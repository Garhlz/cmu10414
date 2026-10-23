"""自动微分系统的核心数据结构与反向传播算法。"""

from __future__ import annotations

from collections.abc import Sequence
from collections import namedtuple
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union, Self
from numpy.typing import ArrayLike, DTypeLike

import numpy

import needle
from needle import init

from .backend_numpy import CPUDevice, Device, all_devices, cpu

# 是否启用惰性求值。关闭时，创建计算图节点后会立刻计算并缓存结果。
LAZY_MODE = False
# 用于调试当前存活的 Value/Tensor 对象数量。
TENSOR_COUNTER = 0

# HW1/HW2 使用 NumPy 作为数组后端；后续作业会替换为自行实现的后端。
import numpy as array_api

NDArray = numpy.ndarray


class Op:
    """计算图中的算子基类，例如加法、矩阵乘法和 ReLU。"""

    def __call__(self, *args: Any, **kwargs: Any) -> Value:
        # *args 将任意数量的位置参数收集为元组，便于统一处理一元和多元算子。
        raise NotImplementedError()

    def compute(self, *args: Any) -> Any:
        """执行算子的正向计算。

        此处接收底层 NDArray，而不是 Tensor。Tensor 负责维护计算图，
        compute 只负责根据输入数组计算输出数组。
        """
        raise NotImplementedError()

    def gradient(
        self, out_grad: "Value", node: "Value"
    ) -> Value | tuple[Value, ...] | list[Value]:
        """根据上游梯度计算当前算子各输入的梯度贡献。

        若当前节点表示 z = f(x, y)，则 out_grad 是 dL/dz。本方法应用
        链式法则，返回 dL/dx = dL/dz * dz/dx 和 dL/dy。

        返回值可以是单个 Value，也可以是与输入一一对应的 Value 元组。
        """
        raise NotImplementedError()

    def gradient_as_tuple(self, out_grad: Value, node: Value) -> tuple[Value, ...]:
        """调用 gradient，并把不同形式的返回值统一转换为元组。"""
        output = self.gradient(out_grad, node)
        if isinstance(output, tuple):
            return output
        elif isinstance(output, list):
            return tuple(output)
        else:
            # (output,) 是单元素元组；逗号不可省略，否则只是加了括号。
            return (output,)


class TensorOp(Op):
    """输出单个 Tensor 的算子基类。

    典型调用链（以 ``c = a + b`` 为例）：

    1. Python 将 ``a + b`` 转换为 ``a.__add__(b)``；
    2. ``Tensor.__add__`` 选择 ``EWiseAdd`` 或 ``AddScalar``；
    3. ``EWiseAdd()(a, b)`` 先构造算子对象，再触发这里的 ``__call__``；
    4. ``Tensor.make_from_op`` 创建输出节点并记录算子与输入；
    5. 非惰性模式下，``realize_cached_data`` 调用算子的 ``compute``；
    6. 反向传播时，再通过输出节点保存的 ``op`` 调用 ``gradient``。

    因此，调用算子对象不仅执行数值运算，还会把这次运算记录到计算图中。
    """

    def __call__(self, *args: Value) -> Tensor:
        # 正向路径第 3 步：op(a, b) 会被 Python 转换为 op.__call__(a, b)。
        # 此时 self 是具体算子对象，args 是包含输入 Tensor 的元组 (a, b)。
        # 这里不直接调用 compute，因为必须先创建节点并保存计算图关系。
        return Tensor.make_from_op(self, args)


class TensorTupleOp(Op):
    """输出 TensorTuple 的算子基类。"""

    def __call__(self, *args: Value) -> TensorTuple:
        return TensorTuple.make_from_op(self, args)


class Value:
    """计算图节点，记录产生它的算子、输入节点及缓存数据。"""

    # Optional[Op] 表示值可以是 Op 或 None；叶子节点的 op 为 None。
    op: Optional[Op]
    # 引号中的 "Value" 是前向引用：类尚未完成定义时也可用于类型标注。
    inputs: Sequence[Value]
    # 动态计算过程中缓存的底层数组，避免重复执行正向计算。
    # Tensor 缓存数组，TensorTuple 缓存数组元组，尚未求值时为 None。
    # 这层使用 Any，由具体算子的 compute 与公开 Tensor 接口约束数据类型。
    cached_data: Any
    requires_grad: bool

    def realize_cached_data(self) -> Any:
        """按需执行正向数值计算并返回缓存结果。

        计算图节点保存的是 Tensor 输入，而具体算子的 compute 接收底层数组。
        本方法负责递归取得输入数组、调用 compute，并缓存输出数组。
        """
        # 正向路径第 5a 步：若已有缓存，直接返回，避免重复计算同一节点。
        if self.cached_data is not None:
            return self.cached_data

        # 正向路径第 5b 步：列表推导式递归取得每个输入节点的底层数组。
        # 前面的 * 再把数组列表解包为 compute 的多个位置参数。
        # 例如 EWiseAdd 节点最终执行 self.op.compute(a_data, b_data)。
        self.cached_data = self.op.compute(
            *[x.realize_cached_data() for x in self.inputs]
        )
        return self.cached_data

    def is_leaf(self) -> bool:
        """没有生成算子的节点即为叶子节点，通常是用户创建的输入。"""
        return self.op is None

    def __del__(self) -> None:
        # global 表示修改模块级变量，而不是创建同名局部变量。
        global TENSOR_COUNTER
        TENSOR_COUNTER -= 1

    def _init(
        self,
        op: Optional[Op],
        inputs: Sequence[Value],
        *,  # * 之后的参数只能通过关键字传入，防止位置参数含义混淆。
        num_outputs: int = 1,
        cached_data: Any = None,
        requires_grad: Optional[bool] = None,
    ) -> None:
        """初始化一个 Value 节点及其计算图元数据。"""
        global TENSOR_COUNTER
        TENSOR_COUNTER += 1
        if requires_grad is None:
            # any 接收生成器；只要任一输入需要梯度，当前节点也需要梯度。
            requires_grad = any(x.requires_grad for x in inputs)
        # 正向路径第 4b 步：保存“由哪个算子产生”和“依赖哪些输入”。
        # 反向传播正是依靠这两项，从输出节点逐步找到所有前驱节点。
        self.op = op
        self.inputs = inputs
        self.num_outputs = num_outputs
        self.cached_data = cached_data
        self.requires_grad = requires_grad

    @classmethod
    def make_const(cls, data: Any, *, requires_grad: bool = False) -> Self:
        """创建不依赖其他节点的常量值。"""
        # __new__ 只分配对象而不调用 __init__，便于统一交给 _init 初始化。
        value = cls.__new__(cls)
        value._init(
            None,
            [],
            cached_data=data,
            requires_grad=requires_grad,
        )
        return value

    @classmethod
    def make_from_op(cls, op: Op, inputs: Sequence[Value]) -> Self:
        """根据算子及输入创建新的计算图节点。"""
        value = cls.__new__(cls)
        value._init(op, inputs)

        if not LAZY_MODE:
            if not value.requires_grad:
                return value.detach()
            value.realize_cached_data()
        return value


# HW1 不需要实现 TensorTuple，保留该结构是为了后续作业扩展。
class TensorTuple(Value):
    """一组 Tensor 的图节点；为简化实现，不支持嵌套 TensorTuple。"""

    def __len__(self) -> int:
        cdata = self.realize_cached_data()
        return len(cdata)

    def __getitem__(self, index: int) -> Tensor:
        # 实现 [] 运算符，例如 tensor_tuple[0]。
        return needle.ops.tuple_get_item(self, index)

    def tuple(self) -> tuple[Tensor, ...]:
        return tuple([x for x in self])

    def __repr__(self) -> str:
        # __repr__ 返回面向开发者的对象表示，交互式环境会自动调用它。
        return "needle.TensorTuple" + str(self.tuple())

    def __str__(self) -> str:
        return self.__repr__()

    def __add__(self, other: TensorTuple) -> TensorTuple:
        assert isinstance(other, TensorTuple)
        assert len(self) == len(other)
        # * 将列表中的元素解包为 make_tuple 的多个位置参数。
        return needle.ops.make_tuple(*[self[i] + other[i] for i in range(len(self))])

    def detach(self) -> TensorTuple:
        """共享底层数据，但创建一个与原计算图断开的新节点。"""
        return TensorTuple.make_const(self.realize_cached_data())


class Tensor(Value):
    """用户直接操作的张量，同时也是计算图中的 Value 节点。"""

    grad: "Tensor"

    def __init__(
        self,
        array: ArrayLike | Tensor,
        *,  # device 及其后的参数必须用关键字传入。
        device: Optional[Device] = None,
        dtype: DTypeLike = None,
        requires_grad: bool = True,
        **kwargs: Any,
    ) -> None:
        if isinstance(array, Tensor):
            if device is None:
                device = array.device
            if dtype is None:
                dtype = array.dtype
            if device == array.device and dtype == array.dtype:
                cached_data = array.realize_cached_data()
            else:
                # 设备或类型不同时，先转成 NumPy，再由目标后端重新创建数组。
                cached_data = Tensor._array_from_numpy(
                    array.numpy(), device=device, dtype=dtype
                )
        else:
            # Python 条件表达式：条件成立取前值，否则取后值。
            device = device if device else cpu()
            cached_data = Tensor._array_from_numpy(array, device=device, dtype=dtype)

        self._init(
            None,
            [],
            cached_data=cached_data,
            requires_grad=requires_grad,
        )

    @staticmethod
    def _array_from_numpy(
        numpy_array: ArrayLike, device: Device, dtype: DTypeLike
    ) -> NDArray:
        """将 Python/NumPy 数据转换为当前后端使用的数组。"""
        if array_api is numpy:
            return numpy.array(numpy_array, dtype=dtype)
        return array_api.array(numpy_array, device=device, dtype=dtype)

    @staticmethod
    def make_from_op(op: Op, inputs: Sequence[Value]) -> Tensor:
        """由算子和输入 Tensor 创建新的输出 Tensor 节点。

        该工厂方法由 ``TensorOp.__call__`` 调用。它使用 ``__new__``
        绕过面向用户数据输入的 ``Tensor.__init__``，因为此处创建的是
        一个有算子、有输入依赖的中间节点，而不是新的叶子 Tensor。
        """
        # 正向路径第 4a 步：只分配 Tensor 对象，暂不调用 Tensor.__init__。
        tensor = Tensor.__new__(Tensor)
        # 记录 op、inputs、requires_grad 等计算图信息；此时通常还没有输出数据。
        tensor._init(op, inputs)

        if not LAZY_MODE:
            # 不需要梯度时，将结果变为常量节点，避免保留无用计算图。
            if not tensor.requires_grad:
                return tensor.detach()
            # 正向路径第 5 步：HW1 默认立即执行 compute 并缓存数值结果。
            tensor.realize_cached_data()
        return tensor

    @staticmethod
    def make_const(data: ArrayLike | Tensor, requires_grad: bool = False) -> Tensor:
        tensor = Tensor.__new__(Tensor)
        tensor._init(
            None,
            [],
            cached_data=data
            if not isinstance(data, Tensor)
            else data.realize_cached_data(),
            requires_grad=requires_grad,
        )
        return tensor

    @property
    def data(self) -> Tensor:
        # @property 让方法可以像普通属性一样通过 tensor.data 访问。
        return self.detach()

    @data.setter
    def data(self, value: Tensor) -> None:
        # @data.setter 定义 tensor.data = value 时的行为。
        assert isinstance(value, Tensor)
        assert value.dtype == self.dtype, "%s %s" % (
            value.dtype,
            self.dtype,
        )
        self.cached_data = value.realize_cached_data()

    def detach(self) -> Tensor:
        """共享底层数据，但创建一个与原计算图断开的 Tensor。"""
        return Tensor.make_const(self.realize_cached_data())

    @property
    def shape(self) -> tuple[int, ...]:
        return self.realize_cached_data().shape

    @property
    def dtype(self) -> numpy.dtype:
        return self.realize_cached_data().dtype

    @property
    def device(self) -> CPUDevice:
        data = self.realize_cached_data()
        # NumPy 数组始终位于 CPU。
        if array_api is numpy:
            return cpu()
        return data.device

    def backward(self, out_grad: Tensor | None = None) -> None:
        """启动反向模式自动微分，将输出对各节点的梯度保存在节点的 grad 中。

        默认梯度种子为全 1；非标量输出时相当于对输出元素之和求导。
        此方法计算梯度，不更新模型参数。
        """
        # 反向路径第 1 步：为最终输出准备梯度种子 dL/dL。
        # 未指定时使用全 1 Tensor；对标量损失而言，它就是数值 1。
        # *self.shape 会把形状元组解包为 init.ones 的位置参数。
        out_grad = (
            out_grad
            if out_grad
            else init.ones(*self.shape, dtype=self.dtype, device=self.device)
        )
        # 反向路径第 2 步：从当前输出出发，沿计算图向输入传播梯度。
        compute_gradient_of_variables(self, out_grad)

    def __repr__(self) -> str:
        return "needle.Tensor(" + str(self.realize_cached_data()) + ")"

    def __str__(self) -> str:
        return self.realize_cached_data().__str__()

    def numpy(self) -> NDArray:
        data = self.realize_cached_data()
        if array_api is numpy:
            return numpy.array(data)
        return data.numpy()

    def __add__(self, other: Tensor | float) -> Tensor:
        """重载 ``+``，并按右操作数类型选择对应的 Needle 算子。"""
        # 正向路径第 1 步：Python 把 a + b 转换为 a.__add__(b)。
        # 正向路径第 2 步：Tensor + Tensor 使用逐元素加法；Tensor + 标量
        # 使用专门的标量加法，避免把普通数字额外包装成 Tensor。
        if isinstance(other, Tensor):
            # 两对括号含义不同：EWiseAdd() 构造对象，随后 (self, other)
            # 调用该对象的 TensorOp.__call__，开始创建计算图节点。
            return needle.ops.EWiseAdd()(self, other)
        else:
            # scalar 保存在 AddScalar 算子对象内部，因此它只有一个 Tensor 输入。
            return needle.ops.AddScalar(other)(self)

    def __mul__(self, other: Tensor | float) -> Tensor:
        if isinstance(other, Tensor):
            return needle.ops.EWiseMul()(self, other)
        else:
            return needle.ops.MulScalar(other)(self)

    def __pow__(self, other: Tensor | float) -> Tensor:
        if isinstance(other, Tensor):
            return needle.ops.EWisePow()(self, other)
        else:
            return needle.ops.PowerScalar(other)(self)

    def __sub__(self, other: Tensor | float) -> Tensor:
        if isinstance(other, Tensor):
            return needle.ops.EWiseAdd()(self, needle.ops.Negate()(other))
        else:
            return needle.ops.AddScalar(-other)(self)

    def __truediv__(self, other: Tensor | float) -> Tensor:
        if isinstance(other, Tensor):
            return needle.ops.EWiseDiv()(self, other)
        else:
            return needle.ops.DivScalar(other)(self)

    def __matmul__(self, other: Tensor) -> Tensor:
        # @ 在 Python 中对应矩阵乘法运算符。
        return needle.ops.MatMul()(self, other)

    def matmul(self, other: Tensor) -> Tensor:
        return needle.ops.MatMul()(self, other)

    def sum(self, axes: int | tuple[int, ...] | None = None) -> Tensor:
        return needle.ops.Summation(axes)(self)

    def broadcast_to(self, shape: tuple[int, ...]) -> Tensor:
        return needle.ops.BroadcastTo(shape)(self)

    def reshape(self, shape: tuple[int, ...]) -> Tensor:
        return needle.ops.Reshape(shape)(self)

    def __neg__(self) -> Tensor:
        # 一元负号 -tensor 会调用 __neg__。
        return needle.ops.Negate()(self)

    def transpose(self, axes: tuple[int, int] | None = None) -> Tensor:
        return needle.ops.Transpose(axes)(self)

    # 反向运算符在左操作数无法处理时调用，例如 2 + tensor。
    __radd__ = __add__
    __rmul__ = __mul__


def compute_gradient_of_variables(output_tensor: Tensor, out_grad: Tensor) -> None:
    """从 output_tensor 开始执行反向模式自动微分。

    out_grad 是最终目标对 output_tensor 的梯度种子。函数不显式返回结果，
    而是把累计梯度写入每个可达 Tensor 的 grad 属性。
    """
    # 反向路径第 3 步：一个节点可能经多条路径影响输出，因此先分别保存
    # 每条路径产生的梯度贡献，到处理该节点时再统一相加。
    node_to_output_grads_list: Dict[Tensor, List[Tensor]] = {}
    # 把 backward 准备的梯度种子放到最终输出节点上。
    node_to_output_grads_list[output_tensor] = [out_grad]

    # 反向路径第 4 步：拓扑序原本是输入到输出；reversed 将其反转，保证
    # 处理一个节点前，所有依赖它的下游节点都已经贡献了梯度。
    reverse_topo_order = list(reversed(find_topo_sort([output_tensor])))

    ### BEGIN YOUR SOLUTION
    for node in reverse_topo_order:
        # 反向路径第 5 步：累计来自不同下游路径的梯度，得到完整的 dL/dnode，也就是gradient()用到的out_grad
        node.grad = sum_node_list(node_to_output_grads_list[node])

        if node.is_leaf():
            continue
        # 反向路径第 6 步：实际调用正向时保存在 node.op 中的局部求导规则(gradient())
        input_grads = node.op.gradient_as_tuple(node.grad, node)

        # 反向路径第 7 步：配对输入节点与对应梯度，将贡献追加到每个输入的列表中
        # 之后轮到该输入时再对所有贡献求和，求出对应的out_grad
        for input_node, input_grad in zip(node.inputs, input_grads):
            if input_node not in node_to_output_grads_list:
                node_to_output_grads_list[input_node] = []  # 初始化
            node_to_output_grads_list[input_node].append(input_grad)
    ### END YOUR SOLUTION


def find_topo_sort(node_list: List[Value]) -> List[Value]:
    """返回所有目标节点及其依赖节点的正向拓扑序。

    使用后序深度优先搜索：先递归访问一个节点的全部输入，再把当前节点
    放入结果，因此每个节点一定出现在所有依赖节点之后。
    """
    ### BEGIN YOUR SOLUTION
    # set 是哈希集合，成员查询平均为 O(1)，用于避免重复访问共享子图。
    visited = set()
    topo_order = []
    for node in node_list:
        topo_sort_dfs(node, visited, topo_order)
    return topo_order
    ### END YOUR SOLUTION


def topo_sort_dfs(node: Value, visited: set[Value], topo_order: list[Value]) -> None:
    """对计算图执行后序 DFS，并把节点追加到 topo_order。"""
    ### BEGIN YOUR SOLUTION
    if node in visited:
        return
    visited.add(node)
    for input_node in node.inputs:
        topo_sort_dfs(input_node, visited, topo_order)
    # 后序遍历的关键：所有前驱处理完成后，才记录当前节点。
    topo_order.append(node)

    ### END YOUR SOLUTION


# 辅助函数


def sum_node_list(node_list: Sequence[Value]) -> Value:
    """累加多个梯度节点，同时避免 Python sum 从整数 0 开始产生冗余节点。"""
    from functools import reduce
    from operator import add

    # reduce(add, [a, b, c]) 等价于 add(add(a, b), c)。
    return reduce(add, node_list)
