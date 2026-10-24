import math
import operator
from functools import reduce
from typing import Any, Callable, Iterable, Union

import numpy as np

from . import ndarray_backend_numpy
from . import ndarray_backend_cpu  # type: ignore[attr-defined]


# Python 3.7 尚未提供 math.prod
def prod(x: Iterable[int]) -> int:
    return reduce(operator.mul, x, 1)


class BackendDevice:
    """后端设备：封装具体的后端实现模块。"""

    def __init__(self, name: str, mod: Any) -> None:
        self.name: str = name
        self.mod: Any = mod

    def __eq__(self, other: object) -> bool:
        return isinstance(other, BackendDevice) and self.name == other.name

    def __repr__(self) -> str:
        return self.name + "()"

    def __getattr__(self, name: str) -> Any:
        return getattr(self.mod, name)

    def enabled(self) -> bool:
        return self.mod is not None

    def randn(self, *shape: int, dtype: str = "float32") -> "NDArray":
        # 注意：NumPy 的标准随机函数不支持指定数据类型，
        # 生成单个元素时也可使用 .astype("float32") 转换类型。
        return NDArray(np.random.randn(*shape).astype(dtype), device=self)

    def rand(self, *shape: int, dtype: str = "float32") -> "NDArray":
        # 注意：NumPy 的标准随机函数不支持指定数据类型，
        # 生成单个元素时也可使用 .astype("float32") 转换类型。
        return NDArray(np.random.rand(*shape).astype(dtype), device=self)

    def one_hot(self, n: int, i: int, dtype: str = "float32") -> "NDArray":
        return NDArray(np.eye(n, dtype=dtype)[i], device=self)

    def empty(self, shape: tuple[int, ...], dtype: str = "float32") -> "NDArray":
        dtype = "float32" if dtype is None else dtype
        assert dtype == "float32"
        return NDArray.make(shape, device=self)

    def full(self, shape: tuple[int, ...], fill_value: float, dtype: str = "float32") -> "NDArray":
        dtype = "float32" if dtype is None else dtype
        assert dtype == "float32"
        arr = self.empty(shape, dtype)
        arr.fill(fill_value)
        return arr


def cuda() -> BackendDevice:
    """返回 CUDA 设备。"""
    try:
        from . import ndarray_backend_cuda  # type: ignore[attr-defined]

        return BackendDevice("cuda", ndarray_backend_cuda)
    except ImportError:
        return BackendDevice("cuda", None)


def cpu_numpy() -> BackendDevice:
    """返回 NumPy 设备。"""
    return BackendDevice("cpu_numpy", ndarray_backend_numpy)


def cpu() -> BackendDevice:
    """返回 CPU 设备。"""
    return BackendDevice("cpu", ndarray_backend_cpu)


def default_device() -> BackendDevice:
    return cpu_numpy()


def all_devices() -> list[BackendDevice]:
    """返回所有后端设备的列表。"""
    return [cpu(), cuda(), cpu_numpy()]


class NDArray:
    """支持多种后端的通用 n 维数组类。

    后端包括 NumPy、原生 CPU 和 GPU。
    本类只提供完成作业编程示例所需的功能。
    为简化实现，目前只支持 float32；如有需要可以扩展。"""

    _shape: tuple[int, ...]
    _strides: tuple[int, ...]
    _offset: int
    _device: BackendDevice
    _handle: Any

    def __init__(self, other, device=None):
        """复制另一个 NDArray，或从 NumPy 数组创建。"""
        if isinstance(other, NDArray):
            # 复制已有的 NDArray
            if device is None:
                device = other.device
            self._init(other.to(device) + 0.0)  # 这里会创建副本
        elif isinstance(other, np.ndarray):
            # 从 NumPy 数组创建副本
            device = device if device is not None else default_device()
            array = self.make(other.shape, device=device)
            array.device.from_numpy(np.ascontiguousarray(other), array._handle)
            self._init(array)
        else:
            # 尝试先将输入转换成 NumPy 数组
            array = NDArray(np.array(other), device=device)
            self._init(array)

    def _init(self, other: "NDArray") -> None:
        self._shape = other._shape
        self._strides = other._strides
        self._offset = other._offset
        self._device = other._device
        self._handle = other._handle

    @staticmethod
    def compact_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
        """计算紧凑布局下各维度的步长。"""
        stride = 1
        res = []
        for i in range(1, len(shape) + 1):
            res.append(stride)
            stride *= shape[-i]
        return tuple(res[::-1])

    @staticmethod
    def make(
        shape: tuple[int, ...],
        strides: tuple[int, ...] | None = None,
        device: BackendDevice | None = None,
        handle: Any = None,
        offset: int = 0,
    ) -> "NDArray":
        """根据给定属性创建新的 NDArray。

        handle=None 时分配内存，否则复用已有数组的 handle。"""
        array = NDArray.__new__(NDArray)
        array._shape = tuple(shape)
        array._strides = NDArray.compact_strides(shape) if strides is None else strides
        array._offset = offset
        array._device = device if device is not None else default_device()
        if handle is None:
            array._handle = array.device.Array(prod(shape))
        else:
            array._handle = handle
        return array

    ### 属性和字符串表示
    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def strides(self) -> tuple[int, ...]:
        return self._strides

    @property
    def device(self) -> BackendDevice:
        return self._device

    @property
    def dtype(self) -> str:
        # 目前只支持 float32
        return "float32"

    @property
    def ndim(self) -> int:
        """返回数组的维度数。"""
        return len(self._shape)

    @property
    def size(self) -> int:
        return prod(self._shape)

    def __repr__(self) -> str:
        return "NDArray(" + self.numpy().__str__() + f", device={self.device})"

    def __str__(self) -> str:
        return self.numpy().__str__()

    ### 基本数组操作
    def fill(self, value: float) -> None:
        """用常量值原地填充数组。"""
        self._device.fill(self._handle, value)

    def to(self, device: BackendDevice) -> "NDArray":
        """通过转换为 NumPy 数组再从 NumPy 导入，在不同设备间转换。"""
        if device == self.device:
            return self
        else:
            return NDArray(self.numpy(), device=device)

    def numpy(self) -> np.ndarray:
        """转换为 NumPy 数组。"""
        return self.device.to_numpy(
            self._handle, self.shape, self.strides, self._offset
        )

    def is_compact(self) -> bool:
        """判断数组是否在内存中紧凑排列，且底层存储大小等于各维度大小的乘积。"""
        return (
            self._strides == self.compact_strides(self._shape)
            and prod(self.shape) == self._handle.size
        )

    def compact(self) -> "NDArray":
        """将数组转换为紧凑布局。"""
        if self.is_compact():
            return self
        else:
            out = NDArray.make(self.shape, device=self.device)
            self.device.compact(
                self._handle, out._handle, self.shape, self.strides, self._offset
            )
            return out

    def as_strided(self, shape: tuple[int, ...], strides: tuple[int, ...]) -> "NDArray":
        """调整数组的形状和步长，不复制内存。"""
        assert len(shape) == len(strides)
        return NDArray.make(
            shape, strides=strides, device=self.device, handle=self._handle, offset=self._offset
        )

    @property
    def flat(self) -> "NDArray":
        return self.reshape((self.size,))

    def reshape(self, new_shape: tuple[int, ...]) -> "NDArray":
        """调整数组形状，不复制内存。

        返回的数组采用新形状，但与原数组指向同一块内存。

        异常：
            ValueError：新旧形状的元素总数不同，或原数组不是紧凑布局。

        参数：
            new_shape (tuple)：数组的新形状。

        返回：
            NDArray：调整形状后与原数组共享内存的数组。"""

        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def permute(self, new_axes: tuple[int, ...]) -> "NDArray":
        """置换各维度的顺序。

        new_axes 指定原有轴的排列顺序，例如：
            - 对于维度顺序为 BHWC 的数组，permute((0, 3, 1, 2)) 将其变为 BCHW。
            - 对于二维数组，permute((1, 0)) 即转置。
        与 reshape 一样，该操作只调整 shape/strides，不复制内存。
        返回的新数组按要求排列维度，但仍指向原数组的内存。

        参数：
            new_axes (tuple)：维度的排列顺序。

        返回：
            NDArray：维度置换后的新数组，与原数组共享内存，仅形状和步长改变。"""

        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def broadcast_to(self, new_shape: tuple[int, ...]) -> "NDArray":
        """将数组广播到新形状。

        除原数组中大小为 1 的维度可广播到任意大小外，new_shape 的各项必须
        与原形状一致。与前面的操作一样，只调整步长，不复制内存。

        异常：
            AssertionError：对于任意 shape[i] != 1 的维度，
                若 new_shape[i] != shape[i]，则触发断言。

        参数：
            new_shape (tuple)：广播的目标形状。

        返回：
            NDArray：具有广播后形状的新数组，与原数组共享内存。"""

        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    ### 读取和设置元素

    def process_slice(self, sl: slice, dim: int) -> slice:
        """将切片转换为具有显式 start/stop/step 的形式。"""
        start, stop, step = sl.start, sl.stop, sl.step
        if start is None:
            start = 0
        if start < 0:
            start = self.shape[dim] + start
        if stop is None:
            stop = self.shape[dim]
        if stop < 0:
            stop = self.shape[dim] + stop
        if step is None:
            step = 1

        # 不处理负步长等情况
        assert stop > start, "Start must be less than stop"
        assert step > 0, "No support for  negative increments"
        return slice(start, stop, step)

    def __getitem__(self, idxs: int | slice | tuple[int | slice, ...]) -> "NDArray":
        """通过 Python 的 __getitem__ 操作访问数组元素。

        对于 a[1:5, :-1:2, 4, :] 这样的表达式，Python 会将索引转换为由
        切片和整数（例如这里的 4）组成的元组。切片的 start、stop、step
        可能为 None 或负数。为简化实现，已提供代码将这些索引规范化为
        切片元组，每个维度对应一个切片。

        根据该元组返回所需的数组子集。只需计算新视图的 shape、strides
        和 offset，使其指向原数组的相应元素，无需复制内存。

        异常：
            AssertionError：切片大小或步长不符合要求，或切片数不等于维度数。
                提供的骨架代码已经包含这些检查。

        参数：
            idxs (tuple)：经骨架代码处理后，为表示目标子集的切片元组。

        返回：
            NDArray：对应所选元素子集的新数组。仅调整形状、步长和偏移量，
                与原数组共享内存。"""

        # 将单个索引包装为元组，并把所有索引转换为切片
        if not isinstance(idxs, tuple):
            idxs = (idxs,)
        idxs = tuple(
            s + self.shape[i] if isinstance(s, int) and s < 0 else s
            for i, s in enumerate(idxs)
        )
        slices = tuple(
            [
                self.process_slice(s, i) if isinstance(s, slice) else slice(s, s + 1, 1)
                for i, s in enumerate(idxs)
            ]
        )
        assert len(slices) == self.ndim, "Need indexes equal to number of dimensions"

        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def __setitem__(self, idxs: int | slice | tuple[int | slice, ...], other: Union["NDArray", float]) -> None:
        """设置数组视图中的值，索引语义与 __getitem__() 相同。"""
        view = self.__getitem__(idxs)
        if isinstance(other, NDArray):
            assert prod(view.shape) == prod(other.shape)
            self.device.ewise_setitem(
                other.compact()._handle,
                view._handle,
                view.shape,
                view.strides,
                view._offset,
            )
        else:
            self.device.scalar_setitem(
                prod(view.shape),
                other,
                view._handle,
                view.shape,
                view.strides,
                view._offset,
            )

    ### 逐元素和标量操作：加法、乘法、布尔运算等

    def ewise_or_scalar(
        self,
        other: Union["NDArray", float],
        ewise_func: Callable[[Any, Any, Any], None],
        scalar_func: Callable[[Any, Any, Any], None],
    ) -> "NDArray":
        """根据 other 是 NDArray 还是标量，调用对应的逐元素或标量版本函数。"""
        out = NDArray.make(self.shape, device=self.device)
        if isinstance(other, NDArray):
            assert self.shape == other.shape, "operation needs two equal-sized arrays"
            ewise_func(self.compact()._handle, other.compact()._handle, out._handle)
        else:
            scalar_func(self.compact()._handle, other, out._handle)
        return out

    def __add__(self, other: Union["NDArray", float]) -> "NDArray":
        return self.ewise_or_scalar(
            other, self.device.ewise_add, self.device.scalar_add
        )

    __radd__ = __add__

    def __sub__(self, other: Union["NDArray", float]) -> "NDArray":
        return self + (-other)

    def __rsub__(self, other: Union["NDArray", float]) -> "NDArray":
        return other + (-self)

    def __mul__(self, other: Union["NDArray", float]) -> "NDArray":
        return self.ewise_or_scalar(
            other, self.device.ewise_mul, self.device.scalar_mul
        )

    __rmul__ = __mul__

    def __truediv__(self, other: Union["NDArray", float]) -> "NDArray":
        return self.ewise_or_scalar(
            other, self.device.ewise_div, self.device.scalar_div
        )

    def __neg__(self) -> "NDArray":
        return self * (-1)

    def __pow__(self, other: float) -> "NDArray":
        out = NDArray.make(self.shape, device=self.device)
        self.device.scalar_power(self.compact()._handle, other, out._handle)
        return out

    def maximum(self, other: Union["NDArray", float]) -> "NDArray":
        return self.ewise_or_scalar(
            other, self.device.ewise_maximum, self.device.scalar_maximum
        )

    ### 比较运算均返回 0.0 或 1.0 浮点值；此处仍有优化空间
    def __eq__(self, other: Any) -> "NDArray":  # type: ignore[override]
        return self.ewise_or_scalar(other, self.device.ewise_eq, self.device.scalar_eq)

    def __ge__(self, other: Any) -> "NDArray":
        return self.ewise_or_scalar(other, self.device.ewise_ge, self.device.scalar_ge)

    def __ne__(self, other: Any) -> "NDArray":  # type: ignore[override]
        return 1 - (self == other)

    def __gt__(self, other: Any) -> "NDArray":
        return (self >= other) * (self != other)

    def __lt__(self, other: Any) -> "NDArray":
        return 1 - (self >= other)

    def __le__(self, other: Any) -> "NDArray":
        return 1 - (self > other)

    ### 逐元素函数

    def log(self) -> "NDArray":
        out = NDArray.make(self.shape, device=self.device)
        self.device.ewise_log(self.compact()._handle, out._handle)
        return out

    def exp(self) -> "NDArray":
        out = NDArray.make(self.shape, device=self.device)
        self.device.ewise_exp(self.compact()._handle, out._handle)
        return out

    def tanh(self) -> "NDArray":
        out = NDArray.make(self.shape, device=self.device)
        self.device.ewise_tanh(self.compact()._handle, out._handle)
        return out

    ### 矩阵乘法
    def __matmul__(self, other: "NDArray") -> "NDArray":
        """计算两个数组的矩阵乘积。

        两个数组必须均为二维（不支持批量矩阵乘法），且相应维度大小必须匹配。

        对于 CPU 后端，当所有维度大小都能被 self.device.__tile_size__ 整除时，
        需要实现高效的分块矩阵乘法。下方代码会调整步长并执行 compact，
        将矩阵转换成分块形式，再传递给对应的 CPU 后端函数。
        若矩阵维度不能被分块大小整除，则回退到朴素 CPU 实现。

        GPU 和 NumPy 后端不提供单独的分块接口；GPU 后端可以在内部对任意
        大小的输入进行分块。"""

        assert self.ndim == 2 and other.ndim == 2
        assert self.shape[1] == other.shape[0]

        m, n, p = self.shape[0], self.shape[1], other.shape[1]

        # 若矩阵各维度满足分块要求，则使用分块矩阵乘法
        if hasattr(self.device, "matmul_tiled") and all(
            d % self.device.__tile_size__ == 0 for d in (m, n, p)
        ):

            def tile(a, tile):
                return a.as_strided(
                    (a.shape[0] // tile, a.shape[1] // tile, tile, tile),
                    (a.shape[1] * tile, tile, a.shape[1], 1),
                )

            t = self.device.__tile_size__
            a = tile(self.compact(), t).compact()
            b = tile(other.compact(), t).compact()
            out = NDArray.make((a.shape[0], b.shape[1], t, t), device=self.device)
            self.device.matmul_tiled(a._handle, b._handle, out._handle, m, n, p)

            return (
                out.permute((0, 2, 1, 3))
                .compact()
                .reshape((self.shape[0], other.shape[1]))
            )

        else:
            out = NDArray.make((m, p), device=self.device)
            self.device.matmul(
                self.compact()._handle, other.compact()._handle, out._handle, m, n, p
            )
            return out

    ### 归约：对所有元素或沿指定轴求和、求最大值
    def reduce_view_out(self, axis: int | tuple[int, ...] | list[int] | None, keepdims: bool = False) -> tuple["NDArray", "NDArray"]:
        """返回适用于归约操作的输入视图，以及用于保存结果的输出数组。"""
        if isinstance(axis, tuple) and not axis:
            raise ValueError("Empty axis in reduce")

        if axis is None:
            view = self.compact().reshape((1,) * (self.ndim - 1) + (prod(self.shape),))
            #out = NDArray.make((1,) * self.ndim, device=self.device)
            out = NDArray.make((1,), device=self.device)

        else:
            if isinstance(axis, (tuple, list)):
                assert len(axis) == 1, "Only support reduction over a single axis"
                axis = axis[0]

            view = self.permute(
                tuple([a for a in range(self.ndim) if a != axis]) + (axis,)
            )
            out = NDArray.make(
                tuple([1 if i == axis else s for i, s in enumerate(self.shape)])
                if keepdims else
                tuple([s for i, s in enumerate(self.shape) if i != axis]),
                device=self.device,
            )
        return view, out

    def sum(self, axis: int | tuple[int, ...] | list[int] | None = None, keepdims: bool = False) -> "NDArray":
        view, out = self.reduce_view_out(axis, keepdims=keepdims)
        self.device.reduce_sum(view.compact()._handle, out._handle, view.shape[-1])
        return out

    def max(self, axis: int | tuple[int, ...] | list[int] | None = None, keepdims: bool = False) -> "NDArray":
        view, out = self.reduce_view_out(axis, keepdims=keepdims)
        self.device.reduce_max(view.compact()._handle, out._handle, view.shape[-1])
        return out



def array(a: Any, dtype: str = "float32", device: BackendDevice | None = None) -> NDArray:
    """提供与 NumPy 接口更接近的便捷方法。"""
    dtype = "float32" if dtype is None else dtype
    assert dtype == "float32"
    return NDArray(a, device=device)


def empty(shape: tuple[int, ...], dtype: str = "float32", device: BackendDevice | None = None) -> NDArray:
    device = device if device is not None else default_device()
    return device.empty(shape, dtype)


def full(shape: tuple[int, ...], fill_value: float, dtype: str = "float32", device: BackendDevice | None = None) -> NDArray:
    device = device if device is not None else default_device()
    return device.full(shape, fill_value, dtype)


def broadcast_to(array: NDArray, new_shape: tuple[int, ...]) -> NDArray:
    return array.broadcast_to(new_shape)


def reshape(array: NDArray, new_shape: tuple[int, ...]) -> NDArray:
    return array.reshape(new_shape)


def maximum(a: NDArray, b: NDArray | float) -> NDArray:
    return a.maximum(b)


def log(a: NDArray) -> NDArray:
    return a.log()


def exp(a: NDArray) -> NDArray:
    return a.exp()


def tanh(a: NDArray) -> NDArray:
    return a.tanh()


def sum(a: NDArray, axis: int | tuple[int] | list[int] | None = None, keepdims: bool = False) -> NDArray:
    return a.sum(axis=axis, keepdims=keepdims)
