"""神经网络模块：参数管理、模块组合，以及常用网络层与损失函数。

与 HW1 的 TensorOp 的分工：
    TensorOp 定义一个基础运算的 compute 和 gradient。
    Module 在 forward 中组合 Tensor 运算，表达一层或整个网络。
    forward 中的 Tensor 运算会继续构建计算图，因此 Module 通常无需手写反向传播。

典型调用流程：
    model(x) -> Module.__call__ -> 具体子类的 forward -> Tensor 算子 -> 计算图
    loss.backward() 沿计算图求导；优化器使用 model.parameters() 获取待更新参数。
"""

from typing import Any
from needle.backend_numpy import CPUDevice
from numpy.typing import DTypeLike
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """标记可学习参数的 Tensor 子类，例如 Linear 的权重与偏置。

    本类只有文档字符串，没有重写方法，直接继承 Tensor 的构造与运算行为。
    标记作用来自 isinstance(value, Parameter)：参数收集器只收集此类型。
    普通 Tensor 即使 requires_grad=True，也不会因此被 parameters() 收集。
    requires_grad 控制是否需要梯度；Parameter 标记是否属于模块的可学习参数。
    """


def _unpack_params(value: object) -> list[Tensor]:
    """递归展开属性中的 Parameter，返回一个扁平列表。

    属性可嵌套 Module、字典、列表和元组，例如 Sequential 保存的模块元组。
    普通 Tensor、整数、布尔值等不属于待收集参数，返回空列表。
    这里不做去重；若多个属性引用同一个 Parameter，可能重复收集它。
    """
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        # 递归收集子模块中的参数
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            # items() 产生 (键, 值) 对；这里只递归处理属性值。
            # 对 list 使用 += 会将右侧列表中的元素追加到当前列表。
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        # isinstance 的第二个参数可以是类型元组，表示匹配其中任意一种类型。
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> list["Module"]:
    """递归查找所有后代模块，供 train()/eval() 统一切换模式。

    注解里的 "Module" 是字符串前向引用：此处 Module 类还未定义。
    """
    if isinstance(value, Module):
        modules = [value]
        # __dict__ 保存对象的实例属性，例如 self.weight、self.fn。
        # extend 将可迭代对象的元素逐个加入列表，而非把整个列表作为一个元素。
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    """所有网络层和网络容器的基类。

    将 Parameter 或子 Module 赋给 self 的属性后，这些递归辅助函数就能找到它们。
    子类通过 forward 定义计算过程；BatchNorm、Dropout 还会读取 training 标志。
    """

    def __init__(self) -> None:
        # 新建模块默认处于训练模式；它与 Tensor.requires_grad 是不同的设置。
        self.training = True

    def parameters(self) -> list[Tensor]:
        """收集本模块及其后代模块的全部 Parameter，供优化器更新(在优化器初始化的时候传入)"""
        # 从实例属性字典开始遍历，因此参数必须保存在实例属性中才能被发现。
        return _unpack_params(self.__dict__)

    def _children(self) -> list["Module"]:
        """返回递归查找到的后代模块，而非仅返回直接子模块。"""
        return _child_modules(self.__dict__)

    def eval(self) -> None:
        """将自身和所有后代设为评估模式；本方法原地修改状态，返回 None。

        eval() 只切换层的行为，不会自动关闭梯度计算或切断计算图。
        """
        self.training = False
        for m in self._children():
            m.training = False

    def train(self) -> None:
        """将自身和所有后代设为训练模式；此方法本身不执行训练循环。"""
        self.training = True
        for m in self._children():
            m.training = True

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        """由子类组合 Tensor 运算实现前向传播，并记录本次运算的计算图。"""
        raise NotImplementedError(f"{type(self).__name__} 尚未实现 forward()")

    def __call__(self, *args: Any, **kwargs: Any) -> Tensor:
        # __call__ 使模块实例可以像函数一样调用：layer(x) 等价于进入此方法。
        # *args 收集位置参数，**kwargs 收集关键字参数；调用时再展开并转发。
        # self.forward 会按实际对象类型找到子类实现，即 Python 的动态方法分派。
        return self.forward(*args, **kwargs)


class Identity(Module):
    """恒等模块：直接返回输入，可作为无需变换时的占位层。"""

    def forward(self, x: Tensor) -> Tensor:
        # 返回同一个对象，不复制数值，也不额外创建算子节点。
        return x


class Linear(Module):
    """全连接层：将每个样本从 in_features 维映射到 out_features 维。

    本作业按 X @ weight + bias 计算：
        X:      (batch_size, in_features)
        weight: (in_features, out_features)，使用 Parameter 包装
        bias:   (out_features,)，可选，使用 Parameter 包装
        输出:   (batch_size, out_features)

    权重和偏置使用 Kaiming 均匀初始化，先初始化权重、再初始化偏置。
    两者的 fan_in 分别为 in_features、out_features；注意作业对顺序的要求。
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: CPUDevice | None = None,
        dtype: DTypeLike = "float32",
    ) -> None:
        super().__init__()
        # super() 调用父类 Module 的初始化逻辑，从而设置 self.training。
        # device/dtype 应继续传给初始化函数，确保参数的设备与类型符合要求。
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        weight = init.kaiming_uniform(
            in_features, out_features, device=device, dtype=dtype, requires_grad=True
        )
        self.weight = Parameter(weight)

        self.bias = None  # 显式初始化为None
        if bias:
            bias = init.kaiming_uniform(
                out_features, 1, device=device, dtype=dtype, requires_grad=True
            )
            bias = ops.reshape(bias, (out_features,))
            self.bias = Parameter(bias)
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        """进行矩阵乘法，并将偏置显式广播到输出形状后相加。"""
        ### BEGIN YOUR SOLUTION
        Z = X @ self.weight
        if self.bias is not None:
            bias = ops.broadcast_to(self.bias, (X.shape[0], self.out_features))
            return Z + bias
        return Z
        ### END YOUR SOLUTION


class Flatten(Module):
    """保留批次维，将其余维度合并：(B, X0, X1, ...) -> (B, X0 * X1 * ...)。

    使用 Tensor 的 reshape 算子才能保留计算图中的形状变换关系。
    """

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size = X.shape[0]
        return ops.reshape(X, (batch_size, -1))
        ### END YOUR SOLUTION


class ReLU(Module):
    """逐元素计算 max(x, 0)，形状不变。

    此层调用已有的 ReLU Tensor 算子；局部求导规则由该算子负责。
    """

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    """按顺序组合多个模块：输出 = 最后一层(...第二层(第一层(x)))。

    容器本身没有额外权重，parameters() 会递归收集各子模块的参数。
    """

    def __init__(self, *modules: Module) -> None:
        super().__init__()
        # *modules 将任意数量的位置参数收集为元组，例如 Sequential(a, b, c)。
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        """依次将上一模块的输出作为下一模块的输入，返回最后的输出。"""
        ### BEGIN YOUR SOLUTION
        output = x
        for module in self.modules:
            output = module(output)
        return output
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    """计算批次平均的 Softmax 交叉熵损失，无可学习参数。

    logits: (B, C)，未经 softmax 的类别得分；y: (B,)，整数类别编号。
    交叉熵损失: -log(e_true / sum(e_i)) = log(sum(e_i)) - y_true
    单样本损失为 log(sum(exp(logits))) - 正确类别的 logit，最后取批次平均。
    使用数值稳定的 LogSumExp 算子；标签可用 init.one_hot 转换。
    返回标量 Tensor，调用其 backward() 即可反向传播到模型参数。
    """

    def forward(self, logits: Tensor, y: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size, num_classes = logits.shape
        y_onehot = init.one_hot(
            num_classes,
            y,
            device=logits.device,
            dtype=logits.dtype,
            requires_grad=False,
        )  # (B, num_classes)，与logits相乘再求和，得到每个batch的true logit

        lse = ops.logsumexp(logits, axes=(1,))
        true_logits = ops.summation(logits * y_onehot, axes=(1,))
        # 以上两者的shape都是(B,)
        losses = lse - true_logits
        return ops.summation(losses) / batch_size  # 求当前所有batch平均的损失
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    """对形状为 (B, dim) 的输入按批次归一化：每个特征单独统计。

    训练时沿批次轴 axis=0 计算均值和有偏方差，再进行归一化及仿射变换：
        输出 = weight * (x - mean) / sqrt(var + eps) + bias
    weight、bias 是形状为 (dim,) 的 Parameter，初始值分别为 1、0。
    running_mean、running_var 是普通 Tensor，初始值分别为 0、1。

    训练时更新运行统计量：running = (1 - momentum) * running + momentum * 当前统计量。
    更新应脱离计算图，以免跨批次保留历史；运行统计量不由优化器更新。
    评估时使用保存的运行统计量，并停止更新它们。
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        device: CPUDevice | None = None,
        dtype: DTypeLike = "float32",
    ) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        weight = init.ones(dim, device=device, dtype=dtype, requires_grad=True)
        bias = init.zeros(dim, device=device, dtype=dtype, requires_grad=True)
        self.weight = Parameter(weight)
        self.bias = Parameter(bias)
        self.running_mean = init.zeros(
            dim, device=device, dtype=dtype, requires_grad=False
        )
        self.running_var = init.ones(
            dim, device=device, dtype=dtype, requires_grad=False
        )
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """根据 self.training 选择统计量；广播后归一化，输出形状与输入一致。"""
        ### BEGIN YOUR SOLUTION
        batch_size = x.shape[0]
        if self.training:
            mean = ops.summation(x, axes=(0,)) / batch_size  # shape=(dim,)
            reshaped_mean = ops.reshape(mean, (1, self.dim))
            centered = x - ops.broadcast_to(reshaped_mean, x.shape)

            # shape=(dim,)
            var = ops.summation(centered**2, axes=(0,)) / batch_size

            self.running_mean = (
                (1 - self.momentum) * self.running_mean + self.momentum * mean.detach()
            ).detach()

            self.running_var = (
                (1 - self.momentum) * self.running_var + self.momentum * var.detach()
            ).detach()

            denom = (var + self.eps) ** 0.5
            denom = ops.reshape(denom, (1, self.dim))

            normed = centered / ops.broadcast_to(denom, x.shape)

            weight = ops.broadcast_to(self.weight, x.shape)
            bias = ops.broadcast_to(self.bias, x.shape)
            return weight * normed + bias
        else:
            mean = ops.reshape(self.running_mean, (1, self.dim))
            mean = ops.broadcast_to(mean, x.shape)
            centered = x - mean

            var = self.running_var
            denom = (var + self.eps) ** 0.5
            denom = ops.reshape(denom, (1, self.dim))
            normed = centered / ops.broadcast_to(denom, x.shape)

            weight = ops.broadcast_to(self.weight, x.shape)
            bias = ops.broadcast_to(self.bias, x.shape)
            return weight * normed + bias
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    """对形状为 (B, dim) 的输入逐样本归一化。

    沿特征轴 axis=1 计算每个样本的均值和有偏方差，公式与 BatchNorm 类似。
    weight、bias 是形状为 (dim,) 的 Parameter，初始值分别为 1、0。
    本层无需运行统计量，训练和评估使用同样的计算方式。
    eps 加在方差上，用于改善数值稳定性。
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        device: CPUDevice | None = None,
        dtype: DTypeLike = "float32",
    ) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        weight = init.ones(dim, device=device, dtype=dtype, requires_grad=True)
        bias = init.zeros(dim, device=device, dtype=dtype, requires_grad=True)
        self.weight = Parameter(weight)
        self.bias = Parameter(bias)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """将逐样本的统计量重塑并广播回输入形状，再应用可学习的缩放和偏置。"""
        ### BEGIN YOUR SOLUTION
        batch_size = x.shape[0]
        mean = ops.summation(x, axes=(1,)) / self.dim  # needle的summation不保留形状的
        mean = ops.reshape(mean, (batch_size, 1))

        centered = x - ops.broadcast_to(mean, x.shape)  # (B, dim)

        var = ops.summation(centered**2, axes=(1,)) / self.dim  # 方差 (B,)

        denom = (var + self.eps) ** 0.5
        denom = ops.reshape(denom, (batch_size, 1))

        # 按axis=1标准化之后的x数据
        normed = centered / ops.broadcast_to(denom, x.shape)

        weight = ops.broadcast_to(self.weight, x.shape)
        bias = ops.broadcast_to(self.bias, x.shape)
        return weight * normed + bias  # 都是element wise的计算
        ### END YOUR SOLUTION


class Dropout(Module):
    """训练时以概率 p 随机将元素置零，评估时直接返回输入。

    保留下来的元素除以 (1 - p)，从而使输出的期望等于输入。
    可使用 init 中的伯努利采样函数构造掩码；掩码无需梯度。
    随机掩码不是可学习参数。输出形状与输入一致。
    """

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if not self.training:
            return x
        # 以概率 p 丢弃元素，掩码为 0。
        # 以概率 1-p 保留元素，掩码为 1
        mask = init.randb(
            *x.shape, p=1 - self.p, device=x.device, dtype=x.dtype, requires_grad=False
        )
        return x * mask / (1 - self.p)
        ### END YOUR SOLUTION


class Residual(Module):
    """包装模块 fn，计算 fn(x) + x，提供残差（跳跃）连接。

    fn(x) 应与 x 具有相同形状。fn 保存为实例属性，因此参数收集和
    train()/eval() 都会继续递归进入它。
    加法算子的反向传播会将两条路径的梯度传播回 x，并在汇合处累加。
    """

    def __init__(self, fn: Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return x + self.fn(x)
        ### END YOUR SOLUTION
