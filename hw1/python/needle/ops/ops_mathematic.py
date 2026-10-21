"""Needle 的数学算子：正向计算及其局部反向传播规则。"""

from numbers import Number
from typing import List, Optional, Tuple, Union

import numpy

from ..autograd import NDArray, Op, Tensor, TensorOp, TensorTuple, TensorTupleOp, Value

# 通过 array_api 别名隔离具体数组后端；HW1 使用 NumPy，后续会替换后端。
BACKEND = "np"
import numpy as array_api


class EWiseAdd(TensorOp):
    """两个同形状 Tensor 的逐元素加法。

    ``EWiseAdd()(a, b)`` 中，TensorOp.__call__ 负责建图；这里的 compute
    只接收 a、b 的底层 NDArray 并完成真正的数值运算。
    """

    def compute(self, a: NDArray, b: NDArray):
        # 正向路径第 6 步：NumPy 的 + 在底层数组上执行逐元素相加。
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        # 反向路径第 6a 步：若 z=a+b，则 dz/da=dz/db=1，因此两个输入
        # 收到的梯度贡献都等于上游梯度 out_grad。返回顺序必须与 inputs 一致。
        return out_grad, out_grad


def add(a, b):
    """函数式加法接口；效果与 Tensor 之间使用 ``a + b`` 相同。"""
    # EWiseAdd() 创建算子实例；紧随其后的 (a, b) 调用实例的 __call__。
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    """Tensor 与 Python 标量的逐元素加法。"""

    def __init__(self, scalar):
        # 普通标量不是计算图输入，直接作为算子对象的配置保存。
        self.scalar = scalar

    def compute(self, a: NDArray):
        # NumPy 会把 scalar 广播到 a 的形状后逐元素相加。
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        # 若 z=a+c 且 c 是常量，则 dz/da=1；常量不需要返回梯度。
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    """两个 Tensor 的逐元素乘法。"""

    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        # 元组解包：node.inputs 中的两个元素分别赋给 lhs 和 rhs。
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    """Tensor 与 Python 标量的逐元素乘法。"""

    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        # 尾随逗号表示返回单元素元组，以便与多输入算子的接口保持一致。
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """逐元素计算 lhs ** rhs，底数和指数均为 Tensor。"""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.pow(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # ** 是 Python 的幂运算符；Tensor.__pow__ 会把它转换为 Needle 算子。
        lhs, rhs = node.inputs
        # d(a^b)/da = b*a^(b-1)，d(a^b)/db = log(a)*a^b。
        return out_grad * rhs * (lhs ** (rhs - 1)), out_grad * log(lhs) * (lhs**rhs)
        ### END YOUR SOLUTION


def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """逐元素计算 Tensor 的整数次幂。"""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.pow(a, self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # node.inputs 是序列；一元算子的唯一输入位于索引 0。
        return out_grad * self.scalar * (node.inputs[0] ** (self.scalar - 1))
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """两个 Tensor 的逐元素真除法。"""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        # rhs ** -2 表示 1 / rhs^2；负指数同样通过 Tensor.__pow__ 建图。
        return out_grad / rhs, out_grad * (-lhs) * (rhs**-2)
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    """Tensor 除以 Python 标量。"""

    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    """交换 Tensor 的两个轴；未指定时交换最后两个轴。"""

    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # None 是 Python 表示“未提供值”的单例对象，应使用 is None 判断。
        if self.axes is None:
            axis1, axis2 = len(a.shape) - 2, len(a.shape) - 1
        else:
            # 将二元素元组直接解包为两个变量。
            axis1, axis2 = self.axes

        return array_api.swapaxes(a, axis1, axis2)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # 同一对轴交换两次会恢复原状，因此转置的逆操作仍是自身。
        return transpose(out_grad, self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    """在元素总数不变的前提下修改 Tensor 形状。"""

    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # 反向传播只需把梯度恢复成输入的原始形状。
        return reshape(out_grad, node.inputs[0].shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


def sum_to_shape(grad, target_shape):
    """沿广播产生的维度求和，并把梯度恢复为 target_shape。"""

    # NumPy 从最右侧对齐形状；较短形状等价于在左侧补若干个 1。
    # (1,) * diff 利用元组乘法重复单元素元组，再用 + 拼接元组。
    diff = len(grad.shape) - len(target_shape)
    padded_shape = (1,) * diff + target_shape

    # 目标维度为 1、梯度对应维度不为 1，说明正向时该轴发生了广播。
    axes = []
    for i in range(len(grad.shape)):
        if padded_shape[i] == 1 and grad.shape[i] != 1:
            axes.append(i)

    # 空列表在布尔上下文中为 False，非空列表为 True。
    if axes:
        grad = summation(grad, axes=tuple(axes))

    # 求和会移除对应轴，最后 reshape 恢复长度为 1 的维度。
    return reshape(grad, target_shape)


class BroadcastTo(TensorOp):
    """依据 NumPy 广播规则把 Tensor 扩展到目标形状。"""

    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_shape = node.inputs[0].shape
        # 广播的反向过程是在所有被复制的轴上累加梯度。
        return sum_to_shape(out_grad, input_shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    """沿指定轴求和；axes=None 表示对所有元素求和。"""

    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_shape = node.inputs[0].shape

        if self.axes is None:
            return broadcast_to(out_grad, input_shape)

        axes = self.axes
        # isinstance 用于运行时类型判断；把单个 int 统一包装成元组便于后续遍历。
        if isinstance(axes, int):
            axes = (axes,)

        # Python/NumPy 允许负轴编号；对 ndim 取模可转换为对应的非负编号。
        ndim = len(input_shape)
        # tuple(generator) 消费生成器表达式，并构造不可变元组。
        axes = tuple(axis % ndim for axis in axes)

        # 求和移除了指定轴；先在这些位置补回长度 1，再广播至输入形状。
        # list 可修改，因此先用列表逐项赋值，最后再转回元组。
        original_shape = list(input_shape)
        for axis in axes:
            original_shape[axis] = 1

        grad = reshape(out_grad, tuple(original_shape))
        return broadcast_to(grad, input_shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    """矩阵乘法，支持 NumPy 风格的批次维度广播。"""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.matmul(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        # 对最后两个矩阵轴应用 d(AB)/dA = dY B^T、d(AB)/dB = A^T dY。
        left_grad = matmul(out_grad, transpose(rhs))
        right_grad = matmul(transpose(lhs), out_grad)

        # 批次维度可能发生广播，需要把梯度归约回各输入的原始形状。
        return sum_to_shape(left_grad, lhs.shape), sum_to_shape(right_grad, rhs.shape)
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    """逐元素取相反数。"""

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -1 * out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    """逐元素计算自然对数。"""

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / node.inputs[0]
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    """逐元素计算自然指数。"""

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * exp(node.inputs[0])
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    """逐元素计算 max(a, 0)。"""

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # Needle Tensor 尚未实现 __gt__，因此读取底层数组来构造布尔掩码。
        input_data = node.inputs[0].realize_cached_data()
        # 比较产生布尔数组；astype 再把 True/False 转成 1.0/0.0。
        mask = (input_data > 0).astype("float32")

        # requires_grad=False 表示掩码是常量，不需要继续对它求导。
        return out_grad * Tensor(mask, requires_grad=False)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)
