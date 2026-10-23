"""数值稳定的对数算子及其梯度；归约轴的约定与 Summation 一致。"""

from typing import Optional, Any, Union
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api


class LogSoftmax(TensorOp):
    """沿二维输入的类别轴计算对数概率，输出形状与输入相同。"""

    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        max_z = array_api.max(Z, axis=1, keepdims=True)

        shifted = Z - max_z

        log_sum = array_api.log(
            array_api.sum(array_api.exp(shifted), axis=1, keepdims=True)
        )

        return shifted - log_sum

        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # node 就是 LogSoftmax(Z) 的输出
        # 因此 exp(node) = Softmax(Z)
        probs = exp(node)

        # 每行共享同一个 LogSumExp 值，反向时累加这些共享路径的梯度。
        grad_sum = summation(out_grad, axes=(1,))

        grad_sum = grad_sum.reshape((out_grad.shape[0], 1))

        grad_sum = grad_sum.broadcast_to(out_grad.shape)

        return out_grad - probs * grad_sum
        ### END YOUR SOLUTION


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    """计算 log(sum(exp(Z)))，正向减去最大值来避免指数上溢。

    只移除 axes 指定的维度；axes=None 时才归约所有元素并返回标量。
    """

    def __init__(self, axes: int | tuple[int, ...] | None = None) -> None:
        self.axes = axes

    # axes=None 时输出形状为 ()，即零维数组。
    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        max_z = array_api.max(Z, axis=self.axes, keepdims=True)
        shifted = Z - max_z
        sum_exp = array_api.sum(array_api.exp(shifted), axis=self.axes, keepdims=True)

        result = array_api.log(sum_exp) + max_z

        if self.axes is None:
            return array_api.reshape(result, ())  # 对所有元素求和，返回的是一个标量

        return array_api.squeeze(result, axis=self.axes)  # 删除指定的、长度为 1 的维度
        ### END YOUR SOLUTION

    # 计算可以发现，这里的梯度实际上相当于对输入的tensor求softmax
    # d(LogSumExp)/dZ = Softmax(Z)
    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        input = node.inputs[0]

        # 确定被求和消除的维度
        if self.axes is None:
            # 函数变换之后变成了标量，所有维度都被消除了
            axes = tuple(range(len(input.shape)))
        elif isinstance(self.axes, int):
            axes = (self.axes,)
        else:
            axes = self.axes

        shape = list(input.shape)
        for axis in axes:
            shape[axis] = 1
        shape = tuple(shape)

        grad = reshape(out_grad, shape)
        grad = broadcast_to(grad, input.shape)

        # d(LogSumExp)/dZ = Softmax(Z)
        # exp(Z - LogSumExp(Z)) 就是 Softmax(Z)
        lse = reshape(node, shape)
        lse = broadcast_to(lse, input.shape)
        softmax = exp(input - lse)

        return grad * softmax
        ### END YOUR SOLUTION


def logsumexp(a: Tensor, axes: int | tuple[int, ...] | None = None) -> Tensor:
    return LogSumExp(axes=axes)(a)
