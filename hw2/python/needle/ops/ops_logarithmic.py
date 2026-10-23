from typing import Optional, Any, Union
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        max_z = array_api.max(Z, axis=1, keepdims=True)
        shifted = Z - max_z
        log_sum = array_api.log(array_api.sum(array_api.exp(shifted), axis=1, keepdims=True))
        return shifted - log_sum
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        probs = exp(node)
        grad_sum = summation(out_grad, axes=(1,))
        grad_sum = grad_sum.reshape((out_grad.shape[0], 1))
        grad_sum = grad_sum.broadcast_to(out_grad.shape)
        return out_grad - probs * grad_sum
        ### END YOUR SOLUTION


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        max_z = array_api.max(Z, axis=self.axes, keepdims=True)
        shifted = Z - max_z
        sum_exp = array_api.sum(array_api.exp(shifted), axis=self.axes, keepdims=True)
        result = array_api.log(sum_exp) + max_z
        if self.axes is None:
            return array_api.reshape(result, ())
        return array_api.squeeze(result, axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        input = node.inputs[0]
        if self.axes is None:
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
        lse = reshape(node, shape)
        lse = broadcast_to(lse, input.shape)
        softmax = exp(input - lse)
        return grad * softmax
        ### END YOUR SOLUTION


def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)