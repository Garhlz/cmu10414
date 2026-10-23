"""The module.
"""
from typing import Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> list[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> list["Module"]:
    if isinstance(value, Module):
        modules = [value]
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
    def __init__(self) -> None:
        self.training = True

    def parameters(self) -> list[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> list["Module"]:
        return _child_modules(self.__dict__)

    def eval(self) -> None:
        self.training = False
        for m in self._children():
            m.training = False

    def train(self) -> None:
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        weight = init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype, requires_grad=True)
        self.weight = Parameter(weight)
        self.bias = None
        if bias:
            bias = init.kaiming_uniform(out_features, 1, device=device, dtype=dtype, requires_grad=True)
            bias = ops.reshape(bias, (out_features,))
            self.bias = Parameter(bias)
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        Z = X @ self.weight
        if self.bias is not None:
            bias = ops.broadcast_to(self.bias, (X.shape[0], self.out_features))
            return Z + bias
        return Z
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size = X.shape[0]
        return ops.reshape(X, (batch_size, -1))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        output = x
        for module in self.modules:
            output = module(output)
        return output
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size, num_classes = logits.shape
        y_onehot = init.one_hot(num_classes, y, device=logits.device, dtype=logits.dtype, requires_grad=False)
        lse = ops.logsumexp(logits, axes=(1,))
        true_logits = ops.summation(logits * y_onehot, axes=(1,))
        losses = lse - true_logits
        return ops.summation(losses) / batch_size
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        weight = init.ones(dim, device=device, dtype=dtype, requires_grad=True)
        bias = init.zeros(dim, device=device, dtype=dtype, requires_grad=True)
        self.weight = Parameter(weight)
        self.bias = Parameter(bias)
        self.running_mean = init.zeros(dim, device=device, dtype=dtype, requires_grad=False)
        self.running_var = init.ones(dim, device=device, dtype=dtype, requires_grad=False)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size = x.shape[0]
        if self.training:
            mean = ops.summation(x, axes=(0,)) / batch_size
            reshaped_mean = ops.reshape(mean, (1, self.dim))
            centered = x - ops.broadcast_to(reshaped_mean, x.shape)
            var = ops.summation(centered ** 2, axes=(0,)) / batch_size
            self.running_mean = ((1 - self.momentum) * self.running_mean + self.momentum * mean.detach()).detach()
            self.running_var = ((1 - self.momentum) * self.running_var + self.momentum * var.detach()).detach()
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
    def __init__(self, dim: int, eps: float = 1e-5, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        weight = init.ones(dim, device=device, dtype=dtype, requires_grad=True)
        bias = init.zeros(dim, device=device, dtype=dtype, requires_grad=True)
        self.weight = Parameter(weight)
        self.bias = Parameter(bias)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size = x.shape[0]
        mean = ops.summation(x, axes=(1,)) / self.dim
        mean = ops.reshape(mean, (batch_size, 1))
        centered = x - ops.broadcast_to(mean, x.shape)
        var = ops.summation(centered ** 2, axes=(1,)) / self.dim
        denom = (var + self.eps) ** 0.5
        denom = ops.reshape(denom, (batch_size, 1))
        normed = centered / ops.broadcast_to(denom, x.shape)
        weight = ops.broadcast_to(self.weight, x.shape)
        bias = ops.broadcast_to(self.bias, x.shape)
        return weight * normed + bias
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if not self.training:
            return x
        mask = init.randb(*x.shape, p=1 - self.p, device=x.device, dtype=x.dtype, requires_grad=False)
        return x * mask / (1 - self.p)
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return x + self.fn(x)
        ### END YOUR SOLUTION
