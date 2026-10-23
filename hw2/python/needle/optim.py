"""优化器：读取反向传播得到的梯度，更新参数数据与历史状态。"""

from collections.abc import Sequence

import needle as ndl
import numpy as np


class Optimizer:
    """优化器基类；所有批次共用同一组参数对象。"""

    def __init__(self, params: Sequence[ndl.Tensor]) -> None:
        self.params = params

    def step(self) -> None:
        raise NotImplementedError()

    def reset_grad(self) -> None:
        """清除上一次反向传播保存的梯度，解除对旧梯度图的引用。"""
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    """带动量和 L2 权重衰减的随机梯度下降。"""

    def __init__(
        self,
        params: Sequence[ndl.Tensor],
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum  # 动量
        self.u: dict[ndl.Tensor, ndl.Tensor | int] = {}
        self.weight_decay = weight_decay
        for param in self.params:
            self.u[param] = 0

    def step(self) -> None:
        ### BEGIN YOUR SOLUTION
        for param in self.params:
            grad = param.grad.detach() + self.weight_decay * param.data  # 权重衰减

            u = self.u[param] * self.momentum + grad * (1 - self.momentum)
            # 保存梯度的移动平均；动量系数 momentum 本身保持不变。
            self.u[param] = u

            # data 的 setter 替换底层数值，保留模型中原 Parameter 对象。
            param.data = param.data - u * self.lr
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm: float = 0.25) -> None:
        """裁剪参数梯度的范数；HW2 不要求实现，保留课程占位接口。"""
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    """保存梯度及其平方的移动平均，并使用偏差修正进行自适应更新。"""

    def __init__(
        self,
        params: Sequence[ndl.Tensor],
        lr: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m: dict[ndl.Tensor, ndl.Tensor | int] = {}
        self.v: dict[ndl.Tensor, ndl.Tensor | int] = {}
        for param in self.params:
            self.m[param] = 0
            self.v[param] = 0

    def step(self) -> None:
        ### BEGIN YOUR SOLUTION
        # t 统计参数更新次数，而非 epoch 数；同一次更新的所有参数共用 t。
        self.t += 1
        for param in self.params:
            grad = param.grad.detach() + self.weight_decay * param.data  # 权重衰减

            m = self.m[param] * self.beta1 + grad * (1 - self.beta1)
            self.m[param] = m

            v = self.v[param] * self.beta2 + (grad**2) * (1 - self.beta2)
            self.v[param] = v

            # 修正值仅用于本次更新，历史字典仍保存未经修正的 m、v。
            m_hat = m / (1 - self.beta1**self.t)
            v_hat = v / (1 - self.beta2**self.t)

            param.data = param.data - self.lr * m_hat / (v_hat**0.5 + self.eps)
        ### END YOUR SOLUTION
