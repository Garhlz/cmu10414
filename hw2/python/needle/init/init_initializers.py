"""权重初始化：使用输入和输出维度控制随机权重的尺度。"""

from __future__ import annotations

import math
from .init_basic import *
from typing import TYPE_CHECKING, TypedDict, Unpack
from numpy.typing import DTypeLike
from ..backend_numpy import CPUDevice

if TYPE_CHECKING:
    # 仅供类型检查器读取，避免初始化模块与 autograd 在运行时循环导入。
    from ..autograd import Tensor


class InitOptions(TypedDict, total=False):
    """可选的初始化关键字；Unpack 将这些键展开为 **kwargs 的类型约束。"""

    device: CPUDevice | None
    dtype: DTypeLike
    requires_grad: bool


def xavier_uniform(
    fan_in: int, fan_out: int, gain: float = 1.0, **kwargs: Unpack[InitOptions]
) -> "Tensor":
    """Xavier 均匀初始化，边界为 gain * sqrt(6 / (fan_in + fan_out))。"""
    ### BEGIN YOUR SOLUTION
    a = gain * math.sqrt(6 / (fan_in + fan_out))
    return rand(fan_in, fan_out, low=-a, high=a, **kwargs)
    ### END YOUR SOLUTION


def xavier_normal(
    fan_in: int, fan_out: int, gain: float = 1.0, **kwargs: Unpack[InitOptions]
) -> "Tensor":
    """Xavier 正态初始化，标准差为 gain * sqrt(2 / (fan_in + fan_out))。"""
    ### BEGIN YOUR SOLUTION
    std = gain * math.sqrt(2 / (fan_in + fan_out))
    return randn(fan_in, fan_out, std=std, **kwargs)
    ### END YOUR SOLUTION


def kaiming_uniform(
    fan_in: int, fan_out: int, nonlinearity: str = "relu", **kwargs: Unpack[InitOptions]
) -> "Tensor":
    """面向 ReLU 的 Kaiming 均匀初始化，增益固定为 sqrt(2)。"""
    assert nonlinearity == "relu", "当前只支持 relu"
    ### BEGIN YOUR SOLUTION
    bound = math.sqrt(2) * math.sqrt(3 / fan_in)
    return rand(fan_in, fan_out, low=-bound, high=bound, **kwargs)
    ### END YOUR SOLUTION


def kaiming_normal(
    fan_in: int, fan_out: int, nonlinearity: str = "relu", **kwargs: Unpack[InitOptions]
) -> "Tensor":
    """面向 ReLU 的 Kaiming 正态初始化，标准差为 sqrt(2 / fan_in)。"""
    assert nonlinearity == "relu", "当前只支持 relu"
    std = math.sqrt(2) * math.sqrt(1 / fan_in)
    return randn(fan_in, fan_out, std=std, **kwargs)
    ### END YOUR SOLUTION
