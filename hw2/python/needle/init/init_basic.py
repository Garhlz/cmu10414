"""基础 Tensor 初始化：随机分布、常量、独热编码及同形状张量。"""

from __future__ import annotations

from numpy.typing import DTypeLike
from ..backend_numpy import CPUDevice
import needle as ndl


def rand(
    *shape: int,
    low: float = 0.0,
    high: float = 1.0,
    device: CPUDevice | None = None,
    dtype: DTypeLike = "float32",
    requires_grad: bool = False,
) -> ndl.Tensor:
    """从 [low, high) 均匀采样；形状元组需通过 *shape 展开后传入。"""
    device = ndl.cpu() if device is None else device
    array = device.rand(*shape) * (high - low) + low
    return ndl.Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


def randn(
    *shape: int,
    mean: float = 0.0,
    std: float = 1.0,
    device: CPUDevice | None = None,
    dtype: DTypeLike = "float32",
    requires_grad: bool = False,
) -> ndl.Tensor:
    """按给定均值和标准差从正态分布采样。"""
    device = ndl.cpu() if device is None else device
    array = device.randn(*shape) * std + mean
    return ndl.Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


def constant(
    *shape: int,
    c: float = 1.0,
    device: CPUDevice | None = None,
    dtype: DTypeLike = "float32",
    requires_grad: bool = False,
) -> ndl.Tensor:
    """创建指定形状、所有元素均为 c 的 Tensor。"""
    device = ndl.cpu() if device is None else device
    array = device.ones(*shape, dtype=dtype) * c  # 最终 dtype 由 Tensor 构造函数保证。
    return ndl.Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


def ones(
    *shape: int,
    device: CPUDevice | None = None,
    dtype: DTypeLike = "float32",
    requires_grad: bool = False,
) -> ndl.Tensor:
    """创建全 1 Tensor。"""
    return constant(
        *shape, c=1.0, device=device, dtype=dtype, requires_grad=requires_grad
    )


def zeros(
    *shape: int,
    device: CPUDevice | None = None,
    dtype: DTypeLike = "float32",
    requires_grad: bool = False,
) -> ndl.Tensor:
    """创建全 0 Tensor。"""
    return constant(
        *shape, c=0.0, device=device, dtype=dtype, requires_grad=requires_grad
    )


def randb(
    *shape: int,
    p: float = 0.5,
    device: CPUDevice | None = None,
    dtype: DTypeLike = "bool",
    requires_grad: bool = False,
) -> ndl.Tensor:
    """创建伯努利随机 Tensor，p 表示元素为 1 的概率。"""
    device = ndl.cpu() if device is None else device
    array = device.rand(*shape) <= p
    return ndl.Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


def one_hot(
    n: int,
    i: ndl.Tensor,
    device: CPUDevice | None = None,
    dtype: DTypeLike = "float32",
    requires_grad: bool = False,
) -> ndl.Tensor:
    """将类别索引 Tensor 转为独热编码，末尾新增长度为 n 的类别维。"""
    device = ndl.cpu() if device is None else device
    return ndl.Tensor(
        device.one_hot(n, i.numpy().astype("int32"), dtype=dtype),
        device=device,
        requires_grad=requires_grad,
    )


def zeros_like(
    array: ndl.Tensor, *, device: CPUDevice | None = None, requires_grad: bool = False
) -> ndl.Tensor:
    """创建与输入形状和类型相同的全 0 Tensor。"""
    device = device if device else array.device
    return zeros(
        *array.shape, dtype=array.dtype, device=device, requires_grad=requires_grad
    )


def ones_like(
    array: ndl.Tensor, *, device: CPUDevice | None = None, requires_grad: bool = False
) -> ndl.Tensor:
    """创建与输入形状和类型相同的全 1 Tensor。"""
    device = device if device else array.device
    return ones(
        *array.shape, dtype=array.dtype, device=device, requires_grad=requires_grad
    )
