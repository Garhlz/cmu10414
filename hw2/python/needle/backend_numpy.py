"""NumPy 数组后端的设备接口；当前实现只支持 CPU。"""

from typing import Any

import numpy
from numpy.typing import DTypeLike


class Device:
    """设备的公共基类，供 Tensor 的设备属性使用。"""


class CPUDevice(Device):
    """CPU 设备，通过 NumPy 创建数组并执行随机采样。"""

    def __repr__(self) -> str:
        return "needle.cpu()"

    def __hash__(self) -> int:
        return self.__repr__().__hash__()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, CPUDevice)

    def enabled(self) -> bool:
        return True

    def zeros(self, *shape: int, dtype: DTypeLike = "float32") -> numpy.ndarray:
        return numpy.zeros(shape, dtype=dtype)

    def ones(self, *shape: int, dtype: DTypeLike = "float32") -> numpy.ndarray:
        return numpy.ones(shape, dtype=dtype)

    def randn(self, *shape: int) -> numpy.ndarray | float:
        # NumPy 的此随机接口不接收 dtype；上层 Tensor 构造函数负责转换类型。
        # 未传形状时可能返回标量，因此返回注解也包含 float。
        return numpy.random.randn(*shape)

    def rand(self, *shape: int) -> numpy.ndarray | float:
        # 随机值范围为 [0, 1)，dtype 由上层初始化函数交给 Tensor 处理。
        return numpy.random.rand(*shape)

    def one_hot(
        self, n: int, i: numpy.ndarray, dtype: DTypeLike = "float32"
    ) -> numpy.ndarray:
        return numpy.eye(n, dtype=dtype)[i]

    def empty(
        self, shape: tuple[int, ...], dtype: DTypeLike = "float32"
    ) -> numpy.ndarray:
        return numpy.empty(shape, dtype=dtype)

    def full(
        self, shape: tuple[int, ...], fill_value: Any, dtype: DTypeLike = "float32"
    ) -> numpy.ndarray:
        return numpy.full(shape, fill_value, dtype=dtype)


def cpu() -> CPUDevice:
    """返回 CPU 设备对象。"""
    return CPUDevice()


def default_device() -> CPUDevice:
    return cpu()


def all_devices() -> list[CPUDevice]:
    """返回当前后端支持的全部设备。"""
    return [cpu()]
