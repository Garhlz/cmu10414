"""使用 NumPy 作为 NDArray 后端时的设备实现。"""
import numpy


class Device:
    """所有计算设备的基类。"""


class CPUDevice(Device):
    """表示存放在 CPU 上、由 NumPy 管理的数据。"""

    def __repr__(self):
        # __repr__ 决定调试或交互式环境中显示设备对象时使用的字符串。
        return "needle.cpu()"

    def __hash__(self):
        # 可哈希对象才能作为 dict 的键或放入 set；这里复用表示字符串的哈希值。
        return self.__repr__().__hash__()

    def __eq__(self, other):
        # isinstance 同时接受当前类及其子类实例。
        return isinstance(other, CPUDevice)

    def enabled(self):
        return True

    def zeros(self, *shape, dtype="float32"):
        # *shape 把若干位置参数收集为元组，例如 zeros(2, 3) 得到 shape=(2, 3)。
        return numpy.zeros(shape, dtype=dtype)

    def ones(self, *shape, dtype="float32"):
        return numpy.ones(shape, dtype=dtype)

    def randn(self, *shape):
        # NumPy 的传统随机函数不直接接受 dtype；Tensor 构造时会再转换类型。
        return numpy.random.randn(*shape)

    def rand(self, *shape):
        # *shape 在调用时又把形状元组解包为 rand 所需的多个位置参数。
        return numpy.random.rand(*shape)

    def one_hot(self, n, i, dtype="float32"):
        # 先创建 n 阶单位矩阵，再用整数数组 i 进行高级索引以选取对应行。
        return numpy.eye(n, dtype=dtype)[i]

    def empty(self, shape, dtype="float32"):
        return numpy.empty(shape, dtype=dtype)

    def full(self, shape, fill_value, dtype="float32"):
        return numpy.full(shape, fill_value, dtype=dtype)


def cpu():
    """返回 CPU 设备对象。"""
    return CPUDevice()


def default_device():
    """返回框架默认设备；HW1 中固定为 CPU。"""
    return cpu()


def all_devices():
    """返回所有可用设备；HW1 只有 CPU。"""
    return [cpu()]
