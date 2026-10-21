import math
import needle as ndl


def rand(*shape, low=0.0, high=1.0, device=None, dtype="float32", requires_grad=False):
    """生成服从 [low, high) 均匀分布的随机 Tensor。"""
    # 条件表达式在 device 未指定时选择 CPU，否则保留调用者传入的设备。
    device = ndl.cpu() if device is None else device
    array = device.rand(*shape) * (high - low) + low
    return ndl.Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


def randn(*shape, mean=0.0, std=1.0, device=None, dtype="float32", requires_grad=False):
    """生成指定均值和标准差的正态分布随机 Tensor。"""
    device = ndl.cpu() if device is None else device
    array = device.randn(*shape) * std + mean
    return ndl.Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


def constant(*shape, c=1.0, device=None, dtype="float32", requires_grad=False):
    """生成所有元素均为常数 c 的 Tensor。"""
    device = ndl.cpu() if device is None else device
    # 先创建全 1 数组再乘 c；dtype 通过关键字参数传入。
    array = device.ones(*shape, dtype=dtype) * c
    return ndl.Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)





def ones(*shape, device=None, dtype="float32", requires_grad=False):
    """生成全 1 Tensor。"""
    # *shape 将形状元组解包后继续转发给 constant。
    return constant(
        *shape, c=1.0, device=device, dtype=dtype, requires_grad=requires_grad
    )


def zeros(*shape, device=None, dtype="float32", requires_grad=False):
    """生成全 0 Tensor。"""
    return constant(
        *shape, c=0.0, device=device, dtype=dtype, requires_grad=requires_grad
    )


def randb(*shape, p=0.5, device=None, dtype="bool", requires_grad=False):
    """按概率 p 生成元素为 True/False 的伯努利随机 Tensor。"""
    device = ndl.cpu() if device is None else device
    array = device.rand(*shape) <= p
    return ndl.Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


def one_hot(n, i, device=None, dtype="float32", requires_grad=False):
    """将整数类别 Tensor i 转换成具有 n 个类别的 one-hot Tensor。"""
    # 后端索引需要整数 NumPy 数组，因此先脱离计算图并转换成 int32。
    device = ndl.cpu() if device is None else device
    return ndl.Tensor(
        device.one_hot(n, i.numpy().astype("int32"), dtype=dtype),
        device=device,
        requires_grad=requires_grad,
    )


def zeros_like(array, *, device=None, requires_grad=False):
    """生成与 array 形状和 dtype 相同的全 0 Tensor。"""
    # 函数签名中的 * 表示后续参数只能按关键字传入。
    device = device if device else array.device
    return zeros(
        *array.shape, dtype=array.dtype, device=device, requires_grad=requires_grad
    )


def ones_like(array, *, device=None, requires_grad=False):
    """生成与 array 形状和 dtype 相同的全 1 Tensor。"""
    device = device if device else array.device
    return ones(
        *array.shape, dtype=array.dtype, device=device, requires_grad=requires_grad
    )
