"""基于 Needle 实现的 MNIST 数据读取、损失函数与训练流程。"""

import struct
import gzip
import numpy as np

import sys

# 将项目内的 python/ 追加到模块搜索路径，随后即可直接 import needle。
# append 修改列表本身；这里的相对路径以运行程序时的当前目录为基准。
sys.path.append("python/")
import needle as ndl


def parse_mnist(image_filename, label_filename):
    """读取 gzip 压缩的 MNIST 图片与标签文件。

    参数：
        image_filename (str)：MNIST 图片文件路径。
        label_filename (str)：MNIST 标签文件路径。

    返回：
        tuple：由 ``(X, y)`` 组成。

        - ``X``：形状为 ``(样本数, 行数 * 列数)`` 的 float32 数组，
          像素值由 [0, 255] 归一化到 [0.0, 1.0]。
        - ``y``：形状为 ``(样本数,)`` 的 uint8 标签数组，值域为 0～9。
    """
    ### BEGIN YOUR CODE
    with gzip.open(image_filename, "rb") as f:
        raw = f.read()
        # struct 格式中 > 表示大端序，I 表示 4 字节无符号整数。
        # raw[:16] 是字节切片；unpack 将其解析为四个 Python 整数。
        magic, num_images, rows, cols = struct.unpack(">IIII", raw[:16])
        assert magic == 2051  # 图像文件

        # frombuffer 不逐个复制字节，而是从跳过 16 字节文件头的位置创建数组视图。
        pixels = np.frombuffer(raw, dtype=np.uint8, offset=16)
        X = pixels.reshape(num_images, rows * cols)
        X = X.astype(np.float32) / 255.0

    with gzip.open(label_filename, "rb") as f:
        raw = f.read()
        magic, num_labels = struct.unpack(">II", raw[:8])
        assert magic == 2049
        assert num_labels == num_images
        labels = np.frombuffer(raw, dtype=np.uint8, offset=8)
    return X, labels
    ### END YOUR CODE


def softmax_loss(Z, y_one_hot) -> ndl.Tensor:
    """计算一个批次的平均 Softmax 交叉熵损失。

    本题不要求使用数值稳定的 log-sum-exp 技巧，可以直接计算指数与对数。

    参数：
        Z：形状为 ``(batch_size, num_classes)`` 的 logits Tensor。
        y_one_hot：同形状的 one-hot 标签 Tensor；真实类别位置为 1，其余为 0。

    返回：
        标量 Needle Tensor，表示批次内样本损失的平均值。
    """

    ### BEGIN YOUR SOLUTION
    batch_size = Z.shape[0]
    exp_Z = ndl.exp(Z)
    normalizer = ndl.summation(exp_Z, (1,))  # 求和会改变形状
    normalizer = ndl.reshape(normalizer, (batch_size, 1))
    normalizer = ndl.broadcast_to(normalizer, Z.shape)

    P = exp_Z / normalizer  # softmax概率 (batch_size, num_classes)

    correct_probs = P * y_one_hot  # 只保留正确class的概率
    correct_probs = ndl.summation(correct_probs, (1,))  # (batch_size, )
    correct_probs = -ndl.log(correct_probs)  # 每个batch的交叉熵损失
    total_loss = ndl.summation(correct_probs)
    return total_loss / batch_size  # 这一批batch的平均交叉熵损失
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1: ndl.Tensor, W2: ndl.Tensor, lr=0.1, batch=100):
    """对无偏置的两层神经网络执行一轮小批量 SGD。

    网络定义为：

        logits = ReLU(X * W1) * W2

    按原顺序遍历 X，不在每轮开始前随机打乱样本。

    参数：
        X：形状为 ``(样本数, 输入维度)`` 的 NumPy 输入数组。
        y：形状为 ``(样本数,)`` 的类别编号数组。
        W1：形状为 ``(输入维度, 隐藏维度)`` 的 Needle Tensor。
        W2：形状为 ``(隐藏维度, 类别数)`` 的 Needle Tensor。
        lr：SGD 学习率。
        batch：小批量大小。

    返回：
        tuple：更新后的 ``(W1, W2)``。
    """

    ### BEGIN YOUR SOLUTION
    total = X.shape[0]
    num_classes = W2.shape[1]
    for start in range(0, total, batch):
        X_b = X[start : start + batch]
        y_b = y[start : start + batch]

        # 转换为needle的tensor类
        X_tensor = ndl.Tensor(X_b, requires_grad=False)
        y_tensor = ndl.Tensor(y_b, requires_grad=False)

        # 转换为独热编码索引 (batch_size, num_classes)
        y_onehot = ndl.init.one_hot(num_classes, y_tensor, requires_grad=False)

        # 调用needle的算子类，实际上完成了算子的创建、正向计算、缓存结果
        # 见TensorOp.__call__相关注释
        Z1 = X_tensor @ W1
        hidden = ndl.relu(Z1)
        Z2 = hidden @ W2

        loss = softmax_loss(Z2, y_onehot)
        loss.backward()

        # 脱离当前计算图，在 NumPy 层执行 SGD。
        new_W1_data = W1.numpy() - lr * W1.grad.numpy()
        new_W2_data = W2.numpy() - lr * W2.grad.numpy()

        # 为下一个 minibatch 创建新的叶子 Tensor。
        W1 = ndl.Tensor(new_W1_data, requires_grad=True)
        W2 = ndl.Tensor(new_W2_data, requires_grad=True)
    return W1, W2
    ### END YOUR SOLUTION


### 以下代码仅作辅助展示，无需修改


def loss_err(h, y):
    """根据 logits 和整数标签同时计算损失与分类错误率。"""
    # NumPy 高级索引同时提供行、列索引，从而把每行真实类别的位置设为 1。
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
