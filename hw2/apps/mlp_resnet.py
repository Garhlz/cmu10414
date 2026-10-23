"""MNIST 残差网络的构建、单轮训练/评估与完整训练入口。"""

from collections.abc import Callable, Iterable
import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os
from typing import Optional

np.random.seed(0)


def ResidualBlock(
    dim: int,
    hidden_dim: int,
    norm: Callable[[int], nn.Module] = nn.BatchNorm1d,
    drop_prob: float = 0.1,
) -> nn.Module:
    """创建独立的残差块；norm 接收模块类，调用 norm(dim) 才创建实例。"""
    # 递归完成一个残差块中所有模块的初始化
    # 输入输出都是dim
    ### BEGIN YOUR SOLUTION
    return nn.Sequential(
        nn.Residual(
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                norm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(drop_prob),
                nn.Linear(hidden_dim, dim),
                norm(dim),
            )
        ),
        nn.ReLU(),
    )
    ### END YOUR SOLUTION


def MLPResNet(
    dim: int,
    hidden_dim: int = 100,
    num_blocks: int = 3,
    num_classes: int = 10,
    norm: Callable[[int], nn.Module] = nn.BatchNorm1d,
    drop_prob: float = 0.1,
) -> nn.Module:
    """按前向执行顺序初始化各层，重复创建的残差块分别持有自己的参数。"""
    # 在这里递归完成了所有子模块的初始化
    ### BEGIN YOUR SOLUTION
    first_layers = [
        nn.Flatten(),
        nn.Linear(dim, hidden_dim),
        nn.ReLU(),
    ]

    blocks = [
        ResidualBlock(hidden_dim, hidden_dim // 2, norm, drop_prob)
        for _ in range(num_blocks)
    ]

    last_layers = [
        nn.Linear(hidden_dim, num_classes),
    ]

    return nn.Sequential(*first_layers, *blocks, *last_layers)
    ### END YOUR SOLUTION


def epoch(
    dataloader: Iterable[tuple[ndl.Tensor, ...]],
    model: nn.Module,
    opt: Optional[ndl.optim.Optimizer] = None,
) -> tuple[float, float]:
    """遍历一轮，返回按样本数量加权的 (错误率, 平均损失)。

    opt=None 时仅评估；否则每个批次清空梯度、反向传播并更新参数。
    接收任意批次可迭代对象，因此也能使用带进度条的 DataLoader 包装器。
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    # 如果opt is None，启用评估模式
    if opt is None:
        # 评估模式
        model.eval()
    else:
        model.train()

    loss_func = nn.SoftmaxLoss()
    total_loss = 0.0
    total_error = 0
    total_samples = 0

    for X, y in dataloader:
        batch_size = X.shape[0]

        if opt is not None:
            opt.reset_grad()  # 重置梯度

        logits = model(X)  # __call__()，递归调用forward函数，完成一次完整的前向传播
        loss = loss_func(logits, y)

        predictions = np.argmax(logits.numpy(), axis=1)
        total_error += np.sum(predictions != y.numpy())
        total_loss += float(loss.numpy()) * batch_size
        total_samples += batch_size

        if opt is not None:
            loss.backward()  # 反向传播，递归计算图，求出并保存每个节点相对于最终输出loss的梯度
            opt.step()  # 对于所有parameter，按照优化器规则更新参数（lr）

    # 返回顺序遵循课程接口：错误率在前，平均损失在后。
    return float(total_error / total_samples), float(total_loss / total_samples)
    ### END YOUR SOLUTION


def train_mnist(
    batch_size: int = 100,
    epochs: int = 10,
    optimizer: Callable[..., ndl.optim.Optimizer] = ndl.optim.Adam,
    lr: float = 0.001,
    weight_decay: float = 0.001,
    hidden_dim: int = 100,
    data_dir: str = "data",
) -> tuple[float, float, float, float]:
    """训练并逐轮评估，返回最终的训练错误率、训练损失、测试错误率和测试损失。

    optimizer 接收优化器类；训练循环外创建实例，以跨批次保留动量等状态。
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    train_dataset = ndl.data.MNISTDataset(
        os.path.join(data_dir, "train-images-idx3-ubyte.gz"),
        os.path.join(data_dir, "train-labels-idx1-ubyte.gz"),
    )

    test_dataset = ndl.data.MNISTDataset(
        os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"),
        os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"),
    )

    train_dataloader = ndl.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    test_dataloader = ndl.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    model = MLPResNet(dim=28 * 28, hidden_dim=hidden_dim)

    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    for i in range(epochs):
        train_error, train_loss = epoch(train_dataloader, model, opt)
        test_error, test_loss = epoch(test_dataloader, model)

        print(
            f"Epoch {i + 1}/{epochs} | "
            f"train loss={train_loss:.4f}, error={train_error:.2%} | "
            f"test loss={test_loss:.4f}, error={test_error:.2%}",
            flush=True,
        )

    return train_error, train_loss, test_error, test_loss
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
