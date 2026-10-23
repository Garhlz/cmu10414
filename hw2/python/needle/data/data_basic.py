"""数据集与批次迭代器：在 NumPy 数据和 Needle Tensor 之间衔接。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeAlias

import numpy as np
from ..autograd import Tensor

# 整数取单个样本；切片、索引序列或布尔数组取一批样本。
Index: TypeAlias = int | slice | Sequence[int] | np.ndarray
Sample: TypeAlias = tuple[np.ndarray | np.generic, ...]
Transform: TypeAlias = Callable[[np.ndarray], np.ndarray]


class Dataset:
    """数据集基类：子类实现按索引取样和样本数量查询。

    返回元组中的每一项代表一种数据，例如图像、标签；可支持任意数量的项。
    图像变换在 NumPy 数组上执行，Tensor 转换由 DataLoader 负责。
    """

    def __init__(self, transforms: Sequence[Transform] | None = None) -> None:
        self.transforms = transforms

    def __getitem__(self, index: Index) -> Sample:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x: np.ndarray) -> np.ndarray:
        """按声明顺序对单个样本应用变换。"""
        if self.transforms is not None:
            # 上一个变换的输出作为下一个变换的输入。
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    """按批次遍历数据集，并把每个数据项转换为不需要梯度的 Tensor。

    batch_size 指定批次大小，最后一批可不足该大小。
    shuffle=True 时，每次开始遍历都会重新打乱样本索引。
    ordering 保存各批次的索引数组；index 是批次编号，而非样本编号。
    """

    dataset: Dataset
    batch_size: int
    ordering: list[np.ndarray]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
    ) -> None:

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.index = 0
        # 先生成全部索引，再按批次边界切分
        if not self.shuffle:
            self.ordering = np.array_split(
                np.arange(len(dataset)), range(batch_size, len(dataset), batch_size)
            )

    def __iter__(self) -> DataLoader:
        """开始新一轮遍历，必要时打乱，并将批次游标归零。"""
        ### BEGIN YOUR SOLUTION
        if self.shuffle:
            shuffled = np.arange(len(self.dataset))
            np.random.shuffle(shuffled)
            self.ordering = np.array_split(
                shuffled, range(self.batch_size, len(self.dataset), self.batch_size)
            )

        self.index = 0
        ### END YOUR SOLUTION
        return self

    def __next__(self) -> tuple[Tensor, ...]:
        """返回下一批 Tensor；耗尽后抛出 StopIteration 结束 for 循环。"""
        ### BEGIN YOUR SOLUTION
        if self.index >= len(self.ordering):
            raise StopIteration
        # 元组长度由数据集决定，不假定一定包含图像和标签两个数组。
        batch = self.dataset[self.ordering[self.index]]
        self.index += 1
        return tuple(
            Tensor(array, dtype=array.dtype, requires_grad=False) for array in batch
        )
        ### END YOUR SOLUTION
