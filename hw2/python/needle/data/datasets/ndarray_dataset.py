from ..data_basic import Dataset
from ..data_basic import Index, Sample
import numpy as np


class NDArrayDataset(Dataset):
    """将多个第一维长度相同的数组包装为数据集，按相同索引同时取样。"""

    def __init__(self, *arrays: np.ndarray) -> None:
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i: Index) -> Sample:
        return tuple([a[i] for a in self.arrays])
