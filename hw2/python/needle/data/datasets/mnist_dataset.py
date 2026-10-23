from collections.abc import Sequence
from ..data_basic import Dataset, Index, Transform
import numpy as np
import gzip
import struct


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Sequence[Transform] | None = None,
    ) -> None:
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)
        self.image_filename = image_filename
        self.label_filename = label_filename

        with gzip.open(image_filename, "rb") as f:
            raw = f.read()
            # > 表示大端字节序。
            # I: 4 字节无符号整数
            # 从字节序列中解析四个大端序整数
            magic, num_images, rows, cols = struct.unpack(">IIII", raw[:16])
            assert magic == 2051  # 图像文件

            pixels = np.frombuffer(raw, dtype=np.uint8, offset=16)
            X = pixels.reshape(num_images, rows, cols, 1)
            X = X.astype(np.float32) / 255.0
            self.imgs = X

        with gzip.open(label_filename, "rb") as f:
            raw = f.read()
            magic, num_labels = struct.unpack(">II", raw[:8])
            assert magic == 2049
            assert num_labels == num_images
            labels = np.frombuffer(raw, dtype=np.uint8, offset=8)
            self.labels = labels
        ### END YOUR SOLUTION

    def __getitem__(self, index: Index) -> tuple[np.ndarray, np.uint8 | np.ndarray]:
        """按整数、切片或一维索引数组取出图像与标签。

        单个索引：返回 (H, W, C) 图像数组和 np.uint8 标签标量。
        批量索引：返回 (B, H, W, C) 图像数组和 (B,) 标签数组。
        Dataset 保留 NumPy 数据类型，交给 DataLoader 再转换为 Tensor。
        """
        ### BEGIN YOUR SOLUTION
        imgs = self.imgs[index]
        labels = self.labels[index]

        if self.transforms:
            if imgs.ndim == 3:
                # 整数索引得到单张图像，变换接收 H × W × C 的数组。
                imgs = self.apply_transforms(imgs.copy())
            elif imgs.shape[0] > 0:
                # 批量取样时逐张独立采样随机变换，再沿新的批次轴合并。
                # copy 避免可能的原地变换修改数据集中保存的原始图像。
                imgs = np.stack(
                    [self.apply_transforms(img.copy()) for img in imgs], axis=0
                )
            # 空批次直接保留原形状，避免 np.stack([]) 报错。

        return imgs, labels
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.imgs.shape[0]
        ### END YOUR SOLUTION
