"""图像数据增强：输入和输出均使用 H × W × C 的 NumPy 数组。"""

import numpy as np


class Transform:
    """图像变换接口；实例可像函数一样调用。"""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """以概率 p 沿宽度轴水平翻转图像，保持形状不变。

        保留课程给定的随机采样方式，以便固定随机种子后复现测试结果。
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        if flip_img:
            # ::-1 表示倒序切片；第 1 轴是宽度，第 2 轴的通道顺序不变。
            return img[:, ::-1, :]
        return img
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding: int = 3) -> None:
        self.padding = padding

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """四周补零后随机裁剪回原尺寸，通道维不补零。

        使用课程给定的偏移采样；shift_x、shift_y 分别对应行和列偏移。
        """
        shift_x, shift_y = np.random.randint(
            low=-self.padding, high=self.padding + 1, size=2
        )
        ### BEGIN YOUR SOLUTION
        p = self.padding
        height, width, _ = img.shape
        padded = np.pad(img, ((p, p), (p, p), (0, 0)), mode="constant")

        # 原图在补零数组中从 (p, p) 开始；偏移后起点在 [0, 2*p] 内。
        start_x = p + shift_x
        start_y = p + shift_y
        return padded[start_x : start_x + height, start_y : start_y + width, :]
        ### END YOUR SOLUTION
