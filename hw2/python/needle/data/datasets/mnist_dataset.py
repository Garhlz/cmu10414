import gzip
import struct
from typing import List, Optional
from ..data_basic import Dataset
import numpy as np

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)
        self.image_filename = image_filename
        self.label_filename = label_filename
        with gzip.open(image_filename, 'rb') as f:
            raw = f.read()
            magic, num_images, rows, cols = struct.unpack('>IIII', raw[:16])
            assert magic == 2051
            pixels = np.frombuffer(raw, dtype=np.uint8, offset=16)
            X = pixels.reshape(num_images, rows, cols, 1)
            X = X.astype(np.float32) / 255.0
            self.imgs = X
        with gzip.open(label_filename, 'rb') as f:
            raw = f.read()
            magic, num_labels = struct.unpack('>II', raw[:8])
            assert magic == 2049
            assert num_labels == num_images
            labels = np.frombuffer(raw, dtype=np.uint8, offset=8)
            self.labels = labels
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        imgs = self.imgs[index]
        labels = self.labels[index]
        if self.transforms:
            if imgs.ndim == 3:
                imgs = self.apply_transforms(imgs.copy())
            elif imgs.shape[0] > 0:
                imgs = np.stack([self.apply_transforms(img.copy()) for img in imgs], axis=0)
        return (imgs, labels)
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.imgs.shape[0]
        ### END YOUR SOLUTION