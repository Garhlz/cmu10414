"""检查已有的 CUDA 内存与填充接口，不调用作业中的待实现算子。"""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from needle import backend_ndarray as nd


def main() -> None:
    device = nd.cuda()
    if not device.enabled():
        raise RuntimeError("CUDA 扩展无法导入，请先运行 uv run make 并检查动态库依赖")
    source = np.arange(24, dtype=np.float32).reshape(4, 6)
    copied = nd.array(source, device=device)
    np.testing.assert_array_equal(copied.numpy(), source)
    filled = nd.full((4, 6), 7.0, device=device)
    np.testing.assert_array_equal(
        filled.numpy(), np.full((4, 6), 7.0, dtype=np.float32)
    )
    print("CUDA 扩展导入：通过")
    print("GPU 显存分配、CPU/GPU 数据传输：通过")
    print("GPU Fill kernel 执行与结果回读：通过")


if __name__ == "__main__":
    main()
