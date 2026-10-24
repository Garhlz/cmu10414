"""从 NVIDIA 官方分发清单安装项目本地 CUDA 编译组件，不安装驱动。"""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request

VERSION = "13.0.2"
BASE = "https://developer.download.nvidia.com/compute/cuda/redist/"
ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / ".tools" / f"cuda-{VERSION}"
# Clang 的 CUDA 兼容头还会引用 cuRAND 头文件，供编辑器解析使用。
COMPONENTS = (
    "cuda_nvcc",
    "cuda_crt",
    "cuda_cudart",
    "cuda_cccl",
    "libnvvm",
    "libcurand",
)


def patch_glibc_compatibility() -> None:
    """CUDA 13.0 的 rsqrt 声明需要与 glibc 2.41+ 的 noexcept 保持一致。

    仅修改项目本地头文件，条件宏保留旧 glibc / 其他平台的原始声明。
    https://forums.developer.nvidia.com/t/362940
    """
    header = DEST / "include" / "crt" / "math_functions.h"
    content = header.read_text()
    for signature in ("rsqrt(double x)", "rsqrtf(float x)"):
        old = next(
            (line for line in content.splitlines()
             if line.startswith("extern ") and line.endswith(signature + ";")),
            None,
        )
        marker = "// CMU10414 glibc compatibility: " + signature
        if old is not None and marker not in content:
            new = (marker + "\n#if defined(__GLIBC__)\n"
                   "#if __GLIBC_PREREQ(2, 41)\n"
                   + old[:-1] + " noexcept(true);\n#else\n" + old
                   + "\n#endif\n#else\n" + old + "\n#endif")
            content = content.replace(old, new, 1)
    header.write_text(content)


def main() -> None:
    marker = DEST / "installed-components.json"
    records = json.loads(marker.read_text()) if marker.exists() else {}
    if all(component in records for component in COMPONENTS):
        patch_glibc_compatibility()
        if (DEST / "lib").is_dir() and not (DEST / "lib64").exists():
            (DEST / "lib64").symlink_to("lib", target_is_directory=True)
        subprocess.run([str(DEST / "bin" / "nvcc"), "--version"], check=True)
        print(f"已安装：{DEST}")
        return
    print(f"读取 NVIDIA CUDA {VERSION} 官方组件清单", flush=True)
    with urllib.request.urlopen(
        BASE + f"redistrib_{VERSION}.json", timeout=60
    ) as response:
        manifest = json.load(response)
    downloads = ROOT / ".tools" / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    DEST.mkdir(parents=True, exist_ok=True)
    for component in COMPONENTS:
        if component in records:
            continue
        info = manifest[component]["linux-x86_64"]
        archive = downloads / Path(info["relative_path"]).name
        expected = info["sha256"]
        print(f"安装 {component}（{int(info['size']) / 1024**2:.1f} MiB）", flush=True)
        if (
            not archive.exists()
            or hashlib.sha256(archive.read_bytes()).hexdigest() != expected
        ):
            # curl 继承代理配置，限制连接等待并重试临时网络错误。
            subprocess.run(
                [
                    "curl",
                    "--fail",
                    "--location",
                    "--retry",
                    "3",
                    "--connect-timeout",
                    "30",
                    "--max-time",
                    "600",
                    "--output",
                    str(archive),
                    BASE + info["relative_path"],
                ],
                check=True,
            )
        if hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"SHA-256 校验失败：{archive}")
        # 独立解压后再合并，保留各组件附带的许可证与相对符号链接。
        with tempfile.TemporaryDirectory(prefix="cuda-component-") as directory:
            with tarfile.open(archive) as package:
                package.extractall(directory, filter="data")
            roots = list(Path(directory).iterdir())
            if len(roots) != 1 or not roots[0].is_dir():
                raise RuntimeError(f"组件目录结构不符合预期：{component}")
            shutil.copytree(roots[0], DEST, dirs_exist_ok=True, symlinks=True)
        records[component] = info
    # NVIDIA 分组件包使用 lib/，而 nvcc 默认链接目录为 lib64/。
    if (DEST / "lib").is_dir() and not (DEST / "lib64").exists():
        (DEST / "lib64").symlink_to("lib", target_is_directory=True)
    marker.write_text(json.dumps(records, indent=2) + "\n")
    patch_glibc_compatibility()
    subprocess.run([str(DEST / "bin" / "nvcc"), "--version"], check=True)
    print(f"安装完成：{DEST}\n接下来运行 uv run make", flush=True)


if __name__ == "__main__":
    main()
