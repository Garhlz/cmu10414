"""安装支持 CUDA 13 的项目本地 clangd，避免依赖系统编辑器工具版本。"""

import hashlib
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
VERSION = "23.1.0"
SHA256 = "e53b1a96196095faedb7642cf64964f7fb9ad4a0c1f00dd2c172a3d9dcbafdfd"


def main() -> None:
    tools = ROOT / ".tools"
    executable = tools / f"clangd_{VERSION}" / "bin" / "clangd"
    if not executable.exists():
        archive = tools / "downloads" / f"clangd-linux-{VERSION}.zip"
        archive.parent.mkdir(parents=True, exist_ok=True)
        if (
            not archive.exists()
            or hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256
        ):
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
                    f"https://github.com/clangd/clangd/releases/download/{VERSION}/clangd-linux-{VERSION}.zip",
                ],
                check=True,
            )
        if hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
            raise RuntimeError("clangd 下载文件 SHA-256 校验失败")
        with zipfile.ZipFile(archive) as package:
            package.extractall(tools)
        executable.chmod(0o755)
    subprocess.run([str(executable), "--version"], check=True)
    print(f"VS Code clangd.path：{executable}")


if __name__ == "__main__":
    main()
