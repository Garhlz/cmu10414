# WSL 开发环境

HW0–HW3 分别使用自己的 `.venv`，已按原有 `uv.lock` 安装 Python 3.11.17 及开发依赖。不要在仓库根目录共用一个 Python 环境：各作业都包含不同版本的 `needle`。

## VS Code

从本目录的 WSL 终端打开：

```nu
code cmu10414.code-workspace
```

确认窗口左下角为 WSL。工作区中每个作业都配置了独立的 `.venv/bin/python`、pytest、模块搜索路径、当前文件调试和 uv 任务。Notebook 在右上角选择所在 HW 的 `.venv/bin/python` 内核。若此前手动选过别的解释器，执行 `Python: Select Interpreter` 重新选择该目录的 `.venv/bin/python`。

HW3 使用项目本地 clangd 23.1.0 和 `build/compile_commands.json` 提供 C++/CUDA 补全。Ctrl+Shift+B 可构建 HW0/HW3；其他命令在“终端 → 运行任务”中选择。工作区默认使用 `/usr/bin/nu`，HW3 集成终端自动加入本地 CUDA 的 PATH。

`.vscode/` 和 `.env` 是本机配置，按各作业的 `.gitignore` 排除；多目录工作区文件可纳入版本控制。

## 日常命令

```nu
cd hw3                  # 可替换为 hw0、hw1、hw2
uv sync --locked        # 恢复依赖，默认包含 dev 组
uv run python --version
uv run make             # 仅 HW0/HW3 需要编译
uv run pytest --collect-only -q
uv run python scripts/check_cuda.py  # 仅 HW3
```

HW3 已完成并通过全部 140 项 CPU/CUDA 本地测试；正式课程提交仍需按 notebook 运行 mugrade 并上传源码归档。

## CUDA 与普通终端

Toolkit 安装在 `hw3/.tools/cuda-13.0.2`，由仓库现有安装脚本下载并校验 SHA-256；不安装 Linux NVIDIA 驱动。系统使用 Windows 提供的 WSL GPU 驱动。

本机 `~/.config/nushell/env.nu` 按基础 PATH → CUDA → `mise activate nu` 的顺序设置环境，每次启动重新生成 `mise.nu`；`config.nu` 加载该模块，不再重设 PATH，也不再添加 CUDA 恢复钩子。新开 Nushell 终端即可使用。迁移到其他机器时，在 `env.nu` 中设置实际的 CUDA 路径，或在生成 mise 模块之前加载 `scripts/wsl-env.nu`。当前已打开的终端建议重新启动 Nushell，以清除旧会话中的恢复钩子；仅临时启用 CUDA 可执行：

```nu
source scripts/wsl-env.nu   # 在仓库根目录执行
nvcc --version
nvidia-smi
```

迁移后重新安装本地工具：

```nu
cd hw3
uv run python scripts/setup_cuda.py
uv run python scripts/setup_clangd.py
uv run make
```

安装脚本包含针对 glibc 2.41+ 的 CUDA 13.0 头文件兼容修复，详情见 HW3 README。扩展通过构建时 RPATH 查找本地 CUDA 运行库，无需全局设置 LD_LIBRARY_PATH。受限沙箱可能屏蔽 GPU；GPU 检查请在普通 WSL 终端运行。

WSL 端已安装并核实 Python、Pylance、debugpy、Jupyter、Ruff、clangd 和 CMake Tools 扩展。

## 本次验证结果

- HW0：C++ 扩展编译成功，`test_add` 通过。
- HW1：29 个测试可收集，12 个 forward 测试通过。
- HW2：93 个测试可收集，7 个 power-scalar / logsumexp forward 测试通过。
- HW3：CPU/CUDA 扩展均编译成功，140 项测试全部通过、无跳过，Ruff 检查通过；RTX 4060 上 CUDA 导入、内存传输与 Fill kernel 检查通过；clangd CUDA 源码检查退出码为 0。
- 新 Nushell 终端可直接找到项目本地 `nvcc`。
