# HW3：NDArray 数组后端

基于 CMU 10-714 HW3 课程骨架，实现数组视图操作，以及 C++ CPU 和 CUDA 后端。作业要求见 [hw3.ipynb](hw3.ipynb)。

本次作业不需要复制 HW1/HW2 的实现：课程骨架的 `autograd.py` 和包入口包含新的后端选择逻辑。实现范围主要为 `python/needle/backend_ndarray/` 和 `src/`。

## 环境初始化

在 `hw3` 目录下运行：

```bash
uv sync --dev
uv run make
```

项目使用 Python 3.11。NumPy 为运行依赖；开发环境包含 pytest、mugrade、Jupyter、Ruff、pybind11、CMake、Ninja 和 clang-format。系统还需要 C/C++ 编译器与 make。

`uv run make` 会通过当前虚拟环境的 Python 查找开发头文件和 pybind11，编译扩展到 `python/needle/backend_ndarray/`。修改 `.cc` 或 `.cu` 文件后需要重新运行构建；只修改 Python 文件时不需要重编译。

## 编辑器

建议将 `hw3` 单独作为 VS Code 工作区打开，并选择 `.venv/bin/python` 作为解释器。

本机 `.vscode/` 已配置 Python/Pylance、Ruff、Jupyter、clangd 和 CMake Tools 的推荐扩展；编辑器目录不纳入版本控制。`Ctrl+Shift+B` 可以调用后端构建任务。

仓库中的 `.clangd` 指向 `build/compile_commands.json`。先构建一次，clangd 才能获取 C++ 的 Python/pybind11 头文件路径和编译参数。如果同时安装微软 C/C++ 扩展，本机配置关闭其 IntelliSense，避免与 clangd 重复诊断。

## 检查与测试

```bash
uv run ruff check python tests
uv run pytest --collect-only -q
uv run jupyter notebook
```

课程的 `pytest.ini` 已将 `python/` 加入模块搜索路径。C++ 骨架部分包含 `assert(false)` 占位，未实现时执行对应测试可能直接中止 Python 进程。初始化阶段先验证构建、导入、已有的内存分配/填充和测试收集，之后按已实现部分筛选测试：

```bash
# 实现对应功能并重新构建后运行
uv run pytest -q -k 'test_permute and cpu'
uv run pytest -q -k 'test_compact and cpu'
uv run pytest -q -k 'test_matmul and cpu'
```

本次初始化已成功收集 **140 项测试**，CPU 扩展的分配、填充和 NumPy 回读检查通过，clangd 对 CPU 源码检查为 0 个错误。尚未执行未完成题目的完整计算测试。

完成全部部分后再运行完整测试：

```bash
uv run make
uv run pytest -q
```

## CUDA

CMake 在找到 CUDA Toolkit 时会尝试构建 GPU 后端；需要 NVIDIA GPU、可用驱动和 Toolkit 提供的 `nvcc` 编译器。仅安装支持 GPU 的 Python 包不能替代 CUDA 编译工具链。

本机 WSL 已确认可以访问 RTX 4060。项目本地 Toolkit 为 CUDA 13.0.2，支持当前 Fedora 42/GCC 15 环境，安装在被忽略的 `.tools/cuda-13.0.2/`。使用 NVIDIA 官方组件清单中的 SHA-256 校验下载，不修改 Windows/WSL 驱动或系统 Python。

新环境可以执行：

```bash
uv run python scripts/setup_cuda.py
uv run python scripts/setup_clangd.py
uv run make
uv run python scripts/check_cuda.py
```

`setup_cuda.py` 下载编译器、运行时、编译头文件、NVVM 以及编辑器兼容头依赖的 cuRAND，约 152 MiB 压缩包；已安装时重复运行不会重新下载。版本来自 [NVIDIA CUDA 13.0.2 分发清单](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.0.2.json)。

`setup_clangd.py` 从 LLVM clangd 官方发布下载 23.1.0 并校验 SHA-256，约 112 MiB 压缩包。系统 clangd 20 对 CUDA 13 的兼容性不足，本机 VS Code 已将 `clangd.path` 指向 `.tools/clangd_23.1.0/bin/clangd`；重新打开工作区或执行 `clangd: Restart language server` 后生效。

`check_cuda.py` 验证扩展导入、显存分配、数据传输和骨架已有的 Fill kernel，不调用尚待实现的计算算子。本机 RTX 4060 已通过这些检查，CUDA 源码的 clangd 解析检查为 0 个错误。

CMake 原生 CUDA 构建默认生成架构 75、86、89 的代码，覆盖课程 T4、Ampere 及本机 RTX 4060。也可指定本机架构后构建：

```bash
uv run cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=89
uv run cmake --build build
```

仅构建 CPU 时，可配置 `-DNEEDLE_ENABLE_CUDA=OFF`；重新启用使用 `ON`。如需其他 Toolkit，可通过 `-DCUDAToolkit_ROOT=/path/to/cuda` 指定；切换编译器或 Toolkit 时建议使用新的构建目录。

VS Code 使用 `build/compile_commands.json` 查找 CUDA 头文件和编译参数。若 CUDA 程序在受限沙箱内报告 GPU 不可用，先在普通 WSL 终端运行 `nvidia-smi` 和检查脚本；本次检查确认沙箱会屏蔽 `/dev/dxg`，普通 WSL 环境可正常访问。

CUDA 后端未构建时，官方测试会跳过相应的 CUDA 用例。`nd.cuda().enabled()` 只表示扩展能导入，实际 GPU 可用性以基础检查脚本为准。编译扩展、Toolkit、虚拟环境、缓存和本机编译数据库均由 `.gitignore` 排除；`uv.lock` 纳入版本控制。
