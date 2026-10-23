# HW2：神经网络库与 MLP ResNet

基于 CMU 10-714 HW2 课程骨架，在 HW1 的自动微分系统上实现网络层、优化器、数据加载和 MNIST 残差网络训练。

- [原始作业要求](hw2.ipynb)
- [中文作业要求](hw2-cn.ipynb)
- [训练入口](apps/mlp_resnet.py)
- [参数搜索脚本](apps/search_mnist.py)

## 环境与测试

在 `hw2` 目录下使用 Python 3.11 和 uv：

```bash
uv sync --dev
uv run pytest -q
uv run ruff check python apps
```

只运行部分测试，`-s` 可以显示代码中的打印输出：

```bash
uv run pytest -v -s -k test_mlp_train_mnist_1
uv run pytest -q -k test_nn_batchnorm
uv run pytest -q -k test_optim_adam
uv run pytest -q -k test_dataloader
```

当前完整课程测试为 **93 项通过**。搜索脚本另做过小样本端到端运行检查。

直接运行默认的 MNIST 训练和评估：

```bash
PYTHONPATH=python uv run python -u -c \
  'from apps.mlp_resnet import train_mnist; print(train_mnist(data_dir="data"))'
```

返回值顺序为：训练错误率、训练平均损失、测试错误率、测试平均损失。

## 一次训练的执行过程

```python
for X, y in dataloader:
    opt.reset_grad()
    logits = model(X)
    loss = loss_fn(logits, y)
    loss.backward()
    opt.step()
```

1. DataLoader 根据本批索引取得 NumPy 数据，再包装成 Tensor。
2. `model(X)` 经 `Module.__call__()` 转发到各层的 `forward()`。
3. 前向中的 Tensor 算子创建中间节点，并保存 `op` 和 `inputs`；损失计算继续扩展计算图。
4. `loss.backward()` 从标量损失的梯度种子 1 开始，按逆拓扑顺序传播并累加梯度，写入节点的 `.grad`。
5. `opt.step()` 读取参数梯度，更新参数的底层数值，以及动量等优化器状态。

反向传播负责求梯度，优化器负责决定如何更新参数。每批重新构建中间计算图，而 Module 和 Parameter 对象持续存在；下一批使用更新后的参数。批次 loss 是该批样本的平均损失，统计整轮损失时需按批次样本数加权。

### 参数、数据与计算图

`Parameter` 继承 Tensor，标记需要交给优化器的可学习参数。`model.parameters()` 沿模块属性递归收集 Parameter；`backward()` 则沿 Tensor 的 `inputs` 遍历计算图，两者通过同一批参数对象连接。

`requires_grad` 决定是否需要梯度，Parameter 决定是否被模块的参数收集器找到。Linear、LayerNorm、BatchNorm 都各自持有独立的权重和偏置，即使属性名相同也不共享参数。

- `tensor.detach()`：返回共享底层数据但不连接原图的新 Tensor，不会修改原 Tensor，也不是深拷贝。
- 读取 `param.data`：调用 Tensor 的属性 getter，返回断开计算图的 Tensor。
- 赋值 `param.data = value`：调用 setter，替换参数缓存的数据，保留原 Parameter 对象。

优化器的动量、Adam 的历史统计量和 BatchNorm 的运行统计量需要跨批次保存，应脱离历史计算图。用于当前批次归一化的均值、方差则必须保留可微关系。

CPython 主要通过引用计数回收对象，并额外处理循环引用。将 loss Tensor 放入长期列表会保留它连接的图；只记录指标时保存 `float(loss.numpy())` 即可。对象释放后，内存分配器可能保留内存供后续复用，进程内存不会总是立即下降。

## 模块与数学要点

### 网络模块

| 模块 | 职责 |
| --- | --- |
| Linear | `X @ weight + bias`；偏置需要显式广播 |
| Flatten | `(B, H, W, C)` 展平为 `(B, H*W*C)` |
| ReLU | 逐元素取输入与 0 的较大值 |
| Sequential | 将上一模块的输出传给下一模块 |
| Residual | 计算 `fn(x) + x`，两条路径在反向时贡献梯度 |
| LayerNorm | 每个样本沿特征轴计算均值和方差 |
| BatchNorm | 每个特征沿批次轴计算均值和方差 |
| Dropout | 训练时随机屏蔽激活并缩放保留值 |

重复构建独立模块时使用列表推导式；`[block] * n` 会重复引用同一个实例并共享参数。`norm(dim)` 中的 `norm` 是传入的构造器（通常是层的类），调用后才得到层实例。

LayerNorm 和 BatchNorm 使用有偏方差，即除以参与统计的元素数量。归一化分母为 `sqrt(var + eps)`，之后应用各层自己的可学习缩放与偏置；它们分别初始化为 1 和 0。

### 训练与评估模式

| 模块 | train 模式 | eval 模式 |
| --- | --- | --- |
| Dropout | 以概率 p 置零，保留值除以 1-p | 原样返回输入 |
| BatchNorm | 使用当前批次统计量并更新运行统计量 | 使用运行统计量且不再更新 |

当前其他模块的前向行为不随模式变化。`Module.train()` / `eval()` 会递归切换子模块的标志；`eval()` 本身不关闭自动求导。`epoch(..., opt=None)` 不调用 backward 或参数更新，当前实现仍可能构建计算图。

### LogSumExp、LogSoftmax 与损失

对于单个样本的类别得分 z：

```text
LogSumExp(z) = log(sum(exp(z)))                 一个标量
Softmax(z)   = exp(z) / sum(exp(z))             概率向量
LogSoftmax(z)= z - LogSumExp(z)                 对数概率向量
```

LogSumExp 的稳定计算是 `m + log(sum(exp(z-m)))`，其中 `m=max(z)`；这与原表达式等价。对输入求导得到 softmax 概率。

若上游梯度为 g，LogSoftmax 的输入梯度为 `g - softmax(z) * sum(g)`。这里的求和来自所有类别共享同一个 LogSumExp 值，反向时需要合并这些路径的贡献。

Softmax 交叉熵对每个样本计算 `LogSumExp(z) - z[真实类别]`，再对批次取平均。Logits 的 argmax 与 softmax 概率的 argmax 相同，统计预测类别无需先计算 softmax。

### 优化器

- SGD：使用梯度和可选的 L2 权重衰减项更新参数；本题动量为 `u = beta*u + (1-beta)*g`。
- Adam：保存梯度和梯度平方的移动平均，分别除以 `1-beta1**t`、`1-beta2**t` 做偏差修正。
- Adam 的 t 是 `step()` 调用次数，每个 minibatch 增加一次；修正后的统计量只用于本次更新，不写回原历史状态。
- Adam 的分母为 `sqrt(v_hat) + eps`，eps 在平方根外。优化器不应原地修改 `.grad`。

## 数据处理

Dataset 返回 NumPy 数据元组；DataLoader 将元组中的每一项转换为 Tensor，因此也支持只含一个数组的 NDArrayDataset。

- 单个 MNIST 索引返回 `(28, 28, 1)` 图像和标签标量。
- 切片或索引数组返回 `(B, 28, 28, 1)` 图像及 `(B,)` 标签；图像变换逐张应用。
- `DataLoader.ordering` 保存各批次的样本索引数组。`__iter__()` 重置游标并按需打乱，`__next__()` 按批次取数据，耗尽后抛出 StopIteration。
- 水平翻转用 `img[:, ::-1, :]`，反转宽度轴。
- 随机裁剪先在高、宽两端各补 p 个零，再从 `(p+shift_x, p+shift_y)` 开始截取原图大小；通道维不补零。

## 超参数搜索与实验结果

搜索脚本默认从原训练集划出 5,000 个固定验证样本，以验证错误率优先、验证损失次优选择配置和轮次。最终仅对最佳检查点评估一次测试集，避免使用测试指标挑选超参数。

本次运行命令：

```bash
uv run python apps/search_mnist.py \
  --trials 12 --epochs 10 \
  --lrs 0.001 0.003 \
  --hidden-dims 100 200 \
  --drop-probs 0.0 0.1 \
  --norms batch layer
```

这是从候选组合中随机抽取 12 组，每组训练 10 轮；并非遍历整个搜索空间。默认随机种子为 42，未显式指定的权重衰减候选为 `0.0` 和 `0.001`。

本次搜索选出的最佳配置（实验运行结果，不代表全局最优）：

| 参数 | 值 |
| --- | --- |
| 优化器 | Adam |
| 学习率 | 0.001 |
| 权重衰减 | 0.0 |
| 隐藏层维度 | 200 |
| 批次大小 | 100 |
| 残差块数量 | 3 |
| Dropout 概率 | 0.1 |
| 归一化 | BatchNorm |
| 最佳检查点 | 第 6 轮 |

测试结果：**loss=0.0980，错误率=2.49%，准确率=97.51%**。

结果保存在 `runs/search-*/`：`search.json` 保存运行参数和候选组合，`history.json` 保存逐轮指标，`best.json` 保存最佳记录，`best.npz` 保存权重及 BatchNorm 运行统计量，`result.json` 保存最终测试指标。该目录已被 Git 忽略；检查点不包含优化器状态，主要用于评估恢复。

快速检查脚本是否能运行：

```bash
uv run python apps/search_mnist.py \
  --trials 1 --epochs 1 --train-size 100 --val-size 50 --test-size 50 \
  --hidden-dims 20 --num-blocks 1 --batch-sizes 25
```

## 当前后端

HW2 使用 NumPy CPU 后端。Tensor 的 device 参数预留了设备抽象，但当前不能仅传入 CUDA 设备就获得 GPU 训练能力；迁移需要数组存储、算子与设备接口共同支持 GPU。

NumPy 的许多运算在编译后的 C 代码中执行，线性代数通常通过 BLAS/LAPACK 数值库加速，具体实现可能包含 C、Fortran 或汇编。Python 在上层组织模块和计算图，主要数值计算交给底层库。

