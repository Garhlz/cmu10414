"""搜索 MLPResNet 超参数，显示训练进度，并保存最佳验证模型。"""

import argparse
from collections.abc import Iterator
import itertools
import json
import math
from pathlib import Path
import random
import sys
from datetime import datetime
from typing import Any, Literal, TypedDict, cast

# 用脚本位置定位项目，支持从任意工作目录执行。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import numpy as np
from tqdm.auto import tqdm

import needle as ndl
from mlp_resnet import MLPResNet, epoch


class SearchConfig(TypedDict):
    """一组候选超参数；TypedDict 仅描述字典的键与值，不改变运行时表示。"""

    optimizer: Literal["adam", "sgd"]
    lr: float
    weight_decay: float
    hidden_dim: int
    batch_size: int
    num_blocks: int
    drop_prob: float
    norm: Literal["batch", "layer"]


class EpochRecord(TypedDict):
    """一组配置在某一轮的训练与验证指标。"""

    trial: int
    epoch: int
    config: SearchConfig
    train_error: float
    train_loss: float
    val_error: float
    val_loss: float


class ProgressLoader:
    """给 DataLoader 增加批次进度条，仍由课程的 epoch 完成训练计算。"""

    def __init__(self, loader: ndl.data.DataLoader, description: str) -> None:
        self.loader = loader
        self.description = description

    def __iter__(self) -> Iterator[tuple[ndl.Tensor, ...]]:
        total = math.ceil(len(self.loader.dataset) / self.loader.batch_size)
        # 生成器在 epoch 设置随机种子之后才开始遍历，保持原来的随机行为。
        with tqdm(total=total, desc=self.description, leave=False) as bar:
            for batch in self.loader:
                yield batch
                bar.update(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--trials", type=int, default=8, help="随机抽取的组合数，0 表示全部组合"
    )
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument(
        "--train-size", type=int, help="可选：限制训练样本数，用于快速实验"
    )
    parser.add_argument("--test-size", type=int, help="可选：限制最终测试样本数")
    parser.add_argument(
        "--optimizers", nargs="+", choices=["adam", "sgd"], default=["adam"]
    )
    parser.add_argument("--lrs", nargs="+", type=float, default=[0.001, 0.003])
    parser.add_argument("--weight-decays", nargs="+", type=float, default=[0.0, 0.001])
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[100, 200])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[100])
    parser.add_argument("--num-blocks", nargs="+", type=int, default=[3])
    parser.add_argument("--drop-probs", nargs="+", type=float, default=[0.0, 0.1])
    parser.add_argument(
        "--norms", nargs="+", choices=["batch", "layer"], default=["batch"]
    )
    args = parser.parse_args()
    if args.epochs < 1 or args.trials < 0 or args.val_size < 1:
        parser.error("epochs、val-size 必须为正数，trials 必须非负")
    if any(v < 1 for v in args.batch_sizes) or any(v < 2 for v in args.hidden_dims):
        parser.error("batch-size 必须为正数，hidden-dim 至少为 2")
    if any(v < 0 for v in args.num_blocks) or any(
        not 0 <= p < 1 for p in args.drop_probs
    ):
        parser.error("num-blocks 必须非负，drop-prob 必须在 [0, 1) 内")
    if any(v <= 0 for v in args.lrs) or any(v < 0 for v in args.weight_decays):
        parser.error("学习率必须为正数，weight-decay 必须非负")
    if any(v is not None and v < 1 for v in (args.train_size, args.test_size)):
        parser.error("train-size 和 test-size 必须为正数")
    return args


def make_model(config: SearchConfig) -> ndl.nn.Module:
    norm = ndl.nn.BatchNorm1d if config["norm"] == "batch" else ndl.nn.LayerNorm1d
    return MLPResNet(
        dim=784,
        hidden_dim=config["hidden_dim"],
        num_blocks=config["num_blocks"],
        norm=norm,
        drop_prob=config["drop_prob"],
    )


def save_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def model_state(model: ndl.nn.Module) -> dict[str, np.ndarray]:
    """保存参数以及 BatchNorm 的运行统计量；后者也影响评估结果。"""
    state = {f"param_{i}": p.numpy().copy() for i, p in enumerate(model.parameters())}
    for i, module in enumerate(model._children()):
        for name in ("running_mean", "running_var"):
            if hasattr(module, name):
                state[f"module_{i}_{name}"] = getattr(module, name).numpy().copy()
    return state


def load_state(model: ndl.nn.Module, state: dict[str, np.ndarray]) -> None:
    """按固定模块遍历顺序恢复参数和运行统计量，不恢复优化器状态。"""
    for i, param in enumerate(model.parameters()):
        param.data = ndl.Tensor(state[f"param_{i}"], requires_grad=False)
    for i, module in enumerate(model._children()):
        for name in ("running_mean", "running_var"):
            if hasattr(module, name):
                setattr(
                    module,
                    name,
                    ndl.Tensor(state[f"module_{i}_{name}"], requires_grad=False),
                )


def main() -> None:
    args = parse_args()
    output = args.output_dir or ROOT / "runs" / datetime.now().strftime(
        "search-%Y%m%d-%H%M%S-%f"
    )
    output.mkdir(parents=True, exist_ok=False)

    dataset = ndl.data.MNISTDataset(
        str(args.data_dir / "train-images-idx3-ubyte.gz"),
        str(args.data_dir / "train-labels-idx1-ubyte.gz"),
    )
    if args.val_size >= len(dataset):
        raise ValueError("val-size 必须小于训练集样本数")
    # 所有试验共用同一个划分，不用测试集挑选超参数。
    indices = np.random.default_rng(args.seed).permutation(len(dataset))
    val_idx, train_idx = indices[: args.val_size], indices[args.val_size :]
    if args.train_size is not None:
        train_idx = train_idx[: args.train_size]
    train = ndl.data.NDArrayDataset(*dataset[train_idx])
    val = ndl.data.NDArrayDataset(*dataset[val_idx])

    names = [
        "optimizer",
        "lr",
        "weight_decay",
        "hidden_dim",
        "batch_size",
        "num_blocks",
        "drop_prob",
        "norm",
    ]
    choices = [
        args.optimizers,
        args.lrs,
        args.weight_decays,
        args.hidden_dims,
        args.batch_sizes,
        args.num_blocks,
        args.drop_probs,
        args.norms,
    ]
    # 组合列表的键来自上面的固定 names，cast 让检查器识别其字典结构。
    configs = cast(
        list[SearchConfig],
        [dict(zip(names, values)) for values in itertools.product(*choices)],
    )
    random.Random(args.seed).shuffle(configs)
    if args.trials:
        configs = configs[: args.trials]
    save_json(
        output / "search.json",
        {
            "args": {
                k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
            },
            "configs": configs,
            "train_samples": len(train),
            "val_samples": len(val),
        },
    )
    tqdm.write(f"搜索 {len(configs)} 组参数；结果目录：{output}")
    history: list[EpochRecord] = []
    best: EpochRecord | None = None
    best_state: dict[str, np.ndarray] | None = None
    for trial, config in enumerate(tqdm(configs, desc="参数搜索"), start=1):
        np.random.seed(args.seed)
        model = make_model(config)
        optimizer = ndl.optim.Adam if config["optimizer"] == "adam" else ndl.optim.SGD
        opt = optimizer(
            model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
        )
        train_loader = ndl.data.DataLoader(
            train, batch_size=config["batch_size"], shuffle=True
        )
        val_loader = ndl.data.DataLoader(val, batch_size=config["batch_size"])
        tqdm.write(f"试验 {trial}: {config}")
        for step in tqdm(range(1, args.epochs + 1), desc="Epoch", leave=False):
            train_error, train_loss = epoch(
                ProgressLoader(train_loader, "训练批次"), model, opt
            )
            val_error, val_loss = epoch(ProgressLoader(val_loader, "验证批次"), model)
            if not np.isfinite([train_loss, val_loss, train_error, val_error]).all():
                tqdm.write("检测到非有限指标，停止本组参数的训练。")
                break
            record: EpochRecord = dict(
                trial=trial,
                epoch=step,
                config=config,
                train_error=train_error,
                train_loss=train_loss,
                val_error=val_error,
                val_loss=val_loss,
            )
            history.append(record)
            tqdm.write(
                f"[{trial}/{len(configs)}] epoch {step}/{args.epochs} | "
                f"train loss={train_loss:.4f}, error={train_error:.2%} | "
                f"val loss={val_loss:.4f}, error={val_error:.2%}"
            )
            # 验证错误率优先；相同时选择验证损失更小的配置与轮次。
            if best is None or (val_error, val_loss) < (
                best["val_error"],
                best["val_loss"],
            ):
                best = record
                best_state = model_state(model)
                np.savez(output / "best.npz", **best_state)
                save_json(output / "best.json", best)
            save_json(output / "history.json", history)

    if best is None:
        raise RuntimeError("没有得到有效模型，请检查学习率与训练实现")
    assert best_state is not None
    model = make_model(best["config"])
    load_state(model, best_state)
    test = ndl.data.MNISTDataset(
        str(args.data_dir / "t10k-images-idx3-ubyte.gz"),
        str(args.data_dir / "t10k-labels-idx1-ubyte.gz"),
    )
    if args.test_size is not None:
        test = ndl.data.NDArrayDataset(*test[: args.test_size])
    test_loader = ndl.data.DataLoader(test, batch_size=best["config"]["batch_size"])
    test_error, test_loss = epoch(ProgressLoader(test_loader, "最佳模型测试"), model)
    result = dict(best=best, test_error=test_error, test_loss=test_loss)
    save_json(output / "result.json", result)
    tqdm.write(f"最佳参数：{best['config']}，epoch={best['epoch']}")
    tqdm.write(
        f"测试 loss={test_loss:.4f}, error={test_error:.2%}, accuracy={1 - test_error:.2%}"
    )


if __name__ == "__main__":
    main()
