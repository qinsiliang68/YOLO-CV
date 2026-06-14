# -*- coding: utf-8 -*-
"""Stage-1 YOLO11-cls 二分类训练调度脚本。

给新手看的总说明：
1. 这个文件不改 YOLO 模型结构，也不改 Ultralytics 官方源码。
2. 它做的事情是：读取我们已经抽好的 CSV manifest，把图片临时整理成 YOLO 分类训练需要的目录。
3. 然后它调用本地 YOLOv11 源码里的 `YOLO(...).train(...)` 开始训练。
4. `smoke` 模式只抽很少的图片，用来确认流程能跑通。
5. `full` 模式才会用完整训练集，默认每个模型训练 200 epoch。
6. 它会写临时数据目录和训练输出目录；不会修改原始 final_sewerml_dataset 图片和 manifest。
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable


# 固定随机种子：所有 smoke 抽样都用它，保证每次抽到同一批小样本。
SEED = 20260606

# 二分类的两个类别名。目录名也会用这两个语义：
#   no_target      = 没有目标缺陷
#   target_defect  = 有 PF/DE/FS/RB/AF/OB 任意一种目标缺陷
CLASS_NAMES = ("no_target", "target_defect")

# 五个 YOLO11 classification 模型规模：
# n 最小最快，x 最大最慢。
MODEL_KEYS = ("n", "s", "m", "l", "x")

# 每个模型代号对应的本地预训练权重文件名。
# 脚本会在仓库根目录、YOLOv11 目录、YOLOv11/weights 目录里查找这些文件。
MODEL_WEIGHTS = {
    "n": "yolo11n-cls.pt",
    "s": "yolo11s-cls.pt",
    "m": "yolo11m-cls.pt",
    "l": "yolo11l-cls.pt",
    "x": "yolo11x-cls.pt",
}

# =========================
# 路径和材料文件名集中配置
# =========================
# 这些变量是给人和机器改的：换训练机、换硬盘、换归档文件名时，优先改这里。
# 命令行参数和环境变量仍然可用；它们的优先级高于这里的默认值。

# YOLOv11 官方源码目录。默认在仓库根目录下。
DEFAULT_YOLO_ROOT = Path("YOLOv11")

# 最终数据集目录。这里不放临时训练目录，只放已经抽样完成的数据集和 manifests。
DEFAULT_DATASET_ROOT = Path("data") / "final_sewerml_dataset"

# YOLO-cls 临时目录。脚本会在这里组装 train/no_target、train/target_defect 等目录。
DEFAULT_WORK_ROOT = Path("data") / "stage1_cls_workdir"

# 每个训练 run 的输出根目录。所有 best.pt、last.pt、日志、meta、清单都在这里下面。
# DEFAULT_RUNS_SUBDIR 是相对 YOLO 根目录的子路径；DEFAULT_RUNS_ROOT 是仓库默认完整相对路径。
DEFAULT_RUNS_SUBDIR = Path("runs") / "stage1_cls_sweep"
DEFAULT_RUNS_ROOT = DEFAULT_YOLO_ROOT / DEFAULT_RUNS_SUBDIR

# summary CSV 文件名。smoke 和 full 分开，避免小测试记录混进正式训练记录。
SUMMARY_FILENAMES = {
    "smoke": "smoke_summary.csv",
    "full": "summary.csv",
}

# run 目录内的材料文件名。
TRAIN_LOG_FILENAME = "train_log.txt"
RUN_META_FILENAME = "run_meta.json"
ARTIFACT_MANIFEST_CSV_FILENAME = "artifact_manifest.csv"
ARTIFACT_MANIFEST_JSON_FILENAME = "artifact_manifest.json"
RESULTS_CSV_FILENAME = "results.csv"
ARGS_YAML_FILENAME = "args.yaml"

# 训练时日志先写到 runs_root/_logs，训练结束后再移动进最终 run 目录。
# 这样即使 YOLO 还没创建 run_dir，日志也不会丢。
TEMP_LOG_DIRNAME = "_logs"
TEMP_LOG_SUFFIX = ".train_log.txt"

# YOLO 默认权重目录名和关键权重文件名。
WEIGHTS_DIRNAME = "weights"
BEST_WEIGHT_FILENAME = "best.pt"
LAST_WEIGHT_FILENAME = "last.pt"

# 训练结束后必须检查的最小正交材料。
# 这些文件要么重新获取成本很高，要么是后续推断训练过程和结果的基础。
REQUIRED_ARTIFACTS = (
    f"{WEIGHTS_DIRNAME}/{BEST_WEIGHT_FILENAME}",
    f"{WEIGHTS_DIRNAME}/{LAST_WEIGHT_FILENAME}",
    RESULTS_CSV_FILENAME,
    ARGS_YAML_FILENAME,
    TRAIN_LOG_FILENAME,
    RUN_META_FILENAME,
)

# 这些 manifest 是数据侧的复现材料。训练 run_meta 会记录它们的 hash，
# 以后别人拿到同一批 CSV 和原始图片，就能知道本次训练用的是哪批图。
TRAIN_MANIFEST_FILES = (
    "train_manifest.csv",
    "normal_train_manifest.csv",
    "val_model_manifest.csv",
    "normal_val_model_manifest.csv",
)

# smoke test 默认图片数量。这个数量故意很小：
# 目的不是训练出好模型，只是验证流程能不能正常生成 best.pt/results.csv。
SMOKE_TRAIN_PER_CLASS = 96
SMOKE_VAL_PER_CLASS = 48

# smoke 默认 2 轮；full 默认 200 轮。
# 你正式训练时一般用 full，脚本默认会用 200 epoch。
SMOKE_EPOCHS = 2
FULL_EPOCHS = 200

# YOLO 分类模型常用 224，显存压力也小。
DEFAULT_IMGSZ = 224


@dataclass(frozen=True)
class Paths:
    """把本脚本会用到的重要目录集中保存。

    `frozen=True` 表示对象创建后不能再改字段，避免训练过程中路径被意外改掉。
    """

    # 仓库根目录，例如 C:\GitHub\YOLO-CV。
    repo_root: Path
    # 官方 YOLOv11 源码目录，本脚本会从这里 import ultralytics。
    yolo_root: Path
    # 已经抽样完成、准备作为最终数据集存档的目录。
    dataset_root: Path
    # CSV 复现材料所在目录，里面记录每张图来自哪里、被分到哪个集合。
    manifest_dir: Path
    # 临时分类数据目录。YOLO-cls 需要 train/类别名、val/类别名 这种目录结构。
    work_root: Path
    # 训练输出目录，best.pt、last.pt、results.csv 会落在这里。
    runs_root: Path


@dataclass(frozen=True)
class TrainConfig:
    """一次训练任务的配置。

    这里不放路径，只放“训练怎么跑”的参数；路径统一放在 `Paths` 里。
    """

    # smoke = 小样本快速试跑；full = 正式全量训练。
    mode: str
    # 要训练哪些模型，例如 ("n",) 或 ("n", "s", "m", "l", "x")。
    models: tuple[str, ...]
    # 随机种子，用来控制 smoke 抽样和 YOLO 训练随机性。
    seed: int
    # epoch 数，简单理解就是完整看多少遍训练集。
    epochs: int
    # 输入图片尺寸，分类模型默认用 224。
    imgsz: int
    # batch size，每一步喂给 GPU 的图片数量。
    batch: int
    # DataLoader 工作线程。Windows 上先用 0 最稳，避免多进程额外问题。
    workers: int
    # 每隔多少个 epoch 保存一次 checkpoint。1 表示每轮都留，-1 表示只留 best/last。
    save_period: int
    # GPU/CPU 选择。一般 "0" 表示第 0 张显卡，"cpu" 表示不用显卡。
    device: str
    # True 表示训练前重建临时数据目录；False 表示复用已有临时目录。
    rebuild_data: bool
    # smoke 模式下每个二分类类别抽多少训练图；full 模式通常是 None，表示全用。
    train_per_class: int | None
    # smoke 模式下每个二分类类别抽多少验证图；full 模式通常是 None，表示全用。
    val_per_class: int | None
    # True 时只准备数据和写 summary，不真正训练，适合检查路径。
    dry_run: bool
    # 是否允许 Ultralytics 复用同名 run 目录；默认不复用，避免覆盖结果。
    exist_ok: bool


@dataclass(frozen=True)
class DatasetCounts:
    """记录实际送进 YOLO 的图片数量。

    这些数量会写入 summary CSV，方便训练结束后追溯这一轮到底用了多少图。
    """

    train_no_target: int
    train_target_defect: int
    val_no_target: int
    val_target_defect: int


def repo_root_from_script() -> Path:
    """根据脚本位置反推仓库根目录。

    当前文件在 `scripts/` 下，所以 `parents[1]` 就是 `C:\GitHub\YOLO-CV`。
    """

    return Path(__file__).resolve().parents[1]


def build_paths(args: argparse.Namespace) -> Paths:
    """把命令行参数转换成绝对路径。

    规则是：用户传了参数就用用户指定的目录；没传就用项目约定的默认目录。
    """

    repo_root = repo_root_from_script()

    # `resolve()` 会把相对路径转成绝对路径，后面打印和排错更清楚。
    yolo_root = Path(args.yolo_root).resolve() if args.yolo_root else repo_root / DEFAULT_YOLO_ROOT
    dataset_root = Path(args.dataset_root).resolve() if args.dataset_root else repo_root / DEFAULT_DATASET_ROOT
    manifest_dir = dataset_root / "manifests"
    work_root = Path(args.work_root).resolve() if args.work_root else repo_root / DEFAULT_WORK_ROOT
    if args.runs_root:
        runs_root = Path(args.runs_root).resolve()
    elif args.yolo_root:
        runs_root = yolo_root / DEFAULT_RUNS_SUBDIR
    else:
        runs_root = repo_root / DEFAULT_RUNS_ROOT
    return Paths(
        repo_root=repo_root,
        yolo_root=yolo_root,
        dataset_root=dataset_root,
        manifest_dir=manifest_dir,
        work_root=work_root,
        runs_root=runs_root,
    )


def parse_models(value: str | None, mode: str) -> tuple[str, ...]:
    """解析要训练的模型列表。

    优先级：命令行 `--models` > 环境变量 `STAGE1_MODELS` > 默认值。
    smoke 默认只跑 n，因为最快；full 默认五个模型都跑。
    """

    raw = value or os.environ.get("STAGE1_MODELS")
    if not raw:
        return ("n",) if mode == "smoke" else MODEL_KEYS

    # 允许写成 `n,s,m`，这里会拆开、去空格、转小写。
    models = tuple(part.strip().lower() for part in raw.split(",") if part.strip())
    unknown = [m for m in models if m not in MODEL_KEYS]
    if unknown:
        raise ValueError(f"Unknown model key(s): {unknown}. Valid keys: {MODEL_KEYS}")
    return models


def read_manifest(path: Path) -> list[dict[str, str]]:
    """读取一个 manifest CSV。

    返回值是“每行一个字典”的列表，例如 row["Filename"] 就是这一行的图片文件名。
    `utf-8-sig` 可以兼容带 BOM 的 CSV，Windows/Excel 导出的文件也更稳。
    """

    if not path.exists():
        raise FileNotFoundError(f"Missing manifest: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def canonical_image_path(row: dict[str, str], dataset_root: Path) -> Path:
    """从 manifest 的一行里找到真实图片路径。

    优先用 `canonical_image_relpath`，因为它是相对最终数据集目录的稳定路径。
    如果这个字段缺失或文件不存在，再退回 `source_image_path` 原始来源路径。
    """

    rel = row.get("canonical_image_relpath", "")
    if rel:
        candidate = dataset_root / Path(rel)
        if candidate.exists():
            return candidate
    source = row.get("source_image_path", "")
    if source:
        candidate = Path(source)
        if candidate.exists():
            return candidate
    filename = row.get("Filename", "<missing filename>")
    raise FileNotFoundError(f"Image not found for {filename}")


def choose_rows(rows: list[dict[str, str]], count: int | None, seed: int, salt: str) -> list[dict[str, str]]:
    """从 manifest 行里抽固定数量的样本。

    `salt` 是额外标签，用来保证 train/val、normal/defect 即使用同一个 seed，
    抽样序列也彼此独立。
    """

    if count is None or count >= len(rows):
        # None 表示不抽样、全量使用；count 比总数还大时也直接全用。
        return list(rows)
    rng = random.Random(f"{seed}:{salt}")
    return rng.sample(rows, count)


def assert_safe_generated_path(path: Path, allowed_root: Path) -> None:
    """删除临时目录前的保险检查。

    本脚本会重建 `data/stage1_cls_workdir/smoke` 或 `full`。
    这个函数确保删除目标一定在 work_root 里面，防止路径写错误删数据集。
    """

    path = path.resolve()
    allowed_root = allowed_root.resolve()
    if path == allowed_root:
        raise ValueError(f"Refusing to remove work root itself: {path}")
    if allowed_root not in path.parents:
        raise ValueError(f"Refusing to remove path outside generated work root: {path}")


def link_or_copy(src: Path, dst: Path) -> str:
    """把图片放进 YOLO 分类目录，优先硬链接，失败再复制。

    硬链接不额外占一份图片空间，速度快；如果跨盘或文件系统不支持硬链接，
    就自动退回普通复制，保证脚本仍然能跑。
    """

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        # 这里删的是临时目录里的目标文件，不是原始数据集里的源图片。
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def populate_class_dir(rows: list[dict[str, str]], class_dir: Path, dataset_root: Path) -> str:
    """把一批图片放进某个类别目录。

    例如把 normal_train_manifest.csv 里的图片放到 `train/no_target/`。
    """

    methods: set[str] = set()
    seen: set[str] = set()
    for row in rows:
        filename = row["Filename"]
        if filename in seen:
            raise ValueError(f"Duplicate filename in selected rows: {filename}")
        seen.add(filename)

        # 先根据 CSV 找到源图，再硬链接/复制到 YOLO 需要的分类目录。
        src = canonical_image_path(row, dataset_root)
        method = link_or_copy(src, class_dir / filename)
        methods.add(method)
    return "+".join(sorted(methods))


def prepare_cls_dataset(paths: Paths, cfg: TrainConfig) -> tuple[Path, DatasetCounts]:
    """准备 YOLO-cls 能直接读取的临时数据目录。

    YOLO 分类训练需要这样的目录：
        train/no_target/*.jpg
        train/target_defect/*.jpg
        val/no_target/*.jpg
        val/target_defect/*.jpg

    我们最终数据集不是这种结构，所以这里按 CSV manifest 临时组装一份。
    """

    # 读取四个已经确定好的 split：
    # train defect、train normal、val_model defect、val_model normal。
    train_target_rows = read_manifest(paths.manifest_dir / "train_manifest.csv")
    train_no_target_rows = read_manifest(paths.manifest_dir / "normal_train_manifest.csv")
    val_target_rows = read_manifest(paths.manifest_dir / "val_model_manifest.csv")
    val_no_target_rows = read_manifest(paths.manifest_dir / "normal_val_model_manifest.csv")

    if cfg.mode == "smoke":
        # smoke 只抽很少的图，目的是快速验证代码、依赖、输出目录是否正常。
        train_target_rows = choose_rows(train_target_rows, cfg.train_per_class, cfg.seed, "smoke-train-target")
        train_no_target_rows = choose_rows(train_no_target_rows, cfg.train_per_class, cfg.seed, "smoke-train-no-target")
        val_target_rows = choose_rows(val_target_rows, cfg.val_per_class, cfg.seed, "smoke-val-target")
        val_no_target_rows = choose_rows(val_no_target_rows, cfg.val_per_class, cfg.seed, "smoke-val-no-target")

    dataset_dir = paths.work_root / cfg.mode
    if cfg.rebuild_data and dataset_dir.exists():
        # 默认每次重建临时目录，避免上一次残留图片污染这一次训练。
        assert_safe_generated_path(dataset_dir, paths.work_root)
        shutil.rmtree(dataset_dir)

    # 这里定义“哪个 split + 哪个类别”对应哪一批 CSV 行。
    split_rows = {
        ("train", "target_defect"): train_target_rows,
        ("train", "no_target"): train_no_target_rows,
        ("val", "target_defect"): val_target_rows,
        ("val", "no_target"): val_no_target_rows,
    }
    for (split, class_name), rows in split_rows.items():
        # 逐类落盘成 YOLO-cls 的目录结构。
        method = populate_class_dir(rows, dataset_dir / split / class_name, paths.dataset_root)
        print(f"prepared {split}/{class_name}: {len(rows)} images ({method})")

    # 把实际图片数量记下来，后面写 summary 用。
    counts = DatasetCounts(
        train_no_target=len(train_no_target_rows),
        train_target_defect=len(train_target_rows),
        val_no_target=len(val_no_target_rows),
        val_target_defect=len(val_target_rows),
    )
    return dataset_dir, counts


def import_local_ultralytics(yolo_root: Path):
    """从本仓库里的 YOLOv11 源码导入 `YOLO` 类。

    这样做的好处是：训练用的是我们仓库里的官方 YOLOv11 代码，
    而不是电脑全局环境里可能安装过的其他 ultralytics 版本。
    """

    if not yolo_root.exists():
        raise FileNotFoundError(f"Missing YOLOv11 root: {yolo_root}")

    # 把 YOLOv11 放到 Python 搜索路径最前面，确保优先 import 本地源码。
    sys.path.insert(0, str(yolo_root))
    from ultralytics import YOLO  # noqa: PLC0415

    return YOLO


def weight_path(paths: Paths, model_key: str) -> Path:
    """找到某个模型规模对应的预训练权重。

    例如 model_key="n" 时，寻找 yolo11n-cls.pt。
    """

    filename = MODEL_WEIGHTS[model_key]

    # 支持三种常见摆放方式，方便你在不同电脑上部署：
    # 1. 仓库根目录
    # 2. YOLOv11 目录
    # 3. YOLOv11/weights 目录
    candidates = [
        paths.repo_root / filename,
        paths.yolo_root / filename,
        paths.yolo_root / "weights" / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing model weight {filename}; checked: {candidates}")


class Tee:
    """把同一段输出同时写到终端和日志文件。

    YOLO 训练时会往 stdout/stderr 打印很多重要信息。这个类让人能在终端实时看，
    同时也把它保存成 `train_log.txt`，避免训练完以后只剩权重、不知道过程。
    """

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


@contextlib.contextmanager
def tee_output(log_path: Path):
    """临时把 stdout/stderr 复制一份到日志文件。"""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        sys.stdout = Tee(old_stdout, log_file)
        sys.stderr = Tee(old_stderr, log_file)
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def sha256_file(path: Path) -> str:
    """计算文件 sha256，用来确认材料没有被篡改或传输损坏。"""

    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, base_dir: Path | None = None) -> dict[str, str]:
    """把一个文件整理成 manifest 里的一行记录。"""

    exists = path.exists()
    rel = path
    if base_dir is not None:
        try:
            rel = path.relative_to(base_dir)
        except ValueError:
            rel = path
    return {
        "path": str(path),
        "relative_path": str(rel).replace("\\", "/"),
        "exists": str(exists),
        "size_bytes": str(path.stat().st_size) if exists else "",
        "sha256": sha256_file(path) if exists and path.is_file() else "",
        "modified_time": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds") if exists else "",
    }


def run_command(args: list[str], cwd: Path) -> str:
    """执行一个只读命令并返回输出；失败时返回错误文本而不是中断训练。"""

    try:
        result = subprocess.run(args, cwd=cwd, text=True, capture_output=True, check=False, timeout=30)
    except Exception as exc:
        return f"<error: {exc}>"
    output = (result.stdout or "").strip()
    error = (result.stderr or "").strip()
    if result.returncode == 0:
        return output
    return f"<exit {result.returncode}> {output} {error}".strip()


def package_version(package_name: str) -> str:
    """读取 Python 包版本；包不存在时返回空字符串。"""

    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return ""


def torch_environment() -> dict[str, object]:
    """记录 torch/CUDA/GPU 信息。

    这部分写进 run_meta，方便以后解释“同一代码在不同机器上为什么结果略有差异”。
    """

    try:
        import torch  # noqa: PLC0415
    except Exception as exc:
        return {"torch_import_error": str(exc)}

    gpu_names: list[str] = []
    if torch.cuda.is_available():
        gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return {
        "torch_version": getattr(torch, "__version__", ""),
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": getattr(torch.version, "cuda", ""),
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "",
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_names": gpu_names,
    }


def manifest_records(paths: Paths) -> list[dict[str, str]]:
    """记录训练用 CSV manifest 的文件 hash。"""

    records = []
    for filename in TRAIN_MANIFEST_FILES:
        path = paths.manifest_dir / filename
        record = file_record(path, paths.dataset_root)
        record["kind"] = "train_manifest"
        records.append(record)
    return records


def git_info(repo_root: Path) -> dict[str, str]:
    """记录代码版本。"""

    status = run_command(["git", "status", "--short"], repo_root)
    return {
        "branch": run_command(["git", "branch", "--show-current"], repo_root),
        "commit": run_command(["git", "rev-parse", "HEAD"], repo_root),
        "commit_short": run_command(["git", "rev-parse", "--short", "HEAD"], repo_root),
        "dirty": str(bool(status)),
        "status_short": status,
    }


def finalize_train_log(temp_log_path: Path, run_dir: Path) -> Path:
    """把临时训练日志移动到 run 目录里的固定文件名。"""

    run_dir.mkdir(parents=True, exist_ok=True)
    final_log = run_dir / TRAIN_LOG_FILENAME
    if temp_log_path.exists():
        if final_log.exists():
            final_log.unlink()
        shutil.move(str(temp_log_path), str(final_log))
    elif not final_log.exists():
        final_log.write_text("", encoding="utf-8")
    return final_log


def write_run_meta(
    paths: Paths,
    cfg: TrainConfig,
    counts: DatasetCounts,
    model_key: str,
    model_weight: Path,
    dataset_dir: Path,
    run_dir: Path,
    run_name: str,
    status: str,
    error: str,
    started_at: str,
    ended_at: str,
    duration: float,
    train_log_path: Path,
) -> Path:
    """写 `run_meta.json`。

    它是本次训练的身份证：代码、数据、命令、环境、输出目录都在这里。
    """

    meta = {
        "run_name": run_name,
        "model_key": model_key,
        "model_weight": str(model_weight),
        "status": status,
        "error": error,
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_sec": round(duration, 2),
        "command": " ".join(sys.argv),
        "paths": {
            "repo_root": str(paths.repo_root),
            "yolo_root": str(paths.yolo_root),
            "dataset_root": str(paths.dataset_root),
            "manifest_dir": str(paths.manifest_dir),
            "dataset_dir": str(dataset_dir),
            "run_dir": str(run_dir),
            "train_log": str(train_log_path),
        },
        "config": {
            "mode": cfg.mode,
            "models": list(cfg.models),
            "seed": cfg.seed,
            "epochs": cfg.epochs,
            "imgsz": cfg.imgsz,
            "batch": cfg.batch,
            "workers": cfg.workers,
            "save_period": cfg.save_period,
            "device": cfg.device,
            "rebuild_data": cfg.rebuild_data,
            "train_per_class": cfg.train_per_class,
            "val_per_class": cfg.val_per_class,
            "dry_run": cfg.dry_run,
            "exist_ok": cfg.exist_ok,
        },
        "dataset_counts": {
            "train_no_target": counts.train_no_target,
            "train_target_defect": counts.train_target_defect,
            "val_no_target": counts.val_no_target,
            "val_target_defect": counts.val_target_defect,
        },
        "git": git_info(paths.repo_root),
        "environment": {
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "packages": {
                "ultralytics": package_version("ultralytics"),
                "torch": package_version("torch"),
                "torchvision": package_version("torchvision"),
                "numpy": package_version("numpy"),
                "opencv-python": package_version("opencv-python"),
            },
            "torch": torch_environment(),
        },
        "manifests": manifest_records(paths),
    }
    path = run_dir / RUN_META_FILENAME
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_artifact_manifest(run_dir: Path) -> tuple[Path, Path, list[str]]:
    """扫描 run 目录，写材料清单 CSV/JSON，并返回缺失的必需材料。"""

    run_dir.mkdir(parents=True, exist_ok=True)
    required_set = set(REQUIRED_ARTIFACTS)
    rows: list[dict[str, str]] = []
    seen: set[str] = set()

    for rel_path in REQUIRED_ARTIFACTS:
        path = run_dir / rel_path
        row = file_record(path, run_dir)
        row["category"] = "required"
        row["required"] = "True"
        rows.append(row)
        seen.add(rel_path)

    for path in sorted(p for p in run_dir.rglob("*") if p.is_file()):
        rel_path = str(path.relative_to(run_dir)).replace("\\", "/")
        if rel_path in seen or rel_path in {ARTIFACT_MANIFEST_CSV_FILENAME, ARTIFACT_MANIFEST_JSON_FILENAME}:
            continue
        row = file_record(path, run_dir)
        row["category"] = "optional"
        row["required"] = "False"
        rows.append(row)
        seen.add(rel_path)

    missing_required = [
        row["relative_path"]
        for row in rows
        if row["relative_path"] in required_set and row["exists"] != "True"
    ]

    fields = [
        "relative_path",
        "path",
        "category",
        "required",
        "exists",
        "size_bytes",
        "sha256",
        "modified_time",
    ]
    csv_path = run_dir / ARTIFACT_MANIFEST_CSV_FILENAME
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    json_path = run_dir / ARTIFACT_MANIFEST_JSON_FILENAME
    json_path.write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "required_artifacts": list(REQUIRED_ARTIFACTS),
                "missing_required_artifacts": missing_required,
                "artifacts": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return csv_path, json_path, missing_required


def summary_path(paths: Paths, mode: str) -> Path:
    """根据 smoke/full 返回本轮 summary CSV 的路径。"""

    name = SUMMARY_FILENAMES[mode]
    return paths.runs_root / name


def append_summary(paths: Paths, cfg: TrainConfig, counts: DatasetCounts, row: dict[str, str]) -> None:
    """把一次模型训练的关键信息追加写入 summary CSV。

    即使训练中途失败，`train_one_model()` 的 finally 也会调用这里，
    所以我们能看到失败发生在哪个模型、用了什么数据、错误是什么。
    """

    path = summary_path(paths, cfg.mode)
    path.parent.mkdir(parents=True, exist_ok=True)

    # CSV 列名固定，方便后续用 Excel、pandas 或论文统计脚本读取。
    fields = [
        "timestamp",
        "mode",
        "model_key",
        "model_weight",
        "epochs",
        "imgsz",
        "batch",
        "workers",
        "save_period",
        "device",
        "seed",
        "dataset_dir",
        "train_no_target",
        "train_target_defect",
        "val_no_target",
        "val_target_defect",
        "run_dir",
        "best_pt_exists",
        "last_pt_exists",
        "results_csv_exists",
        "args_yaml_exists",
        "train_log_exists",
        "run_meta_exists",
        "artifact_manifest_csv_exists",
        "artifact_manifest_json_exists",
        "missing_required_artifacts",
        "status",
        "error",
        "duration_sec",
    ]

    # 这一部分是所有模型通用的信息：模式、epoch、图片数量等。
    # `row` 里面会补上当前模型特有的信息：模型名、run_dir、是否生成 best.pt 等。
    full_row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "mode": cfg.mode,
        "epochs": str(cfg.epochs),
        "imgsz": str(cfg.imgsz),
        "batch": str(cfg.batch),
        "workers": str(cfg.workers),
        "save_period": str(cfg.save_period),
        "device": cfg.device,
        "seed": str(cfg.seed),
        "train_no_target": str(counts.train_no_target),
        "train_target_defect": str(counts.train_target_defect),
        "val_no_target": str(counts.val_no_target),
        "val_target_defect": str(counts.val_target_defect),
        **row,
    }
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)
            existing_fields = reader.fieldnames or []
        if existing_fields != fields:
            # summary 的列可能随脚本升级而增加。这里重写旧行，避免新旧列错位。
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(existing_rows)
                writer.writerow(full_row)
            return

    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not exists:
            # 第一次创建 CSV 时写表头；后面继续追加行。
            writer.writeheader()
        writer.writerow(full_row)


def train_one_model(model_key: str, paths: Paths, cfg: TrainConfig, dataset_dir: Path, counts: DatasetCounts) -> Path:
    """训练一个 YOLO11-cls 模型。

    五个模型 n/s/m/l/x 最终都会走这里，区别只是传入的 `model_key` 不同。
    """

    started = time.time()
    started_at = datetime.now().isoformat(timespec="seconds")
    model_weight = weight_path(paths, model_key)

    # 每次 run 名里加时间戳，避免覆盖旧结果。
    run_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{cfg.mode}_yolo11{model_key}_cls_{run_stamp}"
    run_dir = paths.runs_root / run_name
    temp_log_path = paths.runs_root / TEMP_LOG_DIRNAME / f"{run_name}{TEMP_LOG_SUFFIX}"

    # 默认先认为失败；只有 YOLO 训练完整跑完后才改成 ok。
    status = "failed"
    error = ""
    model = None
    try:
        with tee_output(temp_log_path):
            if cfg.dry_run:
                # dry-run 只检查数据准备和权重路径，不真正占用 GPU 训练。
                print(f"[dry-run] would train {model_key} with {model_weight}")
                status = "dry_run"
            else:
                YOLO = import_local_ultralytics(paths.yolo_root)

                # 从预训练分类权重初始化模型；这就是后面正式训练的起点。
                model = YOLO(str(model_weight))

                # 这一段是 Ultralytics 官方训练入口。真正的训练循环在 YOLOv11 源码里。
                model.train(
                    # data 指向刚刚临时组装好的 YOLO-cls 数据目录。
                    data=str(dataset_dir),
                    # 训练轮数。full 默认 200，smoke 默认 2。
                    epochs=cfg.epochs,
                    # 输入图片尺寸。
                    imgsz=cfg.imgsz,
                    # batch 越大通常越快，但显存占用也越大。
                    batch=cfg.batch,
                    # Windows 上 workers=0 最稳；训练机稳定后可以再调大。
                    workers=cfg.workers,
                    # 每个 epoch 都保存 checkpoint，10TB 网盘场景下优先保留不可再生材料。
                    save_period=cfg.save_period,
                    # 选择训练设备，例如 "0" 表示第 0 张 GPU。
                    device=cfg.device,
                    # project/name 共同决定输出目录。
                    project=str(paths.runs_root),
                    name=run_name,
                    # 默认 False，避免同名目录被复用；需要复用时手动传 --exist-ok。
                    exist_ok=cfg.exist_ok,
                    # 保证训练可复现性；仍然可能受 CUDA/驱动等底层因素影响。
                    seed=cfg.seed,
                    deterministic=True,
                    # 不把图片缓存进内存/磁盘，避免额外占空间。
                    cache=False,
                    # 每轮训练后在 val_model 上验证。
                    val=True,
                    # 生成 results.png、confusion_matrix.png 等图表。
                    plots=True,
                    verbose=True,
                    # 明确告诉 Ultralytics 这是分类任务，不是检测/分割任务。
                    task="classify",
                )
                trainer = getattr(model, "trainer", None)
                save_dir = getattr(trainer, "save_dir", None)
                if save_dir:
                    run_dir = Path(save_dir)
                status = "ok"
    except Exception as exc:  # keep summary even when smoke exposes a bug
        # 训练失败时先把简短错误写入 summary，再把完整 traceback 打印出来。
        error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        full_traceback = traceback.format_exc()
        if temp_log_path.exists():
            with temp_log_path.open("a", encoding="utf-8", errors="replace") as f:
                f.write("\n\n=== Python traceback ===\n")
                f.write(full_traceback)
        print(full_traceback)
        raise
    finally:
        # finally 无论成功/失败都会执行，所以 summary 不会因为异常丢失。
        duration = time.time() - started
        ended_at = datetime.now().isoformat(timespec="seconds")
        train_log_path = finalize_train_log(temp_log_path, run_dir)
        run_meta_path = write_run_meta(
            paths,
            cfg,
            counts,
            model_key,
            model_weight,
            dataset_dir,
            run_dir,
            run_name,
            status,
            error,
            started_at,
            ended_at,
            duration,
            train_log_path,
        )
        artifact_manifest_csv, artifact_manifest_json, missing_required = write_artifact_manifest(run_dir)
        append_summary(
            paths,
            cfg,
            counts,
            {
                "model_key": model_key,
                "model_weight": str(model_weight),
                "dataset_dir": str(dataset_dir),
                "run_dir": str(run_dir),
                # 这些布尔列能快速判断本轮是否真的产出了关键结果文件。
                "best_pt_exists": str((run_dir / WEIGHTS_DIRNAME / BEST_WEIGHT_FILENAME).exists()),
                "last_pt_exists": str((run_dir / WEIGHTS_DIRNAME / LAST_WEIGHT_FILENAME).exists()),
                "results_csv_exists": str((run_dir / RESULTS_CSV_FILENAME).exists()),
                "args_yaml_exists": str((run_dir / ARGS_YAML_FILENAME).exists()),
                "train_log_exists": str(train_log_path.exists()),
                "run_meta_exists": str(run_meta_path.exists()),
                "artifact_manifest_csv_exists": str(artifact_manifest_csv.exists()),
                "artifact_manifest_json_exists": str(artifact_manifest_json.exists()),
                "missing_required_artifacts": ";".join(missing_required),
                "status": status,
                "error": error,
                "duration_sec": f"{duration:.2f}",
            },
        )
    return run_dir


def train_yolo11n_cls(paths: Paths, cfg: TrainConfig, dataset_dir: Path, counts: DatasetCounts) -> Path:
    """独立入口：训练 yolo11n-cls，最小最快。"""

    return train_one_model("n", paths, cfg, dataset_dir, counts)


def train_yolo11s_cls(paths: Paths, cfg: TrainConfig, dataset_dir: Path, counts: DatasetCounts) -> Path:
    """独立入口：训练 yolo11s-cls。"""

    return train_one_model("s", paths, cfg, dataset_dir, counts)


def train_yolo11m_cls(paths: Paths, cfg: TrainConfig, dataset_dir: Path, counts: DatasetCounts) -> Path:
    """独立入口：训练 yolo11m-cls。"""

    return train_one_model("m", paths, cfg, dataset_dir, counts)


def train_yolo11l_cls(paths: Paths, cfg: TrainConfig, dataset_dir: Path, counts: DatasetCounts) -> Path:
    """独立入口：训练 yolo11l-cls。"""

    return train_one_model("l", paths, cfg, dataset_dir, counts)


def train_yolo11x_cls(paths: Paths, cfg: TrainConfig, dataset_dir: Path, counts: DatasetCounts) -> Path:
    """独立入口：训练 yolo11x-cls，最大最慢。"""

    return train_one_model("x", paths, cfg, dataset_dir, counts)


# 训练调度表：字符串模型代号 -> 对应的独立训练函数。
# 后面批量训练时用 TRAINERS[model_key](...) 调用。
TRAINERS: dict[str, Callable[[Paths, TrainConfig, Path, DatasetCounts], Path]] = {
    "n": train_yolo11n_cls,
    "s": train_yolo11s_cls,
    "m": train_yolo11m_cls,
    "l": train_yolo11l_cls,
    "x": train_yolo11x_cls,
}


def run_selected_models(paths: Paths, cfg: TrainConfig) -> list[Path]:
    """准备数据，然后按配置顺序逐个训练模型。"""

    dataset_dir, counts = prepare_cls_dataset(paths, cfg)
    print(f"dataset_dir={dataset_dir}")
    print(f"counts={counts}")

    run_dirs = []
    for model_key in cfg.models:
        # 这里是串行训练：一个模型完成后再训练下一个模型。
        # 两台训练机分工时，只需要在命令行里给每台机器传不同 --models。
        print(f"=== train yolo11{model_key}-cls ({cfg.mode}) ===")
        run_dirs.append(TRAINERS[model_key](paths, cfg, dataset_dir, counts))
    return run_dirs


def parse_args() -> argparse.Namespace:
    """定义命令行参数。

    这个函数只负责“允许用户传什么参数”，不真正执行训练。
    很多参数同时支持环境变量，是为了部署到不同训练机时更方便批量控制。
    """

    parser = argparse.ArgumentParser(description="Stage-1 YOLO11-cls binary training sweep.")

    # 选择小样本试跑还是正式训练。
    parser.add_argument("--mode", choices=("smoke", "full"), default=os.environ.get("STAGE1_MODE", "smoke"))

    # 指定模型列表，例如：
    #   --models n
    #   --models n,s,m
    #   --models l,x
    # 不传时：smoke 默认 n；full 默认 n,s,m,l,x。
    parser.add_argument("--models", default=None, help="Comma-separated model keys, e.g. n,s,m or l,x.")

    # 复现相关参数。
    parser.add_argument("--seed", type=int, default=int(os.environ.get("STAGE1_SEED", SEED)))

    # 训练超参数。epochs 不给时，main() 会根据 mode 自动填 2 或 200。
    parser.add_argument("--epochs", type=int, default=None, help="Defaults to 2 for smoke and 200 for full.")
    parser.add_argument("--imgsz", type=int, default=int(os.environ.get("STAGE1_IMGSZ", DEFAULT_IMGSZ)))
    parser.add_argument("--batch", type=int, default=int(os.environ.get("STAGE1_BATCH", 128)))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("STAGE1_WORKERS", 4)))
    parser.add_argument(
        "--save-period",
        type=int,
        default=int(os.environ.get("STAGE1_SAVE_PERIOD", 1)),
        help="Save checkpoint every N epochs. Use -1 to keep only best.pt and last.pt.",
    )
    parser.add_argument("--device", default=os.environ.get("STAGE1_DEVICE", "0"))

    # smoke 模式专用：控制每类抽多少小样本。
    # full 模式一般不传这两个参数，因为 full 要用完整训练/验证集。
    parser.add_argument("--train-per-class", type=int, default=None)
    parser.add_argument("--val-per-class", type=int, default=None)

    # 数据与运行控制。
    parser.add_argument("--keep-data", action="store_true", help="Reuse generated classification workdir.")
    parser.add_argument("--dry-run", action="store_true", help="Prepare data and write dry-run summary without training.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow Ultralytics to reuse an existing run name.")

    # 路径覆盖参数。默认都用本仓库约定目录；换机器时也可以用这些参数指向其他盘。
    parser.add_argument("--dataset-root", default=os.environ.get("STAGE1_DATASET_ROOT"))
    parser.add_argument("--work-root", default=os.environ.get("STAGE1_WORK_ROOT"))
    parser.add_argument("--runs-root", default=os.environ.get("STAGE1_RUNS_ROOT"))
    parser.add_argument("--yolo-root", default=os.environ.get("STAGE1_YOLO_ROOT"))
    return parser.parse_args()


def main() -> int:
    """脚本主入口：解析参数 -> 组装配置 -> 打印信息 -> 开始训练。"""

    args = parse_args()

    # 把用户传入的模型字符串转换成 ("n", "s", ...) 这种安全的元组。
    models = parse_models(args.models, args.mode)

    # 如果用户没手动指定 epoch，就按模式给默认值。
    epochs = args.epochs if args.epochs is not None else (SMOKE_EPOCHS if args.mode == "smoke" else FULL_EPOCHS)

    train_per_class = args.train_per_class
    val_per_class = args.val_per_class
    if args.mode == "smoke":
        # smoke 不传数量时，用脚本顶部的小样本默认值。
        train_per_class = train_per_class or SMOKE_TRAIN_PER_CLASS
        val_per_class = val_per_class or SMOKE_VAL_PER_CLASS

    # 把零散参数收拢成一个配置对象，后面函数只传 cfg，代码更清楚。
    cfg = TrainConfig(
        mode=args.mode,
        models=models,
        seed=args.seed,
        epochs=epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        save_period=args.save_period,
        device=args.device,
        rebuild_data=not args.keep_data,
        train_per_class=train_per_class,
        val_per_class=val_per_class,
        dry_run=args.dry_run,
        exist_ok=args.exist_ok,
    )

    # 根据参数推导所有目录。
    paths = build_paths(args)

    # 先把关键路径和本轮配置打印出来，方便你在终端第一眼核对有没有跑错目录。
    print(f"repo_root={paths.repo_root}")
    print(f"yolo_root={paths.yolo_root}")
    print(f"dataset_root={paths.dataset_root}")
    print(f"runs_root={paths.runs_root}")
    print(f"mode={cfg.mode} models={','.join(cfg.models)} epochs={cfg.epochs} save_period={cfg.save_period}")

    # 真正开始：准备 YOLO-cls 临时数据目录，并按 models 顺序训练。
    run_dirs = run_selected_models(paths, cfg)

    # 最后把所有输出目录和 summary CSV 位置打印出来，方便人工检查结果。
    print("completed runs:")
    for run_dir in run_dirs:
        print(run_dir)
    print(f"summary={summary_path(paths, cfg.mode)}")
    return 0


if __name__ == "__main__":
    # 只有直接运行这个文件时才进入 main()。
    # 如果别的脚本 import 这个文件，只会拿到函数，不会自动开始训练。
    raise SystemExit(main())
