from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the formal stage-1 capacity-scan bundle.")
    parser.add_argument("--gate-summary-csv", required=True)
    parser.add_argument("--gate-top1-csv", required=True)
    parser.add_argument("--gate-materials-root", required=True)
    parser.add_argument("--gate-run-root", required=False, default="")
    parser.add_argument("--cls6-summary-csv", required=True)
    parser.add_argument("--cls6-materials-root", required=True)
    parser.add_argument("--cls6-run-root", required=True)
    parser.add_argument("--registry-csv", required=True)
    parser.add_argument("--protocol-md", required=True)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, str]] = []
        for row in reader:
            rows.append({str(key).strip(): str(value).strip() for key, value in row.items() if key is not None})
        return rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    write_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def as_float(value: Any) -> float:
    if value in ("", None, "NA"):
        return float("nan")
    return float(value)


def format_num(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "NA"


def normalize_model_key(model: str) -> str:
    text = str(model).strip()
    if text.endswith("-cls"):
        text = text[:-4]
    return text


def gate_rank_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        as_float(row["spec_at_r995"]),
        as_float(row["spec_at_r990"]),
        as_float(row["prec_at_r990"]),
        -as_float(row["ptr_at_r990"]),
    )


def cls6_rank_key(row: dict[str, Any]) -> tuple[float, float, float]:
    return (
        as_float(row["accuracy"]),
        as_float(row["auroc"]),
        as_float(row["auprc"]),
    )


def md_value(value: str, *, best: bool = False, second: bool = False) -> str:
    if best:
        return f"**{value}**"
    if second:
        return f"_{value}_"
    return value


def sort_with_ranks(rows: list[dict[str, Any]], key_fn) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=key_fn, reverse=True)
    for rank, row in enumerate(ordered, start=1):
        row["rank"] = rank
    return ordered


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
    return True


def load_gate_rows(path: Path) -> list[dict[str, Any]]:
    rows = read_csv_rows(path)
    for row in rows:
        row["model_key"] = normalize_model_key(str(row["model"]))
    return sort_with_ranks(rows, gate_rank_key)


def load_gate_top1_rows(path: Path) -> list[dict[str, Any]]:
    rows = read_csv_rows(path)
    for row in rows:
        row["model_key"] = normalize_model_key(str(row["model"]))
        row["model"] = f"{row['model_key']}-cls"
        row["delta_epoch"] = int(row["top1_best_epoch"]) - int(row["gate_best_epoch"])
    return rows


def load_cls6_rows(path: Path) -> list[dict[str, Any]]:
    rows = read_csv_rows(path)
    for row in rows:
        row["model_key"] = normalize_model_key(str(row["model"]))
    return sort_with_ranks(rows, cls6_rank_key)


def build_gate_main_tables(rows: list[dict[str, Any]], table_root: Path) -> None:
    fieldnames = ["Model", "Best Epoch", "Spec@R99.5", "Spec@R99.0", "Prec@R99.0", "PTR@R99.0", "tau_R99.5", "tau_R99.0", "Rank"]
    csv_rows = [
        {
            "Model": row["model"],
            "Best Epoch": row["best_epoch"],
            "Spec@R99.5": format_num(row["spec_at_r995"]),
            "Spec@R99.0": format_num(row["spec_at_r990"]),
            "Prec@R99.0": format_num(row["prec_at_r990"]),
            "PTR@R99.0": format_num(row["ptr_at_r990"]),
            "tau_R99.5": format_num(row["tau_r995"]),
            "tau_R99.0": format_num(row["tau_r990"]),
            "Rank": row["rank"],
        }
        for row in rows
    ]
    write_csv(table_root / "table_stage1_gate_capacity_main.csv", fieldnames, csv_rows)

    metric_fields = ["spec_at_r995", "spec_at_r990", "prec_at_r990"]
    best_values = {field: max(as_float(row[field]) for row in rows) for field in metric_fields}
    second_values: dict[str, float] = {}
    for field in metric_fields:
        ordered = sorted({as_float(row[field]) for row in rows}, reverse=True)
        second_values[field] = ordered[1] if len(ordered) > 1 else ordered[0]
    best_values["ptr_at_r990"] = min(as_float(row["ptr_at_r990"]) for row in rows)
    ordered_ptr = sorted({as_float(row["ptr_at_r990"]) for row in rows})
    second_values["ptr_at_r990"] = ordered_ptr[1] if len(ordered_ptr) > 1 else ordered_ptr[0]

    lines = [
        "# Table: Stage-1 Direct Binary Gate Formal Capacity Scan",
        "",
        "| Model | Best Epoch | Spec@R99.5 (up) | Spec@R99.0 (up) | Prec@R99.0 (up) | PTR@R99.0 (down) | tau_R99.5 | tau_R99.0 | Rank |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        cells = []
        for field in ["spec_at_r995", "spec_at_r990", "prec_at_r990", "ptr_at_r990"]:
            value = as_float(row[field])
            best = value == best_values[field]
            second = value == second_values[field] and not best
            cells.append(md_value(format_num(value), best=best, second=second))
        lines.append(
            "| {model} | {best_epoch} | {spec} | {spec2} | {prec} | {ptr} | {tau1} | {tau2} | {rank} |".format(
                model=row["model"],
                best_epoch=row["best_epoch"],
                spec=cells[0],
                spec2=cells[1],
                prec=cells[2],
                ptr=cells[3],
                tau1=format_num(row["tau_r995"]),
                tau2=format_num(row["tau_r990"]),
                rank=row["rank"],
            )
        )
    lines.extend(
        [
            "",
            "Note:",
            "Binary gate is the primary stage-1 selection view.",
            "Ranking follows `Spec@R99.5 > Spec@R99.0 > Prec@R99.0 > PTR@R99.0`.",
        ]
    )
    write_text(table_root / "table_stage1_gate_capacity_main.md", "\n".join(lines) + "\n")


def build_gate_top1_table(rows: list[dict[str, Any]], table_root: Path) -> None:
    rows = sorted(rows, key=lambda item: int(item["gate_best_epoch"]))
    fieldnames = [
        "Model",
        "Top1-Best Epoch",
        "Top1-Best Top1",
        "Gate-Best Epoch",
        "Gate-Best Spec@R99.5",
        "Gate-Best Spec@R99.0",
        "Delta Epoch",
        "Same Checkpoint?",
    ]
    csv_rows = [
        {
            "Model": row["model"],
            "Top1-Best Epoch": row["top1_best_epoch"],
            "Top1-Best Top1": format_num(row["top1_best_value"]),
            "Gate-Best Epoch": row["gate_best_epoch"],
            "Gate-Best Spec@R99.5": format_num(row["gate_best_spec_at_r995"]),
            "Gate-Best Spec@R99.0": format_num(row["gate_best_spec_at_r990"]),
            "Delta Epoch": row["delta_epoch"],
            "Same Checkpoint?": "Yes" if str(row["same_as_gate_best"]).strip().lower() == "yes" else "No",
        }
        for row in rows
    ]
    write_csv(table_root / "table_stage1_gate_top1_vs_gatebest.csv", fieldnames, csv_rows)

    lines = [
        "# Table: Trainer Top1-Best vs Gate-Best",
        "",
        "| Model | Top1-Best Epoch | Top1-Best Top1 | Gate-Best Epoch | Gate-Best Spec@R99.5 | Gate-Best Spec@R99.0 | Delta Epoch | Same Checkpoint? |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {model} | {top1_epoch} | {top1_val} | {gate_epoch} | {gate_spec} | {gate_spec990} | {delta} | {same} |".format(
                model=row["model"],
                top1_epoch=row["top1_best_epoch"],
                top1_val=format_num(row["top1_best_value"]),
                gate_epoch=row["gate_best_epoch"],
                gate_spec=format_num(row["gate_best_spec_at_r995"]),
                gate_spec990=format_num(row["gate_best_spec_at_r990"]),
                delta=row["delta_epoch"],
                same="Yes" if str(row["same_as_gate_best"]).strip().lower() == "yes" else "No",
            )
        )
    lines.extend(
        [
            "",
            "Note:",
            "Top1-best and gate-best are not aligned under the formal binary-gate objective.",
            "Trainer-side `top1` remains useful for training-health monitoring, but should not be used as the official early-stop or checkpoint-selection criterion for stage-1 gate.",
        ]
    )
    write_text(table_root / "table_stage1_gate_top1_vs_gatebest.md", "\n".join(lines) + "\n")


def build_cls6_main_tables(rows: list[dict[str, Any]], table_root: Path) -> None:
    fieldnames = ["Model", "Best Epoch", "Accuracy", "AUROC", "AUPRC", "Rank"]
    csv_rows = [
        {
            "Model": row["model"],
            "Best Epoch": row["best_epoch"],
            "Accuracy": format_num(row["accuracy"]),
            "AUROC": format_num(row["auroc"]),
            "AUPRC": format_num(row["auprc"]),
            "Rank": row["rank"],
        }
        for row in rows
    ]
    write_csv(table_root / "table_stage1_cls6_capacity_main.csv", fieldnames, csv_rows)

    best_values = {
        "accuracy": max(as_float(row["accuracy"]) for row in rows),
        "auroc": max(as_float(row["auroc"]) for row in rows),
        "auprc": max(as_float(row["auprc"]) for row in rows),
    }
    second_values: dict[str, float] = {}
    for field in ("accuracy", "auroc", "auprc"):
        ordered = sorted({as_float(row[field]) for row in rows}, reverse=True)
        second_values[field] = ordered[1] if len(ordered) > 1 else ordered[0]

    lines = [
        "# Table: Stage-1 Six-Class Source Capacity Scan",
        "",
        "| Model | Best Epoch | Accuracy (up) | AUROC (up) | AUPRC (up) | Rank |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        acc = as_float(row["accuracy"])
        auroc = as_float(row["auroc"])
        auprc = as_float(row["auprc"])
        lines.append(
            "| {model} | {best_epoch} | {acc} | {auroc} | {auprc} | {rank} |".format(
                model=row["model"],
                best_epoch=row["best_epoch"],
                acc=md_value(format_num(acc), best=acc == best_values["accuracy"], second=acc == second_values["accuracy"] and acc != best_values["accuracy"]),
                auroc=md_value(format_num(auroc), best=auroc == best_values["auroc"], second=auroc == second_values["auroc"] and auroc != best_values["auroc"]),
                auprc=md_value(format_num(auprc), best=auprc == best_values["auprc"], second=auprc == second_values["auprc"] and auprc != best_values["auprc"]),
                rank=row["rank"],
            )
        )
    lines.extend(
        [
            "",
            "Note:",
            "The six-class source view is an auxiliary source-side representation reference.",
            "It is not the official selection criterion for the stage-1 gate.",
        ]
    )
    write_text(table_root / "table_stage1_cls6_capacity_main.md", "\n".join(lines) + "\n")


def build_crossview_tables(gate_rows: list[dict[str, Any]], cls6_rows: list[dict[str, Any]], table_root: Path) -> list[dict[str, Any]]:
    gate_rank = {normalize_model_key(str(row["model"])): int(row["rank"]) for row in gate_rows}
    cls6_rank = {normalize_model_key(str(row["model"])): int(row["rank"]) for row in cls6_rows}
    rows: list[dict[str, Any]] = []
    for model_key in sorted(gate_rank):
        gap = cls6_rank[model_key] - gate_rank[model_key]
        if gap > 0:
            interpretation = "The auxiliary cls6 view ranks this model lower than the gate-aware view."
        elif gap < 0:
            interpretation = "The auxiliary cls6 view ranks this model higher than the gate-aware view."
        else:
            interpretation = "Both views agree on this model's relative position."
        rows.append(
            {
                "Model": f"{model_key}-cls",
                "Rank in CLS6": cls6_rank[model_key],
                "Rank in Binary Gate": gate_rank[model_key],
                "Rank Gap": gap,
                "Interpretation": interpretation,
            }
        )
    write_csv(
        table_root / "table_stage1_crossview_rank_gap.csv",
        ["Model", "Rank in CLS6", "Rank in Binary Gate", "Rank Gap", "Interpretation"],
        rows,
    )
    lines = [
        "# Table: Cross-View Rank Gap",
        "",
        "| Model | Rank in CLS6 | Rank in Binary Gate | Rank Gap | Interpretation |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append("| {Model} | {Rank in CLS6} | {Rank in Binary Gate} | {Rank Gap} | {Interpretation} |".format(**row))
    lines.extend(
        [
            "",
            "Note:",
            "The source-side six-class ranking cannot replace direct binary-gate selection.",
        ]
    )
    write_text(table_root / "table_stage1_crossview_rank_gap.md", "\n".join(lines) + "\n")
    return rows


def build_gate_report(rows: list[dict[str, Any]], top1_rows: list[dict[str, Any]], report_root: Path) -> None:
    mismatch_count = sum(1 for row in top1_rows if str(row["same_as_gate_best"]).strip().lower() != "yes")
    leader = rows[0]
    second = rows[1]
    lines = [
        "# Stage-1 Gate Capacity Formal Report",
        "",
        "## Protocol",
        "- Primary task: `direct binary gate`",
        "- Batch size: `24`",
        "- Epochs: `200`",
        "- Checkpoint policy: save every epoch",
        "- Formal ranking: `Spec@R99.5 > Spec@R99.0 > Prec@R99.0 > PTR@R99.0`",
        "- Trainer `top1/acc/loss` is retained only as training-health information and does not determine the formal checkpoint.",
        "",
        "## Main Table",
        "See `paper_main/tables/table_stage1_gate_capacity_main.md`.",
        "",
        "## Final Ranking",
    ]
    for row in rows:
        lines.append(f"{row['rank']}. `{row['model']}`")
    lines.extend(
        [
            "",
            "## Top1-Best vs Gate-Best Mismatch",
            "See `paper_main/tables/table_stage1_gate_top1_vs_gatebest.md`.",
            f"- `{mismatch_count}` out of `5` models show `top1-best != gate-best` under the formal protocol.",
            "- This mismatch indicates that trainer-side `top1` is not aligned with the recall-constrained gate objective and should not be used as the official checkpoint-selection rule.",
            "",
            "## Early-Peak and Checkpoint-Dynamics Analysis",
            "- No model reaches its gate-best checkpoint at epoch 1.",
            "- `yolo11m-cls` peaks at epoch `78`, indicating that the formal optimum is reached in the middle stage rather than at the end of training.",
            "- `yolo11x-cls` peaks at epoch `125`, suggesting that the largest-capacity model benefits from a longer optimization horizon, but still remains below the formal leader.",
            "- `yolo11l-cls` peaks earlier at epoch `54`, which explains why the legacy preference for `l` is not preserved under the new gate-aware selection rule.",
            "",
            "## Why Binary Gate Is the Official Stage-1 Selection View",
            "- Stage-1 is defined as normal filtering under a recall constraint rather than default-threshold classification.",
            "- The formal ranking therefore follows `Spec@R99.5 > Spec@R99.0 > Prec@R99.0 > PTR@R99.0`, which directly reflects the thesis-facing gate objective.",
            "- Under this rule, source-side classification quality remains informative but cannot replace direct binary-gate checkpoint selection.",
            "",
            "## Key Findings",
            f"- Formal gate-capacity leader: `{leader['model']}`",
            f"- Second control model: `{second['model']}`",
            "- The current formal evidence supports `yolo11m-cls` as the primary backbone for subsequent HN, HardMix, and information-driven sampling experiments.",
            "- The capacity scan therefore rewrites the exploratory `l/s`-centric intuition and establishes a new `m/x`-centric mainline for stage-1.",
        ]
    )
    write_text(report_root / "stage1_gate_capacity_formal_report.md", "\n".join(lines) + "\n")


def build_cls6_report(rows: list[dict[str, Any]], report_root: Path) -> None:
    leader = rows[0]
    lines = [
        "# Stage-1 CLS6 Capacity Formal Report",
        "",
        "## Protocol",
        "- Auxiliary task: `source-side six-class capacity scan`",
        "- Batch size: `24`",
        "- Epochs: `200`",
        "- Checkpoint policy: save every epoch",
        "- Formal ranking: `Accuracy > AUROC > AUPRC`",
        "",
        "## Main Table",
        "See `paper_main/tables/table_stage1_cls6_capacity_main.md`.",
        "",
        "## Final Ranking",
    ]
    for row in rows:
        lines.append(f"{row['rank']}. `{row['model']}`")
    lines.extend(
        [
            "",
            "## Interpretation as a Source-Side Representation Reference",
            f"- `{leader['model']}` is the source-side six-class leader under the formal auxiliary rule.",
            "- The six-class ordering is useful for characterizing representation quality on the source task.",
            "- However, it should be interpreted strictly as an auxiliary reference rather than the official selection criterion for the stage-1 gate.",
        ]
    )
    write_text(report_root / "stage1_cls6_capacity_formal_report.md", "\n".join(lines) + "\n")


def build_crossview_report(cross_rows: list[dict[str, Any]], report_root: Path) -> None:
    lines = [
        "# Stage-1 Cross-View Analysis Report",
        "",
        "## Why CLS6 and Binary Gate Should Not Be Conflated",
        "The formal stage-1 objective is direct binary-gate model selection under a recall-constrained filtering setting. The source-side six-class task serves only as an auxiliary representation view and should not be conflated with the official gate objective.",
        "",
        "## Rank-Mismatch Analysis",
        "See `paper_main/tables/table_stage1_crossview_rank_gap.md`.",
        "",
        "## Implications for Stage-1 Model Selection",
        "- `yolo11x-cls` leads the auxiliary cls6 view, but `yolo11m-cls` remains the direct binary-gate leader.",
        "- `yolo11l-cls`, `yolo11s-cls`, and `yolo11n-cls` keep the same relative order across both views, indicating that the main disagreement is concentrated in the two strongest models.",
        "- This mismatch supports the claim that source-side six-class ranking cannot substitute for direct binary-gate selection.",
        "",
        "## Cross-View Table",
    ]
    for row in cross_rows:
        lines.append(f"- `{row['Model']}`: cls6 rank `{row['Rank in CLS6']}`, gate rank `{row['Rank in Binary Gate']}`, gap `{row['Rank Gap']}`")
    write_text(report_root / "stage1_crossview_analysis_report.md", "\n".join(lines) + "\n")


def build_cls6_best_checkpoint_registry(rows: list[dict[str, Any]], output_root: Path) -> None:
    fieldnames = [
        "task",
        "model",
        "summary_dir",
        "best_epoch",
        "best_checkpoint_path",
        "checkpoint_exists",
        "accuracy",
        "auroc",
        "auprc",
    ]
    csv_rows = [
        {
            "task": "cls6",
            "model": row["model"],
            "summary_dir": row.get("summary_dir", ""),
            "best_epoch": row["best_epoch"],
            "best_checkpoint_path": row["best_checkpoint_path"],
            "checkpoint_exists": str(Path(row["best_checkpoint_path"]).exists()).lower(),
            "accuracy": format_num(row["accuracy"]),
            "auroc": format_num(row["auroc"]),
            "auprc": format_num(row["auprc"]),
        }
        for row in rows
    ]
    write_csv(output_root / "cls6_best_checkpoint_registry.csv", fieldnames, csv_rows)
    write_json(output_root / "cls6_best_checkpoint_registry.json", {"rows": csv_rows})
    lines = [
        "# CLS6 Best Checkpoint Registry",
        "",
        "| task | model | summary_dir | best_epoch | best_checkpoint_path | checkpoint_exists | accuracy | auroc | auprc |",
        "| --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for row in csv_rows:
        lines.append(
            "| {task} | {model} | {summary_dir} | {best_epoch} | {best_checkpoint_path} | {checkpoint_exists} | {accuracy} | {auroc} | {auprc} |".format(
                **row
            )
        )
    write_text(output_root / "cls6_best_checkpoint_registry.md", "\n".join(lines) + "\n")


def augment_cls6_epoch_summary(summary_csv: Path, trainer_results_csv: Path) -> list[dict[str, Any]]:
    epoch_rows = read_csv_rows(summary_csv)
    trainer_rows = read_csv_rows(trainer_results_csv) if trainer_results_csv.exists() else []
    trainer_by_epoch = {int(row["epoch"]): row for row in trainer_rows if row.get("epoch")}
    merged: list[dict[str, Any]] = []
    for row in epoch_rows:
        epoch = int(row["epoch"])
        trainer_row = trainer_by_epoch.get(epoch, {})
        merged.append(
            {
                "epoch": epoch,
                "checkpoint_path": row["checkpoint_path"],
                "Accuracy": format_num(row["accuracy"]),
                "AUROC": format_num(row["auroc"]),
                "AUPRC": format_num(row["auprc"]),
                "top1_acc": format_num(trainer_row.get("metrics/accuracy_top1", "NA")),
            }
        )
    return merged


def build_cls6_appendix_tables_and_figures(materials_root: Path, run_root: Path, appendix_tables: Path, appendix_figures: Path, source_root: Path) -> None:
    write_text(
        source_root / "README.md",
        "# CLS6 Source-Of-Truth\n\n"
        "This directory stores the per-model formal cls6 evidence set copied from the formal materials and trainer-run artifacts.\n",
    )
    for summary_dir in sorted(path for path in materials_root.iterdir() if path.is_dir()):
        model_name = summary_dir.name
        short_name = model_name.replace("_formal", "")
        source_of_truth_dir = source_root / model_name
        previous_files = {
            "results.csv": source_of_truth_dir / "results.csv",
            "stdout.log": source_of_truth_dir / "stdout.log",
            "stderr.log": source_of_truth_dir / "stderr.log",
        }
        ensure_clean_dir(source_of_truth_dir)
        for file_name in [
            "all_checkpoints_index.csv",
            "best_epoch_manifest.json",
            "dataset_inventory.csv",
            "dataset_manifest.json",
            "env_snapshot.json",
            "epoch_cls6_summary.csv",
            "epoch_cls6_summary.json",
            "epoch_cls6_summary.md",
            "pip_freeze.txt",
            "run_manifest.json",
        ]:
            copy_if_exists(summary_dir / file_name, source_of_truth_dir / file_name)

        run_name = summary_dir.name.replace("_formal", "_formal_200ep")
        run_dir = run_root / run_name
        if not run_dir.exists():
            alt_run_dir = run_root / summary_dir.name
            if alt_run_dir.exists():
                run_dir = alt_run_dir
        missing_trainer = False
        for file_name in ["results.csv", "stdout.log", "stderr.log"]:
            primary_src = run_dir / file_name
            fallback_src = previous_files[file_name]
            copied = copy_if_exists(primary_src, source_of_truth_dir / file_name)
            if not copied and fallback_src.exists():
                copied = copy_if_exists(fallback_src, source_of_truth_dir / file_name)
            missing_trainer = missing_trainer or not copied
        for optional_file in ["epoch_metrics.csv", "training_runtime.json", "results.png", "confusion_matrix.png", "confusion_matrix_normalized.png", "args.yaml"]:
            copy_if_exists(run_dir / optional_file, source_of_truth_dir / optional_file)
        if missing_trainer:
            write_text(source_of_truth_dir / "MISSING_TRAINER_ARTIFACTS.md", "# Missing trainer artifacts\n")

        trainer_results_csv = run_dir / "results.csv"
        if not trainer_results_csv.exists():
            trainer_results_csv = previous_files["results.csv"]
        merged_rows = augment_cls6_epoch_summary(summary_dir / "epoch_cls6_summary.csv", trainer_results_csv)
        csv_path = appendix_tables / f"{short_name}_epoch_summary.csv"
        md_path = appendix_tables / f"{short_name}_epoch_summary.md"
        write_csv(csv_path, ["epoch", "checkpoint_path", "Accuracy", "AUROC", "AUPRC", "top1_acc"], merged_rows)

        lines = [
            f"# {short_name.replace('_', '-')} CLS6 Epoch Summary",
            "",
            "| epoch | checkpoint_path | Accuracy | AUROC | AUPRC | top1_acc |",
            "| ---: | --- | ---: | ---: | ---: | ---: |",
        ]
        for row in merged_rows:
            lines.append("| {epoch} | {checkpoint_path} | {Accuracy} | {AUROC} | {AUPRC} | {top1_acc} |".format(**row))
        write_text(md_path, "\n".join(lines) + "\n")

        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        epochs = [int(row["epoch"]) for row in merged_rows]
        metrics = [
            ("Accuracy", "Accuracy", "#355C7D"),
            ("AUROC", "AUROC", "#6C5B7B"),
            ("AUPRC", "AUPRC", "#F67280"),
            ("top1_acc", "Top1 Accuracy", "#2A9D8F"),
        ]
        for ax, (field, title, color) in zip(axes.flat, metrics, strict=True):
            values = [as_float(row[field]) for row in merged_rows]
            ax.plot(epochs, values, color=color, linewidth=1.4)
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.grid(alpha=0.2)
        fig.suptitle(f"{short_name} cls6 dashboard", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        appendix_figures.mkdir(parents=True, exist_ok=True)
        fig.savefig(appendix_figures / f"{short_name}_dashboard.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def build_source_of_truth_gate(materials_root: Path, source_root: Path) -> None:
    for summary_dir in sorted(path for path in materials_root.iterdir() if path.is_dir()):
        source_dir = source_root / summary_dir.name
        ensure_clean_dir(source_dir)
        for file_name in [
            "all_checkpoints_index.csv",
            "best_epoch_manifest.json",
            "dataset_manifest.json",
            "epoch_gate_summary.csv",
            "epoch_gate_summary.json",
            "epoch_gate_summary.md",
            "run_manifest.json",
        ]:
            copy_if_exists(summary_dir / file_name, source_dir / file_name)
        write_text(
            source_dir / "MISSING_TRAINER_ARTIFACTS.md",
            "# Missing trainer artifacts\n\nTrainer-side `results.csv`, `stdout.log`, and `stderr.log` are not available on this analysis machine.\n",
        )


def augment_gate_epoch_summary(summary_csv: Path, trainer_results_csv: Path) -> list[dict[str, Any]]:
    epoch_rows = read_csv_rows(summary_csv)
    trainer_rows = read_csv_rows(trainer_results_csv) if trainer_results_csv.exists() else []
    trainer_by_epoch = {int(row["epoch"]): row for row in trainer_rows if row.get("epoch")}
    merged: list[dict[str, Any]] = []
    for row in epoch_rows:
        epoch = int(row["epoch"])
        trainer_row = trainer_by_epoch.get(epoch, {})
        merged.append(
            {
                "epoch": epoch,
                "checkpoint_path": row["checkpoint_path"],
                "temperature_T": format_num(row["temperature_T"] if "temperature_T" in row else row.get("temperature_t", row.get("temperature", "NA"))),
                "tau_r995": format_num(row["tau_r995"]),
                "tau_r990": format_num(row["tau_r990"]),
                "Spec@R99.5": format_num(row["spec_at_r995"] if "spec_at_r995" in row else row["Spec@R99.5"]),
                "Spec@R99.0": format_num(row["spec_at_r990"] if "spec_at_r990" in row else row["Spec@R99.0"]),
                "Prec@R99.0": format_num(row["prec_at_r990"] if "prec_at_r990" in row else row["Prec@R99.0"]),
                "PTR@R99.0": format_num(row["ptr_at_r990"] if "ptr_at_r990" in row else row["PTR@R99.0"]),
                "TN@R99.5": row["tn_at_r995"] if "tn_at_r995" in row else row["TN@R99.5"],
                "FN@R99.5": row["fn_at_r995"] if "fn_at_r995" in row else row["FN@R99.5"],
                "TN@R99.0": row["tn_at_r990"] if "tn_at_r990" in row else row["TN@R99.0"],
                "FN@R99.0": row["fn_at_r990"] if "fn_at_r990" in row else row["FN@R99.0"],
                "top1_acc": format_num(trainer_row.get("metrics/accuracy_top1", "NA")),
            }
        )
    return merged


def build_gate_appendix_tables_and_figures(materials_root: Path, run_root: Path, appendix_tables: Path, appendix_figures: Path, source_root: Path) -> bool:
    write_text(
        source_root / "README.md",
        "# Direct Binary Gate Source-Of-Truth\n\n"
        "This directory stores the per-model formal binary-gate evidence set copied from the formal materials and trainer-run artifacts.\n",
    )
    trainer_available = True
    for summary_dir in sorted(path for path in materials_root.iterdir() if path.is_dir()):
        model_name = summary_dir.name
        short_name = model_name.replace("_gate2_formal", "_gate")
        source_dir = source_root / model_name
        previous_files = {
            "results.csv": source_dir / "results.csv",
            "stdout.log": source_dir / "stdout.log",
            "stderr.log": source_dir / "stderr.log",
        }
        ensure_clean_dir(source_dir)
        for file_name in [
            "all_checkpoints_index.csv",
            "best_epoch_manifest.json",
            "dataset_manifest.json",
            "epoch_gate_summary.csv",
            "epoch_gate_summary.json",
            "epoch_gate_summary.md",
            "run_manifest.json",
        ]:
            copy_if_exists(summary_dir / file_name, source_dir / file_name)

        run_name = model_name.replace("_formal", "_formal_200ep")
        run_dir = run_root / run_name
        missing_trainer = False
        for file_name in ["results.csv", "stdout.log", "stderr.log"]:
            primary_src = run_dir / file_name
            fallback_src = previous_files[file_name]
            copied = copy_if_exists(primary_src, source_dir / file_name)
            if not copied and fallback_src.exists():
                copied = copy_if_exists(fallback_src, source_dir / file_name)
            missing_trainer = missing_trainer or not copied
        for optional_file in ["epoch_metrics.csv", "training_runtime.json", "results.png", "confusion_matrix.png", "confusion_matrix_normalized.png", "args.yaml"]:
            copy_if_exists(run_dir / optional_file, source_dir / optional_file)
        if missing_trainer:
            trainer_available = False
            write_text(
                source_dir / "MISSING_TRAINER_ARTIFACTS.md",
                "# Missing trainer artifacts\n\nTrainer-side `results.csv`, `stdout.log`, and `stderr.log` are not available on this analysis machine.\n",
            )

        trainer_results_csv = run_dir / "results.csv"
        if not trainer_results_csv.exists():
            trainer_results_csv = previous_files["results.csv"]
        merged_rows = augment_gate_epoch_summary(summary_dir / "epoch_gate_summary.csv", trainer_results_csv)
        csv_path = appendix_tables / f"{short_name}_epoch_summary.csv"
        md_path = appendix_tables / f"{short_name}_epoch_summary.md"
        fieldnames = [
            "epoch",
            "checkpoint_path",
            "temperature_T",
            "tau_r995",
            "tau_r990",
            "Spec@R99.5",
            "Spec@R99.0",
            "Prec@R99.0",
            "PTR@R99.0",
            "TN@R99.5",
            "FN@R99.5",
            "TN@R99.0",
            "FN@R99.0",
            "top1_acc",
        ]
        write_csv(csv_path, fieldnames, merged_rows)
        lines = [
            f"# {short_name.replace('_', '-')} Gate Epoch Summary",
            "",
            "| epoch | checkpoint_path | temperature_T | tau_r995 | tau_r990 | Spec@R99.5 | Spec@R99.0 | Prec@R99.0 | PTR@R99.0 | TN@R99.5 | FN@R99.5 | TN@R99.0 | FN@R99.0 | top1_acc |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in merged_rows:
            lines.append(
                f"| {row['epoch']} | {row['checkpoint_path']} | {row['temperature_T']} | {row['tau_r995']} | {row['tau_r990']} | "
                f"{row['Spec@R99.5']} | {row['Spec@R99.0']} | {row['Prec@R99.0']} | {row['PTR@R99.0']} | "
                f"{row['TN@R99.5']} | {row['FN@R99.5']} | {row['TN@R99.0']} | {row['FN@R99.0']} | {row['top1_acc']} |"
            )
        write_text(md_path, "\n".join(lines) + "\n")

        fig, axes = plt.subplots(3, 2, figsize=(10, 9))
        epochs = [int(row["epoch"]) for row in merged_rows]
        metrics = [
            ("Spec@R99.5", "Spec@R99.5", "#355C7D"),
            ("Spec@R99.0", "Spec@R99.0", "#6C5B7B"),
            ("Prec@R99.0", "Prec@R99.0", "#F67280"),
            ("PTR@R99.0", "PTR@R99.0", "#C06C84"),
            ("top1_acc", "Top1 Accuracy", "#2A9D8F"),
        ]
        for ax, (field, title, color) in zip(axes.flat, metrics + [("empty", "", "#000000")], strict=True):
            if field == "empty":
                ax.axis("off")
                continue
            values = [as_float(row[field]) for row in merged_rows]
            ax.plot(epochs, values, color=color, linewidth=1.4)
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.grid(alpha=0.2)
        fig.suptitle(f"{short_name} gate dashboard", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        appendix_figures.mkdir(parents=True, exist_ok=True)
        fig.savefig(appendix_figures / f"{short_name}_dashboard.png", dpi=220, bbox_inches="tight")
        plt.close(fig)
    return trainer_available


def build_gate_appendix_top3(materials_root: Path, appendix_tables: Path) -> None:
    for summary_dir in sorted(path for path in materials_root.iterdir() if path.is_dir()):
        model_name = summary_dir.name.replace("_gate2_formal", "")
        rows = read_csv_rows(summary_dir / "epoch_gate_summary.csv")
        ranked = sorted(rows, key=gate_rank_key, reverse=True)[:3]
        csv_rows = [
            {
                "Rank": index,
                "Epoch": row["epoch"],
                "Spec@R99.5": format_num(row["spec_at_r995"]),
                "Spec@R99.0": format_num(row["spec_at_r990"]),
                "Prec@R99.0": format_num(row["prec_at_r990"]),
                "PTR@R99.0": format_num(row["ptr_at_r990"]),
                "tau_R99.5": format_num(row["tau_r995"]),
                "tau_R99.0": format_num(row["tau_r990"]),
            }
            for index, row in enumerate(ranked, start=1)
        ]
        write_csv(
            appendix_tables / f"{model_name}_gate_top3_checkpoints.csv",
            ["Rank", "Epoch", "Spec@R99.5", "Spec@R99.0", "Prec@R99.0", "PTR@R99.0", "tau_R99.5", "tau_R99.0"],
            csv_rows,
        )
        lines = [
            f"# {model_name} Gate Top-3 Checkpoints",
            "",
            "| Rank | Epoch | Spec@R99.5 | Spec@R99.0 | Prec@R99.0 | PTR@R99.0 | tau_R99.5 | tau_R99.0 |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in csv_rows:
            lines.append(
                "| {rank} | {epoch} | {spec995} | {spec990} | {prec990} | {ptr990} | {tau995} | {tau990} |".format(
                    rank=row["Rank"],
                    epoch=row["Epoch"],
                    spec995=row["Spec@R99.5"],
                    spec990=row["Spec@R99.0"],
                    prec990=row["Prec@R99.0"],
                    ptr990=row["PTR@R99.0"],
                    tau995=row["tau_R99.5"],
                    tau990=row["tau_R99.0"],
                )
            )
        write_text(appendix_tables / f"{model_name}_gate_top3_checkpoints.md", "\n".join(lines) + "\n")


def build_cls6_bar(rows: list[dict[str, Any]], output_path: Path) -> None:
    models = [normalize_model_key(str(row["model"])) for row in rows]
    metrics = [("accuracy", "Accuracy", "#355C7D"), ("auroc", "AUROC", "#6C5B7B"), ("auprc", "AUPRC", "#F67280")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    for ax, (field, title, color) in zip(axes, metrics, strict=True):
        values = [as_float(row[field]) for row in rows]
        ax.bar(models, values, color=color, alpha=0.9)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Formal six-class source capacity scan", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_crossview_figure(cross_rows: list[dict[str, Any]], output_path: Path) -> None:
    models = [row["Model"].replace("-cls", "") for row in cross_rows]
    cls6_rank = [int(row["Rank in CLS6"]) for row in cross_rows]
    gate_rank = [int(row["Rank in Binary Gate"]) for row in cross_rows]
    y = list(range(len(models)))
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for idx, model in enumerate(models):
        ax.plot([cls6_rank[idx], gate_rank[idx]], [idx, idx], color="#6C5B7B", linewidth=2)
        ax.scatter(cls6_rank[idx], idx, color="#F67280", s=60, label="CLS6 rank" if idx == 0 else "")
        ax.scatter(gate_rank[idx], idx, color="#355C7D", s=60, label="Binary gate rank" if idx == 0 else "")
    ax.set_yticks(y, models)
    ax.invert_yaxis()
    ax.set_xlabel("Rank (lower is better)")
    ax.set_title("Cross-view rank gap: cls6 vs direct binary gate")
    ax.grid(axis="x", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_gate_top1_mismatch_figure(
    materials_root: Path,
    run_root: Path,
    gate_top1_rows: list[dict[str, Any]],
    output_path: Path,
) -> bool:
    selected_models = ["yolo11m", "yolo11l"]
    top1_by_model = {row["model_key"]: row for row in gate_top1_rows}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    trainer_curve_ready = True
    for ax, model_key in zip(axes, selected_models, strict=True):
        summary_csv = materials_root / f"{model_key}_gate2_formal" / "epoch_gate_summary.csv"
        rows = read_csv_rows(summary_csv)
        epochs = [int(row["epoch"]) for row in rows]
        spec_values = [as_float(row["spec_at_r995"]) for row in rows]
        gate_row = top1_by_model[model_key]
        gate_epoch = int(gate_row["gate_best_epoch"])
        top1_epoch = int(gate_row["top1_best_epoch"])
        top1_value = as_float(gate_row["top1_best_value"])
        gate_spec = as_float(gate_row["gate_best_spec_at_r995"])

        ax.plot(epochs, spec_values, color="#355C7D", linewidth=1.6, label="Spec@R99.5")
        ax.axvline(gate_epoch, color="#355C7D", linestyle="--", linewidth=1.2, alpha=0.8, label="Gate-best epoch")
        ax.axvline(top1_epoch, color="#F67280", linestyle=":", linewidth=1.4, alpha=0.9, label="Top1-best epoch")
        ax.scatter([gate_epoch], [gate_spec], color="#355C7D", s=45, zorder=5)
        ax.set_title(f"{model_key}-cls")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Spec@R99.5")
        ax.grid(alpha=0.2)

        twin = ax.twinx()
        run_dir = run_root / f"{model_key}_gate2_formal_200ep"
        trainer_results_csv = run_dir / "results.csv"
        if trainer_results_csv.exists():
            trainer_rows = read_csv_rows(trainer_results_csv)
            trainer_by_epoch = {int(row["epoch"]): row for row in trainer_rows if row.get("epoch")}
            top1_curve = [as_float(trainer_by_epoch.get(epoch, {}).get("metrics/accuracy_top1", "nan")) for epoch in epochs]
            twin.plot(epochs, top1_curve, color="#F67280", linewidth=1.3, alpha=0.85, label="Top1 Accuracy")
        else:
            trainer_curve_ready = False
        twin.scatter([top1_epoch], [top1_value], color="#F67280", marker="D", s=48, zorder=6)
        twin.set_ylabel("Top1 Accuracy")
        twin.set_ylim(0.88, 0.95)
        twin.annotate(
            f"top1={top1_value:.4f}",
            xy=(top1_epoch, top1_value),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
            color="#F67280",
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Mismatch between trainer top1-selected and gate-selected checkpoints", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return trainer_curve_ready


def update_bundle_metadata(output_root: Path, gate_trainer_curves_available: bool) -> None:
    gate_curve_note = (
        "The gate top1-vs-spec figure uses full trainer top1 trajectories copied from the formal gate run archives."
        if gate_trainer_curves_available
        else "The gate top1-vs-spec figure marks trainer top1-best points because full gate trainer trajectories are not available on the analysis machine."
    )
    write_json(
        output_root / "bundle_manifest.json",
        {
            "binary_gate": "complete",
            "cls6": "complete",
            "cross_view": "complete",
            "selection_rule_gate": ["Spec@R99.5", "Spec@R99.0", "Prec@R99.0", "PTR@R99.0"],
            "selection_rule_cls6": ["Accuracy", "AUROC", "AUPRC"],
            "trainer_curves_available": (
                [
                    "cls6:yolo11n",
                    "cls6:yolo11s",
                    "cls6:yolo11m",
                    "cls6:yolo11l",
                    "cls6:yolo11x",
                    "gate:yolo11n",
                    "gate:yolo11s",
                    "gate:yolo11m",
                    "gate:yolo11l",
                    "gate:yolo11x",
                ]
                if gate_trainer_curves_available
                else ["cls6:yolo11n", "cls6:yolo11s", "cls6:yolo11m", "cls6:yolo11l", "cls6:yolo11x"]
            ),
            "trainer_curves_missing": [] if gate_trainer_curves_available else ["gate:yolo11n", "gate:yolo11s", "gate:yolo11m", "gate:yolo11l", "gate:yolo11x"],
            "gate_top1_curve_note": gate_curve_note,
        },
    )
    write_text(
        output_root / "README.md",
        "# Stage-1 Formal Capacity Scan Bundle\n\n"
        "- `paper_main/`: thesis-facing main tables, figures and reports.\n"
        "- `appendix/`: detailed appendix tables, figures and protocol.\n"
        "- `source_of_truth/`: formal evidence layer copied from stage1_formal materials/results.\n\n"
        "Current status:\n"
        "- binary gate: complete\n"
        "- cls6: complete\n"
        "- cross-view: complete\n"
        f"- note: {gate_curve_note}\n",
    )


def build_appendix_inventory_report(appendix_reports: Path) -> None:
    write_text(
        appendix_reports / "capacity_scan_appendix_inventory.md",
        "# Capacity-Scan Appendix Inventory\n\n"
        "This appendix directory stores the detailed per-model tables and figures that support the main-text capacity-scan analysis.\n\n"
        "Included materials:\n"
        "- per-model binary-gate full epoch summaries\n"
        "- per-model binary-gate top-3 checkpoint tables\n"
        "- per-model six-class full epoch summaries\n"
        "- all-model gate metric curves\n"
        "- per-model gate dashboards\n"
        "- per-model cls6 dashboards\n"
        "- frozen formal protocol table\n",
    )


def normalize_yes_no(value: str) -> str:
    return "Yes" if str(value).strip().lower() == "yes" else "No"
def main() -> None:
    args = parse_args()
    gate_summary_csv = Path(args.gate_summary_csv).resolve()
    gate_top1_csv = Path(args.gate_top1_csv).resolve()
    gate_materials_root = Path(args.gate_materials_root).resolve()
    gate_run_root = Path(args.gate_run_root).resolve() if str(args.gate_run_root).strip() else Path()
    cls6_summary_csv = Path(args.cls6_summary_csv).resolve()
    cls6_materials_root = Path(args.cls6_materials_root).resolve()
    cls6_run_root = Path(args.cls6_run_root).resolve()
    registry_csv = Path(args.registry_csv).resolve()
    protocol_md = Path(args.protocol_md).resolve()
    output_root = Path(args.output_root).resolve()

    paper_tables = output_root / "paper_main" / "tables"
    paper_figures = output_root / "paper_main" / "figures"
    paper_reports = output_root / "paper_main" / "reports"
    appendix_tables = output_root / "appendix" / "tables"
    appendix_figures = output_root / "appendix" / "figures"
    source_root = output_root / "source_of_truth"
    source_gate = source_root / "gate_capacity"
    source_cls6 = source_root / "cls6_capacity"
    source_manifests = source_root / "manifests"

    gate_rows = load_gate_rows(gate_summary_csv)
    gate_top1_rows = load_gate_top1_rows(gate_top1_csv)
    gate_summary_by_model = {normalize_model_key(str(row["model"])): row for row in gate_rows}
    for row in gate_top1_rows:
        gate_row = gate_summary_by_model[row["model_key"]]
        row["gate_best_spec_at_r995"] = gate_row["spec_at_r995"]
        row["gate_best_spec_at_r990"] = gate_row["spec_at_r990"]
        row["same_as_gate_best"] = normalize_yes_no(str(row["same_as_gate_best"]))
    cls6_rows = load_cls6_rows(cls6_summary_csv)

    build_gate_main_tables(gate_rows, paper_tables)
    build_gate_top1_table(gate_top1_rows, paper_tables)
    build_cls6_main_tables(cls6_rows, paper_tables)
    cross_rows = build_crossview_tables(gate_rows, cls6_rows, paper_tables)

    build_gate_report(gate_rows, gate_top1_rows, paper_reports)
    build_cls6_report(cls6_rows, paper_reports)
    build_crossview_report(cross_rows, paper_reports)

    gate_trainer_available = build_gate_appendix_tables_and_figures(
        gate_materials_root,
        gate_run_root,
        appendix_tables,
        appendix_figures,
        source_gate,
    )
    build_cls6_appendix_tables_and_figures(cls6_materials_root, cls6_run_root, appendix_tables, appendix_figures, source_cls6)
    build_gate_appendix_top3(gate_materials_root, appendix_tables)

    copy_if_exists(protocol_md, source_manifests / "stage1_formal_protocol.md")
    copy_if_exists(registry_csv, source_manifests / "formal_capacity_scan_registry.csv")
    copy_if_exists(Path("research/results/stage1_formal/formal_cleanup_manifest.json").resolve(), source_manifests / "formal_cleanup_manifest.json")
    copy_if_exists(Path("research/results/stage1_formal/formal_cleanup_manifest.md").resolve(), source_manifests / "formal_cleanup_manifest.md")
    copy_if_exists(Path("research/results/stage1_formal/best_checkpoint_registry.csv").resolve(), source_manifests / "best_checkpoint_registry.csv")
    copy_if_exists(Path("research/results/stage1_formal/best_checkpoint_registry.json").resolve(), source_manifests / "best_checkpoint_registry.json")
    copy_if_exists(Path("research/results/stage1_formal/best_checkpoint_registry.md").resolve(), source_manifests / "best_checkpoint_registry.md")
    copy_if_exists(Path("research/materials/stage1_formal/manifests/val_cal_op_split.csv").resolve(), source_manifests / "val_cal_op_split.csv")
    build_cls6_best_checkpoint_registry(cls6_rows, source_manifests)

    build_cls6_bar(cls6_rows, paper_figures / "fig_stage1_cls6_capacity_bar.png")
    build_crossview_figure(cross_rows, paper_figures / "fig_stage1_crossview_rank_gap.png")
    gate_top1_curve_ready = build_gate_top1_mismatch_figure(
        gate_materials_root,
        gate_run_root,
        gate_top1_rows,
        paper_figures / "fig_stage1_gate_top1_vs_spec_dualaxis.png",
    )
    build_appendix_inventory_report(output_root / "appendix" / "reports")
    update_bundle_metadata(output_root, gate_trainer_available and gate_top1_curve_ready)


if __name__ == "__main__":
    main()
