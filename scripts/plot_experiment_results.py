from __future__ import annotations

from pathlib import Path
import argparse
import csv
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN_DIR = ROOT / "results" / "leaderboard"
DEFAULT_OUT_DIR = ROOT / "results" / "leaderboard"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot leaderboard CSV files produced by collect_experiment_results.py.")
    parser.add_argument("--in-dir", type=Path, default=DEFAULT_IN_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input CSV: {path}")
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def to_float(value: str | None, default: float = 0.0) -> float:
    try:
        if value in {None, ""}:
            return default
        return float(value)
    except ValueError:
        return default


def short_labels(labels: list[str], width: int = 28) -> list[str]:
    return [label if len(label) <= width else label[: width - 1] + "…" for label in labels]


def save_bar(path: Path, rows: list[dict[str, str]], metric: str, title: str, ylabel: str, top_k: int) -> None:
    rows = sorted(rows, key=lambda row: to_float(row.get(metric)), reverse=True)[:top_k]
    labels = [row.get("run_name", "") for row in rows]
    values = [to_float(row.get(metric)) for row in rows]
    colors = [family_color(row.get("family", "")) for row in rows]
    fig_height = max(4.0, 0.34 * len(rows))
    plt.figure(figsize=(11, fig_height))
    plt.barh(short_labels(labels), values, color=colors)
    plt.gca().invert_yaxis()
    plt.xlabel(ylabel)
    plt.title(title)
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def family_color(family: str) -> str:
    return {
        "convnext": "#4477aa",
        "vit": "#228833",
        "deit": "#cc6677",
    }.get(family, "#888888")


def save_best_by_family(path: Path, rows: list[dict[str, str]]) -> None:
    rows = sorted(rows, key=lambda row: to_float(row.get("best_valid_roc_auc")), reverse=True)
    labels = [row.get("family", "") for row in rows]
    auc = [to_float(row.get("best_valid_roc_auc")) for row in rows]
    f1 = [to_float(row.get("best_valid_f1")) for row in rows]
    x = list(range(len(rows)))
    width = 0.36
    plt.figure(figsize=(8, 4.5))
    plt.bar([i - width / 2 for i in x], auc, width=width, label="ROC-AUC", color="#4477aa")
    plt.bar([i + width / 2 for i in x], f1, width=width, label="F1", color="#ee7733")
    plt.xticks(x, labels)
    plt.ylim(0, max([0.05] + auc + f1) * 1.12)
    plt.ylabel("metric")
    plt.title("Best Run By Family")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    for idx, row in enumerate(rows):
        plt.text(idx, max(auc[idx], f1[idx]) * 1.02, row.get("run_name", ""), ha="center", va="bottom", fontsize=8, rotation=20)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_by_type_heatmap(path: Path, rows: list[dict[str, str]], top_k: int) -> None:
    if not rows:
        plt.figure(figsize=(7, 3))
        plt.text(0.5, 0.5, "No by-type metrics", ha="center", va="center")
        plt.axis("off")
        plt.savefig(path, dpi=180)
        plt.close()
        return
    run_scores = {}
    for row in rows:
        run = row.get("run_name", "")
        run_scores[run] = max(run_scores.get(run, 0.0), to_float(row.get("roc_auc")))
    run_names = [name for name, _ in sorted(run_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]]
    neg_types = sorted({row.get("negative_type", "") for row in rows if row.get("negative_type")})
    values = [[math.nan for _ in neg_types] for _ in run_names]
    run_index = {name: idx for idx, name in enumerate(run_names)}
    type_index = {name: idx for idx, name in enumerate(neg_types)}
    for row in rows:
        run = row.get("run_name", "")
        neg_type = row.get("negative_type", "")
        if run in run_index and neg_type in type_index:
            values[run_index[run]][type_index[neg_type]] = to_float(row.get("roc_auc"), math.nan)
    plt.figure(figsize=(max(8, 1.2 * len(neg_types)), max(4, 0.35 * len(run_names))))
    image = plt.imshow(values, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    plt.colorbar(image, label="ROC-AUC")
    plt.xticks(range(len(neg_types)), short_labels(neg_types, width=24), rotation=35, ha="right")
    plt.yticks(range(len(run_names)), short_labels(run_names, width=30))
    plt.title("By-Negative-Type ROC-AUC")
    for i, row in enumerate(values):
        for j, value in enumerate(row):
            if not math.isnan(value):
                plt.text(j, i, f"{value:.2f}", ha="center", va="center", color="white" if value < 0.55 else "black", fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    runs = read_csv(args.in_dir / "runs_summary.csv")
    by_type = read_csv(args.in_dir / "by_type_metrics.csv")
    best = read_csv(args.in_dir / "best_by_family.csv")
    save_bar(args.out_dir / "auc_by_run.png", runs, "best_valid_roc_auc", "ROC-AUC By Run", "best valid ROC-AUC", args.top_k)
    save_bar(args.out_dir / "f1_by_run.png", runs, "best_valid_f1", "F1 By Run", "best valid F1", args.top_k)
    save_best_by_family(args.out_dir / "best_by_family.png", best)
    save_by_type_heatmap(args.out_dir / "by_type_auc_heatmap.png", by_type, args.top_k)
    print(f"Wrote plots to {args.out_dir}")


if __name__ == "__main__":
    main()
