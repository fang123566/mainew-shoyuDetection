"""生成 Web 评估分析页的真实图表资产。

本脚本以 Web 默认权重和 YOLO 数据集配置为输入，生成适合课程答辩和报告引用的
Nature 风格高清图表。默认输出到 ``outputs/analysis/``，并同步精选资产到
``new-shoyuDetection/static/analysis/``，保证 Web 评估页 clone 后也能直接展示。
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "datasets" / "sign_language" / "data.yaml"
DEFAULT_WEIGHTS = PROJECT_ROOT / "weights" / "yolov11_best.pt"
DEFAULT_VAL_CSV = PROJECT_ROOT / "outputs" / "csv" / "val_results.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "analysis"
STATIC_DIR = PROJECT_ROOT / "new-shoyuDetection" / "static" / "analysis"
REPORT_SUMMARY = PROJECT_ROOT / "configs" / "report_summary.json"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 Web 评估分析资产")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="YOLO data.yaml 路径")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="模型权重路径")
    parser.add_argument("--metrics-csv", type=Path, default=DEFAULT_VAL_CSV, help="验证指标 CSV")
    parser.add_argument("--conf", type=float, default=0.25, help="混淆矩阵推理置信度阈值")
    parser.add_argument("--iou", type=float, default=0.5, help="混淆矩阵匹配 IoU 阈值")
    parser.add_argument("--max-examples", type=int, default=40, help="最多导出多少个问题样例")
    parser.add_argument("--skip-inference", action="store_true", help="跳过权重推理，仅生成数据集和指标图")
    parser.add_argument("--static-dir", type=Path, default=STATIC_DIR, help="Web 静态评估资产目录")
    return parser.parse_args()


def load_yaml(data_path: Path) -> dict[str, Any]:
    with data_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def normalize_names(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    return {index: str(value) for index, value in enumerate(names or [])}


def resolve_split(data_path: Path, data: dict[str, Any], split: str) -> Path | None:
    raw = data.get(split)
    if raw is None:
        return None
    split_path = Path(str(raw))
    if split_path.is_absolute():
        return split_path
    raw_base = data.get("path")
    base = Path(str(raw_base)) if raw_base is not None else data_path.parent
    if not base.is_absolute():
        # 本项目的 data.yaml 注明需要从仓库根目录运行，Ultralytics 也按当前项目目录解析 path。
        base = PROJECT_ROOT / base
    return (base / split_path).resolve()


def infer_label_dir(image_dir: Path) -> Path:
    parts = list(image_dir.parts)
    if "images" in parts:
        index = len(parts) - 1 - parts[::-1].index("images")
        parts[index] = "labels"
        return Path(*parts)
    return image_dir.parent / "labels" / image_dir.name


def list_images(image_dir: Path | None) -> list[Path]:
    if image_dir is None or not image_dir.exists():
        return []
    return sorted(path for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def read_yolo_labels(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    if not label_path.exists():
        return []
    rows: list[tuple[int, float, float, float, float]] = []
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            rows.append((int(float(parts[0])), *(float(x) for x in parts[1:])))
        except ValueError:
            continue
    return rows


def count_split(image_dir: Path | None, names: dict[int, str]) -> dict[str, Any]:
    if image_dir is None:
        return {"images": 0, "labels": 0, "boxes": 0, "class_counts": {}}
    label_dir = infer_label_dir(image_dir)
    images = list_images(image_dir)
    labels = [path for path in label_dir.rglob("*.txt") if path.name != "classes.txt"] if label_dir.exists() else []
    counts: Counter[int] = Counter()
    boxes = 0
    for label_path in labels:
        for row in read_yolo_labels(label_path):
            class_id = row[0]
            if class_id in names:
                counts[class_id] += 1
                boxes += 1
    return {
        "images": len(images),
        "labels": len(labels),
        "boxes": boxes,
        "class_counts": {names[key]: counts.get(key, 0) for key in sorted(names)},
    }


def read_metrics(metrics_csv: Path) -> dict[str, float]:
    if metrics_csv.exists():
        with metrics_csv.open("r", encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
        if rows:
            row = rows[-1]
            return {
                "precision": float(row.get("precision", 0) or 0),
                "recall": float(row.get("recall", 0) or 0),
                "map50": float(row.get("map50", 0) or 0),
                "map50_95": float(row.get("map50_95", 0) or 0),
            }
    if REPORT_SUMMARY.exists():
        summary = json.loads(REPORT_SUMMARY.read_text(encoding="utf-8"))
        return {key: float(value) for key, value in (summary.get("metrics") or {}).items()}
    return {"precision": 0.0, "recall": 0.0, "map50": 0.0, "map50_95": 0.0}


def configure_plot_style() -> None:
    import matplotlib as mpl
    import seaborn as sns

    sns.set_theme(
        context="paper",
        style="whitegrid",
        palette="colorblind",
        font="DejaVu Sans",
        rc={
            "figure.dpi": 160,
            "savefig.dpi": 480,
            "axes.edgecolor": "#2f3542",
            "axes.linewidth": 0.8,
            "axes.labelcolor": "#1f2937",
            "axes.titlecolor": "#111827",
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "grid.color": "#e5e7eb",
            "grid.linewidth": 0.55,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )
    mpl.rcParams["figure.facecolor"] = "white"
    mpl.rcParams["axes.facecolor"] = "white"


def save_figure(fig, output_name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{output_name}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_DIR / f"{output_name}.svg", bbox_inches="tight", facecolor="white")


def draw_class_distribution(summary: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    train_counts = summary["splits"].get("train", {}).get("class_counts", {})
    val_counts = summary["splits"].get("val", {}).get("class_counts", {})
    labels = list(train_counts.keys())
    values_train = [train_counts[label] for label in labels]
    values_val = [val_counts.get(label, 0) for label in labels]

    fig, ax = plt.subplots(figsize=(15.6, 6.2))
    x = np.arange(len(labels))
    ax.bar(x, values_train, width=0.72, color="#4c78a8", label="Train boxes")
    ax.bar(x, values_val, width=0.72, bottom=values_train, color="#72b7b2", label="Val boxes")
    ax.set_title("Class Distribution of Sign-Language Detection Dataset", fontsize=13, pad=12, weight="bold")
    ax.set_ylabel("Annotated boxes")
    ax.set_xlabel("Gesture class")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7.5)
    ax.legend(ncol=2, loc="upper right")
    ax.margins(x=0.01)
    sns.despine(ax=ax, left=False, bottom=False)
    fig.tight_layout()
    save_figure(fig, "class_distribution")
    plt.close(fig)


def draw_top_bottom(summary: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    train_counts = summary["splits"].get("train", {}).get("class_counts", {})
    sorted_counts = sorted(train_counts.items(), key=lambda item: item[1])
    low = sorted_counts[:8]
    top = sorted_counts[-8:][::-1]
    labels = [item[0] for item in low] + [item[0] for item in top]
    values = [item[1] for item in low] + [item[1] for item in top]
    group = ["Fewer samples"] * len(low) + ["More samples"] * len(top)

    fig, ax = plt.subplots(figsize=(9.8, 6.4))
    sns.barplot(x=values, y=labels, hue=group, dodge=False, palette=["#e45756", "#4c78a8"], ax=ax)
    ax.set_title("Class Balance: Least and Most Represented Training Classes", fontsize=13, pad=12, weight="bold")
    ax.set_xlabel("Annotated boxes in training split")
    ax.set_ylabel("")
    ax.legend(loc="lower right")
    for container in ax.containers:
        ax.bar_label(container, padding=3, fontsize=8)
    sns.despine(ax=ax, left=False, bottom=False)
    fig.tight_layout()
    save_figure(fig, "class_balance_top_bottom")
    plt.close(fig)


def draw_metrics(metrics: dict[str, float]) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    labels = ["Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95"]
    values = [metrics.get("precision", 0), metrics.get("recall", 0), metrics.get("map50", 0), metrics.get("map50_95", 0)]
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    colors = ["#4c78a8", "#72b7b2", "#54a24b", "#f58518"]
    sns.barplot(x=labels, y=values, palette=colors, ax=ax, hue=labels, legend=False)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.set_title("Validation Performance of YOLOv11 Sign-Language Detector", fontsize=13, pad=12, weight="bold")
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    for patch, value in zip(ax.patches, values):
        ax.text(patch.get_x() + patch.get_width() / 2, value + 0.025, f"{value:.2%}", ha="center", va="bottom", fontsize=9, weight="bold")
    sns.despine(ax=ax, left=False, bottom=False)
    fig.tight_layout()
    save_figure(fig, "metrics_summary")
    plt.close(fig)


def label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    if "images" in parts:
        index = len(parts) - 1 - parts[::-1].index("images")
        parts[index] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def yolo_to_xyxy(row: tuple[int, float, float, float, float], width: int, height: int) -> tuple[int, np.ndarray]:
    class_id, xc, yc, bw, bh = row
    x1 = (xc - bw / 2) * width
    y1 = (yc - bh / 2) * height
    x2 = (xc + bw / 2) * width
    y2 = (yc + bh / 2) * height
    return class_id, np.array([x1, y1, x2, y2], dtype=float)


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def build_confusion(data_path: Path, data: dict[str, Any], names: dict[int, str], weights: Path, conf: float, iou_threshold: float, max_examples: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
    try:
        import cv2
        from ultralytics import YOLO
    except Exception as exc:
        print(f"[WARN] 推理依赖不可用，跳过混淆矩阵: {exc}")
        return np.zeros((len(names), len(names)), dtype=int), []

    val_dir = resolve_split(data_path, data, "val")
    images = list_images(val_dir)
    if not images or not weights.exists():
        return np.zeros((len(names), len(names)), dtype=int), []

    model = YOLO(str(weights), task="detect")
    matrix = np.zeros((len(names), len(names)), dtype=int)
    examples: list[dict[str, Any]] = []

    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        gt_rows = read_yolo_labels(label_path_for_image(image_path))
        gt_boxes = [yolo_to_xyxy(row, width, height) for row in gt_rows if row[0] in names]
        result = model.predict(source=str(image_path), conf=conf, iou=iou_threshold, verbose=False)[0]
        pred_boxes: list[tuple[int, float, np.ndarray]] = []
        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().numpy()
            cls = result.boxes.cls.cpu().numpy().astype(int)
            scores = result.boxes.conf.cpu().numpy()
            pred_boxes = [(int(c), float(s), np.array(box, dtype=float)) for c, s, box in zip(cls, scores, xyxy) if int(c) in names]

        used: set[int] = set()
        for gt_class, gt_box in gt_boxes:
            best_idx = -1
            best_iou = 0.0
            best_score = 0.0
            for idx, (pred_class, score, pred_box) in enumerate(pred_boxes):
                if idx in used:
                    continue
                iou = box_iou(gt_box, pred_box)
                if iou > best_iou:
                    best_idx = idx
                    best_iou = iou
                    best_score = score
            if best_idx >= 0 and best_iou >= iou_threshold:
                pred_class = pred_boxes[best_idx][0]
                matrix[gt_class, pred_class] += 1
                used.add(best_idx)
                if pred_class != gt_class and len(examples) < max_examples:
                    examples.append({
                        "file": image_path.name,
                        "reason": f"误分为 {names[pred_class]}",
                        "target": names[gt_class],
                        "prediction": names[pred_class],
                        "confidence": round(best_score, 4),
                        "iou": round(best_iou, 4),
                    })
            elif len(examples) < max_examples:
                examples.append({
                    "file": image_path.name,
                    "reason": "未匹配到有效预测框",
                    "target": names[gt_class],
                    "prediction": "missed",
                    "confidence": 0.0,
                    "iou": round(best_iou, 4),
                })
    return matrix, examples


def draw_confusion(matrix: np.ndarray, names: dict[int, str]) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    labels = [names[index] for index in sorted(names)]
    if not matrix.any():
        return
    fig, ax = plt.subplots(figsize=(14.5, 12.5))
    sns.heatmap(
        matrix,
        cmap="Blues",
        square=True,
        linewidths=0.08,
        linecolor="#f3f4f6",
        cbar_kws={"label": "Matched validation instances"},
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_title("Validation Confusion Matrix (IoU >= 0.50)", fontsize=14, pad=12, weight="bold")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Ground-truth class")
    ax.tick_params(axis="x", labelrotation=60, labelsize=6.8)
    ax.tick_params(axis="y", labelrotation=0, labelsize=6.8)
    fig.tight_layout()
    save_figure(fig, "confusion_matrix")
    plt.close(fig)

    row_sum = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sum, out=np.zeros_like(matrix, dtype=float), where=row_sum != 0)
    fig, ax = plt.subplots(figsize=(14.5, 12.5))
    sns.heatmap(
        normalized,
        cmap="YlGnBu",
        square=True,
        vmin=0,
        vmax=1,
        linewidths=0.08,
        linecolor="#f3f4f6",
        cbar_kws={"label": "Row-normalized proportion"},
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_title("Normalized Validation Confusion Matrix", fontsize=14, pad=12, weight="bold")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Ground-truth class")
    ax.tick_params(axis="x", labelrotation=60, labelsize=6.8)
    ax.tick_params(axis="y", labelrotation=0, labelsize=6.8)
    fig.tight_layout()
    save_figure(fig, "confusion_matrix_normalized")
    plt.close(fig)


def summarize_distribution(summary: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    train_counts = summary["splits"].get("train", {}).get("class_counts", {})
    sorted_counts = sorted(train_counts.items(), key=lambda item: item[1])
    return {
        "top_train": [{"class": cls, "count": count} for cls, count in sorted_counts[-5:][::-1]],
        "low_train": [{"class": cls, "count": count} for cls, count in sorted_counts[:5]],
    }


def sync_static(static_dir: Path) -> None:
    static_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ["*.png", "*.svg", "analysis_summary.json", "error_examples.json"]:
        for source in OUTPUT_DIR.glob(pattern):
            shutil.copy2(source, static_dir / source.name)


def main() -> int:
    args = parse_args()
    data_path = args.data.resolve()
    weights = args.weights.resolve()
    metrics_csv = args.metrics_csv.resolve()
    data = load_yaml(data_path)
    names = normalize_names(data.get("names", {}))
    splits = {split: count_split(resolve_split(data_path, data, split), names) for split in ["train", "val", "test"]}
    metrics = read_metrics(metrics_csv)

    summary: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_yaml": str(data_path),
        "weights": str(weights),
        "metrics_csv": str(metrics_csv),
        "class_count": len(names),
        "splits": splits,
        "metrics": metrics,
        "class_distribution": {},
        "validation_note": "当前 data.yaml 中 test 路径复用 val，本轮真实结果按验证集评估口径解读。",
    }
    summary["class_distribution"] = summarize_distribution(summary)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_plot_style()
    draw_class_distribution(summary)
    draw_top_bottom(summary)
    draw_metrics(metrics)

    examples: list[dict[str, Any]] = []
    if not args.skip_inference:
        matrix, examples = build_confusion(data_path, data, names, weights, args.conf, args.iou, args.max_examples)
        draw_confusion(matrix, names)
        summary["confusion"] = {
            "iou_threshold": args.iou,
            "confidence_threshold": args.conf,
            "matched_instances": int(matrix.sum()),
            "diagonal_instances": int(np.trace(matrix)),
            "matrix_shape": list(matrix.shape),
        }

    (OUTPUT_DIR / "analysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUTPUT_DIR / "error_examples.json").write_text(json.dumps(examples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sync_static(args.static_dir.resolve())
    print(f"[INFO] 分析资产已写入: {OUTPUT_DIR}")
    print(f"[INFO] Web 静态资产已同步: {args.static_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
