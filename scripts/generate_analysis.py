"""生成 Web 评估分析页可选资产。

默认只读取 data.yaml 和标签文件，生成类别分布图与摘要。
如果传入 --with-inference，则额外用当前权重扫描验证集，导出低置信/漏检样例清单。
生成物写入 outputs/analysis/，该目录不提交到 Git。
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "datasets" / "sign_language" / "data.yaml"
DEFAULT_WEIGHTS = PROJECT_ROOT / "weights" / "yolov11_best.pt"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "analysis"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 Web 评估分析资产")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="YOLO data.yaml 路径")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="模型权重路径")
    parser.add_argument("--conf", type=float, default=0.5, help="低置信样例阈值")
    parser.add_argument("--max-examples", type=int, default=40, help="最多导出多少个问题样例")
    parser.add_argument("--with-inference", action="store_true", help="扫描验证集并导出低置信/漏检样例")
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
    base = Path(str(data.get("path", data_path.parent)))
    if not base.is_absolute():
        base = data_path.parent / base
    return (base / split_path).resolve()


def infer_label_dir(image_dir: Path) -> Path:
    parts = list(image_dir.parts)
    if "images" in parts:
        index = parts.index("images")
        parts[index] = "labels"
        return Path(*parts)
    return image_dir.parent / "labels" / image_dir.name


def count_split(image_dir: Path | None, names: dict[int, str]) -> dict[str, Any]:
    if image_dir is None:
        return {"images": 0, "labels": 0, "boxes": 0, "class_counts": {}}
    label_dir = infer_label_dir(image_dir)
    images = [path for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES] if image_dir.exists() else []
    labels = [path for path in label_dir.rglob("*.txt") if path.name != "classes.txt"] if label_dir.exists() else []
    counts: Counter[int] = Counter()
    boxes = 0
    for label_path in labels:
        text = label_path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue
        for line in text.splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            try:
                class_id = int(float(parts[0]))
            except ValueError:
                continue
            if class_id in names:
                counts[class_id] += 1
                boxes += 1
    return {
        "images": len(images),
        "labels": len(labels),
        "boxes": boxes,
        "class_counts": {names[key]: counts.get(key, 0) for key in sorted(names)},
    }


def draw_distribution(summary: dict[str, Any]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[WARN] matplotlib 不可用，跳过类别分布图。")
        return
    train_counts = summary["splits"].get("train", {}).get("class_counts", {})
    labels = list(train_counts.keys())
    values = [train_counts[label] for label in labels]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(max(12, len(labels) * 0.42), 6))
    plt.bar(range(len(labels)), values, color="#6366f1")
    plt.xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=8)
    plt.ylabel("Box count")
    plt.title("Training Class Distribution")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "class_distribution.png", dpi=160)
    plt.close()


def copy_existing_confusion_matrix() -> None:
    candidates = sorted(PROJECT_ROOT.glob("runs/**/confusion_matrix*.png")) + sorted(
        PROJECT_ROOT.glob("new-shoyuDetection/runs/**/confusion_matrix*.png")
    )
    for candidate in candidates:
        target = OUTPUT_DIR / candidate.name
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate, target)


def collect_error_examples(data_path: Path, data: dict[str, Any], weights: Path, conf: float, max_examples: int) -> list[dict[str, Any]]:
    try:
        from ultralytics import YOLO
    except Exception:
        print("[WARN] ultralytics 不可用，跳过错误样例扫描。")
        return []
    val_dir = resolve_split(data_path, data, "val")
    if val_dir is None or not val_dir.exists() or not weights.exists():
        return []
    model = YOLO(str(weights), task="detect")
    examples: list[dict[str, Any]] = []
    for image_path in sorted(path for path in val_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES):
        result = model.predict(source=str(image_path), conf=0.01, verbose=False)[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            examples.append({"file": image_path.name, "reason": "未检测到目标", "confidence": 0.0})
        else:
            best = float(boxes.conf.max().item())
            if best < conf:
                examples.append({"file": image_path.name, "reason": "最高置信度低", "confidence": round(best, 4)})
        if len(examples) >= max_examples:
            break
    return examples


def main() -> int:
    args = parse_args()
    data_path = args.data.resolve()
    data = load_yaml(data_path)
    names = normalize_names(data.get("names", {}))
    splits = {
        split: count_split(resolve_split(data_path, data, split), names)
        for split in ["train", "val", "test"]
    }
    summary = {"data_yaml": str(data_path), "class_count": len(names), "splits": splits}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    draw_distribution(summary)
    copy_existing_confusion_matrix()
    if args.with_inference:
        examples = collect_error_examples(data_path, data, args.weights.resolve(), args.conf, args.max_examples)
        (OUTPUT_DIR / "error_examples.json").write_text(
            json.dumps(examples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(f"[INFO] 分析资产已写入: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
