"""YOLO 数据集体检脚本。

该脚本用于课程设计交付前快速发现数据路径、标签格式、类别越界和图片损坏等问题。
默认检查 datasets/sign_language/data.yaml，也可以通过 --data 指定任意 YOLO data.yaml。
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

try:
    import cv2
    import numpy as np
except Exception:  # pragma: no cover - 环境未安装 OpenCV 时降级
    cv2 = None
    np = None

try:
    from PIL import Image
except Exception:  # pragma: no cover - 环境未安装 Pillow 时降级
    Image = None

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - 允许无图形依赖时只输出 CSV
    plt = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "datasets" / "sign_language" / "data.yaml"
OUTPUT_CSV = PROJECT_ROOT / "outputs" / "csv" / "dataset_summary.csv"
OUTPUT_FIGURE = PROJECT_ROOT / "outputs" / "figures" / "class_distribution.png"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class SplitReport:
    """单个数据划分的检查结果。"""

    name: str
    image_dir: Path | None = None
    label_dir: Path | None = None
    image_count: int = 0
    label_count: int = 0
    box_count: int = 0
    missing_labels: list[Path] = field(default_factory=list)
    empty_labels: list[Path] = field(default_factory=list)
    invalid_labels: list[str] = field(default_factory=list)
    orphan_labels: list[Path] = field(default_factory=list)
    broken_images: list[Path] = field(default_factory=list)
    class_counts: Counter[int] = field(default_factory=Counter)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查 YOLO 手语数据集格式与类别分布")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="data.yaml 路径")
    return parser.parse_args()


def load_yaml(data_path: Path) -> dict[str, Any]:
    if not data_path.exists():
        raise FileNotFoundError(f"未找到 data.yaml: {data_path}")
    with data_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if "names" not in data:
        raise ValueError("data.yaml 缺少 names 字段，无法判断类别范围")
    return data


def normalize_names(names: Any) -> list[str]:
    """将 YOLO 支持的 list/dict 类别名统一成按 class_id 排序的列表。"""
    if isinstance(names, dict):
        return [str(names[index]) for index in sorted(int(key) for key in names for index in [int(key)])]
    if isinstance(names, list):
        return [str(item) for item in names]
    raise ValueError("names 字段必须是 list 或 dict")


def resolve_split_path(data_path: Path, data: dict[str, Any], split: str) -> Path | None:
    raw_value = data.get(split)
    if raw_value is None and split == "valid":
        raw_value = data.get("val")
    if raw_value is None:
        return None

    split_path = Path(str(raw_value))
    if split_path.is_absolute():
        return split_path

    base = Path(str(data.get("path", "")))
    if base and not base.is_absolute():
        base = PROJECT_ROOT / base
    elif not base:
        base = data_path.parent
    return (base / split_path).resolve()


def infer_label_dir(image_dir: Path) -> Path:
    """根据 YOLO 常见目录约定，从 images 目录推导 labels 目录。"""
    parts = list(image_dir.parts)
    if "images" in parts:
        index = parts.index("images")
        parts[index] = "labels"
        return Path(*parts)
    if image_dir.name == "images":
        return image_dir.parent / "labels"
    return image_dir.parent / "labels" / image_dir.name


def read_image(path: Path) -> bool:
    """使用 OpenCV 验证图片是否可读取，兼容中文路径。"""
    try:
        if cv2 is not None and np is not None:
            data = path.read_bytes()
            image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            return image is not None
        if Image is not None:
            with Image.open(path) as image:
                image.verify()
            return True
        return True
    except Exception:
        return False


def validate_label_file(label_path: Path, class_count: int) -> tuple[int, Counter[int], list[str], bool]:
    box_count = 0
    class_counts: Counter[int] = Counter()
    errors: list[str] = []

    text = label_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return 0, class_counts, errors, True

    for line_no, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{label_path}:{line_no} 标签列数应为 5，实际为 {len(parts)}")
            continue
        try:
            class_id = int(float(parts[0]))
            coords = [float(value) for value in parts[1:]]
        except ValueError:
            errors.append(f"{label_path}:{line_no} 存在非数字标签值")
            continue
        if not 0 <= class_id < class_count:
            errors.append(f"{label_path}:{line_no} class_id={class_id} 超出范围 0-{class_count - 1}")
            continue
        if any(value < 0 or value > 1 for value in coords):
            errors.append(f"{label_path}:{line_no} 坐标不在 0-1 范围: {coords}")
            continue
        box_count += 1
        class_counts[class_id] += 1
    return box_count, class_counts, errors, False


def check_split(name: str, image_dir: Path | None, class_count: int) -> SplitReport:
    report = SplitReport(name=name, image_dir=image_dir)
    if image_dir is None:
        return report
    report.label_dir = infer_label_dir(image_dir)

    if not image_dir.exists():
        report.invalid_labels.append(f"{name}: 图片目录不存在: {image_dir}")
        return report
    if not report.label_dir.exists():
        report.invalid_labels.append(f"{name}: 标签目录不存在: {report.label_dir}")

    images = sorted(path for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    labels = sorted(report.label_dir.rglob("*.txt")) if report.label_dir.exists() else []
    report.image_count = len(images)
    report.label_count = len(labels)

    label_by_stem = defaultdict(list)
    for label in labels:
        label_by_stem[label.stem].append(label)

    image_stems = {image.stem for image in images}
    for label in labels:
        if label.stem not in image_stems and label.name != "classes.txt":
            report.orphan_labels.append(label)

    for image in images:
        if not read_image(image):
            report.broken_images.append(image)
        label_path = report.label_dir / f"{image.stem}.txt" if report.label_dir else None
        if label_path is None or not label_path.exists():
            report.missing_labels.append(image)
            continue
        box_count, counts, errors, is_empty = validate_label_file(label_path, class_count)
        report.box_count += box_count
        report.class_counts.update(counts)
        report.invalid_labels.extend(errors)
        if is_empty:
            report.empty_labels.append(label_path)

    return report


def write_summary(reports: list[SplitReport], names: list[str]) -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["split", "class_id", "class_name", "box_count"])
        for report in reports:
            for class_id, class_name in enumerate(names):
                writer.writerow([report.name, class_id, class_name, report.class_counts.get(class_id, 0)])


def draw_distribution(reports: list[SplitReport], names: list[str]) -> None:
    if plt is None:
        print("[WARN] matplotlib 不可用，已跳过类别分布图生成。")
        return

    total_counts: Counter[int] = Counter()
    for report in reports:
        total_counts.update(report.class_counts)
    values = [total_counts.get(index, 0) for index in range(len(names))]

    OUTPUT_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(max(10, len(names) * 0.45), 6))
    plt.bar(range(len(names)), values, color="#2f6f9f")
    plt.xticks(range(len(names)), names, rotation=60, ha="right", fontsize=8)
    plt.xlabel("Class")
    plt.ylabel("Box count")
    plt.title("Class Distribution")
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURE, dpi=160)
    plt.close()


def print_report(reports: list[SplitReport], names: list[str]) -> None:
    print("\n========== 数据集体检摘要 ==========")
    print(f"类别数: {len(names)}")
    for report in reports:
        print(f"\n[{report.name}]")
        print(f"图片目录: {report.image_dir or '未配置'}")
        print(f"标签目录: {report.label_dir or '未推导'}")
        print(f"图片数: {report.image_count} | 标签数: {report.label_count} | 标注框数: {report.box_count}")
        print(f"缺失标签: {len(report.missing_labels)} | 空标签: {len(report.empty_labels)} | 孤立标签: {len(report.orphan_labels)}")
        print(f"损坏图片: {len(report.broken_images)} | 标签错误: {len(report.invalid_labels)}")
        for message in report.invalid_labels[:10]:
            print(f"  [ERROR] {message}")
        if len(report.invalid_labels) > 10:
            print(f"  ... 还有 {len(report.invalid_labels) - 10} 条标签错误未展开")
    print(f"\nCSV 摘要: {OUTPUT_CSV}")
    print(f"类别分布图: {OUTPUT_FIGURE}")


def main() -> int:
    args = parse_args()
    data_path = args.data.resolve()
    data = load_yaml(data_path)
    names = normalize_names(data["names"])

    split_paths = {
        "train": resolve_split_path(data_path, data, "train"),
        "valid": resolve_split_path(data_path, data, "valid"),
        "test": resolve_split_path(data_path, data, "test"),
    }
    reports = [check_split(name, path, len(names)) for name, path in split_paths.items()]

    write_summary(reports, names)
    draw_distribution(reports, names)
    print_report(reports, names)

    has_errors = any(
        report.invalid_labels or report.missing_labels or report.broken_images for report in reports
    )
    return 1 if has_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
