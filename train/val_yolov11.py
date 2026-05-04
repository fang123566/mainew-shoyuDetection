"""YOLOv11 手语识别验证脚本。

示例：
python train/val_yolov11.py --weights weights/yolov11_best.pt --data datasets/sign_language/data.yaml
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "datasets" / "sign_language" / "data.yaml"
DEFAULT_WEIGHTS = PROJECT_ROOT / "weights" / "yolov11_best.pt"
OUTPUT_CSV = PROJECT_ROOT / "outputs" / "csv" / "val_results.csv"
DEFAULT_PROJECT = PROJECT_ROOT / "runs" / "sign_language"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="验证 YOLOv11 手语检测模型")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="待验证权重路径")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="YOLO data.yaml 路径")
    parser.add_argument("--imgsz", type=int, default=640, help="验证图片尺寸")
    parser.add_argument("--batch", type=int, default=16, help="batch size")
    parser.add_argument("--device", type=str, default=None, help="验证设备，如 0 或 cpu；默认自动选择")
    return parser.parse_args()


def auto_device() -> str:
    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def metrics_to_row(weights: Path, metrics: object) -> dict[str, str | float]:
    box = getattr(metrics, "box", None)
    precision = float(getattr(box, "mp", 0.0) or 0.0)
    recall = float(getattr(box, "mr", 0.0) or 0.0)
    map50 = float(getattr(box, "map50", 0.0) or 0.0)
    map5095 = float(getattr(box, "map", 0.0) or 0.0)
    return {
        "weights": str(weights),
        "precision": precision,
        "recall": recall,
        "map50": map50,
        "map50_95": map5095,
    }


def save_csv(row: dict[str, str | float]) -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["weights", "precision", "recall", "map50", "map50_95"])
        writer.writeheader()
        writer.writerow(row)


def main() -> int:
    args = parse_args()
    weights = args.weights.resolve()
    data_path = args.data.resolve()
    if not weights.exists():
        print(f"[ERROR] 未找到权重文件: {weights}")
        print("请将训练得到的 best.pt 放到 weights/yolov11_best.pt，或通过 --weights 指定。")
        return 1
    if not data_path.exists():
        print(f"[ERROR] 未找到数据集配置: {data_path}")
        return 1

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] 未安装 ultralytics。请先运行: pip install -r requirements.txt")
        return 1

    device = args.device if args.device is not None else auto_device()
    try:
        model = YOLO(str(weights))
        metrics = model.val(
            data=str(data_path),
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            project=str(DEFAULT_PROJECT),
            name="val_yolov11",
            exist_ok=True,
        )
        row = metrics_to_row(weights, metrics)
        save_csv(row)
        print("========== 验证指标 ==========")
        print(f"Precision: {row['precision']:.4f}")
        print(f"Recall: {row['recall']:.4f}")
        print(f"mAP@0.5: {row['map50']:.4f}")
        print(f"mAP@0.5:0.95: {row['map50_95']:.4f}")
        print(f"CSV: {OUTPUT_CSV}")
        return 0
    except Exception as exc:
        print("[ERROR] YOLOv11 验证失败。")
        print(f"原因: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

