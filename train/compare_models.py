"""YOLOv8 与 YOLOv11 模型对比实验。

示例：
python train/compare_models.py --data datasets/sign_language/data.yaml --yolov8 weights/yolov8_best.pt --yolov11 weights/yolov11_best.pt
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "datasets" / "sign_language" / "data.yaml"
DEFAULT_YOLOV8 = PROJECT_ROOT / "weights" / "yolov8_best.pt"
DEFAULT_YOLOV11 = PROJECT_ROOT / "weights" / "yolov11_best.pt"
OUTPUT_CSV = PROJECT_ROOT / "outputs" / "csv" / "model_comparison.csv"
OUTPUT_FIGURE = PROJECT_ROOT / "outputs" / "figures" / "model_comparison.png"
DEFAULT_PROJECT = PROJECT_ROOT / "runs" / "sign_language"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对比 YOLOv8 与 YOLOv11 手语检测模型")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="YOLO data.yaml 路径")
    parser.add_argument("--yolov8", type=Path, default=DEFAULT_YOLOV8, help="YOLOv8 权重路径")
    parser.add_argument("--yolov11", type=Path, default=DEFAULT_YOLOV11, help="YOLOv11 权重路径")
    parser.add_argument("--imgsz", type=int, default=640, help="评估图片尺寸")
    parser.add_argument("--batch", type=int, default=16, help="batch size")
    parser.add_argument("--device", type=str, default=None, help="设备，如 0 或 cpu；默认自动选择")
    return parser.parse_args()


def auto_device() -> str:
    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def empty_row(model_name: str, weights: Path, status: str) -> dict[str, Any]:
    return {
        "model": model_name,
        "weights": str(weights),
        "status": status,
        "precision": "",
        "recall": "",
        "map50": "",
        "map50_95": "",
        "avg_inference_ms": "",
        "fps": "",
        "model_size_mb": round(weights.stat().st_size / 1024 / 1024, 2) if weights.exists() else "",
    }


def evaluate_model(
    model_name: str,
    weights: Path,
    data_path: Path,
    imgsz: int,
    batch: int,
    device: str,
) -> dict[str, Any]:
    if not weights.exists():
        print(f"[WARN] {model_name} 权重不存在，已跳过: {weights}")
        return empty_row(model_name, weights, "missing_weights")

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] 未安装 ultralytics。请先运行: pip install -r requirements.txt")
        return empty_row(model_name, weights, "missing_ultralytics")

    try:
        model = YOLO(str(weights))
        metrics = model.val(
            data=str(data_path),
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=str(DEFAULT_PROJECT),
            name=f"compare_{model_name.lower()}",
            exist_ok=True,
            verbose=False,
        )
        box = getattr(metrics, "box", None)
        speed = getattr(metrics, "speed", {}) or {}
        inference_ms = float(speed.get("inference", 0.0) or 0.0)
        fps = round(1000.0 / inference_ms, 2) if inference_ms > 0 else ""
        return {
            "model": model_name,
            "weights": str(weights),
            "status": "ok",
            "precision": round(float(getattr(box, "mp", 0.0) or 0.0), 6),
            "recall": round(float(getattr(box, "mr", 0.0) or 0.0), 6),
            "map50": round(float(getattr(box, "map50", 0.0) or 0.0), 6),
            "map50_95": round(float(getattr(box, "map", 0.0) or 0.0), 6),
            "avg_inference_ms": round(inference_ms, 4) if inference_ms else "",
            "fps": fps,
            "model_size_mb": round(weights.stat().st_size / 1024 / 1024, 2),
        }
    except Exception as exc:
        print(f"[ERROR] {model_name} 评估失败: {exc}")
        row = empty_row(model_name, weights, "eval_failed")
        row["error"] = str(exc)
        return row


def save_csv(rows: list[dict[str, Any]]) -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "weights",
        "status",
        "precision",
        "recall",
        "map50",
        "map50_95",
        "avg_inference_ms",
        "fps",
        "model_size_mb",
        "error",
    ]
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def draw_figure(rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[WARN] matplotlib 不可用，已跳过模型对比图生成。")
        return

    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if not ok_rows:
        print("[WARN] 没有可绘制的有效评估结果，已跳过模型对比图。")
        return

    metrics = ["precision", "recall", "map50", "map50_95", "fps"]
    labels = [row["model"] for row in ok_rows]
    x = range(len(metrics))
    width = 0.35

    OUTPUT_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    for index, row in enumerate(ok_rows):
        values = [float(row.get(metric) or 0.0) for metric in metrics]
        offsets = [item + (index - (len(ok_rows) - 1) / 2) * width for item in x]
        plt.bar(offsets, values, width=width, label=labels[index])
    plt.xticks(list(x), metrics)
    plt.ylabel("Metric value")
    plt.title("YOLOv8 vs YOLOv11 Model Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURE, dpi=160)
    plt.close()


def print_rows(rows: list[dict[str, Any]]) -> None:
    print("\n========== 模型对比结果 ==========")
    for row in rows:
        print(
            f"{row['model']} | status={row['status']} | "
            f"P={row['precision']} R={row['recall']} mAP50={row['map50']} "
            f"mAP50-95={row['map50_95']} FPS={row['fps']} size={row['model_size_mb']}MB"
        )
    print(f"CSV: {OUTPUT_CSV}")
    print(f"Figure: {OUTPUT_FIGURE}")


def main() -> int:
    args = parse_args()
    data_path = args.data.resolve()
    if not data_path.exists():
        print(f"[ERROR] 未找到数据集配置: {data_path}")
        return 1
    device = args.device if args.device is not None else auto_device()
    rows = [
        evaluate_model("YOLOv8", args.yolov8.resolve(), data_path, args.imgsz, args.batch, device),
        evaluate_model("YOLOv11", args.yolov11.resolve(), data_path, args.imgsz, args.batch, device),
    ]
    save_csv(rows)
    draw_figure(rows)
    print_rows(rows)
    return 0 if any(row.get("status") == "ok" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())

