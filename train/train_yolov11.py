"""YOLOv11 手语识别训练脚本。

示例：
python train/train_yolov11.py --data datasets/sign_language/data.yaml --model yolo11n.pt --epochs 100 --imgsz 640 --batch 16 --device 0
"""

from __future__ import annotations

import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "datasets" / "sign_language" / "data.yaml"
DEFAULT_PROJECT = PROJECT_ROOT / "runs" / "sign_language"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 YOLOv11 手语检测模型")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="YOLO data.yaml 路径")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="预训练模型或本地权重路径")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图片尺寸")
    parser.add_argument("--batch", type=int, default=16, help="batch size；显存不足时可调小")
    parser.add_argument("--device", type=str, default=None, help="训练设备，如 0 或 cpu；默认自动选择")
    parser.add_argument("--workers", type=int, default=0, help="Windows 建议从 0 开始，稳定后再调大")
    parser.add_argument("--name", type=str, default="yolov11_sign_language", help="本次实验名称")
    parser.add_argument("--patience", type=int, default=30, help="早停 patience")
    return parser.parse_args()


def auto_device() -> str:
    """自动选择训练设备；未安装 torch 或无 CUDA 时使用 CPU。"""
    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def find_best_weight(save_dir: Path) -> Path | None:
    candidates = [
        save_dir / "weights" / "best.pt",
        save_dir / "best.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = sorted(save_dir.rglob("best.pt"))
    return matches[-1] if matches else None


def main() -> int:
    args = parse_args()
    data_path = args.data.resolve()
    if not data_path.exists():
        print(f"[ERROR] 未找到数据集配置: {data_path}")
        print("请先准备 datasets/sign_language/data.yaml，或通过 --data 指定正确路径。")
        return 1

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] 未安装 ultralytics。请先运行: pip install -r requirements.txt")
        return 1

    device = args.device if args.device is not None else auto_device()
    DEFAULT_PROJECT.mkdir(parents=True, exist_ok=True)

    try:
        print(f"[INFO] 加载模型: {args.model}")
        model = YOLO(args.model)
        print(f"[INFO] 开始训练，device={device}, data={data_path}")
        results = model.train(
            data=str(data_path),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            workers=args.workers,
            project=str(DEFAULT_PROJECT),
            name=args.name,
            patience=args.patience,
            exist_ok=True,
        )

        save_dir = Path(getattr(results, "save_dir", DEFAULT_PROJECT / args.name))
        best_path = find_best_weight(save_dir)
        if best_path:
            print(f"[INFO] 训练完成，best.pt: {best_path.resolve()}")
        else:
            print(f"[WARN] 训练完成，但未在 {save_dir} 下找到 best.pt，请检查 runs 输出。")
        return 0
    except Exception as exc:
        print("[ERROR] YOLOv11 训练失败。")
        print(f"原因: {exc}")
        print("排查建议: 检查 data.yaml 路径、数据集目录、显存/batch size、ultralytics 版本。")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

