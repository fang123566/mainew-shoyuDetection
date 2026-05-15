"""三种检测模式（baseline / motion_roi / motion_filter）在同一视频上的对比评估脚本。

使用方法：
    python scripts/evaluate_demo_video.py --video <视频路径> [--output <输出目录>]
    python scripts/evaluate_demo_video.py --video demo.mp4

输出：
    outputs/csv/motion_ablation.csv      — 各模式指标汇总表
    outputs/figures/motion_ablation.png  — 柱状图对比
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
#  路径设置（与 app/ 模块保持一致）
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = PROJECT_ROOT / "weights"
OUTPUT_CSV_DIR = PROJECT_ROOT / "outputs" / "csv"
OUTPUT_FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

DEFAULT_WEIGHTS = WEIGHTS_DIR / "yolov11_best.pt"


# --------------------------------------------------------------------------- #
#  帧间差分类（内联，避免循环 import）
# --------------------------------------------------------------------------- #

class FrameDifferencer:
    def __init__(
        self,
        diff_threshold: int = 25,
        min_area: int = 200,
        expand_ratio: float = 0.3,
        blur_size: int = 5,
        morph_kernel: int = 3,
    ) -> None:
        self.diff_threshold = diff_threshold
        self.min_area = min_area
        self.expand_ratio = expand_ratio
        self.blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1
        self.morph_kernel = morph_kernel
        self._prev_frame: object | None = None

    def update(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._prev_frame is None:
            self._prev_frame = gray
            return None, None, 0.0

        gray_blur = cv2.GaussianBlur(gray, (self.blur_size, self.blur_size), 0)
        prev_blur = cv2.GaussianBlur(self._prev_frame, (self.blur_size, self.blur_size), 0)
        diff = cv2.absdiff(gray_blur, prev_blur)
        _, mask = cv2.threshold(diff, self.diff_threshold, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morph_kernel, self.morph_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, dilate_kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [c for c in contours if cv2.contourArea(c) >= self.min_area]

        bbox = None
        score = 0.0
        if valid_contours:
            x_min = y_min = float("inf")
            x_max = y_max = float("-inf")
            total_area = 0.0
            for cnt in valid_contours:
                x, y, w, h = cv2.boundingRect(cnt)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x + w)
                y_max = max(y_max, y + h)
                total_area += cv2.contourArea(cnt)
            h_f, w_f = frame.shape[:2]
            expand_w = int((x_max - x_min) * self.expand_ratio)
            expand_h = int((y_max - y_min) * self.expand_ratio)
            x_min_e = max(0, x_min - expand_w)
            y_min_e = max(0, y_min - expand_h)
            x_max_e = min(w_f, x_max + expand_w)
            y_max_e = min(h_f, y_max + expand_h)
            bbox = (int(x_min_e), int(y_min_e), int(x_max_e), int(y_max_e))
            score = float(total_area)

        self._prev_frame = gray
        return mask, bbox, score


# --------------------------------------------------------------------------- #
#  工具函数
# --------------------------------------------------------------------------- #

def compute_iou(box_a, box_b):
    x1_a, y1_a, x2_a, y2_a = box_a
    x1_b, y1_b, x2_b, y2_b = box_b
    xi1 = max(x1_a, x1_b)
    yi1 = max(y1_a, y1_b)
    xi2 = min(x2_a, x2_b)
    yi2 = min(y2_a, y2_b)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area_a = (x2_a - x1_a) * (y2_a - y1_a)
    area_b = (x2_b - x1_b) * (y2_b - y1_b)
    return inter / max(area_a + area_b - inter, 1e-6)


def load_yolo(weights_path: Path):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("未安装 ultralytics: pip install ultralytics") from exc
    return YOLO(str(weights_path))


def run_evaluation(
    video_path: Path,
    weights_path: Path,
    conf: float = 0.5,
    iou: float = 0.45,
    diff_threshold: int = 25,
    min_area: int = 200,
) -> dict[str, dict]:
    """对视频分别运行三种模式，返回各模式指标。"""
    model = load_yolo(weights_path)
    names = getattr(model, "names", {}) or {}
    names = {int(k): str(v) for k, v in names.items()}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    # 各模式统计
    stats = {
        "baseline": {"total": 0, "detected": 0, "conf_sum": 0.0, "empty_frames": 0, "fps_sum": 0.0},
        "motion_roi": {"total": 0, "detected": 0, "conf_sum": 0.0, "empty_frames": 0, "fps_sum": 0.0},
        "motion_filter": {"total": 0, "detected": 0, "conf_sum": 0.0, "empty_frames": 0, "fps_sum": 0.0},
    }

    frame_idx = 0
    differencer = FrameDifferencer(diff_threshold=diff_threshold, min_area=min_area)
    last_time = time.perf_counter()

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1
        now = time.perf_counter()
        fps = 1.0 / max(now - last_time, 1e-6)
        last_time = now

        mask, motion_bbox, motion_score = differencer.update(frame)

        # ---- baseline ----
        results = model.predict(source=frame, conf=conf, iou=iou, verbose=False)
        boxes = results[0].boxes
        det_count = len(boxes) if boxes is not None and len(boxes) > 0 else 0
        confs = boxes.conf.cpu().numpy() if boxes is not None and len(boxes) > 0 else []
        stats["baseline"]["total"] += 1
        stats["baseline"]["fps_sum"] += fps
        if det_count > 0:
            stats["baseline"]["detected"] += 1
            stats["baseline"]["conf_sum"] += float(confs.mean())
        else:
            stats["baseline"]["empty_frames"] += 1

        # ---- motion_roi ----
        if motion_bbox is not None:
            x1, y1, x2, y2 = motion_bbox
            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:
                roi_results = model.predict(source=roi, conf=conf, iou=iou, verbose=False)
                roi_boxes = roi_results[0].boxes
                roi_confs = []
                roi_dets = []
                if roi_boxes is not None and len(roi_boxes) > 0:
                    xyxy = roi_boxes.xyxy.cpu().numpy()
                    for box in xyxy:
                        bx1, by1, bx2, by2 = [int(v) for v in box]
                        roi_dets.append((bx1 + x1, by1 + y1, bx2 + x1, by2 + y1))
                        roi_confs.append(
                            roi_boxes.conf.cpu().numpy()[
                                list(roi_boxes.xyxy.cpu().numpy()).index(box)
                            ]
                        )
                det_count = len(roi_dets)
                conf_sum_roi = sum(roi_confs) if roi_confs else 0.0
            else:
                det_count = 0
                conf_sum_roi = 0.0
        else:
            results = model.predict(source=frame, conf=conf, iou=iou, verbose=False)
            boxes = results[0].boxes
            det_count = len(boxes) if boxes is not None and len(boxes) > 0 else 0
            confs = boxes.conf.cpu().numpy() if boxes is not None and len(boxes) > 0 else []
            conf_sum_roi = float(sum(confs)) if len(confs) > 0 else 0.0

        stats["motion_roi"]["total"] += 1
        stats["motion_roi"]["fps_sum"] += fps
        if det_count > 0:
            stats["motion_roi"]["detected"] += 1
            stats["motion_roi"]["conf_sum"] += conf_sum_roi / max(det_count, 1)
        else:
            stats["motion_roi"]["empty_frames"] += 1

        # ---- motion_filter ----
        all_results = model.predict(source=frame, conf=conf, iou=iou, verbose=False)
        all_boxes = all_results[0].boxes
        if all_boxes is not None and len(all_boxes) > 0:
            xyxy = all_boxes.xyxy.cpu().numpy()
            confs = all_boxes.conf.cpu().numpy()
            filtered = []
            for box, cf in zip(xyxy, confs):
                bx1, by1, bx2, by2 = [int(v) for v in box]
                det_box = (bx1, by1, bx2, by2)
                if motion_bbox is not None and compute_iou(det_box, motion_bbox) > 0.05:
                    filtered.append(cf)
            det_count = len(filtered)
            conf_sum_f = sum(filtered) if filtered else 0.0
        else:
            det_count = 0
            conf_sum_f = 0.0

        stats["motion_filter"]["total"] += 1
        stats["motion_filter"]["fps_sum"] += fps
        if det_count > 0:
            stats["motion_filter"]["detected"] += 1
            stats["motion_filter"]["conf_sum"] += conf_sum_f / max(det_count, 1)
        else:
            stats["motion_filter"]["empty_frames"] += 1

    cap.release()

    # 计算汇总指标
    summary = {}
    for mode, s in stats.items():
        total = max(s["total"], 1)
        avg_conf = s["conf_sum"] / max(s["detected"], 1) if s["detected"] > 0 else 0.0
        avg_fps = s["fps_sum"] / total
        empty_ratio = s["empty_frames"] / total
        summary[mode] = {
            "detection_count": s["detected"],
            "avg_confidence": round(avg_conf, 4),
            "empty_ratio": round(empty_ratio, 4),
            "avg_fps": round(avg_fps, 2),
            "total_frames": total,
        }
    return summary


def save_csv(summary: dict, output_path: Path) -> None:
    """保存 CSV。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["mode", "detection_count", "avg_confidence", "empty_ratio", "avg_fps", "total_frames"]
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for mode, metrics in summary.items():
            row = {"mode": mode, **metrics}
            writer.writerow(row)
    print(f"[INFO] CSV 已保存: {output_path}")


def save_figure(summary: dict, output_path: Path) -> None:
    """生成柱状图对比。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] 未安装 matplotlib，跳过图表生成")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    modes = list(summary.keys())
    detection_counts = [summary[m]["detection_count"] for m in modes]
    avg_confs = [summary[m]["avg_confidence"] for m in modes]
    empty_ratios = [summary[m]["empty_ratio"] for m in modes]
    avg_fps_list = [summary[m]["avg_fps"] for m in modes]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Motion Detection Mode Ablation Study", fontsize=14)

    colors = ["#4c72b0", "#dd8452", "#55a868"]

    axes[0, 0].bar(modes, detection_counts, color=colors)
    axes[0, 0].set_title("Detection Count")
    axes[0, 0].set_ylabel("count")
    for i, v in enumerate(detection_counts):
        axes[0, 0].text(i, v + 0.5, str(v), ha="center", fontsize=10)

    axes[0, 1].bar(modes, avg_confs, color=colors)
    axes[0, 1].set_title("Average Confidence")
    axes[0, 1].set_ylabel("confidence")
    axes[0, 1].set_ylim(0, 1.0)
    for i, v in enumerate(avg_confs):
        axes[0, 1].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)

    axes[1, 0].bar(modes, empty_ratios, color=colors)
    axes[1, 0].set_title("Empty Detection Ratio")
    axes[1, 0].set_ylabel("ratio")
    axes[1, 0].set_ylim(0, 1.0)
    for i, v in enumerate(empty_ratios):
        axes[1, 0].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)

    axes[1, 1].bar(modes, avg_fps_list, color=colors)
    axes[1, 1].set_title("Average FPS")
    axes[1, 1].set_ylabel("FPS")
    for i, v in enumerate(avg_fps_list):
        axes[1, 1].text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[INFO] 图表已保存: {output_path}")


# --------------------------------------------------------------------------- #
#  入口
# --------------------------------------------------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(description="三种检测模式对比评估")
    parser.add_argument("--video", type=Path, required=True, help="演示视频路径")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="模型权重路径")
    parser.add_argument("--conf", type=float, default=0.5, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU 阈值")
    parser.add_argument("--diff-threshold", type=int, default=25, help="帧差阈值")
    parser.add_argument("--min-area", type=int, default=200, help="最小运动区域面积")
    parser.add_argument(
        "--output-csv", type=Path,
        default=OUTPUT_CSV_DIR / "motion_ablation.csv",
        help="CSV 输出路径"
    )
    parser.add_argument(
        "--output-fig", type=Path,
        default=OUTPUT_FIGURE_DIR / "motion_ablation.png",
        help="图表输出路径"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("  帧间差分运动区域检测 — 三模式对比评估")
    print("=" * 60)
    print(f"  视频:        {args.video}")
    print(f"  权重:        {args.weights}")
    print(f"  置信度阈值:  {args.conf}")
    print(f"  帧差阈值:    {args.diff_threshold}")
    print(f"  最小运动面积: {args.min_area}")
    print("=" * 60)

    summary = run_evaluation(
        args.video, args.weights,
        conf=args.conf, iou=args.iou,
        diff_threshold=args.diff_threshold,
        min_area=args.min_area,
    )

    print("\n评估结果：")
    print(f"{'模式':<20} {'检测次数':>8} {'平均置信度':>10} {'空帧比例':>10} {'平均FPS':>10} {'总帧数':>8}")
    print("-" * 70)
    for mode, m in summary.items():
        print(
            f"{mode:<20} {m['detection_count']:>8} "
            f"{m['avg_confidence']:>10.4f} {m['empty_ratio']:>10.4f} "
            f"{m['avg_fps']:>10.2f} {m['total_frames']:>8}"
        )

    save_csv(summary, args.output_csv)
    save_figure(summary, args.output_fig)
    print("\n完成！")


if __name__ == "__main__":
    raise SystemExit(main())
