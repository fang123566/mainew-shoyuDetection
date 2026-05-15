"""多类别手语识别与无障碍交流辅助系统命令行入口。"""

from __future__ import annotations

import argparse
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None
    ImageDraw = None
    ImageFont = None

try:
    from .camera import build_video_writer, cv2, open_capture, parse_source, read_image, resize_to_fit
    from .config import DEFAULT_WEIGHTS, OUTPUT_IMAGE_DIR, OUTPUT_VIDEO_DIR, ensure_output_dirs
    from .detector import DetectionResult, SignLanguageDetector
    from .motion import FrameDifferencer, MotionResult
    from .recorder import RecognitionRecorder
    from .speech import SpeechEngine
    from .translator import LabelTranslator
except ImportError:  # 允许 python app/main.py 直接运行
    from camera import build_video_writer, cv2, open_capture, parse_source, read_image, resize_to_fit
    from config import DEFAULT_WEIGHTS, OUTPUT_IMAGE_DIR, OUTPUT_VIDEO_DIR, ensure_output_dirs
    from detector import DetectionResult, SignLanguageDetector
    from motion import FrameDifferencer, MotionResult
    from recorder import RecognitionRecorder
    from speech import SpeechEngine
    from translator import LabelTranslator


# --------------------------------------------------------------------------- #
#  时序滑动窗口投票器
# --------------------------------------------------------------------------- #

class TemporalVoter:
    """最近 N 帧识别结果滑动窗口投票器，避免单帧误识别导致语音反复播报。"""

    def __init__(self, window_size: int = 5, vote_threshold: int = 3) -> None:
        self.window_size = window_size
        self.vote_threshold = vote_threshold
        self._history: deque[str] = deque(maxlen=window_size)

    def update(self, detections: list[DetectionResult]) -> list[DetectionResult]:
        """将当前帧识别结果加入窗口，返回满足投票阈值的稳定识别结果。"""
        if detections:
            top = detections[0]
            self._history.append(top.label_en)
        else:
            self._history.append("__none__")

        counts = Counter(self._history)
        if counts.most_common(1)[0][1] < self.vote_threshold:
            return []

        # 过滤掉 none，只返回票数达标的类别
        label = counts.most_common(1)[0][0]
        if label == "__none__":
            return []
        return [d for d in detections if d.label_en == label]

    def reset(self) -> None:
        self._history.clear()


# --------------------------------------------------------------------------- #
#  工具函数
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="多类别手语识别与无障碍交流辅助系统")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="模型权重路径")
    parser.add_argument("--source", type=str, default="0", help="0 表示摄像头，也可传入图片或视频路径")
    parser.add_argument("--conf", type=float, default=0.5, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU 阈值")
    parser.add_argument("--speak", action="store_true", help="开启中文语音播报")
    parser.add_argument("--save", action="store_true", help="保存图片或视频检测结果")
    parser.add_argument("--top1", action="store_true", help="只保留最高置信度结果")

    # 运动检测相关参数
    parser.add_argument(
        "--mode",
        type=str,
        default="baseline",
        choices=["baseline", "motion_roi", "motion_filter"],
        help="检测模式：baseline=整图检测，motion_roi=帧差ROI检测，motion_filter=运动过滤",
    )
    parser.add_argument("--show-motion", action="store_true", help="在画面上显示运动区域和ROI框")
    parser.add_argument("--motion-threshold", type=int, default=25, help="帧差二值化阈值")
    parser.add_argument("--min-motion-area", type=int, default=200, help="最小运动区域面积")
    parser.add_argument("--motion-expand", type=float, default=0.3, help="ROI边界扩大比例")
    parser.add_argument("--motion-blur", type=int, default=5, help="高斯模糊核大小（奇数）")
    parser.add_argument("--motion-morph", type=int, default=3, help="形态学操作核大小")

    # 滑动窗口投票参数
    parser.add_argument("--vote-window", type=int, default=5, help="滑动窗口帧数")
    parser.add_argument("--vote-threshold", type=int, default=3, help="投票通过阈值")

    return parser.parse_args()


def load_font(size: int = 24):
    """加载中文字体；找不到字体时使用 PIL 默认字体。"""
    if ImageFont is None:
        return None
    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simsun.ttc"),
        Path(__file__).resolve().parents[1] / "new-shoyuDetection" / "Font" / "platech.ttf",
    ]
    for font_path in candidates:
        if font_path.exists():
            try:
                return ImageFont.truetype(str(font_path), size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def draw_chinese_text(frame, text: str, position: tuple[int, int], font, color=(255, 255, 255)):
    """在 OpenCV 图像上绘制中文文本，Pillow 不可用时降级为 OpenCV 英文绘制。"""
    if Image is None or ImageDraw is None or font is None:
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return frame
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    draw.text(position, text, fill=color, font=font)
    return cv2.cvtColor(__import__("numpy").array(image), cv2.COLOR_RGB2BGR)


def compute_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    """计算两个边界框的 IoU（交并比）。"""
    x1_a, y1_a, x2_a, y2_a = box_a
    x1_b, y1_b, x2_b, y2_b = box_b

    xi1 = max(x1_a, x1_b)
    yi1 = max(y1_a, y1_b)
    xi2 = min(x2_a, x2_b)
    yi2 = min(y2_a, y2_b)

    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h

    area_a = (x2_a - x1_a) * (y2_a - y1_a)
    area_b = (x2_b - x1_b) * (y2_b - y1_b)
    union_area = area_a + area_b - inter_area

    return inter_area / max(union_area, 1e-6)


def draw_detections(
    frame,
    detections: list[DetectionResult],
    translator: LabelTranslator,
    fps: float,
    mode: str = "baseline",
    motion_result: MotionResult | None = None,
    show_motion: bool = False,
    stable_label: str = "",
    motion_area: int = 0,
):
    """绘制检测框、英文标签、中文语义、FPS、模式和运动信息。"""
    font = load_font(22)

    # 绘制运动区域 mask（半透明叠加）
    if show_motion and motion_result is not None and motion_result.mask is not None:
        mask_colored = cv2.applyColorMap(
            (motion_result.mask * 0.4).astype("uint8"), cv2.COLORMAP_JET
        )
        frame = cv2.addWeighted(frame, 0.85, mask_colored, 0.15, 0)

    # 绘制运动 ROI 框（黄色）
    if show_motion and motion_result is not None and motion_result.bbox is not None:
        x1, y1, x2, y2 = motion_result.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(
            frame, "Motion ROI", (x1 + 4, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
        )

    # 绘制 YOLO 检测框（绿色）
    for detection in detections:
        color = (40, 180, 90)
        cv2.rectangle(frame, (detection.xmin, detection.ymin), (detection.xmax, detection.ymax), color, 2)
        label_cn = translator.translate_label(detection.label_en)
        text = f"{detection.label_en} | {label_cn} {detection.confidence:.2f}"
        text_y = max(0, detection.ymin - 30)
        cv2.rectangle(
            frame,
            (detection.xmin, text_y),
            (min(detection.xmin + 360, frame.shape[1] - 1), detection.ymin),
            color, -1,
        )
        frame = draw_chinese_text(frame, text, (detection.xmin + 4, text_y + 3), font)

    # 左上角信息 OSD
    info_lines = [
        f"FPS: {fps:.1f}  模式: {mode}",
        f"当前: {detections[0].label_en if detections else '-'} "
        f"{detections[0].confidence:.2f}" if detections else "当前: -",
        f"运动面积: {motion_area}",
    ]
    if stable_label:
        info_lines.append(f"稳定: {stable_label}")

    y_offset = 34
    for i, line in enumerate(info_lines):
        cv2.putText(
            frame, line, (16, y_offset + i * 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 220, 255), 2,
        )

    return frame


def draw_detections_basic(frame, detections: list[DetectionResult], translator: LabelTranslator, fps: float):
    """兼容旧签名的绘制函数（无运动参数时调用）。"""
    return draw_detections(frame, detections, translator, fps, "baseline", None, False, "", 0)


def handle_detections(
    detections: list[DetectionResult],
    translator: LabelTranslator,
    recorder: RecognitionRecorder,
    speech: SpeechEngine,
) -> None:
    """记录识别结果，并对最高置信度结果做语音播报。"""
    recorder.add_many(detections, translator)
    if detections:
        top = detections[0]
        speech.speak(translator.translate_label(top.label_en))


# --------------------------------------------------------------------------- #
#  处理函数
# --------------------------------------------------------------------------- #

def process_image(args: argparse.Namespace, detector: SignLanguageDetector) -> Path | None:
    source = parse_source(args.source)
    if source.kind != "image":
        raise ValueError("process_image 只能处理图片输入")

    translator = LabelTranslator()
    recorder = RecognitionRecorder()
    speech = SpeechEngine(enabled=args.speak)
    image_path = Path(source.value)
    frame = read_image(image_path)

    start = time.perf_counter()
    detections = detector.predict_image(image_path, top1=args.top1)
    fps = 1.0 / max(time.perf_counter() - start, 1e-6)
    handle_detections(detections, translator, recorder, speech)
    annotated = draw_detections_basic(frame, detections, translator, fps)

    output_path = None
    if args.save:
        OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_IMAGE_DIR / f"{image_path.stem}_detect{image_path.suffix}"
        cv2.imencode(image_path.suffix, annotated)[1].tofile(str(output_path))
        print(f"[INFO] 图片检测结果已保存: {output_path}")

    cv2.imshow("Sign Language Detection", resize_to_fit(annotated))
    cv2.waitKey(0)
    history_path = recorder.export_csv()
    print(f"[INFO] 识别历史已保存: {history_path}")
    return output_path


def process_stream(args: argparse.Namespace, detector: SignLanguageDetector) -> Path | None:
    source = parse_source(args.source)
    if source.kind not in {"camera", "video"}:
        raise ValueError("process_stream 只能处理摄像头或视频输入")

    translator = LabelTranslator()
    recorder = RecognitionRecorder()
    speech = SpeechEngine(enabled=args.speak)

    differencer = FrameDifferencer(
        diff_threshold=args.motion_threshold,
        min_area=args.min_motion_area,
        expand_ratio=args.motion_expand,
        blur_size=args.motion_blur,
        morph_kernel=args.motion_morph,
    )
    voter = TemporalVoter(window_size=args.vote_window, vote_threshold=args.vote_threshold)

    capture = open_capture(source.value)
    writer = None
    output_path = None

    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        fps_source = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
        if args.save:
            OUTPUT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
            suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = OUTPUT_VIDEO_DIR / f"detect_{source.kind}_{suffix}.mp4"
            writer = build_video_writer(output_path, fps_source, (width, height))

        last_time = time.perf_counter()
        while capture.isOpened():
            ok, frame = capture.read()
            if not ok:
                break

            # ---------- 帧差运动检测 ----------
            motion_result = differencer.update(frame)
            motion_area = int(motion_result.score)

            # ---------- 根据模式执行检测 ----------
            if args.mode == "baseline":
                detections = detector.predict_frame(frame, top1=args.top1)

            elif args.mode == "motion_roi":
                if motion_result.bbox is not None:
                    detections = detector.predict_roi(frame, motion_result.bbox, top1=args.top1)
                else:
                    detections = detector.predict_frame(frame, top1=args.top1)

            elif args.mode == "motion_filter":
                all_dets = detector.predict_frame(frame, top1=args.top1)
                if motion_result.bbox is not None and all_dets:
                    filtered = []
                    for det in all_dets:
                        det_box = (det.xmin, det.ymin, det.xmax, det.ymax)
                        iou = compute_iou(det_box, motion_result.bbox)
                        if iou > 0.05:
                            filtered.append(det)
                    detections = filtered
                else:
                    detections = all_dets
            else:
                detections = detector.predict_frame(frame, top1=args.top1)

            # ---------- 滑动窗口投票 ----------
            stable_dets = voter.update(detections)
            stable_label = (
                translator.translate_label(stable_dets[0].label_en)
                if stable_dets
                else ""
            )

            now = time.perf_counter()
            fps = 1.0 / max(now - last_time, 1e-6)
            last_time = now

            # 记录和语音（原始结果）
            handle_detections(detections, translator, recorder, speech)

            # 绘制
            annotated = draw_detections(
                frame, detections, translator, fps,
                mode=args.mode,
                motion_result=motion_result,
                show_motion=args.show_motion,
                stable_label=stable_label,
                motion_area=motion_area,
            )

            if writer is not None:
                writer.write(annotated)
            cv2.imshow("Sign Language Detection", resize_to_fit(annotated))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        history_path = recorder.export_csv()
        print(f"[INFO] 识别历史已保存: {history_path}")
        if output_path:
            print(f"[INFO] 视频检测结果已保存: {output_path}")
    return output_path


# --------------------------------------------------------------------------- #
#  启动入口
# --------------------------------------------------------------------------- #

def main() -> int:
    args = parse_args()
    ensure_output_dirs()

    # 打印启动信息
    print("=" * 60)
    print("  多类别手语识别与无障碍交流辅助系统")
    print("=" * 60)
    print(f"  模式:         {args.mode}")
    print(f"  权重路径:     {Path(args.weights).resolve()}")
    print(f"  置信度阈值:   {args.conf}")
    print(f"  NMS IoU:      {args.iou}")
    print(f"  帧差阈值:     {args.motion_threshold}")
    print(f"  最小运动面积: {args.min_motion_area}")
    print(f"  ROI扩大比例:  {args.motion_expand}")
    print(f"  模糊核大小:   {args.motion_blur}")
    print(f"  形态学核:     {args.motion_morph}")
    print(f"  投票窗口:     {args.vote_window}")
    print(f"  投票阈值:     {args.vote_threshold}")
    print(f"  显示运动区域: {args.show_motion}")
    print(f"  语音播报:     {args.speak}")
    print(f"  保存结果:     {args.save}")
    print("=" * 60)

    try:
        source = parse_source(args.source)
        detector = SignLanguageDetector(args.weights.resolve(), conf=args.conf, iou=args.iou)
        print(f"  类别数量:     {len(detector.names)}")
        print(f"  类别列表:     {list(detector.names.values())}")
        print("=" * 60)

        if source.kind == "image":
            process_image(args, detector)
        else:
            process_stream(args, detector)
        return 0
    except Exception as exc:
        print("[ERROR] 程序运行失败。")
        print(f"原因: {exc}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
