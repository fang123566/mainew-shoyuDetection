"""多类别手语识别与无障碍交流辅助系统命令行入口。"""

from __future__ import annotations

import argparse
import time
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
    from .recorder import RecognitionRecorder
    from .speech import SpeechEngine
    from .translator import LabelTranslator
except ImportError:  # 允许 python app/main.py 直接运行
    from camera import build_video_writer, cv2, open_capture, parse_source, read_image, resize_to_fit
    from config import DEFAULT_WEIGHTS, OUTPUT_IMAGE_DIR, OUTPUT_VIDEO_DIR, ensure_output_dirs
    from detector import DetectionResult, SignLanguageDetector
    from recorder import RecognitionRecorder
    from speech import SpeechEngine
    from translator import LabelTranslator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="多类别手语识别与无障碍交流辅助系统")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="模型权重路径")
    parser.add_argument("--source", type=str, default="0", help="0 表示摄像头，也可传入图片或视频路径")
    parser.add_argument("--conf", type=float, default=0.5, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU 阈值")
    parser.add_argument("--speak", action="store_true", help="开启中文语音播报")
    parser.add_argument("--save", action="store_true", help="保存图片或视频检测结果")
    parser.add_argument("--top1", action="store_true", help="只保留最高置信度结果")
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


def draw_detections(frame, detections: list[DetectionResult], translator: LabelTranslator, fps: float):
    """绘制检测框、英文标签、中文语义和 FPS。"""
    font = load_font(22)
    for detection in detections:
        color = (40, 180, 90)
        cv2.rectangle(frame, (detection.xmin, detection.ymin), (detection.xmax, detection.ymax), color, 2)
        label_cn = translator.translate_label(detection.label_en)
        text = f"{detection.label_en} | {label_cn} {detection.confidence:.2f}"
        text_y = max(0, detection.ymin - 30)
        cv2.rectangle(frame, (detection.xmin, text_y), (min(detection.xmin + 360, frame.shape[1] - 1), detection.ymin), color, -1)
        frame = draw_chinese_text(frame, text, (detection.xmin + 4, text_y + 3), font)
    cv2.putText(frame, f"FPS: {fps:.1f}", (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 220, 255), 2)
    return frame


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
    annotated = draw_detections(frame, detections, translator, fps)

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
            detections = detector.predict_frame(frame, top1=args.top1)
            now = time.perf_counter()
            fps = 1.0 / max(now - last_time, 1e-6)
            last_time = now
            handle_detections(detections, translator, recorder, speech)
            annotated = draw_detections(frame, detections, translator, fps)
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


def main() -> int:
    args = parse_args()
    ensure_output_dirs()
    try:
        source = parse_source(args.source)
        detector = SignLanguageDetector(args.weights.resolve(), conf=args.conf, iou=args.iou)
        if source.kind == "image":
            process_image(args, detector)
        else:
            process_stream(args, detector)
        return 0
    except Exception as exc:
        print("[ERROR] 程序运行失败。")
        print(f"原因: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

