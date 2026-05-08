# -*- coding: utf-8 -*-
"""HTML 演示界面对应的 Flask 后端。

该文件保留原项目的 templates/index.html 作为前端页面，但后端模型、中文映射和路径
全部切换到本次训练完成的 YOLOv11 权重。
"""

from __future__ import annotations

import base64
import json
import os
import time
import uuid
from io import BytesIO
from pathlib import Path

import shutil
import subprocess
import threading

import cv2
import numpy as np
import torch
from flask import Flask, Response, jsonify, render_template, request, send_from_directory
from PIL import Image
from ultralytics import YOLO

try:
    import imageio_ffmpeg
    FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG_EXE = shutil.which("ffmpeg")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = Path(__file__).resolve().parent
WEIGHTS_PATH = PROJECT_ROOT / "weights" / "yolov11_best.pt"
LABEL_MAP_PATH = PROJECT_ROOT / "configs" / "label_map.json"
SAVE_PATH = PROJECT_ROOT / "outputs" / "images"
VIDEO_SAVE_PATH = PROJECT_ROOT / "outputs" / "videos"
UPLOAD_FOLDER = WEB_ROOT / "static" / "uploads"
RESULT_FOLDER = WEB_ROOT / "static" / "results"
VIDEO_RESULT_FOLDER = WEB_ROOT / "static" / "videos"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}

# 视频任务状态：task_id -> {status, progress, result, ...}
VIDEO_TASKS: dict[str, dict] = {}
VIDEO_TASKS_LOCK = threading.Lock()

app = Flask(
    __name__,
    template_folder=str(WEB_ROOT / "templates"),
    static_folder=str(WEB_ROOT / "static"),
)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["RESULT_FOLDER"] = str(RESULT_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULT_FOLDER.mkdir(parents=True, exist_ok=True)
VIDEO_RESULT_FOLDER.mkdir(parents=True, exist_ok=True)
SAVE_PATH.mkdir(parents=True, exist_ok=True)
VIDEO_SAVE_PATH.mkdir(parents=True, exist_ok=True)


def load_label_map() -> dict[str, str]:
    """读取英文到中文的语义映射表。"""
    if not LABEL_MAP_PATH.exists():
        return {}
    with LABEL_MAP_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_model() -> tuple[YOLO, str]:
    """加载 YOLOv11 权重，优先使用 GPU。"""
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"未找到 YOLOv11 权重: {WEIGHTS_PATH}")
    device = "0" if torch.cuda.is_available() else "cpu"
    model = YOLO(str(WEIGHTS_PATH), task="detect")
    model(np.zeros((48, 48, 3), dtype=np.uint8), device=device, verbose=False)
    return model, device


LABEL_MAP = load_label_map()
MODEL, DEVICE = load_model()
MODEL_NAMES = {int(k): str(v) for k, v in dict(MODEL.names).items()}

print("=" * 60)
print("  多类别手语识别与无障碍交流辅助系统 - HTML 演示后端")
print(f"  Model : {WEIGHTS_PATH}")
print(f"  Device: {DEVICE}")
print("  URL   : http://localhost:5000")
print("=" * 60)


def cv2_to_pil(cv_img):
    """OpenCV BGR 图像转 PIL RGB 图像。"""
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def pil_to_b64(img: Image.Image) -> str:
    """PIL 图像转浏览器可直接显示的 base64 JPEG。"""
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=88)
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def label_cn(label_en: str) -> str:
    """英文标签转中文；映射不存在时返回英文。"""
    return LABEL_MAP.get(label_en, label_en)


def extract_detections(results) -> list[dict[str, object]]:
    """从 Ultralytics 结果中提取前端表格需要的数据。"""
    detections: list[dict[str, object]] = []
    boxes = results.boxes
    if boxes is None or len(boxes) == 0:
        return detections

    xyxy = boxes.xyxy.cpu().numpy()
    clses = boxes.cls.cpu().numpy()
    confs = boxes.conf.cpu().numpy()

    for index, box in enumerate(xyxy):
        cls_id = int(clses[index])
        label_en = MODEL_NAMES.get(cls_id, str(cls_id))
        x1, y1, x2, y2 = [int(v) for v in box]
        detections.append(
            {
                "id": index + 1,
                "class_id": cls_id,
                "class": label_en,
                "class_cn": label_cn(label_en),
                "confidence": round(float(confs[index]) * 100, 2),
                "bbox": [x1, y1, x2, y2],
            }
        )
    return detections


def run_detection(image_path: Path, conf: float, iou: float) -> dict[str, object]:
    """对单张图片执行检测，并返回前端可直接使用的结构化结果。"""
    start = time.perf_counter()
    result = MODEL.predict(
        source=str(image_path),
        conf=conf,
        iou=iou,
        device=DEVICE,
        verbose=False,
    )[0]
    elapsed = time.perf_counter() - start

    plotted = result.plot()
    result_name = f"result_{uuid.uuid4().hex}.jpg"
    result_path = RESULT_FOLDER / result_name
    cv2.imwrite(str(result_path), plotted)

    detections = extract_detections(result)
    return {
        "success": True,
        "result_image": pil_to_b64(cv2_to_pil(plotted)),
        "result_path": f"/static/results/{result_name}",
        "detections": detections,
        "count": len(detections),
        "time": round(elapsed, 3),
        "device": DEVICE,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/model_info")
def model_info():
    return jsonify(
        {
            "success": True,
            "weights": str(WEIGHTS_PATH),
            "device": DEVICE,
            "classes": MODEL_NAMES,
        }
    )


@app.route("/detect_frame", methods=["POST"])
def detect_frame():
    """实时摄像头帧检测：直接读取请求体字节，只返回检测框 JSON，不保存图片、不绘制结果。
    极大降低单帧延迟，适合前端用 canvas overlay 叠加框。
    """
    raw = request.get_data()
    if not raw:
        return jsonify({"error": "empty body"}), 400
    try:
        conf = float(request.args.get("conf", 0.25))
        iou = float(request.args.get("iou", 0.45))
        imgsz = int(request.args.get("imgsz", 640))
        # 把 imgsz 限制在 32 的倍数和合理范围内
        imgsz = max(320, min(1280, (imgsz // 32) * 32 or 640))
        arr = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "decode failed"}), 400
        h, w = frame.shape[:2]
        start = time.perf_counter()
        result = MODEL.predict(
            source=frame, conf=conf, iou=iou, imgsz=imgsz, device=DEVICE, verbose=False,
        )[0]
        elapsed = time.perf_counter() - start
        dets = []
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            clses = boxes.cls.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            for i, box in enumerate(xyxy):
                cls_id = int(clses[i])
                label_en = MODEL_NAMES.get(cls_id, str(cls_id))
                x1, y1, x2, y2 = [int(v) for v in box]
                dets.append({
                    "class": label_en,
                    "class_cn": label_cn(label_en),
                    "confidence": round(float(confs[i]) * 100, 2),
                    "bbox": [x1, y1, x2, y2],
                })
        return jsonify({
            "success": True,
            "width": w,
            "height": h,
            "time": round(elapsed, 3),
            "device": DEVICE,
            "detections": dets,
            "count": len(dets),
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/detect_image", methods=["POST"])
def detect_image():
    """检测上传图片或浏览器摄像头帧。"""
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    conf = float(request.form.get("conf", 0.25))
    iou = float(request.form.get("iou", 0.45))
    suffix = Path(file.filename).suffix.lower() or ".jpg"
    if suffix not in IMAGE_SUFFIXES:
        suffix = ".jpg"

    temp_name = f"{uuid.uuid4().hex}{suffix}"
    temp_path = UPLOAD_FOLDER / temp_name
    file.save(temp_path)

    try:
        return jsonify(run_detection(temp_path, conf, iou))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        if temp_path.exists():
            temp_path.unlink()


@app.route("/detect_batch", methods=["POST"])
def detect_batch():
    """批量检测上传图片；兼容旧版按文件夹路径批量检测。"""
    conf = float(request.form.get("conf", 0.25))
    iou = float(request.form.get("iou", 0.45))

    results_list = []
    total_count = 0
    start = time.perf_counter()
    uploaded_files = request.files.getlist("images")

    if uploaded_files:
        for file in uploaded_files:
            if not file or file.filename == "":
                continue
            suffix = Path(file.filename).suffix.lower() or ".jpg"
            if suffix not in IMAGE_SUFFIXES:
                results_list.append({"file": file.filename, "error": "不支持的图片格式"})
                continue

            temp_path = UPLOAD_FOLDER / f"{uuid.uuid4().hex}{suffix}"
            file.save(temp_path)
            try:
                item = run_detection(temp_path, conf, iou)
                item["file"] = file.filename
                item["image_b64"] = item.pop("result_image")
                total_count += int(item["count"])
                results_list.append(item)
            except Exception as exc:
                results_list.append({"file": file.filename, "error": str(exc)})
            finally:
                if temp_path.exists():
                    temp_path.unlink()

        if not results_list:
            return jsonify({"error": "No valid image files provided"}), 400

        return jsonify(
            {
                "success": True,
                "results": results_list,
                "total_images": len(results_list),
                "total_detections": total_count,
                "time": round(time.perf_counter() - start, 3),
            }
        )

    folder_value = request.form.get("folder_path", "").strip()
    if not folder_value:
        return jsonify({"error": "请上传图片，或提供有效的 folder_path"}), 400

    folder_path = Path(folder_value)
    if not folder_path.exists() or not folder_path.is_dir():
        return jsonify({"error": f"Invalid folder path: {folder_path}"}), 400

    for image_path in sorted(folder_path.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        try:
            item = run_detection(image_path, conf, iou)
            item["file"] = image_path.name
            item["path"] = str(image_path)
            item["image_b64"] = item.pop("result_image")
            total_count += int(item["count"])
            results_list.append(item)
        except Exception as exc:
            results_list.append({"file": image_path.name, "error": str(exc)})

    return jsonify(
        {
            "success": True,
            "results": results_list,
            "total_images": len(results_list),
            "total_detections": total_count,
            "time": round(time.perf_counter() - start, 3),
        }
    )


@app.route("/save_detect_result", methods=["POST"])
def save_detect_result():
    """将当前检测结果保存到 outputs/images。"""
    data = request.get_json() or {}
    result_path = data.get("result_path", "")
    save_name = data.get("save_name", "result.jpg")
    if not result_path:
        return jsonify({"error": "No result path provided"}), 400

    src = RESULT_FOLDER / Path(result_path).name
    if not src.exists():
        return jsonify({"error": f"File not found: {src}"}), 404

    dst = SAVE_PATH / save_name
    dst.write_bytes(src.read_bytes())
    return jsonify({"success": True, "save_path": str(dst)})


@app.route("/save_batch_results", methods=["POST"])
def save_batch_results():
    """保存批量检测结果。"""
    data = request.get_json() or {}
    saved = []
    for item in data.get("results", []):
        result_path = item.get("result_path", "")
        if not result_path:
            continue
        src = RESULT_FOLDER / Path(result_path).name
        if src.exists():
            dst = SAVE_PATH / src.name
            dst.write_bytes(src.read_bytes())
            saved.append(str(dst))
    return jsonify({"success": True, "saved_count": len(saved), "save_path": str(SAVE_PATH)})


def _process_video_task(task_id: str, video_path: Path, conf: float, iou: float, frame_skip: int = 1) -> None:
    """后台线程：逐帧检测视频并写入 mp4，更新 VIDEO_TASKS 进度。"""
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            with VIDEO_TASKS_LOCK:
                VIDEO_TASKS[task_id].update(status="error", error="无法打开视频文件")
            return

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_name = f"video_{task_id}.mp4"
        out_path = VIDEO_RESULT_FOLDER / out_name
        # 先用 OpenCV mp4v 写到临时文件，最后再用 ffmpeg 转 H.264 给浏览器播放
        tmp_path = VIDEO_RESULT_FOLDER / f"_tmp_{task_id}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(tmp_path), fourcc, fps, (width, height))

        with VIDEO_TASKS_LOCK:
            VIDEO_TASKS[task_id].update(total_frames=total, fps=fps, width=width, height=height)

        cls_counter: dict[str, int] = {}
        total_detections = 0
        processed = 0
        frame_idx = 0
        start = time.perf_counter()
        last_plot = None

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            run_this = (frame_idx % max(frame_skip, 1)) == 0
            if run_this:
                result = MODEL.predict(
                    source=frame,
                    conf=conf,
                    iou=iou,
                    device=DEVICE,
                    verbose=False,
                )[0]
                last_plot = result.plot()
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    clses = boxes.cls.cpu().numpy()
                    total_detections += len(clses)
                    for cls_id in clses:
                        name = MODEL_NAMES.get(int(cls_id), str(int(cls_id)))
                        cls_counter[name] = cls_counter.get(name, 0) + 1

            output_frame = last_plot if last_plot is not None else frame
            writer.write(output_frame)
            processed += 1

            if total > 0 and (processed % 5 == 0 or processed == total):
                progress = round(processed / total * 100, 1)
                with VIDEO_TASKS_LOCK:
                    VIDEO_TASKS[task_id].update(
                        progress=progress,
                        processed_frames=processed,
                        detections=total_detections,
                    )

            if VIDEO_TASKS.get(task_id, {}).get("cancel"):
                break

        cap.release()
        writer.release()

        # 用 ffmpeg 把 mp4v 视频重编码为浏览器可播放的 H.264 mp4
        transcoded = False
        if FFMPEG_EXE and tmp_path.exists() and tmp_path.stat().st_size > 0:
            try:
                cmd = [
                    FFMPEG_EXE, "-y", "-loglevel", "error",
                    "-i", str(tmp_path),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-preset", "veryfast", "-movflags", "+faststart",
                    "-an",
                    str(out_path),
                ]
                subprocess.run(cmd, check=True)
                transcoded = True
            except Exception as exc:
                with VIDEO_TASKS_LOCK:
                    VIDEO_TASKS[task_id]["ffmpeg_error"] = str(exc)
        if not transcoded:
            # 没有 ffmpeg 或转码失败：直接使用 mp4v 输出（部分浏览器可能播放不了）
            try:
                if tmp_path.exists():
                    tmp_path.replace(out_path)
            except Exception:
                pass
        else:
            try:
                tmp_path.unlink()
            except Exception:
                pass

        elapsed = time.perf_counter() - start

        top_classes = sorted(cls_counter.items(), key=lambda kv: kv[1], reverse=True)[:10]
        top_classes_payload = [
            {"class": name, "class_cn": label_cn(name), "count": cnt}
            for name, cnt in top_classes
        ]

        with VIDEO_TASKS_LOCK:
            VIDEO_TASKS[task_id].update(
                status="done",
                progress=100.0,
                processed_frames=processed,
                detections=total_detections,
                elapsed=round(elapsed, 2),
                video_url=f"/static/videos/{out_name}",
                video_name=out_name,
                top_classes=top_classes_payload,
            )
    except Exception as exc:
        with VIDEO_TASKS_LOCK:
            VIDEO_TASKS[task_id].update(status="error", error=str(exc))
    finally:
        try:
            if video_path.exists():
                video_path.unlink()
        except Exception:
            pass


@app.route("/detect_video", methods=["POST"])
def detect_video():
    """接收视频文件，启动后台检测任务，返回 task_id。"""
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    suffix = Path(file.filename).suffix.lower()
    if suffix not in VIDEO_SUFFIXES:
        return jsonify({"error": f"不支持的视频格式: {suffix}"}), 400

    conf = float(request.form.get("conf", 0.25))
    iou = float(request.form.get("iou", 0.45))
    frame_skip = int(request.form.get("frame_skip", 1))

    task_id = uuid.uuid4().hex
    temp_name = f"{task_id}{suffix}"
    temp_path = UPLOAD_FOLDER / temp_name
    file.save(temp_path)

    with VIDEO_TASKS_LOCK:
        VIDEO_TASKS[task_id] = {
            "status": "running",
            "progress": 0.0,
            "processed_frames": 0,
            "total_frames": 0,
            "detections": 0,
            "filename": file.filename,
        }

    thread = threading.Thread(
        target=_process_video_task,
        args=(task_id, temp_path, conf, iou, frame_skip),
        daemon=True,
    )
    thread.start()

    return jsonify({"success": True, "task_id": task_id})


@app.route("/video_status/<task_id>")
def video_status(task_id: str):
    with VIDEO_TASKS_LOCK:
        info = VIDEO_TASKS.get(task_id)
    if info is None:
        return jsonify({"error": "task not found"}), 404
    return jsonify({"success": True, **info})


@app.route("/video_cancel/<task_id>", methods=["POST"])
def video_cancel(task_id: str):
    with VIDEO_TASKS_LOCK:
        if task_id in VIDEO_TASKS:
            VIDEO_TASKS[task_id]["cancel"] = True
            return jsonify({"success": True})
    return jsonify({"error": "task not found"}), 404


@app.route("/save_video_result", methods=["POST"])
def save_video_result():
    """把检测后的视频复制到 outputs/videos。"""
    data = request.get_json() or {}
    video_name = data.get("video_name", "")
    if not video_name:
        return jsonify({"error": "No video_name provided"}), 400
    src = VIDEO_RESULT_FOLDER / Path(video_name).name
    if not src.exists():
        return jsonify({"error": f"File not found: {src}"}), 404
    dst = VIDEO_SAVE_PATH / src.name
    dst.write_bytes(src.read_bytes())
    return jsonify({"success": True, "save_path": str(dst)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
