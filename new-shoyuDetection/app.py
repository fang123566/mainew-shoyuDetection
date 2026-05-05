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

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, render_template, request
from PIL import Image
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = Path(__file__).resolve().parent
WEIGHTS_PATH = PROJECT_ROOT / "weights" / "yolov11_best.pt"
LABEL_MAP_PATH = PROJECT_ROOT / "configs" / "label_map.json"
SAVE_PATH = PROJECT_ROOT / "outputs" / "images"
UPLOAD_FOLDER = WEB_ROOT / "static" / "uploads"
RESULT_FOLDER = WEB_ROOT / "static" / "results"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

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
SAVE_PATH.mkdir(parents=True, exist_ok=True)


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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
