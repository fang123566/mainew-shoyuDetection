# -*- coding: utf-8 -*-
import os
import time
import uuid
import base64
import numpy as np
from io import BytesIO
from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image
import cv2
import torch
from ultralytics import YOLO
import Config

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Load model
device = 0 if torch.cuda.is_available() else 'cpu'
model = YOLO(Config.model_path, task='detect')
model(np.zeros((48, 48, 3), dtype=np.uint8), device=device)

print(f"[INFO] Model loaded on device: {device}")


def pil_to_b64(img):
    """Convert PIL Image to base64 string."""
    buffer = BytesIO()
    img.save(buffer, format='JPEG', quality=85)
    b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/jpeg;base64,{b64}"


def pil_to_b64_png(img):
    """Convert PIL Image to base64 PNG string."""
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{b64}"


def cv2_to_pil(cv_img):
    """Convert OpenCV image (BGR) to PIL Image (RGB)."""
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/detect_image', methods=['POST'])
def detect_image():
    """Detect objects in a single uploaded image."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    conf = float(request.form.get('conf', 0.25))
    iou = float(request.form.get('iou', 0.45))

    # Save uploaded file
    suffix = os.path.splitext(file.filename)[1].lower()
    temp_name = f"{uuid.uuid4().hex}{suffix}"
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_name)
    file.save(temp_path)

    try:
        # Run detection
        t1 = time.time()
        results = model(temp_path, conf=conf, iou=iou)[0]
        t2 = time.time()

        # Plot result
        plotted = results.plot()
        result_name = f"result_{temp_name}"
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_name)
        cv2.imwrite(result_path, plotted)

        # Extract detection info
        boxes = results.boxes
        detections = []
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            clses = boxes.cls.cpu().numpy()
            confs = boxes.conf.cpu().numpy()

            for i in range(len(xyxy)):
                x1, y1, x2, y2 = map(int, xyxy[i])
                cls_id = int(clses[i])
                detections.append({
                    'id': i + 1,
                    'class': Config.names.get(cls_id, 'unknown'),
                    'class_cn': Config.CH_names[cls_id] if cls_id < len(Config.CH_names) else '未知',
                    'confidence': round(float(confs[i]) * 100, 2),
                    'bbox': [int(x1), int(y1), int(x2), int(y2)]
                })

        result_b64 = pil_to_b64(cv2_to_pil(plotted))

        return jsonify({
            'success': True,
            'result_image': result_b64,
            'result_path': f'/static/results/{result_name}',
            'detections': detections,
            'count': len(detections),
            'time': round(t2 - t1, 3)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.route('/detect_batch', methods=['POST'])
def detect_batch():
    """Batch detect objects in all images from a folder path."""
    folder_path = request.form.get('folder_path', '').strip()
    conf = float(request.form.get('conf', 0.25))
    iou = float(request.form.get('iou', 0.45))

    if not folder_path or not os.path.isdir(folder_path):
        return jsonify({'error': 'Invalid folder path'}), 400

    img_suffix = ['jpg', 'png', 'jpeg', 'bmp']
    results_list = []
    total_count = 0

    for file_name in os.listdir(folder_path):
        full_path = os.path.join(folder_path, file_name)
        if os.path.isfile(full_path) and file_name.split('.')[-1].lower() in img_suffix:
            try:
                results = model(full_path, conf=conf, iou=iou)[0]
                plotted = results.plot()

                result_name = f"result_{uuid.uuid4().hex}.jpg"
                result_path = os.path.join(app.config['RESULT_FOLDER'], result_name)
                cv2.imwrite(result_path, plotted)

                boxes = results.boxes
                detections = []
                if boxes is not None and len(boxes) > 0:
                    xyxy = boxes.xyxy.cpu().numpy()
                    clses = boxes.cls.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()
                    for i in range(len(xyxy)):
                        cls_id = int(clses[i])
                        detections.append({
                            'class': Config.names.get(cls_id, 'unknown'),
                            'class_cn': Config.CH_names[cls_id] if cls_id < len(Config.CH_names) else '未知',
                            'confidence': round(float(confs[i]) * 100, 2),
                        })
                    total_count += len(detections)

                results_list.append({
                    'file': file_name,
                    'path': full_path,
                    'result_path': f'/static/results/{result_name}',
                    'count': len(detections),
                    'detections': detections,
                    'image_b64': pil_to_b64(cv2_to_pil(plotted))
                })
            except Exception as e:
                results_list.append({
                    'file': file_name,
                    'error': str(e)
                })

    return jsonify({
        'success': True,
        'results': results_list,
        'total_images': len(results_list),
        'total_detections': total_count
    })


@app.route('/save_detect_result', methods=['POST'])
def save_detect_result():
    """Save detection result image to Config.save_path."""
    data = request.get_json()
    result_path = data.get('result_path', '')
    save_name = data.get('save_name', 'result.jpg')

    if not result_path:
        return jsonify({'error': 'No result path provided'}), 400

    os.makedirs(Config.save_path, exist_ok=True)
    save_path = os.path.join(Config.save_path, save_name)

    try:
        # result_path is like /static/results/result_xxx.jpg
        src = os.path.join('static/results', os.path.basename(result_path))
        if os.path.exists(src):
            import shutil
            shutil.copy(src, save_path)
        else:
            return jsonify({'error': 'File not found'}), 404

        return jsonify({'success': True, 'save_path': save_path})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/save_batch_results', methods=['POST'])
def save_batch_results():
    """Save batch detection results."""
    data = request.get_json()
    results = data.get('results', [])

    os.makedirs(Config.save_path, exist_ok=True)
    saved = []

    for r in results:
        result_path = r.get('result_path', '')
        if result_path:
            src = os.path.join('static/results', os.path.basename(result_path))
            if os.path.exists(src):
                import shutil
                fname = os.path.basename(result_path)
                dst = os.path.join(Config.save_path, fname)
                shutil.copy(src, dst)
                saved.append(dst)

    return jsonify({'success': True, 'saved_count': len(saved), 'save_path': Config.save_path})


if __name__ == '__main__':
    print("=" * 50)
    print("  手语检测 Web 服务")
    print("  访问地址: http://localhost:5000")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5000, debug=True)
