# 基于 YOLOv11 迁移训练与中文语义映射的多类别手语识别与无障碍交流辅助系统

本项目是计算机视觉课程设计项目，面向多类别手语目标检测、中文语义映射、语音播报和无障碍交流辅助演示。仓库保留旧版 `new-shoyuDetection/` 作为参考，新版代码集中在 `app/`、`train/`、`scripts/`、`configs/` 和 `docs/`。

## 项目体检结论

- 原项目基于 `ultralytics==8.0.199` 和 `yolov8n.pt`，不是完整 YOLOv11 流水线。
- 原项目使用 PyQt5，具备图片、视频、摄像头检测雏形。
- 原 `data.yaml` 存在 `/tmp/pycharm_project_267/...` 绝对路径，且 `val` 指向 train。
- 原数据集目录是 YOLO 风格的 `images/train`、`labels/train`、`images/val`、`labels/val`，新版模板改为 `datasets/sign_language/train|valid|test/images|labels`。
- 原项目有英文类别和中文类别配置，但中文内容出现编码损坏。
- 原项目缺少语音播报、规范历史记录、CSV 导出、模型对比和完整运行文档。

## 功能

- YOLOv11 训练与验证：`train/train_yolov11.py`、`train/val_yolov11.py`
- 数据集体检：路径、标签格式、类别越界、空标签、损坏图片、类别分布图
- 图片、视频、摄像头检测：`app/main.py`、`train/predict_image.py`、`train/predict_video.py`
- 中文语义映射：`configs/label_map.json`
- 中文语音播报：`app/speech.py`
- 识别历史记录与 CSV 导出：`app/recorder.py`
- YOLOv8 与 YOLOv11 指标对比：`train/compare_models.py`
- PyQt5 图形界面：`app/ui_qt.py`

## 项目结构

```text
app/                 # 检测后端、摄像头、语义映射、语音、历史记录、CLI/GUI
train/               # 训练、验证、图片/视频推理、模型对比
scripts/             # 数据集检查脚本
configs/             # 中文标签映射
datasets/sign_language/  # 标准 YOLO 数据集模板，不提交大数据
weights/             # 模型权重目录，不提交 .pt
outputs/             # 运行输出目录，不提交图片/视频/CSV/图表结果
docs/                # 环境、运行、报告素材
new-shoyuDetection/  # 旧项目代码，保留参考
```

## 环境安装

建议 Windows + Python 3.10/3.11：

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

如需 GPU，请按你的 CUDA 版本安装 PyTorch，再安装本项目依赖。

## 数据集准备

将数据放到：

```text
datasets/sign_language/
├── train/images, train/labels
├── valid/images, valid/labels
└── test/images, test/labels
```

每个标签文件为 YOLO 格式：

```text
class_id x_center y_center width height
```

检查数据集：

```powershell
python scripts/check_dataset.py --data datasets/sign_language/data.yaml
```

## 训练与验证

训练 YOLOv11：

```powershell
python train/train_yolov11.py --data datasets/sign_language/data.yaml --model yolo11n.pt --epochs 100 --imgsz 640 --batch 16 --device 0
```

CPU 训练：

```powershell
python train/train_yolov11.py --epochs 1 --batch 2 --device cpu
```

验证：

```powershell
python train/val_yolov11.py --weights weights/yolov11_best.pt --data datasets/sign_language/data.yaml
```

## 检测运行

图片检测：

```powershell
python train/predict_image.py --weights weights/yolov11_best.pt --source path\to\image.jpg --conf 0.5 --save
```

视频检测：

```powershell
python train/predict_video.py --weights weights/yolov11_best.pt --source path\to\video.mp4 --conf 0.5 --save
```

摄像头检测：

```powershell
python app/main.py --weights weights/yolov11_best.pt --source 0 --conf 0.5 --speak
```

PyQt 图形界面：

```powershell
python app/ui_qt.py
```

识别历史默认导出到 `outputs/csv/recognition_history.csv`。

## 模型对比

```powershell
python train/compare_models.py --data datasets/sign_language/data.yaml --yolov8 weights/yolov8_best.pt --yolov11 weights/yolov11_best.pt
```

输出：

- `outputs/csv/model_comparison.csv`
- `outputs/figures/model_comparison.png`

## 常见问题

- `未安装 ultralytics`：运行 `pip install -r requirements.txt`。
- `模型权重不存在`：先训练模型，或把 `best.pt` 放到 `weights/yolov11_best.pt`。
- `data.yaml 路径错误`：使用 `scripts/check_dataset.py` 检查路径。
- 中文语音不标准：Windows 需要安装中文 TTS 声音；没有中文声音时程序会降级，不影响检测。
- 摄像头打不开：尝试 `--source 1` 或检查系统相机权限。

