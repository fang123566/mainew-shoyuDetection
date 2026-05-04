# 基于 YOLOv11 迁移训练与中文语义映射的多类别手语识别与无障碍交流辅助系统

> 计算机视觉课程设计项目 | YOLOv11 迁移训练 | 中文语义映射 | 语音播报 | PyQt5 图形界面

---

## 1. 项目简介

本项目基于 YOLOv11 目标检测框架，对 35 类中国手语动作词进行迁移训练，并构建完整的中文语义映射、语音播报、识别历史记录和图形界面演示系统。适合计算机视觉课程期末答辩展示。

**核心功能：**
- YOLOv11 模型训练与验证
- YOLOv8 与 YOLOv11 模型性能对比实验
- 图片、视频、摄像头实时检测
- 英文标签 → 中文语义映射
- 中文语音播报（TTS）
- 识别历史记录与 CSV 导出
- PyQt5 图形界面

---

## 2. 项目结构

```
sign_language_yolov11/
├── app/                          # 检测后端与用户入口
│   ├── main.py                   # 命令行实时检测入口
│   ├── ui_qt.py                  # PyQt5 图形界面
│   ├── detector.py               # YOLO 模型封装（返回结构化结果）
│   ├── camera.py                 # 摄像头/视频/图片读取工具
│   ├── translator.py             # 英文 → 中文语义映射
│   ├── speech.py                 # 中文语音播报（TTS）
│   ├── recorder.py               # 识别历史记录与 CSV 导出
│   └── config.py                 # 全局路径配置
│
├── train/                        # 训练与推理脚本
│   ├── train_yolov11.py         # YOLOv11 训练
│   ├── val_yolov11.py           # 验证脚本
│   ├── predict_image.py          # 图片推理（复用 main.py）
│   ├── predict_video.py          # 视频推理（复用 main.py）
│   └── compare_models.py         # YOLOv8 vs YOLOv11 对比
│
├── scripts/
│   └── check_dataset.py          # 数据集质量检查脚本
│
├── configs/
│   └── label_map.json           # 英文 → 中文语义映射表
│
├── datasets/
│   └── sign_language/
│       ├── data.yaml             # YOLO 数据集配置
│       ├── train/images/        # 训练图片
│       ├── train/labels/        # 训练标签
│       ├── valid/images/         # 验证图片
│       ├── valid/labels/        # 验证标签
│       ├── test/images/          # 测试图片
│       └── test/labels/          # 测试标签
│
├── weights/                      # 模型权重（.gitignore 忽略 .pt）
│
├── outputs/                      # 运行输出
│   ├── images/                   # 图片检测结果
│   ├── videos/                   # 视频检测结果
│   ├── logs/                    # 日志文件
│   ├── csv/                     # CSV 导出（历史、统计）
│   └── figures/                 # 图表（分布图、对比图）
│
├── docs/                         # 项目文档
│   ├── 项目说明.md
│   ├── 环境配置.md
│   ├── 运行说明.md
│   └── 实验报告素材.md
│
├── new-shoyuDetection/           # 旧版参考代码（保留）
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 3. 环境安装

### 3.1 依赖安装

```powershell
# 方法一：虚拟环境
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# 方法二：conda
conda create -n ai_trading python=3.11 -y
conda activate ai_trading
pip install -r requirements.txt
```

### 3.2 GPU 支持（可选）

```powershell
# 先安装 PyTorch（根据你的 CUDA 版本选择）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# 然后安装其他依赖
pip install -r requirements.txt
```

详细说明见 [docs/环境配置.md](docs/环境配置.md)。

---

## 4. 数据集准备

数据集位于 `datasets/sign_language/`，已配置在 `data.yaml` 中。

数据格式（YOLO 检测格式）：
```
class_id x_center y_center width height   # 归一化坐标 0~1
```

运行数据检查：
```powershell
python scripts/check_dataset.py --data datasets/sign_language/data.yaml
```

---

## 5. 训练与验证

### 5.1 训练 YOLOv11

```powershell
# GPU 训练（推荐）
python train/train_yolov11.py --data datasets/sign_language/data.yaml --model yolo11n.pt --epochs 100 --batch 16 --device 0

# CPU 训练（演示用）
python train/train_yolov11.py --epochs 1 --batch 2 --device cpu
```

训练结果保存在 `runs/sign_language/`。

**训练完成后复制最佳权重：**
```powershell
# 将 best.pt 复制到 weights/ 目录
copy runs\sign_language\<实验名称>\weights\best.pt weights\yolov11_best.pt
```

### 5.2 验证模型

```powershell
python train/val_yolov11.py --weights weights/yolov11_best.pt --data datasets/sign_language/data.yaml
```

输出：Precision、Recall、mAP@0.5、mAP@0.5:0.95，结果保存到 `outputs/csv/val_results.csv`。

---

## 6. 检测运行

### 6.1 图片检测

```powershell
python train/predict_image.py --weights weights/yolov11_best.pt --source test.jpg --conf 0.5 --save
```

### 6.2 视频检测

```powershell
python train/predict_video.py --weights weights/yolov11_best.pt --source video.mp4 --conf 0.5 --save
```

### 6.3 摄像头实时检测

```powershell
python app/main.py --weights weights/yolov11_best.pt --source 0 --conf 0.5 --speak --save
```

按 `Q` 键退出。退出后识别历史自动导出到 `outputs/csv/recognition_history.csv`。

参数说明：
- `--source 0`：默认摄像头（可改为 1、2 等）
- `--speak`：开启中文语音播报
- `--save`：保存检测视频
- `--top1`：只保留最高置信度的检测框

---

## 7. 图形界面

```powershell
python app/ui_qt.py
```

界面功能：
- 左侧：视频/摄像头画面显示区域（带 FPS 指示）
- 右侧：
  - 当前识别结果（中英文标签、置信度、坐标）
  - 置信度阈值调节滑块
  - 历史记录表格（最近 50 条）
- 底部控制栏：
  - 打开图片 / 打开视频 / 打开摄像头 / 停止检测
  - 保存结果 / 清空历史
  - 语音播报开关

---

## 8. 模型对比实验

```powershell
python train/compare_models.py --data datasets/sign_language/data.yaml --yolov8 weights/yolov8_best.pt --yolov11 weights/yolov11_best.pt
```

输出：
- `outputs/csv/model_comparison.csv`
- `outputs/figures/model_comparison.png`

---

## 9. 识别结果导出

识别历史记录默认保存到 `outputs/csv/recognition_history.csv`，格式：

```csv
time,label_en,label_cn,confidence,xmin,ymin,xmax,ymax
2025-01-15 10:30:25,thank you,谢谢,0.9234,120,80,340,300
2025-01-15 10:30:28,hello,你好,0.8912,80,100,280,340
```

---

## 10. 中文语义映射说明

`configs/label_map.json` 定义了英文标签到中文语义的映射。例如：

| 英文标签 | 中文含义 |
| --- | --- |
| thank you | 谢谢 |
| hello | 你好 |
| good | 好 |
| love | 爱 |
| help | 帮助 |
| friend | 朋友 |
| stop | 停止 |
| ... | ... |

用户可自行扩展映射表，格式为 JSON 对象。

---

## 11. 常见问题

| 问题 | 解决方案 |
| --- | --- |
| `ModuleNotFoundError: No module named 'ultralytics'` | 运行 `pip install -r requirements.txt` |
| 模型权重不存在 | 先训练模型，或下载预训练权重到 `weights/` |
| data.yaml 路径错误 | 检查 `datasets/sign_language/data.yaml` 中的路径是否正确 |
| 摄像头打不开 | 检查系统权限，或尝试 `--source 1` |
| 中文语音不工作 | Windows 需安装中文 TTS 声音；pyttsx3 不可用时会自动降级 |
| GUI 闪退 | 检查 PyQt5 是否正确安装：`pip install PyQt5` |
| GPU 显存不足 | 减小 `--batch`，如 `--batch 4` 或 `--batch 8` |

---

## 12. 相关文档

- [环境配置](docs/环境配置.md)
- [运行说明](docs/运行说明.md)
- [实验报告素材](docs/实验报告素材.md)
- [项目说明](docs/项目说明.md)

---

## 13. 致谢

本项目基于以下开源框架和数据集：

- [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)
- [PyQt5](https://riverbankcomputing.com/software/pyqt/)
- [pyttsx3](https://github.com/nateshmbhat/pyttsx3)
