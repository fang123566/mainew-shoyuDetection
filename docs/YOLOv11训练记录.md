# YOLOv11 迁移训练记录

本文档记录 2026 年 5 月 5 日下午完成的 YOLOv11 手语识别模型训练过程，可直接作为课程报告中“实验环境、训练过程与结果分析”章节的依据。

## 1. 训练目标

本次训练的目标是将原项目中的 YOLOv8/旧版检测流程迁移到 YOLOv11，并生成团队可直接复用的手语识别权重，避免每位成员重复训练。

训练完成后，模型需要满足以下要求：

- 能识别 35 类手语动作词；
- 使用 Ultralytics YOLOv11 训练流程；
- 支持 Ubuntu CUDA GPU 训练；
- 输出可复用的 `weights/yolov11_best.pt`；
- 生成验证指标和训练记录，供报告和 PPT 使用；
- 能接入命令行、PyQt 和 HTML 演示页面。

## 2. 训练环境

| 项目 | 配置 |
| --- | --- |
| 训练日期 | 2026-05-05 |
| 操作系统 | Ubuntu WSL2 |
| Conda 环境 | `ai_trading` |
| Python | 3.10.20 |
| PyTorch | 2.0.0+cu117 |
| CUDA | 11.7 |
| Ultralytics | 8.4.46 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| 训练设备参数 | `--device 0` |

训练前使用如下方式确认 GPU 可用：

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ai_trading
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

说明：本项目在 Windows 仓库目录中开发，在 Ubuntu WSL2 中使用 CUDA 训练。WSL 访问仓库路径为：

```bash
/mnt/e/python/CVandPR/mainew-shoyuDetection
```

## 3. 数据集配置

数据集配置文件：

```text
datasets/sign_language/data.yaml
```

当前配置：

```yaml
path: .
train: new-shoyuDetection/datasets/images/train
val: new-shoyuDetection/datasets/images/val
test: new-shoyuDetection/datasets/images/val
nc: 35
```

数据规模：

| 集合 | 路径 | 图片数量 |
| --- | --- | --- |
| 训练集 | `new-shoyuDetection/datasets/images/train` | 2148 |
| 验证集 | `new-shoyuDetection/datasets/images/val` | 210 |
| 测试集 | 当前复用验证集 | 210 |

类别数量：35 类。

YOLO 标签格式：

```text
class_id x_center y_center width height
```

其中 `x_center`、`y_center`、`width`、`height` 均为 0 到 1 之间的归一化坐标。

## 4. 训练命令

从仓库根目录运行：

```bash
cd /mnt/e/python/CVandPR/mainew-shoyuDetection
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ai_trading
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

python train/train_yolov11.py \
  --data datasets/sign_language/data.yaml \
  --model yolo11n.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --device 0 \
  --workers 4 \
  --name yolov11_sign_language_cuda \
  --patience 30
```

## 5. 关键训练参数

| 参数 | 值 | 说明 |
| --- | --- | --- |
| 预训练模型 | `yolo11n.pt` | YOLOv11 nano 版本，适合课程演示和实时推理 |
| 输入尺寸 | 640 | 统一缩放到 640 x 640 |
| Batch Size | 16 | RTX 4060 Laptop GPU 可承受 |
| Epochs | 100 | 设定最大训练轮数 |
| Patience | 30 | 验证指标连续 30 轮不提升则早停 |
| Workers | 4 | Ubuntu 环境下开启多进程读取 |
| AMP | true | 使用混合精度训练 |
| Optimizer | auto | Ultralytics 自动选择 |
| 初始学习率 | 0.01 | 来自训练参数 `lr0` |
| 数据增强 | 默认开启 | Mosaic、HSV、Flip 等 |

训练输出目录：

```text
runs/sign_language/yolov11_sign_language_cuda/
```

关键输出文件：

| 文件 | 作用 |
| --- | --- |
| `args.yaml` | 训练参数记录 |
| `results.csv` | 每轮训练和验证指标 |
| `weights/best.pt` | 验证集上表现最优的权重 |
| `weights/last.pt` | 最后一轮权重 |

## 6. 训练过程

本次训练设置最大 100 轮，实际在第 63 轮触发 EarlyStopping 停止。最佳模型出现在第 33 轮。

训练过程中的指标变化可以概括为：

- 第 1 轮时模型还处于迁移学习初期，mAP@0.5 约为 0.4079；
- 第 2 到第 3 轮指标快速上升，mAP@0.5 已达到 0.97 左右，说明预训练模型迁移效果明显；
- 第 20 到第 50 轮进入稳定优化阶段，mAP@0.5:0.95 在 0.79 到 0.80 附近波动；
- 第 33 轮取得训练日志中的最佳 mAP@0.5:0.95；
- 第 63 轮由于后续 30 轮没有显著提升，触发早停，避免无效训练和过拟合。

训练日志中的最佳轮次：

| Epoch | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
| --- | --- | --- | --- | --- |
| 33 | 0.97297 | 0.98311 | 0.98406 | 0.79982 |

独立验证脚本结果以 `outputs/csv/val_results.csv` 为准：

| Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
| --- | --- | --- | --- |
| 0.9730 | 0.9831 | 0.9834 | 0.7986 |

## 7. 最终权重

训练完成后，将最佳权重复制为项目默认演示权重：

```text
weights/yolov11_best.pt
```

同时保留最后一轮权重：

```text
weights/yolov11_last.pt
```

权重文件大小约为 5.47 MB，适合课程项目提交和本地实时演示。

## 8. 验证命令

```bash
python train/val_yolov11.py \
  --weights weights/yolov11_best.pt \
  --data datasets/sign_language/data.yaml \
  --imgsz 640 \
  --batch 16 \
  --device 0
```

验证输出：

```text
outputs/csv/val_results.csv
```

## 9. 结果分析

从验证指标看，本模型在当前验证集上达到了较高的检测精度：

- Precision 为 0.9730，说明模型预测出的手语目标大部分是正确的；
- Recall 为 0.9831，说明验证集中绝大多数目标都能被检测到；
- mAP@0.5 为 0.9834，说明在较宽松 IoU 阈值下检测效果优秀；
- mAP@0.5:0.95 为 0.7986，说明在更严格定位标准下仍有较好的框定位能力。

需要注意的是，验证集指标不等于真实摄像头演示效果。摄像头场景会受到背景、光照、手部大小、运动模糊和手势停留时间影响。因此演示时建议降低置信度阈值到 0.25 到 0.40，并让手势在画面中央停留 1 到 2 秒。

## 10. 报告可引用结论

本项目完成了从 YOLOv8/旧版手语检测项目到 YOLOv11 的迁移训练。训练在 Ubuntu WSL2 的 CUDA 环境下进行，使用 `yolo11n.pt` 作为预训练模型，训练集 2148 张、验证集 210 张，共 35 类手语动作词。模型在第 33 轮取得最佳效果，最终独立验证 Precision 为 0.9730，Recall 为 0.9831，mAP@0.5 为 0.9834，mAP@0.5:0.95 为 0.7986。训练好的 `weights/yolov11_best.pt` 已接入命令行检测、HTML 演示界面和中文语义映射模块，可直接用于课程答辩演示。
