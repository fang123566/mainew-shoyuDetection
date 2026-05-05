# 基于 YOLOv11 迁移训练与中文语义映射的多类别手语识别与无障碍交流辅助系统

> 计算机视觉课程设计项目。  
> 当前仓库已经完成 YOLOv11 GPU 迁移训练、HTML 演示界面、命令行检测、中文语义映射、语音播报、历史记录、CSV 导出、数据集检查和报告素材整理。

## 1. 给新成员的快速结论

这个项目不是一个单独的模型 Demo，而是一个可训练、可验证、可演示、可写报告的手语识别课程设计系统。

当前推荐演示方式：

```bash
# Ubuntu / WSL2
cd /mnt/e/python/CVandPR/mainew-shoyuDetection
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ai_trading
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
python new-shoyuDetection/app.py
```

浏览器打开：

```text
http://localhost:5000
```

HTML 演示页支持：

- 单张图片检测
- 批量上传图片测试
- 摄像头实时检测
- 后台分段识别，降低前端动态画面卡顿
- 英文标签、中文含义、置信度展示
- 简单短语拼接
- 保存检测结果

如果只是命令行演示摄像头：

```bash
python app/main.py --weights weights/yolov11_best.pt --source 0 --conf 0.5 --speak --save
```

## 2. 项目背景

手语是听障群体重要的沟通方式。传统手语识别系统通常存在几个问题：

- 只做离线图片分类，缺少实时演示能力；
- 识别结果停留在英文标签，不适合中文使用场景；
- 缺少语音播报、历史记录、CSV 导出等辅助交流功能；
- 训练、验证、对比实验流程不完整，答辩时难以解释模型效果；
- 代码结构混乱，团队成员接手成本高。

本项目基于 Ultralytics YOLOv11 进行迁移训练，将手语动作作为目标检测任务处理，并在识别后加入中文语义映射、简单短语拼接、语音播报和结果记录模块，形成一个完整的无障碍交流辅助系统原型。

## 3. 当前模型与训练结果

已训练好的权重位于：

```text
weights/yolov11_best.pt
```

训练环境：

| 项目 | 配置 |
| --- | --- |
| 系统 | Ubuntu WSL2 |
| Conda 环境 | `ai_trading` |
| Python | 3.10.20 |
| PyTorch | 2.0.0+cu117 |
| Ultralytics | 8.4.46 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| CUDA | 11.7 |

训练命令：

```bash
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

训练结果：

| 指标 | 数值 |
| --- | --- |
| 实际完成 epoch | 63 |
| 最佳 epoch | 33 |
| Precision | 0.9730 |
| Recall | 0.9831 |
| mAP@0.5 | 0.9834 |
| mAP@0.5:0.95 | 0.7986 |

详细记录见：

```text
docs/YOLOv11训练记录.md
outputs/csv/val_results.csv
runs/sign_language/yolov11_sign_language_cuda/results.csv
```

团队成员如果只是做演示或写报告，一般不需要重新训练。

## 4. 项目结构

```text
mainew-shoyuDetection/
├── app/                         # 重构后的核心检测模块
│   ├── main.py                  # 命令行入口：图片、视频、摄像头检测
│   ├── detector.py              # YOLO 模型加载和结构化推理结果
│   ├── camera.py                # 图片、视频、摄像头读取工具
│   ├── translator.py            # 英文标签到中文语义映射
│   ├── speech.py                # 中文语音播报
│   ├── recorder.py              # 识别历史和 CSV 导出
│   ├── config.py                # 全局路径配置
│   └── ui_qt.py                 # PyQt5 桌面界面
│
├── train/                       # 训练、验证、推理、模型对比脚本
│   ├── train_yolov11.py
│   ├── val_yolov11.py
│   ├── predict_image.py
│   ├── predict_video.py
│   └── compare_models.py
│
├── scripts/
│   └── check_dataset.py         # YOLO 数据集检查和类别统计
│
├── configs/
│   └── label_map.json           # 英文标签到中文含义映射表
│
├── datasets/
│   └── sign_language/data.yaml  # 当前训练使用的数据集配置
│
├── weights/
│   └── yolov11_best.pt          # 已训练好的 YOLOv11 权重
│
├── outputs/                     # 检测结果、CSV、图表输出
│   ├── csv/
│   ├── figures/
│   ├── images/
│   └── videos/
│
├── docs/                        # 课程设计说明和报告素材
│   ├── YOLOv11训练记录.md
│   ├── 环境配置.md
│   ├── 运行说明.md
│   ├── 实验报告素材.md
│   └── 项目说明.md
│
├── new-shoyuDetection/          # 原项目与当前 HTML 演示入口
│   ├── app.py                   # Flask HTML 演示后端，当前答辩推荐入口
│   ├── templates/index.html     # HTML 演示界面
│   └── datasets/                # 实际训练图片和标签所在目录
│
├── requirements.txt
├── README.md
└── .gitignore
```

说明：

- `app/` 是重构后的通用检测能力，适合命令行、PyQt 和后续二次开发。
- `new-shoyuDetection/app.py` 是当前 HTML 页面演示后端，已经接入 `weights/yolov11_best.pt` 和中文映射。
- `new-shoyuDetection/datasets/` 是当前 `data.yaml` 指向的真实训练数据目录。

## 5. 环境安装

### 5.1 Windows 普通环境

适合运行命令行、PyQt、基础推理：

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 5.2 Conda 环境

```bash
conda create -n ai_trading python=3.10 -y
conda activate ai_trading
pip install -r requirements.txt
```

### 5.3 Ubuntu / WSL2 + CUDA 环境

本仓库训练时使用的是 Ubuntu WSL2 的 `ai_trading` 环境：

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ai_trading
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

如果输出 `True` 并显示显卡名称，说明 GPU 可用。

## 6. 数据集说明

当前数据集配置文件：

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

YOLO 标签格式：

```text
class_id x_center y_center width height
```

其中坐标均为 0 到 1 的归一化值。

检查数据集：

```bash
python scripts/check_dataset.py --data datasets/sign_language/data.yaml
```

输出文件：

```text
outputs/csv/dataset_summary.csv
outputs/figures/class_distribution.png
```

## 7. 支持的识别标签

模型当前训练了 35 个类别：

```text
time, you/your/this, morning, 9, 0, happy, new, wish, please, road,
birthday, flat, safe, friend, 8, know, business card, marry/wife, tea,
have, flavor, today, door, stop, thank you, slow, walk, late/night,
I/me, love, good, person, what, name, introduce
```

中文映射位于：

```text
configs/label_map.json
```

核心映射示例：

| 英文标签 | 中文含义 |
| --- | --- |
| time | 时间 |
| you/your/this | 你/你的/这个 |
| morning | 早上 |
| happy | 开心 |
| please | 请 |
| friend | 朋友 |
| thank you | 谢谢 |
| I/me | 我 |
| love | 爱 |
| good | 好 |
| what | 什么 |
| name | 名字 |
| introduce | 介绍 |

`label_map.json` 中也预留了 `hello`、`help`、`yes`、`no`、`sorry`、`eat`、`drink`、`home`、`school`、`doctor`、`mother`、`father` 等扩展映射。注意：预留映射不代表模型已经训练这些类别，模型能识别的类别以 `data.yaml` 的 35 类为准。

## 8. HTML 演示界面

当前答辩推荐使用 HTML 演示界面。

启动：

```bash
cd /mnt/e/python/CVandPR/mainew-shoyuDetection
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ai_trading
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
python new-shoyuDetection/app.py
```

浏览器打开：

```text
http://localhost:5000
```

功能：

| 功能 | 说明 |
| --- | --- |
| 图片检测 | 上传单张图片，返回带框结果 |
| 批量测试 | 一次选择多张图片，批量返回结果缩略图和表格 |
| 摄像头检测 | 浏览器调用摄像头，实时显示画面 |
| 后台分段识别 | 摄像头画面持续渲染，后台定时抽帧检测，减少卡顿 |
| 中文映射 | 显示英文标签和中文含义 |
| 简单短语 | 根据连续识别结果拼接简单短语 |
| 保存结果 | 将检测结果保存到 `outputs/images/` |

停止服务：

```bash
# WSL / Ubuntu 中查看 5000 端口
ss -ltnp | grep ':5000'

# 结束对应进程
kill <pid>
```

## 9. 命令行检测

### 9.1 图片检测

```bash
python app/main.py --weights weights/yolov11_best.pt --source path/to/image.jpg --conf 0.5 --save
```

或：

```bash
python train/predict_image.py --weights weights/yolov11_best.pt --source path/to/image.jpg --conf 0.5 --save
```

### 9.2 视频检测

```bash
python app/main.py --weights weights/yolov11_best.pt --source path/to/video.mp4 --conf 0.5 --save
```

或：

```bash
python train/predict_video.py --weights weights/yolov11_best.pt --source path/to/video.mp4 --conf 0.5 --save
```

### 9.3 摄像头检测

```bash
python app/main.py --weights weights/yolov11_best.pt --source 0 --conf 0.5 --speak --save
```

常用参数：

| 参数 | 说明 |
| --- | --- |
| `--weights` | 模型权重路径 |
| `--source` | `0` 表示默认摄像头，也可以是图片或视频路径 |
| `--conf` | 置信度阈值 |
| `--iou` | NMS IoU 阈值 |
| `--speak` | 开启中文语音播报 |
| `--save` | 保存检测结果 |
| `--top1` | 只保留最高置信度目标 |

退出后，历史记录会导出到：

```text
outputs/csv/recognition_history.csv
```

## 10. PyQt 桌面界面

运行：

```bash
python app/ui_qt.py
```

功能：

- 打开图片
- 打开视频
- 打开摄像头
- 停止检测
- 保存结果
- 清空历史
- 开启或关闭语音播报
- 调节置信度阈值
- 查看最近识别历史

如果 PyQt5 缺失：

```bash
pip install PyQt5
```

## 11. 训练与验证

### 11.1 重新训练 YOLOv11

通常不需要重新训练。确实需要时，在 Ubuntu CUDA 环境中运行：

```bash
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

训练输出：

```text
runs/sign_language/<实验名称>/
```

训练完成后，如需更新默认演示权重：

```bash
cp runs/sign_language/<实验名称>/weights/best.pt weights/yolov11_best.pt
```

### 11.2 验证模型

```bash
python train/val_yolov11.py \
  --weights weights/yolov11_best.pt \
  --data datasets/sign_language/data.yaml \
  --device 0
```

验证输出：

```text
outputs/csv/val_results.csv
```

### 11.3 YOLOv8 与 YOLOv11 对比

```bash
python train/compare_models.py \
  --data datasets/sign_language/data.yaml \
  --yolov8 weights/yolov8_best.pt \
  --yolov11 weights/yolov11_best.pt
```

输出：

```text
outputs/csv/model_comparison.csv
outputs/figures/model_comparison.png
```

如果 `weights/yolov8_best.pt` 不存在，脚本会给出提示，不会直接崩溃。

## 12. 识别效果不好时怎么办

真实摄像头效果可能弱于验证集指标，原因通常是训练集和现场画面存在差异：

- 摄像头角度、手部大小、背景、光照和训练图片不同；
- 动态手势停留时间太短，模型看到的是模糊过渡帧；
- `yolo11n` 是轻量模型，速度快但容量有限；
- 部分类别手势形态相似，容易混淆。

演示建议：

- 置信度阈值先用 `0.25` 到 `0.40`，不要一开始就设太高；
- 手势尽量在画面中央，手部占画面 1/4 到 1/2；
- 背景保持干净，光线充足；
- 每个手势停留 1 到 2 秒，再切换下一个；
- 使用批量上传功能先测试样例图片，确认模型和页面正常；
- 需要明显提升效果时，采集自己摄像头画面重新标注并微调。

## 13. 输出文件说明

| 路径 | 说明 |
| --- | --- |
| `outputs/images/` | 图片和 HTML 检测结果 |
| `outputs/videos/` | 视频或摄像头保存结果 |
| `outputs/csv/recognition_history.csv` | 识别历史 |
| `outputs/csv/val_results.csv` | 验证指标 |
| `outputs/csv/model_comparison.csv` | 模型对比指标 |
| `outputs/figures/class_distribution.png` | 类别分布图 |
| `outputs/figures/model_comparison.png` | 模型对比图 |

## 14. Git 协作规范

请遵守以下规则，避免把仓库搞大或覆盖队友工作：

- 不要提交大规模数据集、训练缓存和临时视频；
- 不要执行 `git reset --hard`、`git clean -fd` 等危险命令；
- 修改前先 `git status`；
- 提交信息使用清晰的英文规范，例如：
  - `feat: add batch image upload testing`
  - `fix: correct dataset config`
  - `docs: update new member guide`
- 不要自动强推远程分支；
- 如果重新训练模型，先和团队确认是否需要提交新权重。

推荐提交流程：

```bash
git status
git add README.md
git commit -m "docs: update new member guide"
git push origin main
```

## 15. 常见问题

| 问题 | 处理方式 |
| --- | --- |
| `ModuleNotFoundError: ultralytics` | 运行 `pip install -r requirements.txt` |
| `weights/yolov11_best.pt` 不存在 | 拉取最新仓库，或重新训练后复制 `best.pt` |
| HTML 页面打不开 | 确认 `python new-shoyuDetection/app.py` 正在运行，端口是 5000 |
| 5000 端口被占用 | 用 `ss -ltnp | grep ':5000'` 或 Windows 端口工具找到进程后结束 |
| 摄像头打不开 | 检查浏览器权限、系统摄像头权限，或关闭其他占用摄像头的软件 |
| 中文语音不播报 | pyttsx3 或系统中文语音可能缺失，主程序会降级，不影响检测 |
| GPU 不可用 | 检查 CUDA、PyTorch 版本和 `torch.cuda.is_available()` |
| 检测结果全是英文 | 检查 `configs/label_map.json` 是否存在且编码为 UTF-8 |
| 批量检测慢 | 批量图片数量过多或分辨率过高，先用 5 到 20 张样例测试 |

## 16. 课程报告素材

写报告或做答辩 PPT 时，优先参考：

```text
docs/实验报告素材.md
docs/YOLOv11训练记录.md
outputs/csv/val_results.csv
outputs/figures/class_distribution.png
```

建议答辩讲解顺序：

1. 项目背景：无障碍交流和手语识别需求；
2. 技术路线：YOLOv11 目标检测 + 中文语义映射 + 语音播报；
3. 数据集：35 类手语，YOLO 格式标注；
4. 训练过程：迁移训练、GPU 环境、EarlyStopping；
5. 实验指标：Precision、Recall、mAP；
6. 系统演示：图片、批量测试、摄像头；
7. 工程亮点：模块化封装、历史记录、CSV 导出、HTML 演示；
8. 不足与改进：真实场景泛化、动态手势建模、更大模型或补充数据。

## 17. 依赖框架

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)
- [PyTorch](https://pytorch.org/)
- [Flask](https://flask.palletsprojects.com/)
- [PyQt5](https://riverbankcomputing.com/software/pyqt/)
- [pyttsx3](https://github.com/nateshmbhat/pyttsx3)
