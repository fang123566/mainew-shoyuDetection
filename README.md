# 基于 YOLOv11 的多类别手语识别与无障碍交流辅助 Web 系统

本项目是计算机视觉课程设计项目，面向多类别手语动作检测与中文语义展示。系统以 YOLOv11 为核心检测模型，以 Flask Web 页面作为唯一推荐演示入口，支持图片、批量图片、视频文件、摄像头实时识别和模型评估分析。

## 项目亮点

- 使用 YOLOv11n 完成 35 类手语目标检测，默认权重为 `weights/yolov11_best.pt`。
- Web 端统一演示入口，后端复用 `app/` 下的模块化推理能力，减少重复检测逻辑。
- 评估页展示真实验证集指标、类别分布、类别均衡、混淆矩阵和问题样例摘要。
- 评估图表由 `seaborn` + `matplotlib` 生成，采用高清论文风格，静态资产已提交，开箱即可展示。
- 摄像头实时识别加入 ROI、滑动稳定窗口、重复去抖、短语编辑和置信度趋势。
- 协作基线已整理：数据集、训练输出、运行输出和临时权重不进入 Git。

## 快速运行

推荐在 WSL2 / Ubuntu 环境运行：

```bash
cd /mnt/e/python/CVandPR/mainew-shoyuDetection
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ai_trading
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
python new-shoyuDetection/app.py
```

浏览器打开：

```text
http://127.0.0.1:5000
```

Web 页面包含：

- 图片检测
- 批量图片检测
- 视频文件检测
- 摄像头实时识别
- 模型评估分析
- 项目说明与模型信息

## 真实验证结果

本轮使用当前 Web 默认权重 `weights/yolov11_best.pt`，在 `datasets/sign_language/data.yaml` 指定的验证集上重跑评估。验证集包含 210 张图片、210 个标注框。

| 指标 | 数值 |
| --- | ---: |
| Precision | 0.9730 |
| Recall | 0.9831 |
| mAP@0.5 | 0.9834 |
| mAP@0.5:0.95 | 0.7986 |

注意：当前 `data.yaml` 中 `test` 路径复用 `val`，因此最终报告应按验证集结果解读，不声称拥有独立测试集结论。

## 评估图表

精选评估资产位于 `new-shoyuDetection/static/analysis/`，Web 评估页会优先读取这些已提交图表：

- `metrics_summary.png`：核心指标汇总图
- `class_distribution.png`：训练集与验证集类别分布
- `class_balance_top_bottom.png`：样本最多与最少类别对比
- `confusion_matrix.png`：验证集混淆矩阵
- `confusion_matrix_normalized.png`：归一化混淆矩阵

本地重新生成图表：

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ai_trading
python train/val_yolov11.py --weights weights/yolov11_best.pt --data datasets/sign_language/data.yaml --imgsz 640 --batch 16 --device 0
python scripts/generate_analysis.py --data datasets/sign_language/data.yaml --weights weights/yolov11_best.pt
```

生成物会写入 `outputs/analysis/`，并同步精选资产到 `new-shoyuDetection/static/analysis/`。

## 目录结构

```text
app/                          # 模块化推理、翻译、记录和辅助入口
configs/                      # 中文标签映射和评估摘要配置
datasets/sign_language/       # 数据集配置和必要占位文件
new-shoyuDetection/           # Flask Web 演示入口和前端页面
new-shoyuDetection/static/analysis/  # 已提交的真实评估图表
scripts/                      # 数据检查与评估图表生成脚本
train/                        # 训练、验证、预测脚本
weights/yolov11_best.pt        # Web 默认演示权重
outputs/                      # 本地运行输出，不提交 Git
runs/                         # 训练/验证输出，不提交 Git
```

## 协作规范

- Web 是正式演示入口，PyQt/CLI 保留为辅助入口。
- 保留 `weights/yolov11_best.pt` 作为默认演示权重，其他训练过程权重不提交。
- 数据集图片、标注大文件、训练输出、运行输出、上传文件和视频结果不提交。
- 提交前检查：

```bash
git status -sb
python -m py_compile app/*.py scripts/generate_analysis.py new-shoyuDetection/app.py
```

## 答辩表达建议

可以把项目讲成一条完整链路：数据集整理与标注、YOLOv11 迁移训练、统一 Web 推理服务、真实验证评估、实时摄像头交互优化。重点强调本项目不是单张图片 Demo，而是可训练、可验证、可演示、可协作维护的课程系统。
