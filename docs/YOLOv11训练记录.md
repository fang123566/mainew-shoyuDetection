# YOLOv11 训练与验证记录

## 训练环境

- 日期：2026-05-05
- 系统：Ubuntu WSL2
- Conda 环境：`ai_trading`
- Python：3.10.20
- PyTorch：2.0.0+cu117
- Ultralytics：8.4.46
- GPU：NVIDIA GeForce RTX 4060 Laptop GPU
- CUDA：11.7

## 数据集

- 配置文件：`datasets/sign_language/data.yaml`
- 类别数：35
- 训练集：2148 张图片，2148 个标注框
- 验证集：210 张图片，210 个标注框
- 说明：当前 `test` 路径复用 `val`，不作为独立测试集结论。

## 训练命令

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

## 训练结果

- 实际完成：63 epochs
- EarlyStopping：连续 30 epochs 未提升后停止
- 最佳模型 epoch：33
- Web 默认权重：`weights/yolov11_best.pt`
- 原始训练输出：`runs/sign_language/yolov11_sign_language_cuda/`

## 本轮真实验证

验证命令：

```bash
python train/val_yolov11.py --weights weights/yolov11_best.pt --data datasets/sign_language/data.yaml --imgsz 640 --batch 16 --device 0
```

验证结果：

| Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
| ---: | ---: | ---: | ---: |
| 0.9730 | 0.9831 | 0.9834 | 0.7986 |

验证输出：

- `outputs/csv/val_results.csv`
- `runs/sign_language/val_yolov11/`
- `new-shoyuDetection/static/analysis/`

## 图表生成

```bash
python scripts/generate_analysis.py --data datasets/sign_language/data.yaml --weights weights/yolov11_best.pt
```

图表使用 `seaborn` + `matplotlib` 生成，包含核心指标、类别分布、类别均衡、混淆矩阵和归一化混淆矩阵。
