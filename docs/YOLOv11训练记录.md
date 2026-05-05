# YOLOv11 训练记录

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
- 训练集：`new-shoyuDetection/datasets/images/train`
- 验证集：`new-shoyuDetection/datasets/images/val`
- 类别数：35
- 训练图片数：2148
- 验证图片数：210

## 训练命令

从仓库根目录运行：

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
- 最佳权重：`weights/yolov11_best.pt`
- 原始训练输出：`runs/sign_language/yolov11_sign_language_cuda/`

## 验证指标

使用 `weights/yolov11_best.pt` 在验证集上评估：

| Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
| --- | --- | --- | --- |
| 0.9730 | 0.9831 | 0.9834 | 0.7986 |

验证 CSV：`outputs/csv/val_results.csv`

## 复用方式

队友拉取仓库后无需重新训练，直接使用：

```bash
python app/main.py --weights weights/yolov11_best.pt --source 0 --conf 0.5
```

或验证模型：

```bash
python train/val_yolov11.py --weights weights/yolov11_best.pt --data datasets/sign_language/data.yaml --device 0
```

