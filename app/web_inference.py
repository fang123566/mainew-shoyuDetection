"""Web 演示推理服务层。

该模块把 Flask 页面需要的模型加载、推理、标签映射和结果结构化集中起来，
避免 Web 后端重复维护 YOLO 解析逻辑。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np


class WebInferenceService:
    """面向 Web 演示的 YOLO 推理统一入口。"""

    def __init__(self, weights: str | Path, label_map_path: str | Path, device: str | None = None) -> None:
        self.weights = Path(weights)
        self.label_map_path = Path(label_map_path)
        self.device = device or self._auto_device()
        self.label_map = self._load_label_map()
        self._model: Any | None = None
        self._model_names: dict[int, str] = {}

    def _auto_device(self) -> str:
        try:
            import torch

            return "0" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _load_label_map(self) -> dict[str, str]:
        if not self.label_map_path.exists():
            return {}
        try:
            with self.label_map_path.open("r", encoding="utf-8") as file:
                raw = json.load(file)
            return {str(key): str(value) for key, value in raw.items()}
        except Exception:
            return {}

    def load_model(self) -> Any:
        if self._model is not None:
            return self._model
        if not self.weights.exists():
            raise FileNotFoundError(f"未找到 YOLOv11 权重: {self.weights}")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("未安装 ultralytics，请先运行: pip install -r requirements.txt") from exc

        self._model = YOLO(str(self.weights), task="detect")
        names = getattr(self._model, "names", {}) or {}
        if isinstance(names, dict):
            self._model_names = {int(key): str(value) for key, value in names.items()}
        else:
            self._model_names = {index: str(value) for index, value in enumerate(names)}
        return self._model

    @property
    def model(self) -> Any:
        return self.load_model()

    @property
    def model_names(self) -> dict[int, str]:
        self.load_model()
        return dict(self._model_names)

    def warmup(self) -> None:
        """预热模型，减少第一次 Web 请求的延迟。"""
        self.model(np.zeros((48, 48, 3), dtype=np.uint8), device=self.device, verbose=False)

    def label_cn(self, label_en: str) -> str:
        return self.label_map.get(label_en, label_en)

    def predict_image(self, image_path: str | Path, conf: float, iou: float) -> tuple[Any, float]:
        start = time.perf_counter()
        result = self.model.predict(
            source=str(image_path),
            conf=conf,
            iou=iou,
            device=self.device,
            verbose=False,
        )[0]
        return result, time.perf_counter() - start

    def predict_frame(self, frame: Any, conf: float, iou: float, imgsz: int | None = None) -> tuple[Any, float]:
        kwargs: dict[str, Any] = {
            "source": frame,
            "conf": conf,
            "iou": iou,
            "device": self.device,
            "verbose": False,
        }
        if imgsz is not None:
            kwargs["imgsz"] = imgsz
        start = time.perf_counter()
        result = self.model.predict(**kwargs)[0]
        return result, time.perf_counter() - start

    def extract_detections(
        self,
        result: Any,
        *,
        include_id: bool = True,
        confidence_percent: bool = True,
    ) -> list[dict[str, object]]:
        detections: list[dict[str, object]] = []
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        clses = boxes.cls.cpu().numpy()
        confs = boxes.conf.cpu().numpy()

        for index, box in enumerate(xyxy):
            cls_id = int(clses[index])
            label_en = self.model_names.get(cls_id, str(cls_id))
            x1, y1, x2, y2 = [int(v) for v in box]
            confidence = float(confs[index])
            item: dict[str, object] = {
                "class_id": cls_id,
                "class": label_en,
                "class_cn": self.label_cn(label_en),
                "confidence": round(confidence * 100, 2) if confidence_percent else round(confidence, 6),
                "bbox": [x1, y1, x2, y2],
            }
            if include_id:
                item["id"] = index + 1
            detections.append(item)
        return detections

    def class_counts(self, result: Any) -> dict[str, int]:
        counts: dict[str, int] = {}
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return counts
        for cls_id in boxes.cls.cpu().numpy():
            name = self.model_names.get(int(cls_id), str(int(cls_id)))
            counts[name] = counts.get(name, 0) + 1
        return counts
