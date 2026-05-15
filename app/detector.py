"""YOLO 手语检测后端封装。

该模块只负责模型加载和推理结果结构化，不处理界面、语音和历史记录，方便 CLI 与 GUI 复用。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class DetectionResult:
    """单个检测框的结构化结果。"""

    class_id: int
    label_en: str
    confidence: float
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


class SignLanguageDetector:
    """YOLO 模型加载与推理统一入口。"""

    def __init__(
        self,
        weights: str | Path,
        conf: float = 0.5,
        iou: float = 0.45,
        device: str | None = None,
    ) -> None:
        self.weights = Path(weights)
        self.conf = conf
        self.iou = iou
        self.device = device
        self.model = self._load_model()
        self.names = self._load_names()

    def _load_model(self) -> Any:
        if not self.weights.exists():
            raise FileNotFoundError(
                f"模型权重不存在: {self.weights}\n"
                "请先训练模型，或通过 --weights 指定有效的 best.pt。"
            )
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("未安装 ultralytics，请先运行: pip install -r requirements.txt") from exc

        try:
            return YOLO(str(self.weights))
        except Exception as exc:
            raise RuntimeError(f"模型加载失败: {self.weights}，原因: {exc}") from exc

    def _load_names(self) -> dict[int, str]:
        names = getattr(self.model, "names", {}) or {}
        if isinstance(names, dict):
            return {int(key): str(value) for key, value in names.items()}
        if isinstance(names, list):
            return {index: str(value) for index, value in enumerate(names)}
        return {}

    def predict_frame(
        self,
        frame: Any,
        conf: float | None = None,
        iou: float | None = None,
        top1: bool = False,
    ) -> list[DetectionResult]:
        """对视频帧或摄像头帧推理，返回结构化检测结果。"""
        results = self.model.predict(
            source=frame,
            conf=self.conf if conf is None else conf,
            iou=self.iou if iou is None else iou,
            device=self.device,
            verbose=False,
        )
        return self._parse_results(results[0], top1=top1)

    def predict_image(
        self,
        image_path: str | Path,
        conf: float | None = None,
        iou: float | None = None,
        top1: bool = False,
    ) -> list[DetectionResult]:
        """对图片路径推理，兼容 Windows 中文路径由 ultralytics 处理。"""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"图片不存在: {path}")
        results = self.model.predict(
            source=str(path),
            conf=self.conf if conf is None else conf,
            iou=self.iou if iou is None else iou,
            device=self.device,
            verbose=False,
        )
        return self._parse_results(results[0], top1=top1)

    def predict_roi(
        self,
        frame: Any,
        roi_bbox: tuple[int, int, int, int],
        conf: float | None = None,
        iou: float | None = None,
        top1: bool = False,
    ) -> list[DetectionResult]:
        """对 ROI 区域进行推理，并将检测框坐标映射回原图。

        Args:
            frame: 完整帧图像（BGR）。
            roi_bbox: ROI 边界框 (x1, y1, x2, y2)。
            conf: 置信度阈值（None 时使用构造时默认值）。
            iou: NMS IoU 阈值（None 时使用构造时默认值）。
            top1: 是否只返回最高置信度结果。

        Returns:
            映射到原图坐标系的 DetectionResult 列表。
        """
        x1, y1, x2, y2 = roi_bbox
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return []

        roi_detections = self.predict_frame(roi, conf=conf, iou=iou, top1=top1)
        return self._remap_boxes(roi_detections, roi_bbox)

    def _remap_boxes(
        self,
        detections: list[DetectionResult],
        roi_bbox: tuple[int, int, int, int],
    ) -> list[DetectionResult]:
        """将 ROI 内检测框的坐标映射回原图坐标系。

        Args:
            detections: ROI 内检测结果（坐标相对于 ROI 左上角）。
            roi_bbox: ROI 在原图中的边界框 (x1, y1, x2, y2)。

        Returns:
            坐标已映射到原图的 DetectionResult 列表。
        """
        if not detections:
            return []
        x1, y1, _, _ = roi_bbox
        remapped = []
        for det in detections:
            remapped.append(
                DetectionResult(
                    class_id=det.class_id,
                    label_en=det.label_en,
                    confidence=det.confidence,
                    xmin=det.xmin + x1,
                    ymin=det.ymin + y1,
                    xmax=det.xmax + x1,
                    ymax=det.ymax + y1,
                )
            )
        return remapped

    def _parse_results(self, result: Any, top1: bool = False) -> list[DetectionResult]:
        """解析 YOLO 推理结果为 DetectionResult 列表。"""
        detections: list[DetectionResult] = []
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        cls_values = boxes.cls.cpu().numpy()
        conf_values = boxes.conf.cpu().numpy()

        for box, cls_value, conf_value in zip(xyxy, cls_values, conf_values):
            class_id = int(cls_value)
            xmin, ymin, xmax, ymax = [int(round(value)) for value in box.tolist()]
            detections.append(
                DetectionResult(
                    class_id=class_id,
                    label_en=self.names.get(class_id, str(class_id)),
                    confidence=float(conf_value),
                    xmin=xmin,
                    ymin=ymin,
                    xmax=xmax,
                    ymax=ymax,
                )
            )

        detections.sort(key=lambda item: item.confidence, reverse=True)
        return detections[:1] if top1 and detections else detections

