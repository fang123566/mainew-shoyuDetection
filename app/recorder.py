"""识别历史记录与 CSV 导出模块。"""

from __future__ import annotations

import csv
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

try:
    from .config import OUTPUT_CSV_DIR
    from .detector import DetectionResult
except ImportError:  # 允许 python app/main.py 直接运行
    from config import OUTPUT_CSV_DIR
    from detector import DetectionResult


@dataclass
class RecognitionRecord:
    """一次有效识别记录。"""

    time: str
    label_en: str
    label_cn: str
    confidence: float
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    def to_dict(self) -> dict[str, str | int | float]:
        return asdict(self)


class RecognitionRecorder:
    """识别历史管理器，避免同一类别短时间内重复写入。"""

    def __init__(self, cooldown: float = 1.5, output_path: str | Path | None = None) -> None:
        self.cooldown = cooldown
        self.output_path = Path(output_path) if output_path else OUTPUT_CSV_DIR / "recognition_history.csv"
        self.records: list[RecognitionRecord] = []
        self._last_seen: dict[str, float] = {}

    def add_detection(self, detection: DetectionResult, label_cn: str) -> bool:
        """写入单条检测结果；冷却期内重复类别会被忽略。"""
        now = time.time()
        last_time = self._last_seen.get(detection.label_en, 0.0)
        if now - last_time < self.cooldown:
            return False

        record = RecognitionRecord(
            time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            label_en=detection.label_en,
            label_cn=label_cn,
            confidence=round(float(detection.confidence), 6),
            xmin=detection.xmin,
            ymin=detection.ymin,
            xmax=detection.xmax,
            ymax=detection.ymax,
        )
        self.records.append(record)
        self._last_seen[detection.label_en] = now
        return True

    def add_many(self, detections: Iterable[DetectionResult], translator) -> int:
        """批量写入检测结果，translator 需提供 translate_label 方法。"""
        added = 0
        for detection in detections:
            label_cn = translator.translate_label(detection.label_en)
            if self.add_detection(detection, label_cn):
                added += 1
        return added

    def recent(self, n: int = 10) -> list[RecognitionRecord]:
        """获取最近 N 条记录。"""
        return self.records[-n:]

    def clear(self) -> None:
        """清空历史和冷却状态。"""
        self.records.clear()
        self._last_seen.clear()

    def export_csv(self, output_path: str | Path | None = None) -> Path:
        """导出识别历史为 CSV。"""
        path = Path(output_path) if output_path else self.output_path
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["time", "label_en", "label_cn", "confidence", "xmin", "ymin", "xmax", "ymax"]
        with path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for record in self.records:
                writer.writerow(record.to_dict())
        return path

