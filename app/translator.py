"""英文手语类别到中文语义的映射模块。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

try:
    from .config import DEFAULT_LABEL_MAP
except ImportError:  # 允许 python app/main.py 直接运行
    from config import DEFAULT_LABEL_MAP


class LabelTranslator:
    """从 JSON 映射表读取英文类别名，并转换为中文语义。"""

    def __init__(self, label_map_path: str | Path = DEFAULT_LABEL_MAP) -> None:
        self.label_map_path = Path(label_map_path)
        self.label_map = self._load_label_map()

    def _load_label_map(self) -> dict[str, str]:
        if not self.label_map_path.exists():
            print(f"[WARN] 未找到中文映射表: {self.label_map_path}，将使用英文标签。")
            return {}
        try:
            with self.label_map_path.open("r", encoding="utf-8") as file:
                raw = json.load(file)
            return {str(key): str(value) for key, value in raw.items()}
        except Exception as exc:
            print(f"[WARN] 中文映射表读取失败: {exc}，将使用英文标签。")
            return {}

    def translate_label(self, label_en: str) -> str:
        """返回中文语义；映射不存在时返回英文原标签。"""
        return self.label_map.get(label_en, label_en)

    def build_sentence(self, history: Sequence[object], max_items: int = 5) -> str:
        """根据最近识别历史拼接简单中文短语。

        history 可以是 recorder 中的记录 dict，也可以是 DetectionResult 或普通字符串。
        """
        labels: list[str] = []
        for item in list(history)[-max_items:]:
            label_cn = self._extract_cn_label(item)
            if label_cn and (not labels or labels[-1] != label_cn):
                labels.append(label_cn)
        return " ".join(labels)

    def _extract_cn_label(self, item: object) -> str:
        if isinstance(item, str):
            return self.translate_label(item)
        if isinstance(item, Mapping):
            if item.get("label_cn"):
                return str(item["label_cn"])
            if item.get("label_en"):
                return self.translate_label(str(item["label_en"]))
        label_cn = getattr(item, "label_cn", None)
        if label_cn:
            return str(label_cn)
        label_en = getattr(item, "label_en", None)
        if label_en:
            return self.translate_label(str(label_en))
        return ""


_DEFAULT_TRANSLATOR = LabelTranslator()


def translate_label(label_en: str) -> str:
    """便捷函数：英文类别名转中文。"""
    return _DEFAULT_TRANSLATOR.translate_label(label_en)


def build_sentence(history: Iterable[object]) -> str:
    """便捷函数：基于最近识别历史拼接短语。"""
    return _DEFAULT_TRANSLATOR.build_sentence(list(history))

