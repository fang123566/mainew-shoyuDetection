"""中文语音播报模块。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class SpeechEngine:
    """pyttsx3 语音播报封装，支持开关和重复播报冷却。"""

    enabled: bool = False
    cooldown: float = 2.0
    rate: int = 180
    volume: float = 1.0
    _engine: object | None = field(default=None, init=False, repr=False)
    _last_text: str = field(default="", init=False)
    _last_time: float = field(default=0.0, init=False)
    _available: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._engine = self._init_engine()

    def _init_engine(self) -> object | None:
        try:
            import pyttsx3

            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
            engine.setProperty("volume", self.volume)
            self._select_chinese_voice(engine)
            self._available = True
            return engine
        except Exception as exc:
            self._available = False
            print(f"[WARN] pyttsx3 初始化失败，语音播报已降级关闭: {exc}")
            return None

    def _select_chinese_voice(self, engine: object) -> None:
        """尽量选择中文语音；系统没有中文声音时保持默认声音。"""
        try:
            voices = engine.getProperty("voices") or []
            keywords = ("zh", "chinese", "mandarin", "huihui", "kangkang", "yaoyao", "hanhan")
            for voice in voices:
                voice_id = str(getattr(voice, "id", "")).lower()
                voice_name = str(getattr(voice, "name", "")).lower()
                if any(keyword in voice_id or keyword in voice_name for keyword in keywords):
                    engine.setProperty("voice", getattr(voice, "id"))
                    return
        except Exception:
            return

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled

    def speak(self, text: str) -> bool:
        """播报文本；返回是否实际触发播报。"""
        text = text.strip()
        if not text or not self.enabled or not self._available or self._engine is None:
            return False

        now = time.time()
        if text == self._last_text and now - self._last_time < self.cooldown:
            return False

        try:
            self._engine.say(text)
            self._engine.runAndWait()
            self._last_text = text
            self._last_time = now
            return True
        except Exception as exc:
            print(f"[WARN] 语音播报失败，已自动关闭语音: {exc}")
            self.enabled = False
            self._available = False
            return False

