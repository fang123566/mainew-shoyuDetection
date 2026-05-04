"""摄像头、图片和视频输入输出辅助模块。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

try:
    import cv2
except ImportError as exc:  # pragma: no cover - 运行时给出清晰错误
    cv2 = None
    _CV2_IMPORT_ERROR = exc
else:
    _CV2_IMPORT_ERROR = None

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    from .config import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
except ImportError:  # 允许 python app/main.py 直接运行
    from config import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS


@dataclass
class SourceInfo:
    """输入源类型描述。"""

    raw: str
    kind: str
    value: int | Path


def require_cv2() -> None:
    if cv2 is None:
        raise RuntimeError("未安装 opencv-python，请先运行: pip install -r requirements.txt") from _CV2_IMPORT_ERROR


def parse_source(source: str) -> SourceInfo:
    """将命令行 source 解析为 camera/image/video。"""
    if source.isdigit():
        return SourceInfo(raw=source, kind="camera", value=int(source))

    path = Path(source)
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return SourceInfo(raw=source, kind="image", value=path)
    if suffix in VIDEO_EXTENSIONS:
        return SourceInfo(raw=source, kind="video", value=path)
    raise ValueError(f"不支持的输入源: {source}。请传入摄像头编号、图片路径或视频路径。")


def open_capture(source: int | Path) -> object:
    """打开摄像头或视频文件。"""
    require_cv2()
    capture = cv2.VideoCapture(int(source) if isinstance(source, int) else str(source))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频源: {source}")
    return capture


def build_video_writer(output_path: Path, fps: float, frame_size: tuple[int, int]) -> object:
    """创建 mp4 视频写出器，失败时调用方会收到明确异常。"""
    require_cv2()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps if fps > 0 else 25.0, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"无法创建视频输出文件: {output_path}")
    return writer


def read_image(path: Path):
    """读取图片，兼容中文路径。"""
    require_cv2()
    if not path.exists():
        raise FileNotFoundError(f"图片不存在: {path}")
    if np is not None:
        data = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    else:
        image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"图片读取失败: {path}")
    return image


def resize_to_fit(frame, max_width: int = 1280, max_height: int = 720):
    """按比例限制显示尺寸，避免大图超出屏幕。"""
    height, width = frame.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale >= 1.0:
        return frame
    require_cv2()
    return cv2.resize(frame, (int(width * scale), int(height * scale)))

