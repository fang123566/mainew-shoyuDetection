"""项目全局路径与运行参数配置。

所有路径都基于仓库根目录推导，避免在 Windows 或不同机器上出现绝对路径失效。
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

APP_DIR = PROJECT_ROOT / "app"
CONFIG_DIR = PROJECT_ROOT / "configs"
DATASET_DIR = PROJECT_ROOT / "datasets" / "sign_language"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_IMAGE_DIR = OUTPUT_DIR / "images"
OUTPUT_VIDEO_DIR = OUTPUT_DIR / "videos"
OUTPUT_LOG_DIR = OUTPUT_DIR / "logs"
OUTPUT_CSV_DIR = OUTPUT_DIR / "csv"
OUTPUT_FIGURE_DIR = OUTPUT_DIR / "figures"

DEFAULT_DATA_YAML = DATASET_DIR / "data.yaml"
DEFAULT_LABEL_MAP = CONFIG_DIR / "label_map.json"
DEFAULT_WEIGHTS = WEIGHTS_DIR / "yolov11_best.pt"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}


def ensure_output_dirs() -> None:
    """创建运行输出目录，便于脚本首次运行时直接写入结果。"""
    for path in [
        OUTPUT_DIR,
        OUTPUT_IMAGE_DIR,
        OUTPUT_VIDEO_DIR,
        OUTPUT_LOG_DIR,
        OUTPUT_CSV_DIR,
        OUTPUT_FIGURE_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)

