"""视频检测脚本，复用 app/main.py 的检测后端。

示例：
python train/predict_video.py --weights weights/yolov11_best.pt --source test.mp4 --save
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.main import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())

