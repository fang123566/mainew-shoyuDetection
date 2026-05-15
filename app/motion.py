"""帧间差分运动区域检测模块。

通过比较相邻两帧的灰度差异，提取画面中发生变化的运动区域。
可用于：
  1. motion_roi 模式：将运动区域裁剪为 ROI，仅在该区域执行 YOLO 检测；
  2. motion_filter 模式：用运动区域与 YOLO 检测框的重叠度过滤误检。

算法流程：
  当前帧 + 上一帧 → 灰度化 → 高斯平滑 → absdiff → 阈值化
  → 形态学处理（闭运算填空洞、开运算去噪点）→ 轮廓提取
  → 过滤小面积轮廓 → 合并最大轮廓 → 扩大 ROI 边界 → 返回运动区域。
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import cv2
except ImportError as exc:
    cv2 = None
    _CV2_IMPORT_ERROR = exc

try:
    import numpy as np
except ImportError:
    np = None


@dataclass
class MotionResult:
    """帧差结果容器。"""

    mask: object | None  # 二值运动掩码（单通道灰度图），可用 cv2.imread 方式查看
    bbox: tuple[int, int, int, int] | None  # 运动区域边界框 (x1, y1, x2, y2)
    score: float  # 运动得分，等于掩码中白色像素数量


class FrameDifferencer:
    """帧间差分运动区域检测器。"""

    def __init__(
        self,
        diff_threshold: int = 25,
        min_area: int = 200,
        expand_ratio: float = 0.3,
        blur_size: int = 5,
        morph_kernel: int = 3,
    ) -> None:
        """
        Args:
            diff_threshold: 帧差阈值，像素差大于此值视为运动。值越小越敏感。
            min_area: 最小运动区域面积（像素数），小于此值的区域会被过滤。
            expand_ratio: ROI 边界扩大比例，防止裁剪掉手部边缘。
            blur_size: 高斯模糊核大小（必须为奇数）。
            morph_kernel: 形态学操作核大小。
        """
        if cv2 is None:
            raise RuntimeError(
                "未安装 opencv-python，请先运行: pip install -r requirements.txt"
            ) from _CV2_IMPORT_ERROR

        self.diff_threshold = diff_threshold
        self.min_area = min_area
        self.expand_ratio = expand_ratio
        self.blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1
        self.morph_kernel = morph_kernel

        self._prev_frame: object | None = None  # 上一帧（灰度图）

    def update(self, frame: object) -> MotionResult:
        """
        处理新帧，返回当前帧相对于上一帧的运动区域。

        Args:
            frame: BGR 彩色帧（numpy.ndarray，shape HxWx3）。

        Returns:
            MotionResult，包含运动掩码、运动边界框和运动得分。
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self._prev_frame is None:
            self._prev_frame = gray
            return MotionResult(mask=None, bbox=None, score=0.0)

        # 1. 高斯平滑，减少噪声引起的误检
        gray_blur = cv2.GaussianBlur(gray, (self.blur_size, self.blur_size), 0)
        prev_blur = cv2.GaussianBlur(self._prev_frame, (self.blur_size, self.blur_size), 0)

        # 2. 帧间差分取绝对值
        diff = cv2.absdiff(gray_blur, prev_blur)

        # 3. 阈值化得到二值运动掩码
        _, mask = cv2.threshold(diff, self.diff_threshold, 255, cv2.THRESH_BINARY)

        # 4. 形态学处理：先闭运算填空洞，再开运算去噪点
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morph_kernel, self.morph_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # 5. 可选：膨胀一下，避免轮廓收缩导致手部被裁剪
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, dilate_kernel, iterations=1)

        # 6. 找连通域轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 7. 过滤小面积轮廓，收集有效轮廓
        valid_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= self.min_area:
                valid_contours.append(cnt)

        motion_bbox = None
        motion_score = 0.0

        if valid_contours:
            # 8. 合并所有有效轮廓的外接矩形
            x_min = y_min = float("inf")
            x_max = y_max = float("-inf")
            total_area = 0

            for cnt in valid_contours:
                x, y, w, h = cv2.boundingRect(cnt)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x + w)
                y_max = max(y_max, y + h)
                total_area += cv2.contourArea(cnt)

            h_frame, w_frame = frame.shape[:2]

            # 9. 扩大 ROI 边界，防止裁剪掉手部边缘
            expand_w = int((x_max - x_min) * self.expand_ratio)
            expand_h = int((y_max - y_min) * self.expand_ratio)
            x_min_e = max(0, x_min - expand_w)
            y_min_e = max(0, y_min - expand_h)
            x_max_e = min(w_frame, x_max + expand_w)
            y_max_e = min(h_frame, y_max + expand_h)

            motion_bbox = (int(x_min_e), int(y_min_e), int(x_max_e), int(y_max_e))
            motion_score = float(total_area)

        # 更新上一帧
        self._prev_frame = gray

        return MotionResult(mask=mask, bbox=motion_bbox, score=motion_score)

    def reset(self) -> None:
        """重置内部状态，清空上一帧记录。用于切换视频或重新开始检测。"""
        self._prev_frame = None
