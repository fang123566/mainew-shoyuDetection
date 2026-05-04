"""PyQt5 图形界面：多类别手语识别与无障碍交流辅助系统。

设计原则：
- 简洁正式的课程设计风格，不要淘宝风格。
- 左侧显示视频/摄像头画面，右侧显示识别结果和历史。
- 后端全部调用 app/ 下的模块，不在 UI 文件里堆叠业务逻辑。
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
    from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon
    from PyQt5.QtWidgets import (
        QApplication,
        QAction,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QMainWindow,
        QMessageBox,
        QProgressBar,
        QPushButton,
        QSizePolicy,
        QSpinBox,
        QStatusBar,
        QTableWidget,
        QTableWidgetItem,
        QTextEdit,
        QToolBar,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:
    raise RuntimeError("未安装 PyQt5，请先运行: pip install -r requirements.txt") from exc

try:
    from .camera import cv2, open_capture, read_image, resize_to_fit
    from .config import DEFAULT_WEIGHTS, ensure_output_dirs
    from .detector import DetectionResult, SignLanguageDetector
    from .recorder import RecognitionRecorder
    from .speech import SpeechEngine
    from .translator import LabelTranslator
except ImportError:
    from camera import cv2, open_capture, read_image, resize_to_fit
    from config import DEFAULT_WEIGHTS, ensure_output_dirs
    from detector import DetectionResult, SignLanguageDetector
    from recorder import RecognitionRecorder
    from speech import SpeechEngine
    from translator import LabelTranslator


class DetectionSignals(QObject):
    """跨线程信号桥，检测线程完成后通知主线程更新界面。"""
    frame_ready = pyqtSignal(object, object)  # (annotated_frame, detections)


class SignLanguageAssistantUI(QMainWindow):
    """多类别手语识别与无障碍交流辅助系统图形界面。"""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("多类别手语识别与无障碍交流辅助系统")
        self.resize(1400, 860)

        # 核心组件
        self.detector: SignLanguageDetector | None = None
        self.translator = LabelTranslator()
        self.recorder = RecognitionRecorder()
        self.speech = SpeechEngine(enabled=False)
        self.capture = None
        self.is_running = False
        self.current_detections: list[DetectionResult] = []
        self.current_annotated = None
        self.fps_value = 0.0
        self.frame_count = 0
        self.fps_timer = QTimer(self)
        self.fps_timer.timeout.connect(self._update_fps_display)

        ensure_output_dirs()
        self._setup_ui()
        self._connect_signals()
        self._apply_style()
        self.statusBar().showMessage("就绪：请先加载模型或打开媒体源")

    # ------------------------------------------------------------------ #
    #  UI 构建
    # ------------------------------------------------------------------ #

    def _setup_ui(self) -> None:
        """构建完整的 UI 布局。"""
        central = QWidget(self)
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        # 左侧：视频画面 + 控制栏
        left_panel = self._build_left_panel()
        root.addWidget(left_panel, stretch=3)

        # 右侧：状态 + 历史表格
        right_panel = self._build_right_panel()
        root.addWidget(right_panel, stretch=2)

    def _build_left_panel(self) -> QWidget:
        """视频显示区 + 控制栏。"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # 视频画面
        self.video_label = QLabel("请选择图片、视频或摄像头开始检测")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(800, 560)
        self.video_label.setStyleSheet(
            "background:#0d1117;color:#c9d1d9;border:1px solid #30363d;"
            "border-radius:6px;font-size:16px;"
        )
        layout.addWidget(self.video_label, stretch=1)

        # FPS + 状态栏
        self.fps_label = QLabel("FPS: -")
        self.fps_label.setStyleSheet("color:#58a6ff;font-size:13px;font-weight:bold;")
        layout.addWidget(self.fps_label)

        # 控制栏
        controls = self._build_control_bar()
        layout.addWidget(controls)

        return panel

    def _build_control_bar(self) -> QGroupBox:
        """控制按钮、参数调节区。"""
        group = QGroupBox("控制面板")
        grid = QGridLayout(group)

        # -- 置信度调节 --
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.05, 0.95)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.5)
        self.conf_spin.setPrefix("置信度阈值: ")
        self.conf_spin.setSuffix("")
        self.conf_spin.setMaximumWidth(160)

        # -- top1 模式 --
        self.top1_check = QCheckBox("只显示最高置信度")
        self.top1_check.setChecked(False)

        # -- 语音开关 --
        self.speech_check = QCheckBox("开启语音播报")
        self.speech_check.stateChanged.connect(
            lambda state: self.speech.set_enabled(state == Qt.Checked)
        )

        # -- 模型路径 --
        self.weights_label = QLabel(f"模型: {Path(DEFAULT_WEIGHTS).name}")
        self.weights_label.setStyleSheet("color:#8b949e;font-size:12px;")
        self.load_model_btn = QPushButton("加载模型")
        self.load_model_btn.clicked.connect(self._load_model)

        # -- 打开文件按钮 --
        btn_open_img = QPushButton("打开图片")
        btn_open_img.clicked.connect(self._open_image)

        btn_open_vid = QPushButton("打开视频")
        btn_open_vid.clicked.connect(self._open_video)

        btn_open_cam = QPushButton("打开摄像头")
        btn_open_cam.clicked.connect(self._open_camera)

        btn_stop = QPushButton("停止检测")
        btn_stop.setObjectName("btn_stop")
        btn_stop.clicked.connect(self._stop_detection)

        btn_save = QPushButton("保存结果")
        btn_save.clicked.connect(self._save_result)

        btn_clear = QPushButton("清空历史")
        btn_clear.clicked.connect(self._clear_history)

        row = 0
        grid.addWidget(QLabel("参数设置："), row, 0, 1, 2)
        row += 1
        grid.addWidget(self.conf_spin, row, 0, 1, 2)
        row += 1
        grid.addWidget(self.top1_check, row, 0, 1, 2)
        row += 1
        grid.addWidget(self.speech_check, row, 0, 1, 2)
        row += 1
        grid.addWidget(self.weights_label, row, 0)
        grid.addWidget(self.load_model_btn, row, 1)
        row += 1
        grid.addWidget(btn_open_img, row, 0)
        grid.addWidget(btn_open_vid, row, 1)
        row += 1
        grid.addWidget(btn_open_cam, row, 0)
        grid.addWidget(btn_stop, row, 1)
        row += 1
        grid.addWidget(btn_save, row, 0)
        grid.addWidget(btn_clear, row, 1)

        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        return group

    def _build_right_panel(self) -> QWidget:
        """右侧面板：当前识别结果 + 历史记录表格。"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # 当前识别结果
        result_box = self._build_result_box()
        layout.addWidget(result_box)

        # 历史记录
        history_label = QLabel("识别历史记录（最近 50 条）")
        history_label.setStyleSheet("font-size:14px;font-weight:bold;color:#c9d1d9;")
        layout.addWidget(history_label)

        self.history_table = QTableWidget(0, 5)
        self.history_table.setHorizontalHeaderLabels(["时间", "英文标签", "中文含义", "置信度", "坐标"])
        header = self.history_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.Stretch)
        self.history_table.verticalHeader().setVisible(False)
        self.history_table.setStyleSheet(
            "QTableWidget { background:#161b22; color:#c9d1d9; gridline-color:#30363d; "
            "border:1px solid #30363d; font-size:12px; } "
            "QTableWidget::item { padding:4px; } "
            "QHeaderView::section { background:#21262d; color:#c9d1d9; padding:5px; border:none; border-right:1px solid #30363d; border-bottom:1px solid #30363d; }"
        )
        layout.addWidget(self.history_table, stretch=1)

        return panel

    def _build_result_box(self) -> QGroupBox:
        """当前识别结果展示区。"""
        group = QGroupBox("当前识别结果")
        group.setStyleSheet(
            "QGroupBox { font-weight:bold; font-size:13px; border:1px solid #30363d; "
            "border-radius:6px; margin-top:6px; background:#161b22; color:#c9d1d9; } "
            "QGroupBox::title { subcontrol-origin:margin; left:10px; padding:0 4px; }"
        )
        vbox = QVBoxLayout(group)

        self.model_status_label = QLabel("模型状态：未加载")
        self.model_status_label.setStyleSheet("color:#8b949e;font-size:12px;")

        self.current_en_label = QLabel("英文标签：-")
        self.current_cn_label = QLabel("中文含义：-")
        self.current_conf_label = QLabel("置信度：-")
        self.current_coord_label = QLabel("坐标：-")

        for label in [self.current_en_label, self.current_cn_label,
                       self.current_conf_label, self.current_coord_label]:
            label.setStyleSheet("font-size:14px;color:#f0f6fc;")

        vbox.addWidget(self.model_status_label)
        vbox.addWidget(self.current_en_label)
        vbox.addWidget(self.current_cn_label)
        vbox.addWidget(self.current_conf_label)
        vbox.addWidget(self.current_coord_label)

        return group

    # ------------------------------------------------------------------ #
    #  信号连接
    # ------------------------------------------------------------------ #

    def _connect_signals(self) -> None:
        """连接信号槽。"""
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_frame)

    # ------------------------------------------------------------------ #
    #  样式
    # ------------------------------------------------------------------ #

    def _apply_style(self) -> None:
        """统一应用暗色主题。"""
        self.setStyleSheet("""
            QMainWindow { background:#0d1117; }
            QWidget { font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif; }

            QGroupBox { background:#161b22; border:1px solid #30363d; border-radius:6px; margin-top:8px; }
            QGroupBox::title { subcontrol-origin:margin; left:12px; padding:0 6px; color:#c9d1d9; font-size:13px; font-weight:bold; }

            QPushButton {
                background:#238636; color:white; border:none; border-radius:6px;
                padding:8px 12px; font-size:13px; font-weight:500; min-height:28px;
            }
            QPushButton:hover { background:#2ea043; }
            QPushButton:pressed { background:#238636; }
            QPushButton#btn_stop { background:#da3633; }
            QPushButton#btn_stop:hover { background:#f85149; }

            QDoubleSpinBox, QSpinBox, QComboBox {
                background:#0d1117; color:#c9d1d9; border:1px solid #30363d;
                border-radius:4px; padding:4px 6px; font-size:13px; min-height:24px;
            }
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button,
            QSpinBox::up-button, QSpinBox::down-button {
                background:#21262d; border:none; border-radius:2px;
            }

            QCheckBox { color:#c9d1d9; font-size:13px; spacing:6px; }
            QCheckBox::indicator { width:16px; height:16px; border-radius:3px; border:1px solid #30363d; background:#0d1117; }
            QCheckBox::indicator:checked { background:#238636; border-color:#238636; }

            QLabel { color:#c9d1d9; font-size:13px; }
            QLabel[heading="true"] { font-size:16px; font-weight:bold; color:#f0f6fc; }

            QStatusBar { background:#161b22; color:#8b949e; border-top:1px solid #30363d; font-size:12px; }
            QStatusBar::item { border:none; }

            QTableWidget { background:#161b22; color:#c9d1d9; border:1px solid #30363d; border-radius:4px; font-size:12px; }
            QTableWidget::item { padding:3px 5px; }
            QTableWidget::item:selected { background:#1f6feb; color:white; }
            QHeaderView::section { background:#21262d; color:#c9d1d9; padding:5px 6px; border:none; border-right:1px solid #30363d; border-bottom:1px solid #30363d; font-weight:bold; font-size:12px; }
            QScrollBar:vertical { background:#161b22; width:10px; border-radius:5px; }
            QScrollBar::handle:vertical { background:#30363d; border-radius:4px; min-height:30px; }
            QScrollBar::handle:vertical:hover { background:#484f58; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0px; }
        """)

    # ------------------------------------------------------------------ #
    #  模型加载
    # ------------------------------------------------------------------ #

    def _load_model(self) -> None:
        """手动触发模型加载。"""
        if self.detector is not None:
            QMessageBox.information(self, "模型状态", "模型已加载，无需重复加载。")
            return
        self._ensure_detector()

    def _ensure_detector(self) -> bool:
        """确保检测器已加载。"""
        if self.detector is not None:
            return True
        try:
            self.detector = SignLanguageDetector(
                DEFAULT_WEIGHTS, conf=self.conf_spin.value()
            )
            self.model_status_label.setText(
                f"模型状态：已加载 {Path(DEFAULT_WEIGHTS).name}"
            )
            self.statusBar().showMessage(f"模型加载成功: {DEFAULT_WEIGHTS}")
            return True
        except Exception as exc:
            QMessageBox.warning(self, "模型加载失败", str(exc))
            self.model_status_label.setText("模型状态：加载失败")
            return False

    # ------------------------------------------------------------------ #
    #  媒体源操作
    # ------------------------------------------------------------------ #

    def _open_image(self) -> None:
        """打开单张图片并进行检测。"""
        path_text, _ = QFileDialog.getOpenFileName(
            self, "打开图片", str(Path.cwd()),
            "Images (*.jpg *.jpeg *.png *.bmp *.webp)"
        )
        if not path_text:
            return
        self._stop_detection()
        if not self._ensure_detector():
            return

        try:
            frame = read_image(Path(path_text))
            detections = self.detector.predict_frame(
                frame, conf=self.conf_spin.value(), top1=self.top1_check.isChecked()
            )
            self._process_detections(detections, frame)
        except Exception as exc:
            QMessageBox.warning(self, "图片检测失败", str(exc))

    def _open_video(self) -> None:
        """打开视频文件并开始播放检测。"""
        path_text, _ = QFileDialog.getOpenFileName(
            self, "打开视频", str(Path.cwd()),
            "Videos (*.mp4 *.avi *.mov *.mkv *.wmv)"
        )
        if path_text:
            self._start_capture(Path(path_text))

    def _open_camera(self) -> None:
        """打开摄像头并开始实时检测。"""
        if self._ensure_detector():
            self._start_capture(0)

    def _start_capture(self, source: int | Path) -> None:
        """启动视频捕获和定时检测循环。"""
        self._stop_detection()
        try:
            self.capture = open_capture(source)
            self.is_running = True
            self.frame_count = 0
            self.fps_value = 0.0
            self.timer.start(15)  # ~66 fps max
            self.fps_timer.start(1000)
            source_str = "摄像头" if isinstance(source, int) else Path(source).name
            self.statusBar().showMessage(f"正在播放: {source_str}，按停止按钮退出")
        except Exception as exc:
            QMessageBox.warning(self, "视频源打开失败", str(exc))

    def _stop_detection(self) -> None:
        """停止视频捕获和检测循环。"""
        self.timer.stop()
        self.fps_timer.stop()
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.is_running = False
        self.statusBar().showMessage("检测已停止")

    def _update_frame(self) -> None:
        """定时器回调：读取帧、推理、更新画面。"""
        if self.capture is None or self.detector is None:
            return

        ok, frame = self.capture.read()
        if not ok:
            self._stop_detection()
            return

        try:
            detections = self.detector.predict_frame(
                frame, conf=self.conf_spin.value(), top1=self.top1_check.isChecked()
            )
        except Exception:
            return

        self._process_detections(detections, frame)
        self.frame_count += 1

    def _process_detections(self, detections: list[DetectionResult], frame) -> None:
        """统一处理检测结果：更新界面、记录历史、语音播报。"""
        self.current_detections = detections

        # 绘制结果
        annotated = self._draw_annotations(frame, detections)
        self.current_annotated = annotated
        self._show_frame(annotated)

        # 更新右侧结果
        if detections:
            top = detections[0]
            label_cn = self.translator.translate_label(top.label_en)
            self.current_en_label.setText(f"英文标签：{top.label_en}")
            self.current_cn_label.setText(f"中文含义：{label_cn}")
            self.current_conf_label.setText(f"置信度：{top.confidence:.2%}")
            self.current_coord_label.setText(
                f"坐标：({top.xmin}, {top.ymin}) - ({top.xmax}, {top.ymax})"
            )
            # 语音播报
            self.speech.speak(label_cn)
        else:
            self.current_en_label.setText("英文标签：-")
            self.current_cn_label.setText("中文含义：-")
            self.current_conf_label.setText("置信度：-")
            self.current_coord_label.setText("坐标：-")

        # 记录历史
        self.recorder.add_many(detections, self.translator)
        self._refresh_table()

    def _draw_annotations(self, frame, detections: list[DetectionResult]):
        """在帧上绘制检测框和标签，返回标注后的帧。"""
        try:
            from PIL import Image, ImageDraw, ImageFont
        except Exception:
            # 不依赖 PIL 时只用 OpenCV 英文绘制
            for det in detections:
                color = (40, 180, 90)
                cv2.rectangle(frame, (det.xmin, det.ymin), (det.xmax, det.ymax), color, 2)
                label_cn = self.translator.translate_label(det.label_en)
                text = f"{det.label_en}|{label_cn} {det.confidence:.2f}"
                cv2.putText(frame, text, (det.xmin, max(0, det.ymin - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            return frame

        # 尝试加载中文字体
        font = None
        font_paths = [
            Path("C:/Windows/Fonts/msyh.ttc"),
            Path("C:/Windows/Fonts/simsun.ttc"),
            Path(__file__).resolve().parents[1] / "new-shoyuDetection" / "Font" / "platech.ttf",
        ]
        for fp in font_paths:
            if fp.exists():
                try:
                    font = ImageFont.truetype(str(fp), size=22)
                    break
                except Exception:
                    pass

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(image)

        for det in detections:
            color = (40, 180, 90)
            label_cn = self.translator.translate_label(det.label_en)
            text = f"{det.label_en} | {label_cn} {det.confidence:.2f}"
            draw.rectangle([det.xmin, det.ymin, det.xmax, det.ymax], outline=color, width=2)
            draw.rectangle([det.xmin, max(0, det.ymin - 30), det.xmin + 400, det.ymin], fill=color)
            if font:
                draw.text((det.xmin + 4, max(0, det.ymin - 26)), text, fill=(255, 255, 255), font=font)
            else:
                draw.text((det.xmin + 4, max(0, det.ymin - 26)), text, fill=(255, 255, 255))

        return cv2.cvtColor(__import__("numpy").array(image), cv2.COLOR_RGB2BGR)

    def _show_frame(self, frame) -> None:
        """将帧转换为 QPixmap 并显示在 video_label 上。"""
        display = resize_to_fit(frame, max_width=860, max_height=600)
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimage = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimage))

    def _update_fps_display(self) -> None:
        """每秒更新一次 FPS 显示。"""
        if self.is_running and self.frame_count > 0:
            self.fps_value = self.frame_count
            self.frame_count = 0
            self.fps_label.setText(f"FPS: {self.fps_value:.0f} | 检测中...")
        else:
            self.fps_label.setText("FPS: -")

    # ------------------------------------------------------------------ #
    #  历史记录
    # ------------------------------------------------------------------ #

    def _refresh_table(self) -> None:
        """刷新历史记录表格。"""
        records = self.recorder.recent(50)
        self.history_table.setRowCount(len(records))
        for row, record in enumerate(records):
            values = [
                record.time,
                record.label_en,
                record.label_cn,
                f"{record.confidence:.2%}",
                f"({record.xmin},{record.ymin})",
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignCenter)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.history_table.setItem(row, col, item)
        self.history_table.scrollToBottom()

    def _clear_history(self) -> None:
        """清空识别历史。"""
        self.recorder.clear()
        self._refresh_table()
        self.current_en_label.setText("英文标签：-")
        self.current_cn_label.setText("中文含义：-")
        self.current_conf_label.setText("置信度：-")
        self.current_coord_label.setText("坐标：-")

    def _save_result(self) -> None:
        """保存识别历史 CSV 和当前截图。"""
        history_path = self.recorder.export_csv()
        msg_parts = [f"识别历史已保存：\n{history_path}"]

        if self.current_annotated is not None:
            from app.config import OUTPUT_IMAGE_DIR
            OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
            img_path = OUTPUT_IMAGE_DIR / "qt_result.jpg"
            cv2.imwrite(str(img_path), self.current_annotated)
            msg_parts.append(f"\n当前画面已保存：\n{img_path}")

        QMessageBox.information(self, "保存结果", "\n".join(msg_parts))

    # ------------------------------------------------------------------ #
    #  窗口事件
    # ------------------------------------------------------------------ #

    def closeEvent(self, event) -> None:  # noqa: N802
        """关闭时自动保存历史并释放资源。"""
        self._stop_detection()
        self.recorder.export_csv()
        event.accept()


# ---------------------------------------------------------------------- #
#  入口
# ---------------------------------------------------------------------- #

def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("手语识别辅助系统")
    window = SignLanguageAssistantUI()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
