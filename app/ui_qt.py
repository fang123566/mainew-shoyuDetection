"""PyQt5 图形界面：多类别手语识别与无障碍交流辅助系统。"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtGui import QImage, QPixmap
    from PyQt5.QtWidgets import (
        QApplication,
        QCheckBox,
        QFileDialog,
        QGridLayout,
        QHeaderView,
        QLabel,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QDoubleSpinBox,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:  # pragma: no cover - 运行时提示安装 PyQt5
    raise RuntimeError("未安装 PyQt5，请先运行: pip install -r requirements.txt") from exc

try:
    from .camera import cv2, open_capture, read_image, resize_to_fit
    from .config import DEFAULT_WEIGHTS, OUTPUT_IMAGE_DIR, ensure_output_dirs
    from .detector import SignLanguageDetector
    from .main import draw_detections
    from .recorder import RecognitionRecorder
    from .speech import SpeechEngine
    from .translator import LabelTranslator
except ImportError:  # 允许 python app/ui_qt.py 直接运行
    from camera import cv2, open_capture, read_image, resize_to_fit
    from config import DEFAULT_WEIGHTS, OUTPUT_IMAGE_DIR, ensure_output_dirs
    from detector import SignLanguageDetector
    from main import draw_detections
    from recorder import RecognitionRecorder
    from speech import SpeechEngine
    from translator import LabelTranslator


class SignLanguageAssistantUI(QMainWindow):
    """简洁正式的课程设计演示界面。"""

    def __init__(self) -> None:
        super().__init__()
        ensure_output_dirs()
        self.setWindowTitle("多类别手语识别与无障碍交流辅助系统")
        self.resize(1280, 760)

        self.detector: SignLanguageDetector | None = None
        self.translator = LabelTranslator()
        self.recorder = RecognitionRecorder()
        self.speech = SpeechEngine(enabled=False)
        self.capture = None
        self.current_frame = None
        self.current_annotated = None

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.build_ui()
        self.apply_style()

    def build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)

        root = QGridLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setHorizontalSpacing(16)

        self.video_label = QLabel("请选择图片、视频或摄像头")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(760, 520)
        self.video_label.setStyleSheet("background:#111827;color:#d1d5db;border-radius:6px;")
        root.addWidget(self.video_label, 0, 0)

        side = QVBoxLayout()
        root.addLayout(side, 0, 1)

        self.status_label = QLabel("模型状态：未加载")
        self.result_label = QLabel("当前识别：-")
        self.conf_label = QLabel("置信度：-")
        for label in [self.status_label, self.result_label, self.conf_label]:
            label.setWordWrap(True)
            side.addWidget(label)

        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.05, 0.95)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.5)
        self.conf_spin.setPrefix("置信度阈值 ")
        side.addWidget(self.conf_spin)

        self.speech_check = QCheckBox("开启语音播报")
        self.speech_check.stateChanged.connect(lambda state: self.speech.set_enabled(state == Qt.Checked))
        side.addWidget(self.speech_check)

        button_grid = QGridLayout()
        side.addLayout(button_grid)
        buttons = [
            ("打开图片", self.open_image),
            ("打开视频", self.open_video),
            ("打开摄像头", self.open_camera),
            ("停止检测", self.stop_detection),
            ("保存结果", self.save_result),
            ("清空历史", self.clear_history),
        ]
        for index, (text, slot) in enumerate(buttons):
            button = QPushButton(text)
            button.clicked.connect(slot)
            button_grid.addWidget(button, index // 2, index % 2)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["时间", "英文标签", "中文含义", "置信度"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        side.addWidget(self.table, stretch=1)

    def apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow { background: #f8fafc; }
            QLabel { color: #111827; font-size: 15px; }
            QPushButton {
                background: #2563eb; color: white; border: none; border-radius: 6px;
                padding: 9px 12px; font-size: 14px;
            }
            QPushButton:hover { background: #1d4ed8; }
            QDoubleSpinBox, QTableWidget {
                background: white; border: 1px solid #d1d5db; border-radius: 6px; padding: 4px;
            }
            QHeaderView::section { background: #e5e7eb; padding: 6px; border: none; }
            """
        )

    def ensure_detector(self) -> bool:
        if self.detector is not None:
            return True
        try:
            self.detector = SignLanguageDetector(DEFAULT_WEIGHTS, conf=self.conf_spin.value())
            self.status_label.setText(f"模型状态：已加载 {DEFAULT_WEIGHTS}")
            return True
        except Exception as exc:
            QMessageBox.warning(self, "模型加载失败", str(exc))
            self.status_label.setText("模型状态：加载失败")
            return False

    def open_image(self) -> None:
        if not self.ensure_detector():
            return
        path_text, _ = QFileDialog.getOpenFileName(self, "打开图片", str(Path.cwd()), "Images (*.jpg *.jpeg *.png *.bmp *.webp)")
        if not path_text:
            return
        self.stop_detection()
        frame = read_image(Path(path_text))
        detections = self.detector.predict_frame(frame, conf=self.conf_spin.value())
        self.consume_detections(detections)
        self.current_annotated = draw_detections(frame, detections, self.translator, fps=0.0)
        self.show_frame(self.current_annotated)

    def open_video(self) -> None:
        if not self.ensure_detector():
            return
        path_text, _ = QFileDialog.getOpenFileName(self, "打开视频", str(Path.cwd()), "Videos (*.mp4 *.avi *.mov *.mkv *.wmv)")
        if path_text:
            self.start_capture(Path(path_text))

    def open_camera(self) -> None:
        if self.ensure_detector():
            self.start_capture(0)

    def start_capture(self, source: int | Path) -> None:
        self.stop_detection()
        try:
            self.capture = open_capture(source)
            self.timer.start(15)
        except Exception as exc:
            QMessageBox.warning(self, "视频源打开失败", str(exc))

    def update_frame(self) -> None:
        if self.capture is None or self.detector is None:
            return
        ok, frame = self.capture.read()
        if not ok:
            self.stop_detection()
            return
        detections = self.detector.predict_frame(frame, conf=self.conf_spin.value())
        self.consume_detections(detections)
        self.current_annotated = draw_detections(frame, detections, self.translator, fps=0.0)
        self.show_frame(self.current_annotated)

    def consume_detections(self, detections) -> None:
        self.recorder.add_many(detections, self.translator)
        if detections:
            top = detections[0]
            label_cn = self.translator.translate_label(top.label_en)
            self.result_label.setText(f"当前识别：{top.label_en} / {label_cn}")
            self.conf_label.setText(f"置信度：{top.confidence:.2f}")
            self.speech.speak(label_cn)
        self.refresh_table()

    def refresh_table(self) -> None:
        records = self.recorder.recent(20)
        self.table.setRowCount(len(records))
        for row, record in enumerate(records):
            values = [record.time, record.label_en, record.label_cn, f"{record.confidence:.2f}"]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(row, col, item)
        self.table.scrollToBottom()

    def show_frame(self, frame) -> None:
        display = resize_to_fit(frame, max_width=900, max_height=620)
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb.shape
        qimage = QImage(rgb.data, width, height, channel * width, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimage))

    def save_result(self) -> None:
        history_path = self.recorder.export_csv()
        message = f"识别历史已保存：{history_path}"
        if self.current_annotated is not None:
            OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
            image_path = OUTPUT_IMAGE_DIR / "qt_current_result.jpg"
            cv2.imwrite(str(image_path), self.current_annotated)
            message += f"\n当前画面已保存：{image_path}"
        QMessageBox.information(self, "保存结果", message)

    def clear_history(self) -> None:
        self.recorder.clear()
        self.refresh_table()
        self.result_label.setText("当前识别：-")
        self.conf_label.setText("置信度：-")

    def stop_detection(self) -> None:
        self.timer.stop()
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API 命名
        self.stop_detection()
        self.recorder.export_csv()
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    window = SignLanguageAssistantUI()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())

