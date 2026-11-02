"""A small desktop interface for converting one or more score images."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from PySide6.QtCore import QSettings, QThread, Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from score2midi.cli import convert_score


class ConversionWorker(QThread):
    """Convert queued images sequentially without blocking the window."""

    status_changed = Signal(int, str)
    progress_changed = Signal(int)

    def __init__(self, images: list[Path], output_dir: Path, audiveris: str):
        super().__init__()
        self.images = images
        self.output_dir = output_dir
        self.audiveris = audiveris

    def run(self) -> None:
        total = len(self.images)
        for index, image in enumerate(self.images):
            self.status_changed.emit(index, "변환 중…")

            output = self.output_dir / f"{image.stem}.mid"
            suffix = 2
            while output.exists():
                output = self.output_dir / f"{image.stem}_{suffix}.mid"
                suffix += 1

            try:
                convert_score(image, output, self.audiveris)
            except Exception as error:
                self.status_changed.emit(index, f"실패: {error}")
            else:
                self.status_changed.emit(index, f"완료: {output.name}")

            self.progress_changed.emit(int((index + 1) * 100 / total))


class Score2MidiWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Score2MIDI")
        self.resize(760, 520)

        self.output_dir = Path.cwd() / "outputs"
        self.settings = QSettings("Score2MIDI", "Score2MIDI")
        self.audiveris = str(
            self.settings.value("audiveris_path", "audiveris")
        )

        self.file_list = QListWidget()
        self.progress = QProgressBar()
        self.audiveris_label = QLabel(f"Audiveris: {self.audiveris}")
        self.output_label = QLabel(f"저장 위치: {self.output_dir}")

        self.add_button = QPushButton("악보 추가")
        self.remove_button = QPushButton("선택 삭제")
        self.audiveris_button = QPushButton("Audiveris 선택")
        self.output_button = QPushButton("저장 폴더 선택")
        self.convert_button = QPushButton("변환 시작")

        self.add_button.clicked.connect(self.add_files)
        self.remove_button.clicked.connect(self.remove_selected)
        self.audiveris_button.clicked.connect(self.choose_audiveris)
        self.output_button.clicked.connect(self.choose_output_dir)
        self.convert_button.clicked.connect(self.start_conversion)

        audiveris_layout = QHBoxLayout()
        audiveris_layout.addWidget(self.audiveris_label)
        audiveris_layout.addWidget(self.audiveris_button)

        buttons = QHBoxLayout()
        for button in (
            self.add_button,
            self.remove_button,
            self.output_button,
            self.convert_button,
        ):
            buttons.addWidget(button)

        layout = QVBoxLayout()
        layout.addWidget(
            QLabel("변환할 악보를 추가하세요. 여러 파일을 선택할 수 있습니다.")
        )
        layout.addWidget(self.file_list)
        layout.addLayout(audiveris_layout)
        layout.addWidget(self.output_label)
        layout.addWidget(self.progress)
        layout.addLayout(buttons)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def add_files(self) -> None:
        filenames, _ = QFileDialog.getOpenFileNames(
            self,
            "악보 이미지 선택",
            "",
            "악보 이미지 (*.png *.jpg *.jpeg *.tif *.tiff)",
        )
        existing = {
            self.file_list.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self.file_list.count())
        }
        for filename in filenames:
            path = Path(filename)
            if str(path) in existing:
                continue
            item = QListWidgetItem(str(path))
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            self.file_list.addItem(item)

    def remove_selected(self) -> None:
        for item in self.file_list.selectedItems():
            self.file_list.takeItem(self.file_list.row(item))

    def choose_output_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "MIDI 저장 폴더 선택", str(self.output_dir)
        )
        if folder:
            self.output_dir = Path(folder)
            self.output_label.setText(f"저장 위치: {self.output_dir}")

    def choose_audiveris(self) -> None:
        current = Path(self.audiveris)
        start_dir = (
            str(current.parent)
            if current.is_file()
            else r"C:\Program Files\Audiveris"
        )
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Audiveris 실행 파일 선택",
            start_dir,
            "실행 파일 (*.exe);;모든 파일 (*)",
        )
        if filename:
            self.audiveris = filename
            self.settings.setValue("audiveris_path", filename)
            self.audiveris_label.setText(f"Audiveris: {filename}")

    def start_conversion(self) -> None:
        if (
            shutil.which(self.audiveris) is None
            and not Path(self.audiveris).is_file()
        ):
            QMessageBox.warning(
                self,
                "Audiveris를 찾을 수 없음",
                "Audiveris 선택 버튼을 눌러 Audiveris.exe를 지정하세요.",
            )
            return

        images = [
            Path(self.file_list.item(i).data(Qt.ItemDataRole.UserRole))
            for i in range(self.file_list.count())
        ]
        if not images:
            QMessageBox.information(self, "악보 없음", "먼저 악보 파일을 추가하세요.")
            return

        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            QMessageBox.critical(self, "저장 폴더 오류", str(error))
            return

        self.progress.setValue(0)
        self.set_controls_enabled(False)

        self.worker = ConversionWorker(images, self.output_dir, self.audiveris)
        self.worker.status_changed.connect(self.update_status)
        self.worker.progress_changed.connect(self.progress.setValue)
        self.worker.finished.connect(self.conversion_finished)
        self.worker.start()

    def update_status(self, index: int, status: str) -> None:
        item = self.file_list.item(index)
        image_path = item.data(Qt.ItemDataRole.UserRole)
        item.setText(f"{image_path}  —  {status}")

    def set_controls_enabled(self, enabled: bool) -> None:
        for button in (
            self.add_button,
            self.remove_button,
            self.audiveris_button,
            self.output_button,
            self.convert_button,
        ):
            button.setEnabled(enabled)

    def conversion_finished(self) -> None:
        self.set_controls_enabled(True)
        QMessageBox.information(self, "변환 완료", "대기열의 모든 악보 처리가 끝났습니다.")

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt event naming convention
        if hasattr(self, "worker") and self.worker.isRunning():
            QMessageBox.information(
                self,
                "변환 중",
                "변환이 끝난 뒤 프로그램을 닫아주세요.",
            )
            event.ignore()
            return
        event.accept()


def main() -> None:
    app = QApplication(sys.argv)
    window = Score2MidiWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
