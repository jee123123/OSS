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
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from score2midi.cli import convert_pages


class ConversionWorker(QThread):
    """Convert ordered score pages without blocking the window."""

    succeeded = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        images: list[Path],
        output: Path,
        audiveris: str,
        bpm: int,
    ):
        super().__init__()
        self.images = images
        self.output = output
        self.audiveris = audiveris
        self.bpm = bpm

    def run(self) -> None:
        try:
            convert_pages(
                self.images,
                self.output,
                self.audiveris,
                self.bpm,
            )
        except Exception as error:
            log_path = self.output.with_suffix(".txt")
            try:
                log_path.write_text(str(error), encoding="utf-8")
            except OSError:
                self.failed.emit(str(error))
            else:
                self.failed.emit(f"자세한 내용은 {log_path}를 참고하세요.")
        else:
            self.succeeded.emit(str(self.output))


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
        self.status_label = QLabel("대기 중")
        self.output_name = QLineEdit("score.mid")
        self.bpm_input = QSpinBox()
        self.bpm_input.setRange(20, 300)
        self.bpm_input.setValue(int(self.settings.value("bpm", 120)))
        self.bpm_input.setSuffix(" BPM")
        self.audiveris_label = QLabel(f"Audiveris: {self.audiveris}")
        self.output_label = QLabel(f"저장 위치: {self.output_dir}")

        self.add_button = QPushButton("악보 추가")
        self.remove_button = QPushButton("선택 삭제")
        self.move_up_button = QPushButton("위로")
        self.move_down_button = QPushButton("아래로")
        self.audiveris_button = QPushButton("Audiveris 선택")
        self.output_button = QPushButton("저장 폴더 선택")
        self.convert_button = QPushButton("변환 시작")

        self.add_button.clicked.connect(self.add_files)
        self.remove_button.clicked.connect(self.remove_selected)
        self.move_up_button.clicked.connect(lambda: self.move_selected(-1))
        self.move_down_button.clicked.connect(lambda: self.move_selected(1))
        self.audiveris_button.clicked.connect(self.choose_audiveris)
        self.output_button.clicked.connect(self.choose_output_dir)
        self.convert_button.clicked.connect(self.start_conversion)

        audiveris_layout = QHBoxLayout()
        audiveris_layout.addWidget(self.audiveris_label)
        audiveris_layout.addWidget(self.audiveris_button)

        page_buttons = QHBoxLayout()
        for button in (
            self.add_button,
            self.remove_button,
            self.move_up_button,
            self.move_down_button,
        ):
            page_buttons.addWidget(button)

        output_name_layout = QHBoxLayout()
        output_name_layout.addWidget(QLabel("결과 파일 이름:"))
        output_name_layout.addWidget(self.output_name)
        output_name_layout.addWidget(QLabel("BPM:"))
        output_name_layout.addWidget(self.bpm_input)

        buttons = QHBoxLayout()
        for button in (
            self.output_button,
            self.convert_button,
        ):
            buttons.addWidget(button)

        layout = QVBoxLayout()
        layout.addWidget(
            QLabel(
                "다중 페이지 PDF 한 개 또는 순서대로 결합할 "
                "이미지 여러 장을 추가하세요."
            )
        )
        layout.addWidget(self.file_list)
        layout.addLayout(page_buttons)
        layout.addLayout(audiveris_layout)
        layout.addLayout(output_name_layout)
        layout.addWidget(self.output_label)
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress)
        layout.addLayout(buttons)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def add_files(self) -> None:
        filenames, _ = QFileDialog.getOpenFileNames(
            self,
            "악보 선택",
            "",
            "악보 파일 (*.pdf *.png *.jpg *.jpeg *.tif *.tiff)",
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

    def move_selected(self, offset: int) -> None:
        row = self.file_list.currentRow()
        new_row = row + offset
        if row < 0 or not 0 <= new_row < self.file_list.count():
            return

        item = self.file_list.takeItem(row)
        self.file_list.insertItem(new_row, item)
        self.file_list.setCurrentRow(new_row)

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

        pdf_inputs = [
            image for image in images if image.suffix.lower() == ".pdf"
        ]
        if pdf_inputs and len(images) != 1:
            QMessageBox.warning(
                self,
                "입력 형식 오류",
                "PDF는 한 번에 하나만 선택하세요. "
                "PDF와 이미지 파일을 함께 사용할 수 없습니다.",
            )
            return

        filename = self.output_name.text().strip()
        if not filename or Path(filename).name != filename:
            QMessageBox.warning(
                self,
                "파일 이름 오류",
                "폴더 경로를 제외한 MIDI 파일 이름을 입력하세요.",
            )
            return
        if Path(filename).suffix.lower() != ".mid":
            filename += ".mid"

        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            QMessageBox.critical(self, "저장 폴더 오류", str(error))
            return

        output = self.output_dir / filename
        original_stem = output.stem
        suffix = 2
        while output.exists() or output.with_suffix(".musicxml").exists():
            output = output.with_name(f"{original_stem}_{suffix}.mid")
            suffix += 1

        self.progress.setRange(0, 0)
        self.status_label.setText("MIDI와 MusicXML로 변환 중…")
        self.set_controls_enabled(False)
        bpm = self.bpm_input.value()
        self.settings.setValue("bpm", bpm)

        self.worker = ConversionWorker(
            images,
            output,
            self.audiveris,
            bpm,
        )
        self.worker.succeeded.connect(self.conversion_succeeded)
        self.worker.failed.connect(self.conversion_failed)
        self.worker.finished.connect(lambda: self.set_controls_enabled(True))
        self.worker.start()

    def set_controls_enabled(self, enabled: bool) -> None:
        for widget in (
            self.add_button,
            self.remove_button,
            self.move_up_button,
            self.move_down_button,
            self.audiveris_button,
            self.output_button,
            self.convert_button,
            self.output_name,
            self.bpm_input,
        ):
            widget.setEnabled(enabled)

    def conversion_succeeded(self, output: str) -> None:
        self.progress.setRange(0, 100)
        self.progress.setValue(100)
        self.status_label.setText(f"완료: {output}")
        musicxml_path = Path(output).with_suffix(".musicxml")
        details_path = Path(output).with_suffix(".txt")

        message = (
            f"MIDI와 MusicXML 파일로 변환했습니다.\n\n"
            f"MIDI: {output}\n"
            f"MusicXML: {musicxml_path}"
        )
        if details_path.is_file():
            message += f"\n\n세부 내용은 {details_path.name}을 참고하세요."
        QMessageBox.information(self, "변환 완료", message)

    def conversion_failed(self, error: str) -> None:
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.status_label.setText("변환 실패")
        QMessageBox.critical(self, "변환 실패", error)

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
