"""A small desktop interface for converting one or more score images."""

from __future__ import annotations

import shutil
import sys
import re
import tempfile
from pathlib import Path

from PIL.ImageQt import ImageQt
from PySide6.QtCore import QSettings, QSize, QThread, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QIcon, QPixmap
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from score2midi.cli import convert_pages, rewrite_midi_tempo
from score2midi.page_editor import (
    PageSpec,
    load_page_image,
    pages_for_file,
    prepare_conversion_inputs,
)
from score2midi.preprocess import preprocess_page


def _result_summary(details: str) -> str:
    route = re.search(
        r"(\d+)마디를 (\d+)마디의 재생 순서로 전개",
        details,
    )
    measure_reports = sum(
        "마디" in line
        for line in details.splitlines()
        if line.strip()
    )
    playback = (
        f"인식 {route.group(1)}마디 → 재생 {route.group(2)}마디"
        if route
        else "연주 순서: 지면 순서 또는 MusicXML 기호 기준"
    )
    return f"{playback} · 마디 관련 보고 {measure_reports}건"


class ConversionWorker(QThread):
    """Convert ordered score pages without blocking the window."""

    succeeded = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        pages: list[PageSpec],
        output: Path,
        audiveris: str,
        bpm: int,
    ):
        super().__init__()
        self.pages = pages
        self.output = output
        self.audiveris = audiveris
        self.bpm = bpm

    def run(self) -> None:
        try:
            with tempfile.TemporaryDirectory(prefix="score2midi-pages-") as directory:
                inputs = prepare_conversion_inputs(self.pages, Path(directory))
                convert_pages(
                    inputs,
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
        self.resize(1050, 800)

        self.output_dir = Path.cwd() / "outputs"
        self.settings = QSettings("Score2MIDI", "Score2MIDI")
        self.audiveris = str(
            self.settings.value("audiveris_path", "audiveris")
        )
        self.current_output: Path | None = None

        self.file_list = QListWidget()
        self.file_list.setViewMode(QListView.ViewMode.IconMode)
        self.file_list.setResizeMode(QListView.ResizeMode.Adjust)
        self.file_list.setMovement(QListView.Movement.Snap)
        self.file_list.setDragDropMode(
            QAbstractItemView.DragDropMode.InternalMove
        )
        self.file_list.setIconSize(QSize(110, 145))
        self.file_list.setMinimumHeight(190)
        self.file_list.currentItemChanged.connect(self.show_page_preview)
        self.original_preview = QLabel("원본 미리보기")
        self.processed_preview = QLabel("전처리 미리보기")
        self.preview_info = QLabel("페이지를 선택하세요.")
        for preview in self.original_preview, self.processed_preview:
            preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
            preview.setMinimumSize(260, 260)
            preview.setStyleSheet("border: 1px solid #999;")
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
        self.rotate_left_button = QPushButton("왼쪽 회전")
        self.rotate_right_button = QPushButton("오른쪽 회전")
        self.audiveris_button = QPushButton("Audiveris 선택")
        self.output_button = QPushButton("저장 폴더 선택")
        self.convert_button = QPushButton("변환 시작")

        self.add_button.clicked.connect(self.add_files)
        self.remove_button.clicked.connect(self.remove_selected)
        self.move_up_button.clicked.connect(lambda: self.move_selected(-1))
        self.move_down_button.clicked.connect(lambda: self.move_selected(1))
        self.rotate_left_button.clicked.connect(lambda: self.rotate_selected(-90))
        self.rotate_right_button.clicked.connect(lambda: self.rotate_selected(90))
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
            self.rotate_left_button,
            self.rotate_right_button,
        ):
            page_buttons.addWidget(button)

        preview_layout = QHBoxLayout()
        preview_layout.addWidget(self.original_preview)
        preview_layout.addWidget(self.processed_preview)

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
        layout.addLayout(preview_layout)
        layout.addWidget(self.preview_info)
        layout.addLayout(audiveris_layout)
        layout.addLayout(output_name_layout)
        layout.addWidget(self.output_label)
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress)
        layout.addLayout(buttons)

        self.input_page = QWidget()
        self.input_page.setLayout(layout)
        self.result_page = self.build_result_page()
        self.pages = QStackedWidget()
        self.pages.addWidget(self.input_page)
        self.pages.addWidget(self.result_page)
        self.setCentralWidget(self.pages)

    def build_result_page(self) -> QWidget:
        self.result_path_label = QLabel()
        self.result_path_label.setWordWrap(True)
        self.result_summary_label = QLabel()
        self.report_view = QTextEdit()
        self.report_view.setReadOnly(True)

        self.play_button = QPushButton("재생")
        self.pause_button = QPushButton("일시정지")
        self.stop_button = QPushButton("정지")
        self.open_midi_button = QPushButton("MIDI 열기")
        self.open_xml_button = QPushButton("MusicXML 열기")
        self.open_report_button = QPushButton("보고서 열기")
        self.open_folder_button = QPushButton("출력 폴더 열기")
        self.new_conversion_button = QPushButton("다른 악보 변환")
        self.result_bpm_input = QSpinBox()
        self.result_bpm_input.setRange(20, 300)
        self.result_bpm_input.setSuffix(" BPM")
        self.save_bpm_button = QPushButton("새 BPM으로 저장")

        self.audio_output = QAudioOutput(self)
        self.player = QMediaPlayer(self)
        self.player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(0.8)
        self.player.errorOccurred.connect(self.playback_failed)

        self.play_button.clicked.connect(self.play_result)
        self.pause_button.clicked.connect(self.player.pause)
        self.stop_button.clicked.connect(self.player.stop)
        self.open_midi_button.clicked.connect(
            lambda: self.open_result_file(".mid")
        )
        self.open_xml_button.clicked.connect(
            lambda: self.open_result_file(".musicxml")
        )
        self.open_report_button.clicked.connect(
            lambda: self.open_result_file(".txt")
        )
        self.open_folder_button.clicked.connect(self.open_output_folder)
        self.save_bpm_button.clicked.connect(self.save_with_bpm)
        self.new_conversion_button.clicked.connect(
            self.start_new_conversion
        )

        playback_layout = QHBoxLayout()
        for button in self.play_button, self.pause_button, self.stop_button:
            playback_layout.addWidget(button)

        file_layout = QHBoxLayout()
        for button in (
            self.open_midi_button,
            self.open_xml_button,
            self.open_report_button,
            self.open_folder_button,
        ):
            file_layout.addWidget(button)

        bpm_layout = QHBoxLayout()
        bpm_layout.addWidget(QLabel("출력 속도:"))
        bpm_layout.addWidget(self.result_bpm_input)
        bpm_layout.addWidget(self.save_bpm_button)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("변환 결과"))
        layout.addWidget(self.result_path_label)
        layout.addWidget(self.result_summary_label)
        layout.addLayout(playback_layout)
        layout.addLayout(file_layout)
        layout.addLayout(bpm_layout)
        layout.addWidget(QLabel("변환 보고서"))
        layout.addWidget(self.report_view)
        layout.addWidget(self.new_conversion_button)

        page = QWidget()
        page.setLayout(layout)
        return page

    def add_files(self) -> None:
        filenames, _ = QFileDialog.getOpenFileNames(
            self,
            "악보 선택",
            "",
            "악보 파일 (*.pdf *.png *.jpg *.jpeg *.tif *.tiff)",
        )
        existing = {
            self.file_list.item(i).data(Qt.ItemDataRole.UserRole).source.resolve()
            for i in range(self.file_list.count())
        }
        for filename in filenames:
            path = Path(filename)
            if path.resolve() in existing:
                continue
            try:
                specs = pages_for_file(path)
            except (OSError, ValueError) as error:
                QMessageBox.warning(self, "악보 열기 실패", str(error))
                continue
            for spec in specs:
                self.add_page_item(spec)
            existing.add(path.resolve())
        if self.file_list.count() and self.file_list.currentRow() < 0:
            self.file_list.setCurrentRow(0)

    def add_page_item(self, spec: PageSpec) -> None:
        item = QListWidgetItem(spec.label)
        item.setData(Qt.ItemDataRole.UserRole, spec)
        item.setTextAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.update_page_item(item)
        self.file_list.addItem(item)

    def update_page_item(self, item: QListWidgetItem) -> None:
        spec = item.data(Qt.ItemDataRole.UserRole)
        image = load_page_image(spec, dpi=72)
        try:
            image.thumbnail((110, 145))
            item.setIcon(QIcon(QPixmap.fromImage(ImageQt(image))))
            item.setText(spec.label)
        finally:
            image.close()

    def show_page_preview(
        self,
        current: QListWidgetItem | None,
        _previous: QListWidgetItem | None = None,
    ) -> None:
        if current is None:
            self.original_preview.clear()
            self.processed_preview.clear()
            self.preview_info.setText("페이지를 선택하세요.")
            return
        spec = current.data(Qt.ItemDataRole.UserRole)
        original = load_page_image(spec)
        processed = preprocess_page(original)
        try:
            self.set_preview_image(self.original_preview, original)
            self.set_preview_image(self.processed_preview, processed)
            source_type = "PDF" if spec.page_index is not None else "이미지"
            low_resolution = (
                " · 낮은 해상도: 변환 시 자동 확대"
                if spec.page_index is None and max(original.size) < 1500
                else ""
            )
            self.preview_info.setText(
                f"{source_type} · {original.width}×{original.height}px"
                f" · 회전 {spec.rotation}°{low_resolution}"
            )
        finally:
            original.close()
            if processed is not original:
                processed.close()

    @staticmethod
    def set_preview_image(label: QLabel, image) -> None:
        preview = image.copy()
        try:
            preview.thumbnail((340, 300))
            label.setPixmap(QPixmap.fromImage(ImageQt(preview)))
        finally:
            preview.close()

    def rotate_selected(self, degrees: int) -> None:
        item = self.file_list.currentItem()
        if item is None:
            return
        spec = item.data(Qt.ItemDataRole.UserRole).rotate(degrees)
        item.setData(Qt.ItemDataRole.UserRole, spec)
        self.update_page_item(item)
        self.show_page_preview(item)

    def remove_selected(self) -> None:
        for item in self.file_list.selectedItems():
            self.file_list.takeItem(self.file_list.row(item))
        if self.file_list.count() and self.file_list.currentRow() < 0:
            self.file_list.setCurrentRow(0)

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

        page_specs = [
            self.file_list.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self.file_list.count())
        ]
        if not page_specs:
            QMessageBox.information(self, "악보 없음", "먼저 악보 파일을 추가하세요.")
            return

        pdf_sources = {
            spec.source.resolve()
            for spec in page_specs
            if spec.page_index is not None
        }
        if pdf_sources and (
            len(pdf_sources) != 1
            or any(spec.page_index is None for spec in page_specs)
        ):
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
            page_specs,
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
            self.rotate_left_button,
            self.rotate_right_button,
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
        self.show_result(Path(output))

    def show_result(self, output: Path) -> None:
        self.current_output = output
        details_path = output.with_suffix(".txt")
        try:
            details = details_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            details = "별도의 변환 경고가 없습니다."
        except OSError as error:
            details = f"보고서를 읽을 수 없습니다: {error}"

        self.result_path_label.setText(f"MIDI: {output}")
        self.result_summary_label.setText(_result_summary(details))
        self.report_view.setPlainText(details)
        self.result_bpm_input.setValue(self.bpm_input.value())
        self.open_xml_button.setEnabled(output.with_suffix(".musicxml").is_file())
        self.open_report_button.setEnabled(details_path.is_file())
        self.player.stop()
        self.pages.setCurrentWidget(self.result_page)

    def play_result(self) -> None:
        if self.current_output is not None:
            source = QUrl.fromLocalFile(str(self.current_output.resolve()))
            if self.player.source() != source:
                self.player.setSource(source)
            self.player.play()

    def start_new_conversion(self) -> None:
        self.player.stop()
        self.pages.setCurrentWidget(self.input_page)

    def playback_failed(self, _error, message: str) -> None:
        QMessageBox.warning(
            self,
            "내장 재생 불가",
            f"이 환경에서는 MIDI를 바로 재생할 수 없습니다.\n{message}\n\n"
            "MIDI 열기 버튼으로 시스템 기본 앱을 사용하세요.",
        )

    def open_result_file(self, suffix: str) -> None:
        if self.current_output is None:
            return
        path = self.current_output.with_suffix(suffix)
        if path.is_file():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))

    def open_output_folder(self) -> None:
        if self.current_output is not None:
            QDesktopServices.openUrl(
                QUrl.fromLocalFile(str(self.current_output.parent.resolve()))
            )

    def save_with_bpm(self) -> None:
        if self.current_output is None:
            return
        suggested = self.current_output.with_name(
            f"{self.current_output.stem}_{self.result_bpm_input.value()}bpm.mid"
        )
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "BPM을 변경한 MIDI 저장",
            str(suggested),
            "MIDI 파일 (*.mid)",
        )
        if not filename:
            return
        output = Path(filename)
        if output.suffix.lower() != ".mid":
            output = output.with_suffix(".mid")
        try:
            rewrite_midi_tempo(
                self.current_output,
                output,
                self.result_bpm_input.value(),
            )
        except Exception as error:
            QMessageBox.critical(self, "BPM 저장 실패", str(error))
            return
        QMessageBox.information(self, "저장 완료", f"저장했습니다:\n{output}")

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
        self.player.stop()
        event.accept()


def main() -> None:
    app = QApplication(sys.argv)
    window = Score2MidiWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
