"""A small desktop interface for converting one or more score images."""

from __future__ import annotations

import shutil
import sys
import re
import json
import tempfile
from copy import deepcopy
from fractions import Fraction
from pathlib import Path

from music21 import converter, stream
from PIL.ImageQt import ImageQt
from PySide6.QtCore import QBuffer, QIODevice, QPointF, QSettings, QSize, QThread, Qt, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QIcon, QPainter, QPen, QPixmap
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHeaderView,
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
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from score2midi.cli import convert_pages, rewrite_midi_tempo
from score2midi.measure_edit import measure_pitches, rhythm_gaps, save_pitch_edit
from score2midi.measure_preview import render_measure_preview
from score2midi.page_editor import (
    PageSpec,
    load_page_image,
    pages_for_file,
    prepare_conversion_inputs,
)
from score2midi.measure_review import load_measure_image, load_measure_image_layout, reported_measures
from score2midi.measure_retry import (
    RetryCandidate,
    apply_retry_candidate,
    recognize_measure_candidate,
)
from score2midi.navigation import NavigationPlan
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

    succeeded = Signal(str, object, object)
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
                score, navigation_plan = convert_pages(
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
            self.succeeded.emit(str(self.output), score, navigation_plan)


class MeasureRetryWorker(QThread):
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        pages: list[PageSpec],
        musicxml: Path,
        measure: int,
        audiveris: str,
    ):
        super().__init__()
        self.pages = pages
        self.musicxml = musicxml
        self.measure = measure
        self.audiveris = audiveris

    def run(self) -> None:
        try:
            candidate = recognize_measure_candidate(
                self.pages,
                self.musicxml,
                self.measure,
                self.audiveris,
            )
        except Exception as error:
            self.failed.emit(str(error))
        else:
            self.succeeded.emit(candidate)


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
        self.result_score: stream.Score | None = None
        self.result_navigation_plan: NavigationPlan | None = None
        self.converted_pages: list[PageSpec] = []
        self.review_status: dict[str, str] = {}
        self.preview_audio: QBuffer | None = None

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
        self.report_view.setMaximumHeight(150)
        self.problem_list = QListWidget()
        self.problem_list.setMinimumWidth(330)
        self.problem_list.currentItemChanged.connect(
            self.show_problem_measure
        )
        self.problem_preview = QLabel("검토할 마디를 선택하세요.")
        self.problem_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.problem_preview.setMinimumSize(420, 250)
        self.problem_preview.setStyleSheet("border: 1px solid #999;")
        self.review_status_label = QLabel("검토 상태: 미검토")
        self.normal_button = QPushButton("정상")
        self.retry_button = QPushButton("다시 인식")
        self.edit_button = QPushButton("수정 필요")
        self.run_retry_button = QPushButton("선택 마디 400DPI 재인식")
        self.normal_button.clicked.connect(
            lambda: self.set_review_status("정상")
        )
        self.retry_button.clicked.connect(
            lambda: self.set_review_status("다시 인식")
        )
        self.edit_button.clicked.connect(
            lambda: self.set_review_status("수정 필요")
        )
        self.run_retry_button.clicked.connect(self.start_measure_retry)
        self.listen_measure_button = QPushButton("선택 마디 듣기")
        self.listen_measure_button.setEnabled(False)
        self.listen_measure_button.clicked.connect(self.play_selected_measure)
        self.pitch_edit_button = QPushButton("음표·박자 수정")
        self.pitch_edit_button.setEnabled(False)
        self.pitch_edit_button.clicked.connect(self.edit_selected_measure)
        self.include_neighbors = QCheckBox("앞뒤 한 마디 포함")

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

        review_layout = QHBoxLayout()
        review_layout.addWidget(self.problem_list)
        review_layout.addWidget(self.problem_preview, 1)

        review_buttons = QHBoxLayout()
        review_buttons.addWidget(self.review_status_label)
        for button in self.normal_button, self.retry_button, self.edit_button:
            review_buttons.addWidget(button)
        review_buttons.addWidget(self.run_retry_button)

        preview_controls = QHBoxLayout()
        preview_controls.addWidget(self.listen_measure_button)
        preview_controls.addWidget(self.pitch_edit_button)
        preview_controls.addWidget(self.include_neighbors)
        preview_controls.addStretch()

        layout = QVBoxLayout()
        layout.addWidget(QLabel("변환 결과"))
        layout.addWidget(self.result_path_label)
        layout.addWidget(self.result_summary_label)
        layout.addLayout(playback_layout)
        layout.addLayout(file_layout)
        layout.addLayout(bpm_layout)
        layout.addWidget(QLabel("문제 마디 검토"))
        layout.addLayout(review_layout)
        layout.addLayout(preview_controls)
        layout.addLayout(review_buttons)
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
        self.converted_pages = list(page_specs)
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

    def conversion_succeeded(
        self, output: str, score: stream.Score, navigation_plan: NavigationPlan | None
    ) -> None:
        self.result_score = score
        self.result_navigation_plan = navigation_plan
        self.progress.setRange(0, 100)
        self.progress.setValue(100)
        self.status_label.setText(f"완료: {output}")
        self.show_result(Path(output))

    def show_result(self, output: Path) -> None:
        self.stop_measure_preview()
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
        self.load_review_issues(details)
        self.result_bpm_input.setValue(self.bpm_input.value())
        self.open_xml_button.setEnabled(output.with_suffix(".musicxml").is_file())
        self.open_report_button.setEnabled(details_path.is_file())
        self.player.stop()
        self.pages.setCurrentWidget(self.result_page)

    def load_review_issues(self, details: str) -> None:
        self.problem_list.clear()
        self.problem_preview.setText("검토할 마디를 선택하세요.")
        self.review_status = {}
        if self.current_output is None:
            return

        status_path = self.current_output.with_suffix(".review.json")
        try:
            self.review_status = json.loads(status_path.read_text("utf-8"))
        except (FileNotFoundError, OSError, ValueError):
            pass

        musicxml = self.current_output.with_suffix(".musicxml")
        if not musicxml.is_file():
            return
        for issue in reported_measures(details, musicxml):
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, issue)
            item.setToolTip("\n".join(issue.messages))
            self.problem_list.addItem(item)
            self.update_review_item(item)
        if self.problem_list.count():
            self.problem_list.setCurrentRow(0)
        else:
            self.problem_preview.setText("보고된 문제 마디가 없습니다.")

    def update_review_item(self, item: QListWidgetItem) -> None:
        issue = item.data(Qt.ItemDataRole.UserRole)
        status = self.review_status.get(str(issue.measure), "미검토")
        item.setText(f"{issue.measure}마디 · {status}")

    def show_problem_measure(
        self,
        current: QListWidgetItem | None,
        _previous: QListWidgetItem | None = None,
    ) -> None:
        self.stop_measure_preview()
        self.listen_measure_button.setEnabled(
            current is not None and self.current_output is not None
        )
        self.pitch_edit_button.setEnabled(
            self.listen_measure_button.isEnabled() and self.run_retry_button.isEnabled()
        )
        if current is None or self.current_output is None:
            self.review_status_label.setText("검토 상태: 미검토")
            return
        issue = current.data(Qt.ItemDataRole.UserRole)
        status = self.review_status.get(str(issue.measure), "미검토")
        self.review_status_label.setText(f"검토 상태: {status}")
        try:
            image = load_measure_image(
                self.converted_pages,
                self.current_output.with_suffix(".musicxml"),
                issue.measure,
            )
        except (OSError, ValueError, IndexError, StopIteration) as error:
            self.problem_preview.setText(f"마디 미리보기를 만들 수 없습니다.\n{error}")
            return
        try:
            self.set_preview_image(self.problem_preview, image)
        finally:
            image.close()

    def play_selected_measure(self) -> None:
        item = self.problem_list.currentItem()
        if item is not None:
            issue = item.data(Qt.ItemDataRole.UserRole)
            self.play_measure_preview(issue.measure)

    def play_measure_preview(
        self,
        measure: int,
        retry: RetryCandidate | None = None,
        *,
        score: stream.Score | None = None,
    ) -> None:
        if self.current_output is None:
            return
        self.player.stop()
        self.stop_measure_preview()
        try:
            wav = render_measure_preview(
                score if score is not None else self.current_output.with_suffix(".musicxml"),
                measure,
                self.result_bpm_input.value(),
                include_neighbors=self.include_neighbors.isChecked(),
                retry=retry,
            )
        except Exception as error:
            QMessageBox.warning(
                QApplication.activeModalWidget() or self,
                "미리듣기 실패",
                f"재생할 음원을 만들 수 없습니다.\n{error}",
            )
            return
        audio = QBuffer(self)
        audio.setData(wav)
        audio.open(QIODevice.OpenModeFlag.ReadOnly)
        self.preview_audio = audio
        self.player.setSourceDevice(audio, QUrl("preview.wav"))
        self.player.play()

    def stop_measure_preview(self) -> None:
        if self.preview_audio is not None:
            self.player.setSource(QUrl())
            self.preview_audio.deleteLater()
            self.preview_audio = None

    def edit_selected_measure(self) -> None:
        selected = self.problem_list.currentItem()
        if selected is None or self.current_output is None:
            return
        measure = selected.data(Qt.ItemDataRole.UserRole).measure
        try:
            source = self.result_score
            if source is None:
                source = converter.parse(str(self.current_output.with_suffix(".musicxml")))
            draft = deepcopy(source)
            entries = measure_pitches(draft, measure)
        except Exception as error:
            QMessageBox.warning(self, "마디 편집 실패", str(error))
            return
        if not entries:
            QMessageBox.information(self, "음높이 수정", "이 마디에는 편집할 음표가 없습니다.")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle(f"{measure}마디 음표·박자 수정")
        dialog.resize(920, 490)
        layout = QVBoxLayout(dialog)
        content = QHBoxLayout()
        picture = QLabel("원본 마디 미리보기를 사용할 수 없습니다.")
        picture.setWordWrap(True)
        picture.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image_column = QVBoxLayout()
        image_column.addWidget(picture, 1)
        marker_status = QLabel()
        marker_status.setWordWrap(True)
        image_column.addWidget(marker_status)
        content.addLayout(image_column, 1)

        positions = [None] * len(entries)
        pixmap = self.problem_preview.pixmap()
        try:
            image, image_layout = load_measure_image_layout(
                self.converted_pages, self.current_output.with_suffix(".musicxml"), measure,
            )
            with image:
                self.set_preview_image(picture, image)
                pixmap = picture.pixmap()
                scale_x, scale_y = pixmap.width() / image.width, pixmap.height() / image.height
                for index, entry in enumerate(entries):
                    position = entry.original_image_position(image_layout)
                    if position is None:
                        continue
                    x, y = position
                    if 0 <= x < image.width and 0 <= y < image.height:
                        positions[index] = QPointF(x * scale_x, y * scale_y)
        except (OSError, ValueError, IndexError, StopIteration):
            pass

        table = QTableWidget(len(entries), 4)
        table.setHorizontalHeaderLabels(["파트", "시작 위치 (박)", "길이 (박)", "음높이"])
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        initial_pitches = [entry.member.pitch.ps for entry in entries]
        initial_timing = [entry.timing for entry in entries]
        for row, entry in enumerate(entries):
            part = f"파트 {entry.part_index}"
            if len(draft.parts) == 2:
                part += " (위)" if entry.part_index == 1 else " (아래)"
            if isinstance(entry.voice, stream.Voice):
                part += f" · {entry.voice.id}"
            pitch = entry.pitch_label
            if entry.member.tie is not None:
                pitch += " (붙임줄)"
            start, duration = entry.timing
            for column, value in enumerate((part, str(start), str(duration), pitch)):
                table.setItem(row, column, QTableWidgetItem(value))
        content.addWidget(table, 2)
        layout.addLayout(content)

        timing_controls = QHBoxLayout()
        start_input = QLineEdit()
        duration_input = QLineEdit()
        apply_timing = QPushButton("박자 적용")
        apply_timing.setAutoDefault(False)
        for label, field in (("시작 위치 (박)", start_input), ("길이 (박)", duration_input)):
            timing_controls.addWidget(QLabel(label))
            field.setPlaceholderText("예: 1/2")
            field.setMaximumWidth(95)
            field.setMaxLength(28)
            timing_controls.addWidget(field)
        timing_controls.addWidget(apply_timing)
        timing_controls.addStretch()
        layout.addLayout(timing_controls)
        timing_status = QLabel()
        timing_status.setWordWrap(True)
        layout.addWidget(timing_status)

        lower = QPushButton("반음 내리기")
        raise_pitch = QPushButton("반음 올리기")
        listen = QPushButton("수정본 듣기")
        stop = QPushButton("정지")
        save = QPushButton("수정본 저장")
        close = QPushButton("닫기")
        save.setEnabled(False)
        controls = QHBoxLayout()
        for button in (lower, raise_pitch, listen, stop, save, close):
            button.setAutoDefault(False)
            controls.addWidget(button)
        layout.addLayout(controls)

        def update_controls() -> None:
            row = table.currentRow()
            editable = row >= 0 and entries[row].member.tie is None
            lower.setEnabled(editable)
            raise_pitch.setEnabled(editable)
            can_change_timing = row >= 0 and not entries[row].event.duration.isGrace
            for widget in (start_input, duration_input, apply_timing):
                widget.setEnabled(can_change_timing)
            start, duration = entries[row].timing if row >= 0 else ("", "")
            start_input.setText(str(start))
            duration_input.setText(str(duration))
            position = positions[row] if row >= 0 else None
            if not pixmap.isNull():
                highlighted = pixmap.copy()
                if position is not None:
                    painter = QPainter(highlighted)
                    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                    painter.setPen(QPen(QColor("#0b61cc"), 2))
                    painter.setBrush(QColor(56, 147, 250, 46))
                    painter.drawEllipse(position, 7, 6)
                    painter.end()
                picture.setPixmap(highlighted)
            marker_status.setText(
                "" if position is not None or row < 0 else "이 음표의 원본 위치를 찾을 수 없습니다."
            )
            update_save_state()

        def changed_timing():
            return [entry for entry, initial in zip(entries, initial_timing) if entry.timing != initial]

        def update_save_state() -> None:
            edits = changed_timing()
            row = table.currentRow()
            if row >= 0:
                timing_status.setText(
                    f"마디 길이 {entries[row].bar_length}박 · 시작 위치 0부터 · 4분음표 = 1박"
                )
            try:
                rhythm_gaps(edits)
            except ValueError as error:
                timing_status.setText(str(error))
                save.setEnabled(False)
                return
            save.setEnabled(bool(edits) or any(
                entry.member.pitch.ps != initial
                for entry, initial in zip(entries, initial_pitches)
            ))

        def change_timing() -> None:
            row = table.currentRow()
            if row < 0:
                return
            try:
                start = Fraction(start_input.text().strip())
                duration = Fraction(duration_input.text().strip())
            except (ValueError, ZeroDivisionError):
                timing_status.setText("박자는 1/2 또는 0.5처럼 입력해 주세요.")
                return
            try:
                entries[row].change_timing(start, duration)
            except ValueError as error:
                timing_status.setText(str(error))
                return
            self.stop_measure_preview()
            for i, entry in enumerate(entries):
                if entry.event is entries[row].event:
                    start, duration = entry.timing
                    table.item(i, 1).setText(str(start))
                    table.item(i, 2).setText(str(duration))
            update_controls()

        def shift(semitones: int) -> None:
            row = table.currentRow()
            if row < 0:
                return
            try:
                entries[row].shift(semitones)
            except ValueError as error:
                QMessageBox.information(dialog, "음높이 수정", str(error))
                return
            self.stop_measure_preview()
            table.item(row, 3).setText(entries[row].pitch_label)
            update_save_state()

        def save_draft() -> None:
            suggested = self.current_output.with_name(
                f"{self.current_output.stem}_마디{measure}_수정.mid"
            )
            filename, _ = QFileDialog.getSaveFileName(
                dialog, "수정한 MIDI와 MusicXML 저장", str(suggested),
                "MIDI 파일 (*.mid)", options=QFileDialog.Option.DontConfirmOverwrite,
            )
            if not filename:
                return
            output = Path(filename)
            if output.suffix.lower() != ".mid":
                output = output.with_suffix(".mid")
            try:
                xml = save_pitch_edit(
                    draft, output, self.result_bpm_input.value(), self.result_navigation_plan,
                    timing_edits=changed_timing(),
                )
            except Exception as error:
                QMessageBox.warning(dialog, "수정본 저장 실패", str(error))
                return
            self.stop_measure_preview()
            QMessageBox.information(dialog, "수정본 저장 완료", f"{output}\n{xml}")
            dialog.accept()

        table.currentItemChanged.connect(update_controls)
        table.setCurrentCell(0, 0)
        update_controls()
        apply_timing.clicked.connect(change_timing)
        start_input.returnPressed.connect(change_timing)
        duration_input.returnPressed.connect(change_timing)
        lower.clicked.connect(lambda: shift(-1))
        raise_pitch.clicked.connect(lambda: shift(1))
        listen.clicked.connect(lambda: self.play_measure_preview(measure, score=draft))
        stop.clicked.connect(self.stop_measure_preview)
        save.clicked.connect(save_draft)
        close.clicked.connect(dialog.reject)
        self.player.stop()
        try:
            dialog.exec()
        finally:
            self.stop_measure_preview()
            dialog.deleteLater()

    def set_review_status(self, status: str) -> None:
        item = self.problem_list.currentItem()
        if item is None or self.current_output is None:
            return
        issue = item.data(Qt.ItemDataRole.UserRole)
        self.review_status[str(issue.measure)] = status
        try:
            self.current_output.with_suffix(".review.json").write_text(
                json.dumps(self.review_status, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as error:
            QMessageBox.warning(self, "검토 상태 저장 실패", str(error))
            return
        self.update_review_item(item)
        self.review_status_label.setText(f"검토 상태: {status}")

    def start_measure_retry(self) -> None:
        item = self.problem_list.currentItem()
        if item is None or self.current_output is None:
            return
        issue = item.data(Qt.ItemDataRole.UserRole)
        self.set_review_status("다시 인식")
        self.run_retry_button.setEnabled(False)
        self.pitch_edit_button.setEnabled(False)
        self.new_conversion_button.setEnabled(False)
        self.review_status_label.setText(
            f"검토 상태: {issue.measure}마디 400DPI 재인식 중…"
        )
        self.retry_worker = MeasureRetryWorker(
            list(self.converted_pages),
            self.current_output.with_suffix(".musicxml"),
            issue.measure,
            self.audiveris,
        )
        self.retry_worker.succeeded.connect(self.measure_retry_succeeded)
        self.retry_worker.failed.connect(self.measure_retry_failed)
        self.retry_worker.finished.connect(
            lambda: self.run_retry_button.setEnabled(True)
        )
        self.retry_worker.finished.connect(
            lambda: self.new_conversion_button.setEnabled(True)
        )
        self.retry_worker.finished.connect(
            lambda: self.pitch_edit_button.setEnabled(self.problem_list.currentItem() is not None)
        )
        self.retry_worker.start()

    def confirm_retry_candidate(self, retry: RetryCandidate) -> bool:
        dialog = QDialog(self)
        dialog.setWindowTitle(f"{retry.measure}마디 재인식 결과")
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel(retry.comparison))
        controls = QHBoxLayout()
        for label, action in (
            ("기존 결과 듣기", lambda: self.play_measure_preview(retry.measure)),
            ("재인식 결과 듣기", lambda: self.play_measure_preview(retry.measure, retry)),
            ("정지", self.player.stop),
        ):
            button = QPushButton(label)
            button.clicked.connect(action)
            controls.addWidget(button)
        layout.addLayout(controls)

        choices = QDialogButtonBox()
        keep = choices.addButton("기존 유지", QDialogButtonBox.ButtonRole.RejectRole)
        keep.setDefault(True)
        choices.rejected.connect(dialog.reject)
        if retry.safe:
            choices.addButton("새 결과 적용", QDialogButtonBox.ButtonRole.AcceptRole)
            choices.accepted.connect(dialog.accept)
        layout.addWidget(choices)
        self.player.stop()
        try:
            return dialog.exec() == QDialog.DialogCode.Accepted
        finally:
            self.stop_measure_preview()
            dialog.deleteLater()

    def measure_retry_succeeded(self, retry: RetryCandidate) -> None:
        if self.current_output is None:
            return
        if not self.confirm_retry_candidate(retry):
            status = "다시 인식" if retry.safe else "재인식 결과 적용 불가"
            self.review_status_label.setText(f"검토 상태: {status}")
            return
        try:
            self.result_score = apply_retry_candidate(
                self.current_output.with_suffix(".musicxml"),
                self.current_output,
                retry,
                self.result_bpm_input.value(),
                self.result_navigation_plan,
            )
        except Exception as error:
            QMessageBox.critical(self, "재인식 결과 적용 실패", str(error))
            return

        self.review_status[str(retry.measure)] = "정상"
        self.current_output.with_suffix(".review.json").write_text(
            json.dumps(self.review_status, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        output = self.current_output
        self.show_result(output)
        for row in range(self.problem_list.count()):
            item = self.problem_list.item(row)
            if item.data(Qt.ItemDataRole.UserRole).measure == retry.measure:
                self.problem_list.setCurrentRow(row)
                break
        QMessageBox.information(
            self,
            "재인식 적용 완료",
            f"{retry.measure}마디를 교체하고 MIDI를 다시 저장했습니다.",
        )

    def measure_retry_failed(self, error: str) -> None:
        self.review_status_label.setText("검토 상태: 재인식 실패")
        QMessageBox.warning(self, "마디 재인식 실패", error)

    def play_result(self) -> None:
        if self.current_output is not None:
            self.stop_measure_preview()
            source = QUrl.fromLocalFile(str(self.current_output.resolve()))
            if self.player.source() != source:
                self.player.setSource(source)
            self.player.play()

    def start_new_conversion(self) -> None:
        self.stop_measure_preview()
        self.player.stop()
        self.pages.setCurrentWidget(self.input_page)

    def playback_failed(self, _error, message: str) -> None:
        if self.preview_audio is not None:
            QMessageBox.warning(
                QApplication.activeModalWidget() or self,
                "미리듣기 실패",
                f"미리듣기를 재생할 수 없습니다.\n{message}",
            )
            return
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
        if (
            hasattr(self, "worker") and self.worker.isRunning()
        ) or (
            hasattr(self, "retry_worker") and self.retry_worker.isRunning()
        ):
            QMessageBox.information(
                self,
                "변환 중",
                "변환이 끝난 뒤 프로그램을 닫아주세요.",
            )
            event.ignore()
            return
        self.player.stop()
        self.stop_measure_preview()
        event.accept()


def main() -> None:
    app = QApplication(sys.argv)
    window = Score2MidiWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
