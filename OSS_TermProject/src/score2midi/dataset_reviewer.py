"""GUI for reviewing automatically extracted notehead candidates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QKeySequence, QPainter, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from score2midi.ml_dataset import ReviewDataset


class DatasetReviewerWindow(QMainWindow):
    def __init__(
        self,
        dataset: Path | None = None,
        model: Path | None = None,
    ):
        super().__init__()
        self.setWindowTitle("Score2MIDI 학습 데이터 검토")
        self.resize(820, 700)
        self.session: ReviewDataset | None = None
        self.pending: list[int] = []
        self.position = 0
        self.history: list[tuple[int, str]] = []
        self.shortcuts: list[QShortcut] = []
        self.mode_label = "review"
        self.mode_total = 0
        self.predictor = None
        self.predicted_label: str | None = None

        self.dataset_label = QLabel("데이터셋을 선택하세요.")
        self.help_label = QLabel(
            "음표: 빨간 상자 중앙에 검거나 빈 타원형 머리가 있음 "
            "(줄기·빔·오선이 함께 보여도 음표).\n"
            "비음표: 중앙에 빔·오선·줄기·세로줄·이음줄·점·"
            "임시표·글자만 있음. 애매하면 Space로 건너뛰세요."
        )
        self.help_label.setWordWrap(True)
        self.image_label = QLabel("검토할 후보가 없습니다.")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(620, 360)
        self.image_label.setStyleSheet(
            "QLabel { background: white; border: 1px solid #999; }"
        )
        self.info_label = QLabel("")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.suggestion_label = QLabel("모델 추천 없음")
        self.suggestion_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress = QProgressBar()

        self.choose_button = QPushButton("데이터셋 선택")
        self.notehead_button = QPushButton("음표 (1)")
        self.other_button = QPushButton("비음표 (2)")
        self.skip_button = QPushButton("건너뛰기 (Space)")
        self.undo_button = QPushButton("되돌리기 (Ctrl+Z)")
        self.review_mode_button = QPushButton("미검토 후보")
        self.other_mode_button = QPushButton("비음표 다시 검토")
        self.apply_suggestion_button = QPushButton("추천 적용 (Enter)")

        self.choose_button.clicked.connect(self.choose_dataset)
        self.notehead_button.clicked.connect(
            lambda: self.classify_current("notehead")
        )
        self.other_button.clicked.connect(
            lambda: self.classify_current("other")
        )
        self.skip_button.clicked.connect(self.skip_current)
        self.undo_button.clicked.connect(self.undo)
        self.review_mode_button.clicked.connect(
            lambda: self.set_review_mode("review")
        )
        self.other_mode_button.clicked.connect(
            lambda: self.set_review_mode("other")
        )
        self.apply_suggestion_button.clicked.connect(self.apply_suggestion)

        for sequence, callback in (
            ("1", lambda: self.classify_current("notehead")),
            ("2", lambda: self.classify_current("other")),
            ("Space", self.skip_current),
            ("Ctrl+Z", self.undo),
            ("Return", self.apply_suggestion),
        ):
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.activated.connect(callback)
            self.shortcuts.append(shortcut)

        buttons = QHBoxLayout()
        buttons.addWidget(self.notehead_button)
        buttons.addWidget(self.other_button)
        buttons.addWidget(self.skip_button)
        buttons.addWidget(self.apply_suggestion_button)

        bottom_buttons = QHBoxLayout()
        bottom_buttons.addWidget(self.choose_button)
        bottom_buttons.addWidget(self.undo_button)

        mode_buttons = QHBoxLayout()
        mode_buttons.addWidget(self.review_mode_button)
        mode_buttons.addWidget(self.other_mode_button)

        layout = QVBoxLayout()
        layout.addWidget(self.dataset_label)
        layout.addWidget(self.help_label)
        layout.addWidget(self.image_label)
        layout.addWidget(self.info_label)
        layout.addWidget(self.suggestion_label)
        layout.addWidget(self.progress)
        layout.addLayout(buttons)
        layout.addLayout(mode_buttons)
        layout.addLayout(bottom_buttons)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        self.set_review_controls_enabled(False)
        self.undo_button.setEnabled(False)
        self.apply_suggestion_button.setEnabled(False)

        if model is not None:
            try:
                from score2midi.note_classifier import NoteheadPredictor

                self.predictor = NoteheadPredictor(model)
            except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
                QMessageBox.warning(self, "모델 불러오기 실패", str(error))

        if dataset is not None:
            self.load_dataset(dataset)

    def choose_dataset(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            "학습 데이터 폴더 선택",
            str(self.session.root if self.session else Path.cwd()),
        )
        if folder:
            self.load_dataset(Path(folder))

    def load_dataset(self, root: Path) -> None:
        try:
            self.session = ReviewDataset(root)
        except (FileNotFoundError, OSError, ValueError) as error:
            QMessageBox.critical(self, "데이터셋 오류", str(error))
            return

        self.dataset_label.setText(f"데이터셋: {self.session.root}")
        self.set_review_mode("review")

    def set_review_mode(self, label: str) -> None:
        if self.session is None:
            return
        self.mode_label = label
        self.pending = [
            index
            for index, row in enumerate(self.session.rows)
            if row["label"] == label
        ]
        self.mode_total = len(self.pending)
        self.position = 0
        self.history.clear()
        self.show_current()

    def current_index(self) -> int | None:
        if not self.pending:
            return None
        return self.pending[self.position]

    def classify_current(self, label: str) -> None:
        if self.session is None:
            return
        index = self.current_index()
        if index is None:
            return
        old_label = self.session.rows[index]["label"]

        try:
            self.session.classify(index, label)
        except (FileNotFoundError, FileExistsError, OSError, ValueError) as error:
            QMessageBox.critical(self, "저장 오류", str(error))
            return

        self.history.append((index, old_label))
        self.pending.pop(self.position)
        if self.pending and self.position >= len(self.pending):
            self.position = 0
        self.show_current()

    def skip_current(self) -> None:
        if not self.pending:
            return
        self.position = (self.position + 1) % len(self.pending)
        self.show_current()

    def apply_suggestion(self) -> None:
        if self.predicted_label is not None:
            self.classify_current(self.predicted_label)

    def undo(self) -> None:
        if self.session is None or not self.history:
            return

        index, old_label = self.history.pop()
        try:
            self.session.classify(index, old_label)
        except (FileNotFoundError, FileExistsError, OSError, ValueError) as error:
            QMessageBox.critical(self, "되돌리기 오류", str(error))
            return

        self.pending.insert(0, index)
        self.position = 0
        self.show_current()

    def show_current(self) -> None:
        if self.session is None:
            return

        remaining = len(self.pending)
        completed = self.mode_total - remaining
        self.progress.setRange(0, max(1, self.mode_total))
        self.progress.setValue(completed)
        self.progress.setFormat(
            f"이 모드에서 확인 {completed}/{self.mode_total} · "
            f"남은 후보 {remaining}개"
        )
        self.undo_button.setEnabled(bool(self.history))

        index = self.current_index()
        if index is None:
            self.image_label.setPixmap(QPixmap())
            mode_name = "미검토" if self.mode_label == "review" else "비음표"
            self.image_label.setText(
                f"{mode_name} 후보 확인이 끝났습니다."
            )
            self.info_label.setText("")
            self.suggestion_label.setText("모델 추천 없음")
            self.predicted_label = None
            self.apply_suggestion_button.setEnabled(False)
            self.set_review_controls_enabled(False)
            return

        row = self.session.rows[index]
        display_path = row.get("context_path") or row["path"]
        image_path = self.session.root / display_path
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            self.image_label.setText(f"이미지를 열 수 없습니다.\n{image_path}")
        else:
            self.image_label.setText("")
            scaled = pixmap.scaled(
                650,
                360,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            painter = QPainter(scaled)
            painter.setPen(QPen(QColor("#d00000"), 4))
            center = scaled.rect().center()
            painter.drawRect(center.x() - 30, center.y() - 23, 60, 46)
            painter.end()
            self.image_label.setPixmap(scaled)

        self.info_label.setText(
            f"{row['path']}\n"
            f"{row.get('page', '?')}쪽 · {row.get('measure', '?')}마디 · "
            f"파트 {row.get('part', '?')} · 신뢰도 {row.get('confidence', '?')}"
        )
        self.predicted_label = None
        self.apply_suggestion_button.setEnabled(False)
        if self.predictor is None:
            self.suggestion_label.setText("모델 추천 없음")
        else:
            try:
                label, confidence = self.predictor.predict(image_path)
            except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
                self.suggestion_label.setText(f"모델 추천 실패: {error}")
            else:
                self.predicted_label = label
                label_name = "음표" if label == "notehead" else "비음표"
                self.suggestion_label.setText(
                    f"모델 추천: {label_name} ({confidence:.1%}) · "
                    "맞으면 Enter"
                )
                self.apply_suggestion_button.setEnabled(True)
        self.set_review_controls_enabled(True)

    def set_review_controls_enabled(self, enabled: bool) -> None:
        for button in (
            self.notehead_button,
            self.other_button,
            self.skip_button,
        ):
            button.setEnabled(enabled)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="음표 분류 학습 데이터의 검토 후보를 분류합니다."
    )
    parser.add_argument("dataset", nargs="?", type=Path)
    parser.add_argument("--model", type=Path)
    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = DatasetReviewerWindow(args.dataset, args.model)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
