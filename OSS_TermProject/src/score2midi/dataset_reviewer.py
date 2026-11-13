"""GUI for reviewing automatically extracted notehead candidates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QPixmap, QShortcut
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
    def __init__(self, dataset: Path | None = None):
        super().__init__()
        self.setWindowTitle("Score2MIDI 학습 데이터 검토")
        self.resize(620, 620)
        self.session: ReviewDataset | None = None
        self.pending: list[int] = []
        self.position = 0
        self.history: list[int] = []
        self.shortcuts: list[QShortcut] = []

        self.dataset_label = QLabel("데이터셋을 선택하세요.")
        self.image_label = QLabel("검토할 후보가 없습니다.")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(360, 360)
        self.image_label.setStyleSheet(
            "QLabel { background: white; border: 1px solid #999; }"
        )
        self.info_label = QLabel("")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress = QProgressBar()

        self.choose_button = QPushButton("데이터셋 선택")
        self.notehead_button = QPushButton("음표 (1)")
        self.other_button = QPushButton("비음표 (2)")
        self.skip_button = QPushButton("건너뛰기 (Space)")
        self.undo_button = QPushButton("되돌리기 (Ctrl+Z)")

        self.choose_button.clicked.connect(self.choose_dataset)
        self.notehead_button.clicked.connect(
            lambda: self.classify_current("notehead")
        )
        self.other_button.clicked.connect(
            lambda: self.classify_current("other")
        )
        self.skip_button.clicked.connect(self.skip_current)
        self.undo_button.clicked.connect(self.undo)

        for sequence, callback in (
            ("1", lambda: self.classify_current("notehead")),
            ("2", lambda: self.classify_current("other")),
            ("Space", self.skip_current),
            ("Ctrl+Z", self.undo),
        ):
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.activated.connect(callback)
            self.shortcuts.append(shortcut)

        buttons = QHBoxLayout()
        buttons.addWidget(self.notehead_button)
        buttons.addWidget(self.other_button)
        buttons.addWidget(self.skip_button)

        bottom_buttons = QHBoxLayout()
        bottom_buttons.addWidget(self.choose_button)
        bottom_buttons.addWidget(self.undo_button)

        layout = QVBoxLayout()
        layout.addWidget(self.dataset_label)
        layout.addWidget(self.image_label)
        layout.addWidget(self.info_label)
        layout.addWidget(self.progress)
        layout.addLayout(buttons)
        layout.addLayout(bottom_buttons)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        self.set_review_controls_enabled(False)
        self.undo_button.setEnabled(False)

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

        self.pending = self.session.review_indices()
        self.position = 0
        self.history.clear()
        self.dataset_label.setText(f"데이터셋: {self.session.root}")
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

        try:
            self.session.classify(index, label)
        except (FileNotFoundError, FileExistsError, OSError, ValueError) as error:
            QMessageBox.critical(self, "저장 오류", str(error))
            return

        self.history.append(index)
        self.pending.pop(self.position)
        if self.pending and self.position >= len(self.pending):
            self.position = 0
        self.show_current()

    def skip_current(self) -> None:
        if not self.pending:
            return
        self.position = (self.position + 1) % len(self.pending)
        self.show_current()

    def undo(self) -> None:
        if self.session is None or not self.history:
            return

        index = self.history.pop()
        try:
            self.session.classify(index, "review")
        except (FileNotFoundError, FileExistsError, OSError, ValueError) as error:
            QMessageBox.critical(self, "되돌리기 오류", str(error))
            return

        self.pending.insert(0, index)
        self.position = 0
        self.show_current()

    def show_current(self) -> None:
        if self.session is None:
            return

        total = len(self.session.rows)
        remaining = len(self.pending)
        self.progress.setRange(0, total)
        self.progress.setValue(total - remaining)
        self.progress.setFormat(
            f"분류 완료 {total - remaining}/{total} · 남은 검토 {remaining}개"
        )
        self.undo_button.setEnabled(bool(self.history))

        index = self.current_index()
        if index is None:
            self.image_label.setPixmap(QPixmap())
            self.image_label.setText("모든 후보 검토가 끝났습니다.")
            self.info_label.setText("")
            self.set_review_controls_enabled(False)
            return

        row = self.session.rows[index]
        image_path = self.session.root / row["path"]
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            self.image_label.setText(f"이미지를 열 수 없습니다.\n{image_path}")
        else:
            self.image_label.setText("")
            self.image_label.setPixmap(
                pixmap.scaled(
                    320,
                    320,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.FastTransformation,
                )
            )

        self.info_label.setText(
            f"{row['path']}\n"
            f"{row.get('page', '?')}쪽 · {row.get('measure', '?')}마디 · "
            f"파트 {row.get('part', '?')} · 신뢰도 {row.get('confidence', '?')}"
        )
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
    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = DatasetReviewerWindow(args.dataset)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
