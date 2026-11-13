"""Train a small PyTorch classifier for notehead candidate crops."""

from __future__ import annotations

import argparse
import csv
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

IMAGE_SIZE = 32
CLASS_NAMES = ("other", "notehead")
MIN_SAMPLES_PER_CLASS = 20


class NoteheadClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 8 * 8, 32),
            nn.ReLU(),
            nn.Linear(32, len(CLASS_NAMES)),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))


class NoteheadDataset(Dataset):
    def __init__(self, root: Path, rows: list[dict[str, str]]):
        self.root = root
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        with Image.open(self.root / row["path"]) as image:
            image = image.convert("L").resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                Image.Resampling.LANCZOS,
            )
            pixels = np.asarray(image, dtype=np.float32)

        tensor = torch.from_numpy((255.0 - pixels) / 255.0).unsqueeze(0)
        return tensor, CLASS_NAMES.index(row["label"])


def _load_balanced_rows(
    dataset_dir: Path,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]], int]:
    labels_path = dataset_dir / "labels.csv"
    if not labels_path.is_file():
        raise FileNotFoundError(f"labels.csv를 찾을 수 없습니다: {labels_path}")

    with labels_path.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))

    by_label = {
        label: [row for row in rows if row.get("label") == label]
        for label in CLASS_NAMES
    }
    sample_count = min(len(items) for items in by_label.values())
    if sample_count < MIN_SAMPLES_PER_CLASS:
        counts = ", ".join(
            f"{label} {len(items)}개"
            for label, items in by_label.items()
        )
        raise ValueError(
            f"클래스마다 최소 {MIN_SAMPLES_PER_CLASS}개가 필요합니다: {counts}"
        )

    randomizer = random.Random(seed)
    training_rows = []
    validation_rows = []
    validation_count = max(1, round(sample_count * 0.2))

    for label in CLASS_NAMES:
        selected = by_label[label].copy()
        randomizer.shuffle(selected)
        selected = selected[:sample_count]
        validation_rows.extend(selected[:validation_count])
        training_rows.extend(selected[validation_count:])

    randomizer.shuffle(training_rows)
    randomizer.shuffle(validation_rows)
    return training_rows, validation_rows, sample_count


def train_notehead_classifier(
    dataset_dir: Path,
    output: Path,
    epochs: int = 20,
    batch_size: int = 16,
    seed: int = 42,
) -> dict[str, float | int | str]:
    if epochs < 1:
        raise ValueError("학습 횟수는 1 이상이어야 합니다.")
    if batch_size < 1:
        raise ValueError("배치 크기는 1 이상이어야 합니다.")

    training_rows, validation_rows, samples_per_class = (
        _load_balanced_rows(dataset_dir, seed)
    )
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NoteheadClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)
    training_loader = DataLoader(
        NoteheadDataset(dataset_dir, training_rows),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        NoteheadDataset(dataset_dir, validation_rows),
        batch_size=batch_size,
    )
    best_accuracy = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        training_loss = 0.0

        for images, labels in training_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = loss_function(model(images), labels)
            loss.backward()
            optimizer.step()
            training_loss += loss.item() * len(labels)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in validation_loader:
                images = images.to(device)
                labels = labels.to(device)
                predictions = model(images).argmax(dim=1)
                correct += int((predictions == labels).sum().item())
                total += len(labels)

        accuracy = correct / total
        average_loss = training_loss / len(training_rows)
        print(
            f"{epoch:02d}/{epochs} · "
            f"loss {average_loss:.4f} · "
            f"validation {accuracy:.1%}"
        )
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = deepcopy(model.state_dict())

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "class_names": CLASS_NAMES,
            "image_size": IMAGE_SIZE,
            "validation_accuracy": best_accuracy,
            "samples_per_class": samples_per_class,
            "training_samples": len(training_rows),
            "validation_samples": len(validation_rows),
        },
        output,
    )
    return {
        "output": str(output),
        "device": str(device),
        "validation_accuracy": best_accuracy,
        "samples_per_class": samples_per_class,
        "training_samples": len(training_rows),
        "validation_samples": len(validation_rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="검토한 음표 후보로 PyTorch 분류 모델을 학습합니다."
    )
    parser.add_argument("dataset", type=Path, help="학습 데이터 폴더")
    parser.add_argument("output", type=Path, help="모델 저장 경로(.pt)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        result = train_notehead_classifier(
            args.dataset,
            args.output,
            args.epochs,
            args.batch_size,
            args.seed,
        )
    except (FileNotFoundError, OSError, ValueError) as error:
        parser.error(str(error))

    print(
        f"모델 저장: {result['output']} · "
        f"검증 정확도 {result['validation_accuracy']:.1%} · "
        f"장치 {result['device']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
