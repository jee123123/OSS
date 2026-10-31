"""Detect five-line music staves in a printed score image."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def detect_staff_lines(image: np.ndarray) -> list[int]:
    """Return y-coordinates of horizontal staff lines in a grayscale/BGR image.

    Nearby rows belonging to the same printed line are merged. This first
    version assumes a mostly horizontal, clean printed score.
    """
    if image is None or image.size == 0:
        raise ValueError("Image is empty")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    if gray.ndim != 2:
        raise ValueError("Expected a grayscale or color image")

    # Binarize so dark notation becomes white foreground for morphology.
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 12
    )
    width = gray.shape[1]
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(20, width // 25), 1)
    )
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)

    row_strength = np.count_nonzero(horizontal, axis=1)
    # 큰 이미지는 짧은 빔을 제외하고, 작은 이미지는 흐린 오선을 살립니다.
    minimum_ratio = 0.30 if width >= 600 else 0.08
    threshold = max(10, int(width * minimum_ratio))
    candidate_rows = np.flatnonzero(row_strength >= threshold)
    if candidate_rows.size == 0:
        return []

    groups: list[list[int]] = [[int(candidate_rows[0])]]
    for y in candidate_rows[1:]:
        if int(y) - groups[-1][-1] <= 2:
            groups[-1].append(int(y))
        else:
            groups.append([int(y)])

    return [int(round(sum(group) / len(group))) for group in groups]


def group_lines_into_staves(line_ys: list[int]) -> list[list[int]]:
    """Group consecutive, evenly spaced line positions into five-line staves."""
    ys = sorted(int(y) for y in line_ys)
    staves: list[list[int]] = []
    index = 0

    while index + 4 < len(ys):
        candidate = ys[index : index + 5]
        gaps = np.diff(candidate)

        # A staff has four similar gaps. The ratio allows small scan and print
        # variations while rejecting the larger gap between two staves.
        if gaps.min() > 0 and gaps.max() <= gaps.min() * 1.6:
            staves.append(candidate)
            index += 5
        else:
            index += 1

    return staves


def annotate_staves(image: np.ndarray, staves: list[list[int]]) -> np.ndarray:
    """Draw detected lines, a boundary box, and a number for each staff."""
    if image.ndim == 2:
        result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        result = image.copy()

    height, width = result.shape[:2]
    for staff_number, lines in enumerate(staves, start=1):
        for y in lines:
            cv2.line(result, (0, y), (width - 1, y), (0, 0, 255), 1)

        top = max(0, lines[0] - 3)
        bottom = min(height - 1, lines[-1] + 3)
        cv2.rectangle(result, (1, top), (width - 2, bottom), (255, 120, 0), 1)
        label_y = max(14, top - 4)
        cv2.putText(
            result,
            f"Staff {staff_number}",
            (6, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 120, 0),
            1,
            cv2.LINE_AA,
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect staff lines in a score image")
    parser.add_argument("image", type=Path, help="Input score image (PNG/JPEG)")
    parser.add_argument("--output", type=Path, default=Path("staff-overlay.png"))
    args = parser.parse_args()

    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Could not open image: {args.image}")

    ys = detect_staff_lines(image)
    staves = group_lines_into_staves(ys)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), annotate_staves(image, staves)):
        raise SystemExit(f"Could not write image: {args.output}")

    print(f"Detected {len(ys)} horizontal line positions.")
    print(f"Grouped into {len(staves)} five-line staves.")
    for staff_number, lines in enumerate(staves, start=1):
        print(f"  Staff {staff_number}: y={lines}")
    if not staves:
        print("No five-line staves found. Check the image or line-detection threshold.")
    print(f"Saved overlay to {args.output}")


if __name__ == "__main__":
    main()
