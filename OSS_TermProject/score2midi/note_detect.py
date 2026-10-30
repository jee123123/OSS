"""Find filled notehead candidates near one detected staff."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from staff_detect import detect_staff_lines, group_lines_into_staves


def find_staff_x_bounds(image, staff_lines):
    """Return the horizontal extent shared by at least three staff lines."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        12,
    )

    width = gray.shape[1]
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(20, width // 25), 1)
    )
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    line_hits = np.zeros(width, dtype=np.uint8)
    for y in staff_lines:
        y1 = max(0, y - 1)
        y2 = min(gray.shape[0], y + 2)
        line_hits += np.any(horizontal[y1:y2, :] > 0, axis=0)

    staff_x = np.flatnonzero(line_hits >= 1)
    if staff_x.size == 0:
        return 0, width - 1

    return int(staff_x[0]), int(staff_x[-1])


def find_notehead_candidates(image, staff_lines):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    original_staff_lines = np.asarray(staff_lines, dtype=float)
    original_staff_space = float(np.median(np.diff(original_staff_lines)))
    scale = 10.0 / original_staff_space

    # 이진화 전에 회색조 이미지를 확대해 작은 음표 모양을 보존합니다.
    gray = cv2.resize(
        gray,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC,
    )

    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        12,
    )

    image_height, image_width = binary.shape
    staff_space = 10.0
    staff_lines = original_staff_lines * scale
    top, bottom = staff_lines[0], staff_lines[-1]

    # 오선 주변의 세로 범위와 덧줄 음표 영역을 설정합니다.
    y1 = max(0, int(top - staff_space * 1.4))
    y2 = min(image_height, int(bottom + staff_space * 1.9))

    # 수평 형태학 연산으로 오선 마스크를 구합니다.
    staff_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(20, image_width // 25), 1)
    )
    staff_mask = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, staff_kernel
    )

    # 선택한 오선의 실제 가로 범위를 구합니다.
    xs = np.flatnonzero(
        np.sum(staff_mask[int(top):int(bottom) + 1, :], axis=0) > 0
    )
    if xs.size == 0:
        return []

    staff_x0 = int(xs[0])
    staff_x1 = int(xs[-1])

    # 음표 머리 크기에 맞춘 타원 커널로 후보를 추출합니다.
    kernel_width = max(3, int(round(staff_space * 0.8)) | 1)
    kernel_height = max(3, int(round(staff_space * 0.6)) | 1)
    head_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_width, kernel_height)
    )
    # 오선을 제거한 보조 이미지를 만듭니다.
    notes_without_staff = cv2.subtract(binary, staff_mask)

    # 원본과 오선 제거 이미지에서 각각 음표 머리 후보를 찾습니다.
    heads_from_original = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, head_kernel
    )
    heads_from_clean = cv2.morphologyEx(
        notes_without_staff, cv2.MORPH_OPEN, head_kernel
    )

    # 두 경로의 후보를 기존 크기·모양·기둥 필터에 전달합니다.
    heads_only = cv2.bitwise_or(
        heads_from_original, heads_from_clean
    )

    # 오선 시작부의 큰 기호를 찾아 음자리표 영역을 건너뜁니다.
    header_x2 = min(
        staff_x1,
        staff_x0 + int(staff_space * 14),
    )
    band = binary[y1:y2, staff_x0:header_x2]
    band_staff = staff_mask[y1:y2, staff_x0:header_x2]
    band_without_staff = cv2.subtract(band, band_staff)

    count, _, stats, _ = cv2.connectedComponentsWithStats(
        band_without_staff, connectivity=8
    )

    header_end = staff_x0
    for i in range(1, count):
        bx, by, bw, bh, area = stats[i]
        if bh >= staff_space * 2.5 and bx < staff_space * 4:
            header_end = max(header_end, staff_x0 + bx + bw)

    header_end += int(staff_space * 0.3)

    # 음표 머리 후보를 오선 범위 안에서만 분석합니다.
    roi = np.zeros_like(heads_only)
    roi[y1:y2, header_end:max(header_end, staff_x1 - 2)] = 255
    candidates = cv2.bitwise_and(heads_only, roi)

    count, _, stats, _ = cv2.connectedComponentsWithStats(
        candidates, connectivity=8
    )

    def stem_length(x, y, width, height):
        """음표 머리 가장자리에서 위·아래로 이어지는 기둥 길이."""
        best = 0
        center_y = y + height // 2

        for edge_x in (x, x + width - 1):
            for sample_x in (edge_x - 1, edge_x, edge_x + 1):
                if not 0 <= sample_x < image_width:
                    continue

                for direction in (-1, 1):
                    length = 0
                    current_y = center_y

                    while (
                        0 <= current_y < image_height
                        and binary[current_y, sample_x] > 0
                    ):
                        length += 1
                        current_y += direction

                    best = max(best, length)

        return best

    def is_on_staff_position(center_y):
        step = round((bottom - center_y) / (staff_space / 2))
        expected_y = bottom - step * (staff_space / 2)
        return abs(center_y - expected_y) <= staff_space * 0.3

    boxes = []
    for i in range(1, count):
        x, y, width, height, area = stats[i]

        # 후보 크기와 면적을 오선 간격 기준으로 확인합니다.
        if not (
            staff_space * 0.6 <= width <= staff_space * 2.0
            and staff_space * 0.5 <= height <= staff_space * 1.6
        ):
            continue

        fill_ratio = area / float(width * height)
        if fill_ratio < 0.55:
            continue

        # 기둥이 충분히 이어지는 후보만 남깁니다.
        stem = stem_length(x, y, width, height)
        if stem < staff_space * 2.0 and fill_ratio < 0.70:
            continue

        center_x = x + width // 2
        center_y = y + height // 2
        if not is_on_staff_position(center_y):
            continue
        boxes.append(
            (
                int(round(x / scale)),
                int(round(y / scale)),
                int(round(width / scale)),
                int(round(height / scale)),
                int(round(center_x / scale)),
                int(round(center_y / scale)),
            )
        )

    # 속이 빈 음표 머리는 오선 제거 후 닫힘 연산으로 윤곽을 연결합니다.
    outline_source = cv2.subtract(binary, staff_mask)
    outline_kernel_width = max(
        3, int(round(staff_space * 0.9)) | 1
    )
    outline_kernel_height = max(
        3, int(round(staff_space * 0.7)) | 1
    )
    outline_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (outline_kernel_width, outline_kernel_height),
    )
    outline_mask = cv2.morphologyEx(
        outline_source, cv2.MORPH_CLOSE, outline_kernel
    )
    # 채워진 음표 머리 크기의 열림 연산으로 기둥을 분리합니다.
    outline_mask = cv2.morphologyEx(
        outline_mask, cv2.MORPH_OPEN, head_kernel
    )
    outline_mask = cv2.bitwise_and(outline_mask, roi)

    count, _, outline_stats, _ = cv2.connectedComponentsWithStats(
        outline_mask, connectivity=8
    )

    for i in range(1, count):
        x, y, width, height, area = outline_stats[i]

        if not (
            staff_space * 0.6 <= width <= staff_space * 2.0
            and staff_space * 0.5 <= height <= staff_space * 1.6
        ):
            continue

        if area / float(width * height) < 0.55:
            continue

        aspect_ratio = width / height
        if not 0.9 <= aspect_ratio <= 1.8:
            continue

        patch = outline_source[y:y + height, x:x + width]
        inner_x1 = int(width * 0.3)
        inner_x2 = max(inner_x1 + 1, int(width * 0.7))
        inner_y1 = int(height * 0.3)
        inner_y2 = max(inner_y1 + 1, int(height * 0.7))

        center = patch[inner_y1:inner_y2, inner_x1:inner_x2]
        center_ink = np.count_nonzero(center) / center.size

        border = patch.copy()
        border[inner_y1:inner_y2, inner_x1:inner_x2] = 0
        border_ink = np.count_nonzero(border) / border.size

        center_x = x + width // 2
        center_y = y + height // 2
        if not is_on_staff_position(center_y):
            continue
        if center_ink > 0.50 or border_ink < 0.25:
            continue

        original_center_x = int(round(center_x / scale))
        original_center_y = int(round(center_y / scale))

        # 이미 채워진 음표로 검출한 위치와 겹치면 중복 추가하지 않습니다.
        duplicate = any(
            abs(original_center_x - box[4]) < original_staff_space
            and abs(original_center_y - box[5]) < original_staff_space
            for box in boxes
        )
        if duplicate:
            continue

        boxes.append(
            (
                int(round(x / scale)),
                int(round(y / scale)),
                int(round(width / scale)),
                int(round(height / scale)),
                original_center_x,
                original_center_y,
            )
        )

    return boxes


def main():
    parser = argparse.ArgumentParser(
        description="Find notehead candidates in score staves"
    )
    parser.add_argument("image", type=Path, help="Input score image")
    parser.add_argument(
        "--staff",
        type=int,
        default=None,
        help="Process only this staff number; by default, process all staves",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("note-candidates.png")
    )
    args = parser.parse_args()

    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Could not open image: {args.image}")

    lines = detect_staff_lines(image)
    staves = group_lines_into_staves(lines)
    if not staves:
        raise SystemExit("No five-line staves were detected.")

    if args.staff is None:
        staff_numbers = range(1, len(staves) + 1)
    elif 1 <= args.staff <= len(staves):
        staff_numbers = [args.staff]
    else:
        raise SystemExit(
            f"Staff {args.staff} not found; detected {len(staves)} staves."
        )

    all_boxes = []
    for staff_number in staff_numbers:
        boxes = find_notehead_candidates(
            image, staves[staff_number - 1]
        )
        print(f"Staff {staff_number}: {len(boxes)} notehead candidates")
        all_boxes.extend(
            (staff_number, *box) for box in boxes
        )

    result = image.copy()
    for staff_number, x, y, width, height, center_x, center_y in all_boxes:
        cv2.rectangle(
            result, (x, y), (x + width, y + height), (0, 200, 0), 1
        )
        cv2.circle(result, (center_x, center_y), 2, (0, 0, 255), -1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), result):
        raise SystemExit(f"Could not write image: {args.output}")

    print(f"Total: {len(all_boxes)} notehead candidates")
    print(f"Saved overlay to {args.output}")


if __name__ == "__main__":
    main()
