"""Image preparation for optical music recognition."""

from pathlib import Path
from statistics import median

import pypdfium2 as pdfium
from PIL import Image, ImageDraw, ImageOps

MIN_PAGE_LONG_EDGE = 3000
PDF_RENDER_DPI = 300
REFERENCE_STAFF_ROW_RATIO = 0.20
SHORT_STAFF_ROW_RATIO = 0.08
SHORT_SYSTEM_RATIO = 0.55


def _row_bands(values: list[int], threshold: int) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start = None

    for index, value in enumerate(values + [0]):
        if value >= threshold and start is None:
            start = index
        elif value < threshold and start is not None:
            bands.append((start, index - 1))
            start = None

    return bands


def _staff_groups(
    page: Image.Image,
    row_ratio: float,
) -> list[dict[str, object]]:
    binary = page.point(
        lambda value: 0 if value < 180 else 255,
        mode="L",
    )
    ink = ImageOps.invert(binary)
    projection = list(
        ink.resize((1, page.height), Image.Resampling.BOX).tobytes()
    )
    bands = _row_bands(
        projection,
        round(255 * row_ratio),
    )
    centers = [
        max(
            range(start, end + 1),
            key=projection.__getitem__,
        )
        for start, end in bands
    ]
    groups: list[dict[str, object]] = []
    index = 0

    while index <= len(centers) - 5:
        lines = centers[index:index + 5]
        gaps = [right - left for left, right in zip(lines, lines[1:])]
        spacing = median(gaps)
        tolerance = max(1, round(spacing * 0.25))

        if (
            spacing >= 3
            and max(gaps) - min(gaps) <= tolerance
            and spacing <= page.height * 0.03
        ):
            boxes = [
                ink.crop((0, y, page.width, y + 1)).getbbox()
                for y in lines
            ]
            boxes = [box for box in boxes if box is not None]

            if len(boxes) == 5:
                left = round(median(box[0] for box in boxes))
                right = round(median(box[2] for box in boxes))
                groups.append(
                    {
                        "lines": lines,
                        "spacing": spacing,
                        "left": left,
                        "right": right,
                        "width": right - left,
                    }
                )
                index += 5
                continue

        index += 1

    return groups


def extend_short_final_system(page: Image.Image) -> Image.Image:
    groups = _staff_groups(page, REFERENCE_STAFF_ROW_RATIO)
    short_groups = _staff_groups(page, SHORT_STAFF_ROW_RATIO)

    for candidate in short_groups:
        candidate_y = median(candidate["lines"])
        if not any(
            abs(candidate_y - median(group["lines"]))
            <= candidate["spacing"]
            for group in groups
        ):
            groups.append(candidate)

    groups.sort(key=lambda group: median(group["lines"]))

    if len(groups) < 4:
        return page

    upper, lower = groups[-2:]
    references = groups[:-2]
    reference_width = median(group["width"] for group in references)
    reference_left = median(group["left"] for group in references)
    target_right = round(median(group["right"] for group in references))
    upper_width = upper["width"]
    lower_width = lower["width"]
    spacing = round(median([upper["spacing"], lower["spacing"]]))

    if (
        upper_width >= reference_width * SHORT_SYSTEM_RATIO
        or lower_width >= reference_width * SHORT_SYSTEM_RATIO
        or abs(upper_width - lower_width) > spacing * 3
        or abs(upper["left"] - reference_left) > spacing * 3
        or abs(lower["left"] - reference_left) > spacing * 3
    ):
        return page

    old_right = round(median([upper["right"], lower["right"]]))
    if target_right <= old_right + spacing * 4:
        return page

    corrected = page.copy()
    draw = ImageDraw.Draw(corrected)
    all_lines = upper["lines"] + lower["lines"]
    top = min(all_lines)
    bottom = max(all_lines)
    clear_width = max(3, round(spacing * 1.25))
    line_width = max(1, round(spacing * 0.15))
    draw.rectangle(
        (
            old_right - clear_width,
            top - line_width,
            old_right + line_width,
            bottom + line_width,
        ),
        fill=255,
    )

    for y in all_lines:
        draw.line(
            (old_right - clear_width, y, target_right - 1, y),
            fill=0,
            width=line_width,
        )

    thick_width = max(2, line_width * 3)
    thick_x = target_right - thick_width
    thin_x = thick_x - max(2, round(spacing * 0.55))
    draw.line(
        (thin_x, top, thin_x, bottom),
        fill=0,
        width=line_width,
    )
    draw.rectangle(
        (thick_x, top, target_right - 1, bottom),
        fill=0,
    )
    return corrected


def preprocess_page(source: Image.Image) -> Image.Image:
    """Preserve score symbols while ensuring a usable resolution."""
    page = ImageOps.grayscale(source)
    page = extend_short_final_system(page)

    longest_edge = max(page.size)
    if longest_edge < MIN_PAGE_LONG_EDGE:
        scale = MIN_PAGE_LONG_EDGE / longest_edge
        page = page.resize(
            (
                round(page.width * scale),
                round(page.height * scale),
            ),
            Image.Resampling.LANCZOS,
        )

    return page


def preprocess_pdf(
    source: Path,
    dpi: int = PDF_RENDER_DPI,
) -> list[Image.Image]:
    """Render PDF pages without destructive sharpening."""
    document = pdfium.PdfDocument(source)
    pages: list[Image.Image] = []

    try:
        for page_index in range(len(document)):
            page = document[page_index]
            bitmap = page.render(scale=dpi / 72)
            rendered = bitmap.to_pil()

            try:
                pages.append(preprocess_page(rendered))
            finally:
                rendered.close()
                bitmap.close()
                page.close()
    finally:
        document.close()

    if not pages:
        raise ValueError("PDF에 변환할 페이지가 없습니다.")

    return pages
