"""Image preparation for optical music recognition."""

from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image, ImageOps

MIN_PAGE_LONG_EDGE = 3000
PDF_RENDER_DPI = 300


def preprocess_page(source: Image.Image) -> Image.Image:
    """Preserve score symbols while ensuring a usable resolution."""
    page = ImageOps.grayscale(source)

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
