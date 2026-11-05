"""Image preparation for optical music recognition."""

from PIL import Image, ImageFilter, ImageOps

MIN_PAGE_LONG_EDGE = 3000


def preprocess_page(source: Image.Image) -> Image.Image:
    """Improve contrast and sharpness without removing staff lines."""
    page = ImageOps.grayscale(source)
    page = ImageOps.autocontrast(page, cutoff=1)

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

    page = page.filter(
        ImageFilter.UnsharpMask(
            radius=1.2,
            percent=130,
            threshold=3,
        )
    )
    return page.convert("RGB")
