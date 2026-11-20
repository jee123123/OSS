from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image
from pypdf import PdfReader, PdfWriter

from score2midi.page_editor import (
    PageSpec,
    load_page_image,
    pages_for_file,
    prepare_conversion_inputs,
)


def test_reorders_and_rotates_selected_pdf_pages() -> None:
    with TemporaryDirectory() as directory:
        directory = Path(directory)
        source = directory / "score.pdf"
        writer = PdfWriter()
        writer.add_blank_page(width=100, height=200)
        writer.add_blank_page(width=200, height=300)
        with source.open("wb") as stream:
            writer.write(stream)

        pages = pages_for_file(source)
        inputs = prepare_conversion_inputs(
            [pages[1].rotate(90), pages[0]],
            directory,
        )
        result = PdfReader(str(inputs[0]))

        assert len(result.pages) == 2
        assert result.pages[0].rotation == 90
        assert float(result.pages[0].mediabox.width) == 200
        assert float(result.pages[1].mediabox.width) == 100


def test_rotates_image_for_conversion() -> None:
    with TemporaryDirectory() as directory:
        directory = Path(directory)
        source = directory / "page.png"
        Image.new("RGB", (20, 10), "white").save(source)
        spec = PageSpec(source).rotate(90)

        inputs = prepare_conversion_inputs([spec], directory)
        rendered = load_page_image(PageSpec(inputs[0]))
        try:
            assert rendered.size == (10, 20)
        finally:
            rendered.close()
