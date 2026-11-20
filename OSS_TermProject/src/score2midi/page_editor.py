from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image
from pypdf import PdfReader, PdfWriter


@dataclass(frozen=True)
class PageSpec:
    source: Path
    page_index: int | None = None
    rotation: int = 0

    @property
    def label(self) -> str:
        page = f" · {self.page_index + 1}쪽" if self.page_index is not None else ""
        rotation = f" · {self.rotation}°" if self.rotation else ""
        return f"{self.source.name}{page}{rotation}"

    def rotate(self, degrees: int) -> PageSpec:
        return replace(self, rotation=(self.rotation + degrees) % 360)


def pages_for_file(path: Path) -> list[PageSpec]:
    if path.suffix.lower() != ".pdf":
        return [PageSpec(path)]
    document = pdfium.PdfDocument(path)
    try:
        return [PageSpec(path, index) for index in range(len(document))]
    finally:
        document.close()


def load_page_image(spec: PageSpec, dpi: int = 110) -> Image.Image:
    if spec.page_index is None:
        with Image.open(spec.source) as source:
            image = source.convert("RGB")
    else:
        document = pdfium.PdfDocument(spec.source)
        try:
            page = document[spec.page_index]
            bitmap = page.render(scale=dpi / 72)
            rendered = bitmap.to_pil()
            try:
                image = rendered.convert("RGB")
            finally:
                rendered.close()
                bitmap.close()
                page.close()
        finally:
            document.close()
    if spec.rotation:
        rotated = image.rotate(-spec.rotation, expand=True)
        image.close()
        image = rotated
    return image


def prepare_conversion_inputs(
    specs: list[PageSpec],
    temporary_dir: Path,
) -> list[Path]:
    if not specs:
        raise ValueError("변환할 악보 페이지가 없습니다.")

    pdf_specs = [spec for spec in specs if spec.page_index is not None]
    if pdf_specs:
        sources = {spec.source.resolve() for spec in specs}
        if len(pdf_specs) != len(specs) or len(sources) != 1:
            raise ValueError("PDF는 이미지 파일 또는 다른 PDF와 함께 사용할 수 없습니다.")
        reader = PdfReader(str(specs[0].source))
        writer = PdfWriter()
        for spec in specs:
            page = reader.pages[spec.page_index]
            if spec.rotation:
                page.rotate(spec.rotation)
            writer.add_page(page)
        output = temporary_dir / "selected-pages.pdf"
        with output.open("wb") as stream:
            writer.write(stream)
        return [output]

    inputs: list[Path] = []
    for index, spec in enumerate(specs, 1):
        if not spec.rotation:
            inputs.append(spec.source)
            continue
        output = temporary_dir / f"page-{index}.png"
        image = load_page_image(spec)
        try:
            image.save(output, "PNG")
        finally:
            image.close()
        inputs.append(output)
    return inputs
