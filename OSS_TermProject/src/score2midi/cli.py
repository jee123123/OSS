from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from music21 import converter, instrument
from PIL import Image

MIN_PAGE_LONG_EDGE = 3000


def convert_score(image: Path, output: Path, audiveris: str) -> None:
    """Convert a score image to MIDI using Audiveris and music21."""
    convert_pages([image], output, audiveris)


def convert_pages(
    images: list[Path],
    output: Path,
    audiveris: str,
) -> None:
    """Convert ordered score pages into one MIDI file."""
    if not images:
        raise ValueError("변환할 악보가 없습니다.")

    for image in images:
        if not image.is_file():
            raise FileNotFoundError(
                f"악보 파일을 찾을 수 없습니다: {image}"
            )

    pdf_inputs = [
        image for image in images if image.suffix.lower() == ".pdf"
    ]
    if pdf_inputs and len(images) != 1:
        raise ValueError(
            "PDF는 한 번에 하나만 선택할 수 있으며 "
            "이미지 파일과 함께 사용할 수 없습니다."
        )

    executable = shutil.which(audiveris)
    if executable is None and not Path(audiveris).is_file():
        raise FileNotFoundError(
            "Audiveris를 찾을 수 없습니다. "
            "--audiveris 옵션에 실행 파일 경로를 지정하세요."
        )
    audiveris_command = executable or str(Path(audiveris).resolve())

    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="score2midi-") as temp_dir:
        if pdf_inputs:
            input_document = pdf_inputs[0].resolve()
        else:
            pages = []
            for image in images:
                with Image.open(image) as source:
                    page = source.convert("RGB")
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
                    pages.append(page)

            input_document = Path(temp_dir) / "score.pdf"
            try:
                pages[0].save(
                    input_document,
                    "PDF",
                    save_all=True,
                    append_images=pages[1:],
                    resolution=300,
                )
            finally:
                for page in pages:
                    page.close()

        result = subprocess.run(
            [
                audiveris_command,
                "-batch",
                "-transcribe",
                "-export",
                "-output",
                temp_dir,
                str(input_document),
            ],
            capture_output=True,
            check=False,
        )

        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace").strip()
            stdout = result.stdout.decode("utf-8", errors="replace").strip()
            details = stderr or stdout or "자세한 오류 메시지가 없습니다."
            raise RuntimeError(f"Audiveris 실행 실패:\n{details}")

        candidates = (
            list(Path(temp_dir).rglob("*.mxl"))
            + list(Path(temp_dir).rglob("*.musicxml"))
            + list(Path(temp_dir).rglob("*.xml"))
        )
        if not candidates:
            raise FileNotFoundError("Audiveris가 MusicXML 파일을 만들지 않았습니다.")

        score = converter.parse(str(candidates[0]))
        for part in score.parts:
            detected_instruments = list(
                part.recurse().getElementsByClass(instrument.Instrument)
            )
            if detected_instruments:
                for detected in detected_instruments:
                    detected.midiProgram = 0
            else:
                part.insert(0, instrument.Piano())
        score.write("midi", fp=str(output))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="악보 이미지 또는 PDF에서 MIDI 파일을 만듭니다."
    )
    parser.add_argument("image", type=Path, help="입력 악보 이미지 또는 PDF")
    parser.add_argument("output", type=Path, help="저장할 MIDI 파일")
    parser.add_argument(
        "--audiveris",
        default="audiveris",
        help="Audiveris 실행 파일 경로",
    )
    args = parser.parse_args()

    try:
        convert_score(args.image, args.output, args.audiveris)
    except (FileNotFoundError, RuntimeError, OSError) as error:
        print(f"오류: {error}", file=sys.stderr)
        return 1

    print(f"MIDI 파일을 저장했습니다: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
