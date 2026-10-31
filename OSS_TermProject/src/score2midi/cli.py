from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from music21 import converter


def convert_score(image: Path, output: Path, audiveris: str) -> None:
    """Convert a score image to MIDI using Audiveris and music21."""
    if not image.is_file():
        raise FileNotFoundError(f"악보 이미지 파일을 찾을 수 없습니다: {image}")

    executable = shutil.which(audiveris)
    if executable is None and not Path(audiveris).is_file():
        raise FileNotFoundError(
            "Audiveris를 찾을 수 없습니다. "
            "--audiveris 옵션에 실행 파일 경로를 지정하세요."
        )
    audiveris_command = executable or str(Path(audiveris).resolve())

    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="score2midi-") as temp_dir:
        result = subprocess.run(
            [
                audiveris_command,
                "-batch",
                "-transcribe",
                "-export",
                "-output",
                temp_dir,
                str(image.resolve()),
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            details = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"Audiveris 실행 실패:\n{details}")

        candidates = (
            list(Path(temp_dir).rglob("*.mxl"))
            + list(Path(temp_dir).rglob("*.musicxml"))
            + list(Path(temp_dir).rglob("*.xml"))
        )
        if not candidates:
            raise FileNotFoundError("Audiveris가 MusicXML 파일을 만들지 않았습니다.")

        score = converter.parse(str(candidates[0]))
        score.write("midi", fp=str(output))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="악보 이미지에서 MIDI 파일을 만듭니다."
    )
    parser.add_argument("image", type=Path, help="입력 악보 이미지")
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
