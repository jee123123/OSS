from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image, ImageDraw

from score2midi.measure_review import load_measure_image, reported_measures
from score2midi.page_editor import PageSpec


MUSICXML = """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0">
  <defaults><page-layout><page-width>1000</page-width></page-layout></defaults>
  <part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list>
  <part id="P1">
    <measure number="1" width="300"><print new-system="yes"/></measure>
    <measure number="2" width="300"/>
    <measure number="3" width="300"><print new-page="yes"/></measure>
  </part>
</score-partwise>
"""


def test_collects_global_and_page_local_problem_measures() -> None:
    with TemporaryDirectory() as directory:
        musicxml = Path(directory) / "score.musicxml"
        musicxml.write_text(MUSICXML, encoding="utf-8")
        details = "\n".join(
            (
                "1쪽 내 2번째 마디 오른손: 음표가 누락됐을 수 있습니다.",
                "3마디 파트 1: 0.5박이 초과됩니다.",
                "인식 3마디를 3마디의 재생 순서로 전개했습니다.",
            )
        )

        issues = reported_measures(details, musicxml)

    assert [issue.measure for issue in issues] == [2, 3]
    assert len(issues[1].messages) == 1


def test_preview_uses_both_staves_in_the_selected_row(tmp_path):
    musicxml = tmp_path / "score.musicxml"
    musicxml.write_text(
        """<score-partwise><part id="P1">
        <measure number="1" width="530"><attributes><staves>2</staves></attributes></measure>
        <measure number="2" width="260"><print new-system="yes"/></measure>
        <measure number="3" width="270"><print new-system="yes"><system-layout>
          <system-margins><left-margin>260</left-margin></system-margins>
          <system-distance>-100</system-distance>
        </system-layout></print></measure>
        <measure number="4" width="530"><print new-system="yes"/></measure>
        </part></score-partwise>""",
        encoding="utf-8",
    )
    page = tmp_path / "score.png"
    with Image.new("RGB", (600, 900), "white") as image:
        draw = ImageDraw.Draw(image)
        for y in (100, 210, 360, 470, 650, 760):
            for line in range(y, y + 41, 10):
                draw.line((30, line, 560, line), fill="black", width=2)
        for x in (30, 290, 560):
            draw.line((x, 360, x, 510), fill="black")
        draw.rectangle((135, 353, 200, 359), fill="black")
        draw.point((350, 385), fill="red")
        draw.point((350, 495), fill="blue")
        draw.point((150, 385), fill="purple")
        image.save(page)

    with load_measure_image([PageSpec(page)], musicxml, 3) as preview:
        assert preview.width == 286
        assert preview.height == 230
        assert preview.getpixel((68, 65)) == (255, 0, 0)
        assert preview.getpixel((68, 175)) == (0, 0, 255)
