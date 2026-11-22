from pathlib import Path
from tempfile import TemporaryDirectory

from score2midi.measure_review import reported_measures


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
