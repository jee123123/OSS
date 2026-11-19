from __future__ import annotations

import re
import unicodedata
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

from music21 import expressions, harmony, repeat, stream

CommandKind = Literal[
    "ds",
    "ds_al_coda",
    "ds_al_fine",
    "dss",
    "dss_al_coda",
    "dss_al_fine",
    "dc",
    "dc_al_coda",
    "dc_al_fine",
]


@dataclass(frozen=True)
class NavigationCommand:
    measure_index: int
    kind: CommandKind


@dataclass(frozen=True)
class NavigationPlan:
    measure_count: int
    segnos: tuple[int, ...] = ()
    double_segnos: tuple[int, ...] = ()
    coda_marks: tuple[int, ...] = ()
    coda_jumps: tuple[int, ...] = ()
    coda_targets: tuple[int, ...] = ()
    double_coda_jumps: tuple[int, ...] = ()
    double_coda_targets: tuple[int, ...] = ()
    fines: tuple[int, ...] = ()
    commands: tuple[NavigationCommand, ...] = ()


_REPEAT_COMMANDS: tuple[tuple[type, CommandKind], ...] = (
    (repeat.DalSegnoAlCoda, "ds_al_coda"),
    (repeat.DalSegnoAlFine, "ds_al_fine"),
    (repeat.DalSegno, "ds"),
    (repeat.DaCapoAlCoda, "dc_al_coda"),
    (repeat.DaCapoAlFine, "dc_al_fine"),
    (repeat.DaCapo, "dc"),
)


def _navigation_text_kind(text: str) -> str | None:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    key = re.sub(r"[^a-z]", "", normalized)
    commands = {
        "dssalcoda": "dss_al_coda",
        "dssalfine": "dss_al_fine",
        "dss": "dss",
        "dsalcoda": "ds_al_coda",
        "dalsegnoalcoda": "ds_al_coda",
        "dsalfine": "ds_al_fine",
        "dalsegnoalfine": "ds_al_fine",
        "ds": "ds",
        "dalsegno": "ds",
        "dcalcoda": "dc_al_coda",
        "dacapoalcoda": "dc_al_coda",
        "dcalfine": "dc_al_fine",
        "dacapoalfine": "dc_al_fine",
        "dc": "dc",
        "dacapo": "dc",
    }
    if key in commands:
        return commands[key]
    if key in {"tocoda", "alcoda"}:
        return "to_coda"
    if key == "coda":
        return "coda"
    if key in {"segnosegno", "doublesegno"}:
        return "double_segno"
    if key == "segno":
        return "segno"
    if key == "fine":
        return "fine"
    return None


def extract_navigation(part: stream.Part) -> NavigationPlan:
    measures = list(part.getElementsByClass(stream.Measure))
    segno_counts: dict[int, int] = {}
    double_segnos: set[int] = set()
    coda_marks: set[int] = set()
    coda_jumps: set[int] = set()
    coda_targets: set[int] = set()
    fines: set[int] = set()
    commands: dict[int, CommandKind] = {}

    for index, measure in enumerate(measures):
        for marker in measure.recurse().getElementsByClass(
            repeat.RepeatExpression
        ):
            command_kind = next(
                (
                    kind
                    for marker_type, kind in _REPEAT_COMMANDS
                    if isinstance(marker, marker_type)
                ),
                None,
            )
            if command_kind is not None:
                commands.setdefault(index, command_kind)
            elif isinstance(marker, repeat.Segno):
                segno_counts[index] = segno_counts.get(index, 0) + 1
            elif isinstance(marker, repeat.Coda):
                coda_marks.add(index)
            elif isinstance(marker, repeat.Fine):
                fines.add(index)

        for text in measure.recurse().getElementsByClass(
            expressions.TextExpression
        ):
            kind = _navigation_text_kind(str(text.content or ""))
            if kind in {
                "ds",
                "ds_al_coda",
                "ds_al_fine",
                "dss",
                "dss_al_coda",
                "dss_al_fine",
                "dc",
                "dc_al_coda",
                "dc_al_fine",
            }:
                commands.setdefault(index, kind)
            elif kind == "segno":
                segno_counts[index] = max(segno_counts.get(index, 0), 1)
            elif kind == "double_segno":
                double_segnos.add(index)
            elif kind == "to_coda":
                coda_jumps.add(index)
            elif kind == "coda":
                coda_targets.add(index)
            elif kind == "fine":
                fines.add(index)

    for index, count in segno_counts.items():
        if count >= 2:
            double_segnos.add(index)

    return NavigationPlan(
        measure_count=len(measures),
        segnos=tuple(sorted(
            index for index, count in segno_counts.items() if count == 1
        )),
        double_segnos=tuple(sorted(double_segnos)),
        coda_marks=tuple(sorted(coda_marks)),
        coda_jumps=tuple(sorted(coda_jumps)),
        coda_targets=tuple(sorted(coda_targets)),
        fines=tuple(sorted(fines)),
        commands=tuple(
            NavigationCommand(index, kind)
            for index, kind in sorted(commands.items())
        ),
    )


def _last_before(values: tuple[int, ...], index: int) -> int | None:
    return next((value for value in reversed(values) if value < index), None)


def _first_after(values: tuple[int, ...], index: int) -> int | None:
    return next((value for value in values if value > index), None)


def _jump_target(
    plan: NavigationPlan,
    command: NavigationCommand,
) -> int | None:
    if command.kind.startswith("dc"):
        return 0
    if command.kind.startswith("dss"):
        return _last_before(plan.double_segnos, command.measure_index)
    return _last_before(plan.segnos, command.measure_index)


def _coda_route(
    plan: NavigationPlan,
    command: NavigationCommand,
) -> tuple[int, int] | None:
    if command.kind.startswith("dss"):
        jumps = plan.double_coda_jumps
        targets = plan.double_coda_targets
        if jumps and targets:
            jump = _last_before(jumps, command.measure_index)
            target = _first_after(targets, command.measure_index)
            if jump is not None and target is not None:
                return jump, target

    jump = _last_before(
        plan.coda_jumps or plan.coda_marks,
        command.measure_index,
    )
    target = _first_after(
        plan.coda_targets or plan.coda_marks,
        command.measure_index,
    )
    if jump is None or target is None:
        return None
    return jump, target


def resolve_playback_order(
    plan: NavigationPlan,
) -> tuple[list[int], list[str]]:
    linear = list(range(plan.measure_count))
    if plan.measure_count == 0 or not plan.commands:
        return linear, []

    commands = {
        command.measure_index: command
        for command in plan.commands
        if 0 <= command.measure_index < plan.measure_count
    }
    used_commands: set[int] = set()
    warnings: list[str] = []
    order: list[int] = []
    position = 0
    pending_coda: tuple[int, int] | None = None
    stop_at_fine = False
    visited: set[tuple[int, tuple[int, ...], tuple[int, int] | None, bool]] = set()

    while 0 <= position < plan.measure_count:
        state = (
            position,
            tuple(sorted(used_commands)),
            pending_coda,
            stop_at_fine,
        )
        if state in visited:
            warnings.append("연주 순서가 반복되어 탐색을 중단했습니다.")
            break
        visited.add(state)
        order.append(position)

        if stop_at_fine and position in plan.fines:
            break
        if pending_coda is not None and position == pending_coda[0]:
            position = pending_coda[1]
            pending_coda = None
            continue

        command = commands.get(position)
        if command is None or position in used_commands:
            position += 1
            continue

        used_commands.add(position)
        target = _jump_target(plan, command)
        if target is None:
            warnings.append(
                f"{position + 1}번째 마디의 {command.kind} 명령에 필요한 "
                "세뇨를 찾지 못해 원래 순서로 연주합니다."
            )
            position += 1
            continue

        if command.kind.endswith("_al_coda"):
            pending_coda = _coda_route(plan, command)
            if pending_coda is None:
                warnings.append(
                    f"{position + 1}번째 마디의 코다 이동 위치를 찾지 "
                    "못해 원래 순서로 연주합니다."
                )
                position += 1
                continue
        elif command.kind.startswith(("ds", "dss")):
            pending_coda = _coda_route(plan, command)
        stop_at_fine = command.kind.endswith("_al_fine")
        position = target

    return order, warnings


def expand_score(
    score: stream.Score,
    order: list[int],
) -> stream.Score:
    expanded = deepcopy(score)
    for part in expanded.parts:
        measures = list(part.getElementsByClass(stream.Measure))
        if any(not 0 <= index < len(measures) for index in order):
            raise ValueError("파트마다 마디 수가 달라 연주 순서를 적용할 수 없습니다.")
        for measure in measures:
            part.remove(measure)
        for number, index in enumerate(order, 1):
            copied = deepcopy(measures[index])
            copied.number = number
            for marker in list(
                copied.recurse().getElementsByClass(repeat.RepeatExpression)
            ):
                if marker.activeSite is not None:
                    marker.activeSite.remove(marker)
            part.append(copied)
    return expanded


def expand_navigation(
    score: stream.Score,
    plan: NavigationPlan | None = None,
) -> tuple[stream.Score, list[str]]:
    if plan is None:
        plans = [extract_navigation(part) for part in score.parts]
        plan = max(
            plans,
            key=lambda item: (len(item.commands), len(item.coda_marks)),
            default=NavigationPlan(0),
        )
    order, warnings = resolve_playback_order(plan)
    has_navigation_marks = bool(
        list(score.recurse().getElementsByClass(repeat.RepeatExpression))
    )
    if order == list(range(plan.measure_count)):
        if has_navigation_marks:
            return expand_score(score, order), warnings
        return deepcopy(score), warnings
    warnings.append(
        f"연주 순서 기호를 적용해 {plan.measure_count}마디를 "
        f"{len(order)}마디의 재생 순서로 전개했습니다."
    )
    return expand_score(score, order), warnings


def prepare_playback_score(
    score: stream.Score,
    navigation_plan: NavigationPlan | None = None,
) -> tuple[stream.Score, list[str]]:
    playback, warnings = expand_navigation(score, navigation_plan)
    chord_symbols = list(
        playback.recurse().getElementsByClass(harmony.ChordSymbol)
    )
    for symbol in chord_symbols:
        if symbol.activeSite is not None:
            symbol.activeSite.remove(symbol)
    if chord_symbols:
        warnings.append(
            f"코드 이름 {len(chord_symbols)}개를 MIDI 연주에서 제외했습니다."
        )
    return playback, warnings
