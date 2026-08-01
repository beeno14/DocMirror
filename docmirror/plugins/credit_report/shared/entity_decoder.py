# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stateful, source-conserving entity decoding for native credit reports.

The decoder is deliberately post-seal.  It copies the small amount of source
geometry and content needed for scoring and never mutates ``ParseResult``.
Physical page boundaries are observations, not entity boundaries: an open
entity remains active while compatible content is consumed on later pages.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Literal

ReportFamily = Literal["enterprise", "personal_brief", "personal_detail"]
UnitKind = Literal["table", "ledger", "text", "heading"]
EntityKind = Literal["table", "text", "mixed"]
TransitionAction = Literal[
    "same_table",
    "different_table",
    "table_to_text_related",
    "table_to_text_unrelated",
    "text_to_table_related",
    "text_to_table_unrelated",
    "same_text_section",
    "different_text_section",
    "new_section",
]

_ACTIONS: tuple[TransitionAction, ...] = (
    "same_table",
    "different_table",
    "table_to_text_related",
    "table_to_text_unrelated",
    "text_to_table_related",
    "text_to_table_unrelated",
    "same_text_section",
    "different_text_section",
    "new_section",
)
_CONTINUATION_ACTIONS = frozenset(
    {
        "same_table",
        "table_to_text_related",
        "text_to_table_related",
        "same_text_section",
    }
)
_SECTION_MARKERS = frozenset(
    {
        "报告信息",
        "说明",
        "身份标识",
        "信息概要",
        "基本信息",
        "信贷记录",
        "信贷记录明细",
        "相关还款责任信息",
        "非信贷交易记录",
        "公共记录",
        "公共记录明细",
        "查询记录",
        "机构查询记录明细",
        "个人查询记录明细",
        "信用记录补充信息",
        "附件",
    }
)
_TABLE_HEADER_MARKERS = frozenset(
    {
        "账户编号",
        "账户标识",
        "授信机构",
        "管理机构",
        "业务种类",
        "开立日期",
        "开户日期",
        "到期日",
        "币种",
        "金额",
        "余额",
        "信息报告日期",
        "查询日期",
        "查询机构",
        "查询原因",
        "五级分类",
        "合计",
    }
)
_TERMINAL_PUNCTUATION = frozenset("。！？!?；;")
_PAGE_NUMBER_RE = re.compile(
    r"^(?:第\s*\d+\s*页\s*[,，/]\s*共\s*\d+\s*页|"
    r"第\s*\d+\s*页|"
    r"page\s*\d+\s*(?:of|/)\s*\d+)$",
    re.IGNORECASE,
)
_LEDGER_ROW_RE = re.compile(r"^\s*\d{1,4}\s*(?:(?:19|20)\d{2}年|(?:19|20)\d{2}[-/.])")
_NUMBERED_TEXT_RE = re.compile(r"^\s*(\d{1,4})[.、]?\s*")
_DATE_RE = re.compile(r"(?:19|20)\d{2}(?:年|[-/.])\s*\d{1,2}(?:月|[-/.])\s*\d{1,2}")


@dataclass(frozen=True)
class CreditReportUnit:
    """One immutable text/table unit in physical top-to-bottom order."""

    unit_id: str
    page: int
    order: int
    source_index: int
    kind: UnitKind
    text: str
    bbox: tuple[float, float, float, float] | None
    page_width: float
    page_height: float
    table_id: str = ""
    rows: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True)
class TransitionHypothesis:
    """One possible action at a content boundary."""

    action: TransitionAction
    score: float
    signals: tuple[str, ...] = ()


@dataclass(frozen=True)
class EntityTransitionDecision:
    """The selected action and all alternatives for one adjacent unit pair."""

    left_unit_id: str
    right_unit_id: str
    from_page: int
    to_page: int
    hypotheses: tuple[TransitionHypothesis, ...]
    selected: TransitionAction
    confidence: float

    @property
    def continues_entity(self) -> bool:
        return self.selected in _CONTINUATION_ACTIONS

    @property
    def crosses_page(self) -> bool:
        return self.from_page != self.to_page


@dataclass(frozen=True)
class CreditReportEntity:
    """One decoded logical entity spanning one or more physical units/pages."""

    entity_id: str
    kind: EntityKind
    unit_ids: tuple[str, ...]
    pages: tuple[int, ...]
    confidence: float
    provisional_header_only: bool = False


@dataclass(frozen=True)
class CreditReportEntityContext:
    """Read-only decoder result and provenance indexes."""

    report_family: ReportFamily
    units: tuple[CreditReportUnit, ...]
    furniture_unit_ids: tuple[str, ...]
    entities: tuple[CreditReportEntity, ...]
    decisions: tuple[EntityTransitionDecision, ...]
    unassigned_unit_ids: tuple[str, ...]

    @property
    def page_boundary_decisions(self) -> tuple[EntityTransitionDecision, ...]:
        return tuple(decision for decision in self.decisions if decision.crosses_page)

    @property
    def content_conserved(self) -> bool:
        return not self.unassigned_unit_ids and len(self.unit_ids) == len(set(self.unit_ids))

    @property
    def unit_ids(self) -> tuple[str, ...]:
        return tuple(unit.unit_id for entity in self.entities for unit in self.units_for_entity(entity.entity_id))

    def units_for_entity(self, entity_id: str) -> tuple[CreditReportUnit, ...]:
        entity = next((item for item in self.entities if item.entity_id == entity_id), None)
        if entity is None:
            return ()
        by_id = {unit.unit_id: unit for unit in self.units}
        return tuple(by_id[unit_id] for unit_id in entity.unit_ids if unit_id in by_id)

    def entity_for_unit(self, unit_id: str) -> CreditReportEntity | None:
        return next((entity for entity in self.entities if unit_id in entity.unit_ids), None)

    def table_unit_id(self, table_id: str) -> str:
        unit = next((item for item in self.units if item.table_id == table_id), None)
        return unit.unit_id if unit is not None else ""

    def same_table_entity(self, left_table_id: str, right_table_id: str) -> bool:
        left_id = self.table_unit_id(left_table_id)
        right_id = self.table_unit_id(right_table_id)
        if not left_id or not right_id:
            return False
        left_entity = self.entity_for_unit(left_id)
        right_entity = self.entity_for_unit(right_id)
        return bool(
            left_entity is not None
            and right_entity is not None
            and left_entity.entity_id == right_entity.entity_id
            and left_entity.kind in {"table", "mixed"}
        )

    def decision_between(self, left_unit_id: str, right_unit_id: str) -> EntityTransitionDecision | None:
        return next(
            (
                decision
                for decision in self.decisions
                if decision.left_unit_id == left_unit_id and decision.right_unit_id == right_unit_id
            ),
            None,
        )

    def ordered_page_flow(self) -> tuple[tuple[int, str, Any], ...]:
        """Return source content in entity-level y order without rewriting text."""
        flow: list[tuple[int, str, Any]] = []
        for unit in self.units:
            if unit.kind == "table":
                flow.append((unit.page, "table", (unit.table_id, [list(row) for row in unit.rows])))
            else:
                flow.append((unit.page, "text", unit.text))
        return tuple(flow)

    def ordered_text_blocks(self) -> tuple[tuple[int, str], ...]:
        """Order entities geometrically while retaining source text order inside them."""
        blocks: list[tuple[int, str]] = []
        for entity in self.entities:
            text_units = [
                unit
                for unit in self.units_for_entity(entity.entity_id)
                if unit.kind in {"text", "heading", "ledger"} and unit.text.strip()
            ]
            text_units.sort(key=lambda unit: (unit.page, unit.source_index))
            blocks.extend((unit.page, unit.text) for unit in text_units)
        return tuple(blocks)


@dataclass(frozen=True)
class _BeamState:
    closed: tuple[tuple[str, ...], ...]
    open_unit_ids: tuple[str, ...]
    decisions: tuple[EntityTransitionDecision, ...]
    log_score: float


TransitionScorer = Callable[
    [tuple[CreditReportUnit, ...], CreditReportUnit, CreditReportUnit | None],
    tuple[TransitionHypothesis, ...],
]


def _compact(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or ""))


def _finite(value: Any) -> float:
    try:
        result = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def _bbox(value: Any) -> tuple[float, float, float, float] | None:
    raw = getattr(value, "bbox", None)
    if not isinstance(raw, (list, tuple)) or len(raw) < 4:
        return None
    x0, y0, x1, y1 = (_finite(item) for item in raw[:4])
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def _raw_rows(table: Any) -> tuple[tuple[str, ...], ...]:
    metadata = dict(getattr(table, "metadata", None) or {})
    raw_rows = metadata.get("raw_rows")
    if isinstance(raw_rows, list) and raw_rows:
        return tuple(
            tuple(str(value or "").replace("\n", "").strip() for value in row)
            for row in raw_rows
            if isinstance(row, list)
        )
    rows: list[tuple[str, ...]] = []
    headers = tuple(str(value or "").replace("\n", "").strip() for value in (getattr(table, "headers", None) or []))
    if headers:
        rows.append(headers)
    for row in getattr(table, "rows", None) or []:
        rows.append(
            tuple(
                str(getattr(cell, "text", cell) or "").replace("\n", "").strip()
                for cell in (getattr(row, "cells", None) or [])
            )
        )
    return tuple(rows)


def _page_number(page: Any, fallback: int) -> int:
    return int(getattr(page, "source_page_number", 0) or getattr(page, "page_number", 0) or fallback)


def _vertical(value: Any, fallback: float) -> float:
    box = _bbox(value)
    return box[1] if box is not None else fallback


def _is_explicit_page_number(text: str) -> bool:
    return bool(_PAGE_NUMBER_RE.fullmatch(re.sub(r"\s+", "", text).lower()))


def _is_heading(text: str) -> bool:
    compact = _compact(text).strip(":：")
    return compact in _SECTION_MARKERS or any(
        compact.startswith(marker) and len(compact) <= len(marker) + 12 for marker in _SECTION_MARKERS
    )


def _is_ledger(text: str) -> bool:
    return bool(_LEDGER_ROW_RE.match(_compact(text)))


def _collect_units(parse_result: Any) -> tuple[tuple[CreditReportUnit, ...], tuple[str, ...]]:
    pages = list(getattr(parse_result, "pages", None) or [])
    candidates: list[CreditReportUnit] = []
    furniture_ids: set[str] = set()
    edge_occurrences: dict[str, list[str]] = {}

    for page_index, page in enumerate(pages, start=1):
        page_number = _page_number(page, page_index)
        width = _finite(getattr(page, "width", 0))
        height = _finite(getattr(page, "height", 0))
        positioned: list[tuple[float, int, int, CreditReportUnit]] = []
        for text_index, block in enumerate(getattr(page, "texts", None) or []):
            content = str(getattr(block, "content", "") or "").strip()
            if not content:
                continue
            box = _bbox(block)
            unit_id = f"text:p{page_number}:{text_index}"
            kind: UnitKind = "heading" if _is_heading(content) else "ledger" if _is_ledger(content) else "text"
            unit = CreditReportUnit(
                unit_id=unit_id,
                page=page_number,
                order=0,
                source_index=text_index,
                kind=kind,
                text=content,
                bbox=box,
                page_width=width,
                page_height=height,
            )
            positioned.append((_vertical(block, float(text_index)), 0, text_index, unit))
            compact = _compact(content)
            near_edge = bool(box is not None and height > 0 and (box[1] / height <= 0.08 or box[3] / height >= 0.90))
            if _is_explicit_page_number(content):
                furniture_ids.add(unit_id)
            elif near_edge and len(compact) <= 80:
                edge_occurrences.setdefault(compact, []).append(unit_id)

        tables = list(getattr(page, "tables", None) or [])
        for table_index, table in enumerate(tables):
            rows = _raw_rows(table)
            if not rows:
                continue
            table_id = str(getattr(table, "table_id", "") or f"p{page_number}:t{table_index}")
            unit = CreditReportUnit(
                unit_id=f"table:{table_id}",
                page=page_number,
                order=0,
                source_index=table_index,
                kind="table",
                text="\n".join(" | ".join(row) for row in rows),
                bbox=_bbox(table),
                page_width=width,
                page_height=height,
                table_id=table_id,
                rows=rows,
            )
            positioned.append((_vertical(table, 10000.0 + float(table_index)), 1, table_index, unit))

        for order, (_top, _kind_order, _source_index, unit) in enumerate(sorted(positioned)):
            candidates.append(
                CreditReportUnit(
                    **{
                        **unit.__dict__,
                        "order": order,
                    }
                )
            )

    recurrence_minimum = max(2, math.ceil(len(pages) * 0.5))
    for unit_ids in edge_occurrences.values():
        if len({unit_id.split(":", 2)[1] for unit_id in unit_ids}) >= recurrence_minimum:
            furniture_ids.update(unit_ids)

    active = tuple(unit for unit in candidates if unit.unit_id not in furniture_ids)
    return active, tuple(sorted(furniture_ids))


def _table_like(kind: UnitKind) -> bool:
    return kind in {"table", "ledger"}


def _text_like(kind: UnitKind) -> bool:
    return kind in {"text", "heading"}


def _entity_kind(units: Iterable[CreditReportUnit]) -> EntityKind:
    kinds = {unit.kind for unit in units}
    if kinds and all(_table_like(kind) for kind in kinds):
        return "table"
    if kinds and all(_text_like(kind) for kind in kinds):
        return "text"
    return "mixed"


def _row_is_header(row: Iterable[str]) -> bool:
    compact = _compact("".join(row))
    return bool(compact) and sum(marker in compact for marker in _TABLE_HEADER_MARKERS) >= 2


def _table_rows(units: Iterable[CreditReportUnit]) -> tuple[tuple[str, ...], ...]:
    return tuple(row for unit in units if unit.kind == "table" for row in unit.rows)


def _header_tokens(units: Iterable[CreditReportUnit]) -> frozenset[str]:
    tokens: set[str] = set()
    for row in _table_rows(units):
        if not _row_is_header(row):
            continue
        compact = _compact("".join(row))
        tokens.update(marker for marker in _TABLE_HEADER_MARKERS if marker in compact)
    return frozenset(tokens)


def _data_rows(units: Iterable[CreditReportUnit]) -> tuple[tuple[str, ...], ...]:
    return tuple(row for row in _table_rows(units) if any(_compact(value) for value in row) and not _row_is_header(row))


def _median_columns(units: Iterable[CreditReportUnit]) -> int:
    counts = sorted(len(row) for row in _table_rows(units) if any(_compact(value) for value in row))
    return counts[len(counts) // 2] if counts else 0


def _horizontal_overlap(left: CreditReportUnit, right: CreditReportUnit) -> float:
    if left.bbox is None or right.bbox is None:
        return 0.0
    overlap = max(0.0, min(left.bbox[2], right.bbox[2]) - max(left.bbox[0], right.bbox[0]))
    smaller = min(left.bbox[2] - left.bbox[0], right.bbox[2] - right.bbox[0])
    return overlap / smaller if smaller > 0 else 0.0


def _at_bottom(unit: CreditReportUnit) -> bool:
    return bool(unit.bbox is not None and unit.page_height > 0 and unit.bbox[3] / unit.page_height >= 0.72)


def _at_top(unit: CreditReportUnit) -> bool:
    return bool(unit.bbox is not None and unit.page_height > 0 and unit.bbox[1] / unit.page_height <= 0.30)


def _numbered_sequence(text: str) -> int | None:
    match = _NUMBERED_TEXT_RE.match(str(text or ""))
    return int(match.group(1)) if match else None


def _score_transition(
    open_units: tuple[CreditReportUnit, ...],
    candidate: CreditReportUnit,
    lookahead: CreditReportUnit | None,
    *,
    report_family: ReportFamily,
) -> tuple[TransitionHypothesis, ...]:
    tail = open_units[-1]
    values: dict[TransitionAction, tuple[float, list[str]]] = {
        action: (0.001, ["incompatible_content_types"]) for action in _ACTIONS
    }

    def compatible(action: TransitionAction, base: float) -> tuple[float, list[str]]:
        values[action] = (base, [])
        return values[action]

    if _table_like(tail.kind) and _table_like(candidate.kind):
        same_score, same_signals = compatible("same_table", 0.24)
        different_score, different_signals = compatible("different_table", 0.20)
        section_score, section_signals = compatible("new_section", 0.03)
        cross_page = candidate.page != tail.page
        if cross_page:
            same_score += 0.10
            same_signals.append("adjacent_page_candidate")
            if _at_bottom(tail):
                same_score += 0.16
                same_signals.append("source_reaches_page_bottom")
            else:
                different_score += 0.24
                different_signals.append("source_ends_before_page_bottom")
            if _at_top(candidate):
                same_score += 0.14
                same_signals.append("candidate_starts_near_page_top")
        else:
            different_score += 0.08
            different_signals.append("same_page_table_boundary")

        source_columns = _median_columns(open_units)
        candidate_columns = _median_columns((candidate,))
        if source_columns and candidate_columns:
            if source_columns == candidate_columns:
                same_score += 0.32
                same_signals.append("column_count_exact")
            elif abs(source_columns - candidate_columns) == 1:
                same_score += 0.16
                same_signals.append("column_count_near")
            else:
                ratio = min(source_columns, candidate_columns) / max(source_columns, candidate_columns)
                if ratio >= 0.70:
                    same_score += 0.07
                    same_signals.append("column_count_compatible")
                else:
                    different_score += 0.36
                    different_signals.append("column_count_diverges")

        source_headers = _header_tokens(open_units)
        candidate_headers = _header_tokens((candidate,))
        source_data = _data_rows(open_units)
        candidate_data = _data_rows((candidate,))
        if source_headers and candidate_headers:
            overlap = len(source_headers & candidate_headers) / max(len(source_headers), len(candidate_headers), 1)
            if overlap >= 0.60:
                same_score += 0.27
                same_signals.append("header_schema_repeats")
            elif overlap < 0.20:
                different_score += 0.40
                different_signals.append("new_header_schema")
        elif source_headers and candidate_data:
            same_score += 0.34
            same_signals.append("header_to_headerless_body")
        elif candidate_headers and source_data:
            if cross_page:
                same_score += 0.16
                same_signals.append("repeated_header_after_page_break")
            else:
                different_score += 0.18
                different_signals.append("new_same_page_header")
        if source_headers and not source_data and candidate_data:
            same_score += 0.30
            same_signals.append("provisional_header_receives_body")
        if candidate.kind == "ledger" and tail.kind == "ledger":
            left_sequence = _numbered_sequence(tail.text)
            right_sequence = _numbered_sequence(candidate.text)
            if left_sequence is not None and right_sequence == left_sequence + 1:
                same_score += 0.32
                same_signals.append("ledger_sequence_continues")
        overlap = _horizontal_overlap(tail, candidate)
        if overlap >= 0.75:
            same_score += 0.12
            same_signals.append("horizontal_geometry_aligned")
        elif tail.bbox is not None and candidate.bbox is not None and overlap < 0.35:
            different_score += 0.12
            different_signals.append("horizontal_geometry_diverges")
        if (
            lookahead is not None
            and lookahead.page > candidate.page
            and _table_like(lookahead.kind)
            and _at_bottom(candidate)
        ):
            same_score += 0.12
            same_signals.append("lookahead_supports_open_table")
        if candidate.kind == "heading":
            section_score += 0.90
            section_signals.append("candidate_is_section_heading")
        values["same_table"] = (same_score, same_signals)
        values["different_table"] = (different_score, different_signals)
        values["new_section"] = (section_score, section_signals)

    elif _text_like(tail.kind) and _text_like(candidate.kind):
        same_score, same_signals = compatible("same_text_section", 0.25)
        different_score, different_signals = compatible("different_text_section", 0.20)
        section_score, section_signals = compatible("new_section", 0.03)
        left_text = tail.text.strip()
        right_text = candidate.text.strip()
        cross_page = candidate.page != tail.page
        if tail.kind == "heading" and candidate.kind != "heading":
            same_score += 0.30
            same_signals.append("heading_introduces_body")
        if candidate.kind == "heading":
            section_score += 0.95
            different_score += 0.30
            section_signals.append("candidate_is_section_heading")
        if left_text and left_text[-1] not in _TERMINAL_PUNCTUATION:
            same_score += 0.22
            same_signals.append("source_text_not_terminal")
        else:
            different_score += 0.12
            different_signals.append("source_text_terminal")
        if cross_page:
            if _at_bottom(tail):
                same_score += 0.13
                same_signals.append("source_reaches_page_bottom")
            if _at_top(candidate):
                same_score += 0.13
                same_signals.append("candidate_starts_near_page_top")
        left_sequence = _numbered_sequence(left_text)
        right_sequence = _numbered_sequence(right_text)
        if left_sequence is not None and right_sequence == left_sequence + 1:
            same_score += 0.28
            same_signals.append("numbered_text_sequence_continues")
        if _DATE_RE.search(left_text) and _DATE_RE.search(right_text) and candidate.kind != "heading":
            same_score += 0.08
            same_signals.append("record_family_date_pattern")
        if _horizontal_overlap(tail, candidate) >= 0.70:
            same_score += 0.10
            same_signals.append("text_columns_aligned")
        if (
            lookahead is not None
            and lookahead.page > candidate.page
            and _text_like(lookahead.kind)
            and _at_bottom(candidate)
            and candidate.kind != "heading"
        ):
            same_score += 0.10
            same_signals.append("lookahead_supports_open_text")
        values["same_text_section"] = (same_score, same_signals)
        values["different_text_section"] = (different_score, different_signals)
        values["new_section"] = (section_score, section_signals)

    elif _table_like(tail.kind) and _text_like(candidate.kind):
        related_score, related_signals = compatible("table_to_text_related", 0.23)
        unrelated_score, unrelated_signals = compatible("table_to_text_unrelated", 0.24)
        section_score, section_signals = compatible("new_section", 0.03)
        if candidate.kind == "heading":
            section_score += 0.88
            unrelated_score += 0.25
            section_signals.append("candidate_is_section_heading")
        elif candidate.page == tail.page and tail.bbox is not None and candidate.bbox is not None:
            gap = candidate.bbox[1] - tail.bbox[3]
            if -2 <= gap <= max(36.0, tail.page_height * 0.06):
                related_score += 0.25
                related_signals.append("text_follows_table_locally")
            else:
                unrelated_score += 0.13
                unrelated_signals.append("large_table_text_gap")
        elif candidate.page != tail.page and _at_bottom(tail) and _at_top(candidate):
            related_score += 0.12
            related_signals.append("cross_page_table_text_alignment")
        values["table_to_text_related"] = (related_score, related_signals)
        values["table_to_text_unrelated"] = (unrelated_score, unrelated_signals)
        values["new_section"] = (section_score, section_signals)

    elif _text_like(tail.kind) and _table_like(candidate.kind):
        related_score, related_signals = compatible("text_to_table_related", 0.25)
        unrelated_score, unrelated_signals = compatible("text_to_table_unrelated", 0.22)
        section_score, section_signals = compatible("new_section", 0.03)
        if tail.kind == "heading":
            related_score += 0.42
            related_signals.append("heading_introduces_table")
        elif candidate.page == tail.page and tail.bbox is not None and candidate.bbox is not None:
            gap = candidate.bbox[1] - tail.bbox[3]
            if -2 <= gap <= max(42.0, tail.page_height * 0.07):
                related_score += 0.22
                related_signals.append("table_follows_text_locally")
            else:
                unrelated_score += 0.12
                unrelated_signals.append("large_text_table_gap")
        elif candidate.page != tail.page and _at_bottom(tail) and _at_top(candidate):
            related_score += 0.12
            related_signals.append("cross_page_text_table_alignment")
        if _header_tokens((candidate,)):
            related_score += 0.08
            related_signals.append("candidate_has_table_header")
        values["text_to_table_related"] = (related_score, related_signals)
        values["text_to_table_unrelated"] = (unrelated_score, unrelated_signals)
        values["new_section"] = (section_score, section_signals)

    # Personal inquiry rows are visually table-like but may be emitted as text.
    # Their ordering evidence is stronger than the parser's modality label.
    if report_family == "personal_brief" and _is_ledger(candidate.text):
        action: TransitionAction = "same_table" if _table_like(tail.kind) else "text_to_table_related"
        score, signals = values[action]
        values[action] = (score + 0.18, [*signals, "personal_borderless_ledger"])

    total = sum(max(0.0, score) for score, _signals in values.values()) or 1.0
    hypotheses = tuple(
        sorted(
            (
                TransitionHypothesis(
                    action=action,
                    score=round(max(0.0, score) / total, 6),
                    signals=tuple(signals),
                )
                for action, (score, signals) in values.items()
            ),
            key=lambda item: (-item.score, item.action),
        )
    )
    return hypotheses


def _decode_units(
    units: tuple[CreditReportUnit, ...],
    *,
    report_family: ReportFamily,
    beam_width: int,
    transition_scorer: TransitionScorer | None = None,
) -> tuple[tuple[tuple[str, ...], ...], tuple[EntityTransitionDecision, ...]]:
    if not units:
        return (), ()
    by_id = {unit.unit_id: unit for unit in units}
    beam = [
        _BeamState(
            closed=(),
            open_unit_ids=(units[0].unit_id,),
            decisions=(),
            log_score=0.0,
        )
    ]
    for index, candidate in enumerate(units[1:], start=1):
        lookahead = units[index + 1] if index + 1 < len(units) else None
        expanded: list[_BeamState] = []
        for state in beam:
            open_units = tuple(by_id[unit_id] for unit_id in state.open_unit_ids)
            hypotheses = (
                transition_scorer(open_units, candidate, lookahead)
                if transition_scorer is not None
                else _score_transition(
                    open_units,
                    candidate,
                    lookahead,
                    report_family=report_family,
                )
            )
            compatible = [item for item in hypotheses if "incompatible_content_types" not in item.signals]
            for hypothesis in compatible[:3]:
                decision = EntityTransitionDecision(
                    left_unit_id=open_units[-1].unit_id,
                    right_unit_id=candidate.unit_id,
                    from_page=open_units[-1].page,
                    to_page=candidate.page,
                    hypotheses=hypotheses,
                    selected=hypothesis.action,
                    confidence=hypothesis.score,
                )
                continuation = hypothesis.action in _CONTINUATION_ACTIONS
                expanded.append(
                    _BeamState(
                        closed=state.closed if continuation else (*state.closed, state.open_unit_ids),
                        open_unit_ids=(
                            (*state.open_unit_ids, candidate.unit_id) if continuation else (candidate.unit_id,)
                        ),
                        decisions=(*state.decisions, decision),
                        log_score=state.log_score + math.log(max(hypothesis.score, 1e-9)),
                    )
                )
        deduplicated: dict[tuple[tuple[str, ...], int], _BeamState] = {}
        for state in expanded:
            key = (state.open_unit_ids[-3:], len(state.closed))
            incumbent = deduplicated.get(key)
            if incumbent is None or state.log_score > incumbent.log_score:
                deduplicated[key] = state
        beam = sorted(deduplicated.values(), key=lambda state: state.log_score, reverse=True)[: max(1, beam_width)]
    best = max(beam, key=lambda state: state.log_score)
    return (*best.closed, best.open_unit_ids), best.decisions


def score_credit_report_transition(
    open_units: tuple[CreditReportUnit, ...],
    candidate: CreditReportUnit,
    *,
    report_family: ReportFamily,
    lookahead: CreditReportUnit | None = None,
) -> EntityTransitionDecision:
    """Score one candidate against an accumulated open entity."""
    if not open_units:
        raise ValueError("open_units must contain at least one unit")
    hypotheses = _score_transition(
        open_units,
        candidate,
        lookahead,
        report_family=report_family,
    )
    selected = hypotheses[0]
    tail = open_units[-1]
    return EntityTransitionDecision(
        left_unit_id=tail.unit_id,
        right_unit_id=candidate.unit_id,
        from_page=tail.page,
        to_page=candidate.page,
        hypotheses=hypotheses,
        selected=selected.action,
        confidence=selected.score,
    )


def decode_credit_report_entities(
    parse_result: Any,
    *,
    report_family: ReportFamily,
    beam_width: int = 5,
) -> CreditReportEntityContext:
    """Decode native credit-report entities without modifying source objects."""
    if report_family not in {"enterprise", "personal_brief"}:
        raise ValueError(f"unsupported credit-report entity family: {report_family}")
    units, furniture_ids = _collect_units(parse_result)
    return decode_credit_report_units(
        units,
        report_family=report_family,
        furniture_unit_ids=furniture_ids,
        beam_width=beam_width,
    )


def decode_credit_report_units(
    units: Iterable[CreditReportUnit],
    *,
    report_family: ReportFamily,
    furniture_unit_ids: Iterable[str] = (),
    beam_width: int = 5,
    transition_scorer: TransitionScorer | None = None,
    entity_prefix: str | None = None,
) -> CreditReportEntityContext:
    """Decode caller-owned units with an optional variant-local scorer.

    Existing native report families continue to enter through
    :func:`decode_credit_report_entities`; this lower-level seam lets a variant
    provide logical-page units and business contracts without changing the
    shared default collector or score tables.
    """
    if report_family not in {"enterprise", "personal_brief", "personal_detail"}:
        raise ValueError(f"unsupported credit-report entity family: {report_family}")
    ordered_units = tuple(units)
    grouped_ids, decisions = _decode_units(
        ordered_units,
        report_family=report_family,
        beam_width=beam_width,
        transition_scorer=transition_scorer,
    )
    by_id = {unit.unit_id: unit for unit in ordered_units}
    if len(by_id) != len(ordered_units):
        raise ValueError("credit-report unit ids must be unique")
    decision_by_pair = {(decision.left_unit_id, decision.right_unit_id): decision for decision in decisions}
    entities: list[CreditReportEntity] = []
    assigned: list[str] = []
    prefix = entity_prefix or report_family
    for index, unit_ids in enumerate(grouped_ids, start=1):
        grouped_units = tuple(by_id[unit_id] for unit_id in unit_ids)
        internal_decisions = [
            decision_by_pair[(left, right)]
            for left, right in zip(unit_ids, unit_ids[1:])
            if (left, right) in decision_by_pair
        ]
        kind = _entity_kind(grouped_units)
        entities.append(
            CreditReportEntity(
                entity_id=f"{prefix}_entity:{index:05d}",
                kind=kind,
                unit_ids=unit_ids,
                pages=tuple(sorted({unit.page for unit in grouped_units})),
                confidence=round(
                    min((decision.confidence for decision in internal_decisions), default=1.0),
                    6,
                ),
                provisional_header_only=kind == "table"
                and bool(_header_tokens(grouped_units))
                and not bool(_data_rows(grouped_units)),
            )
        )
        assigned.extend(unit_ids)
    active_ids = [unit.unit_id for unit in ordered_units]
    unassigned = tuple(unit_id for unit_id in active_ids if assigned.count(unit_id) != 1)
    return CreditReportEntityContext(
        report_family=report_family,
        units=ordered_units,
        furniture_unit_ids=tuple(furniture_unit_ids),
        entities=tuple(entities),
        decisions=decisions,
        unassigned_unit_ids=unassigned,
    )


__all__ = [
    "CreditReportEntity",
    "CreditReportEntityContext",
    "CreditReportUnit",
    "EntityTransitionDecision",
    "TransitionAction",
    "TransitionHypothesis",
    "decode_credit_report_entities",
    "decode_credit_report_units",
    "score_credit_report_transition",
]
