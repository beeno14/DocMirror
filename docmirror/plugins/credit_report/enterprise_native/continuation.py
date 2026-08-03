# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scored page-boundary resolution for native enterprise credit reports.

The resolver works above immutable physical pages and tables.  It ranks
same-entity, new-entity, and new-section hypotheses from layout, schema, text,
and optional business-row evidence, then builds non-destructive logical
entities across accepted boundaries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from math import isfinite
from typing import Any, Callable, Iterable, Literal

from docmirror.plugins.credit_report.shared.entity_decoder import (
    CreditReportEntityContext,
    CreditReportUnit,
    EntityTransitionDecision,
    TransitionAction,
    decode_credit_report_entities,
    score_credit_report_transition,
)

Row = list[str]
RowPredicate = Callable[[Row], bool]


@dataclass(frozen=True)
class TableFragment:
    """One sealed physical table in document reading order."""

    index: int
    page: int
    table_id: str
    rows: tuple[tuple[str, ...], ...]
    bbox: tuple[float, float, float, float] | None = None
    page_width: float = 0.0
    page_height: float = 0.0
    first_on_page: bool = False
    last_on_page: bool = False

    def mutable_rows(self) -> list[Row]:
        return [list(row) for row in self.rows]


@dataclass(frozen=True)
class ContinuationContract:
    """A declarative, family-specific permission to follow one table."""

    name: str
    # An empty set deliberately means "semantic shape only".  Native PBOC
    # PDFs sometimes materialize merged/spacer cells differently when the
    # same logical row lands at another page position, so an exact physical
    # width must never be the sole continuation key.
    expected_columns: frozenset[int]
    row_predicate: RowPredicate
    max_page_gap: int = 1
    forbidden_markers: tuple[str, ...] = ()


@dataclass(frozen=True)
class ContinuationMatch:
    """An accepted continuation with auditable source coordinates."""

    contract: str
    fragment: TableFragment
    row_index: int
    row: tuple[str, ...]


@dataclass(frozen=True)
class ContinuationRejection:
    """Why a possible continuation was not consumed."""

    contract: str
    source_table_id: str
    candidate_table_id: str
    reason: str


BoundaryKind = Literal[
    "same_table",
    "new_table",
    "same_body_text",
    "new_body_text",
    "table_related_text",
    "table_unrelated_text",
    "new_section",
]


@dataclass(frozen=True)
class BoundaryHypothesis:
    """One ranked interpretation of a physical page boundary."""

    kind: BoundaryKind
    score: float
    signals: tuple[str, ...] = ()


@dataclass(frozen=True)
class PageBoundaryDecision:
    """Scored decision between two physical fragments."""

    left_id: str
    right_id: str
    from_page: int
    to_page: int
    hypotheses: tuple[BoundaryHypothesis, ...]
    selected: BoundaryKind
    confidence: float
    accepted: bool
    context: str = ""


@dataclass(frozen=True)
class BoundaryUnit:
    """A table or text block participating in page-boundary scoring."""

    unit_id: str
    page: int
    kind: Literal["table", "text"]
    order: float
    text: str = ""
    table_id: str = ""
    bbox: tuple[float, float, float, float] | None = None
    page_width: float = 0.0
    page_height: float = 0.0


@dataclass(frozen=True)
class LogicalPageEntity:
    """A non-destructive logical entity assembled across physical pages."""

    entity_id: str
    kind: str
    unit_ids: tuple[str, ...]
    pages: tuple[int, ...]
    confidence: float
    boundary_decisions: tuple[PageBoundaryDecision, ...] = ()


def _raw_rows(table: Any) -> list[Row]:
    metadata = dict(getattr(table, "metadata", None) or {})
    raw_rows = metadata.get("raw_rows")
    if isinstance(raw_rows, list) and raw_rows:
        return [
            [str(value or "").replace("\n", "").strip() for value in row] for row in raw_rows if isinstance(row, list)
        ]
    rows: list[Row] = []
    headers = list(getattr(table, "headers", None) or [])
    if headers:
        rows.append([str(value or "").replace("\n", "").strip() for value in headers])
    for row in getattr(table, "rows", None) or []:
        cells = getattr(row, "cells", None) or []
        rows.append([str(getattr(cell, "text", cell) or "").replace("\n", "").strip() for cell in cells])
    return rows


class EnterpriseContinuationResolver:
    """Score page-boundary hypotheses and expose accepted logical continuations.

    Physical pages and tables stay immutable.  The resolver builds a parallel
    boundary graph, ranks competing interpretations, and exposes compatibility
    helpers for the enterprise business extractors.
    """

    def __init__(self, parse_result: Any, *, acceptance_threshold: float = 0.58):
        self.rejections: list[ContinuationRejection] = []
        self.acceptance_threshold = float(acceptance_threshold)
        self._decision_cache: dict[tuple[int, int, str, int], PageBoundaryDecision] = {}
        self._document_boundary_decisions: tuple[PageBoundaryDecision, ...] = ()
        self._logical_entities: tuple[LogicalPageEntity, ...] = ()
        prebuilt = getattr(parse_result, "continuation_fragments", None)
        if prebuilt is not None:
            self.fragments = tuple(prebuilt)
        else:
            self.fragments = self._build_table_fragments(parse_result)
        source_result = getattr(parse_result, "parse_result", parse_result)
        prebuilt_entity_context = getattr(parse_result, "entity_context", None)
        self.entity_context: CreditReportEntityContext = (
            prebuilt_entity_context
            if isinstance(prebuilt_entity_context, CreditReportEntityContext)
            else decode_credit_report_entities(
                source_result,
                report_family="enterprise",
            )
        )
        self._units = self._build_boundary_units(source_result)
        self._score_document_boundaries()

    @staticmethod
    def _build_table_fragments(parse_result: Any) -> tuple[TableFragment, ...]:
        fragments: list[TableFragment] = []
        pages = list(getattr(parse_result, "pages", None) or [])
        for page in pages:
            page_number = int(getattr(page, "source_page_number", 0) or getattr(page, "page_number", 0) or 0)
            tables = list(getattr(page, "tables", None) or [])
            for table_index, table in enumerate(tables):
                rows = _raw_rows(table)
                if not rows:
                    continue
                bbox = _valid_bbox(getattr(table, "bbox", None))
                fragments.append(
                    TableFragment(
                        index=len(fragments),
                        page=page_number,
                        table_id=str(getattr(table, "table_id", "") or ""),
                        rows=tuple(tuple(value for value in row) for row in rows),
                        bbox=bbox,
                        page_width=_finite_float(getattr(page, "width", 0)),
                        page_height=_finite_float(getattr(page, "height", 0)),
                        first_on_page=table_index == 0,
                        last_on_page=table_index == len(tables) - 1,
                    )
                )
        return tuple(fragments)

    def _build_boundary_units(self, parse_result: Any) -> tuple[BoundaryUnit, ...]:
        fragment_by_table_id = {fragment.table_id: fragment for fragment in self.fragments}
        units: list[BoundaryUnit] = []
        for page_index, page in enumerate(getattr(parse_result, "pages", None) or [], start=1):
            page_number = int(getattr(page, "source_page_number", 0) or getattr(page, "page_number", 0) or page_index)
            width = _finite_float(getattr(page, "width", 0))
            height = _finite_float(getattr(page, "height", 0))
            positioned: list[tuple[float, int, int, BoundaryUnit]] = []
            for text_index, block in enumerate(getattr(page, "texts", None) or []):
                content = str(getattr(block, "content", "") or "").strip()
                if not content or _is_page_noise(content):
                    continue
                bbox = _valid_bbox(getattr(block, "bbox", None))
                positioned.append(
                    (
                        _vertical_position(bbox, float(text_index)),
                        0,
                        text_index,
                        BoundaryUnit(
                            unit_id=f"text:p{page_number}:{text_index}",
                            page=page_number,
                            kind="text",
                            order=0.0,
                            text=content,
                            bbox=bbox,
                            page_width=width,
                            page_height=height,
                        ),
                    )
                )
            for table_index, table in enumerate(getattr(page, "tables", None) or []):
                table_id = str(getattr(table, "table_id", "") or "")
                fragment = fragment_by_table_id.get(table_id)
                if fragment is None:
                    continue
                bbox = fragment.bbox or _valid_bbox(getattr(table, "bbox", None))
                positioned.append(
                    (
                        _vertical_position(bbox, 10000.0 + float(table_index)),
                        1,
                        table_index,
                        BoundaryUnit(
                            unit_id=f"table:{table_id or fragment.index}",
                            page=page_number,
                            kind="table",
                            order=0.0,
                            text="\n".join(" | ".join(row) for row in fragment.rows),
                            table_id=table_id,
                            bbox=bbox,
                            page_width=width,
                            page_height=height,
                        ),
                    )
                )
            for order, (_position, _kind_order, _index, unit) in enumerate(sorted(positioned)):
                units.append(
                    BoundaryUnit(
                        **{
                            **unit.__dict__,
                            "order": float(order),
                        }
                    )
                )
        return tuple(units)

    @property
    def boundary_decisions(self) -> tuple[PageBoundaryDecision, ...]:
        return self._document_boundary_decisions

    @property
    def logical_entities(self) -> tuple[LogicalPageEntity, ...]:
        return self._logical_entities

    def _score_document_boundaries(self) -> None:
        decisions = tuple(_shared_page_decision(decision) for decision in self.entity_context.page_boundary_decisions)
        decision_by_pair = {(decision.left_id, decision.right_id): decision for decision in decisions}
        self._document_boundary_decisions = decisions
        self._logical_entities = tuple(
            LogicalPageEntity(
                entity_id=entity.entity_id,
                kind=entity.kind,
                unit_ids=entity.unit_ids,
                pages=entity.pages,
                confidence=entity.confidence,
                boundary_decisions=tuple(
                    decision_by_pair[(left, right)]
                    for left, right in zip(entity.unit_ids, entity.unit_ids[1:])
                    if (left, right) in decision_by_pair
                ),
            )
            for entity in self.entity_context.entities
        )

    def _fragment_for_id(self, table_id: str) -> TableFragment | None:
        return next((fragment for fragment in self.fragments if fragment.table_id == table_id), None)

    def decide_table_boundary(
        self,
        source: TableFragment,
        candidate: TableFragment,
        contract: ContinuationContract | None = None,
        *,
        candidate_row_index: int = 0,
        candidate_validator: RowPredicate | None = None,
        candidate_validator_scope: Literal["row", "fragment"] = "row",
        context: str = "",
    ) -> PageBoundaryDecision:
        """Rank same-table and split-table hypotheses for one boundary."""
        context_name = context or (contract.name if contract is not None else "generic_table")
        cache_key = (
            source.index,
            candidate.index,
            f"{context_name}:{candidate_validator_scope}",
            candidate_row_index,
        )
        cached = self._decision_cache.get(cache_key)
        if cached is not None:
            return cached
        left_id = self.entity_context.table_unit_id(source.table_id)
        right_id = self.entity_context.table_unit_id(candidate.table_id)
        shared = self.entity_context.decision_between(left_id, right_id)
        semantic_match = _candidate_semantic_match(
            candidate,
            contract=contract,
            candidate_row_index=candidate_row_index,
            candidate_validator=candidate_validator,
            candidate_validator_scope=candidate_validator_scope,
        )
        if shared is not None:
            decision = _shared_page_decision(shared, context=context_name)
            if semantic_match is True and candidate.index == source.index + 1:
                decision = _with_semantic_table_match(decision)
        elif self.entity_context.same_table_entity(source.table_id, candidate.table_id):
            decision = PageBoundaryDecision(
                left_id=left_id or f"table:{source.table_id or source.index}",
                right_id=right_id or f"table:{candidate.table_id or candidate.index}",
                from_page=source.page,
                to_page=candidate.page,
                hypotheses=(
                    BoundaryHypothesis(
                        kind="same_table",
                        score=0.75,
                        signals=("shared_open_entity_membership",),
                    ),
                    BoundaryHypothesis(kind="new_table", score=0.20),
                    BoundaryHypothesis(kind="new_section", score=0.05),
                ),
                selected="same_table",
                confidence=0.75,
                accepted=True,
                context=context_name,
            )
        else:
            decision = _shared_page_decision(
                score_credit_report_transition(
                    (_fragment_entity_unit(source),),
                    _fragment_entity_unit(candidate),
                    report_family="enterprise",
                ),
                context=context_name,
            )
            if semantic_match is True and candidate.index == source.index + 1:
                decision = _with_semantic_table_match(decision)
        if candidate.page - source.page not in {0, 1} or candidate.index != source.index + 1:
            semantic_match = False
        if semantic_match is False and decision.accepted:
            decision = PageBoundaryDecision(
                left_id=decision.left_id,
                right_id=decision.right_id,
                from_page=decision.from_page,
                to_page=decision.to_page,
                hypotheses=decision.hypotheses,
                selected="new_table",
                confidence=decision.confidence,
                accepted=False,
                context=decision.context,
            )
        self._decision_cache[cache_key] = decision
        return decision

    def table_continues(
        self,
        source: TableFragment,
        candidate: TableFragment,
        contract: ContinuationContract | None = None,
        *,
        candidate_row_index: int = 0,
        candidate_validator: RowPredicate | None = None,
        candidate_validator_scope: Literal["row", "fragment"] = "row",
        context: str = "",
    ) -> bool:
        """Return the scored same-table decision without mutating either table."""
        return self.decide_table_boundary(
            source,
            candidate,
            contract,
            candidate_row_index=candidate_row_index,
            candidate_validator=candidate_validator,
            candidate_validator_scope=candidate_validator_scope,
            context=context,
        ).accepted

    def following_fragments(
        self,
        source: TableFragment,
        *,
        candidate_validator: RowPredicate | None = None,
        context: str = "generic_table_chain",
    ) -> tuple[TableFragment, ...]:
        """Follow a same-table chain until a scored boundary selects a split."""
        accepted: list[TableFragment] = []
        previous = source
        candidate_index = source.index + 1
        while candidate_index < len(self.fragments):
            candidate = self.fragments[candidate_index]
            if candidate.page - previous.page not in {0, 1}:
                break
            if not self.table_continues(
                previous,
                candidate,
                candidate_validator=candidate_validator,
                candidate_validator_scope="fragment",
                context=context,
            ):
                break
            accepted.append(candidate)
            previous = candidate
            candidate_index += 1
        return tuple(accepted)

    def following_row(
        self,
        source: TableFragment,
        contract: ContinuationContract,
        *,
        candidate_row_index: int = 0,
    ) -> ContinuationMatch | None:
        """Return the next row when the scored same-table hypothesis wins."""
        candidate_index = source.index + 1
        if candidate_index >= len(self.fragments):
            return None
        candidate = self.fragments[candidate_index]
        page_gap = candidate.page - source.page
        if page_gap < 0 or page_gap > contract.max_page_gap:
            self._reject(source, candidate, contract, "page_gap")
            return None
        if candidate_row_index >= len(candidate.rows):
            self._reject(source, candidate, contract, "missing_candidate_row")
            return None
        row = list(candidate.rows[candidate_row_index])
        signature = "".join(row)
        decision = self.decide_table_boundary(
            source,
            candidate,
            contract,
            candidate_row_index=candidate_row_index,
        )
        if not decision.accepted:
            if contract.expected_columns and len(row) not in contract.expected_columns:
                reason = "column_shape"
            elif any(marker in signature for marker in contract.forbidden_markers):
                reason = "new_header"
            elif not contract.row_predicate(row):
                reason = "row_semantics"
            else:
                reason = "boundary_score"
            self._reject(source, candidate, contract, reason)
            return None
        return ContinuationMatch(
            contract=contract.name,
            fragment=candidate,
            row_index=candidate_row_index,
            row=tuple(row),
        )

    def _reject(
        self,
        source: TableFragment,
        candidate: TableFragment,
        contract: ContinuationContract,
        reason: str,
    ) -> None:
        self.rejections.append(
            ContinuationRejection(
                contract=contract.name,
                source_table_id=source.table_id,
                candidate_table_id=candidate.table_id,
                reason=reason,
            )
        )

    def audit_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "contract": item.contract,
                "source_table_id": item.source_table_id,
                "candidate_table_id": item.candidate_table_id,
                "reason": item.reason,
            }
            for item in self.rejections
        ]

    def decision_rows(self) -> list[dict[str, Any]]:
        """Return scored page-boundary evidence for diagnostics and experiments."""
        decisions = list(self._document_boundary_decisions)
        decisions.extend(decision for decision in self._decision_cache.values() if decision not in decisions)
        return [
            {
                "left_id": decision.left_id,
                "right_id": decision.right_id,
                "from_page": decision.from_page,
                "to_page": decision.to_page,
                "selected": decision.selected,
                "confidence": decision.confidence,
                "accepted": decision.accepted,
                "context": decision.context,
                "hypotheses": [
                    {
                        "kind": hypothesis.kind,
                        "score": hypothesis.score,
                        "signals": list(hypothesis.signals),
                    }
                    for hypothesis in decision.hypotheses
                ],
            }
            for decision in decisions
        ]


_HEADER_MARKERS = (
    "账户编号",
    "授信机构",
    "业务种类",
    "开立日期",
    "到期日",
    "信息报告日期",
    "余额",
    "五级分类",
    "币种",
    "金额",
    "正常类",
    "关注类",
    "不良类",
    "合计",
    "责任类型",
    "保证合同编号",
    "还款责任金额",
)
_SECTION_MARKERS = (
    "身份标识",
    "信息概要",
    "基本信息",
    "信贷记录明细",
    "公共记录明细",
    "信用记录补充信息",
    "附件",
    "报告说明",
    "相关还款责任",
)
_TERMINAL_PUNCTUATION = frozenset("。！？!?；;：:")


def _finite_float(value: Any) -> float:
    try:
        result = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return result if isfinite(result) else 0.0


def _valid_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    x0, y0, x1, y1 = (_finite_float(item) for item in value[:4])
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _vertical_position(
    bbox: tuple[float, float, float, float] | None,
    fallback: float,
) -> float:
    return bbox[1] if bbox is not None else fallback


def _compact_text(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or ""))


def _is_page_noise(text: str) -> bool:
    compact = _compact_text(text)
    return bool(
        re.fullmatch(r"第?\d+页(?:[/／]共?\d+页)?", compact)
        or re.fullmatch(r"NO\.?[0-9A-Z-]{8,}", compact, flags=re.IGNORECASE)
    )


def _score_number_like(value: Any) -> bool:
    compact = _compact_text(value).replace(",", "").replace("，", "")
    if compact in {"", "-", "--", "—"}:
        return False
    try:
        float(compact.rstrip("%"))
    except ValueError:
        return False
    return True


def _date_like_for_score(value: Any) -> bool:
    compact = _compact_text(value)
    return bool(re.fullmatch(r"(?:19|20)\d{2}(?:[-/.年]\d{1,2}){1,2}(?:日)?", compact))


def _id_like(value: Any) -> bool:
    compact = re.sub(r"[^0-9A-Z]", "", str(value or "").upper())
    return len(compact) >= 12 and any(char.isdigit() for char in compact)


def _cell_type(value: Any) -> str:
    compact = _compact_text(value)
    if not compact or compact in {"-", "--", "—"}:
        return "blank"
    if _date_like_for_score(compact):
        return "date"
    if _score_number_like(compact):
        return "number"
    if _id_like(compact):
        return "identifier"
    return "text"


def _row_is_header(row: Iterable[Any]) -> bool:
    values = [_compact_text(value) for value in row]
    signature = "".join(values)
    if not signature:
        return False
    marker_hits = sum(marker in signature for marker in _HEADER_MARKERS)
    if marker_hits >= 2:
        return True
    nonempty = [value for value in values if value]
    numeric = sum(_score_number_like(value) or _date_like_for_score(value) for value in nonempty)
    return marker_hits == 1 and len(nonempty) >= 2 and numeric == 0


def _row_is_section_title(row: Iterable[Any]) -> bool:
    values = [_compact_text(value) for value in row if _compact_text(value)]
    if not values:
        return False
    signature = "".join(values)
    return len(values) <= 2 and any(marker in signature for marker in _SECTION_MARKERS)


def _header_band(rows: tuple[tuple[str, ...], ...]) -> tuple[tuple[str, ...], ...]:
    band: list[tuple[str, ...]] = []
    for row in rows[:4]:
        if _row_is_header(row) or _row_is_section_title(row):
            band.append(row)
            continue
        break
    return tuple(band)


def _header_tokens(rows: tuple[tuple[str, ...], ...]) -> frozenset[str]:
    tokens: set[str] = set()
    for row in _header_band(rows):
        for value in row:
            compact = _compact_text(value)
            if compact:
                tokens.add(compact)
    return frozenset(tokens)


def _median_column_count(rows: tuple[tuple[str, ...], ...]) -> int:
    counts = sorted(len(row) for row in rows if any(_compact_text(value) for value in row))
    return counts[len(counts) // 2] if counts else 0


def _column_profile(rows: tuple[tuple[str, ...], ...]) -> tuple[str, ...]:
    data = [row for row in rows if not _row_is_header(row) and not _row_is_section_title(row)]
    selected = data[-4:] if data else list(rows[-2:])
    width = max((len(row) for row in selected), default=0)
    profile: list[str] = []
    for index in range(width):
        types = [_cell_type(row[index]) for row in selected if index < len(row) and _cell_type(row[index]) != "blank"]
        if not types:
            profile.append("blank")
            continue
        profile.append(max(set(types), key=lambda item: (types.count(item), item)))
    return tuple(profile)


def _profile_similarity(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    if not left or not right:
        return 0.0
    width = max(len(left), len(right))
    matches = 0.0
    for index in range(width):
        left_type = left[index] if index < len(left) else "missing"
        right_type = right[index] if index < len(right) else "missing"
        if left_type == right_type:
            matches += 1.0
        elif "blank" in {left_type, right_type}:
            matches += 0.5
    return matches / width


def _horizontal_overlap(
    left: tuple[float, float, float, float] | None,
    right: tuple[float, float, float, float] | None,
) -> float:
    if left is None or right is None:
        return 0.0
    overlap = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    smaller = min(left[2] - left[0], right[2] - right[0])
    return overlap / smaller if smaller > 0 else 0.0


def _at_page_bottom(fragment: TableFragment) -> bool:
    return bool(
        fragment.bbox is not None and fragment.page_height > 0 and fragment.bbox[3] / fragment.page_height >= 0.72
    )


def _at_page_top(fragment: TableFragment) -> bool:
    return bool(
        fragment.bbox is not None and fragment.page_height > 0 and fragment.bbox[1] / fragment.page_height <= 0.30
    )


def _ranked_hypotheses(
    values: dict[BoundaryKind, tuple[float, list[str]]],
) -> tuple[BoundaryHypothesis, ...]:
    total = sum(max(0.0, score) for score, _signals in values.values()) or 1.0
    hypotheses = [
        BoundaryHypothesis(
            kind=kind,
            score=round(max(0.0, score) / total, 4),
            signals=tuple(signals),
        )
        for kind, (score, signals) in values.items()
    ]
    return tuple(sorted(hypotheses, key=lambda item: (-item.score, item.kind)))


_SHARED_ACTION_KIND: dict[TransitionAction, BoundaryKind] = {
    "same_table": "same_table",
    "different_table": "new_table",
    "table_to_text_related": "table_related_text",
    "table_to_text_unrelated": "table_unrelated_text",
    "text_to_table_related": "table_related_text",
    "text_to_table_unrelated": "table_unrelated_text",
    "same_text_section": "same_body_text",
    "different_text_section": "new_body_text",
    "new_section": "new_section",
}


def _shared_page_decision(
    decision: EntityTransitionDecision,
    *,
    context: str = "",
) -> PageBoundaryDecision:
    compatible = [
        hypothesis for hypothesis in decision.hypotheses if "incompatible_content_types" not in hypothesis.signals
    ]
    hypotheses = tuple(
        BoundaryHypothesis(
            kind=_SHARED_ACTION_KIND[hypothesis.action],
            score=hypothesis.score,
            signals=hypothesis.signals,
        )
        for hypothesis in compatible
    )
    selected = _SHARED_ACTION_KIND[decision.selected]
    return PageBoundaryDecision(
        left_id=decision.left_unit_id,
        right_id=decision.right_unit_id,
        from_page=decision.from_page,
        to_page=decision.to_page,
        hypotheses=hypotheses,
        selected=selected,
        confidence=decision.confidence,
        accepted=decision.continues_entity,
        context=context or "shared_entity_decoder",
    )


def _candidate_semantic_match(
    candidate: TableFragment,
    *,
    contract: ContinuationContract | None,
    candidate_row_index: int,
    candidate_validator: RowPredicate | None,
    candidate_validator_scope: Literal["row", "fragment"],
) -> bool | None:
    validator = candidate_validator or (contract.row_predicate if contract is not None else None)
    candidate_row = list(candidate.rows[candidate_row_index]) if 0 <= candidate_row_index < len(candidate.rows) else []
    if contract is not None and candidate_row:
        if contract.expected_columns and len(candidate_row) not in contract.expected_columns:
            return False
        signature = "".join(candidate_row)
        if any(marker in signature for marker in contract.forbidden_markers):
            return False
    if validator is None:
        return None
    rows = (
        [list(row) for row in candidate.rows]
        if candidate_validator_scope == "fragment"
        else ([candidate_row] if candidate_row else [])
    )
    return any(validator(row) for row in rows)


def _with_semantic_table_match(decision: PageBoundaryDecision) -> PageBoundaryDecision:
    """Re-rank a shared transition when a business row contract also matches."""
    values: dict[BoundaryKind, tuple[float, list[str]]] = {}
    for hypothesis in decision.hypotheses:
        score = hypothesis.score + (0.45 if hypothesis.kind == "same_table" else 0.0)
        signals = list(hypothesis.signals)
        if hypothesis.kind == "same_table":
            signals.append("semantic_row_shape_matches")
        values[hypothesis.kind] = (score, signals)
    hypotheses = _ranked_hypotheses(values)
    selected = hypotheses[0]
    return PageBoundaryDecision(
        left_id=decision.left_id,
        right_id=decision.right_id,
        from_page=decision.from_page,
        to_page=decision.to_page,
        hypotheses=hypotheses,
        selected=selected.kind,
        confidence=selected.score,
        accepted=selected.kind == "same_table",
        context=decision.context,
    )


def _fragment_entity_unit(fragment: TableFragment) -> CreditReportUnit:
    """Adapt the public fragment API to the shared entity scorer."""
    return CreditReportUnit(
        unit_id=f"table:{fragment.table_id or fragment.index}",
        page=fragment.page,
        order=fragment.index,
        source_index=fragment.index,
        kind="table",
        text="\n".join(" | ".join(row) for row in fragment.rows),
        bbox=fragment.bbox,
        page_width=fragment.page_width,
        page_height=fragment.page_height,
        table_id=fragment.table_id,
        rows=fragment.rows,
    )


def _score_table_boundary(
    source: TableFragment,
    candidate: TableFragment,
    *,
    threshold: float,
    contract: ContinuationContract | None,
    candidate_row_index: int,
    candidate_validator: RowPredicate | None,
    candidate_validator_scope: Literal["row", "fragment"],
    context: str,
) -> PageBoundaryDecision:
    same_score = 0.18
    split_score = 0.18
    section_score = 0.02
    same_signals: list[str] = []
    split_signals: list[str] = []
    section_signals: list[str] = []
    page_gap = candidate.page - source.page

    if candidate.index == source.index + 1:
        same_score += 0.10
        same_signals.append("adjacent_physical_fragments")
    else:
        split_score += 0.35
        split_signals.append("intervening_table")
    if page_gap == 1:
        same_score += 0.08
        same_signals.append("adjacent_pages")
    elif page_gap == 0:
        same_score += 0.03
        same_signals.append("same_page_fragment")
    else:
        split_score += 0.75
        split_signals.append("non_adjacent_pages")

    if source.last_on_page:
        same_score += 0.08
        same_signals.append("source_is_last_table")
    if candidate.first_on_page:
        same_score += 0.08
        same_signals.append("candidate_is_first_table")
    if _at_page_bottom(source):
        same_score += 0.09
        same_signals.append("source_reaches_page_bottom")
    if _at_page_top(candidate):
        same_score += 0.09
        same_signals.append("candidate_starts_near_page_top")
    overlap = _horizontal_overlap(source.bbox, candidate.bbox)
    if overlap >= 0.80:
        same_score += 0.08
        same_signals.append("horizontal_geometry_aligned")
    elif source.bbox is not None and candidate.bbox is not None and overlap < 0.35:
        split_score += 0.12
        split_signals.append("horizontal_geometry_diverges")

    source_cols = _median_column_count(source.rows)
    candidate_cols = _median_column_count(candidate.rows)
    if source_cols and candidate_cols:
        difference = abs(source_cols - candidate_cols)
        if difference == 0:
            same_score += 0.20
            same_signals.append("column_count_exact")
        elif difference == 1:
            same_score += 0.10
            same_signals.append("column_count_near")
        else:
            ratio = min(source_cols, candidate_cols) / max(source_cols, candidate_cols)
            if ratio >= 0.70:
                same_score += 0.04
                same_signals.append("column_count_compatible")
            else:
                split_score += 0.28
                split_signals.append("column_count_diverges")

    source_headers = _header_tokens(source.rows)
    candidate_headers = _header_tokens(candidate.rows)
    candidate_data = [row for row in candidate.rows if not _row_is_header(row) and not _row_is_section_title(row)]
    if source_headers and candidate_headers:
        overlap_count = len(source_headers & candidate_headers)
        header_overlap = overlap_count / max(len(source_headers), len(candidate_headers), 1)
        if header_overlap >= 0.60 and candidate_data:
            same_score += 0.18
            same_signals.append("repeated_header_with_body")
        elif not candidate_data:
            split_score += 0.20
            split_signals.append("header_only_candidate")
        elif header_overlap < 0.20:
            split_score += 0.16
            split_signals.append("new_header_schema")
    elif source_headers and candidate_data:
        same_score += 0.16
        same_signals.append("header_to_headerless_body")
    elif candidate_headers and not source_headers:
        split_score += 0.14
        split_signals.append("candidate_introduces_header")

    if candidate.rows and _row_is_section_title(candidate.rows[0]):
        section_score += 0.62
        split_score += 0.22
        section_signals.append("strong_section_marker")

    profile_similarity = _profile_similarity(
        _column_profile(source.rows),
        _column_profile(candidate.rows),
    )
    if profile_similarity >= 0.75:
        same_score += 0.14
        same_signals.append("column_types_match")
    elif profile_similarity and profile_similarity < 0.35:
        split_score += 0.10
        split_signals.append("column_types_diverge")

    candidate_row = list(candidate.rows[candidate_row_index]) if 0 <= candidate_row_index < len(candidate.rows) else []
    validator = candidate_validator or (contract.row_predicate if contract is not None else None)
    semantic_match: bool | None = None
    if validator is not None:
        rows_to_validate = (
            [list(row) for row in candidate.rows]
            if candidate_validator_scope == "fragment"
            else ([candidate_row] if candidate_row else [])
        )
        semantic_match = any(validator(row) for row in rows_to_validate)
        if semantic_match:
            same_score += 0.28
            same_signals.append("semantic_row_shape_matches")
        else:
            split_score += 0.42
            split_signals.append("semantic_row_shape_diverges")
    if contract is not None:
        if candidate_row and (
            not contract.expected_columns or len(candidate_row) in contract.expected_columns
        ):
            same_score += 0.08
            same_signals.append("expected_column_contract")
        elif candidate_row:
            split_score += 0.18
            split_signals.append("unexpected_column_contract")
        signature = "".join(candidate_row)
        if any(marker in signature for marker in contract.forbidden_markers):
            split_score += 0.42
            split_signals.append("new_header_forbidden_by_context")
    if not candidate_data and not (validator is not None and candidate_row and validator(candidate_row)):
        split_score += 0.10
        split_signals.append("no_continuation_body")

    hypotheses = _ranked_hypotheses(
        {
            "same_table": (same_score, same_signals),
            "new_table": (split_score, split_signals),
            "new_section": (section_score, section_signals),
        }
    )
    selected = hypotheses[0]
    same_hypothesis = next(item for item in hypotheses if item.kind == "same_table")
    accepted = bool(
        selected.kind == "same_table"
        and same_hypothesis.score >= threshold
        and semantic_match is not False
        and page_gap in {0, 1}
        and candidate.index == source.index + 1
    )
    return PageBoundaryDecision(
        left_id=f"table:{source.table_id or source.index}",
        right_id=f"table:{candidate.table_id or candidate.index}",
        from_page=source.page,
        to_page=candidate.page,
        hypotheses=hypotheses,
        selected=selected.kind,
        confidence=selected.score,
        accepted=accepted,
        context=context,
    )


def _score_text_boundary(
    left: BoundaryUnit,
    right: BoundaryUnit,
    threshold: float,
) -> PageBoundaryDecision:
    same_score = 0.16
    split_score = 0.18
    section_score = 0.02
    same_signals: list[str] = []
    split_signals: list[str] = []
    section_signals: list[str] = []
    left_text = left.text.strip()
    right_text = right.text.strip()

    if left_text and left_text[-1] not in _TERMINAL_PUNCTUATION:
        same_score += 0.24
        same_signals.append("no_terminal_punctuation")
    else:
        split_score += 0.12
        split_signals.append("terminal_punctuation")
    if any(marker in _compact_text(right_text) for marker in _SECTION_MARKERS):
        section_score += 0.60
        split_score += 0.22
        section_signals.append("strong_section_heading")
    overlap = _horizontal_overlap(left.bbox, right.bbox)
    if overlap >= 0.70:
        same_score += 0.16
        same_signals.append("text_columns_aligned")
    if left.bbox is not None and left.page_height > 0 and left.bbox[3] / left.page_height >= 0.72:
        same_score += 0.12
        same_signals.append("text_reaches_page_bottom")
    if right.bbox is not None and right.page_height > 0 and right.bbox[1] / right.page_height <= 0.30:
        same_score += 0.12
        same_signals.append("text_starts_near_page_top")
    hypotheses = _ranked_hypotheses(
        {
            "same_body_text": (same_score, same_signals),
            "new_body_text": (split_score, split_signals),
            "new_section": (section_score, section_signals),
        }
    )
    selected = hypotheses[0]
    return PageBoundaryDecision(
        left_id=left.unit_id,
        right_id=right.unit_id,
        from_page=left.page,
        to_page=right.page,
        hypotheses=hypotheses,
        selected=selected.kind,
        confidence=selected.score,
        accepted=selected.kind == "same_body_text" and selected.score >= threshold,
        context="body_text",
    )


def _score_mixed_boundary(
    left: BoundaryUnit,
    right: BoundaryUnit,
) -> PageBoundaryDecision:
    related_score = 0.24
    unrelated_score = 0.30
    section_score = 0.02
    related_signals: list[str] = []
    unrelated_signals: list[str] = []
    section_signals: list[str] = []
    text_unit = right if right.kind == "text" else left
    compact = _compact_text(text_unit.text)
    if any(marker in compact for marker in _SECTION_MARKERS):
        section_score += 0.62
        unrelated_score += 0.12
        section_signals.append("strong_section_heading")
    if len(compact) <= 80 and any(marker in compact for marker in _HEADER_MARKERS):
        related_score += 0.34
        related_signals.append("table_heading_or_caption")
    overlap = _horizontal_overlap(left.bbox, right.bbox)
    if overlap >= 0.70:
        related_score += 0.16
        related_signals.append("mixed_blocks_aligned")
    elif left.bbox is not None and right.bbox is not None:
        unrelated_score += 0.12
        unrelated_signals.append("mixed_blocks_not_aligned")
    hypotheses = _ranked_hypotheses(
        {
            "table_related_text": (related_score, related_signals),
            "table_unrelated_text": (unrelated_score, unrelated_signals),
            "new_section": (section_score, section_signals),
        }
    )
    selected = hypotheses[0]
    return PageBoundaryDecision(
        left_id=left.unit_id,
        right_id=right.unit_id,
        from_page=left.page,
        to_page=right.page,
        hypotheses=hypotheses,
        selected=selected.kind,
        confidence=selected.score,
        accepted=False,
        context="mixed_table_text",
    )


def _entities_from_boundary_graph(
    units: tuple[BoundaryUnit, ...],
    decisions: list[PageBoundaryDecision],
) -> tuple[LogicalPageEntity, ...]:
    parent = {unit.unit_id: unit.unit_id for unit in units}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for decision in decisions:
        if decision.accepted and decision.left_id in parent and decision.right_id in parent:
            union(decision.left_id, decision.right_id)

    groups: dict[str, list[BoundaryUnit]] = {}
    for unit in units:
        groups.setdefault(find(unit.unit_id), []).append(unit)
    decision_by_pair = {(decision.left_id, decision.right_id): decision for decision in decisions if decision.accepted}
    entities: list[LogicalPageEntity] = []
    for entity_index, grouped_units in enumerate(groups.values(), start=1):
        ordered = sorted(grouped_units, key=lambda item: (item.page, item.order, item.unit_id))
        entity_decisions = tuple(
            decision_by_pair[(left.unit_id, right.unit_id)]
            for left, right in zip(ordered, ordered[1:])
            if (left.unit_id, right.unit_id) in decision_by_pair
        )
        confidence = min(decision.confidence for decision in entity_decisions) if entity_decisions else 1.0
        kinds = {unit.kind for unit in ordered}
        entities.append(
            LogicalPageEntity(
                entity_id=f"enterprise_entity:{entity_index:05d}",
                kind=next(iter(kinds)) if len(kinds) == 1 else "mixed",
                unit_ids=tuple(unit.unit_id for unit in ordered),
                pages=tuple(sorted({unit.page for unit in ordered})),
                confidence=round(confidence, 4),
                boundary_decisions=entity_decisions,
            )
        )
    return tuple(entities)


def numeric_row(
    row: Iterable[str],
    *,
    numeric_indexes: Iterable[int],
    nonempty_indexes: Iterable[int] = (),
) -> bool:
    """Conservative validator shared by declarative continuation contracts."""
    values = list(row)
    for index in nonempty_indexes:
        if index >= len(values) or not str(values[index] or "").strip():
            return False
    for index in numeric_indexes:
        if index >= len(values):
            return False
        raw = str(values[index] or "").replace(",", "").replace("，", "").replace(" ", "")
        if raw in {"", "-", "--", "—"}:
            return False
        try:
            float(raw)
        except ValueError:
            return False
    return True


def _date_like(value: Any) -> bool:
    raw = str(value or "").replace(" ", "").strip()
    return bool(
        re.fullmatch(
            r"(?:19|20)\d{2}[-年./]\d{1,2}[-月./]\d{1,2}日?",
            raw,
        )
    )


def _number_like(value: Any) -> bool:
    raw = str(value or "").replace(",", "").replace("，", "").replace(" ", "").strip()
    if raw in {"", "-", "--", "—"}:
        return False
    try:
        float(raw)
    except ValueError:
        return False
    return True


def _settled_account_detail_row(row: Row) -> bool:
    values = [str(value or "").strip() for value in row]
    if len(values) < 4:
        return False
    date_indexes = [index for index, value in enumerate(values) if _date_like(value)]
    if len(date_indexes) < 2:
        return False
    first_date, second_date = date_indexes[:2]
    suffix = re.sub(r"[^0-9A-Z]", "", "".join(values[:first_date]).upper())
    classification = next(
        (
            value
            for value in values[first_date + 1 : second_date]
            if value and value not in {"-", "--", "—"}
        ),
        "",
    )
    return bool(
        (not suffix or len(suffix) <= 16)
        and classification
        and not any(marker in classification for marker in ("账户编号", "五级分类", "关闭日期"))
        and any((not value or "还款" in value) for value in values[second_date + 1 :])
    )


def _attachment_history_continuation_row(row: Row) -> bool:
    values = [str(value or "").strip() for value in row]
    signature = "".join(values)
    if "逾期月数" in signature and "最近一次约定还款日期" in signature and "最近一次还款形式" in signature:
        return True

    dense = [value for value in values if value]
    if not dense or not 6 <= len(dense) <= 8:
        return False
    if _date_like(values[0] if values else ""):
        return True
    if values and not values[0]:
        return bool(any(_date_like(value) for value in values[1:]) and any(_number_like(value) for value in values[1:]))
    return False


FACILITY_VALUE_CONTRACT = ContinuationContract(
    name="facility_summary_values",
    expected_columns=frozenset(),
    row_predicate=lambda row: sum(_number_like(value) for value in row) >= 6,
    forbidden_markers=("非循环信用额度", "循环信用额度", "总额", "已用额度"),
)

CLOSED_SUMMARY_BODY_CONTRACT = ContinuationContract(
    name="closed_credit_summary_body",
    expected_columns=frozenset(),
    row_predicate=lambda row: bool(
        next((str(value or "").strip() for value in row if str(value or "").strip()), "")
        and sum(_number_like(value) for value in row) >= 4
    ),
    forbidden_markers=("正常类账户数", "关注类账户数", "不良类账户数"),
)

ACCOUNT_SETTLED_DETAIL_CONTRACT = ContinuationContract(
    name="enterprise_account_settled_detail",
    expected_columns=frozenset(),
    row_predicate=_settled_account_detail_row,
    forbidden_markers=("账户编号", "授信机构", "业务种类", "借款金额"),
)

ATTACHMENT_HISTORY_BODY_CONTRACT = ContinuationContract(
    name="enterprise_attachment_history_body",
    expected_columns=frozenset(),
    row_predicate=_attachment_history_continuation_row,
    forbidden_markers=("账户编号", "授信机构", "开户日期", "开立日期", "信息报告日期"),
)

__all__ = [
    "ACCOUNT_SETTLED_DETAIL_CONTRACT",
    "ATTACHMENT_HISTORY_BODY_CONTRACT",
    "BoundaryHypothesis",
    "BoundaryUnit",
    "CLOSED_SUMMARY_BODY_CONTRACT",
    "ContinuationContract",
    "ContinuationMatch",
    "ContinuationRejection",
    "EnterpriseContinuationResolver",
    "FACILITY_VALUE_CONTRACT",
    "LogicalPageEntity",
    "PageBoundaryDecision",
    "TableFragment",
    "numeric_row",
]
