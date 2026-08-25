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
