# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Logical-page extraction context for personal detailed credit reports.

The context is post-seal and source conserving.  It owns the one cross-page
decode performed for a detailed report and memoizes expensive variant-owned
extractors without exposing mutable cached values to their consumers.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from types import MappingProxyType
from typing import Any, Callable, Iterable, TypeVar, cast

from docmirror.plugins.credit_report.personal_detail_scanned.page_topology import (
    PersonalDetailLogicalPageImageResolver,
    PersonalDetailPageTopology,
)
from docmirror.plugins.credit_report.shared.entity_decoder import (
    CreditReportEntityContext,
    CreditReportUnit,
    TransitionAction,
    TransitionHypothesis,
    UnitKind,
    decode_credit_report_units,
    score_credit_report_transition,
)

_T = TypeVar("_T")

_SECTION_MARKERS = (
    "个人基本信息",
    "信息概要",
    "信贷交易信息明细",
    "非信贷交易信息明细",
    "公共信息明细",
    "查询记录",
    "报告说明",
)
_FAMILY_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("inquiry", ("查询日期", "查询机构", "查询原因")),
    ("public_record", ("欠税记录", "民事判决", "强制执行", "行政处罚", "住房公积金")),
    ("liability", ("相关还款责任", "责任金额", "责任余额")),
    ("credit_line", ("授信协议信息", "授信协议标识", "授信额度用途")),
    ("residence", ("居住地址", "居住状况", "住宅电话")),
    ("employment", ("工作单位", "单位地址", "职业", "职务")),
    ("repayment", ("还款记录", "还款状态", "逾期金额", "月份")),
    ("account", ("账户标识", "账户状态", "管理机构", "发卡机构", "开立日期")),
    ("summary", ("信息概要", "账户数", "业务类型")),
)
_STRONG_FAMILY_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("liability", ("相关还款责任", "保证合同编号")),
    ("credit_line", ("授信协议",)),
    ("residence", ("居住地址",)),
    ("employment", ("工作单位",)),
    ("inquiry", ("查询记录",)),
    ("public_record", ("欠税记录", "民事判决", "强制执行", "行政处罚", "执行法院", "立案法院")),
    ("repayment", ("还款记录",)),
    ("account", ("贷账户", "贷记卡账户", "准贷记卡账户")),
    ("summary", ("信息概要",)),
)
_PAGE_NUMBER_RE = re.compile(r"^(?:第\s*\d+\s*页(?:[,，]\s*共\s*\d+\s*页)?|page\s*\d+)", re.I)
_PRINTED_PAGE_RE = re.compile(
    r"第\s*(?P<page>\d{1,3})\s*页\s*[,，]?\s*共\s*(?P<total>\d{1,3})\s*页"
)
_PRINTED_PAGE_ONLY_RE = re.compile(r"第\s*(?P<page>\d{1,3})\s*[页面]")
_PRINTED_TOTAL_ONLY_RE = re.compile(r"共\s*(?P<total>\d{1,3})\s*页?")
_NUMBERED_RE = re.compile(r"^\s*\d{1,4}[.、)]")
_ACCOUNT_ANCHOR_RE = re.compile(r"(?:账户|业务)\s*[（(]?\s*(\d{1,3})\s*[）)]?")
_BUSINESS_HEADING_RE = re.compile(
    r"^(?:"
    r"[（(][一二三四五六七八九十]+[）)].{0,24}"
    r"|账户\s*\d{1,3}(?:[（(].{0,80}[）)])?"
    r"|授信协议\s*\d{1,3}"
    r")$"
)
_CONTINUATIONS = frozenset(
    {"same_table", "table_to_text_related", "text_to_table_related", "same_text_section"}
)
_SPLIT_FOR: dict[TransitionAction, TransitionAction] = {
    "same_table": "different_table",
    "table_to_text_related": "table_to_text_unrelated",
    "text_to_table_related": "text_to_table_unrelated",
    "same_text_section": "different_text_section",
}


def _compact(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or ""))


def _finite(value: Any) -> float:
    try:
        result = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def _bbox(value: Any) -> tuple[float, float, float, float] | None:
    raw = value.get("bbox") if isinstance(value, dict) else getattr(value, "bbox", None)
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
    headers = tuple(str(value or "").replace("\n", "").strip() for value in getattr(table, "headers", None) or [])
    if headers:
        rows.append(headers)
    for row in getattr(table, "rows", None) or []:
        rows.append(
            tuple(
                str(getattr(cell, "text", cell) or "").replace("\n", "").strip()
                for cell in getattr(row, "cells", None) or []
            )
        )
    return tuple(rows)


def _owned_by_table(
    box: tuple[float, float, float, float] | None,
    table_boxes: Iterable[tuple[float, float, float, float]],
) -> bool:
    if box is None:
        return False
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    center = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
    for table in table_boxes:
        intersection = max(0.0, min(box[2], table[2]) - max(box[0], table[0])) * max(
            0.0, min(box[3], table[3]) - max(box[1], table[1])
        )
        center_inside = table[0] <= center[0] <= table[2] and table[1] <= center[1] <= table[3]
        if center_inside or (area > 0 and intersection / area >= 0.65):
            return True
    return False


def _geometry_owner(
    box: tuple[float, float, float, float] | None,
    owners: Iterable[tuple[tuple[float, float, float, float], str]],
) -> str:
    if box is None:
        return ""
    for owner_box, unit_id in owners:
        if _owned_by_table(box, (owner_box,)):
            return unit_id
    return ""


def _kind(text: str) -> UnitKind:
    compact = _compact(text).strip(":：")
    if any(marker in compact and len(compact) <= len(marker) + 12 for marker in _SECTION_MARKERS) or (
        len(compact) <= 96 and _BUSINESS_HEADING_RE.fullmatch(compact)
    ):
        return "heading"
    if _NUMBERED_RE.match(str(text or "")):
        return "ledger"
    return "text"


def _family(units: Iterable[CreditReportUnit]) -> str:
    text = _compact("\n".join(unit.text for unit in units))
    for family, markers in _STRONG_FAMILY_MARKERS:
        if any(marker in text for marker in markers):
            return family
    ranked = sorted(
        (
            (sum(marker in text for marker in markers), family)
            for family, markers in _FAMILY_MARKERS
        ),
        reverse=True,
    )
    return ranked[0][1] if ranked and ranked[0][0] >= 2 else ""


def _new_account_boundary(left_text: str, right_text: str) -> bool:
    left = _ACCOUNT_ANCHOR_RE.search(left_text)
    right = _ACCOUNT_ANCHOR_RE.search(right_text)
    return bool(left and right and left.group(1) != right.group(1))


def _families_compatible(left: str, right: str) -> bool:
    if not left or not right:
        return True
    if left == right:
        return True
    # Repayment/status grids are components of an account card, not a separate
    # business entity, even when their local schema contains no account header.
    return {left, right} == {"account", "repayment"}


class PersonalDetailTransitionPolicy:
    """Re-rank shared transition hypotheses using personal-report contracts."""

    def __init__(self, *, minimum_confidence: float = 0.46, minimum_margin: float = 0.06) -> None:
        self.minimum_confidence = minimum_confidence
        self.minimum_margin = minimum_margin
        self._cache: dict[tuple[tuple[str, ...], str, str], tuple[TransitionHypothesis, ...]] = {}

    def score(
        self,
        open_units: tuple[CreditReportUnit, ...],
        candidate: CreditReportUnit,
        lookahead: CreditReportUnit | None,
    ) -> tuple[TransitionHypothesis, ...]:
        key = (tuple(unit.unit_id for unit in open_units), candidate.unit_id, lookahead.unit_id if lookahead else "")
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        base = score_credit_report_transition(
            open_units,
            candidate,
            report_family="personal_detail",
            lookahead=lookahead,
        ).hypotheses
        tail = open_units[-1]
        crosses_page = candidate.page != tail.page

        weights = {hypothesis.action: max(hypothesis.score, 0.000001) for hypothesis in base}
        signals = {hypothesis.action: list(hypothesis.signals) for hypothesis in base}
        continuation = next((item for item in base if item.action in _CONTINUATIONS), None)
        if continuation is None:
            self._cache[key] = base
            return base

        split_action = _SPLIT_FOR[continuation.action]
        left_family = _family(open_units)
        right_family = _family((candidate,))
        hard_split = candidate.kind == "heading" or _new_account_boundary(
            _compact("\n".join(unit.text for unit in open_units)),
            _compact(candidate.text),
        )
        if not _families_compatible(left_family, right_family):
            hard_split = True
            signals[split_action].append("personal_detail_family_mismatch")
        elif left_family and right_family:
            weights[continuation.action] *= 1.35
            signals[continuation.action].append(
                f"personal_detail_family_continues:{left_family}:{right_family}"
            )
            if crosses_page and {left_family, right_family} <= {"account", "repayment"}:
                # One account card is composed of several differently shaped
                # grids. A page break can therefore change column count and
                # header schema without opening a new business entity.
                weights[continuation.action] *= 2.25
                weights[split_action] *= 0.50
                signals[continuation.action].append("personal_detail_account_card_continues")

        if hard_split:
            weights[continuation.action] *= 0.01
            weights[split_action] *= 5.0
            signals[split_action].append("personal_detail_semantic_veto")

        normalized = self._normalize(weights, signals)
        selected = normalized[0]
        if crosses_page and selected.action in _CONTINUATIONS:
            runner_up = normalized[1].score if len(normalized) > 1 else 0.0
            if selected.score < self.minimum_confidence or selected.score - runner_up < self.minimum_margin:
                weights[split_action] = max(weights[split_action], weights[selected.action] * 1.05)
                signals[split_action].append("personal_detail_conservative_boundary")
                normalized = self._normalize(weights, signals)
        self._cache[key] = normalized
        return normalized

    @staticmethod
    def _normalize(
        weights: dict[TransitionAction, float],
        signals: dict[TransitionAction, list[str]],
    ) -> tuple[TransitionHypothesis, ...]:
        total = sum(max(value, 0.0) for value in weights.values()) or 1.0
        return tuple(
            sorted(
                (
                    TransitionHypothesis(
                        action=action,
                        score=round(max(value, 0.0) / total, 6),
                        signals=tuple(signals[action]),
                    )
                    for action, value in weights.items()
                ),
                key=lambda item: (-item.score, item.action),
            )
        )


def _domain_specific(parse_result: Any) -> dict[str, Any]:
    value = getattr(getattr(parse_result, "entities", None), "domain_specific", None)
    return value if isinstance(value, dict) else {}


def _printed_reading_order(
    parse_result: Any,
    topology: PersonalDetailPageTopology | None = None,
) -> dict[int, int]:
    """Map sealed logical pages to the report's printed reading order.

    Detailed reports are commonly scanned as two-page spreads. Physical
    sheets can be out of order even though each half retains the report's own
    ``第 N 页，共 M 页`` footer. Provenance page numbers remain unchanged; only
    continuation and evidence traversal use this order.

    Reordering is deliberately conservative: every observed logical page must
    resolve to one unique printed number, either from its footer or from the
    adjacent half of the same source spread, and the observed totals must be
    coherent. Partial OCR therefore cannot invent an isolated page position.
    """
    texts_by_page: dict[int, list[str]] = {}
    observed_pages: set[int] = set()
    source_by_logical: dict[int, int] = {}

    for page_index, page in enumerate(getattr(parse_result, "pages", None) or [], start=1):
        logical = int(getattr(page, "page_number", 0) or page_index)
        observed_pages.add(logical)
        transform = dict(getattr(page, "coordinate_transform", None) or {})
        source_by_logical[logical] = int(
            transform.get("source_page_number")
            or getattr(page, "source_page_number", 0)
            or logical
        )
        texts_by_page.setdefault(logical, []).extend(
            str(getattr(block, "content", "") or "")
            for block in getattr(page, "texts", None) or []
        )

    for bundle in _domain_specific(parse_result).get("_page_evidence_bundles") or []:
        if not isinstance(bundle, dict):
            continue
        local = bundle.get("local_structure_evidence")
        if not isinstance(local, dict):
            continue
        logical = int(bundle.get("page") or local.get("page") or 0)
        if logical <= 0:
            continue
        observed_pages.add(logical)
        source_by_logical.setdefault(
            logical,
            int(bundle.get("source_page_number") or local.get("source_page") or logical),
        )
        texts_by_page.setdefault(logical, []).extend(
            str(line.get("text") or line.get("content") or "")
            for line in local.get("lines") or []
            if isinstance(line, dict)
        )

    identity = {page: page for page in observed_pages}
    if len(observed_pages) < 2:
        return identity

    printed_by_logical: dict[int, int] = {}
    totals: list[int] = []
    for logical in observed_pages:
        exact_matches = {
            (int(match.group("page")), int(match.group("total")))
            for text in texts_by_page.get(logical, ())
            for match in _PRINTED_PAGE_RE.finditer(text)
        }
        if len(exact_matches) == 1:
            printed, total = next(iter(exact_matches))
            if 1 <= printed <= total:
                printed_by_logical[logical] = printed
                totals.append(total)

    if len(printed_by_logical) != len(observed_pages):
        image_pages, image_totals = _ocr_printed_page_footers(
            parse_result,
            missing_pages=observed_pages - printed_by_logical.keys(),
            topology=topology,
        )
        printed_by_logical.update(image_pages)
        totals.extend(image_totals)

    _infer_paired_printed_pages(
        printed_by_logical,
        source_by_logical,
        topology=topology,
    )
    if len(printed_by_logical) != len(observed_pages):
        return identity
    if len(set(printed_by_logical.values())) != len(observed_pages):
        return identity

    total_counts = Counter(total for total in totals if total >= len(observed_pages))
    expected_total = total_counts.most_common(1)[0][0] if total_counts else len(observed_pages)
    printed_pages = set(printed_by_logical.values())
    if (
        max(printed_pages, default=0) > expected_total
        or min(printed_pages, default=1) < 1
        or expected_total - len(observed_pages) > 2
    ):
        return identity

    return {
        logical: index
        for index, logical in enumerate(
            sorted(observed_pages, key=lambda page: (printed_by_logical[page], page)),
            start=1,
        )
    }


def _ocr_printed_page_footers(
    parse_result: Any,
    *,
    missing_pages: Iterable[int],
    topology: PersonalDetailPageTopology | None = None,
) -> tuple[dict[int, int], list[int]]:
    """Read only footer strips for structural page order, never cell values."""
    try:
        from docmirror.ocr.repair.recognizers import rapidocr_recognize
        resolver = PersonalDetailLogicalPageImageResolver(
            parse_result,
            zoom=2.0,
            topology=topology,
        )
    except Exception:
        return {}, []

    printed: dict[int, int] = {}
    totals: list[int] = []
    for logical in sorted(set(int(page) for page in missing_pages if int(page) > 0)):
        rendered = resolver(logical)
        if not rendered:
            continue
        image = rendered.get("image")
        shape = getattr(image, "shape", None)
        if not shape or len(shape) < 2:
            continue
        footer = image[int(shape[0] * 0.82) : shape[0], :]
        words = rapidocr_recognize(footer)
        texts = [
            str(word.get("text") or "")
            for word in words
            if float(word.get("confidence") or 0.0) >= 0.75
        ]
        joined = " ".join(texts)
        pages = {int(match.group("page")) for match in _PRINTED_PAGE_ONLY_RE.finditer(joined)}
        if len(pages) == 1:
            printed[logical] = next(iter(pages))
        totals.extend(
            int(match.group("total"))
            for match in _PRINTED_TOTAL_ONLY_RE.finditer(joined)
        )
    resolver.clear()
    return printed, totals


def _infer_paired_printed_pages(
    printed_by_logical: dict[int, int],
    source_by_logical: dict[int, int],
    *,
    topology: PersonalDetailPageTopology | None = None,
) -> None:
    """Infer one unread footer from a geometry-confirmed adjacent half."""
    logicals_by_source: dict[int, list[int]] = {}
    for logical, source in source_by_logical.items():
        logicals_by_source.setdefault(source, []).append(logical)
    for logicals in logicals_by_source.values():
        ordered = topology.ordered_pair(logicals) if topology is not None else None
        if ordered is None:
            continue
        left, right = ordered
        if left in printed_by_logical and right not in printed_by_logical:
            printed_by_logical[right] = printed_by_logical[left] + 1
        elif right in printed_by_logical and left not in printed_by_logical:
            printed_by_logical[left] = printed_by_logical[right] - 1


def _evidence_key(page: int, line: dict[str, Any], index: int) -> str:
    evidence_ids = tuple(str(value) for value in line.get("evidence_ids") or [] if value)
    if evidence_ids:
        return f"evidence:{'|'.join(evidence_ids)}"
    box = _bbox(line)
    box_key = ":".join(f"{value:.2f}" for value in box) if box else ""
    return f"evidence:p{page}:{index}:{box_key}:{_compact(line.get('text') or line.get('content'))[:80]}"


def _collect_personal_detail_units(
    parse_result: Any,
    *,
    topology: PersonalDetailPageTopology | None = None,
) -> tuple[
    tuple[CreditReportUnit, ...],
    tuple[str, ...],
    dict[str, str],
    dict[int, int],
    dict[int, int],
]:
    candidates: list[CreditReportUnit] = []
    furniture: set[str] = set()
    evidence_units: dict[str, str] = {}
    source_pages: dict[int, int] = {}
    pages = list(getattr(parse_result, "pages", None) or [])
    table_owners: dict[int, list[tuple[tuple[float, float, float, float], str]]] = {}
    text_owners: dict[int, list[tuple[str, tuple[float, float, float, float] | None, str]]] = {}
    edge_occurrences: dict[str, list[tuple[int, str]]] = {}
    reading_order = _printed_reading_order(parse_result, topology)

    for page_index, page in enumerate(pages, start=1):
        logical = int(getattr(page, "page_number", 0) or page_index)
        source = int(getattr(page, "source_page_number", 0) or logical)
        source_pages[logical] = source
        width = _finite(getattr(page, "width", 0))
        height = _finite(getattr(page, "height", 0))
        positioned: list[tuple[float, int, int, CreditReportUnit]] = []
        table_boxes = tuple(box for table in getattr(page, "tables", None) or [] if (box := _bbox(table)))
        for table_index, table in enumerate(getattr(page, "tables", None) or []):
            rows = _raw_rows(table)
            if not rows:
                continue
            table_id = str(getattr(table, "table_id", "") or f"p{logical}:t{table_index}")
            box = _bbox(table)
            unit_id = f"personal_detail:table:p{logical}:{table_id}"
            unit = CreditReportUnit(
                unit_id=unit_id,
                page=reading_order.get(logical, logical),
                order=0,
                source_index=table_index,
                kind="table",
                text="\n".join(" | ".join(row) for row in rows),
                bbox=box,
                page_width=width,
                page_height=height,
                table_id=table_id,
                rows=rows,
            )
            positioned.append((box[1] if box else 10000.0 + table_index, 1, table_index, unit))
            if box is not None:
                table_owners.setdefault(logical, []).append((box, unit_id))

        for text_index, block in enumerate(getattr(page, "texts", None) or []):
            content = str(getattr(block, "content", "") or "").strip()
            box = _bbox(block)
            if not content or _owned_by_table(box, table_boxes):
                continue
            unit_id = f"personal_detail:text:p{logical}:{text_index}"
            unit = CreditReportUnit(
                unit_id=unit_id,
                page=reading_order.get(logical, logical),
                order=0,
                source_index=text_index,
                kind=_kind(content),
                text=content,
                bbox=box,
                page_width=width,
                page_height=height,
            )
            positioned.append((box[1] if box else float(text_index), 0, text_index, unit))
            text_owners.setdefault(logical, []).append((_compact(content), box, unit_id))
            if _PAGE_NUMBER_RE.match(_compact(content)):
                furniture.add(unit_id)
            elif box is not None and height > 0 and (box[1] / height <= 0.08 or box[3] / height >= 0.90):
                compact = _compact(content)
                if len(compact) <= 80:
                    edge_occurrences.setdefault(compact, []).append((logical, unit_id))

        for order, (_top, _kind_order, _index, unit) in enumerate(sorted(positioned)):
            candidates.append(replace(unit, order=order))

    bundles = _domain_specific(parse_result).get("_page_evidence_bundles") or []
    for bundle in bundles:
        if not isinstance(bundle, dict):
            continue
        local = bundle.get("local_structure_evidence")
        if not isinstance(local, dict):
            continue
        logical = int(bundle.get("page") or local.get("page") or 0)
        if logical <= 0:
            continue
        source_pages.setdefault(logical, int(bundle.get("source_page_number") or local.get("source_page") or logical))
        lines = [dict(line) for line in local.get("lines") or [] if isinstance(line, dict)]
        width = _finite(
            local.get("page_width") or bundle.get("page_width") or local.get("width") or bundle.get("width")
        )
        height = _finite(
            local.get("page_height") or bundle.get("page_height") or local.get("height") or bundle.get("height")
        )
        ordered_lines = sorted(
            lines,
            key=lambda item: (
                (_bbox(item) or (0, 0, 0, 0))[1],
                (_bbox(item) or (0, 0, 0, 0))[0],
            ),
        )
        for line_index, line in enumerate(ordered_lines):
            content = str(line.get("text") or line.get("content") or "").strip()
            if not content:
                continue
            evidence_key = _evidence_key(logical, line, line_index)
            box = _bbox(line)
            owner = _geometry_owner(box, table_owners.get(logical, ()))
            if not owner:
                compact = _compact(content)
                owner = next(
                    (
                        unit_id
                        for native_text, native_box, unit_id in text_owners.get(logical, ())
                        if native_text == compact
                        or (native_box is not None and _geometry_owner(box, ((native_box, unit_id),)))
                    ),
                    "",
                )
            if owner:
                evidence_units[evidence_key] = owner
                continue
            unit_id = f"personal_detail:evidence:p{logical}:{line_index}"
            evidence_units[evidence_key] = unit_id
            candidates.append(
                CreditReportUnit(
                    unit_id=unit_id,
                    page=reading_order.get(logical, logical),
                    order=line_index,
                    source_index=line_index,
                    kind=_kind(content),
                    text=content,
                    bbox=box,
                    page_width=width,
                    page_height=height,
                )
            )
            if _PAGE_NUMBER_RE.match(_compact(content)):
                furniture.add(unit_id)
            elif box is not None and height > 0 and (box[1] / height <= 0.08 or box[3] / height >= 0.90):
                compact = _compact(content)
                if len(compact) <= 80:
                    edge_occurrences.setdefault(compact, []).append((logical, unit_id))

    page_count = len(source_pages)
    recurrence_minimum = max(2, math.ceil(page_count * 0.5))
    for occurrences in edge_occurrences.values():
        if len({page for page, _unit_id in occurrences}) >= recurrence_minimum:
            furniture.update(unit_id for _page, unit_id in occurrences)

    active_by_page: dict[int, list[CreditReportUnit]] = {}
    for unit in candidates:
        if unit.unit_id not in furniture:
            active_by_page.setdefault(unit.page, []).append(unit)
    active: list[CreditReportUnit] = []
    for page in sorted(active_by_page):
        page_units = sorted(
            active_by_page[page],
            key=lambda unit: (
                unit.bbox[1] if unit.bbox is not None else 10000.0 + unit.order,
                1 if unit.kind == "table" else 0,
                unit.source_index,
                unit.unit_id,
            ),
        )
        active.extend(replace(unit, order=order) for order, unit in enumerate(page_units))
    return tuple(active), tuple(sorted(furniture)), evidence_units, source_pages, reading_order


class PersonalDetailExtractionContext:
    """Variant-owned logical page graph and copy-on-read extraction cache."""

    def __init__(
        self,
        parse_result: Any,
        entity_context: CreditReportEntityContext,
        *,
        evidence_unit_ids: dict[str, str],
        source_page_by_logical: dict[int, int],
        reading_order_by_logical: dict[int, int],
        page_topology: PersonalDetailPageTopology,
    ) -> None:
        self.parse_result = parse_result
        self.entity_context = entity_context
        self.evidence_unit_ids = MappingProxyType(dict(evidence_unit_ids))
        self.source_page_by_logical = MappingProxyType(dict(source_page_by_logical))
        self.reading_order_by_logical = MappingProxyType(dict(reading_order_by_logical))
        self.page_topology = page_topology
        self._cache: dict[str, Any] = {}
        from docmirror.plugins.credit_report.personal_detail_scanned.ocr_correction import (
            PersonalDetailOCRCorrectionOverlay,
        )

        self._ocr_correction_overlay = PersonalDetailOCRCorrectionOverlay(
            parse_result,
            page_image_resolver=PersonalDetailLogicalPageImageResolver(
                parse_result,
                topology=page_topology,
            ),
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.parse_result, name)

    def cached(self, key: str, factory: Callable[[], _T]) -> _T:
        if key not in self._cache:
            self._cache[key] = deepcopy(factory())
        return cast(_T, deepcopy(self._cache[key]))

    def account_collections(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        from docmirror.plugins.credit_report.personal_detail_scanned.native_extraction import _extract_accounts

        return cast(
            tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]],
            self.cached("account_collections", lambda: _extract_accounts(self)),
        )

    def native_business(self, full_text: str) -> dict[str, Any]:
        from docmirror.plugins.credit_report.personal_detail_scanned.native_extraction import (
            extract_personal_detail_native_business,
        )

        return self.cached(
            "native_business",
            lambda: self._ocr_correction_overlay.correct_business_candidates(
                extract_personal_detail_native_business(self, full_text),
                stage="native_business",
            ),
        )

    def scanned_business(self, full_text: str) -> dict[str, Any]:
        from docmirror.plugins.credit_report.scanned_business import extract_scanned_credit_business

        return self.cached(
            "scanned_business",
            lambda: self._ocr_correction_overlay.correct_business_candidates(
                extract_scanned_credit_business(self, full_text),
                stage="scanned_business",
            ),
        )

    def corrected_repayment_records(self) -> list[dict[str, Any]]:
        """Rebuild monthly cells from images using corrected document order.

        Parse-time micro-grids may contain ``unknown`` placeholders and may
        have augmented a shuffled physical neighbor. Work on a detached copy,
        force grid reconstruction, and enable typed one-cell OCR. The sealed
        evidence plane remains unchanged.
        """

        def rebuild() -> list[dict[str, Any]]:
            from docmirror.models.mirror.domain_access import (
                micro_grid_structures_from_domain_specific,
            )
            from docmirror.plugins.credit_report.micro_grid_materialize import (
                augment_credit_repayment_evidence_bundles,
                materialize_credit_repayment_micro_grids_from_bundles,
            )
            from docmirror.plugins.credit_report.repayment_grid import (
                dedupe_repayment_records,
                records_from_micro_grid_dict,
            )

            detached = deepcopy(_domain_specific(self.parse_result))
            detached.pop("credit_repayment_records", None)
            for bundle in detached.get("_page_evidence_bundles") or []:
                if isinstance(bundle, dict):
                    bundle.pop("micro_grid_structures", None)
            augment_credit_repayment_evidence_bundles(
                detached,
                reading_order_by_logical=dict(self.reading_order_by_logical),
            )
            resolver = PersonalDetailLogicalPageImageResolver(
                self.parse_result,
                topology=self.page_topology,
            )
            try:
                materialize_credit_repayment_micro_grids_from_bundles(
                    detached,
                    page_image_resolver=resolver,
                    enable_cell_ocr=True,
                    extra_status_chars={"A"},
                )
            finally:
                resolver.clear()
            records = [
                record
                for grid in micro_grid_structures_from_domain_specific(detached)
                for record in records_from_micro_grid_dict(grid)
            ]
            return dedupe_repayment_records(records)

        return self.cached("corrected_repayment_records", rebuild)

    def section_content(self, full_text: str) -> dict[str, Any]:
        from docmirror.plugins.credit_report.personal_detail_scanned.native_extraction import (
            extract_personal_detail_section_content,
        )

        return self.cached(
            "section_content",
            lambda: self._ocr_correction_overlay.correct_business_candidates(
                extract_personal_detail_section_content(self, full_text),
                stage="section_content",
            ),
        )

    def corrected_evidence_pages(self) -> list[dict[str, Any]]:
        """Return corrected copies of OCR lines while preserving sealed evidence."""
        domain_specific = _domain_specific(self.parse_result)
        pages: list[dict[str, Any]] = []
        for bundle in domain_specific.get("_page_evidence_bundles") or []:
            if not isinstance(bundle, dict):
                continue
            local = bundle.get("local_structure_evidence")
            if not isinstance(local, dict):
                continue
            lines = [dict(line) for line in local.get("lines") or [] if isinstance(line, dict)]
            if not lines:
                continue
            pages.append(
                {
                    "page": int(bundle.get("page") or local.get("page") or 0),
                    "source_page": int(bundle.get("source_page_number") or local.get("source_page") or 0),
                    "page_width": _finite(
                        local.get("page_width")
                        or bundle.get("page_width")
                        or local.get("width")
                        or bundle.get("width")
                    ),
                    "page_height": _finite(
                        local.get("page_height")
                        or bundle.get("page_height")
                        or local.get("height")
                        or bundle.get("height")
                    ),
                    "lines": sorted(
                        lines,
                        key=lambda line: (
                            (_bbox(line) or (0, 0, 0, 0))[1],
                            (_bbox(line) or (0, 0, 0, 0))[0],
                        ),
                    ),
                }
            )
        return self._ocr_correction_overlay.corrected_evidence_pages(
            sorted(
                pages,
                key=lambda item: (
                    self.reading_order_by_logical.get(item["page"], item["page"]),
                    item["page"],
                ),
            )
        )

    def ocr_correction_audit(self) -> dict[str, Any]:
        """Return a detached audit snapshot for diagnostics and regression tests."""
        return deepcopy(self._ocr_correction_overlay.audit())

    def page_topology_audit(self) -> dict[str, Any]:
        """Return the plugin's detached logical-page validation result."""
        return deepcopy(self.page_topology.audit())

    def tables_continue(self, left_table_id: str, right_table_id: str) -> bool | None:
        left_unit_id = self.entity_context.table_unit_id(left_table_id)
        right_unit_id = self.entity_context.table_unit_id(right_table_id)
        if not left_unit_id or not right_unit_id:
            return None
        left = self.entity_context.entity_for_unit(left_unit_id)
        right = self.entity_context.entity_for_unit(right_unit_id)
        return bool(left is not None and right is not None and left.entity_id == right.entity_id)

    def allows_scanned_line_transition(
        self,
        left_page: int,
        left_line: dict[str, Any],
        left_index: int,
        right_page: int,
        right_line: dict[str, Any],
        right_index: int,
    ) -> bool | None:
        if left_page == right_page:
            return True
        left_id = self.evidence_unit_ids.get(_evidence_key(left_page, left_line, left_index))
        right_id = self.evidence_unit_ids.get(_evidence_key(right_page, right_line, right_index))
        if not left_id or not right_id:
            return None
        left = self.entity_context.entity_for_unit(left_id)
        right = self.entity_context.entity_for_unit(right_id)
        return bool(left is not None and right is not None and left.entity_id == right.entity_id)


def build_personal_detail_extraction_context(parse_result: Any) -> PersonalDetailExtractionContext:
    """Build the detailed-report logical-page graph exactly once."""
    if isinstance(parse_result, PersonalDetailExtractionContext):
        return parse_result
    page_topology = PersonalDetailPageTopology(parse_result)
    units, furniture, evidence_units, source_pages, reading_order = _collect_personal_detail_units(
        parse_result,
        topology=page_topology,
    )
    policy = PersonalDetailTransitionPolicy()
    entity_context = decode_credit_report_units(
        units,
        report_family="personal_detail",
        furniture_unit_ids=furniture,
        transition_scorer=policy.score,
        entity_prefix="personal_detail",
    )
    return PersonalDetailExtractionContext(
        parse_result,
        entity_context,
        evidence_unit_ids=evidence_units,
        source_page_by_logical=source_pages,
        reading_order_by_logical=reading_order,
        page_topology=page_topology,
    )


__all__ = [
    "PersonalDetailExtractionContext",
    "PersonalDetailTransitionPolicy",
    "build_personal_detail_extraction_context",
]
