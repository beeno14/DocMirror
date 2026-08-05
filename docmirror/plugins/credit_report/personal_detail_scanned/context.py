# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Logical-page extraction context for personal detailed credit reports.

The context is post-seal and source conserving.  It owns the one cross-page
decode performed for a detailed report and memoizes expensive variant-owned
extractors without exposing mutable cached values to their consumers.
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from types import MappingProxyType
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping, TypeVar, cast

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
_PRINTED_PAGE_RE = re.compile(r"第\s*(?P<page>\d{1,3})\s*页\s*[,，]?\s*共\s*(?P<total>\d{1,3})\s*页")
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
_CONTINUATIONS = frozenset({"same_table", "table_to_text_related", "text_to_table_related", "same_text_section"})
_SPLIT_FOR: dict[TransitionAction, TransitionAction] = {
    "same_table": "different_table",
    "table_to_text_related": "table_to_text_unrelated",
    "text_to_table_related": "text_to_table_unrelated",
    "same_text_section": "different_text_section",
}
_PAGE_OCR_ANCHORS = (
    "个人信用报告",
    "报告编号",
    "个人基本信息",
    "信息概要",
    "信贷交易信息明细",
    "查询记录",
    "账户",
    "管理机构",
    "查询日期",
)


def _compact(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or ""))


def _finite(value: Any) -> float:
    try:
        result = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def _page_ocr_score(words: Iterable[dict[str, Any]], *, image_shape: Any = None) -> float:
    texts = [str(word.get("text") or "").strip() for word in words if str(word.get("text") or "").strip()]
    joined = " ".join(texts)
    anchors = sum(joined.count(marker) for marker in _PAGE_OCR_ANCHORS)
    long_cjk = sum(1 for text in texts if len(re.findall(r"[\u3400-\u9fff]", text)) >= 2)
    confidences = [float(word.get("confidence") or 0.0) for word in words if word.get("text")]
    mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    horizontal_chars = 0
    vertical_chars = 0
    for word in words:
        text = str(word.get("text") or "").strip()
        box = word.get("bbox")
        if not text or not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        width = max(0.0, float(box[2]) - float(box[0]))
        height = max(0.0, float(box[3]) - float(box[1]))
        weight = max(1, len(text))
        if width >= height * 1.15:
            horizontal_chars += weight
        elif height >= width * 1.5:
            vertical_chars += weight
    portrait_score = 0.0
    if isinstance(image_shape, (list, tuple)) and len(image_shape) >= 2:
        height = float(image_shape[0] or 0.0)
        width = float(image_shape[1] or 0.0)
        if height > 0 and width > 0:
            # Canonical detailed-report logical pages are portrait.  This
            # prior breaks dense-table ties where rotating an already upright
            # page produces more, but vertically fragmented, OCR tokens.
            portrait_score = 240.0 if height >= width else -240.0
    return float(
        len(joined)
        + anchors * 80
        + long_cjk * 4
        + mean_confidence * 30
        + horizontal_chars * 2.5
        - vertical_chars * 3.0
        + portrait_score
    )


def _complete_page_ocr(image: Any) -> tuple[list[dict[str, Any]], Any, int, float, float]:
    """OCR a complete logical page with per-page orientation and deskew selection."""
    import cv2

    from docmirror.ocr.image_preprocessing import deskew_image
    from docmirror.ocr.repair.recognizers import rapidocr_recognize

    rotations = (
        (0, 90, 180, 270) if os.environ.get("DOCMIRROR_PERSONAL_DETAIL_PAGE_ORIENTATION_PROBE", "1") != "0" else (0,)
    )
    candidates: list[tuple[float, int, list[dict[str, Any]], Any]] = []
    for rotation in rotations:
        if rotation == 0:
            oriented = image
        elif rotation == 90:
            oriented = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == 180:
            oriented = cv2.rotate(image, cv2.ROTATE_180)
        else:
            oriented = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        words = rapidocr_recognize(oriented, source="personal_detail_full_page_ocr")
        candidates.append((_page_ocr_score(words, image_shape=getattr(oriented, "shape", None)), rotation, words, oriented))
    score, rotation, words, selected = max(candidates, key=lambda item: item[0])
    deskewed, deskew_angle = deskew_image(selected)
    if abs(float(deskew_angle or 0.0)) >= 0.5:
        deskew_words = rapidocr_recognize(
            deskewed,
            source="personal_detail_full_page_ocr_deskewed",
        )
        deskew_score = _page_ocr_score(deskew_words, image_shape=getattr(deskewed, "shape", None))
        if deskew_score > score:
            return deskew_words, deskewed, rotation, float(deskew_angle), deskew_score
    return words, selected, rotation, 0.0, score


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
        ((sum(marker in text for marker in markers), family) for family, markers in _FAMILY_MARKERS),
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
            signals[continuation.action].append(f"personal_detail_family_continues:{left_family}:{right_family}")
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
            transform.get("source_page_number") or getattr(page, "source_page_number", 0) or logical
        )
        texts_by_page.setdefault(logical, []).extend(
            str(getattr(block, "content", "") or "") for block in getattr(page, "texts", None) or []
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
    if max(printed_pages, default=0) > expected_total or min(printed_pages, default=1) < 1:
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
        texts = [str(word.get("text") or "") for word in words if float(word.get("confidence") or 0.0) >= 0.75]
        joined = " ".join(texts)
        pages = {int(match.group("page")) for match in _PRINTED_PAGE_ONLY_RE.finditer(joined)}
        if len(pages) == 1:
            printed[logical] = next(iter(pages))
        totals.extend(int(match.group("total")) for match in _PRINTED_TOTAL_ONLY_RE.finditer(joined))
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
        # Plugin-owned supplemental subpages are added lazily after a confirmed
        # split replay, so these two ledgers intentionally remain mutable while
        # the sealed ParseResult and evidence IDs stay immutable.
        self.source_page_by_logical = dict(source_page_by_logical)
        self.reading_order_by_logical = dict(reading_order_by_logical)
        self.page_topology = page_topology
        self._cache: dict[str, Any] = {}
        self._page_ocr_cache: dict[int, dict[str, Any] | None] = {}
        self._supplemental_ocr_cache: dict[str, dict[str, Any]] = {}
        self._page_ocr_requests: list[dict[str, Any]] = []
        self._canonical_layout_projection_cache: Any | None = None
        self._canonical_entity_context_ready = False
        self._page_ocr_max_requests = max(
            0, int(os.environ.get("DOCMIRROR_PERSONAL_DETAIL_PAGE_OCR_MAX_REQUESTS", "12"))
        )
        from docmirror.plugins.credit_report.personal_detail_scanned.ocr_correction import (
            PersonalDetailOCRCorrectionOverlay,
        )

        self._page_image_resolver = PersonalDetailLogicalPageImageResolver(
            parse_result,
            topology=page_topology,
        )
        self._ocr_correction_overlay = PersonalDetailOCRCorrectionOverlay(
            parse_result,
            page_image_resolver=self._page_image_resolver,
            full_page_ocr_loader=self.full_page_ocr_evidence,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.parse_result, name)

    @property
    def pages(self) -> list[Any]:
        """Return detached canonical pages, never the sealed ParseResult pages."""
        return list(self._canonical_layout_projection().pages)

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
        """Compatibility view of the single Candidate B business result."""
        return deepcopy(self.candidate_b_extraction(full_text).business)

    def scanned_business(self, full_text: str) -> dict[str, Any]:
        """Compatibility view; no shared scanned extractor is invoked."""
        return deepcopy(self.candidate_b_extraction(full_text).business)

    def candidate_b_extraction(self, full_text: str) -> Any:
        """Build and retain the only business extraction for this document."""
        key = "candidate_b_extraction"
        if key not in self._cache:
            from docmirror.plugins.credit_report.personal_detail_scanned.candidate_b import (
                CandidateBPipeline,
            )

            self._cache[key] = CandidateBPipeline(self, full_text).run()
        return self._cache[key]

    def correct_candidate_b_business(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Apply the branch's correction and cross-field contracts exactly once."""
        corrected = self._ocr_correction_overlay.correct_business_candidates(
            payload,
            stage="candidate_b_schema_extraction",
        )
        return self._ocr_correction_overlay.enforce_cross_field_contracts(
            corrected,
            stage="candidate_b_schema_extraction",
        )

    def correct_assembled_business(self, payload: dict[str, Any], *, stage: str) -> dict[str, Any]:
        """Compatibility hook; Candidate B has already performed this pass."""
        del stage
        return deepcopy(payload)

    def corrected_repayment_records(self) -> list[dict[str, Any]]:
        """Rebuild monthly cells from the same canonical page evidence as all fields.

        No cell-level OCR is permitted here.  Complete-page OCR replay and
        canonical fragment registration happen before the repayment grid is
        materialized, so monthly performance cannot bypass the template layer.
        The sealed evidence plane remains unchanged.
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

            canonical_pages = {
                int(page.get("page") or 0): page
                for page in self.corrected_evidence_pages()
                if isinstance(page, dict) and int(page.get("page") or 0) > 0
            }
            observed_pages: set[int] = set()
            for bundle in detached.get("_page_evidence_bundles") or []:
                if not isinstance(bundle, dict):
                    continue
                local = bundle.get("local_structure_evidence")
                page = int(bundle.get("page") or (local or {}).get("page") or 0)
                canonical = canonical_pages.get(page)
                if canonical is None:
                    continue
                observed_pages.add(page)
                lines = deepcopy(list(canonical.get("lines") or []))
                if not isinstance(local, dict):
                    local = {}
                    bundle["local_structure_evidence"] = local
                local.update(
                    {
                        "page": page,
                        "source_page": int(canonical.get("source_page") or page),
                        "page_width": canonical.get("page_width"),
                        "page_height": canonical.get("page_height"),
                        "lines": lines,
                    }
                )
                grid_evidence = bundle.get("micro_grid_evidence")
                if not isinstance(grid_evidence, dict):
                    grid_evidence = {}
                    bundle["micro_grid_evidence"] = grid_evidence
                grid_evidence.update(
                    {
                        "page": page,
                        "page_width": canonical.get("page_width"),
                        "page_height": canonical.get("page_height"),
                        "lines": deepcopy(lines),
                        # Tokens from the sealed pre-registration page must not
                        # outrank the canonical complete-page evidence.
                        "tokens": deepcopy(lines),
                    }
                )
            for page, canonical in canonical_pages.items():
                if page in observed_pages:
                    continue
                lines = deepcopy(list(canonical.get("lines") or []))
                detached.setdefault("_page_evidence_bundles", []).append(
                    {
                        "page": page,
                        "source_page_number": int(canonical.get("source_page") or page),
                        "local_structure_evidence": {
                            "page": page,
                            "source_page": int(canonical.get("source_page") or page),
                            "page_width": canonical.get("page_width"),
                            "page_height": canonical.get("page_height"),
                            "lines": deepcopy(lines),
                        },
                        "micro_grid_evidence": {
                            "page": page,
                            "page_width": canonical.get("page_width"),
                            "page_height": canonical.get("page_height"),
                            "lines": lines,
                            "tokens": deepcopy(lines),
                        },
                    }
                )
            augment_credit_repayment_evidence_bundles(
                detached,
                reading_order_by_logical=dict(self.reading_order_by_logical),
            )
            materialize_credit_repayment_micro_grids_from_bundles(
                detached,
                page_image_resolver=None,
                enable_cell_ocr=False,
                extra_status_chars={"A"},
            )
            records = [
                record
                for grid in micro_grid_structures_from_domain_specific(detached)
                for record in records_from_micro_grid_dict(grid)
            ]
            deduped = dedupe_repayment_records(records)

            # Build an OCR-free structural witness from the detached sealed
            # page evidence.  It is never returned and none of its cell values
            # can enter the business projection; it only detects complete grid
            # ranges that canonical registration may have missed.
            source_baseline = deepcopy(_domain_specific(self.parse_result))
            source_baseline.pop("credit_repayment_records", None)
            for bundle in source_baseline.get("_page_evidence_bundles") or []:
                if isinstance(bundle, dict):
                    bundle.pop("micro_grid_structures", None)
            augment_credit_repayment_evidence_bundles(
                source_baseline,
                reading_order_by_logical=dict(self.reading_order_by_logical),
            )
            materialize_credit_repayment_micro_grids_from_bundles(
                source_baseline,
                page_image_resolver=None,
                enable_cell_ocr=False,
                extra_status_chars={"A"},
            )
            source_structure_records = dedupe_repayment_records(
                [
                    record
                    for grid in micro_grid_structures_from_domain_specific(source_baseline)
                    for record in records_from_micro_grid_dict(grid)
                ]
            )
            source_structure_count = len(source_structure_records)

            months_by_series: dict[str, set[int]] = {}
            for record in deduped:
                account_id = str(record.get("account_id") or "").strip()
                grid_id = str(record.get("grid_id") or "").strip()
                if not grid_id:
                    grid_id = next(
                        (
                            str(ref.get("grid_id") or "").strip()
                            for ref in record.get("source_cell_refs") or []
                            if isinstance(ref, dict) and str(ref.get("grid_id") or "").strip()
                        ),
                        "",
                    )
                month = str(record.get("performance_month") or "").strip()
                if not month:
                    year_value = int(record.get("year") or 0)
                    month_value = int(record.get("month") or 0)
                    if 2000 <= year_value <= 2099 and 1 <= month_value <= 12:
                        month = f"{year_value:04d}-{month_value:02d}"
                match = re.fullmatch(r"(20\d{2})-(0[1-9]|1[0-2])", month)
                series_id = account_id or grid_id
                if not series_id or match is None:
                    continue
                month_index = int(match.group(1)) * 12 + int(match.group(2))
                months_by_series.setdefault(series_id, set()).add(month_index)
            # The schema requires one observation per account/month.  Once an
            # account has observations on both sides of a month, a hole in
            # that interval is direct structural evidence of missing or
            # mis-linked cells.  This capacity check is independent of the
            # retired typed-cell OCR result and of any fixture-specific count.
            schema_implied_count = sum(
                max(months) - min(months) + 1
                for months in months_by_series.values()
                if months
            )
            structural_expected_count = max(schema_implied_count, source_structure_count)
            missing_month_count = max(0, structural_expected_count - len(deduped))
            if missing_month_count:
                from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
                    make_issue,
                    record_issue,
                )

                record_issue(
                    self,
                    make_issue(
                        category="ocr_structure_correction",
                        issue_code="canonical_monthly_reconstruction_incomplete",
                        message=(
                            "The unified canonical-page pass reconstructed fewer account-month positions than "
                            "the schema and detached source-structure witnesses; missing cells were not silently invented."
                        ),
                        parser_stage="canonical_monthly_grid_materialization",
                        target_dataset="repayment_records",
                        observed_value={"canonical_row_count": len(deduped)},
                        candidate_value={
                            "structural_expected_row_count": structural_expected_count,
                            "schema_implied_row_count": schema_implied_count,
                            "source_structure_row_count": source_structure_count,
                            "missing_month_count": missing_month_count,
                            "affected_account_or_grid_count": sum(
                                1
                                for months in months_by_series.values()
                                if months and max(months) - min(months) + 1 > len(months)
                            ),
                        },
                        reason_codes=(
                            "cell_level_ocr_disabled",
                            "canonical_page_evidence_only",
                            "source_structure_is_audit_only",
                            "dataset_incomplete",
                        ),
                    ),
                )
            return deduped

        return self.cached("corrected_repayment_records", rebuild)

    def section_content(self, full_text: str) -> dict[str, Any]:
        """Return supplemental datasets from the same Candidate B result."""
        return deepcopy(self.candidate_b_extraction(full_text).section_content)

    def corrected_evidence_pages(self) -> list[dict[str, Any]]:
        """Return the registered canonical evidence shared by every extractor."""
        return deepcopy(list(self._canonical_layout_projection().evidence_pages))

    def _source_corrected_evidence_pages(self) -> list[dict[str, Any]]:
        """Return detached pre-template evidence used only to register pages."""
        return self.cached("source_corrected_evidence_pages", self._build_corrected_evidence_pages)

    def _canonical_layout_projection(self) -> Any:
        if self._canonical_layout_projection_cache is None:
            from docmirror.plugins.credit_report.personal_detail_scanned.canonical_layout import (
                PBOCCanonicalTemplateAssembler,
            )

            assembler = PBOCCanonicalTemplateAssembler(
                self.parse_result,
                topology=self.page_topology,
                reading_order_by_logical=self.reading_order_by_logical,
                source_evidence_loader=self._source_corrected_evidence_pages,
                full_page_ocr_loader=self.full_page_ocr_evidence,
                issue_owner=self,
            )
            self._canonical_layout_projection_cache = assembler.build()
            self._adopt_canonical_entity_context()
        return self._canonical_layout_projection_cache

    def _adopt_canonical_entity_context(self) -> None:
        """Rebuild continuation identities over detached canonical pages."""
        if self._canonical_entity_context_ready or self._canonical_layout_projection_cache is None:
            return
        adapter = SimpleNamespace(
            pages=list(self._canonical_layout_projection_cache.pages),
            entities=SimpleNamespace(domain_specific={}),
        )
        units, furniture, evidence_units, source_pages, reading_order = _collect_personal_detail_units(
            adapter,
            topology=self.page_topology,
        )
        if units:
            policy = PersonalDetailTransitionPolicy()
            self.entity_context = decode_credit_report_units(
                units,
                report_family="personal_detail",
                furniture_unit_ids=furniture,
                transition_scorer=policy.score,
                entity_prefix="personal_detail_canonical",
            )
            self.evidence_unit_ids = MappingProxyType(dict(evidence_units))
            self.source_page_by_logical.clear()
            self.source_page_by_logical.update(source_pages)
            self.reading_order_by_logical.clear()
            self.reading_order_by_logical.update(reading_order)
        self._canonical_entity_context_ready = True

    def _build_corrected_evidence_pages(self) -> list[dict[str, Any]]:
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
                        local.get("page_width") or bundle.get("page_width") or local.get("width") or bundle.get("width")
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
        corrected = self._ocr_correction_overlay.corrected_evidence_pages(
            sorted(
                pages,
                key=lambda item: (
                    self.reading_order_by_logical.get(item["page"], item["page"]),
                    item["page"],
                ),
            )
        )
        return self._merge_split_replay_pages(corrected)

    def _merge_split_replay_pages(
        self,
        pages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Overlay actual splitter slices on core pages that remained unsplit."""
        audit = self.page_topology.audit()
        source_candidates = {
            int(source)
            for source, logicals in (audit.get("logical_pages_by_source") or {}).items()
            if len(logicals) == 1
        }
        if not source_candidates:
            return pages
        supplemental = self.supplemental_page_ocr_evidence(
            source_candidates,
            reason="split_result_topology_replay",
        )
        if not supplemental:
            return pages

        base_by_source: dict[int, list[dict[str, Any]]] = {}
        for page in pages:
            base_by_source.setdefault(int(page.get("source_page") or 0), []).append(page)
        supplemental_by_source: dict[int, list[dict[str, Any]]] = {}
        for page in supplemental:
            supplemental_by_source.setdefault(int(page.get("source_page") or 0), []).append(page)

        next_logical = max(self.source_page_by_logical, default=0) + 1
        merged: list[dict[str, Any]] = []
        for source in sorted(set(base_by_source) | set(supplemental_by_source)):
            base_pages = base_by_source.get(source, [])
            replay_pages = sorted(
                supplemental_by_source.get(source, []),
                key=lambda item: int(item.get("segment_index") or 0),
            )
            if not replay_pages:
                merged.extend(base_pages)
                continue
            logicals = self.page_topology.logicals_for_source(source)
            geometries = {
                int(geometry.segment_index): logical
                for logical in logicals
                if (geometry := self.page_topology.geometry(logical)) is not None
                and geometry.split_kind == "two_page_spread"
                and geometry.segment_index in {0, 1}
            }
            base_by_segment: dict[int, dict[str, Any]] = {}
            for page in base_pages:
                geometry = self.page_topology.geometry(int(page.get("page") or 0))
                if (
                    geometry is not None
                    and geometry.split_kind == "two_page_spread"
                    and geometry.segment_index in {0, 1}
                ):
                    base_by_segment[int(geometry.segment_index)] = page

            replay_by_segment = {int(page.get("segment_index") or 0): page for page in replay_pages}
            available_segments = set(base_by_segment) | set(replay_by_segment)
            if available_segments != {0, 1}:
                # Never replace an unsplit/core page with only one half of a
                # splitter-confirmed spread. Preserve the original evidence
                # and publish an explicit page-continuation uncertainty.
                merged.extend(base_pages)
                incomplete_id = f"source:{source}:split-pair"
                if not any(
                    str(item.get("supplemental_page_id") or "") == incomplete_id
                    and item.get("status") == "pair_incomplete"
                    for item in self._page_ocr_requests
                ):
                    self._page_ocr_requests.append(
                        {
                            "supplemental_page_id": incomplete_id,
                            "source_page": source,
                            "expected_segments": [0, 1],
                            "observed_segments": sorted(available_segments),
                            "reason": "split_result_topology_replay",
                            "status": "pair_incomplete",
                        }
                    )
                continue
            unsplit_logical = logicals[0] if len(logicals) == 1 and not geometries else 0
            for segment in (0, 1):
                if segment in base_by_segment:
                    merged.append(base_by_segment[segment])
                    continue
                replay = replay_by_segment.get(segment)
                if replay is None:
                    continue
                logical = geometries.get(segment, 0)
                if not logical and unsplit_logical and segment == 0:
                    logical = unsplit_logical
                if not logical:
                    logical = next_logical
                    next_logical += 1
                replay = dict(replay)
                replay["page"] = logical
                replay["logical_page"] = logical
                replay["plugin_replayed_subpage"] = True
                replay["lines"] = self._ocr_correction_overlay.corrected_evidence_pages([replay])[0]["lines"]
                self.source_page_by_logical[logical] = source
                merged.append(replay)

        order_keys: dict[int, tuple[int, int, int]] = {}
        for logical, source in self.source_page_by_logical.items():
            source_logicals = self.page_topology.logicals_for_source(source)
            source_order = min(
                (self.reading_order_by_logical.get(item, item) for item in source_logicals),
                default=self.reading_order_by_logical.get(logical, logical),
            )
            geometry = self.page_topology.geometry(logical)
            segment = int(geometry.segment_index) if geometry and geometry.segment_index in {0, 1} else 0
            replay = next((item for item in merged if int(item.get("page") or 0) == logical), None)
            if replay and replay.get("plugin_replayed_subpage"):
                segment = int(replay.get("segment_index") or 0)
            order_keys[logical] = (source_order, segment, logical)
        self.reading_order_by_logical.clear()
        self.reading_order_by_logical.update(
            {logical: index for index, logical in enumerate(sorted(order_keys, key=order_keys.get), start=1)}
        )
        return sorted(
            merged,
            key=lambda item: (
                self.reading_order_by_logical.get(int(item.get("page") or 0), int(item.get("page") or 0)),
                int(item.get("segment_index") or 0),
            ),
        )

    def full_page_ocr_evidence(
        self,
        logical_pages: Iterable[int],
        *,
        reason: str,
    ) -> list[dict[str, Any]]:
        """Re-OCR complete logical pages as bounded supplemental evidence.

        This path avoids reliance on a possibly incorrect cell crop.  Results
        retain logical-page coordinates and never replace sealed OCR evidence.
        """
        if os.environ.get("DOCMIRROR_PERSONAL_DETAIL_PAGE_OCR", "1") == "0":
            return []
        requested = sorted({int(page) for page in logical_pages if int(page) > 0})
        canonical_registration = str(reason).startswith("canonical_template_registration")
        output: list[dict[str, Any]] = []
        for logical in requested:
            if logical in self._page_ocr_cache:
                cached = self._page_ocr_cache[logical]
                if cached:
                    output.append(deepcopy(cached))
                continue
            # Canonical registration is not an optional field probe.  Every
            # unregistered page receives one complete-page retry even when an
            # earlier field-repair budget has been consumed.
            if not canonical_registration and len(self._page_ocr_requests) >= self._page_ocr_max_requests:
                break
            request = {"logical_page": logical, "reason": str(reason), "status": "requested"}
            self._page_ocr_requests.append(request)
            rendered = self._page_image_resolver(logical)
            if not rendered:
                request["status"] = "render_failed"
                self._page_ocr_cache[logical] = None
                continue
            image = rendered.get("image")
            shape = getattr(image, "shape", None)
            if not shape or len(shape) < 2 or not shape[0] or not shape[1]:
                request["status"] = "invalid_image"
                self._page_ocr_cache[logical] = None
                continue
            words, selected_image, selected_rotation, deskew_angle, orientation_score = _complete_page_ocr(image)
            selected_shape = getattr(selected_image, "shape", shape)
            zoom = float(rendered.get("zoom") or 1.0)
            page_width = float(selected_shape[1]) / zoom
            page_height = float(selected_shape[0]) / zoom
            scale_x = page_width / float(shape[1])
            scale_y = page_height / float(shape[0])
            if selected_rotation in {90, 270}:
                scale_x = page_width / float(selected_shape[1])
                scale_y = page_height / float(selected_shape[0])
            lines: list[dict[str, Any]] = []
            for index, word in enumerate(words):
                text = str(word.get("text") or "").strip()
                bbox = word.get("bbox")
                confidence = float(word.get("confidence") or 0.0)
                if not text or confidence < 0.45 or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                    continue
                lines.append(
                    {
                        "text": text,
                        "confidence": confidence,
                        "bbox": [
                            float(bbox[0]) * scale_x,
                            float(bbox[1]) * scale_y,
                            float(bbox[2]) * scale_x,
                            float(bbox[3]) * scale_y,
                        ],
                        "evidence_ids": [f"personal_detail_full_page_ocr:p{logical}:w{index}"],
                        "source": "personal_detail_full_page_ocr",
                    }
                )
            if not lines:
                request["status"] = "ocr_empty"
                self._page_ocr_cache[logical] = None
                continue
            request["status"] = "completed"
            request["word_count"] = len(lines)
            request["selected_rotation"] = selected_rotation
            request["deskew_angle"] = deskew_angle
            request["orientation_score"] = orientation_score
            page = {
                "page": logical,
                "source_page": int(rendered.get("source_page") or self.source_page_by_logical.get(logical, logical)),
                "page_width": page_width,
                "page_height": page_height,
                "reason": str(reason),
                "selected_rotation": selected_rotation,
                "deskew_angle": deskew_angle,
                "orientation_score": orientation_score,
                "lines": sorted(lines, key=lambda line: (line["bbox"][1], line["bbox"][0])),
            }
            # Keep the replay raw. Typed correction happens only after a
            # structural parser assigns the page token to a schema role. This
            # also prevents recursive page repair while producing the replay.
            self._page_ocr_cache[logical] = deepcopy(page)
            output.append(page)
        return output

    def supplemental_page_ocr_evidence(
        self,
        source_pages: Iterable[int],
        *,
        reason: str,
    ) -> list[dict[str, Any]]:
        """OCR splitter-confirmed supplemental logical subpages.

        Footer text may corroborate printed order but does not decide whether
        the physical source is subpaged.
        """
        slices = self._page_image_resolver.supplemental_spread_slices(source_pages)
        output: list[dict[str, Any]] = []
        requested_sources = {
            int(item.get("source_page") or 0)
            for item in self._page_ocr_requests
            if item.get("supplemental_page_id") and item.get("status") != "budget_exhausted"
        }
        budget_exhausted_sources: set[int] = set()
        for recovered in slices:
            supplemental_id = str(recovered.get("supplemental_page_id") or "")
            if not supplemental_id:
                continue
            source_page = int(recovered.get("source_page") or 0)
            if source_page in budget_exhausted_sources:
                continue
            if supplemental_id in self._supplemental_ocr_cache:
                output.append(deepcopy(self._supplemental_ocr_cache[supplemental_id]))
                continue
            # Splitter-confirmed subpages are part of the corrected document
            # topology, not optional anomaly probes. Count their budget
            # independently so earlier field-repair OCR cannot silently drop
            # a later half of the document. A physical spread is one atomic
            # budget unit, so both of its splitter slices are always attempted.
            if source_page not in requested_sources and len(requested_sources) >= self._page_ocr_max_requests:
                budget_exhausted_sources.add(source_page)
                self._page_ocr_requests.append(
                    {
                        "supplemental_page_id": f"source:{source_page}:split-pair",
                        "source_page": source_page,
                        "expected_segments": [0, 1],
                        "reason": str(reason),
                        "status": "budget_exhausted",
                    }
                )
                continue
            requested_sources.add(source_page)
            request = {
                "supplemental_page_id": supplemental_id,
                "source_page": source_page,
                "printed_page": int(recovered.get("printed_page") or 0),
                "segment_index": int(recovered.get("segment_index") or 0),
                "subpage_basis": str(recovered.get("subpage_basis") or "core_split_result"),
                "reason": str(reason),
                "status": "requested",
            }
            self._page_ocr_requests.append(request)
            image = recovered.get("image")
            shape = getattr(image, "shape", None)
            if not shape or len(shape) < 2 or not shape[0] or not shape[1]:
                request["status"] = "invalid_image"
                continue
            words, selected_image, page_rotation, deskew_angle, orientation_score = _complete_page_ocr(image)
            selected_shape = getattr(selected_image, "shape", shape)
            zoom = float(recovered.get("zoom") or 1.0)
            page_width = float(selected_shape[1]) / zoom
            page_height = float(selected_shape[0]) / zoom
            scale_x = page_width / float(selected_shape[1])
            scale_y = page_height / float(selected_shape[0])
            lines: list[dict[str, Any]] = []
            for index, word in enumerate(words):
                text = str(word.get("text") or "").strip()
                bbox = word.get("bbox")
                confidence = float(word.get("confidence") or 0.0)
                if not text or confidence < 0.45 or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                    continue
                lines.append(
                    {
                        "text": text,
                        "confidence": confidence,
                        "bbox": [
                            float(bbox[0]) * scale_x,
                            float(bbox[1]) * scale_y,
                            float(bbox[2]) * scale_x,
                            float(bbox[3]) * scale_y,
                        ],
                        "evidence_ids": [f"personal_detail_supplemental_ocr:{supplemental_id}:w{index}"],
                        "source": "personal_detail_supplemental_page_ocr",
                    }
                )
            if not lines:
                request["status"] = "ocr_empty"
                continue
            page = {
                "supplemental_page_id": supplemental_id,
                "printed_page": int(recovered.get("printed_page") or 0),
                "printed_total": int(recovered.get("printed_total") or 0),
                "source_page": int(recovered.get("source_page") or 0),
                "segment_index": int(recovered.get("segment_index") or 0),
                "selected_rotation": int(recovered.get("selected_rotation") or 0),
                "page_rotation": page_rotation,
                "deskew_angle": deskew_angle,
                "orientation_score": orientation_score,
                "split_confidence": float(recovered.get("split_confidence") or 0.0),
                "split_ratio": float(recovered.get("split_ratio") or 0.5),
                "split_consensus_boost": float(recovered.get("split_consensus_boost") or 0.0),
                "subpage_basis": str(recovered.get("subpage_basis") or "core_split_result"),
                "page_width": page_width,
                "page_height": page_height,
                "reason": str(reason),
                "lines": sorted(lines, key=lambda line: (line["bbox"][1], line["bbox"][0])),
            }
            request["status"] = "completed"
            request["word_count"] = len(lines)
            request["page_rotation"] = page_rotation
            request["deskew_angle"] = deskew_angle
            request["orientation_score"] = orientation_score
            self._supplemental_ocr_cache[supplemental_id] = deepcopy(page)
            output.append(page)
        return output

    def ocr_correction_audit(self) -> dict[str, Any]:
        """Return a detached audit snapshot for diagnostics and regression tests."""
        return {
            **deepcopy(self._ocr_correction_overlay.audit()),
            "full_page_ocr_requests": deepcopy(self._page_ocr_requests),
            "full_page_ocr_request_count": len(self._page_ocr_requests),
        }

    def page_topology_audit(self) -> dict[str, Any]:
        """Return the plugin's detached logical-page validation result."""
        return deepcopy(self._page_image_resolver.audit())

    def canonical_layout_audit(self) -> dict[str, Any]:
        """Return the detached template-registration and fragment audit."""
        return deepcopy(self._canonical_layout_projection().audit())

    def tables_continue(self, left_table_id: str, right_table_id: str) -> bool | None:
        left_unit_id = self.entity_context.table_unit_id(left_table_id)
        right_unit_id = self.entity_context.table_unit_id(right_table_id)
        if not left_unit_id or not right_unit_id:
            return None
        left = self.entity_context.entity_for_unit(left_unit_id)
        right = self.entity_context.entity_for_unit(right_unit_id)
        return bool(left is not None and right is not None and left.entity_id == right.entity_id)

    def pages_adjacent_in_reading_order(self, left_page: int, right_page: int) -> bool:
        left_order = self.reading_order_by_logical.get(int(left_page), int(left_page))
        right_order = self.reading_order_by_logical.get(int(right_page), int(right_page))
        return right_order == left_order + 1

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
            left_box = _bbox(left_line)
            right_box = _bbox(right_line)
            left_geometry = self.page_topology.geometry(left_page)
            right_geometry = self.page_topology.geometry(right_page)
            if (
                left_box
                and right_box
                and left_geometry
                and right_geometry
                and left_geometry.height > 0
                and right_geometry.height > 0
                and left_box[3] / left_geometry.height >= 0.75
                and right_box[1] / right_geometry.height <= 0.25
                and self.pages_adjacent_in_reading_order(left_page, right_page)
            ):
                return True
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
