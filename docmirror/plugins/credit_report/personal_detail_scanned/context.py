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
from types import MappingProxyType, SimpleNamespace
from typing import Any, Callable, Iterable, Mapping, TypeVar, cast

from docmirror.plugins.credit_report.personal_detail_scanned.page_reocr import (
    OneShotPageReOCRRegistry,
)
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
_PRINTED_PAGE_ONLY_RE = re.compile(
    r"^\s*第\s*(?P<page>\d{1,3})\s*页\s*[,，。.]?\s*$"
)
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


def _single_page_ocr(image: Any) -> tuple[list[dict[str, Any]], float]:
    """OCR one already-oriented, frozen logical subpage exactly once."""

    from docmirror.ocr.repair.recognizers import rapidocr_recognize

    words = rapidocr_recognize(image, source="personal_detail_page_reocr_once")
    return words, _page_ocr_score(words, image_shape=getattr(image, "shape", None))


def _bbox(value: Any) -> tuple[float, float, float, float] | None:
    raw = value.get("bbox") if isinstance(value, dict) else getattr(value, "bbox", None)
    if not isinstance(raw, (list, tuple)) or len(raw) < 4:
        return None
    x0, y0, x1, y1 = (_finite(item) for item in raw[:4])
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def _bottom_furniture_geometry(value: Any, *, page_height: Any) -> bool:
    """Require one exact local bbox in the page's narrow bottom band."""

    box = _bbox(value)
    height = _finite(page_height)
    if box is None or height <= 0.0:
        return False
    tolerance = max(2.0, height * 0.01)
    return bool(
        box[1] >= height * 0.85
        and box[3] >= height * 0.90
        and box[3] <= height + tolerance
        and box[3] - box[1] <= height * 0.08
    )


def _authoritative_reading_order(resolution: Any) -> bool:
    return bool(
        isinstance(resolution, Mapping)
        and resolution.get("resolved") is True
        and resolution.get("authoritative") is True
    )


def _matrix3(value: Any) -> list[list[float]] | None:
    if not (
        isinstance(value, (list, tuple))
        and len(value) == 3
        and all(isinstance(row, (list, tuple)) and len(row) == 3 for row in value)
    ):
        return None
    return [[float(item) for item in row] for row in value]


def _transform_bbox(
    matrix: list[list[float]],
    bbox: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = bbox
    points = []
    for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
        points.append(
            (
                matrix[0][0] * x + matrix[0][1] * y + matrix[0][2],
                matrix[1][0] * x + matrix[1][1] * y + matrix[1][2],
            )
        )
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def _overlap_fraction(
    bbox: tuple[float, float, float, float],
    crop: tuple[float, float, float, float],
) -> float:
    intersection = max(0.0, min(bbox[2], crop[2]) - max(bbox[0], crop[0])) * max(
        0.0, min(bbox[3], crop[3]) - max(bbox[1], crop[1])
    )
    area = max(1e-6, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    return intersection / area


def _slice_assignment(
    source_bbox: tuple[float, float, float, float],
    recovered_by_segment: Mapping[int, Mapping[str, Any]],
) -> tuple[int | None, bool]:
    ranked = sorted(
        (
            (
                _overlap_fraction(
                    source_bbox,
                    cast(
                        tuple[float, float, float, float],
                        tuple(float(value) for value in recovered["source_crop_bbox"]),
                    ),
                ),
                segment,
            )
            for segment, recovered in recovered_by_segment.items()
        ),
        reverse=True,
    )
    if not ranked or ranked[0][0] < 0.55:
        return None, True
    second = ranked[1][0] if len(ranked) > 1 else 0.0
    ambiguous = second >= 0.20 and ranked[0][0] - second < 0.35
    return (None, True) if ambiguous else (ranked[0][1], False)


def _project_table_for_static_slice(
    table: Any,
    *,
    base_to_source: list[list[float]],
    source_to_slice: list[list[float]],
) -> Any:
    def project(raw: Any) -> list[float] | Any:
        box = _bbox({"bbox": raw})
        if box is None:
            return raw
        source_box = _transform_bbox(base_to_source, box)
        return list(_transform_bbox(source_to_slice, source_box))

    metadata = deepcopy(dict(getattr(table, "metadata", None) or {}))
    cell_boxes = metadata.get("cell_bboxes")
    if isinstance(cell_boxes, list):
        metadata["cell_bboxes"] = [
            [project(box) for box in row] if isinstance(row, list) else row
            for row in cell_boxes
        ]
    raw_rows = _raw_rows(table)
    if raw_rows:
        metadata["raw_rows"] = [list(row) for row in raw_rows]
    table_box = _bbox(table)
    return SimpleNamespace(
        table_id=str(getattr(table, "table_id", "") or ""),
        metadata=metadata,
        headers=[],
        rows=[],
        bbox=project(table_box) if table_box is not None else None,
        confidence=getattr(table, "confidence", None),
    )


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


def _printed_reading_order_resolution(
    parse_result: Any,
    topology: PersonalDetailPageTopology | None = None,
) -> tuple[dict[int, int], dict[str, Any]]:
    """Resolve printed reading order without disguising fallback as proof.

    Detailed reports are commonly scanned as two-page spreads. Physical
    sheets can be out of order even though each half retains the report's own
    ``第 N 页，共 M 页`` footer. Provenance page numbers remain unchanged; only
    continuation and evidence traversal use this order.

    A page-only ``第 N 页`` marker is accepted only after other complete
    footers establish one coherent document total. Its other spread half may
    then be inferred from frozen topology. Every observed logical page must
    ultimately own one unique, in-range printed number. A trailing scan half
    with no sealed text or table content may follow that complete permutation;
    it is retained after the report pages but never assigned a manufactured
    printed number. Failure returns sealed identity order together with explicit
    unresolved provenance so downstream ownership code cannot mistake the
    fallback for an authoritative mapping.
    """
    text_evidence_by_page: dict[int, list[tuple[str, Any]]] = {}
    page_heights_by_logical: dict[int, list[float]] = {}
    observed_pages: set[int] = set()
    source_by_logical: dict[int, int] = {}
    native_pages_by_logical: dict[int, list[Any]] = {}
    evidence_bundles_by_logical: dict[int, list[Mapping[str, Any]]] = {}

    for page_index, page in enumerate(getattr(parse_result, "pages", None) or [], start=1):
        logical = int(getattr(page, "page_number", 0) or page_index)
        observed_pages.add(logical)
        native_pages_by_logical.setdefault(logical, []).append(page)
        transform = dict(getattr(page, "coordinate_transform", None) or {})
        source_by_logical[logical] = int(
            transform.get("source_page_number") or getattr(page, "source_page_number", 0) or logical
        )
        page_height = _finite(getattr(page, "height", 0))
        if page_height > 0.0:
            page_heights_by_logical.setdefault(logical, []).append(page_height)
        text_evidence_by_page.setdefault(logical, []).extend(
            (str(getattr(block, "content", "") or ""), block)
            for block in getattr(page, "texts", None) or []
        )

    for bundle in _domain_specific(parse_result).get("_page_evidence_bundles") or []:
        if not isinstance(bundle, dict):
            continue
        local = bundle.get("local_structure_evidence")
        logical = int(
            bundle.get("page")
            or (local.get("page") if isinstance(local, Mapping) else 0)
            or 0
        )
        if logical > 0:
            evidence_bundles_by_logical.setdefault(logical, []).append(bundle)
        if not isinstance(local, dict):
            continue
        if logical <= 0:
            continue
        observed_pages.add(logical)
        source_by_logical.setdefault(
            logical,
            int(bundle.get("source_page_number") or local.get("source_page") or logical),
        )
        page_height = _finite(
            local.get("page_height")
            or bundle.get("page_height")
            or local.get("height")
            or bundle.get("height")
        )
        if page_height > 0.0:
            page_heights_by_logical.setdefault(logical, []).append(page_height)
        text_evidence_by_page.setdefault(logical, []).extend(
            (str(line.get("text") or line.get("content") or ""), line)
            for line in local.get("lines") or []
            if isinstance(line, dict)
        )

    identity = {page: page for page in sorted(observed_pages)}

    def table_source_content_present(table: Any) -> bool:
        """Check every sealed table representation, not only its preferred rows."""

        def member(owner: Any, name: str, default: Any = None) -> Any:
            return (
                owner.get(name, default)
                if isinstance(owner, Mapping)
                else getattr(owner, name, default)
            )

        def scalar_present(value: Any) -> bool:
            if value is None:
                return False
            if isinstance(value, str):
                return bool(_compact(value))
            return True

        metadata = member(table, "metadata")
        raw_row_representations = [member(table, "raw_rows")]
        if isinstance(metadata, Mapping):
            raw_row_representations.append(metadata.get("raw_rows"))
        for raw_rows in raw_row_representations:
            if raw_rows in (None, [], ()):
                continue
            if not isinstance(raw_rows, (list, tuple)):
                return True
            for raw_row in raw_rows:
                if not isinstance(raw_row, (list, tuple)):
                    return True
                if any(scalar_present(cell) for cell in raw_row):
                    return True

        if any(scalar_present(header) for header in member(table, "headers", ()) or ()):
            return True
        if scalar_present(member(table, "caption")):
            return True
        for row_collection_name in ("rows", "row_models", "logical_rows"):
            rows = member(table, row_collection_name)
            if rows in (None, [], ()):
                continue
            if not isinstance(rows, (list, tuple)):
                return True
            for row in rows:
                cells = member(row, "cells")
                if cells is None and isinstance(row, (list, tuple)):
                    cells = row
                if cells is None or not isinstance(cells, (list, tuple)):
                    return True
                for cell in cells:
                    if isinstance(cell, Mapping):
                        values = (
                            cell.get("text"),
                            cell.get("cleaned"),
                            cell.get("numeric"),
                            cell.get("value"),
                            cell.get("content"),
                        )
                    elif isinstance(cell, (str, int, float, bool)) or cell is None:
                        values = (cell,)
                    else:
                        values = (
                            getattr(cell, "text", None),
                            getattr(cell, "cleaned", None),
                            getattr(cell, "numeric", None),
                        )
                    if any(scalar_present(value) for value in values):
                        return True
        return False

    def source_evidence_empty(logical: int) -> bool:
        """Prove that one registered logical page has no sealed source content."""

        native_pages = native_pages_by_logical.get(logical) or []
        if len(native_pages) != 1:
            return False
        if any(_compact(text) for text, _evidence in text_evidence_by_page.get(logical, ())):
            return False
        page = native_pages[0]
        if any(
            _compact(
                pair.get("key")
                if isinstance(pair, Mapping)
                else getattr(pair, "key", "")
            )
            or _compact(
                pair.get("value")
                if isinstance(pair, Mapping)
                else getattr(pair, "value", "")
            )
            for pair in getattr(page, "key_values", None) or []
        ):
            return False
        if any(
            _compact(
                (line.get("content") or line.get("text"))
                if isinstance(line, Mapping)
                else getattr(line, "content", getattr(line, "text", ""))
            )
            for line in getattr(page, "lines", None) or []
        ):
            return False
        if any(
            table_source_content_present(table)
            for table in getattr(page, "tables", None) or []
        ):
            return False
        sealed_bundle_content_keys = (
            "lines",
            "tokens",
            "candidates",
            "structures",
            "micro_grid_structures",
            "source_table_geometry",
            "tables",
            "table_rows",
            "rows",
            "cells",
        )
        for bundle in evidence_bundles_by_logical.get(logical, ()):
            owners = (
                bundle,
                bundle.get("local_structure_evidence"),
                bundle.get("micro_grid_evidence"),
            )
            if any(
                isinstance(owner, Mapping) and any(owner.get(key) for key in sealed_bundle_content_keys)
                for owner in owners
            ):
                return False
            region_detect = bundle.get("region_detect")
            if isinstance(region_detect, Mapping) and region_detect.get(
                "region_detect_candidates"
            ):
                return False
            morphology_summary = bundle.get("morphology_summary")
            if isinstance(morphology_summary, Mapping) and any(
                value not in (None, "", 0, 0.0, False)
                for value in morphology_summary.values()
            ):
                return False
        return True

    def unresolved(
        reason: str,
        *,
        printed_by_logical: Mapping[int, int] | None = None,
        expected_total: int | None = None,
        full_footer_pages: Iterable[int] = (),
        page_only_footer_pages: Iterable[int] = (),
        paired_inferred_pages: Iterable[int] = (),
        blank_logical_pages: Iterable[int] = (),
    ) -> tuple[dict[int, int], dict[str, Any]]:
        observed = sorted(observed_pages)
        printed = {
            int(logical): int(page)
            for logical, page in (printed_by_logical or {}).items()
        }
        duplicate_printed_pages = sorted(
            page
            for page, count in Counter(printed.values()).items()
            if count > 1
        )
        return identity, {
            "resolved": False,
            "authoritative": False,
            "basis": "unresolved_identity_fallback",
            "reason": reason,
            "observed_logical_pages": observed,
            "identity_fallback": True,
            "printed_page_by_logical": printed,
            "unresolved_logical_pages": sorted(set(observed) - set(printed)),
            "duplicate_printed_pages": duplicate_printed_pages,
            "full_footer_logical_pages": sorted(set(full_footer_pages)),
            "page_only_footer_logical_pages": sorted(set(page_only_footer_pages)),
            "paired_inferred_logical_pages": sorted(set(paired_inferred_pages)),
            "blank_logical_pages": sorted(set(blank_logical_pages)),
            **({"printed_total": expected_total} if expected_total is not None else {}),
        }

    if len(observed_pages) < 2:
        return identity, {
            "resolved": True,
            "authoritative": True,
            "basis": "single_or_empty_page",
            "reason": "cross_page_order_not_required",
            "observed_logical_pages": sorted(observed_pages),
            "identity_fallback": False,
            "printed_page_by_logical": dict(identity),
            "unresolved_logical_pages": [],
            "duplicate_printed_pages": [],
            "full_footer_logical_pages": [],
            "page_only_footer_logical_pages": [],
            "paired_inferred_logical_pages": [],
            "blank_logical_pages": [],
        }

    printed_by_logical: dict[int, int] = {}
    totals: list[int] = []
    full_footer_pages: set[int] = set()
    ambiguous_full_footer_pages: set[int] = set()
    for logical in sorted(observed_pages):
        page_height = max(page_heights_by_logical.get(logical) or (0.0,))
        exact_matches = {
            (int(match.group("page")), int(match.group("total")))
            for text, evidence in text_evidence_by_page.get(logical, ())
            if _bottom_furniture_geometry(evidence, page_height=page_height)
            for match in _PRINTED_PAGE_RE.finditer(text)
        }
        if len(exact_matches) == 1:
            printed, total = next(iter(exact_matches))
            if 1 <= printed <= total:
                printed_by_logical[logical] = printed
                totals.append(total)
                full_footer_pages.add(logical)
        elif len(exact_matches) > 1:
            ambiguous_full_footer_pages.add(logical)

    if ambiguous_full_footer_pages:
        return unresolved(
            "ambiguous_full_footer",
            printed_by_logical=printed_by_logical,
            full_footer_pages=full_footer_pages,
        )

    observed_totals = set(totals)
    if len(observed_totals) != 1 or next(iter(observed_totals), 0) <= 0:
        return unresolved(
            "printed_total_missing_or_conflicting",
            printed_by_logical=printed_by_logical,
            full_footer_pages=full_footer_pages,
        )
    expected_total = next(iter(observed_totals))

    page_only_footer_pages: set[int] = set()
    ambiguous_page_only_pages: set[int] = set()
    for logical in sorted(observed_pages - set(printed_by_logical)):
        page_height = max(page_heights_by_logical.get(logical) or (0.0,))
        page_only_matches = {
            int(match.group("page"))
            for text, evidence in text_evidence_by_page.get(logical, ())
            if _bottom_furniture_geometry(evidence, page_height=page_height)
            if (match := _PRINTED_PAGE_ONLY_RE.fullmatch(text)) is not None
            and 1 <= int(match.group("page")) <= expected_total
        }
        if len(page_only_matches) == 1:
            printed_by_logical[logical] = next(iter(page_only_matches))
            page_only_footer_pages.add(logical)
        elif len(page_only_matches) > 1:
            ambiguous_page_only_pages.add(logical)

    if ambiguous_page_only_pages:
        return unresolved(
            "ambiguous_page_only_footer",
            printed_by_logical=printed_by_logical,
            expected_total=expected_total,
            full_footer_pages=full_footer_pages,
            page_only_footer_pages=page_only_footer_pages,
        )

    # A sparse sealed excerpt (printed total larger than its observed page
    # population) cannot prove an extra scan half. Its empty synthetic/partial
    # siblings remain eligible for bounded footer inference. Blank-tail
    # certification is available only when the observed population could
    # already contain the whole numbered document.
    blank_logical_pages = (
        {
            logical
            for logical in observed_pages - set(printed_by_logical)
            if source_evidence_empty(logical)
        }
        if expected_total <= len(observed_pages)
        else set()
    )
    paired_inferred_pages = _infer_paired_printed_pages(
        printed_by_logical,
        source_by_logical,
        topology=topology,
        expected_total=expected_total,
        excluded_logical_pages=blank_logical_pages,
    )
    unprinted_nonblank_pages = (
        observed_pages - set(printed_by_logical) - blank_logical_pages
    )
    if unprinted_nonblank_pages:
        return unresolved(
            "logical_page_footer_unresolved",
            printed_by_logical=printed_by_logical,
            expected_total=expected_total,
            full_footer_pages=full_footer_pages,
            page_only_footer_pages=page_only_footer_pages,
            paired_inferred_pages=paired_inferred_pages,
            blank_logical_pages=blank_logical_pages,
        )
    if len(set(printed_by_logical.values())) != len(printed_by_logical):
        return unresolved(
            "printed_page_nonunique",
            printed_by_logical=printed_by_logical,
            expected_total=expected_total,
            full_footer_pages=full_footer_pages,
            page_only_footer_pages=page_only_footer_pages,
            paired_inferred_pages=paired_inferred_pages,
            blank_logical_pages=blank_logical_pages,
        )

    printed_pages = set(printed_by_logical.values())
    if max(printed_pages, default=0) > expected_total or min(printed_pages, default=1) < 1:
        return unresolved(
            "printed_page_out_of_range",
            printed_by_logical=printed_by_logical,
            expected_total=expected_total,
            full_footer_pages=full_footer_pages,
            page_only_footer_pages=page_only_footer_pages,
            paired_inferred_pages=paired_inferred_pages,
            blank_logical_pages=blank_logical_pages,
        )

    # Partial sealed contexts may legitimately contain a sparse, uniquely
    # numbered report excerpt.  Once an unnumbered empty scan half is accepted,
    # however, require the numbered side to be the complete 1..M document so a
    # missing business page can never be disguised as that blank tail.
    if blank_logical_pages and printed_pages != set(range(1, expected_total + 1)):
        return unresolved(
            "printed_page_permutation_incomplete",
            printed_by_logical=printed_by_logical,
            expected_total=expected_total,
            full_footer_pages=full_footer_pages,
            page_only_footer_pages=page_only_footer_pages,
            paired_inferred_pages=paired_inferred_pages,
            blank_logical_pages=blank_logical_pages,
        )

    ordered_report_pages = sorted(
        printed_by_logical,
        key=lambda page: (printed_by_logical[page], page),
    )
    ordered_logical_pages = ordered_report_pages + sorted(blank_logical_pages)
    order = {
        logical: index
        for index, logical in enumerate(ordered_logical_pages, start=1)
    }
    has_blank_tail = bool(blank_logical_pages)
    return order, {
        "resolved": True,
        "authoritative": True,
        "basis": (
            "complete_unique_printed_page_permutation_with_blank_tail"
            if has_blank_tail
            else "complete_unique_printed_page_permutation"
        ),
        "reason": (
            "full_page_total_bounded_pair_resolution_and_source_empty_tail"
            if has_blank_tail
            else "full_page_total_and_bounded_pair_resolution"
        ),
        "observed_logical_pages": sorted(observed_pages),
        "identity_fallback": False,
        "printed_page_by_logical": dict(sorted(printed_by_logical.items())),
        "unresolved_logical_pages": [],
        "duplicate_printed_pages": [],
        "full_footer_logical_pages": sorted(full_footer_pages),
        "page_only_footer_logical_pages": sorted(page_only_footer_pages),
        "paired_inferred_logical_pages": sorted(paired_inferred_pages),
        "blank_logical_pages": sorted(blank_logical_pages),
        "printed_total": expected_total,
    }


def _printed_reading_order(
    parse_result: Any,
    topology: PersonalDetailPageTopology | None = None,
) -> dict[int, int]:
    """Compatibility view of the authoritative order or sealed fallback."""

    return _printed_reading_order_resolution(parse_result, topology)[0]


def _infer_paired_printed_pages(
    printed_by_logical: dict[int, int],
    source_by_logical: dict[int, int],
    *,
    topology: PersonalDetailPageTopology | None = None,
    expected_total: int | None = None,
    excluded_logical_pages: Iterable[int] = (),
) -> set[int]:
    """Infer one unread footer from a geometry-confirmed adjacent half."""
    inferred: set[int] = set()
    excluded = {int(value) for value in excluded_logical_pages}
    logicals_by_source: dict[int, list[int]] = {}
    for logical, source in source_by_logical.items():
        logicals_by_source.setdefault(source, []).append(logical)
    for logicals in logicals_by_source.values():
        ordered = topology.ordered_pair(logicals) if topology is not None else None
        if ordered is None:
            continue
        left, right = ordered
        if left in excluded or right in excluded:
            continue
        if left in printed_by_logical and right not in printed_by_logical:
            candidate = printed_by_logical[left] + 1
            if expected_total is None or 1 <= candidate <= expected_total:
                printed_by_logical[right] = candidate
                inferred.add(right)
        elif right in printed_by_logical and left not in printed_by_logical:
            candidate = printed_by_logical[right] - 1
            if expected_total is None or 1 <= candidate <= expected_total:
                printed_by_logical[left] = candidate
                inferred.add(left)
    return inferred


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
    registered_reading_order: Mapping[int, int] | None = None,
    registered_reading_order_resolution: Mapping[str, Any] | None = None,
) -> tuple[
    tuple[CreditReportUnit, ...],
    tuple[str, ...],
    dict[str, str],
    dict[int, int],
    dict[int, int],
    dict[str, Any],
]:
    candidates: list[CreditReportUnit] = []
    furniture: set[str] = set()
    evidence_units: dict[str, str] = {}
    source_pages: dict[int, int] = {}
    pages = list(getattr(parse_result, "pages", None) or [])
    table_owners: dict[int, list[tuple[tuple[float, float, float, float], str]]] = {}
    text_owners: dict[int, list[tuple[str, tuple[float, float, float, float] | None, str]]] = {}
    edge_occurrences: dict[str, list[tuple[int, str]]] = {}
    if registered_reading_order is None:
        reading_order, reading_order_resolution = _printed_reading_order_resolution(
            parse_result,
            topology,
        )
    else:
        reading_order = {
            int(logical): int(position)
            for logical, position in registered_reading_order.items()
        }
        reading_order_resolution = deepcopy(
            dict(registered_reading_order_resolution or {})
        )

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
    return (
        tuple(active),
        tuple(sorted(furniture)),
        evidence_units,
        source_pages,
        reading_order,
        reading_order_resolution,
    )


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
        reading_order_resolution: Mapping[str, Any],
        page_topology: PersonalDetailPageTopology,
    ) -> None:
        self.parse_result = parse_result
        self.entity_context = entity_context
        self.evidence_unit_ids = MappingProxyType(dict(evidence_unit_ids))
        # Plugin-owned logical subpages are added during static topology
        # construction, so these two ledgers intentionally remain mutable while
        # the sealed ParseResult and evidence IDs stay immutable.
        self.source_page_by_logical = dict(source_page_by_logical)
        self.reading_order_by_logical = dict(reading_order_by_logical)
        self.reading_order_resolution = deepcopy(dict(reading_order_resolution))
        self.page_topology = page_topology
        self._cache: dict[str, Any] = {}
        self._frozen_logical_pages: dict[int, Any] = {
            int(getattr(page, "page_number", 0) or index): page
            for index, page in enumerate(getattr(parse_result, "pages", None) or [], start=1)
        }
        self._topology_recovery_issues: list[dict[str, Any]] = []
        self._initial_personal_detail_extraction_issues = deepcopy(
            getattr(self, "_personal_detail_extraction_issues", [])
        )
        self._canonical_layout_projection_cache: Any | None = None
        self._canonical_entity_context_ready = False
        self._business_repair_plan: Any | None = None
        self._business_repair_evidence_by_page: dict[int, dict[str, Any]] = {}
        self._business_repair_active = False
        # The contract is one attempt per frozen logical page, not a document-
        # wide quota.  A fixed quota silently deprived later uncertain business
        # fields of their only field-aware correction pass on longer reports.
        self._page_ocr_max_requests = max(
            len(self._frozen_logical_pages),
            len(self.source_page_by_logical),
        )
        self._page_reocr_registry = OneShotPageReOCRRegistry(max_pages=self._page_ocr_max_requests)
        from docmirror.plugins.credit_report.personal_detail_scanned.ocr_correction import (
            PersonalDetailOCRCorrectionOverlay,
        )

        self._page_image_resolver = PersonalDetailLogicalPageImageResolver(
            parse_result,
            topology=page_topology,
        )
        self._ocr_correction_overlay = PersonalDetailOCRCorrectionOverlay(parse_result)

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

    def prepare_candidate_b_business_repair(self, payload: dict[str, Any]) -> bool:
        """Plan the sole post-schema page repair and prepare a second pass."""
        from docmirror.plugins.credit_report.personal_detail_scanned.business_repair import (
            BusinessUncertaintyRepairCoordinator,
        )

        coordinator = BusinessUncertaintyRepairCoordinator(self.parse_result)
        plan = coordinator.plan(
            payload,
            canonical_audit=self.canonical_layout_audit(),
            extraction_issues=(
                dict(issue)
                for issue in getattr(self, "_personal_detail_extraction_issues", ())
                if isinstance(issue, Mapping)
            ),
        )
        plan = coordinator.resolve_page_evidence(
            plan,
            source_pages=self._source_evidence_pages(),
            page_ocr_loader=self.full_page_ocr_evidence,
        )
        self._business_repair_plan = plan
        self._business_repair_evidence_by_page = deepcopy(plan.page_evidence)
        if not plan.requires_second_pass:
            return False

        # The first pass exists only to discover schema uncertainty.  Every
        # extractor in the second pass observes the same repaired evidence.
        self._business_repair_active = True
        self._personal_detail_extraction_issues = deepcopy(
            self._initial_personal_detail_extraction_issues
        )
        self._cache.clear()
        self._canonical_layout_projection_cache = None
        self._canonical_entity_context_ready = False
        from docmirror.plugins.credit_report.personal_detail_scanned.ocr_correction import (
            PersonalDetailOCRCorrectionOverlay,
        )

        self._ocr_correction_overlay = PersonalDetailOCRCorrectionOverlay(self.parse_result)
        return True

    def correct_candidate_b_datasets(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Correct and validate all final v2 source datasets exactly once."""
        plan = self._business_repair_plan
        if plan is not None and plan.affected_pages:
            self._ocr_correction_overlay.install_business_repair_evidence(
                self.corrected_evidence_pages(),
                affected_pages=plan.affected_pages,
            )
        return self._ocr_correction_overlay.correct_business_candidates(
            payload,
            stage="candidate_b_final_validation",
        )

    def corrected_repayment_records(self) -> list[dict[str, Any]]:
        """Rebuild monthly cells from the same canonical page evidence as all fields.

        No cell-level OCR is permitted here. One-shot page re-OCR and
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
            from docmirror.plugins.credit_report.source_table_month_lattice import (
                detached_source_table_geometry_by_page,
            )

            def strip_cross_page_augmentation(bundle: dict[str, Any]) -> None:
                grid_evidence = bundle.get("micro_grid_evidence")
                if not isinstance(grid_evidence, dict):
                    return
                evidence_page = str(
                    grid_evidence.get("page") or bundle.get("page") or ""
                ).strip()
                grid_evidence["lines"] = [
                    line
                    for line in grid_evidence.get("lines") or []
                    if not (
                        isinstance(line, dict)
                        and (
                            line.get("coordinate_status")
                            == "cross_page_y_shift"
                            or (
                                str(line.get("source_logical_page") or "")
                                not in {"", evidence_page}
                            )
                        )
                    )
                ]
                grid_evidence.pop("credit_cross_page_augmented", None)
                grid_evidence.pop("continuation_logical_pages", None)
                grid_evidence.pop(
                    "continuation_source_table_geometry_by_page",
                    None,
                )

            def detached_geometry_from_unique_pages(
                pages: Any,
                selected_pages: set[int],
            ) -> tuple[dict[int, list[dict[str, Any]]], set[int]]:
                """Detach geometry only when page/table ownership is unique."""

                page_values = list(pages) if isinstance(pages, (list, tuple)) else []

                def freeze_geometry(value: Any) -> Any:
                    if isinstance(value, Mapping):
                        return tuple(
                            sorted(
                                (str(key), freeze_geometry(item))
                                for key, item in value.items()
                                if key not in {"table_id", "logical_page", "source_page"}
                            )
                        )
                    if isinstance(value, (list, tuple)):
                        return tuple(freeze_geometry(item) for item in value)
                    return value

                def raw_physical_signature(table: Any) -> Any | None:
                    metadata = (
                        table.get("metadata")
                        if isinstance(table, Mapping)
                        else getattr(table, "metadata", None)
                    )
                    metadata = metadata if isinstance(metadata, Mapping) else {}
                    geometry = metadata.get("geometry")
                    geometry = geometry if isinstance(geometry, Mapping) else {}
                    cell_bboxes = geometry.get("cell_bboxes")
                    if not isinstance(cell_bboxes, (list, tuple)):
                        cell_bboxes = metadata.get("cell_bboxes")
                    bbox = (
                        table.get("bbox")
                        if isinstance(table, Mapping)
                        else getattr(table, "bbox", None)
                    )
                    if not isinstance(bbox, (list, tuple)) and not isinstance(
                        cell_bboxes, (list, tuple)
                    ):
                        return None
                    return freeze_geometry(
                        {
                            "bbox": bbox if isinstance(bbox, (list, tuple)) else None,
                            "cell_bboxes": (
                                cell_bboxes
                                if isinstance(cell_bboxes, (list, tuple))
                                else None
                            ),
                        }
                    )

                page_counts: Counter[int] = Counter()
                page_values_by_logical: dict[int, list[Any]] = {}
                for fallback_page, page_value in enumerate(page_values, start=1):
                    page_number = (
                        page_value.get("page_number")
                        if isinstance(page_value, Mapping)
                        else getattr(page_value, "page_number", None)
                    )
                    logical_page = (
                        page_number
                        if isinstance(page_number, int)
                        and not isinstance(page_number, bool)
                        else fallback_page
                    )
                    if logical_page in selected_pages:
                        page_counts[logical_page] += 1
                        page_values_by_logical.setdefault(logical_page, []).append(
                            page_value
                        )

                blocked_pages = {
                    page for page in selected_pages if page_counts.get(page, 0) != 1
                }
                raw_table_owners: dict[str, set[int]] = {}
                for page in selected_pages - blocked_pages:
                    page_value = page_values_by_logical[page][0]
                    raw_tables = (
                        page_value.get("tables")
                        if isinstance(page_value, Mapping)
                        else getattr(page_value, "tables", None)
                    )
                    if not isinstance(raw_tables, (list, tuple)):
                        blocked_pages.add(page)
                        continue
                    raw_table_ids = [
                        str(
                            (
                                table.get("table_id") or table.get("id")
                                if isinstance(table, Mapping)
                                else getattr(table, "table_id", None)
                                or getattr(table, "id", None)
                            )
                            or ""
                        ).strip()
                        for table in raw_tables
                    ]
                    nonempty_raw_ids = [
                        table_id for table_id in raw_table_ids if table_id
                    ]
                    raw_signatures = [
                        signature
                        for table in raw_tables
                        if (signature := raw_physical_signature(table)) is not None
                    ]
                    for table_id in set(nonempty_raw_ids):
                        raw_table_owners.setdefault(table_id, set()).add(page)
                    if len(nonempty_raw_ids) != len(set(nonempty_raw_ids)) or len(
                        raw_signatures
                    ) != len(set(raw_signatures)):
                        blocked_pages.add(page)
                        continue
                for owners in raw_table_owners.values():
                    if len(owners) > 1:
                        blocked_pages.update(owners)

                detached_by_page = detached_source_table_geometry_by_page(page_values)
                geometry_by_page: dict[int, list[dict[str, Any]]] = {}
                table_owners: dict[str, set[int]] = {}

                for page in selected_pages - blocked_pages:
                    tables = detached_by_page.get(page) or []
                    if not isinstance(tables, list) or not all(
                        isinstance(table, Mapping) for table in tables
                    ):
                        blocked_pages.add(page)
                        continue
                    table_ids = [
                        str(table.get("table_id") or "").strip() for table in tables
                    ]
                    nonempty_ids = [table_id for table_id in table_ids if table_id]
                    signatures = [freeze_geometry(table) for table in tables]
                    if len(nonempty_ids) != len(set(nonempty_ids)) or len(
                        signatures
                    ) != len(set(signatures)):
                        blocked_pages.add(page)
                        continue
                    geometry_by_page[page] = deepcopy(tables)
                    for table_id in nonempty_ids:
                        table_owners.setdefault(table_id, set()).add(page)

                # Repeated physical layouts are expected across report pages;
                # only one table identity claiming multiple pages is ambiguous.
                for owners in table_owners.values():
                    if len(owners) > 1:
                        blocked_pages.update(owners)
                for page in blocked_pages:
                    geometry_by_page.pop(page, None)
                return geometry_by_page, blocked_pages

            detached = deepcopy(_domain_specific(self.parse_result))
            detached.pop("credit_repayment_records", None)
            cross_page_order_authoritative = _authoritative_reading_order(
                getattr(self, "reading_order_resolution", None)
            )
            for bundle in detached.get("_page_evidence_bundles") or []:
                if isinstance(bundle, dict):
                    bundle.pop("micro_grid_structures", None)
                    strip_cross_page_augmentation(bundle)

            canonical_pages: dict[int, dict[str, Any]] = {}
            geometry_blocked_pages: set[int] = set()
            for canonical_page in self.corrected_evidence_pages():
                if not isinstance(canonical_page, dict):
                    continue
                page = canonical_page.get("page")
                if not isinstance(page, int) or isinstance(page, bool) or page <= 0:
                    continue
                if page in canonical_pages:
                    geometry_blocked_pages.add(page)
                    continue
                canonical_pages[page] = canonical_page

            selected_pages = set(canonical_pages)
            canonical_projection = getattr(
                self,
                "_canonical_layout_projection_cache",
                None,
            )
            if canonical_projection is None:
                # Lightweight/legacy contexts do not own a canonical projection;
                # retain their sealed, ordinary-page geometry behavior.
                geometry_pages = getattr(self.parse_result, "pages", None)
            else:
                # The evidence lines and physical cells must come from the same
                # transformed canonical page plane.  Never mix in sealed values.
                geometry_pages = getattr(canonical_projection, "pages", None)
            source_table_geometry_by_page, nonunique_geometry_pages = (
                detached_geometry_from_unique_pages(geometry_pages, selected_pages)
            )
            geometry_blocked_pages.update(nonunique_geometry_pages)
            for page in geometry_blocked_pages:
                source_table_geometry_by_page.pop(page, None)
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
                # Canonical lines replace the prior evidence view, so any
                # augmentation marker on that prior view is stale. Rebuild the
                # adjacent continuation and its detached geometry atomically.
                grid_evidence.pop("credit_cross_page_augmented", None)
                grid_evidence.pop("continuation_logical_pages", None)
                grid_evidence.pop(
                    "continuation_source_table_geometry_by_page",
                    None,
                )
                grid_evidence.update(
                    {
                        "page": page,
                        "page_width": canonical.get("page_width"),
                        "page_height": canonical.get("page_height"),
                        "lines": deepcopy(lines),
                        # Tokens from the sealed pre-registration page must not
                        # outrank the canonical complete-page evidence.
                        "tokens": deepcopy(lines),
                        "source_table_geometry": deepcopy(
                            source_table_geometry_by_page.get(page) or []
                        ),
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
                            "source_table_geometry": deepcopy(
                                source_table_geometry_by_page.get(page) or []
                            ),
                        },
                    }
                )
            if cross_page_order_authoritative:
                augment_credit_repayment_evidence_bundles(
                    detached,
                    reading_order_by_logical=dict(self.reading_order_by_logical),
                )
            status_glyph_observations: list[dict[str, Any]] = []
            materialize_credit_repayment_micro_grids_from_bundles(
                detached,
                page_image_resolver=getattr(self, "_page_image_resolver", None),
                enable_cell_ocr=False,
                enable_static_status_validation=True,
                extra_status_chars={"A", "#"},
                enable_candidate_b_amount_pairing=True,
                candidate_b_status_glyph_observations=status_glyph_observations,
            )
            # Ephemeral visual evidence is held only on this plugin context.
            # It is never inserted into the detached ParseResult view or any
            # micro-grid/dataset projection.
            self._candidate_b_status_glyph_observations = status_glyph_observations
            corrected_grids = micro_grid_structures_from_domain_specific(detached)
            self._corrected_repayment_micro_grids = deepcopy(corrected_grids)
            records = [
                record
                for grid in corrected_grids
                for record in records_from_micro_grid_dict(
                    grid,
                    accept_exact_row_numeric_status=True,
                )
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
                    strip_cross_page_augmentation(bundle)
            if cross_page_order_authoritative:
                augment_credit_repayment_evidence_bundles(
                    source_baseline,
                    reading_order_by_logical=dict(self.reading_order_by_logical),
                )
            materialize_credit_repayment_micro_grids_from_bundles(
                source_baseline,
                page_image_resolver=None,
                enable_cell_ocr=False,
                extra_status_chars={"A", "#"},
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
                interval_gap_series_count = sum(
                    1
                    for months in months_by_series.values()
                    if months and max(months) - min(months) + 1 > len(months)
                )
                within_series_missing_position_count = max(
                    0, schema_implied_count - len(deduped)
                )
                unlocalized_source_structure_delta = max(
                    0, source_structure_count - max(schema_implied_count, len(deduped))
                )
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
                            "within_series_missing_position_count": within_series_missing_position_count,
                            "unlocalized_source_structure_delta": unlocalized_source_structure_delta,
                            "affected_account_or_grid_count": (
                                interval_gap_series_count or None
                            ),
                            "localization_status": (
                                "localized_to_account_or_grid_intervals"
                                if interval_gap_series_count
                                else "unresolved_from_detached_source_structure"
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
            account_gap_issues = [
                issue
                for issue in getattr(self, "_personal_detail_extraction_issues", ())
                if isinstance(issue, Mapping)
                and issue.get("issue_code") == "candidate_b_account_sequence_gap"
                and str(issue.get("status") or "open") != "resolved"
            ]
            if account_gap_issues:
                from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
                    make_issue,
                    record_issue,
                )

                missing_by_family = {
                    str((issue.get("observed_value") or {}).get("account_type") or "unknown"): list(
                        (issue.get("candidate_value") or {}).get("missing_category_sequences") or ()
                    )
                    for issue in account_gap_issues
                }
                record_issue(
                    self,
                    make_issue(
                        category="schema_incompleteness",
                        issue_code="monthly_population_incomplete_from_account_gap",
                        message=(
                            "Monthly performance cannot be population-complete while one or more account-family "
                            "ordinals are unresolved; no months were invented for the missing accounts."
                        ),
                        parser_stage="canonical_monthly_grid_materialization",
                        target_dataset="repayment_records",
                        observed_value={"canonical_grid_row_count": len(deduped)},
                        candidate_value={"missing_account_category_sequences": missing_by_family},
                        source_refs=(
                            dict(ref)
                            for issue in account_gap_issues
                            for ref in issue.get("source_refs") or ()
                            if isinstance(ref, Mapping)
                        ),
                        reason_codes=(
                            "credit_account_population_incomplete",
                            "monthly_population_cannot_be_complete",
                            "missing_months_not_invented",
                        ),
                    ),
                )
            return deduped

        return self.cached("corrected_repayment_records", rebuild)

    def corrected_repayment_micro_grids(self) -> list[dict[str, Any]]:
        """Return the exact canonical grids used to materialize monthly rows."""
        self.corrected_repayment_records()
        return deepcopy(getattr(self, "_corrected_repayment_micro_grids", []))

    def candidate_b_status_glyph_observations(self) -> list[dict[str, Any]]:
        """Return private document-local glyph evidence for the final gate."""
        self.corrected_repayment_records()
        return deepcopy(
            getattr(self, "_candidate_b_status_glyph_observations", [])
        )

    def section_content(self, full_text: str) -> dict[str, Any]:
        """Return supplemental datasets from the same Candidate B result."""
        return deepcopy(self.candidate_b_extraction(full_text).section_content)

    def corrected_evidence_pages(self) -> list[dict[str, Any]]:
        """Return the registered canonical evidence shared by every extractor."""
        return deepcopy(list(self._canonical_layout_projection().evidence_pages))

    def _source_evidence_pages(self) -> list[dict[str, Any]]:
        """Return detached static evidence used to register canonical pages."""
        return self.cached("source_evidence_pages", self._build_source_evidence_pages)

    def _canonical_layout_projection(self) -> Any:
        if self._canonical_layout_projection_cache is None:
            from docmirror.plugins.credit_report.personal_detail_scanned.canonical_layout import (
                PBOCCanonicalTemplateAssembler,
            )

            assembler = PBOCCanonicalTemplateAssembler(
                self.parse_result,
                topology=self.page_topology,
                reading_order_by_logical=self.reading_order_by_logical,
                source_evidence_loader=self._source_evidence_pages,
                issue_owner=self,
                source_page_loader=lambda: list(self._frozen_logical_pages.values()),
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
        (
            units,
            furniture,
            evidence_units,
            source_pages,
            _reading_order,
            _reading_order_resolution,
        ) = _collect_personal_detail_units(
            adapter,
            topology=self.page_topology,
            registered_reading_order=self.reading_order_by_logical,
            registered_reading_order_resolution=self.reading_order_resolution,
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
        self._canonical_entity_context_ready = True

    def _build_source_evidence_pages(self) -> list[dict[str, Any]]:
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
        ordered = sorted(
            pages,
            key=lambda item: (
                self.reading_order_by_logical.get(item["page"], item["page"]),
                item["page"],
            ),
        )
        static_pages = self._construct_static_topology_pages(ordered)
        if not self._business_repair_active:
            return static_pages

        merged: list[dict[str, Any]] = []
        affected = set(self._business_repair_evidence_by_page)
        for source in static_pages:
            logical = int(source.get("page") or 0)
            replacement = self._business_repair_evidence_by_page.get(logical)
            if replacement is None:
                merged.append(source)
                continue
            merged.append(
                {
                    **source,
                    **deepcopy(replacement),
                    "page": logical,
                    "logical_page": logical,
                    "source_page": int(replacement.get("source_page") or source.get("source_page") or logical),
                }
            )
        known = {int(page.get("page") or 0) for page in merged}
        merged.extend(
            deepcopy(page)
            for logical, page in self._business_repair_evidence_by_page.items()
            if logical not in known
        )
        # Line-level normalization is permitted only now: the first schema
        # pass selected these pages as business-uncertain.
        selected = [page for page in merged if int(page.get("page") or 0) in affected]
        corrected_by_page = {
            int(page.get("page") or 0): page
            for page in self._ocr_correction_overlay.corrected_evidence_pages(selected)
        }
        return [corrected_by_page.get(int(page.get("page") or 0), page) for page in merged]

    def _construct_static_topology_pages(
        self,
        pages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Freeze statically split pages and partition existing evidence."""
        audit = self.page_topology.audit()
        source_candidates = {
            int(source)
            for source, logicals in (audit.get("logical_pages_by_source") or {}).items()
            if len(logicals) == 1
            and (
                (geometry := self.page_topology.geometry(int(logicals[0]))) is not None
                and (geometry.split_kind == "two_page_spread" or geometry.split_confidence >= 0.55)
            )
        }
        if not source_candidates:
            return pages
        split_pages = self._page_image_resolver.static_split_slices(source_candidates)
        for decision in self._page_image_resolver.audit().get("static_split_decisions") or []:
            if not isinstance(decision, Mapping) or decision.get("status") not in {"uncertain", "failed"}:
                continue
            source = int(decision.get("source_page") or 0)
            failed = decision.get("status") == "failed"
            issue_code = "static_page_split_validation_failed" if failed else "static_page_split_uncertain"
            if any(
                item.get("code") == issue_code
                and int(item.get("source_page") or 0) == source
                for item in self._topology_recovery_issues
            ):
                continue
            self._topology_recovery_issues.append(
                {
                    "code": issue_code,
                    "message": (
                        "Static page-split validation could not run; the original logical page was preserved."
                        if failed
                        else "Static image geometry could not safely confirm or reject a potential page split."
                    ),
                    **deepcopy(dict(decision)),
                }
            )
        if not split_pages:
            return pages

        base_by_source: dict[int, list[dict[str, Any]]] = {}
        for page in pages:
            base_by_source.setdefault(int(page.get("source_page") or 0), []).append(page)
        static_by_source: dict[int, list[dict[str, Any]]] = {}
        for page in split_pages:
            static_by_source.setdefault(int(page.get("source_page") or 0), []).append(page)

        next_logical = max(self.source_page_by_logical, default=0) + 1
        merged: list[dict[str, Any]] = []
        for source in sorted(set(base_by_source) | set(static_by_source)):
            base_pages = base_by_source.get(source, [])
            constructed_pages = sorted(
                static_by_source.get(source, []),
                key=lambda item: int(item.get("segment_index") or 0),
            )
            if not constructed_pages:
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

            split_by_segment = {int(page.get("segment_index") or 0): page for page in constructed_pages}
            available_segments = set(base_by_segment) | set(split_by_segment)
            if available_segments != {0, 1}:
                # Never replace a source with only one half of a statically
                # confirmed spread. Preserve the original evidence and report
                # the topology uncertainty without invoking OCR.
                merged.extend(base_pages)
                if not any(
                    item.get("code") == "static_split_pair_incomplete"
                    and int(item.get("source_page") or 0) == source
                    for item in self._topology_recovery_issues
                ):
                    self._topology_recovery_issues.append(
                        {
                            "code": "static_split_pair_incomplete",
                            "message": "Static spread validation did not produce both logical subpages.",
                            "source_page": source,
                            "expected_segments": [0, 1],
                            "observed_segments": sorted(available_segments),
                        }
                    )
                continue
            unsplit_logical = logicals[0] if len(logicals) == 1 and not geometries else 0
            partitioned_lines: dict[int, list[dict[str, Any]]] = {0: [], 1: []}
            partitioned_tables: dict[int, list[Any]] = {0: [], 1: []}
            ambiguous_items = 0
            if unsplit_logical and base_pages:
                base_evidence = base_pages[0]
                raw_page = self._frozen_logical_pages.get(unsplit_logical)
                transform = dict(getattr(raw_page, "coordinate_transform", None) or {})
                base_to_source = _matrix3(transform.get("inverse_matrix"))
                if base_to_source is None:
                    merged.extend(base_pages)
                    self._topology_recovery_issues.append(
                        {
                            "code": "static_split_evidence_transform_unusable",
                            "message": (
                                "The original page evidence could not be projected into the static subpages; "
                                "the unsplit page was preserved."
                            ),
                            "source_page": source,
                            "logical_page": unsplit_logical,
                        }
                    )
                    continue
                for line in base_evidence.get("lines") or []:
                    if not isinstance(line, Mapping) or (box := _bbox(line)) is None:
                        ambiguous_items += 1
                        continue
                    source_box = _transform_bbox(base_to_source, box)
                    segment, ambiguous = _slice_assignment(source_box, split_by_segment)
                    if ambiguous or segment is None:
                        ambiguous_items += 1
                        continue
                    source_to_slice = _matrix3(split_by_segment[segment].get("source_to_logical"))
                    if source_to_slice is None:
                        ambiguous_items += 1
                        continue
                    local = deepcopy(dict(line))
                    local["source_bbox"] = list(source_box)
                    local["bbox"] = list(_transform_bbox(source_to_slice, source_box))
                    partitioned_lines[segment].append(local)
                for table in getattr(raw_page, "tables", None) or []:
                    if (box := _bbox(table)) is None:
                        ambiguous_items += 1
                        continue
                    source_box = _transform_bbox(base_to_source, box)
                    segment, ambiguous = _slice_assignment(source_box, split_by_segment)
                    if ambiguous or segment is None:
                        ambiguous_items += 1
                        continue
                    source_to_slice = _matrix3(split_by_segment[segment].get("source_to_logical"))
                    if source_to_slice is None:
                        ambiguous_items += 1
                        continue
                    partitioned_tables[segment].append(
                        _project_table_for_static_slice(
                            table,
                            base_to_source=base_to_source,
                            source_to_slice=source_to_slice,
                        )
                    )
            if ambiguous_items:
                self._topology_recovery_issues.append(
                    {
                        "code": "static_split_boundary_ambiguous",
                        "message": "Evidence crossing the static split boundary was withheld from both subpages.",
                        "source_page": source,
                        "ambiguous_item_count": ambiguous_items,
                    }
                )
            for segment in (0, 1):
                if segment in base_by_segment:
                    merged.append(base_by_segment[segment])
                    continue
                static_page = split_by_segment.get(segment)
                if static_page is None:
                    continue
                logical = geometries.get(segment, 0)
                if not logical and unsplit_logical and segment == 0:
                    logical = unsplit_logical
                if not logical:
                    logical = next_logical
                    next_logical += 1
                static_page = dict(static_page)
                static_page["page"] = logical
                static_page["logical_page"] = logical
                static_page["plugin_static_subpage"] = True
                static_page["lines"] = sorted(
                    partitioned_lines.get(segment, []),
                    key=lambda line: ((_bbox(line) or (0, 0, 0, 0))[1], (_bbox(line) or (0, 0, 0, 0))[0]),
                )
                self.source_page_by_logical[logical] = source
                self._page_image_resolver.register_static_logical_page(logical, static_page)
                # The resolver owns the frozen pixel surface. Canonical
                # evidence remains lightweight and serializable.
                static_page.pop("image", None)
                self._frozen_logical_pages[logical] = SimpleNamespace(
                    page_number=logical,
                    source_page_number=source,
                    width=float(static_page.get("page_width") or 0.0),
                    height=float(static_page.get("page_height") or 0.0),
                    coordinate_transform=deepcopy(static_page.get("coordinate_transform") or {}),
                    tables=partitioned_tables.get(segment, []),
                    texts=[
                        SimpleNamespace(content=str(line.get("text") or ""), bbox=list(line.get("bbox") or []))
                        for line in static_page["lines"]
                    ],
                )
                merged.append(static_page)

        order_keys: dict[int, tuple[int, int, int]] = {}
        for logical, source in self.source_page_by_logical.items():
            source_logicals = self.page_topology.logicals_for_source(source)
            source_order = min(
                (self.reading_order_by_logical.get(item, item) for item in source_logicals),
                default=self.reading_order_by_logical.get(logical, logical),
            )
            geometry = self.page_topology.geometry(logical)
            segment = int(geometry.segment_index) if geometry and geometry.segment_index in {0, 1} else 0
            static_page = next((item for item in merged if int(item.get("page") or 0) == logical), None)
            if static_page and static_page.get("plugin_static_subpage"):
                segment = int(static_page.get("segment_index") or 0)
            order_keys[logical] = (source_order, segment, logical)
        self.reading_order_by_logical.clear()
        self.reading_order_by_logical.update(
            {logical: index for index, logical in enumerate(sorted(order_keys, key=lambda item: order_keys[item]), start=1)}
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
        """Return the one cached re-OCR result for each frozen logical page."""
        if os.environ.get("DOCMIRROR_PERSONAL_DETAIL_PAGE_OCR", "1") == "0":
            return []
        requested = sorted({int(page) for page in logical_pages if int(page) > 0})
        output: list[dict[str, Any]] = []
        for logical in requested:
            rendered = self._page_image_resolver(logical)
            page_key = self._page_image_resolver.page_key(logical)
            if not page_key:
                continue

            def produce() -> tuple[dict[str, Any] | None, str, dict[str, Any]]:
                if not rendered:
                    return None, "render_failed", {"ocr_invocations": 0}
                image = rendered.get("image")
                shape = getattr(image, "shape", None)
                if not shape or len(shape) < 2 or not shape[0] or not shape[1]:
                    return None, "invalid_image", {"ocr_invocations": 0}
                words, page_score = _single_page_ocr(image)
                page_width = float(rendered.get("page_width") or 0.0)
                page_height = float(rendered.get("page_height") or 0.0)
                if page_width <= 0 or page_height <= 0:
                    zoom = float(rendered.get("zoom") or 1.0)
                    page_width = float(shape[1]) / zoom
                    page_height = float(shape[0]) / zoom
                scale_x = page_width / float(shape[1])
                scale_y = page_height / float(shape[0])
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
                            "evidence_ids": [f"personal_detail_page_reocr:{page_key}:w{index}"],
                            "source": "personal_detail_page_reocr_once",
                        }
                    )
                details = {
                    "ocr_invocations": 1,
                    "word_count": len(lines),
                    "page_score": page_score,
                }
                if not lines:
                    return None, "ocr_empty", details
                transform = dict(rendered.get("coordinate_transform") or {})
                decomposition = dict(transform.get("decomposition") or {})
                return (
                    {
                        "page": logical,
                        "logical_page": logical,
                        "page_key": page_key,
                        "source_page": int(
                            rendered.get("source_page") or self.source_page_by_logical.get(logical, logical)
                        ),
                        "page_width": page_width,
                        "page_height": page_height,
                        "selected_rotation": int(decomposition.get("selected_rotation") or 0),
                        "lines": sorted(lines, key=lambda line: (line["bbox"][1], line["bbox"][0])),
                    },
                    "completed",
                    details,
                )

            page = self._page_reocr_registry.resolve(
                page_key=page_key,
                logical_page=logical,
                reason=str(reason),
                producer=produce,
            )
            if page is not None:
                output.append(page)
        return output

    def ocr_correction_audit(self) -> dict[str, Any]:
        """Return a detached audit snapshot for diagnostics and regression tests."""
        registry = self._page_reocr_registry.audit()
        requests = registry.pop("page_reocr_requests", [])
        return {
            **deepcopy(self._ocr_correction_overlay.audit()),
            **registry,
            "business_repair": (
                deepcopy(self.__dict__["_business_repair_plan"].audit())
                if self.__dict__.get("_business_repair_plan") is not None
                else {
                    "architecture": "schema_triggered_page_repair_v1",
                    "first_pass_uncertainty_count": 0,
                    "affected_pages": [],
                    "second_schema_pass_required": False,
                }
            ),
            "page_reocr_failures": [
                row
                for row in requests
                if isinstance(row, Mapping) and row.get("status") not in {"completed", "requested"}
            ],
        }

    def page_topology_audit(self) -> dict[str, Any]:
        """Return the plugin's detached logical-page validation result."""
        audit = deepcopy(self._page_image_resolver.audit())
        audit["issues"] = [*(audit.get("issues") or []), *deepcopy(self._topology_recovery_issues)]
        audit["ocr_used_for_topology"] = False
        audit["topology_frozen_before_reocr"] = True
        return audit

    def canonical_layout_audit(self) -> dict[str, Any]:
        """Return the detached template-registration and fragment audit."""
        audit = deepcopy(self._canonical_layout_projection().audit())
        audit["reading_order_resolution"] = deepcopy(self.reading_order_resolution)
        return audit

    def tables_continue(self, left_table_id: str, right_table_id: str) -> bool | None:
        left_unit_id = self.entity_context.table_unit_id(left_table_id)
        right_unit_id = self.entity_context.table_unit_id(right_table_id)
        if not left_unit_id or not right_unit_id:
            return None
        units_by_id = {unit.unit_id: unit for unit in self.entity_context.units}
        left_unit = units_by_id.get(left_unit_id)
        right_unit = units_by_id.get(right_unit_id)
        if left_unit is None or right_unit is None:
            return None
        if left_unit.page != right_unit.page:
            if not _authoritative_reading_order(
                getattr(self, "reading_order_resolution", None)
            ) or not (
                isinstance(left_unit.page, int)
                and not isinstance(left_unit.page, bool)
                and isinstance(right_unit.page, int)
                and not isinstance(right_unit.page, bool)
                and right_unit.page == left_unit.page + 1
            ):
                return False
        left = self.entity_context.entity_for_unit(left_unit_id)
        right = self.entity_context.entity_for_unit(right_unit_id)
        return bool(left is not None and right is not None and left.entity_id == right.entity_id)

    def pages_adjacent_in_reading_order(self, left_page: int, right_page: int) -> bool:
        if not _authoritative_reading_order(
            getattr(self, "reading_order_resolution", None)
        ):
            return False
        if (
            not isinstance(left_page, int)
            or isinstance(left_page, bool)
            or left_page <= 0
            or not isinstance(right_page, int)
            or isinstance(right_page, bool)
            or right_page <= 0
        ):
            return False
        registered_positions = list(self.reading_order_by_logical.values())
        registered_logical_pages = set(
            getattr(self, "source_page_by_logical", {}) or {}
        )
        if (
            any(
                not isinstance(position, int)
                or isinstance(position, bool)
                or position <= 0
                for position in registered_positions
            )
            or len(registered_positions) != len(set(registered_positions))
            or (
                registered_logical_pages
                and not registered_logical_pages.issubset(
                    self.reading_order_by_logical
                )
            )
        ):
            return False
        left_order = self.reading_order_by_logical.get(left_page)
        right_order = self.reading_order_by_logical.get(right_page)
        if (
            not isinstance(left_order, int)
            or isinstance(left_order, bool)
            or not isinstance(right_order, int)
            or isinstance(right_order, bool)
        ):
            return False
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
        if not _authoritative_reading_order(
            getattr(self, "reading_order_resolution", None)
        ):
            return False
        if not self.pages_adjacent_in_reading_order(left_page, right_page):
            return False
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
    (
        units,
        furniture,
        evidence_units,
        source_pages,
        reading_order,
        reading_order_resolution,
    ) = _collect_personal_detail_units(parse_result, topology=page_topology)
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
        reading_order_resolution=reading_order_resolution,
        page_topology=page_topology,
    )


__all__ = [
    "PersonalDetailExtractionContext",
    "PersonalDetailTransitionPolicy",
    "build_personal_detail_extraction_context",
]
