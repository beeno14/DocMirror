# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local row reconstruction for personal-brief institution inquiries.

The native PDF lays out ``机构查询记录明细`` as aligned text rather than a
ruled table.  Wrapped institution fragments can consequently occur after the
inquiry-reason fragment in canonical reading order.  This module reconstructs
only the plugin's logical inquiry records; it never mutates ``ParseResult``.
"""

from __future__ import annotations

import re
from typing import Any

from docmirror.plugins.credit_report.reading_order_utils import (
    ordered_document_nodes as _ordered_nodes,
)
from docmirror.plugins.credit_report.reading_order_utils import (
    valid_bbox as _bbox,
)
from docmirror.plugins.credit_report.value_utils import compact_text as _compact

_INSTITUTION_SECTION = "机构查询记录明细"
_SECTION_END_MARKERS = ("个人查询记录明细", "说明")
_ROW_START_RE = re.compile(
    r"^\s*(?P<sequence>\d{1,4})\s+"
    r"(?P<date>20\d{2}年\s*\d{1,2}月\s*\d{1,2}日)\s*"
    r"(?P<institution>.*)$",
    re.DOTALL,
)
_INQUIRY_REASONS = frozenset(
    {
        "法人代表、负责人、高管等资信审查",
        "本人查询（互联网个人信用信息服务平台）",
        "本人查询(互联网个人信用信息服务平台)",
        "本人查询（自助查询机）",
        "本人查询(自助查询机)",
        "担保资格审查",
        "资信审查",
        "融资审批",
        "信用卡审批",
        "贷款审批",
        "贷后管理",
    }
)
_COMPACT_INQUIRY_REASONS = frozenset(re.sub(r"\s+", "", reason) for reason in _INQUIRY_REASONS)


def _row_match(value: Any) -> re.Match[str] | None:
    return _ROW_START_RE.match(str(getattr(value, "text", "") or ""))


def _reason_sequence(values: list[Any]) -> tuple[str, list[Any]]:
    """Find one known reason even when unrelated column fragments interleave it."""
    for reason in sorted(_COMPACT_INQUIRY_REASONS, key=len, reverse=True):
        assembled = ""
        selected: list[Any] = []
        for value in values:
            fragment = _compact(getattr(value, "text", ""))
            if fragment and reason.startswith(assembled + fragment):
                assembled += fragment
                selected.append(value)
                if assembled == reason:
                    return reason.replace("(", "（").replace(")", "）"), selected
    return "", []


def _is_section_end(value: Any) -> bool:
    compact = _compact(getattr(value, "text", ""))
    return any(compact == marker or compact.startswith(marker) for marker in _SECTION_END_MARKERS)


def _is_institution_fragment(anchor: Any, reason: Any, candidate: Any) -> bool:
    """Return whether a node belongs to the current row's institution column."""
    anchor_box = _bbox(anchor)
    reason_box = _bbox(reason)
    candidate_box = _bbox(candidate)
    if anchor_box is None or reason_box is None or candidate_box is None:
        return False
    anchor_page = int(getattr(anchor, "page", 0) or 0)
    if int(getattr(reason, "page", 0) or 0) != anchor_page or int(getattr(candidate, "page", 0) or 0) != anchor_page:
        return False

    text = _compact(getattr(candidate, "text", ""))
    if (
        not text
        or len(text) > 40
        or text in _COMPACT_INQUIRY_REASONS
        or _row_match(candidate) is not None
        or _INSTITUTION_SECTION in text
        or any(marker in text for marker in _SECTION_END_MARKERS)
    ):
        return False

    anchor_x0, anchor_y0, _, anchor_y1 = anchor_box
    reason_x0, reason_y0, _, reason_y1 = reason_box
    candidate_x0, candidate_y0, _, candidate_y1 = candidate_box
    row_height = max(1.0, anchor_y1 - anchor_y0, reason_y1 - reason_y0)
    institution_floor = anchor_x0 + 0.35 * max(1.0, reason_x0 - anchor_x0)

    return (
        institution_floor <= candidate_x0 < reason_x0 - 4.0
        and candidate_y0 >= min(anchor_y0, reason_y0) - row_height * 0.5
        and candidate_y1 - max(anchor_y1, reason_y1) <= row_height * 3.25
    )


def reconstruct_institution_inquiry_rows(parse_result: Any) -> list[dict[str, Any]]:
    """Reconstruct logical institution-inquiry rows from the first DFG flow.

    A returned row contains raw date text and plugin-local provenance.  The
    business extractor remains responsible for canonical dates and record IDs.
    """
    ordered = _ordered_nodes(parse_result)
    if not ordered:
        return []
    section_start = next(
        (index for index, node in enumerate(ordered) if _INSTITUTION_SECTION in _compact(getattr(node, "text", ""))),
        -1,
    )
    if section_start < 0:
        return []
    section_end = next(
        (
            index
            for index, node in enumerate(ordered[section_start + 1 :], start=section_start + 1)
            if _is_section_end(node)
        ),
        len(ordered),
    )

    rows: list[dict[str, Any]] = []
    index = section_start + 1
    while index < section_end:
        anchor = ordered[index]
        anchor_match = _row_match(anchor)
        if anchor_match is None:
            index += 1
            continue
        next_row = next(
            (
                candidate_index
                for candidate_index in range(index + 1, section_end)
                if _row_match(ordered[candidate_index]) is not None
            ),
            section_end,
        )
        row_tail = ordered[index + 1 : next_row]
        reason, reason_nodes = _reason_sequence(row_tail)
        if not reason_nodes or _bbox(anchor) is None:
            index = next_row
            continue

        reason_node = reason_nodes[0]
        if _bbox(reason_node) is None:
            index = next_row
            continue
        reason_node_keys = {id(node) for node in reason_nodes}
        fragments = [
            node
            for node in row_tail
            if id(node) not in reason_node_keys and _is_institution_fragment(anchor, reason_node, node)
        ]
        institution = _compact(anchor_match.group("institution")) + "".join(
            _compact(getattr(fragment, "text", "")) for fragment in fragments
        )
        if not institution or len(institution) > 100:
            index = next_row
            continue

        relevant_node_keys = {id(node) for node in (*reason_nodes, *fragments)}
        source_nodes = [anchor, *(node for node in row_tail if id(node) in relevant_node_keys)]
        rows.append(
            {
                "sequence": int(anchor_match.group("sequence")),
                "query_date_text": anchor_match.group("date"),
                "institution": institution,
                "reason": reason,
                "page": int(getattr(anchor, "page", 0) or 0),
                "anchor_node_id": str(getattr(anchor, "node_id", "") or ""),
                "reason_node_id": str(getattr(reason_node, "node_id", "") or ""),
                "reason_node_ids": [str(getattr(node, "node_id", "") or "") for node in reason_nodes],
                "fragment_node_ids": [str(getattr(node, "node_id", "") or "") for node in fragments],
                "source_node_ids": [str(getattr(node, "node_id", "") or "") for node in source_nodes],
                "evidence_ids": list(
                    dict.fromkeys(
                        str(evidence_id)
                        for node in source_nodes
                        for evidence_id in (getattr(node, "evidence_refs", None) or [])
                        if evidence_id
                    )
                ),
                "confidence": 0.99 if fragments else 0.98,
            }
        )
        index = next_row
    return rows


__all__ = ["reconstruct_institution_inquiry_rows"]
