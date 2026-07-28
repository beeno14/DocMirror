# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-conserving reading repair for numbered personal-brief account lists."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from docmirror.output.reading_projection import ReadingProjection, SliceReflow, SourceTextSlice
from docmirror.plugins.credit_report.reading_order_utils import (
    ordered_document_nodes as _ordered_nodes,
)
from docmirror.plugins.credit_report.reading_order_utils import (
    valid_bbox as _bbox,
)

_DATE_PATTERN = r"20\d{2}年\s*\d{1,2}月\s*\d{1,2}日"
_ACCOUNT_START_RE = re.compile(
    rf"{_DATE_PATTERN}"
    rf"(?=(?:(?!{_DATE_PATTERN}).){{4,100}}?(?:发放的|为(?=.{{0,30}}贷款授信)))",
    re.DOTALL,
)
_LIABILITY_START_RE = re.compile(
    rf"{_DATE_PATTERN}(?=\s*[，,]\s*为(?:(?!{_DATE_PATTERN}).){{1,180}}?承担相关还款责任)",
    re.DOTALL,
)
_ORDINAL_RE = re.compile(r"(?m)^[ \t]*(?P<number>\d{1,3})(?P<punct>\.)[ \t]*$")
_ACCOUNT_TRAILING_MARKERS = ("相关还款责任信息",)


@dataclass(frozen=True)
class _RecordSpec:
    scope: str
    pattern: re.Pattern[str]
    required_markers: tuple[str, ...]
    trailing_markers: tuple[str, ...] = ()


_RECORD_SPECS = (
    _RecordSpec(
        scope="credit_report.personal_brief.credit_accounts",
        pattern=_ACCOUNT_START_RE,
        required_markers=("贷款", "贷记卡", "准贷记卡"),
        trailing_markers=_ACCOUNT_TRAILING_MARKERS,
    ),
    _RecordSpec(
        scope="credit_report.personal_brief.repayment_liabilities",
        pattern=_LIABILITY_START_RE,
        required_markers=("承担相关还款责任",),
    ),
)


def _ordinal_slices(node: Any) -> list[tuple[int, SourceTextSlice]]:
    text = str(getattr(node, "text", "") or "")
    node_id = str(getattr(node, "node_id", "") or "")
    if not text or not node_id:
        return []
    matches = list(_ORDINAL_RE.finditer(text))
    if not matches:
        return []
    cursor = 0
    slices: list[tuple[int, SourceTextSlice]] = []
    for match in matches:
        if text[cursor : match.start()].strip():
            return []
        slices.append(
            (
                int(match.group("number")),
                SourceTextSlice(
                    node_id=node_id,
                    start=match.start("number"),
                    end=match.end("punct"),
                ),
            )
        )
        cursor = match.end()
    return slices if not text[cursor:].strip() else []


def _record_slices(
    node: Any,
    spec: _RecordSpec,
) -> tuple[list[SourceTextSlice], list[SourceTextSlice]]:
    text = str(getattr(node, "text", "") or "")
    node_id = str(getattr(node, "node_id", "") or "")
    matches = list(spec.pattern.finditer(text))
    if len(matches) < 3 or not node_id or text[: matches[0].start()].strip():
        return [], []

    content_end = len(text)
    suffix_start: int | None = None
    for marker in spec.trailing_markers:
        marker_start = text.find(marker, matches[-1].end())
        if marker_start >= 0 and (suffix_start is None or marker_start < suffix_start):
            suffix_start = marker_start
    if suffix_start is not None:
        content_end = suffix_start

    records: list[SourceTextSlice] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else content_end
        if end <= match.start():
            return [], []
        chunk = text[match.start() : end]
        if not any(marker in chunk for marker in spec.required_markers):
            return [], []
        records.append(SourceTextSlice(node_id=node_id, start=match.start(), end=end))

    suffixes = [SourceTextSlice(node_id=node_id, start=suffix_start, end=len(text))] if suffix_start is not None else []
    return records, suffixes


def _label_geometry_matches(label: Any, narrative: Any) -> bool:
    label_box = _bbox(label)
    narrative_box = _bbox(narrative)
    if label_box is None or narrative_box is None:
        return False
    if int(getattr(label, "page", 0) or 0) != int(getattr(narrative, "page", 0) or 0):
        return False
    narrative_width = narrative_box[2] - narrative_box[0]
    return (
        narrative_width >= 120.0
        and label_box[2] <= narrative_box[0] + 60.0
        and abs(label_box[1] - narrative_box[1]) <= 12.0
    )


def _build_reflow(
    ordered: list[Any],
    narrative_index: int,
    spec: _RecordSpec,
) -> SliceReflow | None:
    if narrative_index <= 0:
        return None
    narrative = ordered[narrative_index]
    records, suffixes = _record_slices(narrative, spec)
    if not records:
        return None

    first_label = ordered[narrative_index - 1]
    first_ordinals = _ordinal_slices(first_label)
    if [number for number, _slice in first_ordinals] != [1]:
        return None
    if not _label_geometry_matches(first_label, narrative):
        return None

    labels = list(first_ordinals)
    source_end = narrative_index
    cursor = narrative_index + 1
    while len(labels) < len(records) and cursor < len(ordered):
        node_ordinals = _ordinal_slices(ordered[cursor])
        if not node_ordinals or len(labels) + len(node_ordinals) > len(records):
            return None
        labels.extend(node_ordinals)
        source_end = cursor
        cursor += 1
    if [number for number, _slice in labels] != list(range(1, len(records) + 1)):
        return None

    output_slices: list[SourceTextSlice] = []
    for (_number, label_slice), record_slice in zip(labels, records, strict=True):
        output_slices.extend((label_slice, record_slice))
    output_slices.extend(suffixes)

    source_nodes = ordered[narrative_index - 1 : source_end + 1]
    source_node_ids = tuple(str(getattr(node, "node_id", "") or "") for node in source_nodes)
    if not all(source_node_ids):
        return None
    evidence_ids = tuple(
        dict.fromkeys(
            str(evidence_id)
            for node in source_nodes
            for evidence_id in (getattr(node, "evidence_refs", None) or [])
            if evidence_id
        )
    )
    return SliceReflow(
        scope=spec.scope,
        source_node_ids=source_node_ids,
        anchor_node_id=source_node_ids[0],
        output_slices=tuple(output_slices),
        reason="numbered record labels and narrative rows were emitted as separate PDF text blocks",
        confidence=0.99,
        evidence_ids=evidence_ids,
    )


def build_account_reading_projection(parse_result: Any) -> ReadingProjection | None:
    """Build source-conserving reflows for unambiguously numbered account lists."""
    ordered = _ordered_nodes(parse_result)
    if not ordered:
        return None
    transforms: list[SliceReflow] = []
    claimed: set[str] = set()
    for index, _node in enumerate(ordered):
        for spec in _RECORD_SPECS:
            transform = _build_reflow(ordered, index, spec)
            if transform is None or claimed.intersection(transform.source_node_ids):
                continue
            transforms.append(transform)
            claimed.update(transform.source_node_ids)
            break
    if not transforms:
        return None
    return ReadingProjection(plugin_id="credit_report", transforms=tuple(transforms))


__all__ = ["build_account_reading_projection"]
