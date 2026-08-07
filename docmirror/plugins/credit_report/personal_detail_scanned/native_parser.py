# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schema-directed card decoder for PBOC personal detailed reports.

This is the single Candidate B decoder for labelled credit-agreement and
repayment-responsibility cards. Native tables, registered page rows, and
whole-page OCR retries are observations inside one decoder, not competing
business populations. Only exact canonical labels and explicitly enumerated
OCR-confusion aliases may authorize a business field binding.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
    make_issue,
    record_issue,
)

_LABELS = frozenset(
    {
        "被查询者姓名",
        "被查询者证件类型",
        "被查询者证件号码",
        "查询机构",
        "查询原因",
        "证件类型",
        "证件号码",
        "授信协议标识",
        "管理机构",
        "授信额度用途",
        "生效日期",
        "到期日期",
        "授信额度",
        "授信限额",
        "已用额度",
        "授信限额编号",
        "币种",
        "责任人类型",
        "还款责任金额",
        "保证合同编号",
        "主业务借款人",
        "主业务借款人证件类型",
        "主业务借款人证件号码",
        "开立日期",
        "业务种类",
        "余额",
        "五级分类",
        "逾期月数",
        "还款状态",
    }
)

# Closed-world label repair.  Keep this deliberately small: an entry must be a
# known OCR rendering of one canonical PBOC label, not a spelling-similarity
# candidate.  Unknown/damaged labels stay unresolved so the coordinated page
# repair can observe them without authorizing a neighbouring business value.
_LABEL_OCR_ALIASES = {
    "营理机构": "管理机构",
    "开立白期": "开立日期",
}

_SECTION_MARKERS = {
    "credit_lines": frozenset({"授信协议标识", "授信额度用途"}),
    "repayment_liability_records": frozenset({"责任人类型", "保证合同编号"}),
    "report_header": frozenset({"被查询者姓名", "被查询者证件类型", "被查询者证件号码", "查询机构", "查询原因"}),
}

_EVIDENCE_SECTION_HEADINGS = {
    "credit_lines": "授信协议信息",
    "repayment_liability_records": "相关还款责任信息",
}
_EVIDENCE_ANCHORS = {
    "credit_lines": re.compile(r"授信协议\s*\d{1,3}"),
    "repayment_liability_records": re.compile(r"账户\s*\d{1,3}"),
}
_PRIMARY_RECORD_LABELS = {
    "credit_lines": "授信协议标识",
    "repayment_liability_records": "保证合同编号",
}
_EVIDENCE_SECTION_END_MARKERS = (
    "非信贷交易信息",
    "公共信息",
    "查询记录",
    "本人声明",
    "异议标注",
    "报告说明",
)
def _compact(value: Any) -> str:
    return re.sub(r"[\s:：,，。；;()（）\[\]【】]", "", str(value or "")).strip()


def _rows(table: Any) -> list[list[str]]:
    metadata = getattr(table, "metadata", None) or {}
    raw_rows = metadata.get("raw_rows") if isinstance(metadata, dict) else None
    if isinstance(raw_rows, list) and raw_rows:
        return [[str(cell or "") for cell in row] for row in raw_rows if isinstance(row, list)]
    headers = [str(value or "") for value in getattr(table, "headers", None) or []]
    body = [
        [str(getattr(cell, "text", "") or "") for cell in getattr(row, "cells", None) or []]
        for row in getattr(table, "rows", None) or []
    ]
    return ([headers] if headers else []) + body


def _canonical_label(value: Any) -> tuple[str | None, float]:
    text = _compact(value)
    if not text:
        return None, 0.0
    if text in _LABELS:
        return text, 1.0
    alias = _LABEL_OCR_ALIASES.get(text)
    if alias in _LABELS:
        return alias, 0.96
    return None, 0.0


def _source_ref(page: Any, table: Any) -> dict[str, Any]:
    ref: dict[str, Any] = {
        "source": "native_detail_tolerant_table",
        "logical_page": int(getattr(page, "page_number", 0) or 0),
        "source_page": int(getattr(page, "source_page_number", 0) or getattr(page, "page_number", 0) or 0),
        "table_id": str(getattr(table, "table_id", "") or ""),
    }
    bbox = getattr(table, "bbox", None)
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        ref["bbox"] = list(bbox)
        ref["geometry_scope"] = "table"
    return ref


def _field_source_ref(
    page: Any,
    table: Any,
    *,
    row: int,
    column: int,
    field_name: str,
) -> dict[str, Any]:
    """Return the narrowest source reference available for one bound value."""

    ref = _source_ref(page, table)
    ref.update({"row": row, "column": column, "field_name": field_name, "binding": "label_column"})
    metadata = getattr(table, "metadata", None) or {}
    if isinstance(metadata, dict):
        ref["logical_page"] = int(metadata.get("source_logical_page") or ref["logical_page"])
        ref["source_page"] = int(metadata.get("source_page") or ref["source_page"])
    cell_boxes = (
        metadata.get("source_cell_bboxes") or metadata.get("cell_bboxes")
        if isinstance(metadata, dict)
        else None
    )
    if (
        isinstance(cell_boxes, list)
        and 0 <= row < len(cell_boxes)
        and isinstance(cell_boxes[row], list)
        and 0 <= column < len(cell_boxes[row])
        and isinstance(cell_boxes[row][column], (list, tuple))
        and len(cell_boxes[row][column]) == 4
    ):
        ref["source"] = "native_detail_tolerant_table_cell"
        ref["bbox"] = list(cell_boxes[row][column])
        ref["geometry_scope"] = "cell"
    evidence_ids = metadata.get("cell_evidence_ids") if isinstance(metadata, dict) else None
    if (
        isinstance(evidence_ids, list)
        and 0 <= row < len(evidence_ids)
        and isinstance(evidence_ids[row], list)
        and 0 <= column < len(evidence_ids[row])
        and isinstance(evidence_ids[row][column], list)
    ):
        ref["evidence_ids"] = [str(value) for value in evidence_ids[row][column] if value]
    return ref


def _table_top(table: Any) -> float:
    bbox = getattr(table, "bbox", None)
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        return float(bbox[1])
    value = getattr(bbox, "y0", None)
    return float(value) if isinstance(value, (int, float)) else 0.0


@dataclass(frozen=True)
class NativeLabeledRecord:
    dataset_name: str
    fields: dict[str, str]
    source_refs: tuple[dict[str, Any], ...]
    confidence: float
    source_refs_by_field: dict[str, tuple[dict[str, Any], ...]] = field(default_factory=dict)
    binding_quality_by_field: dict[str, str] = field(default_factory=dict)
    observed_labels: frozenset[str] = frozenset()
    unresolved_labels: frozenset[str] = frozenset()


@dataclass(frozen=True)
class _PositionedEvidenceCell:
    """One corrected-page OCR cell that keeps its canonical coordinates."""

    text: str
    bbox: tuple[float, float, float, float]
    source_bbox: tuple[float, float, float, float]
    logical_page: int
    source_page: int
    confidence: float = 0.0
    evidence_ids: tuple[str, ...] = ()

    @property
    def center_x(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2.0

    def source_ref(self, *, field_name: str) -> dict[str, Any]:
        ref: dict[str, Any] = {
            "source": "personal_detail_corrected_page_cell",
            "logical_page": self.logical_page,
            "source_page": self.source_page,
            "bbox": list(self.source_bbox),
            "geometry_scope": "cell",
            "field_name": field_name,
            "binding": "canonical_label_slot",
        }
        if self.evidence_ids:
            ref["evidence_ids"] = list(self.evidence_ids)
        return ref


def _cell_text(value: Any) -> str:
    return value.text if isinstance(value, _PositionedEvidenceCell) else str(value or "")


class PBOCPersonalDetailNativeParser:
    """Resolve labelled card observations into one canonical record stream."""

    def __init__(self, context: Any) -> None:
        self.context = context

    @staticmethod
    def _canonical_template_id(page: Any, table: Any) -> str:
        page_template = str(getattr(page, "canonical_template_id", "") or "")
        metadata = getattr(table, "metadata", None) or {}
        table_template = str(metadata.get("canonical_template_id") or "") if isinstance(metadata, dict) else ""
        return table_template or page_template

    @staticmethod
    def _printed_sequence_anchor_above_table(
        page: Any,
        table: Any,
        dataset_name: str,
    ) -> tuple[str, dict[str, Any] | None]:
        """Recover a card ordinal and its heading evidence, never row order."""
        table_box = getattr(table, "bbox", None)
        if not isinstance(table_box, (list, tuple)) or len(table_box) != 4:
            return "", None
        table_top = float(table_box[1])
        candidates: list[tuple[float, str, dict[str, Any]]] = []
        pattern = (
            re.compile(r"授信协议\s*(\d{1,3})")
            if dataset_name == "credit_lines"
            else re.compile(r"账户\s*(\d{1,3})")
        )
        for text_item in getattr(page, "texts", None) or ():
            match = pattern.search(str(getattr(text_item, "content", "") or ""))
            text_box = getattr(text_item, "bbox", None)
            if match is None or not isinstance(text_box, (list, tuple)) or len(text_box) != 4:
                continue
            bottom = float(text_box[3])
            if bottom <= table_top + 8.0:
                candidates.append(
                    (
                        bottom,
                        match.group(1),
                        {
                            "source": "native_detail_canonical_anchor_text",
                            "logical_page": int(getattr(page, "page_number", 0) or 0),
                            "source_page": int(
                                getattr(page, "source_page_number", 0)
                                or getattr(page, "page_number", 0)
                                or 0
                            ),
                            "bbox": list(text_box),
                            "geometry_scope": "text",
                            "field_name": "sequence",
                            "binding": "canonical_card_anchor",
                        },
                    )
                )
        if not candidates:
            return "", None
        _bottom, sequence, ref = max(candidates, key=lambda item: item[0])
        return sequence, ref

    def _table_groups(self) -> list[tuple[Any, Any, list[list[str]], tuple[dict[str, Any], ...]]]:
        entries: list[tuple[Any, Any, list[list[str]], dict[str, Any]]] = []
        reading_order = dict(getattr(self.context, "reading_order_by_logical", {}) or {})
        for page in getattr(self.context, "pages", None) or []:
            for table in getattr(page, "tables", None) or []:
                table_rows = _rows(table)
                if table_rows:
                    entries.append((page, table, table_rows, _source_ref(page, table)))
        entries.sort(
            key=lambda item: (
                reading_order.get(
                    int(getattr(item[0], "page_number", 0) or 0), int(getattr(item[0], "page_number", 0) or 0)
                ),
                _table_top(item[1]),
            )
        )
        groups: list[tuple[Any, Any, list[list[str]], tuple[dict[str, Any], ...]]] = []
        for page, table, table_rows, ref in entries:
            if groups:
                previous_page, previous_table, previous_rows, previous_refs = groups[-1]
                continuation = getattr(self.context, "tables_continue", None)
                continues = (
                    continuation(
                        str(getattr(previous_table, "table_id", "") or ""),
                        str(getattr(table, "table_id", "") or ""),
                    )
                    if callable(continuation)
                    else None
                )
                if continues is True:
                    groups[-1] = (
                        previous_page,
                        previous_table,
                        [*previous_rows, *table_rows],
                        (*previous_refs, ref),
                    )
                    continue
            groups.append((page, table, table_rows, (ref,)))
        return groups

    @staticmethod
    def _split_repeated_cards_with_offsets(
        dataset_name: str, rows: list[list[str]]
    ) -> list[tuple[int, list[list[str]]]]:
        """Split repeated labelled cards before converting labels to a dict.

        ``tables_continue`` identifies one continued table entity.  A continued
        entity can still contain several canonical PBOC records, so repeated
        record anchors/primary labels must open a new record rather than being
        collapsed by ``dict.setdefault`` in ``_pairs``.
        """
        primary_label = _PRIMARY_RECORD_LABELS.get(dataset_name)
        anchor = _EVIDENCE_ANCHORS.get(dataset_name)
        if not primary_label or anchor is None:
            return [(0, rows)]

        groups: list[tuple[int, list[list[str]]]] = []
        current: list[list[str]] = []
        current_start = 0
        current_has_primary = False
        for row_index, row in enumerate(rows):
            compact = _compact("".join(str(cell or "") for cell in row))
            has_anchor = bool(anchor.search(compact))
            has_primary = primary_label in compact
            if current and (has_anchor or (has_primary and current_has_primary)):
                groups.append((current_start, current))
                current = []
                current_start = row_index
                current_has_primary = False
            current.append(row)
            current_has_primary = current_has_primary or has_primary
        if current:
            groups.append((current_start, current))
        return groups

    @classmethod
    def _split_repeated_cards(cls, dataset_name: str, rows: list[list[str]]) -> list[list[list[str]]]:
        return [group for _offset, group in cls._split_repeated_cards_with_offsets(dataset_name, rows)]

    @staticmethod
    def _pairs_with_bindings(
        rows: list[list[Any]],
    ) -> tuple[dict[str, str], float, dict[str, tuple[int, int]], frozenset[str], frozenset[str]]:
        """Decode only explicit inline or same-column label/value bindings.

        A previous fallback walked rightward until it found any non-label cell.
        That turns a missing cell into a column shift and can silently bind a
        perfectly valid value to the wrong PBOC field.  Canonical tables use a
        label row followed by its value row, so unequal/merged grids are now
        withheld for the complete-page evidence pass instead of guessed.
        """

        fields: dict[str, str] = {}
        scores: list[float] = []
        positions: dict[str, tuple[int, int]] = {}
        observed: set[str] = set()
        unresolved: set[str] = set()
        for row_index, row in enumerate(rows):
            for column, cell in enumerate(row):
                cell_text = _cell_text(cell)
                label, score = _canonical_label(cell_text)
                if label is None:
                    inline = re.match(r"^\s*([^:：]{2,30})[:：]\s*(.+?)\s*$", cell_text)
                    if inline:
                        label, score = _canonical_label(inline.group(1))
                        if label and inline.group(2).strip():
                            candidate = inline.group(2).strip()
                            if label in fields and _compact(fields[label]) != _compact(candidate):
                                fields.pop(label, None)
                                positions.pop(label, None)
                                unresolved.add(label)
                                continue
                            fields.setdefault(label, candidate)
                            positions.setdefault(label, (row_index, column))
                            observed.add(label)
                            scores.append(score)
                    continue
                observed.add(label)
                value = ""
                value_position: tuple[int, int] | None = None
                if row_index + 1 < len(rows) and len(rows[row_index + 1]) == len(row):
                    below = _cell_text(rows[row_index + 1][column]).strip()
                    below_label, _below_score = _canonical_label(below)
                    if below and below_label is None:
                        value = below
                        value_position = (row_index + 1, column)
                if value:
                    if label in fields and _compact(fields[label]) != _compact(value):
                        fields.pop(label, None)
                        positions.pop(label, None)
                        unresolved.add(label)
                        continue
                    fields.setdefault(label, value)
                    positions.setdefault(label, value_position or (row_index, column))
                    scores.append(score)
                else:
                    unresolved.add(label)
        unresolved.difference_update(fields)
        return (
            fields,
            min(scores) if scores else 0.0,
            positions,
            frozenset(observed),
            frozenset(unresolved),
        )

    @classmethod
    def _pairs(cls, rows: list[list[Any]]) -> tuple[dict[str, str], float]:
        fields, confidence, _positions, _observed, _unresolved = cls._pairs_with_bindings(rows)
        return fields, confidence

    @staticmethod
    def _ocr_positioned_rows(page: dict[str, Any]) -> list[list[_PositionedEvidenceCell]]:
        positioned: list[tuple[float, float, float, _PositionedEvidenceCell]] = []
        logical_page = int(page.get("page") or page.get("logical_page") or 0)
        source_page = int(page.get("source_page") or logical_page)
        for line in page.get("lines") or []:
            if not isinstance(line, dict):
                continue
            bbox = line.get("bbox")
            text = str(line.get("text") or "").strip()
            if not text or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            box = tuple(float(value) for value in bbox)
            raw_source_box = line.get("source_bbox")
            source_box = (
                tuple(float(value) for value in raw_source_box)
                if isinstance(raw_source_box, (list, tuple)) and len(raw_source_box) == 4
                else box
            )
            positioned.append(
                (
                    (box[1] + box[3]) / 2.0,
                    box[0],
                    max(1.0, box[3] - box[1]),
                    _PositionedEvidenceCell(
                        text=text,
                        bbox=box,
                        source_bbox=source_box,
                        logical_page=int(line.get("source_logical_page") or line.get("page") or logical_page),
                        source_page=int(line.get("source_page") or source_page),
                        confidence=float(line.get("confidence") or 0.0),
                        evidence_ids=tuple(str(value) for value in line.get("evidence_ids") or () if value),
                    ),
                )
            )
        rows: list[list[tuple[float, _PositionedEvidenceCell]]] = []
        centers: list[float] = []
        heights: list[float] = []
        for center, left, height, cell in sorted(positioned, key=lambda item: (item[0], item[1])):
            if not rows or abs(center - centers[-1]) > max(6.0, height * 0.75, heights[-1] * 0.75):
                rows.append([(left, cell)])
                centers.append(center)
                heights.append(height)
            else:
                rows[-1].append((left, cell))
                count = len(rows[-1])
                centers[-1] = ((centers[-1] * (count - 1)) + center) / count
                heights[-1] = max(heights[-1], height)
        return [[cell for _left, cell in sorted(row, key=lambda item: item[0])] for row in rows]

    @classmethod
    def _ocr_rows(cls, page: dict[str, Any]) -> list[list[str]]:
        return [[cell.text for cell in row] for row in cls._ocr_positioned_rows(page)]

    def _evidence_record_groups(
        self,
        dataset_name: str,
    ) -> list[tuple[list[list[_PositionedEvidenceCell]], tuple[dict[str, Any], ...]]]:
        """Segment repeated PBOC cards from corrected logical-page rows."""
        loader = getattr(self.context, "corrected_evidence_pages", None)
        if not callable(loader):
            return []
        pages = loader() or []
        if dataset_name == "report_header":
            if not pages:
                return []
            page = min(pages, key=lambda item: int(item.get("page") or 0))
            logical_page = int(page.get("page") or 0)
            source_page = int(page.get("source_page") or logical_page)
            rows = self._ocr_positioned_rows(page)
            return [
                (
                    rows,
                    (
                        {
                            "source": "personal_detail_corrected_page_rows",
                            "logical_page": logical_page,
                            "source_page": source_page,
                            "geometry_scope": "logical_page",
                        },
                    ),
                )
            ] if rows else []

        heading = _EVIDENCE_SECTION_HEADINGS.get(dataset_name)
        anchor = _EVIDENCE_ANCHORS.get(dataset_name)
        primary_label = _PRIMARY_RECORD_LABELS.get(dataset_name, "")
        if not heading or anchor is None:
            return []
        groups: list[tuple[list[list[_PositionedEvidenceCell]], tuple[dict[str, Any], ...]]] = []
        active = False
        current_rows: list[list[_PositionedEvidenceCell]] = []
        current_has_primary = False
        current_refs: list[dict[str, Any]] = []
        referenced_pages: set[tuple[int, int]] = set()
        liability_party_category = ""

        def flush() -> None:
            nonlocal current_rows, current_has_primary, current_refs, referenced_pages
            has_explicit_anchor = any(
                anchor.search(_compact("".join(_cell_text(cell) for cell in row)))
                for row in current_rows
            )
            if current_rows and (dataset_name != "credit_lines" or has_explicit_anchor):
                groups.append((current_rows, tuple(current_refs)))
            current_rows = []
            current_has_primary = False
            current_refs = []
            referenced_pages = set()

        for page in pages:
            logical_page = int(page.get("page") or 0)
            source_page = int(page.get("source_page") or 0)
            page_key = (logical_page, source_page)
            page_ref = {
                "source": "personal_detail_corrected_page_rows",
                "logical_page": logical_page,
                "source_page": source_page,
                "geometry_scope": "logical_page",
            }
            for row in self._ocr_positioned_rows(page):
                compact = _compact("".join(_cell_text(cell) for cell in row))
                if heading in compact:
                    flush()
                    active = True
                    continue
                other_heading = next(
                    (
                        value
                        for name, value in _EVIDENCE_SECTION_HEADINGS.items()
                        if name != dataset_name and value in compact
                    ),
                    None,
                )
                if active and (other_heading or any(marker in compact for marker in _EVIDENCE_SECTION_END_MARKERS)):
                    flush()
                    active = False
                    continue
                if not active:
                    continue
                if dataset_name == "repayment_liability_records":
                    if "有相关还款责任的个人借款" in compact:
                        flush()
                        liability_party_category = "person"
                        continue
                    if "有相关还款责任的企业借款" in compact:
                        flush()
                        liability_party_category = "organization"
                        continue
                has_primary = bool(primary_label and primary_label in compact)
                if anchor.search(compact) or (has_primary and current_has_primary):
                    flush()
                    current_rows = [row]
                elif current_rows:
                    current_rows.append(row)
                current_has_primary = current_has_primary or has_primary
                if current_rows and page_key not in referenced_pages:
                    current_refs.append(
                        {
                            **page_ref,
                            **(
                                {"canonical_party_category": liability_party_category}
                                if liability_party_category
                                else {}
                            ),
                        }
                    )
                    referenced_pages.add(page_key)
        flush()
        return groups

    @staticmethod
    def _record_identity(dataset_name: str, fields: dict[str, str]) -> tuple[str, ...]:
        if dataset_name == "credit_lines":
            return (_compact(fields.get("授信协议标识")),)
        if dataset_name == "report_header":
            return (
                _compact(fields.get("被查询者姓名")),
                _compact(fields.get("被查询者证件号码")),
                _compact(fields.get("查询机构")),
            )
        return (
            _compact(fields.get("保证合同编号")),
            _compact(fields.get("管理机构")),
            _compact(fields.get("开立日期")),
            _compact(fields.get("还款责任金额")),
        )

    @staticmethod
    def _standalone_positioned_label(cell: _PositionedEvidenceCell) -> tuple[str | None, float]:
        text = _compact(cell.text)
        if text in _LABELS:
            return text, 1.0
        contained = [label for label in _LABELS if label in text]
        if len(contained) > 1:
            return None, 0.0
        label, score = _canonical_label(cell.text)
        if label is None:
            return None, score
        # A label token may carry minor punctuation/OCR loss, but it must not be
        # a whole row whose substring happens to contain this label.
        if len(text) > len(_compact(label)) + 2:
            return None, score
        return label, score

    @classmethod
    def _positioned_pairs(
        cls,
        rows: list[list[_PositionedEvidenceCell]],
    ) -> tuple[
        dict[str, str],
        float,
        dict[str, tuple[dict[str, Any], ...]],
        dict[str, str],
        frozenset[str],
        frozenset[str],
    ]:
        """Build a label-to-cell graph over canonical corrected-page boxes."""

        values_by_label: dict[str, list[tuple[str, tuple[_PositionedEvidenceCell, ...], float]]] = {}
        observed: set[str] = set()
        unresolved: set[str] = set()
        for row_index, row in enumerate(rows):
            labels: list[tuple[str, float, _PositionedEvidenceCell]] = []
            for cell in row:
                inline = re.match(r"^\s*([^:：]{2,30})[:：]\s*(.+?)\s*$", cell.text)
                if inline:
                    label, score = _canonical_label(inline.group(1))
                    if label:
                        observed.add(label)
                        values_by_label.setdefault(label, []).append(
                            (inline.group(2).strip(), (cell,), score)
                        )
                    continue
                label, score = cls._standalone_positioned_label(cell)
                if label:
                    labels.append((label, score, cell))
                    observed.add(label)
            if not labels:
                continue

            labels.sort(key=lambda item: item[2].center_x)
            centers = [item[2].center_x for item in labels]
            boundaries = [float("-inf")]
            boundaries.extend((left + right) / 2.0 for left, right in zip(centers, centers[1:], strict=False))
            boundaries.append(float("inf"))

            value_rows: list[list[_PositionedEvidenceCell]] = []
            for candidate_row in rows[row_index + 1 :]:
                if any(
                    re.search(
                        r"截至\s*(?:19|20)\d{2}\s*[.年/-]\s*\d{1,2}\s*[.月/-]\s*\d{1,2}",
                        cell.text,
                    )
                    for cell in candidate_row
                ):
                    # The liability snapshot date is canonical record
                    # furniture, not a wrapped value belonging to the label
                    # row immediately above it.
                    break
                candidate_labels = [
                    cls._standalone_positioned_label(cell)[0]
                    for cell in candidate_row
                ]
                if any(candidate_labels):
                    break
                # Card headings and section furniture are not values.  Once a
                # new printed record starts, this label row is unresolved.
                candidate_text = _compact("".join(cell.text for cell in candidate_row))
                if any(anchor.search(candidate_text) for anchor in _EVIDENCE_ANCHORS.values()):
                    break
                value_rows.append(candidate_row)
                # Wrapped institution/identifier cells occupy at most two OCR
                # baselines before the next canonical label row.
                if len(value_rows) >= 2:
                    break

            for slot_index, (label, score, _label_cell) in enumerate(labels):
                slot_cells: list[_PositionedEvidenceCell] = []
                for candidate_row in value_rows:
                    for cell in candidate_row:
                        if not (boundaries[slot_index] <= cell.center_x < boundaries[slot_index + 1]):
                            continue
                        # A token spanning another column centre is a merged row,
                        # not uniquely bound cell evidence.
                        if any(
                            cell.bbox[0] < other_center < cell.bbox[2]
                            for other_index, other_center in enumerate(centers)
                            if other_index != slot_index
                        ):
                            continue
                        if cls._standalone_positioned_label(cell)[0] is None:
                            slot_cells.append(cell)
                if not slot_cells:
                    unresolved.add(label)
                    continue
                ordered = tuple(sorted(slot_cells, key=lambda item: (item.bbox[1], item.bbox[0])))
                value = "".join(cell.text.strip() for cell in ordered if cell.text.strip()).strip()
                if value:
                    values_by_label.setdefault(label, []).append((value, ordered, score))
                else:
                    unresolved.add(label)

        fields: dict[str, str] = {}
        refs_by_field: dict[str, tuple[dict[str, Any], ...]] = {}
        binding_by_field: dict[str, str] = {}
        scores: list[float] = []
        for label, observations in values_by_label.items():
            distinct = {_compact(value) for value, _cells, _score in observations}
            if len(distinct) != 1:
                unresolved.add(label)
                continue
            value = observations[0][0]
            fields[label] = value
            cells = tuple(cell for _value, selected, _score in observations for cell in selected)
            refs_by_field[label] = tuple(cell.source_ref(field_name=label) for cell in cells)
            binding_by_field[label] = "canonical_cell_slot"
            scores.extend(score for _value, _cells, score in observations)
        unresolved.difference_update(fields)
        return (
            fields,
            min(scores) if scores else 0.0,
            refs_by_field,
            binding_by_field,
            frozenset(observed),
            frozenset(unresolved),
        )

    @classmethod
    def _evidence_fields(
        cls,
        dataset_name: str,
        rows: list[list[_PositionedEvidenceCell]],
    ) -> tuple[
        dict[str, str],
        float,
        dict[str, tuple[dict[str, Any], ...]],
        dict[str, str],
        frozenset[str],
        frozenset[str],
    ]:
        fields, confidence, refs, bindings, observed, unresolved = cls._positioned_pairs(rows)
        if dataset_name in {"credit_lines", "repayment_liability_records"}:
            sequence_pattern = (
                re.compile(r"授信协议\s*(\d{1,3})")
                if dataset_name == "credit_lines"
                else re.compile(r"账户\s*(\d{1,3})")
            )
            sequence_cells = [
                (match.group(1), cell)
                for row in rows
                for cell in row
                if (match := sequence_pattern.search(cell.text))
            ]
            printed_sequences = {value for value, _cell in sequence_cells}
            if len(printed_sequences) == 1:
                printed_sequence = next(iter(printed_sequences))
                fields["__printed_sequence"] = printed_sequence
                refs["__printed_sequence"] = tuple(
                    {
                        **cell.source_ref(field_name="sequence"),
                        "binding": "canonical_card_anchor",
                    }
                    for value, cell in sequence_cells
                    if value == printed_sequence
                )
                bindings["__printed_sequence"] = "canonical_card_anchor"
            elif len(printed_sequences) > 1:
                unresolved = frozenset({*unresolved, "__printed_sequence"})
        if dataset_name == "repayment_liability_records":
            snapshot_cells = [
                cell
                for row in rows
                for cell in row
                if re.search(r"截至\s*(?:19|20)\d{2}\s*[.年/-]\s*\d{1,2}\s*[.月/-]\s*\d{1,2}", cell.text)
            ]
            snapshot_values = {_compact(cell.text) for cell in snapshot_cells}
            if len(snapshot_values) == 1:
                fields["__snapshot_date"] = snapshot_cells[0].text
                refs["__snapshot_date"] = tuple(
                    cell.source_ref(field_name="snapshot_date") for cell in snapshot_cells
                )
                bindings["__snapshot_date"] = "canonical_snapshot_date_cell"
            elif len(snapshot_values) > 1:
                unresolved = frozenset({*unresolved, "__snapshot_date"})
        return fields, confidence, refs, bindings, observed, unresolved

    def records(self, dataset_name: str) -> list[NativeLabeledRecord]:
        required = _SECTION_MARKERS[dataset_name]
        result: list[NativeLabeledRecord] = []
        for _page, _table, rows, refs in self._table_groups():
            template_id = self._canonical_template_id(_page, _table)
            if dataset_name == "credit_lines" and template_id and template_id != "credit_agreement":
                # Account-detail cards also print ``授信协议标识`` in their
                # heading.  They are account evidence, not agreement rows.
                continue
            for row_offset, record_rows in self._split_repeated_cards_with_offsets(dataset_name, rows):
                fields, confidence, positions, observed_labels, unresolved = self._pairs_with_bindings(record_rows)
                refs_by_field = {
                    label: (
                        _field_source_ref(
                            _page,
                            _table,
                            row=row_offset + position[0],
                            column=position[1],
                            field_name=label,
                        ),
                    )
                    for label, position in positions.items()
                }
                bindings_by_field = {label: "native_label_column" for label in positions}
                if dataset_name in {"credit_lines", "repayment_liability_records"}:
                    sequence_pattern = (
                        re.compile(r"授信协议\s*(\d{1,3})")
                        if dataset_name == "credit_lines"
                        else re.compile(r"账户\s*(\d{1,3})")
                    )
                    sequence_cells = [
                        (match.group(1), row_index, column_index)
                        for row_index, row in enumerate(record_rows)
                        for column_index, cell in enumerate(row)
                        if (match := sequence_pattern.search(str(cell or "")))
                    ]
                    printed_sequences = {value for value, _row, _column in sequence_cells}
                    if len(printed_sequences) == 1:
                        printed_sequence = next(iter(printed_sequences))
                        fields["__printed_sequence"] = printed_sequence
                        refs_by_field["__printed_sequence"] = tuple(
                            {
                                **_field_source_ref(
                                    _page,
                                    _table,
                                    row=row_offset + sequence_row,
                                    column=sequence_column,
                                    field_name="sequence",
                                ),
                                "binding": "canonical_card_anchor",
                            }
                            for value, sequence_row, sequence_column in sequence_cells
                            if value == printed_sequence
                        )
                        bindings_by_field["__printed_sequence"] = "canonical_card_anchor"
                    elif len(printed_sequences) > 1:
                        unresolved = frozenset({*unresolved, "__printed_sequence"})
                    else:
                        printed_sequence, anchor_ref = self._printed_sequence_anchor_above_table(
                            _page,
                            _table,
                            dataset_name,
                        )
                        if printed_sequence and anchor_ref:
                            fields["__printed_sequence"] = printed_sequence
                            refs_by_field["__printed_sequence"] = (anchor_ref,)
                            bindings_by_field["__printed_sequence"] = "canonical_card_anchor"
                if dataset_name == "repayment_liability_records":
                    snapshot_positions = [
                        (row_index, column_index, str(cell or ""))
                        for row_index, row in enumerate(record_rows)
                        for column_index, cell in enumerate(row)
                        if re.search(
                            r"截至\s*(?:19|20)\d{2}\s*[.年/-]\s*\d{1,2}\s*[.月/-]\s*\d{1,2}",
                            str(cell or ""),
                        )
                    ]
                    snapshot_values = {_compact(value) for _row, _column, value in snapshot_positions}
                    if len(snapshot_values) == 1:
                        snapshot_row, snapshot_column, snapshot_value = snapshot_positions[0]
                        fields["__snapshot_date"] = snapshot_value
                        refs_by_field["__snapshot_date"] = (
                            _field_source_ref(
                                _page,
                                _table,
                                row=row_offset + snapshot_row,
                                column=snapshot_column,
                                field_name="snapshot_date",
                            ),
                        )
                        bindings_by_field["__snapshot_date"] = "canonical_snapshot_date_cell"
                    elif len(snapshot_values) > 1:
                        unresolved = frozenset({*unresolved, "__snapshot_date"})
                observed_fields = set(fields)
                label_text = _compact("".join(cell for row in record_rows for cell in row))
                marker_hits = {marker for marker in required if marker in observed_fields or marker in label_text}
                identified_credit_agreement = bool(
                    dataset_name == "credit_lines"
                    and (fields.get("授信协议标识") or "授信协议标识" in label_text)
                )
                if len(marker_hits) < max(2, len(required) - 1) and not identified_credit_agreement:
                    # A partial canonical label set is evidence of a damaged
                    # business card, not permission to guess which field an
                    # unknown string resembles.  Keep the card out of business
                    # data and make the page eligible for its coordinated OCR
                    # repair pass.  The higher threshold on report headers
                    # avoids mistaking ordinary inquiry tables for a header.
                    partial_threshold = max(1, len(required) - 1)
                    if len(marker_hits) >= partial_threshold:
                        record_issue(
                            self.context,
                            make_issue(
                                category="ocr_structure_correction",
                                issue_code="canonical_label_authorization_unresolved",
                                message=(
                                    "A canonical table was only partially labelled; unknown or damaged "
                                    "labels were withheld instead of being fuzzy-matched to business fields."
                                ),
                                parser_stage="native_canonical_label_authorization",
                                target_dataset=(
                                    dataset_name
                                    if dataset_name != "report_header"
                                    else "personal_report_metadata"
                                ),
                                observed_value={
                                    "observed_labels": sorted(observed_labels),
                                    "missing_labels": sorted(required - observed_fields),
                                },
                                confidence=confidence or None,
                                source_refs=refs,
                                reason_codes=(
                                    "partial_canonical_label_set",
                                    "unknown_label_not_authorized",
                                    "schema_triggered_page_repair_eligible",
                                    "no_guess_applied",
                                ),
                            ),
                        )
                    continue
                missing = required - observed_fields
                if missing:
                    if dataset_name not in {"credit_lines", "repayment_liability_records"}:
                        record_issue(
                            self.context,
                            make_issue(
                                category="ocr_structure_correction",
                                issue_code="recognized_native_section_missing_required_value",
                                message="A PBOC native table section was recognized but a required labelled value was not recoverable.",
                                parser_stage="native_tolerant_parser",
                                target_dataset=dataset_name if dataset_name != "report_header" else "personal_report_metadata",
                                observed_value={"observed_labels": sorted(observed_fields), "missing_labels": sorted(missing)},
                                confidence=confidence or None,
                                source_refs=refs,
                                reason_codes=("section_anchor_recognized", "required_value_missing", "no_guess_applied"),
                            ),
                        )
                    # Agreement fields are reconciled across all native and
                    # complete-page observations before the final gate.  Do
                    # not publish a transient missing-field issue here.
                    if dataset_name == "credit_lines" and not fields.get("授信协议标识"):
                        continue
                    if dataset_name == "repayment_liability_records" and not (
                        fields.get("保证合同编号") or fields.get("还款责任金额")
                    ):
                        continue
                result.append(
                    NativeLabeledRecord(
                        dataset_name=dataset_name,
                        fields=fields,
                        source_refs=refs,
                        confidence=confidence,
                        source_refs_by_field=refs_by_field,
                        binding_quality_by_field=bindings_by_field,
                        observed_labels=observed_labels,
                        unresolved_labels=unresolved,
                    )
                )
        seen = {self._record_identity(dataset_name, item.fields) for item in result}
        for rows, refs in self._evidence_record_groups(dataset_name):
            (
                fields,
                confidence,
                refs_by_field,
                bindings_by_field,
                observed,
                unresolved,
            ) = self._evidence_fields(dataset_name, rows)
            if dataset_name == "repayment_liability_records":
                party_categories = {
                    str(ref.get("canonical_party_category") or "")
                    for ref in refs
                    if ref.get("canonical_party_category")
                }
                if len(party_categories) == 1:
                    fields["__party_category"] = next(iter(party_categories))
            missing = required - set(fields)
            if missing:
                observed_text = _compact("".join(_cell_text(cell) for row in rows for cell in row))
                primary_label = _PRIMARY_RECORD_LABELS.get(dataset_name, "")
                anchor = _EVIDENCE_ANCHORS.get(dataset_name)
                is_record_observation = bool(
                    dataset_name == "report_header"
                    or (primary_label and primary_label in observed_text)
                    or (anchor is not None and anchor.search(observed_text))
                )
                if is_record_observation and dataset_name not in {
                    "credit_lines",
                    "repayment_liability_records",
                }:
                    record_issue(
                        self.context,
                        make_issue(
                            category="ocr_structure_correction",
                            issue_code="corrected_page_record_missing_required_value",
                            message=(
                                "A canonical record was visible in complete-page evidence but one or more "
                                "required fields could not be decoded."
                            ),
                            parser_stage="candidate_b_corrected_page_record_decoder",
                            target_dataset=(
                                dataset_name if dataset_name != "report_header" else "personal_report_metadata"
                            ),
                            observed_value={"observed_labels": sorted(fields), "missing_labels": sorted(missing)},
                            confidence=confidence or None,
                            source_refs=refs,
                            reason_codes=(
                                "canonical_record_anchor_observed",
                                "required_value_missing",
                                "schema_triggered_page_repair_eligible",
                                "record_not_invented",
                            ),
                        ),
                    )
                if dataset_name == "credit_lines" and not fields.get("授信协议标识"):
                    continue
                if dataset_name == "repayment_liability_records" and not (
                    fields.get("保证合同编号") or fields.get("还款责任金额")
                ):
                    continue
            identity = self._record_identity(dataset_name, fields)
            if not any(identity) or (identity in seen and dataset_name != "credit_lines"):
                continue
            seen.add(identity)
            result.append(
                NativeLabeledRecord(
                    dataset_name=dataset_name,
                    fields=fields,
                    source_refs=refs,
                    confidence=confidence,
                    source_refs_by_field=refs_by_field,
                    binding_quality_by_field=bindings_by_field,
                    observed_labels=observed,
                    unresolved_labels=unresolved,
                )
            )
        if not result:
            evidence_loader = getattr(self.context, "corrected_evidence_pages", None)
            evidence_pages = evidence_loader() if callable(evidence_loader) else []
            candidate_pages: set[int] = set()
            for page in evidence_pages:
                text = _compact("".join(str(line.get("text") or "") for line in page.get("lines") or []))
                if sum(marker in text for marker in required) >= max(2, len(required) - 1):
                    candidate_pages.add(int(page.get("page") or 0))
            if candidate_pages:
                candidate_refs = tuple(
                    {
                        "source": "candidate_b_visible_unparsed_section",
                        "logical_page": page,
                        "geometry_scope": "logical_page",
                    }
                    for page in sorted(candidate_pages)
                )
                record_issue(
                    self.context,
                    make_issue(
                        category="ocr_structure_correction",
                        issue_code="recognized_native_section_not_extracted",
                        message=(
                            "The source section was visible in page evidence but absent from native tables; "
                            "schema-triggered page repair may retry the section once."
                        ),
                        parser_stage="native_tolerant_parser",
                        target_dataset=dataset_name if dataset_name != "report_header" else "personal_report_metadata",
                        source_refs=candidate_refs,
                        reason_codes=("native_table_missing", "page_anchor_observed", "business_data_uncertain"),
                    ),
                )
        return result


__all__ = ["NativeLabeledRecord", "PBOCPersonalDetailNativeParser"]
