# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tolerant native table grammar for PBOC personal detailed reports.

This is a fallback grammar, not an alternate source of truth.  Exact legacy
parsers run first; this parser is used only when an expected dataset or field
is missing.  Fuzzy labels require a unique high-margin match and never cause a
business value to be guessed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
    make_issue,
    record_issue,
)

_LABELS = frozenset(
    {
        "被查询者姓名", "被查询者证件类型", "被查询者证件号码", "查询机构", "查询原因",
        "证件类型", "证件号码", "授信协议标识", "管理机构", "授信额度用途", "生效日期",
        "到期日期", "授信额度", "已用额度", "授信限额编号", "币种", "责任人类型",
        "还款责任金额", "保证合同编号", "主业务借款人", "主业务借款人证件类型",
        "主业务借款人证件号码", "开立日期", "业务种类", "余额", "五级分类", "逾期月数",
        "还款状态",
    }
)

_SECTION_MARKERS = {
    "credit_lines": frozenset({"授信协议标识", "授信额度用途"}),
    "repayment_liability_records": frozenset({"责任人类型", "保证合同编号"}),
    "report_header": frozenset({"被查询者姓名", "被查询者证件类型", "被查询者证件号码", "查询机构", "查询原因"}),
}


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
    contained = [label for label in _LABELS if label in text or text in label]
    if len(contained) == 1 and min(len(text), len(contained[0])) >= 3:
        return contained[0], 0.94
    scored = sorted(
        ((SequenceMatcher(None, text, label).ratio(), label) for label in _LABELS if min(len(text), len(label)) >= 4),
        reverse=True,
    )
    if not scored:
        return None, 0.0
    best_score, best = scored[0]
    runner = scored[1][0] if len(scored) > 1 else 0.0
    if best_score >= 0.90 and best_score - runner >= 0.08:
        return best, best_score
    return None, best_score


def _source_ref(page: Any, table: Any) -> dict[str, Any]:
    ref: dict[str, Any] = {
        "source": "native_detail_tolerant_table",
        "logical_page": int(getattr(page, "page_number", 0) or 0),
        "source_page": int(
            getattr(page, "source_page_number", 0) or getattr(page, "page_number", 0) or 0
        ),
        "table_id": str(getattr(table, "table_id", "") or ""),
    }
    bbox = getattr(table, "bbox", None)
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        ref["bbox"] = list(bbox)
        ref["geometry_scope"] = "table"
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


class PBOCPersonalDetailNativeParser:
    """Parse labelled native table fragments using sealed table evidence."""

    def __init__(self, context: Any) -> None:
        self.context = context

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
                reading_order.get(int(getattr(item[0], "page_number", 0) or 0), int(getattr(item[0], "page_number", 0) or 0)),
                _table_top(item[1]),
            )
        )
        groups: list[tuple[Any, Any, list[list[str]], tuple[dict[str, Any], ...]]] = []
        for page, table, table_rows, ref in entries:
            if groups:
                previous_page, previous_table, previous_rows, previous_refs = groups[-1]
                continuation = getattr(self.context, "tables_continue", None)
                continues = continuation(
                    str(getattr(previous_table, "table_id", "") or ""),
                    str(getattr(table, "table_id", "") or ""),
                ) if callable(continuation) else None
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
    def _pairs(rows: list[list[str]]) -> tuple[dict[str, str], float]:
        fields: dict[str, str] = {}
        scores: list[float] = []
        for row_index, row in enumerate(rows):
            for column, cell in enumerate(row):
                label, score = _canonical_label(cell)
                if label is None:
                    inline = re.match(r"^\s*([^:：]{2,30})[:：]\s*(.+?)\s*$", str(cell or ""))
                    if inline:
                        label, score = _canonical_label(inline.group(1))
                        if label and inline.group(2).strip():
                            fields.setdefault(label, inline.group(2).strip())
                            scores.append(score)
                    continue
                value = ""
                if row_index + 1 < len(rows) and column < len(rows[row_index + 1]):
                    below = str(rows[row_index + 1][column] or "").strip()
                    below_label, _below_score = _canonical_label(below)
                    if below and below_label is None:
                        value = below
                if not value:
                    for right in row[column + 1 :]:
                        right_text = str(right or "").strip()
                        right_label, _right_score = _canonical_label(right_text)
                        if right_text and right_label is None:
                            value = right_text
                            break
                if value:
                    fields.setdefault(label, value)
                    scores.append(score)
        return fields, min(scores) if scores else 0.0

    @staticmethod
    def _ocr_rows(page: dict[str, Any]) -> list[list[str]]:
        positioned: list[tuple[float, float, float, str]] = []
        for line in page.get("lines") or []:
            if not isinstance(line, dict):
                continue
            bbox = line.get("bbox")
            text = str(line.get("text") or "").strip()
            if not text or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            positioned.append(
                (
                    (float(bbox[1]) + float(bbox[3])) / 2.0,
                    float(bbox[0]),
                    max(1.0, float(bbox[3]) - float(bbox[1])),
                    text,
                )
            )
        rows: list[list[tuple[float, str]]] = []
        centers: list[float] = []
        heights: list[float] = []
        for center, left, height, text in sorted(positioned):
            if not rows or abs(center - centers[-1]) > max(6.0, height * 0.75, heights[-1] * 0.75):
                rows.append([(left, text)])
                centers.append(center)
                heights.append(height)
            else:
                rows[-1].append((left, text))
                count = len(rows[-1])
                centers[-1] = ((centers[-1] * (count - 1)) + center) / count
                heights[-1] = max(heights[-1], height)
        return [[text for _left, text in sorted(row)] for row in rows]

    def _full_page_fields(
        self,
        pages: set[int],
        *,
        dataset_name: str,
    ) -> tuple[dict[str, str], tuple[dict[str, Any], ...], float]:
        loader = getattr(self.context, "full_page_ocr_evidence", None)
        if not callable(loader) or not pages:
            return {}, (), 0.0
        evidence_pages = loader(
            pages,
            reason=f"native_parser_missing_required_value:{dataset_name}",
        )
        candidates: list[tuple[dict[str, str], dict[str, Any], float]] = []
        for page in evidence_pages:
            fields, confidence = self._pairs(self._ocr_rows(page))
            if fields:
                candidates.append(
                    (
                        fields,
                        {
                            "source": "personal_detail_full_page_ocr",
                            "logical_page": int(page.get("page") or 0),
                            "source_page": int(page.get("source_page") or 0),
                            "geometry_scope": "logical_page",
                        },
                        confidence,
                    )
                )
        if not candidates:
            return {}, (), 0.0
        merged: dict[str, str] = {}
        ambiguous: set[str] = set()
        refs: list[dict[str, Any]] = []
        confidences: list[float] = []
        for fields, ref, confidence in candidates:
            refs.append(ref)
            if confidence:
                confidences.append(confidence)
            for label, value in fields.items():
                if label in merged and merged[label] != value:
                    ambiguous.add(label)
                else:
                    merged.setdefault(label, value)
        for label in ambiguous:
            merged.pop(label, None)
        return merged, tuple(refs), min(confidences) if confidences else 0.0

    def _supplemental_fields(
        self,
        source_pages: set[int],
        *,
        dataset_name: str,
    ) -> tuple[dict[str, str], tuple[dict[str, Any], ...], float]:
        loader = getattr(self.context, "supplemental_page_ocr_evidence", None)
        if not callable(loader) or not source_pages:
            return {}, (), 0.0
        evidence_pages = loader(
            source_pages,
            reason=f"native_parser_missing_spread_continuation:{dataset_name}",
        )
        merged: dict[str, str] = {}
        ambiguous: set[str] = set()
        refs: list[dict[str, Any]] = []
        confidences: list[float] = []
        for page in evidence_pages:
            fields, confidence = self._pairs(self._ocr_rows(page))
            if not fields:
                continue
            refs.append(
                {
                    "source": "personal_detail_supplemental_page_ocr",
                    "source_page": int(page.get("source_page") or 0),
                    "source_segment_index": int(page.get("segment_index") or 0),
                    "printed_page": int(page.get("printed_page") or 0),
                    "supplemental_page_id": str(page.get("supplemental_page_id") or ""),
                    "geometry_scope": "supplemental_logical_page",
                }
            )
            if confidence:
                confidences.append(confidence)
            for label, value in fields.items():
                if label in merged and merged[label] != value:
                    ambiguous.add(label)
                else:
                    merged.setdefault(label, value)
        for label in ambiguous:
            merged.pop(label, None)
        return merged, tuple(refs), min(confidences) if confidences else 0.0

    def records(self, dataset_name: str) -> list[NativeLabeledRecord]:
        required = _SECTION_MARKERS[dataset_name]
        result: list[NativeLabeledRecord] = []
        for _page, _table, rows, refs in self._table_groups():
            fields, confidence = self._pairs(rows)
            observed = set(fields)
            label_text = _compact("".join(cell for row in rows for cell in row))
            marker_hits = {
                marker for marker in required if marker in observed or marker in label_text
            }
            if len(marker_hits) < max(2, len(required) - 1):
                continue
            missing = required - observed
            if missing:
                pages = {int(ref.get("logical_page") or 0) for ref in refs if ref.get("logical_page")}
                recovered, recovered_refs, recovered_confidence = self._full_page_fields(
                    pages,
                    dataset_name=dataset_name,
                )
                combined = {**recovered, **fields}
                supplemental_refs: tuple[dict[str, Any], ...] = ()
                supplemental_confidence = 0.0
                if not required <= set(combined):
                    source_pages = {
                        int(ref.get("source_page") or 0) for ref in refs if ref.get("source_page")
                    }
                    supplemental, supplemental_refs, supplemental_confidence = self._supplemental_fields(
                        source_pages,
                        dataset_name=dataset_name,
                    )
                    combined = {**supplemental, **combined}
                if required <= set(combined):
                    record_issue(
                        self.context,
                        make_issue(
                            category="ocr_structure_correction",
                            issue_code="full_page_ocr_recovered_native_fields",
                            message=(
                                "Missing labelled values were recovered from a complete logical-page OCR pass; "
                                "a footer-confirmed supplemental spread half was used when necessary."
                            ),
                            severity="info",
                            status="resolved",
                            parser_stage="native_tolerant_parser",
                            target_dataset=dataset_name if dataset_name != "report_header" else "personal_report_metadata",
                            confidence=supplemental_confidence or recovered_confidence or confidence or None,
                            source_refs=(*refs, *recovered_refs, *supplemental_refs),
                            reason_codes=(
                                "logical_page_rerendered",
                                "unique_label_value",
                                "cell_crop_not_required",
                                *("footer_confirmed_supplemental_spread" for _ in [0] if supplemental_refs),
                            ),
                        ),
                    )
                    result.append(
                        NativeLabeledRecord(
                            dataset_name=dataset_name,
                            fields=combined,
                            source_refs=(*refs, *recovered_refs, *supplemental_refs),
                            confidence=min(
                                value for value in (confidence, recovered_confidence, supplemental_confidence) if value
                            ) if (confidence or recovered_confidence or supplemental_confidence) else 0.0,
                        )
                    )
                    continue
                record_issue(
                    self.context,
                    make_issue(
                        category="ocr_structure_correction",
                        issue_code="recognized_native_section_missing_required_value",
                        message="A PBOC native table section was recognized but a required labelled value was not recoverable.",
                        parser_stage="native_tolerant_parser",
                        target_dataset=dataset_name if dataset_name != "report_header" else "personal_report_metadata",
                        observed_value={"observed_labels": sorted(observed), "missing_labels": sorted(missing)},
                        confidence=confidence or None,
                        source_refs=refs,
                        reason_codes=("section_anchor_recognized", "required_value_missing", "no_guess_applied"),
                    ),
                )
                continue
            result.append(
                NativeLabeledRecord(
                    dataset_name=dataset_name,
                    fields=fields,
                    source_refs=refs,
                    confidence=confidence,
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
            recovered, recovered_refs, recovered_confidence = self._full_page_fields(
                candidate_pages,
                dataset_name=dataset_name,
            )
            if required <= set(recovered):
                record_issue(
                    self.context,
                    make_issue(
                        category="ocr_structure_correction",
                        issue_code="full_page_ocr_recovered_unparsed_native_section",
                        message=(
                            "The source section was visible in page evidence but absent from native tables; "
                            "complete-page OCR recovered a unique labelled record."
                        ),
                        severity="info",
                        status="resolved",
                        parser_stage="native_tolerant_parser",
                        target_dataset=dataset_name if dataset_name != "report_header" else "personal_report_metadata",
                        confidence=recovered_confidence or None,
                        source_refs=recovered_refs,
                        reason_codes=("native_table_missing", "page_anchor_observed", "full_page_ocr_unique_values"),
                    ),
                )
                result.append(
                    NativeLabeledRecord(
                        dataset_name=dataset_name,
                        fields=recovered,
                        source_refs=recovered_refs,
                        confidence=recovered_confidence,
                    )
                )
        return result


__all__ = ["NativeLabeledRecord", "PBOCPersonalDetailNativeParser"]
