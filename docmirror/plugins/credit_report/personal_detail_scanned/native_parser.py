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
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
    make_issue,
    record_issue,
)
from docmirror.plugins.credit_report.personal_detail_scanned.liability_clusters import (
    decode_packed_liability_row,
    normalize_packed_liability_header,
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
_CREDIT_AGREEMENT_INLINE_LABELS = tuple(
    sorted(
        (
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
        ),
        key=len,
        reverse=True,
    )
)
_CREDIT_AGREEMENT_INLINE_LABEL_RE = re.compile(
    "|".join(re.escape(label) for label in _CREDIT_AGREEMENT_INLINE_LABELS)
)
_EVIDENCE_SECTION_END_MARKERS = (
    "非信贷交易信息",
    "公共信息",
    "查询记录",
    "本人声明",
    "异议标注",
    "报告说明",
)

_PACKED_LIABILITY_FIELD_LABELS = {
    "institution": "管理机构",
    "business_type": "业务种类",
    "open_date": "开立日期",
    "due_date": "到期日期",
    "responsibility_type": "责任人类型",
    "responsibility_amount": "还款责任金额",
    "currency": "币种",
    "contract_number": "保证合同编号",
}
_PACKED_LIABILITY_LABEL_FIELDS = {
    label: field_name for field_name, label in _PACKED_LIABILITY_FIELD_LABELS.items()
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
    alias = _LABEL_OCR_ALIASES.get(text)
    if alias in _LABELS:
        return alias, 0.96
    return None, 0.0


def _collapsed_credit_agreement_pairs(value: Any) -> list[tuple[str, str]]:
    """Decode exact labels and their suffixes from one collapsed card cell.

    Some OCR table graphs preserve a complete PBOC agreement row as one text
    cell.  This remains schema-bound evidence because every value is delimited
    by an exact printed label; unknown/fuzzy labels are never authorized.
    """

    text = str(value or "").strip()
    if not text:
        return []
    matches = list(_CREDIT_AGREEMENT_INLINE_LABEL_RE.finditer(text))
    if not matches or text[: matches[0].start()].strip(" \t:：,，;；|/"):
        return []
    pairs: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        candidate = text[match.end() : end].strip(" \t:：,，;；|/")
        if candidate:
            pairs.append((match.group(0), candidate))
    return pairs


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


@dataclass(frozen=True, slots=True)
class _PackedLiabilityRowObservation:
    """One adjacent canonical packed header/value witness."""

    header_row_index: int
    value_row_index: int | None
    normalized_labels: tuple[str, ...]
    header_values: tuple[str, ...]
    fields: dict[str, str]
    unresolved_reason: str | None

    @property
    def resolved(self) -> bool:
        return self.unresolved_reason is None


def _packed_liability_row_observations(
    rows: list[list[Any]],
) -> list[_PackedLiabilityRowObservation]:
    """Decode every adjacent complete liability header without fuzzy repair."""

    observations: list[_PackedLiabilityRowObservation] = []
    for header_row_index, header_row in enumerate(rows):
        header_values = [_cell_text(cell) for cell in header_row]
        normalized_labels = normalize_packed_liability_header(header_values)
        if normalized_labels is None:
            continue
        value_row_index = header_row_index + 1 if header_row_index + 1 < len(rows) else None
        value_values = (
            [_cell_text(cell) for cell in rows[value_row_index]]
            if value_row_index is not None
            else []
        )
        source_header_tokens = tuple(_compact(value) for value in header_values if _compact(value))
        source_value_tokens = tuple(_compact(value) for value in value_values if _compact(value))
        if (
            source_header_tokens == tuple(_compact(value) for value in normalized_labels)
            and len(source_value_tokens) == len(normalized_labels)
        ):
            # A clean one-cell-per-slot table already has a narrower exact
            # binding in the ordinary label-column decoder. The packed-row
            # decoder is reserved for packed rows and the helper's exact OCR
            # label aliases, avoiding broader provenance on clean tables.
            continue
        decoded = decode_packed_liability_row(header_values, value_values)
        fields = {
            _PACKED_LIABILITY_FIELD_LABELS[field_name]: str(value)
            for field_name, value in decoded.fields.items()
            if field_name in _PACKED_LIABILITY_FIELD_LABELS
        }
        observations.append(
            _PackedLiabilityRowObservation(
                header_row_index=header_row_index,
                value_row_index=value_row_index,
                normalized_labels=normalized_labels,
                header_values=tuple(header_values),
                fields=fields,
                unresolved_reason=decoded.unresolved_reason,
            )
        )
    return observations


def _packed_value_equivalent(label: str, left: Any, right: Any) -> bool:
    """Compare raw direct-cell and normalized packed-row representations."""

    if label in {"开立日期", "到期日期"}:
        return re.sub(r"\D", "", str(left or "")) == re.sub(r"\D", "", str(right or ""))
    if label == "还款责任金额":
        return re.sub(r"\D", "", str(left or "")) == re.sub(r"\D", "", str(right or ""))
    if label == "币种":
        currencies = {
            "人民币": "CNY",
            "人民币元": "CNY",
            "RMB": "CNY",
            "CNY": "CNY",
            "美元": "USD",
            "USD": "USD",
            "欧元": "EUR",
            "EUR": "EUR",
            "港元": "HKD",
            "HKD": "HKD",
            "日元": "JPY",
            "JPY": "JPY",
            "英镑": "GBP",
            "GBP": "GBP",
        }
        return currencies.get(_compact(left).upper()) == currencies.get(_compact(right).upper())
    return _compact(left).upper() == _compact(right).upper()


def _merge_packed_liability_fields(
    fields: dict[str, str],
    refs_by_field: dict[str, tuple[dict[str, Any], ...]],
    bindings_by_field: dict[str, str],
    observed_labels: frozenset[str],
    unresolved_labels: frozenset[str],
    observations: list[_PackedLiabilityRowObservation],
    packed_refs: dict[int, tuple[dict[str, Any], ...]],
) -> tuple[frozenset[str], frozenset[str]]:
    """Merge helper-authorized fields and retain exact independent bindings.

    An unresolved packed row may still expose uniquely typed fields. The merge
    retains those exact spans and keeps every unsupported slot unresolved.
    """

    observed = set(observed_labels)
    unresolved = set(unresolved_labels)
    for observation_index, observation in enumerate(observations):
        observed.update(observation.normalized_labels)
        row_refs = packed_refs.get(observation_index, ())
        header_compact = _compact("".join(observation.header_values))
        for label in observation.normalized_labels:
            existing = fields.get(label)
            if (
                existing not in (None, "")
                and _compact(existing)
                and _compact(existing) in header_compact
            ):
                # The generic collapsed-agreement reader can see agreement
                # labels inside a packed liability header.  Such header text
                # is not a business value and must not survive either a
                # resolved or unresolved liability-row interpretation.
                fields.pop(label, None)
                refs_by_field.pop(label, None)
                bindings_by_field.pop(label, None)
        for label, candidate in observation.fields.items():
            existing = fields.get(label)
            if existing not in (None, "") and not _packed_value_equivalent(label, existing, candidate):
                fields.pop(label, None)
                refs_by_field.pop(label, None)
                bindings_by_field.pop(label, None)
                unresolved.add(label)
                continue
            if existing in (None, ""):
                fields[label] = candidate
                if row_refs:
                    refs_by_field[label] = tuple({**ref, "field_name": label} for ref in row_refs)
                bindings_by_field[label] = "canonical_packed_liability_row"
            unresolved.discard(label)
        if not observation.resolved:
            # Direct same-column evidence remains valid even when residue in a
            # different slot prevents a complete packed-row interpretation.
            # Exact partial packed-row fields likewise survive, while every
            # absent slot remains explicitly unresolved.
            unresolved.update(label for label in observation.normalized_labels if label not in fields)
    unresolved.difference_update(fields)
    return frozenset(observed), frozenset(unresolved)


class PBOCPersonalDetailNativeParser:
    """Resolve labelled card observations into one canonical record stream."""

    def __init__(self, context: Any) -> None:
        self.context = context

    @staticmethod
    def _native_packed_row_refs(
        page: Any,
        table: Any,
        rows: list[list[Any]],
        *,
        row_offset: int,
        row_indices: tuple[int, ...],
        field_name: str,
        binding: str,
    ) -> tuple[dict[str, Any], ...]:
        refs: list[dict[str, Any]] = []
        for row_index in row_indices:
            if not 0 <= row_index < len(rows):
                continue
            for column_index, cell in enumerate(rows[row_index]):
                if not _cell_text(cell).strip():
                    continue
                refs.append(
                    {
                        **_field_source_ref(
                            page,
                            table,
                            row=row_offset + row_index,
                            column=column_index,
                            field_name=field_name,
                        ),
                        "binding": binding,
                    }
                )
        return tuple(refs)

    @staticmethod
    def _evidence_packed_row_refs(
        rows: list[list[_PositionedEvidenceCell]],
        *,
        row_indices: tuple[int, ...],
        field_name: str,
        binding: str,
    ) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                **cell.source_ref(field_name=field_name),
                "binding": binding,
            }
            for row_index in row_indices
            if 0 <= row_index < len(rows)
            for cell in rows[row_index]
            if cell.text.strip()
        )

    def _record_unresolved_packed_liability_rows(
        self,
        rows: list[list[Any]],
        observations: list[_PackedLiabilityRowObservation],
        witness_refs: dict[int, tuple[dict[str, Any], ...]],
        *,
        parser_stage: str,
    ) -> None:
        for observation_index, observation in enumerate(observations):
            if observation.resolved:
                continue
            header = [_cell_text(cell) for cell in rows[observation.header_row_index]]
            value = (
                [_cell_text(cell) for cell in rows[observation.value_row_index]]
                if observation.value_row_index is not None
                else []
            )
            affected_fields = [
                _PACKED_LIABILITY_LABEL_FIELDS[label]
                for label in observation.normalized_labels
                if label not in observation.fields
                and label in _PACKED_LIABILITY_LABEL_FIELDS
            ]
            printed_sequences = {
                match.group(1)
                for row in rows
                for cell in row
                if (match := re.search(r"账户\s*(\d{1,3})", _cell_text(cell)))
            }
            observed_value: dict[str, Any] = {
                "header": header,
                "value": value,
                "unresolved_reason": observation.unresolved_reason,
                "retained_typed_fields": dict(observation.fields),
                "affected_fields": affected_fields,
            }
            if len(printed_sequences) == 1:
                observed_value["printed_sequence"] = next(iter(printed_sequences))
            record_issue(
                self.context,
                make_issue(
                    category="ocr_cell_level_error",
                    issue_code="candidate_b_packed_liability_row_unresolved",
                    message=(
                        "A complete canonical repayment-responsibility header was observed, but its "
                        "adjacent packed value row did not have one complete typed interpretation. "
                        "Fields without unique typed support were withheld and the source witness was preserved."
                    ),
                    parser_stage=parser_stage,
                    target_dataset="repayment_liability_records",
                    observed_value=observed_value,
                    source_refs=witness_refs.get(observation_index, ()),
                    reason_codes=(
                        "canonical_packed_liability_header",
                        str(observation.unresolved_reason or "packed_row_unresolved"),
                        "unique_typed_fields_retained",
                        "residual_or_ambiguous_fields_reported",
                        "raw_witness_preserved",
                        "unsupported_fields_withheld",
                    ),
                ),
            )

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

    @staticmethod
    def _record_group_top(table: Any, row_offset: int, row_count: int) -> float | None:
        metadata = getattr(table, "metadata", None) or {}
        cell_boxes = (
            metadata.get("source_cell_bboxes") or metadata.get("cell_bboxes")
            if isinstance(metadata, dict)
            else None
        )
        if not isinstance(cell_boxes, list):
            return None
        tops: list[float] = []
        for row in cell_boxes[row_offset : row_offset + row_count]:
            if not isinstance(row, list):
                continue
            for box in row:
                if isinstance(box, (list, tuple)) and len(box) == 4:
                    tops.append(float(box[1]))
        return min(tops) if tops else None

    @classmethod
    def _printed_sequence_anchors_for_groups(
        cls,
        page: Any,
        table: Any,
        dataset_name: str,
        groups: list[tuple[int, list[list[str]]]],
    ) -> dict[int, tuple[str, dict[str, Any]]]:
        """Bind each printed card heading to at most one repeated table card."""

        if dataset_name != "credit_lines" or not groups:
            return {}
        pattern = re.compile(r"授信协议\s*(\d{1,3})")
        anchors: list[tuple[float, str, dict[str, Any]]] = []
        for text_item in getattr(page, "texts", None) or ():
            match = pattern.search(str(getattr(text_item, "content", "") or ""))
            box = getattr(text_item, "bbox", None)
            if match is None or not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            anchors.append(
                (
                    float(box[3]),
                    match.group(1),
                    {
                        "source": "native_detail_canonical_anchor_text",
                        "logical_page": int(getattr(page, "page_number", 0) or 0),
                        "source_page": int(
                            getattr(page, "source_page_number", 0)
                            or getattr(page, "page_number", 0)
                            or 0
                        ),
                        "bbox": list(box),
                        "geometry_scope": "text",
                        "field_name": "sequence",
                        "binding": "canonical_card_anchor",
                    },
                )
            )
        if not anchors:
            return {}
        anchors.sort(key=lambda item: item[0])
        group_tops = [
            cls._record_group_top(table, row_offset, len(record_rows))
            for row_offset, record_rows in groups
        ]
        assigned: dict[int, tuple[str, dict[str, Any]]] = {}
        used: set[int] = set()
        if all(top is not None for top in group_tops):
            for (row_offset, _record_rows), group_top in zip(groups, group_tops, strict=True):
                eligible = [
                    (index, anchor)
                    for index, anchor in enumerate(anchors)
                    if index not in used and anchor[0] <= float(group_top) + 8.0
                ]
                if not eligible:
                    continue
                index, (_bottom, sequence, ref) = max(eligible, key=lambda item: item[1][0])
                used.add(index)
                assigned[row_offset] = (sequence, ref)
            return assigned

        # Geometry-free fallback is authorized only by a complete one-to-one
        # set of explicit headings.  Never reuse one heading for several cards.
        table_box = getattr(table, "bbox", None)
        if isinstance(table_box, (list, tuple)) and len(table_box) == 4:
            nearby = [
                anchor
                for anchor in anchors
                if float(table_box[1]) - 120.0 <= anchor[0] <= float(table_box[3]) + 8.0
            ]
        else:
            nearby = anchors
        if len(nearby) == len(groups) and len({sequence for _bottom, sequence, _ref in nearby}) == len(groups):
            for (row_offset, _record_rows), (_bottom, sequence, ref) in zip(
                groups, nearby, strict=True
            ):
                assigned[row_offset] = (sequence, ref)
        return assigned

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
                collapsed_pairs = _collapsed_credit_agreement_pairs(cell_text)
                if collapsed_pairs:
                    for label, candidate in collapsed_pairs:
                        observed.add(label)
                        if label in fields and _compact(fields[label]) != _compact(candidate):
                            fields.pop(label, None)
                            positions.pop(label, None)
                            unresolved.add(label)
                            continue
                        fields.setdefault(label, candidate)
                        positions.setdefault(label, (row_index, column))
                        scores.append(1.0)
                    continue
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

    def _validated_evidence_page_order(
        self,
        pages: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[int, int] | None]:
        """Return corrected pages in one complete authoritative document order."""

        materialized = list(pages)
        logical_pages: list[int] = []
        try:
            for page in materialized:
                logical_page = int(page.get("page") or 0)
                if logical_page <= 0:
                    return materialized, None
                logical_pages.append(logical_page)
        except (AttributeError, TypeError, ValueError):
            return materialized, None
        if len(set(logical_pages)) <= 1:
            return materialized, {
                logical_page: 1 for logical_page in set(logical_pages)
            }

        resolution = getattr(self.context, "reading_order_resolution", None)
        raw_order = getattr(self.context, "reading_order_by_logical", None)
        if not (
            isinstance(resolution, Mapping)
            and resolution.get("resolved") is True
            and resolution.get("authoritative") is True
            and isinstance(raw_order, Mapping)
            and raw_order
        ):
            return materialized, None

        order: dict[int, int] = {}
        try:
            for raw_logical, raw_position in raw_order.items():
                if isinstance(raw_logical, bool) or isinstance(raw_position, bool):
                    return materialized, None
                logical_page = int(raw_logical)
                position = int(raw_position)
                if logical_page <= 0 or position <= 0 or logical_page in order:
                    return materialized, None
                order[logical_page] = position
        except (TypeError, ValueError):
            return materialized, None
        positions = list(order.values())
        if len(positions) != len(set(positions)) or sorted(positions) != list(
            range(1, len(positions) + 1)
        ):
            return materialized, None

        required_pages = set(logical_pages)
        for page in getattr(self.context, "pages", None) or []:
            try:
                logical_page = int(getattr(page, "page_number", 0) or 0)
            except (TypeError, ValueError):
                return materialized, None
            if logical_page <= 0:
                return materialized, None
            required_pages.add(logical_page)
        if not required_pages.issubset(order):
            return materialized, None

        ordered = [
            page
            for _index, page in sorted(
                enumerate(materialized),
                key=lambda item: (order[int(item[1].get("page") or 0)], item[0]),
            )
        ]
        return ordered, order

    def _record_evidence_page_order_unresolved(
        self,
        dataset_name: str,
        pages: list[dict[str, Any]],
    ) -> None:
        """Localize corrected-page evidence withheld from an unproven order."""

        target_dataset = (
            dataset_name
            if dataset_name != "report_header"
            else "personal_report_metadata"
        )
        refs = tuple(
            {
                "source": "personal_detail_corrected_page_rows",
                "logical_page": int(page.get("page") or 0),
                "source_page": int(page.get("source_page") or page.get("page") or 0),
                "geometry_scope": "logical_page",
            }
            for page in pages
            if isinstance(page, Mapping) and int(page.get("page") or 0) > 0
        )
        resolution = getattr(self.context, "reading_order_resolution", None)
        raw_order = getattr(self.context, "reading_order_by_logical", None)
        record_issue(
            self.context,
            make_issue(
                category="page_continuation",
                issue_code="candidate_b_native_evidence_page_order_unresolved",
                message=(
                    "Corrected-page native evidence was not joined or promoted because "
                    "its document reading order was incomplete or non-authoritative."
                ),
                parser_stage="candidate_b_native_evidence_page_order",
                target_dataset=target_dataset,
                observed_value={
                    "logical_pages": [int(page.get("page") or 0) for page in pages],
                    "reading_order_resolution": (
                        dict(resolution) if isinstance(resolution, Mapping) else None
                    ),
                    "registered_reading_order": (
                        dict(raw_order) if isinstance(raw_order, Mapping) else None
                    ),
                },
                source_refs=refs,
                reason_codes=(
                    "corrected_page_population",
                    "authoritative_complete_order_required",
                    "cross_page_state_not_carried",
                    "record_not_invented",
                ),
            ),
        )

    def _evidence_record_groups(
        self,
        dataset_name: str,
    ) -> list[tuple[list[list[_PositionedEvidenceCell]], tuple[dict[str, Any], ...]]]:
        """Segment repeated PBOC cards from corrected logical-page rows."""
        loader = getattr(self.context, "corrected_evidence_pages", None)
        if not callable(loader):
            return []
        raw_pages = loader() or []
        pages, reading_order = self._validated_evidence_page_order(raw_pages)
        if dataset_name == "report_header":
            if not pages:
                return []
            if len({int(page.get("page") or 0) for page in pages}) > 1 and reading_order is None:
                self._record_evidence_page_order_unresolved(dataset_name, pages)
                return []
            page = pages[0]
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
        previous_logical_page: int | None = None

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
            page_rows = self._ocr_positioned_rows(page)
            page_compact_rows = [
                _compact("".join(_cell_text(cell) for cell in row))
                for row in page_rows
            ]
            same_page = bool(
                logical_page > 0 and previous_logical_page == logical_page
            )
            adjacent_page = bool(
                reading_order is not None
                and previous_logical_page in reading_order
                and logical_page in reading_order
                and reading_order[logical_page]
                == reading_order[previous_logical_page] + 1
            )
            if (
                active
                and previous_logical_page is not None
                and not same_page
                and not adjacent_page
            ):
                flush()
                active = False
                liability_party_category = ""
                starts_new_section = any(heading in text for text in page_compact_rows)
                fragment_labels = set(_SECTION_MARKERS.get(dataset_name, ()))
                if dataset_name == "credit_lines":
                    fragment_labels.update(_CREDIT_AGREEMENT_INLINE_LABELS)
                elif dataset_name == "repayment_liability_records":
                    fragment_labels.update(_PACKED_LIABILITY_LABEL_FIELDS)
                if not starts_new_section and any(
                    label in text
                    for text in page_compact_rows
                    for label in fragment_labels
                ):
                    self._record_evidence_page_order_unresolved(dataset_name, [page])
            page_key = (logical_page, source_page)
            page_ref = {
                "source": "personal_detail_corrected_page_rows",
                "logical_page": logical_page,
                "source_page": source_page,
                "geometry_scope": "logical_page",
            }
            for row in page_rows:
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
                complete_liability_header = bool(
                    dataset_name == "repayment_liability_records"
                    and normalize_packed_liability_header([cell.text for cell in row]) is not None
                )
                if (
                    anchor.search(compact)
                    or (has_primary and current_has_primary)
                    or (complete_liability_header and not current_rows)
                ):
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
            previous_logical_page = logical_page
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
                collapsed_pairs = _collapsed_credit_agreement_pairs(cell.text)
                if collapsed_pairs:
                    for label, candidate in collapsed_pairs:
                        observed.add(label)
                        values_by_label.setdefault(label, []).append(
                            (candidate, (cell,), 1.0)
                        )
                    continue
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
            record_groups = self._split_repeated_cards_with_offsets(dataset_name, rows)
            assigned_sequence_anchors = self._printed_sequence_anchors_for_groups(
                _page,
                _table,
                dataset_name,
                record_groups,
            )
            for row_offset, record_rows in record_groups:
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
                if dataset_name == "repayment_liability_records":
                    packed_observations = _packed_liability_row_observations(record_rows)
                    packed_refs: dict[int, tuple[dict[str, Any], ...]] = {}
                    witness_refs: dict[int, tuple[dict[str, Any], ...]] = {}
                    for observation_index, observation in enumerate(packed_observations):
                        value_indices = (
                            (observation.value_row_index,)
                            if observation.value_row_index is not None
                            else ()
                        )
                        packed_refs[observation_index] = self._native_packed_row_refs(
                            _page,
                            _table,
                            record_rows,
                            row_offset=row_offset,
                            row_indices=value_indices,
                            field_name="packed_liability_value_row",
                            binding="canonical_packed_liability_row",
                        )
                        witness_refs[observation_index] = self._native_packed_row_refs(
                            _page,
                            _table,
                            record_rows,
                            row_offset=row_offset,
                            row_indices=tuple(
                                value
                                for value in (
                                    observation.header_row_index,
                                    observation.value_row_index,
                                )
                                if value is not None
                            ),
                            field_name="packed_liability_witness",
                            binding="canonical_packed_liability_witness",
                        )
                    self._record_unresolved_packed_liability_rows(
                        record_rows,
                        packed_observations,
                        witness_refs,
                        parser_stage="candidate_b_native_packed_liability_decoder",
                    )
                    observed_labels, unresolved = _merge_packed_liability_fields(
                        fields,
                        refs_by_field,
                        bindings_by_field,
                        observed_labels,
                        unresolved,
                        packed_observations,
                        packed_refs,
                    )
                    if any(observation.resolved for observation in packed_observations):
                        confidence = confidence or 1.0
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
                        assigned_anchor = assigned_sequence_anchors.get(row_offset)
                        if assigned_anchor is not None:
                            printed_sequence, anchor_ref = assigned_anchor
                        elif len(record_groups) == 1:
                            printed_sequence, anchor_ref = self._printed_sequence_anchor_above_table(
                                _page,
                                _table,
                                dataset_name,
                            )
                        else:
                            printed_sequence, anchor_ref = "", None
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
                marker_hits = {
                    marker
                    for marker in required
                    if marker in observed_fields or marker in observed_labels or marker in label_text
                }
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
                packed_observations = _packed_liability_row_observations(rows)
                packed_refs: dict[int, tuple[dict[str, Any], ...]] = {}
                witness_refs: dict[int, tuple[dict[str, Any], ...]] = {}
                packed_confidences: list[float] = []
                for observation_index, observation in enumerate(packed_observations):
                    value_indices = (
                        (observation.value_row_index,)
                        if observation.value_row_index is not None
                        else ()
                    )
                    packed_refs[observation_index] = self._evidence_packed_row_refs(
                        rows,
                        row_indices=value_indices,
                        field_name="packed_liability_value_row",
                        binding="canonical_packed_liability_row",
                    )
                    witness_refs[observation_index] = self._evidence_packed_row_refs(
                        rows,
                        row_indices=tuple(
                            value
                            for value in (
                                observation.header_row_index,
                                observation.value_row_index,
                            )
                            if value is not None
                        ),
                        field_name="packed_liability_witness",
                        binding="canonical_packed_liability_witness",
                    )
                    if observation.resolved and observation.value_row_index is not None:
                        packed_confidences.extend(
                            cell.confidence
                            for cell in rows[observation.value_row_index]
                            if cell.text.strip() and cell.confidence > 0.0
                        )
                self._record_unresolved_packed_liability_rows(
                    rows,
                    packed_observations,
                    witness_refs,
                    parser_stage="candidate_b_corrected_page_packed_liability_decoder",
                )
                observed, unresolved = _merge_packed_liability_fields(
                    fields,
                    refs_by_field,
                    bindings_by_field,
                    observed,
                    unresolved,
                    packed_observations,
                    packed_refs,
                )
                if packed_confidences:
                    packed_confidence = min(packed_confidences)
                    confidence = min(confidence, packed_confidence) if confidence > 0.0 else packed_confidence
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
