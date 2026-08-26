"""Source-conserving helpers for financial Community projectors.

The helpers in this module only read sealed ``TableBlock`` facts.  They do not
repair OCR text, infer document types, or mutate ``ParseResult``.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Callable


@dataclass(frozen=True)
class ColumnSpec:
    """One projected business column tied to a physical source column."""

    source_index: int
    key: str
    label: str
    value_type: str = "string"


@dataclass
class ProjectedSegment:
    """One source table segment ready for a domain projector."""

    dataset_id: str
    columns: list[ColumnSpec]
    records: list[dict[str, Any]]
    sections: list[dict[str, Any]]
    source_page: int
    table_id: str
    kind: str
    title: str = ""
    row_groups: list[dict[str, Any]] = field(default_factory=list)
    column_header_bands: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    source_row_refs: list[dict[str, Any]] = field(default_factory=list)


_AMOUNT_RE = re.compile(r"^[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?$")
_PAREN_AMOUNT_RE = re.compile(r"^[（(](\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?[）)]$")
_DECIMAL_PLACEHOLDER_RE = re.compile(r"^[-‐‑‒–—―−－]{1,3}$")
_EMPTY_QUOTE_RE = re.compile(r"“”|‘’|\"\"|''")
_DATE_TOKEN = r"(?P<year>20\d{2})[年./-](?P<month>\d{1,2})[月./-](?P<day>\d{1,2})日?"
_DATE_RE = re.compile(_DATE_TOKEN)
_SUBJECT_ID_RE = re.compile(r"(?<![0-9A-Z])(?:[0-9A-Z]{15}|[0-9A-Z]{18})(?![0-9A-Z])")


def clean_label(value: Any) -> str:
    """Normalize layout whitespace without changing source cell values."""

    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    return re.sub(r"\s+", "", text)


def row_cells_by_column(row: Any, width: int) -> list[Any | None]:
    """Return physical cells indexed by their declared source column."""

    positioned: list[Any | None] = [None] * max(0, width)
    for fallback, cell in enumerate(getattr(row, "cells", None) or []):
        declared = getattr(cell, "col_index", None)
        column = fallback if declared is None else int(declared)
        if 0 <= column < len(positioned):
            positioned[column] = cell
    return positioned


def row_texts(row: Any, width: int) -> list[str]:
    """Return raw source text in physical column order."""

    return [
        str(getattr(cell, "text", "") or "") if cell is not None else "" for cell in row_cells_by_column(row, width)
    ]


def table_width(table: Any) -> int:
    """Return the widest physical row/header width."""

    return max(
        len(getattr(table, "headers", None) or []),
        max((len(getattr(row, "cells", None) or []) for row in getattr(table, "rows", None) or []), default=0),
    )


def flatten_header_rows(rows: list[Any], row_indexes: list[int], width: int) -> list[str]:
    """Flatten evidence-backed header bands into one label per physical column."""

    bands: list[list[str]] = []
    for row_index in row_indexes:
        if not 0 <= row_index < len(rows):
            continue
        expanded = [""] * width
        cells = row_cells_by_column(rows[row_index], width)
        for fallback, cell in enumerate(cells):
            if cell is None:
                continue
            label = clean_label(getattr(cell, "text", ""))
            if not label:
                continue
            start = getattr(cell, "col_index", None)
            start = fallback if start is None else int(start)
            span = max(1, int(getattr(cell, "col_span", 1) or 1))
            for column in range(max(0, start), min(width, start + span)):
                expanded[column] = label
        _fill_unspanned_parent_cells(expanded)
        bands.append(expanded)

    labels: list[str] = []
    for column in range(width):
        path: list[str] = []
        for band in bands:
            label = band[column]
            if label and label not in path:
                path.append(label)
        labels.append("/".join(path))
    return labels


def active_columns(labels: list[str], entries: list[tuple[int, Any]]) -> list[int]:
    """Drop only geometry-only empty columns, preserving explicit source columns."""

    width = len(labels)
    nonempty_by_column = [False] * width
    for _row_index, row in entries:
        for column, value in enumerate(row_texts(row, width)):
            if clean_label(value):
                nonempty_by_column[column] = True

    active: list[int] = []
    for column, label in enumerate(labels):
        if clean_label(label) or nonempty_by_column[column]:
            active.append(column)

    # A merged parent header can be expanded over a geometry column that never
    # owns data.  Remove only that duplicate; unique explicit empty columns stay.
    for column in list(active):
        label = clean_label(labels[column])
        if not label or nonempty_by_column[column]:
            continue
        peers = [
            peer
            for peer in active
            if peer != column and clean_label(labels[peer]) == label and nonempty_by_column[peer]
        ]
        if peers:
            active.remove(column)
    return active


def build_column_specs(
    labels: list[str],
    source_columns: list[int],
    key_builder: Callable[[str, int, int], str],
) -> list[ColumnSpec]:
    """Build unique stable keys while retaining source labels and positions."""

    occurrences: dict[str, int] = {}
    used: set[str] = set()
    specs: list[ColumnSpec] = []
    for source_index in source_columns:
        label = labels[source_index] if source_index < len(labels) else ""
        normalized_label = clean_label(label)
        occurrence = occurrences.get(normalized_label, 0)
        occurrences[normalized_label] = occurrence + 1
        base = key_builder(normalized_label, source_index, occurrence) or f"column_{source_index + 1:02d}"
        key = base
        suffix = 2
        while key in used:
            key = f"{base}_{suffix}"
            suffix += 1
        used.add(key)
        specs.append(
            ColumnSpec(
                source_index=source_index,
                key=key,
                label=label or f"第{source_index + 1}列",
                value_type="decimal" if _amount_column(label) else "string",
            )
        )
    return specs


def build_records(
    table: Any,
    entries: list[tuple[int, Any]],
    columns: list[ColumnSpec],
    *,
    dataset_id: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Project business rows with exact cell-level source references."""

    records: list[dict[str, Any]] = []
    warnings: list[str] = []
    width = table_width(table)
    for ordinal, (table_row_index, row) in enumerate(entries, start=1):
        cells = row_cells_by_column(row, width)
        raw: dict[str, str] = {}
        normalized: dict[str, Any] = {}
        refs: list[dict[str, Any]] = []
        evidence_ids: list[str] = []
        missing_evidence: list[str] = []
        review_reasons: list[str] = []
        page, table_id, source_row = _source_row_coordinates(table, table_row_index, row)

        for spec in columns:
            cell = cells[spec.source_index] if spec.source_index < len(cells) else None
            value = str(getattr(cell, "text", "") or "") if cell is not None else ""
            raw[spec.key] = value
            normalized[spec.key] = normalize_scalar(value, value_type=spec.value_type)
            format_issue = _decimal_format_issue(value, value_type=spec.value_type)
            if format_issue:
                review_reasons.append(format_issue)
                warnings.append(
                    f"precision:financial_{format_issue}:table={table_id}:row={source_row}:field={spec.key}"
                )
            if spec.value_type == "string" and _EMPTY_QUOTE_RE.search(value):
                review_reasons.append("ambiguous_empty_quoted_text")
                warnings.append(
                    f"precision:financial_ambiguous_empty_quoted_text:table={table_id}:"
                    f"row={source_row}:field={spec.key}"
                )
            cell_refs = [dict(ref) for ref in (getattr(cell, "source_cell_refs", None) or []) if isinstance(ref, dict)]
            for ref in cell_refs:
                ref.setdefault("page", page)
                ref.setdefault("table_id", table_id)
                ref.setdefault("row", source_row)
                ref.setdefault("col", spec.source_index)
                ref.setdefault("field_name", spec.key)
                if ref not in refs:
                    refs.append(ref)

            cell_evidence = [str(value) for value in (getattr(cell, "evidence_ids", None) or []) if value]
            evidence_ids.extend(cell_evidence)
            has_geometry = isinstance(getattr(cell, "bbox", None), (list, tuple))
            if not cell_refs and (cell_evidence or has_geometry):
                refs.append(
                    {
                        "page": page,
                        "table_id": table_id,
                        "row": source_row,
                        "col": spec.source_index,
                        "field_name": spec.key,
                    }
                )
            elif value.strip() and not cell_refs and not cell_evidence and not has_geometry:
                missing_evidence.append(spec.key)

        source: dict[str, Any] = {
            "page": page,
            "page_range": [page],
            "table_id": table_id,
            "physical_table_id": table_id,
            "table_row_index": table_row_index,
            "source_row_index": source_row,
        }
        if refs:
            source["source_cell_refs"] = refs
        if evidence_ids:
            source["evidence_ids"] = list(dict.fromkeys(evidence_ids))
        if missing_evidence:
            warnings.append(
                f"precision:financial_cell_evidence_missing:table={table_id}:row={source_row}:"
                f"fields={','.join(missing_evidence)}"
            )

        record: dict[str, Any] = {
            "record_id": f"{dataset_id}:r{ordinal:06d}",
            "raw": raw,
            "canonical_raw": dict(raw),
            "normalized": normalized,
            "source": source,
            "confidence": min(
                [
                    _table_source_confidence(table),
                    *[_cell_source_confidence(cell) for cell in cells if cell is not None],
                ]
            ),
        }
        if refs:
            record["source_cell_refs"] = refs
        if evidence_ids:
            record["evidence_ids"] = list(dict.fromkeys(evidence_ids))
        if missing_evidence:
            record["review"] = {"required": True, "reasons": ["missing_cell_evidence"]}
        if review_reasons:
            record["review"] = {
                "required": True,
                "reasons": list(
                    dict.fromkeys(
                        [
                            *(record.get("review", {}).get("reasons", [])),
                            *review_reasons,
                        ]
                    )
                ),
            }
        records.append(record)
    return records, warnings


def source_row_refs(table: Any, entries: list[tuple[int, Any]]) -> list[dict[str, Any]]:
    """Return ordered physical row identities independently of projected records."""

    refs: list[dict[str, Any]] = []
    for table_row_index, row in entries:
        page, table_id, source_row = _source_row_coordinates(table, table_row_index, row)
        refs.append({"page": page, "table_id": table_id, "source_row_index": source_row})
    return refs


def _source_row_coordinates(table: Any, table_row_index: int, row: Any) -> tuple[int, str, int]:
    page = int(getattr(row, "source_page", 0) or getattr(table, "page", 0) or 1)
    table_id = str(getattr(table, "table_id", "") or "table")
    declared_source_row = getattr(row, "source_row_index", None)
    try:
        source_row = int(declared_source_row) if declared_source_row is not None else table_row_index
    except (TypeError, ValueError):
        source_row = table_row_index
    if source_row < 0:
        source_row = table_row_index
    return page, table_id, source_row


def _table_source_confidence(table: Any) -> float:
    values = [
        value
        for value in (
            getattr(table, "confidence", None),
            getattr(table, "extraction_confidence", None),
        )
        if value is not None
    ]
    confidences: list[float] = []
    for value in values:
        try:
            confidences.append(max(0.0, min(1.0, float(value))))
        except (TypeError, ValueError):
            continue
    return min(confidences or [0.0])


def _cell_source_confidence(cell: Any) -> float:
    """Return the most specific available extraction confidence for a cell."""

    geometry_confidence = getattr(cell, "geometry_confidence", None)
    if geometry_confidence is not None:
        try:
            return max(0.0, min(1.0, float(geometry_confidence)))
        except (TypeError, ValueError):
            return 0.0
    try:
        return max(0.0, min(1.0, float(getattr(cell, "confidence", 0.0) or 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _bounded_bbox(value: Any) -> list[float] | None:
    bbox = getattr(value, "bbox", None)
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        return [float(coordinate) for coordinate in bbox]
    except (TypeError, ValueError):
        return None


def _bbox_union(values: list[Any]) -> list[float] | None:
    boxes = [bbox for value in values if (bbox := _bounded_bbox(value)) is not None]
    if not boxes:
        return None
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


def _bounded_confidence(value: Any, *, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(getattr(value, "confidence", default) or default)))
    except (TypeError, ValueError):
        return max(0.0, min(1.0, default))


def extract_labeled_header_fields(parse_result: Any) -> tuple[dict[str, str], dict[str, Any]]:
    """Extract explicit statement header fields without inferring absent values."""

    fields: dict[str, str] = {}
    details: dict[str, Any] = {}
    for page in getattr(parse_result, "pages", None) or []:
        page_number = int(getattr(page, "page_number", 0) or 1)
        for block in getattr(page, "texts", None) or []:
            text = unicodedata.normalize("NFKC", str(getattr(block, "content", "") or ""))
            evidence = [str(item) for item in (getattr(block, "evidence_ids", None) or []) if item]
            _collect_labeled_header_text(
                text,
                page_number,
                evidence,
                fields,
                details,
                confidence=_bounded_confidence(block),
                bbox=_bounded_bbox(block),
                source_kind="text",
            )
        for key_value in getattr(page, "key_values", None) or []:
            key = str(getattr(key_value, "key", "") or "")
            value = str(getattr(key_value, "value", "") or "")
            evidence = [str(item) for item in (getattr(key_value, "evidence_ids", None) or []) if item]
            _collect_labeled_header_values(
                [key, value],
                page_number,
                evidence,
                fields,
                details,
                confidence=_bounded_confidence(key_value),
                bbox=_bounded_bbox(key_value),
                source_kind="key_value",
            )
        for table in getattr(page, "tables", None) or []:
            header_values = [str(value or "") for value in (getattr(table, "headers", None) or [])]
            if header_values:
                _collect_labeled_header_values(
                    header_values,
                    page_number,
                    [str(item) for item in (getattr(table, "evidence_ids", None) or []) if item],
                    fields,
                    details,
                    confidence=_table_source_confidence(table),
                    source_kind="table_header",
                )
            for row in getattr(table, "rows", None) or []:
                cells = list(getattr(row, "cells", None) or [])
                values = [str(getattr(cell, "text", "") or "") for cell in cells]
                evidence = [
                    str(evidence_id)
                    for cell in cells
                    for evidence_id in (getattr(cell, "evidence_ids", None) or [])
                    if evidence_id
                ]
                confidences = [_cell_source_confidence(cell) for cell in cells]
                _collect_labeled_header_values(
                    values,
                    page_number,
                    evidence,
                    fields,
                    details,
                    confidence=min(confidences or [_table_source_confidence(table)]),
                    bbox=_bbox_union(cells),
                    source_kind="table_row",
                )
    full_text = str(getattr(parse_result, "full_text", "") or getattr(parse_result, "raw_text", "") or "")
    if full_text:
        _collect_labeled_header_text(
            unicodedata.normalize("NFKC", full_text),
            0,
            [],
            fields,
            details,
            confidence=0.6,
            source_kind="full_text",
        )
    return fields, details


def _collect_labeled_header_values(
    values: list[str],
    page: int,
    evidence: list[str],
    fields: dict[str, str],
    details: dict[str, Any],
    *,
    confidence: float = 1.0,
    bbox: list[float] | None = None,
    source_kind: str = "",
) -> None:
    """Extract explicitly paired labels and values from physical table cells."""

    before = set(fields)
    normalized = [unicodedata.normalize("NFKC", str(value or "")).strip() for value in values]
    _collect_labeled_header_text(
        " | ".join(normalized),
        page,
        evidence,
        fields,
        details,
        confidence=confidence,
        bbox=bbox,
        source_kind=source_kind,
    )

    def value_after(index: int) -> str:
        return next((value for value in normalized[index + 1 :] if value), "")

    for index, text in enumerate(normalized):
        compact = clean_label(text)
        if not compact:
            continue
        value = value_after(index)
        if any(label in compact for label in ("纳税人名称", "企业名称", "公司名称")):
            raw_name = _inline_or_adjacent_value(text, value)
            name = normalize_subject_name(raw_name)
            if name and re.search(r"公司|企业|集团|事务所|合作社|学校|医院|中心|研究院", name):
                _set_header_field(
                    fields,
                    details,
                    "subject_name",
                    name,
                    bounded_subject_name_raw(raw_name),
                    page,
                    evidence,
                )
        if any(label in compact for label in ("纳税人识别号", "统一社会信用代码", "社会信用代码")):
            identifier = _inline_or_adjacent_value(text, value).upper()
            match = _SUBJECT_ID_RE.search(identifier)
            if match:
                _set_header_field(fields, details, "subject_id", match.group(0), match.group(0), page, evidence)
        if any(label in compact for label in ("税款所属期起止", "税款所属时间", "税款所属期", "所属期")):
            period = f"{text} {_inline_or_adjacent_value(text, value)}"
            matches = list(_DATE_RE.finditer(period))
            if matches:
                _set_header_field(
                    fields,
                    details,
                    "period_start",
                    _date_value(matches[0]),
                    matches[0].group(0),
                    page,
                    evidence,
                )
                end = matches[1] if len(matches) > 1 else matches[0]
                _set_header_field(
                    fields,
                    details,
                    "period_end",
                    _date_value(end),
                    end.group(0),
                    page,
                    evidence,
                )
        if any(label in compact for label in ("报送日期", "填报日期", "填表日期", "申报日期", "编制日期")):
            date_text = f"{text} {_inline_or_adjacent_value(text, value)}"
            match = _DATE_RE.search(date_text)
            if match:
                _set_header_field(
                    fields,
                    details,
                    "document_date",
                    _date_value(match),
                    match.group(0),
                    page,
                    evidence,
                )
        if any(label in compact for label in ("金额单位", "货币单位", "单位")):
            unit_text = _inline_or_adjacent_value(text, value)
            match = re.search(r"人民币|CNY|元", unit_text, re.IGNORECASE)
            if match:
                _set_header_field(fields, details, "currency_unit", "CNY", match.group(0), page, evidence)
    _apply_header_source_metadata(
        fields,
        details,
        before=before,
        page=page,
        evidence=evidence,
        confidence=confidence,
        bbox=bbox,
        source_kind=source_kind,
    )


def _inline_or_adjacent_value(text: str, adjacent: str) -> str:
    match = re.search(r"[：:]\s*(.+)$", text)
    return match.group(1).strip() if match else adjacent.strip()


def _collect_labeled_header_text(
    text: str,
    page: int,
    evidence: list[str],
    fields: dict[str, str],
    details: dict[str, Any],
    *,
    confidence: float = 1.0,
    bbox: list[float] | None = None,
    source_kind: str = "",
) -> None:
    before = set(fields)
    name_match = re.search(r"(?:纳税人名称|企业名称|公司名称)\s*[：:]\s*([^\r\n]+)", text)
    if name_match:
        raw_name = name_match.group(1).strip()
        name = normalize_subject_name(raw_name)
        if name and re.search(r"公司|企业|集团|事务所|合作社|学校|医院|中心|研究院", name):
            bounded_raw = bounded_subject_name_raw(raw_name)
            _set_header_field(fields, details, "subject_name", name, bounded_raw, page, evidence)
            _set_header_field(fields, details, "organization", name, bounded_raw, page, evidence)

    for label in ("纳税人识别号", "统一社会信用代码", "社会信用代码"):
        label_start = text.find(label)
        if label_start < 0:
            continue
        match = _SUBJECT_ID_RE.search(text[label_start : label_start + 100].upper())
        if match:
            _set_header_field(fields, details, "subject_id", match.group(0), match.group(0), page, evidence)
            break

    for label in ("税款所属期起止", "税款所属时间", "税款所属期", "所属期"):
        label_start = text.find(label)
        if label_start < 0:
            continue
        matches = list(_DATE_RE.finditer(text[label_start : label_start + 100]))
        if matches:
            start = _date_value(matches[0])
            end = _date_value(matches[1]) if len(matches) > 1 else start
            end_match = matches[1] if len(matches) > 1 else matches[0]
            _set_header_field(fields, details, "period_start", start, matches[0].group(0), page, evidence)
            _set_header_field(fields, details, "period_end", end, end_match.group(0), page, evidence)
            break

    date_match = re.search(
        r"(?:报送日期|填报日期|填表日期|申报日期|编制日期)\s*[：:]\s*(" + _DATE_TOKEN + r")",
        text,
    )
    if date_match:
        _set_header_field(
            fields,
            details,
            "document_date",
            _date_value(date_match),
            date_match.group(1),
            page,
            evidence,
        )

    unit_match = re.search(r"(?:金额单位|货币单位|单位)\s*[：:]\s*(人民币|CNY|元)", text, re.IGNORECASE)
    if unit_match:
        _set_header_field(fields, details, "currency_unit", "CNY", unit_match.group(1), page, evidence)
    _apply_header_source_metadata(
        fields,
        details,
        before=before,
        page=page,
        evidence=evidence,
        confidence=confidence,
        bbox=bbox,
        source_kind=source_kind,
    )


def _apply_header_source_metadata(
    fields: dict[str, str],
    details: dict[str, Any],
    *,
    before: set[str],
    page: int,
    evidence: list[str],
    confidence: float,
    bbox: list[float] | None,
    source_kind: str,
) -> None:
    bounded_confidence = max(0.0, min(1.0, float(confidence)))
    source_ref: dict[str, Any] = {}
    if page > 0:
        source_ref["page"] = page
    if bbox:
        source_ref["bbox"] = list(bbox)
    if evidence:
        source_ref["evidence_ids"] = list(dict.fromkeys(evidence))
    has_source = bool(source_ref)
    if source_kind and has_source:
        source_ref["source"] = source_kind
    review = "needs_evidence"
    if has_source and bounded_confidence >= 0.85:
        review = "auto_accepted"
    elif has_source and bounded_confidence >= 0.6:
        review = "manual_optional"
    elif has_source:
        review = "needs_review"
    for key in set(fields) - before:
        detail = details.get(key)
        if not isinstance(detail, dict):
            continue
        if page > 0:
            detail["source_page"] = page
        else:
            detail.pop("source_page", None)
        detail["confidence"] = bounded_confidence
        detail["evidence_ids"] = list(dict.fromkeys(evidence))
        detail["source_refs"] = [source_ref] if source_ref else []
        detail["review"] = review
        if bbox:
            detail["bbox"] = list(bbox)
        if source_kind:
            detail["source_kind"] = source_kind


def _set_header_field(
    fields: dict[str, str],
    details: dict[str, Any],
    key: str,
    value: str,
    raw: str,
    page: int,
    evidence: list[str],
) -> None:
    if key in fields or not value:
        return
    fields[key] = value
    details[key] = {
        "value": value,
        "raw": raw,
        "source_page": page,
        "confidence": 1.0,
        "evidence_ids": evidence,
    }


def _date_value(match: re.Match[str]) -> str:
    return f"{int(match.group('year')):04d}-{int(match.group('month')):02d}-{int(match.group('day')):02d}"


def normalize_subject_name(value: str) -> str:
    """Remove explicit signature artifacts from a displayed organization name."""

    text = bounded_subject_name_raw(value)
    return re.sub(r"\s+", "", text).strip()


def bounded_subject_name_raw(value: str) -> str:
    """Return the exact organization-name value span without label annotations or form controls."""

    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    text = re.sub(r"^[（(]?公章[）)]?\s*", "", text)
    text = re.sub(r"(?:\s+NsrSignkey\b|\bNsrSignkey\b).*$", "", text, flags=re.IGNORECASE | re.DOTALL)
    return text.strip()


def data_dictionary(segments: list[ProjectedSegment]) -> dict[str, Any]:
    """Build dynamic dataset column metadata without changing Bundle 3.0."""

    def column_descriptor(segment: ProjectedSegment, spec: ColumnSpec) -> dict[str, Any]:
        descriptor: dict[str, Any] = {"label": spec.label, "type": spec.value_type}
        source_header_bands = segment.column_header_bands.get(spec.key)
        if source_header_bands:
            descriptor["source_header_bands"] = source_header_bands
        return descriptor

    return {
        "fields": {
            "subject_name": {"label": "纳税人名称", "type": "string"},
            "subject_id": {"label": "纳税人识别号", "type": "string"},
            "period_start": {"label": "所属期开始", "type": "date"},
            "period_end": {"label": "所属期结束", "type": "date"},
            "document_date": {"label": "填报日期", "type": "date"},
            "currency_unit": {"label": "金额单位", "type": "string"},
        },
        "datasets": {
            segment.dataset_id: {"columns": {spec.key: column_descriptor(segment, spec) for spec in segment.columns}}
            for segment in segments
        },
    }


def normalize_scalar(value: str, *, value_type: str) -> str | None:
    """Normalize exact numeric text with Decimal semantics; never repair glyphs."""

    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    if value_type != "decimal":
        return text
    if not text or decimal_placeholder(text):
        return None
    compact = re.sub(r"\s+", "", text).replace("，", ",")
    negative = False
    parenthesized = _PAREN_AMOUNT_RE.fullmatch(compact)
    if parenthesized:
        compact = parenthesized.group(1)
        negative = True
    if not _AMOUNT_RE.fullmatch(compact):
        return None
    try:
        number = Decimal(compact.replace(",", ""))
    except InvalidOperation:
        return None
    if negative:
        number = -abs(number)
    decimals = len(compact.rsplit(".", 1)[1]) if "." in compact else 0
    return format(number, f".{decimals}f") if decimals else format(number, "f")


def row_type(row: Any) -> str:
    """Return the lowercase semantic row type."""

    value = getattr(row, "row_type", "")
    return str(getattr(value, "value", value) or "").lower()


def amount_like(value: Any) -> bool:
    """Return whether a source cell is an exact amount/ordinal token."""

    text = re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(value or ""))).replace("，", ",")
    return bool(_AMOUNT_RE.fullmatch(text) or _PAREN_AMOUNT_RE.fullmatch(text))


def decimal_placeholder(value: Any) -> bool:
    """Return whether a decimal source token explicitly denotes no numeric value."""

    text = re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(value or "")))
    return bool(text and _DECIMAL_PLACEHOLDER_RE.fullmatch(text))


def add_review_reason(record: dict[str, Any], reason: str) -> None:
    """Mark one projected record for review without duplicating reasons."""

    review = record.setdefault("review", {"required": True, "reasons": []})
    review["required"] = True
    reasons = review.setdefault("reasons", [])
    if reason not in reasons:
        reasons.append(reason)


def _decimal_format_issue(value: str, *, value_type: str) -> str:
    """Return a review reason for visibly incomplete monetary OCR text."""

    if value_type != "decimal":
        return ""
    text = re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(value or ""))).replace("，", ",")
    if not text or decimal_placeholder(text):
        return ""
    parenthesized = _PAREN_AMOUNT_RE.fullmatch(text)
    numeric = parenthesized.group(1) if parenthesized else text
    if not _AMOUNT_RE.fullmatch(numeric):
        return "amount_format_invalid"
    if "." in numeric and len(numeric.rsplit(".", 1)[1]) != 2:
        return "decimal_scale_unexpected"
    return ""


def _fill_unspanned_parent_cells(row: list[str]) -> None:
    anchors = [(index, label) for index, label in enumerate(row) if label]
    for anchor_index, (start, label) in enumerate(anchors):
        end = anchors[anchor_index + 1][0] if anchor_index + 1 < len(anchors) else len(row)
        for column in range(start, end):
            if not row[column]:
                row[column] = label


def _amount_column(label: str) -> bool:
    compact = clean_label(label)
    return bool(re.search(r"金额|余额|销售额|税额|价税|本月数|本年累计|本期数|上期数|期初|期末|年初|年末", compact))


__all__ = [
    "ColumnSpec",
    "ProjectedSegment",
    "active_columns",
    "add_review_reason",
    "amount_like",
    "build_column_specs",
    "build_records",
    "clean_label",
    "data_dictionary",
    "decimal_placeholder",
    "extract_labeled_header_fields",
    "flatten_header_rows",
    "normalize_subject_name",
    "normalize_scalar",
    "row_texts",
    "row_type",
    "source_row_refs",
    "table_width",
]
