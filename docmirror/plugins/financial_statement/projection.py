"""Source-conserving projection for balance, income, and cash-flow statements."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any

from docmirror.plugins._base.financial_source_projection import (
    ProjectedSegment,
    active_columns,
    add_review_reason,
    amount_like,
    build_column_specs,
    build_records,
    clean_label,
    data_dictionary,
    extract_labeled_header_fields,
    flatten_header_rows,
    normalize_scalar,
    row_texts,
    row_type,
    source_row_refs,
    table_width,
)
from docmirror.plugins._base.projector import ProjectionData

_HEADER_LABEL_RE = re.compile(
    r"项目|资产|负债|所有者权益|股东权益|行次|栏次|期末余额|年初余额|期初余额|年末余额|"
    r"本年累计|本月金额|本期金额|上期金额|本年发生额|上年发生额"
)
_FOOTER_RE = re.compile(r"负责人|会计机构负责人|制表人|签章|签名|单位负责人")
_STATEMENT_TITLES = {
    "balance_sheet": "资产负债表",
    "income_statement": "利润表",
    "cash_flow_statement": "现金流量表",
    "owners_equity_changes": "所有者权益变动表",
}


def derive_financial_statement_projection(parse_result: Any, *, full_text: str = "") -> ProjectionData:
    """Derive financial datasets without invoking the Generic projector."""

    detected_type = str(getattr(getattr(parse_result, "entities", None), "document_type", "") or "financial_statement")
    segments: list[ProjectedSegment] = []
    warnings: list[str] = []
    type_counts: dict[str, int] = {}
    for page in getattr(parse_result, "pages", None) or []:
        page_number = int(getattr(page, "page_number", 0) or 1)
        for table in getattr(page, "tables", None) or []:
            kind = detect_statement_kind(table, fallback=detected_type, page_text=_page_text(page))
            if kind is None:
                continue
            occurrence = type_counts.get(kind, 0) + 1
            type_counts[kind] = occurrence
            dataset_id = kind if occurrence == 1 else f"{kind}_{occurrence:02d}"
            segment, segment_warnings = project_statement_table(
                table,
                page_number=page_number,
                dataset_id=dataset_id,
                kind=kind,
            )
            warnings.extend(segment_warnings)
            if segment is not None:
                segments.append(segment)

    if not segments:
        warnings.append("precision:financial_statement_table_unresolved")
    entity_fields, field_details = extract_labeled_header_fields(parse_result)
    entity_fields.pop("organization", None)
    field_details.pop("organization", None)
    datasets = {segment.dataset_id: segment.records for segment in segments if segment.records}
    verification_blockers = [
        *[str(item) for item in (getattr(getattr(parse_result, "parser_info", None), "warnings", None) or [])],
        *warnings,
    ]
    column_orders = {
        segment.dataset_id: [column.key for column in segment.columns] for segment in segments if segment.records
    }
    dataset_labels = {
        segment.dataset_id: segment.title or _STATEMENT_TITLES.get(segment.kind, segment.dataset_id)
        for segment in segments
        if segment.records
    }
    dataset_section_ids = {
        segment.dataset_id: f"section_{segment.dataset_id}" for segment in segments if segment.records
    }
    row_groups = {
        segment.dataset_id: list(segment.row_groups) for segment in segments if segment.records and segment.row_groups
    }
    return ProjectionData(
        projector_id="financial_statement",
        document_type=detected_type,
        entity_fields=entity_fields,
        domain_facts={
            "field_details": field_details,
            "data_dictionary": data_dictionary(segments),
            "financial_projection": {
                "strict_source_evidence": True,
                "dataset_count": len(datasets),
                "emitted_rows": sum(len(rows) for rows in datasets.values()),
            },
            "dataset_source_row_refs": {
                segment.dataset_id: list(segment.source_row_refs) for segment in segments if segment.records
            },
            "dataset_verification_blockers": {
                segment.dataset_id: list(dict.fromkeys(verification_blockers))
                for segment in segments
                if segment.records and verification_blockers
            },
        },
        semantic={
            "dataset_column_order": column_orders,
            "dataset_reading_columns": column_orders,
            "dataset_document_order": list(datasets),
            "dataset_labels": dataset_labels,
            "dataset_section_ids": dataset_section_ids,
            "dataset_row_groups": row_groups,
            "enhanced_markdown": {
                "suppress_empty_sections": True,
                "show_top_document_metadata": False,
                "section_layouts": {
                    "financial_statement": {
                        "omit_unlisted": True,
                        "groups": [
                            {
                                "hide_title": True,
                                "fields": [
                                    "period_start",
                                    "period_end",
                                    "document_date",
                                    "currency_unit",
                                    "subject_id",
                                    "subject_name",
                                ],
                            }
                        ],
                    }
                },
                "dataset_layouts": {
                    dataset_id: {
                        "hide_title": True,
                        "prefer_canonical_raw": True,
                        "show_raw_when_normalized_null": True,
                    }
                    for dataset_id in datasets
                },
            },
        },
        datasets=datasets,
        sections=tuple(_dataset_section(segment) for segment in segments if segment.records),
        warnings=tuple(dict.fromkeys(warnings)),
        confidence=1.0 if datasets and not warnings else 0.65,
        reason="post-seal financial-statement projection from sealed table facts",
    )


def detect_statement_kind(table: Any, *, fallback: str = "", page_text: str = "") -> str | None:
    """Recognize one statement table from source headers and nearby page text."""

    values = [*(getattr(table, "headers", None) or [])]
    for row in list(getattr(table, "rows", None) or [])[:8]:
        values.extend(str(getattr(cell, "text", "") or "") for cell in getattr(row, "cells", None) or [])
    haystack = clean_label(" ".join([page_text, *values]))
    if "所有者权益变动表" in haystack or fallback == "owners_equity_changes":
        return "owners_equity_changes"
    if ("资产" in haystack and "负债" in haystack) or fallback == "balance_sheet":
        return "balance_sheet"
    if "现金流量" in haystack or "经营活动产生的现金流量" in haystack or fallback == "cash_flow_statement":
        return "cash_flow_statement"
    if any(marker in haystack for marker in ("营业收入", "利润总额", "净利润")) or fallback == "income_statement":
        return "income_statement"
    return None


def project_statement_table(
    table: Any,
    *,
    page_number: int,
    dataset_id: str,
    kind: str,
) -> tuple[ProjectedSegment | None, list[str]]:
    """Project one physical statement table with exact row/cell provenance."""

    rows = list(getattr(table, "rows", None) or [])
    width = table_width(table)
    if not rows or width < 2:
        return None, [f"precision:financial_table_empty:page={page_number}:table={getattr(table, 'table_id', '')}"]

    source_headers = [clean_label(value) for value in (getattr(table, "headers", None) or [])]
    source_score = _header_score(source_headers)
    embedded = [
        (index, _header_score(row_texts(row, width)))
        for index, row in enumerate(rows[:8])
        if _header_score(row_texts(row, width)) >= 2
    ]
    best_embedded = max(embedded, key=lambda item: item[1], default=None)
    header_indexes: list[int] = []
    data_start = 0
    if best_embedded is not None and (source_score < 2 or best_embedded[1] > source_score):
        start = best_embedded[0]
        header_indexes = [start]
        cursor = start + 1
        while cursor < min(len(rows), start + 3) and _leaf_header_score(row_texts(rows[cursor], width)) >= 2:
            header_indexes.append(cursor)
            cursor += 1
        labels = flatten_header_rows(rows, header_indexes, width)
        data_start = cursor
    elif source_score >= 2:
        labels = [clean_label(value) for value in getattr(table, "headers", None) or []]
        labels.extend([""] * max(0, width - len(labels)))
    else:
        return None, [
            f"precision:financial_header_unresolved:page={page_number}:table={getattr(table, 'table_id', '')}"
        ]

    entries: list[tuple[int, Any]] = []
    for row_index, row in enumerate(rows[data_start:], start=data_start):
        values = row_texts(row, width)
        nonempty = [clean_label(value) for value in values if clean_label(value)]
        if not nonempty or row_type(row) == "separator":
            continue
        if _same_header(values, labels):
            continue
        joined = "".join(nonempty)
        if _FOOTER_RE.search(joined) and not any(amount_like(value) for value in values[1:]):
            break
        entries.append((row_index, row))

    active = active_columns(labels, entries)
    columns = build_column_specs(labels, active, lambda label, index, occurrence: _statement_key(kind, label, index))
    records, warnings = build_records(table, entries, columns, dataset_id=dataset_id)
    if not records:
        warnings.append(
            f"precision:financial_business_rows_missing:page={page_number}:table={getattr(table, 'table_id', '')}"
        )
        return None, warnings
    if kind == "cash_flow_statement":
        _validate_cash_flow_totals(records, table_id=str(getattr(table, "table_id", "") or ""), warnings=warnings)
    _validate_statement_line_sequences(
        records,
        columns=columns,
        dataset_id=dataset_id,
        kind=kind,
        warnings=warnings,
    )
    return (
        ProjectedSegment(
            dataset_id=dataset_id,
            columns=columns,
            records=records,
            sections=[],
            source_page=page_number,
            table_id=str(getattr(table, "table_id", "") or ""),
            kind=kind,
            title=_STATEMENT_TITLES.get(kind, dataset_id),
            source_row_refs=source_row_refs(table, entries),
        ),
        warnings,
    )


def _statement_key(kind: str, label: str, index: int) -> str:
    compact = clean_label(label)
    if kind == "balance_sheet":
        balance_keys = (
            "asset_item",
            "asset_line_no",
            "asset_ending_balance",
            "asset_opening_balance",
            "liability_and_equity_item",
            "liability_line_no",
            "liability_ending_balance",
            "liability_opening_balance",
        )
        return balance_keys[index] if index < len(balance_keys) else f"column_{index + 1:02d}"
    if index == 0:
        return "item"
    if "行次" in compact or "栏次" in compact:
        return "line_no"
    if "本年累计" in compact or "本年发生额" in compact:
        return "year_to_date_amount"
    if "本月" in compact:
        return "current_month_amount"
    if "本期" in compact:
        return "current_period_amount"
    if "上期" in compact or "上年" in compact:
        return "previous_period_amount"
    return f"column_{index + 1:02d}"


def _header_score(values: list[str]) -> int:
    return sum(bool(_HEADER_LABEL_RE.search(clean_label(value))) for value in values if clean_label(value))


def _leaf_header_score(values: list[str]) -> int:
    leaf_re = re.compile(r"行次|栏次|余额|金额|本年|本月|本期|上期|发生额|累计|数量|单价")
    return sum(bool(leaf_re.search(clean_label(value))) for value in values if clean_label(value))


def _same_header(values: list[str], labels: list[str]) -> bool:
    source = [clean_label(value) for value in values]
    expected = [clean_label(value) for value in labels]
    return bool(source and source == expected)


def _validate_statement_line_sequences(
    records: list[dict[str, Any]],
    *,
    columns: list[Any],
    dataset_id: str,
    kind: str,
    warnings: list[str],
) -> None:
    line_keys = [column.key for column in columns if column.key == "line_no" or column.key.endswith("_line_no")]
    for line_key in line_keys:
        line_records: dict[int, dict[str, Any]] = {}
        for record in records:
            value = clean_label((record.get("raw") or {}).get(line_key, ""))
            if not value:
                continue
            if not re.fullmatch(r"\d{1,3}", value):
                warnings.append(
                    "FINANCIAL_LINE_NO_INVALID:"
                    f"dataset={dataset_id}:record={record.get('record_id')}:field={line_key}:value={value}"
                )
                add_review_reason(record, "financial_line_no_invalid")
                continue
            line_records[int(value)] = record
        if not line_records:
            continue
        observed = sorted(line_records)
        missing = sorted(set(range(observed[0], observed[-1] + 1)).difference(observed))
        if kind != "balance_sheet" and observed[0] > 1:
            missing = [*range(1, observed[0]), *missing]
        if missing:
            warnings.append(
                "FINANCIAL_LINE_SEQUENCE_GAP:"
                f"dataset={dataset_id}:field={line_key}:missing={','.join(map(str, missing))}"
            )


def _validate_cash_flow_totals(records: list[dict[str, Any]], *, table_id: str, warnings: list[str]) -> None:
    markers = {
        "operating": "经营活动产生的现金流量净额",
        "investing": "投资活动产生的现金流量净额",
        "financing": "筹资活动产生的现金流量净额",
        "net": "现金净增加额",
    }
    matched: dict[str, dict[str, Any]] = {}
    for record in records:
        item = clean_label((record.get("raw") or {}).get("item", ""))
        for key, marker in markers.items():
            if marker in item:
                matched[key] = record
    if set(matched) != set(markers):
        return

    amount_fields = [
        key
        for key in (matched["net"].get("raw") or {})
        if key.endswith("_amount") or key in {"year_to_date_amount", "current_month_amount"}
    ]
    for field in amount_fields:
        values = {key: _record_decimal(record, field) for key, record in matched.items()}
        if any(value is None for value in values.values()):
            continue
        expected = values["operating"] + values["investing"] + values["financing"]
        actual = values["net"]
        if abs(expected - actual) <= Decimal("0.01"):
            continue
        warnings.append(
            f"precision:cash_flow_net_mismatch:table={table_id}:field={field}:expected={expected}:actual={actual}"
        )
        for record in matched.values():
            add_review_reason(record, "cash_flow_net_mismatch")


def _record_decimal(record: dict[str, Any], field: str) -> Decimal | None:
    value = str((record.get("raw") or {}).get(field, "") or "")
    normalized = normalize_scalar(value, value_type="decimal")
    try:
        return Decimal(normalized)
    except (InvalidOperation, TypeError):
        return None


def _dataset_section(segment: ProjectedSegment) -> dict[str, Any]:
    pages = [int((record.get("source") or {}).get("page") or segment.source_page) for record in segment.records]
    return {
        "id": f"section_{segment.dataset_id}",
        "title": segment.title or _STATEMENT_TITLES.get(segment.kind, segment.dataset_id),
        "type": "financial_statement",
        "page_start": min(pages, default=segment.source_page),
        "page_end": max(pages, default=segment.source_page),
    }


def _page_text(page: Any) -> str:
    return " ".join(str(getattr(block, "content", "") or "") for block in getattr(page, "texts", None) or [])


__all__ = ["derive_financial_statement_projection", "detect_statement_kind", "project_statement_table"]
