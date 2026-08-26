"""Source-conserving tax-return projection from sealed table facts."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from docmirror.plugins._base.financial_source_projection import (
    ColumnSpec,
    ProjectedSegment,
    active_columns,
    add_review_reason,
    amount_like,
    bounded_subject_name_raw,
    build_column_specs,
    build_records,
    clean_label,
    data_dictionary,
    decimal_placeholder,
    flatten_header_rows,
    normalize_subject_name,
    row_cells_by_column,
    row_texts,
    row_type,
    source_row_refs,
    table_width,
)
from docmirror.plugins._base.projector import ProjectionData, load_projection_policy
from docmirror.plugins.financial_statement.projection import detect_statement_kind, project_statement_table

_TAX_HEADER_RE = re.compile(
    r"项目|项目及栏次|栏次|序号|销售额|税额|金额|余额|本月数|本年累计|本期发生额|"
    r"期初余额|期末余额|发票|扣除|抵减|减免性质|免税性质|征收率|税率|合计"
)
_FORM_FOOTER_RE = re.compile(r"声明.*真实性|纳税人.*签章|经办人名称|受理税务机关|代理机构签章|填表日期")
_ORDINAL_RE = re.compile(r"^\d{1,3}[A-Za-z]?(?:[=＝+\-−≤≥<>].+)?$")
_DATE_RE = re.compile(r"(?P<year>20\d{2})[年./-](?P<month>\d{1,2})[月./-](?P<day>\d{1,2})日?")
_SUBJECT_ID_RE = re.compile(r"(?<![0-9A-Z])(?:[0-9A-Z]{15}|[0-9A-Z]{18})(?![0-9A-Z])")
_SUBJECT_LABELS = ("纳税人名称", "企业名称", "公司名称")
_SUBJECT_ID_LABELS = ("纳税人识别号", "统一社会信用代码", "社会信用代码")
_PERIOD_LABELS = ("税款所属时间", "税款所属期", "所属期")
_DOCUMENT_DATE_LABELS = ("报送日期", "填报日期", "填表日期", "申报日期", "编制日期")
_CURRENCY_LABELS = ("金额单位", "货币单位", "单位")
_STATUTORY_ADDITIONAL_TAX_LABELS = [
    "税(费)种",
    "增值税税额",
    "增值税免抵税额",
    "留抵退税本期扣除额",
    "税(费)率",
    "本期应纳税(费)额",
    "减免性质代码",
    "减免税(费)额",
    "减征比例(%)",
    "减征额",
    "减免性质代码",
    "本期抵免金额",
    "本期已缴税(费)额",
    "本期应补(退)税(费)额",
]
_TAX_TEXT_COLUMN_KEYS = {
    "additional_deduction_item",
    "applicability_status",
    "deduction_item",
    "invoice_code",
    "invoice_count",
    "invoice_date",
    "invoice_number",
    "invoice_type",
    "item_and_line_no",
    "item_category",
    "item_name",
    "item_subcategory",
    "line_no",
    "relief_code",
    "relief_code_2",
    "relief_code_and_name",
    "section_name",
    "tax_exemption_code_and_name",
    "tax_or_fee_type",
    "tax_reduction_code_and_name",
}
_TAX_AMOUNT_LABEL_RE = re.compile(
    r"金额|余额|销售额|税额|价税|本月数|本年累计|本期数|上期数|发生额|扣除额|抵减额|"
    r"调减额|减征额|减免税\(费\)额|抵免金额|已缴税|应纳税|应补|免税额|比例|税率|征收率|"
    r"税\(费\)率"
)
_TABLE_CHOICE_MARK_RE = re.compile(r"[□■☐☑☒√✓✔○●]")
_TAX_FORM_PROFILES = tuple(load_projection_policy(__package__).get("form_profiles") or ())


@dataclass(frozen=True)
class _IdentityCandidate:
    """One explicit tax identity value with its source authority."""

    raw: str
    page: int
    score: int
    value: str = ""
    evidence_ids: tuple[str, ...] = ()
    bbox: tuple[float, float, float, float] | None = None
    source_kind: str = ""


def derive_tax_return_projection(parse_result: Any, *, full_text: str = "") -> ProjectionData:
    """Derive tax-return datasets without invoking Generic projection logic."""

    segments: list[ProjectedSegment] = []
    warnings: list[str] = []
    last_tax_segment: ProjectedSegment | None = None
    tax_segment_count = 0
    statement_counts: dict[str, int] = {}

    for page in getattr(parse_result, "pages", None) or []:
        page_number = int(getattr(page, "page_number", 0) or 1)
        page_text = _page_text(page)
        for table in getattr(page, "tables", None) or []:
            statement_kind = detect_statement_kind(table, fallback="", page_text=page_text)
            if statement_kind is not None:
                occurrence = statement_counts.get(statement_kind, 0) + 1
                statement_counts[statement_kind] = occurrence
                dataset_id = statement_kind if occurrence == 1 else f"{statement_kind}_{occurrence:02d}"
                statement, statement_warnings = project_statement_table(
                    table,
                    page_number=page_number,
                    dataset_id=dataset_id,
                    kind=statement_kind,
                )
                warnings.extend(statement_warnings)
                if statement is not None:
                    segments.append(statement)
                last_tax_segment = None
                continue

            projected, table_warnings = _project_tax_table(
                table,
                page_number=page_number,
                start_index=tax_segment_count,
                page_text=page_text,
            )
            warnings.extend(table_warnings)
            if projected:
                for segment in projected:
                    tax_segment_count += 1
                    segment.dataset_id = _tax_dataset_id(tax_segment_count)
                    _renumber_records(segment)
                    segments.append(segment)
                    last_tax_segment = segment
                continue

            continuation_entries = _continuation_entries(table)
            if last_tax_segment is not None and continuation_entries and _continuation_matches(table, last_tax_segment):
                records, continuation_warnings = build_records(
                    table,
                    continuation_entries,
                    last_tax_segment.columns,
                    dataset_id=last_tax_segment.dataset_id,
                )
                last_tax_segment.records.extend(records)
                last_tax_segment.source_row_refs.extend(source_row_refs(table, continuation_entries))
                _renumber_records(last_tax_segment)
                warnings.extend(continuation_warnings)
            elif getattr(table, "rows", None):
                warnings.append(
                    f"precision:tax_table_header_unresolved:page={page_number}:table={getattr(table, 'table_id', '')}"
                )

    for segment in segments:
        if segment.kind != "tax_return":
            continue
        _renumber_records(segment)
        _remove_repeated_code_gridline_artifacts(segment)
        _validate_tax_segment(segment, warnings)

    entity_fields, field_details, identity_warnings = _extract_tax_identity(parse_result)
    entity_fields.pop("organization", None)
    field_details.pop("organization", None)
    warnings.extend(identity_warnings)
    datasets = {segment.dataset_id: segment.records for segment in segments if segment.records}
    column_orders = {
        segment.dataset_id: [column.key for column in segment.columns] for segment in segments if segment.records
    }
    if not datasets:
        warnings.append("precision:tax_return_business_rows_missing")
    emitted_segments = [segment for segment in segments if segment.records]
    verification_blockers = [
        *[str(item) for item in (getattr(getattr(parse_result, "parser_info", None), "warnings", None) or [])],
        *warnings,
    ]
    dataset_labels = {segment.dataset_id: _tax_dataset_title(segment) for segment in emitted_segments}
    dataset_section_ids = {segment.dataset_id: _tax_section_id(segment) for segment in emitted_segments}
    row_groups = {
        segment.dataset_id: list(segment.row_groups) for segment in segments if segment.records and segment.row_groups
    }
    return ProjectionData(
        projector_id="tax_return",
        document_type="tax_return",
        entity_fields=entity_fields,
        domain_facts={
            "field_details": field_details,
            "data_dictionary": data_dictionary(segments),
            "tax_return_projection": {
                "strict_source_evidence": True,
                "dataset_count": len(datasets),
                "emitted_rows": sum(len(rows) for rows in datasets.values()),
            },
            "dataset_source_row_refs": {
                segment.dataset_id: list(segment.source_row_refs) for segment in segments if segment.records
            },
            "dataset_verification_blockers": {
                segment.dataset_id: list(dict.fromkeys(verification_blockers))
                for segment in emitted_segments
                if verification_blockers
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
                    "tax_form": {
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
                    segment.dataset_id: {
                        "hide_title": _tax_dataset_title(segment) == _tax_form_title(segment),
                        "prefer_canonical_raw": True,
                        "prefer_normalized_fields": ["line_no"],
                        "prefer_normalized_source_headers": True,
                        "render_source_header_rows": True,
                        "show_source_header_bands": True,
                        "show_raw_when_normalized_null": True,
                    }
                    for segment in emitted_segments
                },
            },
        },
        datasets=datasets,
        sections=tuple(_tax_form_sections(emitted_segments)),
        warnings=tuple(dict.fromkeys(warnings)),
        confidence=1.0 if datasets and not warnings else 0.65,
        reason="post-seal tax-return projection from sealed table facts",
    )


def _project_tax_table(
    table: Any,
    *,
    page_number: int,
    start_index: int,
    page_text: str,
) -> tuple[list[ProjectedSegment], list[str]]:
    rows = list(getattr(table, "rows", None) or [])
    width = table_width(table)
    if not rows or width < 2:
        return [], []
    detected_bands = _tax_header_bands(rows, width)
    if not detected_bands:
        return [], []
    source_bands = _tax_expand_header_bands(rows, detected_bands, width)

    segments: list[ProjectedSegment] = []
    warnings: list[str] = []
    for band_offset, detected_header_indexes in enumerate(detected_bands):
        header_indexes = source_bands[band_offset]
        start = min(header_indexes)
        data_start = max(detected_header_indexes) + 1
        end = min(source_bands[band_offset + 1]) if band_offset + 1 < len(source_bands) else len(rows)
        ordinal_header_indexes = _tax_ordinal_header_indexes(rows, start=data_start, end=end, width=width)
        labels = _normalize_tax_span_labels(
            [_correct_tax_header_label(label) for label in flatten_header_rows(rows, detected_header_indexes, width)]
        )
        labels = _normalize_tax_main_amount_labels(labels)
        labels = _normalize_statutory_additional_tax_labels(labels)
        entries: list[tuple[int, Any]] = []
        preceding_start = max(source_bands[band_offset - 1]) + 1 if band_offset else 0
        row_groups = _tax_pre_header_fact_groups(
            table,
            rows,
            start=preceding_start,
            end=start,
            width=width,
            page_number=page_number,
        )
        for row_index in range(data_start, end):
            row = rows[row_index]
            values = row_texts(row, width)
            nonempty = [clean_label(value) for value in values if clean_label(value)]
            if not nonempty or row_type(row) in {"header", "separator"} or _is_ordinal_row(values):
                continue
            joined = "".join(nonempty)
            if _FORM_FOOTER_RE.search(joined):
                break
            if _tax_header_score(values) >= 2 and not any(amount_like(value) for value in values):
                continue
            if _is_tax_data_row(values):
                entries.append((row_index, row))
                continue
            if _is_orphan_ordinal_row(values):
                # A statutory tax form can reserve a numbered physical row whose
                # label and values are intentionally blank (for example line 34
                # in the input-tax attachment).  The line number is still a
                # source fact and must remain in JSON/CSV/Markdown.
                entries.append((row_index, row))
                continue
            if len(nonempty) <= 2:
                row_groups.append(
                    {
                        "title": next(value for value in values if clean_label(value)).strip(),
                        "start_ordinal": len(entries) + 1,
                        "source_page": page_number,
                    }
                )

        labels = _normalize_tax_main_amount_labels(labels, entries)
        labels = [_normalize_tax_header_label(label) for label in labels]
        source_title_hint = _tax_header_title_hint(rows, header_indexes, width)
        entry_parts = _split_tax_entries_on_schema_drift(labels, entries)
        for part_labels, part_entries, entry_offset in entry_parts:
            source_columns = active_columns(part_labels, part_entries)
            columns = build_column_specs(part_labels, source_columns, _tax_column_key)
            columns = _normalize_tax_column_specs(columns)
            provisional_index = start_index + len(segments) + 1
            provisional_id = _tax_dataset_id(provisional_index)
            records, record_warnings = build_records(table, part_entries, columns, dataset_id=provisional_id)
            _normalize_tax_text_records(records, columns)
            warnings.extend(record_warnings)
            if not records:
                warnings.append(
                    f"precision:tax_business_rows_missing:page={page_number}:table={getattr(table, 'table_id', '')}:"
                    f"header_row={start}"
                )
                continue
            local_groups = _tax_groups_for_entry_slice(
                row_groups,
                entry_offset=entry_offset,
                entry_count=len(part_entries),
            )
            segment_title = _tax_segment_title(
                rows,
                header_start=start if entry_offset == 0 else part_entries[0][0],
                page_text=page_text,
                fallback_index=provisional_index,
                local_hint=_tax_part_title_hint(part_entries) if entry_offset else source_title_hint,
            )
            warnings.extend(
                _normalize_profiled_line_formulas(
                    records,
                    columns=columns,
                    form_title=segment_title,
                )
            )
            segments.append(
                ProjectedSegment(
                    dataset_id=provisional_id,
                    columns=columns,
                    records=records,
                    sections=[],
                    source_page=page_number,
                    table_id=str(getattr(table, "table_id", "") or ""),
                    kind="tax_return",
                    title=segment_title,
                    row_groups=_resolve_tax_row_groups(local_groups, records),
                    source_row_refs=source_row_refs(table, part_entries),
                    column_header_bands=(
                        _tax_column_header_bands(
                            table,
                            rows,
                            header_indexes=[*header_indexes, *ordinal_header_indexes],
                            ordinal_header_indexes=set(ordinal_header_indexes),
                            columns=columns,
                            page_number=page_number,
                            records=records,
                            form_title=segment_title,
                        )
                        if entry_offset == 0
                        else {}
                    ),
                )
            )
    return segments, warnings


def _tax_header_bands(rows: list[Any], width: int) -> list[list[int]]:
    candidates = [
        index
        for index, row in enumerate(rows)
        if _tax_header_score(row_texts(row, width)) >= 2
        and not any(amount_like(value) for value in row_texts(row, width))
    ]
    groups: list[list[int]] = []
    for index in candidates:
        if groups and index <= groups[-1][-1] + 1:
            groups[-1].append(index)
        else:
            groups.append([index])
    return groups


def _tax_expand_header_bands(rows: list[Any], bands: list[list[int]], width: int) -> list[list[int]]:
    """Include adjacent source parent headers without changing semantic column labels."""

    expanded: list[list[int]] = []
    for band in bands:
        lower_bound = max(expanded[-1]) + 1 if expanded else 0
        indexes = list(band)
        cursor = min(indexes) - 1
        while cursor >= lower_bound and _is_tax_parent_header_row(rows[cursor], width):
            indexes.insert(0, cursor)
            cursor -= 1
        expanded.append(indexes)
    return expanded


def _is_tax_parent_header_row(row: Any, width: int) -> bool:
    values = row_texts(row, width)
    nonempty = [clean_label(value) for value in values if clean_label(value)]
    if not nonempty or any(amount_like(value) for value in values):
        return False
    joined = "".join(nonempty)
    if "是否" in joined or "：" in joined or ":" in joined:
        return False
    amount_header_score = sum(bool(_TAX_AMOUNT_LABEL_RE.search(value)) for value in nonempty)
    header_score = _tax_header_score(values)
    spanning = any(
        max(1, int(getattr(cell, "col_span", 1) or 1)) > 1
        for cell in (getattr(row, "cells", None) or [])
        if clean_label(getattr(cell, "text", ""))
    )
    return (header_score >= 1 or amount_header_score >= 2) and (len(nonempty) >= 2 or spanning)


def _tax_pre_header_fact_groups(
    table: Any,
    rows: list[Any],
    *,
    start: int,
    end: int,
    width: int,
    page_number: int,
) -> list[dict[str, Any]]:
    """Keep evidence-backed table context immediately before a tax data header."""

    seeds: list[int] = []
    for row_index in range(end - 1, start - 1, -1):
        values = row_texts(rows[row_index], width)
        joined = "".join(clean_label(value) for value in values)
        if _is_tax_data_row(values):
            continue
        choice_end = min(end, row_index + 3)
        has_choice = "是否" in joined and any(
            _TABLE_CHOICE_MARK_RE.search("".join(row_texts(rows[candidate], width)))
            for candidate in range(row_index, choice_end)
        )
        has_labeled_value = "：" in joined or ":" in joined
        if has_choice or has_labeled_value:
            seeds.append(row_index)
    if not seeds:
        return []

    anchor = max(seeds)
    context_start = anchor
    while context_start > start and _is_tax_context_continuation_row(rows[context_start - 1], width):
        context_start -= 1
    context_end = anchor + 1
    while context_end < end and _is_tax_context_continuation_row(rows[context_end], width):
        context_end += 1

    facts: list[dict[str, Any]] = []
    source_rows: list[int] = []
    for row_index in range(context_start, context_end):
        row = rows[row_index]
        located = [
            (column, cell, str(getattr(cell, "text", "") or ""))
            for column, cell in enumerate(row_cells_by_column(row, width))
            if cell is not None and clean_label(getattr(cell, "text", ""))
        ]
        item_index = 0
        while item_index < len(located):
            label_column, label_cell, label_raw = located[item_index]
            inline = _split_tax_inline_fact(label_raw)
            if inline is not None:
                label, raw = inline
                source_cells = ((label_column, label_cell, "label_and_raw"),)
                item_index += 1
            elif item_index + 1 < len(located):
                value_column, value_cell, raw = located[item_index + 1]
                label = _normalize_tax_fact_label(label_raw)
                source_cells = ((label_column, label_cell, "label"), (value_column, value_cell, "raw"))
                item_index += 2
            else:
                break
            facts.append(
                {
                    "label": label,
                    "raw": raw,
                    "source": _tax_table_fact_source(
                        table,
                        row,
                        table_row_index=row_index,
                        cells=source_cells,
                        page_number=page_number,
                    ),
                }
            )
        if located:
            source_rows.append(_tax_source_row_index(row, row_index))
    if not facts:
        return []
    return [
        {
            "kind": "table_fact_region",
            "start_ordinal": 1,
            "source_page": page_number,
            "source_row_start": min(source_rows),
            "source_row_end": max(source_rows),
            "facts": facts,
        }
    ]


def _is_tax_context_continuation_row(row: Any, width: int) -> bool:
    values = row_texts(row, width)
    nonempty = [clean_label(value) for value in values if clean_label(value)]
    if not nonempty or _is_tax_data_row(values):
        return False
    joined = "".join(nonempty)
    if "：" in joined or ":" in joined or ("是否" in joined and _TABLE_CHOICE_MARK_RE.search(joined)):
        return True
    amount_header_score = sum(bool(_TAX_AMOUNT_LABEL_RE.search(value)) for value in nonempty)
    if _tax_header_score(values) >= 1 or amount_header_score >= 2:
        return False
    return 2 <= len(nonempty) <= 6


def _split_tax_inline_fact(value: str) -> tuple[str, str] | None:
    match = re.match(r"^(?P<label>[^：:]{1,80})[：:](?P<raw>.*)$", value, flags=re.DOTALL)
    if match is None:
        return None
    raw = match.group("raw").strip()
    if not raw or re.fullmatch(r"[（(][^）)]{1,16}[）)]", clean_label(raw)):
        return None
    return _normalize_tax_fact_label(match.group("label")), raw


def _normalize_tax_fact_label(value: str) -> str:
    text = _normalize_tax_display_text(value).strip().rstrip("：:")
    match = re.match(r"^(?P<label>.*?)[：:](?P<qualifier>[（(][^）)]{1,16}[）)])$", text)
    if match:
        return f"{match.group('label')}{match.group('qualifier')}"
    return text


def _tax_table_fact_source(
    table: Any,
    row: Any,
    *,
    table_row_index: int,
    cells: tuple[tuple[int, Any, str], ...],
    page_number: int,
) -> dict[str, Any]:
    """Build auditable source facts for one label/value pair without normalizing its value."""

    page = int(getattr(row, "source_page", 0) or getattr(table, "page", 0) or page_number)
    table_id = str(getattr(table, "table_id", "") or "table")
    source_row = _tax_source_row_index(row, table_row_index)
    refs: list[dict[str, Any]] = []
    evidence_ids: list[str] = []
    boxes: list[list[float]] = []
    confidences: list[float] = []
    for column, cell, field_name in cells:
        cell_evidence = [str(value) for value in (getattr(cell, "evidence_ids", None) or []) if value]
        evidence_ids.extend(cell_evidence)
        bbox = getattr(cell, "bbox", None)
        bounded_bbox: list[float] | None = None
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            try:
                bounded_bbox = [float(value) for value in bbox]
                boxes.append(bounded_bbox)
            except (TypeError, ValueError):
                pass
        cell_refs = [dict(ref) for ref in (getattr(cell, "source_cell_refs", None) or []) if isinstance(ref, dict)]
        if not cell_refs and (cell_evidence or bounded_bbox is not None):
            cell_refs = [{"page": page, "table_id": table_id, "row": source_row, "col": column}]
        for ref in cell_refs:
            ref.setdefault("page", page)
            ref.setdefault("table_id", table_id)
            ref.setdefault("row", source_row)
            ref.setdefault("col", column)
            ref.setdefault("field_name", field_name)
            if ref not in refs:
                refs.append(ref)
        try:
            confidences.append(max(0.0, min(1.0, float(getattr(cell, "confidence", 1.0) or 0.0))))
        except (TypeError, ValueError):
            pass
    source: dict[str, Any] = {
        "page": page,
        "page_range": [page, page],
        "table_id": table_id,
        "physical_table_id": table_id,
        "table_row_index": table_row_index,
        "source_row_index": source_row,
        "confidence": min(confidences or [0.0]),
    }
    if refs:
        source["source_cell_refs"] = refs
    if evidence_ids:
        source["evidence_ids"] = list(dict.fromkeys(evidence_ids))
    if boxes:
        source["bbox"] = [
            min(box[0] for box in boxes),
            min(box[1] for box in boxes),
            max(box[2] for box in boxes),
            max(box[3] for box in boxes),
        ]
    return source


def _tax_source_row_index(row: Any, fallback: int) -> int:
    declared = getattr(row, "source_row_index", None)
    try:
        value = int(declared) if declared is not None else fallback
    except (TypeError, ValueError):
        return fallback
    return value if value >= 0 else fallback


def _tax_header_score(values: list[str]) -> int:
    labels = [clean_label(value) for value in values if clean_label(value)]
    if any(any(identity in value for identity in (*_SUBJECT_LABELS, *_SUBJECT_ID_LABELS)) for value in labels):
        return 0
    return sum(bool(_TAX_HEADER_RE.search(value)) for value in labels)


def _tax_column_key(label: str, index: int, occurrence: int) -> str:
    compact = clean_label(label)
    if compact == "分组":
        return "section_name"
    if compact == "明细项目":
        return "item_name"
    if compact == "项目类别":
        return "item_category"
    if compact == "项目子类":
        return "item_subcategory"
    if compact == "适用情况":
        return "applicability_status"
    if compact == "项目":
        return "item_category" if occurrence == 0 else "item_name"
    if "项目及栏次" in compact:
        return "item_and_line_no" if occurrence == 0 else "item_name"
    if "栏次" in compact or "序号" in compact:
        return "line_no"
    mappings = {
        "一般项目/本月数": "general_current_month",
        "一般项目/本年累计": "general_year_to_date",
        "即征即退项目/本月数": "immediate_refund_current_month",
        "即征即退项目/本年累计": "immediate_refund_year_to_date",
    }
    if compact in mappings:
        return mappings[compact]
    stable_labels = {
        "序号": "sequence_no",
        "份数": "invoice_count",
        "发票种类": "invoice_type",
        "发票代码": "invoice_code",
        "发票号码": "invoice_number",
        "开票日期": "invoice_date",
        "销售额": "sales_amount",
        "销项(应纳)税额": "output_or_payable_tax_amount",
        "金额": "amount",
        "税额": "tax_amount",
        "价税合计": "amount_with_tax",
        "税率": "tax_rate",
        "征收率": "levy_rate",
        "期初余额": "opening_balance",
        "本期发生额": "current_period_amount",
        "本期扣除额": "current_period_deduction",
        "期末余额": "ending_balance",
        "减免性质代码及名称": "relief_code_and_name",
        "减税性质代码及名称": "tax_reduction_code_and_name",
        "免税性质代码及名称": "tax_exemption_code_and_name",
        "抵减项目": "deduction_item",
        "加计抵减项目": "additional_deduction_item",
        "本期应抵减税额": "current_period_eligible_deduction",
        "本期实际抵减税额": "current_period_actual_deduction",
        "本期调减额": "current_period_reduction",
        "本期可抵减额": "current_period_available_deduction",
        "本期实际抵减额": "current_period_actual_deduction",
        "税(费)种": "tax_or_fee_type",
        "税费种": "tax_or_fee_type",
        "税(费)率": "tax_or_fee_rate",
        "增值税税额": "vat_tax_amount",
        "增值税免抵税额": "vat_exemption_credit_amount",
        "留抵退税本期扣除额": "current_period_refund_deduction",
        "本期应纳税(费)额": "current_period_tax_payable",
        "减免性质代码": "relief_code",
        "减免税(费)额": "relief_amount",
        "减征比例(%)": "reduction_rate",
        "减征额": "reduction_amount",
        "本期抵免金额": "current_period_credit_amount",
        "本期已缴税(费)额": "current_period_tax_paid",
        "本期应补(退)税(费)额": "current_period_tax_due_or_refund",
        "免征增值税项目销售额": "exempt_sales_amount",
        "免税销售额扣除项目本期实际扣除金额": "exempt_sales_deduction_amount",
        "扣除后免税销售额": "exempt_sales_after_deduction",
        "免税销售额对应的进项税额": "input_tax_for_exempt_sales",
        "免税额": "tax_exemption_amount",
    }
    if compact in stable_labels:
        return stable_labels[compact]
    composite_key = _tax_composite_column_key(compact)
    if composite_key:
        return composite_key
    return f"column_{index + 1:02d}"


def _correct_tax_header_label(label: str) -> str:
    compact = clean_label(label)
    return f"一{compact}" if compact.startswith("般项目") else compact


def _normalize_tax_header_label(label: str) -> str:
    """Remove a known merged-parent bleed without changing source cells."""

    compact = clean_label(label)
    service_deduction = "服务、不动产和无形资产扣除项目本期实际扣除金额"
    if compact == f"{service_deduction}/价税合计":
        return service_deduction
    return label


def _normalize_tax_column_specs(columns: list[ColumnSpec]) -> list[ColumnSpec]:
    """Apply tax-form types locally so unrelated financial plugins are unchanged."""

    normalized: list[ColumnSpec] = []
    for column in columns:
        value_type = column.value_type
        if column.key not in _TAX_TEXT_COLUMN_KEYS and _TAX_AMOUNT_LABEL_RE.search(clean_label(column.label)):
            value_type = "decimal"
        normalized.append(
            ColumnSpec(
                source_index=column.source_index,
                key=column.key,
                label=column.label,
                value_type=value_type,
            )
        )
    return normalized


def _normalize_tax_text_records(records: list[dict[str, Any]], columns: list[ColumnSpec]) -> None:
    """Make semantic text readable while preserving exact raw and canonical values."""

    text_keys = {column.key for column in columns if column.value_type == "string"}
    for record in records:
        raw = record.get("raw") if isinstance(record.get("raw"), dict) else {}
        normalized = record.get("normalized") if isinstance(record.get("normalized"), dict) else {}
        for key in text_keys:
            normalized[key] = _normalize_tax_display_text(str(raw.get(key, "") or ""))


def _normalize_profiled_line_formulas(
    records: list[dict[str, Any]],
    *,
    columns: list[ColumnSpec],
    form_title: str,
) -> list[str]:
    """Recover statutory formulas only after a complete form-profile match."""

    matched = _matched_tax_form_profile(records, columns=columns, form_title=form_title)
    if matched is None:
        return []

    records_by_line: dict[int, dict[str, Any]] = {}
    for record in records:
        raw = record.get("raw") if isinstance(record.get("raw"), dict) else {}
        match = re.match(r"^\s*(\d{1,3})", str(raw.get("line_no", "") or ""))
        if match:
            records_by_line[int(match.group(1))] = record

    warnings: list[str] = []
    for line, expected in dict(matched.get("line_formulas") or {}).items():
        record = records_by_line.get(int(line))
        if record is None:
            continue
        raw = record.get("raw") if isinstance(record.get("raw"), dict) else {}
        observed = str(raw.get("line_no", "") or "")
        normalized_expected = str(expected).replace("−", "-")
        if re.sub(r"\D", "", observed) != re.sub(r"\D", "", normalized_expected):
            warnings.append(f"precision:tax_formula_digit_mismatch:line={line}")
            add_review_reason(record, "tax_formula_digit_mismatch")
            continue
        observed_compact = clean_label(observed).replace("−", "-")
        if observed_compact == normalized_expected:
            normalized = record.get("normalized") if isinstance(record.get("normalized"), dict) else {}
            normalized["line_no"] = normalized_expected
            continue
        _replace_projected_value(
            record,
            "line_no",
            normalized_expected,
            method="tax_statutory_formula_profile",
            confidence=1.0,
        )
    return warnings


def _matched_tax_form_profile(
    records: list[dict[str, Any]],
    *,
    columns: list[ColumnSpec],
    form_title: str,
) -> dict[str, Any] | None:
    """Return a tax form profile only after its full structural signature matches."""

    column_keys = {column.key for column in columns}
    observed_lines: set[int] = set()
    for record in records:
        raw = record.get("raw") if isinstance(record.get("raw"), dict) else {}
        match = re.match(r"^\s*(\d{1,3})", str(raw.get("line_no", "") or ""))
        if match:
            observed_lines.add(int(match.group(1)))

    for profile in _TAX_FORM_PROFILES:
        if not isinstance(profile, dict) or str(profile.get("title_contains") or "") not in form_title:
            continue
        required_columns = {str(value) for value in profile.get("required_columns") or []}
        required_lines = {int(value) for value in profile.get("required_lines") or []}
        if not required_columns <= column_keys or not required_lines <= observed_lines:
            continue
        return profile
    return None


def _normalize_tax_display_text(value: str) -> str:
    """Join layout-only Chinese line wraps without rewriting source glyphs."""

    text = unicodedata.normalize("NFC", str(value or "")).strip()
    cjk = r"\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF"
    text = re.sub(rf"(?<=[{cjk}])\s*[\r\n]+\s*(?=[{cjk}])", "", text)
    text = re.sub(r"\s*[\r\n]+\s*", " ", text)
    return re.sub(r"[\t ]+", " ", text).strip()


def _normalize_tax_span_labels(labels: list[str]) -> list[str]:
    normalized = list(labels)
    span_indexes = [index for index, label in enumerate(normalized) if clean_label(label) == "项目及栏次"]
    if len(span_indexes) == 2:
        normalized[span_indexes[0]] = "项目"
        normalized[span_indexes[1]] = "栏次"
    elif len(span_indexes) >= 3:
        replacements = ["项目类别", "项目子类", "明细项目", "栏次"]
        for offset, index in enumerate(span_indexes):
            normalized[index] = replacements[min(offset, len(replacements) - 1)]
    return normalized


def _normalize_tax_main_amount_labels(
    labels: list[str],
    entries: list[tuple[int, Any]] | None = None,
) -> list[str]:
    """Restore the fixed four amount columns when one merged parent label bleeds into its neighbor."""

    amount_indexes = [index for index, label in enumerate(labels) if re.search(r"本月数|本年累计", clean_label(label))]
    if entries:
        active = set(active_columns(labels, entries))
        amount_indexes = [index for index in amount_indexes if index in active]
    compact = "|".join(clean_label(label) for label in labels)
    if len(amount_indexes) != 4 or "一般项目" not in compact or "即征即退项目" not in compact:
        return labels
    normalized = list(labels)
    for index, label in zip(
        amount_indexes,
        ("一般项目/本月数", "一般项目/本年累计", "即征即退项目/本月数", "即征即退项目/本年累计"),
        strict=True,
    ):
        normalized[index] = label
    return normalized


def _normalize_statutory_additional_tax_labels(labels: list[str]) -> list[str]:
    """Restore the official 14-column additional-tax header when merged parents bleed across columns."""

    if len(labels) != len(_STATUTORY_ADDITIONAL_TAX_LABELS):
        return labels
    compact = "|".join(clean_label(label) for label in labels)
    signature = (
        "增值税税额",
        "增值税免抵税额",
        "留抵退税本期扣除额",
        "减免性质代码",
        "减征比例",
        "本期抵免金额",
    )
    return list(_STATUTORY_ADDITIONAL_TAX_LABELS) if all(marker in compact for marker in signature) else labels


def _tax_composite_column_key(compact: str) -> str:
    mappings = (
        ("开具增值税专用发票/销售额", "vat_special_invoice_sales_amount"),
        ("开具增值税专用发票/销项(应纳)税额", "vat_special_invoice_tax_amount"),
        ("开具其他发票/销售额", "other_invoice_sales_amount"),
        ("开具其他发票/销项(应纳)税额", "other_invoice_tax_amount"),
        ("未开具发票/销售额", "unissued_invoice_sales_amount"),
        ("未开具发票/销项(应纳)税额", "unissued_invoice_tax_amount"),
        ("纳税检查调整/销售额", "tax_inspection_adjustment_sales_amount"),
        ("纳税检查调整/销项(应纳)税额", "tax_inspection_adjustment_tax_amount"),
        ("合计/销售额", "total_sales_amount"),
        ("合计/销项(应纳)税额", "total_tax_amount"),
        ("合计/价税合计", "total_amount_with_tax"),
        (
            "服务、不动产和无形资产扣除项目本期实际扣除金额",
            "service_deduction_actual_amount",
        ),
        ("扣除后/含税(免税)销售额", "sales_amount_after_deduction"),
        ("扣除后/销项(应纳)税额", "tax_amount_after_deduction"),
        ("本期服务、不动产和无形资产价税合计额(免税销售额)", "current_service_amount_with_tax"),
        ("服务、不动产和无形资产扣除项目/期初余额", "service_deduction_opening_balance"),
        ("服务、不动产和无形资产扣除项目/本期发生额", "service_deduction_current_amount"),
        ("服务、不动产和无形资产扣除项目/本期应扣除金额", "service_deduction_eligible_amount"),
        ("服务、不动产和无形资产扣除项目/本期实际扣除金额", "service_deduction_actual_amount"),
        ("服务、不动产和无形资产扣除项目/期末余额", "service_deduction_ending_balance"),
    )
    return next((key for label, key in mappings if compact == label), "")


def _is_tax_data_row(values: list[str]) -> bool:
    nonempty = [clean_label(value) for value in values if clean_label(value)]
    if len(nonempty) < 2:
        return False
    numeric = sum(amount_like(value) or bool(_ORDINAL_RE.fullmatch(clean_label(value))) for value in values)
    return numeric >= 1


def _is_ordinal_row(values: list[str]) -> bool:
    nonempty = [clean_label(value) for value in values if clean_label(value)]
    return len(nonempty) >= 3 and sum(bool(_ORDINAL_RE.fullmatch(value)) for value in nonempty) / len(nonempty) >= 0.7


def _tax_ordinal_header_indexes(
    rows: list[Any],
    *,
    start: int,
    end: int,
    width: int,
) -> list[int]:
    """Return source ordinal/formula rows directly beneath one tax header band."""

    indexes: list[int] = []
    for row_index in range(start, end):
        values = row_texts(rows[row_index], width)
        if not any(clean_label(value) for value in values):
            continue
        if _is_ordinal_row(values):
            indexes.append(row_index)
            continue
        break
    return indexes


def _tax_column_header_bands(
    table: Any,
    rows: list[Any],
    *,
    header_indexes: list[int],
    ordinal_header_indexes: set[int],
    columns: list[ColumnSpec],
    page_number: int,
    records: list[dict[str, Any]],
    form_title: str,
) -> dict[str, list[dict[str, Any]]]:
    """Keep physical tax header levels, including statutory formulas, as JSON facts."""

    table_id = str(getattr(table, "table_id", "") or "table")
    result: dict[str, list[dict[str, Any]]] = {}
    for column in columns:
        bands: list[dict[str, Any]] = []
        for level, row_index in enumerate(header_indexes, start=1):
            if not 0 <= row_index < len(rows):
                continue
            row = rows[row_index]
            located = _tax_header_cell_for_column(row, column.source_index)
            if located is None:
                continue
            cell, source_column, column_span = located
            raw = str(getattr(cell, "text", "") or "")
            value = clean_label(raw)
            if not value:
                continue
            source_row = int(getattr(row, "source_row_index", -1) or -1)
            if source_row < 0:
                source_row = row_index
            source: dict[str, Any] = {
                "page": int(getattr(row, "source_page", 0) or getattr(table, "page", 0) or page_number),
                "table_id": table_id,
                "row": source_row,
                "col": source_column,
                "col_span": column_span,
            }
            bbox = getattr(cell, "bbox", None)
            if isinstance(bbox, (list, tuple)):
                source["bbox"] = [float(coordinate) for coordinate in bbox]
            refs = [dict(ref) for ref in (getattr(cell, "source_cell_refs", None) or []) if isinstance(ref, dict)]
            if refs:
                source["source_cell_refs"] = refs
            evidence_ids = [
                str(evidence_id) for evidence_id in (getattr(cell, "evidence_ids", None) or []) if evidence_id
            ]
            if evidence_ids:
                source["evidence_ids"] = list(dict.fromkeys(evidence_ids))
            bands.append(
                {
                    "level": level,
                    "role": "ordinal" if row_index in ordinal_header_indexes else "label",
                    "value": _normalize_tax_display_text(raw),
                    "raw": raw,
                    "confidence": float(getattr(cell, "confidence", 1.0) or 0.0),
                    "source": source,
                }
            )
        if bands:
            result[column.key] = bands
    return _normalize_profiled_tax_header_bands(
        result,
        records=records,
        columns=columns,
        form_title=form_title,
    )


def _normalize_profiled_tax_header_bands(
    bands_by_key: dict[str, list[dict[str, Any]]],
    *,
    records: list[dict[str, Any]],
    columns: list[ColumnSpec],
    form_title: str,
) -> dict[str, list[dict[str, Any]]]:
    """Repair display header paths only after the complete statutory profile matches."""

    if _matched_tax_form_profile(records, columns=columns, form_title=form_title) is None:
        return bands_by_key

    parent_by_key = {
        "general_current_month": "一般项目",
        "general_year_to_date": "一般项目",
        "immediate_refund_current_month": "即征即退项目",
        "immediate_refund_year_to_date": "即征即退项目",
    }
    if not set(parent_by_key) <= {column.key for column in columns}:
        return bands_by_key

    for key, expected_parent in parent_by_key.items():
        bands = bands_by_key.setdefault(key, [])
        parent_band = next(
            (
                band
                for band in bands
                if band.get("role") == "label"
                and (
                    "般项目" in clean_label(band.get("value", ""))
                    or "即征即退项目" in clean_label(band.get("value", ""))
                )
            ),
            None,
        )
        if parent_band is not None:
            parent_band["value"] = expected_parent
            parent_band["raw"] = expected_parent
            continue

        sibling = next(
            (
                band
                for sibling_key, sibling_bands in bands_by_key.items()
                if sibling_key != key and parent_by_key.get(sibling_key) == expected_parent
                for band in sibling_bands
                if band.get("role") == "label" and expected_parent in clean_label(band.get("value", ""))
            ),
            None,
        )
        if sibling is None:
            continue
        derived = {**sibling, "value": expected_parent, "raw": ""}
        if isinstance(sibling.get("source"), dict):
            derived["source"] = dict(sibling["source"])
        bands.append(derived)
        bands.sort(key=lambda band: int(band.get("level") or 0))
    return bands_by_key


def _tax_header_cell_for_column(row: Any, target_column: int) -> tuple[Any, int, int] | None:
    """Locate the physical (possibly spanning) header cell covering one source column."""

    for fallback, cell in enumerate(getattr(row, "cells", None) or []):
        source_column = getattr(cell, "col_index", None)
        source_column = fallback if source_column is None else int(source_column)
        column_span = max(1, int(getattr(cell, "col_span", 1) or 1))
        if source_column <= target_column < source_column + column_span:
            return cell, source_column, column_span
    return None


def _is_orphan_ordinal_row(values: list[str]) -> bool:
    nonempty = [clean_label(value) for value in values if clean_label(value)]
    return len(nonempty) == 1 and bool(_ORDINAL_RE.fullmatch(nonempty[0]))


def _split_tax_entries_on_schema_drift(
    labels: list[str],
    entries: list[tuple[int, Any]],
) -> list[tuple[list[str], list[tuple[int, Any]], int]]:
    """Split a form when numeric columns persistently become a different text schema."""

    if len(entries) < 7:
        return [(labels, entries, 0)]
    width = len(labels)
    drift_at: int | None = None
    for column in range(width):
        baseline = [
            clean_label(row_texts(row, width)[column])
            for _row_index, row in entries[:4]
            if clean_label(row_texts(row, width)[column])
        ]
        if len(baseline) < 3 or sum(_is_tax_profile_number(value) for value in baseline) < 3:
            continue
        for index in range(4, len(entries) - 2):
            values = [clean_label(row_texts(row, width)[column]) for _row_index, row in entries[index : index + 3]]
            if all(len(value) >= 2 and not _is_tax_profile_number(value) for value in values):
                drift_at = index if drift_at is None else min(drift_at, index)
                break
    if drift_at is None:
        return [(labels, entries, 0)]
    trailing = entries[drift_at:]
    return [
        (labels, entries[:drift_at], 0),
        (_tax_drift_labels(width, trailing), trailing, drift_at),
    ]


def _is_tax_profile_number(value: str) -> bool:
    compact = clean_label(value).replace("，", ",").replace("−", "-")
    return bool(amount_like(compact) or _ORDINAL_RE.fullmatch(compact) or re.fullmatch(r"""[-—一二工"'|]+""", compact))


def _tax_drift_labels(width: int, entries: list[tuple[int, Any]]) -> list[str]:
    values_by_column: dict[int, list[str]] = {index: [] for index in range(width)}
    for _row_index, row in entries:
        for column, value in enumerate(row_texts(row, width)):
            compact = clean_label(value)
            if compact:
                values_by_column[column].append(compact)

    ordinal_columns = {
        column
        for column, values in values_by_column.items()
        if len(values) >= 2 and sum(bool(re.fullmatch(r"\d{1,3}", value)) for value in values) / len(values) >= 0.7
    }
    amount_columns = {
        column
        for column, values in values_by_column.items()
        if column not in ordinal_columns
        and len(values) >= 2
        and sum(amount_like(value) for value in values) / len(values) >= 0.7
    }
    applicability_columns = {
        column
        for column, values in values_by_column.items()
        if any(
            re.search(r"□|■|☑|☒", value) or (len(value) <= 8 and value in {"是", "否", "是否", "是/否", "是否适用"})
            for value in values
        )
    }
    text_columns = [
        column
        for column, values in values_by_column.items()
        if values and column not in ordinal_columns | amount_columns | applicability_columns
    ]
    detail_column = max(text_columns, key=lambda column: (len(values_by_column[column]), column), default=None)
    labels = ["" for _column in range(width)]
    for column in ordinal_columns:
        labels[column] = "栏次"
    for column in amount_columns:
        labels[column] = "金额"
    for column in applicability_columns:
        labels[column] = "适用情况"
    for column in text_columns:
        labels[column] = "明细项目" if column == detail_column else "分组"
    return labels


def _tax_groups_for_entry_slice(
    groups: list[dict[str, Any]],
    *,
    entry_offset: int,
    entry_count: int,
) -> list[dict[str, Any]]:
    sliced: list[dict[str, Any]] = []
    start = entry_offset + 1
    end = entry_offset + entry_count
    for group in groups:
        ordinal = int(group.get("start_ordinal") or 0)
        if not start <= ordinal <= end:
            continue
        sliced.append({**group, "start_ordinal": ordinal - entry_offset})
    return sliced


def _tax_part_title_hint(entries: list[tuple[int, Any]]) -> str:
    for _row_index, row in entries[:3]:
        candidates = [clean_label(value) for value in row_texts(row, len(getattr(row, "cells", None) or []))]
        for candidate in candidates:
            if (
                4 <= len(candidate) <= 80
                and not _is_tax_profile_number(candidate)
                and not re.search(r"□|■|☑|☒", candidate)
                and candidate not in {"是", "否", "是否", "是/否", "是否适用"}
            ):
                return candidate
    return ""


def _tax_header_title_hint(rows: list[Any], header_indexes: list[int], width: int) -> str:
    """Return a numbered source section title adjacent to one physical header band."""

    if not header_indexes:
        return ""
    start = max(0, min(header_indexes) - 2)
    for row_index in range(start, max(header_indexes) + 1):
        for value in row_texts(rows[row_index], width):
            candidate = clean_label(value)
            if 3 <= len(candidate) <= 80 and re.match(r"^[一二三四五六七八九十]+[、.]", candidate):
                return candidate
    return ""


def _continuation_entries(table: Any) -> list[tuple[int, Any]]:
    entries: list[tuple[int, Any]] = []
    width = table_width(table)
    for row_index, row in enumerate(getattr(table, "rows", None) or []):
        values = row_texts(row, width)
        if _FORM_FOOTER_RE.search("".join(clean_label(value) for value in values)):
            break
        if _is_tax_data_row(values):
            entries.append((row_index, row))
    return entries


def _continuation_matches(table: Any, segment: ProjectedSegment) -> bool:
    width = table_width(table)
    required_width = max((column.source_index for column in segment.columns), default=-1) + 1
    return width >= required_width and width - required_width <= 1


def _tax_dataset_id(index: int) -> str:
    return "tax_return_main" if index == 1 else f"tax_return_attachment_{index - 1:03d}"


def _renumber_records(segment: ProjectedSegment) -> None:
    replacements: dict[str, str] = {}
    for index, record in enumerate(segment.records, start=1):
        previous = str(record.get("record_id") or "")
        current = f"{segment.dataset_id}:r{index:06d}"
        record["record_id"] = current
        replacements[previous] = current
    for group in segment.row_groups:
        start = str(group.get("start_record_id") or "")
        if start in replacements:
            group["start_record_id"] = replacements[start]


def _resolve_tax_row_groups(groups: list[dict[str, Any]], records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    resolved: list[dict[str, Any]] = []
    seen_groups: set[tuple[str, str, str, int]] = set()
    for group in groups:
        ordinal = int(group.get("start_ordinal") or 0)
        if not 1 <= ordinal <= len(records):
            continue
        start_record_id = str(records[ordinal - 1].get("record_id") or "")
        title = str(group.get("title") or "").strip()
        facts = [fact for fact in (group.get("facts") or []) if isinstance(fact, dict)]
        if not start_record_id or (not title and not facts):
            continue
        signature = (
            start_record_id,
            str(group.get("kind") or "heading"),
            title,
            int(group.get("source_row_start") or -1),
        )
        if signature in seen_groups:
            continue
        seen_groups.add(signature)
        item = {key: value for key, value in group.items() if key != "start_ordinal"}
        item["start_record_id"] = start_record_id
        item["source_page"] = int(group.get("source_page") or 1)
        if not title:
            item.pop("title", None)
        resolved.append(item)
    return resolved


def _tax_segment_title(
    rows: list[Any],
    *,
    header_start: int,
    page_text: str,
    fallback_index: int,
    local_hint: str = "",
) -> str:
    preceding_rows = rows[:header_start]
    local_candidates: list[str] = []
    for row in reversed(preceding_rows[-12:]):
        values = [clean_label(value) for value in row_texts(row, table_width_from_rows(rows)) if clean_label(value)]
        if values:
            local_candidates.append("".join(values))
    table_title_text = "\n".join(
        " ".join(value for value in row_texts(row, table_width_from_rows(rows)) if str(value or "").strip())
        for row in preceding_rows
    )
    page_title = _tax_page_form_title(page_text) or _tax_page_form_title(table_title_text)
    local_title = clean_label(local_hint)
    if not local_title:
        local_title = next(
            (
                candidate
                for candidate in local_candidates
                if 3 <= len(candidate) <= 80
                and (
                    re.search(r"^[一二三四五六七八九十]+[、.]", candidate)
                    or (
                        not re.search(r"\d|[=＝]|[,.]\d", candidate)
                        and re.search(r"进项税额|抵减情况|减税项目|免税项目|^其他$", candidate)
                    )
                )
            ),
            "",
        )
    fallback = _default_tax_title(_tax_dataset_id(fallback_index))
    if page_title and local_title and local_title not in page_title:
        return f"{page_title} - {local_title}"
    return page_title or local_title or fallback


def _tax_page_form_title(page_text: str) -> str:
    """Return the explicit source form title, including its parenthetical subtitle."""

    text = unicodedata.normalize("NFC", str(page_text or ""))
    parenthetical = r"[（(][^（）()\r\n]{2,50}[）)]"
    patterns = (
        rf"增值税及附加税费申报表(?:[（(]一般纳税人适用[）)])?"
        rf"附列资料[（(][一二三四五][）)](?:\s*{parenthetical})?",
        r"增值税减免税申报明细表",
        r"增值税及附加税费申报表(?:\s*[（(]一般纳税人适用[）)])?",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return re.sub(r"\s+", "", match.group(0)).strip()
    return ""


def table_width_from_rows(rows: list[Any]) -> int:
    """Return a safe physical width for a tax title scan."""

    return max((len(getattr(row, "cells", None) or []) for row in rows), default=0)


def _default_tax_title(dataset_id: str) -> str:
    if dataset_id == "tax_return_main":
        return "纳税申报主表"
    match = re.search(r"_(\d{3})$", dataset_id)
    number = int(match.group(1)) if match else 0
    return f"纳税申报表附表{number}" if number else "纳税申报附表"


def _tax_form_title(segment: ProjectedSegment) -> str:
    title = segment.title or _default_tax_title(segment.dataset_id)
    return title.split(" - ", 1)[0] if segment.kind == "tax_return" else title


def _tax_dataset_title(segment: ProjectedSegment) -> str:
    title = segment.title or _default_tax_title(segment.dataset_id)
    if segment.kind == "tax_return" and " - " in title:
        return title.split(" - ", 1)[1]
    return title


def _tax_section_id(segment: ProjectedSegment) -> str:
    if segment.kind == "tax_return":
        return f"section_tax_return_page_{segment.source_page:03d}"
    return f"section_{segment.dataset_id}"


def _tax_form_sections(segments: list[ProjectedSegment]) -> list[dict[str, Any]]:
    """Group logical datasets under their physical source form in page order."""

    sections: dict[str, dict[str, Any]] = {}
    for segment in segments:
        section_id = _tax_section_id(segment)
        pages = [int((record.get("source") or {}).get("page") or segment.source_page) for record in segment.records]
        page_start = min(pages, default=segment.source_page)
        page_end = max(pages, default=segment.source_page)
        existing = sections.get(section_id)
        if existing is not None:
            existing["page_start"] = min(int(existing["page_start"]), page_start)
            existing["page_end"] = max(int(existing["page_end"]), page_end)
            continue
        sections[section_id] = {
            "id": section_id,
            "title": _tax_form_title(segment),
            "type": "financial_statement" if segment.kind != "tax_return" else "tax_form",
            "page_start": page_start,
            "page_end": page_end,
        }
    return list(sections.values())


def _validate_tax_segment(segment: ProjectedSegment, warnings: list[str]) -> None:
    if not segment.records:
        return
    is_main = segment.dataset_id == "tax_return_main"
    if is_main:
        line_key = next(
            (column.key for column in segment.columns if column.key in {"line_no", "item_and_line_no"}),
            "",
        )
        line_numbers: list[int] = []
        for record in segment.records:
            raw = record.get("raw") if isinstance(record.get("raw"), dict) else {}
            normalized = record.get("normalized") if isinstance(record.get("normalized"), dict) else {}
            line_value = clean_label(normalized.get(line_key, "") or raw.get(line_key, "")) if line_key else ""
            match = re.match(r"^(\d{1,3})", line_value)
            if match:
                line_numbers.append(int(match.group(1)))
            if line_value and not _valid_tax_line_expression(line_value):
                warnings.append(
                    "TAX_LINE_NO_INVALID:"
                    f"dataset={segment.dataset_id}:record={record.get('record_id')}:"
                    f"field={line_key}:value={line_value}"
                )
                add_review_reason(record, "tax_line_no_invalid")

        if line_numbers and max(line_numbers) >= 20:
            expected = set(range(min(line_numbers), max(line_numbers) + 1))
            missing = sorted(expected.difference(line_numbers))
            if min(line_numbers) > 1:
                missing = [*range(1, min(line_numbers)), *missing]
            if missing:
                warnings.append(
                    f"TAX_LINE_SEQUENCE_GAP:dataset={segment.dataset_id}:missing={','.join(map(str, missing))}"
                )

    decimal_fields = {column.key for column in segment.columns if column.value_type == "decimal"}
    main_amount_fields = {
        "general_current_month",
        "general_year_to_date",
        "immediate_refund_current_month",
        "immediate_refund_year_to_date",
    }
    require_complete_amount_grid = is_main and main_amount_fields.issubset(decimal_fields)
    for record in segment.records:
        raw = record.get("raw") if isinstance(record.get("raw"), dict) else {}
        for field in decimal_fields:
            value = clean_label(raw.get(field, ""))
            if amount_like(value) or decimal_placeholder(value):
                continue
            if not value and not require_complete_amount_grid:
                continue
            code = "TAX_DECIMAL_TOKEN_MISSING" if not value else "TAX_DECIMAL_TOKEN_INVALID"
            reason = "tax_decimal_token_missing" if not value else "tax_decimal_token_invalid"
            warnings.append(
                f"{code}:dataset={segment.dataset_id}:record={record.get('record_id')}:"
                f"field={field}:value={value or '<empty>'}"
            )
            add_review_reason(record, reason)


def _remove_repeated_code_gridline_artifacts(segment: ProjectedSegment) -> None:
    """Remove a vertical grid stroke OCR'd as text only when it repeats in a code column."""

    for column in segment.columns:
        if "减免性质代码" not in clean_label(column.label):
            continue
        affected = [
            record
            for record in segment.records
            if clean_label((record.get("raw") or {}).get(column.key, "")) in {"|", "I"}
        ]
        if len(affected) < 2:
            continue
        for record in affected:
            _replace_projected_value(
                record,
                column.key,
                "",
                method="tax_repeated_gridline_artifact",
                confidence=1.0,
            )


def _valid_tax_line_expression(value: str) -> bool:
    compact = clean_label(value).replace("＝", "=").replace("−", "-")
    if re.fullmatch(r"\d{1,3}[A-Za-z]?", compact):
        return True
    if re.fullmatch(r"\d{1,3}[\(（].+[\)）]", compact):
        return True
    if not re.fullmatch(r"\d{1,3}=\(?\d{1,3}\)?(?:[+\-]\(?\d{1,3}\)?)+", compact):
        return False
    left, expression = compact.split("=", 1)
    references = [int(item) for item in re.findall(r"\d{1,3}", expression)]
    return bool(references) and all(reference <= max(200, int(left) + 100) for reference in references)


def _replace_projected_value(
    record: dict[str, Any],
    field: str,
    corrected: str,
    *,
    method: str,
    confidence: float,
) -> bool:
    """Replace one deterministically recovered cell and retain its observed OCR value."""

    raw = record.get("raw") if isinstance(record.get("raw"), dict) else {}
    observed = str(raw.get(field, "") or "")
    if observed == corrected:
        return False
    for container_name in ("raw", "canonical_raw", "normalized"):
        container = record.get(container_name)
        if isinstance(container, dict):
            container[field] = corrected
    source = record.setdefault("source", {})
    corrections = source.setdefault("corrections", [])
    refs = [
        dict(ref)
        for ref in source.get("source_cell_refs") or []
        if isinstance(ref, dict) and str(ref.get("field_name") or "") == field
    ]
    correction: dict[str, Any] = {
        "field": field,
        "observed": observed,
        "corrected": corrected,
        "method": method,
        "confidence": confidence,
    }
    if refs:
        correction["source_refs"] = refs
    corrections.append(correction)
    return True


def _extract_tax_identity(parse_result: Any) -> tuple[dict[str, str], dict[str, Any], list[str]]:
    names: list[_IdentityCandidate] = []
    subject_ids: list[_IdentityCandidate] = []
    period_dates: list[_IdentityCandidate] = []
    document_dates: list[_IdentityCandidate] = []
    currencies: list[_IdentityCandidate] = []

    for page in getattr(parse_result, "pages", None) or []:
        page_number = int(getattr(page, "page_number", 0) or 1)
        for block in getattr(page, "texts", None) or []:
            content = str(getattr(block, "content", "") or "")
            confidence = float(getattr(block, "confidence", 0.0) or 0.0)
            evidence = [str(item) for item in (getattr(block, "evidence_ids", None) or []) if item]
            _collect_identity_row(
                [content],
                page_number,
                names,
                subject_ids,
                period_dates,
                document_dates,
                currencies,
                score=max(1, min(100, round(confidence * 100))),
                evidence=evidence,
                bbox=_identity_bbox([block]),
                source_kind="text",
            )
        for kv in getattr(page, "key_values", None) or []:
            key = str(getattr(kv, "key", "") or "")
            value = str(getattr(kv, "value", "") or "")
            evidence = [str(item) for item in getattr(kv, "evidence_ids", None) or []]
            _collect_identity_row(
                [key, value],
                page_number,
                names,
                subject_ids,
                period_dates,
                document_dates,
                currencies,
                score=_identity_score(kv, default=1.0),
                evidence=evidence,
                bbox=_identity_bbox([kv]),
                source_kind="key_value",
            )

        for table in getattr(page, "tables", None) or []:
            table_evidence = [str(item) for item in (getattr(table, "evidence_ids", None) or []) if item]
            header_values = [str(value or "") for value in (getattr(table, "headers", None) or [])]
            if header_values:
                _collect_identity_row(
                    header_values,
                    page_number,
                    names,
                    subject_ids,
                    period_dates,
                    document_dates,
                    currencies,
                    score=_identity_score(table, default=0.8),
                    evidence=table_evidence,
                    source_kind="table_header",
                )
            for row in getattr(table, "rows", None) or []:
                cells = list(getattr(row, "cells", None) or [])
                values = [str(getattr(cell, "text", "") or "") for cell in cells]
                evidence = list(
                    dict.fromkeys(
                        str(evidence_id)
                        for cell in cells
                        for evidence_id in (getattr(cell, "evidence_ids", None) or [])
                        if evidence_id
                    )
                )
                cell_scores = [_identity_score(cell, default=0.8) for cell in cells]
                _collect_identity_row(
                    values,
                    page_number,
                    names,
                    subject_ids,
                    period_dates,
                    document_dates,
                    currencies,
                    score=min(cell_scores or [80]),
                    evidence=evidence,
                    bbox=_identity_bbox(cells),
                    source_kind="table_row",
                )

    full_text = str(getattr(parse_result, "full_text", "") or getattr(parse_result, "raw_text", "") or "")
    _collect_identity_from_text(full_text, names, subject_ids)
    _collect_labeled_scalar_candidates(
        full_text,
        page_number=0,
        score=60,
        evidence=[],
        bbox=None,
        source_kind="full_text",
        period_dates=period_dates,
        document_dates=document_dates,
        currencies=currencies,
    )

    fields: dict[str, str] = {}
    details: dict[str, Any] = {}
    warnings: list[str] = []
    selected_name = _select_candidate(names, subject_name=True)
    selected_id = _select_candidate(subject_ids)
    name_variants = {
        normalize_subject_name(candidate.raw) for candidate in names if normalize_subject_name(candidate.raw)
    }
    if selected_name:
        cleaned_name = normalize_subject_name(selected_name.raw)
        fields["subject_name"] = cleaned_name
        fields["organization"] = cleaned_name
        name_detail = _field_detail(
            selected_name,
            value=cleaned_name,
            raw=bounded_subject_name_raw(selected_name.raw),
        )
        if len(name_variants) > 1:
            name_detail["review"] = "needs_review"
            warnings.append(
                "TAX_SUBJECT_IDENTITY_CONFLICT:"
                f"field=subject_name:selected_page={selected_name.page}:candidate_count={len(name_variants)}"
            )
        details["subject_name"] = name_detail
        details["organization"] = dict(name_detail)
    else:
        warnings.append("precision:tax_subject_name_missing")
    if selected_id:
        fields["subject_id"] = selected_id.raw
        details["subject_id"] = _field_detail(selected_id)
    distinct_period_dates = list(dict.fromkeys(candidate.value for candidate in period_dates if candidate.value))
    if distinct_period_dates:
        period_start = next(candidate for candidate in period_dates if candidate.value == distinct_period_dates[0])
        period_end_value = distinct_period_dates[1] if len(distinct_period_dates) > 1 else distinct_period_dates[0]
        period_end = next(candidate for candidate in period_dates if candidate.value == period_end_value)
        fields["period_start"] = period_start.value
        fields["period_end"] = period_end.value
        details["period_start"] = _field_detail(period_start)
        details["period_end"] = _field_detail(period_end)
    else:
        warnings.append("precision:tax_period_missing")
    if document_dates:
        fields["document_date"] = document_dates[0].value
        details["document_date"] = _field_detail(document_dates[0])
    if currencies:
        fields["currency_unit"] = currencies[0].value
        details["currency_unit"] = _field_detail(currencies[0])
    return fields, details, warnings


def _collect_identity_row(
    values: list[str],
    page_number: int,
    names: list[_IdentityCandidate],
    subject_ids: list[_IdentityCandidate],
    period_dates: list[_IdentityCandidate],
    document_dates: list[_IdentityCandidate],
    currencies: list[_IdentityCandidate],
    *,
    score: int = 80,
    evidence: list[str] | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    source_kind: str = "",
) -> None:
    source_evidence = list(evidence or [])
    for index, raw in enumerate(values):
        text = unicodedata.normalize("NFKC", str(raw or "")).strip()
        if not text:
            continue
        if any(label in clean_label(text) for label in _SUBJECT_LABELS):
            value = _value_after_label(text, _SUBJECT_LABELS) or _next_nonempty(values, index)
            if _valid_subject_name(value):
                names.append(
                    _identity_candidate(
                        value,
                        page_number,
                        score,
                        source_evidence,
                        bbox=bbox,
                        source_kind=source_kind,
                    )
                )
        if any(label in clean_label(text) for label in _SUBJECT_ID_LABELS):
            value = _value_after_label(text, _SUBJECT_ID_LABELS) or _next_nonempty(values, index)
            match = _SUBJECT_ID_RE.search(str(value or "").upper())
            if match:
                subject_ids.append(
                    _identity_candidate(
                        match.group(0),
                        page_number,
                        score,
                        source_evidence,
                        bbox=bbox,
                        source_kind=source_kind,
                    )
                )
        adjacent = _next_nonempty(values, index)
        _collect_labeled_scalar_candidates(
            f"{text} {adjacent}".strip(),
            page_number=page_number,
            score=score,
            evidence=source_evidence,
            bbox=bbox,
            source_kind=source_kind,
            period_dates=period_dates,
            document_dates=document_dates,
            currencies=currencies,
        )


def _collect_labeled_scalar_candidates(
    text: str,
    *,
    page_number: int,
    score: int,
    evidence: list[str],
    bbox: tuple[float, float, float, float] | None,
    source_kind: str,
    period_dates: list[_IdentityCandidate],
    document_dates: list[_IdentityCandidate],
    currencies: list[_IdentityCandidate],
) -> None:
    """Collect explicitly labeled tax dates and currency from one source span."""

    normalized = unicodedata.normalize("NFKC", str(text or ""))

    def candidate(raw: str, value: str) -> _IdentityCandidate:
        return _identity_candidate(
            raw,
            page_number,
            score,
            evidence,
            value=value,
            bbox=bbox,
            source_kind=source_kind,
        )

    for labels, target, limit in (
        (_PERIOD_LABELS, period_dates, 2),
        (_DOCUMENT_DATE_LABELS, document_dates, 1),
    ):
        starts = [normalized.find(label) for label in labels if normalized.find(label) >= 0]
        if not starts:
            continue
        scope = normalized[min(starts) : min(starts) + 120]
        for match in list(_DATE_RE.finditer(scope))[:limit]:
            value = f"{int(match.group('year')):04d}-{int(match.group('month')):02d}-{int(match.group('day')):02d}"
            target.append(candidate(match.group(0), value))

    starts = [normalized.find(label) for label in _CURRENCY_LABELS if normalized.find(label) >= 0]
    if starts:
        match = re.search(r"人民币|CNY|元", normalized[min(starts) : min(starts) + 50], re.IGNORECASE)
        if match:
            currencies.append(candidate(match.group(0), "CNY"))


def _collect_identity_from_text(
    text: str,
    names: list[_IdentityCandidate],
    subject_ids: list[_IdentityCandidate],
) -> None:
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    company_suffix = r"(?:有限责任公司|股份有限公司|有限公司|集团|企业|事务所|合作社|学校|医院|中心|研究院)"
    for label in _SUBJECT_LABELS:
        pattern = re.compile(
            re.escape(label) + r"(?:\([^)]*\)|（[^）]*）)?\s*[：:]?\s*" + rf"([^\n\r:：]{{2,80}}?{company_suffix})"
        )
        for match in pattern.finditer(normalized):
            value = match.group(1).strip()
            if _valid_subject_name(value):
                names.append(_identity_candidate(value, 0, 60, [], source_kind="full_text"))
    for label in _SUBJECT_ID_LABELS:
        start = normalized.find(label)
        if start < 0:
            continue
        match = _SUBJECT_ID_RE.search(normalized[start : start + 100].upper())
        if match:
            subject_ids.append(_identity_candidate(match.group(0), 0, 60, [], source_kind="full_text"))


def _identity_score(value: Any, *, default: float) -> int:
    try:
        confidence = max(0.0, min(1.0, float(getattr(value, "confidence", default) or default)))
    except (TypeError, ValueError):
        confidence = default
    return max(1, min(100, round(confidence * 100)))


def _identity_bbox(values: list[Any]) -> tuple[float, float, float, float] | None:
    boxes: list[tuple[float, float, float, float]] = []
    for value in values:
        bbox = getattr(value, "bbox", None)
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            boxes.append(tuple(float(coordinate) for coordinate in bbox))
        except (TypeError, ValueError):
            continue
    if not boxes:
        return None
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _identity_candidate(
    raw: str,
    page: int,
    score: int,
    evidence: list[str],
    *,
    value: str = "",
    bbox: tuple[float, float, float, float] | None = None,
    source_kind: str = "",
) -> _IdentityCandidate:
    return _IdentityCandidate(
        raw=str(raw or "").strip(),
        page=max(0, int(page or 0)),
        score=max(1, min(100, int(score))),
        value=str(value or "").strip(),
        evidence_ids=tuple(dict.fromkeys(str(item) for item in evidence if item)),
        bbox=bbox,
        source_kind=source_kind,
    )


def _select_candidate(
    candidates: list[_IdentityCandidate],
    *,
    subject_name: bool = False,
) -> _IdentityCandidate | None:
    if not candidates:
        return None

    def semantic_key(candidate: _IdentityCandidate) -> str:
        return normalize_subject_name(candidate.raw) if subject_name else clean_label(candidate.value or candidate.raw)

    grouped: dict[str, list[_IdentityCandidate]] = {}
    for candidate in candidates:
        key = semantic_key(candidate)
        if key:
            grouped.setdefault(key, []).append(candidate)
    if not grouped:
        return None
    selected_group = max(
        grouped.values(),
        key=lambda group: (
            max(candidate.score for candidate in group),
            len(group),
            any(candidate.evidence_ids for candidate in group),
            -min(candidate.page for candidate in group),
        ),
    )

    def candidate_rank(candidate: _IdentityCandidate) -> tuple[bool, int, bool, bool, int, int, int]:
        bounded_raw = bounded_subject_name_raw(candidate.raw) if subject_name else candidate.raw.strip()
        is_clean = candidate.raw.strip() == bounded_raw
        source_authority = {
            "key_value": 4,
            "text": 3,
            "table_row": 2,
            "table_header": 2,
            "full_text": 0,
        }.get(candidate.source_kind, 1)
        return (
            is_clean,
            source_authority,
            candidate.page > 0,
            bool(candidate.evidence_ids or candidate.bbox),
            -candidate.page,
            candidate.score,
            -len(candidate.raw),
        )

    return max(selected_group, key=candidate_rank)


def _field_detail(
    candidate: _IdentityCandidate,
    *,
    value: str | None = None,
    raw: str | None = None,
) -> dict[str, Any]:
    source_ref: dict[str, Any] = {}
    if candidate.page > 0:
        source_ref["page"] = candidate.page
    if candidate.bbox:
        source_ref["bbox"] = list(candidate.bbox)
    if candidate.evidence_ids:
        source_ref["evidence_ids"] = list(candidate.evidence_ids)
    has_source = bool(source_ref)
    if candidate.source_kind and has_source:
        source_ref["source"] = candidate.source_kind
    confidence = candidate.score / 100
    review = "needs_evidence"
    if has_source and confidence >= 0.85:
        review = "auto_accepted"
    elif has_source and confidence >= 0.6:
        review = "manual_optional"
    elif has_source:
        review = "needs_review"
    detail: dict[str, Any] = {
        "value": candidate.value or candidate.raw if value is None else value,
        "raw": candidate.raw if raw is None else raw,
        "confidence": confidence,
        "evidence_ids": list(candidate.evidence_ids),
        "source_refs": [source_ref] if source_ref else [],
        "review": review,
    }
    if candidate.page > 0:
        detail["source_page"] = candidate.page
    if candidate.bbox:
        detail["bbox"] = list(candidate.bbox)
    if candidate.source_kind:
        detail["source_kind"] = candidate.source_kind
    return detail


def _valid_subject_name(value: str) -> bool:
    compact = clean_label(value)
    if not 4 <= len(compact) <= 100:
        return False
    if any(label in compact for label in (*_SUBJECT_LABELS, *_SUBJECT_ID_LABELS, "栏次", "项目")):
        return False
    return bool(re.search(r"公司|企业|集团|事务所|合作社|学校|医院|中心|研究院", compact))


def _value_after_label(text: str, labels: tuple[str, ...]) -> str:
    for label in labels:
        match = re.search(re.escape(label) + r"(?:\([^)]*\)|（[^）]*）)?\s*[：:]?\s*(.+)$", text)
        if match:
            return match.group(1).strip()
    return ""


def _next_nonempty(values: list[str], index: int) -> str:
    return next((str(value).strip() for value in values[index + 1 :] if str(value or "").strip()), "")


def _page_text(page: Any) -> str:
    return "\n".join(str(getattr(block, "content", "") or "") for block in getattr(page, "texts", None) or [])


__all__ = ["derive_tax_return_projection"]
