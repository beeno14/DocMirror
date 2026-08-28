"""Source-conserving audit-report projection built only from sealed facts."""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Iterable
from typing import Any

from docmirror.plugins._base.financial_source_projection import ProjectedSegment
from docmirror.plugins._base.generic_community_adapter import derive_generic_projection
from docmirror.plugins._base.projector import ProjectionData
from docmirror.plugins.audit_report.table_projection import (
    audit_data_dictionary,
    audit_semantic,
    bind_datasets_to_sections,
    blocking_warning,
    canonicalize_audit_dataset_columns,
    dataset_blocking_warnings,
    dataset_pages,
    embedded_financial_pages,
    merge_cross_page_continuations,
    merge_horizontal_note_continuations,
    name_note_datasets,
    normalize_audit_label,
    normalize_audit_record,
    normalize_audit_text,
    page_lines,
    project_embedded_financial_statements,
    quality_warnings,
    record_keys,
    recover_note_text_continuations,
    recover_owner_equity_label_rows,
    repair_note_datasets,
    repair_stacked_note_headers,
    resolve_note_table_candidates,
    statement_kind,
    synchronize_audit_record_sources,
)

_HEADING_RE = re.compile(
    r"^(?P<marker>(?:(?:[一二三四五六七八九十百]{1,4}|\d{1,2})[、.．]|"
    r"[（(](?:[一二三四五六七八九十百]{1,4}|\d{1,2})[）)]))\s*(?P<title>\S.{0,70})$"
)
_AUDIT_NUMBER_RE = re.compile(
    r"(?P<number>[\u3400-\u9fff]{1,20}字\s*[\[【〔（(]\s*\d{4}\s*[\]】〕）)]\s*第?\s*[0-9A-Za-z-]{1,32}\s*号)"
)
_INCOMPLETE_AUDIT_NUMBER_RE = re.compile(
    r"(?P<number>[\u3400-\u9fff]{1,20}字\s*[\[【〔（(]\s*\d{4}\s*[\]】〕）)]\s*第?\s*号)"
)
_REGULATORY_NUMBER_RE = re.compile(r"报告编号\s*[:：]\s*(?P<number>[\u3400-\u9fffA-Za-z0-9-]{6,32})")
_STRICT_REGULATORY_NUMBER_RE = re.compile(r"^[\u3400-\u9fff]{1,3}[0-9A-Z][0-9A-Z-]{7,20}$")
_AUDIT_TITLE_RE = re.compile(r"(?P<title>20\d{2}\s*年度\s*审计报告)")
_AUDITOR_NAME_RE = re.compile(r"(?P<name>[\u3400-\u9fff（）()·]{2,40}会计师事务所有限公司)")
_REPORT_DATE_RE = re.compile(
    r"(?P<date>(?:20\d{2}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日)|"
    r"(?:[二〇○零ＯO一二三四五六七八九十]{4}\s*年\s*"
    r"[〇○零一二三四五六七八九十]{1,3}\s*月\s*[〇○零一二三四五六七八九十]{1,3}\s*日))"
)
_BARE_NOTE_REFERENCE_RE = re.compile(r"^(?:附注\s*)?[一二三四五六七八九十百]{1,4}[、.．]\s*\d{1,3}$")
_CORE_SECTION_TYPES: tuple[tuple[str, str], ...] = (
    ("形成审计意见的基础", "basis_for_opinion"),
    ("关键审计事项", "key_audit_matters"),
    ("持续经营", "going_concern"),
    ("强调事项", "emphasis_of_matter"),
    ("其他信息", "other_information"),
    ("管理层对财务报表的责任", "management_responsibility"),
    ("管理层和治理层", "management_responsibility"),
    ("注册会计师对财务报表审计的责任", "auditor_responsibility"),
    ("审计意见", "audit_opinion"),
)
_PUBLIC_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "subject_name": ("subject_name", "被审计单位", "编制单位", "公司名称", "单位名称"),
    "subject_id": ("subject_id", "统一社会信用代码", "纳税人识别号"),
    "document_date": ("document_date", "报告日期", "审计报告日期"),
    "period_start": ("period_start", "期间开始", "审计期间开始"),
    "period_end": ("period_end", "期间结束", "审计期间结束"),
    "currency_unit": ("currency_unit", "金额单位", "单位"),
    "currency": ("currency", "币种"),
    "auditor_name": ("auditor_name", "会计师事务所", "审计机构"),
}
_REPAIR_EVENT_CODES = frozenset(
    {
        "AUDIT_BALANCE_AMOUNT_SHIFT_RECOVERED",
        "AUDIT_BALANCE_SECTION_LABEL_RECOVERED",
        "AUDIT_CANONICAL_RAW_RECOVERED",
        "AUDIT_CROSS_PAGE_TABLE_REPAIRED",
        "AUDIT_FINANCIAL_PERIOD_HEADERS_RECOVERED",
        "AUDIT_NOTE_ADJACENT_CELLS_SPLIT",
        "AUDIT_NOTE_DUPLICATE_DATASET_REMOVED",
        "AUDIT_NOTE_HEADER_ROWS_PROMOTED",
        "AUDIT_NOTE_HEADER_SOURCE_RECOVERED",
        "AUDIT_NOTE_HORIZONTAL_TABLE_MERGED",
        "AUDIT_NOTE_MIXED_TABLE_SPLIT",
        "AUDIT_NOTE_STACKED_HEADERS_RECOVERED",
        "AUDIT_NOTE_TEXT_CONTINUATION_RECOVERED",
        "AUDIT_OWNER_EQUITY_HEADER_ROWS_REMOVED",
        "AUDIT_OWNER_EQUITY_LABEL_ROWS_RECOVERED",
        "AUDIT_OWNER_EQUITY_LOGICAL_TABLE_REJECTED",
        "AUDIT_OWNER_EQUITY_ROTATION_RECOVERED",
        "AUDIT_STATEMENT_CELLS_RECOVERED",
        "AUDIT_STATEMENT_HEADER_ROWS_REMOVED",
        "AUDIT_STATEMENT_RECOVERY_SELECTED",
        "AUDIT_STATEMENT_TEXT_ROWS_RECOVERED",
        "AUDIT_STATEMENT_TEXT_ROWS_SELECTED",
    }
)


def derive_audit_report_projection(parse_result: Any, *, full_text: str = "") -> ProjectionData:
    """Derive an audit-specific projection without mutating the sealed result."""

    generic = derive_generic_projection(parse_result, "audit_report", full_text)
    datasets = {name: copy.deepcopy(rows) for name, rows in generic.datasets.items()}
    warnings = [warning for warning in generic.warnings if str(warning).strip().lower() != "community_generic_fallback"]

    warnings.extend(repair_note_datasets(datasets))
    warnings.extend(resolve_note_table_candidates(datasets, parse_result))
    warnings.extend(merge_horizontal_note_continuations(datasets, parse_result))
    warnings.extend(merge_cross_page_continuations(datasets, parse_result))
    warnings.extend(repair_stacked_note_headers(datasets))
    financial_segments, financial_warnings = project_embedded_financial_statements(parse_result)
    warnings.extend(financial_warnings)
    warnings.extend(recover_owner_equity_label_rows(financial_segments, parse_result))
    financial_pages = embedded_financial_pages(parse_result)
    unresolved_wide_pages = _unresolved_wide_statement_pages(
        datasets,
        financial_segments,
        known_financial_pages=financial_pages,
    )
    financial_pages.update(unresolved_wide_pages)
    for page, width in unresolved_wide_pages.items():
        warnings.extend(
            (
                f"AUDIT_FINANCIAL_STATEMENT_UNRESOLVED:page={page}:kind=owners_equity_changes",
                f"AUDIT_OWNER_EQUITY_UNRESOLVED:page={page}:width={width}:source=sealed_wide_table",
            )
        )
    for name in list(datasets):
        pages = dataset_pages(datasets[name])
        if pages and pages <= financial_pages:
            datasets.pop(name)
    for segment in financial_segments:
        datasets[segment.dataset_id] = copy.deepcopy(segment.records)

    fields, field_details, metadata_warnings = _audit_fields(generic.domain_facts, parse_result, full_text)
    warnings.extend(metadata_warnings)
    if fields.get("currency") and fields.get("currency_unit"):
        warnings = [
            warning for warning in warnings if not str(warning).startswith("precision:generic_currency_unknown:")
        ]
    sections = _audit_sections(parse_result, generic.sections, financial_segments, field_details)
    semantic_labels, naming_warnings = name_note_datasets(datasets, sections, parse_result)
    warnings.extend(naming_warnings)
    warnings.extend(recover_note_text_continuations(datasets, parse_result))
    warnings.extend(synchronize_audit_record_sources(datasets))
    note_schemas, schema_warnings = canonicalize_audit_dataset_columns(datasets)
    warnings.extend(schema_warnings)
    datasets = {name: [normalize_audit_record(record) for record in rows] for name, rows in datasets.items()}
    section_ids, dataset_labels = bind_datasets_to_sections(datasets, sections, parse_result)
    dataset_labels.update(semantic_labels)
    dataset_labels = {name: normalize_audit_label(label) for name, label in dataset_labels.items()}
    semantic = audit_semantic(generic.semantic, datasets, financial_segments, section_ids, dataset_labels)
    page_dimensions = {
        str(int(getattr(page, "page_number", 0) or 1)): {
            "width": int(getattr(page, "width", 0) or 0),
            "height": int(getattr(page, "height", 0) or 0),
        }
        for page in getattr(parse_result, "pages", None) or []
        if getattr(page, "width", None) and getattr(page, "height", None)
    }
    if page_dimensions:
        semantic.setdefault("enhanced_markdown", {})["page_dimensions"] = page_dimensions
    warnings.extend(quality_warnings(parse_result, fields, sections, datasets, financial_segments))
    warnings = _discard_superseded_generic_precision_warnings(warnings, datasets)
    warnings = _discard_superseded_financial_precision_warnings(warnings, financial_segments)
    warnings, repair_events = _partition_repair_events(warnings)
    dataset_blockers = _dataset_verification_blockers(datasets, financial_segments, warnings)
    source_row_ledgers = _dataset_source_row_ledgers(datasets, financial_segments)

    domain_facts = {
        **fields,
        "field_details": field_details,
        "data_dictionary": audit_data_dictionary(financial_segments, note_schemas),
        "dataset_source_row_refs": source_row_ledgers,
        "dataset_verification_blockers": dataset_blockers,
        "summary": {
            "field_count": len(fields),
            "dataset_count": len(datasets),
            "total_rows": sum(len(rows) for rows in datasets.values()),
            "section_count": len(sections),
        },
        "audit_projection": {
            "dataset_count": len(datasets),
            "financial_statement_count": len(financial_segments),
            "section_count": len(sections),
            "source_conserving": True,
            "repair_events": repair_events,
        },
    }
    return ProjectionData(
        projector_id="audit_report",
        document_type="audit_report",
        entity_fields=_public_entity_fields(fields),
        domain_facts=domain_facts,
        semantic=semantic,
        datasets=datasets,
        sections=tuple(sections),
        warnings=tuple(dict.fromkeys(warnings)),
        confidence=0.85 if financial_segments and not blocking_warning(warnings) else 0.55,
        reason="post-seal audit-report projection from sealed source facts",
    )


def _unresolved_wide_statement_pages(
    datasets: dict[str, list[dict[str, Any]]],
    financial_segments: list[ProjectedSegment],
    *,
    known_financial_pages: set[int],
) -> dict[int, int]:
    """Identify transposed main-statement facts hidden behind generic wide-table rows."""

    cash_pages = {
        page
        for segment in financial_segments
        if segment.kind == "cash_flow_statement"
        for page in dataset_pages(segment.records)
    }
    if not cash_pages:
        return {}
    cash_end = max(cash_pages)
    candidates: dict[int, int] = {}
    for name, rows in datasets.items():
        if not name.startswith("table_") or not rows:
            continue
        pages = dataset_pages(rows)
        if len(pages) != 1:
            continue
        page = next(iter(pages))
        if page in known_financial_pages or not cash_end < page <= cash_end + 2:
            continue
        keys = {
            str(key)
            for record in rows
            for key in ((record.get("raw") or {}).keys() if isinstance(record.get("raw"), dict) else [])
        }
        width = len(keys)
        generic_columns = sum(bool(re.fullmatch(r"(?:col|column)_\d+", key)) for key in keys)
        if width >= 8 and len(rows) < width and generic_columns >= max(6, math.ceil(width * 0.6)):
            candidates[page] = max(candidates.get(page, 0), width)
    return candidates


def _audit_fields(
    domain_facts: dict[str, Any],
    parse_result: Any,
    full_text: str,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    source_details = domain_facts.get("field_details") if isinstance(domain_facts.get("field_details"), dict) else {}
    source_text = full_text or _document_text(parse_result)
    fields, details = _copy_public_fields(domain_facts, source_details)
    warnings: list[str] = []

    regulatory, regulatory_detail, regulatory_warnings = _resolve_regulatory_number(
        domain_facts,
        source_details,
        parse_result,
    )
    warnings.extend(regulatory_warnings)
    if regulatory:
        fields["regulatory_report_id"] = regulatory
        details["regulatory_report_id"] = regulatory_detail

    audit_number, audit_details, audit_warnings = _resolve_audit_number(parse_result, source_text)
    warnings.extend(audit_warnings)
    details.update(audit_details)
    if audit_number:
        fields["audit_document_number"] = audit_number

    title_match = _find_text_match(parse_result, _AUDIT_TITLE_RE)
    if title_match is None:
        title_match = _match_from_text(source_text, _AUDIT_TITLE_RE)
    if title_match is not None:
        title, source = title_match
        fields["document_label"] = re.sub(r"\s+", "", title)
        details["document_label"] = source

    if not fields.get("auditor_name"):
        auditor_match = _find_text_match(parse_result, _AUDITOR_NAME_RE)
        if auditor_match is not None:
            fields["auditor_name"], details["auditor_name"] = auditor_match
    if not fields.get("document_date"):
        date_match = _find_text_match(parse_result, _REPORT_DATE_RE)
        if date_match is not None:
            fields["document_date"], details["document_date"] = date_match

    opinion = _audit_opinion(parse_result, source_text)
    if opinion is not None:
        value, source = opinion
        fields["audit_opinion_type"] = value
        details["audit_opinion_type"] = source
    return fields, details, warnings


def _copy_public_fields(
    domain_facts: dict[str, Any],
    source_details: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    fields: dict[str, Any] = {}
    details: dict[str, Any] = {}
    for target, aliases in _PUBLIC_FIELD_ALIASES.items():
        source_key = next((alias for alias in aliases if domain_facts.get(alias) not in (None, "")), "")
        if not source_key:
            continue
        fields[target] = normalize_audit_text(domain_facts[source_key])
        if isinstance(source_details.get(source_key), dict):
            details[target] = copy.deepcopy(source_details[source_key])
    return fields, details


def _resolve_regulatory_number(
    domain_facts: dict[str, Any],
    source_details: dict[str, Any],
    parse_result: Any,
) -> tuple[str, dict[str, Any], list[str]]:
    value = normalize_audit_text(_field_by_alias(domain_facts, "regulatory_report_id", "报告编号"))
    source_key = next(
        (key for key in ("regulatory_report_id", "报告编号") if isinstance(source_details.get(key), dict)),
        "",
    )
    if _valid_regulatory_number(value):
        return value, copy.deepcopy(source_details.get(source_key) or {}), []
    match = next(
        (
            (candidate, source)
            for candidate, source in _find_text_matches(parse_result, _REGULATORY_NUMBER_RE)
            if _valid_regulatory_number(candidate)
        ),
        None,
    )
    if match is not None:
        return match[0], match[1] or _source_for_marker(parse_result, match[0]), []
    warnings = [f"AUDIT_REGULATORY_REPORT_ID_INVALID:value={value}"] if value else []
    return "", {}, warnings


def _resolve_audit_number(
    parse_result: Any,
    source_text: str,
) -> tuple[str, dict[str, Any], list[str]]:
    number_matches = _find_text_matches(parse_result, _AUDIT_NUMBER_RE)
    incomplete_matches = _find_text_matches(parse_result, _INCOMPLETE_AUDIT_NUMBER_RE)
    if not number_matches:
        fallback = _match_from_text(source_text, _AUDIT_NUMBER_RE)
        number_matches = [fallback] if fallback is not None else []
    if not incomplete_matches:
        fallback = _match_from_text(source_text, _INCOMPLETE_AUDIT_NUMBER_RE)
        incomplete_matches = [fallback] if fallback is not None else []
    incomplete = list(dict.fromkeys(re.sub(r"\s+", "", value) for value, _source in incomplete_matches))
    if not number_matches:
        incomplete_warnings = [f"AUDIT_DOCUMENT_NUMBER_INCOMPLETE:values={'|'.join(incomplete)}"] if incomplete else []
        return "", {}, [*incomplete_warnings, "AUDIT_DOCUMENT_NUMBER_MISSING"]

    normalized = [(re.sub(r"\s+", "", number), source) for number, source in number_matches]
    warnings: list[str] = []
    distinct = list(dict.fromkeys(value for value, _source in normalized))
    top_authority = max(_audit_number_authority(source) for _number, source in normalized)
    authoritative = [
        (number, source) for number, source in normalized if _audit_number_authority(source) == top_authority
    ]
    authoritative_values = list(dict.fromkeys(number for number, _source in authoritative))
    if len(authoritative_values) > 1:
        conflicts = list(dict.fromkeys(authoritative_values))
        details = {
            "audit_document_number_candidates": {
                "values": distinct,
                "incomplete_values": incomplete,
                "sources": [copy.deepcopy(source) for _number, source in normalized],
                "resolution": "unresolved_conflict",
            }
        }
        return "", details, [*warnings, f"AUDIT_DOCUMENT_NUMBER_CONFLICT:values={'|'.join(conflicts)}"]

    number, source = authoritative[0]
    alternates = [value for value in distinct if value != number]
    if alternates:
        warnings.append(f"AUDIT_DOCUMENT_NUMBER_CANDIDATE_VARIANCE:selected={number}:alternates={'|'.join(alternates)}")
    detail = {
        **source,
        "resolution": "authoritative_complete_candidate",
        **({"alternate_candidates": alternates} if alternates else {}),
        **({"incomplete_candidates": incomplete} if incomplete else {}),
    }
    candidate_detail = {
        "values": distinct,
        "incomplete_values": incomplete,
        "sources": [copy.deepcopy(candidate_source) for _candidate, candidate_source in normalized],
        "resolution": "selected_authoritative_complete_candidate",
        "selected": number,
    }
    return (
        number,
        {
            "audit_document_number": detail,
            "audit_document_number_candidates": candidate_detail,
        },
        warnings,
    )


def _audit_sections(
    parse_result: Any,
    generic_sections: Iterable[dict[str, Any]],
    financial_segments: list[ProjectedSegment],
    field_details: dict[str, Any],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    order = 0
    for section in generic_sections:
        title = normalize_audit_text(section.get("title"))
        page_start = int(section.get("page_start") or 1)
        if not _valid_section_heading(title) or _bbox_inside_source_table(
            _page_by_number(parse_result, page_start),
            section.get("bbox"),
        ):
            continue
        candidates.append(
            {
                **copy.deepcopy(section),
                "title": title,
                "page_start": page_start,
                "_order": order,
            }
        )
        order += 1
    for page in getattr(parse_result, "pages", None) or []:
        page_number = int(getattr(page, "page_number", 0) or 1)
        for block in getattr(page, "texts", None) or []:
            for line in str(getattr(block, "content", "") or "").splitlines():
                title = normalize_audit_text(line)
                if not _valid_section_heading(title) or _bbox_inside_source_table(
                    page,
                    getattr(block, "bbox", None),
                ):
                    continue
                candidates.append(
                    {
                        "title": title,
                        "page_start": page_number,
                        "bbox": list(getattr(block, "bbox", None) or []),
                        "_order": order,
                    }
                )
                order += 1
    for segment in financial_segments:
        candidates.append(
            {
                "id": f"section_{segment.dataset_id}",
                "title": segment.title,
                "type": "financial_statement",
                "page_start": segment.source_page,
                "_order": order,
            }
        )
        order += 1

    deduplicated: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for candidate in sorted(candidates, key=_section_sort_key):
        key = (candidate["title"].rstrip(":："), candidate["page_start"])
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(candidate)

    metadata_end = max((page for detail in field_details.values() for page in _detail_pages(detail)), default=1)
    sections: list[dict[str, Any]] = [
        {
            "id": "section_audit_metadata",
            "title": "审计报告基本信息",
            "type": "basic_information",
            "page_start": 1,
            "page_end": metadata_end,
        }
    ]
    used_ids = {"section_audit_metadata"}
    for index, candidate in enumerate(deduplicated, start=1):
        section_type = str(candidate.get("type") or _section_type(candidate["title"]))
        preferred_id = str(candidate.get("id") or f"section_audit_{index:03d}")
        section_id = _unique_id(preferred_id, used_ids)
        used_ids.add(section_id)
        sections.append(
            {
                "id": section_id,
                "title": candidate["title"],
                "type": section_type,
                "page_start": candidate["page_start"],
                **({"bbox": candidate["bbox"]} if candidate.get("bbox") else {}),
            }
        )

    appendix_start = _appendix_start_page(parse_result)
    if appendix_start:
        sections.append(
            {
                "id": "section_appendix",
                "title": "附件",
                "type": "appendix",
                "page_start": appendix_start,
            }
        )
    _set_section_page_ranges(sections, page_count=len(getattr(parse_result, "pages", None) or []))
    return sections


def _set_section_page_ranges(sections: list[dict[str, Any]], *, page_count: int) -> None:
    for index, section in enumerate(sections):
        if section["id"] == "section_audit_metadata":
            continue
        next_page = next(
            (
                int(candidate["page_start"])
                for candidate in sections[index + 1 :]
                if int(candidate.get("page_start") or 0) >= int(section["page_start"])
            ),
            page_count or int(section["page_start"]),
        )
        section["page_end"] = max(int(section["page_start"]), next_page - int(next_page > section["page_start"]))


def _audit_opinion(parse_result: Any, full_text: str) -> tuple[str, dict[str, Any]] | None:
    scoped_lines = _opinion_scope(parse_result)
    if scoped_lines:
        text = normalize_audit_text("\n".join(line for line, _block, _page in scoped_lines))
    else:
        front = _audit_front_text(parse_result) or normalize_audit_text(full_text)
        text = _opinion_scope_from_text(front)
    markers = (
        ("无法表示意见", "disclaimer"),
        ("否定意见", "adverse"),
    )
    for marker, value in markers:
        if marker in text:
            return value, _source_for_scoped_marker(scoped_lines, marker) or _source_for_marker(parse_result, marker)
    qualified = re.search(r"(?<!非无)保留意见|除.{0,160}影响外", text, flags=re.S)
    if qualified:
        marker = qualified.group(0)
        return "qualified", _source_for_scoped_marker(scoped_lines, marker) or _source_for_marker(
            parse_result,
            "保留意见",
        )
    if "我们认为" in text and "公允反映" in text:
        return "unmodified", _source_for_scoped_marker(scoped_lines, "我们认为") or _source_for_marker(
            parse_result,
            "我们认为",
        )
    return None


def _opinion_scope(parse_result: Any) -> list[tuple[str, Any, int]]:
    scoped: list[tuple[str, Any, int]] = []
    active = False
    for page in getattr(parse_result, "pages", None) or []:
        if statement_kind(page) is not None:
            break
        page_number = int(getattr(page, "page_number", 0) or 1)
        for block in getattr(page, "texts", None) or []:
            for raw_line in str(getattr(block, "content", "") or "").splitlines():
                line = normalize_audit_text(raw_line)
                if not line:
                    continue
                section_type = _section_type(line) if _HEADING_RE.fullmatch(line) else ""
                if section_type == "audit_opinion":
                    active = True
                    scoped.append((line, block, page_number))
                    continue
                if active and section_type and section_type != "audit_opinion":
                    return scoped
                if active:
                    scoped.append((line, block, page_number))
    return scoped


def _opinion_scope_from_text(text: str) -> str:
    match = re.search(
        r"(?:[一二三四五六七八九十]+[、.．])?审计意见(?P<body>.*?)(?:"
        r"[一二三四五六七八九十]+[、.．](?:形成审计意见的基础|关键审计事项|强调事项|其他信息|管理层))",
        text,
        flags=re.S,
    )
    return normalize_audit_text(match.group("body")) if match else ""


def _source_for_scoped_marker(
    scoped_lines: list[tuple[str, Any, int]],
    marker: str,
) -> dict[str, Any] | None:
    compact_marker = normalize_audit_text(marker)
    for line, block, page in scoped_lines:
        if compact_marker in line or line in compact_marker:
            return _block_source(block, page, 0.96)
    return None


def _audit_front_text(parse_result: Any) -> str:
    lines: list[str] = []
    for page in getattr(parse_result, "pages", None) or []:
        if statement_kind(page) is not None:
            break
        page_number = int(getattr(page, "page_number", 0) or 1)
        if page_number > 8:
            break
        lines.extend(page_lines(page))
    return normalize_audit_text("\n".join(lines))


def _find_text_match(parse_result: Any, pattern: re.Pattern[str]) -> tuple[str, dict[str, Any]] | None:
    matches = _find_text_matches(parse_result, pattern)
    return matches[0] if matches else None


def _find_text_matches(parse_result: Any, pattern: re.Pattern[str]) -> list[tuple[str, dict[str, Any]]]:
    matches: list[tuple[str, dict[str, Any]]] = []
    for page in getattr(parse_result, "pages", None) or []:
        page_number = int(getattr(page, "page_number", 0) or 1)
        for block in getattr(page, "texts", None) or []:
            text = normalize_audit_text(getattr(block, "content", ""))
            for match in pattern.finditer(text):
                matches.append((_match_value(match), _block_source(block, page_number, 0.92)))
    return matches


def _match_from_text(text: str, pattern: re.Pattern[str]) -> tuple[str, dict[str, Any]] | None:
    match = pattern.search(normalize_audit_text(text))
    if not match:
        return None
    return _match_value(match), {"source": "full_text", "confidence": 0.65, "review": "needs_page_evidence"}


def _match_value(match: re.Match[str]) -> str:
    return next((value for value in match.groupdict().values() if value is not None), match.group(0))


def _valid_regulatory_number(value: Any) -> bool:
    return bool(_STRICT_REGULATORY_NUMBER_RE.fullmatch(normalize_audit_text(value).replace(" ", "").upper()))


def _audit_number_authority(source: dict[str, Any]) -> tuple[int, int]:
    page = int(source.get("page") or 999)
    return int(1 < page <= 5), -page


def _audit_number_year(value: str) -> str:
    match = re.search(r"20\d{2}", value)
    return match.group(0) if match else ""


def _source_for_marker(parse_result: Any, marker: str) -> dict[str, Any]:
    for page in getattr(parse_result, "pages", None) or []:
        page_number = int(getattr(page, "page_number", 0) or 1)
        for block in getattr(page, "texts", None) or []:
            if marker in normalize_audit_text(getattr(block, "content", "")):
                return _block_source(block, page_number, 0.88)
    return {"source": "full_text", "confidence": 0.6, "review": "needs_page_evidence"}


def _block_source(block: Any, page: int, confidence: float) -> dict[str, Any]:
    source: dict[str, Any] = {"source": "canonical_text", "page": page, "confidence": confidence}
    if getattr(block, "bbox", None):
        source["bbox"] = list(block.bbox)
    if getattr(block, "evidence_ids", None):
        source["evidence_ids"] = list(block.evidence_ids)
    return source


def _appendix_start_page(parse_result: Any) -> int:
    for page in getattr(parse_result, "pages", None) or []:
        page_number = int(getattr(page, "page_number", 0) or 1)
        text = normalize_audit_text(" ".join(page_lines(page)))
        if page_number > 1 and "营业执照" in text and "统一社会信用代码" in text:
            return page_number
    return 0


def _valid_section_heading(title: str) -> bool:
    match = _HEADING_RE.fullmatch(title)
    if not match:
        return False
    compact = title.replace(" ", "")
    if _BARE_NOTE_REFERENCE_RE.fullmatch(compact):
        return False
    marker = match.group("marker")
    body = normalize_audit_text(match.group("title"))
    if re.fullmatch(r"[（(]\d{1,2}[）)]", marker):
        return False
    if len(body) > 36 or re.search(r"[，,；;。！？!?]", body):
        return False
    return not bool(re.fullmatch(r"[一二三四五六七八九十百\d、.．（）()]+", compact))


def _bbox_inside_source_table(page: Any, bbox: Any) -> bool:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return False
    left = [float(value) for value in bbox[:4]]
    for table in getattr(page, "tables", None) or []:
        table_bbox = getattr(table, "bbox", None)
        if not isinstance(table_bbox, (list, tuple)) or len(table_bbox) < 4:
            continue
        right = [float(value) for value in table_bbox[:4]]
        center_x = (left[0] + left[2]) / 2
        center_y = (left[1] + left[3]) / 2
        if right[0] <= center_x <= right[2] and right[1] <= center_y <= right[3]:
            return True
    return False


def _page_by_number(parse_result: Any, page_number: int) -> Any | None:
    return next(
        (
            page
            for page in getattr(parse_result, "pages", None) or []
            if int(getattr(page, "page_number", 0) or 1) == page_number
        ),
        None,
    )


def _detail_pages(detail: Any) -> set[int]:
    if not isinstance(detail, dict):
        return set()
    pages: set[int] = set()
    page = detail.get("page")
    if page not in (None, ""):
        try:
            page_number = int(page)
        except (TypeError, ValueError):
            page_number = 0
        if page_number > 0:
            pages.add(page_number)
    page_range = detail.get("page_range")
    if isinstance(page_range, (list, tuple)):
        for value in page_range:
            try:
                page_number = int(value)
            except (TypeError, ValueError):
                continue
            if page_number > 0:
                pages.add(page_number)
    return pages


def _section_type(title: str) -> str:
    for marker, section_type in _CORE_SECTION_TYPES:
        if marker in title:
            return section_type
    if "承诺及或有事项" in title:
        return "commitments_and_contingencies"
    if "资产负债表日后事项" in title:
        return "subsequent_events"
    if "其他重要事项" in title:
        return "other_significant_matters"
    return "financial_statement_note"


def _section_sort_key(section: dict[str, Any]) -> tuple[int, float, int]:
    bbox = section.get("bbox") or []
    top = float(bbox[1]) if len(bbox) >= 2 else float("inf")
    return int(section.get("page_start") or 1), top, int(section.get("_order") or 0)


def _public_entity_fields(fields: dict[str, Any]) -> dict[str, Any]:
    return {
        key: fields[key]
        for key in ("subject_name", "subject_id", "audit_document_number", "regulatory_report_id")
        if fields.get(key) not in (None, "")
    }


def _field_by_alias(fields: dict[str, Any], *aliases: str) -> Any:
    return next((fields[name] for name in aliases if fields.get(name) not in (None, "")), "")


def _document_text(parse_result: Any) -> str:
    return "\n".join(line for page in getattr(parse_result, "pages", None) or [] for line in page_lines(page))


def _unique_id(preferred: str, used: set[str]) -> str:
    if preferred not in used:
        return preferred
    suffix = 2
    while f"{preferred}_{suffix}" in used:
        suffix += 1
    return f"{preferred}_{suffix}"


def _dataset_source_row_ledgers(
    datasets: dict[str, list[dict[str, Any]]],
    segments: list[ProjectedSegment],
) -> dict[str, list[dict[str, Any]]]:
    segment_ledgers = {
        segment.dataset_id: [copy.deepcopy(reference) for reference in segment.source_row_refs]
        for segment in segments
        if segment.records
    }
    ledgers: dict[str, list[dict[str, Any]]] = {}
    for dataset_name, records in datasets.items():
        if dataset_name in segment_ledgers:
            ledgers[dataset_name] = segment_ledgers[dataset_name]
            continue
        ledgers[dataset_name] = [
            {
                "page": int(source.get("page") or 0),
                "table_id": str(source.get("table_id") or source.get("physical_table_id") or ""),
                "source_row_index": source.get("source_row_index"),
            }
            for record in records
            if isinstance((source := record.get("source")), dict)
        ]
    return ledgers


def _discard_superseded_financial_precision_warnings(
    warnings: Iterable[str],
    segments: list[ProjectedSegment],
) -> list[str]:
    emitted_rows = {
        (
            str(source.get("table_id") or source.get("physical_table_id") or ""),
            int(source.get("table_row_index", source.get("source_row_index", -1))),
        )
        for segment in segments
        for record in segment.records
        if isinstance((source := record.get("source")), dict)
    }
    statement_tables = {table_id for table_id, _row_index in emitted_rows if table_id}
    retained: list[str] = []
    for warning in warnings:
        match = re.match(
            r"precision:financial_amount_format_invalid:table=([^:]+):row=(\d+):",
            str(warning),
        )
        if match and match.group(1) in statement_tables and (match.group(1), int(match.group(2))) not in emitted_rows:
            continue
        retained.append(str(warning))
    return retained


def _discard_superseded_generic_precision_warnings(
    warnings: Iterable[str],
    datasets: dict[str, list[dict[str, Any]]],
) -> list[str]:
    final_keys = {str(key) for records in datasets.values() for key in record_keys(records)}
    schemas_resolved = not any(re.fullmatch(r"(?:col|column)_\d+", key) for key in final_keys)
    normalization_loss = any(str(warning).startswith("AUDIT_NORMALIZATION_LOSS") for warning in warnings)
    retained: list[str] = []
    for warning in warnings:
        value = str(warning)
        if value.startswith("precision:generic_low_confidence_text_kv:"):
            continue
        if schemas_resolved and value.startswith(
            ("precision:generic_header_repaired:", "precision:generic_header_repaired_ratio:")
        ):
            continue
        if not normalization_loss and value.startswith("precision:generic_normalization_failed:"):
            continue
        retained.append(value)
    return retained


def _partition_repair_events(warnings: Iterable[str]) -> tuple[list[str], list[str]]:
    """Keep successful audit repairs out of public Community warnings."""

    public: list[str] = []
    repairs: list[str] = []
    for warning in warnings:
        value = str(warning)
        code = value.split(":", 1)[0]
        (repairs if code in _REPAIR_EVENT_CODES else public).append(value)
    return list(dict.fromkeys(public)), list(dict.fromkeys(repairs))


def _dataset_verification_blockers(
    datasets: dict[str, list[dict[str, Any]]],
    segments: list[ProjectedSegment],
    warnings: Iterable[str],
) -> dict[str, list[str]]:
    blockers = dataset_blocking_warnings(warnings)
    segment_kinds = {segment.dataset_id: segment.kind for segment in segments}
    result: dict[str, list[str]] = {}
    for dataset_name, records in datasets.items():
        if not records:
            continue
        pages = dataset_pages(records)
        relevant: list[str] = []
        for warning in blockers:
            dataset_match = re.search(r"dataset=([^:]+)", warning)
            kind_match = re.search(r"kind=([^:]+)", warning)
            warning_pages = {int(match.group(1)) for match in re.finditer(r"page=(\d+)", warning)}
            if dataset_match and dataset_match.group(1) != dataset_name:
                continue
            if kind_match and kind_match.group(1) != segment_kinds.get(dataset_name):
                continue
            if warning_pages and pages and not warning_pages.intersection(pages):
                if not kind_match:
                    continue
            if (
                not dataset_match
                and not kind_match
                and not warning_pages
                and not (dataset_name in segment_kinds and warning.startswith("AUDIT_STATEMENT_TOTAL_MISMATCH"))
            ):
                continue
            relevant.append(warning)
        if relevant:
            result[dataset_name] = relevant
    return result


__all__ = ["derive_audit_report_projection", "normalize_audit_text"]
