# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""F4 validation of ParseResult inputs before enterprise reconstruction."""

from __future__ import annotations

from typing import Any

from docmirror.plugins.credit_report.enterprise_native.quality import EnterpriseQualityFlag


def _page_number(page: Any, fallback: int) -> int:
    for key in ("source_page_number", "page_number", "number", "page"):
        try:
            value = int(getattr(page, key, 0) or 0)
        except (TypeError, ValueError):
            value = 0
        if value > 0:
            return value
    return fallback


def _has_page_content(page: Any) -> bool:
    if any(str(getattr(block, "content", "") or "").strip() for block in (getattr(page, "texts", None) or ())):
        return True
    for table in getattr(page, "tables", None) or ():
        metadata = getattr(table, "metadata", None) or {}
        rows = metadata.get("raw_rows") if isinstance(metadata, dict) else None
        if rows or getattr(table, "rows", None) or getattr(table, "headers", None):
            return True
        if str(getattr(table, "caption", "") or "").strip():
            return True
    if any(
        str(getattr(pair, "key", "") or "").strip()
        or str(getattr(pair, "value", "") or "").strip()
        for pair in (getattr(page, "key_values", None) or ())
    ):
        return True
    return bool(getattr(page, "images", None))


def _page_strings(page: Any) -> tuple[str, ...]:
    values = [str(getattr(block, "content", "") or "") for block in (getattr(page, "texts", None) or ())]
    for table in getattr(page, "tables", None) or ():
        caption = str(getattr(table, "caption", "") or "")
        if caption:
            values.append(caption)
        metadata = getattr(table, "metadata", None) or {}
        rows = metadata.get("raw_rows") if isinstance(metadata, dict) else None
        for row in rows or getattr(table, "rows", None) or ():
            cells = getattr(row, "cells", None)
            if cells is not None:
                values.extend(str(getattr(value, "text", value) or "") for value in cells)
            else:
                values.extend(str(value or "") for value in row)
    for pair in getattr(page, "key_values", None) or ():
        values.append(str(getattr(pair, "key", "") or ""))
        values.append(str(getattr(pair, "value", "") or ""))
    return tuple(values)


def _logical_table_strings(parse_result: Any) -> tuple[str, ...]:
    values: list[str] = []
    for table in getattr(parse_result, "logical_tables", None) or ():
        values.extend(str(value or "") for value in (getattr(table, "headers", None) or ()))
        for row in getattr(table, "rows", None) or ():
            values.extend(
                str(getattr(cell, "text", cell) or "")
                for cell in (getattr(row, "cells", None) or ())
            )
    return tuple(values)


def _flow_strings(parse_result: Any) -> tuple[str, ...]:
    graph = getattr(parse_result, "document_flow", None)
    return tuple(
        str(getattr(node, "text", "") or "")
        for node in (getattr(graph, "nodes", None) or ())
        if str(getattr(node, "text", "") or "")
    )


def _has_alternate_content(parse_result: Any) -> bool:
    return bool(_logical_table_strings(parse_result) or _flow_strings(parse_result))


def _suspicious_glyphs(value: str) -> bool:
    return any(
        character == "\ufffd" or "\ue000" <= character <= "\uf8ff" or "\uac00" <= character <= "\ud7af"
        for character in value
    )


def assess_enterprise_parse_result(parse_result: Any) -> tuple[EnterpriseQualityFlag, ...]:
    """Return structural input findings; extraction may proceed for warnings."""
    flags: list[EnterpriseQualityFlag] = []
    pages = list(getattr(parse_result, "pages", None) or ())
    if not pages:
        if not _has_alternate_content(parse_result):
            return (
                EnterpriseQualityFlag(
                    code="ENTERPRISE_INPUT_NO_PAGES",
                    severity="error",
                    category="parseresult_input",
                    status="bad_input",
                    message="ParseResult contains no business-bearing source view; enterprise fields cannot be reconstructed.",
                ),
            )
        flags.append(
            EnterpriseQualityFlag(
                code="ENTERPRISE_INPUT_NO_PHYSICAL_PAGES",
                severity="warning",
                category="parseresult_input",
                status="alternate_source_views_available",
                message="Physical pages are absent; reconstruction is using logical tables and/or document flow.",
            )
        )

    page_numbers = [_page_number(page, index) for index, page in enumerate(pages, start=1)]
    duplicates = sorted({number for number in page_numbers if page_numbers.count(number) > 1})
    if duplicates:
        flags.append(
            EnterpriseQualityFlag(
                code="ENTERPRISE_INPUT_DUPLICATE_PAGE_NUMBERS",
                severity="warning",
                category="parseresult_input",
                status="logical_page_instances_reconstructed",
                message="Multiple logical page containers reference the same source page; unique page-instance identifiers were assigned.",
                source_pages=tuple(duplicates),
                details={
                    "logical_page_instances": [
                        index
                        for index, number in enumerate(page_numbers, start=1)
                        if number in duplicates
                    ]
                },
            )
        )
    expected = set(range(min(page_numbers), max(page_numbers) + 1)) if page_numbers else set()
    missing = tuple(sorted(expected.difference(set(page_numbers))))
    if missing:
        flags.append(
            EnterpriseQualityFlag(
                code="ENTERPRISE_INPUT_PAGE_GAPS",
                severity="warning",
                category="parseresult_input",
                status="possibly_incomplete",
                message="ParseResult page numbering has gaps; cross-page records may be incomplete.",
                source_pages=missing,
            )
        )
    empty_pages = tuple(number for number, page in zip(page_numbers, pages, strict=True) if not _has_page_content(page))
    if empty_pages:
        flags.append(
            EnterpriseQualityFlag(
                code="ENTERPRISE_INPUT_EMPTY_PAGES",
                severity="warning",
                category="parseresult_input",
                status="possibly_incomplete",
                message="One or more ParseResult pages contain no usable text, table, or image content.",
                source_pages=empty_pages,
            )
        )
    suspicious_pages = tuple(
        number
        for number, page in zip(page_numbers, pages, strict=True)
        if any(_suspicious_glyphs(value) for value in _page_strings(page))
    )
    if suspicious_pages:
        flags.append(
            EnterpriseQualityFlag(
                code="ENTERPRISE_INPUT_SUSPICIOUS_GLYPHS",
                severity="warning",
                category="parseresult_input",
                status="possibly_misdecoded",
                message="ParseResult contains replacement, private-use, or unexpected Hangul glyphs; affected fields may need targeted re-extraction.",
                source_pages=suspicious_pages,
            )
        )

    alternate_suspicious = any(
        _suspicious_glyphs(value)
        for value in (*_logical_table_strings(parse_result), *_flow_strings(parse_result))
    )
    if alternate_suspicious and not suspicious_pages:
        flags.append(
            EnterpriseQualityFlag(
                code="ENTERPRISE_INPUT_SUSPICIOUS_GLYPHS",
                severity="warning",
                category="parseresult_input",
                status="possibly_misdecoded",
                message="Logical-table or document-flow source content contains suspicious glyphs.",
            )
        )

    rejected_logical = tuple(
        str(getattr(table, "logical_id", "") or getattr(table, "table_id", "") or index)
        for index, table in enumerate(getattr(parse_result, "logical_tables", None) or (), start=1)
        if not bool(getattr(table, "quality_passed", True))
    )
    if rejected_logical:
        flags.append(
            EnterpriseQualityFlag(
                code="ENTERPRISE_INPUT_LOGICAL_TABLES_REJECTED",
                severity="warning",
                category="parseresult_input",
                status="physical_fallback_required",
                message="One or more composed logical tables failed ParseResult quality gates; physical evidence remains available.",
                details={"logical_table_ids": list(rejected_logical)},
            )
        )

    parser_info = getattr(parse_result, "parser_info", None)
    parser_warnings = tuple(str(value) for value in (getattr(parser_info, "warnings", None) or ()) if str(value))
    if parser_warnings:
        flags.append(
            EnterpriseQualityFlag(
                code="ENTERPRISE_INPUT_PARSER_WARNINGS",
                severity="warning",
                category="parseresult_input",
                status="possibly_incomplete",
                message="The source parser reported warnings before enterprise reconstruction.",
                details={"parser_warnings": list(parser_warnings)},
            )
        )
    parser_errors = tuple(str(value) for value in (getattr(parse_result, "errors", None) or ()) if str(value))
    if parser_errors or getattr(parse_result, "success", True) is False:
        flags.append(
            EnterpriseQualityFlag(
                code="ENTERPRISE_INPUT_PARSER_ERRORS",
                severity="error",
                category="parseresult_input",
                status="bad_input",
                message="The source parser reported failure or errors before enterprise reconstruction.",
                details={"parser_errors": list(parser_errors)},
            )
        )
    return tuple(flags)


__all__ = ["assess_enterprise_parse_result"]
