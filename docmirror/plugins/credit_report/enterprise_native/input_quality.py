# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""F4 validation of ParseResult inputs before enterprise reconstruction."""

from __future__ import annotations

from typing import Any

from docmirror.plugins.credit_report.enterprise_native.quality import EnterpriseQualityFlag


def _page_number(page: Any, fallback: int) -> int:
    for key in ("page_number", "number", "page"):
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
    return bool(getattr(page, "images", None))


def _page_strings(page: Any) -> tuple[str, ...]:
    values = [str(getattr(block, "content", "") or "") for block in (getattr(page, "texts", None) or ())]
    for table in getattr(page, "tables", None) or ():
        metadata = getattr(table, "metadata", None) or {}
        rows = metadata.get("raw_rows") if isinstance(metadata, dict) else None
        for row in rows or getattr(table, "rows", None) or ():
            values.extend(str(value or "") for value in row)
    return tuple(values)


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
        return (
            EnterpriseQualityFlag(
                code="ENTERPRISE_INPUT_NO_PAGES",
                severity="error",
                category="parseresult_input",
                status="bad_input",
                message="ParseResult contains no pages; enterprise fields cannot be reconstructed.",
            ),
        )

    page_numbers = [_page_number(page, index) for index, page in enumerate(pages, start=1)]
    duplicates = sorted({number for number in page_numbers if page_numbers.count(number) > 1})
    if duplicates:
        flags.append(
            EnterpriseQualityFlag(
                code="ENTERPRISE_INPUT_DUPLICATE_PAGE_NUMBERS",
                severity="error",
                category="parseresult_input",
                status="bad_input",
                message="ParseResult contains duplicate positive page numbers.",
                source_pages=tuple(duplicates),
            )
        )
    expected = set(range(min(page_numbers), max(page_numbers) + 1))
    missing = tuple(sorted(expected.difference(page_numbers)))
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
