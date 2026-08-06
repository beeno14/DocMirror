# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Shared bank statement extract pipeline for Community / Enterprise / Finance.

Single SSOT for StyleContext → StyleDetector → ParserRegistry → identity enrichment
→ style metadata → LTRO audit warnings.

Pipeline role: called by post-seal Community, Enterprise, and Finance
projectors against their own sealed read views.

Key exports: ``BankExtractResult``, ``run_bank_statement_extract``,
``enrich_identity_fields``, ``collect_extract_warnings``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from docmirror.plugins.bank_statement.blo import BankLedgerOrchestrator
from docmirror.plugins.bank_statement.canonical import StyleMeta, build_style_meta
from docmirror.plugins.bank_statement.canonical_quality import (
    audit_row_accounting,
    is_canonical_row,
    physical_transaction_row_estimate,
)
from docmirror.plugins.bank_statement.context import StyleContext, build_style_context
from docmirror.plugins.bank_statement.institution_authority import (
    extract_identity_from_header,
    resolve_institution_from_context,
)
from docmirror.plugins.bank_statement.style_detector import BankStyleDetector, StyleDetectionResult
from docmirror.plugins.bank_statement.style_registry import BankStyleParserRegistry
from docmirror.plugins.bank_statement.wide_table_recovery import (
    audit_bank_statement_invariants,
    count_expected_rows_from_bank_footer,
    page_texts_from_parse_result,
)


@dataclass
class BankExtractResult:
    ctx: StyleContext
    detection: StyleDetectionResult
    records: list[dict[str, Any]]
    identity_fields: dict[str, dict]
    style_meta: StyleMeta
    warnings: list[str]
    parsed_rows: int = 0
    canonical_rows: int = 0
    emitted_rows: int = 0
    candidate_diagnostics: list[dict[str, Any]] | None = None


def enrich_identity_fields(
    identity_fields: dict[str, dict],
    full_text: str,
    parse_result: Any = None,
    institution: str | None = None,
) -> dict[str, dict]:
    """Merge header KV identity into registry identity fields (EIP)."""
    fields = dict(identity_fields)
    header_identity = extract_identity_from_header(full_text)
    for field_name, value in header_identity.items():
        if not value:
            continue
        fields[field_name] = {
            "raw_name": field_name,
            "raw_value": value,
            "normalized_value": value,
            "data_type": "string",
            "source": "header.kv",
        }
    explicit_page_period = _explicit_query_period_from_pages(parse_result)
    if explicit_page_period is not None:
        period_value, source_refs = explicit_page_period
        fields["query_period"] = {
            "raw_name": "起始日期/截止日期",
            "raw_value": period_value,
            "normalized_value": period_value,
            "data_type": "string",
            "source": "page_headers",
            "source_refs": source_refs,
        }
    if parse_result is not None:
        entities = getattr(parse_result, "entities", None)
        metadata = getattr(entities, "metadata", None) if entities is not None else None
        if isinstance(metadata, dict):
            for field_name in (
                "account_holder",
                "account_number",
                "bank_name",
                "query_period",
                "currency",
            ):
                value = metadata.get(field_name)
                if field_name == "account_number" and full_text.strip() and not header_identity.get("account_number"):
                    continue
                if value and field_name not in fields:
                    fields[field_name] = {
                        "raw_name": field_name,
                        "raw_value": str(value),
                        "normalized_value": str(value),
                        "data_type": "string",
                        "source": "metadata",
                    }
        if entities is not None:
            for field_name in ("account_holder", "account_number", "bank_name"):
                value = getattr(entities, field_name, None)
                if field_name == "account_number" and full_text.strip() and not header_identity.get("account_number"):
                    continue
                if value and field_name not in fields:
                    fields[field_name] = {
                        "raw_name": field_name,
                        "raw_value": str(value),
                        "normalized_value": str(value),
                        "data_type": "string",
                        "source": "entities",
                    }
            subject_id = getattr(entities, "subject_id", None)
            if (
                subject_id
                and "account_number" not in fields
                and (not full_text.strip() or bool(header_identity.get("account_number")))
            ):
                fields["account_number"] = {
                    "raw_name": "subject_id",
                    "raw_value": str(subject_id),
                    "normalized_value": str(subject_id),
                    "data_type": "string",
                    "source": "entities.subject_id",
                }
        if "currency" not in fields:
            currency_detail = _currency_from_source_table(parse_result)
            if currency_detail:
                fields["currency"] = currency_detail
        if institution and "bank_name" not in fields:
            fields["bank_name"] = {
                "raw_name": "bank_name",
                "raw_value": institution,
                "normalized_value": institution,
                "data_type": "string",
                "source": "institution_argument",
            }
        if "bank_name" not in fields:
            institution, authority = resolve_institution_from_context(parse_result, full_text)
            if institution:
                fields["bank_name"] = {
                    "raw_name": "bank_name",
                    "raw_value": institution,
                    "normalized_value": institution,
                    "data_type": "string",
                    "source": authority or "institution_authority",
                }
    return fields


def _explicit_query_period_from_pages(parse_result: Any) -> tuple[str, list[dict[str, Any]]] | None:
    """Aggregate explicit issuer periods from every page header."""
    if parse_result is None:
        return None
    periods: list[tuple[str, str, int]] = []
    for page_number, page_text in page_texts_from_parse_result(parse_result):
        start_match = re.search(r"起始日期\s*[:：]\s*(20\d{6}|20\d{2}[-/]\d{2}[-/]\d{2})", page_text)
        end_match = re.search(r"(?:截止日期|终止日期)\s*[:：]\s*(20\d{6}|20\d{2}[-/]\d{2}[-/]\d{2})", page_text)
        if start_match is None or end_match is None:
            continue
        periods.append(
            (
                _normalize_period_date(start_match.group(1)),
                _normalize_period_date(end_match.group(1)),
                int(page_number),
            )
        )
    if not periods:
        return None
    return (
        f"{min(period[0] for period in periods)} ~ {max(period[1] for period in periods)}",
        [
            {
                "source": "page_header_text",
                "source_page": period[2],
                "page_range": [period[2], period[2]],
            }
            for period in periods
        ],
    )


def _normalize_period_date(value: str) -> str:
    text = str(value or "").strip().replace("/", "-")
    if re.fullmatch(r"20\d{6}", text):
        return f"{text[:4]}-{text[4:6]}-{text[6:8]}"
    return text


def _currency_from_source_table(parse_result: Any) -> dict[str, Any]:
    """Recover currency only when a source table explicitly provides it."""
    aliases = {"CNY": "CNY", "RMB": "CNY", "人民币": "CNY"}
    for page in getattr(parse_result, "pages", []) or []:
        page_number = int(getattr(page, "source_page_number", 0) or getattr(page, "page_number", 0) or 0)
        for table in getattr(page, "tables", []) or []:
            from docmirror.plugins.bank_statement.header_resolve import normalize_header_cell

            headers = [str(header or "").strip() for header in getattr(table, "headers", []) or []]
            currency_col = next(
                (
                    index
                    for index, header in enumerate(headers)
                    if any(marker in normalize_header_cell(header).lower() for marker in ("币种", "币别", "currency"))
                ),
                None,
            )
            if currency_col is None:
                continue
            for row_index, row in enumerate(getattr(table, "rows", []) or []):
                cells = getattr(row, "cells", []) or []
                if currency_col >= len(cells):
                    continue
                cell = cells[currency_col]
                raw_value = str(getattr(cell, "text", "") or "").strip()
                normalized_raw_value = normalize_header_cell(raw_value)
                normalized_value = aliases.get(normalized_raw_value.upper(), aliases.get(normalized_raw_value))
                if not normalized_value:
                    continue
                source_refs = list(getattr(cell, "source_cell_refs", []) or [])
                if not source_refs:
                    source_refs = [
                        {
                            "source": "canonical_physical_table",
                            "page": page_number,
                            "table_id": str(getattr(table, "table_id", "") or ""),
                            "row": row_index,
                            "col": currency_col,
                        }
                    ]
                return {
                    "raw_name": headers[currency_col],
                    "raw_value": raw_value,
                    "normalized_value": normalized_value,
                    "data_type": "string",
                    "source": "source_table.currency",
                    "source_refs": source_refs,
                    "evidence_ids": list(getattr(cell, "evidence_ids", []) or []),
                }
    return {}


def collect_extract_warnings(ctx: StyleContext, style_meta: StyleMeta) -> list[str]:
    """LTRO / coverage warnings shared across editions."""
    from docmirror.evidence.spe_consumer import read_structure_spe, spe_ltro_warnings

    warnings: list[str] = []
    if ctx.reconstruction and ctx.reconstruction.pipe_parse_failed:
        warnings.append("pipe_parse_failed:no_silent_ocr_fallback")
    expected = style_meta.expected_primary_rows
    extracted = style_meta.extracted_rows
    if expected > 0 and extracted / expected < 0.8:
        warnings.append("low_coverage:bank_ledger")
    if style_meta.extract_status == "degraded":
        warnings.append("cqf_degraded:canonical_quality")
    elif style_meta.extract_status == "low_coverage":
        warnings.append("cqf_low_coverage:canonical_quality")
    if ctx.parse_result is not None:
        spe = read_structure_spe(ctx.parse_result)
        source = style_meta.reconstruction_source or (ctx.reconstruction.source if ctx.reconstruction else "")
        warnings.extend(spe_ltro_warnings(spe, source))
    return warnings


def _apply_source_reported_transaction_count(
    identity_fields: dict[str, dict],
    source_reported_count: int,
) -> None:
    """Keep the identity aggregate aligned with the independently counted source rows."""
    if source_reported_count <= 0:
        return
    identity_fields["total_transactions"] = {
        "raw_name": "page_footer_transaction_count",
        "raw_value": str(source_reported_count),
        "normalized_value": str(source_reported_count),
        "data_type": "integer",
        "source": "page_footer.sum",
    }


def _physical_logical_row_mismatch_warning(physical_expected: int, style_meta: StyleMeta) -> str:
    """Return a row mismatch warning unless a complete recovery supersedes a sparse table."""
    if physical_expected <= 0 or physical_expected == style_meta.canonical_extracted:
        return ""
    recovery_supersedes_sparse_physical = (
        style_meta.reconstruction_source
        in {
            "canonical_evidence_table",
            "positioned_record_block",
            "native_wide_table",
            "ocr_implicit_table",
        }
        and style_meta.canonical_extracted > physical_expected
        and style_meta.canonical_extracted == style_meta.expected_primary_rows
    )
    if recovery_supersedes_sparse_physical:
        return ""
    return f"BANK_PHYSICAL_LOGICAL_ROW_MISMATCH:physical={physical_expected}:canonical={style_meta.canonical_extracted}"


def run_bank_statement_extract(
    parse_result: Any,
    full_text: str,
    plugin: Any,
) -> BankExtractResult:
    """Run the canonical bank-statement extract pipeline."""
    ctx = build_style_context(parse_result, full_text)
    detection = BankStyleDetector().detect(ctx)
    registry = BankStyleParserRegistry()
    records, identity_fields, blo_meta = BankLedgerOrchestrator(registry).run(
        detection,
        ctx,
        plugin,
    )
    parsed_rows = len(records)
    direction_count = sum(
        1 for record in records if (record.get("normalized") or {}).get("direction") in {"income", "expense"}
    )
    records = [record for record in records if is_canonical_row(record.get("normalized") or {})]
    canonical_rows = len(records)
    emitted_rows = canonical_rows
    blocked_noncanonical_count = parsed_rows - canonical_rows
    identity_fields = enrich_identity_fields(identity_fields, ctx.full_text, parse_result)
    try:
        evidence_identity = plugin._recover_identity_from_evidence(parse_result)
    except Exception:
        evidence_identity = {}
    if evidence_identity:
        for field_name, detail in evidence_identity.items():
            current = identity_fields.get(field_name)
            current_source = str(current.get("source") or "") if isinstance(current, dict) else ""
            evidence_preferred = field_name in {
                "account_holder",
                "account_number",
                "query_period",
                "currency",
                "total_transactions",
            }
            if (
                field_name not in identity_fields
                or field_name == "bank_name"
                or (evidence_preferred and current_source in {"header.kv", "bank_statement.default"})
            ):
                identity_fields[field_name] = detail
    page_texts = page_texts_from_parse_result(parse_result)
    source_reported_count = count_expected_rows_from_bank_footer(
        ctx.full_text,
        page_texts=page_texts,
    )
    _apply_source_reported_transaction_count(identity_fields, source_reported_count)
    style_meta = build_style_meta(
        detection,
        reconstruction=ctx.reconstruction,
        record_count=len(records),
        parse_result=parse_result,
        records=records,
        blo_meta=blo_meta,
        source_reported_count=source_reported_count,
    )
    warnings = collect_extract_warnings(ctx, style_meta)
    physical_expected = physical_transaction_row_estimate(parse_result)
    if mismatch_warning := _physical_logical_row_mismatch_warning(physical_expected, style_meta):
        warnings.append(mismatch_warning)
    if style_meta.expected_primary_rows > 0 and style_meta.canonical_extracted < style_meta.expected_primary_rows:
        warnings.append(
            "BANK_CANONICAL_ROW_COVERAGE_LOW:"
            f"canonical={style_meta.canonical_extracted}:expected={style_meta.expected_primary_rows}"
        )
    if blocked_noncanonical_count > 0:
        warnings.append(
            f"BANK_DATASET_NONCANONICAL_ROWS_BLOCKED:blocked={blocked_noncanonical_count}:parsed={parsed_rows}"
        )
        if style_meta.extract_status == "success":
            style_meta.extract_status = "degraded"
    if parsed_rows > 0 and direction_count < parsed_rows:
        warnings.append(f"BANK_DIRECTION_COVERAGE_LOW:directional={direction_count}:parsed={parsed_rows}")
        style_meta.extract_status = "degraded"
    if physical_expected > 0 and style_meta.canonical_extracted < physical_expected:
        style_meta.extract_status = "degraded"
    sourced_count = sum(1 for record in records if _has_single_page_source(record))
    if records and sourced_count < len(records):
        warnings.append(f"BANK_SOURCE_PAGE_COVERAGE_LOW:sourced={sourced_count}:canonical={len(records)}")
        style_meta.extract_status = "degraded"
    invariant_failures = audit_bank_statement_invariants(
        records,
        ctx.full_text,
        page_texts=page_texts,
    )
    if invariant_failures:
        if any(warning.startswith("bank_invariant_failed:") for warning in invariant_failures):
            style_meta.extract_status = "degraded"
        warnings.extend(invariant_failures)
    accounting_warnings = audit_row_accounting(
        parsed_rows=parsed_rows,
        canonical_rows=canonical_rows,
        emitted_rows=emitted_rows,
    )
    if accounting_warnings:
        style_meta.extract_status = "degraded"
        warnings.extend(accounting_warnings)
    if parsed_rows > 0 and not records:
        style_meta.extract_status = "degraded"
        warnings.append("BANK_CANONICAL_EMPTY_AFTER_FILTER:parsed_records_present")
    return BankExtractResult(
        ctx=ctx,
        detection=detection,
        records=records,
        identity_fields=identity_fields,
        style_meta=style_meta,
        warnings=warnings,
        parsed_rows=parsed_rows,
        canonical_rows=canonical_rows,
        emitted_rows=emitted_rows,
        candidate_diagnostics=list(getattr(blo_meta, "candidate_diagnostics", []) or []),
    )


def _has_single_page_source(record: dict[str, Any]) -> bool:
    source = record.get("source") if isinstance(record.get("source"), dict) else {}
    source_page = source.get("source_page")
    page_range = source.get("page_range")
    single_page = (
        source_page not in (None, "", 0)
        and isinstance(page_range, (list, tuple))
        and len(page_range) == 2
        and page_range[0] == page_range[1] == source_page
    )
    stitched_source = (
        source_page not in (None, "", 0)
        and isinstance(page_range, (list, tuple))
        and len(page_range) == 2
        and page_range[0] == source_page
        and bool(source.get("source_refs"))
    )
    return single_page or stitched_source
