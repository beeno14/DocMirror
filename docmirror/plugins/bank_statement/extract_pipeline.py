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
)


@dataclass
class BankExtractResult:
    ctx: StyleContext
    detection: StyleDetectionResult
    records: list[dict[str, Any]]
    identity_fields: dict[str, dict]
    style_meta: StyleMeta
    warnings: list[str]


def enrich_identity_fields(
    identity_fields: dict[str, dict],
    full_text: str,
    parse_result: Any = None,
    institution: str | None = None,
) -> dict[str, dict]:
    """Merge header KV identity into registry identity fields (EIP)."""
    fields = dict(identity_fields)
    for field_name, value in extract_identity_from_header(full_text).items():
        if not value:
            continue
        fields[field_name] = {
            "raw_name": field_name,
            "raw_value": value,
            "normalized_value": value,
            "data_type": "string",
            "source": "header.kv",
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
                if value and field_name not in fields:
                    fields[field_name] = {
                        "raw_name": field_name,
                        "raw_value": str(value),
                        "normalized_value": str(value),
                        "data_type": "string",
                        "source": "entities",
                    }
            subject_id = getattr(entities, "subject_id", None)
            if subject_id and "account_number" not in fields:
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
    parsed_record_count = len(records)
    direction_count = sum(
        1 for record in records if (record.get("normalized") or {}).get("direction") in {"income", "expense"}
    )
    records = [record for record in records if is_canonical_row(record.get("normalized") or {})]
    blocked_noncanonical_count = parsed_record_count - len(records)
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
    source_reported_count = count_expected_rows_from_bank_footer(ctx.full_text)
    total_detail = identity_fields.get("total_transactions")
    if isinstance(total_detail, dict):
        total_value = next(
            (
                str(total_detail.get(candidate) or "")
                for candidate in ("normalized_value", "value", "raw_value")
                if total_detail.get(candidate) not in (None, "")
            ),
            "",
        )
        match = re.search(r"\d+", total_value)
        if match:
            source_reported_count = int(match.group())
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
    if physical_expected > 0 and physical_expected != style_meta.canonical_extracted:
        warnings.append(
            "BANK_PHYSICAL_LOGICAL_ROW_MISMATCH:"
            f"physical={physical_expected}:canonical={style_meta.canonical_extracted}"
        )
    if style_meta.expected_primary_rows > 0 and style_meta.canonical_extracted < style_meta.expected_primary_rows:
        warnings.append(
            "BANK_CANONICAL_ROW_COVERAGE_LOW:"
            f"canonical={style_meta.canonical_extracted}:expected={style_meta.expected_primary_rows}"
        )
    if blocked_noncanonical_count > 0:
        warnings.append(
            f"BANK_DATASET_NONCANONICAL_ROWS_BLOCKED:blocked={blocked_noncanonical_count}:parsed={parsed_record_count}"
        )
    if parsed_record_count > 0 and direction_count < parsed_record_count:
        warnings.append(f"BANK_DIRECTION_COVERAGE_LOW:directional={direction_count}:parsed={parsed_record_count}")
    sourced_count = sum(1 for record in records if _has_single_page_source(record))
    if records and sourced_count < len(records):
        warnings.append(f"BANK_SOURCE_PAGE_COVERAGE_LOW:sourced={sourced_count}:canonical={len(records)}")
        style_meta.extract_status = "degraded"
    invariant_failures = audit_bank_statement_invariants(records, ctx.full_text)
    if invariant_failures:
        if any(warning.startswith("bank_invariant_failed:") for warning in invariant_failures):
            style_meta.extract_status = "degraded"
        warnings.extend(invariant_failures)
    return BankExtractResult(
        ctx=ctx,
        detection=detection,
        records=records,
        identity_fields=identity_fields,
        style_meta=style_meta,
        warnings=warnings,
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
