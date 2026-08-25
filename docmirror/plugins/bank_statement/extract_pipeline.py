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
from dataclasses import dataclass, replace
from typing import Any

from docmirror.plugins.bank_statement.blo import BankLedgerOrchestrator
from docmirror.plugins.bank_statement.canonical import StyleMeta, build_style_meta
from docmirror.plugins.bank_statement.canonical_quality import (
    audit_row_accounting,
    is_canonical_row,
    physical_transaction_row_estimate,
)
from docmirror.plugins.bank_statement.context import (
    StyleContext,
    build_digital_style_context,
    build_scanned_style_context,
)
from docmirror.plugins.bank_statement.extraction_dispatch import (
    BankExtractionRoute,
    resolve_bank_extraction_route,
)
from docmirror.plugins.bank_statement.institution_authority import (
    extract_header_institution_fields,
    extract_identity_from_header,
)
from docmirror.plugins.bank_statement.statement_context import page_texts_with_business_headers
from docmirror.plugins.bank_statement.style_detector import BankStyleDetector, StyleDetectionResult
from docmirror.plugins.bank_statement.style_registry import BankStyleParserRegistry
from docmirror.plugins.bank_statement.wide_table_recovery import (
    audit_bank_statement_invariants,
    page_texts_from_parse_result,
    resolve_row_count_evidence,
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
    extraction_route: BankExtractionRoute = BankExtractionRoute.DIGITAL


_INFERRED_ISSUER_SOURCES = {
    "domain_specific.institution",
    "entities",
    "entities.organization",
    "filename.token",
    "institution_argument",
    "institution_authority",
    "institution_keywords.header",
    "layout_profile.variant",
    "metadata",
    "plugin.institutions",
}
_DIRECT_ISSUER_SOURCES = {
    "canonical_evidence_atoms",
    "header.kv",
    "page.table_kv",
    "page_headers",
    "parse_result_ocr_text",
}
_ISSUER_SOURCE_LABELS = {
    "bankname",
    "issuerbank",
    "issuermark",
    "issuertitleband",
    "statementtitleissuer",
    "银行名称",
}


def _compact_identity_label(value: Any) -> str:
    return re.sub(r"[\s:：/／_\-()（）]+", "", str(value or "")).casefold()


def is_source_bound_issuer_detail(detail: Any) -> bool:
    """Admit an issuer only from an explicit source label with provenance."""

    if not isinstance(detail, dict):
        return False
    value = next(
        (
            str(detail.get(candidate) or "").strip()
            for candidate in ("raw_value", "value", "normalized_value")
            if detail.get(candidate) not in (None, "")
        ),
        "",
    )
    if not value or not any(marker in value.casefold() for marker in ("银行", "bank", "信用社", "信用联社")):
        return False
    if _compact_identity_label(detail.get("raw_name")) not in _ISSUER_SOURCE_LABELS:
        return False

    source = str(detail.get("source") or "").strip().casefold()
    if source in _INFERRED_ISSUER_SOURCES:
        return False
    refs = [ref for ref in (detail.get("source_refs") or []) if isinstance(ref, dict)]
    direct_ref = any(
        (
            (ref_source := str(ref.get("source") or "").strip().casefold())
            not in _INFERRED_ISSUER_SOURCES
            and (
                ref_source in _DIRECT_ISSUER_SOURCES
                or ref.get("bbox")
                or ref.get("page")
                or ref.get("source_page")
                or ref.get("page_id")
            )
        )
        for ref in refs
    )
    return source in _DIRECT_ISSUER_SOURCES or source.startswith("statement_header") or direct_ref


def _drop_unbound_issuer(fields: dict[str, dict]) -> None:
    if "bank_name" in fields and not is_source_bound_issuer_detail(fields.get("bank_name")):
        fields.pop("bank_name", None)


def enrich_identity_fields(
    identity_fields: dict[str, dict],
    full_text: str,
    parse_result: Any = None,
    institution: str | None = None,
) -> dict[str, dict]:
    """Merge header KV identity into registry identity fields (EIP)."""
    fields = dict(identity_fields)
    header_identity = extract_identity_from_header(full_text)
    header_institution_fields = extract_header_institution_fields(full_text)
    for field_name, value in header_identity.items():
        if not value:
            continue
        raw_name = header_institution_fields.get(field_name, (field_name, value))[0]
        fields[field_name] = {
            "raw_name": raw_name,
            "raw_value": value,
            "normalized_value": value,
            "data_type": "string",
            "source": "header.kv",
            "source_refs": [{"source": "full_text", "scope": "header"}],
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
            for field_name in ("account_holder", "account_number"):
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
    # Institution hints (including ``institution``) are routing metadata.  They
    # never become issuer business data without an explicit source label.
    _drop_unbound_issuer(fields)
    return fields


def _explicit_query_period_from_pages(parse_result: Any) -> tuple[str, list[dict[str, Any]]] | None:
    """Aggregate explicit issuer periods from every page header."""
    if parse_result is None:
        return None
    periods: list[tuple[str, str, int]] = []
    for page_number, page_text in page_texts_with_business_headers(
        parse_result,
        page_texts_from_parse_result(parse_result),
    ):
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


_SPE_FULL_TABLE_PIPE_FALLBACK_WARNING = "spe:mirror_table_extraction_full_used_ltro_fallback"


def _spe_explicitly_proves_no_table_candidates(structure_spe: dict[str, Any] | None) -> bool:
    """Return true only when SPE explicitly proves that no table route existed.

    ``table_extraction=full`` describes the requested extraction policy, not proof that
    Mirror actually produced a usable table.  Keep the ordinary fallback warning unless
    the structure record is complete enough to establish the stronger, layout-neutral
    fact that table extraction was inapplicable and every candidate counter was zero.
    """
    if not isinstance(structure_spe, dict):
        return False
    gate = structure_spe.get("table_reconstruction_gate")
    if not isinstance(gate, dict):
        return False
    return (
        structure_spe.get("table_extraction_skipped_reason") == "no_tabular_signal"
        and structure_spe.get("physical_table_count") == 0
        and structure_spe.get("native_table_candidate_count") == 0
        and "logical_table_count" in structure_spe
        and structure_spe.get("logical_table_count") in {None, 0}
        and gate.get("applicable") is False
        and gate.get("candidate_count") == 0
        and gate.get("physical_table_count") == 0
    )


def _bank_spe_ltro_warnings(
    structure_spe: dict[str, Any] | None,
    reconstruction_source: str,
) -> list[str]:
    """Apply bank-local interpretation without weakening core SPE diagnostics."""
    from docmirror.evidence.spe_consumer import spe_ltro_warnings

    warnings = spe_ltro_warnings(structure_spe, reconstruction_source)
    if _spe_explicitly_proves_no_table_candidates(structure_spe):
        warnings = [warning for warning in warnings if warning != _SPE_FULL_TABLE_PIPE_FALLBACK_WARNING]
    return warnings


def collect_extract_warnings(ctx: StyleContext, style_meta: StyleMeta) -> list[str]:
    """LTRO / coverage warnings shared across editions."""
    from docmirror.evidence.spe_consumer import read_structure_spe

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
        warnings.extend(_bank_spe_ltro_warnings(spe, source))
    return warnings


ISSUER_ROW_COUNT_EVIDENCE_SOURCES = frozenset(
    {
        "split_footer",
        "header_total",
        "statement_header_totals",
        "cumulative_footer_total",
        "page_footer",
    }
)
_ROW_PLANE_COUNT_SOURCES = frozenset(
    {
        "complete_page_local_sequences",
        "ccb_primary_source_sequence",
        "cmb_primary_source_rows",
        "native_page_datetime_census",
        "native_page_signed_ledger_census",
        "ocr_page_ordinal_census",
        "page_transaction_anchors",
        "physical_rows",
        "positioned_date_anchors",
        "positioned_record_blocks",
    }
)


def is_authoritative_issuer_row_count(evidence: Any) -> bool:
    """Return whether ``evidence`` is an exact, issuer-owned document count."""
    if evidence is None or not hasattr(evidence, "source") or not hasattr(evidence, "confidence"):
        return False
    try:
        count = int(getattr(evidence, "count", 0) or 0)
        confidence = float(getattr(evidence, "confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        return False
    source = str(getattr(evidence, "source", "") or "")
    return count > 0 and confidence >= 0.85 and source in ISSUER_ROW_COUNT_EVIDENCE_SOURCES


def authoritative_issuer_transaction_count_detail(evidence: Any) -> dict[str, Any] | None:
    """Build the public identity detail for independently issuer-reported evidence."""
    if not is_authoritative_issuer_row_count(evidence):
        return None
    count = int(evidence.count)
    evidence_source = str(evidence.source)
    detail: dict[str, Any] = {
        "raw_name": f"{evidence_source}_transaction_count",
        "raw_value": str(count),
        "normalized_value": str(count),
        "data_type": "integer",
        "source": f"row_count_evidence.{evidence_source}",
    }
    page = getattr(evidence, "page", None)
    if page is not None:
        detail["source_refs"] = [
            {
                "source": f"row_count_evidence.{evidence_source}",
                "page_id": f"page:{int(page):04d}",
            }
        ]
    evidence_ids = list(getattr(evidence, "evidence_ids", ()) or ())
    if evidence_ids:
        detail["evidence_ids"] = evidence_ids
    return detail


def _apply_source_reported_transaction_count(
    identity_fields: dict[str, dict],
    source_reported_count: Any,
) -> None:
    """Keep the identity aggregate aligned with the independently counted source rows."""
    detail = authoritative_issuer_transaction_count_detail(source_reported_count)
    if detail is not None:
        identity_fields["total_transactions"] = detail


def _physical_logical_row_mismatch_warning(
    physical_expected: int,
    style_meta: StyleMeta,
    reconstruction: Any,
) -> str:
    """Return a mismatch unless independent reconstruction evidence proves completeness."""
    if physical_expected <= 0 or physical_expected == style_meta.canonical_extracted:
        return ""

    evidence_source = str(getattr(reconstruction, "expected_evidence_source", "") or "")
    evidence_confidence = float(getattr(reconstruction, "expected_evidence_confidence", 0.0) or 0.0)
    evidence_count = int(getattr(reconstruction, "expected_primary_rows", 0) or 0)
    authoritative_count = (
        evidence_source in ISSUER_ROW_COUNT_EVIDENCE_SOURCES and evidence_confidence >= 0.85
    )
    evidence_proves_fuller_result = (
        authoritative_count
        and style_meta.canonical_extracted > physical_expected
        and evidence_count == style_meta.canonical_extracted
        and style_meta.expected_primary_rows == style_meta.canonical_extracted
    )
    if evidence_proves_fuller_result:
        return ""
    return f"BANK_PHYSICAL_LOGICAL_ROW_MISMATCH:physical={physical_expected}:canonical={style_meta.canonical_extracted}"


def _run_strategy_deployment(
    context_builder: Any,
    parse_result: Any,
    full_text: str,
    plugin: Any,
    *,
    adaptive: bool,
) -> tuple[StyleContext, StyleDetectionResult, list[dict[str, Any]], dict[str, dict], Any]:
    """Run one fresh BLO attempt with adaptive or exact eager deployment."""

    ctx = context_builder(parse_result, full_text)
    detection = BankStyleDetector().detect(ctx)
    registry = BankStyleParserRegistry(adaptive=adaptive)
    records, identity_fields, blo_meta = BankLedgerOrchestrator(registry).run(
        detection,
        ctx,
        plugin,
    )
    return ctx, detection, records, identity_fields, blo_meta


def _used_lazy_primary(blo_meta: Any) -> bool:
    return any(
        diagnostic.get("deployment_mode") == "lazy_primary"
        and diagnostic.get("completion_state") == "proven"
        for diagnostic in (getattr(blo_meta, "candidate_diagnostics", None) or [])
        if isinstance(diagnostic, dict)
    )


def _row_accounting_view(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, int, int]:
    parsed_rows = len(records)
    direction_count = sum(
        1 for record in records if (record.get("normalized") or {}).get("direction") in {"income", "expense"}
    )
    canonical_records = [
        record for record in records if is_canonical_row(record.get("normalized") or {})
    ]
    sourced_count = sum(1 for record in canonical_records if _has_single_page_source(record))
    return canonical_records, parsed_rows, direction_count, sourced_count


def _lazy_final_gate_failures(
    records: list[dict[str, Any]],
    *,
    parsed_rows: int,
    direction_count: int,
    sourced_count: int,
    physical_expected: int,
    source_reported_count: Any,
    invariant_failures: list[str],
) -> list[str]:
    """Return reasons that invalidate a provisional lazy-primary result."""

    failures: list[str] = []
    canonical_rows = len(records)
    if parsed_rows != canonical_rows:
        failures.append("canonical_filter_changed_row_count")
    if direction_count != parsed_rows:
        failures.append("direction_coverage_incomplete")
    if canonical_rows and sourced_count != canonical_rows:
        failures.append("source_page_coverage_incomplete")
    if is_authoritative_issuer_row_count(source_reported_count) and source_reported_count.count != canonical_rows:
        failures.append("issuer_row_count_mismatch")
    if physical_expected > 0 and physical_expected != canonical_rows:
        failures.append("physical_row_count_mismatch")
    if invariant_failures:
        failures.append("final_invariant_audit_failed")
    return failures


def run_bank_statement_extract(
    parse_result: Any,
    full_text: str,
    plugin: Any,
) -> BankExtractResult:
    """Run the canonical bank-statement extract pipeline."""
    extraction_route = resolve_bank_extraction_route(parse_result)
    context_builder = (
        build_digital_style_context
        if extraction_route is BankExtractionRoute.DIGITAL
        else build_scanned_style_context
    )
    ctx, detection, raw_records, identity_fields, blo_meta = _run_strategy_deployment(
        context_builder,
        parse_result,
        full_text,
        plugin,
        adaptive=True,
    )
    page_texts = page_texts_with_business_headers(
        parse_result,
        page_texts_from_parse_result(parse_result),
    )
    source_reported_count = resolve_row_count_evidence(
        ctx.full_text,
        page_texts=page_texts,
    )
    if source_reported_count.source in _ROW_PLANE_COUNT_SOURCES:
        source_reported_count = replace(
            source_reported_count,
            confidence=min(source_reported_count.confidence, 0.80),
        )
    physical_expected = (
        physical_transaction_row_estimate(parse_result)
        if extraction_route is BankExtractionRoute.DIGITAL
        else 0
    )

    records, parsed_rows, direction_count, sourced_count = _row_accounting_view(raw_records)
    invariant_failures: list[str] | None = None
    if _used_lazy_primary(blo_meta):
        invariant_failures = audit_bank_statement_invariants(
            records,
            ctx.full_text,
            page_texts=page_texts,
            row_count_evidence=source_reported_count,
        )
        final_gate_failures = _lazy_final_gate_failures(
            records,
            parsed_rows=parsed_rows,
            direction_count=direction_count,
            sourced_count=sourced_count,
            physical_expected=physical_expected,
            source_reported_count=source_reported_count,
            invariant_failures=invariant_failures,
        )
        if final_gate_failures:
            prior_lazy_attempt = next(
                (
                    dict(diagnostic)
                    for diagnostic in (getattr(blo_meta, "candidate_diagnostics", None) or [])
                    if isinstance(diagnostic, dict)
                    and diagnostic.get("deployment_mode") == "lazy_primary"
                ),
                None,
            )
            ctx, detection, raw_records, identity_fields, blo_meta = _run_strategy_deployment(
                context_builder,
                parse_result,
                full_text,
                plugin,
                adaptive=False,
            )
            diagnostics = getattr(blo_meta, "candidate_diagnostics", None) or []
            if diagnostics and isinstance(diagnostics[0], dict):
                diagnostics[0]["deployment_mode"] = "eager_final_gate_fallback"
                diagnostics[0]["completion_state"] = "unknown"
                diagnostics[0]["completion_reason"] = ":".join(final_gate_failures)
                if prior_lazy_attempt is not None:
                    diagnostics[0]["prior_lazy_attempt"] = prior_lazy_attempt
            records, parsed_rows, direction_count, sourced_count = _row_accounting_view(raw_records)
            invariant_failures = None

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
    _drop_unbound_issuer(identity_fields)
    # Generic KV/text/atom identity recovery can observe a count label, but it
    # cannot prove that the label owns the complete statement. Fail closed and
    # publish this exact-looking field only from issuer-authoritative evidence.
    identity_fields.pop("total_transactions", None)
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
    if mismatch_warning := _physical_logical_row_mismatch_warning(
        physical_expected,
        style_meta,
        ctx.reconstruction,
    ):
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
    if records and sourced_count < len(records):
        warnings.append(f"BANK_SOURCE_PAGE_COVERAGE_LOW:sourced={sourced_count}:canonical={len(records)}")
        style_meta.extract_status = "degraded"
    if invariant_failures is None:
        invariant_failures = audit_bank_statement_invariants(
            records,
            ctx.full_text,
            page_texts=page_texts,
            row_count_evidence=source_reported_count,
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
        extraction_route=extraction_route,
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
