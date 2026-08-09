# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Canonical transaction record builders and style metadata for bank statements.

Defines ``CANONICAL_FIELDS``, ``StyleMeta`` (style_id, confidence, parser chain),
and helpers to normalize raw style-parser output into edition-ready record dicts with
``raw`` and ``normalized`` sub-objects.

Pipeline role: ``style_registry`` and individual style parsers call these builders
before ``community_plugin`` serializes DEC output.

Key exports: ``CANONICAL_FIELDS``, ``StyleMeta``, ``build_style_meta``,
``ensure_canonical_normalized``, ``records_from_raw_transactions``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

CANONICAL_FIELDS = (
    "date",
    "timestamp",
    "summary",
    "amount",
    "amount_cny",
    "direction",
    "balance",
    "counter_party",
    "counter_account",
    "counterparty_status",
    "reference",
)


@dataclass
class StyleMeta:
    style_id: str
    style_confidence: float
    parser_chain: list[str] = field(default_factory=list)
    institution_hint: str | None = None
    secondary_styles: list[str] = field(default_factory=list)
    reconstruction_source: str = ""
    expected_primary_rows: int = 0
    extracted_rows: int = 0
    coverage_ratio: float = 0.0
    institution_authority: str = ""
    pipe_parse_failed: bool = False
    canonical_expected: int = 0
    canonical_extracted: int = 0
    canonical_ratio: float = 0.0
    extract_status: str = "success"
    blo_tables_parsed: int = 0
    blo_tables_skipped: int = 0

    def to_properties(self) -> dict[str, Any]:
        return {
            "style_id": self.style_id,
            "style_confidence": round(self.style_confidence, 4),
            "parser_chain": list(self.parser_chain),
            "institution_hint": self.institution_hint or "",
            "secondary_styles": list(self.secondary_styles),
            "reconstruction_source": self.reconstruction_source,
            "expected_primary_rows": self.expected_primary_rows,
            "extracted_rows": self.extracted_rows,
            "coverage_ratio": round(self.coverage_ratio, 4),
            "institution_authority": self.institution_authority,
            "pipe_parse_failed": self.pipe_parse_failed,
            "canonical_expected": self.canonical_expected,
            "canonical_extracted": self.canonical_extracted,
            "canonical_ratio": round(self.canonical_ratio, 4),
            "extract_status": self.extract_status,
            "blo_tables_parsed": self.blo_tables_parsed,
            "blo_tables_skipped": self.blo_tables_skipped,
        }


def build_style_meta(
    detection: Any,
    *,
    reconstruction: Any = None,
    record_count: int = 0,
    parse_result: Any = None,
    records: list[dict[str, Any]] | None = None,
    blo_meta: Any = None,
    source_reported_count: int = 0,
) -> StyleMeta:
    expected_candidates: list[int] = []
    source = ""
    pipe_failed = False
    expected_evidence_source = ""
    expected_evidence_confidence = 0.0
    stitched_continuation_rows = int(getattr(reconstruction, "stitched_continuation_rows", 0) or 0)
    if reconstruction is not None:
        reconstruction_expected = int(getattr(reconstruction, "expected_primary_rows", 0) or 0)
        if stitched_continuation_rows > 0:
            reconstruction_expected = max(0, reconstruction_expected - stitched_continuation_rows)
        source = getattr(reconstruction, "source", "") or ""
        expected_evidence_source = str(getattr(reconstruction, "expected_evidence_source", "") or "")
        expected_evidence_confidence = float(getattr(reconstruction, "expected_evidence_confidence", 0.0) or 0.0)
        pipe_failed = bool(getattr(reconstruction, "pipe_parse_failed", False))
        if (
            source
            in {
                "canonical_table",
                "canonical_physical_tables",
                "canonical_evidence_table",
                "native_wide_table",
                "positioned_record_block",
                "ocr_implicit_table",
            }
            and reconstruction_expected > 0
        ):
            expected_candidates.append(reconstruction_expected)

    from docmirror.plugins.bank_statement.canonical_quality import (
        audit_cqf,
        canonical_expected_from_parse_result,
    )

    canonical_expected = canonical_expected_from_parse_result(parse_result)
    if canonical_expected > 0 and stitched_continuation_rows > 0:
        canonical_expected = max(0, canonical_expected - stitched_continuation_rows)
    # A source-reported total is independent of every parser candidate and must
    # remain the strongest denominator. Positioned blocks and a selected evidence
    # table are stronger than a sparse physical-table estimate when no such total exists.
    if source_reported_count > 0:
        expected = int(source_reported_count)
    elif (
        expected_evidence_source
        in {
            "split_footer",
            "header_total",
            "page_footer",
            "page_transaction_anchors",
            "physical_rows",
            "positioned_date_anchors",
            "positioned_record_blocks",
        }
        and expected_evidence_confidence >= 0.85
        and reconstruction_expected > 0
    ):
        expected = max(reconstruction_expected, canonical_expected)
    elif source in {"positioned_record_block", "canonical_evidence_table"} and reconstruction_expected > 0:
        expected = max(reconstruction_expected, canonical_expected)
    elif canonical_expected > 0:
        expected = canonical_expected
    else:
        expected = max(expected_candidates, default=0)

    cqf = audit_cqf(records or [], canonical_expected=expected)
    coverage = cqf.coverage_ratio
    if records is None and expected > 0:
        coverage = min(record_count / expected, 1.0) if record_count > 0 else 0.0
    if expected <= 0 and record_count > 0:
        coverage = 1.0

    blo_parsed = 0
    blo_skipped = 0
    if blo_meta is not None:
        blo_parsed = int(getattr(blo_meta, "tables_parsed", 0) or 0)
        blo_skipped = int(getattr(blo_meta, "tables_skipped", 0) or 0)

    return StyleMeta(
        style_id=detection.primary_style,
        style_confidence=detection.confidence,
        parser_chain=list(detection.parser_chain),
        institution_hint=detection.institution_hint,
        secondary_styles=list(detection.secondary_styles),
        reconstruction_source=source,
        expected_primary_rows=expected,
        extracted_rows=record_count,
        coverage_ratio=coverage,
        institution_authority=getattr(detection, "institution_authority", "") or "",
        pipe_parse_failed=pipe_failed,
        canonical_expected=cqf.canonical_expected,
        canonical_extracted=cqf.canonical_extracted,
        canonical_ratio=cqf.canonical_ratio,
        extract_status=cqf.extract_status,
        blo_tables_parsed=blo_parsed,
        blo_tables_skipped=blo_skipped,
    )


def dedupe_transaction_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Dedupe by (date, amount, balance, counter_party) — ADR-BS-05."""
    seen: set[tuple[Any, ...]] = set()
    out: list[dict[str, Any]] = []
    for rec in records:
        norm = rec.get("normalized") or {}
        raw = rec.get("raw") if isinstance(rec.get("raw"), dict) else {}
        sequence = str(norm.get("sequence_no") or _raw_sequence(raw) or "").strip()
        reference = str(norm.get("reference") or _raw_reference(raw) or "").strip()
        balance = norm.get("balance")
        try:
            balance_key = float(balance) if balance not in (None, "") else None
        except (TypeError, ValueError):
            balance_key = balance
        try:
            amount_key = float(norm.get("amount") or 0)
        except (TypeError, ValueError):
            amount_key = norm.get("amount")
        source = rec.get("source") if isinstance(rec.get("source"), dict) else {}
        source_scope = _source_sequence_scope(source)
        business_key = (
            str(norm.get("date") or ""),
            str(norm.get("direction") or ""),
            amount_key,
            balance_key,
            str(norm.get("counter_party") or ""),
            str(norm.get("counter_account") or ""),
            str(norm.get("summary") or ""),
        )
        if sequence and source_scope:
            key = ("sequence", source_scope, sequence, business_key)
        elif sequence:
            key = ("sequence", sequence, business_key)
        elif reference:
            key = ("reference", reference, business_key)
        elif source_scope:
            key = ("business", source_scope, business_key)
        else:
            key = ("business", business_key)
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(rec))
    for idx, rec in enumerate(out, start=1):
        rec["row_index"] = idx
    return out


def _source_sequence_scope(source: dict[str, Any]) -> tuple[Any, ...]:
    source_page = source.get("source_page")
    page_range = source.get("page_range")
    table_id = str(source.get("table_id") or "").strip()
    row_index = source.get("source_row_index")
    if source_page in (None, "") and not page_range and not table_id and row_index in (None, ""):
        return ()
    normalized_range = tuple(page_range) if isinstance(page_range, (list, tuple)) else ()
    return ("source", source_page, normalized_range, table_id, row_index)


def _raw_sequence(raw: dict[str, Any]) -> str:
    for key, value in raw.items():
        if any(marker in str(key) for marker in ("序号", "交易序号", "Sequence")):
            sequence = str(value or "").strip()
            if re.fullmatch(r"\d{1,8}", sequence):
                return sequence
    return ""


def _raw_reference(raw: dict[str, Any]) -> str:
    for key, value in raw.items():
        if any(marker in str(key) for marker in ("交易流水号", "流水号", "Reference")):
            return str(value or "").strip()
    return ""


def ensure_canonical_normalized(normalized: dict[str, Any], standard_fields: list[str]) -> dict[str, Any]:
    from docmirror.tables.cell_normalizer import normalize_cell_line_breaks

    out = dict(normalized)
    summary = out.get("summary")
    if isinstance(summary, str) and summary:
        out["summary"] = normalize_cell_line_breaks(summary).strip()
    for fld in standard_fields:
        if fld not in out:
            out[fld] = "" if fld not in ("amount", "amount_cny", "balance") else None
    if out.get("amount_cny") is None:
        out["amount_cny"] = out.get("amount")
    if "direction" not in out:
        out["direction"] = "other"
    if not str(out.get("counterparty_status") or "").strip():
        counter_party = str(out.get("counter_party") or "").strip()
        counter_account = str(out.get("counter_account") or "").strip()
        out["counterparty_status"] = "present" if counter_party or counter_account else "source_null"
    return out


def records_from_raw_transactions(
    transactions: list[dict[str, Any]],
    *,
    normalize_fn,
    style_id: str,
    canonical_raw_fn=None,
) -> list[dict[str, Any]]:
    from docmirror.plugins._base.base_table_parser import public_record_raw

    records: list[dict[str, Any]] = []
    for idx, raw_txn in enumerate(transactions, start=1):
        raw = dict(raw_txn)
        raw.setdefault("_style_id", style_id)
        normalized = normalize_fn(raw_txn)
        source_raw = raw_txn.get("_source_raw")
        raw_public = public_record_raw(dict(source_raw)) if isinstance(source_raw, dict) else public_record_raw(raw)
        record = {
            "row_index": idx,
            "raw": raw_public,
            "normalized": normalized,
        }
        if callable(canonical_raw_fn):
            record["canonical_raw"] = canonical_raw_fn(raw_public, normalized)
        source = raw_txn.get("_source")
        if not isinstance(source, dict):
            try:
                source_page = int(str(raw_txn.get("_source_page") or "").strip())
            except (TypeError, ValueError):
                source_page = 0
            if source_page > 0:
                source = {
                    "source_page": source_page,
                    "page_range": [source_page, source_page],
                }
                table_id = str(raw_txn.get("_source_table_id") or "").strip()
                row_index = str(raw_txn.get("_source_row_index") or "").strip()
                if table_id:
                    source["table_id"] = table_id
                if row_index.isdigit():
                    source["source_row_index"] = int(row_index)
        if isinstance(source, dict):
            record["source"] = dict(source)
        records.append(record)
    return records
