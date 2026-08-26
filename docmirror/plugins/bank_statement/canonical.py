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
    "direction",
    "balance",
    "counter_party",
    "counter_account",
    "counterparty_status",
    "reference",
)

_ISSUER_COUNT_SOURCES = frozenset(
    {
        "split_footer",
        "header_total",
        "statement_header_totals",
        "cumulative_footer_total",
        "page_footer",
    }
)

_NATIVE_SOURCE_REPAIR_KIND = "adjacent_summary_signed_amount_spill"
_BORDERLESS_HEADER_NORMALIZATION = "borderless_header_normalization"
_NATIVE_SOURCE_REPAIR_KEYS = frozenset(
    {
        "kind",
        "summary_header",
        "amount_header",
        "summary_prefix",
        "source_summary",
        "source_amount",
        "working_summary",
        "working_amount",
        "working_transform",
    }
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
    source_reported_count: Any = None,
) -> StyleMeta:
    source = ""
    pipe_failed = False
    expected_evidence_source = ""
    expected_evidence_confidence = 0.0
    stitched_continuation_rows = int(getattr(reconstruction, "stitched_continuation_rows", 0) or 0)
    if reconstruction is not None:
        reconstruction_expected = int(getattr(reconstruction, "expected_primary_rows", 0) or 0)
        source = getattr(reconstruction, "source", "") or ""
        expected_evidence_source = str(getattr(reconstruction, "expected_evidence_source", "") or "")
        expected_evidence_confidence = float(getattr(reconstruction, "expected_evidence_confidence", 0.0) or 0.0)
        # A stitched continuation is a source fragment merged into an existing
        # transaction, not an extra transaction.  Only candidate/raw-table row
        # estimates can include that fragment in their denominator.  Issuer
        # totals already count business transactions and must never be reduced
        # by the stitch count. Candidate/source-row-plane counts can include a
        # continuation fragment and therefore receive no such exemption.
        if (
            stitched_continuation_rows > 0
            and expected_evidence_source
            not in {
                "split_footer",
                "header_total",
                "statement_header_totals",
                "cumulative_footer_total",
                "page_footer",
            }
        ):
            reconstruction_expected = max(0, reconstruction_expected - stitched_continuation_rows)
        pipe_failed = bool(getattr(reconstruction, "pipe_parse_failed", False))
    from docmirror.plugins.bank_statement.canonical_quality import (
        audit_cqf,
    )

    # Parser tables, positioned blocks, page anchors, and mirror row estimates
    # describe only the rows a candidate happened to recover. They remain
    # available to candidate selection and diagnostics, but cannot certify that
    # the public dataset covers the complete source document. Only an issuer
    # count may populate the public completeness denominator.
    reported_source = str(getattr(source_reported_count, "source", "") or "")
    if hasattr(source_reported_count, "confidence"):
        confidence = float(getattr(source_reported_count, "confidence", 0.0) or 0.0)
        reported_count = (
            int(getattr(source_reported_count, "count", 0) or 0)
            if confidence >= 0.85 and reported_source in _ISSUER_COUNT_SOURCES
            else 0
        )
    else:
        # A scalar has no provenance and therefore cannot prove that the count
        # came from the issuer instead of the selected parser candidate.
        reported_count = 0
    if reported_count > 0:
        expected = reported_count
    elif (
        expected_evidence_source in _ISSUER_COUNT_SOURCES
        and expected_evidence_confidence >= 0.85
        and reconstruction_expected > 0
    ):
        expected = reconstruction_expected
    else:
        expected = 0

    cqf = audit_cqf(records or [], canonical_expected=expected)
    coverage = cqf.coverage_ratio
    if records is None and expected > 0:
        coverage = min(record_count / expected, 1.0) if record_count > 0 else 0.0

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
    """Dedupe only records that identify the same physical source row.

    Equal business values are not row identity.  Ledgers commonly contain
    repeated fees, reversals, and batch payments on the same page.  A row may
    therefore be collapsed only when its provenance identifies the same table
    row, evidence row, or bounding box; page membership alone is insufficient.
    Alternative extraction planes are reconciled before this function during
    candidate selection rather than by value-only deletion here.
    """
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
        source_row = _source_row_identity(source)
        business_key = (
            str(norm.get("date") or ""),
            str(norm.get("direction") or ""),
            amount_key,
            balance_key,
            str(norm.get("counter_party") or ""),
            str(norm.get("counter_account") or ""),
            str(norm.get("summary") or ""),
        )
        if sequence and source_row:
            key = ("sequence", source_row, sequence, business_key)
        elif reference and source_row:
            key = ("reference", source_row, reference, business_key)
        elif source_row:
            key = ("business", source_row, business_key)
        else:
            # Sequence/reference values can repeat or reset.  Without physical
            # row provenance, equal values still do not prove row identity.
            key = ("unsourced", len(out), business_key)
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(rec))
    for idx, rec in enumerate(out, start=1):
        rec["row_index"] = idx
    return out


def _source_row_identity(source: dict[str, Any]) -> tuple[Any, ...]:
    """Return stable row-level provenance, never a page-only scope."""

    source_page = source.get("source_page")
    page_range = source.get("page_range")
    table_id = str(source.get("table_id") or "").strip()
    row_index = source.get("source_row_index")
    normalized_range = tuple(page_range) if isinstance(page_range, (list, tuple)) else ()
    page_scope = (source_page, normalized_range)

    if table_id and row_index not in (None, ""):
        return ("table_row", *page_scope, table_id, row_index)

    bbox = source.get("bbox") or source.get("source_bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        try:
            normalized_bbox = tuple(round(float(value), 4) for value in bbox[:4])
        except (TypeError, ValueError):
            normalized_bbox = ()
        if normalized_bbox:
            return ("bbox", *page_scope, normalized_bbox)

    evidence_ids = source.get("evidence_ids")
    if isinstance(evidence_ids, (list, tuple, set)):
        normalized_evidence = tuple(sorted({str(value) for value in evidence_ids if str(value)}))
        if normalized_evidence:
            return ("evidence", *page_scope, normalized_evidence)

    return ()


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
    if "direction" not in out:
        out["direction"] = "other"
    if not str(out.get("counterparty_status") or "").strip():
        counter_party = str(out.get("counter_party") or "").strip()
        counter_account = str(out.get("counter_account") or "").strip()
        out["counterparty_status"] = "present" if counter_party or counter_account else "source_null"
    return out


def _canonical_raw_input_for_source_repair(
    raw_txn: dict[str, Any],
    *,
    source_public: dict[str, Any],
    working_public: dict[str, Any],
) -> dict[str, Any]:
    """Choose working cells only under an exact repair or header-normalization contract."""
    canonical_mode = raw_txn.get("_canonical_raw_from_working")
    source_raw = raw_txn.get("_source_raw")
    if canonical_mode == _BORDERLESS_HEADER_NORMALIZATION:
        if (
            not isinstance(source_raw, dict)
            or source_public != source_raw
            or len(source_public) != len(working_public)
            or list(source_public.values()) != list(working_public.values())
        ):
            return source_public
        return working_public
    if canonical_mode is not True:
        return source_public
    manifest = raw_txn.get("_source_repair_manifest")
    if (
        not isinstance(source_raw, dict)
        or not isinstance(manifest, dict)
        or set(manifest) != _NATIVE_SOURCE_REPAIR_KEYS
        or not all(isinstance(value, str) for value in manifest.values())
        or manifest.get("kind") != _NATIVE_SOURCE_REPAIR_KIND
        or set(source_public) != set(working_public)
        or set(source_raw) != set(working_public)
    ):
        return source_public

    if source_public != source_raw:
        return source_public
    from docmirror.plugins.bank_statement.wide_table_recovery import (
        _validated_native_source_repair_working_map,
    )

    expected_working = _validated_native_source_repair_working_map(
        source_raw,
        list(source_raw),
        manifest,
    )
    if expected_working is None or working_public != expected_working:
        return source_public
    return working_public


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
        working_public = public_record_raw(raw)
        source_raw = raw_txn.get("_source_raw")
        raw_public = public_record_raw(dict(source_raw)) if isinstance(source_raw, dict) else working_public
        record = {
            "row_index": idx,
            "raw": raw_public,
            "normalized": normalized,
        }
        scope_text = str(raw_txn.get("_document_scope_text") or "").strip()
        if scope_text:
            record["_document_scope_text"] = scope_text
        if callable(canonical_raw_fn):
            canonical_raw_input = _canonical_raw_input_for_source_repair(
                raw_txn,
                source_public=raw_public,
                working_public=working_public,
            )
            record["canonical_raw"] = canonical_raw_fn(canonical_raw_input, normalized)
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
