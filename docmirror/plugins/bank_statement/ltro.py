# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Logical Table Reconstruction Orchestrator (LTRO) for bank statements.

When canonical physical tables are empty, rebuilds logical ledger grids from full_text
using ordered strategies: pipe text → spaced OCR → none.

Pipeline role: called by a route-specific context builder before style detection.

Key exports: ``ReconstructionMeta``, ``reconstruct_digital_tables``,
``reconstruct_scanned_tables``.

Dependencies: ``pipe_text_table_builder``, ``text_table_builder``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from docmirror.plugins.bank_statement.pipe_text_table_builder import (
    build_tables_from_pipe_text,
    count_expected_primary_rows,
    detect_pipe_header_in_text,
)
from docmirror.plugins.bank_statement.text_table_builder import build_tables_from_spaced_ocr_text

SourceKind = Literal[
    "canonical_table",
    "pipe_text",
    "spaced_ocr",
    "stacked_text",
    "native_wide_table",
    "canonical_evidence_table",
    "positioned_record_block",
    "ocr_implicit_table",
    "none",
]


@dataclass
class ReconstructionMeta:
    source: SourceKind
    expected_primary_rows: int = 0
    stitched_continuation_rows: int = 0
    pipe_header_detected: bool = False
    pipe_parse_failed: bool = False
    pages_scanned: int = 0
    spe_primary: str | None = None
    spe_table_extraction: str | None = None
    expected_evidence_source: str = ""
    expected_evidence_confidence: float = 0.0


def _resolve_source_row_count_evidence(full_text: str, parse_result: Any) -> Any:
    """Resolve source count evidence behind one patchable LTRO seam."""
    from docmirror.plugins.bank_statement.wide_table_recovery import (
        page_texts_from_parse_result,
        resolve_row_count_evidence,
    )

    return resolve_row_count_evidence(
        full_text,
        page_texts=page_texts_from_parse_result(parse_result),
    )


def _canonical_reconstruction(
    canonical_tables: list[list[list[str]]],
    full_text: str,
    *,
    page_count: int = 0,
    structure_spe: dict | None = None,
    parse_result: Any | None = None,
    allow_pipe_metadata: bool = False,
) -> tuple[list[list[list[str]]], ReconstructionMeta] | None:
    spe_primary = (structure_spe or {}).get("primary")
    spe_mode = (structure_spe or {}).get("table_extraction")

    canonical_rows = _usable_bank_ledger_row_count(canonical_tables)
    if canonical_rows > 0:
        expected = _canonical_table_expected_rows(
            canonical_tables,
            parse_result=parse_result,
            structure_spe=structure_spe,
        )
        source_evidence = _resolve_source_row_count_evidence(full_text, parse_result)
        # Import lazily because extract_pipeline imports the context builders
        # that invoke LTRO. The shared predicate keeps count authority identical
        # at reconstruction and public-projection boundaries.
        from docmirror.plugins.bank_statement.extract_pipeline import is_authoritative_issuer_row_count

        authoritative_evidence = is_authoritative_issuer_row_count(source_evidence)
        if authoritative_evidence:
            expected = source_evidence.count
        return canonical_tables, ReconstructionMeta(
            source="canonical_table",
            expected_primary_rows=expected,
            pipe_header_detected=(detect_pipe_header_in_text(full_text) if allow_pipe_metadata else False),
            pages_scanned=page_count,
            spe_primary=spe_primary,
            spe_table_extraction=spe_mode,
            expected_evidence_source=(
                source_evidence.source
                if authoritative_evidence
                else ""
            ),
            expected_evidence_confidence=(
                source_evidence.confidence
                if authoritative_evidence
                else 0.0
            ),
        )

    return None


def reconstruct_digital_tables(
    canonical_tables: list[list[list[str]]],
    full_text: str,
    *,
    page_count: int = 0,
    structure_spe: dict | None = None,
    parse_result: Any | None = None,
) -> tuple[list[list[list[str]]], ReconstructionMeta]:
    """Rebuild a digital ledger without invoking OCR reconstruction."""

    canonical = _canonical_reconstruction(
        canonical_tables,
        full_text,
        page_count=page_count,
        structure_spe=structure_spe,
        parse_result=parse_result,
        allow_pipe_metadata=True,
    )
    if canonical is not None:
        tables, meta = canonical
        pipe_detected = detect_pipe_header_in_text(full_text)
        pipe_tables = build_tables_from_pipe_text(full_text) if pipe_detected else []
        pipe_rows = sum(max(len(table) - 1, 0) for table in pipe_tables)
        if pipe_rows > _usable_bank_ledger_row_count(tables):
            return pipe_tables, ReconstructionMeta(
                source="pipe_text",
                expected_primary_rows=pipe_rows,
                pipe_header_detected=True,
                pages_scanned=page_count,
                spe_primary=(structure_spe or {}).get("primary"),
                spe_table_extraction=(structure_spe or {}).get("table_extraction"),
            )
        return tables, meta

    from docmirror.evidence.spe_consumer import should_block_pipe_ltro, should_force_ltro

    blocked = should_block_pipe_ltro(structure_spe)
    if blocked:
        force_pipe = False
        pipe_detected = False
    else:
        force_pipe = should_force_ltro(
            mirror_tables=canonical_tables,
            full_text=full_text,
            structure_spe=structure_spe,
        )
        pipe_detected = detect_pipe_header_in_text(full_text) or force_pipe
    pipe_tables = build_tables_from_pipe_text(full_text) if pipe_detected else []
    if pipe_tables:
        data_rows = len(pipe_tables[0]) - 1
        return pipe_tables, ReconstructionMeta(
            source="pipe_text",
            expected_primary_rows=data_rows,
            pipe_header_detected=True,
            pages_scanned=page_count,
            spe_primary=(structure_spe or {}).get("primary"),
            spe_table_extraction=(structure_spe or {}).get("table_extraction"),
        )

    if pipe_detected:
        return [], ReconstructionMeta(
            source="none",
            expected_primary_rows=count_expected_primary_rows(full_text),
            pipe_header_detected=True,
            pipe_parse_failed=True,
            pages_scanned=page_count,
            spe_primary=(structure_spe or {}).get("primary"),
            spe_table_extraction=(structure_spe or {}).get("table_extraction"),
        )

    return [], ReconstructionMeta(
        source="none",
        expected_primary_rows=0,
        pipe_header_detected=pipe_detected,
        pages_scanned=page_count,
        spe_primary=(structure_spe or {}).get("primary"),
        spe_table_extraction=(structure_spe or {}).get("table_extraction"),
    )


def reconstruct_scanned_tables(
    canonical_tables: list[list[list[str]]],
    full_text: str,
    *,
    page_count: int = 0,
    structure_spe: dict | None = None,
    parse_result: Any | None = None,
) -> tuple[list[list[list[str]]], ReconstructionMeta]:
    """Rebuild an OCR ledger without invoking native-text reconstruction."""

    canonical = _canonical_reconstruction(
        canonical_tables,
        full_text,
        page_count=page_count,
        structure_spe=structure_spe,
        parse_result=parse_result,
        allow_pipe_metadata=False,
    )
    if canonical is not None:
        return canonical
    ocr_tables = build_tables_from_spaced_ocr_text(full_text)
    if ocr_tables:
        expected = len(ocr_tables[0]) - 1
        return ocr_tables, ReconstructionMeta(
            source="spaced_ocr",
            expected_primary_rows=expected,
            pipe_header_detected=False,
            pages_scanned=page_count,
            spe_primary=(structure_spe or {}).get("primary"),
            spe_table_extraction=(structure_spe or {}).get("table_extraction"),
        )
    return [], ReconstructionMeta(
        source="none",
        pages_scanned=page_count,
        spe_primary=(structure_spe or {}).get("primary"),
        spe_table_extraction=(structure_spe or {}).get("table_extraction"),
    )


def reconstruct_tables(
    canonical_tables: list[list[list[str]]],
    full_text: str,
    **kwargs: Any,
) -> tuple[list[list[list[str]]], ReconstructionMeta]:
    """Compatibility wrapper for explicit route-aware LTRO callers.

    Omitting ``route`` retains the historical digital behavior only.  OCR
    callers must opt into ``route='scanned'``; the wrapper never mixes source
    strategies in one invocation.
    """

    route = str(kwargs.pop("route", "digital") or "digital").lower()
    if route == "digital":
        return reconstruct_digital_tables(canonical_tables, full_text, **kwargs)
    if route == "scanned":
        return reconstruct_scanned_tables(canonical_tables, full_text, **kwargs)
    raise ValueError(f"unsupported bank LTRO route: {route}")


def _usable_bank_ledger_row_count(canonical_tables: list[list[list[str]]]) -> int:
    """Return transaction-like row count in semantically usable bank tables."""
    from docmirror.plugins.bank_statement.community_plugin import BANK_COLUMN_REGISTRY
    from docmirror.plugins.bank_statement.header_resolve import (
        detect_headers,
        has_split_debit_credit_headers,
    )
    from docmirror.plugins.bank_statement.row_extract import count_transaction_data_rows

    count = 0
    for table in canonical_tables:
        if not table:
            continue
        header = detect_headers([table], BANK_COLUMN_REGISTRY)
        if header is None:
            continue
        fields = set(header.col_map)
        has_amount = "amount" in fields or has_split_debit_credit_headers([table])
        if not (has_amount and "balance" in fields and fields.intersection({"date", "timestamp"})):
            continue
        count += count_transaction_data_rows([table], header)
    return count


def _canonical_table_expected_rows(
    canonical_tables: list[list[list[str]]],
    *,
    parse_result: Any | None = None,
    structure_spe: dict | None = None,
) -> int:
    """Mirror SSOT for coverage denominator (ADR-BS-07); raw max fallback."""
    if parse_result is not None:
        from docmirror.evidence.spe_consumer import mirror_expected_primary_rows

        expected = mirror_expected_primary_rows(parse_result, structure_spe)
        if expected > 0:
            return expected
    if canonical_tables:
        from docmirror.plugins.bank_statement.header_resolve import has_split_debit_credit_headers

        if has_split_debit_credit_headers(canonical_tables):
            usable_rows = _usable_bank_ledger_row_count(canonical_tables)
            if usable_rows > 0:
                return usable_rows
        return max((max(len(tbl) - 1, 0) for tbl in canonical_tables if tbl), default=0)
    return 0
