# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Bank Ledger Orchestrator (BLO) — Plugin multi-table export SSOT (ADR-BS-05).

Iterates passed ``LogicalTable`` instances, runs style parser chain per table,
merges and dedupes canonical records. Single-table / pipe LTRO paths preserve
raw behaviour via one synthetic block from ``ctx.tables``.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field, replace
from datetime import date
from typing import Any

from docmirror.models.entities.parse_result import LogicalTable
from docmirror.plugins.bank_statement.canonical import dedupe_transaction_rows
from docmirror.plugins.bank_statement.canonical_quality import is_canonical_row
from docmirror.plugins.bank_statement.context import StyleContext
from docmirror.plugins.bank_statement.row_extract import row_has_transaction_data
from docmirror.plugins.bank_statement.style_detector import BankStyleDetector, StyleDetectionResult
from docmirror.tables.access import get_logical_tables

logger = logging.getLogger(__name__)

_INHERIT_CONFIDENCE = 0.55


@dataclass
class BLOMeta:
    tables_parsed: int = 0
    tables_skipped: int = 0
    logical_table_count: int = 0
    candidate_diagnostics: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, int]:
        return {
            "tables_parsed": self.tables_parsed,
            "tables_skipped": self.tables_skipped,
            "logical_table_count": self.logical_table_count,
        }


def logical_table_to_matrices(lt: LogicalTable) -> list[list[list[str]]]:
    """Convert one logical table to plugin matrix form."""
    matrix: list[list[str]] = []
    headers = list(lt.headers or [])
    if headers:
        matrix.append([str(h or "") for h in headers])
    for row in lt.rows or []:
        cells = [str(getattr(cell, "cleaned", None) or getattr(cell, "text", "") or "") for cell in (row.cells or [])]
        if any(c.strip() for c in cells):
            matrix.append(cells)
    return [matrix] if matrix else []


def _iter_parse_blocks(ctx: StyleContext) -> list[tuple[LogicalTable | None, list[list[list[str]]]]]:
    logical_tables = get_logical_tables(ctx.parse_result) if ctx.parse_result else []
    if logical_tables:
        return [(lt, logical_table_to_matrices(lt)) for lt in logical_tables]
    if ctx.tables:
        return [(None, ctx.tables)]
    return []


def _logical_table_page_bounds(table: LogicalTable) -> tuple[int, int] | None:
    pages: list[int] = []
    for value in getattr(table, "source_pages", []) or []:
        try:
            page = int(value)
        except (TypeError, ValueError):
            continue
        if page > 0:
            pages.append(page)
    if pages:
        unique = sorted(set(pages))
        if unique != list(range(unique[0], unique[-1] + 1)):
            return None
        return unique[0], unique[-1]

    span = getattr(table, "page_span", None)
    if not isinstance(span, (list, tuple)) or len(span) != 2:
        return None
    try:
        start, end = int(span[0]), int(span[1])
    except (TypeError, ValueError):
        return None
    if start <= 0 or end < start:
        return None
    return start, end


def _looks_like_ledger_headers(headers: list[str]) -> bool:
    compact = ["".join(str(value or "").split()).lower() for value in headers]
    has_date = any(any(marker in value for marker in ("日期", "时间", "date", "time")) for value in compact)
    has_money = any(
        any(marker in value for marker in ("金额", "收入", "支出", "借方", "贷方", "amount", "debit", "credit"))
        for value in compact
    )
    return has_date and has_money


def _merge_quarantine_continuation_matrices(
    previous: LogicalTable | None,
    quarantined: LogicalTable,
) -> list[list[list[str]]]:
    """Recover a forward ledger fragment whose first row was promoted to headers.

    LTQG must remain fail-closed for arbitrary failed tables.  The only accepted
    exception is an adjacent ``merge_quarantine`` continuation with the exact
    preceding ledger width, a transaction-shaped promoted header, and an entirely
    transaction-shaped body.  The trusted preceding schema is reused; no failed
    header text is interpreted as a new schema.
    """
    if previous is None or not getattr(previous, "quality_passed", True):
        return []
    if str(getattr(quarantined, "quality_skip_reason", "") or "") != "merge_quarantine":
        return []

    previous_headers = [str(value or "") for value in (getattr(previous, "headers", []) or [])]
    promoted_row = [str(value or "") for value in (getattr(quarantined, "headers", []) or [])]
    body_rows = [
        [str(getattr(cell, "cleaned", None) or getattr(cell, "text", "") or "") for cell in (row.cells or [])]
        for row in (getattr(quarantined, "rows", []) or [])
    ]
    width = len(previous_headers)
    if width < 4 or not _looks_like_ledger_headers(previous_headers):
        return []
    if len(promoted_row) != width or not body_rows or any(len(row) != width for row in body_rows):
        return []
    if not row_has_transaction_data(promoted_row) or not all(row_has_transaction_data(row) for row in body_rows):
        return []

    previous_bounds = _logical_table_page_bounds(previous)
    current_bounds = _logical_table_page_bounds(quarantined)
    if previous_bounds is None or current_bounds is None or current_bounds[0] != previous_bounds[1] + 1:
        return []

    return [[previous_headers, promoted_row, *body_rows]]


def _compact_source_value(value: Any) -> str:
    return "".join(unicodedata.normalize("NFKC", str(value or "")).split())


def _physical_raw_rows(table: Any) -> list[list[str]]:
    metadata = getattr(table, "metadata", None) or {}
    raw_rows = metadata.get("raw_rows") if isinstance(metadata, dict) else None
    if isinstance(raw_rows, list) and raw_rows and all(isinstance(row, list) for row in raw_rows):
        return [[str(value or "") for value in row] for row in raw_rows]

    rows = [[str(value or "") for value in (getattr(table, "headers", []) or [])]]
    rows.extend(
        [str(getattr(cell, "cleaned", None) or getattr(cell, "text", "") or "") for cell in (row.cells or [])]
        for row in (getattr(table, "rows", []) or [])
    )
    return [row for row in rows if any(value.strip() for value in row)]


def _same_source_rows(left: list[list[str]], right: list[list[str]]) -> bool:
    if len(left) != len(right):
        return False
    return all(
        len(left_row) == len(right_row)
        and [_compact_source_value(value) for value in left_row]
        == [_compact_source_value(value) for value in right_row]
        for left_row, right_row in zip(left, right, strict=True)
    )


def _row_bbox(cell_bboxes: Any, row_index: int, fallback: Any = None) -> list[float] | None:
    rows = cell_bboxes if isinstance(cell_bboxes, list) else []
    cells = rows[row_index] if row_index < len(rows) and isinstance(rows[row_index], list) else []
    boxes = [
        [float(value) for value in bbox]
        for bbox in cells
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4
        and all(isinstance(value, (int, float)) for value in bbox)
    ]
    if boxes:
        return [
            min(box[0] for box in boxes),
            min(box[1] for box in boxes),
            max(box[2] for box in boxes),
            max(box[3] for box in boxes),
        ]
    if isinstance(fallback, (list, tuple)) and len(fallback) == 4:
        return [float(value) for value in fallback]
    return None


def _row_evidence_ids(cell_evidence_ids: Any, row_index: int) -> list[str]:
    rows = cell_evidence_ids if isinstance(cell_evidence_ids, list) else []
    cells = rows[row_index] if row_index < len(rows) and isinstance(rows[row_index], list) else []
    return list(
        dict.fromkeys(
            str(evidence_id)
            for cell in cells
            for evidence_id in (cell if isinstance(cell, list) else [])
            if str(evidence_id)
        )
    )


def _physical_quarantine_source_rows(
    parse_result: Any,
    previous_headers: list[str],
    quarantined: LogicalTable,
    rows: list[list[str]],
) -> list[dict[str, Any]]:
    """Return exact physical row provenance for a guarded quarantine tail.

    The failed logical table promoted its first transaction to ``headers``.
    Its physical table still retains both the promoted row and body rows in
    ``metadata.raw_rows`` together with geometry/evidence.  Bind only a unique,
    exact physical match; page proximity or width alone is insufficient.
    """
    target_ids = {str(value) for value in (getattr(quarantined, "source_physical_ids", []) or []) if str(value)}
    bounds = _logical_table_page_bounds(quarantined)
    candidates: list[tuple[int, Any]] = []
    for page in (getattr(parse_result, "pages", []) or []):
        page_number = int(getattr(page, "page_number", 0) or 0)
        if bounds is not None and not (bounds[0] <= page_number <= bounds[1]):
            continue
        for table in (getattr(page, "tables", []) or []):
            table_id = str(getattr(table, "table_id", "") or "")
            if target_ids and table_id not in target_ids:
                continue
            if _same_source_rows(_physical_raw_rows(table), rows):
                candidates.append((page_number, table))
    if len(candidates) != 1:
        return []

    page_number, table = candidates[0]
    table_id = str(getattr(table, "table_id", "") or "")
    metadata = getattr(table, "metadata", None) or {}
    geometry = metadata.get("geometry") if isinstance(metadata, dict) else {}
    geometry = geometry if isinstance(geometry, dict) else {}
    cell_bboxes = geometry.get("cell_bboxes")
    cell_evidence_ids = geometry.get("cell_evidence_ids")
    output: list[dict[str, Any]] = []
    for row_index, values in enumerate(rows):
        evidence_ids = _row_evidence_ids(cell_evidence_ids, row_index)
        source_cell_refs = [
            {
                "source": "canonical_physical_table",
                "page": page_number,
                "table_id": table_id,
                "row": row_index,
                "raw_row": row_index,
                "col": col_index,
            }
            for col_index, value in enumerate(values)
            if str(value or "").strip()
        ]
        source = {
            "source": "canonical_physical_table",
            "source_page": page_number,
            "page_id": f"page:{page_number:04d}",
            "page_range": [page_number, page_number],
            **({"table_id": table_id} if table_id else {}),
            "source_row_index": row_index,
            **(
                {"bbox": bbox}
                if (bbox := _row_bbox(cell_bboxes, row_index, getattr(table, "bbox", None))) is not None
                else {}
            ),
            **({"evidence_ids": evidence_ids} if evidence_ids else {}),
            **({"source_cell_refs": source_cell_refs} if source_cell_refs else {}),
        }
        output.append(
            {
                "source": source,
                "source_raw": dict(zip(previous_headers, values, strict=True)),
            }
        )
    return output


def _logical_quarantine_source_rows(
    previous_headers: list[str],
    quarantined: LogicalTable,
    rows: list[list[str]],
) -> list[dict[str, Any]]:
    """Fallback to logical-table provenance when physical geometry is unavailable."""
    bounds = _logical_table_page_bounds(quarantined)
    page = bounds[0] if bounds is not None else 0
    table_ids = [str(value) for value in (getattr(quarantined, "source_physical_ids", []) or []) if str(value)]
    table_id = table_ids[0] if len(set(table_ids)) == 1 else ""
    logical_rows = list(getattr(quarantined, "rows", []) or [])
    output: list[dict[str, Any]] = []
    for raw_index, values in enumerate(rows):
        logical_row = logical_rows[raw_index - 1] if raw_index > 0 and raw_index - 1 < len(logical_rows) else None
        source_page = int(getattr(logical_row, "source_page", 0) or page or 0)
        row_table_id = str(getattr(logical_row, "source_physical_id", "") or table_id)
        source_row_index = getattr(logical_row, "source_row_index", -1)
        try:
            source_row_index = int(source_row_index)
        except (TypeError, ValueError):
            source_row_index = -1
        if source_row_index < 0:
            source_row_index = raw_index
        source_cell_refs = [
            dict(ref)
            for ref in (getattr(logical_row, "source_cell_refs", []) or [])
            if isinstance(ref, dict)
        ]
        cells = list(getattr(logical_row, "cells", []) or [])
        evidence_ids = list(
            dict.fromkeys(
                str(evidence_id)
                for cell in cells
                for evidence_id in (getattr(cell, "evidence_ids", []) or [])
                if str(evidence_id)
            )
        )
        source = {
            "source": "canonical_table",
            "source_page": source_page,
            **({"page_id": f"page:{source_page:04d}", "page_range": [source_page, source_page]} if source_page else {}),
            **({"table_id": row_table_id} if row_table_id else {}),
            "source_row_index": source_row_index,
            **({"evidence_ids": evidence_ids} if evidence_ids else {}),
            **({"source_cell_refs": source_cell_refs} if source_cell_refs else {}),
        }
        output.append(
            {
                "source": source,
                "source_raw": dict(zip(previous_headers, values, strict=True)),
            }
        )
    return output


def _quarantine_continuation_sources(
    parse_result: Any,
    previous: LogicalTable,
    quarantined: LogicalTable,
    recovered_tables: list[list[list[str]]],
) -> list[dict[str, Any]]:
    if len(recovered_tables) != 1 or len(recovered_tables[0]) < 2:
        return []
    previous_headers = list(recovered_tables[0][0])
    rows = [list(row) for row in recovered_tables[0][1:]]
    return _physical_quarantine_source_rows(parse_result, previous_headers, quarantined, rows) or (
        _logical_quarantine_source_rows(previous_headers, quarantined, rows)
    )


_DATE_TOKEN_RE = re.compile(r"(?<!\d)(20\d{2})[-/.年]?([01]?\d)[-/.月]?([0-3]?\d)(?:日)?(?!\d)")


def _source_row_date(values: dict[str, Any]) -> str:
    for header, value in values.items():
        compact_header = _compact_source_value(header).lower()
        if not any(marker in compact_header for marker in ("日期", "时间", "date", "time")):
            continue
        for match in _DATE_TOKEN_RE.finditer(unicodedata.normalize("NFKC", str(value or ""))):
            try:
                parsed = date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
            except ValueError:
                continue
            return parsed.isoformat()
    return ""


def _attach_quarantine_sources(
    records: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach row-local tail provenance after the guarded parser run.

    Parser order is source order for a single injected matrix.  We nevertheless
    require equal cardinality and a matching source/normalized date for every row
    before binding anything, so a filtering or reordering parser cannot receive
    plausible but incorrect provenance.
    """
    if not records or len(records) != len(source_rows):
        return []
    output: list[dict[str, Any]] = []
    for record, row_source in zip(records, source_rows, strict=True):
        source_date = _source_row_date(row_source.get("source_raw") or {})
        normalized = record.get("normalized") or {}
        record_date = str(normalized.get("date") or normalized.get("timestamp") or "")[:10]
        if not source_date or record_date != source_date:
            return []
        updated = dict(record)
        updated["source"] = dict(row_source.get("source") or {})
        output.append(updated)
    return output


class BankLedgerOrchestrator:
    """Multi logical-table bank export orchestrator."""

    def __init__(self, registry: Any) -> None:
        self._registry = registry
        self._detector = BankStyleDetector()

    def _record_selection_diagnostics(self, meta: BLOMeta) -> dict[str, Any] | None:
        diagnostics = getattr(self._registry, "last_selection_diagnostics", None)
        if isinstance(diagnostics, dict) and diagnostics:
            snapshot = dict(diagnostics)
            meta.candidate_diagnostics.append(snapshot)
            return snapshot
        return None

    @staticmethod
    def _commit_winning_diagnostic(meta: BLOMeta, diagnostic: dict[str, Any] | None) -> None:
        """Put the whole-result winner first for downstream audit consumers."""
        if not diagnostic:
            return
        meta.candidate_diagnostics = [
            diagnostic,
            *(item for item in meta.candidate_diagnostics if item is not diagnostic),
        ]

    @staticmethod
    def _merge_reconstruction_meta(ctx: StyleContext, sub_ctx: StyleContext) -> None:
        """Retain the strongest row-count evidence from a logical-table run."""
        if sub_ctx.reconstruction is None:
            return
        current_expected = int(getattr(ctx.reconstruction, "expected_primary_rows", 0) or 0)
        candidate_expected = int(sub_ctx.reconstruction.expected_primary_rows or 0)
        if ctx.reconstruction is None or candidate_expected >= current_expected:
            ctx.reconstruction = replace(
                sub_ctx.reconstruction,
                expected_primary_rows=max(current_expected, candidate_expected),
            )

    def _run_document_context(
        self,
        detection: StyleDetectionResult,
        ctx: StyleContext,
        plugin: Any,
        meta: BLOMeta,
    ) -> tuple[list[dict[str, Any]], dict[str, dict], StyleContext, dict[str, Any] | None]:
        """Run existing whole-document candidates once against ``ctx.tables``."""
        if not ctx.tables:
            return [], {}, replace(ctx), None
        document_ctx = replace(ctx)
        batch, identity = self._registry.run_parser_chain(detection, document_ctx, plugin)
        diagnostic = self._record_selection_diagnostics(meta)
        return dedupe_transaction_rows(batch), identity, document_ctx, diagnostic

    @staticmethod
    def _prefer_document_result(
        logical_records: list[dict[str, Any]],
        document_records: list[dict[str, Any]],
        *,
        logical_reconstruction: Any = None,
        document_reconstruction: Any = None,
    ) -> bool:
        """Compare whole results without merging rows or fields across them."""
        if not document_records:
            return False
        if not logical_records:
            return True

        def evidence_authority(records: list[dict[str, Any]], reconstruction: Any) -> int:
            if not records or not all(is_canonical_row(record.get("normalized") or {}) for record in records):
                return 0
            source = str(getattr(reconstruction, "expected_evidence_source", "") or "")
            confidence = float(getattr(reconstruction, "expected_evidence_confidence", 0.0) or 0.0)
            expected = int(getattr(reconstruction, "expected_primary_rows", 0) or 0)
            if confidence < 0.85 or expected != len(records):
                return 0
            if source in {
                "split_footer",
                "header_total",
                "statement_header_totals",
                "cumulative_footer_total",
                "page_footer",
            }:
                return 2
            return 0

        logical_authority = evidence_authority(logical_records, logical_reconstruction)
        document_authority = evidence_authority(document_records, document_reconstruction)
        if logical_authority != document_authority:
            return document_authority > logical_authority

        def quality(records: list[dict[str, Any]]) -> tuple[int, float, int]:
            canonical = sum(is_canonical_row(record.get("normalized") or {}) for record in records)
            return canonical, canonical / max(len(records), 1), len(records)

        return quality(document_records) > quality(logical_records)

    def run(
        self,
        detection: StyleDetectionResult,
        ctx: StyleContext,
        plugin: Any,
    ) -> tuple[list[dict[str, Any]], dict[str, dict], BLOMeta]:
        blocks = _iter_parse_blocks(ctx)
        meta = BLOMeta(logical_table_count=len(blocks))

        if len(blocks) <= 1:
            sub_ctx = ctx
            sub_detection = detection
            if blocks:
                lt, sub_tables = blocks[0]
                if lt is not None and not getattr(lt, "quality_passed", True):
                    meta.tables_skipped = 1
                    if ctx.tables:
                        sub_ctx = ctx
                        sub_detection = detection
                    else:
                        return [], plugin._extract_identity(ctx.parse_result), meta
                elif sub_tables:
                    sub_ctx = replace(ctx, tables=sub_tables)
                    sub_detection = self._resolve_detection(detection, sub_ctx)
            records, identity = self._registry.run_parser_chain(sub_detection, sub_ctx, plugin)
            selection_diagnostic = self._record_selection_diagnostics(meta)
            if sub_ctx is not ctx and sub_ctx.reconstruction is not None:
                ctx.reconstruction = sub_ctx.reconstruction
            if blocks and blocks[0][0] is not None and getattr(blocks[0][0], "quality_passed", True):
                meta.tables_parsed = 1
            elif records:
                meta.tables_parsed = 1
            records = dedupe_transaction_rows(records)
            primary_proves_document = bool(
                selection_diagnostic
                and selection_diagnostic.get("completion_state") == "proven"
                and selection_diagnostic.get("deployment_mode") == "lazy_primary"
            )
            if sub_ctx is not ctx and not primary_proves_document:
                document_records, document_identity, document_ctx, document_diagnostic = self._run_document_context(
                    detection, ctx, plugin, meta
                )
                if self._prefer_document_result(
                    records,
                    document_records,
                    logical_reconstruction=sub_ctx.reconstruction,
                    document_reconstruction=document_ctx.reconstruction,
                ):
                    records = document_records
                    ctx.reconstruction = document_ctx.reconstruction
                    self._commit_winning_diagnostic(meta, document_diagnostic)
                    if document_identity:
                        identity = document_identity
            return records, identity, meta

        records: list[dict[str, Any]] = []
        identity_fields = plugin._extract_identity(ctx.parse_result)
        previous_passed_lt: LogicalTable | None = None

        for lt, sub_tables in blocks:
            recovered_quarantine = False
            recovered_sources: list[dict[str, Any]] = []
            if lt is not None and not getattr(lt, "quality_passed", True):
                recovered_tables = _merge_quarantine_continuation_matrices(previous_passed_lt, lt)
                if recovered_tables:
                    sub_tables = recovered_tables
                    recovered_quarantine = True
                    recovered_sources = _quarantine_continuation_sources(
                        ctx.parse_result,
                        previous_passed_lt,
                        lt,
                        recovered_tables,
                    )
                    logger.info(
                        "[BLO] recover adjacent merge_quarantine logical_table=%s rows=%d",
                        getattr(lt, "logical_id", None) or getattr(lt, "table_id", ""),
                        max(len(recovered_tables[0]) - 1, 0),
                    )
                else:
                    meta.tables_skipped += 1
                    logger.info(
                        "[BLO] skip logical_table=%s reason=%s",
                        getattr(lt, "logical_id", None) or getattr(lt, "table_id", ""),
                        getattr(lt, "quality_skip_reason", "ltqg_failed"),
                    )
                    continue
            if not sub_tables:
                meta.tables_skipped += 1
                continue

            sub_ctx = replace(
                ctx,
                tables=sub_tables,
                # The guarded matrix deliberately replaces a failed logical
                # table's promoted-header interpretation.  Without this flag,
                # grid_standard re-enters the original ParseResult and returns
                # the preceding passed plane again, silently dropping the tail.
                prefer_context_tables=True if recovered_quarantine else ctx.prefer_context_tables,
            )
            sub_detection = self._resolve_detection(detection, sub_ctx)
            batch, batch_identity = self._registry.run_parser_chain(sub_detection, sub_ctx, plugin)
            if recovered_quarantine:
                batch = _attach_quarantine_sources(batch, recovered_sources)
                if not batch:
                    meta.tables_skipped += 1
                    logger.info(
                        "[BLO] reject merge_quarantine tail without exact row-local provenance logical_table=%s",
                        getattr(lt, "logical_id", None) or getattr(lt, "table_id", ""),
                    )
                    continue
            self._record_selection_diagnostics(meta)
            self._merge_reconstruction_meta(ctx, sub_ctx)
            if batch_identity:
                identity_fields = batch_identity
            records.extend(batch)
            meta.tables_parsed += 1
            if lt is not None and getattr(lt, "quality_passed", True):
                previous_passed_lt = lt

        document_was_run = False
        document_ctx: StyleContext | None = None
        if not records and ctx.tables:
            logger.info("[BLO] no records from logical tables — fallback to ctx.tables")
            batch, batch_identity, document_ctx, document_diagnostic = self._run_document_context(
                detection, ctx, plugin, meta
            )
            document_was_run = True
            if batch_identity:
                identity_fields = batch_identity
            records.extend(batch)
            if batch:
                meta.tables_parsed = max(meta.tables_parsed, 1)
                self._commit_winning_diagnostic(meta, document_diagnostic)

        records = dedupe_transaction_rows(records)
        if records and ctx.tables and not document_was_run:
            document_records, document_identity, document_ctx, document_diagnostic = self._run_document_context(
                detection, ctx, plugin, meta
            )
            if self._prefer_document_result(
                records,
                document_records,
                logical_reconstruction=ctx.reconstruction,
                document_reconstruction=document_ctx.reconstruction,
            ):
                logger.info(
                    "[BLO] whole-document candidate supersedes sparse logical tables logical=%d document=%d",
                    len(records),
                    len(document_records),
                )
                records = document_records
                ctx.reconstruction = document_ctx.reconstruction
                self._commit_winning_diagnostic(meta, document_diagnostic)
                if document_identity:
                    identity_fields = document_identity
                meta.tables_parsed = max(meta.tables_parsed, 1)
        elif document_was_run and document_ctx is not None:
            ctx.reconstruction = document_ctx.reconstruction
        logger.info(
            "[BLO] parsed=%d skipped=%d records=%d",
            meta.tables_parsed,
            meta.tables_skipped,
            len(records),
        )
        return records, identity_fields, meta

    def _resolve_detection(
        self,
        document_detection: StyleDetectionResult,
        sub_ctx: StyleContext,
    ) -> StyleDetectionResult:
        sub_detection = self._detector.detect(sub_ctx)
        if sub_detection.confidence >= _INHERIT_CONFIDENCE:
            return sub_detection
        return document_detection


__all__ = [
    "BLOMeta",
    "BankLedgerOrchestrator",
    "logical_table_to_matrices",
]
