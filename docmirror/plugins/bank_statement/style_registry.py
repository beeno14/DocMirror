# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Bank statement style parser registry and fallback orchestration.

Maps detected style IDs to parser modules under ``bank_statement.styles``,
runs the primary parser chain, scores record completeness with CAPS coverage,
and falls back to ``grid_standard`` / ``borderless_ocr`` when primary is sparse.

Pipeline role: plugin-local dispatch between ``BankStyleDetector`` and record
builders inside the post-seal bank-statement projector.

Key exports: ``BankStyleParserRegistry``.

Dependencies: ``bank_statement.styles.*``, ``bank_statement.canonical``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, replace
from typing import Any, Callable

from docmirror.plugins.bank_statement.canonical import ensure_canonical_normalized, records_from_raw_transactions
from docmirror.plugins.bank_statement.canonical_quality import (
    canonical_expected_from_parse_result,
    is_canonical_row,
    physical_transaction_row_estimate,
)
from docmirror.plugins.bank_statement.context import StyleContext, collect_physical_tables_from_parse_result
from docmirror.plugins.bank_statement.evidence_atom_table_recovery import (
    recover_evidence_atom_bank_tables,
    recover_positioned_record_block_bank_tables,
    recovered_evidence_atom_expected_row_count,
    recovered_evidence_atom_expected_row_evidence,
    recovered_evidence_atom_row_sources,
)
from docmirror.plugins.bank_statement.ocr_implicit_table_recovery import (
    recover_ocr_implicit_ledger_tables,
    recovered_ocr_implicit_row_count,
)
from docmirror.plugins.bank_statement.style_detector import StyleDetectionResult
from docmirror.plugins.bank_statement.styles import (
    borderless_ocr,
    compact_merged,
    grid_standard,
    kv_identity,
    signed_amount,
)
from docmirror.plugins.bank_statement.text_table_builder import build_tables_from_stacked_bank_text
from docmirror.plugins.bank_statement.wide_table_recovery import (
    SOURCE_REPORTED_ROW_COUNT_SOURCES,
    RowCountEvidence,
    count_expected_rows_from_bank_footer,
    page_texts_from_parse_result,
    recover_wide_bank_tables,
    resolve_row_count_evidence,
)

logger = logging.getLogger(__name__)

_PARSERS = {
    "compact_merged": compact_merged,
    "grid_standard": grid_standard,
    "kv_identity": kv_identity,
    "signed_amount": signed_amount,
    "borderless_ocr": borderless_ocr,
}

_FALLBACK_PARSER_IDS = ("grid_standard", "borderless_ocr", "signed_amount", "compact_merged")
_CAPS_THRESHOLD = 0.55
_COVERAGE_THRESHOLD = 0.80
_INDEPENDENT_ROW_COUNT_SOURCES = frozenset(
    {"split_footer", "header_total", "page_footer", "page_transaction_anchors", "physical_rows"}
)
_SOURCE_DATE_RE = re.compile(r"20\d{2}(?:[-/.]?\d{1,2}){2}")
_SOURCE_DATE_HEADERS = frozenset(
    {
        "date",
        "datetime",
        "timestamp",
        "日期",
        "交易日期",
        "交易时间",
        "记账日期",
        "记账时间",
    }
)


def _field_completeness(records: list[dict[str, Any]], sample: int = 8) -> float:
    if not records:
        return 0.0
    fields = ("date", "amount", "direction", "balance")
    scores = []
    for rec in records[:sample]:
        norm = rec.get("normalized") or {}
        scores.append(sum(1 for f in fields if norm.get(f) not in (None, "", 0)) / len(fields))
    return sum(scores) / len(scores)


def _batch_field_completeness(
    transactions: list[dict[str, str]],
    normalize_fn: Any,
    plugin: Any,
) -> float:
    if not transactions:
        return 0.0
    nf = normalize_fn or plugin._normalize
    balance_expected = any(
        any(
            "余额" in grid_standard.normalize_header_cell(str(key)) or "balance" in str(key).lower()
            for key in transaction
        )
        for transaction in transactions[:8]
    )
    fields = ("date", "amount", "direction", *(("balance",) if balance_expected else ()))
    scores = []
    for txn in transactions[:8]:
        norm = ensure_canonical_normalized(nf(txn), plugin.standard_fields)
        scores.append(sum(1 for f in fields if norm.get(f) not in (None, "", 0)) / len(fields))
    return sum(scores) / len(scores)


def _batch_raw_width(transactions: list[dict[str, str]], sample: int = 8) -> float:
    """Average number of populated source columns, used only as a tie-breaker."""
    if not transactions:
        return 0.0
    widths = [
        sum(bool(str(value or "").strip()) for key, value in transaction.items() if not key.startswith("_"))
        for transaction in transactions[:sample]
    ]
    return sum(widths) / len(widths)


def _batch_source_coverage(transactions: list[dict[str, Any]]) -> float:
    if not transactions:
        return 0.0
    sourced = 0
    for transaction in transactions:
        source = transaction.get("_source")
        if not isinstance(source, dict):
            continue
        source_page = source.get("source_page")
        page_range = source.get("page_range")
        if source_page not in (None, "", 0) and isinstance(page_range, (list, tuple)) and page_range:
            sourced += 1
    return sourced / len(transactions)


def _parser_score(
    transactions: list[dict[str, str]],
    normalize_fn: Any,
    plugin: Any,
    expected_rows: int,
) -> tuple[float, float]:
    if not transactions:
        return 0.0, 0.0
    nf = normalize_fn or plugin._normalize
    canonical_count = sum(
        is_canonical_row(ensure_canonical_normalized(nf(transaction), plugin.standard_fields))
        for transaction in transactions
    )
    expected = max(expected_rows if expected_rows > 0 else len(transactions), 1)
    coverage = min(canonical_count / expected, 1.0)
    completeness = _batch_field_completeness(transactions, normalize_fn, plugin)
    source_coverage = _batch_source_coverage(transactions)
    raw_width = min(_batch_raw_width(transactions) / 8.0, 1.0)
    score = 0.60 * coverage + 0.20 * completeness + 0.15 * source_coverage + 0.05 * raw_width
    return score, coverage


def _expected_rows(ctx: StyleContext) -> int:
    footer_expected = count_expected_rows_from_bank_footer(
        ctx.full_text,
        page_texts=page_texts_from_parse_result(ctx.parse_result),
    )
    if footer_expected > 0:
        return footer_expected
    canonical_expected = canonical_expected_from_parse_result(ctx.parse_result)
    ocr_expected = recovered_ocr_implicit_row_count(ctx.parse_result)
    if canonical_expected > 0 or ocr_expected > 0:
        return max(canonical_expected, ocr_expected)
    candidates: list[int] = []
    if ctx.parse_result is not None:
        from docmirror.evidence.spe_consumer import mirror_expected_primary_rows, read_structure_spe

        candidates.append(mirror_expected_primary_rows(ctx.parse_result, read_structure_spe(ctx.parse_result)))
    if (
        ctx.reconstruction
        and ctx.reconstruction.source in {"canonical_table", "canonical_evidence_table"}
        and ctx.reconstruction.expected_primary_rows > 0
    ):
        candidates.append(ctx.reconstruction.expected_primary_rows)
    return max((int(candidate) for candidate in candidates if int(candidate or 0) > 0), default=0)


def _run_parser(parser_id: str, ctx: StyleContext, plugin: Any) -> tuple[list[dict[str, Any]], Any]:
    if parser_id == "compact_merged":
        batch = compact_merged.extract_transactions(ctx.tables)
        if batch:
            return batch, compact_merged.normalize_record
    if parser_id == "grid_standard":
        batch = grid_standard.extract_transactions(ctx, plugin)
        return batch, lambda raw: grid_standard.normalize_record(raw, plugin)
    if parser_id == "signed_amount":
        batch = signed_amount.extract_transactions(ctx, plugin)
        return batch, lambda raw: signed_amount.normalize_record(raw, plugin)
    if parser_id == "borderless_ocr":
        batch = borderless_ocr.extract_transactions(ctx, plugin)
        return batch, lambda raw: borderless_ocr.normalize_record(raw, plugin)
    return [], None


def _row_value_signature(values: Any) -> tuple[str, ...]:
    if isinstance(values, dict):
        source_values = [value for key, value in values.items() if not str(key).startswith("_")]
    elif isinstance(values, (list, tuple)):
        source_values = list(values)
    else:
        return ()
    normalized = [re.sub(r"\s+", "", str(value or "")).replace(",", "") for value in source_values]
    return tuple(sorted(value for value in normalized if value))


def _attach_recovered_sources(
    transactions: list[dict[str, Any]],
    row_sources: list[dict[str, Any]],
) -> None:
    """Attach positioned row provenance without changing the recovered table contract."""
    if not transactions or not row_sources:
        return
    source_queues: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for source in row_sources:
        signature = _row_value_signature(source.get("row_values") or [])
        if signature:
            source_queues.setdefault(signature, []).append(source)

    unmatched: list[int] = []
    used_source_ids: set[int] = set()
    for index, transaction in enumerate(transactions):
        queue = source_queues.get(_row_value_signature(transaction)) or []
        source = queue.pop(0) if queue else None
        if source is None:
            unmatched.append(index)
            continue
        used_source_ids.add(id(source))
        transaction["_source"] = _single_page_source(source)
        if isinstance(source.get("source_raw"), dict):
            transaction["_source_raw"] = dict(source["source_raw"])

    remaining_sources = [source for source in row_sources if id(source) not in used_source_ids]
    if len(unmatched) != len(remaining_sources):
        return
    for index, source in zip(unmatched, remaining_sources):
        transactions[index]["_source"] = _single_page_source(source)
        if isinstance(source.get("source_raw"), dict):
            transactions[index]["_source_raw"] = dict(source["source_raw"])


def _single_page_source(source: dict[str, Any]) -> dict[str, Any]:
    """Normalize recovered page metadata before it enters canonical records."""
    normalized = {key: value for key, value in source.items() if key not in {"row_values", "source_raw"}}
    try:
        source_page = int(normalized.get("source_page") or normalized.get("page") or 0)
    except (TypeError, ValueError):
        source_page = 0
    if source_page > 0:
        normalized["source_page"] = source_page
        normalized.setdefault("page_range", [source_page, source_page])
    return normalized


def _semantic_text_table_candidates(full_text: str) -> list[list[list[str]]]:
    """Return generic text-reconstructed ledger table candidates."""
    candidates: list[list[list[str]]] = []
    seen: set[tuple[tuple[str, ...], ...]] = set()
    for tables in (
        _solve_split_debit_credit_tables(full_text),
        build_tables_from_stacked_bank_text(full_text),
    ):
        for table in tables:
            signature = tuple(tuple(cell.strip() for cell in row) for row in table)
            if signature in seen:
                continue
            seen.add(signature)
            candidates.append(table)
    return candidates


@dataclass
class BankTableCandidate:
    """A plugin-local normalized bank-table recovery candidate."""

    candidate_id: str
    records: list[dict[str, Any]]
    source: str
    canonical_rows: int
    directional_rows: int
    source_page_rows: int
    expected_rows: RowCountEvidence | None
    balance_chain_score: float
    field_completeness: float
    score: float
    normalize_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    canonical_coverage: float = 0.0
    source_page_coverage: float = 0.0
    extraction_confidence: float = 0.0
    source_column_width: float = 0.0
    sequence_continuity: float = 0.0
    native_cell_coverage: float = 0.0


def _candidate_expected_rows(
    source_evidence: RowCountEvidence,
    *,
    count: int = 0,
    source: str = "candidate_rows",
    confidence: float = 0.55,
) -> RowCountEvidence | None:
    if (
        source_evidence.count > 0
        and source_evidence.confidence >= 0.85
        and (
            source_evidence.source in SOURCE_REPORTED_ROW_COUNT_SOURCES
            or count <= 0
            or source_evidence.count >= count
        )
    ):
        return source_evidence
    if count > 0:
        return RowCountEvidence(count=count, source=source, confidence=confidence)
    return source_evidence if source_evidence.count > 0 else None


def _evidence_atom_expected_rows(
    parse_result: Any,
    atom_tables: list[list[list[str]]],
) -> RowCountEvidence | None:
    """Return independent page-anchor evidence when recovery can prove it."""
    table_rows = sum(max(len(table) - 1, 0) for table in atom_tables)
    evidence_count, evidence_source, evidence_confidence = recovered_evidence_atom_expected_row_evidence(parse_result)
    expected = max(table_rows, evidence_count)
    if expected <= 0:
        return None
    if evidence_count == expected and evidence_source:
        return RowCountEvidence(
            count=expected,
            source=evidence_source,
            confidence=evidence_confidence,
        )
    return RowCountEvidence(count=expected, source="candidate_rows", confidence=0.55)


def _candidate_balance_chain_score(normalized_rows: list[dict[str, Any]]) -> float:
    """Return the better forward/reverse balance-chain agreement ratio."""

    def score(rows: list[dict[str, Any]]) -> float:
        comparisons = 0
        matches = 0
        for previous, current in zip(rows, rows[1:]):
            try:
                previous_balance = float(previous.get("balance"))
                current_balance = float(current.get("balance"))
                amount = abs(float(current.get("amount")))
            except (TypeError, ValueError):
                continue
            direction = str(current.get("direction") or "")
            if direction not in {"income", "expense"} or amount <= 0:
                continue
            comparisons += 1
            expected = previous_balance + amount if direction == "income" else previous_balance - amount
            if abs(round(expected - current_balance, 2)) <= 0.05:
                matches += 1
        return matches / comparisons if comparisons else 0.5

    return max(score(normalized_rows), score(list(reversed(normalized_rows))))


def _candidate_sequence_continuity(normalized_rows: list[dict[str, Any]]) -> float:
    """Return source-order sequence coverage, uniqueness, and continuity."""
    numbers: list[int] = []
    for row in normalized_rows:
        value = str(row.get("sequence_no") or "").strip()
        if re.fullmatch(r"\d{1,9}", value):
            numbers.append(int(value))
    if len(numbers) < 2:
        return 0.0
    coverage = len(numbers) / max(len(normalized_rows), 1)
    uniqueness = len(set(numbers)) / len(numbers)
    continuity = sum(abs(current - previous) == 1 for previous, current in zip(numbers, numbers[1:])) / (
        len(numbers) - 1
    )
    return coverage * uniqueness * continuity


def _candidate_source_page_coverage(transactions: list[dict[str, Any]]) -> float:
    if not transactions:
        return 0.0
    sourced = 0
    for transaction in transactions:
        source = transaction.get("_source")
        if not isinstance(source, dict):
            continue
        source_page = source.get("source_page")
        page_range = source.get("page_range")
        if source_page not in (None, "", 0) and isinstance(page_range, (list, tuple)) and len(page_range) == 2:
            if page_range[0] == page_range[1] == source_page:
                sourced += 1
    return sourced / len(transactions)


def _candidate_native_cell_coverage(transactions: list[dict[str, Any]]) -> float:
    """Return coverage by native PDF grid rows with explicit physical cell bounds."""
    if not transactions:
        return 0.0
    sourced = 0
    for transaction in transactions:
        source = transaction.get("_source")
        if not isinstance(source, dict):
            continue
        table_id = str(source.get("table_id") or source.get("source_table_id") or "")
        bbox = source.get("bbox")
        if not table_id.startswith("native:") or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            x0, y0, x1, y1 = (float(value) for value in bbox)
        except (TypeError, ValueError):
            continue
        if x1 > x0 and y1 > y0:
            sourced += 1
    return sourced / len(transactions)


def _candidate_from_batch(
    *,
    candidate_id: str,
    transactions: list[dict[str, Any]],
    normalize_fn: Callable[[dict[str, Any]], dict[str, Any]] | None,
    plugin: Any,
    source: str,
    expected_rows: RowCountEvidence | None,
    extraction_confidence: float,
) -> BankTableCandidate:
    normalizer = normalize_fn or plugin._normalize
    normalized_rows: list[dict[str, Any]] = []
    canonical_rows = 0
    directional_rows = 0
    for transaction in transactions:
        try:
            normalized = ensure_canonical_normalized(normalizer(transaction), plugin.standard_fields)
        except Exception:
            normalized = {}
        normalized_rows.append(normalized)
        if normalized.get("direction") in {"income", "expense"}:
            directional_rows += 1
        if is_canonical_row(normalized) and not _source_row_contains_multiple_transaction_dates(transaction):
            canonical_rows += 1
    expected_count = int(expected_rows.count or 0) if expected_rows is not None else 0
    if expected_count > 0 and (expected_rows is not None and expected_rows.confidence >= 0.85):
        canonical_coverage = min(canonical_rows / expected_count, 1.0)
    else:
        canonical_coverage = canonical_rows / max(len(transactions), 1)
    field_completeness = _batch_field_completeness(transactions, normalize_fn, plugin)
    source_column_width = _batch_raw_width(transactions)
    source_page_coverage = _candidate_source_page_coverage(transactions)
    native_cell_coverage = _candidate_native_cell_coverage(transactions)
    balance_chain_score = _candidate_balance_chain_score(normalized_rows)
    sequence_continuity = _candidate_sequence_continuity(normalized_rows)
    score = (
        0.35 * canonical_coverage
        + 0.25 * source_page_coverage
        + 0.20 * field_completeness
        + 0.15 * balance_chain_score
        + 0.05 * extraction_confidence
    )
    return BankTableCandidate(
        candidate_id=candidate_id,
        records=transactions,
        source=source,
        canonical_rows=canonical_rows,
        directional_rows=directional_rows,
        source_page_rows=round(source_page_coverage * len(transactions)),
        expected_rows=expected_rows,
        balance_chain_score=balance_chain_score,
        field_completeness=field_completeness,
        score=score,
        normalize_fn=normalize_fn,
        canonical_coverage=canonical_coverage,
        source_page_coverage=source_page_coverage,
        extraction_confidence=extraction_confidence,
        source_column_width=source_column_width,
        sequence_continuity=sequence_continuity,
        native_cell_coverage=native_cell_coverage,
    )


def _source_row_contains_multiple_transaction_dates(transaction: dict[str, Any]) -> bool:
    """Reject a column-aggregated page masquerading as one transaction row.

    A genuine transaction may expose both a date and a timestamp column, so
    occurrences are checked per source cell. Two complete dates inside one
    date-like cell prove that physical row boundaries were lost.
    """
    for raw_header, value in transaction.items():
        if str(raw_header).startswith("_"):
            continue
        header = re.sub(r"\s+", "", str(raw_header or "")).lower()
        normalized_header = re.sub(r"\s+", "", grid_standard.normalize_header_cell(str(raw_header or ""))).lower()
        if header not in _SOURCE_DATE_HEADERS and normalized_header not in _SOURCE_DATE_HEADERS:
            continue
        if len(_SOURCE_DATE_RE.findall(str(value or ""))) > 1:
            return True
    return False


def _candidate_reliable_count_coverage(candidate: BankTableCandidate) -> float | None:
    """Return exact-count agreement only when the candidate evidence is reliable."""
    evidence = candidate.expected_rows
    if (
        evidence is None
        or evidence.count <= 0
        or evidence.confidence < 0.85
        or evidence.source not in _INDEPENDENT_ROW_COUNT_SOURCES
    ):
        return None
    ratio = candidate.canonical_rows / evidence.count
    return max(0.0, 1.0 - abs(1.0 - ratio))


def _has_not_lower_reliable_count(candidate: BankTableCandidate, reference: BankTableCandidate) -> bool:
    reference_coverage = _candidate_reliable_count_coverage(reference)
    if reference_coverage is None:
        return True
    candidate_coverage = _candidate_reliable_count_coverage(candidate)
    return candidate_coverage is not None and candidate_coverage >= reference_coverage - 0.01


def _select_candidate(
    candidates: list[BankTableCandidate],
    *,
    native_text_suspicious: bool = False,
) -> tuple[BankTableCandidate | None, dict[str, Any]]:
    """Select a viable candidate without allowing raw row count to dominate quality."""
    viable = [candidate for candidate in candidates if candidate.canonical_rows > 0]
    candidate_counts = {candidate.candidate_id: len(candidate.records) for candidate in candidates}
    if not viable:
        return None, {
            "selected_candidate": "",
            "candidate_counts": candidate_counts,
            "selection_reason": "no_candidate_has_canonical_rows",
        }

    def selection_key(candidate: BankTableCandidate) -> tuple[float, ...]:
        reliable_coverage = _candidate_reliable_count_coverage(candidate)
        return (
            1.0 if reliable_coverage is not None else 0.0,
            reliable_coverage if reliable_coverage is not None else 0.0,
            candidate.canonical_coverage,
            candidate.sequence_continuity,
            candidate.source_page_coverage,
            candidate.field_completeness,
            candidate.balance_chain_score,
            candidate.extraction_confidence,
            candidate.score,
        )

    # This lexicographic order is intentional.  In particular, a candidate
    # cannot win merely because it has more recovered rows: it must first
    # satisfy reliable count evidence and canonical validation.
    selected = max(viable, key=selection_key)
    selection_note = ""
    source_faithful = [
        candidate
        for candidate in viable
        if candidate.source_column_width > selected.source_column_width + 0.25
        and candidate.canonical_rows >= selected.canonical_rows
        and candidate.canonical_coverage >= selected.canonical_coverage - 0.01
        and candidate.source_page_coverage >= selected.source_page_coverage - 0.01
        and candidate.field_completeness >= selected.field_completeness - 0.01
        and candidate.balance_chain_score >= selected.balance_chain_score - 0.01
    ]
    if source_faithful:
        # A recovery may improve record boundaries, but it must not discard
        # source columns from an equally complete native candidate.
        selected = max(
            source_faithful,
            key=lambda candidate: (
                candidate.source_column_width,
                candidate.candidate_id == "legacy_primary",
                selection_key(candidate),
            ),
        )
    if native_text_suspicious and (
        selected.native_cell_coverage >= 0.99 or selected.source == "native_wide_table"
    ):
        ocr_candidates = [
            candidate
            for candidate in viable
            if candidate.candidate_id == "ocr_implicit_table"
            and candidate.canonical_rows >= selected.canonical_rows
            and candidate.canonical_coverage >= selected.canonical_coverage - 0.01
            and candidate.source_page_coverage >= selected.source_page_coverage - 0.01
            and candidate.field_completeness >= selected.field_completeness - 0.01
            and candidate.balance_chain_score >= selected.balance_chain_score - 0.01
            and _has_not_lower_reliable_count(candidate, selected)
        ]
        if ocr_candidates:
            selected = max(ocr_candidates, key=selection_key)
            selection_note = "native_text_suspicious:ocr_not_lower_quality:"
    native_grid_candidates = [
        candidate
        for candidate in viable
        if candidate.candidate_id == "native_wide_table"
        and candidate.native_cell_coverage >= 0.99
        and len(candidate.records) >= len(selected.records)
        and candidate.canonical_rows >= selected.canonical_rows
        and candidate.canonical_coverage >= selected.canonical_coverage - 0.01
        and candidate.source_page_coverage >= selected.source_page_coverage - 0.01
        and candidate.field_completeness >= selected.field_completeness - 0.01
        and candidate.balance_chain_score >= selected.balance_chain_score - 0.01
        and candidate.source_column_width >= selected.source_column_width - 0.25
        and _has_not_lower_reliable_count(candidate, selected)
    ]
    if native_grid_candidates and not native_text_suspicious:
        # When every semantic gate is equal, explicit native cell boundaries are
        # stronger evidence than a text reconstruction that may shift wrapped
        # fragments into the following transaction.
        selected = max(
            native_grid_candidates, key=lambda candidate: (candidate.native_cell_coverage, selection_key(candidate))
        )
    legacy = next((candidate for candidate in viable if candidate.candidate_id == "legacy_primary"), None)
    if (
        legacy is not None
        and selected is not legacy
        and _candidate_reliable_count_coverage(selected) is None
        and len(selected.records) < len(legacy.records)
        and selected.canonical_rows <= legacy.canonical_rows
    ):
        # A low-confidence recovery cannot manufacture a denominator from its
        # own smaller result and use it to replace an equally canonical source
        # table.  Keep the fuller established candidate until independent row
        # count evidence proves the smaller result is complete.
        selected = legacy
    reason = (
        f"{selection_note}score={selected.score:.3f}:canonical_coverage={selected.canonical_coverage:.3f}:"
        f"sequence_continuity={selected.sequence_continuity:.3f}:"
        f"source_coverage={selected.source_page_coverage:.3f}:"
        f"balance_chain={selected.balance_chain_score:.3f}"
    )
    return selected, {
        "selected_candidate": selected.candidate_id,
        "candidate_counts": candidate_counts,
        "selection_reason": reason,
    }


def _has_suspicious_native_text(parse_result: Any) -> bool:
    parser_info = getattr(parse_result, "parser_info", None)
    warnings = getattr(parser_info, "warnings", ()) or ()
    return "native_text_glyph_mapping_suspected" in warnings


def _collect_table_candidates(
    detection: StyleDetectionResult,
    ctx: StyleContext,
    plugin: Any,
    *,
    legacy_transactions: list[dict[str, Any]],
    legacy_normalize_fn: Callable[[dict[str, Any]], dict[str, Any]] | None,
) -> list[BankTableCandidate]:
    """Materialize every bank recovery source before selecting one candidate."""
    source_evidence = resolve_row_count_evidence(
        ctx.full_text,
        page_texts=page_texts_from_parse_result(ctx.parse_result),
    )
    atom_tables = recover_evidence_atom_bank_tables(ctx.parse_result)
    atom_row_evidence = _evidence_atom_expected_rows(ctx.parse_result, atom_tables)
    atom_expected = atom_row_evidence.count if atom_row_evidence is not None else 0
    if source_evidence.count <= 0 and atom_row_evidence is not None:
        source_evidence = atom_row_evidence
    candidates: list[BankTableCandidate] = []

    def add(
        candidate_id: str,
        batch: list[dict[str, Any]],
        normalize_fn: Callable[[dict[str, Any]], dict[str, Any]] | None,
        source: str,
        expected: RowCountEvidence | None = None,
        confidence: float = 0.7,
    ) -> None:
        if not batch:
            return
        candidates.append(
            _candidate_from_batch(
                candidate_id=candidate_id,
                transactions=batch,
                normalize_fn=normalize_fn,
                plugin=plugin,
                source=source,
                expected_rows=expected or _candidate_expected_rows(source_evidence),
                extraction_confidence=confidence,
            )
        )

    add(
        "legacy_primary",
        legacy_transactions,
        legacy_normalize_fn,
        (ctx.reconstruction.source if ctx.reconstruction is not None else "canonical_table"),
        _candidate_expected_rows(
            source_evidence,
            count=len(legacy_transactions),
        ),
        confidence=0.8,
    )
    for parser_id in dict.fromkeys([*detection.parser_chain, *_FALLBACK_PARSER_IDS]):
        if parser_id == "kv_identity" or parser_id not in _PARSERS:
            continue
        batch, normalize_fn = _run_parser(parser_id, ctx, plugin)
        add(f"parser:{parser_id}", batch, normalize_fn, "canonical_table", confidence=0.8)

    if detection.primary_style == "compact_merged_ledger":
        add(
            "compact_merged",
            compact_merged.extract_transactions(ctx.tables),
            compact_merged.normalize_record,
            "compact_merged",
            confidence=0.75,
        )

    for index, semantic_table in enumerate(_semantic_text_table_candidates(ctx.full_text)):
        semantic_ctx = replace(ctx, tables=[semantic_table], prefer_context_tables=True)
        batch, normalize_fn = _run_parser("grid_standard", semantic_ctx, plugin)
        add(
            f"semantic_text:{index}",
            batch,
            normalize_fn,
            "semantic_text_table",
            _candidate_expected_rows(
                source_evidence,
                count=max(len(semantic_table) - 1, 0),
                source="semantic_text_rows",
                confidence=0.60,
            ),
            confidence=0.65,
        )

    physical_tables = collect_physical_tables_from_parse_result(ctx.parse_result)
    physical_expected = physical_transaction_row_estimate(ctx.parse_result)
    if physical_tables:
        physical_ctx = replace(ctx, tables=physical_tables, prefer_context_tables=True)
        batch, normalize_fn = _run_parser("grid_standard", physical_ctx, plugin)
        add(
            "physical_table",
            batch,
            normalize_fn,
            "canonical_physical_tables",
            _candidate_expected_rows(
                source_evidence,
                count=physical_expected,
                source="physical_rows",
                confidence=0.55,
            ),
            confidence=0.85,
        )

    positioned = recover_positioned_record_block_bank_tables(ctx.parse_result)
    if positioned.tables:
        positioned_ctx = replace(ctx, tables=positioned.tables, prefer_context_tables=True)
        batch, normalize_fn = _run_parser("grid_standard", positioned_ctx, plugin)
        _attach_recovered_sources(batch, positioned.row_sources)
        add(
            "positioned_record_block",
            batch,
            normalize_fn,
            "positioned_record_block",
            _candidate_expected_rows(
                source_evidence,
                count=positioned.expected_rows,
                source="positioned_record_blocks",
                confidence=0.95,
            ),
            confidence=0.95,
        )

    if atom_tables:
        atom_ctx = replace(ctx, tables=atom_tables, prefer_context_tables=True)
        batch, normalize_fn = _run_parser("grid_standard", atom_ctx, plugin)
        _attach_recovered_sources(batch, recovered_evidence_atom_row_sources(ctx.parse_result))
        add(
            "evidence_atom",
            batch,
            normalize_fn,
            "canonical_evidence_table",
            _candidate_expected_rows(
                source_evidence,
                count=atom_expected,
                source=(atom_row_evidence.source if atom_row_evidence is not None else "candidate_rows"),
                confidence=(atom_row_evidence.confidence if atom_row_evidence is not None else 0.55),
            ),
            confidence=0.90,
        )

    wide_tables = recover_wide_bank_tables(ctx.parse_result, ctx.full_text)
    if wide_tables:
        wide_ctx = replace(ctx, tables=wide_tables, prefer_context_tables=True)
        wide_parser_id = "signed_amount" if detection.primary_style == "signed_amount" else "grid_standard"
        batch, normalize_fn = _run_parser(wide_parser_id, wide_ctx, plugin)
        add(
            "native_wide_table",
            batch,
            normalize_fn,
            "native_wide_table",
            _candidate_expected_rows(
                source_evidence,
                count=len(batch),
                source="native_wide_rows",
                confidence=0.70,
            ),
            confidence=0.85,
        )

    ocr_tables = recover_ocr_implicit_ledger_tables(ctx.parse_result, ctx.full_text)
    if ocr_tables:
        ocr_ctx = replace(ctx, tables=ocr_tables, prefer_context_tables=True)
        batch, normalize_fn = _run_parser("grid_standard", ocr_ctx, plugin)
        recovered_count = sum(max(len(table) - 1, 0) for table in ocr_tables)
        positioned_rows_complete = (
            len(batch) == recovered_count and _candidate_source_page_coverage(batch) >= 0.99
        )
        add(
            "ocr_implicit_table",
            batch,
            normalize_fn,
            "ocr_implicit_table",
            _candidate_expected_rows(
                source_evidence,
                count=recovered_count,
                source="positioned_date_anchors" if positioned_rows_complete else "ocr_recovered_rows",
                confidence=0.95 if positioned_rows_complete else 0.70,
            ),
            confidence=0.70,
        )
    return candidates


class BankStyleParserRegistry:
    """Execute parser_chain and produce v2.0 records."""

    def __init__(self) -> None:
        self.last_selection_diagnostics: dict[str, Any] = {}

    def run(
        self,
        detection: StyleDetectionResult,
        ctx: StyleContext,
        plugin: Any,
    ) -> tuple[list[dict[str, Any]], dict[str, dict]]:
        return self.run_parser_chain(detection, ctx, plugin)

    def run_parser_chain(
        self,
        detection: StyleDetectionResult,
        ctx: StyleContext,
        plugin: Any,
    ) -> tuple[list[dict[str, Any]], dict[str, dict]]:
        identity_fields = plugin._extract_identity(ctx.parse_result)
        transactions: list[dict[str, Any]] = []
        normalize_fn = None
        expected = _expected_rows(ctx)

        for parser_id in detection.parser_chain:
            module = _PARSERS.get(parser_id)
            if module is None:
                logger.warning("[BankStyleRegistry] unknown parser: %s", parser_id)
                continue

            if parser_id == "kv_identity":
                identity_fields = kv_identity.enrich_identity_fields(
                    ctx,
                    identity_fields,
                    plugin.identity_fields,
                )
                continue

            batch, norm = _run_parser(parser_id, ctx, plugin)
            if batch:
                transactions = batch
                normalize_fn = norm
                continue

        if not transactions and detection.primary_style == "compact_merged_ledger":
            transactions = compact_merged.extract_transactions(ctx.tables)
            normalize_fn = compact_merged.normalize_record

        if not transactions:
            stacked_tables = _semantic_text_table_candidates(ctx.full_text)
            if stacked_tables:
                stacked_ctx = replace(ctx, tables=stacked_tables, prefer_context_tables=True)
                transactions = grid_standard.extract_transactions(stacked_ctx, plugin)
                if transactions:

                    def _stacked_normalize(raw):
                        return grid_standard.normalize_record(raw, plugin)

                    normalize_fn = _stacked_normalize
                    if ctx.reconstruction is not None:
                        ctx.reconstruction = replace(
                            ctx.reconstruction,
                            source="stacked_text",
                            expected_primary_rows=expected,
                            pipe_parse_failed=False,
                        )
                    logger.info(
                        "[BankStyleRegistry] stacked text fallback rows=%d",
                        len(transactions),
                    )

        primary_parser = (detection.parser_chain or ["grid_standard"])[-1]
        primary_score, coverage = _parser_score(transactions, normalize_fn, plugin, expected)
        physical_tables = collect_physical_tables_from_parse_result(ctx.parse_result)
        physical_expected = physical_transaction_row_estimate(ctx.parse_result)
        if physical_tables and physical_expected > 0:
            physical_ctx = replace(ctx, tables=physical_tables, prefer_context_tables=True)
            physical_batch, physical_norm = _run_parser("grid_standard", physical_ctx, plugin)
            candidate_expected = max(expected, physical_expected)
            physical_score, physical_coverage = _parser_score(
                physical_batch,
                physical_norm,
                plugin,
                candidate_expected,
            )
            if physical_score > primary_score or (
                physical_coverage >= coverage and len(physical_batch) > len(transactions)
            ):
                transactions = physical_batch
                normalize_fn = physical_norm
                primary_score = physical_score
                coverage = physical_coverage
                expected = candidate_expected
                if ctx.reconstruction is not None:
                    ctx.reconstruction = replace(
                        ctx.reconstruction,
                        source="canonical_physical_tables",
                        expected_primary_rows=candidate_expected,
                        pipe_parse_failed=False,
                    )
                logger.info(
                    "[BankStyleRegistry] canonical physical table recovery rows=%d score=%.2f",
                    len(physical_batch),
                    physical_score,
                )
        if primary_score < _CAPS_THRESHOLD or (expected > 0 and coverage < _COVERAGE_THRESHOLD):
            for semantic_table in _semantic_text_table_candidates(ctx.full_text):
                semantic_tables = [semantic_table]
                semantic_expected = max(expected, sum(max(len(table) - 1, 0) for table in semantic_tables))
                semantic_ctx = replace(ctx, tables=semantic_tables, prefer_context_tables=True)
                semantic_batch, semantic_norm = _run_parser("grid_standard", semantic_ctx, plugin)
                semantic_score, semantic_coverage = _parser_score(
                    semantic_batch,
                    semantic_norm,
                    plugin,
                    semantic_expected,
                )
                if semantic_score > primary_score or (
                    semantic_coverage >= coverage and len(semantic_batch) > len(transactions)
                ):
                    transactions = semantic_batch
                    normalize_fn = semantic_norm
                    primary_score = semantic_score
                    coverage = semantic_coverage
                    expected = semantic_expected
                    if ctx.reconstruction is not None:
                        ctx.reconstruction = replace(
                            ctx.reconstruction,
                            source="semantic_text_table",
                            expected_primary_rows=semantic_expected,
                            pipe_parse_failed=False,
                        )
                    logger.info(
                        "[BankStyleRegistry] semantic text table recovery rows=%d score=%.2f",
                        len(semantic_batch),
                        semantic_score,
                    )
        positioned = recover_positioned_record_block_bank_tables(ctx.parse_result)
        if positioned.tables:
            positioned_ctx = replace(ctx, tables=positioned.tables, prefer_context_tables=True)
            positioned_batch, positioned_norm = _run_parser("grid_standard", positioned_ctx, plugin)
            _attach_recovered_sources(positioned_batch, positioned.row_sources)
            positioned_expected = max(expected, positioned.expected_rows)
            positioned_score, positioned_coverage = _parser_score(
                positioned_batch,
                positioned_norm,
                plugin,
                positioned_expected,
            )
            if positioned_score > primary_score or (
                positioned_coverage >= coverage and len(positioned_batch) > len(transactions)
            ):
                transactions = positioned_batch
                normalize_fn = positioned_norm
                primary_score = positioned_score
                coverage = positioned_coverage
                expected = positioned_expected
                if ctx.reconstruction is not None:
                    ctx.reconstruction = replace(
                        ctx.reconstruction,
                        source="positioned_record_block",
                        expected_primary_rows=positioned_expected,
                        pipe_parse_failed=False,
                    )
                logger.info(
                    "[BankStyleRegistry] positioned record-block recovery rows=%d score=%.2f",
                    len(positioned_batch),
                    positioned_score,
                )
        atom_tables = [] if positioned.tables else recover_evidence_atom_bank_tables(ctx.parse_result)
        if atom_tables:
            atom_count = sum(max(len(table) - 1, 0) for table in atom_tables)
            atom_expected = max(
                atom_count,
                recovered_evidence_atom_expected_row_count(ctx.parse_result),
            )
            atom_ctx = replace(ctx, tables=atom_tables, prefer_context_tables=True)
            atom_batch, atom_norm = _run_parser("grid_standard", atom_ctx, plugin)
            _attach_recovered_sources(
                atom_batch,
                recovered_evidence_atom_row_sources(ctx.parse_result),
            )
            atom_score, atom_coverage = _parser_score(atom_batch, atom_norm, plugin, atom_expected)
            primary_comparable_score, primary_comparable_coverage = _parser_score(
                transactions,
                normalize_fn,
                plugin,
                atom_expected,
            )
            richer_equal_coverage = (
                len(atom_batch) >= len(transactions)
                and atom_coverage >= primary_comparable_coverage
                and _batch_raw_width(atom_batch) > _batch_raw_width(transactions) + 1.0
            )
            if atom_score > primary_comparable_score or richer_equal_coverage:
                transactions = atom_batch
                normalize_fn = atom_norm
                primary_score = atom_score
                coverage = atom_coverage
                expected = atom_expected
                if ctx.reconstruction is not None:
                    ctx.reconstruction = replace(
                        ctx.reconstruction,
                        source="canonical_evidence_table",
                        expected_primary_rows=atom_expected,
                        pipe_parse_failed=False,
                    )
                logger.info(
                    "[BankStyleRegistry] canonical evidence table recovery rows=%d score=%.2f",
                    len(atom_batch),
                    atom_score,
                )
        if primary_score < _CAPS_THRESHOLD or (expected > 0 and coverage < _COVERAGE_THRESHOLD):
            wide_tables = recover_wide_bank_tables(ctx.parse_result, ctx.full_text)
            if wide_tables:
                wide_ctx = replace(ctx, tables=wide_tables, prefer_context_tables=True)
                wide_parser_id = "signed_amount" if detection.primary_style == "signed_amount" else "grid_standard"
                wide_batch, wide_norm = _run_parser(wide_parser_id, wide_ctx, plugin)
                wide_score, wide_coverage = _parser_score(wide_batch, wide_norm, plugin, expected)
                if wide_score > primary_score:
                    transactions = wide_batch
                    normalize_fn = wide_norm
                    primary_score = wide_score
                    coverage = wide_coverage
                    if ctx.reconstruction is not None:
                        ctx.reconstruction = replace(
                            ctx.reconstruction,
                            source="native_wide_table",
                            expected_primary_rows=expected,
                            pipe_parse_failed=False,
                        )
                    logger.info(
                        "[BankStyleRegistry] native wide table recovery rows=%d score=%.2f",
                        len(wide_batch),
                        wide_score,
                    )
        primary_score, coverage = _parser_score(transactions, normalize_fn, plugin, expected)
        if primary_score < _CAPS_THRESHOLD or (expected > 0 and coverage < _COVERAGE_THRESHOLD):
            ocr_tables = recover_ocr_implicit_ledger_tables(ctx.parse_result, ctx.full_text)
            if ocr_tables:
                recovered_count = sum(max(len(table) - 1, 0) for table in ocr_tables)
                ocr_expected = max(expected, recovered_count)
                ocr_ctx = replace(ctx, tables=ocr_tables, prefer_context_tables=True)
                ocr_batch, ocr_norm = _run_parser("grid_standard", ocr_ctx, plugin)
                ocr_score, ocr_coverage = _parser_score(ocr_batch, ocr_norm, plugin, ocr_expected)
                if ocr_score > primary_score:
                    transactions = ocr_batch
                    normalize_fn = ocr_norm
                    primary_score = ocr_score
                    coverage = ocr_coverage
                    expected = ocr_expected
                    if ctx.reconstruction is not None:
                        ctx.reconstruction = replace(
                            ctx.reconstruction,
                            source="ocr_implicit_table",
                            expected_primary_rows=expected,
                            pipe_parse_failed=False,
                        )
                    logger.info(
                        "[BankStyleRegistry] OCR implicit table recovery rows=%d score=%.2f",
                        len(ocr_batch),
                        ocr_score,
                    )
        needs_fallback = primary_score < _CAPS_THRESHOLD or (expected > 0 and coverage < _COVERAGE_THRESHOLD)
        if needs_fallback:
            best_batch = transactions
            best_norm = normalize_fn
            best_score = primary_score
            for fallback_id in _FALLBACK_PARSER_IDS:
                if fallback_id == primary_parser:
                    continue
                batch, norm = _run_parser(fallback_id, ctx, plugin)
                score, _ = _parser_score(batch, norm, plugin, expected)
                if score > best_score:
                    logger.info(
                        "[BankStyleRegistry] CAPS fallback parser=%s score=%.2f (was %.2f, %d rows)",
                        fallback_id,
                        score,
                        best_score,
                        len(best_batch),
                    )
                    best_batch = batch
                    best_norm = norm
                    best_score = score
            transactions = best_batch
            normalize_fn = best_norm

        candidates = _collect_table_candidates(
            detection,
            ctx,
            plugin,
            legacy_transactions=transactions,
            legacy_normalize_fn=normalize_fn,
        )
        selected, diagnostics = _select_candidate(
            candidates,
            native_text_suspicious=_has_suspicious_native_text(ctx.parse_result),
        )
        self.last_selection_diagnostics = diagnostics
        if selected is not None:
            transactions = selected.records
            normalize_fn = selected.normalize_fn
            selected_expected = selected.expected_rows
            if ctx.reconstruction is not None:
                ctx.reconstruction = replace(
                    ctx.reconstruction,
                    source=selected.source,
                    expected_primary_rows=(
                        selected_expected.count
                        if selected_expected is not None and selected_expected.confidence >= 0.85
                        else ctx.reconstruction.expected_primary_rows
                    ),
                    expected_evidence_source=(
                        selected_expected.source
                        if selected_expected is not None and selected_expected.confidence >= 0.85
                        else ctx.reconstruction.expected_evidence_source
                    ),
                    expected_evidence_confidence=(
                        selected_expected.confidence
                        if selected_expected is not None and selected_expected.confidence >= 0.85
                        else ctx.reconstruction.expected_evidence_confidence
                    ),
                    pipe_parse_failed=False,
                )
            logger.info(
                "[BankStyleRegistry] selected candidate=%s rows=%d reason=%s",
                selected.candidate_id,
                len(selected.records),
                diagnostics["selection_reason"],
            )

        if not transactions:
            batch, norm = _run_parser("grid_standard", ctx, plugin)
            transactions = batch
            if norm is None:

                def _grid_normalize(raw):
                    return grid_standard.normalize_record(raw, plugin)

                normalize_fn = _grid_normalize
            else:
                normalize_fn = norm

        if normalize_fn is None:

            def _plugin_normalize(raw):
                return plugin._normalize(raw)

            normalize_fn = _plugin_normalize

        def _normalize(raw: dict[str, Any]) -> dict[str, Any]:
            normalized = normalize_fn(raw)
            return ensure_canonical_normalized(normalized, plugin.standard_fields)

        records = records_from_raw_transactions(
            transactions,
            normalize_fn=_normalize,
            canonical_raw_fn=plugin._canonical_raw_values,
            style_id=detection.primary_style,
        )
        grid_standard.refine_missing_directions_from_balance_chain(records)

        if "compact_merged" in detection.parser_chain or detection.primary_style == "compact_merged_ledger":
            compact_merged.refine_directions_from_balance_chain(records)

        logger.info(
            "[BankStyleRegistry] style=%s chain=%s records=%d expected=%d",
            detection.primary_style,
            detection.parser_chain,
            len(records),
            expected,
        )
        return records, identity_fields


def _solve_split_debit_credit_tables(full_text: str) -> list[list[list[str]]]:
    """Use vNext domain solver when debit/credit ledger invariants close."""
    try:
        from docmirror.plugins.bank_statement.semantic_solver import BankStatementSemanticSolver

        solution = BankStatementSemanticSolver().solve(full_text=full_text)
    except Exception as exc:
        logger.debug("[BankStyleRegistry] bank semantic solver skipped: %s", exc)
        return []
    if not solution.success:
        return []
    split_table = (solution.canonical_model or {}).get("split_table")
    if not split_table:
        return []
    logger.info(
        "[BankStyleRegistry] semantic ledger solver rows=%d",
        max(len(split_table) - 1, 0),
    )
    return [split_table]
