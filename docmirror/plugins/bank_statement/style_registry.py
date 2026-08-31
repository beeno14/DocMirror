# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bank-statement table parser and candidate-selection registry.

Maps detected style IDs to parser modules under ``bank_statement.styles`` and
deploys the recovery candidates allowed by the pre-resolved acquisition policy.
A source-complete primary result can stop deployment early; otherwise the
original eager candidate set is materialized and selected as one unit.

Pipeline role: plugin-local dispatch between ``BankStyleDetector`` and record
builders inside the post-seal bank-statement projector.

Key exports: ``BankStyleParserRegistry``.

Dependencies: ``bank_statement.styles.*``, ``bank_statement.canonical``.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, replace
from datetime import date
from typing import Any, Callable

from docmirror.plugins._runtime.evidence_access import evidence_payload
from docmirror.plugins.bank_statement.canonical import ensure_canonical_normalized, records_from_raw_transactions
from docmirror.plugins.bank_statement.canonical_quality import (
    canonical_expected_from_parse_result,
    is_canonical_row,
    physical_transaction_row_estimate,
)
from docmirror.plugins.bank_statement.context import (
    StyleContext,
    collect_physical_table_row_sources_from_parse_result,
    collect_physical_tables_from_parse_result,
)
from docmirror.plugins.bank_statement.evidence_atom_table_recovery import (
    recover_evidence_atom_bank_tables,
    recover_positioned_record_block_bank_tables,
    recovered_evidence_atom_expected_row_evidence,
    recovered_evidence_atom_row_sources,
    recovered_native_datetime_row_evidence,
)
from docmirror.plugins.bank_statement.extraction_dispatch import BankExtractionRoute
from docmirror.plugins.bank_statement.header_resolve import normalize_header_cell
from docmirror.plugins.bank_statement.ocr_implicit_table_recovery import (
    recover_ocr_implicit_ledger_tables,
    recovered_ocr_implicit_row_count,
    recovered_ocr_implicit_row_evidence,
)
from docmirror.plugins.bank_statement.schema_guided_page_text import recover_schema_guided_page_text
from docmirror.plugins.bank_statement.statement_context import statement_scope_count
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
    RowCountEvidence,
    count_expected_rows_from_bank_footer,
    page_texts_from_parse_result,
    recover_wide_bank_tables,
    resolve_row_count_evidence,
)
from docmirror.tables.access import get_logical_tables

logger = logging.getLogger(__name__)

_PARSERS = {
    "compact_merged": compact_merged,
    "grid_standard": grid_standard,
    "kv_identity": kv_identity,
    "signed_amount": signed_amount,
    "borderless_ocr": borderless_ocr,
}

_PARSER_CANDIDATE_ORDER = ("grid_standard", "signed_amount", "compact_merged", "borderless_ocr")
_INDEPENDENT_ROW_COUNT_SOURCES = frozenset(
    {
        "split_footer",
        "header_total",
        "statement_header_totals",
        "cumulative_footer_total",
        "page_footer",
    }
)
_ISSUER_ROW_COUNT_SOURCES = frozenset(
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
        "schema_guided_page_text_rows",
        "positioned_date_anchors",
        "positioned_record_blocks",
    }
)
_SOURCE_DATE_RE = re.compile(r"20\d{2}(?:[-/.]?\d{1,2}){2}")
_SEALED_TIME_RE = re.compile(r"(?<![\d:])\d{1,2}:\d{2}:\d{2}(?![\d:])")
_SEALED_MONEY_RE = re.compile(r"(?<!\d)[+-]?\d[\d,]*\.\d{2}(?!\d)")
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
_SOURCE_SUMMARY_HEADERS = frozenset(
    {
        "summary",
        "摘要",
        "摘要描述",
        "交易摘要",
        "description",
        "memo",
    }
)
_FURNITURE_LABELS = (
    "打印时间",
    "打印日期",
    "记录数",
    "用户所属公司",
    "查询条件",
    "查询账号",
    "账户信息",
)
_SEMANTIC_ECHO_FIELDS = (
    "date",
    "timestamp",
    "summary",
    "purpose",
    "sequence_no",
    "reference",
    "channel",
    "counter_account",
    "counter_party",
)
_DELIMITED_BUSINESS_LAYOUT_HEADERS = frozenset(
    {"序号", "摘要/附言", "币别", "交易日期", "交易类型", "交易金额", "账户余额", "对方账号", "对方户名"}
)
_DELIMITED_BUSINESS_CODES = frozenset({"0WL", "_0WL", "1银联"})
_DELIMITED_BUSINESS_TYPES = frozenset({"WL协议", "WL付款", "WL退款", "银联贷记"})
_DELIMITED_BUSINESS_MARKER_RE = re.compile(r"(?<![0-9A-Za-z_])(?:0WL|_0WL|1银联)#")
_DELIMITED_BUSINESS_REFERENCE_RE = re.compile(r"[0-9S]{6,31}", re.IGNORECASE)


def _batch_field_completeness(
    transactions: list[dict[str, str]],
    normalize_fn: Any,
    plugin: Any,
    *,
    normalized_rows: list[dict[str, Any]] | None = None,
) -> float:
    if not transactions:
        return 0.0
    nf = normalize_fn or plugin._normalize
    balance_expected = any(
        any(
            "余额" in grid_standard.normalize_header_cell(str(key)) or "balance" in str(key).lower()
            for key in transaction
        )
        for transaction in transactions
    )
    fields = ("date", "amount", "direction", *(("balance",) if balance_expected else ()))
    scores = []
    rows = normalized_rows
    if rows is None:
        rows = [ensure_canonical_normalized(nf(txn), plugin.standard_fields) for txn in transactions]
    for norm in rows:
        scores.append(sum(1 for f in fields if norm.get(f) not in (None, "")) / len(fields))
    return sum(scores) / len(scores)


def _batch_raw_width(transactions: list[dict[str, str]], sample: int | None = None) -> float:
    """Average number of populated source columns, used only as a tie-breaker."""
    if not transactions:
        return 0.0
    widths = [
        sum(bool(str(value or "").strip()) for key, value in transaction.items() if not key.startswith("_"))
        for transaction in (transactions if sample is None else transactions[:sample])
    ]
    return sum(widths) / len(widths)


def _expected_rows(ctx: StyleContext) -> int:
    footer_expected = count_expected_rows_from_bank_footer(
        ctx.full_text,
        page_texts=page_texts_from_parse_result(ctx.parse_result),
    )
    if footer_expected > 0:
        return footer_expected
    canonical_expected = canonical_expected_from_parse_result(ctx.parse_result)
    ocr_expected = (
        recovered_ocr_implicit_row_count(ctx.parse_result) if ctx.extraction_route is BankExtractionRoute.SCANNED else 0
    )
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
    from docmirror.plugins.bank_statement.work_cache import active_bank_cache, reuse_bank_work

    if not active_bank_cache(ctx.parse_result):
        return _run_parser_uncached(parser_id, ctx, plugin)
    key = (
        parser_id,
        id(plugin),
        tuple(tuple(tuple(row) for row in table) for table in ctx.tables),
        ctx.full_text,
        ctx.institution,
        ctx.institution_authority,
        ctx.page_count,
        ctx.prefer_context_tables,
        ctx.extraction_route,
        ctx.extraction_policy,
        tuple(vars(ctx.reconstruction).items()) if ctx.reconstruction is not None else None,
    )
    return reuse_bank_work(
        ctx.parse_result,
        _run_parser_uncached,
        key,
        lambda: _run_parser_uncached(parser_id, ctx, plugin),
        capture=lambda: ctx.reconstruction,
        restore=lambda value: setattr(ctx, "reconstruction", value),
    )


def _run_parser_uncached(parser_id: str, ctx: StyleContext, plugin: Any) -> tuple[list[dict[str, Any]], Any]:
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
    identity_queues: dict[tuple[int, str, int], list[dict[str, Any]]] = {}
    source_queues: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for source in row_sources:
        identity = _physical_row_identity(source)
        if identity is not None:
            identity_queues.setdefault(identity, []).append(source)
        signature = _row_value_signature(source.get("row_values") or [])
        if signature:
            source_queues.setdefault(signature, []).append(source)

    unmatched: list[int] = []
    used_source_ids: set[int] = set()
    for index, transaction in enumerate(transactions):
        transaction_source = transaction.get("_source")
        identity = _physical_row_identity(transaction_source) if isinstance(transaction_source, dict) else None
        queue = identity_queues.get(identity) or []
        source = _pop_unused_source(queue, used_source_ids)
        if source is None:
            queue = source_queues.get(_row_value_signature(transaction)) or []
            source = _pop_unused_source(queue, used_source_ids)
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


def _physical_row_identity(source: dict[str, Any]) -> tuple[int, str, int] | None:
    try:
        page = int(source.get("source_page") or source.get("page") or 0)
        row_index = int(source.get("source_row_index"))
    except (TypeError, ValueError):
        return None
    table_id = str(source.get("table_id") or "").strip()
    if page <= 0 or not table_id or row_index < 0:
        return None
    return page, table_id, row_index


def _pop_unused_source(
    queue: list[dict[str, Any]],
    used_source_ids: set[int],
) -> dict[str, Any] | None:
    while queue:
        source = queue.pop(0)
        if id(source) not in used_source_ids:
            return source
    return None


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
    source_fragment_coverage: float = 0.0
    extraction_confidence: float = 0.0
    source_column_width: float = 0.0
    sequence_continuity: float = 0.0
    native_cell_coverage: float = 0.0
    semantic_anomaly_rows: int = 0
    source_role_swap_ratio: float = 0.0
    rejected_row_indexes: tuple[int, ...] = ()
    rejection_reason: str = ""


@dataclass(frozen=True)
class CandidateCompletionProof:
    """Conservative sealed-document proof for stopping after the primary parser."""

    state: str
    reason: str

    @property
    def proven(self) -> bool:
        return self.state == "proven"


def _candidate_expected_rows(
    source_evidence: RowCountEvidence,
    *,
    count: int = 0,
    source: str = "candidate_rows",
    confidence: float = 0.55,
) -> RowCountEvidence | None:
    row_plane_signal = source_evidence.source in _ROW_PLANE_COUNT_SOURCES
    if row_plane_signal:
        source_evidence = replace(source_evidence, confidence=min(source_evidence.confidence, 0.80))
    if source_evidence.count > 0 and source_evidence.confidence >= 0.85:
        return source_evidence
    if count > 0:
        fallback_confidence = min(confidence, 0.80) if source in _ROW_PLANE_COUNT_SOURCES else confidence
        return RowCountEvidence(count=count, source=source, confidence=fallback_confidence)
    if row_plane_signal and source_evidence.count > 0:
        # Keep the bounded row-plane signal available for ranking when this
        # candidate has no count of its own.  When it does, however, describe
        # the candidate with that local count: a separate row plane at 0.80 is
        # structural corroboration, not the candidate's completeness proof.
        return source_evidence
    return source_evidence if source_evidence.count > 0 else None


def _continuous_source_sequence_evidence(transactions: list[dict[str, Any]]) -> RowCountEvidence | None:
    """Return candidate-local continuity evidence for an exact ``1..N`` spine.

    A retained prefix ``1..5`` is still continuous when source row 6 was lost,
    so this signal is useful for ranking candidates but is not an independent
    document-completeness proof.  Other source-row-plane censuses have the
    same limitation and likewise stay below the public count-authority
    threshold.
    """
    values: list[int] = []
    for transaction in transactions:
        if _source_row_contains_multiple_transaction_dates(transaction):
            continue
        raw_sequence = ""
        for key, value in transaction.items():
            if normalize_header_cell(str(key or "")) in {"序号", "交易序号", "sequence"}:
                raw_sequence = re.sub(r"\s+", "", str(value or ""))
                break
        if not re.fullmatch(r"\d{1,9}", raw_sequence):
            return None
        values.append(int(raw_sequence))
    if not values or values != list(range(1, len(values) + 1)):
        return None
    return RowCountEvidence(
        count=len(values),
        source="continuous_source_sequence",
        confidence=0.80,
    )


def _transaction_source_page(transaction: dict[str, Any]) -> int:
    source = transaction.get("_source")
    values = (
        source.get("source_page") if isinstance(source, dict) else None,
        source.get("page") if isinstance(source, dict) else None,
        transaction.get("_source_page"),
    )
    for value in values:
        try:
            page = int(value or 0)
        except (TypeError, ValueError):
            continue
        if page > 0:
            return page
    return 0


def _source_sequence_values(transactions: list[dict[str, Any]]) -> list[tuple[int, int]] | None:
    values: list[tuple[int, int]] = []
    for transaction in transactions:
        if _source_row_contains_multiple_transaction_dates(transaction):
            continue
        raw_sequence = ""
        for key, value in transaction.items():
            if normalize_header_cell(str(key or "")) in {"序号", "交易序号", "sequence"}:
                raw_sequence = re.sub(r"\s+", "", str(value or ""))
                break
        if not re.fullmatch(r"\d{1,9}", raw_sequence):
            return None
        values.append((_transaction_source_page(transaction), int(raw_sequence)))
    return values or None


def _page_complete_sequence_evidence(
    transactions: list[dict[str, Any]],
    *,
    page_count: int,
    page_texts: list[tuple[int, str]] | None = None,
) -> RowCountEvidence | None:
    """Return a source-row-plane consistency signal for page-local ordinals.

    A candidate-local consecutive sequence is not a completeness witness: rows
    ``1..5`` are still consecutive when row 6 was dropped.  The BOC statements
    for which page-local ordinals reset expose a stable bilingual pipe ledger in
    each sealed page text.  Count only those bounded source rows, require their
    page markers and footer band, and then require the candidate ordinals to
    match that source census exactly.
    """
    values = _source_sequence_values(transactions)
    if not values or page_count <= 0:
        return None
    pages = [page for page, _value in values]
    if any(page <= 0 for page in pages) or set(pages) != set(range(1, page_count + 1)):
        return None
    scoped_text = {page: str(text or "") for page, text in (page_texts or [])}
    if set(scoped_text) != set(range(1, page_count + 1)):
        return None
    by_page: dict[int, list[int]] = {}
    for page, value in values:
        by_page.setdefault(page, []).append(value)

    source_by_page: dict[int, list[int]] = {}
    row_pattern = re.compile(
        r"(?m)^\|\s*(?P<sequence>\d{1,4})\s*\|\s*"
        r"(?P<posting>\d{6})\s*\|\s*(?P<value>\d{6})\s*\|"
    )
    page_marker = re.compile(r"\bPage\s+(?P<page>\d+)\s+of\s+(?P<total>\d+)\b", re.I)
    exact_header_markers = ("|No.", "|Bk.D.", "|Val.D.", "| Type ", "| Notes |")

    for page in range(1, page_count + 1):
        text = scoped_text[page]
        marker = page_marker.search(text)
        if (
            marker is None
            or int(marker.group("page")) != page
            or int(marker.group("total")) != page_count
            or not all(label in text for label in exact_header_markers)
            or "Debit Total" not in text
            or "Credit Total" not in text
        ):
            return None
        source_rows: list[int] = []
        for match in row_pattern.finditer(text):
            for field in ("posting", "value"):
                raw_date = match.group(field)
                try:
                    date(2000 + int(raw_date[:2]), int(raw_date[2:4]), int(raw_date[4:6]))
                except ValueError:
                    return None
            source_rows.append(int(match.group("sequence")))
        if not source_rows or source_rows != list(range(1, len(source_rows) + 1)):
            return None
        source_by_page[page] = source_rows

    if by_page != source_by_page:
        return None
    # The candidate and page-text rows are two representations of the same row
    # plane.  Agreement is useful for ranking, but a shared truncation can make
    # both representations agree on a short prefix, so it is not an independent
    # document-level denominator.
    return RowCountEvidence(sum(len(rows) for rows in source_by_page.values()), "complete_page_local_sequences", 0.80)


def _candidate_row_count_evidence(
    transactions: list[dict[str, Any]],
    expected: RowCountEvidence | None,
    *,
    page_count: int = 0,
    page_texts: list[tuple[int, str]] | None = None,
) -> RowCountEvidence | None:
    """Prefer exact candidate sequences without letting a prefix defeat issuer evidence.

    Sequence evidence must be computed for every candidate representation. Doing
    so only for native-wide recovery lets a short 1..N native prefix claim exact
    completeness while a fuller parser candidate carrying the same source
    sequence is scored against weak text anchors.
    """
    # Prefer the independently bounded page-text census.  A one-page table can
    # also be globally 1..N; checking bare continuity first would discard the
    # stronger source-boundary proof and unnecessarily demote the result.
    sequence = _page_complete_sequence_evidence(
        transactions,
        page_count=page_count,
        page_texts=page_texts,
    ) or _continuous_source_sequence_evidence(transactions)
    if sequence is None:
        return expected
    if sequence.confidence < 0.85:
        # Bare continuity is candidate quality, never a reason to replace an
        # independently counted (or even independently anchored) denominator.
        return expected if expected is not None and expected.count > 0 else sequence
    if expected is None or expected.count <= 0 or expected.confidence < 0.85:
        return sequence
    if expected.source in _ISSUER_ROW_COUNT_SOURCES:
        return expected
    if expected.source in {"page_transaction_anchors", "physical_rows"}:
        return sequence if sequence.count >= expected.count else expected
    return expected


def _evidence_atom_expected_rows(
    parse_result: Any,
    atom_tables: list[list[list[str]]],
    *,
    source_route: str | None = None,
) -> RowCountEvidence | None:
    """Return independent page-anchor evidence when recovery can prove it."""
    table_rows = sum(max(len(table) - 1, 0) for table in atom_tables)
    evidence_count, evidence_source, evidence_confidence = recovered_evidence_atom_expected_row_evidence(
        parse_result,
        source_route=source_route,
    )
    expected = max(table_rows, evidence_count)
    if expected <= 0:
        return None
    if evidence_count == expected and evidence_source:
        if evidence_source in _ROW_PLANE_COUNT_SOURCES:
            evidence_confidence = min(evidence_confidence, 0.80)
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


def _candidate_sequence_continuity(
    normalized_rows: list[dict[str, Any]],
    transactions: list[dict[str, Any]] | None = None,
) -> float:
    """Return source-order sequence coverage across legitimate local resets.

    Many statements reset an ordinal to ``1`` at a page, query batch, or
    statement-segment boundary. Global uniqueness therefore measures document
    segmentation rather than extraction quality. A reset counts as continuous
    only when its source provenance moves strictly forward; concatenating a
    duplicate extraction plane jumps back to an earlier page and remains
    penalized.
    """
    numbered: list[tuple[int, dict[str, Any] | None]] = []
    transaction_rows = transactions or []
    for index, row in enumerate(normalized_rows):
        value = str(row.get("sequence_no") or "").strip()
        if re.fullmatch(r"\d{1,9}", value):
            transaction = transaction_rows[index] if index < len(transaction_rows) else None
            numbered.append((int(value), transaction))
    if len(numbered) < 2:
        return 0.0
    coverage = len(numbered) / max(len(normalized_rows), 1)
    continuous = 0
    for (previous, previous_transaction), (current, current_transaction) in zip(numbered, numbered[1:]):
        if (
            previous_transaction is not None
            and current_transaction is not None
            and _candidate_source_page_rewinds(previous_transaction, current_transaction)
        ):
            continue
        if abs(current - previous) == 1:
            continuous += 1
            continue
        if (
            current == 1
            and previous_transaction is not None
            and current_transaction is not None
            and _candidate_source_row_follows(previous_transaction, current_transaction)
        ):
            continuous += 1
    return coverage * continuous / (len(numbered) - 1)


def _candidate_source_page_rewinds(
    previous_transaction: dict[str, Any],
    current_transaction: dict[str, Any],
) -> bool:
    previous_source = previous_transaction.get("_source")
    current_source = current_transaction.get("_source")
    if not isinstance(previous_source, dict) or not isinstance(current_source, dict):
        return False
    try:
        previous_page = int(previous_source.get("source_page") or 0)
        current_page = int(current_source.get("source_page") or 0)
    except (TypeError, ValueError):
        return False
    return previous_page > 0 and current_page > 0 and current_page < previous_page


def _candidate_source_row_follows(
    previous_transaction: dict[str, Any],
    current_transaction: dict[str, Any],
) -> bool:
    previous_source = previous_transaction.get("_source")
    current_source = current_transaction.get("_source")
    if not isinstance(previous_source, dict) or not isinstance(current_source, dict):
        return False
    try:
        previous_page = int(previous_source.get("source_page") or 0)
        current_page = int(current_source.get("source_page") or 0)
    except (TypeError, ValueError):
        return False
    if previous_page <= 0 or current_page <= 0 or current_page < previous_page:
        return False
    if current_page > previous_page:
        return True

    previous_bbox = _candidate_source_bbox(previous_transaction)
    current_bbox = _candidate_source_bbox(current_transaction)
    if previous_bbox is not None and current_bbox is not None:
        return current_bbox[1] > previous_bbox[1] + 0.5
    try:
        previous_index = int(previous_source.get("source_row_index") or 0)
        current_index = int(current_source.get("source_row_index") or 0)
    except (TypeError, ValueError):
        return False
    return previous_index > 0 and current_index > previous_index


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


def _candidate_source_fragment_coverage(transactions: list[dict[str, Any]]) -> float:
    """Return coverage by rows proven from adjacent cross-page source fragments."""
    if not transactions:
        return 0.0
    stitched = 0
    for transaction in transactions:
        source = transaction.get("_source")
        if not isinstance(source, dict):
            continue
        page_range = source.get("page_range")
        refs = source.get("source_refs")
        if not isinstance(page_range, (list, tuple)) or len(page_range) != 2 or not isinstance(refs, list):
            continue
        try:
            first_page, last_page = (int(page) for page in page_range)
            ref_pages = {
                int(ref.get("source_page") or ref.get("page") or 0)
                for ref in refs
                if isinstance(ref, dict)
            }
        except (TypeError, ValueError):
            continue
        if first_page > 0 and last_page == first_page + 1 and {first_page, last_page}.issubset(ref_pages):
            stitched += 1
    return stitched / len(transactions)


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


def _apply_delivery_refinement_for_candidate_quality(
    transactions: list[dict[str, Any]],
    normalized_rows: list[dict[str, Any]],
) -> None:
    """Score candidates after the same deterministic refinement used on output.

    Candidate selection previously measured canonical coverage before the
    already-deployed balance-chain refinement, then applied that refinement
    only to the winning candidate.  That made a complete whole-document plane
    look sparse and allowed a locally perfect page prefix to win.  Use the
    existing refinement here as an evaluation step; no extraction strategy or
    source value is changed.
    """

    if not transactions or len(transactions) != len(normalized_rows):
        return
    from docmirror.plugins._base.base_table_parser import public_record_raw

    provisional_records: list[dict[str, Any]] = []
    for transaction, normalized in zip(transactions, normalized_rows, strict=True):
        record: dict[str, Any] = {
            "raw": public_record_raw(dict(transaction)),
            "normalized": normalized,
        }
        scope_text = str(transaction.get("_document_scope_text") or "").strip()
        if scope_text:
            record["_document_scope_text"] = scope_text
        provisional_records.append(record)
    grid_standard.refine_missing_directions_from_balance_chain(provisional_records)


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
    retained_transactions: list[dict[str, Any]] = []
    normalized_rows: list[dict[str, Any]] = []
    canonical_rows = 0
    directional_rows = 0
    semantic_anomaly_rows = 0
    aggregate_anomaly_rows = 0
    semantic_anomaly_flags: list[bool] = []
    rejected_row_indexes: list[int] = []
    role_swap_eligible = 0
    role_swap_rows = 0
    for row_index, transaction in enumerate(transactions):
        # A native page can expose one column-aggregated pseudo-row alongside
        # all of its correctly bounded physical rows.  Multiple complete dates
        # in a single source date cell prove that this item is not a transaction;
        # discard that furniture without condemning the valid physical rows beside it.
        if _source_row_contains_multiple_transaction_dates(transaction):
            semantic_anomaly_rows += 1
            aggregate_anomaly_rows += 1
            rejected_row_indexes.append(row_index)
            continue
        try:
            normalized = ensure_canonical_normalized(normalizer(transaction), plugin.standard_fields)
        except Exception:
            normalized = {}
        normalized_rows.append(normalized)
        retained_transactions.append(transaction)
        source_date, source_summary = _source_date_and_summary(transaction)
        if source_date and source_summary:
            role_swap_eligible += 1
            if _compact_semantic_value(source_date) == _compact_semantic_value(source_summary):
                role_swap_rows += 1
        semantic_anomaly = _row_has_semantic_anomaly(transaction, normalized)
        semantic_anomaly_flags.append(semantic_anomaly)
        if semantic_anomaly:
            semantic_anomaly_rows += 1
            rejected_row_indexes.append(row_index)
    _apply_delivery_refinement_for_candidate_quality(retained_transactions, normalized_rows)
    directional_rows = sum(row.get("direction") in {"income", "expense"} for row in normalized_rows)
    canonical_rows = sum(
        is_canonical_row(normalized) and not semantic_anomaly
        for normalized, semantic_anomaly in zip(normalized_rows, semantic_anomaly_flags, strict=True)
    )
    role_swap_ratio = role_swap_rows / role_swap_eligible if role_swap_eligible else 0.0
    rejection_reason = ""
    nonaggregate_anomalies = semantic_anomaly_rows - aggregate_anomaly_rows
    if nonaggregate_anomalies:
        # A row with transaction-shaped core values may still be a genuine row
        # whose columns were shifted by reconstruction.  Removing just that row
        # would manufacture incompleteness, so reject the whole alignment and
        # let a source-faithful candidate compete instead.
        canonical_rows = 0
        rejection_reason = "source_role_corruption"
    if role_swap_eligible >= 3 and role_swap_ratio >= 0.80:
        # A whole candidate whose source summary is systematically the source
        # date has lost column roles.  Row counts alone cannot make it viable.
        semantic_anomaly_rows += canonical_rows
        canonical_rows = 0
        rejected_row_indexes = list(range(len(transactions)))
        rejection_reason = "systemic_summary_date_collision"
    expected_count = int(expected_rows.count or 0) if expected_rows is not None else 0
    if expected_count > 0 and (expected_rows is not None and expected_rows.confidence >= 0.85):
        canonical_coverage = min(canonical_rows / expected_count, 1.0)
    else:
        canonical_coverage = canonical_rows / max(len(transactions), 1)
    field_completeness = _batch_field_completeness(
        retained_transactions,
        normalize_fn,
        plugin,
        normalized_rows=normalized_rows,
    )
    source_column_width = _batch_raw_width(retained_transactions)
    source_page_coverage = _candidate_source_page_coverage(retained_transactions)
    source_fragment_coverage = _candidate_source_fragment_coverage(retained_transactions)
    native_cell_coverage = _candidate_native_cell_coverage(retained_transactions)
    balance_chain_score = _candidate_balance_chain_score(normalized_rows)
    sequence_continuity = _candidate_sequence_continuity(normalized_rows, retained_transactions)
    score = (
        0.35 * canonical_coverage
        + 0.25 * source_page_coverage
        + 0.20 * field_completeness
        + 0.15 * balance_chain_score
        + 0.05 * extraction_confidence
    )
    return BankTableCandidate(
        candidate_id=candidate_id,
        records=retained_transactions,
        source=source,
        canonical_rows=canonical_rows,
        directional_rows=directional_rows,
        source_page_rows=round(source_page_coverage * len(retained_transactions)),
        expected_rows=expected_rows,
        balance_chain_score=balance_chain_score,
        field_completeness=field_completeness,
        score=score,
        normalize_fn=normalize_fn,
        canonical_coverage=canonical_coverage,
        source_page_coverage=source_page_coverage,
        source_fragment_coverage=source_fragment_coverage,
        extraction_confidence=extraction_confidence,
        source_column_width=source_column_width,
        sequence_continuity=sequence_continuity,
        native_cell_coverage=native_cell_coverage,
        semantic_anomaly_rows=semantic_anomaly_rows,
        source_role_swap_ratio=role_swap_ratio,
        rejected_row_indexes=tuple(rejected_row_indexes),
        rejection_reason=rejection_reason,
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


def _source_fact_pool(transaction: dict[str, Any]) -> dict[str, Any]:
    source_raw = transaction.get("_source_raw")
    return source_raw if isinstance(source_raw, dict) else transaction


def _normalized_source_header(value: Any) -> str:
    return re.sub(r"\s+", "", grid_standard.normalize_header_cell(str(value or ""))).lower()


def _source_date_and_summary(transaction: dict[str, Any]) -> tuple[str, str]:
    source = _source_fact_pool(transaction)
    source_date = ""
    source_summary = ""
    for raw_header, value in source.items():
        if str(raw_header).startswith("_") or value in (None, ""):
            continue
        header = _normalized_source_header(raw_header)
        header_parts = {
            _normalized_source_header(part) for part in str(raw_header or "").splitlines() if str(part).strip()
        }
        if not source_date and (header in _SOURCE_DATE_HEADERS or header_parts.intersection(_SOURCE_DATE_HEADERS)):
            source_date = str(value).strip()
        if not source_summary and (
            header in _SOURCE_SUMMARY_HEADERS or header_parts.intersection(_SOURCE_SUMMARY_HEADERS)
        ):
            source_summary = str(value).strip()
    return source_date, source_summary


def _compact_semantic_value(value: Any) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", str(value or "").casefold())


def _row_has_semantic_anomaly(transaction: dict[str, Any], normalized: dict[str, Any]) -> bool:
    """Reject only high-specificity row-boundary or field-role corruption.

    Missing optional fields are deliberately not anomalies.  These guards target
    patterns observed in digital source recovery where page metadata or one
    repeated token was copied into several unrelated business roles.
    """
    source = _source_fact_pool(transaction)
    # A delimited source business cell is self-describing: once a recognized
    # code family appears, shifted/truncated geometry must not silently turn it
    # into a transaction. This is grammar evidence, not an issuer/file switch;
    # ordinary free-text summaries and unknown delimiters remain untouched.
    source_headers = {
        re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(raw_header or "")))
        for raw_header in source
        if not str(raw_header or "").startswith("_")
    }
    for raw_header, raw_value in source.items():
        header = re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(raw_header or "")))
        if header not in {"对方账号户名/附言", "对方账户户名/附言"}:
            continue
        source_compound = _compact_semantic_value(raw_value)
        if not source_compound:
            continue
        if any(
            _compact_semantic_value(normalized.get(field)) == source_compound
            for field in ("counter_account", "sub_account")
        ):
            return True
    delimited_layout = _DELIMITED_BUSINESS_LAYOUT_HEADERS.issubset(source_headers)
    for raw_header, value in source.items():
        header = re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(raw_header or "")))
        if not delimited_layout or header != "摘要/附言":
            continue
        compact_value = re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(value or "")))
        parts = compact_value.split("#")
        recognized_marker = _DELIMITED_BUSINESS_MARKER_RE.search(compact_value) is not None
        recognized_grammar = bool(parts and parts[0] in _DELIMITED_BUSINESS_CODES)
        if recognized_marker and (
            not recognized_grammar
            or len(parts) != 4
            or parts[2] not in _DELIMITED_BUSINESS_TYPES
            or _DELIMITED_BUSINESS_REFERENCE_RE.fullmatch(parts[1]) is None
            or not parts[3]
        ):
            return True
    source_text = " ".join(
        str(value or "") for key, value in source.items() if not str(key).startswith("_") and value not in (None, "")
    )
    furniture_hits = sum(label in source_text for label in _FURNITURE_LABELS)
    if furniture_hits >= 2:
        return True

    role_values: dict[str, str] = {}
    for field in _SEMANTIC_ECHO_FIELDS:
        compact = _compact_semantic_value(normalized.get(field))
        if len(compact) < 6 or compact in {"000000", "none", "null", "na"}:
            continue
        role_values[field] = compact
    occurrences: dict[str, set[str]] = {}
    for field, value in role_values.items():
        occurrences.setdefault(value, set()).add(field)
    if any(len(fields) >= 4 for fields in occurrences.values()):
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


def _is_semantically_comparable_fuller_candidate(
    candidate: BankTableCandidate,
    selected: BankTableCandidate,
) -> bool:
    """Return whether ``candidate`` safely demonstrates more canonical rows.

    Candidate identifiers describe extraction routes, not evidence authority.  A
    completeness guard must therefore work for any pair of routes.  The modest
    tolerances account for normal scoring noise while protecting every semantic
    dimension that can make a shorter reconstruction genuinely preferable.
    """
    if len(candidate.records) <= len(selected.records) or candidate.canonical_rows <= selected.canonical_rows:
        return False

    candidate_density = candidate.canonical_rows / max(len(candidate.records), 1)
    selected_density = selected.canonical_rows / max(len(selected.records), 1)
    broadens_source_scope = _candidate_strictly_broadens_source_scope(candidate, selected)
    balance_is_comparable = (
        candidate.balance_chain_score >= selected.balance_chain_score - 0.10
        or (broadens_source_scope and candidate.balance_chain_score >= 0.45)
    )
    sequence_is_comparable = (
        candidate.sequence_continuity >= selected.sequence_continuity - 0.10 or broadens_source_scope
    )
    score_is_comparable = candidate.score >= selected.score - (0.12 if broadens_source_scope else 0.05)
    return (
        candidate_density >= selected_density - 0.01
        and candidate.canonical_coverage >= selected.canonical_coverage - 0.03
        and candidate.source_fragment_coverage >= selected.source_fragment_coverage - 0.01
        and candidate.source_page_coverage >= selected.source_page_coverage - 0.05
        and candidate.field_completeness >= selected.field_completeness - 0.05
        and balance_is_comparable
        and sequence_is_comparable
        and candidate.source_column_width
        >= selected.source_column_width - max(0.25, selected.source_column_width * 0.02)
        and score_is_comparable
    )


def _candidate_strictly_broadens_source_scope(
    candidate: BankTableCandidate,
    selected: BankTableCandidate,
) -> bool:
    candidate_pages = _candidate_source_pages(candidate.records)
    selected_pages = _candidate_source_pages(selected.records)
    return bool(selected_pages and selected_pages < candidate_pages)


def _candidate_source_pages(transactions: list[dict[str, Any]]) -> set[int]:
    pages: set[int] = set()
    for transaction in transactions:
        source = transaction.get("_source")
        if not isinstance(source, dict):
            continue
        try:
            source_page = int(source.get("source_page") or 0)
        except (TypeError, ValueError):
            source_page = 0
        if source_page > 0:
            pages.add(source_page)
        page_range = source.get("page_range")
        if not isinstance(page_range, (list, tuple)) or len(page_range) != 2:
            continue
        try:
            first_page, last_page = (int(value) for value in page_range)
        except (TypeError, ValueError):
            continue
        if 0 < first_page <= last_page and last_page - first_page <= 10_000:
            pages.update(range(first_page, last_page + 1))
    return pages


def _candidate_is_safe_page_component(candidate: BankTableCandidate) -> bool:
    """Return whether a candidate can safely contribute a disjoint page span."""

    row_count = len(candidate.records)
    pages = _candidate_source_pages(candidate.records)
    return bool(
        row_count
        and pages
        and candidate.canonical_rows / row_count >= 0.99
        and candidate.canonical_coverage >= 0.99
        and candidate.source_page_coverage >= 0.99
        and candidate.field_completeness >= 0.95
        and candidate.balance_chain_score >= 0.45
        and not candidate.rejection_reason
        and pages == set(range(min(pages), max(pages) + 1))
    )


def _candidate_is_safe_material_partial_component(
    candidate: BankTableCandidate,
    selected: BankTableCandidate,
) -> bool:
    """Return whether a disjoint partial plane adds many validated rows safely."""

    row_count = len(candidate.records)
    pages = _candidate_source_pages(candidate.records)
    evidence = candidate.expected_rows
    return bool(
        row_count
        and len(pages) >= 2
        and pages.isdisjoint(_candidate_source_pages(selected.records))
        and candidate.canonical_rows >= max(25, selected.canonical_rows * 2)
        and candidate.canonical_rows / row_count >= 0.90
        and candidate.canonical_coverage >= 0.90
        and candidate.source_page_coverage >= 0.99
        and candidate.field_completeness >= 0.95
        and candidate.balance_chain_score >= 0.90
        and candidate.source_column_width >= 4.0
        and not candidate.rejection_reason
        and evidence is not None
        and evidence.count >= candidate.canonical_rows
        and evidence.source in _ROW_PLANE_COUNT_SOURCES
    )


def _material_source_disjoint_components(
    selected: BankTableCandidate,
    candidates: list[BankTableCandidate],
) -> list[BankTableCandidate]:
    """Preserve a large validated partial plane without claiming completeness."""

    supplements = [
        candidate
        for candidate in candidates
        if candidate is not selected and _candidate_is_safe_material_partial_component(candidate, selected)
    ]
    if not supplements:
        return [selected]
    supplement = max(
        supplements,
        key=lambda candidate: (
            candidate.canonical_rows,
            candidate.canonical_rows / max(len(candidate.records), 1),
            candidate.field_completeness,
            candidate.balance_chain_score,
            candidate.score,
        ),
    )
    components = [selected, supplement]
    components.sort(key=lambda candidate: min(_candidate_source_pages(candidate.records)))
    canonical_total = sum(candidate.canonical_rows for candidate in components)
    authoritative_counts = {
        evidence.count
        for candidate in candidates
        if (evidence := candidate.expected_rows) is not None
        and evidence.confidence >= 0.85
        and evidence.source in _ISSUER_ROW_COUNT_SOURCES
    }
    if authoritative_counts and canonical_total > max(authoritative_counts):
        return [selected]
    return components


def _source_disjoint_candidate_components(
    selected: BankTableCandidate,
    candidates: list[BankTableCandidate],
) -> list[BankTableCandidate]:
    """Return an exact, source-disjoint page cover anchored by ``selected``.

    Acquisition routes are alternatives when they observe the same pages, but
    long digital statements can expose adjacent physical components to
    different routes.  Treating those components as competing whole-document
    candidates discards valid pages.  Composition is allowed only when every
    component is locally canonical, has complete provenance, and the chosen
    components form a contiguous exact cover of all safely represented pages.
    """

    if not _candidate_is_safe_page_component(selected):
        return [selected]
    selected_reliable = _candidate_reliable_count_coverage(selected)
    if selected_reliable is not None and selected_reliable >= 0.99:
        return [selected]

    eligible = [candidate for candidate in candidates if _candidate_is_safe_page_component(candidate)]
    page_universe = set().union(*(_candidate_source_pages(candidate.records) for candidate in eligible))
    if not page_universe or page_universe != set(range(min(page_universe), max(page_universe) + 1)):
        return _material_source_disjoint_components(selected, candidates)

    selected_pages = _candidate_source_pages(selected.records)
    missing_pages = page_universe - selected_pages
    if not missing_pages:
        return _material_source_disjoint_components(selected, candidates)

    def component_rank(candidate: BankTableCandidate) -> tuple[float, ...]:
        return (
            candidate.canonical_rows / max(len(candidate.records), 1),
            candidate.source_fragment_coverage,
            candidate.field_completeness,
            candidate.balance_chain_score,
            candidate.source_column_width,
            candidate.score,
            float(candidate.canonical_rows),
        )

    # Candidates over identical pages are alternative planes.  Retain only
    # the strongest one before solving the disjoint cover.
    by_pages: dict[frozenset[int], BankTableCandidate] = {}
    for candidate in eligible:
        if candidate is selected:
            continue
        pages = frozenset(_candidate_source_pages(candidate.records))
        if not pages or not pages.issubset(missing_pages):
            continue
        incumbent = by_pages.get(pages)
        if incumbent is None or component_rank(candidate) > component_rank(incumbent):
            by_pages[pages] = candidate

    options = list(by_pages.values())

    def exact_cover(remaining: set[int]) -> list[BankTableCandidate] | None:
        if not remaining:
            return []
        first_page = min(remaining)
        matches = [
            candidate
            for candidate in options
            if first_page in _candidate_source_pages(candidate.records)
            and _candidate_source_pages(candidate.records).issubset(remaining)
        ]
        matches.sort(key=lambda candidate: (len(_candidate_source_pages(candidate.records)), component_rank(candidate)), reverse=True)
        for candidate in matches:
            pages = _candidate_source_pages(candidate.records)
            tail = exact_cover(remaining - pages)
            if tail is not None:
                return [candidate, *tail]
        return None

    supplements = exact_cover(set(missing_pages))
    if not supplements:
        return _material_source_disjoint_components(selected, candidates)
    components = [selected, *supplements]
    components.sort(key=lambda candidate: min(_candidate_source_pages(candidate.records)))

    canonical_total = sum(candidate.canonical_rows for candidate in components)
    authoritative_counts = {
        evidence.count
        for candidate in candidates
        if (evidence := candidate.expected_rows) is not None
        and evidence.confidence >= 0.85
        and evidence.source in _ISSUER_ROW_COUNT_SOURCES
    }
    if authoritative_counts and canonical_total not in authoritative_counts:
        return [selected]
    return components


def _candidate_overlap_signature(
    transaction: dict[str, Any],
    normalizer: Callable[[dict[str, Any]], dict[str, Any]],
    standard_fields: list[str],
) -> tuple[str, str, str, str] | None:
    """Return a conservative business spine for cross-plane overlap checks."""

    try:
        normalized = ensure_canonical_normalized(normalizer(transaction), standard_fields)
    except Exception:
        return None
    date_value = str(normalized.get("date") or "").strip()
    if not date_value:
        timestamp = str(normalized.get("timestamp") or "").strip()
        match = _SOURCE_DATE_RE.search(timestamp)
        date_value = match.group(0) if match else ""
    amount = normalized.get("amount")
    balance = normalized.get("balance")
    amount_value = "" if amount in (None, "") else _compact_semantic_value(amount)
    balance_value = "" if balance in (None, "") else _compact_semantic_value(balance)
    sequence = _compact_semantic_value(normalized.get("sequence_no"))
    reference = _compact_semantic_value(normalized.get("reference"))
    if not date_value or not amount_value or not (balance_value or sequence or reference):
        return None
    return (date_value, amount_value, balance_value, sequence or reference)


def _candidate_source_bbox(transaction: dict[str, Any]) -> tuple[float, float, float, float] | None:
    source = transaction.get("_source")
    if not isinstance(source, dict):
        source = transaction.get("source")
    if not isinstance(source, dict):
        return None
    bbox = source.get("bbox") or source.get("source_bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    try:
        return tuple(float(value) for value in bbox[:4])
    except (TypeError, ValueError):
        return None


def _source_rows_can_be_the_same(
    left: dict[str, Any],
    right: dict[str, Any],
) -> bool:
    """Return whether two matching business spines can represent one row."""

    left_page = _transaction_source_page(left)
    right_page = _transaction_source_page(right)
    if left_page > 0 and right_page > 0 and left_page != right_page:
        return False
    if left_page <= 0 or right_page <= 0:
        return True
    left_bbox = _candidate_source_bbox(left)
    right_bbox = _candidate_source_bbox(right)
    if left_bbox is None or right_bbox is None:
        return True
    vertical_overlap = min(left_bbox[3], right_bbox[3]) - max(left_bbox[1], right_bbox[1])
    return vertical_overlap >= -1.0


def _native_batches_prove_disjoint(
    batches: list[tuple[list[dict[str, Any]], Callable[[dict[str, Any]], dict[str, Any]] | None]],
    plugin: Any,
) -> bool:
    """Require every native recovery component to be source-disjoint.

    Recovery components are normally alternative acquisition planes.  An
    issuer total can validate their union only after their transaction spines
    also show that no component contains or overlaps another.
    """

    indexed: list[dict[tuple[str, str, str, str], list[dict[str, Any]]]] = []
    for batch, normalize_fn in batches:
        normalizer = normalize_fn or plugin._normalize
        by_signature: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
        for transaction in batch:
            signature = _candidate_overlap_signature(transaction, normalizer, plugin.standard_fields)
            if signature is not None:
                by_signature.setdefault(signature, []).append(transaction)
        if not by_signature:
            return False
        indexed.append(by_signature)

    for left_index, left in enumerate(indexed):
        for right in indexed[left_index + 1 :]:
            for signature in left.keys() & right.keys():
                if any(
                    _source_rows_can_be_the_same(left_row, right_row)
                    for left_row in left[signature]
                    for right_row in right[signature]
                ):
                    return False
    return True


def _select_candidate(candidates: list[BankTableCandidate]) -> tuple[BankTableCandidate | None, dict[str, Any]]:
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
        evidence = candidate.expected_rows
        evidence_authority = 0.0
        if evidence is not None and evidence.confidence >= 0.85:
            # A labelled issuer count describes the whole statement.  An exact
            # 1..N sequence is also strong evidence, but it can describe only a
            # truncated table when the source parser starts the sequence over
            # or stops early.  Never let that candidate-local sequence override
            # a conflicting issuer footer/header total.
            if evidence.source in _ISSUER_ROW_COUNT_SOURCES:
                evidence_authority = 2.0
        return (
            1.0 if reliable_coverage is not None else 0.0,
            reliable_coverage if reliable_coverage is not None else 0.0,
            evidence_authority,
            float(evidence.count) if reliable_coverage is not None and evidence is not None else 0.0,
            candidate.canonical_coverage,
            candidate.source_fragment_coverage,
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
    source_faithful = [
        candidate
        for candidate in viable
        if candidate.source_column_width > selected.source_column_width + 0.25
        and (
            _candidate_reliable_count_coverage(selected) is None
            or (
                _candidate_reliable_count_coverage(candidate) is not None
                and _candidate_reliable_count_coverage(candidate)
                >= (_candidate_reliable_count_coverage(selected) or 0.0)
            )
        )
        and candidate.canonical_rows >= selected.canonical_rows
        and candidate.canonical_coverage >= selected.canonical_coverage - 0.01
        and candidate.source_fragment_coverage >= selected.source_fragment_coverage - 0.01
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
                candidate.candidate_id.startswith("parser:"),
                selection_key(candidate),
            ),
        )
    native_grid_candidates = [
        candidate
        for candidate in viable
        if candidate.candidate_id.startswith("native_wide_table")
        and candidate.native_cell_coverage >= 0.99
        and len(candidate.records) >= len(selected.records)
        and candidate.canonical_rows >= selected.canonical_rows
        and candidate.canonical_coverage >= selected.canonical_coverage - 0.01
        and candidate.source_fragment_coverage >= selected.source_fragment_coverage - 0.01
        and candidate.source_page_coverage >= selected.source_page_coverage - 0.01
        and candidate.field_completeness >= selected.field_completeness - 0.01
        and (candidate.balance_chain_score >= selected.balance_chain_score - 0.01)
        and candidate.source_column_width >= selected.source_column_width - 0.25
        and (
            _candidate_reliable_count_coverage(selected) is None
            or (
                _candidate_reliable_count_coverage(candidate) is not None
                and _candidate_reliable_count_coverage(candidate)
                >= (_candidate_reliable_count_coverage(selected) or 0.0) - 0.01
            )
        )
    ]
    if native_grid_candidates:
        # When every semantic gate is equal, explicit native cell boundaries are
        # stronger evidence than a text reconstruction that may shift wrapped
        # fragments into the following transaction.
        selected = max(
            native_grid_candidates, key=lambda candidate: (candidate.native_cell_coverage, selection_key(candidate))
        )
    if _candidate_reliable_count_coverage(selected) is None:
        fuller_candidates = [
            candidate
            for candidate in viable
            if candidate is not selected and _is_semantically_comparable_fuller_candidate(candidate, selected)
        ]
        if fuller_candidates:
            # A candidate-local denominator cannot certify that a shorter result
            # is complete.  Prefer the strongest fuller representation regardless
            # of extraction route when it preserves the selected row semantics.
            selected = max(
                fuller_candidates,
                key=lambda candidate: (candidate.canonical_rows, len(candidate.records), selection_key(candidate)),
            )
    reason = (
        f"score={selected.score:.3f}:canonical_coverage={selected.canonical_coverage:.3f}:"
        f"fragment_coverage={selected.source_fragment_coverage:.3f}:"
        f"sequence_continuity={selected.sequence_continuity:.3f}:"
        f"source_coverage={selected.source_page_coverage:.3f}:"
        f"balance_chain={selected.balance_chain_score:.3f}"
    )
    return selected, {
        "selected_candidate": selected.candidate_id,
        "candidate_counts": candidate_counts,
        "selection_reason": reason,
        "rejected_rows": len(selected.rejected_row_indexes),
        "rejection_reason": selected.rejection_reason,
    }


def _eligible_strategy_ids(detection: StyleDetectionResult, ctx: StyleContext) -> list[str]:
    """Describe the existing strategy nodes eligible for this extraction scope."""

    policy = ctx.extraction_policy
    strategy_ids = [
        f"parser:{parser_id}"
        for parser_id in dict.fromkeys([*detection.parser_chain, *_PARSER_CANDIDATE_ORDER])
        if parser_id != "kv_identity" and parser_id in _PARSERS and policy.allows_parser(parser_id)
    ]
    if ctx.prefer_context_tables:
        return strategy_ids
    if policy.allow_semantic_text:
        strategy_ids.append("semantic_text")
    if policy.allow_schema_guided_page_text:
        strategy_ids.append("schema_guided_page_text")
    if policy.allow_physical_tables:
        strategy_ids.append("physical_table")
    if policy.allow_positioned_records:
        strategy_ids.append("positioned_record_block")
    if policy.allow_evidence_atoms:
        strategy_ids.append("evidence_atom")
    if policy.allow_native_wide_tables:
        strategy_ids.append("native_wide_table")
    if policy.allow_ocr_implicit_tables:
        strategy_ids.append("ocr_implicit_table")
    return strategy_ids


def _collect_primary_table_candidates(
    detection: StyleDetectionResult,
    ctx: StyleContext,
    plugin: Any,
) -> list[BankTableCandidate]:
    """Run only the first route-allowed transaction parser.

    This preflight intentionally does not invoke any reconstruction provider.
    The unchanged eager collector below remains the fallback whenever the
    primary result cannot prove document-scope completeness.
    """

    policy = ctx.extraction_policy
    parser_id = next(
        (
            candidate_id
            for candidate_id in detection.parser_chain
            if candidate_id != "kv_identity"
            and candidate_id in _PARSERS
            and policy.allows_parser(candidate_id)
        ),
        None,
    )
    if parser_id is None:
        return []

    page_texts = [] if ctx.prefer_context_tables else page_texts_from_parse_result(ctx.parse_result)
    source_evidence = (
        RowCountEvidence.empty()
        if ctx.prefer_context_tables
        else resolve_row_count_evidence(ctx.full_text, page_texts=page_texts)
    )
    if source_evidence.source in _ROW_PLANE_COUNT_SOURCES:
        source_evidence = replace(source_evidence, confidence=min(source_evidence.confidence, 0.80))

    batch, normalize_fn = _run_parser(parser_id, ctx, plugin)
    if not batch:
        return []
    expected = _candidate_row_count_evidence(
        batch,
        _candidate_expected_rows(source_evidence),
        page_count=ctx.page_count,
        page_texts=page_texts,
    )
    return [
        _candidate_from_batch(
            candidate_id=f"parser:{parser_id}",
            transactions=batch,
            normalize_fn=normalize_fn,
            plugin=plugin,
            source=(ctx.reconstruction.source if ctx.reconstruction is not None else "canonical_table"),
            expected_rows=expected,
            extraction_confidence=0.8,
        )
    ]


def _candidate_row_source_identity(transaction: dict[str, Any]) -> tuple[Any, ...] | None:
    source = transaction.get("_source")
    if not isinstance(source, dict):
        return None
    try:
        page = int(source.get("source_page") or 0)
    except (TypeError, ValueError):
        return None
    page_range = source.get("page_range")
    if (
        page <= 0
        or not isinstance(page_range, (list, tuple))
        or len(page_range) != 2
        or page_range[0] != page
        or page_range[1] != page
    ):
        return None

    table_id = str(source.get("table_id") or source.get("source_table_id") or "")
    if table_id and "source_row_index" in source:
        try:
            row_index = int(source["source_row_index"])
        except (TypeError, ValueError):
            row_index = -1
        if row_index >= 0:
            return ("row", page, table_id, row_index)

    bbox = _candidate_source_bbox(transaction)
    if bbox is not None:
        return ("bbox", page, *bbox)
    return None


def _source_cell_signature(value: Any) -> str:
    return unicodedata.normalize("NFKC", str(value or "")).strip()


def _source_row_signature(values: Any) -> tuple[str, ...] | None:
    if not isinstance(values, list):
        return None
    return tuple(_source_cell_signature(value) for value in values)


def _compact_source_cell(value: Any) -> str:
    return re.sub(r"\s+", "", _source_cell_signature(value))


def _physical_row_raw_indexes(row: Any) -> set[int] | None:
    """Return explicit raw-row bindings, or ``None`` when none were supplied."""

    refs = [
        *(getattr(row, "source_cell_refs", None) or []),
        *(
            ref
            for cell in (getattr(row, "cells", None) or [])
            for ref in (getattr(cell, "source_cell_refs", None) or [])
        ),
    ]
    indexes: set[int] = set()
    saw_raw_row = False
    for ref in refs:
        if not isinstance(ref, dict) or ref.get("raw_row") is None:
            continue
        saw_raw_row = True
        try:
            raw_row = int(ref["raw_row"])
        except (TypeError, ValueError):
            return set()
        if raw_row < 0:
            return set()
        indexes.add(raw_row)
    return indexes if saw_raw_row else None


def _physical_body_rows_match_raw_rows(
    table: Any,
    raw_rows: list[list[Any]],
    *,
    header_raw_row: int,
    canonical_width: int,
) -> bool:
    """Bind the physical recovery plane to its raw matrix without guessing."""

    represented: set[int] = set()
    body_rows = list(getattr(table, "rows", None) or [])
    for row_index, row in enumerate(body_rows):
        cells = list(getattr(row, "cells", None) or [])
        if len(cells) != canonical_width:
            return False
        values = [
            getattr(cell, "cleaned", None) or getattr(cell, "text", "") or ""
            for cell in cells
        ]
        explicit_indexes = _physical_row_raw_indexes(row)
        if explicit_indexes is not None:
            if len(explicit_indexes) != 1:
                return False
            raw_row_index = next(iter(explicit_indexes))
        else:
            raw_row_index = row_index + 1
        if (
            raw_row_index == header_raw_row
            or raw_row_index < 0
            or raw_row_index >= len(raw_rows)
            or raw_row_index in represented
            or tuple(_source_cell_signature(value) for value in values)
            != tuple(_source_cell_signature(value) for value in raw_rows[raw_row_index])
        ):
            return False
        represented.add(raw_row_index)

    return represented == (set(range(len(raw_rows))) - {header_raw_row})


def _sealed_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if x1 < x0 or y1 < y0:
        return None
    return (x0, y0, x1, y1)


def _has_sealed_transaction_headers(values: list[str]) -> bool:
    compact_values = [re.sub(r"\s+", "", _source_cell_signature(value)).casefold() for value in values]
    has_date_header = any(
        marker in value
        for value in compact_values
        for marker in ("交易日期", "记账日期", "交易时间", "日期", "transactiondate", "date")
    )
    has_amount_header = any(
        marker in value
        for value in compact_values
        for marker in (
            "交易金额",
            "发生额",
            "支出金额",
            "收入金额",
            "借方金额",
            "贷方金额",
            "transactionamount",
            "amount",
        )
    )
    return has_date_header and has_amount_header


def _has_sealed_transaction_role_fact(values: list[str]) -> bool:
    return any(
        marker in re.sub(r"\s+", "", _source_cell_signature(value)).casefold()
        for value in values
        for marker in (
            "对方账号",
            "对方账户",
            "对方户名",
            "对方行名",
            "对手名称",
            "交易用途",
            "交易摘要",
            "交易流水号",
            "交易渠道",
            "counterparty",
            "reference",
        )
    )


def _looks_like_sealed_positioned_record(value: str) -> bool:
    return (
        len([line for line in str(value or "").splitlines() if line.strip()]) >= 4
        and bool(_SOURCE_DATE_RE.search(value))
        and len(_SEALED_MONEY_RE.findall(value)) >= 2
    )


def _sealed_alternative_planes_match_physical(
    ctx: StyleContext,
    physical_rows_by_position: dict[tuple[int, int], list[list[Any]]],
    normalized_headers: tuple[str, ...],
    canonical_width: int,
) -> bool:
    """Inventory sealed inputs consumed by skipped reconstruction strategies.

    The evidence-plane table index must be a row/column-identical view of the
    physical tables, and all positioned text inside those table bands must be
    owned by their cell geometry.  Cheap structural checks also reject a richer
    table advertised only by the semantic-text or positioned-atom planes.  No
    reconstruction provider is invoked while building this certificate.
    """

    payload = evidence_payload(ctx.parse_result)
    indexes = payload.get("indexes") if isinstance(payload, dict) else None
    indexed_tables = indexes.get("table_candidates") if isinstance(indexes, dict) else None
    text_atoms = payload.get("text_atoms") if isinstance(payload, dict) else None
    if not isinstance(indexed_tables, list) or not indexed_tables or not isinstance(text_atoms, list):
        return False

    atoms_by_id: dict[str, dict[str, Any]] = {}
    for atom in text_atoms:
        if not isinstance(atom, dict):
            return False
        atom_id = str(atom.get("id") or "")
        if not atom_id or atom_id in atoms_by_id:
            return False
        atoms_by_id[atom_id] = atom

    positioned_blocks_by_page: dict[str, list[str]] = {}
    for page in getattr(ctx.parse_result, "pages", None) or []:
        try:
            page_number = int(getattr(page, "page_number", 0) or 0)
        except (TypeError, ValueError):
            return False
        if page_number <= 0:
            return False
        page_id = f"page:{page_number:04d}"
        for block in getattr(page, "texts", None) or []:
            content = str(getattr(block, "content", "") or "").strip()
            bbox = getattr(block, "bbox", None)
            if not content or not isinstance(bbox, list) or len(bbox) < 4:
                continue
            evidence_ids = [str(item or "") for item in (getattr(block, "evidence_ids", None) or [])]
            if not evidence_ids or any(not item or item not in atoms_by_id for item in evidence_ids):
                return False
            atom_text = "".join(str(atoms_by_id[item].get("text") or "") for item in evidence_ids)
            if _compact_source_cell(content) != _compact_source_cell(atom_text):
                return False
            positioned_blocks_by_page.setdefault(page_id, []).append(content)

    for block_texts in positioned_blocks_by_page.values():
        if (
            _has_sealed_transaction_role_fact(block_texts)
            or sum(_looks_like_sealed_positioned_record(text) for text in block_texts) >= 2
        ):
            return False
        block_text = "\n".join(block_texts)
        if (
            _has_sealed_transaction_headers(block_texts)
            and len(_SOURCE_DATE_RE.findall(block_text)) >= 2
            and len(_SEALED_MONEY_RE.findall(block_text)) >= 2
        ):
            return False

    candidates_by_position: dict[tuple[int, int], dict[str, Any]] = {}
    for indexed_table in indexed_tables:
        if not isinstance(indexed_table, dict):
            return False
        try:
            key = (int(indexed_table.get("page_number") or 0), int(indexed_table["table_index"]))
        except (KeyError, TypeError, ValueError):
            return False
        if key[0] <= 0 or key in candidates_by_position:
            return False
        candidates_by_position[key] = indexed_table
    if set(candidates_by_position) != set(physical_rows_by_position):
        return False

    table_bands: list[tuple[str, float, float, set[str]]] = []
    for position, raw_rows in physical_rows_by_position.items():
        indexed_table = candidates_by_position[position]
        indexed_rows = indexed_table.get("rows")
        if not isinstance(indexed_rows, list) or len(indexed_rows) != len(raw_rows):
            return False
        for indexed_row, raw_row in zip(indexed_rows, raw_rows, strict=True):
            if (
                _source_row_signature(indexed_row) is None
                or len(indexed_row) != canonical_width
                or _source_row_signature(indexed_row) != _source_row_signature(raw_row)
            ):
                return False

        geometry = indexed_table.get("geometry")
        cell_evidence = geometry.get("cell_evidence_ids") if isinstance(geometry, dict) else None
        if (
            not isinstance(cell_evidence, list)
            or len(cell_evidence) != len(raw_rows)
            or any(not isinstance(row, list) or len(row) != canonical_width for row in cell_evidence)
        ):
            return False

        covered_ids: set[str] = set()
        for row_index, (raw_row, evidence_row) in enumerate(
            zip(raw_rows, cell_evidence, strict=True)
        ):
            is_canonical_header = _source_row_signature(raw_row) == normalized_headers
            header_atom_ids: list[str] = []
            for col_index, evidence_ids in enumerate(evidence_row):
                if not isinstance(evidence_ids, list):
                    return False
                atom_texts: list[str] = []
                for evidence_id in evidence_ids:
                    atom_id = str(evidence_id or "")
                    atom = atoms_by_id.get(atom_id)
                    if not atom_id or atom is None:
                        return False
                    covered_ids.add(atom_id)
                    if atom_id not in header_atom_ids:
                        header_atom_ids.append(atom_id)
                    atom_texts.append(str(atom.get("text") or ""))
                # Merged source headers do not always have a one-cell/one-atom
                # encoding.  Body rows, including promoted transaction headers,
                # must retain exact text composition.
                if not is_canonical_header and _compact_source_cell("".join(atom_texts)) != _compact_source_cell(
                    raw_row[col_index]
                ):
                    return False
            if is_canonical_header and _compact_source_cell(
                "".join(str(atoms_by_id[atom_id].get("text") or "") for atom_id in header_atom_ids)
            ) != _compact_source_cell("".join(str(value or "") for value in raw_row)):
                return False

        bbox = _sealed_bbox(indexed_table.get("bbox"))
        page_id = str(indexed_table.get("page_id") or f"page:{position[0]:04d}")
        if bbox is None or not page_id:
            return False
        table_bands.append((page_id, bbox[1], bbox[3], covered_ids))

    outside_by_page: dict[str, list[str]] = {}
    for atom_id, atom in atoms_by_id.items():
        page_id = str(atom.get("page_id") or "")
        bbox = _sealed_bbox(atom.get("bbox"))
        matching_bands = [
            covered
            for band_page, y0, y1, covered in table_bands
            if band_page == page_id and bbox is not None and y0 <= (bbox[1] + bbox[3]) / 2.0 <= y1
        ]
        if matching_bands:
            if not any(atom_id in covered for covered in matching_bands):
                return False
            continue
        outside_by_page.setdefault(page_id, []).append(str(atom.get("text") or ""))

    # A second positioned ledger outside the known table bands is not safe to
    # skip.  Requiring repeated date/time/amount anchors avoids treating ordinary
    # statement identity and totals as a transaction plane.
    for outside_texts in outside_by_page.values():
        if (
            _has_sealed_transaction_role_fact(outside_texts)
            or sum(_looks_like_sealed_positioned_record(text) for text in outside_texts) >= 2
        ):
            return False
        dates = sum(bool(_SOURCE_DATE_RE.search(text)) for text in outside_texts)
        times = sum(bool(_SEALED_TIME_RE.search(text)) for text in outside_texts)
        amounts = sum(bool(_SEALED_MONEY_RE.search(text)) for text in outside_texts)
        if dates >= 2 and amounts >= 2 and (times >= 2 or _has_sealed_transaction_headers(outside_texts)):
            return False

    non_pipe_lines: list[str] = []
    for line in str(ctx.full_text or "").splitlines():
        if line.count("|") >= 2:
            cells = line.strip().strip("|").split("|")
            if len(cells) > canonical_width:
                return False
        else:
            non_pipe_lines.append(line)
    non_pipe_text = "\n".join(non_pipe_lines)
    row_like_lines = sum(
        bool(_SOURCE_DATE_RE.search(line) and len(_SEALED_MONEY_RE.findall(line)) >= 2)
        for line in non_pipe_lines
    )
    if row_like_lines >= 2 or (
        len(_SOURCE_DATE_RE.findall(non_pipe_text)) >= 2
        and len(_SEALED_TIME_RE.findall(non_pipe_text)) >= 2
        and len(_SEALED_MONEY_RE.findall(non_pipe_text)) >= 2
    ):
        return False
    return True


def _candidate_preserves_canonical_source_columns(
    candidate: BankTableCandidate,
    ctx: StyleContext,
) -> bool:
    """Prove a row/column/value bijection with the sealed physical source.

    Core normalized fields cannot certify that an alternative would not retain
    additional source business facts.  A primary candidate may stop deployment
    only when every physical source column survives under its original canonical
    header, every value matches its referenced source cell, and every non-header
    physical row is represented exactly once.  This certificate is deliberately
    bounded to the sealed ParseResult: source-PDF reopening remains a recovery
    strategy for cases where that sealed evidence cannot establish completeness.
    """

    header_signatures: set[tuple[str, ...]] = set()
    for table in ctx.tables:
        if not table:
            continue
        headers = tuple(str(header or "") for header in table[0])
        if (
            len(headers) >= 3
            and all(header and not header.startswith("_") for header in headers)
            and len(set(headers)) == len(headers)
        ):
            header_signatures.add(headers)
    if len(header_signatures) != 1:
        return False
    canonical_headers = next(iter(header_signatures))
    canonical_width = len(canonical_headers)

    normalized_headers = tuple(_source_cell_signature(header) for header in canonical_headers)
    physical_tables: dict[tuple[int, str], Any] = {}
    physical_rows_by_position: dict[tuple[int, int], list[list[Any]]] = {}
    promoted_header_rows: set[tuple[int, str, int]] = set()
    canonical_header_tables = 0
    for page in getattr(ctx.parse_result, "pages", None) or []:
        try:
            page_number = int(
                getattr(page, "source_page_number", 0) or getattr(page, "page_number", 0) or 0
            )
        except (TypeError, ValueError):
            return False
        if page_number <= 0:
            return False
        for table_index, table in enumerate(getattr(page, "tables", None) or []):
            table_id = str(getattr(table, "table_id", "") or f"pt_{page_number}_{table_index}")
            table_key = (page_number, table_id)
            if table_key in physical_tables:
                return False
            metadata = getattr(table, "metadata", None) or {}
            raw_rows = metadata.get("raw_rows") if isinstance(metadata, dict) else None
            if not isinstance(raw_rows, list) or not raw_rows:
                return False
            if any(not isinstance(row, list) or len(row) != canonical_width for row in raw_rows):
                return False
            table_headers = tuple(str(header or "") for header in getattr(table, "headers", None) or [])
            if len(table_headers) != canonical_width:
                return False
            geometry = metadata.get("geometry")
            if isinstance(geometry, dict):
                for plane_name in ("cell_bboxes", "cell_evidence_ids"):
                    plane = geometry.get(plane_name)
                    if plane is None:
                        continue
                    if (
                        not isinstance(plane, list)
                        or len(plane) != len(raw_rows)
                        or any(not isinstance(row, list) or len(row) != canonical_width for row in plane)
                    ):
                        return False
            normalized_table_headers = tuple(_source_cell_signature(header) for header in table_headers)
            matching_header_rows = [
                raw_row_index
                for raw_row_index, raw_values in enumerate(raw_rows)
                if tuple(_source_cell_signature(value) for value in raw_values)
                == normalized_table_headers
            ]
            if len(matching_header_rows) != 1:
                return False
            header_raw_row = matching_header_rows[0]
            if not _physical_body_rows_match_raw_rows(
                table,
                raw_rows,
                header_raw_row=header_raw_row,
                canonical_width=canonical_width,
            ):
                return False
            if normalized_table_headers == normalized_headers:
                canonical_header_tables += 1
            else:
                promoted_header_rows.add((page_number, table_id, header_raw_row))
            physical_tables[table_key] = table
            physical_rows_by_position[(page_number, table_index)] = raw_rows
    if not physical_tables or canonical_header_tables <= 0:
        return False
    if not _sealed_alternative_planes_match_physical(
        ctx,
        physical_rows_by_position,
        normalized_headers,
        canonical_width,
    ):
        return False

    represented_source_rows: set[tuple[int, str, int]] = set()
    for transaction in candidate.records:
        business_headers = tuple(str(key) for key in transaction if not str(key).startswith("_"))
        if business_headers != canonical_headers:
            return False
        source = transaction.get("_source")
        refs = source.get("source_cell_refs") if isinstance(source, dict) else None
        if not isinstance(refs, list) or len(refs) != canonical_width:
            return False
        try:
            source_page = int(source.get("source_page") or 0)
            source_row = int(source["source_row_index"])
        except (KeyError, TypeError, ValueError):
            return False
        source_table = str(source.get("table_id") or source.get("source_table_id") or "")
        if source_page <= 0 or source_row < 0 or not source_table:
            return False
        physical_table = physical_tables.get((source_page, source_table))
        if physical_table is None:
            return False
        source_columns: dict[int, int] = {}
        for ref in refs:
            if not isinstance(ref, dict):
                return False
            try:
                ref_page = int(ref.get("page") or ref.get("source_page") or 0)
                ref_row = int(ref.get("row") if ref.get("row") is not None else ref.get("source_row_index"))
                ref_raw_row = int(ref.get("raw_row")) if ref.get("raw_row") is not None else -1
                ref_col = int(ref.get("col") if ref.get("col") is not None else ref.get("source_col_index"))
            except (TypeError, ValueError):
                return False
            ref_table = str(ref.get("table_id") or ref.get("source_table_id") or "")
            if (
                ref_page != source_page
                or ref_table != source_table
                or source_row not in {ref_row, ref_raw_row}
            ):
                return False
            if ref_col in source_columns or ref_raw_row < 0:
                return False
            source_columns[ref_col] = ref_raw_row
        if set(source_columns) != set(range(canonical_width)):
            return False

        raw_row_indexes = set(source_columns.values())
        if len(raw_row_indexes) != 1:
            return False
        raw_row_index = next(iter(raw_row_indexes))
        source_identity = (source_page, source_table, raw_row_index)
        if source_identity in represented_source_rows:
            return False
        represented_source_rows.add(source_identity)
        raw_rows = physical_table.metadata.get("raw_rows")
        if raw_row_index >= len(raw_rows):
            return False
        raw_values = raw_rows[raw_row_index]
        if any(
            _source_cell_signature(transaction.get(header)) != _source_cell_signature(raw_values[col_index])
            for col_index, header in enumerate(canonical_headers)
        ):
            return False

        geometry = physical_table.metadata.get("geometry")
        cell_evidence = geometry.get("cell_evidence_ids") if isinstance(geometry, dict) else None
        if isinstance(cell_evidence, list):
            row_evidence = {
                str(evidence_id)
                for cell_ids in cell_evidence[raw_row_index]
                if isinstance(cell_ids, list)
                for evidence_id in cell_ids
                if str(evidence_id)
            }
            candidate_evidence = {
                str(item) for item in (source.get("evidence_ids") or []) if str(item)
            }
            if row_evidence and not row_evidence.issubset(candidate_evidence):
                return False

    if not promoted_header_rows.issubset(represented_source_rows):
        return False
    for (page_number, table_id), table in physical_tables.items():
        for raw_row_index, raw_values in enumerate(table.metadata["raw_rows"]):
            if (page_number, table_id, raw_row_index) in represented_source_rows:
                continue
            if tuple(_source_cell_signature(value) for value in raw_values) != normalized_headers:
                return False
    return True


def _prove_primary_candidate_complete(
    candidate: BankTableCandidate | None,
    detection: StyleDetectionResult,
    ctx: StyleContext,
) -> CandidateCompletionProof:
    """Fail closed unless the primary parser proves a complete canonical document."""

    if candidate is None:
        return CandidateCompletionProof("unknown", "primary_parser_returned_no_candidate")
    if ctx.prefer_context_tables:
        return CandidateCompletionProof("unknown", "local_context_cannot_prove_document_scope")
    if ctx.extraction_route is not BankExtractionRoute.DIGITAL:
        return CandidateCompletionProof("unknown", "scanned_route_requires_reconstruction_comparison")
    reconstruction = ctx.reconstruction
    if reconstruction is None or reconstruction.source != "canonical_table":
        return CandidateCompletionProof("unknown", "primary_source_is_not_canonical_table")
    if reconstruction.pipe_parse_failed:
        return CandidateCompletionProof("unknown", "earlier_table_reconstruction_failed")
    if detection.confidence < 0.75:
        return CandidateCompletionProof("unknown", "style_detection_below_completion_threshold")

    logical_tables = get_logical_tables(ctx.parse_result) if ctx.parse_result is not None else []
    if len(logical_tables) > 1:
        return CandidateCompletionProof("unknown", "multiple_logical_scopes_require_comparison")
    if any(not getattr(table, "quality_passed", True) for table in logical_tables):
        return CandidateCompletionProof("unknown", "logical_table_quality_not_proven")
    if statement_scope_count(ctx.parse_result) != 1:
        return CandidateCompletionProof("unknown", "single_statement_scope_not_proven")
    if not _candidate_preserves_canonical_source_columns(candidate, ctx):
        return CandidateCompletionProof("unknown", "canonical_source_columns_not_conserved")

    evidence = candidate.expected_rows
    if (
        evidence is None
        or evidence.count <= 0
        or evidence.confidence < 0.85
        or evidence.source not in _ISSUER_ROW_COUNT_SOURCES
    ):
        return CandidateCompletionProof("unknown", "issuer_row_count_is_not_authoritative")
    row_count = len(candidate.records)
    if not (
        evidence.count
        == row_count
        == candidate.canonical_rows
        == candidate.directional_rows
    ):
        return CandidateCompletionProof("unknown", "primary_rows_do_not_match_issuer_count")
    if (
        candidate.canonical_coverage < 0.999
        or candidate.source_page_coverage < 0.999
        or candidate.field_completeness < 0.999
    ):
        return CandidateCompletionProof("unknown", "primary_quality_coverage_is_incomplete")
    if (
        candidate.semantic_anomaly_rows
        or candidate.rejected_row_indexes
        or candidate.rejection_reason
        or candidate.source_role_swap_ratio > 0.0
    ):
        return CandidateCompletionProof("unknown", "primary_rows_have_semantic_anomalies")

    source_identities = [_candidate_row_source_identity(row) for row in candidate.records]
    if any(identity is None for identity in source_identities):
        return CandidateCompletionProof("unknown", "row_local_provenance_is_incomplete")
    if len(set(source_identities)) != len(source_identities):
        return CandidateCompletionProof("unknown", "duplicate_row_local_provenance")
    if any(
        _candidate_source_page_rewinds(previous, current)
        for previous, current in zip(candidate.records, candidate.records[1:])
    ):
        return CandidateCompletionProof("unknown", "source_page_order_rewinds")

    if ctx.extraction_route is BankExtractionRoute.DIGITAL and ctx.parse_result is not None:
        physical_rows = physical_transaction_row_estimate(ctx.parse_result)
        if physical_rows > 0 and physical_rows != row_count:
            return CandidateCompletionProof("unknown", "physical_row_census_conflicts")

    return CandidateCompletionProof(
        "proven",
        f"issuer_count={evidence.count}:canonical_rows={candidate.canonical_rows}:provenance=complete",
    )


def _collect_table_candidates(
    detection: StyleDetectionResult,
    ctx: StyleContext,
    plugin: Any,
) -> list[BankTableCandidate]:
    """Materialize the candidates allowed by the context's acquisition policy."""
    context_tables_only = bool(ctx.prefer_context_tables)
    scope_page_texts = [] if context_tables_only else page_texts_from_parse_result(ctx.parse_result)
    source_evidence = (
        RowCountEvidence(0, "none", 0.0)
        if context_tables_only
        else resolve_row_count_evidence(ctx.full_text, page_texts=scope_page_texts)
    )
    policy = ctx.extraction_policy
    if not context_tables_only and policy.route is BankExtractionRoute.DIGITAL:
        count, source, confidence = recovered_native_datetime_row_evidence(
            ctx.parse_result,
            source_route=policy.route.value,
        )
        if count > 0 and source and confidence >= 0.85 and source in _INDEPENDENT_ROW_COUNT_SOURCES:
            source_evidence = RowCountEvidence(count=count, source=source, confidence=confidence)
    atom_tables = (
        recover_evidence_atom_bank_tables(ctx.parse_result, source_route=policy.route.value)
        if policy.allow_evidence_atoms and not context_tables_only
        else []
    )
    atom_row_evidence = (
        _evidence_atom_expected_rows(
            ctx.parse_result,
            atom_tables,
            source_route=policy.route.value,
        )
        if atom_tables
        else None
    )
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
        candidate_expected = _candidate_row_count_evidence(
            batch,
            expected or _candidate_expected_rows(source_evidence),
            page_count=ctx.page_count,
            page_texts=scope_page_texts,
        )
        candidates.append(
            _candidate_from_batch(
                candidate_id=candidate_id,
                transactions=batch,
                normalize_fn=normalize_fn,
                plugin=plugin,
                source=source,
                expected_rows=candidate_expected,
                extraction_confidence=confidence,
            )
        )

    for parser_id in dict.fromkeys([*detection.parser_chain, *_PARSER_CANDIDATE_ORDER]):
        if parser_id == "kv_identity" or parser_id not in _PARSERS or not policy.allows_parser(parser_id):
            continue
        batch, normalize_fn = _run_parser(parser_id, ctx, plugin)
        add(
            f"parser:{parser_id}",
            batch,
            normalize_fn,
            (ctx.reconstruction.source if ctx.reconstruction is not None else "canonical_table"),
            confidence=0.8,
        )

    if context_tables_only:
        # ``prefer_context_tables`` is a top-level acquisition boundary, not
        # merely a hint to grid_standard. BLO uses it for a strictly guarded
        # continuation fragment; admitting document-wide physical/native/OCR
        # planes here would contaminate that local batch before final dedupe.
        return candidates

    semantic_tables = _semantic_text_table_candidates(ctx.full_text) if policy.allow_semantic_text else []
    for index, semantic_table in enumerate(semantic_tables):
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

    schema_guided = (
        recover_schema_guided_page_text(ctx.parse_result, source_route=policy.route.value)
        if policy.allow_schema_guided_page_text
        else None
    )
    if schema_guided is not None and schema_guided.records:
        add(
            "schema_guided_page_text",
            schema_guided.records,
            lambda transaction: signed_amount.normalize_record(transaction, plugin),
            "schema_guided_page_text",
            _candidate_expected_rows(
                source_evidence,
                count=schema_guided.expected_rows,
                source="schema_guided_page_text_rows",
                confidence=0.80,
            ),
            confidence=0.92,
        )

    physical_tables = (
        collect_physical_tables_from_parse_result(ctx.parse_result) if policy.allow_physical_tables else []
    )
    physical_expected = physical_transaction_row_estimate(ctx.parse_result) if physical_tables else 0
    if physical_tables:
        physical_ctx = replace(ctx, tables=physical_tables, prefer_context_tables=True)
        batch, normalize_fn = _run_parser("grid_standard", physical_ctx, plugin)
        _attach_recovered_sources(
            batch,
            collect_physical_table_row_sources_from_parse_result(ctx.parse_result),
        )
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

    positioned = (
        recover_positioned_record_block_bank_tables(ctx.parse_result, source_route=policy.route.value)
        if policy.allow_positioned_records
        else None
    )
    if positioned is not None and positioned.tables:
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
        _attach_recovered_sources(
            batch,
            recovered_evidence_atom_row_sources(
                ctx.parse_result,
                source_route=policy.route.value,
            ),
        )
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

    wide_tables = recover_wide_bank_tables(ctx.parse_result, ctx.full_text) if policy.allow_native_wide_tables else []
    if wide_tables:
        wide_parser_id = (
            "signed_amount"
            if detection.primary_style == "signed_amount" or signed_amount.table_has_signed_amount_cells(wide_tables)
            else "grid_standard"
        )
        # Native and borderless recovery streams are whole-table alternatives,
        # not rows to union implicitly.  Materialize each separately so source
        # evidence and semantic quality can choose the faithful representation.
        # Keep a combined candidate only for documents whose tables are truly
        # disjoint statement sections; an independent total can then validate
        # that larger interpretation.
        native_batches: list[tuple[list[dict[str, Any]], Callable[[dict[str, Any]], dict[str, Any]] | None]] = []
        for index, table in enumerate(wide_tables):
            candidate_id = f"native_wide_table:{index}"
            candidate_tables = [table]
            wide_ctx = replace(ctx, tables=candidate_tables, prefer_context_tables=True)
            batch, normalize_fn = _run_parser(wide_parser_id, wide_ctx, plugin)
            native_batches.append((batch, normalize_fn))
            add(
                candidate_id,
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
        issuer_proves_union = (
            source_evidence.count > 0
            and source_evidence.confidence >= 0.85
            and source_evidence.source in _ISSUER_ROW_COUNT_SOURCES
        )
        if len(wide_tables) > 1 and issuer_proves_union and _native_batches_prove_disjoint(native_batches, plugin):
            wide_ctx = replace(ctx, tables=wide_tables, prefer_context_tables=True)
            batch, normalize_fn = _run_parser(wide_parser_id, wide_ctx, plugin)
            if len(batch) == source_evidence.count:
                add(
                    "native_wide_table:combined",
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

    ocr_tables = (
        recover_ocr_implicit_ledger_tables(ctx.parse_result, ctx.full_text) if policy.allow_ocr_implicit_tables else []
    )
    if ocr_tables:
        ocr_count, ocr_source, ocr_confidence = recovered_ocr_implicit_row_evidence(ctx.parse_result)
        if ocr_source in _ROW_PLANE_COUNT_SOURCES:
            ocr_confidence = min(ocr_confidence, 0.80)
        ocr_expected = (
            RowCountEvidence(ocr_count, ocr_source, ocr_confidence)
            if (
                ocr_count > 0 and ocr_source and ocr_confidence >= 0.85 and ocr_source in _INDEPENDENT_ROW_COUNT_SOURCES
            )
            else _candidate_expected_rows(
                source_evidence,
                count=sum(max(len(table) - 1, 0) for table in ocr_tables),
                source="ocr_recovered_rows",
                confidence=0.70,
            )
        )
        ocr_ctx = replace(ctx, tables=ocr_tables, prefer_context_tables=True)
        batch, normalize_fn = _run_parser("grid_standard", ocr_ctx, plugin)
        add(
            "ocr_implicit_table",
            batch,
            normalize_fn,
            "ocr_implicit_table",
            ocr_expected,
            confidence=0.70,
        )
    return candidates


class BankStyleParserRegistry:
    """Execute parser_chain and produce v2.0 records."""

    def __init__(self, *, adaptive: bool = True) -> None:
        self._adaptive = adaptive
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
        if "kv_identity" in detection.parser_chain:
            identity_fields = kv_identity.enrich_identity_fields(
                ctx,
                identity_fields,
                plugin.identity_fields,
            )

        eligible_strategies = _eligible_strategy_ids(detection, ctx)
        completion = CandidateCompletionProof("unknown", "adaptive_deployment_disabled")
        deployment_mode = "eager_forced"
        attempted_strategies = list(eligible_strategies)
        skipped_strategies: list[str] = []

        if self._adaptive:
            preflight_ctx = replace(
                ctx,
                reconstruction=(replace(ctx.reconstruction) if ctx.reconstruction is not None else None),
            )
            primary_candidates = _collect_primary_table_candidates(detection, preflight_ctx, plugin)
            primary_candidate = primary_candidates[0] if primary_candidates else None
            completion = _prove_primary_candidate_complete(primary_candidate, detection, preflight_ctx)
            primary_strategy = (
                primary_candidate.candidate_id
                if primary_candidate is not None
                else next((item for item in eligible_strategies if item.startswith("parser:")), "")
            )
            if completion.proven:
                candidates = primary_candidates
                ctx.reconstruction = preflight_ctx.reconstruction
                deployment_mode = "lazy_primary"
                attempted_strategies = [primary_strategy] if primary_strategy else []
                skipped_strategies = [
                    strategy_id for strategy_id in eligible_strategies if strategy_id != primary_strategy
                ]
            else:
                # This is intentionally the pre-adaptive collector, invoked as
                # one unit so UNKNOWN retains the existing candidate universe,
                # scoring, and winning-record behavior.
                candidates = _collect_table_candidates(detection, ctx, plugin)
                deployment_mode = "eager_fallback"
                attempted_strategies = list(dict.fromkeys([primary_strategy, *eligible_strategies]))
                attempted_strategies = [item for item in attempted_strategies if item]
        else:
            candidates = _collect_table_candidates(detection, ctx, plugin)

        selected, diagnostics = _select_candidate(candidates)
        diagnostics.update(
            {
                "deployment_mode": deployment_mode,
                "completion_state": completion.state,
                "completion_reason": completion.reason,
                "attempted_strategies": attempted_strategies,
                "skipped_strategies": skipped_strategies,
            }
        )
        self.last_selection_diagnostics = diagnostics
        transactions: list[dict[str, Any]] = []
        normalize_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None
        record_normalizers: dict[int, Callable[[dict[str, Any]], dict[str, Any]] | None] = {}
        if selected is not None:
            components = _source_disjoint_candidate_components(selected, candidates)
            rejected_count = sum(len(candidate.rejected_row_indexes) for candidate in components)
            transactions = [record for candidate in components for record in candidate.records]
            for candidate in components:
                for record in candidate.records:
                    record_normalizers[id(record)] = candidate.normalize_fn
            normalize_fn = selected.normalize_fn
            selected_expected = selected.expected_rows
            deployment_source = selected.source
            if len(components) > 1:
                component_ids = [candidate.candidate_id for candidate in components]
                canonical_total = sum(candidate.canonical_rows for candidate in components)
                composed_pages = set().union(
                    *(_candidate_source_pages(candidate.records) for candidate in components)
                )
                exact_contiguous_cover = (
                    all(_candidate_is_safe_page_component(candidate) for candidate in components)
                    and composed_pages
                    == set(range(min(composed_pages), max(composed_pages) + 1))
                )
                matching_evidence = [
                    evidence
                    for candidate in candidates
                    if (evidence := candidate.expected_rows) is not None and evidence.count == canonical_total
                ]
                if matching_evidence:
                    selected_expected = max(matching_evidence, key=lambda evidence: evidence.confidence)
                deployment_source = "source_disjoint_page_components"
                diagnostics.update(
                    {
                        "selected_candidate": f"composite:{'+'.join(component_ids)}",
                        "composed_candidates": component_ids,
                        "composition_reason": (
                            "contiguous_exact_source_page_cover"
                            if exact_contiguous_cover
                            else "material_source_disjoint_partial_recovery"
                        ),
                        "composed_canonical_rows": canonical_total,
                    }
                )
            if selected_expected is not None:
                diagnostics.update(
                    {
                        "selected_expected_rows": selected_expected.count,
                        "selected_expected_source": selected_expected.source,
                        "selected_expected_confidence": selected_expected.confidence,
                    }
                )
            if ctx.reconstruction is not None:
                retained_evidence_source = ctx.reconstruction.expected_evidence_source
                retained_evidence_confidence = ctx.reconstruction.expected_evidence_confidence
                retained_expected_rows = ctx.reconstruction.expected_primary_rows
                if retained_evidence_source in _ROW_PLANE_COUNT_SOURCES:
                    retained_evidence_confidence = min(retained_evidence_confidence, 0.80)
                    retained_expected_rows = 0
                if (
                    selected_expected is not None
                    and selected_expected.source in _ROW_PLANE_COUNT_SOURCES
                    and not retained_evidence_source
                    and retained_expected_rows == selected_expected.count
                ):
                    retained_evidence_source = selected_expected.source
                    retained_evidence_confidence = min(selected_expected.confidence, 0.80)
                    retained_expected_rows = 0
                ctx.reconstruction = replace(
                    ctx.reconstruction,
                    source=deployment_source,
                    expected_primary_rows=(
                        selected_expected.count
                        if selected_expected is not None and selected_expected.confidence >= 0.85
                        else retained_expected_rows
                    ),
                    expected_evidence_source=(
                        selected_expected.source
                        if selected_expected is not None and selected_expected.confidence >= 0.85
                        else retained_evidence_source
                    ),
                    expected_evidence_confidence=(
                        selected_expected.confidence
                        if selected_expected is not None and selected_expected.confidence >= 0.85
                        else retained_evidence_confidence
                    ),
                    pipe_parse_failed=False,
                )
            logger.info(
                "[BankStyleRegistry] selected candidate=%s rows=%d rejected=%d reason=%s",
                diagnostics["selected_candidate"],
                len(transactions),
                rejected_count,
                diagnostics["selection_reason"],
            )

        if not transactions and not candidates:
            # Candidate collection already executes every route-eligible whole
            # extraction strategy.  If candidates exist but semantic validation
            # rejects all of them, rerunning a parser here would bypass those
            # gates and emit the exact rows we just proved unreliable.
            fallback_id = "borderless_ocr" if ctx.extraction_route is BankExtractionRoute.SCANNED else "grid_standard"
            transactions, normalize_fn = _run_parser(fallback_id, ctx, plugin)

        if normalize_fn is None:

            def _plugin_normalize(raw):
                return plugin._normalize(raw)

            normalize_fn = _plugin_normalize

        def _normalizer_for(raw: dict[str, Any]) -> Callable[[dict[str, Any]], dict[str, Any]]:
            per_record = record_normalizers.get(id(raw), normalize_fn)
            return per_record or plugin._normalize

        def _normalize(raw: dict[str, Any]) -> dict[str, Any]:
            normalized = _normalizer_for(raw)(raw)
            # ``amount_cny`` is not a second source business fact.  It is only
            # valid when an actual FX conversion was evidenced, which none of
            # the bank style normalizers performs.  Keep the source currency
            # and amount, and do not manufacture a duplicate CNY column.
            normalized.pop("amount_cny", None)
            return ensure_canonical_normalized(normalized, plugin.standard_fields)

        def _canonical_raw(raw: dict[str, Any], normalized: dict[str, Any]) -> dict[str, Any]:
            if _normalizer_for(raw) is compact_merged.normalize_record:
                return compact_merged.canonical_raw_values(raw, normalized, plugin)
            return plugin._canonical_raw_values(raw, normalized)

        records = records_from_raw_transactions(
            transactions,
            normalize_fn=_normalize,
            canonical_raw_fn=_canonical_raw,
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
            _expected_rows(ctx),
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
