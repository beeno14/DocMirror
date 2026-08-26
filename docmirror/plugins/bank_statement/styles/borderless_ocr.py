# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Borderless OCR bank ledger style parser.

Handles char-strategy / weak-border tables with relaxed header detection,
institution-specific header normalization, and grid-standard fallback normalization.

Pipeline role: registered in ``style_registry._PARSERS`` as ``borderless_ocr``;
selected when OCR-dominant Mirror output or style detector scores favor borderless layouts.

Key exports: ``PARSER_ID``, ``STYLE_ID``, ``is_ocr_dominant``, ``extract_transactions``.

Dependencies: ``header_resolve`` (relaxed mode), ``row_extract``, ``grid_standard``.
"""

from __future__ import annotations

from typing import Any

from docmirror.plugins._base.standardizer import normalize_amount
from docmirror.plugins.bank_statement.context import StyleContext
from docmirror.plugins.bank_statement.header_resolve import (
    RELAXED_MIN_COLUMNS,
    canonical_key_for_field,
    detect_headers,
    has_split_debit_credit_headers,
    registry_strict_header_match_count,
)
from docmirror.plugins.bank_statement.header_resolve import (
    registry_strict_header_match_count as strict_header_match_count,
)
from docmirror.plugins.bank_statement.institution import match_institution, normalize_table_headers
from docmirror.plugins.bank_statement.row_extract import (
    count_transaction_data_rows,
    extract_rows_from_header,
    row_has_transaction_data,
)
from docmirror.plugins.bank_statement.styles import grid_standard

PARSER_ID = "borderless_ocr"
STYLE_ID = "borderless_ocr"
_REQUIRED_TRANSACTION_ROLES = frozenset({"date", "amount", "balance"})
_CANONICAL_HEADER_NORMALIZATION = "borderless_header_normalization"


def _prepare_tables(ctx: StyleContext) -> list[list[list[str]]]:
    variant = match_institution(ctx.full_text, ctx.institution)
    return normalize_table_headers(ctx.tables, variant=variant)


def _has_incomplete_transaction_plane(
    tables: list[list[list[str]]],
    registry: dict[str, Any],
) -> bool:
    """Reject a multi-table plane when any transaction-bearing segment loses a required role."""

    for table in tables:
        if not table:
            continue
        header = detect_headers([table], registry, prefer_strict=False)
        if header is None:
            if any(row_has_transaction_data(row, strict_first_col=False) for row in table):
                return True
            continue
        if count_transaction_data_rows([table], header) <= 0:
            continue
        if not _REQUIRED_TRANSACTION_ROLES.issubset(header.col_map):
            return True
    return False


def _working_transaction(
    row: list[str],
    col_map: dict[str, int],
    registry: dict[str, Any],
) -> dict[str, str]:
    return {
        canonical_key_for_field(field_name, registry): str(row[column] or "").strip()
        for field_name, column in col_map.items()
        if column < len(row)
    }


def _source_metadata(source_headers: list[str], source_row: list[str]) -> dict[str, Any]:
    indexes = {str(header or "").strip(): index for index, header in enumerate(source_headers)}

    def value(name: str) -> str:
        index = indexes.get(name, -1)
        return str(source_row[index] or "").strip() if 0 <= index < len(source_row) else ""

    try:
        source_page = int(value("_source_page"))
    except (TypeError, ValueError):
        source_page = 0
    if source_page <= 0:
        return {}

    source: dict[str, Any] = {
        "source_page": source_page,
        "page_range": [source_page, source_page],
    }
    table_id = value("_source_table_id")
    if table_id:
        source["table_id"] = table_id
    try:
        source_row_index = int(value("_source_row_index"))
    except (TypeError, ValueError):
        source_row_index = -1
    if source_row_index >= 0:
        source["source_row_index"] = source_row_index
    return source


def _extract_with_source_lineage(
    source_tables: list[list[list[str]]],
    prepared_tables: list[list[list[str]]],
    registry: dict[str, Any],
) -> list[dict[str, Any]]:
    """Extract from normalized headers while retaining the exact source row envelope."""

    transactions: list[dict[str, Any]] = []
    seen_tables: set[tuple[tuple[str, ...], ...]] = set()
    for table_index, prepared_table in enumerate(prepared_tables):
        if not prepared_table:
            continue
        signature = tuple(tuple(str(cell or "").strip() for cell in row) for row in prepared_table)
        if signature in seen_tables:
            continue
        seen_tables.add(signature)
        header = detect_headers([prepared_table], registry, prefer_strict=False)
        if header is None:
            continue
        extracted = extract_rows_from_header(
            [prepared_table],
            header,
            registry,
            strict_first_col=False,
        )
        source_table = source_tables[table_index] if table_index < len(source_tables) else prepared_table
        source_headers = (
            source_table[header.row_index]
            if header.row_index < len(source_table)
            else prepared_table[header.row_index]
        )
        search_from = header.row_index + 1
        for transaction in extracted:
            matched_index = next(
                (
                    row_index
                    for row_index in range(search_from, len(prepared_table))
                    if _working_transaction(prepared_table[row_index], header.col_map, registry) == transaction
                ),
                None,
            )
            if matched_index is not None and matched_index < len(source_table):
                source_row = source_table[matched_index]
                source_raw = {
                    str(source_headers[column] or "").strip(): str(source_row[column] or "").strip()
                    for column in header.col_map.values()
                    if column < len(source_headers)
                    and column < len(source_row)
                    and str(source_headers[column] or "").strip()
                    and not str(source_headers[column] or "").strip().startswith("_")
                }
                if source_raw:
                    transaction["_source_raw"] = source_raw
                    transaction["_canonical_raw_from_working"] = _CANONICAL_HEADER_NORMALIZATION
                if source := _source_metadata(source_headers, source_row):
                    transaction["_source"] = source
                search_from = matched_index + 1
            transactions.append(transaction)
    return transactions


def is_ocr_dominant(ctx: StyleContext) -> bool:
    pr = ctx.parse_result
    if pr is None:
        return False
    info = getattr(pr, "parser_info", None)
    if info is None:
        return False
    method = getattr(info, "extraction_method", None)
    if method is None:
        return False
    val = method.value if hasattr(method, "value") else str(method)
    return val in ("ocr", "hybrid", "image")


def table_is_borderless_ocr(ctx: StyleContext, registry: dict[str, Any] | None = None) -> bool:
    """True when strict headers fail but relaxed headers + data rows succeed."""
    from docmirror.plugins.bank_statement.styles.compact_merged import table_has_compact_ledger
    from docmirror.plugins.bank_statement.styles.signed_amount import table_has_signed_amount_cells

    if not ctx.tables:
        return False
    if table_has_compact_ledger(ctx.tables) or table_has_signed_amount_cells(ctx.tables):
        return False

    if registry is None:
        from docmirror.plugins.bank_statement.community_plugin import BANK_COLUMN_REGISTRY

        registry = BANK_COLUMN_REGISTRY

    prepared = _prepare_tables(ctx)
    if has_split_debit_credit_headers(prepared) and registry_strict_header_match_count(prepared, registry) >= 2:
        return False

    registry_strict = registry_strict_header_match_count(prepared, registry)
    if registry_strict >= 3:
        return False

    header = detect_headers(prepared, registry, prefer_strict=False)
    if header is None or len(header.col_map) < RELAXED_MIN_COLUMNS:
        return False

    if count_transaction_data_rows(prepared, header) < 2:
        return False

    return is_ocr_dominant(ctx) or registry_strict < 3


def extract_transactions(ctx: StyleContext, plugin: Any) -> list[dict[str, Any]]:
    prepared = _prepare_tables(ctx)
    if _has_incomplete_transaction_plane(prepared, plugin.column_registry):
        return []
    batch = _extract_with_source_lineage(
        ctx.tables,
        prepared,
        plugin.column_registry,
    )
    if batch:
        return batch
    return grid_standard.extract_transactions(ctx, plugin)


def normalize_record(raw_txn: dict[str, str], plugin: Any) -> dict[str, Any]:
    split = grid_standard.normalize_split_debit_credit(raw_txn, plugin)
    if split is not None:
        return split

    normalized = plugin._normalize(raw_txn)
    if normalized.get("amount") is None:
        for key, value in raw_txn.items():
            if any(n in key for n in ("金额", "发生", "Amount")):
                amount = normalize_amount(value)
                if amount is not None:
                    normalized["amount"] = float(amount)
                    normalized["amount_cny"] = float(amount)
                    break
    return normalized


def detect_headers_relaxed(
    tables: list[list[list[str]]],
    registry: dict[str, Any],
) -> tuple[int, list[str], dict[str, int]]:
    header = detect_headers(tables, registry, prefer_strict=False)
    if header is None:
        return 0, [], {}
    return header.row_index, header.raw_headers, header.col_map


__all__ = [
    "PARSER_ID",
    "STYLE_ID",
    "detect_headers_relaxed",
    "extract_transactions",
    "is_ocr_dominant",
    "normalize_record",
    "strict_header_match_count",
    "table_is_borderless_ocr",
]
