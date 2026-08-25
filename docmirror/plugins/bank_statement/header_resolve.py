# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unified bank statement header resolution — SSOT for strict and relaxed matching.

Merges OCR header aliases, layout profile aliases, and ``ColumnMatcher`` into
``detect_headers`` with configurable strictness (minimum columns, lookahead rows).
Single entry for all bank style parsers when locating header rows and column maps.

Pipeline role: used by ``row_extract``, ``grid_standard``, and ``borderless_ocr``
before row iteration; bridges ``core.profile.registry`` with plugin column registry.

Key exports: ``HeaderMatch``, ``detect_headers``, ``canonical_key_for_field``,
``has_split_debit_credit_headers``, strict/relaxed threshold constants.

Dependencies: ``column_registry.ColumnMatcher``, ``bank_statement.institution``.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from docmirror.layout.profile.registry import resolve_header_aliases
from docmirror.plugins._base.column_registry import ColumnMatcher
from docmirror.plugins.bank_statement.institution import get_bank_layout_profile

# OCR / regional header variants merged into plugin-layer SSOT (not Mirror EPO).
_OCR_HEADER_ALIASES: dict[str, str] = {
    "值日": "交易日期",
    "交易日": "交易日期",
    "记账日": "交易日期",
    "交易说明": "摘要",
    "交易摘要": "摘要",
    "发生金额": "交易金额",
    "发生额(元)": "交易金额",
    "交易额(元)": "交易金额",
    "账面余领": "余额",
    "账 面余额": "余额",
    "账 面 余 额": "余额",
    "账面余额": "余额",
    "卡余额": "余额",
    "本次余额": "余额",
    "对手信息": "对方户名",
    "对⼿信息": "对方户名",
    "交易地点/附言": "摘要",
}
_PROFILE_TO_REGISTRY: dict[str, str] = {
    "交易时间": "交易日期",
    "账户余额": "余额",
}

STRICT_MIN_COLUMNS = 3
RELAXED_MIN_COLUMNS = 2
STRICT_LOOKAHEAD = 8
RELAXED_LOOKAHEAD = 15
_BANK_MATCH_TRANSLATION = str.maketrans(
    {
        "戶": "户",
        "賬": "账",
        "帳": "账",
        "幣": "币",
        "⼈": "人",
        "⺠": "民",
        "⾏": "行",
        "⽌": "止",
        "⽇": "日",
        "⾦": "金",
        "⼿": "手",
    }
)
_TIME_VALUE_RE = re.compile(r"^(?:\d{6}|\d{1,2}:\d{2}(?::\d{2})?)$")
_MONEY_VALUE_RE = re.compile(r"^[+-]?(?:[¥￥$])?\d[\d,]*\.\d{1,2}$")
_TRANSACTION_MONEY_VALUE_RE = re.compile(r"^[+-]?(?:[¥￥$])?\d[\d,]*(?:\.\d{1,2})?$")
_TRANSACTION_DATE_VALUE_RE = re.compile(
    r"^(?:(?:19|20)\d{2}(?:[-/.]\d{1,2}){2}|(?:19|20)\d{6})(?:[ T]\d{1,2}:?\d{2}(?::?\d{2})?)?$"
)


@dataclass(frozen=True)
class HeaderMatch:
    table_index: int
    row_index: int
    raw_headers: list[str]
    col_map: dict[str, int]
    mode: str  # strict | relaxed


@lru_cache(maxsize=2048)
def normalize_header_cell(text: str) -> str:
    cell = normalize_bank_matching_text(text).strip()
    if not cell:
        return cell
    profile = get_bank_layout_profile()
    cell = resolve_header_aliases(profile, cell)
    cell = _OCR_HEADER_ALIASES.get(cell, cell)
    cell = _PROFILE_TO_REGISTRY.get(cell, cell)
    return re.sub(r"[\s\n\r\t\u3000]", "", cell).replace("\u00a0", "")


def normalize_bank_matching_text(text: str) -> str:
    """Normalize compatibility glyphs for matching without changing source facts."""
    return unicodedata.normalize("NFKC", str(text or "")).translate(_BANK_MATCH_TRANSLATION)


def align_bank_ledger_row(headers: list[str], values: list[str]) -> list[str]:
    """Restore one omitted optional time cell using only shared ledger semantics."""
    if not headers or not values:
        return list(values)
    normalized_headers = [re.sub(r"\s+", "", normalize_bank_matching_text(header)) for header in headers]
    time_index = next(
        (index for index, header in enumerate(normalized_headers) if header in {"交易时间", "时间", "Time"}),
        -1,
    )
    amount_index = next(
        (
            index
            for index, header in enumerate(normalized_headers)
            if any(marker in header for marker in ("交易金额", "发生额", "收入金额", "支出金额"))
        ),
        -1,
    )
    if time_index <= 0 or amount_index <= time_index or amount_index >= len(values):
        return list(values)

    time_value = re.sub(r"\s+", "", normalize_bank_matching_text(values[time_index]))
    displaced_amount = re.sub(r"\s+", "", normalize_bank_matching_text(values[amount_index - 1]))
    current_amount = re.sub(r"\s+", "", normalize_bank_matching_text(values[amount_index]))
    if _TIME_VALUE_RE.fullmatch(time_value):
        return list(values)
    if not _MONEY_VALUE_RE.fullmatch(displaced_amount) or not _MONEY_VALUE_RE.fullmatch(current_amount):
        return list(values)

    aligned = [*values[:time_index], "", *values[time_index:]]
    return aligned[: len(headers)]


def canonical_key_for_field(field_name: str, registry: dict[str, Any]) -> str:
    for canonical_name, mapping in registry.items():
        if mapping.field == field_name:
            return canonical_name
    return field_name


def _match_row(
    row: list[str],
    registry: dict[str, Any],
    *,
    min_columns: int,
) -> tuple[list[str], dict[str, int]] | None:
    matcher = ColumnMatcher(registry)
    normalized_row = [normalize_header_cell(c) for c in row]
    col_map = matcher.match(normalized_row)
    col_map = prefer_explicit_counterparty_columns(normalized_row, col_map, registry)
    col_map = prefer_explicit_direction_column(normalized_row, col_map)
    if len(col_map) >= min_columns:
        return [str(c or "").strip() for c in row], col_map
    return None


def prefer_explicit_counterparty_columns(
    header_cells: list[str],
    col_map: dict[str, int],
    registry: dict[str, Any],
) -> dict[str, int]:
    """Let an exact counterparty header override an earlier fuzzy account match."""
    resolved = dict(col_map)
    for field_name in ("counter_account", "counter_party"):
        variants: set[str] = set()
        for canonical_name, mapping in registry.items():
            if getattr(mapping, "field", "") != field_name:
                continue
            variants.add(normalize_header_cell(canonical_name))
            variants.update(normalize_header_cell(alias) for alias in (getattr(mapping, "aliases", None) or []))
        exact_index = next((index for index, cell in enumerate(header_cells) if cell in variants), None)
        if exact_index is not None:
            resolved[field_name] = exact_index
    return resolved


def prefer_explicit_direction_column(header_cells: list[str], col_map: dict[str, int]) -> dict[str, int]:
    """Prefer a dedicated debit/credit flag over a transaction-type column."""
    explicit_markers = {
        "借贷",
        "借/贷",
        "借贷标志",
        "收/支",
        "收入/支出",
        "方向",
        "交易方向",
        "dcflg",
    }
    explicit_index = next(
        (
            index
            for index, cell in enumerate(header_cells)
            if re.sub(r"\s+", "", str(cell or "")).lower() in explicit_markers
        ),
        None,
    )
    if explicit_index is None:
        return col_map
    return {**col_map, "direction": explicit_index}


def best_header_match(
    tables: list[list[list[str]]],
    registry: dict[str, Any],
    *,
    max_rows: int,
    min_columns: int,
) -> HeaderMatch | None:
    best: HeaderMatch | None = None
    best_count = 0

    for tbl_idx, tbl in enumerate(tables):
        if not tbl:
            continue
        for row_idx, row in enumerate(tbl[:max_rows]):
            matched = _match_row(row, registry, min_columns=min_columns)
            if matched is None:
                continue
            raw_headers, col_map = matched
            count = len(col_map)
            mode = "strict" if count >= STRICT_MIN_COLUMNS else "relaxed"
            candidate = HeaderMatch(tbl_idx, row_idx, raw_headers, col_map, mode)
            if count > best_count:
                best = candidate
                best_count = count
            if count >= STRICT_MIN_COLUMNS:
                return candidate

    return best


def detect_headers(
    tables: list[list[list[str]]],
    registry: dict[str, Any],
    *,
    prefer_strict: bool = True,
) -> HeaderMatch | None:
    """Cascade strict → relaxed header detection."""
    if prefer_strict:
        strict = best_header_match(
            tables,
            registry,
            max_rows=STRICT_LOOKAHEAD,
            min_columns=STRICT_MIN_COLUMNS,
        )
        if strict is not None:
            return _merge_stacked_header_row(tables, strict, registry)
    relaxed = best_header_match(
        tables,
        registry,
        max_rows=RELAXED_LOOKAHEAD,
        min_columns=RELAXED_MIN_COLUMNS,
    )
    return _merge_stacked_header_row(tables, relaxed, registry) if relaxed is not None else None


def registry_strict_header_match_count(
    tables: list[list[list[str]]],
    registry: dict[str, Any],
) -> int:
    """Max ColumnMatcher hits without OCR alias normalization (institution cells only)."""
    matcher = ColumnMatcher(registry)
    best = 0
    for tbl in tables:
        if not tbl:
            continue
        for row in tbl[:STRICT_LOOKAHEAD]:
            col_map = matcher.match([str(c or "").strip() for c in row])
            best = max(best, len(col_map))
    return best


def strict_header_match_count(tables: list[list[list[str]]], registry: dict[str, Any]) -> int:
    """Max normalized ColumnMatcher hits (includes OCR + profile aliases)."""
    best = 0
    for tbl in tables:
        if not tbl:
            continue
        for row in tbl[:STRICT_LOOKAHEAD]:
            matched = _match_row(row, registry, min_columns=1)
            if matched:
                best = max(best, len(matched[1]))
    return best


_INCOME_CELL_KEYS: tuple[str, ...] = ("收入", "贷方发生额", "贷方", "Credit")
_EXPENSE_CELL_KEYS: tuple[str, ...] = ("支出", "借方发生额", "借方", "Debit")


def _cell_has_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(n in text for n in needles)


def has_merged_amount_header(tables: list[list[list[str]]]) -> bool:
    """True when a header cell contains BOTH income and expense keywords (e.g. 收入/支出金额).

    This signals a single combined amount column, NOT split debit/credit columns."""
    for tbl in tables:
        for row in tbl[:RELAXED_LOOKAHEAD]:
            for cell in row:
                text = normalize_header_cell(cell)
                if _cell_has_any(text, _INCOME_CELL_KEYS) and _cell_has_any(text, _EXPENSE_CELL_KEYS):
                    return True
    return False


def has_split_debit_credit_headers(tables: list[list[list[str]]]) -> bool:
    """True when income and expense keywords appear in DIFFERENT header cells.

    This function explicitly resists false positives from merged headers like
    ``收入/支出金额`` where both keywords land in one cell.  A merged header
    is a signed-amount column, not separate debit/credit columns.

    Cell-level matching is the single most impactful fix for ICBC-style
    statements (BS-018)."""
    for tbl in tables:
        for row in tbl[:RELAXED_LOOKAHEAD]:
            cells = [normalize_header_cell(c) for c in row]

            if any(_cell_has_any(c, _INCOME_CELL_KEYS) and _cell_has_any(c, _EXPENSE_CELL_KEYS) for c in cells):
                continue

            has_income = any(_cell_has_any(c, _INCOME_CELL_KEYS) for c in cells)
            has_expense = any(_cell_has_any(c, _EXPENSE_CELL_KEYS) for c in cells)
            if has_income and has_expense:
                return True
    return False


def _merge_stacked_header_row(
    tables: list[list[list[str]]],
    header: HeaderMatch,
    registry: dict[str, Any],
) -> HeaderMatch:
    """Merge a parent header with an immediately following debit/credit subheader.

    Corporate electronic statements commonly put ``Transaction Amount`` and
    ``Counterparty Information`` in the first header row, then ``Debit/Credit``
    and the counterparty subcolumns in the next row. Keeping the physical column
    positions while combining both labels lets the existing split-column parser
    retain every semantic column.
    """
    table = tables[header.table_index]
    next_index = header.row_index + 1
    if next_index >= len(table):
        return header
    child_row = [str(cell or "").strip() for cell in table[next_index]]
    if _row_is_transaction_shaped(child_row):
        # A transaction can legitimately contain both an expense marker and an
        # income-looking counterparty/summary.  It must never be consumed as a
        # debit/credit subheader merely because those words occur in different
        # business cells.  Stacked header rows contain labels, not a valid
        # transaction date plus monetary values.
        return header
    if not has_split_debit_credit_headers([[child_row]]):
        return header

    width = max(len(header.raw_headers), len(child_row))
    merged_headers: list[str] = []
    for column_index in range(width):
        parent = header.raw_headers[column_index] if column_index < len(header.raw_headers) else ""
        child = child_row[column_index] if column_index < len(child_row) else ""
        if parent and child:
            merged_headers.append(f"{parent}\n{child}")
        else:
            merged_headers.append(parent or child)

    matched = _match_row(merged_headers, registry, min_columns=RELAXED_MIN_COLUMNS)
    return HeaderMatch(
        table_index=header.table_index,
        row_index=header.row_index,
        raw_headers=merged_headers,
        col_map=matched[1] if matched is not None else header.col_map,
        mode=header.mode,
    )


def _row_is_transaction_shaped(row: list[str]) -> bool:
    """Return whether a prospective child header is clearly a ledger row."""

    cells = [re.sub(r"\s+", "", normalize_bank_matching_text(cell)) for cell in row if str(cell or "").strip()]
    has_date = any(_TRANSACTION_DATE_VALUE_RE.fullmatch(cell) for cell in cells)
    monetary_cells = sum(bool(_TRANSACTION_MONEY_VALUE_RE.fullmatch(cell)) for cell in cells)
    return has_date and monetary_cells >= 1
