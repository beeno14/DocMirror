# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Canonical Quality Floor (CQF) — bank statement export audit (ADR-BS-06).

Redefines coverage as canonical_extracted / canonical_expected where canonical rows
satisfy date + (amount with income/expense direction). Drives community degraded
status and honest coverage metrics (BS-013, BS-009).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from math import isfinite
from typing import Any

_DATE_CELL_RE = re.compile(r"^(?:19|20)\d{2}(?:[-/.年]?\d{1,2})(?:[-/.月]?\d{1,2})(?:日)?(?:\d{2}:\d{2}:\d{2})?$")
_MONEY_CELL_RE = re.compile(r"^[+-]?(?:[¥￥$])?\d[\d,]*\.\d{1,2}$")
_NON_TRANSACTION_MARKERS = ("合计", "小计", "总计", "本页", "期初", "期末")


def is_canonical_row(norm: dict[str, Any]) -> bool:
    """BS-A1: date plus an explicit, parseable directional amount.

    A source value of zero is a valid ledger fact. Missing or unparseable
    amounts remain non-canonical and must never be defaulted to zero.
    """
    if not norm.get("date"):
        return False
    direction = norm.get("direction")
    amount = norm.get("amount")
    if direction not in ("income", "expense") or amount in (None, ""):
        return False
    try:
        numeric_amount = float(amount)
    except (TypeError, ValueError):
        return False
    return isfinite(numeric_amount) and numeric_amount >= 0.0


def canonical_expected_from_parse_result(parse_result: Any) -> int:
    """Return the strongest independent transaction-row estimate."""
    if parse_result is None:
        return 0
    from docmirror.evidence.spe_consumer import read_ltqg_summary, read_structure_spe
    from docmirror.tables.access import get_logical_tables
    from docmirror.tables.compose.ledger_quality import sum_passed_data_row_estimates

    spe = read_structure_spe(parse_result)
    summary = read_ltqg_summary(spe, parse_result)
    candidates: list[int] = []
    if summary.get("enabled"):
        candidates.append(int(summary.get("expected_data_rows") or 0))

    logical_tables = get_logical_tables(parse_result)
    if logical_tables:
        candidates.append(sum_passed_data_row_estimates(logical_tables))

    from docmirror.evidence.spe_consumer import mirror_expected_primary_rows

    candidates.extend(
        (
            mirror_expected_primary_rows(parse_result, spe),
            physical_transaction_row_estimate(parse_result),
        )
    )
    return max((candidate for candidate in candidates if candidate > 0), default=0)


def physical_transaction_row_estimate(parse_result: Any) -> int:
    """Estimate visible ledger rows without summing overlapping table candidates."""
    if parse_result is None:
        return 0
    total = 0
    for page in getattr(parse_result, "pages", []) or []:
        table_counts: list[int] = []
        for table in getattr(page, "tables", []) or []:
            from docmirror.plugins.bank_statement.header_resolve import align_bank_ledger_row

            headers = [str(header or "") for header in getattr(table, "headers", []) or []]
            count = sum(
                _looks_like_physical_transaction_row(
                    align_bank_ledger_row(
                        headers,
                        [str(getattr(cell, "text", "") or "") for cell in getattr(row, "cells", []) or []],
                    ),
                    headers,
                )
                for row in getattr(table, "rows", []) or []
            )
            if count > 0:
                table_counts.append(count)
        if table_counts:
            total += max(table_counts)
    return total


def _looks_like_physical_transaction_row(values: list[str], headers: list[str]) -> bool:
    from docmirror.plugins.bank_statement.header_resolve import normalize_header_cell

    compact_values = [
        re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(value or "")))
        for value in values
        if str(value or "").strip()
    ]
    if not compact_values or any(marker in "".join(compact_values) for marker in _NON_TRANSACTION_MARKERS):
        return False
    has_date = any(_DATE_CELL_RE.fullmatch(value) for value in compact_values)
    has_money = any(_MONEY_CELL_RE.fullmatch(value.replace("元", "")) for value in compact_values)
    normalized_headers = [normalize_header_cell(header) for header in headers]
    amount_indices = [
        index
        for index, header in enumerate(normalized_headers)
        if any(marker in header for marker in ("交易金额", "发生额", "收入金额", "支出金额", "借方", "贷方"))
    ]
    if amount_indices:
        source_amounts = [_money_value(values[index]) for index in amount_indices if index < len(values)]
        has_parseable_amount = any(amount is not None for amount in source_amounts)
    else:
        has_parseable_amount = has_money
    return has_date and has_money and has_parseable_amount


def _money_value(value: str) -> float | None:
    compact = re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(value or "")))
    compact = compact.replace(",", "").replace("¥", "").replace("￥", "").replace("$", "")
    try:
        return abs(float(compact))
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class CQFResult:
    canonical_expected: int
    canonical_extracted: int
    coverage_ratio: float
    canonical_ratio: float
    extract_status: str  # success | low_coverage | degraded

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_expected": self.canonical_expected,
            "canonical_extracted": self.canonical_extracted,
            "coverage_ratio": round(self.coverage_ratio, 4),
            "canonical_ratio": round(self.canonical_ratio, 4),
            "extract_status": self.extract_status,
        }


def resolve_extract_status(
    *,
    coverage_ratio: float,
    canonical_ratio: float,
) -> str:
    """Map CQF ratios to export status (community / finance alignment)."""
    if coverage_ratio >= 1.0 and canonical_ratio >= 1.0:
        return "success"
    if coverage_ratio < 0.50 or canonical_ratio < 0.50:
        return "degraded"
    return "low_coverage"


def audit_cqf(
    records: list[dict[str, Any]],
    *,
    canonical_expected: int,
) -> CQFResult:
    """Audit extracted records against canonical expected denominator."""
    canonical_extracted = sum(1 for rec in records if is_canonical_row(rec.get("normalized") or {}))
    expected = max(int(canonical_expected or 0), 0)
    if expected <= 0:
        coverage_ratio = 1.0 if canonical_extracted > 0 else 0.0
        canonical_ratio = coverage_ratio
    else:
        coverage_ratio = min(canonical_extracted / expected, 1.0)
        canonical_ratio = min(canonical_extracted / expected, 1.0)

    status = resolve_extract_status(
        coverage_ratio=coverage_ratio,
        canonical_ratio=canonical_ratio,
    )
    if expected > 0 and canonical_extracted != expected and status == "success":
        status = "low_coverage"
    return CQFResult(
        canonical_expected=expected,
        canonical_extracted=canonical_extracted,
        coverage_ratio=coverage_ratio,
        canonical_ratio=canonical_ratio,
        extract_status=status,
    )


def audit_row_accounting(*, parsed_rows: int, canonical_rows: int, emitted_rows: int) -> list[str]:
    """Validate the plugin-local row lifecycle before records reach a Dataset."""
    parsed = max(int(parsed_rows or 0), 0)
    canonical = max(int(canonical_rows or 0), 0)
    emitted = max(int(emitted_rows or 0), 0)
    warnings: list[str] = []
    if canonical != emitted:
        warnings.append(f"BANK_CANONICAL_EMITTED_ROW_MISMATCH:canonical={canonical}:emitted={emitted}")
    if emitted > parsed:
        warnings.append(f"BANK_EMITTED_ROWS_EXCEED_PARSED:parsed={parsed}:emitted={emitted}")
    return warnings


__all__ = [
    "CQFResult",
    "audit_cqf",
    "audit_row_accounting",
    "canonical_expected_from_parse_result",
    "is_canonical_row",
    "physical_transaction_row_estimate",
    "resolve_extract_status",
]
