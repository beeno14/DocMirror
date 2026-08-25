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
from datetime import date
from math import isfinite
from typing import Any

_DATE_CELL_RE = re.compile(r"^(?:19|20)\d{2}(?:[-/.年]?\d{1,2})(?:[-/.月]?\d{1,2})(?:日)?(?:\d{2}:\d{2}:\d{2})?$")
_MONEY_CELL_RE = re.compile(r"^[+-]?(?:[¥￥$])?\d[\d,]*\.\d{1,2}$")
_NON_TRANSACTION_MARKERS = ("合计", "小计", "总计", "本页", "期初", "期末")
_SPLIT_INCOME_HEADERS = ("收入", "收入金额", "贷方发生额", "贷方", "转入金额")
_SPLIT_EXPENSE_HEADERS = ("支出", "支出金额", "借方发生额", "借方", "转出金额")


def is_canonical_row(norm: dict[str, Any]) -> bool:
    """BS-A1: date plus an explicit, parseable directional amount.

    A source value of zero is a valid ledger fact. Missing or unparseable
    amounts remain non-canonical and must never be defaulted to zero.
    """
    if not _is_valid_calendar_date(norm.get("date")):
        return False
    amount = norm.get("amount")
    if amount in (None, ""):
        return False
    try:
        numeric_amount = float(amount)
    except (TypeError, ValueError):
        return False
    if not isfinite(numeric_amount) or numeric_amount < 0.0:
        return False
    if numeric_amount == 0.0:
        # A source-backed zero-value ledger event (for example an interest-tax
        # line) has no mathematically meaningful debit/credit direction.  Keep
        # the business row without fabricating one, but still require another
        # transaction semantic so a bare date/zero fragment cannot become a
        # dataset row. Non-zero amounts require an explicit or uniquely
        # derived direction below.
        return norm.get("direction") in ("income", "expense") or any(
            norm.get(field) not in (None, "")
            for field in ("balance", "summary", "reference", "note", "counter_party", "counter_account")
        )
    return norm.get("direction") in ("income", "expense")


def _is_valid_calendar_date(value: Any) -> bool:
    """Return whether a normalized ledger date names a real calendar day.

    Candidate selection runs before the public dataset is assembled.  Treating
    a merely non-empty date as canonical allowed reconstruction furniture such
    as ``2023-06-00`` to compete with genuine transactions.  Normalizers may
    still present compact or slash-separated source dates, so validate those
    losslessly instead of requiring one display format here.
    """
    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    match = re.fullmatch(
        r"(?P<year>(?:19|20)\d{2})[-/.年]?(?P<month>\d{1,2})[-/.月]?(?P<day>\d{1,2})(?:日)?",
        text,
    )
    if match is None:
        return False
    try:
        date(int(match.group("year")), int(match.group("month")), int(match.group("day")))
    except ValueError:
        return False
    return True


def audit_amount_consistency(records: list[dict[str, Any]]) -> list[str]:
    """Compare explicit split source amounts with normalized facts without dropping zero-value rows."""
    warnings: list[str] = []
    for row_index, record in enumerate(records, start=1):
        raw = record.get("raw") if isinstance(record.get("raw"), dict) else {}
        normalized = record.get("normalized") if isinstance(record.get("normalized"), dict) else {}
        income_values = _source_amount_values(raw, _SPLIT_INCOME_HEADERS)
        expense_values = _source_amount_values(raw, _SPLIT_EXPENSE_HEADERS)
        source_values = [*income_values, *expense_values]
        if not source_values:
            continue

        record_id = str(record.get("record_id") or f"row:{row_index}")
        parsed_values = [value for _raw_value, value in source_values if value is not None]
        nonzero_values = [value for value in parsed_values if value != 0]
        normalized_amount = _normalized_amount(normalized.get("amount"))
        if source_values and all(not raw_value for raw_value, _value in source_values) and normalized_amount == 0.0:
            warnings.append(f"BANK_AMOUNT_DEFAULTED_TO_ZERO:row={row_index}:record_id={record_id}")
        if nonzero_values and normalized_amount in (None, 0.0):
            warnings.append(
                f"BANK_NONZERO_AMOUNT_LOST:row={row_index}:record_id={record_id}:"
                f"source_nonzero={','.join(str(value) for value in nonzero_values)}:normalized={normalized_amount}"
            )

        source_is_explicit_zero = bool(parsed_values) and not nonzero_values
        direction = str(normalized.get("direction") or "")
        if source_is_explicit_zero and normalized_amount == 0.0 and direction not in {"income", "expense"}:
            warnings.append(f"BANK_ZERO_AMOUNT_DIRECTION_UNKNOWN:row={row_index}:record_id={record_id}")

        income_nonzero = any(value not in (None, 0.0) for _raw_value, value in income_values)
        expense_nonzero = any(value not in (None, 0.0) for _raw_value, value in expense_values)
        if income_nonzero and expense_nonzero:
            warnings.append(f"BANK_SPLIT_AMOUNT_CONFLICT:row={row_index}:record_id={record_id}")
    return warnings


def _source_amount_values(raw: dict[str, Any], aliases: tuple[str, ...]) -> list[tuple[str, float | None]]:
    compact_aliases = {_compact_header(alias) for alias in aliases}
    values: list[tuple[str, float | None]] = []
    for header, raw_value in raw.items():
        header_text = str(header or "")
        compact_header = _compact_header(header_text)
        header_parts = {_compact_header(part) for part in header_text.splitlines() if part.strip()}
        if compact_header not in compact_aliases and not compact_aliases.intersection(header_parts):
            continue
        text = str(raw_value or "").strip()
        values.append((text, _money_value(text) if text else None))
    return values


def _compact_header(value: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(value or ""))).lower()


def _normalized_amount(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        amount = abs(float(value))
    except (TypeError, ValueError):
        return None
    return amount if isfinite(amount) else None


def canonical_expected_from_parse_result(parse_result: Any) -> int:
    """Return the strongest candidate-local transaction-row estimate.

    The result helps rank and diagnose parser candidates. It is not independent
    issuer evidence and must not be exposed as an exact document-completeness
    denominator.
    """
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
    pages = list(getattr(parse_result, "pages", []) or [])
    schema_signatures = {
        signature
        for page in pages
        for table in getattr(page, "tables", []) or []
        if (headers := [str(header or "") for header in getattr(table, "headers", []) or []])
        and (signature := _physical_ledger_role_signature(headers)) is not None
    }
    total = 0
    for page in pages:
        table_counts: list[int] = []
        for table in getattr(page, "tables", []) or []:
            from docmirror.plugins.bank_statement.header_resolve import align_bank_ledger_row

            headers = [str(header or "") for header in getattr(table, "headers", []) or []]
            row_count = sum(
                _looks_like_physical_transaction_row(
                    align_bank_ledger_row(
                        headers,
                        [str(getattr(cell, "text", "") or "") for cell in getattr(row, "cells", []) or []],
                    ),
                    headers,
                )
                for row in getattr(table, "rows", []) or []
            )
            # Some table extractors promote the first transaction on a
            # continuation page into ``headers``.  Count it only when the same
            # table also contains an ordinary transaction row.  This local
            # lineage witness avoids treating an isolated date-and-money
            # furniture table as a ledger page.
            promoted_transaction_header = (
                row_count > 0
                and _looks_like_physical_transaction_row(headers, headers)
                and any(_transaction_values_match_role_signature(headers, signature) for signature in schema_signatures)
            )
            count = row_count + int(promoted_transaction_header)
            if count > 0:
                table_counts.append(count)
        if table_counts:
            total += max(table_counts)
    return total


def _looks_like_physical_ledger_header(headers: list[str]) -> bool:
    """Return whether a source header independently names date and amount roles."""
    return _physical_ledger_role_signature(headers) is not None


def _physical_ledger_role_signature(
    headers: list[str],
) -> tuple[int, tuple[int, ...], tuple[int, ...], tuple[int, ...]] | None:
    """Describe independently labelled ledger roles by their physical columns."""
    compact = [_compact_header(header) for header in headers]
    date_roles = (
        "交易日期",
        "记账日期",
        "交易时间",
        "日期",
        "transactiondate",
        "bookingdate",
        "valuedate",
    )
    amount_roles = (
        "交易金额",
        "发生额",
        "借方",
        "贷方",
        "收入金额",
        "支出金额",
        "transactionamount",
        "debit",
        "credit",
    )
    balance_roles = ("余额", "账户余额", "balance")
    date_indices = tuple(
        index for index, value in enumerate(compact) if any(role in value for role in date_roles)
    )
    amount_indices = tuple(
        index for index, value in enumerate(compact) if any(role in value for role in amount_roles)
    )
    balance_indices = tuple(
        index for index, value in enumerate(compact) if any(role in value for role in balance_roles)
    )
    if not date_indices or not amount_indices:
        return None
    return len(headers), date_indices, amount_indices, balance_indices


def _transaction_values_match_role_signature(
    values: list[str],
    signature: tuple[int, tuple[int, ...], tuple[int, ...], tuple[int, ...]],
) -> bool:
    """Require a promoted row to occupy the same roles as a real source schema."""
    width, date_indices, amount_indices, balance_indices = signature
    if len(values) != width:
        return False

    compact_values = [re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(value or ""))) for value in values]
    if not any(_DATE_CELL_RE.fullmatch(compact_values[index]) for index in date_indices):
        return False
    if not any(_money_value(compact_values[index]) is not None for index in amount_indices):
        return False
    return not balance_indices or any(_money_value(compact_values[index]) is not None for index in balance_indices)


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
        # Extracted rows cannot prove that no rows are missing. Without an
        # independently supplied denominator, completeness remains unknown and
        # must never self-certify as 100%/success.
        coverage_ratio = 0.0
        canonical_ratio = 0.0
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
    "audit_amount_consistency",
    "audit_cqf",
    "audit_row_accounting",
    "canonical_expected_from_parse_result",
    "is_canonical_row",
    "physical_transaction_row_estimate",
    "resolve_extract_status",
]
