# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Single-column signed amount bank ledger style parser.

Handles ledgers with one amount column where sign denotes direction (+ income /
- expense) rather than separate debit/credit columns.

Pipeline role: registered as ``signed_amount`` in ``style_registry`` when detector
identifies signed-amount layout signatures.

Key exports: ``PARSER_ID``, ``STYLE_ID``, ``parse_signed_amount``, ``extract_transactions``.

Dependencies: ``grid_standard`` (shared row paths), ``standardizer.normalize_amount``.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import defaultdict, deque
from typing import Any

from docmirror.plugins._base.standardizer import normalize_amount
from docmirror.plugins.bank_statement.context import StyleContext
from docmirror.plugins.bank_statement.header_resolve import normalize_header_cell
from docmirror.plugins.bank_statement.styles import grid_standard

PARSER_ID = "signed_amount"
STYLE_ID = "signed_amount"

_SIGNED_PREFIX_RE = re.compile(r"^[+-]")
_AMOUNT_HEADER_NEEDLES = ("交易金额", "金额", "发生额")
_SPLIT_HEADER_NEEDLES = ("收入", "支出", "借方发生额", "贷方发生额")
_STRICT_MONEY_CELL_RE = re.compile(r"^[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)\.\d{2}$")
_COMPACT_DATE_CELL_RE = re.compile(r"^(?:19|20)\d{6}$")
_PROMOTED_VOUCHER_RE = re.compile(r"(?=.*\d)[A-Za-z0-9]{8,32}")
_PROMOTED_HEADER_ROLE_ALIASES: dict[str, tuple[str, ...]] = {
    "date": ("日期", "交易日期", "Date", "Transaction Date"),
    "transaction_type": (
        "业务类型",
        "Business Type",
        "业务类型 Business Type",
        "Transaction Type",
    ),
    "voucher": ("票据号", "Bill No.", "票据号 Bill No.", "Voucher No."),
    "summary": ("摘要", "Description", "摘要 Description", "Summary"),
    "amount": (
        "借方/贷方金额",
        "Debit/Credit Amount",
        "借方/贷方金额 Debit/Credit Amount",
        "Signed Amount",
    ),
    "balance": ("余额", "Balance", "余额 Balance", "Account Balance"),
    "counterparty": (
        "对手户名",
        "Counterparty Account Name",
        "对手户名 Counterparty Account Name",
        "Counterparty Name",
    ),
}
_REQUIRED_PROMOTED_HEADER_ROLES = frozenset(_PROMOTED_HEADER_ROLE_ALIASES)


def _cell_value(raw_txn: dict[str, str], *needles: str) -> str:
    for key, value in raw_txn.items():
        normalized_key = normalize_header_cell(key)
        for needle in needles:
            normalized_needle = normalize_header_cell(needle)
            if normalized_key == normalized_needle or normalized_needle in normalized_key:
                return str(value or "").strip()
    return ""


def parse_signed_amount(text: str) -> tuple[float | None, str]:
    """Return (abs_amount, direction) from a signed amount string."""
    raw = (text or "").strip().replace(",", "").replace("¥", "").replace("￥", "")
    if not raw:
        return None, "other"
    if raw.startswith("-"):
        amount = normalize_amount(raw)
        return (abs(float(amount)), "expense") if amount is not None else (None, "other")
    if raw.startswith("+"):
        amount = normalize_amount(raw.lstrip("+"))
        return (float(amount), "income") if amount is not None else (None, "other")
    amount = normalize_amount(raw)
    if amount is None:
        return None, "other"
    return abs(float(amount)), "income"


def table_has_signed_amount_cells(tables: list[list[list[str]]]) -> bool:
    """True when one amount column contains enough sign evidence for direction."""
    for tbl in tables:
        if not tbl:
            continue
        header_idx = -1
        amount_col = -1
        for i, row in enumerate(tbl[:10]):
            for j, cell in enumerate(row):
                text = normalize_header_cell(cell)
                if any(n in text for n in _AMOUNT_HEADER_NEEDLES):
                    # Distinguish merged columns (收入/支出金额) from true split columns.
                    # Merged columns contain both income and expense keywords in one cell;
                    # true split columns have separate cells for income and expense.
                    if any(s in text for s in _SPLIT_HEADER_NEEDLES):
                        # Check if this is a merged header (both income + expense in one cell)
                        from docmirror.plugins.bank_statement.header_resolve import has_merged_amount_header

                        if has_merged_amount_header([tbl]):
                            # Merged column like 收入/支出金额 → don't short-circuit,
                            # let the signed-amount path run.
                            pass
                        else:
                            return False
                    header_idx = i
                    amount_col = j
                    break
            if amount_col >= 0:
                break
        if amount_col < 0:
            continue

        signed_rows = 0
        checked_rows = 0
        for row in tbl[header_idx + 1 : header_idx + 12]:
            if amount_col >= len(row):
                continue
            cell = str(row[amount_col] or "").strip()
            if not cell or not re.search(r"\d", cell):
                continue
            checked_rows += 1
            if _SIGNED_PREFIX_RE.match(cell):
                signed_rows += 1
        # Many bank ledgers omit the plus sign for credits while retaining a
        # minus sign for debits. One explicit sign makes unsigned positives
        # unambiguous only when the source has a single amount column.
        if checked_rows >= 2 and signed_rows >= 1:
            return True
    return False


def extract_transactions(ctx: StyleContext, plugin: Any) -> list[dict[str, str]]:
    transactions = grid_standard.extract_transactions(ctx, plugin)
    if transactions and not _has_promoted_table_candidate(ctx.parse_result):
        return transactions

    recovered = _recover_promoted_signed_amount_rows(ctx.parse_result)
    if not recovered:
        return transactions
    recovered_transactions = grid_standard._finalize_transactions(recovered, ctx.parse_result, ctx.full_text)
    if len(recovered_transactions) > len(transactions):
        return recovered_transactions
    return transactions


def _compact_source_text(value: Any) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(value or ""))).strip()


_PROMOTED_HEADER_ROLE_SIGNATURES = {
    role: {_compact_source_text(alias) for alias in aliases}
    for role, aliases in _PROMOTED_HEADER_ROLE_ALIASES.items()
}


def _read_parse_result(parse_result: Any) -> Any:
    if parse_result is None:
        return None
    if getattr(parse_result, "pages", None) is not None:
        return parse_result
    to_read_view = getattr(parse_result, "to_read_view", None)
    if callable(to_read_view):
        try:
            return to_read_view()
        except Exception:
            return parse_result
    return parse_result


def _promoted_header_roles(page: Any) -> frozenset[str]:
    """Return exact issuer-neutral ledger roles present as source text atoms."""
    atom_signatures = {
        signature
        for text in (getattr(page, "texts", []) or [])
        if (signature := _compact_source_text(getattr(text, "content", "")))
    }
    return frozenset(
        role
        for role, signatures in _PROMOTED_HEADER_ROLE_SIGNATURES.items()
        if atom_signatures.intersection(signatures)
    )


def _has_promoted_header_contract(page: Any) -> bool:
    return _promoted_header_roles(page) == _REQUIRED_PROMOTED_HEADER_ROLES


def _has_promoted_table_metadata(table: Any) -> bool:
    metadata = getattr(table, "metadata", None) or {}
    return bool(
        isinstance(metadata, dict)
        and metadata.get("header_source") == "data_row"
        and metadata.get("preserve_headers") is False
        and metadata.get("source") == "geometric_reconstructor"
    )


def _table_body_values(row: Any) -> list[str]:
    return [
        str(getattr(cell, "cleaned", None) or getattr(cell, "text", "") or "").strip()
        for cell in (getattr(row, "cells", []) or [])
    ]


def _looks_like_promoted_transaction(values: list[str]) -> bool:
    if not values or not _COMPACT_DATE_CELL_RE.fullmatch(_compact_source_text(values[0])):
        return False
    return any(_STRICT_MONEY_CELL_RE.fullmatch(_compact_source_text(value)) for value in values[1:])


def _promoted_transaction_like_tables(page: Any) -> list[Any]:
    candidates: list[Any] = []
    for table in getattr(page, "tables", []) or []:
        if not _has_promoted_table_metadata(table):
            continue
        promoted = [str(value or "").strip() for value in (getattr(table, "headers", []) or [])]
        body_rows = [_table_body_values(row) for row in (getattr(table, "rows", []) or [])]
        if any(_looks_like_promoted_transaction(values) for values in (promoted, *body_rows)):
            candidates.append(table)
    return candidates


def _has_promoted_table_candidate(parse_result: Any) -> bool:
    """Return whether an exact promoted-ledger contract makes a retry relevant."""
    parse_result = _read_parse_result(parse_result)
    if parse_result is None:
        return False
    for page in getattr(parse_result, "pages", []) or []:
        if not _has_promoted_header_contract(page):
            continue
        if _promoted_transaction_like_tables(page):
            return True
    return False


def _strict_money_pair(values: list[str]) -> tuple[int, int] | None:
    money_indexes = [
        index
        for index, value in enumerate(values)
        if _STRICT_MONEY_CELL_RE.fullmatch(_compact_source_text(value))
    ]
    adjacent_pairs = [
        (left, right)
        for left, right in zip(money_indexes, money_indexes[1:])
        if right == left + 1
    ]
    return adjacent_pairs[0] if len(adjacent_pairs) == 1 else None


def _recover_promoted_row(values: list[str]) -> dict[str, str] | None:
    source_values = [str(value or "").strip() for value in values]
    while source_values and not source_values[-1]:
        source_values.pop()
    if len(source_values) < 6 or not _COMPACT_DATE_CELL_RE.fullmatch(_compact_source_text(source_values[0])):
        return None

    money_pair = _strict_money_pair(source_values)
    if money_pair is None:
        return None
    amount_index, balance_index = money_pair
    if amount_index < 3 or balance_index + 1 >= len(source_values):
        return None

    narrative = [value for value in source_values[2:amount_index] if value]
    if not narrative:
        return None
    bill_number = ""
    if len(narrative) >= 2 and _PROMOTED_VOUCHER_RE.fullmatch(_compact_source_text(narrative[0])):
        bill_number = narrative.pop(0)
    summary = "\n".join(narrative).strip()
    counterparty = "\n".join(value for value in source_values[balance_index + 1 :] if value).strip()
    if not summary or not counterparty:
        return None

    return {
        "日期": source_values[0],
        "业务类型": source_values[1],
        "票据号": bill_number,
        "摘要": summary,
        "借方/贷方金额": source_values[amount_index],
        "余额": source_values[balance_index],
        "对手户名": counterparty,
    }


def _source_text_rows(page: Any) -> dict[str, deque[Any]]:
    rows: dict[str, deque[Any]] = defaultdict(deque)
    for text in getattr(page, "texts", []) or []:
        signature = _compact_source_text(getattr(text, "content", ""))
        if signature:
            rows[signature].append(text)
    return rows


def _source_text_transaction_count(page: Any) -> int:
    count = 0
    for text in getattr(page, "texts", []) or []:
        values = [str(value or "").strip() for value in str(getattr(text, "content", "") or "").splitlines()]
        if not values or not _COMPACT_DATE_CELL_RE.fullmatch(_compact_source_text(values[0])):
            continue
        if sum(bool(_STRICT_MONEY_CELL_RE.fullmatch(_compact_source_text(value))) for value in values) >= 2:
            count += 1
    return count


def _source_for_promoted_row(
    text: Any,
    *,
    page_number: int,
    table_id: str,
    row_index: int,
    row_role: str,
    reconstructed_row_index: int,
) -> dict[str, Any] | None:
    raw_bbox = getattr(text, "bbox", None)
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        return None
    try:
        bbox = [float(value) for value in raw_bbox]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in bbox) or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None
    evidence_ids = list(
        dict.fromkeys(
            str(value).strip()
            for value in (getattr(text, "evidence_ids", []) or [])
            if str(value).strip()
        )
    )
    if not evidence_ids:
        return None
    source_ref = {
        "source": "canonical_page_text",
        "source_page": page_number,
        "page_range": [page_number, page_number],
        "bbox": bbox,
        "evidence_ids": evidence_ids,
    }
    return {
        "source": "promoted_data_row_table",
        "source_page": page_number,
        "page_id": f"page:{page_number:04d}",
        "page_range": [page_number, page_number],
        "table_id": table_id,
        "source_row_index": row_index,
        "source_row_role": row_role,
        "reconstructed_row_index": reconstructed_row_index,
        **({"header_source": "data_row"} if row_role == "promoted_header" else {}),
        "evidence_ids": evidence_ids,
        "source_refs": [source_ref],
    }


def _recover_promoted_signed_amount_rows(parse_result: Any) -> list[dict[str, Any]]:
    """Recover a signed ledger whose geometric header is its first transaction.

    The branch is issuer-neutral and all-or-nothing.  It requires an exact set
    of source header roles, explicit ``data_row`` promotion metadata, a unique
    amount/balance pair in every transaction, and a matching canonical text
    block for every row.
    """
    parse_result = _read_parse_result(parse_result)
    if parse_result is None:
        return []

    recovered: list[dict[str, Any]] = []
    pages = list(getattr(parse_result, "pages", []) or [])
    source_transaction_count = sum(_source_text_transaction_count(page) for page in pages)
    for page in pages:
        candidate_tables = _promoted_transaction_like_tables(page)
        if not candidate_tables:
            continue
        if not _has_promoted_header_contract(page):
            return []
        page_number = int(getattr(page, "source_page_number", 0) or getattr(page, "page_number", 0) or 0)
        if page_number <= 0:
            return []
        text_rows = _source_text_rows(page)
        page_recovered: list[dict[str, Any]] = []

        for table in candidate_tables:
            promoted = [str(value or "").strip() for value in (getattr(table, "headers", []) or [])]
            row_values: list[tuple[list[str], int, str]] = [(promoted, -1, "promoted_header")]
            body_source_indexes: list[int] = []
            for row in getattr(table, "rows", []) or []:
                values = _table_body_values(row)
                try:
                    source_row_index = getattr(row, "source_row_index", None)
                    if isinstance(source_row_index, bool):
                        return []
                    physical_row_index = int(source_row_index)
                except (TypeError, ValueError):
                    return []
                if physical_row_index < 0:
                    return []
                body_source_indexes.append(physical_row_index)
                row_values.append((values, physical_row_index, "table_body"))
            if len(set(body_source_indexes)) != len(body_source_indexes) or any(
                right <= left for left, right in zip(body_source_indexes, body_source_indexes[1:])
            ):
                return []
            table_id = str(getattr(table, "table_id", "") or "")
            if not table_id:
                return []

            table_recovered: list[dict[str, Any]] = []
            for reconstructed_index, (values, source_index, source_role) in enumerate(row_values):
                if not values or not _COMPACT_DATE_CELL_RE.fullmatch(_compact_source_text(values[0])):
                    continue
                transaction = _recover_promoted_row(values)
                signature = _compact_source_text("\n".join(values))
                matching_texts = text_rows.get(signature)
                if transaction is None or not matching_texts:
                    return []
                source_text = matching_texts.popleft()
                source = _source_for_promoted_row(
                    source_text,
                    page_number=page_number,
                    table_id=table_id,
                    row_index=source_index,
                    row_role=source_role,
                    reconstructed_row_index=reconstructed_index,
                )
                if source is None:
                    return []
                transaction["_source"] = source
                table_recovered.append(transaction)

            if table_recovered:
                page_recovered.extend(table_recovered)

        recovered.extend(page_recovered)
    if recovered and len(recovered) != source_transaction_count:
        return []
    return recovered


def normalize_record(raw_txn: dict[str, str], plugin: Any) -> dict[str, Any]:
    amount_text = _cell_value(raw_txn, *_AMOUNT_HEADER_NEEDLES)
    parsed_amount, direction = parse_signed_amount(amount_text)
    normalized = grid_standard.normalize_record(raw_txn, plugin)
    if parsed_amount is not None:
        normalized["amount"] = parsed_amount
        normalized["amount_cny"] = parsed_amount
        normalized["direction"] = direction
    return normalized
