# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schema-guided transaction candidates from canonical digital page text.

Some native PDFs expose each visual ledger row as one newline-delimited
``TextBlock``.  A generic geometric table can be absent even though that block
already preserves the business row boundary.  This module treats a proven
six-role signed-ledger schema as document context and emits page-local source
records directly.  It never manufactures rows, requires exact date/currency/
money anchors in every emitted block, and does not use balance continuity as an
admission condition.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from docmirror.plugins.bank_statement.extraction_dispatch import BankExtractionRoute
from docmirror.plugins.bank_statement.header_resolve import normalize_header_cell
from docmirror.plugins.bank_statement.work_cache import memoize_bank_document_work

_ROLE_ORDER = ("date", "currency", "amount", "balance", "summary", "counterparty")
_ROLE_HEADERS = {
    "date": "记账日期",
    "currency": "货币",
    "amount": "交易金额",
    "balance": "联机余额",
    "summary": "交易摘要",
    "counterparty": "对手信息",
}
_ROLE_ALIASES = {
    "date": ("记账日期", "交易日期", "交易时间", "Date", "Transaction Date"),
    "currency": ("货币", "币种", "币别", "Currency"),
    "amount": ("交易金额", "发生额", "借方/贷方金额", "Transaction Amount", "Amount"),
    "balance": ("联机余额", "账户余额", "余额", "Balance", "Account Balance"),
    "summary": ("交易摘要", "摘要", "交易类型", "Transaction Type", "Summary"),
    "counterparty": ("对手信息", "对方信息", "对手户名", "Counter Party", "Counterparty"),
}
_ROLE_SIGNATURES = {
    role: frozenset(normalize_header_cell(alias) for alias in aliases)
    for role, aliases in _ROLE_ALIASES.items()
}
_DATE_RE = re.compile(
    r"^(?:19|20)\d{2}[-/.]\d{1,2}[-/.]\d{1,2}"
    r"(?:[ T]\d{1,2}:\d{2}(?::\d{2})?)?$"
)
_CURRENCY_RE = re.compile(r"^[A-Z]{3}$")
_MONEY_RE = re.compile(r"^[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)\.\d{2}$")


@dataclass(frozen=True)
class SchemaGuidedPageText:
    """One document candidate and the pages whose schema was inherited."""

    records: list[dict[str, Any]]
    schema_pages: tuple[int, ...]
    inherited_pages: tuple[int, ...]

    @property
    def expected_rows(self) -> int:
        return len(self.records)


def _read_parse_result(parse_result: Any) -> Any:
    if parse_result is None or getattr(parse_result, "pages", None) is not None:
        return parse_result
    to_read_view = getattr(parse_result, "to_read_view", None)
    if callable(to_read_view):
        try:
            return to_read_view()
        except Exception:
            return parse_result
    return parse_result


def _text_lines(value: Any) -> list[str]:
    return [
        unicodedata.normalize("NFKC", line).strip()
        for line in str(value or "").splitlines()
        if line.strip()
    ]


def _header_roles(value: Any) -> tuple[str, ...] | None:
    roles: list[str] = []
    for line in _text_lines(value):
        signature = normalize_header_cell(line)
        matches = [role for role, signatures in _ROLE_SIGNATURES.items() if signature in signatures]
        if len(matches) != 1:
            return None
        roles.append(matches[0])
    return tuple(roles) if tuple(roles) == _ROLE_ORDER else None


def _page_has_schema(page: Any) -> bool:
    if any(_header_roles(getattr(text, "content", "")) is not None for text in getattr(page, "texts", []) or []):
        return True
    for table in getattr(page, "tables", []) or []:
        if _header_roles("\n".join(str(value or "") for value in getattr(table, "headers", []) or [])) is not None:
            return True
    return False


def _valid_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        bbox = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in bbox) or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None
    return bbox


def _transaction_values(value: Any) -> dict[str, str] | None:
    lines = _text_lines(value)
    if len(lines) < 5:
        return None
    date_value, currency, amount, balance = lines[:4]
    if (
        _DATE_RE.fullmatch(date_value) is None
        or _CURRENCY_RE.fullmatch(currency.upper()) is None
        or _MONEY_RE.fullmatch(amount) is None
        or _MONEY_RE.fullmatch(balance) is None
    ):
        return None
    summary = lines[4]
    if not summary or normalize_header_cell(summary) in set().union(*_ROLE_SIGNATURES.values()):
        return None
    return {
        _ROLE_HEADERS["date"]: date_value,
        _ROLE_HEADERS["currency"]: currency,
        _ROLE_HEADERS["amount"]: amount,
        _ROLE_HEADERS["balance"]: balance,
        _ROLE_HEADERS["summary"]: summary,
        _ROLE_HEADERS["counterparty"]: "\n".join(lines[5:]),
    }


def _row_source(
    text: Any,
    *,
    page_number: int,
    block_index: int,
    schema_page: int,
    inherited_schema: bool,
) -> dict[str, Any] | None:
    bbox = _valid_bbox(getattr(text, "bbox", None))
    evidence_ids = list(
        dict.fromkeys(
            str(value).strip()
            for value in (getattr(text, "evidence_ids", None) or [])
            if str(value).strip()
        )
    )
    if bbox is None or not evidence_ids:
        return None
    source_ref = {
        "source": "canonical_page_text",
        "source_page": page_number,
        "page_range": [page_number, page_number],
        "block_index": block_index,
        "bbox": bbox,
        "evidence_ids": evidence_ids,
    }
    return {
        "source": "schema_guided_page_text",
        "source_page": page_number,
        "page_id": f"page:{page_number:04d}",
        "page_range": [page_number, page_number],
        "table_id": f"schema_page_text:{schema_page:04d}",
        "source_row_index": block_index,
        "schema_source_page": schema_page,
        "schema_inherited": inherited_schema,
        "bbox": bbox,
        "evidence_ids": evidence_ids,
        "source_refs": [source_ref],
    }


@memoize_bank_document_work
def recover_schema_guided_page_text(
    parse_result: Any,
    *,
    source_route: str | None = None,
) -> SchemaGuidedPageText:
    """Emit exact page-text rows under a proven, carried digital ledger schema."""

    if source_route not in (None, "", BankExtractionRoute.DIGITAL.value):
        return SchemaGuidedPageText([], (), ())
    parse_result = _read_parse_result(parse_result)
    if parse_result is None:
        return SchemaGuidedPageText([], (), ())

    records: list[dict[str, Any]] = []
    schema_pages: list[int] = []
    inherited_pages: list[int] = []
    active_schema_page = 0
    pages = sorted(
        getattr(parse_result, "pages", []) or [],
        key=lambda page: int(getattr(page, "source_page_number", 0) or getattr(page, "page_number", 0) or 0),
    )
    for page in pages:
        try:
            page_number = int(
                getattr(page, "source_page_number", 0) or getattr(page, "page_number", 0) or 0
            )
        except (TypeError, ValueError):
            active_schema_page = 0
            continue
        if page_number <= 0:
            active_schema_page = 0
            continue

        local_schema = _page_has_schema(page)
        if local_schema:
            active_schema_page = page_number
            schema_pages.append(page_number)
        if active_schema_page <= 0:
            continue

        page_records: list[dict[str, Any]] = []
        for block_index, text in enumerate(getattr(page, "texts", []) or []):
            transaction = _transaction_values(getattr(text, "content", ""))
            if transaction is None:
                continue
            source = _row_source(
                text,
                page_number=page_number,
                block_index=block_index,
                schema_page=active_schema_page,
                inherited_schema=not local_schema,
            )
            if source is None:
                continue
            transaction["_source"] = source
            page_records.append(transaction)

        if page_records:
            records.extend(page_records)
            if not local_schema:
                inherited_pages.append(page_number)
        elif not local_schema:
            # Do not carry one statement schema across an unrelated empty page.
            active_schema_page = 0

    return SchemaGuidedPageText(
        records,
        tuple(dict.fromkeys(schema_pages)),
        tuple(dict.fromkeys(inherited_pages)),
    )


__all__ = ["SchemaGuidedPageText", "recover_schema_guided_page_text"]
