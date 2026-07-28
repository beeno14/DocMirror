# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guarded continuation resolution for native enterprise credit reports.

The resolver deliberately works above sealed physical tables.  It never joins
tables merely because they are adjacent or have the same number of columns.
Every accepted continuation must satisfy a named family contract, be the next
physical table, stay within the configured page gap, and pass a row-level
shape validator.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable

Row = list[str]
RowPredicate = Callable[[Row], bool]


@dataclass(frozen=True)
class TableFragment:
    """One sealed physical table in document reading order."""

    index: int
    page: int
    table_id: str
    rows: tuple[tuple[str, ...], ...]

    def mutable_rows(self) -> list[Row]:
        return [list(row) for row in self.rows]


@dataclass(frozen=True)
class ContinuationContract:
    """A declarative, family-specific permission to follow one table."""

    name: str
    expected_columns: frozenset[int]
    row_predicate: RowPredicate
    max_page_gap: int = 1
    forbidden_markers: tuple[str, ...] = ()


@dataclass(frozen=True)
class ContinuationMatch:
    """An accepted continuation with auditable source coordinates."""

    contract: str
    fragment: TableFragment
    row_index: int
    row: tuple[str, ...]


@dataclass(frozen=True)
class ContinuationRejection:
    """Why a possible continuation was not consumed."""

    contract: str
    source_table_id: str
    candidate_table_id: str
    reason: str


def _raw_rows(table: Any) -> list[Row]:
    metadata = dict(getattr(table, "metadata", None) or {})
    raw_rows = metadata.get("raw_rows")
    if isinstance(raw_rows, list) and raw_rows:
        return [
            [str(value or "").replace("\n", "").strip() for value in row] for row in raw_rows if isinstance(row, list)
        ]
    rows: list[Row] = []
    headers = list(getattr(table, "headers", None) or [])
    if headers:
        rows.append([str(value or "").replace("\n", "").strip() for value in headers])
    for row in getattr(table, "rows", None) or []:
        cells = getattr(row, "cells", None) or []
        rows.append([str(getattr(cell, "text", cell) or "").replace("\n", "").strip() for cell in cells])
    return rows


class EnterpriseContinuationResolver:
    """Resolve only explicitly authorized enterprise table continuations."""

    def __init__(self, parse_result: Any):
        self.rejections: list[ContinuationRejection] = []
        prebuilt = getattr(parse_result, "continuation_fragments", None)
        if prebuilt is not None:
            self.fragments = tuple(prebuilt)
            return
        fragments: list[TableFragment] = []
        for page in getattr(parse_result, "pages", None) or []:
            page_number = int(getattr(page, "source_page_number", 0) or getattr(page, "page_number", 0) or 0)
            for table in getattr(page, "tables", None) or []:
                rows = _raw_rows(table)
                if not rows:
                    continue
                fragments.append(
                    TableFragment(
                        index=len(fragments),
                        page=page_number,
                        table_id=str(getattr(table, "table_id", "") or ""),
                        rows=tuple(tuple(value for value in row) for row in rows),
                    )
                )
        self.fragments = tuple(fragments)

    def following_row(
        self,
        source: TableFragment,
        contract: ContinuationContract,
        *,
        candidate_row_index: int = 0,
    ) -> ContinuationMatch | None:
        """Return the next row only when every contract guard passes."""
        candidate_index = source.index + 1
        if candidate_index >= len(self.fragments):
            return None
        candidate = self.fragments[candidate_index]
        page_gap = candidate.page - source.page
        if page_gap < 0 or page_gap > contract.max_page_gap:
            self._reject(source, candidate, contract, "page_gap")
            return None
        if candidate_row_index >= len(candidate.rows):
            self._reject(source, candidate, contract, "missing_candidate_row")
            return None
        row = list(candidate.rows[candidate_row_index])
        signature = "".join(row)
        if len(row) not in contract.expected_columns:
            self._reject(source, candidate, contract, "column_shape")
            return None
        if any(marker in signature for marker in contract.forbidden_markers):
            self._reject(source, candidate, contract, "new_header")
            return None
        if not contract.row_predicate(row):
            self._reject(source, candidate, contract, "row_semantics")
            return None
        return ContinuationMatch(
            contract=contract.name,
            fragment=candidate,
            row_index=candidate_row_index,
            row=tuple(row),
        )

    def _reject(
        self,
        source: TableFragment,
        candidate: TableFragment,
        contract: ContinuationContract,
        reason: str,
    ) -> None:
        self.rejections.append(
            ContinuationRejection(
                contract=contract.name,
                source_table_id=source.table_id,
                candidate_table_id=candidate.table_id,
                reason=reason,
            )
        )

    def audit_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "contract": item.contract,
                "source_table_id": item.source_table_id,
                "candidate_table_id": item.candidate_table_id,
                "reason": item.reason,
            }
            for item in self.rejections
        ]


def numeric_row(
    row: Iterable[str],
    *,
    numeric_indexes: Iterable[int],
    nonempty_indexes: Iterable[int] = (),
) -> bool:
    """Conservative validator shared by declarative continuation contracts."""
    values = list(row)
    for index in nonempty_indexes:
        if index >= len(values) or not str(values[index] or "").strip():
            return False
    for index in numeric_indexes:
        if index >= len(values):
            return False
        raw = str(values[index] or "").replace(",", "").replace("，", "").replace(" ", "")
        if raw in {"", "-", "--", "—"}:
            return False
        try:
            float(raw)
        except ValueError:
            return False
    return True


def _date_like(value: Any) -> bool:
    raw = str(value or "").replace(" ", "").strip()
    return bool(
        re.fullmatch(
            r"(?:19|20)\d{2}[-年./]\d{1,2}[-月./]\d{1,2}日?",
            raw,
        )
    )


def _number_like(value: Any) -> bool:
    raw = str(value or "").replace(",", "").replace("，", "").replace(" ", "").strip()
    if raw in {"", "-", "--", "—"}:
        return False
    try:
        float(raw)
    except ValueError:
        return False
    return True


def _settled_account_detail_row(row: Row) -> bool:
    values = [str(value or "").strip() for value in row]
    if len(values) != 8:
        return False
    suffix = re.sub(r"[^0-9A-Z]", "", values[0].upper())
    return bool(
        (not suffix or len(suffix) <= 16)
        and _date_like(values[1])
        and values[2] in {"正常", "关注", "次级", "可疑", "损失", "违约", "未分类"}
        and _date_like(values[3])
        and (not values[5] or "还款" in values[5])
    )


def _attachment_history_continuation_row(row: Row) -> bool:
    values = [str(value or "").strip() for value in row]
    signature = "".join(values)
    if "逾期月数" in signature and "最近一次约定还款日期" in signature and "最近一次还款形式" in signature:
        return True

    dense = [value for value in values if value]
    if not dense or not 6 <= len(dense) <= 8:
        return False
    if _date_like(values[0] if values else ""):
        return True
    if values and not values[0]:
        return bool(any(_date_like(value) for value in values[1:]) and any(_number_like(value) for value in values[1:]))
    return False


FACILITY_VALUE_CONTRACT = ContinuationContract(
    name="facility_summary_values",
    expected_columns=frozenset({6}),
    row_predicate=lambda row: numeric_row(row, numeric_indexes=range(6)),
    forbidden_markers=("非循环信用额度", "循环信用额度", "总额", "已用额度"),
)

CLOSED_SUMMARY_BODY_CONTRACT = ContinuationContract(
    name="closed_credit_summary_body",
    expected_columns=frozenset({5}),
    row_predicate=lambda row: numeric_row(
        row,
        numeric_indexes=range(1, 5),
        nonempty_indexes=(0,),
    ),
    forbidden_markers=("正常类账户数", "关注类账户数", "不良类账户数"),
)

ACCOUNT_SETTLED_DETAIL_CONTRACT = ContinuationContract(
    name="enterprise_account_settled_detail",
    expected_columns=frozenset({8}),
    row_predicate=_settled_account_detail_row,
    forbidden_markers=("账户编号", "授信机构", "业务种类", "借款金额"),
)

ATTACHMENT_HISTORY_BODY_CONTRACT = ContinuationContract(
    name="enterprise_attachment_history_body",
    expected_columns=frozenset({7, 8, 11}),
    row_predicate=_attachment_history_continuation_row,
    forbidden_markers=("账户编号", "授信机构", "开户日期", "开立日期", "信息报告日期"),
)

__all__ = [
    "ACCOUNT_SETTLED_DETAIL_CONTRACT",
    "ATTACHMENT_HISTORY_BODY_CONTRACT",
    "CLOSED_SUMMARY_BODY_CONTRACT",
    "ContinuationContract",
    "ContinuationMatch",
    "ContinuationRejection",
    "EnterpriseContinuationResolver",
    "FACILITY_VALUE_CONTRACT",
    "TableFragment",
    "numeric_row",
]
