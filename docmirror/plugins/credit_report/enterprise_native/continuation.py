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
    require_next_physical_table: bool = True


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
            [str(value or "").replace("\n", "").strip() for value in row]
            for row in raw_rows
            if isinstance(row, list)
        ]
    rows: list[Row] = []
    headers = list(getattr(table, "headers", None) or [])
    if headers:
        rows.append([str(value or "").replace("\n", "").strip() for value in headers])
    for row in getattr(table, "rows", None) or []:
        cells = getattr(row, "cells", None) or []
        rows.append(
            [
                str(getattr(cell, "text", cell) or "").replace("\n", "").strip()
                for cell in cells
            ]
        )
    return rows


class EnterpriseContinuationResolver:
    """Resolve only explicitly authorized enterprise table continuations."""

    def __init__(self, parse_result: Any):
        fragments: list[TableFragment] = []
        for page in getattr(parse_result, "pages", None) or []:
            page_number = int(
                getattr(page, "source_page_number", 0)
                or getattr(page, "page_number", 0)
                or 0
            )
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
        self.rejections: list[ContinuationRejection] = []

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
        raw = (
            str(values[index] or "")
            .replace(",", "")
            .replace("，", "")
            .replace(" ", "")
        )
        if raw in {"", "-", "--", "—"}:
            return False
        try:
            float(raw)
        except ValueError:
            return False
    return True


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

ATTACHMENT_DETAIL_BODY_CONTRACT = ContinuationContract(
    name="attachment_credit_detail_body",
    expected_columns=frozenset({6, 7, 8, 9, 10}),
    row_predicate=lambda row: any(str(value or "").strip() for value in row),
    forbidden_markers=("账户编号", "开户日期", "开立日期", "信息报告日期"),
)


__all__ = [
    "ATTACHMENT_DETAIL_BODY_CONTRACT",
    "CLOSED_SUMMARY_BODY_CONTRACT",
    "ContinuationContract",
    "ContinuationMatch",
    "ContinuationRejection",
    "EnterpriseContinuationResolver",
    "FACILITY_VALUE_CONTRACT",
    "TableFragment",
    "numeric_row",
]
