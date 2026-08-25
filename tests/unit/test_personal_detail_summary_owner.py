# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from docmirror.plugins.credit_report.personal_detail_scanned.native_extraction import (
    _extract_summary_datasets,
)

_SUMMARY_ROLE = "information_summary"


def _table(
    table_id: str,
    rows: list[list[str]],
    *,
    role: str,
    logical_page: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        table_id=table_id,
        metadata={
            "raw_rows": rows,
            "canonical_template_id": role,
            "source_logical_page": logical_page,
            "source_page": logical_page,
        },
        headers=[],
        rows=[],
        bbox=[20.0, 20.0, 580.0, 220.0],
    )


def _context(
    tables: list[SimpleNamespace],
    *,
    page_role: str,
    logical_page: int,
) -> SimpleNamespace:
    page = SimpleNamespace(
        page_number=logical_page,
        source_page_number=logical_page,
        canonical_template_id=page_role,
        tables=tables,
        texts=[],
    )
    return SimpleNamespace(
        pages=[page],
        reading_order_by_logical={logical_page: 1},
        tables_continue=lambda _left, _right: False,
        corrected_evidence_pages=lambda: [],
        _personal_detail_extraction_issues=[],
    )


def _overview_rows(population: list[tuple[str, str, str, str]]) -> list[list[str]]:
    return [
        ["信用业务概要", "", "", ""],
        ["业务分组", "业务类型", "账户数", "首笔业务发放月份"],
        *[list(row) for row in population],
    ]


def test_summary_shaped_public_information_table_is_not_admitted() -> None:
    logical_page = 9
    table = _table(
        "public-summary-lookalike",
        _overview_rows([("贷款", "个人住房贷款", "1", "2024.01")]),
        role="public_information",
        logical_page=logical_page,
    )
    context = _context(
        [table],
        page_role="public_information",
        logical_page=logical_page,
    )

    records, cells = _extract_summary_datasets(context)

    assert records == []
    assert cells == []


@pytest.mark.parametrize(
    ("owned_table_index", "population"),
    [
        (0, [("贷款", "个人住房贷款", "2", "2018.04")]),
        (
            1,
            [
                ("贷款", "个人商用房贷款（包括商住两用房）", "3", "2019.05"),
                ("信用卡", "贷记卡", "4", "2020.06"),
                ("贷款", "其他类贷款", "5", "2021.07"),
            ],
        ),
    ],
)
def test_owned_summary_admission_is_independent_of_table_order_and_population(
    owned_table_index: int,
    population: list[tuple[str, str, str, str]],
) -> None:
    logical_page = 4 + len(population)
    summary = _table(
        "owned-summary",
        _overview_rows(population),
        role=_SUMMARY_ROLE,
        logical_page=logical_page,
    )
    foreign = _table(
        "foreign-lookalike",
        _overview_rows([("贷款", "个人住房贷款", "99", "2001.01")]),
        role="public_information",
        logical_page=logical_page,
    )
    tables = [summary, foreign]
    if owned_table_index == 1:
        tables.reverse()
    context = _context(
        tables,
        page_role=_SUMMARY_ROLE,
        logical_page=logical_page,
    )

    records, cells = _extract_summary_datasets(context)

    assert len(records) == 1
    assert records[0]["source_table_id"] == "owned-summary"
    assert records[0]["source_row_count"] == len(population)
    assert [cell["value"] for cell in cells if cell["column_label"] == "业务类型"] == [
        row[1] for row in population
    ]
    assert [cell["value"] for cell in cells if cell["column_label"] == "账户数"] == [
        row[2] for row in population
    ]
    assert all(cell["value"] != "99" for cell in cells)
