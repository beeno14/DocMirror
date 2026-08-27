# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic regressions for wrapped native bank grids and logical pseudo-headers."""

from __future__ import annotations

import pytest

from docmirror.models.entities.parse_result import (
    CellValue,
    DocumentEntities,
    LogicalTable,
    PageContent,
    ParseResult,
    RowProvenance,
    TableRow,
)
from docmirror.plugins.bank_statement.community_plugin import BankStatementCommunityPlugin
from docmirror.plugins.bank_statement.context import StyleContext
from docmirror.plugins.bank_statement.extraction_dispatch import (
    BankExtractionPolicy,
    BankExtractionRoute,
)
from docmirror.plugins.bank_statement.ltro import ReconstructionMeta
from docmirror.plugins.bank_statement.row_extract import extract_logical_rows_with_provenance
from docmirror.plugins.bank_statement.style_detector import BankStyleDetector, StyleDetectionResult
from docmirror.plugins.bank_statement.style_registry import BankStyleParserRegistry
from docmirror.plugins.bank_statement.styles.grid_standard import (
    _finalize_transactions,
    normalize_record,
    refine_missing_directions_from_balance_chain,
)

_WIDE_HEADERS = [
    "账号",
    "交易时间",
    "借方发生额",
    "贷方发生额",
    "余额",
    "币种",
    "对方户名",
    "对方账号",
    "对方开户机构",
    "记账日期",
    "摘要",
    "备注",
    "账户明细编号-交易流水号",
    "企业流水号",
    "凭证种类",
    "凭证号",
    "交易介质编号",
]


def _wrap_arbitrarily(value: str, seed: int) -> str:
    """Insert deterministic, non-semantic wraps at changing offsets."""
    if not value:
        return value
    chunks: list[str] = []
    cursor = 0
    while cursor < len(value):
        width = 1 + ((seed + cursor) % 4)
        chunks.append(value[cursor : cursor + width])
        cursor += width
    return "\n".join(chunks)


def _wide_source_row(*, income: bool, wrap_seed: int) -> list[str]:
    values = [
        "880000001234",
        "2024022912:34:56" if income else "2024030109:08:07",
        "0.00" if income else "12.30",
        "88.20" if income else "0.00",
        "188.20" if income else "175.90",
        "人民币元",
        "样例交易对手甲" if income else "样例交易对手乙",
        "990000004321" if income else "990000004322",
        "样例对方开户机构",
        "20240301",
        "电子汇入" if income else "电子转账",
        "完整业务备注甲" if income else "完整业务备注乙",
        "SERIAL-A-001" if income else "SERIAL-B-002",
        "ENTERPRISE-A" if income else "ENTERPRISE-B",
        "电子凭证",
        "VOUCHER-001" if income else "VOUCHER-002",
        "MEDIUM-01" if income else "MEDIUM-02",
    ]
    return [_wrap_arbitrarily(value, wrap_seed + index) for index, value in enumerate(values)]


@pytest.mark.parametrize("wrap_seed", [0, 3, 7])
def test_wrapped_wide_grid_preserves_every_raw_business_cell_and_canonicalizes_dates(
    wrap_seed: int,
) -> None:
    wrapped_headers = [
        _wrap_arbitrarily(header, wrap_seed + index) for index, header in enumerate(_WIDE_HEADERS)
    ]
    source_rows = [
        _wide_source_row(income=True, wrap_seed=wrap_seed + 11),
        _wide_source_row(income=False, wrap_seed=wrap_seed + 29),
    ]
    ctx = StyleContext(
        tables=[[wrapped_headers, *source_rows]],
        full_text="企业账户交易明细",
        institution=None,
        page_count=1,
    )
    detection = BankStyleDetector().detect(ctx)

    records, _identity = BankStyleParserRegistry().run(
        detection,
        ctx,
        BankStatementCommunityPlugin(),
    )

    assert len(records) == 2
    for record, source_row in zip(records, source_rows, strict=True):
        assert record["raw"] == dict(zip(wrapped_headers, source_row, strict=True))
    assert [record["normalized"]["date"] for record in records] == [
        "2024-02-29",
        "2024-03-01",
    ]
    assert [record["normalized"]["timestamp"] for record in records] == [
        "2024-02-29T12:34:56",
        "2024-03-01T09:08:07",
    ]
    assert [record["normalized"]["direction"] for record in records] == ["income", "expense"]
    assert [record["normalized"]["amount"] for record in records] == pytest.approx([88.2, 12.3])
    assert [record["normalized"]["balance"] for record in records] == pytest.approx([188.2, 175.9])


_TWO_PAGE_GRID_HEADERS = [
    "序号",
    "交易日期",
    "收/支",
    "交易金额",
    "余额",
    "对方账号",
    "对方户名",
    "摘要",
]
_TWO_PAGE_GRID_VALUES = [
    ["1", "20240801", "贷", "100.00", "1100.00", "622200001111", "甲公司", "收款"],
    ["2", "20240802", "借", "20.00", "1080.00", "622200002222", "乙公司", "付款"],
]


def _two_page_grid_row(values: list[str], *, page: int, row_index: int) -> TableRow:
    table_id = f"grid:p{page}"
    refs = [
        {
            "page": page,
            "table_id": table_id,
            "row": row_index,
            "raw_row": row_index + 1,
            "col": col_index,
        }
        for col_index in range(len(values))
    ]
    return TableRow(
        cells=[
            CellValue(
                text=value,
                evidence_ids=[f"ev:p{page}:r{row_index}:c{col_index}"],
                source_cell_refs=[refs[col_index]],
            )
            for col_index, value in enumerate(values)
        ],
        source_page=page,
        source_physical_id=table_id,
        source_row_index=row_index,
        source_cell_refs=refs,
    )


def test_grid_strategy_conserves_two_page_raw_canonical_normalized_and_source_lineage() -> None:
    source_rows = [
        _two_page_grid_row(_TWO_PAGE_GRID_VALUES[0], page=1, row_index=4),
        _two_page_grid_row(_TWO_PAGE_GRID_VALUES[1], page=2, row_index=2),
    ]
    parse_result = ParseResult(
        pages=[PageContent(page_number=1), PageContent(page_number=2)],
        entities=DocumentEntities(document_type="bank_statement"),
        logical_tables=[
            LogicalTable(
                table_id="logical:grid:two-page",
                headers=list(_TWO_PAGE_GRID_HEADERS),
                rows=source_rows,
                row_count=2,
                data_row_estimate=2,
                source_pages=[1, 2],
                source_physical_ids=["grid:p1", "grid:p2"],
                page_span=(1, 2),
                provenance=[
                    RowProvenance(
                        source_page=row.source_page,
                        source_table_id=row.source_physical_id,
                        source_row_index=row.source_row_index,
                    )
                    for row in source_rows
                ],
                quality_passed=True,
            )
        ],
    )
    policy = BankExtractionPolicy(
        route=BankExtractionRoute.DIGITAL,
        allowed_parser_ids=frozenset({"grid_standard"}),
    )
    ctx = StyleContext(
        tables=[[_TWO_PAGE_GRID_HEADERS, *_TWO_PAGE_GRID_VALUES]],
        full_text="企业账户交易明细 第1/2页 第2/2页",
        institution=None,
        page_count=2,
        parse_result=parse_result,
        reconstruction=ReconstructionMeta(source="canonical_table"),
        extraction_route=policy.route,
        extraction_policy=policy,
    )
    detection = StyleDetectionResult(
        primary_style="grid_standard",
        confidence=1.0,
        parser_chain=["grid_standard"],
    )

    records, _identity = BankStyleParserRegistry(adaptive=False).run(
        detection,
        ctx,
        BankStatementCommunityPlugin(),
    )

    assert [record["raw"] for record in records] == [
        dict(zip(_TWO_PAGE_GRID_HEADERS, values, strict=True))
        for values in _TWO_PAGE_GRID_VALUES
    ]
    assert [record["canonical_raw"] for record in records] == [
        {
            "sequence_no": "1",
            "date": "20240801",
            "direction": "贷",
            "amount": "100.00",
            "balance": "1100.00",
            "counter_account": "622200001111",
            "counter_party": "甲公司",
            "summary": "收款",
        },
        {
            "sequence_no": "2",
            "date": "20240802",
            "direction": "借",
            "amount": "20.00",
            "balance": "1080.00",
            "counter_account": "622200002222",
            "counter_party": "乙公司",
            "summary": "付款",
        },
    ]
    assert [
        {
            key: record["normalized"][key]
            for key in (
                "sequence_no",
                "date",
                "direction",
                "amount",
                "balance",
                "counter_account",
                "counter_party",
                "summary",
            )
        }
        for record in records
    ] == [
        {
            "sequence_no": "1",
            "date": "2024-08-01",
            "direction": "income",
            "amount": 100.0,
            "balance": 1100.0,
            "counter_account": "622200001111",
            "counter_party": "甲公司",
            "summary": "收款",
        },
        {
            "sequence_no": "2",
            "date": "2024-08-02",
            "direction": "expense",
            "amount": 20.0,
            "balance": 1080.0,
            "counter_account": "622200002222",
            "counter_party": "乙公司",
            "summary": "付款",
        },
    ]
    assert [record["source"] for record in records] == [
        {
            "source": "canonical_table",
            "source_page": 1,
            "page_id": "page:0001",
            "table_id": "grid:p1",
            "source_row_index": 4,
            "page_range": [1, 1],
            "evidence_ids": [f"ev:p1:r4:c{col_index}" for col_index in range(8)],
            "source_cell_refs": [
                {
                    "page": 1,
                    "table_id": "grid:p1",
                    "row": 4,
                    "raw_row": 5,
                    "col": col_index,
                }
                for col_index in range(8)
            ],
        },
        {
            "source": "canonical_table",
            "source_page": 2,
            "page_id": "page:0002",
            "table_id": "grid:p2",
            "source_row_index": 2,
            "page_range": [2, 2],
            "evidence_ids": [f"ev:p2:r2:c{col_index}" for col_index in range(8)],
            "source_cell_refs": [
                {
                    "page": 2,
                    "table_id": "grid:p2",
                    "row": 2,
                    "raw_row": 3,
                    "col": col_index,
                }
                for col_index in range(8)
            ],
        },
    ]


def _logical_parse_result(*, rows: list[TableRow]) -> ParseResult:
    return ParseResult(
        pages=[PageContent(page_number=1), PageContent(page_number=2)],
        entities=DocumentEntities(document_type="bank_statement"),
        logical_tables=[
            LogicalTable(
                table_id="logical:synthetic",
                headers=[
                    "账户明细 账号:SYNTHETIC 户名:样例企业 币种:人民币",
                    "",
                    "",
                    "",
                    "",
                    "",
                ],
                rows=rows,
                row_count=len(rows),
                source_pages=[1, 2],
                source_physical_ids=["physical:1", "physical:2"],
                page_span=(1, 2),
                quality_passed=True,
            )
        ],
    )


def _logical_row(values: list[str], *, page: int, row_index: int) -> TableRow:
    return TableRow(
        cells=[CellValue(text=value) for value in values],
        source_page=page,
        source_physical_id=f"physical:{page}",
        source_row_index=row_index,
    )


def test_recurrent_real_header_is_promoted_ahead_of_logical_identity_pseudo_header() -> None:
    real_header = ["交易时间", "收入金额", "支出金额", "账户余额", "交易用途", "会计日期"]
    parse_result = _logical_parse_result(
        rows=[
            _logical_row(real_header, page=1, row_index=0),
            _logical_row(
                ["2024050620:20:17", "", "2.25", "368.79", "服务费", "20240506"],
                page=1,
                row_index=1,
            ),
            _logical_row(real_header, page=2, row_index=0),
            _logical_row(
                ["2024050710:11:12", "1500.00", "", "1868.79", "收款", "20240507"],
                page=2,
                row_index=1,
            ),
        ]
    )
    plugin = BankStatementCommunityPlugin()

    transactions = extract_logical_rows_with_provenance(
        parse_result,
        plugin.column_registry,
        strict_first_col=True,
    )

    assert len(transactions) == 2
    assert [transaction["_source"]["source_page"] for transaction in transactions] == [1, 2]
    normalized = [normalize_record(transaction, plugin) for transaction in transactions]
    assert [row["date"] for row in normalized] == ["2024-05-06", "2024-05-07"]
    assert [row["timestamp"] for row in normalized] == [
        "2024-05-06T20:20:17",
        "2024-05-07T10:11:12",
    ]


def test_identity_pseudo_header_does_not_promote_a_single_data_like_row() -> None:
    parse_result = _logical_parse_result(
        rows=[
            _logical_row(
                ["2024050620:20:17", "", "2.25", "368.79", "服务费", "20240506"],
                page=1,
                row_index=0,
            )
        ]
    )

    transactions = extract_logical_rows_with_provenance(
        parse_result,
        BankStatementCommunityPlugin().column_registry,
        strict_first_col=True,
    )

    assert transactions == []


def test_dedicated_bilingual_direction_outranks_earlier_transaction_type() -> None:
    raw = {
        "序号": "37",
        "交易日期": "2024-06-30",
        # Deliberately inserted first: the business name contains 贷 but is not
        # itself a bounded direction label.
        "交易类型": "贷款到期归还",
        "借贷": "借 Dr",
        "交易金额": "4,966.76",
        "余额": "281,454.81",
    }
    plugin = BankStatementCommunityPlugin()

    normalized = normalize_record(raw, plugin)
    assert normalized["direction"] == "expense"

    # Even a contradictory adjacent balance may not rewrite an explicit B3
    # direction fact.
    records = [
        {"raw": {"序号": "36"}, "normalized": {"amount": 1.0, "balance": 100.0}},
        {"raw": raw, "normalized": {**normalized, "direction": "income", "balance": 105.0}},
    ]
    refine_missing_directions_from_balance_chain(records)
    assert records[1]["normalized"]["direction"] == "expense"


@pytest.mark.parametrize(
    ("counterparty_column", "source_value", "expected_account", "expected_party"),
    [
        ("对方账号", "320975300000315768300样例姓名", "320975300000315768300", "样例姓名"),
        ("对方账号与户名", "884400001234/样例公司", "884400001234", "样例公司"),
        ("对方账号与户名", "A84x9Z00231Q/另一公司", "A84x9Z00231Q", "另一公司"),
    ],
)
def test_same_row_compound_counterparty_blocks_flattened_page_enrichment(
    counterparty_column: str,
    source_value: str,
    expected_account: str,
    expected_party: str,
) -> None:
    source_row = {
        "序号": "41",
        "交易日期": "2024-07-01",
        "交易时间": "09:10:11",
        "借贷": "贷 Cr",
        "交易金额": "10.00",
        "余额": "110.00",
        counterparty_column: source_value,
        "_source": {"source_page": 1, "page_range": [1, 1]},
    }
    if counterparty_column == "对方账号":
        source_row["对方户名"] = ""
    exact_source_business_cells = {
        key: value for key, value in source_row.items() if not key.startswith("_")
    }
    following_row = {
        "序号": "42",
        "交易日期": "2024-07-02",
        "交易时间": "12:13:14",
        "借贷": "借 Dr",
        "交易金额": "20.00",
        "余额": "90.00",
        "对方账号": "991100001111",
        "对方户名": "相邻交易对手",
        "_source": {"source_page": 1, "page_range": [1, 1]},
    }
    flattened_page_text = (
        "41 2024-07-01 09:10:11 10.00 110.00 "
        f"{source_value} "
        "42 2024-07-02 12:13:14 20.00 90.00 991100001111 相邻交易对手"
    )

    finalized = _finalize_transactions(
        [source_row, following_row],
        full_text=flattened_page_text,
    )

    assert {
        key: value for key, value in finalized[0].items() if not key.startswith("_")
    } == exact_source_business_cells
    if counterparty_column == "对方账号与户名":
        assert "对方户名" not in finalized[0]
    normalized = normalize_record(finalized[0], BankStatementCommunityPlugin())
    assert normalized["counter_account"] == expected_account
    assert normalized["counter_party"] == expected_party


def test_page_recovery_never_skips_an_adjacent_unbounded_row() -> None:
    first = {
        "序号": "51",
        "交易日期": "2024-08-01",
        "交易时间": "01:02:03",
        "借贷": "借 Dr",
        "交易金额": "5.00",
        "余额": "95.00",
        "对方账号": "880000005151",
        "对方户名": "",
        "_source": {"source_page": 1, "page_range": [1, 1]},
    }
    adjacent = {
        "序号": "52",
        "交易日期": "2024-08-02",
        "交易时间": "04:05:06",
        "借贷": "贷 Cr",
        "交易金额": "15.00",
        "余额": "110.00",
        "对方账号": "880000005252",
        "对方户名": "相邻姓名",
        "_source": {"source_page": 1, "page_range": [1, 1]},
    }
    source_text = (
        "51 2024-08-01 01:02:03 5.00 95.00 880000005151 "
        "52 2024-08-02 04:05:06 15.00 110.00 880000005252 相邻姓名"
    )

    finalized = _finalize_transactions([first, adjacent], full_text=source_text)

    assert finalized[0]["对方户名"] == ""
    assert normalize_record(finalized[0], BankStatementCommunityPlugin())["counter_party"] == ""
