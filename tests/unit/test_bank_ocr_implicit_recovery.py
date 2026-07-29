# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OCR implicit table recovery tests for scanned bank ledgers."""

from __future__ import annotations

import pytest

from docmirror.models.entities.parse_result import (
    CellValue,
    DocumentEntities,
    PageContent,
    ParseResult,
    TableBlock,
    TableRow,
    TextBlock,
)
from docmirror.plugins.bank_statement.community_plugin import BankStatementCommunityPlugin
from docmirror.plugins.bank_statement.context import StyleContext
from docmirror.plugins.bank_statement.ocr_implicit_table_recovery import (
    recover_ocr_implicit_ledger_tables,
    recovered_ocr_implicit_row_count,
)
from docmirror.plugins.bank_statement.styles import grid_standard


def _table_result(rows: list[list[str]]) -> ParseResult:
    table = TableBlock(
        headers=rows[0],
        rows=[TableRow(cells=[CellValue(text=text) for text in row]) for row in rows[1:]],
    )
    return ParseResult(pages=[PageContent(page_number=1, tables=[table])])


def test_recover_ocr_implicit_table_keeps_valid_rows_with_page_noise() -> None:
    parse_result = _table_result(
        [
            ["交易日期", "月收/支", "交易金额", "账户余额", "摘要", "对方账号", "对方户名凭证序号", "机构", "柜员", "备注信息"],
            ["20221008", "支出", "4.00", "1256.57", "POS消费", "第3页", "Q", "OB8B9E", "ORD95E", "D02"],
            ["20221009", "支山", "4.00", "1252.57", "POS消费", "", "", "", "", ""],
        ]
    )
    tables = recover_ocr_implicit_ledger_tables(parse_result, "")
    assert len(tables) == 1
    assert len(tables[0]) == 3

    ctx = StyleContext(tables=tables, full_text="", institution=None, page_count=1, parse_result=parse_result)
    plugin = BankStatementCommunityPlugin()
    raw = grid_standard.extract_transactions(ctx, plugin)
    records = [grid_standard.normalize_record(row, plugin) for row in raw]

    assert len(records) == 2
    assert records[0]["date"] == "2022-10-08"
    assert records[0]["direction"] == "expense"
    assert records[0]["amount"] == pytest.approx(4.0)
    assert records[1]["direction"] == "expense"


def test_recover_paragraph_ledger_rows_and_repairs_amount_balance_orientation() -> None:
    lines = [
        ("交易日期 收/支 交易金额 摘要 账户余额 对方账号 柜员 备注信息 对方户名凭证序号 机构", [30, 100, 560, 120]),
        ("20220419 支出 2.00 944.75 网络付款 1500947831 0098 NY0035 2号生活馆", [30, 130, 520, 145]),
        ("20220419 支出 800.00 144.75 网络付款 1000050001 0098 NY0024 微信转账", [30, 146, 520, 160]),
        ("扫二维码 支出 20220420 网络付款 125.75 19. 00 1000107101 0098 NY0016 付款", [30, 161, 520, 176]),
        ("收入 20220425 800.00 925.75 网络收款 243300133 0098 NY0062", [30, 177, 520, 192]),
    ]
    parse_result = ParseResult(
        pages=[PageContent(page_number=1, texts=[TextBlock(content=text, bbox=bbox) for text, bbox in lines])]
    )

    tables = recover_ocr_implicit_ledger_tables(parse_result, "")
    assert len(tables) == 1
    assert len(tables[0]) == 5

    ctx = StyleContext(tables=tables, full_text="", institution=None, page_count=1, parse_result=parse_result)
    plugin = BankStatementCommunityPlugin()
    raw = grid_standard.extract_transactions(ctx, plugin)
    records = [grid_standard.normalize_record(row, plugin) for row in raw]

    assert [(r["amount"], r["balance"]) for r in records] == [
        (2.0, 944.75),
        (800.0, 144.75),
        (19.0, 125.75),
        (800.0, 925.75),
    ]


def test_descending_signed_paragraph_rows_keep_amount_balance_and_summary_columns() -> None:
    lines = [
        ("序号交易时间交易金额\n余额\n摘要\n交易对手信息", [30, 100, 560, 120]),
        ("1\n20220426\n-1,911.00\n0.47\n转账\n招商银行股份有限公司-彭磊-\n6214831213283929", [30, 130, 520, 170]),
        ("2\n20220426\n-1,500.00\n1,911.47\n转账\n招商银行股份有限公司-彭磊-\n6214831213283929", [30, 171, 520, 211]),
        ("3\n20220408\n-42.80\n3,411.47\n第三方支付\n杭州兑吧网络科技有限公司-\n4061601", [30, 212, 520, 252]),
    ]
    parse_result = ParseResult(
        pages=[PageContent(page_number=1, texts=[TextBlock(content=text, bbox=bbox) for text, bbox in lines])]
    )

    tables = recover_ocr_implicit_ledger_tables(parse_result, "")

    assert len(tables) == 1
    assert [row[:7] for row in tables[0][1:]] == [
        ["2022-04-26", "支出", "1911.00", "0.47", "转账", "6214831213283929", "招商银行股份有限公司-彭磊-"],
        ["2022-04-26", "支出", "1500.00", "1911.47", "转账", "6214831213283929", "招商银行股份有限公司-彭磊-"],
        ["2022-04-08", "支出", "42.80", "3411.47", "第三方支付", "4061601", "杭州兑吧网络科技有限公司-"],
    ]


def test_multi_row_sequence_date_block_does_not_merge_neighbor_transactions() -> None:
    lines = [
        ("序号交易时间交易金额\n余额\n摘要\n交易对手信息", [30, 100, 560, 120]),
        (
            "7\n20220402\n-420,000.00\n5,658.41\n转账\n中国银行总行-彭磊-\n6216690800004884536\n"
            "8\n20220401\n+94.89\n425,658.41\n结息\n"
            "9\n20220328\n-187.18\n425,563.52\n还信用卡",
            [30, 130, 520, 210],
        ),
    ]
    parse_result = ParseResult(
        pages=[PageContent(page_number=1, texts=[TextBlock(content=text, bbox=bbox) for text, bbox in lines])]
    )

    tables = recover_ocr_implicit_ledger_tables(parse_result, "")

    assert len(tables) == 1
    assert [row[:7] for row in tables[0][1:]] == [
        ["2022-04-02", "支出", "420000.00", "5658.41", "转账", "6216690800004884536", "中国银行总行-彭磊-"],
        ["2022-04-01", "收入", "94.89", "425658.41", "结息", "", ""],
        ["2022-03-28", "支出", "187.18", "425563.52", "还信用卡", "", ""],
    ]


def test_signed_amount_orientation_is_not_swapped_to_fit_noisy_neighbor_order() -> None:
    lines = [
        ("序号交易时间交易金额\n余额\n摘要\n交易对手信息", [30, 100, 560, 120]),
        ("招商银行股份有限公司-\n6214832145808973-\n6214832145808973", [390, 121, 520, 145]),
        ("64\n20210508\n+40,000.00\n59,777.29\n转账", [30, 130, 520, 150]),
        ("招商银行股份有限公司-\n6214832145808973-\n6214832145808973\n66\n20210501\n+60.10\n59,777.29\n结息", [30, 151, 520, 190]),
        ("65\n20210508\n-40,000.00\n19,777.29\n转账", [30, 191, 520, 211]),
        ("67\n20210427\n-4,950.00\n59,717.19\n转账\n长沙银行股份有限公司-刘梦云\n-6214467873120749558", [30, 212, 520, 252]),
    ]
    parse_result = ParseResult(
        pages=[PageContent(page_number=1, texts=[TextBlock(content=text, bbox=bbox) for text, bbox in lines])]
    )

    tables = recover_ocr_implicit_ledger_tables(parse_result, "")

    row_20210508_income = next(row for row in tables[0][1:] if row[0] == "2021-05-08" and row[1] == "收入")
    row_20210508_expense = next(
        row for row in tables[0][1:] if row[0] == "2021-05-08" and row[1] == "支出" and row[2] == "40000.00"
    )
    row_20210501 = next(row for row in tables[0][1:] if row[0] == "2021-05-01")
    assert row_20210508_income[5:7] == ["6214832145808973", "招商银行股份有限公司-"]
    assert row_20210508_expense[5:7] == ["6214832145808973", "招商银行股份有限公司-"]
    assert row_20210501[:5] == ["2021-05-01", "收入", "60.10", "59777.29", "结息"]


def test_recover_distributed_signed_amount_ledger_blocks() -> None:
    lines = [
        ("交易日期", [30, 100, 90, 112]),
        ("借/贷方发生额", [100, 100, 180, 112]),
        ("余额", [190, 100, 240, 112]),
        ("摘要", [250, 100, 300, 112]),
        ("对方账户", [310, 100, 390, 112]),
        ("20230309", [30, 130, 90, 142]),
        ("-10.00", [100, 130, 150, 142]),
        ("990.00", [190, 130, 240, 142]),
        ("网络付款", [250, 130, 300, 142]),
        ("6222000000000001", [310, 130, 410, 142]),
        ("20230310", [30, 150, 90, 162]),
        ("-20.00", [100, 150, 150, 162]),
        ("970.00", [190, 150, 240, 162]),
        ("微信转账", [250, 150, 300, 162]),
        ("6222000000000002", [310, 150, 410, 162]),
        ("20230311", [30, 170, 90, 182]),
        ("+30.00", [100, 170, 150, 182]),
        ("1000.00", [190, 170, 240, 182]),
        ("网络收款", [250, 170, 300, 182]),
        ("6222000000000003", [310, 170, 410, 182]),
    ]
    parse_result = ParseResult(
        pages=[PageContent(page_number=1, texts=[TextBlock(content=text, bbox=bbox) for text, bbox in lines])]
    )

    tables = recover_ocr_implicit_ledger_tables(parse_result, "")

    assert len(tables) == 1
    assert [row[:4] for row in tables[0][1:]] == [
        ["2023-03-09", "支出", "10.00", "990.00"],
        ["2023-03-10", "支出", "20.00", "970.00"],
        ["2023-03-11", "收入", "30.00", "1000.00"],
    ]


def test_recover_distributed_explicit_direction_ledger_blocks() -> None:
    lines = [
        ("序号", [10, 90, 40, 102]),
        ("交易日期", [50, 90, 110, 102]),
        ("收入/支出", [120, 90, 180, 102]),
        ("交易金额", [190, 90, 250, 102]),
        ("账户余额", [260, 90, 320, 102]),
        ("对方账号", [330, 90, 410, 102]),
        ("摘要", [420, 90, 470, 102]),
        ("20221231", [50, 120, 110, 132]),
        ("支出", [120, 120, 150, 132]),
        ("100.00", [190, 120, 240, 132]),
        ("900.00", [260, 120, 310, 132]),
        ("6222000000000001", [330, 120, 430, 132]),
        ("转账", [420, 120, 460, 132]),
        ("20230101", [50, 140, 110, 152]),
        ("收入", [120, 140, 150, 152]),
        ("50.00", [190, 140, 240, 152]),
        ("950.00", [260, 140, 310, 152]),
        ("6222000000000002", [330, 140, 430, 152]),
        ("转账", [420, 140, 460, 152]),
        ("20230102", [50, 160, 110, 172]),
        ("支出", [120, 160, 150, 172]),
        ("10.00", [190, 160, 240, 172]),
        ("940.00", [260, 160, 310, 172]),
        ("6222000000000003", [330, 160, 430, 172]),
        ("转账", [420, 160, 460, 172]),
    ]
    parse_result = ParseResult(
        pages=[PageContent(page_number=1, texts=[TextBlock(content=text, bbox=bbox) for text, bbox in lines])]
    )

    tables = recover_ocr_implicit_ledger_tables(parse_result, "")

    assert len(tables) == 1
    assert len(tables[0]) == 4


def test_distributed_ledger_requires_at_least_three_valid_rows() -> None:
    lines = [
        ("交易日期", [30, 100, 90, 112]),
        ("交易金额", [100, 100, 160, 112]),
        ("余额", [170, 100, 220, 112]),
        ("收/支", [230, 100, 270, 112]),
        ("摘要", [280, 100, 320, 112]),
        ("20230309", [30, 130, 90, 142]),
        ("支出 10.00 990.00", [100, 130, 240, 142]),
        ("20230310", [30, 150, 90, 162]),
        ("支出 20.00 970.00", [100, 150, 240, 162]),
    ]
    parse_result = ParseResult(
        pages=[PageContent(page_number=1, texts=[TextBlock(content=text, bbox=bbox) for text, bbox in lines])]
    )

    assert recover_ocr_implicit_ledger_tables(parse_result, "") == []


def test_complete_paragraph_header_keeps_two_existing_rows() -> None:
    lines = [
        ("交易日期 收/支 交易金额 账户余额 摘要 对方账号", [30, 100, 400, 112]),
        ("20230309 支出 10.00 990.00 网络付款 6222000000000001", [30, 130, 420, 142]),
        ("20230310 支出 20.00 970.00 微信转账 6222000000000002", [30, 150, 420, 162]),
    ]
    parse_result = ParseResult(
        pages=[PageContent(page_number=1, texts=[TextBlock(content=text, bbox=bbox) for text, bbox in lines])]
    )

    tables = recover_ocr_implicit_ledger_tables(parse_result, "")

    assert len(tables) == 1
    assert len(tables[0]) == 3


def test_recover_borderless_page_emitted_as_one_ocr_text_block() -> None:
    full_text = """
上页余额: 1000.00
交易日期
借方/贷方金额
余额
交易流水号
摘要
100.00借 900.00 转账 W0001-202401010001 20240101
50.00贷 950.00 网银来账 W0001-202401020002 20240102
20.00借 930.00 手续费 W0001-202401030003 20240103
"""
    parse_result = ParseResult(
        pages=[PageContent(page_number=1, texts=[TextBlock(content=full_text)])]
    )

    tables = recover_ocr_implicit_ledger_tables(parse_result, full_text)

    assert len(tables) == 1
    assert [row[:4] for row in tables[0][1:]] == [
        ["2024-01-01", "支出", "100.00", "900.00"],
        ["2024-01-02", "收入", "50.00", "950.00"],
        ["2024-01-03", "支出", "20.00", "930.00"],
    ]


def test_borderless_debit_marker_distinguishes_amount_from_leading_balance() -> None:
    full_text = """
上页余额: 500010.00
交易日期
借方/贷方金额
余额
交易流水号
摘要
450010.00贷 50000.00借 转账 W0001-202401010001 20240101
405010.00贷 45000.00借 转账 W0001-202401020002 20240102
357010.00贷 48000.00借 转账 W0001-202401030003 20240103
"""
    parse_result = ParseResult(
        pages=[PageContent(page_number=1, texts=[TextBlock(content=full_text)])]
    )

    tables = recover_ocr_implicit_ledger_tables(parse_result, full_text)

    assert [row[:4] for row in tables[0][1:]] == [
        ["2024-01-01", "支出", "50000.00", "450010.00"],
        ["2024-01-02", "支出", "45000.00", "405010.00"],
        ["2024-01-03", "支出", "48000.00", "357010.00"],
    ]


def test_recover_ocr_implicit_table_caches_tables_on_parse_result() -> None:
    parse_result = _table_result(
        [
            ["交易日期", "收/支", "交易金额", "账户余额"],
            ["20240101", "收入", "10.00", "10.00"],
            ["20240102", "支出", "3.00", "7.00"],
        ]
    )
    parse_result.entities = DocumentEntities(domain_specific={})

    first = recover_ocr_implicit_ledger_tables(parse_result, "")
    assert recovered_ocr_implicit_row_count(parse_result) == 2

    first[0].append(["mutated"])
    parse_result.pages = []
    second = recover_ocr_implicit_ledger_tables(parse_result, "")

    assert len(second) == 1
    assert len(second[0]) == 3
    assert recovered_ocr_implicit_row_count(parse_result) == 2
