# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OCR implicit table recovery tests for scanned bank ledgers."""

from __future__ import annotations

import pytest

from docmirror.models.entities.parse_result import (
    CanonicalEvidencePlane,
    CellValue,
    DocumentEntities,
    ExtractionMethod,
    PageContent,
    ParseResult,
    ParserInfo,
    TableBlock,
    TableRow,
    TextBlock,
)
from docmirror.models.mirror.vnext import EvidenceAtom, EvidenceStore
from docmirror.plugins.bank_statement.community_plugin import BankStatementCommunityPlugin
from docmirror.plugins.bank_statement.context import StyleContext
from docmirror.plugins.bank_statement.ocr_implicit_table_recovery import (
    _repair_balance_chain_rows,
    recover_ocr_implicit_ledger_tables,
    recovered_ocr_implicit_row_count,
    recovered_ocr_implicit_row_evidence,
)
from docmirror.plugins.bank_statement.styles import grid_standard


def _table_result(rows: list[list[str]]) -> ParseResult:
    table = TableBlock(
        headers=rows[0],
        rows=[TableRow(cells=[CellValue(text=text) for text in row]) for row in rows[1:]],
    )
    return ParseResult(pages=[PageContent(page_number=1, tables=[table])])


def test_single_recovered_row_does_not_expose_internal_repair_metadata() -> None:
    row = [
        "2024-01-01",
        "收入",
        "100.00",
        "100.00",
        "转账",
        "6222000000000000",
        "测试用户",
        "",
        "",
        "__docmirror_repair_meta__:signed=1",
        "1",
    ]

    repaired = _repair_balance_chain_rows([row])

    assert repaired[0][9] == ""


def test_internal_ocr_sequence_becomes_canonical_sequence() -> None:
    normalized = grid_standard.normalize_record(
        {
            "交易日期": "2023-03-09",
            "收/支": "支出",
            "交易金额": "4819.00",
            "账户余额": "401143.31",
            "_source_sequence_no": "1",
        },
        BankStatementCommunityPlugin(),
    )

    assert normalized["sequence_no"] == "1"


def test_recover_ocr_implicit_table_keeps_valid_rows_with_page_noise() -> None:
    parse_result = _table_result(
        [
            [
                "交易日期",
                "月收/支",
                "交易金额",
                "账户余额",
                "摘要",
                "对方账号",
                "对方户名凭证序号",
                "机构",
                "柜员",
                "备注信息",
            ],
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
        (
            "招商银行股份有限公司-\n6214832145808973-\n6214832145808973\n66\n20210501\n+60.10\n59,777.29\n结息",
            [30, 151, 520, 190],
        ),
        ("65\n20210508\n-40,000.00\n19,777.29\n转账", [30, 191, 520, 211]),
        (
            "67\n20210427\n-4,950.00\n59,717.19\n转账\n长沙银行股份有限公司-刘梦云\n-6214467873120749558",
            [30, 212, 520, 252],
        ),
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


def test_recover_all_pages_with_stacked_ocr_header_and_debit_credit_markers() -> None:
    pages: list[PageContent] = []
    for page_no in range(1, 6):
        month = f"{page_no:02d}"
        header_prefix = "交易口 借贷标" if page_no == 1 else "交易日 借贷标"
        lines = [
            ("某银行公司账户交易明细清单", [200, 10, 400, 24]),
            ("起止日期:2024/01/01-2024/12/31", [20, 40, 250, 54]),
            (header_prefix, [20, 62, 85, 75]),
            ("期 志 交易金额 余额 对方户名 对方账号 摘要", [20, 73, 500, 88]),
            (
                f"2024{month}01 借 100.00 900.00 待报解预算收入 6222000000000001 划缴税款",
                [20, 96, 520, 112],
            ),
            (
                f"2024{month}02 贷 50.00 950.00 付款单位 6222000000000002 汇款汇入",
                [20, 112, 520, 128],
            ),
            (
                f"2024{month}03 借 -10.00 960.00 付款单位 6222000000000003 冲正",
                [20, 128, 520, 144],
            ),
        ]
        pages.append(
            PageContent(
                page_number=page_no,
                texts=[TextBlock(content=text, bbox=bbox) for text, bbox in lines],
            )
        )
    parse_result = ParseResult(pages=pages)

    tables = recover_ocr_implicit_ledger_tables(parse_result, "")

    assert len(tables) == 5
    assert sum(len(table) - 1 for table in tables) == 15
    assert [[row[1] for row in table[1:]] for table in tables] == [["支出", "收入", "收入"]] * 5
    assert [[row[10] for row in table[1:]] for table in tables] == [[str(page_no)] * 3 for page_no in range(1, 6)]
    assert tables[0][1][4:7] == ["划缴税款", "6222000000000001", "待报解预算收入"]


def test_scanned_ledger_prefers_positioned_ocr_lines_over_scattered_tokens() -> None:
    bundles = []
    for page_no in (1, 2):
        line_texts = [
            "某银行公司账户交易明细清单",
            "交易口 借贷标",
            "期 志 交易金额 余额 对方户名 对方账号 摘要",
            f"20240{page_no}01 借 100.00 900.00 付款单位 6222000000000001 转账",
            f"20240{page_no}02 贷 50.00 950.00 付款单位 6222000000000002 汇入",
            f"20240{page_no}03 借 20.00 930.00 付款单位 6222000000000003 手续费",
        ]
        bundles.append(
            {
                "local_structure_evidence": {
                    "page": page_no,
                    "lines": [
                        {
                            "page": page_no,
                            "text": text,
                            "bbox": [20, 20 + index * 16, 520, 32 + index * 16],
                        }
                        for index, text in enumerate(line_texts)
                    ],
                }
            }
        )
    parse_result = ParseResult(
        pages=[
            PageContent(
                page_number=page_no,
                texts=[
                    TextBlock(content="交易口", bbox=[20, 60, 50, 72]),
                    TextBlock(content="20240101", bbox=[20, 100, 80, 112]),
                    TextBlock(content="100.00", bbox=[100, 100, 150, 112]),
                ],
            )
            for page_no in (1, 2)
        ],
        entities=DocumentEntities(domain_specific={"_page_evidence_bundles": bundles}),
    )

    tables = recover_ocr_implicit_ledger_tables(parse_result, "")

    assert len(tables) == 2
    assert [len(table) - 1 for table in tables] == [3, 3]
    assert [[row[10] for row in table[1:]] for table in tables] == [["1"] * 3, ["2"] * 3]


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


def test_complete_paragraph_header_accepts_plain_date_and_signed_amount_column() -> None:
    lines = [
        (
            "序号 日期 借/贷方发生额 余额 对方户名 对方账户 传票号 摘要",
            [30, 100, 760, 118],
        ),
        (
            "1 20230309 -4,819.00 401,143.31 杨光 6226192013864418 备用金",
            [30, 130, 760, 145],
        ),
        (
            "2 20230309 -13,300.00 387,843.31 谢林华 6226192011784154 备用金",
            [30, 146, 760, 161],
        ),
        (
            "3 20230310 +18,400.00 406,243.31 梁远述 6228480460934190410 往来款",
            [30, 162, 760, 177],
        ),
    ]
    parse_result = ParseResult(
        pages=[PageContent(page_number=1, texts=[TextBlock(content=text, bbox=bbox) for text, bbox in lines])]
    )

    tables = recover_ocr_implicit_ledger_tables(parse_result, "")

    assert len(tables) == 1
    assert [row[:7] for row in tables[0][1:]] == [
        ["2023-03-09", "支出", "4819.00", "401143.31", "备用金", "6226192013864418", "杨光"],
        ["2023-03-09", "支出", "13300.00", "387843.31", "备用金", "6226192011784154", "谢林华"],
        ["2023-03-10", "收入", "18400.00", "406243.31", "往来款", "6228480460934190410", "梁远述"],
    ]
    assert tables[0][0][-1] == "_source_sequence_no"
    assert [row[-1] for row in tables[0][1:]] == ["1", "2", "3"]


def test_recover_native_corporate_detail_with_transaction_occurrence_amount_header() -> None:
    lines = [
        ("交易日期\n交易发生金额\n账户余额\n对方账号\n对方户名\n摘要\n备注", [30, 100, 520, 118]),
        (
            "20250102\n-2000000.00\n1458306.91\n650987227500015\n"
            "重庆正大软件（集团）\n有限公司\n网银转账\n用途:往来结算款;",
            [30, 130, 520, 150],
        ),
        (
            "20250102\n-2.00\n1458304.91\n5106010150380000027\n企网汇划手续费收入\n手续费\n用途:往来结算款;",
            [30, 151, 520, 171],
        ),
        (
            "20250102\n+2900000.00\n4358304.91\n650987227500015\n"
            "重庆正大软件（集团）\n有限公司\n跨行转账\n附言:往来款;",
            [30, 172, 520, 192],
        ),
        (
            "20250321\n+383.76\n32111.68\n5106010131210000841\n"
            "应付单位活期存款利息活期结息\n对公活期自动结息;利息:383.76;结息积数:138154517.00;"
            "利率:0.1000000;所属时间:20241221至20250321;",
            [30, 193, 520, 213],
        ),
        (
            "20250414\n-20000.00\n732076.68\n"
            "641301106013000859983重庆正大华日软件有限公司银川分公司\n"
            "实时汇款\n用途:往来结算款;",
            [30, 214, 520, 234],
        ),
    ]
    parse_result = ParseResult(
        pages=[PageContent(page_number=3, texts=[TextBlock(content=text, bbox=bbox) for text, bbox in lines])]
    )

    tables = recover_ocr_implicit_ledger_tables(parse_result, "")

    assert len(tables) == 1
    assert [row[:4] for row in tables[0][1:]] == [
        ["2025-01-02", "支出", "2000000.00", "1458306.91"],
        ["2025-01-02", "支出", "2.00", "1458304.91"],
        ["2025-01-02", "收入", "2900000.00", "4358304.91"],
        ["2025-03-21", "收入", "383.76", "32111.68"],
        ["2025-04-14", "支出", "20000.00", "732076.68"],
    ]
    assert [row[5] for row in tables[0][1:]] == [
        "650987227500015",
        "5106010150380000027",
        "650987227500015",
        "5106010131210000841",
        "641301106013000859983",
    ]
    assert [row[6] for row in tables[0][1:]] == [
        "重庆正大软件（集团）有限公司",
        "企网汇划手续费收入",
        "重庆正大软件（集团）有限公司",
        "应付单位活期存款利息",
        "重庆正大华日软件有限公司银川分公司",
    ]
    assert all(row[10] == "3" for row in tables[0][1:])

    ctx = StyleContext(
        tables=tables,
        full_text="",
        institution=None,
        page_count=1,
        parse_result=parse_result,
        prefer_context_tables=True,
    )
    raw_records = grid_standard.extract_transactions(ctx, BankStatementCommunityPlugin())
    assert len(raw_records) == 5
    assert raw_records[1]["对方户名"] == "企网汇划手续费收入"


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
    parse_result = ParseResult(pages=[PageContent(page_number=1, texts=[TextBlock(content=full_text)])])

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
    parse_result = ParseResult(pages=[PageContent(page_number=1, texts=[TextBlock(content=full_text)])])

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


def _pab_ordinal_page(
    page_number: int,
    total_pages: int,
    month: str,
    ordinals: list[int],
    *,
    page_mode: str = "scanned_ocr",
    account: str = "11005350836201",
) -> PageContent:
    columns = [
        ("序号", 30.0, 58.0),
        ("日期", 80.0, 104.0),
        ("借/贷方发生额", 140.0, 212.0),
        ("余额", 246.0, 272.0),
        ("对方户名", 353.0, 399.0),
        ("对方账户", 538.0, 585.0),
        ("传票号", 666.0, 703.0),
        ("摘要", 721.0, 747.0),
    ]
    header = [
        ("中国平安", [35.0, 30.0, 79.0, 44.0]),
        ("PINGANBANK", [103.0, 46.0, 183.0, 58.0]),
        ("平安银行", [101.0, 25.0, 185.0, 45.0]),
        ("客户存款月结单", [225.0, 47.0, 303.0, 61.0]),
        ("结单号:23090821289990000809", [337.0, 46.0, 492.0, 61.0]),
        (month, [565.0, 47.0, 620.0, 60.0]),
        (f"第{page_number}页共{total_pages}页", [650.0, 49.0, 716.0, 62.0]),
        (f"账号:{account}", [33.0, 81.0, 139.0, 93.0]),
    ]
    blocks = [TextBlock(content=text, bbox=bbox) for text, bbox in header]
    blocks.extend(TextBlock(content=text, bbox=[x0, 103.0, x1, 119.0]) for text, x0, x1 in columns)
    year_month = month.replace("年", "").replace("月", "")
    for offset, ordinal in enumerate(ordinals):
        y = 127.0 + offset * 15.0
        blocks.extend(
            [
                TextBlock(content=str(ordinal), bbox=[31.0, y, 40.0, y + 11.0]),
                TextBlock(content=f"{year_month}{offset + 1:02d}", bbox=[79.0, y, 123.0, y + 11.0]),
                TextBlock(content=f"-{ordinal:,}.00", bbox=[140.0, y, 190.0, y + 11.0]),
                TextBlock(content=f"{1000 - ordinal:,}.00", bbox=[246.0, y, 300.0, y + 11.0]),
            ]
        )
    footer_y = 127.0 + len(ordinals) * 15.0 + 8.0
    blocks.extend(
        TextBlock(content=text, bbox=[x0, footer_y + offset * 13.0, x1, footer_y + offset * 13.0 + 11.0])
        for offset, (text, x0, x1) in enumerate(
            [
                ("平安银行", 100.0, 185.0),
                ("电子回单专用章", 330.0, 420.0),
                ("已打印次数:2", 35.0, 120.0),
                ("打印时间:2023-09-08", 160.0, 290.0),
                ("打印方式:系统PDF生成", 320.0, 470.0),
                ("设备编号:0000", 500.0, 590.0),
                ("柜员号:3100525", 620.0, 720.0),
            ]
        )
    )
    return PageContent(
        page_number=page_number,
        source_page_number=page_number,
        page_mode=page_mode,
        texts=blocks,
    )


def _pab_ordinal_result(pages: list[PageContent], *, method: ExtractionMethod = ExtractionMethod.OCR) -> ParseResult:
    native_atoms = []
    for page in pages:
        page_id = f"page:{page.page_number:04d}"
        for index, block in enumerate(page.texts):
            native_atoms.append(
                EvidenceAtom(
                    id=f"native:{page.page_number}:{index}",
                    source_kind="pdf_native",
                    page_id=page_id,
                    text=block.content,
                    bbox=list(block.bbox or []),
                )
            )
    return ParseResult(
        pages=pages,
        entities=DocumentEntities(domain_specific={}),
        parser_info=ParserInfo(extraction_method=method, page_count=len(pages)),
        evidence_plane=CanonicalEvidencePlane(evidence=EvidenceStore(text_atoms=native_atoms)),
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "ocr_title",
        "ocr_missing_header",
        "ocr_header_order",
        "ocr_page_marker",
        "ocr_footer",
        "native_missing_header",
        "native_header_order",
        "native_footer",
    ],
)
def test_pab_page_ordinal_census_fails_closed_on_layout_contract_mutation(mutation: str) -> None:
    page = _pab_ordinal_page(1, 1, "2023年04月", [1, 2, 3])
    parse_result = _pab_ordinal_result([page])

    if mutation == "ocr_title":
        next(block for block in page.texts if block.content == "客户存款月结单").content = "账户明细"
    elif mutation == "ocr_missing_header":
        page.texts = [block for block in page.texts if block.content != "余额"]
    elif mutation == "ocr_header_order":
        sequence = next(block for block in page.texts if block.content == "序号")
        balance = next(block for block in page.texts if block.content == "余额")
        sequence.bbox, balance.bbox = balance.bbox, sequence.bbox
    elif mutation == "ocr_page_marker":
        next(block for block in page.texts if block.content == "第1页共1页").content = "第2页共2页"
    elif mutation == "ocr_footer":
        page.texts = [block for block in page.texts if block.content != "电子回单专用章"]
    elif mutation == "native_missing_header":
        atoms = parse_result.evidence_plane.evidence.text_atoms
        parse_result.evidence_plane.evidence.text_atoms = [atom for atom in atoms if atom.text != "余额"]
    elif mutation == "native_header_order":
        atoms = parse_result.evidence_plane.evidence.text_atoms
        sequence = next(atom for atom in atoms if atom.text == "序号")
        balance = next(atom for atom in atoms if atom.text == "余额")
        sequence.bbox, balance.bbox = balance.bbox, sequence.bbox
    elif mutation == "native_footer":
        atoms = parse_result.evidence_plane.evidence.text_atoms
        parse_result.evidence_plane.evidence.text_atoms = [
            atom
            for atom in atoms
            if not (atom.text.startswith("已打印次数:") or atom.text.startswith("打印时间:"))
        ]

    recover_ocr_implicit_ledger_tables(parse_result, "")

    assert recovered_ocr_implicit_row_evidence(parse_result) == (0, "", 0.0)


def test_scanned_pab_page_ordinal_census_caches_low_confidence_row_coverage() -> None:
    parse_result = _pab_ordinal_result(
        [
            _pab_ordinal_page(1, 3, "2023年03月", [1, 2, 3]),
            _pab_ordinal_page(2, 3, "2023年04月", [1, 2, 3, 4]),
            _pab_ordinal_page(3, 3, "2023年04月", [5, 6]),
        ]
    )

    recover_ocr_implicit_ledger_tables(parse_result, "")

    assert recovered_ocr_implicit_row_evidence(parse_result) == (9, "ocr_page_ordinal_census", 0.80)
    assert parse_result.entities.domain_specific["_bank_ocr_implicit_recovery"]["expected_row_page_counts"] == [
        3,
        4,
        2,
    ]


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("digital", (0, "", 0.0)),
        ("ordinal_gap", (0, "", 0.0)),
        ("missing_amount", (0, "", 0.0)),
        ("account_change", (0, "", 0.0)),
    ],
)
def test_pab_page_ordinal_census_rejects_unproved_or_non_scanned_sources(
    mutation: str,
    expected: tuple[int, str, float],
) -> None:
    pages = [
        _pab_ordinal_page(1, 2, "2023年04月", [1, 2, 3]),
        _pab_ordinal_page(2, 2, "2023年04月", [4, 5]),
    ]
    method = ExtractionMethod.OCR
    if mutation == "digital":
        method = ExtractionMethod.DIGITAL
        for page in pages:
            page.page_mode = None
    elif mutation == "ordinal_gap":
        sequence = next(block for block in pages[1].texts if block.content == "5" and block.bbox[0] < 50.0)
        sequence.content = "6"
    elif mutation == "missing_amount":
        pages[0].texts = [block for block in pages[0].texts if block.content != "-2.00"]
    elif mutation == "account_change":
        account = next(block for block in pages[1].texts if block.content.startswith("账号:"))
        account.content = "账号:11005350836202"
    parse_result = _pab_ordinal_result(pages, method=method)

    recover_ocr_implicit_ledger_tables(parse_result, "")

    assert recovered_ocr_implicit_row_evidence(parse_result) == expected


def test_pab_page_ordinal_census_rejects_whole_ocr_terminal_row_loss() -> None:
    pages = [
        _pab_ordinal_page(1, 2, "2023年04月", [1, 2, 3]),
        _pab_ordinal_page(2, 2, "2023年04月", [4, 5, 6]),
    ]
    parse_result = _pab_ordinal_result(pages)
    # The independent native evidence plane was sealed before OCR loss.  Remove
    # the complete final OCR row: ordinal continuity alone would falsely call
    # the shorter 1..5 prefix document-complete.
    final_y = next(
        block.bbox[1]
        for block in pages[1].texts
        if block.content == "6" and block.bbox is not None and block.bbox[0] < 50.0
    )
    pages[1].texts = [
        block
        for block in pages[1].texts
        if block.bbox is None or abs(float(block.bbox[1]) - float(final_y)) > 1.0
    ]

    recover_ocr_implicit_ledger_tables(parse_result, "")

    assert recovered_ocr_implicit_row_evidence(parse_result) == (0, "", 0.0)


def test_pab_page_ordinal_census_symmetric_terminal_deletion_remains_low_confidence() -> None:
    pages = [
        _pab_ordinal_page(1, 2, "2023年04月", [1, 2, 3]),
        _pab_ordinal_page(2, 2, "2023年04月", [4, 5, 6]),
    ]
    parse_result = _pab_ordinal_result(pages)
    final_y = next(
        block.bbox[1]
        for block in pages[1].texts
        if block.content == "6" and block.bbox is not None and block.bbox[0] < 50.0
    )
    pages[1].texts = [
        block
        for block in pages[1].texts
        if block.bbox is None or abs(float(block.bbox[1]) - float(final_y)) > 1.0
    ]
    native_atoms = parse_result.evidence_plane.evidence.text_atoms
    parse_result.evidence_plane.evidence.text_atoms = [
        atom
        for atom in native_atoms
        if atom.page_id != "page:0002"
        or atom.bbox is None
        or abs(float(atom.bbox[1]) - float(final_y)) > 1.0
    ]

    # Both representations now agree on a shorter contiguous prefix.  That
    # agreement remains useful for ranking, but cannot certify the missing tail.
    recover_ocr_implicit_ledger_tables(parse_result, "")

    assert recovered_ocr_implicit_row_evidence(parse_result) == (5, "ocr_page_ordinal_census", 0.80)


def test_pab_page_ordinal_census_requires_independent_native_row_spine() -> None:
    parse_result = _pab_ordinal_result([_pab_ordinal_page(1, 1, "2023年04月", [1, 2, 3])])
    parse_result.evidence_plane = None

    recover_ocr_implicit_ledger_tables(parse_result, "")

    assert recovered_ocr_implicit_row_evidence(parse_result) == (0, "", 0.0)
