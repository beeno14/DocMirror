# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Institution column maps + grid_standard / split_debit_credit integration."""

from __future__ import annotations

import csv
import io
from types import SimpleNamespace

import pytest

from docmirror.models.entities.parse_result import (
    CellValue,
    DocumentEntities,
    LogicalTable,
    PageContent,
    ParseResult,
    RowProvenance,
    TableBlock,
    TableRow,
    TextBlock,
)
from docmirror.models.sealed import seal_parse_result
from docmirror.plugins.bank_statement.community_plugin import BankStatementCommunityPlugin, _sanitize_bank_records
from docmirror.plugins.bank_statement.context import StyleContext, build_style_context
from docmirror.plugins.bank_statement.extract_pipeline import (
    _apply_source_reported_transaction_count,
    run_bank_statement_extract,
)
from docmirror.plugins.bank_statement.header_resolve import detect_headers
from docmirror.plugins.bank_statement.institution import match_institution, normalize_table_headers
from docmirror.plugins.bank_statement.ltro import ReconstructionMeta
from docmirror.plugins.bank_statement.style_detector import BankStyleDetector
from docmirror.plugins.bank_statement.style_registry import BankStyleParserRegistry
from docmirror.plugins.bank_statement.styles.grid_standard import normalize_record, normalize_split_debit_credit
from docmirror.plugins.bank_statement.wide_table_recovery import (
    _normalize_native_grid_table,
    _select_wide_bank_table,
    is_wide_bank_header,
)


def test_match_institution_ccb():
    variant = match_institution("中国建设银行账户明细信息")
    assert variant is not None
    assert variant.id == "ccb"
    assert variant.column_map.get("交易日期") == "交易时间"


def test_normalize_table_headers_ccb_alias():
    variant = match_institution("中国建设银行")
    tables = [
        [
            ["交易日期", "摘要", "余额", "对方账号", "对方户名"],
            ["2024-01-01", "转账", "100.00", "6210000000000000", "测试公司"],
        ]
    ]
    normalized = normalize_table_headers(tables, variant=variant)
    assert normalized[0][0][0] == "交易时间"
    assert normalized[0][0][-2:] == ["对方账号", "对方户名"]


def test_split_debit_credit_style_detection():
    ctx = StyleContext(
        tables=[
            [
                ["交易日期", "摘要", "收入", "支出", "余额"],
                ["2024-01-01", "工资入账", "5000.00", "0.00", "8000.00"],
            ]
        ],
        full_text="中国工商银行 个人客户交易明细",
        institution=None,
        page_count=1,
    )
    result = BankStyleDetector().detect(ctx)
    assert result.primary_style == "split_debit_credit"


def test_style_registry_icbc_split_columns():
    ctx = StyleContext(
        tables=[
            [
                ["交易日期", "摘要", "收入", "支出", "余额"],
                ["2024-01-01", "工资入账", "5000.00", "0.00", "8000.00"],
                ["2024-01-02", "转账支出", "0.00", "200.00", "7800.00"],
                ["2024-01-03", "消费", "0.00", "50.00", "7750.00"],
            ]
        ],
        full_text="中国工商银行\n个人客户交易明细\n户名：张三",
        institution=None,
        page_count=1,
    )
    detection = BankStyleDetector().detect(ctx)
    plugin = BankStatementCommunityPlugin()
    records, _identity = BankStyleParserRegistry().run(detection, ctx, plugin)
    assert len(records) >= 3
    directions = {r["normalized"].get("direction") for r in records}
    assert "income" in directions
    assert "expense" in directions


def test_recovery_candidate_is_consumed_instead_of_reopening_sparse_logical_table():
    """A selected generic recovery table must not be replaced by the original narrow logical table."""
    sparse = LogicalTable(
        table_id="lt_sparse",
        headers=["交易日期", "交易金额", "余额"],
        rows=[TableRow(cells=[CellValue(text="2024-01-01"), CellValue(text="+10.00"), CellValue(text="10.00")])],
        row_count=1,
        data_row_estimate=1,
        quality_passed=True,
    )
    physical_values = [
        ["2024-01-01", "+10.00", "10.00", "甲公司"],
        ["2024-01-02", "-2.00", "8.00", "乙公司"],
        ["2024-01-03", "+3.00", "11.00", "丙公司"],
    ]
    parse_result = ParseResult(
        pages=[
            PageContent(
                page_number=1,
                tables=[
                    TableBlock(
                        table_id="pt_1_0",
                        headers=["交易日期", "交易金额", "余额", "对方户名"],
                        rows=[
                            TableRow(
                                cells=[CellValue(text=value) for value in row],
                                source_page=1,
                                source_physical_id="pt_1_0",
                                source_row_index=index,
                            )
                            for index, row in enumerate(physical_values)
                        ],
                    )
                ],
            )
        ],
        logical_tables=[sparse],
    )

    ctx = StyleContext(
        tables=[[sparse.headers, *[[cell.text for cell in row.cells] for row in sparse.rows]]],
        full_text="某银行交易明细 总笔数：3",
        institution=None,
        page_count=1,
        parse_result=parse_result,
        reconstruction=ReconstructionMeta(source="canonical_table", expected_primary_rows=1),
    )
    records, _ = BankStyleParserRegistry().run(
        BankStyleDetector().detect(ctx),
        ctx,
        BankStatementCommunityPlugin(),
    )

    assert len(records) == 3
    assert [record["normalized"]["direction"] for record in records] == ["income", "expense", "income"]


def test_normalize_split_debit_credit_direct():
    plugin = BankStatementCommunityPlugin()
    norm = normalize_split_debit_credit(
        {
            "交易日期": "2024-01-02",
            "摘要": "转账支出",
            "收入": "0.00",
            "支出": "200.00",
            "余额": "7800.00",
        },
        plugin,
    )
    assert norm is not None
    assert norm["amount"] == 200.0
    assert norm["direction"] == "expense"


def test_normalize_transaction_location_as_channel_alias():
    plugin = BankStatementCommunityPlugin()
    norm = normalize_split_debit_credit(
        {
            "交易日期": "2025-07-10",
            "贷方发生额": "30,000.00",
            "借方发生额": "",
            "余额": "36,989.93",
            "交易地点": "支付平台",
        },
        plugin,
    )

    assert norm is not None
    assert norm["channel"] == "支付平台"


def test_normalize_merged_balance_and_timestamp_split_columns():
    plugin = BankStatementCommunityPlugin()
    norm = normalize_split_debit_credit(
        {
            "交易时间": "2025/01/0316:18:35",
            "摘要": "个人所得税",
            "借方发生额": "15.00",
            "贷方发生额": "0.00",
            "账户余额流水号": "363,693.0255420250100824712870",
        },
        plugin,
    )
    assert norm is not None
    assert norm["amount"] == 15.0
    assert norm["direction"] == "expense"
    assert norm["balance"] == 363693.02


def test_normalize_direction_embedded_after_amount():
    plugin = BankStatementCommunityPlugin()
    norm = normalize_split_debit_credit(
        {
            "交易日期": "2023-10-02",
            "摘要": "跨行代付",
            "支/收交易金额": "23,903.69付收",
            "账户余额": "23,903.69",
        },
        plugin,
    )
    assert norm is not None
    assert norm["amount"] == 23903.69
    assert norm["direction"] == "income"
    assert norm["balance"] == 23903.69


@pytest.mark.parametrize(
    ("raw_direction", "expected"),
    [("贷Cr", "income"), ("借Dr", "expense")],
)
def test_normalize_bilingual_debit_credit_flag(raw_direction, expected):
    plugin = BankStatementCommunityPlugin()
    norm = normalize_split_debit_credit(
        {
            "交易日期": "2022-08-05",
            "借贷": raw_direction,
            "交易金额": "40.00",
            "余额": "41.06",
        },
        plugin,
    )

    assert norm is not None
    assert norm["direction"] == expected


@pytest.mark.parametrize(
    ("raw_direction", "expected"),
    [
        ("转入", "income"),
        ("转出", "expense"),
        ("转\n入", "income"),
        ("转 出", "expense"),
        ("收入", "income"),
        ("支出", "expense"),
    ],
)
def test_normalize_transaction_category_direction(raw_direction, expected):
    plugin = BankStatementCommunityPlugin()
    norm = normalize_split_debit_credit(
        {
            "交易日期": "2023-08-29",
            "交易类别": raw_direction,
            "交易金额": "228.00",
            "账户余额": "372.38",
        },
        plugin,
    )

    assert norm is not None
    assert norm["amount"] == 228.0
    assert norm["direction"] == expected


@pytest.mark.parametrize(
    ("raw_direction", "expected"),
    [("收入", "income"), ("支出", "expense")],
)
def test_normalize_transaction_type_direction_alias(raw_direction, expected):
    plugin = BankStatementCommunityPlugin()
    norm = normalize_split_debit_credit(
        {
            "交易日期": "2023-08-31",
            "交易类型": raw_direction,
            "交易金额": "31.00",
            "账户余额": "99.79",
        },
        plugin,
    )

    assert norm is not None
    assert norm["amount"] == 31.0
    assert norm["direction"] == expected


def test_canonical_logical_grid_preserves_generic_row_provenance_and_raw_columns():
    headers = ["交易日期", "交易金额", "交易类别", "账户余额", "对方账号", "对方户名", "备注", "交易机构"]
    raw_rows = [
        ["20230829", "228.00", "转出", "372.38", "243300133", "扫二维码付款", "财付通支\n付", "101001"],
        ["20230828", "1600.00", "转入", "1972.38", "6230000000000000000", "测试对手方", "转账", "101001"],
    ]
    rows = []
    provenance = []
    for row_index, (page_number, values) in enumerate(zip((1, 2), raw_rows, strict=True)):
        cells = [
            CellValue(
                text=value,
                evidence_ids=[f"ev:{page_number:04d}:{row_index:06d}:{col_index:02d}"],
                source_cell_refs=[
                    {
                        "page": page_number,
                        "table_id": f"pt_{page_number}_0",
                        "row": row_index,
                        "col": col_index,
                    }
                ],
            )
            for col_index, value in enumerate(values)
        ]
        rows.append(
            TableRow(
                cells=cells,
                source_page=page_number,
                source_physical_id=f"pt_{page_number}_0",
                source_row_index=row_index,
            )
        )
        provenance.append(
            RowProvenance(
                source_page=page_number,
                source_table_id=f"pt_{page_number}_0",
                source_row_index=row_index,
            )
        )

    logical_table = LogicalTable(
        table_id="lt_transactions",
        headers=headers,
        rows=rows,
        source_physical_ids=["pt_1_0", "pt_2_0"],
        source_pages=[1, 2],
        page_span=(1, 2),
        row_count=2,
        provenance=provenance,
    )
    parse_result = ParseResult(logical_tables=[logical_table])
    ctx = StyleContext(
        tables=[[headers, *raw_rows]],
        full_text="银行账户交易明细",
        institution=None,
        page_count=2,
        parse_result=parse_result,
        reconstruction=ReconstructionMeta(source="canonical_table", expected_primary_rows=2),
    )
    detection = BankStyleDetector().detect(ctx)
    records, _identity = BankStyleParserRegistry().run(
        detection,
        ctx,
        BankStatementCommunityPlugin(),
    )

    assert len(records) == 2
    assert [record["normalized"]["direction"] for record in records] == ["expense", "income"]
    assert records[0]["normalized"]["summary"] == "财付通支付"
    assert records[0]["raw"]["备注"] == "财付通支\n付"
    assert records[0]["raw"]["交易机构"] == "101001"
    assert "交易机构" not in records[0]["normalized"]
    assert "交易机构" not in records[0]["canonical_raw"]
    assert [record["source"]["source_page"] for record in records] == [1, 2]
    assert [record["source"]["table_id"] for record in records] == ["pt_1_0", "pt_2_0"]
    assert all(record["source"]["source_cell_refs"] for record in records)
    assert all(record["source"]["evidence_ids"] for record in records)


def test_canonical_split_unit_grid_preserves_wrapped_counter_accounts() -> None:
    headers = [
        "序号",
        "交易日期",
        "交易流水号",
        "支出（元）",
        "收入（元）",
        "账户余额（元）",
        "对方账号",
        "对方户名",
        "摘要",
    ]
    raw_rows = [
        [
            "1",
            "2025-01-24\n16:38:19",
            "004010100551005",
            "200000.00",
            "",
            "2369231.13",
            "830100788013000002\n20",
            "重庆中链农科技有限公司",
            "企业网银-跨行转账（实时）",
        ],
        [
            "2",
            "2025-03-10\n12:02:13",
            "004010100245394",
            "100.00",
            "",
            "9481.13",
            "120023710020000001\n988",
            "重庆数宜信信用管理有限公司",
            "企业网银-跨行转账（实时）",
        ],
    ]
    rows = [
        TableRow(
            cells=[CellValue(text=value) for value in values],
            source_page=1,
            source_physical_id="pt_1_0",
            source_row_index=row_index,
        )
        for row_index, values in enumerate(raw_rows)
    ]
    parse_result = ParseResult(
        logical_tables=[
            LogicalTable(
                table_id="lt_transactions",
                headers=headers,
                rows=rows,
                source_physical_ids=["pt_1_0"],
                source_pages=[1],
                page_span=(1, 1),
                row_count=2,
            )
        ]
    )

    ctx = build_style_context(parse_result, "银行账户交易明细")
    detection = BankStyleDetector().detect(ctx)
    records, _identity = BankStyleParserRegistry().run(
        detection,
        ctx,
        BankStatementCommunityPlugin(),
    )

    assert ctx.reconstruction is not None
    assert ctx.reconstruction.source == "canonical_table"
    assert len(records) == 2
    assert [record["normalized"]["counter_account"] for record in records] == [
        "83010078801300000220",
        "120023710020000001988",
    ]
    assert [record["normalized"]["counter_party"] for record in records] == [
        "重庆中链农科技有限公司",
        "重庆数宜信信用管理有限公司",
    ]


def test_canonical_stacked_bilingual_headers_preserve_debit_credit_and_counterparty() -> None:
    headers = [
        "交易日期\nTransaction Date",
        "交易流水号\nTeller's Serial Number",
        "发生额\nTransaction Amount",
        "",
        "账户余额\nAccount Balance",
        "交易对手信息\nCounterparty Information",
        "",
        "摘要代码\nAbstract Code",
        "备注\nDescription",
    ]
    raw_rows = [
        ["", "", "借方\nDebit", "贷方\nCredit", "", "对手机构", "对手名称", "", ""],
        ["2025/01/02", "0001", "50.00", "", "100.00", "浦发银行重庆分行", "甲公司", "S1", "付款"],
        ["2025/01/03", "0002", "", "75.00", "175.00", "招商银行重庆分行", "乙公司", "S2", "收款"],
    ]
    rows = [
        TableRow(
            cells=[CellValue(text=value) for value in values],
            source_page=1,
            source_physical_id="pt_1_0",
            source_row_index=row_index,
        )
        for row_index, values in enumerate(raw_rows)
    ]
    parse_result = ParseResult(
        logical_tables=[
            LogicalTable(
                table_id="lt_transactions",
                headers=headers,
                rows=rows,
                source_physical_ids=["pt_1_0"],
                source_pages=[1],
                page_span=(1, 1),
                row_count=2,
            )
        ]
    )

    ctx = build_style_context(parse_result, "企业电子对账单")
    detection = BankStyleDetector().detect(ctx)
    records, _identity = BankStyleParserRegistry().run(
        detection,
        ctx,
        BankStatementCommunityPlugin(),
    )

    assert ctx.reconstruction is not None
    assert ctx.reconstruction.source == "canonical_table"
    assert len(records) == 2
    assert [record["normalized"]["direction"] for record in records] == ["expense", "income"]
    assert [record["normalized"]["amount"] for record in records] == [50.0, 75.0]
    assert [record["normalized"]["balance"] for record in records] == [100.0, 175.0]
    assert [record["normalized"]["counter_party"] for record in records] == ["甲公司", "乙公司"]
    assert [record["source"]["source_page"] for record in records] == [1, 1]

    sanitized = _sanitize_bank_records(records)
    assert [record["canonical_raw"]["amount"] for record in sanitized] == ["50.00", "75.00"]


def test_split_grid_reads_bilingual_counterparty_and_repairs_canonical_raw_amount() -> None:
    raw = {
        "交易日期\nTransaction Date": "2025-01-02",
        "发生额\nTransaction Amount\n借方\nDebit": "88.20",
        "发生额\nTransaction Amount\n贷方\nCredit": "",
        "余额\nBalance": "911.80",
        "交易对手信息\nCounterparty Information\n对手机构\nCounterparty\nInstitution": "测试银行科技支行",
        "对手名称\nCounterparty Name": "测试供应链有限公司",
        "备注\nDescription": "采购付款",
    }
    normalized = normalize_split_debit_credit(raw, BankStatementCommunityPlugin())
    assert normalized is not None

    records = _sanitize_bank_records(
        [
            {
                "raw": raw,
                "normalized": normalized,
                "canonical_raw": {"amount": "", "amount_cny": "", "counter_party": ""},
            }
        ]
    )

    assert normalized["direction"] == "expense"
    assert normalized["amount"] == 88.2
    assert normalized["counter_party"] == "测试供应链有限公司"
    assert normalized["counter_bank_name"] == "测试银行科技支行"
    assert records[0]["canonical_raw"]["amount"] == "88.20"
    assert records[0]["canonical_raw"]["amount_cny"] == "88.20"
    assert records[0]["canonical_raw"]["counter_party"] == "测试供应链有限公司"
    assert records[0]["canonical_raw"]["counter_bank_name"] == "测试银行科技支行"


def test_stacked_split_grid_infers_single_page_sources_from_logical_rows():
    headers = [
        "序号",
        "交易日期",
        "交易时间",
        "摘要",
        "凭证种类",
        "借方发生额",
        "贷方发生额",
        "余额",
        "对方账户",
        "对方户名",
    ]
    raw_rows: list[list[str]] = []
    table_rows: list[TableRow] = []
    provenance: list[RowProvenance] = []
    page_counts = {1: 23, 2: 23, 3: 23, 4: 22}
    sequence = 1
    for page_number, count in page_counts.items():
        for row_index in range(count):
            amount = "10.00" if sequence % 2 else ""
            credit = "" if sequence % 2 else "20.00"
            values = [
                str(sequence),
                "2022-06-01",
                "2022-06-01 10:00:00",
                "测试",
                "",
                amount,
                credit,
                f"{1000 + sequence}.00",
                f"62220000{sequence:04d}",
                f"测试对手方{sequence}",
            ]
            raw_rows.append(values)
            refs = [
                {"page": page_number, "table_id": f"pt_{page_number}_0", "row": row_index, "col": col_index}
                for col_index, _value in enumerate(values)
            ]
            cells = [
                CellValue(text=value, source_cell_refs=[refs[col_index]]) for col_index, value in enumerate(values)
            ]
            table_rows.append(
                TableRow(
                    cells=cells,
                    source_page=page_number,
                    source_physical_id=f"pt_{page_number}_0",
                    source_row_index=row_index,
                    source_cell_refs=refs,
                )
            )
            provenance.append(
                RowProvenance(
                    source_page=page_number,
                    source_table_id=f"pt_{page_number}_0",
                    source_row_index=row_index,
                )
            )
            sequence += 1

    logical_table = LogicalTable(
        table_id="lt_stacked",
        headers=headers,
        rows=table_rows,
        source_physical_ids=[f"pt_{page}_0" for page in page_counts],
        source_pages=list(page_counts),
        page_span=(1, 4),
        row_count=len(table_rows),
        provenance=provenance,
    )
    ctx = StyleContext(
        tables=[[headers, *raw_rows]],
        full_text="江苏银行交易明细",
        institution=None,
        page_count=4,
        parse_result=ParseResult(logical_tables=[logical_table]),
        reconstruction=ReconstructionMeta(source="stacked_text", expected_primary_rows=91),
    )

    detection = BankStyleDetector().detect(ctx)
    records, _identity = BankStyleParserRegistry().run(detection, ctx, BankStatementCommunityPlugin())

    assert len(records) == 91
    distribution: dict[int, int] = {}
    for record in records:
        source = record["source"]
        source_page = source["source_page"]
        distribution[source_page] = distribution.get(source_page, 0) + 1
        assert source["page_range"] == [source_page, source_page]
        assert source["source_cell_refs"]
    assert distribution == page_counts


def test_stacked_split_grid_infers_sources_from_page_text_anchors_when_tables_are_absent():
    headers = ["序号", "交易日期", "交易时间", "摘要", "借方发生额", "贷方发生额", "余额", "对方账户", "对方户名"]
    raw_rows = [
        ["1", "2022-06-01", "2022-06-01 10:00:00", "往来款", "100.00", "", "900.00", "622200001", "甲公司"],
        ["2", "2022-06-02", "2022-06-02 10:00:00", "收费", "2.00", "", "898.00", "622200002", "手续费收入"],
        ["3", "2022-07-01", "2022-07-01 10:00:00", "往来款", "", "200.00", "1098.00", "622200003", "乙公司"],
        ["4", "2022-08-01", "2022-08-01 10:00:00", "往来款", "50.00", "", "1048.00", "622200004", "丙公司"],
    ]
    pages = [
        PageContent(
            page_number=1, texts=[TextBlock(content="1 2022-06-01 10:00:00 往来款 100.00 900.00 622200001 甲公司")]
        ),
        PageContent(
            page_number=1, texts=[TextBlock(content="2 2022-06-02 10:00:00 收费 2.00 898.00 622200002 手续费收入")]
        ),
        PageContent(
            page_number=2, texts=[TextBlock(content="3 2022-07-01 10:00:00 往来款 200.00 1,098.00 622200003 乙公司")]
        ),
        PageContent(
            page_number=3, texts=[TextBlock(content="4 2022-08-01 10:00:00 往来款 50.00 1,048.00 622200004 丙公司")]
        ),
    ]
    ctx = StyleContext(
        tables=[[headers, *raw_rows]],
        full_text="银行交易明细",
        institution=None,
        page_count=3,
        parse_result=ParseResult(pages=pages),
        reconstruction=ReconstructionMeta(source="stacked_text", expected_primary_rows=4),
    )

    detection = BankStyleDetector().detect(ctx)
    records, _identity = BankStyleParserRegistry().run(detection, ctx, BankStatementCommunityPlugin())

    assert len(records) == 4
    assert [record["source"]["source_page"] for record in records] == [1, 1, 2, 3]
    assert [record["source"]["page_range"] for record in records] == [[1, 1], [1, 1], [2, 2], [3, 3]]


def test_split_grid_recovers_empty_counterparty_from_same_page_source_text():
    headers = ["序号", "交易日期", "交易时间", "摘要", "借方发生额", "贷方发生额", "余额", "对方账户", "对方户名"]
    raw_rows = [
        [
            "13",
            "2022-06-13",
            "2022-06-13 18:19:36",
            "公共耗能和水电费用",
            "101.80",
            "",
            "54.15",
            "6232511300395178",
            "限公司",
        ],
        ["14", "2022-06-13", "2022-06-13 18:19:36", "收费", "2.00", "", "52.15", "70650107360000033", "入"],
        ["15", "2022-06-21", "2022-06-21 00:21:02", "结息", "", "53.14", "6226.06", "", ""],
        [
            "16",
            "2022-08-03",
            "2022-08-03 17:35:14",
            "tips扣税",
            "2159.00",
            "",
            "1320.91",
            "70010151830005003",
            "代收）",
        ],
    ]
    source_text = "\n".join(
        [
            "序号",
            "交易日期",
            "交易时间",
            "摘要",
            "借方发生额",
            "贷方发生额",
            "余额",
            "对方账户",
            "对方户名",
            "13",
            "2022-06-13",
            "18:19:36",
            "公共耗能和水电费用",
            "101.80",
            "54.15",
            "6232511300395",
            "178",
            "镇江大学科技园",
            "资产经营管理有",
            "限公司",
            "14",
            "2022-06-13",
            "18:19:36",
            "收费",
            "2.00",
            "52.15",
            "7065010736000",
            "0033",
            "企业电子渠道跨",
            "行转账手续费收",
            "入",
            "15",
            "2022-06-21",
            "00:21:02",
            "结息",
            "53.14",
            "6226.06",
            "null",
            "16",
            "2022-08-03",
            "17:35:14",
            "tips扣税",
            "2,159.00",
            "1,320.91",
            "7001015183000",
            "5003",
            "待报解预算收入",
            "（财税库银联网",
            "代收）",
        ]
    )
    page_anchor_text = "\n".join(
        [
            "13 2022-06-13 18:19:36 公共耗能和水电费用 101.80 54.15 6232511300395178",
            "14 2022-06-13 18:19:36 收费 2.00 52.15 70650107360000033",
            "15 2022-06-21 00:21:02 结息 53.14 6226.06",
            "16 2022-08-03 17:35:14 tips扣税 2159.00 1320.91 70010151830005003",
        ]
    )
    ctx = StyleContext(
        tables=[[headers, *raw_rows]],
        full_text=source_text,
        institution=None,
        page_count=1,
        parse_result=ParseResult(pages=[PageContent(page_number=1, texts=[TextBlock(content=page_anchor_text)])]),
        reconstruction=ReconstructionMeta(source="stacked_text", expected_primary_rows=4),
    )

    detection = BankStyleDetector().detect(ctx)
    records, _identity = BankStyleParserRegistry().run(detection, ctx, BankStatementCommunityPlugin())

    assert len(records) == 4
    assert records[0]["raw"]["对方户名"] == "镇江大学科技园资产经营管理有限公司"
    assert records[0]["normalized"]["counter_party"] == "镇江大学科技园资产经营管理有限公司"
    assert records[1]["normalized"]["counter_party"] == "企业电子渠道跨行转账手续费收入"
    assert records[2]["normalized"]["counter_party"] == ""
    assert records[3]["normalized"]["counter_party"] == "待报解预算收入（财税库银联网代收）"


def test_split_grid_rejects_column_ordered_page_text_as_counterparty():
    headers = ["序号", "交易日期", "交易时间", "摘要", "借方发生额", "贷方发生额", "余额", "对方账户", "对方户名"]
    raw_rows = [
        ["159", "2023-01-27", "", "转出", "20,000.00", "", "73,155.95", "1000050001", "限公司"],
        ["160", "2023-01-27", "", "转出", "50,000.00", "", "23,155.95", "215500690", "WL支付宝"],
    ]
    column_ordered_text = "\n".join(
        [
            "159",
            "160",
            "2023-01-27",
            "2023-01-27",
            "20,000.00",
            "50,000.00",
            "73,155.95",
            "23,155.95",
            "1000050001",
            "215500690",
            "清单支出算术合计:19,756,586.06",
            "打印渠道:远程视频柜员机",
            "打印机构:907072604",
            "WL财付通微信转账:微信转账",
            "WL财付通微信转账:微信转账",
            "WL支付宝",
            "对方户名张祝祥陈元友",
        ]
    )
    ctx = StyleContext(
        tables=[[headers, *raw_rows]],
        full_text=column_ordered_text,
        institution=None,
        page_count=1,
        parse_result=ParseResult(pages=[PageContent(page_number=1, texts=[TextBlock(content=column_ordered_text)])]),
        reconstruction=ReconstructionMeta(source="canonical_evidence_table", expected_primary_rows=2),
    )

    detection = BankStyleDetector().detect(ctx)
    records, _identity = BankStyleParserRegistry().run(detection, ctx, BankStatementCommunityPlugin())
    records = _sanitize_bank_records(records)

    assert len(records) == 2
    assert records[0]["normalized"]["counter_party"] == ""
    assert "清单支出算术合计" not in records[0]["raw"]["对方户名"]
    assert records[1]["normalized"]["counter_party"] == "WL支付宝"


def test_signed_grid_keeps_source_null_interest_counterparty_empty():
    headers = ["交易日期", "对方户名", "对方账号/卡号", "交易摘要", "发生额", "余额", "币种"]
    row = ["2024-06-20\n20:59:30", "", "0000000000000", "结存款息", "4.37", "17486.33", "CNY"]
    source_text = "2024-06-20 20:59:30 0000000000000 结存款息 4.37 17486.33 CNY"
    ctx = StyleContext(
        tables=[[headers, row]],
        full_text=source_text,
        institution=None,
        page_count=1,
        parse_result=ParseResult(pages=[PageContent(page_number=1, texts=[TextBlock(content=source_text)])]),
        reconstruction=ReconstructionMeta(source="canonical_table", expected_primary_rows=1),
    )

    records, _identity = BankStyleParserRegistry().run(
        BankStyleDetector().detect(ctx),
        ctx,
        BankStatementCommunityPlugin(),
    )

    assert len(records) == 1
    assert records[0]["raw"]["对方户名"] == ""
    assert records[0]["normalized"]["counter_party"] == ""


def test_registry_prefers_semantic_text_table_when_canonical_grid_coverage_is_low():
    bad_headers = [
        "序号",
        "交易日期",
        "交易时间",
        "摘要",
        "凭证种类",
        "借方发生额",
        "贷方发生额",
        "余额",
        "对方账户",
        "对方户名",
    ]
    bad_rows = [
        [
            "1",
            "2023-06-01",
            "11:47:14",
            "往来款",
            "16,500.00",
            "17,286.21",
            "7065018800015",
            "6836",
            "镇江一生一世好",
            "",
        ],
        ["3", "2023-06-01", "11:48:53", "工资", "514.46", "16,674.25", "6228760805004", "170034", "俞佩", ""],
    ]
    full_text = "\n".join(
        [
            "借方笔数：2   借方发生总额：611.96   贷方笔数：1   贷方发生总额：16,500.00   合计笔数：3",
            "序号",
            "交易日期",
            "交易时间",
            "摘要",
            "凭证种类",
            "借方发生额",
            "贷方发生额",
            "余额",
            "对方账户",
            "对方户名",
            "1",
            "2023-06-01",
            "11:47:14",
            "往来款",
            "16,500.00",
            "17,286.21",
            "7065018800015",
            "6836",
            "镇江一生一世好",
            "游戏有限公司",
            "2",
            "2023-06-01",
            "11:48:53",
            "工资",
            "97.50",
            "17,188.71",
            "6228760801004",
            "812493",
            "杨洁",
            "3",
            "2023-06-01",
            "11:48:53",
            "工资",
            "514.46",
            "16,674.25",
            "6228760805004",
            "170034",
            "俞佩",
        ]
    )
    ctx = StyleContext(
        tables=[[bad_headers, *bad_rows]],
        full_text=full_text,
        institution=None,
        page_count=1,
        parse_result=ParseResult(pages=[PageContent(page_number=1, texts=[TextBlock(content=full_text)])]),
        reconstruction=ReconstructionMeta(source="canonical_table", expected_primary_rows=3),
    )

    records, _identity = BankStyleParserRegistry().run(
        BankStyleDetector().detect(ctx),
        ctx,
        BankStatementCommunityPlugin(),
    )

    assert len(records) == 3
    assert ctx.reconstruction is not None
    assert ctx.reconstruction.source == "semantic_text_table"
    assert records[0]["normalized"]["amount"] == 16500.0
    assert records[0]["normalized"]["balance"] == 17286.21
    assert records[0]["normalized"]["counter_account"] == "70650188000156836"
    assert records[0]["normalized"]["counter_party"] == "镇江一生一世好游戏有限公司"
    assert records[1]["normalized"]["direction"] == "expense"
    assert records[1]["normalized"]["amount"] == 97.5


def test_wide_table_accepts_date_anchored_rows_without_sequence_column():
    table = [
        ["交易时间", "摘要", "借方发生额", "贷方发生额", "账户余额流水号"],
        ["2025/01/0316:18:35", "个人所得税", "15.00", "0.00", "363,693.0255420250100824712870"],
        ["2025/01/0710:41:35", "社保费", "113.16", "0.00", "363,579.8650400202507000001657"],
    ]
    assert is_wide_bank_header(table[0]) is True
    assert len(_select_wide_bank_table(table)) == 3


def test_wide_table_accepts_direction_embedded_amount_header():
    table = [
        ["交易日期", "记账日期", "摘要", "支/收交易金额", "账户余额"],
        ["2023-10-02", "2023-10-02", "跨行代付", "23,903.69付收", "23,903.69"],
        ["2023-10-07", "2023-10-07", "跨行代付", "13,610.09付收", "13,610.09"],
    ]
    assert is_wide_bank_header(table[0]) is True
    assert len(_select_wide_bank_table(table)) == 3


def test_explicit_counter_account_header_overrides_earlier_own_account_column() -> None:
    table = [
        ["交易日期", "账号", "摘要", "收入/支出金额", "余额", "对方户名", "对方账号"],
        ["2023-03-03", "1910213201003000000", "转账", "-10.00", "90.00", "测试对方", "6222000000000000"],
    ]

    header = detect_headers([table], BankStatementCommunityPlugin().column_registry)

    assert header is not None
    assert header.col_map["counter_account"] == 6


def test_source_reported_total_overrides_single_page_identity_count() -> None:
    identity = {
        "total_transactions": {
            "raw_value": "25",
            "normalized_value": "25",
            "source": "header.kv",
        }
    }

    _apply_source_reported_transaction_count(identity, 146)

    assert identity["total_transactions"] == {
        "raw_name": "page_footer_transaction_count",
        "raw_value": "146",
        "normalized_value": "146",
        "data_type": "integer",
        "source": "page_footer.sum",
    }


def test_native_grid_recovers_watermarked_combined_amount_header_with_provenance() -> None:
    matrix = [
        ["交易日期", "摘要", "2 C 9 收入/支出金额", "余额", "对方户名"],
        ["2023-03-03 12:43:17", "开户", "8 4 +0.00", "0.00", "（空）"],
        ["行 20银23-03-03 12:51:10", "卡存", "9 2 +60,000.00", "6 2 60,000.00", "（空）"],
    ]
    rows = [
        SimpleNamespace(cells=[(10.0, float(index * 20), 20.0, float(index * 20 + 10))] * len(matrix[0]))
        for index in range(len(matrix))
    ]
    table = SimpleNamespace(extract=lambda: matrix, rows=rows)

    recovered = _normalize_native_grid_table(
        table,
        page_number=3,
        table_index=1,
        money_hints={
            ("2023-03-03", "12:43:17"): [("+0.00", "0.00")],
            ("2023-03-03", "12:51:10"): [("+60,000.00", "60,000.00")],
        },
    )

    assert is_wide_bank_header(recovered[0]) is True
    assert recovered[1][2:4] == ["+0.00", "0.00"]
    assert recovered[2][0] == "2023-03-03 12:51:10"
    assert recovered[2][2:4] == ["+60,000.00", "60,000.00"]
    assert recovered[1][-4:] == ["3", "10.000,20.000,20.000,30.000", "native:p3:t1", "1"]
    assert len(_select_wide_bank_table(recovered)) == 3


_SHANGRAO_HEADERS = [
    "序号",
    "交易时间",
    "流水号",
    "对方账号",
    "对方户名",
    "支出",
    "收入",
    "账户余额",
    "摘要",
    "附言",
]


def _sourced_bank_row(values: list[str], *, page: int, row_index: int) -> TableRow:
    refs = [
        {
            "page": page,
            "table_id": f"pt_{page}_0",
            "row": row_index,
            "raw_row": row_index + 1,
            "col": col_index,
        }
        for col_index, _value in enumerate(values)
    ]
    return TableRow(
        cells=[
            CellValue(
                text=value,
                evidence_ids=[f"ev:{page:04d}:text:{row_index:06d}:{col_index:02d}"],
                source_cell_refs=[refs[col_index]],
            )
            for col_index, value in enumerate(values)
        ],
        source_page=page,
        source_physical_id=f"pt_{page}_0",
        source_row_index=row_index,
        source_cell_refs=refs,
    )


def _cross_page_bank_parse_result(
    *,
    valid_page_two_row: bool = False,
    repeated_header_fragment: bool = False,
    fragment_page: int = 2,
    fragment_row_index: int = 0,
) -> ParseResult:
    page_one_values = [
        "13",
        "2023-",
        "1112052",
        "7272798",
        "江西昌荣",
        "",
        "1000000",
        "1006296.",
        "超网-贷记",
        "转户",
    ]
    page_two_values = (
        [
            "",
            "2023-\n06-28\n18:00:00",
            "",
            "7272798\n0000001\n1760",
            "江西昌荣\n供应链有限公司",
            "",
            "500",
            "1006796.\n3",
            "超网-贷记\n转入",
            "转户",
        ]
        if valid_page_two_row
        else [
            "",
            "",
            "",
            "",
            "",
            "借方\nDebit",
            "贷方\nCredit",
            "",
            "对手名称\nCounterparty Name",
            "备注\nDescription",
        ]
        if repeated_header_fragment
        else [
            "",
            "06-28\n19:50:16",
            "",
            "0000001\n1760",
            "供应链有\n限公司",
            "",
            "",
            "3",
            "转入",
            "",
        ]
    )
    final_values = [
        "14",
        "2023-\n06-28\n11:24:57",
        "1069557",
        "7272798\n0000001\n1760",
        "江西昌荣\n供应链有\n限公司",
        "780000",
        "",
        "6296.3",
        "超网-贷记\n转出",
        "转户",
    ]
    rows = [
        _sourced_bank_row(page_one_values, page=1, row_index=12),
        _sourced_bank_row(page_two_values, page=fragment_page, row_index=fragment_row_index),
        _sourced_bank_row(final_values, page=2, row_index=1),
    ]
    provenance = [
        RowProvenance(source_page=1, source_table_id="pt_1_0", source_row_index=12),
        RowProvenance(
            source_page=fragment_page,
            source_table_id=f"pt_{fragment_page}_0",
            source_row_index=fragment_row_index,
            is_continuation=True,
        ),
        RowProvenance(source_page=2, source_table_id="pt_2_0", source_row_index=1, is_continuation=True),
    ]
    return ParseResult(
        pages=[
            PageContent(page_number=1, texts=[TextBlock(content="上饶银行账户交易明细")]),
            PageContent(page_number=2),
        ],
        entities=DocumentEntities(document_type="bank_statement"),
        logical_tables=[
            LogicalTable(
                table_id="lt_0",
                headers=_SHANGRAO_HEADERS,
                rows=rows,
                row_count=len(rows),
                source_pages=[1, 2],
                source_physical_ids=["pt_1_0", "pt_2_0"],
                page_span=(1, 2),
                provenance=provenance,
                quality_passed=True,
                data_row_estimate=len(rows),
            )
        ],
    )


def test_cross_page_split_grid_stitches_one_business_record_with_two_page_sources():
    result = run_bank_statement_extract(
        _cross_page_bank_parse_result(),
        "上饶银行账户交易明细",
        BankStatementCommunityPlugin(),
    )

    assert len(result.records) == 2
    first, second = result.records
    expected = {
        "sequence_no": "13",
        "date": "2023-06-28",
        "timestamp": "2023-06-28T19:50:16",
        "reference": "1112052",
        "counter_account": "727279800000011760",
        "counter_party": "江西昌荣供应链有限公司",
        "direction": "income",
        "amount": 1000000.0,
        "balance": 1006296.3,
        "summary": "超网-贷记转入",
    }
    assert {key: first["normalized"][key] for key in expected} == expected
    assert first["source"]["source_page"] == 1
    assert first["source"]["page_range"] == [1, 2]
    assert {ref["page"] for ref in first["source"]["source_cell_refs"]} == {1, 2}
    assert len(first["source"]["source_refs"]) == 2
    assert second["normalized"]["sequence_no"] == "14"
    assert second["normalized"]["date"] == "2023-06-28"
    assert second["source"]["page_range"] == [2, 2]
    assert result.ctx.reconstruction is not None
    assert result.ctx.reconstruction.stitched_continuation_rows == 1
    assert result.style_meta.expected_primary_rows == 2
    assert result.style_meta.extracted_rows == 2
    assert result.style_meta.canonical_expected == 2
    assert result.style_meta.canonical_extracted == 2


def test_cross_page_stitch_does_not_merge_valid_page_two_transaction():
    result = run_bank_statement_extract(
        _cross_page_bank_parse_result(valid_page_two_row=True),
        "上饶银行账户交易明细",
        BankStatementCommunityPlugin(),
    )

    assert len(result.records) == 2
    assert result.ctx.reconstruction is not None
    assert result.ctx.reconstruction.stitched_continuation_rows == 0
    assert result.records[0]["normalized"]["date"] == "2023-06-28"
    assert result.records[0]["normalized"]["amount"] == 500.0
    assert "source_refs" not in result.records[0]["source"]


def test_cross_page_stitch_does_not_merge_repeated_page_header():
    result = run_bank_statement_extract(
        _cross_page_bank_parse_result(repeated_header_fragment=True),
        "上饶银行账户交易明细",
        BankStatementCommunityPlugin(),
    )

    assert len(result.records) == 1
    assert result.records[0]["normalized"]["sequence_no"] == "14"
    assert result.records[0]["source"]["page_range"] == [2, 2]
    assert result.ctx.reconstruction is not None
    assert result.ctx.reconstruction.stitched_continuation_rows == 0


@pytest.mark.parametrize(
    ("fragment_page", "fragment_row_index"),
    [(1, 13), (2, 3)],
)
def test_cross_page_stitch_requires_next_page_top(fragment_page: int, fragment_row_index: int):
    result = run_bank_statement_extract(
        _cross_page_bank_parse_result(
            fragment_page=fragment_page,
            fragment_row_index=fragment_row_index,
        ),
        "上饶银行账户交易明细",
        BankStatementCommunityPlugin(),
    )

    assert len(result.records) == 1
    assert result.records[0]["normalized"]["sequence_no"] == "14"
    assert result.ctx.reconstruction is not None
    assert result.ctx.reconstruction.stitched_continuation_rows == 0


def test_grid_normalization_removes_layout_wraps_only_in_typed_fields():
    normalized = normalize_record(
        {
            "序号": "474",
            "交易时间": "2023-\n01-03\n19:07:12",
            "流水号": "1408124",
            "对方账号": "7270991\n0000001\n7378",
            "对方户名": "九江冠泽\n建材贸易\n有限公司",
            "支出": "1000000\n0",
            "收入": "",
            "账户余额": "610082.5\n2",
            "摘要": "超网-贷记\n转出",
        },
        BankStatementCommunityPlugin(),
    )

    assert normalized["date"] == "2023-01-03"
    assert normalized["timestamp"] == "2023-01-03T19:07:12"
    assert normalized["amount"] == 10000000.0
    assert normalized["balance"] == 610082.52
    assert normalized["counter_account"] == "727099100000017378"
    assert normalized["counter_party"] == "九江冠泽建材贸易有限公司"


def test_cross_page_records_stay_consistent_across_community_artifacts():
    bundle = BankStatementCommunityPlugin().project_bundle(
        seal_parse_result(_cross_page_bank_parse_result()),
        file_id="001",
        document_id="doc_cross_page_bank",
    )

    assert bundle is not None
    semantic = bundle.semantic_payload()
    payload = bundle.json_payload(semantic)
    dataset = payload["datasets"][0]
    json_rows = dataset["rows"]
    csv_text = bundle.render_dataset_csvs(semantic)[dataset["csv"]]
    csv_rows = list(csv.DictReader(io.StringIO(csv_text.lstrip("\ufeff"))))
    audit_rows = list(csv.DictReader(io.StringIO(bundle.render_audit_csv(semantic).lstrip("\ufeff"))))
    markdown = bundle.render_markdown()

    record_ids = [row["record_id"] for row in json_rows]
    assert dataset["row_count"] == len(json_rows) == len(csv_rows) == 2
    assert record_ids == ["records:r000001", "records:r000002"]
    assert [row["record_id"] for row in csv_rows] == record_ids
    assert {row["record_id"] for row in audit_rows} == set(record_ids)
    assert bundle.conservation_issues(payload=payload, dataset_csvs={dataset["csv"]: csv_text}) == []

    first_json = json_rows[0]
    first_csv = csv_rows[0]
    first_audit = {row["field_key"]: row for row in audit_rows if row["record_id"] == record_ids[0]}
    assert first_json["normalized"]["date"] == "2023-06-28"
    assert first_json["normalized"]["amount"] == "1000000.0"
    assert first_json["normalized"]["balance"] == "1006296.3"
    assert first_json["source"]["page_range"] == [1, 2]
    assert first_csv["date"] == "2023-06-28"
    assert first_csv["amount"] == "1000000.0"
    assert first_csv["balance"] == "1006296.3"
    assert (first_csv["_page_start"], first_csv["_page_end"]) == ("1", "2")
    assert first_audit["amount"]["value"] == "1000000.0"
    assert first_audit["amount"]["raw"] == "1000000"
    assert first_audit["balance"]["value"] == "1006296.3"
    assert first_audit["balance"]["raw"] == "1006296.3"
    assert first_audit["date"]["value"] == "2023-06-28"
    assert "| 序号 | 交易时间 | 流水号 | 对方账号 | 对方户名 | 支出 | 收入 | 账户余额 | 摘要 | 附言 |" in markdown
    first_markdown_row = (
        "| 13 | 2023-06-28 19:50:16 | 1112052 | 727279800000011760 | 江西昌荣供应链有限公司 | "
        " | 1000000 | 1006296.3 | 超网-贷记转入 | 转户 |"
    )
    second_markdown_row = (
        "| 14 | 2023-06-28 11:24:57 | 1069557 | 727279800000011760 | 江西昌荣供应链有限公司 | "
        "780000 |  | 6296.3 | 超网-贷记转出 | 转户 |"
    )
    assert first_markdown_row in markdown
    assert second_markdown_row in markdown
    assert markdown.index(first_markdown_row) < markdown.index(second_markdown_row)
