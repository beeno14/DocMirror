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
from docmirror.plugins.bank_statement import style_registry as style_registry_module
from docmirror.plugins.bank_statement.community_plugin import BankStatementCommunityPlugin, _sanitize_bank_records
from docmirror.plugins.bank_statement.context import StyleContext, build_style_context
from docmirror.plugins.bank_statement.extract_pipeline import (
    _apply_source_reported_transaction_count,
    run_bank_statement_extract,
)
from docmirror.plugins.bank_statement.header_resolve import detect_headers
from docmirror.plugins.bank_statement.institution import match_institution, normalize_table_headers
from docmirror.plugins.bank_statement.ltro import ReconstructionMeta
from docmirror.plugins.bank_statement.row_extract import extract_logical_rows_with_provenance
from docmirror.plugins.bank_statement.style_detector import BankStyleDetector
from docmirror.plugins.bank_statement.style_registry import BankStyleParserRegistry
from docmirror.plugins.bank_statement.styles.grid_standard import (
    _normalize_direction_text,
    normalize_record,
    normalize_split_debit_credit,
    source_owned_signed_directional_amount,
)
from docmirror.plugins.bank_statement.wide_table_recovery import (
    _annotate_native_grid_matrix,
    _normalize_native_grid_table,
    _normalize_table,
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


def test_normalize_table_headers_preserves_exact_bojs_date_role() -> None:
    variant = match_institution("江苏银行")
    headers = ["序号", "摘要/附言", "币别", "交易日期", "交易类型", "交易金额", "账户余额", "对方账号", "对方户名"]

    normalized = normalize_table_headers([[headers]], variant=variant)

    assert normalized[0][0] == headers


def test_exact_counterparty_unit_label_outranks_fuzzy_neighboring_fields() -> None:
    raw = {
        "交易日期": "2023-07-26",
        "交易时间": "10:52:44",
        "借贷标志": "贷",
        "交易金额": "200.00",
        "余额": "300.00",
        "对方单位": "胡晓敏",
    }
    plugin = BankStatementCommunityPlugin()

    normalized = normalize_record(raw, plugin)
    canonical_raw = plugin._canonical_raw_values(raw, normalized)

    assert normalized["counter_party"] == "胡晓敏"
    assert canonical_raw["counter_party"] == "胡晓敏"


def test_bojs_grid_extraction_preserves_exact_source_roles_end_to_end() -> None:
    headers = ["序号", "摘要/附言", "币别", "交易日期", "交易类型", "交易金额", "账户余额", "对方账号", "对方户名"]
    row = [
        "1",
        "0WL#2023083116926046280500090200404#WL协议#百果汇",
        "人民币",
        "20230831",
        "支出",
        "31.00",
        "99.79",
        "215500690",
        "支付宝（中国）网络技术有限公司",
    ]
    ctx = StyleContext(
        tables=[[headers, row]],
        full_text="江苏银行交易明细",
        institution="江苏银行",
        page_count=1,
        reconstruction=ReconstructionMeta(source="physical_table", expected_primary_rows=1),
    )

    from docmirror.plugins.bank_statement.styles.grid_standard import extract_transactions

    raw = extract_transactions(ctx, BankStatementCommunityPlugin())[0]
    normalized = normalize_record(raw, BankStatementCommunityPlugin())

    assert {key: raw[key] for key in headers} == dict(zip(headers, row, strict=True))
    assert "交易时间" not in raw
    assert normalized["date"] == "2023-08-31"
    assert normalized["timestamp"] == ""
    assert normalized["direction"] == "expense"
    assert normalized["summary"] == "百果汇"
    assert normalized["reference"] == "2023083116926046280500090200404"


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


@pytest.mark.parametrize(("source", "expected"), [("收", "income"), ("支", "expense")])
def test_single_character_direction_flag_is_explicit(source: str, expected: str) -> None:
    assert _normalize_direction_text(source) == expected


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


def test_explicit_split_expense_wins_over_zero_income_and_summary_suffix() -> None:
    plugin = BankStatementCommunityPlugin()

    norm = normalize_split_debit_credit(
        {
            "交易时间": "20220929",
            "收入金额": "0",
            "支出金额": "2.25",
            "账户余额": "1507.17",
            "摘要": "费用外收",
        },
        plugin,
    )

    assert norm is not None
    assert norm["amount"] == 2.25
    assert norm["amount_cny"] == 2.25
    assert norm["direction"] == "expense"


def test_explicit_zero_split_transaction_is_preserved_without_invented_direction() -> None:
    plugin = BankStatementCommunityPlugin()

    norm = normalize_split_debit_credit(
        {
            "交易时间": "20220929",
            "收入金额": "0",
            "支出金额": "0.00",
            "账户余额": "1507.17",
            "摘要": "零金额业务",
        },
        plugin,
    )

    assert norm is not None
    assert norm["amount"] == 0.0
    assert norm["amount_cny"] == 0.0
    assert norm["direction"] == ""


def test_split_normalization_does_not_promote_remarks_to_counterparty() -> None:
    normalized = normalize_split_debit_credit(
        {
            "交易日期": "2025-01-02",
            "支出": "10.00",
            "收入": "",
            "余额": "90.00",
            "备注": "退奥迪A6押金",
        },
        BankStatementCommunityPlugin(),
    )

    assert normalized is not None
    assert normalized["amount"] == 10.0
    assert normalized["direction"] == "expense"
    assert normalized["counter_party"] == ""
    assert normalized["note"] == "退奥迪A6押金"


def test_explicit_counterparty_wins_and_remarks_remain_separate() -> None:
    normalized = normalize_split_debit_credit(
        {
            "交易日期": "2025-01-02",
            "支出": "10.00",
            "收入": "",
            "余额": "90.00",
            "对方户名": "甲公司",
            "Remarks": "采购付款",
        },
        BankStatementCommunityPlugin(),
    )

    assert normalized is not None
    assert normalized["counter_party"] == "甲公司"
    assert normalized["note"] == "采购付款"


def test_shrcb_exact_counter_account_name_keeps_remittance_note_separate() -> None:
    normalized = normalize_record(
        {
            "交易日期": "2023-08-08",
            "借方发生额": "14,350.00",
            "贷方发生额": "",
            "账户余额": "72,789.38",
            "对方账号": "6210000000000001",
            "对方账户名": "佛山市橡茂橡胶\n原料有限公司",
            "附言": "运费",
        },
        BankStatementCommunityPlugin(),
    )

    assert normalized["counter_party"] == "佛山市橡茂橡胶原料有限公司"
    assert normalized["remittance_note"] == "运费"
    assert normalized.get("purpose", "") == ""


def test_shrcb_source_counterparty_prevents_page_text_from_appending_remittance_note() -> None:
    headers = [
        "交易日期",
        "借方发生额",
        "贷方发生额",
        "账户余额",
        "对方账号",
        "对方账户名",
        "附言",
    ]
    raw_row = [
        "2023-08-08",
        "14,350.00",
        "",
        "72,789.38",
        "6210000000000001",
        "佛山市橡茂橡胶\n原料有限公司",
        "运费",
    ]
    page_text = "2023-08-08 14,350.00 72,789.38 6210000000000001 佛山市橡茂橡胶原料有限公司 运费"
    ctx = StyleContext(
        tables=[[headers, raw_row]],
        full_text=page_text,
        institution=None,
        page_count=1,
        parse_result=ParseResult(pages=[PageContent(page_number=1, texts=[TextBlock(content=page_text)])]),
        reconstruction=ReconstructionMeta(source="canonical_evidence_table", expected_primary_rows=1),
    )

    records, _identity = BankStyleParserRegistry().run(
        BankStyleDetector().detect(ctx),
        ctx,
        BankStatementCommunityPlugin(),
    )

    assert len(records) == 1
    assert records[0]["raw"]["对方账户名"] == "佛山市橡茂橡胶\n原料有限公司"
    assert records[0]["raw"]["附言"] == "运费"
    assert records[0]["normalized"]["counter_party"] == "佛山市橡茂橡胶原料有限公司"
    assert records[0]["normalized"]["remittance_note"] == "运费"


@pytest.mark.parametrize(
    "contaminated_party",
    [
        "贷款还款 2023-05-20 15:16:48 96000.00 96174.57 6214680161989726 彭超杰北京银行转存",
        "彭超杰 宁波银行股份有限公司 转存 第4页/共6页 交易时间 收入金额 支出金额 账户余额",
    ],
)
def test_counterparty_rejects_embedded_transaction_or_page_furniture(contaminated_party: str) -> None:
    normalized = normalize_record(
        {
            "交易时间": "2023-05-20 15:16:48",
            "收入金额": "",
            "支出金额": "96,000.00",
            "账户余额": "96,174.57",
            "对方账号": "10130256900000185",
            "对方户名": contaminated_party,
            "摘要": "贷款还款",
        },
        BankStatementCommunityPlugin(),
    )

    assert normalized["counter_party"] == ""
    assert normalized["counter_account"] == "10130256900000185"


def test_counterparty_contamination_guard_preserves_legitimate_digit_bearing_company() -> None:
    normalized = normalize_record(
        {
            "交易时间": "2023-05-20 15:16:48",
            "收入金额": "10.00",
            "支出金额": "",
            "账户余额": "100.00",
            "对方账号": "6210000000000001",
            "对方户名": "重庆2023数字科技有限公司3G事业部",
            "摘要": "往来款",
        },
        BankStatementCommunityPlugin(),
    )

    assert normalized["counter_party"] == "重庆2023数字科技有限公司3G事业部"


def test_compound_counterparty_decomposes_source_roles_without_losing_raw_cell() -> None:
    plugin = BankStatementCommunityPlugin()
    raw = {
        "交易日期": "20220105",
        "交易时间": "120114",
        "借/贷": "1",
        "交易金额": "500000.00",
        "账户余额": "591664.74",
        "交易对手信息": "丁梦琴 6228480938715161379 中国农业银行九龙支行158966653255",
    }

    normalized = normalize_record(raw, plugin)

    assert normalized["counter_party"] == "丁梦琴"
    assert normalized["counter_account"] == "6228480938715161379"
    assert normalized["counter_bank_name"] == "中国农业银行九龙支行"
    assert normalized["counter_bank_code"] == "158966653255"
    canonical_raw = plugin._canonical_raw_values(raw, normalized)
    assert canonical_raw["counter_party"] == "丁梦琴"
    assert canonical_raw["counter_account"] == "6228480938715161379"
    assert canonical_raw["counter_bank_name"] == "中国农业银行九龙支行"
    assert canonical_raw["counter_bank_code"] == "158966653255"
    assert raw["交易对手信息"] == "丁梦琴 6228480938715161379 中国农业银行九龙支行158966653255"


def test_compound_counterparty_fails_closed_when_numeric_roles_are_ambiguous() -> None:
    normalized = normalize_record(
        {
            "交易日期": "20220105",
            "借/贷": "1",
            "交易金额": "10.00",
            "账户余额": "20.00",
            "交易对手信息": "甲公司 12345678 乙银行 87654321 99999999",
        },
        BankStatementCommunityPlugin(),
    )

    assert normalized["counter_party"] == ""
    assert normalized["counter_account"] == ""
    assert normalized["counter_bank_name"] == ""


def test_compound_counterparty_maps_exact_account_only_value() -> None:
    normalized = normalize_record(
        {
            "交易时间": "2023/01/22\n05:30:27",
            "交易金额": "-2,061.15",
            "账户余额": "0.00",
            "对方户名/账号": "6226230006293805",
        },
        BankStatementCommunityPlugin(),
    )

    assert normalized["counter_party"] == ""
    assert normalized["counter_account"] == "6226230006293805"


def test_second_tier_counterparty_line_preserves_only_source_proved_roles() -> None:
    plugin = BankStatementCommunityPlugin()
    raw = {
        "日期时间": "20220911 112748",
        "日志号": "797063924",
        "短摘要": "微信支付",
        "交易金额": "-12.13",
        "本次余额": "13181.12",
        "对方账号户名/附言": "243300133 UA0911754783924513扫二维码付款",
    }

    normalized = normalize_record(raw, plugin)

    assert normalized["counter_account"] == "243300133"
    assert normalized["sub_account"] == ""
    assert normalized["counter_party"] == ""
    assert normalized["remittance_note"] == "UA0911754783924513扫二维码付款"
    canonical_raw = plugin._canonical_raw_values(raw, normalized)
    assert canonical_raw["counter_account"] == "243300133"
    assert canonical_raw["remittance_note"] == "UA0911754783924513扫二维码付款"
    assert "sub_account" not in canonical_raw
    assert raw["对方账号户名/附言"] == "243300133 UA0911754783924513扫二维码付款"


def test_second_tier_undelimited_tail_is_not_guessed_into_typed_roles() -> None:
    normalized = normalize_record(
        {
            "日期时间": "20220911 145445",
            "交易金额": "19100.00",
            "本次余额": "32281.12",
            "对方账号户名/附言": "6214673590001999710曹社兵",
        },
        BankStatementCommunityPlugin(),
    )

    assert normalized["counter_account"] == "6214673590001999710"
    assert normalized["sub_account"] == ""
    assert normalized["counter_party"] == ""
    assert normalized["remittance_note"] == ""


def test_completion_proof_rejects_whole_compound_copied_into_identifier_roles() -> None:
    raw = {"对方账号户名/附言": "243300133 UA0911754783924513扫二维码付款"}
    polluted = {
        "counter_account": "243300133 UA0911754783924513扫二维码付款",
        "sub_account": "243300133 UA0911754783924513扫二维码付款",
    }

    assert style_registry_module._row_has_semantic_anomaly(raw, polluted)


def test_labelled_note_preserves_source_and_decomposes_business_roles() -> None:
    normalized = normalize_record(
        {
            "交易日期": "2025-06-27",
            "交易发生金额": "-198.87",
            "账户余额": "144.74",
            "备注": "普通汇兑;业务编号:DEN0201;用途:往来结算款;附言:往来款;",
        },
        BankStatementCommunityPlugin(),
    )

    assert normalized["note"] == "普通汇兑;业务编号:DEN0201;用途:往来结算款;附言:往来款;"
    assert normalized["reference"] == "DEN0201"
    assert normalized["purpose"] == "往来结算款"
    assert normalized["remittance_note"] == "往来款"


def test_pab_placeholder_note_stays_raw_only_and_labelled_transfer_maps_reference() -> None:
    plugin = BankStatementCommunityPlugin()
    pab_headers = {
        "序号\nNo.": "721",
        "交易日期\nDate": "2024-01-01",
        "交易金额\nTransaction\nAmount": "-1.00",
        "余额\nBalance": "999.00",
        "交易地点\nTrading Place": "平安银行",
        "摘要\nRemark": "取现",
    }
    placeholder_raw = {**pab_headers, "备注\nNotes": "/"}
    labelled_raw = {
        **pab_headers,
        "交易日期\nDate": "2024-01-07",
        "交易金额\nTransaction\nAmount": "-500.00",
        "备注\nNotes": "平安信用卡口袋银行取现转出资金划拨: MQ20240107711839;信用卡账号:0002998000455801183",
    }

    placeholder = normalize_record(placeholder_raw, plugin)
    labelled = normalize_record(labelled_raw, plugin)
    labelled_canonical = plugin._canonical_raw_values(labelled_raw, labelled)

    assert placeholder["note"] == ""
    assert placeholder_raw["备注\nNotes"] == "/"
    assert labelled["note"] == labelled_raw["备注\nNotes"]
    assert labelled["business_system_reference"] == "MQ20240107711839"
    assert labelled_canonical["business_system_reference"] == "MQ20240107711839"
    assert labelled["counter_account"] == ""

    cross_layout = normalize_record(
        {
            "交易日期": "2024-01-07",
            "交易金额": "-500.00",
            "备注": labelled_raw["备注\nNotes"],
        },
        plugin,
    )
    assert cross_layout["note"] == labelled_raw["备注\nNotes"]
    assert cross_layout["business_system_reference"] == ""
    assert plugin._canonical_raw_values(
        {
            "交易日期": "2024-01-07",
            "交易金额": "-500.00",
            "备注": labelled_raw["备注\nNotes"],
        },
        cross_layout,
    ).get("business_system_reference") is None

    incomplete_layout_raw = dict(labelled_raw)
    incomplete_layout_raw.pop("交易地点\nTrading Place")
    incomplete_layout = normalize_record(incomplete_layout_raw, plugin)
    assert incomplete_layout["note"] == labelled_raw["备注\nNotes"]
    assert incomplete_layout["business_system_reference"] == ""
    assert plugin._canonical_raw_values(incomplete_layout_raw, incomplete_layout).get(
        "business_system_reference"
    ) is None

    explicit_reference_raw = {**labelled_raw, "业务系统参考号": "SOURCE-REF-1"}
    explicit_reference = normalize_record(explicit_reference_raw, plugin)
    assert explicit_reference["business_system_reference"] == "SOURCE-REF-1"
    assert (
        plugin._canonical_raw_values(explicit_reference_raw, explicit_reference)["business_system_reference"]
        == "SOURCE-REF-1"
    )

    slash_serialized_raw = {
        "序号/No.": "721",
        "交易日期/Date": "2024-01-07",
        "交易金额/Transaction Amount": "-500.00",
        "余额/Balance": "999.00",
        "交易地点/Trading Place": "平安银行",
        "摘要/Remark": "取现",
        "备注/Notes": labelled_raw["备注\nNotes"],
    }
    slash_serialized = normalize_record(slash_serialized_raw, plugin)
    assert slash_serialized["business_system_reference"] == "MQ20240107711839"
    assert (
        plugin._canonical_raw_values(slash_serialized_raw, slash_serialized)["business_system_reference"]
        == "MQ20240107711839"
    )

    malformed_note_raw = {**labelled_raw, "备注\nNotes": "资金划拨 MQ20240107711839"}
    malformed_note = normalize_record(malformed_note_raw, plugin)
    assert malformed_note["business_system_reference"] == ""
    assert plugin._canonical_raw_values(malformed_note_raw, malformed_note).get(
        "business_system_reference"
    ) is None


@pytest.mark.parametrize(
    ("compound", "code", "reference", "transaction_name", "summary"),
    [
        ("0WL#2023083116926046280500090200404#WL协议#百果汇", "0WL", "2023083116926046280500090200404", "WL协议", "百果汇"),
        ("0WL#20230531131922923977443S0110301#WL退款#退款", "0WL", "20230531131922923977443S0110301", "WL退款", "退款"),
        ("1银联#520032#银联贷记#财付通支付科技有限公司/银联入账,微信零钱提现", "1银联", "520032", "银联贷记", "财付通支付科技有限公司/银联入账,微信零钱提现"),
    ],
)
def test_bojs_compound_memo_maps_distinct_source_roles(
    compound: str,
    code: str,
    reference: str,
    transaction_name: str,
    summary: str,
) -> None:
    plugin = BankStatementCommunityPlugin()
    raw = {
        "序号": "1",
        "摘要/附言": compound,
        "币别": "人民币",
        "交易日期": "20230831",
        "交易类型": "支出",
        "交易金额": "31.00",
        "账户余额": "99.79",
        "对方账号": "215500690",
        "对方户名": "支付宝（中国）网络技术有限公司",
    }

    normalized = normalize_record(raw, plugin)
    canonical = plugin._canonical_raw_values(raw, normalized)

    assert normalized["date"] == "2023-08-31"
    assert normalized["timestamp"] == ""
    assert normalized["direction"] == "expense"
    assert normalized["currency"] == "人民币"
    assert normalized["transaction_code"] == code
    assert normalized["reference"] == reference
    assert normalized["transaction_name"] == transaction_name
    assert normalized["summary"] == summary
    assert normalized["business_detail"] == compound
    assert canonical["date"] == "20230831"
    assert canonical["direction"] == "支出"
    assert canonical["currency"] == "人民币"
    assert canonical["business_detail"] == compound
    assert "timestamp" not in canonical


@pytest.mark.parametrize("compound", ["0WL#123456#WL协议", "0WL#123456#WL协议#memo#extra", "UNKNOWN#123456#kind#memo"])
def test_bojs_compound_memo_fails_closed_for_unproven_grammar(compound: str) -> None:
    raw = {
        "序号": "1",
        "摘要/附言": compound,
        "币别": "人民币",
        "交易日期": "20230831",
        "交易类型": "收入",
        "交易金额": "1.00",
        "账户余额": "1.00",
        "对方账号": "",
        "对方户名": "",
    }

    normalized = normalize_record(raw, BankStatementCommunityPlugin())

    assert normalized["summary"] == compound
    assert normalized["business_detail"] == ""
    assert normalized["reference"] == ""
    assert normalized["transaction_code"] == ""
    assert normalized["transaction_name"] == ""


def test_psbc_subaccount_is_a_distinct_source_role() -> None:
    plugin = BankStatementCommunityPlugin()
    raw = {
        "交易日期": "20240101",
        "子账号": "0001",
        "储种": "活期",
        "币种": "人民币",
        "钞汇": "钞",
        "交易金额": "-10.00",
        "账户余额": "90.00",
        "对方户名": "甲公司",
        "对方账号": "123456789",
        "摘要": "转账",
        "交易渠道": "网银",
    }

    normalized = normalize_record(raw, plugin)
    canonical = plugin._canonical_raw_values(raw, normalized)

    assert normalized["sub_account"] == "0001"
    assert normalized["own_account"] == ""
    assert normalized["deposit_type"] == "活期"
    assert normalized["counter_account"] == "123456789"
    assert normalized["timestamp"] == ""
    assert canonical["sub_account"] == "0001"


def test_single_date_summary_layout_does_not_duplicate_posting_date() -> None:
    normalized = normalize_record(
        {
            "序号": "1",
            "记账日期": "2024-01-01",
            "交易金额": "-1.00",
            "账户余额": "9.00",
            "摘要描述": "手续费",
            "对方户名": "甲公司",
        },
        BankStatementCommunityPlugin(),
    )

    assert normalized["date"] == "2024-01-01"
    assert normalized["posting_date"] == ""
    assert normalized["timestamp"] == ""


def test_counterparty_null_token_is_source_null_without_erasing_source() -> None:
    raw = {
        "交易日期": "2022-06-21",
        "交易金额": "53.14",
        "对方户名": "NULL",
    }
    record = {
        "raw": dict(raw),
        "canonical_raw": {"counter_party": "NULL"},
        "normalized": {"counter_party": "NULL", "counter_account": ""},
    }

    sanitized = _sanitize_bank_records([record])[0]

    assert sanitized["raw"] == raw
    assert sanitized["canonical_raw"] == {"counter_party": "NULL"}
    assert sanitized["normalized"]["counter_party"] == ""
    assert sanitized["normalized"]["counterparty_status"] == "source_null"


def test_cscb_exact_compound_memo_and_number_map_to_distinct_roles() -> None:
    raw = {
        "交易日期": "20221202",
        "交易金额": "-2,000.00",
        "账户余额": "451,517.76",
        "对方户名": "贺哲尧",
        "对方账号": "6214857219035997",
        "摘要/备注": "转账/其他合法款项-退奥迪A6押金",
        "编号": "99018150246\n4669",
    }

    normalized = normalize_record(raw, BankStatementCommunityPlugin())
    canonical = BankStatementCommunityPlugin()._canonical_raw_values(raw, normalized)

    assert normalized["counter_party"] == "贺哲尧"
    assert normalized["counter_account"] == "6214857219035997"
    assert normalized["summary"] == "转账/其他合法款项-退奥迪A6押金"
    assert normalized["note"] == ""
    assert normalized["reference"] == "990181502464669"
    assert normalized["timestamp"] == ""
    assert canonical["summary"] == "转账/其他合法款项-退奥迪A6押金"
    assert canonical["reference"] == "99018150246\n4669"
    assert "note" not in canonical


@pytest.mark.parametrize("missing_header", ["账户余额", "对方户名", "编号"])
def test_cscb_mapping_requires_the_complete_source_layout(missing_header: str) -> None:
    raw = {
        "交易日期": "20221202",
        "交易金额": "-2,000.00",
        "账户余额": "451,517.76",
        "对方户名": "贺哲尧",
        "对方账号": "6214857219035997",
        "摘要/备注": "CUSTOM/free/form",
        "编号": "99018150246\n4669",
    }
    raw.pop(missing_header)

    normalized = normalize_record(raw, BankStatementCommunityPlugin())
    canonical = BankStatementCommunityPlugin()._canonical_raw_values(raw, normalized)

    # Without the full source contract, the generic mapping must not invent
    # CSCB-specific role claims from globally ambiguous compound headers.
    assert normalized.get("reference", "") == ""
    assert canonical.get("reference", "") == ""
    if missing_header != "编号":
        assert normalized.get("note", "") == "CUSTOM/free/form"


def test_exact_electronic_receipt_number_maps_to_reference_not_sequence() -> None:
    normalized = normalize_record(
        {
            "交易时间": "2022-07-06 14:02:35",
            "电子回单编号": "22187000001",
            "交易类型": "转账",
            "交易金额": "27,500.00",
            "收/支": "收入",
            "余额": "250,324.09",
        },
        BankStatementCommunityPlugin(),
    )

    assert normalized["reference"] == "22187000001"
    assert normalized["sequence_no"] == ""
    assert normalized["transaction_name"] == "转账"


def test_embedded_unique_business_identifier_is_promoted_without_truncating_summary() -> None:
    summary = "汇兑-实时代收业务唯一编号202308150008"
    normalized = normalize_record(
        {
            "序号": "1",
            "记账日期": "2023-08-15",
            "交易金额": "-2,500.00",
            "账户余额": "13,804.01",
            "摘要描述": summary,
            "对方户名": "上海市公积金管理中心(房改资金)",
        },
        BankStatementCommunityPlugin(),
    )

    assert normalized["summary"] == summary
    assert normalized["reference"] == "202308150008"


def test_explicit_value_date_alias_stays_distinct_from_transaction_date() -> None:
    normalized = normalize_record(
        {
            "交易日期": "2023-06-01",
            "交易时间": "10:57:00",
            "起息日期": "2023-06-02",
            "支出金额": "503.00",
            "收入金额": "0.00",
            "余额": "85,623.04",
        },
        BankStatementCommunityPlugin(),
    )

    assert normalized["date"] == "2023-06-01"
    assert normalized["value_date"] == "2023-06-02"


def test_boc_compound_business_layout_preserves_and_decomposes_roles_once() -> None:
    plugin = BankStatementCommunityPlugin()
    raw = {
        "序号": "3",
        "记账日": "251215",
        "起息日": "251216",
        "交易类型": "代发划转",
        "凭证": "",
        "凭证号码/业务编号/用途/摘要": ("A2755427C1202512155G001001/2025年11月工资/OBSS003568953795GIRO000000000000"),
        "借方发生额": "61,073.80",
        "贷方发生额": "",
        "余额": "38,069.41",
        "机构/柜员/流水": "12313/9880105/267174050",
        "备注": "重庆正大华日软件有限公司/重庆农村商业银行",
    }

    normalized = normalize_record(raw, plugin)
    canonical = plugin._canonical_raw_values(raw, normalized)

    assert normalized["date"] == "2025-12-15"
    assert normalized["value_date"] == "2025-12-16"
    assert normalized["posting_date"] == ""
    assert normalized["transaction_name"] == "代发划转"
    assert normalized["transaction_institution"] == "12313"
    assert normalized["teller_id"] == "9880105"
    assert normalized["bank_serial"] == "267174050"
    assert normalized["reference"] == "A2755427C1202512155G001001"
    assert normalized["purpose"] == "2025年11月工资"
    assert normalized["business_system_reference"] == "OBSS003568953795GIRO000000000000"
    assert normalized["summary"] == ""
    assert normalized["counter_party"] == "重庆正大华日软件有限公司"
    assert normalized["counter_bank_name"] == "重庆农村商业银行"
    assert canonical["business_detail"] == raw["凭证号码/业务编号/用途/摘要"]
    assert canonical["value_date"] == "251216"
    assert "summary" not in canonical
    assert "posting_date" not in canonical


def test_boc_plain_detail_maps_to_purpose_without_inventing_reference() -> None:
    raw = {
        "序号": "7",
        "记账日": "251226",
        "起息日": "251226",
        "交易类型": "收费",
        "凭证": "",
        "凭证号码/业务编号/用途/摘要": "对公跨行转账汇款手续费",
        "借方发生额": "4.50",
        "贷方发生额": "",
        "余额": "25,180.12",
        "机构/柜员/流水": "12313/9880105/238695600",
        "备注": "国家金库长沙县支库（346）",
    }

    normalized = normalize_record(raw, BankStatementCommunityPlugin())

    assert normalized["purpose"] == "对公跨行转账汇款手续费"
    assert normalized["reference"] == ""
    assert normalized["counter_party"] == "国家金库长沙县支库（346）"
    assert normalized["counter_bank_name"] == ""


def test_boc_reference_with_embedded_tax_context_is_not_invented_as_purpose() -> None:
    raw = {
        "序号": "1",
        "记账日": "251209",
        "起息日": "251209",
        "交易类型": "实时缴税",
        "凭证": "",
        "凭证号码/业务编号/用途/摘要": (
            "19077378/2025120964867670 重庆正大华日软件有限公司长沙分公司 91430100MADRF3UN4A 国家税务"
        ),
        "借方发生额": "113.76",
        "贷方发生额": "",
        "余额": "31,622.44",
        "机构/柜员/流水": "12313/9880800/256775918",
        "备注": "国家金库长沙县支库（346）",
    }

    normalized = normalize_record(raw, BankStatementCommunityPlugin())

    assert normalized["reference"] == "19077378"
    assert normalized["purpose"] == ""
    assert normalized["business_context"].startswith("2025120964867670")
    assert normalized["counter_party"] == "国家金库长沙县支库（346）"


def test_boc_slash_note_requires_a_bank_shaped_right_side() -> None:
    raw = {
        "序号": "2",
        "记账日": "251210",
        "起息日": "251210",
        "交易类型": "网上支付",
        "凭证": "",
        "凭证号码/业务编号/用途/摘要": "3146530000112025121142301261/往来结算款",
        "借方发生额": "10.00",
        "贷方发生额": "",
        "余额": "20.00",
        "机构/柜员/流水": "12313/9880105/1",
        "备注": "甲公司/项目一部",
    }

    normalized = normalize_record(raw, BankStatementCommunityPlugin())

    assert normalized["note"] == "甲公司/项目一部"
    assert normalized["counter_party"] == "甲公司"
    assert normalized["counter_bank_name"] == ""


def test_boc_truncated_bank_prefix_completes_only_from_unique_same_row_source() -> None:
    plugin = BankStatementCommunityPlugin()
    raw = {
        "序号": "1",
        "记账日": "220401",
        "起息日": "220401",
        "交易类型": "网上支付",
        "凭证": "",
        "凭证号码/业务编号/用途/摘要": "3235840008882022040194831167/贴现款",
        "借方发生额": "",
        "贷方发生额": "49,234.67",
        "余额": "56,020.44",
        "机构/柜员/流水": "06257/9880809/43627150",
        "备注": "深圳前海微众银行股份有限公司/深圳前海微众银行股份有",
    }

    normalized = normalize_record(raw, plugin)
    canonical = plugin._canonical_raw_values(raw, normalized)

    assert normalized["counter_bank_name"] == "深圳前海微众银行股份有限公司"
    assert canonical["counter_bank_name"] == "深圳前海微众银行股份有"


def test_boc_unsupported_truncated_bank_prefix_is_preserved() -> None:
    raw = {
        "序号": "1",
        "记账日": "220401",
        "起息日": "220401",
        "交易类型": "网上支付",
        "凭证": "",
        "凭证号码/业务编号/用途/摘要": "1041000000042022040119694857/采购款",
        "借方发生额": "150,000.00",
        "贷方发生额": "",
        "余额": "1,222,783.07",
        "机构/柜员/流水": "06257/9880105/106234268",
        "备注": "镇江世泽机电设备有限公司/招商银行股份有限公司天",
    }

    normalized = normalize_record(raw, BankStatementCommunityPlugin())

    assert normalized["counter_bank_name"] == "招商银行股份有限公司天"


def test_directional_payer_payee_mapping_uses_only_the_counterparty_side() -> None:
    plugin = BankStatementCommunityPlugin()
    income = normalize_record(
        {
            "交易日期\n交易日期": "2025-10-28",
            "付款账号\n付款账号": "651204680300015",
            "付款账户名\n付款账户名": "重庆恒腾科技有限公司",
            "收入\n收入": "3000000.00",
            "收款账号\n收款账号": "100102029005622957",
            "收款账户名\n收款账户名": "重庆数宜信信用管理有限公司",
            "支出\n支出": "",
            "余额\n余额": "3036962.83",
        },
        plugin,
    )
    expense = normalize_record(
        {
            "交易日期\n交易日期": "2025-10-28",
            "付款账号\n付款账号": "100102029005622957",
            "付款账户名\n付款账户名": "重庆数宜信信用管理有限公司",
            "收入\n收入": "",
            "收款账号\n收款账号": "01041560012000235",
            "收款账户名\n收款账户名": "重庆正大能科科技有限公司",
            "支出\n支出": "3418450.00",
            "余额\n余额": "21507.83",
        },
        plugin,
    )
    fee = normalize_record(
        {
            "交易日期\n交易日期": "2025-10-28",
            "付款账号\n付款账号": "100102029005622957",
            "付款账户名\n付款账户名": "重庆数宜信信用管理有限公司",
            "收入\n收入": "",
            "收款账号\n收款账号": "",
            "收款账户名\n收款账户名": "",
            "支出\n支出": "5.00",
            "余额\n余额": "3436957.83",
        },
        plugin,
    )

    assert (income["counter_account"], income["counter_party"]) == (
        "651204680300015",
        "重庆恒腾科技有限公司",
    )
    assert (expense["counter_account"], expense["counter_party"]) == (
        "01041560012000235",
        "重庆正大能科科技有限公司",
    )
    assert fee["counter_account"] == ""
    assert fee["counter_party"] == ""


def test_slash_delimited_counter_account_and_party_are_distinct() -> None:
    normalized = normalize_record(
        {
            "交易日期": "20240102",
            "交易金额": "80000.00",
            "账户余额": "102214.76",
            "对方账号与户名": "35001677107*****5957/顺***融竹木有限公司",
            "_document_scope_text": "中国建设银行个人活期账户收入交易明细",
        },
        BankStatementCommunityPlugin(),
    )

    assert normalized["direction"] == "income"
    assert normalized["counter_account"] == "35001677107*****5957"
    assert normalized["counter_party"] == "顺***融竹木有限公司"


def test_summary_direction_with_signed_amount_keeps_source_sign_only_in_raw() -> None:
    raw = {
        "交易时间": "20240102123045",
        "短摘要": "转支",
        "交易金额": "-13,900.00",
        "本次余额": "18,381.12",
    }

    normalized = normalize_record(raw, BankStatementCommunityPlugin())

    assert normalized["direction"] == "expense"
    assert normalized["amount"] == 13_900.0
    assert raw["交易金额"] == "-13,900.00"


def test_signed_amount_prefix_survives_trailing_page_overlay_text() -> None:
    raw = {
        "交易时间": "20240102123045",
        "交易金额": "-500.00https://secure.example/statement",
        "余额": "2,533.08",
        "交易类型": "转账汇款",
    }

    normalized = normalize_record(raw, BankStatementCommunityPlugin())

    assert normalized["direction"] == "expense"
    assert normalized["amount"] == 500.0
    assert raw["交易金额"] == "-500.00https://secure.example/statement"


def test_reverse_slash_counterparty_decomposes_without_guessing_fee_code() -> None:
    plugin = BankStatementCommunityPlugin()
    raw = {
        "交易时间": "2025/01/0316:18:35",
        "借方发生额": "15.00",
        "贷方发生额": "0.00",
        "账户余额": "363,693.02",
        "流水号": "554202501030\n08247128705",
        "对方户名/账号": "暂收款/190700000003371002",
        "对方行名": "中华人民共和国国家金库江门市中心支库",
    }

    normalized = normalize_record(raw, plugin)
    canonical_raw = plugin._canonical_raw_values(raw, normalized)

    assert normalized["date"] == "2025-01-03"
    assert normalized["timestamp"] == "2025-01-03T16:18:35"
    assert normalized["balance"] == 363693.02
    assert normalized["reference"] == "55420250103008247128705"
    assert normalized["counter_party"] == "暂收款"
    assert normalized["counter_account"] == "190700000003371002"
    assert normalized["counter_bank_name"] == "中华人民共和国国家金库江门市中心支库"
    assert canonical_raw["counter_party"] == "暂收款"
    assert canonical_raw["counter_account"] == "190700000003371002"

    ambiguous = normalize_record(
        {
            "交易时间": "2025/01/0316:18:35",
            "借方发生额": "7.50",
            "贷方发生额": "0.00",
            "账户余额": "363,685.52",
            "对方户名/账号": "4501-C070470",
        },
        plugin,
    )
    assert ambiguous["counter_party"] == ""
    assert ambiguous["counter_account"] == ""


def test_date_only_source_does_not_claim_canonical_raw_timestamp() -> None:
    plugin = BankStatementCommunityPlugin()
    raw = {"交易日期": "20221201", "交易金额": "10.00", "收/支": "收入"}
    normalized = normalize_record(raw, plugin)

    canonical_raw = plugin._canonical_raw_values(raw, normalized)

    assert canonical_raw["date"] == "20221201"
    assert "timestamp" not in canonical_raw


def test_combined_counter_account_and_bank_are_decomposed() -> None:
    normalized = normalize_record(
        {
            "交易日期": "2022-01-15",
            "支/收": "支",
            "交易金额": "-133.80",
            "账户余额": "0.00",
            "对方户名": "兴业消费金融股份公司",
            "对方账户/对方银行": "129970100100190487 兴业银行厦门科技支行",
        },
        BankStatementCommunityPlugin(),
    )

    assert normalized["counter_account"] == "129970100100190487"
    assert normalized["counter_bank_name"] == "兴业银行厦门科技支行"


@pytest.mark.parametrize(
    ("party", "account_bank", "expected_party", "expected_account", "expected_bank"),
    [
        ("宋鹏", "6230523170029107378中国农业银行", "宋鹏", "6230523170029107378", "中国农业银行"),
        (
            "郑萍杰",
            "AW8BAGYF12345enWGZfG1财付通支付科技有限公司",
            "郑萍杰",
            "AW8BAGYF12345enWGZfG1",
            "财付通支付科技有限公司",
        ),
        (
            "微信转账1000050001",
            "财付通支付科技有限公司",
            "微信转账",
            "1000050001",
            "财付通支付科技有限公司",
        ),
    ],
)
def test_exact_counterparty_and_account_bank_columns_are_fully_decomposed(
    party: str,
    account_bank: str,
    expected_party: str,
    expected_account: str,
    expected_bank: str,
) -> None:
    plugin = BankStatementCommunityPlugin()
    raw = {
        "交易日期": "2024-02-20",
        "支/收": "收",
        "交易金额": "100.00",
        "账户余额": "108.95",
        "对方户名": party,
        "对方账户/对方银行": account_bank,
    }

    normalized = normalize_record(raw, plugin)
    canonical_raw = plugin._canonical_raw_values(raw, normalized)

    assert raw["对方户名"] == party
    assert raw["对方账户/对方银行"] == account_bank
    assert normalized["counter_party"] == expected_party
    assert normalized["counter_account"] == expected_account
    assert normalized["counter_bank_name"] == expected_bank
    assert canonical_raw["counter_party"] == expected_party
    assert canonical_raw["counter_account"] == expected_account
    assert canonical_raw["counter_bank_name"] == expected_bank


def test_exact_source_roles_override_reconstructed_headers_without_reverting_core_cells() -> None:
    raw = {
        "交易日期": "2023-04-15",
        "摘要": "汇款汇入",
        "收/支": "支出",
        "交易金额": "50,000.00",
        "余额": "150,050.05",
        "对方户名": "样例公司",
        "对方账号": "A84x9Z00231Q样例银行",
        "_source_raw": {
            "交易日期": "2023-04-15",
            "摘要": "汇款汇入",
            "支/收": "收",
            "交易金额": "50,000.00",
            "账户余额": "150,050.05",
            "对方户名": "样例公司",
            "对方账户/对方银行": "A84x9Z00231Q样例银行",
        },
    }

    normalized = normalize_record(raw, BankStatementCommunityPlugin())

    assert normalized["date"] == "2023-04-15"
    assert normalized["amount"] == 50_000.0
    assert normalized["balance"] == 150_050.05
    assert normalized["direction"] == "income"
    assert normalized["counter_party"] == "样例公司"
    assert normalized["counter_account"] == "A84x9Z00231Q"
    assert normalized["counter_bank_name"] == "样例银行"
    assert raw["收/支"] == "支出"
    assert raw["对方账号"] == "A84x9Z00231Q样例银行"


def test_blank_exact_source_direction_does_not_erase_source_backed_reconstruction() -> None:
    normalized = normalize_record(
        {
            "交易日期": "2023-05-25",
            "摘要": "快捷支付支",
            "收/支": "支出",
            "交易金额": "-2,000.00",
            "余额": "7,424.19",
            "对方户名": "微信转账1000050001",
            "对方账号": "财付通支付科技有限公司",
            "_source_raw": {
                "交易日期": "2023-05-25",
                "摘要": "快捷支付支",
                "支/收": "",
                "交易金额": "-2,000.00",
                "账户余额": "7,424.19",
                "对方户名": "微信转账1000050001",
                "对方账户/对方银行": "财付通支付科技有限公司",
            },
        },
        BankStatementCommunityPlugin(),
    )

    assert normalized["direction"] == "expense"
    assert normalized["counter_party"] == "微信转账"
    assert normalized["counter_account"] == "1000050001"
    assert normalized["counter_bank_name"] == "财付通支付科技有限公司"


def test_income_scope_requires_exact_statement_title() -> None:
    raw = {
        "交易日期": "20240102",
        "交易金额": "100.00",
        "账户余额": "100.00",
        "_document_scope_text": "中国建设银行个人活期账户交易明细",
    }

    assert normalize_record(raw, BankStatementCommunityPlugin())["direction"] == ""


def test_finalization_preserves_exact_private_scope_when_parse_text_lacks_title() -> None:
    from docmirror.plugins.bank_statement.styles.grid_standard import _finalize_transactions

    rows = _finalize_transactions(
        [
            {
                "交易日期": "20240102",
                "交易金额": "100.00",
                "账户余额": "100.00",
                "_document_scope_text": "中国建设银行个人活期账户收入交易明细",
            }
        ],
        full_text="序号 摘要 币别 钞汇 交易日期 交易金额 账户余额",
    )

    assert rows[0]["_document_scope_text"] == "中国建设银行个人活期账户收入交易明细"
    assert normalize_record(rows[0], BankStatementCommunityPlugin())["direction"] == "income"


def test_income_scope_is_not_rewritten_by_discontinuous_balance_chain() -> None:
    from docmirror.plugins.bank_statement.canonical import records_from_raw_transactions
    from docmirror.plugins.bank_statement.styles.grid_standard import refine_missing_directions_from_balance_chain

    title = "中国建设银行个人活期账户收入交易明细"
    transactions = [
        {
            "序号": "136",
            "交易日期": "2024-03-01",
            "摘要": "往来款",
            "交易金额": "100000.00",
            "账户余额": "108801.16",
            "_document_scope_text": title,
        },
        {
            "序号": "137",
            "交易日期": "2024-03-01",
            "摘要": "往来款",
            "交易金额": "100000.00",
            "账户余额": "208801.16",
            "_document_scope_text": title,
        },
        {
            "序号": "138",
            "交易日期": "2024-03-01",
            "摘要": "支付机构提现",
            "交易金额": "100000.00",
            "账户余额": "108801.16",
            "_document_scope_text": title,
        },
    ]
    plugin = BankStatementCommunityPlugin()
    records = records_from_raw_transactions(
        transactions,
        normalize_fn=lambda raw: normalize_record(raw, plugin),
        style_id="grid_standard",
    )

    refine_missing_directions_from_balance_chain(records)

    assert [record["normalized"]["direction"] for record in records] == ["income", "income", "income"]


def test_explicit_split_amount_direction_is_not_rewritten_by_same_time_balance_order() -> None:
    from docmirror.plugins.bank_statement.styles.grid_standard import refine_missing_directions_from_balance_chain

    records = [
        {
            "raw": {"交易日期": "2024-01-01", "借方发生额": "", "贷方发生额": "3,000,000.00"},
            "normalized": {
                "date": "2024-01-01",
                "timestamp": "2024-01-01T10:00:00",
                "amount": 3_000_000.0,
                "direction": "income",
                "balance": 2_000_000.0,
            },
        },
        {
            "raw": {"交易日期": "2024-01-01", "借方发生额": "1,000,000.00", "贷方发生额": ""},
            "normalized": {
                "date": "2024-01-01",
                "timestamp": "2024-01-01T10:00:00",
                "amount": 1_000_000.0,
                "direction": "expense",
                "balance": 1_000_000.0,
            },
        },
    ]

    refine_missing_directions_from_balance_chain(records)

    assert [record["normalized"]["direction"] for record in records] == ["income", "expense"]


def test_explicit_direction_label_is_not_rewritten_for_negative_reversal() -> None:
    from docmirror.plugins.bank_statement.styles.grid_standard import refine_missing_directions_from_balance_chain

    records = [
        {
            "raw": {"序号": "1", "借/贷": "借方", "交易金额": "100.00"},
            "normalized": {
                "sequence_no": "1",
                "date": "2024-01-01",
                "amount": 100.0,
                "direction": "expense",
                "balance": 100.0,
            },
        },
        {
            "raw": {"序号": "2", "借/贷": "借方", "交易金额": "-100.00"},
            "normalized": {
                "sequence_no": "2",
                "date": "2024-01-02",
                "amount": 100.0,
                "direction": "expense",
                "balance": 200.0,
            },
        },
    ]

    refine_missing_directions_from_balance_chain(records)

    assert records[1]["normalized"]["direction"] == "expense"


@pytest.mark.parametrize(
    ("raw", "canonical_raw", "expected"),
    [
        (
            {"借/贷": "借方", "交易金额": "-100.00"},
            {"direction": "借方", "amount": "-100.00"},
            ("expense", -100.0),
        ),
        (
            {"收入": "", "支出": "-3,260.00"},
            {"direction": "expense", "amount": "-3,260.00"},
            ("expense", -3260.0),
        ),
    ],
    ids=["dedicated-direction", "split-columns"],
)
def test_signed_reversal_aggregate_fact_requires_source_owned_direction(
    raw: dict[str, str],
    canonical_raw: dict[str, str],
    expected: tuple[str, float],
) -> None:
    normalized = {"direction": "expense", "amount": abs(expected[1])}

    assert source_owned_signed_directional_amount(raw, normalized, canonical_raw) == expected


@pytest.mark.parametrize(
    ("raw", "canonical_raw"),
    [
        (
            {"交易金额": "-100.00"},
            {"amount": "-100.00"},
        ),
        (
            {"收入": "5.00", "支出": "-100.00"},
            {"direction": "expense", "amount": "-100.00"},
        ),
        (
            {"借/贷": "借方", "交易金额": "-100.00"},
            {"direction": "借方", "amount": "-99.00"},
        ),
        (
            {"借/贷": "借方", "交易金额": "-100.00"},
            {"direction": "贷方", "amount": "-100.00"},
        ),
        (
            {"借/贷": "借方", "交易金额": "-100.00"},
            {"direction": "借方", "amount": "-100.0"},
        ),
        (
            {"借/贷": "借方", "交易金额": "-100.00附言"},
            {"direction": "借方", "amount": "-100.00附言"},
        ),
    ],
    ids=[
        "sign-only",
        "both-split-sides",
        "canonical-amount-mismatch",
        "canonical-direction-mismatch",
        "raw-canonical-text-mismatch",
        "trailing-source-junk",
    ],
)
def test_signed_reversal_aggregate_fact_fails_closed_without_complete_provenance(
    raw: dict[str, str],
    canonical_raw: dict[str, str],
) -> None:
    normalized = {"direction": "expense", "amount": 100.0}

    assert source_owned_signed_directional_amount(raw, normalized, canonical_raw) is None


def test_missing_source_direction_still_allows_unique_balance_inference() -> None:
    from docmirror.plugins.bank_statement.styles.grid_standard import refine_missing_directions_from_balance_chain

    records = [
        {
            "raw": {"序号": "1", "交易金额": "100.00"},
            "normalized": {
                "sequence_no": "1",
                "date": "2024-01-01",
                "amount": 100.0,
                "direction": "income",
                "balance": 100.0,
            },
        },
        {
            "raw": {"序号": "2", "交易金额": "20.00"},
            "normalized": {
                "sequence_no": "2",
                "date": "2024-01-02",
                "amount": 20.0,
                "direction": "expense",
                "balance": 120.0,
            },
        },
    ]

    refine_missing_directions_from_balance_chain(records)

    assert records[1]["normalized"]["direction"] == "income"


def test_numeric_dedicated_debit_credit_flag_is_respected() -> None:
    plugin = BankStatementCommunityPlugin()
    income = normalize_record(
        {"交易日期": "2025-01-02", "借/贷": "1", "交易金额": "10.00", "余额": "110.00"},
        plugin,
    )
    expense = normalize_record(
        {"交易日期": "2025-01-03", "借/贷": "0", "交易金额": "5.00", "余额": "105.00"},
        plugin,
    )

    assert income["direction"] == "income"
    assert expense["direction"] == "expense"


def test_sanitizer_preserves_source_layers_when_normalized_amount_differs() -> None:
    records = _sanitize_bank_records(
        [
            {
                "raw": {"收入金额": "0", "支出金额": "2.25"},
                "normalized": {"amount": 2.25, "amount_cny": 2.25, "direction": "expense"},
                "canonical_raw": {"amount": "0", "amount_cny": "0"},
            }
        ]
    )

    assert records[0]["raw"] == {"收入金额": "0", "支出金额": "2.25"}
    assert records[0]["canonical_raw"] == {"amount": "0", "amount_cny": "0"}


def test_normalize_transaction_location_as_distinct_business_field():
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
    assert norm["channel"] == ""
    assert norm["transaction_location"] == "支付平台"


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
    assert norm["reference"] == "55420250100824712870"


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
    assert records[0]["normalized"]["summary"] == ""
    assert records[0]["normalized"]["note"] == "财付通支付"
    assert records[0]["raw"]["备注"] == "财付通支\n付"
    assert records[0]["raw"]["交易机构"] == "101001"
    assert "交易机构" not in records[0]["normalized"]
    assert "交易机构" not in records[0]["canonical_raw"]
    assert [record["source"]["source_page"] for record in records] == [1, 2]
    assert [record["source"]["table_id"] for record in records] == ["pt_1_0", "pt_2_0"]
    assert all(record["source"]["source_cell_refs"] for record in records)
    assert all(record["source"]["evidence_ids"] for record in records)


def test_native_split_grid_keeps_wrapped_cells_in_their_physical_rows() -> None:
    matrix = _normalize_table(
        [
            ["交易时间", "收入金额", "支出金额", "账户余额", "对方账号", "对方户名", "对方开户行", "摘要"],
            [
                "20221102",
                "5647.20",
                "0",
                "5729.46",
                "121940438010301",
                "上海赫程国际旅行社有限公司南通分\n公司",
                "中国农业银行股份\n有限公司镇江京口\n支行",
                "转存",
            ],
            ["20221102", "0", "5000.00", "729.46", "6212261104003648184", "徐双根", "", "转取"],
        ]
    )

    annotated = _annotate_native_grid_matrix(
        matrix,
        page_number=6,
        table_index=0,
        money_hints={},
        row_bboxes=[
            (0.0, 0.0, 100.0, 10.0),
            (0.0, 10.0, 100.0, 20.0),
            (0.0, 20.0, 100.0, 30.0),
        ],
    )

    assert annotated[0][-4:] == ["_source_page", "_source_bbox", "_source_table_id", "_source_row_index"]
    assert annotated[1][-4:] == ["6", "0.000,10.000,100.000,20.000", "native:p6:t0", "1"]
    assert annotated[2][-4:] == ["6", "0.000,20.000,100.000,30.000", "native:p6:t0", "2"]

    first_raw = dict(zip(annotated[0], annotated[1]))
    second_raw = dict(zip(annotated[0], annotated[2]))
    plugin = BankStatementCommunityPlugin()
    first = normalize_split_debit_credit(first_raw, plugin)
    second = normalize_split_debit_credit(second_raw, plugin)

    assert first is not None
    assert second is not None
    assert first["counter_party"] == "上海赫程国际旅行社有限公司南通分公司"
    assert first["counter_bank_name"] == "中国农业银行股份有限公司镇江京口支行"
    assert second["counter_party"] == "徐双根"
    assert second.get("counter_bank_name", "") == ""


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
    assert [record["raw"] for record in sanitized] == [record["raw"] for record in records]


def test_split_grid_keeps_bilingual_counterparty_normalized_without_rewriting_source_layers() -> None:
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
    assert records[0]["raw"] == raw
    assert records[0]["canonical_raw"] == {"amount": "", "amount_cny": "", "counter_party": ""}


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


def test_page_source_inference_uses_boc_posting_day_to_break_repeated_row_tie() -> None:
    from docmirror.plugins.bank_statement.styles.grid_standard import _text_page_row_sources

    transaction = {
        "序号": "1",
        "记账日": "220414",
        "借方发生额": "100,000.00",
        "贷方发生额": "",
        "余额": "98,133.91",
    }
    parse_result = ParseResult(
        pages=[
            PageContent(
                page_number=2,
                texts=[TextBlock(content="1 220401 100,000.00 98,133.91")],
            ),
            PageContent(
                page_number=3,
                texts=[TextBlock(content="1 220414 100,000.00 98,133.91")],
            ),
        ]
    )

    sources = _text_page_row_sources([transaction], parse_result)

    assert sources[0]["source_page"] == 3


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


def test_split_grid_keeps_named_source_null_party_when_tail_is_bank_and_summary() -> None:
    headers = ["交易时间", "收入金额", "支出金额", "账户余额", "对方账号", "对方户名", "对方开户行", "摘要"]
    row = ["2023-04-21\n05:22:53", "177.11", "", "525.78", "10311101940040251", "", "999999", "转存"]
    source_text = "2023-04-21 05:22:53 177.11 525.78 10311101940040251 999999 转存"
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
    assert records[0]["normalized"]["counter_bank_name"] == "999999"
    assert records[0]["normalized"]["summary"] == "转存"


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
    assert header.col_map["own_account"] == 1
    assert header.col_map["counter_account"] == 6


def test_icbc_bare_account_headers_are_distinct_source_business_fields() -> None:
    raw = {
        "交易日期": "2022-09-0410:01:14",
        "账号": "1104060001031076947",
        "储种": "活期",
        "序号": "00000",
        "币种": "人民币",
        "钞汇": "钞",
        "摘要": "消费",
        "地区": "1104",
        "收入/支出金额": "-23.00",
        "余额": "268.08",
        "渠道": "快捷支付",
    }

    normalized = normalize_record(raw, BankStatementCommunityPlugin())

    assert normalized["own_account"] == "1104060001031076947"
    assert normalized["deposit_type"] == "活期"
    assert normalized["region_code"] == "1104"
    assert normalized["sub_account"] == ""
    assert normalized["counter_account"] == ""
    assert normalized["counter_party"] == ""


def _ccb_enterprise_17_column_row(
    *,
    stacked_headers: bool,
    counter_header: str,
    counter_account: str,
    counter_party: str,
) -> dict[str, str]:
    headers = [
        "账号",
        "交易时\n间" if stacked_headers else "交易时间",
        "借方发\n生额" if stacked_headers else "借方发生额",
        "贷方发\n生额" if stacked_headers else "贷方发生额",
        "余额",
        "币种",
        "对方户\n名" if stacked_headers else "对方户名",
        counter_header[:2] + "\n" + counter_header[2:] if stacked_headers else counter_header,
        "对方开户\n机构" if stacked_headers else "对方开户机构",
        "记账日\n期" if stacked_headers else "记账日期",
        "摘要",
        "备注",
        "账户明细编号-\n企业网银流水号" if stacked_headers else "账户明细编号-企业网银流水号",
        "企业流水号",
        "凭证种类",
        "凭证号",
        "交易介质编号",
    ]
    values = [
        "51001660\n04305250\n1060",
        "20240321\n00:41:05",
        "1018.22",
        "0.00",
        "134967.51",
        "人民币元",
        counter_party,
        counter_account,
        "建行四川省分行营运管理部核算中心",
        "20240321",
        "收回贷款本息",
        "",
        "2170-51000170816XP2VG71F",
        "",
        "",
        "",
        "",
    ]
    return dict(zip(headers, values))


@pytest.mark.parametrize("stacked_headers", [False, True])
@pytest.mark.parametrize("counter_header", ["对方账号", "对方账户"])
@pytest.mark.parametrize(
    ("counter_account", "counter_party"),
    [("", ""), ("6212261408013357748", "乔羽")],
)
def test_distinct_ccb_account_headers_preserve_exact_ownership_roles(
    stacked_headers: bool,
    counter_header: str,
    counter_account: str,
    counter_party: str,
) -> None:
    raw = _ccb_enterprise_17_column_row(
        stacked_headers=stacked_headers,
        counter_header=counter_header,
        counter_account=counter_account,
        counter_party=counter_party,
    )
    source_row = dict(raw)

    normalized = normalize_record(raw, BankStatementCommunityPlugin())

    assert raw == source_row
    assert normalized["own_account"] == "51001660043052501060"
    assert normalized["sub_account"] == ""
    assert normalized["counter_account"] == counter_account
    assert normalized["counter_party"] == counter_party


def test_ambiguous_bare_account_without_distinct_counter_header_stays_unowned() -> None:
    raw = _ccb_enterprise_17_column_row(
        stacked_headers=False,
        counter_header="往来账号",
        counter_account="6212261408013357748",
        counter_party="乔羽",
    )

    normalized = normalize_record(raw, BankStatementCommunityPlugin())

    assert normalized["own_account"] == ""


def test_distinct_account_headers_preserve_an_explicit_sub_account() -> None:
    raw = {
        "交易日期": "2024-03-21",
        "账号": "51001660043052501060",
        "子账号": "51001660043052501060-01",
        "对方账号": "6212261408013357748",
        "交易金额": "-10.00",
        "余额": "90.00",
    }

    normalized = normalize_record(raw, BankStatementCommunityPlugin())

    assert normalized["own_account"] == "51001660043052501060"
    assert normalized["sub_account"] == "51001660043052501060-01"
    assert normalized["counter_account"] == "6212261408013357748"


def test_explicit_counter_account_does_not_backfill_own_account() -> None:
    normalized = BankStatementCommunityPlugin()._normalize(
        {
            "交易日期": "2025-01-02",
            "交易金额": "10.00",
            "余额": "90.00",
            "对方账号": "215500690",
        }
    )

    assert normalized["counter_account"] == "215500690"
    assert normalized["own_account"] == ""


def test_ambiguous_bare_account_requires_source_owned_layout() -> None:
    plugin = BankStatementCommunityPlugin()
    ambiguous = plugin._normalize(
        {
            "交易日期": "2025-01-02",
            "交易金额": "10.00",
            "余额": "90.00",
            "账号": "6222000000000000",
        }
    )
    explicit = plugin._normalize(
        {
            "交易日期": "2025-01-02",
            "交易金额": "10.00",
            "余额": "90.00",
            "本方账号": "6222000000000000",
        }
    )

    assert ambiguous["own_account"] == ""
    assert plugin._canonical_raw_values({"账号": "6222000000000000"}, ambiguous).get("own_account") is None
    assert explicit["own_account"] == "6222000000000000"


def test_scalar_count_cannot_override_single_page_identity_count() -> None:
    identity = {
        "total_transactions": {
            "raw_value": "25",
            "normalized_value": "25",
            "source": "header.kv",
        }
    }

    _apply_source_reported_transaction_count(identity, 146)

    assert identity["total_transactions"]["normalized_value"] == "25"


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
    assert result.style_meta.expected_primary_rows == 0
    assert result.style_meta.extracted_rows == 2
    assert result.style_meta.canonical_expected == 0
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


def test_spdb_short_dates_use_same_page_period_and_preserve_business_roles() -> None:
    normalized = normalize_record(
        {
            "交易日\n期": "08-01",
            "柜员流水号": "999795591893",
            "发生额\n借方": "",
            "贷方": "100,000.00",
            "账户余额": "168,083.80",
            "交易对手信息\n对手机构": "浦发银行大众大厦支行",
            "对手名称": "田野",
            "摘要代码": "转账汇款借款",
            "备注": "借款",
            "_source_page_scope_text": "账单统计日期\n2022年08月31日\n第1页,共8页",
        },
        BankStatementCommunityPlugin(),
    )

    assert normalized["date"] == "2022-08-01"
    assert normalized["reference"] == "999795591893"
    assert normalized["direction"] == "income"
    assert normalized["amount"] == 100000.0
    assert normalized["balance"] == 168083.8
    assert normalized["counter_bank_name"] == "浦发银行大众大厦支行"
    assert normalized["counter_party"] == "田野"
    assert normalized["summary"] == "转账汇款借款"
    assert normalized["note"] == "借款"


def test_spdb_cross_page_continuation_survives_repeated_child_header() -> None:
    headers = ["交易日期", "柜员流水号", "发生额", "", "账户余额", "交易对手信息", "", "摘要代码", "备注"]
    rows = [
        _sourced_bank_row(
            [
                "08-08",
                "999572280710",
                "14,350.00",
                "",
                "72,789.38",
                "中国农业银行股份有",
                "无锡康城物流有限公",
                "电子渠道转账",
                "郑州中海唯宝运费",
            ],
            page=1,
            row_index=18,
        ),
        _sourced_bank_row(["", "", "借方", "贷方", "", "对手机构", "对手名称", "", ""], page=2, row_index=0),
        _sourced_bank_row(["", "", "", "", "", "限公司无锡石塘湾支行", "司", "", ""], page=2, row_index=1),
        _sourced_bank_row(
            ["08-08", "999572280710", "7.50", "", "72,781.88", "", "", "跨行转账(网银异地)", ""], page=2, row_index=2
        ),
    ]
    parse_result = ParseResult(
        pages=[
            PageContent(page_number=1, texts=[TextBlock(content="2022年08月31日")]),
            PageContent(page_number=2, texts=[TextBlock(content="2022年08月31日")]),
        ],
        entities=DocumentEntities(document_type="bank_statement"),
        logical_tables=[
            LogicalTable(
                table_id="lt_spdb",
                headers=headers,
                rows=rows,
                row_count=len(rows),
                source_pages=[1, 2],
                source_physical_ids=["pt_1_0", "pt_2_0"],
                page_span=(1, 2),
                provenance=[
                    RowProvenance(source_page=1, source_table_id="pt_1_0", source_row_index=18),
                    RowProvenance(source_page=2, source_table_id="pt_2_0", source_row_index=0, is_continuation=True),
                    RowProvenance(source_page=2, source_table_id="pt_2_0", source_row_index=1, is_continuation=True),
                    RowProvenance(source_page=2, source_table_id="pt_2_0", source_row_index=2, is_continuation=True),
                ],
                quality_passed=True,
                data_row_estimate=2,
            )
        ],
    )
    stats: dict[str, int] = {}
    extracted = extract_logical_rows_with_provenance(
        parse_result,
        BankStatementCommunityPlugin().column_registry,
        strict_first_col=True,
        stats=stats,
    )

    assert len(extracted) == 2
    first = extracted[0]
    assert first["交易对手信息"].replace("\n", "") == "中国农业银行股份有限公司无锡石塘湾支行"
    assert first["col_6"].replace("\n", "") == "无锡康城物流有限公司"
    assert first["_source"]["page_range"] == [1, 2]
    assert stats["stitched_continuation_rows"] == 1


def test_cross_page_continuation_without_sequence_or_reference_uses_date_anchor() -> None:
    headers = ["交易时间", "收入金额", "支出金额", "账户余额", "对方账号", "对方户名", "对方开户行", "摘要"]
    rows = [
        _sourced_bank_row(
            [
                "2023-05-19",
                "35464.67",
                "",
                "36149.52",
                "62122611040036481",
                "上海赫程国际旅行",
                "招商银行股份有限",
                "转存",
            ],
            page=1,
            row_index=13,
        ),
        _sourced_bank_row(
            ["16:24:44", "", "", "", "84", "社有限公司南通分公司", "公司上海分行营业部", ""],
            page=2,
            row_index=0,
        ),
        _sourced_bank_row(
            ["2023-05-18\n09:00:00", "", "100.00", "36049.52", "123456", "下一交易", "下一银行", "转取"],
            page=2,
            row_index=1,
        ),
    ]
    parse_result = ParseResult(
        pages=[PageContent(page_number=1), PageContent(page_number=2)],
        entities=DocumentEntities(document_type="bank_statement"),
        logical_tables=[
            LogicalTable(
                table_id="lt_no_sequence",
                headers=headers,
                rows=rows,
                row_count=len(rows),
                source_pages=[1, 2],
                source_physical_ids=["pt_1_0", "pt_2_0"],
                page_span=(1, 2),
                provenance=[
                    RowProvenance(source_page=1, source_table_id="pt_1_0", source_row_index=13),
                    RowProvenance(source_page=2, source_table_id="pt_2_0", source_row_index=0, is_continuation=True),
                    RowProvenance(source_page=2, source_table_id="pt_2_0", source_row_index=1, is_continuation=True),
                ],
                quality_passed=True,
                data_row_estimate=2,
            )
        ],
    )
    stats: dict[str, int] = {}

    extracted = extract_logical_rows_with_provenance(
        parse_result,
        BankStatementCommunityPlugin().column_registry,
        stats=stats,
    )
    normalized = [normalize_record(record, BankStatementCommunityPlugin()) for record in extracted]

    assert len(extracted) == 2
    assert normalized[0]["timestamp"] == "2023-05-19T16:24:44"
    assert normalized[0]["counter_account"] == "6212261104003648184"
    assert normalized[0]["counter_party"] == "上海赫程国际旅行社有限公司南通分公司"
    assert normalized[0]["counter_bank_name"] == "招商银行股份有限公司上海分行营业部"
    assert extracted[0]["_source"]["page_range"] == [1, 2]
    assert len(extracted[0]["_source"]["source_refs"]) == 2
    assert normalized[1]["counter_party"] == "下一交易"
    assert stats["stitched_continuation_rows"] == 1


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
    source_markdown = bundle.render_source_markdown()

    record_ids = [row["extraction"]["record_id"] for row in json_rows]
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
    assert first_json["extraction"]["page_range"] == [1, 2]
    assert first_csv["date"] == "2023-06-28"
    assert first_csv["amount"] == "1000000.0"
    assert first_csv["balance"] == "1006296.3"
    assert (first_csv["_page_start"], first_csv["_page_end"]) == ("1", "2")
    assert first_audit["amount"]["value"] == "1000000.0"
    assert first_audit["amount"]["raw"] == "1000000"
    assert first_audit["balance"]["value"] == "1006296.3"
    assert first_audit["balance"]["raw"] == "1006296.\n3"
    assert first_audit["date"]["value"] == "2023-06-28"
    from scripts.validate.bank_business_exports import assert_business_markdown_values

    assert markdown == bundle.render_enhanced_markdown(semantic)
    assert_business_markdown_values(payload, markdown)
    business_rows = [line for line in markdown.splitlines() if line.startswith(("| 13 |", "| 14 |"))]
    assert len(business_rows) == 2
    for row, line in zip(json_rows, business_rows, strict=True):
        for key in ("sequence_no", "date", "amount", "balance", "counter_account", "counter_party"):
            assert str(row["normalized"][key]) in line
    assert business_rows[0].startswith("| 13 |")
    assert business_rows[1].startswith("| 14 |")
    assert "| 序号 | 交易时间 | 流水号 | 对方账号 | 对方户名 | 支出 | 收入 | 账户余额 | 摘要 | 附言 |" in source_markdown
    first_markdown_row = (
        "| 13 | 2023-06-28 19:50:16 | 1112052 | 727279800000011760 | 江西昌荣供应链有限公司 | "
        " | 1000000 | 1006296.3 | 超网-贷记转入 | 转户 |"
    )
    second_markdown_row = (
        "| 14 | 2023-06-28 11:24:57 | 1069557 | 727279800000011760 | 江西昌荣供应链有限公司 | "
        "780000 |  | 6296.3 | 超网-贷记转出 | 转户 |"
    )
    assert first_markdown_row in source_markdown
    assert second_markdown_row in source_markdown
    assert source_markdown.index(first_markdown_row) < source_markdown.index(second_markdown_row)
