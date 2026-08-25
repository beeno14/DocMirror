# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Institution Authority Stack (IAS) tests."""

from __future__ import annotations

from docmirror.plugins.bank_statement.context import StyleContext
from docmirror.plugins.bank_statement.institution_authority import (
    extract_identity_from_header,
    resolve_institution_hint,
)


def test_organization_over_body_keyword():
    header = "开户行     中国银行南京浦东路支行"
    body = "对方户名 李正华/中国建设银行股份有限公司"
    text = header + "\n|序号|记账日|" + body
    ctx = StyleContext(
        tables=[],
        full_text=text,
        institution="中国银行",
        institution_authority="entities.organization",
        page_count=1,
    )
    hint, authority = resolve_institution_hint(ctx, {"中国建设银行": ["建设银行"]})
    assert hint == "中国银行"
    assert authority == "entities.organization"


def test_header_only_ccb_not_in_transactions():
    header = "开户行     中国银行南京浦东路支行\n账户名称  测试公司"
    body = "交易明细\n| 1 |220401|...|中国建设银行股份有限公司|"
    text = header + "\n|序号|记账日|借方发生额|贷方发生额|\n" + body
    ctx = StyleContext(
        tables=[],
        full_text=text,
        institution=None,
        page_count=1,
    )
    hint, _ = resolve_institution_hint(
        ctx,
        {"中国建设银行": ["建设银行"], "中国银行": ["中国银行"]},
    )
    assert hint is not None
    assert "中国银行" in hint


def test_filename_bank_token_priority_over_body_keyword():
    from docmirror.plugins.bank_statement.context import StyleContext
    from docmirror.plugins.bank_statement.institution_authority import resolve_institution_hint

    text = "对方户名 李正华/中国建设银行股份有限公司\n|序号|记账日|"
    ctx = StyleContext(
        tables=[],
        full_text=text,
        institution=None,
        page_count=1,
        parse_result=type("PR", (), {"file_path": "/tmp/中国银行-南京创沃电气设备有限公司_1.pdf"})(),
    )
    hint, authority = resolve_institution_hint(ctx, {"中国建设银行": ["建设银行"]})
    assert hint is not None
    assert "中国银行" in hint
    assert authority == "filename.token"


def test_context_institution_authority_is_not_laundered() -> None:
    ctx = StyleContext(
        tables=[],
        full_text="交易明细",
        institution="中国银行",
        institution_authority="filename.token",
        page_count=1,
    )

    assert resolve_institution_hint(ctx, {}) == ("中国银行", "filename.token")


def test_personal_bank_product_name_is_not_institution_hint():
    ctx = StyleContext(
        tables=[],
        full_text="个人银行结算账户 客户姓名 于鑫日 客户账号 621700001234567890",
        institution="个人银行",
        page_count=1,
        parse_result=type("PR", (), {"file_path": "/tmp/于鑫日_银行流水_个人银行.pdf"})(),
    )
    hint, authority = resolve_institution_hint(ctx, {"中国建设银行": ["建设银行"]})
    assert hint is None
    assert authority == ""


def test_extract_identity_from_header():
    text = (
        "账号     544362180589         账户名称  南京创沃电气设备有限公司"
        "                                        开户行     中国银行南京浦东路支行"
        "起始日期20220401                              截止日期 20220430"
        "\n|序号|记账日|"
    )
    identity = extract_identity_from_header(text)
    assert identity["account_holder"] == "南京创沃电气设备有限公司"
    assert identity["account_number"] == "544362180589"
    assert "2022-04-01" in identity["query_period"]
    assert identity["branch_name"] == "中国银行南京浦东路支行"
    assert "bank_name" not in identity


def test_explicit_bank_name_is_source_issuer() -> None:
    identity = extract_identity_from_header("银行名称：中国银行\n账号：12345678")

    assert identity["bank_name"] == "中国银行"


def test_extract_identity_accepts_rural_credit_cooperative_as_institution():
    text = (
        "交易明细清单\n"
        "户名：吴文坤\n"
        "账号/卡号：6230361108033553943\n"
        "开户机构：东山县农村信用合作联社陈城信用社\n"
        "序号 交易日期 收入/支出 交易金额 账户余额 对方户名\n"
    )

    identity = extract_identity_from_header(text)

    assert identity["branch_name"] == "东山县农村信用合作联社陈城信用社"
    assert "bank_name" not in identity


def test_extract_identity_from_header_when_holder_is_nearby_line():
    text = (
        "账号: 6230 **** 3462 开户行: 江苏镇江农村商业银行 起止日期: 2024-07-14 — 2025-01-14\n"
        "户名: 币种：人民币\n"
        "交易明细详情\n"
        "申请时间: 2025-01-17\n"
        "曹兴勇\n"
        "|序号|交易日期|"
    )
    identity = extract_identity_from_header(text)
    assert identity["account_holder"] == "曹兴勇"
    assert identity["account_number"] == "6230 **** 3462"
    assert identity["currency"] == "CNY"


def test_extract_identity_from_customer_name_header():
    text = (
        "中国建设银行个人活期账户收入交易明细\n"
        "卡号/账号:6227001863030091717    客户名称：郑云华    起始日期：20240101    结束日期：20241231\n"
        "序号 摘要 币别 钞汇 交易日期 交易金额 账户余额 交易地点/附言 对方账号与户名\n"
    )

    identity = extract_identity_from_header(text)

    assert identity["account_holder"] == "郑云华"
    assert identity["account_number"] == "6227001863030091717"


def test_extract_masked_card_and_explicit_period_without_crossing_into_first_transaction():
    text = (
        "户名: 测试用户\n"
        "账号/卡号: 6230\\*\\*\\*\\*6516\n"
        "币种: 人民币\n"
        "起止日期: 2023-01-01至2023-06-30\n"
        "序号 交易日期 交易金额 账户余额 用途\n"
        "1 2023-01-01 10.00 90.00 20230101782001954508147\n"
    )

    identity = extract_identity_from_header(text)

    assert identity["account_number"] == "6230****6516"
    assert identity["query_period"] == "2023-01-01 ~ 2023-06-30"


def test_header_identity_rejects_carry_forward_as_holder():
    text = "\n".join(
        [
            "641301106013000859983",
            "测试软件有限公司银川分公司",
            "2025",
            "人民币",
            "某银行分行明细对账单",
            "账号：",
            "开户机构：某银行开发区支行",
            "户名：",
            "年份：",
            "币种：",
            "承前",
            "贷方发生额",
        ]
    )

    identity = extract_identity_from_header(text)

    assert identity.get("account_holder") != "承前"


def test_horizontal_header_supports_hyphenated_account_and_chinese_dates():
    text = (
        "账户明细\n"
        "账号:31-080201040015288 户名:测试农业科技有限公司币种:人民币 "
        "起止日期: 2025年11月01日 - 2025年12月31日\n"
        "交易时间 收入金额 支出金额 账户余额 对方账号 对方户名 对方开户行 摘要"
    )

    identity = extract_identity_from_header(text)

    assert identity["account_number"] == "31-080201040015288"
    assert identity["account_holder"] == "测试农业科技有限公司"
    assert identity["currency"] == "CNY"
    assert identity["query_period"] == "2025-11-01 ~ 2025-12-31"


def test_header_transaction_time_range_is_an_explicit_query_period() -> None:
    text = (
        "银行交易明细\n"
        "账号:651204680300015 账户名:测试科技有限公司 币种:人民币\n"
        "交易时间:2025-07-01 至 2025-12-31\n"
        "日期 支出 收入 余额 对方账户 对方户名 摘要/附言\n"
        "2025-09-21 0.04 306.09 结息\n"
    )

    identity = extract_identity_from_header(text)

    assert identity["query_period"] == "2025-07-01 ~ 2025-12-31"


def test_vertical_bilingual_header_stops_before_counterparty_labels() -> None:
    text = (
        "交通银行个人客户交易清单\n"
        "6222620640011272413\n"
        "账号/卡号Account/Card No:\n"
        "周深\n"
        "户名Account Name:\n"
        "2022-05-26\n"
        "查询起日Query Starting Date:\n"
        "2023-05-26\n"
        "查询止日Query Ending Date:\n"
        "人民币 CNY\n"
        "币种Currency:\n"
        "Serial\n"
        "交易日期\n"
        "交易时间\n"
        "交易类型\n"
        "借贷\n"
        "交易金额\n"
        "余额\n"
        "对方账号\n"
        "对方户名\n"
        "序号\n"
        "1\n2022-08-05\n"
    )

    identity = extract_identity_from_header(text)

    assert identity["account_holder"] == "周深"
    assert identity["account_number"] == "6222620640011272413"
    assert identity["query_period"] == "2022-05-26 ~ 2023-05-26"


def test_split_debit_credit_table_at_document_start_does_not_promote_summary_account() -> None:
    text = (
        "交易日期\n借方(出账)\n贷方(入账)\n余额摘要\n收(付)方名称\n收(付)方账号\n交易类型\n"
        "2025-03-21 31.55 27,486.46 收息，结息账号:999019305110001 "
        "应付利息-单位活期存款利息 912351220611001810 账户结息\n"
    )

    assert extract_identity_from_header(text) == {}
