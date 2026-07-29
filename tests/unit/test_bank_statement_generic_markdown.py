# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for generic bank statement Markdown and record cleanup."""

from __future__ import annotations

from docmirror.plugins.bank_statement.canonical import dedupe_transaction_rows
from docmirror.plugins.bank_statement.community_plugin import (
    _render_bank_statement_content_markdown,
    _sanitize_bank_records,
)


def test_bank_statement_dedupe_keeps_repeated_page_sequences() -> None:
    records = [
        {
            "normalized": {"sequence_no": "1", "date": "2023-03-09", "amount": 1, "balance": 9},
            "raw": {},
            "source": {"table_id": "bank_table_p1", "page_range": [1, 1]},
        },
        {
            "normalized": {"sequence_no": "1", "date": "2023-04-01", "amount": 2, "balance": 11},
            "raw": {},
            "source": {"table_id": "bank_table_p2", "page_range": [2, 2]},
        },
        {
            "normalized": {"sequence_no": "1", "date": "2023-04-01", "amount": 2, "balance": 11},
            "raw": {},
            "source": {"table_id": "bank_table_p2", "page_range": [2, 2]},
        },
    ]

    deduped = dedupe_transaction_rows(records)

    assert len(deduped) == 2


def test_bank_statement_content_markdown_is_record_complete_and_generic() -> None:
    records = [
        {
            "raw": {
                "序号": "1",
                "摘要": "往来款",
                "交易日期": "20240102",
                "交易金额": "80,000.00",
                "余额": "102,214.76",
                "对方户名": "35001677107*****5957/顺***融竹木有限公司",
            },
            "normalized": {"direction": "income"},
            "source": {"source_page": 1, "page_range": [1, 1]},
        },
        {
            "raw": {
                "序号": "549",
                "摘要": "电子汇入生成时间:2025-02-0710:29:08",
                "交易日期": "20241231",
                "交易金额": "40,000.00",
                "余额": "168,166.41",
                "对方户名": "6230522020*****8471/陈*明总页数：29",
            },
            "normalized": {"direction": "income"},
            "source": {"source_page": 29, "page_range": [29, 29]},
        },
    ]

    markdown = _render_bank_statement_content_markdown(
        records,
        {"account_holder": "郑云华", "account_number": "6227001863030091717", "currency": "CNY"},
        {"start": "2024-01-02", "end": "2024-12-31"},
    )

    business_rows = [
        line for line in markdown.splitlines() if line.startswith("| ") and line.split("|", 3)[1].strip().isdigit()
    ]
    assert len(business_rows) == 2
    assert "| 序号 |" not in markdown
    assert "| 日期 | 收/支 | 交易金额 | 账户余额 | 对方户名 | 对方账号 | 摘要 |" in markdown
    assert business_rows[0].startswith("| 20240102 |")
    assert markdown.count("docmirror:page") == 2
    assert "## 第 1 页" in markdown
    assert "## 第 29 页" in markdown
    assert "郑云华" in markdown
    assert "生成时间" not in markdown
    assert "总页数" not in markdown
    assert "35001677107*****5957/顺***融竹木有限公司" in markdown


def test_bank_statement_markdown_prefers_source_reading_table_when_raw_headers_are_complete() -> None:
    records = [
        {
            "raw": {
                "交易日期": "20230829",
                "交易金额": "228.00",
                "交易类别": "转出",
                "账户余额": "372.38",
                "对方账号": "243300133",
                "对方户名": "扫二维码付款",
                "备注": "财付通支\n付",
                "交易机构": "101001",
            },
            "normalized": {"direction": "expense", "amount": 228, "counter_party": "扫二维码付款"},
            "source": {"source_page": 1, "page_range": [1, 1]},
        }
    ]
    source_text = "\n".join(
        [
            "常熟农商银行个人账户交易流水",
            "打印日期：20230830",
            "卡/账号： 6230710101240694111",
            "户名： 陈从兵",
            "开户行： 江苏常熟农村商业银行股份有限公司镇江分行",
            "起始日期：20230731 终止日期：20230829",
            "@常熟农商银行",
            "第 1 页，共 1页",
        ]
    )

    markdown = _render_bank_statement_content_markdown(
        records,
        {
            "account_holder": "陈从兵",
            "account_number": "6230710101240694111",
            "bank_branch": "江苏常熟农村商业银行股份有限公司镇江分行",
        },
        {"start": "2023-07-31", "end": "2023-08-29"},
        source_text,
    )

    assert "# 银行流水" not in markdown
    assert "## 第 1 页" not in markdown
    assert "常熟农商银行个人账户交易流水  \n打印日期：20230830" in markdown
    assert "常熟农商银行个人账户交易流水" in markdown
    assert "打印日期：20230830" in markdown
    assert "| 交易日期 | 交易金额 | 交易类别 | 账户余额 | 对方账号 | 对方户名 | 备注 | 交易机构 |" in markdown
    assert "| 20230829 | 228.00 | 转出 | 372.38 | 243300133 | 扫二维码付款 | 财付通支付 | 101001 |" in markdown
    assert "@常熟农商银行" in markdown


def test_bank_statement_markdown_uses_source_headers_when_raw_headers_are_partial() -> None:
    records = [
        {
            "raw": {
                "交易日期": "20230622",
                "摘要": "微信⽀付",
                "交易金额": "-500.00",
                "余额": "10705.87",
                "对方户名": "243300133",
                "序号": "457272650",
                "交易渠道": "电⼦商务",
            },
            "normalized": {
                "date": "2023-06-22",
                "amount": 500,
                "balance": "10705.87",
                "counter_party": "243300133",
                "sequence_no": "457272650",
                "summary": "微信⽀付",
                "channel": "电⼦商务",
                "direction": "expense",
            },
            "source": {"source_page": 1, "page_range": [1, 1]},
        },
        {
            "raw": {
                "交易日期": "20231221",
                "摘要": "⽀付宝",
                "交易金额": "-215",
                "余额": "2308.2",
                "对方户名": "国家税务总局江苏省税务局",
                "序号": "324030752",
                "交易渠道": "电⼦商务",
            },
            "normalized": {"date": "2023-12-21", "amount": 215, "balance": "2308.2", "direction": "expense"},
            "source": {"source_page": 3, "page_range": [3, 3]},
        },
    ]
    source_text = "\n".join(
        [
            "中国农业银⾏账⼾活期交易明细清单",
            "⼾名：徐雪 账⼾：6228481048429419672",
            "币种：⼈⺠币 钞汇标识：本币",
            "起⽌⽇期：20230622-20231221 电⼦流⽔号：2312211530229831551",
            "交易⽇期",
            "交易时间",
            "交易摘要",
            "交易⾦额",
            "本次余额",
            "对⼿信息",
            "⽇ 志号",
            "交易渠道",
            "交易附⾔",
            "20230622 195925 微信⽀付 -500.00 10705.87 243300133 457272650 电⼦商务",
            "@中国农业银⾏",
            "第 1 页，共 3页",
            "20231221 110342 ⽀付宝 -215 2308.2 324030752 电⼦商务",
            "截至打印时间下方无其他明细内容，交易明细截止2023年08月30日16时00分",
            "@中国农业银⾏",
            "第 3 页，共 3页",
        ]
    )

    markdown = _render_bank_statement_content_markdown(
        records,
        {"bank_name": "中国农业银行", "currency": "CNY"},
        {"start": "2023-06-22", "end": "2023-12-21"},
        source_text,
    )

    assert "| 交易⽇期 |" not in markdown
    assert "| 交易日期 | 交易时间 | 交易摘要 | 交易金额 | 本次余额 | 对手信息 | 日志号 | 交易渠道 | 交易附言 |" in markdown
    assert "| 20230622 |  | 微信⽀付 | -500.00 | 10705.87 | 243300133 | 457272650 | 电⼦商务 |  |" in markdown
    assert "截至打印时间下方无其他明细内容，交易明细截止2023年08月30日16时00分" in markdown
    assert "@中国农业银⾏  \n第 3 页，共 3页" in markdown
    assert "# 银行流水" not in markdown


def test_bank_statement_markdown_infers_generic_raw_headers_and_transaction_type_direction() -> None:
    records = [
        {
            "raw": {
                "序号": "1",
                "摘要/附言": "协议付款",
                "币别": "人民币",
                "交易日期": "20230831",
                "交易类型": "支出",
                "交易金额": "31.00",
                "账户余额": "99.79",
                "对方账号": "215500690",
                "对方户名": "支付宝（中国）网络技术有限公司",
            },
            "normalized": {
                "date": "2023-08-31",
                "amount": 31.0,
                "balance": 99.79,
                "direction": "",
                "summary": "协议付款",
            },
            "source": {"source_page": 1, "page_range": [1, 1]},
        },
        {
            "raw": {
                "序号": "2",
                "摘要/附言": "工资",
                "币别": "人民币",
                "交易日期": "20230830",
                "交易类型": "收入",
                "交易金额": "296.00",
                "账户余额": "299.69",
                "对方账号": "70070188000077841",
                "对方户名": "",
            },
            "normalized": {
                "date": "2023-08-30",
                "amount": 296.0,
                "balance": 299.69,
                "direction": "",
                "summary": "工资",
            },
            "source": {"source_page": 1, "page_range": [1, 1]},
        },
    ]

    markdown = _render_bank_statement_content_markdown(
        records,
        {"bank_name": "江苏银行", "account_holder": "李秀茹"},
        {"start": "2022-09-09", "end": "2023-08-31"},
    )

    assert "| 序号 |" not in markdown
    assert "| 币别 |" not in markdown
    assert "| 交易日期 | 交易类型 | 交易金额 | 账户余额 | 对方账号 | 对方户名 | 摘要/附言 |" in markdown
    assert "| 20230831 | 支出 | 31.00 | 99.79 | 215500690 | 支付宝（中国）网络技术有限公司 | 协议付款 |" in markdown
    assert "| 20230830 | 收入 | 296.00 | 299.69 | 70070188000077841 |  | 工资 |" in markdown


def test_bank_statement_record_sanitizer_removes_footer_and_money_noise() -> None:
    records = [
        {
            "raw": {
                "摘要": "电子汇入生成时间:2025-02-0710:29:08",
                "余额": "168,166.41***",
                "对方户名": "6230522020*****8471/陈*明总页数：29",
            },
            "canonical_raw": {
                "summary": "电子汇入生成时间:2025-02-0710:29:08",
                "balance": "168,166.41***",
                "counter_party": "6230522020*****8471/陈*明总页数：29",
            },
            "normalized": {
                "summary": "电子汇入生成时间:2025-02-0710:29:08",
                "balance": "168,166.41***",
                "counter_party": "6230522020*****8471/陈*明总页数：29",
            },
        }
    ]

    sanitized = _sanitize_bank_records(records)

    assert sanitized[0]["raw"]["摘要"] == "电子汇入"
    assert sanitized[0]["raw"]["余额"] == "168,166.41"
    assert sanitized[0]["raw"]["对方户名"] == "6230522020*****8471/陈*明"
    assert sanitized[0]["canonical_raw"]["balance"] == "168,166.41"
    assert sanitized[0]["normalized"]["counter_party"] == "6230522020*****8471/陈*明"


def test_bank_statement_record_sanitizer_removes_counterparty_pollution() -> None:
    records = [
        {
            "raw": {
                "摘要": "往来款",
                "对方账户": "1104010309000388824",
                "对方户名": "镇江小松鼠计算机技术服务有限公司企业电子渠道跨行转账手续费收",
            },
            "canonical_raw": {
                "summary": "往来款",
                "counter_account": "1104010309000388824",
                "counter_party": "镇江小松鼠计算机技术服务有限公司企业电子渠道跨行转账手续费收",
            },
            "normalized": {
                "summary": "往来款",
                "counter_account": "1104010309000388824",
                "counter_party": "镇江小松鼠计算机技术服务有限公司企业电子渠道跨行转账手续费收",
            },
        },
        {
            "raw": {"摘要": "收费", "对方账户": "70650107360000033", "对方户名": "入"},
            "canonical_raw": {"summary": "收费", "counter_account": "70650107360000033", "counter_party": "入"},
            "normalized": {"summary": "收费", "counter_account": "70650107360000033", "counter_party": "入"},
        },
        {
            "raw": {
                "摘要": "住房公积金",
                "对方账户": "32001755236052503998",
                "对方户名": "镇江市住房公积金管理中心序号交易日期交易时间摘要凭证种类借方发生额贷方发生额余额对方账户对方户名",
            },
            "canonical_raw": {
                "summary": "住房公积金",
                "counter_account": "32001755236052503998",
                "counter_party": "镇江市住房公积金管理中心序号交易日期交易时间摘要凭证种类借方发生额贷方发生额余额对方账户对方户名",
            },
            "normalized": {
                "summary": "住房公积金",
                "counter_account": "32001755236052503998",
                "counter_party": "镇江市住房公积金管理中心序号交易日期交易时间摘要凭证种类借方发生额贷方发生额余额对方账户对方户名",
            },
        },
        {
            "raw": {"摘要": "报销", "对方账户": "6228760805002812421", "对方户名": "夏炎"},
            "canonical_raw": {"summary": "报销", "counter_account": "6228760805002812421", "counter_party": "夏炎"},
            "normalized": {"summary": "报销", "counter_account": "6228760805002812421", "counter_party": "夏炎"},
        },
        {
            "raw": {"摘要": "报销", "对方账户": "6228760805002812421", "对方户名": "夏炎镇江大学科技园资产经营管理有"},
            "canonical_raw": {
                "summary": "报销",
                "counter_account": "6228760805002812421",
                "counter_party": "夏炎镇江大学科技园资产经营管理有",
            },
            "normalized": {
                "summary": "报销",
                "counter_account": "6228760805002812421",
                "counter_party": "夏炎镇江大学科技园资产经营管理有",
            },
        },
        {
            "raw": {"摘要": "公共耗能和水电费用", "对方账户": "6232511300395178", "对方户名": "限公司"},
            "canonical_raw": {"summary": "公共耗能和水电费用", "counter_account": "6232511300395178", "counter_party": "限公司"},
            "normalized": {"summary": "公共耗能和水电费用", "counter_account": "6232511300395178", "counter_party": "限公司"},
        },
        {
            "raw": {"摘要": "tips扣税", "对方账户": "70010151830005003", "对方户名": "代收）"},
            "canonical_raw": {"summary": "tips扣税", "counter_account": "70010151830005003", "counter_party": "代收）"},
            "normalized": {"summary": "tips扣税", "counter_account": "70010151830005003", "counter_party": "代收）"},
        },
        {
            "raw": {"摘要": "收费", "对方账户": "70650107360000033", "对方户名": "企业电子渠道跨行转账手续费收入"},
            "canonical_raw": {
                "summary": "收费",
                "counter_account": "70650107360000033",
                "counter_party": "企业电子渠道跨行转账手续费收入",
            },
            "normalized": {
                "summary": "收费",
                "counter_account": "70650107360000033",
                "counter_party": "企业电子渠道跨行转账手续费收入",
            },
        },
        {
            "raw": {"摘要": "tips扣税", "对方账户": "70010151830005003", "对方户名": "待报解预算收入（财税库银联网代收）"},
            "canonical_raw": {
                "summary": "tips扣税",
                "counter_account": "70010151830005003",
                "counter_party": "待报解预算收入（财税库银联网代收）",
            },
            "normalized": {
                "summary": "tips扣税",
                "counter_account": "70010151830005003",
                "counter_party": "待报解预算收入（财税库银联网代收）",
            },
        },
    ]

    sanitized = _sanitize_bank_records(records)

    assert sanitized[0]["normalized"]["counter_party"] == "镇江小松鼠计算机技术服务有限公司"
    assert sanitized[0]["raw"]["对方户名"] == "镇江小松鼠计算机技术服务有限公司"
    assert sanitized[1]["normalized"]["counter_party"] == "企业电子渠道跨行转账手续费收入"
    assert sanitized[1]["normalized"]["counterparty_status"] == "present"
    assert sanitized[2]["normalized"]["counter_party"] == "镇江市住房公积金管理中心"
    assert sanitized[3]["normalized"]["counter_party"] == "夏炎"
    assert sanitized[4]["normalized"]["counter_party"] == "夏炎"
    assert sanitized[5]["normalized"]["counter_party"] == ""
    assert sanitized[6]["normalized"]["counter_party"] == "待报解预算收入（财税库银联网代收）"
    assert sanitized[7]["normalized"]["counter_party"] == "企业电子渠道跨行转账手续费收入"
    assert sanitized[8]["normalized"]["counter_party"] == "待报解预算收入（财税库银联网代收）"
