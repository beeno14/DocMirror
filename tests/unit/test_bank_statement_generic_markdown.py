# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for generic bank statement Markdown and record cleanup."""

from __future__ import annotations

import copy

from docmirror.models.entities.parse_result import DocumentEntities, ParseResult
from docmirror.plugins.bank_statement.canonical import dedupe_transaction_rows
from docmirror.plugins.bank_statement.community_plugin import (
    BANK_DATA_DICTIONARY,
    _mark_represented_header_delivery_exclusions,
    _parse_result_source_table_headers,
    _raw_statement_after_table_lines,
    _raw_statement_header_lines,
    _render_bank_statement_content_markdown,
    _sanitize_bank_records,
    plugin,
)


def test_bank_statement_projection_publishes_chinese_reading_labels() -> None:
    projection = plugin.derive(
        ParseResult(entities=DocumentEntities(document_type="bank_statement")),
        "",
    )

    dictionary = projection.domain_facts["data_dictionary"]

    assert dictionary == BANK_DATA_DICTIONARY
    assert dictionary["fields"]["period_start"]["label"] == "账期开始"
    assert dictionary["fields"]["extract_status"]["label"] == "提取状态"
    assert dictionary["fields"]["document_scene_refined"]["label"] == "修正文档场景"
    assert dictionary["fields"]["layout_profile_id_refined"]["label"] == "修正版式配置"
    assert dictionary["record_columns"]["amount"]["label"] == "交易金额"
    assert dictionary["record_columns"]["counter_account"]["label"] == "对方账号"
    assert dictionary["record_columns"]["timestamp"]["label"] == "交易时间"
    assert dictionary["enums"]["direction"] == {"income": "收入", "expense": "支出"}
    assert dictionary["enums"]["layout_profile_id_refined"]["borderless_ledger_bank"] == "无框银行流水版式"


def test_bank_statement_dedupe_keeps_unproven_repeated_rows_on_the_same_page() -> None:
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

    # With no bbox, evidence id, or source-row index, the final two records
    # may be two genuine same-value ledger events.  Plane overlap is handled
    # upstream; the canonical row layer must preserve ambiguous source rows.
    assert len(deduped) == 3


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
                "摘要": "电子汇入",
                "交易日期": "20241231",
                "交易金额": "40,000.00",
                "余额": "168,166.41",
                "对方户名": "6230522020*****8471/陈*明",
            },
            "normalized": {"direction": "income"},
            "source": {"source_page": 29, "page_range": [29, 29]},
        },
    ]

    markdown = _render_bank_statement_content_markdown(
        _sanitize_bank_records(records),
        {"account_holder": "郑云华", "account_number": "6227001863030091717", "currency": "CNY"},
        {"start": "2024-01-02", "end": "2024-12-31"},
    )

    business_rows = [
        line for line in markdown.splitlines() if line.startswith("| ") and line.split("|", 3)[1].strip().isdigit()
    ]
    assert len(business_rows) == 2
    assert "| 序号 | 摘要 | 交易日期 | 交易金额 | 余额 | 对方户名 |" in markdown
    assert "| 日期 | 收/支 |" not in markdown
    assert business_rows[0].startswith("| 1 | 往来款 | 20240102 |")
    assert markdown.count("docmirror:page") == 2
    assert "## 第 1 页" not in markdown
    assert "## 第 29 页" not in markdown
    assert "郑云华" in markdown
    assert "生成时间" not in markdown
    assert "总页数" not in markdown
    assert "35001677107*****5957/顺***融竹木有限公司" in markdown


def test_bank_statement_summary_markdown_compacts_wrapped_numeric_counter_account() -> None:
    records = [
        {
            "raw": {
                "交易日期": "2025-01-24",
                "支出金额": "200000.00",
                "余额": "2369231.13",
                "对方账号": "830100788013000002\n20",
                "对方户名": "重庆中链农科技有限公司",
                "摘要": "跨行转账",
            },
            "normalized": {
                "date": "2025-01-24",
                "direction": "expense",
                "amount": 200000.0,
                "balance": 2369231.13,
                "counter_account": "83010078801300000220",
                "counter_party": "重庆中链农科技有限公司",
                "summary": "跨行转账",
            },
            "source": {"source_page": 1, "page_range": [1, 1]},
        }
    ]

    markdown = _render_bank_statement_content_markdown(records, {}, {})

    assert "83010078801300000220" in markdown
    assert "830100788013000002 20" not in markdown


def test_positioned_signed_amount_ocr_markdown_restores_source_column_order() -> None:
    records = [
        {
            "raw": {
                "交易时间": "2023-03-09",
                "收/支": "支出",
                "交易金额": "4819.00",
                "账户余额": "401143.31",
                "摘要": "备用金",
                "对方账号": "6226192013864418",
                "对方户名": "杨光",
            },
            "normalized": {
                "sequence_no": "1",
                "date": "2023-03-09",
                "direction": "expense",
                "amount": 4819.0,
                "balance": 401143.31,
                "counter_account": "6226192013864418",
                "counter_party": "杨光",
                "summary": "备用金",
            },
            "source": {"source_page": 1, "page_range": [1, 1]},
        }
    ]
    labels = [
        ("承前余额：405,962.31", [1089, 162, 1293, 188]),
        ("序号", [63, 206, 114, 237]),
        ("对方账户", [1075, 206, 1168, 237]),
        ("传票号", [1333, 206, 1405, 238]),
        ("余额", [493, 207, 544, 237]),
        ("摘要", [1442, 207, 1493, 237]),
        ("日期", [161, 208, 208, 235]),
        ("对方户名", [706, 208, 796, 236]),
        ("借/贷方发生额", [280, 210, 424, 234]),
    ]
    parse_result = ParseResult(
        entities=DocumentEntities(
            domain_specific={
                "_page_evidence_bundles": [
                    {
                        "page": 1,
                        "local_structure_evidence": {"tokens": [{"text": text, "bbox": bbox} for text, bbox in labels]},
                    }
                ]
            }
        )
    )
    headers = _parse_result_source_table_headers(parse_result)
    expected_headers = [
        "序号",
        "日期",
        "借/贷方发生额",
        "余额",
        "对方户名",
        "对方账户",
        "传票号",
        "摘要",
    ]
    markdown = _render_bank_statement_content_markdown(records, {}, {}, source_headers=headers)

    assert headers == expected_headers
    assert "| " + " | ".join(expected_headers) + " |" in markdown
    assert "| 1 | 2023-03-09 | -4819.00 | 401143.31 | 杨光 | 6226192013864418 |  | 备用金 |" in markdown
    assert "| 交易时间 | 收/支 |" not in markdown


def test_source_markdown_keeps_dynamic_raw_columns_without_normalized_fallback() -> None:
    headers = ["交易日期", "交易时间", "交易摘要", "交易金额", "本次余额", "对手信息", "日志号", "交易渠道", "交易附言"]
    records = [
        {
            "raw": dict(
                zip(
                    headers,
                    [
                        "20230622",
                        "195925",
                        "微信支付",
                        "-500.00",
                        "10705.87",
                        "243300133",
                        "457272650",
                        "电子商务",
                        "二维码付款",
                    ],
                    strict=True,
                )
            ),
            "normalized": {"date": "2023-06-22", "timestamp": "2023-06-22T19:59:25", "amount": 500.0},
            "source": {"source_page": 1, "page_range": [1, 1]},
        },
        {
            "raw": dict(
                zip(
                    headers,
                    ["20231221", "", "利息税", "+0.00", "2953.20", "--", "1", "", "个人活期结息"],
                    strict=True,
                )
            ),
            "normalized": {"date": "2023-12-21", "timestamp": "2023-12-21T00:00:00", "amount": 0.0},
            "source": {"source_page": 2, "page_range": [2, 2]},
        },
    ]
    page_one = "\n".join(
        [
            "中国农业银行账户活期交易明细清单",
            "户名：徐雪 账户：6228481048429419672",
            "币种：人民币 钞汇标识：本币",
            "起止日期：20230622-20231221 电子流水号：2312211530229831551",
            " ".join(headers),
        ]
    )
    disclaimer = "该交易明细因不可预测的非人控技术原因可能导致数据缺失，明细内容仅供参考。"
    page_two = "\n".join([" ".join(headers), disclaimer])

    markdown = _render_bank_statement_content_markdown(
        records,
        {"account_holder": "徐雪", "account_number": "6228481048429419672", "currency": "CNY"},
        {"start": "2023-06-22", "end": "2023-12-21"},
        page_one,
        source_pages={1: page_one, 2: page_two},
    )

    assert "币种：人民币 钞汇标识：本币" in markdown
    assert "电子流水号：2312211530229831551" in markdown
    assert markdown.count("户名：徐雪") == 1
    assert "| 20230622 | 195925 | 微信支付 | -500.00 |" in markdown
    assert "| 20231221 |  | 利息税 | +0.00 |" in markdown
    assert "2023-12-21T00:00:00" not in markdown
    assert markdown.rfind(disclaimer) > markdown.rfind("| 20231221 |")


def test_source_header_stops_before_individually_positioned_table_labels() -> None:
    source_text = "\n".join(
        [
            "中国农业银行账户活期交易明细清单",
            "户名：徐雪",
            "账户：6228481048429419672",
            "币种：人民币",
            "起止日期：20230622-20231221",
            "交易日期",
            "交易时间",
            "交易摘要",
        ]
    )

    lines = _raw_statement_header_lines({}, {}, source_text)

    assert lines[-1] == "起止日期：20230622-20231221"
    assert "交易日期" not in lines


def test_source_header_stops_before_concatenated_table_label_fragments() -> None:
    source_text = "\n".join(
        [
            "交易明细清单",
            "户名：测试用户",
            "账号/卡号：6230000000000000000",
            "币种：人民币",
            "开户机构：测试农村信用社",
            "起止日期：2025-01-01至2025-12-31",
            "钞汇标志：本币",
            "序号交易日期",
            "收入/支出交易金额账户余额",
        ]
    )

    lines = _raw_statement_header_lines({}, {}, source_text)

    assert lines[-1] == "钞汇标志：本币"
    assert "序号交易日期" not in lines


def test_source_header_stops_before_split_partial_table_labels() -> None:
    source_text = "\n".join(
        [
            "测试银行公司账户交易明细清单",
            "户名:测试企业有限公司",
            "账号:727150100100143834",
            "起止日期:2022/05/18-2023/05/17",
            "账户号:000001",
            "借贷标",
            "交易口",
            "志",
        ]
    )

    lines = _raw_statement_header_lines({}, {}, source_text)

    assert lines[-1] == "账户号:000001"
    assert "借贷标" not in lines
    assert "交易口" not in lines
    assert "志" not in lines

    interleaved_lines = _raw_statement_header_lines(
        {},
        {},
        "\n".join(
            [
                "测试银行公司账户交易明细清单",
                "户名:测试企业有限公司",
                "账号:727150100100143834",
                "交易口借贷标",
                "志",
            ]
        ),
    )
    assert interleaved_lines[-1] == "账号:727150100100143834"


def test_source_header_stops_before_isolated_table_row_ordinals() -> None:
    source_text = "\n".join(
        [
            "测试银行账户交易明细表",
            "（代回单）",
            "打印日期：2026-02-24",
            "交易时段：2025-01-01 至 2025-12-31",
            "总条数：38",
            "户名：测试企业有限公司",
            "账号：0111014170000993",
            "币种：人民币",
            "2.",
            "3.",
        ]
    )

    lines = _raw_statement_header_lines({}, {}, source_text)

    assert lines[-1] == "币种：人民币"


def test_source_header_repairs_two_character_kv_labels_split_by_ocr() -> None:
    source_text = "\n".join(
        [
            "账户明细对账单",
            "客户名称:测试企业",
            "种:人民币",
            "起始日期:20230601",
            "币",
            "位:元",
            "开户机构:测试银行",
            "截止日期:20230630",
            "单",
            "序号",
            "记账日期",
            "交易金额",
            "账户余额",
        ]
    )

    lines = _raw_statement_header_lines({}, {}, source_text)

    assert "币种:人民币" in lines
    assert "单位:元" in lines
    assert "种:人民币" not in lines
    assert "币" not in lines
    assert "单" not in lines
    assert "位:元" not in lines
    assert "2." not in lines
    assert "3." not in lines


def test_source_header_keeps_split_source_page_number() -> None:
    source_text = "\n".join(
        [
            "测试银行账户交易明细表",
            "户名：测试用户",
            "账号：6230000000000000000",
            "第 2",
            "/",
            "4 页",
            "序号",
        ]
    )

    lines = _raw_statement_header_lines({}, {}, source_text)

    assert "第 2 / 4 页" in lines


def test_source_header_stops_before_numbered_hyphenated_transaction_line() -> None:
    source_text = "\n".join(
        [
            "测试银行账户交易明细表",
            "户名：测试用户",
            "账号：6230000000000000000",
            "1 2025-01-24 14:23:24 100.00 900.00",
        ]
    )

    lines = _raw_statement_header_lines({}, {}, source_text)

    assert lines[-1] == "账号：6230000000000000000"


def test_source_footer_furniture_moves_out_of_header_and_stays_complete() -> None:
    legal_notice = "本回单被伪造、变造、篡改的，不具有法律效力；"
    source_text = "\n".join(
        [
            "测试银行账户交易明细表",
            "打印日期：2026-02-24",
            "户名：测试企业有限公司",
            "账号：0111014170000993",
            "风险提示：",
            legal_notice,
            "测试银行代回单第",
            "页",
            "4页 / 共4",
        ]
    )

    header_lines = _raw_statement_header_lines({}, {}, source_text)
    footer_lines = _raw_statement_after_table_lines(source_text, 4)

    assert header_lines[-1] == "账号：0111014170000993"
    assert "风险提示：" not in header_lines
    assert footer_lines == ["测试银行代回单第4页 / 共4", "风险提示：", legal_notice]


def test_source_first_markdown_fills_split_header_values_and_keeps_footer_outside_table() -> None:
    records = [
        {
            "raw": {
                "序号": "1",
                "交易日期": "20250710",
                "贷方发生额": "30,000.00",
                "余额": "36,989.93",
                "摘要": "往来款",
            },
            "normalized": {
                "date": "2025-07-10",
                "direction": "income",
                "amount": 30000.0,
                "balance": 36989.93,
                "summary": "往来款",
            },
            "source": {"source_page": 1, "page_range": [1, 1]},
        }
    ]
    source_text = "\n".join(
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
            "序号 会计日期 交易日期 借方发生额 贷方发生额 余额 摘要",
            "当前账单借方发生数：10 当前账单贷方发生数：1",
            "本月累计借方发生额：28,314.48 本月累计贷方发生额：30,000.00",
            "出单截至日期：2025-07-31",
        ]
    )

    markdown = _render_bank_statement_content_markdown(
        records,
        {
            "account_holder": "测试软件有限公司银川分公司",
            "account_number": "641301106013000859983",
            "bank_name": "某银行开发区支行",
            "currency": "CNY",
        },
        {"start": "2025-07-01", "end": "2025-07-31"},
        source_text,
        document_type="bank_reconciliation",
        source_pages={1: source_text},
    )

    assert "账号：641301106013000859983" in markdown
    assert "户名：测试软件有限公司银川分公司" in markdown
    assert "币种：CNY" in markdown
    assert "账期：20250701 至 20250731" in markdown
    assert markdown.count("| 1 |") == 1
    assert "当前账单借方发生数：10 当前账单贷方发生数：1" in markdown
    assert "出单截至日期：2025-07-31" in markdown
    assert not any("当前账单" in line for line in markdown.splitlines() if line.startswith("| "))


def test_source_first_markdown_prefers_page_local_reconstructed_footer() -> None:
    records = [
        {
            "raw": {
                "序号": "1",
                "交易日期": "20250710",
                "贷方发生额": "30,000.00",
                "余额": "36,989.93",
            },
            "normalized": {
                "date": "2025-07-10",
                "direction": "income",
                "amount": 30000.0,
                "balance": 36989.93,
            },
            "source": {"source_page": 1, "page_range": [1, 1]},
        }
    ]
    native_text = "某银行明细对账单\n序号 交易日期 借方发生额 贷方发生额 余额"
    page_text = "\n".join(
        [
            native_text,
            "开户机构：某银行开发区支行 页码：本月第1份-第1页",
            "当前账单借方发生数：10 当前账单贷方发生数：1",
            "本月累计借方发生额：28,314.48 本月累计贷方发生额：30,000.00",
            "1 20250710 30,000.00 36,989.93 往来款",
            "出单截至日期：2025-07-31",
        ]
    )

    markdown = _render_bank_statement_content_markdown(
        records,
        {},
        {},
        native_text,
        document_type="bank_reconciliation",
        source_pages={1: page_text},
    )

    assert "当前账单借方发生数：10 当前账单贷方发生数：1" in markdown
    assert "本月累计借方发生额：28,314.48 本月累计贷方发生额：30,000.00" in markdown
    assert "出单截至日期：2025-07-31" in markdown


def test_source_first_markdown_preserves_inline_page_totals_footer() -> None:
    records = [
        {
            "raw": {
                "交易日期": "2024-03-11 22:27:09",
                "发生额": "-45.00",
                "余额": "15214.50",
            },
            "normalized": {
                "date": "2024-03-11",
                "timestamp": "2024-03-11T22:27:09",
                "direction": "expense",
                "amount": 45.0,
                "balance": 15214.5,
            },
            "source": {"source_page": 1, "page_range": [1, 1]},
        }
    ]
    footer = "第1页共4页本页支出合计 : 17546.22 本页收入合计: 17639.64 本页交易笔数: 28"
    page_text = "\n".join(
        [
            "湖北农商银行个人交易流水",
            "交易日期 对方户名 对方账号/卡号 交易摘要 发生额 余额 币种",
            "2024-03-11 22:27:09 汇款 -45.00 15214.50 CNY",
            footer,
        ]
    )

    markdown = _render_bank_statement_content_markdown(
        records,
        {"account_holder": "测试客户", "account_number": "6224000000000000"},
        {"start": "2024-03-10", "end": "2025-03-08"},
        page_text,
        source_pages={1: page_text},
    )

    assert footer in markdown
    assert f"| {footer} |" not in markdown
    assert markdown.count("1 20250710 30,000.00 36,989.93 往来款") == 0
    assert "页码：本月第1份-第1页" not in markdown


def test_bank_reconciliation_markdown_uses_recovered_source_title() -> None:
    records = [
        {
            "raw": {"交易日期": "2025/01/01", "借方发生额": "1.00", "账户余额": "9.00"},
            "normalized": {
                "date": "2025-01-01",
                "direction": "expense",
                "amount": 1.0,
                "balance": 9.0,
            },
            "source": {"source_page": 1, "page_range": [1, 1]},
        }
    ]

    markdown = _render_bank_statement_content_markdown(
        records,
        {"statement_title": "上海浦东发展银行电子对账单"},
        {"start": "2025-01-01", "end": "2025-12-31"},
    )

    assert "# 上海浦东发展银行电子对账单" in markdown
    assert "# 银行流水" not in markdown


def test_bank_reconciliation_markdown_uses_alias_title_when_source_title_is_unavailable() -> None:
    records = [
        {
            "raw": {"交易日期": "2025/01/01", "借方发生额": "1.00", "账户余额": "9.00"},
            "normalized": {
                "date": "2025-01-01",
                "direction": "expense",
                "amount": 1.0,
                "balance": 9.0,
            },
            "source": {"source_page": 1, "page_range": [1, 1]},
        }
    ]

    markdown = _render_bank_statement_content_markdown(
        records,
        {},
        {},
        document_type="bank_reconciliation",
    )

    assert "# 银行对账单" in markdown
    assert "# 银行流水" not in markdown


def test_bank_reconciliation_markdown_preserves_non_transaction_remarks_page() -> None:
    records = [
        {
            "raw": {"交易日期": "2025/12/31", "借方发生额": "0.90", "账户余额": "24,175.16"},
            "normalized": {
                "date": "2025-12-31",
                "direction": "expense",
                "amount": 0.9,
                "balance": 24175.16,
            },
            "source": {"source_page": 1, "page_range": [1, 1]},
        }
    ]
    source_pages = {
        1: "上海浦东发展银行电子对账单\nDate | 交易流水号 | 发生额 | 账户余额",
        2: "\n".join(
            [
                "第2页,共2页",
                "提示 Remarks:",
                "1、请及时核实可能发生的财务差错。",
                "2、交易日期为银行记账日期。",
                "2026/02/24",
                "2VU4LBSCQ0LN8UT",
                "账单生成日期 Statement Generation Date",
            ]
        ),
    }

    markdown = _render_bank_statement_content_markdown(
        records,
        {
            "statement_title": "上海浦东发展银行电子对账单",
            "account_holder": "测试有限公司",
            "account_number": "001234",
        },
        {"start": "2025-01-01", "end": "2025-12-31"},
        source_pages[1],
        document_type="bank_reconciliation",
        source_pages=source_pages,
    )

    assert markdown.count("docmirror:page") == 2
    assert '<!-- docmirror:page logical="2" source="2" -->' in markdown
    assert "提示 Remarks:" in markdown
    assert "1、请及时核实可能发生的财务差错。" in markdown
    assert "账单生成日期 Statement Generation Date 2026/02/24" in markdown
    assert "2VU4LBSCQ0LN8UT" not in markdown
    assert "Date | 交易流水号 | 发生额 | 账户余额" not in markdown


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
    assert (
        "| 交易日期 | 交易时间 | 交易摘要 | 交易金额 | 本次余额 | 对手信息 | 日志号 | 交易渠道 | 交易附言 |" in markdown
    )
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

    assert "| 序号 | 摘要/附言 | 币别 | 交易日期 | 交易类型 | 交易金额 | 账户余额 | 对方账号 | 对方户名 |" in markdown
    assert (
        "| 1 | 协议付款 | 人民币 | 20230831 | 支出 | 31.00 | 99.79 | 215500690 | 支付宝（中国）网络技术有限公司 |"
        in markdown
    )
    assert "| 2 | 工资 | 人民币 | 20230830 | 收入 | 296.00 | 299.69 | 70070188000077841 |  |" in markdown


def test_bank_statement_markdown_omits_empty_schema_padding_columns() -> None:
    records = [
        {
            "raw": {
                "交易时间": "2022-05-20",
                "收/支": "支出",
                "交易金额": "13567.84",
                "账户余额": "22564.73",
                "摘要": "财税库行联网划缴税款",
                "对方账号": "081010143711000196",
                "对方账号与户名": "待报解预算收入",
                "机构": "",
                "柜员": "",
                "备注": "",
            },
            "normalized": {
                "date": "2022-05-20",
                "direction": "expense",
                "amount": 13567.84,
                "balance": 22564.73,
            },
            "source": {"source_page": 1, "page_range": [1, 1]},
        }
    ]
    source_text = "\n".join(
        [
            "测试银行公司账户交易明细清单",
            "户名:测试企业有限公司",
            "账号:727150100100143834",
            "借贷标",
            "志",
        ]
    )

    markdown = _render_bank_statement_content_markdown(records, {}, {}, source_text)

    assert "| 交易时间 | 收/支 | 交易金额 | 账户余额 | 摘要 | 对方账号 | 对方账号与户名 |" in markdown
    assert "| 机构 |" not in markdown
    assert "| 柜员 |" not in markdown
    assert "| 备注 |" not in markdown


def test_bank_statement_markdown_preserves_unusual_source_columns_and_order() -> None:
    records = [
        {
            "raw": {
                "记账日期": "2025-01-02",
                "起息日": "2025-01-03",
                "凭证号": "V0007",
                "借方发生额": "88.20",
                "贷方发生额": "",
                "余额": "911.80",
                "用途": "采购款",
                "附言": "合同A-17",
            },
            "normalized": {
                "date": "2025-01-02",
                "direction": "expense",
                "amount": 88.2,
                "balance": 911.8,
                "summary": "采购款",
            },
            "source": {"source_page": 1, "page_range": [1, 1]},
        }
    ]

    markdown = _render_bank_statement_content_markdown(records, {}, {})

    assert "| 记账日期 | 起息日 | 凭证号 | 借方发生额 | 贷方发生额 | 余额 | 用途 | 附言 |" in markdown
    assert "| 2025-01-02 | 2025-01-03 | V0007 | 88.20 |  | 911.80 | 采购款 | 合同A-17 |" in markdown
    assert "| 日期 | 收/支 |" not in markdown


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

    source_records = copy.deepcopy(records)
    sanitized = _sanitize_bank_records(records)

    assert sanitized[0]["raw"] == source_records[0]["raw"]
    assert sanitized[0]["canonical_raw"] == source_records[0]["canonical_raw"]
    assert sanitized[0]["normalized"]["summary"] == "电子汇入"
    assert sanitized[0]["normalized"]["balance"] == "168,166.41"
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
            "canonical_raw": {
                "summary": "公共耗能和水电费用",
                "counter_account": "6232511300395178",
                "counter_party": "限公司",
            },
            "normalized": {
                "summary": "公共耗能和水电费用",
                "counter_account": "6232511300395178",
                "counter_party": "限公司",
            },
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
            "raw": {
                "摘要": "tips扣税",
                "对方账户": "70010151830005003",
                "对方户名": "待报解预算收入（财税库银联网代收）",
            },
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
        {
            "raw": {
                "摘要": "转出",
                "对方账户": "1000050001",
                "对方户名": (
                    "1000107101 1000050001 清单支出算术合计:19,756,586.06 "
                    "打印渠道:远程视频柜员机 WL财付通微信转账:微信转账 "
                    "WL财付通微信转账:微信转账 对方户名张祝祥陈元友"
                ),
            },
            "canonical_raw": {
                "summary": "转出",
                "counter_account": "1000050001",
                "counter_party": (
                    "1000107101 1000050001 清单支出算术合计:19,756,586.06 "
                    "打印渠道:远程视频柜员机 WL财付通微信转账:微信转账 "
                    "WL财付通微信转账:微信转账 对方户名张祝祥陈元友"
                ),
            },
            "normalized": {
                "summary": "转出",
                "counter_account": "1000050001",
                "counter_party": (
                    "1000107101 1000050001 清单支出算术合计:19,756,586.06 "
                    "打印渠道:远程视频柜员机 WL财付通微信转账:微信转账 "
                    "WL财付通微信转账:微信转账 对方户名张祝祥陈元友"
                ),
            },
        },
    ]

    source_records = copy.deepcopy(records)
    sanitized = _sanitize_bank_records(records)

    for source, cleaned in zip(source_records, sanitized, strict=True):
        assert cleaned["raw"] == source["raw"]
        assert cleaned["canonical_raw"] == source["canonical_raw"]

    assert sanitized[0]["normalized"]["counter_party"] == "镇江小松鼠计算机技术服务有限公司"
    assert sanitized[1]["normalized"]["counter_party"] == ""
    assert sanitized[1]["normalized"]["counterparty_status"] == "present"
    assert sanitized[2]["normalized"]["counter_party"] == "镇江市住房公积金管理中心"
    assert sanitized[3]["normalized"]["counter_party"] == "夏炎"
    assert sanitized[4]["normalized"]["counter_party"] == "夏炎"
    assert sanitized[5]["normalized"]["counter_party"] == ""
    assert sanitized[6]["normalized"]["counter_party"] == ""
    assert sanitized[7]["normalized"]["counter_party"] == "企业电子渠道跨行转账手续费收入"
    assert sanitized[8]["normalized"]["counter_party"] == "待报解预算收入（财税库银联网代收）"
    assert sanitized[9]["normalized"]["counter_party"] == ""


def test_bank_statement_record_sanitizer_never_backfills_a_party_from_another_row() -> None:
    records = [
        {
            "raw": {"对方账号与户名": "210401324测试供应商有限公司"},
            "canonical_raw": {
                "counter_account": "210401324",
                "counter_party": "测试供应商有限公司",
            },
            "normalized": {
                "counter_account": "210401324",
                "counter_party": "测试供应商有限公司",
            },
        },
        {
            "raw": {"对方账号与户名": "210401324"},
            "canonical_raw": {"counter_account": "210401324", "counter_party": ""},
            "normalized": {"counter_account": "210401324", "counter_party": ""},
        },
        {
            "raw": {"对方账号与户名": "210401324", "对方户名": "限公司"},
            "canonical_raw": {"counter_account": "210401324", "counter_party": "限公司"},
            "normalized": {"counter_account": "210401324", "counter_party": "限公司"},
        },
    ]

    source_records = copy.deepcopy(records)
    sanitized = _sanitize_bank_records(records)

    assert sanitized[0]["normalized"]["counter_party"] == "测试供应商有限公司"
    assert sanitized[1]["normalized"]["counter_party"] == ""
    assert sanitized[1]["normalized"]["counterparty_status"] == "present"
    assert sanitized[2]["normalized"]["counter_party"] == ""
    assert sanitized[2]["normalized"]["counterparty_status"] == "present"
    assert [row["raw"] for row in sanitized] == [row["raw"] for row in source_records]
    assert [row["canonical_raw"] for row in sanitized] == [row["canonical_raw"] for row in source_records]


def test_bank_statement_counterparty_pollution_allows_one_long_identifier_only() -> None:
    records = [
        {
            "raw": {"对方户名": "供应商统一代码 123456789012345678"},
            "canonical_raw": {"counter_party": "供应商统一代码 123456789012345678"},
            "normalized": {"counter_party": "供应商统一代码 123456789012345678"},
        },
        {
            "raw": {"对方户名": "供应商 12345678 87654321"},
            "canonical_raw": {"counter_party": "供应商 12345678 87654321"},
            "normalized": {"counter_party": "供应商 12345678 87654321"},
        },
    ]

    sanitized = _sanitize_bank_records(records)

    assert sanitized[0]["normalized"]["counter_party"] == "供应商统一代码 123456789012345678"
    assert sanitized[1]["normalized"]["counter_party"] == ""


def test_bank_statement_counterparty_sanitizer_rejects_concatenated_page_marker() -> None:
    record = {
        "raw": {
            "对方账号": "10311101940040251",
            "对方户名": "999999 转存第10页/共29页",
            "对方开户行": "999999",
            "摘要": "转存",
        },
        "canonical_raw": {
            "counter_account": "10311101940040251",
            "counter_party": "999999 转存第10页/共29页",
            "counter_bank_name": "999999",
            "summary": "转存",
        },
        "normalized": {
            "counter_account": "10311101940040251",
            "counter_party": "999999 转存第10页/共29页",
            "counter_party_original": "999999 转存第10页/共29页",
            "counter_bank_name": "999999",
            "summary": "转存",
            "additional_fields": [
                {
                    "name": "对方户名",
                    "field": "counter_party",
                    "value": "999999 转存第10页/共29页",
                }
            ],
        },
    }

    sanitized = _sanitize_bank_records([record])[0]

    assert sanitized["raw"] == record["raw"]
    assert sanitized["canonical_raw"] == record["canonical_raw"]
    assert sanitized["normalized"]["counter_party"] == ""
    assert sanitized["normalized"]["counter_party_original"] == ""
    assert sanitized["normalized"]["additional_fields"][0]["value"] == ""
    assert sanitized["normalized"]["counterparty_status"] == "present"
    exclusions = sanitized["source"]["_delivery_value_exclusions"]
    assert {item["pool"] for item in exclusions} == {"raw", "canonical_raw"}


def test_bank_statement_delivery_does_not_repromote_excluded_counterparty_source_noise() -> None:
    from docmirror.output.normalized_records import additional_business_fields

    record = {
        "raw": {"对方户名": "999999 转存第10页/共29页", "摘要": "转存"},
        "canonical_raw": {
            "counter_party": "999999 转存第10页/共29页",
            "summary": "转存",
        },
        "normalized": {"counter_party": "", "summary": "转存"},
    }
    sanitized = _sanitize_bank_records([record])[0]
    columns = [
        {"key": "counter_party", "label": "对方户名", "type": "string"},
        {"key": "summary", "label": "摘要", "type": "string"},
    ]

    assert additional_business_fields(
        sanitized,
        columns,
        {"counter_party": ["对方户名"]},
    ) == []
    assert sanitized["raw"] == record["raw"]
    assert sanitized["canonical_raw"] == record["canonical_raw"]


def test_bank_statement_delivery_does_not_duplicate_a_reconstructed_disclaimer_fragment() -> None:
    from docmirror.output.normalized_records import additional_business_fields

    first = "本明细仅限于查询账户交易流水使用,在跨行退回、日终冲账等特殊情况下存在后续变动可能,"
    complete = "重要提示:" + first + "若与实际交易不符,以银行对账单为准。"
    record = {
        "raw": {"重要提示": first, "document_disclaimer": complete},
        "canonical_raw": {"statement_disclaimer": complete},
        "normalized": {"statement_disclaimer": complete},
        "source": {},
    }
    _mark_represented_header_delivery_exclusions([record])

    assert additional_business_fields(
        record,
        [{"key": "statement_disclaimer", "label": "流水免责声明", "type": "string"}],
        {},
    ) == []
    assert record["raw"]["重要提示"] == first
    assert record["canonical_raw"]["statement_disclaimer"] == complete
