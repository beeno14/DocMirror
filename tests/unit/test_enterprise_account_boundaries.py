# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

from docmirror.models.entities.parse_result import PageContent, TableBlock, TextBlock
from docmirror.plugins.credit_report.enterprise_native.extraction import (
    extract_enterprise_accounts_from_tables,
)
from docmirror.plugins.credit_report.enterprise_native.ir import (
    build_canonical_enterprise_document,
)


def _table(table_id: str, rows: list[list[str]]) -> TableBlock:
    return TableBlock(table_id=table_id, metadata={"raw_rows": rows})


def test_active_account_card_closes_before_adjacent_facility_table() -> None:
    result = SimpleNamespace(
        pages=[
            PageContent(
                page_number=1,
                texts=[TextBlock(content="企业信用报告（自主查询版）\n信贷记录明细")],
                tables=[
                    _table(
                        "account",
                        [
                            ["循环透支", "", "", "", "共1笔", "", "", ""],
                            [
                                "账户编号",
                                "授信机构",
                                "业务种类",
                                "开立日期",
                                "到期日",
                                "币种",
                                "信用额度",
                                "发放形式",
                            ],
                            [
                                "",
                                "担保方式",
                                "余额",
                                "五级分类",
                                "逾期总额",
                                "逾期本金",
                                "逾期月数",
                                "最近一次还款日期",
                            ],
                            [
                                "",
                                "最近一次还款总额",
                                "最近一次还款形式",
                                "剩余还款月数",
                                "特定交易提示",
                                "授信协议编号",
                                "历史表现",
                                "信息报告日期",
                            ],
                            [
                                "N101W5810H0008321",
                                "甲银行",
                                "法人账户透支",
                                "2014-08-01",
                                "2017-07-31",
                                "人民币",
                                "100",
                                "新增",
                            ],
                            ["", "抵押", "50", "正常", "0", "0", "0", "2015-10-01"],
                            [
                                "",
                                "3",
                                "正常还款",
                                "22",
                                "展期",
                                "N101W1100H0019025015",
                                "见附件",
                                "2015-10-02",
                            ],
                        ],
                    ),
                    _table(
                        "facility",
                        [
                            ["授信信息", "", "", "共1笔", "", ""],
                            [
                                "授信协议编号",
                                "授信机构",
                                "授信额度类型",
                                "额度循环标志",
                                "生效日期",
                                "到期日",
                                "信息报告日期",
                            ],
                            ["", "币种", "授信额度", "已用额度", "授信限额", "授信限额编号", ""],
                            [
                                "N101W1100H0019025015",
                                "甲银行",
                                "贷款",
                                "是",
                                "2014-01-01",
                                "2017-12-31",
                                "2014-01-02",
                            ],
                            ["", "人民币", "500", "100", "400", "N101W1100H0051239548", ""],
                        ],
                    ),
                ],
            )
        ],
        confidence=1.0,
    )

    accounts = extract_enterprise_accounts_from_tables(
        build_canonical_enterprise_document(result)
    )

    assert len(accounts) == 1
    account = accounts[0]
    assert account["balance"] == 50
    assert account["five_tier_class"] == "正常"
    assert account["overdue_total"] == 0
    assert account["overdue_principal"] == 0
    assert account["overdue_months"] == 0
    assert account["remaining_periods"] == 22
    assert account["credit_agreement_identifier"] == "N101W1100H0019025015"
    assert account["special_transaction"] == "展期"
