from __future__ import annotations

from types import SimpleNamespace

from docmirror.plugins.credit_report.enterprise_native.continuation import (
    ACCOUNT_SETTLED_DETAIL_CONTRACT,
    ATTACHMENT_HISTORY_BODY_CONTRACT,
    CLOSED_SUMMARY_BODY_CONTRACT,
    FACILITY_VALUE_CONTRACT,
    EnterpriseContinuationResolver,
)
from docmirror.plugins.credit_report.enterprise_native.extraction import (
    extract_enterprise_accounts_from_tables,
    extract_enterprise_attachment_datasets,
    extract_enterprise_continuation_audit,
    extract_enterprise_public_record_datasets,
    extract_enterprise_public_records_from_tables,
)


def _table(
    table_id: str,
    rows: list[list[str]],
    *,
    bbox: list[float] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        table_id=table_id,
        headers=[],
        rows=[],
        metadata={"raw_rows": rows},
        bbox=bbox or [],
    )


def _result(*pages: tuple[int, list[SimpleNamespace]]) -> SimpleNamespace:
    return SimpleNamespace(
        pages=[
            SimpleNamespace(
                page_number=page_number,
                tables=tables,
                texts=[],
            )
            for page_number, tables in pages
        ]
    )


def _text(content: str, top: float) -> SimpleNamespace:
    return SimpleNamespace(
        content=content,
        bbox=[0.0, top, 100.0, top + 5.0],
    )


def _flow_result(
    *pages: tuple[int, list[SimpleNamespace], list[SimpleNamespace]],
) -> SimpleNamespace:
    return SimpleNamespace(
        pages=[
            SimpleNamespace(
                page_number=page_number,
                texts=texts,
                tables=tables,
            )
            for page_number, texts, tables in pages
        ]
    )


def test_facility_continuation_accepts_only_valid_next_value_row() -> None:
    result = _result(
        (
            1,
            [
                _table(
                    "header",
                    [
                        ["非循环信用额度", "", "", "循环信用额度", "", ""],
                        ["总额", "已用额度", "剩余可用额度", "总额", "已用额度", "剩余可用额度"],
                    ],
                )
            ],
        ),
        (2, [_table("values", [["3000", "3000", "0", "4900", "4900", "0"]])]),
    )
    resolver = EnterpriseContinuationResolver(result)

    match = resolver.following_row(
        resolver.fragments[0],
        FACILITY_VALUE_CONTRACT,
    )

    assert match is not None
    assert match.fragment.table_id == "values"
    assert list(match.row) == ["3000", "3000", "0", "4900", "4900", "0"]


def test_same_column_count_does_not_authorize_unrelated_table_merge() -> None:
    result = _result(
        (
            1,
            [
                _table(
                    "closed_header",
                    [["", "正常类账户数", "关注类账户数", "不良类账户数", "合计"]],
                )
            ],
        ),
        (
            2,
            [
                _table(
                    "unrelated_header",
                    [["类型", "正常类账户数", "关注类账户数", "不良类账户数", "合计"]],
                )
            ],
        ),
    )
    resolver = EnterpriseContinuationResolver(result)

    assert (
        resolver.following_row(
            resolver.fragments[0],
            CLOSED_SUMMARY_BODY_CONTRACT,
        )
        is None
    )
    assert resolver.audit_rows() == [
        {
            "contract": "closed_credit_summary_body",
            "source_table_id": "closed_header",
            "candidate_table_id": "unrelated_header",
            "reason": "new_header",
        }
    ]
    decision = resolver.decision_rows()[-1]
    assert decision["selected"] == "new_table"
    assert {item["kind"] for item in decision["hypotheses"]} == {
        "same_table",
        "new_table",
        "new_section",
    }


def test_scored_boundaries_chain_a_table_across_three_pages() -> None:
    pages = [
        SimpleNamespace(
            page_number=1,
            source_page_number=1,
            width=600,
            height=800,
            texts=[],
            tables=[
                _table(
                    "table_page_1",
                    [
                        ["账户编号", "开立日期", "到期日", "币种", "金额"],
                        ["A123456789012", "2025-01-01", "2025-06-01", "人民币元", "100"],
                    ],
                    bbox=[20, 500, 580, 790],
                )
            ],
        ),
        SimpleNamespace(
            page_number=2,
            source_page_number=2,
            width=600,
            height=800,
            texts=[],
            tables=[
                _table(
                    "table_page_2",
                    [["A123456789013", "2025-01-02", "2025-06-02", "人民币元", "200"]],
                    bbox=[20, 20, 580, 790],
                )
            ],
        ),
        SimpleNamespace(
            page_number=3,
            source_page_number=3,
            width=600,
            height=800,
            texts=[],
            tables=[
                _table(
                    "table_page_3",
                    [["A123456789014", "2025-01-03", "2025-06-03", "人民币元", "300"]],
                    bbox=[20, 20, 580, 400],
                )
            ],
        ),
    ]
    resolver = EnterpriseContinuationResolver(SimpleNamespace(pages=pages))

    assert [fragment.table_id for fragment in resolver.following_fragments(resolver.fragments[0])] == [
        "table_page_2",
        "table_page_3",
    ]
    assert [decision.selected for decision in resolver.boundary_decisions] == [
        "same_table",
        "same_table",
    ]
    assert all(decision.accepted for decision in resolver.boundary_decisions)
    spanning = [entity for entity in resolver.logical_entities if len(entity.pages) == 3]
    assert len(spanning) == 1
    assert spanning[0].unit_ids == (
        "table:table_page_1",
        "table:table_page_2",
        "table:table_page_3",
    )


def test_fragment_validator_scores_rows_below_a_split_header() -> None:
    result = _result(
        (
            1,
            [
                _table(
                    "detail_header",
                    [
                        ["账户编号", "开立日期", "金额"],
                        ["A123456789012", "2025-01-01", "100"],
                    ],
                )
            ],
        ),
        (
            2,
            [
                _table(
                    "split_header_and_body",
                    [
                        ["账户明细", "", ""],
                        ["A123456789013", "2025-01-02", "200"],
                    ],
                )
            ],
        ),
    )
    resolver = EnterpriseContinuationResolver(result)

    matches = resolver.following_fragments(
        resolver.fragments[0],
        candidate_validator=lambda row: len(row) == 3 and row[1].startswith("2025-"),
        context="split_header",
    )

    assert [fragment.table_id for fragment in matches] == ["split_header_and_body"]


def test_single_row_validator_cannot_be_outvoted_by_geometry() -> None:
    pages = [
        SimpleNamespace(
            page_number=1,
            source_page_number=1,
            width=600,
            height=800,
            texts=[],
            tables=[
                _table(
                    "history_header",
                    [["信息报告日期", "余额", "余额变化日期"]],
                    bbox=[20, 500, 580, 790],
                )
            ],
        ),
        SimpleNamespace(
            page_number=2,
            source_page_number=2,
            width=600,
            height=800,
            texts=[],
            tables=[
                _table(
                    "different_record",
                    [["账户编号", "授信机构", "业务种类"]],
                    bbox=[20, 20, 580, 790],
                )
            ],
        ),
    ]
    resolver = EnterpriseContinuationResolver(SimpleNamespace(pages=pages))

    assert (
        resolver.table_continues(
            resolver.fragments[0],
            resolver.fragments[1],
            candidate_validator=lambda row: row[0].startswith("2025-"),
            context="history_row",
        )
        is False
    )


def test_scored_text_boundary_distinguishes_continuation_from_new_section() -> None:
    continuing = SimpleNamespace(
        pages=[
            SimpleNamespace(
                page_number=1,
                source_page_number=1,
                width=600,
                height=800,
                tables=[],
                texts=[_text("本段文字尚未结束并在下一页", 760)],
            ),
            SimpleNamespace(
                page_number=2,
                source_page_number=2,
                width=600,
                height=800,
                tables=[],
                texts=[_text("继续说明相关事项。", 20)],
            ),
        ]
    )
    new_section = SimpleNamespace(
        pages=[
            SimpleNamespace(
                page_number=1,
                source_page_number=1,
                width=600,
                height=800,
                tables=[],
                texts=[_text("上一节已经结束。", 760)],
            ),
            SimpleNamespace(
                page_number=2,
                source_page_number=2,
                width=600,
                height=800,
                tables=[],
                texts=[_text("公共记录明细", 20)],
            ),
        ]
    )

    continuation_decision = EnterpriseContinuationResolver(continuing).boundary_decisions[0]
    section_decision = EnterpriseContinuationResolver(new_section).boundary_decisions[0]

    assert continuation_decision.selected == "same_body_text"
    assert continuation_decision.accepted is True
    assert section_decision.selected == "new_section"
    assert section_decision.accepted is False


def test_nonadjacent_and_distant_tables_are_never_skipped_into_a_merge() -> None:
    result = _result(
        (
            1,
            [
                _table(
                    "closed_header",
                    [["", "正常类账户数", "关注类账户数", "不良类账户数", "合计"]],
                ),
                _table("intervening", [["说明", "值"]]),
            ],
        ),
        (3, [_table("plausible_but_distant", [["合计", "1", "0", "0", "1"]])]),
    )
    resolver = EnterpriseContinuationResolver(result)

    assert (
        resolver.following_row(
            resolver.fragments[0],
            CLOSED_SUMMARY_BODY_CONTRACT,
        )
        is None
    )
    assert resolver.audit_rows()[0]["candidate_table_id"] == "intervening"
    assert resolver.audit_rows()[0]["reason"] == "column_shape"


def test_account_and_history_continuation_contracts_require_business_shapes() -> None:
    result = _result(
        (
            1,
            [
                _table(
                    "account_source",
                    [["D10123320H000170060110009", "机构", "贷款", "2022-02-11"]],
                ),
                _table(
                    "account_detail",
                    [["028463", "2023-01-12", "正常", "2023-01-12", "", "正常还款", "", "见附件"]],
                ),
                _table(
                    "history_source",
                    [["信息报告日期", "余额", "余额变化日期", "五级分类", "认定日期", "逾期总额", "逾期本金"]],
                ),
                _table(
                    "history_body",
                    [
                        [
                            "",
                            "逾期月数",
                            "最近一次约定还款日期",
                            "最近一次应还总额",
                            "最近一次实际还款日期",
                            "最近一次实还总额",
                            "最近一次还款形式",
                        ]
                    ],
                ),
            ],
        )
    )
    resolver = EnterpriseContinuationResolver(result)

    account_match = resolver.following_row(
        resolver.fragments[0],
        ACCOUNT_SETTLED_DETAIL_CONTRACT,
    )
    history_match = resolver.following_row(
        resolver.fragments[2],
        ATTACHMENT_HISTORY_BODY_CONTRACT,
    )

    assert account_match is not None
    assert account_match.fragment.table_id == "account_detail"
    assert history_match is not None
    assert history_match.fragment.table_id == "history_body"


def test_enterprise_accounts_keep_short_ids_and_append_settled_page_suffix() -> None:
    result = _result(
        (
            1,
            [
                _table(
                    "accounts_page_1",
                    [
                        ["短期借款", "已结清 共3笔", "", "", "", "", "", ""],
                        [
                            "账户编号",
                            "授信机构",
                            "业务种类",
                            "开立日期",
                            "到期日",
                            "",
                            "币种",
                            "借款金额",
                        ],
                        [
                            "",
                            "关闭日期",
                            "五级分类",
                            "最后一次还款日期",
                            "",
                            "最后一次还款形式",
                            "",
                            "历史表现",
                        ],
                        [
                            "2142019656",
                            "招商银行上海宝山支行",
                            "流动资金贷款",
                            "2020-12-02",
                            "2021-06-01",
                            "",
                            "人民币元",
                            "490",
                        ],
                        ["", "2021-06-01", "正常", "2021-06-01", "", "正常还款", "", "见附件"],
                        [
                            "2142018846",
                            "招商银行上海宝山支行",
                            "流动资金贷款",
                            "2020-06-24",
                            "2020-12-23",
                            "",
                            "人民币元",
                            "200",
                        ],
                        ["", "2020-12-01", "正常", "2020-12-01", "", "正常还款", "", "见附件"],
                        [
                            "D10123320H000170060110009",
                            "宁波银行上海分行",
                            "流动资金贷款",
                            "2022-02-11",
                            "2023-01-13",
                            "",
                            "人民币元",
                            "31",
                        ],
                    ],
                )
            ],
        ),
        (
            2,
            [
                _table(
                    "accounts_page_2",
                    [["028463", "2023-01-12", "正常", "2023-01-12", "", "正常还款", "", "见附件"]],
                )
            ],
        ),
    )

    accounts = extract_enterprise_accounts_from_tables(result)

    assert [record["account_identifier"] for record in accounts] == [
        "2142019656",
        "2142018846",
        "D10123320H000170060110009028463",
    ]
    assert [record["loan_amount"] for record in accounts] == [490, 200, 31]


def test_enterprise_account_recovers_header_only_page_then_shifted_settled_rows() -> None:
    result = _result(
        (
            5,
            [
                _table(
                    "settled_header_only",
                    [
                        ["短期借款", "已结清 共1笔", "", "", "", "", ""],
                        [
                            "账户编号",
                            "授信机构",
                            "业务种类",
                            "开立日期",
                            "到期日",
                            "币种",
                            "借款金额",
                        ],
                    ],
                )
            ],
        ),
        (
            6,
            [
                _table(
                    "settled_shifted_body",
                    [
                        ["", "关闭日期", "五级分类", "最后一次还款日期", "", "最后一次还款形式", "", "历史表现"],
                        [
                            "B10611000H00011811138121928001",
                            "中信银行",
                            "流动资金贷款",
                            "2024-08-10",
                            "2025-08-09",
                            "",
                            "人民币元",
                            "1000",
                        ],
                        ["", "2025-08-09", "正常", "2025-08-09", "", "正常还款", "", "见附件"],
                    ],
                )
            ],
        ),
    )

    accounts = extract_enterprise_accounts_from_tables(result)

    assert len(accounts) == 1
    assert accounts[0]["account_identifier"] == "B10611000H00011811138121928001"
    assert accounts[0]["account_status"] == "settled"
    assert accounts[0]["loan_amount"] == 1000
    assert accounts[0]["currency"] == "CNY"
    assert accounts[0]["close_date"] == "2025-08-09"
    assert accounts[0]["five_tier_class"] == "正常"
    assert [ref["page"] for ref in accounts[0]["source_refs"]] == [5, 6, 6]


def test_enterprise_account_maps_issuance_form_and_validates_legacy_dates() -> None:
    result = _result(
        (
            1,
            [
                _table(
                    "active_account",
                    [
                        ["短期借款", "未结清 共1笔", "", "", "", "", "", ""],
                        [
                            "账户编号",
                            "授信机构",
                            "业务种类",
                            "开立日期",
                            "到期日",
                            "币种",
                            "借款金额",
                            "发放形式",
                        ],
                        [
                            "ABC123",
                            "示例银行",
                            "流动资金贷款",
                            "1999年1月2日",
                            "2000/02/03",
                            "人民币元",
                            "100",
                            "新增",
                        ],
                    ],
                )
            ],
        ),
    )

    accounts = extract_enterprise_accounts_from_tables(result)

    assert len(accounts) == 1
    assert accounts[0]["open_date"] == "1999-01-02"
    assert accounts[0]["due_date"] == "2000-02-03"
    assert accounts[0]["issuance_form"] == "新增"


def test_attachment_history_recovers_both_cross_page_split_shapes() -> None:
    primary_header = [
        "信息报告日期",
        "余额",
        "余额变化日期",
        "五级分类",
        "五级分类认定日期",
        "逾期总额",
        "逾期本金",
    ]
    secondary_header = [
        "",
        "逾期月数",
        "最近一次约定还款日期",
        "最近一次应还总额",
        "最近一次实际还款日期",
        "最近一次实还总额",
        "最近一次还款形式",
    ]
    result = _flow_result(
        (
            1,
            [
                _text("附件", 0),
                _text("短期借款的历史表现", 10),
                _text(
                    "1.已结清账户编号：B11313900H0001216450100300036861\n"
                    "授信机构：兴业银行股份有限公司上海五角场支行\n"
                    "业务种类：流动资金贷款",
                    20,
                ),
            ],
            [_table("history_header_only", [primary_header], bbox=[0, 100, 100, 120])],
        ),
        (
            2,
            [],
            [
                _table(
                    "history_header_continuation",
                    [
                        secondary_header,
                        ["2023-03-30", "0", "2023-03-29", "正常", "2022-03-30", "0", "0"],
                        ["", "0", "2023-03-29", "200", "2023-03-29", "200", "正常还款"],
                    ],
                    bbox=[0, 0, 100, 50],
                )
            ],
        ),
        (
            3,
            [
                _text(
                    "2.已结清账户编号：0231000272200408000100\n"
                    "授信机构：中国邮政储蓄银行上海浦东新区分行\n"
                    "业务种类：其他贷款",
                    0,
                )
            ],
            [
                _table(
                    "history_primary_only",
                    [
                        primary_header,
                        secondary_header,
                        ["2020-10-16", "0", "2020-10-16", "正常", "--", "--", "--"],
                    ],
                    bbox=[0, 100, 100, 140],
                )
            ],
        ),
        (
            4,
            [],
            [
                _table(
                    "history_secondary_only",
                    [["", "--", "--", "--", "2020-10-16", "480", "正常还款"]],
                    bbox=[0, 0, 100, 20],
                )
            ],
        ),
    )

    datasets = extract_enterprise_attachment_datasets(result)
    histories = datasets["enterprise_credit_supplement"]

    assert [(record["account_identifier"], record["report_date"]) for record in histories] == [
        ("B11313900H0001216450100300036861", "2023-03-30"),
        ("0231000272200408000100", "2020-10-16"),
    ]


def test_attachment_detail_inherits_classification_and_maps_active_fields() -> None:
    result = _flow_result(
        (
            1,
            [
                _text("附件", 0),
                _text("银行承兑汇票和信用证的信贷明细", 10),
                _text(
                    "1.未结清业务\n授信机构：示例银行\n业务种类：银行承兑汇票\n五级分类：正常",
                    20,
                ),
                _text("贴现的信贷明细", 100),
                _text(
                    "1.未结清账户编号：D10123320H00012025070160898220\n"
                    "授信机构：宁波银行股份有限公司上海分行\n"
                    "业务种类：有追索权的银行承兑汇票贴现",
                    110,
                ),
            ],
            [
                _table(
                    "business_details",
                    [
                        ["信贷明细", "", "", "", ""],
                        ["账户编号", "开立日期", "到期日", "币种", "金额"],
                        ["20201209791117834", "2020-12-10", "2021-06-10", "人民币元", "10"],
                    ],
                    bbox=[0, 50, 100, 90],
                ),
                _table(
                    "active_discount",
                    [
                        ["信贷明细", "", "", "", "", "", "", ""],
                        [
                            "开户日期",
                            "到期日",
                            "币种",
                            "贴现金额",
                            "担保方式",
                            "五级分类",
                            "授信协议编号",
                            "信息报告日期",
                        ],
                        [
                            "2025-07-01",
                            "2025-12-25",
                            "人民币元",
                            "5.68",
                            "信用/无担保",
                            "正常",
                            "--",
                            "2025-07-01",
                        ],
                    ],
                    bbox=[0, 140, 100, 180],
                ),
            ],
        )
    )

    details = extract_enterprise_attachment_datasets(result)["enterprise_attachment_credit_details"]
    inherited, active_discount = details

    assert inherited["five_tier_class"] == "正常"
    assert inherited["five_tier_class_source"] == "parent_attachment_heading"
    assert len(inherited["source_refs"]) == 2
    assert active_discount["guarantee_type"] == "信用/无担保"
    assert active_discount["snapshot_date"] == "2025-07-01"
    assert active_discount["credit_agreement_identifier"] == ""
    assert active_discount["credit_agreement_status"] == "not_reported"
    assert active_discount["five_tier_class_source"] == "detail_table"


def test_attachment_detail_pairs_double_headers_with_double_data_rows() -> None:
    result = _flow_result(
        (
            76,
            [
                _text("附件", 0),
                _text("银行承兑汇票和信用证的信贷明细", 10),
                _text(
                    "1.未结清业务\n授信机构：宣城皖南农村商业银行股份有限公司\n业务种类：银行承兑汇票\n五级分类：正常",
                    20,
                ),
            ],
            [
                _table(
                    "double_header_guarantees",
                    [
                        ["账户编号", "开立日期", "到期日", "币种", "金额", "反担保方式"],
                        ["", "保证金比例", "余额", "风险敞口", "授信协议编号", "信息报告日期"],
                        [
                            "G10423771H00065402377100013202503281003069991742282",
                            "2025-03-28",
                            "2025-09-27",
                            "人民币元",
                            "2000",
                            "信用/无担保/保证金",
                        ],
                        [
                            "",
                            "50%",
                            "2000",
                            "1000",
                            "G10423771H00065879024720250329",
                            "2025-03-29",
                        ],
                        [
                            "G10323310H0001DLC8022025000052",
                            "2025-05-13",
                            "2025-05-31",
                            "人民币元",
                            "2500",
                            "信用/无担保/保证金",
                        ],
                        ["", "50%", "2500", "--", "--", "2025-05-13"],
                    ],
                    bbox=[0, 50, 100, 140],
                )
            ],
        )
    )

    details = extract_enterprise_attachment_datasets(result)["enterprise_attachment_credit_details"]
    reported, not_reported = details

    assert reported["guarantee_type"] == ""
    assert reported["counter_guarantee_type"] == "信用/无担保/保证金"
    assert reported["deposit_ratio"] == 0.5
    assert reported["balance"] == 2000
    assert reported["risk_exposure_amount"] == 1000
    assert reported["credit_agreement_identifier"] == "G10423771H00065879024720250329"
    assert reported["credit_agreement_status"] == "reported"
    assert reported["snapshot_date"] == "2025-03-29"
    assert len(reported["source_refs"]) == 3
    assert not_reported["deposit_ratio"] == 0.5
    assert not_reported["balance"] == 2500
    assert not_reported["risk_exposure_amount"] is None
    assert not_reported["credit_agreement_identifier"] == ""
    assert not_reported["credit_agreement_status"] == "not_reported"
    assert not_reported["snapshot_date"] == "2025-05-13"


def test_attachment_detail_keeps_schema_across_headerless_final_page() -> None:
    result = _flow_result(
        (
            12,
            [
                _text("附件", 0),
                _text("银行承兑汇票和信用证的信贷明细", 10),
                _text("2.已结清业务", 20),
            ],
            [
                _table(
                    "settled_acceptance_header_and_body",
                    [
                        ["账户编号", "开立日期", "到期日", "币种", "金额", "关闭日期", "垫款标志"],
                        [
                            "B11115840H0001B24010140000001",
                            "2024-01-01",
                            "2024-06-01",
                            "人民币元",
                            "100",
                            "2024-06-01",
                            "否",
                        ],
                        [
                            "B11115840H0001B24010240000002",
                            "2024-01-02",
                            "2024-06-02",
                            "人民币元",
                            "200",
                            "2024-06-02",
                            "否",
                        ],
                    ],
                )
            ],
        ),
        (
            13,
            [],
            [
                _table(
                    "settled_acceptance_headerless_body",
                    [
                        [
                            "B11115840H0001B24010340000003",
                            "2024-01-03",
                            "2024-06-03",
                            "人民币元",
                            "300",
                            "2024-06-03",
                            "否",
                        ]
                    ],
                )
            ],
        ),
    )

    datasets = extract_enterprise_attachment_datasets(result)
    details = datasets["enterprise_attachment_credit_details"]

    assert [record["account_identifier"] for record in details] == [
        "B11115840H0001B24010140000001",
        "B11115840H0001B24010240000002",
        "B11115840H0001B24010340000003",
    ]
    assert [record["source_page"] for record in details] == [12, 12, 13]
    assert all(record["account_status"] == "settled" for record in details)


def test_public_records_expose_typed_fields_and_raw_attributes() -> None:
    result = _result(
        (
            6,
            [
                _table(
                    "housing_fund",
                    [
                        [
                            "统计年月：2021-06",
                            "初缴年月：2021-02",
                            "职工人数：3",
                            "缴费基数（元）：9,040",
                            "最近一次缴费日期：2021-06-28",
                            "缴至年月：2021-06",
                            "缴费状态：正常缴费",
                            "累计欠费金额（元）：0",
                        ]
                    ],
                ),
                _table(
                    "license",
                    [
                        ["许可部门", "许可类型", "许可日期", "截止日期", "许可内容"],
                        ["南京市行政审批局", "登记", "2024-01-11", "2099-12-31", "登字"],
                    ],
                ),
            ],
        )
    )

    housing, license_record = extract_enterprise_public_records_from_tables(result)

    assert housing["record_type"] == "social_security_payment"
    assert housing["statistics_month"] == "2021-06"
    assert housing["employee_count"] == 3
    assert housing["contribution_base"] == 9040
    assert housing["last_contribution_date"] == "2021-06-28"
    assert housing["cumulative_arrears"] == 0
    assert housing["attributes"]["payment_status"] == "正常缴费"
    assert housing["details"]["初缴年月"] == "2021-02"
    assert license_record["licensing_authority"] == "南京市行政审批局"
    assert license_record["license_date"] == "2024-01-11"
    assert license_record["license_content"] == "登字"


def test_public_records_are_projected_as_lossless_source_chart_datasets() -> None:
    result = _result(
        (
            6,
            [
                _table(
                    "housing_fund",
                    [
                        [
                            "统计年月：2021-06",
                            "初缴年月：2021-02",
                            "职工人数：3",
                            "缴费基数（元）：9,040",
                            "最近一次缴费日期：2021-06-28",
                            "缴至年月：2021-06",
                            "缴费状态：正常缴费",
                            "累计欠费金额（元）：0",
                        ]
                    ],
                ),
                _table(
                    "license",
                    [
                        ["许可部门", "许可类型", "许可日期", "截止日期", "许可内容"],
                        ["南京市行政审批局", "登记", "2024-01-11", "2099-12-31", "登字"],
                    ],
                ),
            ],
        ),
        (
            7,
            [
                _table(
                    "certification",
                    [
                        ["认证部门", "认证类型", "认证日期", "截止日期", "认证内容"],
                        ["国家税务总局", "纳税信用A级纳税人", "--", "2030-12-31", "2024年度"],
                    ],
                )
            ],
        ),
    )

    datasets = extract_enterprise_public_record_datasets(result)

    assert list(datasets) == [
        "enterprise_public_social_security_payment_records",
        "enterprise_public_license_records",
        "enterprise_public_certification_records",
    ]
    assert datasets["enterprise_public_social_security_payment_records"][0]["contribution_base"] == 9040
    license_record = datasets["enterprise_public_license_records"][0]
    assert set(license_record) == {
        "public_record_id",
        "sequence",
        "licensing_authority",
        "license_type",
        "license_date",
        "license_expiry_date",
        "license_content",
        "source_page",
        "source_table_id",
        "source",
        "source_refs",
        "confidence",
    }
    assert license_record["licensing_authority"] == "南京市行政审批局"
    assert license_record["license_type"] == "登记"
    assert license_record["license_date"] == "2024-01-11"
    assert license_record["license_expiry_date"] == "2099-12-31"
    assert license_record["license_content"] == "登字"
    assert datasets["enterprise_public_certification_records"][0]["certification_date"] == "--"


def test_attachment_detail_audit_reconciles_reported_settled_business_count() -> None:
    result = _result((1, []))
    datasets = {
        "enterprise_current_credit_summary": [],
        "enterprise_closed_credit_summary": [
            {
                "transaction_group": "银行承兑汇票和信用证",
                "business_category": "银行承兑汇票",
                "total_account_count": 3,
                "is_total": False,
            }
        ],
        "enterprise_repayment_responsibility_summary": [],
        "repayment_liability_records": [],
        "enterprise_attachment_accounts": [
            {
                "attachment_record_type": "business",
                "account_status": "settled",
                "business_category": "银行承兑汇票和信用证",
            }
        ],
        "enterprise_attachment_credit_details": [
            {
                "account_status": "settled",
                "business_category": "银行承兑汇票和信用证",
            },
            {
                "account_status": "settled",
                "business_category": "银行承兑汇票和信用证",
            },
        ],
    }

    audits = extract_enterprise_continuation_audit(result, datasets=datasets)
    detail_audit = next(row for row in audits if row["continuation_family"] == "attachment_credit_detail")

    assert detail_audit["expected_record_count"] == 3
    assert detail_audit["extracted_record_count"] == 2
    assert detail_audit["unresolved_record_count"] == 1
    assert detail_audit["reconciliation_status"] == "unresolved"


def test_continuation_audit_distinguishes_unexpected_records_without_mutating_input() -> None:
    result = _result((1, []))
    datasets = {
        "enterprise_current_credit_summary": [{"current_summary_id": "unexpected"}],
        "enterprise_closed_credit_summary": [],
        "enterprise_repayment_responsibility_summary": [],
        "repayment_liability_records": [],
        "enterprise_attachment_accounts": [],
    }

    audits = extract_enterprise_continuation_audit(result, datasets=datasets)

    assert datasets == {
        "enterprise_current_credit_summary": [{"current_summary_id": "unexpected"}],
        "enterprise_closed_credit_summary": [],
        "enterprise_repayment_responsibility_summary": [],
        "repayment_liability_records": [],
        "enterprise_attachment_accounts": [],
    }
    assert audits[0]["unresolved_record_count"] == 0
    assert audits[0]["unexpected_record_count"] == 1
    assert audits[0]["reconciliation_status"] == "unresolved"
