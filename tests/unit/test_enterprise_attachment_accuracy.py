from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

from docmirror.plugins.credit_report.enterprise_native.extraction import (
    _attachment_label,
    _attachment_spatial_heading_contexts,
    extract_enterprise_attachment_datasets,
)


def _text(content: str, top: float) -> SimpleNamespace:
    return SimpleNamespace(content=content, bbox=[0.0, top, 100.0, top + 5.0])


def _table(table_id: str, rows: list[list[str]], top: float) -> SimpleNamespace:
    return SimpleNamespace(
        table_id=table_id,
        headers=[],
        rows=[],
        metadata={"raw_rows": rows},
        bbox=[0.0, top, 100.0, top + 20.0],
    )


def _result(*pages: tuple[int, list[SimpleNamespace], list[SimpleNamespace]]) -> SimpleNamespace:
    return SimpleNamespace(
        pages=[
            SimpleNamespace(page_number=number, texts=texts, tables=tables)
            for number, texts, tables in pages
        ]
    )


_RECOVERY_HEADER = [
    "信息报告日期",
    "余额",
    "余额变化日期",
    "五级分类",
    "五级分类认定日期",
    "最近一次实际还款日期",
    "最近一次实还总额",
    "最近一次还款形式",
]
_LOAN_PRIMARY_HEADER = [
    "信息报告日期",
    "余额",
    "余额变化日期",
    "五级分类",
    "五级分类认定日期",
    "逾期总额",
    "逾期本金",
]
_LOAN_SECONDARY_HEADER = [
    "",
    "逾期月数",
    "最近一次约定还款日期",
    "最近一次应还总额",
    "最近一次实际还款日期",
    "最近一次实还总额",
    "最近一次还款形式",
]


def test_attachment_retains_every_canonical_history_shape_across_page_split() -> None:
    result = _result(
        (
            1,
            [
                _text("附件", 0),
                _text("被追偿业务的历史表现", 5),
                _text(
                    "1.未结清账户编号：RECOVERY0001 "
                    "授信机构：甲银行业务种类：资产处置",
                    10,
                ),
                _text(
                    "2.已结清账户编号：RECOVERY0002 "
                    "授信机构：乙银行业务种类：资产处置",
                    100,
                ),
                _text("中长期借款的历史表现", 200),
                _text(
                    "1.未结清账户编号：LONGTERM0001 "
                    "授信机构：丙银行业务种类：固定资产贷款",
                    210,
                ),
                _text(
                    "2.已结清账户编号：LONGTERM0002 "
                    "授信机构：丁银行业务种类：流动资金贷款",
                    300,
                ),
                _text("循环透支的历史表现", 400),
                _text(
                    "1.未结清账户编号：REVOLVING001 "
                    "授信机构：戊银行业务种类：法人账户透支",
                    410,
                ),
            ],
            [
                _table(
                    "recovery_active",
                    [
                        _RECOVERY_HEADER,
                        ["2024-04-30", "40", "2024-04-29", "可疑", "2024-01-01", "2024-04-29", "5", "以资抵债"],
                        ["2024-03-31", "45", "2024-03-30", "可疑", "2024-01-01", "2024-03-30", "5", "以资抵债"],
                    ],
                    20,
                ),
                _table(
                    "recovery_settled",
                    [
                        _RECOVERY_HEADER,
                        ["2024-02-29", "0", "2024-02-28", "次级", "2024-01-01", "2024-02-28", "20", "诉讼追偿"],
                        ["2024-01-31", "20", "2024-01-30", "次级", "2024-01-01", "2024-01-30", "8", "以资抵债"],
                    ],
                    110,
                ),
                _table(
                    "long_active",
                    [
                        _LOAN_PRIMARY_HEADER,
                        _LOAN_SECONDARY_HEADER,
                        ["2024-06-30", "50", "2024-06-29", "正常", "2024-01-01", "0", "0"],
                        ["", "0", "2024-06-29", "20", "2024-06-29", "30", "正常还款"],
                    ],
                    220,
                ),
                _table(
                    "long_settled",
                    [
                        _LOAN_PRIMARY_HEADER,
                        _LOAN_SECONDARY_HEADER,
                        ["2024-05-31", "0", "2024-05-30", "--", "2024-01-01", "0", "0"],
                        ["", "0", "2024-05-30", "80", "2024-05-30", "80", "正常还款"],
                    ],
                    310,
                ),
                _table(
                    "revolving_split_header",
                    [["信息报告", "余额", "余额变化", "五级分类", "五级分类", "逾期总额", "逾期本金", "逾期月数"]],
                    420,
                ),
            ],
        ),
        (
            2,
            [
                _text(
                    "2.已结清账户编号：REVOLVING002 "
                    "授信机构：己银行业务种类：法人账户透支",
                    100,
                )
            ],
            [
                _table(
                    "revolving_split_body",
                    [
                        ["日期", "", "日期", "", "", "认定日期", "", "", "", ""],
                        ["", "最近一次约定还款日期", "最近一次应还总额", "", "最近一次实际还款日期", "最近一次实还总额", "", "最近一次还款形式", "剩余还款月数", ""],
                        ["2024-08-31", "50", "2024-08-30", "", "正常", "2024-08-01", "", "0", "0", "0"],
                        ["", "2024-08-30", "20", "", "2024-08-30", "50", "", "正常还款", "3", ""],
                        ["2024-07-31", "100", "2024-07-30", "", "正常", "2024-07-01", "", "0", "0", "0"],
                        ["", "2024-07-30", "20", "", "2024-07-30", "20", "", "正常还款", "4", ""],
                    ],
                    0,
                ),
                _table(
                    "revolving_settled",
                    [
                        _LOAN_PRIMARY_HEADER + ["逾期月数"],
                        ["", "最近一次约定还款日期", "最近一次应还总额", "最近一次实际还款日期", "最近一次实还总额", "最近一次还款形式", "剩余还款月数", ""],
                        ["2024-06-30", "0", "2024-06-29", "正常", "2024-01-01", "0", "0", "0"],
                        ["", "2024-06-29", "100", "2024-06-29", "100", "正常还款", "0", ""],
                    ],
                    120,
                ),
            ],
        ),
    )

    histories = extract_enterprise_attachment_datasets(result)["enterprise_credit_supplement"]

    assert len(histories) == 9
    assert Counter(record["business_category"] for record in histories) == {
        "被追偿业务": 4,
        "中长期借款": 2,
        "循环透支": 3,
    }
    assert [record["report_date"] for record in histories[-3:]] == [
        "2024-08-31",
        "2024-07-31",
        "2024-06-30",
    ]
    assert histories[0]["actual_repayment_amount"] == 5
    assert histories[0]["scheduled_repayment_date"] == ""
    assert histories[0]["overdue_total"] is None


def test_attachment_details_retain_row_institution_and_heading_business_type_across_split() -> None:
    result = _result(
        (
            1,
            [
                _text("附件", 0),
                _text("银行承兑汇票和信用证的信贷明细", 5),
                _text("1.未结清业务业务种类：银行承兑汇票五级分类：正常", 10),
            ],
            [
                _table(
                    "detail_header",
                    [["账户编号", "授信机构", "开立日期", "到期日", "币种", "金额", "反担保方式"]],
                    20,
                )
            ],
        ),
        (
            2,
            [],
            [
                _table(
                    "detail_body",
                    [
                        ["", "保证金比例", "余额", "风险敞口", "授信协议编号", "信息报告日期", ""],
                        ["DETAILACCOUNT01", "甲银行北京分行", "2024-01-01", "2024-12-31", "人民币", "20", "抵押"],
                        ["", "20%", "20", "0", "AGREEMENT001", "2024-01-02", ""],
                        ["DETAILACCOUNT02", "甲银行上海分行", "2024-02-01", "2025-01-31", "人民币", "100", "抵押"],
                        ["", "20%", "100", "0", "AGREEMENT002", "2024-02-02", ""],
                    ],
                    0,
                )
            ],
        ),
    )

    details = extract_enterprise_attachment_datasets(result)["enterprise_attachment_credit_details"]

    assert len(details) == 2
    assert [record["institution"] for record in details] == [
        "甲银行北京分行",
        "甲银行上海分行",
    ]
    assert {record["business_type"] for record in details} == {"银行承兑汇票"}
    assert all(record["institution"] for record in details)
    assert all(record["business_type"] for record in details)


def test_attachment_reconstructs_split_heading_context_and_propagates_it_to_children() -> None:
    result = _result(
        (
            1,
            [
                _text("附件", 0),
                _text("中长期借款的历史表现", 5),
                _text("1.未结清账户编号：SPLITHEADING01", 10),
                _text("授信机构：", 11),
                _text("甲银行股份", 12),
                _text("有限公司", 13),
                _text("业务种类：", 14),
                _text("流动资金", 15),
                _text("贷款", 16),
            ],
            [
                _table(
                    "history",
                    [
                        _LOAN_PRIMARY_HEADER,
                        _LOAN_SECONDARY_HEADER,
                        ["2025-01-31", "10", "2025-01-30", "正常", "2025-01-01", "0", "0"],
                        ["", "0", "2025-01-30", "5", "2025-01-30", "5", "正常还款"],
                    ],
                    20,
                )
            ],
        )
    )

    datasets = extract_enterprise_attachment_datasets(result)
    context = datasets["enterprise_attachment_accounts"][0]
    history = datasets["enterprise_credit_supplement"][0]

    assert context["institution"] == "甲银行股份有限公司"
    assert context["business_type"] == "流动资金贷款"
    assert history["institution"] == context["institution"]
    assert history["business_type"] == context["business_type"]


def test_next_numbered_attachment_section_does_not_extend_business_type() -> None:
    result = _result(
        (
            1,
            [
                _text("附件", 0),
                _text("（二）短期借款的历史表现", 5),
                _text(
                    "1.未结清账户编号：SECTIONBOUNDARY01"
                    "授信机构：甲银行业务种类：流动资金贷款",
                    10,
                ),
                _text("（三）循环透支的", 20),
                _text("历史表现", 21),
            ],
            [],
        )
    )

    contexts = extract_enterprise_attachment_datasets(result)["enterprise_attachment_accounts"]

    assert len(contexts) == 1
    assert contexts[0]["business_category"] == "短期借款"
    assert contexts[0]["business_type"] == "流动资金贷款"


def test_attachment_label_stops_only_at_known_numbered_subsection_prefixes() -> None:
    institution = "中国银行股份有限公司上海市浦东分行公司金融二部"

    assert _attachment_label(
        f"授信机构：{institution}（二）短期借款的",
        "授信机构",
    ) == institution
    assert _attachment_label("五级分类：正常（五）银行保函及其他业务的", "五级分类") == "正常"
    assert _attachment_label(
        "授信机构：招商银行股份有限公司上海分行票据中心（待注销）",
        "授信机构",
    ).endswith("（待注销）")


def test_attachment_geometry_matches_column_major_heading_blocks_to_their_rows() -> None:
    def segment(segment_id: str, text: str, bbox: list[float]) -> dict[str, object]:
        return {
            "id": segment_id,
            "source_id": f"source:{segment_id}",
            "kind": "text",
            "source_page": 10,
            "bbox": bbox,
            "text": text,
        }

    # This is the native ordering seen in the large reports: one anchor, then
    # several middle-column institutions, then right-column products, while
    # the second anchor lives in a different connected component.
    parse_result = SimpleNamespace(
        components=(
            SimpleNamespace(
                segments=(
                    segment("a1", "1.已结清账户编号：ACCOUNT000001", [40, 100, 205, 132]),
                    segment("i1", "授信机构：甲银行", [214, 100, 370, 121]),
                    segment("i2", "授信机构：乙银行", [214, 300, 370, 321]),
                    segment("b1", "业务种类：流动资金贷款", [385, 100, 500, 111]),
                    segment("b2", "业务种类：固定资产贷款", [385, 300, 500, 311]),
                )
            ),
            SimpleNamespace(
                segments=(
                    segment("a2", "2.已结清账户编号：ACCOUNT000002", [40, 300, 205, 332]),
                )
            ),
        )
    )

    contexts = _attachment_spatial_heading_contexts(parse_result)

    first = contexts[(10, "account", "settled", 1, "ACCOUNT000001")][0]
    second = contexts[(10, "account", "settled", 2, "ACCOUNT000002")][0]
    assert (first["institution"], first["business_type"]) == ("甲银行", "流动资金贷款")
    assert (second["institution"], second["business_type"]) == ("乙银行", "固定资产贷款")
