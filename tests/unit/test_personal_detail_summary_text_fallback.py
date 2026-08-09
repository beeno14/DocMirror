# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from docmirror.plugins.credit_report.personal_detail_scanned.native_extraction import (
    _extract_summary_datasets,
)
from docmirror.plugins.credit_report.personal_detail_scanned.summary_fallback import (
    decode_credit_business_overview_text_line,
    decode_credit_business_overview_text_lines,
    is_credit_business_overview_text_header,
)


def test_exact_text_header_and_first_lin_overview_row_are_decoded() -> None:
    assert is_credit_business_overview_text_header("业务类型 账户数 首笔业务发放月份")
    assert decode_credit_business_overview_text_line(
        "贷款 信用卡 个人住房贷款 2 2017.06"
    ) == {
        "business_type": "个人住房贷款",
        "account_count": 2,
        "first_business_issue_month": "2017-06",
    }


def test_colon_month_is_a_field_scoped_typographic_normalization() -> None:
    assert decode_credit_business_overview_text_line("贷记卡 22 2007:01") == {
        "business_type": "贷记卡",
        "account_count": 22,
        "first_business_issue_month": "2007-01",
    }


def test_ambiguous_or_polluted_overview_lines_are_not_guessed() -> None:
    assert decode_credit_business_overview_text_line(
        "个人住房贷款 个人商用房贷款(包括商住两用房) 2 2017.06"
    ) is None
    assert decode_credit_business_overview_text_line("个人住房贷款 S 2 2017.06") is None
    assert decode_credit_business_overview_text_line("个人住房贷款 2 2017.16") is None


def test_failed_category_line_is_retained_as_an_unresolved_witness() -> None:
    rows, unresolved = decode_credit_business_overview_text_lines(
        [
            "业务类型 账户数 首笔业务发放月份",
            "贷款 信用卡 个人住房贷款 2 2017.06",
            "其他类贷款 So 2013.07",
            "逾期(透支)信息汇总",
            "贷记卡 99 2020.01",
        ]
    )

    assert rows == [
        {
            "business_type": "个人住房贷款",
            "account_count": 2,
            "first_business_issue_month": "2017-06",
        }
    ]
    assert unresolved == ["其他类贷款 So 2013.07"]


def test_text_header_and_headerless_table_form_one_credit_overview() -> None:
    table = SimpleNamespace(
        table_id="overview-tail",
        metadata={
            "raw_rows": [
                ["", "其他类贷款", "20", "2013.07"],
                ["", "2 贷记卡 n", "22", "2007:01"],
            ]
        },
        headers=[],
        rows=[],
        bbox=[20, 120, 580, 260],
    )
    context = SimpleNamespace(
        pages=[SimpleNamespace(page_number=3, source_page_number=2, tables=[table])],
        corrected_evidence_pages=lambda: [
            {
                "page": 3,
                "source_page": 2,
                "lines": [
                    {"text": "业务类型 账户数 首笔业务发放月份", "bbox": [20, 20, 580, 40]},
                    {"text": "贷款 信用卡 个人住房贷款 2 2017.06", "bbox": [20, 50, 580, 70]},
                    {"text": "其他类贷款 So 2013.07", "bbox": [20, 80, 580, 100]},
                    {"text": "逾期(透支)信息汇总", "bbox": [20, 280, 580, 300]},
                ],
            }
        ],
        _personal_detail_extraction_issues=[],
    )

    records, cells = _extract_summary_datasets(context)

    assert len(records) == 1
    assert records[0]["title"] == "信用业务概要"
    assert records[0]["source_row_count"] == 3
    first_row = {
        cell["column_label"]: cell["value"]
        for cell in cells
        if cell["row_index"] == 1
    }
    assert first_row == {
        "业务类型": "个人住房贷款",
        "账户数": "2",
        "首笔业务发放月份": "2017-06",
    }
    assert any(
        issue.get("issue_code") == "candidate_b_summary_text_row_unresolved"
        and issue.get("observed_value") == "其他类贷款 So 2013.07"
        for issue in context._personal_detail_extraction_issues
    )


def test_corrected_text_row_does_not_duplicate_same_native_summary_row() -> None:
    table = SimpleNamespace(
        table_id="overview-complete",
        metadata={
            "raw_rows": [
                ["信用业务概要", "", "", ""],
                ["", "业务类型", "账户数", "首笔业务发放月份"],
                ["贷款", "个人住房贷款", "2", "2017.06"],
            ]
        },
        headers=[],
        rows=[],
        bbox=[20, 20, 580, 200],
    )
    context = SimpleNamespace(
        pages=[SimpleNamespace(page_number=3, source_page_number=2, tables=[table])],
        corrected_evidence_pages=lambda: [
            {
                "page": 3,
                "source_page": 2,
                "lines": [
                    {"text": "业务类型 账户数 首笔业务发放月份"},
                    {"text": "贷款 信用卡 个人住房贷款 2 2017.06"},
                    {"text": "逾期(透支)信息汇总"},
                ],
            }
        ],
        _personal_detail_extraction_issues=[],
    )

    records, cells = _extract_summary_datasets(context)

    assert records[0]["source_row_count"] == 1
    assert len({(cell["row_index"], cell["column_label"]) for cell in cells}) == 3


def test_summary_text_anchor_with_zero_usable_rows_is_reported() -> None:
    context = SimpleNamespace(
        pages=[],
        corrected_evidence_pages=lambda: [
            {
                "page": 3,
                "source_page": 2,
                "lines": [
                    {"text": "业务类型 账户数 首笔业务发放月份", "bbox": [20, 20, 580, 40]},
                    {"text": "逾期(透支)信息汇总", "bbox": [20, 80, 580, 100]},
                ],
            }
        ],
        _personal_detail_extraction_issues=[],
    )

    records, cells = _extract_summary_datasets(context)

    assert len(records) == 1
    assert cells == []
    issue = next(
        issue
        for issue in context._personal_detail_extraction_issues
        if issue["issue_code"] == "candidate_b_summary_anchor_without_usable_rows"
    )
    assert issue["target_record_id"] == records[0]["summary_record_id"]


def test_summary_table_anchor_with_zero_usable_rows_is_reported() -> None:
    table = SimpleNamespace(
        table_id="overview-empty",
        metadata={
            "raw_rows": [
                ["信用业务概要", "", "", ""],
                ["", "业务类型", "账户数", "首笔业务发放月份"],
            ]
        },
        headers=[],
        rows=[],
        bbox=[20, 20, 580, 120],
    )
    context = SimpleNamespace(
        pages=[SimpleNamespace(page_number=3, source_page_number=2, tables=[table])],
        corrected_evidence_pages=lambda: [],
        _personal_detail_extraction_issues=[],
    )

    records, cells = _extract_summary_datasets(context)

    assert len(records) == 1
    assert cells == []
    assert any(
        issue["issue_code"] == "candidate_b_summary_anchor_without_usable_rows"
        for issue in context._personal_detail_extraction_issues
    )


def test_summary_count_month_category_collision_withholds_category_without_duplicate() -> None:
    table = SimpleNamespace(
        table_id="overview-conflict",
        metadata={
            "raw_rows": [
                ["信用业务概要", "", "", ""],
                ["", "业务类型", "账户数", "首笔业务发放月份"],
                ["贷款", "个人住房贷款", "2", "2017.06"],
            ]
        },
        headers=[],
        rows=[],
        bbox=[20, 20, 580, 200],
    )
    context = SimpleNamespace(
        pages=[SimpleNamespace(page_number=3, source_page_number=2, tables=[table])],
        corrected_evidence_pages=lambda: [
            {
                "page": 3,
                "source_page": 2,
                "lines": [
                    {"text": "业务类型 账户数 首笔业务发放月份"},
                    {"text": "其他类贷款 2 2017.06"},
                    {"text": "逾期(透支)信息汇总"},
                ],
            }
        ],
        _personal_detail_extraction_issues=[],
    )

    records, cells = _extract_summary_datasets(context)

    assert records[0]["source_row_count"] == 1
    assert [cell["column_label"] for cell in cells] == ["账户数", "首笔业务发放月份"]
    assert [cell["value"] for cell in cells] == ["2", "2017-06"]
    assert any(
        issue["issue_code"] == "candidate_b_summary_category_collision_unresolved"
        and issue["field_name"] == "business_type"
        for issue in context._personal_detail_extraction_issues
    )
