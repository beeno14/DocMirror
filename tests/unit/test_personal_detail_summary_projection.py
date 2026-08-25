# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Closed-catalog projection tests for scanned detailed-report summaries."""

from __future__ import annotations

import pytest

from docmirror.plugins.credit_report.personal_detail_scanned.schema import (
    project_personal_detail_datasets,
)
from docmirror.plugins.credit_report.personal_detail_scanned.source_projection import (
    _summary_value,
    prepare_personal_detail_source_collections,
)


@pytest.mark.parametrize("raw", ("1,2", "1,,2", "1,23,456", "12,34.5"))
def test_summary_scalar_never_concatenates_malformed_grouping(raw: str) -> None:
    value_type, normalized, status = _summary_value(raw)

    assert (value_type, normalized, status) == ("text", None, "reported")


@pytest.mark.parametrize(
    ("raw", "expected_type", "expected"),
    (
        ("1,234", "integer", "1234"),
        ("12,345.60", "decimal", "12345.60"),
        ("1,234.5%", "percentage", "1234.5"),
    ),
)
def test_summary_scalar_accepts_registered_numeric_presentations(
    raw: str,
    expected_type: str,
    expected: str,
) -> None:
    value_type, normalized, status = _summary_value(raw)

    assert (value_type, normalized, status) == (
        expected_type,
        expected,
        "reported",
    )


def _summary_cell(
    cell_id: str,
    *,
    title: str,
    column_label: str = "账户数",
    value: str = "2",
    row_index: int = 1,
    column_index: int = 1,
) -> dict[str, object]:
    return {
        "record_id": cell_id,
        "summary_cell_id": cell_id,
        "summary_record_id": "summary:1",
        "summary_type": title,
        "title": title,
        "row_index": row_index,
        "column_index": column_index,
        "column_label": column_label,
        "value": value,
    }


@pytest.mark.parametrize(
    ("observed_title", "canonical_type", "canonical_title", "summary_code"),
    (
        (
            "逾期(透支)信息汇总",
            "逾期（透支）",
            "逾期（透支）信息汇总",
            "delinquency_overdraft",
        ),
        (
            "+ 非循环贷账户信息汇总",
            "非循环贷账户",
            "非循环贷账户信息汇总",
            "non_revolving_loan",
        ),
        (
            "今 R 循环贷账户二信息汇总 囍",
            "循环贷账户二",
            "循环贷账户二信息汇总",
            "revolving_loan_account",
        ),
        (
            "No 贷记卡账户信息汇总 券 ?",
            "贷记卡账户",
            "贷记卡账户信息汇总",
            "credit_card",
        ),
    ),
)
def test_exact_catalog_phrase_survives_title_prefix_and_suffix_noise(
    observed_title: str,
    canonical_type: str,
    canonical_title: str,
    summary_code: str,
) -> None:
    content = {
        "facts": {},
        "datasets": {
            "personal_detail_summary_cells": [
                _summary_cell("cell:1", title=observed_title)
            ]
        },
    }

    prepared = prepare_personal_detail_source_collections(content)
    metric = prepared["datasets"]["personal_detail_credit_summary_metrics"][0]

    assert metric["summary_type"] == canonical_type
    assert metric["title"] == canonical_title
    assert metric["summary_code"] == summary_code
    assert metric["metric_code"] == "account_count"
    assert metric["numeric_value"] == "2"
    assert project_personal_detail_datasets(prepared["datasets"])[
        "credit_business_overview"
    ][0]["summary_code"] == summary_code


@pytest.mark.parametrize(
    "observed_title",
    ("未知新型业务概要", "公共事业缴费信息概要", "新型贷记卡账户风险试算"),
)
def test_unknown_summary_title_is_preserved_unmapped_and_reported(observed_title: str) -> None:
    source_cell = _summary_cell("cell:unknown", title=observed_title)
    content = {
        "facts": {},
        "datasets": {"personal_detail_summary_cells": [source_cell]},
    }

    prepared = prepare_personal_detail_source_collections(content)
    projected = project_personal_detail_datasets(prepared["datasets"])

    assert "personal_detail_credit_summary_metrics" not in prepared["datasets"]
    assert "credit_business_overview" not in projected
    assert prepared["datasets"]["personal_detail_summary_cells"] == [source_cell]
    issues = [row.get("normalized", row) for row in projected["extraction_issues"]]
    issue = next(
        row for row in issues if row["issue_code"] == "canonical_summary_cell_unmapped"
    )
    assert issue.get("target_record_id") is None
    assert issue["target_dataset"] == "credit_business_overview"
    assert issue["observed_value"] == "2"


def test_title_recovery_does_not_turn_polluted_scalar_into_business_value() -> None:
    content = {
        "facts": {},
        "datasets": {
            "personal_detail_summary_cells": [
                _summary_cell(
                    "cell:polluted",
                    title="+ 非循环贷账户信息汇总",
                    value="9户",
                )
            ]
        },
    }

    prepared = prepare_personal_detail_source_collections(content)
    metric = prepared["datasets"]["personal_detail_credit_summary_metrics"][0]

    assert metric["summary_code"] == "non_revolving_loan"
    assert metric["reporting_status"] == "unknown"
    assert metric["value_type"] == "unknown"
    assert "numeric_value" not in metric
    assert "text_value" not in metric
    assert any(
        issue["issue_code"] == "candidate_b_summary_scalar_unresolved"
        and issue["target_record_id"] == "cell:polluted"
        for issue in prepared["datasets"]["personal_detail_extraction_issues"]
    )


def test_precise_text_dimension_wins_without_mutating_source_cell_population() -> None:
    source_cells = [
        _summary_cell(
            "cell:type",
            title="信用业务概要",
            column_label="业务类型",
            value="个人住房贷款",
            column_index=2,
        ),
        _summary_cell(
            "cell:count",
            title="信用业务概要",
            column_label="账户数",
            value="3",
            column_index=3,
        ),
    ]
    content = {
        "facts": {},
        "datasets": {"personal_detail_summary_cells": source_cells.copy()},
    }

    prepared = prepare_personal_detail_source_collections(content)
    metrics = prepared["datasets"]["personal_detail_credit_summary_metrics"]
    count_metric = next(row for row in metrics if row["metric_code"] == "account_count")

    assert count_metric["row_dimension_name"] == "业务类型"
    assert count_metric["row_dimension_value"] == "个人住房贷款"
    assert count_metric["business_category"] == "个人住房贷款"
    assert len(prepared["datasets"]["personal_detail_summary_cells"]) == 2
    assert {
        row["summary_cell_id"]
        for row in prepared["datasets"]["personal_detail_summary_cells"]
    } == {"cell:type", "cell:count"}
    assert all(
        row.get("text_value") != "贷款" and row.get("row_dimension_value") != "贷款"
        for row in metrics
    )


@pytest.mark.parametrize(
    ("title", "column_label", "observed_value", "canonical_value"),
    (
        ("信用业务概要", "业务类型", "2 贷记卡 n", "贷记卡"),
        ("逾期（透支）信息汇总", "账户类型", "No 准贷记卡账户 ?", "准贷记卡账户"),
        ("被追偿信息汇总", "业务类型", "资产处置业务", "资产处置业务"),
        ("被追偿信息汇总", "业务类型", "合计", "合计"),
        ("后付费业务欠费信息汇总", "业务类型", "电信业务", "电信业务"),
        ("公共信息汇总", "信息类型", "行政处罚信息", "行政处罚信息"),
    ),
)
def test_text_dimensions_use_context_scoped_finite_catalog(
    title: str,
    column_label: str,
    observed_value: str,
    canonical_value: str,
) -> None:
    content = {
        "facts": {},
        "datasets": {
            "personal_detail_summary_cells": [
                _summary_cell(
                    "cell:dimension",
                    title=title,
                    column_label=column_label,
                    value=observed_value,
                )
            ]
        },
    }

    prepared = prepare_personal_detail_source_collections(content)
    metric = prepared["datasets"]["personal_detail_credit_summary_metrics"][0]

    assert metric["text_value"] == canonical_value
    assert metric["row_dimension_value"] == canonical_value
    if column_label != "信息类型":
        assert metric["business_category"] == canonical_value
    assert prepared["datasets"]["personal_detail_summary_cells"][0]["value"] == observed_value
    assert not any(
        issue["issue_code"] == "candidate_b_summary_text_dimension_unresolved"
        for issue in prepared["datasets"].get("personal_detail_extraction_issues", [])
    )


@pytest.mark.parametrize(
    ("title", "observed_value", "expected_reason"),
    (
        ("信用业务概要", "个人住房贷款贷记卡", "text_value_ambiguous"),
        ("信用业务概要", "新型融资", "text_value_unknown"),
        ("信用业务概要", "错误贷记卡", "text_value_unsafe_noise"),
        ("后付费业务欠费信息汇总", "贷记卡", "text_value_unknown"),
    ),
)
def test_unknown_ambiguous_or_cross_context_text_dimensions_are_withheld(
    title: str,
    observed_value: str,
    expected_reason: str,
) -> None:
    content = {
        "facts": {},
        "datasets": {
            "personal_detail_summary_cells": [
                _summary_cell(
                    "cell:bad-dimension",
                    title=title,
                    column_label="业务类型",
                    value=observed_value,
                ),
                _summary_cell(
                    "cell:count",
                    title=title,
                    column_label="账户数",
                    value="2",
                    column_index=2,
                ),
            ]
        },
    }

    prepared = prepare_personal_detail_source_collections(content)
    metrics = prepared["datasets"]["personal_detail_credit_summary_metrics"]
    dimension_metric = next(row for row in metrics if row["metric_code"] == "business_type")
    count_metric = next(row for row in metrics if row["metric_code"] == "account_count")

    assert dimension_metric["reporting_status"] == "unknown"
    assert dimension_metric["value_type"] == "unknown"
    assert "text_value" not in dimension_metric
    assert "row_dimension_value" not in count_metric
    assert "business_category" not in count_metric
    issue = next(
        issue
        for issue in prepared["datasets"]["personal_detail_extraction_issues"]
        if issue["issue_code"] == "candidate_b_summary_text_dimension_unresolved"
    )
    assert issue["target_record_id"] == "cell:bad-dimension"
    assert issue["observed_value"] == observed_value
    assert expected_reason in issue["reason_codes"]
