# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CQF canonical row audit tests."""

from __future__ import annotations

from types import SimpleNamespace

from docmirror.plugins.bank_statement.canonical_quality import (
    audit_amount_consistency,
    audit_cqf,
    is_canonical_row,
    physical_transaction_row_estimate,
    resolve_extract_status,
)


def _physical_table(headers: list[str], rows: list[list[str]]) -> SimpleNamespace:
    return SimpleNamespace(
        headers=headers,
        rows=[SimpleNamespace(cells=[SimpleNamespace(text=value) for value in row]) for row in rows],
    )


def test_physical_estimate_counts_transaction_promoted_into_continuation_header() -> None:
    schema = ["序号", "交易日期", "交易金额", "余额"]
    continuation_header = ["2", "2024-01-02", "-2.00", "7.00"]
    parse_result = SimpleNamespace(
        pages=[
            SimpleNamespace(tables=[_physical_table(schema, [["1", "2024-01-01", "-1.00", "9.00"]])]),
            SimpleNamespace(tables=[_physical_table(continuation_header, [["3", "2024-01-03", "+3.00", "10.00"]])]),
        ]
    )

    assert physical_transaction_row_estimate(parse_result) == 3


def test_physical_estimate_never_counts_schema_header_as_transaction() -> None:
    schema = ["序号", "交易日期", "交易金额", "余额"]
    parse_result = SimpleNamespace(
        pages=[
            SimpleNamespace(
                tables=[
                    _physical_table(schema, [["1", "2024-01-01", "-1.00", "9.00"]]),
                    _physical_table(schema, [["1", "2024-01-01", "-1.00", "9.00"]]),
                ]
            )
        ]
    )

    # Overlapping physical candidates are still max-per-page, not summed.
    assert physical_transaction_row_estimate(parse_result) == 1


def test_physical_estimate_does_not_trust_isolated_transaction_shaped_header() -> None:
    transaction_header = ["2", "2024-01-02", "-2.00", "7.00"]
    parse_result = SimpleNamespace(
        pages=[SimpleNamespace(tables=[_physical_table(transaction_header, [["打印时间", "", "", ""]])])]
    )

    assert physical_transaction_row_estimate(parse_result) == 0


def test_physical_estimate_requires_matching_source_schema_width() -> None:
    schema = ["序号", "交易日期", "交易金额", "余额"]
    unrelated_transaction_header = ["2", "2024-01-02", "备注", "-2.00", "7.00"]
    parse_result = SimpleNamespace(
        pages=[
            SimpleNamespace(tables=[_physical_table(schema, [["1", "2024-01-01", "-1.00", "9.00"]])]),
            SimpleNamespace(
                tables=[_physical_table(unrelated_transaction_header, [["", "2024-01-03", "", "+3.00", "10.00"]])]
            ),
        ]
    )

    assert physical_transaction_row_estimate(parse_result) == 2


def test_physical_estimate_requires_matching_source_role_positions() -> None:
    schema = ["序号", "交易日期", "交易金额", "余额"]
    same_width_wrong_roles = ["2", "-2.00", "2024-01-02", "7.00"]
    parse_result = SimpleNamespace(
        pages=[
            SimpleNamespace(tables=[_physical_table(schema, [["1", "2024-01-01", "-1.00", "9.00"]])]),
            SimpleNamespace(
                tables=[_physical_table(same_width_wrong_roles, [["3", "+3.00", "2024-01-03", "10.00"]])]
            ),
        ]
    )

    assert physical_transaction_row_estimate(parse_result) == 2


def test_is_canonical_row_requires_directional_amount():
    assert is_canonical_row({"date": "2024-01-01", "direction": "income", "amount": 10.0})
    assert is_canonical_row({"date": "2024-01-01", "direction": "income", "amount": 0.0})
    assert is_canonical_row({"date": "2024-01-01", "direction": "", "amount": 0.0, "summary": "利息税"})
    assert not is_canonical_row({"date": "2024-01-01", "direction": "", "amount": 0.0})
    assert not is_canonical_row({"date": "2024-01-01", "direction": "other", "amount": 10.0})
    assert not is_canonical_row({"direction": "income", "amount": 10.0})
    assert not is_canonical_row({"date": "2024-01-01", "direction": "income", "amount": None})
    assert not is_canonical_row({"date": "2024-01-01", "direction": "income", "amount": ""})


def test_is_canonical_row_requires_real_calendar_date():
    assert is_canonical_row({"date": "20240229", "direction": "income", "amount": 1.0})
    assert is_canonical_row({"date": "2024/02/29", "direction": "expense", "amount": 0.0})
    assert not is_canonical_row({"date": "2023-06-00", "direction": "income", "amount": 1.0})
    assert not is_canonical_row({"date": "2023-02-29", "direction": "income", "amount": 1.0})
    assert not is_canonical_row({"date": "2024-13-01", "direction": "income", "amount": 1.0})
    assert not is_canonical_row({"date": "2024-01-01garbage", "direction": "income", "amount": 1.0})
    assert not is_canonical_row({"date": "2024010120240102", "direction": "income", "amount": 1.0})


def test_amount_consistency_detects_nonzero_source_amount_lost_to_zero() -> None:
    warnings = audit_amount_consistency(
        [
            {
                "record_id": "records:r000020",
                "raw": {"收入金额": "0", "支出金额": "2.25"},
                "normalized": {"amount": 0.0, "direction": "income"},
            }
        ]
    )

    assert any(warning.startswith("BANK_NONZERO_AMOUNT_LOST:") for warning in warnings)


def test_amount_consistency_preserves_explicit_zero_and_flags_unknown_direction() -> None:
    warnings = audit_amount_consistency(
        [
            {
                "record_id": "records:r000001",
                "raw": {"收入金额": "0", "支出金额": "0.00"},
                "normalized": {"amount": 0.0, "direction": ""},
            }
        ]
    )

    assert not any(warning.startswith("BANK_NONZERO_AMOUNT_LOST:") for warning in warnings)
    assert any(warning.startswith("BANK_ZERO_AMOUNT_DIRECTION_UNKNOWN:") for warning in warnings)


def test_amount_consistency_detects_missing_source_amount_defaulted_to_zero() -> None:
    warnings = audit_amount_consistency(
        [
            {
                "record_id": "records:r000002",
                "raw": {"收入金额": "", "支出金额": ""},
                "normalized": {"amount": 0.0, "direction": "income"},
            }
        ]
    )

    assert any(warning.startswith("BANK_AMOUNT_DEFAULTED_TO_ZERO:") for warning in warnings)


def test_audit_cqf_degraded_when_canonical_low():
    records = [
        {"normalized": {"date": "2024-01-01", "direction": "other", "amount": 1.0}},
        {"normalized": {"date": "2024-01-02", "direction": "other", "amount": 2.0}},
    ]
    result = audit_cqf(records, canonical_expected=100)
    assert result.extract_status == "degraded"
    assert result.canonical_extracted == 0


def test_audit_cqf_success_requires_full_coverage():
    records = [{"normalized": {"date": "2024-01-01", "direction": "income", "amount": 1.0}} for _ in range(100)]
    result = audit_cqf(records, canonical_expected=100)
    assert result.extract_status == "success"
    assert result.canonical_ratio == 1.0


def test_audit_cqf_does_not_mark_over_extraction_as_success():
    records = [{"normalized": {"date": "2024-01-01", "direction": "income", "amount": 1.0}} for _ in range(2)]

    result = audit_cqf(records, canonical_expected=1)

    assert result.extract_status == "low_coverage"
    assert result.coverage_ratio == 1.0
    assert result.canonical_ratio == 1.0


def test_resolve_extract_status_thresholds():
    assert resolve_extract_status(coverage_ratio=1.0, canonical_ratio=1.0) == "success"
    assert resolve_extract_status(coverage_ratio=0.9, canonical_ratio=0.9) == "low_coverage"
    assert resolve_extract_status(coverage_ratio=0.6, canonical_ratio=0.6) == "low_coverage"
    assert resolve_extract_status(coverage_ratio=0.3, canonical_ratio=0.3) == "degraded"
