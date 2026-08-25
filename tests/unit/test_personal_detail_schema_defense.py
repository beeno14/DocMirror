from __future__ import annotations

import json
from typing import Any

import pytest

from docmirror.plugins.credit_report.personal_detail_scanned.schema import (
    project_personal_detail_datasets,
)

_POSTPAID_STATUS_CODES = ("*", "N", "0", "1", "2", "3", "4", "5", "6", "C", "G", "#")
_LEGACY_POSTPAID_STATUS_CODES = ("A", "B", "D", "M", "Z", "7", "/")
_POSTPAID_BUSINESS_TYPES = (
    "固定电话",
    "固定电话后付费",
    "移动电话",
    "移动电话后付费",
    "有线电视",
    "有线电视后付费",
    "电信业务",
    "水费",
    "电费",
    "燃气费",
    "水电气等公用事业",
)


def _values(record: dict[str, Any]) -> dict[str, Any]:
    normalized = record.get("normalized")
    return normalized if isinstance(normalized, dict) else record


def _field_ref(field_name: str, marker: str) -> dict[str, Any]:
    return {
        "field_name": field_name,
        "source": "schema_defense_test",
        "marker": marker,
    }


def _postpaid_account(
    *,
    business_type: Any = "移动电话",
    payment_status: Any = "正常",
) -> dict[str, Any]:
    return {
        "record_id": "postpaid:account:1",
        "normalized": {
            "postpaid_record_id": "postpaid:account:1",
            "institution": "某通信公司",
            "business_type": business_type,
            "payment_status": payment_status,
            "billing_month": "2024-01",
        },
        "canonical_raw": {
            "business_type": business_type,
            "payment_status": payment_status,
        },
        "source_refs_by_field": {
            "business_type": [_field_ref("business_type", "business-own")],
            "payment_status": [_field_ref("payment_status", "payment-own")],
        },
        "source_refs": [_field_ref("institution", "foreign")],
    }


def _postpaid_month(status: Any) -> dict[str, Any]:
    return {
        "record_id": "postpaid:month:1",
        "normalized": {
            "postpaid_payment_history_id": "postpaid:month:1",
            "postpaid_record_id": "postpaid:account:1",
            "year": 2024,
            "month": 1,
            "status": status,
        },
        "canonical_raw": {"status": status},
        "source_refs_by_field": {
            "status_code": [_field_ref("status_code", "status-own")],
        },
        "source_refs": [_field_ref("business_type", "foreign")],
    }


def _project_postpaid(*, status: Any = "N", account: dict[str, Any] | None = None):
    return project_personal_detail_datasets(
        {
            "postpaid_records": [account or _postpaid_account()],
            "postpaid_payment_history": [_postpaid_month(status)],
        }
    )


@pytest.mark.parametrize("status", _POSTPAID_STATUS_CODES)
def test_postpaid_monthly_gate_accepts_only_native_status_codes(status: str) -> None:
    projected = _project_postpaid(status=status)

    month = _values(projected["postpaid_monthly_performance"][0])
    assert month["status_code"] == status
    assert not any(
        _values(issue).get("issue_code") == "canonical_postpaid_monthly_status_invalid"
        for issue in projected.get("extraction_issues", [])
    )


@pytest.mark.parametrize("status", _LEGACY_POSTPAID_STATUS_CODES)
def test_postpaid_monthly_gate_withholds_legacy_status_with_one_local_issue(
    status: str,
) -> None:
    projected = _project_postpaid(status=status)

    month_record = projected["postpaid_monthly_performance"][0]
    month = _values(month_record)
    assert month.get("status_code") is None
    assert month.get("status_code") != "unknown"
    assert month_record["canonical_raw"]["status_code"] == status
    issues = [
        _values(issue)
        for issue in projected["extraction_issues"]
        if _values(issue).get("issue_code") == "canonical_postpaid_monthly_status_invalid"
    ]
    assert len(issues) == 1
    assert issues[0]["target_dataset"] == "postpaid_monthly_performance"
    assert issues[0]["target_record_id"] == "postpaid:month:1"
    assert issues[0]["field_name"] == "status_code"
    assert issues[0]["observed_value"] == status
    assert issues[0]["source_refs"] == [_field_ref("status_code", "status-own")]


@pytest.mark.parametrize("status", [None, "", "--"])
def test_postpaid_monthly_gate_keeps_null_or_source_absence_null(status: Any) -> None:
    projected = _project_postpaid(status=status)

    month = _values(projected["postpaid_monthly_performance"][0])
    assert month.get("status_code") is None
    assert not any(
        _values(issue).get("issue_code") == "canonical_postpaid_monthly_status_invalid"
        for issue in projected.get("extraction_issues", [])
    )


def test_postpaid_monthly_conflicting_aliases_are_withheld_once_with_both_raw_values() -> None:
    month = _postpaid_month("A")
    month["normalized"]["status_code"] = "N"
    month["canonical_raw"]["status_code"] = "N"
    month["source_refs_by_field"]["status"] = [
        _field_ref("status", "source-status-own")
    ]

    projected = project_personal_detail_datasets(
        {
            "postpaid_records": [_postpaid_account()],
            "postpaid_payment_history": [month],
        }
    )

    month_record = projected["postpaid_monthly_performance"][0]
    assert _values(month_record).get("status_code") is None
    assert month_record["canonical_raw"]["status"] == "A"
    assert month_record["canonical_raw"]["status_code"] == "N"
    issue_records = [
        issue
        for issue in projected["extraction_issues"]
        if _values(issue).get("issue_code")
        == "canonical_postpaid_monthly_status_invalid"
    ]
    assert len(issue_records) == 1
    issue = _values(issue_records[0])
    assert issue["target_dataset"] == "postpaid_monthly_performance"
    assert issue["target_record_id"] == "postpaid:month:1"
    assert issue["field_name"] == "status_code"
    assert issue_records[0]["source_refs"] == [
        _field_ref("status", "source-status-own"),
        _field_ref("status_code", "status-own"),
    ]
    observed_evidence = {
        _values(evidence)["evidence_path"]: _values(evidence)["string_value"]
        for evidence in projected["extraction_issue_evidence"]
        if _values(evidence).get("extraction_issue_id")
        == issue["extraction_issue_id"]
        and _values(evidence).get("evidence_kind") == "observed"
    }
    assert observed_evidence == {"status": "A", "status_code": "N"}
@pytest.mark.parametrize("business_type", _POSTPAID_BUSINESS_TYPES)
def test_postpaid_account_gate_accepts_native_business_types(business_type: str) -> None:
    projected = _project_postpaid(account=_postpaid_account(business_type=business_type))

    account = _values(projected["postpaid_accounts"][0])
    assert account["business_type"] == business_type
    assert not any(
        _values(issue).get("issue_code") == "canonical_postpaid_business_type_invalid"
        for issue in projected.get("extraction_issues", [])
    )


@pytest.mark.parametrize("payment_status", ["正常", "欠费"])
def test_postpaid_account_gate_accepts_native_payment_status(payment_status: str) -> None:
    projected = _project_postpaid(account=_postpaid_account(payment_status=payment_status))

    account = _values(projected["postpaid_accounts"][0])
    assert account["payment_status"] == payment_status
    assert not any(
        _values(issue).get("issue_code") == "canonical_postpaid_payment_status_invalid"
        for issue in projected.get("extraction_issues", [])
    )


def test_postpaid_account_gate_withholds_each_invalid_enum_before_parent_use() -> None:
    projected = _project_postpaid(
        account=_postpaid_account(
            business_type="业务类型移动电话",
            payment_status="正常欠费",
        )
    )

    account_record = projected["postpaid_accounts"][0]
    account = _values(account_record)
    assert account.get("business_type") is None
    assert account.get("payment_status") is None
    assert account_record["canonical_raw"]["business_type"] == "业务类型移动电话"
    assert account_record["canonical_raw"]["payment_status"] == "正常欠费"
    issues = {
        _values(issue)["field_name"]: _values(issue)
        for issue in projected["extraction_issues"]
        if _values(issue).get("issue_code")
        in {
            "canonical_postpaid_business_type_invalid",
            "canonical_postpaid_payment_status_invalid",
        }
    }
    assert set(issues) == {"business_type", "payment_status"}
    assert issues["business_type"]["source_refs"] == [
        _field_ref("business_type", "business-own")
    ]
    assert issues["payment_status"]["source_refs"] == [
        _field_ref("payment_status", "payment-own")
    ]
    month = _values(projected["postpaid_monthly_performance"][0])
    assert month.get("business_type") is None


_PUBLIC_MONEY_FIELDS = (
    ("tax_arrears", "tax_arrears_records", "arrears_amount"),
    ("civil_judgment", "civil_judgment_records", "claim_amount"),
    ("enforcement", "enforcement_records", "executed_amount"),
    ("enforcement", "enforcement_records", "requested_amount"),
    ("administrative_penalty", "administrative_penalty_records", "penalty_amount"),
    ("housing_fund", "housing_fund_records", "monthly_contribution"),
)


def _public_record(
    *,
    record_type: str,
    field_name: str,
    parsed_value: Any,
    raw_value: Any,
) -> dict[str, Any]:
    return {
        "record_id": f"public:{record_type}:1",
        "normalized": {
            "public_record_id": f"public:{record_type}:1",
            "record_type": record_type,
            "sequence": 1,
            "content": json.dumps({field_name: parsed_value}, ensure_ascii=False),
        },
        "canonical_raw": {field_name: raw_value},
        "source_refs_by_field": {
            field_name: [_field_ref(field_name, "money-own")],
        },
        "source_refs": [_field_ref("foreign_money_field", "foreign")],
    }


@pytest.mark.parametrize("raw_value", ["1 2", "1, 234"])
@pytest.mark.parametrize("record_type,target_dataset,field_name", _PUBLIC_MONEY_FIELDS)
def test_public_content_money_requires_complete_field_local_raw_decimal(
    record_type: str,
    target_dataset: str,
    field_name: str,
    raw_value: str,
) -> None:
    projected = project_personal_detail_datasets(
        {
            "public_records": [
                _public_record(
                    record_type=record_type,
                    field_name=field_name,
                    parsed_value=12,
                    raw_value=raw_value,
                )
            ]
        }
    )

    public_record = projected[target_dataset][0]
    assert _values(public_record).get(field_name) is None
    assert public_record["canonical_raw"][field_name] == raw_value
    issues = [
        _values(issue)
        for issue in projected["extraction_issues"]
        if _values(issue).get("issue_code") == "canonical_public_money_raw_invalid"
    ]
    assert len(issues) == 1
    assert issues[0]["target_dataset"] == target_dataset
    assert issues[0]["target_record_id"] == f"public:{record_type}:1"
    assert issues[0]["field_name"] == field_name
    assert issues[0]["observed_value"] == raw_value
    assert issues[0]["source_refs"] == [_field_ref(field_name, "money-own")]


def test_public_content_money_requires_semantic_equality_with_valid_raw() -> None:
    projected = project_personal_detail_datasets(
        {
            "public_records": [
                _public_record(
                    record_type="tax_arrears",
                    field_name="arrears_amount",
                    parsed_value="1235.00",
                    raw_value="1,234",
                )
            ]
        }
    )

    record = projected["tax_arrears_records"][0]
    assert _values(record).get("arrears_amount") is None
    issue = next(
        _values(issue)
        for issue in projected["extraction_issues"]
        if _values(issue).get("issue_code") == "canonical_public_money_raw_mismatch"
    )
    assert issue["observed_value"] == "1,234"
    assert issue["candidate_value"] == "1235.00"
    assert issue["source_refs"] == [_field_ref("arrears_amount", "money-own")]


def test_public_content_money_accepts_grouped_raw_with_equal_decimal_value() -> None:
    projected = project_personal_detail_datasets(
        {
            "public_records": [
                _public_record(
                    record_type="tax_arrears",
                    field_name="arrears_amount",
                    parsed_value="1234.00",
                    raw_value="1,234",
                )
            ]
        }
    )

    assert _values(projected["tax_arrears_records"][0])["arrears_amount"] == "1234.00"
    assert not any(
        _values(issue).get("issue_code", "").startswith("canonical_public_money_raw_")
        for issue in projected.get("extraction_issues", [])
    )
