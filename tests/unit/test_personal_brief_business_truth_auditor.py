"""The independent regression oracle must reject lost and invented values."""

from __future__ import annotations

import copy

import pytest

from tests._personal_brief_business_truth import audit_business_truth, same_value


def _fixture():
    account = {"account_id": "a1", "institution": "示例银行", "balance": "0", "is_revolving": False}
    payload = {
        "schema": {"version": "4.0.0"}, "document": {"type": "personal_credit_report_brief"},
        "datasets": [{"name": "credit_accounts", "rows": [{"record_id": "r1", "normalized": account}]}],
        "sections": [{"type": "inquiries", "items": [{"key": "lookback_years", "value": 2}]}],
    }
    standard = {
        "case_id": "synthetic", "datasets": {"credit_accounts": [{**account, "account_id": {"$account_ref": 0}}]},
        "sections": {"inquiries": {"lookback_years": 2}},
        "facts": [
            {"id": "balance", "checks": [{"dataset": "credit_accounts", "row": 0, "field": "balance", "expected": "0"}]},
            {"id": "scope", "checks": [{"section": "inquiries", "field": "lookback_years", "expected": 2}]},
        ],
    }
    return standard, payload


def test_source_truth_is_read_only_and_accepts_correct_output():
    standard, payload = _fixture()
    before = copy.deepcopy((standard, payload))
    result = audit_business_truth(standard, payload)
    assert result["passed"] and result["correct"] == result["source_facts"] == 2
    assert (standard, payload) == before


@pytest.mark.parametrize("mutation", ["missing", "wrong", "extra", "boolean", "source_wrapper", "scope", "row", "dataset"])
def test_strict_audit_rejects_regressions(mutation):
    standard, payload = _fixture()
    row = payload["datasets"][0]["rows"][0]
    if mutation == "missing":
        del row["normalized"]["balance"]
    elif mutation == "wrong":
        row["normalized"]["balance"] = "1"
    elif mutation == "extra":
        row["normalized"]["account_currency"] = "CNY"
    elif mutation == "boolean":
        row["normalized"]["is_revolving"] = 0
    elif mutation == "source_wrapper":
        row["source"] = {}
    elif mutation == "scope":
        payload["sections"][0]["items"][0]["value"] = 5
    elif mutation == "row":
        payload["datasets"][0]["rows"].append(copy.deepcopy(row))
    else:
        payload["datasets"].append(copy.deepcopy(payload["datasets"][0]))
    assert not audit_business_truth(standard, payload)["passed"]


def test_exact_enum_boolean_date_and_money_semantics():
    assert same_value("0", 0, "balance")
    assert not same_value("0", False, "balance")
    assert not same_value(False, 0, "is_revolving")
    assert not same_value(2, "2", "lookback_years")
    assert same_value("2025-02-21T00:20:03", "2025-02-21T00:20:03+08:00", "report_time")
    assert not same_value("2025-02-21T00:20:03", "2025-02-21T00:20:03+00:00", "report_time")
    assert not same_value("2025-02", "2025-02-01", "snapshot_date")
