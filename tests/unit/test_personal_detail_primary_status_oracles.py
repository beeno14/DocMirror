from copy import deepcopy

import pytest

from tests.regression.test_personal_detail_ocr_quality_private import (
    _assert_lin_low_confidence_legal_status_oracle,
)


def _lin_source_status_case():
    return {
        "datasets": [
            {
                "name": "credit_account_monthly_performance",
                "rows": [
                    {
                        "record_id": f"mg_p19_repayment_0:{month}",
                        "normalized": {"status_code": "N", "status_amount": "0"},
                        "review": {"status": "clean"},
                    }
                    for month in ("2019-07", "2021-09")
                ],
            },
            {"name": "extraction_issues", "rows": []},
        ]
    }


def _add_unresolved_status(payload, index=0):
    row = payload["datasets"][0]["rows"][index]
    row["normalized"]["status_code"] = None
    row["review"]["status"] = "requires_review"
    issue = {
        "normalized": {
            "target_dataset": "credit_account_monthly_performance",
            "target_record_id": row["record_id"],
            "field_name": "status_code",
            "issue_code": "candidate_b_monthly_source_ocr_confidence_unresolved",
            "status": "requires_review",
        },
        "source": {"page_range": [19, 19]},
    }
    payload["datasets"][1]["rows"].append(issue)
    return issue


def test_primary_lin_status_oracle_accepts_source_value_or_localized_withholding():
    payload = _lin_source_status_case()
    original = deepcopy(payload)
    _assert_lin_low_confidence_legal_status_oracle(payload)
    assert payload == original
    _add_unresolved_status(payload)
    _assert_lin_low_confidence_legal_status_oracle(payload)


@pytest.mark.parametrize("row_index", (0, 1))
def test_primary_lin_status_oracle_rejects_legal_but_wrong_status_even_with_issue(row_index):
    payload = _lin_source_status_case()
    _add_unresolved_status(payload, row_index)
    payload["datasets"][0]["rows"][row_index]["normalized"]["status_code"] = "M"
    with pytest.raises(AssertionError, match="published 'M'"):
        _assert_lin_low_confidence_legal_status_oracle(payload)


@pytest.mark.parametrize("damage", ("wrong_page", "wrong_field", "resolved", "generic_issue", "missing_row", "lost_amount"))
def test_primary_lin_status_oracle_rejects_unlocalized_or_destructive_withholding(damage):
    payload = _lin_source_status_case()
    issue = _add_unresolved_status(payload)
    if damage == "wrong_page":
        issue["source"]["page_range"] = [20, 20]
    elif damage == "wrong_field":
        issue["normalized"]["field_name"] = "status_amount"
    elif damage == "resolved":
        issue["normalized"]["status"] = "resolved"
    elif damage == "generic_issue":
        issue["normalized"]["issue_code"] = "some_other_uncertainty"
    elif damage == "missing_row":
        payload["datasets"][0]["rows"].pop(0)
    else:
        payload["datasets"][0]["rows"][0]["normalized"]["status_amount"] = None
    with pytest.raises(AssertionError):
        _assert_lin_low_confidence_legal_status_oracle(payload)
