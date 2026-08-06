# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from copy import deepcopy
from pathlib import Path

import pytest

from docmirror.input.entry.factory import PerceiveOptions, perceive_document
from docmirror.input.entry.options import normalize_parse_policy
from docmirror.models.schemas.registry import validate_projection_payload
from docmirror.server.output_builder import build_community_bundle

pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.tier_slow]
_FIXTURE = Path("tests/fixtures-private/credit_report/个人信用报告（本人版）展示样本.pdf")


def test_personal_detail_sample_uses_canonical_typed_datasets() -> None:
    if not _FIXTURE.exists():
        pytest.skip("personal detailed display sample is unavailable")

    sealed = asyncio.run(
        perceive_document(
            _FIXTURE,
            PerceiveOptions(
                policy=normalize_parse_policy(
                    enhance_mode="standard",
                    doc_type_hint="credit_report:force",
                )
            ),
        )
    )
    bundle = build_community_bundle(sealed, file_path=str(_FIXTURE))
    semantic = bundle.semantic_payload()
    payload = bundle.json_payload(semantic)
    datasets = {dataset["name"]: dataset for dataset in payload["datasets"]}

    assert payload["document"]["type"] == "personal_credit_report_detailed"
    assert payload["document"]["domain_schema"]["version"] == "2.0.0"
    assert semantic["domain"]["data_dictionary"]["version"] == "2.0.0"
    assert validate_projection_payload("community", payload).valid
    domain_validation = validate_projection_payload("personal_credit_report_detailed", payload)
    assert domain_validation.valid, domain_validation.errors

    expected_counts = {
        "report_metadata": 1,
        "report_query": 1,
        "subject_profile": 1,
        "subject_identity_documents": 2,
        "subject_mobile_phones": 5,
        "subject_spouse": 1,
        "subject_residences": 5,
        "subject_employment": 5,
        "recovery_account_details": 2,
        "credit_accounts": 15,
        "credit_agreements": 5,
        "repayment_responsibilities": 4,
        "credit_account_monthly_performance": 449,
        "postpaid_accounts": 3,
        "postpaid_monthly_performance": 48,
        "credit_business_overview": 127,
        "tax_arrears_records": 1,
        "civil_judgment_records": 2,
        "enforcement_records": 2,
        "administrative_penalty_records": 1,
        "housing_fund_records": 1,
        "professional_qualification_records": 2,
        "administrative_award_records": 1,
        "inquiries": 12,
        "annotation_statement_groups": 5,
        "annotation_statements": 5,
    }
    assert {name: datasets[name]["row_count"] for name in expected_counts} == expected_counts

    event_count = sum(
        datasets.get(name, {}).get("row_count", 0)
        for name in (
            "credit_account_latest_repayments",
            "credit_account_special_transactions",
            "credit_account_special_events",
            "credit_card_large_installments",
        )
    )
    assert event_count == 8

    accounts = {
        row["normalized"]["account_id"] for row in datasets["credit_accounts"]["rows"]
    }
    monthly = [
        row["normalized"] for row in datasets["credit_account_monthly_performance"]["rows"]
    ]
    assert all(row["account_id"] in accounts for row in monthly)
    assert all(row["performance_month"] for row in monthly)

    observations = [
        row["normalized"] for row in datasets.get("field_observations", {}).get("rows", [])
    ]
    assert all(
        row["observation_status"]
        in {"ocr_corrected", "inferred", "ambiguous", "unreadable", "not_observed"}
        for row in observations
    )
    statuses = [row["normalized"] for row in datasets["dataset_status"]["rows"]]
    assert all(
        row["presence_status"] in {"not_observed", "partial", "extraction_failed", "unknown"}
        for row in statuses
    )

    malformed = deepcopy(payload)
    malformed["datasets"].append(deepcopy(malformed["datasets"][0]))
    assert not validate_projection_payload("personal_credit_report_detailed", malformed).valid
