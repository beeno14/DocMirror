# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from docmirror.input.entry.factory import PerceiveOptions, perceive_document
from docmirror.input.entry.options import normalize_parse_policy
from docmirror.models.schemas.registry import validate_projection_payload
from docmirror.plugins.credit_report.personal_detail_scanned.schema import (
    PBOC_DATASET_ORDER,
)
from docmirror.plugins.credit_report.personal_detail_scanned.source_projection import (
    PERSONAL_PROFILE_FIELDS,
)
from docmirror.server.output_builder import build_community_bundle

pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.tier_slow]
_FIXTURE = Path("tests/fixtures-private/credit_report/Scanned Personal Detailed/叶永燕征信.pdf")


def test_ye_yongyan_personal_detail_schema_contract() -> None:
    """Regress the plugin contract without asserting OCR extraction accuracy."""
    if not _FIXTURE.exists():
        pytest.skip("叶永燕 personal detailed report fixture is unavailable")

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
    assert validate_projection_payload("community", payload).valid
    domain_validation = validate_projection_payload("personal_credit_report_detailed", payload)
    status_counts = {
        row["normalized"]["dataset_name"]: row["normalized"]["observed_row_count"]
        for row in datasets["dataset_status"]["rows"]
    }
    assert domain_validation.valid, {
        "errors": domain_validation.errors,
        "credit_agreements": (
            datasets.get("credit_agreements", {}).get("row_count", 0),
            status_counts.get("credit_agreements"),
        ),
        "monthly_performance": (
            datasets.get("credit_account_monthly_performance", {}).get("row_count", 0),
            status_counts.get("credit_account_monthly_performance"),
        ),
    }
    dictionary = semantic["domain"]["data_dictionary"]
    assert dictionary["schema_id"] == "personal_credit_report_detailed"
    assert dictionary["version"] == "2.0.0"

    profile = datasets["subject_profile"]["rows"][0]["normalized"]
    assert profile["subject_name"]
    assert profile["primary_id_number"]
    assert profile["mobile_phone"]

    observations = [row["normalized"] for row in datasets["field_observations"]["rows"]]
    profile_observations = [row for row in observations if row["dataset_name"] == "subject_profile"]
    assert {row["field_name"] for row in profile_observations} <= set(PERSONAL_PROFILE_FIELDS)
    assert all(
        row["observation_status"] in {"ambiguous", "unreadable"}
        for row in observations
        if row["dataset_name"] != "subject_profile"
    )
    assert {row["observation_status"] for row in observations} <= {
        "ocr_corrected",
        "inferred",
        "ambiguous",
        "unreadable",
        "not_observed",
    }
    assert all(
        row.get("reason") == "no_field_observation_emitted"
        for row in observations
        if row["observation_status"] == "not_observed"
    )
    assert all(row["confidence_status"] in {"available", "not_available"} for row in observations)

    summary_metrics = datasets["credit_business_overview"]
    assert summary_metrics["row_count"] > 0
    assert all(row["normalized"]["reporting_status"] in {"reported", "not_reported"} for row in summary_metrics["rows"])

    statuses = {
        row["normalized"]["dataset_name"]: row["normalized"]
        for row in datasets["dataset_status"]["rows"]
    }
    assert set(statuses) <= set(PBOC_DATASET_ORDER) - {
        "field_observations",
        "extraction_issues",
        "pboc_extension_fields",
        "dataset_status",
    }
    assert all(
        row["presence_status"] in {"not_observed", "partial", "extraction_failed", "unknown"}
        for row in statuses.values()
    )
    assert "credit_accounts" not in statuses
    assert "subject_profile" not in statuses
    assert "subject_mobile_phones" not in statuses
    assert "subject_spouse" not in statuses

    # Source-grounded structure: these counts protect the continuation repair
    # and the positional profile-table extraction from schema-only regressions.
    assert datasets["credit_accounts"]["row_count"] == 42
    assert datasets["subject_residences"]["row_count"] == 5
    assert datasets["subject_employment"]["row_count"] == 5
    assert datasets["subject_mobile_phones"]["row_count"] == 1
    assert datasets["subject_spouse"]["row_count"] == 1

    account_ids = {row["normalized"]["account_id"] for row in datasets["credit_accounts"]["rows"]}
    repayment_rows = [
        row["normalized"] for row in datasets["credit_account_monthly_performance"]["rows"]
    ]
    assert repayment_rows
    assert all(row.get("account_id") in account_ids for row in repayment_rows)

    contract = semantic["domain"]["extensions"]["personal_detail_contract"]
    assert contract["absence_requires_explicit_source_evidence"] is True
    assert contract["empty_dataset_means_absent"] is False
    assert set(contract["canonical_public_record_datasets"]) <= set(dictionary["datasets"])
