# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from docmirror.input.entry.factory import PerceiveOptions, perceive_document
from docmirror.input.entry.options import normalize_parse_policy
from docmirror.models.schemas.registry import validate_projection_payload
from docmirror.plugins.credit_report.personal_detail_scanned.contract_projection import (
    PERSONAL_DETAIL_BUSINESS_DATASETS,
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
        for row in datasets["personal_detail_dataset_status"]["rows"]
    }
    assert domain_validation.valid, {
        "errors": domain_validation.errors,
        "credit_lines": (datasets.get("credit_lines", {}).get("row_count", 0), status_counts["credit_lines"]),
        "overdue_records": (
            datasets.get("overdue_records", {}).get("row_count", 0),
            status_counts["overdue_records"],
        ),
    }
    dictionary = semantic["domain"]["data_dictionary"]
    assert dictionary["schema_id"] == "personal_credit_report_detailed"
    assert dictionary["version"] == "1.2.0"

    profile = datasets["personal_profile"]["rows"][0]["normalized"]
    assert profile["subject_name"]
    assert profile["primary_id_number"]
    assert profile["mobile_phone"]

    observations = [
        row["normalized"] for row in datasets["personal_detail_field_observations"]["rows"]
    ]
    assert {row["field_name"] for row in observations} == set(PERSONAL_PROFILE_FIELDS)
    assert {row["observation_status"] for row in observations} <= {
        "observed",
        "normalized",
        "ocr_corrected",
        "inferred",
        "ambiguous",
        "unreadable",
        "not_observed",
        "explicitly_absent",
        "not_applicable",
    }
    assert all(
        row.get("reason") == "no_field_observation_emitted"
        for row in observations
        if row["observation_status"] == "not_observed"
    )
    assert all(row["confidence_status"] in {"available", "not_available"} for row in observations)

    summary_metrics = datasets["personal_detail_credit_summary_metrics"]
    assert summary_metrics["row_count"] == datasets["personal_detail_summary_cells"]["row_count"]
    assert all(
        row["normalized"]["reporting_status"] in {"reported", "not_reported"}
        for row in summary_metrics["rows"]
    )

    statuses = {
        row["normalized"]["dataset_name"]: row["normalized"]
        for row in datasets["personal_detail_dataset_status"]["rows"]
    }
    assert set(statuses) == set(PERSONAL_DETAIL_BUSINESS_DATASETS)
    assert statuses["credit_accounts"]["presence_status"] == "observed_nonempty"
    assert statuses["personal_profile"]["presence_status"] == "observed_nonempty"
    assert statuses["mobile_phone_records"]["presence_status"] == "observed_nonempty"
    assert statuses["spouse_records"]["presence_status"] == "observed_nonempty"
    assert not any(row["presence_status"] == "explicitly_empty" for row in statuses.values())

    # Source-grounded structure: these counts protect the continuation repair
    # and the positional profile-table extraction from schema-only regressions.
    assert datasets["credit_accounts"]["row_count"] == 42
    assert datasets["residence_records"]["row_count"] == 5
    assert datasets["employment_records"]["row_count"] == 5
    assert datasets["mobile_phone_records"]["row_count"] == 1
    assert datasets["spouse_records"]["row_count"] == 1

    account_ids = {
        row["normalized"]["account_id"] for row in datasets["credit_accounts"]["rows"]
    }
    repayment_rows = [
        row["normalized"] for row in datasets["repayment_records"]["rows"]
    ]
    assert repayment_rows
    assert all(row.get("account_id") in account_ids for row in repayment_rows)

    contract = semantic["domain"]["extensions"]["personal_detail_contract"]
    assert contract["absence_requires_explicit_source_evidence"] is True
    assert contract["empty_dataset_means_absent"] is False
    assert set(contract["canonical_public_record_datasets"]) <= set(dictionary["datasets"])
