# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-variant credit-report schema integrity tests."""

from docmirror.models.schemas.registry import _personal_detail_invariant_errors
from docmirror.plugins.credit_report.contracts import CONTENT_MODE_SCANNED
from docmirror.plugins.credit_report.personal_detail_scanned.contract_projection import (
    _summary_value,
    apply_personal_detail_contract,
)
from docmirror.plugins.credit_report.personal_detail_scanned.schema import (
    PERSONAL_DETAIL_DATASET_ORDER,
    personal_detail_semantic_extensions,
)
from docmirror.plugins.credit_report.personal_detail_scanned.variant import (
    PersonalDetailScannedVariant,
)
from docmirror.plugins.credit_report.semantic_enrichment import credit_report_data_dictionary
from docmirror.plugins.credit_report.variant_router import registered_credit_report_variants


def test_every_variant_dataset_has_a_dictionary_definition() -> None:
    for variant in registered_credit_report_variants():
        datasets = variant.data_dictionary()["datasets"]
        assert set(variant.dataset_names()) <= set(datasets), variant.variant_id


def test_scanned_content_mode_uses_the_runtime_vocabulary() -> None:
    enums = credit_report_data_dictionary()["enums"]["content_mode"]
    assert CONTENT_MODE_SCANNED in enums
    assert "scanned" not in enums


def test_nonempty_personal_detail_dataset_may_be_explicitly_partial() -> None:
    payload = {
        "document": {"domain_schema": {"id": "personal_credit_report_detailed"}},
        "datasets": [
            {"name": "personal_profile", "row_count": 0, "rows": []},
            {"name": "personal_detail_field_observations", "row_count": 0, "rows": []},
            {
                "name": "credit_lines",
                "row_count": 1,
                "rows": [{"record_id": "credit_line:1", "normalized": {"credit_line_id": "credit_line:1"}}],
            },
            {
                "name": "personal_detail_dataset_status",
                "row_count": 1,
                "rows": [
                    {
                        "record_id": "dataset_status:credit_lines",
                        "normalized": {
                            "dataset_name": "credit_lines",
                            "observed_row_count": 1,
                            "presence_status": "partial",
                        },
                    }
                ],
            },
        ],
    }

    assert _personal_detail_invariant_errors(payload) == ()


def test_summary_numeric_values_require_complete_decimal_lexemes() -> None:
    assert _summary_value("2.0") == ("decimal", "2.0", "reported")
    assert _summary_value("12.5%") == ("percentage", "12.5", "reported")
    assert _summary_value("2.") == ("text", None, "reported")
    assert _summary_value(".0") == ("text", None, "reported")


def test_scanned_variant_forwards_precomputed_auxiliary_records() -> None:
    variant = PersonalDetailScannedVariant()
    content = variant.build_section_content(
        object(),
        "",
        auxiliary_business={
            "subject_profile": {"subject_name": {"value": "示例"}},
            "residence_records": [{"sequence": 1, "values": {"居住地址": "示例地址"}}],
            "employment_records": [{"sequence": 1, "values": {"工作单位": "示例单位"}}],
            "statements": [{"id": "statement:1", "text": "本人声明"}],
            "annotations": [{"id": "annotation:1", "text": "异议标注"}],
        },
    )

    assert content["facts"]["subject_profile"]["subject_name"]["value"] == "示例"
    assert {
        "residence_records",
        "employment_records",
        "statements",
        "annotations",
    } <= set(content["datasets"])
    assert "personal_profile" in content["datasets"]
    assert "personal_detail_field_observations" in content["datasets"]
    assert "personal_detail_dataset_status" in content["datasets"]


def test_personal_detail_dictionary_covers_profile_summary_public_absence_and_uncertainty() -> None:
    variant = PersonalDetailScannedVariant()
    dictionary = variant.data_dictionary()
    datasets = dictionary["datasets"]
    semantic = personal_detail_semantic_extensions()

    assert dictionary["version"] == "1.2.0"
    assert {
        "gender",
        "birth_date",
        "employment_status",
        "education_level",
        "degree",
        "nationality",
        "mobile_phone",
        "work_phone",
        "residence_phone",
        "email",
        "mailing_address",
        "household_address",
    } <= set(dictionary["fields"])
    assert {
        "personal_profile",
        "personal_detail_credit_summary_metrics",
        "tax_arrears_records",
        "civil_judgment_records",
        "enforcement_records",
        "administrative_penalty_records",
        "personal_housing_fund_records",
        "professional_qualification_records",
        "award_records",
        "personal_detail_dataset_status",
        "personal_detail_field_observations",
    } <= set(datasets)
    assert set(PERSONAL_DETAIL_DATASET_ORDER) <= set(datasets)
    for dataset_name, reading_columns in semantic["dataset_reading_columns"].items():
        assert set(reading_columns) <= set(datasets[dataset_name]["columns"]), dataset_name
    contract = semantic["personal_detail_contract"]
    assert contract["canonical_profile_dataset"] == "personal_profile"
    assert contract["canonical_credit_summary_dataset"] == "personal_detail_credit_summary_metrics"
    assert contract["absence_dataset"] == "personal_detail_dataset_status"
    assert contract["uncertainty_dataset"] == "personal_detail_field_observations"
    assert contract["absence_requires_explicit_source_evidence"] is True
    assert contract["empty_dataset_means_absent"] is False
    assert semantic["domain_schema"]["version"] == "1.2.0"
    assert semantic["presentation_policy"]["enhanced_markdown_display"] == "full"
    assert semantic["personal_detail_contract"]["uncertainty_coverage"]["mode"] == (
        "potentially_flawed_only"
    )
    assert dictionary["datasets"]["repayment_records"]["columns"]["status"]["enum_ref"] == ("repayment_status_code")


def test_personal_detail_contract_projects_values_without_inventing_absence() -> None:
    content = {
        "facts": {
            "subject_name": "示例用户",
            "subject_profile": {
                "gender": {"value": "女", "raw": "女"},
                "birth_date": {"value": "1980-11-04", "raw": "1980.11.04"},
            },
            "personal_detail_dataset_states": {
                "spouse_records": {
                    "presence_status": "explicitly_empty",
                    "source_statement": "配偶信息：无",
                }
            },
        },
        "datasets": {
            "personal_report_metadata": [
                {
                    "personal_report_metadata_id": "metadata:1",
                    "subject_name": "示例用户",
                    "primary_id_type": "身份证",
                    "primary_id_number": "110101198011040000",
                }
            ],
            "personal_detail_summary_records": [
                {
                    "summary_record_id": "summary:1",
                    "summary_type": "account_count",
                    "title": "信贷交易信息提示",
                    "source_table_id": "table:1",
                }
            ],
            "personal_detail_summary_cells": [
                {
                    "summary_cell_id": "cell:1",
                    "summary_record_id": "summary:1",
                    "summary_type": "account_count",
                    "title": "信贷交易信息提示",
                    "row_index": 1,
                    "column_index": 1,
                    "column_label": "账户类型",
                    "value": "贷款",
                },
                {
                    "summary_cell_id": "cell:2",
                    "summary_record_id": "summary:1",
                    "summary_type": "account_count",
                    "title": "信贷交易信息提示",
                    "row_index": 1,
                    "column_index": 2,
                    "column_label": "账户数",
                    "value": "23,505",
                },
            ],
        },
    }
    auxiliary = {
        "public_records": [
            {
                "public_record_id": "public:1",
                "sequence": 1,
                "record_type": "housing_fund",
                "authority": "示例单位",
                "content": {
                    "employer": "示例单位",
                    "monthly_contribution": 3000,
                    "payment_status": "缴交",
                },
            }
        ]
    }

    projected = apply_personal_detail_contract(
        content,
        auxiliary,
        final_dataset_counts={"credit_accounts": 15, "inquiry_records": 12},
    )
    datasets = projected["datasets"]
    profile = datasets["personal_profile"][0]
    assert profile["subject_name"] == "示例用户"
    assert profile["birth_date"] == "1980-11-04"

    observations = {row["field_name"]: row for row in datasets["personal_detail_field_observations"]}
    assert "gender" not in observations
    assert "birth_date" not in observations
    assert observations["work_phone"]["observation_status"] == "not_observed"
    assert observations["work_phone"]["confidence_status"] == "not_available"

    metrics = datasets["personal_detail_credit_summary_metrics"]
    account_count = next(row for row in metrics if row["metric_name"] == "账户数")
    assert account_count["row_dimension_name"] == "账户类型"
    assert account_count["row_dimension_value"] == "贷款"
    assert account_count["business_category"] == "贷款"
    assert account_count["numeric_value"] == "23505"
    assert account_count["reporting_status"] == "reported"
    assert account_count["metric_code"] == "account_count"

    housing_fund = datasets["personal_housing_fund_records"][0]
    assert housing_fund["public_record_id"] == "public:1"
    assert housing_fund["monthly_contribution"] == 3000

    statuses = {row["dataset_name"]: row for row in datasets["personal_detail_dataset_status"]}
    assert "public_records" not in statuses
    assert "credit_accounts" not in statuses
    assert "inquiry_records" not in statuses
    assert statuses["mobile_phone_records"]["presence_status"] == "not_observed"
    assert statuses["mobile_phone_records"]["reason"] == "no_explicit_absence_evidence"
    assert "spouse_records" not in statuses
