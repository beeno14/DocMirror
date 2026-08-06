# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-variant credit-report schema integrity tests."""

from docmirror.models.schemas.registry import _personal_detail_invariant_errors
from docmirror.plugins.credit_report.contracts import CONTENT_MODE_SCANNED
from docmirror.plugins.credit_report.personal_detail_scanned.schema import (
    PBOC_DATASET_ORDER,
    personal_detail_semantic_extensions,
)
from docmirror.plugins.credit_report.personal_detail_scanned.source_projection import (
    _summary_value,
    prepare_personal_detail_source_collections,
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
            {"name": "report_metadata", "row_count": 0, "rows": []},
            {"name": "report_query", "row_count": 0, "rows": []},
            {"name": "subject_profile", "row_count": 0, "rows": []},
            {
                "name": "credit_agreements",
                "row_count": 1,
                "rows": [
                    {
                        "record_id": "credit_agreement:1",
                        "normalized": {"credit_agreement_id": "credit_agreement:1"},
                    }
                ],
            },
            {
                "name": "dataset_status",
                "row_count": 1,
                "rows": [
                    {
                        "record_id": "dataset_status:credit_agreements",
                        "normalized": {
                            "dataset_name": "credit_agreements",
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


def test_scanned_variant_does_not_merge_precomputed_auxiliary_records() -> None:
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

    assert content == {}


def test_personal_detail_dictionary_covers_profile_summary_public_absence_and_uncertainty() -> None:
    variant = PersonalDetailScannedVariant()
    dictionary = variant.data_dictionary()
    datasets = dictionary["datasets"]
    semantic = personal_detail_semantic_extensions()

    assert dictionary["version"] == "2.0.0"
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
        "subject_profile",
        "credit_business_overview",
        "tax_arrears_records",
        "civil_judgment_records",
        "enforcement_records",
        "administrative_penalty_records",
        "housing_fund_records",
        "professional_qualification_records",
        "administrative_award_records",
        "dataset_status",
        "field_observations",
        "extraction_issues",
    } <= set(datasets)
    assert set(PBOC_DATASET_ORDER) <= set(datasets)
    for dataset_name, reading_columns in semantic["dataset_reading_columns"].items():
        assert set(reading_columns) <= set(datasets[dataset_name]["columns"]), dataset_name
    contract = semantic["personal_detail_contract"]
    assert contract["canonical_profile_dataset"] == "subject_profile"
    assert contract["canonical_credit_summary_dataset"] == "credit_business_overview"
    assert contract["absence_dataset"] == "dataset_status"
    assert contract["uncertainty_dataset"] == "field_observations"
    assert contract["extraction_issue_dataset"] == "extraction_issues"
    assert contract["absence_requires_explicit_source_evidence"] is True
    assert contract["empty_dataset_means_absent"] is False
    assert semantic["domain_schema"]["version"] == "2.0.0"
    assert semantic["presentation_policy"]["enhanced_markdown_display"] == "full"
    assert semantic["personal_detail_contract"]["uncertainty_coverage"]["mode"] == (
        "potentially_flawed_only"
    )
    assert dictionary["datasets"]["credit_account_monthly_performance"]["columns"]["status_code"][
        "enum_ref"
    ] == "repayment_status_code"


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

    projected = prepare_personal_detail_source_collections(
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
    assert "work_phone" not in observations

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
    assert statuses["mobile_phone_records"]["presence_status"] == "unknown"
    assert statuses["mobile_phone_records"]["reason"] == "source_presence_not_established"
    assert "spouse_records" not in statuses
