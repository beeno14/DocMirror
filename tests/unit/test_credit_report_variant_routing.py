# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Architecture contracts for the three credit-report variant packages."""

from types import SimpleNamespace

from docmirror.plugins.credit_report.personal_brief_native import extraction as personal_brief_extraction
from docmirror.plugins.credit_report.variant_router import (
    registered_credit_report_variants,
    resolve_credit_report_variant,
)


def test_credit_report_registers_three_internal_document_variants() -> None:
    variants = registered_credit_report_variants()

    assert [variant.variant_id for variant in variants] == [
        "personal_brief_native",
        "enterprise_native",
        "personal_detail_scanned",
    ]
    assert [variant.report_subtype for variant in variants] == [
        "personal_brief",
        "enterprise",
        "personal_detail",
    ]


def test_credit_report_variant_routing_is_subtype_owned() -> None:
    assert resolve_credit_report_variant("personal_brief", "native_text").variant_id == "personal_brief_native"
    assert resolve_credit_report_variant("enterprise", "native_text").variant_id == "enterprise_native"
    assert resolve_credit_report_variant("personal_detail", "scanned_ocr").variant_id == "personal_detail_scanned"

    # Content-mode mismatches remain diagnosable but do not silently switch the
    # document family or execute another family's heuristics.
    detail = resolve_credit_report_variant("personal_detail", "native_text")
    assert detail.variant_id == "personal_detail_scanned"
    assert detail.content_mode_is_expected("native_text") is True


def test_credit_report_variant_dataset_contracts_preserve_existing_outputs() -> None:
    personal_brief = resolve_credit_report_variant("personal_brief", "native_text")
    enterprise = resolve_credit_report_variant("enterprise", "native_text")
    personal_detail = resolve_credit_report_variant("personal_detail", "scanned_ocr")

    assert "credit_lines" not in personal_brief.dataset_names()
    assert "enterprise_credit_facilities" in enterprise.dataset_names()
    assert "credit_lines" not in enterprise.dataset_names()
    assert "credit_agreements" in personal_detail.dataset_names()
    assert "credit_lines" not in personal_detail.dataset_names()
    assert personal_brief.keep_query_institution is False
    assert enterprise.keep_query_institution is True
    assert personal_detail.keep_query_institution is True


def test_personal_detail_presentation_overrides_are_variant_local() -> None:
    personal_brief = resolve_credit_report_variant("personal_brief", "native_text")
    enterprise = resolve_credit_report_variant("enterprise", "native_text")
    personal_detail = resolve_credit_report_variant("personal_detail", "native_text")

    overrides = personal_detail.semantic_extensions()["community_projection_overrides"]
    assert overrides["dataset_labels"]["report_metadata"] == "报告元数据"
    assert overrides["dataset_labels"]["credit_accounts"] == "信贷交易账户"
    assert overrides["section_markers"]["annotation_statements"] == ["statements", "annotations"]
    assert "community_projection_overrides" not in personal_brief.semantic_extensions()
    assert "community_projection_overrides" not in enterprise.semantic_extensions()


def test_unknown_credit_report_variant_preserves_legacy_fallback_shape() -> None:
    unknown = resolve_credit_report_variant("unknown", "native_text")

    assert unknown.variant_id == "unknown"
    assert unknown.report_subtype == "unknown"
    assert "credit_lines" in unknown.dataset_names()


def test_native_business_extraction_is_owned_by_non_enterprise_variants(monkeypatch) -> None:
    parse_result = SimpleNamespace()
    personal_brief = resolve_credit_report_variant("personal_brief", "native_text")
    enterprise = resolve_credit_report_variant("enterprise", "native_text")
    personal_detail = resolve_credit_report_variant("personal_detail", "scanned_ocr")
    monkeypatch.setattr(
        personal_brief_extraction,
        "extract_personal_brief_native_business",
        lambda _result, _text: {"owner": "personal_brief_native"},
    )
    assert personal_brief.extract_native_business(
        parse_result,
        "",
        content_mode="native_text",
    ) == {"owner": "personal_brief_native"}
    assert "extract_native_business" not in type(enterprise).__dict__
    assert (
        personal_detail.extract_native_business(
            parse_result,
            "",
            content_mode="scanned_ocr",
        )
        == {}
    )


def test_enterprise_semantics_do_not_inherit_personal_brief_identity_or_relationships() -> None:
    personal_brief = resolve_credit_report_variant("personal_brief", "native_text")
    enterprise = resolve_credit_report_variant("enterprise", "native_text")

    personal_dictionary = personal_brief.data_dictionary()
    enterprise_dictionary = enterprise.data_dictionary()
    personal_semantic = personal_brief.semantic_extensions()
    enterprise_semantic = enterprise.semantic_extensions()

    assert "id_number" in personal_dictionary["fields"]
    assert "marital_status" in personal_dictionary["fields"]
    assert "id_number" not in enterprise_dictionary["fields"]
    assert "marital_status" not in enterprise_dictionary["fields"]
    assert "institution_credit_code" not in personal_dictionary["fields"]
    assert enterprise_dictionary["fields"]["institution_credit_code"] == {
        "label": "机构信用代码",
        "type": "string",
        "format": "long_id",
        "sensitive": True,
    }
    assert "个人简版" in personal_dictionary["datasets"]["credit_accounts"]["definition"]
    assert "个人简版" not in enterprise_dictionary["datasets"]["enterprise_credit_accounts"]["definition"]
    assert personal_semantic["presentation_policy"]["classification"] == ("highly_sensitive_personal_financial_data")
    assert enterprise_semantic["presentation_policy"]["classification"] == ("sensitive_enterprise_credit_data")
    assert enterprise_semantic["dataset_relationships"]["enterprise_credit_facilities"]["relationship"] == (
        "independent_enterprise_facility_records"
    )
    assert personal_semantic["dataset_document_order"][:5] == [
        "personal_report_metadata",
        "report_notes",
        "identity_documents",
        "personal_credit_summary_records",
        "asset_disposition_records",
    ]
    assert personal_semantic["dataset_document_order"][-1] == "inquiry_records"
    enterprise_order = enterprise_semantic["dataset_document_order"]
    assert enterprise_order[:4] == [
        "enterprise_report_metadata",
        "report_notes",
        "enterprise_exchange_rates",
        "enterprise_report_identity",
    ]
    assert enterprise_order[4] == "enterprise_section_presence"
    assert (
        enterprise_order.index("enterprise_current_credit_summary")
        < enterprise_order.index("enterprise_facility_summary")
        < enterprise_order.index("enterprise_closed_credit_summary")
    )
    assert (
        enterprise_order.index("enterprise_profile")
        < enterprise_order.index("enterprise_capital_summary")
        < enterprise_order.index("enterprise_contributors")
    )
    assert (
        enterprise_order.index("enterprise_credit_accounts")
        < enterprise_order.index("enterprise_displayed_credit_summary")
        < enterprise_order.index("enterprise_credit_facilities")
    )
    assert "enterprise_extraction_audit" not in enterprise_order
    assert enterprise_dictionary["schema_id"] == "enterprise_credit_report"
    assert enterprise_dictionary["version"] == "3.0.0"
    assert "identity_documents" not in enterprise_dictionary["datasets"]
    assert "enterprise_credit_accounts" in enterprise_dictionary["datasets"]


def test_enterprise_official_public_record_lexicon_is_complete() -> None:
    enterprise = resolve_credit_report_variant("enterprise", "native_text")
    dictionary = enterprise.data_dictionary()
    dataset_layouts = enterprise.semantic_extensions()["enhanced_markdown"]["dataset_layouts"]

    assert dictionary["fields"]["extracted_public_record_type_counts"]["map_key_enum"] == "record_type"
    assert "public_records" not in dictionary["datasets"]
    assert "public_records" not in dataset_layouts
    assert list(dictionary["datasets"]["enterprise_public_license_records"]["columns"]) == [
        "sequence",
        "public_record_id",
        "licensing_authority",
        "license_type",
        "license_date",
        "license_expiry_date",
        "license_content",
        "source_page",
        "source_table_id",
    ]
    assert enterprise.semantic_extensions()["enhanced_markdown"]["dataset_layouts"][
        "enterprise_public_certification_records"
    ]["columns"] == [
        "sequence",
        "certification_authority",
        "certification_type",
        "certification_date",
        "certification_expiry_date",
        "certification_content",
    ]
    assert set(dictionary["enums"]["record_type"]) == {
        "non_credit_accounts",
        "utility_payment",
        "tax_arrears",
        "civil_judgment",
        "enforcement",
        "administrative_penalty",
        "social_security_payment",
        "license",
        "certification",
        "qualification",
        "award",
        "export_quality",
        "inspection_exemption",
        "regulatory_supervision",
        "patent",
        "financing_restriction",
        "data_provider_statement",
        "credit_bureau_statement",
        "subject_statement",
        "dispute_annotation",
    }
    assert all(dictionary["enums"]["record_type"].values())
