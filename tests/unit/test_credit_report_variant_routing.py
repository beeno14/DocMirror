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
    personal_semantic = personal_brief.semantic_extensions()
    personal_overrides = personal_semantic["community_projection_overrides"]
    assert personal_overrides["section_markers"]["personal_report_metadata"] == [
        "report_header"
    ]
    assert personal_overrides["section_markers"]["report_notes"] == ["notes"]
    assert "appendix" not in personal_semantic["enhanced_markdown"]
    assert personal_semantic["enhanced_markdown"]["dataset_layouts"]["public_records"] == {
        "hidden": True
    }
    personal_layouts = personal_semantic["enhanced_markdown"]["dataset_layouts"]
    assert personal_layouts["overdue_records"] == {"hidden": True}
    account_partitions = personal_layouts["credit_accounts"]["partitions"]
    assert [partition["prepend_partition"]["title"] for partition in account_partitions] == [
        "信用卡逾期记录",
        "贷款逾期记录",
        "其他业务逾期记录",
    ]
    assert all(
        partition["prepend_partition"]["dataset"] == "overdue_records"
        and partition["prepend_partition"]["join_on"] == "account_id"
        for partition in account_partitions
    )
    for other_semantic in (
        enterprise.semantic_extensions(),
        personal_detail.semantic_extensions(),
    ):
        other_layouts = other_semantic.get("enhanced_markdown", {}).get(
            "dataset_layouts", {}
        )
        assert not any(
            "prepend_partition" in partition
            for layout in other_layouts.values()
            if isinstance(layout, dict)
            for partition in layout.get("partitions", [])
            if isinstance(partition, dict)
        )
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
        "identity_documents",
        "personal_credit_summary_records",
        "asset_disposition_records",
        "guarantor_compensation_records",
    ]
    assert personal_semantic["dataset_document_order"][-2:] == [
        "inquiry_records",
        "report_notes",
    ]
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
        enterprise_order.index("enterprise_displayed_credit_summary")
        < enterprise_order.index("enterprise_credit_accounts")
        < enterprise_order.index("enterprise_credit_facilities")
    )
    assert "enterprise_extraction_audit" not in enterprise_order
    assert enterprise_dictionary["schema_id"] == "enterprise_credit_report"
    assert enterprise_dictionary["version"] == "4.0.0"
    assert "identity_documents" not in enterprise_dictionary["datasets"]
    assert "enterprise_credit_accounts" in enterprise_dictionary["datasets"]


def test_enterprise_enhanced_markdown_uses_business_only_allowlists() -> None:
    enterprise = resolve_credit_report_variant("enterprise", "native_text")
    semantic = enterprise.semantic_extensions()
    enhanced = semantic["enhanced_markdown"]
    layouts = enhanced["dataset_layouts"]

    assert enhanced["suppress_empty_columns"] is True
    assert layouts["report_notes"] == {"hidden": True}
    assert layouts["enterprise_section_presence"] == {"hidden": True}
    assert layouts["enterprise_report_identity"] == {"hidden": True}
    assert layouts["enterprise_credit_overview"]["columns"] == [
        "source_display_limited", "source_display_scopes",
    ]
    assert layouts["enterprise_credit_overview"]["hide_record_titles"] is True
    assert set(enterprise.dataset_names()) == set(layouts)
    assert all(layout.get("hidden") or layout.get("columns") for layout in layouts.values())
    assert enhanced["section_layouts"]["identity"]["omit_unlisted"] is True
    assert enhanced["section_layouts"]["credit_summary"]["omit_unlisted"] is True

    technical_columns = {
        "sequence",
        "source_page",
        "source_page_end",
        "source_table_id",
        "source_table_id_end",
        "presence_status",
        "heading_detected",
        "count_scope",
        "represented_dataset",
        "reported_record_count_status",
        "continuation_complete",
    }
    for layout in layouts.values():
        assert not (technical_columns & set(layout.get("columns") or ()))

    detail_groups = layouts["enterprise_credit_detail_groups"]
    assert detail_groups["title_fields"] == ["group_phase", "business_category"]
    assert enterprise.data_dictionary()["enums"]["group_phase"] == {
        "active": "未结清信贷",
        "recovered": "被追偿",
        "settled": "已结清信贷",
        "repayment_responsibility": "相关还款责任",
    }


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
        "housing_fund_payment",
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


def test_enterprise_attachment_amount_schema_prefers_typed_business_fields() -> None:
    enterprise = resolve_credit_report_variant("enterprise", "native_text")
    dictionary = enterprise.data_dictionary()
    columns = dictionary["datasets"]["enterprise_attachment_credit_details"]["columns"]
    layout = enterprise.semantic_extensions()["enhanced_markdown"]["dataset_layouts"][
        "enterprise_attachment_credit_details"
    ]["columns"]

    assert {
        "amount",
        "amount_kind",
        "credit_limit",
        "loan_amount",
        "discount_amount",
        "instrument_amount",
        "guarantee_amount",
        "balance",
        "risk_exposure_amount",
    } <= set(columns)
    assert columns["amount"]["label"] == "金额"
    assert "amount_kind" in columns["amount"]["definition"]
    assert "amount" not in layout
    assert "amount_kind" in layout
    assert "instrument_amount" in layout
    assert "guarantee_amount" in layout
