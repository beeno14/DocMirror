# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Architecture contracts for the three credit-report variant packages."""

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
    assert detail.content_mode_is_expected("native_text") is False


def test_credit_report_variant_dataset_contracts_preserve_existing_outputs() -> None:
    personal_brief = resolve_credit_report_variant("personal_brief", "native_text")
    enterprise = resolve_credit_report_variant("enterprise", "native_text")
    personal_detail = resolve_credit_report_variant("personal_detail", "scanned_ocr")

    assert "credit_lines" not in personal_brief.dataset_names()
    assert enterprise.dataset_names()[1] == "credit_lines"
    assert personal_detail.dataset_names()[1] == "credit_lines"
    assert personal_brief.keep_query_institution is False
    assert enterprise.keep_query_institution is True
    assert personal_detail.keep_query_institution is True


def test_unknown_credit_report_variant_preserves_legacy_fallback_shape() -> None:
    unknown = resolve_credit_report_variant("unknown", "native_text")

    assert unknown.variant_id == "unknown"
    assert unknown.report_subtype == "unknown"
    assert "credit_lines" in unknown.dataset_names()
