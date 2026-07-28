# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-variant credit-report schema integrity tests."""

from docmirror.plugins.credit_report.contracts import CONTENT_MODE_SCANNED
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
    assert set(content["datasets"]) == {
        "residence_records",
        "employment_records",
        "statements",
        "annotations",
    }
