# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from docmirror.models.entities.parse_result import PageContent, TableBlock, TextBlock
from docmirror.models.schemas.registry import validate_projection_payload
from docmirror.plugins.credit_report.enterprise_native.extraction_validation import (
    build_enterprise_extraction_report,
)
from docmirror.plugins.credit_report.enterprise_native.ir import (
    CanonicalEnterpriseDocumentIR,
    build_canonical_enterprise_document,
)
from docmirror.plugins.credit_report.enterprise_native.pipeline import (
    _dataset_completeness,
    _normalize_passthrough,
    extract_enterprise_semantic_document,
    run_enterprise_pipeline,
)
from docmirror.plugins.credit_report.enterprise_native.projector import derive_enterprise_projection
from docmirror.plugins.credit_report.enterprise_native.source_quality import (
    assess_enterprise_source_information,
)


def _enterprise_result() -> SimpleNamespace:
    return SimpleNamespace(
        pages=[
            PageContent(
                page_number=1,
                texts=[TextBlock(content="企业信用报告（自主查询版）\n身份标识")],
                tables=[
                    TableBlock(
                        table_id="identity",
                        metadata={
                            "raw_rows": [
                                ["企业名称", "示例企业有限公司"],
                                ["统一社会信用代码", "91110000123456789X"],
                            ]
                        },
                    )
                ],
            )
        ],
        confidence=0.99,
    )


def _all_mapping_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        keys.update(map(str, value))
        for nested in value.values():
            keys.update(_all_mapping_keys(nested))
    elif isinstance(value, (list, tuple)):
        for nested in value:
            keys.update(_all_mapping_keys(nested))
    return keys


def test_enterprise_pipeline_retains_private_ir_and_semantic_json() -> None:
    artifacts = run_enterprise_pipeline(_enterprise_result())

    assert isinstance(artifacts.document_ir, CanonicalEnterpriseDocumentIR)
    assert not hasattr(artifacts.document_ir, "parse_result")
    assert not hasattr(artifacts.document_ir, "source_result")

    ir_payload = json.loads(artifacts.ir_debug_json)
    semantic_payload = json.loads(artifacts.semantic_debug_json)
    assert ir_payload["schema"]["id"] == "canonical_enterprise_document_ir"
    assert ir_payload["content_conserved"] is True
    assert ir_payload["components"]
    assert ir_payload["table_rows"][0]["table_id"] == "identity"
    assert semantic_payload["schema"]["id"] == "enterprise_credit_report"
    assert semantic_payload["extraction"]["protocol"] == "pboc-enterprise-extraction-failure"
    assert semantic_payload["extraction"]["status"] == "complete"
    assert semantic_payload["facts"]["subject_name"] == "示例企业有限公司"


def test_enterprise_schema_rejects_parse_result_input() -> None:
    with pytest.raises(TypeError, match="CanonicalEnterpriseDocumentIR"):
        extract_enterprise_semantic_document(_enterprise_result())  # type: ignore[arg-type]


def test_enterprise_public_projection_does_not_expose_debug_json() -> None:
    projection = derive_enterprise_projection(
        SimpleNamespace(projector_id="credit_report", domain_name="credit_report"),
        _enterprise_result(),
    )

    keys = _all_mapping_keys(projection.model_dump(mode="python"))
    assert "ir_debug_json" not in keys
    assert "semantic_debug_json" not in keys
    assert "document_ir" not in keys
    assert "semantic_document" not in keys


def test_enterprise_projection_contains_structured_extraction_report() -> None:
    projection = derive_enterprise_projection(
        SimpleNamespace(projector_id="credit_report", domain_name="credit_report"),
        _enterprise_result(),
    )

    report = projection.domain_facts["extraction_report"]
    assert report["protocol"] == "pboc-enterprise-extraction-failure"
    assert report["version"] == "1.0.0"
    assert report["status"] == "complete"
    assert report["failures"] == []


def test_build_canonical_enterprise_document_is_idempotent() -> None:
    document = build_canonical_enterprise_document(_enterprise_result())
    assert build_canonical_enterprise_document(document) is document


def test_f1_uses_exact_heading_blocks_instead_of_prose_substrings() -> None:
    result = _enterprise_result()
    result.pages.append(
        PageContent(
            page_number=2,
            texts=[TextBlock(content="本报告所展示的基本信息来自多个数据来源。")],
        )
    )

    semantic = run_enterprise_pipeline(result).semantic_document
    section_types = {section["type"] for section in semantic.sections}

    assert "report_metadata" in section_types
    assert "identity" in section_types
    assert "basic_information" not in section_types


def test_f3_normalizes_source_placeholders_to_typed_null_and_status() -> None:
    record = _normalize_passthrough(
        {"certification_date": "--", "certification_content": "质量体系认证"}
    )

    assert record["normalized"]["certification_date"] is None
    assert record["normalized"]["certification_date_status"] == "not_reported"
    assert record["normalized"]["certification_content"] == "质量体系认证"


def test_f4_flags_bad_parseresult_input_in_the_canonical_ir() -> None:
    document = build_canonical_enterprise_document(SimpleNamespace(pages=[], confidence=0.0))

    assert document.input_quality_flags[0]["code"] == "ENTERPRISE_INPUT_NO_PAGES"
    assert document.input_quality_flags[0]["status"] == "bad_input"


def test_f4_flags_suspicious_parser_glyphs_for_targeted_reextraction() -> None:
    result = _enterprise_result()
    result.pages[0].texts.append(TextBlock(content="历史表现：숨맒숭"))

    document = build_canonical_enterprise_document(result)

    assert any(
        flag["code"] == "ENTERPRISE_INPUT_SUSPICIOUS_GLYPHS"
        and flag["status"] == "possibly_misdecoded"
        for flag in document.input_quality_flags
    )


def test_f6_semantic_document_contains_only_canonical_enterprise_views() -> None:
    semantic = run_enterprise_pipeline(_enterprise_result()).semantic_document

    assert "credit_accounts" not in semantic.datasets
    assert "credit_lines" not in semantic.datasets
    assert "repayment_liability_records" not in semantic.datasets
    assert "enterprise_profile_fields" not in semantic.datasets
    assert "enterprise_stakeholders" not in semantic.datasets
    assert "enterprise_extraction_audit" not in semantic.datasets
    assert semantic.continuation_audit


def test_f4_flags_source_truncation_without_inventing_missing_text() -> None:
    document = build_canonical_enterprise_document(_enterprise_result())
    address = "福建省福州市仓山区临江街道六一南路与朝阳路交叉处中"

    flags = assess_enterprise_source_information(
        document,
        {
            "enterprise_profile": [
                {
                    "operating_address": address,
                    "source_refs": [{"source_page": 3}],
                }
            ]
        },
    )

    truncation = next(flag for flag in flags if flag.code == "ENTERPRISE_SOURCE_FIELD_TRUNCATED")
    assert truncation.status == "source_truncated"
    assert truncation.details["source_value"] == address


def test_f5_does_not_compare_a_scoped_audit_count_to_a_whole_dataset() -> None:
    rows = [{"source_page": 7} for _ in range(43)]
    scoped_audit = [
        {
            "continuation_family": "attachment_credit_detail",
            "business_category": "已结清业务",
            "expected_record_count": 29,
            "extracted_record_count": 29,
            "unresolved_record_count": 0,
            "unexpected_record_count": 0,
        }
    ]

    completeness = _dataset_completeness(
        {"enterprise_attachment_credit_details": rows},
        scoped_audit,
        (),
    )["enterprise_attachment_credit_details"]

    assert completeness["expected_row_count"] == 43
    assert completeness["verified"] is True
    assert completeness["basis"] == "canonical_source_component_count"


def _validation_dictionary() -> dict[str, Any]:
    return {
        "datasets": {
            "enterprise_profile": {
                "columns": {
                    "operating_address": {
                        "label": "\u529e\u516c/\u7ecf\u8425\u5730\u5740",
                        "type": "string",
                    }
                }
            },
            "enterprise_public_record_counts": {
                "columns": {
                    "record_type": {"label": "\u8bb0\u5f55\u7c7b\u578b", "type": "string"},
                    "record_count": {"label": "\u8bb0\u5f55\u6761\u6570", "type": "integer"},
                }
            },
        },
        "enums": {},
    }


def test_failure_protocol_reports_populated_source_field_missing_from_semantic_data() -> None:
    result = SimpleNamespace(
        pages=[
            PageContent(
                page_number=1,
                tables=[
                    TableBlock(
                        table_id="profile",
                        metadata={
                            "raw_rows": [["\u529e\u516c/\u7ecf\u8425\u5730\u5740", "\u5317\u4eac\u5e02\u6d77\u6dc0\u533a"]]
                        },
                    )
                ],
            )
        ],
        confidence=1.0,
    )
    document = build_canonical_enterprise_document(result)
    report = build_enterprise_extraction_report(
        document,
        {
            "enterprise_profile": [
                {
                    "enterprise_profile_id": "profile:1",
                    "operating_address": None,
                    "operating_address_status": "not_reported",
                    "source_refs": [
                        {"source": "canonical_physical_table", "page": 1, "table_id": "profile", "row": 0}
                    ],
                    "normalized": {
                        "enterprise_profile_id": "profile:1",
                        "operating_address": None,
                        "operating_address_status": "not_reported",
                    },
                }
            ]
        },
        continuation_audit=(),
        dataset_completeness={},
        data_dictionary=_validation_dictionary(),
    ).to_payload()

    assert report["status"] == "partial"
    assert report["summary"]["checked_field_count"] == 1
    assert report["summary"]["failed_field_count"] == 1
    assert report["failures"][0]["code"] == "EXPECTED_FIELD_NOT_EXTRACTED"
    assert report["failures"][0]["path"] == "/data/enterprise_profile/profile:1/operating_address"
    assert report["failures"][0]["evidence"]["source_values"] == ["\u5317\u4eac\u5e02\u6d77\u6dc0\u533a"]


def test_failure_protocol_accepts_explicit_source_placeholder_and_zero_record_section() -> None:
    result = SimpleNamespace(
        pages=[
            PageContent(
                page_number=1,
                tables=[
                    TableBlock(
                        table_id="profile",
                        metadata={"raw_rows": [["\u529e\u516c/\u7ecf\u8425\u5730\u5740", "--"]]},
                    )
                ],
            )
        ],
        confidence=1.0,
    )
    document = build_canonical_enterprise_document(result)
    report = build_enterprise_extraction_report(
        document,
        {
            "enterprise_profile": [
                {
                    "enterprise_profile_id": "profile:1",
                    "normalized": {
                        "enterprise_profile_id": "profile:1",
                        "operating_address": None,
                        "operating_address_status": "not_reported",
                    },
                }
            ],
            "enterprise_public_record_counts": [
                {
                    "normalized": {"record_type": "tax_arrears", "record_count": 0},
                    "source_page": 1,
                }
            ],
        },
        continuation_audit=(),
        dataset_completeness={},
        data_dictionary=_validation_dictionary(),
    ).to_payload()

    assert report["status"] == "complete"
    assert report["failures"] == []


def test_failure_protocol_treats_numeric_zero_as_a_retained_value() -> None:
    result = SimpleNamespace(
        pages=[
            PageContent(
                page_number=1,
                tables=[
                    TableBlock(
                        table_id="overview",
                        metadata={"raw_rows": [["发生信贷交易的机构数", "0"]]},
                    )
                ],
            )
        ],
        confidence=1.0,
    )
    document = build_canonical_enterprise_document(result)
    dictionary = {
        "datasets": {
            "enterprise_credit_overview": {
                "columns": {
                    "credit_institution_count": {
                        "label": "发生信贷交易的机构数",
                        "type": "integer",
                    }
                }
            }
        },
        "enums": {},
    }
    report = build_enterprise_extraction_report(
        document,
        {
            "enterprise_credit_overview": [
                {
                    "credit_overview_id": "overview:1",
                    "normalized": {"credit_institution_count": 0},
                }
            ]
        },
        continuation_audit=(),
        dataset_completeness={},
        data_dictionary=dictionary,
    ).to_payload()

    assert report["status"] == "complete"
    assert report["summary"]["checked_field_count"] == 1
    assert report["summary"]["satisfied_field_count"] == 1
    assert report["failures"] == []


def test_failure_protocol_reports_continuation_record_mismatch() -> None:
    document = build_canonical_enterprise_document(_enterprise_result())
    report = build_enterprise_extraction_report(
        document,
        {"enterprise_attachment_accounts": [{"source_page": 1}]},
        continuation_audit=(
            {
                "continuation_family": "attachment_account",
                "expected_record_count": 2,
                "extracted_record_count": 1,
                "unresolved_record_count": 1,
                "unexpected_record_count": 0,
            },
        ),
        dataset_completeness={},
        data_dictionary={"datasets": {}, "enums": {}},
    ).to_payload()

    assert report["status"] == "partial"
    assert report["failures"][0]["code"] == "RECORD_RECONSTRUCTION_MISMATCH"
    assert report["failures"][0]["dataset"] == "enterprise_attachment_accounts"


def test_failure_protocol_reports_bad_input_as_failed() -> None:
    document = build_canonical_enterprise_document(SimpleNamespace(pages=[], confidence=0.0))
    report = build_enterprise_extraction_report(
        document,
        {},
        continuation_audit=(),
        dataset_completeness={},
        data_dictionary={"datasets": {}, "enums": {}},
    ).to_payload()

    assert report["status"] == "failed"
    assert report["failures"][0]["code"] == "INPUT_INTEGRITY_VIOLATION"


def test_enterprise_semantic_schema_requires_the_failure_protocol() -> None:
    pytest.importorskip("jsonschema")
    payload = {
        "schema": {
            "name": "docmirror.community.semantic",
            "version": "1.0.0",
            "edition": "community",
            "document_type": "enterprise_credit_report",
        },
        "source": {"parse_result_schema": "docmirror.parse_result", "fingerprint": "abc"},
        "classification": {
            "document_type": "enterprise_credit_report",
            "projector_id": "credit_report",
            "support_level": "canonical",
        },
        "document": {
            "id": "doc-1",
            "type": "enterprise_credit_report",
            "title": "企业信用报告",
            "page_count": 1,
            "language": "zh-CN",
            "source_file": "enterprise.pdf",
            "units": [],
        },
        "structure": {"sections": [], "blocks": [], "source_tables": []},
        "datasets": [],
        "bindings": [],
        "domain": {},
        "reading": {},
        "files": {
            "content_md": "content.md",
            "enhanced_reading_md": "enhanced_reading.md",
            "datasets_dir": "datasets",
            "dataset_audit_csv": "datasets/_audit_cells.csv",
        },
        "warnings": [],
        "diagnostics": {},
    }

    missing = validate_projection_payload("community_semantic", payload)
    assert not missing.valid
    payload["extraction"] = {
        "protocol": "pboc-enterprise-extraction-failure",
        "version": "1.0.0",
        "status": "complete",
        "summary": {
            "failure_count": 0,
            "warning_count": 0,
            "checked_field_count": 0,
            "satisfied_field_count": 0,
            "failed_field_count": 0,
            "record_contract_count": 0,
            "source_component_count": 0,
            "source_components_conserved": True,
        },
        "failures": [],
    }
    assert validate_projection_payload("community_semantic", payload).valid
