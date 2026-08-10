from __future__ import annotations

import csv
import io
import json

from docmirror.models.entities.parse_result import (
    CellValue,
    DocumentEntities,
    KeyValuePair,
    PageContent,
    ParseResult,
    RowType,
    TableBlock,
    TableRow,
    TextBlock,
    TextLevel,
)
from docmirror.models.schemas.registry import validate_projection_payload
from docmirror.models.sealed import seal_parse_result
from docmirror.output.community_bundle import (
    _dataset_section_id,
    render_community_reading_markdown,
)
from docmirror.output.community_bundle import (
    project_community_bundle as _project_community_bundle,
)
from docmirror.plugins._base.projector import load_projection_policy
from docmirror.plugins.bank_statement.community_plugin import BANK_DATA_DICTIONARY

_PROJECTIONS: dict[int, dict] = {}


def project_community_bundle(result, **kwargs):
    projection_data = _PROJECTIONS.get(id(result))
    document_type = str((projection_data or {}).get("document_type") or "generic")
    policy = load_projection_policy(f"docmirror.plugins.{document_type}")
    return _project_community_bundle(
        seal_parse_result(result),
        projection_data=projection_data,
        projection_policy=policy,
        **kwargs,
    )


def _candidate(records: list[dict] | None = None) -> dict:
    return {
        "document": {
            "document_type": "credit_report",
            "document_name": "个人征信.pdf",
            "page_count": 1,
            "language": "zh",
            "properties": {"report_subtype": "personal_detailed"},
        },
        "plugin": {"name": "credit_report"},
        "metadata": {"route_type": "core_domain", "domain_status": "ga"},
        "status": {"success": True, "warnings": [], "errors": []},
        "quality": {"issues": []},
        "data": {
            "fields": {"subject_name": "洪晓鑫"},
            "field_details": {"subject_name": {"raw": "洪晓鑫"}},
            "sections": [{"id": "sec_credit", "title": "信贷记录明细", "source_page_start": 1}],
            "tables": [
                {
                    "id": "table:repayment_records",
                    "section_id": "sec_credit",
                    "data_ref": {"path": "/data/repayment_records"},
                }
            ],
            "repayment_records": records or [],
            "data_dictionary": {
                "fields": {"subject_name": {"label": "姓名", "type": "string"}},
                "datasets": {
                    "repayment_records": {
                        "columns": {
                            "month": {"label": "月份", "type": "string"},
                            "status": {"label": "还款状态", "type": "string"},
                        }
                    }
                },
            },
        },
    }


def _with_projection(result: ParseResult, candidate: dict) -> ParseResult:
    data = candidate["data"]
    fields = dict(data.get("fields") or {})
    datasets: dict[str, list[dict]] = {}
    for key, rows in data.items():
        if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
            continue
        datasets[key] = [
            {
                **dict(row),
                "record_id": str(row.get("record_id") or f"{key}:r{index:06d}"),
            }
            for index, row in enumerate(rows, start=1)
        ]
    existing_type = str(result.entities.document_type or "")
    _PROJECTIONS[id(result)] = {
        "projector_id": "test-fixture",
        "document_type": existing_type
        if existing_type not in {"", "generic", "unknown"}
        else candidate["document"]["document_type"],
        "entity_fields": {"subject_name": fields["subject_name"]} if fields.get("subject_name") else {},
        "domain_facts": {
            **candidate["document"].get("properties", {}),
            **fields,
            "field_details": data.get("field_details", {}),
            "data_dictionary": data.get("data_dictionary", {}),
        },
        "datasets": datasets,
        "sections": tuple(data.get("sections") or ()),
    }
    return result


def test_public_json_has_reading_model_and_complete_dataset_rows() -> None:
    records = [{"repayment_id": f"rep_{i}", "month": f"2025-{i:02d}", "status": "N"} for i in range(1, 13)]
    result = _with_projection(
        ParseResult(entities=DocumentEntities(document_type="credit_report")),
        _candidate(records),
    )
    bundle = project_community_bundle(result, file_id="001", document_id="doc_test")
    semantic = bundle.semantic_payload()
    payload = bundle.json_payload(semantic)

    assert set(payload) == {"schema", "document", "sections", "datasets", "reading", "files", "warnings"}
    assert semantic["schema"] == {
        "name": "docmirror.community.semantic",
        "version": "1.0.0",
        "edition": "community",
        "document_type": "personal_credit_report_detailed",
    }
    assert semantic["source"]["fingerprint"]
    assert semantic["classification"]["projector_id"]
    assert semantic["datasets"] == payload["datasets"]
    assert len(semantic["bindings"]) == 12
    assert {binding["record_id"] for binding in semantic["bindings"]} == {
        row["record_id"] for row in semantic["datasets"][0]["rows"]
    }
    assert semantic["reading"] == payload["reading"]
    assert semantic["structure"]["sections"][0]["id"] == payload["sections"][0]["id"]
    assert validate_projection_payload("community_semantic", semantic).valid
    assert payload["schema"]["version"] == "3.0.0"
    assert payload["schema"]["domain"] == "personal_credit_report_detailed"
    assert payload["datasets"][0]["row_count"] == 12
    assert len(payload["datasets"][0]["rows"]) == 12
    assert payload["datasets"][0]["primary_key"] == "record_id"
    assert payload["datasets"][0]["completeness"] == {
        "expected_row_count": 12,
        "emitted_row_count": 12,
        "omitted_row_count": 0,
        "verified": False,
        "basis": "emitted_records_only",
    }
    assert payload["datasets"][0]["status"] == "partial"
    assert payload["warnings"] == [
        {
            "code": "DATASET_COMPLETENESS_UNVERIFIED",
            "level": "warning",
            "message": "dataset ds_repayment_records has 12 emitted records but no independent source count",
            "dataset_id": "ds_repayment_records",
        }
    ]
    assert [row["record_id"] for row in payload["datasets"][0]["rows"]] == [
        f"repayment_records:r{index:06d}" for index in range(1, 13)
    ]
    assert payload["datasets"][0]["rows"][0]["normalized"]["month"] == "2025-01"
    assert payload["reading"]["profile"] == "community"
    assert payload["reading"]["tables"][0]["dataset_id"] == payload["datasets"][0]["id"]
    assert validate_projection_payload("community", payload).valid
    duplicated_payload = dict(payload)
    duplicated_payload["public_records"] = {}
    assert not validate_projection_payload("community", duplicated_payload).valid
    duplicated_semantic = dict(semantic)
    duplicated_semantic["public_records"] = {}
    assert not validate_projection_payload(
        "community_semantic",
        duplicated_semantic,
    ).valid


def test_dataset_section_fallback_uses_first_source_page() -> None:
    sections = [
        {"id": "sec_report", "type": "report_metadata", "page_range": [1, 1]},
        {"id": "sec_public", "type": "public_records", "page_range": [6, 6]},
        {"id": "sec_supplement", "type": "credit_supplement", "page_range": [8, 8]},
    ]
    data = {
        "enterprise_public_unconfigured_records": [
            {"source_page": 7},
            {"source_page": 6},
        ]
    }

    assert (
        _dataset_section_id(
            data,
            "enterprise_public_unconfigured_records",
            sections,
            {},
        )
        == "sec_public"
    )


def test_dataset_document_order_is_applied_only_when_configured() -> None:
    candidate = _candidate()
    candidate["data"]["first_records"] = [{"value": "first"}]
    candidate["data"]["second_records"] = [{"value": "second"}]
    result = _with_projection(
        ParseResult(entities=DocumentEntities(document_type="credit_report")),
        candidate,
    )
    _PROJECTIONS[id(result)]["semantic"] = {
        "dataset_document_order": ["second_records", "first_records"],
    }

    payload = project_community_bundle(result).json_payload()

    assert [dataset["name"] for dataset in payload["datasets"]] == [
        "second_records",
        "first_records",
    ]


def test_enhanced_markdown_omits_sections_without_renderable_content() -> None:
    payload = {
        "schema": {"name": "docmirror.community"},
        "document": {"id": "doc_empty_section", "title": "测试报告", "type": "generic"},
        "domain": {
            "extensions": {
                "enhanced_markdown": {
                    "suppress_empty_sections": True,
                }
            }
        },
        "sections": [
            {
                "id": "sec_empty",
                "title": "空章节",
                "type": "empty",
                "items": [],
                "groups": [],
                "dataset_refs": [],
            },
            {
                "id": "sec_data",
                "title": "有内容章节",
                "type": "details",
                "items": [
                    {
                        "key": "status",
                        "label": "状态",
                        "value": "正常",
                        "type": "string",
                    }
                ],
                "groups": [],
                "dataset_refs": [],
            },
        ],
        "datasets": [],
        "reading": {
            "document_flow": [
                {"order": 1, "kind": "document", "ref_id": "doc_empty_section"},
                {"order": 2, "kind": "section", "ref_id": "sec_empty"},
                {"order": 3, "kind": "section", "ref_id": "sec_data"},
            ],
            "tables": [],
        },
    }

    enhanced = render_community_reading_markdown(payload)

    assert "## 空章节" not in enhanced
    assert "## 有内容章节" in enhanced
    assert "**状态:** 正常" in enhanced


def test_semantic_extension_can_override_dataset_reading_columns() -> None:
    result = _with_projection(
        ParseResult(entities=DocumentEntities(document_type="credit_report")),
        _candidate(
            [
                {
                    "repayment_id": "rep_1",
                    "month": "2025-01",
                    "status": "N",
                }
            ]
        ),
    )
    _PROJECTIONS[id(result)]["semantic"] = {
        "dataset_reading_columns": {
            "repayment_records": ["status"],
        }
    }

    semantic = project_community_bundle(result).semantic_payload()
    reading_table = next(
        table
        for table in semantic["reading"]["tables"]
        if table["dataset_id"] == "ds_repayment_records"
    )

    assert reading_table["column_keys"] == ["status"]


def test_source_report_count_can_verify_dataset_completeness() -> None:
    candidate = _candidate()
    candidate["data"]["credit_accounts"] = [
        {"account_id": "account:1"},
        {"account_id": "account:2"},
    ]
    result = _with_projection(
        ParseResult(entities=DocumentEntities(document_type="credit_report")),
        candidate,
    )
    _PROJECTIONS[id(result)]["domain_facts"]["credit_summary"] = {
        "reported_account_count": 2,
    }

    payload = project_community_bundle(result, document_id="doc_verified").json_payload()
    dataset = next(item for item in payload["datasets"] if item["name"] == "credit_accounts")

    assert dataset["status"] == "complete"
    assert dataset["completeness"] == {
        "expected_row_count": 2,
        "emitted_row_count": 2,
        "omitted_row_count": 0,
        "verified": True,
        "basis": "source_report_summary",
    }
    assert payload["warnings"] == []


def test_bank_footer_count_can_verify_transaction_dataset_completeness() -> None:
    result = ParseResult(entities=DocumentEntities(document_type="bank_statement"))
    _PROJECTIONS[id(result)] = {
        "projector_id": "bank_statement",
        "document_type": "bank_statement",
        "domain_facts": {
            "source_reported_transaction_count": 2,
        },
        "datasets": {
            "records": [
                {"record_id": "transactions:r000001", "normalized": {"date": "2025-01-01"}},
                {"record_id": "transactions:r000002", "normalized": {"date": "2025-01-02"}},
            ]
        },
    }

    payload = project_community_bundle(result, document_id="doc_bank_verified").json_payload()
    dataset = next(item for item in payload["datasets"] if item["name"] == "transactions")

    assert dataset["status"] == "complete"
    assert dataset["completeness"] == {
        "expected_row_count": 2,
        "emitted_row_count": 2,
        "omitted_row_count": 0,
        "verified": True,
        "basis": "source_footer_transaction_count",
    }
    assert payload["warnings"] == []


def test_bank_degraded_status_blocks_dataset_verification_even_when_row_count_matches() -> None:
    result = ParseResult(entities=DocumentEntities(document_type="bank_statement"))
    _PROJECTIONS[id(result)] = {
        "projector_id": "bank_statement",
        "document_type": "bank_statement",
        "domain_facts": {
            "expected_primary_rows": 2,
            "extract_status": "degraded",
        },
        "datasets": {
            "records": [
                {"record_id": "transactions:r000001", "normalized": {"date": "2025-01-01"}},
                {"record_id": "transactions:r000002", "normalized": {"date": "2025-01-02"}},
            ]
        },
    }

    payload = project_community_bundle(result, document_id="doc_bank_degraded").json_payload()
    dataset = next(item for item in payload["datasets"] if item["name"] == "transactions")

    assert dataset["status"] == "partial"
    assert dataset["completeness"]["expected_row_count"] == 2
    assert dataset["completeness"]["emitted_row_count"] == 2
    assert dataset["completeness"]["verified"] is False
    assert any(warning["code"] == "DATASET_VERIFICATION_BLOCKED" for warning in payload["warnings"])


def test_semantic_schema_allows_document_type_specific_extensions() -> None:
    result = _with_projection(
        ParseResult(entities=DocumentEntities(document_type="credit_report")),
        _candidate([{"month": "2025-01", "status": "N"}]),
    )
    bundle = project_community_bundle(result, document_id="doc_semantic_extensions")
    semantic = bundle.semantic_payload()
    semantic["domain"]["credit_report"] = {
        "reported_summary_grid": {
            "headers": [["信用卡", "贷款"], ["", "购房", "其他"]],
            "cells": [{"metric": "账户数", "values": [21, 2, 22]}],
        },
        "说明": ["公开语义扩展可以因文档类型而异。"],
    }
    semantic["structure"]["sections"][0]["document_specific_role"] = "credit_summary"

    assert validate_projection_payload("community_semantic", semantic).valid
    assert validate_projection_payload("community", bundle.json_payload(semantic)).valid


def test_renderers_use_the_same_frozen_semantic_result() -> None:
    result = _with_projection(
        ParseResult(entities=DocumentEntities(document_type="credit_report")),
        _candidate([{"month": "2025-01", "status": "N"}]),
    )
    bundle = project_community_bundle(result, document_id="doc_semantic_source")
    semantic = bundle.semantic_payload()
    semantic_record_id = semantic["datasets"][0]["rows"][0]["record_id"]

    bundle.datasets[0].rows.clear()
    community = bundle.json_payload(semantic)
    csv_rows = list(
        csv.DictReader(
            io.StringIO(bundle.render_dataset_csvs(semantic)[semantic["datasets"][0]["csv"]].lstrip("\ufeff"))
        )
    )
    enhanced = bundle.render_enhanced_markdown(semantic)

    assert community["datasets"][0]["rows"][0]["record_id"] == semantic_record_id
    assert csv_rows[0]["record_id"] == semantic_record_id
    assert "| 2025-01 | N |" in enhanced


def test_markdown_contains_every_physical_table_row_without_preview_limit() -> None:
    rows = [
        TableRow(cells=[CellValue(text=str(index)), CellValue(text=f"状态{index}")], row_type=RowType.DATA)
        for index in range(1, 177)
    ]
    rows.append(TableRow(cells=[CellValue(text="合计"), CellValue(text="176")], row_type=RowType.SUMMARY))
    result = ParseResult(
        pages=[
            PageContent(
                page_number=1,
                texts=[TextBlock(content="个人信用报告", level=TextLevel.TITLE, bbox=[0, 0, 100, 10])],
                key_values=[KeyValuePair(key="姓名", value="洪晓鑫", bbox=[0, 12, 100, 20])],
                tables=[TableBlock(headers=["序号", "还款状态"], rows=rows, bbox=[0, 30, 100, 700])],
            )
        ],
        entities=DocumentEntities(document_type="credit_report"),
    )
    _with_projection(result, _candidate())
    bundle = project_community_bundle(result, document_id="doc_test")
    markdown = bundle.render_markdown()
    semantic = bundle.semantic_payload()

    assert "# 个人信用报告" in markdown
    assert "**姓名:** 洪晓鑫" in markdown
    for index in range(1, 177):
        assert f"| {index} | 状态{index} |" in markdown
    assert "| 合计 | 176 |" in markdown
    assert any(
        block["kind"] == "heading" and block["text"] == "个人信用报告" for block in semantic["structure"]["blocks"]
    )
    assert semantic["structure"]["source_tables"][0]["rows"][-1] == ["合计", "176"]


def test_dataset_bundle_has_one_wide_row_per_record_and_cell_audit() -> None:
    records = [
        {
            "repayment_id": f"rep_{index}",
            "account_id": "shared_account",
            "normalized": {"month": f"2025-{index:02d}", "status": "N"},
            "raw": {"month": f"2025-{index:02d}", "status": "N"},
        }
        for index in range(1, 13)
    ]
    result = _with_projection(
        ParseResult(entities=DocumentEntities(document_type="credit_report")),
        _candidate(records),
    )
    bundle = project_community_bundle(result, document_id="doc_test")
    dataset_csvs = bundle.render_dataset_csvs()
    rows = list(csv.DictReader(io.StringIO(dataset_csvs["001_datasets/repayment_records.csv"].lstrip("\ufeff"))))
    audit_rows = list(csv.DictReader(io.StringIO(bundle.render_audit_csv().lstrip("\ufeff"))))

    assert len(rows) == 12
    assert len({row["record_id"] for row in rows}) == 12
    assert list(rows[0]) == ["record_id", "_page_start", "_page_end", "month", "status"]
    assert rows[0]["month"] == "2025-01"
    assert len(audit_rows) == 24
    assert {row["dataset_id"] for row in audit_rows} == {"ds_repayment_records"}
    assert "subject_name" not in {row["field_key"] for row in audit_rows}


def test_audit_resolves_field_evidence_from_structural_table_refs_for_all_domains() -> None:
    candidate = _candidate(
        [
            {
                "normalized": {"month": "2025-01", "status": "N", "sequence": 1},
                "raw": {"month": "2025-01", "status": "N", "sequence": 1},
                "source": {
                    "source_refs": [
                        {
                            "source": "canonical_physical_table",
                            "page": 4,
                            "table_id": "pt_4_2",
                            "row": 0,
                        }
                    ],
                    "confidence": 1.0,
                },
            }
        ]
    )
    candidate["data"]["data_dictionary"]["datasets"]["repayment_records"]["columns"]["sequence"] = {
        "label": "序号",
        "type": "integer",
    }
    result = _with_projection(
        ParseResult(entities=DocumentEntities(document_type="credit_report")),
        candidate,
    )
    bundle = project_community_bundle(result, document_id="doc_test")
    semantic = bundle.semantic_payload()
    semantic["classification"]["document_type"] = "bank_statement"
    semantic["structure"]["source_tables"] = [
        {
            "id": "pt_4_2",
            "page": 4,
            "order": 1,
            "headers": ["月份", "状态"],
            "rows": [["2025-01", "N"]],
            "row_models": [
                {
                    "source_row_index": 0,
                    "cells": [
                        {
                            "text": "2025-01",
                            "col_index": 0,
                            "bbox": [10, 20, 40, 30],
                            "evidence_ids": ["ev:0004:text:000001"],
                        },
                        {
                            "text": "N",
                            "col_index": 1,
                            "bbox": [40, 20, 60, 30],
                            "evidence_ids": ["ev:0004:text:000002"],
                        },
                    ],
                }
            ],
        }
    ]
    audit_rows = list(csv.DictReader(io.StringIO(bundle.render_audit_csv(semantic).lstrip("\ufeff"))))

    month = next(row for row in audit_rows if row["field_key"] == "month")
    status = next(row for row in audit_rows if row["field_key"] == "status")
    sequence = next(row for row in audit_rows if row["field_key"] == "sequence")
    assert month["evidence_ref"] == '["ev:0004:text:000001"]'
    assert month["bbox"] == "[10.0,20.0,40.0,30.0]"
    assert month["confidence"] == "1.0"
    assert status["evidence_ref"] == '["ev:0004:text:000002"]'
    assert status["bbox"] == "[40.0,20.0,60.0,30.0]"
    assert json.loads(sequence["evidence_ref"]) == [
        "ev:0004:text:000001",
        "ev:0004:text:000002",
    ]
    assert sequence["bbox"] == "[10.0,20.0,60.0,30.0]"


def test_dataset_and_audit_csv_use_nested_source_page_range() -> None:
    records = [
        {
            "normalized": {"month": "2025-01", "status": "N"},
            "raw": {"month": "2025-01", "status": "N"},
            "source": {
                "source": "canonical_table",
                "page_id": "page:0007",
                "page_range": [7, 7],
            },
        }
    ]
    result = _with_projection(
        ParseResult(entities=DocumentEntities(document_type="credit_report")),
        _candidate(records),
    )
    bundle = project_community_bundle(result, document_id="doc_test")
    rows = list(
        csv.DictReader(io.StringIO(bundle.render_dataset_csvs()["001_datasets/repayment_records.csv"].lstrip("\ufeff")))
    )
    audit_rows = list(csv.DictReader(io.StringIO(bundle.render_audit_csv().lstrip("\ufeff"))))

    assert rows[0]["_page_start"] == "7"
    assert rows[0]["_page_end"] == "7"
    assert audit_rows
    assert {row["page_start"] for row in audit_rows} == {"7"}
    assert {row["page_end"] for row in audit_rows} == {"7"}


def test_csv_preserves_signed_numbers_but_neutralizes_text_formulas() -> None:
    candidate = _candidate(
        [
            {
                "repayment_id": "rep_1",
                "normalized": {"amount": "-10.25", "status": "=CMD()"},
                "raw": {"amount": "-10.25", "status": "=CMD()"},
            }
        ]
    )
    candidate["data"]["data_dictionary"]["datasets"]["repayment_records"]["columns"]["amount"] = {
        "label": "金额",
        "type": "money",
    }
    result = _with_projection(ParseResult(entities=DocumentEntities(document_type="credit_report")), candidate)
    bundle = project_community_bundle(result, document_id="doc_test")
    rows = list(
        csv.DictReader(io.StringIO(bundle.render_dataset_csvs()["001_datasets/repayment_records.csv"].lstrip("\ufeff")))
    )
    audit_rows = list(csv.DictReader(io.StringIO(bundle.render_audit_csv().lstrip("\ufeff"))))

    assert rows[0]["amount"] == "-10.25"
    assert rows[0]["status"] == "'=CMD()"
    amount = next(row for row in audit_rows if row["field_key"] == "amount")
    status = next(row for row in audit_rows if row["field_key"] == "status")
    assert amount["value"] == "-10.25"
    assert amount["raw"] == "-10.25"
    assert amount["csv_escape_applied"] == "false"
    assert status["value"] == "'=CMD()"
    assert status["raw"] == "'=CMD()"
    assert status["csv_escape_applied"] == "true"


def test_personal_detail_markdown_shows_sensitive_identifiers_and_phone_numbers() -> None:
    candidate = _candidate(
        [
            {
                "repayment_id": "rep_1",
                "month": "2025-01",
                "status": "N",
                "account_identifier": "123456789012345678",
            }
        ]
    )
    candidate["data"]["fields"]["id_number"] = "350600198703032041"
    candidate["data"]["fields"]["mobile_phone"] = "13812345678"
    candidate["data"]["data_dictionary"]["fields"]["id_number"] = {
        "label": "证件号码",
        "type": "long_id",
        "sensitive": True,
        "display": "masked",
    }
    candidate["data"]["data_dictionary"]["fields"]["mobile_phone"] = {
        "label": "手机号码",
        "type": "string",
        "sensitive": True,
        "display": "masked",
    }
    candidate["data"]["data_dictionary"]["datasets"]["repayment_records"]["columns"]["account_identifier"] = {
        "label": "账户标识",
        "type": "long_id",
        "sensitive": True,
        "display": "masked",
    }
    result = _with_projection(
        ParseResult(entities=DocumentEntities(document_type="credit_report")),
        candidate,
    )
    bundle = project_community_bundle(result, file_id="001", document_id="doc_sensitive")
    csv_content = next(iter(bundle.render_dataset_csvs().values()))
    rows = list(csv.DictReader(io.StringIO(csv_content.lstrip("\ufeff"))))
    enhanced = bundle.render_enhanced_markdown()

    assert rows[0]["account_identifier"] == "'123456789012345678"
    assert "350600198703032041" in enhanced
    assert "13812345678" in enhanced
    assert "123456789012345678" in enhanced
    assert "3506**********2041" not in enhanced
    assert "1234**********5678" not in enhanced


def test_enterprise_markdown_keeps_sensitive_identifiers_masked() -> None:
    candidate = _candidate(
        [
            {
                "repayment_id": "rep_1",
                "month": "2025-01",
                "status": "N",
                "account_identifier": "123456789012345678",
            }
        ]
    )
    candidate["data"]["data_dictionary"]["datasets"]["repayment_records"]["columns"][
        "account_identifier"
    ] = {
        "label": "账户标识",
        "type": "long_id",
        "sensitive": True,
        "display": "masked",
    }
    result = _with_projection(
        ParseResult(entities=DocumentEntities(document_type="enterprise_credit_report")),
        candidate,
    )

    enhanced = _project_community_bundle(
        seal_parse_result(result),
        file_id="001",
        document_id="doc_enterprise_sensitive",
        projection_data=_PROJECTIONS[id(result)],
        projection_policy={},
    ).render_enhanced_markdown()

    assert "1234**********5678" in enhanced
    assert "123456789012345678" not in enhanced


def test_audit_uses_canonical_field_keys_with_original_source_values() -> None:
    candidate = _candidate(
        [
            {
                "normalized": {"direction": "expense", "amount": "35.00"},
                "raw": {"收/支": "支出", "金额": "35.00"},
                "canonical_raw": {"direction": "支出", "amount": "35.00"},
            }
        ]
    )
    candidate["data"]["data_dictionary"]["datasets"]["repayment_records"]["columns"] = {
        "direction": {"label": "收/支", "type": "enum"},
        "amount": {"label": "金额", "type": "money"},
    }
    result = _with_projection(ParseResult(entities=DocumentEntities(document_type="credit_report")), candidate)
    bundle = project_community_bundle(result, document_id="doc_test")

    wide_rows = list(
        csv.DictReader(io.StringIO(bundle.render_dataset_csvs()["001_datasets/repayment_records.csv"].lstrip("\ufeff")))
    )
    audit_rows = list(csv.DictReader(io.StringIO(bundle.render_audit_csv().lstrip("\ufeff"))))

    assert list(wide_rows[0]) == ["record_id", "_page_start", "_page_end", "amount", "direction"]
    direction = next(row for row in audit_rows if row["field_key"] == "direction")
    assert direction["value"] == "expense"
    assert direction["raw"] == "支出"
    assert {row["field_key"] for row in audit_rows} == {"direction", "amount"}


def test_different_logical_datasets_are_written_to_different_wide_csvs() -> None:
    candidate = _candidate([{"month": "2025-01", "status": "N"}])
    candidate["data"]["inquiry_records"] = [{"query_date": "2025-02-01", "institution": "银行"}]
    candidate["data"]["data_dictionary"]["datasets"]["inquiry_records"] = {
        "columns": {
            "query_date": {"label": "查询日期", "type": "date"},
            "institution": {"label": "查询机构", "type": "string"},
        }
    }
    result = _with_projection(ParseResult(entities=DocumentEntities(document_type="credit_report")), candidate)
    bundle = project_community_bundle(result, document_id="doc_test")

    csvs = bundle.render_dataset_csvs()

    assert set(csvs) == {
        "001_datasets/repayment_records.csv",
        "001_datasets/inquiry_records.csv",
    }
    repayment_header = next(csv.reader(io.StringIO(csvs["001_datasets/repayment_records.csv"].lstrip("\ufeff"))))
    inquiry_header = next(csv.reader(io.StringIO(csvs["001_datasets/inquiry_records.csv"].lstrip("\ufeff"))))
    assert repayment_header == ["record_id", "_page_start", "_page_end", "month", "status"]
    assert inquiry_header == ["record_id", "_page_start", "_page_end", "institution", "query_date"]


def test_json_csv_and_enhanced_markdown_share_public_inquiry_occurrences() -> None:
    candidate = _candidate()
    candidate["data"]["inquiry_records"] = [
        {
            "inquiry_id": "credit_inquiry:25",
            "sequence": 25,
            "inquiry_date": "2024-09-10",
            "institution": "中国银行股份有限公司福建省分行",
            "reason": "贷后管理",
            "inquiry_type": "institution",
        },
        {
            "inquiry_id": "credit_inquiry:26",
            "sequence": 26,
            "inquiry_date": "2024-09-10",
            "institution": "中国银行股份有限公司福建省分行",
            "reason": "贷后管理",
            "inquiry_type": "institution",
        },
    ]
    result = _with_projection(
        ParseResult(entities=DocumentEntities(document_type="credit_report")),
        candidate,
    )
    bundle = project_community_bundle(result, file_id="001", document_id="doc_test")

    payload = bundle.json_payload()
    inquiry_dataset = next(dataset for dataset in payload["datasets"] if dataset["name"] == "inquiry_records")
    csv_rows = list(csv.DictReader(io.StringIO(bundle.render_dataset_csvs()[inquiry_dataset["csv"]].lstrip("\ufeff"))))
    enhanced = bundle.render_enhanced_markdown()

    json_ids = [row["record_id"] for row in inquiry_dataset["rows"]]
    assert len(json_ids) == len(set(json_ids)) == 2
    assert [row["record_id"] for row in csv_rows] == json_ids
    assert "#### institution" in enhanced
    assert "| 25 | 2024-09-10 | 中国银行股份有限公司福建省分行 | 贷后管理 | institution |" in enhanced
    assert "| 26 | 2024-09-10 | 中国银行股份有限公司福建省分行 | 贷后管理 | institution |" in enhanced


def test_record_card_title_separator_is_plugin_configurable() -> None:
    candidate = _candidate(
        [
            {
                "repayment_id": "rep_1",
                "sequence": 1,
                "business_category": "中长期贷款",
            }
        ]
    )
    candidate["data"]["data_dictionary"]["datasets"]["repayment_records"]["columns"].update(
        {
            "sequence": {"label": "序号", "type": "integer"},
            "business_category": {"label": "业务类别", "type": "string"},
        }
    )
    result = _with_projection(
        ParseResult(entities=DocumentEntities(document_type="credit_report")),
        candidate,
    )
    _PROJECTIONS[id(result)]["semantic"] = {
        "enhanced_markdown": {
            "dataset_layouts": {
                "repayment_records": {
                    "mode": "record_cards",
                    "title_fields": ["sequence", "business_category"],
                    "title_separator": ". ",
                    "columns": ["sequence", "business_category"],
                }
            }
        }
    }

    enhanced = project_community_bundle(result, document_id="doc_title_separator").render_enhanced_markdown()

    assert "#### 1. 中长期贷款" in enhanced
    assert "#### 1 · 中长期贷款" not in enhanced


def test_partitioned_tables_use_declared_order_and_type_specific_columns() -> None:
    candidate = _candidate()
    candidate["data"]["credit_accounts"] = [
        {
            "account_id": "card:1",
            "sequence": 1,
            "account_type": "credit_card",
            "institution": "发卡银行",
            "credit_limit": 2240000,
        },
        {
            "account_id": "loan:1",
            "sequence": 1,
            "account_type": "loan",
            "institution": "贷款银行",
            "loan_amount": 315000,
        },
        {
            "account_id": "other:1",
            "sequence": 1,
            "account_type": "other",
            "institution": "其他机构",
        },
    ]
    candidate["data"]["data_dictionary"]["datasets"]["credit_accounts"] = {
        "columns": {
            "sequence": {"label": "组内序号", "type": "integer"},
            "account_type": {"label": "账户类型", "type": "enum"},
            "institution": {"label": "管理机构", "type": "string"},
            "credit_limit": {"label": "信用额度", "type": "money"},
            "loan_amount": {"label": "贷款发放金额", "type": "money"},
        }
    }
    candidate["data"]["data_dictionary"]["enums"] = {
        "account_type": {
            "credit_card": "信用卡",
            "loan": "贷款",
            "other": "其他账户",
        }
    }
    result = _with_projection(
        ParseResult(entities=DocumentEntities(document_type="credit_report")),
        candidate,
    )
    _PROJECTIONS[id(result)]["semantic"] = {
        "enhanced_markdown": {
            "dataset_layouts": {
                "credit_accounts": {
                    "mode": "partitioned_tables",
                    "partition_by": "account_type",
                    "partitions": [
                        {
                            "value": "credit_card",
                            "title": "信用卡账户",
                            "columns": ["sequence", "institution", "credit_limit"],
                        },
                        {
                            "value": "loan",
                            "title": "贷款账户",
                            "columns": ["sequence", "institution", "loan_amount"],
                        },
                    ],
                }
            }
        }
    }

    enhanced = project_community_bundle(result, document_id="doc_partitioned").render_enhanced_markdown()
    card_table = enhanced.split("#### 信用卡账户", maxsplit=1)[1].split("\n#### ", maxsplit=1)[0]
    loan_table = enhanced.split("#### 贷款账户", maxsplit=1)[1].split("\n#### ", maxsplit=1)[0]

    assert enhanced.index("#### 信用卡账户") < enhanced.index("#### 贷款账户")
    assert "| 组内序号 | 管理机构 | 信用额度 |" in card_table
    assert "贷款发放金额" not in card_table
    assert "| 1 | 发卡银行 | 2240000 |" in card_table
    assert "| 组内序号 | 管理机构 | 贷款发放金额 |" in loan_table
    assert "信用额度" not in loan_table
    assert "| 1 | 贷款银行 | 315000 |" in loan_table
    assert "#### 其他账户" in enhanced
    assert "其他机构" in enhanced
    assert "2,240,000" not in enhanced


def test_audit_reconciliation_can_render_in_appendix_and_audit_csv_only() -> None:
    result = _with_projection(
        ParseResult(entities=DocumentEntities(document_type="credit_report")),
        _candidate([{"repayment_id": "rep_1", "month": "2025-01", "status": "N"}]),
    )
    projection = _PROJECTIONS[id(result)]
    projection["domain_facts"]["credit_extraction_audit"] = {
        "reconciliations": [
            {
                "name": "credit_account_balance",
                "expected": 65.41,
                "actual": 65.42,
                "difference": 0.01,
                "tolerance": 0.02,
                "matched": True,
                "status": "within_rounding_tolerance",
            }
        ]
    }
    projection["semantic"] = {
        "audit_csv": {
            "reconciliations": [
                {
                    "name": "credit_account_balance",
                    "fields": ["expected", "actual", "difference", "tolerance", "status"],
                }
            ]
        },
        "enhanced_markdown": {
            "appendix": {
                "title": "附录：审计信息",
                "audit_reconciliations": [
                    {
                        "name": "credit_account_balance",
                        "title": "源文余额矛盾",
                        "fields": [
                            {"key": "expected", "label": "源报告值"},
                            {"key": "actual", "label": "明细计算值"},
                            {"key": "difference", "label": "差额"},
                        ],
                        "note": "仅作审计提示，不改写业务数据。",
                    }
                ],
            }
        },
    }
    bundle = project_community_bundle(result, document_id="doc_audit_reconciliation")
    semantic = bundle.semantic_payload()
    enhanced = bundle.render_enhanced_markdown(semantic)
    audit_rows = list(csv.DictReader(io.StringIO(bundle.render_audit_csv(semantic).lstrip("\ufeff"))))

    assert "### 源文余额矛盾" in enhanced
    assert "**源报告值:** 65.41" in enhanced
    assert "**明细计算值:** 65.42" in enhanced
    assert "不改写业务数据" in enhanced
    reconciliation_rows = [
        row for row in audit_rows if row["dataset_id"] == "_audit_reconciliations"
    ]
    assert {row["field_key"]: row["value"] for row in reconciliation_rows} == {
        "expected": "65.41",
        "actual": "65.42",
        "difference": "0.01",
        "tolerance": "0.02",
        "status": "within_rounding_tolerance",
    }
    assert len(bundle.json_payload(semantic)["datasets"][0]["rows"]) == 1


def test_dataset_can_be_deferred_to_final_markdown_appendix() -> None:
    result = _with_projection(
        ParseResult(entities=DocumentEntities(document_type="credit_report")),
        _candidate(
            [
                {
                    "repayment_id": "rep_1",
                    "month": "2025-01",
                    "status": "N",
                }
            ]
        ),
    )
    _PROJECTIONS[id(result)]["semantic"] = {
        "enhanced_markdown": {
            "dataset_layouts": {
                "repayment_records": {
                    "placement": "appendix",
                    "columns": ["month", "status"],
                }
            },
            "appendix": {"title": "附录：审计信息"},
        }
    }

    enhanced = project_community_bundle(
        result,
        document_id="doc_deferred_dataset",
    ).render_enhanced_markdown()
    body, appendix = enhanced.split("## 附录：审计信息", maxsplit=1)

    assert "2025-01" not in body
    assert "### 还款记录" in appendix
    assert "| 月份 | 还款状态 |" in appendix
    assert "| 2025-01 | N |" in appendix
    assert enhanced.rstrip().endswith("| 2025-01 | N |")


def test_payment_records_use_transaction_business_name() -> None:
    candidate = _candidate([{"normalized": {"amount": "10.00"}, "raw": {"amount": "10.00"}}])
    candidate["data"]["records"] = candidate["data"].pop("repayment_records")
    candidate["data"]["data_dictionary"]["record_columns"] = {"amount": {"label": "金额", "type": "money"}}
    result = _with_projection(ParseResult(entities=DocumentEntities(document_type="alipay_payment")), candidate)
    bundle = project_community_bundle(result, document_id="doc_test")

    assert bundle.datasets[0].public["id"] == "ds_transactions"
    assert bundle.datasets[0].public["name"] == "transactions"
    assert bundle.datasets[0].public["type"] == "transaction"
    assert bundle.datasets[0].public["csv"] == "001_datasets/transactions.csv"


def test_bank_statement_enhanced_markdown_uses_chinese_labels_without_changing_keys() -> None:
    candidate = _candidate(
        [
            {
                "normalized": {
                    "amount": "10.00",
                    "amount_cny": "10.00",
                    "balance": "90.00",
                    "channel": "网银",
                    "counter_account": "001234",
                    "counter_bank_code": "102",
                    "counter_bank_name": "示例银行",
                    "counter_party": "示例公司",
                    "counterparty_status": "present",
                    "date": "2025-07-01",
                    "direction": "expense",
                    "purpose": "货款",
                    "sequence_no": "0001",
                    "summary": "转账",
                    "timestamp": "2025-07-01 09:00:00",
                },
                "raw": {},
            }
        ]
    )
    candidate["data"]["records"] = candidate["data"].pop("repayment_records")
    candidate["data"]["data_dictionary"] = BANK_DATA_DICTIONARY
    candidate["data"]["fields"].update(
        {
            "document_scene_refined": "bank_statement",
            "layout_profile_id_refined": "borderless_ledger_bank",
            "layout_profile_refine_confidence": 0.9997,
        }
    )
    result = _with_projection(
        ParseResult(entities=DocumentEntities(document_type="bank_statement")),
        candidate,
    )

    bundle = project_community_bundle(result, document_id="doc_bank_chinese_reading")
    enhanced = bundle.render_enhanced_markdown()
    transaction_header = next(
        line for line in enhanced.splitlines() if line.startswith("| ") and "交易金额" in line
    )

    assert "**文档类型:** 银行流水" in enhanced
    assert "**页数:** 0" in enhanced
    assert "**修正文档场景:** 银行流水" in enhanced
    assert "**修正版式配置:** 无框银行流水版式" in enhanced
    assert "**版式修正置信度:** 0.9997" in enhanced
    assert "document scene refined" not in enhanced
    assert "layout profile id refined" not in enhanced
    assert "交易金额" in transaction_header
    assert "折合人民币金额" in transaction_header
    assert "账户余额" in transaction_header
    assert "对方账号" in transaction_header
    assert "对方银行名称" in transaction_header
    assert "收支方向" in transaction_header
    assert "交易时间" in transaction_header
    assert " amount " not in transaction_header
    assert " counter account " not in transaction_header
    assert bundle.datasets[0].public["columns"][0]["key"] == "amount"


def test_markdown_escapes_table_delimiters_but_preserves_content() -> None:
    result = ParseResult(
        pages=[
            PageContent(
                page_number=1,
                tables=[
                    TableBlock(
                        headers=["内容"],
                        rows=[TableRow(cells=[CellValue(text="A|B\nC")])],
                    )
                ],
            )
        ]
    )
    _with_projection(result, _candidate())
    markdown = project_community_bundle(result, document_id="doc_test").render_markdown()
    assert "A\\|B C" in markdown
    assert "<br>" not in markdown


def test_markdown_renders_payment_record_promoted_to_header_as_data() -> None:
    result = ParseResult(
        pages=[
            PageContent(
                page_number=1,
                tables=[
                    TableBlock(
                        headers=["不计\n收支", "第一条交易"],
                        rows=[TableRow(cells=[CellValue(text="收入"), CellValue(text="第二条交易")])],
                    )
                ],
            )
        ]
    )
    markdown = project_community_bundle(result, document_id="doc_payment_header").render_markdown()

    assert "| 不计收支 | 第一条交易 |" in markdown
    assert "| 收入 | 第二条交易 |" in markdown
    assert "<table>" not in markdown


def test_markdown_image_omission_adds_idempotent_info_warning() -> None:
    result = ParseResult(
        pages=[
            PageContent(
                page_number=1,
                texts=[TextBlock(content='<div><img src="imgs/missing.jpg" alt="Image" /></div>')],
            )
        ]
    )
    _with_projection(result, _candidate())
    bundle = project_community_bundle(result, document_id="doc_test")

    first = bundle.render_markdown()
    second = bundle.render_markdown()
    warnings = [warning for warning in bundle.json_payload()["warnings"] if warning["code"] == "MARKDOWN_IMAGE_OMITTED"]

    assert first == second
    assert "<img" not in first
    assert warnings == [
        {
            "code": "MARKDOWN_IMAGE_OMITTED",
            "level": "info",
            "message": "Unmaterialized source images were omitted from content Markdown.",
        }
    ]


def test_large_dataset_is_not_truncated_and_json_csv_ids_match() -> None:
    records = [
        {
            "record_id": f"repayment:{index:06d}",
            "normalized": {"month": f"2025-{((index - 1) % 12) + 1:02d}", "status": "N"},
            "raw": {"month": f"2025-{((index - 1) % 12) + 1:02d}", "status": "N"},
        }
        for index in range(1, 5001)
    ]
    result = _with_projection(
        ParseResult(entities=DocumentEntities(document_type="credit_report")),
        _candidate(records),
    )
    bundle = project_community_bundle(result, document_id="doc_large")

    payload = bundle.json_payload()
    dataset = payload["datasets"][0]
    csvs = bundle.render_dataset_csvs()
    csv_rows = list(csv.DictReader(io.StringIO(csvs["001_datasets/repayment_records.csv"].lstrip("\ufeff"))))

    assert dataset["row_count"] == 5000
    assert len(dataset["rows"]) == 5000
    assert dataset["rows"][-1]["record_id"] == "repayment:005000"
    assert [row["record_id"] for row in dataset["rows"]] == [row["record_id"] for row in csv_rows]
    assert bundle.conservation_issues(payload=payload, dataset_csvs=csvs) == []


def test_conservation_gate_rejects_json_or_csv_row_loss() -> None:
    records = [{"month": f"2025-{index:02d}", "status": "N"} for index in range(1, 4)]
    result = _with_projection(
        ParseResult(entities=DocumentEntities(document_type="credit_report")),
        _candidate(records),
    )
    bundle = project_community_bundle(result, document_id="doc_gate")
    payload = bundle.json_payload()
    csvs = bundle.render_dataset_csvs()

    payload["datasets"][0]["rows"].pop()
    assert any(":row_count=" in issue for issue in bundle.conservation_issues(payload=payload))

    intact_payload = bundle.json_payload()
    csv_lines = csvs["001_datasets/repayment_records.csv"].splitlines(keepends=True)
    truncated_csvs = dict(csvs)
    truncated_csvs["001_datasets/repayment_records.csv"] = "".join(csv_lines[:-1])
    assert any(
        ":csv=" in issue for issue in bundle.conservation_issues(payload=intact_payload, dataset_csvs=truncated_csvs)
    )


def test_nested_summary_map_keys_use_parent_enum_lexicon() -> None:
    candidate = _candidate()
    candidate["data"]["fields"]["credit_summary"] = {
        "public_record_type_counts": {
            "license": 2,
            "certification": 1,
            "patent": 3,
        }
    }
    candidate["data"]["data_dictionary"]["fields"]["public_record_type_counts"] = {
        "label": "公共记录明细类型统计",
        "type": "object",
        "map_key_enum": "record_type",
    }
    candidate["data"]["data_dictionary"]["enums"] = {
        "record_type": {
            "license": "许可记录",
            "certification": "认证记录",
            "patent": "专利记录",
        }
    }
    result = _with_projection(
        ParseResult(entities=DocumentEntities(document_type="credit_report")),
        candidate,
    )

    semantic = project_community_bundle(
        result,
        document_id="doc_nested_enum",
    ).semantic_payload()
    group = next(
        group
        for section in semantic["structure"]["sections"]
        for group in section.get("groups", [])
        if group["key"] == "public_record_type_counts"
    )

    assert {item["key"]: item["label"] for item in group["items"]} == {
        "license": "许可记录",
        "certification": "认证记录",
        "patent": "专利记录",
    }
