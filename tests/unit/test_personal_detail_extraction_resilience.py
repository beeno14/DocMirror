from __future__ import annotations

from types import SimpleNamespace

from docmirror.plugins.credit_report.personal_detail_scanned.context import _printed_reading_order
from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
    collect_extraction_issues,
    dataset_states_from_issues,
    liability_record_is_substantive,
    make_issue,
)
from docmirror.plugins.credit_report.personal_detail_scanned.field_contracts import validate_pboc_field
from docmirror.plugins.credit_report.personal_detail_scanned.native_parser import (
    PBOCPersonalDetailNativeParser,
)
from docmirror.plugins.credit_report.personal_detail_scanned.schema_v2 import (
    project_personal_detail_v2_datasets,
)


def _table(table_id: str, rows: list[list[str]]) -> SimpleNamespace:
    return SimpleNamespace(
        table_id=table_id,
        metadata={"raw_rows": rows},
        headers=[],
        rows=[],
        bbox=[20, 20, 580, 300],
    )


def test_uncertain_cell_is_preserved_and_published_for_review() -> None:
    context = SimpleNamespace(
        ocr_correction_audit=lambda: {
            "cell_anomalies": [
                {
                    "stage": "native_business",
                    "path": "credit_accounts[0].balance",
                    "role": "amount",
                    "value": "12O0",
                    "reason_codes": ["role_validation_failed", "preserved_unresolved_value"],
                    "source_refs": [{"logical_page": 3, "source_page": 2}],
                }
            ]
        },
        page_topology_audit=lambda: {"issues": []},
    )

    issues = collect_extraction_issues(context)

    assert len(issues) == 1
    assert issues[0]["observed_value"] == "12O0"
    assert issues[0]["status"] == "requires_review"
    assert dataset_states_from_issues(issues) == {}


def test_pboc_controlled_vocabulary_is_diagnostic_and_broad() -> None:
    assert validate_pboc_field("贷后管理", "inquiry_reason").valid is True
    assert validate_pboc_field("司法调查", "inquiry_reason").valid is True
    invalid = validate_pboc_field("贷后管埋", "inquiry_reason")
    assert invalid.assessed is True
    assert invalid.valid is False
    assert invalid.reason_code == "controlled_vocabulary_mismatch"


def test_only_row_blocking_structure_failures_degrade_dataset_presence() -> None:
    issue = make_issue(
        category="ocr_structure_correction",
        issue_code="recognized_native_section_missing_required_value",
        message="required value missing",
        target_dataset="credit_lines",
    )

    assert dataset_states_from_issues([issue])["credit_lines"]["presence_status"] == "extraction_failed"


def test_tolerant_native_parser_accepts_unique_high_margin_label_damage() -> None:
    table = _table(
        "credit-line",
        [
            ["授信协议标", "授信额度用途", "管理机构"],
            ["AGREEMENT0001", "循环额度", "示例银行股份有限公司"],
        ],
    )
    page = SimpleNamespace(page_number=1, source_page_number=1, tables=[table])
    context = SimpleNamespace(
        pages=[page],
        reading_order_by_logical={1: 1},
        tables_continue=lambda _left, _right: None,
    )

    records = PBOCPersonalDetailNativeParser(context).records("credit_lines")

    assert len(records) == 1
    assert records[0].fields["授信协议标识"] == "AGREEMENT0001"


def test_native_parser_uses_whole_page_ocr_when_cell_structure_is_incomplete() -> None:
    table = _table(
        "credit-line-incomplete",
        [
            ["授信协议标识", "授信额度用途"],
            ["", "循环额度"],
        ],
    )
    page = SimpleNamespace(page_number=1, source_page_number=1, tables=[table])
    recorded: list[dict[str, object]] = []

    def full_page_ocr(_pages: set[int], *, reason: str) -> list[dict[str, object]]:
        assert "missing_required_value" in reason
        return [
            {
                "page": 1,
                "source_page": 1,
                "lines": [
                    {"text": "授信协议标识", "bbox": [20, 100, 180, 125]},
                    {"text": "授信额度用途", "bbox": [300, 100, 450, 125]},
                    {"text": "AGREEMENT0002", "bbox": [20, 145, 180, 170]},
                    {"text": "循环额度", "bbox": [300, 145, 450, 170]},
                ],
            }
        ]

    context = SimpleNamespace(
        pages=[page],
        reading_order_by_logical={1: 1},
        tables_continue=lambda _left, _right: None,
        full_page_ocr_evidence=full_page_ocr,
        _personal_detail_extraction_issues=recorded,
    )

    records = PBOCPersonalDetailNativeParser(context).records("credit_lines")

    assert len(records) == 1
    assert records[0].fields["授信协议标识"] == "AGREEMENT0002"
    assert context._personal_detail_extraction_issues[0]["status"] == "resolved"


def test_printed_reading_order_tolerates_large_observed_page_gaps() -> None:
    pages = [
        SimpleNamespace(page_number=30, source_page_number=3, texts=[SimpleNamespace(content="第 8 页，共 8 页")]),
        SimpleNamespace(page_number=10, source_page_number=1, texts=[SimpleNamespace(content="第 1 页，共 8 页")]),
        SimpleNamespace(page_number=20, source_page_number=2, texts=[SimpleNamespace(content="第 4 页，共 8 页")]),
    ]
    result = SimpleNamespace(pages=pages, entities=SimpleNamespace(domain_specific={}))

    assert _printed_reading_order(result) == {10: 1, 20: 2, 30: 3}


def test_identifier_only_liability_rows_are_redundant_not_business_records() -> None:
    assert liability_record_is_substantive({"liability_id": "row:1", "source_refs": []}) is False
    assert liability_record_is_substantive(
        {"liability_id": "row:2", "responsibility_type": "保证人"}
    ) is True


def test_v2_projection_keeps_extraction_issues_in_extension_fields() -> None:
    issue = make_issue(
        category="ocr_cell_level_error",
        issue_code="pboc_cell_contract_unresolved",
        message="review",
        target_dataset="credit_accounts",
        field_name="balance",
        observed_value="12O0",
    )

    projected = project_personal_detail_v2_datasets(
        {"personal_detail_extraction_issues": [issue]}
    )
    extension_row = projected["pboc_extension_fields"][0]
    extension = extension_row.get("normalized", extension_row)

    assert extension["source_dataset"] == "personal_detail_extraction_issues"
    assert extension["field_name"] == "pboc_cell_contract_unresolved"
    assert "12O0" in extension["value"]
