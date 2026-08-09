from __future__ import annotations

from types import SimpleNamespace

from docmirror.plugins.credit_report.personal_detail_scanned.native_parser import (
    PBOCPersonalDetailNativeParser,
)

PACKED_HEADER = (
    "管理机构 业务种类 成立日期 到期日期 贵任人类型 "
    "还款贵任金额 币种 保证合同编号"
)
PACKED_VALUE = (
    "梅赛德斯-奔驰汽车金融有限公司 贷款 2021.08.03 2024.08.03 "
    "保证 629,860 人民币元 Y10061000H0001EIP1967714G01"
)


def _native_context(value: str) -> SimpleNamespace:
    table = SimpleNamespace(
        table_id="liability-packed-1",
        metadata={
            "raw_rows": [[PACKED_HEADER], [value]],
            "source_cell_bboxes": [
                [[20.0, 40.0, 580.0, 58.0]],
                [[20.0, 70.0, 580.0, 88.0]],
            ],
            "cell_evidence_ids": [[['header-evidence']], [['value-evidence']]],
        },
        headers=[],
        rows=[],
        bbox=[20.0, 40.0, 580.0, 88.0],
    )
    page = SimpleNamespace(
        page_number=6,
        source_page_number=3,
        canonical_template_id="repayment_liability",
        texts=[],
        tables=[table],
    )
    return SimpleNamespace(
        pages=[page],
        reading_order_by_logical={6: 6},
        tables_continue=lambda _left, _right: None,
        corrected_evidence_pages=lambda: [],
        _personal_detail_extraction_issues=[],
    )


def _line(text: str, y: float, *, evidence_id: str) -> dict[str, object]:
    return {
        "text": text,
        "bbox": [20.0, y, 580.0, y + 18.0],
        "source_bbox": [25.0, y + 5.0, 585.0, y + 23.0],
        "confidence": 0.97,
        "evidence_ids": [evidence_id],
    }


def test_native_parser_decodes_packed_liability_with_exact_cell_provenance() -> None:
    context = _native_context(PACKED_VALUE)

    records = PBOCPersonalDetailNativeParser(context).records("repayment_liability_records")

    assert len(records) == 1
    record = records[0]
    assert record.fields == {
        "管理机构": "梅赛德斯-奔驰汽车金融有限公司",
        "业务种类": "贷款",
        "开立日期": "2021-08-03",
        "到期日期": "2024-08-03",
        "责任人类型": "保证",
        "还款责任金额": "629860",
        "币种": "CNY",
        "保证合同编号": "Y10061000H0001EIP1967714G01",
    }
    assert record.observed_labels == frozenset(record.fields)
    assert record.unresolved_labels == frozenset()
    institution_ref = record.source_refs_by_field["管理机构"][0]
    assert institution_ref == {
        "source": "native_detail_tolerant_table_cell",
        "logical_page": 6,
        "source_page": 3,
        "table_id": "liability-packed-1",
        "bbox": [20.0, 70.0, 580.0, 88.0],
        "geometry_scope": "cell",
        "row": 1,
        "column": 0,
        "field_name": "管理机构",
        "binding": "canonical_packed_liability_row",
        "evidence_ids": ["value-evidence"],
    }
    assert record.binding_quality_by_field["管理机构"] == "canonical_packed_liability_row"
    assert context._personal_detail_extraction_issues == []


def test_native_parser_retains_safe_partial_fields_and_preserves_issue_witness() -> None:
    raw_value = PACKED_VALUE.replace("人民币元", "多 民币元")
    context = _native_context(raw_value)

    records = PBOCPersonalDetailNativeParser(context).records("repayment_liability_records")

    assert len(records) == 1
    record = records[0]
    assert record.fields["管理机构"] == "梅赛德斯-奔驰汽车金融有限公司"
    assert record.fields["还款责任金额"] == "629860"
    assert record.fields["保证合同编号"] == "Y10061000H0001EIP1967714G01"
    assert "币种" not in record.fields
    assert "币种" in record.unresolved_labels
    assert len(context._personal_detail_extraction_issues) == 1
    issue = context._personal_detail_extraction_issues[0]
    assert issue["issue_code"] == "candidate_b_packed_liability_row_unresolved"
    assert issue["observed_value"] == {
        "header": [PACKED_HEADER],
        "value": [raw_value],
        "unresolved_reason": "typed_spans_with_ocr_residue",
        "retained_typed_fields": {
            "管理机构": "梅赛德斯-奔驰汽车金融有限公司",
            "业务种类": "贷款",
            "开立日期": "2021-08-03",
            "到期日期": "2024-08-03",
            "责任人类型": "保证",
            "还款责任金额": "629860",
            "保证合同编号": "Y10061000H0001EIP1967714G01",
        },
        "affected_fields": ["currency"],
    }
    assert {(ref["row"], ref["column"]) for ref in issue["source_refs"]} == {(0, 0), (1, 0)}
    assert "raw_witness_preserved" in issue["reason_codes"]
    assert "unique_typed_fields_retained" in issue["reason_codes"]


def test_unresolved_packed_issue_retains_one_printed_liability_ordinal() -> None:
    raw_value = PACKED_VALUE.replace("人民币元", "多 民币元")
    context = _native_context(raw_value)
    table = context.pages[0].tables[0]
    table.metadata["raw_rows"].insert(0, ["账户 7"])
    table.metadata["source_cell_bboxes"].insert(0, [[20.0, 15.0, 580.0, 33.0]])
    table.metadata["cell_evidence_ids"].insert(0, [["ordinal-evidence"]])

    PBOCPersonalDetailNativeParser(context).records("repayment_liability_records")

    issue = context._personal_detail_extraction_issues[0]
    assert issue["observed_value"]["printed_sequence"] == "7"
    assert issue["observed_value"]["affected_fields"] == ["currency"]


def test_corrected_page_parser_starts_record_at_unanchored_complete_packed_header() -> None:
    context = SimpleNamespace(
        pages=[],
        _personal_detail_extraction_issues=[],
        corrected_evidence_pages=lambda: [
            {
                "page": 7,
                "source_page": 4,
                "lines": [
                    _line("(六)相关还款责任信息", 10.0, evidence_id="section"),
                    _line(PACKED_HEADER, 40.0, evidence_id="header"),
                    _line(PACKED_VALUE, 70.0, evidence_id="value"),
                    _line("(七)授信协议信息", 110.0, evidence_id="next-section"),
                ],
            }
        ],
    )

    records = PBOCPersonalDetailNativeParser(context).records("repayment_liability_records")

    assert len(records) == 1
    record = records[0]
    assert record.fields["管理机构"] == "梅赛德斯-奔驰汽车金融有限公司"
    assert record.fields["保证合同编号"] == "Y10061000H0001EIP1967714G01"
    ref = record.source_refs_by_field["保证合同编号"][0]
    assert ref["source"] == "personal_detail_corrected_page_cell"
    assert ref["bbox"] == [25.0, 75.0, 585.0, 93.0]
    assert ref["evidence_ids"] == ["value"]
    assert ref["field_name"] == "保证合同编号"
    assert ref["binding"] == "canonical_packed_liability_row"
    assert context._personal_detail_extraction_issues == []


def test_corrected_page_partial_packed_row_keeps_fields_and_cell_witness() -> None:
    raw_value = PACKED_VALUE.replace("人民币元", "多 民币元")
    context = SimpleNamespace(
        pages=[],
        _personal_detail_extraction_issues=[],
        corrected_evidence_pages=lambda: [
            {
                "page": 7,
                "source_page": 4,
                "lines": [
                    _line("(六)相关还款责任信息", 10.0, evidence_id="section"),
                    _line(PACKED_HEADER, 40.0, evidence_id="header"),
                    _line(raw_value, 70.0, evidence_id="value"),
                    _line("(七)授信协议信息", 110.0, evidence_id="next-section"),
                ],
            }
        ],
    )

    records = PBOCPersonalDetailNativeParser(context).records("repayment_liability_records")

    assert len(records) == 1
    assert records[0].fields["还款责任金额"] == "629860"
    assert records[0].fields["保证合同编号"] == "Y10061000H0001EIP1967714G01"
    assert "币种" not in records[0].fields
    assert len(context._personal_detail_extraction_issues) == 1
    issue = context._personal_detail_extraction_issues[0]
    assert issue["issue_code"] == "candidate_b_packed_liability_row_unresolved"
    assert issue["observed_value"]["value"] == [raw_value]
    assert {ref["evidence_ids"][0] for ref in issue["source_refs"]} == {"header", "value"}
    assert all(ref["source"] == "personal_detail_corrected_page_cell" for ref in issue["source_refs"])


def test_unresolved_packed_gate_does_not_erase_exact_same_column_fields() -> None:
    header = ["管理机构", "业务种类", "成立日期", "到期日期", "责任人类型", "还款责任金额", "币种", "保证合同编号"]
    values = [
        "示例银行股份有限公司",
        "贷款",
        "2021.01.01",
        "2022.01.01",
        "保证人",
        "1,000",
        "多 民币元",
        "HT0000001",
    ]
    table = SimpleNamespace(
        table_id="liability-columns-1",
        metadata={"raw_rows": [header, values]},
        headers=[],
        rows=[],
        bbox=[20.0, 40.0, 580.0, 88.0],
    )
    page = SimpleNamespace(
        page_number=8,
        source_page_number=4,
        canonical_template_id="repayment_liability",
        texts=[],
        tables=[table],
    )
    context = SimpleNamespace(
        pages=[page],
        reading_order_by_logical={8: 8},
        tables_continue=lambda _left, _right: None,
        corrected_evidence_pages=lambda: [],
        _personal_detail_extraction_issues=[],
    )

    records = PBOCPersonalDetailNativeParser(context).records("repayment_liability_records")

    assert len(records) == 1
    assert records[0].fields["管理机构"] == "示例银行股份有限公司"
    assert records[0].fields["还款责任金额"] == "1,000"
    assert records[0].fields["币种"] == "多 民币元"
    assert records[0].binding_quality_by_field["管理机构"] == "native_label_column"
    assert len(context._personal_detail_extraction_issues) == 1
    assert (
        context._personal_detail_extraction_issues[0]["issue_code"]
        == "candidate_b_packed_liability_row_unresolved"
    )
