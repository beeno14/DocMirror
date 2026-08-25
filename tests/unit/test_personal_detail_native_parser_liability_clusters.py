from __future__ import annotations

from types import SimpleNamespace

from docmirror.plugins.credit_report.personal_detail_scanned import native_extraction
from docmirror.plugins.credit_report.personal_detail_scanned.native_parser import (
    PBOCPersonalDetailNativeParser,
    _packed_value_equivalent,
)

PACKED_HEADER = (
    "管理机构 业务种类 成立日期 到期日期 贵任人类型 "
    "还款贵任金额 币种 保证合同编号"
)
PACKED_VALUE = (
    "梅赛德斯-奔驰汽车金融有限公司 贷款 2021.08.03 2024.08.03 "
    "保证 629,860 人民币元 Y10061000H0001EIP1967714G01"
)


def test_packed_liability_equivalence_rejects_deleted_residue() -> None:
    assert not _packed_value_equivalent("开立日期", "2024?01?02", "2024-01-02")
    assert not _packed_value_equivalent("开立日期", "任意文本", "其他文本")
    assert not _packed_value_equivalent("还款责任金额", "1,2", "12")
    assert not _packed_value_equivalent("还款责任金额", "$1,200", "1200")


def test_packed_liability_equivalence_accepts_registered_presentations() -> None:
    assert _packed_value_equivalent("开立日期", "2024.01.02", "2024-01-02")
    assert _packed_value_equivalent("还款责任金额", "1,200", "1200")


def _owned_liability_table(table: SimpleNamespace) -> SimpleNamespace:
    table.metadata["canonical_template_id"] = "repayment_responsibility"
    return table


def _native_context(value: str) -> SimpleNamespace:
    table = _owned_liability_table(SimpleNamespace(
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
    ))
    page = SimpleNamespace(
        page_number=6,
        source_page_number=3,
        canonical_template_id="repayment_responsibility",
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


def _adjacent_liability_anchor_context(
    *,
    anchor_bbox: list[float] | None = None,
) -> SimpleNamespace:
    headers = [
        "管理机构",
        "业务种类",
        "开立日期",
        "到期日期",
        "责任人类型",
        "还款责任金额",
        "币种",
        "保证合同编号",
    ]
    values = [
        "中国农业银行股份有限公司大理分行",
        "贷款",
        "2023.09.18",
        "2024.09.13",
        "共同借款人",
        "",
        "人民币元",
        "",
    ]
    column_width = 44.0
    source_cell_bboxes = [
        [
            [
                47.0 + column * column_width,
                339.5 if row == 0 else 352.5,
                47.0 + (column + 1) * column_width,
                352.5 if row == 0 else 378.0,
            ]
            for column in range(8)
        ]
        for row in range(2)
    ]
    table = _owned_liability_table(SimpleNamespace(
        table_id="pt_19_2",
        metadata={
            "raw_rows": [headers, values],
            "source_cell_bboxes": source_cell_bboxes,
            "cell_evidence_ids": [
                [
                    [f"pt19:{row}:{column}"]
                    if [headers, values][row][column]
                    else []
                    for column in range(8)
                ]
                for row in range(2)
            ],
        },
        headers=[],
        rows=[],
        bbox=[47.0, 339.5, 399.0, 444.0],
    ))
    page = SimpleNamespace(
        page_number=19,
        source_page_number=10,
        canonical_template_id="repayment_responsibility",
        texts=[
            SimpleNamespace(
                content="账户4",
                bbox=anchor_bbox or [49.5, 329.0, 70.0, 340.5],
            )
        ],
        tables=[table],
    )
    return SimpleNamespace(
        pages=[page],
        reading_order_by_logical={19: 19},
        reading_order_resolution={"resolved": True, "authoritative": True},
        tables_continue=lambda _left, _right: None,
        corrected_evidence_pages=lambda: [],
        _personal_detail_extraction_issues=[],
    )


def test_native_liability_card_can_use_one_adjacent_printed_ordinal() -> None:
    context = _adjacent_liability_anchor_context()

    records = PBOCPersonalDetailNativeParser(context).records(
        "repayment_liability_records"
    )

    assert len(records) == 1
    record = records[0]
    assert record.fields["__printed_sequence"] == "4"
    assert record.fields["管理机构"] == "中国农业银行股份有限公司大理分行"
    assert record.fields["开立日期"] == "2023.09.18"
    assert record.binding_quality_by_field["__printed_sequence"] == (
        "canonical_card_anchor"
    )
    sequence_ref = record.source_refs_by_field["__printed_sequence"][0]
    assert sequence_ref["source"] == "native_detail_canonical_anchor_text"
    assert sequence_ref["logical_page"] == 19
    assert sequence_ref["source_page"] == 10
    assert sequence_ref["bbox"] == [49.5, 329.0, 70.0, 340.5]
    assert "还款责任金额" not in record.fields
    assert "保证合同编号" not in record.fields
    assert {"还款责任金额", "保证合同编号"} <= record.unresolved_labels


def test_native_liability_card_rejects_distant_printed_ordinal() -> None:
    context = _adjacent_liability_anchor_context(
        anchor_bbox=[49.5, 280.0, 70.0, 300.0]
    )

    records = PBOCPersonalDetailNativeParser(context).records(
        "repayment_liability_records"
    )

    assert records == []


def test_liability_extraction_retains_ordinal_card_and_reports_blank_slots() -> None:
    context = _adjacent_liability_anchor_context()

    records = native_extraction._extract_liabilities(context)

    assert len(records) == 1
    record = records[0]
    assert record["_printed_sequence"] == 4
    assert record["institution"] == "中国农业银行股份有限公司大理分行"
    assert record["open_date"] == "2023-09-18"
    assert record.get("responsibility_amount") is None
    assert record.get("contract_number") is None
    assert record.get("responsibility_amount") != 700210
    unresolved_issues = {
        str(issue.get("field_name") or "")
        for issue in context._personal_detail_extraction_issues
        if issue.get("issue_code")
        == "candidate_b_repayment_responsibility_required_field_unresolved"
        and issue.get("target_record_id") == record["liability_id"]
    }
    assert {"responsibility_amount", "contract_number"} <= unresolved_issues


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
    table = _owned_liability_table(SimpleNamespace(
        table_id="liability-columns-1",
        metadata={"raw_rows": [header, values]},
        headers=[],
        rows=[],
        bbox=[20.0, 40.0, 580.0, 88.0],
    ))
    page = SimpleNamespace(
        page_number=8,
        source_page_number=4,
        canonical_template_id="repayment_responsibility",
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


def test_native_liability_requires_coherent_page_and_table_owner() -> None:
    for page_role, table_role in (
        ("public_information", "public_information"),
        ("information_summary", "information_summary"),
        ("", ""),
        ("repayment_responsibility", "public_information"),
        ("public_information", "repayment_responsibility"),
    ):
        context = _native_context(PACKED_VALUE)
        page = context.pages[0]
        table = page.tables[0]
        page.canonical_template_id = page_role
        table.metadata["canonical_template_id"] = table_role

        assert (
            PBOCPersonalDetailNativeParser(context).records(
                "repayment_liability_records"
            )
            == []
        )


def test_native_liability_owner_accepts_reordered_columns_and_variable_rows() -> None:
    context = _native_context(PACKED_VALUE)
    table = context.pages[0].tables[0]
    headers = [
        "保证合同编号",
        "币种",
        "还款责任金额",
        "责任人类型",
        "到期日期",
        "开立日期",
        "业务种类",
        "管理机构",
    ]
    first = [
        "GENERIC-CONTRACT-ALPHA",
        "人民币元",
        "25,000",
        "保证",
        "2027.11.03",
        "2025.02.14",
        "贷款",
        "甲银行",
    ]
    second = [
        "GENERIC-CONTRACT-BETA",
        "人民币元",
        "98,765",
        "共同借款人",
        "2031.01.09",
        "2024.12.28",
        "贷款",
        "乙银行",
    ]
    table.metadata["raw_rows"] = [headers, first, headers, second]
    table.metadata["source_cell_bboxes"] = [
        [
            [20.0 + column * 60.0, 40.0 + row * 22.0, 76.0 + column * 60.0, 58.0 + row * 22.0]
            for column in range(8)
        ]
        for row in range(4)
    ]
    table.metadata["cell_evidence_ids"] = [
        [[f"liability:{row}:{column}"] for column in range(8)]
        for row in range(4)
    ]

    records = PBOCPersonalDetailNativeParser(context).records(
        "repayment_liability_records"
    )

    assert [record.fields["保证合同编号"] for record in records] == [
        "GENERIC-CONTRACT-ALPHA",
        "GENERIC-CONTRACT-BETA",
    ]
    assert [record.fields["管理机构"] for record in records] == ["甲银行", "乙银行"]


def test_corrected_liability_rejects_heading_inside_prose_or_contents() -> None:
    for heading in (
        "目录 （六）相关还款责任信息 ........ 9",
        "本页说明相关还款责任信息的填报规则",
    ):
        context = SimpleNamespace(
            pages=[],
            _personal_detail_extraction_issues=[],
            corrected_evidence_pages=lambda heading=heading: [
                {
                    "page": 27,
                    "source_page": 14,
                    "canonical_template_id": "public_information",
                    "lines": [
                        _line(heading, 10.0, evidence_id="prose"),
                        _line(PACKED_HEADER, 40.0, evidence_id="header"),
                        _line(PACKED_VALUE, 70.0, evidence_id="value"),
                    ],
                }
            ],
        )

        assert (
            PBOCPersonalDetailNativeParser(context).records(
                "repayment_liability_records"
            )
            == []
        )


def test_corrected_liability_rejects_exact_heading_on_incompatible_page_owner() -> None:
    context = SimpleNamespace(
        pages=[],
        _personal_detail_extraction_issues=[],
        corrected_evidence_pages=lambda: [
            {
                "page": 31,
                "source_page": 16,
                "canonical_template_id": "public_information",
                "lines": [
                    _line("（九）相关还款责任信息", 10.0, evidence_id="heading"),
                    _line(PACKED_HEADER, 40.0, evidence_id="header"),
                    _line(PACKED_VALUE, 70.0, evidence_id="value"),
                ],
            }
        ],
    )

    assert (
        PBOCPersonalDetailNativeParser(context).records(
            "repayment_liability_records"
        )
        == []
    )
