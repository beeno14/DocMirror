from __future__ import annotations

from docmirror.models.entities.parse_result import (
    CellValue,
    DocumentEntities,
    LogicalTable,
    PageContent,
    ParseResult,
    ResultStatus,
    TableBlock,
    TableRow,
    TextBlock,
)
from docmirror.models.sealed import seal_parse_result
from docmirror.plugins._runtime.plugin_registry import PluginRegistry
from docmirror.plugins.audit_report import derive_audit_report_projection
from docmirror.plugins.audit_report.community_plugin import AuditReportPlugin
from docmirror.plugins.generic.community_plugin import GenericCommunityPlugin


def _row(page: int, table_id: str, row_index: int, values: list[str]) -> TableRow:
    return TableRow(
        cells=[
            CellValue(
                text=value,
                row_index=row_index,
                col_index=column,
                bbox=[column * 100.0, row_index * 20.0, (column + 1) * 100.0, (row_index + 1) * 20.0],
                evidence_ids=[f"ev:{page:04d}:{table_id}:{row_index:04d}:{column:02d}"],
                geometry_status="exact",
            )
            for column, value in enumerate(values)
        ],
        source_page=page,
        source_physical_id=table_id,
        source_row_index=row_index,
    )


def _audit_result(*pages: PageContent) -> ParseResult:
    return ParseResult(
        status=ResultStatus.SUCCESS,
        pages=list(pages),
        entities=DocumentEntities(document_type="audit_report"),
    )


def test_audit_projection_separates_report_numbers_and_filters_note_fragments() -> None:
    result = _audit_result(
        PageContent(
            page_number=1,
            texts=[
                TextBlock(
                    content=(
                        "被审计单位:测试科技股份有限公司\n"
                        "2024年度审计报告\n"
                        "鼎迈会师审字〔2025〕第0123号\n"
                        "报告编号:沪25ABC123\n"
                        "资本化期间:这不是文档基本信息"
                    ),
                    bbox=[10.0, 10.0, 500.0, 120.0],
                    evidence_ids=["ev:audit:cover"],
                )
            ],
        ),
        PageContent(
            page_number=2,
            texts=[
                TextBlock(
                    content="一、审计意见\n我们认为，财务报表在所有重大方面公允反映了公司的财务状况。",
                    bbox=[20.0, 30.0, 500.0, 100.0],
                    evidence_ids=["ev:audit:opinion"],
                ),
                TextBlock(
                    content="二、形成审计意见的基础\n我们按照中国注册会计师审计准则的规定执行了审计工作。",
                    bbox=[20.0, 120.0, 500.0, 200.0],
                    evidence_ids=["ev:audit:basis"],
                ),
            ],
        ),
    )

    projection = derive_audit_report_projection(result)

    assert projection.projector_id == "audit_report"
    assert projection.domain_facts["subject_name"] == "测试科技股份有限公司"
    assert projection.domain_facts["audit_document_number"] == "鼎迈会师审字〔2025〕第0123号"
    assert projection.domain_facts["regulatory_report_id"] == "沪25ABC123"
    assert projection.domain_facts["audit_opinion_type"] == "unmodified"
    assert "community_generic_fallback" not in projection.warnings
    assert "资本化期间" not in projection.domain_facts
    assert "资本化期间" not in projection.entity_fields
    assert projection.sections[0]["id"] == "section_audit_metadata"
    assert {section["type"] for section in projection.sections} >= {"audit_opinion", "basis_for_opinion"}


def test_audit_projection_reuses_financial_statement_projection_and_preserves_raw_glyphs() -> None:
    table = TableBlock(
        table_id="pt_5_0",
        page=5,
        headers=["项目", "附注", "2024年12月31日", "2023年12月31日"],
        rows=[
            _row(5, "pt_5_0", 0, ["人⺠币资金", "五、1", "1,200.00", "900.00"]),
            _row(5, "pt_5_0", 1, ["资产总计", "", "1,200.00", "900.00"]),
        ],
        bbox=[20.0, 100.0, 800.0, 500.0],
        evidence_ids=["ev:table:balance"],
    )
    result = _audit_result(
        PageContent(
            page_number=1,
            texts=[TextBlock(content="一、审计意见\n我们认为财务报表公允反映了财务状况。")],
        ),
        PageContent(
            page_number=2,
            texts=[TextBlock(content="二、形成审计意见的基础")],
        ),
        PageContent(page_number=5, texts=[TextBlock(content="资产负债表")], tables=[table]),
    )

    projection = derive_audit_report_projection(result)
    rows = projection.datasets["balance_sheet"]

    assert list(projection.datasets) == ["balance_sheet"]
    assert rows[0]["raw"]["item"] == "人⺠币资金"
    assert rows[0]["canonical_raw"]["item"] == "人⺠币资金"
    assert rows[0]["normalized"]["item"] == "人民币资金"
    assert rows[0]["source"]["page"] == 5
    assert rows[0]["source"]["evidence_ids"]
    assert projection.semantic["dataset_labels"]["balance_sheet"] == "资产负债表"

    projector = PluginRegistry().get_projector("audit_report", "community")
    assert isinstance(projector, AuditReportPlugin)
    bundle = projector.project_bundle(seal_parse_result(result), file_path="audit.pdf")
    assert bundle is not None
    dataset = next(item for item in bundle.json_payload()["datasets"] if item["name"] == "balance_sheet")
    assert dataset["completeness"]["verified"] is True
    assert dataset["completeness"]["basis"] == "source_statement_rows"
    assert dataset["status"] == "complete"


def test_audit_projection_recovers_positioned_statement_text_when_table_is_missing() -> None:
    def text(value: str, bbox: list[float], evidence: str) -> TextBlock:
        return TextBlock(content=value, bbox=bbox, confidence=0.98, evidence_ids=[evidence])

    result = _audit_result(
        PageContent(
            page_number=5,
            texts=[
                text("资产负债表", [240.0, 40.0, 340.0, 55.0], "ev:title"),
                text("项目", [100.0, 80.0, 160.0, 90.0], "ev:header:item"),
                text("附注", [250.0, 80.0, 290.0, 90.0], "ev:header:note"),
                text("2024年12月31日", [350.0, 80.0, 410.0, 90.0], "ev:header:current"),
                text("2023年12月31日", [450.0, 80.0, 510.0, 90.0], "ev:header:previous"),
                text("流动资产：", [100.0, 100.0, 170.0, 110.0], "ev:row:0:item"),
                text("货币资金", [100.0, 120.0, 170.0, 130.0], "ev:row:1:item"),
                text("五、1", [250.0, 120.0, 290.0, 130.0], "ev:row:1:note"),
                text("1,200.00", [350.0, 120.0, 410.0, 130.0], "ev:row:1:current"),
                text("900.00", [450.0, 120.0, 510.0, 130.0], "ev:row:1:previous"),
                text("资产总计", [100.0, 140.0, 170.0, 150.0], "ev:row:2:item"),
                text("1,200.00", [350.0, 140.0, 410.0, 150.0], "ev:row:2:current"),
                text("900.00", [450.0, 140.0, 510.0, 150.0], "ev:row:2:previous"),
                text("法定代表人：", [100.0, 170.0, 180.0, 180.0], "ev:footer"),
            ],
        )
    )

    projection = derive_audit_report_projection(result)

    rows = projection.datasets["balance_sheet"]
    assert len(rows) == 3
    assert rows[1]["raw"] == {
        "item": "货币资金",
        "note_reference": "五、1",
        "current_period_amount": "1,200.00",
        "previous_period_amount": "900.00",
    }
    assert rows[1]["source"]["recovery"] == "positioned_text_rows"
    assert rows[1]["source"]["page"] == 5
    assert any(warning.startswith("AUDIT_STATEMENT_TEXT_ROWS_RECOVERED") for warning in projection.warnings)


def test_audit_projection_does_not_guess_statement_rows_without_positioned_header() -> None:
    result = _audit_result(
        PageContent(
            page_number=5,
            texts=[TextBlock(content="资产负债表\n项目 附注 期末余额 期初余额\n货币资金 五、1 1,200.00 900.00")],
        )
    )

    projection = derive_audit_report_projection(result)

    assert "balance_sheet" not in projection.datasets
    assert "AUDIT_FINANCIAL_STATEMENT_UNRESOLVED:page=5:kind=balance_sheet" in projection.warnings


def test_audit_projection_recovers_cross_page_header_promoted_to_data() -> None:
    first = TableBlock(
        table_id="pt_40_0",
        page=40,
        headers=["项目", "序号", "期末余额", "期初余额"],
        rows=[_row(40, "pt_40_0", 0, ["关联方款项", "8", "100.00", "80.00"])],
        bbox=[20.0, 100.0, 800.0, 760.0],
        evidence_ids=["ev:table:40"],
    )
    continuation = TableBlock(
        table_id="pt_41_0",
        page=41,
        headers=["其他往来单位", "9", "920,895.22", "608,519.54"],
        rows=[_row(41, "pt_41_0", 1, ["代垫款", "10", "50.00", "30.00"])],
        bbox=[20.0, 20.0, 800.0, 300.0],
        evidence_ids=["ev:table:41"],
    )
    second_continuation = TableBlock(
        table_id="pt_42_0",
        page=42,
        headers=["其他往来单位二", "11", "70.00", "40.00"],
        rows=[_row(42, "pt_42_0", 1, ["预付款", "12", "20.00", "10.00"])],
        bbox=[20.0, 20.0, 800.0, 300.0],
        evidence_ids=["ev:table:42"],
    )
    result = _audit_result(
        PageContent(page_number=40, tables=[first]),
        PageContent(page_number=41, tables=[continuation]),
        PageContent(page_number=42, tables=[second_continuation]),
    )

    projection = derive_audit_report_projection(result)
    rows = next(iter(projection.datasets.values()))

    assert len(projection.datasets) == 1
    assert rows[1]["raw"] == {
        "项目": "其他往来单位",
        "序号": "9",
        "期末余额": "920,895.22",
        "期初余额": "608,519.54",
    }
    assert rows[2]["raw"] == {
        "项目": "代垫款",
        "序号": "10",
        "期末余额": "50.00",
        "期初余额": "30.00",
    }
    assert rows[3]["raw"]["项目"] == "其他往来单位二"
    assert rows[4]["source"]["page"] == 42
    assert "canonical_raw" not in rows[2]
    assert sum("mode=data_header" in warning for warning in projection.warnings) == 2


def test_audit_projection_recovers_only_evidence_backed_missing_outer_columns() -> None:
    first = TableBlock(
        table_id="pt_46_0",
        page=46,
        headers=["类别", "项目", "账面余额", "坏账准备", "账面价值", "备注"],
        rows=[_row(46, "pt_46_0", 0, ["其他应收款", "保证金", "100.00", "5.00", "95.00", ""])],
        bbox=[20.0, 100.0, 800.0, 760.0],
    )
    continuation = TableBlock(
        table_id="pt_47_0",
        page=47,
        headers=["项目", "账面余额", "坏账准备", "账面价值"],
        rows=[
            _row(47, "pt_47_0", 0, ["备用金", "20.00", "1.00", "19.00"]),
            _row(47, "pt_47_0", 1, ["代垫款", "30.00", "2.00", "28.00"]),
        ],
        bbox=[20.0, 20.0, 800.0, 300.0],
    )
    result = _audit_result(
        PageContent(page_number=46, tables=[first]),
        PageContent(
            page_number=47,
            texts=[
                TextBlock(
                    content="其他应收款\n其他应收款",
                    bbox=[10.0, 10.0, 120.0, 40.0],
                    evidence_ids=["ev:category:47"],
                )
            ],
            tables=[continuation],
        ),
    )

    projection = derive_audit_report_projection(result)
    rows = next(iter(projection.datasets.values()))

    assert len(projection.datasets) == 1
    assert rows[1]["raw"] == {
        "类别": "其他应收款",
        "项目": "备用金",
        "账面余额": "20.00",
        "坏账准备": "1.00",
        "账面价值": "19.00",
        "备注": "",
    }
    assert rows[2]["source"]["page"] == 47
    category_ref = next(ref for ref in rows[1]["source"]["source_cell_refs"] if ref["field_name"] == "类别")
    assert category_ref["evidence_ids"] == ["ev:category:47"]
    assert any("mode=missing_outer" in warning for warning in projection.warnings)


def test_audit_projection_promotes_note_header_rows_instead_of_emitting_them_as_data() -> None:
    table = TableBlock(
        table_id="pt_20_0",
        page=20,
        headers=["项目", "col_1", "col_2", "col_3", "col_4"],
        rows=[
            _row(20, "pt_20_0", 0, ["担保方", "担保金额", "担保起始日担保到期日", "", "是否履行完毕"]),
            _row(20, "pt_20_0", 1, ["甲公司", "25,000.00", "2024-01-01", "2025-01-01", "否"]),
        ],
    )
    result = _audit_result(PageContent(page_number=20, texts=[TextBlock(content="财务报表附注")], tables=[table]))

    projection = derive_audit_report_projection(result)
    rows = next(iter(projection.datasets.values()))

    assert len(rows) == 1
    assert rows[0]["raw"] == {
        "担保方": "甲公司",
        "担保金额": "25,000.00",
        "担保起始日": "2024-01-01",
        "担保到期日": "2025-01-01",
        "是否履行完毕": "否",
    }
    assert rows[0]["source"]["header_recovery"] == "promoted_source_rows"
    assert any(warning.startswith("AUDIT_NOTE_HEADER_ROWS_PROMOTED") for warning in projection.warnings)


def test_audit_projection_splits_only_exact_adjacent_note_values() -> None:
    comparative = TableBlock(
        table_id="pt_21_0",
        page=21,
        headers=["项目", "期末余额", "期初余额"],
        rows=[_row(21, "pt_21_0", 0, ["合计", "153,098.4376,403.83", ""])],
    )
    aging = TableBlock(
        table_id="pt_22_0",
        page=22,
        headers=["项目", "期末余额", "账龄"],
        rows=[_row(22, "pt_22_0", 0, ["往来款", "1,648,700.001年以内", ""])],
    )
    result = _audit_result(
        PageContent(page_number=21, texts=[TextBlock(content="财务报表附注")], tables=[comparative]),
        PageContent(page_number=22, tables=[aging]),
    )

    projection = derive_audit_report_projection(result)
    rows = [row for dataset_rows in projection.datasets.values() for row in dataset_rows]

    assert rows[0]["raw"]["期末余额"] == "153,098.43"
    assert rows[0]["raw"]["期初余额"] == "76,403.83"
    assert rows[1]["raw"]["期末余额"] == "1,648,700.00"
    assert rows[1]["raw"]["账龄"] == "1年以内"
    assert rows[0]["source"]["recovery_raw"]["期末余额"] == "153,098.4376,403.83"
    assert rows[0]["source_cell_refs"][-1]["field_name"] == "期初余额"
    assert any(warning.startswith("AUDIT_NOTE_ADJACENT_CELLS_SPLIT") for warning in projection.warnings)


def test_generic_plugin_keeps_non_audit_documents_on_original_path() -> None:
    result = ParseResult(
        status=ResultStatus.SUCCESS,
        pages=[PageContent(page_number=1, texts=[TextBlock(content="名称:测试")])],
        entities=DocumentEntities(document_type="generic"),
    )

    projection = GenericCommunityPlugin().derive(result, "名称:测试")

    assert projection.projector_id == "generic"
    assert "audit_projection" not in projection.domain_facts


def test_audit_projection_keeps_bundle_v3_and_emits_only_curated_metadata() -> None:
    result = _audit_result(
        PageContent(
            page_number=1,
            texts=[
                TextBlock(
                    content=(
                        "被审计单位:测试科技股份有限公司\n"
                        "鼎迈会师审字〔2025〕第0123号\n"
                        "资本化期间:这不是文档基本信息\n"
                        "一、审计意见\n我们认为，财务报表公允反映了财务状况。\n"
                        "二、形成审计意见的基础"
                    ),
                    evidence_ids=["ev:audit:page1"],
                )
            ],
        )
    )

    projector = PluginRegistry().get_projector("audit_report", "community")
    assert isinstance(projector, AuditReportPlugin)
    bundle = projector.project_bundle(seal_parse_result(result), file_path="audit.pdf")
    assert bundle is not None
    payload = bundle.json_payload()

    assert {"schema", "document", "sections", "datasets", "files", "warnings"} <= set(payload)
    assert "domain" not in payload
    assert payload["schema"]["version"] == "3.0.0"
    assert payload["document"]["type"] == "audit_report"
    assert bundle.classification["projector_id"] == "audit_report"
    metadata = next(section for section in payload["sections"] if section["id"] == "section_audit_metadata")
    metadata_keys = {item["key"] for item in metadata["items"]}
    assert {"subject_name", "audit_document_number", "audit_opinion_type"} <= metadata_keys
    assert "资本化期间" not in metadata_keys

    direct_bundle = AuditReportPlugin().project_bundle(seal_parse_result(result), file_path="audit.pdf")
    assert direct_bundle is not None
    assert direct_bundle.classification["projector_id"] == "audit_report"


def test_audit_projection_recovers_concatenated_comparative_amounts() -> None:
    table = TableBlock(
        table_id="pt_7_0",
        page=7,
        headers=["项目", "附注2022年度2021年度", ""],
        rows=[
            _row(7, "pt_7_0", 0, ["项目", "附注2022年度2021年度", ""]),
            _row(7, "pt_7_0", 1, ["一、营业收入", "五、27177,192,895.90171,385,322.20", ""]),
            _row(7, "pt_7_0", 2, ["其中：利息费用", "3,742,068,202,493,030.46", ""]),
            _row(7, "pt_7_0", 3, ["利息收入", "7,444,975,148.42", ""]),
            _row(7, "pt_7_0", 4, ["投资活动现金流量净额", "-8,257.51", "5.2311,325,620.03"]),
            _row(7, "pt_7_0", 5, ["末列单一金额", "", "6,618.98"]),
            _row(7, "pt_7_0", 6, ["本期单一金额", "419,585.74", ""]),
        ],
    )
    result = _audit_result(
        PageContent(page_number=7, texts=[TextBlock(content="利润表")], tables=[table]),
    )

    projection = derive_audit_report_projection(result)
    rows = projection.datasets["income_statement"]

    assert rows[0]["raw"] == {
        "item": "一、营业收入",
        "note_reference": "五、27",
        "current_period_amount": "177,192,895.90",
        "previous_period_amount": "171,385,322.20",
    }
    assert rows[1]["raw"]["current_period_amount"] == "3,742,068.20"
    assert rows[1]["raw"]["previous_period_amount"] == "2,493,030.46"
    assert rows[2]["raw"]["current_period_amount"] == "7,444.97"
    assert rows[2]["raw"]["previous_period_amount"] == "5,148.42"
    assert rows[3]["raw"]["current_period_amount"] == "-8,257,515.23"
    assert rows[3]["raw"]["previous_period_amount"] == "11,325,620.03"
    assert rows[4]["raw"]["current_period_amount"] == "6,618.98"
    assert rows[4]["raw"]["previous_period_amount"] == ""
    assert rows[5]["raw"]["current_period_amount"] == "419,585.74"
    assert rows[5]["raw"]["previous_period_amount"] == ""
    assert rows[4]["source"]["recovery"] == "source_cell_repartition"
    assert rows[5]["source"]["recovery"] == "source_cell_repartition"
    assert rows[1]["source"]["recovery_raw"] == "其中：利息费用 | 3,742,068,202,493,030.46"
    assert rows[1]["review"]["required"] is True
    assert any(warning.startswith("AUDIT_AMOUNT_SPLIT_INFERRED") for warning in projection.warnings)


def test_audit_projection_prefers_recovered_comparatives_when_amounts_are_stuck_in_item() -> None:
    table = TableBlock(
        table_id="pt_7_0",
        page=7,
        headers=["项目附注本期金额上期金额", "", ""],
        rows=[
            _row(7, "pt_7_0", 0, ["项目附注本期金额上期金额", "", ""]),
            _row(7, "pt_7_0", 1, ["一、营业收入五、27177,192,895.90171,385,322.20", "", ""]),
            _row(7, "pt_7_0", 2, ["信用减值损失-3,711,229.951,443,795.31", "", ""]),
        ],
    )
    result = _audit_result(PageContent(page_number=7, texts=[TextBlock(content="利润表")], tables=[table]))

    projection = derive_audit_report_projection(result)
    rows = projection.datasets["income_statement"]

    assert rows[0]["raw"] == {
        "item": "一、营业收入",
        "note_reference": "五、27",
        "current_period_amount": "177,192,895.90",
        "previous_period_amount": "171,385,322.20",
    }
    assert rows[1]["raw"]["item"] == "信用减值损失"
    assert rows[1]["normalized"]["current_period_amount"] == "-3711229.95"
    assert rows[1]["normalized"]["previous_period_amount"] == "1443795.31"
    assert all(not any(char.isdigit() for char in row["raw"]["item"][-8:]) for row in rows)
    assert any(warning.startswith("AUDIT_STATEMENT_CELLS_RECOVERED") for warning in projection.warnings)


def test_audit_projection_removes_malformed_comparative_header_row() -> None:
    table = TableBlock(
        table_id="pt_8_0",
        page=8,
        headers=["项目", "附注", "本期金额", "上期金额"],
        rows=[
            _row(8, "pt_8_0", 0, ["一项目附注", "", "本期金额", "上期金额"]),
            _row(8, "pt_8_0", 1, ["经营活动产生的现金流量净额", "", "10.00", "9.00"]),
        ],
    )
    result = _audit_result(PageContent(page_number=8, texts=[TextBlock(content="现金流量表")], tables=[table]))

    projection = derive_audit_report_projection(result)

    assert [row["raw"]["item"] for row in projection.datasets["cash_flow_statement"]] == ["经营活动产生的现金流量净额"]
    assert any(warning.startswith("AUDIT_STATEMENT_HEADER_ROWS_REMOVED") for warning in projection.warnings)


def test_audit_projection_preserves_single_comparative_amount_period_position() -> None:
    table = TableBlock(
        table_id="pt_8_0",
        page=8,
        headers=["项目", "附注", "本期金额", "上期金额"],
        rows=[
            _row(8, "pt_8_0", 0, ["收到其他款项", "", "", "6,618.98"]),
            _row(8, "pt_8_0", 1, ["偿还债务支付的现金", "", "55,557,000.00", ""]),
        ],
    )
    result = _audit_result(PageContent(page_number=8, texts=[TextBlock(content="现金流量表")], tables=[table]))

    projection = derive_audit_report_projection(result)
    rows = projection.datasets["cash_flow_statement"]

    assert rows[0]["raw"]["current_period_amount"] == ""
    assert rows[0]["raw"]["previous_period_amount"] == "6,618.98"
    assert rows[1]["raw"]["current_period_amount"] == "55,557,000.00"
    assert rows[1]["raw"]["previous_period_amount"] == ""


def test_audit_projection_merges_balance_continuation_with_liability_columns() -> None:
    assets = TableBlock(
        table_id="pt_5_0",
        page=5,
        headers=["", "项目", "附注2022年12月31日", "", "", "2021年12月31日", ""],
        rows=[
            _row(5, "pt_5_0", 0, ["", "项目", "附注2022年12月31日", "", "", "2021年12月31日", ""]),
            _row(5, "pt_5_0", 1, ["", "货币资金", "五、11,600,060.84", "", "", "8,145,477.09", ""]),
            _row(5, "pt_5_0", 2, ["", "应收账款", "五、2", "41,097,688.41", "", "23,724,610.17", ""]),
        ],
    )
    liabilities = TableBlock(
        table_id="pt_6_0",
        page=6,
        headers=["项目附注2022年12月31日", "", "", "", "2021年12月31日"],
        rows=[
            _row(6, "pt_6_0", 0, ["项目附注2022年12月31日", "", "", "", "2021年12月31日"]),
            _row(6, "pt_6_0", 1, ["短期借款五、1237,900,000.00", "", "", "", "46,100,000.00"]),
        ],
    )
    result = _audit_result(
        PageContent(page_number=5, texts=[TextBlock(content="资产负债表")], tables=[assets]),
        PageContent(page_number=6, texts=[TextBlock(content="资产负债表(续)")], tables=[liabilities]),
    )

    projection = derive_audit_report_projection(result)
    rows = projection.datasets["balance_sheet"]

    assert list(projection.datasets) == ["balance_sheet"]
    assert len(rows) == 3
    assert rows[0]["raw"]["note_reference"] == "五、1"
    assert rows[0]["raw"]["current_period_amount"] == "1,600,060.84"
    assert rows[1]["raw"]["note_reference"] == "五、2"
    assert rows[1]["raw"]["current_period_amount"] == "41,097,688.41"
    assert rows[2]["raw"]["item"] == "短期借款"
    assert rows[2]["raw"]["note_reference"] == "五、12"
    assert rows[2]["raw"]["current_period_amount"] == "37,900,000.00"
    assert rows[2]["source"]["page"] == 6


def test_audit_projection_preserves_clean_four_cell_balance_continuation() -> None:
    table = TableBlock(
        table_id="pt_6_0",
        page=6,
        headers=[],
        rows=[
            _row(6, "pt_6_0", 0, ["", "", "", ""]),
            _row(6, "pt_6_0", 1, ["短期借款", "五、12", "29,000,000.00", "37,900,000.00"]),
            _row(6, "pt_6_0", 2, ["盈余公积五、252,784,842.642,584,135.99", "", "", ""]),
            _row(6, "pt_6_0", 3, ["负债和所有者（股东）权益总计129,651,353.06130,327,981.38", "", "", ""]),
        ],
    )
    result = _audit_result(PageContent(page_number=6, texts=[TextBlock(content="资产负债表（续）")], tables=[table]))

    projection = derive_audit_report_projection(result)

    rows = projection.datasets["balance_sheet"]
    assert rows[0]["raw"] == {
        "item": "短期借款",
        "note_reference": "五、12",
        "current_period_amount": "29,000,000.00",
        "previous_period_amount": "37,900,000.00",
    }
    assert rows[1]["raw"] == {
        "item": "盈余公积",
        "note_reference": "五、25",
        "current_period_amount": "2,784,842.64",
        "previous_period_amount": "2,584,135.99",
    }
    assert rows[2]["raw"] == {
        "item": "负债和所有者（股东）权益总计",
        "note_reference": "",
        "current_period_amount": "129,651,353.06",
        "previous_period_amount": "130,327,981.38",
    }


def test_audit_projection_recovers_evidence_backed_balance_section_label() -> None:
    table = TableBlock(
        table_id="pt_6_0",
        page=6,
        headers=["项目", "附注", "期末余额", "期初余额"],
        rows=[_row(6, "pt_6_0", 0, ["短期借款", "五、12", "29,000,000.00", "37,900,000.00"])],
    )
    result = _audit_result(
        PageContent(
            page_number=6,
            texts=[
                TextBlock(content="资产负债表（续）"),
                TextBlock(content="流动负债:", bbox=[20.0, 80.0, 120.0, 95.0], evidence_ids=["ev:label"]),
            ],
            tables=[table],
        )
    )

    projection = derive_audit_report_projection(result)
    rows = projection.datasets["balance_sheet"]

    assert [row["raw"]["item"] for row in rows] == ["流动负债:", "短期借款"]
    assert rows[0]["source"]["recovery"] == "canonical_text_balance_section_label"
    assert rows[0]["source"]["evidence_ids"] == ["ev:label"]
    assert any(warning.startswith("AUDIT_BALANCE_SECTION_LABEL_RECOVERED") for warning in projection.warnings)


def test_audit_projection_repairs_exact_embedded_balance_amount_shifts() -> None:
    table = TableBlock(
        table_id="pt_5_0",
        page=5,
        headers=["项目", "附注", "期末余额", "期初余额"],
        rows=[
            _row(5, "pt_5_0", 0, ["递延所得税资产五、11 1,860,765.41", "", "1,051,966.08", ""]),
            _row(5, "pt_5_0", 1, ["固定资产五、7 363.10", "", "0.69", "648,839.90"]),
            _row(5, "pt_5_0", 2, ["长期股权投资五、6 2,856.93", "", "7,321,860,000.00", ""]),
            _row(5, "pt_5_0", 3, ["应付票据五、135,005,000.00", "", "", ""]),
        ],
    )
    result = _audit_result(PageContent(page_number=5, texts=[TextBlock(content="资产负债表")], tables=[table]))

    projection = derive_audit_report_projection(result)
    rows = projection.datasets["balance_sheet"]

    assert rows[0]["raw"] == {
        "item": "递延所得税资产",
        "note_reference": "五、11",
        "current_period_amount": "1,860,765.41",
        "previous_period_amount": "1,051,966.08",
    }
    assert rows[1]["raw"]["current_period_amount"] == "363,100.69"
    assert rows[1]["raw"]["previous_period_amount"] == "648,839.90"
    assert rows[2]["raw"]["current_period_amount"] == "2,856,937.32"
    assert rows[2]["raw"]["previous_period_amount"] == "1,860,000.00"
    assert rows[3]["raw"] == {
        "item": "应付票据",
        "note_reference": "五、13",
        "current_period_amount": "5,005,000.00",
        "previous_period_amount": "",
    }
    assert rows[0]["source"]["recovery"] == "embedded_amount_shift"
    assert rows[2]["review"]["required"] is True
    assert any(warning.startswith("AUDIT_BALANCE_AMOUNT_SHIFT_RECOVERED") for warning in projection.warnings)
    assert any(warning.startswith("AUDIT_AMOUNT_FRAGMENT_INFERRED") for warning in projection.warnings)


def test_audit_projection_expands_exact_standard_item_sequence_and_keeps_review_status() -> None:
    table = TableBlock(
        table_id="pt_6_0",
        page=6,
        headers=["项目", "附注", "期末余额", "期初余额"],
        rows=[
            _row(6, "pt_6_0", 0, ["交易性金融负债衍生金融负债应付票据", "", "", ""]),
            _row(6, "pt_6_0", 1, ["", "五、13", "5,000.00", "5,005,000.00"]),
        ],
    )
    result = _audit_result(PageContent(page_number=6, texts=[TextBlock(content="资产负债表（续）")], tables=[table]))

    projection = derive_audit_report_projection(result)
    rows = projection.datasets["balance_sheet"]

    assert [row["raw"]["item"] for row in rows] == ["交易性金融负债", "衍生金融负债", "应付票据"]
    assert rows[-1]["raw"]["note_reference"] == "五、13"
    assert rows[-1]["raw"]["current_period_amount"] == "5,000.00"
    assert all(row["review"]["required"] is True for row in rows)
    assert any(warning.startswith("AUDIT_MERGED_ITEM_ROWS_INFERRED") for warning in projection.warnings)
    assert projection.domain_facts["dataset_verification_blockers"]["balance_sheet"]


def test_audit_projection_recovers_evidence_backed_transposed_owner_equity_table() -> None:
    cash = TableBlock(
        table_id="pt_8_0",
        page=8,
        headers=["项目", "本期金额", "上期金额"],
        rows=[_row(8, "pt_8_0", 0, ["经营活动产生的现金流量净额", "10.00", "9.00"])],
    )
    logical = [
        ["项目", "本期金额", "上期金额", "所有者权益合计"],
        ["一、上年年末余额", "1.00", "2.00", "3.00"],
        ["加：会计政策变更", "", "", ""],
        ["二、本年年初余额", "1.00", "2.00", "3.00"],
        ["三、本年增减变动金额", "4.00", "5.00", "9.00"],
        ["（一）综合收益总额", "4.00", "5.00", "9.00"],
        ["（二）所有者投入资本", "", "", ""],
        ["四、本年期末余额", "5.00", "7.00", "12.00"],
    ]
    physical = [list(column) for column in zip(*logical, strict=True)]
    owner_table = TableBlock(
        table_id="pt_9_0",
        page=9,
        headers=[],
        rows=[_row(9, "pt_9_0", row_index, values) for row_index, values in enumerate(physical)],
    )
    result = _audit_result(
        PageContent(page_number=8, texts=[TextBlock(content="现金流量表")], tables=[cash]),
        PageContent(page_number=9, tables=[owner_table]),
        PageContent(page_number=10, texts=[TextBlock(content="财务报表附注")]),
    )

    projection = derive_audit_report_projection(result)
    rows = projection.datasets["owners_equity_changes"]

    assert len(rows) == 7
    assert rows[-1]["raw"]["item"] == "四、本年期末余额"
    assert rows[-1]["source"]["recovery"] == "rotated_table_transpose"
    assert any(warning.startswith("AUDIT_OWNER_EQUITY_ROTATION_RECOVERED") for warning in projection.warnings)


def test_audit_projection_drops_unreliable_landscape_pseudotable() -> None:
    cash = TableBlock(
        table_id="pt_8_0",
        page=8,
        headers=["项目", "本期金额", "上期金额"],
        rows=[_row(8, "pt_8_0", 0, ["经营活动产生的现金流量净额", "10.00", "9.00"])],
    )
    noise = TableBlock(
        table_id="pt_9_0",
        page=9,
        headers=[],
        rows=[_row(9, "pt_9_0", 0, ["889181", "", "0998.119", "", "90", "", "08", ""])],
    )
    result = _audit_result(
        PageContent(page_number=8, texts=[TextBlock(content="现金流量表")], tables=[cash]),
        PageContent(page_number=9, tables=[noise]),
        PageContent(page_number=10, texts=[TextBlock(content="财务报表附注")]),
    )

    projection = derive_audit_report_projection(result)

    assert list(projection.datasets) == ["cash_flow_statement"]
    assert any(warning.startswith("AUDIT_OWNER_EQUITY_UNRESOLVED:page=9") for warning in projection.warnings)


def test_audit_projection_uses_single_page_logical_owner_equity_fallback() -> None:
    headers = [
        "项目",
        "实收资本",
        "优先股",
        "永续债",
        "其他权益工具",
        "资本公积",
        "减：库存股",
        "其他综合收益",
        "专项储备",
        "盈余公积",
        "未分配利润",
        "所有者权益合计",
    ]
    logical = LogicalTable(
        table_id="lt_owner_9",
        logical_id="lt_owner_9",
        headers=headers,
        rows=[
            _row(9, "pt_9_0", 0, ["一、上年年末余额", "10.00", "", "", "", "20.00", "", "", "", "", "30.00", "60.00"]),
            _row(9, "pt_9_0", 1, ["三、本年增减变动金额", "", "", "", "", "", "", "", "", "", "5.00", "5.00"]),
            _row(9, "pt_9_0", 2, ["四、本年期末余额", "10.00", "", "", "", "20.00", "", "", "", "", "35.00", "65.00"]),
        ],
        source_physical_ids=["pt_9_0"],
        source_pages=[9],
        page_span=(9, 9),
        row_count=3,
    )
    result = _audit_result(
        PageContent(page_number=9, texts=[TextBlock(content="所有者（股东）权益变动表")]),
    ).model_copy(update={"logical_tables": [logical]})

    projection = derive_audit_report_projection(result)
    rows = projection.datasets["owners_equity_changes"]

    assert [row["raw"]["item"] for row in rows] == ["一、上年年末余额", "三、本年增减变动金额", "四、本年期末余额"]
    assert rows[-1]["raw"]["total_equity"] == "65.00"
    assert rows[-1]["source"]["table_id"] == "lt_owner_9"


def test_audit_projection_suppresses_unreliable_logical_owner_pseudotable() -> None:
    cash = TableBlock(
        table_id="pt_8_0",
        page=8,
        headers=["项目", "本期金额", "上期金额"],
        rows=[_row(8, "pt_8_0", 0, ["经营活动产生的现金流量净额", "10.00", "9.00"])],
    )
    noise = LogicalTable(
        table_id="lt_noise_9",
        logical_id="lt_noise_9",
        headers=[],
        rows=[_row(9, "pt_9_0", 0, ["889181", "", "0998.119", "", "90", "", "08", ""])],
        source_physical_ids=["pt_9_0"],
        source_pages=[9],
        page_span=(9, 9),
        row_count=1,
    )
    result = _audit_result(
        PageContent(page_number=8, texts=[TextBlock(content="现金流量表")], tables=[cash]),
        PageContent(page_number=9),
        PageContent(page_number=10, texts=[TextBlock(content="财务报表附注")]),
    ).model_copy(update={"logical_tables": [noise]})

    projection = derive_audit_report_projection(result)

    assert list(projection.datasets) == ["cash_flow_statement"]
    assert any(warning.startswith("AUDIT_OWNER_EQUITY_UNRESOLVED:page=9") for warning in projection.warnings)


def test_audit_projection_suppresses_sealed_wide_pseudotable_when_page_metadata_misses_it(monkeypatch) -> None:
    cash = TableBlock(
        table_id="pt_8_0",
        page=8,
        headers=["项目", "本期金额", "上期金额"],
        rows=[_row(8, "pt_8_0", 0, ["经营活动产生的现金流量净额", "10.00", "9.00"])],
    )
    noise = TableBlock(
        table_id="pt_9_0",
        page=9,
        headers=[],
        rows=[
            _row(
                9,
                "pt_9_0",
                row_index,
                ["", f"88918{row_index}", "", "", "6228961820", "988", "29997", "0998.119", *([""] * 9)],
            )
            for row_index in range(7)
        ],
    )
    result = _audit_result(
        PageContent(page_number=8, texts=[TextBlock(content="现金流量表")], tables=[cash]),
        PageContent(page_number=9, tables=[noise]),
    )
    monkeypatch.setattr(
        "docmirror.plugins.audit_report.table_projection.embedded_financial_pages",
        lambda _result: {8},
    )
    monkeypatch.setattr(
        "docmirror.plugins.audit_report.projection.embedded_financial_pages",
        lambda _result: {8},
    )

    projection = derive_audit_report_projection(result)

    assert list(projection.datasets) == ["cash_flow_statement"]
    assert "AUDIT_FINANCIAL_STATEMENT_UNRESOLVED:page=9:kind=owners_equity_changes" in projection.warnings
    assert "AUDIT_OWNER_EQUITY_UNRESOLVED:page=9:width=17:source=sealed_wide_table" in projection.warnings


def test_audit_projection_reports_cross_statement_total_mismatch_without_rewriting_values() -> None:
    balance = TableBlock(
        table_id="pt_5_0",
        page=5,
        headers=["项目", "附注", "期末余额", "期初余额", "项目", "附注", "期末余额", "期初余额"],
        rows=[
            _row(5, "pt_5_0", 0, ["资产总计", "", "100.00", "", "所有者权益合计", "", "55.00", ""]),
            _row(5, "pt_5_0", 1, ["", "", "", "", "负债和所有者权益总计", "", "100.00", ""]),
        ],
    )
    owners = TableBlock(
        table_id="pt_9_0",
        page=9,
        headers=["项目", "本期金额", "上期金额", "所有者权益合计"],
        rows=[_row(9, "pt_9_0", 0, ["四、本年期末余额", "10.00", "20.00", "50.00"])],
    )
    result = _audit_result(
        PageContent(page_number=5, texts=[TextBlock(content="资产负债表")], tables=[balance]),
        PageContent(page_number=9, texts=[TextBlock(content="所有者（股东）权益变动表")], tables=[owners]),
    )

    projection = derive_audit_report_projection(result)

    balance_rows = projection.datasets["balance_sheet"]
    assert len(balance_rows) == 3
    equity = next(row for row in balance_rows if row["raw"]["item"] == "所有者权益合计")
    assert equity["raw"]["current_period_amount"] == "55.00"
    assert equity["source"]["source_region"] == "liabilities_and_equity"
    assert projection.datasets["owners_equity_changes"][0]["raw"]["column_04"] == "50.00"
    mismatch = next(warning for warning in projection.warnings if warning.startswith("AUDIT_STATEMENT_TOTAL_MISMATCH"))
    assert "metric=ending_equity" in mismatch
    assert mismatch in projection.domain_facts["dataset_verification_blockers"]["balance_sheet"]


def test_audit_projection_removes_owner_equity_period_header_from_business_rows() -> None:
    owners = TableBlock(
        table_id="pt_9_0",
        page=9,
        headers=["项目", "实收资本", "资本公积", "所有者权益合计"],
        rows=[
            _row(9, "pt_9_0", 0, ["", "", "本期金额", ""]),
            _row(9, "pt_9_0", 1, ["项目", "实收资本", "资本公积", "所有者权益合计"]),
            _row(9, "pt_9_0", 2, ["四、本期期末余额", "10.00", "20.00", "30.00"]),
        ],
    )
    result = _audit_result(
        PageContent(page_number=9, texts=[TextBlock(content="所有者（股东）权益变动表")], tables=[owners])
    )

    projection = derive_audit_report_projection(result)

    rows = projection.datasets["owners_equity_changes"]
    assert len(rows) == 1
    assert rows[0]["raw"]["item"] == "四、本期期末余额"
    assert any(warning.startswith("AUDIT_OWNER_EQUITY_HEADER_ROWS_REMOVED") for warning in projection.warnings)


def test_audit_opinion_ignores_non_unmodified_phrase_in_auditor_responsibilities() -> None:
    result = _audit_result(
        PageContent(
            page_number=2,
            texts=[
                TextBlock(
                    content=(
                        "一、审计意见\n"
                        "我们认为，财务报表在所有重大方面公允反映了公司的财务状况。\n"
                        "二、形成审计意见的基础"
                    ),
                    evidence_ids=["ev:opinion"],
                )
            ],
        ),
        PageContent(
            page_number=3,
            texts=[
                TextBlock(
                    content=("五、注册会计师对财务报表审计的责任\n如果披露不充分，我们应当发表非无保留意见。"),
                    evidence_ids=["ev:responsibility"],
                )
            ],
        ),
    )

    projection = derive_audit_report_projection(result)

    assert projection.domain_facts["audit_opinion_type"] == "unmodified"
    assert projection.domain_facts["field_details"]["audit_opinion_type"]["page"] == 2
    assert projection.domain_facts["field_details"]["audit_opinion_type"]["evidence_ids"] == ["ev:opinion"]


def test_audit_opinion_keeps_true_qualified_language_inside_opinion_section() -> None:
    result = _audit_result(
        PageContent(
            page_number=2,
            texts=[
                TextBlock(
                    content=(
                        "一、审计意见\n"
                        "除无法取得充分证据事项可能产生的影响外，财务报表公允反映了财务状况。\n"
                        "二、形成保留意见的基础"
                    ),
                    evidence_ids=["ev:qualified"],
                )
            ],
        )
    )

    projection = derive_audit_report_projection(result)

    assert projection.domain_facts["audit_opinion_type"] == "qualified"
    assert projection.domain_facts["field_details"]["audit_opinion_type"]["page"] == 2


def test_audit_metadata_leaves_conflicting_complete_numbers_unresolved() -> None:
    result = _audit_result(
        PageContent(
            page_number=1,
            texts=[
                TextBlock(
                    content="2024年度审计报告\n鼎迈会师审字〔2025〕第0123号\n报告编号:沪23ANP3CE鼎",
                    evidence_ids=["ev:cover"],
                )
            ],
        ),
        PageContent(
            page_number=3,
            texts=[
                TextBlock(
                    content=(
                        "鼎迈会师审字〔2024〕第0554号\n"
                        "一、审计意见\n我们认为财务报表公允反映了财务状况。\n"
                        "二、形成审计意见的基础"
                    ),
                    evidence_ids=["ev:body"],
                )
            ],
        ),
    )

    projection = derive_audit_report_projection(result)
    bundle = AuditReportPlugin().project_bundle(seal_parse_result(result), file_path="source-file.pdf")

    assert "audit_document_number" not in projection.domain_facts
    assert projection.domain_facts["field_details"]["audit_document_number_candidates"]["values"] == [
        "鼎迈会师审字〔2025〕第0123号",
        "鼎迈会师审字〔2024〕第0554号",
    ]
    assert "regulatory_report_id" not in projection.domain_facts
    assert any(warning.startswith("AUDIT_DOCUMENT_NUMBER_CONFLICT") for warning in projection.warnings)
    assert any(warning.startswith("AUDIT_REGULATORY_REPORT_ID_INVALID") for warning in projection.warnings)
    assert bundle is not None
    assert bundle.json_payload()["document"]["title"] == "2024年度审计报告"


def test_audit_metadata_warns_when_cover_number_is_incomplete_and_year_conflicts() -> None:
    result = _audit_result(
        PageContent(page_number=1, texts=[TextBlock(content="2024年度审计报告\n会师审字[2025]第 号")]),
        PageContent(page_number=3, texts=[TextBlock(content="鼎迈会师审字[2024]第0554号")]),
    )

    projection = derive_audit_report_projection(result)

    assert "audit_document_number" not in projection.domain_facts
    candidates = projection.domain_facts["field_details"]["audit_document_number_candidates"]
    assert candidates["values"] == ["鼎迈会师审字[2024]第0554号"]
    assert candidates["incomplete_values"] == ["会师审字[2025]第号"]
    assert any(warning.startswith("AUDIT_DOCUMENT_NUMBER_INCOMPLETE") for warning in projection.warnings)
    assert any(warning.startswith("AUDIT_DOCUMENT_NUMBER_CONFLICT") for warning in projection.warnings)


def test_audit_projection_normalizes_supported_cjk_radical_variants() -> None:
    table = TableBlock(
        table_id="pt_20_0",
        page=20,
        headers=["项目", "金额"],
        rows=[_row(20, "pt_20_0", 0, ["江⻄⻋船税", "10.00"])],
    )
    result = _audit_result(PageContent(page_number=20, texts=[TextBlock(content="财务报表附注")], tables=[table]))

    projection = derive_audit_report_projection(result)
    row = next(iter(projection.datasets.values()))[0]

    assert row["raw"]["项目"] == "江⻄⻋船税"
    assert row["normalized"]["项目"] == "江西车船税"
    assert "AUDIT_NORMALIZED_GLYPH_VARIANT_REMAINS" not in projection.warnings


def test_audit_sections_ignore_numbered_financial_items_inside_table_bbox() -> None:
    table = TableBlock(
        table_id="pt_12_0",
        page=12,
        headers=["项目", "金额"],
        rows=[_row(12, "pt_12_0", 0, ["一、营业收入", "10.00"])],
        bbox=[50.0, 80.0, 500.0, 500.0],
    )
    result = _audit_result(
        PageContent(
            page_number=12,
            texts=[
                TextBlock(content="一、营业收入", bbox=[100.0, 100.0, 200.0, 120.0]),
                TextBlock(content="一、公司基本情况", bbox=[100.0, 600.0, 260.0, 620.0]),
            ],
            tables=[table],
        )
    )

    projection = derive_audit_report_projection(result)
    titles = {section["title"] for section in projection.sections}

    assert "一、营业收入" not in titles
    assert "一、公司基本情况" in titles


def test_audit_statement_candidate_scoring_prefers_complete_positioned_text() -> None:
    def text(value: str, bbox: list[float], evidence: str) -> TextBlock:
        return TextBlock(content=value, bbox=bbox, confidence=0.98, evidence_ids=[evidence])

    bad_table = TableBlock(
        table_id="pt_5_0",
        page=5,
        headers=["项目", "附注", "期末余额", "期初余额"],
        rows=[_row(5, "pt_5_0", 0, ["乱码", "", "", ""])],
    )
    result = _audit_result(
        PageContent(
            page_number=5,
            texts=[
                text("资产负债表", [240.0, 40.0, 340.0, 55.0], "ev:title"),
                text("项目", [100.0, 80.0, 160.0, 90.0], "ev:header:item"),
                text("附注", [250.0, 80.0, 290.0, 90.0], "ev:header:note"),
                text("2024年12月31日", [350.0, 80.0, 410.0, 90.0], "ev:header:current"),
                text("2023年12月31日", [450.0, 80.0, 510.0, 90.0], "ev:header:previous"),
                text("货币资金", [100.0, 120.0, 170.0, 130.0], "ev:row:1:item"),
                text("五、1", [250.0, 120.0, 290.0, 130.0], "ev:row:1:note"),
                text("1,200.00", [350.0, 120.0, 410.0, 130.0], "ev:row:1:current"),
                text("900.00", [450.0, 120.0, 510.0, 130.0], "ev:row:1:previous"),
                text("资产总计", [100.0, 140.0, 170.0, 150.0], "ev:row:2:item"),
                text("1,200.00", [350.0, 140.0, 410.0, 150.0], "ev:row:2:current"),
                text("900.00", [450.0, 140.0, 510.0, 150.0], "ev:row:2:previous"),
                text("应收账款", [100.0, 160.0, 170.0, 170.0], "ev:row:3:item"),
                text("五、2", [250.0, 160.0, 290.0, 170.0], "ev:row:3:note"),
                text("200.00", [350.0, 160.0, 410.0, 170.0], "ev:row:3:current"),
                text("100.00", [450.0, 160.0, 510.0, 170.0], "ev:row:3:previous"),
            ],
            tables=[bad_table],
        )
    )

    projection = derive_audit_report_projection(result)
    rows = projection.datasets["balance_sheet"]

    assert [row["raw"]["item"] for row in rows] == ["货币资金", "资产总计", "应收账款"]
    assert all(row["source"]["recovery"] == "positioned_text_rows" for row in rows)
    assert any(warning.startswith("AUDIT_STATEMENT_TEXT_ROWS_SELECTED") for warning in projection.warnings)


def test_audit_owner_equity_uses_named_columns_and_removes_stacked_headers() -> None:
    headers = [
        "项目",
        "实收资本",
        "优先股",
        "永续债",
        "其他权益工具",
        "资本公积",
        "减：库存股",
        "其他综合收益",
        "专项储备",
        "盈余公积",
        "未分配利润",
        "所有者权益合计",
    ]
    table = TableBlock(
        table_id="pt_9_0",
        page=9,
        headers=headers,
        rows=[
            _row(9, "pt_9_0", 0, ["", "本期金额", "", "", "", "", "", "", "", "", "", ""]),
            _row(9, "pt_9_0", 1, ["", *headers[1:]]),
            _row(
                9,
                "pt_9_0",
                2,
                ["四、本期期末余额", "10.00", "", "", "", "20.00", "", "", "", "", "30.00", "60.00"],
            ),
        ],
    )
    result = _audit_result(
        PageContent(page_number=9, texts=[TextBlock(content="所有者（股东）权益变动表")], tables=[table])
    )

    projection = derive_audit_report_projection(result)
    rows = projection.datasets["owners_equity_changes"]

    assert len(rows) == 1
    assert rows[0]["raw"]["period_role"] == "current"
    assert rows[0]["raw"]["paid_in_capital"] == "10.00"
    assert rows[0]["raw"]["capital_reserve"] == "20.00"
    assert rows[0]["raw"]["retained_earnings"] == "30.00"
    assert rows[0]["raw"]["total_equity"] == "60.00"
    assert not any(key.startswith("column_") for key in rows[0]["raw"])
    assert any(warning.startswith("AUDIT_OWNER_EQUITY_HEADER_ROWS_REMOVED") for warning in projection.warnings)
