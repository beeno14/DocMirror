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
from docmirror.plugins.audit_report.table_projection import (
    audit_data_dictionary,
    canonicalize_audit_dataset_columns,
    dataset_blocking_warnings,
    merge_cross_page_continuations,
    normalize_audit_label,
    normalize_audit_record,
    normalize_audit_value,
    quality_warnings,
    record_keys,
    repair_stacked_note_headers,
    resolve_note_table_candidates,
    synchronize_audit_record_sources,
)
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


def _repair_events(projection: object) -> list[str]:
    domain_facts = getattr(projection, "domain_facts")
    return list(domain_facts["audit_projection"]["repair_events"])


def _projected_record(
    dataset: str,
    ordinal: int,
    raw: dict[str, str],
    *,
    table_id: str,
    physical_table_id: str,
    page: int,
) -> dict[str, object]:
    return {
        "record_id": f"{dataset}:r{ordinal:06d}",
        "raw": dict(raw),
        "canonical_raw": dict(raw),
        "normalized": dict(raw),
        "source": {
            "table_id": table_id,
            "physical_table_id": physical_table_id,
            "table_row_index": ordinal - 1,
            "source_row_index": ordinal - 1,
            "page": page,
            "page_range": [page, page],
        },
    }


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
    assert projection.semantic["dataset_section_ids"]["balance_sheet"] == "section_balance_sheet"

    projector = PluginRegistry().get_projector("audit_report", "community")
    assert isinstance(projector, AuditReportPlugin)
    bundle = projector.project_bundle(seal_parse_result(result), file_path="audit.pdf")
    assert bundle is not None
    dataset = next(item for item in bundle.json_payload()["datasets"] if item["name"] == "balance_sheet")
    assert dataset["completeness"]["verified"] is True
    assert dataset["completeness"]["basis"] == "source_statement_rows"
    assert dataset["status"] == "complete"


def test_audit_projection_verifies_source_backed_note_dataset_rows() -> None:
    table = TableBlock(
        table_id="pt_12_0",
        page=12,
        headers=["项目", "金额"],
        rows=[
            _row(12, "pt_12_0", 0, ["库存现金", "1,200.00"]),
            _row(12, "pt_12_0", 1, ["银行存款", "9,800.00"]),
        ],
        evidence_ids=["ev:table:cash"],
    )
    result = _audit_result(
        PageContent(
            page_number=12,
            texts=[TextBlock(content="财务报表附注\n1、货币资金")],
            tables=[table],
        )
    )

    bundle = AuditReportPlugin().project_bundle(seal_parse_result(result), file_path="audit.pdf")

    assert bundle is not None
    dataset = bundle.json_payload()["datasets"][0]
    assert dataset["row_count"] == 2
    assert dataset["completeness"] == {
        "expected_row_count": 2,
        "emitted_row_count": 2,
        "omitted_row_count": 0,
        "verified": True,
        "basis": "source_table_rows",
    }
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
    assert any(warning.startswith("AUDIT_STATEMENT_TEXT_ROWS_RECOVERED") for warning in _repair_events(projection))


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
    assert [row["source"]["source_row_index"] for row in rows[1:3]] == [0, 1]
    assert [row["source"]["source_row_index"] for row in rows[3:]] == [0, 1]
    assert rows[2]["canonical_raw"] == rows[2]["raw"]
    assert {ref["field_name"] for ref in rows[1]["source"]["source_cell_refs"]} == set(rows[1]["raw"])
    assert sum("mode=data_header" in warning for warning in _repair_events(projection)) == 2


def test_audit_projection_names_cross_page_note_table_from_source_caption() -> None:
    first = TableBlock(
        table_id="pt_40_4",
        page=40,
        headers=["项目", "期末余额", "期初余额"],
        rows=[
            _row(40, "pt_40_4", 0, ["关联方款项", "20,081,697.07", "6,391,299.27"]),
            _row(40, "pt_40_4", 1, ["保证金备用金", "622,025.91", "650,832.36"]),
        ],
        bbox=[88.0, 690.0, 533.0, 758.0],
        evidence_ids=["ev:table:40"],
    )
    continuation = TableBlock(
        table_id="pt_41_0",
        page=41,
        headers=["其他往来单位", "9,920,895.22", "608,519.54"],
        rows=[
            _row(41, "pt_41_0", 0, ["代垫款", "32,048.70", "146,237.18"]),
            _row(41, "pt_41_0", 1, ["借款", "10,907,734.71", "17,723,899.77"]),
            _row(41, "pt_41_0", 2, ["代扣代缴款项", "71,603.55", "3,546.07"]),
            _row(41, "pt_41_0", 3, ["合计", "41,636,005.16", "25,524,334.19"]),
        ],
        bbox=[88.0, 72.0, 533.0, 186.0],
        evidence_ids=["ev:table:41"],
    )
    result = _audit_result(
        PageContent(
            page_number=40,
            texts=[
                TextBlock(content="18、其他应付款", bbox=[114.0, 484.0, 196.0, 501.0]),
                TextBlock(content="总体情况列示", bbox=[111.0, 507.0, 175.0, 522.0]),
                TextBlock(content="其他应付款", bbox=[111.0, 648.0, 164.0, 663.0]),
                TextBlock(content="（1）按款项性质列示其他应付款", bbox=[111.0, 668.0, 262.0, 684.0]),
            ],
            tables=[first],
        ),
        PageContent(page_number=41, texts=[TextBlock(content="财务报表附注")], tables=[continuation]),
    )

    projection = derive_audit_report_projection(result)
    bundle = AuditReportPlugin().project_bundle(seal_parse_result(result), file_path="audit.pdf")

    assert list(projection.datasets) == ["other_payables"]
    rows = projection.datasets["other_payables"]
    assert len(rows) == 7
    assert [row["record_id"] for row in rows] == [f"other_payables:r{index:06d}" for index in range(1, 8)]
    assert {row["source"]["page"] for row in rows} == {40, 41}
    assert projection.semantic["dataset_labels"]["other_payables"] == ("（1）按款项性质列示其他应付款")
    assert bundle is not None
    dataset = bundle.json_payload()["datasets"][0]
    assert dataset["id"] == "ds_other_payables"
    assert dataset["name"] == "other_payables"
    assert dataset["label"] == "（1）按款项性质列示其他应付款"
    assert dataset["csv"] == "001_datasets/other_payables.csv"
    assert dataset["status"] == "complete"


def test_audit_projection_numbers_distinct_tables_with_the_same_subject() -> None:
    summary = TableBlock(
        table_id="pt_40_0",
        page=40,
        headers=["项目", "期末余额", "期初余额"],
        rows=[_row(40, "pt_40_0", 0, ["其他应付款", "100.00", "80.00"])],
        bbox=[88.0, 540.0, 533.0, 620.0],
    )
    by_nature = TableBlock(
        table_id="pt_42_0",
        page=42,
        headers=["款项性质", "期末余额", "期初余额"],
        rows=[_row(42, "pt_42_0", 0, ["保证金", "60.00", "50.00"])],
        bbox=[88.0, 120.0, 533.0, 200.0],
    )
    result = _audit_result(
        PageContent(
            page_number=40,
            texts=[
                TextBlock(content="18、其他应付款", bbox=[114.0, 480.0, 196.0, 500.0]),
                TextBlock(content="总体情况列示", bbox=[114.0, 510.0, 196.0, 530.0]),
            ],
            tables=[summary],
        ),
        PageContent(
            page_number=42,
            texts=[TextBlock(content="（1）按款项性质列示其他应付款", bbox=[114.0, 80.0, 300.0, 100.0])],
            tables=[by_nature],
        ),
    )

    projection = derive_audit_report_projection(result)
    bundle = AuditReportPlugin().project_bundle(seal_parse_result(result), file_path="audit.pdf")

    assert list(projection.datasets) == ["other_payables_01", "other_payables_02"]
    assert projection.datasets["other_payables_01"][0]["record_id"] == "other_payables_01:r000001"
    assert projection.datasets["other_payables_02"][0]["record_id"] == "other_payables_02:r000001"
    assert projection.semantic["dataset_labels"] == {
        "other_payables_01": "总体情况列示",
        "other_payables_02": "（1）按款项性质列示其他应付款",
    }
    assert bundle is not None
    assert [dataset["csv"] for dataset in bundle.json_payload()["datasets"]] == [
        "001_datasets/other_payables_01.csv",
        "001_datasets/other_payables_02.csv",
    ]


def test_audit_projection_rejects_a_conflicting_preceding_subject_label() -> None:
    table = TableBlock(
        table_id="pt_31_0",
        page=31,
        headers=["项目", "可抵扣暂时性差异", "递延所得税资产"],
        rows=[_row(31, "pt_31_0", 0, ["资产减值准备", "100.00", "25.00"])],
        bbox=[88.0, 140.0, 533.0, 220.0],
    )
    result = _audit_result(
        PageContent(
            page_number=31,
            texts=[TextBlock(content="10、长期待摊费用", bbox=[114.0, 80.0, 240.0, 100.0])],
            tables=[table],
        )
    )

    projection = derive_audit_report_projection(result)

    assert list(projection.datasets) == ["deferred_tax"]
    assert projection.semantic["dataset_labels"]["deferred_tax"] == "递延所得税资产"


def test_audit_projection_keeps_unresolved_generic_table_name_and_warns() -> None:
    table = TableBlock(
        table_id="pt_20_0",
        page=20,
        headers=["项目", "金额"],
        rows=[_row(20, "pt_20_0", 0, ["无法确定用途", "10.00"])],
        bbox=[88.0, 120.0, 533.0, 180.0],
    )
    result = _audit_result(
        PageContent(
            page_number=20,
            texts=[TextBlock(content="财务报表附注", bbox=[90.0, 40.0, 530.0, 57.0])],
            tables=[table],
        )
    )

    projection = derive_audit_report_projection(result)

    assert list(projection.datasets) == ["records"]
    assert any(warning.startswith("AUDIT_TABLE_TITLE_UNRESOLVED") for warning in projection.warnings)


def test_audit_projection_merges_repeated_header_continuation_before_naming() -> None:
    first = TableBlock(
        table_id="pt_42_4",
        page=42,
        headers=["项目", "本期发生额", "上期发生额"],
        rows=[_row(42, "pt_42_4", 0, ["运输费", "100.00", "80.00"])],
        bbox=[84.0, 696.0, 537.0, 755.0],
    )
    continuation = TableBlock(
        table_id="pt_43_0",
        page=43,
        headers=["项目", "本期发生额", "上期发生额"],
        rows=[_row(43, "pt_43_0", 0, ["广告费", "50.00", "40.00"])],
        bbox=[84.0, 72.0, 537.0, 271.0],
    )
    result = _audit_result(
        PageContent(
            page_number=42,
            texts=[TextBlock(content="28、销售费用", bbox=[114.0, 672.0, 184.0, 689.0])],
            tables=[first],
        ),
        PageContent(
            page_number=43,
            texts=[
                TextBlock(content="财务报表附注", bbox=[90.0, 41.0, 530.0, 57.0]),
                TextBlock(content="29、管理费用", bbox=[114.0, 277.0, 184.0, 294.0]),
            ],
            tables=[continuation],
        ),
    )

    projection = derive_audit_report_projection(result)

    assert list(projection.datasets) == ["selling_expenses"]
    assert len(projection.datasets["selling_expenses"]) == 2
    assert projection.semantic["dataset_labels"]["selling_expenses"] == "28、销售费用"
    assert any("mode=repeated_header" in warning for warning in _repair_events(projection))


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
        "category": "其他应收款",
        "item": "备用金",
        "book_balance": "20.00",
        "loss_allowance": "1.00",
        "carrying_amount": "19.00",
    }
    assert rows[2]["source"]["page"] == 47
    category_ref = next(ref for ref in rows[1]["source"]["source_cell_refs"] if ref["field_name"] == "category")
    assert category_ref["evidence_ids"] == ["ev:category:47"]
    assert rows[1]["source"]["source_cell_refs"][0] == category_ref
    assert any("mode=missing_outer" in warning for warning in _repair_events(projection))


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
    assert any(warning.startswith("AUDIT_NOTE_HEADER_ROWS_PROMOTED") for warning in _repair_events(projection))


def test_audit_projection_promotes_semantic_two_column_note_header() -> None:
    table = TableBlock(
        table_id="pt_19_0",
        page=19,
        headers=["col_0", "col_1"],
        rows=[
            _row(19, "pt_19_0", 0, ["项目", "确定组合的依据"]),
            _row(19, "pt_19_0", 1, ["银行承兑汇票", "承兑人为信用风险较小的银行"]),
            _row(19, "pt_19_0", 2, ["商业承兑汇票", "根据承兑人的信用风险划分"]),
        ],
    )
    result = _audit_result(PageContent(page_number=19, texts=[TextBlock(content="财务报表附注")], tables=[table]))

    projection = derive_audit_report_projection(result)
    rows = next(iter(projection.datasets.values()))

    assert len(rows) == 2
    assert rows[0]["raw"] == {
        "item": "银行承兑汇票",
        "portfolio_basis": "承兑人为信用风险较小的银行",
    }
    assert rows[0]["normalized"] == {
        "item": "银行承兑汇票",
        "portfolio_basis": "承兑人为信用风险较小的银行",
    }
    assert any(warning.startswith("AUDIT_NOTE_HEADER_ROWS_PROMOTED") for warning in _repair_events(projection))
    assert not any(warning.startswith("precision:generic_header_repaired") for warning in projection.warnings)


def test_audit_output_normalization_preserves_raw_and_uses_english_evidence_backed_columns() -> None:
    table = TableBlock(
        table_id="pt_20_0",
        page=20,
        headers=["单位名称", "期末余额"],
        rows=[_row(20, "pt_20_0", 0, ["江西斯菲尔物流有限公 司", "1,200.00"])],
    )
    result = _audit_result(
        PageContent(
            page_number=20,
            texts=[TextBlock(content="4、其他应收款", bbox=[100.0, 80.0, 240.0, 100.0])],
            tables=[table],
        )
    )

    bundle = AuditReportPlugin().project_bundle(seal_parse_result(result), file_path="audit.pdf")

    assert bundle is not None
    dataset = bundle.datasets[0]
    assert [column["key"] for column in dataset.public["columns"]] == ["entity_name", "ending_balance"]
    assert [column["label"] for column in dataset.public["columns"]] == ["单位名称", "期末余额"]
    assert all(column["evidence_available"] for column in dataset.public["columns"])
    assert dataset.rows[0]["raw"]["entity_name"] == "江西斯菲尔物流有限公 司"
    assert dataset.rows[0]["normalized"]["entity_name"] == "江西斯菲尔物流有限公司"
    assert dataset.rows[0]["source"]["source_cell_refs"][0]["field_name"] == "entity_name"
    assert normalize_audit_label("2应收账款") == "2、应收账款"
    assert normalize_audit_label("项⽬") == "项目"
    assert normalize_audit_value("ABC Logistics Ltd.") == "ABC Logistics Ltd."


def test_audit_note_candidate_resolution_prefers_complete_physical_dataset() -> None:
    physical = TableBlock(
        table_id="pt_39_0",
        page=39,
        headers=["项目", "期初余额", "期末余额"],
        rows=[
            _row(39, "pt_39_0", 0, ["装修费", "100.00", "80.00"]),
            _row(39, "pt_39_0", 1, ["合计", "100.00", "80.00"]),
        ],
        bbox=[80.0, 100.0, 530.0, 180.0],
    )
    result = _audit_result(PageContent(page_number=39, tables=[physical]))
    logical_rows = [
        _projected_record(
            "table_001",
            ordinal,
            dict(zip(physical.headers, values, strict=True)),
            table_id="lt_18",
            physical_table_id="pt_39_9",
            page=39,
        )
        for ordinal, values in enumerate((["装修费", "100.00", "80.00"], ["合计", "100.00", "80.00"]), start=1)
    ]
    physical_rows = [
        _projected_record(
            "table_002",
            ordinal,
            dict(zip(physical.headers, values, strict=True)),
            table_id="pt_39_0",
            physical_table_id="pt_39_0",
            page=39,
        )
        for ordinal, values in enumerate((["装修费", "100.00", "80.00"], ["合计", "100.00", "80.00"]), start=1)
    ]
    datasets = {"table_001": logical_rows, "table_002": physical_rows}

    warnings = resolve_note_table_candidates(datasets, result)

    assert list(datasets) == ["table_002"]
    assert datasets["table_002"][0]["source"]["physical_table_id"] == "pt_39_0"
    assert any(warning.startswith("AUDIT_NOTE_DUPLICATE_DATASET_REMOVED") for warning in warnings)


def test_audit_note_candidate_resolution_splits_mixed_subjects_by_physical_table() -> None:
    current_assets = TableBlock(
        table_id="pt_35_1",
        page=35,
        headers=["项目", "期末余额", "期初余额"],
        rows=[
            _row(35, "pt_35_1", 0, ["待抵扣进项税额", "290.69", "290.69"]),
            _row(35, "pt_35_1", 1, ["合计", "290.69", "290.69"]),
        ],
        bbox=[80.0, 180.0, 530.0, 260.0],
    )
    equity = TableBlock(
        table_id="pt_35_2",
        page=35,
        headers=["项目", "期末余额", "期初余额"],
        rows=[
            _row(35, "pt_35_2", 0, ["对子公司投资", "3,361,800.00", "3,361,800.00"]),
            _row(35, "pt_35_2", 1, ["对联营、合营企业投资", "", ""]),
            _row(35, "pt_35_2", 2, ["合计", "3,361,800.00", "3,361,800.00"]),
        ],
        bbox=[80.0, 320.0, 530.0, 430.0],
    )
    result = _audit_result(
        PageContent(
            page_number=35,
            texts=[
                TextBlock(content="5、其他流动资产", bbox=[100.0, 150.0, 250.0, 170.0]),
                TextBlock(content="（1）长期股权投资分类", bbox=[100.0, 290.0, 300.0, 310.0]),
            ],
            tables=[current_assets, equity],
        )
    )
    values = [
        ["待抵扣进项税额", "290.69", "290.69"],
        ["合计", "290.69", "290.69"],
        ["对子公司投资", "3,361,800.00", "3,361,800.00"],
        ["对联营、合营企业投资", "", ""],
        ["合计", "3,361,800.00", "3,361,800.00"],
    ]
    datasets = {
        "table_001": [
            _projected_record(
                "table_001",
                ordinal,
                dict(zip(current_assets.headers, row, strict=True)),
                table_id="lt_14",
                physical_table_id="pt_35_0",
                page=35,
            )
            for ordinal, row in enumerate(values, start=1)
        ]
    }

    warnings = resolve_note_table_candidates(datasets, result)

    assert list(datasets) == ["table_001__split_01", "table_001__split_02"]
    assert len(datasets["table_001__split_01"]) == 2
    assert len(datasets["table_001__split_02"]) == 3
    assert datasets["table_001__split_01"][0]["source"]["physical_table_id"] == "pt_35_1"
    assert datasets["table_001__split_02"][0]["source"]["physical_table_id"] == "pt_35_2"
    assert sum(warning.startswith("AUDIT_NOTE_MIXED_TABLE_SPLIT") for warning in warnings) == 2


def test_audit_note_candidate_resolution_recovers_data_promoted_to_physical_header() -> None:
    table = TableBlock(
        table_id="pt_32_2",
        page=32,
        headers=["单项计提坏账准备的应收账款", "", "", ""],
        rows=[
            _row(32, "pt_32_2", 4, ["按组合计提坏账准备的应收账款", "54,656,458.32", "100", ""]),
            _row(32, "pt_32_2", 5, ["合计", "54,656,458.32", "100", ""]),
        ],
        bbox=[80.0, 100.0, 530.0, 260.0],
        evidence_ids=["ev:pt_32_2"],
    )
    result = _audit_result(PageContent(page_number=32, tables=[table]))
    rows = [
        _projected_record(
            "table_001",
            1,
            {"类别": "单项计提坏账准备的应收账款", "金额": "", "比例": "", "账面价值": ""},
            table_id="lt_4:segment_2",
            physical_table_id="pt_32_0",
            page=32,
        ),
        _projected_record(
            "table_001",
            2,
            {"类别": "按组合计提坏账准备的应收账款", "金额": "54,656,458.32", "比例": "100", "账面价值": ""},
            table_id="lt_4:segment_2",
            physical_table_id="pt_32_0",
            page=32,
        ),
        _projected_record(
            "table_001",
            3,
            {"类别": "合计", "金额": "54,656,458.32", "比例": "100", "账面价值": ""},
            table_id="lt_4:segment_2",
            physical_table_id="pt_32_0",
            page=32,
        ),
    ]
    datasets = {"table_001": rows}

    warnings = resolve_note_table_candidates(datasets, result)

    recovered = datasets["table_001"][0]["source"]
    assert recovered["physical_table_id"] == "pt_32_2"
    assert [row["source"]["source_row_index"] for row in datasets["table_001"]] == [0, 1, 2]
    assert recovered["physical_source_row_index"] == -1
    assert recovered["source_resolution"] == "matched_physical_table_header"
    assert not any(warning.startswith("AUDIT_NOTE_PHYSICAL_SOURCE_UNRESOLVED") for warning in warnings)


def test_audit_note_candidate_resolution_uses_stacked_period_headers_to_break_sparse_row_tie() -> None:
    ending = TableBlock(
        table_id="pt_32_1",
        page=32,
        headers=[],
        rows=[
            _row(32, "pt_32_1", 0, ["类别", "期末余额"]),
            _row(32, "pt_32_1", 1, ["", "账面余额"]),
            _row(32, "pt_32_1", 2, ["", "金额"]),
            _row(32, "pt_32_1", 3, ["单项计提坏账准备的应收账款", ""]),
        ],
    )
    opening = TableBlock(
        table_id="pt_32_2",
        page=32,
        headers=[],
        rows=[
            _row(32, "pt_32_2", 0, ["类别", "期初余额"]),
            _row(32, "pt_32_2", 1, ["", "账面余额"]),
            _row(32, "pt_32_2", 2, ["", "金额"]),
            _row(32, "pt_32_2", 3, ["单项计提坏账准备的应收账款", ""]),
        ],
    )
    result = _audit_result(PageContent(page_number=32, tables=[ending, opening]))
    datasets = {
        "table_001": [
            _projected_record(
                "table_001",
                1,
                {"类别": "单项计提坏账准备的应收账款", "期初余额/账面余额/金额": ""},
                table_id="lt_4:segment_2",
                physical_table_id="pt_32_0",
                page=32,
            )
        ]
    }

    warnings = resolve_note_table_candidates(datasets, result)

    recovered = datasets["table_001"][0]["source"]
    assert recovered["physical_table_id"] == "pt_32_2"
    assert recovered["source_row_index"] == 3
    assert recovered["source_resolution"] == "matched_physical_row"
    assert not any(warning.startswith("AUDIT_NOTE_PHYSICAL_SOURCE_UNRESOLVED") for warning in warnings)


def test_audit_projection_merges_horizontal_note_continuations_by_business_anchor() -> None:
    left = TableBlock(
        table_id="pt_35_0",
        page=35,
        headers=["被投资单位", "期初余额", "追加投资", "减少投资"],
        rows=[
            _row(35, "pt_35_0", 0, ["测试子公司", "100.00", "", ""]),
            _row(35, "pt_35_0", 1, ["合计", "100.00", "", ""]),
        ],
        bbox=[80.0, 130.0, 530.0, 220.0],
    )
    right = TableBlock(
        table_id="pt_35_1",
        page=35,
        headers=["被投资单位", "宣告发放现金股利或利润", "期末余额", "减值准备期末余额"],
        rows=[
            _row(35, "pt_35_1", 0, ["测试子公司", "", "100.00", ""]),
            _row(35, "pt_35_1", 1, ["合计", "", "100.00", ""]),
        ],
        bbox=[80.0, 260.0, 530.0, 350.0],
    )
    result = _audit_result(
        PageContent(
            page_number=35,
            texts=[
                TextBlock(content="6、长期股权投资", bbox=[100.0, 70.0, 260.0, 90.0]),
                TextBlock(content="（2）对子公司投资", bbox=[100.0, 100.0, 260.0, 120.0]),
                TextBlock(content="（续）", bbox=[100.0, 230.0, 180.0, 250.0]),
            ],
            tables=[left, right],
        )
    )

    projection = derive_audit_report_projection(result)

    assert list(projection.datasets) == ["long_term_equity_investments"]
    rows = projection.datasets["long_term_equity_investments"]
    assert len(rows) == 2
    assert rows[0]["raw"]["opening_balance"] == "100.00"
    assert rows[0]["raw"]["ending_balance"] == "100.00"
    assert rows[0]["source"]["physical_table_ids"] == ["pt_35_0", "pt_35_1"]
    assert any(warning.startswith("AUDIT_NOTE_HORIZONTAL_TABLE_MERGED") for warning in _repair_events(projection))


def test_audit_projection_recovers_stable_stacked_note_schemas() -> None:
    datasets = {
        "table_001": [
            _projected_record(
                "table_001",
                1,
                {"col_0": "账龄组合", "col_1": "100.00", "坏账准备": "5.00", "计提比例(%)": "5.00"},
                table_id="pt_32_3",
                physical_table_id="pt_32_3",
                page=32,
            )
        ],
        "table_002": [
            _projected_record(
                "table_002",
                1,
                {
                    "col_0": "其他应收款",
                    "col_1": "测试关联方",
                    "账面余额": "100.00",
                    "坏账准备": "",
                    "账面余额_2": "80.00",
                    "坏账准备_2": "",
                },
                table_id="pt_46_0",
                physical_table_id="pt_46_0",
                page=46,
            )
        ],
        "table_003": [
            _projected_record(
                "table_003",
                1,
                {
                    "类别": "按组合计提坏账准备的应收账款",
                    "期初余额/账面余额/金额": "54,656,458.32",
                    "比例 (%)": "100.00",
                    "期初余额/坏账准备/金额": "3,541,438.79",
                    "计提比例(%)": "6.48",
                    "期初余额/账面价值": "51,115,019.53",
                },
                table_id="pt_32_2",
                physical_table_id="pt_32_2",
                page=32,
            )
        ],
    }

    warnings = repair_stacked_note_headers(datasets)

    assert list(datasets["table_001"][0]["raw"]) == [
        "名称",
        "期末余额/应收账款",
        "期末余额/坏账准备",
        "期末余额/计提比例(%)",
    ]
    assert list(datasets["table_002"][0]["raw"]) == [
        "项目",
        "关联方名称",
        "期末余额/账面余额",
        "期末余额/坏账准备",
        "期初余额/账面余额",
        "期初余额/坏账准备",
    ]
    assert list(datasets["table_003"][0]["raw"]) == [
        "类别",
        "期初余额/账面余额/金额",
        "期初余额/账面余额/比例(%)",
        "期初余额/坏账准备/金额",
        "期初余额/坏账准备/计提比例(%)",
        "期初余额/账面价值",
    ]
    assert len(warnings) == 3


def test_audit_note_schema_uses_all_evidence_backed_fields_when_first_canonical_row_is_sparse() -> None:
    source_keys = [
        "类别",
        "期末余额/账面余额/金额",
        "期末余额/账面余额/比例(%)",
        "期末余额/坏账准备/金额",
        "期末余额/坏账准备/计提比例(%)",
        "期末余额/账面价值",
    ]
    rows = [
        {
            "record_id": "table_001:r000001",
            "raw": dict.fromkeys(source_keys, "") | {"类别": "单项计提坏账准备的应收账款"},
            "canonical_raw": {"类别": "单项计提坏账准备的应收账款"},
            "normalized": dict.fromkeys(source_keys, "") | {"类别": "单项计提坏账准备的应收账款"},
            "source": {
                "page": 32,
                "physical_table_id": "pt_32_1",
                "source_cell_refs": [
                    {"page": 32, "table_id": "pt_32_1", "row": 3, "col": col, "field_name": key}
                    for col, key in enumerate(source_keys)
                ],
            },
        },
        {
            "record_id": "table_001:r000002",
            "raw": dict(
                zip(
                    source_keys,
                    ["按组合计提坏账准备的应收账款", "58,251,600.81", "100", "4,485,273.71", "7.70", "53,766,327.10"],
                    strict=True,
                )
            ),
            "canonical_raw": {"类别": "按组合计提坏账准备的应收账款"},
            "normalized": dict(
                zip(
                    source_keys,
                    ["按组合计提坏账准备的应收账款", "58,251,600.81", "100", "4,485,273.71", "7.70", "53,766,327.10"],
                    strict=True,
                )
            ),
            "source": {
                "page": 32,
                "physical_table_id": "pt_32_1",
                "source_cell_refs": [
                    {"page": 32, "table_id": "pt_32_1", "row": 4, "col": col, "field_name": key}
                    for col, key in enumerate(source_keys)
                ],
            },
        },
    ]
    datasets = {"table_001": rows}

    assert record_keys(rows) == source_keys

    schemas, warnings = canonicalize_audit_dataset_columns(datasets)

    expected_keys = [
        "category",
        "ending_balance_book_balance_amount",
        "ending_balance_book_balance_ratio",
        "ending_balance_loss_allowance_amount",
        "ending_balance_loss_allowance_provision_ratio",
        "ending_balance_carrying_amount",
    ]
    assert not warnings
    assert list(schemas["table_001"]) == expected_keys
    assert schemas["table_001"]["category"]["type"] == "string"
    assert schemas["table_001"]["ending_balance_book_balance_amount"]["type"] == "decimal"
    assert list(datasets["table_001"][0]["raw"]) == expected_keys
    assert datasets["table_001"][1]["raw"]["ending_balance_carrying_amount"] == "53,766,327.10"
    assert [ref["field_name"] for ref in datasets["table_001"][0]["source"]["source_cell_refs"]] == expected_keys


def test_audit_note_schema_keeps_multi_value_numeric_ranges_as_source_strings() -> None:
    datasets = {
        "taxes": [
            {
                "record_id": "taxes:r000001",
                "raw": {"税种": "增值税", "税率(%)": "9、6"},
                "canonical_raw": {"税种": "增值税", "税率(%)": "9、6"},
                "normalized": {"税种": "增值税", "税率(%)": "9、6"},
                "source": {
                    "page": 31,
                    "evidence_ids": ["ev:source"],
                    "source_cell_refs": [
                        {"field_name": "税种", "page": 31},
                        {"field_name": "税率(%)", "page": 31},
                    ],
                },
            }
        ]
    }

    schemas, warnings = canonicalize_audit_dataset_columns(datasets)
    normalized = normalize_audit_record(datasets["taxes"][0])

    assert not warnings
    assert schemas["taxes"]["tax_rate"]["type"] == "string"
    assert normalized["normalized"]["tax_rate"] == "9、6"


def test_audit_note_quality_blocks_missing_physical_columns_and_nonempty_cells() -> None:
    table = TableBlock(
        table_id="pt_32_1",
        page=32,
        headers=[],
        rows=[
            _row(32, "pt_32_1", 0, ["类别", "期末余额", "", "", "", ""]),
            _row(32, "pt_32_1", 1, ["", "账面余额", "", "坏账准备", "", "账面价值"]),
            _row(32, "pt_32_1", 2, ["", "金额", "比例（%）", "金额", "计提比例（%）", ""]),
            _row(
                32,
                "pt_32_1",
                3,
                ["按组合计提坏账准备的应收账款", "58,251,600.81", "100", "4,485,273.71", "7.70", "53,766,327.10"],
            ),
        ],
    )
    field_names = ["category", "book_balance", "ratio", "loss_allowance", "provision_ratio", "carrying_amount"]
    rows = [
        {
            "record_id": "accounts_receivable:r000001",
            "raw": {"category": "按组合计提坏账准备的应收账款"},
            "canonical_raw": {"category": "按组合计提坏账准备的应收账款"},
            "normalized": {"category": "按组合计提坏账准备的应收账款"},
            "source": {
                "page": 32,
                "physical_table_id": "pt_32_1",
                "source_cell_refs": [
                    {"page": 32, "table_id": "pt_32_1", "row": 3, "col": col, "field_name": field_name}
                    for col, field_name in enumerate(field_names)
                ],
            },
        }
    ]
    result = _audit_result(PageContent(page_number=32, tables=[table]))

    warnings = quality_warnings(
        result,
        {"audit_document_number": "测试审字〔2025〕第1号"},
        [{"type": "audit_opinion"}, {"type": "basis_for_opinion"}],
        {"accounts_receivable": rows},
        [],
    )

    blockers = dataset_blocking_warnings(warnings)
    assert any(warning.startswith("AUDIT_NOTE_SOURCE_COLUMN_OMITTED") for warning in blockers)
    assert any(warning.startswith("AUDIT_NOTE_SOURCE_CELL_OMITTED") for warning in blockers)


def test_audit_record_normalization_keeps_dash_amount_placeholders_empty() -> None:
    record = {
        "raw": {"项目": "信用减值损失", "本期发生额": "—", "上期发生额": "-"},
        "canonical_raw": {"项目": "信用减值损失", "本期发生额": "—", "上期发生额": "-"},
        "normalized": {},
    }

    normalized = normalize_audit_record(record)

    assert normalized["normalized"] == {
        "项目": "信用减值损失",
        "本期发生额": None,
        "上期发生额": None,
    }


def test_audit_record_normalization_recomputes_source_backed_decimals_without_rounding() -> None:
    record = {
        "raw": {"share_count_ten_thousand": "1,099.6063"},
        "canonical_raw": {"share_count_ten_thousand": "1,099.6063"},
        "normalized": {"share_count_ten_thousand": {"value": 1099.61, "currency": "CNY"}},
        "source": {
            "evidence_ids": ["ev:source"],
            "source_cell_refs": [{"field_name": "share_count_ten_thousand", "page": 12}],
        },
    }

    normalized = normalize_audit_record(record)

    assert normalized["normalized"]["share_count_ten_thousand"] == "1099.6063"
    assert normalized["canonical_raw"]["share_count_ten_thousand"] == "1,099.6063"


def test_audit_note_dictionary_declares_semantic_numeric_columns_as_decimal() -> None:
    dictionary = audit_data_dictionary(
        [],
        {
            "credit_impairment_losses": {
                "item": {"label": "项目", "type": "string"},
                "current_period_amount": {"label": "本期发生额", "type": "decimal"},
                "shareholding_ratio": {"label": "持股比例(%)", "type": "decimal"},
            }
        },
    )
    columns = dictionary["datasets"]["credit_impairment_losses"]["columns"]

    assert columns["item"]["type"] == "string"
    assert columns["current_period_amount"]["type"] == "decimal"
    assert columns["shareholding_ratio"]["type"] == "decimal"


def test_audit_quality_blocks_source_backed_numeric_drift() -> None:
    rows = [
        {
            "raw": {"share_count_ten_thousand": "1,099.6063"},
            "canonical_raw": {"share_count_ten_thousand": "1,099.6063"},
            "normalized": {"share_count_ten_thousand": {"value": 1099.61, "currency": "CNY"}},
            "source": {
                "evidence_ids": ["ev:source"],
                "source_cell_refs": [{"field_name": "share_count_ten_thousand", "page": 12}],
            },
        }
    ]

    warnings = quality_warnings(
        _audit_result(PageContent(page_number=12)),
        {"audit_document_number": "测试审字〔2025〕第1号"},
        [{"type": "audit_opinion"}, {"type": "basis_for_opinion"}],
        {"capital_change_history": rows},
        [],
    )

    assert "AUDIT_NORMALIZED_NUMERIC_MISMATCH:dataset=capital_change_history:count=1" in warnings
    assert any(warning.startswith("AUDIT_NORMALIZED_NUMERIC_MISMATCH") for warning in dataset_blocking_warnings(warnings))


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

    assert rows[0]["raw"]["ending_balance"] == "153,098.43"
    assert rows[0]["raw"]["opening_balance"] == "76,403.83"
    assert rows[1]["raw"]["ending_balance"] == "1,648,700.00"
    assert rows[1]["raw"]["aging_bucket"] == "1年以内"
    assert rows[0]["source"]["recovery_raw"]["期末余额"] == "153,098.4376,403.83"
    assert rows[0]["source_cell_refs"][-1]["field_name"] == "opening_balance"
    assert any(warning.startswith("AUDIT_NOTE_ADJACENT_CELLS_SPLIT") for warning in _repair_events(projection))


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
    assert any(warning.startswith("AUDIT_STATEMENT_CELLS_RECOVERED") for warning in _repair_events(projection))


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
    assert any(warning.startswith("AUDIT_STATEMENT_HEADER_ROWS_REMOVED") for warning in _repair_events(projection))


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
    assert any(warning.startswith("AUDIT_BALANCE_SECTION_LABEL_RECOVERED") for warning in _repair_events(projection))


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
    assert any(warning.startswith("AUDIT_BALANCE_AMOUNT_SHIFT_RECOVERED") for warning in _repair_events(projection))
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
    assert any(warning.startswith("AUDIT_OWNER_EQUITY_ROTATION_RECOVERED") for warning in _repair_events(projection))


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


def test_audit_projection_preserves_cross_statement_values_without_judging_source_consistency() -> None:
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
    assert not any(warning.startswith("AUDIT_STATEMENT_TOTAL_MISMATCH") for warning in projection.warnings)
    assert "balance_sheet" not in projection.domain_facts["dataset_verification_blockers"]
    assert "owners_equity_changes" not in projection.domain_facts["dataset_verification_blockers"]


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
    assert any(warning.startswith("AUDIT_OWNER_EQUITY_HEADER_ROWS_REMOVED") for warning in _repair_events(projection))


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


def test_audit_metadata_prefers_complete_body_number_over_complete_cover_candidate() -> None:
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

    assert projection.domain_facts["audit_document_number"] == "鼎迈会师审字〔2024〕第0554号"
    assert projection.domain_facts["field_details"]["audit_document_number_candidates"]["values"] == [
        "鼎迈会师审字〔2025〕第0123号",
        "鼎迈会师审字〔2024〕第0554号",
    ]
    assert "regulatory_report_id" not in projection.domain_facts
    assert any(warning.startswith("AUDIT_DOCUMENT_NUMBER_CANDIDATE_VARIANCE") for warning in projection.warnings)
    assert not any(warning.startswith("AUDIT_DOCUMENT_NUMBER_CONFLICT") for warning in projection.warnings)
    assert any(warning.startswith("AUDIT_REGULATORY_REPORT_ID_INVALID") for warning in projection.warnings)
    assert bundle is not None
    assert bundle.json_payload()["document"]["title"] == "2024年度审计报告"


def test_audit_metadata_records_incomplete_cover_placeholder_without_warning() -> None:
    result = _audit_result(
        PageContent(page_number=1, texts=[TextBlock(content="2024年度审计报告\n会师审字[2025]第 号")]),
        PageContent(page_number=3, texts=[TextBlock(content="鼎迈会师审字[2024]第0554号")]),
    )

    projection = derive_audit_report_projection(result)

    assert projection.domain_facts["audit_document_number"] == "鼎迈会师审字[2024]第0554号"
    candidates = projection.domain_facts["field_details"]["audit_document_number_candidates"]
    assert candidates["values"] == ["鼎迈会师审字[2024]第0554号"]
    assert candidates["incomplete_values"] == ["会师审字[2025]第号"]
    assert not any(warning.startswith("AUDIT_DOCUMENT_NUMBER_INCOMPLETE") for warning in projection.warnings)
    assert not any(warning.startswith("AUDIT_DOCUMENT_NUMBER_CONFLICT") for warning in projection.warnings)


def test_audit_metadata_warns_when_only_incomplete_number_exists() -> None:
    result = _audit_result(
        PageContent(page_number=1, texts=[TextBlock(content="2024年度审计报告\n会师审字[2025]第 号")]),
    )

    projection = derive_audit_report_projection(result)

    assert "audit_document_number" not in projection.domain_facts
    assert any(warning.startswith("AUDIT_DOCUMENT_NUMBER_INCOMPLETE") for warning in projection.warnings)
    assert "AUDIT_DOCUMENT_NUMBER_MISSING" in projection.warnings


def test_audit_projection_normalizes_supported_cjk_radical_variants() -> None:
    table = TableBlock(
        table_id="pt_20_0",
        page=20,
        headers=["项目", "金额"],
        rows=[_row(20, "pt_20_0", 0, ["江⻄⻋船税及⻮轮", "10.00"])],
    )
    result = _audit_result(PageContent(page_number=20, texts=[TextBlock(content="财务报表附注")], tables=[table]))

    projection = derive_audit_report_projection(result)
    row = next(iter(projection.datasets.values()))[0]

    assert row["raw"]["item"] == "江⻄⻋船税及⻮轮"
    assert row["normalized"]["item"] == "江西车船税及齿轮"
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


def test_audit_sections_reject_numbered_responsibility_sentences() -> None:
    result = _audit_result(
        PageContent(
            page_number=4,
            texts=[
                TextBlock(content="五、注册会计师对财务报表审计的责任"),
                TextBlock(content="(1)识别和评估由于舞弊或错误导致的财务报表重大错报风险，设计和实施审计程序。"),
                TextBlock(content="(2)了解与审计相关的内部控制，以设计恰当的审计程序。"),
            ],
        )
    )

    projection = derive_audit_report_projection(result)
    titles = {section["title"] for section in projection.sections}

    assert "五、注册会计师对财务报表审计的责任" in titles
    assert not any(title.startswith("(1)") or title.startswith("(2)") for title in titles)


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
    assert any(warning.startswith("AUDIT_STATEMENT_TEXT_ROWS_SELECTED") for warning in _repair_events(projection))


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
    assert "period_role" not in rows[0]["raw"]
    assert "period_role" not in rows[0]["canonical_raw"]
    assert rows[0]["normalized"]["period_role"] == "current"
    assert rows[0]["raw"]["paid_in_capital"] == "10.00"
    assert rows[0]["raw"]["capital_reserve"] == "20.00"
    assert rows[0]["raw"]["retained_earnings"] == "30.00"
    assert rows[0]["raw"]["total_equity"] == "60.00"
    assert not any(key.startswith("column_") for key in rows[0]["raw"])
    dictionary_columns = projection.domain_facts["data_dictionary"]["datasets"]["owners_equity_changes"]["columns"]
    assert list(dictionary_columns)[:2] == ["item", "period_role"]
    assert [descriptor["label"] for descriptor in list(dictionary_columns.values())[2:]] == [
        "实收资本（或股本）",
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
    assert any(warning.startswith("AUDIT_OWNER_EQUITY_HEADER_ROWS_REMOVED") for warning in _repair_events(projection))


def test_audit_owner_equity_recovers_source_confirmed_sparse_label_row() -> None:
    headers = [
        "项目",
        "实收资本（或股本）",
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
        table_id="pt_10_0",
        page=10,
        headers=headers,
        rows=[
            _row(
                10, "pt_10_0", 0, ["一、上年期末余额", "10.00", "", "", "", "20.00", "", "", "", "", "30.00", "60.00"]
            ),
            _row(10, "pt_10_0", 1, ["（二）所有者投入和减少资本", "", "", "", "", "", "", "", "", "", "", ""]),
            _row(10, "pt_10_0", 2, ["1.所有者（股东）投入的普通股", "", "", "", "", "", "", "", "", "", "", ""]),
            _row(10, "pt_10_0", 3, ["2.其他权益工具持有者投入资本", "", "", "", "", "", "", "", "", "", "", ""]),
            _row(10, "pt_10_0", 4, ["4.其他", "", "", "", "", "", "", "", "", "", "", ""]),
        ],
    )
    result = _audit_result(
        PageContent(
            page_number=10,
            texts=[
                TextBlock(content="所有者（股东）权益变动表"),
                TextBlock(
                    content="3.股份支付计入所有者权益的金额",
                    bbox=[80.0, 300.0, 320.0, 320.0],
                    evidence_ids=["ev:owner:share_payment"],
                ),
            ],
            tables=[table],
        )
    )

    projection = derive_audit_report_projection(result)
    rows = projection.datasets["owners_equity_changes"]

    assert [row["raw"]["item"] for row in rows][3:6] == [
        "2.其他权益工具持有者投入资本",
        "3.股份支付计入所有者权益的金额",
        "4.其他",
    ]
    recovered = rows[4]
    assert recovered["source"]["page"] == 10
    assert recovered["source"]["source_row_index"] == 1
    assert recovered["source"]["source_resolution"] == "canonical_text_label_row"
    assert recovered["source"]["evidence_ids"] == ["ev:owner:share_payment"]
    assert recovered["source"]["field_sources"]["period_role"] == {
        "source": "derived.statement_period_role",
        "page": 10,
        "derivation": "source_statement_period_header",
    }
    assert "period_role" not in recovered["raw"]
    assert "period_role" not in recovered["canonical_raw"]
    bundle = AuditReportPlugin().project_bundle(seal_parse_result(result), file_path="audit.pdf")
    assert bundle is not None
    payload = bundle.json_payload()
    dataset = next(item for item in payload["datasets"] if item["name"] == "owners_equity_changes")
    period_column = next(column for column in dataset["columns"] if column["key"] == "period_role")
    assert period_column["raw_available"] is False
    assert any(warning.startswith("AUDIT_OWNER_EQUITY_LABEL_ROWS_RECOVERED") for warning in _repair_events(projection))


def test_audit_projection_recovers_accounts_payable_total_from_next_page_text() -> None:
    table = TableBlock(
        table_id="pt_39_0",
        page=39,
        headers=["项目", "期末余额", "期初余额"],
        rows=[_row(39, "pt_39_0", 0, ["运输服务款", "21,733,124.67", "17,279,642.10"])],
        bbox=[80.0, 200.0, 530.0, 260.0],
    )
    result = _audit_result(
        PageContent(
            page_number=39,
            texts=[
                TextBlock(content="14、应付账款", bbox=[100.0, 140.0, 240.0, 160.0]),
                TextBlock(content="（1）应付账款列示", bbox=[100.0, 170.0, 260.0, 190.0]),
            ],
            tables=[table],
        ),
        PageContent(
            page_number=40,
            texts=[
                TextBlock(
                    content="合 计 19,233,124.67 17,279,642.10",
                    bbox=[80.0, 70.0, 530.0, 90.0],
                    evidence_ids=["ev:accounts_payable:total"],
                )
            ],
        ),
    )

    projection = derive_audit_report_projection(result)
    rows = projection.datasets["accounts_payable"]

    assert len(rows) == 2
    assert rows[1]["raw"] == {
        "item": "合 计",
        "ending_balance": "19,233,124.67",
        "opening_balance": "17,279,642.10",
    }
    assert rows[1]["normalized"]["item"] == "合计"
    assert rows[1]["source"]["page"] == 40
    assert rows[1]["source"]["source_row_index"] == 0
    assert rows[1]["source"]["evidence_ids"] == ["ev:accounts_payable:total"]
    assert any(warning.startswith("AUDIT_NOTE_TEXT_CONTINUATION_RECOVERED") for warning in _repair_events(projection))
    assert not any(warning.startswith("AUDIT_NOTE_TOTAL_MISMATCH") for warning in projection.warnings)


def test_audit_projection_merges_headerless_total_table_from_next_page() -> None:
    detail = TableBlock(
        table_id="pt_33_0",
        page=33,
        headers=["预付对象", "期末余额", "占预付款项合计数比例(%)"],
        rows=[
            _row(33, "pt_33_0", 0, ["甲公司", "60.00", "60.00"]),
            _row(33, "pt_33_0", 1, ["乙公司", "40.00", "40.00"]),
        ],
    )
    total = TableBlock(
        table_id="pt_34_0",
        page=34,
        headers=[],
        rows=[_row(34, "pt_34_0", 0, ["合 计", "100.00", "100.00"])],
    )
    unrelated = TableBlock(
        table_id="pt_34_1",
        page=34,
        headers=["项目", "期末余额"],
        rows=[_row(34, "pt_34_1", 0, ["其他应收款", "20.00"])],
    )
    result = _audit_result(
        PageContent(
            page_number=33,
            texts=[
                TextBlock(content="3、预付款项"),
                TextBlock(content="（2）按预付对象归集的期末余额前五名的预付款情况"),
            ],
            tables=[detail],
        ),
        PageContent(page_number=34, texts=[TextBlock(content="4、其他应收款")], tables=[total, unrelated]),
    )

    projection = derive_audit_report_projection(result)
    rows = next(
        records
        for records in projection.datasets.values()
        if any((record.get("normalized") or {}).get("prepayment_counterparty") == "甲公司" for record in records)
    )

    assert len(rows) == 3
    assert rows[-1]["normalized"] == {
        "prepayment_counterparty": "合计",
        "ending_balance": "100.00",
        "prepayments_share_ratio": "100.00",
    }
    assert rows[-1]["source"]["page"] == 34
    assert rows[-1]["source"]["physical_table_id"] == "pt_34_0"
    assert {ref["field_name"] for ref in rows[-1]["source"]["source_cell_refs"]} == {
        "prepayment_counterparty",
        "ending_balance",
        "prepayments_share_ratio",
    }
    assert any("mode=orphan_total" in warning for warning in _repair_events(projection))


def test_audit_projection_does_not_merge_orphan_total_after_completed_table() -> None:
    previous = TableBlock(
        table_id="pt_1_0",
        page=1,
        headers=["项目", "金额"],
        rows=[_row(1, "pt_1_0", 0, ["合计", "100.00"])],
    )
    next_page = TableBlock(
        table_id="pt_2_0",
        page=2,
        headers=[],
        rows=[_row(2, "pt_2_0", 0, ["合计", "200.00"])],
    )
    result = _audit_result(
        PageContent(page_number=1, tables=[previous]),
        PageContent(page_number=2, tables=[next_page]),
    )
    datasets = {
        "table_001": [
            _projected_record(
                "table_001",
                1,
                {"项目": "合计", "金额": "100.00"},
                table_id="pt_1_0",
                physical_table_id="pt_1_0",
                page=1,
            )
        ]
    }

    warnings = merge_cross_page_continuations(datasets, result)

    assert len(datasets["table_001"]) == 1
    assert not any("mode=orphan_total" in warning for warning in warnings)


def test_audit_source_sync_only_backfills_evidence_backed_canonical_raw() -> None:
    datasets = {
        "table_001": [
            {
                "record_id": "table_001:r000001",
                "raw": {"项目": "测试公司", "期末余额": "100.00"},
                "canonical_raw": {"项目": "", "期末余额": ""},
                "normalized": {"项目": "测试公司", "期末余额": "100.00", "period_role": "current"},
                "source": {
                    "page": 1,
                    "evidence_ids": ["ev:source"],
                    "source_cell_refs": [
                        {"field_name": "项目", "page": 1},
                        {"field_name": "期末余额", "page": 1},
                    ],
                    "field_sources": {
                        "period_role": {
                            "source": "derived.statement_period_role",
                            "page": 1,
                            "derivation": "source_statement_period_header",
                        }
                    },
                },
            }
        ]
    }

    warnings = synchronize_audit_record_sources(datasets)
    row = datasets["table_001"][0]

    assert row["canonical_raw"] == {"项目": "测试公司", "期末余额": "100.00"}
    assert "period_role" not in row["canonical_raw"]
    assert warnings == ["AUDIT_CANONICAL_RAW_RECOVERED:fields=2"]


def test_audit_bundle_propagates_document_units() -> None:
    table = TableBlock(
        table_id="pt_6_0",
        page=6,
        headers=["项目", "期末余额"],
        rows=[_row(6, "pt_6_0", 0, ["货币资金", "1,200.00"])],
    )
    result = _audit_result(
        PageContent(
            page_number=6,
            texts=[TextBlock(content="资产负债表\n编制单位：测试科技股份有限公司\n单位：元\n币种：人民币")],
            tables=[table],
        )
    )

    projection = derive_audit_report_projection(result)
    bundle = AuditReportPlugin().project_bundle(seal_parse_result(result), file_path="audit.pdf")

    assert projection.domain_facts["currency"] == "人民币"
    assert projection.domain_facts["currency_unit"] == "元"
    assert not any(warning.startswith("precision:generic_currency_unknown") for warning in projection.warnings)
    assert bundle is not None
    assert bundle.json_payload()["document"]["units"] == {"currency": "人民币", "currency_unit": "元"}
