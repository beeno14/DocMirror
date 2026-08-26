from __future__ import annotations

from docmirror.models.entities.parse_result import (
    CellValue,
    DocumentEntities,
    KeyValuePair,
    PageContent,
    ParseResult,
    ResultStatus,
    RowType,
    TableBlock,
    TableRow,
    TextBlock,
)
from docmirror.models.sealed import seal_parse_result
from docmirror.plugins._base.financial_source_projection import (
    ColumnSpec,
    build_records,
    extract_labeled_header_fields,
)
from docmirror.plugins._runtime.plugin_registry import PluginRegistry
from docmirror.plugins.tax_return.projection import (
    _normalize_profiled_line_formulas,
    _normalize_profiled_tax_header_bands,
)
from docmirror.server.output_builder import build_community_bundle


def _row(page: int, row_index: int, values: list[str], *, row_type: RowType = RowType.DATA) -> TableRow:
    cells = [
        CellValue(
            text=value,
            row_index=row_index,
            col_index=column,
            bbox=[float(column * 100), float(row_index * 20), float((column + 1) * 100), float((row_index + 1) * 20)],
            evidence_ids=[f"ev:{page:04d}:{row_index:04d}:{column:02d}"],
            geometry_status="exact",
        )
        for column, value in enumerate(values)
    ]
    return TableRow(
        cells=cells,
        row_type=row_type,
        source_page=page,
        source_row_index=row_index,
        source_physical_id=f"pt_{page}_0",
    )


def _dataset(payload: dict, dataset_id: str) -> dict:
    return next(dataset for dataset in payload["datasets"] if dataset["id"] == f"ds_{dataset_id}")


def test_financial_records_use_geometry_confidence_and_review_unresolved_empty_quotes() -> None:
    cells = [
        CellValue(
            text="说明：符号写作“”",
            row_index=0,
            col_index=0,
            bbox=[0.0, 0.0, 100.0, 20.0],
            geometry_status="exact",
            geometry_confidence=0.73,
            evidence_ids=["ev:text"],
        ),
        CellValue(
            text="1",
            row_index=0,
            col_index=1,
            bbox=[100.0, 0.0, 120.0, 20.0],
            geometry_status="exact",
            geometry_confidence=0.91,
            evidence_ids=["ev:line"],
        ),
    ]
    row = TableRow(cells=cells, source_page=1, source_row_index=0)
    table = TableBlock(table_id="pt_1_0", page=1, headers=["项目", "行次"], rows=[row])

    records, warnings = build_records(
        table,
        [(0, row)],
        [
            ColumnSpec(source_index=0, key="item", label="项目", value_type="string"),
            ColumnSpec(source_index=1, key="line_no", label="行次", value_type="string"),
        ],
        dataset_id="income_statement",
    )

    assert records[0]["confidence"] == 0.73
    assert records[0]["review"] == {
        "required": True,
        "reasons": ["ambiguous_empty_quoted_text"],
    }
    assert any("financial_ambiguous_empty_quoted_text" in warning for warning in warnings)


def test_financial_header_full_text_fallback_does_not_fabricate_page_one_source() -> None:
    fields, details = extract_labeled_header_fields(
        ParseResult(raw_text="纳税人名称:测试科技股份有限公司", entities=DocumentEntities())
    )

    assert fields["subject_name"] == "测试科技股份有限公司"
    assert "source_page" not in details["subject_name"]
    assert details["subject_name"]["source_refs"] == []
    assert details["subject_name"]["review"] == "needs_evidence"


def test_financial_record_confidence_is_bounded_by_table_extraction_confidence() -> None:
    row = _row(1, 0, ["营业收入", "1", "100.00"])
    table = TableBlock(
        table_id="pt_1_0",
        page=1,
        headers=["项目", "行次", "本期金额"],
        rows=[row],
        confidence=0.64,
        extraction_confidence=0.64,
    )

    records, _warnings = build_records(
        table,
        [(0, row)],
        [
            ColumnSpec(source_index=0, key="item", label="项目", value_type="string"),
            ColumnSpec(source_index=1, key="line_no", label="行次", value_type="string"),
            ColumnSpec(source_index=2, key="amount", label="本期金额", value_type="decimal"),
        ],
        dataset_id="income_statement",
    )

    assert records[0]["confidence"] == 0.64


def test_financial_statement_alias_routes_to_isolated_projector_and_preserves_source_columns() -> None:
    table = TableBlock(
        table_id="pt_1_0",
        page=1,
        headers=["项目", "行次", "本年累计金额", "本月金额"],
        rows=[
            _row(1, 0, ["项目", "行次", "本年累计金额", "本月金额"], row_type=RowType.HEADER),
            _row(1, 1, ["一、经营活动产生的现金流量", "", "", ""]),
            _row(1, 2, ["销售商品、提供劳务收到的现金", "1", "100.00", "20.00"]),
            _row(1, 3, ["经营活动产生的现金流量净额", "2", "80.0", "10.00"]),
        ],
    )
    result = ParseResult(
        status=ResultStatus.SUCCESS,
        pages=[
            PageContent(
                page_number=1,
                texts=[
                    TextBlock(
                        content=(
                            "现金流量表\n税款所属期起止:2025-05-01 至 2025-05-31\n"
                            "纳税人识别号:913100005515731558\n报送日期:2025-06-23\n"
                            "纳税人名称:(公章) 测试科技股份有限公司\n单位：元"
                        )
                    )
                ],
                tables=[table],
            )
        ],
        entities=DocumentEntities(document_type="cash_flow_statement"),
    )
    sealed = seal_parse_result(result)
    before = sealed.integrity_fingerprint

    bundle = build_community_bundle(sealed)
    payload = bundle.json_payload(bundle.semantic_payload())
    dataset = _dataset(payload, "cash_flow_statement")

    assert bundle.classification["projector_id"] == "financial_statement"
    assert "fallback_reason" not in bundle.classification
    assert [column["key"] for column in dataset["columns"]] == [
        "item",
        "line_no",
        "year_to_date_amount",
        "current_month_amount",
    ]
    assert dataset["row_count"] == 3
    assert dataset["rows"][0]["raw"] == {
        "item": "一、经营活动产生的现金流量",
        "line_no": "",
        "year_to_date_amount": "",
        "current_month_amount": "",
    }
    assert dataset["rows"][1]["raw"]["item"] == "销售商品、提供劳务收到的现金"
    assert dataset["rows"][1]["source"]["source_cell_refs"][0]["row"] == 2
    assert dataset["rows"][2]["review"]["required"] is True
    assert "decimal_scale_unexpected" in dataset["rows"][2]["review"]["reasons"]
    assert any("financial_decimal_scale_unexpected" in warning["message"] for warning in payload["warnings"])
    assert dataset["section_id"] == "section_cash_flow_statement"
    assert [section["title"] for section in payload["sections"]] == ["现金流量表"]
    reading_table = next(table for table in payload["reading"]["tables"] if table["dataset_id"] == dataset["id"])
    assert not reading_table.get("row_groups")
    enhanced = bundle.render_enhanced_markdown(bundle.semantic_payload())
    assert "#### 一、经营活动产生的现金流量" not in enhanced
    assert "| 一、经营活动产生的现金流量 |  |  |  |" in enhanced
    document_items = {item["key"]: item for section in payload["sections"] for item in section["items"]}
    assert document_items["subject_name"]["value"] == "测试科技股份有限公司"
    assert document_items["subject_name"]["raw"] == "测试科技股份有限公司"
    assert document_items["subject_name"]["source"]["page_range"] == [1, 1]
    assert document_items["subject_name"]["confidence"] == 1.0
    assert document_items["subject_name"]["review"] == "auto_accepted"
    assert document_items["subject_id"]["value"] == "913100005515731558"
    assert document_items["period_start"]["value"] == "2025-05-01"
    assert document_items["period_end"]["value"] == "2025-05-31"
    assert document_items["period_end"]["raw"] == "2025-05-31"
    assert document_items["document_date"]["value"] == "2025-06-23"
    assert document_items["currency_unit"]["value"] == "CNY"
    assert document_items["currency_unit"]["raw"] == "元"
    assert sealed.integrity_fingerprint == before
    assert sealed.verify_integrity()


def test_cash_flow_preserves_all_physical_rows_in_one_ungrouped_dataset() -> None:
    source_rows = [
        ["一、经营活动产生的现金流量：", "", "", ""],
        *[[f"经营活动项目{line}", str(line), "0.00", "0.00"] for line in range(1, 7)],
        ["经营活动产生的现金流量净额", "7", "0.00", "0.00"],
        ["二、投资活动产生的现金流量：", "", "", ""],
        *[[f"投资活动项目{line}", str(line), "0.00", "0.00"] for line in range(8, 13)],
        ["投资活动产生的现金流量净额", "13", "0.00", "0.00"],
        ["三、筹资活动所产生的现金流量：", "", "", ""],
        *[[f"筹资活动项目{line}", str(line), "0.00", "0.00"] for line in range(14, 19)],
        ["筹资活动产生的现金流量净额", "19", "0.00", "0.00"],
        ["四、现金净增加额", "20", "0.00", "0.00"],
        ["加：期初现金余额", "21", "0.00", "0.00"],
        ["五、期末现金余额", "22", "0.00", "0.00"],
    ]
    table = TableBlock(
        table_id="pt_1_0",
        page=1,
        headers=["项目", "行次", "本年累计金额", "本月金额"],
        rows=[
            _row(1, 0, ["项目", "行次", "本年累计金额", "本月金额"], row_type=RowType.HEADER),
            *[_row(1, row_index, values) for row_index, values in enumerate(source_rows, start=1)],
        ],
    )
    result = ParseResult(
        status=ResultStatus.SUCCESS,
        pages=[PageContent(page_number=1, tables=[table])],
        entities=DocumentEntities(document_type="cash_flow_statement"),
    )

    bundle = build_community_bundle(seal_parse_result(result))
    payload = bundle.json_payload(bundle.semantic_payload())
    dataset = _dataset(payload, "cash_flow_statement")
    enhanced = bundle.render_enhanced_markdown(bundle.semantic_payload())

    assert dataset["row_count"] == 25
    assert [row["raw"]["item"] for row in dataset["rows"]] == [row[0] for row in source_rows]
    assert [row["raw"]["line_no"] for row in dataset["rows"] if row["raw"]["line_no"]] == [
        str(line) for line in range(1, 23)
    ]
    assert not dataset.get("row_groups")
    reading_table = next(table for table in payload["reading"]["tables"] if table["dataset_id"] == dataset["id"])
    assert not reading_table.get("row_groups")
    assert "#### 一、经营活动产生的现金流量" not in enhanced
    assert "#### 二、投资活动产生的现金流量" not in enhanced
    assert "#### 三、筹资活动所产生的现金流量" not in enhanced
    assert "| 一、经营活动产生的现金流量： |  |  |  |" in enhanced
    assert "| 二、投资活动产生的现金流量： |  |  |  |" in enhanced
    assert "| 三、筹资活动所产生的现金流量： |  |  |  |" in enhanced
    assert "| 四、现金净增加额 | 20 | 0.00 | 0.00 |" in enhanced
    assert "| 五、期末现金余额 | 22 | 0.00 | 0.00 |" in enhanced


def test_financial_enhanced_markdown_preserves_source_format_and_hides_technical_metadata() -> None:
    table = TableBlock(
        table_id="pt_1_0",
        page=1,
        headers=["项目", "行次", "本年累计金额", "本月金额"],
        rows=[
            _row(1, 0, ["项目", "行次", "本年累计金额", "本月金额"], row_type=RowType.HEADER),
            _row(1, 1, ["减：营业成本", "1", "21,141,633.42", "1,234.50"]),
        ],
    )
    result = ParseResult(
        status=ResultStatus.SUCCESS,
        pages=[PageContent(page_number=1, tables=[table])],
        entities=DocumentEntities(document_type="income_statement"),
    )

    bundle = build_community_bundle(seal_parse_result(result))
    enhanced = bundle.render_enhanced_markdown(bundle.semantic_payload())

    assert "| 减：营业成本 | 1 | 21,141,633.42 | 1,234.50 |" in enhanced
    assert "21141633.42" not in enhanced
    assert "**文档类型:**" not in enhanced
    assert "canonical document type" not in enhanced
    assert "**language:**" not in enhanced


def test_balance_sheet_preserves_two_sided_section_rows_as_source_records() -> None:
    headers = [
        "资产",
        "行次",
        "期末余额",
        "年初余额",
        "负债和所有者权益",
        "行次",
        "期末余额",
        "年初余额",
    ]
    source_rows = [
        ["流动资产：", "", "", "", "流动负债：", "", "", ""],
        ["货币资金", "1", "100.00", "80.00", "短期借款", "31", "20.00", "10.00"],
    ]
    table = TableBlock(
        table_id="pt_1_0",
        page=1,
        headers=headers,
        rows=[
            _row(1, 0, headers, row_type=RowType.HEADER),
            _row(1, 1, source_rows[0], row_type=RowType.HEADER),
            _row(1, 2, source_rows[1]),
        ],
    )
    result = ParseResult(
        status=ResultStatus.SUCCESS,
        pages=[PageContent(page_number=1, tables=[table])],
        entities=DocumentEntities(document_type="balance_sheet"),
    )

    bundle = build_community_bundle(seal_parse_result(result))
    payload = bundle.json_payload(bundle.semantic_payload())
    dataset = _dataset(payload, "balance_sheet")

    assert dataset["row_count"] == 2
    assert dataset["rows"][0]["raw"] == {
        "asset_item": "流动资产：",
        "asset_line_no": "",
        "asset_ending_balance": "",
        "asset_opening_balance": "",
        "liability_and_equity_item": "流动负债：",
        "liability_line_no": "",
        "liability_ending_balance": "",
        "liability_opening_balance": "",
    }
    assert dataset["completeness"] == {
        "expected_row_count": 2,
        "emitted_row_count": 2,
        "omitted_row_count": 0,
        "verified": True,
        "basis": "source_row_ledger",
    }
    assert dataset["status"] == "complete"


def test_tax_return_uses_embedded_header_band_and_keeps_headerless_continuation() -> None:
    page_one = TableBlock(
        table_id="pt_1_0",
        page=1,
        headers=[
            "纳税人名称",
            "测试科技\n股份有限公司\nNsrSignkey",
            "税款所属时间",
            "2025-05-01 至 2025-05-31",
            "",
            "",
            "",
        ],
        rows=[
            _row(
                1,
                0,
                [
                    "纳税人名称",
                    "测试科技\n股份有限公司\nNsrSignkey",
                    "税款所属时间",
                    "2025-05-01 至 2025-05-31",
                    "",
                    "",
                    "",
                ],
            ),
            _row(1, 1, ["项目", "", "栏次", "一般项目", "", "即征即退项目", ""]),
            _row(1, 2, ["", "", "", "本月数", "本年累计", "本月数", "本年累计"]),
            _row(1, 3, ["销售额", "按适用税率计税销售额", "1", "100.00", "500.00", "0.00", "0.00"]),
            _row(1, 4, ["其他免税项目", "", "", "", "", "", ""]),
            _row(1, 5, ["", "免税货物销售额", "9", "0.00", "0.00", "--", "--"]),
            _row(1, 6, ["", "", "34", "", "", "", ""]),
            _row(1, 7, ["填表日期", "2025-06-09", "金额单位", "元", "", "", ""]),
        ],
    )
    page_two = TableBlock(
        table_id="pt_2_0",
        page=2,
        headers=[],
        rows=[
            _row(2, 0, ["附加税费", "城市维护建设税", "39", "7.00", "77.00", "--", "--"]),
        ],
    )
    result = ParseResult(
        status=ResultStatus.SUCCESS,
        pages=[
            PageContent(page_number=1, tables=[page_one]),
            PageContent(
                page_number=2,
                texts=[TextBlock(content="纳税人名称:测试科技股份有限公司", confidence=0.99)],
                tables=[page_two],
            ),
        ],
        entities=DocumentEntities(document_type="tax_return"),
    )

    bundle = build_community_bundle(seal_parse_result(result))
    payload = bundle.json_payload(bundle.semantic_payload())
    dataset = _dataset(payload, "tax_return_main")

    assert bundle.classification["projector_id"] == "tax_return"
    assert [column["key"] for column in dataset["columns"]] == [
        "item_category",
        "item_name",
        "line_no",
        "general_current_month",
        "general_year_to_date",
        "immediate_refund_current_month",
        "immediate_refund_year_to_date",
    ]
    assert dataset["row_count"] == 4
    assert [record["source"]["page_range"] for record in dataset["rows"]] == [
        [1, 1],
        [1, 1],
        [1, 1],
        [2, 2],
    ]
    assert dataset["rows"][2]["raw"] == {
        "item_category": "",
        "item_name": "",
        "line_no": "34",
        "general_current_month": "",
        "general_year_to_date": "",
        "immediate_refund_current_month": "",
        "immediate_refund_year_to_date": "",
    }
    assert all(record["raw"]["item_category"] != "其他免税项目" for record in dataset["rows"])
    assert [section["title"] for section in payload["sections"]] == ["纳税申报主表"]
    reading_table = next(table for table in payload["reading"]["tables"] if table["dataset_id"] == dataset["id"])
    assert any(group["title"] == "其他免税项目" for group in reading_table["row_groups"])
    assert not any("tax_orphan_line_no" in warning["message"] for warning in payload["warnings"])
    assert payload["document"]["type"] == "tax_return"
    document_items = {item["key"]: item for section in payload["sections"] for item in section["items"]}
    assert document_items["subject_name"]["value"] == "测试科技股份有限公司"
    assert document_items["subject_name"]["raw"] == "测试科技股份有限公司"
    assert document_items["subject_name"]["source"]["page_range"] == [2, 2]
    assert document_items["period_start"]["value"] == "2025-05-01"
    assert document_items["period_end"]["value"] == "2025-05-31"
    assert document_items["document_date"]["value"] == "2025-06-09"
    assert document_items["currency_unit"]["value"] == "CNY"


def test_tax_return_keeps_pre_header_choice_region_inside_existing_reading_table() -> None:
    parent_header = _row(
        6,
        6,
        [
            "税（费）种",
            "计税（费）依据",
            "",
            "本期减免税（费）额",
            "",
            "本期应补（退）税（费）额",
        ],
    )
    parent_header.cells[1].col_span = 2
    parent_header.cells[3].col_span = 2
    table = TableBlock(
        table_id="pt_6_0",
        page=6,
        headers=[],
        rows=[
            _row(6, 0, ["增值税及附加税费申报表附列资料", "", "", "", "", ""]),
            _row(6, 1, ["（附加税费情况表）", "", "", "", "", ""]),
            _row(6, 2, ["税（费）款所属时间：2025年5月1日至2025年5月31日", "", "", "", "", ""]),
            _row(
                6,
                3,
                [
                    "纳税人名称：（公\n章）",
                    "测试科技股份有限公司",
                    "",
                    "",
                    "金额单位：元（列至角分）",
                    "",
                ],
            ),
            _row(
                6,
                4,
                [
                    "本期是否适用优惠政策",
                    "",
                    "□ 是 ■ 否",
                    "适用主体",
                    "□ 个体工商户 □ 小型微利企业",
                    "",
                ],
            ),
            _row(6, 5, ["", "", "", "适用政策起止时间", "年 月 日 至 年 月 日", ""]),
            parent_header,
            _row(
                6,
                7,
                ["", "增值税税额", "增值税免抵税额", "减免性质代码", "减免税（费）额", "本期应补（退）税额"],
            ),
            _row(6, 8, ["", "1", "2", "3", "4", "5=3-4"]),
            _row(6, 9, ["分项明细", "", "", "", "", ""]),
            _row(6, 10, ["城市维护建设税", "31,628.60", "0.05", "1,581.43", "0.00", "1,581.43"]),
        ],
    )
    result = ParseResult(
        status=ResultStatus.SUCCESS,
        pages=[PageContent(page_number=6, tables=[table])],
        entities=DocumentEntities(document_type="tax_return"),
    )

    bundle = build_community_bundle(seal_parse_result(result))
    payload = bundle.json_payload(bundle.semantic_payload())
    dataset = _dataset(payload, "tax_return_main")
    reading_table = next(item for item in payload["reading"]["tables"] if item["dataset_id"] == dataset["id"])
    fact_group = next(group for group in reading_table["row_groups"] if group.get("kind") == "table_fact_region")
    heading_group = next(group for group in reading_table["row_groups"] if group.get("title") == "分项明细")

    assert len(payload["datasets"]) == 1
    assert dataset["row_count"] == 1
    assert fact_group["start_record_id"] == dataset["rows"][0]["record_id"]
    assert heading_group["start_record_id"] == dataset["rows"][0]["record_id"]
    assert [(fact["label"], fact["raw"]) for fact in fact_group["facts"]] == [
        ("税（费）款所属时间", "2025年5月1日至2025年5月31日"),
        ("纳税人名称（公章）", "测试科技股份有限公司"),
        ("金额单位", "元（列至角分）"),
        ("本期是否适用优惠政策", "□ 是 ■ 否"),
        ("适用主体", "□ 个体工商户 □ 小型微利企业"),
        ("适用政策起止时间", "年 月 日 至 年 月 日"),
    ]
    assert fact_group["source_row_start"] == 2
    assert fact_group["source_row_end"] == 5
    assert fact_group["facts"][3]["source"] == {
        "page": 6,
        "page_range": [6, 6],
        "table_id": "pt_6_0",
        "physical_table_id": "pt_6_0",
        "table_row_index": 4,
        "source_row_index": 4,
        "source_cell_refs": [
            {"page": 6, "table_id": "pt_6_0", "row": 4, "col": 0, "field_name": "label"},
            {"page": 6, "table_id": "pt_6_0", "row": 4, "col": 2, "field_name": "raw"},
        ],
        "confidence": 1.0,
        "evidence_ids": ["ev:0006:0004:00", "ev:0006:0004:02"],
        "bbox": [0.0, 80.0, 300.0, 100.0],
    }
    basis_column = next(
        column
        for column in dataset["columns"]
        if any(band.get("value") == "计税（费）依据" for band in column.get("source_header_bands", []))
    )
    assert [band["value"] for band in basis_column["source_header_bands"] if band["role"] == "label"] == [
        "计税（费）依据",
        "增值税税额",
    ]
    enhanced = bundle.render_enhanced_markdown(bundle.semantic_payload())
    period_at = enhanced.index("**税（费）款所属时间：** 2025年5月1日至2025年5月31日")
    declaration_at = enhanced.index("**本期是否适用优惠政策：** □ 是 ■ 否")
    heading_at = enhanced.index("#### 分项明细")
    parent_header_at = enhanced.index("| 税（费）种 | 计税（费）依据 |")
    assert enhanced != bundle.render_markdown()
    assert "计税（费）依据 / 增值税税额（1）" not in enhanced
    assert "|  | 增值税税额 | 增值税免抵税额 | 减免性质代码 | 减免税（费）额 |" in enhanced
    assert "|  | 1 | 2 | 3 | 4 | 5=3-4 |" in enhanced
    assert "| 城市维护建设税 | 31,628.60 | 0.05 | 1,581.43 | 0.00 | 1,581.43 |" in enhanced
    assert period_at < declaration_at < heading_at < parent_header_at


def test_tax_return_does_not_promote_pre_header_question_without_choice_evidence() -> None:
    table = TableBlock(
        table_id="pt_1_0",
        page=1,
        headers=[],
        rows=[
            _row(1, 0, ["是否申报相关事项", "", "待确认", ""]),
            _row(1, 1, ["项目", "栏次", "本月数", "本年累计"]),
            _row(1, 2, ["销售额", "1", "100.00", "500.00"]),
        ],
    )
    result = ParseResult(
        status=ResultStatus.SUCCESS,
        pages=[PageContent(page_number=1, tables=[table])],
        entities=DocumentEntities(document_type="tax_return"),
    )

    payload = build_community_bundle(seal_parse_result(result)).json_payload()
    dataset = _dataset(payload, "tax_return_main")
    reading_table = next(item for item in payload["reading"]["tables"] if item["dataset_id"] == dataset["id"])

    assert dataset["row_count"] == 1
    assert not any(group.get("kind") == "table_fact_region" for group in reading_table.get("row_groups", []))


def test_tax_return_keeps_labeled_pre_header_context_without_checkbox() -> None:
    table = TableBlock(
        table_id="pt_1_0",
        page=1,
        headers=[],
        rows=[
            _row(1, 0, ["申报附表", "", "", ""]),
            _row(1, 1, ["税款所属期间：2025年5月1日至2025年5月31日", "", "", ""]),
            _row(1, 2, ["项目", "栏次", "本月数", "本年累计"]),
            _row(1, 3, ["销售额", "1", "100.00", "500.00"]),
        ],
    )
    result = ParseResult(
        status=ResultStatus.SUCCESS,
        pages=[PageContent(page_number=1, tables=[table])],
        entities=DocumentEntities(document_type="tax_return"),
    )

    payload = build_community_bundle(seal_parse_result(result)).json_payload()
    dataset = _dataset(payload, "tax_return_main")
    fact_group = next(group for group in dataset["row_groups"] if group.get("kind") == "table_fact_region")

    assert [(fact["label"], fact["raw"]) for fact in fact_group["facts"]] == [
        ("税款所属期间", "2025年5月1日至2025年5月31日")
    ]
    assert fact_group["source_row_start"] == fact_group["source_row_end"] == 1


def test_tax_return_prefers_clean_main_form_identity_over_high_confidence_seal_attachment() -> None:
    table = TableBlock(
        table_id="pt_1_0",
        page=1,
        headers=["项目", "栏次", "本月数", "本年累计"],
        rows=[
            _row(1, 0, ["项目", "栏次", "本月数", "本年累计"], row_type=RowType.HEADER),
            _row(1, 1, ["销售额", "1", "100.00", "500.00"]),
        ],
    )
    result = ParseResult(
        status=ResultStatus.SUCCESS,
        pages=[
            PageContent(
                page_number=1,
                texts=[
                    TextBlock(
                        content="纳税人名称:测试(上海)科技股份有限公司",
                        confidence=0.8,
                        bbox=[10.0, 10.0, 300.0, 30.0],
                        evidence_ids=["ev:main-form-name"],
                    )
                ],
                tables=[table],
            ),
            PageContent(
                page_number=2,
                key_values=[
                    KeyValuePair(
                        key="纳税人名称(公章)",
                        value="(公章) 测试(上海)科技股份有限公司",
                        confidence=1.0,
                        bbox=[20.0, 20.0, 320.0, 40.0],
                        evidence_ids=["ev:seal-attachment-name"],
                    )
                ],
            ),
        ],
        entities=DocumentEntities(document_type="tax_return"),
    )

    bundle = build_community_bundle(seal_parse_result(result))
    payload = bundle.json_payload(bundle.semantic_payload())
    document_items = {item["key"]: item for section in payload["sections"] for item in section["items"]}
    subject_name = document_items["subject_name"]

    assert subject_name["value"] == "测试(上海)科技股份有限公司"
    assert subject_name["raw"] == "测试(上海)科技股份有限公司"
    assert subject_name["source"]["page_range"] == [1, 1]
    assert subject_name["source"]["bbox"] == [10.0, 10.0, 300.0, 30.0]
    assert subject_name["source"]["evidence_ids"] == ["ev:main-form-name"]
    assert subject_name["confidence"] == 0.8
    assert subject_name["review"] == "manual_optional"


def test_tax_return_preserves_ordinal_formula_header_band_in_json_and_enhanced_markdown() -> None:
    table = TableBlock(
        table_id="pt_7_0",
        page=7,
        headers=[
            "减税性质代码及名称",
            "栏次",
            "期初余额",
            "本期发生额",
            "本期应抵减税额",
            "本期实际抵减税额",
            "期末余额",
        ],
        rows=[
            _row(7, 0, ["一、减税项目", "", "", "", "", "", ""]),
            _row(
                7,
                1,
                [
                    "减税性质代码及名称",
                    "栏次",
                    "期初余额",
                    "本期发生额",
                    "本期应抵减税额",
                    "本期实际抵减税额",
                    "期末余额",
                ],
                row_type=RowType.HEADER,
            ),
            _row(7, 2, ["", "", "1", "2", "3=1+2", "4≤3", "5=3-4"]),
            _row(7, 3, ["合计", "1", "0.00", "0.00", "0.00", "0.00", "0.00"]),
        ],
    )
    result = ParseResult(
        status=ResultStatus.SUCCESS,
        pages=[PageContent(page_number=7, tables=[table])],
        entities=DocumentEntities(document_type="tax_return"),
    )

    bundle = build_community_bundle(seal_parse_result(result))
    semantic = bundle.semantic_payload()
    payload = bundle.json_payload(semantic)
    dataset = _dataset(payload, "tax_return_main")
    eligible = next(column for column in dataset["columns"] if column["key"] == "current_period_eligible_deduction")

    assert dataset["row_count"] == 1
    assert eligible["source_header_bands"][-1] == {
        "level": 2,
        "role": "ordinal",
        "value": "3=1+2",
        "raw": "3=1+2",
        "confidence": 1.0,
        "source": {
            "page": 7,
            "table_id": "pt_7_0",
            "row": 2,
            "col": 4,
            "col_span": 1,
            "bbox": [400.0, 40.0, 500.0, 60.0],
            "evidence_ids": ["ev:0007:0002:04"],
        },
    }
    assert dataset["rows"][0]["raw"]["line_no"] == "1"
    enhanced = bundle.render_enhanced_markdown(semantic)
    assert enhanced != bundle.render_markdown()
    assert "本期应抵减税额（3=1+2）" not in enhanced
    assert "|  |  | 1 | 2 | 3=1+2 | 4≤3 | 5=3-4 |" in enhanced
    assert "| 合计 | 1 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |" in enhanced


def test_tax_return_uses_current_source_section_header_instead_of_previous_data_row_as_title() -> None:
    first_title = _row(3, 0, ["一、申报抵扣的进项税额", "", "", "", ""])
    first_title.cells[0].col_span = 5
    second_title = _row(3, 3, ["二、进项税额转出额", "", "", "", ""])
    second_title.cells[0].col_span = 5
    table = TableBlock(
        table_id="pt_3_0",
        page=3,
        headers=[],
        rows=[
            first_title,
            _row(3, 1, ["项目", "栏次", "份数", "金额", "税额"], row_type=RowType.HEADER),
            _row(3, 2, ["当期申报抵扣进项税额合计", "12=1+4+11", "7", "5,388,717.34", "517,254.02"]),
            second_title,
            _row(3, 4, ["项目", "栏次", "税额", "", ""], row_type=RowType.HEADER),
            _row(3, 5, ["本期进项税转出额", "13=14至23之和", "0.00", "", ""]),
        ],
    )
    result = ParseResult(
        status=ResultStatus.SUCCESS,
        pages=[
            PageContent(
                page_number=3,
                texts=[
                    TextBlock(content="增值税及附加税费申报表附列资料（二）"),
                    TextBlock(content="（本期进项税额明细）"),
                ],
                tables=[table],
            )
        ],
        entities=DocumentEntities(document_type="tax_return"),
    )

    bundle = build_community_bundle(seal_parse_result(result))
    payload = bundle.json_payload(bundle.semantic_payload())
    enhanced = bundle.render_enhanced_markdown(bundle.semantic_payload())

    assert [dataset["label"] for dataset in payload["datasets"]] == [
        "一、申报抵扣的进项税额",
        "二、进项税额转出额",
    ]
    assert all("5,388,717.34" not in dataset["label"] for dataset in payload["datasets"])
    assert "### 一、申报抵扣的进项税额" in enhanced
    assert "### 二、进项税额转出额" in enhanced
    assert "### 当期申报抵扣进项税额合计" not in enhanced
    assert "| 当期申报抵扣进项税额合计 | 12=1+4+11 | 7 | 5,388,717.34 | 517,254.02 |" in enhanced


def test_tax_main_invalid_line_and_corrupt_placeholders_require_review() -> None:
    table = TableBlock(
        table_id="pt_1_0",
        page=1,
        headers=[
            "项目",
            "栏次",
            "一般项目/本月数",
            "一般项目/本年累计",
            "即征即退项目/本月数",
            "即征即退项目/本年累计",
        ],
        rows=[
            _row(
                1,
                0,
                [
                    "项目",
                    "栏次",
                    "一般项目/本月数",
                    "一般项目/本年累计",
                    "即征即退项目/本月数",
                    "即征即退项目/本年累计",
                ],
                row_type=RowType.HEADER,
            ),
            _row(1, 1, ["销售额", "2-28-29", "100.00", "500.00", '"', "一"]),
            _row(1, 2, ["销项税额", "20", "13.00", "65.00", "二", ""]),
        ],
    )
    result = ParseResult(
        status=ResultStatus.SUCCESS,
        pages=[PageContent(page_number=1, tables=[table])],
        entities=DocumentEntities(document_type="tax_return"),
    )

    bundle = build_community_bundle(seal_parse_result(result))
    payload = bundle.json_payload(bundle.semantic_payload())
    dataset = _dataset(payload, "tax_return_main")

    assert any(warning["code"] == "TAX_LINE_NO_INVALID" for warning in payload["warnings"])
    assert any(warning["code"] == "TAX_LINE_SEQUENCE_GAP" for warning in payload["warnings"])
    assert sum(warning["code"] == "TAX_DECIMAL_TOKEN_INVALID" for warning in payload["warnings"]) == 3
    assert sum(warning["code"] == "TAX_DECIMAL_TOKEN_MISSING" for warning in payload["warnings"]) == 1
    assert all(row["review"]["required"] is True for row in dataset["rows"])


def test_tax_decimal_contract_applies_to_attachments_and_preserves_placeholders_for_reading() -> None:
    main_headers = ["项目", "栏次", "本月数"]
    attachment_headers = ["减税项目", "栏次", "本期发生额"]
    result = ParseResult(
        status=ResultStatus.SUCCESS,
        pages=[
            PageContent(
                page_number=1,
                tables=[
                    TableBlock(
                        table_id="pt_1_0",
                        page=1,
                        headers=main_headers,
                        rows=[
                            _row(1, 0, main_headers, row_type=RowType.HEADER),
                            _row(1, 1, ["销售额", "1", "100.00"]),
                        ],
                    ),
                    TableBlock(
                        table_id="pt_1_1",
                        page=1,
                        headers=attachment_headers,
                        rows=[
                            _row(1, 0, attachment_headers, row_type=RowType.HEADER),
                            _row(1, 1, ["合法占位", "1", "——"]),
                            _row(1, 2, ["异常识别", "2", "一"]),
                        ],
                    ),
                ],
            )
        ],
        entities=DocumentEntities(document_type="tax_return"),
    )

    bundle = build_community_bundle(seal_parse_result(result))
    payload = bundle.json_payload(bundle.semantic_payload())
    attachment = _dataset(payload, "tax_return_attachment_001")
    placeholder, invalid = attachment["rows"]

    assert placeholder["raw"]["current_period_amount"] == "——"
    assert placeholder["normalized"]["current_period_amount"] is None
    assert invalid["raw"]["current_period_amount"] == "一"
    assert invalid["normalized"]["current_period_amount"] is None
    assert "tax_decimal_token_invalid" in invalid["review"]["reasons"]
    assert any(
        warning["code"] == "TAX_DECIMAL_TOKEN_INVALID" and "dataset=tax_return_attachment_001" in warning["message"]
        for warning in payload["warnings"]
    )
    assert "——" in bundle.render_enhanced_markdown(bundle.semantic_payload())
    assert bundle.conservation_issues(payload=payload) == []
    invalid["normalized"]["current_period_amount"] = "—"
    assert any(
        issue.endswith("current_period_amount:normalized_decimal_invalid")
        for issue in bundle.conservation_issues(payload=payload)
    )


def test_tax_projector_does_not_invent_a_missing_placeholder_without_source_evidence() -> None:
    headers = [
        "项目",
        "栏次",
        "一般项目/本月数",
        "一般项目/本年累计",
        "一般项目/本月数",
        "即征即退项目/本年累计",
        "即征即退项目/本年累计",
    ]
    table = TableBlock(
        table_id="pt_1_0",
        page=1,
        headers=headers,
        rows=[
            _row(1, 0, headers, row_type=RowType.HEADER),
            _row(1, 1, ["免、抵、退办法出口销售额", "7", "0.00", "0.00", "—", "", ""]),
            _row(1, 2, ["免税销售额", "8", "0.00", "0.00", "—", "—", ""]),
            _row(1, 3, ["其中:免税货物销售额", "9", "0.00", "0.00", "—", "—", ""]),
        ],
    )
    result = ParseResult(
        status=ResultStatus.SUCCESS,
        pages=[PageContent(page_number=1, tables=[table])],
        entities=DocumentEntities(document_type="tax_return"),
    )

    bundle = build_community_bundle(seal_parse_result(result))
    payload = bundle.json_payload(bundle.semantic_payload())
    dataset = _dataset(payload, "tax_return_main")

    assert [column["key"] for column in dataset["columns"]][-4:] == [
        "general_current_month",
        "general_year_to_date",
        "immediate_refund_current_month",
        "immediate_refund_year_to_date",
    ]
    assert dataset["rows"][0]["raw"]["immediate_refund_year_to_date"] == ""
    assert "tax_decimal_token_missing" in dataset["rows"][0]["review"]["reasons"]
    assert any(warning["code"] == "TAX_DECIMAL_TOKEN_MISSING" for warning in payload["warnings"])


def test_tax_projector_preserves_source_labels_without_a_statutory_answer_table() -> None:
    headers = [
        "项目",
        "",
        "栏次",
        "一般项目/本月数",
        "一般项目/本年累计",
        "即征即退项目/本月数",
        "即征即退项目/本年累计",
    ]
    table = TableBlock(
        table_id="pt_1_0",
        page=1,
        headers=headers,
        rows=[
            _row(1, 0, headers, row_type=RowType.HEADER),
            _row(1, 1, ["", "来源项目甲", "1", "10.00", "20.00", "0.00", "0.00"]),
            _row(1, 2, ["源分组", "合法中文项目", "2", "0.05", "0.00", "0.00", "0.00"]),
            _row(1, 3, ["", "来源项目乙", "3", "0.00", "0.00", "0.00", "0.00"]),
        ],
    )
    result = ParseResult(
        status=ResultStatus.SUCCESS,
        pages=[PageContent(page_number=1, tables=[table])],
        entities=DocumentEntities(document_type="tax_return"),
    )

    bundle = build_community_bundle(seal_parse_result(result))
    payload = bundle.json_payload(bundle.semantic_payload())
    rows = _dataset(payload, "tax_return_main")["rows"]

    assert len(rows) == 3
    assert [row["raw"]["item_name"] for row in rows] == ["来源项目甲", "合法中文项目", "来源项目乙"]
    assert [row["raw"]["item_category"] for row in rows] == ["", "源分组", ""]
    assert rows[1]["raw"]["general_current_month"] == "0.05"
    assert rows[1]["normalized"]["general_current_month"] == "0.05"


def test_tax_projector_marks_an_invalid_amount_instead_of_guessing() -> None:
    headers = [
        "项目",
        "",
        "栏次",
        "一般项目/本月数",
        "一般项目/本年累计",
        "即征即退项目/本月数",
        "即征即退项目/本年累计",
    ]
    table = TableBlock(
        table_id="pt_1_0",
        page=1,
        headers=headers,
        rows=[
            _row(1, 0, headers, row_type=RowType.HEADER),
            _row(1, 1, ["", "来源项目甲", "1", "0.05", "0.00", "0.00", "0.00"]),
            _row(1, 2, ["", "来源项目乙", "2", "一", "0.00", "0.00", "0.00"]),
            _row(1, 3, ["", "来源项目丙", "3", "0.00", "0.00", "0.00", "0.00"]),
        ],
    )
    result = ParseResult(
        status=ResultStatus.SUCCESS,
        pages=[PageContent(page_number=1, tables=[table])],
        entities=DocumentEntities(document_type="tax_return"),
    )

    bundle = build_community_bundle(seal_parse_result(result))
    payload = bundle.json_payload(bundle.semantic_payload())
    rows = _dataset(payload, "tax_return_main")["rows"]
    invalid_row = next(row for row in rows if row["raw"]["line_no"] == "2")

    assert rows[0]["raw"]["general_current_month"] == "0.05"
    assert invalid_row["raw"]["general_current_month"] == "一"
    assert "tax_decimal_token_invalid" in invalid_row["review"]["reasons"]
    assert any(
        warning["code"] == "TAX_DECIMAL_TOKEN_INVALID" and "record=tax_return_main:r000002" in warning["message"]
        for warning in payload["warnings"]
    )


def test_cash_flow_arithmetic_mismatch_requires_review_without_repairing_source_value() -> None:
    table = TableBlock(
        table_id="pt_1_0",
        page=1,
        headers=["项目", "行次", "本年累计金额", "本月金额"],
        rows=[
            _row(1, 0, ["项目", "行次", "本年累计金额", "本月金额"], row_type=RowType.HEADER),
            _row(1, 1, ["经营活动产生的现金流量净额", "7", "100.00", "10.00"]),
            _row(1, 2, ["投资活动产生的现金流量净额", "13", "0.00", "0.00"]),
            _row(1, 3, ["筹资活动产生的现金流量净额", "19", "50.00", "5.00"]),
            _row(1, 4, ["四、现金净增加额", "20", "50.00", "15.00"]),
        ],
    )
    result = ParseResult(
        status=ResultStatus.SUCCESS,
        pages=[PageContent(page_number=1, tables=[table])],
        entities=DocumentEntities(document_type="cash_flow_statement"),
    )

    bundle = build_community_bundle(seal_parse_result(result))
    payload = bundle.json_payload(bundle.semantic_payload())
    dataset = _dataset(payload, "cash_flow_statement")

    assert dataset["row_count"] == 4
    assert dataset["rows"][-1]["raw"]["year_to_date_amount"] == "50.00"
    assert all("cash_flow_net_mismatch" in row["review"]["reasons"] for row in dataset["rows"])
    assert any("expected=150.00:actual=50.00" in warning["message"] for warning in payload["warnings"])


def test_income_statement_preserves_numbered_rows_with_blank_amounts() -> None:
    table = TableBlock(
        table_id="pt_1_0",
        page=1,
        headers=["项目", "行次", "本年累计金额", "本月金额"],
        rows=[
            _row(1, 0, ["项目", "行次", "本年累计金额", "本月金额"], row_type=RowType.HEADER),
            _row(1, 1, ["一、营业收入", "1", "", ""]),
            _row(1, 2, ["减：营业成本", "2", "50.00", "5.00"]),
        ],
    )
    result = ParseResult(
        status=ResultStatus.SUCCESS,
        pages=[PageContent(page_number=1, tables=[table])],
        entities=DocumentEntities(document_type="income_statement"),
    )

    bundle = build_community_bundle(seal_parse_result(result))
    payload = bundle.json_payload(bundle.semantic_payload())
    dataset = _dataset(payload, "income_statement")

    assert dataset["row_count"] == 2
    assert dataset["rows"][0]["raw"]["line_no"] == "1"
    assert dataset["rows"][0]["raw"]["year_to_date_amount"] == ""
    assert not any(warning["code"] == "FINANCIAL_LINE_SEQUENCE_GAP" for warning in payload["warnings"])


def test_tax_line_explanation_is_not_treated_as_ocr_corruption() -> None:
    table = TableBlock(
        table_id="pt_1_0",
        page=1,
        headers=["项目", "栏次", "一般项目/本月数", "一般项目/本年累计"],
        rows=[
            _row(1, 0, ["项目", "栏次", "一般项目/本月数", "一般项目/本年累计"], row_type=RowType.HEADER),
            _row(1, 1, ["应纳税额", "18(如17<11,则为17,否则为11)", "13.00", "65.00"]),
        ],
    )
    result = ParseResult(
        status=ResultStatus.SUCCESS,
        pages=[PageContent(page_number=1, tables=[table])],
        entities=DocumentEntities(document_type="tax_return"),
    )

    bundle = build_community_bundle(seal_parse_result(result))
    payload = bundle.json_payload(bundle.semantic_payload())

    assert not any(warning["code"] == "TAX_LINE_NO_INVALID" for warning in payload["warnings"])


def test_tax_projector_preserves_source_formula_instead_of_guessing_operators() -> None:
    table = TableBlock(
        table_id="pt_1_0",
        page=1,
        headers=["项目", "栏次", "一般项目/本月数", "一般项目/本年累计"],
        rows=[
            _row(1, 0, ["项目", "栏次", "一般项目/本月数", "一般项目/本年累计"], row_type=RowType.HEADER),
            _row(1, 1, ["应抵扣税额合计", "17-12-13-14-1516", "100.00", "500.00"]),
        ],
    )
    result = ParseResult(
        status=ResultStatus.SUCCESS,
        pages=[
            PageContent(
                page_number=1,
                texts=[
                    TextBlock(
                        content="纳税人名称:斯菲尔(土海)智能科技股份有限公司",
                        confidence=0.8,
                    ),
                    TextBlock(content="纳税人识别号:913100005515731558", confidence=0.99),
                ],
                tables=[table],
            ),
            PageContent(
                page_number=2,
                texts=[
                    TextBlock(
                        content="纳税人名称:斯菲尔(上海)智能科技股份有限公司",
                        confidence=0.99,
                    ),
                    TextBlock(content="纳税人识别号:913100005515731558", confidence=0.99),
                ],
            ),
        ],
        entities=DocumentEntities(document_type="tax_return"),
    )

    bundle = build_community_bundle(seal_parse_result(result))
    payload = bundle.json_payload(bundle.semantic_payload())
    row = _dataset(payload, "tax_return_main")["rows"][0]
    content_markdown = bundle.render_markdown()

    assert row["raw"]["line_no"] == "17-12-13-14-1516"
    assert not any("tax_formula_glyphs_recovered" in warning["message"] for warning in payload["warnings"])
    assert "17-12-13-14-1516" in content_markdown
    assert content_markdown.count("斯菲尔(土海)智能科技股份有限公司") == 1
    assert content_markdown.count("斯菲尔(上海)智能科技股份有限公司") == 1
    assert "tax_line_no_invalid" in row["review"]["reasons"]
    subject_name = next(
        item for section in payload["sections"] for item in section["items"] if item["key"] == "subject_name"
    )
    assert subject_name["review"] == "needs_review"
    assert any(warning["code"] == "TAX_SUBJECT_IDENTITY_CONFLICT" for warning in payload["warnings"])
    assert any(
        warning["code"] == "TAX_LINE_NO_INVALID" and "17-12-13-14-1516" in warning["message"]
        for warning in payload["warnings"]
    )


def test_tax_form_profile_normalizes_formula_only_after_full_signature_match() -> None:
    column_keys = [
        "item_name",
        "line_no",
        "general_current_month",
        "general_year_to_date",
        "immediate_refund_current_month",
        "immediate_refund_year_to_date",
    ]
    columns = [ColumnSpec(source_index=index, key=key, label=key) for index, key in enumerate(column_keys)]
    observed_lines = {
        1: "1",
        17: "17-12-13-14-1516",
        24: "24＝19+21－\n23",
        27: "27=28-29-30-31",
        32: "32=24-2526-27",
        38: "38=16-22-36-37",
    }
    records = [
        {
            "raw": {"line_no": observed},
            "canonical_raw": {"line_no": observed},
            "normalized": {"line_no": observed},
            "source": {
                "source_cell_refs": [{"page": 1, "table_id": "pt_1_0", "row": line, "col": 2, "field_name": "line_no"}]
            },
        }
        for line, observed in observed_lines.items()
    ]

    warnings = _normalize_profiled_line_formulas(
        records,
        columns=columns,
        form_title="增值税及附加税费申报表（一般纳税人适用）",
    )

    assert warnings == []
    assert records[1]["raw"]["line_no"] == "17=12+13-14-15+16"
    assert records[1]["canonical_raw"]["line_no"] == "17=12+13-14-15+16"
    assert records[1]["normalized"]["line_no"] == "17=12+13-14-15+16"
    assert records[2]["raw"]["line_no"] == "24＝19+21－\n23"
    assert records[2]["canonical_raw"]["line_no"] == "24＝19+21－\n23"
    assert records[2]["normalized"]["line_no"] == "24=19+21-23"
    assert records[3]["normalized"]["line_no"] == "27=28+29+30+31"
    assert records[4]["normalized"]["line_no"] == "32=24+25+26-27"
    assert records[5]["normalized"]["line_no"] == "38=16+22+36-37"
    assert "corrections" not in records[2]["source"]
    correction = records[1]["source"]["corrections"][0]
    assert correction == {
        "field": "line_no",
        "observed": "17-12-13-14-1516",
        "corrected": "17=12+13-14-15+16",
        "method": "tax_statutory_formula_profile",
        "confidence": 1.0,
        "source_refs": [{"page": 1, "table_id": "pt_1_0", "row": 17, "col": 2, "field_name": "line_no"}],
    }


def test_tax_full_profile_enhanced_uses_normalized_formula_and_source_amount_format() -> None:
    headers = [
        "项目",
        "",
        "栏次",
        "一般项目/本月数",
        "一般项目/本年累计",
        "即征即退项目/本月数",
        "即征即退项目/本年累计",
    ]
    lines = [
        ("销售额", "1", "1,234.50"),
        ("应抵扣税额合计", "17-12-13-14-1516", "2,345.60"),
        ("应纳税额合计", "24=19+21-23", "0.00"),
        ("本期已缴税额", "27=28-29-30-31", "0.00"),
        ("期末未缴税额", "32=24-2526-27", "0.00"),
        ("期末税额", "38=16-22-36-37", "0.00"),
    ]
    table = TableBlock(
        table_id="pt_1_0",
        page=1,
        headers=headers,
        rows=[
            _row(1, 0, headers, row_type=RowType.HEADER),
            *[
                _row(
                    1,
                    index,
                    ["一、销售额" if index == 1 else "", item, line, amount, amount, "0.00", "0.00"],
                )
                for index, (item, line, amount) in enumerate(lines, start=1)
            ],
        ],
    )
    result = ParseResult(
        status=ResultStatus.SUCCESS,
        pages=[
            PageContent(
                page_number=1,
                texts=[TextBlock(content="增值税及附加税费申报表（一般纳税人适用）")],
                tables=[table],
            )
        ],
        entities=DocumentEntities(document_type="tax_return"),
    )

    bundle = build_community_bundle(seal_parse_result(result))
    payload = bundle.json_payload(bundle.semantic_payload())
    enhanced = bundle.render_enhanced_markdown(bundle.semantic_payload())
    formula_row = _dataset(payload, "tax_return_main")["rows"][1]

    assert formula_row["raw"]["line_no"] == "17=12+13-14-15+16"
    assert formula_row["canonical_raw"]["line_no"] == "17=12+13-14-15+16"
    assert formula_row["normalized"]["line_no"] == "17=12+13-14-15+16"
    correction = formula_row["source"]["corrections"][0]
    assert correction["observed"] == "17-12-13-14-1516"
    assert correction["corrected"] == "17=12+13-14-15+16"
    assert correction["method"] == "tax_statutory_formula_profile"
    assert "17=12+13-14-15+16" in enhanced
    assert "17-12-13-14-1516" not in enhanced
    assert "1,234.50" in enhanced


def test_tax_full_profile_repairs_only_display_header_bands_and_preserves_observed_raw() -> None:
    keys = [
        "item_name",
        "line_no",
        "general_current_month",
        "general_year_to_date",
        "immediate_refund_current_month",
        "immediate_refund_year_to_date",
    ]
    columns = [ColumnSpec(source_index=index, key=key, label=key) for index, key in enumerate(keys)]
    records = [
        {"raw": {"line_no": line}}
        for line in ("1", "17-12-13-14-1516", "24=19+21-23", "27=28-29-30-31", "32=24-2526-27", "38=16-22-36-37")
    ]
    bands = {
        "general_current_month": [
            {
                "level": 1,
                "role": "label",
                "value": "般项目",
                "raw": "般项目",
                "source": {"page": 1, "table_id": "pt_1_0", "row": 2, "col": 2},
            }
        ],
        "general_year_to_date": [
            {
                "level": 1,
                "role": "label",
                "value": "般项目",
                "raw": "般项目",
                "source": {"page": 1, "table_id": "pt_1_0", "row": 2, "col": 2},
            }
        ],
        "immediate_refund_current_month": [],
        "immediate_refund_year_to_date": [
            {
                "level": 1,
                "role": "label",
                "value": "即征即退项目",
                "raw": "即征即退项目",
                "source": {"page": 1, "table_id": "pt_1_0", "row": 2, "col": 5},
            }
        ],
    }

    normalized = _normalize_profiled_tax_header_bands(
        bands,
        records=records,
        columns=columns,
        form_title="增值税及附加税费申报表（一般纳税人适用）",
    )

    assert normalized["general_current_month"][0]["value"] == "一般项目"
    assert normalized["general_current_month"][0]["raw"] == "一般项目"
    assert normalized["immediate_refund_current_month"][0]["value"] == "即征即退项目"
    assert normalized["immediate_refund_current_month"][0]["raw"] == ""


def test_tax_form_profile_rejects_a_formula_with_different_source_references() -> None:
    column_keys = [
        "item_name",
        "line_no",
        "general_current_month",
        "general_year_to_date",
        "immediate_refund_current_month",
        "immediate_refund_year_to_date",
    ]
    columns = [ColumnSpec(source_index=index, key=key, label=key) for index, key in enumerate(column_keys)]
    observed_lines = [
        "1",
        "17=12+13-14-15+18",
        "24=19+21-23",
        "27=28+29+30+31",
        "32=24+25+26-27",
        "38=16+22+36-37",
    ]
    records = [{"raw": {"line_no": observed}, "normalized": {"line_no": observed}} for observed in observed_lines]

    warnings = _normalize_profiled_line_formulas(
        records,
        columns=columns,
        form_title="增值税及附加税费申报表（一般纳税人适用）",
    )

    assert warnings == ["precision:tax_formula_digit_mismatch:line=17"]
    assert records[1]["normalized"]["line_no"] == "17=12+13-14-15+18"
    assert records[1]["review"] == {"required": True, "reasons": ["tax_formula_digit_mismatch"]}
    assert "corrections" not in records[1].get("source", {})


def test_additional_tax_form_restores_statutory_columns_and_removes_repeated_grid_strokes() -> None:
    group_header = [
        "税(费)种",
        "计税(费)依据",
        "",
        "",
        "税(费)率(%)",
        "本期应纳税(费)额",
        "本期减免税(费)额",
        "",
        "小微企业减免政策",
        "",
        "试点建设培育产教融合型企业",
        "",
        "本期已缴税(费)额",
        "本期应补(退)税(费)额",
    ]
    detail_header = [
        "",
        "增值税税额",
        "增值税免抵税额",
        "留抵退税本期扣除额",
        "",
        "",
        "减免性质代码",
        "减免税(费)额",
        "减征比例(%)",
        "减征额",
        "减免性质代码",
        "本期抵免金额",
        "",
        "",
    ]
    data = [
        [
            "城市维护建设税",
            "31,628.60",
            "0.00",
            "0.00",
            "0.05",
            "1,581.43",
            "",
            "0.00",
            "0.00",
            "0.00",
            "|",
            "0.00",
            "0.00",
            "1,581.43",
        ],
        [
            "教育费附加",
            "31,628.60",
            "0.00",
            "0.00",
            "0.03",
            "948.86",
            "",
            "0.00",
            "0.00",
            "0.00",
            "|",
            "0.00",
            "0.00",
            "948.86",
        ],
        [
            "地方教育附加",
            "31,628.60",
            "0.00",
            "0.00",
            "0.02",
            "632.57",
            "",
            "0.00",
            "0.00",
            "0.00",
            "|",
            "0.00",
            "0.00",
            "632.57",
        ],
        ["合计", "--", "--", "--", "--", "3,162.86", "--", "0.00", "--", "0.00", "--", "0.00", "0.00", "3,162.86"],
    ]
    table = TableBlock(
        table_id="pt_6_0",
        page=6,
        rows=[
            _row(6, 0, group_header, row_type=RowType.HEADER),
            _row(6, 1, detail_header, row_type=RowType.HEADER),
            _row(
                6,
                2,
                ["", "1", "2", "3", "4", "5=(1+2-3)*4", "6", "7", "8", "9=(5-7)*8", "10", "11", "12", "13=5-7-9-11-12"],
                row_type=RowType.HEADER,
            ),
            *[_row(6, index + 3, values) for index, values in enumerate(data)],
        ],
    )
    result = ParseResult(
        status=ResultStatus.SUCCESS,
        pages=[
            PageContent(
                page_number=6,
                texts=[TextBlock(content="增值税及附加税费申报表附列资料(五)")],
                tables=[table],
            )
        ],
        entities=DocumentEntities(document_type="tax_return"),
    )

    bundle = build_community_bundle(seal_parse_result(result))
    payload = bundle.json_payload(bundle.semantic_payload())
    dataset = payload["datasets"][0]

    assert [column["key"] for column in dataset["columns"]] == [
        "tax_or_fee_type",
        "vat_tax_amount",
        "vat_exemption_credit_amount",
        "current_period_refund_deduction",
        "tax_or_fee_rate",
        "current_period_tax_payable",
        "relief_code",
        "relief_amount",
        "reduction_rate",
        "reduction_amount",
        "relief_code_2",
        "current_period_credit_amount",
        "current_period_tax_paid",
        "current_period_tax_due_or_refund",
    ]
    assert dataset["rows"][0]["raw"]["tax_or_fee_rate"] == "0.05"
    assert dataset["rows"][0]["raw"]["current_period_tax_payable"] == "1,581.43"
    assert dataset["rows"][0]["normalized"]["current_period_tax_payable"] == "1581.43"
    assert dataset["rows"][0]["raw"]["relief_code_2"] == ""
    column_types = {column["key"]: column["type"] for column in dataset["columns"]}
    assert column_types["current_period_refund_deduction"] == "decimal"
    assert column_types["tax_or_fee_rate"] == "decimal"
    assert column_types["current_period_tax_payable"] == "decimal"
    assert column_types["relief_amount"] == "decimal"
    assert column_types["reduction_rate"] == "decimal"
    assert column_types["current_period_tax_due_or_refund"] == "decimal"
    assert not any("tax_blank_code_gridline_removed" in warning["message"] for warning in payload["warnings"])
    correction = dataset["rows"][0]["source"]["corrections"][0]
    assert correction["method"] == "tax_repeated_gridline_artifact"
    assert correction["observed"] == "|"
    assert correction["corrected"] == ""


def test_tax_enhanced_markdown_uses_source_title_and_readable_source_text() -> None:
    table = TableBlock(
        table_id="pt_1_0",
        page=1,
        headers=["项目", "栏次", "一般项目/本月数", "一般项目/本年累计"],
        rows=[
            _row(
                1,
                0,
                ["项目", "栏次", "一般项目/本月数", "一般项目/本年累计"],
                row_type=RowType.HEADER,
            ),
            _row(1, 1, ["①本期缴纳\n上期应纳税额", "30", "1,234.50", "2,345.60"]),
        ],
    )
    result = ParseResult(
        status=ResultStatus.SUCCESS,
        pages=[
            PageContent(
                page_number=1,
                texts=[
                    TextBlock(content="增值税及附加税费申报表"),
                    TextBlock(content="（一般纳税人适用）"),
                ],
                tables=[table],
            )
        ],
        entities=DocumentEntities(document_type="tax_return"),
    )

    bundle = build_community_bundle(seal_parse_result(result))
    payload = bundle.json_payload(bundle.semantic_payload())
    dataset = _dataset(payload, "tax_return_main")
    enhanced = bundle.render_enhanced_markdown()

    assert dataset["label"] == "增值税及附加税费申报表（一般纳税人适用）"
    assert dataset["rows"][0]["raw"]["item_category"] == "①本期缴纳\n上期应纳税额"
    assert dataset["rows"][0]["normalized"]["item_category"] == "①本期缴纳上期应纳税额"
    assert dataset["rows"][0]["normalized"]["general_current_month"] == "1234.50"
    assert "## 增值税及附加税费申报表（一般纳税人适用）" in enhanced
    assert "①本期缴纳 上期应纳税额" in enhanced
    assert "| ①本期缴纳 上期应纳税额 | 30 | 1,234.50 | 2,345.60 |" in enhanced
    assert enhanced != bundle.render_markdown()
    assert "classification source" not in enhanced
    assert "doc type hint source" not in enhanced
    assert "**文档类型:**" not in enhanced


def test_tax_form_splits_persistent_schema_drift_into_a_separate_dataset() -> None:
    headers = [
        "税(费)种",
        "增值税税额",
        "税(费)率",
        "本期应纳税(费)额",
        "减免性质代码",
        "减免税(费)额",
    ]
    table = TableBlock(
        table_id="pt_1_0",
        page=1,
        headers=headers,
        rows=[
            _row(1, 0, headers, row_type=RowType.HEADER),
            _row(1, 1, ["", "1", "2", "3=1+2", "4", "5"]),
            _row(1, 2, ["城建税", "100.00", "0.05", "5.00", "", "0.00"]),
            _row(1, 3, ["教育费附加", "100.00", "0.03", "3.00", "", "0.00"]),
            _row(1, 4, ["地方教育附加", "100.00", "0.02", "2.00", "", "0.00"]),
            _row(1, 5, ["合计", "--", "--", "10.00", "--", "0.00"]),
            _row(1, 6, ["本期是否适用抵免政策", "", "□是 ■否", "当期新增投资额", "5", "0.00"]),
            _row(1, 7, ["", "", "", "上期留抵可抵免金额", "6", "0.00"]),
            _row(1, 8, ["留抵退税额使用情况", "", "", "结转下期可抵免金额", "7", "0.00"]),
        ],
    )
    result = ParseResult(
        status=ResultStatus.SUCCESS,
        pages=[
            PageContent(
                page_number=1,
                texts=[TextBlock(content="增值税及附加税费申报表(一般纳税人适用)附列资料(五)")],
                tables=[table],
            )
        ],
        entities=DocumentEntities(document_type="tax_return"),
    )

    bundle = build_community_bundle(seal_parse_result(result))
    payload = bundle.json_payload(bundle.semantic_payload())

    assert [dataset["row_count"] for dataset in payload["datasets"]] == [4, 3]
    trailing = _dataset(payload, "tax_return_attachment_001")
    assert [column["key"] for column in trailing["columns"]] == [
        "section_name",
        "applicability_status",
        "item_name",
        "line_no",
        "amount",
    ]
    assert trailing["rows"][0]["raw"]["line_no"] == "5"
    assert "本期是否适用抵免政策" in trailing["label"]
    assert all("source_header_bands" not in column for column in trailing["columns"])


def test_tax_project_and_line_span_gets_stable_source_columns() -> None:
    table = TableBlock(
        table_id="pt_1_0",
        page=1,
        headers=["项目及栏次", "项目及栏次", "项目及栏次", "项目及栏次", "销售额", "税额"],
        rows=[
            _row(
                1,
                0,
                ["项目及栏次", "项目及栏次", "项目及栏次", "项目及栏次", "销售额", "税额"],
                row_type=RowType.HEADER,
            ),
            _row(1, 1, ["一般计税", "全部项目", "13%税率项目", "1", "100.00", "13.00"]),
        ],
    )
    result = ParseResult(
        status=ResultStatus.SUCCESS,
        pages=[PageContent(page_number=1, tables=[table])],
        entities=DocumentEntities(document_type="tax_return"),
    )

    bundle = build_community_bundle(seal_parse_result(result))
    payload = bundle.json_payload(bundle.semantic_payload())
    dataset = _dataset(payload, "tax_return_main")

    assert [column["key"] for column in dataset["columns"]] == [
        "item_category",
        "item_subcategory",
        "item_name",
        "line_no",
        "sales_amount",
        "tax_amount",
    ]


def test_tax_service_deduction_header_bleed_uses_stable_key() -> None:
    headers = [
        "项目及栏次",
        "项目及栏次",
        "项目及栏次",
        "项目及栏次",
        "合计/价税合计",
        "服务、不动产和无形资产扣除项目本期实际扣除金额/价税合计",
    ]
    table = TableBlock(
        table_id="pt_1_0",
        page=1,
        headers=headers,
        rows=[
            _row(1, 0, ["增值税及附加税费申报表附列资料（一）", "", "", "", "", ""]),
            _row(1, 1, ["（本期销售情况明细）", "", "", "", "", ""]),
            _row(1, 2, headers, row_type=RowType.HEADER),
            _row(1, 3, ["一般计税", "全部项目", "9%税率项目", "1", "109.00", "9.00"]),
        ],
    )
    result = ParseResult(
        status=ResultStatus.SUCCESS,
        pages=[PageContent(page_number=1, tables=[table])],
        entities=DocumentEntities(document_type="tax_return"),
    )

    bundle = build_community_bundle(seal_parse_result(result))
    payload = bundle.json_payload(bundle.semantic_payload())
    dataset = _dataset(payload, "tax_return_main")

    assert dataset["label"] == "增值税及附加税费申报表附列资料（一）（本期销售情况明细）"
    assert [column["key"] for column in dataset["columns"]][-2:] == [
        "total_amount_with_tax",
        "service_deduction_actual_amount",
    ]
    assert dataset["columns"][-1]["label"] == "服务、不动产和无形资产扣除项目本期实际扣除金额"
    assert dataset["rows"][0]["normalized"]["service_deduction_actual_amount"] == "9.00"


def test_financial_plugin_registration_does_not_replace_existing_domains() -> None:
    registry = PluginRegistry()

    assert registry.get_projector("tax_return", "community").domain_name == "tax_return"
    assert registry.get_projector("income_statement", "community").domain_name == "income_statement"
    assert registry.get_projector("bank_statement", "community").domain_name == "bank_statement"
    assert registry.get_projector("wechat_payment", "community").domain_name == "wechat_payment"
    assert registry.get_projector("generic", "community").domain_name == "generic"
