"""Unit tests for generic community output v2.1 — type detection, standardization, identity extraction."""

from __future__ import annotations

import pytest

from docmirror.models.entities.parse_result import (
    CellValue,
    DocumentEntities,
    KeyValuePair,
    LogicalTable,
    PageContent,
    ParseResult,
    ResultStatus,
    RowProvenance,
    RowType,
    TableBlock,
    TableRow,
    TextBlock,
    TextLevel,
)
from docmirror.plugins._base.generic_community_adapter import (
    _GENERIC_WARNING,
    _build_normalized_record,
    _collect_sections,
    _collect_table_descriptors,
    _collect_table_records,
    _collect_text_key_values,
    _extract_identities,
    _infer_generic_type,
    _infer_table_column_types,
    _select_generic_tables,
    _standardize_value,
    _type_detect_column,
    derive_generic_projection,
)


def build_generic_community_output(result, document_type, text=""):
    """Expose ProjectionData contents in the compact shape used by collector tests."""
    patch = derive_generic_projection(result, document_type, text)
    reserved = {"field_details", "summary", "normalized_fields", "field_schema", "columns", "identities"}
    return {
        "data": {
            "fields": {key: value for key, value in patch.domain_facts.items() if key not in reserved},
            "field_metadata": patch.domain_facts["field_details"],
            "records": patch.datasets.get("records", []),
            "tables": _collect_table_descriptors(result, _select_generic_tables(result)),
        }
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  _type_detect_column
# ═══════════════════════════════════════════════════════════════════════════════


class TestTypeDetectColumn:
    def test_detect_date(self):
        col_type, conf = _type_detect_column(["2024-01-01", "2024-02-15", "2024-03-20"])
        assert col_type == "date"
        assert conf >= 0.6

    def test_detect_amount(self):
        col_type, conf = _type_detect_column(["1,000.00", "500.50", "-50.00", "¥100.00"])
        assert col_type == "amount"
        assert conf >= 0.6

    def test_detect_phone(self):
        col_type, conf = _type_detect_column(["13800138000", "13912345678", "18600001111"])
        assert col_type == "phone"
        assert conf >= 0.6

    def test_detect_id_number(self):
        col_type, conf = _type_detect_column(["110101199001011234", "32010219850706321X", "440305200112123456"])
        assert col_type == "id_number"
        assert conf >= 0.6

    def test_detect_email(self):
        col_type, conf = _type_detect_column(["user@example.com", "test@company.cn", "admin@test.org"])
        assert col_type == "email"
        assert conf >= 0.6

    def test_detect_percentage(self):
        col_type, conf = _type_detect_column(["5%", "12.5%", "99.9%", "0%"])
        assert col_type == "percentage"
        assert conf >= 0.6

    def test_text_fallback(self):
        col_type, conf = _type_detect_column(["报销差旅费", "办公用品采购", "交通费", "餐费补贴"])
        assert col_type == "text"

    def test_mixed_column_fallback_to_text(self):
        """Mixed types (dates + amounts) should fall back to text."""
        col_type, conf = _type_detect_column(["2024-01-01", "1,000.00", "摘要说明", "张三"])
        assert col_type == "text"
        assert conf < 0.6

    def test_empty_values(self):
        col_type, conf = _type_detect_column([])
        assert col_type == "text"
        assert conf == 0.0

    def test_dash_values_skipped(self):
        col_type, conf = _type_detect_column(["-", "—", "2024-01-01", "2024-02-01"])
        assert col_type == "date"
        assert conf >= 0.6

    def test_detect_account_number(self):
        col_type, conf = _type_detect_column(["6222021234567890123", "6228480012345678901", "6217001234567890123"])
        assert col_type == "account"
        assert conf >= 0.6


class TestGenericSemanticTypeInference:
    def test_invalid_calendar_dates_fall_back_to_text(self):
        assert _infer_generic_type("日期", ["2024-99-99", "2024-02-31", "2024-13-01"]) == ("text", 0.0)

    def test_compact_dates_use_header_semantics(self):
        assert _infer_generic_type("交易日期", ["20240101", "20240201", "20240301"]) == ("date", 1.0)

    def test_formatted_phone_values_are_detected(self):
        assert _infer_generic_type("联系电话", ["138-0013-8000", "139 1234 5678", "+86 18600001111"]) == ("phone", 1.0)

    def test_header_disambiguates_large_amount_from_account(self):
        values = ["100000000000000", "200000000000000", "300000000000000"]
        assert _infer_generic_type("总金额", values)[0] == "amount"
        assert _infer_generic_type("银行账号", values)[0] == "account"
        assert _infer_generic_type("col_0", values)[0] == "text"

    def test_identifier_header_preserves_leading_zeroes_as_text(self):
        assert _infer_generic_type("订单号", ["00001234"])[0] == "text"

    def test_identifier_semantics_win_over_amount_keyword(self):
        col_type, confidence = _infer_generic_type("费用编号", ["000123"])
        assert col_type == "text"
        assert confidence >= 0.9

    def test_strong_amount_header_keeps_parseable_minorities_typed(self):
        values = ["1,000.00", "2,000.00", "3,000.00", "年末余额", "账面价值", "项目", "计提理由"]
        col_type, confidence = _infer_generic_type("年末余额", values)
        assert col_type == "amount"
        assert 0.4 <= confidence < 0.6


# ═══════════════════════════════════════════════════════════════════════════════
#  _standardize_value
# ═══════════════════════════════════════════════════════════════════════════════


class TestStandardizeValue:
    def test_amount_float(self):
        result = _standardize_value("1,000.00", "amount")
        assert isinstance(result, dict)
        assert result["value"] == 1000.0
        assert "currency" not in result

    def test_amount_negative(self):
        result = _standardize_value("-500.00", "amount")
        assert isinstance(result, dict)
        assert result["value"] == -500.0

    def test_amount_with_currency_symbol(self):
        result = _standardize_value("¥100.00", "amount")
        assert isinstance(result, dict)
        assert result["value"] == 100.0
        assert result["currency"] == "CNY"

    def test_amount_usd(self):
        result = _standardize_value("$50.00", "amount")
        assert isinstance(result, dict)
        assert result["value"] == 50.0
        assert result["currency"] == "USD"

    def test_amount_uses_only_explicit_currency_context(self):
        assert _standardize_value("50.00", "amount", currency_hint="CNY") == {
            "value": 50.0,
            "currency": "CNY",
        }

    def test_amount_non_numeric(self):
        result = _standardize_value("N/A", "amount")
        assert result is None

    @pytest.mark.parametrize("placeholder", ["-", "—", "–", "--"])
    def test_amount_placeholder_is_typed_null(self, placeholder):
        assert _standardize_value(placeholder, "amount") is None

    def test_amount_repairs_unambiguous_scanned_thousands_separator(self):
        assert _standardize_value("127,500.000.00", "amount", currency_hint="CNY") == {
            "value": 127500000.0,
            "currency": "CNY",
        }

    def test_date_iso(self):
        result = _standardize_value("2024-01-01", "date")
        assert isinstance(result, dict)
        assert result["value"] == "2024-01-01"

    def test_date_chinese_format(self):
        result = _standardize_value("2024年01月01日", "date")
        assert isinstance(result, dict)
        assert "2024" in result["value"]

    def test_invalid_date_is_preserved_as_raw_text(self):
        assert _standardize_value("2024-99-99", "date") == "2024-99-99"

    def test_compact_date_is_normalized(self):
        assert _standardize_value("20240101", "date") == {"value": "2024-01-01"}

    def test_percentage(self):
        result = _standardize_value("12.5%", "percentage")
        assert isinstance(result, dict)
        assert result["value"] == 12.5
        assert result["unit"] == "%"

    def test_phone_digits(self):
        result = _standardize_value("138-0013-8000", "phone")
        assert isinstance(result, dict)
        assert result["value"] == "13800138000"

    def test_text_passthrough(self):
        result = _standardize_value("报销差旅费", "text")
        assert result == "报销差旅费"


# ═══════════════════════════════════════════════════════════════════════════════
#  _extract_identities
# ═══════════════════════════════════════════════════════════════════════════════


class TestExtractIdentities:
    def test_extract_name_by_key(self):
        fields = {"客户名称": "张三", "金额": "1,000"}
        ids = _extract_identities(fields)
        assert "name" in ids
        assert ids["name"]["key"] == "客户名称"
        assert ids["name"]["value"] == "张三"

    def test_extract_phone_by_value(self):
        fields = {"联系电话": "13800138000"}
        ids = _extract_identities(fields)
        assert "phone" in ids
        assert ids["phone"]["value"] == "13800138000"

    def test_extract_id_by_key(self):
        fields = {"身份证号": "110101199001011234"}
        ids = _extract_identities(fields)
        assert "id_number" in ids
        assert ids["id_number"]["value"] == "110101199001011234"

    @pytest.mark.parametrize(
        "key,value",
        [
            ("护照号码", "E12345678"),
            ("驾驶证号", "420106198801011234"),
            ("社会保障号码", "110101199001011234"),
        ],
    )
    def test_extract_document_identity_by_confirmed_key(self, key, value):
        identity = _extract_identities({key: value})["id_number"]
        assert identity["key"] == key
        assert identity["value"] == value

    def test_extract_account_by_key(self):
        fields = {"银行账号": "6222021234567890123"}
        ids = _extract_identities(fields)
        assert "account" in ids
        assert ids["account"]["key"] == "银行账号"

    def test_extract_address(self):
        fields = {"住址": "北京市朝阳区建国路88号"}
        ids = _extract_identities(fields)
        assert "address" in ids
        assert ids["address"]["value"] == "北京市朝阳区建国路88号"

    def test_english_field_names(self):
        fields = {"Name": "Alice", "Phone": "13800138000"}
        ids = _extract_identities(fields)
        assert "name" in ids
        assert ids["name"]["value"] == "Alice"

    def test_empty_fields(self):
        ids = _extract_identities({})
        assert ids == {}

    def test_value_pattern_fallback(self):
        """When key doesn't match, phone pattern should still detect."""
        fields = {"联系方式": "13912345678"}
        ids = _extract_identities(fields)
        assert "phone" in ids
        assert ids["phone"]["value"] == "13912345678"

    def test_short_id_alias_does_not_match_paid_field(self):
        assert "id_number" not in _extract_identities({"paid_amount": "110101199001011234"})

    def test_formatted_phone_value_fallback(self):
        assert _extract_identities({"联系方式": "138-0013-8000"})["phone"]["value"] == "138-0013-8000"


# ═══════════════════════════════════════════════════════════════════════════════
#  _build_normalized_record
# ═══════════════════════════════════════════════════════════════════════════════


class TestBuildNormalizedRecord:
    def test_normalized_record_with_types(self):
        raw = {"日期": "2024-01-01", "金额": "1,000.00", "摘要": "报销差旅费"}
        col_types = {
            "日期": {"type": "date", "confidence": 0.95, "null_ratio": 0.0},
            "金额": {"type": "amount", "confidence": 0.90, "null_ratio": 0.0},
            "摘要": {"type": "text", "confidence": 0.80, "null_ratio": 0.0},
        }
        normalized = _build_normalized_record(raw, col_types)
        assert isinstance(normalized["日期"], dict)
        assert normalized["金额"]["value"] == 1000.0
        assert normalized["摘要"] == "报销差旅费"

    def test_normalized_unknown_column_type(self):
        raw = {"未知字段": "随便什么内容"}
        col_types = {}
        normalized = _build_normalized_record(raw, col_types)
        assert normalized["未知字段"] == "随便什么内容"

    def test_normalized_empty_raw(self):
        normalized = _build_normalized_record({}, {})
        assert normalized == {}


class TestGenericTextAndTableRecovery:
    def test_multiple_logical_tables_keep_independent_datasets(self):
        result = ParseResult(
            pages=[
                PageContent(
                    page_number=1,
                    tables=[
                        TableBlock(
                            table_id="sales",
                            headers=["项目", "本月数"],
                            rows=[TableRow(cells=[CellValue(text="销售额"), CellValue(text="10.00")])],
                        ),
                        TableBlock(
                            table_id="rates",
                            headers=["税种", "税率"],
                            rows=[TableRow(cells=[CellValue(text="增值税"), CellValue(text="13%")])],
                        ),
                    ],
                )
            ]
        )

        projection = derive_generic_projection(result, "tax_return")

        assert list(projection.datasets) == ["table_001", "table_002"]
        assert projection.datasets["table_001"][0]["raw"] == {"项目": "销售额", "本月数": "10.00"}
        assert projection.datasets["table_002"][0]["raw"] == {"税种": "增值税", "税率": "13%"}

    def test_generic_projection_splits_explicit_repeated_header_bands(self):
        table = TableBlock(
            table_id="combined_form",
            headers=["表名", "附列资料", "", ""],
            rows=[
                TableRow(
                    cells=[
                        CellValue(text="序号"),
                        CellValue(text="抵减项目"),
                        CellValue(text="期初余额"),
                        CellValue(text="本期发生额"),
                    ]
                ),
                TableRow(
                    cells=[
                        CellValue(text="1"),
                        CellValue(text="销售额"),
                        CellValue(text="10.00"),
                        CellValue(text="1.00"),
                    ]
                ),
                TableRow(
                    cells=[
                        CellValue(text="序号"),
                        CellValue(text="加计抵减项目"),
                        CellValue(text="期初余额"),
                        CellValue(text="本期可抵减额"),
                    ]
                ),
                TableRow(
                    cells=[
                        CellValue(text="2"),
                        CellValue(text="当期计提"),
                        CellValue(text="20.00"),
                        CellValue(text="2.00"),
                    ]
                ),
            ],
        )

        projection = derive_generic_projection(
            ParseResult(pages=[PageContent(page_number=1, tables=[table])]), "tax_return"
        )

        assert list(projection.datasets) == ["table_001", "table_002"]
        assert projection.datasets["table_001"][0]["raw"] == {
            "序号": "1",
            "抵减项目": "销售额",
            "期初余额": "10.00",
            "本期发生额": "1.00",
        }
        assert projection.datasets["table_002"][0]["raw"] == {
            "序号": "2",
            "加计抵减项目": "当期计提",
            "期初余额": "20.00",
            "本期可抵减额": "2.00",
        }

    def test_explicit_grouped_parent_headers_consume_textual_child_header_row(self):
        table = TableBlock(
            table_id="deferred_tax_liability",
            headers=["项目", "期末余额", "", "期初余额", ""],
            rows=[
                TableRow(
                    cells=[
                        CellValue(),
                        CellValue(text="应纳税暂时性差异"),
                        CellValue(text="递延所得税负债"),
                        CellValue(text="应纳税暂时性差异"),
                        CellValue(text="递延所得税负债"),
                    ]
                ),
                TableRow(
                    cells=[
                        CellValue(text="使用权资产税会差异"),
                        CellValue(text="775,858.27"),
                        CellValue(text="116,378.74"),
                        CellValue(text="67,014.02"),
                        CellValue(text="16,753.50"),
                    ]
                ),
            ],
        )

        records = _collect_table_records(
            ParseResult(pages=[PageContent(page_number=39, tables=[table])]),
            allow_embedded_headers=True,
        )

        assert [record["raw"] for record in records] == [
            {
                "项目": "使用权资产税会差异",
                "期末余额/应纳税暂时性差异": "775,858.27",
                "期末余额/递延所得税负债": "116,378.74",
                "期初余额/应纳税暂时性差异": "67,014.02",
                "期初余额/递延所得税负债": "16,753.50",
            }
        ]

    def test_adjacent_grouped_table_replaces_child_labels_without_emitting_header_row(self):
        table = TableBlock(
            table_id="deferred_tax_tables",
            headers=[
                "项目",
                "期末余额/可抵扣暂时性差异",
                "期末余额/递延所得税资产",
                "期初余额/可抵扣暂时性差异",
                "期初余额/递延所得税资产",
            ],
            rows=[
                TableRow(
                    cells=[
                        CellValue(text="信用减值准备"),
                        CellValue(text="10,599,372.91"),
                        CellValue(text="1,589,905.94"),
                        CellValue(text="5,840,764.64"),
                        CellValue(text="1,460,191.16"),
                    ]
                ),
                TableRow(
                    cells=[
                        CellValue(),
                        CellValue(text="应纳税暂时性差异"),
                        CellValue(text="递延所得税负债"),
                        CellValue(text="应纳税暂时性差异"),
                        CellValue(text="递延所得税负债"),
                    ]
                ),
                TableRow(
                    cells=[
                        CellValue(text="使用权资产税会差异"),
                        CellValue(text="775,858.27"),
                        CellValue(text="116,378.74"),
                        CellValue(text="67,014.02"),
                        CellValue(text="16,753.50"),
                    ]
                ),
            ],
        )

        records = _collect_table_records(
            ParseResult(pages=[PageContent(page_number=39, tables=[table])]),
            allow_embedded_headers=True,
        )

        assert len(records) == 2
        assert records[0]["raw"]["期末余额/递延所得税资产"] == "1,589,905.94"
        assert records[1]["raw"] == {
            "项目": "使用权资产税会差异",
            "期末余额/应纳税暂时性差异": "775,858.27",
            "期末余额/递延所得税负债": "116,378.74",
            "期初余额/应纳税暂时性差异": "67,014.02",
            "期初余额/递延所得税负债": "16,753.50",
        }

    def test_parent_only_source_headers_split_two_adjacent_grouped_child_schemas(self):
        table = TableBlock(
            table_id="deferred_tax_parent_only",
            headers=["项目", "期末余额", "", "期初余额", ""],
            rows=[
                TableRow(
                    cells=[
                        CellValue(),
                        CellValue(text="可抵扣暂时性差异"),
                        CellValue(text="递延所得税资产"),
                        CellValue(text="可抵扣暂时性差异"),
                        CellValue(text="递延所得税资产"),
                    ]
                ),
                TableRow(
                    cells=[
                        CellValue(text="信用减值准备"),
                        CellValue(text="10,599,372.91"),
                        CellValue(text="1,589,905.94"),
                        CellValue(text="5,840,764.64"),
                        CellValue(text="1,460,191.16"),
                    ]
                ),
                TableRow(
                    cells=[
                        CellValue(),
                        CellValue(text="应纳税暂时性差异"),
                        CellValue(text="递延所得税负债"),
                        CellValue(text="应纳税暂时性差异"),
                        CellValue(text="递延所得税负债"),
                    ]
                ),
                TableRow(
                    cells=[
                        CellValue(text="使用权资产税会差异"),
                        CellValue(text="775,858.27"),
                        CellValue(text="116,378.74"),
                        CellValue(text="67,014.02"),
                        CellValue(text="16,753.50"),
                    ]
                ),
            ],
        )

        records = _collect_table_records(
            ParseResult(pages=[PageContent(page_number=39, tables=[table])]),
            allow_embedded_headers=True,
        )

        assert len(records) == 2
        assert list(records[0]["raw"]) == [
            "项目",
            "期末余额/可抵扣暂时性差异",
            "期末余额/递延所得税资产",
            "期初余额/可抵扣暂时性差异",
            "期初余额/递延所得税资产",
        ]
        assert list(records[1]["raw"]) == [
            "项目",
            "期末余额/应纳税暂时性差异",
            "期末余额/递延所得税负债",
            "期初余额/应纳税暂时性差异",
            "期初余额/递延所得税负债",
        ]

        projection = derive_generic_projection(
            ParseResult(pages=[PageContent(page_number=39, tables=[table])]), "audit_report"
        )

        assert projection.datasets["table_002"][0]["normalized"] == {
            "项目": "使用权资产税会差异",
            "期末余额/应纳税暂时性差异": {"value": 775858.27},
            "期末余额/递延所得税负债": {"value": 116378.74},
            "期初余额/应纳税暂时性差异": {"value": 67014.02},
            "期初余额/递延所得税负债": {"value": 16753.5},
        }

    def test_explicit_deepest_headers_merge_with_stacked_parent_rows(self):
        table = TableBlock(
            table_id="equity_changes",
            headers=["项目", "", "优先股", "永续债", "其他", "", "", "", "", "", "", ""],
            rows=[
                TableRow(
                    cells=[
                        CellValue(text="项目", row_span=3),
                        CellValue(text="本期金额", col_span=11),
                        *[CellValue() for _ in range(10)],
                    ]
                ),
                TableRow(
                    cells=[
                        CellValue(),
                        CellValue(text="实收资本（或股本）", row_span=2),
                        CellValue(text="其他权益工具", col_span=3),
                        CellValue(),
                        CellValue(),
                        CellValue(text="资本公积", row_span=2),
                        CellValue(text="减：库存股", row_span=2),
                        CellValue(text="其他综合收益", row_span=2),
                        CellValue(text="专项储备", row_span=2),
                        CellValue(text="盈余公积", row_span=2),
                        CellValue(text="未分配利润", row_span=2),
                        CellValue(text="所有者权益合计", row_span=2),
                    ]
                ),
                TableRow(
                    cells=[
                        CellValue(),
                        CellValue(),
                        CellValue(text="优先股"),
                        CellValue(text="永续债"),
                        CellValue(text="其他"),
                        *[CellValue() for _ in range(7)],
                    ]
                ),
                TableRow(
                    cells=[
                        CellValue(text="一、上年期末余额"),
                        CellValue(text="27,366,313.00"),
                        CellValue(),
                        CellValue(),
                        CellValue(),
                        CellValue(text="12,144,181.02"),
                        CellValue(),
                        CellValue(),
                        CellValue(),
                        CellValue(text="2,784,842.64"),
                        CellValue(text="10,846,294.04"),
                        CellValue(text="53,141,630.70"),
                    ]
                ),
            ],
        )

        records = _collect_table_records(
            ParseResult(pages=[PageContent(page_number=10, tables=[table])]),
            allow_embedded_headers=True,
        )

        assert len(records) == 1
        assert records[0]["raw"] == {
            "项目": "一、上年期末余额",
            "本期金额/实收资本(或股本)": "27,366,313.00",
            "本期金额/其他权益工具/优先股": "",
            "本期金额/其他权益工具/永续债": "",
            "本期金额/其他权益工具/其他": "",
            "本期金额/资本公积": "12,144,181.02",
            "本期金额/减:库存股": "",
            "本期金额/其他综合收益": "",
            "本期金额/专项储备": "",
            "本期金额/盈余公积": "2,784,842.64",
            "本期金额/未分配利润": "10,846,294.04",
            "本期金额/所有者权益合计": "53,141,630.70",
        }

    def test_embedded_header_scan_reaches_adjacent_table_after_twenty_rows(self):
        first_rows = [
            TableRow(
                cells=[
                    CellValue(text=f"项目{index}"),
                    CellValue(text=f"{index}.00"),
                    CellValue(text=f"{index + 1}.00"),
                ]
            )
            for index in range(1, 22)
        ]
        table = TableBlock(
            table_id="adjacent_tables",
            headers=["项目", "期末余额", "期初余额"],
            rows=first_rows
            + [
                TableRow(
                    cells=[
                        CellValue(text="项目"),
                        CellValue(text="本期发生额"),
                        CellValue(text="上期发生额"),
                    ]
                ),
                TableRow(
                    cells=[
                        CellValue(text="管理费用"),
                        CellValue(text="10.00"),
                        CellValue(text="9.00"),
                    ]
                ),
            ],
        )

        records = _collect_table_records(
            ParseResult(pages=[PageContent(page_number=43, tables=[table])]),
            allow_embedded_headers=True,
        )

        assert len(records) == 22
        assert records[20]["raw"] == {"项目": "项目21", "期末余额": "21.00", "期初余额": "22.00"}
        assert records[21]["raw"] == {"项目": "管理费用", "本期发生额": "10.00", "上期发生额": "9.00"}

    def test_generic_projection_uses_positional_columns_after_persistent_schema_drift(self):
        table = TableBlock(
            table_id="combined_form",
            headers=["项目", "本期应纳税额", "本期已缴税额"],
            rows=[
                TableRow(cells=[CellValue(text="项目甲"), CellValue(text="10.00"), CellValue(text="1.00")]),
                TableRow(cells=[CellValue(text="项目乙"), CellValue(text="20.00"), CellValue(text="2.00")]),
                TableRow(cells=[CellValue(text="项目丙"), CellValue(text="30.00"), CellValue(text="3.00")]),
                TableRow(cells=[CellValue(text="合计"), CellValue(text="60.00"), CellValue(text="6.00")]),
                TableRow(
                    cells=[CellValue(text="政策适用情况"), CellValue(text="当期新增投资额"), CellValue(text="0.00")]
                ),
                TableRow(cells=[CellValue(), CellValue(text="上期留抵可抵免金额"), CellValue(text="0.00")]),
                TableRow(cells=[CellValue(), CellValue(text="结转下期可抵免金额"), CellValue(text="0.00")]),
            ],
        )

        projection = derive_generic_projection(
            ParseResult(pages=[PageContent(page_number=1, tables=[table])]), "tax_return"
        )

        assert list(projection.datasets) == ["table_001", "table_002"]
        assert projection.datasets["table_001"][-1]["raw"] == {
            "项目": "合计",
            "本期应纳税额": "60.00",
            "本期已缴税额": "6.00",
        }
        assert projection.datasets["table_002"][0]["raw"] == {
            "col_0": "政策适用情况",
            "col_1": "当期新增投资额",
            "col_2": "0.00",
        }

    def test_formula_ordinals_do_not_trigger_schema_drift(self):
        table = TableBlock(
            table_id="return",
            headers=["项目", "栏次", "本月数"],
            rows=[
                TableRow(cells=[CellValue(text=f"项目{index}"), CellValue(text=str(index)), CellValue(text="0.00")])
                for index in range(1, 5)
            ]
            + [
                TableRow(cells=[CellValue(text="期末余额"), CellValue(text="32=24+25-27"), CellValue(text="0.00")]),
                TableRow(cells=[CellValue(text="欠缴税额"), CellValue(text="33=25+26-27"), CellValue(text="0.00")]),
                TableRow(cells=[CellValue(text="应补税额"), CellValue(text="34＝24-28-29"), CellValue(text="0.00")]),
            ],
        )

        projection = derive_generic_projection(
            ParseResult(pages=[PageContent(page_number=1, tables=[table])]), "tax_return"
        )

        assert list(projection.datasets) == ["records"]
        assert projection.datasets["records"][-1]["raw"]["栏次"] == "34＝24-28-29"

    def test_projection_does_not_trim_rows_by_sequence_values(self):
        table = TableBlock(
            table_id="return",
            headers=["项目", "栏次", "本月数"],
            rows=[
                TableRow(cells=[CellValue(text="纳税人信息"), CellValue(), CellValue()]),
                TableRow(cells=[CellValue(text="销售额"), CellValue(text="1"), CellValue(text="10.00")]),
                TableRow(cells=[CellValue(text="销项税额"), CellValue(text="2"), CellValue(text="1.30")]),
                TableRow(cells=[CellValue(text="应纳税额"), CellValue(text="3"), CellValue(text="1.30")]),
                TableRow(cells=[CellValue(text="其中甲项"), CellValue(text="3a"), CellValue(text="0.80")]),
                TableRow(cells=[CellValue(text="其中乙项"), CellValue(text="3b"), CellValue(text="0.50")]),
                TableRow(cells=[CellValue(text="合计"), CellValue(), CellValue(text="13.10")]),
                TableRow(cells=[CellValue(text="经办人：张三"), CellValue(), CellValue()]),
            ],
        )

        records = _collect_table_records(ParseResult(pages=[PageContent(page_number=1, tables=[table])]))

        assert records[0]["raw"]["项目"] == "纳税人信息"
        assert records[-1]["raw"]["项目"] == "经办人：张三"

    def test_projection_does_not_invent_an_isolated_sequence_value(self):
        table = TableBlock(
            table_id="return",
            headers=["项目", "栏次", "金额"],
            rows=[
                TableRow(cells=[CellValue(text="项目一"), CellValue(), CellValue(text="10.00")]),
                TableRow(cells=[CellValue(text="项目二"), CellValue(text="2"), CellValue(text="20.00")]),
                TableRow(cells=[CellValue(text="项目三"), CellValue(text="3"), CellValue(text="30.00")]),
            ],
        )

        projection = derive_generic_projection(
            ParseResult(pages=[PageContent(page_number=1, tables=[table])]), "tax_return"
        )

        assert [row["raw"]["栏次"] for row in projection.datasets["records"]] == ["", "2", "3"]

    def test_projection_does_not_replace_canonical_headers_from_embedded_rows(self):
        table = TableBlock(
            table_id="return",
            headers=["纳税人名称", "测试公司", "法定代表人", "张三"],
            rows=[
                TableRow(
                    cells=[
                        CellValue(text="开户银行"),
                        CellValue(text="测试银行"),
                        CellValue(text="注册地址"),
                        CellValue(text="上海"),
                    ]
                ),
                TableRow(
                    cells=[
                        CellValue(text="项目"),
                        CellValue(text="栏次"),
                        CellValue(text="本月数"),
                        CellValue(text="本年累计"),
                    ]
                ),
                TableRow(
                    cells=[CellValue(text="销售额"), CellValue(), CellValue(text="10.00"), CellValue(text="20.00")]
                ),
                TableRow(
                    cells=[CellValue(text="税额"), CellValue(text="2"), CellValue(text="1.30"), CellValue(text="2.60")]
                ),
                TableRow(
                    cells=[
                        CellValue(text="调整额"),
                        CellValue(text="3"),
                        CellValue(text="0.00"),
                        CellValue(text="0.00"),
                    ]
                ),
                TableRow(
                    cells=[
                        CellValue(text="合计"),
                        CellValue(text="4"),
                        CellValue(text="11.30"),
                        CellValue(text="22.60"),
                    ]
                ),
                TableRow(cells=[CellValue(text="受理人"), CellValue(), CellValue(), CellValue()]),
            ],
        )

        records = _collect_table_records(ParseResult(pages=[PageContent(page_number=1, tables=[table])]))

        assert list(records[0]["raw"]) == ["纳税人名称", "测试公司", "法定代表人", "张三"]
        assert records[0]["raw"]["纳税人名称"] == "开户银行"
        assert records[1]["raw"]["纳税人名称"] == "项目"

    def test_section_title_after_header_is_preserved_as_first_record(self):
        table = TableBlock(
            table_id="cash_flow",
            rows=[
                TableRow(
                    cells=[
                        CellValue(text="项目"),
                        CellValue(text="行次"),
                        CellValue(text="本月金额"),
                        CellValue(text="本年累计金额"),
                    ]
                ),
                TableRow(cells=[CellValue(text="一、经营活动产生的现金流量"), CellValue(), CellValue(), CellValue()]),
                TableRow(
                    cells=[
                        CellValue(text="销售商品收到的现金"),
                        CellValue(text="1"),
                        CellValue(text="10.00"),
                        CellValue(text="20.00"),
                    ]
                ),
            ],
        )

        records = _collect_table_records(ParseResult(pages=[PageContent(page_number=1, tables=[table])]))

        assert records[0]["raw"] == {
            "项目": "一、经营活动产生的现金流量",
            "行次": "",
            "本月金额": "",
            "本年累计金额": "",
        }

    def test_adjacent_headerless_table_does_not_inherit_schema_without_merge_evidence(self):
        first = TableBlock(
            table_id="pt_1_0",
            headers=["项目", "栏次", "金额"],
            rows=[
                TableRow(cells=[CellValue(text="销售额"), CellValue(text="1"), CellValue(text="10.00")]),
                TableRow(cells=[CellValue(text="税额"), CellValue(text="2"), CellValue(text="1.30")]),
                TableRow(cells=[CellValue(text="合计"), CellValue(text="3"), CellValue(text="11.30")]),
            ],
        )
        continuation = TableBlock(
            table_id="pt_2_0",
            rows=[
                TableRow(cells=[CellValue(text="附加税"), CellValue(text="4"), CellValue(text="0.50")]),
                TableRow(cells=[CellValue(text="城建税"), CellValue(text="5"), CellValue(text="0.30")]),
                TableRow(cells=[CellValue(text="教育费附加"), CellValue(text="6"), CellValue(text="0.20")]),
            ],
        )
        result = ParseResult(
            pages=[
                PageContent(page_number=1, tables=[first]),
                PageContent(page_number=2, tables=[continuation]),
            ]
        )

        records = _collect_table_records(result)

        assert {record["source"]["table_id"] for record in records} == {"pt_1_0", "pt_2_0"}
        continuation_records = [record for record in records if record["source"]["table_id"] == "pt_2_0"]
        assert continuation_records
        assert "栏次" not in continuation_records[0]["raw"]

    def test_source_text_does_not_guess_placeholder_headers(self):
        table = TableBlock(
            table_id="return",
            headers=["项目", "col_1", "金额", "col_3"],
            rows=[
                TableRow(cells=[CellValue(text="销售额"), CellValue(text="1"), CellValue(text="10.00"), CellValue()]),
                TableRow(cells=[CellValue(text="税额"), CellValue(text="2"), CellValue(text="1.30"), CellValue()]),
                TableRow(cells=[CellValue(text="合计"), CellValue(text="3"), CellValue(text="11.30"), CellValue()]),
            ],
        )
        result = ParseResult(
            pages=[
                PageContent(
                    page_number=1,
                    texts=[TextBlock(content="项目 栏次 金额")],
                    tables=[table],
                )
            ]
        )

        projection = derive_generic_projection(result, "tax_return")

        assert list(projection.datasets["records"][0]["raw"]) == ["项目", "col_1", "金额", "col_3"]

    def test_explicit_non_data_row_types_do_not_become_records(self):
        table = TableBlock(
            table_id="typed_rows",
            headers=["项目", "金额"],
            rows=[
                TableRow(
                    row_type=RowType.HEADER,
                    cells=[CellValue(text="项目"), CellValue(text="金额")],
                ),
                TableRow(cells=[CellValue(text="收入"), CellValue(text="10.00")]),
                TableRow(row_type=RowType.SEPARATOR, cells=[CellValue(text="---"), CellValue()]),
            ],
        )

        records = _collect_table_records(ParseResult(pages=[PageContent(page_number=1, tables=[table])]))

        assert [record["raw"] for record in records] == [{"项目": "收入", "金额": "10.00"}]

    def test_column_ordinal_band_does_not_become_a_business_record(self):
        table = TableBlock(
            table_id="form",
            headers=["序号", "抵减项目", "期初余额", "本期发生额", "期末余额"],
            rows=[
                TableRow(
                    cells=[
                        CellValue(),
                        CellValue(),
                        CellValue(text="1"),
                        CellValue(text="2"),
                        CellValue(text="3=1+2"),
                    ]
                ),
                TableRow(
                    cells=[
                        CellValue(text="1"),
                        CellValue(text="测试项目"),
                        CellValue(text="10.00"),
                        CellValue(text="2.00"),
                        CellValue(text="12.00"),
                    ]
                ),
            ],
        )

        records = _collect_table_records(ParseResult(pages=[PageContent(page_number=1, tables=[table])]))

        assert len(records) == 1
        assert records[0]["raw"]["序号"] == "1"

    def test_generic_records_retain_cell_level_source_refs(self):
        table = TableBlock(
            table_id="pt_2_0",
            headers=["项目", "金额"],
            rows=[
                TableRow(
                    cells=[
                        CellValue(
                            text="收入",
                            source_cell_refs=[{"page": 2, "table_id": "pt_2_0", "row": 5, "col": 0}],
                        ),
                        CellValue(
                            text="10.00",
                            source_cell_refs=[{"page": 2, "table_id": "pt_2_0", "row": 5, "col": 1}],
                        ),
                    ]
                )
            ],
        )

        record = _collect_table_records(ParseResult(pages=[PageContent(page_number=2, tables=[table])]))[0]

        assert record["source"]["source_cell_refs"] == [
            {"page": 2, "table_id": "pt_2_0", "row": 5, "col": 0, "field_name": "项目"},
            {"page": 2, "table_id": "pt_2_0", "row": 5, "col": 1, "field_name": "金额"},
        ]

    @pytest.mark.parametrize("line", ("12:30", "08：45", "第1章：总则", "第十二条：付款条件"))
    def test_non_kv_lines_are_rejected(self, line: str):
        assert _collect_text_key_values(line) == {}

    def test_valid_ratio_and_note_kv_are_preserved(self):
        assert _collect_text_key_values("比例：1:2\n备注：差旅报销") == {"比例": "1:2", "备注": "差旅报销"}

    def test_prose_and_markdown_table_lines_are_not_kv_fields(self):
        text = "\n".join(
            [
                "法定代表人：许水均;",
                "统一社会信用代码：91330109MA27XQ7P70;",
                "为：以摊余成本计量的金融资产;以公允价值计量的金融资产;",
                "差额,除：按照借款费用资本化处理",
                "一揽子交易进行会计处理：1这些交易同时订立;2这些交易整体实现商业结果",
                "| 减：坏账准备 | 68,333,689.66 | 54,767,811.77 |",
            ]
        )

        assert _collect_text_key_values(text) == {
            "法定代表人": "许水均",
            "统一社会信用代码": "91330109MA27XQ7P70",
        }

    def test_decimal_scalar_is_not_treated_as_numbered_list(self):
        assert _collect_text_key_values("估值：123.45") == {"估值": "123.45"}

    def test_body_numbered_headings_are_recovered_with_page_sources(self):
        result = ParseResult(
            pages=[
                PageContent(
                    page_number=2,
                    texts=[
                        TextBlock(
                            content="一、 审计意见\n我们审计了相关财务报表。\n二、 形成审计意见的基础",
                            bbox=[10, 20, 500, 300],
                        )
                    ],
                ),
                PageContent(
                    page_number=49,
                    texts=[TextBlock(content="六、 合并财务报表项目注释\n1、 货币资金")],
                ),
            ]
        )

        sections = _collect_sections(result)

        assert [item["title"] for item in sections] == [
            "一、审计意见",
            "二、形成审计意见的基础",
            "六、合并财务报表项目注释",
            "1、货币资金",
        ]
        assert sections[0]["page_start"] == 2
        assert sections[0]["bbox"] == [10.0, 20.0, 500.0, 300.0]

    def test_table_aligned_numbered_rows_are_not_recovered_as_sections(self):
        result = ParseResult(
            pages=[
                PageContent(
                    page_number=59,
                    texts=[
                        TextBlock(content="3、本年减少金额", bbox=[81, 118, 160, 130]),
                        TextBlock(content="四、账面价值", bbox=[81, 180, 144, 191]),
                        TextBlock(content="11、", bbox=[102, 248, 123, 260]),
                        TextBlock(content="无形资产", bbox=[128, 248, 181, 260]),
                        TextBlock(content="12、", bbox=[102, 288, 123, 300]),
                        TextBlock(content="长期待摊费用", bbox=[128, 288, 202, 300]),
                        TextBlock(content="1、收入确认和计量所采用的会计政策", bbox=[102, 330, 300, 342]),
                    ],
                )
            ]
        )

        sections = _collect_sections(result)

        assert [item["title"] for item in sections] == [
            "11、无形资产",
            "12、长期待摊费用",
            "1、收入确认和计量所采用的会计政策",
        ]

    def test_monotonic_table_headings_fill_outline_without_metric_rows(self):
        table = TableBlock(
            table_id="page_wide_table",
            rows=[
                TableRow(cells=[CellValue(text="2、应收账款", bbox=[102, 220, 170, 232])]),
                TableRow(cells=[CellValue(text="3、本年减少金额", bbox=[81, 250, 170, 262])]),
                TableRow(cells=[CellValue(text="4、预付款项", bbox=[102, 280, 170, 292])]),
                TableRow(
                    cells=[
                        CellValue(text="6、", bbox=[102, 326, 123, 338]),
                        CellValue(text="存货", bbox=[124, 326, 150, 338]),
                        CellValue(),
                    ]
                ),
            ],
        )
        result = ParseResult(
            pages=[
                PageContent(
                    page_number=49,
                    texts=[
                        TextBlock(content="六、合并财务报表项目注释", bbox=[102, 120, 230, 132]),
                        TextBlock(content="1、", bbox=[102, 180, 123, 192]),
                        TextBlock(content="货币资金", bbox=[128, 180, 181, 192]),
                        TextBlock(content="5、", bbox=[102, 320, 123, 332]),
                        TextBlock(content="其他应收款", bbox=[128, 320, 191, 332]),
                        TextBlock(content="16、", bbox=[102, 338, 128, 350]),
                        TextBlock(content="短期借款", bbox=[133, 338, 186, 350]),
                    ],
                    tables=[
                        table,
                        TableBlock(
                            table_id="later_page_grid",
                            rows=[
                                TableRow(cells=[CellValue(text="四、一年内到期", bbox=[81, 350, 170, 362])]),
                                TableRow(cells=[CellValue(text="6、短期带薪缺勤", bbox=[81, 380, 170, 392])]),
                            ],
                        ),
                    ],
                )
            ]
        )

        sections = _collect_sections(result)

        assert [item["title"] for item in sections] == [
            "六、合并财务报表项目注释",
            "1、货币资金",
            "2、应收账款",
            "4、预付款项",
            "5、其他应收款",
            "6、存货",
            "16、短期借款",
        ]

    def test_text_kv_metadata_is_mapped_back_to_page_evidence(self):
        text = "统一社会信用代码：91330109MA27XQ7P70;"
        result = ParseResult(
            status=ResultStatus.SUCCESS,
            pages=[
                PageContent(
                    page_number=3,
                    texts=[
                        TextBlock(
                            content=text,
                            confidence=0.96,
                            bbox=[10, 20, 300, 40],
                            evidence_ids=["ev:text:3"],
                        )
                    ],
                )
            ],
            entities=DocumentEntities(document_type="audit_report"),
        )

        out = build_generic_community_output(result, "audit_report", text)

        assert out["data"]["fields"]["统一社会信用代码"] == "91330109MA27XQ7P70"
        assert out["data"]["field_metadata"]["统一社会信用代码"] == {
            "source": "canonical_text",
            "page": 3,
            "confidence": 0.96,
            "bbox": [10.0, 20.0, 300.0, 40.0],
            "evidence_ids": ["ev:text:3"],
        }

    def test_ocr_generic_fields_drop_noise_and_duplicate_label_variants(self):
        result = ParseResult(
            pages=[
                PageContent(
                    page_number=1,
                    page_mode="scanned_ocr",
                    texts=[TextBlock(content="审计报告", evidence_ids=["ocr:p0001:0001"])],
                )
            ],
            entities=DocumentEntities(
                document_type="audit_report",
                domain_specific={
                    "编制单位": "杭州华英新塘羽绒制品有限公司",
                    "缤制单位": "杭州华英新塘羽绒制品有限公司",
                    "综制单位": "杭州华英新塘羽绒制品有限公司",
                    "法定代表人": "许水均",
                    "主管会计工作负责人": "不",
                    "查报告": "办理企业",
                    "按照应收金额计量的政府补助应同时符合以下条件": "1、应收补助款已经确认",
                    "国家企业信用信息公示系统网址": "http://www.gsxt.gov.cn",
                },
            ),
        )

        out = build_generic_community_output(result, "audit_report", "审计报告")
        fields = out["data"]["fields"]

        assert fields["编制单位"] == "杭州华英新塘羽绒制品有限公司"
        assert fields["法定代表人"] == "许水均"
        assert fields["国家企业信用信息公示系统网址"] == "http://www.gsxt.gov.cn"
        assert {"缤制单位", "综制单位", "主管会计工作负责人", "查报告"}.isdisjoint(fields)
        assert not any("政府补助" in key for key in fields)

    def test_ocr_generic_fields_drop_statement_clauses_and_redundant_unit(self):
        result = ParseResult(
            pages=[PageContent(page_number=1, page_mode="scanned_ocr")],
            entities=DocumentEntities(
                document_type="audit_report",
                domain_specific={
                    "金额单位": "人民币元",
                    "单位": "元",
                    "调整年初未分配利润明细": "由于会计政策变更，影响年初未分配利润560,601.33元。",
                    "调整后年初未分配利润 加": "本年归属于母公司所有者的净利润 -24,693,753.54",
                    "3年末现金及现金等价物余额 其中": "受限现金18,401,847.82",
                },
            ),
        )

        out = build_generic_community_output(result, "audit_report", "")

        assert out["data"]["fields"] == {"金额单位": "人民币元"}

    def test_ocr_long_field_joins_only_adjacent_continuation_lines(self):
        result = ParseResult(
            pages=[
                PageContent(
                    page_number=18,
                    page_mode="scanned_ocr",
                    texts=[
                        TextBlock(
                            content="经营范围:一般项目:羽毛(绒)及制品制造;服装制",
                            bbox=[109, 378.5, 515, 393.5],
                            evidence_ids=["ocr:p0018:0015"],
                        ),
                        TextBlock(content="造;服饰制造;电子", bbox=[89, 399.5, 514.5, 416.5]),
                        TextBlock(content="产品销售。", bbox=[90, 422.5, 513.5, 438]),
                        TextBlock(content="二、财务报表的编制基础", bbox=[108, 510.5, 218.5, 523.5]),
                    ],
                )
            ],
            entities=DocumentEntities(
                document_type="audit_report",
                domain_specific={"经营范围": "一般项目:羽毛(绒)及制品制造;服装制"},
            ),
        )

        out = build_generic_community_output(result, "audit_report", "")

        assert out["data"]["fields"]["经营范围"] == ("一般项目:羽毛(绒)及制品制造;服装制造;服饰制造;电子产品销售。")
        assert out["data"]["field_metadata"]["经营范围"]["page"] == 18

    def test_audit_report_number_is_recovered_without_replacing_qr_number(self):
        result = ParseResult(
            pages=[PageContent(page_number=1)],
            entities=DocumentEntities(
                document_type="audit_report",
                domain_specific={"报告编号": "京2491UTVQSC"},
            ),
        )

        out = build_generic_community_output(
            result,
            "audit_report",
            "杭州华英新塘羽绒制品有限公司\n亚会审字（2024）第01310141号",
        )

        assert out["data"]["fields"]["报告编号"] == "京2491UTVQSC"
        assert out["data"]["fields"]["审计报告文号"] == "亚会审字(2024)第01310141号"

    def test_duplicate_and_empty_headers_preserve_every_cell(self):
        table = TableBlock(
            table_id="dup_headers",
            headers=["金额", "金额", ""],
            rows=[TableRow(cells=[CellValue(text="1"), CellValue(text="2"), CellValue(text="3")])],
        )
        result = ParseResult(pages=[PageContent(page_number=1, tables=[table])])
        records = _collect_table_records(result)
        projection = derive_generic_projection(result, "tax_return")

        assert records[0]["raw"] == {"金额": "1", "金额_2": "2", "col_2": "3"}
        assert records[0]["source"]["header_repaired"] is True
        assert projection.semantic["dataset_column_order"]["records"] == ["金额", "金额_2", "col_2"]

    def test_empty_trailing_synthetic_column_is_not_projected(self):
        table = TableBlock(
            table_id="trailing_empty",
            extraction_layer="scanned_image_line_grid",
            metadata={"source": "scanned_bordered_table_reconstructor"},
            rows=[
                TableRow(
                    cells=[
                        CellValue(text="项目"),
                        CellValue(text="栏次"),
                        CellValue(text="本月数"),
                        CellValue(text=""),
                    ]
                ),
                TableRow(
                    cells=[
                        CellValue(text="销售额"),
                        CellValue(text="1"),
                        CellValue(text="100.00"),
                        CellValue(text=""),
                    ]
                ),
            ],
        )

        records = _collect_table_records(
            ParseResult(pages=[PageContent(page_number=1, tables=[table])]),
            allow_embedded_headers=True,
        )

        assert records[0]["raw"] == {"项目": "销售额", "栏次": "1", "本月数": "100.00"}

    def test_layout_spacing_inside_chinese_headers_is_normalized_for_records(self):
        table = TableBlock(
            table_id="native_table",
            headers=["项 目", "年末余额", "年初余额"],
            rows=[
                TableRow(
                    cells=[
                        CellValue(text="银行存款"),
                        CellValue(text="64,822,045.96"),
                        CellValue(text="18,410,772.82"),
                    ]
                )
            ],
        )

        records = _collect_table_records(ParseResult(pages=[PageContent(page_number=49, tables=[table])]))

        assert records[0]["raw"] == {
            "项目": "银行存款",
            "年末余额": "64,822,045.96",
            "年初余额": "18,410,772.82",
        }
        assert "header_repaired" not in records[0]["source"]

    def test_scanned_multirow_headers_are_promoted_and_removed_from_records(self):
        table = TableBlock(
            table_id="note_table",
            extraction_layer="scanned_image_line_grid",
            metadata={"source": "scanned_bordered_table_reconstructor"},
            rows=[
                TableRow(
                    cells=[
                        CellValue(text="项目", col_span=1),
                        CellValue(text="年末余额", col_span=3),
                        CellValue(),
                        CellValue(),
                    ]
                ),
                TableRow(
                    cells=[
                        CellValue(),
                        CellValue(text="账面余额"),
                        CellValue(text="存货跌价准备"),
                        CellValue(text="账面价值"),
                    ]
                ),
                TableRow(
                    cells=[
                        CellValue(text="原材料"),
                        CellValue(text="1,297,676.15"),
                        CellValue(),
                        CellValue(text="1,297,676.15"),
                    ]
                ),
            ],
        )

        result = ParseResult(pages=[PageContent(page_number=55, tables=[table])])
        records = _collect_table_records(result)
        out = build_generic_community_output(result, "audit_report", "")

        assert len(records) == 1
        assert records[0]["raw"] == {
            "项目": "原材料",
            "年末余额/账面余额": "1,297,676.15",
            "年末余额/存货跌价准备": "",
            "年末余额/账面价值": "1,297,676.15",
        }
        assert out["data"]["tables"][0]["headers"] == [
            "项目",
            "年末余额/账面余额",
            "年末余额/存货跌价准备",
            "年末余额/账面价值",
        ]
        assert "header_repaired" not in out["data"]["tables"][0]

    def test_scanned_investor_header_labels_are_not_projected_as_a_record(self):
        table = TableBlock(
            table_id="investors",
            extraction_layer="scanned_image_line_grid",
            metadata={"source": "scanned_bordered_table_reconstructor"},
            rows=[
                TableRow(
                    cells=[
                        CellValue(text="投资方"),
                        CellValue(text="股份数(万股)"),
                        CellValue(text="持股比例(%)"),
                    ]
                ),
                TableRow(cells=[CellValue(text="甲公司"), CellValue(text="10.5000"), CellValue(text="0.38")]),
            ],
        )

        records = _collect_table_records(
            ParseResult(pages=[PageContent(page_number=11, tables=[table])]),
            allow_embedded_headers=True,
        )

        assert [record["raw"]["投资方"] for record in records] == ["甲公司"]

    def test_scanned_tax_header_uses_multi_rate_value_as_numeric_following_evidence(self):
        table = TableBlock(
            table_id="tax_rates",
            extraction_layer="scanned_image_line_grid",
            metadata={"source": "scanned_bordered_table_reconstructor"},
            rows=[
                TableRow(cells=[CellValue(text="税种"), CellValue(text="计税依据"), CellValue(text="税率(%)")]),
                TableRow(cells=[CellValue(text="增值税"), CellValue(text="销售额扣除进项税"), CellValue(text="9、6")]),
                TableRow(cells=[CellValue(text="城市维护建设税"), CellValue(text="应交流转税"), CellValue(text="5")]),
                TableRow(cells=[CellValue(text="教育费附加"), CellValue(text="应交流转税"), CellValue(text="3")]),
                TableRow(cells=[CellValue(text="地方教育附加"), CellValue(text="应交流转税"), CellValue(text="2")]),
            ],
        )

        result = ParseResult(pages=[PageContent(page_number=32, tables=[table])])
        records = _collect_table_records(
            result,
            allow_embedded_headers=True,
        )

        assert len(records) == 4
        assert records[0]["raw"]["税率(%)"] == "9、6"

    def test_data_contaminated_first_row_is_not_promoted_to_header(self):
        table = TableBlock(
            table_id="contaminated_header",
            extraction_layer="scanned_image_line_grid",
            metadata={"source": "scanned_bordered_table_reconstructor"},
            rows=[
                TableRow(
                    cells=[
                        CellValue(text="项目 甲公司 乙公司"),
                        CellValue(text="年初余额 10.00 20.00"),
                    ]
                ),
                TableRow(cells=[CellValue(text="合计"), CellValue(text="30.00")]),
            ],
        )

        records = _collect_table_records(ParseResult(pages=[PageContent(page_number=1, tables=[table])]))

        assert records[0]["raw"] == {
            "col_0": "项目 甲公司 乙公司",
            "col_1": "年初余额 10.00 20.00",
        }
        assert len(records) == 2

    def test_table_local_type_inference_does_not_share_damaged_samples(self):
        amount_table = TableBlock(
            table_id="amounts",
            headers=["金额"],
            rows=[
                TableRow(cells=[CellValue(text="10.00")]),
                TableRow(cells=[CellValue(text="20.00")]),
                TableRow(cells=[CellValue(text="30.00")]),
            ],
        )
        note_table = TableBlock(
            table_id="notes",
            headers=["金额"],
            rows=[
                TableRow(cells=[CellValue(text="不适用")]),
                TableRow(cells=[CellValue(text="详见附注")]),
            ],
        )

        inferred = _infer_table_column_types(
            [(amount_table, [1], "physical_table"), (note_table, [2], "physical_table")]
        )

        assert inferred["amounts"]["金额"]["type"] == "amount"
        assert inferred["notes"]["金额"]["type"] == "text"

    def test_currency_context_is_read_from_table_cells(self):
        unit_table = TableBlock(
            table_id="unit",
            rows=[TableRow(cells=[CellValue(text="金额单位:人民币元")])],
        )
        amount_table = TableBlock(
            table_id="amounts",
            headers=["项目", "金额"],
            rows=[TableRow(cells=[CellValue(text="资本"), CellValue(text="10.00")])],
        )
        result = ParseResult(
            pages=[PageContent(page_number=1, tables=[unit_table, amount_table])],
            entities=DocumentEntities(document_type="audit_report"),
        )

        out = build_generic_community_output(result, "audit_report", "")
        record = next(record for record in out["data"]["records"] if record["raw"].get("项目") == "资本")

        assert record["normalized"]["金额"] == {"value": 10.0, "currency": "CNY"}

    def test_logical_ocr_table_recovers_group_header_and_strips_merged_amount(self):
        table = LogicalTable(
            table_id="lt_inventory",
            logical_id="lt_inventory",
            rows=[
                TableRow(
                    cells=[
                        CellValue(),
                        CellValue(),
                        CellValue(text="年末余额"),
                        CellValue(),
                    ]
                ),
                TableRow(
                    cells=[
                        CellValue(text="项 目"),
                        CellValue(text="账面余额 1,297,676.15"),
                        CellValue(text="存货跌价准备"),
                        CellValue(text="账面价值"),
                    ]
                ),
                TableRow(
                    cells=[
                        CellValue(text="库存商品"),
                        CellValue(text="10.00"),
                        CellValue(),
                        CellValue(text="10.00"),
                    ]
                ),
            ],
        )

        records = _collect_table_records(ParseResult(logical_tables=[table]))

        assert records[0]["raw"] == {
            "项目": "库存商品",
            "年末余额/账面余额": "10.00",
            "年末余额/存货跌价准备": "",
            "年末余额/账面价值": "10.00",
        }

    def test_existing_heading_levels_use_monotonic_scope_and_table_recovery(self):
        result = ParseResult(
            pages=[
                PageContent(
                    page_number=80,
                    texts=[
                        TextBlock(content="六、合并财务报表项目注释", level=TextLevel.H1),
                        TextBlock(content="27、递延收益", level=TextLevel.H2),
                    ],
                    tables=[
                        TableBlock(
                            table_id="capital_notes",
                            rows=[
                                TableRow(cells=[CellValue(text="28、实收资本")]),
                                TableRow(cells=[CellValue(text="29、资本公积")]),
                                TableRow(cells=[CellValue(text="30、盈余公积")]),
                                TableRow(cells=[CellValue(text="31、未分配利润")]),
                            ],
                        )
                    ],
                ),
                PageContent(
                    page_number=82,
                    texts=[
                        TextBlock(content="一、短期薪酬", level=TextLevel.H1),
                        TextBlock(content="1、工资、奖金、津贴和补贴", level=TextLevel.H2),
                        TextBlock(content="32、营业收入和营业成本", level=TextLevel.H2),
                    ],
                ),
            ]
        )

        titles = [section["title"] for section in _collect_sections(result)]

        assert {"28、实收资本", "29、资本公积", "30、盈余公积", "31、未分配利润"} <= set(titles)
        assert "32、营业收入和营业成本" in titles
        assert "一、短期薪酬" not in titles
        assert "1、工资、奖金、津贴和补贴" not in titles

    def test_audit_outline_excludes_statement_rows_before_notes_and_nested_note_rows(self):
        result = ParseResult(
            pages=[
                PageContent(
                    page_number=3,
                    texts=[
                        TextBlock(content="一、审计意见", level=TextLevel.H1),
                        TextBlock(content="二、形成审计意见的基础", level=TextLevel.H1),
                        TextBlock(content="三、管理层和治理层对财务报表的责任", level=TextLevel.H1),
                        TextBlock(content="四、注册会计师对财务报表审计的责任", level=TextLevel.H1),
                    ],
                ),
                PageContent(
                    page_number=8,
                    texts=[TextBlock(content="六、其他综合收益", level=TextLevel.H1)],
                ),
                PageContent(
                    page_number=9,
                    texts=[TextBlock(content="二、投资活动产生的现金流量", level=TextLevel.H1)],
                ),
                PageContent(
                    page_number=18,
                    texts=[TextBlock(content="一、公司基本情况", level=TextLevel.H1)],
                ),
                PageContent(
                    page_number=48,
                    texts=[
                        TextBlock(content="六、合并财务报表项目注释", level=TextLevel.H1),
                        TextBlock(content="19、应付职工薪酬", level=TextLevel.H2),
                    ],
                ),
                PageContent(
                    page_number=63,
                    texts=[
                        TextBlock(content="1、短期薪酬", level=TextLevel.H2),
                        TextBlock(content="20、应交税费", level=TextLevel.H2),
                    ],
                ),
                PageContent(
                    page_number=65,
                    texts=[
                        TextBlock(content="盈余公积", bbox=[126, 380.5, 171.5, 396]),
                        TextBlock(content="30、", bbox=[109, 383.5, 132.5, 394]),
                    ],
                ),
            ]
        )

        titles = [section["title"] for section in _collect_sections(result, document_type="audit_report")]

        assert "六、其他综合收益" not in titles
        assert "二、投资活动产生的现金流量" not in titles
        assert "1、短期薪酬" not in titles
        assert {"一、审计意见", "一、公司基本情况", "20、应交税费", "30、盈余公积"} <= set(titles)

    def test_contained_page_local_ocr_grid_is_deduplicated(self):
        small = TableBlock(
            table_id="small",
            rows=[
                TableRow(cells=[CellValue(text="项目"), CellValue(text="年初余额")]),
                TableRow(cells=[CellValue(text="法定盈余公积"), CellValue(text="29,011,115.74")]),
            ],
        )
        rich = TableBlock(
            table_id="rich",
            rows=[
                TableRow(
                    cells=[
                        CellValue(text="项目"),
                        CellValue(text="年初余额"),
                        CellValue(text="本年增加"),
                        CellValue(text="年末余额"),
                    ]
                ),
                TableRow(
                    cells=[
                        CellValue(text="法定盈余公积"),
                        CellValue(text="29,011,115.74"),
                        CellValue(),
                        CellValue(text="29,011,115.74"),
                    ]
                ),
            ],
        )
        result = ParseResult(pages=[PageContent(page_number=65, tables=[small, rich])])

        out = build_generic_community_output(result, "audit_report", "")

        assert [table["table_id"] for table in out["data"]["tables"]] == ["rich"]

    def test_overlapping_ocr_table_views_with_minor_text_drift_are_deduplicated(self):
        small = TableBlock(
            table_id="small",
            bbox=[10, 10, 200, 120],
            rows=[
                TableRow(cells=[CellValue(text="项目"), CellValue(text="年末余额")]),
                TableRow(cells=[CellValue(text="资本"), CellValue(text="10,00")]),
            ],
        )
        rich = TableBlock(
            table_id="rich",
            bbox=[8, 8, 202, 122],
            rows=[
                TableRow(cells=[CellValue(text="项目"), CellValue(text="年末余额")]),
                TableRow(cells=[CellValue(text="资本"), CellValue(text="10.00")]),
                TableRow(cells=[CellValue(text="合计"), CellValue(text="10.00")]),
            ],
        )
        result = ParseResult(pages=[PageContent(page_number=65, tables=[small, rich])])

        out = build_generic_community_output(result, "audit_report", "")

        assert [table["table_id"] for table in out["data"]["tables"]] == ["rich"]

    def test_audit_appendix_fields_are_retained_but_confidence_is_demoted(self):
        result = ParseResult(
            pages=[
                PageContent(
                    page_number=83,
                    page_mode="scanned_ocr",
                    texts=[
                        TextBlock(
                            content="本页无正文，为审计报告签字盖章页",
                            evidence_ids=["ocr:p0083:0001"],
                        )
                    ],
                ),
                PageContent(
                    page_number=84,
                    key_values=[
                        KeyValuePair(
                            key="国家企业信用信息公示系统网址",
                            value="http://www.gsxt.gov.cn",
                            confidence=0.98,
                            evidence_ids=["ocr:p0084:0046"],
                        )
                    ],
                ),
            ],
            entities=DocumentEntities(document_type="audit_report"),
        )

        out = build_generic_community_output(result, "audit_report", "")

        assert out["data"]["fields"]["国家企业信用信息公示系统网址"] == "http://www.gsxt.gov.cn"
        assert out["data"]["field_metadata"]["国家企业信用信息公示系统网址"]["confidence"] == 0.79

    def test_outline_keeps_late_headings_beyond_two_hundred_candidates(self):
        result = ParseResult(
            pages=[
                PageContent(
                    page_number=1,
                    texts=[TextBlock(content=f"附录标题{index}", level=TextLevel.H3) for index in range(205)],
                ),
                PageContent(
                    page_number=87,
                    texts=[TextBlock(content="最终事项", level=TextLevel.H3)],
                ),
            ]
        )

        titles = [section["title"] for section in _collect_sections(result)]

        assert len(titles) == 206
        assert titles[-1] == "最终事项"

    def test_exact_repeated_header_row_is_skipped(self):
        table = TableBlock(
            table_id="repeated_header",
            headers=["日期", "金额"],
            rows=[
                TableRow(cells=[CellValue(text="日期"), CellValue(text="金额")]),
                TableRow(cells=[CellValue(text="2024-01-01"), CellValue(text="10.00")]),
            ],
        )
        records = _collect_table_records(ParseResult(pages=[PageContent(page_number=1, tables=[table])]))
        assert len(records) == 1
        assert records[0]["raw"]["日期"] == "2024-01-01"

    def test_structural_rows_are_preserved_but_merged_headers_are_not_records(self):
        table = TableBlock(
            table_id="page_wide_table",
            headers=["项目", "本年发生额", "上年发生额"],
            rows=[
                TableRow(cells=[CellValue(text="44、所得税费用"), CellValue(), CellValue()]),
                TableRow(cells=[CellValue(text="项目"), CellValue(), CellValue(text="本年发生额上年发生额")]),
                TableRow(cells=[CellValue(text="当期所得税费用"), CellValue(text="10.00"), CellValue(text="20.00")]),
            ],
        )

        records = _collect_table_records(ParseResult(pages=[PageContent(page_number=70, tables=[table])]))

        assert len(records) == 2
        assert records[0]["raw"] == {
            "项目": "44、所得税费用",
            "本年发生额": "",
            "上年发生额": "",
        }
        assert records[1]["raw"] == {
            "项目": "当期所得税费用",
            "本年发生额": "10.00",
            "上年发生额": "20.00",
        }

    def test_low_confidence_non_contiguous_logical_table_uses_physical_tables(self):
        physical_1 = TableBlock(
            table_id="pt_1_0",
            headers=["项目", "年末余额"],
            rows=[TableRow(cells=[CellValue(text="银行存款"), CellValue(text="10.00")])],
        )
        physical_3 = TableBlock(
            table_id="pt_3_0",
            headers=["项目", "年末余额"],
            rows=[TableRow(cells=[CellValue(text="固定资产"), CellValue(text="20.00")])],
        )
        logical = LogicalTable(
            table_id="lt_0",
            logical_id="lt_0",
            headers=["项目", "年末余额"],
            rows=[TableRow(cells=[CellValue(text="错误合并"), CellValue(text="30.00")])],
            source_physical_ids=["pt_1_0", "pt_3_0"],
            source_pages=[1, 3],
            merge_confidence=0.62,
            provenance=[RowProvenance(source_page=1, source_table_id="pt_1_0")],
        )
        result = ParseResult(
            pages=[
                PageContent(page_number=1, tables=[physical_1]),
                PageContent(page_number=3, tables=[physical_3]),
            ],
            logical_tables=[logical],
        )

        records = _collect_table_records(result)

        assert [record["source"]["table_id"] for record in records] == ["pt_1_0", "pt_3_0"]
        assert [record["raw"]["项目"] for record in records] == ["银行存款", "固定资产"]

    def test_high_confidence_contiguous_logical_table_is_preserved(self):
        logical = LogicalTable(
            table_id="lt_0",
            logical_id="lt_0",
            headers=["项目", "年末余额"],
            rows=[TableRow(cells=[CellValue(text="银行存款"), CellValue(text="10.00")])],
            source_pages=[1, 2],
            merge_confidence=0.9,
            provenance=[RowProvenance(source_page=1, source_table_id="pt_1_0")],
        )
        result = ParseResult(logical_tables=[logical])

        records = _collect_table_records(result)

        assert records[0]["source"]["table_id"] == "lt_0"

    def test_uncovered_physical_table_is_not_hidden_by_other_logical_tables(self):
        covered = TableBlock(
            table_id="pt_1_0",
            headers=["项目", "金额"],
            rows=[TableRow(cells=[CellValue(text="销售额"), CellValue(text="10.00")])],
        )
        uncovered = TableBlock(
            table_id="pt_2_0",
            headers=["税种", "税额"],
            rows=[TableRow(cells=[CellValue(text="增值税"), CellValue(text="1.30")])],
        )
        covered_late = TableBlock(
            table_id="pt_3_0",
            headers=["项目", "金额"],
            rows=[TableRow(cells=[CellValue(text="利润总额"), CellValue(text="8.70")])],
        )
        logical = LogicalTable(
            table_id="lt_0",
            logical_id="lt_0",
            headers=["项目", "金额"],
            rows=covered.rows,
            source_physical_ids=["pt_1_0"],
            source_pages=[1],
            provenance=[RowProvenance(source_page=1, source_table_id="pt_1_0")],
        )
        logical_late = LogicalTable(
            table_id="lt_1",
            logical_id="lt_1",
            headers=["项目", "金额"],
            rows=covered_late.rows,
            source_physical_ids=["pt_3_0"],
            source_pages=[3],
            provenance=[RowProvenance(source_page=3, source_table_id="pt_3_0")],
        )
        result = ParseResult(
            pages=[
                PageContent(page_number=1, tables=[covered]),
                PageContent(page_number=2, tables=[uncovered]),
                PageContent(page_number=3, tables=[covered_late]),
            ],
            logical_tables=[logical, logical_late],
        )

        records = _collect_table_records(result)

        assert [record["source"]["table_id"] for record in records] == ["lt_0", "pt_2_0", "lt_1"]

    def test_canonical_domain_collections_do_not_reenter_scalar_fields(self):
        result = ParseResult(
            entities=DocumentEntities(
                document_type="audit_report",
                domain_specific={
                    "法定代表人": "许水均",
                    "line_items": [],
                    "notes": [],
                    "summary": {"field_count": 1},
                    "records": [{"name": "row"}],
                },
            )
        )

        output = build_generic_community_output(result, "audit_report", "")

        assert output["data"]["fields"] == {"法定代表人": "许水均"}
