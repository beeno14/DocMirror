from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from docmirror.plugins.credit_report.personal_detail_scanned import native_extraction
from docmirror.plugins.credit_report.personal_detail_scanned.schema import (
    _decimal_string,
    personal_detail_data_dictionary,
    project_personal_detail_datasets,
)


def _table(table_id: str, rows: list[list[str]]) -> SimpleNamespace:
    bboxes = [
        [[column * 100, row * 20, (column + 1) * 100, (row + 1) * 20] for column in range(len(cells))]
        for row, cells in enumerate(rows)
    ]
    return SimpleNamespace(
        table_id=table_id,
        bbox=[0, 0, 1200, max(20, len(rows) * 20)],
        confidence=0.96,
        metadata={"raw_rows": rows, "source_cell_bboxes": bboxes},
    )


@pytest.mark.parametrize("malformed", ("1,2", "1,,2", "1,23,456", "12,34.50"))
def test_schema_decimal_never_concatenates_malformed_grouping(
    malformed: str,
) -> None:
    assert _decimal_string(malformed) is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    (("0", "0"), ("-12.50", "-12.5"), ("1,234", "1234"), ("12,345.60", "12345.6")),
)
def test_schema_decimal_accepts_only_plain_or_registered_thousands_groups(
    raw: str,
    expected: str,
) -> None:
    assert _decimal_string(raw) == expected


def _seal_table(table: SimpleNamespace) -> SimpleNamespace:
    rows = table.metadata["raw_rows"]
    column_count = max((len(row) for row in rows), default=0)
    table.metadata["geometry"] = {
        "row_bands": [
            {"index": row, "y0": row * 20, "y1": (row + 1) * 20}
            for row in range(len(rows))
        ],
        "col_bands": [
            {"index": column, "x0": column * 100, "x1": (column + 1) * 100}
            for column in range(column_count)
        ],
        "cell_bboxes": table.metadata["source_cell_bboxes"],
        "cell_geometry_status": [["exact" for _cell in row] for row in rows],
        "cell_evidence_ids": [
            [[f"{table.table_id}:{row_index}:{column}"] for column, _cell in enumerate(row)]
            for row_index, row in enumerate(rows)
        ],
    }
    return table


def _result(
    *tables: SimpleNamespace,
    role: str | None = None,
) -> SimpleNamespace:
    if role:
        for table in tables:
            table.metadata["canonical_template_id"] = role
    return SimpleNamespace(
        pages=[
            SimpleNamespace(
                page_number=1,
                source_page_number=1,
                canonical_template_id=role,
                tables=list(tables),
            )
        ]
    )


def _issues(result: SimpleNamespace) -> list[dict]:
    return list(getattr(result, "_personal_detail_extraction_issues", []) or [])


def test_public_rows_keep_physical_columns_and_report_missing_cell() -> None:
    result = _result(
        _seal_table(
            _table(
                "tax",
            [
                ["编号", "主管税务机关", "欠税总额", "欠税统计日期"],
                ["1", "某市税务局", "", "2024-01-31"],
                ["2", "某区税务局", "500", "2024-02-29"],
            ],
            )
        ),
        role="public_information",
    )

    records = native_extraction._extract_public_records(result)

    assert len(records) == 2
    first = json.loads(records[0]["content"])
    second = json.loads(records[1]["content"])
    assert "arrears_amount" not in first
    assert first["statistics_date"] == "2024-01-31"
    assert second["arrears_amount"] == 500
    assert second["statistics_date"] == "2024-02-29"
    assert any(
        issue.get("issue_code") == "candidate_b_public_record_cell_unresolved"
        and issue.get("field_name") == "arrears_amount"
        for issue in _issues(result)
    )


@pytest.mark.parametrize(
    "malformed",
    ("1,2", "1,,2", "1,23,456", "1 2", "1  234", "1, 234"),
)
def test_public_amount_rejects_malformed_grouping_without_digit_concatenation(
    malformed: str,
) -> None:
    result = _result(
        _seal_table(
            _table(
                "tax-malformed-amount",
            [
                ["编号", "主管税务机关", "欠税总额", "欠税统计日期"],
                ["1", "某市税务局", malformed, "2024-01-31"],
                ["2", "某区税务局", "1,234", "2024-02-29"],
            ],
            )
        ),
        role="public_information",
    )

    records = native_extraction._extract_public_records(result)
    first, second = (json.loads(record["content"]) for record in records)
    observed = " ".join(malformed.split())

    assert "arrears_amount" not in first
    assert second["arrears_amount"] == 1234
    assert records[0]["canonical_raw"]["arrears_amount"] == [observed]
    assert records[0]["source_refs_by_field"]["arrears_amount"][0]["canonical_column"] == 2
    assert any(
        issue.get("field_name") == "arrears_amount"
        and issue.get("observed_value") == [observed]
        and issue.get("source_refs", [{}])[0].get("canonical_column") == 2
        for issue in _issues(result)
    )


def test_public_whitespace_amount_is_withheld_and_localized_in_final_projection() -> None:
    result = _result(
        _seal_table(
            _table(
                "tax-whitespace-amount",
            [
                ["编号", "主管税务机关", "欠税总额", "欠税统计日期"],
                ["1", "某市税务局", "1 2", "2024-01-31"],
            ],
            )
        ),
        role="public_information",
    )
    records = native_extraction._extract_public_records(result)

    projected = project_personal_detail_datasets(
        {
            "public_records": records,
            "personal_detail_extraction_issues": _issues(result),
        }
    )

    assert "arrears_amount" not in projected["tax_arrears_records"][0]
    issue = next(
        row
        for row in projected["extraction_issues"]
        if row.get("field_name") == "arrears_amount"
    )
    assert issue["source_refs"][0]["canonical_column"] == 2
    assert any(
        row.get("extraction_issue_id") == issue["extraction_issue_id"]
        and row.get("evidence_kind") == "observed"
        and row.get("string_value") == "1 2"
        for row in projected["extraction_issue_evidence"]
    )


def test_public_unreadable_sequence_reports_row_and_keeps_later_rows() -> None:
    result = _result(
        _seal_table(
            _table(
                "tax-sequence",
            [
                ["编号", "主管税务机关", "欠税总额", "欠税统计日期"],
                ["X", "某市税务局", "100", "2024-01-31"],
                ["2", "某区税务局", "500", "2024-02-29"],
            ],
            )
        ),
        role="public_information",
    )

    records = native_extraction._extract_public_records(result)

    assert len(records) == 1
    assert json.loads(records[0]["content"])["tax_authority"] == "某区税务局"
    assert any(
        issue.get("issue_code") == "candidate_b_sequence_cell_unresolved"
        and issue.get("observed_value", {}).get("physical_cells", [None])[0] == "X"
        for issue in _issues(result)
    )


def test_public_scalar_date_with_multiple_valid_spans_is_withheld_and_reported() -> None:
    result = _result(
        _seal_table(
            _table(
                "tax-date",
            [
                ["编号", "主管税务机关", "欠税总额", "欠税统计日期"],
                ["1", "某市税务局", "100", "2024-01-31 2024-02-29"],
                ["2", "某区税务局", "500", "2024-02-29"],
            ],
            )
        ),
        role="public_information",
    )

    records = native_extraction._extract_public_records(result)
    first, second = (json.loads(record["content"]) for record in records)

    assert "statistics_date" not in first
    assert second["statistics_date"] == "2024-02-29"
    assert any(
        issue.get("field_name") == "statistics_date"
        and issue.get("observed_value") == ["2024-01-31 2024-02-29"]
        for issue in _issues(result)
    )


def test_public_text_slots_reject_layout_labels_and_multiple_typed_markers() -> None:
    result = _result(
        _seal_table(
            _table(
                "civil-text",
            [
                ["编号", "立案法院", "案由", "立案日期", "结案方式"],
                ["1", "立案法院 某法院 立案日期 2024-01-01", "借款纠纷", "2024-01-01", "判决"],
                ["2", "某法院 2024-02-01", "合同纠纷", "2024-02-01", "调解"],
                ["3", "某法院", "劳动纠纷", "2024-03-01", "撤诉"],
            ],
            )
        ),
        role="public_information",
    )

    records = native_extraction._extract_public_records(result)
    contents = [json.loads(record["content"]) for record in records]

    assert "filing_court" not in contents[0]
    assert "filing_court" not in contents[1]
    assert contents[2]["filing_court"] == "某法院"
    assert sum(issue.get("field_name") == "filing_court" for issue in _issues(result)) == 2


@pytest.mark.parametrize(
    "raw",
    (
        "导中国工商银行股份有限公司",
        "S 中国工商银行股份有限公司",
        "中国工商银行股份有限公司 Ss",
    ),
)
def test_public_business_name_never_deletes_unowned_glyphs(raw: str) -> None:
    assert native_extraction._public_value(raw, "employer_name") is None
    assert (
        native_extraction._public_value("中国工商银行股份有限公司", "employer_name")
        == "中国工商银行股份有限公司"
    )


def test_account_institution_exact_slot_rejects_contamination() -> None:
    assert native_extraction._account_institution("中国银行股份有限公司厦门分行") == (
        "中国银行股份有限公司厦门分行"
    )
    assert native_extraction._account_institution(
        "甲银行股份有限公司 乙银行股份有限公司"
    ) is None
    assert native_extraction._account_institution(
        "中国银行股份有限公司 开立日期"
    ) is None


def test_public_two_part_civil_record_joins_by_printed_sequence() -> None:
    result = _result(
        _seal_table(
            _table(
                "civil",
            [
                ["编号", "立案法院", "案由", "立案日期", "结案方式"],
                ["1", "某法院", "合同纠纷", "2023-01-02", "判决"],
                ["编号", "判决/调解结果", "判决/调解生效日期", "诉讼标的", "诉讼标的金额"],
                ["1", "被告偿还借款", "2023-05-06", "借款", "10000"],
            ],
            )
        ),
        role="public_information",
    )

    records = native_extraction._extract_public_records(result)

    assert len(records) == 1
    content = json.loads(records[0]["content"])
    assert content["filing_court"] == "某法院"
    assert content["judgment_result"] == "被告偿还借款"
    assert content["claim_amount"] == 10000
    assert records[0]["source_refs_by_field"]["claim_amount"][0]["geometry_scope"] == "cell"


def test_postpaid_card_does_not_shift_after_blank_value() -> None:
    result = _result(
        _seal_table(
            _table(
                "postpaid",
                [
                    ["机构名称", "业务类型", "业务开通日期", "当前缴费状态", "当前欠费金额", "记账年月"],
                    ["某通信公司", "移动电话", "2020-01-02", "正常", "", "2024-06"],
                ],
            )
        ),
        role="postpaid_detail",
    )

    records = native_extraction._extract_postpaid_records(result)

    assert len(records) == 1
    assert records[0]["billing_month"] == "2024-06"
    assert "current_arrears_amount" not in records[0]
    assert records[0]["source_refs_by_field"]["billing_month"][0]["canonical_column"] == 5
    assert any(
        issue.get("target_record_id") == records[0]["postpaid_record_id"]
        and issue.get("field_name") == "current_arrears_amount"
        for issue in _issues(result)
    )


@pytest.mark.parametrize("malformed", ("1,2", "1,,2", "1,23,456"))
def test_postpaid_amount_never_concatenates_malformed_grouping(
    malformed: str,
) -> None:
    result = _result(
        _seal_table(
            _table(
                "postpaid-malformed-amount",
                [
                    ["机构名称", "业务类型", "业务开通日期", "当前缴费状态", "当前欠费金额", "记账年月"],
                    ["某通信公司", "移动电话", "2020-01-02", "欠费", malformed, "2024-06"],
                ],
            )
        ),
        role="postpaid_detail",
    )

    records = native_extraction._extract_postpaid_records(result)

    assert len(records) == 1
    assert "current_arrears_amount" not in records[0]
    assert any(
        issue.get("target_record_id") == records[0]["postpaid_record_id"]
        and issue.get("field_name") == "current_arrears_amount"
        for issue in _issues(result)
    )


def test_postpaid_short_row_cannot_left_shift_text_fields() -> None:
    result = _result(
        _seal_table(
            _table(
                "postpaid-short-row",
                [
                    ["机构名称", "业务类型", "业务开通日期", "当前缴费状态", "当前欠费金额", "记账年月"],
                    ["移动通信", "2024-01-01", "正常", "100", "2024-06"],
                ],
            )
        ),
        role="postpaid_detail",
    )

    assert native_extraction._extract_postpaid_records(result) == []
    assert any(
        issue.get("issue_code") == "candidate_b_canonical_header_graph_unresolved"
        and issue.get("target_dataset") == "postpaid_records"
        for issue in _issues(result)
    )


def test_postpaid_months_use_exact_header_columns_and_retain_bad_cell_as_issue() -> None:
    result = _result(
        _seal_table(
            _table(
                "postpaid-history",
            [
                ["机构名称", "业务类型", "记账年月"],
                ["某通信公司", "移动电话", "2024-06"],
                ["缴费记录", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"],
                ["2024", "N", "N", "Q", "N", "N", "N", "N", "N", "N", "N", "N", "N"],
            ],
            )
        ),
        role="postpaid_detail",
    )

    records = native_extraction._extract_postpaid_payment_history(result)

    assert len(records) == 12
    march = next(record for record in records if record["month"] == 3)
    april = next(record for record in records if record["month"] == 4)
    assert "status" not in march
    assert april["status"] == "N"
    assert april["source_refs"][0]["canonical_column"] == 4
    assert april["source_refs_by_field"]["institution"][0]["canonical_column"] == 0
    assert april["source_refs_by_field"]["business_type"][0]["canonical_column"] == 1
    assert april["source_refs_by_field"]["status"][0]["canonical_column"] == 4
    assert any(
        issue.get("target_record_id") == march["postpaid_payment_history_id"]
        and issue.get("field_name") == "status"
        for issue in _issues(result)
    )


@pytest.mark.parametrize(
    "row",
    (
        ["1", "某市税务局", "100"],
        ["1", "某市税务局", "100", "2024-01-31", "尾部残留"],
    ),
)
def test_public_short_or_extra_row_is_issue_only(row: list[str]) -> None:
    result = _result(
        _seal_table(
            _table(
                "public-width-mismatch",
                [
                    ["编号", "主管税务机关", "欠税总额", "欠税统计日期"],
                    row,
                ],
            )
        ),
        role="public_information",
    )

    assert native_extraction._extract_public_records(result) == []
    issues = [
        issue
        for issue in _issues(result)
        if issue.get("issue_code") == "candidate_b_public_lattice_row_unresolved"
    ]
    assert {issue.get("field_name") for issue in issues} == {
        "tax_authority",
        "arrears_amount",
        "statistics_date",
    }
    assert all(issue["source_refs"][0]["row"] == 1 for issue in issues)


def test_public_header_aliases_must_occupy_the_whole_cell() -> None:
    result = _result(
        _seal_table(
            _table(
                "public-header-residue",
                [
                    ["编号说明", "主管税务机关", "欠税总额", "欠税统计日期"],
                    ["1", "某市税务局", "100", "2024-01-31"],
                ],
            )
        ),
        role="public_information",
    )

    assert native_extraction._extract_public_records(result) == []
    assert any(
        issue.get("issue_code") == "candidate_b_canonical_header_graph_unresolved"
        and issue.get("source_refs", [{}])[0].get("row") == 0
        for issue in _issues(result)
    )


def _civil_base_table(table_id: str, court: str) -> SimpleNamespace:
    return _seal_table(
        _table(
            table_id,
            [
                ["编号", "立案法院", "案由", "立案日期", "结案方式"],
                ["1", court, "合同纠纷", "2024-01-02", "判决"],
            ],
        )
    )


def _civil_detail_table(table_id: str) -> SimpleNamespace:
    return _seal_table(
        _table(
            table_id,
            [
                ["编号", "判决/调解结果", "判决/调解生效日期", "诉讼标的", "诉讼标的金额"],
                ["1", "被告偿还借款", "2024-05-06", "借款", "10000"],
            ],
        )
    )


def test_public_same_sequence_in_distinct_tables_never_merges_globally() -> None:
    result = _result(
        _civil_base_table("civil-a", "甲法院"),
        _civil_base_table("civil-b", "乙法院"),
        role="public_information",
    )

    records = native_extraction._extract_public_records(result)

    assert len(records) == 2
    assert {json.loads(record["content"])["filing_court"] for record in records} == {
        "甲法院",
        "乙法院",
    }
    assert len({record["public_record_id"] for record in records}) == 2


def test_public_duplicate_sequence_rows_are_not_silently_collapsed() -> None:
    row = ["1", "甲法院", "合同纠纷", "2024-01-02", "判决"]
    result = _result(
        _seal_table(
            _table(
                "civil-duplicate-physical-row",
                [
                    ["编号", "立案法院", "案由", "立案日期", "结案方式"],
                    row,
                    list(row),
                ],
            )
        ),
        role="public_information",
    )

    records = native_extraction._extract_public_records(result)

    assert len(records) == 1
    duplicate_issues = [
        issue
        for issue in _issues(result)
        if issue.get("issue_code")
        == "candidate_b_public_duplicate_sequence_owner_unresolved"
    ]
    assert {issue.get("field_name") for issue in duplicate_issues} == {
        "filing_court",
        "case_number",
        "cause",
        "filing_date",
        "closure_method",
    } - {"case_number"}
    assert all(issue["source_refs"][0]["row"] == 2 for issue in duplicate_issues)


def test_public_detail_requires_unique_approved_adjacent_base() -> None:
    result = _result(
        _civil_base_table("civil-base", "甲法院"),
        _civil_detail_table("civil-detail"),
        role="public_information",
    )

    records = native_extraction._extract_public_records(result)
    contents = [json.loads(record["content"]) for record in records]

    assert len(records) == 2
    assert not any("filing_court" in content and "judgment_result" in content for content in contents)
    assert any(
        issue.get("issue_code") == "candidate_b_public_record_join_unresolved"
        for issue in _issues(result)
    )


def test_public_detail_joins_one_explicitly_approved_adjacent_base() -> None:
    result = _result(
        _civil_base_table("civil-base-approved", "甲法院"),
        _civil_detail_table("civil-detail-approved"),
        role="public_information",
    )
    result.tables_continue = lambda left, right: (
        left,
        right,
    ) == ("civil-base-approved", "civil-detail-approved")

    records = native_extraction._extract_public_records(result)

    assert len(records) == 1
    content = json.loads(records[0]["content"])
    assert content["filing_court"] == "甲法院"
    assert content["judgment_result"] == "被告偿还借款"


def test_public_optional_identifiers_publish_only_from_exact_whole_cells() -> None:
    result = _result(
        _seal_table(
            _table(
                "tax-id",
                [
                    ["编号", "主管税务机关", "纳税人识别号", "欠税总额", "欠税统计日期"],
                    ["1", "某市税务局", "91310000MA1K12345X", "100", "2024-01-31"],
                ],
            )
        ),
        _seal_table(
            _table(
                "civil-case-number",
                [
                    ["编号", "立案法院", "案号", "案由", "立案日期", "结案方式"],
                    ["1", "某法院", "(2024)京0101民初123号", "合同纠纷", "2024-01-02", "判决"],
                ],
            )
        ),
        _seal_table(
            _table(
                "penalty-document-number",
                [
                    [
                        "编号",
                        "处罚机构",
                        "文书编号",
                        "处罚内容",
                        "处罚金额",
                        "生效日期",
                        "截止日期",
                        "行政复议结果",
                    ],
                    ["1", "某市监管局", "京市监罚〔2024〕12号", "警告", "400", "2024-02-01", "--", "--"],
                ],
            )
        ),
        role="public_information",
    )

    contents = {
        record["record_type"]: json.loads(record["content"])
        for record in native_extraction._extract_public_records(result)
    }

    assert contents["tax_arrears"]["taxpayer_identifier"] == "91310000MA1K12345X"
    assert contents["civil_judgment"]["case_number"] == "(2024)京0101民初123号"
    assert contents["administrative_penalty"]["document_number"] == "京市监罚〔2024〕12号"


def test_public_invalid_identifier_preserves_raw_and_exact_ref() -> None:
    result = _result(
        _seal_table(
            _table(
                "tax-id-invalid",
                [
                    ["编号", "主管税务机关", "纳税人识别号", "欠税总额", "欠税统计日期"],
                    ["1", "某市税务局", "91310000ma1k12345x", "100", "2024-01-31"],
                ],
            )
        ),
        role="public_information",
    )

    record = native_extraction._extract_public_records(result)[0]

    assert "taxpayer_identifier" not in json.loads(record["content"])
    assert record["canonical_raw"]["taxpayer_identifier"] == ["91310000ma1k12345x"]
    issue = next(
        issue for issue in _issues(result) if issue.get("field_name") == "taxpayer_identifier"
    )
    assert issue["source_refs"][0]["canonical_column"] == 2
    assert issue["source_refs"][0]["evidence_ids"] == ["tax-id-invalid:1:2"]


@pytest.mark.parametrize(
    ("raw", "kind"),
    (
        ("AAAAAAAAAAAA", "taxpayer_identifier"),
        ("000000000000", "taxpayer_identifier"),
        ("错误123号", "case_number"),
        ("京市监罚〔2024〕12号", "case_number"),
        ("(2024)京0101民初123号", "document_number"),
    ),
)
def test_public_identifier_roles_reject_cross_type_or_unstructured_values(
    raw: str,
    kind: str,
) -> None:
    assert native_extraction._strict_public_identifier(raw, kind) is None


def test_visible_social_assistance_rows_are_issue_only_and_catalog_safe() -> None:
    result = _result(
        _seal_table(
            _table(
                "social-assistance",
                [
                    ["社会救助记录"],
                    ["人员类别", "所在地区"],
                    ["低保", "某市"],
                ],
            )
        ),
        role="public_information",
    )

    assert native_extraction._extract_public_records(result) == []
    issues = [
        issue
        for issue in _issues(result)
        if issue.get("issue_code") == "candidate_b_social_assistance_source_row_unresolved"
    ]
    assert len(issues) == 2
    assert all(not issue.get("field_name") for issue in issues)
    assert all(issue.get("source_refs") for issue in issues)
    projected = project_personal_detail_datasets(
        {"personal_detail_extraction_issues": issues}
    )
    assert len(projected["extraction_issues"]) == 2
    assert set(
        personal_detail_data_dictionary()["datasets"]["social_assistance_records"]["columns"]
    ) == {"social_assistance_record_id"}


@pytest.mark.parametrize("defect", ("duplicate_marker", "unsealed_marker", "marker_row_residue"))
def test_social_assistance_malformed_markers_still_conserve_every_visible_business_row(
    defect: str,
) -> None:
    rows = [
        ["社会救助记录"],
        ["人员类别", "所在地区"],
        ["低保", "某市"],
    ]
    if defect == "duplicate_marker":
        rows.extend([["社会救助记录"], ["人员类别", "所在地区"], ["低收入", "某区"]])
    elif defect == "marker_row_residue":
        rows[0].append("来源甲")
    table = _seal_table(_table(f"social-{defect}", rows))
    result = _result(table, role="public_information")
    if defect == "unsealed_marker":
        table.metadata["geometry"]["cell_evidence_ids"][0][0] = list(
            table.metadata["geometry"]["cell_evidence_ids"][1][0]
        )

    assert native_extraction._extract_public_records(result) == []
    row_issues = [
        issue
        for issue in _issues(result)
        if issue.get("issue_code") == "candidate_b_social_assistance_source_row_unresolved"
    ]
    expected_rows = {1, 2, 4, 5} if defect == "duplicate_marker" else {1, 2}
    if defect == "marker_row_residue":
        expected_rows.add(0)
    assert {
        issue["source_refs"][0].get("row")
        for issue in row_issues
    } == expected_rows


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    (
        ("institution", "某通信公司正常"),
        ("institution", "某通信公司正常说明"),
        ("business_type", "移动电话正常"),
        ("payment_status", "正常欠费"),
    ),
)
def test_postpaid_card_rejects_multi_token_residue(
    field_name: str,
    bad_value: str,
) -> None:
    values = {
        "institution": "某通信公司",
        "business_type": "移动电话",
        "service_start_date": "2020-01-02",
        "payment_status": "正常",
        "current_arrears_amount": "0",
        "billing_month": "2024-06",
    }
    values[field_name] = bad_value
    result = _result(
        _seal_table(
            _table(
                f"postpaid-residue-{field_name}",
                [
                    ["机构名称", "业务类型", "业务开通日期", "当前缴费状态", "当前欠费金额", "记账年月"],
                    list(values.values()),
                ],
            )
        ),
        role="postpaid_detail",
    )

    record = native_extraction._extract_postpaid_records(result)[0]

    assert field_name not in record
    assert record["canonical_raw"][field_name] == [bad_value]
    assert any(
        issue.get("field_name") == field_name
        and issue.get("observed_value") == [bad_value]
        for issue in _issues(result)
    )


def test_postpaid_history_accepts_only_the_finite_twelve_status_codes() -> None:
    statuses = ["*", "N", "0", "1", "2", "3", "4", "5", "6", "C", "G", "#"]
    result = _result(
        _seal_table(
            _table(
                "postpaid-history-finite",
                [
                    ["机构名称", "业务类型", "记账年月"],
                    ["某通信公司", "移动电话", "2024-06"],
                    ["缴费记录", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"],
                    ["2024", *statuses],
                ],
            )
        ),
        role="postpaid_detail",
    )

    records = native_extraction._extract_postpaid_payment_history(result)

    assert [record["status"] for record in records] == statuses
    assert not _issues(result)


@pytest.mark.parametrize(
    "year_row",
    (
        ["2024", *(["N"] * 11)],
        ["2024", *(["N"] * 13)],
    ),
)
def test_postpaid_history_short_or_extra_year_row_is_issue_only(
    year_row: list[str],
) -> None:
    result = _result(
        _seal_table(
            _table(
                "postpaid-history-width",
                [
                    ["机构名称", "业务类型", "记账年月"],
                    ["某通信公司", "移动电话", "2024-06"],
                    ["缴费记录", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"],
                    year_row,
                ],
            )
        ),
        role="postpaid_detail",
    )

    assert native_extraction._extract_postpaid_payment_history(result) == []
    assert any(
        issue.get("issue_code") == "candidate_b_postpaid_history_lattice_row_unresolved"
        and issue.get("source_refs", [{}])[0].get("row") == 3
        for issue in _issues(result)
    )


def test_postpaid_second_value_row_is_localized_instead_of_silently_ignored() -> None:
    result = _result(
        _seal_table(
            _table(
                "postpaid-extra-value-row",
                [
                    ["机构名称", "业务类型", "业务开通日期", "当前缴费状态", "当前欠费金额", "记账年月"],
                    ["甲通信公司", "移动电话", "2020-01-02", "正常", "0", "2024-06"],
                    ["乙通信公司", "移动电话", "2021-02-03", "正常", "0", "2024-07"],
                ],
            )
        ),
        role="postpaid_detail",
    )

    assert len(native_extraction._extract_postpaid_records(result)) == 1
    issues = [
        issue
        for issue in _issues(result)
        if issue.get("issue_code") == "candidate_b_postpaid_card_pair_owner_unresolved"
    ]
    assert {issue.get("field_name") for issue in issues} == set(
        native_extraction._POSTPAID_CARD_FIELDS
    )
    assert all(issue["source_refs"][0]["row"] == 2 for issue in issues)


def test_duplicate_postpaid_card_identity_keeps_unique_physical_owners() -> None:
    rows = [
        ["机构名称", "业务类型", "业务开通日期", "当前缴费状态", "当前欠费金额", "记账年月"],
        ["某通信公司", "移动电话", "2020-01-02", "正常", "0", "2024-06"],
    ]
    result = _result(
        _seal_table(_table("postpaid-duplicate-a", rows)),
        _seal_table(_table("postpaid-duplicate-b", rows)),
        role="postpaid_detail",
    )

    records = native_extraction._extract_postpaid_records(result)

    assert len(records) == 2
    assert len({record["postpaid_record_id"] for record in records}) == 2
    assert sum(
        issue.get("issue_code") == "candidate_b_postpaid_identity_not_unique"
        for issue in _issues(result)
    ) == 2


def test_multiple_exact_postpaid_pairs_in_one_table_are_each_localized() -> None:
    result = _result(
        _seal_table(
            _table(
                "postpaid-multiple-pairs",
                [
                    ["机构名称", "业务类型", "业务开通日期", "当前缴费状态", "当前欠费金额", "记账年月"],
                    ["甲通信公司", "移动电话", "2020-01-02", "正常", "0", "2024-06"],
                    ["机构名称", "业务类型", "业务开通日期", "当前缴费状态", "当前欠费金额", "记账年月"],
                    ["乙通信公司", "移动电话", "2021-02-03", "正常", "0", "2024-07"],
                ],
            )
        ),
        role="postpaid_detail",
    )

    assert native_extraction._extract_postpaid_records(result) == []
    issues = [
        issue
        for issue in _issues(result)
        if issue.get("issue_code") == "candidate_b_postpaid_card_pair_owner_unresolved"
    ]
    assert len(issues) == 12
    assert len({issue["target_record_id"] for issue in issues}) == 2
    assert {issue["source_refs"][0]["row"] for issue in issues} == {1, 3}


def test_duplicate_postpaid_history_year_rows_are_both_localized() -> None:
    result = _result(
        _seal_table(
            _table(
                "postpaid-duplicate-year",
                [
                    ["机构名称", "业务类型", "记账年月"],
                    ["某通信公司", "移动电话", "2024-06"],
                    ["缴费记录", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"],
                    ["2024", *(["N"] * 12)],
                    ["2024", *(["N"] * 12)],
                ],
            )
        ),
        role="postpaid_detail",
    )

    assert native_extraction._extract_postpaid_payment_history(result) == []
    issues = [
        issue
        for issue in _issues(result)
        if issue.get("issue_code")
        == "candidate_b_postpaid_history_duplicate_year_owner_unresolved"
        and issue.get("field_name") == "year"
    ]
    assert len(issues) == 2
    assert {issue["source_refs"][0]["row"] for issue in issues} == {3, 4}
    status_issues = [
        issue
        for issue in _issues(result)
        if issue.get("issue_code")
        == "candidate_b_postpaid_history_duplicate_year_owner_unresolved"
        and issue.get("field_name") == "status"
    ]
    assert len(status_issues) == 24
    assert {issue["source_refs"][0]["canonical_column"] for issue in status_issues} == set(
        range(1, 13)
    )


def test_duplicate_postpaid_month_identity_across_tables_keeps_physical_owners() -> None:
    rows = [
        ["机构名称", "业务类型", "记账年月"],
        ["某通信公司", "移动电话", "2024-06"],
        ["缴费记录", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"],
        ["2024", *(["N"] * 12)],
    ]
    result = _result(
        _seal_table(_table("postpaid-history-duplicate-a", rows)),
        _seal_table(_table("postpaid-history-duplicate-b", rows)),
        role="postpaid_detail",
    )

    records = native_extraction._extract_postpaid_payment_history(result)

    assert len(records) == 24
    assert len({record["postpaid_payment_history_id"] for record in records}) == 24
    assert sum(
        issue.get("issue_code") == "candidate_b_postpaid_history_identity_not_unique"
        for issue in _issues(result)
    ) == 24


def test_postpaid_history_identity_must_immediately_own_month_grid() -> None:
    result = _result(
        _seal_table(
            _table(
                "postpaid-history-nonadjacent-identity",
                [
                    ["机构名称", "业务类型", "记账年月"],
                    ["某通信公司", "移动电话", "2024-06"],
                    [""],
                    ["缴费记录", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"],
                    ["2024", *(["N"] * 12)],
                ],
            )
        ),
        role="postpaid_detail",
    )

    records = native_extraction._extract_postpaid_payment_history(result)

    assert len(records) == 12
    assert all(record.get("institution") is None for record in records)
    assert any(
        issue.get("issue_code") == "candidate_b_canonical_header_graph_unresolved"
        and issue.get("target_dataset") == "postpaid_payment_history"
        for issue in _issues(result)
    )


@pytest.mark.parametrize("section", ("public", "postpaid"))
@pytest.mark.parametrize(
    "defect",
    (
        "malformed_container",
        "malformed_line_container",
        "malformed_page_container",
        "malformed_lines_container",
        "cross_cell_replay",
        "far_line_replay",
    ),
)
def test_public_and_postpaid_lattices_reject_malformed_or_replayed_evidence(
    section: str,
    defect: str,
) -> None:
    if section == "public":
        table = _seal_table(
            _table(
                "public-evidence",
                [
                    ["编号", "主管税务机关", "欠税总额", "欠税统计日期"],
                    ["1", "某市税务局", "100", "2024-01-31"],
                ],
            )
        )
        role = "public_information"
        extractor = native_extraction._extract_public_records
    else:
        table = _seal_table(
            _table(
                "postpaid-evidence",
                [
                    ["机构名称", "业务类型", "业务开通日期", "当前缴费状态", "当前欠费金额", "记账年月"],
                    ["某通信公司", "移动电话", "2020-01-02", "正常", "0", "2024-06"],
                ],
            )
        )
        role = "postpaid_detail"
        extractor = native_extraction._extract_postpaid_records
    result = _result(table, role=role)
    evidence = table.metadata["geometry"]["cell_evidence_ids"]
    deciding_id = evidence[0][0][0]
    if defect == "malformed_container":
        evidence[0][0] = deciding_id
    elif defect == "malformed_line_container":
        result.corrected_evidence_pages = lambda: [
            {
                "page": 1,
                "lines": [
                    {
                        "text": "原位别名",
                        "bbox": [10, 2, 90, 18],
                        "evidence_ids": deciding_id,
                    }
                ],
            }
        ]
    elif defect == "malformed_page_container":
        result.corrected_evidence_pages = lambda: [None]
    elif defect == "malformed_lines_container":
        result.corrected_evidence_pages = lambda: [{"page": 1, "lines": "not-a-line-list"}]
    elif defect == "cross_cell_replay":
        evidence[0][1] = [deciding_id]
    else:
        result.corrected_evidence_pages = lambda: [
            {
                "page": 1,
                "lines": [
                    {
                        "text": "远处重放",
                        "bbox": [900, 900, 980, 920],
                        "evidence_ids": [deciding_id],
                    }
                ],
            }
        ]

    assert extractor(result) == []
    assert any(
        issue.get("issue_code") == "candidate_b_canonical_header_graph_unresolved"
        for issue in _issues(result)
    )


@pytest.mark.parametrize("section", ("public", "postpaid"))
def test_exact_contained_line_alias_does_not_count_as_evidence_replay(section: str) -> None:
    if section == "public":
        table = _seal_table(
            _table(
                "public-contained-alias",
                [
                    ["编号", "主管税务机关", "欠税总额", "欠税统计日期"],
                    ["1", "某市税务局", "100", "2024-01-31"],
                ],
            )
        )
        role = "public_information"
        extractor = native_extraction._extract_public_records
    else:
        table = _seal_table(
            _table(
                "postpaid-contained-alias",
                [
                    ["机构名称", "业务类型", "业务开通日期", "当前缴费状态", "当前欠费金额", "记账年月"],
                    ["某通信公司", "移动电话", "2020-01-02", "正常", "0", "2024-06"],
                ],
            )
        )
        role = "postpaid_detail"
        extractor = native_extraction._extract_postpaid_records
    result = _result(table, role=role)
    deciding_id = table.metadata["geometry"]["cell_evidence_ids"][0][0][0]
    result.corrected_evidence_pages = lambda: [
        {
            "page": 1,
            "lines": [
                {
                    "text": table.metadata["raw_rows"][0][0],
                    "bbox": [10, 2, 90, 18],
                    "evidence_ids": [deciding_id],
                }
            ],
        }
    ]

    assert len(extractor(result)) == 1


def test_public_and_postpaid_share_one_document_global_evidence_namespace() -> None:
    public = _seal_table(
        _table(
            "public-global-evidence",
            [
                ["编号", "主管税务机关", "欠税总额", "欠税统计日期"],
                ["1", "某市税务局", "100", "2024-01-31"],
            ],
        )
    )
    postpaid = _seal_table(
        _table(
            "postpaid-global-evidence",
            [
                ["机构名称", "业务类型", "业务开通日期", "当前缴费状态", "当前欠费金额", "记账年月"],
                ["某通信公司", "移动电话", "2020-01-02", "正常", "0", "2024-06"],
            ],
        )
    )
    public.metadata["canonical_template_id"] = "public_information"
    postpaid.metadata["canonical_template_id"] = "postpaid_detail"
    deciding_id = public.metadata["geometry"]["cell_evidence_ids"][0][0][0]
    postpaid.metadata["geometry"]["cell_evidence_ids"][0][0] = [deciding_id]
    result = SimpleNamespace(
        pages=[
            SimpleNamespace(
                page_number=1,
                source_page_number=1,
                canonical_template_id="public_information",
                tables=[public],
            ),
            SimpleNamespace(
                page_number=2,
                source_page_number=2,
                canonical_template_id="postpaid_detail",
                tables=[postpaid],
            ),
        ]
    )

    assert native_extraction._extract_public_records(result) == []
    assert native_extraction._extract_postpaid_records(result) == []
    assert sum(
        issue.get("issue_code") == "candidate_b_canonical_header_graph_unresolved"
        for issue in _issues(result)
    ) == 2


def test_note_table_keeps_blank_text_separate_from_date() -> None:
    result = _result(
        _table(
            "notes",
            [
                ["编号", "标注内容", "添加日期"],
                ["1", "", "2024-01-02"],
            ],
        ),
        role="annotations_and_inquiries",
    )

    annotations, statements = native_extraction._extract_personal_notes(result)

    assert statements == []
    assert len(annotations) == 1
    assert "text" not in annotations[0]
    assert annotations[0]["added_date"] == "2024-01-02"
    assert any(
        issue.get("target_dataset") == "annotation_statements"
        and issue.get("target_record_id") == f"annotation_statement:{annotations[0]['id']}"
        and issue.get("field_name") == "text"
        for issue in _issues(result)
    )


def test_recovery_card_preserves_later_slots_when_amount_is_blank() -> None:
    table = _table(
        "recovery",
        [
            [
                "账户状态",
                "原债权人",
                "管理机构",
                "债权接收日期",
                "余额",
                "业务种类",
                "债权金额",
                "原债务业务种类",
                "账户关闭日期",
                "债权转移时的还款状态",
                "最近一次还款日期",
            ],
            [
                "结清",
                "某银行",
                "某资产公司",
                "2022-01-02",
                "0",
                "资产处置",
                "",
                "个人贷款",
                "2023-02-03",
                "逾期",
                "2023-01-02",
            ],
        ],
    )
    table.metadata["canonical_template_id"] = "credit_account_detail"
    result = _result(table)
    result.pages[0].canonical_template_id = "credit_account_detail"

    records = native_extraction._extract_recovery_records(result)

    assert len(records) == 1
    assert "debt_amount" not in records[0]
    assert records[0]["account_status"] == "结清"
    assert records[0]["balance"] == 0
    assert any(
        issue.get("target_record_id") == records[0]["recovery_record_id"]
        and issue.get("field_name") == "debt_amount"
        for issue in _issues(result)
    )


@pytest.mark.parametrize(
    ("page_role", "table_role"),
    [
        ("public_information", "public_information"),
        ("information_summary", "information_summary"),
        (None, None),
        ("credit_account_detail", "public_information"),
    ],
)
def test_recovery_labels_do_not_cross_section_ownership(
    page_role: str | None,
    table_role: str | None,
) -> None:
    table = _table(
        "foreign-recovery-shape",
        [
            ["原债权人", "债权接收日期", "管理机构", "账户状态"],
            ["某银行", "2024-03-08", "某资产公司", "结清"],
        ],
    )
    if table_role is not None:
        table.metadata["canonical_template_id"] = table_role
    result = _result(table)
    result.pages[0].canonical_template_id = page_role

    assert native_extraction._extract_recovery_records(result) == []


def test_schema_withholds_unknown_public_type_without_generic_extension() -> None:
    source_ref = {
        "ref_type": "source_cell",
        "ref_id": "public:unknown:content",
        "field_name": "content",
        "page": 1,
    }
    projected = project_personal_detail_datasets(
        {
            "public_records": [
                {
                    "record_id": "public:unknown",
                    "public_record_id": "public:unknown",
                    "record_type": "future_public_type",
                    "content": '{"field_a":"alpha"}',
                    "source_refs": [source_ref],
                }
            ]
        }
    )

    assert "pboc_extension_fields" not in projected
    issue = next(
        row
        for row in projected["extraction_issues"]
        if row["issue_code"] == "canonical_public_record_type_unresolved"
    )
    assert issue["target_record_id"] == "public:unknown"
    assert "target_dataset" not in issue
    assert "field_name" not in issue
    assert issue["source_refs"] == [source_ref]
    assert any(
        row.get("evidence_kind") == "observed" and row.get("string_value") == "alpha"
        for row in projected["extraction_issue_evidence"]
    )


def test_schema_known_public_structural_failure_is_fieldless_and_keeps_source() -> None:
    source_ref = {
        "ref_type": "source_cell",
        "ref_id": "public:known:content",
        "field_name": "content",
        "page": 2,
    }
    projected = project_personal_detail_datasets(
        {
            "public_records": [
                {
                    "record_id": "public:known",
                    "public_record_id": "public:known",
                    "record_type": "tax_arrears",
                    "content": "not-structured-json",
                    "source_refs": [source_ref],
                }
            ]
        }
    )

    issue = next(
        row
        for row in projected["extraction_issues"]
        if row["issue_code"] == "canonical_public_content_unresolved"
    )
    assert issue["target_dataset"] == "tax_arrears_records"
    assert issue["target_record_id"] == "public:known"
    assert "field_name" not in issue
    assert issue["observed_value"] == "not-structured-json"
    assert issue["source_refs"] == [source_ref]


def test_schema_removes_noncatalog_public_scalar_and_reports_it() -> None:
    source_ref = {
        "ref_type": "source_cell",
        "ref_id": "tax:1:unmapped",
        "field_name": "unmapped_content",
        "page": 3,
    }
    projected = project_personal_detail_datasets(
        {
            "tax_arrears_records": [
                {
                    "record_id": "tax:1",
                    "tax_arrears_id": "tax:1",
                    "tax_authority": "某税务局",
                    "arrears_amount": 100,
                    "unmapped_content": "多个字段被错误拼接",
                    "source_refs": [source_ref],
                }
            ]
        }
    )

    record = projected["tax_arrears_records"][0]
    assert "unmapped_content" not in record
    assert "unmapped_content" not in record.get("normalized", {})
    issue = next(
        row
        for row in projected["extraction_issues"]
        if row["issue_code"] == "canonical_field_outside_closed_catalog"
    )
    assert issue["target_dataset"] == "tax_arrears_records"
    assert issue["target_record_id"] == "tax:1"
    assert "field_name" not in issue
    assert issue["source_refs"] == [source_ref]
    observed_evidence = {
        (row.get("evidence_path"), row.get("string_value"))
        for row in projected["extraction_issue_evidence"]
        if row.get("extraction_issue_id") == issue["extraction_issue_id"]
        and row.get("evidence_kind") == "observed"
    }
    assert observed_evidence == {
        ("source_field_name", "unmapped_content"),
        ("source_value", "多个字段被错误拼接"),
    }


def test_schema_does_not_treat_noncatalog_dash_as_business_absence() -> None:
    projected = project_personal_detail_datasets(
        {
            "tax_arrears_records": [
                {
                    "record_id": "tax:dash",
                    "tax_arrears_id": "tax:dash",
                    "tax_authority": "某税务局",
                    "arrears_amount": 100,
                    "unmapped_content": "--",
                }
            ]
        }
    )

    record = projected["tax_arrears_records"][0]
    assert "unmapped_content" not in record
    issue = next(
        row
        for row in projected["extraction_issues"]
        if row["issue_code"] == "canonical_field_outside_closed_catalog"
    )
    assert "field_name" not in issue
    assert any(
        row.get("evidence_kind") == "observed"
        and row.get("evidence_path") == "source_value"
        and row.get("string_value") == "--"
        for row in projected["extraction_issue_evidence"]
    )


def test_schema_withholds_known_event_extra_scalar_instead_of_extension() -> None:
    projected = project_personal_detail_datasets(
        {
            "credit_accounts": [
                {"record_id": "account:1", "account_id": "account:1", "account_type": "credit_card"}
            ],
            "personal_detail_account_events": [
                {
                    "record_id": "event:1",
                    "account_event_id": "event:1",
                    "account_id": "account:1",
                    "event_type": "special_event",
                    "details": "已知事件",
                    "future_scalar": "不应发布",
                }
            ],
        }
    )

    event = projected["credit_account_special_events"][0]
    assert event["details"] == "已知事件"
    assert "future_scalar" not in event
    assert "pboc_extension_fields" not in projected
    issue = next(
        row
        for row in projected["extraction_issues"]
        if row["issue_code"] == "canonical_field_outside_closed_catalog"
    )
    assert "field_name" not in issue
    assert any(
        row.get("evidence_kind") == "observed"
        and row.get("evidence_path") == "source_field_name"
        and row.get("string_value") == "future_scalar"
        for row in projected["extraction_issue_evidence"]
    )


def test_schema_postpaid_month_requires_canonical_parent() -> None:
    projected = project_personal_detail_datasets(
        {
            "postpaid_payment_history": [
                {
                    "record_id": "month:1",
                    "postpaid_payment_history_id": "month:1",
                    "postpaid_record_id": "postpaid:missing",
                    "institution": "某通信公司",
                    "business_type": "移动电话",
                    "year": 2024,
                    "month": 1,
                    "status": "N",
                }
            ]
        }
    )

    month = projected["postpaid_monthly_performance"][0]
    assert month["postpaid_record_id"] is None
    assert any(
        row["issue_code"] == "unresolved_postpaid_parent_identity"
        and row["target_record_id"] == "month:1"
        for row in projected["extraction_issues"]
    )


def test_schema_postpaid_month_inherits_complete_parent_identity_silently() -> None:
    projected = project_personal_detail_datasets(
        {
            "postpaid_records": [
                {
                    "record_id": "postpaid:1",
                    "postpaid_record_id": "postpaid:1",
                    "institution": "某通信公司",
                    "business_type": "移动电话",
                    "billing_month": "2024-01",
                }
            ],
            "postpaid_payment_history": [
                {
                    "record_id": "month:1",
                    "postpaid_payment_history_id": "month:1",
                    "postpaid_record_id": "postpaid:1",
                    "year": 2024,
                    "month": 1,
                    "status": "N",
                }
            ],
        }
    )

    month = projected["postpaid_monthly_performance"][0]
    assert month["institution"] == "某通信公司"
    assert month["business_type"] == "移动电话"
    assert "extraction_issues" not in projected


def test_schema_postpaid_month_reports_incomplete_but_linked_parent_identity() -> None:
    projected = project_personal_detail_datasets(
        {
            "postpaid_records": [
                {
                    "record_id": "postpaid:partial",
                    "postpaid_record_id": "postpaid:partial",
                    "institution": "某通信公司",
                    "business_type": None,
                    "billing_month": "2024-01",
                }
            ],
            "postpaid_payment_history": [
                {
                    "record_id": "month:partial",
                    "postpaid_payment_history_id": "month:partial",
                    "postpaid_record_id": "postpaid:partial",
                    "year": 2024,
                    "month": 1,
                    "status": "N",
                }
            ],
        }
    )

    month = projected["postpaid_monthly_performance"][0]
    assert month["postpaid_record_id"] == "postpaid:partial"
    assert any(
        row["issue_code"] == "postpaid_parent_identity_incomplete"
        and row["target_record_id"] == "month:partial"
        for row in projected["extraction_issues"]
    )
