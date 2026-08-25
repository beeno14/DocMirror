# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from docmirror.plugins.credit_report.personal_detail_scanned.native_extraction import (
    _enforce_employment_record_contracts,
    _extract_employment_records,
    _finite_employment_value,
)


def test_employment_vocab_does_not_delete_or_insert_punctuation() -> None:
    assert _finite_employment_value("高级,领导", ("高级领导",)) is None
    assert _finite_employment_value("个体私营企业", ("个体、私营企业",)) is None
    assert _finite_employment_value("高级 领导", ("高级领导",)) == "高级领导"


def _result(
    *,
    address: str = "福建省厦门市思明区样例路1号",
    industry: str = "制造业",
    occupation: str = "专业技术人员",
    position: str = "一般员工",
    professional_title: str = "中级",
) -> SimpleNamespace:
    rows = [
        ["编号", "工作单位", "单位性质", "单位地址", "单位电话", "", ""],
        ["1", "样例科技有限公司", "私营企业", address, "010-12345678", "", ""],
        ["编号", "职业", "行业", "职务", "职称", "进入本单位年份", "信息更新日期"],
        ["1", occupation, industry, position, professional_title, "2020", "2025.01.02"],
        ["编号", "数据发生机构名称", "", "", "", "", ""],
        ["1", "样例银行股份有限公司", "", "", "", "", ""],
    ]
    row_bands = [
        {"index": row, "y0": row * 19.0, "y1": (row + 1) * 19.0}
        for row in range(len(rows))
    ]
    widths = (47.0, 193.0, 81.0, 211.0, 93.0, 57.0, 119.0)
    edges = [0.0]
    for width in widths:
        edges.append(edges[-1] + width)
    col_bands = [
        {"index": column, "x0": edges[column], "x1": edges[column + 1]}
        for column in range(len(widths))
    ]
    evidence = [
        [([f"contract:{row}:{column}"] if str(value).strip() else []) for column, value in enumerate(values)]
        for row, values in enumerate(rows)
    ]
    table = SimpleNamespace(
        table_id="employment-contracts",
        metadata={
            "raw_rows": rows,
            "canonical_template_id": "report_header_and_identity",
            "source_logical_page": 2,
            "source_page": 1,
            "geometry": {
                "coordinate_system": "pdf_points_top_left",
                "row_bands": row_bands,
                "col_bands": col_bands,
                "cell_bboxes": [
                    [
                        [edges[column], row * 19.0, edges[column + 1], (row + 1) * 19.0]
                        for column in range(len(widths))
                    ]
                    for row in range(len(rows))
                ],
                "cell_geometry_status": [["exact"] * len(widths) for _row in rows],
                "cell_evidence_ids": evidence,
                "cell_token_ids": deepcopy(evidence),
                "cell_spans": [],
            },
        },
        headers=[],
        rows=[],
        bbox=[20, 20, 580, 700],
    )
    return SimpleNamespace(
        pages=[
            SimpleNamespace(
                page_number=2,
                source_page_number=1,
                canonical_template_id="report_header_and_identity",
                canonical_fragment_logical_pages=(2,),
                coordinate_transform={"source_page_numbers": [1]},
                tables=[table],
            )
        ],
        tables_continue=lambda _left, _right: False,
        _personal_detail_extraction_issues=[],
    )


def test_distinct_employment_columns_publish_exact_finite_values() -> None:
    result = _result()

    records = _extract_employment_records(result)

    assert len(records) == 1
    record = records[0]
    assert {
        field: record[field]
        for field in ("industry", "occupation", "position", "professional_title")
    } == {
        "industry": "制造业",
        "occupation": "专业技术人员",
        "position": "一般员工",
        "professional_title": "中级",
    }
    assert result._personal_detail_extraction_issues == []


@pytest.mark.parametrize(
    ("field_name", "raw"),
    (
        ("industry", "梦行业"),
        ("occupation", "4人"),
        ("position", "签月**"),
        ("professional_title", "中级38"),
    ),
)
def test_distinct_employment_columns_withhold_noncanonical_scalars(
    field_name: str, raw: str
) -> None:
    kwargs = {field_name: raw}
    result = _result(**kwargs)

    record = _extract_employment_records(result)[0]

    assert field_name not in record
    assert record["canonical_raw"][field_name] == [raw]
    assert any(
        issue.get("issue_code") == "candidate_b_exact_slot_value_invalid"
        and issue.get("target_record_id") == record["employment_record_id"]
        and issue.get("field_name") == field_name
        and issue.get("observed_value") == [raw]
        for issue in result._personal_detail_extraction_issues
    )


def test_employer_address_repeating_employer_is_withheld_whole() -> None:
    address = "福建省厦门市样例科技有限公司思明区样例路1号"
    result = _result(address=address)

    record = _extract_employment_records(result)[0]

    assert "employer_address" not in record
    assert record["canonical_raw"]["employer_address"] == [address]
    assert any(
        issue.get("issue_code") == "candidate_b_exact_slot_value_invalid"
        and issue.get("field_name") == "employer_address"
        and issue.get("observed_value") == [address]
        for issue in result._personal_detail_extraction_issues
    )


@pytest.mark.parametrize("suffix", ("一般员工", "中级"))
def test_employer_address_with_appended_position_or_title_is_withheld(suffix: str) -> None:
    address = f"福建省厦门市思明区样例路1号 {suffix}"
    result = _result(address=address)

    record = _extract_employment_records(result)[0]

    assert "employer_address" not in record
    assert record["canonical_raw"]["employer_address"] == [address]
    assert any(
        issue.get("field_name") == "employer_address"
        and issue.get("observed_value") == [address]
        for issue in result._personal_detail_extraction_issues
    )


def test_explicit_absent_distinct_employment_scalars_remain_source_absent() -> None:
    result = _result(
        industry="--",
        occupation="--",
        position="--",
        professional_title="--",
    )

    record = _extract_employment_records(result)[0]

    assert {"industry", "occupation", "position", "professional_title"} <= set(
        record["_source_absent_fields"]
    )
    assert not any(
        issue.get("issue_code") == "candidate_b_exact_slot_value_invalid"
        and issue.get("field_name")
        in {"industry", "occupation", "position", "professional_title"}
        for issue in result._personal_detail_extraction_issues
    )


def test_legitimate_address_with_internal_title_word_is_not_withheld() -> None:
    # Short title glyphs are not treated as contamination when they are part
    # of an address-internal proper name rather than an appended field token.
    address = "福建省厦门市中级人民法院旁样例路1号"
    result = _result(address=address)

    record = _extract_employment_records(result)[0]

    assert record["employer_address"] == address


def test_employer_phone_normalizes_one_layout_spaced_value_to_digits() -> None:
    result = _result()
    rows = result.pages[0].tables[0].metadata["raw_rows"]
    rows[1][4] = "010 12345678"

    record = _extract_employment_records(result)[0]

    assert record["employer_phone"] == "01012345678"
    assert record["canonical_raw"]["employer_phone"] == "010 12345678"


@pytest.mark.parametrize(
    "raw",
    (
        "059215250467 509",
        "01012345678 销售部",
        "01012345678A",
        "01012345678 13900000000",
        "010--12345678",
        "010))12345678",
        "1-2-3-4-5",
        "12345",
    ),
)
def test_employer_phone_with_residue_or_multiple_business_values_is_withheld(
    raw: str,
) -> None:
    result = _result()
    rows = result.pages[0].tables[0].metadata["raw_rows"]
    rows[1][4] = raw

    record = _extract_employment_records(result)[0]

    assert "employer_phone" not in record
    assert record["canonical_raw"]["employer_phone"] == [raw]
    assert record["source_refs_by_field"]["employer_phone"][0]["geometry_scope"] == "cell"
    assert any(
        issue.get("issue_code") == "candidate_b_exact_slot_value_invalid"
        and issue.get("field_name") == "employer_phone"
        and issue.get("observed_value") == [raw]
        and issue.get("source_refs", [{}])[0].get("geometry_scope") == "cell"
        for issue in result._personal_detail_extraction_issues
    )


def test_address_with_full_employer_and_exact_department_suffix_is_retained() -> None:
    address = "福建省厦门市思明区样例路1号样例科技有限公司销售部"
    result = _result(address=address)

    record = _extract_employment_records(result)[0]

    assert record["employer_address"] == address
    assert not any(
        issue.get("field_name") == "employer_address"
        for issue in result._personal_detail_extraction_issues
    )


def test_bare_employer_and_department_is_not_promoted_to_an_address() -> None:
    address = "样例科技有限公司销售部"
    result = _result(address=address)

    record = _extract_employment_records(result)[0]

    assert "employer_address" not in record
    assert any(
        issue.get("field_name") == "employer_address"
        for issue in result._personal_detail_extraction_issues
    )


def _final_merged_employment(address: str) -> dict[str, object]:
    return {
        "record_id": "employment:final:1",
        "employment_record_id": "employment:final:1",
        "employer": "福州众恒房产代理有限公司",
        "employer_address": address,
        "position": "一般员工",
        "professional_title": "中级",
        "source_refs": [{"logical_page": 2, "table_id": "employment-final", "row": 1}],
        "source_refs_by_field": {
            "employer_address": [
                {
                    "logical_page": 2,
                    "table_id": "employment-final",
                    "row": 1,
                    "column": 3,
                }
            ]
        },
        "canonical_raw": {"employer_address": address},
    }


def test_final_merged_employment_contract_withholds_long_same_employer_prefix() -> None:
    address = "福建省福州市仓山区金山碧水中区朱菊苑4号楼04店面福州众恒房产代"
    record = _final_merged_employment(address)
    result = SimpleNamespace(_personal_detail_extraction_issues=[])

    _enforce_employment_record_contracts(result, [record])

    assert "employer_address" not in record
    assert record["canonical_raw"]["employer_address"] == [address]
    assert any(
        issue.get("target_record_id") == "employment:final:1"
        and issue.get("field_name") == "employer_address"
        and issue.get("observed_value") == [address]
        for issue in result._personal_detail_extraction_issues
    )


def test_final_merged_employment_contract_keeps_clean_address_silent() -> None:
    address = "福建省福州市仓山区金山碧水中区朱菊苑4号楼04店面"
    record = _final_merged_employment(address)
    result = SimpleNamespace(_personal_detail_extraction_issues=[])

    _enforce_employment_record_contracts(result, [record])

    assert record["employer_address"] == address
    assert result._personal_detail_extraction_issues == []
