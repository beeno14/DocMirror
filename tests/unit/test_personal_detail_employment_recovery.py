# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from docmirror.plugins.credit_report.personal_detail_scanned.native_extraction import (
    _employment_provider_header,
    _extract_employment_records,
    _strict_employment_provider_span,
)
from tests.unit.personal_detail_employment_test_support import own_employment_table


def _table(
    *rows: list[str],
    exact_blank_slots: tuple[tuple[int, int], ...] = (),
) -> SimpleNamespace:
    return SimpleNamespace(
        table_id="employment",
        metadata={"raw_rows": list(rows)},
        headers=[],
        rows=[],
        bbox=[20, 20, 580, 700],
        exact_blank_slots=exact_blank_slots,
    )


def _result(table: SimpleNamespace) -> SimpleNamespace:
    logical_page = 2
    source_page = 1
    if "geometry" not in table.metadata:
        own_employment_table(
            table,
            logical_page=logical_page,
            source_page=source_page,
        )
    else:
        # The merged-ordinal fixtures deliberately carry custom exact spans.
        # Preserve that topology while attaching the same generic PBOC owner.
        table.metadata.update(
            {
                "canonical_template_id": "report_header_and_identity",
                "source_logical_page": logical_page,
                "source_page": source_page,
            }
        )
    geometry = table.metadata["geometry"]
    raw_rows = table.metadata["raw_rows"]
    for row, column in getattr(table, "exact_blank_slots", ()):
        assert not str(raw_rows[row][column]).strip()
        evidence_ids = [f"employment:exact-blank:{table.table_id}:{row}:{column}"]
        geometry["cell_evidence_ids"][row][column] = evidence_ids
        geometry["cell_token_ids"][row][column] = list(evidence_ids)
    return SimpleNamespace(
        pages=[
            SimpleNamespace(
                page_number=logical_page,
                source_page_number=source_page,
                canonical_template_id="report_header_and_identity",
                canonical_fragment_logical_pages=(logical_page,),
                coordinate_transform={"source_page_numbers": [source_page]},
                tables=[table],
                texts=[],
            )
        ],
        tables_continue=lambda _left, _right: False,
    )


def _exact_employment_geometry(rows: list[list[str]]) -> dict[str, object]:
    """Build an exact five-column lattice with no synthetic business evidence."""

    width = 100.0
    height = 20.0
    row_bands = [
        {"index": row, "y0": row * height, "y1": (row + 1) * height}
        for row in range(len(rows))
    ]
    col_bands = [
        {"index": column, "x0": column * width, "x1": (column + 1) * width}
        for column in range(5)
    ]
    bboxes = [
        [
            [column * width, row * height, (column + 1) * width, (row + 1) * height]
            for column in range(5)
        ]
        for row in range(len(rows))
    ]
    statuses = [["exact"] * 5 for _row in rows]
    evidence = [
        [[f"ocr:test:{row}:{column}"] for column in range(5)]
        for row in range(len(rows))
    ]
    return {
        "coordinate_system": "pdf_points_top_left",
        "row_bands": row_bands,
        "col_bands": col_bands,
        "cell_bboxes": bboxes,
        "cell_geometry_status": statuses,
        "cell_evidence_ids": evidence,
        "cell_token_ids": deepcopy(evidence),
        "cell_spans": [],
    }


def _set_exact_span(
    geometry: dict[str, object],
    *,
    row: int,
    column: int,
    row_span: int,
    column_span: int,
) -> None:
    """Replace unit cells with one exact span and empty derived covered slots."""

    row_bands = geometry["row_bands"]
    col_bands = geometry["col_bands"]
    bboxes = geometry["cell_bboxes"]
    statuses = geometry["cell_geometry_status"]
    evidence = geometry["cell_evidence_ids"]
    tokens = geometry["cell_token_ids"]
    owner_bbox = [
        col_bands[column]["x0"],
        row_bands[row]["y0"],
        col_bands[column + column_span - 1]["x1"],
        row_bands[row + row_span - 1]["y1"],
    ]
    bboxes[row][column] = owner_bbox
    owner_evidence = evidence[row][column]
    geometry["cell_spans"].append(
        {
            "row": row,
            "col": column,
            "row_span": row_span,
            "col_span": column_span,
            "bbox": owner_bbox,
            "evidence_ids": list(owner_evidence),
        }
    )
    for covered_row in range(row, row + row_span):
        for covered_column in range(column, column + column_span):
            if (covered_row, covered_column) == (row, column):
                continue
            bboxes[covered_row][covered_column] = None
            statuses[covered_row][covered_column] = "derived"
            evidence[covered_row][covered_column] = []
            tokens[covered_row][covered_column] = []


def _merged_ordinal_employment_table(*, nonterminal: bool = False) -> SimpleNamespace:
    """Create a generic exact table with a two-row ordinal owner and detail grid."""

    rows = [
        ["编号", "工作单位", "单位性质", "单位地址", "单位电话"],
        ["1", "甲一科技有限公司", "私营企业", "福建省福州市甲一路1号", "01011111111"],
        ["2", "乙二科技有限公司", "私营企业", "福建省福州市乙二路2号", "01022222222"],
        ["3", "丙三科技有限公司", "私营企业", "福建省福州市丙三路3号", "01033333333"],
        ["5", "丁四科技有限公司", "私营企业", "福建省福州市丁四路4号", "01044444444"],
        ["", "戊五工程有限公司", "福建省福州市戊五路501- 01055555555 502室", "", ""],
    ]
    if nonterminal:
        rows.append(["6", "己六科技有限公司", "私营企业", "福建省福州市己六路6号", "01066666666"])
    detail_header = len(rows)
    rows.append(["编号", "行业 信息更新日期 职务 进入本单位年份 职业 职称", "", "", ""])
    detail_count = 6 if nonterminal else 5
    for sequence in range(1, detail_count + 1):
        rows.append(
            [
                str(sequence),
                f"专业技术人员 信息传输、软件和信息技术服务业 无 2025.01.{sequence:02d} 一般员工 2020",
                "",
                "",
                "",
            ]
        )
    geometry = _exact_employment_geometry(rows)
    _set_exact_span(geometry, row=4, column=0, row_span=2, column_span=1)
    _set_exact_span(geometry, row=5, column=2, row_span=1, column_span=3)
    _set_exact_span(geometry, row=detail_header, column=1, row_span=1, column_span=4)
    for row in range(detail_header + 1, len(rows)):
        _set_exact_span(geometry, row=row, column=1, row_span=1, column_span=4)
    return SimpleNamespace(
        table_id="employment-merged-ordinal",
        metadata={"raw_rows": rows, "geometry": geometry},
        headers=[],
        rows=[],
        bbox=[0.0, 0.0, 500.0, len(rows) * 20.0],
    )


def test_employment_recovers_only_validated_fields_from_hong_shaped_ocr() -> None:
    result = _result(
        _table(
            ["编母", "美 工作单位", "单位性质 个体私 营企业", "单位地址", "单位电话", ""],
            ["1", "显微镜(厦门)贸易有限公司", "", "中国福建省厦门市思明区台车路157号 404之六", "059215250467 509", ""],
            ["子 2 2", "厦门阀竹源贸易有限公司", "个体、私 营企业", "福建省厦门市思明区塔埔东路166号观音山 国际商务营运中心11号楼3楼", "059213950015 475", ""],
            ["3", "福建省南安市石井镇岑兜村石基牛片区 46号", "其他包 括三资企 业、民营 企业、民 间团体 等)", "*a", "", ""],
            ["编号", "行业 职业", "", "进入本单位年份 职称 取务", "信息更新日期", ""],
            ["1", '" 商业、服务业人员', "", "2024 一般员工 “", "2025.05.15", ""],
            ["2", "批发和零售业 办事人员和有关人员", "", '一般员工 “ "', "2024.07.17", ""],
            ["3", "不便分类的其他从业 ** 人员", "", '无 * ""', "2022.11.14", ""],
            ["编号", "", "", "数据发生机构名称", "", ""],
            ["1", "", "", "中国工商银行股份有限公司厦门市分行", "", ""],
            ["2", "", "", "华夏银行股份有限公司信用卡中心", "", ""],
            ["3", "$", "", "兰州银行股份有限公司", "", ""],
            exact_blank_slots=((1, 2),),
        )
    )

    records = _extract_employment_records(result)

    assert [record["sequence"] for record in records] == [1, 2, 3]
    first, second, third = records
    assert first["employer"] == "显微镜(厦门)贸易有限公司"
    assert "employer_type" not in first
    assert first["employer_address"] == "中国福建省厦门市思明区台车路157号 404之六"
    assert first["occupation"] == "商业、服务业人员"
    assert first["position"] == "一般员工"
    assert first["entry_year"] == 2024
    assert first["information_updated_date"] == "2025-05-15"
    assert first["data_provider"] == "中国工商银行股份有限公司厦门市分行"

    assert second["employer"] == "厦门阀竹源贸易有限公司"
    assert second["employer_type"] == "个体、私营企业"
    assert second["industry"] == "批发和零售业"
    assert second["occupation"] == "办事人员和有关人员"
    assert second["position"] == "一般员工"
    assert second["information_updated_date"] == "2024-07-17"

    assert third["employer"] == "福建省南安市石井镇岑兜村石基牛片区 46号"
    assert "employer_type" not in third
    assert third["canonical_raw"]["employer_type"] == [
        "其他包 括三资企 业、民营 企业、民 间团体 等)"
    ]
    assert "occupation" not in third
    assert "position" not in third
    assert third["professional_title"] == "无"
    assert third["information_updated_date"] == "2022-11-14"

    # Watermark-contaminated phones and the non-address glyph are evidence,
    # never silently accepted business values.
    assert all("employer_phone" not in record for record in records)
    assert "employer_address" not in third
    invalid_fields = {
        (issue.get("target_record_id"), issue.get("field_name"))
        for issue in result._personal_detail_extraction_issues
        if issue["issue_code"] == "candidate_b_exact_slot_value_invalid"
    }
    assert (first["employment_record_id"], "employer_phone") in invalid_fields
    assert (second["employment_record_id"], "employer_phone") in invalid_fields
    assert (third["employment_record_id"], "employer_type") in invalid_fields
    assert (third["employment_record_id"], "employer_address") in invalid_fields
    assert any(
        issue["issue_code"] == "candidate_b_employment_recovered_header_cell_unresolved"
        and issue.get("target_record_id") == first["employment_record_id"]
        and issue.get("field_name") == "employer_type"
        for issue in result._personal_detail_extraction_issues
    )

    unresolved = {
        (issue.get("target_record_id"), issue.get("field_name"))
        for issue in result._personal_detail_extraction_issues
        if issue["issue_code"] == "candidate_b_employment_cluster_field_unresolved"
    }
    assert (first["employment_record_id"], "industry") in unresolved
    assert (second["employment_record_id"], "entry_year") in unresolved
    assert (third["employment_record_id"], "occupation") in unresolved
    assert (third["employment_record_id"], "position") in unresolved
    assert not any(
        issue["issue_code"] == "candidate_b_employment_component_missing"
        for issue in result._personal_detail_extraction_issues
    )
    assert any(
        issue["issue_code"] == "candidate_b_employment_cluster_residue_unresolved"
        and issue.get("target_record_id") == third["employment_record_id"]
        and issue.get("field_name") == "professional_title"
        and "candidate_value_retained_with_uncertainty" in issue.get("reason_codes", ())
        for issue in result._personal_detail_extraction_issues
    )


def test_employment_cluster_retains_full_raw_and_reports_unconsumed_residue() -> None:
    main_raw = "制造业 专业技术人员 乱码"
    secondary_raw = "2024 一般员工 无"
    result = _result(
        _table(
            ["编号", "工作单位", "单位性质", "单位地址", "单位电话"],
            ["1", "样例公司", "私营企业", "福建省福州市样例路1号", "010-12345678"],
            ["编号", "行业 职业", "", "进入本单位年份 职称 职务", "信息更新日期"],
            ["1", main_raw, "", secondary_raw, "1999.01.02"],
        )
    )

    record = _extract_employment_records(result)[0]

    assert record["industry"] == "制造业"
    assert record["occupation"] == "专业技术人员"
    assert record["position"] == "一般员工"
    assert record["professional_title"] == "无"
    assert record["entry_year"] == 2024
    assert record["information_updated_date"] == "1999-01-02"
    assert record["canonical_raw"]["industry"] == main_raw
    assert record["canonical_raw"]["occupation"] == main_raw
    residue_issues = [
        issue
        for issue in result._personal_detail_extraction_issues
        if issue["issue_code"] == "candidate_b_employment_cluster_residue_unresolved"
    ]
    assert {issue["field_name"] for issue in residue_issues} == {"industry", "occupation"}
    assert all(
        issue["observed_value"]
        == {"raw_cluster": main_raw, "unconsumed_residue": "乱码"}
        for issue in residue_issues
    )


def test_employment_does_not_infer_sequence_from_noncanonical_column_order() -> None:
    result = _result(
        _table(
            ["编母", "单位性质", "工作单位", "单位地址", "单位电话"],
            ["1", "私营企业", "样例公司", "样例路1号", "010-12345678"],
        )
    )

    assert _extract_employment_records(result) == []
    assert any(
        issue["issue_code"] == "candidate_b_canonical_header_graph_unresolved"
        for issue in result._personal_detail_extraction_issues
    )


def test_employment_cluster_never_guesses_unknown_or_cross_role_values() -> None:
    result = _result(
        _table(
            ["编号", "工作单位", "单位性质", "单位地址", "单位电话"],
            ["2", "样例公司", "私营企业", "样例路2号", "010-12345678"],
            ["编号", "行业 职业", "", "进入本单位年份 职称 职务", "信息更新日期"],
            ["2", "神秘行业 神秘职业", "", "未知", "2025.01.02"],
        )
    )

    records = _extract_employment_records(result)

    assert len(records) == 1
    record = records[0]
    assert "industry" not in record
    assert "occupation" not in record
    # "未知" belongs to more than one field vocabulary in the merged cell,
    # so it cannot be assigned to either role.
    assert "position" not in record
    assert "professional_title" not in record
    unresolved_fields = {
        issue.get("field_name")
        for issue in result._personal_detail_extraction_issues
        if issue["issue_code"] == "candidate_b_employment_cluster_field_unresolved"
    }
    assert {"industry", "occupation", "position", "professional_title"} <= unresolved_fields


def test_employment_cluster_never_spell_corrects_a_business_value() -> None:
    result = _result(
        _table(
            ["编号", "工作单位", "单位性质", "单位地址", "单位电话"],
            ["1", "样例公司", "私营企业", "样例路1号", "010-12345678"],
            ["编号", "行业 职业", "", "进入本单位年份 职称 职务", "信息更新日期"],
            ["1", "专业技木人员", "", "一般员工", "2025.01.02"],
        )
    )

    record = _extract_employment_records(result)[0]

    assert "occupation" not in record
    assert any(
        issue["issue_code"] == "candidate_b_employment_cluster_field_unresolved"
        and issue.get("field_name") == "occupation"
        for issue in result._personal_detail_extraction_issues
    )


def test_employment_sequence_repair_requires_repeated_identical_digits() -> None:
    result = _result(
        _table(
            ["编号", "工作单位", "单位性质", "单位地址", "单位电话"],
            ["子 2 3", "不可绑定公司", "个体、私营企业", "样例路23号", "010-12345678"],
        )
    )

    assert _extract_employment_records(result) == []
    assert any(
        issue["issue_code"] == "candidate_b_sequence_cell_unresolved"
        for issue in result._personal_detail_extraction_issues
    )


def test_employment_header_residue_is_never_used_as_an_immediate_or_later_value() -> None:
    result = _result(
        _table(
            ["编母", "噪 工作单位", "单位性质", "单位地址", "单位电话"],
            ["1", "", "私营企业", "样例路1号", "010-12345678"],
            ["2", "第二家公司", "私营企业", "样例路2号", "010-87654321"],
            exact_blank_slots=((1, 1),),
        )
    )

    records = _extract_employment_records(result)

    assert [record["sequence"] for record in records] == [1, 2]
    assert "employer" not in records[0]
    assert records[1]["employer"] == "第二家公司"
    assert any(
        issue["issue_code"] == "candidate_b_employment_recovered_header_cell_unresolved"
        and issue["target_record_id"] == records[0]["employment_record_id"]
        and issue["field_name"] == "employer"
        for issue in result._personal_detail_extraction_issues
    )
    assert any(
        issue["issue_code"] == "candidate_b_canonical_header_graph_unresolved"
        and issue["observed_value"]["physical_header_cells"][1] == "噪 工作单位"
        for issue in result._personal_detail_extraction_issues
    )


def test_canonical_employment_runs_recover_bracketed_and_population_ordinals() -> None:
    result = _result(
        _table(
            ["编号", "工作单位", "单位性质", "单位地址", "单位电话"],
            ["1", "样例甲科技有限公司", "私营企业", "样例路1号", "010 12345678"],
            ["2", "样例乙科技有限公司", "私营企业", "样例路2号", "010 22345678"],
            ["3", "样例丙科技有限公司", "私营企业", "样例路3号", "010 32345678"],
            ["", "样例丁科技有限公司", "私营企业", "样例路4号", "010 42345678"],
            ["5", "样例戊科技有限公司", "私营企业", "样例路5号", "010 52345678"],
            ["编号", "职业 行业", "", "进入本单位年份 职称 职务", "信息更新日期"],
            ["", "专业技术人员", "", "一般员工", "2025.01.01"],
            ["2", "制造业", "", "--", "2025.01.02"],
            ["3", "办事人员和有关人员", "", "中级领导", "2025.01.03"],
            ["4", "制造业", "", "无 一般员工", "2025.01.04"],
            ["点", "专业技术人员", "制造业", "2013 --", "2025.01.05"],
            ["编号", "", "", "数据发生机构名称", ""],
            ["1", "样例甲银行股份有限公司", "", "", ""],
            ["2", "样例乙银行股份有限公司", "", "", ""],
            ["3", "样例丙银行股份有限公司", "", "", ""],
            ["4", "样例丁银行股份有限公司", "", "", ""],
            ["5", "样例戊银行股份有限公司", "", "", ""],
        )
    )

    records = _extract_employment_records(result)

    assert [record["sequence"] for record in records] == [1, 2, 3, 4, 5]
    first, second, third, fourth, fifth = records
    assert first["occupation"] == "专业技术人员"
    assert first["position"] == "一般员工"
    assert first["information_updated_date"] == "2025-01-01"
    assert second["industry"] == "制造业"
    assert second["information_updated_date"] == "2025-01-02"
    assert third["occupation"] == "办事人员和有关人员"
    assert third["position"] == "中级领导"
    assert fourth["industry"] == "制造业"
    assert fourth["position"] == "一般员工"
    assert "professional_title" not in fourth
    assert fifth["occupation"] == "专业技术人员"
    assert fifth["industry"] == "制造业"
    assert fifth["entry_year"] == 2013
    assert [record["data_provider"] for record in records] == [
        "样例甲银行股份有限公司",
        "样例乙银行股份有限公司",
        "样例丙银行股份有限公司",
        "样例丁银行股份有限公司",
        "样例戊银行股份有限公司",
    ]
    assert all(record["employer_phone"].isdigit() for record in records)
    assert not any(
        issue["issue_code"] in {
            "candidate_b_continuation_sequence_unresolved",
            "candidate_b_employment_component_missing",
            "candidate_b_employment_provider_cell_unresolved",
        }
        for issue in result._personal_detail_extraction_issues
    )


def test_employment_sequence_repair_rejects_unbracketed_or_conflicting_runs() -> None:
    result = _result(
        _table(
            ["编号", "工作单位", "单位性质", "单位地址", "单位电话"],
            ["", "不可绑定甲公司", "私营企业", "样例路1号", "01012345678"],
            ["2", "可绑定乙公司", "私营企业", "样例路2号", "01022345678"],
            ["9", "可绑定异常公司", "私营企业", "样例路3号", "01032345678"],
            ["", "不可绑定丁公司", "私营企业", "样例路4号", "01042345678"],
            ["5", "可绑定戊公司", "私营企业", "样例路5号", "01052345678"],
        )
    )

    records = _extract_employment_records(result)

    assert [record["sequence"] for record in records] == [2, 5, 9]
    assert sum(
        issue["issue_code"] == "candidate_b_sequence_cell_unresolved"
        for issue in result._personal_detail_extraction_issues
    ) == 2


def test_provider_row_with_multiple_institution_cells_is_withheld_and_reported() -> None:
    result = _result(
        _table(
            ["编号", "工作单位", "单位性质", "单位地址", "单位电话"],
            ["1", "样例科技有限公司", "私营企业", "样例路1号", "01012345678"],
            ["编号", "", "数据发生机构名称", ""],
            ["1", "甲银行股份有限公司", "乙银行股份有限公司", ""],
        )
    )

    record = _extract_employment_records(result)[0]

    assert "data_provider" not in record
    assert any(
        issue["issue_code"] == "candidate_b_employment_provider_cell_unresolved"
        and issue["field_name"] == "data_provider"
        and issue["observed_value"]["resolution"] == "ambiguous"
        for issue in result._personal_detail_extraction_issues
    )


def test_employment_sequence_repair_never_replaces_printed_dash() -> None:
    result = _result(
        _table(
            ["编号", "工作单位", "单位性质", "单位地址", "单位电话"],
            ["1", "样例甲公司", "私营企业", "样例路1号", "01012345678"],
            ["--", "不可绑定公司", "私营企业", "样例路2号", "01022345678"],
            ["3", "样例丙公司", "私营企业", "样例路3号", "01032345678"],
        )
    )

    records = _extract_employment_records(result)

    assert [record["sequence"] for record in records] == [1, 3]
    assert any(
        issue["issue_code"] == "candidate_b_sequence_cell_unresolved"
        and issue["observed_value"]["physical_cells"][0] == "--"
        for issue in result._personal_detail_extraction_issues
    )


def test_invalid_update_date_year_is_not_reused_as_entry_year() -> None:
    result = _result(
        _table(
            ["编号", "工作单位", "单位性质", "单位地址", "单位电话"],
            ["1", "样例公司", "私营企业", "样例路1号", "01012345678"],
            ["编号", "职业 行业", "", "进入本单位年份 职称 职务", "信息更新日期"],
            ["1", "专业技术人员", "", "一般员工", "2025.13.40"],
        )
    )

    record = _extract_employment_records(result)[0]

    assert "entry_year" not in record
    assert "information_updated_date" not in record
    unresolved = {
        issue.get("field_name")
        for issue in result._personal_detail_extraction_issues
        if issue["issue_code"] == "candidate_b_employment_cluster_field_unresolved"
    }
    assert {"entry_year", "information_updated_date"} <= unresolved


def test_collapsed_detail_date_residue_is_reported_only_on_the_date_field() -> None:
    result = _result(
        _table(
            ["编号", "工作单位", "单位性质", "单位地址", "单位电话"],
            ["1", "样例公司", "私营企业", "样例路1号", "01012345678"],
            ["编号", "职业 行业", "", "进入本单位年份 职称 职务", "信息更新日期"],
            ["1", "专业技术人员", "", "一般员工", "2025.01.02 x"],
        )
    )

    record = _extract_employment_records(result)[0]

    assert record["information_updated_date"] == "2025-01-02"
    residue_issues = [
        issue
        for issue in result._personal_detail_extraction_issues
        if issue["issue_code"] == "candidate_b_employment_cluster_residue_unresolved"
        and issue["observed_value"]["unconsumed_residue"] == "x"
    ]
    assert [issue["field_name"] for issue in residue_issues] == [
        "information_updated_date"
    ]


def test_lin_shaped_employment_rows_recover_only_exact_roles_and_ordinal_providers() -> None:
    result = _result(
        _table(
            ["编号", "工作单位", "单位性质", "福 单位地址", "单位电话", ""],
            ["1", "样例甲科技有限公司", "", "福建省福州市样例路1号", "01012345671", ""],
            ["2", "样例乙科技有限公司", "", "福建省福州市样例路2号", "01012345672", ""],
            ["3", "样例丙科技有限公司", "", "福建省福州市样例路3号", "01012345673", ""],
            ["4", "样例丁科技有限公司", "", "福建省福州市样例路4号", "01012345674", ""],
            ["5", "样例戊科技有限公司", "", "福建省福州市样例路5号", "01012345675", ""],
            ["编号", "? 职业 行业", "", "职务 职称 进入本单位年份", "信息更新日期", ""],
            ["", "商业、服务业人员 -", "", "一般员工 多", "2022.11.29", ""],
            [
                "2",
                "安 不便分类的其他从业 批发和零售业 2022.09.14 1 入员 等",
                "",
                "",
                "",
                "",
            ],
            ["3", "专业技术人员", "", "高级领导", "2022.03.29", ""],
            ["4", "住宿和餐饮业", "", "高级领导", "2021.09.27", ""],
            ["点", "商业、服务业人员 制造业 2013 2021.09.26", "", "", "", ""],
            ["编号", "", "", "数据发生机构名称", "", ""],
            ["1", "", "", "样例甲银行股份有限公司信用卡中心 A", "", ""],
            ["2", "福 样例乙消费金融有限公司 水", "", "", "", ""],
            ["3", "", "样例丙银行股份有限公司", "", "", ""],
            ["4", "", "", "", "样例丁银行股份有限公司", ""],
            ["5", "样例戊银行股份有限公司", "", "", "", ""],
        )
    )

    records = _extract_employment_records(result)

    assert [record["sequence"] for record in records] == [1, 2, 3, 4, 5]
    assert records[0]["occupation"] == "商业、服务业人员"
    assert "occupation" not in records[1]
    assert records[4]["occupation"] == "商业、服务业人员"
    assert "position" not in records[1]
    assert [record.get("data_provider") for record in records] == [
        None,
        None,
        "样例丙银行股份有限公司",
        "样例丁银行股份有限公司",
        "样例戊银行股份有限公司",
    ]

    issues = result._personal_detail_extraction_issues
    assert {
        issue.get("target_record_id")
        for issue in issues
        if issue["issue_code"] == "candidate_b_employment_provider_cell_unresolved"
    } >= {
        records[0]["employment_record_id"],
        records[1]["employment_record_id"],
    }
    assert not any(
        issue["issue_code"] == "candidate_b_employment_cluster_field_unresolved"
        and issue.get("field_name") == "occupation"
        and issue.get("target_record_id")
        in {
            records[0]["employment_record_id"],
            records[4]["employment_record_id"],
        }
        for issue in issues
    )
    for field_name in ("occupation", "position"):
        assert any(
            issue["issue_code"] == "candidate_b_employment_cluster_field_unresolved"
            and issue.get("target_record_id") == records[1]["employment_record_id"]
            and issue.get("field_name") == field_name
            for issue in issues
        )


def test_employment_provider_requires_complete_cell_without_deleting_edge_glyphs() -> None:
    assert _employment_provider_header(("编号", "", "数据发生机构名称", "")) == {
        "sequence": 0,
        "data_provider": 2,
    }
    assert _employment_provider_header(("编号", "", "福 数据发生机构名称", "")) is None
    assert _employment_provider_header(("编号", "", "污染噪数据发生机构名称", "")) is None
    assert _strict_employment_provider_span("福 样例消费金融有限公司 水") is None
    assert (
        _strict_employment_provider_span("样例银行股份有限公 司")
        == "样例银行股份有限公司"
    )
    assert (
        _strict_employment_provider_span("兴 业银行股份有限公司")
        == "兴业银行股份有限公司"
    )
    assert (
        _strict_employment_provider_span(
            "样例甲银行股份有限公司 样例乙银行股份有限公司"
        )
        is None
    )


def test_resolved_employment_field_prunes_only_stale_missing_issue() -> None:
    result = _result(
        _table(
            ["编号", "工作单位", "单位性质", "单位地址", "单位电话"],
            ["1", "样例科技有限公司", "私营企业", "福建省福州市样例路1号", "01012345678"],
            ["编号", "职业 行业", "", "进入本单位年份 职称 职务", "信息更新日期"],
            ["1", "乱码", "", "", "2025.01.02"],
            ["1", "专业技术人员 乱码", "", "", "2025.01.02"],
            ["编号", "", "数据发生机构名称", ""],
            ["1", "样例银行股份有限公司", "", ""],
        )
    )

    record = _extract_employment_records(result)[0]

    assert record["occupation"] == "专业技术人员"
    assert "occupation" not in set(record.get("_unresolved_fields") or ())
    occupation_issues = [
        issue
        for issue in result._personal_detail_extraction_issues
        if issue.get("target_record_id") == record["employment_record_id"]
        and issue.get("field_name") == "occupation"
    ]
    assert not any(
        issue["issue_code"] == "candidate_b_employment_cluster_field_unresolved"
        for issue in occupation_issues
    )
    assert any(
        issue["issue_code"] == "candidate_b_employment_cluster_residue_unresolved"
        for issue in occupation_issues
    )


@pytest.mark.parametrize("nonterminal", [False, True])
def test_exact_merged_ordinal_recovers_terminal_and_nonterminal_business_rows(
    nonterminal: bool,
) -> None:
    result = _result(_merged_ordinal_employment_table(nonterminal=nonterminal))

    records = _extract_employment_records(result)

    assert [record["sequence"] for record in records] == list(
        range(1, 7 if nonterminal else 6)
    )
    fourth = records[3]
    fifth = records[4]
    assert fourth["employer"] == "丁四科技有限公司"
    assert fourth["employer_address"] == "福建省福州市丁四路4号"
    assert fifth["employer"] == "戊五工程有限公司"
    assert fifth["employer_address"] == "福建省福州市戊五路501- 502室"
    assert fifth["employer_phone"] == "01055555555"
    assert fifth["occupation"] == "专业技术人员"
    assert fifth["position"] == "一般员工"
    assert fifth["entry_year"] == 2020
    assert fifth["information_updated_date"] == "2025-01-05"
    assert "employer_type" not in fifth
    assert any(
        issue["issue_code"] == "candidate_b_employment_canonical_cell_unresolved"
        and issue.get("target_record_id") == fifth["employment_record_id"]
        and issue.get("field_name") == "employer_type"
        for issue in result._personal_detail_extraction_issues
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_sequence_span",
        "cross_row_value_span",
        "malformed_phone",
        "latin_prefixed_address",
        "nonexact_prior_anchor",
        "sparse_detail_population",
    ],
)
def test_exact_merged_ordinal_repair_fails_closed_on_near_misses(
    mutation: str,
) -> None:
    table = _merged_ordinal_employment_table()
    rows = table.metadata["raw_rows"]
    geometry = table.metadata["geometry"]
    terminal_employer = rows[5][1]
    if mutation == "duplicate_sequence_span":
        geometry["cell_spans"].append(deepcopy(geometry["cell_spans"][0]))
    elif mutation == "cross_row_value_span":
        value_span = geometry["cell_spans"][1]
        value_span["row"] = 4
        geometry["cell_bboxes"][4][2] = geometry["cell_bboxes"][5][2]
        geometry["cell_bboxes"][5][2] = None
        geometry["cell_geometry_status"][5][2] = "derived"
        geometry["cell_evidence_ids"][5][2] = []
        geometry["cell_token_ids"][5][2] = []
    elif mutation == "malformed_phone":
        rows[5][2] = "福建省福州市戊五路501- 05555555 502室"
    elif mutation == "latin_prefixed_address":
        rows[5][2] = "C 福建省福州市戊五路501- 01055555555 502室"
    elif mutation == "nonexact_prior_anchor":
        geometry["cell_geometry_status"][3][0] = "projected"
    elif mutation == "sparse_detail_population":
        rows[8][0] = "7"

    result = _result(table)
    records = _extract_employment_records(result)

    assert all(record.get("employer") != terminal_employer for record in records)
    assert any(
        issue["issue_code"] == "candidate_b_sequence_cell_unresolved"
        and terminal_employer in issue["observed_value"]["physical_cells"]
        for issue in result._personal_detail_extraction_issues
    )


def test_exact_merged_type_address_phone_cell_withholds_type_contaminated_address() -> None:
    table = _merged_ordinal_employment_table()
    table.metadata["raw_rows"][5][2] = (
        "私营企业 福建省福州市戊五路501- 01055555555 502室"
    )

    result = _result(table)
    records = _extract_employment_records(result)

    assert all(record.get("employer") != "戊五工程有限公司" for record in records)
    assert all(
        "私营企业" not in str(record.get("employer_address") or "")
        for record in records
    )
    assert any(
        issue["issue_code"] == "candidate_b_sequence_cell_unresolved"
        and "私营企业" in "".join(issue["observed_value"]["physical_cells"])
        for issue in result._personal_detail_extraction_issues
    )


def test_invalid_phone_truncated_employer_and_latin_address_are_field_local() -> None:
    result = _result(
        _table(
            ["编号", "工作单位", "单位性质", "单位地址", "单位电话"],
            ["1", "甲一科技有限公司", "私营企业", "福建省福州市甲一路1号", "06927225"],
            ["2", "乙二科技有限公", "私营企业", "福建省福州市乙二路2号", "01022222222"],
            ["3", "丙三科技有限公司", "私营企业", "C 福建省福州市丙三路3号", "01033333333"],
        )
    )

    records = _extract_employment_records(result)

    assert "employer_phone" not in records[0]
    assert "employer" not in records[1]
    assert "employer_address" not in records[2]
    invalid = {
        (issue.get("target_record_id"), issue.get("field_name"))
        for issue in result._personal_detail_extraction_issues
        if issue["issue_code"] == "candidate_b_exact_slot_value_invalid"
    }
    assert (records[0]["employment_record_id"], "employer_phone") in invalid
    assert (records[1]["employment_record_id"], "employer") in invalid
    assert (records[2]["employment_record_id"], "employer_address") in invalid
