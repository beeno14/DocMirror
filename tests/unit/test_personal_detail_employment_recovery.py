# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

from docmirror.plugins.credit_report.personal_detail_scanned.native_extraction import (
    _employment_provider_header,
    _extract_employment_records,
    _strict_employment_provider_span,
)


def _table(*rows: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        table_id="employment",
        metadata={"raw_rows": list(rows)},
        headers=[],
        rows=[],
        bbox=[20, 20, 580, 700],
    )


def _result(table: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(
        pages=[SimpleNamespace(page_number=2, source_page_number=1, tables=[table])],
        tables_continue=lambda _left, _right: False,
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
        )
    )

    records = _extract_employment_records(result)

    assert [record["sequence"] for record in records] == [1, 2, 3]
    first, second, third = records
    assert first["employer"] == "显微镜(厦门)贸易有限公司"
    assert first["employer_type"] == "个体、私营企业"
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
    assert third["employer_type"] == "其他（包括三资企业、民营企业、民间团体等）"
    assert third["occupation"] == "不便分类的其他从业人员"
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
    assert (third["employment_record_id"], "employer_address") in invalid_fields

    unresolved = {
        (issue.get("target_record_id"), issue.get("field_name"))
        for issue in result._personal_detail_extraction_issues
        if issue["issue_code"] == "candidate_b_employment_cluster_field_unresolved"
    }
    assert (first["employment_record_id"], "industry") in unresolved
    assert (second["employment_record_id"], "entry_year") in unresolved
    assert (third["employment_record_id"], "position") in unresolved
    assert not any(
        issue["issue_code"] == "candidate_b_employment_component_missing"
        for issue in result._personal_detail_extraction_issues
    )
    assert not any(
        issue["issue_code"] == "candidate_b_employment_cluster_residue_unresolved"
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


def test_employment_sequence_repair_requires_repeated_identical_digits() -> None:
    result = _result(
        _table(
            ["编母", "工作单位", "单位性质", "单位地址", "单位电话"],
            ["子 2 3", "不可绑定公司", "个体、私营企业", "样例路23号", "010-12345678"],
        )
    )

    assert _extract_employment_records(result) == []
    assert any(
        issue["issue_code"] == "candidate_b_sequence_cell_unresolved"
        for issue in result._personal_detail_extraction_issues
    )


def test_employment_header_overflow_is_never_carried_to_a_later_record() -> None:
    result = _result(
        _table(
            ["编母", "噪 工作单位", "单位性质", "单位地址", "单位电话"],
            ["1", "第一家公司", "私营企业", "样例路1号", "010-12345678"],
            ["2", "", "私营企业", "样例路2号", "010-87654321"],
        )
    )

    records = _extract_employment_records(result)

    assert [record["sequence"] for record in records] == [1, 2]
    assert records[0]["employer"] == "第一家公司"
    assert "employer" not in records[1]
    assert any(
        issue["issue_code"] == "candidate_b_employment_recovered_header_cell_unresolved"
        and issue["target_record_id"] == records[1]["employment_record_id"]
        and issue["field_name"] == "employer"
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


def test_lin_shaped_employment_rows_recover_finite_roles_and_ordinal_providers() -> None:
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
            ["编号", "", "", "福 数据发生机构名称", "", ""],
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
    assert records[1]["occupation"] == "不便分类的其他从业人员"
    assert records[4]["occupation"] == "商业、服务业人员"
    assert "position" not in records[1]
    assert [record["data_provider"] for record in records] == [
        "样例甲银行股份有限公司信用卡中心",
        "样例乙消费金融有限公司",
        "样例丙银行股份有限公司",
        "样例丁银行股份有限公司",
        "样例戊银行股份有限公司",
    ]
    assert (
        records[0]["canonical_raw"]["data_provider"]
        == "样例甲银行股份有限公司信用卡中心 A"
    )
    assert (
        records[1]["canonical_raw"]["data_provider"]
        == "福 样例乙消费金融有限公司 水"
    )

    issues = result._personal_detail_extraction_issues
    assert not any(
        issue["issue_code"]
        in {
            "candidate_b_employment_provider_cell_unresolved",
            "candidate_b_employment_component_missing",
        }
        for issue in issues
    )
    assert not any(
        issue["issue_code"] == "candidate_b_employment_cluster_field_unresolved"
        and issue.get("field_name") == "occupation"
        and issue.get("target_record_id")
        in {
            records[0]["employment_record_id"],
            records[1]["employment_record_id"],
            records[4]["employment_record_id"],
        }
        for issue in issues
    )
    assert any(
        issue["issue_code"] == "candidate_b_employment_cluster_field_unresolved"
        and issue.get("target_record_id") == records[1]["employment_record_id"]
        and issue.get("field_name") == "position"
        for issue in issues
    )


def test_employment_provider_correction_is_bounded_and_requires_one_span() -> None:
    assert _employment_provider_header(("编号", "", "福 数据发生机构名称", "")) == {
        "sequence": 0,
        "data_provider": 2,
    }
    assert _employment_provider_header(("编号", "", "污染噪数据发生机构名称", "")) is None
    assert (
        _strict_employment_provider_span("福 样例消费金融有限公司 水")
        == "样例消费金融有限公司"
    )
    assert (
        _strict_employment_provider_span("样例银行股份有限公 司")
        == "样例银行股份有限公司"
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
