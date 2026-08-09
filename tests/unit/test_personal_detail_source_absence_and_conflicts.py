from __future__ import annotations

from types import SimpleNamespace

import pytest

from docmirror.plugins.credit_report.personal_detail_scanned import native_extraction
from docmirror.plugins.credit_report.personal_detail_scanned.candidate_b import (
    _withhold_independent_plane_conflicts,
)
from docmirror.plugins.credit_report.personal_detail_scanned.field_contracts import (
    is_explicit_source_absence,
)
from docmirror.plugins.credit_report.personal_detail_scanned.profile_extraction import (
    extract_candidate_b_profile,
)
from docmirror.plugins.credit_report.personal_detail_scanned.source_projection import (
    prepare_personal_detail_source_collections,
)


def _table(table_id: str, rows: list[list[str]]) -> SimpleNamespace:
    return SimpleNamespace(
        table_id=table_id,
        bbox=[10.0, 10.0, 590.0, 700.0],
        metadata={"raw_rows": rows, "canonical_template_id": "report_header_and_identity"},
        rows=[],
    )


def _context(*tables: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(
        pages=[
            SimpleNamespace(
                page_number=1,
                source_page_number=1,
                canonical_template_id="report_header_and_identity",
                tables=list(tables),
            )
        ],
        tables_continue=lambda _left, _right: False,
        _personal_detail_extraction_issues=[],
    )


@pytest.mark.parametrize("value", ("-", "--", "－", "——", "‐", "‑", "‒", "–", "—", "―"))
def test_source_absence_contract_accepts_only_nonblank_dash_glyphs(value: str) -> None:
    assert is_explicit_source_absence(value)
    assert is_explicit_source_absence(f"  {value}  ")


@pytest.mark.parametrize("value", ("", "   ", "未知", "-- 缺失", "A-B"))
def test_source_absence_contract_does_not_hide_blank_or_business_text(value: str) -> None:
    assert not is_explicit_source_absence(value)


def test_profile_degree_dash_variant_is_source_absent_without_issue() -> None:
    context = _context(_table("profile", [["学位"], ["—"]]))

    profile = extract_candidate_b_profile(context)

    assert profile["degree"]["normalized_value"] is None
    assert profile["degree"]["observation_status"] == "source_absent"
    assert context._personal_detail_extraction_issues == []


def test_profile_source_absence_stays_null_and_silent_through_public_projection() -> None:
    context = _context(_table("profile", [["学位"], ["--"]]))
    profile = extract_candidate_b_profile(context)

    projected = prepare_personal_detail_source_collections(
        {"facts": {"subject_profile": profile}, "datasets": {}}
    )

    assert "degree" not in projected["datasets"]["personal_profile"][0]
    assert projected["datasets"]["personal_detail_field_observations"] == []
    assert context._personal_detail_extraction_issues == []


def test_profile_degree_glyph_guess_is_withheld_with_local_raw_issue() -> None:
    context = _context(_table("profile", [["学位"], ["光"]]))

    profile = extract_candidate_b_profile(context)

    assert profile["degree"]["normalized_value"] is None
    assert profile["degree"]["observation_status"] == "unreadable"
    assert profile["degree"]["raw"] == ["光"]
    assert any(
        issue.get("issue_code") == "candidate_b_profile_contract_unresolved"
        and issue.get("field_name") == "degree"
        and issue.get("observed_value") == ["光"]
        for issue in context._personal_detail_extraction_issues
    )


def test_profile_exact_degree_value_remains_silent() -> None:
    context = _context(_table("profile", [["学位"], ["无"]]))

    profile = extract_candidate_b_profile(context)

    assert profile["degree"]["normalized_value"] == "无"
    assert profile["degree"]["observation_status"] == "observed"
    assert context._personal_detail_extraction_issues == []


def test_spouse_dash_variants_are_null_source_absence_without_false_issue() -> None:
    table = _table(
        "spouse",
        [
            ["姓名", "证件类型", "证件号码", "工作单位", "联系电话"],
            ["—", "－", "--", "–", "―"],
            ["数据发生机构名称", "", "", "", ""],
            ["——", "", "", "", ""],
        ],
    )
    context = _context(table)

    spouse_records = native_extraction._extract_profile_detail_records(context)["spouse_records"]

    # The entire canonical spouse row explicitly says no source value.  The
    # dataset-level absence is therefore silent; no empty business row is made.
    assert spouse_records == []
    assert context._personal_detail_extraction_issues == []


def test_residence_phone_dash_variant_is_source_absent_not_invalid() -> None:
    table = _table(
        "residence",
        [
            ["编号", "居住地址", "住宅电话", "居住状况", "信息更新日期"],
            ["1", "福建省福州市示例路1号", "—", "自置", "2025.01.02"],
            ["编号", "数据发生机构名称", "", "", ""],
            ["1", "示例银行股份有限公司", "", "", ""],
        ],
    )
    context = _context(table)

    record = native_extraction._extract_residence_records(context)[0]

    assert "residential_phone" not in record
    assert "residential_phone" in record["_source_absent_fields"]
    assert not any(
        issue.get("field_name") == "residential_phone"
        for issue in context._personal_detail_extraction_issues
    )


def test_agreement_optional_dash_variants_are_null_source_absence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from docmirror.plugins.credit_report.personal_detail_scanned.native_parser import (
        PBOCPersonalDetailNativeParser,
    )

    candidate = SimpleNamespace(
        fields={
            "授信协议标识": "T10151210H0001ABC12345",
            "管理机构": "—",
            "授信额度用途": "－",
            "生效日期": "–",
            "到期日期": "―",
            "授信额度": "——",
            "授信限额": "--",
            "已用额度": "‐",
            "授信限额编号": "‑",
            "币种": "‒",
        },
        source_refs=[],
        source_refs_by_field={},
        binding_quality_by_field={},
        unresolved_labels=frozenset(),
        observed_labels=frozenset(),
        confidence=0.9,
    )
    monkeypatch.setattr(
        PBOCPersonalDetailNativeParser,
        "records",
        lambda _self, dataset_name: [candidate] if dataset_name == "credit_lines" else [],
    )
    context = SimpleNamespace(_personal_detail_extraction_issues=[])

    rows = native_extraction.reconcile_candidate_b_credit_lines(
        context,
        native_extraction._extract_credit_lines(context),
    )

    absent = {
        "institution",
        "facility_type",
        "effective_date",
        "due_date",
        "total_limit",
        "credit_limit",
        "used_limit",
        "limit_identifier",
        "currency",
    }
    assert absent <= set(rows[0]["_source_absent_fields"])
    assert all(rows[0].get(field_name) is None for field_name in absent)
    assert not any(
        issue.get("field_name") in absent
        for issue in context._personal_detail_extraction_issues
    )


def _employment_context(*, address: str, position: str = "一般员工") -> SimpleNamespace:
    return _context(
        _table(
            "employment",
            [
                ["编号", "工作单位", "单位性质", "单位地址", "单位电话", "", ""],
                ["1", "福州众恒房产代理有限公司", "—", address, "－", "", ""],
                ["编号", "职业", "行业", "职务", "职称", "进入本单位年份", "信息更新日期"],
                ["1", "专业技术人员", "制造业", position, "–", "―", "2025.01.02"],
                ["编号", "数据发生机构名称", "", "", "", "", ""],
                ["1", "示例银行股份有限公司", "", "", "", "", ""],
            ],
        )
    )


def test_employment_dash_variants_are_source_absent_and_clean() -> None:
    context = _employment_context(address="福建省福州市仓山区样例路1号")

    record = native_extraction._extract_employment_records(context)[0]

    assert {"employer_type", "employer_phone", "professional_title", "entry_year"} <= set(
        record["_source_absent_fields"]
    )
    assert not any(
        issue.get("field_name")
        in {"employer_type", "employer_phone", "professional_title", "entry_year"}
        for issue in context._personal_detail_extraction_issues
    )


def test_blank_employment_slot_is_reported_field_locally_not_source_absent() -> None:
    context = _employment_context(
        address="福建省福州市仓山区样例路1号",
        position="",
    )

    record = native_extraction._extract_employment_records(context)[0]

    assert "position" not in record
    assert "position" not in set(record.get("_source_absent_fields") or ())
    assert any(
        issue.get("issue_code") == "candidate_b_employment_canonical_cell_unresolved"
        and issue.get("field_name") == "position"
        and issue.get("target_record_id") == record["employment_record_id"]
        for issue in context._personal_detail_extraction_issues
    )


def test_employment_address_ending_in_long_employer_prefix_is_withheld() -> None:
    address = "福建省福州市仓山区金山碧水中区朱菊苑4号楼04店面福州众恒房产代"
    context = _employment_context(address=address)

    record = native_extraction._extract_employment_records(context)[0]

    assert "employer_address" not in record
    assert any(
        issue.get("field_name") == "employer_address"
        and issue.get("observed_value") == [address]
        for issue in context._personal_detail_extraction_issues
    )


@pytest.mark.parametrize(
    ("dataset", "identity_field", "record_id", "field_name", "native_value", "corrected_value"),
    (
        ("residence_records", "residence_record_id", "residence:5", "address", "卢滨路19号", "泸滨路19号"),
        (
            "credit_accounts",
            "account_id",
            "credit_account:22",
            "management_institution",
            "重庆市蚂蚁商诚小额贷款有限公司",
            "重庆市蚂蚊商诚小额贷款有限公司",
        ),
        (
            "credit_lines",
            "credit_line_id",
            "credit_line:11",
            "institution",
            "示例银行股份有限公司",
            "示例银衍股份有限公司",
        ),
    ),
)
def test_independent_plane_single_glyph_conflicts_are_withheld_and_reported(
    dataset: str,
    identity_field: str,
    record_id: str,
    field_name: str,
    native_value: str,
    corrected_value: str,
) -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    native = {dataset: [{identity_field: record_id, field_name: native_value}]}
    corrected_row = {identity_field: record_id, field_name: corrected_value}
    corrected = {dataset: [corrected_row]}

    _withhold_independent_plane_conflicts(context, native, corrected)

    assert field_name not in corrected_row
    issue = context._personal_detail_extraction_issues[0]
    assert issue["issue_code"] == "candidate_b_independent_plane_field_conflict"
    assert issue["field_name"] == field_name
    assert issue["target_record_id"] == record_id


def test_cross_plane_clean_value_preserves_business_data_silently() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    native = {
        "residence_records": [
            {"residence_record_id": "residence:1", "address": "福建省 福州市示例路1号"}
        ]
    }
    corrected_row = {
        "residence_record_id": "residence:1",
        "address": "福建省福州市示例路1号",
    }

    _withhold_independent_plane_conflicts(
        context,
        native,
        {"residence_records": [corrected_row]},
    )

    assert corrected_row["address"] == "福建省福州市示例路1号"
    assert context._personal_detail_extraction_issues == []


def test_cross_plane_institution_alias_disagreement_is_withheld_without_guessing() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    native = {
        "credit_accounts": [
            {
                "account_id": "credit_account:1",
                "management_institution": "重庆蚂蚊消费金融有限公司",
            }
        ]
    }
    corrected_row = {
        "account_id": "credit_account:1",
        "management_institution": "重庆蚂蚁消费金融有限公司",
    }

    _withhold_independent_plane_conflicts(
        context,
        native,
        {"credit_accounts": [corrected_row]},
    )

    assert "management_institution" not in corrected_row
    assert corrected_row["canonical_raw"]["management_institution"] == [
        "重庆蚂蚊消费金融有限公司",
        "重庆蚂蚁消费金融有限公司",
    ]
    assert context._personal_detail_extraction_issues[0]["issue_code"] == (
        "candidate_b_independent_plane_field_conflict"
    )
