from __future__ import annotations

from types import SimpleNamespace

import pytest

from docmirror.plugins.credit_report.personal_detail_scanned.native_extraction import (
    _extract_profile_detail_records,
)

_PROFILE_ROLE = "report_header_and_identity"
_PUBLIC_ROLE = "public_information"


def _table(
    table_id: str,
    rows: list[list[str]],
    *,
    role: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        table_id=table_id,
        metadata={"raw_rows": rows, "canonical_template_id": role},
        headers=[],
        rows=[],
    )


def _context(
    *tables: SimpleNamespace,
    page_role: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        pages=[
            SimpleNamespace(
                page_number=4,
                source_page_number=2,
                canonical_template_id=page_role,
                tables=list(tables),
            )
        ],
        tables_continue=lambda _left, _right: False,
    )


def _issues(context: SimpleNamespace) -> list[dict]:
    return list(getattr(context, "_personal_detail_extraction_issues", []) or [])


def _mobile_rows() -> list[list[str]]:
    # Reordered columns and three business rows prove that neither source
    # column order nor the canonical sample's population size is authority.
    return [
        ["数据发生机构名称", "编号", "信息更新日期", "手机号码"],
        ["中国工商银行股份有限公司", "7", "2025.01.03", "13800138007"],
        ["中国建设银行股份有限公司", "2", "2024.12.01", "13900139002"],
        ["中国农业银行股份有限公司", "11", "2023.06.30", "13700137011"],
    ]


def _spouse_rows() -> list[list[str]]:
    return [
        ["联系电话", "工作单位", "姓名", "证件号码", "证件类型"],
        ["13800138000", "示例科技有限公司", "张甲", "110101199001010011", "身份证"],
    ]


def test_owned_reordered_variable_profile_tables_are_published() -> None:
    context = _context(
        _table("mobile", _mobile_rows(), role=_PROFILE_ROLE),
        _table("spouse", _spouse_rows(), role=_PROFILE_ROLE),
        page_role=_PROFILE_ROLE,
    )

    details = _extract_profile_detail_records(context)

    assert [row["sequence"] for row in details["mobile_phone_records"]] == [2, 7, 11]
    assert [row["mobile_phone"] for row in details["mobile_phone_records"]] == [
        "13900139002",
        "13800138007",
        "13700137011",
    ]
    assert details["spouse_records"][0]["name"] == "张甲"
    assert details["spouse_records"][0]["phone"] == "13800138000"


@pytest.mark.parametrize(
    "raw_provider",
    (
        "导中国工商银行股份有限公司",
        "S 中国工商银行股份有限公司",
        "中国工商银行股份有限公司 Ss",
    ),
)
def test_mobile_provider_never_deletes_unowned_glyphs(raw_provider: str) -> None:
    rows = [
        _mobile_rows()[0],
        [raw_provider, "1", "2025.01.03", "13800138007"],
    ]
    context = _context(
        _table("mobile-provider-glyphs", rows, role=_PROFILE_ROLE),
        page_role=_PROFILE_ROLE,
    )

    record = _extract_profile_detail_records(context)["mobile_phone_records"][0]

    assert "data_provider" not in record
    assert any(
        issue.get("target_dataset") == "mobile_phone_records"
        and issue.get("field_name") == "data_provider"
        for issue in _issues(context)
    )


@pytest.mark.parametrize(
    "raw_phone",
    ("$13800138007", "13800138007?", "138,0013,8007", "手机13800138007"),
)
def test_mobile_phone_rejects_arbitrary_nonformat_glyphs(raw_phone: str) -> None:
    rows = [
        _mobile_rows()[0],
        ["中国工商银行股份有限公司", "1", "2025.01.03", raw_phone],
    ]
    context = _context(
        _table("mobile-phone-glyphs", rows, role=_PROFILE_ROLE),
        page_role=_PROFILE_ROLE,
    )

    record = _extract_profile_detail_records(context)["mobile_phone_records"][0]

    assert "mobile_phone" not in record
    assert any(
        issue.get("target_dataset") == "mobile_phone_records"
        and issue.get("field_name") == "mobile_phone"
        for issue in _issues(context)
    )


def test_mobile_phone_accepts_only_registered_presentation_separators() -> None:
    rows = [
        _mobile_rows()[0],
        ["中国工商银行股份有限公司", "1", "2025.01.03", "+86 138-0013-8007"],
    ]
    context = _context(
        _table("mobile-phone-format", rows, role=_PROFILE_ROLE),
        page_role=_PROFILE_ROLE,
    )

    record = _extract_profile_detail_records(context)["mobile_phone_records"][0]
    assert record["mobile_phone"] == "13800138007"
    assert record["canonical_raw"]["mobile_phone"] == "+86 138-0013-8007"


@pytest.mark.parametrize("raw_employer", ('".', "导示例科技有限公司"))
def test_spouse_employer_requires_a_complete_whole_field_value(raw_employer: str) -> None:
    rows = [
        _spouse_rows()[0],
        ["13800138000", raw_employer, "张甲", "110101199001010011", "身份证"],
    ]
    context = _context(
        _table("spouse-employer", rows, role=_PROFILE_ROLE),
        page_role=_PROFILE_ROLE,
    )

    record = _extract_profile_detail_records(context)["spouse_records"][0]

    assert record["name"] == "张甲"
    assert "employer" not in record
    assert any(
        issue.get("target_dataset") == "spouse_records"
        and issue.get("field_name") == "employer"
        for issue in _issues(context)
    )


def test_public_information_owner_cannot_publish_profile_tables() -> None:
    context = _context(
        _table("public-mobile", _mobile_rows(), role=_PUBLIC_ROLE),
        _table("public-spouse", _spouse_rows(), role=_PUBLIC_ROLE),
        page_role=_PUBLIC_ROLE,
    )

    assert _extract_profile_detail_records(context) == {
        "mobile_phone_records": [],
        "spouse_records": [],
    }


@pytest.mark.parametrize(
    ("page_role", "table_role"),
    (
        (_PROFILE_ROLE, _PUBLIC_ROLE),
        (_PUBLIC_ROLE, _PROFILE_ROLE),
    ),
)
def test_page_table_owner_mismatch_cannot_publish_profile_tables(
    page_role: str,
    table_role: str,
) -> None:
    context = _context(
        _table("mismatched-mobile", _mobile_rows(), role=table_role),
        _table("mismatched-spouse", _spouse_rows(), role=table_role),
        page_role=page_role,
    )

    assert _extract_profile_detail_records(context) == {
        "mobile_phone_records": [],
        "spouse_records": [],
    }
