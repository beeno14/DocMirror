from __future__ import annotations

from types import SimpleNamespace

import pytest

from docmirror.plugins.credit_report.personal_detail_scanned.canonical_layout import (
    _classify,
)
from docmirror.plugins.credit_report.personal_detail_scanned.native_parser import (
    PBOCPersonalDetailNativeParser,
)


def _table(rows: list[list[str]], *, top: float = 100.0) -> SimpleNamespace:
    return SimpleNamespace(
        table_id="wang-agreement",
        bbox=[20.0, top, 580.0, top + 120.0],
        headers=[],
        rows=[],
        metadata={
            "raw_rows": rows,
            "canonical_template_id": "annotations_and_inquiries",
        },
    )


def _context(
    table: SimpleNamespace,
    *,
    heading: str,
    heading_bbox: list[float],
    template_id: str = "annotations_and_inquiries",
) -> SimpleNamespace:
    table.metadata["canonical_template_id"] = template_id
    page = SimpleNamespace(
        page_number=33,
        source_page_number=17,
        canonical_template_id=template_id,
        tables=[table],
        texts=[SimpleNamespace(content=heading, bbox=heading_bbox)],
    )
    return SimpleNamespace(
        pages=[page],
        reading_order_by_logical={33: 33},
        tables_continue=lambda _left, _right: None,
        _personal_detail_extraction_issues=[],
    )


@pytest.mark.parametrize(
    "heading, primary, purpose",
    [
        ("投值协议3", "投值协议标识", "投值额度用途"),
        ("授伯协议11", "授伯协议标识", "授伯额度用途"),
        ("投值协这2", "投信协议标识", "投信额度用途"),
    ],
)
def test_wang_numbered_agreement_schema_registers_without_fuzzy_matching(
    heading: str,
    primary: str,
    purpose: str,
) -> None:
    classified = _classify(
        f"{heading} 管理机构 {primary} 生效日期 到期日期 {purpose}"
    )

    assert classified is not None
    assert classified[0] == "credit_agreement"
    assert classified[2] == ("numbered_agreement_card_schema",)


def test_explicit_summary_role_wins_over_agreement_mentions() -> None:
    classified = _classify(
        "信息概要 授信协议1 管理机构 授信协议标识 生效日期 到期日期 授信额度用途"
    )

    assert classified is not None
    assert classified[0] == "information_summary"


def test_leading_boundary_fragment_and_next_numbered_card_register_as_agreement() -> None:
    classified = _classify(
        "投信额度 投伯限额 已用额度 币种 18,100 17,168 人民币元 "
        "投信协议8 管理机构 投值协议标识 生效日期 到期日期 投伯额度用途"
    )

    assert classified is not None
    assert classified[0] == "credit_agreement"


def test_late_mixed_page_agreement_section_is_not_promoted_wholesale() -> None:
    classified = _classify(
        "账户12 发卡机构 账户标识 账户状态 余额 相关还款责任信息 "
        "投伯协议1 管理机构 投值协议标识 生效日期 到期日期 投信额度用途"
    )

    assert classified is not None
    assert classified[0] != "credit_agreement"


def test_agreement_heading_lookalike_without_card_schema_fails_closed() -> None:
    assert _classify("投值协议9 投值协议标识 授信总额合计") is None


def test_registered_agreement_page_admits_only_geometry_bound_wang_card() -> None:
    table = _table(
        [
            ["营理机构", "投值协议标识", "生效日期", "到期日期", "投值额度用途"],
            ["机构甲", "AGREEMENT0019", "2020.01.02", "长期", "信用卡共享额度"],
            ["投值额度", "投值限额", "投值限额编号", "已用额度", "币种"],
            ["50,000", "--", "--", "1,000", "人民币元"],
        ]
    )
    context = _context(
        table,
        heading="投信协议19",
        heading_bbox=[20.0, 78.0, 120.0, 98.0],
        template_id="credit_agreement",
    )

    records = PBOCPersonalDetailNativeParser(context).records("credit_lines")

    assert len(records) == 1
    assert records[0].fields == {
        "管理机构": "机构甲",
        "授信协议标识": "AGREEMENT0019",
        "生效日期": "2020.01.02",
        "到期日期": "长期",
        "授信额度用途": "信用卡共享额度",
        "授信额度": "50,000",
        "授信限额": "--",
        "授信限额编号": "--",
        "已用额度": "1,000",
        "币种": "人民币元",
        "__printed_sequence": "19",
    }


@pytest.mark.parametrize(
    "heading, heading_bbox",
    [
        ("账户19（授信协议标识：AGREEMENT0019）", [20.0, 78.0, 300.0, 98.0]),
        ("投信协议19", [20.0, 20.0, 120.0, 40.0]),
        ("投信协议十九", [20.0, 78.0, 120.0, 98.0]),
    ],
)
def test_registered_agreement_page_lookalikes_never_authorize_a_record(
    heading: str,
    heading_bbox: list[float],
) -> None:
    table = _table(
        [
            ["管理机构", "授信协议标识", "生效日期", "到期日期", "授信额度用途"],
            ["机构甲", "AGREEMENT0019", "2020.01.02", "长期", "信用卡共享额度"],
        ],
        top=100.0,
    )
    context = _context(
        table,
        heading=heading,
        heading_bbox=heading_bbox,
        template_id="credit_agreement",
    )

    assert PBOCPersonalDetailNativeParser(context).records("credit_lines") == []


def test_merged_agreement_header_recovers_only_unique_typed_slots() -> None:
    table = _table(
        [
            ["管理机构", "生效日期 投信协议标识", "", "到期日期", "投值额度用途"],
            [
                "招商银行股份有限公司",
                "B11115840H000100000000000000000002",
                "2023.07.02",
                "长期",
                "信用卡共享额度",
            ],
            ["授伯额度", "投伯限额", "投伯限额编号", "已用额度", "币种"],
            ["22,000", "--", "--", "21,839", "人民币元"],
        ]
    )
    table.metadata["canonical_template_id"] = "credit_agreement"
    context = _context(
        table,
        heading="投伯协议16",
        heading_bbox=[20.0, 78.0, 120.0, 98.0],
        template_id="credit_agreement",
    )

    records = PBOCPersonalDetailNativeParser(context).records("credit_lines")

    assert len(records) == 1
    assert records[0].fields["授信协议标识"] == "B11115840H000100000000000000000002"
    assert records[0].fields["生效日期"] == "2023.07.02"
    assert records[0].fields["__printed_sequence"] == "16"


def test_merged_agreement_header_does_not_shift_identifier_into_empty_column() -> None:
    table = _table(
        [
            ["管理机构", "生效日期 投信协议标识", "", "到期日期", "投值额度用途"],
            [
                "招商银行股份有限公司",
                "B22222222",
                "C33333333",
                "长期",
                "信用卡共享额度",
            ],
        ]
    )
    table.metadata["canonical_template_id"] = "credit_agreement"
    context = _context(
        table,
        heading="投伯协议16",
        heading_bbox=[20.0, 78.0, 120.0, 98.0],
        template_id="credit_agreement",
    )

    records = PBOCPersonalDetailNativeParser(context).records("credit_lines")

    assert records == []
