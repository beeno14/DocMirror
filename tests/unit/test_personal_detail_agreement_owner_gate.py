from __future__ import annotations

from types import SimpleNamespace

from docmirror.plugins.credit_report.personal_detail_scanned.native_parser import (
    PBOCPersonalDetailNativeParser,
)


def _agreement_table(*, registered: bool) -> SimpleNamespace:
    rows = [
        ["管理机构", "授信协议标识", "生效日期", "到期日期", "授信额度用途"],
        [
            "中国工商银行股份有限公司",
            "T10151024H0002HT11202211122200008302277",
            "2025.01.01",
            "2030.01.01",
            "循环贷款额度",
        ],
    ]
    return SimpleNamespace(
        table_id="agreement-owner-contract",
        bbox=[20.0, 100.0, 580.0, 180.0],
        headers=[],
        rows=[],
        metadata={
            "raw_rows": rows,
            **({"canonical_template_id": "credit_agreement"} if registered else {}),
        },
    )


def _context(
    *,
    registered: bool,
    competing_heading: bool = False,
    template_id: str | None = None,
) -> SimpleNamespace:
    table = _agreement_table(registered=registered)
    if template_id is not None:
        table.metadata["canonical_template_id"] = template_id
    page = SimpleNamespace(
        page_number=8,
        source_page_number=4,
        canonical_template_id="credit_agreement" if registered else "",
        tables=[table],
        texts=[
            SimpleNamespace(
                content="授信协议6",
                bbox=[20.0, 78.0, 120.0, 98.0],
            )
        ],
    )
    if competing_heading:
        page.texts.append(
            SimpleNamespace(
                content=page.texts[0].content,
                bbox=[150.0, 78.0, 250.0, 98.0],
            )
        )
    return SimpleNamespace(
        pages=[page],
        reading_order_by_logical={8: 8},
        tables_continue=lambda _left, _right: None,
        _personal_detail_extraction_issues=[],
    )


def test_exact_agreement_labels_without_registered_owner_are_withheld() -> None:
    records = PBOCPersonalDetailNativeParser(
        _context(registered=False)
    ).records("credit_lines")

    assert records == []


def test_exact_agreement_labels_with_arbitrary_template_are_withheld() -> None:
    records = PBOCPersonalDetailNativeParser(
        _context(registered=False, template_id="arbitrary_unregistered_template")
    ).records("credit_lines")

    assert records == []


def test_registered_numbered_agreement_card_owner_is_retained() -> None:
    records = PBOCPersonalDetailNativeParser(
        _context(registered=True)
    ).records("credit_lines")

    assert len(records) == 1
    assert records[0].fields["授信协议标识"] == (
        "T10151024H0002HT11202211122200008302277"
    )
    assert records[0].fields["__printed_sequence"] == "6"


def test_registered_agreement_with_competing_local_owners_is_withheld() -> None:
    records = PBOCPersonalDetailNativeParser(
        _context(registered=True, competing_heading=True)
    ).records("credit_lines")

    assert records == []


def test_registered_agreement_with_heading_substring_is_withheld() -> None:
    context = _context(registered=True)
    context.pages[0].texts[0].content = "璐︽埛6锛堟巿淇″崗璁?锛?"

    records = PBOCPersonalDetailNativeParser(context).records("credit_lines")

    assert records == []


def test_registered_agreement_with_nonfinite_heading_geometry_is_withheld() -> None:
    context = _context(registered=True)
    context.pages[0].texts[0].bbox = [20.0, 78.0, 120.0, float("nan")]

    records = PBOCPersonalDetailNativeParser(context).records("credit_lines")

    assert records == []
