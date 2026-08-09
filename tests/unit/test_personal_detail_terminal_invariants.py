from types import SimpleNamespace

from docmirror.plugins.credit_report.personal_detail_scanned.native_extraction import (
    _extract_employment_records,
    _extract_inquiries,
    _extract_profile_detail_records,
    _extract_residence_records,
)


def _context(rows: list[list[str]], table_id: str = "terminal") -> SimpleNamespace:
    table = SimpleNamespace(
        table_id=table_id,
        metadata={"raw_rows": rows},
        headers=[],
        rows=[],
        bbox=[10, 10, 590, 200],
        confidence=1.0,
    )
    return SimpleNamespace(
        pages=[SimpleNamespace(page_number=3, source_page_number=2, tables=[table])],
        tables_continue=lambda _left, _right: None,
        corrected_evidence_pages=lambda: [],
        _personal_detail_extraction_issues=[],
    )


def test_observed_detail_headers_cannot_terminate_silently() -> None:
    cases = (
        (
            _extract_profile_detail_records,
            [["编号", "手机号码", "信息更新日期", "数据发生机构名称"]],
            "mobile_phone_records",
        ),
        (
            _extract_profile_detail_records,
            [["姓名", "证件类型", "证件号码", "工作单位", "联系电话"]],
            "spouse_records",
        ),
        (
            _extract_residence_records,
            [["编号", "居住地址", "住宅电话", "居住状况", "信息更新日期"]],
            "residence_records",
        ),
        (
            _extract_employment_records,
            [["编号", "工作单位", "单位性质", "单位地址", "单位电话"]],
            "employment_records",
        ),
        (
            _extract_inquiries,
            [["编号", "查询日期", "查询机构", "查询原因"]],
            "inquiry_records",
        ),
    )
    for extractor, rows, dataset in cases:
        context = _context(rows, dataset)
        extractor(context)
        assert any(
            issue["issue_code"] == "candidate_b_observed_header_without_terminal_row"
            and issue["target_dataset"] == dataset
            for issue in context._personal_detail_extraction_issues
        )


def test_explicit_dash_absence_satisfies_terminal_header_invariant() -> None:
    context = _context(
        [
            ["姓名", "证件类型", "证件号码", "工作单位", "联系电话"],
            ["--", "--", "--", "--", "--"],
        ],
        "spouse-absent",
    )

    result = _extract_profile_detail_records(context)

    assert result["spouse_records"] == []
    assert not any(
        issue["issue_code"] == "candidate_b_observed_header_without_terminal_row"
        for issue in context._personal_detail_extraction_issues
    )

