from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from docmirror.plugins.credit_report.personal_detail_scanned import (
    native_extraction,
)
from docmirror.plugins.credit_report.personal_detail_scanned import (
    schema as personal_detail_schema,
)
from docmirror.plugins.credit_report.personal_detail_scanned.context import (
    _printed_reading_order_resolution,
    build_personal_detail_extraction_context,
)
from docmirror.plugins.credit_report.personal_detail_scanned.native_parser import (
    PBOCPersonalDetailNativeParser,
)


def _table(
    table_id: str,
    rows: list[list[str]],
    *,
    top: float = 100.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        table_id=table_id,
        metadata={"raw_rows": rows},
        headers=[],
        rows=[],
        bbox=[10.0, top, 590.0, top + 120.0],
        confidence=0.99,
    )


def _page(
    logical_page: int,
    tables: list[SimpleNamespace],
    *,
    template: str = "credit_account_detail",
) -> SimpleNamespace:
    return SimpleNamespace(
        page_number=logical_page,
        source_page_number=logical_page,
        canonical_template_id=template,
        tables=tables,
        texts=[],
        height=800.0,
    )


def _exact_merged_geometry_table(
    table_id: str,
    labels: tuple[str, ...],
    values: tuple[str, ...],
    *,
    top: float = 100.0,
    interleaved: bool = False,
) -> SimpleNamespace:
    assert len(labels) == len(values)
    column_count = len(labels) * 2 + 1 if interleaved else len(labels)
    bottom = top + 40.0
    header_bottom = top + 20.0
    column_bands = [
        {"index": index, "x0": float(index * 100), "x1": float((index + 1) * 100)}
        for index in range(column_count)
    ]
    value_row = ["" for _index in range(column_count)]
    for index, value in enumerate(values):
        value_row[1 + index * 2 if interleaved else index] = value
    rows = [
        [" ".join(labels), *("" for _index in range(column_count - 1))],
        value_row,
    ]
    geometry = {
        "cell_bboxes": [
            [
                [0.0, top, float(column_count * 100), header_bottom],
                *(None for _index in range(column_count - 1)),
            ],
            [
                [float(index * 100), header_bottom, float((index + 1) * 100), bottom]
                for index in range(column_count)
            ],
        ],
        "cell_geometry_status": [
            ["exact", *("derived" for _index in range(column_count - 1))],
            ["exact" for _index in range(column_count)],
        ],
        "cell_evidence_ids": [
            [[f"header:{table_id}"], *([] for _index in range(column_count - 1))],
            [
                [f"value:{table_id}:{index}"] if value_row[index] else []
                for index in range(column_count)
            ],
        ],
        "cell_spans": [
            {
                "row": 0,
                "col": 0,
                "row_span": 1,
                "col_span": column_count,
                "bbox": [0.0, top, float(column_count * 100), header_bottom],
            }
        ],
        "row_bands": [
            {"index": 0, "y0": top, "y1": header_bottom},
            {"index": 1, "y0": header_bottom, "y1": bottom},
        ],
        "col_bands": column_bands,
    }
    return SimpleNamespace(
        table_id=table_id,
        metadata={"raw_rows": rows, "geometry": geometry},
        headers=[],
        rows=[],
        bbox=[0.0, top, float(column_count * 100), bottom],
        confidence=0.99,
    )


_CARD_HEADER = [
    "发卡机构",
    "账户标识",
    "开立日期",
    "账户授信额度",
    "共享授信额度",
    "币种",
    "业务种类",
    "担保方式",
]


def _card_values(identifier: str, *, institution: str = "招商银行股份有限公司") -> list[str]:
    return [
        institution,
        identifier,
        "2020.01.02",
        "50000",
        "50000",
        "人民币",
        "贷记卡",
        "信用",
    ]


def test_document_local_inquiry_repair_is_shared_by_rows_and_source_coverage() -> None:
    table = _table(
        "inquiries",
        [
            ["编号", "查询日期", "查询机构", "查询原因"],
            ["88", "2022.05.31", "机构甲", "贷款审批"],
            ["789", "2022.05.22", "兴业银行股份有限公司", "贷后管理"],
            ["90", "2022.05.20", "机构丙", "信用卡审批"],
        ],
    )
    context = SimpleNamespace(
        pages=[_page(29, [table], template="annotations_and_inquiries")],
        corrected_evidence_pages=lambda: [],
    )

    records = native_extraction._extract_inquiries(context)
    coverage = native_extraction._inquiry_source_coverage(context)

    assert [record["sequence"] for record in records] == [88, 89, 90]
    assert coverage["sequence_endpoints"] == {"institution": 90}
    assert coverage["observed_sequences"] == {"institution": [88, 89, 90]}
    assert coverage["sequence_outliers"] == {"institution": [789]}
    assert any(
        issue.get("issue_code") == "candidate_b_inquiry_sequence_prefix_noise_corrected"
        and issue.get("observed_value", {}).get("raw_sequence") == 789
        and issue.get("candidate_value", {}).get("normalized_sequence") == 89
        for issue in context._personal_detail_extraction_issues
    )


def test_document_local_inquiry_repair_preserves_real_high_ordinals() -> None:
    assert native_extraction._document_local_inquiry_ordinals([100, 101, 102]) == [
        (100, None),
        (101, None),
        (102, None),
    ]
    assert native_extraction._document_local_inquiry_ordinals([788, 789, 790]) == [
        (788, None),
        (789, None),
        (790, None),
    ]
    assert native_extraction._document_local_inquiry_ordinals([89, 789]) == [
        (89, None),
        (789, None),
    ]


def test_document_local_inquiry_repair_resolves_independent_isolated_gaps() -> None:
    assert native_extraction._document_local_inquiry_ordinals(
        [1, None, 3, 4, None, 6]
    ) == [
        (1, None),
        (2, "missing"),
        (3, None),
        (4, None),
        (5, "missing"),
        (6, None),
    ]


@pytest.mark.parametrize(
    ("raw_sequences", "expected_repair"),
    [
        ([None, 2, 3], "leading_boundary_missing"),
        ([1, 2, None], "trailing_boundary_missing"),
    ],
)
def test_document_local_inquiry_repair_localizes_single_boundary_gap(
    raw_sequences: list[int | None],
    expected_repair: str,
) -> None:
    normalized = native_extraction._document_local_inquiry_ordinals(raw_sequences)

    assert [value for value, _repair in normalized if value is None] == [None]
    assert [repair for value, repair in normalized if value is None] == [
        expected_repair
    ]


def test_document_local_inquiry_repair_marks_adjacent_run_as_multiple() -> None:
    assert native_extraction._document_local_inquiry_ordinals(
        [1, None, None, 4]
    ) == [
        (1, None),
        (None, "multiple_missing"),
        (None, "multiple_missing"),
        (4, None),
    ]


def test_exact_inquiry_table_withholds_adjacent_missing_run() -> None:
    rows = [["\u7f16\u53f7", "\u67e5\u8be2\u65e5\u671f", "\u67e5\u8be2\u673a\u6784", "\u67e5\u8be2\u539f\u56e0"]]
    rows.extend(
        [
            "\u574f" if sequence in {2, 3} else str(sequence),
            f"2024.01.{sequence:02d}",
            "\u672c\u4eba",
            "\u672c\u4eba\u67e5\u8be2",
        ]
        for sequence in range(1, 7)
    )
    table = _table("two-missing-inquiries", rows)
    context = SimpleNamespace(
        pages=[_page(30, [table], template="annotations_and_inquiries")],
        corrected_evidence_pages=lambda: [],
    )

    records = native_extraction._extract_inquiries(context)
    coverage = native_extraction._inquiry_source_coverage(context)

    assert [record["sequence"] for record in records] == [1, 4, 5, 6]
    assert coverage["observed_sequences"] == {"personal": [1, 4, 5, 6]}
    issues = [
        issue
        for issue in context._personal_detail_extraction_issues
        if issue.get("issue_code")
        == "candidate_b_inquiry_multiple_missing_sequences_unresolved"
    ]
    assert len(issues) == 2
    assert {issue["source_refs"][0]["row"] for issue in issues} == {2, 3}
    assert all(issue.get("field_name") == "sequence" for issue in issues)
    unresolved_targets = {
        str(issue.get("target_record_id") or "") for issue in issues
    }
    assert len(unresolved_targets) == 2
    assert all(
        target.startswith("credit_inquiry_unresolved_sequence:")
        for target in unresolved_targets
    )
    emitted_inquiry_ids = {
        str(record.get("inquiry_id") or "") for record in records
    }
    assert unresolved_targets.isdisjoint(emitted_inquiry_ids)

    compact_issues, evidence_rows = personal_detail_schema._issue_evidence_rows(issues)
    reasons_by_issue: dict[str, set[str]] = {}
    for evidence in evidence_rows:
        if evidence.get("evidence_kind") == "reason" and evidence.get("string_value"):
            reasons_by_issue.setdefault(
                str(evidence["extraction_issue_id"]), set()
            ).add(str(evidence["string_value"]))
    non_emission_markers = (
        "withheld",
        "suppressed",
        "not_invented",
        "not_emitted",
        "unresolved",
        "record_not_silently_dropped",
        "silent_drop_prevented",
    )
    assert all(
        issue.get("target_dataset") == "inquiries"
        and str(issue.get("target_record_id") or "") not in emitted_inquiry_ids
        and any(
            marker in reason
            for reason in reasons_by_issue.get(
                str(issue["extraction_issue_id"]), set()
            )
            for marker in non_emission_markers
        )
        for issue in compact_issues
    )


def test_collapsed_four_column_personal_inquiry_header_recovers_one_bad_ordinal() -> None:
    rows = [["编号 查询日期", "", "查询机构", "查询原因"]]
    rows.extend(
        [
            "坏" if sequence == 9 else str(sequence),
            f"2024.01.{sequence:02d}",
            "本人",
            "本人查询",
        ]
        for sequence in range(1, 17)
    )
    table = _table("personal-inquiries", rows)
    context = SimpleNamespace(
        pages=[_page(30, [table], template="annotations_and_inquiries")],
        corrected_evidence_pages=lambda: [],
    )

    records = native_extraction._extract_inquiries(context)
    coverage = native_extraction._inquiry_source_coverage(context)

    assert [record["sequence"] for record in records] == list(range(1, 17))
    assert {record["inquiry_type"] for record in records} == {"personal"}
    assert coverage["sequence_endpoints"] == {"personal": 16}
    assert coverage["observed_sequences"] == {"personal": list(range(1, 17))}
    assert any(
        issue.get("issue_code") == "candidate_b_inquiry_sequence_inferred_from_row_order"
        and issue.get("candidate_value", {}).get("normalized_sequence") == 9
        for issue in context._personal_detail_extraction_issues
    )


def test_ye_inquiry_population_is_personal_1_to_16_and_institution_1_to_96_except_27() -> None:
    institution_sequences = [sequence for sequence in range(1, 97) if sequence != 27]
    institution_rows = [["编号", "查询日期", "查询机构", "查询原因"]]
    institution_rows.extend([str(sequence), "2022.05.22", "示例银行", "贷后管理"] for sequence in institution_sequences)
    personal_rows = [["编号 查询日期", "", "查询机构", "查询原因"]]
    personal_rows.extend(
        [
            "坏" if sequence == 9 else str(sequence),
            "2022.05.22",
            "本人",
            "本人查询",
        ]
        for sequence in range(1, 17)
    )
    context = SimpleNamespace(
        pages=[
            _page(
                29,
                [
                    _table("institution-inquiries", institution_rows),
                    _table("personal-inquiries", personal_rows, top=400.0),
                ],
                template="annotations_and_inquiries",
            )
        ],
        corrected_evidence_pages=lambda: [],
    )

    records = native_extraction._extract_inquiries(context)
    ledger = native_extraction._source_completeness_ledger(context)
    institution = [record["sequence"] for record in records if record["inquiry_type"] == "institution"]
    personal = [record["sequence"] for record in records if record["inquiry_type"] == "personal"]

    assert institution == institution_sequences
    assert personal == list(range(1, 17))
    assert ledger["inquiry_sequence_endpoints"] == {
        "institution": 96,
        "personal": 16,
    }
    assert ledger["inquiry_records"] == 112


@pytest.mark.parametrize(
    "header, body",
    [
        (
            ["查询日期 编号", "", "查询机构", "查询原因"],
            ["1", "2024.01.01", "本人", "本人查询"],
        ),
        (
            ["编号 查询日期", "", "查询机构", "查询原因"],
            ["1", "2024.01.01", "--", "本人查询"],
        ),
    ],
)
def test_collapsed_inquiry_header_repair_fails_closed_on_order_or_body_contract(
    header: list[str],
    body: list[str],
) -> None:
    rows = [header, body, ["2", "2024.01.02", "本人", "本人查询"]]
    assert native_extraction._bounded_collapsed_inquiry_header_slots(rows) is None


def _inquiry_header() -> list[str]:
    return [
        "\u7f16\u53f7",
        "\u67e5\u8be2\u65e5\u671f",
        "\u67e5\u8be2\u673a\u6784",
        "\u67e5\u8be2\u539f\u56e0",
    ]


@pytest.mark.parametrize(
    "resolution",
    [
        {"resolved": False, "authoritative": False},
        {"resolved": True, "authoritative": False},
    ],
)
def test_cross_page_inquiry_schema_carry_requires_authoritative_order(
    resolution: dict[str, bool],
) -> None:
    headed = _table(
        "headed-inquiries",
        [
            _inquiry_header(),
            ["1", "2024.01.01", "\u673a\u6784\u7532", "\u8d37\u6b3e\u5ba1\u6279"],
        ],
    )
    headerless = _table(
        "headerless-inquiries",
        [["2", "2024.01.02", "\u673a\u6784\u4e59", "\u8d37\u540e\u7ba1\u7406"]],
    )
    context = SimpleNamespace(
        pages=[
            _page(20, [headed], template="annotations_and_inquiries"),
            _page(17, [headerless], template="annotations_and_inquiries"),
        ],
        reading_order_by_logical={20: 1, 17: 2},
        reading_order_resolution=resolution,
        corrected_evidence_pages=lambda: [],
    )

    records = native_extraction._extract_inquiries(context)
    coverage = native_extraction._inquiry_source_coverage(context)

    assert [record["sequence"] for record in records] == [1]
    assert coverage["observed_sequences"] == {"institution": [1]}
    assert coverage["sequence_endpoints"] == {"institution": 1}
    assert any(
        issue.get("issue_code")
        == "candidate_b_inquiry_cross_page_schema_unresolved"
        and issue.get("source_refs", [{}])[0].get("table_id")
        == "headerless-inquiries"
        for issue in context._personal_detail_extraction_issues
    )


def test_cross_page_inquiry_schema_carry_accepts_authoritative_adjacency() -> None:
    headed = _table(
        "headed-inquiries",
        [
            _inquiry_header(),
            ["1", "2024.01.01", "\u673a\u6784\u7532", "\u8d37\u6b3e\u5ba1\u6279"],
        ],
    )
    headerless = _table(
        "headerless-inquiries",
        [["2", "2024.01.02", "\u673a\u6784\u4e59", "\u8d37\u540e\u7ba1\u7406"]],
    )
    context = SimpleNamespace(
        pages=[
            _page(20, [headed], template="annotations_and_inquiries"),
            _page(17, [headerless], template="annotations_and_inquiries"),
        ],
        reading_order_by_logical={20: 1, 17: 2},
        reading_order_resolution={"resolved": True, "authoritative": True},
        corrected_evidence_pages=lambda: [],
    )

    records = native_extraction._extract_inquiries(context)
    coverage = native_extraction._inquiry_source_coverage(context)

    assert [record["sequence"] for record in records] == [1, 2]
    assert coverage["observed_sequences"] == {"institution": [1, 2]}
    assert not hasattr(context, "_personal_detail_extraction_issues")


def test_cross_page_inquiry_schema_carry_rejects_a_partial_authoritative_map() -> None:
    headed = _table(
        "headed-inquiries",
        [
            _inquiry_header(),
            ["1", "2024.01.01", "\u673a\u6784\u7532", "\u8d37\u6b3e\u5ba1\u6279"],
        ],
    )
    headerless = _table(
        "headerless-inquiries",
        [["2", "2024.01.02", "\u673a\u6784\u4e59", "\u8d37\u540e\u7ba1\u7406"]],
    )
    context = SimpleNamespace(
        pages=[
            _page(20, [headed], template="annotations_and_inquiries"),
            _page(18, [], template="annotations_and_inquiries"),
            _page(17, [headerless], template="annotations_and_inquiries"),
        ],
        reading_order_by_logical={20: 1, 17: 2},
        reading_order_resolution={"resolved": True, "authoritative": True},
        corrected_evidence_pages=lambda: [],
    )

    records = native_extraction._extract_inquiries(context)
    coverage = native_extraction._inquiry_source_coverage(context)

    assert [record["sequence"] for record in records] == [1]
    assert coverage["observed_sequences"] == {"institution": [1]}


def test_same_page_inquiry_schema_carry_does_not_require_document_order() -> None:
    headed = _table(
        "headed-inquiries",
        [
            _inquiry_header(),
            ["1", "2024.01.01", "\u673a\u6784\u7532", "\u8d37\u6b3e\u5ba1\u6279"],
        ],
    )
    headerless = _table(
        "headerless-inquiries",
        [["2", "2024.01.02", "\u673a\u6784\u4e59", "\u8d37\u540e\u7ba1\u7406"]],
    )
    context = SimpleNamespace(
        pages=[
            _page(
                20,
                [headed, headerless],
                template="annotations_and_inquiries",
            )
        ],
        reading_order_by_logical={20: 1},
        reading_order_resolution={"resolved": False, "authoritative": False},
        corrected_evidence_pages=lambda: [],
    )

    records = native_extraction._extract_inquiries(context)
    coverage = native_extraction._inquiry_source_coverage(context)

    assert [record["sequence"] for record in records] == [1, 2]
    assert coverage["observed_sequences"] == {"institution": [1, 2]}
    assert not hasattr(context, "_personal_detail_extraction_issues")


def _canonical_inquiry_line_page(
    logical_page: int,
    sequence: int,
    *,
    date: str,
    institution: str,
    reason: str,
) -> dict[str, object]:
    return {
        "page": logical_page,
        "source_page": logical_page,
        "canonical_template_id": "annotations_and_inquiries",
        "lines": [
            {
                "text": f"{sequence} {date} {institution} {reason}",
                "bbox": [10.0, 100.0, 500.0, 120.0],
                "confidence": 0.99,
            }
        ],
    }


def test_unresolved_order_keeps_exact_canonical_line_sequences_page_local() -> None:
    evidence = [
        _canonical_inquiry_line_page(
            20,
            90,
            date="2024.01.01",
            institution="\u673a\u6784\u7532",
            reason="\u8d37\u6b3e\u5ba1\u6279",
        ),
        _canonical_inquiry_line_page(
            17,
            1,
            date="2024.01.02",
            institution="\u673a\u6784\u4e59",
            reason="\u8d37\u540e\u7ba1\u7406",
        ),
    ]
    context = SimpleNamespace(
        pages=[],
        reading_order_by_logical={20: 1, 17: 2},
        reading_order_resolution={"resolved": False, "authoritative": False},
        corrected_evidence_pages=lambda: evidence,
    )

    records = native_extraction._extract_inquiries(context)

    assert [record["sequence"] for record in records] == [1, 90]


def test_unresolved_order_canonical_line_exact_duplicates_still_dedupe() -> None:
    evidence = [
        _canonical_inquiry_line_page(
            logical_page,
            1,
            date="2024.01.01",
            institution="\u673a\u6784\u7532",
            reason="\u8d37\u6b3e\u5ba1\u6279",
        )
        for logical_page in (20, 17)
    ]
    context = SimpleNamespace(
        pages=[],
        reading_order_by_logical={20: 1, 17: 2},
        reading_order_resolution={"resolved": False, "authoritative": False},
        corrected_evidence_pages=lambda: evidence,
    )

    records = native_extraction._extract_inquiries(context)

    assert len(records) == 1
    assert records[0]["sequence"] == 1
    assert {ref["logical_page"] for ref in records[0]["source_refs"]} == {17, 20}


def test_unresolved_order_does_not_classify_a_separate_headed_inquiry_group() -> None:
    typed = _table(
        "typed-inquiries",
        [
            _inquiry_header(),
            ["1", "2024.01.01", "\u673a\u6784\u7532", "\u8d37\u6b3e\u5ba1\u6279"],
        ],
    )
    untyped = _table(
        "untyped-inquiries",
        [_inquiry_header(), ["2", "2024.01.02", "", ""]],
    )
    context = SimpleNamespace(
        pages=[
            _page(20, [typed], template="annotations_and_inquiries"),
            _page(17, [untyped], template="annotations_and_inquiries"),
        ],
        reading_order_by_logical={20: 1, 17: 2},
        reading_order_resolution={"resolved": False, "authoritative": False},
    )

    coverage = native_extraction._inquiry_source_coverage(context)

    assert coverage["observed_sequences"] == {"institution": [1]}
    assert coverage["unclassified_sequence_endpoints"] == [2]


def test_authoritative_order_classifies_an_adjacent_repeated_inquiry_header() -> None:
    typed = _table(
        "typed-inquiries",
        [
            _inquiry_header(),
            ["1", "2024.01.01", "\u673a\u6784\u7532", "\u8d37\u6b3e\u5ba1\u6279"],
        ],
    )
    untyped = _table(
        "untyped-inquiries",
        [_inquiry_header(), ["2", "2024.01.02", "", ""]],
    )
    context = SimpleNamespace(
        pages=[
            _page(20, [typed], template="annotations_and_inquiries"),
            _page(17, [untyped], template="annotations_and_inquiries"),
        ],
        reading_order_by_logical={20: 1, 17: 2},
        reading_order_resolution={"resolved": True, "authoritative": True},
    )

    coverage = native_extraction._inquiry_source_coverage(context)

    assert coverage["observed_sequences"] == {"institution": [1, 2]}
    assert "unclassified_sequence_endpoints" not in coverage


def _housing_fund_split_tables() -> tuple[SimpleNamespace, SimpleNamespace]:
    layouts = {
        str(layout["name"]): layout
        for layout in native_extraction._PUBLIC_CANONICAL_LAYOUTS
    }
    base = layouts["housing_fund_base"]
    provider = layouts["housing_fund_provider"]

    def header(layout: dict[str, object]) -> list[str]:
        aliases = layout["aliases"]
        fields = layout["fields"]
        assert isinstance(aliases, dict)
        assert isinstance(fields, dict)
        return [aliases[role][0] for role in fields]

    return (
        _table(
            "housing-fund-base",
            [
                header(base),
                [
                    "Fuzhou",
                    "2018.09.03",
                    "2018.09",
                    "2023.08",
                    "active",
                    "906",
                    "6%",
                    "6%",
                ],
            ],
        ),
        _table(
            "housing-fund-provider",
            [
                header(provider),
                ["\u793a\u4f8b\u79d1\u6280\u6709\u9650\u516c\u53f8", "2023.08"],
            ],
        ),
    )


@pytest.mark.parametrize(
    "resolution",
    [
        {"resolved": False, "authoritative": False},
        {"resolved": True, "authoritative": False},
    ],
)
def test_housing_fund_cross_page_provider_requires_authoritative_order(
    resolution: dict[str, bool],
) -> None:
    base, provider = _housing_fund_split_tables()
    context = SimpleNamespace(
        pages=[_page(1, [base]), _page(2, [provider])],
        reading_order_by_logical={1: 1, 2: 2},
        reading_order_resolution=resolution,
        _personal_detail_extraction_issues=[],
    )

    records = native_extraction._extract_public_records(context)
    housing = [record for record in records if record["record_type"] == "housing_fund"]

    assert len(housing) == 1
    assert housing[0]["contribution_location"] == "Fuzhou"
    assert "employer" not in housing[0]
    issues = context._personal_detail_extraction_issues
    assert [issue["issue_code"] for issue in issues] == [
        "candidate_b_public_record_continuation_missing",
        "candidate_b_public_record_continuation_unowned",
    ]
    assert issues[0]["target_record_id"] == housing[0]["public_record_id"]
    assert issues[0]["candidate_value"]["missing_fields"] == [
        "employer",
        "information_updated_month",
    ]
    assert issues[1]["source_refs"][0]["table_id"] == "housing-fund-provider"


def test_housing_fund_cross_page_provider_rejects_a_partial_order_plane() -> None:
    base, provider = _housing_fund_split_tables()
    context = SimpleNamespace(
        pages=[_page(1, [base]), _page(2, [provider]), _page(3, [])],
        reading_order_by_logical={1: 1, 2: 2},
        reading_order_resolution={"resolved": True, "authoritative": True},
        _personal_detail_extraction_issues=[],
    )

    records = native_extraction._extract_public_records(context)
    housing = [record for record in records if record["record_type"] == "housing_fund"]

    assert len(housing) == 1
    assert "employer" not in housing[0]
    assert [
        issue["issue_code"]
        for issue in context._personal_detail_extraction_issues
    ] == [
        "candidate_b_public_record_continuation_missing",
        "candidate_b_public_record_continuation_unowned",
    ]


def test_housing_fund_cross_page_provider_accepts_authoritative_adjacency() -> None:
    base, provider = _housing_fund_split_tables()
    context = SimpleNamespace(
        pages=[_page(1, [base]), _page(2, [provider])],
        reading_order_by_logical={1: 1, 2: 2},
        reading_order_resolution={"resolved": True, "authoritative": True},
        _personal_detail_extraction_issues=[],
    )

    records = native_extraction._extract_public_records(context)
    housing = [record for record in records if record["record_type"] == "housing_fund"]

    assert len(housing) == 1
    assert housing[0]["employer"] == "\u793a\u4f8b\u79d1\u6280\u6709\u9650\u516c\u53f8"
    assert not context._personal_detail_extraction_issues


def test_housing_fund_same_page_provider_does_not_require_document_order() -> None:
    base, provider = _housing_fund_split_tables()
    context = SimpleNamespace(
        pages=[_page(1, [base, provider])],
        reading_order_by_logical={1: 1},
        reading_order_resolution={"resolved": False, "authoritative": False},
        _personal_detail_extraction_issues=[],
    )

    records = native_extraction._extract_public_records(context)
    housing = [record for record in records if record["record_type"] == "housing_fund"]

    assert len(housing) == 1
    assert housing[0]["employer"] == "\u793a\u4f8b\u79d1\u6280\u6709\u9650\u516c\u53f8"
    assert not context._personal_detail_extraction_issues


def _agreement_ledger_context(
    resolution: dict[str, bool],
    *,
    partial: bool = False,
) -> SimpleNamespace:
    evidence = [
        {
            "page": 1,
            "source_page": 1,
            "lines": [
                {"text": "\u6388\u4fe1\u534f\u8bae\u4fe1\u606f"},
                {"text": "\u6388\u4fe1\u534f\u8bae\u6807\u8bc6 A"},
            ],
        },
        {
            "page": 2,
            "source_page": 2,
            "lines": [{"text": "\u6388\u4fe1\u534f\u8bae\u6807\u8bc6 B"}],
        },
    ]
    pages = [_page(1, []), _page(2, [])]
    if partial:
        pages.append(_page(3, []))
        evidence.append({"page": 3, "source_page": 3, "lines": []})
    return SimpleNamespace(
        pages=pages,
        reading_order_by_logical={1: 1, 2: 2},
        reading_order_resolution=resolution,
        corrected_evidence_pages=lambda: evidence,
        _personal_detail_extraction_issues=[],
    )


@pytest.mark.parametrize(
    "resolution, partial",
    [
        ({"resolved": False, "authoritative": False}, False),
        ({"resolved": True, "authoritative": False}, False),
        ({"resolved": True, "authoritative": True}, True),
    ],
)
def test_agreement_source_ledger_does_not_carry_unproven_page_state(
    resolution: dict[str, bool],
    partial: bool,
) -> None:
    context = _agreement_ledger_context(resolution, partial=partial)

    ledger = native_extraction._source_completeness_ledger(context)

    assert ledger["credit_agreements"] == 1


def test_agreement_source_ledger_carries_authoritative_adjacent_page_state() -> None:
    context = _agreement_ledger_context(
        {"resolved": True, "authoritative": True}
    )

    ledger = native_extraction._source_completeness_ledger(context)

    assert ledger["credit_agreements"] == 2


def test_agreement_source_ledger_keeps_same_page_section_state() -> None:
    context = SimpleNamespace(
        pages=[_page(1, [])],
        reading_order_by_logical={1: 1},
        reading_order_resolution={"resolved": False, "authoritative": False},
        corrected_evidence_pages=lambda: [
            {
                "page": 1,
                "source_page": 1,
                "lines": [
                    {"text": "\u6388\u4fe1\u534f\u8bae\u4fe1\u606f"},
                    {"text": "\u6388\u4fe1\u534f\u8bae\u6807\u8bc6 A"},
                    {"text": "\u6388\u4fe1\u534f\u8bae\u6807\u8bc6 B"},
                ],
            }
        ],
        _personal_detail_extraction_issues=[],
    )

    ledger = native_extraction._source_completeness_ledger(context)

    assert ledger["credit_agreements"] == 2


def _native_evidence_line(
    text: str,
    x: float,
    y: float,
    *,
    width: float = 140.0,
) -> dict[str, object]:
    return {
        "text": text,
        "bbox": [x, y, x + width, y + 18.0],
        "confidence": 0.99,
    }


def _split_credit_agreement_evidence() -> list[dict[str, object]]:
    return [
        {
            "page": 1,
            "source_page": 1,
            "lines": [
                _native_evidence_line("\u6388\u4fe1\u534f\u8bae\u4fe1\u606f", 20, 10),
                _native_evidence_line("\u6388\u4fe1\u534f\u8bae1", 20, 40),
                _native_evidence_line("\u6388\u4fe1\u534f\u8bae\u6807\u8bc6", 20, 70),
                _native_evidence_line("AGREEMENT0001", 20, 100),
            ],
        },
        {
            "page": 2,
            "source_page": 2,
            "lines": [
                _native_evidence_line("\u6388\u4fe1\u989d\u5ea6\u7528\u9014", 20, 20),
                _native_evidence_line("\u5faa\u73af\u989d\u5ea6", 20, 50),
            ],
        },
    ]


@pytest.mark.parametrize(
    "resolution, partial",
    [
        ({"resolved": False, "authoritative": False}, False),
        ({"resolved": True, "authoritative": False}, False),
        ({"resolved": True, "authoritative": True}, True),
    ],
)
def test_native_parser_does_not_merge_credit_line_across_unproven_page_order(
    resolution: dict[str, bool],
    partial: bool,
) -> None:
    pages = [_page(1, []), _page(2, [])]
    if partial:
        pages.append(_page(3, []))
    evidence = _split_credit_agreement_evidence()
    context = SimpleNamespace(
        pages=pages,
        reading_order_by_logical={1: 1, 2: 2},
        reading_order_resolution=resolution,
        corrected_evidence_pages=lambda: evidence,
        _personal_detail_extraction_issues=[],
    )

    records = PBOCPersonalDetailNativeParser(context).records("credit_lines")

    assert len(records) == 1
    assert records[0].fields["\u6388\u4fe1\u534f\u8bae\u6807\u8bc6"] == "AGREEMENT0001"
    assert "\u6388\u4fe1\u989d\u5ea6\u7528\u9014" not in records[0].fields
    assert {ref["logical_page"] for ref in records[0].source_refs} == {1}
    assert any(
        issue.get("issue_code")
        == "candidate_b_native_evidence_page_order_unresolved"
        and issue.get("target_dataset") == "credit_lines"
        and issue.get("source_refs", [{}])[0].get("logical_page") == 2
        for issue in context._personal_detail_extraction_issues
    )


def test_native_parser_does_not_merge_liability_across_unresolved_page_order() -> None:
    evidence = [
        {
            "page": 1,
            "source_page": 1,
            "lines": [
                _native_evidence_line(
                    "\u76f8\u5173\u8fd8\u6b3e\u8d23\u4efb\u4fe1\u606f",
                    20,
                    10,
                ),
                _native_evidence_line("\u8d26\u62371", 20, 40),
                _native_evidence_line("\u8d23\u4efb\u4eba\u7c7b\u578b", 20, 70),
                _native_evidence_line("\u8fd8\u6b3e\u8d23\u4efb\u91d1\u989d", 200, 70),
                _native_evidence_line("\u4fdd\u8bc1\u4eba", 20, 100),
                _native_evidence_line("200000", 200, 100),
            ],
        },
        {
            "page": 2,
            "source_page": 2,
            "lines": [
                _native_evidence_line("\u4fdd\u8bc1\u5408\u540c\u7f16\u53f7", 20, 20),
                _native_evidence_line("G-UNOWNED", 20, 50),
            ],
        },
    ]
    context = SimpleNamespace(
        pages=[_page(1, []), _page(2, [])],
        reading_order_by_logical={1: 1, 2: 2},
        reading_order_resolution={"resolved": False, "authoritative": False},
        corrected_evidence_pages=lambda: evidence,
        _personal_detail_extraction_issues=[],
    )

    records = PBOCPersonalDetailNativeParser(context).records(
        "repayment_liability_records"
    )

    assert len(records) == 1
    assert records[0].fields["\u8fd8\u6b3e\u8d23\u4efb\u91d1\u989d"] == "200000"
    assert "\u4fdd\u8bc1\u5408\u540c\u7f16\u53f7" not in records[0].fields
    assert {ref["logical_page"] for ref in records[0].source_refs} == {1}
    assert any(
        issue.get("issue_code")
        == "candidate_b_native_evidence_page_order_unresolved"
        and issue.get("target_dataset") == "repayment_liability_records"
        for issue in context._personal_detail_extraction_issues
    )


def test_native_parser_merges_credit_line_across_authoritative_adjacency() -> None:
    evidence = _split_credit_agreement_evidence()
    context = SimpleNamespace(
        pages=[_page(1, []), _page(2, [])],
        reading_order_by_logical={1: 1, 2: 2},
        reading_order_resolution={"resolved": True, "authoritative": True},
        corrected_evidence_pages=lambda: evidence,
        _personal_detail_extraction_issues=[],
    )

    records = PBOCPersonalDetailNativeParser(context).records("credit_lines")

    assert len(records) == 1
    assert records[0].fields["\u6388\u4fe1\u989d\u5ea6\u7528\u9014"] == "\u5faa\u73af\u989d\u5ea6"
    assert {ref["logical_page"] for ref in records[0].source_refs} == {1, 2}
    assert not context._personal_detail_extraction_issues


def test_native_parser_merges_credit_line_fragments_on_the_same_page() -> None:
    evidence = _split_credit_agreement_evidence()
    evidence[1]["page"] = 1
    evidence[1]["source_page"] = 1
    context = SimpleNamespace(
        pages=[_page(1, [])],
        reading_order_by_logical={1: 1},
        reading_order_resolution={"resolved": False, "authoritative": False},
        corrected_evidence_pages=lambda: evidence,
        _personal_detail_extraction_issues=[],
    )

    records = PBOCPersonalDetailNativeParser(context).records("credit_lines")

    assert len(records) == 1
    assert records[0].fields["\u6388\u4fe1\u989d\u5ea6\u7528\u9014"] == "\u5faa\u73af\u989d\u5ea6"
    assert not context._personal_detail_extraction_issues


def _report_header_evidence_page(logical_page: int) -> dict[str, object]:
    labels = (
        "\u88ab\u67e5\u8be2\u8005\u59d3\u540d",
        "\u88ab\u67e5\u8be2\u8005\u8bc1\u4ef6\u7c7b\u578b",
        "\u88ab\u67e5\u8be2\u8005\u8bc1\u4ef6\u53f7\u7801",
        "\u67e5\u8be2\u673a\u6784",
        "\u67e5\u8be2\u539f\u56e0",
    )
    values = (
        "\u5f20\u4e09",
        "\u8eab\u4efd\u8bc1",
        "110101199001011234",
        "\u4e2d\u56fd\u4eba\u6c11\u94f6\u884c\u5f81\u4fe1\u4e2d\u5fc3",
        "\u672c\u4eba\u67e5\u8be2",
    )
    return {
        "page": logical_page,
        "source_page": logical_page,
        "lines": [
            *(
                _native_evidence_line(label, index * 180.0, 20.0, width=160.0)
                for index, label in enumerate(labels)
            ),
            *(
                _native_evidence_line(value, index * 180.0, 50.0, width=160.0)
                for index, value in enumerate(values)
            ),
        ],
    }


def test_native_parser_report_header_uses_authoritative_first_page() -> None:
    report_header = _report_header_evidence_page(20)
    summary = {
        "page": 17,
        "source_page": 17,
        "lines": [_native_evidence_line("\u4fe1\u606f\u6982\u8981", 20, 20)],
    }
    context = SimpleNamespace(
        pages=[_page(20, []), _page(17, [])],
        reading_order_by_logical={20: 1, 17: 2},
        reading_order_resolution={"resolved": True, "authoritative": True},
        corrected_evidence_pages=lambda: [summary, report_header],
        _personal_detail_extraction_issues=[],
    )

    records = PBOCPersonalDetailNativeParser(context).records("report_header")

    assert len(records) == 1
    assert records[0].fields["\u88ab\u67e5\u8be2\u8005\u59d3\u540d"] == "\u5f20\u4e09"
    assert {ref["logical_page"] for ref in records[0].source_refs} == {20}


@pytest.mark.parametrize(
    "resolution, order, extra_page",
    [
        ({"resolved": False, "authoritative": False}, {20: 1, 17: 2}, False),
        ({"resolved": True, "authoritative": False}, {20: 1, 17: 2}, False),
        ({"resolved": True, "authoritative": True}, {20: 1}, True),
    ],
)
def test_native_parser_report_header_fails_closed_without_complete_authoritative_order(
    resolution: dict[str, bool],
    order: dict[int, int],
    extra_page: bool,
) -> None:
    report_header = _report_header_evidence_page(20)
    summary = {
        "page": 17,
        "source_page": 17,
        "lines": [_native_evidence_line("\u4fe1\u606f\u6982\u8981", 20, 20)],
    }
    pages = [_page(20, []), _page(17, [])]
    if extra_page:
        pages.append(_page(18, []))
    context = SimpleNamespace(
        pages=pages,
        reading_order_by_logical=order,
        reading_order_resolution=resolution,
        corrected_evidence_pages=lambda: [report_header, summary],
        _personal_detail_extraction_issues=[],
    )

    records = PBOCPersonalDetailNativeParser(context).records("report_header")

    assert records == []
    assert any(
        issue.get("issue_code")
        == "candidate_b_native_evidence_page_order_unresolved"
        and issue.get("target_dataset") == "personal_report_metadata"
        for issue in context._personal_detail_extraction_issues
    )


def test_native_parser_report_header_accepts_authoritative_identity_order() -> None:
    report_header = _report_header_evidence_page(1)
    summary = {
        "page": 2,
        "source_page": 2,
        "lines": [_native_evidence_line("\u4fe1\u606f\u6982\u8981", 20, 20)],
    }
    context = SimpleNamespace(
        pages=[_page(1, []), _page(2, [])],
        reading_order_by_logical={1: 1, 2: 2},
        reading_order_resolution={"resolved": True, "authoritative": True},
        corrected_evidence_pages=lambda: [report_header, summary],
        _personal_detail_extraction_issues=[],
    )

    records = PBOCPersonalDetailNativeParser(context).records("report_header")

    assert len(records) == 1
    assert records[0].fields["\u88ab\u67e5\u8be2\u8005\u59d3\u540d"] == "\u5f20\u4e09"


def test_account_endpoint_accepts_consecutive_high_tail_but_not_sparse_joined_value() -> None:
    assert native_extraction._credible_sequence_endpoint({1, 2, 3, 10, 11, 12}) == (
        12,
        [],
    )
    assert native_extraction._credible_sequence_endpoint({1, 3, 115}) == (3, [115])


def test_account_source_ledger_keeps_exact_family_populations_without_cancellation() -> None:
    lines: list[dict[str, str]] = []
    for heading, endpoint in (
        ("（一）非循环贷账户", 18),
        ("（二）循环贷账户一", 6),
        ("（三）循环贷账户二", 6),
        ("（四）贷记卡账户", 12),
    ):
        lines.append({"text": heading})
        lines.extend({"text": f"账户{sequence}"} for sequence in range(1, endpoint + 1))
    lines.append({"text": "（五）授信协议信息"})
    context = SimpleNamespace(
        pages=[],
        reading_order_by_logical={1: 1},
        corrected_evidence_pages=lambda: [{"page": 1, "source_page": 1, "lines": lines}],
    )

    ledger = native_extraction._source_completeness_ledger(context)

    assert ledger["credit_accounts"] == 42
    assert ledger["account_family_source_populations"] == {
        "credit_card": 12,
        "non_revolving_loan": 18,
        "revolving_loan_account": 6,
        "revolving_loan_subaccount": 6,
    }


def test_scrambled_logical_pages_reclassify_cards_and_preserve_strong_ids() -> None:
    card_ids = [
        "B10911000H000115603050013394541",
        "B11313900H000115603090424251222",
        "D10123910H000115604050032149",
    ]
    evidence = [
        {
            "page": 16,
            "source_page": 8,
            "lines": [
                {"text": "循环贷账户（二）", "bbox": [10, 10, 200, 25]},
                {"text": "账户 6", "bbox": [10, 40, 150, 55]},
            ],
        },
        {
            # Stored before logical 19, but printed after its card-family heading.
            "page": 17,
            "source_page": 9,
            "lines": [
                {"text": "账户 4", "bbox": [10, 20, 150, 35]},
                {"text": "账户 5", "bbox": [10, 220, 150, 235]},
                {"text": "账户 6", "bbox": [10, 420, 150, 435]},
            ],
        },
        {
            "page": 19,
            "source_page": 10,
            "lines": [
                {"text": "贷记卡账户", "bbox": [10, 10, 200, 25]},
                {"text": "账户 1", "bbox": [10, 40, 150, 55]},
            ],
        },
    ]
    pages = [
        _page(16, []),
        _page(
            17,
            [
                _table(f"card-{sequence}", [_CARD_HEADER, _card_values(identifier)], top=top)
                for sequence, identifier, top in zip(
                    (4, 5, 6),
                    card_ids,
                    (60.0, 260.0, 460.0),
                    strict=True,
                )
            ],
        ),
        _page(
            19,
            [
                _table(
                    "card-1",
                    [_CARD_HEADER, _card_values("B10000000H000100000000000000001")],
                    top=60.0,
                )
            ],
        ),
    ]
    context = SimpleNamespace(
        pages=pages,
        reading_order_by_logical={16: 1, 19: 2, 17: 3},
        corrected_evidence_pages=lambda: evidence,
        tables_continue=lambda _left, _right: None,
    )

    accounts, _repayments, _events = native_extraction._extract_accounts(context)
    recovered = [account for account in accounts if account.get("account_identifier") in set(card_ids)]

    assert [account["category_sequence"] for account in recovered] == [4, 5, 6]
    assert [account["account_type"] for account in recovered] == [
        "credit_card",
        "credit_card",
        "credit_card",
    ]
    assert [account["account_identifier"] for account in recovered] == card_ids


def _headerless_card_context(
    *,
    candidate_identifier: str = "B10611000H00016226880219191368607",
    candidate_currency: str = "人民币",
) -> SimpleNamespace:
    card7 = "B10411000H000115602800002159651279117266"
    card9 = "B11911000H000115661000042356833"
    previous = _table(
        "pt_23_1",
        [
            _CARD_HEADER,
            _card_values(card7),
            _CARD_HEADER,
        ],
        top=100.0,
    )
    candidate_values = _card_values(candidate_identifier)
    candidate_values[5] = candidate_currency
    candidate = _table("pt_24_0", [candidate_values], top=10.0)
    following = _table("pt_25_0", [_CARD_HEADER, _card_values(card9)], top=100.0)
    evidence = [
        {
            "page": 23,
            "source_page": 12,
            "lines": [
                {"text": "贷记卡账户", "bbox": [10, 10, 200, 25]},
                {"text": "账户 7", "bbox": [10, 40, 150, 55]},
                {"text": "账户 8", "bbox": [10, 320, 150, 335]},
            ],
        },
        {
            "page": 24,
            "source_page": 12,
            "lines": [{"text": "续页", "bbox": [10, 10, 60, 25]}],
        },
        {
            "page": 25,
            "source_page": 13,
            "lines": [{"text": "账户 9", "bbox": [10, 40, 150, 55]}],
        },
    ]
    return SimpleNamespace(
        pages=[_page(23, [previous]), _page(24, [candidate]), _page(25, [following])],
        reading_order_by_logical={23: 1, 24: 2, 25: 3},
        corrected_evidence_pages=lambda: evidence,
        tables_continue=lambda left, right: (left, right) == ("pt_23_1", "pt_24_0"),
        allows_scanned_line_transition=lambda *_args: False,
    )


def test_headerless_next_page_card_gets_distinct_anchor_and_table_ownership() -> None:
    context = _headerless_card_context()

    accounts, _repayments, _events = native_extraction._extract_accounts(context)
    by_sequence = {
        int(account["category_sequence"]): account
        for account in accounts
        if account.get("account_type") == "credit_card"
    }

    assert sorted(by_sequence) == [7, 8, 9]
    assert by_sequence[8]["account_identifier"] == "B10611000H00016226880219191368607"
    assert "pt_24_0" not in {ref.get("table_id") for ref in by_sequence[7].get("source_refs") or ()}
    assert "pt_24_0" in {ref.get("table_id") for ref in by_sequence[8].get("source_refs") or ()}
    assert any(
        issue.get("issue_code") == "candidate_b_headerless_account_owner_resolved"
        and issue.get("target_record_id") == by_sequence[8]["account_id"]
        for issue in context._personal_detail_extraction_issues
    )


@pytest.mark.parametrize(
    "identifier",
    [
        "B10911000H000115603050013394541",
        "B11313900H000115603090424251222",
        "D10123910H000115604050032149",
        "B10411000H000115602800002159651279117266",
        "B10611000H00016226880219191368607",
        "B11911000H000115661000042356833",
    ],
)
def test_headerless_identifier_contract_preserves_all_six_ye_cards(
    identifier: str,
) -> None:
    assert native_extraction._canonical_pboc_account_identifier(identifier) == identifier


@pytest.mark.parametrize(
    "candidate_identifier, candidate_currency",
    [
        # Exact replay cannot become a new entity.
        ("B10411000H000115602800002159651279117266", "人民币"),
        # A weak identity-bearing value row fails the finite card contract.
        ("BAD", "人民币"),
        ("A00000000000", "人民币"),
        ("ABCD12345678", "人民币"),
        ("B10611000H00016226880219191368607", "未知币种"),
    ],
)
def test_headerless_card_split_rejects_replay_and_weak_identity_rows(
    candidate_identifier: str,
    candidate_currency: str,
) -> None:
    context = _headerless_card_context(
        candidate_identifier=candidate_identifier,
        candidate_currency=candidate_currency,
    )

    table_accounts, _repayments, _events = native_extraction._extract_table_accounts(context)

    assert not any(account.get("_pending_anchor_account_id") for account in table_accounts)
    assert not any(
        issue.get("issue_code") == "candidate_b_headerless_account_owner_resolved"
        for issue in getattr(context, "_personal_detail_extraction_issues", [])
    )


def test_headerless_card_split_rejects_repayment_only_continuation() -> None:
    context = _headerless_card_context()
    context.pages[1].tables[0].metadata["raw_rows"] = [
        ["还款记录", "2024.01", "N"],
    ]

    table_accounts, _repayments, _events = native_extraction._extract_table_accounts(context)

    assert not any(account.get("_pending_anchor_account_id") for account in table_accounts)


def test_nonunique_account_reading_order_falls_back_with_explicit_issue() -> None:
    pages = [_page(1, []), _page(2, [])]
    context = SimpleNamespace(reading_order_by_logical={1: 1, 2: 1})

    assert native_extraction._account_ordered_pages(context, pages) == pages
    issue = context._personal_detail_extraction_issues[0]
    assert issue["issue_code"] == "candidate_b_account_reading_order_unresolved"
    assert issue["observed_value"]["duplicate_reading_positions"] == [1]


@pytest.mark.parametrize(
    "reading_order, continuation_decision",
    [
        ({1: 1}, None),
        ({1: 1}, True),
        ({1: 1, 2: 1}, True),
    ],
)
def test_unresolved_account_order_blocks_every_cross_page_table_owner(
    reading_order: dict[int, int],
    continuation_decision: bool | None,
) -> None:
    first = _table(
        "base-1",
        [_CARD_HEADER, _card_values("B10000000H000100000000000000001")],
        top=100.0,
    )
    fragment = _table(
        "fragment",
        [["账户状态", "账户关闭日期"], ["正常", "2024.01.01"]],
        top=10.0,
    )
    second = _table(
        "base-2",
        [_CARD_HEADER, _card_values("B10000000H000100000000000000002")],
        top=300.0,
    )
    context = SimpleNamespace(
        pages=[_page(1, [first]), _page(2, [fragment, second])],
        reading_order_by_logical=reading_order,
        corrected_evidence_pages=lambda: [],
        tables_continue=lambda left, right: (
            continuation_decision
            if (left, right) == ("base-1", "fragment")
            else None
        ),
        allows_scanned_line_transition=lambda *_args: False,
    )

    accounts, _repayments, _events = native_extraction._extract_table_accounts(context)

    assert [
        ref.get("table_id") for ref in accounts[0].get("source_refs") or ()
    ] == ["base-1"]
    assert any(
        issue.get("issue_code") == "candidate_b_account_reading_order_unresolved"
        for issue in context._personal_detail_extraction_issues
    )


def test_same_page_geometric_account_owner_survives_order_hardening() -> None:
    first = _table(
        "base-1",
        [_CARD_HEADER, _card_values("B10000000H000100000000000000001")],
        top=100.0,
    )
    fragment = _table(
        "fragment",
        [["账户状态", "账户关闭日期"], ["正常", "2024.01.01"]],
        top=240.0,
    )
    context = SimpleNamespace(
        pages=[_page(1, [first, fragment])],
        reading_order_by_logical={1: 1},
        corrected_evidence_pages=lambda: [],
        tables_continue=lambda *_args: None,
    )

    accounts, _repayments, _events = native_extraction._extract_table_accounts(context)

    assert [
        ref.get("table_id") for ref in accounts[0].get("source_refs") or ()
    ] == ["base-1", "fragment"]


def test_partial_order_disables_headerless_next_page_card_owner() -> None:
    context = _headerless_card_context()
    context.reading_order_by_logical = {23: 1, 24: 2}

    table_accounts, _repayments, _events = native_extraction._extract_table_accounts(
        context
    )

    assert not any(
        account.get("_pending_anchor_account_id") for account in table_accounts
    )
    assert not any(
        issue.get("issue_code") == "candidate_b_headerless_account_owner_resolved"
        for issue in getattr(context, "_personal_detail_extraction_issues", [])
    )


def test_repeated_page_object_cannot_remap_suppressed_instance_children() -> None:
    rows = [
        _CARD_HEADER,
        _card_values("B10000000H000100000000000000001"),
        ["特殊事件说明"],
        ["测试事件"],
        ["", *(str(month) for month in range(1, 13))],
        ["2024", *("N" for _month in range(1, 13))],
        ["", *("--" for _month in range(1, 13))],
    ]
    table = _table("same-table", rows, top=100.0)
    repeated_page = _page(1, [table])
    evidence = [
        {
            "page": 1,
            "source_page": 1,
            "lines": [
                {"text": "（四）贷记卡账户", "bbox": [0, 0, 200, 10]},
                {"text": "账户 1", "bbox": [0, 20, 100, 30]},
            ],
        }
    ]
    context = SimpleNamespace(
        pages=[repeated_page, repeated_page],
        reading_order_by_logical={1: 1},
        corrected_evidence_pages=lambda: evidence,
        tables_continue=lambda *_args: None,
        allows_scanned_line_transition=lambda *_args: False,
    )

    accounts, repayments, events = native_extraction._extract_accounts(context)

    assert [account["account_id"] for account in accounts] == [
        "credit_account:credit_card:1"
    ]
    assert len(repayments) == 12
    assert len(events) == 1
    accepted_instance = events[0]["_table_observation_instance_id"]
    assert {
        repayment["_table_observation_instance_id"] for repayment in repayments
    } == {accepted_instance}
    suppressed = next(
        issue
        for issue in context._personal_detail_extraction_issues
        if issue.get("issue_code") == "candidate_b_unmatched_account_table_suppressed"
    )
    observed = suppressed["observed_value"]
    assert observed["account_observation_instance_id"] != accepted_instance
    assert suppressed["target_record_id"] == observed[
        "account_observation_instance_id"
    ]
    assert observed["suppressed_child_counts_by_dataset"] == {
        "credit_account_monthly_performance": 12,
        "credit_account_special_events": 1,
    }
    assert "record_not_emitted_due_to_unresolved_account_ownership" in suppressed[
        "reason_codes"
    ]


def test_suppressed_unmatched_account_issue_retains_locator_and_nonemission_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observation_id = "credit_account_table_observation:unmatched"
    table = {
        "account_id": observation_id,
        "_table_observation_id": observation_id,
        "account_type": "credit_card",
        "source_refs": [{"logical_page": 4, "table_id": "pt_4_0"}],
    }
    anchor = {
        "account_id": "credit_account:credit_card:1",
        "account_type": "credit_card",
        "category_sequence": 1,
        "account_family_quality": "exact",
        "_printed_ordinal_status": "printed_unique",
        "source_refs": [{"logical_page": 3, "bbox": [10, 20, 100, 40]}],
    }
    context = SimpleNamespace(
        _personal_detail_extraction_issues=[
            {
                "issue_code": "candidate_b_account_cluster_residue",
                "target_dataset": "credit_accounts",
                "target_record_id": observation_id,
                "field_name": "open_date",
                "reason_codes": ["uniquely_typed_value_retained", "cell_residue_reported"],
            }
        ]
    )
    monkeypatch.setattr(
        native_extraction,
        "_extract_table_accounts",
        lambda _context: ([table], [], []),
    )
    monkeypatch.setattr(
        native_extraction,
        "_account_anchor_skeletons",
        lambda _context: [anchor],
    )

    accounts, _repayments, _events = native_extraction._extract_accounts(context)

    assert accounts == [anchor]
    original = context._personal_detail_extraction_issues[0]
    assert original["target_record_id"] == observation_id
    assert "record_not_emitted_due_to_unresolved_account_ownership" in original["reason_codes"]


@pytest.mark.parametrize(
    ("shape", "labels", "values", "expected"),
    [
        (
            "pt_6_2_non_revolving",
            ("管理机构", "账户标识", "开立日期"),
            ("上海汽车集团财务有限责任公司", "N10252900H00013539300", "2017.12.15"),
            {
                "management_institution": "上海汽车集团财务有限责任公司",
                "account_identifier": "N10252900H00013539300",
                "open_date": "2017-12-15",
            },
        ),
        (
            "r1_credit_terms",
            ("账户授信额度", "币种", "业务种类", "担保方式"),
            ("120000", "人民币元", "个人汽车消费贷款", "抵押"),
            {
                "credit_limit": 120000,
                "currency": "CNY",
                "account_currency": "CNY",
                "business_type": "个人汽车消费贷款",
                "guarantee_type": "抵押",
            },
        ),
        (
            "r2_repayment_terms",
            ("业务种类", "担保方式", "还款期数"),
            ("个人住房商业贷款", "抵押", "288"),
            {
                "business_type": "个人住房商业贷款",
                "guarantee_type": "抵押",
                "repayment_periods": 288,
            },
        ),
        (
            "credit_card",
            ("发卡机构", "账户标识", "开立日期"),
            ("招商银行股份有限公司", "B10911000H000115603050013394541", "2020.01.02"),
            {
                "management_institution": "招商银行股份有限公司",
                "account_identifier": "B10911000H000115603050013394541",
                "open_date": "2020-01-02",
            },
        ),
    ],
)
def test_exact_merged_account_geometry_binds_only_its_physical_cells(
    shape: str,
    labels: tuple[str, ...],
    values: tuple[str, ...],
    expected: dict[str, object],
) -> None:
    interleaved = shape == "pt_6_2_non_revolving"
    table = _exact_merged_geometry_table(
        shape,
        labels,
        values,
        interleaved=interleaved,
    )
    result = SimpleNamespace(_personal_detail_extraction_issues=[])
    account = {"account_id": f"account:{shape}", "canonical_raw": {}}

    native_extraction._apply_collapsed_account_clusters(
        result,
        account,
        table.metadata["raw_rows"],
        page=_page(1, [table]),
        table=table,
        physical_row_indices=None,
    )

    assert {field: account[field] for field in expected} == expected
    exact_refs = [
        ref
        for refs in account.get("source_refs_by_field", {}).values()
        for ref in refs
        if ref.get("binding") == "closed_canonical_account_merged_header_geometry"
    ]
    expected_columns = (
        {1 + index * 2 for index in range(len(labels))}
        if interleaved
        else set(range(len(labels)))
    )
    assert {int(ref["column"]) for ref in exact_refs} == expected_columns
    assert all(ref.get("geometry_scope") == "cell" for ref in exact_refs)
    assert result._personal_detail_extraction_issues == []


@pytest.mark.parametrize(
    "defect",
    [
        "shifted_value_bbox",
        "oversized_value_bbox",
        "extra_competing_value",
        "oversized_sparse_span",
        "empty_value",
        "merged_value",
        "malformed_span",
        "missing_span_index",
        "boolean_span",
        "fractional_span",
        "numeric_string_span",
        "malformed_row_band_index",
        "duplicate_column_band_index",
        "competing_header_cell",
    ],
)
def test_exact_merged_account_geometry_fails_closed_on_physical_ambiguity(
    defect: str,
) -> None:
    labels = ("管理机构", "账户标识", "开立日期")
    table = _exact_merged_geometry_table(
        f"bad-{defect}",
        labels,
        ("上海汽车集团财务有限责任公司", "N10252900H00013539300", "2017.12.15"),
    )
    rows = table.metadata["raw_rows"]
    geometry = table.metadata["geometry"]
    if defect == "shifted_value_bbox":
        geometry["cell_bboxes"][1][1] = [110.0, 120.0, 210.0, 140.0]
    elif defect == "oversized_value_bbox":
        geometry["cell_bboxes"][1][1] = [100.0, 120.0, 225.0, 140.0]
    elif defect == "extra_competing_value":
        rows[0].append("")
        rows[1].append("2024.01.02")
        geometry["cell_bboxes"][0].append(None)
        geometry["cell_bboxes"][0][0] = [0.0, 100.0, 400.0, 120.0]
        geometry["cell_bboxes"][1].append([300.0, 120.0, 400.0, 140.0])
        geometry["cell_geometry_status"][0].append("derived")
        geometry["cell_geometry_status"][1].append("exact")
        geometry["cell_evidence_ids"][0].append([])
        geometry["cell_evidence_ids"][1].append(["extra"])
        geometry["cell_spans"][0]["col_span"] = 4
        geometry["cell_spans"][0]["bbox"] = [0.0, 100.0, 400.0, 120.0]
        geometry["col_bands"].append({"index": 3, "x0": 300.0, "x1": 400.0})
    elif defect == "oversized_sparse_span":
        rows[0].append("")
        rows[1].append(rows[1][2])
        rows[1][2] = ""
        geometry["cell_bboxes"][0].append(None)
        geometry["cell_bboxes"][0][0] = [0.0, 100.0, 400.0, 120.0]
        geometry["cell_bboxes"][1].append([300.0, 120.0, 400.0, 140.0])
        geometry["cell_geometry_status"][0].append("derived")
        geometry["cell_geometry_status"][1].append("exact")
        geometry["cell_evidence_ids"][0].append([])
        geometry["cell_evidence_ids"][1].append(["migrated"])
        geometry["cell_spans"][0]["col_span"] = 4
        geometry["cell_spans"][0]["bbox"] = [0.0, 100.0, 400.0, 120.0]
        geometry["col_bands"].append({"index": 3, "x0": 300.0, "x1": 400.0})
    elif defect == "empty_value":
        rows[1][1] = ""
    elif defect == "merged_value":
        geometry["cell_spans"].append(
            {
                "row": 1,
                "col": 1,
                "row_span": 1,
                "col_span": 2,
                "bbox": [100.0, 120.0, 300.0, 140.0],
            }
        )
    elif defect == "malformed_span":
        geometry["cell_spans"][0]["row"] = "not-an-index"
    elif defect == "missing_span_index":
        geometry["cell_spans"][0].pop("row")
    elif defect == "boolean_span":
        geometry["cell_spans"][0]["row_span"] = True
    elif defect == "fractional_span":
        geometry["cell_spans"][0]["col_span"] = 3.5
    elif defect == "numeric_string_span":
        geometry["cell_spans"][0]["col"] = "0"
    elif defect == "malformed_row_band_index":
        geometry["row_bands"][0]["index"] = "not-an-index"
    elif defect == "duplicate_column_band_index":
        geometry["col_bands"][1]["index"] = 0
    elif defect == "competing_header_cell":
        rows[0][1] = "到期日期"
        geometry["cell_bboxes"][0][1] = [100.0, 100.0, 200.0, 120.0]
        geometry["cell_geometry_status"][0][1] = "exact"
        geometry["cell_evidence_ids"][0][1] = ["competing-header"]

    status, values = native_extraction._account_merged_header_geometry_values(
        table,
        rows,
        header_row=0,
        header_column=0,
        labels=frozenset(labels),
    )

    assert status == "rejected"
    assert values == []
    result = SimpleNamespace(_personal_detail_extraction_issues=[])
    account = {"account_id": f"account:{defect}", "canonical_raw": {}}
    native_extraction._apply_collapsed_account_clusters(
        result,
        account,
        rows,
        page=_page(1, [table]),
        table=table,
        physical_row_indices=None,
    )
    assert not set(expected_field for expected_field in (
        "management_institution",
        "account_identifier",
        "open_date",
    ) if expected_field in account)
    assert any(
        issue.get("issue_code") == "candidate_b_account_cluster_field_unresolved"
        for issue in result._personal_detail_extraction_issues
    )


def _exact_interval_anchor(
    *,
    account_type: str = "revolving_loan_subaccount",
    ordinal_status: str = "printed_unique",
) -> dict[str, object]:
    return {
        "account_id": f"anchor:{account_type}:1",
        "account_type": account_type,
        "account_family_quality": "exact",
        "_printed_ordinal_status": ordinal_status,
        "category_sequence": 1,
        "source": "candidate_b_account_anchor",
        "page": 11,
        "bbox": [20.0, 100.0, 250.0, 112.0],
        "_canonical_segment": {
            "ownership_basis": "printed_anchor_to_next_anchor",
            "pages": [{"logical_page": 11, "min_y": 100.0, "max_y": 300.0}],
        },
        "source_refs": [
            {"logical_page": 11, "bbox": [20.0, 100.0, 250.0, 112.0]}
        ],
    }


def _owned_native_table(
    observation: str,
    *,
    account_type: str,
    basis: str,
    top: float = 120.0,
    pending_anchor: str = "",
) -> dict[str, object]:
    return {
        "account_id": observation,
        "_table_observation_id": observation,
        "account_type": account_type,
        "_table_account_family_basis": basis,
        "source": "native_detail_account_table",
        "_pending_anchor_account_id": pending_anchor,
        "source_refs": [
            {
                "logical_page": 11,
                "table_id": observation,
                "bbox": [20.0, top, 580.0, top + 80.0],
            }
        ],
    }


@pytest.mark.parametrize(
    ("anchor_type", "table_type", "basis"),
    [
        (
            "revolving_loan_subaccount",
            "non_revolving_loan",
            "non_revolving_table_signature",
        ),
        (
            "revolving_loan_account",
            "revolving_loan_subaccount",
            "shared_revolving_credit_limit_signature",
        ),
    ],
)
def test_exact_printed_revolving_anchor_resolves_one_owned_native_signature(
    anchor_type: str,
    table_type: str,
    basis: str,
) -> None:
    anchor = _exact_interval_anchor(account_type=anchor_type)
    table = _owned_native_table(
        "table:owned",
        account_type=table_type,
        basis=basis,
    )

    matches = native_extraction._match_account_table_observations([anchor], [table])
    native_extraction._resolve_owned_revolving_table_families(
        [anchor],
        [table],
        matches,
    )

    assert matches == {0: 0}
    assert table["account_type"] == anchor_type
    assert table["_table_account_type_candidate"] == table_type
    assert table["_table_account_family_resolution"] == (
        "exact_printed_anchor_unique_native_signature_interval"
    )


@pytest.mark.parametrize("reverse_registration", [False, True])
@pytest.mark.parametrize("alias_precedes_exact", [False, True])
def test_exact_family_table_wins_over_interval_alias_in_every_permutation(
    reverse_registration: bool,
    alias_precedes_exact: bool,
) -> None:
    anchor = _exact_interval_anchor(account_type="revolving_loan_account")
    alias_top, exact_top = (
        (120.0, 210.0) if alias_precedes_exact else (210.0, 120.0)
    )
    alias = _owned_native_table(
        "table:positional-alias",
        account_type="non_revolving_loan",
        basis="non_revolving_table_signature",
        top=alias_top,
    )
    exact = _owned_native_table(
        "table:exact-family",
        account_type="revolving_loan_account",
        basis="revolving_table_phase_carry",
        top=exact_top,
    )
    tables = [alias, exact]
    if reverse_registration:
        tables.reverse()

    matches = native_extraction._match_account_table_observations([anchor], tables)

    assert set(matches) == {0}
    matched_table = tables[matches[0]]
    assert matched_table["_table_observation_id"] == "table:exact-family"
    assert matched_table["account_type"] == "revolving_loan_account"


@pytest.mark.parametrize(
    "defect",
    [
        "wrong_interval",
        "duplicate_candidate",
        "wrong_signature",
        "headerless",
        "unreadable_anchor",
        "non_revolving_anchor",
        "strong_identity_conflict",
    ],
)
def test_interval_family_resolution_rejects_every_unbounded_variant(
    defect: str,
) -> None:
    anchor = _exact_interval_anchor(
        account_type=(
            "non_revolving_loan"
            if defect == "non_revolving_anchor"
            else "revolving_loan_subaccount"
        ),
        ordinal_status="printed_unreadable" if defect == "unreadable_anchor" else "printed_unique",
    )
    table = _owned_native_table(
        "table:candidate",
        account_type=(
            "revolving_loan_subaccount"
            if defect == "non_revolving_anchor"
            else "non_revolving_loan"
        ),
        basis=(
            "shared_revolving_credit_limit_signature"
            if defect == "non_revolving_anchor"
            else "revolving_table_phase_carry"
            if defect == "wrong_signature"
            else "non_revolving_table_signature"
        ),
        top=320.0 if defect == "wrong_interval" else 120.0,
        pending_anchor=str(anchor["account_id"]) if defect == "headerless" else "",
    )
    tables = [table]
    if defect == "strong_identity_conflict":
        anchor["account_identifier"] = "A12345678B1234567890123456"
        table["account_identifier"] = "C12345678D1234567890123456"
    if defect == "duplicate_candidate":
        tables.append(
            _owned_native_table(
                "table:competitor",
                account_type="non_revolving_loan",
                basis="non_revolving_table_signature",
                top=210.0,
            )
        )

    matches = native_extraction._match_account_table_observations([anchor], tables)

    assert matches == {}


def _spread_page(
    logical_page: int,
    source_page: int,
    segment_index: int,
    tables: list[SimpleNamespace],
) -> SimpleNamespace:
    source_offset = 600.0 * segment_index
    return SimpleNamespace(
        page_number=logical_page,
        source_page_number=source_page,
        width=600.0,
        height=800.0,
        tables=tables,
        texts=[],
        coordinate_transform={
            "source_page_number": source_page,
            "display_width": 600.0,
            "display_height": 800.0,
            "matrix": [
                [1.0, 0.0, -source_offset],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            "inverse_matrix": [
                [1.0, 0.0, source_offset],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            "source_crop_bbox": [source_offset, 0.0, source_offset + 600.0, 800.0],
            "decomposition": {
                "kind": "two_page_spread",
                "segment_index": segment_index,
                "selected_rotation": 0,
                "confidence": 0.99,
            },
        },
    )


def _single_native_page(
    logical_page: int,
    tables: list[SimpleNamespace],
) -> SimpleNamespace:
    return SimpleNamespace(
        page_number=logical_page,
        source_page_number=logical_page,
        width=600.0,
        height=800.0,
        tables=tables,
        texts=[],
        coordinate_transform={
            "source_page_number": logical_page,
            "display_width": 600.0,
            "display_height": 800.0,
            "matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "inverse_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "source_crop_bbox": [0.0, 0.0, 600.0, 800.0],
            "decomposition": {
                "kind": "none",
                "selected_rotation": 0,
                "confidence": 1.0,
            },
        },
    )


def _evidence_bundle(
    logical_page: int,
    source_page: int,
    lines: list[tuple[str, list[float]]],
) -> dict[str, object]:
    return {
        "page": logical_page,
        "source_page_number": source_page,
        "local_structure_evidence": {
            "page": logical_page,
            "source_page": source_page,
            "page_width": 600.0,
            "page_height": 800.0,
            "lines": [
                {"text": text, "bbox": bbox, "evidence_ids": [f"p{logical_page}:{index}"]}
                for index, (text, bbox) in enumerate(lines)
            ],
        },
    }


def test_conflicting_full_footer_total_never_becomes_authoritative_order() -> None:
    pages = [
        SimpleNamespace(
            page_number=page,
            source_page_number=page,
            width=600.0,
            height=800.0,
            tables=[],
            texts=[
                SimpleNamespace(
                    content=f"第 {page} 页，共 {2 if page == 2 else 4} 页",
                    bbox=[220.0, 760.0, 380.0, 785.0],
                )
            ],
        )
        for page in range(1, 5)
    ]
    result = SimpleNamespace(
        pages=pages,
        entities=SimpleNamespace(domain_specific={}),
    )

    _order, resolution = _printed_reading_order_resolution(result)

    assert resolution["resolved"] is False
    assert resolution["authoritative"] is False
    assert resolution["reason"] == "printed_total_missing_or_conflicting"


def test_full_context_adoption_preserves_ye_printed_order_and_cards_four_to_nine() -> None:
    card_ids = {
        4: "B10911000H000115603050013394541",
        5: "B11313900H000115603090424251222",
        6: "D10123910H000115604050032149",
        7: "B10411000H000115602800002159651279117266",
        8: "B10611000H00016226880219191368607",
        9: "B11911000H000115661000042356833",
    }
    card_tables_17 = [
        _table(
            f"pt_17_{sequence - 4}",
            [_CARD_HEADER, _card_values(card_ids[sequence])],
            top=top,
        )
        for sequence, top in ((4, 60.0), (5, 260.0), (6, 460.0))
    ]
    card7_with_card8_header = _table(
        "pt_23_1",
        [_CARD_HEADER, _card_values(card_ids[7]), _CARD_HEADER],
        top=60.0,
    )
    card8_values = _table("pt_24_0", [_card_values(card_ids[8])], top=10.0)
    card9 = _table(
        "pt_24_1",
        [_CARD_HEADER, _card_values(card_ids[9])],
        top=300.0,
    )
    pages = [
        _spread_page(17, 9, 0, card_tables_17),
        _spread_page(18, 9, 1, []),
        _spread_page(19, 10, 0, []),
        _spread_page(20, 10, 1, []),
        _spread_page(21, 11, 0, []),
        _spread_page(22, 11, 1, []),
        _spread_page(23, 12, 0, [card7_with_card8_header]),
        _spread_page(24, 12, 1, [card8_values, card9]),
    ]
    bundles = [
        _evidence_bundle(
            17,
            9,
            [
                ("账户 4", [20.0, 40.0, 180.0, 55.0]),
                ("账户 5", [20.0, 240.0, 180.0, 255.0]),
                ("账户 6", [20.0, 440.0, 180.0, 455.0]),
                ("第 19 页", [240.0, 760.0, 360.0, 785.0]),
            ],
        ),
        _evidence_bundle(18, 9, []),
        _evidence_bundle(
            19,
            10,
            [
                ("（四）贷记卡账户", [20.0, 20.0, 220.0, 35.0]),
                ("第 17 页，共 24 页", [220.0, 760.0, 380.0, 785.0]),
            ],
        ),
        _evidence_bundle(
            20,
            10,
            [("第 18 页，共 24 页", [220.0, 760.0, 380.0, 785.0])],
        ),
        _evidence_bundle(
            21,
            11,
            [("第 23 页，共 24 页", [220.0, 760.0, 380.0, 785.0])],
        ),
        _evidence_bundle(22, 11, []),
        _evidence_bundle(
            23,
            12,
            [
                ("账户 7", [20.0, 40.0, 180.0, 55.0]),
                ("账户 8", [20.0, 320.0, 180.0, 335.0]),
                ("第 21 页，共 24 页", [220.0, 760.0, 380.0, 785.0]),
            ],
        ),
        _evidence_bundle(
            24,
            12,
            [
                ("账户 9", [20.0, 280.0, 180.0, 295.0]),
                ("第 22 页，共 24 页", [220.0, 760.0, 380.0, 785.0]),
            ],
        ),
    ]
    result = SimpleNamespace(
        pages=pages,
        entities=SimpleNamespace(domain_specific={"_page_evidence_bundles": bundles}),
    )

    context = build_personal_detail_extraction_context(result)
    expected_order = {19: 1, 20: 2, 17: 3, 18: 4, 23: 5, 24: 6, 21: 7, 22: 8}
    assert context.reading_order_by_logical == expected_order
    assert context.reading_order_resolution["resolved"] is True
    _canonical_pages = context.pages
    assert context.reading_order_by_logical == expected_order
    assert context.reading_order_resolution["printed_page_by_logical"] == {
        17: 19,
        18: 20,
        19: 17,
        20: 18,
        21: 23,
        22: 24,
        23: 21,
        24: 22,
    }

    accounts, _repayments, _events = context.account_collections()
    by_sequence = {
        int(account["category_sequence"]): account
        for account in accounts
        if account.get("account_type") == "credit_card"
        and account.get("category_sequence") in card_ids
    }

    assert sorted(by_sequence) == list(card_ids)
    for sequence, identifier in card_ids.items():
        account = by_sequence[sequence]
        assert account["account_identifier"] == identifier
        assert account["management_institution"] == "招商银行股份有限公司"
        assert account["open_date"] == "2020-01-02"


def test_full_context_attaches_pt6_and_exact_r1_r2_native_tables() -> None:
    pt6 = _exact_merged_geometry_table(
        "pt_6_2",
        ("管理机构", "账户标识", "开立日期"),
        ("上海汽车集团财务有限责任公司", "N10252900H00013539300", "2017.12.15"),
        top=60.0,
        interleaved=True,
    )
    r1_identifier = "D10053310H00012022052901021012089466554314"
    r2_identifier = "D10123910H000115604050032149"
    revolving_header = [
        "管理机构",
        "账户标识",
        "开立日期",
        "账户授信额度",
        "币种",
        "业务种类",
        "担保方式",
    ]
    r1 = _table(
        "pt_11_3",
        [
            revolving_header,
            [
                "示例银行股份有限公司",
                r1_identifier,
                "2019.01.02",
                "100000",
                "人民币元",
                "个人消费贷款",
                "信用",
            ],
        ],
        top=60.0,
    )
    r2 = _table(
        "pt_14_2",
        [
            revolving_header,
            [
                "示例银行股份有限公司",
                r2_identifier,
                "2020.03.04",
                "200000",
                "人民币元",
                "个人消费贷款",
                "抵押",
            ],
        ],
        top=60.0,
    )
    pages = [
        _single_native_page(6, [pt6]),
        _single_native_page(11, [r1]),
        _single_native_page(14, [r2]),
    ]
    bundles = [
        _evidence_bundle(
            6,
            6,
            [
                ("（一）非循环贷账户", [20.0, 20.0, 240.0, 35.0]),
                ("账户 6", [20.0, 40.0, 180.0, 55.0]),
                ("第 1 页，共 3 页", [220.0, 760.0, 380.0, 785.0]),
            ],
        ),
        _evidence_bundle(
            11,
            11,
            [
                ("（二）循环贷账户一", [20.0, 20.0, 240.0, 35.0]),
                ("账户 1", [20.0, 40.0, 180.0, 55.0]),
                ("第 2 页，共 3 页", [220.0, 760.0, 380.0, 785.0]),
            ],
        ),
        _evidence_bundle(
            14,
            14,
            [
                ("（三）循环贷账户二", [20.0, 20.0, 240.0, 35.0]),
                ("账户 1", [20.0, 40.0, 180.0, 55.0]),
                ("第 3 页，共 3 页", [220.0, 760.0, 380.0, 785.0]),
            ],
        ),
    ]
    result = SimpleNamespace(
        pages=pages,
        entities=SimpleNamespace(domain_specific={"_page_evidence_bundles": bundles}),
    )

    context = build_personal_detail_extraction_context(result)
    _canonical_pages = context.pages
    accounts, _repayments, _events = context.account_collections()
    by_family = {account["account_type"]: account for account in accounts}

    assert by_family["non_revolving_loan"]["account_identifier"] == "N10252900H00013539300"
    assert by_family["non_revolving_loan"]["management_institution"] == (
        "上海汽车集团财务有限责任公司"
    )
    assert by_family["non_revolving_loan"]["open_date"] == "2017-12-15"
    assert by_family["revolving_loan_subaccount"]["account_identifier"] == r1_identifier
    assert by_family["revolving_loan_subaccount"]["credit_limit"] == 100000
    assert by_family["revolving_loan_account"]["account_identifier"] == r2_identifier
    assert by_family["revolving_loan_account"]["credit_limit"] == 200000
    assert {
        ref.get("table_id")
        for account in by_family.values()
        for ref in account.get("source_refs") or ()
    } >= {"pt_6_2", "pt_11_3", "pt_14_2"}


_GEOMETRIC_LOAN_BASE_ROWS = [
    [
        "\u7ba1\u7406\u673a\u6784",
        "\u8d26\u6237\u6807\u8bc6",
        "\u5f00\u7acb\u65e5\u671f",
        "\u5230\u671f\u65e5\u671f",
        "\u501f\u6b3e\u91d1\u989d",
        "\u8d26\u6237\u5e01\u79cd",
    ],
    [
        "\u793a\u4f8b\u94f6\u884c",
        "J10158510H000110000000640557",
        "2021.01.12",
        "2024.01.12",
        "140000",
        "\u4eba\u6c11\u5e01\u5143",
    ],
]
_GEOMETRIC_LOAN_DETAIL_ROWS = [
    [
        "\u4e94\u7ea7\u5206\u7c7b",
        "\u8d26\u6237\u72b6\u6001",
        "\u4f59\u989d",
        "\u5269\u4f59\u8fd8\u6b3e\u671f\u6570",
    ],
    ["\u6b63\u5e38", "\u6b63\u5e38", "55046", "13"],
]


def _geometric_continuation_context(
    *,
    left_logical_page: int,
    right_logical_page: int,
    page_order: list[int],
    reading_order: dict[int, int] | None,
    resolution: dict[str, bool] | None,
) -> SimpleNamespace:
    pages_by_logical = {
        left_logical_page: _page(
            left_logical_page,
            [_table("geometric-base", _GEOMETRIC_LOAN_BASE_ROWS, top=100.0)],
        ),
        right_logical_page: _page(
            right_logical_page,
            [
                _table(
                    "geometric-detail",
                    _GEOMETRIC_LOAN_DETAIL_ROWS,
                    top=20.0,
                ),
                _table(
                    "geometric-next-base",
                    _GEOMETRIC_LOAN_BASE_ROWS,
                    top=300.0,
                ),
            ],
        ),
    }
    for logical_page in page_order:
        pages_by_logical.setdefault(logical_page, _page(logical_page, []))
    return SimpleNamespace(
        pages=[pages_by_logical[logical_page] for logical_page in page_order],
        reading_order_by_logical=reading_order,
        reading_order_resolution=resolution,
        corrected_evidence_pages=lambda: [],
        tables_continue=lambda *_args: None,
    )


def test_geometric_account_continuation_requires_immediate_registered_page_edge() -> None:
    context = _geometric_continuation_context(
        left_logical_page=1,
        right_logical_page=3,
        page_order=[1, 2, 3],
        reading_order={1: 1, 2: 2, 3: 3},
        resolution={"resolved": True, "authoritative": True},
    )

    accounts, _repayments, _events = native_extraction._extract_table_accounts(
        context
    )

    assert "balance" not in accounts[0]
    assert {ref.get("table_id") for ref in accounts[0]["source_refs"]} == {
        "geometric-base"
    }


def test_geometric_account_continuation_accepts_authoritative_adjacency() -> None:
    context = _geometric_continuation_context(
        left_logical_page=1,
        right_logical_page=2,
        page_order=[1, 2],
        reading_order={1: 1, 2: 2},
        resolution={"resolved": True, "authoritative": True},
    )

    accounts, _repayments, _events = native_extraction._extract_table_accounts(
        context
    )

    assert accounts[0]["balance"] == 55046
    assert {ref.get("table_id") for ref in accounts[0]["source_refs"]} == {
        "geometric-base",
        "geometric-detail",
    }


def test_geometric_account_continuation_accepts_reordered_authoritative_edge() -> None:
    context = _geometric_continuation_context(
        left_logical_page=20,
        right_logical_page=17,
        page_order=[17, 20],
        reading_order={20: 1, 17: 2},
        resolution={"resolved": True, "authoritative": True},
    )

    accounts, _repayments, _events = native_extraction._extract_table_accounts(
        context
    )

    assert accounts[0]["balance"] == 55046
    assert accounts[0]["source_refs"][0]["logical_page"] == 20


@pytest.mark.parametrize(
    ("reading_order", "resolution"),
    [
        ({1: 1, 2: 2}, {"resolved": False, "authoritative": False}),
        ({1: 1, 2: 2}, {"resolved": True, "authoritative": False}),
        ({1: 1, 2: 1}, {"resolved": True, "authoritative": True}),
        (None, {"resolved": True, "authoritative": True}),
        (None, None),
    ],
)
def test_geometric_account_continuation_rejects_unproven_order(
    reading_order: dict[int, int] | None,
    resolution: dict[str, bool] | None,
) -> None:
    context = _geometric_continuation_context(
        left_logical_page=1,
        right_logical_page=2,
        page_order=[1, 2],
        reading_order=reading_order,
        resolution=resolution,
    )

    accounts, _repayments, _events = native_extraction._extract_table_accounts(
        context
    )

    assert "balance" not in accounts[0]


_REVOLVING_PHASE_HEADER = [
    "\u7ba1\u7406\u673a\u6784",
    "\u8d26\u6237\u6807\u8bc6",
    "\u5f00\u7acb\u65e5\u671f",
    "\u8d26\u6237\u6388\u4fe1\u989d\u5ea6",
    "\u5e01\u79cd",
    "\u4e1a\u52a1\u79cd\u7c7b",
    "\u62c5\u4fdd\u65b9\u5f0f",
]


def _phase_context(*, adjacent: bool) -> SimpleNamespace:
    second_page = 2 if adjacent else 3
    first = _table(
        "phase-r1",
        [
            _REVOLVING_PHASE_HEADER,
            [
                "\u793a\u4f8b\u94f6\u884c",
                "D10053310H00012022052901021012089466554314",
                "2019.01.02",
                "100000",
                "\u4eba\u6c11\u5e01\u5143",
                "\u4e2a\u4eba\u6d88\u8d39\u8d37\u6b3e",
                "\u4fe1\u7528",
            ],
        ],
    )
    second_rows = [
        _GEOMETRIC_LOAN_BASE_ROWS[0],
        [
            "\u793a\u4f8b\u94f6\u884c",
            "D10123910H000115604050032149",
            "2020.03.04",
            "2025.03.04",
            "200000",
            "\u4eba\u6c11\u5e01\u5143",
        ],
    ]
    pages = [_page(1, [first])]
    if not adjacent:
        pages.append(_page(2, []))
    pages.append(_page(second_page, [_table("phase-next", second_rows)]))
    return SimpleNamespace(
        pages=pages,
        reading_order_by_logical={page.page_number: index for index, page in enumerate(pages, 1)},
        reading_order_resolution={"resolved": True, "authoritative": True},
        corrected_evidence_pages=lambda: [],
        tables_continue=lambda *_args: None,
    )


def test_revolving_phase_does_not_cross_blank_registered_page() -> None:
    context = _phase_context(adjacent=False)

    accounts, _repayments, _events = native_extraction._extract_table_accounts(
        context
    )

    assert [account["account_type"] for account in accounts] == [
        "revolving_loan_subaccount",
        "non_revolving_loan",
    ]
    assert accounts[1]["_table_account_family_basis"] == (
        "non_revolving_table_signature"
    )


def test_revolving_phase_accepts_immediate_authoritative_page_edge() -> None:
    context = _phase_context(adjacent=True)

    accounts, _repayments, _events = native_extraction._extract_table_accounts(
        context
    )

    assert [account["account_type"] for account in accounts] == [
        "revolving_loan_subaccount",
        "revolving_loan_account",
    ]
    assert accounts[1]["_table_account_family_basis"] == "revolving_table_phase_carry"


def _family_state_context(*, gap: bool) -> SimpleNamespace:
    second_page = 3 if gap else 2
    evidence = [
        {
            "page": 1,
            "source_page": 1,
            "lines": [
                {
                    "text": "\uff08\u4e00\uff09\u975e\u5faa\u73af\u8d37\u8d26\u6237",
                    "bbox": [10, 10, 220, 25],
                },
                {"text": "\u8d26\u6237 1\uff1a", "bbox": [10, 40, 120, 55]},
            ],
        },
        *(
            [{"page": 2, "source_page": 2, "lines": []}]
            if gap
            else []
        ),
        {
            "page": second_page,
            "source_page": second_page,
            "lines": [
                {"text": "\u8d26\u6237 2\uff1a", "bbox": [10, 40, 120, 55]}
            ],
        },
    ]
    pages = [_page(1, [])]
    if gap:
        pages.append(_page(2, []))
    pages.append(_page(second_page, []))
    return SimpleNamespace(
        pages=pages,
        reading_order_by_logical={page.page_number: index for index, page in enumerate(pages, 1)},
        reading_order_resolution={"resolved": True, "authoritative": True},
        corrected_evidence_pages=lambda: evidence,
        allows_scanned_line_transition=lambda *_args: False,
    )


def test_anchor_and_ledger_family_state_stop_at_blank_registered_page() -> None:
    context = _family_state_context(gap=True)

    skeletons = native_extraction._account_anchor_skeletons(context)
    ledger = native_extraction._source_completeness_ledger(context)

    assert [row["category_sequence"] for row in skeletons] == [1]
    assert ledger["account_family_endpoints"] == {"non_revolving_loan": 1}
    assert ledger["credit_accounts"] == 1


def test_anchor_and_ledger_family_state_accept_immediate_authoritative_edge() -> None:
    context = _family_state_context(gap=False)

    skeletons = native_extraction._account_anchor_skeletons(context)
    ledger = native_extraction._source_completeness_ledger(context)

    assert [row["category_sequence"] for row in skeletons] == [1, 2]
    assert ledger["account_family_endpoints"] == {"non_revolving_loan": 2}
    assert ledger["credit_accounts"] == 2


def test_page_local_account_table_family_uses_canonical_closed_signatures() -> None:
    card = _table(
        "canonical-card",
        [_CARD_HEADER, _card_values("B10911000H000115603050013394541")],
    )
    revolving = _table(
        "shared-revolving",
        [
            _REVOLVING_PHASE_HEADER,
            [
                "\u793a\u4f8b\u94f6\u884c",
                "D10053310H00012022052901021012089466554314",
                "2019.01.02",
                "100000",
                "\u4eba\u6c11\u5e01\u5143",
                "\u4e2a\u4eba\u6d88\u8d39\u8d37\u6b3e",
                "\u4fe1\u7528",
            ],
        ],
    )
    card_context = SimpleNamespace(pages=[_page(1, [card])])
    revolving_context = SimpleNamespace(pages=[_page(1, [revolving])])

    assert native_extraction._account_page_table_evidence(card_context, 1) == (
        ("credit_card", "exact"),
        False,
    )
    assert native_extraction._account_page_table_evidence(revolving_context, 1) == (
        None,
        True,
    )


def test_page_local_account_table_family_rejects_conflicting_exact_signatures() -> None:
    card = _table(
        "canonical-card",
        [_CARD_HEADER, _card_values("B10911000H000115603050013394541")],
    )
    loan = _table("canonical-loan", _GEOMETRIC_LOAN_BASE_ROWS)
    context = SimpleNamespace(pages=[_page(1, [card, loan])])

    assert native_extraction._account_page_table_evidence(context, 1) == (None, False)


def test_conflicting_table_families_cannot_refresh_anchor_or_ledger_state() -> None:
    context = _family_state_context(gap=True)
    card = _table(
        "conflicting-card",
        [_CARD_HEADER, _card_values("B10911000H000115603050013394541")],
    )
    loan = _table("conflicting-loan", _GEOMETRIC_LOAN_BASE_ROWS)
    context.pages[1].tables = [card, loan]

    skeletons = native_extraction._account_anchor_skeletons(context)
    ledger = native_extraction._source_completeness_ledger(context)

    assert [row["category_sequence"] for row in skeletons] == [1]
    assert ledger["account_family_endpoints"] == {"non_revolving_loan": 1}
    assert ledger["credit_accounts"] == 1


@pytest.mark.parametrize(
    ("heading", "expected_sequences"),
    [
        (
            "\uff08\u4e8c\uff09\u5faa\u73af\u8d37\u8d26\u6237\u4e00",
            [1, 2],
        ),
        ("\uff08\u4e00\uff09\u975e\u5faa\u73af\u8d37\u8d26\u6237", [1]),
    ],
)
def test_shared_revolving_table_refreshes_only_compatible_family_state(
    heading: str,
    expected_sequences: list[int],
) -> None:
    shared_revolving = _table(
        "shared-revolving-carry",
        [
            _REVOLVING_PHASE_HEADER,
            [
                "\u793a\u4f8b\u94f6\u884c",
                "D10053310H00012022052901021012089466554314",
                "2019.01.02",
                "100000",
                "\u4eba\u6c11\u5e01\u5143",
                "\u4e2a\u4eba\u6d88\u8d39\u8d37\u6b3e",
                "\u4fe1\u7528",
            ],
        ],
    )
    evidence = [
        {
            "page": 1,
            "source_page": 1,
            "lines": [
                {"text": heading, "bbox": [10, 10, 220, 25]},
                {"text": "\u8d26\u6237 1\uff1a", "bbox": [10, 40, 120, 55]},
            ],
        },
        {"page": 2, "source_page": 2, "lines": []},
        {
            "page": 3,
            "source_page": 3,
            "lines": [
                {"text": "\u8d26\u6237 2\uff1a", "bbox": [10, 40, 120, 55]}
            ],
        },
    ]
    context = SimpleNamespace(
        pages=[_page(1, []), _page(2, [shared_revolving]), _page(3, [])],
        reading_order_by_logical={1: 1, 2: 2, 3: 3},
        reading_order_resolution={"resolved": True, "authoritative": True},
        corrected_evidence_pages=lambda: evidence,
        allows_scanned_line_transition=lambda *_args: False,
    )

    skeletons = native_extraction._account_anchor_skeletons(context)
    ledger = native_extraction._source_completeness_ledger(context)

    assert [row["category_sequence"] for row in skeletons] == expected_sequences
    account_type = native_extraction._account_family_from_heading(
        native_extraction._compact(heading)
    )[0]
    assert ledger["account_family_endpoints"] == {
        account_type: max(expected_sequences)
    }


def _exact_two_cell_card_table(*, trailing_residue: str = "爱") -> SimpleNamespace:
    rows = [
        [
            "发卡机构 账户标识 开立日期 账户授信额度",
            "共享授信额度 币种 业务种类 担保方式",
        ],
        [
            "D10123910H 福建海峡银行 00011560405 2017.02.04 30,000 "
            "股份有限公司 0032149",
            f"人民币元 贷记卡 信用/无担保 {trailing_residue}".strip(),
        ],
    ]
    geometry = {
        "cell_bboxes": [
            [[51.5, 386.5, 226.0, 401.0], [226.0, 386.5, 401.5, 401.0]],
            [[51.5, 401.0, 226.0, 426.5], [226.0, 401.0, 401.5, 426.5]],
        ],
        "cell_geometry_status": [["exact", "exact"], ["exact", "exact"]],
        "cell_evidence_ids": [
            [["card6:h0"], ["card6:h1"]],
            [["card6:v0"], ["card6:v1"]],
        ],
        "cell_spans": [],
        "row_bands": [
            {"index": 0, "y0": 386.5, "y1": 401.0},
            {"index": 1, "y0": 401.0, "y1": 426.5},
        ],
        "col_bands": [
            {"index": 0, "x0": 51.5, "x1": 226.0},
            {"index": 1, "x0": 226.0, "x1": 401.5},
        ],
    }
    return SimpleNamespace(
        table_id="pt_18_1",
        metadata={"raw_rows": rows, "geometry": geometry},
        headers=[],
        rows=[],
        bbox=[51.5, 386.5, 401.5, 426.5],
        confidence=0.99,
    )


def test_exact_two_cell_card_cluster_recovers_card6_without_xau() -> None:
    table = _exact_two_cell_card_table()
    evidence = [
        {
            "page": 18,
            "source_page": 9,
            "lines": [
                {
                    "text": "（四）贷记卡账户",
                    "bbox": [51.5, 350.0, 180.0, 365.0],
                    "evidence_ids": ["card-family"],
                },
                {
                    "text": "账户 6：",
                    "bbox": [51.5, 368.0, 150.0, 382.0],
                    "evidence_ids": ["card6-anchor"],
                },
            ],
        }
    ]
    context = SimpleNamespace(
        pages=[_page(18, [table])],
        reading_order_by_logical={18: 1},
        reading_order_resolution={"resolved": True, "authoritative": True},
        corrected_evidence_pages=lambda: evidence,
        allows_scanned_line_transition=lambda *_args: False,
        tables_continue=lambda *_args: None,
    )

    accounts, _repayments, _events = native_extraction._extract_accounts(context)
    card6 = next(
        account
        for account in accounts
        if account.get("account_type") == "credit_card"
        and account.get("category_sequence") == 6
    )

    assert card6["management_institution"] == "福建海峡银行股份有限公司"
    assert card6["account_identifier"] == "D10123910H000115604050032149"
    assert card6["open_date"] == "2017-02-04"
    assert card6["credit_limit"] == 30000
    assert card6["currency"] == "CNY"
    assert card6["account_currency"] == "CNY"
    assert card6["business_type"] == "贷记卡"
    assert card6["guarantee_type"] == "信用/无担保"
    assert card6.get("shared_credit_limit") is None
    assert "XAU" not in str(card6)
    assert any(
        issue.get("issue_code") == "candidate_b_account_cluster_field_unresolved"
        and issue.get("field_name") == "shared_credit_limit"
        and issue.get("source_refs", [{}])[0].get("row") == 1
        and issue.get("source_refs", [{}])[0].get("column") == 1
        for issue in context._personal_detail_extraction_issues
    )
    assert any(
        issue.get("issue_code") == "candidate_b_account_cluster_residue_unresolved"
        and issue.get("observed_value", {}).get("unconsumed_residue") == "爱"
        for issue in context._personal_detail_extraction_issues
    )


@pytest.mark.parametrize("defect", ["two_han_residue", "spanned_value", "derived_cell"])
def test_exact_two_cell_card_cluster_rejects_nearby_shapes(defect: str) -> None:
    table = _exact_two_cell_card_table(
        trailing_residue="爱福" if defect == "two_han_residue" else "爱"
    )
    geometry = table.metadata["geometry"]
    if defect == "spanned_value":
        geometry["cell_spans"] = [
            {"row": 1, "col": 0, "row_span": 1, "col_span": 2}
        ]
    elif defect == "derived_cell":
        geometry["cell_geometry_status"][1][1] = "derived"

    assert (
        native_extraction._exact_two_cell_card_cluster_values(
            table, table.metadata["raw_rows"]
        )
        is None
    )


def _exact_merged_r1_identity_table() -> SimpleNamespace:
    rows = [
        [
            "管理机构 账户标识 开立日期",
            "",
            "",
            "",
            "",
            "",
            "",
            "到期日期 借款金额 账户币种",
            "",
            "",
            "",
            "",
            "",
            "",
        ],
        [
            "中信银行股份有限 B10611000H0001 2022.06.22 公司福州分行 811132137961001",
            "",
            "",
            "",
            "",
            "",
            "",
            "人民币元 50,000",
            "",
            "",
            "",
            "",
            "",
            "",
        ],
    ]

    column_bands = [
        {"index": index, "x0": float(index * 25), "x1": float((index + 1) * 25)}
        for index in range(14)
    ]
    row_bands = [
        {"index": 0, "y0": 157.5, "y1": 171.5},
        {"index": 1, "y0": 171.5, "y1": 190.5},
    ]
    left_header_bbox = [0.0, 157.5, 175.0, 171.5]
    left_value_bbox = [0.0, 171.5, 175.0, 190.5]
    right_header_bbox = [175.0, 157.5, 350.0, 171.5]
    right_value_bbox = [175.0, 171.5, 350.0, 190.5]
    geometry = {
        "coordinate_system": "pdf_points_top_left",
        "cell_bboxes": [
            [left_header_bbox, *([None] * 6), right_header_bbox, *([None] * 6)],
            [left_value_bbox, *([None] * 6), right_value_bbox, *([None] * 6)],
        ],
        "cell_geometry_status": [
            ["exact", *(["derived"] * 6), "exact", *(["derived"] * 6)],
            ["exact", *(["derived"] * 6), "exact", *(["derived"] * 6)],
        ],
        "cell_evidence_ids": [
            [["r1:h0"], *([[]] * 6), ["r1:h7"], *([[]] * 6)],
            [
                ["r1:v:institution", "r1:v:identifier", "r1:v:date"],
                *([[]] * 6),
                ["r1:v:currency", "r1:v:amount"],
                *([[]] * 6),
            ],
        ],
        "cell_spans": [
            {
                "row": 0,
                "col": 0,
                "row_span": 1,
                "col_span": 7,
                "bbox": left_header_bbox,
            },
            {"row": 1, "col": 0, "row_span": 1, "col_span": 7, "bbox": left_value_bbox},
            {
                "row": 0,
                "col": 7,
                "row_span": 1,
                "col_span": 7,
                "bbox": right_header_bbox,
            },
            {
                "row": 1,
                "col": 7,
                "row_span": 1,
                "col_span": 7,
                "bbox": right_value_bbox,
            },
        ],
        "row_bands": row_bands,
        "col_bands": column_bands,
    }
    return SimpleNamespace(
        table_id="pt_14_1",
        metadata={"raw_rows": rows, "geometry": geometry},
        headers=[],
        rows=[],
        bbox=[0.0, 157.5, 350.0, 190.5],
        confidence=0.99,
    )
def test_document_local_inquiry_repair_resolves_one_closed_noisy_pair() -> None:
    assert native_extraction._document_local_inquiry_ordinals(
        [7, None, None, 10],
        noisy_candidates=[None, (8, "suffix_noise"), None, None],
    ) == [
        (7, None),
        (8, "paired_suffix_noise"),
        (9, "paired_missing"),
        (10, None),
    ]
@pytest.mark.parametrize(
    ("raw_sequences", "noisy_candidates"),
    [
        ([1, None, None, 4], [None, None, None, None]),
        ([7, None, None, 10], [None, (9, "suffix_noise"), None, None]),
        ([7, None, None, 10], [None, None, (9, "suffix_noise"), None]),
        (
            [7, None, None, 10],
            [None, (8, "suffix_noise"), (9, "suffix_noise"), None],
        ),
        ([7, None, None, 11], [None, (8, "suffix_noise"), None, None]),
        ([None, None, 3], [(1, "suffix_noise"), None, None]),
        (
            [6, None, None, None, 10],
            [None, (7, "suffix_noise"), None, None, None],
        ),
        ([7, None, None, 10], [None, (8, "prefixed_noise"), None, None]),
        (
            [8, 7, None, None, 10],
            [None, None, (8, "suffix_noise"), None, None],
        ),
    ],
)
def test_document_local_inquiry_repair_rejects_unclosed_noisy_pairs(
    raw_sequences: list[int | None],
    noisy_candidates: list[tuple[int, str] | None],
) -> None:
    normalized = native_extraction._document_local_inquiry_ordinals(
        raw_sequences,
        noisy_candidates=noisy_candidates,
    )

    assert not any(repair == "paired_suffix_noise" for _value, repair in normalized)
@pytest.mark.parametrize("raw", ["8\u56cdX", "\u56cd8\u56cd"])
def test_bounded_inquiry_suffix_candidate_rejects_extra_edge_glyphs(raw: str) -> None:
    assert native_extraction._bounded_inquiry_sequence_noise_candidate(raw) is None
def test_exact_merged_r1_identity_cell_remains_localized_without_token_geometry() -> (
    None
):
    table = _exact_merged_r1_identity_table()
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    account = {
        "account_id": "credit_account:revolving_loan_subaccount:6",
        "account_type": "revolving_loan_subaccount",
        "canonical_raw": {},
    }

    native_extraction._apply_collapsed_account_clusters(
        context,
        account,
        table.metadata["raw_rows"],
        page=_page(14, [table]),
        table=table,
        physical_row_indices=None,
    )

    assert account.get("management_institution") in (None, "")
    assert account.get("account_identifier") in (None, "")
    assert {
        issue.get("field_name")
        for issue in context._personal_detail_extraction_issues
        if issue.get("issue_code") == "candidate_b_account_cluster_field_unresolved"
    }.issuperset({"management_institution", "account_identifier"})
def _ye_family_population_context(*, insert_gap: bool = False) -> SimpleNamespace:
    evidence: list[dict[str, object]] = []
    pages: list[SimpleNamespace] = []

    def evidence_page(page: int, lines: list[tuple[str, float]]) -> None:
        evidence.append(
            {
                "page": page,
                "source_page": page,
                "lines": [
                    {
                        "text": text,
                        "bbox": [20.0, top, 260.0, top + 12.0],
                        "evidence_ids": [f"p{page}:{index}"],
                    }
                    for index, (text, top) in enumerate(lines)
                ],
            }
        )

    nr_lines = [("（一）非循环贷账户", 10.0)] + [
        (f"账户 {sequence}：", 30.0 + sequence * 18.0)
        for sequence in range(1, 19)
    ]
    evidence_page(1, nr_lines)
    pages.append(_page(1, []))

    page11_table = _table("r1-1", _GEOMETRIC_LOAN_BASE_ROWS, top=535.0)
    pages.append(_page(11, [page11_table]))
    evidence_page(
        11,
        [("（二）循环贷账户一", 501.5), ("账户 1：", 516.5)],
    )

    continuation_pages = [(12, (2, 3)), (13, (4, 5))]
    for page_number, ordinals in continuation_pages:
        pages.append(
            _page(
                page_number,
                [
                    _table(
                        f"r1-{ordinal}",
                        _GEOMETRIC_LOAN_BASE_ROWS,
                        top=80.0 + index * 220.0,
                    )
                    for index, ordinal in enumerate(ordinals)
                ],
            )
        )
        evidence_page(
            page_number,
            [
                (f"账户 {ordinal}：", 60.0 + index * 220.0)
                for index, ordinal in enumerate(ordinals)
            ],
        )

    page14_tables = [
        _table("r1-6", _GEOMETRIC_LOAN_BASE_ROWS, top=170.0),
        *[
            _table(
                f"r2-{sequence}",
                [
                    _REVOLVING_PHASE_HEADER,
                    [
                        "示例银行",
                        f"D10053310H000120220529010210120894665543{sequence:02d}",
                        "2021.01.02",
                        "100000",
                        "人民币元",
                        "个人消费贷款",
                        "信用",
                    ],
                ],
                top=380.0 + sequence * 55.0,
            )
            for sequence in range(1, 7)
        ],
    ]
    pages.append(_page(14, page14_tables))
    evidence_page(
        14,
        [
            ("账户 6：", 148.0),
            ("（三）循环贷账户二", 349.5),
            *[
                (f"账户 {sequence}：", 365.0 + sequence * 55.0)
                for sequence in range(1, 7)
            ],
        ],
    )
    pages.append(_page(17, []))
    evidence_page(
        17,
        [("（四）贷记卡账户", 10.0)]
        + [
            (f"账户 {sequence}：", 30.0 + sequence * 18.0)
            for sequence in range(1, 13)
        ],
    )

    if insert_gap:
        pages.insert(3, _page(120, []))
        evidence.insert(
            3,
            {"page": 120, "source_page": 120, "lines": []},
        )
    reading_order = {
        page.page_number: index for index, page in enumerate(pages, 1)
    }
    return SimpleNamespace(
        pages=pages,
        reading_order_by_logical=reading_order,
        reading_order_resolution={"resolved": True, "authoritative": True},
        corrected_evidence_pages=lambda: evidence,
        allows_scanned_line_transition=lambda *_args: False,
    )


def test_generic_loan_morphology_preserves_exact_ye_family_populations() -> None:
    context = _ye_family_population_context()

    skeletons = native_extraction._account_anchor_skeletons(context)
    ledger = native_extraction._source_completeness_ledger(context)
    counts: dict[str, int] = {}
    for skeleton in skeletons:
        family = str(skeleton["account_type"])
        counts[family] = counts.get(family, 0) + 1

    assert counts == {
        "non_revolving_loan": 18,
        "revolving_loan_subaccount": 6,
        "revolving_loan_account": 6,
        "credit_card": 12,
    }
    assert ledger["account_family_endpoints"] == {
        "non_revolving_loan": 18,
        "revolving_loan_subaccount": 6,
        "revolving_loan_account": 6,
        "credit_card": 12,
    }
    assert ledger["credit_accounts"] == 42


def test_generic_loan_family_carry_stops_at_registered_blank_gap() -> None:
    context = _ye_family_population_context(insert_gap=True)

    skeletons = native_extraction._account_anchor_skeletons(context)
    r1_sequences = [
        row.get("category_sequence")
        for row in skeletons
        if row.get("account_type") == "revolving_loan_subaccount"
    ]

    assert r1_sequences == [1, 2, 3]


def test_generic_loan_family_carry_rejects_competing_family_before_heading() -> None:
    context = _ye_family_population_context()
    page14 = next(page for page in context.pages if page.page_number == 14)
    page14.tables.insert(
        1,
        _table(
            "competing-r2-before-heading",
            [
                _REVOLVING_PHASE_HEADER,
                [
                    "示例银行",
                    "D10053310H00012022052901021012089466554399",
                    "2021.01.02",
                    "100000",
                    "人民币元",
                    "个人消费贷款",
                    "信用",
                ],
            ],
            top=250.0,
        ),
    )

    ledger = native_extraction._source_completeness_ledger(context)

    assert ledger["account_family_endpoints"]["revolving_loan_subaccount"] == 5
    assert ledger["credit_accounts"] == 41


@pytest.mark.parametrize(
    "defect",
    [
        "table_outside_anchor_interval",
        "two_tables_in_one_interval",
        "missing_anchor_bbox",
        "non_increasing_anchor_tops",
        "nan_anchor_top",
        "nan_table_top",
    ],
)
def test_generic_loan_family_carry_requires_exact_anchor_table_intervals(
    defect: str,
) -> None:
    context = _ye_family_population_context()
    page13 = next(page for page in context.pages if page.page_number == 13)
    evidence13 = next(
        page for page in context.corrected_evidence_pages() if page["page"] == 13
    )
    if defect == "table_outside_anchor_interval":
        page13.tables[0].bbox[1] = 290.0
    elif defect == "two_tables_in_one_interval":
        page13.tables[1].bbox[1] = 100.0
    elif defect == "non_increasing_anchor_tops":
        evidence13["lines"][1]["bbox"][1:4:2] = [50.0, 62.0]
    elif defect == "nan_anchor_top":
        evidence13["lines"][1]["bbox"][1] = float("nan")
    elif defect == "nan_table_top":
        page13.tables[0].bbox[1] = float("nan")
    else:
        evidence13["lines"][0].pop("bbox")

    ledger = native_extraction._source_completeness_ledger(context)

    assert ledger["account_family_endpoints"]["revolving_loan_subaccount"] == 3
    assert ledger["credit_accounts"] == 39


def _exact_sparse_card7_table() -> SimpleNamespace:
    identifier = "B10411000H000115602800002159651279117266"
    header = ["" for _column in range(13)]
    values = ["" for _column in range(13)]
    for column, label, value in zip(
        # The actual prior card table prints its own currency header in
        # primitive column 9, while the separately printed next-account header
        # maps currency to primitive column 8.
        [0, 2, 4, 5, 7, 9, 10, 12],
        _CARD_HEADER,
        _card_values(identifier),
        strict=True,
    ):
        header[column] = label
        values[column] = value
    column_bands = [
        {"index": index, "x0": 42.5 + 27.0 * index, "x1": 69.5 + 27.0 * index}
        for index in range(13)
    ]
    row_bands = [
        {"index": 0, "y0": 220.0, "y1": 240.0},
        {"index": 1, "y0": 240.0, "y1": 529.0},
    ]
    geometry = {
        "coordinate_system": "pdf_points_top_left",
        "cell_bboxes": [
            [[band["x0"], row["y0"], band["x1"], row["y1"]] for band in column_bands]
            for row in row_bands
        ],
        "cell_geometry_status": [
            ["exact" for _column in range(13)] for _row in range(2)
        ],
        "cell_evidence_ids": [
            [
                [f"card7:{row}:{column}"] if [header, values][row][column] else []
                for column in range(13)
            ]
            for row in range(2)
        ],
        "cell_spans": [],
        "row_bands": row_bands,
        "col_bands": column_bands,
    }
    return SimpleNamespace(
        table_id="pt_23_1",
        metadata={"raw_rows": [header, values], "geometry": geometry},
        headers=[],
        rows=[],
        bbox=[42.5, 220.0, 393.5, 529.0],
        confidence=0.99,
    )
def _exact_sparse_card8_table(*, left: float = 52.5) -> SimpleNamespace:
    values = ["" for _column in range(13)]
    values[0] = "中信银行股份 有限公司信用 卡中心"
    values[2] = "B10611000H 00016226880 21919136860 7"
    values[4] = "2018.10.24"
    values[5] = "13,000 福"
    values[8] = "人民币元"
    values[10] = "贷记卡"
    values[12] = "信用/无担保"
    column_bands = [
        {"index": index, "x0": left + 27.0 * index, "x1": left + 27.0 * (index + 1)}
        for index in range(13)
    ]
    geometry = {
        "coordinate_system": "pdf_points_top_left",
        "cell_bboxes": [
            [[band["x0"], 51.0, band["x1"], 84.0] for band in column_bands]
        ],
        "cell_geometry_status": [["exact" for _column in range(13)]],
        "cell_evidence_ids": [
            [
                [f"card8:value:{column}"] if values[column] else []
                for column in range(13)
            ]
        ],
        "cell_spans": [],
        "row_bands": [{"index": 0, "y0": 51.0, "y1": 84.0}],
        "col_bands": column_bands,
    }
    return SimpleNamespace(
        table_id="pt_24_0",
        metadata={"raw_rows": [values], "geometry": geometry},
        headers=[],
        rows=[],
        bbox=[left, 51.0, left + 351.0, 84.0],
        confidence=0.99,
    )


def _real_headerless_card8_context(
    *,
    header_defect: str = "",
    continuation_decision: bool = True,
    candidate_left: float = 52.5,
) -> SimpleNamespace:
    card9_identifier = "B11911000H000115661000042356833"
    previous = _exact_sparse_card7_table()
    card8 = _exact_sparse_card8_table(left=candidate_left)
    card9 = _table(
        "pt_24_1",
        [_CARD_HEADER, _card_values(card9_identifier)],
        top=300.0,
    )
    header_columns = [0, 2, 4, 5, 7, 8, 10, 12]
    header_lines = []
    for index, (label, column) in enumerate(
        zip(_CARD_HEADER, header_columns, strict=True)
    ):
        left = 42.5 + 27.0 * column + 2.0
        evidence_ids = (
            [] if header_defect == f"missing_evidence_{index}" else [f"card8:h:{index}"]
        )
        header_lines.append(
            {
                "text": label,
                "bbox": [left, 546.0 + (index % 2), left + 20.0, 558.0 + (index % 2)],
                "evidence_ids": evidence_ids,
            }
        )
    if header_defect == "transposed_currency_business":
        header_lines[5]["bbox"], header_lines[6]["bbox"] = (
            header_lines[6]["bbox"],
            header_lines[5]["bbox"],
        )
    evidence = [
        {
            "page": 23,
            "source_page": 12,
            "lines": [
                {
                    "text": "（四）贷记卡账户",
                    "bbox": [52.5, 180.0, 200.0, 192.0],
                    "evidence_ids": ["card-family"],
                },
                {
                    "text": "账户 7：",
                    "bbox": [52.5, 200.0, 150.0, 213.0],
                    "evidence_ids": ["card7-anchor"],
                },
                {
                    "text": "账户 8（授信协议标识:B10611000H00016226880219191368607）",
                    "bbox": [52.5, 534.5, 360.0, 545.0],
                    "evidence_ids": ["card8-anchor"],
                },
                *header_lines,
            ],
        },
        {
            "page": 24,
            "source_page": 12,
            "lines": [
                {
                    "text": "账户 9：",
                    "bbox": [52.5, 280.0, 150.0, 293.0],
                    "evidence_ids": ["card9-anchor"],
                }
            ],
        },
    ]
    return SimpleNamespace(
        pages=[_page(23, [previous]), _page(24, [card8, card9])],
        reading_order_by_logical={23: 1, 24: 2},
        reading_order_resolution={"resolved": True, "authoritative": True},
        corrected_evidence_pages=lambda: evidence,
        tables_continue=lambda left, right: (
            continuation_decision if (left, right) == ("pt_23_1", "pt_24_0") else None
        ),
        allows_scanned_line_transition=lambda *_args: False,
    )


@pytest.mark.parametrize("continuation_decision", [False, True])
@pytest.mark.parametrize("candidate_left", [52.5, 72.5])
def test_real_anchor_header_lattice_splits_card8_from_card7(
    continuation_decision: bool,
    candidate_left: float,
) -> None:
    context = _real_headerless_card8_context(
        continuation_decision=continuation_decision,
        candidate_left=candidate_left,
    )

    accounts, _repayments, _events = native_extraction._extract_accounts(context)
    by_sequence = {
        int(account["category_sequence"]): account
        for account in accounts
        if account.get("account_type") == "credit_card"
    }

    assert sorted(by_sequence) == [7, 8, 9]
    assert by_sequence[8]["account_identifier"] == "B10611000H00016226880219191368607"
    assert by_sequence[8]["management_institution"] == "中信银行股份有限公司信用卡中心"
    assert by_sequence[8]["open_date"] == "2018-10-24"
    assert by_sequence[8]["credit_limit"] == 13000
    assert by_sequence[8]["currency"] == "CNY"
    assert by_sequence[8]["business_type"] == "贷记卡"
    assert by_sequence[8]["guarantee_type"] == "信用/无担保"
    card7_tables = {
        ref.get("table_id") for ref in by_sequence[7].get("source_refs") or ()
    }
    card8_tables = {
        ref.get("table_id") for ref in by_sequence[8].get("source_refs") or ()
    }
    assert "pt_24_0" not in card7_tables
    assert "pt_24_0" in card8_tables
    assert any(
        issue.get("issue_code") == "candidate_b_headerless_account_owner_resolved"
        and issue.get("observed_value", {}).get("ownership_basis")
        == "printed_anchor_header_lattice"
        for issue in context._personal_detail_extraction_issues
    )


def test_top_zero_card_continuation_updates_boundary_state_atomically() -> None:
    base = _real_headerless_card8_context(continuation_decision=True)
    card7 = base.pages[0].tables[0]
    candidate = base.pages[1].tables[0]
    card9 = base.pages[1].tables[1]
    continuation = deepcopy(candidate)
    continuation.table_id = "pt_24_continuation"
    continuation.bbox[1:4:2] = [0.0, 40.0]
    continuation_geometry = continuation.metadata["geometry"]
    continuation_geometry["row_bands"][0]["y0"] = 0.0
    continuation_geometry["row_bands"][0]["y1"] = 40.0
    continuation_geometry["cell_bboxes"][0] = [
        [band["x0"], 0.0, band["x1"], 40.0]
        for band in continuation_geometry["col_bands"]
    ]
    continuation_values = ["" for _column in range(13)]
    for column in (0, 2, 4, 5, 7, 8, 10, 12):
        continuation_values[column] = f"continuation-{column}"
        continuation_geometry["cell_evidence_ids"][0][column] = [
            f"continuation:{column}"
        ]
    continuation.metadata["raw_rows"] = [continuation_values]
    candidate.table_id = "pt_25_0"

    evidence = base.corrected_evidence_pages()
    page23_lines = evidence[0]["lines"][:2]
    page24_lines = deepcopy(evidence[0]["lines"][2:])
    page24_lines[0]["bbox"][1:4:2] = [50.0, 59.0]
    for index, line in enumerate(page24_lines[1:]):
        line["bbox"][1:4:2] = [60.0 + index % 2, 72.0 + index % 2]
    page25_lines = evidence[1]["lines"]
    corrected = [
        {"page": 23, "source_page": 12, "lines": page23_lines},
        {"page": 24, "source_page": 12, "lines": page24_lines},
        {"page": 25, "source_page": 13, "lines": page25_lines},
    ]
    context = SimpleNamespace(
        pages=[
            _page(23, [card7]),
            _page(24, [continuation]),
            _page(25, [candidate, card9]),
        ],
        reading_order_by_logical={23: 1, 24: 2, 25: 3},
        reading_order_resolution={"resolved": True, "authoritative": True},
        corrected_evidence_pages=lambda: corrected,
        tables_continue=lambda left, right: (
            True
            if (left, right)
            in {
                ("pt_23_1", "pt_24_continuation"),
                ("pt_24_continuation", "pt_25_0"),
            }
            else None
        ),
        allows_scanned_line_transition=lambda *_args: True,
    )

    accounts, _repayments, _events = native_extraction._extract_accounts(context)
    by_sequence = {
        int(account["category_sequence"]): account
        for account in accounts
        if account.get("account_type") == "credit_card"
    }
    assert "pt_25_0" not in {
        ref.get("table_id") for ref in by_sequence[7].get("source_refs") or ()
    }
    assert "pt_25_0" in {
        ref.get("table_id") for ref in by_sequence[8].get("source_refs") or ()
    }


@pytest.mark.parametrize(
    "header_defect",
    ["missing_evidence_3", "transposed_currency_business"],
)
def test_real_anchor_header_lattice_rejects_incomplete_or_transposed_header(
    header_defect: str,
) -> None:
    context = _real_headerless_card8_context(header_defect=header_defect)

    table_accounts, _repayments, _events = native_extraction._extract_table_accounts(
        context
    )

    assert not any(
        account.get("_pending_anchor_account_id")
        and any(
            ref.get("table_id") == "pt_24_0"
            for ref in account.get("source_refs") or ()
        )
        for account in table_accounts
    )


def _owns_real_card8_table(context: SimpleNamespace) -> bool:
    table_accounts, _repayments, _events = native_extraction._extract_table_accounts(
        context
    )
    return any(
        any(ref.get("table_id") == "pt_24_0" for ref in account.get("source_refs") or ())
        for account in table_accounts
    )


@pytest.mark.parametrize(
    "defect",
    [
        "unsupported_equal_coordinate_plane",
        "coordinate_plane_mismatch",
        "empty_prior_table_id",
        "duplicate_prior_table_id",
        "duplicate_prior_physical_lattice",
        "duplicate_prior_lattice_with_equivalent_spans",
        "nan_prior_table_bbox",
        "nan_pending_anchor_bbox",
        "malformed_hidden_anchor_before_valid_later_anchor",
        "prior_bbox_mismatch",
        "candidate_bbox_mismatch",
        "candidate_cell_nonexact",
        "candidate_projected_cell_evidence_missing",
        "prior_projected_cell_nonexact",
        "candidate_lattice_incomplete",
        "candidate_interior_boundary_warp",
        "wrong_role_ordinal",
        "boundary_ambiguous_label",
        "competing_first_candidate_table",
        "weak_prior_owner_identifier",
        "invalid_candidate_currency",
        "replayed_prior_identifier",
    ],
)
@pytest.mark.parametrize("continuation_decision", [False, True])
def test_real_anchor_header_lattice_rejects_unowned_or_malformed_planes(
    defect: str,
    continuation_decision: bool,
) -> None:
    context = _real_headerless_card8_context(
        continuation_decision=continuation_decision
    )
    prior = context.pages[0].tables[0]
    candidate = context.pages[1].tables[0]
    prior_geometry = prior.metadata["geometry"]
    candidate_geometry = candidate.metadata["geometry"]
    if defect == "unsupported_equal_coordinate_plane":
        prior_geometry["coordinate_system"] = "image_pixels"
        candidate_geometry["coordinate_system"] = "image_pixels"
    elif defect == "coordinate_plane_mismatch":
        candidate_geometry["coordinate_system"] = "image_pixels"
    elif defect == "empty_prior_table_id":
        prior.table_id = ""
    elif defect == "duplicate_prior_table_id":
        context.pages[0].tables.append(deepcopy(prior))
    elif defect == "duplicate_prior_physical_lattice":
        duplicate = deepcopy(prior)
        duplicate.table_id = "pt_23_duplicate"
        context.pages[0].tables.append(duplicate)
    elif defect == "duplicate_prior_lattice_with_equivalent_spans":
        bands = prior_geometry["col_bands"]
        rows = prior_geometry["row_bands"]
        prior_geometry["cell_spans"] = [
            {
                "row": 1,
                "col": column,
                "row_span": 1,
                "col_span": 1,
                "bbox": [
                    bands[column]["x0"],
                    rows[1]["y0"],
                    bands[column]["x1"],
                    rows[1]["y1"],
                ],
            }
            for column in (1, 3)
        ]
        duplicate = deepcopy(prior)
        duplicate.table_id = "pt_23_duplicate"
        duplicate.metadata["geometry"]["cell_spans"].reverse()
        duplicate.metadata["geometry"]["cell_spans"][0].pop("bbox")
        context.pages[0].tables.append(duplicate)
    elif defect == "nan_prior_table_bbox":
        prior.bbox[1] = float("nan")
    elif defect == "nan_pending_anchor_bbox":
        context.corrected_evidence_pages()[0]["lines"][2]["bbox"][1] = float(
            "nan"
        )
    elif defect == "malformed_hidden_anchor_before_valid_later_anchor":
        lines = context.corrected_evidence_pages()[0]["lines"]
        lines[2]["bbox"][1] = float("nan")
        lines.insert(
            3,
            {
                "text": "账户 10：",
                "bbox": [52.5, 535.0, 150.0, 545.0],
                "evidence_ids": ["card10-anchor"],
            },
        )
    elif defect == "prior_bbox_mismatch":
        prior.bbox[0] += 20.0
    elif defect == "candidate_bbox_mismatch":
        candidate.bbox[2] += 20.0
    elif defect == "candidate_cell_nonexact":
        candidate_geometry["cell_geometry_status"][0][4] = "derived"
    elif defect == "candidate_projected_cell_evidence_missing":
        candidate_geometry["cell_evidence_ids"][0][8] = []
    elif defect == "prior_projected_cell_nonexact":
        prior_geometry["cell_geometry_status"][0][8] = "derived"
    elif defect == "candidate_lattice_incomplete":
        candidate_geometry["col_bands"].pop()
    elif defect == "candidate_interior_boundary_warp":
        candidate_geometry["col_bands"][4]["x1"] += 4.0
        candidate_geometry["col_bands"][5]["x0"] += 4.0
        candidate_geometry["cell_bboxes"][0][4][2] += 4.0
        candidate_geometry["cell_bboxes"][0][5][0] += 4.0
    elif defect in {"wrong_role_ordinal", "boundary_ambiguous_label"}:
        header_line = context.corrected_evidence_pages()[0]["lines"][5]
        if defect == "wrong_role_ordinal":
            left = 42.5 + 27.0 * 3 + 2.0
            header_line["bbox"] = [left, 546.0, left + 20.0, 558.0]
        else:
            boundary = 69.5
            header_line["bbox"] = [boundary - 2.0, 546.0, boundary + 2.0, 558.0]
    elif defect == "competing_first_candidate_table":
        context.pages[1].tables.insert(
            0,
            _table("unrelated-top-table", [["unrelated"]], top=5.0),
        )
    elif defect == "weak_prior_owner_identifier":
        prior.metadata["raw_rows"][1][2] = "A00000000000"
    elif defect == "invalid_candidate_currency":
        candidate.metadata["raw_rows"][0][8] = "NOT-A-CURRENCY"
    else:
        candidate.metadata["raw_rows"][0][2] = prior.metadata["raw_rows"][1][2]

    assert not _owns_real_card8_table(context)
    assert any(
        issue.get("issue_code")
        in {
            "candidate_b_headerless_account_owner_unresolved",
            "candidate_b_headerless_account_value_contract_unresolved",
        }
        and any(
            ref.get("table_id") == "pt_24_0"
            for ref in issue.get("source_refs") or ()
        )
        for issue in getattr(context, "_personal_detail_extraction_issues", ())
    )
def _inquiry_value_row(sequence: int, raw_sequence: str) -> list[str]:
    return [
        raw_sequence,
        f"2024.{1 + (sequence - 1) // 28:02d}.{1 + (sequence - 1) % 28:02d}",
        f"示例银行{sequence}股份有限公司",
        "贷后管理",
    ]


def _exact_collapsed_inquiry_table(
    rows: list[list[str]],
    *,
    span_column: int = 1,
) -> SimpleNamespace:
    assert span_column in {0, 1}
    x_bands = [(44.5, 84.0), (84.0, 161.5), (161.5, 316.0), (316.0, 395.0)]
    row_bands = [
        {"index": index, "y0": 93.5 + index * 13.0, "y1": 106.5 + index * 13.0}
        for index in range(len(rows))
    ]
    cell_bboxes: list[list[list[float] | None]] = []
    cell_status: list[list[str]] = []
    cell_evidence_ids: list[list[list[str]]] = []
    for row_index, band in enumerate(row_bands):
        if row_index == 0:
            header_boxes: list[list[float] | None] = [
                [left, band["y0"], right, band["y1"]] for left, right in x_bands
            ]
            header_boxes[span_column] = [
                x_bands[span_column][0],
                band["y0"],
                x_bands[span_column + 1][1],
                band["y1"],
            ]
            header_boxes[span_column + 1] = None
            header_status = ["exact" for _column in range(4)]
            header_status[span_column + 1] = "derived"
            header_evidence = [[f"inq:h:{column}"] for column in range(4)]
            header_evidence[span_column] = [
                f"inq:h:{span_column}",
                f"inq:h:{span_column + 1}",
            ]
            header_evidence[span_column + 1] = []
            cell_bboxes.append(header_boxes)
            cell_status.append(header_status)
            cell_evidence_ids.append(header_evidence)
            continue
        cell_bboxes.append(
            [[left, band["y0"], right, band["y1"]] for left, right in x_bands]
        )
        cell_status.append(["exact", "exact", "exact", "exact"])
        cell_evidence_ids.append([[f"inq:{row_index}:{column}"] for column in range(4)])
    geometry = {
        "cell_bboxes": cell_bboxes,
        "cell_geometry_status": cell_status,
        "cell_evidence_ids": cell_evidence_ids,
        "cell_spans": [{"row": 0, "col": span_column, "row_span": 1, "col_span": 2}],
        "row_bands": row_bands,
        "col_bands": [
            {"index": index, "x0": left, "x1": right}
            for index, (left, right) in enumerate(x_bands)
        ],
    }
    return SimpleNamespace(
        table_id="pt_27_0",
        metadata={"raw_rows": rows, "geometry": geometry},
        headers=[],
        rows=[],
        bbox=[44.5, 93.5, 395.0, row_bands[-1]["y1"]],
        confidence=0.99,
    )


def _full_ye_inquiry_context() -> SimpleNamespace:
    pt27_rows = [["编号", "? 查询日期 查询机构 X", "", "查询原因"]]
    for sequence in range(1, 37):
        raw_sequence = (
            ""
            if sequence in {5, 8, 9, 29}
            else "27 多"
            if sequence == 27
            else str(sequence)
        )
        pt27_rows.append(_inquiry_value_row(sequence, raw_sequence))
    pt28_rows = []
    for sequence in range(37, 77):
        raw_sequence = (
            "芬 48"
            if sequence == 48
            else ""
            if sequence == 67
            else "K69"
            if sequence == 69
            else str(sequence)
        )
        pt28_rows.append(_inquiry_value_row(sequence, raw_sequence))
    pt29_rows = [
        _inquiry_value_row(sequence, str(sequence))
        for sequence in range(77, 97)
    ]
    pages = [
        _page(27, [_exact_collapsed_inquiry_table(pt27_rows)], template="annotations_and_inquiries"),
        _page(28, [_table("pt_28_0", pt28_rows)], template="annotations_and_inquiries"),
        _page(29, [_table("pt_29_0", pt29_rows)], template="annotations_and_inquiries"),
    ]
    witness_sequences = {27: (8, 9)}
    evidence = []
    for page_number, sequences in witness_sequences.items():
        evidence.append(
            {
                "page": page_number,
                "source_page": page_number,
                "canonical_template_id": "annotations_and_inquiries",
                "lines": [
                    {
                        "text": " ".join(_inquiry_value_row(sequence, str(sequence))),
                        "bbox": [
                            44.5,
                            93.5 + sequence * 13.0,
                            395.0,
                            106.5 + sequence * 13.0,
                        ],
                        "evidence_ids": [f"inq:{sequence}:1"],
                        "confidence": 0.99,
                    }
                    for sequence in sequences
                ],
            }
        )
    return SimpleNamespace(
        pages=pages,
        reading_order_by_logical={27: 1, 28: 2, 29: 3},
        reading_order_resolution={"resolved": True, "authoritative": True},
        corrected_evidence_pages=lambda: evidence,
    )


def test_ye_inquiry_planes_emit_all_institution_sequences_one_to_96() -> None:
    context = _full_ye_inquiry_context()

    records = native_extraction._extract_inquiries(context)
    coverage = native_extraction._inquiry_source_coverage(context)
    institution = sorted(
        int(record["sequence"])
        for record in records
        if record.get("inquiry_type") == "institution"
    )

    assert institution == list(range(1, 97))
    assert coverage["sequence_endpoints"] == {"institution": 96}
    assert any(
        issue.get("issue_code")
        == "candidate_b_inquiry_sequence_inferred_from_row_order"
        and issue.get("candidate_value", {}).get("normalized_sequence") == 5
        for issue in context._personal_detail_extraction_issues
    )
    assert any(
        issue.get("issue_code")
        == "candidate_b_inquiry_sequence_suffix_noise_corrected"
        and issue.get("candidate_value", {}).get("normalized_sequence") == 27
        for issue in context._personal_detail_extraction_issues
    )
    assert any(
        issue.get("issue_code")
        == "candidate_b_inquiry_sequence_prefix_noise_corrected"
        and issue.get("candidate_value", {}).get("normalized_sequence") == 69
        for issue in context._personal_detail_extraction_issues
    )
    assert sorted(
        int(record["sequence"])
        for record in native_extraction._extract_inquiries(context)
        if record.get("inquiry_type") == "institution"
    ) == list(range(1, 97))


def test_inquiry_line_witness_rejects_duplicate_fingerprint_on_different_row() -> None:
    rows = [
        ["缂栧彿", "? 鏌ヨ鏃ユ湡 鏌ヨ鏈烘瀯 X", "", "鏌ヨ鍘熷洜"],
        _inquiry_value_row(1, "1"),
        _inquiry_value_row(2, ""),
        _inquiry_value_row(3, "3"),
    ]
    table = _exact_collapsed_inquiry_table(rows)
    target = rows[2]
    duplicate_business_row = {
        "inquiry_id": "credit_inquiry:institution:2",
        "sequence": 2,
        "inquiry_date": target[1],
        "institution": target[2],
        "reason": target[3],
        "source_refs": [
            {
                "source": "candidate_b_canonical_inquiry_line",
                "logical_page": 27,
                "bbox": [44.5, 132.5, 395.0, 145.5],
                "geometry_scope": "row",
                "evidence_ids": ["inq:3:1"],
            }
        ],
    }

    assert (
        native_extraction._exact_inquiry_line_sequence_witness(
            [duplicate_business_row],
            table=table,
            row_index=2,
            repaired_line_ids=set(),
            logical_page=27,
            observed={
                "inquiry_date": target[1],
                "institution": target[2],
                "reason": target[3],
            },
            already_observed_sequences={1, 3},
        )
        is None
    )


def test_noisy_inquiry_candidate_leaves_one_uniquely_bracketed_blank() -> None:
    assert native_extraction._document_local_inquiry_ordinals(
        [4, None, 6, 68, None, 70],
        noisy_candidates=[None, None, None, None, (69, "prefixed_noise"), None],
    ) == [
        (4, None),
        (5, "missing"),
        (6, None),
        (68, None),
        (69, "prefixed_noise"),
        (70, None),
    ]


def test_collapsed_inquiry_header_rejects_nonexact_middle_span() -> None:
    rows = [
        ["编号", "? 查询日期 查询机构 X", "", "查询原因"],
        *[_inquiry_value_row(sequence, str(sequence)) for sequence in range(1, 4)],
    ]
    table = _exact_collapsed_inquiry_table(rows)
    table.metadata["geometry"]["cell_spans"][0]["col_span"] = 3

    assert (
        native_extraction._bounded_collapsed_inquiry_header_slots(
            rows, table=table
        )
        is None
    )
def test_left_merged_personal_inquiry_header_emits_all_sixteen_rows() -> None:
    dates = [
        "2024.01.15",
        "2023.10.28",
        "2023.09.22",
        "2023.08.28",
        "2023.07.20",
        "2023.06.16",
        "离 2023.04.25",
        "2023.03.26",
        "2023.02.17",
        "2023.01.04",
        "2022.12.11",
        "2022.10.11",
        "2022.07.21",
        "2022.07.05",
        "2022.05.30",
        "2022.04.27",
    ]
    institutions = ["本人" for _sequence in range(16)]
    institutions[11] = "您 本人 业"
    institutions[12] = "真 本人"
    institutions[13] = "苏 本人 6"
    rows = [["编号 查询日期", "", "查询机构", "查询原因"]]
    rows.extend(
        [
            (
                "8 \u56cd"
                if sequence == 8
                else "\u7248"
                if sequence == 9
                else str(sequence)
            ),
            dates[sequence - 1],
            institutions[sequence - 1],
            "本人查询(自助查询机)",
        ]
        for sequence in range(1, 17)
    )
    table = _exact_collapsed_inquiry_table(rows, span_column=0)
    table.table_id = "pt_29_1"
    context = SimpleNamespace(
        pages=[_page(29, [table], template="annotations_and_inquiries")],
        corrected_evidence_pages=lambda: [],
    )

    records = native_extraction._extract_inquiries(context)
    coverage = native_extraction._inquiry_source_coverage(context)

    assert [record["sequence"] for record in records] == list(range(1, 17))
    assert {record["inquiry_type"] for record in records} == {"personal"}
    assert coverage["sequence_endpoints"] == {"personal": 16}
    assert coverage["expected_row_count"] == 16
    assert any(
        issue.get("issue_code") == "candidate_b_inquiry_sequence_suffix_noise_corrected"
        and issue.get("candidate_value", {}).get("normalized_sequence") == 8
        and issue.get("observed_value", {}).get("raw_sequence_text") == "8 \u56cd"
        and "exact_two_row_outer_bracket_sequence_proof"
        in issue.get("reason_codes", [])
        for issue in context._personal_detail_extraction_issues
    )
    assert any(
        issue.get("issue_code")
        == "candidate_b_inquiry_sequence_inferred_from_row_order"
        and issue.get("candidate_value", {}).get("normalized_sequence") == 9
        and "exact_two_row_outer_bracket_sequence_proof"
        in issue.get("reason_codes", [])
        for issue in context._personal_detail_extraction_issues
    )
