from __future__ import annotations

from types import SimpleNamespace

import pytest

from docmirror.plugins.credit_report.personal_detail_scanned.agreement_ocr import (
    canonical_credit_agreement_heading,
)
from docmirror.plugins.credit_report.personal_detail_scanned.canonical_layout import (
    _classify,
    _classify_page,
)
from docmirror.plugins.credit_report.personal_detail_scanned.native_extraction import (
    _account_detail_section_end,
    _exact_account_detail_section_heading,
    _source_completeness_ledger,
)
from docmirror.plugins.credit_report.personal_detail_scanned.section_headings import (
    canonical_account_family_heading,
    canonical_registered_section_heading,
)


def _exact_table(table_id: str, rows: list[list[str]]) -> SimpleNamespace:
    width = max((len(row) for row in rows), default=0)
    return SimpleNamespace(
        table_id=table_id,
        metadata={
            "raw_rows": rows,
            "geometry": {
                "coordinate_system": "logical_page_pixels",
                "row_bands": [
                    {"index": row, "y0": 20.0 + row * 20.0, "y1": 40.0 + row * 20.0}
                    for row in range(len(rows))
                ],
                "col_bands": [
                    {"index": column, "x0": 20.0 + column * 100.0, "x1": 120.0 + column * 100.0}
                    for column in range(width)
                ],
                "cell_bboxes": [
                    [
                        [
                            20.0 + column * 100.0,
                            20.0 + row * 20.0,
                            120.0 + column * 100.0,
                            40.0 + row * 20.0,
                        ]
                        for column in range(width)
                    ]
                    for row in range(len(rows))
                ],
                "cell_geometry_status": [["exact"] * width for _row in rows],
                "cell_evidence_ids": [
                    [[f"native:{table_id}:{row}:{column}"] for column in range(width)]
                    for row in range(len(rows))
                ],
                "cell_spans": [],
            },
        },
        headers=[],
        rows=[],
        bbox=[20.0, 20.0, 20.0 + width * 100.0, 20.0 + len(rows) * 20.0],
    )


def _profile_ledger(rows: list[list[str]]) -> dict[str, object]:
    table = _exact_table(
        "mobile-sequences",
        [
            ["编号", "手机号码", "信息更新日期", "数据发生机构名称"],
            *rows,
        ],
    )
    context = SimpleNamespace(
        pages=[SimpleNamespace(page_number=1, source_page_number=1, tables=[table])],
        corrected_evidence_pages=lambda: [],
    )
    return _source_completeness_ledger(context)


def _mobile_row(sequence: int) -> list[str]:
    return [str(sequence), f"138{sequence:08d}", "2024.01.01", "示例机构"]


def test_profile_source_ledger_keeps_dense_ordinals_above_twenty() -> None:
    ledger = _profile_ledger([_mobile_row(sequence) for sequence in range(1, 22)])

    assert ledger["sequence_endpoints"]["mobile_phone_records"] == 21
    assert ledger["sequence_observed_sequences"]["mobile_phone_records"] == list(range(1, 22))
    assert "21" in ledger["sequence_ordinal_observations"]["mobile_phone_records"]
    assert "sequence_outliers" not in ledger


def test_profile_source_ledger_rejects_sparse_high_ordinal_as_outlier() -> None:
    ledger = _profile_ledger([_mobile_row(sequence) for sequence in (1, 2, 999)])

    assert ledger["sequence_endpoints"]["mobile_phone_records"] == 2
    assert ledger["sequence_observed_sequences"]["mobile_phone_records"] == [1, 2]
    assert ledger["sequence_outliers"] == {"mobile_phone_records": [999]}
    assert "999" not in ledger["sequence_ordinal_observations"]["mobile_phone_records"]


def test_profile_source_ledger_keeps_duplicate_ordinal_aggregate_only() -> None:
    ledger = _profile_ledger([_mobile_row(sequence) for sequence in (1, 2, 2)])

    assert ledger["sequence_endpoints"]["mobile_phone_records"] == 2
    assert ledger["sequence_observed_sequences"]["mobile_phone_records"] == [1, 2]
    assert "2" not in ledger["sequence_ordinal_observations"]["mobile_phone_records"]


@pytest.mark.parametrize(
    ("observed", "canonical"),
    [
        ("授信协议1", "授信协议1"),
        ("授伯协议11", "授信协议11"),
        ("投值协议 3", "授信协议3"),
        ("投信协议8", "授信协议8"),
        ("投值协这2", "授信协议2"),
    ],
)
def test_finite_agreement_heading_aliases_canonicalize_exactly(
    observed: str,
    canonical: str,
) -> None:
    assert canonical_credit_agreement_heading(observed) == canonical


@pytest.mark.parametrize(
    "observed",
    [
        "投直协议2",
        "授伯协议信息",
        "投值协议二",
        "投值协这2附注",
    ],
)
def test_unregistered_agreement_heading_near_misses_fail_closed(observed: str) -> None:
    assert canonical_credit_agreement_heading(observed) is None


def test_alias_only_agreements_remain_visible_to_source_completeness() -> None:
    raw_page = SimpleNamespace(
        page_number=1,
        source_page_number=1,
        tables=[],
        texts=[
            SimpleNamespace(
                content=text,
                bbox=[20.0, 20.0 + index * 20.0, 180.0, 36.0 + index * 20.0],
                evidence_ids=[f"agreement:{index}"],
            )
            for index, text in enumerate(
                (
                    "（七）投值协议信息",
                    "投值协议1",
                    "投值协议标识",
                    "授伯协议2",
                    "授伯协议标识",
                    "公共信息明细",
                )
            )
        ],
    )
    context = SimpleNamespace(
        parse_result=SimpleNamespace(pages=[raw_page]),
        _frozen_logical_pages={1: raw_page},
        pages=[],
        corrected_evidence_pages=lambda: [],
        reading_order_by_logical={1: 1},
        reading_order_resolution={"resolved": True, "authoritative": True},
    )

    ledger = _source_completeness_ledger(context)

    assert ledger["credit_agreements"] == 2
    assert ledger["credit_agreement_sequence_endpoint"] == 2
    assert ledger["credit_agreement_observed_sequences"] == [1, 2]


def test_unregistered_agreement_near_miss_is_not_population_evidence() -> None:
    text = "投直协议2 管理机构 投直协议标识 生效日期 到期日期 投直额度用途"
    context = SimpleNamespace(
        pages=[],
        corrected_evidence_pages=lambda: [
            {
                "page": 1,
                "source_page": 1,
                "lines": [{"text": text}],
            }
        ],
    )

    assert _classify(text) is None
    assert "credit_agreements" not in _source_completeness_ledger(context)


@pytest.mark.parametrize("padding_length", [0, 79, 80, 149, 150, 5000])
def test_agreement_text_classification_is_invariant_to_unrelated_padding(
    padding_length: int,
) -> None:
    card = "投值协议7 管理机构 投值协议标识 生效日期 到期日期 投值额度用途"

    classified = _classify("附注" * padding_length + card)

    assert classified is not None
    assert classified[0] == "credit_agreement"
    assert classified[2] == ("numbered_agreement_card_schema",)


def _sealed_line(
    text: str,
    top: float,
    *,
    evidence: bool = True,
    scale: float = 1.0,
) -> dict[str, object]:
    return {
        "text": text,
        "bbox": [20.0 * scale, top, 580.0 * scale, top + 16.0 * scale],
        "evidence_ids": [f"atom:{top}"] if evidence else [],
    }


def test_whole_page_classification_rejects_lower_mixed_section_without_heading() -> None:
    page = SimpleNamespace(width=600.0, height=800.0, tables=[], texts=[])
    evidence = {
        "page_width": 600.0,
        "page_height": 800.0,
        "lines": [
            _sealed_line("账户12 发卡机构 账户标识 账户状态 余额", 40.0),
            _sealed_line("投值协议7", 420.0),
            _sealed_line("管理机构 投值协议标识 生效日期 到期日期 投值额度用途", 450.0),
        ],
    }

    classified = _classify_page(page, evidence)

    assert classified is None


def test_headingless_card_schema_is_not_a_whole_page_role_across_scales() -> None:
    for scale in (0.25, 1.0, 4.0):
        page = SimpleNamespace(width=600.0 * scale, height=800.0 * scale, tables=[], texts=[])
        evidence = {
            "page_width": 600.0 * scale,
            "page_height": 800.0 * scale,
            "lines": [
                _sealed_line("投值协议7", 20.0 * scale, scale=scale),
                _sealed_line(
                    "管理机构 投值协议标识 生效日期 到期日期 投值额度用途",
                    50.0 * scale,
                    scale=scale,
                ),
            ],
        }

        classified = _classify_page(page, evidence)

        assert classified is None


@pytest.mark.parametrize("missing_proof", ["evidence_ids", "bbox"])
def test_mixed_runtime_evidence_without_exact_geometry_fails_closed(
    missing_proof: str,
) -> None:
    page = SimpleNamespace(width=600.0, height=800.0, tables=[], texts=[])
    account_line = _sealed_line("账户12 发卡机构 账户标识 账户状态 余额", 40.0)
    if missing_proof == "evidence_ids":
        account_line["evidence_ids"] = []
    else:
        account_line.pop("bbox")
    evidence = {
        "page_width": 600.0,
        "page_height": 800.0,
        "lines": [
            account_line,
            _sealed_line("投值协议7", 420.0),
            _sealed_line("管理机构 投值协议标识 生效日期 到期日期 投值额度用途", 450.0),
        ],
    }

    assert _classify_page(page, evidence) is None


def test_text_only_legacy_agreement_requires_explicit_opt_in() -> None:
    text = "投值协议7 管理机构 投值协议标识 生效日期 到期日期 投值额度用途"
    page = SimpleNamespace(
        width=600.0,
        height=800.0,
        tables=[],
        texts=[SimpleNamespace(content=text)],
    )
    evidence = {"page_width": 600.0, "page_height": 800.0, "lines": []}

    assert _classify_page(page, evidence) is None
    classified = _classify_page(page, evidence, allow_legacy_text_only=True)
    assert classified is not None
    assert classified[0] == "credit_agreement"


def test_registered_heading_requires_sealed_exact_runtime_line() -> None:
    page = SimpleNamespace(width=600.0, height=800.0, tables=[], texts=[])
    evidence = {
        "page_width": 600.0,
        "page_height": 800.0,
        "lines": [
            {"text": "（十一）授信协议信息"},
            {"text": "投值协议7 管理机构 投值协议标识 生效日期 到期日期 投值额度用途"},
        ],
    }

    classified = _classify_page(page, evidence)

    assert classified is None


@pytest.mark.parametrize("scale", [0.25, 1.0, 4.0])
def test_unique_sealed_registered_heading_assigns_role_at_any_page_scale(
    scale: float,
) -> None:
    page = SimpleNamespace(
        width=600.0 * scale,
        height=800.0 * scale,
        tables=[],
        texts=[],
    )
    evidence = {
        "page_width": page.width,
        "page_height": page.height,
        "lines": [
            _sealed_line("（九）授信协议信息", 20.0 * scale, scale=scale),
            _sealed_line(
                "授信协议1 管理机构 授信协议标识 生效日期 到期日期 授信额度用途",
                60.0 * scale,
                scale=scale,
            ),
        ],
    }

    assert _classify_page(page, evidence) == (
        "credit_agreement",
        0.99,
        ("授信协议信息",),
    )


@pytest.mark.parametrize(
    ("heading", "expected"),
    [
        ("一个人基本信息", "个人基本信息"),
        ("十二授信协议信息", "授信协议信息"),
        ("（十二）授信协议信息", "授信协议信息"),
    ],
)
def test_registered_section_title_accepts_pboc_numeral_presentation_only(
    heading: str,
    expected: str,
) -> None:
    assert canonical_registered_section_heading(heading) == expected


@pytest.mark.parametrize(
    "lines",
    [
        # A table of contents lists multiple complete PBOC headings.
        ["目录", "（一）个人基本信息", "（二）信息概要", "（三）信贷交易信息明细"],
        # A mixed page contains two real sections and needs section-local extraction.
        ["（七）查询记录明细", "（八）授信协议信息"],
        # A duplicated sealed atom does not prove a unique physical heading line.
        ["（八）授信协议信息", "（八）授信协议信息"],
    ],
)
def test_multiple_sealed_registered_heading_lines_fail_closed(
    lines: list[str],
) -> None:
    page = SimpleNamespace(width=1200.0, height=1600.0, tables=[], texts=[])
    evidence = {
        "page_width": page.width,
        "page_height": page.height,
        "lines": [
            _sealed_line(text, 20.0 + index * 40.0, scale=2.0)
            for index, text in enumerate(lines)
        ],
    }

    assert _classify_page(page, evidence) is None


@pytest.mark.parametrize(
    "lines",
    [
        ["三信贷交易信息明细", "（一）非循环贷账户", "（二）循环贷账户一"],
        ["（三）循环贷账户一", "（四）循环贷账户二"],
        ["（六）贷记卡账户", "（七）准贷记卡账户"],
    ],
)
def test_distinct_pboc_account_subheadings_may_share_one_semantic_page_role(
    lines: list[str],
) -> None:
    page = SimpleNamespace(width=600.0, height=800.0, tables=[], texts=[])
    evidence = {
        "page_width": page.width,
        "page_height": page.height,
        "lines": [
            _sealed_line(text, 20.0 + index * 40.0)
            for index, text in enumerate(lines)
        ],
    }

    classified = _classify_page(page, evidence)

    assert classified is not None
    assert classified[0] == "credit_account_detail"
    expected_signals: list[str] = []
    for text in lines:
        title = canonical_registered_section_heading(text)
        family = canonical_account_family_heading(text)
        if title is not None:
            expected_signals.append(title)
        elif family is not None:
            expected_signals.append(family)
    assert classified[2] == tuple(expected_signals)


def test_registered_heading_substring_in_prose_is_not_a_page_role() -> None:
    page = SimpleNamespace(width=600.0, height=800.0, tables=[], texts=[])
    evidence = {
        "page_width": page.width,
        "page_height": page.height,
        "lines": [
            _sealed_line("报告说明中提到还款状态说明仅用于解释状态符号。", 20.0),
            _sealed_line("贷记卡账户和准贷记卡账户均可能显示还款记录。", 60.0),
        ],
    }

    assert _classify_page(page, evidence) is None


@pytest.mark.parametrize(
    ("heading", "template_id"),
    [
        ("（六）准贷记卡账户", "credit_account_detail"),
        ("（八）授信协议信息", "credit_agreement"),
        ("（十一）授信协议信息", "credit_agreement"),
        ("（十二）报告说明", "report_explanation"),
    ],
)
def test_role_semantics_do_not_depend_on_sample_section_numeral(
    heading: str,
    template_id: str | None,
) -> None:
    page = SimpleNamespace(width=600.0, height=800.0, tables=[], texts=[])
    evidence = {
        "page_width": page.width,
        "page_height": page.height,
        "lines": [_sealed_line(heading, 20.0)],
    }

    classified = _classify_page(page, evidence)
    if template_id is None:
        assert classified is None
    else:
        assert classified is not None
        assert classified[0] == template_id


@pytest.mark.parametrize(
    "heading",
    [
        "授信协议信息",
        "(一)授信协议信息",
        "（四）授信协议信息",
        "（十一）授信协议信息",
        "（一百零二）授信协议信息：",
    ],
)
def test_registered_section_heading_ignores_only_complete_chinese_numeral(
    heading: str,
) -> None:
    assert canonical_registered_section_heading(heading) == "授信协议信息"


@pytest.mark.parametrize(
    "heading",
    [
        "四)授信协议信息",
        "(4)授信协议信息",
        "A(四)授信协议信息",
        "(四)授信协议信息附注",
        "(四)授信协议",
    ],
)
def test_registered_section_heading_near_misses_fail_closed(heading: str) -> None:
    assert canonical_registered_section_heading(heading) is None


@pytest.mark.parametrize("prefix", ["A", "AB", "x", "ZZ"])
def test_latin_prefixed_account_section_heading_fails_closed(prefix: str) -> None:
    line = _sealed_line(f"{prefix}（四）贷记卡账户", 20.0)

    assert _exact_account_detail_section_heading(line) is None


@pytest.mark.parametrize(
    ("heading", "expected_family"),
    [
        ("（六）准贷记卡账户", "quasi_credit_card"),
        ("（九）贷记卡账户", "credit_card"),
        ("（十二）非循环贷账户", "non_revolving_loan"),
        ("（八）循环贷账户（二）", "revolving_loan_account"),
    ],
)
def test_account_family_title_not_outer_numeral_owns_semantics(
    heading: str,
    expected_family: str,
) -> None:
    line = _sealed_line(heading, 20.0)

    assert _exact_account_detail_section_heading(line) == (
        expected_family,
        "exact",
    )
    assert canonical_account_family_heading(heading) == expected_family


@pytest.mark.parametrize(
    "heading",
    [
        "(6)准贷记卡账户",
        "六）准贷记卡账户",
        "（六）未知账户",
        "（六）贷记卡账户准贷记卡账户",
        "（六）本段说明准贷记卡账户",
    ],
)
def test_account_family_heading_numeral_and_title_near_misses_fail_closed(
    heading: str,
) -> None:
    assert _exact_account_detail_section_heading(_sealed_line(heading, 20.0)) is None


def test_account_section_end_uses_exact_title_not_sample_numeral() -> None:
    assert _account_detail_section_end("（十二）授信协议信息")
    assert _account_detail_section_end("（三）授信协议信息")
    assert not _account_detail_section_end("（六）授信协议")
    assert not _account_detail_section_end("（六）未知账户")


def _agreement_ledger(sequences: list[int]) -> dict[str, object]:
    texts = [
        SimpleNamespace(
            content="（七）授信协议信息",
            bbox=[20.0, 10.0, 180.0, 26.0],
            evidence_ids=["agreement:section"],
        ),
        *[
            SimpleNamespace(
                content=f"授信协议{sequence}",
                bbox=[20.0, 40.0 + index * 20.0, 180.0, 56.0 + index * 20.0],
                evidence_ids=[f"agreement:{index}"],
            )
            for index, sequence in enumerate(sequences)
        ],
        SimpleNamespace(
            content="公共信息明细",
            bbox=[20.0, 60.0 + len(sequences) * 20.0, 180.0, 76.0 + len(sequences) * 20.0],
            evidence_ids=["agreement:boundary"],
        ),
    ]
    raw_page = SimpleNamespace(
        page_number=1,
        source_page_number=1,
        tables=[],
        texts=texts,
    )
    context = SimpleNamespace(
        parse_result=SimpleNamespace(pages=[raw_page]),
        _frozen_logical_pages={1: raw_page},
        pages=[],
        corrected_evidence_pages=lambda: [],
        reading_order_by_logical={1: 1},
        reading_order_resolution={"resolved": True, "authoritative": True},
    )
    return _source_completeness_ledger(context)


def test_dense_agreement_ordinals_above_one_hundred_are_retained() -> None:
    ledger = _agreement_ledger(list(range(1, 104)))

    assert ledger["credit_agreements"] == 103
    assert ledger["credit_agreement_sequence_endpoint"] == 103
    assert ledger["credit_agreement_observed_sequences"][-3:] == [101, 102, 103]
    assert "credit_agreement_sequence_outliers" not in ledger


def test_sparse_high_agreement_ordinal_fails_closed() -> None:
    ledger = _agreement_ledger([1, 2, 999])

    assert "credit_agreements" not in ledger
    assert "credit_agreement_sequence_endpoint" not in ledger
    assert "credit_agreement_observed_sequences" not in ledger


def test_duplicate_agreement_ordinal_fails_closed() -> None:
    ledger = _agreement_ledger([1, 2, 2])

    assert "credit_agreements" not in ledger
    assert "credit_agreement_ordinal_observations" not in ledger
