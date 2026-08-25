from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from docmirror.plugins.credit_report.personal_detail_scanned import native_extraction
from docmirror.plugins.credit_report.personal_detail_scanned.source_projection import (
    prepare_personal_detail_source_collections,
)


def _text(
    content: str,
    top: float,
    evidence_id: str,
    *,
    scale: float = 1.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        content=content,
        bbox=[17.0 * scale, top * scale, 260.0 * scale, (top + 9.0) * scale],
        evidence_ids=[evidence_id],
    )


def _account_table(table_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        table_id=table_id,
        bbox=[10.0, 180.0, 500.0, 260.0],
        metadata={
            "raw_rows": [
                ["管理机构", "账户标识", "开立日期", "账户币种"],
                ["某银行", "D10000000H00012024010101021012000000000001", "2024.01.02", "人民币元"],
            ]
        },
        headers=[],
        rows=[],
    )


def _page(
    logical_page: int,
    source_page: int,
    *texts: SimpleNamespace,
    with_account_table: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        page_number=logical_page,
        source_page_number=source_page,
        width=600.0,
        height=800.0,
        texts=list(texts),
        tables=[_account_table(f"account:{logical_page}")] if with_account_table else [],
    )


def _context(
    pages: list[SimpleNamespace],
    *,
    order: dict[int, int] | None = None,
    resolved: bool = True,
    authoritative: bool = True,
    corrected_pages: list[dict[str, object]] | None = None,
) -> SimpleNamespace:
    registered_order = order or {
        page.page_number: index for index, page in enumerate(pages, start=1)
    }
    return SimpleNamespace(
        parse_result=SimpleNamespace(pages=pages),
        _frozen_logical_pages={page.page_number: page for page in pages},
        pages=[],
        corrected_evidence_pages=lambda: corrected_pages or [],
        reading_order_by_logical=registered_order,
        reading_order_resolution={
            "resolved": resolved,
            "authoritative": authoritative,
            **(
                {}
                if resolved and authoritative
                else {
                    "basis": "unresolved_identity_fallback",
                    "identity_fallback": True,
                }
            ),
        },
        tables_continue=lambda _left, _right: None,
        _personal_detail_extraction_issues=[],
    )


def _dense_context(
    *,
    count: int = 5,
    scale: float = 1.0,
    family_heading: str = "（一）非循环贷账户",
) -> SimpleNamespace:
    page = _page(
        41,
        23,
        _text("（五）信贷交易信息明细", 10.0, "account:start", scale=scale),
        _text(family_heading, 30.0, "family:start", scale=scale),
        *(
            _text(
                (
                    f"账户{ordinal}(授信协议标识:D10000000H000120240101{ordinal:04d})"
                    if ordinal == count
                    else f"账户{ordinal}"
                ),
                45.0 + ordinal * 13.0,
                f"anchor:{ordinal}",
                scale=scale,
            )
            for ordinal in range(1, count + 1)
        ),
        _text(
            "（六）授信协议信息",
            65.0 + count * 13.0,
            "account:boundary",
            scale=scale,
        ),
    )
    return _context([page], order={41: 1})


def _corrected_page(page: SimpleNamespace) -> dict[str, object]:
    return {
        "page": page.page_number,
        "source_page": page.source_page_number,
        "page_width": page.width,
        "page_height": page.height,
        "lines": [
            {
                "text": text.content,
                "bbox": list(text.bbox),
                "evidence_ids": list(text.evidence_ids),
                "confidence": 0.99,
            }
            for text in page.texts
        ],
    }


def _install_unsplit_topology(context: SimpleNamespace) -> None:
    source_by_logical = {
        page.page_number: page.source_page_number
        for page in context._frozen_logical_pages.values()
    }
    groups = {
        page.source_page_number: (page.page_number,)
        for page in context._frozen_logical_pages.values()
    }
    geometries = {
        page.page_number: SimpleNamespace(
            source_page=page.source_page_number,
            width=600.0,
            height=800.0,
            split_kind="none",
            segment_index=None,
            selected_rotation=0,
            source_crop_bbox=None,
            transform_usable=True,
        )
        for page in context._frozen_logical_pages.values()
    }
    context.source_page_by_logical = source_by_logical
    context.page_topology = SimpleNamespace(
        audit=lambda: {"valid": True},
        geometry=lambda logical: geometries.get(logical),
        ordered_fragments=lambda source: groups.get(source, ()),
        ordered_pair=lambda _logicals: None,
    )


@pytest.mark.parametrize(
    ("count", "scale"),
    ((1, 0.55), (5, 1.0), (21, 1.8)),
)
def test_raw_account_anchor_census_scales_without_a_fixture_count_ceiling(
    count: int,
    scale: float,
) -> None:
    census = native_extraction._sealed_raw_account_population_census(
        _dense_context(count=count, scale=scale)
    )

    assert census is not None
    assert census["sequences"] == {"non_revolving_loan": list(range(1, count + 1))}
    assert census["endpoints"] == {"non_revolving_loan": count}
    for ordinal, observation in census["ordinal_observations"]["non_revolving_loan"].items():
        assert observation["account_id"] == f"credit_account:non_revolving_loan:{ordinal}"
        assert set(observation) == {
            "account_id",
            "account_type",
            "category_sequence",
            "source_refs",
        }
        assert observation["source_refs"][0]["binding"] == "printed_account_ordinal"


def test_raw_account_anchor_census_uses_registered_order_not_page_numbers() -> None:
    first = _page(
        50,
        10,
        _text("信贷交易信息明细", 10.0, "start"),
        _text("（四）贷记卡账户", 30.0, "family"),
        _text("账户1", 50.0, "anchor:1"),
        _text("账户2", 70.0, "anchor:2"),
    )
    second = _page(
        7,
        11,
        _text("账户3", 20.0, "anchor:3"),
        _text("账户4", 40.0, "anchor:4"),
        _text("账户5", 60.0, "anchor:5"),
        _text("授信协议信息", 90.0, "boundary"),
    )
    context = _context([second, first], order={50: 1, 7: 2})

    census = native_extraction._sealed_raw_account_population_census(context)

    assert census is not None
    assert census["sequences"] == {"credit_card": [1, 2, 3, 4, 5]}


def _independently_sealed_family_context() -> SimpleNamespace:
    page = _page(
        61,
        31,
        _text("信贷交易信息明细", 10.0, "account:start"),
        # This OCR-damaged family title cannot own the following anchors.
        _text("（一）丰循环贷账户", 25.0, "damaged:family"),
        _text("账户1", 38.0, "damaged:1"),
        _text("账户2", 51.0, "damaged:2"),
        _text("（二）循环贷账户二", 65.0, "revolving:family"),
        _text("账户1", 78.0, "revolving:1"),
        # The unmatched metadata suffix keeps ordinal 2 outside the exact
        # heading grammar, so this whole family interval must be quarantined.
        _text(
            "账户2(授信协议标识:D10056510H000120230109175437",
            91.0,
            "revolving:damaged:2",
        ),
        _text("账户3", 104.0, "revolving:3"),
        _text("（三）贷记卡账户", 118.0, "card:family"),
        _text("账户1", 131.0, "card:1"),
        _text("账户2", 144.0, "card:2"),
        _text("账户3", 157.0, "card:3"),
        _text("（五）授信协议信息", 171.0, "account:boundary"),
    )
    return _context([page], order={61: 1})


def test_raw_account_census_quarantines_only_invalid_family_intervals() -> None:
    census = native_extraction._sealed_raw_account_population_census(
        _independently_sealed_family_context()
    )

    assert census is not None
    assert census["sequences"] == {"credit_card": [1, 2, 3]}
    assert census["endpoints"] == {"credit_card": 3}
    assert set(census["ordinal_observations"]) == {"credit_card"}


def test_raw_family_census_extends_only_its_derived_family_population(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _independently_sealed_family_context()

    def skeleton(account_family: str, ordinal: int, offset: int) -> dict[str, object]:
        return {
            "account_id": f"credit_account:{account_family}:{ordinal}",
            "account_type": account_family,
            "account_family_quality": "exact",
            "category_sequence": ordinal,
            "_printed_ordinal_status": "printed_unique",
            "_canonical_segment": {
                "ownership_basis": "printed_anchor_to_next_anchor"
            },
            "source_refs": [
                {
                    "source": "candidate_b_account_anchor",
                    "logical_page": 61,
                    "source_page": 31,
                    "bbox": [
                        300.0,
                        float(offset),
                        340.0,
                        float(offset + 9),
                    ],
                    "evidence_ids": [f"derived:{account_family}:{ordinal}"],
                }
            ],
        }

    derived = [
        skeleton(account_family, ordinal, base + ordinal * 10)
        for account_family, count, base in (
            ("non_revolving_loan", 2, 200),
            ("revolving_loan_account", 2, 230),
            ("credit_card", 2, 260),
        )
        for ordinal in range(1, count + 1)
    ]
    monkeypatch.setattr(
        native_extraction,
        "_account_anchor_skeletons",
        lambda _context: deepcopy(derived),
    )
    monkeypatch.setattr(
        native_extraction,
        "_repair_complete_account_anchor_skeletons",
        lambda _context, rows: rows,
    )

    ledger = native_extraction._source_completeness_ledger(context)

    assert ledger["credit_accounts"] == 7
    assert ledger["account_family_endpoints"] == {
        "non_revolving_loan": 2,
        "revolving_loan_account": 2,
        "credit_card": 3,
    }
    assert ledger["account_raw_anchor_sequence_endpoints"] == {"credit_card": 3}


def _family_interval_barrier_context(barrier_text: str) -> SimpleNamespace:
    page = _page(
        62,
        32,
        _text("信贷交易信息明细", 10.0, "account:start"),
        _text("（一）非循环贷账户", 25.0, "loan:family"),
        _text("账户1", 38.0, "loan:1"),
        _text("账户2", 51.0, "loan:2"),
        _text(barrier_text, 65.0, "interval:barrier"),
        _text("账户3", 78.0, "untyped:3"),
        _text("账户4", 91.0, "untyped:4"),
        _text("（三）贷记卡账户", 105.0, "card:family"),
        _text("账户1", 118.0, "card:1"),
        _text("账户2", 131.0, "card:2"),
        _text("（五）授信协议信息", 145.0, "account:boundary"),
    )
    return _context([page], order={62: 1})


@pytest.mark.parametrize(
    "barrier",
    (
        "（二）丰循环贷账户",
        "(4)丰循环贷账户",
        "（4）丰循环贷账户",
        "( 4 ) 丰循环贷账户",
        "（ 4 ） 丰循环贷账户",
        "(４)丰循环贷账户",
        "（４）丰循环贷账户",
        "(１２)丰循环贷账户",
        "（ １２ ） 丰循环贷账户",
        "（1２）丰循环贷账户",
        "（１2）丰循环贷账户",
        "（肆）丰循环贷账户",
        "（贰）丰循环贷账户",
        "（貳）丰循环贷账户",
        "（壹拾贰）丰循环贷账户",
        "（壹拾貳）丰循环贷账户",
        "二丰循环贷账户",
        "肆丰循环贷账户",
        "壹拾贰丰循环贷账户",
        "(二）丰循环贷账户",
        "（二)丰循环贷账户",
        "丰循环贷账户",
        "（二）循坏贷账户一",
        "（二）循坏贷账户二",
        "（二）循坏贷账户（一）",
        "（二）循坏贷账户（二）",
    ),
)
def test_untyped_family_barrier_prevents_prior_family_ordinal_leakage(
    barrier: str,
) -> None:
    census = native_extraction._sealed_raw_account_population_census(
        _family_interval_barrier_context(barrier)
    )

    assert census is not None
    assert census["sequences"] == {
        "non_revolving_loan": [1, 2],
        "credit_card": [1, 2],
    }
    assert all(
        ordinal not in census["ordinal_observations"]["non_revolving_loan"]
        for ordinal in (3, 4)
    )


@pytest.mark.parametrize(
    ("heading", "family"),
    (
        ("贷记卡账户", "credit_card"),
        ("循环贷账户一", "revolving_loan_subaccount"),
        ("循环贷账户二", "revolving_loan_account"),
        ("循环贷账户（一）", "revolving_loan_subaccount"),
        ("循环贷账户（二）", "revolving_loan_account"),
    ),
)
def test_unnumbered_canonical_family_heading_remains_typed(
    heading: str,
    family: str,
) -> None:
    census = native_extraction._sealed_raw_account_population_census(
        _dense_context(count=2, family_heading=heading)
    )

    assert census is not None
    assert census["sequences"] == {family: [1, 2]}


def _section_interval_barrier_context(barrier_text: str) -> SimpleNamespace:
    page = _page(
        63,
        33,
        _text("信贷交易信息明细", 10.0, "account:start"),
        _text("（三）贷记卡账户", 25.0, "card:family"),
        _text("账户1", 38.0, "card:1"),
        _text("账户2", 51.0, "card:2"),
        _text(barrier_text, 65.0, "damaged:section"),
        _text("账户3", 78.0, "liability:3"),
        _text("账户4", 91.0, "liability:4"),
        _text("（五）授信协议信息", 105.0, "account:boundary"),
    )
    return _context([page], order={63: 1})


@pytest.mark.parametrize(
    "barrier",
    (
        "（四）相关还款责住信息",
        "(4)相关还款责住信息",
        "（4）相关还款责住信息",
        "( 4 ) 相关还款责住信息",
        "（ 4 ） 相关还款责住信息",
        "(４)相关还款责住信息",
        "（４）相关还款责住信息",
        "(１２)相关还款责住信息",
        "（ １２ ） 相关还款责住信息",
        "（1２）相关还款责住信息",
        "（１2）相关还款责住信息",
        "（肆）相关还款责住信息",
        "（贰）相关还款责住信息",
        "（貳）相关还款责住信息",
        "（壹拾贰）相关还款责住信息",
        "（壹拾貳）相关还款责住信息",
        "四相关还款责住信息",
        "肆相关还款责住信息",
        "壹拾贰相关还款责住信息",
        "(四）相关还款责住信息",
        "（四)相关还款责住信息",
        "相关还款责住信息",
    ),
)
def test_untyped_section_barrier_prevents_liability_ordinal_leakage(
    barrier: str,
) -> None:
    census = native_extraction._sealed_raw_account_population_census(
        _section_interval_barrier_context(barrier)
    )

    assert census is not None
    assert census["sequences"] == {"credit_card": [1, 2]}


def _exact_boundary_leak_context(boundary_heading: str) -> SimpleNamespace:
    page = _page(
        64,
        34,
        _text("信贷交易信息明细", 10.0, "account:start"),
        _text("贷记卡账户", 25.0, "card:family"),
        _text("账户1", 38.0, "card:1"),
        _text("账户2", 51.0, "card:2"),
        _text(boundary_heading, 65.0, "exact:boundary"),
        _text("账户3", 78.0, "foreign:3"),
        _text("账户4", 91.0, "foreign:4"),
        _text("公共信息明细", 105.0, "next:boundary"),
    )
    return _context([page], order={64: 1})


@pytest.mark.parametrize(
    "subsection_heading",
    (
        "（九）后付费记录",
        "后付费记录账户",
        "（十）欠税记录",
        "住房公积金参缴记录",
        "(9)后付费记录",
        "（９）后付费记录",
        "玖后付费记录",
        "（玖）后付费记录",
        "(9)后付费记录账户",
        "玖后付费记录账户",
    ),
)
def test_registered_subsection_closes_account_interval(
    subsection_heading: str,
) -> None:
    assert (
        native_extraction._sealed_raw_account_population_census(
            _exact_boundary_leak_context(subsection_heading)
        )
        is None
    )


@pytest.mark.parametrize(
    "label",
    (
        "(9后付费记录",
        "9)后付费记录",
        "9后付费记录",
        "(123456)后付费记录",
        "(A)后付费记录",
        "(9)九后付费记录",
        "(9)后付费记绿",
        "说明(9)后付费记录",
        "(9)后付费记录说明",
        "(9)后付费记录：",
    ),
)
def test_invalid_subsection_prefix_title_and_prose_remain_nonstructural(
    label: str,
) -> None:
    census = native_extraction._sealed_raw_account_population_census(
        _exact_boundary_leak_context(label)
    )

    assert census is not None
    assert census["sequences"] == {"credit_card": [1, 2, 3, 4]}


def test_exact_top_level_section_remains_account_boundary() -> None:
    assert (
        native_extraction._sealed_raw_account_population_census(
            _exact_boundary_leak_context("授信协议信息")
        )
        is None
    )


@pytest.mark.parametrize(
    "label",
    (
        "后付费记录账户合计",
        "住房公积金参缴记录说明",
        "账户数合计",
        "账户信息",
    ),
)
def test_account_subtotals_and_titles_are_not_subsection_boundaries(
    label: str,
) -> None:
    census = native_extraction._sealed_raw_account_population_census(
        _family_interval_barrier_context(label)
    )

    assert census is not None
    assert census["sequences"] == {
        "non_revolving_loan": [1, 2, 3, 4],
        "credit_card": [1, 2],
    }


@pytest.mark.parametrize(
    "ordinal",
    (
        "0",
        "01",
        "０",
        "０１",
        "0１",
        "０1",
        "001",
        "００１",
        "0０1",
        "12345",
        "１２３４５",
        "1２3４5",
    ),
)
def test_bounded_arabic_family_enumerator_is_barrier(
    ordinal: str,
) -> None:
    census = native_extraction._sealed_raw_account_population_census(
        _family_interval_barrier_context(f"（{ordinal}）丰循环贷账户")
    )

    assert census is not None
    assert census["sequences"] == {
        "non_revolving_loan": [1, 2],
        "credit_card": [1, 2],
    }


@pytest.mark.parametrize(
    "ordinal",
    (
        "0",
        "01",
        "０",
        "０１",
        "0１",
        "０1",
        "001",
        "００１",
        "0０1",
        "12345",
        "１２３４５",
        "1２3４5",
    ),
)
def test_bounded_arabic_section_enumerator_is_barrier(
    ordinal: str,
) -> None:
    census = native_extraction._sealed_raw_account_population_census(
        _section_interval_barrier_context(f"({ordinal})相关还款责住信息")
    )

    assert census is not None
    assert census["sequences"] == {"credit_card": [1, 2]}


@pytest.mark.parametrize(
    "heading",
    (
        "(123456)丰循环贷账户",
        "（１２３４５６）丰循环贷账户",
        "（1２3４5６）丰循环贷账户",
        "（壹贰叁肆伍陆）丰循环贷账户",
        "（二丰循环贷账户",
        "二）丰循环贷账户",
        "(二丰循环贷账户",
        "二)丰循环贷账户",
        "2丰循环贷账户",
        "(A)丰循环贷账户",
        "（甲）丰循环贷账户",
        "（４A）丰循环贷账户",
        "(4)丰循环贷账户:",
        "（二）循坏贷账户三",
        "（二）循坏贷账户（三）",
        "（二）循坏贷账户（一",
        "（二）循坏贷账户一：",
    ),
)
def test_out_of_grammar_family_enumerator_remains_nonstructural(
    heading: str,
) -> None:
    census = native_extraction._sealed_raw_account_population_census(
        _family_interval_barrier_context(heading)
    )

    assert census is not None
    assert census["sequences"] == {
        "non_revolving_loan": [1, 2, 3, 4],
        "credit_card": [1, 2],
    }


@pytest.mark.parametrize(
    "heading",
    (
        "(123456)相关还款责住信息",
        "（１２３４５６）相关还款责住信息",
        "（1２3４5６）相关还款责住信息",
        "（壹贰叁肆伍陆）相关还款责住信息",
        "（四相关还款责住信息",
        "四）相关还款责住信息",
        "(四相关还款责住信息",
        "四)相关还款责住信息",
        "4相关还款责住信息",
        "(A)相关还款责住信息",
        "（甲）相关还款责住信息",
        "（４A）相关还款责住信息",
        "（4）相关还款责住信息：",
    ),
)
def test_out_of_grammar_section_enumerator_remains_nonstructural(
    heading: str,
) -> None:
    census = native_extraction._sealed_raw_account_population_census(
        _section_interval_barrier_context(heading)
    )

    assert census is not None
    assert census["sequences"] == {"credit_card": [1, 2, 3, 4]}


@pytest.mark.parametrize(
    "prose",
    (
        "说明（二）丰循环贷账户",
        "（二）丰循环贷账户说明",
        "（四）相关还款责住信息补充",
        "说明(4)丰循环贷账户",
        "(4)丰循环贷账户说明",
        "（ 4 ）相关还款责住信息补充",
        "说明二丰循环贷账户",
        "二丰循环贷账户说明",
        "说明四相关还款责住信息",
        "四相关还款责住信息补充",
        "说明（二）循坏贷账户一",
        "（二）循坏贷账户一说明",
    ),
)
def test_untyped_interval_barrier_rejects_prose_and_substrings(prose: str) -> None:
    census = native_extraction._sealed_raw_account_population_census(
        _family_interval_barrier_context(prose)
    )

    assert census is not None
    assert census["sequences"] == {
        "non_revolving_loan": [1, 2, 3, 4],
        "credit_card": [1, 2],
    }


def _responsibility_suffix_context(
    mutation: str | None = None,
    *,
    account_extra: str | None = None,
    responsibility_extras: tuple[str, ...] = (),
    after_close_extras: tuple[str, ...] = (),
    responsibility_heading: str = "（四）相关还款责任信息",
) -> SimpleNamespace:
    account_and_responsibility = _page(
        50,
        10,
        _text("信贷交易信息明细", 10.0, "account:start"),
        _text("（三）贷记卡账户", 30.0, "account:family"),
        _text("账户1", 50.0, "account:1"),
        _text("账户2", 70.0, "account:2"),
        _text(responsibility_heading, 90.0, "responsibility:start"),
        _text("账户1", 110.0, "responsibility:1"),
        _text("账户2", 130.0, "responsibility:2"),
    )
    if account_extra is not None:
        account_and_responsibility.texts.append(
            _text(account_extra, 80.0, "account:extra")
        )
    account_and_responsibility.texts.extend(
        _text(value, 150.0 + index * 20.0, f"responsibility:extra:{index}")
        for index, value in enumerate(responsibility_extras)
    )
    next_section = _page(
        51,
        11,
        _text("（五）授信协议信息", 20.0, "agreement:start"),
        with_account_table=False,
    )
    next_section.texts.extend(
        _text(value, 40.0 + index * 20.0, f"agreement:extra:{index}")
        for index, value in enumerate(after_close_extras)
    )
    if mutation == "missing_close":
        next_section.texts.clear()
    elif mutation == "duplicate_responsibility_start":
        next_section.texts[0].content = "（五）相关还款责任信息"
    elif mutation == "invalid_prior_section_close":
        next_section.texts[0].content = "（五）信息概要"
    elif mutation == "responsibility_gap":
        account_and_responsibility.texts[-1].content = "账户3"
    elif mutation == "responsibility_duplicate":
        account_and_responsibility.texts[-1].content = "账户1"
    elif mutation == "responsibility_family_heading":
        account_and_responsibility.texts[-1].content = "（四）贷记卡账户"
    elif mutation == "anchor_after_close":
        next_section.texts.append(_text("账户3", 40.0, "agreement:account"))
    return _context(
        [next_section, account_and_responsibility],
        order={50: 1, 51: 2},
    )


def test_raw_account_census_excludes_closed_responsibility_account_ordinals() -> None:
    census = native_extraction._sealed_raw_account_population_census(
        _responsibility_suffix_context()
    )

    assert census is not None
    assert census["sequences"] == {"credit_card": [1, 2]}
    assert census["endpoints"] == {"credit_card": 2}
    assert set(census["ordinal_observations"]["credit_card"]) == {1, 2}


def test_unnumbered_canonical_responsibility_heading_remains_typed() -> None:
    census = native_extraction._sealed_raw_account_population_census(
        _responsibility_suffix_context(
            responsibility_heading="相关还款责任信息",
        )
    )

    assert census is not None
    assert census["sequences"] == {"credit_card": [1, 2]}


_MALFORMED_ACCOUNT_ORDINAL_TOKENS = (
    "账户O",
    "账户〇",
    "账户I",
    "账户l",
    "账户0",
    "账户01",
    "账户i",
    "账户L",
    "账户○",
)


@pytest.mark.parametrize("token", _MALFORMED_ACCOUNT_ORDINAL_TOKENS)
def test_raw_account_census_rejects_confusable_responsibility_ordinal(
    token: str,
) -> None:
    assert (
        native_extraction._sealed_raw_account_population_census(
            _responsibility_suffix_context(responsibility_extras=(token,))
        )
        is None
    )


@pytest.mark.parametrize("token", _MALFORMED_ACCOUNT_ORDINAL_TOKENS)
def test_raw_account_census_rejects_confusable_ordinal_inside_active_family(
    token: str,
) -> None:
    assert (
        native_extraction._sealed_raw_account_population_census(
            _responsibility_suffix_context(account_extra=token)
        )
        is None
    )


def test_raw_account_census_preserves_valid_dense_active_account_ordinals() -> None:
    census = native_extraction._sealed_raw_account_population_census(
        _responsibility_suffix_context(account_extra="账户3"),
    )

    assert census is not None
    assert census["sequences"] == {"credit_card": [1, 2, 3]}


@pytest.mark.parametrize("text", ("账户状态", "账户信息"))
def test_raw_account_census_ignores_non_heading_account_prose_in_active_family(
    text: str,
) -> None:
    census = native_extraction._sealed_raw_account_population_census(
        _responsibility_suffix_context(account_extra=text),
    )

    assert census is not None
    assert census["sequences"] == {"credit_card": [1, 2]}


def test_raw_account_census_preserves_valid_dense_responsibility_ordinals() -> None:
    census = native_extraction._sealed_raw_account_population_census(
        _responsibility_suffix_context(responsibility_extras=("账户3",)),
    )

    assert census is not None
    assert census["sequences"] == {"credit_card": [1, 2]}


def test_raw_account_census_ignores_non_heading_account_prose_in_responsibility() -> None:
    census = native_extraction._sealed_raw_account_population_census(
        _responsibility_suffix_context(
            responsibility_extras=("账户状态", "账户信息"),
        )
    )

    assert census is not None
    assert census["sequences"] == {"credit_card": [1, 2]}


def test_arabic_enumerated_report_explanation_labels_remain_nonstructural() -> None:
    census = native_extraction._sealed_raw_account_population_census(
        _responsibility_suffix_context(
            after_close_extras=("(1)贷记卡账户:", "（ 2 ） 准贷记卡账户："),
        )
    )

    assert census is not None
    assert census["sequences"] == {"credit_card": [1, 2]}


@pytest.mark.parametrize(
    "barrier",
    ("（二）丰循环贷账户", "（四）相关还款责住信息"),
)
def test_untyped_interval_barrier_cannot_bypass_post_boundary_veto(
    barrier: str,
) -> None:
    assert (
        native_extraction._sealed_raw_account_population_census(
            _responsibility_suffix_context(responsibility_extras=(barrier,))
        )
        is None
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_close",
        "duplicate_responsibility_start",
        "invalid_prior_section_close",
        "responsibility_gap",
        "responsibility_duplicate",
        "responsibility_family_heading",
        "anchor_after_close",
    ),
)
def test_raw_account_census_rejects_unclosed_or_ambiguous_responsibility_suffix(
    mutation: str,
) -> None:
    assert (
        native_extraction._sealed_raw_account_population_census(
            _responsibility_suffix_context(mutation)
        )
        is None
    )


def _local_topology_context(*, include_outside: bool = False) -> SimpleNamespace:
    first = _page(
        50,
        10,
        _text("信贷交易信息明细", 10.0, "start"),
        _text("（二）循环贷账户一", 30.0, "family"),
        _text("账户1", 50.0, "anchor:1"),
    )
    second = _page(
        12,
        11,
        _text("账户2", 20.0, "anchor:2"),
        _text("账户3", 40.0, "anchor:3"),
    )
    boundary = _page(
        88,
        12,
        _text("授信协议信息", 20.0, "boundary"),
        with_account_table=False,
    )
    pages = [second, boundary, first]
    if include_outside:
        pages.append(
            _page(
                4,
                13,
                _text("报告说明", 20.0, "outside"),
                with_account_table=False,
            )
        )
    corrected = [_corrected_page(page) for page in pages]
    context = _context(
        pages,
        order={page.page_number: page.page_number for page in pages},
        resolved=False,
        authoritative=False,
        corrected_pages=corrected,
    )
    _install_unsplit_topology(context)
    return context


def test_complete_local_topology_proves_population_but_no_values() -> None:
    context = _local_topology_context()

    census = native_extraction._sealed_raw_account_population_census(context)

    assert census is not None
    assert census["local_topology_authority"] is True
    assert census["sequences"] == {"revolving_loan_subaccount": [1, 2, 3]}
    assert all(
        "printed_fields" not in observation and "field_source_refs" not in observation
        for observation in census["ordinal_observations"]["revolving_loan_subaccount"].values()
    )


def test_identity_fallback_alone_cannot_authorize_population_or_values() -> None:
    context = _local_topology_context()
    del context.page_topology
    del context.source_page_by_logical

    assert native_extraction._sealed_raw_account_population_census(context) is None
    assert (
        native_extraction._registered_account_section_plane(
            context,
            context.corrected_evidence_pages(),
            {},
        )
        is None
    )


def test_local_topology_value_plane_is_clipped_to_the_closed_envelope() -> None:
    context = _local_topology_context(include_outside=True)

    result = native_extraction._registered_account_section_plane(
        context,
        context.corrected_evidence_pages(),
        {},
    )

    assert result is not None
    ordered_pages, plane = result
    assert [page["page"] for page in ordered_pages] == [50, 12, 88]
    assert set(plane) == {50, 12, 88}
    assert 4 not in plane


@pytest.mark.parametrize(
    "mutation",
    [
        "toc_start",
        "prose_anchor",
        "duplicate_ordinal",
        "ordinal_gap",
        "replayed_evidence",
        "duplicate_bbox",
        "blank_evidence",
        "nonstring_evidence",
        "missing_boundary",
        "missing_account_table",
        "cross_section_anchor",
    ],
)
def test_raw_account_anchor_census_fails_closed_on_ambiguity(mutation: str) -> None:
    context = _dense_context(count=3)
    page = context._frozen_logical_pages[41]
    if mutation == "toc_start":
        page.texts[0].content = "目录信贷交易信息明细"
    elif mutation == "prose_anchor":
        page.texts[2].content = "详见账户1"
    elif mutation == "duplicate_ordinal":
        page.texts[3].content = "账户1"
    elif mutation == "ordinal_gap":
        page.texts[3].content = "账户3"
    elif mutation == "replayed_evidence":
        page.texts[2].evidence_ids = list(page.texts[1].evidence_ids)
    elif mutation == "duplicate_bbox":
        page.texts.append(
            SimpleNamespace(
                content="普通正文",
                bbox=list(page.texts[2].bbox),
                evidence_ids=["ordinary"],
            )
        )
    elif mutation == "blank_evidence":
        page.texts[2].evidence_ids = [""]
    elif mutation == "nonstring_evidence":
        page.texts[2].evidence_ids = [7]
    elif mutation == "missing_boundary":
        page.texts.pop()
    elif mutation == "missing_account_table":
        page.tables = []
    elif mutation == "cross_section_anchor":
        page.texts.append(_text("账户4", 140.0, "outside:anchor"))

    assert native_extraction._sealed_raw_account_population_census(context) is None


def test_raw_anchor_ledger_adds_typed_population_without_business_fields() -> None:
    ledger = native_extraction._source_completeness_ledger(_dense_context(count=4))

    assert ledger["credit_accounts"] == 4
    assert ledger["account_family_endpoints"] == {"non_revolving_loan": 4}
    assert ledger["account_raw_anchor_sequence_endpoints"] == {
        "non_revolving_loan": 4
    }
    assert ledger["account_raw_anchor_observed_sequences"] == {
        "non_revolving_loan": [1, 2, 3, 4]
    }
    assert all(
        "printed_fields" not in observation and "field_source_refs" not in observation
        for observation in ledger["account_family_ordinal_observations"][
            "non_revolving_loan"
        ].values()
    )


def _registered_card_population_inventory(
    *,
    count: int = 16,
) -> tuple[list[dict[str, int]], dict[int, list[dict[str, object]]]]:
    pages = [
        {"page": 80, "source_page": 40},
        {"page": 81, "source_page": 41},
        {"page": 82, "source_page": 42},
    ]
    plane: dict[int, list[dict[str, object]]] = {
        int(page["page"]): [] for page in pages
    }
    for ordinal in range(1, count + 1):
        page = pages[min((ordinal - 1) // 6, len(pages) - 1)]
        logical_page = int(page["page"])
        local_index = (ordinal - 1) % 6
        plane[logical_page].append(
            {
                "text": f"账户{ordinal}",
                "page": logical_page,
                "source_page": int(page["source_page"]),
                "account_type": "credit_card",
                "account_family_quality": "exact",
                "bbox": [20.0, 40.0 + local_index * 30.0, 120.0, 55.0 + local_index * 30.0],
                "evidence_ids": [f"registered-card:{ordinal}"],
            }
        )
    return pages, plane


def _registered_card_skeleton(
    inventory: tuple[list[dict[str, int]], dict[int, list[dict[str, object]]]],
    ordinal: int,
) -> dict[str, object]:
    line = next(
        line
        for lines in inventory[1].values()
        for line in lines
        if line["text"] == f"账户{ordinal}"
    )
    return {
        "account_id": f"credit_account:credit_card:{ordinal}",
        "account_type": "credit_card",
        "account_family_quality": "exact",
        "category_sequence": ordinal,
        "_printed_ordinal_status": "printed_unique",
        "_canonical_segment": {"ownership_basis": "printed_anchor_to_next_anchor"},
        "source_refs": [
            {
                "source": "candidate_b_account_anchor",
                "logical_page": line["page"],
                "source_page": line["source_page"],
                "bbox": list(line["bbox"]),
                "evidence_ids": list(line["evidence_ids"]),
            }
        ],
    }


def test_registered_section_plane_conserves_population_filtered_out_of_value_skeletons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory = _registered_card_population_inventory()
    materializable_ordinals = (1, 2, 3, 4, 7, 8, 9)
    skeletons = [
        _registered_card_skeleton(inventory, ordinal)
        for ordinal in materializable_ordinals
    ]
    context = SimpleNamespace(
        pages=[],
        corrected_evidence_pages=lambda: [],
        reading_order_by_logical={80: 1, 81: 2, 82: 3},
        reading_order_resolution={"resolved": True, "authoritative": True},
        _personal_detail_extraction_issues=[],
    )
    monkeypatch.setattr(
        native_extraction,
        "_account_anchor_skeletons",
        lambda _context: deepcopy(skeletons),
    )
    monkeypatch.setattr(
        native_extraction,
        "_repair_complete_account_anchor_skeletons",
        lambda _context, rows: rows,
    )
    monkeypatch.setattr(
        native_extraction,
        "_registered_account_section_plane",
        lambda *_args, **_kwargs: deepcopy(inventory),
    )
    monkeypatch.setattr(
        native_extraction,
        "_exact_account_table_cell_anchors",
        lambda _context: {},
    )
    monkeypatch.setattr(
        native_extraction,
        "_sealed_raw_account_population_census",
        lambda _context: None,
    )
    monkeypatch.setattr(
        native_extraction,
        "_sealed_agreement_population_census",
        lambda _context: None,
    )
    monkeypatch.setattr(
        native_extraction,
        "_inquiry_source_coverage",
        lambda _context: {},
    )

    ledger = native_extraction._source_completeness_ledger(context)
    observations = ledger["account_family_ordinal_observations"]["credit_card"]

    assert ledger["credit_accounts"] == 16
    assert ledger["account_family_endpoints"] == {"credit_card": 16}
    assert set(observations) == {str(ordinal) for ordinal in range(1, 17)}
    assert all(
        "printed_fields" not in observation
        and "field_source_refs" not in observation
        for ordinal, observation in observations.items()
        if int(ordinal) not in materializable_ordinals
    )

    content = prepare_personal_detail_source_collections(
        {
            "facts": {"personal_detail_source_completeness_ledger": ledger},
            "datasets": {
                "credit_accounts": [
                    {
                        "account_id": f"credit_account:credit_card:{ordinal}",
                        "account_type": "credit_card",
                        "category_sequence": ordinal,
                    }
                    for ordinal in materializable_ordinals
                ]
            },
        }
    )
    omissions = [
        issue
        for issue in content["datasets"]["personal_detail_extraction_issues"]
        if issue.get("issue_code") == "source_account_record_omitted"
    ]
    assert len(materializable_ordinals) + len(omissions) == 16
    assert {issue["target_record_id"] for issue in omissions} == {
        f"credit_account:credit_card:{ordinal}"
        for ordinal in (5, 6, 10, 11, 12, 13, 14, 15, 16)
    }


def test_registered_population_survives_complete_page_repair_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory = _registered_card_population_inventory()
    materializable_ordinals = (1, 2, 3, 4, 7, 8, 9)
    context = SimpleNamespace(
        pages=[],
        corrected_evidence_pages=lambda: [],
        reading_order_by_logical={80: 1, 81: 2, 82: 3},
        reading_order_resolution={"resolved": True, "authoritative": True},
        _personal_detail_extraction_issues=[],
        _business_repair_active=False,
    )

    def phase_skeletons(_context: object) -> list[dict[str, object]]:
        rows = [
            _registered_card_skeleton(inventory, ordinal)
            for ordinal in materializable_ordinals
        ]
        if context._business_repair_active:
            for row in rows:
                ref = row["source_refs"][0]
                ref["bbox"] = [value + 1.0 for value in ref["bbox"]]
                ref["evidence_ids"] = [
                    f"complete-page-repair:{row['category_sequence']}"
                ]
        return rows

    monkeypatch.setattr(
        native_extraction,
        "_account_anchor_skeletons",
        phase_skeletons,
    )
    monkeypatch.setattr(
        native_extraction,
        "_repair_complete_account_anchor_skeletons",
        lambda _context, rows: rows,
    )
    monkeypatch.setattr(
        native_extraction,
        "_registered_account_section_plane",
        lambda *_args, **_kwargs: (
            None
            if context._business_repair_active
            else deepcopy(inventory)
        ),
    )
    monkeypatch.setattr(
        native_extraction,
        "_exact_account_table_cell_anchors",
        lambda _context: {},
    )
    monkeypatch.setattr(
        native_extraction,
        "_sealed_raw_account_population_census",
        lambda _context: None,
    )
    monkeypatch.setattr(
        native_extraction,
        "_sealed_agreement_population_census",
        lambda _context: None,
    )
    monkeypatch.setattr(
        native_extraction,
        "_inquiry_source_coverage",
        lambda _context: {},
    )

    discovery_ledger = native_extraction._source_completeness_ledger(context)
    context._business_repair_active = True
    repaired_ledger = native_extraction._source_completeness_ledger(context)

    assert discovery_ledger["account_family_endpoints"] == {"credit_card": 16}
    assert repaired_ledger["credit_accounts"] == 16
    assert repaired_ledger["account_family_endpoints"] == {"credit_card": 16}
    observations = repaired_ledger["account_family_ordinal_observations"][
        "credit_card"
    ]
    assert set(observations) == {str(ordinal) for ordinal in range(1, 17)}
    assert all(
        "printed_fields" not in observations[str(ordinal)]
        and "field_source_refs" not in observations[str(ordinal)]
        for ordinal in (5, 6, 10, 11, 12, 13, 14, 15, 16)
    )
    assert observations["16"]["source_refs"][0]["evidence_ids"] == [
        "registered-card:16"
    ]


def test_candidate_b_two_pass_preserves_registered_population_for_final_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from docmirror.plugins.credit_report.personal_detail_scanned.candidate_b import (
        CandidateBPipeline,
    )

    inventory = _registered_card_population_inventory()
    materializable_ordinals = (1, 2, 3, 4, 7, 8, 9)

    class StaticTwoPassContext:
        def __init__(self) -> None:
            self.pages: list[object] = []
            self.reading_order_by_logical = {80: 1, 81: 2, 82: 3}
            self.reading_order_resolution = {
                "resolved": True,
                "authoritative": True,
            }
            self._personal_detail_extraction_issues: list[dict[str, object]] = []
            self._business_repair_active = False

        def account_collections(self):
            return (
                [
                    {
                        "account_id": f"credit_account:credit_card:{ordinal}",
                        "account_type": "credit_card",
                        "category_sequence": ordinal,
                    }
                    for ordinal in materializable_ordinals
                ],
                [],
                [],
            )

        def corrected_repayment_records(self):
            return []

        def corrected_repayment_micro_grids(self):
            return []

        def corrected_evidence_pages(self):
            return []

        def candidate_b_status_glyph_observations(self):
            return []

        def prepare_candidate_b_business_repair(self, _payload):
            self._business_repair_active = True
            self._personal_detail_extraction_issues = []
            return True

        def correct_candidate_b_datasets(self, payload):
            return deepcopy(payload)

        def canonical_layout_audit(self):
            return {}

        def page_topology_audit(self):
            return {}

        def ocr_correction_audit(self):
            return {
                "business_repair": {"second_schema_pass_required": True}
            }

    context = StaticTwoPassContext()

    def phase_skeletons(_context: object) -> list[dict[str, object]]:
        rows = [
            _registered_card_skeleton(inventory, ordinal)
            for ordinal in materializable_ordinals
        ]
        if context._business_repair_active:
            for row in rows:
                ref = row["source_refs"][0]
                ref["bbox"] = [value + 1.0 for value in ref["bbox"]]
                ref["evidence_ids"] = [
                    f"complete-page-repair:{row['category_sequence']}"
                ]
        return rows

    monkeypatch.setattr(
        native_extraction,
        "_account_anchor_skeletons",
        phase_skeletons,
    )
    monkeypatch.setattr(
        native_extraction,
        "_repair_complete_account_anchor_skeletons",
        lambda _context, rows: rows,
    )
    monkeypatch.setattr(
        native_extraction,
        "_registered_account_section_plane",
        lambda *_args, **_kwargs: (
            None
            if context._business_repair_active
            else deepcopy(inventory)
        ),
    )
    monkeypatch.setattr(
        native_extraction,
        "_exact_account_table_cell_anchors",
        lambda _context: {},
    )
    monkeypatch.setattr(
        native_extraction,
        "_sealed_raw_account_population_census",
        lambda _context: None,
    )
    monkeypatch.setattr(
        native_extraction,
        "_sealed_agreement_population_census",
        lambda _context: None,
    )
    monkeypatch.setattr(
        native_extraction,
        "_inquiry_source_coverage",
        lambda _context: {},
    )

    def empty(_context):
        return []

    for name in (
        "_extract_employment_records",
        "_extract_inquiries",
        "_extract_liabilities",
        "_extract_postpaid_payment_history",
        "_extract_postpaid_records",
        "_extract_public_records",
        "_extract_recovery_records",
        "_extract_residence_records",
        "_extract_source_rows",
    ):
        monkeypatch.setattr(native_extraction, name, empty)
    monkeypatch.setattr(
        native_extraction,
        "_extract_header_datasets",
        lambda _context, _text: {},
    )
    monkeypatch.setattr(
        native_extraction,
        "_extract_personal_notes",
        lambda _context: ([], []),
    )
    monkeypatch.setattr(
        native_extraction,
        "_extract_profile_detail_records",
        lambda _context: {},
    )
    monkeypatch.setattr(
        native_extraction,
        "_extract_summary_datasets",
        lambda _context: ([], []),
    )
    monkeypatch.setattr(
        "docmirror.plugins.credit_report.personal_detail_scanned.profile_extraction.extract_candidate_b_profile",
        lambda _context: {},
    )
    monkeypatch.setattr(
        "docmirror.plugins.credit_report.personal_detail_scanned.relations.candidate_b_repayment_anchor_ledger",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        "docmirror.plugins.credit_report.personal_detail_scanned.relations.link_candidate_b_repayments",
        lambda rows, *_args, **_kwargs: rows,
    )
    monkeypatch.setattr(
        "docmirror.plugins.credit_report.personal_detail_scanned.relations.derive_candidate_b_overdue_records",
        lambda *_args: [],
    )
    monkeypatch.setattr(
        "docmirror.plugins.credit_report.personal_detail_scanned.consistency_ledger.apply_document_consistency_ledger",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        "docmirror.plugins.credit_report.personal_detail_scanned.document_glyph_bank.apply_document_local_status_glyph_bank",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        "docmirror.plugins.credit_report.personal_detail_scanned.native_status_conflict.apply_candidate_b_native_status_conflict_guard",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        "docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues.register_final_liability_issue_records",
        lambda *_args, **_kwargs: None,
    )

    projected: dict[str, object] = {}

    def capture_projection(content, business, **kwargs):
        result = prepare_personal_detail_source_collections(
            content,
            business,
            **kwargs,
        )
        projected.update(deepcopy(result))
        return result

    monkeypatch.setattr(
        "docmirror.plugins.credit_report.personal_detail_scanned.source_projection.prepare_personal_detail_source_collections",
        capture_projection,
    )

    result = CandidateBPipeline(context, "").run()

    assert result.audit["schema_extraction_pass_count"] == 2
    assert len(result.business["credit_accounts"]) == len(materializable_ordinals)
    ledger = projected["facts"]["personal_detail_source_completeness_ledger"]
    assert ledger["credit_accounts"] == 16
    assert ledger["account_family_endpoints"] == {"credit_card": 16}
    omissions = [
        issue
        for issue in projected["datasets"]["personal_detail_extraction_issues"]
        if issue.get("issue_code") == "source_account_record_omitted"
    ]
    assert {issue["target_record_id"] for issue in omissions} == {
        f"credit_account:credit_card:{ordinal}"
        for ordinal in (5, 6, 10, 11, 12, 13, 14, 15, 16)
    }


@pytest.mark.parametrize(
    "mutation",
    ("replayed_evidence", "reversed_order", "foreign_owner"),
)
def test_registered_population_lifecycle_snapshot_fails_closed_when_mutated(
    mutation: str,
) -> None:
    inventory = _registered_card_population_inventory(count=3)
    context = SimpleNamespace(_business_repair_active=False)

    discovery = native_extraction._registered_account_population_observations(
        inventory
    )
    assert discovery is not None
    assert (
        native_extraction._registered_account_population_lifecycle_observations(
            context,
            discovery,
            inventory,
        )
        is not None
    )
    snapshot = context.__dict__[
        "_candidate_b_pre_repair_registered_account_population_observations"
    ]
    if mutation == "replayed_evidence":
        snapshot["credit_card"][3]["source_refs"][0]["evidence_ids"] = [
            "registered-card:2"
        ]
    elif mutation == "reversed_order":
        first_ref = snapshot["credit_card"][1]["source_refs"][0]
        second_ref = snapshot["credit_card"][2]["source_refs"][0]
        first_ref["bbox"], second_ref["bbox"] = (
            list(second_ref["bbox"]),
            list(first_ref["bbox"]),
        )
    else:
        foreign_ref = snapshot["credit_card"][1]["source_refs"][0]
        foreign_ref["logical_page"] = 999
        foreign_ref["source_page"] = 999
        foreign_ref["bbox"] = [1.0, 2.0, 3.0, 4.0]
        foreign_ref["evidence_ids"] = ["foreign:account-anchor"]
    context._business_repair_active = True

    assert (
        native_extraction._registered_account_population_lifecycle_observations(
            context,
            None,
            None,
        )
        is None
    )


def test_registered_population_lifecycle_reseals_cross_pass_family_merge() -> None:
    discovery_inventory = _registered_card_population_inventory(count=1)
    context = SimpleNamespace(_business_repair_active=False)
    discovery = native_extraction._registered_account_population_observations(
        discovery_inventory
    )
    assert discovery is not None
    assert (
        native_extraction._registered_account_population_lifecycle_observations(
            context,
            discovery,
            discovery_inventory,
        )
        is not None
    )

    repair_inventory = deepcopy(discovery_inventory)
    repair_inventory[1][80][0]["account_type"] = "quasi_credit_card"
    current = native_extraction._registered_account_population_observations(
        repair_inventory
    )
    assert current is not None
    context._business_repair_active = True

    assert (
        native_extraction._registered_account_population_lifecycle_observations(
            context,
            current,
            repair_inventory,
        )
        is None
    )


def test_registered_population_rejects_promoted_logical_copy_of_physical_anchor() -> None:
    bbox = [20.0, 40.0, 120.0, 55.0]
    inventory = (
        [
            {"page": 80, "source_page": 40},
            {"page": 81, "source_page": 40},
        ],
        {
            80: [
                {
                    "text": "账户1",
                    "page": 80,
                    "source_page": 40,
                    "account_type": "credit_card",
                    "account_family_quality": "exact",
                    "bbox": list(bbox),
                    "evidence_ids": ["promoted-card:logical-80"],
                }
            ],
            81: [
                {
                    "text": "账户2",
                    "page": 81,
                    "source_page": 40,
                    "account_type": "credit_card",
                    "account_family_quality": "exact",
                    "bbox": list(bbox),
                    "evidence_ids": ["promoted-card:logical-81"],
                }
            ],
        },
    )

    assert native_extraction._registered_account_population_observations(
        inventory
    ) is None


def test_registered_population_preserves_two_up_anchors_on_one_source_page() -> None:
    inventory = (
        [{"page": 80, "source_page": 40}],
        {
            80: [
                {
                    "text": "账户1",
                    "page": 80,
                    "source_page": 40,
                    "account_type": "credit_card",
                    "account_family_quality": "exact",
                    "bbox": [20.0, 40.0, 120.0, 55.0],
                    "evidence_ids": ["two-up-card:1"],
                },
                {
                    "text": "账户2",
                    "page": 80,
                    "source_page": 40,
                    "account_type": "credit_card",
                    "account_family_quality": "exact",
                    "bbox": [300.0, 40.0, 400.0, 55.0],
                    "evidence_ids": ["two-up-card:2"],
                },
            ]
        },
    )

    observations = native_extraction._registered_account_population_observations(
        inventory
    )
    assert observations is not None
    assert sorted(observations["credit_card"]) == [1, 2]
    authority = native_extraction._registered_account_population_authority_seal(
        inventory
    )
    assert authority is not None
    assert (
        native_extraction._registered_account_population_from_authority_seal(
            authority
        )
        == observations
    )


def test_registered_population_authority_rejects_promoted_logical_copy() -> None:
    inventory = (
        [
            {"page": 80, "source_page": 40},
            {"page": 81, "source_page": 40},
        ],
        {
            80: [
                {
                    "text": "账户1",
                    "page": 80,
                    "source_page": 40,
                    "account_type": "credit_card",
                    "account_family_quality": "exact",
                    "bbox": [20.0, 40.0, 120.0, 55.0],
                    "evidence_ids": ["authority-copy:logical-80"],
                }
            ],
            81: [
                {
                    "text": "账户2",
                    "page": 81,
                    "source_page": 40,
                    "account_type": "credit_card",
                    "account_family_quality": "exact",
                    "bbox": [20.0, 70.0, 120.0, 85.0],
                    "evidence_ids": ["authority-copy:logical-81"],
                }
            ],
        },
    )
    authority = native_extraction._registered_account_population_authority_seal(
        inventory
    )
    assert authority is not None
    payload, _digest = authority
    version, page_order, owners = payload
    first_owner, second_owner = owners
    promoted_second_owner = (
        *second_owner[:5],
        first_owner[5],
        second_owner[6],
    )
    promoted_payload = (
        version,
        page_order,
        (first_owner, promoted_second_owner),
    )
    promoted_authority = (
        promoted_payload,
        native_extraction._registered_account_population_authority_digest(
            promoted_payload
        ),
    )

    assert (
        native_extraction._registered_account_population_from_authority_seal(
            promoted_authority
        )
        is None
    )


def test_registered_population_lifecycle_rejects_cross_family_promoted_copy() -> None:
    discovery_inventory = (
        [{"page": 80, "source_page": 40}],
        {
            80: [
                {
                    "text": "账户1",
                    "page": 80,
                    "source_page": 40,
                    "account_type": "credit_card",
                    "account_family_quality": "exact",
                    "bbox": [20.0, 40.0, 120.0, 55.0],
                    "evidence_ids": ["discovery-copy:card"],
                }
            ]
        },
    )
    context = SimpleNamespace(_business_repair_active=False)
    discovery = native_extraction._registered_account_population_observations(
        discovery_inventory
    )
    assert discovery is not None
    assert (
        native_extraction._registered_account_population_lifecycle_observations(
            context,
            discovery,
            discovery_inventory,
        )
        is not None
    )

    repair_inventory = (
        [{"page": 81, "source_page": 40}],
        {
            81: [
                {
                    "text": "账户1",
                    "page": 81,
                    "source_page": 40,
                    "account_type": "quasi_credit_card",
                    "account_family_quality": "exact",
                    "bbox": [20.0, 40.0, 120.0, 55.0],
                    "evidence_ids": ["repair-copy:quasi-card"],
                }
            ]
        },
    )
    current = native_extraction._registered_account_population_observations(
        repair_inventory
    )
    assert current is not None
    context._business_repair_active = True

    assert (
        native_extraction._registered_account_population_lifecycle_observations(
            context,
            current,
            repair_inventory,
        )
        is None
    )


@pytest.mark.parametrize(
    "mutation",
    ("duplicate_ordinal", "replayed_evidence", "duplicate_physical_owner", "reversed_order"),
)
def test_registered_section_population_fails_closed_on_owner_conflicts(
    mutation: str,
) -> None:
    inventory = _registered_card_population_inventory(count=3)
    lines = inventory[1][80]
    if mutation == "duplicate_ordinal":
        duplicate = deepcopy(lines[0])
        duplicate["bbox"] = [20.0, 140.0, 120.0, 155.0]
        duplicate["evidence_ids"] = ["registered-card:duplicate"]
        lines.append(duplicate)
    elif mutation == "replayed_evidence":
        lines[1]["evidence_ids"] = list(lines[0]["evidence_ids"])
    elif mutation == "duplicate_physical_owner":
        lines[1]["bbox"] = list(lines[0]["bbox"])
    elif mutation == "reversed_order":
        lines[0]["bbox"], lines[1]["bbox"] = (
            list(lines[1]["bbox"]),
            list(lines[0]["bbox"]),
        )

    assert native_extraction._registered_account_population_observations(
        inventory
    ) is None


def _typed_skeleton_from_raw_anchor(context: SimpleNamespace, *, evidence_id: str) -> dict[str, object]:
    anchor = context._frozen_logical_pages[41].texts[2]
    return {
        "account_id": "credit_account:non_revolving_loan:1",
        "account_type": "non_revolving_loan",
        "account_family_quality": "exact",
        "category_sequence": 1,
        "_printed_ordinal_status": "printed_unique",
        "_canonical_segment": {"ownership_basis": "printed_anchor_to_next_anchor"},
        "source_refs": [
            {
                "source": "candidate_b_account_anchor",
                "logical_page": 41,
                "source_page": 23,
                "bbox": list(anchor.bbox),
                "evidence_ids": [evidence_id],
            }
        ],
    }


def test_raw_and_typed_anchor_dedupe_only_on_exact_immutable_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _dense_context(count=1)
    skeleton = _typed_skeleton_from_raw_anchor(context, evidence_id="anchor:1")
    monkeypatch.setattr(native_extraction, "_account_anchor_skeletons", lambda _context: [deepcopy(skeleton)])
    monkeypatch.setattr(
        native_extraction,
        "_repair_complete_account_anchor_skeletons",
        lambda _context, skeletons: skeletons,
    )

    ledger = native_extraction._source_completeness_ledger(context)

    assert ledger["credit_accounts"] == 1
    observation = ledger["account_family_ordinal_observations"]["non_revolving_loan"]["1"]
    assert len(observation["source_refs"]) == 1
    assert "account_raw_anchor_ordinal_conflicts" not in ledger


def test_frozen_raw_anchor_population_cannot_be_vetoed_by_a_repair_plane_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _dense_context(count=1)
    skeleton = _typed_skeleton_from_raw_anchor(context, evidence_id="different-owner")
    monkeypatch.setattr(native_extraction, "_account_anchor_skeletons", lambda _context: [deepcopy(skeleton)])
    monkeypatch.setattr(
        native_extraction,
        "_repair_complete_account_anchor_skeletons",
        lambda _context, skeletons: skeletons,
    )

    ledger = native_extraction._source_completeness_ledger(context)

    assert ledger["credit_accounts"] == 1
    assert ledger["account_family_endpoints"] == {"non_revolving_loan": 1}
    observation = ledger["account_family_ordinal_observations"][
        "non_revolving_loan"
    ]["1"]
    assert observation["source_refs"][0]["evidence_ids"] == ["anchor:1"]
    assert "printed_fields" not in observation
    assert "field_source_refs" not in observation
    assert "account_raw_anchor_ordinal_conflicts" not in ledger


@pytest.mark.parametrize(
    ("count", "scale", "reverse"),
    ((2, 0.55, False), (6, 1.0, True), (19, 1.8, False)),
)
def test_frozen_dense_family_replaces_variable_repair_plane_observations(
    monkeypatch: pytest.MonkeyPatch,
    count: int,
    scale: float,
    reverse: bool,
) -> None:
    context = _dense_context(count=count, scale=scale)
    page = context._frozen_logical_pages[41]
    skeletons = []
    for ordinal in range(1, count + 1):
        anchor = page.texts[ordinal + 1]
        skeletons.append(
            {
                "account_id": f"credit_account:non_revolving_loan:{ordinal}",
                "account_type": "non_revolving_loan",
                "account_family_quality": "exact",
                "category_sequence": ordinal,
                "_printed_ordinal_status": "printed_unique",
                "_canonical_segment": {
                    "ownership_basis": "printed_anchor_to_next_anchor"
                },
                "source_refs": [
                    {
                        "source": "candidate_b_account_anchor",
                        "logical_page": 41,
                        "source_page": 23,
                        "bbox": [
                            float(anchor.bbox[0]) + scale,
                            float(anchor.bbox[1]),
                            float(anchor.bbox[2]) + scale,
                            float(anchor.bbox[3]),
                        ],
                        "evidence_ids": [f"repair-generation:{ordinal}"],
                    }
                ],
            }
        )
    if reverse:
        skeletons.reverse()
    monkeypatch.setattr(
        native_extraction,
        "_account_anchor_skeletons",
        lambda _context: deepcopy(skeletons),
    )
    monkeypatch.setattr(
        native_extraction,
        "_repair_complete_account_anchor_skeletons",
        lambda _context, rows: rows,
    )

    ledger = native_extraction._source_completeness_ledger(context)

    assert ledger["credit_accounts"] == count
    assert ledger["account_family_endpoints"] == {
        "non_revolving_loan": count
    }
    assert ledger["account_raw_anchor_observed_sequences"] == {
        "non_revolving_loan": list(range(1, count + 1))
    }
    assert set(
        ledger["account_family_ordinal_observations"]["non_revolving_loan"]
    ) == {str(ordinal) for ordinal in range(1, count + 1)}
    assert all(
        observation["source_refs"][0]["evidence_ids"] == [f"anchor:{ordinal}"]
        and "printed_fields" not in observation
        and "field_source_refs" not in observation
        for ordinal, observation in (
            (int(raw_ordinal), raw_observation)
            for raw_ordinal, raw_observation in ledger[
                "account_family_ordinal_observations"
            ]["non_revolving_loan"].items()
        )
    )
    assert "account_raw_anchor_ordinal_conflicts" not in ledger


@pytest.mark.parametrize(
    ("count", "emitted_ordinals"),
    ((3, ()), (6, (1, 3, 6)), (17, (2, 4, 8, 16))),
)
def test_frozen_family_population_is_exactly_conserved_as_localized_omissions(
    count: int,
    emitted_ordinals: tuple[int, ...],
) -> None:
    ledger = native_extraction._source_completeness_ledger(
        _dense_context(count=count)
    )
    content = prepare_personal_detail_source_collections(
        {
            "facts": {"personal_detail_source_completeness_ledger": ledger},
            "datasets": {
                "credit_accounts": [
                    {
                        "account_id": (
                            f"credit_account:non_revolving_loan:{ordinal}"
                        ),
                        "account_type": "non_revolving_loan",
                        "category_sequence": ordinal,
                    }
                    for ordinal in reversed(emitted_ordinals)
                ]
            },
        }
    )
    omissions = [
        issue
        for issue in content["datasets"]["personal_detail_extraction_issues"]
        if issue.get("issue_code") == "source_account_record_omitted"
    ]

    assert len(emitted_ordinals) + len(omissions) == count
    assert {issue["target_record_id"] for issue in omissions} == {
        f"credit_account:non_revolving_loan:{ordinal}"
        for ordinal in range(1, count + 1)
        if ordinal not in emitted_ordinals
    }
    assert all(
        issue["source_refs"][0]["binding"] == "printed_account_ordinal"
        and issue["source_refs"][0]["evidence_ids"]
        == [f"anchor:{issue['observed_value']['category_sequence']}"]
        for issue in omissions
    )


def _projection_ref(evidence_ids: object) -> dict[str, object]:
    return {
        "source": "candidate_b_account_anchor",
        "logical_page": 41,
        "source_page": 23,
        "geometry_scope": "line",
        "binding": "printed_account_ordinal",
        "binding_quality": "printed_account_ordinal",
        "account_type": "non_revolving_loan",
        "category_sequence": 1,
        "bbox": [10.0, 20.0, 80.0, 40.0],
        "evidence_ids": evidence_ids,
    }


def _projection_issues(ref: dict[str, object]) -> list[dict[str, object]]:
    ledger = {
        "account_family_endpoints": {"non_revolving_loan": 1},
        "account_family_ordinal_observations": {
            "non_revolving_loan": {
                "1": {
                    "account_id": "credit_account:non_revolving_loan:1",
                    "source_refs": [ref],
                }
            }
        },
    }
    content = prepare_personal_detail_source_collections(
        {
            "facts": {"personal_detail_source_completeness_ledger": ledger},
            "datasets": {"credit_accounts": []},
        }
    )
    return [
        issue
        for issue in content["datasets"]["personal_detail_extraction_issues"]
        if issue.get("issue_code") == "source_account_record_omitted"
    ]


def test_projection_accepts_only_the_exact_raw_account_ordinal_contract() -> None:
    assert len(_projection_issues(_projection_ref(["anchor:1"]))) == 1


@pytest.mark.parametrize(
    "evidence_ids",
    [[], [""], [7], ["anchor:1", "anchor:1"], ("anchor:1",)],
)
def test_projection_rejects_malformed_or_replayed_raw_anchor_ids(
    evidence_ids: object,
) -> None:
    assert _projection_issues(_projection_ref(evidence_ids)) == []


def test_projection_rejects_an_anchor_ref_without_the_printed_ordinal_binding() -> None:
    ref = _projection_ref(["anchor:1"])
    ref.pop("binding")

    assert _projection_issues(ref) == []
