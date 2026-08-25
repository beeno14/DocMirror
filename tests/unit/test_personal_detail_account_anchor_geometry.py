from __future__ import annotations

from types import SimpleNamespace

import pytest

from docmirror.plugins.credit_report.personal_detail_scanned import native_extraction


def _line(
    text: str,
    bbox: list[float],
    *,
    page: int = 1,
    source_page: int = 1,
    evidence: str | None = None,
) -> dict:
    return {
        "text": text,
        "bbox": bbox,
        "page": page,
        "source_page": source_page,
        "evidence_ids": [evidence or f"e:{page}:{bbox[0]}:{bbox[1]}"],
    }


def _card_header(y: float, *, page: int = 1, source_page: int = 1) -> list[dict]:
    labels = (
        ("发卡机构", 20, 70),
        ("账户标识", 70, 125),
        ("开立日期", 125, 175),
        ("账户授信额度", 175, 225),
        ("共享授信额度", 225, 275),
        ("币种", 275, 315),
        ("业务种类", 315, 360),
        ("担保方式", 360, 410),
    )
    return [_line(text, [left, y, right, y + 12], page=page, source_page=source_page) for text, left, right in labels]


def _card_values(
    y: float,
    *,
    page: int = 1,
    source_page: int = 1,
    currency_x: tuple[float, float] = (278, 312),
) -> list[dict]:
    return [
        _line("中国工商银行", [22, y, 68, y + 9], page=page, source_page=source_page),
        _line("B10111000H", [76, y, 121, y + 9], page=page, source_page=source_page),
        _line("2021.05.31", [130, y, 170, y + 9], page=page, source_page=source_page),
        _line("50,000", [184, y, 216, y + 9], page=page, source_page=source_page),
        _line("--", [242, y, 255, y + 9], page=page, source_page=source_page),
        _line(
            "人民币元",
            [currency_x[0], y, currency_x[1], y + 9],
            page=page,
            source_page=source_page,
        ),
        _line("贷记卡", [322, y, 350, y + 9], page=page, source_page=source_page),
        _line("信用/免担保", [364, y, 408, y + 9], page=page, source_page=source_page),
        _line("股份有限公司", [22, y + 10, 68, y + 19], page=page, source_page=source_page),
        _line("00014100000", [76, y + 10, 121, y + 19], page=page, source_page=source_page),
        _line("厦门市分行", [22, y + 20, 68, y + 29], page=page, source_page=source_page),
        _line("02100474560", [76, y + 20, 121, y + 29], page=page, source_page=source_page),
        _line("01", [90, y + 30, 105, y + 39], page=page, source_page=source_page),
    ]


def _single_card_context(lines: list[dict], *, transition=True) -> SimpleNamespace:
    pages: dict[int, dict] = {}
    for line in lines:
        page = int(line["page"])
        pages.setdefault(
            page,
            {
                "page": page,
                "source_page": int(line["source_page"]),
                "lines": [],
            },
        )["lines"].append(line)
    return SimpleNamespace(
        corrected_evidence_pages=lambda: list(pages.values()),
        allows_scanned_line_transition=lambda *_args: transition,
        _personal_detail_extraction_issues=[],
    )


def _table_observation(*, top: float = 38) -> dict:
    return {
        "account_id": "credit_account_table_observation:test",
        "_table_observation_id": "credit_account_table_observation:test",
        "account_type": "credit_card",
        "source": "native_detail_account_table",
        "source_refs": [
            {
                "source": "native_detail_table",
                "logical_page": 1,
                "source_page": 1,
                "table_id": "pt_test",
                "bbox": [18, top, 412, 240],
            }
        ],
        "canonical_raw": {},
    }


def _extract_one(monkeypatch, lines: list[dict], *, table: dict | None = None) -> tuple[dict, SimpleNamespace]:
    context = _single_card_context(lines)
    monkeypatch.setattr(
        native_extraction,
        "_extract_table_accounts",
        lambda _context: (([table] if table is not None else []), [], []),
    )
    accounts, _, _ = native_extraction._extract_accounts(context)
    assert len(accounts) == 1
    return accounts[0], context


def _issue_fields(context: SimpleNamespace, issue_code: str) -> list[str]:
    return [
        str(issue.get("field_name") or "")
        for issue in context._personal_detail_extraction_issues
        if issue.get("issue_code") == issue_code
    ]


def _loan_header(y: float) -> list[dict]:
    return [
        _line(text, [left, y, right, y + 12])
        for text, left, right in (
            ("管理机构", 20, 80),
            ("账户标识", 80, 145),
            ("开立日期", 145, 200),
            ("到期日期", 200, 255),
            ("借款金额", 255, 330),
            ("账户币种", 330, 400),
        )
    ]


def _loan_values(y: float, *, include_open_date: bool = True) -> list[dict]:
    values = [
        _line("五矿国际信托有限公司", [22, y, 76, y + 9]),
        _line("J10158510H000110000000640557", [84, y, 141, y + 9]),
        _line("2031.05.31", [205, y, 250, y + 9]),
        _line("140,000", [265, y, 320, y + 9]),
        _line("人民币元", [340, y, 390, y + 9]),
    ]
    if include_open_date:
        values.insert(2, _line("2021.01.12", [150, y, 195, y + 9]))
    return values


def _r2_header(y: float) -> list[dict]:
    return [
        _line(text, [left, y, right, y + 12])
        for text, left, right in (
            ("管理机构", 20, 80),
            ("账户标识", 80, 145),
            ("开立日期", 145, 200),
            ("到期日期", 200, 255),
            ("账户授信额度", 255, 330),
            ("账户币种", 330, 400),
        )
    ]


def _loan_second_header(y: float) -> list[dict]:
    return [
        _line(text, [left, y, right, y + 12])
        for text, left, right in (
            ("业务种类", 20, 85),
            ("担保方式", 85, 145),
            ("还款期数", 145, 205),
            ("还款频率", 205, 270),
            ("还款方式", 270, 340),
            ("共同借款标志", 340, 410),
        )
    ]


def _loan_second_values(y: float) -> list[dict]:
    return [
        _line("个人消费贷款", [22, y, 82, y + 9]),
        _line("抵押", [95, y, 135, y + 9]),
        _line("36", [165, y, 190, y + 9]),
        _line("月", [225, y, 250, y + 9]),
        _line("等额本息", [280, y, 330, y + 9]),
        _line("否", [360, y, 390, y + 9]),
    ]


def _with_duplicate_identifier_fragment(values: list[dict]) -> list[dict]:
    fragment = next(value for value in values if value["text"] == "00014100000")
    return [
        *values,
        {**fragment, "evidence_ids": list(fragment["evidence_ids"])},
    ]


def test_anchor_interval_recovers_basic_fields_when_line_grid_misses_the_table(
    monkeypatch,
) -> None:
    lines = [
        _line("贷记卡账户", [20, 10, 120, 22]),
        _line("账户1（授信协议标识：A00000001）", [20, 25, 250, 36]),
        *_card_header(40),
        *_card_values(55),
        _line("截至2022年12月14日", [160, 100, 270, 112]),
        _line("账户状态", [20, 120, 70, 132]),
    ]
    context = _single_card_context(lines)
    monkeypatch.setattr(
        native_extraction,
        "_extract_table_accounts",
        lambda _context: ([], [], []),
    )

    accounts, repayments, events = native_extraction._extract_accounts(context)

    assert repayments == []
    assert events == []
    assert len(accounts) == 1
    account = accounts[0]
    assert account["management_institution"] == "中国工商银行股份有限公司厦门市分行"
    assert account["account_identifier"] == "B10111000H000141000000210047456001"
    assert account["open_date"] == "2021-05-31"
    assert account["credit_limit"] == 50_000
    assert "shared_credit_limit" not in account
    assert "shared_credit_limit" in account["_source_absent_fields"]
    assert account["currency"] == "CNY"
    assert account["account_currency"] == "CNY"
    assert account["business_type"] == "贷记卡"
    assert account["guarantee_type"] == "信用/免担保"
    assert all(
        ref["binding"] == "canonical_account_header_geometry"
        for field_name in (
            "management_institution",
            "account_identifier",
            "open_date",
            "credit_limit",
            "currency",
            "business_type",
            "guarantee_type",
        )
        for ref in account["source_refs_by_field"][field_name]
    )
    assert any(
        issue["issue_code"] == "candidate_b_account_table_missing"
        for issue in context._personal_detail_extraction_issues
    )


def test_anchor_interval_currency_residue_is_withheld_with_exact_field_issues(
    monkeypatch,
) -> None:
    values = _card_values(55)
    currency_line = next(line for line in values if line["text"] == "人民币元")
    currency_line["text"] = "非人民币"
    lines = [
        _line("贷记卡账户", [20, 10, 120, 22]),
        _line("账户1（授信协议标识：A00000001）", [20, 25, 250, 36]),
        *_card_header(40),
        *values,
        _line("截至2022年12月14日", [160, 100, 270, 112]),
    ]

    account, context = _extract_one(monkeypatch, lines)

    assert "currency" not in account
    assert "account_currency" not in account
    assert account["canonical_raw"]["currency"] == ["非人民币"]
    assert account["canonical_raw"]["account_currency"] == ["非人民币"]
    issues = [
        issue
        for issue in context._personal_detail_extraction_issues
        if issue.get("issue_code") == "candidate_b_exact_slot_value_invalid"
        and issue.get("field_name") in {"currency", "account_currency"}
    ]
    assert {issue["field_name"] for issue in issues} == {"currency", "account_currency"}
    assert all(issue["observed_value"] == ["非人民币"] for issue in issues)
    assert all(issue["status"] == "requires_review" for issue in issues)


def test_anchor_interval_enriches_collapsed_table_without_required_field_issues(
    monkeypatch,
) -> None:
    lines = [
        _line("贷记卡账户", [20, 10, 120, 22]),
        _line("账户1（授信协议标识：A00000001）", [20, 25, 250, 36]),
        *_card_header(40),
        *_card_values(55),
        _line("截至2022年12月14日", [160, 100, 270, 112]),
    ]
    context = _single_card_context(lines)
    monkeypatch.setattr(
        native_extraction,
        "_extract_table_accounts",
        lambda _context: ([_table_observation()], [], []),
    )

    accounts, _, _ = native_extraction._extract_accounts(context)

    assert len(accounts) == 1
    assert accounts[0]["management_institution"].endswith("厦门市分行")
    assert accounts[0]["account_identifier"].endswith("01")
    assert accounts[0]["open_date"] == "2021-05-31"
    assert accounts[0]["credit_limit"] == 50_000
    assert accounts[0]["account_currency"] == "CNY"
    assert accounts[0]["business_type"] == "贷记卡"
    assert accounts[0]["guarantee_type"] == "信用/免担保"
    assert not any(
        issue["issue_code"] == "candidate_b_account_required_field_unresolved"
        and issue.get("field_name") in {"management_institution", "account_identifier", "open_date", "currency"}
        for issue in context._personal_detail_extraction_issues
    )


def test_anchor_interval_joins_bottom_header_to_verified_next_page_value_row(
    monkeypatch,
) -> None:
    lines = [
        _line("贷记卡账户", [20, 660, 120, 672]),
        _line("账户17（授信协议标识：A00000017）", [20, 680, 250, 692]),
        *_card_header(700),
        _line("第21页，共30页", [180, 760, 250, 772]),
        *_card_values(20, page=2, source_page=2),
        _line("截至2022年12月14日", [160, 66, 270, 78], page=2, source_page=2),
        _line("账户18", [20, 200, 80, 212], page=2, source_page=2),
    ]
    context = _single_card_context(lines, transition=True)
    monkeypatch.setattr(
        native_extraction,
        "_extract_table_accounts",
        lambda _context: ([], [], []),
    )

    accounts, _, _ = native_extraction._extract_accounts(context)

    assert accounts[0]["account_identifier"] == "B10111000H000141000000210047456001"
    assert accounts[0]["open_date"] == "2021-05-31"
    assert accounts[0]["currency"] == "CNY"
    assert {ref["logical_page"] for ref in accounts[0]["source_refs_by_field"]["currency"]} == {2}


def test_anchor_interval_does_not_cross_unverified_page_boundary(monkeypatch) -> None:
    lines = [
        _line("贷记卡账户", [20, 660, 120, 672]),
        _line("账户1", [20, 680, 80, 692]),
        *_card_header(700),
        *_card_values(20, page=2, source_page=2),
        _line("截至2022年12月14日", [160, 66, 270, 78], page=2, source_page=2),
    ]
    context = _single_card_context(lines, transition=None)
    monkeypatch.setattr(
        native_extraction,
        "_extract_table_accounts",
        lambda _context: ([], [], []),
    )

    accounts, _, _ = native_extraction._extract_accounts(context)

    assert "account_identifier" not in accounts[0]
    assert "open_date" not in accounts[0]
    assert "currency" not in accounts[0]


def test_anchor_interval_does_not_shift_neighbor_column_currency(monkeypatch) -> None:
    lines = [
        _line("贷记卡账户", [20, 10, 120, 22]),
        _line("账户1", [20, 25, 80, 36]),
        *_card_header(40),
        # Put the visible currency token in the business-type column.  It must
        # not jump left into the empty currency slot.
        *_card_values(55, currency_x=(320, 350)),
        _line("截至2022年12月14日", [160, 100, 270, 112]),
    ]
    context = _single_card_context(lines)
    monkeypatch.setattr(
        native_extraction,
        "_extract_table_accounts",
        lambda _context: ([_table_observation()], [], []),
    )

    accounts, _, _ = native_extraction._extract_accounts(context)

    assert "currency" not in accounts[0]
    assert "account_currency" not in accounts[0]
    assert any(
        issue["issue_code"] == "candidate_b_exact_slot_value_invalid" and issue.get("field_name") == "account_currency"
        for issue in context._personal_detail_extraction_issues
    )


def test_anchor_interval_withholds_conflicting_open_dates(monkeypatch) -> None:
    lines = [
        _line("贷记卡账户", [20, 10, 120, 22]),
        _line("账户1", [20, 25, 80, 36]),
        *_card_header(40),
        *_card_values(55),
        _line("2022.05.31", [130, 86, 170, 95]),
        _line("截至2022年12月14日", [160, 100, 270, 112]),
    ]
    context = _single_card_context(lines)
    monkeypatch.setattr(
        native_extraction,
        "_extract_table_accounts",
        lambda _context: ([_table_observation()], [], []),
    )

    accounts, _, _ = native_extraction._extract_accounts(context)

    assert "open_date" not in accounts[0]
    assert any(
        issue["issue_code"] == "candidate_b_exact_slot_value_conflict" and issue.get("field_name") == "open_date"
        for issue in context._personal_detail_extraction_issues
    )


@pytest.mark.parametrize(
    "mutator",
    (
        lambda values: [*values, _line("88", [90, 95, 105, 99])],
        _with_duplicate_identifier_fragment,
        lambda values: [
            *values[:-1],
            _line("8888", [76, 80, 121, 84]),
            values[-1],
        ],
        lambda values: [
            *values[:-1],
            _line("88888888", [80.5, 80, 116.5, 84]),
            values[-1],
        ],
    ),
)
def test_identifier_wrap_rejects_extra_or_replayed_fragments(monkeypatch, mutator) -> None:
    values = _card_values(55)
    lines = [
        _line("贷记卡账户", [20, 10, 120, 22]),
        _line("账户1", [20, 25, 80, 36]),
        *_card_header(40),
        *mutator(values),
        _line("截至2022年12月24日", [160, 105, 270, 117]),
    ]

    account, context = _extract_one(monkeypatch, lines)

    assert "account_identifier" not in account
    assert "account_identifier" in _issue_fields(
        context,
        "candidate_b_exact_slot_value_invalid",
    )


def test_identifier_wrap_requires_evidence_order_to_match_physical_order(monkeypatch) -> None:
    values = _card_values(55)
    suffix_one = next(value for value in values if value["text"] == "00014100000")
    suffix_two = next(value for value in values if value["text"] == "02100474560")
    first_index = values.index(suffix_one)
    second_index = values.index(suffix_two)
    values[first_index], values[second_index] = values[second_index], values[first_index]
    lines = [
        _line("贷记卡账户", [20, 10, 120, 22]),
        _line("账户1", [20, 25, 80, 36]),
        *_card_header(40),
        *values,
        _line("截至2022年12月24日", [160, 105, 270, 117]),
    ]

    account, context = _extract_one(monkeypatch, lines)

    assert "account_identifier" not in account
    assert "account_identifier" in _issue_fields(
        context,
        "candidate_b_exact_slot_value_invalid",
    )


def test_identifier_accepts_one_exact_typed_box(monkeypatch) -> None:
    values = [
        value for value in _card_values(55) if value["text"] not in {"B10111000H", "00014100000", "02100474560", "01"}
    ]
    values.append(
        _line(
            "B10111000H000141000000210047456001",
            [76, 55, 121, 64],
        )
    )
    lines = [
        _line("贷记卡账户", [20, 10, 120, 22]),
        _line("账户1", [20, 25, 80, 36]),
        *_card_header(40),
        *values,
        _line("截至2022年12月24日", [160, 105, 270, 117]),
    ]

    account, _context = _extract_one(monkeypatch, lines)

    assert account["account_identifier"] == "B10111000H000141000000210047456001"


def test_identifier_accepts_generic_two_letter_prefix_and_one_glyph_final_wrap(
    monkeypatch,
) -> None:
    prefix = "RL2014082100004"
    final = "6"
    values = [value for value in _loan_values(55) if not str(value["text"]).startswith("J101")]
    values.extend(
        (
            _line(prefix, [84, 55, 141, 64]),
            _line(final, [105, 65, 111, 74]),
        )
    )
    lines = [
        _line("非循环贷账户", [20, 10, 140, 22]),
        _line("账户1", [20, 25, 80, 36]),
        *_loan_header(40),
        *values,
        _line("截至2022年12月24日", [160, 105, 270, 117]),
    ]

    account, _context = _extract_one(monkeypatch, lines)

    assert account["account_identifier"] == prefix + final


def test_duplicate_complete_headers_are_rejected_not_concatenated(monkeypatch) -> None:
    lines = [
        _line("贷记卡账户", [20, 10, 120, 22]),
        _line("账户1", [20, 25, 80, 36]),
        *_card_header(40),
        *_card_values(55),
        *_card_header(120),
        *_card_values(135),
        _line("截至2022年12月24日", [160, 185, 270, 197]),
    ]

    account, context = _extract_one(monkeypatch, lines)

    assert "account_identifier" not in account
    assert "management_institution" not in account
    assert set(_issue_fields(context, "candidate_b_exact_slot_value_invalid")) >= {
        "account_identifier",
        "management_institution",
        "business_type",
        "guarantee_type",
    }


def test_repeated_continuation_header_uses_only_populated_header_cluster(
    monkeypatch,
) -> None:
    lines = [
        _line("贷记卡账户", [20, 660, 120, 672]),
        _line("账户1", [20, 680, 80, 692]),
        *_card_header(700),
        *_card_header(20, page=2, source_page=2),
        *_card_values(35, page=2, source_page=2),
        _line("截至2022年12月24日", [160, 90, 270, 102], page=2, source_page=2),
    ]

    account, context = _extract_one(monkeypatch, lines)

    assert account["account_identifier"] == "B10111000H000141000000210047456001"
    assert account["management_institution"].endswith("厦门市分行")
    assert "account_identifier" not in _issue_fields(
        context,
        "candidate_b_exact_slot_value_invalid",
    )


def test_transposed_header_roles_withhold_all_geometry_fields(monkeypatch) -> None:
    header = _card_header(40)
    institution = next(line for line in header if line["text"] == "发卡机构")
    business = next(line for line in header if line["text"] == "业务种类")
    institution["bbox"], business["bbox"] = business["bbox"], institution["bbox"]
    lines = [
        _line("贷记卡账户", [20, 10, 120, 22]),
        _line("账户1", [20, 25, 80, 36]),
        *header,
        *_card_values(55),
        _line("截至2022年12月24日", [160, 105, 270, 117]),
    ]

    account, context = _extract_one(monkeypatch, lines, table=_table_observation())

    assert "management_institution" not in account
    assert "business_type" not in account
    assert set(_issue_fields(context, "candidate_b_exact_slot_value_invalid")) >= {
        "management_institution",
        "business_type",
    }


def test_combined_header_or_value_boxes_never_guess_individual_fields(monkeypatch) -> None:
    header = [line for line in _card_header(40) if line["text"] not in {"发卡机构", "账户标识"}]
    values = [
        line
        for line in _card_values(55)
        if line["text"] not in {"B10111000H", "00014100000", "02100474560", "01", "2021.05.31"}
    ]
    lines = [
        _line("贷记卡账户", [20, 10, 120, 22]),
        _line("账户1", [20, 25, 80, 36]),
        _line("发卡机构账户标识", [20, 40, 125, 52]),
        *header,
        _line(
            "B10111000H000141000000210047456001 2021.05.31",
            [75, 55, 170, 64],
        ),
        *values,
        _line("截至2022年12月24日", [160, 105, 270, 117]),
    ]

    account, context = _extract_one(monkeypatch, lines)

    assert "account_identifier" not in account
    assert "open_date" not in account
    assert set(_issue_fields(context, "candidate_b_exact_slot_value_invalid")) >= {
        "account_identifier",
        "open_date",
    }


def test_geometryless_composite_header_cannot_invent_equal_columns(monkeypatch) -> None:
    header = [line for line in _card_header(40) if line["text"] not in {"开立日期", "账户授信额度"}]
    lines = [
        _line("贷记卡账户", [20, 10, 120, 22]),
        _line("账户1", [20, 25, 80, 36]),
        *header,
        _line("开立日期 账户授信额度", [125, 40, 225, 52]),
        *_card_values(55),
        _line("截至2022年12月24日", [160, 105, 270, 117]),
    ]

    account, _context = _extract_one(monkeypatch, lines)

    assert "open_date" not in account
    assert "credit_limit" not in account


def test_exact_unequal_composite_header_tokens_bind_unique_roles(monkeypatch) -> None:
    header = [line for line in _card_header(40) if line["text"] not in {"开立日期", "账户授信额度"}]
    composite = _line(
        "开立日期账户授信额度",
        [125, 40, 225, 52],
        evidence="composite-header",
    )
    composite["evidence_ids"] = ["header-open", "header-limit"]
    lines = [
        _line("贷记卡账户", [20, 10, 120, 22]),
        _line("账户1", [20, 25, 80, 36]),
        *header,
        composite,
        *_card_values(55),
        _line("截至2022年12月24日", [160, 105, 270, 117]),
    ]
    context = _single_card_context(lines)
    context.evidence_plane = SimpleNamespace(
        evidence=SimpleNamespace(
            text_atoms=[
                {"id": "header-limit", "text": "账户授信额度", "bbox": [170, 40, 225, 52]},
                {"id": "header-open", "text": "开立日期", "bbox": [125, 40, 165, 52]},
            ]
        )
    )
    monkeypatch.setattr(
        native_extraction,
        "_extract_table_accounts",
        lambda _context: ([], [], []),
    )

    accounts, _, _ = native_extraction._extract_accounts(context)

    assert len(accounts) == 1
    assert accounts[0]["open_date"] == "2021-05-31"
    assert accounts[0]["credit_limit"] == 50_000


@pytest.mark.parametrize(
    ("role_order", "split"),
    [
        (("credit_limit", "open_date"), 0.7),
        (("open_date", "credit_limit"), 0.3),
    ],
)
def test_exact_reordered_scaled_composite_tokens_bind_semantic_roles(
    role_order: tuple[str, str],
    split: float,
) -> None:
    scale = 1.8
    labels = {"credit_limit": "账户授信额度", "open_date": "开立日期"}
    left = 40.0 * scale
    right = 180.0 * scale
    boundary = left + (right - left) * split
    line = _line(
        "".join(labels[role] for role in role_order),
        [left, 30 * scale, right, 44 * scale],
        page=9,
        source_page=4,
        evidence="unused",
    )
    line["evidence_ids"] = ["left-token", "right-token"]
    context = SimpleNamespace(
        evidence_plane=SimpleNamespace(
            evidence=SimpleNamespace(
                text_atoms=[
                    {
                        "id": "right-token",
                        "text": labels[role_order[1]],
                        "bbox": [boundary + scale, 30 * scale, right, 44 * scale],
                    },
                    {
                        "id": "left-token",
                        "text": labels[role_order[0]],
                        "bbox": [left, 30 * scale, boundary - scale, 44 * scale],
                    },
                ]
            )
        )
    )

    parts = native_extraction._account_composite_header_parts(
        context,
        line,
        template=native_extraction._ACCOUNT_BASIC_CARD_TEMPLATE,
    )

    assert [part["_account_composite_role"] for part in parts or ()] == list(role_order)
    assert [part["evidence_ids"] for part in parts or ()] == [
        ["left-token"],
        ["right-token"],
    ]


@pytest.mark.parametrize(
    "defect",
    ("missing", "duplicate_store", "partial", "shared", "overlap", "geometryless"),
)
def test_composite_header_token_ownership_defects_fail_closed(defect: str) -> None:
    line = _line(
        "开立日期到期日期",
        [100, 30, 250, 44],
        evidence="unused",
    )
    line["evidence_ids"] = ["open-token", "due-token"]
    atoms = [
        {"id": "open-token", "text": "开立日期", "bbox": [100, 30, 145, 44]},
        {"id": "due-token", "text": "到期日期", "bbox": [190, 30, 250, 44]},
    ]
    if defect == "missing":
        line["evidence_ids"] = []
    elif defect == "duplicate_store":
        atoms.append({**atoms[0]})
    elif defect == "partial":
        atoms = atoms[:1]
    elif defect == "shared":
        line["evidence_ids"] = ["open-token", "open-token"]
    elif defect == "overlap":
        atoms[1] = {**atoms[1], "bbox": [140, 30, 250, 44]}
    elif defect == "geometryless":
        atoms[1] = {"id": "due-token", "text": "到期日期"}
    context = SimpleNamespace(evidence_plane=SimpleNamespace(evidence=SimpleNamespace(text_atoms=atoms)))

    assert (
        native_extraction._account_composite_header_parts(
            context,
            line,
            template=("open_date", "due_date"),
        )
        is None
    )


def test_unequal_merged_true_due_date_never_routes_to_open_date(monkeypatch) -> None:
    header = [line for line in _loan_header(40) if line["text"] not in {"开立日期", "到期日期"}]
    composite = _line(
        "开立日期到期日期",
        [145, 40, 255, 52],
        evidence="unused",
    )
    composite["evidence_ids"] = ["open-header", "due-header"]
    values = _loan_values(55, include_open_date=False)
    due = next(line for line in values if line["text"] == "2031.05.31")
    # The true due value sits left of the former equal-width split (x=200),
    # but inside the exact wide due-date header token region.
    due["bbox"] = [180, 55, 190, 64]
    lines = [
        _line("非循环贷账户", [20, 10, 140, 22]),
        _line("账户1", [20, 25, 80, 36]),
        *header,
        composite,
        *values,
        _line("截至2022年12月24日", [160, 105, 270, 117]),
    ]
    context = _single_card_context(lines)
    context.evidence_plane = SimpleNamespace(
        evidence=SimpleNamespace(
            text_atoms=[
                {"id": "due-header", "text": "到期日期", "bbox": [180, 40, 255, 52]},
                {"id": "open-header", "text": "开立日期", "bbox": [145, 40, 175, 52]},
            ]
        )
    )
    monkeypatch.setattr(
        native_extraction,
        "_extract_table_accounts",
        lambda _context: ([], [], []),
    )

    accounts, _, _ = native_extraction._extract_accounts(context)

    assert len(accounts) == 1
    assert "open_date" not in accounts[0]
    assert accounts[0]["due_date"] == "2031-05-31"


def test_composite_atom_gap_never_becomes_a_semantic_column(monkeypatch) -> None:
    header = [line for line in _loan_header(40) if line["text"] not in {"开立日期", "到期日期"}]
    composite = _line(
        "开立日期到期日期",
        [145, 40, 255, 52],
        evidence="unused",
    )
    composite["evidence_ids"] = ["open-header", "due-header"]
    values = _loan_values(55, include_open_date=False)
    due = next(line for line in values if line["text"] == "2031.05.31")
    due["bbox"] = [176, 55, 179, 64]
    lines = [
        _line("非循环贷账户", [20, 10, 140, 22]),
        _line("账户1", [20, 25, 80, 36]),
        *header,
        composite,
        *values,
        _line("截至2022年12月24日", [160, 105, 270, 117]),
    ]
    context = _single_card_context(lines)
    context.evidence_plane = SimpleNamespace(
        evidence=SimpleNamespace(
            text_atoms=[
                {"id": "open-header", "text": "开立日期", "bbox": [145, 40, 175, 52]},
                {"id": "due-header", "text": "到期日期", "bbox": [180, 40, 255, 52]},
            ]
        )
    )
    monkeypatch.setattr(
        native_extraction,
        "_extract_table_accounts",
        lambda _context: ([], [], []),
    )

    accounts, _, _ = native_extraction._extract_accounts(context)

    assert len(accounts) == 1
    assert "open_date" not in accounts[0]
    assert "due_date" not in accounts[0]


def test_narrow_composite_header_cannot_invent_column_geometry(monkeypatch) -> None:
    header = [line for line in _card_header(40) if line["text"] not in {"开立日期", "账户授信额度"}]
    lines = [
        _line("贷记卡账户", [20, 10, 120, 22]),
        _line("账户1", [20, 25, 80, 36]),
        *header,
        _line("开立日期账户授信额度", [125, 40, 155, 52]),
        *_card_values(55),
        _line("截至2022年12月24日", [160, 105, 270, 117]),
    ]

    account, context = _extract_one(monkeypatch, lines)

    assert "credit_limit" not in account
    assert "shared_credit_limit" not in account
    assert {"credit_limit", "shared_credit_limit"} <= set(
        _issue_fields(context, "candidate_b_exact_slot_value_invalid")
    )


def test_substantive_value_and_dash_in_one_slot_are_withheld(monkeypatch) -> None:
    values = _card_values(55)
    credit = next(value for value in values if value["text"] == "50,000")
    credit["bbox"] = [232, 55, 260, 64]
    lines = [
        _line("贷记卡账户", [20, 10, 120, 22]),
        _line("账户1", [20, 25, 80, 36]),
        *_card_header(40),
        *values,
        _line("截至2022年12月24日", [160, 105, 270, 117]),
    ]

    account, context = _extract_one(monkeypatch, lines)

    assert "shared_credit_limit" not in account
    assert "shared_credit_limit" in _issue_fields(
        context,
        "candidate_b_exact_slot_value_invalid",
    )


@pytest.mark.parametrize("prefix", ("附注", "其他", "本栏说明", "提示", "贷款说明"))
def test_structural_prefix_cannot_contaminate_institution(monkeypatch, prefix: str) -> None:
    lines = [
        _line("贷记卡账户", [20, 10, 120, 22]),
        _line("账户1", [20, 25, 80, 36]),
        *_card_header(40),
        _line(prefix, [22, 53, 55, 54.5]),
        *_card_values(55),
        _line("截至2022年12月24日", [160, 105, 270, 117]),
    ]

    account, context = _extract_one(monkeypatch, lines)

    assert account["management_institution"] == "中国工商银行股份有限公司厦门市分行"
    assert "management_institution" in _issue_fields(
        context,
        "candidate_b_account_cluster_residue_unresolved",
    )


def test_shifted_next_page_values_are_reported_not_relabelled(monkeypatch) -> None:
    shifted = [
        {**line, "bbox": [value + 210 if index % 2 == 0 else value for index, value in enumerate(line["bbox"])]}
        for line in _card_values(20, page=2, source_page=2)
    ]
    lines = [
        _line("贷记卡账户", [20, 660, 120, 672]),
        _line("账户1", [20, 680, 80, 692]),
        *_card_header(700),
        *shifted,
        _line("截至2022年12月24日", [160, 80, 270, 92], page=2, source_page=2),
    ]
    context = _single_card_context(lines, transition=True)
    monkeypatch.setattr(native_extraction, "_extract_table_accounts", lambda _context: ([], [], []))

    accounts, _, _ = native_extraction._extract_accounts(context)

    assert "management_institution" not in accounts[0]
    assert "management_institution" in _issue_fields(
        context,
        "candidate_b_exact_slot_value_invalid",
    )


def test_verified_page_transition_may_join_anchor_to_next_page_header(monkeypatch) -> None:
    lines = [
        _line("贷记卡账户", [20, 660, 120, 672]),
        _line("账户1", [20, 680, 80, 692]),
        *_card_header(20, page=2, source_page=2),
        *_card_values(38, page=2, source_page=2),
        _line("截至2022年12月24日", [160, 90, 270, 102], page=2, source_page=2),
    ]
    context = _single_card_context(lines, transition=True)
    monkeypatch.setattr(native_extraction, "_extract_table_accounts", lambda _context: ([], [], []))

    accounts, _, _ = native_extraction._extract_accounts(context)

    assert accounts[0]["account_identifier"] == "B10111000H000141000000210047456001"
    assert accounts[0]["open_date"] == "2021-05-31"


def test_explicit_dashes_mark_canonical_source_absence(monkeypatch) -> None:
    values = [
        _line("--", [22, 55, 68, 64]),
        _line("--", [76, 55, 121, 64]),
        _line("--", [130, 55, 170, 64]),
        _line("50,000", [184, 55, 216, 64]),
        _line("--", [242, 55, 255, 64]),
        _line("--", [278, 55, 312, 64]),
        _line("贷记卡", [322, 55, 350, 64]),
        _line("信用/免担保", [364, 55, 408, 64]),
    ]
    lines = [
        _line("贷记卡账户", [20, 10, 120, 22]),
        _line("账户1", [20, 25, 80, 36]),
        *_card_header(40),
        *values,
        _line("截至2022年12月24日", [160, 105, 270, 117]),
    ]

    account, context = _extract_one(monkeypatch, lines)

    assert set(account["_source_absent_fields"]) >= {
        "management_institution",
        "account_identifier",
        "open_date",
        "shared_credit_limit",
        "currency",
        "account_currency",
    }
    assert not set(_issue_fields(context, "candidate_b_exact_slot_value_invalid")).intersection(
        {"management_institution", "account_identifier", "open_date", "currency"}
    )


def test_cross_column_finite_values_are_retained_only_with_residue_issues(monkeypatch) -> None:
    values = [line for line in _card_values(55) if line["text"] not in {"贷记卡", "信用/免担保"}]
    values.append(_line("贷记卡 信用/免担保", [315, 55, 410, 64]))
    lines = [
        _line("贷记卡账户", [20, 10, 120, 22]),
        _line("账户1", [20, 25, 80, 36]),
        *_card_header(40),
        *values,
        _line("截至2022年12月24日", [160, 105, 270, 117]),
    ]

    account, context = _extract_one(monkeypatch, lines)

    assert account["business_type"] == "贷记卡"
    assert account["guarantee_type"] == "信用/免担保"
    assert set(_issue_fields(context, "candidate_b_account_cluster_residue_unresolved")) == {
        "business_type",
        "guarantee_type",
    }


def test_geometry_header_must_follow_owning_anchor(monkeypatch) -> None:
    lines = [
        _line("贷记卡账户", [20, 10, 120, 22]),
        _line("账户2", [20, 100, 80, 112]),
        *_card_header(40),
        *_card_values(55),
        _line("截至2022年12月24日", [160, 155, 270, 167]),
    ]

    account, context = _extract_one(monkeypatch, lines)

    assert "management_institution" not in account
    assert "account_identifier" not in account
    assert set(_issue_fields(context, "candidate_b_exact_slot_value_invalid")) >= {
        "management_institution",
        "account_identifier",
    }


def test_geometry_values_must_be_physically_below_header(monkeypatch) -> None:
    lines = [
        _line("贷记卡账户", [20, 10, 120, 22]),
        _line("账户1", [20, 25, 80, 36]),
        *_card_header(100),
        *_card_values(55),
        _line("截至2022年12月24日", [160, 155, 270, 167]),
    ]

    account, context = _extract_one(monkeypatch, lines)

    assert "open_date" not in account
    assert "open_date" in _issue_fields(
        context,
        "candidate_b_exact_slot_value_invalid",
    )


def test_clean_loan_template_keeps_open_and_due_date_in_distinct_slots(monkeypatch) -> None:
    lines = [
        _line("非循环贷账户", [20, 10, 140, 22]),
        _line("账户1", [20, 25, 80, 36]),
        *_loan_header(40),
        *_loan_values(55),
        _line("截至2022年12月24日", [160, 105, 270, 117]),
    ]

    account, _context = _extract_one(monkeypatch, lines)

    assert account["open_date"] == "2021-01-12"
    assert account["due_date"] == "2031-05-31"
    assert account["loan_amount"] == 140_000


def test_missing_loan_open_date_never_promotes_visible_due_date(monkeypatch) -> None:
    lines = [
        _line("非循环贷账户", [20, 10, 140, 22]),
        _line("账户1", [20, 25, 80, 36]),
        *_loan_header(40),
        *_loan_values(55, include_open_date=False),
        _line("截至2022年12月24日", [160, 105, 270, 117]),
    ]

    account, context = _extract_one(monkeypatch, lines)

    assert "open_date" not in account
    assert account["due_date"] == "2031-05-31"
    assert "open_date" in _issue_fields(
        context,
        "candidate_b_exact_slot_value_invalid",
    )


def test_r2_long_term_due_slot_preserves_perpetual_validity(monkeypatch) -> None:
    values = [
        _line("五矿国际信托有限公司", [22, 55, 76, 64]),
        _line("RL20140821000046", [84, 55, 141, 64]),
        _line("2021.01.12", [150, 55, 195, 64]),
        _line("长期", [205, 55, 250, 64]),
        _line("140,000", [265, 55, 320, 64]),
        _line("人民币元", [340, 55, 390, 64]),
    ]
    lines = [
        _line("循环贷账户二", [20, 10, 140, 22]),
        _line("账户1", [20, 25, 80, 36]),
        *_r2_header(40),
        *values,
        _line("截至2022年12月24日", [160, 105, 270, 117]),
    ]

    account, context = _extract_one(monkeypatch, lines)

    assert "due_date" not in account
    assert account["validity_type"] == "perpetual"
    assert account["canonical_raw"]["due_date"] == "长期"
    assert "due_date" not in _issue_fields(context, "candidate_b_exact_slot_value_invalid")


def test_loan_second_row_geometry_recovers_typed_fields_without_native_table(
    monkeypatch,
) -> None:
    lines = [
        _line("非循环贷账户", [20, 10, 140, 22]),
        _line("账户1", [20, 25, 80, 36]),
        *_loan_header(40),
        *_loan_values(55),
        *_loan_second_header(90),
        *_loan_second_values(105),
        _line("截至2022年12月24日", [160, 140, 270, 152]),
    ]

    account, context = _extract_one(monkeypatch, lines)

    assert account["business_type"] == "个人消费贷款"
    assert account["guarantee_type"] == "抵押"
    assert account["repayment_periods"] == 36
    assert account["repayment_frequency"] == "月"
    assert account["repayment_method"] == "等额本息"
    assert account["co_borrower_flag"] == "否"
    assert not set(_ACCOUNT_SECOND_ROW_FIELDS).intersection(
        _issue_fields(context, "candidate_b_exact_slot_value_invalid")
    )


_ACCOUNT_SECOND_ROW_FIELDS = {
    "business_type",
    "guarantee_type",
    "repayment_periods",
    "repayment_frequency",
    "repayment_method",
    "co_borrower_flag",
}


def test_loan_second_row_dash_and_residue_are_field_local(monkeypatch) -> None:
    values = _loan_second_values(105)
    next(value for value in values if value["text"] == "否")["text"] = "--"
    next(value for value in values if value["text"] == "月")["text"] = "月提示"
    lines = [
        _line("非循环贷账户", [20, 10, 140, 22]),
        _line("账户1", [20, 25, 80, 36]),
        *_loan_header(40),
        *_loan_values(55),
        *_loan_second_header(90),
        *values,
        _line("截至2022年12月24日", [160, 140, 270, 152]),
    ]

    account, context = _extract_one(monkeypatch, lines)

    assert "co_borrower_flag" not in account
    assert "co_borrower_flag" in account["_source_absent_fields"]
    assert "repayment_frequency" not in account
    assert "repayment_frequency" in _issue_fields(
        context,
        "candidate_b_exact_slot_value_invalid",
    )
    assert account["business_type"] == "个人消费贷款"
    assert account["guarantee_type"] == "抵押"


def test_loan_second_row_observations_replay_into_matched_table(monkeypatch) -> None:
    table = {
        **_table_observation(),
        "account_type": "non_revolving_loan",
        "management_institution": "五矿国际信托有限公司",
        "account_identifier": "J10158510H000110000000640557",
        "open_date": "2021-01-12",
        "due_date": "2031-05-31",
        "loan_amount": 140_000,
        "currency": "CNY",
        "account_currency": "CNY",
    }
    lines = [
        _line("非循环贷账户", [20, 10, 140, 22]),
        _line("账户1", [20, 25, 80, 36]),
        *_loan_header(40),
        *_loan_values(55),
        *_loan_second_header(90),
        *_loan_second_values(105),
        _line("截至2022年12月24日", [160, 140, 270, 152]),
    ]

    account, _context = _extract_one(monkeypatch, lines, table=table)

    assert account["business_type"] == "个人消费贷款"
    assert account["guarantee_type"] == "抵押"
    assert account["repayment_periods"] == 36
    assert account["repayment_frequency"] == "月"
    assert account["repayment_method"] == "等额本息"
    assert account["co_borrower_flag"] == "否"


def test_invalid_geometry_does_not_erase_independently_valid_table_values(
    monkeypatch,
) -> None:
    table = {
        **_table_observation(),
        "management_institution": "中国工商银行股份有限公司厦门市分行",
        "account_identifier": "B10111000H000141000000210047456001",
        "open_date": "2021-05-31",
        "credit_limit": 50_000,
        "currency": "CNY",
        "account_currency": "CNY",
        "business_type": "贷记卡",
        "guarantee_type": "信用/免担保",
    }
    header = [line for line in _card_header(40) if line["text"] not in {"开立日期", "账户授信额度"}]
    lines = [
        _line("贷记卡账户", [20, 10, 120, 22]),
        _line("账户1", [20, 25, 80, 36]),
        *header,
        _line("开立日期账户授信额度X", [125, 40, 225, 52]),
        *_card_values(55),
        _line("截至2022年12月24日", [160, 105, 270, 117]),
    ]

    account, context = _extract_one(monkeypatch, lines, table=table)

    for field_name, expected in {
        "management_institution": "中国工商银行股份有限公司厦门市分行",
        "account_identifier": "B10111000H000141000000210047456001",
        "open_date": "2021-05-31",
        "credit_limit": 50_000,
        "currency": "CNY",
        "business_type": "贷记卡",
        "guarantee_type": "信用/免担保",
    }.items():
        assert account[field_name] == expected
        assert field_name not in _issue_fields(
            context,
            "candidate_b_exact_slot_value_invalid",
        )


def test_table_matching_requires_account_family_compatibility() -> None:
    skeleton = {
        "account_type": "credit_card",
        "page": 1,
        "bbox": [20, 20, 100, 30],
        "_canonical_segment": {
            "pages": [{"logical_page": 1, "min_y": 20, "max_y": 200}],
        },
    }
    table = {
        "account_type": "non_revolving_loan",
        "page": 1,
        "bbox": [20, 40, 400, 180],
        "source_refs": [{"logical_page": 1, "bbox": [20, 40, 400, 180]}],
    }

    assert native_extraction._match_account_table_observations([skeleton], [table]) == {}
