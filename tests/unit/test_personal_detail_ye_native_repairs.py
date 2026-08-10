from __future__ import annotations

from types import SimpleNamespace

import pytest

from docmirror.plugins.credit_report.personal_detail_scanned import native_extraction


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


def test_document_local_inquiry_repair_withholds_two_missing_ordinals() -> None:
    assert native_extraction._document_local_inquiry_ordinals(
        [1, None, 3, 4, None, 6]
    ) == [
        (1, None),
        (None, "multiple_missing"),
        (3, None),
        (4, None),
        (None, "multiple_missing"),
        (6, None),
    ]


def test_exact_inquiry_table_localizes_two_missing_ordinals_without_emission() -> None:
    rows = [["\u7f16\u53f7", "\u67e5\u8be2\u65e5\u671f", "\u67e5\u8be2\u673a\u6784", "\u67e5\u8be2\u539f\u56e0"]]
    rows.extend(
        [
            "\u574f" if sequence in {2, 5} else str(sequence),
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

    assert [record["sequence"] for record in records] == [1, 3, 4, 6]
    assert coverage["observed_sequences"] == {"personal": [1, 3, 4, 6]}
    issues = [
        issue
        for issue in context._personal_detail_extraction_issues
        if issue.get("issue_code")
        == "candidate_b_inquiry_multiple_missing_sequences_unresolved"
    ]
    assert len(issues) == 2
    assert {issue["source_refs"][0]["row"] for issue in issues} == {2, 5}
    assert all(issue.get("field_name") == "sequence" for issue in issues)


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
