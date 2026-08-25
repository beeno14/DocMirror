from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from docmirror.plugins.credit_report.personal_detail_scanned import native_extraction

_IDENTIFIER_BASE = "D10000000H00012024010101021012000000000001"


def _identifier(index: int) -> str:
    return f"{_IDENTIFIER_BASE[:-2]}{index:02d}"


def _physical_account_fixture(
    index: int,
    *,
    scale: float = 1.0,
    width: float = 420.0,
    template_id: str = "",
) -> tuple[SimpleNamespace, dict[str, object]]:
    labels = ["管理机构", "账户标识", "开立日期", "账户授信额度", "币种", "业务种类"]
    values = [
        f"机构{index}股份有限公司",
        _identifier(index),
        "2024.01.02",
        str(1000 + index),
        "人民币元",
        "个人消费",
    ]
    table_id = f"physical-account-{index}"
    left = 11.0 * scale
    top = (20.0 + index * 80.0) * scale
    row_height = 20.0 * scale
    weights = (0.9, 1.35, 0.8, 1.15, 0.75, 1.25)
    unit = width * scale / sum(weights)
    boundaries = [left]
    for weight in weights:
        boundaries.append(boundaries[-1] + weight * unit)
    cell_bboxes = [
        [
            [boundaries[column], top + row * row_height, boundaries[column + 1], top + (row + 1) * row_height]
            for column in range(len(labels))
        ]
        for row in range(2)
    ]
    cell_evidence_ids = [
        [[f"physical:{index}:r{row}:c{column}"] for column in range(len(labels))]
        for row in range(2)
    ]
    table_bbox = [left, top, boundaries[-1], top + 2 * row_height]
    metadata: dict[str, object] = {
        "raw_rows": [labels, values],
        "cell_bboxes": cell_bboxes,
        "cell_evidence_ids": cell_evidence_ids,
        "cell_geometry_status": [["exact"] * len(labels) for _row in range(2)],
    }
    if template_id:
        metadata["canonical_template_id"] = template_id
    table = SimpleNamespace(
        table_id=table_id,
        bbox=table_bbox,
        metadata=metadata,
        headers=[],
        rows=[],
    )

    def field_ref(field_name: str, column: int) -> dict[str, object]:
        return {
            "source": "native_detail_table_cell",
            "geometry_scope": "cell",
            "logical_page": 7,
            "source_page": 4,
            "table_id": table_id,
            "row": 1,
            "column": column,
            "canonical_row": 1,
            "canonical_column": column,
            "bbox": list(cell_bboxes[1][column]),
            "evidence_ids": list(cell_evidence_ids[1][column]),
            "binding": "canonical_field_slot",
            "binding_quality": "canonical_header_column",
            "field_name": field_name,
        }

    observation_id = f"table-observation-{index}"
    observation: dict[str, object] = {
        "account_id": observation_id,
        "_table_observation_id": observation_id,
        "_table_observation_instance_id": f"table-instance-{index}",
        "source": "native_detail_account_table",
        "account_type": "non_revolving_loan",
        "sequence": index,
        "category_sequence": index,
        "management_institution": values[0],
        "account_identifier": values[1],
        "open_date": "2024-01-02",
        "credit_limit": 1000 + index,
        "currency": "CNY",
        "account_currency": "CNY",
        # A normalized value without a field-local exact ref must never leak.
        "loan_amount": 900000 + index,
        "confidence": 0.99,
        "canonical_raw": {
            "management_institution": values[0],
            "account_identifier": values[1],
            "open_date": values[2],
            "credit_limit": values[3],
            "currency": values[4],
            "account_currency": values[4],
        },
        "source_refs": [
            {
                "source": "native_detail_table",
                "geometry_scope": "table",
                "logical_page": 7,
                "source_page": 4,
                "table_id": table_id,
                "bbox": list(table_bbox),
            }
        ],
        "source_refs_by_field": {
            "management_institution": [field_ref("management_institution", 0)],
            "account_identifier": [field_ref("account_identifier", 1)],
            "open_date": [field_ref("open_date", 2)],
            "credit_limit": [field_ref("credit_limit", 3)],
            "currency": [field_ref("currency", 4)],
            "account_currency": [field_ref("account_currency", 4)],
        },
    }
    return table, observation


def _context(tables: list[SimpleNamespace]) -> SimpleNamespace:
    return SimpleNamespace(
        pages=[
            SimpleNamespace(
                page_number=7,
                source_page_number=4,
                tables=tables,
            )
        ],
        corrected_evidence_pages=lambda: [],
    )


@pytest.mark.parametrize(
    "raw",
    (
        "导中国工商银行股份有限公司",
        "S 中国工商银行股份有限公司",
        "中国工商银行股份有限公司 Ss",
    ),
)
def test_account_institution_publication_never_deletes_source_glyphs(raw: str) -> None:
    # The legacy normalizer can recognize the likely name by deleting debris,
    # but an exact account field cell owns the complete printed value.
    assert native_extraction._account_institution(raw) == "中国工商银行股份有限公司"
    assert native_extraction._exact_source_account_institution(raw) is None

    table, _observation = _physical_account_fixture(1)
    table.metadata["raw_rows"][1][0] = raw

    assert native_extraction._raw_physical_account_table_observations(
        _context([table])
    ) == []


def test_account_institution_publication_never_substitutes_source_glyphs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from docmirror.plugins.credit_report.personal_detail_scanned import ocr_correction

    raw = "中国建没银行股份有限公司"
    corrected = "中国建设银行股份有限公司"
    monkeypatch.setattr(
        ocr_correction,
        "normalize_institution_name",
        lambda value: corrected if value == raw else value,
    )

    assert native_extraction._account_institution(raw) == corrected
    assert native_extraction._exact_source_account_institution(raw) is None
    assert (
        native_extraction._exact_source_account_institution(corrected)
        == corrected
    )


@pytest.mark.parametrize(
    ("population", "scale", "width", "reverse"),
    ((1, 0.55, 275.0, False), (3, 1.0, 430.0, True), (5, 1.8, 610.0, False)),
)
def test_physical_account_population_is_variable_and_order_independent(
    monkeypatch: pytest.MonkeyPatch,
    population: int,
    scale: float,
    width: float,
    reverse: bool,
) -> None:
    fixtures = [
        _physical_account_fixture(index, scale=scale, width=width)
        for index in range(1, population + 1)
    ]
    tables = [item[0] for item in fixtures]
    observations = [item[1] for item in fixtures]
    context = _context(list(reversed(tables)) if reverse else tables)
    ordered = list(reversed(observations)) if reverse else observations
    monkeypatch.setattr(
        native_extraction,
        "_extract_table_accounts",
        lambda _context: (deepcopy(ordered), [], []),
    )

    accounts, repayments, events = native_extraction._extract_accounts(context)

    assert repayments == []
    assert events == []
    assert len(accounts) == population
    assert len({account["account_id"] for account in accounts}) == population
    assert all(account["_physical_account_identity_provisional"] for account in accounts)
    assert all(
        not {"account_type", "credit_card_type", "category_sequence", "sequence", "loan_amount"}
        .intersection(account)
        for account in accounts
    )
    assert all(
        set(account["source_refs_by_field"]) >= {
            "management_institution",
            "account_identifier",
            "open_date",
            "currency",
        }
        for account in accounts
    )
    issue_rows = [
        issue
        for issue in context._personal_detail_extraction_issues
        if issue["issue_code"] == "candidate_b_account_physical_identity_provisional"
    ]
    assert len(issue_rows) == population
    assert all(
        "typed_family_ordinal_completeness_not_satisfied" in issue["reason_codes"]
        for issue in issue_rows
    )

    forward = native_extraction._provisional_physical_account_records(
        context,
        deepcopy(observations),
    )
    backward = native_extraction._provisional_physical_account_records(
        context,
        list(reversed(deepcopy(observations))),
    )
    assert {row["account_id"] for row in forward.values()} == {
        row["account_id"] for row in backward.values()
    }


def test_physical_account_exact_owner_dedupes_replays_without_value_or_order_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table, observation = _physical_account_fixture(1)
    replay = deepcopy(observation)
    replay["account_id"] = "observation-replay"
    replay["_table_observation_id"] = "observation-replay"
    replay["_table_observation_instance_id"] = "instance-replay"
    monkeypatch.setattr(
        native_extraction,
        "_extract_table_accounts",
        lambda _context: ([replay, deepcopy(observation)], [], []),
    )
    context = _context([table])

    accounts, _repayments, _events = native_extraction._extract_accounts(context)

    assert len(accounts) == 1
    assert accounts[0]["_physical_account_identity_provisional"] is True
    assert any(
        issue["issue_code"] == "candidate_b_duplicate_account_table_source_suppressed"
        for issue in context._personal_detail_extraction_issues
    )


@pytest.mark.parametrize(
    "defect",
    (
        "duplicate_identifier",
        "replayed_evidence",
        "blank_evidence_member",
        "non_string_evidence_member",
        "inexact_geometry",
        "wrong_field_owner",
        "foreign_section",
    ),
)
def test_physical_account_population_fails_closed_on_ambiguous_or_foreign_owners(
    defect: str,
) -> None:
    table_one, observation_one = _physical_account_fixture(1)
    table_two, observation_two = _physical_account_fixture(
        2,
        template_id="credit_agreement" if defect == "foreign_section" else "",
    )
    tables = [table_one, table_two]
    observations = [observation_one, observation_two]
    if defect == "duplicate_identifier":
        observation_two["account_identifier"] = observation_one["account_identifier"]
        observation_two["canonical_raw"]["account_identifier"] = observation_one[
            "account_identifier"
        ]
    elif defect == "replayed_evidence":
        table_two.metadata["cell_evidence_ids"][0][0] = list(
            table_one.metadata["cell_evidence_ids"][0][0]
        )
    elif defect == "blank_evidence_member":
        table_two.metadata["cell_evidence_ids"][1][1].append("   ")
        observation_two["source_refs_by_field"]["account_identifier"][0][
            "evidence_ids"
        ].append("   ")
    elif defect == "non_string_evidence_member":
        table_two.metadata["cell_evidence_ids"][1][1].append(7)
        observation_two["source_refs_by_field"]["account_identifier"][0][
            "evidence_ids"
        ].append(7)
    elif defect == "inexact_geometry":
        table_two.metadata["cell_geometry_status"][1][1] = "derived"
    elif defect == "wrong_field_owner":
        ref = observation_two["source_refs_by_field"]["account_identifier"][0]
        ref["column"] = 2
        ref["canonical_column"] = 2
        ref["bbox"] = list(table_two.metadata["cell_bboxes"][1][2])
        ref["evidence_ids"] = list(table_two.metadata["cell_evidence_ids"][1][2])

    records = native_extraction._provisional_physical_account_records(
        _context(tables),
        deepcopy(observations),
    )

    if defect in {"duplicate_identifier", "replayed_evidence"}:
        assert records == {}
    else:
        assert len(records) == 1
        assert next(iter(records.values()))["account_identifier"] == _identifier(1)


def test_physical_account_never_satisfies_a_missing_typed_family_ordinal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixtures = [_physical_account_fixture(index) for index in (1, 2, 3)]
    tables = [item[0] for item in fixtures]
    observations = [item[1] for item in fixtures]
    anchors = [
        {
            "account_id": f"credit_account:non_revolving_loan:{ordinal}",
            "account_type": "non_revolving_loan",
            "account_family_quality": "exact",
            "_printed_ordinal_status": "printed_unique",
            "category_sequence": ordinal,
            "sequence": ordinal,
            "source_refs": [],
        }
        for ordinal in (1, 3)
    ]
    context = _context(tables)
    monkeypatch.setattr(
        native_extraction,
        "_extract_table_accounts",
        lambda _context: (deepcopy(observations), [], []),
    )
    monkeypatch.setattr(
        native_extraction,
        "_account_anchor_skeletons",
        lambda _context: deepcopy(anchors),
    )
    monkeypatch.setattr(
        native_extraction,
        "_repair_complete_account_anchor_skeletons",
        lambda _context, skeletons: skeletons,
    )
    monkeypatch.setattr(
        native_extraction,
        "_match_account_table_observations",
        lambda _skeletons, _tables, *, parse_result=None: {0: 0, 1: 2},
    )

    accounts, _repayments, _events = native_extraction._extract_accounts(context)

    typed = [account for account in accounts if account.get("account_type")]
    physical = [
        account
        for account in accounts
        if account.get("_physical_account_identity_provisional")
    ]
    assert {account["category_sequence"] for account in typed} == {1, 3}
    assert len(physical) == 1
    assert "account_type" not in physical[0]
    assert "category_sequence" not in physical[0]
    gap = next(
        issue
        for issue in context._personal_detail_extraction_issues
        if issue["issue_code"] == "candidate_b_account_sequence_gap"
    )
    assert gap["candidate_value"]["missing_category_sequences"] == [2]


def _typed_anchor(ordinal: int) -> dict[str, object]:
    return {
        "account_id": f"credit_account:non_revolving_loan:{ordinal}",
        "account_type": "non_revolving_loan",
        "account_family_quality": "exact",
        "_printed_ordinal_status": "printed_unique",
        "category_sequence": ordinal,
        "sequence": ordinal,
        "source_refs": [],
    }


def test_typed_account_gap_contract_accepts_a_dense_family_longer_than_any_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context([])
    monkeypatch.setattr(
        native_extraction,
        "_extract_table_accounts",
        lambda _context: ([], [], []),
    )
    monkeypatch.setattr(
        native_extraction,
        "_account_anchor_skeletons",
        lambda _context: [_typed_anchor(ordinal) for ordinal in range(1, 22)],
    )
    monkeypatch.setattr(
        native_extraction,
        "_repair_complete_account_anchor_skeletons",
        lambda _context, skeletons: skeletons,
    )

    accounts, _repayments, _events = native_extraction._extract_accounts(context)

    assert len(accounts) == 21
    assert not any(
        issue["issue_code"] == "candidate_b_account_sequence_gap"
        for issue in context._personal_detail_extraction_issues
    )


def test_typed_account_gap_contract_isolates_a_sparse_high_ordinal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context([])
    monkeypatch.setattr(
        native_extraction,
        "_extract_table_accounts",
        lambda _context: ([], [], []),
    )
    monkeypatch.setattr(
        native_extraction,
        "_account_anchor_skeletons",
        lambda _context: [_typed_anchor(ordinal) for ordinal in (1, 3, 21)],
    )
    monkeypatch.setattr(
        native_extraction,
        "_repair_complete_account_anchor_skeletons",
        lambda _context, skeletons: skeletons,
    )

    native_extraction._extract_accounts(context)

    gap = next(
        issue
        for issue in context._personal_detail_extraction_issues
        if issue["issue_code"] == "candidate_b_account_sequence_gap"
    )
    assert gap["candidate_value"]["missing_category_sequences"] == [2]
    assert gap["candidate_value"]["outlier_category_sequences"] == [21]


def test_source_ledger_keeps_raw_physical_accounts_outside_typed_completeness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixtures = [_physical_account_fixture(index) for index in (1, 2)]
    context = _context([item[0] for item in fixtures])
    monkeypatch.setattr(
        native_extraction,
        "_account_anchor_skeletons",
        lambda _context: [],
    )
    monkeypatch.setattr(
        native_extraction,
        "_repair_complete_account_anchor_skeletons",
        lambda _context, skeletons: skeletons,
    )
    monkeypatch.setattr(
        native_extraction,
        "_registered_account_section_plane",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        native_extraction,
        "_exact_account_table_cell_anchors",
        lambda _context: {},
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

    assert "credit_accounts" not in ledger
    assert "account_family_endpoints" not in ledger
    assert ledger["account_raw_physical_count"] == 2
    assert len(ledger["account_raw_physical_observations"]) == 2
    assert all(
        observation["account_type"] is None
        and observation["category_sequence"] is None
        for observation in ledger["account_raw_physical_observations"]
    )


def _anchor_owned_value_skeleton(*, exact_field_refs: bool) -> dict[str, object]:
    identifier = _identifier(1)
    anchor_ref = {
        "source": "candidate_b_account_anchor",
        "logical_page": 7,
        "source_page": 4,
        "bbox": [10.0, 10.0, 120.0, 30.0],
        "evidence_ids": ["account-anchor:1"],
    }
    skeleton = {
        **_typed_anchor(1),
        "account_identifier": identifier,
        "account_identifier_source": "canonical_anchor_table_row",
        "validity_type": "perpetual",
        "canonical_raw": {
            "account_identifier": identifier,
            "due_date": "长期",
        },
        "source_refs": [anchor_ref],
    }
    if exact_field_refs:
        skeleton["source_refs_by_field"] = {
            field_name: [
                {
                    "source": "candidate_b_account_anchor_interval",
                    "logical_page": 7,
                    "source_page": 4,
                    "bbox": [20.0, top, 260.0, top + 18.0],
                    "evidence_ids": [evidence_id],
                    "field_name": field_name,
                    "binding": "canonical_account_header_geometry",
                    "binding_quality": "canonical_account_header_geometry",
                }
            ]
            for field_name, top, evidence_id in (
                ("account_identifier", 40.0, "identifier:value"),
                ("due_date", 70.0, "due:value"),
            )
        }
    return skeleton


def _run_anchor_owned_value_case(
    monkeypatch: pytest.MonkeyPatch,
    *,
    exact_field_refs: bool,
) -> tuple[SimpleNamespace, dict[str, object]]:
    table, observation = _physical_account_fixture(1)
    observation.pop("account_identifier", None)
    observation["canonical_raw"].pop("account_identifier", None)
    observation["source_refs_by_field"].pop("account_identifier", None)
    context = _context([table])
    skeleton = _anchor_owned_value_skeleton(exact_field_refs=exact_field_refs)
    monkeypatch.setattr(
        native_extraction,
        "_extract_table_accounts",
        lambda _context: ([deepcopy(observation)], [], []),
    )
    monkeypatch.setattr(
        native_extraction,
        "_account_anchor_skeletons",
        lambda _context: [deepcopy(skeleton)],
    )
    monkeypatch.setattr(
        native_extraction,
        "_repair_complete_account_anchor_skeletons",
        lambda _context, skeletons: skeletons,
    )
    monkeypatch.setattr(
        native_extraction,
        "_match_account_table_observations",
        lambda _skeletons, _tables, *, parse_result=None: {0: 0},
    )

    accounts, _repayments, _events = native_extraction._extract_accounts(context)

    return context, accounts[0]


def test_anchor_only_identifier_and_perpetual_token_are_withheld(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, account = _run_anchor_owned_value_case(
        monkeypatch,
        exact_field_refs=False,
    )

    assert "account_identifier" not in account
    assert "validity_type" not in account
    assert set(account["_unresolved_fields"]) >= {"account_identifier", "due_date"}
    unresolved = [
        issue
        for issue in context._personal_detail_extraction_issues
        if issue["issue_code"] == "candidate_b_account_field_owner_unresolved"
    ]
    assert {issue["field_name"] for issue in unresolved} == {
        "account_identifier",
        "due_date",
    }
    assert all(
        "account_anchor_is_population_only" in issue["reason_codes"]
        for issue in unresolved
    )


def test_exact_field_local_identifier_and_perpetual_token_remain_publishable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _context_value, account = _run_anchor_owned_value_case(
        monkeypatch,
        exact_field_refs=True,
    )

    assert account["account_identifier"] == _identifier(1)
    assert account["validity_type"] == "perpetual"
    assert account["canonical_raw"]["due_date"] == "长期"
    assert set(account["source_refs_by_field"]) >= {
        "account_identifier",
        "due_date",
    }
