from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from docmirror.plugins.credit_report.personal_detail_scanned import native_extraction
from docmirror.plugins.credit_report.personal_detail_scanned.native_parser import (
    PBOCPersonalDetailNativeParser,
)


def _registered_owner(
    family: str,
    ordinal: int,
    *,
    page: int,
    source_page: int,
    top: float,
    evidence_id: str | None = None,
) -> dict:
    return {
        "account_id": f"credit_account:{family}:{ordinal}",
        "account_type": family,
        "category_sequence": ordinal,
        "source_refs": [
            {
                "source": "candidate_b_account_anchor",
                "logical_page": page,
                "source_page": source_page,
                "geometry_scope": "line",
                "binding": "printed_account_ordinal",
                "binding_quality": "printed_account_ordinal",
                "account_type": family,
                "category_sequence": ordinal,
                "bbox": [20.0, top, 180.0, top + 12.0],
                "evidence_ids": [evidence_id or f"account:{family}:{ordinal}"],
            }
        ],
    }


def _existing_skeleton(owner: dict, **updates: object) -> dict:
    family = str(owner["account_type"])
    ordinal = int(owner["category_sequence"])
    row = {
        "account_id": f"credit_account:{family}:{ordinal}",
        "account_type": family,
        "category_sequence": ordinal,
        "account_family_quality": "exact",
        "_printed_ordinal_status": "printed_unique",
        "_canonical_segment": {
            "ownership_basis": "printed_anchor_to_next_anchor",
            "pages": [],
        },
        "source_refs": [
            {
                key: deepcopy(value)
                for key, value in owner["source_refs"][0].items()
                if key
                not in {
                    "geometry_scope",
                    "binding",
                    "binding_quality",
                    "account_type",
                    "category_sequence",
                }
            }
        ],
        "confidence": 0.9,
    }
    row.update(updates)
    return row


def _install_registered_population(
    monkeypatch: pytest.MonkeyPatch,
    observations: dict | None,
) -> None:
    monkeypatch.setattr(
        native_extraction,
        "_registered_account_section_plane",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        native_extraction,
        "_exact_account_table_cell_anchors",
        lambda *_args: {},
    )
    monkeypatch.setattr(
        native_extraction,
        "_registered_account_population_lifecycle_observations",
        lambda *_args: deepcopy(observations),
    )


def test_registered_materialization_refreshes_cache_with_zero_additions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_registered_population(monkeypatch, None)
    repaired = {
        "account_id": "credit_account:credit_card:1",
        "account_type": "credit_card",
        "category_sequence": 1,
        "repair_marker": "post-repair",
    }
    context = SimpleNamespace(
        corrected_evidence_pages=lambda: [],
        _candidate_b_account_anchor_skeleton_cache=[{"repair_marker": "stale"}],
    )

    rows = native_extraction._materialize_registered_account_population_skeletons(
        context,
        [repaired],
    )

    assert rows == [repaired]
    assert context._candidate_b_account_anchor_skeleton_cache == [repaired]
    assert context._candidate_b_account_anchor_skeleton_cache is not rows


def test_registered_materialization_uses_global_same_page_and_section_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observations = {
        "credit_card": {
            1: _registered_owner("credit_card", 1, page=8, source_page=4, top=100.0),
            2: _registered_owner("credit_card", 2, page=8, source_page=4, top=200.0),
        },
        "quasi_credit_card": {
            1: _registered_owner(
                "quasi_credit_card",
                1,
                page=8,
                source_page=4,
                top=300.0,
            )
        },
    }
    _install_registered_population(monkeypatch, observations)
    boundary = {
        "text": "授信协议信息",
        "bbox": [20.0, 400.0, 180.0, 412.0],
        "evidence_ids": ["account-section:boundary"],
    }
    context = SimpleNamespace(
        corrected_evidence_pages=lambda: [
            {"page": 8, "source_page": 4, "lines": [boundary]}
        ]
    )

    rows = native_extraction._materialize_registered_account_population_skeletons(
        context,
        [],
    )

    assert [row["account_id"] for row in rows] == [
        "credit_account:credit_card:1",
        "credit_account:credit_card:2",
        "credit_account:quasi_credit_card:1",
    ]
    assert [row["_canonical_segment"]["pages"][0]["max_y"] for row in rows] == [
        200.0,
        300.0,
        400.0,
    ]
    assert all(
        row["_canonical_segment"]["pages"]
        == [
            {
                "logical_page": 8,
                "min_y": row["bbox"][1],
                "max_y": row["_canonical_segment"]["pages"][0]["max_y"],
                "continuation_verified": False,
            }
        ]
        for row in rows
    )
    assert all(
        row["_canonical_segment"]["cross_page_continuation_verified"] is False
        for row in rows
    )


def test_registered_materialization_never_invents_cross_page_continuation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observations = {
        "credit_card": {
            1: _registered_owner("credit_card", 1, page=8, source_page=4, top=100.0),
            2: _registered_owner("credit_card", 2, page=10, source_page=5, top=100.0),
        }
    }
    _install_registered_population(monkeypatch, observations)
    context = SimpleNamespace(
        corrected_evidence_pages=lambda: [],
        allows_scanned_line_transition=lambda *_args: False,
    )

    rows = native_extraction._materialize_registered_account_population_skeletons(
        context,
        [],
    )

    assert [[page["logical_page"] for page in row["_canonical_segment"]["pages"]] for row in rows] == [
        [8],
        [10],
    ]
    assert all(
        row["_canonical_segment"]["pages"][0]["continuation_verified"] is False
        for row in rows
    )


def _account_table_observation(
    family: str,
    *,
    page: int,
    source_page: int,
    top: float,
) -> dict:
    return {
        "account_type": family,
        "page": page,
        "source_page": source_page,
        "bbox": [20.0, top, 520.0, top + 40.0],
        "source": "native_detail_account_table",
        "_table_observation_id": f"table:{family}:{page}:{top}",
        "_table_account_family_basis": "shared_card_table_signature",
        "source_refs": [],
    }


def test_retained_segment_clamps_to_added_same_family_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner1 = _registered_owner(
        "credit_card",
        1,
        page=8,
        source_page=4,
        top=100.0,
    )
    owner2 = _registered_owner(
        "credit_card",
        2,
        page=8,
        source_page=4,
        top=200.0,
    )
    _install_registered_population(
        monkeypatch,
        {"credit_card": {1: owner1, 2: owner2}},
    )
    retained = _existing_skeleton(owner1)
    retained["_canonical_segment"]["pages"] = [
        {
            "logical_page": 8,
            "min_y": 100.0,
            "max_y": 500.0,
            "continuation_verified": False,
        }
    ]
    context = SimpleNamespace(corrected_evidence_pages=lambda: [])

    rows = native_extraction._materialize_registered_account_population_skeletons(
        context,
        [retained],
    )

    assert [row["account_id"] for row in rows] == [
        "credit_account:credit_card:1",
        "credit_account:credit_card:2",
    ]
    assert rows[0]["_canonical_segment"]["pages"][0]["max_y"] == 200.0
    matches = native_extraction._match_account_table_observations(
        rows,
        [
            _account_table_observation(
                "credit_card",
                page=8,
                source_page=4,
                top=240.0,
            )
        ],
    )
    assert matches == {1: 0}


def test_retained_verified_continuation_clamps_to_added_cross_family_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retained_owner = _registered_owner(
        "credit_card",
        1,
        page=8,
        source_page=4,
        top=100.0,
    )
    added_owner = _registered_owner(
        "quasi_credit_card",
        1,
        page=9,
        source_page=5,
        top=200.0,
    )
    _install_registered_population(
        monkeypatch,
        {
            "credit_card": {1: retained_owner},
            "quasi_credit_card": {1: added_owner},
        },
    )
    retained = _existing_skeleton(retained_owner)
    retained["_canonical_segment"]["pages"] = [
        {
            "logical_page": 8,
            "min_y": 100.0,
            "max_y": None,
            "continuation_verified": False,
        },
        {
            "logical_page": 9,
            "min_y": 0.0,
            "max_y": 500.0,
            "continuation_verified": True,
        },
    ]
    retained["_canonical_segment"]["cross_page_continuation_verified"] = True
    context = SimpleNamespace(corrected_evidence_pages=lambda: [])

    rows = native_extraction._materialize_registered_account_population_skeletons(
        context,
        [retained],
    )

    retained_row = next(
        row
        for row in rows
        if row["account_id"] == "credit_account:credit_card:1"
    )
    continuation = retained_row["_canonical_segment"]["pages"][1]
    assert continuation == {
        "logical_page": 9,
        "min_y": 0.0,
        "max_y": 200.0,
        "continuation_verified": True,
    }
    assert retained_row["_canonical_segment"][
        "cross_page_continuation_verified"
    ] is True
    matches = native_extraction._match_account_table_observations(
        rows,
        [
            _account_table_observation(
                "credit_card",
                page=9,
                source_page=5,
                top=240.0,
            )
        ],
    )
    assert matches == {}


def test_retained_continuation_uses_authoritative_nonmonotonic_page_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retained_owner = _registered_owner(
        "credit_card",
        1,
        page=10,
        source_page=4,
        top=100.0,
    )
    added_owner = _registered_owner(
        "quasi_credit_card",
        1,
        page=8,
        source_page=5,
        top=200.0,
    )
    _install_registered_population(
        monkeypatch,
        {
            "credit_card": {1: retained_owner},
            "quasi_credit_card": {1: added_owner},
        },
    )
    retained = _existing_skeleton(retained_owner)
    retained["_canonical_segment"]["pages"] = [
        {
            "logical_page": 10,
            "min_y": 100.0,
            "max_y": None,
            "continuation_verified": False,
        },
        {
            "logical_page": 8,
            "min_y": 0.0,
            "max_y": 500.0,
            "continuation_verified": True,
        },
    ]
    retained["_canonical_segment"]["cross_page_continuation_verified"] = True
    context = SimpleNamespace(
        corrected_evidence_pages=lambda: [],
        reading_order_by_logical={10: 1, 8: 2},
        reading_order_resolution={"resolved": True, "authoritative": True},
    )

    rows = native_extraction._materialize_registered_account_population_skeletons(
        context,
        [retained],
    )

    retained_row = next(
        row
        for row in rows
        if row["account_id"] == "credit_account:credit_card:1"
    )
    assert retained_row["_canonical_segment"]["pages"][1] == {
        "logical_page": 8,
        "min_y": 0.0,
        "max_y": 200.0,
        "continuation_verified": True,
    }
    matches = native_extraction._match_account_table_observations(
        rows,
        [
            _account_table_observation(
                "credit_card",
                page=8,
                source_page=5,
                top=120.0,
            )
        ],
    )
    assert matches == {0: 0}


@pytest.mark.parametrize("conflict", ("weak", "duplicate", "wrong_owner"))
def test_registered_materialization_rejects_conflicted_family_atomically(
    monkeypatch: pytest.MonkeyPatch,
    conflict: str,
) -> None:
    owner1 = _registered_owner("credit_card", 1, page=8, source_page=4, top=100.0)
    owner2 = _registered_owner("credit_card", 2, page=8, source_page=4, top=200.0)
    observations = {"credit_card": {1: owner1, 2: owner2}}
    _install_registered_population(monkeypatch, observations)
    first = _existing_skeleton(owner1)
    if conflict == "weak":
        first["account_family_quality"] = "inferred"
        skeletons = [first]
    elif conflict == "duplicate":
        skeletons = [first, deepcopy(first)]
    else:
        first["source_refs"][0]["bbox"] = [20.0, 130.0, 180.0, 142.0]
        skeletons = [first]
    context = SimpleNamespace(corrected_evidence_pages=lambda: [])

    rows = native_extraction._materialize_registered_account_population_skeletons(
        context,
        skeletons,
    )

    assert "credit_account:credit_card:2" not in {
        str(row.get("account_id") or "") for row in rows
    }
    assert context._candidate_b_account_anchor_skeleton_cache == rows


def _table(
    table_id: str,
    rows: list[list[str]],
    *,
    top: float = 100.0,
    missing_evidence: tuple[int, int] | None = None,
) -> SimpleNamespace:
    cell_bboxes = [
        [
            [20.0 + column * 60.0, top + row * 18.0, 80.0 + column * 60.0, top + 18.0 + row * 18.0]
            for column in range(len(values))
        ]
        for row, values in enumerate(rows)
    ]
    cell_evidence_ids = [
        [
            []
            if missing_evidence == (row, column)
            else [f"ocr:{table_id}:{row}:{column}"]
            for column in range(len(values))
        ]
        for row, values in enumerate(rows)
    ]
    return SimpleNamespace(
        table_id=table_id,
        bbox=[20.0, top, 20.0 + len(rows[0]) * 60.0, top + len(rows) * 18.0],
        headers=[],
        rows=[],
        metadata={
            "raw_rows": deepcopy(rows),
            "cell_bboxes": cell_bboxes,
            "cell_evidence_ids": cell_evidence_ids,
        },
    )


def _agreement_census(count: int, *, top: float = 100.0) -> dict:
    return {
        "sequences": list(range(1, count + 1)),
        "ordinal_observations": {
            sequence: {
                "sequence": sequence,
                "source_refs": [
                    {
                        "source": "candidate_b_source_coverage_ledger",
                        "logical_page": 12,
                        "source_page": 6,
                        "geometry_scope": "line",
                        "binding": "printed_credit_agreement_ordinal",
                        "binding_quality": "printed_credit_agreement_ordinal",
                        "sequence": sequence,
                        "bbox": [20.0, top - 22.0, 150.0, top - 2.0],
                        "evidence_ids": [f"agreement:heading:{sequence}"],
                    }
                ],
            }
            for sequence in range(1, count + 1)
        },
        "source_refs": [],
    }


def _frozen_context(*tables: SimpleNamespace) -> SimpleNamespace:
    page = SimpleNamespace(
        page_number=12,
        source_page_number=6,
        tables=list(tables),
        texts=[],
    )
    return SimpleNamespace(
        pages=[page],
        _frozen_logical_pages={12: page},
        _personal_detail_extraction_issues=[],
    )


def test_merged_agreement_header_recovers_guangda_with_raw_correction_trail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identifier = "B10711000H0001100000111111111498898000000"
    rows = [
        ["管理机构", "授信协议标识", "生效日期", "到期日期 授信额度用途", ""],
        [
            "公司 中国光大银行股份有限",
            identifier,
            "2019.05.21",
            "长期",
            "信用卡共享额度",
        ],
        ["授信额度", "授信限额", "授信限额编号", "已用额度", "币种"],
        ["36,400", "--", "--", "36,393", "人民币元"],
    ]
    context = _frozen_context(_table("agreement-guangda", rows))
    monkeypatch.setattr(
        native_extraction,
        "_sealed_agreement_population_census",
        lambda _context: _agreement_census(1),
    )
    monkeypatch.setattr(
        PBOCPersonalDetailNativeParser,
        "records",
        lambda _parser, _dataset: [],
    )

    extracted = native_extraction._extract_credit_lines(context)

    assert len(extracted) == 1
    row = extracted[0]
    assert row["account_identifier"] == identifier
    assert row["institution"] == "中国光大银行股份有限公司"
    assert row["facility_type"] == "信用卡共享额度"
    assert row["validity_type"] == "perpetual"
    assert row["canonical_raw"]["institution"] == "公司 中国光大银行股份有限"
    institution_refs = row["source_refs_by_field"]["institution"]
    assert len(institution_refs) == 1
    assert institution_refs[0]["geometry_scope"] == "cell"
    assert institution_refs[0]["evidence_ids"]
    corrections = [
        issue
        for issue in context._personal_detail_extraction_issues
        if issue.get("issue_code")
        == "candidate_b_credit_agreement_corporate_suffix_rotation_recovered"
    ]
    assert len(corrections) == 1
    assert corrections[0]["observed_value"] == "公司 中国光大银行股份有限"
    assert corrections[0]["candidate_value"] == "中国光大银行股份有限公司"
    assert corrections[0]["source_refs"] == institution_refs


def _agreement_heading_witness(
    ordinal: int,
    identifier: str,
) -> tuple[dict, dict]:
    bbox = [20.0, 100.0 + ordinal * 30.0, 520.0, 112.0 + ordinal * 30.0]
    evidence_id = f"agreement-account:{ordinal}:{identifier}"
    anchor_ref = {
        "source": "candidate_b_account_anchor",
        "logical_page": 20,
        "source_page": 10,
        "geometry_scope": "line",
        "bbox": bbox,
        "evidence_ids": [evidence_id],
    }
    account = {
        "account_id": f"credit_account:credit_card:{ordinal}",
        "account_type": "credit_card",
        "category_sequence": ordinal,
        "account_family_quality": "exact",
        "_printed_ordinal_status": "printed_unique",
        "credit_agreement_identifier": identifier,
        "source_refs": [deepcopy(anchor_ref)],
        "source_refs_by_field": {
            "credit_agreement_identifier": [
                {
                    **anchor_ref,
                    "field_name": "credit_agreement_identifier",
                    "binding": "canonical_account_anchor",
                    "binding_quality": "canonical_account_anchor",
                }
            ]
        },
    }
    heading = f"账户{ordinal}（授信协议标识：{identifier}）（卡片尾号：3906）"
    line = {
        "text": heading,
        "ocr_original_text": heading,
        "bbox": bbox,
        "evidence_ids": [evidence_id],
    }
    return account, line


def _prefix_residue_agreement_context(
    identifier: str,
    account_identifiers: list[str],
) -> SimpleNamespace:
    rows = [
        ["管理机构", "授信协议标识", "生效日期", "到期日期 授信额度用途", ""],
        [
            "公司 中国光大银行股份有限",
            identifier,
            "2019.05.21",
            "长期",
            "信用卡共享额度",
        ],
        ["授信额度", "授信限额", "授信限额编号", "已用额度", "币种"],
        ["36,400 2", "--", "--", "36,393", "人民币元"],
    ]
    context = _frozen_context(_table("agreement-prefix-residue", rows))
    witnesses = [
        _agreement_heading_witness(index + 1, account_identifier)
        for index, account_identifier in enumerate(account_identifiers)
    ]
    context.account_collections = lambda: (
        deepcopy([account for account, _line in witnesses]),
        [],
        [],
    )
    context.corrected_evidence_pages = lambda: [
        {
            "page": 20,
            "source_page": 10,
            "lines": deepcopy([line for _account, line in witnesses]),
        }
    ]
    return context


def test_separated_agreement_prefix_uses_exact_field_grammar_and_preserves_raw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identifier = "B10711000H0001100000111111111498898000000"
    raw_identifier = (
        "R B10711000H00011000 0011111111149889800 0000"
    )
    context = _prefix_residue_agreement_context(
        raw_identifier,
        [identifier, identifier],
    )
    monkeypatch.setattr(
        native_extraction,
        "_sealed_agreement_population_census",
        lambda _context: _agreement_census(1),
    )
    monkeypatch.setattr(
        PBOCPersonalDetailNativeParser,
        "records",
        lambda _parser, _dataset: [],
    )

    rows = native_extraction.reconcile_candidate_b_credit_lines(
        context,
        native_extraction._extract_credit_lines(context),
    )

    assert len(rows) == 1
    assert rows[0]["account_identifier"] == identifier
    assert rows[0]["total_limit"] is None
    assert rows[0]["canonical_raw"]["account_identifier"] == raw_identifier
    issues = [
        issue
        for issue in context._personal_detail_extraction_issues
        if issue.get("issue_code")
        == "candidate_b_credit_agreement_identifier_invalid_leading_glyph_repaired"
    ]
    assert len(issues) == 1
    assert issues[0]["observed_value"]["raw_agreement_cell_identifier"] == raw_identifier
    assert len(issues[0]["source_refs"]) == 1


@pytest.mark.parametrize(
    ("raw_identifier", "account_identifiers", "expected_repair"),
    (
        (
            "R B10711000H00011000 0011111111149889800 0000",
            ["B10711000H0001100000111111111498898000000"],
            True,
        ),
        (
            "RB10711000H0001100000111111111498898000000",
            [
                "B10711000H0001100000111111111498898000000",
                "B10711000H0001100000111111111498898000000",
            ],
            False,
        ),
        (
            "R B10711000H00011000 0011111111149889800 0000",
            [
                "B10711000H0001100000111111111498898000000",
                "B10711000H0001100000111111111498898000000",
                "RB10711000H0001100000111111111498898000000",
            ],
            True,
        ),
    ),
    ids=("one_witness", "unseparated_prefix", "competing_exact_entity"),
)
def test_agreement_prefix_residue_repair_is_local_and_grammar_bounded(
    monkeypatch: pytest.MonkeyPatch,
    raw_identifier: str,
    account_identifiers: list[str],
    expected_repair: bool,
) -> None:
    context = _prefix_residue_agreement_context(
        raw_identifier,
        account_identifiers,
    )
    monkeypatch.setattr(
        native_extraction,
        "_sealed_agreement_population_census",
        lambda _context: _agreement_census(1),
    )
    monkeypatch.setattr(
        PBOCPersonalDetailNativeParser,
        "records",
        lambda _parser, _dataset: [],
    )

    rows = native_extraction.reconcile_candidate_b_credit_lines(
        context,
        native_extraction._extract_credit_lines(context),
    )

    assert len(rows) == 1
    expected_identifier = (
        "B10711000H0001100000111111111498898000000"
        if expected_repair
        else raw_identifier.replace(" ", "")
    )
    assert rows[0]["account_identifier"] == expected_identifier
    repaired = [
        issue
        for issue in context._personal_detail_extraction_issues
        if issue.get("issue_code")
        == "candidate_b_credit_agreement_identifier_invalid_leading_glyph_repaired"
    ]
    assert bool(repaired) is expected_repair
    assert not any(
        issue.get("issue_code")
        == "candidate_b_credit_agreement_identifier_separated_prefix_recovered"
        for issue in context._personal_detail_extraction_issues
    )


def _headerless_agreement_rows(identifier: str) -> list[list[str]]:
    return [
        [
            "招商银行股份有限公司信用卡中心",
            identifier,
            "2016.03.28",
            "长期",
            "信用卡共享额度",
        ],
        ["授信额度", "授信限额", "授信限额编号", "已用额度", "币种"],
        ["50,000", "--", "--", "50,998", "人民币元"],
    ]


def test_headerless_agreement_card_conserves_only_identity_and_ordinal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identifier = "B11115840H0001000000000000000000000001"
    context = _frozen_context(
        _table("agreement-headerless", _headerless_agreement_rows(identifier))
    )
    monkeypatch.setattr(
        native_extraction,
        "_sealed_agreement_population_census",
        lambda _context: _agreement_census(1),
    )

    candidates = native_extraction._sealed_agreement_identity_table_candidates(context)

    assert len(candidates) == 1
    assert candidates[0].fields == {
        "授信限额": "--",
        "授信限额编号": "--",
        "授信协议标识": identifier,
        "__printed_sequence": "1",
    }
    assert candidates[0].source_refs_by_field["授信协议标识"][0]["geometry_scope"] == "cell"
    assert "管理机构" in candidates[0].unresolved_labels
    assert "授信限额" not in candidates[0].unresolved_labels
    assert set(candidates[0].agreement_raw_observations) == (
        set(native_extraction._CREDIT_AGREEMENT_FIELD_NAMES)
        - {"授信协议标识"}
    )
    assert all(
        candidates[0].source_refs_by_field[label][0]["binding"]
        == "canonical_label_slot"
        for label in native_extraction._CREDIT_AGREEMENT_FIELD_NAMES
        if label != "授信协议标识"
    )


def test_headerless_agreement_withheld_slots_emit_raw_field_local_issues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identifier = "B11115840H0001000000000000000000000001"
    context = _frozen_context(
        _table("agreement-headerless-audit", _headerless_agreement_rows(identifier))
    )
    monkeypatch.setattr(
        native_extraction,
        "_sealed_agreement_population_census",
        lambda _context: _agreement_census(1),
    )
    monkeypatch.setattr(
        PBOCPersonalDetailNativeParser,
        "records",
        lambda _parser, _dataset: [],
    )

    rows = native_extraction.reconcile_candidate_b_credit_lines(
        context,
        native_extraction._extract_credit_lines(context),
    )

    assert len(rows) == 1
    row = rows[0]
    expected_raw = {
        "institution": "招商银行股份有限公司信用卡中心",
        "effective_date": "2016.03.28",
        "due_date": "长期",
        "facility_type": "信用卡共享额度",
        "total_limit": "50,000",
        "used_limit": "50,998",
        "currency": "人民币元",
    }
    assert {
        field_name: row["canonical_raw"][field_name]
        for field_name in expected_raw
    } == expected_raw
    assert {"credit_limit", "limit_identifier"}.issubset(
        row["_source_absent_fields"]
    )
    assert row["validity_type"] == "perpetual"
    assert len(row["source_refs_by_field"]["due_date"]) == 1
    expected_issue_raw = {
        field_name: raw_value
        for field_name, raw_value in expected_raw.items()
        if field_name != "due_date"
    }
    unresolved_issues = [
        issue
        for issue in context._personal_detail_extraction_issues
        if issue.get("issue_code")
        == "candidate_b_credit_agreement_required_field_unresolved"
    ]
    assert {issue["field_name"] for issue in unresolved_issues} == set(
        expected_issue_raw
    )
    for issue in unresolved_issues:
        field_name = issue["field_name"]
        assert issue["observed_value"] == expected_issue_raw[field_name]
        assert len(issue["source_refs"]) == 1
        assert issue["source_refs"][0]["geometry_scope"] == "cell"
        assert issue["source_refs"][0]["evidence_ids"]


def test_headerless_agreement_identity_requires_evidence_for_every_fixed_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identifier = "B11115840H0001000000000000000000000001"
    context = _frozen_context(
        _table(
            "agreement-headerless-missing-slot",
            _headerless_agreement_rows(identifier),
            missing_evidence=(2, 4),
        )
    )
    monkeypatch.setattr(
        native_extraction,
        "_sealed_agreement_population_census",
        lambda _context: _agreement_census(1),
    )

    assert native_extraction._sealed_agreement_identity_table_candidates(context) == []


def test_headerless_agreement_identity_fails_closed_on_two_table_owners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identifier = "B11115840H0001000000000000000000000001"
    rows = _headerless_agreement_rows(identifier)
    context = _frozen_context(
        _table("agreement-left", rows),
        _table("agreement-right", rows),
    )
    monkeypatch.setattr(
        native_extraction,
        "_sealed_agreement_population_census",
        lambda _context: _agreement_census(1),
    )

    assert native_extraction._sealed_agreement_identity_table_candidates(context) == []


def _liability_rows(
    contract: str,
    *,
    open_label: str = "开立日期",
    responsibility_label: str = "责任人类型",
    amount_label: str = "还款责任金额",
    amount: str = "56,000",
) -> list[list[str]]:
    return [
        [
            "管理机构",
            "业务种类",
            open_label,
            "",
            "到期日期",
            responsibility_label,
            amount_label,
            "",
            "币种",
            "保证合同编号",
        ],
        [
            "华能贵诚信托有限公司",
            "贷款",
            "2022.09.02",
            "",
            "2024.09.07",
            "保证人",
            amount,
            "",
            "人民币元",
            contract,
        ],
        [
            "主业务借款人",
            "",
            "",
            "主业务借款人证件类型",
            "",
            "",
            "",
            "主业务借款人证件号码",
            "",
            "",
        ],
        [
            "厦门雯玥轩商贸有限公司",
            "",
            "",
            "中征码",
            "",
            "",
            "",
            "35020300119734252",
            "",
            "",
        ],
        ["截至2023年01月07日", "", "", "", "", "", "", "", "", ""],
        ["余额", "", "", "五级分类", "", "", "", "逾期月数", "", ""],
        ["46,667", "", "", "正常", "", "", "", "0", "", ""],
    ]


def test_sealed_liability_tables_materialize_each_unique_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tables = [
        _table(
            "liability-one",
            _liability_rows(
                "Y10061000H0001EIP1967714G01",
                responsibility_label="贵任人类型",
                amount_label="还款贵任金额",
            ),
            top=100.0,
        ),
        _table(
            "liability-two",
            _liability_rows(
                "D10055840H0001DB20220228XS000000109",
                open_label="成立日期",
                amount_label="还款贵任金额",
            ),
            top=260.0,
        ),
        _table(
            "liability-three",
            _liability_rows(
                "70105501018BZYQ20220902XS0M00000460",
                amount="福 成 56,000",
            ),
            top=420.0,
        ),
    ]
    context = _frozen_context(*tables)
    monkeypatch.setattr(
        PBOCPersonalDetailNativeParser,
        "records",
        lambda _parser, _dataset: [],
    )

    rows = native_extraction._extract_liabilities(context)

    assert len(rows) == 3
    assert {row["contract_number"] for row in rows} == {
        "Y10061000H0001EIP1967714G01",
        "D10055840H0001DB20220228XS000000109",
        "70105501018BZYQ20220902XS0M00000460",
    }
    assert all(row["open_date"] == "2022-09-02" for row in rows)
    contaminated = next(
        row
        for row in rows
        if row["contract_number"] == "70105501018BZYQ20220902XS0M00000460"
    )
    assert contaminated["responsibility_amount"] is None
    assert "responsibility_amount" in contaminated["_unresolved_fields"]
    assert contaminated["canonical_raw"]["responsibility_amount"] == "福 成 56,000"
    corrections = [
        issue
        for issue in context._personal_detail_extraction_issues
        if issue.get("issue_code")
        == "candidate_b_repayment_responsibility_fixed_slot_header_alias_recovered"
    ]
    assert len(corrections) == 4
    open_date_correction = next(
        issue
        for issue in corrections
        if issue.get("observed_value") == "成立日期"
    )
    assert open_date_correction["field_name"] == "open_date"
    assert open_date_correction["candidate_value"] == "开立日期"
    assert "slot_semantics_not_global_text_rewrite" in open_date_correction[
        "reason_codes"
    ]
    assert len(open_date_correction["source_refs"]) == 1
    assert open_date_correction["source_refs"][0]["row"] == 0
    assert open_date_correction["source_refs"][0]["column"] == 2


def test_sealed_liability_population_fails_closed_on_duplicate_contracts() -> None:
    contract = "Y10061000H0001EIP1967714G01"
    context = _frozen_context(
        _table("liability-left", _liability_rows(contract), top=100.0),
        _table("liability-right", _liability_rows(contract), top=260.0),
    )

    assert native_extraction._sealed_liability_table_candidates(context) == []


def test_sealed_liability_rejects_duplicate_contract_across_ordinary_and_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = "Y10061000H0001EIP1967714G01"
    context = _frozen_context(
        _table(
            "liability-fallback-owner",
            _liability_rows(contract, amount_label="还款贵任金额"),
            top=260.0,
        )
    )
    ordinary_table_ref = {
        "source": "native_detail_tolerant_table",
        "logical_page": 12,
        "source_page": 6,
        "table_id": "liability-ordinary-owner",
        "geometry_scope": "table",
        "bbox": [20.0, 100.0, 620.0, 226.0],
    }
    ordinary_contract_ref = {
        **ordinary_table_ref,
        "source": "native_detail_tolerant_table_cell",
        "geometry_scope": "cell",
        "row": 1,
        "column": 9,
        "field_name": "contract_number",
        "binding": "label_column",
        "bbox": [560.0, 118.0, 620.0, 136.0],
        "evidence_ids": ["liability:ordinary:contract"],
    }
    ordinary = SimpleNamespace(
        fields={"保证合同编号": contract},
        source_refs=(ordinary_table_ref,),
        confidence=1.0,
        source_refs_by_field={"保证合同编号": (ordinary_contract_ref,)},
        binding_quality_by_field={"保证合同编号": "native_label_column"},
        observed_labels=frozenset({"保证合同编号"}),
        unresolved_labels=frozenset(),
    )
    monkeypatch.setattr(
        PBOCPersonalDetailNativeParser,
        "records",
        lambda _parser, _dataset: [ordinary],
    )

    rows = native_extraction._extract_liabilities(context)

    assert len(rows) == 1
    assert rows[0]["contract_number"] == contract
    assert {
        ref.get("table_id") for ref in rows[0]["source_refs"]
    } == {"liability-ordinary-owner"}


def test_sealed_liability_requires_exact_header_cell_evidence() -> None:
    context = _frozen_context(
        _table(
            "liability-missing-evidence",
            _liability_rows("Y10061000H0001EIP1967714G01"),
            missing_evidence=(0, 6),
        )
    )

    assert native_extraction._sealed_liability_table_candidates(context) == []


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("open_label", "设立日期"),
        ("responsibility_label", "义务人类型"),
        ("amount_label", "担保责任金额"),
    ),
)
def test_sealed_liability_rejects_unlisted_header_alias(
    field: str,
    value: str,
) -> None:
    kwargs = {field: value}
    context = _frozen_context(
        _table(
            "liability-unlisted-header",
            _liability_rows("Y10061000H0001EIP1967714G01", **kwargs),
        )
    )

    assert native_extraction._sealed_liability_table_candidates(context) == []
