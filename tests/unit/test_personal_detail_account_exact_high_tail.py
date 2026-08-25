from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from docmirror.plugins.credit_report.community_plugin import (
    _compact_personal_detail_public_projection,
)
from docmirror.plugins.credit_report.personal_detail_scanned import native_extraction
from docmirror.plugins.credit_report.personal_detail_scanned.schema import (
    project_personal_detail_datasets,
)
from docmirror.plugins.credit_report.personal_detail_scanned.source_projection import (
    prepare_personal_detail_source_collections,
)

_FAMILY = "credit_card"


def _sealed_anchor_fixture(
    values: set[int],
    *,
    scale: float = 1.0,
    reverse_observations: bool = False,
) -> tuple[SimpleNamespace, dict[int, dict[str, object]]]:
    ordered = sorted(values)
    observations: dict[int, dict[str, object]] = {}
    document_order: dict[int, int] = {}
    for index, ordinal in enumerate(ordered):
        page_slot = index // 2
        logical_page = 900 - page_slot * 17
        source_page = 40 + page_slot
        document_order[logical_page] = page_slot + 1
        top = (20.0 + (index % 2) * 30.0) * scale
        observations[ordinal] = {
            "account_id": f"credit_account:{_FAMILY}:{ordinal}",
            "source_refs": [
                {
                    "source": "candidate_b_account_anchor",
                    "logical_page": logical_page,
                    "source_page": source_page,
                    "geometry_scope": "line",
                    "binding": "printed_account_ordinal",
                    "binding_quality": "printed_account_ordinal",
                    "account_type": _FAMILY,
                    "category_sequence": ordinal,
                    "bbox": [10.0 * scale, top, 110.0 * scale, top + 8.0 * scale],
                    "evidence_ids": [f"anchor:{ordinal}"],
                }
            ],
        }
    if reverse_observations:
        observations = {ordinal: observations[ordinal] for ordinal in reversed(list(observations))}
        document_order = {logical_page: document_order[logical_page] for logical_page in reversed(list(document_order))}
    context = SimpleNamespace(
        pages=[],
        reading_order_by_logical=document_order,
        reading_order_resolution={"resolved": True, "authoritative": True},
        corrected_evidence_pages=lambda: [],
    )
    return context, observations


@pytest.mark.parametrize(
    ("values", "endpoint", "scale", "reverse_observations"),
    [
        ({1, 8, 9}, 9, 0.55, False),
        ({1, 2, 3, 12, 13}, 13, 1.0, True),
        ({1, 2, 3, 4, 5, 6, 18, 19}, 19, 2.4, False),
    ],
)
def test_exact_source_high_tail_accepts_variable_population_order_and_scale(
    values: set[int],
    endpoint: int,
    scale: float,
    reverse_observations: bool,
) -> None:
    context, observations = _sealed_anchor_fixture(
        values,
        scale=scale,
        reverse_observations=reverse_observations,
    )

    result = native_extraction._exact_source_supported_account_sequence_endpoint(
        context,
        values,
        family=_FAMILY,
        observations=observations,
    )

    assert result == (endpoint, [])


def test_yu_like_tail_adds_only_the_exact_missing_ordinal_range() -> None:
    values = {1, 2, 3, 12, 13}
    context, observations = _sealed_anchor_fixture(values)

    generic = native_extraction._credible_sequence_endpoint(values)
    endpoint, outliers = native_extraction._exact_source_supported_account_sequence_endpoint(
        context,
        values,
        family=_FAMILY,
        observations=observations,
    )

    assert generic == (3, [12, 13])
    assert endpoint == 13
    assert outliers == []
    assert sorted(set(range(1, endpoint + 1)) - values) == list(range(4, 12))


def test_lin_like_sparse_exact_tail_reaches_the_printed_high_endpoint() -> None:
    values = {1, 2, 5, 6, 7, 8, 9, 13, 14, 18, 19}
    context, observations = _sealed_anchor_fixture(
        values,
        scale=1.7,
        reverse_observations=True,
    )

    assert native_extraction._credible_sequence_endpoint(values) == (
        13,
        [14, 18, 19],
    )
    assert native_extraction._exact_source_supported_account_sequence_endpoint(
        context,
        values,
        family=_FAMILY,
        observations=observations,
    ) == (19, [])


def test_isolated_exact_high_ordinal_remains_an_outlier() -> None:
    values = {1, 2, 3, 115}
    context, observations = _sealed_anchor_fixture(values)

    assert native_extraction._exact_source_supported_account_sequence_endpoint(
        context,
        values,
        family=_FAMILY,
        observations=observations,
    ) == (3, [115])


@pytest.mark.parametrize(
    "defect",
    [
        "duplicate_owner",
        "replayed_evidence",
        "replayed_geometry",
        "reversed_document_order",
        "foreign_family",
        "unsealed_order",
    ],
)
def test_exact_source_high_tail_fails_closed_on_ambiguous_ownership(
    defect: str,
) -> None:
    values = {1, 2, 3, 12, 13}
    context, observations = _sealed_anchor_fixture(values)
    tail_left = observations[12]["source_refs"][0]
    tail_right = observations[13]["source_refs"][0]
    assert isinstance(tail_left, dict)
    assert isinstance(tail_right, dict)

    if defect == "duplicate_owner":
        observations[13]["source_refs"].append(deepcopy(tail_right))
    elif defect == "replayed_evidence":
        tail_right["evidence_ids"] = list(tail_left["evidence_ids"])
    elif defect == "replayed_geometry":
        tail_right["source_page"] = tail_left["source_page"]
        tail_right["bbox"] = list(tail_left["bbox"])
    elif defect == "reversed_document_order":
        left_page = int(tail_left["logical_page"])
        right_page = int(tail_right["logical_page"])
        context.reading_order_by_logical[left_page], context.reading_order_by_logical[right_page] = (
            context.reading_order_by_logical[right_page],
            context.reading_order_by_logical[left_page],
        )
    elif defect == "foreign_family":
        tail_right["account_type"] = "non_revolving_loan"
    elif defect == "unsealed_order":
        context.reading_order_resolution = {
            "resolved": True,
            "authoritative": False,
        }

    assert native_extraction._exact_source_supported_account_sequence_endpoint(
        context,
        values,
        family=_FAMILY,
        observations=observations,
    ) == (3, [12, 13])


def _skeletons_from_observations(
    observations: dict[int, dict[str, object]],
) -> list[dict[str, object]]:
    skeletons: list[dict[str, object]] = []
    for ordinal in reversed(list(observations)):
        observation = observations[ordinal]
        refs = observation["source_refs"]
        assert isinstance(refs, list) and len(refs) == 1
        skeletons.append(
            {
                "account_id": f"credit_account:{_FAMILY}:{ordinal}",
                "account_type": _FAMILY,
                "account_family_quality": "exact",
                "category_sequence": ordinal,
                "_printed_ordinal_status": "printed_unique",
                "_canonical_segment": {"ownership_basis": "printed_anchor_to_next_anchor"},
                "source_refs": [deepcopy(refs[0])],
            }
        )
    return skeletons


def test_account_gap_issue_uses_the_exact_high_tail_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = {1, 2, 3, 12, 13}
    context, observations = _sealed_anchor_fixture(values)
    context._personal_detail_extraction_issues = []
    skeletons = _skeletons_from_observations(observations)
    monkeypatch.setattr(
        native_extraction,
        "_extract_table_accounts",
        lambda _context: ([], [], []),
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

    accounts, repayments, events = native_extraction._extract_accounts(context)
    provisional = next(
        issue
        for issue in context._personal_detail_extraction_issues
        if issue.get("issue_code") == "candidate_b_account_sequence_gap"
    )
    assert provisional["candidate_value"]["outlier_category_sequences"] == [12, 13]
    ledger = {
        "credit_accounts": 13,
        "account_family_endpoints": {_FAMILY: 13},
        "account_family_anchor_inventory_sequences": {
            _FAMILY: [1, 2, 3, 12, 13]
        },
        "account_family_ordinal_observations": {
            _FAMILY: {
                str(ordinal): deepcopy(observation)
                for ordinal, observation in observations.items()
            }
        },
        "account_family_exact_source_high_tail_endpoints": {_FAMILY: 13},
    }
    first_audit = native_extraction.reconcile_candidate_b_account_sequence_issues(
        context,
        ledger,
        accounts,
    )
    # Reconciliation is idempotent and must retain one stable family-local issue.
    second_audit = native_extraction.reconcile_candidate_b_account_sequence_issues(
        context,
        ledger,
        accounts,
    )

    gaps = [
        issue
        for issue in context._personal_detail_extraction_issues
        if issue.get("issue_code") == "candidate_b_account_sequence_gap"
    ]
    assert len(accounts) == len(values)
    assert repayments == []
    assert events == []
    assert len(gaps) == 1
    assert first_audit == second_audit == {
        "reconciled_families": [_FAMILY],
        "localized_missing_identities": sorted(
            f"credit_account:{_FAMILY}:{ordinal}" for ordinal in range(4, 12)
        ),
    }
    gap = gaps[0]
    assert gap["target_dataset"] == "credit_accounts"
    assert gap["observed_value"] == {
        "account_type": _FAMILY,
        "observed_category_sequences": [1, 2, 3, 12, 13],
    }
    assert gap["candidate_value"]["missing_category_sequences"] == list(
        range(4, 12)
    )
    assert gap["candidate_value"]["outlier_category_sequences"] == []
    assert gap["candidate_value"]["authoritative_family_endpoint"] == 13
    assert [ref["category_sequence"] for ref in gap["source_refs"]] == [
        1,
        2,
        3,
        12,
        13,
    ]
    assert all(
        ref.get("source") == "candidate_b_account_anchor"
        and ref.get("binding") == "printed_account_ordinal"
        and ref.get("binding_quality") == "printed_account_ordinal"
        and ref.get("account_type") == _FAMILY
        for ref in gap["source_refs"]
    )
    projected = prepare_personal_detail_source_collections(
        {
            "facts": {"personal_detail_source_completeness_ledger": ledger},
            "datasets": {
                "credit_accounts": accounts,
                "personal_detail_extraction_issues": gaps,
            },
        }
    )
    projected_gap = next(
        issue
        for issue in projected["datasets"]["personal_detail_extraction_issues"]
        if issue.get("issue_code") == "candidate_b_account_sequence_gap"
    )
    assert projected_gap["candidate_value"] == gap["candidate_value"]
    assert len(accounts) + len(
        projected_gap["candidate_value"]["missing_category_sequences"]
    ) == ledger["credit_accounts"]
    schema_projection = project_personal_detail_datasets(
        {"personal_detail_extraction_issues": [projected_gap]}
    )
    schema_issue = schema_projection["extraction_issues"][0]
    schema_evidence = schema_projection["extraction_issue_evidence"]
    page_range = [
        min(ref["logical_page"] for ref in gap["source_refs"]),
        max(ref["logical_page"] for ref in gap["source_refs"]),
    ]
    community = {
        "datasets": [
            {
                "name": "extraction_issues",
                "rows": [
                    {
                        "record_id": schema_issue["record_id"],
                        "normalized": deepcopy(schema_issue),
                        "canonical_raw": {},
                        "raw": {},
                        "source": {"page_range": page_range},
                    }
                ],
                "columns": [],
            },
            {
                "name": "extraction_issue_evidence",
                "rows": [
                    {
                        "record_id": evidence["record_id"],
                        "normalized": deepcopy(evidence),
                        "canonical_raw": {},
                        "raw": {},
                        "source": deepcopy(evidence.get("source") or {}),
                    }
                    for evidence in schema_evidence
                ],
                "columns": [],
            },
        ]
    }
    _compact_personal_detail_public_projection(
        community,
        source_datasets=[
            SimpleNamespace(
                public={"name": "extraction_issues"},
                rows=schema_projection["extraction_issues"],
            ),
            SimpleNamespace(
                public={"name": "extraction_issue_evidence"},
                rows=schema_evidence,
            ),
        ],
    )
    community_datasets = {
        dataset["name"]: dataset for dataset in community["datasets"]
    }
    community_issue = community_datasets["extraction_issues"]["rows"][0]
    assert community_issue["normalized"]["issue_code"] == (
        "candidate_b_account_sequence_gap"
    )
    community_evidence = [
        row["normalized"]
        for row in community_datasets["extraction_issue_evidence"]["rows"]
    ]
    assert {
        row["integer_value"]
        for row in community_evidence
        if row.get("evidence_kind") == "candidate"
        and str(row.get("evidence_path") or "").startswith(
            "missing_category_sequences["
        )
    } == set(range(4, 12))
    assert any(
        row.get("evidence_kind") == "observed"
        and row.get("evidence_path") == "account_type"
        and row.get("string_value") == _FAMILY
        for row in community_evidence
    )


def test_source_ledger_uses_only_the_sealed_high_tail_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = {1, 2, 3, 12, 13}
    context, observations = _sealed_anchor_fixture(values, reverse_observations=True)
    skeletons = _skeletons_from_observations(observations)
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
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        native_extraction,
        "_exact_account_table_cell_anchors",
        lambda _context: {},
    )
    monkeypatch.setattr(
        native_extraction,
        "_sealed_raw_profile_population_census",
        lambda _context: None,
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
        "_raw_physical_account_source_coverage",
        lambda _context: {},
    )
    monkeypatch.setattr(
        native_extraction,
        "_inquiry_source_coverage",
        lambda _context: {},
    )

    ledger = native_extraction._source_completeness_ledger(context)

    assert ledger["credit_accounts"] == 13
    assert ledger["account_family_endpoints"] == {_FAMILY: 13}
    assert ledger["account_family_source_populations"] == {_FAMILY: 13}
    assert ledger["account_family_exact_source_high_tail_endpoints"] == {_FAMILY: 13}
    assert ledger["account_family_anchor_inventory_sequences"] == {_FAMILY: [1, 2, 3, 12, 13]}
    assert "account_family_sequence_outliers" not in ledger
