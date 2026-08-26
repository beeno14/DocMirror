# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Executor-level parity for deployed Candidate B extraction strategies.

The planner tests prove graph decisions; this simulated full PBOC document
proves that the actual ``CandidateBPipeline`` dispatcher invokes the unchanged
stage callbacks in legacy order and produces identical business content under
explicit eager, complete-census lazy, and fail-closed eager-fallback plans.
"""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest

from docmirror.plugins.credit_report.personal_detail_scanned import (
    candidate_b,
    consistency_ledger,
    document_glyph_bank,
    extraction_issues,
    native_extraction,
    native_status_conflict,
    profile_extraction,
    relations,
    source_projection,
)
from docmirror.plugins.credit_report.personal_detail_scanned.candidate_b import (
    CandidateBPipeline,
)
from docmirror.plugins.credit_report.personal_detail_scanned.candidate_b_strategy import (
    CANDIDATE_B_STAGE_REGISTRY,
    REPORT_HEADER_SECTION,
    SECTION_TO_CANONICAL_DATASETS,
)
from docmirror.plugins.credit_report.personal_detail_scanned.extraction_strategy import (
    EagerExtractionStrategy,
    LazyExtractionStrategy,
)


# Each deployed stage is tied to the exact public Candidate B boundary where a
# business caller observes its unchanged strategy output.  This makes the
# parity oracle fail if a stage is dispatched but its result is dropped,
# renamed, or moved during lazy orchestration.
_STAGE_PUBLIC_OUTPUT_ORACLES: dict[
    str,
    tuple[tuple[str, str], ...],
] = {
    "account_inventory": (
        ("business", "credit_accounts"),
        ("datasets", "personal_detail_account_events"),
    ),
    "monthly_repayments": (("business", "repayment_records"),),
    "credit_agreements": (("business", "credit_lines"),),
    "liabilities": (("business", "repayment_liability_records"),),
    "overdue": (("business", "overdue_records"),),
    "inquiries": (("business", "inquiry_records"),),
    "public": (("business", "public_records"),),
    "notes": (
        ("datasets", "annotations"),
        ("datasets", "statements"),
    ),
    "summary": (
        ("datasets", "personal_detail_summary_records"),
        ("datasets", "personal_detail_summary_cells"),
    ),
    "header": (
        ("datasets", "personal_report_metadata"),
        ("datasets", "identity_documents"),
    ),
    "recovery": (("datasets", "recovery_records"),),
    "postpaid_records": (("datasets", "postpaid_records"),),
    "postpaid_history": (("datasets", "postpaid_payment_history"),),
    "residence": (("datasets", "residence_records"),),
    "employment": (("datasets", "employment_records"),),
    "source_rows": (("datasets", "personal_detail_source_rows"),),
    "profile_details": (
        ("datasets", "mobile_phone_records"),
        ("datasets", "spouse_records"),
    ),
    "profile": (("facts", "subject_profile"),),
}


def test_every_deployed_stage_has_an_exact_public_business_oracle() -> None:
    assert tuple(_STAGE_PUBLIC_OUTPUT_ORACLES) == (
        CANDIDATE_B_STAGE_REGISTRY.ordered()
    )
    for stage in CANDIDATE_B_STAGE_REGISTRY.stages:
        public_names = {
            output_name
            for _boundary, output_name in _STAGE_PUBLIC_OUTPUT_ORACLES[
                stage.name
            ]
        }
        assert public_names == set(stage.output_names).difference(
            {"status_glyph_observations"}
        )


def _full_business_audit(*, unresolved_pages: tuple[int, ...] = ()) -> dict[str, Any]:
    sections = tuple(SECTION_TO_CANONICAL_DATASETS)
    registrations = [
        {
            "logical_page": page,
            "source_page": (page + 1) // 2,
            "status": "registered",
            "template_id": section,
            "basis": "simulated_exact_heading_and_table_signature",
            "affected_source_datasets": sorted(
                SECTION_TO_CANONICAL_DATASETS[section]
            ),
            "printed_page": page,
            "printed_total": len(sections),
        }
        for page, section in enumerate(sections, start=1)
    ]
    assert registrations[0]["template_id"] == REPORT_HEADER_SECTION
    return {
        "corrected_evidence_conservation": {
            "valid": True,
            "conserved_logical_pages": list(range(1, len(sections) + 1)),
        },
        "canonical_subset_conservation": {"valid": True},
        "unresolved_pages": list(unresolved_pages),
        "registrations": registrations,
        "fragment_groups": [
            {
                "template_id": section,
                "fragment_logical_pages": [page],
                "canonical_page": page,
                "coverage_status": "full",
                "coverage_ratio": 1.0,
            }
            for page, section in enumerate(sections, start=1)
        ],
    }


def _header_only_audit() -> dict[str, Any]:
    return {
        "corrected_evidence_conservation": {
            "valid": True,
            "conserved_logical_pages": [1],
        },
        "canonical_subset_conservation": {"valid": True},
        "unresolved_pages": [],
        "registrations": [
            {
                "logical_page": 1,
                "source_page": 1,
                "status": "registered",
                "template_id": REPORT_HEADER_SECTION,
                "basis": "simulated_exact_report_header",
                "affected_source_datasets": sorted(
                    SECTION_TO_CANONICAL_DATASETS[REPORT_HEADER_SECTION]
                ),
                "printed_page": 1,
                "printed_total": 1,
            }
        ],
        "fragment_groups": [
            {
                "template_id": REPORT_HEADER_SECTION,
                "fragment_logical_pages": [1],
                "canonical_page": 1,
                "coverage_status": "full",
                "coverage_ratio": 1.0,
            }
        ],
    }


def _context(audit: dict[str, Any]) -> SimpleNamespace:
    context = SimpleNamespace(
        _cache={},
        _personal_detail_extraction_issues=[],
        pages=[],
        reading_order_by_logical={},
        stage_calls=[],
        status_glyph_calls=[],
        corrected_repayment_micro_grids=lambda: [],
        corrected_evidence_pages=lambda: [],
        candidate_b_status_glyph_observations=None,
        correct_candidate_b_datasets=lambda payload: deepcopy(payload),
        prepare_candidate_b_business_repair=lambda _payload: False,
        canonical_layout_audit=lambda: deepcopy(audit),
        page_topology_audit=lambda: {"valid": True},
        ocr_correction_audit=lambda: {
            "business_repair": {"second_schema_pass_required": False}
        },
    )

    def status_glyph_observations():
        context.status_glyph_calls.append("monthly_repayments")
        return [{"observation_id": "status-glyph:1"}]

    def account_collections():
        context.stage_calls.append("account_inventory")
        return (
            [
                {
                    "account_id": "credit_account:credit_card:1",
                    "account_type": "credit_card",
                    "institution": "示例银行股份有限公司",
                }
            ],
            [],
            [{"account_event_id": "account-event:1"}],
        )

    def repayments():
        context.stage_calls.append("monthly_repayments")
        return [
            {
                "repayment_id": "grid:1:2024-01",
                "account_id": "credit_account:credit_card:1",
                "grid_id": "grid:1",
                "year": 2024,
                "month": 1,
                "status": "N",
                "overdue_amount": "0",
            }
        ]

    context.account_collections = account_collections
    context.corrected_repayment_records = repayments
    context.candidate_b_status_glyph_observations = status_glyph_observations
    return context


def _install_pipeline_stubs(
    monkeypatch: pytest.MonkeyPatch,
    context: SimpleNamespace,
) -> None:
    def one_row(stage_name: str, dataset_name: str):
        def extract(_context: Any):
            assert _context is context
            context.stage_calls.append(stage_name)
            return [{f"{dataset_name[:-1] if dataset_name.endswith('s') else dataset_name}_id": f"{dataset_name}:1"}]

        return extract

    def extract_credit_lines(_context: Any):
        assert _context is context
        context.stage_calls.append("credit_agreements")
        return [{"credit_line_id": "credit-line:1", "institution": "示例银行股份有限公司"}]

    def extract_liabilities(_context: Any):
        assert _context is context
        context.stage_calls.append("liabilities")
        return [
            {
                "repayment_responsibility_id": "liability:1",
                "institution": "示例担保机构",
                "business_type": "贷款",
            }
        ]

    def extract_inquiries(_context: Any):
        assert _context is context
        context.stage_calls.append("inquiries")
        return [
            {
                "inquiry_id": "inquiry:1",
                "inquiry_type": "institution",
                "sequence": 1,
            }
        ]

    def extract_public(_context: Any):
        assert _context is context
        context.stage_calls.append("public")
        return [{"public_record_id": "public:1"}]

    def extract_notes(_context: Any):
        assert _context is context
        context.stage_calls.append("notes")
        return (
            [{"annotation_id": "annotation:1"}],
            [{"statement_id": "statement:1"}],
        )

    def extract_summary(_context: Any):
        assert _context is context
        context.stage_calls.append("summary")
        return (
            [{"summary_record_id": "summary:1"}],
            [{"summary_cell_id": "summary-cell:1"}],
        )

    def extract_header(_context: Any, full_text: str):
        assert _context is context
        assert full_text == "simulated full PBOC report"
        context.stage_calls.append("header")
        return {
            "personal_report_metadata": [{"report_id": "report:1"}],
            "identity_documents": [{"identity_document_id": "identity:1"}],
        }

    def extract_profile_details(_context: Any):
        assert _context is context
        context.stage_calls.append("profile_details")
        return {
            "mobile_phone_records": [{"mobile_phone_id": "mobile:1"}],
            "spouse_records": [{"spouse_id": "spouse:1"}],
        }

    def extract_profile(_context: Any):
        assert _context is context
        context.stage_calls.append("profile")
        return {"subject_id": "subject:1", "subject_name": "示例客户"}

    def overdue(accounts: Any, repayments: Any):
        assert accounts and repayments
        context.stage_calls.append("overdue")
        return [{"overdue_id": "overdue:1"}]

    monkeypatch.setattr(
        native_extraction,
        "_extract_credit_lines",
        extract_credit_lines,
    )
    monkeypatch.setattr(
        native_extraction,
        "_extract_liabilities",
        extract_liabilities,
    )
    monkeypatch.setattr(
        native_extraction,
        "_extract_inquiries",
        extract_inquiries,
    )
    monkeypatch.setattr(
        native_extraction,
        "_extract_public_records",
        extract_public,
    )
    monkeypatch.setattr(
        native_extraction,
        "_extract_personal_notes",
        extract_notes,
    )
    monkeypatch.setattr(
        native_extraction,
        "_extract_summary_datasets",
        extract_summary,
    )
    monkeypatch.setattr(
        native_extraction,
        "_extract_header_datasets",
        extract_header,
    )
    monkeypatch.setattr(
        native_extraction,
        "_extract_recovery_records",
        one_row("recovery", "recovery_records"),
    )
    monkeypatch.setattr(
        native_extraction,
        "_extract_postpaid_records",
        one_row("postpaid_records", "postpaid_records"),
    )
    monkeypatch.setattr(
        native_extraction,
        "_extract_postpaid_payment_history",
        one_row("postpaid_history", "postpaid_payment_history"),
    )
    monkeypatch.setattr(
        native_extraction,
        "_extract_residence_records",
        one_row("residence", "residence_records"),
    )
    monkeypatch.setattr(
        native_extraction,
        "_extract_employment_records",
        one_row("employment", "employment_records"),
    )
    monkeypatch.setattr(
        native_extraction,
        "_extract_source_rows",
        one_row("source_rows", "personal_detail_source_rows"),
    )
    monkeypatch.setattr(
        native_extraction,
        "_extract_profile_detail_records",
        extract_profile_details,
    )
    monkeypatch.setattr(
        native_extraction,
        "_record_pre_repair_source_gaps",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        native_extraction,
        "_source_completeness_ledger",
        lambda _context: {},
    )
    monkeypatch.setattr(
        native_extraction,
        "_enforce_employment_record_contracts",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        native_extraction,
        "reconcile_candidate_b_credit_lines",
        lambda _context, rows: deepcopy(rows),
    )
    monkeypatch.setattr(
        native_extraction,
        "reconcile_candidate_b_account_sequence_issues",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        profile_extraction,
        "extract_candidate_b_profile",
        extract_profile,
    )
    monkeypatch.setattr(
        relations,
        "candidate_b_repayment_anchor_ledger",
        lambda *_args: {},
    )
    monkeypatch.setattr(
        relations,
        "link_candidate_b_repayments",
        lambda rows, *_args, **_kwargs: deepcopy(rows),
    )
    monkeypatch.setattr(
        relations,
        "derive_candidate_b_overdue_records",
        overdue,
    )
    monkeypatch.setattr(
        candidate_b,
        "_reconcile_candidate_b_header_lifecycle",
        lambda _context, _before, after: after,
    )
    monkeypatch.setattr(
        candidate_b,
        "_withhold_independent_plane_conflicts",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        candidate_b,
        "_reconcile_final_account_field_issues",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        consistency_ledger,
        "apply_document_consistency_ledger",
        lambda *_args: {},
    )
    monkeypatch.setattr(
        document_glyph_bank,
        "apply_document_local_status_glyph_bank",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        native_status_conflict,
        "apply_candidate_b_native_status_conflict_guard",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        extraction_issues,
        "register_final_liability_issue_records",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        extraction_issues,
        "collect_extraction_issues",
        lambda issue_context: deepcopy(
            getattr(issue_context, "_personal_detail_extraction_issues", [])
        ),
    )
    monkeypatch.setattr(
        extraction_issues,
        "dataset_states_from_issues",
        lambda _issues: {},
    )
    monkeypatch.setattr(
        source_projection,
        "prepare_personal_detail_source_collections",
        lambda payload, _business, **_kwargs: deepcopy(payload),
    )


def _make_optional_stages_empty(
    monkeypatch: pytest.MonkeyPatch,
    context: SimpleNamespace,
) -> None:
    """Keep eager callbacks observable while modelling absent source sections."""

    def empty_account_collections():
        context.stage_calls.append("account_inventory")
        return [], [], []

    def empty_repayments():
        context.stage_calls.append("monthly_repayments")
        return []

    def empty(stage_name: str):
        def extract(_context: Any):
            assert _context is context
            context.stage_calls.append(stage_name)
            return []

        return extract

    def empty_pair(stage_name: str):
        def extract(_context: Any):
            assert _context is context
            context.stage_calls.append(stage_name)
            return [], []

        return extract

    def empty_overdue(_accounts: Any, _repayments: Any):
        context.stage_calls.append("overdue")
        return []

    context.account_collections = empty_account_collections
    context.corrected_repayment_records = empty_repayments
    for function_name, stage_name in (
        ("_extract_credit_lines", "credit_agreements"),
        ("_extract_liabilities", "liabilities"),
        ("_extract_inquiries", "inquiries"),
        ("_extract_public_records", "public"),
        ("_extract_recovery_records", "recovery"),
        ("_extract_postpaid_records", "postpaid_records"),
        ("_extract_postpaid_payment_history", "postpaid_history"),
    ):
        monkeypatch.setattr(
            native_extraction,
            function_name,
            empty(stage_name),
        )
    monkeypatch.setattr(
        native_extraction,
        "_extract_personal_notes",
        empty_pair("notes"),
    )
    monkeypatch.setattr(
        native_extraction,
        "_extract_summary_datasets",
        empty_pair("summary"),
    )
    monkeypatch.setattr(
        relations,
        "derive_candidate_b_overdue_records",
        empty_overdue,
    )


def _detached_public_output(result: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compare business semantics, keeping strategy audit assertions separate."""

    assert result.business["credit_extraction_audit"] is result.audit
    business = {
        key: deepcopy(value)
        for key, value in result.business.items()
        if key != "credit_extraction_audit"
    }
    return business, deepcopy(result.section_content)


def test_deployed_executor_preserves_full_business_output_across_strategies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = (
        (
            "legacy_eager",
            EagerExtractionStrategy(),
            _full_business_audit(),
            "eager",
        ),
        (
            "complete_census_lazy",
            LazyExtractionStrategy(),
            _full_business_audit(),
            "lazy",
        ),
        (
            "ambiguous_census_eager_fallback",
            LazyExtractionStrategy(),
            _full_business_audit(unresolved_pages=(2,)),
            "eager_fallback",
        ),
    )
    baseline: tuple[dict[str, Any], dict[str, Any]] | None = None
    for case_name, strategy, audit, expected_mode in cases:
        context = _context(audit)
        _install_pipeline_stubs(monkeypatch, context)

        result = CandidateBPipeline(
            context,
            "simulated full PBOC report",
            extraction_strategy=strategy,
        ).run()

        assert context.stage_calls == list(CANDIDATE_B_STAGE_REGISTRY.ordered())
        assert context.status_glyph_calls == ["monthly_repayments"]
        plan = result.audit["source_extraction_strategy"]["discovery"]["plan"]
        assert plan["mode"] == expected_mode
        assert plan["ordered_stages"] == list(CANDIDATE_B_STAGE_REGISTRY.ordered())
        assert plan["skipped_stages"] == []
        snapshots = {
            row["stage"]: row
            for row in result.audit["source_extraction_strategy"]["discovery"][
                "stage_snapshots"
            ]
        }
        assert snapshots["monthly_repayments"]["record_counts"][
            "status_glyph_observations"
        ] == 1

        public_boundaries = {
            "business": result.business,
            "datasets": result.section_content["datasets"],
            "facts": result.section_content["facts"],
        }
        for stage_name, oracles in _STAGE_PUBLIC_OUTPUT_ORACLES.items():
            for boundary_name, output_name in oracles:
                assert output_name in public_boundaries[boundary_name], (
                    stage_name,
                    boundary_name,
                    output_name,
                )
                assert public_boundaries[boundary_name][output_name], (
                    stage_name,
                    boundary_name,
                    output_name,
                )

        expected_business_ids = {
            "credit_accounts": "credit_account:credit_card:1",
            "repayment_records": "grid:1:2024-01",
            "overdue_records": "overdue:1",
            "credit_lines": "credit-line:1",
            "repayment_liability_records": "liability:1",
            "inquiry_records": "inquiry:1",
            "public_records": "public:1",
        }
        for dataset_name, expected_id in expected_business_ids.items():
            assert len(result.business[dataset_name]) == 1
            assert expected_id in set(result.business[dataset_name][0].values())

        supplemental = result.section_content["datasets"]
        for dataset_name in (
            "personal_report_metadata",
            "identity_documents",
            "personal_detail_account_events",
            "recovery_records",
            "postpaid_records",
            "postpaid_payment_history",
            "personal_detail_summary_records",
            "personal_detail_summary_cells",
            "residence_records",
            "employment_records",
            "annotations",
            "statements",
            "personal_detail_source_rows",
            "mobile_phone_records",
            "spouse_records",
        ):
            assert len(supplemental[dataset_name]) == 1, dataset_name
        assert result.section_content["facts"]["subject_profile"] == {
            "subject_id": "subject:1",
            "subject_name": "示例客户",
        }

        detached = _detached_public_output(result)
        if baseline is None:
            baseline = detached
        else:
            assert detached == baseline, case_name


def test_sparse_report_lazy_skips_absent_optional_callbacks_with_eager_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mandatory_stages = (
        "header",
        "residence",
        "employment",
        "source_rows",
        "profile_details",
        "profile",
    )
    lazy_output: tuple[dict[str, Any], dict[str, Any]] | None = None
    for case_name, strategy, expected_calls, expected_status_glyph_calls in (
        (
            "header_only_lazy",
            LazyExtractionStrategy(),
            mandatory_stages,
            (),
        ),
        (
            "header_only_explicit_eager",
            EagerExtractionStrategy(),
            CANDIDATE_B_STAGE_REGISTRY.ordered(),
            ("monthly_repayments",),
        ),
    ):
        context = _context(_header_only_audit())
        _install_pipeline_stubs(monkeypatch, context)
        _make_optional_stages_empty(monkeypatch, context)

        result = CandidateBPipeline(
            context,
            "simulated full PBOC report",
            extraction_strategy=strategy,
        ).run()

        assert tuple(context.stage_calls) == expected_calls
        assert tuple(context.status_glyph_calls) == expected_status_glyph_calls
        plan = result.audit["source_extraction_strategy"]["discovery"]["plan"]
        assert tuple(plan["ordered_stages"]) == expected_calls
        if case_name == "header_only_lazy":
            assert set(plan["skipped_stages"]) == set(
                CANDIDATE_B_STAGE_REGISTRY.names
            ).difference(mandatory_stages)
        else:
            assert plan["skipped_stages"] == []

        detached = _detached_public_output(result)
        if lazy_output is None:
            lazy_output = detached
        else:
            assert detached == lazy_output, case_name
