# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

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
    ACCOUNT_SECTION,
    LIABILITY_SECTION,
    REPORT_HEADER_SECTION,
    SECTION_TO_CANONICAL_DATASETS,
)


def _canonical_audit() -> dict[str, object]:
    sections = (REPORT_HEADER_SECTION, ACCOUNT_SECTION, LIABILITY_SECTION)
    registrations = [
        {
            "logical_page": page,
            "source_page": page,
            "status": "registered",
            "template_id": section,
            "basis": "printed_heading_and_table_signature",
            "affected_source_datasets": sorted(
                SECTION_TO_CANONICAL_DATASETS[section]
            ),
            "printed_page": page,
            "printed_total": len(sections),
        }
        for page, section in enumerate(sections, start=1)
    ]
    return {
        "corrected_evidence_conservation": {
            "valid": True,
            "conserved_logical_pages": [1, 2, 3],
        },
        "canonical_subset_conservation": {"valid": True},
        "unresolved_pages": [],
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


def test_lazy_repair_rehydrates_shared_diagnostics_before_dirty_stage(
    monkeypatch,
) -> None:
    """A dirty stage observes every clean predecessor's replayed ownership."""

    shared_issue = extraction_issues.make_issue(
        category="schema_incompleteness",
        issue_code="candidate_b_test_shared_diagnostic",
        message="Synthetic shared diagnostic used to verify stage replay.",
        target_dataset="credit_accounts",
        target_record_id="credit_account:test:shared",
        field_name="account_identifier",
        observed_value="unreadable",
    )
    issue_id = str(shared_issue["extraction_issue_id"])
    remap_edge = (
        "credit_account_observation:test:shared",
        "credit_account:test:shared",
    )
    status_issue = extraction_issues.make_issue(
        category="ocr_cell_level_error",
        issue_code="candidate_b_test_status_observation_diagnostic",
        message="Synthetic late status-loader diagnostic.",
        target_dataset="repayment_records",
        target_record_id="repayment:test:status",
        field_name="status",
        observed_value="?",
    )
    status_issue_id = str(status_issue["extraction_issue_id"])
    status_remap_edge = (
        "repayment_observation:test:status",
        "repayment:test:status",
    )
    status_observations = [{"observation_id": "status:test"}]
    glyph_observations: list[dict[str, object]] = []
    liability_calls = 0
    account_factory_calls = 0
    account_anchor_sentinel = [{"anchor_id": "account:test:anchor"}]
    discovery_liability = {
        "repayment_responsibility_id": "liability:test:field-local-repair",
        "institution": "深圳前海微众银行股份有限公司",
        "business_type": "爱贷款",
        "contract_number": "D10055840H0001DB20220228XS000000109",
    }
    repaired_liability = {
        **discovery_liability,
        "business_type": "贷款",
    }

    context = SimpleNamespace(
        _cache={},
        pages=[],
        reading_order_by_logical={},
        corrected_repayment_micro_grids=lambda: [],
        correct_candidate_b_datasets=lambda payload: payload,
        ocr_correction_audit=lambda: {
            "business_repair": {"second_schema_pass_required": True}
        },
        canonical_layout_audit=_canonical_audit,
        page_topology_audit=lambda: {},
    )

    def emit_shared_diagnostic() -> None:
        extraction_issues.record_issue(context, shared_issue)
        extraction_issues.register_issue_target_remap(context, *remap_edge)

    def account_collections():
        nonlocal account_factory_calls
        if "account_collections" in context._cache:
            return deepcopy(context._cache["account_collections"])
        account_factory_calls += 1
        emit_shared_diagnostic()
        context._candidate_b_account_anchor_skeleton_cache = deepcopy(
            account_anchor_sentinel
        )
        collections = ([], [], [])
        context._cache["account_collections"] = deepcopy(collections)
        return deepcopy(collections)

    def repayments():
        # Duplicate suppression keeps the discovery row/edge singular, while
        # the sidecars must still remember this independent second owner.
        emit_shared_diagnostic()
        return []

    def prepare_repair(_payload):
        context._business_repair_plan = SimpleNamespace(
            affected_pages=(3,),
            uncertainties=(
                SimpleNamespace(
                    logical_pages=(3,),
                    dataset_name="repayment_liability_records",
                    field_name="business_type",
                ),
            ),
        )
        context._cache.clear()
        context.__dict__.pop(
            "_candidate_b_account_anchor_skeleton_cache",
            None,
        )
        return True

    def load_status_observations():
        extraction_issues.record_issue(context, status_issue)
        extraction_issues.register_issue_target_remap(
            context,
            *status_remap_edge,
        )
        return status_observations

    def liabilities(_context):
        nonlocal liability_calls
        liability_calls += 1
        if liability_calls == 2:
            assert context._cache["account_collections"] == ([], [], [])
            assert context._candidate_b_account_anchor_skeleton_cache == (
                account_anchor_sentinel
            )
            assert (
                context._candidate_b_account_anchor_skeleton_cache
                is not account_anchor_sentinel
            )
            assert context.account_collections() == ([], [], [])
            assert context._candidate_b_issue_stage_owners[issue_id] == {
                "account_inventory",
                "monthly_repayments",
            }
            assert context._candidate_b_remap_stage_owners[remap_edge] == {
                "account_inventory",
                "monthly_repayments",
            }
            assert [
                issue
                for issue in context._personal_detail_extraction_issues
                if issue.get("extraction_issue_id") == issue_id
            ] == [shared_issue]
            assert context._personal_detail_issue_target_remaps[
                remap_edge[0]
            ] == {remap_edge[1]}
            assert context._candidate_b_issue_stage_owners[status_issue_id] == {
                "monthly_repayments"
            }
            assert context._candidate_b_remap_stage_owners[
                status_remap_edge
            ] == {"monthly_repayments"}
            assert [
                issue
                for issue in context._personal_detail_extraction_issues
                if issue.get("extraction_issue_id") == status_issue_id
            ] == [status_issue]
            assert context._personal_detail_issue_target_remaps[
                status_remap_edge[0]
            ] == {status_remap_edge[1]}
        return deepcopy(
            [
                discovery_liability
                if liability_calls == 1
                else repaired_liability
            ]
        )

    context.account_collections = account_collections
    context.corrected_repayment_records = repayments
    context.prepare_candidate_b_business_repair = prepare_repair
    context.candidate_b_status_glyph_observations = load_status_observations

    def empty(_context):
        return []

    for name in (
        "_extract_credit_lines",
        "_extract_employment_records",
        "_extract_inquiries",
        "_extract_postpaid_payment_history",
        "_extract_postpaid_records",
        "_extract_public_records",
        "_extract_recovery_records",
        "_extract_residence_records",
        "_extract_source_rows",
    ):
        monkeypatch.setattr(native_extraction, name, empty)
    monkeypatch.setattr(native_extraction, "_extract_liabilities", liabilities)
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

    def reconcile_credit_lines(repair_context, rows):
        repair_context.account_collections()
        return rows

    monkeypatch.setattr(
        native_extraction,
        "reconcile_candidate_b_credit_lines",
        reconcile_credit_lines,
    )
    monkeypatch.setattr(
        native_extraction,
        "reconcile_candidate_b_account_sequence_issues",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        profile_extraction,
        "extract_candidate_b_profile",
        lambda _context: {},
    )
    monkeypatch.setattr(
        relations,
        "candidate_b_repayment_anchor_ledger",
        lambda *_args: {},
    )
    monkeypatch.setattr(
        relations,
        "link_candidate_b_repayments",
        lambda rows, *_args, **_kwargs: rows,
    )
    monkeypatch.setattr(
        relations,
        "derive_candidate_b_overdue_records",
        lambda *_args: [],
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
        consistency_ledger,
        "apply_document_consistency_ledger",
        lambda *_args: {},
    )
    def glyph_bank(_records, observations, **_kwargs):
        glyph_observations.extend(observations)
        return {}

    monkeypatch.setattr(
        document_glyph_bank,
        "apply_document_local_status_glyph_bank",
        glyph_bank,
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
        lambda issue_context: list(
            getattr(issue_context, "_personal_detail_extraction_issues", ())
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
        lambda payload, _business, **_kwargs: payload,
    )

    result = CandidateBPipeline(context, "").run()

    repair_plan = result.audit["source_extraction_strategy"]["repair"]["plan"]
    assert repair_plan["mode"] == "lazy"
    assert repair_plan["ordered_stages"] == ["liabilities", "source_rows"]
    assert {
        "account_inventory",
        "monthly_repayments",
    }.issubset(repair_plan["reused_stages"])
    assert liability_calls == 2
    assert account_factory_calls == 1
    assert glyph_observations == status_observations
    assert result.business["repayment_liability_records"] == [
        repaired_liability
    ]
    assert result.business["repayment_liability_records"][0][
        "institution"
    ] == discovery_liability["institution"]
    assert result.business["repayment_liability_records"][0][
        "contract_number"
    ] == discovery_liability["contract_number"]
    assert result.business["repayment_liability_records"][0][
        "business_type"
    ] != discovery_liability["business_type"]
    strategy_audit = result.audit["source_extraction_strategy"]
    assert strategy_audit["canonical_census_mode"] == (
        "always_recomputed_before_stage_planning"
    )
    assert strategy_audit["repair"]["repaired_census"]["complete"] is True
    assert strategy_audit["repair"]["repaired_census"][
        "fallback_reason"
    ] is None
    assert strategy_audit["repair"]["repaired_census"] == (
        strategy_audit["discovery"]["census"]
    )
    assert strategy_audit["shared_release_gates"] == {
        "mode": "always_eager_once_after_source_materialization",
        "stages": [
            "final_dataset_correction",
            "employment_contract_enforcement",
            "credit_line_reconciliation",
            "document_consistency",
            "final_account_field_issue_reconciliation",
            "document_local_status_glyph_bank",
            "native_source_cell_status_guard",
            "final_liability_issue_registration",
            "source_completeness",
            "account_sequence_issue_reconciliation",
            "extraction_issue_collection",
            "source_projection",
        ],
    }
