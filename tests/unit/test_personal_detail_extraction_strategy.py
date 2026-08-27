# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from docmirror.plugins.credit_report.personal_detail_scanned.extraction_strategy import (
    EagerExtractionStrategy,
    ExtractionPlanError,
    ExtractionRequest,
    ExtractionStage,
    LazyExtractionStrategy,
    MaterializationMode,
    SectionCensus,
    SectionState,
    StageRegistry,
    StageSnapshot,
)


def _registry() -> StageRegistry:
    # The declaration order is intentionally not topological.  Stable ordering
    # must come from the dependency graph rather than incidental registration.
    return StageRegistry(
        (
            ExtractionStage(
                "overdue",
                dependencies=frozenset({"accounts", "repayments"}),
                output_names=("overdue_records",),
            ),
            ExtractionStage(
                "inquiry_summary",
                dependencies=frozenset({"inquiries"}),
                section="inquiries",
                optional=True,
            ),
            ExtractionStage("topology"),
            ExtractionStage(
                "employment",
                dependencies=frozenset({"topology"}),
                section="employment",
                optional=True,
            ),
            ExtractionStage(
                "accounts",
                dependencies=frozenset({"topology"}),
            ),
            ExtractionStage(
                "repayments",
                dependencies=frozenset({"accounts"}),
            ),
            ExtractionStage(
                "inquiries",
                dependencies=frozenset({"topology"}),
                section="inquiries",
                optional=True,
            ),
        )
    )


def _complete_census(
    *,
    inquiries: SectionState = SectionState.OBSERVED,
    employment: SectionState = SectionState.ABSENT_PROVEN,
) -> SectionCensus:
    return SectionCensus.from_mapping(
        {
            "employment": employment,
            "inquiries": inquiries,
        }
    )


def test_registry_resolves_deterministic_dependency_and_dependent_closures():
    registry = _registry()

    assert registry.ordered() == (
        "topology",
        "employment",
        "accounts",
        "repayments",
        "overdue",
        "inquiries",
        "inquiry_summary",
    )
    assert registry.dependency_closure({"overdue"}) == {
        "topology",
        "accounts",
        "repayments",
        "overdue",
    }
    assert registry.dependent_closure({"inquiries"}) == {
        "inquiries",
        "inquiry_summary",
    }


def test_registry_rejects_unknown_dependencies_and_cycles_before_execution():
    with pytest.raises(ExtractionPlanError, match="unknown dependencies: missing"):
        StageRegistry(
            (
                ExtractionStage("one", dependencies=frozenset({"missing"})),
            )
        )

    with pytest.raises(ExtractionPlanError, match="dependency cycle"):
        StageRegistry(
            (
                ExtractionStage("one", dependencies=frozenset({"two"})),
                ExtractionStage("two", dependencies=frozenset({"one"})),
            )
        )


def test_eager_strategy_materializes_every_registered_stage():
    registry = _registry()

    plan = EagerExtractionStrategy().plan(
        registry,
        ExtractionRequest.initial(_complete_census()),
    )

    assert plan.mode is MaterializationMode.EAGER
    assert plan.ordered_stage_names == registry.ordered()
    assert plan.reused_stage_names == ()
    assert plan.skipped_stage_names == ()
    assert not plan.fallback_used


def test_lazy_initial_plan_skips_only_optional_absent_or_empty_sections():
    registry = _registry()

    plan = LazyExtractionStrategy().plan(
        registry,
        ExtractionRequest.initial(_complete_census()),
    )

    assert plan.mode is MaterializationMode.LAZY
    assert plan.ordered_stage_names == (
        "topology",
        "accounts",
        "repayments",
        "overdue",
        "inquiries",
        "inquiry_summary",
    )
    assert plan.skipped_stage_names == ("employment",)

    empty_plan = LazyExtractionStrategy().plan(
        registry,
        ExtractionRequest.initial(
            _complete_census(inquiries=SectionState.EXPLICITLY_EMPTY)
        ),
    )
    assert empty_plan.skipped_stage_names == (
        "employment",
        "inquiries",
        "inquiry_summary",
    )


def test_explicit_request_materializes_absent_optional_section_and_dependencies():
    registry = _registry()

    plan = LazyExtractionStrategy().plan(
        registry,
        ExtractionRequest.initial(
            _complete_census(
                inquiries=SectionState.ABSENT_PROVEN,
                employment=SectionState.ABSENT_PROVEN,
            ),
            requested_stage_names={"employment"},
        ),
    )

    assert plan.requested_stage_names == ("employment",)
    assert "employment" in plan.ordered_stage_names
    assert "topology" in plan.ordered_stage_names
    assert "inquiries" in plan.skipped_stage_names


@pytest.mark.parametrize(
    ("census", "reason"),
    (
        (
            SectionCensus.from_mapping(
                {
                    "employment": SectionState.ABSENT_PROVEN,
                    "inquiries": SectionState.UNRESOLVED,
                }
            ),
            "section_census_unresolved:inquiries",
        ),
        (
            SectionCensus.from_mapping(
                {
                    "employment": SectionState.ABSENT_PROVEN,
                    "inquiries": SectionState.OBSERVED,
                },
                complete=False,
                incomplete_reason="page_registration_incomplete",
            ),
            "page_registration_incomplete",
        ),
    ),
)
def test_lazy_strategy_conservatively_falls_back_for_unresolved_census(
    census: SectionCensus,
    reason: str,
):
    registry = _registry()

    plan = LazyExtractionStrategy().plan(
        registry,
        ExtractionRequest.initial(census),
    )

    assert plan.mode is MaterializationMode.EAGER_FALLBACK
    assert plan.ordered_stage_names == registry.ordered()
    assert plan.skipped_stage_names == ()
    assert plan.fallback_reason == reason


def test_lazy_strategy_falls_back_when_optional_section_is_not_in_census():
    registry = _registry()
    census = SectionCensus.from_mapping(
        {"inquiries": SectionState.OBSERVED}
    )

    plan = LazyExtractionStrategy().plan(
        registry,
        ExtractionRequest.initial(census),
    )

    assert plan.mode is MaterializationMode.EAGER_FALLBACK
    assert plan.fallback_reason == "section_census_missing_sections:employment"


def test_lazy_repair_recomputes_dirty_dependents_and_reuses_clean_snapshots():
    registry = _registry()
    initial_plan = LazyExtractionStrategy().plan(
        registry,
        ExtractionRequest.initial(_complete_census()),
    )

    plan = LazyExtractionStrategy().plan(
        registry,
        ExtractionRequest.repair(
            _complete_census(),
            available_stage_names=initial_plan.ordered_stage_names,
            dirty_section_names={"inquiries"},
        ),
    )

    assert plan.mode is MaterializationMode.LAZY
    assert plan.dirty_stage_names == ("inquiries", "inquiry_summary")
    assert plan.ordered_stage_names == ("inquiries", "inquiry_summary")
    assert plan.reused_stage_names == (
        "topology",
        "accounts",
        "repayments",
        "overdue",
    )
    assert plan.skipped_stage_names == ("employment",)


def test_lazy_repair_uses_transitive_stage_closure_and_materializes_missing_inputs():
    registry = _registry()

    plan = LazyExtractionStrategy().plan(
        registry,
        ExtractionRequest.repair(
            _complete_census(),
            available_stage_names={"topology", "accounts", "inquiries", "inquiry_summary"},
            dirty_stage_names={"accounts"},
        ),
    )

    assert plan.dirty_stage_names == ("accounts",)
    assert plan.ordered_stage_names == ("accounts", "repayments", "overdue")
    assert plan.reused_stage_names == (
        "topology",
        "inquiries",
        "inquiry_summary",
    )


def test_lazy_repair_preserves_available_sectionless_optional_dependents():
    registry = StageRegistry(
        (
            ExtractionStage("source"),
            ExtractionStage(
                "optional_projection",
                dependencies=frozenset({"source"}),
                optional=True,
            ),
        )
    )

    plan = LazyExtractionStrategy().plan(
        registry,
        ExtractionRequest.repair(
            SectionCensus.from_mapping({}),
            available_stage_names=registry.names,
            dirty_stage_names={"source"},
        ),
    )

    assert plan.ordered_stage_names == ("source", "optional_projection")
    assert plan.reused_stage_names == ()
    assert plan.skipped_stage_names == ()


def test_lazy_repair_falls_back_when_dependency_closure_is_not_proven():
    registry = _registry()

    plan = LazyExtractionStrategy().plan(
        registry,
        ExtractionRequest.repair(
            _complete_census(),
            available_stage_names=registry.names,
            dirty_stage_names={"accounts"},
            dependency_closure_known=False,
        ),
    )

    assert plan.mode is MaterializationMode.EAGER_FALLBACK
    assert plan.ordered_stage_names == registry.ordered()
    assert plan.reused_stage_names == ()
    assert plan.fallback_reason == "dependency_closure_unknown"


def test_plan_audit_and_snapshot_are_immutable_and_deterministic():
    registry = _registry()
    plan = LazyExtractionStrategy().plan(
        registry,
        ExtractionRequest.initial(_complete_census()),
    )
    snapshot = StageSnapshot(
        stage_name="accounts",
        dependency_generations=(("topology", 1),),
        output_names=("credit_accounts",),
        record_counts=(("credit_accounts", 4),),
    )

    assert plan.audit().to_dict() == {
        "strategy": "lazy",
        "mode": "lazy",
        "purpose": "initial",
        "requested_stages": [],
        "ordered_stages": [
            "topology",
            "accounts",
            "repayments",
            "overdue",
            "inquiries",
            "inquiry_summary",
        ],
        "reused_stages": [],
        "skipped_stages": ["employment"],
        "dirty_stages": [],
        "dirty_sections": [],
        "fallback_reason": None,
    }
    assert snapshot.to_audit_dict() == {
        "stage": "accounts",
        "generation": 1,
        "dependency_generations": {"topology": 1},
        "outputs": ["credit_accounts"],
        "record_counts": {"credit_accounts": 4},
    }
    with pytest.raises(FrozenInstanceError):
        snapshot.generation = 2
