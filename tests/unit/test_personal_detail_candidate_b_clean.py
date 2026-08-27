from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from docmirror.plugins.credit_report.personal_detail_scanned import (
    candidate_b,
    native_extraction,
)
from docmirror.plugins.credit_report.personal_detail_scanned import (
    relations as relations_mod,
)
from docmirror.plugins.credit_report.personal_detail_scanned.candidate_b import CandidateBPipeline
from docmirror.plugins.credit_report.personal_detail_scanned.context import (
    PersonalDetailExtractionContext,
)
from docmirror.plugins.credit_report.personal_detail_scanned.profile_extraction import (
    extract_candidate_b_profile,
)
from docmirror.plugins.credit_report.personal_detail_scanned.relations import (
    candidate_b_repayment_anchor_ledger,
    link_candidate_b_repayments,
)
from docmirror.plugins.credit_report.personal_detail_scanned.schema import (
    project_personal_detail_datasets,
)
from docmirror.plugins.credit_report.personal_detail_scanned.variant import (
    PersonalDetailScannedVariant,
)
from docmirror.plugins.credit_report.value_utils import stable_record_id


def test_repaired_inquiries_close_ye_shaped_source_population_aggregate() -> None:
    institution = list(range(1, 97))
    personal = list(range(1, 17))
    ledger = {
        "inquiry_records": 36,
        "inquiry_sequence_endpoints": {
            "institution": 96,
            "personal": 16,
        },
        "inquiry_observed_sequences": {
            "institution": [*institution, 789],
            "personal": personal,
        },
        "inquiry_sequence_outliers": {"institution": [789]},
        # Mirrors the real source ledger: population evidence is complete,
        # while only a subset of ordinals owns a publishable exact row ref.
        "inquiry_ordinal_observations": {
            "institution": {
                str(sequence): {"sequence": sequence, "inquiry_type": "institution"}
                for sequence in institution
                if sequence not in {67, 93, 94, 95, 96}
            },
            "personal": {"16": {"sequence": 16, "inquiry_type": "personal"}},
        },
    }
    rows = []
    for inquiry_type, sequences in (
        ("institution", institution),
        ("personal", personal),
    ):
        for sequence in sequences:
            rows.append(
                {
                    "inquiry_id": stable_record_id("credit_inquiry", inquiry_type, sequence),
                    "inquiry_type": inquiry_type,
                    "sequence": sequence,
                    "source_refs": [
                        {
                            "source": "native_detail_table",
                            "logical_page": 28 if inquiry_type == "institution" else 29,
                            "source_page": 14 if inquiry_type == "institution" else 15,
                            "table_id": f"{inquiry_type}-inquiries",
                            "row": sequence,
                            "geometry_scope": "row",
                            "bbox": [0.0, float(sequence), 10.0, float(sequence + 1)],
                        }
                    ],
                }
            )

    reconciled = candidate_b._reconcile_repaired_inquiry_source_population(
        ledger,
        rows,
    )

    assert reconciled["inquiry_records"] == 112
    assert reconciled["inquiry_sequence_endpoints"] == {
        "institution": 96,
        "personal": 16,
    }


def test_repaired_inquiry_population_reconciliation_rejects_duplicate_identity() -> None:
    ledger = {
        "inquiry_records": 1,
        "inquiry_sequence_endpoints": {"institution": 2},
        "inquiry_observed_sequences": {"institution": [1, 2]},
    }
    rows = [
        {
            "inquiry_id": stable_record_id("credit_inquiry", "institution", sequence),
            "inquiry_type": "institution",
            "sequence": sequence,
            "source_refs": [
                {
                    "source": "native_detail_table",
                    "logical_page": 27,
                    "source_page": 14,
                    "table_id": "institution-inquiries",
                    "row": sequence,
                }
            ],
        }
        for sequence in (1, 2, 2)
    ]

    reconciled = candidate_b._reconcile_repaired_inquiry_source_population(
        ledger,
        rows,
    )

    assert reconciled["inquiry_records"] == 1


def test_account_repair_lifecycle_conserves_the_ye_shaped_42_account_population() -> None:
    family_counts = (
        ("non_revolving_loan", 18),
        ("revolving_loan_subaccount", 6),
        ("revolving_loan_account", 6),
        ("credit_card", 12),
    )
    discovery: list[dict[str, object]] = []
    sequence = 0
    for family, count in family_counts:
        for ordinal in range(1, count + 1):
            sequence += 1
            account_id = f"credit_account:{family}:{ordinal}"
            discovery.append(
                {
                    "account_id": account_id,
                    "account_type": family,
                    "category_sequence": ordinal,
                    "sequence": sequence,
                    "account_identifier": f"YEACCOUNT{sequence:04d}",
                    "management_institution": f"discovery-{sequence}",
                    "account_family_quality": "exact",
                    "_printed_ordinal_status": "printed_unique",
                    "_canonical_segment": {"ownership_basis": "printed_anchor_to_next_anchor"},
                    # A discovered anchor is normally enriched by an owned
                    # business table before lifecycle reconciliation.  The
                    # immutable anchor remains in ``source_refs``; the
                    # top-level producer correctly describes the richer row.
                    "source": "native_detail_table",
                    "page": sequence,
                    "source_page": sequence,
                    "bbox": [10.0, 20.0, 80.0, 30.0],
                    "source_refs": [
                        {
                            "source": "candidate_b_account_anchor",
                            "logical_page": sequence,
                            "source_page": sequence,
                            "bbox": [10.0, 20.0, 80.0, 30.0],
                            "evidence_ids": [f"ye-account-anchor:{sequence}"],
                        },
                        {
                            "source": "native_detail_table",
                            "logical_page": sequence,
                            "source_page": sequence,
                            "table_id": f"discovery-{sequence}",
                        },
                    ],
                }
            )

    repaired = deepcopy(discovery)
    missing_id = "credit_account:revolving_loan_subaccount:6"
    repaired = [row for row in repaired if row["account_id"] != missing_id]
    remapped_ids: dict[str, str] = {}
    for canonical_id in (
        "credit_account:non_revolving_loan:4",
        "credit_account:revolving_loan_subaccount:4",
        "credit_account:revolving_loan_subaccount:5",
    ):
        row = next(item for item in repaired if item["account_id"] == canonical_id)
        provisional_id = f"credit_account_provisional:{len(remapped_ids) + 1}"
        remapped_ids[provisional_id] = canonical_id
        row["account_id"] = provisional_id
        row["account_type"] = "non_revolving_loan"
        row.pop("category_sequence", None)
        row.pop("account_family_quality", None)
        row.pop("_printed_ordinal_status", None)
        row.pop("_canonical_segment", None)
        row["source"] = "native_detail_table"
        row["management_institution"] = f"repaired-{canonical_id}"
        row["source_refs"] = [
            {
                "source": "native_detail_table",
                "logical_page": int(row["sequence"]),
                "source_page": int(row["sequence"]),
                "table_id": f"repaired-{row['sequence']}",
            }
        ]

    context = SimpleNamespace()
    reconciled = candidate_b._reconcile_account_population_lifecycle(
        context,
        discovery,
        repaired,
    )

    assert len(reconciled) == 42
    assert {row["account_id"] for row in reconciled} == {
        f"credit_account:{family}:{ordinal}" for family, count in family_counts for ordinal in range(1, count + 1)
    }
    by_id = {row["account_id"]: row for row in reconciled}
    assert by_id[missing_id]["management_institution"] == "discovery-24"
    for provisional_id, canonical_id in remapped_ids.items():
        row = by_id[canonical_id]
        family, ordinal = canonical_id.rsplit(":", 2)[1:]
        assert row["account_type"] == family
        assert row["category_sequence"] == int(ordinal)
        assert row["management_institution"] == f"repaired-{canonical_id}"
        assert {ref["source"] for ref in row["source_refs"]} == {
            "candidate_b_account_anchor",
            "native_detail_table",
        }
        assert context._personal_detail_issue_target_remaps[provisional_id] == {canonical_id}


def test_account_repair_stage_conserves_real_shaped_ye_anchor_and_table_identities() -> None:
    """A dirty account stage must expose the sealed population to dependants."""

    identities = (
        ("non_revolving_loan", 4, 5, [43.0, 477.0, 62.5, 486.5]),
        ("revolving_loan_subaccount", 4, 13, [44.5, 250.5, 64.0, 260.0]),
        ("revolving_loan_subaccount", 5, 13, [44.0, 463.0, 67.0, 474.0]),
        ("revolving_loan_subaccount", 6, 14, [57.5, 148.0, 215.0, 158.0]),
    )
    inventory: list[dict[str, object]] = []
    discovery: list[dict[str, object]] = []
    for sequence, (family, ordinal, page, bbox) in enumerate(identities, 1):
        account_id = f"credit_account:{family}:{ordinal}"
        skeleton: dict[str, object] = {
            "account_id": account_id,
            "record_id": account_id,
            "account_type": family,
            "category_sequence": ordinal,
            "sequence": sequence,
            "account_family_quality": "exact",
            "_printed_ordinal_status": "printed_unique",
            "_canonical_segment": {
                "ownership_basis": "printed_anchor_to_next_anchor",
                "anchor_logical_page": page,
                "anchor_bbox": list(bbox),
                "pages": [{"logical_page": page, "min_y": bbox[1], "max_y": None}],
            },
            "page": page,
            "source_page": page,
            "bbox": list(bbox),
            "source_refs": [
                {
                    "source": "candidate_b_account_anchor",
                    "logical_page": page,
                    "source_page": page,
                    "binding": "printed_account_ordinal",
                    "account_type": family,
                    "category_sequence": ordinal,
                    "bbox": list(bbox),
                    "evidence_ids": [f"ye-anchor:{family}:{ordinal}"],
                }
            ],
        }
        inventory.append(skeleton)
        table_observation_id = f"credit_account_table_observation:ye-{family}-{ordinal}"
        table_instance_id = f"credit_account_table_observation_instance:ye-{family}-{ordinal}"
        row = deepcopy(skeleton)
        row.update(
            {
                "_table_observation_id": table_observation_id,
                "_table_observation_instance_id": table_instance_id,
            }
        )
        if ordinal != 6:
            row["account_identifier"] = f"YEACCOUNT{sequence:04d}"
            row["management_institution"] = f"discovery-{sequence}"
        # The discovery dataset owns table values while the separately sealed
        # pre-repair inventory owns the immutable family/ordinal identity.
        for private_field in (
            "account_family_quality",
            "_printed_ordinal_status",
            "_canonical_segment",
        ):
            row.pop(private_field, None)
        discovery.append(row)

    repaired_boxes = (
        [43.72, 477.40, 62.41, 486.07],
        [44.723, 251.210, 63.413, 259.885],
        [42.777, 458.631, 71.394, 476.040],
    )
    repaired: list[dict[str, object]] = []
    for index, (discovery_row, repaired_bbox) in enumerate(
        zip(discovery[:3], repaired_boxes, strict=True),
        1,
    ):
        row = deepcopy(discovery_row)
        row["account_id"] = f"credit_account_provisional:ye-{index}"
        row["record_id"] = row["account_id"]
        row["account_type"] = "non_revolving_loan"
        row.pop("category_sequence", None)
        row["management_institution"] = f"repaired-{index}"
        anchor_ref = deepcopy(row["source_refs"][0])
        anchor_ref["bbox"] = list(repaired_bbox)
        anchor_ref.pop("account_type", None)
        anchor_ref.pop("category_sequence", None)
        row["source_refs"] = [anchor_ref]
        repaired.append(row)

    context = SimpleNamespace(
        _cache={
            "account_collections": (
                deepcopy(repaired),
                [{"discarded": True}],
                [{"account_event_id": "event:1"}],
            )
        },
        _candidate_b_pre_repair_account_anchor_inventory=tuple(deepcopy(inventory)),
        _personal_detail_extraction_issues=[],
    )
    discovery_payload = candidate_b._StagePayload(business={"credit_accounts": deepcopy(discovery)})
    repaired_payload = candidate_b._StagePayload(
        business={"credit_accounts": deepcopy(repaired)},
        context_cache_entries=deepcopy(context._cache),
    )

    candidate_b._reconcile_repaired_account_stage_payload(
        context,
        discovery_payload,
        repaired_payload,
    )

    reconciled = repaired_payload.business["credit_accounts"]
    expected_ids = {f"credit_account:{family}:{ordinal}" for family, ordinal, _page, _bbox in identities}
    assert len(reconciled) == 4
    assert {row["account_id"] for row in reconciled} == expected_ids
    by_id = {row["account_id"]: row for row in reconciled}
    assert by_id["credit_account:non_revolving_loan:4"]["management_institution"] == "repaired-1"
    assert by_id["credit_account:revolving_loan_subaccount:4"]["management_institution"] == "repaired-2"
    assert by_id["credit_account:revolving_loan_subaccount:5"]["management_institution"] == "repaired-3"
    assert by_id["credit_account:revolving_loan_subaccount:6"].get("account_identifier") is None
    cached_accounts, cached_discarded, cached_events = context._cache["account_collections"]
    assert cached_accounts == reconciled
    assert cached_discarded == [{"discarded": True}]
    assert cached_events == [{"account_event_id": "event:1"}]
    assert repaired_payload.context_cache_entries["account_collections"][0] == (reconciled)

    remaps = context._personal_detail_issue_target_remaps
    for repaired_row, (family, ordinal, _page, _bbox) in zip(
        repaired,
        identities[:3],
        strict=True,
    ):
        canonical_id = f"credit_account:{family}:{ordinal}"
        for alias_field in (
            "account_id",
            "record_id",
            "_table_observation_id",
            "_table_observation_instance_id",
        ):
            assert remaps[str(repaired_row[alias_field])] == {canonical_id}
    r1_6 = discovery[-1]
    for alias_field in ("_table_observation_id", "_table_observation_instance_id"):
        assert remaps[str(r1_6[alias_field])] == {"credit_account:revolving_loan_subaccount:6"}


def test_account_repair_stage_recovers_exact_source_census_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The independent source census survives a lossy repair-stage cache."""

    account_id = "credit_account:revolving_loan_subaccount:6"
    anchor_ref = {
        "source": "candidate_b_account_anchor",
        "logical_page": 14,
        "source_page": 7,
        "geometry_scope": "line",
        "binding": "printed_account_ordinal",
        "binding_quality": "printed_account_ordinal",
        "account_type": "revolving_loan_subaccount",
        "category_sequence": 6,
        "bbox": [57.5, 148.0, 215.0, 158.0],
        "evidence_ids": ["ocr:sp0007:lp0014:0041"],
    }
    table_observation_id = "credit_account_table_observation:ye-r1-6"
    table_instance_id = "credit_account_table_observation_instance:ye-r1-6"
    discovery = {
        "account_id": account_id,
        "record_id": account_id,
        "account_type": "revolving_loan_subaccount",
        "category_sequence": 6,
        "sequence": 24,
        "_table_observation_id": table_observation_id,
        "_table_observation_instance_id": table_instance_id,
        "source_refs": [deepcopy(anchor_ref)],
    }
    ledger = {
        "account_family_ordinal_observations": {
            "revolving_loan_subaccount": {
                "6": {
                    "account_id": account_id,
                    "account_type": "revolving_loan_subaccount",
                    "category_sequence": 6,
                    "source_refs": [deepcopy(anchor_ref)],
                }
            }
        }
    }
    monkeypatch.setattr(
        native_extraction,
        "_source_completeness_ledger",
        lambda _context: deepcopy(ledger),
    )
    context = SimpleNamespace(
        _cache={"account_collections": ([], [], [])},
        _personal_detail_extraction_issues=[],
    )
    discovery_payload = candidate_b._StagePayload(business={"credit_accounts": [deepcopy(discovery)]})
    repaired_payload = candidate_b._StagePayload(
        business={"credit_accounts": []},
        context_cache_entries=deepcopy(context._cache),
    )

    candidate_b._reconcile_repaired_account_stage_payload(
        context,
        discovery_payload,
        repaired_payload,
    )

    assert repaired_payload.business["credit_accounts"] == [discovery]
    assert "_canonical_segment" not in repaired_payload.business["credit_accounts"][0]
    assert context._cache["account_collections"][0] == [discovery]
    assert context._personal_detail_issue_target_remaps[table_observation_id] == {account_id}
    assert context._personal_detail_issue_target_remaps[table_instance_id] == {account_id}


def test_empty_repair_stage_preserves_source_backed_discovery_output() -> None:
    from docmirror.plugins.credit_report.personal_detail_scanned.candidate_b_strategy import (
        CANDIDATE_B_STAGE_REGISTRY,
    )

    discovery_rows = [
        {
            "record_id": "personal_detail_summary_record:account_count",
            "metric_code": "account_count",
            "numeric_value": 45,
            "source_refs": [
                {
                    "source": "native_detail_table_cell",
                    "logical_page": 3,
                    "table_id": "pt_3_0",
                    "row": 1,
                    "column": 2,
                    "geometry_scope": "cell",
                    "bbox": [120.0, 80.0, 160.0, 94.0],
                    "evidence_ids": ["ocr:sp0002:lp0003:0012"],
                }
            ],
        }
    ]
    discovery = candidate_b._StagePayload(
        datasets={
            "personal_detail_summary_records": deepcopy(discovery_rows),
            "personal_detail_summary_cells": [{"summary_cell_id": "cell:1"}],
        }
    )
    repaired = candidate_b._StagePayload(
        datasets={
            "personal_detail_summary_records": [],
            "personal_detail_summary_cells": [],
        }
    )
    preserved = candidate_b._preserve_discovery_stage_outputs_on_empty_repair(
        CANDIDATE_B_STAGE_REGISTRY.stage("summary"),
        discovery,
        repaired,
    )

    assert preserved is True
    assert repaired.datasets == discovery.datasets


def test_nonempty_repair_stage_is_not_replaced_by_discovery() -> None:
    from docmirror.plugins.credit_report.personal_detail_scanned.candidate_b_strategy import (
        CANDIDATE_B_STAGE_REGISTRY,
    )

    discovery = candidate_b._StagePayload(datasets={"residence_records": [{"sequence": 1}, {"sequence": 2}]})
    repaired = candidate_b._StagePayload(datasets={"residence_records": [{"sequence": 2}]})

    preserved = candidate_b._preserve_discovery_stage_outputs_on_empty_repair(
        CANDIDATE_B_STAGE_REGISTRY.stage("residence"),
        discovery,
        repaired,
    )

    assert preserved is False
    assert repaired.datasets["residence_records"] == [{"sequence": 2}]


def test_final_account_issues_localize_unique_table_alias_and_suppress_generic_duplicate() -> None:
    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
        collect_extraction_issues,
        make_issue,
    )

    account_id = "credit_account:revolving_loan_subaccount:6"
    table_observation_id = "credit_account_table_observation:ye-r1-6"
    merged = "中信银行股份有限 B10611000H0001 2022.06.22 公司福州分行 811132137961001"
    table_ref = {
        "source": "native_detail_table",
        "logical_page": 14,
        "source_page": 7,
        "table_id": "pt_14_1",
        "coordinate_system": "pdf_points_top_left",
        "geometry_scope": "table",
        "bbox": [54.0, 157.5, 403.5, 335.0],
    }
    cell_ref = {
        **table_ref,
        "source": "native_detail_table_cell",
        "geometry_scope": "cell",
        "row": 1,
        "column": 0,
        "bbox": [54.0, 171.5, 228.5, 190.5],
        "evidence_ids": ["r1:v:institution", "r1:v:identifier", "r1:v:date"],
        "binding": "closed_canonical_account_cell_cluster",
        "binding_quality": "closed_canonical_account_cell_cluster",
    }
    issues = [
        make_issue(
            category="ocr_structure_correction",
            issue_code="candidate_b_account_cluster_field_unresolved",
            message="The merged exact cell did not uniquely type this field.",
            parser_stage="candidate_b_account_closed_cell_cluster",
            target_dataset="credit_accounts",
            target_record_id=table_observation_id,
            field_name=field_name,
            observed_value=[merged],
            source_refs=(cell_ref,),
            reason_codes=(
                "closed_canonical_header_label_set",
                "field_not_uniquely_typed",
                "normalized_value_withheld",
                "record_not_emitted_due_to_unresolved_account_ownership",
            ),
        )
        for field_name in ("management_institution", "account_identifier")
    ]
    issues.append(
        make_issue(
            category="ocr_cell_level_error",
            issue_code="pboc_cell_contract_unresolved",
            message="The generic final contract also rejected the identifier.",
            parser_stage="candidate_b_final_validation",
            target_dataset="credit_accounts",
            target_record_id=account_id,
            field_name="account_identifier",
            observed_value=merged,
            source_refs=(cell_ref,),
            reason_codes=("normalized_value_withheld",),
        )
    )
    context = SimpleNamespace(
        _personal_detail_extraction_issues=issues,
        _personal_detail_issue_target_remaps={},
    )
    records = [
        {
            "account_id": account_id,
            "account_type": "revolving_loan_subaccount",
            "category_sequence": 6,
            "management_institution": None,
            "account_identifier": None,
            "source_refs": [table_ref],
            "_unresolved_fields": [
                "management_institution",
                "account_identifier",
            ],
        }
    ]

    candidate_b._reconcile_final_account_field_issues(context, records)
    published = collect_extraction_issues(context)

    assert context._personal_detail_issue_target_remaps[table_observation_id] == {account_id}
    for field_name in ("management_institution", "account_identifier"):
        active = [
            issue
            for issue in published
            if issue.get("target_record_id") == account_id
            and issue.get("field_name") == field_name
            and issue.get("status") == "requires_review"
        ]
        assert len(active) == 1
        assert active[0]["issue_code"] == ("candidate_b_account_cluster_field_unresolved")
        assert active[0]["parser_stage"] == ("candidate_b_account_closed_cell_cluster")
        assert active[0]["category"] == "ocr_structure_correction"
        assert active[0]["reason_codes"] == [
            "closed_canonical_header_label_set",
            "field_not_uniquely_typed",
            "normalized_value_withheld",
        ]


def test_account_repair_lifecycle_does_not_guess_ambiguous_anchor_geometry() -> None:
    def sealed(account_id: str) -> dict[str, object]:
        family, ordinal = account_id.rsplit(":", 2)[1:]
        return {
            "account_id": account_id,
            "account_type": family,
            "category_sequence": int(ordinal),
            "account_family_quality": "exact",
            "_printed_ordinal_status": "printed_unique",
            "_canonical_segment": {"ownership_basis": "printed_anchor_to_next_anchor"},
            "source_refs": [
                {
                    "source": "candidate_b_account_anchor",
                    "logical_page": 13,
                    "source_page": 7,
                    "bbox": [44.0, 250.0, 70.0, 262.0],
                    "evidence_ids": [f"anchor:{account_id}"],
                }
            ],
        }

    discovery = [
        sealed("credit_account:revolving_loan_subaccount:4"),
        sealed("credit_account:revolving_loan_subaccount:5"),
    ]
    repaired = [
        {
            "account_id": "credit_account_provisional:ambiguous",
            "source_refs": [
                {
                    "source": "candidate_b_account_anchor",
                    "logical_page": 13,
                    "source_page": 7,
                    "bbox": [44.5, 250.5, 69.5, 261.5],
                    "evidence_ids": ["anchor:repaired"],
                }
            ],
        }
    ]

    context = SimpleNamespace()
    reconciled = candidate_b._reconcile_account_population_lifecycle(
        context,
        discovery,
        repaired,
    )

    assert {row["account_id"] for row in reconciled} == {
        "credit_account_provisional:ambiguous",
        "credit_account:revolving_loan_subaccount:4",
        "credit_account:revolving_loan_subaccount:5",
    }
    assert not hasattr(context, "_personal_detail_issue_target_remaps")


def _account_issue_ref(*, row: int, column: int, field_name: str) -> dict[str, object]:
    return {
        "source": "native_detail_table",
        "logical_page": 4,
        "source_page": 2,
        "table_id": "bad-table",
        "row": row,
        "column": column,
        "field_name": field_name,
        "binding_quality": "closed_canonical_account_cell_cluster",
    }


def _exact_account_field_ref(*, bbox_top: float, field_name: str) -> dict[str, object]:
    return {
        "source": "candidate_b_account_anchor_interval",
        "logical_page": 4,
        "source_page": 2,
        "bbox": [10.0, bbox_top, 500.0, bbox_top + 20.0],
        "field_name": field_name,
        "binding_quality": "canonical_account_header_geometry",
    }


def _active_account_issue(
    *, record_id: str, field_name: str, issue_code: str, source_ref: dict[str, object]
) -> dict[str, object]:
    return {
        "target_dataset": "credit_accounts",
        "target_record_id": record_id,
        "field_name": field_name,
        "issue_code": issue_code,
        "status": "requires_review",
        "source_refs": [source_ref],
    }


def test_final_account_issue_reconciliation_closes_only_independently_exact_bad_alternates() -> None:
    cases = (
        (
            "currency",
            "account_currency",
            "CNY",
            "candidate_b_account_cluster_residue_unresolved",
        ),
        (
            "business",
            "business_type",
            "其他个人消费贷款",
            "candidate_b_account_cluster_field_unresolved",
        ),
        (
            "guarantee",
            "guarantee_type",
            "信用/免担保",
            "candidate_b_exact_slot_value_invalid",
        ),
        (
            "identifier",
            "account_identifier",
            "B11614560H0001310333515600026815",
            "candidate_b_account_cluster_field_unresolved",
        ),
        (
            "institution-prefix",
            "management_institution",
            "重庆市蚂蚁商诚小额贷款有限公司",
            "candidate_b_institution_leading_boundary_ambiguous",
        ),
        (
            "institution-branch",
            "management_institution",
            "中国建设银行股份有限公司福建自贸试验区福州片区分行",
            "candidate_b_institution_branch_without_legal_root",
        ),
    )
    records = []
    issues = []
    for index, (suffix, field_name, value, issue_code) in enumerate(cases, start=1):
        record_id = f"credit_account:test:{suffix}"
        ref_field = "currency" if field_name == "account_currency" else field_name
        record = {
            "account_id": record_id,
            field_name: value,
            "source_refs_by_field": {
                ref_field: [
                    _exact_account_field_ref(
                        bbox_top=100.0 + index * 30.0,
                        field_name=ref_field,
                    )
                ]
            },
            "_unresolved_fields": [field_name],
            "_invalid_observation_fields": [field_name],
        }
        records.append(record)
        issues.append(
            _active_account_issue(
                record_id=record_id,
                field_name=field_name,
                issue_code=issue_code,
                source_ref=_account_issue_ref(
                    row=index,
                    column=0,
                    field_name=field_name,
                ),
            )
        )
    context = SimpleNamespace(_personal_detail_extraction_issues=issues)

    candidate_b._reconcile_final_account_field_issues(context, records)

    assert context._personal_detail_extraction_issues == []
    assert all("_unresolved_fields" not in record for record in records)
    assert all("_invalid_observation_fields" not in record for record in records)


def test_final_account_issue_reconciliation_preserves_same_source_invalid_and_conflict_cases() -> None:
    same_ref = _account_issue_ref(row=1, column=0, field_name="business_type")
    same_source = {
        "account_id": "credit_account:test:same-source",
        "business_type": "其他个人消费贷款",
        "source_refs_by_field": {"business_type": [same_ref]},
        "_unresolved_fields": ["business_type"],
    }
    invalid = {
        "account_id": "credit_account:test:invalid",
        "business_type": "货记卡",
        "source_refs_by_field": {
            "business_type": [_exact_account_field_ref(bbox_top=200.0, field_name="business_type")]
        },
        "_unresolved_fields": ["business_type"],
    }
    conflict = {
        "account_id": "credit_account:test:conflict",
        "account_currency": "CNY",
        "source_refs_by_field": {"currency": [_exact_account_field_ref(bbox_top=240.0, field_name="currency")]},
        "_reported_field_conflicts": ["account_currency"],
        "_unresolved_fields": ["account_currency"],
    }
    absent = {
        "account_id": "credit_account:test:absent",
        "guarantee_type": "信用/免担保",
        "source_refs_by_field": {
            "guarantee_type": [_exact_account_field_ref(bbox_top=280.0, field_name="guarantee_type")]
        },
        "_source_absent_fields": ["guarantee_type"],
        "_unresolved_fields": ["guarantee_type"],
    }
    records = [same_source, invalid, conflict, absent]
    issues = [
        _active_account_issue(
            record_id="credit_account:test:same-source",
            field_name="business_type",
            issue_code="candidate_b_account_cluster_field_unresolved",
            source_ref=same_ref,
        ),
        _active_account_issue(
            record_id="credit_account:test:invalid",
            field_name="business_type",
            issue_code="candidate_b_account_cluster_field_unresolved",
            source_ref=_account_issue_ref(row=2, column=0, field_name="business_type"),
        ),
        _active_account_issue(
            record_id="credit_account:test:conflict",
            field_name="account_currency",
            issue_code="candidate_b_account_cluster_residue_unresolved",
            source_ref=_account_issue_ref(row=3, column=0, field_name="account_currency"),
        ),
        _active_account_issue(
            record_id="credit_account:test:conflict",
            field_name="account_currency",
            issue_code="candidate_b_exact_slot_value_conflict",
            source_ref=_account_issue_ref(row=4, column=0, field_name="account_currency"),
        ),
        _active_account_issue(
            record_id="credit_account:test:absent",
            field_name="guarantee_type",
            issue_code="candidate_b_exact_slot_value_invalid",
            source_ref=_account_issue_ref(row=5, column=0, field_name="guarantee_type"),
        ),
    ]
    context = SimpleNamespace(_personal_detail_extraction_issues=issues)

    candidate_b._reconcile_final_account_field_issues(context, records)

    assert context._personal_detail_extraction_issues == issues
    assert all(record.get("_unresolved_fields") for record in records)


def test_new_currency_issues_require_evidence_bearing_corrected_cell_ref() -> None:
    record_id = "credit_account:test:currency"
    geometry_only_ref = _exact_account_field_ref(
        bbox_top=300.0,
        field_name="currency",
    )
    record = {
        "account_id": record_id,
        "account_currency": "CNY",
        "source_refs_by_field": {"currency": [geometry_only_ref]},
        "_unresolved_fields": ["currency"],
        "_invalid_observation_fields": ["currency"],
        "_reported_invalid_fields": ["currency"],
    }
    issue_ref = _account_issue_ref(row=1, column=5, field_name="currency")
    issues = [
        _active_account_issue(
            record_id=record_id,
            field_name="account_currency",
            issue_code=issue_code,
            source_ref=issue_ref,
        )
        for issue_code in (
            "candidate_b_account_required_field_unresolved",
            "candidate_b_exact_slot_value_unreadable",
        )
    ]
    context = SimpleNamespace(_personal_detail_extraction_issues=issues)

    candidate_b._reconcile_final_account_field_issues(context, [record])

    assert context._personal_detail_extraction_issues == issues
    assert record["_unresolved_fields"] == ["currency"]
    assert record["_invalid_observation_fields"] == ["currency"]
    assert record["_reported_invalid_fields"] == ["currency"]


def test_final_currency_issue_reconciliation_uses_unambiguous_target_remap() -> None:
    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
        register_issue_target_remap,
    )

    provisional_id = "credit_account_table_observation:pt_23_1"
    record_id = "credit_account:credit_card:20"
    issue_ref = _account_issue_ref(row=1, column=5, field_name="currency")
    corrected_ref = {
        "source": "personal_detail_corrected_page_cell",
        "logical_page": 4,
        "source_page": 2,
        "bbox": [270.0, 120.0, 300.0, 135.0],
        "geometry_scope": "cell",
        "evidence_ids": ["repair:currency:20"],
        "binding": "canonical_field_slot",
        "binding_quality": "canonical_field_slot",
        "field_slot_role": "value",
        "field_name": "currency",
    }
    record = {
        "account_id": record_id,
        "account_currency": "CNY",
        "source_refs_by_field": {"currency": [corrected_ref]},
        "_unresolved_fields": ["currency"],
        "_invalid_observation_fields": ["currency"],
        "_reported_invalid_fields": ["currency"],
    }
    issues = [
        _active_account_issue(
            record_id=provisional_id,
            field_name="account_currency",
            issue_code="candidate_b_exact_slot_value_unreadable",
            source_ref=issue_ref,
        ),
        _active_account_issue(
            record_id=record_id,
            field_name="currency",
            issue_code="candidate_b_account_required_field_unresolved",
            source_ref=issue_ref,
        ),
    ]
    context = SimpleNamespace(_personal_detail_extraction_issues=issues)
    register_issue_target_remap(context, provisional_id, record_id)

    candidate_b._reconcile_final_account_field_issues(context, [record])

    assert context._personal_detail_extraction_issues == []
    assert "_unresolved_fields" not in record
    assert "_invalid_observation_fields" not in record
    assert "_reported_invalid_fields" not in record


def test_candidate_b_branch_has_no_shared_extraction_or_assembly_imports() -> None:
    root = Path("docmirror/plugins/credit_report/personal_detail_scanned")
    active = "\n".join(
        (root / name).read_text(encoding="utf-8") for name in ("candidate_b.py", "context.py", "variant.py")
    )

    assert "extract_scanned_credit_business" not in active
    assert "link_repayment_records_to_accounts" not in active
    assert "assemble_credit_report_business" not in active


def test_only_business_repair_coordinator_can_request_plugin_page_ocr() -> None:
    root = Path("docmirror/plugins/credit_report/personal_detail_scanned")
    forbidden = "\n".join(
        (root / name).read_text(encoding="utf-8")
        for name in (
            "canonical_layout.py",
            "native_extraction.py",
            "native_parser.py",
            "ocr_correction.py",
            "page_topology.py",
        )
    )
    coordinator = (root / "business_repair.py").read_text(encoding="utf-8")

    assert "full_page_ocr_evidence" not in forbidden
    assert "page_ocr_loader" not in forbidden
    assert "page_ocr_loader" in coordinator


def test_variant_discards_conflicting_projector_candidates() -> None:
    authoritative = {
        "credit_accounts": [{"account_id": "candidate-b"}],
        "repayment_records": [],
    }
    context = SimpleNamespace(candidate_b_extraction=lambda _text: SimpleNamespace(business=authoritative))

    result = PersonalDetailScannedVariant().assemble_business(
        SimpleNamespace(),
        "",
        content_mode="scanned_ocr",
        existing_collections={"credit_accounts": [{"account_id": "legacy"}]},
        existing_summary={"projected_account_count": 999},
        variant_input=context,
    )

    assert result == authoritative
    assert result is not authoritative


def test_context_builds_candidate_b_once(monkeypatch) -> None:
    marker = SimpleNamespace(business={}, section_content={}, audit={})
    calls: list[str] = []

    class Pipeline:
        def __init__(self, _context, full_text: str) -> None:
            calls.append(full_text)

        def run(self):
            return marker

    monkeypatch.setattr(
        "docmirror.plugins.credit_report.personal_detail_scanned.candidate_b.CandidateBPipeline",
        Pipeline,
    )
    context = object.__new__(PersonalDetailExtractionContext)
    context._cache = {}

    assert context.candidate_b_extraction("source") is marker
    assert context.candidate_b_extraction("source") is marker
    assert calls == ["source"]


def test_profile_uncertainty_reaches_the_single_repair_planner(monkeypatch) -> None:
    profile_table = SimpleNamespace(
        table_id="profile-unreadable",
        metadata={
            "canonical_template_id": "report_header_and_identity",
            "raw_rows": [["性别"], ["无法辨认"]],
        },
        rows=[],
    )
    profile_page = SimpleNamespace(
        page_number=1,
        source_page_number=1,
        canonical_template_id="report_header_and_identity",
        tables=[profile_table],
    )
    planned_issues: list[dict[str, object]] = []

    context = SimpleNamespace(
        pages=[profile_page],
        reading_order_by_logical={},
        account_collections=lambda: ([], [], []),
        corrected_repayment_records=lambda: [],
        corrected_repayment_micro_grids=lambda: [],
        prepare_candidate_b_business_repair=lambda _payload: (
            planned_issues.extend(list(getattr(context, "_personal_detail_extraction_issues", ()))) or False
        ),
        correct_candidate_b_datasets=lambda payload: payload,
        ocr_correction_audit=lambda: {"business_repair": {}},
        canonical_layout_audit=lambda: {},
        page_topology_audit=lambda: {},
    )

    def empty(_context):
        return []

    for name in (
        "_extract_credit_lines",
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
    monkeypatch.setattr(native_extraction, "_extract_header_datasets", lambda _context, _text: {})
    monkeypatch.setattr(native_extraction, "_extract_personal_notes", lambda _context: ([], []))
    monkeypatch.setattr(native_extraction, "_extract_profile_detail_records", lambda _context: {})
    monkeypatch.setattr(native_extraction, "_extract_summary_datasets", lambda _context: ([], []))
    monkeypatch.setattr(native_extraction, "_record_pre_repair_source_gaps", lambda *_args: None)
    monkeypatch.setattr(native_extraction, "_source_completeness_ledger", lambda _context: {})
    monkeypatch.setattr(native_extraction, "reconcile_candidate_b_credit_lines", lambda _context, rows: rows)
    monkeypatch.setattr(
        "docmirror.plugins.credit_report.personal_detail_scanned.relations.link_candidate_b_repayments",
        lambda rows, *_args, **_kwargs: rows,
    )
    monkeypatch.setattr(
        "docmirror.plugins.credit_report.personal_detail_scanned.relations.derive_candidate_b_overdue_records",
        lambda *_args: [],
    )
    monkeypatch.setattr(
        "docmirror.plugins.credit_report.personal_detail_scanned.source_projection.prepare_personal_detail_source_collections",
        lambda payload, _business, **_kwargs: payload,
    )

    CandidateBPipeline(context, "").run()

    assert any(
        issue.get("field_name") == "gender" and issue.get("issue_code") == "candidate_b_profile_contract_unresolved"
        for issue in planned_issues
    )


def test_profile_schema_withholds_concatenated_multi_region_address() -> None:
    table = SimpleNamespace(
        table_id="profile",
        metadata={
            "canonical_template_id": "report_header_and_identity",
            "raw_rows": [
                ["性别", "出生日期", "通讯地址"],
                ["男", "1990.01.02", "福建省福州市某路1号 江西省上饶市某村2号"],
            ],
        },
        rows=[],
    )
    page = SimpleNamespace(
        page_number=1,
        source_page_number=1,
        canonical_template_id="report_header_and_identity",
        tables=[table],
    )
    context = SimpleNamespace(pages=[page])

    profile = extract_candidate_b_profile(context)

    assert profile["gender"]["normalized_value"] == "男"
    assert profile["birth_date"]["normalized_value"] == "1990-01-02"
    assert profile["mailing_address"]["normalized_value"] is None
    assert profile["mailing_address"]["observation_status"] == "unreadable"
    assert context._personal_detail_extraction_issues[0]["issue_code"] == "candidate_b_profile_contract_unresolved"


def _profile_with_mailing_address(address: str) -> tuple[dict, SimpleNamespace]:
    table = SimpleNamespace(
        table_id="profile-address-contract",
        metadata={
            "canonical_template_id": "report_header_and_identity",
            "raw_rows": [["性别", "通讯地址"], ["男", address]],
        },
        rows=[],
    )
    page = SimpleNamespace(
        page_number=1,
        source_page_number=1,
        canonical_template_id="report_header_and_identity",
        tables=[table],
    )
    context = SimpleNamespace(pages=[page])
    return extract_candidate_b_profile(context), context


def test_profile_address_preserves_province_name_inside_community_proper_name() -> None:
    address = "福建省福州市鼓楼区华林路福建省政府宿舍12号"

    profile, context = _profile_with_mailing_address(address)

    assert profile["mailing_address"]["normalized_value"] == address
    assert profile["mailing_address"]["observation_status"] == "observed"
    assert not hasattr(context, "_personal_detail_extraction_issues")


def _profile_address_token_residue_context(
    *,
    raw: str = "2 福建省福州市鼓楼区华林路1号",
    ordinal_text: str = "2",
    address_text: str = "福建省福州市鼓楼区华林路1号",
    ordinal_bbox: tuple[float, float, float, float] = (2.0, 12.0, 5.0, 18.0),
    address_bbox: tuple[float, float, float, float] = (10.0, 12.0, 90.0, 18.0),
    cell_span: int = 2,
    evidence_ids: tuple[str, ...] = ("address-ordinal", "address-value"),
    token_ids: tuple[str, ...] = ("address-ordinal", "address-value"),
) -> SimpleNamespace:
    value_cell = SimpleNamespace(
        bbox=[0.0, 30.0, 100.0, 40.0],
        geometry_status="exact",
        evidence_ids=list(evidence_ids),
        token_ids=list(token_ids),
        row_span=1,
        col_span=cell_span,
    )
    table = SimpleNamespace(
        table_id="profile-address-token-residue",
        metadata={
            "canonical_template_id": "report_header_and_identity",
            "source_logical_page": 1,
            "source_page": 1,
            "raw_rows": [
                ["性别", "", "", ""],
                ["男", "", "", ""],
                ["通讯地址", "", "户籍地址", ""],
                [raw, "", "--", ""],
            ],
            "source_cell_bboxes": [
                [[0, 0, 25, 10], None, None, None],
                [[0, 10, 25, 20], None, None, None],
                [[0, 20, 100, 30], None, [100, 20, 200, 30], None],
                [[0, 30, 100, 40], None, [100, 30, 200, 40], None],
            ],
            "cell_evidence_ids": [
                [["gender-header"], [], [], []],
                [["gender-value"], [], [], []],
                [["address-header"], [], ["household-header"], []],
                [list(evidence_ids), [], ["household-value"], []],
            ],
            "cell_geometry_status": [
                ["exact", "derived", "derived", "derived"],
                ["exact", "derived", "derived", "derived"],
                ["exact", "derived", "exact", "derived"],
                ["exact", "derived", "exact", "derived"],
            ],
        },
        rows=[],
        source_cell_objects=[
            [None, None, None, None],
            [None, None, None, None],
            [None, None, None, None],
            [value_cell, None, None, None],
        ],
    )
    page = SimpleNamespace(
        page_number=1,
        source_page_number=1,
        canonical_template_id="report_header_and_identity",
        tables=[table],
    )
    atoms = [
        {
            "id": "address-ordinal",
            "text": ordinal_text,
            "bbox": [ordinal_bbox[0], ordinal_bbox[1] + 20, ordinal_bbox[2], ordinal_bbox[3] + 20],
        },
        {
            "id": "address-value",
            "text": address_text,
            "bbox": [address_bbox[0], address_bbox[1] + 20, address_bbox[2], address_bbox[3] + 20],
        },
    ]
    return SimpleNamespace(
        pages=[page],
        evidence_plane=SimpleNamespace(evidence=SimpleNamespace(text_atoms=atoms)),
    )


def test_profile_address_removes_independently_owned_sequence_residue() -> None:
    context = _profile_address_token_residue_context()

    profile = extract_candidate_b_profile(context)

    assert profile["mailing_address"]["normalized_value"] == "福建省福州市鼓楼区华林路1号"
    assert profile["mailing_address"]["raw"] == "2 福建省福州市鼓楼区华林路1号"
    assert not hasattr(context, "_personal_detail_extraction_issues")


@pytest.mark.parametrize(
    ("overrides", "expected_raw"),
    [
        ({"ordinal_text": "20A"}, "2 福建省福州市鼓楼区华林路1号"),
        ({"address_text": "示例路1号"}, "2 福建省福州市鼓楼区华林路1号"),
        ({"ordinal_bbox": (2.0, 12.0, 20.0, 18.0)}, "2 福建省福州市鼓楼区华林路1号"),
        ({"address_bbox": (4.0, 12.0, 90.0, 18.0)}, "2 福建省福州市鼓楼区华林路1号"),
        ({"cell_span": 1}, "2 福建省福州市鼓楼区华林路1号"),
        (
            {"evidence_ids": ("address-ordinal",), "token_ids": ("address-ordinal",)},
            "2 福建省福州市鼓楼区华林路1号",
        ),
        ({"raw": "3 福建省福州市鼓楼区华林路1号"}, "3 福建省福州市鼓楼区华林路1号"),
    ],
)
def test_profile_address_sequence_residue_fails_closed(
    overrides: dict[str, object],
    expected_raw: str,
) -> None:
    context = _profile_address_token_residue_context(**overrides)

    profile = extract_candidate_b_profile(context)

    assert profile["mailing_address"]["normalized_value"] == expected_raw
    assert profile["mailing_address"]["normalized_value"] != "福建省福州市鼓楼区华林路1号"


def test_profile_address_rejects_structural_region_and_provider_concatenation() -> None:
    contaminated = (
        "福建省福州市鼓楼区华林路1号 福建省厦门市思明区湖滨路2号",
        "福建省福州市鼓楼区华林路1号 江西省上饶市信州区广信路2号",
    )

    for address in contaminated:
        profile, context = _profile_with_mailing_address(address)

        assert profile["mailing_address"]["normalized_value"] is None
        assert profile["mailing_address"]["observation_status"] == "unreadable"
        assert profile["mailing_address"]["raw"] == [address]
        assert any(
            issue.get("field_name") == "mailing_address" and issue.get("observed_value") == [address]
            for issue in context._personal_detail_extraction_issues
        )


def test_profile_address_decodes_exact_provider_boundary_and_preserves_full_raw() -> None:
    address = "福建省福州市鼓楼区华林路1号"
    provider = "某银行股份有限公司"
    raw = f"{address} 数据发生机构名称 {provider}"

    profile, context = _profile_with_mailing_address(raw)
    entry = profile["mailing_address"]

    assert entry["normalized_value"] == address
    assert entry["raw"] == raw
    assert entry["provider_evidence"]["normalized_value"] == provider
    assert entry["provider_evidence"]["observation_status"] == "observed"
    assert entry["provider_evidence"]["source_refs"] == entry["source_refs"]
    assert not hasattr(context, "_personal_detail_extraction_issues")


def test_profile_address_keeps_safe_prefix_but_reports_invalid_provider() -> None:
    address = "福建省福州市鼓楼区华林路1号"
    raw = f"{address} 数据发生机构名称 月38"

    profile, context = _profile_with_mailing_address(raw)
    entry = profile["mailing_address"]

    assert entry["normalized_value"] == address
    assert entry["raw"] == raw
    assert entry["provider_evidence"]["normalized_value"] is None
    assert entry["provider_evidence"]["observation_status"] == "unreadable"
    assert any(
        issue.get("issue_code") == "candidate_b_profile_provider_contract_unresolved"
        and issue.get("field_name") == "mailing_address.data_provider"
        and issue.get("observed_value") == "月38"
        for issue in context._personal_detail_extraction_issues
    )


def test_profile_address_with_multiple_provider_markers_is_withheld_and_reported() -> None:
    address = "福建省福州市鼓楼区华林路1号"
    raw = f"{address} 数据发生机构名称 某银行股份有限公司 数据发生机构名称 某征信中心"

    profile, context = _profile_with_mailing_address(raw)
    entry = profile["mailing_address"]

    assert entry["normalized_value"] is None
    assert entry["raw"] == [raw]
    assert entry["provider_evidence"][0]["observation_status"] == "ambiguous"
    assert {issue.get("field_name") for issue in context._personal_detail_extraction_issues} >= {
        "mailing_address",
        "mailing_address.data_provider",
    }


def test_profile_schema_withholds_collapsed_scalars_without_exact_token_owners() -> None:
    table = SimpleNamespace(
        table_id="identity-profile-collapsed",
        metadata={
            "canonical_template_id": "report_header_and_identity",
            "raw_rows": [
                ["出生日期 性别", "d 婚姻状况", "就业状况"],
                [
                    "2002.08.03 男 数据发生机构名称 示例银行",
                    "未婚 数据发生机构名称 示例银行",
                    "职员 数据发生机构名称 示例银行",
                ],
                ["学位 学历", "国赣", "电子邮箱"],
                ["光 大专", "中国（含港澳台）", "1838961623@qq.com"],
            ],
        },
        rows=[],
    )
    page = SimpleNamespace(
        page_number=1,
        source_page_number=1,
        canonical_template_id="report_header_and_identity",
        tables=[table],
    )
    context = SimpleNamespace(pages=[page])

    profile = extract_candidate_b_profile(context)

    assert {
        field: entry["normalized_value"] for field, entry in profile.items() if entry["normalized_value"] is not None
    } == {
        "email": "1838961623@qq.com",
    }
    assert "nationality" not in profile
    assert not any(
        issue.get("field_name") == "nationality" for issue in getattr(context, "_personal_detail_extraction_issues", ())
    )
    assert "marital_status" not in profile
    assert not any(
        issue.get("field_name") == "marital_status"
        for issue in getattr(context, "_personal_detail_extraction_issues", ())
    )
    for field in (
        "gender",
        "birth_date",
        "employment_status",
        "education_level",
        "degree",
    ):
        assert profile[field]["normalized_value"] is None
        assert any(
            issue.get("issue_code") == "candidate_b_profile_contract_unresolved" and issue.get("field_name") == field
            for issue in context._personal_detail_extraction_issues
        )


@pytest.mark.parametrize(
    "provider",
    (
        "导某银行股份有限公司",
        "S 某银行股份有限公司",
        "某银行股份有限公司 Ss",
    ),
)
def test_profile_address_provider_never_deletes_unowned_source_glyphs(
    provider: str,
) -> None:
    address = "福建省福州市鼓楼区华林路1号"
    raw = f"{address} 数据发生机构名称 {provider}"

    profile, context = _profile_with_mailing_address(raw)
    entry = profile["mailing_address"]

    assert entry["normalized_value"] == address
    assert entry["raw"] == raw
    assert entry["provider_evidence"]["normalized_value"] is None
    assert entry["provider_evidence"]["observation_status"] == "unreadable"
    assert any(
        issue.get("issue_code") == "candidate_b_profile_provider_contract_unresolved"
        and issue.get("field_name") == "mailing_address.data_provider"
        and issue.get("observed_value") == provider
        for issue in context._personal_detail_extraction_issues
    )


def test_candidate_b_publishes_owner_census_before_projection_only(
    monkeypatch,
) -> None:
    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
        make_issue,
    )
    from docmirror.plugins.credit_report.personal_detail_scanned.source_projection import (
        prepare_personal_detail_source_collections as real_prepare,
    )

    captured_facts: dict[str, object] = {}
    metadata_id = "personal_report_metadata:production-owner"
    canonical_audit = {
        "registrations": [
            {
                "logical_page": 1,
                "source_page": 1,
                "template_id": "report_header_and_identity",
                "status": "registered",
            },
            {
                "logical_page": 3,
                "source_page": 3,
                "template_id": "report_header_and_identity",
                "status": "registered",
            },
        ],
        "fragment_groups": [
            {
                "template_id": "report_header_and_identity",
                "fragment_logical_pages": [1],
                "source_pages": [1],
            },
            {
                "template_id": "report_header_and_identity",
                "fragment_logical_pages": [3],
                "source_pages": [3],
            },
        ],
    }
    context = SimpleNamespace(
        pages=[],
        reading_order_by_logical={},
        account_collections=lambda: ([], [], []),
        corrected_repayment_records=lambda: [],
        corrected_repayment_micro_grids=lambda: [],
        prepare_candidate_b_business_repair=lambda _payload: False,
        correct_candidate_b_datasets=lambda payload: payload,
        ocr_correction_audit=lambda: {"business_repair": {}},
        canonical_layout_audit=lambda: canonical_audit,
        page_topology_audit=lambda: {},
        _personal_detail_extraction_issues=[
            make_issue(
                category="ocr_cell_level_error",
                issue_code="candidate_b_exact_slot_value_unreadable",
                message="Exact report-header subject name was unreadable.",
                parser_stage="candidate_b_report_header_exact_slot",
                target_dataset="personal_report_metadata",
                target_record_id=metadata_id,
                field_name="subject_name",
                observed_value={"raw": "", "slot_state": "blank_or_unreadable"},
                source_refs=(
                    {
                        "source": "native_detail_table_cell",
                        "logical_page": 3,
                        "source_page": 3,
                        "table_id": "report-header-page-three",
                        "row": 1,
                        "column": 0,
                        "bbox": [20.0, 40.0, 120.0, 60.0],
                        "geometry_scope": "cell",
                        "evidence_ids": ["native:report-header-page-three:1:0"],
                        "binding": "canonical_field_slot",
                        "binding_quality": "canonical_header_column",
                        "field_name": "subject_name",
                    },
                ),
                reason_codes=(
                    "canonical_field_slot",
                    "normalized_value_withheld",
                ),
            )
        ],
    )

    def empty(_context):
        return []

    for name in (
        "_extract_credit_lines",
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
        lambda _context, _text: {
            "personal_report_metadata": [
                {
                    "record_id": metadata_id,
                    "personal_report_metadata_id": metadata_id,
                    "subject_name": None,
                    "primary_id_type": None,
                    "primary_id_number": None,
                }
            ]
        },
    )
    monkeypatch.setattr(native_extraction, "_extract_personal_notes", lambda _context: ([], []))
    monkeypatch.setattr(native_extraction, "_extract_profile_detail_records", lambda _context: {})
    monkeypatch.setattr(native_extraction, "_extract_summary_datasets", lambda _context: ([], []))
    monkeypatch.setattr(native_extraction, "_record_pre_repair_source_gaps", lambda *_args: None)
    monkeypatch.setattr(native_extraction, "_source_completeness_ledger", lambda _context: {})
    monkeypatch.setattr(
        native_extraction,
        "reconcile_candidate_b_credit_lines",
        lambda _context, rows: rows,
    )
    monkeypatch.setattr(
        "docmirror.plugins.credit_report.personal_detail_scanned.profile_extraction.extract_candidate_b_profile",
        lambda _context: {},
    )
    monkeypatch.setattr(
        "docmirror.plugins.credit_report.personal_detail_scanned.relations.link_candidate_b_repayments",
        lambda rows, *_args, **_kwargs: rows,
    )
    monkeypatch.setattr(
        "docmirror.plugins.credit_report.personal_detail_scanned.relations.derive_candidate_b_overdue_records",
        lambda *_args: [],
    )

    def capture_projection(payload, _business, **_kwargs):
        captured_facts.update(payload["facts"])
        return real_prepare(payload, _business, **_kwargs)

    monkeypatch.setattr(
        "docmirror.plugins.credit_report.personal_detail_scanned.source_projection.prepare_personal_detail_source_collections",
        capture_projection,
    )

    result = CandidateBPipeline(context, "").run()

    assert captured_facts["_personal_detail_canonical_layout_owner_census"] == canonical_audit
    assert "_personal_detail_canonical_layout_owner_census" not in result.section_content["facts"]
    assert result.audit["canonical_layout"] == canonical_audit
    assert result.audit["canonical_layout"] is not canonical_audit
    mirror = [
        issue
        for issue in result.section_content["datasets"]["personal_detail_extraction_issues"]
        if issue.get("issue_code") == "source_bound_profile_field_omitted"
    ]
    assert len(mirror) == 1
    assert mirror[0]["target_dataset"] == "personal_profile"
    assert mirror[0]["target_record_id"] == "personal_profile:1"
    assert mirror[0]["field_name"] == "subject_name"


def test_profile_schema_recovers_clipped_nationality_header_only_from_exact_four_column_signature() -> None:
    header_atom = SimpleNamespace(id="h2", text="国", bbox=[55.0, 1.0, 65.0, 9.0])
    value_atom = SimpleNamespace(
        id="v2",
        text="中国(含港澳台)",
        bbox=[52.0, 11.0, 73.0, 19.0],
    )
    header_cell = SimpleNamespace(
        text="国",
        geometry_status="exact",
        evidence_ids=["h2"],
        token_ids=["h2"],
        bbox=[50.0, 0.0, 75.0, 10.0],
    )
    value_cell = SimpleNamespace(
        text="中国(含港澳台)",
        geometry_status="exact",
        evidence_ids=["v2"],
        token_ids=["v2"],
        bbox=[50.0, 10.0, 75.0, 20.0],
    )
    table = SimpleNamespace(
        table_id="identity-profile-clipped-nationality",
        metadata={
            "canonical_template_id": "report_header_and_identity",
            "raw_rows": [
                ["学历", "学位", "国", "电子邮箱"],
                ["大专", "--", "中国(含港澳台)", "user@example.com"],
            ],
            "source_cell_bboxes": [
                [[0, 0, 25, 10], [25, 0, 50, 10], [50, 0, 75, 10], [75, 0, 100, 10]],
                [[0, 10, 25, 20], [25, 10, 50, 20], [50, 10, 75, 20], [75, 10, 100, 20]],
            ],
            "cell_evidence_ids": [
                [["h0"], ["h1"], ["h2"], ["h3"]],
                [["v0"], ["v1"], ["v2"], ["v3"]],
            ],
            "cell_geometry_status": [["exact"] * 4, ["exact"] * 4],
        },
        rows=[],
        source_cell_objects=[
            [None, None, header_cell, None],
            [None, None, value_cell, None],
        ],
    )
    page = SimpleNamespace(
        page_number=1,
        source_page_number=1,
        canonical_template_id="report_header_and_identity",
        tables=[table],
    )

    profile = extract_candidate_b_profile(
        SimpleNamespace(
            pages=[page],
            evidence_plane=SimpleNamespace(evidence=SimpleNamespace(text_atoms=[header_atom, value_atom])),
        )
    )

    assert profile["nationality"]["normalized_value"] == "中国(含港澳台)"
    ref = profile["nationality"]["source_refs"][0]
    assert ref["bbox"] == [50, 10, 75, 20]
    assert ref["evidence_ids"] == ["v2"]
    assert ref["geometry_status"] == "exact"


def test_profile_schema_does_not_infer_clipped_nationality_without_complete_signature() -> None:
    table = SimpleNamespace(
        table_id="identity-profile-clipped-nationality-incomplete",
        metadata={
            "canonical_template_id": "report_header_and_identity",
            "raw_rows": [
                ["学历", "学位", "国", ""],
                ["大专", "--", "中国(含港澳台)", ""],
            ],
        },
        rows=[],
    )
    page = SimpleNamespace(
        page_number=1,
        source_page_number=1,
        canonical_template_id="report_header_and_identity",
        tables=[table],
    )

    profile = extract_candidate_b_profile(SimpleNamespace(pages=[page]))

    assert "nationality" not in profile


def _visual_profile_degree_context(*, dash_only: bool) -> SimpleNamespace:
    import cv2
    import numpy as np

    image = np.full((100, 200, 3), 255, dtype=np.uint8)
    if dash_only:
        cv2.line(image, (82, 40), (90, 40), (0, 0, 0), 2)
        cv2.line(image, (98, 40), (106, 40), (0, 0, 0), 2)
    else:
        cv2.rectangle(image, (84, 31), (102, 49), (0, 0, 0), -1)
    value_cell = SimpleNamespace(
        bbox=[20.0, 20.0, 180.0, 60.0],
        geometry_status="exact",
        evidence_ids=["degree-value"],
    )
    table = SimpleNamespace(
        table_id="identity-profile-visual-degree",
        metadata={
            "canonical_template_id": "report_header_and_identity",
            "source_logical_page": 1,
            "source_page": 1,
            "raw_rows": [["学位"], ["中"]],
            "source_cell_bboxes": [[[20, 5, 180, 20]], [[20, 20, 180, 60]]],
            "cell_evidence_ids": [[["degree-header"]], [["degree-value"]]],
            "cell_geometry_status": [["exact"], ["exact"]],
        },
        rows=[],
        source_cell_objects=[[None], [value_cell]],
    )
    page = SimpleNamespace(
        page_number=1,
        source_page_number=1,
        canonical_template_id="report_header_and_identity",
        tables=[table],
    )
    return SimpleNamespace(
        pages=[page],
        _page_image_resolver=lambda _page: {
            "image": image,
            "page_width": 200.0,
            "page_height": 100.0,
        },
    )


def test_profile_schema_visual_dash_shape_recovers_exact_source_absence() -> None:
    context = _visual_profile_degree_context(dash_only=True)

    profile = extract_candidate_b_profile(context)

    assert profile["degree"]["normalized_value"] is None
    assert profile["degree"]["observation_status"] == "source_absent"
    assert profile["degree"]["raw"] == ["--"]
    assert not hasattr(context, "_personal_detail_extraction_issues")


def test_profile_schema_visual_dash_shape_rejects_non_dash_foreground() -> None:
    context = _visual_profile_degree_context(dash_only=False)

    profile = extract_candidate_b_profile(context)

    assert profile["degree"]["normalized_value"] is None
    assert profile["degree"]["observation_status"] == "unreadable"
    assert profile["degree"]["raw"] == ["中"]
    assert context._personal_detail_extraction_issues[0]["field_name"] == "degree"


def test_profile_schema_visual_dash_shape_ignores_pale_watermark_and_specks() -> None:
    import cv2
    import numpy as np

    context = _visual_profile_degree_context(dash_only=False)
    image = context._page_image_resolver(1)["image"]
    image[:] = 255
    cv2.line(image, (94, 40), (99, 40), (65, 65, 65), 1)
    cv2.line(image, (102, 40), (107, 40), (65, 65, 65), 1)
    cv2.circle(image, (155, 35), 11, (190, 190, 190), 2)
    image[35, 55] = np.asarray((100, 100, 100), dtype=np.uint8)

    profile = extract_candidate_b_profile(context)

    assert profile["degree"]["normalized_value"] is None
    assert profile["degree"]["observation_status"] == "source_absent"
    assert profile["degree"]["raw"] == ["--"]
    assert not hasattr(context, "_personal_detail_extraction_issues")


def test_profile_schema_does_not_mine_provider_suffix_for_identity_value() -> None:
    table = SimpleNamespace(
        table_id="identity-profile-provider-suffix",
        metadata={
            "canonical_template_id": "report_header_and_identity",
            "raw_rows": [["性别"], ["数据发生机构名称 女士银行"]],
        },
        rows=[],
    )
    page = SimpleNamespace(
        page_number=1,
        source_page_number=1,
        canonical_template_id="report_header_and_identity",
        tables=[table],
    )
    context = SimpleNamespace(pages=[page])

    profile = extract_candidate_b_profile(context)

    assert profile["gender"]["normalized_value"] is None
    assert profile["gender"]["observation_status"] == "unreadable"
    assert context._personal_detail_extraction_issues[0]["field_name"] == "gender"


def test_profile_schema_keeps_printed_placeholder_as_source_absent() -> None:
    table = SimpleNamespace(
        table_id="identity-profile-placeholder",
        metadata={
            "canonical_template_id": "report_header_and_identity",
            "raw_rows": [["性别", "婚姻状况"], ["--", "--"]],
        },
        rows=[],
    )
    page = SimpleNamespace(
        page_number=1,
        source_page_number=1,
        canonical_template_id="report_header_and_identity",
        tables=[table],
    )
    context = SimpleNamespace(pages=[page])

    profile = extract_candidate_b_profile(context)

    assert profile["gender"]["normalized_value"] is None
    assert profile["gender"]["observation_status"] == "source_absent"
    assert profile["marital_status"]["normalized_value"] is None
    assert profile["marital_status"]["observation_status"] == "source_absent"
    assert not hasattr(context, "_personal_detail_extraction_issues")


def test_profile_schema_does_not_infer_household_address_from_damaged_label() -> None:
    table = SimpleNamespace(
        table_id="identity-profile-address-slots",
        metadata={
            "canonical_template_id": "report_header_and_identity",
            "raw_rows": [
                ["性别", "", ""],
                ["男", "", ""],
                ["通讯地址", "", "户瓣地址"],
                ["福建省泉州市示例路1号", "", ""],
            ],
        },
        rows=[],
    )
    page = SimpleNamespace(
        page_number=1,
        source_page_number=1,
        canonical_template_id="report_header_and_identity",
        tables=[table],
    )
    context = SimpleNamespace(pages=[page])

    profile = extract_candidate_b_profile(context)

    assert profile["mailing_address"]["normalized_value"] == "福建省泉州市示例路1号"
    assert "household_address" not in profile
    assert not hasattr(context, "_personal_detail_extraction_issues")


def test_profile_schema_recovers_exact_mailing_header_with_one_han_residue() -> None:
    """Mirror the Lin profile row without depending on the private PDF."""

    header_cell = SimpleNamespace(
        bbox=[0.0, 20.0, 100.0, 30.0],
        geometry_status="exact",
        evidence_ids=["mailing-residue", "mailing-label"],
        token_ids=["mailing-residue", "mailing-label"],
        row_span=1,
        col_span=1,
    )
    value_cell = SimpleNamespace(
        bbox=[0.0, 30.0, 100.0, 40.0],
        geometry_status="exact",
        evidence_ids=["mailing-value"],
        token_ids=["mailing-value"],
        row_span=1,
        col_span=1,
    )
    address = "福建省福州市仓山区仓山镇仓山村委会卢滨路中庚城19座704"
    table = SimpleNamespace(
        table_id="lin-shaped-profile-address",
        metadata={
            "canonical_template_id": "report_header_and_identity",
            "source_logical_page": 1,
            "source_page": 1,
            "raw_rows": [
                ["性别", "出生日期", "婚姻状况", "就业状况"],
                ["男", "1983.11.01", "已婚", "在职"],
                ["人 通讯地址", "", "户籍地址", "A"],
                [address, "", "", "福"],
            ],
            "source_cell_bboxes": [
                [[0, 0, 25, 10], [25, 0, 50, 10], [50, 0, 75, 10], [75, 0, 100, 10]],
                [[0, 10, 25, 20], [25, 10, 50, 20], [50, 10, 75, 20], [75, 10, 100, 20]],
                [[0, 20, 100, 30], None, [100, 20, 180, 30], [180, 20, 200, 30]],
                [[0, 30, 100, 40], None, [100, 30, 180, 40], [180, 30, 200, 40]],
            ],
            "cell_geometry_status": [
                ["exact", "exact", "exact", "exact"],
                ["exact", "exact", "exact", "exact"],
                ["exact", "derived", "exact", "exact"],
                ["exact", "derived", "exact", "exact"],
            ],
        },
        rows=[],
        source_cell_objects=[
            [None, None, None, None],
            [None, None, None, None],
            [header_cell, None, None, None],
            [value_cell, None, None, None],
        ],
    )
    page = SimpleNamespace(
        page_number=1,
        source_page_number=1,
        canonical_template_id="report_header_and_identity",
        tables=[table],
    )
    context = SimpleNamespace(
        pages=[page],
        evidence_plane=SimpleNamespace(
            evidence=SimpleNamespace(
                text_atoms=[
                    {
                        "id": "mailing-residue",
                        "text": "人",
                        "bbox": [2.0, 22.0, 8.0, 28.0],
                        "logical_page": 1,
                    },
                    {
                        "id": "mailing-label",
                        "text": "通讯地址",
                        # Mirrors the native OCR seam: immutable owners are
                        # distinct even though their token boxes overlap.
                        "bbox": [7.0, 22.0, 45.0, 28.0],
                        "logical_page": 1,
                    },
                    {
                        "id": "mailing-value",
                        "text": address,
                        "bbox": [2.0, 32.0, 95.0, 38.0],
                        "logical_page": 1,
                    },
                ]
            )
        ),
    )

    profile = extract_candidate_b_profile(context)

    assert profile["mailing_address"]["normalized_value"] == address
    assert profile["mailing_address"]["observation_status"] == "observed"


def test_profile_schema_ignores_residence_and_employment_contact_tables() -> None:
    profile_table = SimpleNamespace(
        table_id="identity-profile",
        metadata={
            "canonical_template_id": "report_header_and_identity",
            "raw_rows": [
                [
                    "性别",
                    "出生日期",
                    "婚姻状况",
                    "就业状况",
                    "学历",
                    "学位",
                    "国籍",
                    "手机号码",
                    "住宅电话",
                    "单位电话",
                    "电子邮箱",
                    "通讯地址",
                    "户籍地址",
                ],
                [
                    "男",
                    "--",
                    "--",
                    "--",
                    "--",
                    "--",
                    "--",
                    "13800138000",
                    "010-12345678",
                    "021-87654321",
                    "--",
                    "北京市朝阳区示例路1号",
                    "--",
                ],
            ],
        },
        rows=[],
    )
    residence_table = SimpleNamespace(
        table_id="residence-history",
        metadata={
            "canonical_template_id": "report_header_and_identity",
            "raw_rows": [
                ["编号", "居住地址", "住宅电话", "通讯地址", "居住状况"],
                ["1", "上海市浦东新区示例路2号", "010-99999999", "上海市浦东新区示例路2号", "租房"],
            ],
        },
        rows=[],
    )
    employment_table = SimpleNamespace(
        table_id="employment-history",
        metadata={
            "canonical_template_id": "report_header_and_identity",
            "raw_rows": [
                ["编号", "工作单位", "单位地址", "单位电话", "通讯地址"],
                ["1", "示例单位", "深圳市南山区示例路3号", "0755-99999999", "深圳市南山区示例路3号"],
            ],
        },
        rows=[],
    )
    page = SimpleNamespace(
        page_number=1,
        source_page_number=1,
        canonical_template_id="report_header_and_identity",
        tables=[profile_table, residence_table, employment_table],
    )
    context = SimpleNamespace(pages=[page])

    profile = extract_candidate_b_profile(context)

    assert profile["mobile_phone"]["normalized_value"] == "13800138000"
    assert profile["residence_phone"]["normalized_value"] == "010-12345678"
    assert profile["work_phone"]["normalized_value"] == "021-87654321"
    assert profile["mailing_address"]["normalized_value"] == "北京市朝阳区示例路1号"
    assert not hasattr(context, "_personal_detail_extraction_issues")


def test_profile_schema_ignores_noncanonical_profile_like_tables() -> None:
    table = SimpleNamespace(
        table_id="unrelated-contact-table",
        metadata={
            "canonical_template_id": "credit_account_detail",
            "raw_rows": [
                ["性别", "手机号码", "通讯地址"],
                ["女", "13900139000", "广东省深圳市南山区示例路1号"],
            ],
        },
        rows=[],
    )
    page = SimpleNamespace(
        page_number=5,
        source_page_number=3,
        canonical_template_id="credit_account_detail",
        tables=[table],
    )

    assert extract_candidate_b_profile(SimpleNamespace(pages=[page])) == {}


def test_profile_schema_rejects_alphanumeric_phone_contamination() -> None:
    table = SimpleNamespace(
        table_id="identity-profile",
        metadata={
            "canonical_template_id": "report_header_and_identity",
            "raw_rows": [
                ["性别", "手机号码", "住宅电话", "单位电话"],
                ["男", "138O0138000", "010A12345678", "021B87654321"],
            ],
        },
        rows=[],
    )
    page = SimpleNamespace(
        page_number=1,
        source_page_number=1,
        canonical_template_id="report_header_and_identity",
        tables=[table],
    )
    context = SimpleNamespace(pages=[page])

    profile = extract_candidate_b_profile(context)

    for field in ("mobile_phone", "residence_phone", "work_phone"):
        assert profile[field]["normalized_value"] is None
        assert profile[field]["observation_status"] == "unreadable"
    assert {issue["field_name"] for issue in context._personal_detail_extraction_issues} >= {
        "mobile_phone",
        "residence_phone",
        "work_phone",
    }


def test_profile_schema_does_not_report_roles_absent_from_printed_template() -> None:
    table = SimpleNamespace(
        table_id="identity-profile",
        metadata={
            "canonical_template_id": "report_header_and_identity",
            "raw_rows": [
                ["性别", "手机号码", "通讯地址"],
                ["女", "13800138000", "北京市朝阳区示例路1号"],
            ],
        },
        rows=[],
    )
    page = SimpleNamespace(
        page_number=1,
        source_page_number=1,
        canonical_template_id="report_header_and_identity",
        tables=[table],
    )
    context = SimpleNamespace(pages=[page])

    profile = extract_candidate_b_profile(context)

    assert "work_phone" not in profile
    assert "residence_phone" not in profile
    assert not hasattr(context, "_personal_detail_extraction_issues")


def test_profile_schema_reports_visible_label_with_unreadable_value() -> None:
    table = SimpleNamespace(
        table_id="identity-profile",
        metadata={
            "canonical_template_id": "report_header_and_identity",
            "raw_rows": [["性别", "国籍"], ["女", ""]],
        },
        rows=[],
    )
    page = SimpleNamespace(
        page_number=1,
        source_page_number=1,
        canonical_template_id="report_header_and_identity",
        tables=[table],
    )
    context = SimpleNamespace(pages=[page])

    profile = extract_candidate_b_profile(context)

    assert profile["nationality"]["observation_status"] == "unreadable"
    assert context._personal_detail_extraction_issues[0]["field_name"] == "nationality"


def test_account_schema_withholds_unreadable_ordinal_and_uses_provisional_id() -> None:
    context = SimpleNamespace(
        corrected_evidence_pages=lambda: [
            {
                "page": 1,
                "source_page": 1,
                "lines": [
                    {"text": "贷记卡账户"},
                    {"text": "账户 1：发卡机构甲"},
                    {"text": "账户（发卡机构乙）"},
                ],
            }
        ]
    )

    rows = native_extraction._account_anchor_skeletons(context)

    assert [row["account_type"] for row in rows] == ["credit_card", "credit_card"]
    assert rows[0]["category_sequence"] == 1
    assert "category_sequence" not in rows[1]
    assert rows[1]["account_id"].startswith("credit_account_provisional:")
    issue = context._personal_detail_extraction_issues[0]
    assert issue["issue_code"] == "candidate_b_account_printed_ordinal_unresolved"
    assert issue["target_record_id"] == rows[1]["account_id"]
    assert "encounter_order_not_used" in issue["reason_codes"]


def test_account_schema_withholds_every_duplicate_printed_ordinal() -> None:
    context = SimpleNamespace(
        corrected_evidence_pages=lambda: [
            {
                "page": 1,
                "source_page": 1,
                "lines": [
                    {"text": "贷记卡账户", "bbox": [10, 10, 100, 20]},
                    {"text": "账户 1：发卡机构甲", "bbox": [10, 30, 200, 40]},
                    {"text": "账户 1：发卡机构乙", "bbox": [10, 130, 200, 140]},
                ],
            }
        ]
    )

    rows = native_extraction._account_anchor_skeletons(context)

    assert len(rows) == 2
    assert all("category_sequence" not in row for row in rows)
    assert len({row["account_id"] for row in rows}) == 2
    assert all(row["account_id"].startswith("credit_account_provisional:") for row in rows)
    issues = [
        issue
        for issue in context._personal_detail_extraction_issues
        if issue["issue_code"] == "candidate_b_account_printed_ordinal_unresolved"
    ]
    assert len(issues) == 2
    assert all(issue["observed_value"]["ordinal_status"] == "printed_duplicate" for issue in issues)


def test_account_schema_completes_exact_canonical_singleton_ordinal(monkeypatch) -> None:
    context = SimpleNamespace(
        corrected_evidence_pages=lambda: [
            {
                "page": 4,
                "source_page": 2,
                "lines": [
                    {"text": "\u5faa\u73af\u8d37\u8d26\u6237\uff08\u4e8c\uff09", "bbox": [10, 10, 200, 30]},
                    {"text": "\u8d26\u6237", "bbox": [10, 40, 100, 60]},
                ],
            }
        ]
    )
    table_id = "credit_account_table_observation:singleton"
    table = {
        "account_id": table_id,
        "_table_observation_id": table_id,
        "account_type": "revolving_loan_account",
        "source": "native_detail_account_table",
        "management_institution": "\u793a\u4f8b\u94f6\u884c",
        "account_identifier": "D10053310H00012022052901021012089466554314",
        "open_date": "2022-05-29",
        "currency": "CNY",
        "source_refs": [{"logical_page": 4, "bbox": [10, 100, 500, 180]}],
    }
    monkeypatch.setattr(
        native_extraction,
        "_extract_table_accounts",
        lambda _context: ([table], [], []),
    )

    accounts, repayments, events = native_extraction._extract_accounts(context)

    assert repayments == []
    assert events == []
    assert len(accounts) == 1
    assert accounts[0]["account_id"] == "credit_account:revolving_loan_account:1"
    assert accounts[0]["category_sequence"] == 1
    assert accounts[0]["_printed_ordinal_status"] == "canonical_singleton_inferred"
    issue_codes = {issue.get("issue_code") for issue in getattr(context, "_personal_detail_extraction_issues", ())}
    assert "candidate_b_account_printed_ordinal_unresolved" not in issue_codes
    assert "candidate_b_account_sequence_gap" not in issue_codes
    assert "monthly_population_incomplete_from_account_gap" not in issue_codes


def test_account_schema_completes_singleton_after_same_card_native_replay(
    monkeypatch,
) -> None:
    context = SimpleNamespace(
        corrected_evidence_pages=lambda: [
            {
                "page": 4,
                "source_page": 2,
                "lines": [
                    {"text": "循环贷账户（二）", "bbox": [10, 10, 200, 30]},
                    {"text": "账户", "bbox": [10, 40, 100, 60]},
                ],
            }
        ]
    )
    account_identifier = "D10000000H00012022052901021012000000000001"
    primary = {
        "account_id": "credit_account_table_observation:primary",
        "_table_observation_id": "credit_account_table_observation:primary",
        "account_type": "revolving_loan_account",
        "source": "native_detail_account_table",
        "account_identifier": account_identifier,
        "management_institution": "示例银行",
        "source_refs": [
            {
                "logical_page": 4,
                "source_page": 2,
                "table_id": "account-card-primary",
                "bbox": [10, 100, 500, 280],
            }
        ],
    }
    replay = {
        **primary,
        "account_id": "credit_account_table_observation:replay",
        "_table_observation_id": "credit_account_table_observation:replay",
        "source_refs": [
            {
                "logical_page": 4,
                "source_page": 2,
                "table_id": "account-card-replay",
                "bbox": [11, 101, 499, 279],
            }
        ],
    }
    monkeypatch.setattr(
        native_extraction,
        "_extract_table_accounts",
        lambda _context: ([primary, replay], [], []),
    )

    accounts, repayments, events = native_extraction._extract_accounts(context)

    assert repayments == []
    assert events == []
    assert len(accounts) == 1
    assert accounts[0]["account_id"] == "credit_account:revolving_loan_account:1"
    assert accounts[0]["category_sequence"] == 1
    assert accounts[0]["_printed_ordinal_status"] == "canonical_singleton_inferred"
    assert {ref.get("table_id") for ref in accounts[0]["source_refs"] if ref.get("table_id")} >= {
        "account-card-primary",
        "account-card-replay",
    }
    issue_codes = {issue.get("issue_code") for issue in getattr(context, "_personal_detail_extraction_issues", ())}
    assert not issue_codes.intersection(
        {
            "candidate_b_account_printed_ordinal_unresolved",
            "candidate_b_account_sequence_gap",
            "candidate_b_unmatched_account_table_suppressed",
            "monthly_population_incomplete_from_account_gap",
        }
    )


def test_account_schema_does_not_promote_two_distinct_unnumbered_cards(
    monkeypatch,
) -> None:
    context = SimpleNamespace(
        corrected_evidence_pages=lambda: [
            {
                "page": 4,
                "source_page": 2,
                "lines": [
                    {"text": "循环贷账户（二）", "bbox": [10, 10, 200, 30]},
                    {"text": "账户", "bbox": [10, 40, 100, 60]},
                ],
            }
        ]
    )
    tables = [
        {
            "account_id": f"credit_account_table_observation:{index}",
            "_table_observation_id": f"credit_account_table_observation:{index}",
            "account_type": "revolving_loan_account",
            "source": "native_detail_account_table",
            "account_identifier": account_identifier,
            "source_refs": [
                {
                    "logical_page": 4,
                    "source_page": 2,
                    "table_id": f"account-card-{index}",
                    "bbox": bbox,
                }
            ],
        }
        for index, account_identifier, bbox in (
            ("one", "D10000000000000000000000000000000000000001", [10, 100, 500, 220]),
            ("two", "D10000000000000000000000000000000000000002", [10, 240, 500, 360]),
        )
    ]
    monkeypatch.setattr(
        native_extraction,
        "_extract_table_accounts",
        lambda _context: (tables, [], []),
    )

    accounts, _repayments, _events = native_extraction._extract_accounts(context)

    assert len(accounts) == 1
    assert accounts[0]["account_id"].startswith("credit_account_provisional:")
    assert "category_sequence" not in accounts[0]
    issue_codes = {issue.get("issue_code") for issue in getattr(context, "_personal_detail_extraction_issues", ())}
    assert {
        "candidate_b_account_printed_ordinal_unresolved",
        "candidate_b_account_sequence_gap",
        "candidate_b_unmatched_account_table_suppressed",
    } <= issue_codes


def test_singleton_completion_rejects_multi_account_or_ambiguous_families() -> None:
    unnumbered = {
        "account_id": "credit_account_provisional:one",
        "account_type": "revolving_loan_account",
        "account_family_quality": "exact",
        "_printed_ordinal_status": "printed_unreadable",
    }
    numbered = {
        "account_id": "credit_account:revolving_loan_account:2",
        "account_type": "revolving_loan_account",
        "account_family_quality": "exact",
        "_printed_ordinal_status": "printed_unique",
        "category_sequence": 2,
    }
    tables = [
        {
            "_table_observation_id": f"table:{index}",
            "account_type": "revolving_loan_account",
            "source": "native_detail_account_table",
        }
        for index in (1, 2)
    ]

    assert (
        native_extraction._canonical_singleton_account_matches(
            [unnumbered, numbered],
            tables,
            {0: 0, 1: 1},
        )
        == {}
    )

    ambiguous = {
        **unnumbered,
        "account_family_quality": "ambiguous_missing_variant",
    }
    assert (
        native_extraction._canonical_singleton_account_matches(
            [ambiguous],
            [tables[0]],
            {0: 0},
        )
        == {}
    )


def test_singleton_completion_requires_one_native_same_family_owned_table() -> None:
    skeleton = {
        "account_id": "credit_account_provisional:one",
        "account_type": "revolving_loan_account",
        "account_family_quality": "exact",
        "_printed_ordinal_status": "printed_unreadable",
    }
    table = {
        "_table_observation_id": "table:1",
        "account_type": "revolving_loan_account",
        "source": "native_detail_account_table",
    }

    assert native_extraction._canonical_singleton_account_matches(
        [skeleton],
        [table],
        {0: 0},
    ) == {0: 0}
    assert (
        native_extraction._canonical_singleton_account_matches(
            [skeleton],
            [{**table, "source": "candidate_b_account_anchor"}],
            {0: 0},
        )
        == {}
    )
    assert (
        native_extraction._canonical_singleton_account_matches(
            [skeleton],
            [{**table, "account_type": "revolving_loan_subaccount"}],
            {0: 0},
        )
        == {}
    )
    assert (
        native_extraction._canonical_singleton_account_matches(
            [skeleton],
            [table],
            {},
        )
        == {}
    )


def test_account_schema_reconstructs_split_identifier_at_date_boundary() -> None:
    context = SimpleNamespace(
        corrected_evidence_pages=lambda: [
            {
                "page": 4,
                "source_page": 2,
                "lines": [
                    {"text": "循环贷账户（二）"},
                    {"text": "账户 1"},
                    {"text": "管理机构 账户标识 开立日期 信用额度"},
                    {"text": "D10053310H0001"},
                    {"text": "某银行 2022052901021012089466554314 2022.05.29 10000"},
                ],
            }
        ]
    )

    rows = native_extraction._account_anchor_skeletons(context)

    assert rows[0]["account_identifier"] == "D10053310H00012022052901021012089466554314"
    assert rows[0]["account_identifier_source"] == "canonical_anchor_table_row"


def test_account_schema_suppresses_unmatched_table_in_anchored_category(monkeypatch) -> None:
    table = {
        "account_id": "credit_account:credit_card:2",
        "account_type": "credit_card",
        "category_sequence": 2,
        "source_refs": [],
    }
    anchor = {
        "account_id": "credit_account:credit_card:1",
        "account_type": "credit_card",
        "category_sequence": 1,
        "source_refs": [],
    }
    context = SimpleNamespace()
    monkeypatch.setattr(native_extraction, "_extract_table_accounts", lambda _context: ([table], [], []))
    monkeypatch.setattr(native_extraction, "_account_anchor_skeletons", lambda _context: [anchor])

    accounts, _repayments, _events = native_extraction._extract_accounts(context)

    assert accounts == [anchor]
    assert any(
        issue.get("issue_code") == "candidate_b_unmatched_account_table_suppressed"
        for issue in context._personal_detail_extraction_issues
    )


def test_unmatched_account_report_identifies_exact_suppressed_child_population(monkeypatch) -> None:
    table_id = "credit_account_table_observation:unmatched"
    table_ref = {"logical_page": 4, "table_id": "table:unmatched", "bbox": [10, 200, 500, 300]}
    table = {
        "account_id": table_id,
        "_table_observation_id": table_id,
        "account_type": "credit_card",
        "category_sequence": 2,
        "source_refs": [table_ref],
    }
    anchor = {
        "account_id": "credit_account:credit_card:1",
        "account_type": "credit_card",
        "category_sequence": 1,
        "_canonical_segment": {"pages": [{"logical_page": 4, "min_y": 20, "max_y": 100}]},
        "source_refs": [{"logical_page": 4, "bbox": [10, 20, 100, 40]}],
    }
    repayments = [
        {
            "repayment_id": "repayment:2024-01",
            "account_id": table_id,
            "year": 2024,
            "month": 1,
            "source_refs": [{"logical_page": 4, "table_id": "monthly:1", "row": 2}],
        },
        {
            "repayment_id": "repayment:2024-02",
            "account_id": table_id,
            "year": 2024,
            "month": 2,
            "source_refs": [{"logical_page": 4, "table_id": "monthly:1", "row": 3}],
        },
    ]
    events = [
        {
            "account_event_id": "event:special",
            "account_id": table_id,
            "event_type": "special_transaction",
            "source_refs": [{"logical_page": 4, "table_id": "event:1", "row": 5}],
        },
        {
            "account_event_id": "event:latest",
            "account_id": table_id,
            "event_type": "latest_repayment",
            "source_refs": [{"logical_page": 4, "table_id": "event:2", "row": 7}],
        },
    ]
    context = SimpleNamespace()
    monkeypatch.setattr(
        native_extraction,
        "_extract_table_accounts",
        lambda _context: ([table], repayments, events),
    )
    monkeypatch.setattr(native_extraction, "_account_anchor_skeletons", lambda _context: [anchor])

    accounts, filtered_repayments, filtered_events = native_extraction._extract_accounts(context)

    assert accounts == [anchor]
    assert filtered_repayments == []
    assert filtered_events == []
    issue = next(
        issue
        for issue in context._personal_detail_extraction_issues
        if issue["issue_code"] == "candidate_b_unmatched_account_table_suppressed"
    )
    assert issue["target_record_id"] == table_id
    assert issue["observed_value"]["suppressed_child_count"] == 4
    assert issue["observed_value"]["suppressed_child_counts_by_dataset"] == {
        "credit_account_latest_repayments": 1,
        "credit_account_monthly_performance": 2,
        "credit_account_special_transactions": 1,
    }
    assert issue["observed_value"]["affected_child_datasets"] == [
        "credit_account_latest_repayments",
        "credit_account_monthly_performance",
        "credit_account_special_transactions",
    ]
    child_rows = issue["observed_value"]["suppressed_child_observations"]
    assert {row["child_observation_id"] for row in child_rows} == {
        "repayment:2024-01",
        "repayment:2024-02",
        "event:special",
        "event:latest",
    }
    assert all(row["account_observation_id"] == table_id for row in child_rows)
    assert all(row["source_refs"] for row in child_rows)
    assert issue["candidate_value"]["same_category_emitted_account_ids"] == ["credit_account:credit_card:1"]
    assert {ref.get("table_id") for ref in issue["source_refs"]} == {
        "table:unmatched",
        "monthly:1",
        "event:1",
        "event:2",
    }
    assert "related_child_observations_suppressed" in issue["reason_codes"]


def test_account_schema_withholds_unanchored_table_as_reported_observation(monkeypatch) -> None:
    table = {
        "account_id": "credit_account_table_observation:abc",
        "account_type": "credit_card",
        "sequence": 1,
        "category_sequence": 1,
        "management_institution": "样例银行",
        "source_refs": [{"logical_page": 2, "bbox": [10, 100, 500, 200]}],
    }
    context = SimpleNamespace()
    monkeypatch.setattr(native_extraction, "_extract_table_accounts", lambda _context: ([table], [], []))

    accounts, repayments, events = native_extraction._extract_accounts(context)

    assert repayments == []
    assert events == []
    assert accounts == []
    issue = context._personal_detail_extraction_issues[0]
    assert issue["issue_code"] == "candidate_b_account_anchor_population_missing"
    assert issue["target_record_id"] == table["account_id"]
    assert "encounter_order_not_used" in issue["reason_codes"]
    assert "record_not_emitted_due_to_unresolved_account_ownership" in issue["reason_codes"]


def test_account_schema_resolves_shared_revolving_table_signature_from_exact_owned_anchor(
    monkeypatch,
) -> None:
    identifier = "D10053310H00012022052901021012089466554314"
    table_id = "credit_account_table_observation:r2_shared_signature"
    table = {
        "account_id": table_id,
        "_table_observation_id": table_id,
        "account_type": "revolving_loan_subaccount",
        "_table_account_family_basis": "shared_revolving_credit_limit_signature",
        "source": "native_detail_account_table",
        "account_identifier": identifier,
        "balance": "8667",
        "source_refs_by_field": {
            "account_identifier": [
                {
                    "source": "native_detail_table_cell",
                    "geometry_scope": "cell",
                    "logical_page": 12,
                    "source_page": 6,
                    "table_id": "pt_12_2",
                    "row": 1,
                    "column": 1,
                    "bbox": [120, 401, 250, 421],
                    "evidence_ids": ["pt_12_2:r1:c1"],
                    "binding": "canonical_field_slot",
                    "binding_quality": "canonical_header_column",
                    "field_name": "account_identifier",
                }
            ]
        },
        "source_refs": [
            {
                "source": "native_detail_table",
                "logical_page": 12,
                "source_page": 6,
                "table_id": "pt_12_2",
                "bbox": [52, 370, 402, 551],
            }
        ],
    }
    anchor_id = "credit_account_provisional:r2_anchor"
    anchor = {
        "account_id": anchor_id,
        "account_type": "revolving_loan_account",
        "account_family_quality": "exact",
        "_printed_ordinal_status": "printed_unreadable",
        "account_identifier": identifier,
        "page": 12,
        "source_page": 6,
        "bbox": [55, 361, 262, 370],
        "_canonical_segment": {"pages": [{"logical_page": 12, "min_y": 361, "max_y": None}]},
        "source_refs": [
            {
                "source": "candidate_b_account_anchor",
                "logical_page": 12,
                "source_page": 6,
                "bbox": [55, 361, 262, 370],
            }
        ],
    }
    repayment = {
        "account_id": table_id,
        "year": 2022,
        "month": 12,
        "status": "N",
    }
    context = SimpleNamespace()
    monkeypatch.setattr(
        native_extraction,
        "_extract_table_accounts",
        lambda _context: ([table], [repayment], []),
    )
    monkeypatch.setattr(
        native_extraction,
        "_account_anchor_skeletons",
        lambda _context: [anchor],
    )

    accounts, repayments, events = native_extraction._extract_accounts(context)

    assert events == []
    assert len(accounts) == 1
    assert accounts[0]["account_id"] == "credit_account:revolving_loan_account:1"
    assert accounts[0]["account_type"] == "revolving_loan_account"
    assert accounts[0]["category_sequence"] == 1
    assert accounts[0]["account_identifier"] == identifier
    assert accounts[0]["balance"] == "8667"
    assert {ref.get("source") for ref in accounts[0]["source_refs"]} == {
        "candidate_b_account_anchor",
        "native_detail_table",
    }
    assert repayments[0]["account_id"] == accounts[0]["account_id"]
    issue_codes = {issue.get("issue_code") for issue in getattr(context, "_personal_detail_extraction_issues", ())}
    assert not issue_codes.intersection(
        {
            "candidate_b_account_table_missing",
            "candidate_b_account_category_anchor_missing",
            "candidate_b_account_sequence_gap",
            "monthly_population_incomplete_from_account_gap",
        }
    )


def test_account_schema_does_not_join_misclassified_table_by_stream_position(monkeypatch) -> None:
    table = {
        "account_id": "credit_account:revolving_loan_subaccount:1",
        "_table_observation_id": "credit_account:revolving_loan_subaccount:1",
        "account_type": "revolving_loan_subaccount",
        "_table_account_family_basis": "shared_revolving_credit_limit_signature",
        "source": "native_detail_account_table",
        "category_sequence": 1,
        "account_identifier": "D0206000CA202506XZ20011136047",
        "source_refs": [{"logical_page": 4, "bbox": [10, 220, 500, 280]}],
    }
    anchor = {
        "account_id": "credit_account:revolving_loan_account:2",
        "account_type": "revolving_loan_account",
        "account_family_quality": "exact",
        "category_sequence": 2,
        "account_identifier": "D0206000CA202506XZ20011136048",
        "page": 4,
        "bbox": [10, 200, 300, 210],
        "source_refs": [{"logical_page": 4, "bbox": [10, 200, 300, 210]}],
    }
    repayment = {
        "account_id": "credit_account:revolving_loan_subaccount:1",
        "year": 2025,
        "month": 1,
        "status": "N",
    }
    context = SimpleNamespace()
    monkeypatch.setattr(native_extraction, "_extract_table_accounts", lambda _context: ([table], [repayment], []))
    monkeypatch.setattr(native_extraction, "_account_anchor_skeletons", lambda _context: [anchor])

    accounts, repayments, _events = native_extraction._extract_accounts(context)

    assert len(accounts) == 1
    assert accounts[0]["account_id"] == "credit_account:revolving_loan_account:2"
    assert accounts[0]["account_type"] == "revolving_loan_account"
    assert accounts[0]["account_identifier"] == "D0206000CA202506XZ20011136048"
    assert repayments == []
    issue = next(
        issue
        for issue in context._personal_detail_extraction_issues
        if issue["issue_code"] == "candidate_b_account_category_anchor_missing"
    )
    assert issue["target_record_id"] == table["account_id"]
    assert "record_not_emitted_due_to_unresolved_account_ownership" in issue["reason_codes"]


def test_account_stream_match_never_spills_replay_into_older_unmatched_anchor() -> None:
    anchors = [
        {
            "account_id": "account:1",
            "account_type": "credit_card",
            "page": 1,
            "bbox": [0, 100, 100, 110],
        },
        {
            "account_id": "account:2",
            "account_type": "credit_card",
            "page": 2,
            "bbox": [0, 100, 100, 110],
        },
    ]
    tables = [
        {
            "account_id": "table:1",
            "account_type": "credit_card",
            "source_refs": [{"logical_page": 2, "bbox": [0, 120, 100, 180]}],
        },
        {
            "account_id": "table:replay",
            "account_type": "credit_card",
            "source_refs": [{"logical_page": 3, "bbox": [0, 20, 100, 80]}],
        },
    ]

    matches = native_extraction._match_account_table_observations(anchors, tables)

    assert matches == {1: 0}


def test_account_table_match_does_not_use_category_or_encounter_order_without_geometry() -> None:
    anchors = [
        {
            "account_id": "credit_account:credit_card:1",
            "account_type": "credit_card",
            "category_sequence": 1,
            "source_refs": [],
        }
    ]
    tables = [
        {
            "account_id": "credit_account_table_observation:abc",
            "account_type": "credit_card",
            "category_sequence": 1,
            "source_refs": [],
        }
    ]

    assert native_extraction._match_account_table_observations(anchors, tables) == {}


def test_account_table_match_requires_verified_segment_for_later_page() -> None:
    anchor = {
        "account_id": "account:1",
        "account_type": "credit_card",
        "page": 1,
        "bbox": [0, 100, 100, 110],
        "_canonical_segment": {"pages": [{"logical_page": 1, "min_y": 100.0, "max_y": None}]},
    }
    table = {
        "account_id": "table:1",
        "account_type": "credit_card",
        "source_refs": [{"logical_page": 2, "bbox": [0, 20, 100, 80]}],
    }

    assert native_extraction._match_account_table_observations([anchor], [table]) == {}

    anchor["_canonical_segment"]["pages"].append(
        {
            "logical_page": 2,
            "min_y": 0.0,
            "max_y": 90.0,
            "continuation_verified": True,
        }
    )
    assert native_extraction._match_account_table_observations([anchor], [table]) == {0: 0}


def test_credit_agreement_schema_reports_identity_conflict_and_emits_one_row() -> None:
    context = SimpleNamespace()
    records = [
        {
            "credit_line_id": "obsolete:1",
            "account_identifier": "T10151210H0001ABC12345",
            "institution": "机构甲",
            "source_refs": [{"logical_page": 8}],
            "confidence": 0.9,
        },
        {
            "credit_line_id": "obsolete:2",
            "account_identifier": "T10151210H0001ABC12345",
            "institution": "机构乙",
            "source_refs": [{"logical_page": 9}],
            "confidence": 0.8,
        },
    ]

    reconciled = native_extraction.reconcile_candidate_b_credit_lines(context, records)

    assert len(reconciled) == 1
    # Equal-provenance conflicts are not resolved by confidence.  Confidence
    # describes OCR certainty, not which of two incompatible business values
    # belongs in the canonical slot.
    assert reconciled[0]["institution"] is None
    assert len(reconciled[0]["source_refs"]) == 2
    assert context._personal_detail_extraction_issues[0]["issue_code"] == (
        "candidate_b_credit_agreement_observation_conflict"
    )


def test_credit_agreement_does_not_merge_damaged_identifier_from_business_similarity() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    shared = {
        "institution": "示例银行股份有限公司",
        "facility_type": "循环额度",
        "effective_date": "2024-01-01",
        "total_limit": "100000",
    }
    records = [
        {
            "account_identifier": "T10151210H0001ABC12345",
            "_printed_sequence": 1,
            **shared,
            "confidence": 0.9,
        },
        {
            "account_identifier": "T10151210H0001ABC1234?",
            "_printed_sequence": 1,
            **shared,
            "confidence": 0.8,
        },
    ]

    reconciled = native_extraction.reconcile_candidate_b_credit_lines(context, records)

    assert len(reconciled) == 2
    assert {row.get("account_identifier") for row in reconciled} == {
        "T10151210H0001ABC12345",
        None,
    }
    invalid = next(row for row in reconciled if row.get("account_identifier") is None)
    assert invalid["canonical_raw"]["account_identifier"] == "T10151210H0001ABC1234?"
    assert "account_identifier" in invalid["_unresolved_fields"]
    issues = [
        issue
        for issue in context._personal_detail_extraction_issues
        if issue.get("issue_code") == "candidate_b_credit_agreement_identifier_invalid"
    ]
    assert len(issues) == 1
    assert issues[0]["status"] == "requires_review"
    assert issues[0]["target_record_id"] == invalid["credit_line_id"]
    assert issues[0]["field_name"] == "account_identifier"
    assert issues[0]["observed_value"] == "T10151210H0001ABC1234?"
    assert "fuzzy_identifier_repair_forbidden" in issues[0]["reason_codes"]


def test_credit_agreement_same_page_without_shared_card_geometry_is_not_merge_authority() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    shared = {
        "institution": "示例银行股份有限公司",
        "facility_type": "循环额度",
        "effective_date": "2024-01-01",
        "total_limit": "100000",
    }
    first = {
        "account_identifier": "T10151210H0001ABC123456789",
        "_printed_sequence": 4,
        "_canonical_card_key": "credit_agreement:4",
        "_canonical_card_anchor_refs": [
            {
                "source": "native_detail_canonical_anchor_text",
                "binding": "canonical_card_anchor",
            }
        ],
        **shared,
        "source_refs": [{"logical_page": 7}],
    }
    same_page_variant = {
        "account_identifier": "T10151210H0001ABC1234567",
        "_printed_sequence": 4,
        "_canonical_card_key": "credit_agreement:4",
        "_canonical_card_anchor_refs": [
            {
                "source": "native_detail_canonical_anchor_text",
                "binding": "canonical_card_anchor",
            }
        ],
        **shared,
        "source_refs": [{"logical_page": 7}],
    }
    other_page_agreement = {
        "account_identifier": "T10151210H0001ABC1234568",
        "_printed_sequence": 5,
        "_canonical_card_key": "credit_agreement:5",
        "_canonical_card_anchor_refs": [
            {
                "source": "native_detail_canonical_anchor_text",
                "binding": "canonical_card_anchor",
            }
        ],
        **shared,
        "source_refs": [{"logical_page": 8}],
    }

    reconciled = native_extraction.reconcile_candidate_b_credit_lines(
        context,
        [first, same_page_variant, other_page_agreement],
    )

    assert len(reconciled) == 3
    assert {row["account_identifier"] for row in reconciled} == {
        "T10151210H0001ABC123456789",
        "T10151210H0001ABC1234567",
        "T10151210H0001ABC1234568",
    }
    assert [row.get("sequence") for row in reconciled].count(5) == 1
    assert [row.get("sequence") for row in reconciled].count(None) == 2
    assert all("_printed_sequence" not in row for row in reconciled)
    assert any(
        issue.get("issue_code") == "candidate_b_credit_agreement_identity_ambiguous"
        and "canonical_card_anchor_not_cross_plane_unique" in issue.get("reason_codes", ())
        for issue in context._personal_detail_extraction_issues
    )


def test_credit_agreement_prefixes_without_shared_sequence_are_never_collapsed() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    shared = {
        "institution": "示例银行股份有限公司",
        "facility_type": "循环贷款额度",
        "effective_date": "2024-01-01",
        "total_limit": "100000",
        "source_refs": [{"logical_page": 7}],
    }
    records = [
        {"account_identifier": "T10151210H0001ABC1234", "_printed_sequence": 1, **shared},
        {"account_identifier": "T10151210H0001ABC12345", **shared},
    ]

    reconciled = native_extraction.reconcile_candidate_b_credit_lines(context, records)

    assert len(reconciled) == 2
    assert {row["account_identifier"] for row in reconciled} == {
        "T10151210H0001ABC1234",
        "T10151210H0001ABC12345",
    }
    # Shared business fields and an identifier prefix are not structural
    # evidence that two canonical agreements are one record.  Retain both,
    # without falsely flagging otherwise valid agreements as an identity error.
    assert not any(
        issue.get("issue_code") == "candidate_b_credit_agreement_identity_ambiguous"
        for issue in context._personal_detail_extraction_issues
    )


def test_credit_agreement_same_physical_page_and_containment_do_not_prove_identity() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    shared = {
        "effective_date": "2019-05-21",
        "used_limit": "36393",
    }
    damaged = {
        "account_identifier": "RB10711000H0001100000111111111498898000000",
        "institution": None,
        "facility_type": "",
        "total_limit": "364002",
        "_printed_sequence": 7,
        "source_refs": [{"logical_page": 25, "source_page": 13}],
        **shared,
    }
    complete = {
        "account_identifier": "B10711000H000110000011111111149889800",
        "institution": "中国光大银行股份有限公司",
        "facility_type": "信用卡共享额度",
        "total_limit": "36400",
        "credit_limit": "--",
        "limit_identifier": "--",
        "currency": "CNY",
        "due_date": "2029-05-21",
        "_printed_sequence": 7,
        "source_refs": [{"logical_page": 26, "source_page": 13}],
        "source_refs_by_field": {
            field_name: [
                {
                    "logical_page": 26,
                    "source_page": 13,
                    "geometry_scope": "cell",
                    "binding": "canonical_label_slot",
                }
            ]
            for field_name in (
                "institution",
                "facility_type",
                "effective_date",
                "due_date",
                "total_limit",
                "credit_limit",
                "used_limit",
                "limit_identifier",
                "currency",
            )
        },
        "_field_binding_quality": {
            field_name: "canonical_cell_slot"
            for field_name in (
                "institution",
                "facility_type",
                "effective_date",
                "due_date",
                "total_limit",
                "credit_limit",
                "used_limit",
                "limit_identifier",
                "currency",
            )
        },
        **shared,
    }

    reconciled = native_extraction.reconcile_candidate_b_credit_lines(context, [damaged, complete])

    assert len(reconciled) == 2
    assert {row.get("account_identifier") for row in reconciled} == {
        damaged["account_identifier"],
        complete["account_identifier"],
    }
    assert all(row.get("sequence") is None for row in reconciled)
    assert any(
        issue.get("issue_code") == "candidate_b_credit_agreement_identity_ambiguous"
        and "exact_card_identity_not_proven" in issue.get("reason_codes", ())
        for issue in context._personal_detail_extraction_issues
    )


def test_credit_agreement_leading_suffix_shape_does_not_prove_same_card() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    shared = {
        "_printed_sequence": 7,
        "effective_date": "2019-05-21",
        "used_limit": "36393",
        "source_refs": [{"logical_page": 25, "source_page": 13}],
    }
    damaged = {
        "account_identifier": "RB10711000H0001100000111111111498898000000",
        **shared,
    }
    canonical = {
        "account_identifier": "B10711000H0001100000111111111498898000000",
        "institution": "中国光大银行股份有限公司",
        "facility_type": "信用卡共享额度",
        **shared,
    }

    reconciled = native_extraction.reconcile_candidate_b_credit_lines(
        context,
        [damaged, canonical],
    )

    assert len(reconciled) == 2
    assert {row["account_identifier"] for row in reconciled} == {
        damaged["account_identifier"],
        canonical["account_identifier"],
    }
    assert all(row.get("sequence") is None for row in reconciled)
    assert any(
        issue.get("issue_code") == "candidate_b_credit_agreement_identity_ambiguous"
        and "fuzzy_identifier_merge_forbidden" in issue.get("reason_codes", ())
        for issue in context._personal_detail_extraction_issues
    )


def test_credit_agreement_exact_ordinal_and_verified_continuation_cannot_choose_an_identifier() -> None:
    context = SimpleNamespace(
        _personal_detail_extraction_issues=[],
        tables_continue=lambda left, right: (left, right) == ("agreement:left", "agreement:right"),
    )
    shared = {
        "institution": "示例银行股份有限公司",
        "facility_type": "循环额度",
        "effective_date": "2024-01-01",
        "used_limit": "50000",
    }
    weak = {
        "account_identifier": "T10151210H0001ABC12340",
        "_printed_sequence": 4,
        "source_refs": [{"logical_page": 7, "table_id": "agreement:left"}],
        **shared,
    }
    strong = {
        "account_identifier": "T10151210H0001ABC12345",
        "_printed_sequence": 4,
        "source_refs": [
            {
                "logical_page": 8,
                "source_page": 8,
                "table_id": "agreement:right",
            }
        ],
        "source_refs_by_field": {
            "account_identifier": [
                {
                    "source": "personal_detail_corrected_page_cell",
                    "logical_page": 8,
                    "source_page": 8,
                    "table_id": "agreement:right",
                    "bbox": [20.0, 100.0, 220.0, 120.0],
                    "geometry_scope": "cell",
                    "binding": "canonical_label_slot",
                    "field_name": "account_identifier",
                    "evidence_ids": ["agreement:right:identifier"],
                }
            ]
        },
        "_field_binding_quality": {"account_identifier": "canonical_cell_slot"},
        **shared,
    }

    reconciled = native_extraction.reconcile_candidate_b_credit_lines(context, [weak, strong])

    assert len(reconciled) == 2
    assert {row["account_identifier"] for row in reconciled} == {
        weak["account_identifier"],
        strong["account_identifier"],
    }
    assert all(row.get("sequence") is None for row in reconciled)
    assert any(
        issue.get("issue_code") == "candidate_b_credit_agreement_identity_ambiguous"
        and "fuzzy_identifier_merge_forbidden" in issue.get("reason_codes", ())
        for issue in context._personal_detail_extraction_issues
    )


def test_credit_agreement_exact_cross_plane_card_anchor_rejects_distinct_identifiers() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    shared = {
        "_printed_sequence": 2,
        "_canonical_card_key": "credit_agreement:2",
        "institution": "中国光大银行股份有限公司",
        "effective_date": "2019-12-01",
        "used_limit": "0",
    }
    native = {
        "account_identifier": "B10711000H0001100001111112446567900000",
        "_canonical_card_anchor_refs": [
            {
                "source": "native_detail_canonical_anchor_text",
                "binding": "canonical_card_anchor",
                "logical_page": 15,
                "source_page": 15,
                "bbox": [20.0, 80.0, 220.0, 95.0],
                "evidence_ids": ["native:anchor"],
            }
        ],
        "source_refs": [
            {
                "source": "native_detail_tolerant_table",
                "logical_page": 15,
                "source_page": 15,
                "table_id": "agreement:native",
            }
        ],
        "source_refs_by_field": {
            "account_identifier": [
                {
                    "source": "native_detail_tolerant_table_cell",
                    "logical_page": 15,
                    "source_page": 15,
                    "table_id": "agreement:native",
                    "row": 1,
                    "column": 1,
                    "bbox": [20.0, 100.0, 220.0, 120.0],
                    "geometry_scope": "cell",
                    "binding": "label_column",
                    "field_name": "account_identifier",
                    "evidence_ids": ["native:identifier"],
                }
            ]
        },
        "_field_binding_quality": {"account_identifier": "native_label_column"},
        **shared,
    }
    corrected = {
        "account_identifier": "B10711000H0001100000111111112446567900000",
        "facility_type": "信用卡共享额度",
        "total_limit": "0",
        "currency": "CNY",
        "validity_type": "perpetual",
        "_canonical_card_anchor_refs": [
            {
                "source": "personal_detail_corrected_page_cell",
                "binding": "canonical_card_anchor",
                "logical_page": 15,
                "source_page": 15,
                "bbox": [20.0, 80.0, 220.0, 95.0],
                "evidence_ids": ["corrected:anchor"],
            }
        ],
        "source_refs": [
            {
                "source": "personal_detail_corrected_page",
                "logical_page": 15,
                "source_page": 15,
            }
        ],
        "source_refs_by_field": {
            "account_identifier": [
                {
                    "source": "personal_detail_corrected_page_cell",
                    "logical_page": 15,
                    "source_page": 15,
                    "bbox": [20.0, 100.0, 220.0, 120.0],
                    "geometry_scope": "cell",
                    "binding": "canonical_label_slot",
                    "field_name": "account_identifier",
                    "evidence_ids": ["corrected:identifier"],
                }
            ]
        },
        "_field_binding_quality": {"account_identifier": "canonical_cell_slot"},
        **shared,
    }

    reconciled = native_extraction.reconcile_candidate_b_credit_lines(
        context,
        [native, corrected],
    )

    assert len(reconciled) == 2
    assert {row["account_identifier"] for row in reconciled} == {
        native["account_identifier"],
        corrected["account_identifier"],
    }
    assert all(row.get("sequence") is None for row in reconciled)
    assert any(
        issue.get("issue_code") == "candidate_b_credit_agreement_identity_ambiguous"
        and "canonical_card_anchor_not_cross_plane_unique" in issue.get("reason_codes", ())
        for issue in context._personal_detail_extraction_issues
    )


def test_credit_agreement_cross_plane_anchor_does_not_override_business_conflict() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    shared = {
        "_printed_sequence": 2,
        "_canonical_card_key": "credit_agreement:2",
        "institution": "示例银行股份有限公司",
        "facility_type": "信用卡共享额度",
        "effective_date": "2019-12-01",
    }
    native = {
        "account_identifier": "B10711000H0001100001111112446567900000",
        "used_limit": "0",
        "_canonical_card_anchor_refs": [
            {
                "source": "native_detail_canonical_anchor_text",
                "binding": "canonical_card_anchor",
            }
        ],
        **shared,
    }
    corrected = {
        "account_identifier": native["account_identifier"],
        "used_limit": "999",
        "_canonical_card_anchor_refs": [
            {
                "source": "personal_detail_corrected_page_cell",
                "binding": "canonical_card_anchor",
            }
        ],
        **shared,
    }

    reconciled = native_extraction.reconcile_candidate_b_credit_lines(
        context,
        [native, corrected],
    )

    assert len(reconciled) == 1
    assert reconciled[0]["sequence"] == 2
    assert reconciled[0]["used_limit"] is None
    assert any(
        issue.get("issue_code") == "candidate_b_credit_agreement_observation_conflict"
        and "used_limit" in issue.get("observed_value", {}).get("conflicting_fields", ())
        for issue in context._personal_detail_extraction_issues
    )


def test_credit_agreement_business_similarity_alone_does_not_raise_identity_issue() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    shared = {
        "institution": "示例银行股份有限公司",
        "facility_type": "信用卡共享额度",
        "effective_date": "2019-12-01",
        "used_limit": "0",
    }

    reconciled = native_extraction.reconcile_candidate_b_credit_lines(
        context,
        [
            {
                "account_identifier": "B10711000H0001100001111112446567900000",
                "_printed_sequence": 1,
                "source_refs": [{"table_id": "agreement-list"}],
                **shared,
            },
            {
                "account_identifier": "B10711000H0001100000111111112446567900000",
                "_printed_sequence": 2,
                "source_refs": [{"table_id": "agreement-list"}],
                **shared,
            },
        ],
    )

    assert len(reconciled) == 2
    assert [row.get("sequence") for row in reconciled] == [1, 2]
    assert not any(
        issue.get("issue_code") == "candidate_b_credit_agreement_identity_ambiguous"
        for issue in context._personal_detail_extraction_issues
    )


def test_credit_agreement_reports_required_fields_missing_after_final_merge() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    missing_fields = {
        "institution",
        "facility_type",
        "due_date",
        "total_limit",
        "credit_limit",
        "used_limit",
        "limit_identifier",
        "currency",
    }
    exact_field_refs = {
        field_name: [
            {
                "source": "personal_detail_corrected_page_cell",
                "logical_page": 8,
                "source_page": 4,
                "bbox": [1.0, 2.0, 30.0, 40.0],
                "geometry_scope": "cell",
                "field_name": field_name,
                "binding": "canonical_label_slot",
                "evidence_ids": [f"ocr:agreement:{field_name}"],
            }
        ]
        for field_name in missing_fields
    }
    rows = native_extraction.reconcile_candidate_b_credit_lines(
        context,
        [
            {
                "credit_line_id": "credit-line:missing-fields",
                "account_identifier": "B10512900H00010010011135264974289",
                "institution": "",
                "facility_type": None,
                "effective_date": "2020-03-29",
                "_printed_sequence": 1,
                "source_refs": [{"logical_page": 8, "bbox": [1, 2, 30, 40]}],
                "source_refs_by_field": exact_field_refs,
            }
        ],
    )

    assert len(rows) == 1
    final_id = rows[0]["credit_line_id"]
    issues = [
        issue
        for issue in context._personal_detail_extraction_issues
        if issue.get("issue_code") == "candidate_b_credit_agreement_required_field_unresolved"
    ]
    assert {issue["field_name"] for issue in issues} == missing_fields
    assert {issue["target_record_id"] for issue in issues} == {final_id}
    assert all(issue["source_refs"][0]["logical_page"] == 8 for issue in issues)
    assert all(issue["source_refs"][0]["field_name"] == issue["field_name"] for issue in issues)
    assert all(
        issue["reason_codes"]
        == [
            "required_field_missing",
            "canonical_credit_agreement_field_unresolved",
            "field_slot_not_safely_bound",
            "preserved_unknown_value",
        ]
        for issue in issues
    )


def test_credit_agreement_withholds_non_unique_printed_sequences() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    rows = native_extraction.reconcile_candidate_b_credit_lines(
        context,
        [
            {
                "account_identifier": "T10151210H0001ABC12345",
                "_printed_sequence": 3,
                "source_refs": [{"logical_page": 7}],
            },
            {
                "account_identifier": "B10151210H0001XYZ12345",
                "_printed_sequence": 3,
                "source_refs": [{"logical_page": 8}],
            },
        ],
    )

    assert all("sequence" not in row for row in rows)
    assert (
        sum(
            issue["issue_code"] == "candidate_b_credit_agreement_sequence_unresolved"
            for issue in context._personal_detail_extraction_issues
        )
        == 2
    )


def test_credit_agreement_withholds_concatenated_limit_identifiers(monkeypatch) -> None:
    from docmirror.plugins.credit_report.personal_detail_scanned.native_parser import (
        PBOCPersonalDetailNativeParser,
    )

    candidate = SimpleNamespace(
        fields={
            "授信协议标识": "T10151210H0001ABC12345",
            "授信限额编号": "B10411000H0001799190000103302585" * 3,
        },
        source_refs=[],
        confidence=0.9,
    )
    monkeypatch.setattr(
        PBOCPersonalDetailNativeParser,
        "records",
        lambda _self, dataset_name: [candidate] if dataset_name == "credit_lines" else [],
    )
    context = SimpleNamespace(_personal_detail_extraction_issues=[])

    rows = native_extraction._extract_credit_lines(context)

    assert rows[0]["limit_identifier"] is None
    assert context._personal_detail_extraction_issues[0]["issue_code"] == (
        "candidate_b_credit_limit_identifier_unresolved"
    )
    assert not context._personal_detail_extraction_issues[0].get("source_refs")


def test_credit_agreement_dash_glyphs_are_explicit_absence_not_failures(monkeypatch) -> None:
    from docmirror.plugins.credit_report.personal_detail_scanned.native_parser import (
        PBOCPersonalDetailNativeParser,
    )

    dash_fields = {
        "授信额度": "total_limit",
        "授信限额": "credit_limit",
        "已用额度": "used_limit",
        "授信限额编号": "limit_identifier",
        "到期日期": "due_date",
    }
    candidate = SimpleNamespace(
        fields={
            "授信协议标识": "T10151210H0001ABC12345",
            "授信额度": "－",
            "授信限额": "——",
            "已用额度": "--",
            "授信限额编号": "--",
            "到期日期": "—",
        },
        source_refs=[],
        source_refs_by_field={
            label: [
                {
                    "source": "personal_detail_corrected_page_cell",
                    "logical_page": 8,
                    "source_page": 4,
                    "bbox": [10.0, 20.0, 80.0, 36.0],
                    "geometry_scope": "cell",
                    "field_name": field_name,
                    "binding": "canonical_label_slot",
                    "evidence_ids": [f"ocr:agreement:{field_name}:dash"],
                }
            ]
            for label, field_name in dash_fields.items()
        },
        binding_quality_by_field={},
        unresolved_labels=frozenset(),
        observed_labels=frozenset(),
        confidence=0.9,
    )
    monkeypatch.setattr(
        PBOCPersonalDetailNativeParser,
        "records",
        lambda _self, dataset_name: [candidate] if dataset_name == "credit_lines" else [],
    )
    context = SimpleNamespace(_personal_detail_extraction_issues=[])

    extracted = native_extraction._extract_credit_lines(context)
    rows = native_extraction.reconcile_candidate_b_credit_lines(context, extracted)

    assert len(rows) == 1
    assert rows[0]["total_limit"] is None
    assert rows[0]["credit_limit"] is None
    assert rows[0]["used_limit"] is None
    assert rows[0]["limit_identifier"] is None
    assert rows[0]["validity_type"] == "unknown"
    assert {
        "total_limit",
        "credit_limit",
        "used_limit",
        "limit_identifier",
        "due_date",
    } <= set(rows[0]["_source_absent_fields"])
    assert rows[0]["canonical_raw"] == {
        "total_limit": "－",
        "credit_limit": "——",
        "used_limit": "--",
        "limit_identifier": "--",
        "due_date": "—",
    }
    assert not any(
        issue.get("field_name") in {"total_limit", "credit_limit", "used_limit", "limit_identifier", "due_date"}
        for issue in context._personal_detail_extraction_issues
    )


def test_inquiry_boundary_and_normalization_differences_require_exact_institution_identity() -> None:
    assert native_extraction._inquiry_business_equivalent(
        {"inquiry_date": "2024.01.02", "institution": " 示例银行股份有限公司 ", "reason": "货款审批"},
        {"inquiry_date": "2024-01-02", "institution": "示例银行股份有限公司", "reason": "贷款审批"},
    )
    assert not native_extraction._inquiry_business_equivalent(
        {"inquiry_date": "2024-01-02", "institution": "安 本人", "reason": "本人查询"},
        {
            "inquiry_date": "2024-01-02",
            "institution": "本人 安",
            "reason": "本人查询(自助查询机)",
        },
    )
    assert not native_extraction._inquiry_business_equivalent(
        {"inquiry_date": "2024-01-02", "institution": "美 兴业银行股份有限公司 你", "reason": "贷后管理"},
        {"inquiry_date": "2024-01-02", "institution": "兴业银行股份有限公司 你 美", "reason": "贷后管理"},
    )
    assert not native_extraction._inquiry_business_equivalent(
        {
            "inquiry_date": "2024-01-02",
            "institution": "深圳前海微众银行股份有限公司 法人代表、负责人、高管等",
            "reason": "资信审查",
        },
        {
            "inquiry_date": "2024-01-02",
            "institution": "深圳前海微众银行股份有限公司",
            "reason": "法人代表、负责人、高管等资信审查",
        },
    )
    assert native_extraction._normalize_inquiry_reason("货后管理") == "贷后管理"
    assert native_extraction._normalize_inquiry_reason("某货后管理服务有限公司") == "某货后管理服务有限公司"
    assert native_extraction._bounded_canonical_inquiry_reason("公积金提取复核") == "公积金提取复核"
    assert native_extraction._bounded_canonical_inquiry_reason("本人查询（临柜）") == "本人查询"
    assert native_extraction._bounded_canonical_inquiry_reason("本人查询（未来渠道）") is None
    assert native_extraction._bounded_canonical_inquiry_reason("未来用途审查") is None
    assert native_extraction._bounded_canonical_inquiry_reason("其他") == "其他"
    assert native_extraction._bounded_canonical_inquiry_reason("X贷后管理Y") is None
    assert native_extraction._bounded_canonical_inquiry_reason("???贷后管理!!!") is None
    assert native_extraction._longest_inquiry_reason_suffix("示例征信服务其他") == ""
    assert native_extraction._longest_inquiry_reason_suffix("示例银行 公积金提取复核") == "公积金提取复核"


def _sealed_public_table(table_id: str, rows: list[list[str]]) -> SimpleNamespace:
    column_count = max((len(row) for row in rows), default=0)
    bboxes = [
        [[column * 100, row_index * 20, (column + 1) * 100, (row_index + 1) * 20] for column in range(len(row))]
        for row_index, row in enumerate(rows)
    ]
    return SimpleNamespace(
        table_id=table_id,
        bbox=[0, 0, max(100, column_count * 100), max(20, len(rows) * 20)],
        metadata={
            "raw_rows": rows,
            "canonical_template_id": "public_information",
            "geometry": {
                "row_bands": [{"index": row, "y0": row * 20, "y1": (row + 1) * 20} for row in range(len(rows))],
                "col_bands": [
                    {"index": column, "x0": column * 100, "x1": (column + 1) * 100} for column in range(column_count)
                ],
                "cell_bboxes": bboxes,
                "cell_geometry_status": [["exact"] * len(row) for row in rows],
                "cell_evidence_ids": [
                    [[f"{table_id}:{row_index}:{column}"] for column in range(len(row))]
                    for row_index, row in enumerate(rows)
                ],
            },
        },
    )


def test_public_record_projection_keeps_canonical_authorities_and_typed_fields() -> None:
    def table(table_id: str, rows: list[list[str]]) -> SimpleNamespace:
        return _sealed_public_table(table_id, rows)

    page = SimpleNamespace(
        page_number=13,
        source_page_number=13,
        canonical_template_id="public_information",
        tables=[
            table(
                "tax",
                [
                    ["编号", "主管税务机关", "欠税总额", "欠税统计日期"],
                    ["1", "某市税务局", "500", "2023.03.17"],
                ],
            ),
            table(
                "penalty",
                [
                    ["编号", "处罚机构", "处罚内容", "处罚金额", "生效日期", "截止日期", "行政复议结果"],
                    ["1", "某市监管局", "警告", "400", "2021.08", "2024.07", "--"],
                ],
            ),
            table(
                "award",
                [
                    ["编号", "奖励机构", "奖励内容", "生效日期", "截止日期"],
                    ["1", "某市总工会", "先进工作者", "2023.02", "2033.06"],
                ],
            ),
        ],
    )

    public_rows = native_extraction._extract_public_records(SimpleNamespace(pages=[page]))
    projected = project_personal_detail_datasets({"public_records": public_rows})

    tax = projected["tax_arrears_records"][0]
    assert tax["tax_authority"] == "某市税务局"
    assert tax["arrears_amount"] == 500
    assert tax["reporting_amount_currency"] == "CNY"
    penalty = projected["administrative_penalty_records"][0]
    assert penalty["authority"] == "某市监管局"
    assert penalty["penalty_content"] == "警告"
    assert penalty["administrative_review_result"] is None
    award = projected["administrative_award_records"][0]
    assert award["authority"] == "某市总工会"
    assert award["award_content"] == "先进工作者"


def test_housing_fund_blocks_use_canonical_boundaries_across_pages() -> None:
    layouts = {layout["name"]: layout for layout in native_extraction._PUBLIC_CANONICAL_LAYOUTS}
    base = layouts["housing_fund_base"]
    provider = layouts["housing_fund_provider"]

    def header(layout: dict[str, object]) -> list[str]:
        aliases = layout["aliases"]
        fields = layout["fields"]
        assert isinstance(aliases, dict)
        assert isinstance(fields, dict)
        return [aliases[role][0] for role in fields]

    def table(table_id: str, rows: list[list[str]]) -> SimpleNamespace:
        return _sealed_public_table(table_id, rows)

    page_1 = SimpleNamespace(
        page_number=13,
        source_page_number=13,
        canonical_template_id="public_information",
        tables=[
            table(
                "housing-base-1",
                [
                    header(base),
                    ["", "", "", "", "", "", "", ""],
                    ["Fuzhou", "2018.09.03", "2018.09", "2023.08", "active", "906", "6%", "6%"],
                ],
            )
        ],
    )
    page_2 = SimpleNamespace(
        page_number=14,
        source_page_number=14,
        canonical_template_id="public_information",
        tables=[
            table(
                "housing-continuation-and-second-record",
                [
                    header(provider),
                    ["", ""],
                    ["示例科技有限公司", "2023.08"],
                    header(base),
                    ["Xiamen", "2015.06.25", "2015.06", "2018.08", "closed", "1023", "8%", "8%"],
                    header(provider),
                    ["示例服务有限公司", "2023.08"],
                ],
            )
        ],
    )
    context = SimpleNamespace(
        pages=[page_1, page_2],
        _personal_detail_extraction_issues=[],
    )

    public_rows = native_extraction._extract_public_records(context)
    projected = project_personal_detail_datasets({"public_records": public_rows})
    housing = projected["housing_fund_records"]

    assert [row["sequence"] for row in housing] == [1, 2]
    assert [row["employer"] for row in housing] == ["示例科技有限公司", "示例服务有限公司"]
    assert [row["contribution_location"] for row in housing] == ["Fuzhou", "Xiamen"]
    assert len({row["public_record_id"] for row in housing}) == 2
    assert not context._personal_detail_extraction_issues


def test_housing_fund_continuation_without_start_is_reported_and_not_invented() -> None:
    provider = next(
        layout for layout in native_extraction._PUBLIC_CANONICAL_LAYOUTS if layout["name"] == "housing_fund_provider"
    )
    header = [provider["aliases"][role][0] for role in provider["fields"]]
    page = SimpleNamespace(
        page_number=14,
        source_page_number=14,
        canonical_template_id="public_information",
        tables=[
            _sealed_public_table(
                "orphan-housing-provider",
                [header, ["Employer A", "2023.08"]],
            )
        ],
    )
    context = SimpleNamespace(pages=[page], _personal_detail_extraction_issues=[])

    assert native_extraction._extract_public_records(context) == []
    assert [issue["issue_code"] for issue in context._personal_detail_extraction_issues] == [
        "candidate_b_public_record_continuation_unowned"
    ]


def test_housing_fund_nonadjacent_continuation_is_not_attached() -> None:
    layouts = {layout["name"]: layout for layout in native_extraction._PUBLIC_CANONICAL_LAYOUTS}
    base = layouts["housing_fund_base"]
    provider = layouts["housing_fund_provider"]

    def header(layout: dict[str, object]) -> list[str]:
        aliases = layout["aliases"]
        fields = layout["fields"]
        assert isinstance(aliases, dict)
        assert isinstance(fields, dict)
        return [aliases[role][0] for role in fields]

    def page(number: int, table_id: str, rows: list[list[str]]) -> SimpleNamespace:
        return SimpleNamespace(
            page_number=number,
            source_page_number=number,
            canonical_template_id="public_information",
            tables=[_sealed_public_table(table_id, rows)],
        )

    context = SimpleNamespace(
        pages=[
            page(
                13,
                "housing-base",
                [
                    header(base),
                    ["Fuzhou", "2018.09.03", "2018.09", "2023.08", "active", "906", "6%", "6%"],
                ],
            ),
            page(15, "late-provider", [header(provider), ["Employer A", "2023.08"]]),
        ],
        reading_order_by_logical={13: 13, 15: 15},
        _personal_detail_extraction_issues=[],
    )

    rows = native_extraction._extract_public_records(context)
    housing = [row for row in rows if row["record_type"] == "housing_fund"]
    issues = context._personal_detail_extraction_issues

    assert len(housing) == 1
    assert "employer" not in housing[0]
    assert [issue["issue_code"] for issue in issues] == [
        "candidate_b_public_record_continuation_missing",
        "candidate_b_public_record_continuation_unowned",
    ]
    assert issues[0]["candidate_value"]["missing_fields"] == [
        "employer",
        "information_updated_month",
    ]


def test_projection_keeps_schema_values_separate_from_raw_evidence() -> None:
    rows = project_personal_detail_datasets(
        {
            "credit_accounts": [
                {
                    "account_id": "credit_account:credit_card:1",
                    "account_type": "credit_card",
                    "canonical_raw": {"management_institution": "OCR evidence"},
                }
            ]
        }
    )["credit_accounts"]

    assert rows[0]["normalized"]["account_id"] == "credit_account:credit_card:1"
    assert rows[0]["normalized"]["pboc_account_type_code"] == "R3"
    assert rows[0]["canonical_raw"] == {"management_institution": "OCR evidence"}


def test_monthly_link_recovers_grid_geometry_from_cells() -> None:
    accounts = [
        {"account_id": "account:1", "page": 4, "bbox": [10, 20, 100, 100], "sequence": 1},
        {"account_id": "account:2", "page": 4, "bbox": [10, 400, 100, 450], "sequence": 2},
    ]
    repayments = [{"grid_id": "grid:1", "year": 2024, "month": 1, "status": "N"}]
    grids = [
        {
            "grid_id": "grid:1",
            "page": 4,
            "cells": [[{"bbox": [10, 200, 100, 220], "text": "N"}]],
        }
    ]

    linked = link_candidate_b_repayments(repayments, accounts, grids)

    assert linked[0]["account_id"] == "account:1"


def test_monthly_link_withholds_owner_when_grid_geometry_is_missing() -> None:
    context = SimpleNamespace()
    accounts = [
        {"account_id": "account:1", "page": 4, "bbox": [10, 20, 100, 100], "sequence": 1},
        {"account_id": "account:2", "page": 4, "bbox": [10, 400, 100, 450], "sequence": 2},
    ]

    linked = link_candidate_b_repayments(
        [{"grid_id": "grid:1", "year": 2024, "month": 1, "status": "N"}],
        accounts,
        [{"grid_id": "grid:1", "page": 4, "cells": []}],
        issue_context=context,
    )

    assert linked == []
    assert context._personal_detail_extraction_issues[0]["issue_code"] == ("candidate_b_monthly_grid_owner_unresolved")
    assert "target_record_id" not in context._personal_detail_extraction_issues[0]
    assert context._personal_detail_extraction_issues[0]["observed_value"]["observed_candidate_count"] == 1


def test_monthly_cross_page_predecessor_is_not_used_as_an_owner() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    repayment_id = "grid:continued:2024-01"
    linked = link_candidate_b_repayments(
        [
            {
                "repayment_id": repayment_id,
                "grid_id": "grid:continued",
                "year": 2024,
                "month": 1,
                "status": "N",
                "overdue_amount": "0",
                "source_cell_refs": [{"grid_id": "grid:continued", "logical_page": 6}],
            }
        ],
        [
            {
                "account_id": "account:1",
                "page": 4,
                "bbox": [10, 20, 100, 100],
                "sequence": 1,
            }
        ],
        [{"grid_id": "grid:continued", "page": 6, "bbox": [10, 100, 200, 200]}],
        reading_order_by_logical={4: 4, 6: 6},
        issue_context=context,
    )

    assert linked == []
    issue = context._personal_detail_extraction_issues[0]
    assert issue["issue_code"] == "candidate_b_monthly_grid_owner_unresolved"
    assert "target_record_id" not in issue
    assert issue["observed_value"]["grid_id"] == "grid:continued"
    assert issue["observed_value"]["observed_candidate_months"] == ["2024-01"]


def test_monthly_verified_cross_page_account_segment_is_silent() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    linked = link_candidate_b_repayments(
        [
            {
                "repayment_id": "grid:continued:2024-01",
                "grid_id": "grid:continued",
                "year": 2024,
                "month": 1,
                "status": "N",
                "overdue_amount": "0",
            }
        ],
        [
            {
                "account_id": "account:1",
                "page": 4,
                "bbox": [10, 20, 100, 100],
                "sequence": 1,
                "_canonical_segment": {
                    "pages": [
                        {"logical_page": 4, "min_y": 20.0, "max_y": None},
                        {
                            "logical_page": 6,
                            "min_y": 0.0,
                            "max_y": 300.0,
                            "continuation_verified": True,
                        },
                    ]
                },
            }
        ],
        [{"grid_id": "grid:continued", "page": 6, "bbox": [10, 100, 200, 200]}],
        issue_context=context,
    )

    assert linked[0]["account_id"] == "account:1"
    assert "account_linkage" not in linked[0].get("audit", {})
    assert context._personal_detail_extraction_issues == []


def _owned_table_continuation_record() -> dict:
    table_ref = {
        "source": "native_detail_table",
        "logical_page": 17,
        "source_page": 9,
        "table_id": "pt_17_0",
        "geometry_scope": "table",
        "coordinate_system": "pdf_points_top_left",
        "bbox": [52.0, 55.0, 402.0, 252.0],
    }
    return {
        "account_id": "credit_account:credit_card:3",
        "_canonical_segment": {
            "ownership_basis": "printed_anchor_to_next_anchor",
            "anchor_logical_page": 20,
            "anchor_bbox": [47.0, 514.5, 280.5, 527.0],
            "pages": [
                {
                    "logical_page": 20,
                    "min_y": 514.5,
                    "max_y": None,
                    "continuation_verified": False,
                }
            ],
            "cross_page_continuation_verified": False,
        },
        "source_refs": [table_ref],
        "_owned_account_table_continuation_refs": [
            {
                **table_ref,
                "binding": "account_table_continuation",
                "binding_quality": "entity_table_continuation",
            }
        ],
    }


def test_exact_owned_table_continuation_extends_nonmonotonic_account_segment() -> None:
    context = SimpleNamespace(
        reading_order_by_logical={20: 18, 17: 19},
        reading_order_resolution={"resolved": True, "authoritative": True},
        _personal_detail_extraction_issues=[],
    )
    account = _owned_table_continuation_record()

    native_extraction._extend_account_segment_with_owned_table_continuations(
        context,
        account,
    )

    assert "_owned_account_table_continuation_refs" not in account
    assert account["_canonical_segment"]["cross_page_continuation_verified"] is True
    assert account["_canonical_segment"]["pages"] == [
        {
            "logical_page": 20,
            "min_y": 514.5,
            "max_y": None,
            "continuation_verified": False,
        },
        {
            "logical_page": 17,
            "min_y": 55.0,
            "max_y": 252.0,
            "continuation_verified": True,
            "ownership_basis": "exact_owned_account_table_continuation",
            "source_table_id": "pt_17_0",
            "binding_quality": "entity_table_continuation",
        },
    ]
    linked = link_candidate_b_repayments(
        [
            {
                "repayment_id": "mg_p17_repayment_0:2019-04",
                "grid_id": "mg_p17_repayment_0",
                "year": 2019,
                "month": 4,
                "status": "N",
                "overdue_amount": "0",
            }
        ],
        [account],
        [
            {
                "grid_id": "mg_p17_repayment_0",
                "page": 17,
                "bbox": [54.0, 57.0, 400.0, 250.0],
            }
        ],
        issue_context=context,
    )
    assert [row["account_id"] for row in linked] == ["credit_account:credit_card:3"]
    assert "account_identifier" not in linked[0]
    assert context._personal_detail_extraction_issues == []


@pytest.mark.parametrize(
    "failure_mode",
    (
        "missing_native_table_ref",
        "unauthoritative_order",
        "reversed_order",
        "malformed_marker",
        "unmarked_table",
    ),
)
def test_owned_table_segment_extension_fails_closed(failure_mode: str) -> None:
    context = SimpleNamespace(
        reading_order_by_logical={20: 18, 17: 19},
        reading_order_resolution={"resolved": True, "authoritative": True},
    )
    account = _owned_table_continuation_record()
    if failure_mode == "missing_native_table_ref":
        account["source_refs"] = []
    elif failure_mode == "unauthoritative_order":
        context.reading_order_resolution["authoritative"] = False
    elif failure_mode == "reversed_order":
        context.reading_order_by_logical = {20: 19, 17: 18}
    elif failure_mode == "malformed_marker":
        account["_owned_account_table_continuation_refs"][0]["binding_quality"] = "nearby_table"
    elif failure_mode == "unmarked_table":
        account.pop("_owned_account_table_continuation_refs")

    native_extraction._extend_account_segment_with_owned_table_continuations(
        context,
        account,
    )

    assert "_owned_account_table_continuation_refs" not in account
    assert [page["logical_page"] for page in account["_canonical_segment"]["pages"]] == [20]
    assert account["_canonical_segment"]["cross_page_continuation_verified"] is False


def test_monthly_ambiguous_account_segments_withhold_orphan_rows() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    accounts = [
        {
            "account_id": account_id,
            "_canonical_segment": {"pages": [{"logical_page": 4, "min_y": 20.0, "max_y": 300.0}]},
        }
        for account_id in ("account:1", "account:2")
    ]

    linked = link_candidate_b_repayments(
        [{"repayment_id": "grid:1:2024-01", "grid_id": "grid:1", "year": 2024, "month": 1, "status": "N"}],
        accounts,
        [{"grid_id": "grid:1", "page": 4, "bbox": [10, 100, 200, 200]}],
        issue_context=context,
    )

    assert linked == []
    issue = context._personal_detail_extraction_issues[0]
    assert issue["issue_code"] == "candidate_b_monthly_grid_owner_unresolved"
    assert issue["observed_value"]["linkage_basis"] == "ambiguous_account_segments"
    assert issue["observed_value"]["candidate_account_ids"] == ["account:1", "account:2"]


def test_monthly_explicit_cross_page_owner_requires_segment_proof() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    linked = link_candidate_b_repayments(
        [
            {
                "repayment_id": "grid:continued:2024-01",
                "grid_id": "grid:continued",
                "account_id": "account:1",
                "year": 2024,
                "month": 1,
                "status": "N",
                "overdue_amount": "0",
            }
        ],
        [{"account_id": "account:1", "page": 4, "bbox": [10, 20, 100, 100], "sequence": 1}],
        [{"grid_id": "grid:continued", "page": 6, "bbox": [10, 100, 200, 200]}],
        issue_context=context,
    )

    assert linked == []
    assert context._personal_detail_extraction_issues[0]["issue_code"] == ("candidate_b_monthly_grid_owner_unresolved")


def test_native_table_source_ref_propagates_only_declared_coordinate_system() -> None:
    page = SimpleNamespace(page_number=12, source_page_number=6)
    declared = SimpleNamespace(
        table_id="pt_12_0",
        bbox=[52.0, 55.0, 402.0, 252.0],
        metadata={"geometry": {"coordinate_system": "pdf_points_top_left"}},
    )
    undeclared = SimpleNamespace(
        table_id="pt_12_1",
        bbox=[52.0, 274.0, 402.0, 500.0],
        metadata={"geometry": {}},
    )

    assert native_extraction._source_ref(page, declared)["coordinate_system"] == ("pdf_points_top_left")
    assert "coordinate_system" not in native_extraction._source_ref(page, undeclared)


def _exact_source_table_monthly_link_fixture() -> tuple[list[dict], list[dict], list[dict]]:
    grid_id = "mg_p12_repayment_0"
    table_id = "pt_12_0"

    def source_ref(*, month: int, field_name: str, row: int, y0: float) -> dict:
        x0 = 80.0 + 26.5 * (month - 1)
        return {
            "page": 12,
            "logical_page": 12,
            "geometry_scope": "cell",
            "coordinate_system": "pdf_points_top_left",
            "grid_id": grid_id,
            "row": row,
            "col": month,
            "field_name": field_name,
            "bbox": [x0, y0, x0 + 26.5, y0 + 12.0],
            "geometry_provenance": {
                "source": "source_table_geometry",
                "selection_basis": "source_table_year_plus_twelve_ownership",
                "reason": "exact_source_table_month_lattice_calibration",
                "table_id": table_id,
                "logical_page": 12,
                "coordinate_system": "pdf_points_top_left",
                "calibrated_from_source_table_geometry": True,
                "active_cell_geometry_exact": True,
                "value_inputs_used": False,
            },
        }

    accounts = [
        {
            "account_id": "credit_account:revolving_loan_subaccount:1",
            "account_identifier": "B10611000H0001811132129417001",
            "_canonical_segment": {
                "ownership_basis": "printed_anchor_to_next_anchor",
                "anchor_logical_page": 11,
                "anchor_bbox": [47.0, 514.5, 280.5, 527.0],
                "pages": [{"logical_page": 11, "min_y": 514.5, "max_y": None}],
            },
            "source_refs": [
                {
                    "source": "native_detail_table",
                    "logical_page": 12,
                    "source_page": 6,
                    "table_id": table_id,
                    "geometry_scope": "table",
                    "coordinate_system": "pdf_points_top_left",
                    "bbox": [52.0, 55.0, 402.0, 252.0],
                }
            ],
        }
    ]
    month_positions = [(value // 12, value % 12 + 1) for value in range(2022 * 12 + 5, 2024 * 12 + 3)]
    repayments = [
        {
            "repayment_id": f"{grid_id}:{year:04d}-{month:02d}",
            "grid_id": grid_id,
            "year": year,
            "month": month,
            "performance_month": f"{year:04d}-{month:02d}",
            "status": "N",
            "overdue_amount": "0",
            "source_cell_refs": [
                source_ref(
                    month=month,
                    field_name="status",
                    row=2 + 2 * (year - 2022),
                    y0=100.0 + 45.0 * (year - 2022),
                ),
                source_ref(
                    month=month,
                    field_name="overdue_amount",
                    row=3 + 2 * (year - 2022),
                    y0=112.0 + 45.0 * (year - 2022),
                ),
            ],
        }
        for year, month in month_positions
    ]
    grids = [
        {
            "grid_id": grid_id,
            "page": 12,
            "bbox": [52.0, 55.5, 401.5, 251.0],
            "coordinate_system": "pdf_points_top_left",
            "col_bands": [
                {
                    "index": month,
                    "header": str(month),
                    "role": "month",
                    "bbox": [
                        80.0 + 26.5 * (month - 1),
                        80.0,
                        80.0 + 26.5 * month,
                        90.0,
                    ],
                    "geometry_status": "exact",
                    "geometry_source": "source_table_geometry",
                }
                for month in range(1, 13)
            ],
            "audit": {
                "date_range": {
                    "start_year": 2022,
                    "start_month": 6,
                    "end_year": 2024,
                    "end_month": 3,
                },
                "visual_month_geometry_by_page": {
                    "12": {
                        "source": "source_table_geometry",
                        "selection_basis": "source_table_year_plus_twelve_ownership",
                        "reason": "exact_source_table_month_lattice_calibration",
                        "table_id": table_id,
                        "logical_page": 12,
                        "coordinate_system": "pdf_points_top_left",
                        "calibrated_from_source_table_geometry": True,
                        "active_cell_geometry_exact": True,
                        "value_inputs_used": False,
                    }
                },
            },
        }
    ]
    return accounts, repayments, grids


def test_monthly_exact_source_table_owner_recovers_headerless_continuation() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    accounts, repayments, grids = _exact_source_table_monthly_link_fixture()

    linked = link_candidate_b_repayments(
        repayments,
        accounts,
        grids,
        issue_context=context,
    )

    assert len(linked) == 22
    assert linked[0]["repayment_id"] == "mg_p12_repayment_0:2022-06"
    assert linked[-1]["repayment_id"] == "mg_p12_repayment_0:2024-03"
    assert {row["account_id"] for row in linked} == {"credit_account:revolving_loan_subaccount:1"}
    assert {row["account_identifier"] for row in linked} == {"B10611000H0001811132129417001"}
    assert context._personal_detail_extraction_issues == []


def test_monthly_exact_source_table_owner_accepts_absent_year_month_aliases() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    accounts, repayments, grids = _exact_source_table_monthly_link_fixture()
    for repayment in repayments:
        repayment.pop("year")
        repayment.pop("month")

    linked = link_candidate_b_repayments(
        repayments,
        accounts,
        grids,
        issue_context=context,
    )

    assert len(linked) == 22
    assert context._personal_detail_extraction_issues == []


def _monthly_owner_issue_basis(context: SimpleNamespace) -> str:
    owner_issue = next(
        issue
        for issue in context._personal_detail_extraction_issues
        if issue["issue_code"] == "candidate_b_monthly_grid_owner_unresolved"
    )
    return str(owner_issue["observed_value"]["linkage_basis"])


def test_monthly_source_table_owner_rejects_unowned_partial_account() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    accounts, repayments, grids = _exact_source_table_monthly_link_fixture()
    accounts[0].pop("_canonical_segment")
    accounts[0]["_ownership_status"] = "printed_category_anchor_missing"

    linked = link_candidate_b_repayments(
        repayments,
        accounts,
        grids,
        issue_context=context,
    )

    assert linked == []
    assert _monthly_owner_issue_basis(context) == "source_table_account_owner_not_observed"


def test_monthly_source_table_owner_withholds_noncanonical_owner_identifier() -> None:
    canonical = "B10611000H0001811132129417001"
    for invalid_identifier in (None, "NOT_CANONICAL", f" {canonical} "):
        context = SimpleNamespace(_personal_detail_extraction_issues=[])
        accounts, repayments, grids = _exact_source_table_monthly_link_fixture()
        if invalid_identifier is None:
            accounts[0].pop("account_identifier")
        else:
            accounts[0]["account_identifier"] = invalid_identifier

        linked = link_candidate_b_repayments(
            repayments,
            accounts,
            grids,
            issue_context=context,
        )

        assert len(linked) == len(repayments)
        assert {row["account_id"] for row in linked} == {"credit_account:revolving_loan_subaccount:1"}
        assert all("account_identifier" not in row for row in linked)
        assert context._personal_detail_extraction_issues == []


def test_monthly_source_table_owner_requires_native_int_anchor_page_identity() -> None:
    for field in ("anchor_logical_page", "segment_logical_page"):
        for invalid_page in (11.5, "11", True):
            context = SimpleNamespace(_personal_detail_extraction_issues=[])
            accounts, repayments, grids = _exact_source_table_monthly_link_fixture()
            segment = accounts[0]["_canonical_segment"]
            if field == "anchor_logical_page":
                segment["anchor_logical_page"] = invalid_page
            else:
                segment["pages"][0]["logical_page"] = invalid_page

            linked = link_candidate_b_repayments(
                repayments,
                accounts,
                grids,
                issue_context=context,
            )

            assert linked == []
            assert _monthly_owner_issue_basis(context) == ("source_table_account_owner_not_observed")


def test_monthly_source_table_owner_fails_closed_for_duplicate_account_owners() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    accounts, repayments, grids = _exact_source_table_monthly_link_fixture()
    duplicate = {
        **accounts[0],
        "account_id": "credit_account:revolving_loan_subaccount:2",
        "account_identifier": "B10111000H0001140201019800104930030100000661000020000001",
        "source_refs": [dict(accounts[0]["source_refs"][0])],
    }
    accounts.append(duplicate)

    linked = link_candidate_b_repayments(
        repayments,
        accounts,
        grids,
        issue_context=context,
    )

    assert linked == []
    owner_issue = next(
        issue
        for issue in context._personal_detail_extraction_issues
        if issue["issue_code"] == "candidate_b_monthly_grid_owner_unresolved"
    )
    assert owner_issue["observed_value"]["linkage_basis"] == ("ambiguous_source_table_account_owners")


def test_monthly_source_table_owner_rejects_conflicting_account_table_boxes() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    accounts, repayments, grids = _exact_source_table_monthly_link_fixture()
    accounts[0]["source_refs"].append(
        {
            **accounts[0]["source_refs"][0],
            "bbox": [200.0, 55.0, 550.0, 252.0],
        }
    )

    linked = link_candidate_b_repayments(
        repayments,
        accounts,
        grids,
        issue_context=context,
    )

    assert linked == []
    assert _monthly_owner_issue_basis(context) == "source_table_account_geometry_conflict"


def test_monthly_source_table_owner_fails_closed_without_exact_cell_provenance() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    accounts, repayments, grids = _exact_source_table_monthly_link_fixture()
    repayments[0]["source_cell_refs"][0]["geometry_provenance"].pop("table_id")

    linked = link_candidate_b_repayments(
        repayments,
        accounts,
        grids,
        issue_context=context,
    )

    assert linked == []
    owner_issue = next(
        issue
        for issue in context._personal_detail_extraction_issues
        if issue["issue_code"] == "candidate_b_monthly_grid_owner_unresolved"
    )
    assert owner_issue["observed_value"]["linkage_basis"] == ("source_table_exact_provenance_unresolved")


def test_monthly_source_table_owner_rejects_conflicting_table_id_aliases() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    accounts, repayments, grids = _exact_source_table_monthly_link_fixture()
    repayments[0]["source_cell_refs"][0]["table_id"] = "pt_12_conflict"

    linked = link_candidate_b_repayments(
        repayments,
        accounts,
        grids,
        issue_context=context,
    )

    assert linked == []
    assert _monthly_owner_issue_basis(context) == "source_table_exact_provenance_unresolved"


def test_monthly_source_table_owner_never_uses_value_based_provenance() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    accounts, repayments, grids = _exact_source_table_monthly_link_fixture()
    repayments[0]["source_cell_refs"][0]["geometry_provenance"]["value_inputs_used"] = True

    linked = link_candidate_b_repayments(
        repayments,
        accounts,
        grids,
        issue_context=context,
    )

    assert linked == []
    owner_issue = next(
        issue
        for issue in context._personal_detail_extraction_issues
        if issue["issue_code"] == "candidate_b_monthly_grid_owner_unresolved"
    )
    assert owner_issue["observed_value"]["linkage_basis"] == ("source_table_exact_provenance_unresolved")


def test_monthly_source_table_owner_rejects_image_pixel_cell_coordinates() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    accounts, repayments, grids = _exact_source_table_monthly_link_fixture()
    source_ref = repayments[0]["source_cell_refs"][0]
    source_ref["coordinate_system"] = "image_pixels"
    source_ref["geometry_provenance"]["coordinate_system"] = "image_pixels"

    linked = link_candidate_b_repayments(
        repayments,
        accounts,
        grids,
        issue_context=context,
    )

    assert linked == []
    assert _monthly_owner_issue_basis(context) == "source_table_exact_provenance_unresolved"


def test_monthly_source_table_owner_rejects_non_pdf_grid_coordinates() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    accounts, repayments, grids = _exact_source_table_monthly_link_fixture()
    grids[0]["coordinate_system"] = "image_pixels"

    linked = link_candidate_b_repayments(
        repayments,
        accounts,
        grids,
        issue_context=context,
    )

    assert linked == []
    assert _monthly_owner_issue_basis(context) == ("source_table_grid_geometry_unresolved")


def test_monthly_source_table_owner_rejects_non_pdf_account_table_coordinates() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    accounts, repayments, grids = _exact_source_table_monthly_link_fixture()
    accounts[0]["source_refs"][0]["coordinate_system"] = "image_pixels"

    linked = link_candidate_b_repayments(
        repayments,
        accounts,
        grids,
        issue_context=context,
    )

    assert linked == []
    assert _monthly_owner_issue_basis(context) == "source_table_account_geometry_conflict"


def test_monthly_source_table_owner_fails_closed_for_page_mismatch() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    accounts, repayments, grids = _exact_source_table_monthly_link_fixture()
    source_ref = repayments[0]["source_cell_refs"][0]
    source_ref["page"] = 13
    source_ref["logical_page"] = 13
    source_ref["geometry_provenance"]["logical_page"] = 13

    linked = link_candidate_b_repayments(
        repayments,
        accounts,
        grids,
        issue_context=context,
    )

    assert linked == []
    owner_issue = next(
        issue
        for issue in context._personal_detail_extraction_issues
        if issue["issue_code"] == "candidate_b_monthly_grid_owner_unresolved"
    )
    assert owner_issue["observed_value"]["linkage_basis"] == "source_table_page_conflict"


def test_monthly_source_table_owner_rejects_conflicting_page_aliases() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    accounts, repayments, grids = _exact_source_table_monthly_link_fixture()
    repayments[0]["source_cell_refs"][0]["page"] = 13

    linked = link_candidate_b_repayments(
        repayments,
        accounts,
        grids,
        issue_context=context,
    )

    assert linked == []
    assert _monthly_owner_issue_basis(context) == "source_table_exact_provenance_unresolved"


def test_monthly_source_table_owner_requires_native_int_cell_page_aliases() -> None:
    for invalid_page in (12.5, "12", True):
        context = SimpleNamespace(_personal_detail_extraction_issues=[])
        accounts, repayments, grids = _exact_source_table_monthly_link_fixture()
        repayments[0]["source_cell_refs"][0]["page"] = invalid_page

        linked = link_candidate_b_repayments(
            repayments,
            accounts,
            grids,
            issue_context=context,
        )

        assert linked == []
        assert _monthly_owner_issue_basis(context) == ("source_table_exact_provenance_unresolved")


def test_monthly_source_table_owner_requires_native_int_grid_page() -> None:
    for invalid_page in (12.5, "12", True):
        context = SimpleNamespace(_personal_detail_extraction_issues=[])
        accounts, repayments, grids = _exact_source_table_monthly_link_fixture()
        grids[0]["page"] = invalid_page

        linked = link_candidate_b_repayments(
            repayments,
            accounts,
            grids,
            issue_context=context,
        )

        assert linked == []
        assert _monthly_owner_issue_basis(context) == "source_table_page_conflict"


def test_monthly_source_table_owner_requires_native_int_account_table_page() -> None:
    for invalid_page in (12.5, "12", True):
        context = SimpleNamespace(_personal_detail_extraction_issues=[])
        accounts, repayments, grids = _exact_source_table_monthly_link_fixture()
        accounts[0]["source_refs"][0]["logical_page"] = invalid_page

        linked = link_candidate_b_repayments(
            repayments,
            accounts,
            grids,
            issue_context=context,
        )

        assert linked == []
        assert _monthly_owner_issue_basis(context) == ("source_table_account_owner_not_observed")


def test_monthly_source_table_owner_rejects_conflicting_grid_id_aliases() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    accounts, repayments, grids = _exact_source_table_monthly_link_fixture()
    repayments[0]["source_cell_refs"][0]["grid_id"] = "mg_p12_repayment_conflict"

    linked = link_candidate_b_repayments(
        repayments,
        accounts,
        grids,
        issue_context=context,
    )

    assert linked == []
    assert _monthly_owner_issue_basis(context) == "source_table_grid_identity_conflict"


def test_monthly_source_table_owner_rejects_duplicate_grid_ids() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    accounts, repayments, grids = _exact_source_table_monthly_link_fixture()
    grids.append(dict(grids[0]))

    linked = link_candidate_b_repayments(
        repayments,
        accounts,
        grids,
        issue_context=context,
    )

    assert linked == []
    assert _monthly_owner_issue_basis(context) == "duplicate_grid_id"


def test_monthly_source_table_owner_fails_closed_for_shifted_month_ref() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    accounts, repayments, grids = _exact_source_table_monthly_link_fixture()
    repayments[0]["source_cell_refs"][0]["col"] = 7

    linked = link_candidate_b_repayments(
        repayments,
        accounts,
        grids,
        issue_context=context,
    )

    assert linked == []
    owner_issue = next(
        issue
        for issue in context._personal_detail_extraction_issues
        if issue["issue_code"] == "candidate_b_monthly_grid_owner_unresolved"
    )
    assert owner_issue["observed_value"]["linkage_basis"] == ("source_table_grid_month_contract_unresolved")


def test_monthly_source_table_owner_requires_native_int_month_column() -> None:
    for invalid_column in (6.0, "6", True):
        context = SimpleNamespace(_personal_detail_extraction_issues=[])
        accounts, repayments, grids = _exact_source_table_monthly_link_fixture()
        repayments[0]["source_cell_refs"][0]["col"] = invalid_column

        linked = link_candidate_b_repayments(
            repayments,
            accounts,
            grids,
            issue_context=context,
        )

        assert linked == []
        assert _monthly_owner_issue_basis(context) == ("source_table_grid_month_contract_unresolved")


def test_monthly_source_table_owner_requires_exact_performance_month() -> None:
    for invalid_performance_month in (None, "2022-6", "2022-07", "2022-06 "):
        context = SimpleNamespace(_personal_detail_extraction_issues=[])
        accounts, repayments, grids = _exact_source_table_monthly_link_fixture()
        repayments[0]["performance_month"] = invalid_performance_month

        linked = link_candidate_b_repayments(
            repayments,
            accounts,
            grids,
            issue_context=context,
        )

        assert linked == []
        assert _monthly_owner_issue_basis(context) == ("source_table_grid_month_contract_unresolved")


def test_monthly_source_table_owner_requires_native_int_grid_date_range() -> None:
    for invalid_start_year in ("2022", 2022.5, True):
        context = SimpleNamespace(_personal_detail_extraction_issues=[])
        accounts, repayments, grids = _exact_source_table_monthly_link_fixture()
        grids[0]["audit"]["date_range"]["start_year"] = invalid_start_year

        linked = link_candidate_b_repayments(
            repayments,
            accounts,
            grids,
            issue_context=context,
        )

        assert linked == []
        assert _monthly_owner_issue_basis(context) == ("source_table_grid_month_contract_unresolved")

    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    accounts, repayments, grids = _exact_source_table_monthly_link_fixture()
    date_range = grids[0]["audit"]["date_range"]
    for field_name, value in list(date_range.items()):
        date_range[field_name] = str(value)

    linked = link_candidate_b_repayments(
        repayments,
        accounts,
        grids,
        issue_context=context,
    )

    assert linked == []
    assert _monthly_owner_issue_basis(context) == ("source_table_grid_month_contract_unresolved")


def test_monthly_source_table_owner_rejects_coerced_calendar_aliases() -> None:
    for field, invalid_value in (
        ("year", 2022.5),
        ("year", "2022"),
        ("year", True),
        ("month", 6.0),
        ("month", "6"),
        ("month", True),
    ):
        context = SimpleNamespace(_personal_detail_extraction_issues=[])
        accounts, repayments, grids = _exact_source_table_monthly_link_fixture()
        repayments[0][field] = invalid_value

        linked = link_candidate_b_repayments(
            repayments,
            accounts,
            grids,
            issue_context=context,
        )

        assert linked == []
        assert _monthly_owner_issue_basis(context) == ("source_table_grid_month_contract_unresolved")


def test_monthly_complete_native_lattice_supersedes_unordered_derived_bands() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    accounts, repayments, grids = _exact_source_table_monthly_link_fixture()
    month_bands = grids[0]["col_bands"]
    month_bands[5]["bbox"], month_bands[6]["bbox"] = (
        month_bands[6]["bbox"],
        month_bands[5]["bbox"],
    )

    linked = link_candidate_b_repayments(
        repayments,
        accounts,
        grids,
        issue_context=context,
    )

    assert len(linked) == len(repayments)
    assert {row["account_id"] for row in linked} == {"credit_account:revolving_loan_subaccount:1"}
    assert all(
        row["_account_month_identity_proof"]["owner_basis"] == "exact_source_table_account_owner" for row in linked
    )


def test_monthly_multi_year_native_lattices_share_one_physical_month_axis(
    monkeypatch,
) -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    accounts, repayments, grids = _exact_source_table_monthly_link_fixture()
    month_bands = grids[0]["col_bands"]
    month_bands[5]["bbox"], month_bands[6]["bbox"] = (
        month_bands[6]["bbox"],
        month_bands[5]["bbox"],
    )
    for source_ref in repayments[0]["source_cell_refs"]:
        source_ref["geometry_provenance"]["value_inputs_used"] = True

    month_boxes = tuple(
        (
            80.0 + 26.5 * (month - 1),
            100.0,
            80.0 + 26.5 * month,
            112.0,
        )
        for month in range(1, 13)
    )

    def native_lattice(
        _context,
        _source_ref,
        *,
        expected_year: int,
        expected_month: int,
    ):
        del expected_month
        status_row = 2 + 2 * (expected_year - 2022)
        return SimpleNamespace(
            logical_page=12,
            table_id="pt_12_0",
            year_anchor_row_index=status_row,
            status_row_index=status_row,
            amount_row_index=status_row + 1,
            month_bboxes=month_boxes,
        )

    monkeypatch.setattr(
        relations_mod,
        "_native_month_lattice_from_exact_ref",
        native_lattice,
    )

    linked = link_candidate_b_repayments(
        repayments,
        accounts,
        grids,
        issue_context=context,
    )

    assert len(linked) == len(repayments)
    assert {row["account_id"] for row in linked} == {"credit_account:revolving_loan_subaccount:1"}
    assert context._personal_detail_extraction_issues == []


def test_monthly_sparse_rule_derived_refs_are_reproved_before_table_ownership(
    monkeypatch,
) -> None:
    """Model the p17 continuation: sparse cells come from exact physical rules."""

    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    accounts, repayments, grids = _exact_source_table_monthly_link_fixture()
    month_bands = grids[0]["col_bands"]
    month_bands[5]["bbox"], month_bands[6]["bbox"] = (
        month_bands[6]["bbox"],
        month_bands[5]["bbox"],
    )
    witness_month_by_year = {2022: 6, 2023: 1, 2024: 1}
    for repayment in repayments:
        year = repayment["year"]
        month = repayment["month"]
        for source_ref in repayment["source_cell_refs"]:
            if source_ref["field_name"] == "status" and month == witness_month_by_year[year]:
                provenance = source_ref["geometry_provenance"]
                status_row = 2 + 2 * (year - 2022)
                provenance.update(
                    {
                        "active_cell_geometry_exact": False,
                        "active_cell_rule_derived_count": 2,
                        "rule_count": 14,
                        "column_count": 13,
                        "month_column_count": 12,
                        "status_row_index": status_row,
                        "amount_row_index": status_row + 1,
                        "year_anchor_row_index": status_row,
                    }
                )
                continue
            source_ref.clear()
            source_ref.update(
                {
                    "page": 12,
                    "logical_page": 12,
                    "grid_id": "mg_p12_repayment_0",
                    "row": 0,
                    "col": month,
                    "field_name": "status",
                    "geometry_scope": "logical_page",
                    "geometry_status": "unresolved",
                }
            )

    month_boxes = tuple(
        (
            80.0 + 26.5 * (month - 1),
            100.0,
            80.0 + 26.5 * month,
            112.0,
        )
        for month in range(1, 13)
    )

    def native_lattice(
        _context,
        source_ref,
        *,
        expected_year: int,
        expected_month: int,
    ):
        assert source_ref["col"] == expected_month
        status_row = 2 + 2 * (expected_year - 2022)
        return SimpleNamespace(
            logical_page=12,
            table_id="pt_12_0",
            year_anchor_row_index=status_row,
            status_row_index=status_row,
            amount_row_index=status_row + 1,
            month_bboxes=month_boxes,
        )

    monkeypatch.setattr(
        relations_mod,
        "_native_month_lattice_from_exact_ref",
        native_lattice,
    )

    linked = link_candidate_b_repayments(
        repayments,
        accounts,
        grids,
        issue_context=context,
    )

    assert len(linked) == len(repayments)
    assert {row["account_id"] for row in linked} == {"credit_account:revolving_loan_subaccount:1"}
    assert context._personal_detail_extraction_issues == []


def test_monthly_damaged_bands_require_complete_exact_native_lattice() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    accounts, repayments, grids = _exact_source_table_monthly_link_fixture()
    month_bands = grids[0]["col_bands"]
    month_bands[5]["bbox"], month_bands[6]["bbox"] = (
        month_bands[6]["bbox"],
        month_bands[5]["bbox"],
    )
    for source_ref in repayments[0]["source_cell_refs"]:
        source_ref["geometry_provenance"]["value_inputs_used"] = True

    linked = link_candidate_b_repayments(
        repayments,
        accounts,
        grids,
        issue_context=context,
    )

    assert linked == []
    assert _monthly_owner_issue_basis(context) == ("source_table_grid_month_geometry_unresolved")


def test_monthly_source_table_owner_rejects_reversed_cell_x_bands() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    accounts, repayments, grids = _exact_source_table_monthly_link_fixture()
    for repayment in repayments:
        if repayment["month"] not in {6, 7}:
            continue
        x0 = 350.0 if repayment["month"] == 6 else 80.0
        for source_ref in repayment["source_cell_refs"]:
            source_ref["bbox"][0] = x0
            source_ref["bbox"][2] = x0 + 26.5

    linked = link_candidate_b_repayments(
        repayments,
        accounts,
        grids,
        issue_context=context,
    )

    assert linked == []
    assert _monthly_owner_issue_basis(context) == "source_table_grid_geometry_conflict"


def test_monthly_complete_exact_cells_supersede_grid_bbox_extent() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    accounts, repayments, grids = _exact_source_table_monthly_link_fixture()
    accounts[0]["source_refs"][0]["bbox"] = [40.0, 40.0, 410.0, 300.0]
    repayments[0]["source_cell_refs"][0]["bbox"][1:] = [251.5, 239.0, 263.5]

    linked = link_candidate_b_repayments(
        repayments,
        accounts,
        grids,
        issue_context=context,
    )

    assert len(linked) == len(repayments)
    assert {row["account_id"] for row in linked} == {"credit_account:revolving_loan_subaccount:1"}
    assert context._personal_detail_extraction_issues == []


def test_monthly_source_table_owner_rejects_cell_outside_owner_table() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    accounts, repayments, grids = _exact_source_table_monthly_link_fixture()
    repayments[0]["source_cell_refs"][0]["bbox"][1:] = [501.5, 239.0, 513.5]

    linked = link_candidate_b_repayments(
        repayments,
        accounts,
        grids,
        issue_context=context,
    )

    assert linked == []
    assert _monthly_owner_issue_basis(context) == "source_table_grid_geometry_conflict"


def test_monthly_source_table_owner_rejects_nonfinite_cell_bbox() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    accounts, repayments, grids = _exact_source_table_monthly_link_fixture()
    repayments[0]["source_cell_refs"][0]["bbox"][2] = float("inf")

    linked = link_candidate_b_repayments(
        repayments,
        accounts,
        grids,
        issue_context=context,
    )

    assert linked == []
    assert _monthly_owner_issue_basis(context) == "source_table_exact_provenance_unresolved"


def test_monthly_source_table_owner_fails_closed_for_grid_table_mismatch() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    accounts, repayments, grids = _exact_source_table_monthly_link_fixture()
    grids[0]["audit"]["visual_month_geometry_by_page"]["12"]["table_id"] = "pt_12_1"

    linked = link_candidate_b_repayments(
        repayments,
        accounts,
        grids,
        issue_context=context,
    )

    assert linked == []
    owner_issue = next(
        issue
        for issue in context._personal_detail_extraction_issues
        if issue["issue_code"] == "candidate_b_monthly_grid_owner_unresolved"
    )
    assert owner_issue["observed_value"]["linkage_basis"] == ("source_table_grid_provenance_conflict")


def test_monthly_source_table_owner_fails_closed_for_identifier_mismatch() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    accounts, repayments, grids = _exact_source_table_monthly_link_fixture()
    repayments[0]["account_identifier"] = "B10611000H0001811132137961001"

    linked = link_candidate_b_repayments(
        repayments,
        accounts,
        grids,
        issue_context=context,
    )

    assert linked == []
    owner_issue = next(
        issue
        for issue in context._personal_detail_extraction_issues
        if issue["issue_code"] == "candidate_b_monthly_grid_owner_unresolved"
    )
    assert owner_issue["observed_value"]["linkage_basis"] == ("account_identifier_source_table_conflict")


def test_monthly_equivalent_duplicate_replays_merge_without_false_review() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    accounts = [{"account_id": "account:1", "page": 4, "bbox": [10, 20, 100, 100], "sequence": 1}]
    grid = {"grid_id": "grid:1", "page": 4, "bbox": [10, 120, 100, 220]}
    rows = [
        {
            "repayment_id": "grid:1:2024-01",
            "grid_id": "grid:1",
            "year": 2024,
            "month": 1,
            "status": "N",
            "overdue_amount": "0.00",
            "confidence": 0.8,
            "source_cell_refs": [{"grid_id": "grid:1", "cell": "a"}],
        },
        {
            "repayment_id": "grid:1:2024-01",
            "grid_id": "grid:1",
            "year": 2024,
            "month": 1,
            "status": "N",
            "overdue_amount": "0",
            "confidence": 0.9,
            "source_cell_refs": [{"grid_id": "grid:1", "cell": "b"}],
        },
    ]

    linked = link_candidate_b_repayments(rows, accounts, [grid], issue_context=context)

    assert len(linked) == 1
    assert linked[0]["confidence"] == 0.9
    assert linked[0]["source_cell_refs"] == [
        {"grid_id": "grid:1", "cell": "b"},
        {"grid_id": "grid:1", "cell": "a"},
    ]
    assert "duplicate_month_candidates" not in linked[0].get("audit", {})
    assert not any(
        issue["issue_code"] == "candidate_b_monthly_duplicate_conflict"
        for issue in context._personal_detail_extraction_issues
    )


def test_monthly_same_grid_replay_stays_clean_with_open_account_gap() -> None:
    gap = {
        "issue_code": "candidate_b_account_sequence_gap",
        "status": "requires_review",
        "observed_value": {"account_type": "credit_card"},
        "candidate_value": {"missing_category_sequences": [2]},
    }
    context = SimpleNamespace(_personal_detail_extraction_issues=[gap])
    accounts = [
        {
            "account_id": "account:1",
            "_canonical_segment": {"pages": [{"logical_page": 4, "min_y": 20.0, "max_y": 300.0}]},
        }
    ]
    grid = {"grid_id": "grid:1", "page": 4, "bbox": [10, 120, 100, 220]}
    rows = [
        {
            "repayment_id": "grid:1:2024-01",
            "grid_id": "grid:1",
            "year": 2024,
            "month": 1,
            "status": "N",
            "overdue_amount": "0",
            "source_cell_refs": [{"grid_id": "grid:1", "cell": cell}],
        }
        for cell in ("a", "b")
    ]

    linked = link_candidate_b_repayments(rows, accounts, [grid], issue_context=context)

    assert len(linked) == 1
    assert {ref["cell"] for ref in linked[0]["source_cell_refs"]} == {"a", "b"}
    assert not any(
        issue.get("issue_code") == "monthly_linkage_collision_from_account_gap"
        for issue in context._personal_detail_extraction_issues
    )


def test_monthly_conflicting_duplicate_is_selected_and_reported() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    accounts = [{"account_id": "account:1", "page": 4, "bbox": [10, 20, 100, 100], "sequence": 1}]
    grid = {"grid_id": "grid:1", "page": 4, "bbox": [10, 120, 100, 220]}
    rows = [
        {
            "repayment_id": "grid:1:2024-01",
            "grid_id": "grid:1",
            "year": 2024,
            "month": 1,
            "status": "unknown",
            "overdue_amount": None,
            "confidence": 0.4,
        },
        {
            "repayment_id": "grid:1:2024-01",
            "grid_id": "grid:1",
            "year": 2024,
            "month": 1,
            "status": "N",
            "overdue_amount": "0",
            "confidence": 0.9,
        },
    ]

    linked = link_candidate_b_repayments(rows, accounts, [grid], issue_context=context)

    assert len(linked) == 1
    assert linked[0]["status"] == "N"
    assert "extraction_status" not in linked[0]
    assert not any(
        issue["issue_code"]
        in {
            "candidate_b_monthly_duplicate_conflict",
            "candidate_b_monthly_status_grid_unresolved",
        }
        for issue in context._personal_detail_extraction_issues
    )


def test_monthly_unresolved_status_survives_linking_for_final_correction() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    accounts = [{"account_id": "account:1", "page": 4, "bbox": [10, 20, 100, 100], "sequence": 1}]
    grid = {
        "grid_id": "grid:1",
        "page": 4,
        "bbox": [10, 120, 100, 220],
        "audit": {"date_range": {"start_year": 2024, "start_month": 1, "end_year": 2024, "end_month": 1}},
    }

    linked = link_candidate_b_repayments(
        [{"repayment_id": "grid:1:2024-01", "grid_id": "grid:1", "year": 2024, "month": 1, "status": "unknown"}],
        accounts,
        [grid],
        issue_context=context,
    )

    assert len(linked) == 1
    assert linked[0]["account_id"] == "account:1"
    assert linked[0]["status"] == "unknown"
    assert not any(
        issue["issue_code"] == "candidate_b_monthly_status_grid_unresolved"
        for issue in context._personal_detail_extraction_issues
    )


def test_inquiry_schema_withholds_boundary_row_without_unique_sequence() -> None:
    context = SimpleNamespace(
        corrected_evidence_pages=lambda: [
            {
                "page": 8,
                "source_page": 4,
                "canonical_template_id": "annotations_and_inquiries",
                "lines": [
                    {"text": "1 2024.01.02 银行甲 贷款审批", "bbox": [50, 10, 390, 18]},
                    {"text": "2024,01.01", "bbox": [110, 30, 170, 38]},
                    {"text": "银行乙", "bbox": [200, 31, 280, 39]},
                    {"text": "贷后管理", "bbox": [345, 29, 390, 37]},
                ],
            }
        ]
    )

    rows = native_extraction._canonical_inquiry_line_rows(context)

    assert [row["sequence"] for row in rows] == [1]
    unresolved = context._personal_detail_extraction_issues[0]
    assert unresolved["issue_code"] == "candidate_b_inquiry_boundary_sequence_unresolved"
    assert unresolved["field_name"] == "sequence"
    assert unresolved["observed_value"] == {
        "inquiry_type": "institution",
        "missing_ocr_sequence": True,
        "boundary": "trailing",
    }
    assert unresolved["source_refs"] == [
        {
            "source": "candidate_b_canonical_inquiry_line",
            "logical_page": 8,
            "source_page": 4,
            "bbox": [110.0, 29.0, 390.0, 39.0],
            "geometry_scope": "row",
            "evidence_ids": [],
        }
    ]
    assert "ordinal_missing_at_population_boundary" in unresolved["reason_codes"]
    assert "record_not_emitted" in unresolved["reason_codes"]


def test_account_schema_reports_non_dense_family_ordinals_without_inventing_rows(monkeypatch) -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    monkeypatch.setattr(native_extraction, "_extract_table_accounts", lambda _context: ([], [], []))
    monkeypatch.setattr(
        native_extraction,
        "_account_anchor_skeletons",
        lambda _context: [
            {
                "account_id": "credit_account:credit_card:1",
                "sequence": 1,
                "category_sequence": 1,
                "account_type": "credit_card",
                "source_refs": [{"logical_page": 8, "bbox": [10, 10, 40, 20]}],
            },
            {
                "account_id": "credit_account:credit_card:3",
                "sequence": 2,
                "category_sequence": 3,
                "account_type": "credit_card",
                "source_refs": [{"logical_page": 9, "bbox": [10, 10, 40, 20]}],
            },
        ],
    )

    accounts, repayments, events = native_extraction._extract_accounts(context)

    assert len(accounts) == 2
    assert repayments == []
    assert events == []
    gap = next(
        issue
        for issue in context._personal_detail_extraction_issues
        if issue["issue_code"] == "candidate_b_account_sequence_gap"
    )
    assert gap["candidate_value"]["missing_category_sequences"] == [2]


def test_account_event_does_not_shift_first_nonempty_cell_into_missing_note_slot() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    page = SimpleNamespace(page_number=8, source_page_number=4, height=800)
    table = SimpleNamespace(
        table_id="account:event",
        metadata={},
        confidence=0.95,
        bbox=[10, 10, 590, 120],
    )
    account = {"account_id": "credit_account:1"}
    rows = [
        ["其他字段", "特殊事件说明"],
        ["不属于说明字段的值", ""],
    ]

    events = native_extraction._account_events(context, account, page, table, rows)

    assert len(events) == 1
    assert events[0]["event_type"] == "special_event_note"
    assert "details" not in events[0]
    assert any(
        issue.get("issue_code") == "candidate_b_account_event_slot_unresolved"
        and issue.get("field_name") == "details"
        and "positional_fallback_forbidden" in issue.get("reason_codes", ())
        for issue in context._personal_detail_extraction_issues
    )


def test_account_family_state_preserves_parenthesized_revolving_variant() -> None:
    context = SimpleNamespace(
        corrected_evidence_pages=lambda: [
            {
                "page": 3,
                "source_page": 2,
                "lines": [
                    {"text": "循环贷账户（二）", "bbox": [10, 10, 200, 30]},
                    {"text": "账户 1：", "bbox": [10, 40, 100, 60]},
                    {"text": "账户标识", "bbox": [10, 70, 100, 90]},
                    {"text": "查询记录", "bbox": [10, 200, 100, 220]},
                ],
            }
        ]
    )

    skeletons = native_extraction._account_anchor_skeletons(context)

    assert len(skeletons) == 1
    assert skeletons[0]["account_type"] == "revolving_loan_account"
    assert skeletons[0]["account_family_quality"] == "exact"


def test_account_anchor_segment_crosses_page_only_after_verified_transition() -> None:
    pages = [
        {
            "page": 1,
            "source_page": 1,
            "lines": [
                {"text": "贷记卡账户", "bbox": [10, 10, 200, 20]},
                {"text": "账户 1：", "bbox": [10, 30, 100, 40]},
                {"text": "账户标识", "bbox": [10, 700, 100, 720]},
            ],
        },
        {
            "page": 2,
            "source_page": 2,
            "lines": [
                {"text": "还款记录", "bbox": [10, 20, 100, 40]},
                {"text": "查询记录", "bbox": [10, 300, 100, 320]},
            ],
        },
    ]
    denied = SimpleNamespace(
        corrected_evidence_pages=lambda: pages,
        allows_scanned_line_transition=lambda *_args: None,
    )
    verified = SimpleNamespace(
        corrected_evidence_pages=lambda: pages,
        allows_scanned_line_transition=lambda *_args: True,
    )

    denied_row = native_extraction._account_anchor_skeletons(denied)[0]
    verified_row = native_extraction._account_anchor_skeletons(verified)[0]

    assert [page["logical_page"] for page in denied_row["_canonical_segment"]["pages"]] == [1]
    assert [page["logical_page"] for page in verified_row["_canonical_segment"]["pages"]] == [1, 2]
    assert verified_row["_canonical_segment"]["pages"][1]["continuation_verified"] is True


def test_monthly_link_reports_population_loss_when_account_ordinals_are_missing() -> None:
    gap = {
        "issue_code": "candidate_b_account_sequence_gap",
        "status": "requires_review",
        "observed_value": {"account_type": "credit_card"},
        "candidate_value": {"missing_category_sequences": [2]},
        "source_refs": [{"logical_page": 1, "bbox": [10, 10, 50, 20]}],
    }
    context = SimpleNamespace(_personal_detail_extraction_issues=[gap])
    accounts = [
        {
            "account_id": "credit_account:credit_card:1",
            "sequence": 1,
            "source_refs": [{"logical_page": 1, "bbox": [10, 10, 50, 20]}],
        }
    ]
    repayments = [
        {
            "grid_id": f"grid:{index}",
            "year": 2024,
            "month": 1,
            "status": status,
            "source_cell_refs": [{"logical_page": 1, "bbox": [10, 30, 20, 40]}],
        }
        for index, status in enumerate(("N", "1"), start=1)
    ]

    linked = link_candidate_b_repayments(
        repayments,
        accounts,
        [],
        reading_order_by_logical={1: 1},
        issue_context=context,
    )

    assert len(linked) == 1
    linkage_gap = next(
        issue
        for issue in context._personal_detail_extraction_issues
        if issue["issue_code"] == "monthly_linkage_collision_from_account_gap"
    )
    assert linkage_gap["observed_value"]["final_linked_row_count"] == 1
    assert linkage_gap["candidate_value"]["pre_deduplication_row_count"] == 2


def test_monthly_equal_values_from_distinct_grids_report_account_gap_collision() -> None:
    gap = {
        "issue_code": "candidate_b_account_sequence_gap",
        "status": "requires_review",
        "observed_value": {"account_type": "credit_card"},
        "candidate_value": {"missing_category_sequences": [2]},
        "source_refs": [{"logical_page": 4, "bbox": [10, 10, 50, 20]}],
    }
    context = SimpleNamespace(_personal_detail_extraction_issues=[gap])
    accounts = [
        {
            "account_id": "account:1",
            "_canonical_segment": {"pages": [{"logical_page": 4, "min_y": 20.0, "max_y": 300.0}]},
        }
    ]
    grids = [
        {"grid_id": grid_id, "page": 4, "bbox": [10, top, 100, top + 30]}
        for grid_id, top in (("grid:1", 100), ("grid:2", 180))
    ]
    rows = [
        {
            "repayment_id": f"{grid_id}:2024-01",
            "grid_id": grid_id,
            "year": 2024,
            "month": 1,
            "status": "N",
            "overdue_amount": "0",
            "source_cell_refs": [{"grid_id": grid_id, "logical_page": 4, "bbox": [10, top, 20, top + 10]}],
        }
        for grid_id, top in (("grid:1", 100), ("grid:2", 180))
    ]

    linked = link_candidate_b_repayments(rows, accounts, grids, issue_context=context)

    assert len(linked) == 1
    assert not any(
        issue.get("issue_code") == "candidate_b_monthly_duplicate_conflict"
        for issue in context._personal_detail_extraction_issues
    )
    collision = next(
        issue
        for issue in context._personal_detail_extraction_issues
        if issue.get("issue_code") == "monthly_linkage_collision_from_account_gap"
    )
    assert collision["target_record_id"] == "grid:1:2024-01"
    expected_observed = {
        "account_id": "account:1",
        "performance_month": "2024-01",
        "colliding_grid_ids": ["grid:1", "grid:2"],
        "distinct_grid_count": 2,
        "suppressed_candidate_count": 1,
    }
    assert {key: collision["observed_value"][key] for key in expected_observed} == expected_observed
    assert collision["candidate_value"]["collapsed_candidate_count"] == 1
    assert collision["candidate_value"]["missing_account_category_sequences"] == {"credit_card": [2]}
    assert {ref["grid_id"] for ref in collision["source_refs"]} == {"grid:1", "grid:2"}


def test_monthly_grid_uses_one_owner_and_matches_printed_date_range() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    accounts = [
        {
            "account_id": "account:1",
            "account_identifier": "蚂蚁借呗合并BAD12345",
            "page": 4,
            "bbox": [10, 20, 100, 100],
            "sequence": 1,
        }
    ]
    grid = {
        "grid_id": "grid:range",
        "page": 4,
        "bbox": [10, 120, 100, 220],
        "audit": {"date_range": {"start_year": 2024, "start_month": 1, "end_year": 2024, "end_month": 2}},
    }
    rows = [{"grid_id": "grid:range", "year": 2024, "month": month, "status": "N"} for month in (1, 2)]

    linked = link_candidate_b_repayments(rows, accounts, [grid], issue_context=context)

    assert {row["account_id"] for row in linked} == {"account:1"}
    assert all("account_identifier" not in row for row in linked)
    assert not any(
        issue["issue_code"] == "candidate_b_monthly_grid_contract_unresolved"
        for issue in context._personal_detail_extraction_issues
    )


def test_monthly_grid_reports_missing_printed_month_without_inventing_it() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    accounts = [{"account_id": "account:1", "page": 4, "bbox": [10, 20, 100, 100], "sequence": 1}]
    grid = {
        "grid_id": "grid:range",
        "page": 4,
        "bbox": [10, 120, 100, 220],
        "audit": {"date_range": {"start_year": 2024, "start_month": 1, "end_year": 2024, "end_month": 2}},
    }

    linked = link_candidate_b_repayments(
        [{"grid_id": "grid:range", "year": 2024, "month": 1, "status": "N"}],
        accounts,
        [grid],
        issue_context=context,
    )

    assert len(linked) == 1
    issue = next(
        issue
        for issue in context._personal_detail_extraction_issues
        if issue["issue_code"] == "candidate_b_monthly_grid_contract_unresolved"
    )
    assert issue["observed_value"]["linked_month_count"] == 1
    assert issue["observed_value"]["linked_months"] == ["2024-01"]
    assert issue["observed_value"]["grid_id"] == "grid:range"
    assert issue["candidate_value"]["printed_month_count"] == 2
    assert issue["candidate_value"]["printed_months"] == ["2024-01", "2024-02"]
    assert "target_record_id" not in issue


def test_monthly_grid_reports_each_missing_month_field_with_exact_identity() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    accounts = [
        {
            "account_id": "account:1",
            "page": 4,
            "bbox": [10, 20, 100, 100],
            "sequence": 1,
        }
    ]
    grid = {
        "grid_id": "grid:range",
        "page": 4,
        "bbox": [10, 120, 220, 220],
        "audit": {
            "date_range": {
                "start_year": 2024,
                "start_month": 1,
                "end_year": 2024,
                "end_month": 2,
            }
        },
    }

    linked = link_candidate_b_repayments(
        [
            {
                "grid_id": "grid:range",
                "year": 2024,
                "month": 1,
                "status": "N",
                "source_cell_refs": [
                    {
                        "logical_page": 4,
                        "grid_id": "grid:range",
                        "field_name": "status",
                        "bbox": [20, 140, 30, 150],
                        "geometry_scope": "cell",
                    }
                ],
            }
        ],
        accounts,
        [grid],
        issue_context=context,
    )

    assert len(linked) == 1
    field_issues = [
        issue
        for issue in context._personal_detail_extraction_issues
        if issue["issue_code"] == "candidate_b_monthly_grid_contract_missing_field"
    ]
    assert {(issue["target_record_id"], issue["field_name"]) for issue in field_issues} == {
        ("grid:range:2024-02", "performance_month"),
        ("grid:range:2024-02", "status_code"),
    }
    assert all(issue["source_refs"] for issue in field_issues)
    assert all(ref["geometry_scope"] == "grid" for issue in field_issues for ref in issue["source_refs"])
    assert all(
        ref["grid_id"] == "grid:range"
        and ref["performance_month"] == "2024-02"
        and ref["field_name"] == issue["field_name"]
        for issue in field_issues
        for ref in issue["source_refs"]
    )
    assert not any(issue["field_name"] == "status_amount" for issue in field_issues)


def test_credit_agreement_missing_fields_never_inherit_record_geometry() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])

    native_extraction.reconcile_candidate_b_credit_lines(
        context,
        [
            {
                "account_identifier": "B10512900H00010010011135264974289",
                "_printed_sequence": 1,
                "source_refs": [
                    {
                        "logical_page": 8,
                        "source_page": 4,
                        "bbox": [1.0, 2.0, 300.0, 400.0],
                        "geometry_scope": "table",
                    }
                ],
            }
        ],
    )

    assert not any(
        issue.get("issue_code") == "candidate_b_credit_agreement_required_field_unresolved"
        for issue in context._personal_detail_extraction_issues
    )


def test_monthly_owner_withholding_is_field_local_without_inventing_values() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    row = {
        "grid_id": "grid:unowned",
        "performance_month": "2024-01",
        "year": 2024,
        "month": 1,
        "status": "N",
        "overdue_amount": "88",
        "source_cell_refs": [
            {
                "logical_page": 6,
                "grid_id": "grid:unowned",
                "col": 1,
                "field_name": "status",
                "bbox": [20, 140, 30, 150],
                "geometry_scope": "cell",
            },
            {
                "logical_page": 6,
                "grid_id": "grid:unowned",
                "col": 1,
                "field_name": "overdue_amount",
                "bbox": [20, 160, 30, 170],
                "geometry_scope": "cell",
            },
        ],
    }
    grid = {
        "grid_id": "grid:unowned",
        "page": 6,
        "bbox": [10, 120, 220, 220],
        "audit": {
            "date_range": {
                "start_year": 2024,
                "start_month": 1,
                "end_year": 2024,
                "end_month": 1,
            }
        },
    }

    assert (
        link_candidate_b_repayments(
            [row],
            [],
            [grid],
            issue_context=context,
        )
        == []
    )

    aggregate = [
        issue
        for issue in context._personal_detail_extraction_issues
        if issue["issue_code"] == "candidate_b_monthly_grid_owner_unresolved"
    ]
    field_issues = [
        issue
        for issue in context._personal_detail_extraction_issues
        if issue["issue_code"] == "candidate_b_monthly_grid_owner_unresolved_field"
    ]
    assert len(aggregate) == 1
    assert "target_record_id" not in aggregate[0]
    assert {(issue["target_record_id"], issue["field_name"]) for issue in field_issues} == {
        ("grid:unowned:2024-01", "performance_month"),
        ("grid:unowned:2024-01", "status_code"),
        ("grid:unowned:2024-01", "status_amount"),
    }
    assert next(issue for issue in field_issues if issue["field_name"] == "status_code")["observed_value"][
        "source_observations"
    ] == ["N"]
    assert next(issue for issue in field_issues if issue["field_name"] == "status_amount")["observed_value"][
        "source_observations"
    ] == ["88"]
    assert all(issue["source_refs"] for issue in field_issues)
    assert len({(issue["target_record_id"], issue["field_name"]) for issue in field_issues}) == len(field_issues)

    projected = project_personal_detail_datasets(
        {
            "repayment_records": [],
            "personal_detail_extraction_issues": context._personal_detail_extraction_issues,
        }
    )
    public_issues = [
        issue
        for issue in projected["extraction_issues"]
        if issue["issue_code"] == "candidate_b_monthly_grid_owner_unresolved_field"
    ]
    assert {(issue["target_record_id"], issue["field_name"]) for issue in public_issues} == {
        ("grid:unowned:2024-01", "performance_month"),
        ("grid:unowned:2024-01", "status_code"),
        ("grid:unowned:2024-01", "status_amount"),
    }
    issue_ids = {issue["extraction_issue_id"] for issue in public_issues}
    evidence_by_issue = {
        issue_id: [
            evidence
            for evidence in projected["extraction_issue_evidence"]
            if evidence["extraction_issue_id"] == issue_id
        ]
        for issue_id in issue_ids
    }
    assert all(evidence_by_issue.values())
    assert all(
        any(
            evidence["evidence_kind"] == "reason" and evidence.get("string_value") == "normalized_value_withheld"
            for evidence in evidence_by_issue[issue_id]
        )
        for issue_id in issue_ids
    )


def test_monthly_owner_withholding_keeps_ambiguous_month_identity_aggregate_only() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    grid = {"grid_id": "grid:ambiguous", "page": 6, "bbox": [10, 120, 220, 220]}

    assert (
        link_candidate_b_repayments(
            [
                {
                    "grid_id": "grid:ambiguous",
                    "performance_month": "2024-13",
                    "status": "N",
                    "source_cell_refs": [
                        {
                            "logical_page": 6,
                            "grid_id": "grid:ambiguous",
                            "field_name": "status",
                            "bbox": [20, 140, 30, 150],
                            "geometry_scope": "cell",
                        }
                    ],
                }
            ],
            [],
            [grid],
            issue_context=context,
        )
        == []
    )

    assert any(
        issue["issue_code"] == "candidate_b_monthly_grid_owner_unresolved"
        for issue in context._personal_detail_extraction_issues
    )
    assert not any(
        issue["issue_code"] == "candidate_b_monthly_grid_owner_unresolved_field"
        for issue in context._personal_detail_extraction_issues
    )


def test_monthly_owner_withholding_does_not_report_unobserved_default_amount() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    grid = {
        "grid_id": "grid:no-amount-cell",
        "page": 6,
        "bbox": [10, 120, 220, 220],
        "audit": {
            "date_range": {
                "start_year": 2024,
                "start_month": 1,
                "end_year": 2024,
                "end_month": 1,
            }
        },
    }
    row = {
        "grid_id": "grid:no-amount-cell",
        "performance_month": "2024-01",
        "year": 2024,
        "month": 1,
        "status": "N",
        # A compatibility/default value is not proof that an amount cell was visible.
        "overdue_amount": "0",
        "source_cell_refs": [
            {
                "logical_page": 6,
                "grid_id": "grid:no-amount-cell",
                "col": 1,
                "field_name": "status",
                "bbox": [20, 140, 30, 150],
                "geometry_scope": "cell",
            }
        ],
    }

    assert link_candidate_b_repayments([row], [], [grid], issue_context=context) == []
    field_issues = [
        issue
        for issue in context._personal_detail_extraction_issues
        if issue["issue_code"] == "candidate_b_monthly_grid_owner_unresolved_field"
    ]
    assert {issue["field_name"] for issue in field_issues} == {
        "performance_month",
        "status_code",
    }


def test_monthly_anchor_ledger_is_independent_and_account_bound() -> None:
    accounts = [
        {
            "account_id": "account:1",
            "_canonical_segment": {"pages": [{"logical_page": 4, "min_y": 20, "max_y": 300}]},
        },
        {
            "account_id": "account:2",
            "_canonical_segment": {"pages": [{"logical_page": 4, "min_y": 300, "max_y": 700}]},
        },
    ]
    evidence = [
        {
            "page": 4,
            "source_page": 2,
            "lines": [
                {
                    "text": "2024年01月-2024年02月的还款记录",
                    "bbox": [100, 120, 400, 140],
                }
            ],
        }
    ]

    ledger = candidate_b_repayment_anchor_ledger(evidence, accounts)

    assert len(ledger) == 1
    assert ledger[0]["account_id"] == "account:1"
    assert ledger[0]["date_range"] == {
        "start_year": 2024,
        "start_month": 1,
        "end_year": 2024,
        "end_month": 2,
    }
    assert ledger[0]["source_refs"][0]["source"] == "candidate_b_monthly_anchor_ledger"


def test_monthly_visible_anchor_with_zero_detector_grids_is_localized_and_reported() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    accounts = [
        {
            "account_id": "account:1",
            "_canonical_segment": {"pages": [{"logical_page": 4, "min_y": 20, "max_y": 300}]},
        }
    ]
    ledger = candidate_b_repayment_anchor_ledger(
        [
            {
                "page": 4,
                "source_page": 2,
                "lines": [
                    {
                        "text": "2024年01月-2024年02月的还款记录",
                        "bbox": [100, 120, 400, 140],
                    }
                ],
            }
        ],
        accounts,
    )

    linked = link_candidate_b_repayments(
        [],
        accounts,
        [],
        issue_context=context,
        repayment_anchors=ledger,
    )

    assert linked == []
    issue = next(
        issue
        for issue in context._personal_detail_extraction_issues
        if issue["issue_code"] == "candidate_b_monthly_anchor_grid_missing"
    )
    assert issue["target_record_id"] == "account:1"
    assert issue["field_name"] == "account_id"
    assert issue["observed_value"]["materialized_grid_count"] == 0
    assert issue["observed_value"]["date_range"]["start_month"] == 1


def test_monthly_visible_anchor_reconciles_to_materialized_grid() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    accounts = [
        {
            "account_id": "account:1",
            "_canonical_segment": {
                "ownership_basis": "printed_anchor_to_next_anchor",
                "anchor_logical_page": 4,
                "anchor_bbox": [10, 20, 90, 30],
                "pages": [{"logical_page": 4, "min_y": 20, "max_y": 300}],
            },
        }
    ]
    ledger = candidate_b_repayment_anchor_ledger(
        [
            {
                "page": 4,
                "source_page": 2,
                "lines": [
                    {
                        "text": "2024年01月-2024年01月的还款记录",
                        "bbox": [100, 120, 400, 140],
                    }
                ],
            }
        ],
        accounts,
    )
    grid = {
        "grid_id": "grid:range",
        "page": 4,
        "bbox": [100, 140, 400, 220],
        "audit": {
            "date_range": {
                "start_year": 2024,
                "start_month": 1,
                "end_year": 2024,
                "end_month": 1,
            }
        },
    }

    linked = link_candidate_b_repayments(
        [{"grid_id": "grid:range", "year": 2024, "month": 1, "status": "N"}],
        accounts,
        [grid],
        issue_context=context,
        repayment_anchors=ledger,
    )

    assert len(linked) == 1
    assert linked[0]["account_id"] == "account:1"
    assert not any(
        issue["issue_code"] == "candidate_b_monthly_anchor_grid_missing"
        for issue in context._personal_detail_extraction_issues
    )


def _monthly_anchor_evidence(
    text: str,
    *,
    page: int = 4,
    source_page: int = 2,
    evidence_ids: list[str] | None = None,
) -> list[dict[str, object]]:
    return [
        {
            "page": page,
            "source_page": source_page,
            "lines": [
                {
                    "text": text,
                    "bbox": [100, 120, 400, 140],
                    "evidence_ids": evidence_ids or ["ocr:monthly-range:1"],
                    "line_id": "ocr:monthly-range:line:1",
                }
            ],
        }
    ]


def _monthly_anchor_account(
    *,
    account_id: str = "account:1",
    page: int = 4,
    minimum: float = 20,
    maximum: float = 300,
) -> dict[str, object]:
    return {
        "account_id": account_id,
        "_canonical_segment": {
            "ownership_basis": "printed_anchor_to_next_anchor",
            "anchor_logical_page": page,
            "anchor_bbox": [10, minimum, 90, minimum + 10],
            "pages": [
                {
                    "logical_page": page,
                    "min_y": minimum,
                    "max_y": maximum,
                }
            ],
        },
    }


def test_monthly_account_range_reports_every_missing_month_without_status_value() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    accounts = [_monthly_anchor_account()]
    ledger = candidate_b_repayment_anchor_ledger(
        _monthly_anchor_evidence("2023年11月一2024年02月的还款记录"),
        accounts,
    )

    assert ledger[0]["date_range"] == {
        "start_year": 2023,
        "start_month": 11,
        "end_year": 2024,
        "end_month": 2,
    }
    assert ledger[0]["source_refs"][0]["evidence_ids"] == ["ocr:monthly-range:1"]

    assert link_candidate_b_repayments([], accounts, [], issue_context=context, repayment_anchors=ledger) == []
    field_issues = [
        issue
        for issue in context._personal_detail_extraction_issues
        if issue["issue_code"]
        in {
            "candidate_b_monthly_account_range_missing_month",
            "candidate_b_monthly_account_range_status_grid_unavailable",
        }
    ]
    assert len(field_issues) == 8
    assert {issue["observed_value"]["performance_month"] for issue in field_issues} == {
        "2023-11",
        "2023-12",
        "2024-01",
        "2024-02",
    }
    assert {issue["field_name"] for issue in field_issues} == {
        "performance_month",
        "status_code",
    }
    assert all(
        issue["observed_value"]["account_id"] == "account:1"
        and issue["target_record_id"].startswith("source_account_month:")
        and issue["source_refs"][0]["evidence_ids"] == ["ocr:monthly-range:1"]
        and issue["source_refs"][0]["logical_page"] == 4
        and issue["source_refs"][0]["source_page"] == 2
        for issue in field_issues
    )
    status_issues = [issue for issue in field_issues if issue["field_name"] == "status_code"]
    assert all(
        issue["source_refs"][0]["binding"] == "source_account_month_identity"
        and "source_observations" not in issue["observed_value"]
        and "status_source_grid_unavailable" in issue["reason_codes"]
        for issue in status_issues
    )

    projected = project_personal_detail_datasets(
        {
            "repayment_records": [],
            "personal_detail_extraction_issues": (context._personal_detail_extraction_issues),
        }
    )
    public_issues = [
        issue
        for issue in projected["extraction_issues"]
        if issue["issue_code"].startswith("candidate_b_monthly_account_range_")
    ]
    assert len(public_issues) == 8
    assert {(issue["target_record_id"], issue["field_name"]) for issue in public_issues} == {
        (issue["target_record_id"], issue["field_name"]) for issue in field_issues
    }
    assert all(
        issue["target_dataset"] == "credit_account_monthly_performance"
        and issue["source_refs"][0]["geometry_scope"] == "line"
        and issue["source_refs"][0]["evidence_ids"] == ["ocr:monthly-range:1"]
        for issue in public_issues
    )
    assert all(
        issue["source_refs"][0]["binding"] not in {"grid_month_cell", "monthly_grid_cell", "source_monthly_field_cell"}
        for issue in public_issues
        if issue["field_name"] == "status_code"
    )


def test_monthly_account_range_overlap_reports_only_unrepresented_months() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    accounts = [_monthly_anchor_account()]
    ledger = candidate_b_repayment_anchor_ledger(
        _monthly_anchor_evidence("2024年01月—2024年03月的还款记录"),
        accounts,
    )
    grid = {
        "grid_id": "grid:overlap",
        "account_id": "account:1",
        "page": 4,
        "bbox": [100, 140, 400, 220],
        "audit": {
            "date_range": {
                "start_year": 2024,
                "start_month": 1,
                "end_year": 2024,
                "end_month": 2,
            }
        },
    }
    rows = [
        {
            "grid_id": "grid:overlap",
            "account_id": "account:1",
            "performance_month": f"2024-0{month}",
            "year": 2024,
            "month": month,
            "status": "N",
        }
        for month in (1, 2)
    ]

    linked = link_candidate_b_repayments(
        rows,
        accounts,
        [grid],
        issue_context=context,
        repayment_anchors=ledger,
    )

    assert len(linked) == 2
    range_issues = [
        issue
        for issue in context._personal_detail_extraction_issues
        if issue["issue_code"].startswith("candidate_b_monthly_account_range_")
    ]
    assert {(issue["observed_value"]["performance_month"], issue["field_name"]) for issue in range_issues} == {
        ("2024-03", "performance_month"),
        ("2024-03", "status_code"),
    }


@pytest.mark.parametrize(
    "text",
    [
        "2024年01月附近2024年02月的还款记录",
        "2024年01月—2024年02月—2024年03月的还款记录",
        "2024年01月—2024年13月的还款记录",
        "2024年02月—2024年01月的还款记录",
        "2024年01月—2024年02月的查询记录",
    ],
)
def test_monthly_account_range_rejects_malformed_near_misses(text: str) -> None:
    ledger = candidate_b_repayment_anchor_ledger(_monthly_anchor_evidence(text), [_monthly_anchor_account()])

    assert not ledger or ledger[0]["date_range"] is None


def test_monthly_account_range_rejects_ambiguous_duplicate_owner() -> None:
    accounts = [
        _monthly_anchor_account(account_id="account:1"),
        _monthly_anchor_account(account_id="account:2"),
    ]

    assert (
        candidate_b_repayment_anchor_ledger(_monthly_anchor_evidence("2024年01月—2024年02月的还款记录"), accounts) == []
    )


def test_monthly_account_range_rejects_cross_page_segment_mismatch() -> None:
    account = _monthly_anchor_account(page=5)

    assert (
        candidate_b_repayment_anchor_ledger(
            _monthly_anchor_evidence("2024年01月—2024年02月的还款记录", page=4, source_page=2),
            [account],
        )
        == []
    )


def test_monthly_account_range_missing_evidence_stays_aggregate_only() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    accounts = [_monthly_anchor_account()]
    evidence = _monthly_anchor_evidence("2024年01月—2024年02月的还款记录")
    evidence[0]["lines"][0].pop("evidence_ids")
    ledger = candidate_b_repayment_anchor_ledger(evidence, accounts)

    link_candidate_b_repayments([], accounts, [], issue_context=context, repayment_anchors=ledger)

    assert any(
        issue["issue_code"] == "candidate_b_monthly_anchor_grid_missing"
        for issue in context._personal_detail_extraction_issues
    )
    assert not any(
        issue["issue_code"].startswith("candidate_b_monthly_account_range_")
        for issue in context._personal_detail_extraction_issues
    )


def test_account_month_closure_spans_variable_ranges_and_cross_page_grids() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    account = _monthly_anchor_account(maximum=300)
    account["_canonical_segment"]["pages"].append({"logical_page": 5, "min_y": 0, "max_y": 700})
    ledger = candidate_b_repayment_anchor_ledger(
        _monthly_anchor_evidence(
            "2023年11月—2024年02月的还款记录",
            page=4,
        ),
        [account],
    )
    grids = [
        {
            "grid_id": "grid:later-scaled",
            "page": 5,
            "bbox": [40, 80, 560, 260],
            "audit": {
                "date_range": {
                    "start_year": 2024,
                    "start_month": 1,
                    "end_year": 2024,
                    "end_month": 2,
                }
            },
        },
        {
            "grid_id": "grid:earlier",
            "page": 4,
            "bbox": [100, 140, 400, 220],
            "audit": {
                "date_range": {
                    "start_year": 2023,
                    "start_month": 11,
                    "end_year": 2023,
                    "end_month": 12,
                }
            },
        },
    ]
    rows = [
        {
            "grid_id": grid_id,
            "year": year,
            "month": month,
            "status": "N",
        }
        for grid_id, year, month in (
            ("grid:later-scaled", 2024, 2),
            ("grid:earlier", 2023, 12),
            ("grid:later-scaled", 2024, 1),
            ("grid:earlier", 2023, 11),
        )
    ]

    linked = link_candidate_b_repayments(
        rows,
        [account],
        grids,
        issue_context=context,
        repayment_anchors=ledger,
    )

    assert {(row["account_id"], row["year"], row["month"]) for row in linked} == {
        ("account:1", 2023, 11),
        ("account:1", 2023, 12),
        ("account:1", 2024, 1),
        ("account:1", 2024, 2),
    }
    assert all(row["_account_month_identity_proof"]["unique_owner"] is True for row in linked)
    assert not any(
        issue["issue_code"] == "candidate_b_monthly_anchor_grid_missing"
        for issue in context._personal_detail_extraction_issues
    )


def test_account_month_closure_deduplicates_reordered_grid_aliases() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    account = _monthly_anchor_account()
    grids = [
        {
            "grid_id": grid_id,
            "page": 4,
            "bbox": bbox,
            "audit": {
                "date_range": {
                    "start_year": 2024,
                    "start_month": 1,
                    "end_year": 2024,
                    "end_month": 2,
                }
            },
        }
        for grid_id, bbox in (
            ("grid:alias-b", [60, 150, 500, 250]),
            ("grid:alias-a", [100, 140, 400, 220]),
        )
    ]
    rows = [
        {"grid_id": grid_id, "year": 2024, "month": month, "status": "N"}
        for grid_id, month in (
            ("grid:alias-b", 2),
            ("grid:alias-a", 1),
            ("grid:alias-b", 1),
            ("grid:alias-a", 2),
        )
    ]

    linked = link_candidate_b_repayments(
        rows,
        [account],
        grids,
        issue_context=context,
    )

    assert {(row["year"], row["month"]) for row in linked} == {
        (2024, 1),
        (2024, 2),
    }
    assert not any(
        issue["issue_code"]
        in {
            "candidate_b_monthly_grid_contract_missing_field",
            "candidate_b_monthly_grid_contract_unresolved",
        }
        for issue in context._personal_detail_extraction_issues
    )
    alias_issues = [
        issue
        for issue in context._personal_detail_extraction_issues
        if issue["issue_code"] == "candidate_b_monthly_source_position_alias_reconciled"
    ]
    assert len(alias_issues) == 2
    assert {issue["observed_value"]["performance_month"] for issue in alias_issues} == {"2024-01", "2024-02"}
    assert all(
        issue["status"] == "informational"
        and issue["observed_value"]["source_position_state"] == "owner_bound_alias"
        and issue["source_refs"]
        and issue["source_refs"][0]["grid_id"] == issue["observed_value"]["grid_id"]
        and issue["source_refs"][0]["performance_month"] == issue["observed_value"]["performance_month"]
        for issue in alias_issues
    )


def test_detached_source_position_promotes_only_with_unique_exact_owner() -> None:
    source_grid = {
        "grid_id": "grid:detached",
        "page": 4,
        "bbox": [100, 140, 400, 220],
        "audit": {
            "date_range": {
                "start_year": 2024,
                "start_month": 1,
                "end_year": 2024,
                "end_month": 1,
            }
        },
    }
    source_record = {
        "grid_id": "grid:detached",
        "year": 2024,
        "month": 1,
        "status": "unknown",
        "source_cell_refs": [
            {
                "source": "sealed_native_physical_table_cell",
                "logical_page": 4,
                "source_page": 2,
                "table_id": "monthly-table:detached",
                "grid_id": "grid:detached",
                "row": 2,
                "col": 1,
                "field_name": "status",
                "performance_month": "2024-01",
                "bbox": [110, 160, 130, 175],
                "geometry_scope": "cell",
                "evidence_ids": ["native:monthly-table:detached:2:1"],
                "binding": "source_monthly_field_cell",
                "binding_quality": "source_monthly_field_cell",
            }
        ],
    }
    exact_context = SimpleNamespace(
        _personal_detail_extraction_issues=[],
        _candidate_b_monthly_source_structure_grids=[source_grid],
        _candidate_b_monthly_source_structure_records=[source_record],
    )

    assert (
        link_candidate_b_repayments(
            [],
            [_monthly_anchor_account()],
            [],
            issue_context=exact_context,
        )
        == []
    )
    exact_issues = [
        issue
        for issue in exact_context._personal_detail_extraction_issues
        if issue["issue_code"] == "candidate_b_monthly_owned_grid_missing_field"
    ]
    assert {issue["field_name"] for issue in exact_issues} == {"performance_month"}
    assert all(
        issue["target_record_id"].startswith("source_account_month:")
        and issue["observed_value"] == {"account_id": "account:1", "performance_month": "2024-01"}
        and len(issue["source_refs"]) == 1
        and issue["source_refs"][0]["source"] == "candidate_b_monthly_owned_grid_cell"
        and issue["source_refs"][0]["binding"] == "source_account_month_identity"
        for issue in exact_issues
    )

    untrusted_record = {
        **source_record,
        "source_cell_refs": [{**source_record["source_cell_refs"][0], "source": "unrelated_cell"}],
    }
    untrusted_context = SimpleNamespace(
        _personal_detail_extraction_issues=[],
        _candidate_b_monthly_source_structure_grids=[source_grid],
        _candidate_b_monthly_source_structure_records=[untrusted_record],
    )
    link_candidate_b_repayments(
        [],
        [_monthly_anchor_account()],
        [],
        issue_context=untrusted_context,
    )
    assert not any(
        issue["issue_code"] == "candidate_b_monthly_owned_grid_missing_field"
        for issue in untrusted_context._personal_detail_extraction_issues
    )

    competing_cell_record = {
        **source_record,
        "source_cell_refs": [
            source_record["source_cell_refs"][0],
            {
                **source_record["source_cell_refs"][0],
                "column": 2,
                "col": 2,
                "bbox": [135, 160, 155, 175],
                "evidence_ids": ["native:monthly-table:detached:2:2"],
            },
        ],
    }
    competing_cell_context = SimpleNamespace(
        _personal_detail_extraction_issues=[],
        _candidate_b_monthly_source_structure_grids=[source_grid],
        _candidate_b_monthly_source_structure_records=[competing_cell_record],
    )
    link_candidate_b_repayments(
        [],
        [_monthly_anchor_account()],
        [],
        issue_context=competing_cell_context,
    )
    assert not any(
        issue["issue_code"] == "candidate_b_monthly_owned_grid_missing_field"
        for issue in competing_cell_context._personal_detail_extraction_issues
    )

    for malformed_evidence_ids in (
        ["native:duplicate", "native:duplicate"],
        ["native:valid", ""],
        ["native:valid", 7],
    ):
        malformed_evidence_context = SimpleNamespace(
            _personal_detail_extraction_issues=[],
            _candidate_b_monthly_source_structure_grids=[source_grid],
            _candidate_b_monthly_source_structure_records=[
                {
                    **source_record,
                    "source_cell_refs": [
                        {
                            **source_record["source_cell_refs"][0],
                            "evidence_ids": malformed_evidence_ids,
                        }
                    ],
                }
            ],
        )
        link_candidate_b_repayments(
            [],
            [_monthly_anchor_account()],
            [],
            issue_context=malformed_evidence_context,
        )
        assert not any(
            issue["issue_code"] == "candidate_b_monthly_owned_grid_missing_field"
            for issue in malformed_evidence_context._personal_detail_extraction_issues
        )

    ambiguous_context = SimpleNamespace(
        _personal_detail_extraction_issues=[],
        _candidate_b_monthly_source_structure_grids=[source_grid],
        _candidate_b_monthly_source_structure_records=[source_record],
    )
    link_candidate_b_repayments(
        [],
        [
            _monthly_anchor_account(account_id="account:1"),
            _monthly_anchor_account(account_id="account:2"),
        ],
        [],
        issue_context=ambiguous_context,
    )
    assert not any(
        issue.get("observed_value", {}).get("account_id")
        for issue in ambiguous_context._personal_detail_extraction_issues
    )


def test_detached_exact_source_alias_is_preserved_without_double_counting() -> None:
    date_range = {
        "start_year": 2024,
        "start_month": 1,
        "end_year": 2024,
        "end_month": 1,
    }
    primary_grid = {
        "grid_id": "grid:primary",
        "page": 4,
        "bbox": [100, 140, 400, 220],
        "audit": {"date_range": date_range},
    }
    detached_grid = {
        **primary_grid,
        "grid_id": "grid:detached-alias",
        "bbox": [90, 145, 420, 225],
    }
    detached_record = {
        "grid_id": "grid:detached-alias",
        "year": 2024,
        "month": 1,
        "status": "unknown",
        "source_cell_refs": [
            {
                "logical_page": 4,
                "grid_id": "grid:detached-alias",
                "row": 2,
                "col": 1,
                "field_name": "status",
                "bbox": [110, 160, 130, 175],
                "geometry_scope": "cell",
            }
        ],
    }
    context = SimpleNamespace(
        _personal_detail_extraction_issues=[],
        _candidate_b_monthly_source_structure_grids=[detached_grid],
        _candidate_b_monthly_source_structure_records=[detached_record],
    )

    linked = link_candidate_b_repayments(
        [
            {
                "grid_id": "grid:primary",
                "year": 2024,
                "month": 1,
                "status": "N",
            }
        ],
        [_monthly_anchor_account()],
        [primary_grid],
        issue_context=context,
    )

    assert len(linked) == 1
    alias_issues = [
        issue
        for issue in context._personal_detail_extraction_issues
        if issue["issue_code"] == "candidate_b_monthly_source_position_alias_reconciled"
    ]
    assert len(alias_issues) == 1
    assert alias_issues[0]["observed_value"] == {
        "account_id": "account:1",
        "grid_id": "grid:detached-alias",
        "performance_month": "2024-01",
        "source_position_state": "owner_bound_alias",
        "account_month_owner_basis": "canonical_account_segment",
    }
    assert alias_issues[0]["source_refs"][0]["geometry_scope"] == "cell"
    assert not any(
        issue["issue_code"] == "canonical_monthly_source_structure_missing_field"
        for issue in context._personal_detail_extraction_issues
    )
