from __future__ import annotations

from types import SimpleNamespace

from docmirror.plugins.credit_report.personal_detail_scanned import (
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


def _repayment_record(record_id: str, status: str) -> dict[str, object]:
    grid_id, performance_month = record_id.split(":", 1)
    return {
        "repayment_id": record_id,
        "grid_id": grid_id,
        "performance_month": performance_month,
        "status": status,
        "overdue_amount": "0",
    }


def test_candidate_b_native_guard_runs_after_glyph_bank_on_final_statuses(
    monkeypatch,
) -> None:
    records = [
        _repayment_record("mg_p19_repayment_0:2022-08", "N"),
        _repayment_record("mg_agreement:2022-08", "N"),
        _repayment_record("mg_p13_repayment_1:2023-05", "*"),
        _repayment_record("mg_p13_repayment_1:2019-08", "*"),
    ]
    native_statuses = {
        "mg_p19_repayment_0": "N",
        "mg_agreement": "N",
        "mg_p13_repayment_1": {
            "2023-05": "3",
            "2019-08": "#",
        },
    }
    calls: list[str] = []

    context = SimpleNamespace(
        pages=[],
        parse_result=SimpleNamespace(),
        reading_order_by_logical={},
        account_collections=lambda: ([], [], []),
        corrected_repayment_records=lambda: records,
        corrected_repayment_micro_grids=lambda: [],
        prepare_candidate_b_business_repair=lambda _payload: False,
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
    monkeypatch.setattr(
        native_extraction,
        "reconcile_candidate_b_credit_lines",
        lambda _context, rows: rows,
    )
    monkeypatch.setattr(profile_extraction, "extract_candidate_b_profile", lambda _context: {})
    monkeypatch.setattr(relations, "candidate_b_repayment_anchor_ledger", lambda *_args: {})
    monkeypatch.setattr(relations, "link_candidate_b_repayments", lambda rows, *_args, **_kwargs: rows)
    monkeypatch.setattr(relations, "derive_candidate_b_overdue_records", lambda *_args: [])
    monkeypatch.setattr(
        consistency_ledger,
        "apply_document_consistency_ledger",
        lambda *_args: {},
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
    monkeypatch.setattr(extraction_issues, "dataset_states_from_issues", lambda _issues: {})
    monkeypatch.setattr(
        source_projection,
        "prepare_personal_detail_source_collections",
        lambda payload, _business, **_kwargs: payload,
    )

    def glyph_bank(final_records, *_args, **_kwargs):
        calls.append("glyph_bank")
        assert final_records[0]["status"] == "N"
        final_records[0]["status"] = "M"
        return {"enabled": True, "promoted_count": 1}

    monkeypatch.setattr(
        document_glyph_bank,
        "apply_document_local_status_glyph_bank",
        glyph_bank,
    )
    monkeypatch.setattr(
        native_status_conflict,
        "_exact_final_status_ref",
        lambda record, **_kwargs: {
            "field_name": "status",
            "grid_id": record["grid_id"],
            "performance_month": record["performance_month"],
        },
    )
    monkeypatch.setattr(
        native_status_conflict,
        "_exact_final_amount_ref",
        lambda record, **_kwargs: {
            "field_name": "overdue_amount",
            "grid_id": record["grid_id"],
            "performance_month": record["performance_month"],
        },
    )

    def native_candidates(_context, *, final_ref, **_kwargs):
        grid_id = str(final_ref["grid_id"])
        performance_month = str(final_ref["performance_month"])
        native_status = native_statuses[grid_id]
        if isinstance(native_status, dict):
            native_status = native_status[performance_month]
        return [
            (
                native_status,
                {"field_name": "status"},
                {"field_name": "overdue_amount"},
            )
        ]

    monkeypatch.setattr(native_status_conflict, "_native_candidates", native_candidates)
    real_guard = native_status_conflict.apply_candidate_b_native_status_conflict_guard

    def tracked_guard(*args, **kwargs):
        calls.append("native_guard")
        return real_guard(*args, **kwargs)

    monkeypatch.setattr(
        native_status_conflict,
        "apply_candidate_b_native_status_conflict_guard",
        tracked_guard,
    )

    result = CandidateBPipeline(context, "").run()

    assert calls == ["glyph_bank", "native_guard"]
    by_id = {str(record["repayment_id"]): record for record in result.business["repayment_records"]}
    assert "status" not in by_id["mg_p19_repayment_0:2022-08"]
    assert by_id["mg_p19_repayment_0:2022-08"]["overdue_amount"] == "0"
    assert by_id["mg_agreement:2022-08"]["status"] == "N"
    assert by_id["mg_p13_repayment_1:2023-05"]["status"] == "*"
    assert "status" not in by_id["mg_p13_repayment_1:2019-08"]
    assert result.audit["native_source_cell_status_guard"] == {
        "enabled": True,
        "records_checked": 4,
        "unique_native_witnesses": 4,
        "native_numeric_witnesses_rejected_for_nonpositive_amount": 1,
        "agreements": 1,
        "conflicts_withheld": 2,
    }
    issues = context._personal_detail_extraction_issues
    assert {
        issue["target_record_id"] for issue in issues
    } == {
        "mg_p19_repayment_0:2022-08",
        "mg_p13_repayment_1:2019-08",
    }
