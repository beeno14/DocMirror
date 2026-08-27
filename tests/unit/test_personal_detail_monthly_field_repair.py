"""Printed year/12-month excerpts through the real sealed-evidence contracts.

These fixtures reproduce the assembler's typed/raw row offset, source token
IDs, scanned-grid geometry and evidence-plane producer. They do not open PDFs
or run OCR; the independent full-page acquisition is an explicit test input.
"""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest

from docmirror.evidence.plane import EvidencePlaneBuilder
from docmirror.models.entities.parse_result import CellValue, PageContent, ParseResult, TableBlock, TableRow
from docmirror.plugins.credit_report.personal_detail_scanned.business_repair import (
    BusinessUncertaintyRepairCoordinator,
    apply_planned_monthly_field_repairs,
)
from docmirror.plugins.credit_report.personal_detail_scanned.candidate_b import (
    _overdue_view_input_basis,
    _refresh_final_overdue_view,
    _withhold_repayment_plane_conflicts,
)
from docmirror.plugins.credit_report.personal_detail_scanned.context import PersonalDetailExtractionContext
from docmirror.plugins.credit_report.personal_detail_scanned.native_status_conflict import (
    MONTHLY_SOURCE_OCR_CONFIDENCE_MINIMUM,
    apply_candidate_b_native_status_conflict_guard,
    authenticated_monthly_field_slots,
    resolve_sealed_monthly_field_slot,
)
from docmirror.plugins.credit_report.personal_detail_scanned.ocr_correction import PersonalDetailOCRCorrectionOverlay
from docmirror.plugins.credit_report.personal_detail_scanned.relations import derive_candidate_b_overdue_records


def _printed_monthly_case(
    *,
    native_status: str = "M",
    published_status: Any = "M",
    native_amount: str = "0",
    published_amount: Any = "0",
    source_confidence: float = 0.0767,
    year: int = 2019,
    month: int = 7,
    preserve_headers: bool = False,
    year_span: int = 2,
    transformed: bool = False,
    evidence_store: str = "both",
) -> tuple[PersonalDetailExtractionContext, dict[str, Any], dict[str, Any]]:
    logical, physical = 19, 20
    registered_page = 7 if transformed else logical
    table_id = "pt_19_0"
    edges = [45.0, 73.0, *[100.0 + 27.0 * index for index in range(12)]]
    ys = [157.5, 170.5, 183.5, 196.5]
    printed_rows = [
        ["", *[str(value) for value in range(1, 13)]],
        [str(year), *[native_status if value == month else "N" for value in range(1, 13)]],
        ["", *[native_amount if value == month else "0" for value in range(1, 13)]],
    ]
    boxes: list[list[Any]] = []
    states: list[list[str]] = []
    ids_matrix: list[list[list[str]]] = []
    typed_rows: list[TableRow] = []
    tokens: list[dict[str, Any]] = []
    offset = 1 if preserve_headers else 0
    for raw_row, text_row in enumerate(printed_rows):
        cell_boxes, cell_states, cell_ids, typed_cells = [], [], [], []
        for col, text in enumerate(text_row):
            box = [edges[col], ys[raw_row], edges[col + 1], ys[raw_row + 1]]
            status = "exact"
            row_span = year_span if (raw_row, col) == (1, 0) else 1
            if (raw_row, col) == (1, 0) and year_span == 2:
                box[3] = ys[3]
            if (raw_row, col) == (2, 0) and year_span == 2:
                box, status = None, "derived"
            ids = [f"ocr:sp0020:lp0019:r{raw_row}:c{col}"] if text else []
            if text and box is not None:
                score = source_confidence if (raw_row, col) == (1, month) else 0.99
                tokens.append({
                    "token_id": ids[0], "evidence_ids": list(ids), "page": logical,
                    "source_page": physical, "content": text, "confidence": score,
                    "bbox": [box[0] + 3.0, box[1] + 3.0, box[2] - 3.0, min(box[3] - 3.0, box[1] + 10.0)],
                    "source": "scanned_page_ocr", "coordinate_system": "pdf_points_top_left",
                })
            cell_boxes.append(box)
            cell_states.append(status)
            cell_ids.append(ids)
            if raw_row >= offset:
                typed_row = raw_row - offset
                typed_cells.append(CellValue(
                    text=text, row_index=typed_row, col_index=col, row_span=row_span,
                    bbox=box, geometry_status=status, geometry_source="scanned_image_line_grid",
                    confidence=0.99, geometry_confidence=0.99,
                    evidence_ids=list(ids), token_ids=list(ids),
                    source_cell_refs=[{"page": logical, "table_id": table_id, "row": typed_row, "raw_row": raw_row, "col": col}],
                ))
        boxes.append(cell_boxes)
        states.append(cell_states)
        ids_matrix.append(cell_ids)
        if typed_cells:
            typed_rows.append(TableRow(cells=typed_cells, source_page=logical, source_physical_id=table_id, source_row_index=raw_row - offset))
    geometry = {
        "geometry_source": "scanned_image_line_grid", "coordinate_system": "pdf_points_top_left",
        "geometry_confidence": 0.99, "cell_bboxes": boxes, "cell_geometry_status": states,
        "cell_evidence_ids": deepcopy(ids_matrix), "cell_token_ids": deepcopy(ids_matrix),
        "cell_spans": ([{"row": 1, "col": 0, "row_span": 2, "col_span": 1}] if year_span == 2 else []),
        "row_bands": [{"index": row, "y0": ys[row], "y1": ys[row + 1]} for row in range(3)],
        "col_bands": [{"index": col, "x0": edges[col], "x1": edges[col + 1]} for col in range(13)],
        "vertical_lines": [{"x": edge, "y0": ys[0], "y1": ys[-1]} for edge in edges],
        "horizontal_lines": [{"y": edge, "x0": edges[0], "x1": edges[-1]} for edge in ys],
    }
    table = TableBlock(
        table_id=table_id, page=logical, rows=typed_rows,
        headers=printed_rows[0] if preserve_headers else [],
        bbox=[edges[0], ys[0], edges[-1], ys[-1]], extraction_layer="scanned_image_line_grid",
        metadata={"preserve_headers": preserve_headers, "raw_rows": deepcopy(printed_rows), "geometry": geometry},
    )
    result = ParseResult(pages=[PageContent(page_number=logical, source_page_number=physical, width=600, height=800, tables=[table])])
    keys = ("micro_grid_evidence", "local_structure_evidence") if evidence_store == "both" else (evidence_store,)
    bundle = {"page": logical, **{key: {"page": logical, "source_page": physical, "tokens": deepcopy(tokens)} for key in keys}}
    result.entities.domain_specific["_page_evidence_bundles"] = [bundle]
    owner = SimpleNamespace(pages=result.pages, entities=result.entities, evidence_plane=EvidencePlaneBuilder().build(result))
    canonical = result.pages[0].model_copy(deep=True)
    sx, sy, ox, oy = (1.2, 0.8, 11.0, 220.0) if transformed else (1.0, 1.0, 0.0, 0.0)
    if transformed:
        canonical.page_number = registered_page
        canonical.tables[0].metadata.update({
            "source_logical_page": logical, "coordinate_logical_page": registered_page, "source_page": physical,
            "source_to_canonical_affine": {"scale_x": sx, "scale_y": sy, "offset_x": ox, "offset_y": oy},
        })
    context = object.__new__(PersonalDetailExtractionContext)
    context.parse_result = owner
    context._canonical_layout_projection_cache = SimpleNamespace(pages=[canonical])
    context._cache = {}
    context._personal_detail_extraction_issues = []
    context._initial_personal_detail_extraction_issues = []
    context._business_repair_active = False
    context._business_repair_plan = None
    context._ocr_correction_overlay = PersonalDetailOCRCorrectionOverlay(owner)
    context.source_page_by_logical = {registered_page: physical, logical: physical}
    context._source_evidence_pages = lambda: []
    context.canonical_layout_audit = lambda: {"unresolved_pages": []}
    grid = "mg_p19_repayment_0"
    refs = []
    for field, raw_row, local_row in (("status", 1, 4), ("overdue_amount", 2, 5)):
        raw_box = boxes[raw_row][month]
        refs.append({
            "logical_page": registered_page, "page": registered_page, "grid_id": grid,
            "row": local_row, "col": month, "field_name": field,
            "bbox": [raw_box[0] * sx + ox, raw_box[1] * sy + oy, raw_box[2] * sx + ox, raw_box[3] * sy + oy],
            "geometry_scope": "cell", "geometry_status": "exact", "coordinate_system": "pdf_points_top_left",
            "geometry_provenance": {
                "selection_basis": "source_table_year_plus_twelve_ownership", "source": "source_table_geometry",
                "table_id": table_id, "status_row_index": 1, "amount_row_index": 2, "year_anchor_row_index": 1,
                "logical_page": registered_page, "source_logical_page": logical, "source_page": physical,
                "value_inputs_used": False, "column_count": 13, "month_column_count": 12, "rule_count": 14,
            },
        })
    record = {
        "repayment_id": f"{grid}:{year:04d}-{month:02d}", "account_id": "credit_account:credit_card:1",
        "grid_id": grid, "performance_month": f"{year:04d}-{month:02d}", "year": year, "month": month,
        "status": published_status, "overdue_amount": published_amount, "source_cell_refs": refs,
    }
    return context, record, {"result": result, "tokens": tokens, "boxes": boxes, "month": month, "logical": logical, "physical": physical}


def _page_evidence(case: dict[str, Any], values: dict[str, tuple[str, float]]) -> dict[str, Any]:
    lines = []
    for index, (field, (text, confidence)) in enumerate(values.items()):
        row = 1 if field == "status" else 2
        box = case["boxes"][row][case["month"]]
        lines.append({
            "text": text, "confidence": confidence, "bbox": [box[0] + 3, box[1] + 3, box[2] - 3, box[3] - 3],
            "evidence_ids": [f"personal_detail_page_reocr:monthly-excerpt:w{index}"], "source": "personal_detail_page_reocr_once",
        })
    return {
        "page": case["logical"], "source_page": case["physical"], "page_key": "monthly-excerpt",
        "coordinate_system": "pdf_points_top_left", "lines": lines,
    }


def _prepare(context: PersonalDetailExtractionContext, records: list[dict[str, Any]], page: dict[str, Any] | None) -> list[set[int]]:
    calls = []

    def load(pages, *, reason):
        assert reason == "business_field_context_rich_reocr_required"
        calls.append(set(pages))
        return [deepcopy(page)] if page is not None else []

    context.full_page_ocr_evidence = load
    context.prepare_candidate_b_business_repair({"repayment_records": records})
    return calls


@pytest.mark.parametrize("preserve_headers", (False, True))
@pytest.mark.parametrize("transformed", (False, True))
def test_weak_legal_m_is_planned_before_one_page_ocr_and_only_status_changes(preserve_headers: bool, transformed: bool) -> None:
    context, record, case = _printed_monthly_case(preserve_headers=preserve_headers, transformed=transformed)
    before = case["result"].model_dump()
    original_amount_ref = deepcopy(record["source_cell_refs"][1])
    page = _page_evidence(case, {"status": ("N", 0.99), "overdue_amount": ("999", 0.99)})
    assert _prepare(context, [record], page) == [{19}]
    plan = context._business_repair_plan
    assert len(plan.field_repairs) == 1
    repair = plan.field_repairs[0]
    assert repair.field_name == "status" and repair.observed_value == "M"
    assert "low_source_ocr_confidence" in repair.reason_codes
    assert repair.source_refs[0]["source_ocr_confidence"] == 0.0767
    assert plan.reconstruction_evidence == {}
    assert plan.page_decisions[0]["page_reconstruction"] is False

    fixed = apply_planned_monthly_field_repairs(context, {"repayment_records": [record]})["repayment_records"][0]
    assert fixed["status"] == "N" and fixed["overdue_amount"] == "0"
    assert next(ref for ref in fixed["source_cell_refs"] if ref["field_name"] == "overdue_amount") == original_amount_ref
    audit = apply_candidate_b_native_status_conflict_guard(context, [fixed], enabled=True)
    assert fixed["status"] == "N"
    assert audit["independent_monthly_field_confirmations"] == 1 and audit["conflicts_withheld"] == 0
    assert case["result"].model_dump() == before
    assert record["status"] == "M"


@pytest.mark.parametrize("source_confidence", (0.72, 0.99))
def test_good_legal_m_stays_m_and_does_not_request_ocr(source_confidence: float) -> None:
    context, record, case = _printed_monthly_case(source_confidence=source_confidence, published_amount=0)
    assert _prepare(context, [record], _page_evidence(case, {"status": ("N", 0.99)})) == []
    assert not context._business_repair_plan.field_repairs
    audit = apply_candidate_b_native_status_conflict_guard(context, [record], enabled=True)
    assert record["status"] == "M" and record["overdue_amount"] == 0
    assert audit["low_source_ocr_confidence_withheld"] == 0


def test_independent_same_m_confirms_confidence_without_claiming_a_value_edit() -> None:
    context, record, case = _printed_monthly_case()
    _prepare(context, [record], _page_evidence(case, {"status": ("M", 0.98)}))
    fixed = apply_planned_monthly_field_repairs(context, {"repayment_records": [record]})["repayment_records"][0]
    assert fixed["status"] == "M"
    assert context._ocr_correction_overlay.audit()["confirmed_count"] == 1
    assert context._ocr_correction_overlay.audit()["applied_count"] == 0
    audit = apply_candidate_b_native_status_conflict_guard(context, [fixed], enabled=True)
    assert audit["independent_monthly_field_confirmations"] == 1
    assert fixed["status"] == "M"


@pytest.mark.parametrize("failure", ("failed_ocr", "weak", "wrong_page", "wrong_source_page", "neighbor", "contradiction", "forged_audit"))
def test_unconfirmed_weak_m_is_explicit_and_amount_survives(failure: str) -> None:
    context, record, case = _printed_monthly_case()
    page = _page_evidence(case, {"status": ("N", 0.99)})
    if failure == "failed_ocr":
        page = None
    elif failure == "weak":
        page["lines"][0]["confidence"] = 0.61
    elif failure == "wrong_page":
        page["page"] = 18
    elif failure == "wrong_source_page":
        page["source_page"] = 21
    elif failure == "neighbor":
        page["lines"][0]["bbox"] = [value + 27 if index % 2 == 0 else value for index, value in enumerate(page["lines"][0]["bbox"])]
    elif failure == "contradiction":
        other = deepcopy(page["lines"][0])
        other.update(text="M", evidence_ids=["personal_detail_page_reocr:monthly-excerpt:w99"])
        page["lines"].append(other)
    elif failure == "forged_audit":
        record["audit"] = {"monthly_field_repairs": [{"corrected": "N", "selected_ocr_confidence": 0.99}]}
        page = None
    _prepare(context, [record], page)
    fixed = apply_planned_monthly_field_repairs(context, {"repayment_records": [record]})["repayment_records"][0]
    audit = apply_candidate_b_native_status_conflict_guard(context, [fixed], enabled=True)
    assert "status" not in fixed and fixed["overdue_amount"] == "0"
    assert audit["low_source_ocr_confidence_withheld"] == 1
    issue = next(row for row in context._personal_detail_extraction_issues if row["issue_code"] == "candidate_b_monthly_source_ocr_confidence_unresolved")
    assert issue["field_name"] == "status_code" and issue["target_record_id"] == record["repayment_id"]
    assert issue["observed_value"]["source_ocr_confidence"] == 0.0767


@pytest.mark.parametrize("field", ("status", "overdue_amount"))
@pytest.mark.parametrize("transformed", (False, True))
def test_blank_monthly_slot_is_authenticated_without_fake_token_ids_and_materialized(field: str, transformed: bool) -> None:
    kwargs = {"native_status": "", "published_status": None} if field == "status" else {"native_status": "N", "published_status": "N", "native_amount": "", "published_amount": None}
    context, record, case = _printed_monthly_case(**kwargs, transformed=transformed, source_confidence=0.99)
    # Reproduce one surviving exact locator. Deriving the other field must use
    # its own row/bbox, not simply rename this witness's field_name.
    witness = "overdue_amount" if field == "status" else "status"
    record["source_cell_refs"] = [ref for ref in record["source_cell_refs"] if ref["field_name"] == witness]
    slots = authenticated_monthly_field_slots(context, record)
    assert slots[field]["evidence_ids"] == []
    assert resolve_sealed_monthly_field_slot(context.parse_result, slots[field]) is not None
    wanted = "N" if field == "status" else "0"
    assert _prepare(context, [record], _page_evidence(case, {field: (wanted, 0.99)})) == [{19}]
    fixed = apply_planned_monthly_field_repairs(context, {"repayment_records": [record]})["repayment_records"][0]
    assert fixed[field] == wanted
    repaired_ref = next(ref for ref in fixed["source_cell_refs"] if ref["field_name"] == field)
    assert repaired_ref["bbox"] == slots[field]["registered_bbox"]
    assert repaired_ref["row"] == (4 if field == "status" else 5)
    assert repaired_ref["row"] != record["source_cell_refs"][0]["row"]


@pytest.mark.parametrize("native_status,confidence", (("N", 0.99), ("M", 0.0767)))
def test_singleton_year_and_missing_amount_target_fields_independently(native_status: str, confidence: float) -> None:
    context, record, case = _printed_monthly_case(native_status=native_status, published_status=native_status, source_confidence=confidence, native_amount="", published_amount=None, year_span=1)
    fields = {"overdue_amount": ("0", 0.99)}
    if native_status == "M":
        fields["status"] = ("N", 0.99)
    assert _prepare(context, [record], _page_evidence(case, fields)) == [{19}]
    assert {repair.field_name for repair in context._business_repair_plan.field_repairs} == set(fields)
    fixed = apply_planned_monthly_field_repairs(context, {"repayment_records": [record]})["repayment_records"][0]
    assert fixed["status"] == "N" and fixed["overdue_amount"] == "0"


@pytest.mark.parametrize("plane", ("low_level", "independent_page"))
@pytest.mark.parametrize("confidence", (0.60, 0.99))
def test_previously_withheld_exact_source_conflict_keeps_specific_diagnostic(plane: str, confidence: float) -> None:
    raw = "N" if plane == "low_level" else "#"
    other = "M" if plane == "low_level" else "*"
    context, record, _case = _printed_monthly_case(native_status=raw, published_status=None, source_confidence=confidence, year=2022, month=8)
    if plane == "low_level":
        record["audit"] = {"reason": "corrected_status_planes_disagree", "observations": {"fallback": [other], "exact_source_cell": [raw], "ordinary": []}}
    else:
        native, corrected = deepcopy(record), record
        native["status"], corrected["status"] = raw, other
        _withhold_repayment_plane_conflicts(context, {"repayment_records": [native]}, {"repayment_records": [corrected]})
        assert "status" not in corrected
    audit = apply_candidate_b_native_status_conflict_guard(context, [record], enabled=True)
    issue = next(row for row in context._personal_detail_extraction_issues if row["issue_code"] == "candidate_b_native_source_cell_repayment_status_conflict")
    assert issue["observed_value"]["corrected_final"] == other
    assert issue["observed_value"]["sealed_native_source_cell"] == raw
    assert issue["observed_value"]["paired_status_amount"] == "0"
    assert issue["observed_value"]["corrected_final_already_withheld"] is True
    assert audit["preserved_source_plane_conflicts"] == 1
    assert "status" not in record and record["overdue_amount"] == "0"


@pytest.mark.parametrize("mutation", ("raw_physical_page", "canonical_physical_page", "ref_origin", "header_ids", "wrong_token_page", "duplicate_raw_token", "conflicting_cross_view_confidence"))
def test_monthly_target_rejects_malformed_source_ownership(mutation: str) -> None:
    context, record, case = _printed_monthly_case(preserve_headers=True)
    raw_table = case["result"].pages[0].tables[0]
    year_token = next(token for token in case["tokens"] if token["content"] == str(record["year"]))
    if mutation == "raw_physical_page":
        raw_table.metadata["source_page"] = 21
    elif mutation == "canonical_physical_page":
        context.pages[0].tables[0].metadata["source_page"] = 21
    elif mutation == "ref_origin":
        for ref in record["source_cell_refs"]:
            ref["geometry_provenance"]["source_logical_page"] = 18
    elif mutation == "header_ids":
        raw_table.metadata["geometry"]["cell_evidence_ids"][0] = [[] for _ in range(13)]
    elif mutation == "wrong_token_page":
        for atom in context.parse_result.evidence_plane.evidence.text_atoms:
            if atom.source_refs == [year_token["token_id"]] and "token" in atom.source_kind:
                atom.page_id = "page:0018"
    else:
        bundle = context.parse_result.entities.domain_specific["_page_evidence_bundles"][0]
        if mutation == "duplicate_raw_token":
            bundle["micro_grid_evidence"]["tokens"].append(deepcopy(year_token))
        else:
            next(token for token in bundle["local_structure_evidence"]["tokens"] if token["token_id"] == year_token["token_id"])["confidence"] = 0.1
    assert authenticated_monthly_field_slots(context, record) == {}
    plan = BusinessUncertaintyRepairCoordinator(context.parse_result, monthly_context=context).plan({"repayment_records": [record]}, canonical_audit={"unresolved_pages": []})
    assert not plan.field_repairs


@pytest.mark.parametrize("change", ("other_account", "unrelated_record", "different_published_value", "two_owners"))
def test_monthly_directive_cannot_migrate_to_another_owner_or_new_value(change: str) -> None:
    context, record, case = _printed_monthly_case()
    _prepare(context, [record], _page_evidence(case, {"status": ("N", 0.99)}))
    final = deepcopy(record)
    records = [final]
    if change == "other_account":
        final["account_id"] = "credit_account:credit_card:2"
    elif change == "unrelated_record":
        final["repayment_id"] = "some-unrelated-manual-record"
    elif change == "different_published_value":
        final["status"] = "#"
    else:
        other = deepcopy(final)
        other["account_id"] = "credit_account:credit_card:2"
        records.append(other)
    fixed = apply_planned_monthly_field_repairs(context, {"repayment_records": records})["repayment_records"]
    assert fixed == records


def test_generated_grid_name_can_change_without_broadening_account_month_scope() -> None:
    context, record, case = _printed_monthly_case()
    _prepare(context, [record], _page_evidence(case, {"status": ("N", 0.99)}))
    final = deepcopy(record)
    final["grid_id"] = "mg_p19_rebuilt_2"
    final["repayment_id"] = "mg_p19_rebuilt_2:2019-07"
    for ref in final["source_cell_refs"]:
        ref["grid_id"] = final["grid_id"]
    fixed = apply_planned_monthly_field_repairs(context, {"repayment_records": [final]})["repayment_records"][0]
    assert fixed["status"] == "N" and fixed["repayment_id"] == final["repayment_id"]


def test_already_reconstructed_n_is_confirmed_only_when_approved_same_slot_ocr_agrees() -> None:
    context, record, case = _printed_monthly_case()
    _prepare(context, [record], _page_evidence(case, {"status": ("N", 0.99)}))
    rebuilt = deepcopy(record)
    rebuilt["status"] = "N"
    fixed = apply_planned_monthly_field_repairs(context, {"repayment_records": [rebuilt]})["repayment_records"][0]
    assert fixed["status"] == "N" and fixed["overdue_amount"] == "0"
    overlay_audit = context._ocr_correction_overlay.audit()
    assert overlay_audit["confirmed_count"] == 1 and overlay_audit["applied_count"] == 0
    guard = apply_candidate_b_native_status_conflict_guard(context, [fixed], enabled=True)
    assert guard["independent_monthly_field_confirmations"] == 1
    assert fixed["status"] == "N"


def test_existing_amount_locator_keeps_its_own_detector_row_and_never_status_token_ids() -> None:
    context, record, case = _printed_monthly_case(native_status="N", published_status="N", source_confidence=0.99, native_amount="", published_amount=None)
    record["source_cell_refs"][0]["evidence_ids"] = ["status-only-provenance"]
    record["source_cell_refs"][1]["row"] = 9
    slots = authenticated_monthly_field_slots(context, record)
    assert slots["overdue_amount"]["registered_source_ref"]["row"] == 9
    assert "status-only-provenance" not in repr(slots["overdue_amount"]["registered_source_ref"])
    _prepare(context, [record], _page_evidence(case, {"overdue_amount": ("0", 0.99)}))
    fixed = apply_planned_monthly_field_repairs(context, {"repayment_records": [record]})["repayment_records"][0]
    assert fixed["overdue_amount"] == "0"
    amount_ref = next(ref for ref in fixed["source_cell_refs"] if ref["field_name"] == "overdue_amount")
    assert amount_ref["row"] == 9 and amount_ref["bbox"] == slots["overdue_amount"]["registered_bbox"]


@pytest.mark.parametrize("status,amount,new_status,new_amount", (("1", "150", "N", "150"), (None, None, "1", "150"), ("1", "15", "1", "150")))
def test_final_derived_overdue_view_tracks_repaired_or_withheld_months(status, amount, new_status, new_amount) -> None:
    account = {"account_id": "credit_account:credit_card:1", "account_type": "credit_card"}
    row = {"account_id": account["account_id"], "year": 2019, "month": 7, "status": status, "overdue_amount": amount}
    payload = {"credit_accounts": [account], "repayment_records": [row], "overdue_records": derive_candidate_b_overdue_records([account], [row])}
    before = _overdue_view_input_basis(payload)
    row.update(status=new_status, overdue_amount=new_amount)
    _refresh_final_overdue_view(payload, before)
    if new_status == "N":
        assert payload["overdue_records"] == []
    else:
        assert len(payload["overdue_records"]) == 1
        assert payload["overdue_records"][0]["overdue_amount"] == "150"
    before = _overdue_view_input_basis(payload)
    row.pop("status")
    _refresh_final_overdue_view(payload, before)
    assert payload["overdue_records"] == []


def test_monthly_source_threshold_is_the_existing_page_reading_threshold_not_geometry_score() -> None:
    assert MONTHLY_SOURCE_OCR_CONFIDENCE_MINIMUM == 0.72


@pytest.mark.parametrize("raw_row,col,state,expected", (
    (0, 0, "missing", True),
    (0, 0, "derived", True),
    (0, 6, "missing", False),
    (0, 6, "derived", False),
    (1, 0, "missing", False),
))
def test_preserved_header_ignores_unused_label_but_keeps_month_and_year_proofs(raw_row: int, col: int, state: str, expected: bool) -> None:
    context, record, case = _printed_monthly_case(preserve_headers=True)
    geometry = case["result"].pages[0].tables[0].metadata["geometry"]
    geometry["cell_geometry_status"][raw_row][col] = state
    if state == "missing":
        geometry["cell_bboxes"][raw_row][col] = None
    slots = authenticated_monthly_field_slots(context, record)
    assert bool(slots) is expected
    plan = BusinessUncertaintyRepairCoordinator(context.parse_result, monthly_context=context).plan(
        {"repayment_records": [record]}, canonical_audit={"unresolved_pages": []},
    )
    assert bool(plan.field_repairs) is expected
    if expected:
        assert [repair.field_name for repair in plan.field_repairs] == ["status"]
        assert slots["status"]["monthly_slot_proof"]["year_row"] == 1


@pytest.mark.parametrize("preserved", (False, True))
def test_weak_native_digit_with_zero_amount_cannot_be_a_conflict_witness(preserved: bool) -> None:
    context, record, _case = _printed_monthly_case(native_status="1", published_status=None if preserved else "N", source_confidence=0.1)
    if preserved:
        record["audit"] = {"reason": "corrected_status_planes_disagree", "observations": {"fallback": ["N"], "exact_source_cell": ["1"]}}
    audit = apply_candidate_b_native_status_conflict_guard(context, [record], enabled=True)
    assert audit["conflicts_withheld"] == 0
    assert not any(row["issue_code"] == "candidate_b_native_source_cell_repayment_status_conflict" for row in context._personal_detail_extraction_issues)
    if not preserved:
        assert record["status"] == "N"
        assert audit["native_numeric_witnesses_rejected_for_nonpositive_amount"] == 1
