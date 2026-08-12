from types import SimpleNamespace

from docmirror.plugins.credit_report.personal_detail_scanned import native_extraction
from docmirror.plugins.credit_report.personal_detail_scanned.consistency_ledger import (
    apply_document_consistency_ledger,
)


def _row_ref(source: str, *, top: float = 100.0) -> dict[str, object]:
    return {
        "source": source,
        "logical_page": 26,
        "source_page": 13,
        "table_id": "pt_26_0" if source == "native_detail_table" else "",
        "row": 4,
        "bbox": [50.0, top, 390.0, top + 12.0],
        "geometry_scope": "row",
        "evidence_ids": [f"{source}:row:4"],
    }


def _unresolved_context(*, top: float = 100.0) -> SimpleNamespace:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    native_extraction._record_inquiry_ordinal_unresolved(
        context,
        inquiry_type="institution",
        source_ref=_row_ref(
            "candidate_b_canonical_inquiry_line",
            top=top,
        ),
        observed_row={
            "raw_sequence": None,
            "inquiry_date": "2022-12-13",
            "institution": "浙江网商银行股份有限公司",
            "reason": "贷后管理",
        },
        repair_kind="multiple_missing",
    )
    return context


def _emitted_record(sequence: int, *, top: float = 100.0) -> dict[str, object]:
    return {
        "inquiry_id": f"credit_inquiry:test:{sequence}",
        "sequence": sequence,
        "inquiry_type": "institution",
        "inquiry_date": "2022-12-13",
        "institution": "浙江网商银行股份有限公司",
        "reason": "贷后管理",
        "source_refs": [_row_ref("native_detail_table", top=top)],
    }


def test_unique_exact_cross_plane_row_closes_false_non_emission() -> None:
    context = _unresolved_context()

    native_extraction._reconcile_unresolved_inquiry_issues_to_emitted_records(
        context,
        [_emitted_record(5)],
    )

    assert len(context._personal_detail_extraction_issues) == 1
    issue = context._personal_detail_extraction_issues[0]
    assert (
        issue["issue_code"]
        == "candidate_b_inquiry_unresolved_sequence_reconciled_to_emitted_record"
    )
    assert issue["status"] == "resolved"
    assert issue["target_record_id"] == "credit_inquiry:test:5"
    assert issue["candidate_value"] == {"normalized_sequence": 5}
    assert "record_emitted" in issue["reason_codes"]
    assert "record_not_emitted" not in issue["reason_codes"]


def test_blank_ordinal_keeps_an_exact_three_business_cell_row_ref() -> None:
    table = _exact_inquiry_table(
        [
            ["编号", "查询日期", "查询机构", "查询原因"],
            ["", "2022.12.13", "浙江网商银行股份有限公司", "贷后管理"],
        ]
    )
    geometry = table.metadata["geometry"]
    geometry["cell_bboxes"][1][0] = None
    geometry["cell_evidence_ids"][1][0] = []
    page = SimpleNamespace(page_number=26, source_page_number=13)

    assert (
        native_extraction._exact_native_inquiry_row_ref(
            page,
            table,
            row_index=1,
        )
        is None
    )
    business_ref = native_extraction._exact_native_inquiry_business_row_ref(
        page,
        table,
        row_index=1,
    )
    assert business_ref is not None
    assert business_ref["geometry_scope"] == "row"
    assert business_ref["bbox"] == [130.0, 34.0, 400.0, 48.0]
    assert len(business_ref["evidence_ids"]) == 3


def test_missing_or_non_unique_cross_plane_row_stays_actionable() -> None:
    missing = _unresolved_context()
    native_extraction._reconcile_unresolved_inquiry_issues_to_emitted_records(
        missing,
        [],
    )
    assert missing._personal_detail_extraction_issues[0]["status"] == "requires_review"
    assert "record_not_emitted" in missing._personal_detail_extraction_issues[0][
        "reason_codes"
    ]

    ambiguous = _unresolved_context()
    native_extraction._reconcile_unresolved_inquiry_issues_to_emitted_records(
        ambiguous,
        [_emitted_record(5), _emitted_record(6)],
    )
    assert ambiguous._personal_detail_extraction_issues[0]["status"] == "requires_review"
    assert "record_not_emitted" in ambiguous._personal_detail_extraction_issues[0][
        "reason_codes"
    ]

    mismatched_type = _unresolved_context()
    personal = _emitted_record(5)
    personal["inquiry_type"] = "personal"
    native_extraction._reconcile_unresolved_inquiry_issues_to_emitted_records(
        mismatched_type,
        [personal],
    )
    assert (
        mismatched_type._personal_detail_extraction_issues[0]["status"]
        == "requires_review"
    )
    assert "record_not_emitted" in mismatched_type._personal_detail_extraction_issues[
        0
    ]["reason_codes"]


def _exact_inquiry_table(rows: list[list[str]]) -> SimpleNamespace:
    row_bands = [
        {"index": index, "y0": 20.0 + index * 14.0, "y1": 34.0 + index * 14.0}
        for index in range(len(rows))
    ]
    column_bands = [
        {"index": index, "x0": 40.0 + index * 90.0, "x1": 130.0 + index * 90.0}
        for index in range(4)
    ]
    cell_bboxes = [
        [
            [column["x0"], row["y0"], column["x1"], row["y1"]]
            for column in column_bands
        ]
        for row in row_bands
    ]
    return SimpleNamespace(
        table_id="pt_26_0",
        bbox=[40.0, 20.0, 400.0, row_bands[-1]["y1"]],
        confidence=0.95,
        rows=[],
        metadata={
            "raw_rows": rows,
            "geometry": {
                "coordinate_system": "pdf_points_top_left",
                "row_bands": row_bands,
                "col_bands": column_bands,
                "cell_bboxes": cell_bboxes,
                "cell_geometry_status": [["exact"] * 4 for _row in rows],
                "cell_evidence_ids": [
                    [
                        [f"ocr:sp0013:lp0026:r{row}:c{column}"]
                        for column in range(4)
                    ]
                    for row in range(len(rows))
                ],
                "cell_spans": [],
            },
        },
    )


def test_final_inquiry_fields_withhold_leading_han_and_bound_reason_residue() -> None:
    table = _exact_inquiry_table(
        [
            ["编号", "查询日期", "查询机构", "查询原因"],
            ["1", "2022.04.20", "讯 广州广汽租赁有限公司 2", "公 贷后管理"],
        ]
    )
    page = SimpleNamespace(
        page_number=26,
        source_page_number=13,
        canonical_template_id="annotations_and_inquiries",
        tables=[table],
        texts=[],
    )
    context = SimpleNamespace(
        pages=[page],
        corrected_evidence_pages=lambda: [],
    )

    records = native_extraction._extract_inquiries(context)

    assert len(records) == 1
    record = records[0]
    assert record["institution"] is None
    assert record["canonical_raw"]["institution"] == [
        "讯 广州广汽租赁有限公司 2"
    ]
    assert record["reason"] == "贷后管理"
    assert record["source_reason"] == "公 贷后管理"
    active_institution = next(
        issue
        for issue in context._personal_detail_extraction_issues
        if issue["issue_code"]
        == "candidate_b_inquiry_institution_leading_boundary_ambiguous"
    )
    assert active_institution["status"] == "requires_review"
    assert active_institution["field_name"] == "institution"
    assert "normalized_value_withheld" in active_institution["reason_codes"]
    reason_correction = next(
        issue
        for issue in context._personal_detail_extraction_issues
        if issue["issue_code"]
        == "candidate_b_inquiry_reason_edge_residue_corrected"
    )
    assert reason_correction["status"] == "resolved"
    assert reason_correction["candidate_value"] == {
        "normalized_reason": "贷后管理"
    }

    apply_document_consistency_ledger(
        context,
        {"inquiry_records": records},
    )
    active_institution_issues = [
        issue
        for issue in context._personal_detail_extraction_issues
        if issue.get("target_record_id") == record["inquiry_id"]
        and issue.get("field_name") == "institution"
        and issue.get("status") == "requires_review"
    ]
    assert len(active_institution_issues) == 1


def test_same_row_clean_line_cannot_override_dropped_leading_han_institution() -> None:
    noisy_institution = "守 中信银行股份有限公司个人信贷部"
    clean_institution = "中信银行股份有限公司个人信贷部"
    table = _exact_inquiry_table(
        [
            ["编号", "查询日期", "查询机构", "查询原因"],
            ["69", "2022.10.11", noisy_institution, "贷后管理"],
        ]
    )
    page = SimpleNamespace(
        page_number=26,
        source_page_number=13,
        canonical_template_id="annotations_and_inquiries",
        tables=[table],
        texts=[],
    )
    corrected_page = {
        "page": 26,
        "source_page": 13,
        "canonical_template_id": "annotations_and_inquiries",
        "lines": [
            {
                "text": f"69 2022.10.11 {clean_institution} 贷后管理",
                "bbox": [50.0, 35.0, 390.0, 47.0],
                "evidence_ids": ["personal_detail_page_reocr:source:13:w69"],
                "confidence": 0.8,
            }
        ],
    }
    context = SimpleNamespace(
        pages=[page],
        corrected_evidence_pages=lambda: [corrected_page],
    )

    records = native_extraction._extract_inquiries(context)

    assert len(records) == 1
    record = records[0]
    assert record["sequence"] == 69
    assert record["institution"] is None
    apply_document_consistency_ledger(
        context,
        {"inquiry_records": records},
    )
    assert record["institution"] is None
    conflicts = [
        issue
        for issue in context._personal_detail_extraction_issues
        if issue.get("target_record_id") == record["inquiry_id"]
        and issue.get("field_name") == "institution"
        and issue.get("status") == "requires_review"
    ]
    assert len(conflicts) == 1
    assert conflicts[0]["issue_code"] == "candidate_b_inquiry_field_conflict"
    assert conflicts[0]["observed_value"] == [
        noisy_institution,
        clean_institution,
    ]
    assert conflicts[0]["reason_codes"] == [
        "same_printed_inquiry_row",
        "equal_field_provenance",
        "conflicting_values_withheld",
    ]


def test_different_row_clean_line_cannot_authorize_or_create_field_conflict() -> None:
    noisy_institution = "守 中信银行股份有限公司个人信贷部"
    clean_institution = "中信银行股份有限公司个人信贷部"
    table = _exact_inquiry_table(
        [
            ["编号", "查询日期", "查询机构", "查询原因"],
            ["69", "2022.10.11", noisy_institution, "贷后管理"],
        ]
    )
    page = SimpleNamespace(
        page_number=26,
        source_page_number=13,
        canonical_template_id="annotations_and_inquiries",
        tables=[table],
        texts=[],
    )
    context = SimpleNamespace(
        pages=[page],
        corrected_evidence_pages=lambda: [
            {
                "page": 26,
                "source_page": 13,
                "canonical_template_id": "annotations_and_inquiries",
                "lines": [
                    {
                        "text": f"69 2022.10.11 {clean_institution} 贷后管理",
                        "bbox": [50.0, 70.0, 390.0, 82.0],
                        "evidence_ids": [
                            "personal_detail_page_reocr:source:13:w69"
                        ],
                        "confidence": 0.8,
                    }
                ],
            }
        ],
    )

    records = native_extraction._extract_inquiries(context)

    assert len(records) == 1
    assert records[0]["institution"] is None
    active_institution_issues = [
        issue
        for issue in context._personal_detail_extraction_issues
        if issue.get("target_record_id") == records[0]["inquiry_id"]
        and issue.get("field_name") == "institution"
        and issue.get("status") == "requires_review"
    ]
    assert len(active_institution_issues) == 1
    assert (
        active_institution_issues[0]["issue_code"]
        == "candidate_b_inquiry_institution_leading_boundary_ambiguous"
    )


def test_final_inquiry_boundary_issue_cannot_be_overridden_by_root_witnesses() -> None:
    target_id = "credit_inquiry:target"
    authoritative = {
        "issue_code": "candidate_b_inquiry_institution_leading_boundary_ambiguous",
        "status": "requires_review",
        "target_dataset": "inquiry_records",
        "target_record_id": target_id,
        "field_name": "institution",
        "reason_codes": ("normalized_value_withheld",),
    }
    context = SimpleNamespace(_personal_detail_extraction_issues=[authoritative])
    datasets = {
        "inquiry_records": [
            {
                "record_id": target_id,
                "inquiry_id": target_id,
                "institution": None,
                "canonical_raw": {
                    "institution": "福 中国建设银行股份有限公司北京市分行"
                },
                "source_refs_by_field": {"institution": [_row_ref("native_detail_table")]},
            },
            {
                "record_id": "credit_inquiry:witness",
                "inquiry_id": "credit_inquiry:witness",
                "institution": "中国建设银行股份有限公司福州城东支行",
                "source_refs_by_field": {
                    "institution": [
                        {
                            **_row_ref("native_detail_table", top=130.0),
                            "logical_page": 27,
                            "source_page": 14,
                        }
                    ]
                },
            },
        ],
        "credit_accounts": [
            {
                "record_id": "credit_account:witness",
                "account_id": "credit_account:witness",
                "account_type": "credit_card",
                "management_institution": (
                    "中国建设银行股份有限公司福建自贸试验区福州片区分行"
                ),
                "source_refs_by_field": {
                    "management_institution": [
                        {
                            **_row_ref("native_detail_table", top=160.0),
                            "logical_page": 23,
                            "source_page": 12,
                        }
                    ]
                },
            }
        ],
    }

    audit = apply_document_consistency_ledger(context, datasets)

    assert datasets["inquiry_records"][0].get("institution") is None
    assert context._personal_detail_extraction_issues == [authoritative]
    assert audit["institution_prefix_issue_reused"] == 1
    assert audit["institution_prefix_unresolved"] == 1
    assert audit.get("institution_prefix_resolved", 0) == 0


def test_reason_token_inside_damaged_long_reason_is_not_shortened() -> None:
    assert (
        native_extraction._bounded_canonical_inquiry_reason(
            "法人代表、费责人、高管等资信审查"
        )
        is None
    )
    assert (
        native_extraction._bounded_canonical_inquiry_reason("贷后管理 司 %5")
        == "贷后管理"
    )


def test_bounded_inquiry_date_accepts_one_missing_printed_separator() -> None:
    assert native_extraction._bounded_inquiry_date("2022.1214") == "2022-12-14"
    assert native_extraction._bounded_inquiry_date("20221214") == "2022-12-14"
    assert native_extraction._bounded_inquiry_date("2022.1314") is None
    assert native_extraction._bounded_inquiry_date("12022.1214") is None
