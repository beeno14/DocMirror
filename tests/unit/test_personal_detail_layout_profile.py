from types import SimpleNamespace

import pytest

from docmirror.plugins.credit_report.personal_detail_scanned.layout_profile import (
    InquiryLocalRepairProof,
    InquiryNumberingModel,
    LayoutCapability,
    detect_pboc_layout_profile,
    exact_inquiry_header_owner,
)

_ROLES = {
    "sequence": "编号",
    "inquiry_date": "查询日期",
    "institution": "查询机构",
    "reason": "查询原因",
}


def _geometry(rows: list[list[str]], *, owner: str) -> dict:
    width = len(rows[0])
    return {
        "row_bands": [
            {"index": row, "y0": float(row), "y1": float(row + 1)}
            for row in range(len(rows))
        ],
        "col_bands": [
            {"index": column, "x0": float(column), "x1": float(column + 1)}
            for column in range(width)
        ],
        "cell_bboxes": [
            [[float(column), float(row), float(column + 1), float(row + 1)] for column in range(width)]
            for row in range(len(rows))
        ],
        "cell_geometry_status": [["exact"] * width for _row in rows],
        "cell_evidence_ids": [
            [[f"{owner}:{row}:{column}"] for column in range(width)]
            for row in range(len(rows))
        ],
        "cell_spans": [],
    }


def _table(
    *,
    owner: str,
    order: tuple[str, ...] = ("sequence", "inquiry_date", "institution", "reason"),
    inquiry_type: str = "institution",
    start: int = 1,
    count: int = 3,
) -> SimpleNamespace:
    header = [_ROLES[role] for role in order]
    semantic_rows: list[dict[str, str]] = []
    for sequence in range(start, start + count):
        semantic_rows.append(
            {
                "sequence": str(sequence),
                "inquiry_date": f"2025-01-{sequence:02d}",
                "institution": "本人" if inquiry_type == "personal" else f"示例银行{sequence}",
                "reason": "本人查询(互联网个人信用信息服务平台)" if inquiry_type == "personal" else "贷后管理",
            }
        )
    rows = [header, *[[row[role] for role in order] for row in semantic_rows]]
    return SimpleNamespace(
        metadata={"raw_rows": rows, "geometry": _geometry(rows, owner=owner)},
        headers=[],
        rows=[],
        bbox=[0.0, 0.0, float(len(header)), float(len(rows))],
    )


def _pages(*tables: SimpleNamespace) -> list[SimpleNamespace]:
    return [SimpleNamespace(tables=[table]) for table in tables]


def test_shared_exact_inquiry_header_owner_maps_reordered_unequal_columns() -> None:
    order = ("reason", "institution", "sequence", "inquiry_date")
    table = _table(owner="unequal", order=order, count=2)
    widths = (1.5, 4.0, 0.75, 2.25)
    lefts = [0.0]
    for width in widths:
        lefts.append(lefts[-1] + width)
    geometry = table.metadata["geometry"]
    for row_index, row in enumerate(geometry["cell_bboxes"]):
        for column, _bbox in enumerate(row):
            row[column] = [
                lefts[column],
                float(row_index),
                lefts[column + 1],
                float(row_index + 1),
            ]
    table.bbox = [0.0, 0.0, lefts[-1], 3.0]

    owner = exact_inquiry_header_owner(table)

    assert owner is not None
    assert owner.columns() == {
        "sequence": 2,
        "inquiry_date": 3,
        "institution": 1,
        "reason": 0,
    }
    assert owner.binding == "exact_single_row_header_lattice"


def test_shared_exact_inquiry_header_owner_accepts_only_exact_split_lattice() -> None:
    rows = [
        ["编号", "", "", ""],
        ["", "查询日期", "查询机构", "查询原因"],
        ["1", "2025-01-01", "本人", "本人查询(自助查询机)"],
    ]
    geometry = _geometry(rows, owner="split")
    geometry["cell_geometry_status"][1][0] = "derived"
    geometry["cell_evidence_ids"][1][0] = []
    geometry["cell_spans"] = [
        {"row": 0, "col": 0, "row_span": 2, "col_span": 1}
    ]
    table = SimpleNamespace(
        metadata={"raw_rows": rows, "geometry": geometry},
        headers=[],
        rows=[],
        bbox=[0.0, 0.0, 4.0, 3.0],
    )

    owner = exact_inquiry_header_owner(table)

    assert owner is not None
    assert owner.header_rows == (0, 1)
    assert owner.body_start == 2
    assert owner.binding == "exact_complementary_header_lattice"


def _collapsed_header_table(*, merged_label: str) -> SimpleNamespace:
    rows = [
        [merged_label, "", "查询机构", "查询原因"],
        ["1", "2025-01-01", "示例机构", "贷后管理"],
    ]
    geometry = _geometry(rows, owner="collapsed")
    geometry["cell_bboxes"][0][0] = [0.0, 0.0, 2.0, 1.0]
    geometry["cell_bboxes"][0][1] = None
    geometry["cell_geometry_status"][0][1] = "derived"
    geometry["cell_evidence_ids"][0][1] = []
    geometry["cell_spans"] = [
        {"row": 0, "col": 0, "row_span": 1, "col_span": 2}
    ]
    return SimpleNamespace(
        metadata={"raw_rows": rows, "geometry": geometry},
        headers=[],
        rows=[],
        bbox=[0.0, 0.0, 4.0, 2.0],
    )


def test_shared_exact_inquiry_header_owner_accepts_clean_collapsed_colspan() -> None:
    owner = exact_inquiry_header_owner(
        _collapsed_header_table(merged_label="编号查询日期")
    )

    assert owner is not None
    assert owner.columns() == {
        "sequence": 0,
        "inquiry_date": 1,
        "institution": 2,
        "reason": 3,
    }
    assert owner.binding == "exact_collapsed_colspan_header_lattice"


@pytest.mark.parametrize(
    "defect",
    [
        "collapsed_residue",
        "unknown_business_label",
        "residual_business_header",
        "repeated_role",
        "missing_exact_geometry",
        "missing_evidence",
        "replayed_owner",
        "multiple_headers",
    ],
)
def test_shared_exact_inquiry_header_owner_fails_closed(defect: str) -> None:
    if defect == "collapsed_residue":
        table = _collapsed_header_table(merged_label="?编号查询日期X")
    elif defect == "unknown_business_label":
        table = _table(owner="unknown", count=1)
        rows = table.metadata["raw_rows"]
        rows[0].append("业务类型")
        rows[1].append("贷款")
        table.metadata["geometry"] = _geometry(rows, owner="unknown-wide")
        table.bbox = [0.0, 0.0, 5.0, 2.0]
    elif defect == "residual_business_header":
        table = _table(owner="residual", count=1)
        table.metadata["raw_rows"].insert(0, ["业务类型", "", "", ""])
        table.metadata["geometry"] = _geometry(
            table.metadata["raw_rows"], owner="residual"
        )
        table.bbox = [0.0, 0.0, 4.0, 3.0]
    else:
        table = _table(owner=f"defect:{defect}", count=1)
        geometry = table.metadata["geometry"]
        if defect == "repeated_role":
            table.metadata["raw_rows"][0][3] = "查询机构"
        elif defect == "missing_exact_geometry":
            geometry["cell_geometry_status"][0][2] = "derived"
        elif defect == "missing_evidence":
            geometry["cell_evidence_ids"][0][2] = []
        elif defect == "replayed_owner":
            geometry["cell_evidence_ids"][0][2] = list(
                geometry["cell_evidence_ids"][0][1]
            )
        elif defect == "multiple_headers":
            header = list(table.metadata["raw_rows"][0])
            table.metadata["raw_rows"].insert(1, header)
            table.metadata["geometry"] = _geometry(
                table.metadata["raw_rows"], owner="multiple"
            )
            table.bbox = [0.0, 0.0, 4.0, 3.0]

    assert exact_inquiry_header_owner(table) is None


def test_registered_pboc_layout_maps_roles_without_granting_repair_capabilities() -> None:
    profile = detect_pboc_layout_profile(
        _pages(
            _table(owner="institution", count=5),
            _table(owner="personal", inquiry_type="personal", count=2),
        )
    )

    assert profile.pboc_family == "pboc_personal_detailed"
    assert profile.layout_revision == "unknown"
    assert profile.inquiry_schema_profile == "pboc_personal_detailed_inquiry_four_column"
    assert profile.inquiry_columns() == {
        "sequence": 0,
        "inquiry_date": 1,
        "institution": 2,
        "reason": 3,
    }
    assert profile.inquiry_numbering_model is InquiryNumberingModel.INDEPENDENT_RESTARTS
    assert not profile.capabilities
    assert not profile.allows_local_proof(LayoutCapability.COLLAPSED_HEADER)


def test_numbering_model_detects_combined_continuity_from_two_exact_subsections() -> None:
    profile = detect_pboc_layout_profile(
        _pages(
            _table(owner="institution", count=4),
            _table(owner="personal", inquiry_type="personal", start=5, count=3),
        )
    )

    assert profile.inquiry_numbering_model is InquiryNumberingModel.COMBINED_CONTINUITY


def test_numbering_model_does_not_infer_restart_from_unanchored_first_tail() -> None:
    profile = detect_pboc_layout_profile(
        _pages(
            _table(owner="institution-tail", start=122, count=8),
            _table(owner="personal", inquiry_type="personal", count=1),
        )
    )

    assert profile.inquiry_numbering_model is InquiryNumberingModel.UNKNOWN
    assert "numbering_requires_two_exact_groups" in profile.detection_reasons


def test_reordered_exact_pboc_header_keeps_semantic_map_but_no_revision_repairs() -> None:
    order = ("reason", "sequence", "institution", "inquiry_date")
    profile = detect_pboc_layout_profile(
        _pages(
            _table(owner="institution", order=order, count=3),
            _table(owner="personal", order=order, inquiry_type="personal", count=2),
        )
    )

    assert profile.pboc_family == "pboc_personal_detailed"
    assert profile.layout_revision == "unknown"
    assert profile.inquiry_schema_profile == "unregistered_semantic_role_map"
    assert profile.inquiry_columns() == {
        "sequence": 1,
        "inquiry_date": 3,
        "institution": 2,
        "reason": 0,
    }
    assert profile.inquiry_numbering_model is InquiryNumberingModel.INDEPENDENT_RESTARTS
    assert not profile.capabilities
    assert "layout_revision_not_registered" in profile.detection_reasons


@pytest.mark.parametrize("defect", ["conflicting_map", "duplicate_owner", "inexact_header"])
def test_profile_detection_fails_closed_on_conflicting_or_inexact_evidence(defect: str) -> None:
    first = _table(owner="first", count=3)
    second = _table(owner="second", inquiry_type="personal", count=2)
    if defect == "conflicting_map":
        second = _table(
            owner="second",
            order=("reason", "sequence", "institution", "inquiry_date"),
            inquiry_type="personal",
            count=2,
        )
    elif defect == "duplicate_owner":
        second.metadata["geometry"]["cell_evidence_ids"][0][0] = list(
            first.metadata["geometry"]["cell_evidence_ids"][0][0]
        )
    else:
        first.metadata["geometry"]["cell_geometry_status"][0][2] = "derived"

    profile = detect_pboc_layout_profile(_pages(first, second))

    if defect == "conflicting_map":
        assert profile.inquiry_schema_profile == "unknown"
        assert profile.inquiry_columns() == {}
        assert "conflicting_exact_inquiry_role_maps" in profile.detection_reasons
    elif defect == "duplicate_owner":
        assert profile.inquiry_numbering_model is InquiryNumberingModel.UNKNOWN
        assert "exact_header_owner_conflict" in profile.detection_reasons
    else:
        assert profile.inquiry_numbering_model is InquiryNumberingModel.UNKNOWN
        assert profile.exact_header_count == 1
    assert not profile.capabilities or defect == "inexact_header"


@pytest.mark.parametrize(
    ("second_start", "expected_reason"),
    [(2, "numbering_restart_and_continuity_not_uniquely_proven"), (8, "numbering_restart_and_continuity_not_uniquely_proven")],
)
def test_numbering_model_rejects_conflicting_restart_and_continuity_evidence(
    second_start: int,
    expected_reason: str,
) -> None:
    profile = detect_pboc_layout_profile(
        _pages(
            _table(owner="institution", count=4),
            _table(owner="personal", inquiry_type="personal", start=second_start, count=2),
        )
    )

    assert profile.inquiry_numbering_model is InquiryNumberingModel.UNKNOWN
    assert expected_reason in profile.detection_reasons


def test_numbering_model_requires_two_independently_exact_groups() -> None:
    profile = detect_pboc_layout_profile(_pages(_table(owner="one", count=7)))

    assert profile.layout_revision == "unknown"
    assert profile.inquiry_schema_profile == "pboc_personal_detailed_inquiry_four_column"
    assert profile.inquiry_numbering_model is InquiryNumberingModel.UNKNOWN
    assert "numbering_requires_two_exact_groups" in profile.detection_reasons


def test_unknown_document_has_explicit_audit_reason_and_no_repairs() -> None:
    profile = detect_pboc_layout_profile(
        [SimpleNamespace(tables=[SimpleNamespace(metadata={"raw_rows": [["任意", "表格"]]})])]
    )

    audit = profile.audit()
    assert audit["pboc_family"] == "unknown"
    assert audit["layout_revision"] == "unknown"
    assert audit["inquiry_schema_profile"] == "unknown"
    assert audit["capabilities"] == []
    assert "no_unique_exact_pboc_inquiry_header" in audit["detection_reasons"]
    assert audit["fixture_identity_used"] is False
    assert audit["ocr_used"] is False
    assert audit["section_graph_authority"] is False
    assert audit["pagination_authority"] is False
    assert audit["capabilities_require_local_proof"] is True


@pytest.mark.parametrize("capability", list(LayoutCapability))
def test_unknown_document_rejects_every_capability_without_local_proof(
    capability: LayoutCapability,
) -> None:
    profile = detect_pboc_layout_profile([])

    assert not profile.allows_local_proof(capability)


@pytest.mark.parametrize("capability", list(LayoutCapability))
def test_canonical_ordinary_header_alone_never_authorizes_repair(
    capability: LayoutCapability,
) -> None:
    profile = detect_pboc_layout_profile(_pages(_table(owner="ordinary")))

    assert not profile.allows_local_proof(capability)


def test_reordered_schema_rejects_even_a_complete_capability_local_proof() -> None:
    profile = detect_pboc_layout_profile(
        _pages(
            _table(
                owner="reordered",
                order=("reason", "sequence", "institution", "inquiry_date"),
            )
        )
    )
    proof = InquiryLocalRepairProof.create(
        LayoutCapability.COLLAPSED_HEADER,
        inquiry_role_columns={
            "reason": 0,
            "sequence": 1,
            "institution": 2,
            "inquiry_date": 3,
        },
        evidence_ids=("local:0", "local:1", "local:2", "local:3"),
        geometry_bbox=(0.0, 0.0, 4.0, 1.0),
        local_trait="exact_collapsed_header_lattice",
    )

    assert proof is not None
    assert not profile.allows_local_proof(
        LayoutCapability.COLLAPSED_HEADER,
        proof=proof,
    )


def test_reordered_schema_allows_owned_sealed_mixed_page_header_in_any_role_order() -> None:
    order = ("reason", "sequence", "institution", "inquiry_date")
    profile = detect_pboc_layout_profile(
        _pages(_table(owner="reordered-mixed", order=order))
    )
    columns = {
        "reason": 0,
        "sequence": 1,
        "institution": 2,
        "inquiry_date": 3,
    }
    proof = InquiryLocalRepairProof.create(
        LayoutCapability.MIXED_PAGE_HEADER,
        inquiry_role_columns=columns,
        evidence_ids=("mixed:0", "mixed:1", "mixed:2", "mixed:3"),
        geometry_bbox=(0.0, 0.0, 4.0, 1.0),
        local_trait="exact_mixed_page_heading_header_lattice",
        section_owner_role="annotations_and_inquiries",
    )
    mismatched = InquiryLocalRepairProof.create(
        LayoutCapability.MIXED_PAGE_HEADER,
        inquiry_role_columns={
            "sequence": 0,
            "inquiry_date": 1,
            "institution": 2,
            "reason": 3,
        },
        evidence_ids=("other:0", "other:1", "other:2", "other:3"),
        geometry_bbox=(0.0, 0.0, 4.0, 1.0),
        local_trait="exact_mixed_page_heading_header_lattice",
        section_owner_role="annotations_and_inquiries",
    )
    unowned = InquiryLocalRepairProof.create(
        LayoutCapability.MIXED_PAGE_HEADER,
        inquiry_role_columns=columns,
        evidence_ids=("unowned:0", "unowned:1", "unowned:2", "unowned:3"),
        geometry_bbox=(0.0, 0.0, 4.0, 1.0),
        local_trait="exact_mixed_page_heading_header_lattice",
    )

    assert proof is not None
    assert mismatched is not None
    assert unowned is not None
    assert profile.allows_local_proof(
        LayoutCapability.MIXED_PAGE_HEADER,
        proof=proof,
    )
    assert profile.allows_local_proof(
        LayoutCapability.MIXED_PAGE_HEADER,
        proof=mismatched,
    )
    assert not profile.allows_local_proof(
        LayoutCapability.MIXED_PAGE_HEADER,
        proof=unowned,
    )


def test_unknown_profile_rejects_even_a_complete_capability_matching_local_proof() -> None:
    profile = detect_pboc_layout_profile([])
    proof = InquiryLocalRepairProof.create(
        LayoutCapability.TWO_ROW_CELL_SPLIT,
        inquiry_role_columns={
            "sequence": 0,
            "inquiry_date": 1,
            "institution": 2,
            "reason": 3,
        },
        evidence_ids=("local:0", "local:1", "local:2", "local:3"),
        geometry_bbox=(0.0, 0.0, 4.0, 1.0),
        local_trait="exact_two_row_token_lattice",
    )

    assert proof is not None
    assert not profile.allows_local_proof(
        LayoutCapability.TWO_ROW_CELL_SPLIT,
        proof=proof,
    )
    assert not profile.allows_local_proof(
        LayoutCapability.TOKEN_BRIDGE,
        proof=proof,
    )


@pytest.mark.parametrize(
    ("evidence_ids", "geometry_bbox", "local_trait"),
    [
        ((), (0.0, 0.0, 4.0, 1.0), "exact_collapsed_header_lattice"),
        (("same", "same"), (0.0, 0.0, 4.0, 1.0), "exact_collapsed_header_lattice"),
        (("local",), (0.0, 0.0, float("nan"), 1.0), "exact_collapsed_header_lattice"),
        (("local",), (0.0, 0.0, 4.0, 1.0), "wrong_trait"),
    ],
)
def test_malformed_local_proof_cannot_be_constructed(
    evidence_ids: tuple[str, ...],
    geometry_bbox: tuple[float, float, float, float],
    local_trait: str,
) -> None:
    assert (
        InquiryLocalRepairProof.create(
            LayoutCapability.COLLAPSED_HEADER,
            inquiry_role_columns={
                "sequence": 0,
                "inquiry_date": 1,
                "institution": 2,
                "reason": 3,
            },
            evidence_ids=evidence_ids,
            geometry_bbox=geometry_bbox,
            local_trait=local_trait,
        )
        is None
    )


def test_conflicting_exact_role_maps_veto_even_complete_local_repair_attempts() -> None:
    profile = detect_pboc_layout_profile(
        _pages(
            _table(owner="canonical", count=3),
            _table(
                owner="reordered",
                order=("reason", "sequence", "institution", "inquiry_date"),
                inquiry_type="personal",
                count=2,
            ),
        )
    )

    proof = InquiryLocalRepairProof.create(
        LayoutCapability.COLLAPSED_HEADER,
        inquiry_role_columns={
            "sequence": 0,
            "inquiry_date": 1,
            "institution": 2,
            "reason": 3,
        },
        evidence_ids=("local:0",),
        geometry_bbox=(0.0, 0.0, 4.0, 1.0),
        local_trait="exact_collapsed_header_lattice",
    )
    assert proof is not None
    assert not profile.allows_local_proof(
        LayoutCapability.COLLAPSED_HEADER,
        proof=proof,
    )


def test_conflicting_table_orders_do_not_suppress_owned_mixed_header_proof() -> None:
    profile = detect_pboc_layout_profile(
        _pages(
            _table(owner="canonical", count=3),
            _table(
                owner="reordered",
                order=("reason", "sequence", "institution", "inquiry_date"),
                inquiry_type="personal",
                count=2,
            ),
        )
    )
    proof = InquiryLocalRepairProof.create(
        LayoutCapability.MIXED_PAGE_HEADER,
        inquiry_role_columns={
            "sequence": 0,
            "inquiry_date": 1,
            "institution": 2,
            "reason": 3,
        },
        evidence_ids=("mixed:0", "mixed:1", "mixed:2", "mixed:3"),
        geometry_bbox=(0.0, 0.0, 4.0, 1.0),
        local_trait="exact_mixed_page_heading_header_lattice",
        section_owner_role="annotations_and_inquiries",
    )

    assert "conflicting_exact_inquiry_role_maps" in profile.detection_reasons
    assert proof is not None
    assert profile.allows_local_proof(
        LayoutCapability.MIXED_PAGE_HEADER,
        proof=proof,
    )


def test_ordinary_canonical_schema_does_not_prove_any_local_repair_shape() -> None:
    """An official-layout header authorizes attempts, never a repair outcome."""

    ordinary = _table(owner="ordinary", count=3)
    unrelated_collapsed_candidate = SimpleNamespace(
        metadata={
            "raw_rows": [
                ["编号查询日期", "", "查询机构", "查询原因"],
                ["4", "2025-01-04", "另一银行", "贷后管理"],
            ],
            # No exact merged-cell lattice: the local collapsed-header proof
            # must reject this even though the document schema is registered.
            "geometry": {},
        },
        headers=[],
        rows=[],
    )
    profile = detect_pboc_layout_profile(_pages(ordinary, unrelated_collapsed_candidate))

    assert not profile.allows(LayoutCapability.COLLAPSED_HEADER)
    assert not profile.allows_local_proof(LayoutCapability.COLLAPSED_HEADER)
    from docmirror.plugins.credit_report.personal_detail_scanned.native_extraction import (
        _bounded_collapsed_inquiry_header_slots,
    )

    assert (
        _bounded_collapsed_inquiry_header_slots(
            unrelated_collapsed_candidate.metadata["raw_rows"],
            table=unrelated_collapsed_candidate,
        )
        is None
    )
