from __future__ import annotations

from types import SimpleNamespace

from docmirror.plugins.credit_report.personal_detail_scanned import native_extraction


def _line(
    text: str,
    bbox: list[float],
    evidence_id: str,
) -> dict[str, object]:
    return {
        "text": text,
        "bbox": bbox,
        "evidence_ids": [evidence_id],
        "confidence": 0.99,
    }


def _cell_table(
    table_id: str,
    *,
    text: str,
    bbox: list[float],
    evidence_id: str,
    status: str = "exact",
    raw_rows: list[list[str]] | None = None,
    source_logical_page: int = 1,
) -> SimpleNamespace:
    rows = raw_rows if raw_rows is not None else [[text]]
    row = next(row_index for row_index, values in enumerate(rows) if text in values)
    column = rows[row].index(text)
    bboxes = [[None for _cell in values] for values in rows]
    evidence = [[None for _cell in values] for values in rows]
    statuses = [[None for _cell in values] for values in rows]
    confidences = [[None for _cell in values] for values in rows]
    bboxes[row][column] = bbox
    evidence[row][column] = [evidence_id]
    statuses[row][column] = status
    confidences[row][column] = 0.98
    return SimpleNamespace(
        table_id=table_id,
        bbox=[0, 0, 500, 500],
        metadata={
            "canonical_template_id": "credit_account_detail",
            "source_logical_page": source_logical_page,
            "source_page": 1,
            "raw_rows": rows,
            "geometry": {
                "coordinate_system": "logical_page_pixels",
                "cell_bboxes": bboxes,
                "cell_evidence_ids": evidence,
                "cell_geometry_status": statuses,
                "cell_confidences": confidences,
            },
        },
        rows=[],
    )


def _context(
    *,
    lines: list[dict[str, object]],
    tables: list[SimpleNamespace],
) -> SimpleNamespace:
    page = SimpleNamespace(
        page_number=1,
        source_page_number=1,
        tables=tables,
    )
    return SimpleNamespace(
        pages=[page],
        corrected_evidence_pages=lambda: [{"page": 1, "source_page": 1, "lines": lines}],
        _personal_detail_extraction_issues=[],
    )


def _two_page_context(
    *,
    second_lines: list[dict[str, object]],
    second_tables: list[SimpleNamespace],
) -> SimpleNamespace:
    first_table = _cell_table(
        "page-one-table",
        text="not an anchor",
        bbox=[10, 20, 80, 30],
        evidence_id="page-one-table-evidence",
    )
    for table in second_tables:
        table.metadata["source_logical_page"] = 2
        table.metadata["source_page"] = 2
    return SimpleNamespace(
        pages=[
            SimpleNamespace(page_number=1, source_page_number=1, tables=[first_table]),
            SimpleNamespace(page_number=2, source_page_number=2, tables=second_tables),
        ],
        corrected_evidence_pages=lambda: [
            {
                "page": 1,
                "source_page": 1,
                "lines": [
                    _line("\u975e\u5faa\u73af\u8d37\u8d26\u6237", [10, 5, 200, 15], "heading"),
                    _line("\u8d26\u62371", [10, 40, 80, 60], "page-one-anchor"),
                ],
            },
            {"page": 2, "source_page": 2, "lines": second_lines},
        ],
        reading_order_by_logical={1: 1, 2: 2},
        reading_order_resolution={"resolved": True, "authoritative": True},
        _personal_detail_extraction_issues=[],
    )


def _issue_codes(context: SimpleNamespace) -> list[str]:
    return [str(issue.get("issue_code") or "") for issue in context._personal_detail_extraction_issues]


def _registered_section_context(
    evidence_pages: list[dict[str, object]],
    *,
    order: dict[int, int] | None = None,
    resolution: dict[str, object] | None = None,
) -> SimpleNamespace:
    pages = [
        SimpleNamespace(
            page_number=int(page["page"]),
            source_page_number=int(page.get("source_page") or page["page"]),
            tables=[],
        )
        for page in evidence_pages
    ]
    return SimpleNamespace(
        pages=pages,
        corrected_evidence_pages=lambda: evidence_pages,
        reading_order_by_logical=order
        or {page.page_number: index for index, page in enumerate(pages, 1)},
        reading_order_resolution=resolution
        or {
            "resolved": True,
            "authoritative": True,
            "basis": "complete_unique_printed_page_permutation",
        },
        _personal_detail_extraction_issues=[],
    )


def test_exact_table_cell_anchors_join_geometry_order_not_table_order() -> None:
    context = _context(
        lines=[_line("\u975e\u5faa\u73af\u8d37\u8d26\u6237", [10, 5, 200, 15], "heading")],
        # Deliberately reverse native encounter order.
        tables=[
            _cell_table(
                "later",
                text="\u8d26\u62372 \u4e0d\u5f97\u63d0\u5347\u4e3a\u5b57\u6bb5\u503c",
                bbox=[10, 80, 240, 100],
                evidence_id="cell-2",
            ),
            _cell_table(
                "earlier",
                text="\u8d26\u62371",
                bbox=[10, 40, 80, 60],
                evidence_id="cell-1",
            ),
        ],
    )

    rows = native_extraction._account_anchor_skeletons(context)

    assert [row["category_sequence"] for row in rows] == [1, 2]
    assert [row["account_id"] for row in rows] == [
        "credit_account:non_revolving_loan:1",
        "credit_account:non_revolving_loan:2",
    ]
    assert all(
        any(
            ref.get("source") == "native_detail_table_cell"
            and ref.get("geometry_status") == "exact"
            and ref.get("anchor_binding") == "printed_account_ordinal"
            for ref in row["source_refs"]
        )
        for row in rows
    )
    # Cell residue remains evidence only; it cannot leak into value extraction.
    assert all(
        "\u4e0d\u5f97\u63d0\u5347\u4e3a\u5b57\u6bb5\u503c" not in detail["text"]
        for row in rows
        for detail in row["raw_detail_lines"]
    )


def test_exact_line_and_cell_anchor_dedupe_only_for_unique_owner() -> None:
    anchor = _line("\u8d26\u62371", [10, 40, 80, 60], "same-anchor")
    context = _context(
        lines=[
            _line("\u975e\u5faa\u73af\u8d37\u8d26\u6237", [10, 5, 200, 15], "heading"),
            anchor,
        ],
        tables=[
            _cell_table(
                "cell-owner",
                text="\u8d26\u62371",
                bbox=[8, 38, 100, 62],
                evidence_id="same-anchor",
            )
        ],
    )

    rows = native_extraction._account_anchor_skeletons(context)

    assert len(rows) == 1
    assert rows[0]["category_sequence"] == 1
    assert rows[0]["_printed_ordinal_status"] == "printed_unique"
    assert any(ref.get("table_id") == "cell-owner" for ref in rows[0]["source_refs"])
    assert "candidate_b_account_printed_ordinal_unresolved" not in _issue_codes(context)


def test_distinct_line_and_cell_anchors_stay_duplicate_and_provisional() -> None:
    context = _context(
        lines=[
            _line("\u975e\u5faa\u73af\u8d37\u8d26\u6237", [10, 5, 200, 15], "heading"),
            _line("\u8d26\u62371", [10, 40, 80, 60], "line-owner"),
        ],
        tables=[
            _cell_table(
                "distinct-cell-owner",
                text="\u8d26\u62371",
                bbox=[10, 80, 80, 100],
                evidence_id="cell-owner",
            )
        ],
    )

    rows = native_extraction._account_anchor_skeletons(context)

    assert len(rows) == 2
    assert all("category_sequence" not in row for row in rows)
    assert all(row["_printed_ordinal_status"] == "printed_duplicate" for row in rows)
    assert len({row["account_id"] for row in rows}) == 2
    assert _issue_codes(context).count("candidate_b_account_printed_ordinal_unresolved") == 2


def test_competing_cell_owners_fail_closed_and_are_reported() -> None:
    context = _context(
        lines=[_line("\u975e\u5faa\u73af\u8d37\u8d26\u6237", [10, 5, 200, 15], "heading")],
        tables=[
            _cell_table(
                "owner-a",
                text="\u8d26\u62371",
                bbox=[10, 40, 80, 60],
                evidence_id="shared-cell-evidence",
            ),
            _cell_table(
                "owner-b",
                text="\u8d26\u62371",
                bbox=[10, 80, 80, 100],
                evidence_id="shared-cell-evidence",
            ),
        ],
    )

    rows = native_extraction._account_anchor_skeletons(context)

    assert rows == []
    assert _issue_codes(context).count("candidate_b_account_table_cell_anchor_owner_unresolved") == 2


def test_one_line_with_two_distinct_cell_owners_is_not_deduped() -> None:
    context = _context(
        lines=[
            _line("\u975e\u5faa\u73af\u8d37\u8d26\u6237", [10, 5, 200, 15], "heading"),
            _line("\u8d26\u62371", [10, 40, 80, 60], "shared-line-evidence"),
        ],
        tables=[
            _cell_table(
                "cell-a",
                text="\u8d26\u62371",
                bbox=[8, 38, 90, 62],
                evidence_id="shared-line-evidence",
            ),
            _cell_table(
                "cell-b",
                text="\u8d26\u62371",
                bbox=[8, 78, 90, 102],
                evidence_id="shared-line-evidence",
            ),
        ],
    )

    rows = native_extraction._account_anchor_skeletons(context)

    # Both cells fail unique source ownership; the line remains the sole anchor.
    assert [row.get("category_sequence") for row in rows] == [1]
    assert _issue_codes(context).count(
        "candidate_b_account_table_cell_anchor_owner_unresolved"
    ) == 2


def test_exact_cell_without_unique_family_is_withheld_and_reported() -> None:
    context = _context(
        lines=[],
        tables=[
            _cell_table(
                "unowned-family",
                text="\u8d26\u62371",
                bbox=[10, 40, 80, 60],
                evidence_id="unowned-family-evidence",
            )
        ],
    )

    rows = native_extraction._account_anchor_skeletons(context)

    assert rows == []
    assert _issue_codes(context) == ["candidate_b_account_table_cell_anchor_family_unresolved"]


def test_inexact_or_unbound_cell_cannot_enter_anchor_plane() -> None:
    inexact = _cell_table(
        "inexact",
        text="\u8d26\u62371",
        bbox=[10, 40, 80, 60],
        evidence_id="inexact-evidence",
        status="estimated",
    )
    missing_evidence = _cell_table(
        "missing-evidence",
        text="\u8d26\u62372",
        bbox=[10, 80, 80, 100],
        evidence_id="placeholder",
    )
    missing_evidence.metadata["geometry"]["cell_evidence_ids"] = [[[]]]
    context = _context(
        lines=[_line("\u975e\u5faa\u73af\u8d37\u8d26\u6237", [10, 5, 200, 15], "heading")],
        tables=[inexact, missing_evidence],
    )

    assert native_extraction._account_anchor_skeletons(context) == []
    assert context._personal_detail_extraction_issues == []


def test_exact_cell_backed_dense_next_ordinal_carries_exact_family() -> None:
    cell = _cell_table(
        "page-two-cell",
        text="\u8d26\u62372",
        bbox=[8, 38, 90, 62],
        evidence_id="page-two-anchor",
    )
    context = _two_page_context(
        second_lines=[_line("\u8d26\u62372", [10, 40, 80, 60], "page-two-anchor")],
        second_tables=[cell],
    )

    rows = native_extraction._account_anchor_skeletons(context)

    assert [row.get("category_sequence") for row in rows] == [1, 2]
    assert all(row["account_type"] == "non_revolving_loan" for row in rows)


def test_cell_backed_non_dense_ordinal_does_not_carry_family() -> None:
    cell = _cell_table(
        "page-two-gap",
        text="\u8d26\u62373",
        bbox=[8, 38, 90, 62],
        evidence_id="page-two-gap-anchor",
    )
    context = _two_page_context(
        second_lines=[
            _line("\u8d26\u62373", [10, 40, 80, 60], "page-two-gap-anchor")
        ],
        second_tables=[cell],
    )

    rows = native_extraction._account_anchor_skeletons(context)

    assert [row.get("category_sequence") for row in rows] == [1]
    assert "candidate_b_account_table_cell_anchor_family_unresolved" in _issue_codes(
        context
    )


def test_exact_cell_anchor_only_binds_values_from_its_native_table() -> None:
    skeleton = {
        "account_id": "credit_account:non_revolving_loan:1",
        "account_type": "non_revolving_loan",
        "account_family_quality": "exact",
        "category_sequence": 1,
        "_printed_ordinal_status": "printed_unique",
        "_account_anchor_origin": "exact_table_cell",
        "_account_table_cell_ref": {
            "source": "native_detail_table_cell",
            "table_id": "owner-table",
            "logical_page": 1,
            "bbox": [10, 40, 80, 60],
        },
        "page": 1,
        "bbox": [10, 40, 80, 60],
        "_canonical_segment": {
            "pages": [{"logical_page": 1, "min_y": 40, "max_y": None}]
        },
    }
    wrong_table = {
        "account_id": "table-observation:wrong",
        "account_type": "non_revolving_loan",
        "source_refs": [
            {
                "source": "native_detail_table",
                "table_id": "neighbor-table",
                "logical_page": 1,
                "bbox": [10, 70, 400, 180],
            }
        ],
    }

    assert native_extraction._match_account_table_observations(
        [skeleton], [wrong_table]
    ) == {}

    owner_table = {
        **wrong_table,
        "source_refs": [
            {
                "source": "native_detail_table",
                "table_id": "owner-table",
                "logical_page": 1,
                "bbox": [10, 70, 400, 180],
            }
        ],
    }
    assert native_extraction._match_account_table_observations(
        [skeleton], [owner_table]
    ) == {0: 0}


def test_generated_exact_cell_skeleton_keeps_native_table_value_owner() -> None:
    context = _context(
        lines=[_line("\u975e\u5faa\u73af\u8d37\u8d26\u6237", [10, 5, 200, 15], "heading")],
        tables=[
            _cell_table(
                "owner-table",
                text="\u8d26\u62371",
                bbox=[10, 40, 80, 60],
                evidence_id="owner-cell",
            )
        ],
    )
    skeletons = native_extraction._account_anchor_skeletons(context)
    assert len(skeletons) == 1

    wrong_table = {
        "account_id": "table-observation:wrong",
        "account_type": "non_revolving_loan",
        "source_refs": [
            {
                "source": "native_detail_table",
                "table_id": "neighbor-table",
                "logical_page": 1,
                "bbox": [10, 70, 400, 180],
            }
        ],
    }
    assert native_extraction._match_account_table_observations(
        skeletons, [wrong_table]
    ) == {}

    owner_table = {
        **wrong_table,
        "source_refs": [
            {
                "source": "native_detail_table",
                "table_id": "owner-table",
                "logical_page": 1,
                "bbox": [10, 70, 400, 180],
            }
        ],
    }
    assert native_extraction._match_account_table_observations(
        skeletons, [owner_table]
    ) == {0: 0}


def test_exact_cell_owner_guard_uses_each_candidate_not_stale_loop_row() -> None:
    cell_skeleton = {
        "account_id": "credit_account:non_revolving_loan:1",
        "account_type": "non_revolving_loan",
        "account_family_quality": "exact",
        "category_sequence": 1,
        "_printed_ordinal_status": "printed_unique",
        "source_refs": [
            {
                "source": "native_detail_table_cell",
                "table_id": "owner-table",
                "logical_page": 1,
                "geometry_status": "exact",
                "anchor_binding": "printed_account_ordinal",
                "bbox": [10, 40, 80, 60],
            }
        ],
        "page": 1,
        "bbox": [10, 40, 80, 60],
        "_canonical_segment": {
            "pages": [{"logical_page": 1, "min_y": 40, "max_y": 200}]
        },
    }
    ordinary_skeleton = {
        "account_id": "credit_account:non_revolving_loan:2",
        "account_type": "non_revolving_loan",
        "account_family_quality": "exact",
        "category_sequence": 2,
        "_printed_ordinal_status": "printed_unique",
        "page": 1,
        "bbox": [10, 200, 80, 220],
        "_canonical_segment": {
            "pages": [{"logical_page": 1, "min_y": 200, "max_y": None}]
        },
    }
    neighbor_table = {
        "account_id": "table-observation:neighbor",
        "account_type": "non_revolving_loan",
        "source_refs": [
            {
                "source": "native_detail_table",
                "table_id": "neighbor-table",
                "logical_page": 1,
                "bbox": [10, 70, 400, 180],
            }
        ],
    }

    assert native_extraction._match_account_table_observations(
        [cell_skeleton, ordinary_skeleton], [neighbor_table]
    ) == {}


def test_registered_printed_heading_owns_cross_page_family_not_generic_shape() -> None:
    context = _registered_section_context(
        [
            {
                "page": 10,
                "source_page": 5,
                "lines": [
                    _line("(二)循环贷账户一", [10, 5, 200, 15], "r1-heading"),
                    _line("账户1", [10, 40, 80, 60], "r1-account-1"),
                    _line("账户2", [10, 80, 80, 100], "r1-account-2"),
                ],
            },
            {
                "page": 11,
                "source_page": 6,
                "lines": [
                    _line("账户3", [10, 40, 80, 60], "r1-account-3"),
                    _line("账户4", [10, 80, 80, 100], "r1-account-4"),
                ],
            },
        ]
    )

    rows = native_extraction._account_anchor_skeletons(context)

    assert [(row["account_type"], row["category_sequence"]) for row in rows] == [
        ("revolving_loan_subaccount", 1),
        ("revolving_loan_subaccount", 2),
        ("revolving_loan_subaccount", 3),
        ("revolving_loan_subaccount", 4),
    ]


def test_registered_mixed_page_boundary_assigns_prefix_to_prior_heading() -> None:
    context = _registered_section_context(
        [
            {
                "page": 1,
                "source_page": 1,
                "lines": [
                    _line("(二)循环贷账户一", [10, 5, 200, 15], "r1-heading"),
                    _line("账户1", [10, 40, 80, 60], "r1-account-1"),
                ],
            },
            {
                "page": 2,
                "source_page": 2,
                "lines": [
                    _line("账户2", [10, 40, 80, 60], "r1-account-2"),
                    _line("(三)循环贷账户二", [10, 80, 200, 100], "r2-heading"),
                    _line("账户1", [10, 120, 80, 140], "r2-account-1"),
                ],
            },
        ]
    )

    rows = native_extraction._account_anchor_skeletons(context)

    assert [(row["account_type"], row["category_sequence"]) for row in rows] == [
        ("revolving_loan_subaccount", 1),
        ("revolving_loan_subaccount", 2),
        ("revolving_loan_account", 1),
    ]


def test_summary_cannot_establish_family_but_outer_section_number_is_not_authority() -> None:
    summary_context = _registered_section_context(
        [
            {
                "page": 1,
                "source_page": 1,
                "lines": [
                    _line("循环贷账户一信息汇总", [10, 5, 200, 15], "summary"),
                    _line("账户1", [10, 40, 80, 60], "summary-account"),
                ],
            }
        ]
    )
    reordered_number_context = _registered_section_context(
        [
            {
                "page": 1,
                "source_page": 1,
                "lines": [
                    _line("(三)循环贷账户一", [10, 5, 200, 15], "wrong-number"),
                    _line("账户1", [10, 40, 80, 60], "wrong-account"),
                ],
            }
        ]
    )

    assert native_extraction._account_anchor_skeletons(summary_context) == []
    rows = native_extraction._account_anchor_skeletons(reordered_number_context)
    # PBOC revisions can reorder or insert sections. The exact semantic family
    # title owns the role; its outer Chinese numeral is presentation evidence,
    # not a hard-coded family discriminator.
    assert [(row["account_type"], row["category_sequence"]) for row in rows] == [
        ("revolving_loan_subaccount", 1)
    ]


def test_registered_duplicate_exact_fragment_is_suppressed_and_reported() -> None:
    replay = _line("账户1", [10, 40, 80, 60], "same-fragment")
    context = _registered_section_context(
        [
            {
                "page": 1,
                "source_page": 1,
                "lines": [
                    _line("(一)非循环贷账户", [10, 5, 200, 15], "heading"),
                    replay,
                    dict(replay),
                ],
            }
        ]
    )

    rows = native_extraction._account_anchor_skeletons(context)

    assert [row.get("category_sequence") for row in rows] == [1]
    assert _issue_codes(context).count(
        "candidate_b_account_anchor_duplicate_fragment_suppressed"
    ) == 1


def test_registered_heading_owns_exact_dot_suffixed_agreement_anchor() -> None:
    context = _registered_section_context(
        [
            {
                "page": 14,
                "source_page": 7,
                "lines": [
                    _line("(二)循环贷账户一", [10, 5, 200, 15], "r1-heading"),
                    _line("账户11", [10, 40, 80, 60], "r1-account-11"),
                ],
            },
            {
                "page": 15,
                "source_page": 8,
                "lines": [
                    _line(
                        "账户12.(授信协议标识:B1011000H000114090201980008376003010000616100001X)",
                        [52.5, 88.0, 292.5, 99.5],
                        "ocr:sp0008:lp0015:0014",
                    ),
                    _line("账户13", [10, 120, 80, 140], "r1-account-13"),
                ],
            },
        ]
    )

    rows = native_extraction._account_anchor_skeletons(context)
    ledger = native_extraction._source_completeness_ledger(context)

    assert [row.get("category_sequence") for row in rows] == [11, 12, 13]
    account12 = rows[1]
    assert account12["account_type"] == "revolving_loan_subaccount"
    assert account12["credit_agreement_identifier"] == (
        "B1011000H000114090201980008376003010000616100001X"
    )
    assert ledger["account_family_anchor_inventory_sequences"][
        "revolving_loan_subaccount"
    ] == [11, 12, 13]
    assert "12" in ledger["account_family_ordinal_observations"][
        "revolving_loan_subaccount"
    ]


def test_registered_dot_suffix_exception_requires_exact_unique_agreement_source() -> None:
    bad_lines = [
        _line("账户12.任意说明", [10, 40, 180, 60], "wrong-suffix"),
        _line("账户13.(授信协议标识:TOO-SHORT)", [10, 80, 180, 100], "short-id"),
        _line(
            "账户14.(授信协议标识:B1011000H000114090201980008376003010000616100001X)尾注",
            [10, 120, 280, 140],
            "trailing-text",
        ),
    ]
    context = _registered_section_context(
        [
            {
                "page": 1,
                "source_page": 1,
                "lines": [
                    _line("(二)循环贷账户一", [10, 5, 200, 15], "heading"),
                    *bad_lines,
                ],
            }
        ]
    )

    assert native_extraction._account_anchor_skeletons(context) == []


def test_registered_dot_suffix_exception_rejects_replayed_evidence_owner() -> None:
    anchor = _line(
        "账户12.(授信协议标识:B1011000H000114090201980008376003010000616100001X)",
        [10, 40, 280, 60],
        "replayed-owner",
    )
    context = _registered_section_context(
        [
            {
                "page": 1,
                "source_page": 1,
                "lines": [
                    _line("(二)循环贷账户一", [10, 5, 200, 15], "heading"),
                    anchor,
                    {**anchor, "bbox": [10, 80, 280, 100]},
                ],
            }
        ]
    )

    assert native_extraction._account_anchor_skeletons(context) == []


def test_registered_ordinary_d1_anchor_six_remains_owned() -> None:
    context = _registered_section_context(
        [
            {
                "page": 1,
                "source_page": 1,
                "lines": [
                    _line("(一)非循环贷账户", [10, 5, 200, 15], "d1-heading"),
                    _line("账户6", [10, 40, 80, 60], "d1-account-6"),
                ],
            }
        ]
    )

    rows = native_extraction._account_anchor_skeletons(context)
    ledger = native_extraction._source_completeness_ledger(context)

    assert [(row["account_type"], row["category_sequence"]) for row in rows] == [
        ("non_revolving_loan", 6)
    ]
    assert ledger["account_family_anchor_inventory_sequences"] == {
        "non_revolving_loan": [6]
    }
    assert "6" in ledger["account_family_ordinal_observations"][
        "non_revolving_loan"
    ]


def test_registered_distinct_duplicate_ordinal_remains_provisional() -> None:
    context = _registered_section_context(
        [
            {
                "page": 1,
                "source_page": 1,
                "lines": [
                    _line("(一)非循环贷账户", [10, 5, 200, 15], "heading"),
                    _line("账户1", [10, 40, 80, 60], "account-a"),
                    _line("账户1", [10, 80, 80, 100], "account-b"),
                ],
            }
        ]
    )

    rows = native_extraction._account_anchor_skeletons(context)

    assert len(rows) == 2
    assert all("category_sequence" not in row for row in rows)
    assert all(row["_printed_ordinal_status"] == "printed_duplicate" for row in rows)
    assert _issue_codes(context).count(
        "candidate_b_account_printed_ordinal_unresolved"
    ) == 2


def test_registered_heading_and_anchor_require_exact_source_binding() -> None:
    inexact_heading = _line("(一)非循环贷账户", [10, 5, 200, 15], "heading")
    inexact_heading["evidence_ids"] = []
    context = _registered_section_context(
        [
            {
                "page": 1,
                "source_page": 1,
                "lines": [
                    inexact_heading,
                    _line("账户1", [10, 40, 80, 60], "account"),
                ],
            }
        ]
    )

    assert native_extraction._account_anchor_skeletons(context) == []


def test_registered_ledger_uses_heading_owned_family_endpoints_only() -> None:
    context = _registered_section_context(
        [
            {
                "page": 1,
                "source_page": 1,
                "lines": [
                    _line("(一)非循环贷账户", [10, 5, 200, 15], "d1-heading"),
                    _line("账户1", [10, 40, 80, 60], "d1-account-1"),
                    _line("账户2", [10, 80, 80, 100], "d1-account-2"),
                ],
            },
            {
                "page": 2,
                "source_page": 2,
                "lines": [
                    _line("账户3", [10, 40, 80, 60], "d1-account-3"),
                    _line("(二)循环贷账户一", [10, 80, 200, 100], "r1-heading"),
                    _line("账户1", [10, 120, 80, 140], "r1-account-1"),
                ],
            },
            {
                "page": 3,
                "source_page": 3,
                "lines": [
                    _line("账户2", [10, 40, 80, 60], "r1-account-2"),
                    # Broad morphology/text matching sees this phrase, but it is
                    # summary prose and cannot create a quasi-card family.
                    _line(
                        "准贷记卡账户信息汇总",
                        [10, 80, 200, 100],
                        "summary",
                    ),
                    _line("账户9", [10, 120, 80, 140], "summary-account"),
                ],
            },
        ]
    )

    ledger = native_extraction._source_completeness_ledger(context)

    # The trailing account-shaped line is not owned by an exact family
    # heading.  Its gap makes the whole carried page ambiguous, so the strict
    # family population stops at the last sealed page instead of retaining a
    # convenient dense prefix from the same unclosed page.
    assert ledger["account_family_endpoints"] == {
        "non_revolving_loan": 3,
        "revolving_loan_subaccount": 1,
    }
    assert ledger["credit_accounts"] == 4
    assert "quasi_credit_card" not in ledger["account_family_endpoints"]


def test_registered_ledger_counts_one_exact_unnumbered_family_anchor() -> None:
    context = _registered_section_context(
        [
            {
                "page": 1,
                "source_page": 1,
                "lines": [
                    _line("(五)准贷记卡账户", [10, 5, 200, 15], "quasi-heading"),
                    _line("账户", [10, 40, 80, 60], "quasi-account"),
                ],
            }
        ]
    )

    ledger = native_extraction._source_completeness_ledger(context)

    assert ledger["credit_accounts"] == 1
    assert ledger["account_family_unnumbered_anchor_counts"] == {
        "quasi_credit_card": 1
    }
    assert "account_family_endpoints" not in ledger


def test_registered_ledger_does_not_count_ambiguous_unnumbered_owners() -> None:
    context = _registered_section_context(
        [
            {
                "page": 1,
                "source_page": 1,
                "lines": [
                    _line("(五)准贷记卡账户", [10, 5, 200, 15], "quasi-heading"),
                    _line("账户", [10, 40, 80, 60], "quasi-account-a"),
                    _line("账户", [10, 80, 80, 100], "quasi-account-b"),
                ],
            }
        ]
    )

    ledger = native_extraction._source_completeness_ledger(context)

    assert "credit_accounts" not in ledger
    assert "account_family_unnumbered_anchor_counts" not in ledger


def test_legacy_ledger_keeps_table_guarded_family_scan() -> None:
    context = _two_page_context(
        second_lines=[
            _line("账户2", [10, 40, 80, 60], "page-two-anchor")
        ],
        second_tables=[
            _cell_table(
                "page-two-cell",
                text="账户2",
                bbox=[8, 38, 90, 62],
                evidence_id="page-two-anchor",
            )
        ],
    )
    # No production provenance basis: this fixture remains on the older
    # native-table guarded path rather than bypassing its family proofs.
    context.reading_order_resolution = {"resolved": True, "authoritative": True}

    ledger = native_extraction._source_completeness_ledger(context)

    assert ledger["account_family_endpoints"] == {"non_revolving_loan": 2}
    assert ledger["credit_accounts"] == 2
