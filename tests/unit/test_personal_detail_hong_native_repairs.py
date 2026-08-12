from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from docmirror.plugins.credit_report.personal_detail_scanned import native_extraction


def _table(
    table_id: str,
    rows: list[list[str]],
    *,
    top: float,
    geometry: dict | None = None,
) -> SimpleNamespace:
    metadata: dict = {"raw_rows": rows}
    if geometry is not None:
        metadata["geometry"] = geometry
    return SimpleNamespace(
        table_id=table_id,
        metadata=metadata,
        headers=[],
        rows=[],
        bbox=[40.0, top, 400.0, top + max(20.0, len(rows) * 10.0)],
        confidence=0.99,
    )


def _page(
    logical_page: int,
    tables: list[SimpleNamespace],
    *,
    template: str,
    texts: list[SimpleNamespace] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        page_number=logical_page,
        source_page_number=(logical_page + 1) // 2,
        canonical_template_id=template,
        tables=tables,
        texts=texts or [],
        height=595.0,
    )


def _line(text: str, top: float, evidence_id: str) -> dict:
    return {
        "text": text,
        "bbox": [50.0, top, 360.0, top + 10.0],
        "evidence_ids": [evidence_id],
    }


def _text(text: str, top: float) -> SimpleNamespace:
    return SimpleNamespace(content=text, text=text, bbox=[50.0, top, 360.0, top + 10.0])


def _geometry(
    rows: list[list[str]],
    *,
    top: float,
    spanning_column: int | None = None,
) -> dict:
    width = max((len(row) for row in rows), default=0)
    row_bands = [
        {"index": row, "y0": top + row * 10.0, "y1": top + (row + 1) * 10.0}
        for row in range(len(rows))
    ]
    col_bands = [
        {"index": column, "x0": 40.0 + column * 90.0, "x1": 130.0 + column * 90.0}
        for column in range(width)
    ]
    cell_bboxes: list[list[list[float] | None]] = []
    cell_status: list[list[str]] = []
    cell_evidence: list[list[list[str]]] = []
    for row_index, row in enumerate(rows):
        boxes: list[list[float] | None] = []
        statuses: list[str] = []
        evidence: list[list[str]] = []
        for column in range(width):
            if row_index == 1 and spanning_column == column:
                boxes.append(None)
                statuses.append("derived")
                evidence.append([])
                continue
            bottom = (
                row_bands[1]["y1"]
                if row_index == 0 and spanning_column == column
                else row_bands[row_index]["y1"]
            )
            boxes.append(
                [
                    col_bands[column]["x0"],
                    row_bands[row_index]["y0"],
                    col_bands[column]["x1"],
                    bottom,
                ]
            )
            statuses.append("exact")
            evidence.append(
                [f"cell:{row_index}:{column}"]
                if column < len(row) and str(row[column] or "")
                else []
            )
        cell_bboxes.append(boxes)
        cell_status.append(statuses)
        cell_evidence.append(evidence)
    spans = []
    if spanning_column is not None:
        spans.append(
            {
                "row": 0,
                "col": spanning_column,
                "row_span": 2,
                "col_span": 1,
                "bbox": cell_bboxes[0][spanning_column],
            }
        )
    return {
        "row_bands": row_bands,
        "col_bands": col_bands,
        "cell_bboxes": cell_bboxes,
        "cell_geometry_status": cell_status,
        "cell_evidence_ids": cell_evidence,
        "cell_spans": spans,
    }


@pytest.mark.parametrize(
    ("logical_page", "current_top", "candidate_top", "anchor_top"),
    [(4, 36.0, 190.5, 128.0), (5, 193.5, 431.5, 421.0)],
)
def test_hong_exact_anchor_vetoes_damaged_new_base_as_prior_continuation(
    logical_page: int,
    current_top: float,
    candidate_top: float,
    anchor_top: float,
) -> None:
    rows = [
        ["营理机构", "账户标识", "开立日期", "账户授信额度"],
        ["兰州银行股份有限公司", "D10128210H0001", "2022.11.14", "100"],
        ["账户状态", "账户关闭日期", "结清", "2023.08.11"],
    ]
    candidate = _table("candidate", rows, top=candidate_top)
    context = SimpleNamespace(
        corrected_evidence_pages=lambda: [
            {
                "page": logical_page,
                "lines": [_line("账户4", anchor_top, "next-account")],
            }
        ]
    )

    assert not native_extraction._geometric_prior_account_continuation(
        parse_result=context,
        page=SimpleNamespace(page_number=logical_page),
        table=candidate,
        page_tables=[candidate],
        current_logical_page=logical_page,
        current_table_top=current_top,
        rows=rows,
        pending_labels=None,
        cross_page_order_resolved=True,
    )


def test_hong_geometric_continuation_still_accepts_fragment_without_new_anchor() -> None:
    rows = [["账户状态", "余额"], ["正常", "0"]]
    candidate = _table("continuation", rows, top=190.5)
    context = SimpleNamespace(corrected_evidence_pages=lambda: [{"page": 4, "lines": []}])

    assert native_extraction._geometric_prior_account_continuation(
        parse_result=context,
        page=SimpleNamespace(page_number=4),
        table=candidate,
        page_tables=[candidate],
        current_logical_page=4,
        current_table_top=36.0,
        rows=rows,
        pending_labels=None,
        cross_page_order_resolved=True,
    )


def _hong_card_prefix_context(*, defect: str | None = None) -> tuple[SimpleNamespace, dict]:
    leading_rows = [
        ["", "★", "*", "*", "★", "*", "N", "N", "N", "*", "★", ""],
        ["2025", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "", ""],
        ["", "", "", "", "", "#", "", "N", "N", "N", "N", "★", ""],
        ["2024", "o", "", "", "", "真", "", "0", "0", "0", "0", "0", "0"],
        ["", "", "", "", "", "", "", "", "", "", "", "", ""],
    ]
    if defect == "leading_grid_prose":
        leading_rows[0][1] = "查询机构"
    leading_grid = _table(
        "pt_7_0",
        leading_rows,
        top=39.5,
    )
    damaged_header = "共享授信额度 账户授信额度 开立日期 账户标识 发卡机树"
    if defect == "second_damaged_role":
        damaged_header = damaged_header.replace("共享授信额度", "")
    card_3 = _table(
        "pt_7_1",
        [
            [damaged_header, "担保方式 业务种类 币种", ""],
            [
                "B10111000H 中国工商银行 00014100000 2025.05.15 0 股份有限公司 "
                "02101441840 厦门市分行 01",
                "大额专项分期 抵押 人民币元 卡",
                "",
            ],
            ["截至2025年10月18日", "", ""],
        ],
        top=116.0,
    )
    agreement = _table(
        "pt_7_2",
        [["管理机构", "授信协议标识", "生效日期", "授信额度用途"]],
        top=370.5,
    )
    evidence_page = {
        "page": 7,
        "source_page": 4,
        "lines": [
            _line("账户3(卡片那号:5140)", 108.5, "card-3"),
            _line("(四)授信协议信息", 338.0, "agreement-boundary"),
        ],
    }
    if defect == "missing_partition":
        evidence_page["lines"] = evidence_page["lines"][:1]
    context = SimpleNamespace(
        pages=[_page(7, [leading_grid, card_3, agreement], template="credit_agreement")],
        reading_order_by_logical={6: 1, 7: 2},
        reading_order_resolution={"resolved": True, "authoritative": True},
    )
    return context, evidence_page


def test_hong_card3_prefix_survives_one_damaged_issuer_header() -> None:
    context, evidence_page = _hong_card_prefix_context()

    assert native_extraction._bounded_credit_card_prefix_carry_on_mixed_page(
        context,
        page=evidence_page,
        active_family="credit_card",
        active_family_quality="exact",
        active_family_logical_page=6,
        active_family_last_ordinal=2,
        local_table_family=None,
        cross_page_order_resolved=True,
    )


@pytest.mark.parametrize(
    "defect",
    ["missing_partition", "second_damaged_role", "leading_grid_prose"],
)
def test_hong_card3_prefix_fails_closed_outside_exact_shape(defect: str) -> None:
    context, evidence_page = _hong_card_prefix_context(defect=defect)

    assert not native_extraction._bounded_credit_card_prefix_carry_on_mixed_page(
        context,
        page=evidence_page,
        active_family="credit_card",
        active_family_quality="exact",
        active_family_logical_page=6,
        active_family_last_ordinal=2,
        local_table_family=None,
        cross_page_order_resolved=True,
    )


def _hong_inquiry_context(*, defect: str | None = None) -> SimpleNamespace:
    institutional_header = ["编号", "查询日期", "查询机构", "查均原因"]
    page_8_rows = [
        ["", "", "", ""],
        institutional_header,
        ["1", "2025.11.02", "招商银行股份有限公司信用卡中心", "贷后管理"],
        ["2", "2025.11.01", "华夏银行股份有限公司信用卡中心", "贷后管理"],
        ["3", "2025.08.06", "华夏银行股份有限公司信用卡中心", "贷后管理"],
        [
            "4",
            "2025.06.30",
            "中国工商银行股份有限公司银行卡业务部(牡丹卡中心)",
            "信用卡审批",
        ],
    ]
    page_8_geometry = _geometry(page_8_rows, top=489.0)
    if defect == "damaged_mixed_header_geometry":
        page_8_geometry["cell_geometry_status"][1][2] = "derived"
    page_8_table = _table(
        "pt_8_5",
        page_8_rows,
        top=489.0,
        geometry=page_8_geometry,
    )
    page_8_texts = [
        _text("四查询记录", 457.0),
        _text("机构查均记录明细", 480.5),
    ]
    if defect == "missing_mixed_boundary":
        page_8_texts = page_8_texts[1:]

    page_9_rows = [
        ["", "", "", "", ""],
        [
            "5",
            "2025.05.07",
            "中国工商银行股份有限公司银行卡业务部(牡丹卡中心)",
            "信用卡审批",
            "",
        ],
        ["6", "2025.05.06", "招商银行股份有限公司", "贷款审批", ""],
        [
            "7",
            "2025.05.06",
            "中国农业银行股份有限公司厦门市分行",
            "信用卡审批",
            "",
        ],
        ["8", "2025.04.29", "华夏银行股份有限公司信用卡中心", "贷后管理", ""],
        ["9", "2025.04.02", "中 招商银行股份有限公司信用卡中心", "贷后管理", ""],
        ["10%", "2025.01.25", "华夏银行股份有限公司信用卡中心", "贷后管理", ""],
        ["", "8", "", "贷后管理", ""],
        ["Te 11", "2024.10.29", "华夏银行股份有限公司信用卡中心", "", ""],
        [
            "13",
            "2024.10.28 2024.07.17",
            "招商银行股份有限公司信用卡中心 华夏银行股份有限公司信用卡中心",
            "贷后管理 信用卡审批",
            "",
        ],
        ["14", "2024.04.30", "招商银行股份有限公司信用卡中心", "贷后管理", ""],
        ["15", "2024.03.30", "招商银行股份有限公司信用卡中心", "贷后管理", ""],
        ["16", "2023.12.29", "重庆市与众融资担保有限公司", "担保资格审查", ""],
        ["A", "2023.12.20", "招商银行股份有限公司信用卡中心", "贷后管理", ""],
        ["", "", "", "", ""],
    ]
    if defect == "populated_trailing_column":
        page_9_rows[1][4] = "competing"
    page_9_table = _table(
        "pt_9_0",
        page_9_rows,
        top=39.0,
        geometry=_geometry(page_9_rows, top=39.0),
    )

    personal_rows = [
        ["", "", "", "查询原因"],
        ["编号", "查询日期", "查询机构", ""],
        ["→", "2025.07.18", "本人", "本人查询(自助查询机)"],
        ["2", "2025.07.15", "本人", "银行) 本人查询(商业银行网上"],
        ["3", "2025.07.15", "本人", "用信息服务平台) 本人查询(互联网个人信"],
        ["", "", "", ""],
    ]
    if defect == "transposed_personal_header":
        personal_rows[1][1], personal_rows[1][2] = (
            personal_rows[1][2],
            personal_rows[1][1],
        )
    elif defect == "unknown_wrapped_personal_channel":
        personal_rows[3][3] = "渠道) 本人查询(未知"
    elif defect == "competing_personal_reason":
        personal_rows[2][3] = "本人查询(自助查询机) 贷款审批"
    personal_table = _table(
        "pt_9_1",
        personal_rows,
        top=233.0,
        geometry=_geometry(personal_rows, top=233.0, spanning_column=3),
    )
    context = SimpleNamespace(
        pages=[
            _page(
                8,
                [page_8_table],
                template="credit_agreement",
                texts=page_8_texts,
            ),
            _page(
                9,
                [page_9_table, personal_table],
                template="annotations_and_inquiries",
            ),
        ],
        reading_order_by_logical={8: 1, 9: 2},
        reading_order_resolution={"resolved": True, "authoritative": True},
        corrected_evidence_pages=lambda: [],
        _personal_detail_extraction_issues=[],
    )
    if defect == "non_authoritative_order":
        context.reading_order_resolution = {"resolved": True, "authoritative": False}
    return context


def _hong_canonical_line_rows(*, omit_sequence: int | None = None) -> list[dict]:
    dates = {
        5: "2025-05-07",
        6: "2025-05-06",
        7: "2025-05-06",
        8: "2025-04-29",
        9: "2025-04-02",
        10: "2025-01-25",
        11: "2024-10-29",
        12: "2024-10-28",
        13: "2024-07-17",
        14: "2024-04-30",
        15: "2024-03-30",
    }
    return [
        {
            "inquiry_id": f"line:{sequence}",
            "sequence": sequence,
            "inquiry_date": dates[sequence],
            "institution": f"机构{sequence}",
            "reason": "贷后管理",
            "source_reason": "贷后管理",
            "query_channel": "institution",
            "inquiry_type": "institution",
            "source": "candidate_b_canonical_inquiry_line",
            "source_refs": [],
            "confidence": 0.8,
        }
        for sequence in range(5, 16)
        if sequence != omit_sequence
    ]


def _install_hong_canonical_lines(
    monkeypatch: pytest.MonkeyPatch,
    *,
    omit_sequence: int | None = None,
) -> None:
    monkeypatch.setattr(
        native_extraction,
        "_canonical_inquiry_line_rows",
        lambda _context: _hong_canonical_line_rows(omit_sequence=omit_sequence),
    )


def test_hong_mixed_inquiry_regions_emit_17_institution_and_3_personal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_hong_canonical_lines(monkeypatch)
    context = _hong_inquiry_context()

    rows = native_extraction._extract_inquiries(context)
    coverage = native_extraction._inquiry_source_coverage(context)

    institutional = [row for row in rows if row["inquiry_type"] == "institution"]
    personal = [row for row in rows if row["inquiry_type"] == "personal"]
    assert [row["sequence"] for row in institutional] == list(range(1, 18))
    assert [row["sequence"] for row in personal] == [1, 2, 3]
    expected_raw_reasons = [
        "本人查询(自助查询机)",
        "银行) 本人查询(商业银行网上",
        "用信息服务平台) 本人查询(互联网个人信",
    ]
    assert [row["reason"] for row in personal] == expected_raw_reasons
    assert [row["source_reason"] for row in personal] == expected_raw_reasons
    assert coverage["sequence_endpoints"] == {"institution": 17, "personal": 3}
    assert coverage["expected_row_count"] == 20


@pytest.mark.parametrize(
    "defect",
    [
        "missing_mixed_boundary",
        "damaged_mixed_header_geometry",
        "non_authoritative_order",
        "populated_trailing_column",
        "transposed_personal_header",
        "unknown_wrapped_personal_channel",
        "competing_personal_reason",
    ],
)
def test_hong_mixed_inquiry_repairs_fail_closed(
    defect: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_hong_canonical_lines(monkeypatch)
    context = _hong_inquiry_context(defect=defect)

    rows = native_extraction._extract_inquiries(context)
    institutional = [row for row in rows if row["inquiry_type"] == "institution"]
    personal = [row for row in rows if row["inquiry_type"] == "personal"]

    if defect in {
        "transposed_personal_header",
        "unknown_wrapped_personal_channel",
        "competing_personal_reason",
    }:
        assert [row["sequence"] for row in institutional] == list(range(1, 18))
        assert personal == []
    elif defect == "non_authoritative_order":
        assert [row["sequence"] for row in institutional] == list(range(1, 16))
        assert [row["sequence"] for row in personal] == [1, 2, 3]
    elif defect == "populated_trailing_column":
        assert [row["sequence"] for row in institutional] == list(range(1, 16))
        assert [row["sequence"] for row in personal] == [1, 2, 3]
    else:
        assert [row["sequence"] for row in institutional] == list(range(5, 16))
        assert [row["sequence"] for row in personal] == [1, 2, 3]


def test_hong_terminal_inquiry_ordinal_requires_complete_line_population(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_hong_canonical_lines(monkeypatch, omit_sequence=12)
    context = _hong_inquiry_context()

    rows = native_extraction._extract_inquiries(context)

    institutional_sequences = {
        row["sequence"] for row in rows if row["inquiry_type"] == "institution"
    }
    assert 16 in institutional_sequences
    assert 17 not in institutional_sequences


def _account_anchor_ref() -> dict:
    return {
        "source": "candidate_b_account_anchor",
        "logical_page": 4,
        "source_page": 2,
        "bbox": [48.5, 477.0, 264.5, 491.5],
        "evidence_ids": ["account-2"],
    }


def test_hong_partial_account_identifier_is_withheld_on_invalid_full_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = {
        "account_id": "table:r2:2",
        "_table_observation_id": "table:r2:2",
        "sequence": 1,
        "account_type": "revolving_loan_account",
        "account_identifier": "D0206000CA202506XZ20011136047",
        "source": "native_detail_account_table",
        "source_refs": [
            {
                "source": "native_detail_table",
                "logical_page": 4,
                "source_page": 2,
                "table_id": "pt_4_2",
                "bbox": [47.5, 486.0, 395.0, 556.0],
            }
        ],
        "canonical_raw": {
            "account_identifier": "D0206000CA2025 06XZ20011136047"
        },
    }
    skeleton = {
        "account_id": "credit_account:revolving_loan_account:2",
        "sequence": 2,
        "category_sequence": 2,
        "account_type": "revolving_loan_account",
        "account_family_quality": "exact",
        "_printed_ordinal_status": "printed_unique",
        "source": "candidate_b_account_anchor",
        "source_refs": [_account_anchor_ref()],
        "_invalid_observation_fields": ["account_identifier"],
        "canonical_raw": {"account_identifier": "X320501...D0206"},
        "source_refs_by_field": {
            "account_identifier": [
                {
                    **_account_anchor_ref(),
                    "field_name": "account_identifier",
                    "binding": "canonical_account_header_geometry",
                }
            ]
        },
    }
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    monkeypatch.setattr(
        native_extraction,
        "_extract_table_accounts",
        lambda _context: ([deepcopy(table)], [], []),
    )
    monkeypatch.setattr(
        native_extraction,
        "_account_anchor_skeletons",
        lambda _context: [deepcopy(skeleton)],
    )
    monkeypatch.setattr(
        native_extraction,
        "_match_account_table_observations",
        lambda _skeletons, _tables, *, parse_result=None: {0: 0},
    )

    accounts, _repayments, _events = native_extraction._extract_accounts(context)

    assert len(accounts) == 1
    assert accounts[0].get("account_identifier") is None
    assert any(
        issue.get("issue_code") == "candidate_b_exact_slot_value_invalid"
        and issue.get("field_name") == "account_identifier"
        for issue in context._personal_detail_extraction_issues
    )


def test_missing_account_table_reports_each_unrecovered_basic_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skeleton = {
        "account_id": "credit_account:credit_card:3",
        "sequence": 8,
        "category_sequence": 3,
        "account_type": "credit_card",
        "management_institution": "中国工商银行股份有限公司厦门市分行",
        "source": "candidate_b_account_anchor",
        "source_refs": [_account_anchor_ref()],
        "_source_absent_fields": ["account_currency", "currency"],
    }
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    monkeypatch.setattr(
        native_extraction,
        "_extract_table_accounts",
        lambda _context: ([], [], []),
    )
    monkeypatch.setattr(
        native_extraction,
        "_account_anchor_skeletons",
        lambda _context: [deepcopy(skeleton)],
    )

    accounts, _repayments, _events = native_extraction._extract_accounts(context)

    assert len(accounts) == 1
    field_issues = {
        issue.get("field_name")
        for issue in context._personal_detail_extraction_issues
        if issue.get("issue_code") == "candidate_b_account_basic_slot_unresolved"
    }
    assert field_issues == {"account_identifier", "open_date"}
    assert any(
        issue.get("issue_code") == "candidate_b_account_table_missing"
        and issue.get("field_name") is None
        for issue in context._personal_detail_extraction_issues
    )
