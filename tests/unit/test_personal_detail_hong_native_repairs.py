from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from docmirror.models.mirror.vnext import EvidenceAtom, EvidenceStore
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
    for table in tables:
        metadata = getattr(table, "metadata", None)
        if isinstance(metadata, dict):
            metadata.setdefault("canonical_template_id", template)
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
    evidence_prefix: str = "cell",
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
                [f"{evidence_prefix}:{row_index}:{column}"]
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
        "cell_token_ids": deepcopy(cell_evidence),
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
                template="annotations_and_inquiries",
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
    assert coverage["numbering_model"] == "unknown"
    assert "expected_row_count" not in coverage


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

    if defect in {"unknown_wrapped_personal_channel", "competing_personal_reason"}:
        assert [row["sequence"] for row in institutional] == list(range(1, 18))
        assert personal == []
    elif defect == "transposed_personal_header":
        # The malformed local personal topology stays withheld, while the
        # independently complete mixed-page institutional topology can prove
        # its own ordinary PBOC role map without global-profile authority.
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


def _collapsed_token_inquiry_context(*, defect: str | None = None) -> SimpleNamespace:
    rows = [
        ["编号", "查询日期", "查询机构", "查询原因"],
        ["143", "2023.08.11", "招商银行股份有限公司", "贷后管理"],
        [
            "144 145",
            "2023.08.11 2023.08.05",
            "上海浦东发展银行股份有限公司 中国工商银行股份有限公司",
            "贷款审批 贷后管理",
        ],
    ]
    geometry = _geometry(rows, top=40.0)
    token_specs = {
        (2, 0): [("144", [50.0, 61.0, 70.0, 65.0]), ("145", [50.0, 66.0, 70.0, 70.0])],
        (2, 1): [("2023.08.11", [140.0, 61.0, 205.0, 65.0]), ("2023.08.05", [140.0, 66.0, 205.0, 70.0])],
        (2, 2): [
            ("上海浦东发展银行股份有限公司", [230.0, 61.0, 300.0, 65.0]),
            ("中国工商银行股份有限公司", [230.0, 66.0, 300.0, 70.0]),
        ],
        (2, 3): [("贷款审批", [320.0, 61.0, 360.0, 65.0]), ("贷后管理", [320.0, 66.0, 360.0, 70.0])],
    }
    atoms: list[EvidenceAtom] = []
    geometry["cell_token_ids"] = deepcopy(geometry["cell_evidence_ids"])
    for (row, column), specs in token_specs.items():
        ids = [f"merged:{row}:{column}:{index}" for index in range(len(specs))]
        geometry["cell_evidence_ids"][row][column] = ids
        geometry["cell_token_ids"][row][column] = list(ids)
        for token_id, (text, bbox) in zip(ids, specs, strict=True):
            atoms.append(EvidenceAtom(id=token_id, text=text, bbox=bbox))
    if defect == "foreign_evidence":
        geometry["cell_token_ids"][2][3].append("foreign")
    elif defect == "overlapping_token_rows":
        atoms[-2].bbox = [320.0, 61.0, 360.0, 67.0]
        atoms[-1].bbox = [320.0, 64.0, 360.0, 70.0]
    elif defect == "nonconsecutive_ordinals":
        atoms[1].text = "146"
    elif defect == "ambiguous_reason":
        atoms[-1].text = "未知用途"
    elif defect == "duplicate_evidence_owner":
        duplicate_id = geometry["cell_token_ids"][2][0][0]
        replaced_id = geometry["cell_token_ids"][2][1][0]
        geometry["cell_token_ids"][2][1][0] = duplicate_id
        geometry["cell_evidence_ids"][2][1][0] = duplicate_id
        atoms = [atom for atom in atoms if atom.id != replaced_id]
    elif defect == "personal_reason_with_bank":
        atoms[-2].text = "本人查询(自助查询机)"
    elif defect == "mixed_inquiry_types":
        atoms[-3].text = "本人"
        atoms[-1].text = "本人查询(自助查询机)"

    table = _table("pt_54_0", rows, top=40.0, geometry=geometry)
    context = SimpleNamespace(
        pages=[
            _page(
                54,
                [table],
                template="annotations_and_inquiries",
            )
        ],
        evidence_plane=SimpleNamespace(evidence=EvidenceStore(text_atoms=atoms)),
        reading_order_by_logical={54: 1},
        reading_order_resolution={"resolved": True, "authoritative": True},
        corrected_evidence_pages=lambda: [],
        _personal_detail_extraction_issues=[],
    )
    return context


def test_exact_token_rows_split_one_collapsed_inquiry_band() -> None:
    rows = native_extraction._extract_inquiries(_collapsed_token_inquiry_context())

    assert [(row["sequence"], row["institution"], row["reason"]) for row in rows] == [
        (143, "招商银行股份有限公司", "贷后管理"),
        (144, "上海浦东发展银行股份有限公司", "贷款审批"),
        (145, "中国工商银行股份有限公司", "贷后管理"),
    ]
    for row in rows[1:]:
        assert row["source"] == "native_detail_inquiry_token_rows"
        assert row["source_refs"][0]["geometry_scope"] == "token_row"
        assert len(row["source_refs"][0]["evidence_ids"]) == 4


def test_inquiry_tables_are_consumed_top_to_bottom_before_schema_carry() -> None:
    upper = _table("pt_54_0", [["144"]], top=40.0)
    lower = _table("pt_54_1", [["缂栧彿"]], top=380.0)
    unboxed_first = _table("unboxed-first", [["noise"]], top=0.0)
    unboxed_first.bbox = None
    unboxed_second = _table("unboxed-second", [["noise"]], top=0.0)
    unboxed_second.bbox = None
    page = _page(
        54,
        [lower, unboxed_first, upper, unboxed_second],
        template="annotations_and_inquiries",
    )

    assert [
        table.table_id
        for table in native_extraction._inquiry_page_tables_in_physical_order(page)
    ] == ["pt_54_0", "pt_54_1", "unboxed-first", "unboxed-second"]


def _headerless_institution_tail_context(
    monkeypatch: pytest.MonkeyPatch,
    *,
    include_later_header: bool = True,
    defect: str | None = None,
    tail_count: int = 8,
) -> SimpleNamespace:
    tail_rows = [
        [
            str(sequence),
            f"2024.01.{sequence - 121:02d}",
            f"样例银行股份有限公司{sequence}",
            "贷后管理",
        ]
        for sequence in range(122, 122 + tail_count)
    ]
    upper = _table(
        "pt_54_0",
        tail_rows,
        top=40.0,
        geometry=_geometry(tail_rows, top=40.0),
    )
    lower_rows = [
        ["编号", "查询日期", "查询机构", "查询原因"],
        ["1", "2025.05.07", "本人", "本人查询(自助查询机)"],
    ]
    lower = _table(
        "pt_54_1",
        lower_rows if include_later_header else [["noise"]],
        top=380.0,
        geometry=_geometry(
            lower_rows if include_later_header else [["noise"]],
            top=380.0,
            evidence_prefix="lower",
        ),
    )
    if defect == "nonconsecutive_ordinals":
        upper.metadata["raw_rows"][4][0] = "127"
    elif defect == "personal_marker":
        upper.metadata["raw_rows"][4][2] = "本人"
        upper.metadata["raw_rows"][4][3] = "本人查询(自助查询机)"
    elif defect == "duplicate_evidence_owner":
        geometry = upper.metadata["geometry"]
        geometry["cell_evidence_ids"][4][0] = list(
            geometry["cell_evidence_ids"][3][0]
        )
    elif defect == "lower_header_not_exact":
        lower.metadata["geometry"]["cell_geometry_status"][0][2] = "derived"
    elif defect == "lower_first_row_not_personal":
        lower.metadata["raw_rows"][1][2] = "样例银行"
        lower.metadata["raw_rows"][1][3] = "贷后管理"
    elif defect == "overlapping_table_boxes":
        upper.bbox[3] = lower.bbox[1] + 1.0
    later_tables = [lower]
    if defect in {"competing_lower_seal", "duplicate_lower_seal"}:
        competing = deepcopy(lower)
        competing.table_id = "pt_54_2" if defect == "competing_lower_seal" else lower.table_id
        competing.bbox = [40.0, 470.0, 400.0, 490.0]
        if defect == "competing_lower_seal":
            competing.metadata["geometry"] = _geometry(
                deepcopy(lower_rows),
                top=470.0,
                evidence_prefix="competing-lower",
            )
        later_tables.append(competing)
    monkeypatch.setattr(native_extraction, "_canonical_inquiry_line_rows", lambda _context: [])
    return SimpleNamespace(
        pages=[
            _page(
                54,
                # Deliberately reverse the reconstruction order.  Physical
                # bbox order must put the headerless institutional tail first.
                [*later_tables, upper],
                template="annotations_and_inquiries",
            )
        ],
        evidence_plane=SimpleNamespace(evidence=EvidenceStore(text_atoms=[])),
        reading_order_by_logical={54: 1},
        reading_order_resolution={"status": "ambiguous"},
        corrected_evidence_pages=lambda: [],
        _personal_detail_extraction_issues=[],
    )


def test_headerless_upper_inquiry_tail_bootstrap_consumes_first_data_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = native_extraction._extract_inquiries(
        _headerless_institution_tail_context(monkeypatch)
    )

    institutional = sorted(
        row["sequence"] for row in rows if row["inquiry_type"] == "institution"
    )
    assert institutional == list(range(122, 130))
    assert any(
        row["inquiry_type"] == "personal" and row["sequence"] == 1 for row in rows
    )
    coverage = native_extraction._inquiry_source_coverage(
        _headerless_institution_tail_context(monkeypatch)
    )
    assert coverage["sequence_endpoints"] == {"institution": 129, "personal": 1}
    assert coverage["numbering_model"] == "unknown"
    assert "expected_row_count" not in coverage
    assert sorted(
        int(sequence)
        for sequence in coverage["ordinal_observations"]["institution"]
    ) == list(range(122, 130))


@pytest.mark.parametrize("tail_count", [2, 4, 8])
def test_headerless_inquiry_bootstrap_uses_smallest_structurally_sealed_tail(
    monkeypatch: pytest.MonkeyPatch,
    tail_count: int,
) -> None:
    context = _headerless_institution_tail_context(monkeypatch, tail_count=tail_count)
    page = context.pages[0]
    upper = min(page.tables, key=lambda table: table.bbox[1])

    descriptor = native_extraction._bounded_headerless_inquiry_table_bootstrap(
        context,
        page,
        upper,
        native_extraction._table_rows(upper),
        slots={"sequence": 0, "inquiry_date": 1, "institution": 2, "reason": 3},
        later_tables=[table for table in page.tables if table is not upper],
    )

    assert descriptor is not None
    assert descriptor["first_sequence"] == 122
    assert descriptor["last_sequence"] == 121 + tail_count


@pytest.mark.parametrize(
    ("tail_count", "defect"),
    [(1, None), (2, "competing_lower_seal"), (2, "duplicate_lower_seal")],
)
def test_headerless_inquiry_bootstrap_rejects_ambiguous_or_singleton_tail(
    monkeypatch: pytest.MonkeyPatch,
    tail_count: int,
    defect: str | None,
) -> None:
    context = _headerless_institution_tail_context(
        monkeypatch,
        tail_count=tail_count,
        defect=defect,
    )
    page = context.pages[0]
    upper = min(page.tables, key=lambda table: table.bbox[1])

    assert (
        native_extraction._bounded_headerless_inquiry_table_bootstrap(
            context,
            page,
            upper,
            native_extraction._table_rows(upper),
            slots={"sequence": 0, "inquiry_date": 1, "institution": 2, "reason": 3},
            later_tables=[table for table in page.tables if table is not upper],
        )
        is None
    )


@pytest.mark.parametrize(
    "defect",
    [
        "no_later_header",
        "nonconsecutive_ordinals",
        "personal_marker",
        "duplicate_evidence_owner",
        "lower_header_not_exact",
        "lower_first_row_not_personal",
        "overlapping_table_boxes",
    ],
)
def test_headerless_upper_inquiry_tail_bootstrap_requires_independent_seals(
    monkeypatch: pytest.MonkeyPatch,
    defect: str,
) -> None:
    context = _headerless_institution_tail_context(
        monkeypatch,
        include_later_header=defect != "no_later_header",
        defect=defect,
    )

    rows = native_extraction._extract_inquiries(context)

    institutional = {
        row["sequence"] for row in rows if row["inquiry_type"] == "institution"
    }
    assert institutional == ({1} if defect == "lower_first_row_not_personal" else set())
    coverage = native_extraction._inquiry_source_coverage(context)
    assert coverage.get("sequence_endpoints", {}).get("institution") == (
        1 if defect == "lower_first_row_not_personal" else None
    )


def _production_shaped_headerless_tail_context(
    monkeypatch: pytest.MonkeyPatch,
    *,
    expose_merged_business_tokens: bool = True,
    expose_collapsed_business_tokens: bool = True,
    include_prior_institution_group: bool = False,
) -> SimpleNamespace:
    tail_rows = [
        [
            str(sequence),
            f"2024.01.{(sequence - 121):02d}",
            f"样例银行股份有限公司{sequence}",
            "贷后管理",
        ]
        for sequence in range(122, 144)
    ]
    # Population ownership must survive business-field OCR failures.  These
    # fields remain subject to normal row validation and are not silently fixed.
    tail_rows[3][3] = "信用审批"
    tail_rows[9][1] = "2023.11.23 招商银行股份有限公司"
    tail_rows[9][2] = ""
    tail_rows[19][1] = "2023.09.05 中国建设银行股份有限公司"
    tail_rows[19][2] = ""
    tail_rows.append(
        [
            "144 145",
            "2023.08.11 2023.08.05",
            "上海浦东发展银行股份有限公司 中国工商银行股份有限公司",
            "贷款审批 贷后管理",
        ]
    )
    upper_geometry = _geometry(tail_rows, top=40.0)
    token_specs = {
        (22, 0): [("144", [50.0, 262.0, 70.0, 266.0]), ("145", [50.0, 267.0, 70.0, 271.0])],
        (22, 1): [("2023.08.11", [140.0, 262.0, 205.0, 266.0]), ("2023.08.05", [140.0, 267.0, 205.0, 271.0])],
        (22, 2): [("上海浦东发展银行股份有限公司", [230.0, 262.0, 300.0, 266.0]), ("中国工商银行股份有限公司", [230.0, 267.0, 300.0, 271.0])],
        (22, 3): [("贷款审批", [320.0, 262.0, 390.0, 266.0]), ("贷后管理", [320.0, 267.0, 390.0, 271.0])],
    }
    atoms: list[EvidenceAtom] = []
    upper_geometry["cell_token_ids"] = deepcopy(
        upper_geometry["cell_evidence_ids"]
    )
    for row, date_text, institution_text in (
        (9, "2023.11.23", "招商银行股份有限公司"),
        (19, "2023.09.05", "中国建设银行股份有限公司"),
    ):
        y0 = 40.0 + row * 10.0
        ids = [f"tail:{row}:span:date", f"tail:{row}:span:institution"]
        upper_geometry["cell_bboxes"][row][1] = [130.0, y0, 310.0, y0 + 10.0]
        upper_geometry["cell_evidence_ids"][row][1] = ids
        upper_geometry["cell_token_ids"][row][1] = list(ids)
        upper_geometry["cell_bboxes"][row][2] = None
        upper_geometry["cell_geometry_status"][row][2] = "derived"
        upper_geometry["cell_evidence_ids"][row][2] = []
        upper_geometry["cell_token_ids"][row][2] = []
        upper_geometry["cell_spans"].append(
            {
                "row": row,
                "col": 1,
                "row_span": 1,
                "col_span": 2,
                "bbox": upper_geometry["cell_bboxes"][row][1],
            }
        )
        date_bbox = [140.0, y0 + 2.0, 205.0, y0 + 8.0]
        institution_bbox = [230.0, y0 + 2.0, 300.0, y0 + 8.0]
        if expose_merged_business_tokens:
            atoms.extend(
                [
                    EvidenceAtom(id=ids[0], text=date_text, bbox=date_bbox),
                    EvidenceAtom(id=ids[1], text=institution_text, bbox=institution_bbox),
                ]
            )
    for (row, column), tokens in token_specs.items():
        ids = [f"tail:{row}:{column}:{index}" for index in range(len(tokens))]
        upper_geometry["cell_evidence_ids"][row][column] = ids
        upper_geometry["cell_token_ids"][row][column] = list(ids)
        if column != 0 and not expose_collapsed_business_tokens:
            continue
        for token_id, (text, bbox) in zip(ids, tokens, strict=True):
            atoms.append(EvidenceAtom(id=token_id, text=text, bbox=bbox))
    upper = _table(
        "pt_54_0",
        tail_rows,
        top=40.0,
        geometry=upper_geometry,
    )
    lower_rows = [
        ["编号", "查询日期", "查询机构", "查询原因"],
        ["1", "2025.05.07", "本人", "本人查询(自助查询机)"],
    ]
    lower = _table(
        "pt_54_1",
        lower_rows,
        top=380.0,
        geometry=_geometry(
            lower_rows,
            top=380.0,
            evidence_prefix="lower",
        ),
    )
    pages = [_page(54, [lower, upper], template="annotations_and_inquiries")]
    reading_order = {54: 1}
    reading_order_resolution = {"status": "ambiguous"}
    if include_prior_institution_group:
        prior_rows = [
            ["缂栧彿", "鏌ヨ鏃ユ湡", "鏌ヨ鏈烘瀯", "鏌ヨ鍘熷洜"],
            ["121", "2024.02.01", "鏍蜂緥閾惰121", "璐峰悗绠＄悊"],
        ]
        prior = _table(
            "pt_53_0",
            prior_rows,
            top=300.0,
            geometry=_geometry(
                prior_rows,
                top=300.0,
                evidence_prefix="prior",
            ),
        )
        pages.insert(
            0,
            _page(53, [prior], template="annotations_and_inquiries"),
        )
        reading_order = {53: 1, 54: 2}
        reading_order_resolution = {
            "resolved": True,
            "authoritative": True,
        }
    monkeypatch.setattr(native_extraction, "_canonical_inquiry_line_rows", lambda _context: [])
    return SimpleNamespace(
        pages=pages,
        evidence_plane=SimpleNamespace(evidence=EvidenceStore(text_atoms=atoms)),
        reading_order_by_logical=reading_order,
        reading_order_resolution=reading_order_resolution,
        corrected_evidence_pages=lambda: [],
        _personal_detail_extraction_issues=[],
    )


def _damaged_prior_schema_headerless_tail_context(
    monkeypatch: pytest.MonkeyPatch,
    *,
    prior_token_defect: str | None = None,
) -> SimpleNamespace:
    context = _production_shaped_headerless_tail_context(monkeypatch)
    upper = next(
        table for table in context.pages[0].tables if table.table_id == "pt_54_0"
    )
    lower_rows = [
        ["编号", "查询日期", "查询机构", "查询原因"],
        *[
            [
                str(sequence),
                f"2025.05.{8 - sequence:02d}",
                "本人",
                "本人查询(自助查询机)",
            ]
            for sequence in range(1, 7)
        ],
    ]
    lower = _table(
        "pt_54_1",
        lower_rows,
        top=380.0,
        geometry=_geometry(lower_rows, top=380.0, evidence_prefix="lower-six"),
    )
    lower.metadata["canonical_template_id"] = "annotations_and_inquiries"
    context.pages[0].tables = [lower, upper]

    prior_rows = [
        ["查询日期", "查询机构", "查询原因", ""],
        ["119", "2024.02.01", "样例银行119", "贷后管理"],
        [
            "120 121",
            "2024.02.07 2024.01.30",
            "甲银行 乙银行",
            "贷后管理 贷后管理",
        ],
    ]
    prior_geometry = _geometry(
        prior_rows,
        top=300.0,
        evidence_prefix="prior-damaged",
    )
    prior_geometry["cell_token_ids"] = deepcopy(
        prior_geometry["cell_evidence_ids"]
    )
    token_texts = (
        ("120", "122" if prior_token_defect == "nonconsecutive" else "121"),
        ("2024.02.07", "2024.01.30"),
        ("甲银行", "乙银行"),
        ("贷后管理", "贷后管理"),
    )
    for column, texts in enumerate(token_texts):
        ids = [f"prior-terminal:{column}:0", f"prior-terminal:{column}:1"]
        prior_geometry["cell_evidence_ids"][2][column] = ids
        prior_geometry["cell_token_ids"][2][column] = list(ids)
        for index, (token_id, text) in enumerate(zip(ids, texts, strict=True)):
            context.evidence_plane.evidence.text_atoms.append(
                EvidenceAtom(
                    id=token_id,
                    text=text,
                    bbox=[
                        50.0 + column * 90.0,
                        321.0 + index * 4.0,
                        120.0 + column * 90.0,
                        324.0 + index * 4.0,
                    ],
                )
            )
    prior = _table(
        "pt_53_0",
        prior_rows,
        top=300.0,
        geometry=prior_geometry,
    )
    context.pages.insert(
        0,
        _page(53, [prior], template="annotations_and_inquiries"),
    )
    context.reading_order_by_logical = {53: 1, 54: 2}
    context.reading_order_resolution = {"resolved": True, "authoritative": True}
    return context


def test_headerless_population_coverage_bridges_exact_prior_terminal_token_band(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _damaged_prior_schema_headerless_tail_context(monkeypatch)

    coverage = native_extraction._inquiry_source_coverage(context)

    assert coverage["sequence_endpoints"] == {
        "institution": 145,
        "personal": 6,
    }
    # The bridge proves terminal ordinals, but the visible institution group
    # starts at 119.  It cannot prove that both subsections restart at one.
    assert coverage["numbering_model"] == "unknown"
    assert "expected_row_count" not in coverage
    institutional = coverage["ordinal_observations"]["institution"]
    assert {"120", "121", "122", "144", "145"} <= set(institutional)
    assert institutional["120"]["source_refs"][0]["geometry_scope"] == "token_row"


def test_headerless_population_coverage_rejects_nonconsecutive_prior_token_bridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _damaged_prior_schema_headerless_tail_context(
        monkeypatch,
        prior_token_defect="nonconsecutive",
    )

    coverage = native_extraction._inquiry_source_coverage(context)

    assert coverage.get("sequence_endpoints", {}).get("institution") != 145
    assert coverage.get("expected_row_count") != 151


def test_production_shaped_headerless_tail_owns_122_through_145_without_promoting_bad_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _production_shaped_headerless_tail_context(monkeypatch)

    rows = native_extraction._extract_inquiries(context)
    coverage = native_extraction._inquiry_source_coverage(context)

    institutional = {
        row["sequence"]: row for row in rows if row["inquiry_type"] == "institution"
    }
    # The exact span witnesses and the independently valid reason cells prove
    # that 131 and 141 are physical records.  Their damaged fields stay null
    # and local instead of causing the whole rows to disappear.
    assert set(institutional) == set(range(122, 146))
    assert institutional[125]["reason"] is None
    assert institutional[125]["extraction_status"] == "review"
    assert "reason" in institutional[125]["_unresolved_fields"]
    assert institutional[144]["institution"] == "上海浦东发展银行股份有限公司"
    assert institutional[145]["institution"] == "中国工商银行股份有限公司"
    for sequence in (131, 141):
        assert institutional[sequence]["inquiry_date"] is None
        assert institutional[sequence]["institution"] is None
        assert institutional[sequence]["reason"] == "贷后管理"
        assert institutional[sequence]["extraction_status"] == "review"
        assert institutional[sequence]["_unresolved_fields"] == [
            "inquiry_date",
            "institution",
        ]
    assert coverage["sequence_endpoints"] == {"institution": 145, "personal": 1}
    assert coverage["numbering_model"] == "unknown"
    assert "expected_row_count" not in coverage
    assert "125" in coverage["ordinal_observations"]["institution"]
    for sequence in (131, 141):
        observation = coverage["ordinal_observations"]["institution"][str(sequence)]
        assert {"inquiry_date", "institution"} <= set(
            observation["printed_fields"]
        )
        assert all(
            ref["geometry_scope"] == "token"
            for field in ("inquiry_date", "institution")
            for ref in observation["field_source_refs"][field]
        )


def test_production_shaped_headerless_population_survives_detached_business_token_plane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replay the final pt54 topology when only ordinal tokens are resolvable.

    The final repaired table can own exact cell/token IDs that are absent from
    the sealed evidence store used by an earlier schema pass.  Exact span
    geometry still proves rows 131/141, and the two sequence tokens still prove
    144/145; none of their unresolved business fields may be invented.
    """

    context = _production_shaped_headerless_tail_context(
        monkeypatch,
        expose_merged_business_tokens=False,
        expose_collapsed_business_tokens=False,
    )

    descriptor = native_extraction._bounded_headerless_inquiry_table_bootstrap(
        context,
        context.pages[0],
        context.pages[0].tables[1],
        native_extraction._table_rows(context.pages[0].tables[1]),
        slots={
            "sequence": 0,
            "inquiry_date": 1,
            "institution": 2,
            "reason": 3,
        },
        later_tables=[context.pages[0].tables[0]],
    )
    rows = native_extraction._extract_inquiries(context)
    coverage = native_extraction._inquiry_source_coverage(context)

    assert descriptor is not None
    assert descriptor["last_sequence"] == 145
    assert coverage["sequence_endpoints"] == {"institution": 145, "personal": 1}
    assert coverage["numbering_model"] == "unknown"
    assert "expected_row_count" not in coverage
    assert set(range(122, 146)) <= {
        int(sequence)
        for sequence in coverage["ordinal_observations"]["institution"]
    }
    institutional = {
        row["sequence"]: row for row in rows if row["inquiry_type"] == "institution"
    }
    assert {144, 145}.isdisjoint(institutional)
    assert {131, 141} <= set(institutional)
    for sequence in (131, 141):
        assert institutional[sequence]["inquiry_date"] is None
        assert institutional[sequence]["institution"] is None
        assert institutional[sequence]["reason"] == "贷后管理"
        assert institutional[sequence]["_unresolved_fields"] == [
            "inquiry_date",
            "institution",
        ]


def _move_headerless_tail_atoms_to_page_bundle(
    context: SimpleNamespace,
    *,
    defect: str | None = None,
) -> None:
    atoms = list(context.evidence_plane.evidence.text_atoms)
    context.evidence_plane.evidence.text_atoms = []
    tokens = [
        {
            "token_id": atom.id,
            "evidence_ids": [atom.id],
            "page": 54,
            "text": atom.text,
            "content": atom.text,
            "bbox": list(atom.bbox or []),
        }
        for atom in atoms
    ]
    bundles = [{"page": 54, "tokens": tokens}]
    if defect == "wrong_page":
        bundles[0]["page"] = 53
    elif defect == "duplicate_id":
        duplicate = dict(next(token for token in tokens if token["token_id"] == "tail:22:0:0"))
        bundles.append({"page": 54, "tokens": [duplicate]})
    context.entities = SimpleNamespace(
        domain_specific={"_page_evidence_bundles": bundles}
    )


def test_production_shaped_headerless_tail_resolves_exact_tokens_from_sealed_page_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _production_shaped_headerless_tail_context(
        monkeypatch,
        include_prior_institution_group=True,
    )
    _move_headerless_tail_atoms_to_page_bundle(context)

    rows = native_extraction._extract_inquiries(context)
    ledger = native_extraction._source_completeness_ledger(context)

    institutional = {
        row["sequence"] for row in rows if row["inquiry_type"] == "institution"
    }
    assert {122, 143, 144, 145} <= institutional
    assert ledger["inquiry_sequence_endpoints"] == {
        "institution": 145,
        "personal": 1,
    }
    assert {"144", "145"} <= set(
        ledger["inquiry_ordinal_observations"]["institution"]
    )


@pytest.mark.parametrize("defect", ["wrong_page", "duplicate_id"])
def test_headerless_page_bundle_token_resolution_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    defect: str,
) -> None:
    context = _production_shaped_headerless_tail_context(
        monkeypatch,
        include_prior_institution_group=True,
    )
    _move_headerless_tail_atoms_to_page_bundle(context, defect=defect)

    rows = native_extraction._extract_inquiries(context)
    ledger = native_extraction._source_completeness_ledger(context)

    institutional = {
        row["sequence"] for row in rows if row["inquiry_type"] == "institution"
    }
    assert {144, 145}.isdisjoint(institutional)
    assert ledger["inquiry_sequence_endpoints"].get("institution") != 145


def test_production_shaped_headerless_tail_extends_active_institution_ledger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replay the real p53-to-p54 runtime state, not a cold p54 bootstrap."""

    context = _production_shaped_headerless_tail_context(
        monkeypatch,
        include_prior_institution_group=True,
    )

    rows = native_extraction._extract_inquiries(context)
    ledger = native_extraction._source_completeness_ledger(context)

    institutional = {
        row["sequence"] for row in rows if row["inquiry_type"] == "institution"
    }
    assert {122, 143, 144, 145} <= institutional
    assert context._candidate_b_canonical_inquiry_line_sequences == {}
    assert ledger["inquiry_sequence_endpoints"] == {
        "institution": 145,
        "personal": 1,
    }
    assert "inquiry_records" not in ledger
    terminal = ledger["inquiry_ordinal_observations"]["institution"]
    assert {"144", "145"} <= set(terminal)
    assert all(
        terminal[ordinal]["source_refs"][0]["binding"]
        == "printed_inquiry_ordinal_token"
        for ordinal in ("144", "145")
    )


@pytest.mark.parametrize(
    "defect",
    [
        "foreign_evidence",
        "overlapping_token_rows",
        "nonconsecutive_ordinals",
        "ambiguous_reason",
        "duplicate_evidence_owner",
        "personal_reason_with_bank",
        "mixed_inquiry_types",
    ],
)
def test_collapsed_inquiry_token_rows_fail_closed(defect: str) -> None:
    rows = native_extraction._extract_inquiries(
        _collapsed_token_inquiry_context(defect=defect)
    )

    assert [row["sequence"] for row in rows] == [143]


def _merged_personal_inquiry_context(*, defect: str | None = None) -> SimpleNamespace:
    rows = [
        ["编号", "查询日期", "查询机构", "查询原因"],
        ["1", "2025.05.07", "本人", "本人查询(自助查询机)"],
        ["2", "本人 2025.02.11", "", "本人查询(自助查询机)"],
    ]
    geometry = _geometry(rows, top=40.0)
    geometry["cell_bboxes"][2][1] = [130.0, 60.0, 310.0, 70.0]
    geometry["cell_bboxes"][2][2] = None
    geometry["cell_geometry_status"][2][2] = "derived"
    geometry["cell_evidence_ids"][2][2] = []
    geometry["cell_spans"] = [
        {
            "row": 2,
            "col": 1,
            "row_span": 1,
            "col_span": 2,
            "bbox": [130.0, 60.0, 310.0, 70.0],
        }
    ]
    geometry["cell_token_ids"] = deepcopy(geometry["cell_evidence_ids"])
    specs = {
        (2, 0): [("2", [50.0, 62.0, 70.0, 67.0])],
        (2, 1): [
            ("2025.02.11", [140.0, 62.0, 190.0, 67.0]),
            ("本人", [235.0, 62.0, 260.0, 67.0]),
        ],
        (2, 3): [("本人查询(自助查询机)", [320.0, 62.0, 390.0, 67.0])],
    }
    atoms: list[EvidenceAtom] = []
    for (row, column), tokens in specs.items():
        ids = [f"personal:{row}:{column}:{index}" for index in range(len(tokens))]
        geometry["cell_evidence_ids"][row][column] = ids
        geometry["cell_token_ids"][row][column] = list(ids)
        for token_id, (text, bbox) in zip(ids, tokens, strict=True):
            atoms.append(EvidenceAtom(id=token_id, text=text, bbox=bbox))
    if defect == "institution_in_date_band":
        next(atom for atom in atoms if atom.text == "本人").bbox = [180.0, 62.0, 200.0, 67.0]
    elif defect == "extra_span_token":
        token_id = "personal:2:1:extra"
        geometry["cell_evidence_ids"][2][1].append(token_id)
        geometry["cell_token_ids"][2][1].append(token_id)
        atoms.append(EvidenceAtom(id=token_id, text="残", bbox=[270.0, 62.0, 280.0, 67.0]))
    elif defect == "nonpersonal_institution":
        next(atom for atom in atoms if atom.text == "本人").text = "某银行"
    elif defect == "wrong_span":
        geometry["cell_spans"][0]["col_span"] = 3
    elif defect == "duplicate_evidence_owner":
        sequence_id = geometry["cell_token_ids"][2][0][0]
        reason_id = geometry["cell_token_ids"][2][3][0]
        geometry["cell_token_ids"][2][3][0] = sequence_id
        geometry["cell_evidence_ids"][2][3][0] = sequence_id
        atoms = [atom for atom in atoms if atom.id != reason_id]
    elif defect == "vertically_displaced_reason":
        next(atom for atom in atoms if atom.id.startswith("personal:2:3:")).bbox = [
            320.0,
            68.5,
            390.0,
            69.5,
        ]

    table = _table("pt_54_1", rows, top=40.0, geometry=geometry)
    return SimpleNamespace(
        pages=[_page(54, [table], template="annotations_and_inquiries")],
        evidence_plane=SimpleNamespace(evidence=EvidenceStore(text_atoms=atoms)),
        reading_order_by_logical={54: 1},
        reading_order_resolution={"resolved": True, "authoritative": True},
        corrected_evidence_pages=lambda: [],
        _personal_detail_extraction_issues=[],
    )


def test_exact_tokens_recover_personal_inquiry_from_date_institution_span() -> None:
    rows = native_extraction._extract_inquiries(_merged_personal_inquiry_context())

    assert [(row["sequence"], row["inquiry_date"], row["institution"]) for row in rows] == [
        (1, "2025-05-07", "本人"),
        (2, "2025-02-11", "本人"),
    ]
    assert rows[1]["source"] == "native_detail_inquiry_token_rows"
    assert rows[1]["source_refs_by_field"]["institution"][0]["geometry_scope"] == "token"
    assert rows[1]["reason"] == rows[1]["source_reason"]


@pytest.mark.parametrize(
    "defect",
    [
        "institution_in_date_band",
        "extra_span_token",
        "nonpersonal_institution",
        "wrong_span",
        "duplicate_evidence_owner",
        "vertically_displaced_reason",
    ],
)
def test_personal_date_institution_span_recovery_fails_closed(defect: str) -> None:
    context = _merged_personal_inquiry_context(defect=defect)
    rows = native_extraction._extract_inquiries(context)

    assert [row["sequence"] for row in rows] == [1, 2]
    unresolved = rows[1]
    assert unresolved["inquiry_type"] == "personal"
    assert unresolved["inquiry_date"] is None
    assert unresolved["institution"] is None
    assert unresolved["reason"] == "本人查询(自助查询机)"
    assert unresolved["extraction_status"] == "review"
    assert unresolved["_unresolved_fields"] == [
        "inquiry_date",
        "institution",
    ]
    issues = [
        issue
        for issue in context._personal_detail_extraction_issues
        if issue.get("target_record_id") == unresolved["inquiry_id"]
        and issue.get("issue_code") == "candidate_b_inquiry_row_cells_unresolved"
    ]
    assert {issue["field_name"] for issue in issues} == {
        "inquiry_date",
        "institution",
    }
    assert all(
        issue["reason_codes"][-2:]
        == ["physical_record_identity_conserved", "field_local_value_withheld"]
        for issue in issues
    )


def test_token_split_helpers_reject_duplicate_slot_ownership() -> None:
    collapsed = _collapsed_token_inquiry_context()
    collapsed_table = collapsed.pages[0].tables[0]
    merged = _merged_personal_inquiry_context()
    merged_table = merged.pages[0].tables[0]
    duplicate_slots = {
        "sequence": 0,
        "inquiry_date": 1,
        "institution": 1,
        "reason": 3,
    }

    assert (
        native_extraction._bounded_two_row_inquiry_cell_split(
            collapsed,
            collapsed_table,
            row_index=2,
            slots=duplicate_slots,
        )
        is None
    )
    assert (
        native_extraction._bounded_personal_inquiry_merged_institution_row(
            merged,
            merged_table,
            row_index=2,
            slots=duplicate_slots,
        )
        is None
    )


def test_personal_token_recovery_preserves_raw_reason_channel() -> None:
    rows = native_extraction._extract_inquiries(_merged_personal_inquiry_context())

    assert rows[1]["reason"] == "本人查询(自助查询机)"
    assert rows[1]["source_reason"] == "本人查询(自助查询机)"


def test_exact_inquiry_field_ref_accepts_only_complete_token_contract() -> None:
    exact_token_ref = {
        "source": "native_detail_inquiry_token",
        "geometry_scope": "token",
        "binding": "canonical_header_column_token",
        "binding_quality": "exact_token_in_canonical_cell",
        "field_name": "inquiry_date",
        "table_id": "pt_54_1",
        "row": 2,
        "column": 1,
        "bbox": [140.0, 62.0, 190.0, 67.0],
        "evidence_ids": ["token:1"],
    }

    assert native_extraction._exact_inquiry_field_ref(
        exact_token_ref,
        field_name="inquiry_date",
    )
    for field_name in ("source", "binding_quality", "field_name", "bbox", "evidence_ids"):
        damaged = dict(exact_token_ref)
        damaged.pop(field_name)
        assert not native_extraction._exact_inquiry_field_ref(
            damaged,
            field_name="inquiry_date",
        )
    assert not native_extraction._exact_inquiry_field_ref(
        {**exact_token_ref, "evidence_ids": ["token:1", "token:2"]},
        field_name="inquiry_date",
    )


def test_personal_token_ordinals_enter_source_coverage_with_field_local_row() -> None:
    context = _merged_personal_inquiry_context(defect="extra_span_token")

    rows = native_extraction._extract_inquiries(context)
    coverage = native_extraction._inquiry_source_coverage(context)

    assert [row["sequence"] for row in rows] == [1, 2]
    assert rows[1]["inquiry_date"] is None
    assert rows[1]["institution"] is None
    assert rows[1]["reason"] == "本人查询(自助查询机)"
    assert rows[1]["extraction_status"] == "review"
    assert coverage["sequence_endpoints"] == {"personal": 2}
    assert coverage["observed_sequences"] == {"personal": [1, 2]}
    omitted = coverage["ordinal_observations"]["personal"]["2"]
    assert omitted["printed_fields"] == []
    assert omitted["field_source_refs"] == {}
    assert omitted["source_refs"][0]["binding"] == "printed_inquiry_ordinal_token"


def _two_personal_ordinal_tokens_partition_record_identities_only() -> None:
    context = _collapsed_token_inquiry_context()
    table = context.pages[0].tables[0]
    table.metadata["raw_rows"][1] = [
        "1",
        "2025.05.07",
        "鏈汉",
        "鏈汉鏌ヨ(鑷姪鏌ヨ鏈?",
    ]
    sequence_atoms = [
        atom
        for atom in context.evidence_plane.evidence.text_atoms
        if atom.id.startswith("merged:2:0:")
    ]
    sequence_atoms[0].text = "2"
    sequence_atoms[1].text = "3"
    for atom in context.evidence_plane.evidence.text_atoms:
        if atom.id.startswith("merged:2:2:"):
            atom.text = "鏈汉"
        elif atom.id.startswith("merged:2:3:"):
            atom.text = "鏈汉鏌ヨ(鑷姪鏌ヨ鏈?"

    observations = native_extraction._bounded_inquiry_token_ordinal_observations(
        context,
        context.pages[0],
        table,
        row_index=2,
        slots={
            "sequence": 0,
            "inquiry_date": 1,
            "institution": 2,
            "reason": 3,
        },
    )

    assert [observation["sequence"] for observation in observations] == [2, 3]
    assert all(observation["inquiry_type"] == "personal" for observation in observations)


@pytest.mark.parametrize(
    "defect",
    ["duplicate_evidence_owner", "nonconsecutive_ordinals", "foreign_evidence"],
)
def test_token_ordinal_source_coverage_fails_closed(defect: str) -> None:
    context = _collapsed_token_inquiry_context(defect=defect)
    table = context.pages[0].tables[0]
    table.metadata["raw_rows"][1] = [
        "1",
        "2025.05.07",
        "鏈汉",
        "鏈汉鏌ヨ(鑷姪鏌ヨ鏈?",
    ]
    for atom in context.evidence_plane.evidence.text_atoms:
        if atom.id.startswith("merged:2:0:0"):
            atom.text = "2"
        elif atom.id.startswith("merged:2:0:1") and defect != "nonconsecutive_ordinals":
            atom.text = "3"
        elif atom.id.startswith("merged:2:2:"):
            atom.text = "鏈汉"
        elif atom.id.startswith("merged:2:3:"):
            atom.text = "鏈汉鏌ヨ(鑷姪鏌ヨ鏈?"

    coverage = native_extraction._inquiry_source_coverage(context)

    assert coverage.get("sequence_endpoints") != {"personal": 3}


@pytest.mark.parametrize(
    "defect",
    [
        None,
        "duplicate_evidence_owner",
        "reversed_ordinals",
        "nonconsecutive_ordinals",
        "wrong_channel",
        "foreign_table",
    ],
)
def test_huang_personal_double_band_reports_identities_only(defect: str | None) -> None:
    context = _collapsed_token_inquiry_context()
    table = context.pages[0].tables[0]
    table.table_id = "pt_54_1"
    table.metadata["raw_rows"][0] = ["编号", "查询日期", "查询机构", "查询原因"]
    table.metadata["raw_rows"][1] = ["1", "2025.05.07", "本人", "本人查询(自助查询机)"]
    sequence_atoms = [
        atom
        for atom in context.evidence_plane.evidence.text_atoms
        if atom.id.startswith("merged:2:0:")
    ]
    sequence_atoms[0].text = "3"
    sequence_atoms[1].text = "4"
    institution_atoms = [
        atom
        for atom in context.evidence_plane.evidence.text_atoms
        if atom.id.startswith("merged:2:2:")
    ]
    institution_atoms[0].text = "本人"
    institution_atoms[1].text = "本人"
    reason_atoms = [
        atom
        for atom in context.evidence_plane.evidence.text_atoms
        if atom.id.startswith("merged:2:3:")
    ]
    reason_atoms[0].text = "本人查询(自助查询机)"
    reason_atoms[1].text = "银行)本人查询(商业银行网上"

    if defect == "duplicate_evidence_owner":
        duplicate_id = table.metadata["geometry"]["cell_token_ids"][2][0][0]
        replaced_id = table.metadata["geometry"]["cell_token_ids"][2][1][0]
        table.metadata["geometry"]["cell_token_ids"][2][1][0] = duplicate_id
        table.metadata["geometry"]["cell_evidence_ids"][2][1][0] = duplicate_id
        context.evidence_plane.evidence.text_atoms = [
            atom
            for atom in context.evidence_plane.evidence.text_atoms
            if atom.id != replaced_id
        ]
    elif defect == "reversed_ordinals":
        sequence_atoms[0].text, sequence_atoms[1].text = "4", "3"
    elif defect == "nonconsecutive_ordinals":
        sequence_atoms[1].text = "5"
    elif defect == "wrong_channel":
        reason_atoms[1].text = "贷款审批"
    elif defect == "foreign_table":
        context.pages[0].canonical_template_id = "credit_accounts"

    coverage = native_extraction._inquiry_source_coverage(context)

    if defect is None:
        assert coverage["observed_sequences"]["personal"] == [1, 3, 4]
        for ordinal in ("3", "4"):
            observation = coverage["ordinal_observations"]["personal"][ordinal]
            assert observation["printed_fields"] == []
            assert observation["field_source_refs"] == {}
            assert observation["source_refs"][0]["source_band_bbox"]
    else:
        assert not ({3, 4} <= set(coverage.get("observed_sequences", {}).get("personal", ())))


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
    assert field_issues == {
        "account_identifier",
        "open_date",
        "credit_limit",
        "shared_credit_limit",
        "business_type",
        "guarantee_type",
        "snapshot_date",
        "account_state",
    }
    assert any(
        issue.get("issue_code") == "candidate_b_account_table_missing"
        and issue.get("field_name") is None
        for issue in context._personal_detail_extraction_issues
    )


def test_missing_loan_table_reports_both_printed_template_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skeleton = {
        "account_id": "credit_account:non_revolving_loan:9",
        "sequence": 9,
        "category_sequence": 9,
        "account_type": "non_revolving_loan",
        "management_institution": "中国建设银行股份有限公司厦门市分行",
        "source": "candidate_b_account_anchor",
        "source_refs": [_account_anchor_ref()],
        "_source_absent_fields": ["co_borrower_flag"],
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
    assert field_issues == {
        "account_identifier",
        "open_date",
        "due_date",
        "loan_amount",
        "account_currency",
        "business_type",
        "guarantee_type",
        "repayment_periods",
        "repayment_frequency",
        "repayment_method",
        "snapshot_date",
        "account_state",
    }


def test_native_detail_account_reports_missing_template_fields_but_not_printed_absence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    account = {
        "account_id": "credit_account:non_revolving_loan:4",
        "sequence": 4,
        "category_sequence": 4,
        "account_type": "non_revolving_loan",
        "management_institution": "中国银行股份有限公司厦门市分行",
        "account_identifier": "D10111000H000000000000000000000004",
        "open_date": "2024-01-01",
        "currency": "CNY",
        "account_currency": "CNY",
        "account_state": "unknown",
        "source": "native_detail_account_table",
        "source_refs": [_account_anchor_ref()],
        "_source_absent_fields": ["co_borrower_flag"],
    }
    skeleton = {
        "account_id": "credit_account:non_revolving_loan:4",
        "sequence": 4,
        "category_sequence": 4,
        "account_type": "non_revolving_loan",
        "account_family_quality": "exact",
        "_printed_ordinal_status": "printed_unique",
        "source": "candidate_b_account_anchor",
        "source_refs": [_account_anchor_ref()],
        "_source_absent_fields": ["co_borrower_flag"],
    }
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    monkeypatch.setattr(
        native_extraction,
        "_extract_table_accounts",
        lambda _context: ([deepcopy(account)], [], []),
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
    field_issues = {
        issue.get("field_name")
        for issue in context._personal_detail_extraction_issues
        if issue.get("issue_code")
        == "candidate_b_account_required_field_unresolved"
    }
    assert field_issues == {
        "due_date",
        "loan_amount",
        "business_type",
        "guarantee_type",
        "repayment_periods",
        "repayment_frequency",
        "repayment_method",
        "snapshot_date",
        "account_state",
    }
    assert "co_borrower_flag" not in field_issues
    assert set(accounts[0]["_unresolved_fields"]) == field_issues
