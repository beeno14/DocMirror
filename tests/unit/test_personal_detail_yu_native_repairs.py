from __future__ import annotations

from types import SimpleNamespace

import pytest

from docmirror.plugins.credit_report.personal_detail_scanned import (
    native_extraction,
)

_CARD_HEADER = [
    "发卡机构",
    "账户标识",
    "开立日期",
    "账户授信额度",
    "共享授信额度",
    "币种",
    "业务种类",
    "担保方式",
]


def _table(
    table_id: str,
    rows: list[list[str]],
    *,
    top: float,
    bottom: float | None = None,
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
        bbox=[50.0, top, 402.0, bottom if bottom is not None else top + 60.0],
        confidence=0.99,
    )


def _page(
    logical_page: int,
    tables: list[SimpleNamespace],
    *,
    template: str = "credit_account_detail",
) -> SimpleNamespace:
    return SimpleNamespace(
        page_number=logical_page,
        source_page_number=(logical_page + 1) // 2,
        canonical_template_id=template,
        tables=tables,
        texts=[],
        height=595.0,
    )


def _line(text: str, bbox: list[float], evidence_id: str) -> dict:
    return {"text": text, "bbox": bbox, "evidence_ids": [evidence_id]}


def _yu_account_context(
    *,
    second_ordinal: int = 13,
    include_partition_heading: bool = True,
    damage_second_card_table: bool = False,
) -> SimpleNamespace:
    prior_card = _table(
        "pt_13_2",
        [
            _CARD_HEADER,
            [
                "中国建设银行股份有限公司厦门市分行",
                "CR6221682281065024",
                "2014.09.30",
                "4,999",
                "4,999",
                "人民币元",
                "贷记卡",
                "信用/免担保",
            ],
        ],
        top=464.5,
        bottom=546.5,
    )
    leading_months = _table(
        "pt_14_0",
        [
            ["", "", "2017年08月-2018年07月的还款记录", ""],
            ["2018", "N", "N", "C"],
        ],
        top=37.5,
        bottom=120.5,
    )
    card_12 = _table(
        "pt_14_1",
        [
            _CARD_HEADER,
            [
                "中国农业银行股份有限公司",
                "40321001660006384",
                "2016.01.13",
                "500",
                "0",
                "人民币元",
                "贷记卡",
                "信用/免担保",
            ],
        ],
        top=140.0,
        bottom=215.0,
    )
    second_header = list(_CARD_HEADER)
    if damage_second_card_table:
        second_header[6] = ""
    card_13 = _table(
        "pt_14_2",
        [
            second_header,
            [
                "交通银行股份有限公司太平洋信用卡中心",
                "B10512900H00010790011130206473619",
                "2017.08.08",
                "10",
                "0",
                "人民币元",
                "贷记卡",
                "信用/免担保",
            ],
        ],
        top=236.5,
        bottom=297.5,
    )
    quasi = _table(
        "pt_14_3",
        [
            [
                "发卡机构",
                "账户标识",
                "开立日期",
                "账户授信额度",
                "共享授信额度",
                "币种",
                "担保方式",
            ],
            [
                "中国工商银行股份有限公司厦门市分行",
                "B10111000H000141005000220002430",
                "2014.07.08",
                "0",
                "0",
                "人民币元",
                "信用/免担保",
            ],
        ],
        top=333.0,
        bottom=562.0,
    )
    evidence_13 = {
        "page": 13,
        "source_page": 7,
        "lines": [
            _line("（二）贷记卡账户", [52.0, 150.0, 180.0, 163.0], "card-family"),
            _line("账户11", [52.0, 452.0, 95.0, 463.0], "card-11"),
        ],
    }
    evidence_14_lines = [
        _line("账户12", [54.5, 131.0, 77.0, 140.0], "card-12"),
        _line(
            f"账户{second_ordinal}(授信协议标识:B10512900H00010010011135264974289)(卡片尾号:2115)",
            [53.0, 226.0, 291.5, 238.0],
            "card-13",
        ),
    ]
    if include_partition_heading:
        evidence_14_lines.append(
            _line("(三)准贷记卡账户", [193.5, 309.5, 255.5, 322.5], "quasi-family")
        )
        evidence_14_lines.append(
            _line(
                "账户(授信协议标识:B10111000H000141005000530084646)(卡片尾号:2185)",
                [52.5, 322.5, 277.5, 335.0],
                "quasi-account",
            )
        )
    evidence = [
        evidence_13,
        {"page": 14, "source_page": 7, "lines": evidence_14_lines},
    ]
    pages = [
        _page(13, [prior_card]),
        _page(14, [leading_months, card_12, card_13, quasi]),
    ]
    return SimpleNamespace(
        pages=pages,
        reading_order_by_logical={13: 1, 14: 2},
        reading_order_resolution={"resolved": True, "authoritative": True},
        corrected_evidence_pages=lambda: evidence,
        allows_scanned_line_transition=lambda *_args: True,
        tables_continue=lambda *_args: None,
        _personal_detail_extraction_issues=[],
    )


def _table_observation(
    account_type: str,
    table_id: str,
    top: float,
    identifier: str,
) -> dict:
    return {
        "account_id": f"credit_account_table_observation:{table_id}",
        "_table_observation_id": f"credit_account_table_observation:{table_id}",
        "account_type": account_type,
        "account_identifier": identifier,
        "source": "native_detail_account_table",
        "source_refs": [
            {
                "source": "native_detail_table",
                "logical_page": 14,
                "source_page": 7,
                "table_id": table_id,
                "bbox": [50.0, top, 402.0, top + 60.0],
            }
        ],
    }


def test_yu_mixed_page_recovers_dense_card_12_and_13_anchors() -> None:
    context = _yu_account_context()

    skeletons = native_extraction._account_anchor_skeletons(context)
    cards = [row for row in skeletons if row.get("account_type") == "credit_card"]

    assert [row["category_sequence"] for row in cards] == [11, 12, 13]
    observations = [
        _table_observation(
            "credit_card",
            "pt_14_1",
            140.0,
            "40321001660006384",
        ),
        _table_observation(
            "credit_card",
            "pt_14_2",
            236.5,
            "B10512900H00010790011130206473619",
        ),
    ]
    matches = native_extraction._match_account_table_observations(cards, observations)
    assert matches == {1: 0, 2: 1}

    ledger = native_extraction._source_completeness_ledger(context)
    assert ledger["account_family_endpoints"]["credit_card"] == 13


@pytest.mark.parametrize(
    "defect",
    ["non_dense_ordinal", "missing_partition", "wrong_prefix_morphology"],
)
def test_yu_credit_card_prefix_carry_fails_closed(defect: str) -> None:
    context = _yu_account_context(
        second_ordinal=14 if defect == "non_dense_ordinal" else 13,
        include_partition_heading=defect != "missing_partition",
        damage_second_card_table=defect == "wrong_prefix_morphology",
    )

    skeletons = native_extraction._account_anchor_skeletons(context)
    cards = [row for row in skeletons if row.get("account_type") == "credit_card"]

    assert [row["category_sequence"] for row in cards] == [11]


def _split_personal_geometry(*, body_count: int = 2) -> dict:
    row_bands = [
        {"index": 0, "y0": 254.5, "y1": 257.5},
        {"index": 1, "y0": 257.5, "y1": 268.5},
        *[
            {
                "index": row_index,
                "y0": 268.5 + (row_index - 2) * 13.0,
                "y1": 268.5 + (row_index - 1) * 13.0,
            }
            for row_index in range(2, 2 + body_count)
        ],
        {
            "index": body_count + 2,
            "y0": 268.5 + body_count * 13.0,
            "y1": 271.5 + body_count * 13.0,
        },
    ]
    cell_bboxes: list[list[list[float] | None]] = [
        [
            [52.0, 254.5, 92.0, 268.5],
            [92.0, 254.5, 171.0, 257.5],
            [171.0, 254.5, 324.0, 257.5],
            [324.0, 254.5, 401.5, 257.5],
        ],
        [
            None,
            [92.0, 257.5, 171.0, 268.5],
            [171.0, 257.5, 324.0, 268.5],
            [324.0, 257.5, 401.5, 268.5],
        ],
    ]
    cell_status = [
        ["exact", "exact", "exact", "exact"],
        ["derived", "exact", "exact", "exact"],
    ]
    cell_evidence = [
        [["header:sequence"], [], [], []],
        [[], ["header:date"], ["header:institution"], ["header:reason"]],
    ]
    for sequence in range(1, body_count + 1):
        band = row_bands[sequence + 1]
        cell_bboxes.append(
            [
                [52.0, band["y0"], 92.0, band["y1"]],
                [92.0, band["y0"], 171.0, band["y1"]],
                [171.0, band["y0"], 324.0, band["y1"]],
                [324.0, band["y0"], 401.5, band["y1"]],
            ]
        )
        cell_status.append(["exact", "exact", "exact", "exact"])
        cell_evidence.append(
            [
                [f"row:{sequence}:sequence"],
                [f"row:{sequence}:date"],
                [f"row:{sequence}:institution"],
                [f"row:{sequence}:reason"],
            ]
        )
    trailing_band = row_bands[-1]
    cell_bboxes.append(
        [
            [52.0, trailing_band["y0"], 92.0, trailing_band["y1"]],
            [92.0, trailing_band["y0"], 171.0, trailing_band["y1"]],
            [171.0, trailing_band["y0"], 324.0, trailing_band["y1"]],
            [324.0, trailing_band["y0"], 401.5, trailing_band["y1"]],
        ]
    )
    cell_status.append(["exact", "exact", "exact", "derived"])
    cell_evidence.append([[], [], [], []])
    return {
        "row_bands": row_bands,
        "col_bands": [
            {"index": 0, "x0": 52.0, "x1": 92.0},
            {"index": 1, "x0": 92.0, "x1": 171.0},
            {"index": 2, "x0": 171.0, "x1": 324.0},
            {"index": 3, "x0": 324.0, "x1": 401.5},
        ],
        "cell_bboxes": cell_bboxes,
        "cell_geometry_status": cell_status,
        "cell_evidence_ids": cell_evidence,
        "cell_spans": [
            {
                "row": 0,
                "col": 0,
                "row_span": 2,
                "col_span": 1,
                "bbox": [52.0, 254.5, 92.0, 268.5],
            }
        ],
    }


def _yu_inquiry_context(
    *,
    defect: str | None = None,
    personal_count: int = 2,
) -> SimpleNamespace:
    institution_rows = [["编号", "查询日期", "查询机构", "查询原因"]]
    institution_rows.extend(
        [str(sequence), "2022.04.26", f"机构{sequence}", "贷款审批"]
        for sequence in range(1, 25)
    )
    personal_rows = [
        ["编号", "", "", ""],
        ["", "查询日期", "查询机构", "查询原因"],
        *[
            [
                str(sequence),
                (
                    ["2022.06.13", "2022.02.18"][sequence - 1]
                    if personal_count == 2
                    else f"2022.{((sequence - 1) % 12) + 1:02d}.13"
                ),
                "本人",
                "本人查询(自助查询机)",
            ]
            for sequence in range(1, personal_count + 1)
        ],
        ["", "", "", ""],
    ]
    geometry = _split_personal_geometry(body_count=personal_count)
    if defect == "transposed_header":
        personal_rows[1][1], personal_rows[1][2] = personal_rows[1][2], personal_rows[1][1]
    elif defect == "missing_row_span":
        geometry["cell_spans"] = []
    elif defect == "non_personal_body":
        personal_rows[3][2] = "某银行"
    elif defect == "sequence_gap":
        personal_rows[-2][0] = str(personal_count + 1)
    elif defect == "duplicate_sequence":
        personal_rows[-2][0] = str(max(1, personal_count - 1))
    elif defect == "outlier_sequence":
        personal_rows[-2][0] = "999"
    elif defect == "inexact_body_cell":
        geometry["cell_geometry_status"][-2][2] = "derived"
    elif defect == "ambiguous_header_span":
        geometry["cell_spans"].append(
            {
                "row": 0,
                "col": 1,
                "row_span": 2,
                "col_span": 1,
                "bbox": geometry["cell_bboxes"][0][1],
            }
        )
    elif defect == "extra_section_after_blank":
        personal_rows.append(
            ["1", "2021.12.01", "本人", "本人查询(自助查询机)"]
        )
    elif defect == "damaged_first_ordinal":
        personal_rows[2][0] = "?"
    elif defect == "second_unreadable_ordinal":
        personal_rows[2][0] = "?"
        personal_rows[3][0] = "#"
    elif defect == "alphanumeric_first_residue":
        personal_rows[2][0] = "A?"
    tables = [
        _table("pt_16_4", institution_rows, top=430.0, bottom=561.5),
        _table(
            "pt_17_1",
            personal_rows,
            top=254.5,
            bottom=298.0,
            geometry=geometry,
        ),
    ]
    for table in tables:
        table.metadata["canonical_template_id"] = "annotations_and_inquiries"
    return SimpleNamespace(
        pages=[_page(17, tables, template="annotations_and_inquiries")],
        corrected_evidence_pages=lambda: [],
        _personal_detail_extraction_issues=[],
    )


def test_yu_split_personal_inquiry_header_emits_both_channels_and_coverage() -> None:
    context = _yu_inquiry_context()

    rows = native_extraction._extract_inquiries(context)
    coverage = native_extraction._inquiry_source_coverage(context)

    institutional = [row for row in rows if row["inquiry_type"] == "institution"]
    personal = [row for row in rows if row["inquiry_type"] == "personal"]
    assert [row["sequence"] for row in institutional] == list(range(1, 25))
    assert [row["sequence"] for row in personal] == [1, 2]
    assert [row["inquiry_date"] for row in personal] == ["2022-06-13", "2022-02-18"]
    assert coverage["sequence_endpoints"] == {"institution": 24, "personal": 2}
    assert coverage["numbering_model"] == "unknown"
    assert "expected_row_count" not in coverage


@pytest.mark.parametrize("personal_count", [1, 4, 7])
def test_split_personal_inquiry_header_derives_variable_dense_population(
    personal_count: int,
) -> None:
    context = _yu_inquiry_context(personal_count=personal_count)

    rows = native_extraction._extract_inquiries(context)
    coverage = native_extraction._inquiry_source_coverage(context)

    personal = [row for row in rows if row["inquiry_type"] == "personal"]
    assert [row["sequence"] for row in personal] == list(
        range(1, personal_count + 1)
    )
    assert coverage["sequence_endpoints"]["personal"] == personal_count
    assert coverage["numbering_model"] == "unknown"
    assert "expected_row_count" not in coverage


@pytest.mark.parametrize("personal_count", [2, 3, 4, 7])
def test_split_personal_inquiry_header_repairs_one_damaged_first_ordinal_for_variable_population(
    personal_count: int,
) -> None:
    context = _yu_inquiry_context(
        defect="damaged_first_ordinal",
        personal_count=personal_count,
    )

    rows = native_extraction._extract_inquiries(context)
    coverage = native_extraction._inquiry_source_coverage(context)

    personal = [row for row in rows if row["inquiry_type"] == "personal"]
    assert [row["sequence"] for row in personal] == list(range(1, personal_count + 1))
    assert coverage["sequence_endpoints"]["personal"] == personal_count


@pytest.mark.parametrize(
    "defect",
    [
        "sequence_gap",
        "duplicate_sequence",
        "second_unreadable_ordinal",
        "alphanumeric_first_residue",
        "inexact_body_cell",
        "extra_section_after_blank",
    ],
)
def test_split_personal_damaged_first_ordinal_fails_closed_without_complete_exact_lattice(
    defect: str,
) -> None:
    context = _yu_inquiry_context(defect=defect, personal_count=4)
    if defect in {"sequence_gap", "duplicate_sequence", "inexact_body_cell", "extra_section_after_blank"}:
        context.pages[0].tables[1].metadata["raw_rows"][2][0] = "?"

    rows = native_extraction._extract_inquiries(context)
    coverage = native_extraction._inquiry_source_coverage(context)

    assert not [row for row in rows if row["inquiry_type"] == "personal"]
    assert coverage["sequence_endpoints"] == {"institution": 24}


@pytest.mark.parametrize(
    "defect",
    [
        "transposed_header",
        "missing_row_span",
        "non_personal_body",
        "sequence_gap",
        "duplicate_sequence",
        "outlier_sequence",
        "inexact_body_cell",
        "ambiguous_header_span",
        "extra_section_after_blank",
    ],
)
def test_yu_split_personal_inquiry_header_fails_closed(defect: str) -> None:
    context = _yu_inquiry_context(defect=defect, personal_count=4)

    rows = native_extraction._extract_inquiries(context)
    coverage = native_extraction._inquiry_source_coverage(context)

    assert len(rows) == 24
    assert {row["inquiry_type"] for row in rows} == {"institution"}
    assert coverage["sequence_endpoints"] == {"institution": 24}
    assert coverage["expected_row_count"] == 24
