from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from docmirror.plugins.credit_report.personal_detail_scanned import native_extraction
from docmirror.plugins.credit_report.personal_detail_scanned.canonical_layout import (
    PBOCCanonicalTemplateAssembler,
    _mixed_account_liability_table_owners,
    _sealed_liability_page_continuation_proved,
)
from docmirror.plugins.credit_report.personal_detail_scanned.native_parser import (
    PBOCPersonalDetailNativeParser,
)

LIABILITY_ROLES = (
    "管理机构",
    "业务种类",
    "开立日期",
    "到期日期",
    "责任人类型",
    "还款责任金额",
    "币种",
    "保证合同编号",
)

ACCOUNT_ROLES = (
    "发卡机构",
    "账户标识",
    "开立日期",
    "账户授信额度",
    "共享授信额度",
    "币种",
    "业务种类",
    "担保方式",
)


def _line(
    text: str,
    bbox: list[float],
    *,
    evidence_id: str,
) -> dict[str, object]:
    return {"text": text, "bbox": bbox, "evidence_ids": [evidence_id]}


def _page(number: int, tables: list[SimpleNamespace]) -> SimpleNamespace:
    return SimpleNamespace(
        page_number=number,
        source_page_number=number,
        width=560.0,
        height=700.0,
        tables=tables,
        texts=[],
    )


def _evidence(
    number: int,
    printed_page: int,
    *lines: dict[str, object],
) -> dict[str, object]:
    return {
        "page": number,
        "source_page": number,
        "page_width": 560.0,
        "page_height": 700.0,
        "lines": [
            *lines,
            _line(
                f"第{printed_page}页，共24页",
                [220.0, 660.0, 340.0, 680.0],
                evidence_id=f"footer:{number}",
            ),
        ],
    }


def _grid_table(
    table_id: str,
    *,
    top: float,
    header: list[str],
    values: list[str],
    widths: list[float],
    extra_rows: list[list[str]] | None = None,
    exact_mask: list[list[bool]] | None = None,
) -> SimpleNamespace:
    assert len(header) == len(values) == len(widths)
    rows = [header, values, *(extra_rows or [])]
    assert all(len(row) == len(widths) for row in rows)
    exact_mask = exact_mask or [
        [bool(value) for value in row]
        for row in rows
    ]
    assert len(exact_mask) == len(rows)
    assert all(len(row) == len(widths) for row in exact_mask)
    left = 30.0
    scale = 500.0 / sum(widths)
    bands: list[tuple[float, float]] = []
    for width in widths:
        right = left + width * scale
        bands.append((left, right))
        left = right
    row_height = 48.0 / len(rows)
    cell_bboxes = [
        [
            [x0, top + row * row_height, x1, top + (row + 1) * row_height]
            for x0, x1 in bands
        ]
        for row in range(len(rows))
    ]
    statuses = [
        [
            "exact" if exact_mask[row][column] else "derived"
            for column in range(len(header))
        ]
        for row in range(len(rows))
    ]
    evidence_ids = [
        [
            [f"{table_id}:r{row}:c{column}"]
            if exact_mask[row][column] and bool(rows[row][column])
            else []
            for column in range(len(header))
        ]
        for row in range(len(rows))
    ]
    geometry = {
        "cell_bboxes": deepcopy(cell_bboxes),
        "cell_geometry_status": deepcopy(statuses),
        "cell_evidence_ids": deepcopy(evidence_ids),
        "coordinate_system": "pdf_points_top_left",
    }
    return SimpleNamespace(
        table_id=table_id,
        bbox=[30.0, top, 530.0, top + 48.0],
        headers=[],
        rows=[],
        metadata={
            "raw_rows": deepcopy(rows),
            "cell_bboxes": deepcopy(cell_bboxes),
            "cell_geometry_status": deepcopy(statuses),
            "cell_evidence_ids": deepcopy(evidence_ids),
            "geometry": geometry,
        },
    )


def _liability_table(
    sequence: int,
    *,
    top: float,
    reordered: bool,
    merged_slots: bool,
    optional_blanks: bool = False,
    compact_graph: bool = False,
) -> SimpleNamespace:
    role_order = (
        tuple(reversed(LIABILITY_ROLES))
        if reordered
        else LIABILITY_ROLES
    )
    label_by_role = {role: role for role in LIABILITY_ROLES}
    if sequence % 2 == 0:
        label_by_role["开立日期"] = "成立日期"
    value_by_role = {
        "管理机构": f"测试银行股份有限公司{sequence}支行",
        "业务种类": "贷款",
        "开立日期": f"2023.{sequence:02d}.01",
        "到期日期": f"2028.{sequence:02d}.01",
        "责任人类型": "共同借款人" if sequence == 4 else "保证",
        "还款责任金额": "" if optional_blanks else str(sequence * 100000),
        "币种": "人民币元",
        "保证合同编号": "" if optional_blanks else f"GENERIC-CONTRACT-{sequence:03d}",
    }
    header: list[str] = []
    values: list[str] = []
    role_at_column: list[str | None] = []
    for role in role_order:
        header.append(label_by_role[role])
        values.append(value_by_role[role])
        role_at_column.append(role)
        if merged_slots and role in {"开立日期", "还款责任金额"}:
            header.append("")
            values.append("")
            role_at_column.append(None)
    base_exact_mask = [
        [role is not None for role in role_at_column],
        [
            role is not None
            for role in role_at_column
        ],
    ]
    column_count = len(header)
    party_header = [""] * column_count
    party_values = [""] * column_count
    if compact_graph:
        party_header[0] = (
            "0主业务借款人证件类型"
            "主业务借款人证件号码"
            "主业务借款人"
        )
        party_values[0] = f"测试借款人{sequence} 中征码 532901000000{sequence:04d}"
    else:
        party_columns = (0, 3, 7) if column_count == 10 else (0, 3, 6)
        party_header[party_columns[0]] = {
            2: "民主业务借款人",
            4: "2主业务借款人",
        }.get(sequence, "主业务借款人")
        party_header[party_columns[1]] = (
            "多主业务借款人证件类型"
            if sequence == 3
            else "主业务借款人证件类型"
        )
        party_header[party_columns[2]] = "主业务借款人证件号码"
        party_values[party_columns[0]] = f"测试借款人{sequence}"
        party_values[party_columns[1]] = "中征码"
        party_values[party_columns[2]] = f"532901000000{sequence:04d}"
    snapshot = [""] * column_count
    snapshot[0] = (
        "2" if sequence == 2 else "人" if sequence == 80 else ""
    ) + f"截至2024年07月{sequence:02d}日"
    extra_rows = [party_header, party_values, snapshot]
    if not compact_graph:
        status_columns = (0, 3, 7) if column_count == 10 else (0, 3, 6)
        status_header = [""] * column_count
        status_values = [""] * column_count
        status_header[status_columns[0]] = "余额"
        status_header[status_columns[1]] = (
            "五级分类 囍" if sequence == 1 else "五级分类"
        )
        status_header[status_columns[2]] = "逾期月数"
        status_values[status_columns[0]] = str(sequence * 90000)
        status_values[status_columns[1]] = "正常"
        status_values[status_columns[2]] = "0"
        extra_rows.extend([status_header, status_values])
    exact_mask = [
        *base_exact_mask,
        *(
            [bool(value) for value in row]
            for row in extra_rows
        ),
    ]
    widths = [
        1.5 + ((sequence + column * 3) % 7)
        for column in range(len(header))
    ]
    return _grid_table(
        f"liability-{sequence}",
        top=top,
        header=header,
        values=values,
        widths=widths,
        extra_rows=extra_rows,
        exact_mask=exact_mask,
    )


def _foreign_table(table_id: str, *, top: float) -> SimpleNamespace:
    return _grid_table(
        table_id,
        top=top,
        header=["字段甲", "字段乙"],
        values=["值甲", "值乙"],
        widths=[3.0, 8.0],
    )


def _account_table(sequence: int, *, top: float) -> SimpleNamespace:
    return _grid_table(
        f"account-{sequence}",
        top=top,
        header=list(ACCOUNT_ROLES),
        values=[
            f"测试银行股份有限公司{sequence}支行",
            f"B10111000H0001000000000000000{sequence:02d}",
            "2020.01.01",
            "10000",
            "10000",
            "人民币元",
            "贷记卡",
            "信用/免担保",
        ],
        widths=[5.0, 8.0, 4.0, 4.0, 4.0, 3.0, 3.0, 4.0],
    )


def _composite_account_table(sequence: int, *, top: float) -> SimpleNamespace:
    """Model a real account card whose OCR packed its header lattice."""

    return _grid_table(
        f"account-composite-{sequence}",
        top=top,
        header=[
            "发卡机构账户标识开立日期账户授信额度共享授信额度币种业务种类担保方式",
            "",
        ],
        values=[
            f"测试银行股份有限公司{sequence}支行 "
            f"B10111000H0001000000000000000{sequence:02d} 2020.01.01",
            "",
        ],
        widths=[9.0, 1.0],
    )


def _monthly_tail_table(*, top: float) -> SimpleNamespace:
    return _grid_table(
        "account-19-monthly-tail",
        top=top,
        header=["", "10", "11", "12"],
        values=["2022", "N", "N", "N"],
        widths=[2.0, 2.0, 2.0, 2.0],
    )


def _append_exact_row(table: SimpleNamespace, row: list[str]) -> None:
    """Append one sealed physical row to both geometry copies."""

    metadata = table.metadata
    raw_rows = metadata["raw_rows"]
    assert len(row) == len(raw_rows[0])
    row_index = len(raw_rows)
    raw_rows.append(list(row))
    previous_boxes = metadata["cell_bboxes"][-1]
    row_height = max(box[3] for box in previous_boxes) - min(
        box[1] for box in previous_boxes
    )
    appended_boxes = [
        [box[0], box[3], box[2], box[3] + row_height]
        for box in previous_boxes
    ]
    statuses = ["exact" if value else "derived" for value in row]
    evidence_ids = [
        [f"{table.table_id}:r{row_index}:c{column}"] if value else []
        for column, value in enumerate(row)
    ]
    for owner in (metadata, metadata["geometry"]):
        owner["cell_bboxes"].append(deepcopy(appended_boxes))
        owner["cell_geometry_status"].append(list(statuses))
        owner["cell_evidence_ids"].append(deepcopy(evidence_ids))
    table.bbox[3] += row_height


def _set_exact_cell(
    table: SimpleNamespace,
    *,
    row: int,
    column: int,
    value: str,
) -> None:
    table.metadata["raw_rows"][row][column] = value
    for owner in (table.metadata, table.metadata["geometry"]):
        owner["cell_geometry_status"][row][column] = "exact"
        owner["cell_evidence_ids"][row][column] = [
            f"{table.table_id}:r{row}:c{column}:residue"
        ]


def _case(
    *,
    current_count: int = 4,
) -> tuple[list[SimpleNamespace], list[dict[str, object]]]:
    personal = _liability_table(
        80,
        top=280.0,
        reordered=False,
        merged_slots=True,
    )
    terminal = _liability_table(
        1,
        top=420.0,
        reordered=False,
        merged_slots=True,
    )
    previous = _page(
        18,
        [
            _foreign_table("preceding-account-a", top=50.0),
            _foreign_table("preceding-account-b", top=150.0),
            personal,
            terminal,
        ],
    )
    current_tables = [
        _liability_table(
            sequence,
            top=65.0 + index * 125.0,
            reordered=index % 2 == 1,
            merged_slots=index % 2 == 0,
            optional_blanks=sequence == 4,
            compact_graph=index == current_count - 1,
        )
        for index, sequence in enumerate(range(2, 2 + current_count))
    ]
    current = _page(19, list(reversed(current_tables)))
    previous_evidence = _evidence(
        18,
        18,
        _line(
            "(四)相关还款责任信息",
            [150.0, 240.0, 410.0, 260.0],
            evidence_id="liability-section",
        ),
        _line(
            "账户1",
            [35.0, 405.0, 120.0, 416.0],
            evidence_id="liability-anchor:1",
        ),
    )
    current_evidence = _evidence(
        19,
        19,
        *(
            _line(
                f"账户{sequence}",
                [35.0, 50.0 + index * 125.0, 120.0, 61.0 + index * 125.0],
                evidence_id=f"liability-anchor:{sequence}",
            )
            for index, sequence in enumerate(range(2, 2 + current_count))
        ),
    )
    return [previous, current], [previous_evidence, current_evidence]


def _lin_production_shaped_mixed_case() -> tuple[
    SimpleNamespace,
    dict[str, object],
    SimpleNamespace,
    dict[str, object],
]:
    previous = _page(
        22,
        [
            _foreign_table("account-17-monthly-tail", top=35.0),
            _composite_account_table(18, top=120.0),
            _composite_account_table(19, top=245.0),
        ],
    )
    liability = _liability_table(
        1,
        top=428.0,
        reordered=False,
        merged_slots=True,
    )
    raw_rows = liability.metadata["raw_rows"]
    for row in raw_rows:
        for column, value in enumerate(row):
            replacements = {
                "责任人类型": "贵任人类型",
                "还款责任金额": "还款贵任金额",
                "五级分类 囍": "7五级分类S",
            }
            if value in replacements:
                row[column] = replacements[value]
    raw_rows[4][0] = "#截至2024年07月01日"
    current = _page(
        23,
        [
            _monthly_tail_table(top=35.0),
            _account_table(20, top=99.0),
            _account_table(21, top=207.0),
            _account_table(22, top=315.0),
            liability,
        ],
    )
    previous_evidence = _evidence(
        22,
        22,
        _line("账户18", [35.0, 104.0, 120.0, 115.0], evidence_id="lin-account:18"),
        _line("账户19", [35.0, 229.0, 120.0, 240.0], evidence_id="lin-account:19"),
    )
    current_evidence = _evidence(
        23,
        23,
        _line("账户20", [35.0, 83.0, 120.0, 94.0], evidence_id="lin-account:20"),
        _line("账户21", [35.0, 191.0, 120.0, 202.0], evidence_id="lin-account:21"),
        _line(
            "账户22(授信协议标识:CREDIT-LINE-22)",
            [35.0, 299.0, 250.0, 310.0],
            evidence_id="lin-account:22",
        ),
        _line(
            "(四)相关还款责任信息",
            [180.0, 384.0, 370.0, 396.0],
            evidence_id="lin-liability-section",
        ),
        _line("账户1", [35.0, 412.0, 120.0, 423.0], evidence_id="lin-liability:1"),
    )
    return previous, previous_evidence, current, current_evidence


def _build(
    pages: list[SimpleNamespace],
    evidence: list[dict[str, object]],
    *,
    continuations: set[tuple[str, str]] | None = None,
):
    owner = SimpleNamespace(
        tables_continue=lambda left, right: (left, right) in (continuations or set()),
        reading_order_resolution=None,
        _personal_detail_extraction_issues=[],
    )
    projection = PBOCCanonicalTemplateAssembler(
        SimpleNamespace(pages=pages),
        topology=SimpleNamespace(geometry=lambda _logical: None),
        reading_order_by_logical={18: 1, 19: 2},
        source_evidence_loader=lambda: evidence,
        issue_owner=owner,
    ).build()
    return projection, owner


def test_account_cards_before_first_liability_heading_keep_table_local_owners() -> None:
    previous = _page(18, [_account_table(18, top=100.0), _account_table(19, top=210.0)])
    current = _page(
        19,
        [
            _monthly_tail_table(top=35.0),
            _account_table(20, top=99.0),
            _account_table(21, top=207.0),
            _account_table(22, top=315.0),
            _liability_table(
                1,
                top=428.0,
                reordered=False,
                merged_slots=True,
            ),
        ],
    )
    previous_evidence = _evidence(
        18,
        18,
        _line(
            "三信贷交易信息明细",
            [180.0, 30.0, 370.0, 48.0],
            evidence_id="account-section",
        ),
        _line("账户18", [35.0, 84.0, 120.0, 95.0], evidence_id="account-anchor:18"),
        _line("账户19", [35.0, 194.0, 120.0, 205.0], evidence_id="account-anchor:19"),
    )
    current_evidence = _evidence(
        19,
        19,
        _line("账户20", [35.0, 83.0, 120.0, 94.0], evidence_id="account-anchor:20"),
        _line("账户21", [35.0, 191.0, 120.0, 202.0], evidence_id="account-anchor:21"),
        _line("账户22", [35.0, 299.0, 120.0, 310.0], evidence_id="account-anchor:22"),
        _line(
            "(四)相关还款责任信息",
            [180.0, 384.0, 370.0, 396.0],
            evidence_id="liability-section:current",
        ),
        _line("账户1", [35.0, 412.0, 120.0, 423.0], evidence_id="liability-anchor:1"),
    )

    projection, _owner = _build(
        [previous, current],
        [previous_evidence, current_evidence],
        continuations={("account-19", "account-19-monthly-tail")},
    )

    registration = {
        item["logical_page"]: item for item in projection.registrations
    }[19]
    assert registration["template_id"] == "mixed_pboc_sections"
    owners = registration["section_table_owners"]
    assert {
        table_id: owner["template_id"] for table_id, owner in owners.items()
    } == {
        "account-19-monthly-tail": "credit_account_detail",
        "account-20": "credit_account_detail",
        "account-21": "credit_account_detail",
        "account-22": "credit_account_detail",
        "liability-1": "repayment_responsibility",
    }
    assert owners["account-19-monthly-tail"]["binding"] == (
        "authoritative_prior_account_table_continuation"
    )


def test_mixed_account_liability_accepts_registered_flow_continuation_predecessor() -> None:
    previous = _page(18, [_account_table(18, top=100.0), _account_table(19, top=210.0)])
    current = _page(
        19,
        [
            _monthly_tail_table(top=35.0),
            _account_table(20, top=99.0),
            _account_table(21, top=207.0),
            _account_table(22, top=315.0),
            _liability_table(1, top=428.0, reordered=False, merged_slots=True),
        ],
    )
    previous_evidence = _evidence(
        18,
        18,
        _line("账户18", [35.0, 84.0, 120.0, 95.0], evidence_id="account-anchor:18"),
        _line("账户19", [35.0, 194.0, 120.0, 205.0], evidence_id="account-anchor:19"),
    )
    current_evidence = _evidence(
        19,
        19,
        _line("账户20", [35.0, 83.0, 120.0, 94.0], evidence_id="account-anchor:20"),
        _line("账户21", [35.0, 191.0, 120.0, 202.0], evidence_id="account-anchor:21"),
        _line("账户22", [35.0, 299.0, 120.0, 310.0], evidence_id="account-anchor:22"),
        _line(
            "(四)相关还款责任信息",
            [180.0, 384.0, 370.0, 396.0],
            evidence_id="liability-section:current",
        ),
        _line("账户1", [35.0, 412.0, 120.0, 423.0], evidence_id="liability-anchor:1"),
    )

    owners = _mixed_account_liability_table_owners(
        previous,
        previous_evidence,
        {
            "status": "registered",
            "template_id": "credit_account_detail",
            "basis": "canonical_flow_continuation",
        },
        current,
        current_evidence,
        tables_continue=lambda left, right: (
            left,
            right,
        ) == ("account-19", "account-19-monthly-tail"),
    )

    assert set(owners) == {
        "account-19-monthly-tail",
        "account-20",
        "account-21",
        "account-22",
        "liability-1",
    }


def test_lin_shaped_packed_predecessor_and_finite_liability_ocr_variants_are_owned() -> None:
    previous, previous_evidence, current, current_evidence = (
        _lin_production_shaped_mixed_case()
    )

    owners = _mixed_account_liability_table_owners(
        previous,
        previous_evidence,
        {
            "status": "registered",
            "template_id": "credit_account_detail",
            "basis": "source_page_evidence",
        },
        current,
        current_evidence,
        tables_continue=lambda left, right: (left, right)
        == ("account-composite-19", "account-19-monthly-tail"),
    )

    assert {
        table_id: owner["template_id"] for table_id, owner in owners.items()
    } == {
        "account-19-monthly-tail": "credit_account_detail",
        "account-20": "credit_account_detail",
        "account-21": "credit_account_detail",
        "account-22": "credit_account_detail",
        "liability-1": "repayment_responsibility",
    }
    assert owners["account-22"]["printed_sequence"] == 22


@pytest.mark.parametrize(
    "defect",
    (
        "unknown_liability_header",
        "malformed_full_account_heading",
        "missing_full_account_heading_evidence",
        "footer_gap",
        "continuation_denied",
        "intervening_section",
    ),
)
def test_lin_shaped_mixed_transition_remains_fail_closed(defect: str) -> None:
    previous, previous_evidence, current, current_evidence = (
        _lin_production_shaped_mixed_case()
    )
    continuation_allowed = True
    if defect == "unknown_liability_header":
        current.tables[-1].metadata["raw_rows"][0][4] = "未知责任类型"
    elif defect == "malformed_full_account_heading":
        current_evidence["lines"][2]["text"] = "账户22(其他标识:CREDIT-LINE-22)"
    elif defect == "missing_full_account_heading_evidence":
        current_evidence["lines"][2]["evidence_ids"] = []
    elif defect == "footer_gap":
        current_evidence["lines"][-1]["text"] = "第24页，共24页"
    elif defect == "continuation_denied":
        continuation_allowed = False
    else:
        previous_evidence["lines"].insert(
            -1,
            _line(
                "(四)相关还款责任信息",
                [180.0, 620.0, 370.0, 635.0],
                evidence_id="lin-intervening-section",
            ),
        )

    owners = _mixed_account_liability_table_owners(
        previous,
        previous_evidence,
        {
            "status": "registered",
            "template_id": "credit_account_detail",
            "basis": "source_page_evidence",
        },
        current,
        current_evidence,
        tables_continue=lambda left, right: continuation_allowed
        and (left, right) == ("account-composite-19", "account-19-monthly-tail"),
    )

    assert owners == {}


@pytest.mark.parametrize("current_count", [1, 3, 5])
def test_liability_continuation_accepts_variable_card_population_and_geometry(
    current_count: int,
) -> None:
    pages, evidence = _case(current_count=current_count)
    prior_registration = {
        "status": "registered",
        "template_id": "repayment_responsibility",
        "basis": "source_page_evidence",
    }

    assert _sealed_liability_page_continuation_proved(
        pages[0],
        evidence[0],
        prior_registration,
        pages[1],
        evidence[1],
    )


def test_yang_shape_continuation_projects_all_four_cards_for_native_parser() -> None:
    pages, evidence = _case(current_count=4)

    projection, owner = _build(pages, evidence)

    registration = {
        item["logical_page"]: item
        for item in projection.registrations
    }[19]
    assert registration["template_id"] == "repayment_responsibility"
    assert registration["basis"] == "source_page_evidence"
    assert "exact_cross_page_liability_continuation" in registration["signals"]
    current = next(page for page in projection.pages if page.page_number == 19)
    assert current.canonical_template_id == "repayment_responsibility"
    assert len(current.tables) == 4
    assert {
        table.metadata["canonical_template_id"]
        for table in current.tables
    } == {"repayment_responsibility"}

    def native_context() -> SimpleNamespace:
        return SimpleNamespace(
            pages=list(projection.pages),
            reading_order_by_logical={18: 1, 19: 2},
            tables_continue=lambda _left, _right: False,
            corrected_evidence_pages=lambda: list(projection.evidence_pages),
            _personal_detail_extraction_issues=[],
        )

    context = native_context()
    records = PBOCPersonalDetailNativeParser(context).records(
        "repayment_liability_records"
    )

    assert len(records) == 6
    assert {
        record.fields.get("__printed_sequence")
        for record in records
        if record.fields.get("__printed_sequence")
    } == {"1", "2", "3", "4", "5"}
    optional = next(
        record
        for record in records
        if record.fields.get("__printed_sequence") == "4"
    )
    assert "还款责任金额" not in optional.fields
    assert "保证合同编号" not in optional.fields

    extracted = native_extraction._extract_liabilities(native_context())
    assert len(extracted) == 6
    assert {
        item.get("_printed_sequence")
        for item in extracted
        if item.get("_printed_sequence") is not None
    } == {1, 2, 3, 4, 5}
    extracted_optional = next(
        item for item in extracted if item.get("_printed_sequence") == 4
    )
    assert extracted_optional["institution"] == "测试银行股份有限公司4支行"
    assert extracted_optional["open_date"] == "2023-04-01"
    assert extracted_optional["responsibility_type"] == "共同借款人"
    assert extracted_optional.get("responsibility_amount") is None
    assert extracted_optional.get("contract_number") is None
    assert owner._personal_detail_extraction_issues == []


@pytest.mark.parametrize(
    "defect",
    [
        "ordinal_gap",
        "ordinal_duplicate",
        "missing_id",
        "replayed_id",
        "malformed_geometry",
        "table_anchor_mismatch",
        "competing_boundary",
        "footer_gap",
        "foreign_table",
        "cross_section_rows",
        "extra_row",
        "inactive_column_residue",
        "duplicate_graph",
    ],
)
def test_liability_continuation_rejects_incomplete_or_ambiguous_proof(
    defect: str,
) -> None:
    pages, evidence = _case(current_count=4)
    current = pages[1]
    current_evidence = evidence[1]
    anchors = current_evidence["lines"][:-1]
    if defect == "ordinal_gap":
        for offset, line in enumerate(anchors[1:], start=4):
            line["text"] = f"账户{offset}"
    elif defect == "ordinal_duplicate":
        anchors[1]["text"] = "账户2"
    elif defect == "missing_id":
        anchors[0]["evidence_ids"] = []
    elif defect == "replayed_id":
        first = current.tables[-1].metadata["geometry"]["cell_evidence_ids"]
        second = current.tables[-2].metadata["geometry"]["cell_evidence_ids"]
        second[0][0] = list(first[0][0])
        current.tables[-2].metadata["cell_evidence_ids"][0][0] = list(first[0][0])
    elif defect == "malformed_geometry":
        current.tables[-1].metadata["geometry"]["cell_bboxes"][0][0][2] = float("nan")
        current.tables[-1].metadata["cell_bboxes"][0][0][2] = float("nan")
    elif defect == "table_anchor_mismatch":
        anchors[0]["bbox"] = [535.0, 50.0, 550.0, 61.0]
    elif defect == "competing_boundary":
        current_evidence["lines"].insert(
            -1,
            _line(
                "授信协议信息",
                [170.0, 620.0, 390.0, 640.0],
                evidence_id="foreign-section",
            ),
        )
    elif defect == "footer_gap":
        current_evidence["lines"][-1]["text"] = "第20页，共24页"
    elif defect == "foreign_table":
        current.tables.append(_foreign_table("foreign-current", top=590.0))
    elif defect == "cross_section_rows":
        table = current.tables[-1]
        header = [""] * len(table.metadata["raw_rows"][0])
        header[0] = "授信协议信息"
        value = [""] * len(header)
        value[0] = "FOREIGN-VALUE"
        _append_exact_row(table, header)
        _append_exact_row(table, value)
    elif defect == "extra_row":
        table = current.tables[-1]
        row = [""] * len(table.metadata["raw_rows"][0])
        row[0] = "未注册业务残留"
        _append_exact_row(table, row)
    elif defect == "inactive_column_residue":
        table = current.tables[-1]
        _set_exact_cell(table, row=1, column=3, value="FOREIGN-VALUE")
    else:
        table = current.tables[-1]
        replay_rows = deepcopy(table.metadata["raw_rows"])
        for replay_row in replay_rows:
            _append_exact_row(table, replay_row)

    assert not _sealed_liability_page_continuation_proved(
        pages[0],
        evidence[0],
        {
            "status": "registered",
            "template_id": "repayment_responsibility",
            "basis": "source_page_evidence",
        },
        pages[1],
        evidence[1],
    )

    projection, _owner = _build(
        pages,
        evidence,
        continuations={("liability-1", "liability-2")},
    )

    registration = {
        item["logical_page"]: item
        for item in projection.registrations
    }[19]
    assert registration["template_id"] != "repayment_responsibility"
    if defect not in {"competing_boundary", "cross_section_rows"}:
        assert registration["status"] == "unresolved"
        assert registration["template_id"] == "unresolved"
        assert 19 in projection.unresolved_pages
