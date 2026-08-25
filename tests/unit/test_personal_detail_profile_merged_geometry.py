from __future__ import annotations

from types import SimpleNamespace

from docmirror.plugins.credit_report.personal_detail_scanned.native_extraction import (
    _extract_profile_detail_records,
)
from docmirror.plugins.credit_report.personal_detail_scanned.profile_extraction import (
    _bounded_profile_address_sequence_residue,
)

_PROFILE_TEMPLATE = "report_header_and_identity"


def _table(
    table_id: str,
    rows: list[list[str]],
    atoms: list[dict],
    *,
    scale: float = 1.0,
    owned: bool = True,
) -> SimpleNamespace:
    row_bands = [{"index": row, "y0": row * 20.0 * scale, "y1": (row + 1) * 20.0 * scale} for row in range(len(rows))]
    bands = [
        (0.0 * scale, 80.0 * scale),
        (80.0 * scale, 180.0 * scale),
        (180.0 * scale, 280.0 * scale),
        (280.0 * scale, 400.0 * scale),
    ]
    col_bands = [{"index": index, "x0": bounds[0], "x1": bounds[1]} for index, bounds in enumerate(bands)]
    boxes: list[list[list[float] | None]] = []
    statuses: list[list[str]] = []
    evidence: list[list[list[str]]] = []
    spans: list[dict] = []
    for row_index, row in enumerate(rows):
        row_boxes: list[list[float] | None] = []
        row_statuses: list[str] = []
        row_evidence: list[list[str]] = []
        for column in range(4):
            ids = [
                atom["id"]
                for atom in atoms
                if row_index * 20.0 * scale <= (atom["bbox"][1] + atom["bbox"][3]) / 2 <= (row_index + 1) * 20.0 * scale
                and bands[column][0] <= (atom["bbox"][0] + atom["bbox"][2]) / 2 <= bands[column][1]
            ]
            row_boxes.append(
                [
                    bands[column][0],
                    row_index * 20.0 * scale,
                    bands[column][1],
                    (row_index + 1) * 20.0 * scale,
                ]
            )
            row_statuses.append("exact")
            row_evidence.append(ids)
        boxes.append(row_boxes)
        statuses.append(row_statuses)
        evidence.append(row_evidence)

    def merge(row: int, column: int) -> None:
        evidence[row][column] += evidence[row][column + 1]
        evidence[row][column + 1] = []
        boxes[row][column] = [
            bands[column][0],
            row * 20.0 * scale,
            bands[column + 1][1],
            (row + 1) * 20.0 * scale,
        ]
        boxes[row][column + 1] = None
        statuses[row][column + 1] = "derived"
        spans.append({"row": row, "col": column, "row_span": 1, "col_span": 2})

    if table_id == "mobile":
        merge(0, 0)
        merge(0, 2)
        for row in range(1, len(rows)):
            merge(row, 0)
    else:
        merge(1, 0)
    geometry = {
        "row_bands": row_bands,
        "col_bands": col_bands,
        "cell_bboxes": boxes,
        "cell_geometry_status": statuses,
        "cell_evidence_ids": evidence,
        "cell_token_ids": evidence,
        "cell_spans": spans,
    }
    metadata = {"raw_rows": rows, "geometry": geometry}
    if owned:
        metadata.update(
            canonical_template_id=_PROFILE_TEMPLATE,
            source_logical_page=7,
            source_page=3,
        )
    return SimpleNamespace(
        table_id=table_id,
        metadata=metadata,
        headers=[],
        rows=[],
    )


def _atom(atom_id: str, text: str, bbox: list[float]) -> dict:
    return {"id": atom_id, "text": text, "bbox": bbox}


def _scaled_atom(atom_id: str, text: str, bbox: list[float], scale: float) -> dict:
    return _atom(atom_id, text, [coordinate * scale for coordinate in bbox])


def _page(*tables: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(
        page_number=11,
        source_page_number=3,
        tables=list(tables),
        canonical_template_id=_PROFILE_TEMPLATE,
        canonical_fragment_logical_pages=(7, 11),
        coordinate_transform={"kind": "plugin_canonical_template", "source_page_numbers": [3, 5]},
    )


def _exact_table(
    table_id: str,
    rows: list[list[str]],
    atoms: list[dict],
    *,
    scale: float,
    owned: bool = True,
) -> SimpleNamespace:
    width = 100.0 * scale
    height = 20.0 * scale
    column_count = len(rows[0])
    row_bands = [{"index": row, "y0": row * height, "y1": (row + 1) * height} for row in range(len(rows))]
    col_bands = [{"index": column, "x0": column * width, "x1": (column + 1) * width} for column in range(column_count)]
    boxes: list[list[list[float]]] = []
    evidence: list[list[list[str]]] = []
    for row_index in range(len(rows)):
        row_boxes: list[list[float]] = []
        row_evidence: list[list[str]] = []
        for column in range(column_count):
            row_boxes.append([column * width, row_index * height, (column + 1) * width, (row_index + 1) * height])
            row_evidence.append(
                [
                    atom["id"]
                    for atom in atoms
                    if row_index * height <= (atom["bbox"][1] + atom["bbox"][3]) / 2 <= (row_index + 1) * height
                    and column * width <= (atom["bbox"][0] + atom["bbox"][2]) / 2 <= (column + 1) * width
                ]
            )
        boxes.append(row_boxes)
        evidence.append(row_evidence)
    geometry = {
        "row_bands": row_bands,
        "col_bands": col_bands,
        "cell_bboxes": boxes,
        "cell_geometry_status": [["exact"] * column_count for _ in rows],
        "cell_evidence_ids": evidence,
        "cell_token_ids": evidence,
        "cell_spans": [],
    }
    metadata: dict[str, object] = {"raw_rows": rows, "geometry": geometry}
    if owned:
        metadata.update(
            canonical_template_id=_PROFILE_TEMPLATE,
            source_logical_page=7,
            source_page=3,
        )
    return SimpleNamespace(table_id=table_id, metadata=metadata, headers=[], rows=[])


def _spouse_provider_fixture(provider: str, *, owned: bool = True) -> tuple[SimpleNamespace, list[dict]]:
    scale = 1.4
    rows = [
        ["姓名", "证件类型", "证件号码", "工作单位", "联系电话"],
        ["林航", "身份证", "350102198806013017", "示例科技有限公司", "13763822211"],
        ["数据发生机构名称", "编号", "", "", ""],
        [provider, "1", "", "", ""],
    ]
    texts = [*rows[0], *rows[1], *rows[2][:2], *rows[3][:2]]
    positions = [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (1, 0),
        (1, 1),
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 0),
        (2, 1),
        (3, 0),
        (3, 1),
    ]
    atoms = [
        _scaled_atom(
            f"sp-{index}",
            text,
            [column * 100 + 10, row * 20 + 4, column * 100 + 90, row * 20 + 16],
            scale,
        )
        for index, (text, (row, column)) in enumerate(zip(texts, positions, strict=True))
    ]
    return _exact_table("spouse-provider", rows, atoms, scale=scale, owned=owned), atoms


def test_exact_merged_mobile_and_spouse_tables_recover_token_owned_fields() -> None:
    scale = 1.65
    atoms = [
        _scaled_atom("mh1", "编号", [10, 4, 35, 16], scale),
        _scaled_atom("mh2", "手机号码", [100, 4, 145, 16], scale),
        _scaled_atom("mh3", "信息更新日期", [195, 4, 260, 16], scale),
        _scaled_atom("mh4", "数据发生机构名称", [300, 4, 390, 16], scale),
        _scaled_atom("m11", "1", [10, 24, 20, 36], scale),
        _scaled_atom("m12", "18046145176", [100, 24, 160, 36], scale),
        _scaled_atom("m13", "2024.11.15", [200, 24, 260, 36], scale),
        _scaled_atom("m14", "样例银行股份有限公司", [290, 24, 390, 36], scale),
        _scaled_atom("m21", "2", [10, 44, 20, 56], scale),
        _scaled_atom("m22", "8618046145176", [95, 44, 165, 56], scale),
        _scaled_atom("m23", "2015.12.29", [200, 44, 260, 56], scale),
        _scaled_atom("m24", "中国工商银行股份有限公司福州分行", [285, 44, 395, 56], scale),
    ]
    mobile = _table(
        "mobile",
        [
            ["编号 手机号码", "", "信息更新日期 数据发生机构名称", ""],
            ["1 18046145176", "", "2024.11.15", "样例银行股份有限公司"],
            ["2 8618046145176", "", "2015.12.29", "中国工商银行股份有限公司福州分行"],
        ],
        atoms,
        scale=scale,
    )
    spouse_atoms = [
        _scaled_atom("sh1", "姓名", [20, 4, 50, 16], scale),
        _scaled_atom("sh2", "证件类型", [90, 4, 120, 16], scale),
        _scaled_atom("sh3", "证件号码", [145, 4, 175, 16], scale),
        _scaled_atom("sh4", "工作单位", [200, 4, 250, 16], scale),
        _scaled_atom("sh5", "联系电话", [310, 4, 360, 16], scale),
        _scaled_atom("sv1", "陶亚利", [20, 24, 55, 36], scale),
        _scaled_atom("sv2", "身份证", [90, 24, 120, 36], scale),
        _scaled_atom("sv3", "340406198604163842", [130, 24, 175, 36], scale),
        _scaled_atom("sv4", "漳州市龙文区仟麦商务服务有限公司", [190, 24, 270, 36], scale),
        _scaled_atom("sv5", "18259616116", [310, 24, 380, 36], scale),
    ]
    spouse = _table(
        "spouse",
        [
            ["姓名", "证件号码 证件类型", "工作单位", "联系电话"],
            ["陶亚利 身份证 340406198604163842", "", "漳州市龙文区仟麦商务服务有限公司", "18259616116"],
        ],
        spouse_atoms,
        scale=scale,
    )
    page = _page(spouse, mobile)
    result = SimpleNamespace(
        pages=[page],
        tables_continue=lambda _left, _right: False,
        evidence_plane=SimpleNamespace(
            evidence=SimpleNamespace(text_atoms=[*reversed(spouse_atoms), *reversed(atoms)])
        ),
    )

    details = _extract_profile_detail_records(result)

    assert [record["sequence"] for record in details["mobile_phone_records"]] == [1, 2]
    assert [record["mobile_phone"] for record in details["mobile_phone_records"]] == [
        "18046145176",
        "18046145176",
    ]
    assert details["mobile_phone_records"][0]["source_refs_by_field"]["mobile_phone"][0]["geometry_scope"] == "token"
    assert (
        details["mobile_phone_records"][0]["source_refs_by_field"]["data_provider"][0]["geometry_scope"] == "token_span"
    )
    assert details["mobile_phone_records"][0]["source_refs_by_field"]["mobile_phone"][0]["logical_page"] == 7
    assert details["mobile_phone_records"][0]["source_refs_by_field"]["mobile_phone"][0]["source_page"] == 3
    assert details["spouse_records"][0]["document_number"] == "340406198604163842"
    assert details["spouse_records"][0]["employer"] == "漳州市龙文区仟麦商务服务有限公司"


def test_merged_spouse_short_phone_is_withheld_with_exact_token_evidence() -> None:
    scale = 1.65
    atoms = [
        _scaled_atom("sh1", "姓名", [20, 4, 50, 16], scale),
        _scaled_atom("sh2", "证件类型", [90, 4, 120, 16], scale),
        _scaled_atom("sh3", "证件号码", [145, 4, 175, 16], scale),
        _scaled_atom("sh4", "工作单位", [200, 4, 250, 16], scale),
        _scaled_atom("sh5", "联系电话", [310, 4, 360, 16], scale),
        _scaled_atom("sv1", "陶亚利", [20, 24, 55, 36], scale),
        _scaled_atom("sv2", "身份证", [90, 24, 120, 36], scale),
        _scaled_atom("sv3", "340406198604163842", [130, 24, 175, 36], scale),
        _scaled_atom("sv4", "漳州市龙文区仟麦商务服务有限公司", [190, 24, 270, 36], scale),
        _scaled_atom("sv5", "12345", [310, 24, 380, 36], scale),
    ]
    table = _table(
        "spouse-short-phone",
        [
            ["姓名", "证件号码 证件类型", "工作单位", "联系电话"],
            ["陶亚利 身份证 340406198604163842", "", "漳州市龙文区仟麦商务服务有限公司", "12345"],
        ],
        atoms,
        scale=scale,
    )
    result = SimpleNamespace(
        pages=[_page(table)],
        tables_continue=lambda _left, _right: False,
        evidence_plane=SimpleNamespace(evidence=SimpleNamespace(text_atoms=atoms)),
    )

    spouse = _extract_profile_detail_records(result)["spouse_records"][0]

    assert "phone" not in spouse
    assert spouse["canonical_raw"]["phone"] == ["12345"]
    assert spouse["source_refs_by_field"]["phone"][0]["geometry_scope"] == "token"
    assert any(
        issue.get("issue_code") == "candidate_b_exact_slot_value_invalid"
        and issue.get("target_record_id") == spouse["spouse_record_id"]
        and issue.get("field_name") == "phone"
        and issue.get("observed_value") == ["12345"]
        and issue.get("source_refs", [{}])[0].get("evidence_ids") == ["sv5"]
        for issue in result._personal_detail_extraction_issues
    )


def test_merged_mobile_without_exact_token_evidence_fails_closed() -> None:
    atoms = [
        _atom("h1", "编号", [10, 4, 35, 16]),
        _atom("h2", "手机号码", [100, 4, 145, 16]),
        _atom("h3", "信息更新日期", [195, 4, 260, 16]),
        _atom("h4", "数据发生机构名称", [300, 4, 390, 16]),
    ]
    table = _table(
        "mobile",
        [["编号 手机号码", "", "信息更新日期 数据发生机构名称", ""]],
        atoms,
    )
    result = SimpleNamespace(
        pages=[_page(table)],
        tables_continue=lambda _left, _right: False,
        evidence_plane=SimpleNamespace(evidence=SimpleNamespace(text_atoms=[])),
    )

    assert _extract_profile_detail_records(result)["mobile_phone_records"] == []
    assert any(
        issue["issue_code"] == "candidate_b_canonical_header_graph_unresolved"
        for issue in result._personal_detail_extraction_issues
    )


def test_arbitrary_unregistered_collapsed_mobile_fails_closed() -> None:
    atoms = [
        _atom("uh1", "编号", [10, 4, 35, 16]),
        _atom("uh2", "手机号码", [100, 4, 145, 16]),
        _atom("uh3", "信息更新日期", [195, 4, 260, 16]),
        _atom("uh4", "数据发生机构名称", [300, 4, 390, 16]),
        _atom("uv1", "1", [10, 24, 20, 36]),
        _atom("uv2", "15260467509", [100, 24, 160, 36]),
        _atom("uv3", "2025.08.03", [200, 24, 260, 36]),
        _atom("uv4", "深圳市乐信融资担保有限公司", [290, 24, 390, 36]),
    ]
    table = _table(
        "arbitrary-unregistered-mobile",
        [
            ["编号 手机号码", "", "信息更新日期 数据发生机构名称", ""],
            ["1 15260467509", "", "2025.08.03", "深圳市乐信融资担保有限公司"],
        ],
        atoms,
        owned=False,
    )
    result = SimpleNamespace(
        pages=[SimpleNamespace(page_number=1, source_page_number=1, tables=[table])],
        tables_continue=lambda _left, _right: False,
        evidence_plane=SimpleNamespace(evidence=SimpleNamespace(text_atoms=list(reversed(atoms)))),
    )

    details = _extract_profile_detail_records(result)

    assert details["mobile_phone_records"] == []


def test_merged_mobile_rejects_unassigned_or_reused_token_evidence() -> None:
    atoms = [
        _atom("dh1", "编号", [10, 4, 35, 16]),
        _atom("dh2", "手机号码", [100, 4, 145, 16]),
        _atom("dh3", "信息更新日期", [195, 4, 260, 16]),
        _atom("dh4", "数据发生机构名称", [300, 4, 390, 16]),
        _atom("dv1", "1", [10, 24, 20, 36]),
        _atom("dv2", "15260467509", [100, 24, 160, 36]),
        _atom("debris", "任意备注", [120, 24, 165, 36]),
        _atom("dv3", "2025.08.03", [200, 24, 260, 36]),
        _atom("dv4", "深圳市乐信融资担保有限公司", [290, 24, 390, 36]),
    ]
    table = _table(
        "mobile-distinct-evidence",
        [
            ["编号 手机号码", "", "信息更新日期 数据发生机构名称", ""],
            ["1 15260467509 任意备注", "", "2025.08.03", "深圳市乐信融资担保有限公司"],
        ],
        atoms,
    )
    result = SimpleNamespace(
        pages=[_page(table)],
        tables_continue=lambda _left, _right: False,
        evidence_plane=SimpleNamespace(evidence=SimpleNamespace(text_atoms=atoms)),
    )
    assert _extract_profile_detail_records(result)["mobile_phone_records"] == []

    table.metadata["geometry"]["cell_evidence_ids"][1][0] = ["dh1", "dv2"]
    table.metadata["geometry"]["cell_token_ids"][1][0] = ["dh1", "dv2"]
    reused = SimpleNamespace(
        pages=[_page(table)],
        tables_continue=lambda _left, _right: False,
        evidence_plane=SimpleNamespace(evidence=SimpleNamespace(text_atoms=atoms)),
    )
    assert _extract_profile_detail_records(reused)["mobile_phone_records"] == []


def test_spouse_provider_uses_exact_reordered_scaled_label_value_band() -> None:
    table, atoms = _spouse_provider_fixture("示例消费金融有限公司")
    result = SimpleNamespace(
        pages=[_page(table)],
        tables_continue=lambda _left, _right: False,
        evidence_plane=SimpleNamespace(evidence=SimpleNamespace(text_atoms=list(reversed(atoms)))),
    )

    spouse = _extract_profile_detail_records(result)["spouse_records"][0]

    assert spouse["data_provider"] == "示例消费金融有限公司"
    provider_ref = spouse["source_refs_by_field"]["data_provider"][0]
    assert provider_ref["column"] == 0
    assert provider_ref["logical_page"] == 7
    assert provider_ref["source_page"] == 3
    assert provider_ref["binding_quality"] == "exact_label_value_same_source_owned_band"
    assert provider_ref["label_evidence_ids"] == ["sp-10"]
    assert provider_ref["evidence_ids"] == ["sp-12"]


def test_spouse_provider_rejects_arbitrary_note_after_exact_label() -> None:
    valid_table, atoms = _spouse_provider_fixture("示例消费金融有限公司")
    invalid_table, _ = _spouse_provider_fixture("任意备注")
    invalid_table.metadata["geometry"] = valid_table.metadata["geometry"]
    atoms[12] = {**atoms[12], "text": "任意备注"}
    table = invalid_table
    result = SimpleNamespace(
        pages=[_page(table)],
        tables_continue=lambda _left, _right: False,
        evidence_plane=SimpleNamespace(evidence=SimpleNamespace(text_atoms=atoms)),
    )

    spouse = _extract_profile_detail_records(result)["spouse_records"][0]

    assert spouse["name"] == "林航"
    assert "data_provider" not in spouse


def test_spouse_provider_requires_profile_owner_and_distinct_label_value_evidence() -> None:
    unowned, unowned_atoms = _spouse_provider_fixture("示例消费金融有限公司", owned=False)
    unowned_result = SimpleNamespace(
        pages=[SimpleNamespace(page_number=1, source_page_number=1, tables=[unowned])],
        tables_continue=lambda _left, _right: False,
        evidence_plane=SimpleNamespace(evidence=SimpleNamespace(text_atoms=unowned_atoms)),
    )
    assert _extract_profile_detail_records(unowned_result) == {
        "mobile_phone_records": [],
        "spouse_records": [],
    }

    reused, reused_atoms = _spouse_provider_fixture("示例消费金融有限公司")
    geometry = reused.metadata["geometry"]
    geometry["cell_evidence_ids"][3][0] = ["sp-10"]
    geometry["cell_token_ids"][3][0] = ["sp-10"]
    reused_result = SimpleNamespace(
        pages=[_page(reused)],
        tables_continue=lambda _left, _right: False,
        evidence_plane=SimpleNamespace(evidence=SimpleNamespace(text_atoms=reused_atoms)),
    )
    assert "data_provider" not in _extract_profile_detail_records(reused_result)["spouse_records"][0]


def test_profile_address_removes_one_exact_right_edge_sequence_residue() -> None:
    address = "漳州市龙文区国贸润园31栋703室"
    cell = SimpleNamespace(
        text=f"2 {address}",
        geometry_status="exact",
        evidence_ids=["address", "ordinal"],
        token_ids=["address", "ordinal"],
        bbox=[47.0, 409.5, 221.5, 422.5],
        row_span=1,
        col_span=2,
    )
    table = SimpleNamespace(metadata={}, source_cell_objects=[[cell]])
    result = SimpleNamespace(
        evidence_plane=SimpleNamespace(
            evidence=SimpleNamespace(
                text_atoms=[
                    _atom("address", address, [89.5, 413.5, 181.5, 422.0]),
                    _atom("ordinal", "2", [206.5, 410.5, 221.5, 423.0]),
                ]
            )
        )
    )

    assert (
        _bounded_profile_address_sequence_residue(
            result,
            table,
            0,
            0,
            cell.text,
            logical_page=1,
        )
        == address
    )
