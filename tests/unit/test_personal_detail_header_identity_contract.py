from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from docmirror.plugins.credit_report.personal_detail_scanned import native_extraction
from docmirror.plugins.credit_report.personal_detail_scanned.candidate_b import (
    CandidateBPipeline,
    _reconcile_candidate_b_header_lifecycle,
)
from docmirror.plugins.credit_report.personal_detail_scanned.native_extraction import (
    _extract_header_datasets,
)
from docmirror.plugins.credit_report.personal_detail_scanned.native_parser import (
    NativeLabeledRecord,
    PBOCPersonalDetailNativeParser,
)


def _exact_table(table_id: str, rows: list[list[str]]) -> SimpleNamespace:
    width = max(len(row) for row in rows)
    normalized = [row + [""] * (width - len(row)) for row in rows]
    bboxes = [
        [
            [10.0 + column * 100.0, 20.0 + row * 20.0, 110.0 + column * 100.0, 40.0 + row * 20.0]
            for column in range(width)
        ]
        for row in range(len(normalized))
    ]
    evidence = [
        [
            [f"source:{table_id}:{row}:{column}"] if value else []
            for column, value in enumerate(values)
        ]
        for row, values in enumerate(normalized)
    ]
    statuses = [
        ["exact" if value else "derived" for value in values]
        for values in normalized
    ]
    return SimpleNamespace(
        table_id=table_id,
        metadata={
            "raw_rows": normalized,
            "canonical_template_id": "report_header_and_identity",
            "source_cell_bboxes": bboxes,
            "cell_evidence_ids": evidence,
            "cell_geometry_status": statuses,
            "geometry": {
                "cell_bboxes": bboxes,
                "cell_evidence_ids": evidence,
                "cell_geometry_status": statuses,
            },
        },
        headers=[],
        rows=[],
        bbox=[10.0, 20.0, 10.0 + width * 100.0, 20.0 + len(normalized) * 20.0],
        confidence=1.0,
    )


def _primary_table() -> SimpleNamespace:
    return _exact_table(
        "primary",
        [
            ["被查询者姓名", "被查询者证件类型", "被查询者证件号码", "查询机构", "查询原因"],
            ["信小达", "身份证", "350121199101285219", "中国人民银行营业管理部", "本人查询(临柜)"],
        ],
    )


def _context(*tables: SimpleNamespace, owned: bool = True) -> SimpleNamespace:
    page = SimpleNamespace(
        page_number=3,
        source_page_number=2,
        canonical_template_id="report_header_and_identity" if owned else None,
        tables=list(tables),
    )
    return SimpleNamespace(
        pages=[page],
        _personal_detail_extraction_issues=[],
    )


@pytest.fixture(autouse=True)
def _disable_tolerant_header_parser(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(PBOCPersonalDetailNativeParser, "records", lambda _self, _name: [])


def _corrected_name_ref(
    value: str,
    *,
    logical_page: int,
    source_page: int,
    evidence_id: str = "corrected:header:subject-name",
) -> dict[str, object]:
    return {
        "source": "personal_detail_corrected_page_cell",
        "logical_page": logical_page,
        "source_page": source_page,
        "bbox": [20.0, 60.0, 120.0, 78.0],
        "geometry_scope": "cell",
        "field_name": "被查询者姓名",
        "binding": "canonical_label_slot",
        "evidence_ids": [evidence_id],
    }


def _header_candidate(
    value: str,
    *,
    refs: tuple[dict[str, object], ...] = (),
    binding_quality: str = "canonical_cell_slot",
) -> NativeLabeledRecord:
    return NativeLabeledRecord(
        dataset_name="report_header",
        fields={"被查询者姓名": value},
        source_refs=(),
        confidence=1.0,
        source_refs_by_field={"被查询者姓名": refs} if refs else {},
        binding_quality_by_field={"被查询者姓名": binding_quality},
        observed_labels=frozenset({"被查询者姓名"}),
        unresolved_labels=frozenset(),
    )


def _corrected_name_context(
    value: str,
    *,
    logical_page: int,
    source_page: int,
    owner: str = "report_header_and_identity",
) -> SimpleNamespace:
    ref = _corrected_name_ref(
        value,
        logical_page=logical_page,
        source_page=source_page,
    )
    page = SimpleNamespace(
        page_number=logical_page,
        source_page_number=source_page,
        canonical_template_id=owner,
        tables=[],
    )
    evidence_page = {
        "page": logical_page,
        "source_page": source_page,
        "canonical_template_id": owner,
        "lines": [
            {
                "text": value,
                "bbox": list(ref["bbox"]),
                "source_bbox": list(ref["bbox"]),
                "evidence_ids": list(ref["evidence_ids"]),
            }
        ],
    }
    return SimpleNamespace(
        pages=[page],
        corrected_evidence_pages=lambda: [evidence_page],
        _personal_detail_extraction_issues=[],
    )


def _parser_records(
    monkeypatch: pytest.MonkeyPatch,
    *candidates: NativeLabeledRecord,
) -> None:
    monkeypatch.setattr(
        PBOCPersonalDetailNativeParser,
        "records",
        lambda _self, name: list(candidates) if name == "report_header" else [],
    )


def test_parser_header_candidate_accepts_one_exact_owned_cell_on_arbitrary_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logical_page = 17
    source_page = 9
    value = "周明远"
    ref = _corrected_name_ref(
        value,
        logical_page=logical_page,
        source_page=source_page,
    )
    context = _corrected_name_context(
        value,
        logical_page=logical_page,
        source_page=source_page,
    )
    _parser_records(monkeypatch, _header_candidate(value, refs=(ref,)))

    metadata = _extract_header_datasets(context, "")["personal_report_metadata"][0]

    assert metadata["subject_name"] == value
    subject_ref = next(
        source_ref
        for source_ref in metadata["source_refs"]
        if source_ref["field_name"] == "subject_name"
    )
    assert subject_ref["logical_page"] == logical_page
    assert subject_ref["source_page"] == source_page
    assert subject_ref["canonical_template_id"] == "report_header_and_identity"
    assert subject_ref["geometry_scope"] == "cell"
    assert subject_ref["evidence_ids"]


def test_parser_header_candidate_without_field_ref_is_observed_but_not_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = "周明远"
    context = _context()
    _parser_records(monkeypatch, _header_candidate(value))

    metadata = _extract_header_datasets(context, "")["personal_report_metadata"][0]

    assert metadata["subject_name"] is None
    issue = next(
        issue
        for issue in context._personal_detail_extraction_issues
        if issue.get("field_name") == "subject_name"
    )
    assert value in issue["observed_value"]
    assert "parser_candidate_exact_source_ownership_unresolved" in issue["reason_codes"]


def test_parser_header_candidate_with_foreign_owner_is_not_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = "周明远"
    logical_page = 23
    source_page = 11
    context = _corrected_name_context(
        value,
        logical_page=logical_page,
        source_page=source_page,
        owner="public_information",
    )
    ref = _corrected_name_ref(
        value,
        logical_page=logical_page,
        source_page=source_page,
    )
    _parser_records(monkeypatch, _header_candidate(value, refs=(ref,)))

    metadata = _extract_header_datasets(context, "")["personal_report_metadata"][0]

    assert metadata["subject_name"] is None
    issue = next(
        issue
        for issue in context._personal_detail_extraction_issues
        if issue.get("field_name") == "subject_name"
    )
    assert value in issue["observed_value"]


def test_parser_header_candidate_with_duplicate_field_refs_is_not_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = "周明远"
    logical_page = 29
    source_page = 14
    context = _corrected_name_context(
        value,
        logical_page=logical_page,
        source_page=source_page,
    )
    ref = _corrected_name_ref(
        value,
        logical_page=logical_page,
        source_page=source_page,
    )
    _parser_records(monkeypatch, _header_candidate(value, refs=(ref, dict(ref))))

    metadata = _extract_header_datasets(context, "")["personal_report_metadata"][0]

    assert metadata["subject_name"] is None
    assert any(
        issue.get("field_name") == "subject_name"
        and value in issue.get("observed_value", ())
        for issue in context._personal_detail_extraction_issues
    )


def test_parser_header_candidate_with_duplicate_registered_owner_is_not_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = "周明远"
    logical_page = 31
    source_page = 15
    context = _corrected_name_context(
        value,
        logical_page=logical_page,
        source_page=source_page,
    )
    context.pages.append(
        SimpleNamespace(
            page_number=logical_page,
            source_page_number=source_page,
            canonical_template_id="report_header_and_identity",
            tables=[],
        )
    )
    ref = _corrected_name_ref(
        value,
        logical_page=logical_page,
        source_page=source_page,
    )
    _parser_records(monkeypatch, _header_candidate(value, refs=(ref,)))

    metadata = _extract_header_datasets(context, "")["personal_report_metadata"][0]

    assert metadata["subject_name"] is None


def test_native_and_independent_exact_header_values_conflict_and_are_withheld(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native_value = "信小达"
    corrected_value = "周明远"
    table = _primary_table()
    context = _context(table)
    corrected_ref = _corrected_name_ref(
        corrected_value,
        logical_page=3,
        source_page=2,
        evidence_id="corrected:independent:subject-name",
    )
    context.corrected_evidence_pages = lambda: [
        {
            "page": 3,
            "source_page": 2,
            "canonical_template_id": "report_header_and_identity",
            "lines": [
                {
                    "text": corrected_value,
                    "bbox": list(corrected_ref["bbox"]),
                    "source_bbox": list(corrected_ref["bbox"]),
                    "evidence_ids": list(corrected_ref["evidence_ids"]),
                }
            ],
        }
    ]
    native_ref = {
        "source": "native_detail_tolerant_table_cell",
        "logical_page": 3,
        "source_page": 2,
        "table_id": "primary",
        "row": 1,
        "column": 0,
        "bbox": list(table.metadata["source_cell_bboxes"][1][0]),
        "geometry_scope": "cell",
        "field_name": "被查询者姓名",
        "binding": "label_column",
        "evidence_ids": list(table.metadata["cell_evidence_ids"][1][0]),
    }
    _parser_records(
        monkeypatch,
        _header_candidate(
            native_value,
            refs=(native_ref,),
            binding_quality="native_label_column",
        ),
        _header_candidate(corrected_value, refs=(corrected_ref,)),
    )

    metadata = _extract_header_datasets(context, "")["personal_report_metadata"][0]

    assert metadata["subject_name"] is None
    issue = next(
        issue
        for issue in context._personal_detail_extraction_issues
        if issue.get("field_name") == "subject_name"
    )
    assert set(issue["observed_value"]) == {native_value, corrected_value}
    assert len(issue["source_refs"]) >= 2


@pytest.mark.parametrize(
    "raw_ids",
    (
        ["source:primary:1:0", ""],
        ["source:primary:1:0", 7],
        ["source:primary:1:0", "source:primary:1:0"],
        [" source:primary:1:0"],
    ),
)
def test_direct_header_cell_rejects_lossy_or_replayed_raw_evidence_ids(
    raw_ids: list[object],
) -> None:
    table = _primary_table()
    table.metadata["cell_evidence_ids"][1][0] = raw_ids
    context = _context(table)

    metadata = _extract_header_datasets(context, "")["personal_report_metadata"][0]

    assert metadata["subject_name"] is None
    assert any(
        issue.get("field_name") == "subject_name"
        and "信小达" in issue.get("observed_value", ())
        for issue in context._personal_detail_extraction_issues
    )


@pytest.mark.parametrize(
    ("key", "value"),
    (
        ("logical_page", True),
        ("source_page", "9"),
        ("evidence_ids", ["corrected:header:subject-name", ""]),
        ("evidence_ids", ["corrected:header:subject-name", 9]),
    ),
)
def test_parser_header_ref_rejects_coerced_coordinates_or_lossy_ids(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    value: object,
) -> None:
    name = "周明远"
    context = _corrected_name_context(
        name,
        logical_page=17,
        source_page=9,
    )
    ref = _corrected_name_ref(name, logical_page=17, source_page=9)
    ref[key] = value
    _parser_records(monkeypatch, _header_candidate(name, refs=(ref,)))

    metadata = _extract_header_datasets(context, "")["personal_report_metadata"][0]

    assert metadata["subject_name"] is None


def test_native_header_candidate_rejects_evidence_replayed_in_another_cell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = "周明远"
    table = _exact_table("replayed-native", [[value, value]])
    replayed_id = table.metadata["cell_evidence_ids"][0][0]
    table.metadata["cell_evidence_ids"][0][1] = list(replayed_id)
    context = _context(table)
    ref = {
        "source": "native_detail_tolerant_table_cell",
        "logical_page": 3,
        "source_page": 2,
        "table_id": "replayed-native",
        "row": 0,
        "column": 0,
        "bbox": list(table.metadata["source_cell_bboxes"][0][0]),
        "geometry_scope": "cell",
        "field_name": "被查询者姓名",
        "binding": "label_column",
        "evidence_ids": list(replayed_id),
    }
    _parser_records(
        monkeypatch,
        _header_candidate(
            value,
            refs=(ref,),
            binding_quality="native_label_column",
        ),
    )

    metadata = _extract_header_datasets(context, "")["personal_report_metadata"][0]

    assert metadata["subject_name"] is None


def test_corrected_header_candidate_rejects_evidence_replayed_on_another_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = "周明远"
    context = _corrected_name_context(
        value,
        logical_page=17,
        source_page=9,
    )
    evidence_page = context.corrected_evidence_pages()[0]
    replayed = deepcopy(evidence_page["lines"][0])
    replayed["bbox"] = [140.0, 60.0, 240.0, 78.0]
    replayed["source_bbox"] = list(replayed["bbox"])
    context.corrected_evidence_pages = lambda: [
        {**evidence_page, "lines": [*evidence_page["lines"], replayed]}
    ]
    ref = _corrected_name_ref(value, logical_page=17, source_page=9)
    _parser_records(monkeypatch, _header_candidate(value, refs=(ref,)))

    metadata = _extract_header_datasets(context, "")["personal_report_metadata"][0]

    assert metadata["subject_name"] is None


def test_owned_reordered_other_identity_table_extracts_without_page_authority() -> None:
    additional = _exact_table(
        "additional",
        [
            ["其他证件信息", ""],
            ["证件号码", "证件类型"],
            ["CHN1234567", "护照"],
        ],
    )
    context = _context(_primary_table(), additional)

    datasets = _extract_header_datasets(
        context,
        "报告编号:2025052510051518624525 报告时间:2025.05.25 10:05:15",
    )

    metadata = datasets["personal_report_metadata"][0]
    assert metadata["query_institution"] == "中国人民银行营业管理部"
    assert metadata["query_reason"] == "本人查询(临柜)"
    assert {ref["logical_page"] for ref in metadata["source_refs"]} == {3}
    assert all(ref["geometry_scope"] == "cell" for ref in metadata["source_refs"])
    assert [row["sequence"] for row in datasets["identity_documents"]] == [1, 2]
    secondary = datasets["identity_documents"][1]
    assert secondary["document_type"] == "护照"
    assert secondary["document_number"] == "CHN1234567"
    assert {ref["column"] for ref in secondary["source_refs"]} == {0, 1}
    assert all(ref["evidence_ids"] for ref in secondary["source_refs"])


def test_spouse_columns_without_other_identity_owner_never_create_subject_document() -> None:
    spouse = _exact_table(
        "spouse",
        [
            ["姓名", "证件类型", "证件号码", "工作单位", "联系电话"],
            ["李某", "护照", "CHN7654321", "某公司", "13899999999"],
        ],
    )

    datasets = _extract_header_datasets(_context(_primary_table(), spouse), "")

    assert len(datasets["identity_documents"]) == 1
    assert datasets["identity_documents"][0]["is_primary"] is True


def test_unregistered_exact_label_table_does_not_own_additional_identity() -> None:
    additional = _exact_table(
        "unowned",
        [
            ["其他证件信息", ""],
            ["证件类型", "证件号码"],
            ["护照", "CHN1234567"],
        ],
    )
    additional.metadata.pop("canonical_template_id")

    datasets = _extract_header_datasets(_context(additional, owned=False), "")

    assert datasets["identity_documents"] == []


@pytest.mark.parametrize("logical_page", [1, 19])
def test_complete_header_table_on_public_page_is_not_header_authority(
    logical_page: int,
) -> None:
    table = _primary_table()
    table.metadata["canonical_template_id"] = "public_information"
    page = SimpleNamespace(
        page_number=logical_page,
        source_page_number=logical_page + 4,
        canonical_template_id="public_information",
        tables=[table],
    )
    context = SimpleNamespace(
        pages=[page],
        _personal_detail_extraction_issues=[],
    )

    datasets = _extract_header_datasets(
        context,
        "报告编号:2025052510051518624525 报告时间:2025.05.25 10:05:15",
    )

    metadata = datasets["personal_report_metadata"][0]
    assert metadata["subject_name"] is None
    assert metadata["report_number"] is None
    assert datasets["identity_documents"] == []


def test_duplicate_other_identity_owners_fail_closed() -> None:
    rows = [
        ["其他证件信息", ""],
        ["证件类型", "证件号码"],
        ["护照", "CHN1234567"],
    ]

    datasets = _extract_header_datasets(
        _context(_primary_table(), _exact_table("a", rows), _exact_table("b", rows)),
        "",
    )

    assert len(datasets["identity_documents"]) == 1


def test_invalid_owned_primary_number_is_withheld_with_exact_field_ref() -> None:
    table = _primary_table()
    table.metadata["raw_rows"][1][2] = "622926198108151111"
    context = _context(table)

    datasets = _extract_header_datasets(context, "")

    assert datasets["personal_report_metadata"][0]["primary_id_number"] is None
    assert datasets["identity_documents"][0]["document_number"] == "622926198108151111"
    issue = next(
        issue
        for issue in context._personal_detail_extraction_issues
        if issue.get("field_name") == "primary_id_number"
    )
    assert issue["source_refs"][0]["field_name"] == "primary_id_number"
    assert issue["source_refs"][0]["geometry_scope"] == "cell"
    assert issue["source_refs"][0]["logical_page"] == 3
    assert issue["source_refs"][0]["evidence_ids"]


def _lifecycle_ref(
    field_name: str,
    *,
    row: int,
    column: int,
    table_id: str = "header-owner",
    evidence_id: str | None = None,
) -> dict[str, object]:
    return {
        "source": "native_detail_table_cell",
        "logical_page": 3,
        "source_page": 2,
        "table_id": table_id,
        "row": row,
        "column": column,
        "canonical_row": row,
        "canonical_column": column,
        "bbox": [
            10.0 + column * 100.0,
            20.0 + row * 20.0,
            110.0 + column * 100.0,
            40.0 + row * 20.0,
        ],
        "geometry_scope": "cell",
        "evidence_ids": [evidence_id or f"source:{table_id}:{row}:{column}"],
        "binding": "canonical_field_slot",
        "binding_quality": "canonical_header_column",
        "field_name": field_name,
        "canonical_template_id": "report_header_and_identity",
    }


def _lifecycle_metadata(
    *,
    subject_name: str = "周明远",
    refs: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "personal_report_metadata_id": "personal_report_metadata:discovery",
        "report_number": None,
        "report_time": None,
        "subject_name": subject_name,
        "primary_id_type": "身份证",
        "primary_id_number": "110101199003078451",
        "query_institution": None,
        "query_reason": None,
        "reporting_currency": "CNY",
        "reporting_amount_unit": "yuan",
        "source": "native_detail_header",
        "source_refs": refs
        or [
            _lifecycle_ref("subject_name", row=1, column=0),
            _lifecycle_ref("primary_id_type", row=1, column=1),
            _lifecycle_ref("primary_id_number", row=1, column=2),
        ],
        "confidence": 1.0,
    }


def _lifecycle_identity(
    *,
    is_primary: bool,
    document_type: str,
    document_number: str,
    row: int,
    table_id: str = "header-owner",
    sequence: int = 1,
) -> dict[str, object]:
    return {
        "identity_document_id": f"identity:{table_id}:{row}",
        "sequence": sequence,
        "holder_name": "周明远",
        "document_type": document_type,
        "document_number": document_number,
        "is_primary": is_primary,
        "source": "native_detail_header",
        "source_refs": [
            _lifecycle_ref(
                "document_type",
                row=row,
                column=1,
                table_id=table_id,
            ),
            _lifecycle_ref(
                "document_number",
                row=row,
                column=2,
                table_id=table_id,
            ),
        ],
        "confidence": 1.0,
    }


def _reconcile(
    discovery: dict[str, list[dict[str, object]]],
    repaired: dict[str, list[dict[str, object]]],
) -> tuple[dict[str, list[dict[str, object]]], SimpleNamespace]:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    return (
        _reconcile_candidate_b_header_lifecycle(context, discovery, repaired),
        context,
    )


def test_candidate_b_lifecycle_preserves_exact_primary_when_repair_keeps_secondary() -> None:
    primary = _lifecycle_identity(
        is_primary=True,
        document_type="身份证",
        document_number="110101199003078451",
        row=1,
    )
    secondary = _lifecycle_identity(
        is_primary=False,
        document_type="护照",
        document_number="EA1234567",
        row=4,
        table_id="other-identity-owner",
        sequence=2,
    )

    reconciled, context = _reconcile(
        {
            "personal_report_metadata": [_lifecycle_metadata()],
            "identity_documents": [primary, secondary],
        },
        {
            "personal_report_metadata": [_lifecycle_metadata(refs=[])],
            "identity_documents": [secondary],
        },
    )

    assert [
        (row["document_type"], row["is_primary"])
        for row in reconciled["identity_documents"]
    ] == [("身份证", True), ("护照", False)]
    assert reconciled["personal_report_metadata"][0]["subject_name"] == "周明远"
    assert all(
        ref["canonical_template_id"] == "report_header_and_identity"
        for row in reconciled["identity_documents"]
        for ref in row["source_refs"]
    )
    assert context._personal_detail_extraction_issues == []


def test_candidate_b_lifecycle_does_not_preserve_unregistered_or_geometry_only_rows() -> None:
    primary = _lifecycle_identity(
        is_primary=True,
        document_type="身份证",
        document_number="110101199003078451",
        row=1,
    )
    primary["source_refs"][0].pop("canonical_template_id")
    primary["source_refs"][1]["geometry_scope"] = "canonical_field_slot"
    primary["source_refs"][1].pop("bbox")
    metadata = _lifecycle_metadata()
    for ref in metadata["source_refs"]:
        ref.pop("canonical_template_id")

    reconciled, _context = _reconcile(
        {
            "personal_report_metadata": [metadata],
            "identity_documents": [primary],
        },
        {"personal_report_metadata": [], "identity_documents": []},
    )

    assert reconciled["personal_report_metadata"] == []
    assert reconciled["identity_documents"] == []


def test_candidate_b_lifecycle_exact_primary_conflict_withholds_both_planes() -> None:
    discovery = _lifecycle_identity(
        is_primary=True,
        document_type="身份证",
        document_number="110101199003078451",
        row=1,
    )
    repaired = _lifecycle_identity(
        is_primary=True,
        document_type="身份证",
        document_number="110101199003078452",
        row=1,
    )

    reconciled, context = _reconcile(
        {"identity_documents": [discovery]},
        {"identity_documents": [repaired]},
    )

    assert reconciled["identity_documents"] == []
    issue = next(
        issue
        for issue in context._personal_detail_extraction_issues
        if issue["issue_code"] == "candidate_b_header_lifecycle_conflict"
    )
    assert "independent_exact_identity_cell_conflict" in issue["reason_codes"]


def test_candidate_b_lifecycle_duplicate_identity_owner_fails_closed() -> None:
    first = _lifecycle_identity(
        is_primary=True,
        document_type="身份证",
        document_number="110101199003078451",
        row=1,
    )
    duplicate = _lifecycle_identity(
        is_primary=True,
        document_type="身份证",
        document_number="110101199003078451",
        row=2,
        table_id="duplicate-owner",
    )

    reconciled, context = _reconcile(
        {"identity_documents": [first, duplicate]},
        {"identity_documents": []},
    )

    assert reconciled["identity_documents"] == []
    assert any(
        "duplicate_exact_primary_identity_owner" in issue.get("reason_codes", ())
        for issue in context._personal_detail_extraction_issues
    )


def test_candidate_b_lifecycle_reordered_exact_slots_remain_authoritative() -> None:
    primary = _lifecycle_identity(
        is_primary=True,
        document_type="身份证",
        document_number="110101199003078451",
        row=7,
        table_id="reordered-header-owner",
    )
    primary["source_refs"][0]["column"] = 4
    primary["source_refs"][0]["canonical_column"] = 4
    primary["source_refs"][0]["bbox"] = [410.0, 160.0, 510.0, 180.0]
    primary["source_refs"][1]["column"] = 0
    primary["source_refs"][1]["canonical_column"] = 0
    primary["source_refs"][1]["bbox"] = [10.0, 160.0, 110.0, 180.0]

    reconciled, _context = _reconcile(
        {"identity_documents": [primary]},
        {"identity_documents": []},
    )

    assert len(reconciled["identity_documents"]) == 1
    assert reconciled["identity_documents"][0]["document_number"] == "110101199003078451"


def test_candidate_b_lifecycle_malformed_repaired_header_row_cannot_bypass_owner_gate() -> None:
    malformed = _lifecycle_identity(
        is_primary=True,
        document_type="身份证",
        document_number="110101199003078451",
        row=1,
    )
    malformed["source_refs"][0]["canonical_row"] = 2

    reconciled, _context = _reconcile(
        {"identity_documents": []},
        {"identity_documents": [malformed]},
    )

    assert reconciled["identity_documents"] == []


def test_candidate_b_lifecycle_duplicate_metadata_owner_withholds_field() -> None:
    duplicate_refs = [
        _lifecycle_ref("subject_name", row=1, column=0),
        _lifecycle_ref(
            "subject_name",
            row=2,
            column=0,
            table_id="duplicate-header-owner",
        ),
    ]
    metadata = _lifecycle_metadata(refs=duplicate_refs)

    reconciled, context = _reconcile(
        {"personal_report_metadata": [metadata]},
        {"personal_report_metadata": [_lifecycle_metadata(refs=[])]},
    )

    assert reconciled["personal_report_metadata"][0]["subject_name"] is None
    assert any(
        "duplicate_exact_header_field_owner" in issue.get("reason_codes", ())
        for issue in context._personal_detail_extraction_issues
    )


def test_candidate_b_second_pass_reconciles_discovery_header_before_final_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the actual CandidateB repair boundary, not only its helper."""

    primary = _lifecycle_identity(
        is_primary=True,
        document_type="身份证",
        document_number="110101199003078451",
        row=1,
    )
    secondary = _lifecycle_identity(
        is_primary=False,
        document_type="护照",
        document_number="EA1234567",
        row=4,
        table_id="other-identity-owner",
        sequence=2,
    )
    pass_index = 0
    corrected_header: dict[str, list[dict[str, object]]] = {}

    context = SimpleNamespace(
        pages=[],
        reading_order_by_logical={},
        account_collections=lambda: ([], [], []),
        corrected_repayment_records=lambda: [],
        corrected_repayment_micro_grids=lambda: [],
        corrected_evidence_pages=lambda: [],
        candidate_b_status_glyph_observations=lambda: [],
        prepare_candidate_b_business_repair=lambda _payload: True,
        ocr_correction_audit=lambda: {
            "business_repair": {"second_schema_pass_required": True}
        },
        canonical_layout_audit=lambda: {},
        page_topology_audit=lambda: {},
        _personal_detail_extraction_issues=[],
    )

    def correct(payload: dict[str, object]) -> dict[str, object]:
        corrected_header["identity_documents"] = list(
            payload.get("identity_documents") or []
        )
        return payload

    context.correct_candidate_b_datasets = correct

    def header_datasets(_context: object, _text: str) -> dict[str, list[dict[str, object]]]:
        nonlocal pass_index
        pass_index += 1
        if pass_index == 1:
            return {
                "personal_report_metadata": [_lifecycle_metadata()],
                "identity_documents": [primary, secondary],
            }
        return {
            "personal_report_metadata": [_lifecycle_metadata(refs=[])],
            "identity_documents": [secondary],
        }

    monkeypatch.setattr(native_extraction, "_extract_header_datasets", header_datasets)

    def empty(_context: object) -> list[dict[str, object]]:
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
        "reconcile_candidate_b_credit_lines",
        lambda _context, rows: rows,
    )
    monkeypatch.setattr(
        "docmirror.plugins.credit_report.personal_detail_scanned.profile_extraction.extract_candidate_b_profile",
        lambda _context: {},
    )
    monkeypatch.setattr(
        "docmirror.plugins.credit_report.personal_detail_scanned.relations.link_candidate_b_repayments",
        lambda rows, *_args, **_kwargs: rows,
    )
    monkeypatch.setattr(
        "docmirror.plugins.credit_report.personal_detail_scanned.relations.derive_candidate_b_overdue_records",
        lambda *_args: [],
    )
    monkeypatch.setattr(
        "docmirror.plugins.credit_report.personal_detail_scanned.consistency_ledger.apply_document_consistency_ledger",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        "docmirror.plugins.credit_report.personal_detail_scanned.document_glyph_bank.apply_document_local_status_glyph_bank",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        "docmirror.plugins.credit_report.personal_detail_scanned.native_status_conflict.apply_candidate_b_native_status_conflict_guard",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        "docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues.register_final_liability_issue_records",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "docmirror.plugins.credit_report.personal_detail_scanned.source_projection.prepare_personal_detail_source_collections",
        lambda content, _business, **_kwargs: content,
    )

    result = CandidateBPipeline(context, "").run()

    assert pass_index == 2
    assert [
        (row["document_type"], row["is_primary"])
        for row in corrected_header["identity_documents"]
    ] == [("身份证", True), ("护照", False)]
    assert len(result.section_content["datasets"]["identity_documents"]) == 2
