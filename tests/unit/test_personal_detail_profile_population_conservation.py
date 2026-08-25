from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
    make_issue,
)
from docmirror.plugins.credit_report.personal_detail_scanned.fail_closed_field_reporting import (
    append_fail_closed_field_issues,
)
from docmirror.plugins.credit_report.personal_detail_scanned.native_extraction import (
    _record_pre_repair_source_gaps,
    _sealed_raw_profile_population_census,
    _source_completeness_ledger,
)
from docmirror.plugins.credit_report.personal_detail_scanned.source_projection import (
    prepare_personal_detail_source_collections,
)
from docmirror.plugins.credit_report.value_utils import stable_record_id


def _text(content: str, top: float, evidence_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        content=content,
        bbox=[20.0, top, 180.0, top + 8.0],
        evidence_ids=[evidence_id],
    )


def _table(
    table_id: str,
    rows: list[list[str]],
    *,
    top: float,
) -> SimpleNamespace:
    column_count = max(len(row) for row in rows)
    row_bands = [
        {"index": index, "y0": top + index * 12.0, "y1": top + (index + 1) * 12.0}
        for index in range(len(rows))
    ]
    column_bands = [
        {"index": index, "x0": 20.0 + index * 80.0, "x1": 100.0 + index * 80.0}
        for index in range(column_count)
    ]
    cell_bboxes = [
        [
            [
                column_bands[column]["x0"],
                row_bands[row]["y0"],
                column_bands[column]["x1"],
                row_bands[row]["y1"],
            ]
            for column in range(column_count)
        ]
        for row in range(len(rows))
    ]
    evidence = [
        [f"{table_id}:r{row}:c{column}" for column in range(column_count)]
        for row in range(len(rows))
    ]
    return SimpleNamespace(
        table_id=table_id,
        bbox=[20.0, top, 20.0 + column_count * 80.0, top + len(rows) * 12.0],
        metadata={
            "raw_rows": deepcopy(rows),
            "geometry": {
                "row_bands": row_bands,
                "col_bands": column_bands,
                "cell_bboxes": cell_bboxes,
                "cell_geometry_status": [
                    ["exact"] * column_count for _row in rows
                ],
                "cell_evidence_ids": [
                    [[evidence_id] for evidence_id in row] for row in evidence
                ],
            },
        },
    )


def _profile_context() -> SimpleNamespace:
    mobile = _table(
        "mobile-source",
        [
            ["编号", "手机号码", "信息更新日期", "数据发生机构名称"],
            ["1", "13800000000", "2025-01-01", "甲机构"],
            ["2", "13900000000", "2025-02-02", "乙机构"],
        ],
        top=35.0,
    )
    residence = _table(
        "residence-source",
        [
            ["编号", "居住地址", "住宅电话 居住状况 信息更新日期"],
            ["1", "甲地址", "01000000000 自置 2025-01-01"],
            ["2", "乙地址", "02000000000 租赁 2025-02-02"],
            ["编号", "数据发生机构名称"],
            ["1", "甲机构"],
            ["2", "乙机构"],
        ],
        top=125.0,
    )
    employment = _table(
        "employment-source",
        [
            ["编号", "工作单位 单位性质 单位电话 单位地址"],
            ["1", "甲单位 国有 01000000000 甲地址"],
            ["2", "乙单位 民营 02000000000 乙地址"],
            ["编号", "职业 行业 职务 职称 进入本单位年份 信息更新日期"],
            ["1", "职员 制造业 经理 中级 2018 2025-01-01"],
            ["2", "职员 服务业 主管 初级 2020 2025-02-02"],
            # A damaged supplemental header must not erase the already-proven
            # primary population or manufacture provider-field ownership.
            ["编号 1", "数据发生机构名称"],
            ["1", "甲机构"],
            ["2", "乙机构"],
        ],
        top=275.0,
    )
    page = SimpleNamespace(
        page_number=73,
        source_page_number=41,
        texts=[
            _text("（九）个人基本信息", 5.0, "profile:start"),
            _text("（三）居住信息", 105.0, "profile:residence"),
            _text("（四）职业信息", 255.0, "profile:employment"),
            _text("（十）信息概要", 410.0, "profile:boundary"),
        ],
        tables=[employment, mobile, residence],
    )
    return SimpleNamespace(
        pages=[],
        _frozen_logical_pages={73: page},
        reading_order_by_logical={73: 1},
        reading_order_resolution={"resolved": True, "authoritative": True},
        _personal_detail_extraction_issues=[],
    )


def test_frozen_profile_census_survives_canonical_registration_failure() -> None:
    context = _profile_context()

    census = _sealed_raw_profile_population_census(context)

    assert census is not None
    assert census["sequences"] == {
        "mobile_phone_records": [1, 2],
        "residence_records": [1, 2],
        "employment_records": [1, 2],
    }
    assert not census["vetoed_datasets"]
    residence = census["ordinal_observations"]["residence_records"][1]
    assert set(residence["printed_fields"]) == {"address", "data_provider"}
    assert not {
        "residential_phone",
        "residence_status",
        "information_updated_date",
    }.intersection(residence["printed_fields"])
    employment = census["ordinal_observations"]["employment_records"][1]
    assert employment["printed_fields"] == []
    assert all(
        ref["canonical_template_id"] == "report_header_and_identity"
        and ref["binding"] == "printed_profile_sequence"
        for observations in census["ordinal_observations"].values()
        for observation in observations.values()
        for ref in observation["source_refs"]
    )
    assert all(
        field not in observation
        for observations in census["ordinal_observations"].values()
        for observation in observations.values()
        for field in (
            "mobile_phone",
            "address",
            "employer",
            "data_provider",
        )
    )

    ledger = _source_completeness_ledger(context)
    assert ledger["sequence_endpoints"] == {
        "mobile_phone_records": 2,
        "residence_records": 2,
        "employment_records": 2,
    }


def test_shared_frozen_topology_orders_split_profile_without_page_number_guessing() -> None:
    context = _profile_context()
    original = context._frozen_logical_pages[73]
    left = SimpleNamespace(
        page_number=73,
        source_page_number=41,
        texts=original.texts[:2],
        tables=[
            table
            for table in original.tables
            if table.table_id in {"mobile-source", "residence-source"}
        ],
    )
    right = SimpleNamespace(
        page_number=9,
        source_page_number=41,
        texts=original.texts[2:],
        tables=[
            table
            for table in original.tables
            if table.table_id == "employment-source"
        ],
    )
    context._frozen_logical_pages = {9: right, 73: left}
    context.reading_order_by_logical = {73: 73, 9: 9}
    context.reading_order_resolution = {"resolved": False, "authoritative": False}
    context.source_page_by_logical = {73: 41, 9: 41}
    geometries = {
        logical: SimpleNamespace(
            source_page=41,
            width=800.0,
            height=600.0,
            split_kind="two_page_spread",
            segment_index=segment,
            selected_rotation=0,
            source_crop_bbox=(segment * 400.0, 0.0, (segment + 1) * 400.0, 600.0),
            transform_usable=True,
        )
        for segment, logical in enumerate((73, 9))
    }
    context.page_topology = SimpleNamespace(
        audit=lambda: {"valid": True},
        geometry=lambda logical: geometries.get(logical),
        ordered_fragments=lambda source: (73, 9) if source == 41 else (),
        ordered_pair=lambda logicals: (73, 9)
        if set(logicals) == {73, 9}
        else None,
    )

    census = _sealed_raw_profile_population_census(context)

    assert census is not None
    assert census["sequences"] == {
        "mobile_phone_records": [1, 2],
        "residence_records": [1, 2],
        "employment_records": [1, 2],
    }


@pytest.mark.parametrize("bad_page_id", [True, "73", 0])
def test_profile_census_rejects_coerced_or_nonpositive_page_ids(
    bad_page_id: object,
) -> None:
    context = _profile_context()
    page = context._frozen_logical_pages.pop(73)
    page.page_number = bad_page_id
    context._frozen_logical_pages = {bad_page_id: page}
    context.reading_order_by_logical = {bad_page_id: 1}

    assert _sealed_raw_profile_population_census(context) is None


@pytest.mark.parametrize(
    "sequences",
    [
        ("1", "1"),
        ("1", "3"),
        ("2", "1"),
        ("01", "2"),
    ],
)
def test_profile_census_never_fills_or_reorders_printed_ordinals(
    sequences: tuple[str, str],
) -> None:
    context = _profile_context()
    mobile = next(
        table
        for table in context._frozen_logical_pages[73].tables
        if table.table_id == "mobile-source"
    )
    mobile.metadata["raw_rows"][1][0] = sequences[0]
    mobile.metadata["raw_rows"][2][0] = sequences[1]

    census = _sealed_raw_profile_population_census(context)

    assert census is not None
    assert "mobile_phone_records" not in census["sequences"]
    assert "mobile_phone_records" in census["vetoed_datasets"]


def test_profile_census_rejects_replayed_cell_evidence() -> None:
    context = _profile_context()
    page = context._frozen_logical_pages[73]
    mobile = next(table for table in page.tables if table.table_id == "mobile-source")
    residence = next(
        table for table in page.tables if table.table_id == "residence-source"
    )
    residence.metadata["geometry"]["cell_evidence_ids"][1][0] = deepcopy(
        mobile.metadata["geometry"]["cell_evidence_ids"][1][0]
    )

    census = _sealed_raw_profile_population_census(context)

    assert census is not None
    assert "mobile_phone_records" not in census["sequences"]
    assert "residence_records" not in census["sequences"]
    assert {"mobile_phone_records", "residence_records"} <= set(
        census["vetoed_datasets"]
    )


@pytest.mark.parametrize(
    "bad_ids",
    (
        ["mobile-source:r1:c0", ""],
        ["mobile-source:r1:c0", "mobile-source:r1:c0"],
        [" mobile-source:r1:c0"],
        [17],
    ),
)
def test_profile_census_rejects_noncanonical_raw_evidence_id_lists(
    bad_ids: list[object],
) -> None:
    context = _profile_context()
    mobile = next(
        table
        for table in context._frozen_logical_pages[73].tables
        if table.table_id == "mobile-source"
    )
    mobile.metadata["geometry"]["cell_evidence_ids"][1][0] = bad_ids

    census = _sealed_raw_profile_population_census(context)

    assert census is not None
    assert "mobile_phone_records" not in census["sequences"]
    assert "mobile_phone_records" in census["vetoed_datasets"]


def test_profile_table_under_foreign_subsection_cannot_own_population() -> None:
    context = _profile_context()
    page = context._frozen_logical_pages[73]
    employment_heading = next(
        text for text in page.texts if "职业信息" in text.content
    )
    employment_heading.bbox = [20.0, 115.0, 180.0, 123.0]

    census = _sealed_raw_profile_population_census(context)

    assert census is not None
    assert "residence_records" not in census["sequences"]
    assert "residence_records" in census["vetoed_datasets"]


def test_pre_repair_profile_gaps_use_each_missing_ordinal_cell_not_endpoint_pages() -> None:
    context = _profile_context()

    ledger = _record_pre_repair_source_gaps(
        context,
        {
            "mobile_phone_records": [],
            "residence_records": [],
            "employment_records": [],
        },
    )

    assert ledger["sequence_endpoints"] == {
        "mobile_phone_records": 2,
        "residence_records": 2,
        "employment_records": 2,
    }
    profile_issues = [
        issue
        for issue in context._personal_detail_extraction_issues
        if issue.get("parser_stage") == "candidate_b_pre_repair_source_coverage"
        and issue.get("target_dataset")
        in {
            "mobile_phone_records",
            "residence_records",
            "employment_records",
        }
    ]
    assert len(profile_issues) == 6
    assert all(
        len(issue["source_refs"]) == 1
        and issue["source_refs"][0]["source"]
        == "candidate_b_raw_profile_sequence_cell"
        and issue["source_refs"][0]["sequence"]
        == issue["observed_value"]["source_sequence"]
        for issue in profile_issues
    )


def _projection_content(
    ledger: dict[str, object],
    *,
    datasets: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "facts": {"personal_detail_source_completeness_ledger": deepcopy(ledger)},
        "datasets": {
            "mobile_phone_records": [],
            "residence_records": [],
            "employment_records": [],
            "personal_detail_extraction_issues": [],
            **(deepcopy(datasets) if datasets else {}),
        },
    }


def _issue_values(content: dict[str, object]) -> list[dict[str, object]]:
    datasets = content["datasets"]
    assert isinstance(datasets, dict)
    return [
        row.get("normalized", row)
        for row in datasets["personal_detail_extraction_issues"]
        if isinstance(row, dict)
    ]


def test_projection_conserves_rows_and_only_exact_whole_field_cells() -> None:
    ledger = _source_completeness_ledger(_profile_context())
    content = _projection_content(ledger)

    prepare_personal_detail_source_collections(content)
    prepare_personal_detail_source_collections(content)

    issues = _issue_values(content)
    assert {
        (row.get("issue_code"), row.get("target_dataset"), row.get("target_record_id"))
        for row in issues
        if str(row.get("issue_code") or "").endswith("_record_omitted")
    } == {
        (
            f"source_{stem}_record_omitted",
            dataset,
            stable_record_id(prefix, ordinal),
        )
        for dataset, stem, prefix in (
            ("mobile_phone_records", "mobile", "personal_mobile_phone"),
            ("residence_records", "residence", "credit_residence"),
            ("employment_records", "employment", "credit_employment"),
        )
        for ordinal in (1, 2)
    }
    exact_fields = {
        (
            row.get("target_dataset"),
            row.get("target_record_id"),
            row.get("field_name"),
        )
        for row in issues
        if str(row.get("issue_code") or "").endswith("_field_omitted")
    }
    assert exact_fields == {
        (
            "mobile_phone_records",
            stable_record_id("personal_mobile_phone", ordinal),
            field_name,
        )
        for ordinal in (1, 2)
        for field_name in (
            "mobile_phone",
            "information_updated_date",
            "data_provider",
        )
    } | {
        (
            "residence_records",
            stable_record_id("credit_residence", ordinal),
            field_name,
        )
        for ordinal in (1, 2)
        for field_name in ("address", "data_provider")
    }
    assert not any(
        dataset == "employment_records"
        for dataset, _target, _field in exact_fields
    )
    facts = content["facts"]
    assert isinstance(facts, dict)
    states = facts["personal_detail_dataset_states"]
    assert all(
        states[dataset]["presence_status"] == "partial"
        for dataset in (
            "mobile_phone_records",
            "residence_records",
            "employment_records",
        )
    )


def test_existing_field_absence_and_active_issue_are_not_reported_as_silent_loss() -> None:
    ledger = _source_completeness_ledger(_profile_context())
    target_id = stable_record_id("personal_mobile_phone", 1)
    active = make_issue(
        category="ocr_cell_level_error",
        issue_code="candidate_b_mobile_provider_unresolved",
        message="provider withheld",
        target_dataset="mobile_phone_records",
        target_record_id=target_id,
        field_name="data_provider",
        status="requires_review",
        source_refs=ledger["sequence_ordinal_observations"]["mobile_phone_records"]["1"][
            "field_source_refs"
        ]["data_provider"],
    )
    content = _projection_content(
        ledger,
        datasets={
            "mobile_phone_records": [
                {
                    "record_id": target_id,
                    "mobile_phone_record_id": target_id,
                    "sequence": 1,
                    "mobile_phone": "13800000000",
                    "information_updated_date": None,
                    "data_provider": None,
                    "_source_absent_fields": ["information_updated_date"],
                }
            ],
            "personal_detail_extraction_issues": [active],
        },
    )

    prepare_personal_detail_source_collections(content)

    issues = _issue_values(content)
    assert not any(
        row.get("issue_code") == "source_mobile_field_omitted"
        and row.get("target_record_id") == target_id
        for row in issues
    )
    assert sum(
        row.get("target_record_id") == target_id
        and row.get("field_name") == "data_provider"
        for row in issues
    ) == 1


@pytest.mark.parametrize(
    "mutation",
    (
        "endpoint_only",
        "missing_ordinal_ref",
        "wrong_source",
        "wrong_binding",
        "wrong_template",
        "replayed_evidence",
        "bool_page",
        "duplicate_ids",
        "empty_id",
        "multiple_sequence_slots",
        "shared_field_slot",
    ),
)
def test_projection_never_promotes_incomplete_or_foreign_profile_ledgers(
    mutation: str,
) -> None:
    ledger = deepcopy(_source_completeness_ledger(_profile_context()))
    mobile_observations = ledger["sequence_ordinal_observations"][
        "mobile_phone_records"
    ]
    if mutation == "endpoint_only":
        ledger.pop("sequence_observed_sequences")
    elif mutation == "missing_ordinal_ref":
        mobile_observations.pop("2")
    elif mutation == "replayed_evidence":
        mobile_observations["2"]["source_refs"][0]["evidence_ids"] = deepcopy(
            mobile_observations["1"]["source_refs"][0]["evidence_ids"]
        )
    elif mutation == "bool_page":
        mobile_observations["1"]["source_refs"][0]["logical_page"] = True
    elif mutation == "duplicate_ids":
        ref = mobile_observations["1"]["source_refs"][0]
        ref["evidence_ids"] = [ref["evidence_ids"][0], ref["evidence_ids"][0]]
    elif mutation == "empty_id":
        mobile_observations["1"]["source_refs"][0]["evidence_ids"].append("")
    elif mutation == "multiple_sequence_slots":
        mobile_observations["1"]["source_refs"].append(
            deepcopy(mobile_observations["1"]["source_refs"][0])
        )
    elif mutation == "shared_field_slot":
        sequence_ref = mobile_observations["1"]["source_refs"][0]
        field_ref = mobile_observations["1"]["field_source_refs"][
            "mobile_phone"
        ][0]
        for key in (
            "logical_page",
            "source_page",
            "table_id",
            "row",
            "column",
            "bbox",
            "evidence_ids",
        ):
            field_ref[key] = deepcopy(sequence_ref[key])
    else:
        ref = mobile_observations["1"]["source_refs"][0]
        if mutation == "wrong_source":
            ref["source"] = "native_detail_table_cell"
        elif mutation == "wrong_binding":
            ref["binding"] = "canonical_header_column"
        else:
            ref["canonical_template_id"] = "information_summary"
    content = _projection_content(ledger)

    prepare_personal_detail_source_collections(content)

    assert not any(
        row.get("target_dataset") == "mobile_phone_records"
        and str(row.get("issue_code") or "").startswith("source_mobile_")
        for row in _issue_values(content)
    )


def test_community_consumer_has_strict_row_branches_for_all_profile_datasets() -> None:
    ledger = _source_completeness_ledger(_profile_context())
    facts = {"personal_detail_source_completeness_ledger": ledger}
    datasets: dict[str, object] = {
        "mobile_phone_records": [],
        "residence_records": [],
        "employment_records": [],
        "personal_detail_extraction_issues": [],
    }

    append_fail_closed_field_issues(facts, datasets)
    append_fail_closed_field_issues(facts, datasets)

    rows = datasets["personal_detail_extraction_issues"]
    assert isinstance(rows, list)
    record_issues = [
        row
        for row in rows
        if isinstance(row, dict)
        and str(row.get("issue_code") or "").endswith("_record_omitted")
    ]
    assert len(record_issues) == 6
    assert {row["target_dataset"] for row in record_issues} == {
        "mobile_phone_records",
        "residence_records",
        "employment_records",
    }
    assert all(
        row["source_refs"][0]["source"]
        == "candidate_b_raw_profile_sequence_cell"
        for row in record_issues
    )
