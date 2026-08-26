from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from docmirror.plugins.credit_report.personal_detail_scanned.candidate_b_strategy import (
    CANDIDATE_B_STAGE_REGISTRY,
    INQUIRY_SECTION,
    REPORT_HEADER_SECTION,
    SECTION_TO_CANONICAL_DATASETS,
    plan_candidate_b_initial_extraction,
    stage_names_for_datasets,
)
from docmirror.plugins.credit_report.personal_detail_scanned.canonical_layout import (
    PBOCCanonicalTemplateAssembler,
)
from docmirror.plugins.credit_report.personal_detail_scanned.context import (
    PersonalDetailExtractionContext,
)
from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
    collect_extraction_issues,
)
from docmirror.plugins.credit_report.personal_detail_scanned.extraction_strategy import (
    MaterializationMode,
    SectionState,
)
from docmirror.plugins.credit_report.personal_detail_scanned.native_extraction import (
    _extract_inquiries,
    _inquiry_source_coverage,
    _source_completeness_ledger,
)
from docmirror.plugins.credit_report.personal_detail_scanned.page_topology import (
    LogicalPageGeometry,
)
from docmirror.plugins.credit_report.personal_detail_scanned.schema import (
    project_personal_detail_datasets,
)
from docmirror.plugins.credit_report.personal_detail_scanned.source_projection import (
    prepare_personal_detail_source_collections,
)
from docmirror.plugins.credit_report.personal_detail_scanned.table_ownership import (
    INQUIRY_AUTHORITY_SCHEMA_CARRY_ONLY,
    canonical_inquiry_population_metadata,
    canonical_table_role,
)
from docmirror.plugins.credit_report.shared.entity_decoder import (
    CreditReportEntity,
    CreditReportEntityContext,
    CreditReportUnit,
)


_FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "personal_detail"
    / "lin_inquiry_production_replay.json"
)
_UNSET = object()


def _spec() -> dict[str, Any]:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


def _line(raw: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "text": str(raw["text"]),
        "bbox": [float(value) for value in raw["bbox"]],
        "evidence_ids": [str(raw["evidence_id"])],
    }


def _default_inquiry_row(sequence: int) -> list[str]:
    offset = sequence - 1
    year = 2023 - offset // 336
    month = (offset // 28) % 12 + 1
    day = offset % 28 + 1
    return [
        str(sequence),
        f"{year:04d}.{month:02d}.{day:02d}",
        f"查询机构{sequence}",
        "贷后管理" if sequence % 2 else "贷款审批",
    ]


def _table(raw: Mapping[str, Any]) -> SimpleNamespace:
    start = int(raw["population_start"])
    endpoint = int(raw["population_endpoint"])
    overrides = {
        int(sequence): [str(value or "") for value in values]
        for sequence, values in raw.get("row_overrides", {}).items()
    }
    body = [
        overrides.get(sequence, _default_inquiry_row(sequence))
        for sequence in range(start, endpoint + 1)
    ]
    include_header = raw.get("include_header") is True
    rows = [list(raw["header"]), *body] if include_header else body
    horizontal = [float(value) for value in raw["horizontal_lines"]]
    bands = [
        (float(pair[0]), float(pair[1]))
        for pair in raw["column_bands"]
    ]
    assert len(horizontal) == len(rows) + 1
    cell_bboxes: list[list[list[float] | None]] = [
        [
            [x0, horizontal[row], x1, horizontal[row + 1]]
            for x0, x1 in bands
        ]
        for row in range(len(rows))
    ]
    statuses = [["exact"] * 4 for _row in rows]
    evidence_ids = [
        [
            [f"{raw['table_id']}:r{row}:c{column}"]
            if str(rows[row][column]).strip()
            else []
            for column in range(4)
        ]
        for row in range(len(rows))
    ]
    for row, column in raw.get("derived_empty_cells", []):
        row = int(row)
        column = int(column)
        assert rows[row][column] == ""
        statuses[row][column] = "derived"
        cell_bboxes[row][column] = None
        evidence_ids[row][column] = []
    for row, column in raw.get("exact_empty_cells", []):
        row = int(row)
        column = int(column)
        assert rows[row][column] == ""
        statuses[row][column] = "exact"
        evidence_ids[row][column] = []
    return SimpleNamespace(
        table_id=str(raw["table_id"]),
        bbox=[float(value) for value in raw["bbox"]],
        headers=[],
        rows=[],
        metadata={
            "raw_rows": rows,
            "geometry": {
                "cell_bboxes": cell_bboxes,
                "cell_geometry_status": statuses,
                "cell_evidence_ids": evidence_ids,
                "cell_spans": [],
                "coordinate_system": "pdf_points_top_left",
            },
        },
    )


def _pages_and_evidence(
    spec: Mapping[str, Any],
) -> tuple[list[SimpleNamespace], list[dict[str, Any]]]:
    pages: list[SimpleNamespace] = []
    evidence: list[dict[str, Any]] = []
    for raw_page in spec["pages"]:
        lines = [_line(value) for value in raw_page["lines"]]
        pages.append(
            SimpleNamespace(
                page_number=int(raw_page["logical_page"]),
                source_page_number=int(raw_page["source_page"]),
                width=float(raw_page["width"]),
                height=float(raw_page["height"]),
                tables=[_table(value) for value in raw_page["tables"]],
                texts=[
                    SimpleNamespace(
                        content=value["text"],
                        bbox=list(value["bbox"]),
                        evidence_ids=list(value["evidence_ids"]),
                    )
                    for value in lines
                ],
            )
        )
        evidence.append(
            {
                "page": int(raw_page["logical_page"]),
                "source_page": int(raw_page["source_page"]),
                "page_width": float(raw_page["width"]),
                "page_height": float(raw_page["height"]),
                "lines": deepcopy(lines),
            }
        )
    return pages, evidence


class _ReplayTopology:
    def __init__(self, raw: Mapping[str, Any]) -> None:
        self._geometries = {
            int(logical): LogicalPageGeometry(
                logical_page=int(logical),
                source_page=int(value["source_page"]),
                width=float(value["width"]),
                height=float(value["height"]),
                split_kind=str(value["split_kind"]),
                segment_index=int(value["segment_index"]),
                selected_rotation=int(value["selected_rotation"]),
                split_confidence=0.99,
                source_crop_bbox=tuple(
                    float(item) for item in value["source_crop_bbox"]
                ),
                transform_usable=value["transform_usable"] is True,
            )
            for logical, value in raw["geometries"].items()
        }
        self._audit = {
            "valid": raw["valid"] is True,
            "logical_pages_by_source": deepcopy(
                raw["logical_pages_by_source"]
            ),
            "topology_frozen_before_reocr": (
                raw["topology_frozen_before_reocr"] is True
            ),
            "invalid_source_pages": [],
            "issues": [],
        }

    def geometry(self, logical_page: int) -> LogicalPageGeometry | None:
        return self._geometries.get(int(logical_page))

    def audit(self) -> dict[str, Any]:
        return deepcopy(self._audit)

    def replace_geometry(self, logical_page: int, **changes: Any) -> None:
        self._geometries[logical_page] = replace(
            self._geometries[logical_page],
            **changes,
        )


def _resolution(spec: Mapping[str, Any]) -> dict[str, Any]:
    raw = deepcopy(spec["reading_order_resolution"])
    raw["printed_page_by_logical"] = {
        int(logical): int(printed)
        for logical, printed in raw["printed_page_by_logical"].items()
    }
    return raw


def _entity_context(
    pages: list[SimpleNamespace],
    *,
    mode: str = "separate",
) -> CreditReportEntityContext:
    units: list[CreditReportUnit] = []
    for page in pages:
        for order, table in enumerate(page.tables):
            if mode == "missing_pt_27_0" and table.table_id == "pt_27_0":
                continue
            rows = tuple(
                tuple(str(value or "") for value in row)
                for row in table.metadata["raw_rows"]
            )
            units.append(
                CreditReportUnit(
                    unit_id=f"unit:{table.table_id}",
                    page=page.page_number,
                    order=order,
                    source_index=len(units),
                    kind="table",
                    text="\n".join(" | ".join(row) for row in rows),
                    bbox=tuple(table.bbox),
                    page_width=page.width,
                    page_height=page.height,
                    table_id=table.table_id,
                    rows=rows,
                )
            )
    by_id = {unit.table_id: unit for unit in units}
    institution_ids = tuple(
        table_id
        for table_id in ("pt_26_3", "pt_27_0", "pt_28_0")
        if table_id in by_id
    )
    if mode == "same_institution_entity":
        institution_entities = (
            CreditReportEntity(
                entity_id="entity:lin:institution-chain",
                kind="table",
                unit_ids=tuple(
                    by_id[table_id].unit_id for table_id in institution_ids
                ),
                pages=tuple(by_id[table_id].page for table_id in institution_ids),
                confidence=0.99,
            ),
        )
    elif mode == "unassigned_pt_27_0":
        institution_entities = tuple(
            CreditReportEntity(
                entity_id=f"entity:{table_id}",
                kind="table",
                unit_ids=(by_id[table_id].unit_id,),
                pages=(by_id[table_id].page,),
                confidence=0.99,
            )
            for table_id in institution_ids
            if table_id != "pt_27_0"
        )
    else:
        institution_entities = tuple(
            CreditReportEntity(
                entity_id=f"entity:{table_id}",
                kind="table",
                unit_ids=(by_id[table_id].unit_id,),
                pages=(by_id[table_id].page,),
                confidence=0.99,
            )
            for table_id in institution_ids
        )
    personal = by_id.get("pt_28_1")
    personal_entities = (
        CreditReportEntity(
            entity_id="entity:pt_28_1",
            kind="table",
            unit_ids=(personal.unit_id,),
            pages=(personal.page,),
            confidence=0.99,
        ),
    ) if personal is not None else ()
    assigned = {
        unit_id
        for entity in (*institution_entities, *personal_entities)
        for unit_id in entity.unit_ids
    }
    return CreditReportEntityContext(
        report_family="personal_detail",
        units=tuple(units),
        furniture_unit_ids=(),
        entities=(*institution_entities, *personal_entities),
        decisions=(),
        unassigned_unit_ids=tuple(
            unit.unit_id for unit in units if unit.unit_id not in assigned
        ),
    )


class _ReplayContext:
    tables_continue = PersonalDetailExtractionContext.tables_continue

    def __init__(
        self,
        *,
        pages: list[SimpleNamespace],
        evidence: list[dict[str, Any]],
        topology: _ReplayTopology,
        resolution: dict[str, Any],
        reading_order: dict[int, int],
        entity_context: CreditReportEntityContext,
    ) -> None:
        self.pages = pages
        self.reading_order_by_logical = reading_order
        self.reading_order_resolution = resolution
        self.page_topology = topology
        self.entity_context = entity_context
        self.page_topology_audit = lambda: {
            **topology.audit(),
            "topology_frozen_before_reocr": True,
        }
        self.corrected_evidence_pages = lambda: deepcopy(evidence)


def _page(pages: list[SimpleNamespace], logical: int) -> SimpleNamespace:
    return next(page for page in pages if page.page_number == logical)


def _physical_table(
    pages: list[SimpleNamespace],
    table_id: str,
) -> SimpleNamespace:
    return next(
        table
        for page in pages
        for table in page.tables
        if table.table_id == table_id
    )


def _apply_producer_defect(
    defect: str,
    *,
    pages: list[SimpleNamespace],
    evidence: list[dict[str, Any]],
    topology: _ReplayTopology,
    resolution: dict[str, Any],
    reading_order: dict[int, int],
) -> None:
    page27 = _page(pages, 27)
    evidence27 = next(row for row in evidence if row["page"] == 27)
    seed = _physical_table(pages, "pt_26_3")
    continuation = _physical_table(pages, "pt_27_0")
    if defect == "page_order":
        reading_order[26], reading_order[27] = 2, 1
    elif defect == "printed_map":
        resolution["printed_page_by_logical"][27] = 28
    elif defect == "footer_missing":
        evidence27["lines"].clear()
        page27.texts.clear()
    elif defect == "footer_duplicate":
        evidence27["lines"].append(deepcopy(evidence27["lines"][-1]))
        page27.texts.append(deepcopy(page27.texts[-1]))
    elif defect == "footer_body_geometry":
        evidence27["lines"][-1]["bbox"] = [100.0, 10.0, 200.0, 20.0]
        page27.texts[-1].bbox = [100.0, 10.0, 200.0, 20.0]
    elif defect == "footer_gap":
        evidence27["lines"][-1]["text"] = "第29页,共30页"
        page27.texts[-1].content = "第29页,共30页"
    elif defect == "footer_total":
        evidence27["lines"][-1]["text"] = "第27页,共31页"
        page27.texts[-1].content = "第27页,共31页"
    elif defect == "footer_unsealed":
        evidence27["lines"][-1]["evidence_ids"] = []
        page27.texts[-1].evidence_ids = []
    elif defect == "topology_source":
        topology.replace_geometry(27, source_page=99)
    elif defect == "topology_nonfinite_crop":
        topology.replace_geometry(
            27,
            source_crop_bbox=(0.0, 0.0, math.inf, 595.5),
        )
    elif defect == "seed_header":
        seed.metadata["raw_rows"][0][1] = "查询日"
    elif defect == "foreign_header":
        foreign = {
            "text": "公共信息明细",
            "bbox": [160.0, 8.0, 280.0, 20.0],
            "evidence_ids": ["lin:p27:foreign-heading"],
        }
        evidence27["lines"].insert(0, foreign)
        page27.texts.insert(
            0,
            SimpleNamespace(
                content=foreign["text"],
                bbox=list(foreign["bbox"]),
                evidence_ids=list(foreign["evidence_ids"]),
            ),
        )
    elif defect == "schema_start":
        continuation.metadata["raw_rows"][0][0] = "18"
    elif defect == "schema_inexact_cell":
        continuation.metadata["geometry"]["cell_geometry_status"][2][2] = (
            "derived"
        )
    elif defect == "schema_overlap":
        continuation.metadata["geometry"]["cell_bboxes"][2][2] = list(
            continuation.metadata["geometry"]["cell_bboxes"][2][1]
        )
    elif defect == "schema_replayed_evidence":
        continuation.metadata["geometry"]["cell_evidence_ids"][2][2] = list(
            continuation.metadata["geometry"]["cell_evidence_ids"][1][2]
        )
    elif defect == "schema_extra_table":
        extra = deepcopy(continuation)
        extra.table_id = "pt_27_extra"
        page27.tables.append(extra)
    elif defect == "resolution_unresolved":
        resolution["resolved"] = False
    elif defect == "resolution_non_authoritative":
        resolution["authoritative"] = False
    elif defect == "resolution_identity_fallback":
        resolution["identity_fallback"] = True
    elif defect == "resolution_total_missing":
        resolution.pop("printed_total")
    else:
        raise AssertionError(f"unknown producer defect: {defect}")


def _projection_case(
    *,
    defect: str = "",
    entity_mode: str = "separate",
    tables_continue: Any = _UNSET,
) -> tuple[
    Any,
    list[dict[str, Any]],
    _ReplayTopology,
    dict[str, Any],
    dict[int, int],
    CreditReportEntityContext,
]:
    spec = _spec()
    pages, evidence = _pages_and_evidence(spec)
    topology = _ReplayTopology(spec["topology"])
    resolution = _resolution(spec)
    reading_order = {26: 1, 27: 2, 28: 3}
    if defect:
        _apply_producer_defect(
            defect,
            pages=pages,
            evidence=evidence,
            topology=topology,
            resolution=resolution,
            reading_order=reading_order,
        )
    entities = _entity_context(pages, mode=entity_mode)
    owner = _ReplayContext(
        pages=pages,
        evidence=evidence,
        topology=topology,
        resolution=resolution,
        reading_order=reading_order,
        entity_context=entities,
    )
    if tables_continue is not _UNSET:
        owner.tables_continue = tables_continue
    projection = PBOCCanonicalTemplateAssembler(
        SimpleNamespace(pages=pages),
        topology=topology,
        reading_order_by_logical=reading_order,
        source_evidence_loader=lambda: deepcopy(evidence),
        issue_owner=owner,
    ).build()
    return (
        projection,
        evidence,
        topology,
        resolution,
        reading_order,
        entities,
    )


def _consumer_context(
    projection: Any,
    evidence: list[dict[str, Any]],
    topology: _ReplayTopology,
    resolution: dict[str, Any],
    reading_order: dict[int, int],
    entities: CreditReportEntityContext,
) -> _ReplayContext:
    consumer_evidence = deepcopy(evidence)
    spec = _spec()
    for raw in spec["supplemental_exact_lines"]:
        target = next(
            page
            for page in consumer_evidence
            if page["page"] == int(raw["logical_page"])
        )
        target["canonical_template_id"] = "annotations_and_inquiries"
        target["lines"].append(_line(raw))
    return _ReplayContext(
        pages=list(projection.pages),
        evidence=consumer_evidence,
        topology=topology,
        resolution=deepcopy(resolution),
        reading_order=deepcopy(reading_order),
        entity_context=entities,
    )


def _projected_table(projection: Any, table_id: str) -> tuple[Any, Any]:
    return next(
        (page, table)
        for page in projection.pages
        for table in page.tables
        if table.table_id == table_id
    )


def _has_projected_table(projection: Any, table_id: str) -> bool:
    return any(
        table.table_id == table_id
        for page in projection.pages
        for table in page.tables
    )


def test_lin_replay_fixture_is_a_minimized_copy_of_fresh_audit_geometry() -> None:
    spec = _spec()

    assert spec["provenance"] == {
        "source_pdf_sha256": (
            "a44515a83ae226d19008437ac6a757fa58dabc14d3f1fb5ac9a01c4441cdfdd2"
        ),
        "source_artifact": (
            "artifacts/personal_detail_six_live_iteration_20260826_linfix4/"
            "林岚挺征信.semantic.json"
        ),
        "captured_at": "2026-08-26",
        "scope": "logical pages 26-28 only; no PDF or OCR required",
        "artifact_observation": {
            "owned_tables": ["pt_26_3", "pt_28_1"],
            "withheld_headerless_tables": ["pt_27_0", "pt_28_0"],
            "emitted_inquiries": 12,
            "reported_expected_inquiries": 90,
        },
    }
    assert [page["logical_page"] for page in spec["pages"]] == [26, 27, 28]
    assert [page["source_page"] for page in spec["pages"]] == [13, 14, 14]
    assert {
        table["table_id"]: (
            table["bbox"],
            table["population_start"],
            table["population_endpoint"],
            len(table["horizontal_lines"]) - 1,
        )
        for page in spec["pages"]
        for table in page["tables"]
    } == {
        "pt_26_3": ([50.5, 325.0, 401.0, 561.5], 1, 16, 17),
        "pt_27_0": ([44.5, 35.0, 398.5, 557.0], 17, 54, 38),
        "pt_28_0": ([52.0, 34.5, 402.5, 487.5], 55, 89, 35),
        "pt_28_1": ([53.0, 512.0, 402.5, 540.0], 1, 1, 2),
    }
    assert spec["expected"] == {
        "source_rows": 90,
        "emitted_rows": 87,
        "institution_endpoint": 89,
        "personal_endpoint": 1,
        "omitted_institution_sequences": [4, 66, 67],
        "localized_omission_raw_dates": [
            "2022.1214",
            "2021.09.30",
            "2021.09.03",
        ],
        "generic_inferred_sequences": [27, 46, 87],
        "authority_mode": "schema_carry_only",
    }


def test_lin_production_replay_reaches_87_of_90_with_only_localized_omissions() -> None:
    spec = _spec()
    expected = spec["expected"]
    projection, evidence, topology, resolution, order, entities = (
        _projection_case()
    )
    context = _consumer_context(
        projection,
        evidence,
        topology,
        resolution,
        order,
        entities,
    )

    assert {
        table.table_id
        for page in projection.pages
        for table in page.tables
    } == {"pt_26_3", "pt_27_0", "pt_28_0", "pt_28_1"}
    continuation_metadata = {}
    for table_id in ("pt_27_0", "pt_28_0"):
        page, table = _projected_table(projection, table_id)
        metadata = canonical_inquiry_population_metadata(context, page, table)
        assert metadata is not None
        continuation_metadata[table_id] = metadata
    assert {
        metadata["authority_mode"]
        for metadata in continuation_metadata.values()
    } == {INQUIRY_AUTHORITY_SCHEMA_CARRY_ONLY}

    records = _extract_inquiries(context)
    coverage = _inquiry_source_coverage(context)
    institution_sequences = sorted(
        row["sequence"]
        for row in records
        if row["inquiry_type"] == "institution"
    )
    personal_sequences = sorted(
        row["sequence"]
        for row in records
        if row["inquiry_type"] == "personal"
    )
    assert len(records) == expected["emitted_rows"]
    assert institution_sequences == [
        sequence
        for sequence in range(1, expected["institution_endpoint"] + 1)
        if sequence not in expected["omitted_institution_sequences"]
    ]
    assert personal_sequences == list(
        range(1, expected["personal_endpoint"] + 1)
    )
    assert coverage["expected_row_count"] == expected["source_rows"]
    assert len(coverage["raw_physical_positions"]) == expected["source_rows"]
    assert coverage["sequence_endpoints"] == {
        "institution": expected["institution_endpoint"],
        "personal": expected["personal_endpoint"],
    }

    issues = context._personal_detail_extraction_issues
    assert {
        issue["candidate_value"]["normalized_sequence"]
        for issue in issues
        if issue.get("issue_code")
        == "candidate_b_inquiry_sequence_inferred_from_row_order"
    } == set(expected["generic_inferred_sequences"])
    assert not any(
        issue.get("issue_code")
        == "candidate_b_inquiry_sequence_owner_ordinal_corrected"
        for issue in issues
    )
    emitted_business_rows = {
        (row["inquiry_date"], row["institution"], row["reason"])
        for row in records
        if row["inquiry_type"] == "institution"
    }
    localized_omissions = [
        issue
        for issue in issues
        if issue.get("issue_code")
        == "candidate_b_inquiry_multiple_missing_sequences_unresolved"
        and (
            issue["observed_value"]["row"]["inquiry_date"],
            issue["observed_value"]["row"]["institution"],
            issue["observed_value"]["row"]["reason"],
        )
        not in emitted_business_rows
    ]
    assert {
        issue["observed_value"]["row"]["raw_inquiry_date"]
        for issue in localized_omissions
    } == set(expected["localized_omission_raw_dates"])
    assert all(
        issue["reason_codes"][-1] == "record_not_emitted"
        for issue in localized_omissions
    )


def test_lin_87_of_90_replay_survives_source_and_public_projection() -> None:
    spec = _spec()
    expected = spec["expected"]
    projection, evidence, topology, resolution, order, entities = (
        _projection_case()
    )
    context = _consumer_context(
        projection,
        evidence,
        topology,
        resolution,
        order,
        entities,
    )

    inquiry_records = _extract_inquiries(context)
    source_ledger = _source_completeness_ledger(context)
    source_issues = collect_extraction_issues(context)
    prepared = prepare_personal_detail_source_collections(
        {
            "facts": {
                "personal_detail_source_completeness_ledger": source_ledger,
            },
            "datasets": {
                "inquiry_records": inquiry_records,
                "personal_detail_extraction_issues": source_issues,
            },
        },
        final_dataset_counts={"inquiry_records": len(inquiry_records)},
    )
    public = project_personal_detail_datasets(prepared["datasets"])

    assert source_ledger["inquiry_records"] == expected["source_rows"]
    assert source_ledger["inquiry_sequence_endpoints"] == {
        "institution": expected["institution_endpoint"],
        "personal": expected["personal_endpoint"],
    }
    assert len(source_ledger["inquiry_raw_physical_positions"]) == expected[
        "source_rows"
    ]

    public_rows = [
        row.get("normalized", row)
        for row in public["inquiries"]
        if isinstance(row, dict)
    ]
    assert len(public_rows) == expected["emitted_rows"]
    assert len({row["inquiry_id"] for row in public_rows}) == len(public_rows)
    assert {
        row["sequence"]
        for row in public_rows
        if row["inquiry_type"] == "institution"
    } == {
        sequence
        for sequence in range(1, expected["institution_endpoint"] + 1)
        if sequence not in expected["omitted_institution_sequences"]
    }
    assert {
        row["sequence"]
        for row in public_rows
        if row["inquiry_type"] == "personal"
    } == set(range(1, expected["personal_endpoint"] + 1))
    assert all(
        row["inquiry_id"]
        == f"credit_inquiry:{row['inquiry_type']}:{row['sequence']}"
        for row in public_rows
    )

    public_issues = [
        row.get("normalized", row)
        for row in public["extraction_issues"]
        if isinstance(row, dict)
    ]
    expected_omission_ids = {
        f"credit_inquiry:institution:{sequence}"
        for sequence in expected["omitted_institution_sequences"]
    }
    canonical_omissions = [
        issue
        for issue in public_issues
        if issue.get("target_dataset") == "inquiries"
        and issue.get("issue_code") == "source_inquiry_record_omitted"
    ]
    assert {
        issue["target_record_id"] for issue in canonical_omissions
    } == expected_omission_ids
    assert len(canonical_omissions) == len(expected_omission_ids)

    forbidden_row_omission_codes = {
        "candidate_b_inquiry_boundary_sequence_unresolved",
        "candidate_b_inquiry_sequence_unresolved",
        "source_inquiry_physical_record_omitted",
    }
    assert not any(
        issue.get("target_dataset") == "inquiries"
        and issue.get("issue_code") in forbidden_row_omission_codes
        for issue in public_issues
    )

    localized_omissions = [
        issue
        for issue in public_issues
        if issue.get("target_dataset") == "inquiries"
        and issue.get("issue_code")
        == "candidate_b_inquiry_multiple_missing_sequences_unresolved"
    ]
    assert len(localized_omissions) == len(expected_omission_ids)
    assert len(
        {issue["target_record_id"] for issue in localized_omissions}
    ) == len(localized_omissions)

    omission_issue_ids = {
        issue["extraction_issue_id"] for issue in localized_omissions
    }
    omission_evidence = [
        row.get("normalized", row)
        for row in public["extraction_issue_evidence"]
        if isinstance(row, dict)
        and row.get("normalized", row).get("extraction_issue_id")
        in omission_issue_ids
    ]
    assert {
        row["string_value"]
        for row in omission_evidence
        if row.get("evidence_kind") == "observed"
        and row.get("evidence_path") == "row.raw_inquiry_date"
    } == set(expected["localized_omission_raw_dates"])
    assert not any(
        row.get("evidence_kind") == "candidate"
        for row in omission_evidence
    )

    field_omissions = [
        issue
        for issue in public_issues
        if issue.get("target_dataset") == "inquiries"
        and issue.get("issue_code") == "source_inquiry_field_omitted"
    ]
    assert {
        issue["target_record_id"] for issue in field_omissions
    } == expected_omission_ids
    assert len(field_omissions) == 3 * len(expected_omission_ids)
    assert all(
        {
            issue["field_name"]
            for issue in field_omissions
            if issue["target_record_id"] == omission_id
        }
        == {"inquiry_date", "institution", "reason"}
        for omission_id in expected_omission_ids
    )

    population_gaps = [
        issue
        for issue in public_issues
        if issue.get("target_dataset") == "inquiries"
        and issue.get("issue_code") == "source_sequence_or_count_gap"
    ]
    assert len(population_gaps) == 1

    inquiry_statuses = [
        row.get("normalized", row)
        for row in public["dataset_status"]
        if isinstance(row, dict)
        and row.get("normalized", row).get("dataset_name") == "inquiries"
    ]
    assert len(inquiry_statuses) == 1
    assert {
        key: inquiry_statuses[0][key]
        for key in (
            "presence_status",
            "observed_row_count",
            "expected_row_count",
            "reason",
        )
    } == {
        "presence_status": "partial",
        "observed_row_count": expected["emitted_rows"],
        "expected_row_count": expected["source_rows"],
        "reason": "source_partially_observed",
    }


def test_distinct_physical_entities_and_one_joined_entity_have_output_parity() -> None:
    outputs = []
    proof_kinds = []
    for mode in ("separate", "same_institution_entity"):
        projection, evidence, topology, resolution, order, entities = (
            _projection_case(entity_mode=mode)
        )
        context = _consumer_context(
            projection,
            evidence,
            topology,
            resolution,
            order,
            entities,
        )
        outputs.append(_extract_inquiries(context))
        proof_kinds.append(
            {
                _projected_table(projection, table_id)[1]
                .metadata["canonical_section_owner"]["adjacency_proof"]["kind"]
                for table_id in ("pt_27_0", "pt_28_0")
            }
        )

    assert outputs[0] == outputs[1]
    assert proof_kinds == [
        {"exact_printed_footer_schema_carry_bridge"},
        {"exact_printed_footer_table_edge"},
    ]


@pytest.mark.parametrize(
    "defect",
    (
        "page_order",
        "printed_map",
        "footer_missing",
        "footer_duplicate",
        "footer_body_geometry",
        "footer_gap",
        "footer_total",
        "footer_unsealed",
        "topology_source",
        "topology_nonfinite_crop",
        "seed_header",
        "foreign_header",
        "schema_start",
        "schema_inexact_cell",
        "schema_overlap",
        "schema_replayed_evidence",
        "schema_extra_table",
        "resolution_unresolved",
        "resolution_non_authoritative",
        "resolution_identity_fallback",
        "resolution_total_missing",
    ),
)
def test_lin_producer_mutations_cannot_authorize_first_headerless_table(
    defect: str,
) -> None:
    baseline = _projection_case()[0]
    assert _has_projected_table(baseline, "pt_27_0")

    mutant = _projection_case(defect=defect)[0]
    assert not _has_projected_table(mutant, "pt_27_0")


@pytest.mark.parametrize(
    ("entity_mode", "tables_continue"),
    (
        ("missing_pt_27_0", _UNSET),
        ("unassigned_pt_27_0", _UNSET),
        ("separate", None),
    ),
)
def test_lin_producer_requires_a_decidable_entity_table_relation(
    entity_mode: str,
    tables_continue: Any,
) -> None:
    baseline = _projection_case()[0]
    assert _has_projected_table(baseline, "pt_27_0")

    override = (
        (lambda _left, _right: None)
        if tables_continue is None
        else _UNSET
    )
    mutant = _projection_case(
        entity_mode=entity_mode,
        tables_continue=override,
    )[0]
    assert not _has_projected_table(mutant, "pt_27_0")


def test_lin_producer_contains_tables_continue_exceptions() -> None:
    baseline = _projection_case()[0]
    assert _has_projected_table(baseline, "pt_27_0")

    def broken_relation(_left: str, _right: str) -> bool:
        raise ValueError("adversarial entity relation failure")

    mutant = _projection_case(tables_continue=broken_relation)[0]
    assert not _has_projected_table(mutant, "pt_27_0")


@pytest.mark.parametrize(
    "defect",
    (
        "page_order",
        "printed_map",
        "footer_body_geometry",
        "footer_duplicate",
        "topology_crop",
        "topology_rotation",
        "header_binding",
        "schema_roles",
        "authority_injection",
        "proof_kind",
        "prior_table",
        "source_page",
        "resolution_missing",
        "resolution_non_authoritative",
        "entity_joined",
        "entity_missing",
        "tables_continue_none",
        "tables_continue_exception",
    ),
)
def test_lin_consumer_replays_every_schema_bridge_seal(defect: str) -> None:
    projection, evidence, topology, resolution, order, entities = (
        _projection_case()
    )
    context = _consumer_context(
        projection,
        evidence,
        topology,
        resolution,
        order,
        entities,
    )
    page, table = _projected_table(projection, "pt_27_0")
    assert canonical_inquiry_population_metadata(context, page, table) is not None

    owner = table.metadata["canonical_section_owner"]
    if defect == "page_order":
        context.reading_order_by_logical[26], context.reading_order_by_logical[27] = (
            2,
            1,
        )
    elif defect == "printed_map":
        context.reading_order_resolution["printed_page_by_logical"][27] = 28
    elif defect == "footer_body_geometry":
        page.texts[-1].bbox = [100.0, 10.0, 200.0, 20.0]
    elif defect == "footer_duplicate":
        page.texts.append(deepcopy(page.texts[-1]))
    elif defect == "topology_crop":
        topology.replace_geometry(
            27,
            source_crop_bbox=(0.0, 0.0, 403.0, 595.5),
        )
    elif defect == "topology_rotation":
        topology.replace_geometry(27, selected_rotation=180)
    elif defect == "header_binding":
        owner["header_binding"] = "unsealed_inherited_lattice"
    elif defect == "schema_roles":
        owner["inquiry_role_columns"] = {
            "sequence": 1,
            "inquiry_date": 0,
            "institution": 2,
            "reason": 3,
        }
    elif defect == "authority_injection":
        owner["authority_mode"] = "closed_physical_ordinal"
    elif defect == "proof_kind":
        owner["adjacency_proof"]["kind"] = "exact_printed_footer_table_edge"
    elif defect == "prior_table":
        owner["prior_table_id"] = "pt_unrelated"
    elif defect == "source_page":
        page.source_page_number = 99
    elif defect == "resolution_missing":
        context.reading_order_resolution = None
    elif defect == "resolution_non_authoritative":
        context.reading_order_resolution["authoritative"] = False
    elif defect == "entity_joined":
        context.entity_context = _entity_context(
            context.pages,
            mode="same_institution_entity",
        )
    elif defect == "entity_missing":
        context.entity_context = _entity_context(
            context.pages,
            mode="missing_pt_27_0",
        )
    elif defect == "tables_continue_none":
        context.tables_continue = lambda _left, _right: None
    elif defect == "tables_continue_exception":
        def broken_relation(_left: str, _right: str) -> bool:
            raise ValueError("adversarial entity relation failure")

        context.tables_continue = broken_relation
    else:
        raise AssertionError(defect)

    assert canonical_inquiry_population_metadata(context, page, table) is None
    assert canonical_table_role(context, page, table) is None


@pytest.mark.parametrize(
    "defect",
    (
        "derived_non_sequence_cell",
        "derived_sequence_has_text",
        "unresolved_sequence_geometry",
    ),
)
def test_lin_consumer_allows_only_source_preserving_seed_sequence_omissions(
    defect: str,
) -> None:
    projection, evidence, topology, resolution, order, entities = (
        _projection_case()
    )
    context = _consumer_context(
        projection,
        evidence,
        topology,
        resolution,
        order,
        entities,
    )
    seed = _projected_table(projection, "pt_26_3")[1]
    page, table = _projected_table(projection, "pt_27_0")
    geometry = seed.metadata["geometry"]
    assert canonical_inquiry_population_metadata(context, page, table) is not None

    if defect == "derived_non_sequence_cell":
        geometry["cell_geometry_status"][4][1] = "derived"
        geometry["cell_bboxes"][4][1] = None
        geometry["cell_evidence_ids"][4][1] = []
    elif defect == "derived_sequence_has_text":
        seed.metadata["raw_rows"][4][0] = "4"
    elif defect == "unresolved_sequence_geometry":
        geometry["cell_geometry_status"][4][0] = "unresolved"
    else:
        raise AssertionError(defect)

    assert canonical_inquiry_population_metadata(context, page, table) is None
    assert canonical_table_role(context, page, table) is None


def _strategy_audit(projection: Any) -> dict[str, Any]:
    projection_audit = projection.audit()
    inquiry_registrations = [
        deepcopy(row) for row in projection_audit["registrations"]
    ]
    inquiry_groups = [
        deepcopy(row) for row in projection_audit["fragment_groups"]
    ]
    header_registration = {
        "logical_page": 1,
        "source_page": 1,
        "status": "registered",
        "template_id": REPORT_HEADER_SECTION,
        "basis": "production_replay_required_header",
        "affected_source_datasets": sorted(
            SECTION_TO_CANONICAL_DATASETS[REPORT_HEADER_SECTION]
        ),
        "printed_page": 1,
        "printed_total": 30,
    }
    return {
        "corrected_evidence_conservation": {
            "valid": True,
            "conserved_logical_pages": [1, 26, 27, 28],
        },
        "canonical_subset_conservation": {"valid": True},
        "unresolved_pages": list(projection_audit["unresolved_pages"]),
        "registrations": [header_registration, *inquiry_registrations],
        "fragment_groups": [
            {
                "template_id": REPORT_HEADER_SECTION,
                "fragment_logical_pages": [1],
                "canonical_page": 1,
                "coverage_status": "full",
                "coverage_ratio": 1.0,
            },
            *inquiry_groups,
        ],
    }


def test_lin_replay_selects_the_deployed_inquiry_stages_without_fallback() -> None:
    projection = _projection_case()[0]
    census, plan = plan_candidate_b_initial_extraction(
        _strategy_audit(projection)
    )

    assert census.complete is True
    assert census.census.state_for(INQUIRY_SECTION) is SectionState.OBSERVED
    assert plan.mode is MaterializationMode.LAZY
    assert {"inquiries", "notes"}.issubset(plan.ordered_stage_names)
    assert "inquiries" not in plan.skipped_stage_names
    assert "notes" not in plan.skipped_stage_names
    assert stage_names_for_datasets({"inquiry_records"}) == ("inquiries",)
    inquiry_stage = CANDIDATE_B_STAGE_REGISTRY.stage("inquiries")
    assert inquiry_stage.section == INQUIRY_SECTION
    assert inquiry_stage.optional is True
    assert inquiry_stage.output_names == ("inquiry_records",)


@pytest.mark.parametrize(
    "defect",
    (
        "resolution_non_authoritative",
        "footer_gap",
        "schema_start",
    ),
)
def test_lin_replay_strategy_census_falls_back_when_inquiry_owner_is_unresolved(
    defect: str,
) -> None:
    baseline = _projection_case()[0]
    baseline_census, baseline_plan = plan_candidate_b_initial_extraction(
        _strategy_audit(baseline)
    )
    assert baseline_census.complete is True
    assert baseline_plan.mode is MaterializationMode.LAZY

    projection = _projection_case(defect=defect)[0]
    census, plan = plan_candidate_b_initial_extraction(
        _strategy_audit(projection)
    )
    assert census.complete is False
    assert plan.mode is MaterializationMode.EAGER_FALLBACK
    assert plan.fallback_reason
