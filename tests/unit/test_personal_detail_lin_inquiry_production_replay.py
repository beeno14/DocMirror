from __future__ import annotations

import json
import math
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

import pytest

from docmirror.models.mirror.vnext import EvidenceAtom, EvidenceStore
from docmirror.plugins.credit_report.personal_detail_scanned.business_repair import (
    BusinessUncertaintyRepairCoordinator,
)
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
    _sealed_raw_inquiry_population_census,
    _source_completeness_ledger,
)
from docmirror.plugins.credit_report.personal_detail_scanned.ocr_correction import (
    PersonalDetailOCRCorrectionOverlay,
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


def _fixture_institution_rows(
    spec: Mapping[str, Any],
) -> dict[int, dict[str, Any]]:
    """Index each institution source row by its sealed population position."""

    indexed: dict[int, dict[str, Any]] = {}
    for raw_page in spec["pages"]:
        for raw_table in raw_page["tables"]:
            if raw_table["table_id"] == "pt_28_1":
                continue
            start = int(raw_table["population_start"])
            endpoint = int(raw_table["population_endpoint"])
            row_offset = 1 if raw_table.get("include_header") is True else 0
            for sequence in range(start, endpoint + 1):
                indexed[sequence] = {
                    "logical_page": int(raw_page["logical_page"]),
                    "table_id": str(raw_table["table_id"]),
                    "row": sequence - start + row_offset,
                    "values": [
                        str(value or "")
                        for value in raw_table["row_overrides"][str(sequence)]
                    ],
                }
    return indexed


def _fixture_sequence_for_issue(
    issue: Mapping[str, Any],
    fixture_rows: Mapping[int, Mapping[str, Any]],
) -> int | None:
    positions = {
        (
            int(row["logical_page"]),
            str(row["table_id"]),
            int(row["row"]),
        ): sequence
        for sequence, row in fixture_rows.items()
    }
    matches = {
        positions.get(
            (
                int(ref.get("logical_page") or 0),
                str(ref.get("table_id") or ""),
                int(ref.get("row") if ref.get("row") is not None else -1),
            )
        )
        for ref in issue.get("source_refs") or ()
        if isinstance(ref, Mapping)
    }
    matches.discard(None)
    assert len(matches) <= 1
    return next(iter(matches), None)


def _line(raw: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "text": str(raw["text"]),
        "bbox": [float(value) for value in raw["bbox"]],
        "evidence_ids": [str(raw["evidence_id"])],
    }


def _table(
    raw: Mapping[str, Any],
    *,
    atom_sink: list[EvidenceAtom] | None = None,
) -> SimpleNamespace:
    start = int(raw["population_start"])
    endpoint = int(raw["population_endpoint"])
    overrides = {
        int(sequence): [str(value or "") for value in values]
        for sequence, values in raw.get("row_overrides", {}).items()
    }
    expected_sequences = set(range(start, endpoint + 1))
    assert set(overrides) == expected_sequences
    body = [
        overrides[sequence]
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
    evidence_ids: list[list[list[str]]] = []
    token_ids: list[list[list[str]]] = []
    for row in range(len(rows)):
        evidence_line: list[list[str]] = []
        token_line: list[list[str]] = []
        for column in range(4):
            text = str(rows[row][column]).strip()
            ids = (
                [f"{raw['table_id']}:r{row}:c{column}"]
                if text
                else []
            )
            evidence_line.append(list(ids))
            token_line.append(list(ids))
            if atom_sink is not None and ids:
                bbox = cell_bboxes[row][column]
                assert bbox is not None
                atom_sink.append(
                    EvidenceAtom(
                        id=ids[0],
                        text=text,
                        bbox=[
                            bbox[0] + 1.0,
                            bbox[1] + 1.0,
                            bbox[2] - 1.0,
                            bbox[3] - 1.0,
                        ],
                    )
                )
        evidence_ids.append(evidence_line)
        token_ids.append(token_line)
    for row, column in raw.get("derived_empty_cells", []):
        row = int(row)
        column = int(column)
        assert rows[row][column] == ""
        statuses[row][column] = "derived"
        cell_bboxes[row][column] = None
        evidence_ids[row][column] = []
        token_ids[row][column] = []
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
                "row_bands": [
                    {
                        "index": row,
                        "y0": horizontal[row],
                        "y1": horizontal[row + 1],
                    }
                    for row in range(len(rows))
                ],
                "col_bands": [
                    {"index": column, "x0": x0, "x1": x1}
                    for column, (x0, x1) in enumerate(bands)
                ],
                "cell_bboxes": cell_bboxes,
                "cell_geometry_status": statuses,
                "cell_evidence_ids": evidence_ids,
                "cell_token_ids": token_ids,
                "cell_spans": [],
                "coordinate_system": "pdf_points_top_left",
            },
        },
    )


def _pages_and_evidence(
    spec: Mapping[str, Any],
    *,
    atom_sink: list[EvidenceAtom] | None = None,
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
                tables=[
                    _table(value, atom_sink=atom_sink)
                    for value in raw_page["tables"]
                ],
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


def _frozen_raw_inquiry_plane(
    spec: Mapping[str, Any],
) -> tuple[dict[int, SimpleNamespace], list[EvidenceAtom], dict[int, int]]:
    """Rebuild the sealed source plane independently of canonical mutation."""

    atoms: list[EvidenceAtom] = []
    pages, _evidence = _pages_and_evidence(spec, atom_sink=atoms)
    raw_envelope = spec["raw_plane_envelope"]
    start = raw_envelope["section_start"]
    start_page = _page(pages, int(start["logical_page"]))
    start_page.texts.insert(
        0,
        SimpleNamespace(
            content=str(start["text"]),
            bbox=[float(value) for value in start["bbox"]],
            evidence_ids=[str(start["evidence_id"])],
        ),
    )

    boundary = raw_envelope["section_boundary"]
    pages.append(
        SimpleNamespace(
            page_number=int(boundary["logical_page"]),
            source_page_number=int(boundary["source_page"]),
            width=float(boundary["width"]),
            height=float(boundary["height"]),
            tables=[],
            texts=[
                SimpleNamespace(
                    content=str(boundary["text"]),
                    bbox=[float(value) for value in boundary["bbox"]],
                    evidence_ids=[str(boundary["evidence_id"])],
                )
            ],
        )
    )
    ordered = sorted(pages, key=lambda page: int(page.page_number))
    return (
        {int(page.page_number): page for page in ordered},
        atoms,
        {
            int(page.page_number): position
            for position, page in enumerate(ordered, start=1)
        },
    )


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
        self._logicals_by_source = {
            int(source): tuple(int(logical) for logical in logical_pages)
            for source, logical_pages in raw["logical_pages_by_source"].items()
        }
        self._invalid_sources = {
            int(source) for source in raw.get("invalid_source_pages", ())
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

    def ordered_pair(
        self,
        logical_pages: Iterable[int],
    ) -> tuple[int, int] | None:
        requested = {int(page) for page in logical_pages if int(page) > 0}
        if len(requested) != 2:
            return None
        sources = {
            geometry.source_page
            for logical in requested
            if (geometry := self._geometries.get(logical)) is not None
        }
        if len(sources) != 1:
            return None
        source = next(iter(sources))
        ordered = self._logicals_by_source.get(source, ())
        if (
            source in self._invalid_sources
            or len(ordered) != 2
            or set(ordered) != requested
        ):
            return None
        return (ordered[0], ordered[1])

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
    candidate_b_planned_field_repair = (
        PersonalDetailExtractionContext.candidate_b_planned_field_repair
    )
    candidate_b_field_repair = PersonalDetailExtractionContext.candidate_b_field_repair

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
        self._business_repair_plan = None
        self._business_repair_active = False
        self._ocr_correction_overlay = PersonalDetailOCRCorrectionOverlay(self)


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
    frozen_pages, raw_atoms, frozen_order = _frozen_raw_inquiry_plane(spec)
    assert {
        logical: frozen_order[logical] for logical in reading_order
    } == reading_order
    consumer_resolution = deepcopy(resolution)
    consumer_resolution["printed_page_by_logical"][29] = 29
    context = _ReplayContext(
        pages=list(projection.pages),
        evidence=consumer_evidence,
        topology=topology,
        resolution=consumer_resolution,
        reading_order=frozen_order,
        entity_context=entities,
    )
    context._frozen_logical_pages = frozen_pages
    context.evidence_plane = SimpleNamespace(
        evidence=EvidenceStore(text_atoms=raw_atoms)
    )
    return context


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
        "scope": (
            "logical pages 26-28 inquiry tables plus page 29 section "
            "boundary; no PDF or OCR required"
        ),
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
        "emitted_rows": 73,
        "target_repaired_rows": 87,
        "institution_endpoint": 89,
        "personal_endpoint": 1,
        "omitted_institution_sequences": [
            3,
            4,
            8,
            9,
            25,
            26,
            40,
            41,
            44,
            59,
            65,
            66,
            67,
            79,
            81,
            83,
            84,
        ],
        "deferred_repair_sequences": [
            3,
            8,
            9,
            25,
            26,
            40,
            41,
            44,
            59,
            65,
            79,
            81,
            83,
            84,
        ],
        "target_final_omitted_institution_sequences": [4, 66, 67],
        "target_final_omission_raw_dates": [
            "2022.1214",
            "2021.09.30",
            "2021.09.03",
        ],
        "generic_inferred_sequences": [27, 46, 87],
        "authority_mode": "schema_carry_only",
    }
    expected = spec["expected"]
    assert expected["emitted_rows"] == expected["source_rows"] - len(
        expected["omitted_institution_sequences"]
    )
    assert expected["target_repaired_rows"] == expected["source_rows"] - len(
        expected["target_final_omitted_institution_sequences"]
    )
    assert set(expected["deferred_repair_sequences"]) == set(
        expected["omitted_institution_sequences"]
    ) - set(expected["target_final_omitted_institution_sequences"])


def test_lin_strict_discovery_localizes_repairs_without_guessing() -> None:
    spec = _spec()
    expected = spec["expected"]
    fixture_rows = _fixture_institution_rows(spec)
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

    raw_census = _sealed_raw_inquiry_population_census(context)
    assert raw_census is not None
    frozen_census = deepcopy(raw_census)
    assert len(raw_census["raw_physical_positions"]) == expected["source_rows"]
    assert len(
        {
            row["source_physical_row_id"]
            for row in raw_census["raw_physical_positions"]
        }
    ) == expected["source_rows"]
    assert all(
        len(row["source_refs"]) == 1
        and row["source_refs"][0].get("evidence_ids")
        for row in raw_census["raw_physical_positions"]
    )
    assert Counter(
        (
            row["source_refs"][0]["logical_page"],
            row["source_refs"][0]["table_id"],
        )
        for row in raw_census["raw_physical_positions"]
    ) == {
        (26, "pt_26_3"): 16,
        (27, "pt_27_0"): 38,
        (28, "pt_28_0"): 35,
        (28, "pt_28_1"): 1,
    }
    assert context._frozen_logical_pages[26] is not _page(context.pages, 26)

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
    assert coverage["raw_physical_positions"] == raw_census[
        "raw_physical_positions"
    ]
    assert _sealed_raw_inquiry_population_census(context) == frozen_census

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
    missing_sequences = set(expected["omitted_institution_sequences"])
    deferred_sequences = set(expected["deferred_repair_sequences"])
    final_omissions = set(
        expected["target_final_omitted_institution_sequences"]
    )
    assert len(missing_sequences) == 17
    assert len(deferred_sequences) == 14
    assert len(final_omissions) == 3
    assert deferred_sequences.isdisjoint(final_omissions)
    assert deferred_sequences | final_omissions == missing_sequences

    localized_by_sequence = {
        sequence: issue
        for issue in issues
        if issue.get("issue_code")
        in {
            "candidate_b_inquiry_row_cells_unresolved",
            "candidate_b_inquiry_multiple_missing_sequences_unresolved",
        }
        and (
            sequence := _fixture_sequence_for_issue(issue, fixture_rows)
        ) in missing_sequences
    }
    assert set(localized_by_sequence) == missing_sequences

    deferred_date_repairs = deferred_sequences - {8, 9}
    assert {
        sequence
        for sequence, issue in localized_by_sequence.items()
        if issue["issue_code"] == "candidate_b_inquiry_row_cells_unresolved"
    } == deferred_date_repairs
    assert all(
        localized_by_sequence[sequence]["field_name"] == "inquiry_date"
        and localized_by_sequence[sequence]["observed_value"]["row"]
        == fixture_rows[sequence]["values"]
        and localized_by_sequence[sequence].get("candidate_value")
        == {"missing_fields": ["inquiry_date"]}
        and localized_by_sequence[sequence]["reason_codes"][-1]
        == "record_not_invented"
        and len(
            [
                ref
                for ref in localized_by_sequence[sequence]["source_refs"]
                if ref.get("field_name") == "inquiry_date"
                and ref.get("geometry_scope") == "cell"
                and ref.get("evidence_ids")
            ]
        )
        == 1
        for sequence in deferred_date_repairs
    )

    unresolved_ordinal_sequences = {4, 8, 9, 66, 67}
    assert unresolved_ordinal_sequences == ({8, 9} | final_omissions)
    assert {
        sequence
        for sequence, issue in localized_by_sequence.items()
        if issue["issue_code"]
        == "candidate_b_inquiry_multiple_missing_sequences_unresolved"
    } == unresolved_ordinal_sequences
    for sequence in unresolved_ordinal_sequences:
        issue = localized_by_sequence[sequence]
        observed_row = issue["observed_value"]["row"]
        assert issue["field_name"] == "sequence"
        assert observed_row["raw_sequence"] == fixture_rows[sequence]["values"][0]
        assert observed_row["raw_inquiry_date"] == fixture_rows[sequence][
            "values"
        ][1]
        assert "candidate_value" not in issue
        assert issue["reason_codes"][-1] == "record_not_emitted"
        assert len(issue["source_refs"]) == 2
        assert any(
            ref.get("geometry_scope") == "row" and ref.get("evidence_ids")
            for ref in issue["source_refs"]
        )
        sequence_refs = [
            ref
            for ref in issue["source_refs"]
            if ref.get("field_name") == "sequence"
        ]
        assert len(sequence_refs) == 1
        assert sequence_refs[0]["geometry_scope"] in {
            "cell",
            "canonical_field_slot",
        }
        assert sequence_refs[0]["binding"] == "canonical_header_column"
        assert sequence_refs[0]["column"] == 0
    for sequence in {8, 9}:
        exact_sequence_ref = next(
            ref
            for ref in localized_by_sequence[sequence]["source_refs"]
            if ref.get("field_name") == "sequence"
        )
        assert exact_sequence_ref["geometry_scope"] == "cell"
        assert exact_sequence_ref.get("bbox")
        assert exact_sequence_ref.get("evidence_ids")

    assert {
        fixture_rows[sequence]["values"][1] for sequence in final_omissions
    } == set(expected["target_final_omission_raw_dates"])


def test_lin_field_repair_policy_reaches_target_without_replacing_pages() -> None:
    spec = _spec()
    expected = spec["expected"]
    fixture_rows = _fixture_institution_rows(spec)
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
    discovery_records = _extract_inquiries(context)
    discovery_issues = deepcopy(context._personal_detail_extraction_issues)
    coordinator = BusinessUncertaintyRepairCoordinator(context)
    plan = coordinator.plan(
        {"inquiry_records": deepcopy(discovery_records)},
        canonical_audit={"unresolved_pages": []},
        extraction_issues=discovery_issues,
    )
    by_position = {
        (
            int(row["logical_page"]),
            str(row["table_id"]),
            int(row["row"]),
        ): sequence
        for sequence, row in fixture_rows.items()
    }
    clean_reocr_dates = {
        3: "2023.01.03",
        59: "2021.11.26",
    }
    assert {
        sequence: fixture_rows[sequence]["values"][1]
        for sequence in clean_reocr_dates
    } == {
        3: "2023.01.03 20",
        59: "2021.11.26 22",
    }
    ocr_calls: list[tuple[set[int], str]] = []

    def page_ocr_loader(pages: set[int], *, reason: str) -> list[dict[str, Any]]:
        ocr_calls.append((set(pages), reason))
        acquired: list[dict[str, Any]] = []
        for logical_page in sorted(pages):
            repairs = [
                repair
                for repair in plan.field_repairs
                if repair.mode == "context_rich_reocr"
                and any(
                    int(ref.get("logical_page") or 0) == logical_page
                    for ref in repair.source_refs
                )
            ]
            assert repairs
            source_page = int(repairs[0].source_refs[0]["source_page"])
            page_key = f"lin-inquiry-{logical_page}"
            lines: list[dict[str, Any]] = [
                {
                    "text": "四查询记录",
                    "content": "四查询记录",
                    "confidence": 0.99,
                    "bbox": [1.0, 1.0, 20.0, 10.0],
                    "evidence_ids": [
                        f"personal_detail_page_reocr:{page_key}:w0"
                    ],
                    "source": "personal_detail_page_reocr_once",
                }
            ]
            for repair in repairs:
                ref = repair.source_refs[0]
                position = (
                    int(ref.get("logical_page") or 0),
                    str(ref.get("table_id") or ""),
                    int(ref.get("row") if ref.get("row") is not None else -1),
                )
                sequence = by_position.get(position)
                if repair.field_name != "inquiry_date" or sequence not in clean_reocr_dates:
                    continue
                word_index = len(lines)
                lines.append(
                    {
                        "text": clean_reocr_dates[sequence],
                        "content": clean_reocr_dates[sequence],
                        "confidence": 0.99,
                        "bbox": list(ref["bbox"]),
                        "evidence_ids": [
                            f"personal_detail_page_reocr:{page_key}:w{word_index}"
                        ],
                        "source": "personal_detail_page_reocr_once",
                    }
                )
            page: dict[str, Any] = {
                "page": logical_page,
                "logical_page": logical_page,
                "source_page": source_page,
                "page_key": page_key,
                "lines": lines,
            }
            coordinate_system = str(
                repairs[0].source_refs[0].get("coordinate_system") or ""
            )
            if coordinate_system:
                page["coordinate_system"] = coordinate_system
            acquired.append(page)
        return acquired

    coordinator.resolve_page_evidence(
        plan,
        source_pages=evidence,
        page_ocr_loader=page_ocr_loader,
    )
    context._business_repair_plan = plan
    context._business_repair_active = True
    context._personal_detail_extraction_issues = []
    context._ocr_correction_overlay = PersonalDetailOCRCorrectionOverlay(context)
    context._ocr_correction_overlay.install_business_repair_evidence(
        plan.page_evidence.values(),
        affected_pages=plan.affected_pages,
        allowed_target_refs=(
            {**dict(ref), "field_name": repair.field_name}
            for repair in plan.field_repairs
            for ref in repair.source_refs
        ),
    )

    repaired_records = _extract_inquiries(context)
    repaired_by_sequence = {
        row["sequence"]: row
        for row in repaired_records
        if row["inquiry_type"] == "institution"
    }
    discovery_by_key = {
        (row["inquiry_type"], row["sequence"]): row
        for row in discovery_records
    }
    repaired_by_key = {
        (row["inquiry_type"], row["sequence"]): row
        for row in repaired_records
    }

    assert len(repaired_records) == expected["target_repaired_rows"]
    assert set(range(1, expected["institution_endpoint"] + 1)).difference(
        repaired_by_sequence
    ) == set(expected["target_final_omitted_institution_sequences"])
    assert plan.reconstruction_evidence == {}
    expected_ocr_pages = {
        int(ref["logical_page"])
        for repair in plan.field_repairs
        if repair.mode == "context_rich_reocr"
        for ref in repair.source_refs
    }
    assert {page for pages, _reason in ocr_calls for page in pages} == expected_ocr_pages
    assert all(
        reason == "business_field_context_rich_reocr_required"
        for _pages, reason in ocr_calls
    )
    assert all(
        decision["page_reconstruction"] is False
        for decision in plan.page_decisions
        if decision["ocr_invocations"] == 1
    )
    for key, discovery in discovery_by_key.items():
        repaired = repaired_by_key[key]
        assert {
            field_name: repaired[field_name]
            for field_name in ("inquiry_date", "institution", "reason")
        } == {
            field_name: discovery[field_name]
            for field_name in ("inquiry_date", "institution", "reason")
        }

    deferred_sequences = set(expected["deferred_repair_sequences"])
    date_repair_sequences = deferred_sequences.difference({8, 9})
    assert date_repair_sequences.issubset(repaired_by_sequence)
    assert {8, 9}.issubset(repaired_by_sequence)
    assert all(
        repaired_by_sequence[sequence]["canonical_raw"]["inquiry_date"]
        == fixture_rows[sequence]["values"][1]
        for sequence in date_repair_sequences
    )
    decisions = context._ocr_correction_overlay.audit()["decisions"]
    assert sum(
        decision["method"] == "schema_bound_deterministic_field_repair"
        for decision in decisions
    ) == len(date_repair_sequences.difference(clean_reocr_dates))
    assert sum(
        decision["method"] == "schema_bound_page_evidence_reparse"
        for decision in decisions
    ) == len(clean_reocr_dates)


def test_lin_strict_discovery_survives_source_and_public_projection() -> None:
    spec = _spec()
    expected = spec["expected"]
    fixture_rows = _fixture_institution_rows(spec)
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
    frozen_source_ledger = deepcopy(source_ledger)
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
    assert len(
        {
            row["source_physical_row_id"]
            for row in source_ledger["inquiry_raw_physical_positions"]
        }
    ) == expected["source_rows"]
    assert all(
        len(row["source_refs"]) == 1
        and row["source_refs"][0].get("evidence_ids")
        for row in source_ledger["inquiry_raw_physical_positions"]
    )
    assert source_ledger == frozen_source_ledger

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
        isinstance(row.get("inquiry_id"), str)
        and row["inquiry_id"].startswith("credit_inquiry:")
        and len(row["inquiry_id"]) > len("credit_inquiry:")
        for row in public_rows
    )

    public_issues = [
        row.get("normalized", row)
        for row in public["extraction_issues"]
        if isinstance(row, dict)
    ]
    missing_sequences = set(expected["omitted_institution_sequences"])
    anonymous_missing_sequences = {4, 8, 9, 66, 67}
    exact_ordinal_missing_sequences = (
        missing_sequences - anonymous_missing_sequences
    )
    expected_omission_ids = {
        f"credit_inquiry:institution:{sequence}"
        for sequence in exact_ordinal_missing_sequences
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
    assert all(
        len(issue["source_refs"]) == 1
        and issue["source_refs"][0].get("evidence_ids")
        and _fixture_sequence_for_issue(issue, fixture_rows)
        in exact_ordinal_missing_sequences
        for issue in canonical_omissions
    )

    forbidden_row_omission_codes = {
        "candidate_b_inquiry_boundary_sequence_unresolved",
        "candidate_b_inquiry_sequence_unresolved",
    }
    assert not any(
        issue.get("target_dataset") == "inquiries"
        and issue.get("issue_code") in forbidden_row_omission_codes
        for issue in public_issues
    )

    physical_omissions = [
        issue
        for issue in public_issues
        if issue.get("target_dataset") == "inquiries"
        and issue.get("issue_code")
        == "source_inquiry_physical_record_omitted"
    ]
    assert {
        _fixture_sequence_for_issue(issue, fixture_rows)
        for issue in physical_omissions
    } == {4, 5, 8, 9, 66, 67}
    assert all(
        len(issue["source_refs"]) == 1
        and issue["source_refs"][0].get("evidence_ids")
        for issue in physical_omissions
    )
    assert 5 not in missing_sequences
    assert any(
        row["inquiry_type"] == "institution" and row["sequence"] == 5
        for row in public_rows
    )

    localized_ordinal_issues = [
        issue
        for issue in public_issues
        if issue.get("target_dataset") == "inquiries"
        and issue.get("issue_code")
        == "candidate_b_inquiry_multiple_missing_sequences_unresolved"
    ]
    assert {
        _fixture_sequence_for_issue(issue, fixture_rows)
        for issue in localized_ordinal_issues
    } == {4, 5, 8, 9, 66, 67}
    localized_omissions = [
        issue
        for issue in localized_ordinal_issues
        if _fixture_sequence_for_issue(issue, fixture_rows)
        in anonymous_missing_sequences
    ]
    assert len(localized_omissions) == len(anonymous_missing_sequences)
    assert len(
        {issue["target_record_id"] for issue in localized_omissions}
    ) == len(localized_omissions)
    assert all(
        len(issue["source_refs"]) == 2
        and any(
            ref.get("geometry_scope") == "row" and ref.get("evidence_ids")
            for ref in issue["source_refs"]
        )
        and len(
            [
                ref
                for ref in issue["source_refs"]
                if ref.get("field_name") == "sequence"
                and ref.get("geometry_scope")
                in {"cell", "canonical_field_slot"}
                and ref.get("binding") == "canonical_header_column"
                and ref.get("column") == 0
            ]
        )
        == 1
        for issue in localized_omissions
    )

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
    } == {
        fixture_rows[sequence]["values"][1]
        for sequence in anonymous_missing_sequences
    }
    assert not any(
        row.get("evidence_kind") == "candidate"
        for row in omission_evidence
    )

    field_omissions = [
        issue
        for issue in public_issues
        if issue.get("target_dataset") == "inquiries"
        and issue.get("issue_code") == "source_inquiry_field_omitted"
        and issue.get("target_record_id") in expected_omission_ids
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
    assert all(
        len(issue["source_refs"]) == 1
        and len(issue["source_refs"][0].get("evidence_ids") or ()) == 1
        and issue["source_refs"][0].get("field_name")
        == issue["field_name"]
        and _fixture_sequence_for_issue(issue, fixture_rows)
        in exact_ordinal_missing_sequences
        for issue in field_omissions
    )

    localized_issue_ids = {
        issue["extraction_issue_id"]
        for issue in (
            *canonical_omissions,
            *field_omissions,
            *localized_omissions,
        )
    }
    localized_evidence = [
        row.get("normalized", row)
        for row in public["extraction_issue_evidence"]
        if isinstance(row, dict)
        and row.get("normalized", row).get("extraction_issue_id")
        in localized_issue_ids
    ]
    assert not any(
        row.get("evidence_kind") == "candidate"
        and row.get("evidence_path")
        in {"inquiry_date", "institution", "reason"}
        for row in localized_evidence
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
    if defect == "foreign_header":
        assert _has_projected_table(mutant, "pt_27_0")
        _page_owner, table = _projected_table(mutant, "pt_27_0")
        owner = table.metadata.get("canonical_section_owner")
        assert not isinstance(owner, Mapping) or owner.get("template_id") != (
            "annotations_and_inquiries"
        )
        return
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
