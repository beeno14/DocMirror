from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from docmirror.models.mirror.vnext import EvidenceAtom, EvidenceStore
from docmirror.plugins.credit_report.community_plugin import (
    _compact_personal_detail_public_projection,
)
from docmirror.plugins.credit_report.personal_detail_scanned import native_extraction
from docmirror.plugins.credit_report.personal_detail_scanned.schema import (
    project_personal_detail_datasets,
)
from docmirror.plugins.credit_report.personal_detail_scanned.source_projection import (
    prepare_personal_detail_source_collections,
)

_SLOTS = {
    "sequence": 0,
    "inquiry_date": 1,
    "institution": 2,
    "reason": 3,
}


def _physical_inquiry_context(
    *,
    band_count: int = 3,
    defect: str | None = None,
    exact_sequences: bool = False,
) -> SimpleNamespace:
    top = 40.0
    bottom = top + band_count * 11.0 + 4.0
    column_edges = (40.0, 100.0, 180.0, 320.0, 400.0)
    token_specs: dict[int, list[tuple[str, list[float], str]]] = {
        column: [] for column in range(4)
    }
    for index in range(band_count):
        y0 = top + 2.0 + index * 11.0
        y1 = y0 + 7.0
        token_specs[0].append(
            (
                str(index + 1) if exact_sequences else "不可判定",
                [50.0, y0, 82.0, y1],
                f"band:{index}:sequence:0",
            )
        )
        token_specs[1].append(
            (
                f"2024.01.{index + 1:02d}",
                [112.0, y0 + 1.0, 154.0, y1],
                f"band:{index}:date:0",
            )
        )
        # Vary atom counts independently of the number of printed bands.
        token_specs[2].append(
            ("示例", [190.0, y0, 218.0, y1], f"band:{index}:institution:0")
        )
        for fragment in range(index % 3 + 1):
            token_specs[2].append(
                (
                    f"机构{fragment}",
                    [220.0 + fragment * 25.0, y0, 243.0 + fragment * 25.0, y1],
                    f"band:{index}:institution:{fragment + 1}",
                )
            )
        token_specs[3].extend(
            [
                ("贷后", [330.0, y0, 348.0, y1], f"band:{index}:reason:0"),
                ("管理", [349.0, y0, 370.0, y1], f"band:{index}:reason:1"),
            ]
        )

    if defect == "cross_band_token":
        token_specs[2][0] = (
            token_specs[2][0][0],
            [190.0, top + 7.0, 218.0, top + 15.0],
            token_specs[2][0][2],
        )
    elif defect == "overlapping_dates":
        text, _bbox, token_id = token_specs[1][1]
        token_specs[1][1] = (text, [112.0, top + 7.0, 154.0, top + 15.0], token_id)
    elif defect == "unknown_reason":
        _text, bbox, token_id = token_specs[3][0]
        token_specs[3][0] = ("未知", bbox, token_id)
    elif defect == "duplicate_evidence_owner":
        text, bbox, _token_id = token_specs[3][0]
        token_specs[3][0] = (text, bbox, token_specs[2][0][2])

    cell_bboxes = [
        [column_edges[column], top, column_edges[column + 1], bottom]
        for column in range(4)
    ]
    cell_token_ids = [
        [token_id for _text, _bbox, token_id in token_specs[column]]
        for column in range(4)
    ]
    geometry = {
        "row_bands": [{"index": 0, "y0": top, "y1": bottom}],
        "col_bands": [
            {
                "index": column,
                "x0": column_edges[column],
                "x1": column_edges[column + 1],
            }
            for column in range(4)
        ],
        "cell_bboxes": [cell_bboxes],
        "cell_geometry_status": [["exact"] * 4],
        "cell_evidence_ids": [[list(ids) for ids in cell_token_ids]],
        "cell_token_ids": [[list(ids) for ids in cell_token_ids]],
        "cell_spans": [],
    }
    table = SimpleNamespace(
        table_id="physical-inquiry-table",
        metadata={
            "raw_rows": [["", "", "", ""]],
            "geometry": geometry,
            "canonical_template_id": "annotations_and_inquiries",
        },
        headers=[],
        rows=[],
        bbox=[column_edges[0], top, column_edges[-1], bottom],
    )
    page_template = (
        "credit_account_detail"
        if defect == "foreign_owner"
        else "annotations_and_inquiries"
    )
    if defect == "foreign_owner":
        table.metadata["canonical_template_id"] = page_template
    page = SimpleNamespace(
        page_number=8,
        source_page_number=4,
        canonical_template_id=page_template,
        tables=[table],
        texts=[],
    )
    atoms_by_id: dict[str, EvidenceAtom] = {}
    for column_specs in token_specs.values():
        for text, bbox, token_id in column_specs:
            atoms_by_id.setdefault(
                token_id,
                EvidenceAtom(id=token_id, text=text, bbox=bbox),
            )
    return SimpleNamespace(
        pages=[page],
        evidence_plane=SimpleNamespace(
            evidence=EvidenceStore(text_atoms=list(atoms_by_id.values()))
        ),
    )


def _typed_token_ordinal(
    sequence: int,
    *,
    inquiry_type: str = "institution",
) -> dict[str, object]:
    return {
        "sequence": sequence,
        "inquiry_type": inquiry_type,
        "printed_fields": [],
        "field_source_refs": {},
        "source_refs": [
            {
                "source": "native_detail_inquiry_token_ordinal",
                "logical_page": 8,
                "source_page": 4,
                "table_id": "physical-inquiry-table",
                "row": 0,
                "column": 0,
                "sequence": sequence,
                "bbox": [50.0, 42.0 + (sequence - 1) * 11.0, 82.0, 49.0 + (sequence - 1) * 11.0],
                "geometry_scope": "token",
                "evidence_ids": [f"band:{sequence - 1}:sequence:0"],
                "binding": "printed_inquiry_ordinal_token",
                "binding_quality": "exact_token_in_sequence_cell",
            }
        ],
        "_source_owner_markers": [f"owner:{inquiry_type}:{sequence}"],
    }


@pytest.mark.parametrize("band_count", [2, 3, 5])
def test_exact_physical_inquiry_fields_accept_arbitrary_y_band_count(
    band_count: int,
) -> None:
    context = _physical_inquiry_context(band_count=band_count)
    page = context.pages[0]

    observations = native_extraction._bounded_inquiry_physical_field_observations(
        context,
        page,
        page.tables[0],
        row_index=0,
        slots=_SLOTS,
    )

    assert len(observations) == band_count
    assert len({item["source_physical_row_id"] for item in observations}) == band_count
    assert all(
        set(item["printed_fields"])
        == {"inquiry_date", "institution", "reason"}
        for item in observations
    )
    assert all(
        not {
            "inquiry_type",
            "sequence",
            "value",
            "raw_value",
            "normalized_value",
        }.intersection(item)
        for item in observations
    )
    assert any(
        len(item["field_source_refs"]["institution"][0]["evidence_ids"]) > 2
        for item in observations
    )


@pytest.mark.parametrize(
    "defect",
    [
        "cross_band_token",
        "overlapping_dates",
        "unknown_reason",
        "duplicate_evidence_owner",
        "foreign_owner",
    ],
)
def test_exact_physical_inquiry_fields_fail_closed(defect: str) -> None:
    context = _physical_inquiry_context(defect=defect)
    page = context.pages[0]

    assert (
        native_extraction._bounded_inquiry_physical_field_observations(
            context,
            page,
            page.tables[0],
            row_index=0,
            slots=_SLOTS,
        )
        == []
    )


def test_exact_physical_inquiry_fields_require_four_unique_role_columns() -> None:
    context = _physical_inquiry_context()
    page = context.pages[0]

    assert (
        native_extraction._bounded_inquiry_physical_field_observations(
            context,
            page,
            page.tables[0],
            row_index=0,
            slots={**_SLOTS, "institution": 1},
        )
        == []
    )


def test_exact_physical_fields_merge_into_one_local_typed_ordinal() -> None:
    context = _physical_inquiry_context(band_count=2, exact_sequences=True)
    page = context.pages[0]
    physical = native_extraction._bounded_inquiry_physical_field_observations(
        context,
        page,
        page.tables[0],
        row_index=0,
        slots=_SLOTS,
    )
    ordinals = {
        "institution": {
            sequence: _typed_token_ordinal(sequence)
            for sequence in (1, 2)
        }
    }

    native_extraction._merge_unique_inquiry_physical_fields_into_ordinals(
        context,
        physical,
        ordinals,
    )

    for sequence, observation in ordinals["institution"].items():
        assert set(observation["printed_fields"]) == {
            "inquiry_date",
            "institution",
            "reason",
        }
        assert all(
            observation["field_source_refs"][field_name]
            == physical[sequence - 1]["field_source_refs"][field_name]
            for field_name in ("inquiry_date", "institution", "reason")
        )
    assert all(
        not {"inquiry_type", "sequence"}.intersection(observation)
        for observation in physical
    )


def test_exact_physical_fields_do_not_merge_with_second_typed_candidate() -> None:
    context = _physical_inquiry_context(band_count=2, exact_sequences=True)
    page = context.pages[0]
    physical = native_extraction._bounded_inquiry_physical_field_observations(
        context,
        page,
        page.tables[0],
        row_index=0,
        slots=_SLOTS,
    )
    institution = _typed_token_ordinal(1)
    personal = _typed_token_ordinal(1, inquiry_type="personal")
    ordinals = {
        "institution": {1: institution},
        "personal": {1: personal},
    }

    native_extraction._merge_unique_inquiry_physical_fields_into_ordinals(
        context,
        physical,
        ordinals,
    )

    assert institution["printed_fields"] == []
    assert personal["printed_fields"] == []


def test_unconsumed_physical_inquiry_band_reports_only_three_local_fields() -> None:
    context = _physical_inquiry_context(band_count=2)
    page = context.pages[0]
    observation = native_extraction._bounded_inquiry_physical_field_observations(
        context,
        page,
        page.tables[0],
        row_index=0,
        slots=_SLOTS,
    )[0]
    content = {
        "facts": {
            "personal_detail_source_completeness_ledger": {
                "inquiry_physical_field_observations": [observation]
            }
        },
        "datasets": {"inquiry_records": []},
    }

    prepare_personal_detail_source_collections(content)
    prepare_personal_detail_source_collections(content)

    issues = [
        issue
        for issue in content["datasets"]["personal_detail_extraction_issues"]
        if issue.get("target_record_id")
        == observation["source_physical_row_id"]
    ]
    assert len(issues) == 3
    assert {issue["field_name"] for issue in issues} == {
        "inquiry_date",
        "institution",
        "reason",
    }
    assert all(issue["issue_code"] == "source_inquiry_field_omitted" for issue in issues)
    assert all(issue["observed_value"] == {"source_field_observed": True} for issue in issues)
    assert all(
        set(issue["candidate_value"]) == {"source_physical_row_id"}
        for issue in issues
    )
    assert not any(
        issue.get("issue_code") == "source_inquiry_record_omitted"
        and issue.get("target_record_id")
        == observation["source_physical_row_id"]
        for issue in content["datasets"]["personal_detail_extraction_issues"]
    )

    projected = project_personal_detail_datasets(content["datasets"])
    public_issues = [row.get("normalized", row) for row in projected["extraction_issues"]]
    physical_issues = [
        issue
        for issue in public_issues
        if issue.get("target_record_id") == observation["source_physical_row_id"]
    ]
    assert len(physical_issues) == 3
    assert {issue["target_dataset"] for issue in physical_issues} == {"inquiries"}
    assert all(issue["observed_value_type"] == "object" for issue in physical_issues)
    assert all(issue["candidate_value_type"] == "object" for issue in physical_issues)
    evidence_values = [
        row.get("normalized", row)
        for row in projected["extraction_issue_evidence"]
    ]
    for issue in physical_issues:
        issue_evidence = [
            row
            for row in evidence_values
            if row.get("extraction_issue_id") == issue["extraction_issue_id"]
        ]
        assert [
            row.get("boolean_value")
            for row in issue_evidence
            if row.get("evidence_kind") == "observed"
            and row.get("evidence_path") == "source_field_observed"
        ] == [True]
        assert [
            row.get("string_value")
            for row in issue_evidence
            if row.get("evidence_kind") == "candidate"
            and row.get("evidence_path") == "source_physical_row_id"
        ] == [observation["source_physical_row_id"]]

    public_payload = {
        "datasets": [
            {
                "name": name,
                "rows": [
                    {
                        "record_id": row["record_id"],
                        "normalized": deepcopy(row),
                        "canonical_raw": {},
                        "raw": {},
                        "source": {"page_range": [8, 8]},
                    }
                    for row in projected[name]
                ],
                "columns": [],
            }
            for name in ("extraction_issues", "extraction_issue_evidence")
        ]
    }
    _compact_personal_detail_public_projection(
        public_payload,
        source_datasets=[
            SimpleNamespace(public={"name": name}, rows=deepcopy(projected[name]))
            for name in ("extraction_issues", "extraction_issue_evidence")
        ],
    )
    compact_issues = [
        row
        for row in public_payload["datasets"][0]["rows"]
        if row.get("normalized", row).get("target_record_id")
        == observation["source_physical_row_id"]
    ]
    assert len(compact_issues) == 3
    assert all(
        row.get("source", {}).get("source_refs")
        and row["source"]["source_refs"][0]["source"]
        == "native_detail_inquiry_physical_field"
        for row in compact_issues
    )


@pytest.mark.parametrize(
    ("emitted_mode", "expected_record_issues", "expected_field_issues"),
    [
        ("absent", 1, 3),
        ("exact_evidence", 0, 0),
        ("identity_without_evidence", 0, 3),
    ],
)
def test_unique_typed_physical_band_requires_field_local_consumption(
    emitted_mode: str,
    expected_record_issues: int,
    expected_field_issues: int,
) -> None:
    context = _physical_inquiry_context(band_count=2, exact_sequences=True)
    page = context.pages[0]
    physical = native_extraction._bounded_inquiry_physical_field_observations(
        context,
        page,
        page.tables[0],
        row_index=0,
        slots=_SLOTS,
    )[0]
    ordinal = _typed_token_ordinal(1)
    internal_ordinals = {"institution": {1: ordinal}}
    native_extraction._merge_unique_inquiry_physical_fields_into_ordinals(
        context,
        [physical],
        internal_ordinals,
    )
    public_ordinal = {
        key: value
        for key, value in ordinal.items()
        if key != "_source_owner_markers"
    }
    emitted_rows: list[dict[str, object]] = []
    if emitted_mode != "absent":
        emitted_row: dict[str, object] = {
            "inquiry_id": "credit_inquiry:institution:1",
            "inquiry_type": "institution",
            "sequence": 1,
        }
        if emitted_mode == "exact_evidence":
            emitted_row["source_refs"] = [
                {
                    "source": "native_detail_table",
                    "geometry_scope": "row",
                    "evidence_ids": physical["source_refs"][0]["evidence_ids"],
                }
            ]
        emitted_rows.append(emitted_row)
    content = {
        "facts": {
            "personal_detail_source_completeness_ledger": {
                "inquiry_records": 1,
                "inquiry_sequence_endpoints": {"institution": 1},
                "inquiry_observed_sequences": {"institution": [1]},
                "inquiry_ordinal_observations": {
                    "institution": {"1": public_ordinal}
                },
                "inquiry_physical_field_observations": [physical],
            }
        },
        "datasets": {"inquiry_records": emitted_rows},
    }

    prepare_personal_detail_source_collections(content)

    typed_target = "credit_inquiry:institution:1"
    issues = content["datasets"]["personal_detail_extraction_issues"]
    assert sum(
        issue.get("issue_code") == "source_inquiry_record_omitted"
        and issue.get("target_record_id") == typed_target
        for issue in issues
    ) == expected_record_issues
    typed_field_issues = [
        issue
        for issue in issues
        if issue.get("issue_code") == "source_inquiry_field_omitted"
        and issue.get("target_record_id") == typed_target
    ]
    assert len(typed_field_issues) == expected_field_issues
    assert not any(
        issue.get("target_record_id") == physical["source_physical_row_id"]
        for issue in issues
    )


def test_emitted_inquiry_row_consumes_exact_physical_field_band() -> None:
    context = _physical_inquiry_context(band_count=2)
    page = context.pages[0]
    observation = native_extraction._bounded_inquiry_physical_field_observations(
        context,
        page,
        page.tables[0],
        row_index=0,
        slots=_SLOTS,
    )[0]
    content = {
        "facts": {
            "personal_detail_source_completeness_ledger": {
                "inquiry_physical_field_observations": [observation]
            }
        },
        "datasets": {
            "inquiry_records": [
                {
                    "source_refs": [
                        {
                            "source": "candidate_b_canonical_inquiry_line",
                            "geometry_scope": "row",
                            "evidence_ids": observation["source_refs"][0][
                                "evidence_ids"
                            ],
                        }
                    ]
                }
            ]
        },
    }

    prepare_personal_detail_source_collections(content)

    assert not any(
        issue.get("target_record_id")
        == observation["source_physical_row_id"]
        for issue in content["datasets"]["personal_detail_extraction_issues"]
    )


def test_physical_inquiry_observation_with_identity_payload_is_rejected() -> None:
    context = _physical_inquiry_context(band_count=2)
    page = context.pages[0]
    observation = native_extraction._bounded_inquiry_physical_field_observations(
        context,
        page,
        page.tables[0],
        row_index=0,
        slots=_SLOTS,
    )[0]
    observation["sequence"] = 99
    content = {
        "facts": {
            "personal_detail_source_completeness_ledger": {
                "inquiry_physical_field_observations": [observation]
            }
        },
        "datasets": {"inquiry_records": []},
    }

    prepare_personal_detail_source_collections(content)

    assert not any(
        issue.get("target_record_id")
        == observation["source_physical_row_id"]
        for issue in content["datasets"]["personal_detail_extraction_issues"]
    )
