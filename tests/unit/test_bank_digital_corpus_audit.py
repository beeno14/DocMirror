from __future__ import annotations

import importlib.util
import sys
from decimal import Decimal
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from docmirror.models.mirror.vnext import EvidenceStore
from docmirror.plugins.bank_statement import statement_context


@pytest.fixture(scope="module")
def audit_module() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "tmp" / "pdfs" / "bank_digital_corpus_audit.py"
    spec = importlib.util.spec_from_file_location("bank_digital_corpus_audit_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _manifest_payload(audit_module: ModuleType, corpus: Path) -> dict:
    entry = {
        "case_number": 1,
        "filename": "case_0001_fixture.pdf",
        "relative_path": "case_0001_fixture.pdf",
        "source_sha256": "a" * 64,
        "source_bytes": 1,
        "effective_page_count": 1,
        "effective_page_count_basis": "focused_fixture",
        "page_count_candidates": {"focused_fixture": 1},
        "page_count_disagreement": False,
        "content_group_size": 1,
    }
    return {
        "schema": audit_module.MANIFEST_SCHEMA,
        "generated_at": "2026-01-01T00:00:00+00:00",
        "corpus_root": str(corpus.resolve()),
        "file_count": 1,
        "unique_sha256_count": 1,
        "effective_page_count": 1,
        "page_count_disagreement_count": 0,
        "entries_fingerprint": audit_module._fingerprint([entry]),
        "entries": [entry],
    }


def test_harness_derives_manifest_from_work_root(tmp_path, audit_module):
    corpus = tmp_path / "Primary"
    corpus.mkdir()
    work_root = tmp_path / "work"
    manifest_path = work_root / "manifest.json"
    audit_module._atomic_write_json(manifest_path, _manifest_payload(audit_module, corpus))
    args = audit_module._parser().parse_args(["reproject", "--work-root", str(work_root)])

    manifest = audit_module._load_or_build_manifest(args)

    assert Path(args.manifest) == manifest_path.resolve()
    assert Path(args.corpus) == corpus.resolve()
    assert manifest["file_count"] == 1


def test_projection_fingerprint_covers_shared_export_code(audit_module):
    _fingerprint, files = audit_module._bank_code_fingerprint()
    assert "docmirror/output/community_bundle.py" in files
    assert "docmirror/server/edition_outputs.py" in files
    assert "docmirror/output/normalized_records.py" in files
    assert "docmirror/output/bank_business_view.py" in files
    assert "docmirror/models/schemas/registry.py" in files


def test_export_validation_options_do_not_enable_pdf_perception(audit_module):
    args = audit_module._parser().parse_args(
        ["reproject", "--validate-exports", "--baseline-report", "prior.json"]
    )
    assert args.validate_exports is True
    assert args.baseline_report == "prior.json"
    assert args.refresh is False
    assert args.command == "reproject"


def test_harness_rejects_manifest_corpus_mismatch_before_processing(tmp_path, audit_module):
    corpus = tmp_path / "Primary"
    other_corpus = tmp_path / "All"
    corpus.mkdir()
    other_corpus.mkdir()
    manifest_path = tmp_path / "manifest.json"
    audit_module._atomic_write_json(manifest_path, _manifest_payload(audit_module, corpus))
    args = audit_module._parser().parse_args(
        [
            "reproject",
            "--work-root",
            str(tmp_path / "work"),
            "--manifest",
            str(manifest_path),
            "--corpus",
            str(other_corpus),
        ]
    )

    with pytest.raises(ValueError, match="manifest corpus_root does not match --corpus"):
        audit_module._load_or_build_manifest(args)


def test_harness_cli_stops_before_projection_on_manifest_corpus_mismatch(
    tmp_path,
    audit_module,
    monkeypatch,
    capsys,
):
    corpus = tmp_path / "Primary"
    other_corpus = tmp_path / "All"
    corpus.mkdir()
    other_corpus.mkdir()
    manifest_path = tmp_path / "manifest.json"
    audit_module._atomic_write_json(manifest_path, _manifest_payload(audit_module, corpus))
    projected = False

    def unexpected_projection(*_args, **_kwargs):
        nonlocal projected
        projected = True
        raise AssertionError("projection must not run")

    monkeypatch.setattr(audit_module, "_reproject_selected", unexpected_projection)

    status = audit_module.main(
        [
            "reproject",
            "--manifest",
            str(manifest_path),
            "--corpus",
            str(other_corpus),
        ]
    )

    assert status == 2
    assert projected is False
    assert "Manifest configuration error" in capsys.readouterr().err


def test_harness_rejects_manifest_entry_that_escapes_corpus(tmp_path, audit_module):
    corpus = tmp_path / "Primary"
    corpus.mkdir()
    manifest = _manifest_payload(audit_module, corpus)
    manifest["entries"][0]["relative_path"] = "../outside.pdf"
    manifest["entries_fingerprint"] = audit_module._fingerprint(manifest["entries"])
    manifest_path = tmp_path / "manifest.json"
    audit_module._atomic_write_json(manifest_path, manifest)
    args = audit_module._parser().parse_args(["report", "--manifest", str(manifest_path)])

    with pytest.raises(ValueError, match="unsafe relative_path"):
        audit_module._load_or_build_manifest(args)


def _terminal_source(field: str, *, page: int = 3) -> dict:
    return {
        "raw_name": f"issuer_{field}",
        "source": "canonical_evidence_atoms",
        "source_refs": [
            {
                "source": "canonical_evidence_atoms",
                "source_page": page,
                "bbox": [10.0, 500.0, 100.0, 515.0],
            }
        ],
        "evidence_ids": [f"aggregate:{field}"],
    }


def _header() -> dict:
    terminal = {"debit_count": 3, "debit_total": "31.00", "credit_count": 1, "credit_total": "5.00"}
    return {
        "record_id": "statement_header:r000001",
        "normalized": dict(terminal),
        "canonical_raw": {key: str(value) for key, value in terminal.items()},
        "raw": {"本月累计": dict(terminal)},
        "source": {
            "source": "statement_header_scope",
            "page_range": [1, 3],
            "field_sources": {key: _terminal_source(key) for key in terminal},
        },
        "confidence": 1.0,
    }


def _transactions() -> list[dict]:
    values = [
        (1, "expense", "10.00", "90.00"),
        (2, "income", "5.00", "94.00"),
        (3, "expense", "20.00", "74.00"),
    ]
    return [
        {
            "record_id": f"bank:r{index:06d}",
            "normalized": {
                "statement_header_id": "statement_header:r000001",
                "direction": direction,
                "amount": amount,
                "balance": balance,
            },
            "canonical_raw": {"direction": direction, "amount": amount, "balance": balance},
            "raw": {},
            "source": {
                "source": "physical_table",
                "source_page": page,
                "page_range": [page, page],
                "bbox": [10.0, 100.0 + index * 20.0, 400.0, 115.0 + index * 20.0],
                "evidence_ids": [f"row:{index}"],
            },
        }
        for index, (page, direction, amount, balance) in enumerate(values, start=1)
    ]


def _physical_parse_result(*specs: dict) -> SimpleNamespace:
    pages = []
    for spec in specs:
        cell_bbox = spec.get("cell_bbox")
        cell_evidence_ids = list(spec.get("cell_evidence_ids") or [])
        geometry_bbox = spec.get("geometry_bbox", cell_bbox)
        geometry_evidence_ids = list(spec.get("geometry_evidence_ids", cell_evidence_ids) or [])
        cell = SimpleNamespace(bbox=cell_bbox, evidence_ids=cell_evidence_ids)
        row = SimpleNamespace(cells=[cell], source_row_index=0)
        table = SimpleNamespace(
            table_id=spec["table_id"],
            rows=[row],
            metadata={
                "raw_rows": [["value"]],
                "geometry": {
                    "cell_bboxes": [[geometry_bbox]],
                    "cell_evidence_ids": [[geometry_evidence_ids]],
                },
            },
        )
        pages.append(
            SimpleNamespace(
                page_number=spec["page"],
                source_page_number=spec["page"],
                tables=[table],
            )
        )
    return SimpleNamespace(pages=pages)


def _cell_ref(page: int, table_id: str, *, row: int = 0, raw_row: int = 0) -> dict:
    return {"page": page, "table_id": table_id, "row": row, "raw_row": raw_row, "col": 0}


def _reconciled_payload(monkeypatch) -> dict:
    def fact(page: int, value: str) -> statement_context._HeaderFact:
        return statement_context._HeaderFact(
            "brought_forward_balance",
            "承前余额",
            value,
            value,
            page,
            f"page:{page:04d}",
            (20.0, 60.0, 120.0, 75.0),
            (f"carry:{page}",),
        )

    monkeypatch.setattr(
        statement_context,
        "_page_header_facts",
        lambda _result: ({1: [], 2: [fact(2, "89.00")], 3: [fact(3, "94.00")]}, {}),
    )
    monkeypatch.setattr(
        statement_context,
        "_cached_independent_row_anchor_evidence",
        lambda _result, source_route: {
            "expected_rows": 3,
            "source": "positioned_date_anchors",
            "confidence": 0.80,
            "row_sources": [{"source_page": page} for page in (1, 2, 3)],
            "pages": [1, 2, 3],
        },
    )
    transactions = _transactions()
    [header] = statement_context.reconcile_source_unitemized_residuals(
        SimpleNamespace(),
        transactions,
        [_header()],
        source_route="digital",
        selected_source="canonical_table",
    )
    return {
        "datasets": [
            {"name": "statement_header", "rows": [header]},
            {"name": "transactions", "rows": transactions},
        ]
    }


def test_corpus_audit_accepts_exact_source_unitemized_reconciliation(monkeypatch, audit_module):
    payload = _reconciled_payload(monkeypatch)

    audit = audit_module.audit_community_payload(payload, effective_page_count=3)

    assert audit["status"] == "pass"
    assert "header_direction_count_mismatch" not in audit["finding_counts"]
    assert "header_direction_distribution_mismatch" not in audit["finding_counts"]
    assert "header_source_unitemized_reconciliation_invalid" not in audit["finding_counts"]
    assert len(payload["datasets"][1]["rows"]) == 3


def test_corpus_audit_accepts_atomic_scoped_source_metadata_between_business_datasets(
    monkeypatch, audit_module
):
    payload = _reconciled_payload(monkeypatch)
    metadata = {
        "record_id": "source_metadata:r000001",
        "normalized": {
            "metadata_field": "seal_code",
            "metadata_name": "业务印章编码",
            "metadata_value": "8DD4EA031026",
            "source_page_start": 2,
            "scope": "page",
        },
        "canonical_raw": {"metadata_name": "业务印章编码", "metadata_value": "8DD4EA031026"},
        "raw": {"metadata_name": "业务印章编码", "metadata_value": "8DD4EA031026"},
        "source": {"page_range": [2, 2]},
    }
    payload["datasets"].insert(1, {"name": "source_metadata", "rows": [metadata]})

    audit = audit_module.audit_community_payload(payload, effective_page_count=3)

    assert audit["status"] == "pass"
    assert audit["dataset_order"] == ["statement_header", "source_metadata", "transactions"]
    assert audit["source_metadata_rows"] == 1


def test_corpus_audit_rejects_unstructured_or_mis_scoped_source_metadata(monkeypatch, audit_module):
    payload = _reconciled_payload(monkeypatch)
    metadata = {
        "record_id": "source_metadata:r000001",
        "normalized": {
            "metadata_field": "other",
            "metadata_name": "终端号",
            "metadata_value": "T-01",
            "source_page_start": 2,
            "scope": "document",
            "content": "终端号=T-01",
        },
        "canonical_raw": {"metadata_name": "终端号", "metadata_value": "T-01"},
        "raw": {"metadata_name": "终端号", "metadata_value": "T-01"},
        "source": {"page_range": [2, 2]},
    }
    payload["datasets"].insert(1, {"name": "source_metadata", "rows": [metadata]})

    audit = audit_module.audit_community_payload(payload, effective_page_count=3)

    assert audit["status"] == "fail"
    assert audit["finding_counts"]["source_metadata_unstructured_fields"] == 1
    assert audit["finding_counts"]["source_metadata_scope_invalid"] == 1


def test_corpus_audit_rejects_normalized_opaque_identifier_corruption(monkeypatch, audit_module):
    payload = _reconciled_payload(monkeypatch)
    row = payload["datasets"][1]["rows"][0]
    row["raw"]["日志号"] = "546276664"
    row["canonical_raw"]["sequence_no"] = "546276664"
    row["normalized"]["sequence_no"] = "2"

    audit = audit_module.audit_community_payload(payload, effective_page_count=3)

    assert audit["status"] == "fail"
    assert audit["finding_counts"]["source_identifier_contradiction"] == 1
    [finding] = [item for item in audit["findings"] if item["code"] == "source_identifier_contradiction"]
    assert finding["detail"] == {
        "field": "sequence_no",
        "source": "546276664",
        "normalized": "2",
    }


def test_corpus_audit_rejects_recoverable_physical_provenance_loss(monkeypatch, audit_module):
    payload = _reconciled_payload(monkeypatch)
    source = payload["datasets"][1]["rows"][0]["source"]
    source.pop("bbox")
    source.pop("evidence_ids")
    source["table_id"] = "pt_1_0"
    source["source_cell_refs"] = [_cell_ref(1, "pt_1_0")]
    parse_result = _physical_parse_result(
        {
            "page": 1,
            "table_id": "pt_1_0",
            "cell_bbox": [10.0, 100.0, 20.0, 110.0],
            "cell_evidence_ids": ["ev:0001:text:000001"],
        }
    )

    audit = audit_module.audit_community_payload(
        payload,
        effective_page_count=3,
        parse_result=parse_result,
    )

    assert audit["status"] == "fail"
    assert audit["finding_counts"]["recoverable_row_bbox_missing"] == 1
    assert audit["finding_counts"]["recoverable_row_evidence_missing"] == 1


def test_corpus_audit_accepts_source_that_truly_lacks_physical_provenance(monkeypatch, audit_module):
    payload = _reconciled_payload(monkeypatch)
    source = payload["datasets"][1]["rows"][0]["source"]
    source.pop("bbox")
    source.pop("evidence_ids")
    source["table_id"] = "geo_table_0"
    source["source_cell_refs"] = [_cell_ref(1, "geo_table_0")]
    parse_result = _physical_parse_result(
        {
            "page": 1,
            "table_id": "geo_table_0",
            "cell_bbox": None,
            "cell_evidence_ids": [],
        }
    )

    audit = audit_module.audit_community_payload(
        payload,
        effective_page_count=3,
        parse_result=parse_result,
    )

    assert audit["status"] == "pass"
    assert "recoverable_row_bbox_missing" not in audit["finding_counts"]
    assert "recoverable_row_evidence_missing" not in audit["finding_counts"]


def test_corpus_audit_checks_exact_raw_row_geometry(monkeypatch, audit_module):
    payload = _reconciled_payload(monkeypatch)
    source = payload["datasets"][1]["rows"][0]["source"]
    source.pop("bbox")
    source.pop("evidence_ids")
    source["table_id"] = "pt_1_0"
    source["source_cell_refs"] = [_cell_ref(1, "pt_1_0")]
    parse_result = _physical_parse_result(
        {
            "page": 1,
            "table_id": "pt_1_0",
            "cell_bbox": None,
            "cell_evidence_ids": [],
            "geometry_bbox": [10.0, 100.0, 20.0, 110.0],
            "geometry_evidence_ids": ["ev:0001:text:000001"],
        }
    )

    audit = audit_module.audit_community_payload(
        payload,
        effective_page_count=3,
        parse_result=parse_result,
    )

    assert audit["status"] == "fail"
    assert audit["finding_counts"]["recoverable_row_bbox_missing"] == 1
    assert audit["finding_counts"]["recoverable_row_evidence_missing"] == 1


def test_corpus_audit_accepts_stitched_row_local_source_refs(monkeypatch, audit_module):
    payload = _reconciled_payload(monkeypatch)
    source = payload["datasets"][1]["rows"][0]["source"]
    source.pop("bbox")
    source.pop("evidence_ids")
    source["page_range"] = [1, 2]
    source["source_cell_refs"] = [
        _cell_ref(1, "pt_1_0"),
        _cell_ref(2, "pt_2_0"),
    ]
    source["source_refs"] = [
        {
            "source_page": 1,
            "page_range": [1, 1],
            "bbox": [10.0, 100.0, 20.0, 110.0],
            "evidence_ids": ["ev:0001:text:000001"],
        },
        {
            "source_page": 2,
            "page_range": [2, 2],
            "bbox": [10.0, 10.0, 20.0, 20.0],
            "evidence_ids": ["ev:0002:text:000001"],
        },
    ]
    parse_result = _physical_parse_result(
        {
            "page": 1,
            "table_id": "pt_1_0",
            "cell_bbox": [10.0, 100.0, 20.0, 110.0],
            "cell_evidence_ids": ["ev:0001:text:000001"],
        },
        {
            "page": 2,
            "table_id": "pt_2_0",
            "cell_bbox": [10.0, 10.0, 20.0, 20.0],
            "cell_evidence_ids": ["ev:0002:text:000001"],
        },
    )

    audit = audit_module.audit_community_payload(
        payload,
        effective_page_count=3,
        parse_result=parse_result,
    )

    assert audit["status"] == "pass"
    assert "recoverable_row_bbox_missing" not in audit["finding_counts"]
    assert "recoverable_row_evidence_missing" not in audit["finding_counts"]


def test_corpus_audit_does_not_guess_unresolved_physical_refs(monkeypatch, audit_module):
    payload = _reconciled_payload(monkeypatch)
    source = payload["datasets"][1]["rows"][0]["source"]
    source.pop("bbox")
    source.pop("evidence_ids")
    source["table_id"] = "pt_1_0"
    source["source_cell_refs"] = [_cell_ref(1, "pt_1_0", row=99, raw_row=99)]
    parse_result = _physical_parse_result(
        {
            "page": 1,
            "table_id": "pt_1_0",
            "cell_bbox": [10.0, 100.0, 20.0, 110.0],
            "cell_evidence_ids": ["ev:0001:text:000001"],
        }
    )

    audit = audit_module.audit_community_payload(
        payload,
        effective_page_count=3,
        parse_result=parse_result,
    )

    assert audit["status"] == "pass"
    assert "recoverable_row_bbox_missing" not in audit["finding_counts"]
    assert "recoverable_row_evidence_missing" not in audit["finding_counts"]


def test_corpus_audit_recovers_strict_native_alias_from_one_exact_public_raw_row(
    monkeypatch,
    audit_module,
):
    payload = _reconciled_payload(monkeypatch)
    row = payload["datasets"][1]["rows"][0]
    row["raw"] = {"source_column": "value"}
    source = row["source"]
    source.pop("evidence_ids")
    source["table_id"] = "native:p1:t0"
    # Native row indexes are not proof of cached row identity.  The exact full
    # public raw row below must be what resolves this source.
    source["source_row_index"] = 99
    # This is the same physical row in a quarter-turned top-left coordinate
    # system.  The gate should preserve/accept the native bbox, not compare it
    # numerically with or replace it from the cached physical table.
    source["bbox"] = [15.0, 76.0, 827.0, 102.0]
    source["source_refs"] = [
        {
            "source_page": 1,
            "page_range": [1, 1],
            "bbox": list(source["bbox"]),
            "source": "native_pdf_words",
        }
    ]
    parse_result = _physical_parse_result(
        {
            "page": 1,
            "table_id": "pt_1_0",
            "cell_bbox": [493.0, 15.0, 519.0, 827.0],
            "cell_evidence_ids": ["ev:0001:text:000001"],
        }
    )

    audit = audit_module.audit_community_payload(
        payload,
        effective_page_count=3,
        parse_result=parse_result,
    )

    assert audit["status"] == "fail"
    assert audit["finding_counts"]["recoverable_row_source_cell_refs_missing"] == 1
    assert audit["finding_counts"]["recoverable_row_evidence_missing"] == 1
    assert "recoverable_row_bbox_missing" not in audit["finding_counts"]


def test_corpus_audit_accepts_native_top_level_source_with_exact_pt_cell_refs(
    monkeypatch,
    audit_module,
):
    payload = _reconciled_payload(monkeypatch)
    row = payload["datasets"][1]["rows"][0]
    row["raw"] = {"source_column": "value"}
    source = row["source"]
    source["table_id"] = "native:p1:t0"
    source["source_cell_refs"] = [_cell_ref(1, "pt_1_0")]
    parse_result = _physical_parse_result(
        {
            "page": 1,
            "table_id": "pt_1_0",
            "cell_bbox": [10.0, 100.0, 20.0, 110.0],
            "cell_evidence_ids": ["ev:0001:text:000001"],
        }
    )

    audit = audit_module.audit_community_payload(
        payload,
        effective_page_count=3,
        parse_result=parse_result,
    )

    assert audit["status"] == "pass"
    assert "recoverable_row_source_cell_refs_missing" not in audit["finding_counts"]
    assert "recoverable_row_evidence_missing" not in audit["finding_counts"]


@pytest.mark.parametrize("invalid_route", ["raw_mismatch", "duplicate_raw", "noncanonical_table_index"])
def test_corpus_audit_native_alias_fallback_fails_closed(
    monkeypatch,
    audit_module,
    invalid_route,
):
    payload = _reconciled_payload(monkeypatch)
    row = payload["datasets"][1]["rows"][0]
    row["raw"] = {"source_column": "different" if invalid_route == "raw_mismatch" else "value"}
    source = row["source"]
    source.pop("evidence_ids")
    source.pop("bbox")
    source["source_refs"] = []
    source["table_id"] = "native:p1:t1" if invalid_route == "noncanonical_table_index" else "native:p1:t0"
    parse_result = _physical_parse_result(
        {
            "page": 1,
            # pt_1_1 at enumerated index zero must not satisfy native ...:t1.
            "table_id": "pt_1_1" if invalid_route == "noncanonical_table_index" else "pt_1_0",
            "cell_bbox": [10.0, 100.0, 20.0, 110.0],
            "cell_evidence_ids": ["ev:0001:text:000001"],
        }
    )
    if invalid_route == "duplicate_raw":
        table = parse_result.pages[0].tables[0]
        table.metadata["raw_rows"].append(["value"])
        table.metadata["geometry"]["cell_bboxes"].append([[10.0, 120.0, 20.0, 130.0]])
        table.metadata["geometry"]["cell_evidence_ids"].append([["ev:0001:text:000002"]])

    audit = audit_module.audit_community_payload(
        payload,
        effective_page_count=3,
        parse_result=parse_result,
    )

    assert "recoverable_row_source_cell_refs_missing" not in audit["finding_counts"]
    assert "recoverable_row_bbox_missing" not in audit["finding_counts"]
    assert "recoverable_row_evidence_missing" not in audit["finding_counts"]


def test_corpus_audit_rejects_unproven_source_unitemized_reconciliation(monkeypatch, audit_module):
    payload = _reconciled_payload(monkeypatch)
    header = payload["datasets"][0]["rows"][0]
    header["source"]["field_sources"]["source_unitemized_debit_count"].pop("derivation")

    audit = audit_module.audit_community_payload(payload, effective_page_count=3)

    assert audit["status"] == "fail"
    assert audit["finding_counts"]["header_source_unitemized_reconciliation_invalid"] == 1
    assert audit["finding_counts"]["header_direction_count_mismatch"] == 1


def test_corpus_audit_rejects_source_unitemized_amount_that_does_not_reconcile(monkeypatch, audit_module):
    payload = _reconciled_payload(monkeypatch)
    payload["datasets"][0]["rows"][0]["normalized"]["source_unitemized_debit_amount"] = "2.00"

    audit = audit_module.audit_community_payload(payload, effective_page_count=3)

    assert audit["status"] == "fail"
    assert audit["finding_counts"]["header_source_unitemized_reconciliation_invalid"] == 1


def test_corpus_audit_rejects_source_unitemized_field_in_issuer_raw(monkeypatch, audit_module):
    payload = _reconciled_payload(monkeypatch)
    payload["datasets"][0]["rows"][0]["raw"]["source_unitemized_debit_count"] = 1

    audit = audit_module.audit_community_payload(payload, effective_page_count=3)

    assert audit["status"] == "fail"
    assert audit["finding_counts"]["header_source_unitemized_reconciliation_invalid"] == 1


def test_corpus_audit_rejects_source_unitemized_field_in_canonical_raw(monkeypatch, audit_module):
    payload = _reconciled_payload(monkeypatch)
    payload["datasets"][0]["rows"][0]["canonical_raw"]["source_unitemized_debit_count"] = 1

    audit = audit_module.audit_community_payload(payload, effective_page_count=3)

    assert audit["status"] == "fail"
    assert audit["finding_counts"]["header_source_unitemized_reconciliation_invalid"] == 1


def _period_source(field: str) -> dict:
    return {
        "raw_name": field,
        "source": "canonical_evidence_atoms",
        "source_refs": [
            {
                "source": "canonical_evidence_atoms",
                "source_page": 1,
                "bbox": [10.0, 10.0, 100.0, 20.0],
            }
        ],
        "evidence_ids": [f"period:{field}"],
    }


def _period_payload(
    header_normalized: dict,
    transaction_normalized: list[dict],
    *,
    propagate: bool = False,
    source_only: bool = False,
) -> dict:
    period_fields = {
        field: value for field in ("period_start", "period_end") if (value := header_normalized.get(field))
    }
    header_field_sources = {field: _period_source(field) for field in period_fields}
    header = {
        "record_id": "statement_header:r000001",
        "normalized": dict(header_normalized),
        "canonical_raw": dict(header_normalized),
        "raw": {},
        "source": {
            "source": "statement_header_scope",
            "source_page": 1,
            "page_range": [1, 1],
            "field_sources": header_field_sources,
        },
    }
    transactions = []
    for index, values in enumerate(transaction_normalized, start=1):
        normalized = {"statement_header_id": header["record_id"], **values}
        canonical = {field: values[field] for field in ("period_start", "period_end") if field in values}
        field_sources = {}
        if propagate and not source_only:
            normalized.update(period_fields)
            canonical.update(period_fields)
        if propagate or source_only:
            field_sources = {field: dict(header_field_sources[field]) for field in period_fields}
        transactions.append(
            {
                "record_id": f"bank:r{index:06d}",
                "normalized": normalized,
                "canonical_raw": canonical,
                "raw": {},
                "source": {
                    "source": "physical_table",
                    "source_page": 1,
                    "page_range": [1, 1],
                    "bbox": [10.0, 40.0 + index * 10.0, 100.0, 45.0 + index * 10.0],
                    "evidence_ids": [f"row:{index}"],
                    "field_sources": field_sources,
                },
            }
        )
    return {
        "datasets": [
            {"name": "statement_header", "rows": [header]},
            {"name": "transactions", "rows": transactions},
        ]
    }


def test_period_audit_preserves_coherent_inclusive_header_context(audit_module):
    payload = _period_payload(
        {"period_start": "2023-03-10", "period_end": "2023-09-10"},
        [{"date": "2023-03-10"}, {"date": "2023-09-10"}],
        propagate=True,
    )

    audit = audit_module.audit_community_payload(payload, effective_page_count=1)

    assert audit["status"] == "pass"
    assert "header_period_incoherent_with_linked_transactions" not in audit["finding_counts"]
    assert "header_context_not_conserved" not in audit["finding_counts"]


@pytest.mark.parametrize(
    "transaction",
    [
        {"date": "2023-03-09", "period_start": "2023-03-10"},
        {"date": "2023-09-11", "period_end": "2023-09-10"},
    ],
)
def test_period_audit_rejects_transaction_own_open_bound_exclusion(audit_module, transaction):
    payload = _period_payload({}, [transaction])

    audit = audit_module.audit_community_payload(payload, effective_page_count=1)

    assert audit["status"] == "fail"
    assert audit["finding_counts"]["transaction_date_outside_own_period"] == 1


def test_period_audit_allows_stripped_context_for_one_incoherent_header(audit_module):
    payload = _period_payload(
        {"period_start": "2023-03-10", "period_end": "2023-09-10"},
        [{"date": "2022-09-11"}, {"date": "2023-03-10"}],
    )

    audit = audit_module.audit_community_payload(payload, effective_page_count=1)

    assert audit["status"] == "pass"
    assert audit["finding_counts"]["header_period_incoherent_with_linked_transactions"] == 1
    assert audit["severity_counts"] == {"diagnostic": 1}
    assert "header_context_not_conserved" not in audit["finding_counts"]


def test_period_audit_rejects_values_propagated_from_incoherent_header(audit_module):
    payload = _period_payload(
        {"period_start": "2023-03-10", "period_end": "2023-09-10"},
        [{"date": "2022-09-11"}],
        propagate=True,
    )

    audit = audit_module.audit_community_payload(payload, effective_page_count=1)

    assert audit["status"] == "fail"
    assert audit["finding_counts"]["header_period_incoherent_with_linked_transactions"] == 1
    assert audit["finding_counts"]["incoherent_header_period_context_propagated"] == 2
    assert audit["finding_counts"]["transaction_date_outside_own_period"] == 1


def test_period_audit_rejects_source_only_propagation_from_incoherent_header(audit_module):
    payload = _period_payload(
        {"period_start": "2023-03-10", "period_end": "2023-09-10"},
        [{"date": "2022-09-11"}],
        source_only=True,
    )

    audit = audit_module.audit_community_payload(payload, effective_page_count=1)

    assert audit["status"] == "fail"
    assert audit["finding_counts"]["incoherent_header_period_context_propagated"] == 2
    assert "transaction_date_outside_own_period" not in audit["finding_counts"]


@pytest.mark.parametrize(
    ("header_period", "dates"),
    [
        ({"period_start": "2023-03-10"}, ["2023-03-10", "2023-09-11"]),
        ({"period_end": "2023-09-10"}, ["2023-03-09", "2023-09-10"]),
    ],
)
def test_period_audit_preserves_coherent_one_sided_header_context(
    audit_module,
    header_period,
    dates,
):
    payload = _period_payload(
        header_period,
        [{"date": date_value} for date_value in dates],
        propagate=True,
    )

    audit = audit_module.audit_community_payload(payload, effective_page_count=1)

    assert audit["status"] == "pass"
    assert "header_period_incoherent_with_linked_transactions" not in audit["finding_counts"]


def test_period_audit_does_not_treat_cutoff_as_a_period_bound(audit_module):
    payload = _period_payload(
        {"statement_cutoff_date": "2023-03-10"},
        [{"date": "2022-09-11"}, {"date": "2023-09-11"}],
    )

    audit = audit_module.audit_community_payload(payload, effective_page_count=1)

    assert audit["status"] == "pass"
    assert "header_period_incoherent_with_linked_transactions" not in audit["finding_counts"]
    assert "transaction_date_outside_own_period" not in audit["finding_counts"]


def test_period_audit_uses_timestamp_when_normalized_date_is_absent(audit_module):
    payload = _period_payload(
        {"period_start": "2023-03-10"},
        [{"timestamp": "2023-03-09T23:59:59"}],
    )

    audit = audit_module.audit_community_payload(payload, effective_page_count=1)

    assert audit["status"] == "pass"
    assert audit["finding_counts"]["header_period_incoherent_with_linked_transactions"] == 1
    assert audit["findings"][0]["detail"]["before_start_rows"] == 1


def _native_text_atom(
    evidence_id: str,
    text: str,
    bbox: list[float],
    *,
    source_kind: str = "pdf_native",
) -> SimpleNamespace:
    return SimpleNamespace(
        id=evidence_id,
        text=text,
        bbox=bbox,
        page_id="page:0001",
        source_kind=source_kind,
        confidence=1.0,
    )


def _balance_header_atoms(*, include_right_neighbor: bool = True) -> list[SimpleNamespace]:
    atoms = [
        _native_text_atom("header:amount", "交易金额", [258.0, 80.0, 290.0, 88.0]),
        _native_text_atom("header:balance:en", "Balance", [337.0, 70.0, 365.0, 78.0]),
        _native_text_atom("header:balance:zh", "余额", [337.0, 80.0, 353.0, 88.0]),
    ]
    if include_right_neighbor:
        atoms.extend(
            [
                _native_text_atom("header:counter", "对方账号", [411.0, 80.0, 443.0, 88.0]),
                _native_text_atom("header:summary", "摘要", [667.0, 80.0, 683.0, 88.0]),
            ]
        )
    return atoms


def _balance_parse_result(
    row_atoms: list[SimpleNamespace],
    *,
    include_right_neighbor: bool = True,
) -> SimpleNamespace:
    atoms = [*_balance_header_atoms(include_right_neighbor=include_right_neighbor), *row_atoms]
    return SimpleNamespace(
        pages=[SimpleNamespace(page_number=1, source_page_number=1, tables=[])],
        evidence_plane=SimpleNamespace(
            evidence=[SimpleNamespace(text_atoms=atoms)],
        ),
    )


def _evidence_store_parse_results(
    atoms: list[SimpleNamespace],
) -> tuple[SimpleNamespace, dict]:
    serialized_atoms = [dict(vars(atom)) for atom in atoms]
    store = EvidenceStore.model_validate({"text_atoms": serialized_atoms})
    typed = SimpleNamespace(
        pages=[SimpleNamespace(page_number=1, source_page_number=1, tables=[])],
        evidence_plane=SimpleNamespace(evidence=store),
    )
    serialized = {
        "pages": [{"page_number": 1, "source_page_number": 1, "tables": []}],
        "evidence_plane": {"evidence": {"text_atoms": serialized_atoms}},
    }
    return typed, serialized


def _balance_payload(*rows: dict) -> dict:
    header_id = "statement_header:r000001"
    header = {
        "record_id": header_id,
        "normalized": {},
        "canonical_raw": {},
        "raw": {},
        "source": {
            "source": "statement_header_scope",
            "source_page": 1,
            "page_range": [1, 1],
        },
    }
    transactions = []
    for index, spec in enumerate(rows, start=1):
        normalized = {
            "statement_header_id": header_id,
            "balance": spec["normalized_balance"],
            **spec.get("normalized", {}),
        }
        canonical = {
            "balance": spec["canonical_balance"],
            **spec.get("canonical", {}),
        }
        source = {
            "source": "canonical_evidence_table",
            "source_page": 1,
            "page_range": [1, 1],
            "bbox": spec["bbox"],
            "evidence_ids": list(spec["evidence_ids"]),
        }
        if "reconstruction_repairs" in spec:
            source["reconstruction_repairs"] = spec["reconstruction_repairs"]
        transactions.append(
            {
                "record_id": f"records:r{index:06d}",
                "normalized": normalized,
                "canonical_raw": canonical,
                "raw": {"余额": spec["canonical_balance"]},
                "source": source,
            }
        )
    return {
        "datasets": [
            {"name": "statement_header", "rows": [header]},
            {"name": "transactions", "rows": transactions},
        ]
    }


def test_balance_source_role_gate_rejects_normalized_mismatch(audit_module):
    balance_atom = _native_text_atom(
        "row:balance",
        "45,462.51",
        [332.0, 120.0, 368.0, 128.0],
    )
    payload = _balance_payload(
        {
            "normalized_balance": "6.18",
            "canonical_balance": "45,462.51",
            "bbox": [52.0, 110.0, 731.0, 140.0],
            "evidence_ids": [balance_atom.id],
        }
    )

    audit = audit_module.audit_community_payload(
        payload,
        effective_page_count=1,
        parse_result=_balance_parse_result([balance_atom]),
    )

    assert audit["status"] == "fail"
    assert audit["finding_counts"]["transaction_balance_source_role_mismatch"] == 1
    [finding] = [item for item in audit["findings"] if item["code"] == "transaction_balance_source_role_mismatch"]
    assert finding["path"] == "datasets.transactions.rows.0.normalized.balance"
    assert finding["detail"]["reasons"] == ["normalized_value_mismatch"]
    assert finding["detail"]["source_balance"]["evidence_id"] == "row:balance"


def test_balance_source_role_gate_rejects_cross_column_canonical_residue(audit_module):
    balance_atom = _native_text_atom(
        "row:balance",
        "51,492.51",
        [332.0, 120.0, 368.0, 128.0],
    )
    summary_atom = _native_text_atom(
        "row:summary",
        "6.17两单6030",
        [667.0, 120.0, 715.0, 128.0],
    )
    payload = _balance_payload(
        {
            "normalized_balance": "51492.51",
            "canonical_balance": "51,492.516.17",
            "bbox": [52.0, 110.0, 731.0, 140.0],
            "evidence_ids": [balance_atom.id, summary_atom.id],
            "reconstruction_repairs": [
                {
                    "field": "余额",
                    "ocr_raw": "51,492.516.17",
                    "reconstructed": "51,492.51",
                }
            ],
        }
    )

    audit = audit_module.audit_community_payload(
        payload,
        effective_page_count=1,
        parse_result=_balance_parse_result([balance_atom, summary_atom]),
    )

    assert audit["status"] == "fail"
    assert audit["finding_counts"]["transaction_balance_source_role_mismatch"] == 1
    [finding] = [item for item in audit["findings"] if item["code"] == "transaction_balance_source_role_mismatch"]
    assert finding["path"] == "datasets.transactions.rows.0.canonical_raw.balance"
    assert finding["detail"]["reasons"] == ["cross_column_residue"]
    assert finding["detail"]["cross_column_sources"] == [
        {
            "evidence_id": "row:summary",
            "text": "6.17两单6030",
            "bbox": [667.0, 120.0, 715.0, 128.0],
            "role": "summary",
            "residue": "6.17",
        }
    ]


def test_balance_source_role_gate_accepts_numeric_summary_kept_in_its_column(audit_module):
    balance_atom = _native_text_atom(
        "row:balance",
        "51,492.51",
        [332.0, 120.0, 368.0, 128.0],
    )
    summary_atom = _native_text_atom(
        "row:summary",
        "6.17两单6030",
        [667.0, 120.0, 715.0, 128.0],
    )
    payload = _balance_payload(
        {
            "normalized_balance": "51492.51",
            "canonical_balance": "51,492.51",
            "bbox": [52.0, 110.0, 731.0, 140.0],
            "evidence_ids": [balance_atom.id, summary_atom.id],
        }
    )

    audit = audit_module.audit_community_payload(
        payload,
        effective_page_count=1,
        parse_result=_balance_parse_result([balance_atom, summary_atom]),
    )

    assert audit["status"] == "pass"
    assert "transaction_balance_source_role_mismatch" not in audit["finding_counts"]


@pytest.mark.parametrize("ambiguity", ["missing_neighbor", "multiple_candidates", "ocr_atom"])
def test_balance_source_role_gate_skips_ambiguous_evidence(audit_module, ambiguity):
    balance_atom = _native_text_atom(
        "row:balance",
        "45,462.51",
        [332.0, 120.0, 368.0, 128.0],
        source_kind="ocr" if ambiguity == "ocr_atom" else "pdf_native",
    )
    row_atoms = [balance_atom]
    evidence_ids = [balance_atom.id]
    if ambiguity == "multiple_candidates":
        duplicate = _native_text_atom(
            "row:balance:duplicate",
            "45,462.52",
            [334.0, 130.0, 370.0, 138.0],
        )
        row_atoms.append(duplicate)
        evidence_ids.append(duplicate.id)
    payload = _balance_payload(
        {
            "normalized_balance": "6.18",
            "canonical_balance": "45,462.51",
            "bbox": [52.0, 110.0, 731.0, 145.0],
            "evidence_ids": evidence_ids,
        }
    )

    audit = audit_module.audit_community_payload(
        payload,
        effective_page_count=1,
        parse_result=_balance_parse_result(
            row_atoms,
            include_right_neighbor=ambiguity != "missing_neighbor",
        ),
    )

    assert audit["status"] == "pass"
    assert "transaction_balance_source_role_mismatch" not in audit["finding_counts"]


def test_balance_source_role_gate_does_not_hard_fail_filtered_ledger_gap(audit_module):
    first_balance = _native_text_atom(
        "row:balance:1",
        "100.00",
        [337.0, 120.0, 365.0, 128.0],
    )
    second_balance = _native_text_atom(
        "row:balance:2",
        "50.00",
        [337.0, 150.0, 365.0, 158.0],
    )
    payload = _balance_payload(
        {
            "normalized_balance": "100.00",
            "canonical_balance": "100.00",
            "normalized": {"direction": "expense", "amount": "1.00"},
            "canonical": {"direction": "expense", "amount": "1.00"},
            "bbox": [52.0, 110.0, 731.0, 140.0],
            "evidence_ids": [first_balance.id],
        },
        {
            "normalized_balance": "50.00",
            "canonical_balance": "50.00",
            "normalized": {"direction": "expense", "amount": "1.00"},
            "canonical": {"direction": "expense", "amount": "1.00"},
            "bbox": [52.0, 145.0, 731.0, 175.0],
            "evidence_ids": [second_balance.id],
        },
    )

    audit = audit_module.audit_community_payload(
        payload,
        effective_page_count=1,
        parse_result=_balance_parse_result([first_balance, second_balance]),
    )

    assert audit["status"] == "pass"
    assert "transaction_balance_source_role_mismatch" not in audit["finding_counts"]


def test_balance_source_role_gate_has_typed_evidence_store_mapping_parity(audit_module):
    balance_atom = _native_text_atom(
        "row:balance",
        "45,462.51",
        [332.0, 120.0, 368.0, 128.0],
    )
    atoms = [*_balance_header_atoms(), balance_atom]
    typed, serialized = _evidence_store_parse_results(atoms)
    payload = _balance_payload(
        {
            "normalized_balance": "6.18",
            "canonical_balance": "45,462.51",
            "bbox": [52.0, 110.0, 731.0, 140.0],
            "evidence_ids": [balance_atom.id],
        }
    )

    typed_audit = audit_module.audit_community_payload(
        payload,
        effective_page_count=1,
        parse_result=typed,
    )
    serialized_audit = audit_module.audit_community_payload(
        payload,
        effective_page_count=1,
        parse_result=serialized,
    )

    assert audit_module._native_text_atoms(typed) == audit_module._native_text_atoms(serialized)
    assert typed_audit == serialized_audit
    assert typed_audit["finding_counts"]["transaction_balance_source_role_mismatch"] == 1


def test_fused_direction_amount_header_does_not_widen_balance_band(audit_module):
    header_atoms = [
        _native_text_atom("header:summary", "摘要", [131.0, 80.0, 150.0, 88.0]),
        _native_text_atom("header:amount", "⽀/收交易⾦额", [208.0, 80.0, 270.0, 88.0]),
        _native_text_atom("header:balance", "账户余额", [280.0, 80.0, 319.0, 88.0]),
        _native_text_atom("header:location", "交易地点", [329.0, 80.0, 368.0, 88.0]),
        _native_text_atom("header:party", "对⽅户名", [410.0, 80.0, 449.0, 88.0]),
    ]
    amount_atom = _native_text_atom("row:amount", "-30,000.00", [210.0, 120.0, 258.0, 128.0])
    fused_balance = _native_text_atom(
        "row:balance-location",
        "170,015.87兴业银行漳州高新",
        [280.0, 120.0, 407.0, 128.0],
    )
    payload = _balance_payload(
        {
            "normalized_balance": "170015.87",
            "canonical_balance": "170,015.87",
            "bbox": [29.0, 110.0, 560.0, 140.0],
            "evidence_ids": [amount_atom.id, fused_balance.id],
        }
    )
    typed, serialized = _evidence_store_parse_results([*header_atoms, amount_atom, fused_balance])

    typed_audit = audit_module.audit_community_payload(
        payload,
        effective_page_count=1,
        parse_result=typed,
    )
    serialized_audit = audit_module.audit_community_payload(
        payload,
        effective_page_count=1,
        parse_result=serialized,
    )

    assert audit_module._source_header_role("⽀/收交易⾦额") == "amount"
    assert typed_audit == serialized_audit
    assert typed_audit["status"] == "pass"
    assert "transaction_balance_source_role_mismatch" not in typed_audit["finding_counts"]


def _cross_row_role_parse_result(*, include_crossing_account: bool = True) -> SimpleNamespace:
    atoms = [
        _native_text_atom("header:amount", "交易金额", [208.0, 80.0, 270.0, 88.0]),
        _native_text_atom("header:balance", "账户余额", [280.0, 80.0, 319.0, 88.0]),
        _native_text_atom("header:location", "交易地点", [329.0, 80.0, 368.0, 88.0]),
        _native_text_atom("header:party", "对方户名", [394.0, 80.0, 432.0, 88.0]),
        _native_text_atom("header:account", "对方账户/对方银行", [469.0, 80.0, 550.0, 88.0]),
        _native_text_atom("row:previous:balance", "100.00", [280.0, 116.0, 319.0, 124.0]),
        _native_text_atom("row:previous:party", "支付宝(中国)", [394.0, 140.0, 460.0, 148.0]),
        _native_text_atom("row:following:balance", "75.00", [280.0, 152.0, 319.0, 160.0]),
        _native_text_atom("row:following:party", "网络技术有限公司", [394.0, 152.0, 460.0, 160.0]),
        _native_text_atom("row:following:account", "支付宝(中国)网络技术有限公司", [469.0, 152.0, 565.0, 160.0]),
    ]
    if include_crossing_account:
        atoms.append(_native_text_atom("row:previous:account", "215500690", [469.0, 140.0, 518.0, 148.0]))
    return SimpleNamespace(
        pages=[SimpleNamespace(page_number=1, source_page_number=1, tables=[])],
        evidence_plane=SimpleNamespace(evidence=[SimpleNamespace(text_atoms=atoms)]),
    )


def _cross_row_role_payload(*, include_crossing_account: bool = True) -> dict:
    previous_ids = ["row:previous:balance", "row:previous:party"]
    if include_crossing_account:
        previous_ids.append("row:previous:account")
    return _balance_payload(
        {
            "normalized_balance": "100.00",
            "canonical_balance": "100.00",
            "bbox": [29.0, 108.0, 565.0, 150.0],
            "evidence_ids": previous_ids,
        },
        {
            "normalized_balance": "75.00",
            "canonical_balance": "75.00",
            "bbox": [29.0, 146.0, 565.0, 170.0],
            "evidence_ids": [
                "row:following:balance",
                "row:following:party",
                "row:following:account",
            ],
        },
    )


def test_cross_row_source_role_gate_requires_two_independent_columns(audit_module):
    failing = audit_module.audit_community_payload(
        _cross_row_role_payload(),
        effective_page_count=1,
        parse_result=_cross_row_role_parse_result(),
    )
    ambiguous = audit_module.audit_community_payload(
        _cross_row_role_payload(include_crossing_account=False),
        effective_page_count=1,
        parse_result=_cross_row_role_parse_result(include_crossing_account=False),
    )

    assert failing["finding_counts"]["transaction_cross_row_source_role_ownership_mismatch"] == 1
    assert failing["findings"][-1]["path"] == "datasets.transactions.rows.0.source.evidence_ids"
    assert "transaction_cross_row_source_role_ownership_mismatch" not in ambiguous["finding_counts"]


def _direction_distribution_payload(
    rows: list[tuple[str, str]],
    *,
    debit_count: int,
    debit_total: str | None,
    credit_count: int,
    credit_total: str | None,
) -> dict:
    header_id = "statement_header:r000001"
    terminal = {
        "debit_count": debit_count,
        "debit_total": debit_total,
        "credit_count": credit_count,
        "credit_total": credit_total,
        "total_transactions": len(rows),
    }
    header = {
        "record_id": header_id,
        "normalized": dict(terminal),
        "canonical_raw": {key: str(value) for key, value in terminal.items()},
        "raw": {"合计": {key: str(value) for key, value in terminal.items()}},
        "source": {
            "source": "statement_header_scope",
            "source_page": 1,
            "page_range": [1, 1],
            "field_sources": {
                key: _terminal_source(key, page=1) for key, value in terminal.items() if value is not None
            },
        },
    }
    transactions = []
    balance = 1000
    for index, (direction, amount) in enumerate(rows, start=1):
        transactions.append(
            {
                "record_id": f"records:r{index:06d}",
                "normalized": {
                    "statement_header_id": header_id,
                    "direction": direction,
                    "amount": amount,
                    "balance": str(balance),
                },
                "canonical_raw": {
                    "direction": direction,
                    "amount": amount,
                    "balance": str(balance),
                },
                "raw": {},
                "source": {
                    "source": "physical_table",
                    "source_page": 1,
                    "page_range": [1, 1],
                    "bbox": [10.0, 100.0 + index * 20.0, 400.0, 115.0 + index * 20.0],
                    "evidence_ids": [f"row:{index}"],
                },
            }
        )
    return {
        "datasets": [
            {"name": "statement_header", "rows": [header]},
            {"name": "transactions", "rows": transactions},
        ]
    }


def _drop_header_fields(payload: dict, *fields: str) -> None:
    header = payload["datasets"][0]["rows"][0]
    for field in fields:
        header["normalized"].pop(field, None)
        header["canonical_raw"].pop(field, None)
        header["source"]["field_sources"].pop(field, None)


def _set_source_bound_header_field(payload: dict, field: str, value: str) -> None:
    header = payload["datasets"][0]["rows"][0]
    header["normalized"][field] = value
    header["canonical_raw"][field] = value
    header["source"]["field_sources"][field] = _terminal_source(field, page=1)


def _set_derived_page_total(
    payload: dict,
    field: str,
    raw_name: str,
    page_values: list[str],
    *,
    declared_value: str | None = None,
    sign_normalization: str | None = None,
) -> None:
    header = payload["datasets"][0]["rows"][0]
    total = sum((Decimal(value) for value in page_values), Decimal("0"))
    total_text = declared_value if declared_value is not None else format(total, "f")
    header["normalized"][field] = total_text
    header["canonical_raw"][field] = total_text
    header["source"]["page_range"] = [1, len(page_values)]
    field_source = {
        "raw_name": raw_name,
        "source": "derived_explicit_page_aggregate",
        "derivation": "sum_explicit_page_totals",
        "components": [
            {
                "page": page,
                "raw_name": raw_name,
                "raw_value": value,
                "normalized_value": value,
                "bbox": [10.0, 500.0, 150.0, 515.0],
                "evidence_ids": [f"page-total:{field}:{page}"],
                "source": "canonical_evidence_atoms",
            }
            for page, value in enumerate(page_values, start=1)
        ],
    }
    if sign_normalization is not None:
        field_source["sign_normalization"] = sign_normalization
    header["source"]["field_sources"][field] = field_source


def _icbc_page_total_payload() -> dict:
    payload = _direction_distribution_payload(
        [("expense", "10.00"), ("income", "5.00"), ("expense", "20.00")],
        debit_count=2,
        debit_total="30.00",
        credit_count=1,
        credit_total="5.00",
    )
    _drop_header_fields(payload, "total_transactions", "debit_count", "credit_count")
    _set_derived_page_total(payload, "debit_total", "本页支出算术合计:", ["10.00", "20.00"])
    _set_derived_page_total(payload, "credit_total", "本页收入算术合计:", ["2.00", "3.00"])
    return payload


def test_completion_checker_marks_authoritative_count_and_totals_complete(audit_module):
    payload = _direction_distribution_payload(
        [("expense", "10.00"), ("income", "5.00"), ("expense", "20.00")],
        debit_count=2,
        debit_total="30.00",
        credit_count=1,
        credit_total="5.00",
    )

    audit = audit_module.audit_community_payload(payload, effective_page_count=1)

    assert audit["status"] == "pass"
    assert audit["completion_status"] == "complete"
    assert audit["completion_status_counts"] == {"complete": 1}
    [check] = audit["completion_checks"]
    assert check["status"] == "complete"
    assert check["accounted_transactions"] == 3
    assert check["accepted_direction_mappings"] == ["direct"]


def test_completion_checker_accepts_issuer_count_unit_in_canonical_value(audit_module):
    payload = _direction_distribution_payload(
        [("expense", "10.00"), ("income", "5.00"), ("expense", "20.00")],
        debit_count=2,
        debit_total="30.00",
        credit_count=1,
        credit_total="5.00",
    )
    header = payload["datasets"][0]["rows"][0]
    header["canonical_raw"]["debit_count"] = "２ 笔"
    header["canonical_raw"]["credit_count"] = "1笔"

    audit = audit_module.audit_community_payload(payload, effective_page_count=1)

    assert audit["status"] == "pass"
    assert audit["completion_status"] == "complete"
    assert "header_aggregate_provenance_invalid" not in audit["finding_counts"]


@pytest.mark.parametrize(
    "canonical_value",
    [
        "2页",
        "共2笔",
        "2.5笔",
        "-2笔",
        Decimal("NaN"),
        Decimal("Infinity"),
        float("nan"),
        float("inf"),
    ],
)
def test_completion_checker_rejects_noncanonical_count_unit_values(audit_module, canonical_value):
    payload = _direction_distribution_payload(
        [("expense", "10.00"), ("income", "5.00"), ("expense", "20.00")],
        debit_count=2,
        debit_total="30.00",
        credit_count=1,
        credit_total="5.00",
    )
    payload["datasets"][0]["rows"][0]["canonical_raw"]["debit_count"] = canonical_value

    audit = audit_module.audit_community_payload(payload, effective_page_count=1)

    assert audit["status"] == "fail"
    assert audit["completion_status"] == "inconsistent"
    assert audit["finding_counts"]["header_aggregate_provenance_invalid"] == 1


@pytest.mark.parametrize("canonical_value", ["NaN", "Infinity", "-Infinity"])
def test_completion_checker_rejects_nonfinite_aggregate_values(audit_module, canonical_value):
    payload = _direction_distribution_payload(
        [("expense", "10.00"), ("income", "5.00"), ("expense", "20.00")],
        debit_count=2,
        debit_total="30.00",
        credit_count=1,
        credit_total="5.00",
    )
    payload["datasets"][0]["rows"][0]["canonical_raw"]["debit_total"] = canonical_value

    audit = audit_module.audit_community_payload(payload, effective_page_count=1)

    assert audit["status"] == "fail"
    assert audit["completion_status"] == "inconsistent"
    assert audit["finding_counts"]["header_aggregate_provenance_invalid"] == 1


def test_completion_checker_marks_unproven_header_aggregates_unverified(audit_module):
    payload = _direction_distribution_payload(
        [("expense", "10.00"), ("income", "5.00")],
        debit_count=1,
        debit_total="10.00",
        credit_count=1,
        credit_total="5.00",
    )
    payload["datasets"][0]["rows"][0]["source"]["field_sources"] = {}

    audit = audit_module.audit_community_payload(payload, effective_page_count=1)

    assert audit["status"] == "pass"
    assert audit["completion_status"] == "unverified"
    [check] = audit["completion_checks"]
    assert check["authoritative_fields"] == []
    assert check["ignored_fields"]["total_transactions"] == "source_missing"
    assert "authoritative_transaction_count_missing" in check["reasons"]


def test_completion_checker_accepts_issuer_row_count_evidence_contract(audit_module):
    payload = _direction_distribution_payload(
        [("expense", "10.00"), ("income", "5.00")],
        debit_count=1,
        debit_total="10.00",
        credit_count=1,
        credit_total="5.00",
    )
    _drop_header_fields(payload, "debit_count", "debit_total", "credit_count", "credit_total")
    header = payload["datasets"][0]["rows"][0]
    header["source"]["field_sources"]["total_transactions"] = {
        "raw_name": "split_footer_transaction_count",
        "source": "row_count_evidence.split_footer",
        "raw_value": "2",
        "normalized_value": "2",
    }

    audit = audit_module.audit_community_payload(payload, effective_page_count=1)

    assert audit["status"] == "pass"
    assert audit["completion_status"] == "complete"
    assert audit["completion_checks"][0]["authoritative_fields"] == ["total_transactions"]


def test_completion_checker_rejects_aggregate_cross_layer_contradiction(audit_module):
    payload = _direction_distribution_payload(
        [("expense", "10.00"), ("income", "5.00")],
        debit_count=1,
        debit_total="10.00",
        credit_count=1,
        credit_total="5.00",
    )
    payload["datasets"][0]["rows"][0]["canonical_raw"]["total_transactions"] = "3"

    audit = audit_module.audit_community_payload(payload, effective_page_count=1)

    assert audit["status"] == "fail"
    assert audit["completion_status"] == "inconsistent"
    assert audit["finding_counts"]["header_aggregate_provenance_invalid"] == 1


def test_completion_checker_marks_authoritative_count_mismatch_inconsistent(audit_module):
    payload = _direction_distribution_payload(
        [("expense", "10.00"), ("income", "5.00")],
        debit_count=1,
        debit_total="10.00",
        credit_count=1,
        credit_total="5.00",
    )
    _drop_header_fields(payload, "debit_count", "debit_total", "credit_count", "credit_total")
    header = payload["datasets"][0]["rows"][0]
    header["normalized"]["total_transactions"] = 3
    header["canonical_raw"]["total_transactions"] = "3"

    audit = audit_module.audit_community_payload(payload, effective_page_count=1)

    assert audit["status"] == "fail"
    assert audit["completion_status"] == "inconsistent"
    assert audit["finding_counts"]["header_transaction_count_mismatch"] == 1


def test_completion_checker_applies_declared_amount_unit_with_decimal_math(audit_module):
    payload = _direction_distribution_payload(
        [("expense", "10.00"), ("income", "5.00"), ("expense", "20.00")],
        debit_count=2,
        debit_total="0.003",
        credit_count=1,
        credit_total="0.0005",
    )
    _set_source_bound_header_field(payload, "amount_unit", "万元")
    _set_source_bound_header_field(payload, "currency", "CNY")
    for row in payload["datasets"][1]["rows"]:
        row["normalized"]["currency"] = "CNY"
        row["canonical_raw"]["currency"] = "CNY"

    audit = audit_module.audit_community_payload(payload, effective_page_count=1)

    assert audit["status"] == "pass"
    assert audit["completion_status"] == "complete"
    assert audit["completion_checks"][0]["accepted_direction_mappings"] == ["direct"]


def test_completion_checker_rejects_currency_scope_contradiction(audit_module):
    payload = _direction_distribution_payload(
        [("expense", "10.00"), ("income", "5.00")],
        debit_count=1,
        debit_total="10.00",
        credit_count=1,
        credit_total="5.00",
    )
    _set_source_bound_header_field(payload, "currency", "CNY")
    for row in payload["datasets"][1]["rows"]:
        row["normalized"]["currency"] = "USD"
        row["canonical_raw"]["currency"] = "USD"

    audit = audit_module.audit_community_payload(payload, effective_page_count=1)

    assert audit["status"] == "fail"
    assert audit["completion_status"] == "inconsistent"
    assert audit["finding_counts"]["header_transaction_currency_mismatch"] == 1


def test_completion_checker_accepts_source_bound_filtered_scope(audit_module):
    payload = _direction_distribution_payload(
        [("income", "10.00"), ("income", "5.00")],
        debit_count=0,
        debit_total="0.00",
        credit_count=2,
        credit_total="15.00",
    )
    _set_source_bound_header_field(payload, "direction_filter", "收入")

    audit = audit_module.audit_community_payload(payload, effective_page_count=1)

    assert audit["status"] == "pass"
    assert audit["completion_status"] == "complete"


def test_completion_checker_rejects_transaction_outside_filtered_scope(audit_module):
    payload = _direction_distribution_payload(
        [("income", "10.00"), ("expense", "5.00")],
        debit_count=1,
        debit_total="5.00",
        credit_count=1,
        credit_total="10.00",
    )
    _set_source_bound_header_field(payload, "direction_filter", "收入")

    audit = audit_module.audit_community_payload(payload, effective_page_count=1)

    assert audit["status"] == "fail"
    assert audit["completion_status"] == "inconsistent"
    assert audit["finding_counts"]["header_filter_scope_mismatch"] == 1


def test_completion_checker_does_not_treat_amount_only_contract_as_row_complete(audit_module):
    payload = _direction_distribution_payload(
        [("expense", "10.00"), ("income", "5.00")],
        debit_count=1,
        debit_total="10.00",
        credit_count=1,
        credit_total="5.00",
    )
    _drop_header_fields(payload, "total_transactions", "debit_count", "credit_count")

    audit = audit_module.audit_community_payload(payload, effective_page_count=1)

    assert audit["status"] == "pass"
    assert audit["completion_status"] == "unverified"
    assert "authoritative_amounts_without_authoritative_direction_counts" in audit["completion_checks"][0]["reasons"]


def test_completion_checker_accepts_evidenced_icbc_page_arithmetic_totals(audit_module):
    payload = _icbc_page_total_payload()

    audit = audit_module.audit_community_payload(payload, effective_page_count=2)

    assert audit["status"] == "pass"
    assert audit["completion_status"] == "unverified"
    [check] = audit["completion_checks"]
    assert check["authoritative_fields"] == ["credit_total", "debit_total"]
    assert check["ignored_fields"] == {}
    assert "authoritative_amounts_without_authoritative_direction_counts" in check["reasons"]
    assert "authoritative_transaction_count_missing" in check["reasons"]


def test_completion_checker_accepts_icbc_page_totals_with_issuer_spelling_and_signed_debits(
    audit_module,
):
    payload = _direction_distribution_payload(
        [("expense", "10.00"), ("income", "5.00"), ("expense", "20.00")],
        debit_count=2,
        debit_total="30.00",
        credit_count=1,
        credit_total="5.00",
    )
    _set_derived_page_total(
        payload,
        "debit_total",
        "本页支出算数合计:",
        ["-10.00", "-20.00"],
        declared_value="30.00",
        sign_normalization="magnitude_from_nonpositive_expense_page_totals",
    )
    _set_derived_page_total(
        payload,
        "credit_total",
        "本页收入算数合计:",
        ["2.00", "3.00"],
    )

    audit = audit_module.audit_community_payload(payload, effective_page_count=2)

    assert audit["status"] == "pass"
    assert audit["completion_status"] == "complete"
    assert audit["completion_checks"][0]["accepted_direction_mappings"] == ["direct"]


def test_completion_checker_accepts_nonnegative_expense_page_total_magnitude_contract(audit_module):
    payload = _direction_distribution_payload(
        [("expense", "10.00"), ("income", "5.00"), ("expense", "20.00")],
        debit_count=2,
        debit_total="30.00",
        credit_count=1,
        credit_total="5.00",
    )
    _set_derived_page_total(
        payload,
        "debit_total",
        "本页支出算术合计:",
        ["10.00", "20.00"],
        declared_value="30.00",
        sign_normalization="magnitude_from_nonnegative_expense_page_totals",
    )
    _set_derived_page_total(
        payload,
        "credit_total",
        "本页收入算术合计:",
        ["5.00", "0.00"],
    )

    audit = audit_module.audit_community_payload(payload, effective_page_count=2)

    assert audit["status"] == "pass"
    assert audit["completion_status"] == "complete"
    assert audit["completion_checks"][0]["accepted_direction_mappings"] == ["direct"]


@pytest.mark.parametrize(
    ("field", "page_values", "declared_value", "sign_normalization"),
    [
        ("debit_total", ["-10.00", "-20.00"], "30.00", None),
        (
            "debit_total",
            ["10.00", "20.00"],
            "30.00",
            "magnitude_from_nonpositive_expense_page_totals",
        ),
        (
            "debit_total",
            ["-10.00", "-20.00"],
            "30.00",
            "unsupported_sign_contract",
        ),
        (
            "credit_total",
            ["-2.00", "-3.00"],
            "5.00",
            "magnitude_from_nonpositive_expense_page_totals",
        ),
        (
            "debit_total",
            ["-10.00", "-20.00"],
            "30.00",
            "magnitude_from_nonnegative_expense_page_totals",
        ),
        (
            "credit_total",
            ["2.00", "3.00"],
            "5.00",
            "magnitude_from_nonnegative_expense_page_totals",
        ),
    ],
)
def test_completion_checker_rejects_uncontracted_or_invalid_page_total_sign_normalization(
    audit_module,
    field,
    page_values,
    declared_value,
    sign_normalization,
):
    payload = _direction_distribution_payload(
        [("expense", "10.00"), ("income", "5.00"), ("expense", "20.00")],
        debit_count=2,
        debit_total="30.00",
        credit_count=1,
        credit_total="5.00",
    )
    raw_name = "本页支出算数合计:" if field == "debit_total" else "本页收入算数合计:"
    _set_derived_page_total(
        payload,
        field,
        raw_name,
        page_values,
        declared_value=declared_value,
        sign_normalization=sign_normalization,
    )

    audit = audit_module.audit_community_payload(payload, effective_page_count=2)

    assert audit["status"] == "fail"
    assert audit["completion_status"] == "inconsistent"
    assert audit["finding_counts"]["header_aggregate_provenance_invalid"] == 1


@pytest.mark.parametrize(
    "damage",
    ["missing-page", "component-sum", "raw-mismatch", "inferred-source", "duplicate-page"],
)
def test_completion_checker_rejects_incomplete_derived_page_total_provenance(audit_module, damage):
    payload = _icbc_page_total_payload()
    header = payload["datasets"][0]["rows"][0]
    debit_source = header["source"]["field_sources"]["debit_total"]
    if damage == "missing-page":
        debit_source["components"].pop()
    elif damage == "component-sum":
        debit_source["components"][1]["normalized_value"] = "19.99"
    elif damage == "raw-mismatch":
        debit_source["components"][1]["raw_value"] = "19.99"
    elif damage == "inferred-source":
        debit_source["components"][1]["source"] = "inferred_page_total"
    else:
        debit_source["components"][1]["page"] = 1

    audit = audit_module.audit_community_payload(payload, effective_page_count=2)

    assert audit["status"] == "fail"
    assert audit["completion_status"] == "inconsistent"
    assert audit["finding_counts"]["header_aggregate_provenance_invalid"] == 1


def test_completion_checker_does_not_promote_page_total_label_without_page_derivation(audit_module):
    payload = _direction_distribution_payload(
        [("expense", "10.00")],
        debit_count=1,
        debit_total="10.00",
        credit_count=0,
        credit_total=None,
    )
    _drop_header_fields(payload, "total_transactions", "debit_count", "credit_count", "credit_total")
    header = payload["datasets"][0]["rows"][0]
    header["source"]["field_sources"]["debit_total"]["raw_name"] = "本页支出算术合计:"

    audit = audit_module.audit_community_payload(payload, effective_page_count=1)

    assert audit["status"] == "pass"
    assert audit["completion_status"] == "unverified"
    [check] = audit["completion_checks"]
    assert check["authoritative_fields"] == []
    assert check["ignored_fields"]["debit_total"] == "aggregate_label_not_authoritative"


def test_completion_checker_fails_closed_on_unknown_amount_unit(audit_module):
    payload = _direction_distribution_payload(
        [("expense", "10.00"), ("income", "5.00")],
        debit_count=1,
        debit_total="10.00",
        credit_count=1,
        credit_total="5.00",
    )
    _set_source_bound_header_field(payload, "amount_unit", "袋")

    audit = audit_module.audit_community_payload(payload, effective_page_count=1)

    assert audit["status"] == "pass"
    assert audit["completion_status"] == "unverified"
    assert "amount_unit_semantics_unverified" in audit["completion_checks"][0]["reasons"]


def _attach_signed_source_plane(
    payload: dict,
    source_amounts: list[str],
    *,
    split_columns: bool = False,
) -> list[dict]:
    rows = payload["datasets"][1]["rows"]
    assert len(rows) == len(source_amounts)
    for row, source_amount in zip(rows, source_amounts, strict=True):
        direction = row["normalized"]["direction"]
        if split_columns:
            row["raw"] = {
                "收入": source_amount if direction == "income" else "",
                "支出": source_amount if direction == "expense" else "",
            }
            row["canonical_raw"]["direction"] = direction
        else:
            source_direction = "贷方" if direction == "income" else "借方"
            row["raw"] = {"借/贷": source_direction, "交易金额": source_amount}
            row["canonical_raw"]["direction"] = source_direction
        row["canonical_raw"]["amount"] = source_amount
    return rows


@pytest.mark.parametrize(
    ("debit_count", "debit_total", "credit_count", "credit_total"),
    [
        (2, "30.00", 1, "5.00"),
        (1, "5.00", 2, "30.00"),
    ],
    ids=["direct", "swapped"],
)
def test_header_direction_distribution_accepts_exact_direct_or_swapped_mapping(
    audit_module,
    debit_count,
    debit_total,
    credit_count,
    credit_total,
):
    payload = _direction_distribution_payload(
        [("expense", "10.00"), ("income", "5.00"), ("expense", "20.00")],
        debit_count=debit_count,
        debit_total=debit_total,
        credit_count=credit_count,
        credit_total=credit_total,
    )

    audit = audit_module.audit_community_payload(payload, effective_page_count=1)

    assert audit["status"] == "pass"
    assert "header_direction_distribution_mismatch" not in audit["finding_counts"]


def test_header_direction_distribution_rejects_when_neither_mapping_reconciles(audit_module):
    payload = _direction_distribution_payload(
        [("expense", "10.00"), ("income", "5.00"), ("expense", "20.00")],
        debit_count=2,
        debit_total="29.00",
        credit_count=1,
        credit_total="6.00",
    )

    audit = audit_module.audit_community_payload(payload, effective_page_count=1)

    assert audit["status"] == "fail"
    assert audit["finding_counts"]["header_direction_distribution_mismatch"] == 1
    [finding] = [item for item in audit["findings"] if item["code"] == "header_direction_distribution_mismatch"]
    assert finding["detail"]["accepted_mappings"] == []
    assert finding["detail"]["actual_expense_total"] == "30.00"
    assert finding["detail"]["actual_income_total"] == "5.00"


def test_header_direction_distribution_accepts_exact_ambiguous_mapping(audit_module):
    payload = _direction_distribution_payload(
        [("expense", "10.00"), ("income", "10.00")],
        debit_count=1,
        debit_total="10.00",
        credit_count=1,
        credit_total="10.00",
    )

    audit = audit_module.audit_community_payload(payload, effective_page_count=1)

    assert audit["status"] == "pass"
    assert "header_direction_distribution_mismatch" not in audit["finding_counts"]


def test_header_direction_distribution_accepts_count_only_direct_mapping(audit_module):
    payload = _direction_distribution_payload(
        [("expense", "10.00"), ("income", "5.00"), ("expense", "20.00")],
        debit_count=2,
        debit_total=None,
        credit_count=1,
        credit_total=None,
    )

    audit = audit_module.audit_community_payload(payload, effective_page_count=1)

    assert audit["status"] == "pass"
    assert "header_direction_distribution_mismatch" not in audit["finding_counts"]


@pytest.mark.parametrize(
    (
        "debit_count",
        "debit_total",
        "credit_count",
        "credit_total",
        "split_columns",
        "expected",
    ),
    [
        (2, "7.00", 1, "5.00", False, ("direct",)),
        (1, "5.00", 2, "7.00", True, ("swapped",)),
    ],
    ids=["dedicated-direct", "split-swapped"],
)
def test_direction_distribution_accepts_complete_source_signed_reversal_plane(
    audit_module,
    debit_count,
    debit_total,
    credit_count,
    credit_total,
    split_columns,
    expected,
):
    payload = _direction_distribution_payload(
        [("expense", "10.00"), ("expense", "3.00"), ("income", "5.00")],
        debit_count=debit_count,
        debit_total=debit_total,
        credit_count=credit_count,
        credit_total=credit_total,
    )
    rows = _attach_signed_source_plane(payload, ["10.00", "-3.00", "5.00"], split_columns=split_columns)

    _, candidates = audit_module._direction_distribution_candidates(
        rows,
        declared_counts={"debit": debit_count, "credit": credit_count},
        declared_amounts={"debit": Decimal(debit_total), "credit": Decimal(credit_total)},
        residual_counts={"debit": 0, "credit": 0},
        residual_amounts={"debit": Decimal("0"), "credit": Decimal("0")},
    )
    audit = audit_module.audit_community_payload(payload, effective_page_count=1)

    assert candidates == expected
    assert audit["status"] == "pass"
    assert "header_direction_distribution_mismatch" not in audit["finding_counts"]


def test_direction_distribution_signed_plane_fails_closed_on_incomplete_provenance(audit_module):
    def signed_rows(source_amounts: list[str]) -> list[dict]:
        payload = _direction_distribution_payload(
            [("expense", "10.00"), ("expense", "3.00"), ("income", "5.00")],
            debit_count=2,
            debit_total="7.00",
            credit_count=1,
            credit_total="5.00",
        )
        return _attach_signed_source_plane(payload, source_amounts)

    missing_canonical = signed_rows(["10.00", "-3.00", "5.00"])
    missing_canonical[1]["canonical_raw"].pop("amount")
    magnitude_mismatch = signed_rows(["10.00", "-2.00", "5.00"])
    direction_mismatch = signed_rows(["10.00", "-3.00", "5.00"])
    direction_mismatch[1]["canonical_raw"]["direction"] = "贷方"
    unowned_positive = signed_rows(["10.00", "-3.00", "5.00"])
    contaminated_direction = "借方，以网点对账单为准。客服电话：95595"
    unowned_positive[0]["raw"]["借/贷"] = contaminated_direction
    unowned_positive[0]["canonical_raw"]["direction"] = contaminated_direction

    common = {
        "declared_counts": {"debit": 2, "credit": 1},
        "residual_counts": {"debit": 0, "credit": 0},
        "residual_amounts": {"debit": Decimal("0"), "credit": Decimal("0")},
    }
    _, missing_candidates = audit_module._direction_distribution_candidates(
        missing_canonical,
        declared_amounts={"debit": Decimal("7.00"), "credit": Decimal("5.00")},
        **common,
    )
    _, magnitude_candidates = audit_module._direction_distribution_candidates(
        magnitude_mismatch,
        declared_amounts={"debit": Decimal("8.00"), "credit": Decimal("5.00")},
        **common,
    )
    _, direction_candidates = audit_module._direction_distribution_candidates(
        direction_mismatch,
        declared_amounts={"debit": Decimal("7.00"), "credit": Decimal("5.00")},
        **common,
    )
    _, unowned_positive_candidates = audit_module._direction_distribution_candidates(
        unowned_positive,
        declared_amounts={"debit": Decimal("7.00"), "credit": Decimal("5.00")},
        **common,
    )
    residual_rows = signed_rows(["10.00", "-3.00", "5.00"])
    _, residual_candidates = audit_module._direction_distribution_candidates(
        residual_rows,
        declared_counts={"debit": 3, "credit": 1},
        declared_amounts={"debit": Decimal("8.00"), "credit": Decimal("5.00")},
        residual_counts={"debit": 1, "credit": 0},
        residual_amounts={"debit": Decimal("1.00"), "credit": Decimal("0")},
    )

    assert missing_candidates == ()
    assert magnitude_candidates == ()
    assert direction_candidates == ()
    assert unowned_positive_candidates == ()
    assert residual_candidates == ()


def test_direction_distribution_accepts_swapped_mapping_with_declared_side_residual(audit_module):
    payload = _direction_distribution_payload(
        [("expense", "10.00"), ("expense", "20.00"), ("income", "5.00")],
        debit_count=1,
        debit_total="5.00",
        credit_count=3,
        credit_total="31.00",
    )

    _, candidates = audit_module._direction_distribution_candidates(
        payload["datasets"][1]["rows"],
        declared_counts={"debit": 1, "credit": 3},
        declared_amounts={"debit": Decimal("5.00"), "credit": Decimal("31.00")},
        residual_counts={"debit": 0, "credit": 1},
        residual_amounts={"debit": Decimal("0"), "credit": Decimal("1.00")},
    )

    assert candidates == ("swapped",)


@pytest.mark.parametrize(
    ("rows", "counts", "amounts", "expected"),
    [
        (
            [("expense", "10.00"), ("expense", "20.00"), ("income", "5.00")],
            {"debit": 2, "credit": 1},
            {"debit": Decimal("30.00"), "credit": Decimal("5.00")},
            ("direct",),
        ),
        (
            [("expense", "10.00"), ("expense", "20.00"), ("income", "5.00")],
            {"debit": 1, "credit": 2},
            {"debit": Decimal("5.00"), "credit": Decimal("30.00")},
            ("swapped",),
        ),
        (
            [("expense", "10.00"), ("expense", "20.00"), ("income", "5.00")],
            {"debit": 2, "credit": 1},
            {"debit": Decimal("29.00"), "credit": Decimal("6.00")},
            (),
        ),
        (
            [("expense", "10.00"), ("income", "10.00")],
            {"debit": 1, "credit": 1},
            {"debit": Decimal("10.00"), "credit": Decimal("10.00")},
            ("direct", "swapped"),
        ),
        (
            [("expense", "10.00"), ("expense", "20.00"), ("income", "5.00")],
            {"debit": 2, "credit": 1},
            {"debit": None, "credit": None},
            ("direct",),
        ),
    ],
    ids=["direct-only", "swapped-only", "neither", "ambiguous-both", "count-only"],
)
def test_direction_distribution_candidate_truth_table(
    audit_module,
    rows,
    counts,
    amounts,
    expected,
):
    payload = _direction_distribution_payload(
        rows,
        debit_count=counts["debit"],
        debit_total=None,
        credit_count=counts["credit"],
        credit_total=None,
    )
    transaction_rows = payload["datasets"][1]["rows"]

    _, candidates = audit_module._direction_distribution_candidates(
        transaction_rows,
        declared_counts=counts,
        declared_amounts=amounts,
        residual_counts={"debit": 0, "credit": 0},
        residual_amounts={"debit": Decimal("0"), "credit": Decimal("0")},
    )

    assert candidates == expected


def test_direction_distribution_accepts_source_bound_zero_amount_count_slack_without_inference(
    audit_module,
):
    payload = _direction_distribution_payload(
        [
            ("expense", "10.00"),
            ("expense", "20.00"),
            ("income", "5.00"),
            ("", "0.00"),
            ("", "0.00"),
            ("", "0.00"),
            ("", "0.00"),
        ],
        debit_count=2,
        debit_total="30.00",
        credit_count=5,
        credit_total="5.00",
    )
    rows = payload["datasets"][1]["rows"]

    visible, candidates = audit_module._direction_distribution_candidates(
        rows,
        declared_counts={"debit": 2, "credit": 5},
        declared_amounts={"debit": Decimal("30.00"), "credit": Decimal("5.00")},
        residual_counts={"debit": 0, "credit": 0},
        residual_amounts={"debit": Decimal("0"), "credit": Decimal("0")},
    )
    audit = audit_module.audit_community_payload(payload, effective_page_count=1)

    assert candidates == ("direct",)
    assert visible == {
        "expense": {"count": 2, "amount": Decimal("30.00")},
        "income": {"count": 1, "amount": Decimal("5.00")},
    }
    assert audit["status"] == "pass"
    assert "header_direction_distribution_mismatch" not in audit["finding_counts"]
    assert [(row["normalized"]["direction"], row["canonical_raw"]["direction"]) for row in rows[3:]] == [("", "")] * 4


@pytest.mark.parametrize(
    "rejection",
    [
        "directionless-nonzero",
        "source-zero-missing",
        "source-direction-present",
        "totals-mismatch",
        "count-only",
        "row-count-mismatch",
    ],
)
def test_direction_distribution_zero_amount_count_slack_fails_closed(
    audit_module,
    rejection,
):
    debit_total = None if rejection == "count-only" else "30.00"
    credit_total = None if rejection == "count-only" else "5.00"
    if rejection == "totals-mismatch":
        credit_total = "6.00"
    credit_count = 4 if rejection == "row-count-mismatch" else 5
    payload = _direction_distribution_payload(
        [
            ("expense", "10.00"),
            ("expense", "20.00"),
            ("income", "5.00"),
            ("", "0.00"),
            ("", "0.00"),
            ("", "0.00"),
            ("", "0.00"),
        ],
        debit_count=2,
        debit_total=debit_total,
        credit_count=credit_count,
        credit_total=credit_total,
    )
    rows = payload["datasets"][1]["rows"]
    if rejection == "directionless-nonzero":
        rows[3]["normalized"]["amount"] = "1.00"
        rows[3]["canonical_raw"]["amount"] = "1.00"
    elif rejection == "source-zero-missing":
        rows[3]["canonical_raw"].pop("amount")
    elif rejection == "source-direction-present":
        rows[3]["canonical_raw"]["direction"] = "income"

    visible, candidates = audit_module._direction_distribution_candidates(
        rows,
        declared_counts={"debit": 2, "credit": credit_count},
        declared_amounts={
            "debit": Decimal(debit_total) if debit_total is not None else None,
            "credit": Decimal(credit_total) if credit_total is not None else None,
        },
        residual_counts={"debit": 0, "credit": 0},
        residual_amounts={"debit": Decimal("0"), "credit": Decimal("0")},
    )
    audit = audit_module.audit_community_payload(payload, effective_page_count=1)

    assert visible is not None
    assert candidates == ()
    assert audit["status"] == "fail"
    assert audit["finding_counts"]["header_direction_distribution_mismatch"] == 1


@pytest.mark.parametrize("evidence_state", ["retained", "missing", "different_normalized"])
def test_normalized_delivery_is_audited_against_retained_internal_evidence(audit_module, evidence_state):
    import copy

    evidence = _direction_distribution_payload(
        [("expense", "10.00"), ("income", "5.00")],
        debit_count=1, debit_total="10.00", credit_count=1, credit_total="5.00",
    )
    payload = copy.deepcopy(evidence)
    payload["schema"] = {"version": "4.0.0"}
    for dataset in payload["datasets"]:
        for row in dataset["rows"]:
            row.pop("raw")
            row.pop("canonical_raw")
    if evidence_state == "different_normalized":
        payload["datasets"][1]["rows"][0]["normalized"]["amount"] = "999.00"
    result = audit_module.audit_community_payload(
        payload, effective_page_count=1,
        evidence_payload=None if evidence_state == "missing" else evidence,
    )
    if evidence_state == "retained":
        assert result["status"] == "pass"
    elif evidence_state == "missing":
        assert result["finding_counts"]["internal_evidence_required_for_normalized_export"] == 1
    else:
        assert result["finding_counts"]["internal_evidence_mismatch"] == 1


def test_normalized_projection_cache_requires_hash_checked_evidence(tmp_path, audit_module):
    payload_path = tmp_path / "sample.community.json"
    meta_path = tmp_path / "sample.meta.json"
    evidence_path = payload_path.with_suffix(".evidence.json")
    audit_module._atomic_write_json(payload_path, {"schema": {"version": "4.0.0"}})
    meta = {"community_sha256": audit_module._sha256_file(payload_path)}
    audit_module._atomic_write_json(meta_path, meta)
    assert not audit_module._projection_payload_is_valid(payload_path, meta_path, {})
    audit_module._atomic_write_json(evidence_path, {"datasets": []})
    meta["evidence_sha256"] = audit_module._sha256_file(evidence_path)
    audit_module._atomic_write_json(meta_path, meta)
    assert audit_module._projection_payload_is_valid(payload_path, meta_path, {})
    audit_module._atomic_write_json(evidence_path, {"datasets": ["changed"]})
    assert not audit_module._projection_payload_is_valid(payload_path, meta_path, {})
