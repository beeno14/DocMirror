from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.verify_personal_detail_final_audit import (
    COMPLETION_SCHEMA,
    EXPECTED_SOURCE_PDFS,
    MANIFEST_SCHEMA,
    MATRIX_SCHEMA,
    NODE_RESULT_SCHEMA,
    SAVED_CASES,
    SNAPSHOT_SCHEMA,
    AuditError,
    SourcePdf,
    atomic_write_json,
    capture_pdf_snapshot,
    compare_snapshots,
    file_record,
    finalize_audit,
    parse_junit,
    sha256_file,
    validate_node,
    validate_result_records,
    validate_saved_pair,
    verify_completed_audit,
)


def _write_junit(path: Path, *, classname: str, name: str, skipped: bool = False) -> None:
    skipped_count = 1 if skipped else 0
    skipped_child = "<skipped />" if skipped else ""
    path.write_text(
        '<?xml version="1.0" encoding="utf-8"?>'
        '<testsuites name="pytest tests">'
        f'<testsuite name="pytest" errors="0" failures="0" skipped="{skipped_count}" tests="1">'
        f'<testcase classname="{classname}" name="{name}" time="0.1">{skipped_child}</testcase>'
        "</testsuite></testsuites>",
        encoding="utf-8",
    )


def _snapshot(fingerprint: str = "a" * 64) -> dict:
    return {
        "schema": SNAPSHOT_SCHEMA,
        "label": "synthetic",
        "composite_sha256": fingerprint,
        "code": {"fingerprint_sha256": "b" * 64},
        "source_pdfs": {"fingerprint_sha256": "c" * 64},
        "runtime": {"fingerprint_sha256": "d" * 64},
    }


def _node_records() -> list[dict]:
    records = []
    for case in EXPECTED_SOURCE_PDFS:
        records.append(
            {
                "schema": NODE_RESULT_SCHEMA,
                "case_id": case.case_id,
                "phase": "live",
                "node": case.node,
                "status": "passed",
                "exit_code": 0,
                "errors": [],
            }
        )
    for case in SAVED_CASES:
        records.append(
            {
                "schema": NODE_RESULT_SCHEMA,
                "case_id": case.case_id,
                "phase": "saved",
                "node": case.node,
                "status": "passed",
                "exit_code": 0,
                "errors": [],
            }
        )
    return records


def _community_and_semantic(case: SourcePdf) -> tuple[dict, dict]:
    datasets = []
    for name in (
        "credit_accounts",
        "credit_account_monthly_performance",
        "credit_agreements",
        "inquiries",
    ):
        datasets.append(
            {
                "name": name,
                "row_count": 0,
                "rows": [],
                "status": "complete",
                "completeness": {
                    "expected_row_count": 0,
                    "emitted_row_count": 0,
                    "omitted_row_count": 0,
                    "verified": True,
                    "basis": "synthetic",
                },
            }
        )
    source = {"name": case.filename, "sha256": f"sha256:{case.sha256}"}
    community = {"document": {"source_file": source}, "datasets": datasets}
    conservation = {
        "valid": True,
        "frozen_before_business_repair": True,
        "business_repair_can_mutate_conserved_plane": False,
        "raw_bundle_count": 1,
        "conserved_page_count": 1,
        "conserved_plane_sha256": "e" * 64,
    }
    semantic = {
        "source": {"file": source},
        "domain": {
            "facts": {
                "credit_extraction_audit": {
                    "page_topology": {
                        "valid": True,
                        "source_page_count": 1,
                        "logical_page_count": 1,
                        "ocr_used_for_topology": False,
                        "topology_frozen_before_reocr": True,
                        "corrected_evidence_conservation": conservation,
                    },
                    "ocr_correction": {"business_repair": {}},
                },
                "personal_detail_source_completeness_ledger": {},
                "personal_detail_account_month_closure": {},
            }
        },
    }
    return community, semantic


def test_parse_junit_requires_one_exact_clean_pass(tmp_path: Path) -> None:
    path = tmp_path / "result.xml"
    node = "tests/regression/example.py::test_example[param]"
    _write_junit(path, classname="tests.regression.example", name="test_example[param]")

    result = parse_junit(path, node)

    assert result["tests"] == 1
    assert result["skipped"] == 0


def test_parse_junit_rejects_skip_even_when_pytest_would_exit_zero(tmp_path: Path) -> None:
    path = tmp_path / "result.xml"
    _write_junit(path, classname="tests.regression.example", name="test_example", skipped=True)

    with pytest.raises(AuditError, match="not one clean pass"):
        parse_junit(path, "tests/regression/example.py::test_example")


def test_compare_snapshots_rejects_each_frozen_component_drift() -> None:
    baseline = _snapshot()
    observed = _snapshot()
    compare_snapshots(baseline, observed)
    observed["runtime"]["fingerprint_sha256"] = "f" * 64

    with pytest.raises(AuditError, match="runtime fingerprint"):
        compare_snapshots(baseline, observed)


def test_pdf_snapshot_rejects_extra_and_hash_changed_fixtures(tmp_path: Path) -> None:
    source = tmp_path / "one.pdf"
    source.write_bytes(b"signed source")
    expected = (SourcePdf("01", "one", "one", "one.pdf", sha256_file(source), "tests/x.py::test_x"),)
    snapshot = capture_pdf_snapshot(tmp_path, expected=expected)
    assert snapshot["files"][0]["sha256"] == expected[0].sha256

    (tmp_path / "extra.pdf").write_bytes(b"extra")
    with pytest.raises(AuditError, match="catalog mismatch"):
        capture_pdf_snapshot(tmp_path, expected=expected)
    (tmp_path / "extra.pdf").unlink()
    source.write_bytes(b"changed")
    with pytest.raises(AuditError, match="hash mismatch"):
        capture_pdf_snapshot(tmp_path, expected=expected)


def test_saved_pair_binds_source_schemas_counts_and_audit(tmp_path: Path) -> None:
    case = SourcePdf("01", "one", "one", "one.pdf", "a" * 64, "tests/x.py::test_x")
    community, semantic = _community_and_semantic(case)
    atomic_write_json(tmp_path / "one.community.json", community)
    atomic_write_json(tmp_path / "one.semantic.json", semantic)

    result = validate_saved_pair(tmp_path, case, schema_errors=lambda _name, _payload: [])

    assert result["schema_validations"] == {
        "community": True,
        "personal_credit_report_detailed": True,
        "community_semantic": True,
    }
    assert result["audit"]["conserved_evidence"]["conserved_plane_sha256"] == "e" * 64

    community["document"]["source_file"]["sha256"] = "sha256:" + "f" * 64
    atomic_write_json(tmp_path / "one.community.json", community)
    with pytest.raises(AuditError, match="not PDF-bound"):
        validate_saved_pair(tmp_path, case, schema_errors=lambda _name, _payload: [])


def test_reduced_contract_preserves_remaining_case_ids_and_hashes() -> None:
    assert [case.case_id for case in EXPECTED_SOURCE_PDFS] == [
        "01",
        "02",
        "03",
        "04",
        "05",
        "07",
    ]
    assert [case.case_id for case in SAVED_CASES] == [
        "08",
        "09",
        "10",
        "11",
        "12",
        "13",
        "15",
    ]
    assert {case.filename: case.sha256 for case in EXPECTED_SOURCE_PDFS} == {
        "余泽熙7.15征信.pdf": "efbc80bd09546cb96de1d7da531596aada793f13c55fb07e5d09420f069caf0b",
        "杨松林个人征信24.7.29.pdf": "42ff7181ea1594ae1fbe85bd997d23f77716c5b7a526e031d15e03eaf1a4e117",
        "林岚挺征信.pdf": "a44515a83ae226d19008437ac6a757fa58dabc14d3f1fb5ac9a01c4441cdfdd2",
        "叶永燕征信.pdf": "7c986d8f07021027836a9df27704afd4746b714ae3e40e7ecf4eace099ea5193",
        "王根镇征信.pdf": "eb6e963fefab972c1d74147be4943741233d912c06f57e9d60d2316dd62aecd2",
        "黄圣辉_个人详版征信报告.pdf": "8d5482f9e354d2d9b02614942283c0c62fce56b49228e8c5b3069e5aaf85f0a5",
    }
    assert all(case.slug != "saved-cao-population" for case in SAVED_CASES)


def test_result_matrix_requires_exact_unique_six_plus_seven() -> None:
    records = _node_records()
    live, saved = validate_result_records(records)
    assert len(live) == 6
    assert len(saved) == 7

    with pytest.raises(AuditError, match="catalog mismatch"):
        validate_result_records(records[:-1])
    with pytest.raises(AuditError, match="Duplicate"):
        validate_result_records([*records, records[0]])


def test_failed_node_writes_diagnostic_result_but_cannot_pass(tmp_path: Path) -> None:
    audit_dir = tmp_path / "audit"
    audit_dir.mkdir()
    (audit_dir / "node-results").mkdir()
    stdout = audit_dir / "08-saved-catalog.stdout.log"
    junit = audit_dir / "08-saved-catalog.junit.xml"
    stdout.write_text("skipped\n", encoding="utf-8")
    _write_junit(
        junit,
        classname="tests.regression.test_personal_detail_ocr_quality_private",
        name="test_saved_five_community_dataset_catalog",
        skipped=True,
    )
    baseline = audit_dir / "baseline.json"
    checkpoint = audit_dir / "checkpoint.json"
    atomic_write_json(baseline, _snapshot())
    atomic_write_json(checkpoint, _snapshot())
    output = audit_dir / "node-results" / "08-saved-catalog.json"

    with pytest.raises(AuditError, match="Node validation failed"):
        validate_node(
            audit_dir=audit_dir,
            fixture_dir=tmp_path,
            case_id="08",
            junit_path=junit,
            stdout_path=stdout,
            checkpoint_path=checkpoint,
            baseline_path=baseline,
            output_path=output,
            exit_code=0,
            started_utc="2026-08-24T00:00:00+00:00",
            completed_utc="2026-08-24T00:00:01+00:00",
        )
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "failed"
    assert (audit_dir / "result-matrix.partial.json").is_file()


def test_completed_manifest_detects_artifact_tampering(tmp_path: Path) -> None:
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    atomic_write_json(snapshots / "00-pre.json", _snapshot())
    atomic_write_json(snapshots / "99-end.json", _snapshot())
    evidence = tmp_path / "evidence.txt"
    evidence.write_text("sealed", encoding="utf-8")
    node_records = _node_records()
    live_count = len(EXPECTED_SOURCE_PDFS)
    matrix = {
        "schema": MATRIX_SCHEMA,
        "complete": True,
        "live_results": node_records[:live_count],
        "saved_results": node_records[live_count:],
    }
    atomic_write_json(tmp_path / "result-matrix.json", matrix)
    artifact_paths = [
        evidence,
        tmp_path / "result-matrix.json",
        snapshots / "00-pre.json",
        snapshots / "99-end.json",
    ]
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "artifacts": [file_record(path, relative_to=tmp_path) for path in artifact_paths],
    }
    atomic_write_json(tmp_path / "manifest.json", manifest)
    marker = {
        "schema": COMPLETION_SCHEMA,
        "status": "complete",
        "manifest_sha256": sha256_file(tmp_path / "manifest.json"),
        "result_matrix_sha256": sha256_file(tmp_path / "result-matrix.json"),
        "baseline_composite_sha256": "a" * 64,
        "end_composite_sha256": "a" * 64,
        "live_expected": len(EXPECTED_SOURCE_PDFS),
        "live_passed": len(EXPECTED_SOURCE_PDFS),
        "saved_expected": len(SAVED_CASES),
        "saved_passed": len(SAVED_CASES),
    }
    atomic_write_json(tmp_path / "AUDIT_COMPLETE.json", marker)

    assert verify_completed_audit(tmp_path)["status"] == "complete"
    evidence.write_text("tampered", encoding="utf-8")
    with pytest.raises(AuditError, match="modified"):
        verify_completed_audit(tmp_path)


def test_failed_finalization_never_creates_completion_marker(tmp_path: Path) -> None:
    with pytest.raises(AuditError):
        finalize_audit(tmp_path, tmp_path)
    assert not (tmp_path / "AUDIT_COMPLETE.json").exists()
