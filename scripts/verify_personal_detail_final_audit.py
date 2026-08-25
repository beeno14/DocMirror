#!/usr/bin/env python3
"""Fail-closed attestation for the private personal-detail OCR audit.

The PowerShell launcher owns process scheduling.  This module owns the immutable
7-live + 8-saved contract, snapshot comparison, persisted-output validation,
JUnit interpretation, result-matrix construction, and completion attestation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import locale
import os
import platform
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

SNAPSHOT_SCHEMA = "docmirror.personal_detail.final_audit_snapshot.v1"
CONTRACT_SCHEMA = "docmirror.personal_detail.final_audit_contract.v1"
NODE_RESULT_SCHEMA = "docmirror.personal_detail.final_audit_node_result.v1"
MATRIX_SCHEMA = "docmirror.personal_detail.final_audit_result_matrix.v1"
MANIFEST_SCHEMA = "docmirror.personal_detail.final_audit_manifest.v1"
COMPLETION_SCHEMA = "docmirror.personal_detail.final_audit_completion.v1"


class AuditError(RuntimeError):
    """Raised when an attestation requirement is not proven."""


@dataclass(frozen=True)
class SourcePdf:
    case_id: str
    slug: str
    stem: str
    filename: str
    sha256: str
    node: str


@dataclass(frozen=True)
class SavedCase:
    case_id: str
    slug: str
    node: str


EXPECTED_SOURCE_PDFS: tuple[SourcePdf, ...] = (
    SourcePdf(
        "01",
        "yu-zexi",
        "余泽熙7.15征信",
        "余泽熙7.15征信.pdf",
        "efbc80bd09546cb96de1d7da531596aada793f13c55fb07e5d09420f069caf0b",
        r"tests/regression/test_personal_detail_ocr_quality_private.py::test_personal_detail_ocr_correction_invariants[\u4f59\u6cfd\u71997.15\u5f81\u4fe1.pdf]",
    ),
    SourcePdf(
        "02",
        "yang-songlin",
        "杨松林个人征信24.7.29",
        "杨松林个人征信24.7.29.pdf",
        "42ff7181ea1594ae1fbe85bd997d23f77716c5b7a526e031d15e03eaf1a4e117",
        r"tests/regression/test_personal_detail_ocr_quality_private.py::test_personal_detail_ocr_correction_invariants[\u6768\u677e\u6797\u4e2a\u4eba\u5f81\u4fe124.7.29.pdf]",
    ),
    SourcePdf(
        "03",
        "lin-lanting",
        "林岚挺征信",
        "林岚挺征信.pdf",
        "a44515a83ae226d19008437ac6a757fa58dabc14d3f1fb5ac9a01c4441cdfdd2",
        r"tests/regression/test_personal_detail_ocr_quality_private.py::test_personal_detail_ocr_correction_invariants[\u6797\u5c9a\u633a\u5f81\u4fe1.pdf]",
    ),
    SourcePdf(
        "04",
        "ye-yongyan",
        "叶永燕征信",
        "叶永燕征信.pdf",
        "7c986d8f07021027836a9df27704afd4746b714ae3e40e7ecf4eace099ea5193",
        r"tests/regression/test_personal_detail_ocr_quality_private.py::test_personal_detail_ocr_correction_invariants[\u53f6\u6c38\u71d5\u5f81\u4fe1.pdf]",
    ),
    SourcePdf(
        "05",
        "wang-genzhen",
        "王根镇征信",
        "王根镇征信.pdf",
        "eb6e963fefab972c1d74147be4943741233d912c06f57e9d60d2316dd62aecd2",
        r"tests/regression/test_personal_detail_ocr_quality_private.py::test_personal_detail_ocr_correction_invariants[\u738b\u6839\u9547\u5f81\u4fe1.pdf]",
    ),
    SourcePdf(
        "06",
        "cao-moyan",
        "曹末艳-征信",
        "曹末艳-征信.pdf",
        "1a33a80b4b818640105e94488db00f1656d8f3e86004caa92f0e98163d523eee",
        r"tests/regression/test_personal_detail_ocr_quality_private.py::test_personal_detail_ocr_correction_invariants[\u66f9\u672b\u8273-\u5f81\u4fe1.pdf]",
    ),
    SourcePdf(
        "07",
        "huang-shenghui",
        "黄圣辉_个人详版征信报告",
        "黄圣辉_个人详版征信报告.pdf",
        "8d5482f9e354d2d9b02614942283c0c62fce56b49228e8c5b3069e5aaf85f0a5",
        r"tests/regression/test_personal_detail_ocr_quality_private.py::test_personal_detail_ocr_correction_invariants[\u9ec4\u5723\u8f89_\u4e2a\u4eba\u8be6\u7248\u5f81\u4fe1\u62a5\u544a.pdf]",
    ),
)

SAVED_CASES: tuple[SavedCase, ...] = (
    SavedCase(
        "08",
        "saved-catalog",
        "tests/regression/test_personal_detail_ocr_quality_private.py::test_saved_five_community_dataset_catalog",
    ),
    SavedCase(
        "09",
        "saved-lin-semantic",
        "tests/regression/test_personal_detail_ocr_quality_private.py::test_saved_lin_semantic_account_fragment_oracle",
    ),
    SavedCase(
        "10",
        "saved-ye-population",
        "tests/regression/test_personal_detail_ocr_quality_private.py::test_saved_ye_population_and_month_geometry_oracle",
    ),
    SavedCase(
        "11",
        "saved-lin-inquiries",
        "tests/regression/test_personal_detail_lin_inquiry_saved_private.py::test_saved_lin_community_inquiry_lifecycle_and_normalized_fields",
    ),
    SavedCase(
        "12",
        "saved-wang-population",
        r"tests/regression/test_personal_detail_expanded_saved_population_private.py::test_saved_expanded_personal_detail_population_acceptance[\u738b\u6839\u9547\u5f81\u4fe1]",
    ),
    SavedCase(
        "13",
        "saved-huang-population",
        r"tests/regression/test_personal_detail_expanded_saved_population_private.py::test_saved_expanded_personal_detail_population_acceptance[\u9ec4\u5723\u8f89_\u4e2a\u4eba\u8be6\u7248\u5f81\u4fe1\u62a5\u544a]",
    ),
    SavedCase(
        "14",
        "saved-cao-population",
        r"tests/regression/test_personal_detail_expanded_saved_population_private.py::test_saved_expanded_personal_detail_population_acceptance[\u66f9\u672b\u8273-\u5f81\u4fe1]",
    ),
    SavedCase(
        "15",
        "saved-ye-schema",
        "tests/regression/test_personal_detail_schema_contract_ye_yongyan_private.py::test_ye_yongyan_personal_detail_schema_contract",
    ),
)

KEY_DATASETS = (
    "credit_accounts",
    "credit_account_monthly_performance",
    "credit_agreements",
    "inquiries",
)

RUNTIME_ENV_NAMES = frozenset(
    {
        "DOCMIRROR_ALLOW_NETWORK",
        "DOCMIRROR_MAX_PAGE_CONCURRENCY",
        "DOCMIRROR_MAX_PROCESS_WORKERS",
        "DOCMIRROR_PAGE_EXECUTOR",
        "DOCMIRROR_PERSONAL_DETAIL_AUDIT_DIR",
        "DOCMIRROR_PERSONAL_DETAIL_EXPANDED_SAVED_AUDIT_DIR",
        "DOCMIRROR_PERSONAL_DETAIL_FIXTURE_DIR",
        "DOCMIRROR_PERSONAL_DETAIL_PAGE_OCR",
        "DOCMIRROR_PERSONAL_DETAIL_SAVED_LIN_AUDIT_DIR",
        "DOCMIRROR_PERSONAL_DETAIL_SAVED_YE_AUDIT_DIR",
        "DOCMIRROR_PRIVACY_MODE",
        "CUDA_VISIBLE_DEVICES",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "PYTHONHASHSEED",
        "PYTHONUTF8",
        "PYTEST_ADDOPTS",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
    }
)

FROZEN_TEST_FILES = (
    "tests/conftest.py",
    "tests/regression/conftest.py",
    "tests/regression/test_personal_detail_ocr_quality_private.py",
    "tests/regression/test_personal_detail_lin_inquiry_saved_private.py",
    "tests/regression/test_personal_detail_expanded_saved_population_private.py",
    "tests/regression/test_personal_detail_schema_contract_ye_yongyan_private.py",
)

FROZEN_SUPPORT_FILES = (
    "pyproject.toml",
    "requirements.txt",
    "scripts/verify_personal_detail_final_audit.py",
    "artifacts/personal_detail_100pct_iteration_20260820/run_final_ocr.ps1",
    "artifacts/personal_detail_100pct_iteration_20260813/huang_month_truth/ledger.jsonl",
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AuditError(f"Invalid JSON {path}: {exc}") from exc


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    data = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    try:
        temporary.write_text(data, encoding="utf-8", newline="\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def file_record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    label = resolved.relative_to(relative_to.resolve()).as_posix() if relative_to else str(resolved)
    return {"path": label, "bytes": resolved.stat().st_size, "sha256": sha256_file(resolved)}


def contract_payload() -> dict[str, Any]:
    return {
        "schema": CONTRACT_SCHEMA,
        "live_case_count": len(EXPECTED_SOURCE_PDFS),
        "saved_case_count": len(SAVED_CASES),
        "live_cases": [asdict(case) for case in EXPECTED_SOURCE_PDFS],
        "saved_cases": [asdict(case) for case in SAVED_CASES],
    }


def _is_frozen_source_file(path: Path) -> bool:
    return path.is_file() and "__pycache__" not in path.parts and path.suffix.lower() not in {".pyc", ".pyo"}


def frozen_code_paths(workspace: Path) -> list[Path]:
    workspace = workspace.resolve()
    paths = [
        path
        for root in (workspace / "docmirror", workspace / "scripts")
        for path in root.rglob("*")
        if _is_frozen_source_file(path)
    ]
    for relative in (*FROZEN_TEST_FILES, *FROZEN_SUPPORT_FILES):
        path = workspace / relative
        if not path.is_file():
            raise AuditError(f"Frozen input is missing: {path}")
        paths.append(path)
    return sorted(set(paths), key=lambda path: path.resolve().relative_to(workspace).as_posix())


def _git_context(workspace: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        try:
            result = subprocess.run(
                ["git", *arguments],
                cwd=workspace,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise AuditError(f"Git context failed: {arguments!r}: {exc}") from exc
        return result.stdout.strip()

    return {
        "head": run("rev-parse", "HEAD"),
        # Informational only: content fingerprints, not status text, seal the run.
        "status_porcelain": run("status", "--porcelain=v1", "--untracked-files=all").splitlines(),
    }


def capture_code_snapshot(workspace: Path) -> dict[str, Any]:
    workspace = workspace.resolve()
    files = [file_record(path, relative_to=workspace) for path in frozen_code_paths(workspace)]
    git = _git_context(workspace)
    fingerprint_body = {"git_head": git["head"], "files": files}
    return {"fingerprint_sha256": canonical_json_sha256(fingerprint_body), "git": git, "files": files}


def capture_pdf_snapshot(
    fixture_dir: Path,
    *,
    expected: Sequence[SourcePdf] = EXPECTED_SOURCE_PDFS,
) -> dict[str, Any]:
    fixture_dir = fixture_dir.resolve()
    if not fixture_dir.is_dir():
        raise AuditError(f"Fixture directory is missing: {fixture_dir}")
    expected_names = {case.filename for case in expected}
    actual_names = {path.name for path in fixture_dir.glob("*.pdf") if path.is_file()}
    if actual_names != expected_names:
        raise AuditError(
            "Fixture PDF catalog mismatch: "
            f"missing={sorted(expected_names - actual_names)!r}, extra={sorted(actual_names - expected_names)!r}"
        )
    files: list[dict[str, Any]] = []
    for case in expected:
        path = fixture_dir / case.filename
        record = file_record(path)
        record.update({"case_id": case.case_id, "name": case.filename})
        if record["sha256"] != case.sha256:
            raise AuditError(f"Source PDF hash mismatch for {case.filename}: {record['sha256']} != {case.sha256}")
        files.append(record)
    fingerprint_body = [{key: row[key] for key in ("case_id", "name", "bytes", "sha256")} for row in files]
    return {"fingerprint_sha256": canonical_json_sha256(fingerprint_body), "files": files}


def _distribution_inventory() -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for distribution in importlib.metadata.distributions():
        name = str(distribution.metadata.get("Name") or "").strip()
        if not name:
            continue
        record_text = distribution.read_text("RECORD")
        direct_url = distribution.read_text("direct_url.json")
        inventory.append(
            {
                "name": name.casefold(),
                "version": distribution.version,
                "record_sha256": sha256_bytes(record_text.encode("utf-8")) if record_text is not None else None,
                "direct_url_sha256": sha256_bytes(direct_url.encode("utf-8")) if direct_url is not None else None,
            }
        )
    return sorted(inventory, key=lambda row: (row["name"], row["version"], str(row["record_sha256"])))


def _rapidocr_model_inventory() -> list[dict[str, Any]]:
    spec = importlib.util.find_spec("rapidocr_onnxruntime")
    if spec is None:
        raise AuditError("rapidocr_onnxruntime is unavailable")
    roots = [Path(item).resolve() for item in spec.submodule_search_locations or ()]
    if not roots and spec.origin:
        roots = [Path(spec.origin).resolve().parent]
    files: list[dict[str, Any]] = []
    for root in roots:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".onnx", ".yaml", ".yml"}:
                files.append(
                    {
                        "path": f"{root.name}/{path.relative_to(root).as_posix()}",
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
    if not any(row["path"].endswith(".onnx") for row in files):
        raise AuditError("RapidOCR model inventory contains no ONNX model")
    return sorted(files, key=lambda row: row["path"])


def _onnx_providers() -> list[str]:
    try:
        import onnxruntime
    except Exception as exc:  # pragma: no cover - environment failure detail
        raise AuditError(f"onnxruntime is unavailable: {exc}") from exc
    return list(onnxruntime.get_available_providers())


def capture_runtime_snapshot() -> dict[str, Any]:
    unexpected_docmirror = sorted(
        name for name in os.environ if name.startswith("DOCMIRROR_") and name not in RUNTIME_ENV_NAMES
    )
    if unexpected_docmirror:
        raise AuditError(f"Unexpected DOCMIRROR environment variables: {unexpected_docmirror!r}")
    executable = Path(sys.executable).resolve()
    environment = {name: os.environ.get(name) for name in sorted(RUNTIME_ENV_NAMES)}
    details = {
        "python": {
            "executable": str(executable),
            "executable_bytes": executable.stat().st_size,
            "executable_sha256": sha256_file(executable),
            "version": sys.version,
            "implementation": platform.python_implementation(),
            "cache_tag": sys.implementation.cache_tag,
        },
        "platform": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
        },
        "locale": {
            "encoding": locale.getencoding(),
            "preferred_encoding": locale.getpreferredencoding(False),
            "locale": list(locale.getlocale()),
            "timezone": list(time.tzname),
        },
        "environment": environment,
        "path_sha256": sha256_bytes(os.environ.get("PATH", "").encode("utf-8")),
        "distributions": _distribution_inventory(),
        "rapidocr_models": _rapidocr_model_inventory(),
        "onnx_available_providers": _onnx_providers(),
    }
    return {"fingerprint_sha256": canonical_json_sha256(details), **details}


def capture_snapshot(workspace: Path, fixture_dir: Path, *, label: str) -> dict[str, Any]:
    code = capture_code_snapshot(workspace)
    source_pdfs = capture_pdf_snapshot(fixture_dir)
    runtime = capture_runtime_snapshot()
    fingerprint_body = {
        "code": code["fingerprint_sha256"],
        "source_pdfs": source_pdfs["fingerprint_sha256"],
        "runtime": runtime["fingerprint_sha256"],
    }
    return {
        "schema": SNAPSHOT_SCHEMA,
        "label": label,
        "captured_utc": utc_now(),
        "composite_sha256": canonical_json_sha256(fingerprint_body),
        "code": code,
        "source_pdfs": source_pdfs,
        "runtime": runtime,
    }


def compare_snapshots(baseline: Mapping[str, Any], observed: Mapping[str, Any]) -> None:
    defects = []
    if baseline.get("schema") != SNAPSHOT_SCHEMA or observed.get("schema") != SNAPSHOT_SCHEMA:
        defects.append("snapshot schema mismatch")
    for component in ("code", "source_pdfs", "runtime"):
        expected = (baseline.get(component) or {}).get("fingerprint_sha256")
        actual = (observed.get(component) or {}).get("fingerprint_sha256")
        if expected != actual:
            defects.append(f"{component} fingerprint {actual!r} != baseline {expected!r}")
    if baseline.get("composite_sha256") != observed.get("composite_sha256"):
        defects.append(
            f"composite fingerprint {observed.get('composite_sha256')!r} "
            f"!= baseline {baseline.get('composite_sha256')!r}"
        )
    if defects:
        raise AuditError("Frozen snapshot drift: " + "; ".join(defects))


def parse_junit(path: Path, expected_node: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise AuditError(f"JUnit artifact is missing or empty: {path}")
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise AuditError(f"Invalid JUnit XML {path}: {exc}") from exc
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    testcases = list(root.iter("testcase"))
    if len(testcases) != 1:
        raise AuditError(f"JUnit must contain exactly one testcase, observed {len(testcases)}: {path}")
    counters = {
        key: sum(int(suite.get(key, "0")) for suite in suites) for key in ("tests", "errors", "failures", "skipped")
    }
    if counters != {"tests": 1, "errors": 0, "failures": 0, "skipped": 0}:
        raise AuditError(f"JUnit is not one clean pass: {counters!r}: {path}")
    testcase = testcases[0]
    if any(testcase.find(tag) is not None for tag in ("failure", "error", "skipped")):
        raise AuditError(f"JUnit testcase contains a failure/error/skip child: {path}")
    try:
        module_path, expected_name = expected_node.split("::", 1)
    except ValueError as exc:
        raise AuditError(f"Invalid expected node id: {expected_node!r}") from exc
    expected_classname = module_path.removesuffix(".py").replace("/", ".").replace("\\", ".")
    observed_identity = (testcase.get("classname"), testcase.get("name"))
    expected_identity = (expected_classname, expected_name)
    if observed_identity != expected_identity:
        raise AuditError(f"JUnit testcase identity {observed_identity!r} != {expected_identity!r}")
    return {
        "tests": 1,
        "errors": 0,
        "failures": 0,
        "skipped": 0,
        "classname": observed_identity[0],
        "name": observed_identity[1],
        "time_seconds": testcase.get("time"),
    }


def _schema_errors(schema_name: str, payload: dict[str, Any]) -> list[str]:
    from docmirror.models.schemas.registry import validate_projection_payload

    result = validate_projection_payload(schema_name, payload)
    return [] if result.valid else [str(error) for error in result.errors]


def _source_metadata(payload: dict[str, Any], semantic: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    community_source = (payload.get("document") or {}).get("source_file") or {}
    semantic_source = (semantic.get("source") or {}).get("file") or {}
    return community_source, semantic_source


def _dataset_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    datasets: dict[str, dict[str, Any]] = {}
    for dataset in payload.get("datasets") or ():
        if not isinstance(dataset, dict) or not dataset.get("name"):
            raise AuditError("Community payload has a malformed dataset entry")
        name = str(dataset["name"])
        if name in datasets:
            raise AuditError(f"Community payload has duplicate dataset {name!r}")
        datasets[name] = dataset
    return datasets


def _validate_dataset_counts(datasets: Mapping[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    for name, dataset in datasets.items():
        rows = dataset.get("rows") or []
        row_count = dataset.get("row_count")
        if type(row_count) is not int or row_count < 0 or row_count != len(rows):
            raise AuditError(f"{name}: row_count={row_count!r}, actual rows={len(rows)}")
    summaries: dict[str, dict[str, Any]] = {}
    for name in KEY_DATASETS:
        if name not in datasets:
            raise AuditError(f"Required source-audited dataset is missing: {name}")
        dataset = datasets[name]
        completeness = dataset.get("completeness") or {}
        expected = completeness.get("expected_row_count")
        emitted = completeness.get("emitted_row_count")
        omitted = completeness.get("omitted_row_count")
        if not all(type(value) is int and value >= 0 for value in (expected, emitted, omitted)):
            raise AuditError(f"{name}: invalid completeness counts {completeness!r}")
        if emitted != dataset["row_count"] or expected != emitted + omitted:
            raise AuditError(
                f"{name}: non-conserving completeness expected={expected}, emitted={emitted}, "
                f"omitted={omitted}, row_count={dataset['row_count']}"
            )
        summaries[name] = {
            "status": dataset.get("status"),
            "row_count": dataset["row_count"],
            "expected_row_count": expected,
            "emitted_row_count": emitted,
            "omitted_row_count": omitted,
            "verified": completeness.get("verified"),
            "basis": completeness.get("basis"),
        }
    return summaries


def _extract_audit_summary(semantic: dict[str, Any]) -> dict[str, Any]:
    facts = (semantic.get("domain") or {}).get("facts") or {}
    extraction = facts.get("credit_extraction_audit") or {}
    topology = extraction.get("page_topology") or {}
    conservation = topology.get("corrected_evidence_conservation") or {}
    if topology.get("valid") is not True:
        raise AuditError("Semantic page-topology audit is not valid")
    if topology.get("ocr_used_for_topology") is not False:
        raise AuditError("Semantic page topology was influenced by repair OCR")
    if topology.get("topology_frozen_before_reocr") is not True:
        raise AuditError("Semantic page topology was not frozen before repair OCR")
    if conservation.get("valid") is not True:
        raise AuditError("Semantic corrected-evidence conservation is not valid")
    if conservation.get("frozen_before_business_repair") is not True:
        raise AuditError("Semantic corrected evidence was not frozen before business repair")
    if conservation.get("business_repair_can_mutate_conserved_plane") is not False:
        raise AuditError("Semantic business repair can mutate the conserved plane")
    plane_hash = str(conservation.get("conserved_plane_sha256") or "")
    if len(plane_hash) != 64:
        raise AuditError("Semantic conserved-plane SHA-256 is missing")
    correction = extraction.get("ocr_correction") or {}
    business_repair = correction.get("business_repair") or {}
    return {
        "page_topology": {
            "valid": True,
            "source_page_count": topology.get("source_page_count"),
            "logical_page_count": topology.get("logical_page_count"),
            "ocr_used_for_topology": False,
            "topology_frozen_before_reocr": True,
        },
        "conserved_evidence": {
            "valid": True,
            "raw_bundle_count": conservation.get("raw_bundle_count"),
            "conserved_page_count": conservation.get("conserved_page_count"),
            "conserved_plane_sha256": plane_hash,
        },
        "ocr_correction": {
            "decision_count": correction.get("decision_count"),
            "applied_count": correction.get("applied_count"),
            "suggested_count": correction.get("suggested_count"),
            "page_reocr_page_count": correction.get("page_reocr_page_count"),
            "page_reocr_engine_invocation_count": correction.get("page_reocr_engine_invocation_count"),
            "one_shot_per_page_enforced": correction.get("one_shot_per_page_enforced"),
            "max_ocr_invocations_per_page": correction.get("max_ocr_invocations_per_page"),
            "page_reocr_failure_count": len(correction.get("page_reocr_failures") or ()),
            "topology_ocr_requests": business_repair.get("topology_ocr_requests"),
            "field_triggered_ocr_requests": business_repair.get("field_triggered_ocr_requests"),
        },
        "source_completeness_ledger": facts.get("personal_detail_source_completeness_ledger"),
        "account_month_closure": facts.get("personal_detail_account_month_closure"),
    }


def validate_saved_pair(
    audit_dir: Path,
    case: SourcePdf,
    *,
    schema_errors: Callable[[str, dict[str, Any]], list[str]] = _schema_errors,
) -> dict[str, Any]:
    community_path = audit_dir / f"{case.stem}.community.json"
    semantic_path = audit_dir / f"{case.stem}.semantic.json"
    for path in (community_path, semantic_path):
        if not path.is_file() or path.stat().st_size <= 0:
            raise AuditError(f"Required saved artifact is missing or empty: {path}")
    community = read_json(community_path)
    semantic = read_json(semantic_path)
    if not isinstance(community, dict) or not isinstance(semantic, dict):
        raise AuditError(f"Saved pair does not contain JSON objects for {case.stem}")
    validations = {
        schema_name: schema_errors(schema_name, payload)
        for schema_name, payload in (
            ("community", community),
            ("personal_credit_report_detailed", community),
            ("community_semantic", semantic),
        )
    }
    invalid = {name: errors for name, errors in validations.items() if errors}
    if invalid:
        raise AuditError(f"Schema validation failed for {case.stem}: {invalid!r}")
    expected_hash = f"sha256:{case.sha256}"
    for label, source in zip(("Community", "Semantic"), _source_metadata(community, semantic), strict=True):
        if source.get("name") != case.filename or source.get("sha256") != expected_hash:
            raise AuditError(f"{case.stem}: {label} source metadata is not PDF-bound: {source!r}")
    datasets = _dataset_map(community)
    dataset_summaries = _validate_dataset_counts(datasets)
    return {
        "case_id": case.case_id,
        "pdf": case.filename,
        "pdf_sha256": case.sha256,
        "community": file_record(community_path, relative_to=audit_dir),
        "semantic": file_record(semantic_path, relative_to=audit_dir),
        "schema_validations": {name: True for name in validations},
        "datasets": dataset_summaries,
        "audit": _extract_audit_summary(semantic),
    }


def _all_cases() -> dict[str, tuple[str, str, SourcePdf | None]]:
    rows: dict[str, tuple[str, str, SourcePdf | None]] = {}
    for case in EXPECTED_SOURCE_PDFS:
        rows[case.case_id] = ("live", case.node, case)
    for case in SAVED_CASES:
        rows[case.case_id] = ("saved", case.node, None)
    return rows


def _partial_matrix(audit_dir: Path) -> dict[str, Any]:
    results = []
    results_dir = audit_dir / "node-results"
    if results_dir.is_dir():
        for path in sorted(results_dir.glob("*.json")):
            value = read_json(path)
            if isinstance(value, dict):
                results.append(value)
    return {
        "schema": MATRIX_SCHEMA,
        "updated_utc": utc_now(),
        "complete": False,
        "live_results": [row for row in results if row.get("phase") == "live"],
        "saved_results": [row for row in results if row.get("phase") == "saved"],
    }


def validate_node(
    *,
    audit_dir: Path,
    fixture_dir: Path,
    case_id: str,
    junit_path: Path,
    stdout_path: Path,
    checkpoint_path: Path,
    baseline_path: Path,
    output_path: Path,
    exit_code: int,
    started_utc: str,
    completed_utc: str,
) -> dict[str, Any]:
    cases = _all_cases()
    if case_id not in cases:
        raise AuditError(f"Unknown contract case id: {case_id}")
    phase, node, live_case = cases[case_id]
    errors: list[str] = []
    junit: dict[str, Any] | None = None
    report: dict[str, Any] | None = None
    checkpoint: dict[str, Any] | None = None
    if exit_code != 0:
        errors.append(f"pytest exit code={exit_code}")
    try:
        junit = parse_junit(junit_path, node)
    except AuditError as exc:
        errors.append(str(exc))
    try:
        if not stdout_path.is_file() or stdout_path.stat().st_size <= 0:
            raise AuditError(f"Stdout artifact is missing or empty: {stdout_path}")
    except OSError as exc:
        errors.append(str(exc))
    try:
        baseline = read_json(baseline_path)
        checkpoint = read_json(checkpoint_path)
        compare_snapshots(baseline, checkpoint)
    except AuditError as exc:
        errors.append(str(exc))
    if phase == "live" and live_case is not None:
        try:
            source_pdf = fixture_dir / live_case.filename
            if sha256_file(source_pdf) != live_case.sha256:
                raise AuditError(f"Live source PDF drifted before persisted-pair validation: {live_case.filename}")
            report = validate_saved_pair(audit_dir, live_case)
        except (AuditError, OSError) as exc:
            errors.append(str(exc))
    record: dict[str, Any] = {
        "schema": NODE_RESULT_SCHEMA,
        "case_id": case_id,
        "phase": phase,
        "node": node,
        "status": "failed" if errors else "passed",
        "started_utc": started_utc,
        "completed_utc": completed_utc,
        "exit_code": exit_code,
        "errors": errors,
        "junit": junit,
        "junit_file": file_record(junit_path, relative_to=audit_dir) if junit_path.is_file() else None,
        "stdout_file": file_record(stdout_path, relative_to=audit_dir) if stdout_path.is_file() else None,
        "checkpoint_file": (file_record(checkpoint_path, relative_to=audit_dir) if checkpoint_path.is_file() else None),
        "checkpoint_composite_sha256": checkpoint.get("composite_sha256") if checkpoint else None,
        "report": report,
    }
    atomic_write_json(output_path, record)
    atomic_write_json(audit_dir / "result-matrix.partial.json", _partial_matrix(audit_dir))
    if errors:
        raise AuditError(f"Node validation failed for {case_id}: " + "; ".join(errors))
    return record


def validate_result_records(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    expected = _all_cases()
    by_id: dict[str, Mapping[str, Any]] = {}
    for record in records:
        case_id = str(record.get("case_id") or "")
        if case_id in by_id:
            raise AuditError(f"Duplicate node result case id: {case_id}")
        by_id[case_id] = record
    if set(by_id) != set(expected):
        raise AuditError(
            f"Node-result catalog mismatch: missing={sorted(set(expected) - set(by_id))!r}, "
            f"extra={sorted(set(by_id) - set(expected))!r}"
        )
    for case_id, (phase, node, _) in expected.items():
        record = by_id[case_id]
        if record.get("schema") != NODE_RESULT_SCHEMA:
            raise AuditError(f"{case_id}: node-result schema mismatch")
        if record.get("phase") != phase or record.get("node") != node or record.get("status") != "passed":
            raise AuditError(f"{case_id}: node result does not prove an exact pass")
        if record.get("exit_code") != 0 or record.get("errors"):
            raise AuditError(f"{case_id}: node result carries an exit/error defect")
    live = [by_id[case.case_id] for case in EXPECTED_SOURCE_PDFS]
    saved = [by_id[case.case_id] for case in SAVED_CASES]
    return live, saved


def _expected_evidence_catalog() -> set[str]:
    paths = {
        "contract.json",
        "freeze-context.json",
        "result-matrix.partial.json",
        "result-matrix.json",
    }
    for case in (*EXPECTED_SOURCE_PDFS, *SAVED_CASES):
        paths.add(f"{case.case_id}-{case.slug}.junit.xml")
        paths.add(f"{case.case_id}-{case.slug}.stdout.log")
        paths.add(f"node-results/{case.case_id}-{case.slug}.json")
        paths.add(f"snapshots/{case.case_id}-before-{case.slug}.json")
    for case in EXPECTED_SOURCE_PDFS:
        paths.add(f"{case.stem}.community.json")
        paths.add(f"{case.stem}.semantic.json")
    paths.update({"snapshots/00-pre.json", "snapshots/99-end.json"})
    return paths


def _current_evidence_catalog(audit_dir: Path, *, include_attestation: bool) -> set[str]:
    excluded = {"manifest.json", "AUDIT_COMPLETE.json"} if include_attestation else set()
    return {
        path.relative_to(audit_dir).as_posix()
        for path in audit_dir.rglob("*")
        if path.is_file() and path.name not in excluded and not path.name.endswith(".tmp")
    }


def _load_node_results(audit_dir: Path) -> list[dict[str, Any]]:
    directory = audit_dir / "node-results"
    return [read_json(path) for path in sorted(directory.glob("*.json"))] if directory.is_dir() else []


def _assert_checkpoint_catalog(audit_dir: Path, baseline: Mapping[str, Any]) -> None:
    expected = {"00-pre.json", "99-end.json"}
    expected.update(f"{case.case_id}-before-{case.slug}.json" for case in (*EXPECTED_SOURCE_PDFS, *SAVED_CASES))
    directory = audit_dir / "snapshots"
    actual = {path.name for path in directory.glob("*.json")} if directory.is_dir() else set()
    if actual != expected:
        raise AuditError(
            f"Snapshot catalog mismatch: missing={sorted(expected - actual)!r}, extra={sorted(actual - expected)!r}"
        )
    for name in sorted(expected - {"00-pre.json"}):
        compare_snapshots(baseline, read_json(directory / name))


def _artifact_manifest(audit_dir: Path) -> list[dict[str, Any]]:
    return [
        file_record(path, relative_to=audit_dir)
        for path in sorted(
            (
                path
                for path in audit_dir.rglob("*")
                if path.is_file() and path.name not in {"manifest.json", "AUDIT_COMPLETE.json"}
            ),
            key=lambda path: path.relative_to(audit_dir).as_posix(),
        )
    ]


def finalize_audit(audit_dir: Path, fixture_dir: Path) -> dict[str, Any]:
    audit_dir = audit_dir.resolve()
    fixture_dir = fixture_dir.resolve()
    marker_path = audit_dir / "AUDIT_COMPLETE.json"
    manifest_path = audit_dir / "manifest.json"
    if marker_path.exists() or manifest_path.exists():
        raise AuditError("Completion artifacts already exist; finalization is append-only")
    temporary_files = [path for path in audit_dir.rglob("*.tmp") if path.is_file()]
    if temporary_files:
        raise AuditError(f"Temporary evidence files remain: {temporary_files!r}")
    baseline = read_json(audit_dir / "snapshots" / "00-pre.json")
    end_snapshot = read_json(audit_dir / "snapshots" / "99-end.json")
    compare_snapshots(baseline, end_snapshot)
    _assert_checkpoint_catalog(audit_dir, baseline)
    live_results, saved_results = validate_result_records(_load_node_results(audit_dir))
    reports = [validate_saved_pair(audit_dir, case) for case in EXPECTED_SOURCE_PDFS]
    matrix = {
        "schema": MATRIX_SCHEMA,
        "completed_utc": utc_now(),
        "complete": True,
        "baseline_composite_sha256": baseline["composite_sha256"],
        "end_composite_sha256": end_snapshot["composite_sha256"],
        "summary": {
            "live_expected": len(EXPECTED_SOURCE_PDFS),
            "live_passed": len(live_results),
            "saved_expected": len(SAVED_CASES),
            "saved_passed": len(saved_results),
            "failures": 0,
            "skipped": 0,
        },
        "live_results": live_results,
        "saved_results": saved_results,
        "reports": reports,
    }
    atomic_write_json(audit_dir / "result-matrix.json", matrix)
    expected_catalog = _expected_evidence_catalog()
    actual_catalog = _current_evidence_catalog(audit_dir, include_attestation=False)
    if actual_catalog != expected_catalog:
        raise AuditError(
            f"Evidence catalog mismatch: missing={sorted(expected_catalog - actual_catalog)!r}, "
            f"extra={sorted(actual_catalog - expected_catalog)!r}"
        )
    # Recompute the source set at the finalization boundary as an independent PDF binding.
    source_pdfs = capture_pdf_snapshot(fixture_dir)
    if source_pdfs["fingerprint_sha256"] != baseline["source_pdfs"]["fingerprint_sha256"]:
        raise AuditError("Final source-PDF fingerprint differs from the baseline")
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "completed_utc": utc_now(),
        "baseline_composite_sha256": baseline["composite_sha256"],
        "source_pdfs": source_pdfs,
        "artifacts": _artifact_manifest(audit_dir),
    }
    atomic_write_json(manifest_path, manifest)
    marker = {
        "schema": COMPLETION_SCHEMA,
        "status": "complete",
        "completed_utc": utc_now(),
        "manifest_sha256": sha256_file(manifest_path),
        "result_matrix_sha256": sha256_file(audit_dir / "result-matrix.json"),
        "baseline_composite_sha256": baseline["composite_sha256"],
        "end_composite_sha256": end_snapshot["composite_sha256"],
        "live_expected": len(EXPECTED_SOURCE_PDFS),
        "live_passed": len(live_results),
        "saved_expected": len(SAVED_CASES),
        "saved_passed": len(saved_results),
    }
    atomic_write_json(marker_path, marker)
    return marker


def verify_completed_audit(audit_dir: Path) -> dict[str, Any]:
    audit_dir = audit_dir.resolve()
    manifest_path = audit_dir / "manifest.json"
    marker_path = audit_dir / "AUDIT_COMPLETE.json"
    manifest = read_json(manifest_path)
    marker = read_json(marker_path)
    if manifest.get("schema") != MANIFEST_SCHEMA or marker.get("schema") != COMPLETION_SCHEMA:
        raise AuditError("Manifest/completion schema mismatch")
    if marker.get("status") != "complete" or marker.get("manifest_sha256") != sha256_file(manifest_path):
        raise AuditError("Completion marker does not authenticate the manifest")
    matrix_path = audit_dir / "result-matrix.json"
    if marker.get("result_matrix_sha256") != sha256_file(matrix_path):
        raise AuditError("Completion marker does not authenticate the result matrix")
    matrix = read_json(matrix_path)
    if matrix.get("schema") != MATRIX_SCHEMA or matrix.get("complete") is not True:
        raise AuditError("Result matrix is not complete")
    live, saved = validate_result_records([*matrix.get("live_results", ()), *matrix.get("saved_results", ())])
    expected_totals = {
        "live_expected": len(EXPECTED_SOURCE_PDFS),
        "live_passed": len(live),
        "saved_expected": len(SAVED_CASES),
        "saved_passed": len(saved),
    }
    if any(marker.get(key) != value for key, value in expected_totals.items()):
        raise AuditError(f"Completion totals do not prove 7+8 passes: {marker!r}")
    manifest_records = {str(row.get("path") or ""): row for row in manifest.get("artifacts") or ()}
    actual_paths = _current_evidence_catalog(audit_dir, include_attestation=True)
    if set(manifest_records) != actual_paths:
        raise AuditError("Manifest artifact catalog differs from the completed audit directory")
    for relative, expected in manifest_records.items():
        path = audit_dir / relative
        if not path.is_file():
            raise AuditError(f"Manifest artifact is missing: {relative}")
        if path.stat().st_size != expected.get("bytes") or sha256_file(path) != expected.get("sha256"):
            raise AuditError(f"Manifest artifact was modified: {relative}")
    baseline = read_json(audit_dir / "snapshots" / "00-pre.json")
    end_snapshot = read_json(audit_dir / "snapshots" / "99-end.json")
    compare_snapshots(baseline, end_snapshot)
    if marker.get("baseline_composite_sha256") != baseline.get("composite_sha256"):
        raise AuditError("Completion marker baseline fingerprint mismatch")
    if marker.get("end_composite_sha256") != end_snapshot.get("composite_sha256"):
        raise AuditError("Completion marker end fingerprint mismatch")
    return marker


def _case_slug(case_id: str) -> str:
    for case in (*EXPECTED_SOURCE_PDFS, *SAVED_CASES):
        if case.case_id == case_id:
            return case.slug
    raise AuditError(f"Unknown case id: {case_id}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    contract = subparsers.add_parser("contract")
    contract.add_argument("--output", type=Path, required=True)

    snapshot = subparsers.add_parser("snapshot")
    snapshot.add_argument("--workspace", type=Path, required=True)
    snapshot.add_argument("--fixture-dir", type=Path, required=True)
    snapshot.add_argument("--label", required=True)
    snapshot.add_argument("--output", type=Path, required=True)
    snapshot.add_argument("--baseline", type=Path)

    validate = subparsers.add_parser("validate-node")
    validate.add_argument("--audit-dir", type=Path, required=True)
    validate.add_argument("--fixture-dir", type=Path, required=True)
    validate.add_argument("--case-id", required=True)
    validate.add_argument("--junit", type=Path, required=True)
    validate.add_argument("--stdout", type=Path, required=True)
    validate.add_argument("--checkpoint", type=Path, required=True)
    validate.add_argument("--baseline", type=Path, required=True)
    validate.add_argument("--output", type=Path, required=True)
    validate.add_argument("--exit-code", type=int, required=True)
    validate.add_argument("--started-utc", required=True)
    validate.add_argument("--completed-utc", required=True)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--audit-dir", type=Path, required=True)
    finalize.add_argument("--fixture-dir", type=Path, required=True)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--audit-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "contract":
            atomic_write_json(args.output, contract_payload())
        elif args.command == "snapshot":
            snapshot = capture_snapshot(args.workspace, args.fixture_dir, label=args.label)
            if args.baseline:
                baseline = read_json(args.baseline)
                try:
                    compare_snapshots(baseline, snapshot)
                    snapshot["matches_baseline"] = True
                    snapshot["baseline_composite_sha256"] = baseline.get("composite_sha256")
                except AuditError:
                    snapshot["matches_baseline"] = False
                    snapshot["baseline_composite_sha256"] = baseline.get("composite_sha256")
                    atomic_write_json(args.output, snapshot)
                    raise
            atomic_write_json(args.output, snapshot)
        elif args.command == "validate-node":
            validate_node(
                audit_dir=args.audit_dir,
                fixture_dir=args.fixture_dir,
                case_id=args.case_id,
                junit_path=args.junit,
                stdout_path=args.stdout,
                checkpoint_path=args.checkpoint,
                baseline_path=args.baseline,
                output_path=args.output,
                exit_code=args.exit_code,
                started_utc=args.started_utc,
                completed_utc=args.completed_utc,
            )
        elif args.command == "finalize":
            finalize_audit(args.audit_dir, args.fixture_dir)
        elif args.command == "verify":
            marker = verify_completed_audit(args.audit_dir)
            print(json.dumps(marker, ensure_ascii=False, sort_keys=True))
        else:  # pragma: no cover - argparse owns this branch
            raise AuditError(f"Unsupported command: {args.command}")
    except (AuditError, OSError) as exc:
        print(f"PERSONAL_DETAIL_FINAL_AUDIT_ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
