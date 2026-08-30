"""One fresh Primary/Secondary pass with frozen evidence and export comparisons.

Run tiers in separate processes for concurrency. The All corpus is deliberately
not an option. A new output directory is required; frozen baselines are read-only.
Projection-only retries may reuse this runner's checksummed fresh perception if
all perception code hashes still match. This is a preservation check, not an
independent certification of extraction recall.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import sys
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

from docmirror.models.entities.parse_result import ParseResult
from docmirror.models.fingerprint import canonical_fact_payload
from docmirror.models.sealed import seal_parse_result
from scripts.validate.bank_compact_exports import _first_difference, validate_and_write_bank_exports

REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hashes() -> dict[str, str]:
    paths = [
        path
        for path in (REPO_ROOT / "docmirror").rglob("*")
        if path.is_file() and path.suffix in {".py", ".yaml", ".json"}
    ]
    paths.extend((REPO_ROOT / "scripts" / "validate").glob("*.py"))
    source_auditor = REPO_ROOT / "tmp" / "pdfs" / "bank_digital_corpus_audit.py"
    if source_auditor.is_file():
        paths.append(source_auditor)
    return {path.relative_to(REPO_ROOT).as_posix(): _sha(path) for path in sorted(paths)}


def _perception_hashes(hashes: dict[str, str]) -> dict[str, str]:
    return {
        key: value
        for key, value in hashes.items()
        if key.startswith("docmirror/")
        and not key.startswith(("docmirror/plugins/", "docmirror/output/", "docmirror/server/"))
        and key != "docmirror/configs/schemas/community_bundle.schema.json"
    }


def _bank_runtime_hashes(hashes: dict[str, str]) -> dict[str, str]:
    """Keep shared code and the forced-bank provider; other providers are independent."""
    return {
        key: value
        for key, value in hashes.items()
        if key.startswith("docmirror/")
        and (
            not key.startswith("docmirror/plugins/")
            or key.startswith("docmirror/plugins/_")
            or key.startswith("docmirror/plugins/bank_statement/")
        )
    }


def _replay_code_changes(before: dict[str, str], after: dict[str, str]) -> list[str]:
    before, after = _bank_runtime_hashes(before), _bank_runtime_hashes(after)
    changed = sorted(key for key in before.keys() | after.keys() if before.get(key) != after.get(key))
    # This one seam is evaluated against both saved inputs in every replay.
    # Any other bank/shared-code change requires a real projection/perception
    # retry; saved artifacts alone cannot certify changed extraction code.
    if set(changed) - {"docmirror/plugins/_runtime/evidence_access.py"}:
        raise AssertionError("bank extraction code changed; artifact-only replay is insufficient")
    return changed


def assert_equal(expected: Any, actual: Any, subject: str) -> None:
    if difference := _first_difference(expected, actual):
        # Do not print private account/transaction values in failure logs.
        raise AssertionError(f"{subject}: {difference}")


def assert_parse_preserved(expected: ParseResult, actual: ParseResult) -> None:
    """Compare all normative facts plus routing, operations, and diagnostics."""
    assert_equal(canonical_fact_payload(expected), canonical_fact_payload(actual), "canonical facts")
    assert_equal(expected.table_operations, actual.table_operations, "table operations")
    if expected.evidence_plane is not None:
        assert actual.evidence_plane is not None
        assert_equal(expected.evidence_plane.diagnostics, actual.evidence_plane.diagnostics, "evidence diagnostics")
    expected_info = expected.parser_info.model_dump(mode="json")
    actual_info = actual.parser_info.model_dump(mode="json")
    for info in (expected_info, actual_info):
        info.pop("elapsed_ms", None)
        if isinstance(info.get("structure"), dict):
            info["structure"].pop("step_timings", None)
    assert_equal(expected_info, actual_info, "parser routing and metadata")


def assert_semantic_preserved(expected: dict, actual: dict, expected_seal: str, actual_seal: str) -> None:
    """Bind each snapshot fingerprint to its own evidence; compare every other field."""
    assert_equal(expected["source"]["fingerprint"], expected_seal, "frozen semantic snapshot binding")
    assert_equal(actual["source"]["fingerprint"], actual_seal, "fresh semantic snapshot binding")
    # Sealed fingerprints include timing/staging paths. Those are verified by
    # assert_parse_preserved, which compares facts plus non-timing metadata.
    comparable = {**actual, "source": {**actual["source"], "fingerprint": expected_seal}}
    assert_equal(expected, comparable, "raw, canonical, provenance and selection")


def _policy_from_dict(payload: dict):
    from docmirror.input.entry.options import (
        DocTypeHint,
        PageSelection,
        ParsePolicy,
        SafetyControl,
        normalize_parse_policy,
    )

    data = dict(payload)
    pages = dict(data.pop("pages"))
    pages["ranges"] = tuple(tuple(pair) for pair in pages.get("ranges", ()))
    hint = data.pop("doc_type_hint", None)
    safety = data.pop("safety")
    data["ocr_correction_packs"] = tuple(data.get("ocr_correction_packs", ()))
    return normalize_parse_policy(
        ParsePolicy(
            pages=PageSelection(**pages),
            doc_type_hint=DocTypeHint(**hint) if hint else None,
            safety=SafetyControl(**safety),
            **data,
        )
    )


def _load_auditor():
    path = REPO_ROOT / "tmp" / "pdfs" / "bank_digital_corpus_audit.py"
    spec = importlib.util.spec_from_file_location("bank_runtime_source_auditor", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("private source auditor is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _baseline(entry: dict, frozen: Path, result: dict) -> dict:
    source_sha = entry["source_sha256"]
    paths = list((frozen / "cache" / source_sha[:2] / source_sha).glob("*.parse_result.json"))
    if len(paths) != 1:
        raise AssertionError("expected exactly one frozen perception baseline")
    parse_path = paths[0]
    parse_meta = _read(parse_path.with_name(parse_path.name.replace(".parse_result.json", ".meta.json")))
    if parse_meta["source_sha256"] != source_sha or _sha(parse_path) != parse_meta["artifact_sha256"]:
        raise AssertionError("frozen perception checksum/source mismatch")
    parse_result = ParseResult.model_validate_json(parse_path.read_bytes())
    if seal_parse_result(parse_result).integrity_fingerprint != parse_meta["sealed_fingerprint"]:
        raise AssertionError("frozen sealed evidence mismatch")
    community_path = Path(result["community"])
    export_meta = _read(community_path.with_name(community_path.name.replace(".community.json", ".meta.json")))
    evidence_path = community_path.with_suffix(".evidence.json")
    if _sha(community_path) != export_meta["community_sha256"] or _sha(evidence_path) != export_meta["evidence_sha256"]:
        raise AssertionError("frozen export/evidence checksum mismatch")
    public = _read(community_path)
    artifacts = result["export_validation"]["artifacts"]
    assert_equal(public, _read(Path(artifacts["community"])), "frozen delivery/cache")
    return {
        "parse_result": parse_result,
        "parse_meta": parse_meta,
        "public": public,
        "semantic": _read(evidence_path),
        "artifacts": artifacts,
    }


@contextmanager
def _measure_bank_reuse(stats: dict):
    from docmirror.plugins.bank_statement import work_cache

    original = work_cache.bank_work_session

    @contextmanager
    def measured(document, **kwargs):
        with original(document, **kwargs) as cache:
            yield cache
        stats.update(hits=dict(cache.hits), misses=dict(cache.misses))

    with patch.object(work_cache, "bank_work_session", measured):
        yield


async def _perceive(source: Path, policy: Any, workers: int, stats: dict):
    from docmirror.input.entry.factory import PerceiveOptions, perceive_document
    from docmirror.input.extraction.extractor import CoreExtractor

    original = CoreExtractor.extract_parse_result

    async def measured(extractor, *args, **kwargs):
        result = await original(extractor, *args, **kwargs)
        stats.update(extractor.native_page_execution_stats)
        return result

    with patch.object(CoreExtractor, "extract_parse_result", measured):
        return await perceive_document(source, PerceiveOptions(policy=policy, max_workers=workers))


def _run_one(
    entry: dict,
    baseline: dict,
    source: Path,
    output: Path,
    args: Any,
    code_hashes: dict,
    auditor: Any,
    golden: dict[str, Any] | None,
) -> dict:
    from docmirror.plugins.bank_statement.community_plugin import plugin

    started = time.perf_counter()
    result: dict[str, Any] = {
        "filename": entry["filename"],
        "source_sha256": entry["source_sha256"],
        "errors": [],
        "timings": {},
        "native_pages": {},
        "bank_reuse": {},
    }
    output.mkdir(parents=True, exist_ok=False)
    policy = _policy_from_dict(baseline["parse_meta"]["parse_policy"])
    if policy.fingerprint() != baseline["parse_meta"]["parse_policy_fingerprint"]:
        raise AssertionError("baseline parse policy no longer matches")
    phase_start = time.perf_counter()
    if args.reuse_extraction_from:
        prior = Path(args.reuse_extraction_from).resolve() / source.stem
        meta = _read(prior / "parse.meta.json")
        parse_path = prior / "parse_result.json"
        assert_equal(_perception_hashes(meta["code_hashes"]), _perception_hashes(code_hashes), "reused perception code")
        if meta["source_sha256"] != entry["source_sha256"] or meta["artifact_sha256"] != _sha(parse_path):
            raise AssertionError("fresh perception retry cache checksum/source mismatch")
        sealed = seal_parse_result(ParseResult.model_validate_json(parse_path.read_bytes()))
        if sealed.integrity_fingerprint != meta["sealed_fingerprint"]:
            raise AssertionError("fresh perception retry cache integrity mismatch")
        result["native_pages"] = meta["native_pages"]
        result["extraction_executed"] = False
    else:
        sealed = asyncio.run(_perceive(source, policy, args.page_workers, result["native_pages"]))
        result["extraction_executed"] = True
    result["timings"]["perception_seconds"] = round(time.perf_counter() - phase_start, 3)
    actual = sealed.to_read_view()
    if not actual.success or len(actual.pages) != entry["effective_page_count"]:
        raise AssertionError("fresh perception failed or lost pages")
    if not sealed.verify_integrity():
        raise AssertionError("fresh perception integrity failed")
    parse_path = output / "parse_result.json"
    auditor._atomic_write_text(parse_path, sealed.model_dump_json())
    auditor._atomic_write_json(
        output / "parse.meta.json",
        {
            "source_sha256": entry["source_sha256"],
            "artifact_sha256": _sha(parse_path),
            "sealed_fingerprint": sealed.integrity_fingerprint,
            "code_hashes": code_hashes,
            "native_pages": result["native_pages"],
        },
    )
    try:
        assert_parse_preserved(baseline["parse_result"], actual)
        result["canonical_facts_unchanged"] = True
    except AssertionError as exc:
        result["errors"].append(str(exc))
    phase_start = time.perf_counter()
    with _measure_bank_reuse(result["bank_reuse"]):
        bundle = plugin.project_bundle(sealed)
    if bundle is None:
        raise AssertionError("bank plugin returned no bundle")
    result["timings"]["projection_seconds"] = round(time.perf_counter() - phase_start, 3)
    phase_start = time.perf_counter()
    public, export_validation = validate_and_write_bank_exports(bundle, output / "exports")
    semantic = bundle.semantic_payload()
    auditor._atomic_write_json(output / "projection.community.json", public)
    auditor._atomic_write_json(output / "projection.community.evidence.json", semantic)
    if not args.accept_projection_changes:
        try:
            assert_equal(baseline["public"], public, "public output")
            assert_semantic_preserved(
                baseline["semantic"],
                semantic,
                baseline["parse_meta"]["sealed_fingerprint"],
                sealed.integrity_fingerprint,
            )
        except AssertionError as exc:
            result["errors"].append(str(exc))
    artifacts = export_validation["artifacts"]
    if not args.accept_projection_changes:
        for kind in ("content", "enhanced_reading"):
            if Path(artifacts[kind]).read_bytes() != Path(baseline["artifacts"][kind]).read_bytes():
                result["errors"].append(f"{kind} Markdown differs from frozen output")
        expected_csvs = Path(baseline["artifacts"]["datasets"])
        actual_csvs = Path(artifacts["datasets"])
        expected_names = sorted(path.name for path in expected_csvs.glob("*.csv"))
        actual_names = sorted(path.name for path in actual_csvs.glob("*.csv"))
        if expected_names != actual_names:
            result["errors"].append("dataset/audit CSV file inventory differs")
        for name in expected_names:
            if (
                not (actual_csvs / name).is_file()
                or (actual_csvs / name).read_bytes() != (expected_csvs / name).read_bytes()
            ):
                result["errors"].append(f"dataset/audit CSV differs: {name}")
    audit = auditor.audit_community_payload(
        public,
        effective_page_count=entry["effective_page_count"],
        parse_result=actual,
        evidence_payload=semantic,
        golden=golden,
    )
    auditor._atomic_write_json(output / "audit.json", audit)
    if audit["status"] != "pass":
        result["errors"].append("independent source/semantic auditor reported errors")
    if not sealed.verify_integrity():
        result["errors"].append("projection or export changed sealed evidence")
    result.update(
        transaction_rows=audit["transaction_rows"],
        statement_header_rows=audit["statement_header_rows"],
        completion_status=audit["completion_status"],
        audit_status=audit["status"],
        golden_checked=audit["golden_checked"],
        golden_check_count=(
            len((golden or {}).get("dataset_row_counts") or {})
            + len((golden or {}).get("assertions") or [])
            + int(isinstance((golden or {}).get("dataset_order"), list))
        ),
        export_validation=export_validation,
    )
    result["timings"]["export_and_audit_seconds"] = round(time.perf_counter() - phase_start, 3)
    result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    result["status"] = "pass" if not result["errors"] else "fail"
    return result


def _replay_one(entry: dict, baseline: dict, prior: Path, output: Path, code_hashes: dict, auditor: Any) -> dict:
    """Recheck current serializers and saved output without reading PDF content."""
    from docmirror.output.community_bundle import render_community_reading_markdown
    from docmirror.plugins._runtime.evidence_access import text_atoms
    from docmirror.server.output_builder import materialize_community_bundle
    from scripts.validate.bank_business_exports import assert_business_value_conservation, evidence_delivery
    from scripts.validate.bank_compact_exports import bind_internal_evidence
    from scripts.validate.validate_community_artifacts import validate_community_artifacts

    started = time.perf_counter()
    directory = prior / Path(entry["filename"]).stem
    meta = _read(directory / "parse.meta.json")
    changed = _replay_code_changes(meta["code_hashes"], code_hashes)
    parse_path = directory / "parse_result.json"
    if meta["source_sha256"] != entry["source_sha256"] or meta["artifact_sha256"] != _sha(parse_path):
        raise AssertionError("saved fresh perception checksum/source mismatch")
    actual = ParseResult.model_validate_json(parse_path.read_bytes())
    sealed = seal_parse_result(actual)
    assert_equal(meta["sealed_fingerprint"], sealed.integrity_fingerprint, "saved fresh snapshot")
    assert_parse_preserved(baseline["parse_result"], actual)
    # Re-evaluate the only post-run production edit, against the original
    # declared-store serialization on both real source-derived inputs.
    for document in (baseline["parse_result"], actual):
        store = document.evidence_plane.evidence
        expected_atoms = store.model_dump(mode="json", exclude_none=True)["text_atoms"]
        assert_equal(expected_atoms, text_atoms(document), "current text-only serializer")
    public = _read(directory / "projection.community.json")
    semantic = _read(directory / "projection.community.evidence.json")
    assert_equal(baseline["public"], public, "frozen public output")
    assert_semantic_preserved(
        baseline["semantic"], semantic, baseline["parse_meta"]["sealed_fingerprint"], sealed.integrity_fingerprint
    )
    bind_internal_evidence(public, semantic)
    assert_business_value_conservation(evidence_delivery(semantic), public)
    previous = next(
        item for item in _read(prior / "report.json")["results"] if item["source_sha256"] == entry["source_sha256"]
    )
    artifacts = previous["export_validation"]["artifacts"]
    assert_equal(public, _read(Path(artifacts["community"])), "written Community JSON")
    issues = validate_community_artifacts(Path(artifacts["community"]))
    if issues:
        raise AssertionError("artifact contract failed: " + "; ".join(issues[:10]))
    replayed = materialize_community_bundle(public, actual)
    assert_equal(public, replayed.json_payload(), "current public JSON replay")
    for kind in ("content", "enhanced_reading"):
        if Path(artifacts[kind]).read_bytes() != Path(baseline["artifacts"][kind]).read_bytes():
            raise AssertionError(f"{kind} Markdown changed")
    if render_community_reading_markdown(semantic).encode("utf-8") != Path(artifacts["enhanced_reading"]).read_bytes():
        raise AssertionError("current enhanced Markdown differs")
    output_bundle = replace(replayed, domain=semantic["domain"], diagnostics={})
    for relative, content in output_bundle.render_dataset_csvs(semantic).items():
        if (Path(artifacts["community"]).parent / relative).read_bytes() != content.encode("utf-8"):
            raise AssertionError("current dataset CSV differs")
    expected_csvs = Path(baseline["artifacts"]["datasets"])
    actual_csvs = Path(artifacts["datasets"])
    assert_equal(
        sorted(path.name for path in expected_csvs.glob("*.csv")),
        sorted(path.name for path in actual_csvs.glob("*.csv")),
        "CSV inventory",
    )
    for path in expected_csvs.glob("*.csv"):
        if path.read_bytes() != (actual_csvs / path.name).read_bytes():
            raise AssertionError("frozen dataset/audit CSV differs")
    if output_bundle.render_audit_csv(semantic).encode("utf-8") != (actual_csvs / "_audit_cells.csv").read_bytes():
        raise AssertionError("current audit CSV differs")
    audit = auditor.audit_community_payload(
        public, effective_page_count=entry["effective_page_count"], parse_result=actual, evidence_payload=semantic
    )
    if audit["status"] != "pass" or not sealed.verify_integrity():
        raise AssertionError("source audit or sealed integrity failed")
    output.mkdir(parents=True, exist_ok=False)
    auditor._atomic_write_json(output / "audit.json", audit)
    artifact_paths = [
        parse_path,
        directory / "projection.community.json",
        directory / "projection.community.evidence.json",
        *(Path(artifacts[kind]) for kind in ("community", "content", "enhanced_reading")),
        *actual_csvs.glob("*.csv"),
    ]
    return {
        "filename": entry["filename"],
        "source_sha256": entry["source_sha256"],
        "status": "pass",
        "errors": [],
        "extraction_executed": False,
        "projection_executed": False,
        "origin": str(directory),
        "canonical_facts_unchanged": True,
        "business_and_provenance_unchanged": True,
        "current_serializer_equivalent": True,
        "current_exporters_checked": True,
        "reevaluated_code_changes": changed,
        "artifact_hashes": {str(path): _sha(path) for path in artifact_paths},
        "native_pages": previous["native_pages"],
        "bank_reuse": previous["bank_reuse"],
        "fresh_run_timings": previous["timings"],
        "transaction_rows": audit["transaction_rows"],
        "statement_header_rows": audit["statement_header_rows"],
        "completion_status": audit["completion_status"],
        "audit_status": audit["status"],
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=("primary", "secondary"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--page-workers", type=int, default=4)
    parser.add_argument("--cases", type=int, nargs="+")
    parser.add_argument("--retry-failures-from", type=Path)
    parser.add_argument("--golden", type=Path)
    parser.add_argument(
        "--accept-projection-changes",
        action="store_true",
        help=(
            "Allow reviewed bank-projection fixes to differ from the frozen output; "
            "perception/facts, export contracts, the source auditor, and --golden still gate the run"
        ),
    )
    replay = parser.add_mutually_exclusive_group()
    replay.add_argument("--reuse-extraction-from", type=Path)
    replay.add_argument("--replay-only-from", type=Path)
    args = parser.parse_args(argv)
    frozen = REPO_ROOT / "tmp" / "pdfs" / f"bank_digital_{args.tier}_frozen_20260826"
    manifest = _read(frozen / "manifest.json")
    baseline_report = _read(REPO_ROOT / "tmp" / "pdfs" / "bank_business_output_20260827" / f"{args.tier}_final.json")
    by_sha = {item["source_sha256"]: item for item in baseline_report["results"]}
    entries = list(manifest["entries"])
    if args.cases:
        if not set(args.cases) <= {entry["case_number"] for entry in entries}:
            raise ValueError("requested case is not in the selected tier")
        entries = [entry for entry in entries if entry["case_number"] in args.cases]
    if args.retry_failures_from:
        failures = {
            item["source_sha256"] for item in _read(args.retry_failures_from)["results"] if item["status"] != "pass"
        }
        entries = [entry for entry in entries if entry["source_sha256"] in failures]
    if not entries:
        raise ValueError("no files selected")
    output = args.output_root.resolve()
    output.relative_to(REPO_ROOT / "tmp" / "pdfs")
    if output.exists():
        raise ValueError("use a new output directory to preserve prior validation artifacts")
    corpus = REPO_ROOT / "tests" / "fixtures-private" / "bank_cashflow" / "Digital" / args.tier.title()
    # Preflight every selected baseline and source before the first PDF parse.
    baselines = {}
    for entry in entries:
        source = corpus / entry["filename"]
        if _sha(source) != entry["source_sha256"]:
            raise AssertionError("source changed since frozen manifest")
        baselines[entry["source_sha256"]] = _baseline(entry, frozen, by_sha[entry["source_sha256"]])
        meta = baselines[entry["source_sha256"]]["parse_meta"]
        if _policy_from_dict(meta["parse_policy"]).fingerprint() != meta["parse_policy_fingerprint"]:
            raise AssertionError("baseline parse policy no longer matches")
    code_hashes = _hashes()
    auditor = _load_auditor()
    golden_by_sha, golden_fingerprint = auditor._golden_entries(args.golden.resolve() if args.golden else None)
    selected_shas = {entry["source_sha256"] for entry in entries}
    missing_goldens = sorted(selected_shas - set(golden_by_sha)) if args.golden else []
    if missing_goldens:
        raise ValueError(f"selected sources are absent from the golden manifest: {missing_goldens}")
    if args.accept_projection_changes and not args.golden:
        raise ValueError("--accept-projection-changes requires --golden")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report: dict[str, Any] = {
        "tier": args.tier,
        "file_count": len(entries),
        "results": [],
        "code_hashes": code_hashes,
        "baseline_report": f"bank_business_output_20260827/{args.tier}_final.json",
        "operation": "artifact_replay" if args.replay_only_from else "fresh_validation",
        "golden_fingerprint": golden_fingerprint,
        "accept_projection_changes": bool(args.accept_projection_changes),
    }
    for entry in entries:
        print(f"START {args.tier} {entry['filename']}", flush=True)
        try:
            if args.replay_only_from:
                result = _replay_one(
                    entry,
                    baselines[entry["source_sha256"]],
                    args.replay_only_from.resolve(),
                    output / Path(entry["filename"]).stem,
                    code_hashes,
                    auditor,
                )
            else:
                result = _run_one(
                    entry,
                    baselines[entry["source_sha256"]],
                    corpus / entry["filename"],
                    output / Path(entry["filename"]).stem,
                    args,
                    code_hashes,
                    auditor,
                    golden_by_sha.get(entry["source_sha256"]),
                )
        except Exception as exc:
            result = {
                "filename": entry["filename"],
                "source_sha256": entry["source_sha256"],
                "status": "fail",
                "errors": [f"{type(exc).__name__}: {exc}"],
            }
        report["results"].append(result)
        auditor._atomic_write_json(output / "report.json", report)
        print(
            f"{result['status'].upper()} {entry['filename']}: {result.get('elapsed_seconds', 0)}s {result['errors']}",
            flush=True,
        )
    report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    report["passed"] = sum(result["status"] == "pass" for result in report["results"])
    report["failed"] = len(entries) - report["passed"]
    after_hashes = _hashes()
    changed_during_run = {
        key for key in code_hashes.keys() | after_hashes.keys() if code_hashes.get(key) != after_hashes.get(key)
    }
    report["unrelated_provider_changes_during_run"] = sorted(
        key
        for key in changed_during_run
        if key.startswith("docmirror/plugins/")
        and key not in _bank_runtime_hashes(code_hashes)
        and key not in _bank_runtime_hashes(after_hashes)
    )
    report["code_unchanged_during_run"] = not changed_during_run - set(report["unrelated_provider_changes_during_run"])
    auditor._atomic_write_json(output / "report.json", report)
    print(
        json.dumps(
            {key: report[key] for key in ("tier", "passed", "failed", "elapsed_seconds", "code_unchanged_during_run")}
        ),
        flush=True,
    )
    return 0 if not report["failed"] and report["code_unchanged_during_run"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
