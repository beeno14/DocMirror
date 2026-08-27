"""Replay audited Primary/Secondary semantic snapshots through consumer exporters.

No PDF parsing or bank strategy runs are permitted. This validates an intentional
delivery-only cleanup, not extraction recall. The source report and every input
artifact are read-only; use a new output directory for each validation pass.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch

from docmirror.models.entities.parse_result import ParseResult
from docmirror.models.sealed import seal_parse_result
from docmirror.output.bank_business_view import business_view
from docmirror.output.community_bundle import CommunityBundle, CommunityDataset
from docmirror.server.artifact_writer import ArtifactWriter
from docmirror.server.edition_outputs import _write_community_bundle_files
from docmirror.server.output_builder import materialize_community_bundle
from scripts.validate.bank_business_exports import (
    assert_business_csv_conservation,
    assert_business_markdown_values,
    assert_business_value_conservation,
    evidence_delivery,
)
from scripts.validate.bank_compact_exports import _audit_additional_fields, bind_internal_evidence
from scripts.validate.bank_runtime_equivalence import (
    REPO_ROOT,
    _bank_runtime_hashes,
    _hashes,
    _load_auditor,
    _read,
    _sha,
    assert_equal,
)
from scripts.validate.validate_community_artifacts import validate_community_artifacts

_DELIVERY_ONLY_FILES = {
    "docmirror/output/bank_business_view.py",
    "docmirror/output/community_bundle.py",
    "docmirror/server/edition_outputs.py",
    "docmirror/configs/schemas/community_bundle.schema.json",
}


def assert_output_only_changes(before: dict, after: dict) -> list[str]:
    before, after = _bank_runtime_hashes(before), _bank_runtime_hashes(after)
    changed = sorted(key for key in before.keys() | after.keys() if before.get(key) != after.get(key))
    if set(changed) - _DELIVERY_ONLY_FILES:
        raise AssertionError("extraction/shared code changed; saved evidence cannot validate those changes")
    return changed


def _validation_hashes(hashes: dict) -> dict:
    checked = _bank_runtime_hashes(hashes)
    checked.update({key: value for key, value in hashes.items()
                    if key.startswith("scripts/validate/") or key == "tmp/pdfs/bank_digital_corpus_audit.py"})
    return checked


@contextmanager
def without_extraction():
    """Fail immediately if a replay accidentally tries to parse or reconstruct."""
    with ExitStack() as stack:
        for target in (
            "docmirror.input.entry.factory.perceive_document",
            "docmirror.input.extraction.extractor.CoreExtractor.extract_parse_result",
            "docmirror.plugins.bank_statement.community_plugin.BankStatementCommunityPlugin.project_bundle",
            "docmirror.plugins.bank_statement.extract_pipeline.run_bank_statement_extract",
            "fitz.open",
            "pdfplumber.open",
        ):
            stack.enter_context(patch(target, side_effect=AssertionError("extraction is forbidden during output replay")))
        yield


@dataclass
class _SavedSemanticBundle(CommunityBundle):
    """The validated semantic artifact is the input stage, not new raw cells.

Re-running semantic normalization on serialized raw pools could treat an encoded
list as fresh source text. Freeze that stage while exercising the real JSON,
Markdown, CSV, schema, record-conservation and artifact-writing implementations.
"""

    saved_semantic: dict = field(default_factory=dict, repr=False)

    def semantic_payload(self) -> dict:
        for key in ("document", "files", "domain", "warnings"):
            assert_equal(self.saved_semantic[key], getattr(self, key), f"saved semantic {key}")
        assert_equal(self.saved_semantic["datasets"],
                     [{**dataset.public, "rows": dataset.rows} for dataset in self.datasets], "saved internal datasets")
        return copy.deepcopy(self.saved_semantic)


def bundle_from_snapshot(semantic: dict, document: ParseResult, source_markdown: str = "") -> CommunityBundle:
    snapshot = copy.deepcopy(semantic)
    return _SavedSemanticBundle(
        schema={"version": "5.0.0", "domain": "bank_statement"},
        document=copy.deepcopy(snapshot["document"]),
        sections=copy.deepcopy(snapshot["structure"]["sections"]),
        datasets=[CommunityDataset(public=copy.deepcopy({key: value for key, value in dataset.items() if key != "rows"}),
                                   rows=copy.deepcopy(dataset["rows"])) for dataset in snapshot["datasets"]],
        files=copy.deepcopy(snapshot["files"]),
        warnings=copy.deepcopy(snapshot["warnings"]),
        result=document,
        source_fingerprint=snapshot["source"]["fingerprint"],
        parse_result_schema=snapshot["source"]["parse_result_schema"],
        classification=copy.deepcopy(snapshot["classification"]),
        domain=copy.deepcopy(snapshot["domain"]),
        diagnostics=copy.deepcopy(snapshot["diagnostics"]),
        content_markdown_override=source_markdown,
        saved_semantic=snapshot,
    )


def _checked_path(path: Path) -> Path:
    path = path.resolve()
    if not path.is_relative_to(REPO_ROOT / "tmp" / "pdfs"):
        raise ValueError("validation artifacts must be inside the repository tmp/pdfs directory")
    return path


def _verify_inputs(entry: dict) -> tuple[Path, dict[str, str]]:
    if entry.get("status") != "pass" or entry.get("audit_status") != "pass":
        raise AssertionError("only previously audited passing cases may be refreshed")
    origin = _checked_path(Path(entry["origin"]))
    hashes = entry["artifact_hashes"]
    for name, expected in hashes.items():
        path = _checked_path(Path(name))
        if not path.is_relative_to(origin) or _sha(path) != expected:
            raise AssertionError("saved input artifact checksum or ownership differs")
    required = [origin / name for name in ("parse_result.json", "projection.community.json", "projection.community.evidence.json")]
    required.extend(origin / "exports" / name for name in ("001_community.json", "001_content.md", "001_enhanced_reading.md"))
    required.append(origin / "exports" / "001_datasets" / "_audit_cells.csv")
    if not all(str(path) in hashes for path in required):
        raise AssertionError("prior report does not bind every required artifact")
    return origin, hashes


def refresh_case(entry: dict, output: Path, auditor) -> dict:
    started = time.perf_counter()
    origin, input_hashes = _verify_inputs(entry)
    previous = _read(origin / "projection.community.json")
    semantic = _read(origin / "projection.community.evidence.json")
    parse_meta = _read(origin / "parse.meta.json")
    document = ParseResult.model_validate_json((origin / "parse_result.json").read_bytes())
    sealed = seal_parse_result(document)
    assert_equal(entry["source_sha256"], parse_meta["source_sha256"], "source checksum")
    assert_equal(parse_meta["artifact_sha256"], _sha(origin / "parse_result.json"), "parse checksum")
    assert_equal(parse_meta["sealed_fingerprint"], sealed.integrity_fingerprint, "sealed snapshot")
    assert_equal(semantic["source"]["fingerprint"], sealed.integrity_fingerprint, "semantic evidence binding")
    assert_equal(previous, _read(origin / "exports" / "001_community.json"), "prior delivery/cache")
    if previous["schema"]["domain"] != "bank_statement" or previous["schema"]["version"] != "5.0.0":
        raise AssertionError("consumer refresh is restricted to audited digital-bank v5 output")
    original = evidence_delivery(semantic)
    # Recheck all original source fields before the explicit delivery omissions.
    aliases = semantic["domain"]["extensions"]["compact_output"].get("source_aliases") or {}
    for dataset in semantic["datasets"]:
        for row in dataset["rows"]:
            before = {**row, "normalized": {key: value for key, value in row["normalized"].items() if key != "additional_fields"}}
            _audit_additional_fields(before, row, dataset["columns"], aliases.get(dataset["name"], {}), row,
                                     serialized_sources=True)
    source_markdown = (origin / "exports" / "001_content.md").read_text(encoding="utf-8")
    bundle = bundle_from_snapshot(semantic, document, source_markdown)
    output.mkdir(parents=True, exist_ok=False)
    paths = _write_community_bundle_files(bundle, output / "exports", file_id="001", document_id=bundle.document["id"])
    public = _read(paths["community"])
    assert_business_value_conservation(original, public)
    assert_equal(business_view(previous), public, "only explicit consumer omissions")
    bind_internal_evidence(public, semantic)
    assert_equal(semantic, bundle.semantic_payload(), "internal semantic evidence")
    for kind in ("content", "enhanced_reading"):
        assert_business_markdown_values(public, paths[kind].read_text(encoding="utf-8"))
    assert_equal(paths["content"].read_bytes(), paths["enhanced_reading"].read_bytes(), "consumer Markdown agreement")
    assert_equal(source_markdown, bundle.render_source_markdown(), "internal source Markdown")
    prior_datasets = {dataset["id"]: dataset for dataset in previous["datasets"]}
    for dataset in public["datasets"]:
        relative = dataset["csv"]
        old_path, new_path = origin / "exports" / relative, paths["community"].parent / relative
        if not old_path.resolve().is_relative_to(origin) or not new_path.resolve().is_relative_to(output):
            raise AssertionError("dataset CSV escapes its artifact directory")
        if str(old_path) not in input_hashes:
            raise AssertionError("prior dataset CSV was not checksummed")
        assert_business_csv_conservation(old_path.read_text(encoding="utf-8"), new_path.read_text(encoding="utf-8"),
                                         dataset, original_dataset=prior_datasets[dataset["id"]])
    assert_equal((origin / "exports" / "001_datasets" / "_audit_cells.csv").read_bytes(),
                 (paths["datasets"] / "_audit_cells.csv").read_bytes(), "internal audit CSV")
    if issues := validate_community_artifacts(paths["community"]):
        raise AssertionError("artifact contract failed: " + "; ".join(issues[:10]))
    restored = materialize_community_bundle(public, document)
    assert_equal(public, restored.json_payload(), "consumer JSON replay")
    assert_equal(bundle.render_enhanced_markdown(semantic, public_payload=public), restored.render_markdown(), "Markdown replay")
    assert_equal(bundle.render_dataset_csvs(semantic, public_payload=public), restored.render_dataset_csvs(), "CSV replay")
    audit = auditor.audit_community_payload(public, effective_page_count=public["document"]["page_count"],
                                           parse_result=document, evidence_payload=semantic)
    ArtifactWriter(output).write_json("audit.json", audit)
    if audit["status"] != "pass" or not sealed.verify_integrity():
        raise AssertionError("independent source audit or sealed integrity failed")
    assert_equal(entry["completion_status"], audit["completion_status"], "audit completion status")
    assert_equal(entry["transaction_rows"], audit["transaction_rows"], "transaction count")
    assert_equal(entry["statement_header_rows"], audit["statement_header_rows"], "account count")
    _verify_inputs(entry)
    removed = {}
    for before, after in zip(previous["datasets"], public["datasets"], strict=True):
        for old_row, new_row in zip(before["rows"], after["rows"], strict=True):
            for key in old_row["normalized"].keys() - new_row["normalized"].keys():
                name = f"{before['name']}.{key}"
                removed[name] = removed.get(name, 0) + 1
    return {
        "filename": entry["filename"], "source_sha256": entry["source_sha256"], "status": "pass", "errors": [],
        "extraction_executed": False, "projection_executed": False, "internal_evidence_unchanged": True,
        "business_values_preserved_except_requested_page_summaries": True,
        "source_fields_accounted_for": True, "artifact_contract_checked": True, "replay_checked": True,
        "audit_status": audit["status"], "completion_status": audit["completion_status"],
        "transaction_rows": audit["transaction_rows"], "statement_header_rows": audit["statement_header_rows"],
        "removed_fields": removed, "input_artifact_hashes": input_hashes,
        "output_artifact_hashes": {str(path): _sha(path) for path in output.rglob("*") if path.is_file()},
        "json_bytes_before": (origin / "exports" / "001_community.json").stat().st_size,
        "json_bytes_after": paths["community"].stat().st_size,
        "artifacts": {key: str(path) for key, path in paths.items()},
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    previous = _read(_checked_path(args.source_report))
    if (previous.get("tier") not in {"primary", "secondary"} or previous.get("failed") != 0
            or previous.get("code_unchanged_during_run") is not True
            or previous.get("passed") != len(previous["results"]) or not previous["results"]):
        raise ValueError("a passing Primary or Secondary report is required")
    code_hashes = _hashes()
    changes = assert_output_only_changes(previous["code_hashes"], code_hashes)
    for entry in previous["results"]:
        _verify_inputs(entry)
        if Path(entry["filename"]).name != entry["filename"]:
            raise ValueError("invalid case filename")
    output = _checked_path(args.output_root)
    if output.exists():
        raise ValueError("use a new output directory; prior results must not be overwritten")
    output.mkdir(parents=True, exist_ok=False)
    writer = ArtifactWriter(output)
    report = {"tier": previous["tier"], "file_count": len(previous["results"]), "operation": "consumer_output_replay",
              "source_report": str(args.source_report.resolve()), "extraction_executed": False,
              "allowed_output_code_changes": changes, "code_hashes": code_hashes, "results": []}
    auditor = _load_auditor()
    started = time.perf_counter()
    with without_extraction():
        for entry in previous["results"]:
            print(f"START {report['tier']} {entry['filename']}", flush=True)
            try:
                result = refresh_case(entry, output / Path(entry["filename"]).stem, auditor)
            except Exception as exc:
                result = {"filename": entry["filename"], "source_sha256": entry["source_sha256"], "status": "fail",
                          "errors": [f"{type(exc).__name__}: {exc}"]}
            report["results"].append(result)
            writer.write_json("report.json", report)
            print(json.dumps({key: result[key] for key in ("filename", "status", "errors")}), flush=True)
    after_hashes = _hashes()
    report.update(passed=sum(result["status"] == "pass" for result in report["results"]),
                  failed=sum(result["status"] != "pass" for result in report["results"]),
                  elapsed_seconds=round(time.perf_counter() - started, 3),
                  code_unchanged_during_run=_validation_hashes(code_hashes) == _validation_hashes(after_hashes))
    writer.write_json("report.json", report)
    print(json.dumps({key: report[key] for key in ("tier", "passed", "failed", "elapsed_seconds", "code_unchanged_during_run")}), flush=True)
    return 0 if not report["failed"] and report["code_unchanged_during_run"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
