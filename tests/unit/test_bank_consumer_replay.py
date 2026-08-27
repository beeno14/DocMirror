"""Output replay must exercise real exporters without rerunning source extraction."""

from __future__ import annotations

import copy
import json

import pytest

from docmirror.models.entities.parse_result import ParseResult
from docmirror.server.edition_outputs import _write_community_bundle_files
from scripts.validate.bank_consumer_exports import (
    _verify_inputs,
    assert_output_only_changes,
    bundle_from_snapshot,
    without_extraction,
)
from scripts.validate.validate_community_artifacts import validate_community_artifacts
from tests.unit.test_bank_consumer_output import _consumer_bundle


def test_saved_semantic_replay_uses_the_production_writer_with_frozen_source_pools(tmp_path):
    original, sealed = _consumer_bundle()
    evidence = original.semantic_payload()
    before = copy.deepcopy(evidence)
    restored = bundle_from_snapshot(evidence, ParseResult(), original.render_source_markdown())
    with without_extraction():
        paths = _write_community_bundle_files(restored, tmp_path, file_id="001", document_id=original.document["id"])
    assert json.loads(paths["community"].read_text(encoding="utf-8")) == original.json_payload()
    assert restored.semantic_payload() == before == evidence
    assert paths["content"].read_text(encoding="utf-8") == original.render_markdown()
    assert paths["enhanced_reading"].read_bytes() == paths["content"].read_bytes()
    assert (paths["datasets"] / "_audit_cells.csv").read_text(encoding="utf-8") == original.render_audit_csv(evidence)
    assert validate_community_artifacts(paths["community"]) == []
    assert sealed.verify_integrity()


@pytest.mark.parametrize("mutation", ["amount", "raw", "order", "document"])
def test_saved_semantic_bundle_rejects_fact_mutation(mutation):
    original, _ = _consumer_bundle()
    restored = bundle_from_snapshot(original.semantic_payload(), ParseResult())
    if mutation == "order":
        restored.datasets.reverse()
    elif mutation == "document":
        restored.document["page_count"] = 999
    else:
        row = restored.datasets[1].rows[0]
        if mutation == "raw":
            row["raw"]["交易金额"] = "WRONG"
        else:
            row["normalized"]["amount"] = 999
    with pytest.raises(AssertionError):
        restored.semantic_payload()


@pytest.mark.parametrize("path", [
    "docmirror/input/extraction/extractor.py", "docmirror/plugins/bank_statement/statement_context.py",
    "docmirror/plugins/bank_statement/work_cache.py", "docmirror/plugins/_runtime/evidence_access.py",
    "docmirror/output/normalized_records.py", "docmirror/tables/native_pdf_candidates.py",
])
def test_output_replay_cannot_certify_changes_to_extraction_or_unapproved_shared_code(path):
    with pytest.raises(AssertionError, match="extraction/shared code changed"):
        assert_output_only_changes({path: "before"}, {path: "after"})


def test_output_replay_permits_only_the_named_output_seams_and_ignores_other_providers():
    paths = ["docmirror/output/bank_business_view.py", "docmirror/output/community_bundle.py",
             "docmirror/configs/schemas/community_bundle.schema.json", "docmirror/server/edition_outputs.py"]
    before = {path: "before" for path in paths}
    before["docmirror/plugins/credit_report/community_plugin.py"] = "before"
    after = dict.fromkeys(before, "after")
    assert assert_output_only_changes(before, after) == sorted(paths)


def test_output_replay_disallows_bank_extraction_and_pdf_opens():
    import fitz
    import pdfplumber

    from docmirror.plugins.bank_statement import extract_pipeline
    from docmirror.plugins.bank_statement.community_plugin import BankStatementCommunityPlugin

    with without_extraction():
        for call in (lambda: fitz.open("never.pdf"), lambda: pdfplumber.open("never.pdf"),
                     lambda: BankStatementCommunityPlugin().project_bundle(None),
                     lambda: extract_pipeline.run_bank_statement_extract(None, "", None)):
            with pytest.raises(AssertionError, match="extraction is forbidden"):
                call()


def test_replay_refuses_unbound_or_modified_input_artifacts(tmp_path, monkeypatch):
    from scripts.validate import bank_consumer_exports as runner

    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    origin = tmp_path / "tmp" / "pdfs" / "case"
    origin.mkdir(parents=True)
    path = origin / "parse_result.json"
    path.write_text("{}", encoding="utf-8")
    entry = {"status": "pass", "audit_status": "pass", "origin": str(origin),
             "artifact_hashes": {str(path): runner._sha(path)}}
    with pytest.raises(AssertionError, match="every required artifact"):
        _verify_inputs(entry)
    path.write_text("{\"modified\":true}", encoding="utf-8")
    with pytest.raises(AssertionError, match="checksum or ownership"):
        _verify_inputs(entry)
