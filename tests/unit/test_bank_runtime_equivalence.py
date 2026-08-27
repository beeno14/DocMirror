"""Negative controls for the fresh-run preservation auditor (no real PDFs)."""

import json
from copy import deepcopy

import pytest

from docmirror.models.entities.parse_result import PageContent, ParseResult, ParserInfo, ProvenanceInfo, TextBlock
from scripts.validate.bank_runtime_equivalence import (
    _perception_hashes,
    _policy_from_dict,
    _replay_code_changes,
    assert_equal,
    assert_parse_preserved,
    assert_semantic_preserved,
    main,
)


def _source():
    return ParseResult(
        raw_text="Account 000123; 2026-08-01 -12.00 988.00",
        pages=[
            PageContent(texts=[TextBlock(content="2026-08-01 -12.00 988.00", evidence_ids=["ev:0001:text:000001"])])
        ],
        provenance=ProvenanceInfo(file_path="old/source.pdf", file_hash="same-content"),
        parser_info=ParserInfo(
            elapsed_ms=100,
            structure={"step_timings": {"extract": 80}, "primary": "table_led"},
            options={"ocr_mode": "off", "selected_pages": [1]},
        ),
    )


def test_runtime_audit_ignores_only_nonfactual_location_and_timings():
    expected = _source()
    actual = deepcopy(expected)
    actual.provenance.file_path = "new/source.pdf"
    actual.parser_info.elapsed_ms = 20
    actual.parser_info.structure["step_timings"] = {"extract": 10}
    assert_parse_preserved(expected, actual)


@pytest.mark.parametrize("pages", [None, "1-3,5-", "last:2"])
def test_saved_policy_roundtrip_preserves_all_fact_options(pages):
    from docmirror.input.entry.options import normalize_parse_policy

    original = normalize_parse_policy(pages=pages, ocr="off", doc_type="bank_statement", doc_type_policy="force")
    loaded = _policy_from_dict(json.loads(json.dumps(original.to_dict())))
    assert loaded == original
    assert loaded.fingerprint() == original.fingerprint()


@pytest.mark.parametrize(
    "change",
    [
        lambda value: setattr(value.pages[0].texts[0], "content", "2026-08-01 -1.00 988.00"),
        lambda value: setattr(value.pages[0].texts[0], "evidence_ids", ["wrong-page-ownership"]),
        lambda value: setattr(value.pages[0], "page_number", 2),
        lambda value: value.parser_info.options.update(ocr_mode="auto"),
        lambda value: value.parser_info.structure.update(primary="other_route"),
        lambda value: value.parser_info.warnings.append("new warning"),
    ],
)
def test_runtime_audit_detects_value_provenance_and_routing_changes(change):
    expected = _source()
    actual = deepcopy(expected)
    change(actual)
    with pytest.raises(AssertionError):
        assert_parse_preserved(expected, actual)


def test_runtime_audit_reports_location_not_private_values():
    with pytest.raises(AssertionError, match=r"\$\.account") as error:
        assert_equal({"account": "000123"}, {"account": "000124"}, "business row")
    assert "000123" not in str(error.value)
    assert "000124" not in str(error.value)


def test_projection_only_retry_never_accepts_changed_perception_code():
    before = {"docmirror/evidence/plane.py": "old", "docmirror/plugins/bank_statement/work_cache.py": "old"}
    projection_change = {**before, "docmirror/plugins/bank_statement/work_cache.py": "new"}
    assert _perception_hashes(before) == _perception_hashes(projection_change)
    assert _perception_hashes(before) != _perception_hashes({**before, "docmirror/evidence/plane.py": "new"})


def test_runtime_runner_does_not_allow_all_corpus(tmp_path):
    with pytest.raises(SystemExit) as error:
        main(["--tier", "all", "--output-root", str(tmp_path / "not-created")])
    assert error.value.code == 2
    assert not (tmp_path / "not-created").exists()


def test_semantic_snapshot_comparison_verifies_each_binding_without_mutation():
    expected = {
        "source": {"fingerprint": "old", "file": {"name": "same.pdf"}},
        "rows": [{"amount": "12.00", "raw": {"value": "12.00"}}],
    }
    actual = deepcopy(expected)
    actual["source"]["fingerprint"] = "new"
    before = deepcopy((expected, actual))
    assert_semantic_preserved(expected, actual, "old", "new")
    assert (expected, actual) == before


@pytest.mark.parametrize(
    "change",
    [
        lambda value: value["source"].update(fingerprint="unbound"),
        lambda value: value["source"]["file"].update(name="other.pdf"),
        lambda value: value["rows"][0].update(amount="13.00"),
        lambda value: value["rows"][0]["raw"].update(value="13.00"),
    ],
)
def test_semantic_snapshot_comparison_cannot_hide_lost_values_or_wrong_binding(change):
    expected = {
        "source": {"fingerprint": "old", "file": {"name": "same.pdf"}},
        "rows": [{"amount": "12.00", "raw": {"value": "12.00"}}],
    }
    actual = deepcopy(expected)
    actual["source"]["fingerprint"] = "new"
    change(actual)
    with pytest.raises(AssertionError):
        assert_semantic_preserved(expected, actual, "old", "new")


@pytest.mark.parametrize(
    "path",
    [
        "docmirror/evidence/plane.py",
        "docmirror/plugins/bank_statement/style_registry.py",
        "docmirror/plugins/_base/projector.py",
        "docmirror/output/community_bundle.py",
    ],
)
def test_saved_replay_refuses_changed_extraction_or_output_code(path):
    with pytest.raises(AssertionError, match="artifact-only replay is insufficient"):
        _replay_code_changes({path: "old"}, {path: "new"})


def test_saved_replay_separates_other_providers_and_requires_serializer_reevaluation():
    serializer = "docmirror/plugins/_runtime/evidence_access.py"
    other_provider = "docmirror/plugins/credit_report/community_plugin.py"
    assert _replay_code_changes({other_provider: "old"}, {other_provider: "new"}) == []
    assert _replay_code_changes({serializer: "old"}, {serializer: "new"}) == [serializer]
