"""Output and end-to-end synthetic extraction equivalence for C1/C3."""

from copy import deepcopy

from docmirror.models.entities.parse_result import DocumentEntities, ParseResult
from docmirror.models.sealed import SealedParseResult, seal_parse_result
from docmirror.plugins.bank_statement.community_plugin import BankStatementCommunityPlugin
from docmirror.plugins.bank_statement.extract_pipeline import run_bank_statement_extract
from docmirror.plugins.bank_statement.work_cache import bank_work_session
from tests.unit.test_community_bank_business_view import _business_bundle
from tests.unit.test_pipe_text_table_builder import _synthetic_boc_text


def test_synthetic_business_document_cache_on_off_equivalence():
    text = _synthetic_boc_text()
    original = ParseResult(raw_text=text, full_text=text, entities=DocumentEntities(document_type="bank_statement"))
    snapshots = []
    for enabled in (False, True):
        result = deepcopy(original)
        with bank_work_session(result, enabled=enabled) as cache:
            output = run_bank_statement_extract(result, text, BankStatementCommunityPlugin())
            snapshots.append(
                (
                    output.records,
                    output.identity_fields,
                    output.warnings,
                    output.style_meta.to_properties(),
                    output.candidate_diagnostics,
                )
            )
            if enabled:
                assert sum(cache.hits.values()) > 0
    assert snapshots[0] == snapshots[1]
    assert snapshots[0][0]


def test_business_renderers_reuse_public_view_without_changing_artifacts(monkeypatch):
    bundle, sealed = _business_bundle()
    semantic = bundle.semantic_payload()
    public = bundle.json_payload(semantic)
    expected_markdown = bundle.render_enhanced_markdown(semantic)
    expected_csvs = bundle.render_dataset_csvs(semantic)
    before = deepcopy((semantic, public))

    def unexpected_projection(*args, **kwargs):
        raise AssertionError("public business view was rebuilt")

    monkeypatch.setattr(bundle, "json_payload", unexpected_projection)
    assert bundle.render_enhanced_markdown(semantic, public_payload=public) == expected_markdown
    assert bundle.render_dataset_csvs(semantic, public_payload=public) == expected_csvs
    assert (semantic, public) == before
    assert sealed.verify_integrity()


def test_projector_reuses_support_read_view_but_retains_separate_bundle_view(monkeypatch):
    original = ParseResult(entities=DocumentEntities(document_type="bank_statement"))
    sealed = seal_parse_result(original)
    views = []
    method = SealedParseResult.to_read_view

    def read_view(self):
        result = method(self)
        views.append(result)
        return result

    monkeypatch.setattr(SealedParseResult, "to_read_view", read_view)
    bundle = BankStatementCommunityPlugin().project_bundle(sealed)
    assert bundle is not None
    assert len(views) == 2  # One derivation view and one isolated output view.
    assert views[0] is not views[1]
    views[0].raw_text = "mutated derivation"
    assert views[1].raw_text != views[0].raw_text
    assert sealed.verify_integrity()


def test_projector_retains_instance_support_override_without_deserializing(monkeypatch):
    sealed = seal_parse_result(ParseResult(entities=DocumentEntities(document_type="bank_statement")))
    projector = BankStatementCommunityPlugin()
    calls = []

    def reject(result):
        calls.append(result)
        return False

    def unexpected_view(*args, **kwargs):
        raise AssertionError("rejected custom support must not materialize a read view")

    monkeypatch.setattr(projector, "supports", reject)
    monkeypatch.setattr(SealedParseResult, "to_read_view", unexpected_view)
    assert projector.project_bundle(sealed) is None
    assert calls == [sealed]


def test_production_writer_materializes_business_view_only_once(monkeypatch, tmp_path):
    from docmirror.server.edition_outputs import _write_community_bundle_files

    bundle, sealed = _business_bundle()
    original = bundle.json_payload
    calls = []

    def counted(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(bundle, "json_payload", counted)
    paths = _write_community_bundle_files(bundle, tmp_path, file_id="001", document_id=bundle.document["id"])
    assert calls == [True]
    assert paths["community"].is_file()
    assert paths["enhanced_reading"].is_file()
    assert sealed.verify_integrity()
