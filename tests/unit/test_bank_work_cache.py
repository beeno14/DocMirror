"""Behavior and isolation contracts for request-local bank work reuse."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from docmirror.plugins.bank_statement import evidence_atom_table_recovery as atoms
from docmirror.plugins.bank_statement import style_registry as registry
from docmirror.plugins.bank_statement import wide_table_recovery as wide
from docmirror.plugins.bank_statement.context import StyleContext
from docmirror.plugins.bank_statement.ltro import ReconstructionMeta
from docmirror.plugins.bank_statement.work_cache import (
    active_bank_cache,
    bank_work_session,
    memoize_bank_document_work,
    reuse_bank_work,
)


def test_cached_results_are_isolated_and_only_live_for_one_request():
    result = SimpleNamespace()
    calls = []

    def compute():
        calls.append(True)
        return {"rows": [{"amount": "-12.00", "account": "000012"}]}

    with bank_work_session(result) as cache:
        first = reuse_bank_work(result, "rows", (), compute)
        first["rows"][0]["amount"] = "corrupted"
        second = reuse_bank_work(result, "rows", (), compute)
        assert second["rows"][0]["amount"] == "-12.00"
        second["rows"].clear()
        assert len(reuse_bank_work(result, "rows", (), compute)["rows"]) == 1
        assert cache.hits["rows"] == 2
    assert cache.entries == {}
    assert not active_bank_cache(result)
    with bank_work_session(result):
        assert reuse_bank_work(result, "rows", (), compute)["rows"]
    assert len(calls) == 2


def test_document_identity_and_arguments_are_part_of_reuse_boundary():
    result, other = SimpleNamespace(), SimpleNamespace()
    calls = []

    @memoize_bank_document_work
    def compute(document, *, source_route=None):
        calls.append((document, source_route))
        return [source_route]

    with bank_work_session(result):
        assert compute(result, source_route="digital") == ["digital"]
        assert compute(result, source_route="digital") == ["digital"]
        assert compute(result, source_route="scanned") == ["scanned"]
        assert compute(other, source_route="digital") == ["digital"]
        with bank_work_session(other):
            assert compute(other, source_route="digital") == ["digital"]
        assert compute(result, source_route="digital") == ["digital"]
    assert len(calls) == 4


def test_empty_source_reads_and_exceptions_are_not_cached():
    result = SimpleNamespace()
    outcomes = iter([OSError("temporary read failure"), [], ["recovered"]])

    def compute():
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    with bank_work_session(result):
        with pytest.raises(OSError):
            reuse_bank_work(result, "native", (), compute, cache_empty=False)
        assert reuse_bank_work(result, "native", (), compute, cache_empty=False) == []
        assert reuse_bank_work(result, "native", (), compute, cache_empty=False) == ["recovered"]
        assert reuse_bank_work(result, "native", (), compute, cache_empty=False) == ["recovered"]


def test_disabled_session_remains_disabled_inside_pipeline_session():
    result = SimpleNamespace()
    calls = []
    with bank_work_session(result, enabled=False):
        with bank_work_session(result):
            for _ in range(2):
                reuse_bank_work(result, "work", (), lambda: calls.append(True))
    assert len(calls) == 2


def test_simultaneous_requests_do_not_share_mutable_results():
    result = SimpleNamespace()

    async def one_request(value):
        with bank_work_session(result):
            first = reuse_bank_work(result, "same-key", (), lambda: [value])
            await asyncio.sleep(0)
            second = reuse_bank_work(result, "same-key", (), lambda: ["wrong"])
            assert first == second == [value]

    async def run():
        await asyncio.gather(one_request("left"), one_request("right"))

    asyncio.run(run())


def test_atom_recovery_replays_counts_and_provenance_without_reconstruction(monkeypatch):
    result = SimpleNamespace(entities=SimpleNamespace(domain_specific={}))
    calls = []
    expected_tables = [[["日期", "金额"], ["2026-08-01", "-12.00"]]]
    sources = [{"source_page": 1, "source_row_index": 2, "evidence_ids": ["source:a"]}]

    def recover(document, *, source_route):
        calls.append(source_route)
        atoms._store_recovery_cache(document, expected_tables, sources, 1, source_route=source_route)
        return deepcopy(expected_tables)

    monkeypatch.setattr(atoms, "_recover_evidence_atom_bank_tables", recover)
    with bank_work_session(result):
        first = atoms.recover_evidence_atom_bank_tables(result, source_route="digital")
        first[0][1][1] = "wrong"
        result.entities.domain_specific["_bank_evidence_atom_recovery:digital"] = {"status": "wrong"}
        second = atoms.recover_evidence_atom_bank_tables(result, source_route="digital")
        assert second == expected_tables
        assert atoms.recovered_evidence_atom_row_sources(result, source_route="digital") == sources
        assert atoms.recovered_evidence_atom_expected_row_count(result, source_route="digital") == 1
    assert calls == ["digital"]


def test_native_recovery_reuses_same_source_but_invalidates_changed_file(monkeypatch, tmp_path):
    path = tmp_path / "source.bin"
    path.write_bytes(b"original")
    result = SimpleNamespace()
    calls = []
    monkeypatch.setattr(wide, "_source_pdf_path", lambda _: path)

    def recover(document, text):
        calls.append(text)
        return [[[path.read_text(), text]]]

    monkeypatch.setattr(wide, "_recover_wide_bank_tables", recover)
    with bank_work_session(result):
        assert wide.recover_wide_bank_tables(result, "same") == [[["original", "same"]]]
        assert wide.recover_wide_bank_tables(result, "same") == [[["original", "same"]]]
        assert wide.recover_wide_bank_tables(result, "other") == [[["original", "other"]]]
        path.write_bytes(b"changed-source-size")
        assert wide.recover_wide_bank_tables(result, "same") == [[["changed-source-size", "same"]]]
    assert calls == ["same", "other", "same"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("full_text", "different text"),
        ("page_count", 2),
        ("prefer_context_tables", True),
        ("tables", [[["different"]]]),
        ("institution", "different institution"),
        ("reconstruction", ReconstructionMeta("pipe_text")),
    ],
)
def test_parser_reuse_preserves_context_boundaries_and_reconstruction_effects(monkeypatch, field, value):
    result, plugin = SimpleNamespace(), SimpleNamespace()
    ctx = StyleContext(
        tables=[[["date", "amount"], ["2026-08-01", "1.00"]]],
        full_text="source",
        institution=None,
        page_count=1,
        parse_result=result,
        reconstruction=ReconstructionMeta("canonical_table"),
    )
    calls = []

    def compute(parser_id, context, owner):
        calls.append(parser_id)
        context.reconstruction = ReconstructionMeta("pipe_text", expected_primary_rows=1)
        return ([{"金额": "1.00", "_source": {"source_page": 1}}], None)

    monkeypatch.setattr(registry, "_run_parser_uncached", compute)
    with bank_work_session(result):
        first_ctx, second_ctx = replace(ctx), replace(ctx)
        first, _ = registry._run_parser("grid_standard", first_ctx, plugin)
        first[0]["金额"] = "bad"
        second, _ = registry._run_parser("grid_standard", second_ctx, plugin)
        assert second[0]["金额"] == "1.00"
        assert first_ctx.reconstruction == second_ctx.reconstruction
        registry._run_parser("grid_standard", replace(ctx, **{field: value}), plugin)
    assert len(calls) == 2


def test_unhashable_extension_arguments_bypass_reuse_without_breaking_extraction():
    result = SimpleNamespace()
    with bank_work_session(result):
        assert reuse_bank_work(result, "extension", {"unhashable": True}, lambda: "ok") == "ok"
