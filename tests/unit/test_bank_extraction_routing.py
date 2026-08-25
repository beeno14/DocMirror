# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts for the pre-context digital/scanned bank pipeline split."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from docmirror.models.entities.parse_result import (
    CanonicalEvidencePlane,
    DocumentEntities,
    EvidencePageSnapshot,
    ExtractionMethod,
    PageContent,
    ParseResult,
    ParserInfo,
    TextBlock,
)
from docmirror.models.mirror.vnext import EvidenceAtom, EvidenceStore
from docmirror.models.sealed import seal_parse_result
from docmirror.plugins.bank_statement.community_plugin import BankStatementCommunityPlugin
from docmirror.plugins.bank_statement.context import (
    StyleContext,
    build_digital_style_context,
    build_scanned_style_context,
)
from docmirror.plugins.bank_statement.extraction_dispatch import (
    DIGITAL_POLICY,
    SCANNED_POLICY,
    BankExtractionRoute,
    resolve_bank_extraction_route,
)
from docmirror.plugins.bank_statement.ltro import ReconstructionMeta
from docmirror.plugins.bank_statement.style_detector import StyleDetectionResult
from docmirror.plugins.bank_statement.style_registry import BankStyleParserRegistry, _expected_rows


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        (ExtractionMethod.DIGITAL, BankExtractionRoute.DIGITAL),
        (ExtractionMethod.OCR, BankExtractionRoute.SCANNED),
        (ExtractionMethod.IMAGE, BankExtractionRoute.SCANNED),
    ],
)
def test_resolve_bank_extraction_route(method: ExtractionMethod, expected: BankExtractionRoute) -> None:
    result = ParseResult(parser_info=ParserInfo(extraction_method=method))

    assert resolve_bank_extraction_route(result) is expected


def test_hybrid_route_is_rejected_instead_of_mixing_pipelines() -> None:
    result = ParseResult(parser_info=ParserInfo(extraction_method=ExtractionMethod.HYBRID))

    with pytest.raises(ValueError, match="hybrid bank statements are not supported"):
        resolve_bank_extraction_route(result)


@pytest.mark.parametrize("blank_page_numbers", [(1,), (2,), (1, 3)])
def test_hybrid_route_treats_empty_pages_as_neutral(blank_page_numbers: tuple[int, ...]) -> None:
    native_page = next(number for number in (1, 2, 3) if number not in blank_page_numbers)
    pages = [
        PageContent(
            page_number=number,
            page_mode="native_text" if number == native_page else "scanned_ocr",
            texts=[TextBlock(content="source ledger", bbox=[0, 0, 10, 10])] if number == native_page else [],
        )
        for number in sorted({native_page, *blank_page_numbers})
    ]
    result = ParseResult(
        parser_info=ParserInfo(extraction_method=ExtractionMethod.HYBRID),
        pages=pages,
    )

    assert resolve_bank_extraction_route(result) is BankExtractionRoute.DIGITAL


def test_hybrid_route_still_rejects_content_bearing_native_and_scanned_pages() -> None:
    result = ParseResult(
        parser_info=ParserInfo(extraction_method=ExtractionMethod.HYBRID),
        pages=[
            PageContent(
                page_number=1,
                page_mode="native_text",
                texts=[TextBlock(content="native ledger", bbox=[0, 0, 10, 10])],
            ),
            PageContent(
                page_number=2,
                page_mode="scanned_ocr",
                texts=[TextBlock(content="OCR ledger", bbox=[0, 0, 10, 10])],
            ),
        ],
    )

    with pytest.raises(ValueError, match="hybrid bank statements are not supported"):
        resolve_bank_extraction_route(result)


def test_hybrid_projection_fails_before_context_construction(monkeypatch) -> None:
    import docmirror.plugins.bank_statement.extract_pipeline as pipeline

    monkeypatch.setattr(
        pipeline,
        "build_digital_style_context",
        lambda *_args, **_kwargs: pytest.fail("hybrid projection built digital context"),
    )
    monkeypatch.setattr(
        pipeline,
        "build_scanned_style_context",
        lambda *_args, **_kwargs: pytest.fail("hybrid projection built scanned context"),
    )
    sealed = seal_parse_result(
        ParseResult(
            parser_info=ParserInfo(extraction_method=ExtractionMethod.HYBRID),
            entities=DocumentEntities(document_type="bank_statement"),
        )
    )

    with pytest.raises(ValueError, match="hybrid bank statements are not supported"):
        BankStatementCommunityPlugin().project(sealed)


def test_route_resolution_happens_before_context_construction(monkeypatch) -> None:
    import docmirror.plugins.bank_statement.extract_pipeline as pipeline

    calls: list[str] = []

    def resolve(_parse_result):
        calls.append("resolve")
        return BankExtractionRoute.DIGITAL

    def build(_parse_result, _text):
        assert calls == ["resolve"]
        calls.append("context")
        raise RuntimeError("context sentinel")

    monkeypatch.setattr(pipeline, "resolve_bank_extraction_route", resolve)
    monkeypatch.setattr(pipeline, "build_digital_style_context", build)
    monkeypatch.setattr(
        pipeline,
        "build_scanned_style_context",
        lambda *_args, **_kwargs: pytest.fail("scanned context must not be built"),
    )

    with pytest.raises(RuntimeError, match="context sentinel"):
        pipeline.run_bank_statement_extract(ParseResult(), "", BankStatementCommunityPlugin())

    assert calls == ["resolve", "context"]


def _context(policy) -> StyleContext:
    return StyleContext(
        tables=[],
        full_text="",
        institution=None,
        page_count=1,
        parse_result=ParseResult(parser_info=ParserInfo(extraction_method=ExtractionMethod.DIGITAL)),
        reconstruction=ReconstructionMeta(source="none"),
        extraction_route=policy.route,
        extraction_policy=policy,
    )


def _detection() -> StyleDetectionResult:
    return StyleDetectionResult(primary_style="grid_standard", parser_chain=["grid_standard"])


def test_digital_registry_never_invokes_ocr_recovery(monkeypatch) -> None:
    import docmirror.plugins.bank_statement.style_registry as registry_module

    monkeypatch.setattr(
        registry_module,
        "recover_ocr_implicit_ledger_tables",
        lambda *_args, **_kwargs: pytest.fail("digital route invoked OCR implicit recovery"),
    )
    registry = BankStyleParserRegistry()

    registry.run_parser_chain(_detection(), _context(DIGITAL_POLICY), BankStatementCommunityPlugin())

    assert "ocr_implicit_table" not in registry.last_selection_diagnostics["candidate_counts"]


def test_scanned_registry_never_invokes_native_wide_recovery(monkeypatch) -> None:
    import docmirror.plugins.bank_statement.style_registry as registry_module

    monkeypatch.setattr(
        registry_module,
        "recover_wide_bank_tables",
        lambda *_args, **_kwargs: pytest.fail("scanned route invoked native PDF recovery"),
    )
    registry = BankStyleParserRegistry()

    registry.run_parser_chain(_detection(), _context(SCANNED_POLICY), BankStatementCommunityPlugin())

    assert "native_wide_table" not in registry.last_selection_diagnostics["candidate_counts"]


def test_digital_expected_rows_ignore_stale_ocr_cache() -> None:
    result = ParseResult(parser_info=ParserInfo(extraction_method=ExtractionMethod.DIGITAL))
    result.entities.domain_specific["_bank_ocr_implicit_recovery"] = {
        "status": "ready",
        "row_count": 999,
        "tables": [],
    }
    ctx = StyleContext(
        tables=[],
        full_text="",
        institution=None,
        page_count=1,
        parse_result=result,
        reconstruction=ReconstructionMeta(source="canonical_table", expected_primary_rows=7),
        extraction_route=DIGITAL_POLICY.route,
        extraction_policy=DIGITAL_POLICY,
    )

    assert _expected_rows(ctx) == 7


def test_digital_context_never_invokes_spaced_ocr_reconstruction(monkeypatch) -> None:
    import docmirror.plugins.bank_statement.ltro as ltro

    monkeypatch.setattr(
        ltro,
        "build_tables_from_spaced_ocr_text",
        lambda *_args, **_kwargs: pytest.fail("digital context invoked spaced OCR reconstruction"),
    )

    context = build_digital_style_context(ParseResult(), "")

    assert context.extraction_route is BankExtractionRoute.DIGITAL


def test_scanned_context_never_invokes_pipe_reconstruction(monkeypatch) -> None:
    import docmirror.plugins.bank_statement.ltro as ltro

    monkeypatch.setattr(
        ltro,
        "detect_pipe_header_in_text",
        lambda *_args, **_kwargs: pytest.fail("scanned context inspected native pipe text"),
    )
    monkeypatch.setattr(
        ltro,
        "build_tables_from_pipe_text",
        lambda *_args, **_kwargs: pytest.fail("scanned context invoked native pipe reconstruction"),
    )
    result = ParseResult(parser_info=ParserInfo(extraction_method=ExtractionMethod.OCR))

    context = build_scanned_style_context(result, "")

    assert context.extraction_route is BankExtractionRoute.SCANNED


def test_context_rejects_route_policy_mismatch() -> None:
    with pytest.raises(ValueError, match="route/policy mismatch"):
        StyleContext(
            tables=[],
            full_text="",
            institution=None,
            page_count=1,
            extraction_route=BankExtractionRoute.SCANNED,
            extraction_policy=DIGITAL_POLICY,
        )


def test_evidence_atoms_are_filtered_by_acquisition_route(monkeypatch) -> None:
    import docmirror.plugins.bank_statement.evidence_atom_table_recovery as recovery

    result = ParseResult(
        evidence_plane=CanonicalEvidencePlane(
            evidence=EvidenceStore(
                text_atoms=[
                    EvidenceAtom(
                        id="native",
                        source_kind="pdf_native",
                        page_id="page:0001",
                        text="native",
                        bbox=[0, 0, 10, 10],
                    ),
                    EvidenceAtom(
                        id="ocr",
                        source_kind="metadata_ocr_token",
                        page_id="page:0001",
                        text="ocr",
                        bbox=[0, 20, 10, 30],
                    ),
                ]
            )
        )
    )

    digital = recovery._atoms_by_page(result, source_route="digital")
    scanned = recovery._atoms_by_page(result, source_route="scanned")

    assert [atom["id"] for atom in digital["page:0001"]] == ["native"]
    assert [atom["id"] for atom in scanned["page:0001"]] == ["ocr"]


def test_generic_evidence_atoms_follow_sealed_page_content_mode() -> None:
    import docmirror.plugins.bank_statement.evidence_atom_table_recovery as recovery

    def result_for(content_mode: str) -> ParseResult:
        return ParseResult(
            evidence_plane=CanonicalEvidencePlane(
                pages=[
                    EvidencePageSnapshot(
                        page_id="page:0001",
                        page_index=0,
                        page_number=1,
                        content_mode=content_mode,
                    )
                ],
                evidence=EvidenceStore(
                    text_atoms=[
                        EvidenceAtom(
                            id="generic",
                            source_kind="parse_result_text",
                            page_id="page:0001",
                            text="ledger row",
                            bbox=[0, 0, 10, 10],
                        )
                    ]
                ),
            )
        )

    native = result_for("native_text")
    scanned = result_for("scanned_ocr")

    assert "page:0001" in recovery._atoms_by_page(native, source_route="digital")
    assert recovery._atoms_by_page(native, source_route="scanned") == {}
    assert "page:0001" in recovery._atoms_by_page(scanned, source_route="scanned")
    assert recovery._atoms_by_page(scanned, source_route="digital") == {}

    scanned.evidence_plane.evidence.text_atoms.extend(
        [
            EvidenceAtom(
                id="explicit-native",
                source_kind="pdf_native",
                page_id="page:0001",
                text="native layer",
                bbox=[0, 20, 10, 30],
            ),
            EvidenceAtom(
                id="explicit-ocr",
                source_kind="metadata_ocr_token",
                page_id="page:0001",
                text="ocr layer",
                bbox=[0, 40, 10, 50],
            ),
        ]
    )

    assert [
        atom["id"] for atom in recovery._atoms_by_page(scanned, source_route="scanned")["page:0001"]
    ] == ["generic", "explicit-ocr"]


def test_route_evidence_fallback_is_merged_for_missing_pages_only() -> None:
    import docmirror.plugins.bank_statement.evidence_atom_table_recovery as recovery

    result = ParseResult(
        pages=[
            PageContent(
                page_number=1,
                page_mode="native_text",
                texts=[TextBlock(content="page one fallback", bbox=[0, 0, 10, 10])],
            ),
            PageContent(
                page_number=2,
                page_mode="native_text",
                texts=[TextBlock(content="page two fallback", bbox=[0, 0, 10, 10])],
            ),
        ],
        evidence_plane=CanonicalEvidencePlane(
            evidence=EvidenceStore(
                text_atoms=[
                    EvidenceAtom(
                        id="sealed-page-one",
                        source_kind="pdf_native",
                        page_id="page:0001",
                        text="sealed page one",
                        bbox=[0, 0, 10, 10],
                    )
                ]
            )
        ),
    )

    grouped = recovery._atoms_by_page(result, source_route="digital")

    assert [atom["id"] for atom in grouped["page:0001"]] == ["sealed-page-one"]
    assert [atom["text"] for atom in grouped["page:0002"]] == ["page two fallback"]


def test_evidence_recovery_cache_is_namespaced_by_route() -> None:
    import docmirror.plugins.bank_statement.evidence_atom_table_recovery as recovery

    result = ParseResult()
    recovery._store_recovery_cache(
        result,
        [],
        [{"route": "digital"}],
        3,
        source_route="digital",
    )
    recovery._store_recovery_cache(
        result,
        [],
        [{"route": "scanned"}],
        7,
        source_route="scanned",
    )

    assert recovery.recovered_evidence_atom_expected_row_count(result, source_route="digital") == 3
    assert recovery.recovered_evidence_atom_expected_row_count(result, source_route="scanned") == 7
    assert recovery.recovered_evidence_atom_row_sources(result, source_route="digital") == [
        {"route": "digital"}
    ]
    assert recovery.recovered_evidence_atom_row_sources(result, source_route="scanned") == [
        {"route": "scanned"}
    ]
