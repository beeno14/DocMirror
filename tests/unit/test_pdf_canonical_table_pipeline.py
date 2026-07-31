from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from docmirror.evidence.plane import EvidencePlaneBuilder, _pymupdf_watermark_line_keys
from docmirror.input.extraction.extractor import CoreExtractor
from docmirror.models.sealed import seal_parse_result
from docmirror.output.community_bundle import project_community_bundle
from docmirror.output.mirror_projector import project_mirror
from docmirror.tables.native_pdf_candidates import extract_pymupdf_table_candidates


def _write_vector_table_pdf(path) -> None:
    canvas_mod = pytest.importorskip("reportlab.pdfgen.canvas")
    canvas = canvas_mod.Canvas(str(path), pagesize=(300, 220))
    xs = [30, 120, 270]
    ys = [180, 150, 120]
    for x in xs:
        canvas.line(x, ys[-1], x, ys[0])
    for y in ys:
        canvas.line(xs[0], y, xs[-1], y)
    canvas.drawString(40, 160, "Name")
    canvas.drawString(130, 160, "Alice")
    canvas.drawString(40, 130, "Status")
    canvas.drawString(130, 130, "Active")
    canvas.save()


def test_native_pdf_table_evidence_survives_into_parse_result(tmp_path) -> None:
    pytest.importorskip("fitz")
    path = tmp_path / "vector-table.pdf"
    _write_vector_table_pdf(path)

    plane = EvidencePlaneBuilder().build(path)
    candidates = plane.evidence.indexes["table_candidates"]

    assert plane.counts["vector_atoms"] > 0
    assert plane.counts["table_candidates"] == 1
    assert candidates[0]["geometry"]["cell_bboxes"][0][0]
    assert candidates[0]["geometry"]["cell_token_ids"][0][0]
    assert plane.evidence.vector_atoms[0].metadata["geometry"]["items"]

    result = asyncio.run(CoreExtractor().extract_parse_result(path))

    assert result.total_tables == 1
    assert result.parser_info.table_engine == "pymupdf_native"
    assert result.evidence_plane is not None
    assert len(result.evidence_plane.evidence.vector_atoms) == plane.counts["vector_atoms"]
    assert result.document_flow is not None
    assert any(node.type == "physical_table" for node in result.document_flow.nodes)

    mirror = project_mirror(seal_parse_result(result), source_filename=str(path))
    assert len(mirror["evidence"]["vector_atoms"]) == plane.counts["vector_atoms"]
    assert mirror["evidence"]["indexes"]["table_candidates"]

    table = result.pages[0].tables[0]
    table_token_ids = {token_id for row in table.rows for cell in row.cells for token_id in cell.token_ids}
    body_token_ids = {token_id for text in result.pages[0].texts for token_id in text.evidence_ids}
    assert table_token_ids
    assert not table_token_ids & body_token_ids

    markdown = project_community_bundle(seal_parse_result(result), document_id="doc_vector").render_markdown()
    assert "|" in markdown
    assert "<table>" not in markdown
    assert markdown.count("Alice") == 1
    assert markdown.count("Active") == 1


def test_missing_table_reconstruction_is_not_reported_as_ready(tmp_path, monkeypatch) -> None:
    pytest.importorskip("fitz")
    path = tmp_path / "vector-table.pdf"
    _write_vector_table_pdf(path)
    monkeypatch.setattr("docmirror.input.extraction.extractor._native_table_blocks", lambda *_a, **_kw: [])

    result = asyncio.run(CoreExtractor().extract_parse_result(path))

    assert result.status.value == "partial"
    assert "native_table_evidence_not_reconstructed" in result.parser_info.warnings
    gate = result.parser_info.structure["table_reconstruction_gate"]
    assert gate["applicable"] is True
    assert gate["passed"] is False


def test_translucent_watermark_lines_are_excluded_from_native_table_text() -> None:
    class FakePage:
        def get_text(self, mode):
            assert mode == "dict"
            return {
                "blocks": [
                    {
                        "number": 0,
                        "type": 0,
                        "lines": [
                            {"dir": (1.0, 0.0), "spans": [{"text": "Alice", "alpha": 255}]},
                            {"dir": (0.86, -0.5), "spans": [{"text": "WATERMARK", "alpha": 51}]},
                        ],
                    }
                ]
            }

        def find_tables(self):
            native = SimpleNamespace(
                bbox=(0, 0, 200, 40),
                rows=[SimpleNamespace(cells=[(0, 0, 100, 40), (100, 0, 200, 40)])],
                extract=lambda: [["Alice\nWATERMARK", "100.00"]],
            )
            return SimpleNamespace(tables=[native])

    page = FakePage()
    assert _pymupdf_watermark_line_keys(page) == {(0, 1)}

    atoms = [
        SimpleNamespace(
            id="ev:name",
            text="Alice",
            bbox=[10, 10, 40, 20],
            metadata={"block_no": 0, "line_no": 0, "word_no": 0},
        ),
        SimpleNamespace(
            id="ev:amount",
            text="100.00",
            bbox=[120, 10, 160, 20],
            metadata={"block_no": 1, "line_no": 0, "word_no": 0},
        ),
    ]
    candidates = extract_pymupdf_table_candidates(
        page,
        page_number=1,
        page_id="p1",
        normalize_bbox=lambda bbox: bbox,
        text_atoms=atoms,
        vector_atoms=[],
        prefer_owned_text=True,
    )

    assert candidates[0]["rows"] == [["Alice", "100.00"]]
    assert candidates[0]["geometry"]["text_source"] == "owned_non_watermark_atoms"
