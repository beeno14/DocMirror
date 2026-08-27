"""Lightweight native PDF equivalence, indexing, and worker-failure tests."""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from docmirror.evidence import plane as plane_module
from docmirror.evidence.plane import EvidencePlaneBuilder, _iter_native_pdf_pages
from docmirror.input.extraction.extractor import _native_table_blocks
from docmirror.models.mirror.vnext import EvidenceAtom, EvidenceStore
from docmirror.plugins._runtime.evidence_access import evidence_payload, text_atoms


@pytest.fixture
def native_pdf(tmp_path):
    fitz = pytest.importorskip("fitz")
    path = tmp_path / "native-equivalence.pdf"
    with fitz.open() as document:
        for index in range(9):
            page = document.new_page(width=400, height=400)
            if index == 4:
                continue
            page.insert_text((20, 30), f"Statement page {index + 1}; account 00000123")
            for x in (20, 140, 260, 380):
                page.draw_line((x, 60), (x, 140))
            for y in (60, 80, 100, 120, 140):
                page.draw_line((20, y), (380, y))
            for row, cells in enumerate(
                [
                    ["Date", "Amount", "Balance"],
                    ["2026-08-01", "-12.00", "988.00"],
                    ["2026-08-02", "0.00", "988.00"],
                    ["2026-08-03", "12.00", "1000.00"],
                ]
            ):
                for column, value in enumerate(cells):
                    page.insert_text((24 + 120 * column, 75 + 20 * row), value, fontsize=9)
            if index == 2:
                page.insert_text((25, 250), "FAINT WATERMARK", fill_opacity=0.1)
            if index == 3:
                page.set_rotation(90)
            if index == 5:
                pixmap = fitz.Pixmap(fitz.csRGB, (0, 0, 4, 4), False)
                pixmap.clear_with(0)
                page.insert_image(fitz.Rect(20, 180, 40, 200), pixmap=pixmap)
            if index in (6, 7):
                page.set_rotation(180 if index == 6 else 270)
        document.save(path)
    return path


@pytest.mark.parametrize("workers", [2, 4])
def test_native_process_results_exactly_match_serial_evidence(native_pdf, workers):
    serial_builder = EvidencePlaneBuilder(max_page_workers=1)
    parallel_builder = EvidencePlaneBuilder(max_page_workers=workers)
    serial = serial_builder.build(native_pdf)
    parallel = parallel_builder.build(native_pdf)
    assert asdict(parallel) == asdict(serial)
    assert serial.counts["pages"] == 9
    assert serial.counts["table_candidates"] >= 7
    assert serial.counts["image_atoms"] == 1
    assert serial.pages[3].original_rotation == 90
    if (plane_module.os.cpu_count() or 1) > 1:
        assert parallel_builder.execution_stats["mode"] == "process"
        assert parallel_builder.execution_stats["fallback_chunks"] == 0


def test_reusing_native_drawings_matches_original_table_detection(native_pdf, monkeypatch):
    from docmirror.tables import native_pdf_candidates

    optimized = EvidencePlaneBuilder().build(native_pdf)
    original = native_pdf_candidates.extract_pymupdf_table_candidates

    def without_reuse(page, **kwargs):
        kwargs.pop("paths", None)
        return original(page, **kwargs)

    monkeypatch.setattr(native_pdf_candidates, "extract_pymupdf_table_candidates", without_reuse)
    baseline = EvidencePlaneBuilder().build(native_pdf)
    assert asdict(optimized) == asdict(baseline)


def test_drawing_reuse_retains_older_page_adapter_compatibility():
    from docmirror.tables.native_pdf_candidates import extract_pymupdf_table_candidates

    calls = []

    class LegacyPage:
        def find_tables(self):
            calls.append(True)
            return SimpleNamespace(tables=[])

    assert (
        extract_pymupdf_table_candidates(
            LegacyPage(),
            page_number=1,
            page_id="page:0001",
            normalize_bbox=lambda bbox: bbox,
            text_atoms=[],
            vector_atoms=[],
            paths=[],
        )
        == []
    )
    assert calls == [True]


@pytest.mark.parametrize("failure", ["exception", "missing", "reordered"])
def test_worker_failures_retry_only_the_affected_chunk_in_source_order(monkeypatch, tmp_path, failure):
    import concurrent.futures

    fake_pages = list(range(40))
    serial_calls = []
    queue = {"outstanding": 0, "peak": 0, "submitted": 0}

    def item(index):
        return SimpleNamespace(page=SimpleNamespace(page_index=index))

    def serial(page, index):
        serial_calls.append(index)
        return item(index)

    class TrackingFuture(Future):
        def result(self, timeout=None):
            queue["outstanding"] -= 1
            return super().result(timeout)

    class FakePool:
        def __init__(self, **kwargs):
            assert kwargs["max_workers"] == 2
            assert kwargs["mp_context"].get_start_method() == "spawn"

        def submit(self, function, path, start, stop):
            queue["submitted"] += 1
            queue["outstanding"] += 1
            queue["peak"] = max(queue["peak"], queue["outstanding"])
            future = TrackingFuture()
            values = [item(index) for index in range(start, stop)]
            if start == 4:
                if failure == "exception":
                    future.set_exception(RuntimeError("worker interrupted"))
                    return future
                values = values[:-1] if failure == "missing" else list(reversed(values))
            future.set_result(values)
            return future

        def shutdown(self, **kwargs):
            assert kwargs == {"wait": True, "cancel_futures": True}

    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", FakePool)
    monkeypatch.setattr(plane_module, "_extract_native_pdf_page", serial)
    monkeypatch.setattr(plane_module.os, "cpu_count", lambda: 4)
    stats = {}
    results = list(_iter_native_pdf_pages(tmp_path / "not-read.pdf", fake_pages, 2, stats))
    assert [value.page.page_index for value in results] == list(range(40))
    assert serial_calls == [4, 5, 6, 7]
    assert stats["fallback_chunks"] == 1
    assert queue["peak"] <= 4
    assert queue["outstanding"] == 0


def test_pool_start_failure_keeps_complete_serial_output(monkeypatch, tmp_path):
    import concurrent.futures

    def unavailable(**kwargs):
        raise OSError("process creation unavailable")

    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", unavailable)
    monkeypatch.setattr(plane_module.os, "cpu_count", lambda: 4)
    monkeypatch.setattr(plane_module, "_extract_native_pdf_page", lambda page, index: index)
    stats = {}
    assert list(_iter_native_pdf_pages(tmp_path / "not-read.pdf", list(range(8)), 2, stats)) == list(range(8))
    assert stats["mode"] == "serial"
    assert stats["fallback_chunks"] == 1


def test_text_atom_projection_does_not_serialize_unrelated_table_geometry(monkeypatch):
    evidence = EvidenceStore(
        text_atoms=[
            EvidenceAtom(
                id="ev:0001:text:000001",
                kind="text_token",
                source_kind="pdf_native",
                page_id="page:0001",
                text="000123",
                bbox=[1, 2, 3, 4],
                metadata={"nullable": None, "zero": 0},
            )
        ],
        indexes={"table_candidates": [{"geometry": {"large": [1, 2, 3]}}]},
    )
    result = SimpleNamespace(evidence_plane=SimpleNamespace(evidence=evidence))
    expected = evidence_payload(result)["text_atoms"]

    original_dump = EvidenceStore.model_dump

    def selected_dump(self, *args, **kwargs):
        assert kwargs.get("include") == {"text_atoms"}
        return original_dump(self, *args, **kwargs)

    monkeypatch.setattr(EvidenceStore, "model_dump", selected_dump)
    assert text_atoms(result) == expected
    first = text_atoms(result)
    first[0]["metadata"]["zero"] = 99
    assert text_atoms(result)[0]["metadata"]["zero"] == 0


def test_text_atom_projection_preserves_declared_schema_for_extension_atoms():
    class ExtensionAtom(EvidenceAtom):
        extension_detail: str = "subclass-only metadata"

    store = EvidenceStore(
        text_atoms=[
            ExtensionAtom(
                id="source:1",
                kind="text_token",
                source_kind="pdf_native",
                page_id="page:0001",
                text="000123",
            )
        ]
    )
    result = SimpleNamespace(evidence_plane=SimpleNamespace(evidence=store))
    expected = evidence_payload(result)["text_atoms"]
    assert "extension_detail" not in expected[0]
    assert text_atoms(result) == expected


def test_text_atom_projection_retains_custom_store_serializer():
    class CustomStore(EvidenceStore):
        def model_dump(self, **kwargs):
            return {"text_atoms": [{"text": "custom serialized value"}]}

    result = SimpleNamespace(evidence_plane=SimpleNamespace(evidence=CustomStore()))
    assert text_atoms(result) == [{"text": "custom serialized value"}]


def test_unrecoverable_native_page_error_is_not_silently_dropped(monkeypatch, tmp_path):
    import concurrent.futures

    def unavailable(**kwargs):
        raise OSError("process creation unavailable")

    def damaged_page(page, index):
        raise ValueError("unrecoverable source page")

    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", unavailable)
    monkeypatch.setattr(plane_module.os, "cpu_count", lambda: 4)
    monkeypatch.setattr(plane_module, "_extract_native_pdf_page", damaged_page)
    with pytest.raises(ValueError, match="unrecoverable source page"):
        list(_iter_native_pdf_pages(tmp_path / "not-read.pdf", list(range(8)), 2, {}))


def test_indexed_table_lookup_preserves_original_projection():
    candidates = [
        {
            "candidate_id": f"table:{page}",
            "page_id": f"page:{page:04d}",
            "rows": [["Date", "Amount"], ["2026-08-01", "1.00"]],
            "bbox": [1, 2, 3, 4],
        }
        for page in range(1, 5)
    ]
    plane = SimpleNamespace(evidence=SimpleNamespace(indexes={"table_candidates": candidates}))
    original = _native_table_blocks(plane, page_id="page:0003", page_number=3)
    indexed = _native_table_blocks(plane, page_id="page:0003", page_number=3, candidates=[candidates[2]])
    assert indexed == original
