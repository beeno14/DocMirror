from __future__ import annotations

from types import SimpleNamespace

import pytest

from docmirror.plugins.credit_report.personal_detail_scanned.context import (
    PersonalDetailExtractionContext,
)
from docmirror.plugins.credit_report.personal_detail_scanned.page_reocr import (
    OneShotPageReOCRRegistry,
)


def test_registry_requires_an_explicit_report_derived_page_budget() -> None:
    with pytest.raises(TypeError, match="max_pages"):
        OneShotPageReOCRRegistry()  # type: ignore[call-arg]


def test_registry_runs_successful_producer_once_and_tracks_all_consumers() -> None:
    calls = 0
    registry = OneShotPageReOCRRegistry(max_pages=4)

    def produce():
        nonlocal calls
        calls += 1
        return {"page": 1, "lines": [{"text": "value"}]}, "completed", {"ocr_invocations": 1}

    first = registry.resolve(page_key="page:1", logical_page=1, reason="registration", producer=produce)
    second = registry.resolve(page_key="page:1", logical_page=1, reason="schema", producer=produce)

    assert calls == 1
    assert first == second
    audit = registry.audit()
    assert audit["page_reocr_engine_invocation_count"] == 1
    assert audit["max_ocr_invocations_per_page"] == 1
    assert audit["page_reocr_requests"][0]["reasons"] == ["registration", "schema"]


def test_empty_or_failed_attempt_is_terminal_and_never_retried() -> None:
    calls = 0
    registry = OneShotPageReOCRRegistry(max_pages=4)

    def produce():
        nonlocal calls
        calls += 1
        return None, "ocr_empty", {"ocr_invocations": 1}

    assert registry.resolve(page_key="page:1", logical_page=1, reason="registration", producer=produce) is None
    assert registry.resolve(page_key="page:1", logical_page=1, reason="native_parser", producer=produce) is None
    assert calls == 1
    assert registry.audit()["page_reocr_requests"][0]["status"] == "ocr_empty"


def test_registry_rejects_a_producer_that_claims_multiple_ocr_invocations() -> None:
    registry = OneShotPageReOCRRegistry(max_pages=4)

    with pytest.raises(RuntimeError, match="violated one-shot contract"):
        registry.resolve(
            page_key="page:1",
            logical_page=1,
            reason="test",
            producer=lambda: (None, "ocr_empty", {"ocr_invocations": 2}),
        )


def test_producer_exception_is_terminal_and_not_retried() -> None:
    calls = 0
    registry = OneShotPageReOCRRegistry(max_pages=4)

    def fail():
        nonlocal calls
        calls += 1
        raise RuntimeError("engine failure")

    assert registry.resolve(page_key="page:1", logical_page=1, reason="registration", producer=fail) is None
    assert registry.resolve(page_key="page:1", logical_page=1, reason="schema", producer=fail) is None
    assert calls == 1
    assert registry.audit()["page_reocr_requests"][0]["status"] == "producer_failed"


class _FrozenResolver:
    def __init__(self) -> None:
        self.image = SimpleNamespace(shape=(100, 200, 3))

    def __call__(self, logical_page: int):
        return {
            "image": self.image,
            "page_width": 100.0,
            "page_height": 50.0,
            "logical_page": logical_page,
            "source_page": logical_page,
            "coordinate_transform": {
                "decomposition": {"selected_rotation": 0},
            },
        }

    @staticmethod
    def page_key(logical_page: int) -> str:
        return f"source:{logical_page}:crop:0:0:100:50:rotation:0"


def test_context_shares_one_page_reocr_between_business_repair_consumers(monkeypatch) -> None:
    calls = 0

    def recognize(_image):
        nonlocal calls
        calls += 1
        return ([{"text": "报告编号", "confidence": 0.95, "bbox": [10, 10, 80, 30]}], 100.0)

    monkeypatch.setattr(
        "docmirror.plugins.credit_report.personal_detail_scanned.context._single_page_ocr",
        recognize,
    )
    context = object.__new__(PersonalDetailExtractionContext)
    context._page_image_resolver = _FrozenResolver()
    context._page_reocr_registry = OneShotPageReOCRRegistry(max_pages=4)
    context._ocr_correction_overlay = SimpleNamespace(audit=lambda: {})
    context.source_page_by_logical = {1: 1}

    first = context.full_page_ocr_evidence({1}, reason="business_field_evidence_insufficient")
    second = context.full_page_ocr_evidence({1}, reason="business_schema_template_unresolved")

    assert calls == 1
    assert first[0]["lines"] == second[0]["lines"]
    assert context._page_reocr_registry.audit()["max_ocr_invocations_per_page"] == 1
    context.parse_result = SimpleNamespace(
        pages=[
            SimpleNamespace(
                page_number=1,
                source_page_number=1,
                coordinate_transform={
                    "deskew_method": "hough_lines_p_v1",
                    "deskew_applied": True,
                    "deskew_angle": -0.75,
                    "deskew_reason": "applied",
                    "decomposition": {"deskew_support_line_count": 7},
                },
            )
        ]
    )
    public_audit = context.ocr_correction_audit()
    assert public_audit["page_reocr_failures"] == []
    assert "page_reocr_requests" not in public_audit
    assert public_audit["core_ocr_deskew"] == [
        {
            "logical_page": 1,
            "source_page": 1,
            "method": "hough_lines_p_v1",
            "applied": True,
            "angle": -0.75,
            "reason": "applied",
            "support_line_count": 7,
        }
    ]


def test_single_page_reocr_deskews_before_ocr_and_restores_bboxes(monkeypatch) -> None:
    import numpy as np

    from docmirror.plugins.credit_report.personal_detail_scanned import context as context_module

    image = np.zeros((100, 200, 3), dtype=np.uint8)
    corrected = np.ones_like(image)
    observed = []
    deskew = {
        "method": "hough_lines_p_v1",
        "applied": True,
        "angle": -1.0,
        "reason": "applied",
        "support_line_count": 8,
        "forward_matrix": [[1.0, 0.0, 2.0], [0.0, 1.0, 3.0], [0.0, 0.0, 1.0]],
        "inverse_matrix": [[1.0, 0.0, -2.0], [0.0, 1.0, -3.0], [0.0, 0.0, 1.0]],
    }

    monkeypatch.setattr(
        "docmirror.layout.normalization.hough_deskew_image",
        lambda supplied: (corrected, deskew),
    )

    def recognize(supplied, *, source):
        observed.append((supplied, source))
        return [{"text": "value", "confidence": 0.9, "bbox": [12.0, 13.0, 42.0, 33.0]}]

    monkeypatch.setattr("docmirror.ocr.repair.recognizers.rapidocr_recognize", recognize)

    words, _score, audit = context_module._single_page_ocr(image)

    assert observed == [(corrected, "personal_detail_page_reocr_once")]
    assert words[0]["bbox"] == pytest.approx([10.0, 10.0, 40.0, 30.0])
    assert audit == {
        "method": "hough_lines_p_v1",
        "applied": True,
        "angle": -1.0,
        "reason": "applied",
        "support_line_count": 8,
    }
