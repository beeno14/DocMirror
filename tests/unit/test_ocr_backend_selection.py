from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from docmirror.ocr.vision import engine as selector


class _Backend:
    def __init__(self, name: str, *, fail: bool = False) -> None:
        self.engine_id = name
        self.source_id = name
        self.is_available = True
        self.fail = fail
        self.calls = 0

    def detect_image_words(self, _img, multi_scale: bool = False):
        self.calls += 1
        if self.fail:
            raise RuntimeError("primary failed")
        return [(0.0, 0.0, 1.0, 1.0, self.engine_id, 0, 0, 0, 0.9)]

    def force_recognize_regions(self, _img, regions):
        self.calls += 1
        if self.fail:
            raise RuntimeError("primary failed")
        return [(*map(float, region), self.engine_id, 0.9) for region in regions]


@pytest.fixture(autouse=True)
def _reset_selector(monkeypatch):
    for name in ("DOCMIRROR_OCR_BACKEND", "DOCMIRROR_PADDLE_DEVICE", "DOCMIRROR_PADDLE_PROFILE"):
        monkeypatch.delenv(name, raising=False)
    selector.reset_ocr_engine()
    yield
    selector.reset_ocr_engine()


def test_rapid_profile_never_probes_paddle(monkeypatch) -> None:
    rapid = _Backend("rapidocr_onnxruntime")
    monkeypatch.setenv("DOCMIRROR_OCR_BACKEND", "rapidocr")
    monkeypatch.setattr(selector, "_new_rapidocr", lambda: rapid)
    monkeypatch.setattr(
        selector,
        "paddle_runtime_status",
        lambda _device=None, **_kwargs: pytest.fail("Rapid-only selection must not probe Paddle"),
    )

    assert selector.get_ocr_engine() is rapid


def test_auto_uses_rapid_when_paddle_runtime_is_unavailable(monkeypatch) -> None:
    rapid = _Backend("rapidocr_onnxruntime")
    monkeypatch.setenv("DOCMIRROR_OCR_BACKEND", "auto")
    monkeypatch.setattr(selector, "paddle_runtime_status", lambda _device=None, **_kwargs: (False, "not_installed"))
    monkeypatch.setattr(selector, "_new_rapidocr", lambda: rapid)

    assert selector.get_ocr_engine() is rapid
    assert selector.get_ocr_backend_status(initialize=True)["fallback_reason"] == "not_installed"


def test_auto_selects_paddle_with_lazy_rapid_fallback(monkeypatch) -> None:
    paddle = _Backend("paddleocr")
    monkeypatch.setenv("DOCMIRROR_OCR_BACKEND", "auto")
    monkeypatch.setattr(selector, "paddle_runtime_status", lambda _device=None, **_kwargs: (True, "available"))
    monkeypatch.setattr(selector, "_new_paddleocr", lambda **_kwargs: paddle)
    monkeypatch.setattr(
        selector,
        "_new_rapidocr",
        lambda: pytest.fail("Fallback must remain lazy while Paddle succeeds"),
    )

    engine = selector.get_ocr_engine()
    assert engine.engine_id == "paddleocr"
    assert engine.detect_image_words(object())[0][4] == "paddleocr"


def test_auto_demotes_to_rapid_after_paddle_inference_failure(monkeypatch) -> None:
    paddle = _Backend("paddleocr", fail=True)
    rapid = _Backend("rapidocr_onnxruntime")
    monkeypatch.setenv("DOCMIRROR_OCR_BACKEND", "auto")
    monkeypatch.setattr(selector, "paddle_runtime_status", lambda _device=None, **_kwargs: (True, "available"))
    monkeypatch.setattr(selector, "_new_paddleocr", lambda **_kwargs: paddle)
    monkeypatch.setattr(selector, "_new_rapidocr", lambda: rapid)

    engine = selector.get_ocr_engine()
    words = engine.detect_image_words(object())

    assert words[0][4] == "rapidocr_onnxruntime"
    assert engine.engine_id == "rapidocr_onnxruntime"
    assert paddle.calls == 1
    assert rapid.calls == 1
    assert "primary failed" in str(engine.fallback_reason)
    assert engine.backend_revision == 1
    assert selector.get_ocr_backend_status(initialize=False)["fallback_reason"].endswith("primary failed")


def test_explicit_paddle_is_strict_when_unavailable(monkeypatch) -> None:
    monkeypatch.setenv("DOCMIRROR_OCR_BACKEND", "paddleocr")
    monkeypatch.setattr(selector, "paddle_runtime_status", lambda _device=None, **_kwargs: (False, "no_cuda_device"))

    with pytest.raises(selector.OCRBackendUnavailable, match="no_cuda_device"):
        selector.get_ocr_engine()


def test_legacy_backend_aliases_are_accepted(monkeypatch) -> None:
    rapid = _Backend("rapidocr_onnxruntime")
    monkeypatch.setenv("DOCMIRROR_OCR_BACKEND", "rapid")
    monkeypatch.setattr(selector, "_new_rapidocr", lambda: rapid)

    assert selector.get_ocr_engine() is rapid


def test_invalid_backend_fails_with_actionable_message(monkeypatch) -> None:
    monkeypatch.setenv("DOCMIRROR_OCR_BACKEND", "unknown")

    with pytest.raises(ValueError, match="DOCMIRROR_OCR_BACKEND"):
        selector.get_ocr_engine()


def test_rapid_module_factory_is_a_compatibility_shim(monkeypatch) -> None:
    selected = SimpleNamespace(engine_id="selected")
    monkeypatch.setattr(selector, "get_ocr_engine", lambda: selected)

    from docmirror.ocr.vision.rapidocr_engine import get_ocr_engine

    assert get_ocr_engine() is selected


def test_auto_requires_gpu_but_explicit_paddle_allows_cpu(monkeypatch) -> None:
    calls: list[tuple[str | None, bool]] = []

    def _status(device=None, *, require_gpu=False):
        calls.append((device, require_gpu))
        return (not require_gpu, "available" if not require_gpu else "auto_requires_gpu")

    rapid = _Backend("rapidocr_onnxruntime")
    monkeypatch.setenv("DOCMIRROR_PADDLE_DEVICE", "cpu")
    monkeypatch.setenv("DOCMIRROR_OCR_BACKEND", "auto")
    monkeypatch.setattr(selector, "paddle_runtime_status", _status)
    monkeypatch.setattr(selector, "_new_rapidocr", lambda: rapid)
    assert selector.get_ocr_engine() is rapid
    assert calls[-1] == ("cpu", True)

    selector.reset_ocr_engine()
    paddle = _Backend("paddleocr")
    monkeypatch.setenv("DOCMIRROR_OCR_BACKEND", "paddleocr")
    monkeypatch.setattr(selector, "_new_paddleocr", lambda **_kwargs: paddle)
    assert selector.get_ocr_engine() is paddle
    assert calls[-1] == ("cpu", False)


def test_concurrent_primary_failures_all_retry_the_same_fallback() -> None:
    barrier = threading.Barrier(2)

    class _ConcurrentFailure(_Backend):
        def detect_image_words(self, _img, multi_scale: bool = False):
            self.calls += 1
            barrier.wait(timeout=5)
            raise RuntimeError("concurrent primary failure")

    primary = _ConcurrentFailure("paddleocr")
    fallback = _Backend("rapidocr_onnxruntime")
    engine = selector.FailoverOCREngine(primary, lambda: fallback)
    outputs: list[list[tuple]] = []
    errors: list[Exception] = []

    def _call() -> None:
        try:
            outputs.append(engine.detect_image_words(object()))
        except Exception as exc:  # pragma: no cover - asserted empty below
            errors.append(exc)

    threads = [threading.Thread(target=_call) for _index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not errors
    assert len(outputs) == 2
    assert all(words[0][4] == "rapidocr_onnxruntime" for words in outputs)
    assert fallback.calls == 2
    assert engine.backend_revision == 1
