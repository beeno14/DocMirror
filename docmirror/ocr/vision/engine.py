# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hardware-aware local OCR selection with a RapidOCR safety fallback."""

from __future__ import annotations

import logging
import os
import threading
from importlib.util import find_spec
from typing import Any, Callable

from docmirror.ocr.vision.backend import OCRBackend, OCRRegionResult, OCRWord, backend_is_available

logger = logging.getLogger(__name__)

_VALID_BACKENDS = {"auto", "rapidocr", "paddleocr"}
_ALIASES = {"rapid": "rapidocr", "paddle": "paddleocr"}


class OCRBackendUnavailable(RuntimeError):
    """Raised when a requested OCR backend cannot be initialized."""


def _normalise_backend(value: str | None) -> str:
    backend = _ALIASES.get(str(value or "").strip().lower(), str(value or "").strip().lower())
    if not backend:
        backend = "auto"
    if backend not in _VALID_BACKENDS:
        choices = ", ".join(sorted(_VALID_BACKENDS))
        raise ValueError(f"Invalid DOCMIRROR_OCR_BACKEND={value!r}; expected one of: {choices}")
    return backend


def _configured_backend() -> str:
    explicit = os.getenv("DOCMIRROR_OCR_BACKEND", "").strip()
    if explicit:
        return _normalise_backend(explicit)
    try:
        from docmirror.configs.runtime.settings import get_settings

        return _normalise_backend(getattr(get_settings(), "ocr_backend", "auto"))
    except Exception:
        return "auto"


def _configured_paddle_device() -> str:
    explicit = os.getenv("DOCMIRROR_PADDLE_DEVICE", "").strip()
    if explicit:
        return explicit
    try:
        from docmirror.configs.runtime.settings import get_settings

        return str(getattr(get_settings(), "paddle_ocr_device", "gpu:0") or "gpu:0")
    except Exception:
        return "gpu:0"


def _configured_paddle_profile() -> str:
    explicit = os.getenv("DOCMIRROR_PADDLE_PROFILE", "").strip()
    if explicit:
        return explicit
    try:
        from docmirror.configs.runtime.settings import get_settings

        return str(getattr(get_settings(), "paddle_ocr_profile", "server") or "server")
    except Exception:
        return "server"


def paddle_runtime_status(
    device: str | None = None,
    *,
    require_gpu: bool = False,
) -> tuple[bool, str]:
    """Probe package and device availability without constructing OCR models."""

    device = str(device or _configured_paddle_device()).strip().lower()
    if require_gpu and not device.startswith("gpu"):
        return False, "auto_requires_gpu"
    if find_spec("paddleocr") is None:
        return False, "paddleocr_not_installed"
    if find_spec("paddle") is None:
        return False, "paddle_runtime_not_installed"
    try:
        import paddle
    except Exception as exc:
        return False, f"paddle_import_failed:{type(exc).__name__}"

    if not device.startswith("gpu"):
        return True, "available"
    try:
        if not bool(paddle.is_compiled_with_cuda()):
            return False, "paddle_not_compiled_with_cuda"
        count = int(paddle.device.cuda.device_count())
        if count < 1:
            return False, "no_cuda_device"
        if ":" in device:
            index_text = device.split(":", 1)[1].split(",", 1)[0]
            index = int(index_text)
            if index < 0 or index >= count:
                return False, f"cuda_device_out_of_range:{index}/{count}"
    except Exception as exc:
        return False, f"cuda_probe_failed:{type(exc).__name__}"
    return True, "available"


def _new_rapidocr() -> OCRBackend:
    from docmirror.ocr.vision.rapidocr_engine import RapidOCREngine

    engine = RapidOCREngine()
    if not backend_is_available(engine):
        raise OCRBackendUnavailable("RapidOCR is unavailable; install with: pip install 'docmirror[ocr]'")
    return engine


def _new_paddleocr(*, device: str, profile: str) -> OCRBackend:
    from docmirror.ocr.vision.paddleocr_engine import PaddleOCREngine

    engine = PaddleOCREngine(device=device, profile=profile)
    if not backend_is_available(engine):
        raise OCRBackendUnavailable("PaddleOCR failed to initialize")
    return engine


class FailoverOCREngine:
    """Use a primary backend until it fails, then demote to a lazy fallback."""

    def __init__(self, primary: OCRBackend, fallback_factory: Callable[[], OCRBackend]) -> None:
        self._primary = primary
        self._active: OCRBackend = primary
        self._fallback_factory = fallback_factory
        self._using_fallback = False
        self._fallback_reason: str | None = None
        self._backend_revision = 0
        self._lock = threading.RLock()

    @property
    def engine_id(self) -> str:
        return self._active.engine_id

    @property
    def source_id(self) -> str:
        return self._active.source_id

    @property
    def is_available(self) -> bool:
        return backend_is_available(self._active)

    @property
    def fallback_reason(self) -> str | None:
        return self._fallback_reason

    @property
    def backend_revision(self) -> int:
        """Incremented when the active backend changes.

        Multi-pass callers use this value to discard any primary-backend
        candidates and restart the current page after a demotion.
        """

        return self._backend_revision

    def _demote(self, exc: Exception, *, failed_backend: OCRBackend) -> OCRBackend:
        with self._lock:
            if not self._using_fallback:
                fallback = self._fallback_factory()
                self._active = fallback
                self._using_fallback = True
                self._fallback_reason = f"{type(exc).__name__}: {exc}"
                self._backend_revision += 1
                logger.warning(
                    "PaddleOCR inference failed; using RapidOCR for the rest of this process: %s",
                    self._fallback_reason,
                )
                close = getattr(failed_backend, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception as close_exc:
                        logger.debug("PaddleOCR cleanup after demotion failed: %s", close_exc)
            return self._active

    def detect_image_words(self, img: Any, multi_scale: bool = False) -> list[OCRWord]:
        backend = self._active
        try:
            return backend.detect_image_words(img, multi_scale=multi_scale)
        except Exception as exc:
            # A concurrent request may have already demoted the shared engine.
            # Calls that failed on the former primary must still retry against
            # the now-active fallback; failures from the fallback itself remain
            # visible to the caller.
            if backend is not self._primary:
                raise
            return self._demote(exc, failed_backend=backend).detect_image_words(img, multi_scale=multi_scale)

    def force_recognize_regions(
        self,
        img: Any,
        regions: list[tuple[int, int, int, int]],
    ) -> list[OCRRegionResult]:
        backend = self._active
        try:
            return backend.force_recognize_regions(img, regions)
        except Exception as exc:
            if backend is not self._primary:
                raise
            return self._demote(exc, failed_backend=backend).force_recognize_regions(img, regions)


_engine: OCRBackend | None = None
_engine_lock = threading.RLock()
_selection_status: dict[str, Any] = {}


def _build_engine() -> OCRBackend:
    global _selection_status

    requested = _configured_backend()
    device = _configured_paddle_device()
    profile = _configured_paddle_profile()
    _selection_status = {
        "requested": requested,
        "active": None,
        "paddle_device": device,
        "paddle_profile": profile,
        "fallback_reason": None,
    }

    if requested == "rapidocr":
        selected = _new_rapidocr()
        _selection_status["active"] = selected.engine_id
        return selected

    available, reason = paddle_runtime_status(device, require_gpu=requested == "auto")
    if requested == "paddleocr" and not available:
        raise OCRBackendUnavailable(f"PaddleOCR requested but unavailable: {reason}")

    if available:
        try:
            paddle_engine = _new_paddleocr(device=device, profile=profile)
        except Exception as exc:
            if requested == "paddleocr":
                raise OCRBackendUnavailable(f"PaddleOCR requested but initialization failed: {exc}") from exc
            reason = f"paddle_initialization_failed:{type(exc).__name__}"
            logger.warning("PaddleOCR initialization failed; falling back to RapidOCR: %s", exc)
        else:
            _selection_status["active"] = paddle_engine.engine_id
            logger.info("Selected PaddleOCR backend (%s, profile=%s)", device, profile)
            if requested == "auto":
                return FailoverOCREngine(paddle_engine, _new_rapidocr)
            return paddle_engine

    selected = _new_rapidocr()
    _selection_status["active"] = selected.engine_id
    _selection_status["fallback_reason"] = reason
    logger.info("Selected RapidOCR backend (PaddleOCR unavailable: %s)", reason)
    return selected


def get_ocr_engine() -> OCRBackend:
    """Return the process-local OCR backend selected for this environment."""

    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = _build_engine()
    return _engine


def get_ocr_backend_status(*, initialize: bool = False) -> dict[str, Any]:
    """Return selection diagnostics suitable for logs and the doctor command."""

    if initialize:
        engine = get_ocr_engine()
        status = dict(_selection_status)
        status["active"] = engine.engine_id
        status["fallback_reason"] = getattr(engine, "fallback_reason", status.get("fallback_reason"))
        return status
    requested = _configured_backend()
    device = _configured_paddle_device()
    available, reason = paddle_runtime_status(device, require_gpu=requested == "auto")
    live_fallback_reason = getattr(_engine, "fallback_reason", None)
    return {
        "requested": requested,
        "active": getattr(_engine, "engine_id", None),
        "paddle_device": device,
        "paddle_profile": _configured_paddle_profile(),
        "paddle_available": available,
        "paddle_status": reason,
        "fallback_reason": live_fallback_reason or _selection_status.get("fallback_reason"),
    }


def reset_ocr_engine() -> None:
    """Clear the selector cache. Intended for tests and controlled reconfiguration."""

    global _engine, _selection_status
    with _engine_lock:
        _engine = None
        _selection_status = {}


__all__ = [
    "FailoverOCREngine",
    "OCRBackendUnavailable",
    "get_ocr_backend_status",
    "get_ocr_engine",
    "paddle_runtime_status",
    "reset_ocr_engine",
]
