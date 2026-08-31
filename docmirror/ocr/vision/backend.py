# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend-neutral contracts shared by DocMirror's local OCR engines."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

# PyMuPDF-compatible word tuple plus confidence.  This is the stable shape used
# throughout DocMirror, regardless of the native result shape of an OCR package.
OCRWord = tuple[float, float, float, float, str, int, int, int, float]

# Forced recognition deliberately preserves the caller-supplied region.
OCRRegionResult = tuple[float, float, float, float, str, float]


@runtime_checkable
class OCRBackend(Protocol):
    """Public interface implemented by every local OCR backend."""

    @property
    def engine_id(self) -> str:
        """Stable implementation identifier for parser metadata."""

    @property
    def source_id(self) -> str:
        """Short identifier used in evidence provenance."""

    @property
    def is_available(self) -> bool:
        """Whether the backend completed initialization and can infer."""

    def detect_image_words(self, img: Any, multi_scale: bool = False) -> list[OCRWord]: ...

    def force_recognize_regions(
        self,
        img: Any,
        regions: list[tuple[int, int, int, int]],
    ) -> list[OCRRegionResult]: ...


def backend_source(backend: OCRBackend, scope: str | None = None) -> str:
    """Build a provenance value while retaining RapidOCR's legacy spelling."""

    source = str(getattr(backend, "source_id", "rapidocr") or "rapidocr")
    return f"{source}_{scope}" if scope else source


def backend_is_available(backend: Any) -> bool:
    """Read the public availability contract without leaking private engines."""

    try:
        value = getattr(backend, "is_available")
        return bool(value() if callable(value) else value)
    except Exception:
        return False


def backend_revision(backend: Any) -> int:
    """Return the active-backend generation used by multi-pass workflows."""

    try:
        return int(getattr(backend, "backend_revision", 0) or 0)
    except (TypeError, ValueError):
        return 0


__all__ = [
    "OCRBackend",
    "OCRRegionResult",
    "OCRWord",
    "backend_is_available",
    "backend_revision",
    "backend_source",
]
