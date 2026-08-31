# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recognizer adapters for local OCR repair."""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


def local_ocr_recognize(image: Any, *, source: str | None = None) -> list[dict[str, Any]]:
    """Run the selected local OCR backend on an image crop."""
    explicit_source = source
    try:
        from docmirror.ocr.vision.engine import get_ocr_engine

        engine = get_ocr_engine()
        words = engine.detect_image_words(image, multi_scale=True)
        source = explicit_source or engine.source_id
    except Exception as exc:
        return [
            {
                "source": source or "local_ocr",
                "status": "not_evaluated",
                "reason": str(exc),
                "text": "",
                "confidence": 0.0,
            }
        ]

    out: list[dict[str, Any]] = []
    for word in words:
        text = str(word[4] if len(word) > 4 else "").strip()
        if not text:
            continue
        confidence = float(word[8] if len(word) > 8 else 0.0)
        out.append({"source": source, "text": text, "confidence": confidence, "bbox": list(word[:4])})
    return out


def rapidocr_recognize(image: Any, *, source: str | None = None) -> list[dict[str, Any]]:
    """Compatibility alias for the historical recognizer name."""

    return local_ocr_recognize(image, source=source)


def tesseract_recognize(
    image: Any, *, source: str = "tesseract", lang: str = "eng", psm: int = 7
) -> list[dict[str, Any]]:
    """Run Tesseract on an image crop when available."""
    try:
        import cv2

        from docmirror.ocr.backends.tesseract import TesseractBackend
    except Exception as exc:
        return [{"source": source, "status": "not_evaluated", "reason": str(exc), "text": "", "confidence": 0.0}]

    try:
        backend = TesseractBackend()
        if not backend.is_available:
            return [
                {
                    "source": source,
                    "status": "not_evaluated",
                    "reason": "backend_unavailable",
                    "text": "",
                    "confidence": 0.0,
                }
            ]
        with NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            cv2.imwrite(str(tmp_path), image)
            result = backend.ocr(tmp_path.read_bytes(), lang=lang, psm=psm, timeout=10)
        finally:
            tmp_path.unlink(missing_ok=True)
        return [{"source": source, "text": result.text, "confidence": result.confidence}]
    except Exception as exc:
        return [{"source": source, "status": "not_evaluated", "reason": str(exc), "text": "", "confidence": 0.0}]


__all__ = ["local_ocr_recognize", "rapidocr_recognize", "tesseract_recognize"]
