# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PaddleOCR adapter that emits DocMirror's backend-neutral OCR contracts."""

from __future__ import annotations

import gc
import logging
import os
import threading
from importlib.util import find_spec
from typing import Any, Mapping

from docmirror.ocr.vision.backend import OCRRegionResult, OCRWord
from docmirror.runtime.optional_deps import require_optional_module

logger = logging.getLogger(__name__)

HAS_PADDLEOCR = find_spec("paddleocr") is not None and find_spec("paddle") is not None

_PROFILE_MODELS: dict[str, tuple[str, str]] = {
    "small": ("PP-OCRv6_small_det", "PP-OCRv6_small_rec"),
    # MinerU-like balance: light detector with the stronger recognizer.
    "server": ("PP-OCRv6_small_det", "PP-OCRv6_medium_rec"),
    "medium": ("PP-OCRv6_medium_det", "PP-OCRv6_medium_rec"),
}


def _result_payload(result: Any) -> dict[str, Any]:
    """Convert PaddleX/PaddleOCR Result objects to their inner result mapping."""

    payload = getattr(result, "json", result)
    if callable(payload):
        payload = payload()
    if isinstance(payload, Mapping):
        payload = dict(payload)
    else:
        try:
            payload = dict(payload)
        except Exception:
            return {}
    inner = payload.get("res")
    return dict(inner) if isinstance(inner, Mapping) else payload


def _rect_from_box(box: Any) -> tuple[float, float, float, float] | None:
    try:
        values = list(box)
    except Exception:
        return None
    if len(values) == 4 and not hasattr(values[0], "__len__"):
        x0, y0, x1, y1 = values
        return float(x0), float(y0), float(x1), float(y1)
    if not values:
        return None
    try:
        xs = [float(point[0]) for point in values]
        ys = [float(point[1]) for point in values]
    except Exception:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def _split_ocr_block(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    text: str,
) -> list[tuple[float, float, float, float, str]]:
    """Reuse RapidOCR's compatibility split rules without importing its package."""

    # Importing this module is safe: rapidocr_engine imports its package lazily.
    from docmirror.ocr.vision.rapidocr_engine import _split_ocr_block as split

    return split(x0, y0, x1, y1, text)


def _intersection_metrics(a: OCRWord, b: OCRWord) -> tuple[float, float]:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    iou = intersection / union if union > 0 else 0.0
    containment = intersection / min(area_a, area_b) if min(area_a, area_b) > 0 else 0.0
    return iou, containment


class PaddleOCREngine:
    """Official PaddleOCR pipeline wrapped in DocMirror's canonical tuple API."""

    engine_id = "paddleocr"
    source_id = "paddleocr"

    def __init__(self, *, device: str = "gpu:0", profile: str = "server") -> None:
        if not HAS_PADDLEOCR:
            raise RuntimeError("PaddleOCR is not installed; install the DocMirror ocr-paddle profile")
        profile = str(profile or "server").strip().lower()
        if profile not in _PROFILE_MODELS:
            choices = ", ".join(sorted(_PROFILE_MODELS))
            raise ValueError(f"Unknown PaddleOCR profile {profile!r}; expected one of: {choices}")

        self.device = str(device or "gpu:0")
        self.profile = profile
        self._det_model, self._rec_model = _PROFILE_MODELS[profile]
        self._lock = threading.RLock()
        self._recognizer: Any | None = None

        paddleocr = require_optional_module(
            "paddleocr",
            feature="PaddleOCR GPU OCR",
            extra="ocr-paddle",
        )
        engine = os.getenv("DOCMIRROR_PADDLE_INFERENCE_ENGINE", "paddle_static").strip() or "paddle_static"
        batch_size = max(1, int(os.getenv("DOCMIRROR_PADDLE_REC_BATCH_SIZE", "16")))
        logger.info(
            "Initializing PaddleOCR (device=%s, profile=%s, det=%s, rec=%s)",
            self.device,
            self.profile,
            self._det_model,
            self._rec_model,
        )
        self._pipeline = paddleocr.PaddleOCR(
            text_detection_model_name=self._det_model,
            text_recognition_model_name=self._rec_model,
            text_recognition_batch_size=batch_size,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device=self.device,
            engine=engine,
        )

    @property
    def is_available(self) -> bool:
        return self._pipeline is not None

    def _predict_once(self, img: Any, *, scale: float = 1.0) -> list[OCRWord]:
        results = list(self._pipeline.predict(input=img))
        words: list[OCRWord] = []
        word_index = 0
        for result_index, result in enumerate(results):
            payload = _result_payload(result)
            texts_raw = payload.get("rec_texts")
            texts = list(texts_raw) if texts_raw is not None else []
            scores_raw = payload.get("rec_scores")
            boxes_raw = payload.get("rec_boxes")
            if boxes_raw is None:
                boxes_raw = payload.get("rec_polys")
            scores = list(scores_raw) if scores_raw is not None else []
            boxes = list(boxes_raw) if boxes_raw is not None else []
            for line_index, (box, text) in enumerate(zip(boxes, texts)):
                text = str(text or "").strip()
                rect = _rect_from_box(box)
                if not text or rect is None:
                    continue
                x0, y0, x1, y1 = (coordinate / scale for coordinate in rect)
                confidence = float(scores[line_index]) if line_index < len(scores) else 0.0
                for sx0, sy0, sx1, sy1, sub_text in _split_ocr_block(x0, y0, x1, y1, text):
                    words.append(
                        (
                            float(sx0),
                            float(sy0),
                            float(sx1),
                            float(sy1),
                            str(sub_text),
                            result_index,
                            line_index,
                            word_index,
                            confidence,
                        )
                    )
                    word_index += 1
        return words

    @staticmethod
    def _merge_multiscale(words: list[OCRWord]) -> list[OCRWord]:
        # RapidOCR's multi-scale path favours larger overlapping detections. Match
        # that behavior while using only PaddleOCR's public full-pipeline API.
        ranked = sorted(
            words,
            key=lambda word: (
                max(0.0, word[2] - word[0]) * max(0.0, word[3] - word[1]),
                word[8],
            ),
            reverse=True,
        )
        kept: list[OCRWord] = []
        for word in ranked:
            if any(
                (iou >= 0.30 or containment >= 0.75)
                for other in kept
                for iou, containment in [_intersection_metrics(word, other)]
            ):
                continue
            kept.append(word)
        kept.sort(key=lambda word: (word[1], word[0]))
        return [(*word[:7], index, word[8]) for index, word in enumerate(kept)]

    def detect_image_words(self, img: Any, multi_scale: bool = False) -> list[OCRWord]:
        if img is None or not hasattr(img, "shape"):
            raise TypeError("PaddleOCR expects a numpy image array")
        with self._lock:
            words = self._predict_once(img, scale=1.0)
            if not multi_scale:
                return words
            cv2 = require_optional_module(
                "cv2",
                feature="PaddleOCR multi-scale detection",
                extra="ocr-paddle",
            )
            scaled = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LANCZOS4)
            words.extend(self._predict_once(scaled, scale=2.0))
            return self._merge_multiscale(words)

    def close(self) -> None:
        """Release model references and return cached GPU memory when possible."""

        with self._lock:
            resources = (self._recognizer, getattr(self, "_pipeline", None))
            self._recognizer = None
            self._pipeline = None
            for resource in resources:
                close = getattr(resource, "close", None)
                if callable(close):
                    close()
        gc.collect()
        try:
            paddle = require_optional_module(
                "paddle",
                feature="PaddleOCR GPU cleanup",
                extra="ocr-paddle",
            )
            empty_cache = getattr(getattr(paddle.device, "cuda", None), "empty_cache", None)
            if callable(empty_cache):
                empty_cache()
        except Exception as exc:
            logger.debug("PaddleOCR GPU cache cleanup skipped: %s", exc)

    def _get_recognizer(self) -> Any:
        if self._recognizer is None:
            paddleocr = require_optional_module(
                "paddleocr",
                feature="PaddleOCR forced text recognition",
                extra="ocr-paddle",
            )
            engine = os.getenv("DOCMIRROR_PADDLE_INFERENCE_ENGINE", "paddle_static").strip() or "paddle_static"
            self._recognizer = paddleocr.TextRecognition(
                model_name=self._rec_model,
                device=self.device,
                engine=engine,
            )
        return self._recognizer

    def force_recognize_regions(
        self,
        img: Any,
        regions: list[tuple[int, int, int, int]],
    ) -> list[OCRRegionResult]:
        crops: list[Any] = []
        valid_regions: list[tuple[int, int, int, int]] = []
        for x0, y0, x1, y1 in regions:
            crop = img[int(y0) : int(y1), int(x0) : int(x1)]
            if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
                continue
            crops.append(crop)
            valid_regions.append((x0, y0, x1, y1))
        if not crops:
            return []

        threshold = float(os.getenv("DOCMIRROR_PADDLE_FORCE_REC_MIN_CONFIDENCE", "0.5"))
        batch_size = max(1, int(os.getenv("DOCMIRROR_PADDLE_REC_BATCH_SIZE", "16")))
        with self._lock:
            predictions = list(
                self._get_recognizer().predict(
                    input=crops,
                    batch_size=min(batch_size, len(crops)),
                )
            )
        recognized: list[OCRRegionResult] = []
        for region, prediction in zip(valid_regions, predictions):
            payload = _result_payload(prediction)
            text = str(payload.get("rec_text") or "").strip()
            confidence = float(payload.get("rec_score") or 0.0)
            if text and confidence >= threshold:
                x0, y0, x1, y1 = region
                recognized.append((float(x0), float(y0), float(x1), float(y1), text, confidence))
        return recognized


__all__ = ["HAS_PADDLEOCR", "PaddleOCREngine"]
