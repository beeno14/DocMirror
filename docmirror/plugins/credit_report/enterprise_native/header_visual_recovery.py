# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded visual recovery for image-backed enterprise cover fields.

Digital PBOC enterprise reports are decoded from native text and tables.  A
small number of cover values can nevertheless be stored as embedded raster
strips.  This module deliberately inspects only likely page-one header strips;
it is not a second full-document OCR pipeline.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from docmirror.plugins.credit_report.enterprise_native.quality import EnterpriseQualityFlag

logger = logging.getLogger(__name__)

_EXISTING_REPORT_NUMBER_RE = re.compile(
    r"(?:NO\.?|\u62a5\u544a\u7f16\u53f7[:\uff1a]?)[0-9]{16,32}",
    re.IGNORECASE,
)
_PREFIXED_REPORT_NUMBER_RE = re.compile(
    r"(?:N[O0Q][.\u3002:\uff1a-]?|\u62a5\u544a\u7f16\u53f7[:\uff1a]?)(?P<value>[0-9]{16,32})",
    re.IGNORECASE,
)
_BARE_REPORT_NUMBER_RE = re.compile(r"(?<![0-9])(?P<value>[0-9]{20,28})(?![0-9])")


@dataclass(frozen=True)
class RecoveredEnterpriseHeaderField:
    """One uniquely recovered header field ready for canonical IR insertion."""

    field_name: str
    value: str
    source_page: int
    bbox: tuple[float, float, float, float]
    confidence: float
    source_text: str


@dataclass(frozen=True)
class EnterpriseHeaderVisualRecovery:
    """Recovered values and warning-only diagnostics from the bounded pass."""

    fields: tuple[RecoveredEnterpriseHeaderField, ...] = ()
    quality_flags: tuple[EnterpriseQualityFlag, ...] = ()


def _value(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _compact(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or ""))


def _source_pdf_path(parse_result: Any) -> Path | None:
    provenance = _value(parse_result, "provenance")
    evidence_plane = _value(parse_result, "evidence_plane")
    evidence_source = _value(evidence_plane, "source")
    candidates = (
        _value(parse_result, "file_path", ""),
        _value(provenance, "file_path", ""),
        _value(evidence_source, "filename", ""),
    )
    for raw in candidates:
        if not raw:
            continue
        try:
            path = Path(str(raw)).expanduser().resolve()
        except (OSError, RuntimeError, ValueError):
            continue
        if path.is_file() and path.suffix.lower() == ".pdf":
            return path
    return None


def _likely_header_box(
    bbox: tuple[float, float, float, float],
    *,
    page_width: float,
    page_height: float,
) -> bool:
    x0, y0, x1, y1 = bbox
    box_width = x1 - x0
    box_height = y1 - y0
    return bool(
        page_width > 0.0
        and page_height > 0.0
        and x0 >= page_width * 0.42
        and y0 >= 0.0
        and y1 <= page_height * 0.40
        and 60.0 <= box_width <= page_width * 0.55
        and 7.0 <= box_height <= 65.0
        and box_width / max(box_height, 1.0) >= 2.5
    )


def _has_likely_header_image_evidence(parse_result: Any) -> bool:
    plane = _value(parse_result, "evidence_plane")
    pages = list(_value(plane, "pages") or ())
    if not pages:
        return False
    first_page = min(
        pages,
        key=lambda page: int(_value(page, "page_index", 0) or 0),
    )
    page_id = str(_value(first_page, "page_id", "") or "")
    try:
        width = float(_value(first_page, "width", 0.0) or 0.0)
        height = float(_value(first_page, "height", 0.0) or 0.0)
    except (TypeError, ValueError):
        return False
    evidence = _value(plane, "evidence")
    for atom in _value(evidence, "image_atoms") or ():
        if page_id and str(_value(atom, "page_id", "") or "") != page_id:
            continue
        raw_bbox = _value(atom, "bbox") or _value(atom, "source_bbox")
        if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) < 4:
            continue
        try:
            bbox = tuple(float(value) for value in raw_bbox[:4])
        except (TypeError, ValueError):
            continue
        if _likely_header_box(bbox, page_width=width, page_height=height):
            return True
    return False


def _candidate_header_image_boxes(page: Any) -> tuple[tuple[float, float, float, float], ...]:
    page_rect = page.rect
    width = float(page_rect.width or 0.0)
    height = float(page_rect.height or 0.0)
    if width <= 0.0 or height <= 0.0:
        return ()
    try:
        blocks = (page.get_text("dict") or {}).get("blocks") or ()
    except Exception:
        return ()
    boxes: list[tuple[float, float, float, float]] = []
    for block in blocks:
        if not isinstance(block, dict) or int(block.get("type", -1)) != 1:
            continue
        raw_bbox = block.get("bbox")
        if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) < 4:
            continue
        try:
            x0, y0, x1, y1 = (float(value) for value in raw_bbox[:4])
        except (TypeError, ValueError):
            continue
        if _likely_header_box(
            (x0, y0, x1, y1),
            page_width=width,
            page_height=height,
        ):
            boxes.append((x0, y0, x1, y1))
    return tuple(dict.fromkeys(boxes))


def _report_number_from_ocr_text(value: str, *, allow_bare: bool) -> str:
    compact = _compact(value).upper()
    match = _PREFIXED_REPORT_NUMBER_RE.search(compact)
    if match:
        return match.group("value")
    if allow_bare:
        bare = _BARE_REPORT_NUMBER_RE.search(compact)
        if bare:
            return bare.group("value")
    return ""


def _ocr_header_strip(page: Any, bbox: tuple[float, float, float, float]) -> tuple[tuple[str, float], ...]:
    import fitz

    from docmirror.ocr.vision.rapidocr_engine import get_ocr_engine
    from docmirror.runtime.optional_deps import require_optional_module

    np = require_optional_module(
        "numpy",
        feature="enterprise report header visual recovery",
        extra="ocr",
    )
    page_rect = page.rect
    x0, y0, x1, y1 = bbox
    clip = fitz.Rect(
        max(float(page_rect.x0), x0 - 4.0),
        max(float(page_rect.y0), y0 - 3.0),
        min(float(page_rect.x1), x1 + 4.0),
        min(float(page_rect.y1), y1 + 3.0),
    )
    zoom = 4.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip, alpha=False)
    image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if image.ndim == 3 and image.shape[2] >= 3:
        image = image[:, :, :3][:, :, ::-1].copy()  # PyMuPDF RGB -> OpenCV BGR
    if pix.n >= 3:
        image = image[:, :, :3]
    engine = get_ocr_engine()
    results: list[tuple[str, float]] = []

    forced = engine.force_recognize_regions(
        image,
        [(0, 0, int(image.shape[1]), int(image.shape[0]))],
    )
    for item in forced:
        if len(item) >= 6 and str(item[4] or "").strip():
            results.append((str(item[4]).strip(), float(item[5] or 0.0)))

    words = engine.detect_image_words(image, multi_scale=True)
    ordered = sorted(
        (word for word in words if len(word) >= 5 and str(word[4] or "").strip()),
        key=lambda word: (float(word[1]), float(word[0])),
    )
    if ordered:
        text = "".join(str(word[4]).strip() for word in ordered)
        confidences = [float(word[8]) for word in ordered if len(word) >= 9]
        results.append(
            (
                text,
                sum(confidences) / len(confidences) if confidences else 0.8,
            )
        )
    return tuple(results)


def recover_enterprise_header_visual_fields(
    parse_result: Any,
    *,
    existing_text: str,
) -> EnterpriseHeaderVisualRecovery:
    """Recover a missing report number from unique page-one raster evidence."""

    if _EXISTING_REPORT_NUMBER_RE.search(_compact(existing_text)):
        return EnterpriseHeaderVisualRecovery()
    source_path = _source_pdf_path(parse_result)
    if source_path is None:
        if _has_likely_header_image_evidence(parse_result):
            return EnterpriseHeaderVisualRecovery(
                quality_flags=(
                    EnterpriseQualityFlag(
                        code="ENTERPRISE_HEADER_VISUAL_RECOVERY_UNAVAILABLE",
                        severity="warning",
                        category="parseresult_input",
                        status="source_file_unavailable",
                        message=(
                            "Likely page-one image-backed report metadata is present, but the "
                            "local source PDF is unavailable for bounded recovery."
                        ),
                        source_pages=(1,),
                        dataset="enterprise_report_metadata",
                        field_name="report_number",
                    ),
                )
            )
        return EnterpriseHeaderVisualRecovery()

    try:
        import fitz

        with fitz.open(source_path) as document:
            if not document:
                return EnterpriseHeaderVisualRecovery()
            page = document[0]
            boxes = _candidate_header_image_boxes(page)
            if not boxes:
                return EnterpriseHeaderVisualRecovery()
            candidates: list[tuple[str, float, str, tuple[float, float, float, float]]] = []
            errors: list[str] = []
            for bbox in boxes:
                try:
                    for text, confidence in _ocr_header_strip(page, bbox):
                        number = _report_number_from_ocr_text(text, allow_bare=True)
                        if number:
                            candidates.append((number, confidence, text, bbox))
                except Exception as exc:  # optional OCR must not suppress native extraction
                    errors.append(type(exc).__name__)
                    logger.debug("enterprise header strip OCR failed", exc_info=True)
    except Exception as exc:
        logger.debug("enterprise header visual recovery failed", exc_info=True)
        return EnterpriseHeaderVisualRecovery(
            quality_flags=(
                EnterpriseQualityFlag(
                    code="ENTERPRISE_HEADER_VISUAL_RECOVERY_UNAVAILABLE",
                    severity="warning",
                    category="parseresult_input",
                    status="targeted_recovery_unavailable",
                    message=(
                        "A page-one image-backed report field could not be inspected; "
                        "native enterprise values were retained unchanged."
                    ),
                    source_pages=(1,),
                    dataset="enterprise_report_metadata",
                    field_name="report_number",
                    details={"error_type": type(exc).__name__},
                ),
            )
        )

    values = {number for number, _confidence, _text, _bbox in candidates}
    if len(values) == 1:
        value = next(iter(values))
        matching = [candidate for candidate in candidates if candidate[0] == value]
        number, confidence, source_text, bbox = max(matching, key=lambda item: item[1])
        return EnterpriseHeaderVisualRecovery(
            fields=(
                RecoveredEnterpriseHeaderField(
                    field_name="report_number",
                    value=number,
                    source_page=1,
                    bbox=bbox,
                    confidence=round(max(0.0, min(1.0, confidence)), 4),
                    source_text=source_text,
                ),
            )
        )

    status = "ambiguous" if values else "not_recognized"
    message = (
        "Page-one image evidence produced multiple report-number candidates; "
        "no report number was selected."
        if values
        else "A likely page-one report-number image was present but could not be recognized uniquely."
    )
    return EnterpriseHeaderVisualRecovery(
        quality_flags=(
            EnterpriseQualityFlag(
                code="ENTERPRISE_HEADER_VISUAL_FIELD_UNRESOLVED",
                severity="warning",
                category="canonical_field_conservation",
                status=status,
                message=message,
                source_pages=(1,),
                dataset="enterprise_report_metadata",
                field_name="report_number",
                details={
                    "candidate_value_count": len(values),
                    "candidate_image_count": len(boxes),
                    "ocr_error_types": sorted(set(errors)),
                },
            ),
        )
    )


__all__ = [
    "EnterpriseHeaderVisualRecovery",
    "RecoveredEnterpriseHeaderField",
    "recover_enterprise_header_visual_fields",
]
