# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Credit-report repayment micro-grid reconstruction."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from typing import Any, cast

from docmirror.ocr.micro_grid.cell_recognition import (
    extract_micro_cell_glyph_template,
    normalize_allowlist_text,
    recognize_micro_cell_from_image,
)
from docmirror.ocr.micro_grid.detect import detect_micro_grid_candidates
from docmirror.ocr.micro_grid.models import BBox, MicroGrid, MicroGridCell, OCRToken
from docmirror.ocr.micro_grid.reconstruct import (
    assign_tokens_to_col_bands,
    build_cell,
    cell_bbox,
    dedupe_visual_tokens,
    equal_col_bands,
    expand_tokens_to_char_tokens,
)

_RANGE_RE = re.compile(r"(20\d{2})年\s*(\d{1,2})月\s*[-—一至~～]\s*(20\d{2})年\s*(\d{1,2})月.*还款记录")
_YEAR_RE = re.compile(r"^20\d{2}(?=\s|$)")
_STATUS_CHARS = {"*", "/", "N", "C", "1", "2", "3", "4", "5", "6", "7", "B", "M", "D", "Z", "G"}
_ZERO_OVERDUE_STATUSES = {"*", "/", "N", "C"}


@dataclass(frozen=True)
class RepaymentExtraction:
    micro_grid: MicroGrid | None
    records: list[dict[str, Any]]
    audit: dict[str, Any]


def _bbox(obj: Any) -> BBox | None:
    raw = obj.get("bbox") if isinstance(obj, dict) else getattr(obj, "bbox", None)
    if raw and len(raw) == 4:
        return (float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))
    return None


def _text(obj: Any) -> str:
    if isinstance(obj, dict):
        return str(obj.get("content") or obj.get("text") or "").strip()
    return str(getattr(obj, "text", "") or "").strip()


def _confidence(obj: Any) -> float:
    val = obj.get("confidence") if isinstance(obj, dict) else getattr(obj, "confidence", 1.0)
    try:
        return float(val if val is not None else 1.0)
    except (TypeError, ValueError):
        return 1.0


def _line_items(lines: Iterable[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, line in enumerate(lines):
        b = _bbox(line)
        t = _text(line)
        if not b or not t:
            continue
        source_logical_page = (
            line.get("source_logical_page") if isinstance(line, dict) else getattr(line, "source_logical_page", None)
        )
        out.append(
            {
                "idx": idx,
                "text": t,
                "bbox": b,
                "confidence": _confidence(line),
                **({"source_logical_page": int(source_logical_page)} if source_logical_page else {}),
            }
        )
    out.sort(key=lambda x: (x["bbox"][1], x["bbox"][0]))
    return out


def _months_between(start_year: int, start_month: int, end_year: int, end_month: int) -> list[tuple[int, int]]:
    months: list[tuple[int, int]] = []
    y, m = start_year, start_month
    while (y, m) <= (end_year, end_month):
        months.append((y, m))
        m += 1
        if m > 12:
            y += 1
            m = 1
    return months


def _expand_line_to_char_tokens(line: dict[str, Any], *, page: int, prefix: str) -> list[OCRToken]:
    """Split a merged OCR line into approximate single-character tokens."""
    text = line["text"]
    chars = [ch for ch in text if not ch.isspace()]
    if not chars:
        return []
    x0, y0, x1, y1 = line["bbox"]
    width = max(x1 - x0, 1.0)
    step = width / len(chars)
    tokens = []
    for i, ch in enumerate(chars):
        tokens.append(
            OCRToken(
                token_id=f"{prefix}_{line['idx']}_{i}",
                text=ch,
                bbox=(x0 + step * i, y0, x0 + step * (i + 1), y1),
                confidence=line.get("confidence", 1.0),
                page=page,
                source="ocr_line_split",
                source_token_id=f"line_{line['idx']}",
            )
        )
    return tokens


def _cell_bbox(row_band: dict[str, Any], col_band: dict[str, Any]) -> BBox:
    return cell_bbox(row_band, col_band)


def _assign_row(
    tokens: list[OCRToken], row_band: dict[str, Any], cols: list[dict[str, Any]]
) -> dict[int, list[OCRToken]]:
    return cast(dict[int, list[OCRToken]], assign_tokens_to_col_bands(tokens, row_band, cols))


def _token_text(tokens: list[OCRToken], *, allowed: set[str] | None = None) -> str:
    ordered = sorted(tokens, key=lambda t: (t.bbox[0], t.bbox[1]))
    text = "".join(t.text for t in ordered).strip()
    if allowed is not None:
        text = "".join(ch for ch in text if ch in allowed)
    return text


def _line_split_tokens_under_anchor(line_items: list[dict[str, Any]], *, ay1: float, page: int) -> list[OCRToken]:
    out: list[OCRToken] = []
    for line in line_items:
        ly0 = line["bbox"][1]
        if ay1 <= ly0 <= ay1 + 170:
            out.extend(_expand_line_to_char_tokens(line, page=page, prefix=f"ocr_p{page}_repay"))
    return out


def _normalize_amount_text(text: str) -> str:
    normalized = normalize_allowlist_text(text, set("0123456789.,"), max_chars=16)
    compact = normalized.replace(",", "").replace(".", "")
    if compact and set(compact) == {"0"}:
        return "0"
    return str(normalized)


def _repayment_business_signature(record: dict[str, Any]) -> tuple[str, str | None]:
    """Return the normalized business values that define one monthly result."""
    status = str(record.get("status_code") or record.get("status") or "").strip().upper()
    raw_amount = record.get("overdue_amount")
    if raw_amount in (None, ""):
        raw_amount = record.get("status_amount")
    if raw_amount in (None, ""):
        amount = None
    else:
        compact = re.sub(r"[,，\s]", "", str(raw_amount))
        try:
            amount = format(Decimal(compact).normalize(), "f")
            if amount == "-0":
                amount = "0"
        except (InvalidOperation, TypeError, ValueError):
            amount = f"raw:{compact}"
    return status, amount


def _merged_source_cell_refs(*records: dict[str, Any]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    markers: set[str] = set()
    for record in records:
        for ref in record.get("source_cell_refs") or ():
            if not isinstance(ref, dict):
                continue
            marker = repr(sorted(ref.items()))
            if marker in markers:
                continue
            markers.add(marker)
            refs.append(dict(ref))
    return refs


def _strong_visual_status(recognition: Any) -> bool:
    """Return whether crop evidence is strong enough to override row OCR."""
    if not str(getattr(recognition, "text", "") or ""):
        return False
    if (
        str(getattr(recognition, "source", "") or "")
        not in {
            "cell_crop_consensus",
            "unavailable",
            "none",
        }
        and float(getattr(recognition, "confidence", 0.0) or 0.0) >= 0.8
    ):
        return True
    audit = dict(getattr(recognition, "audit", {}) or {})
    if int(audit.get("consensus_count") or 0) >= 2:
        return True
    for vote in audit.get("votes") or []:
        if not isinstance(vote, dict):
            continue
        if (
            vote.get("variant") in {"glyph_shape", "glyph_shape_n"}
            and str(vote.get("text") or "") == str(getattr(recognition, "text", "") or "")
            and float(vote.get("confidence") or 0.0) >= 0.8
        ):
            return True
    return False


@lru_cache(maxsize=1)
def _static_status_reference_bank() -> tuple[Any, tuple[str, ...]]:
    """Build a compact OCR-free status bank from OpenCV's bundled fonts."""

    try:
        import cv2
        import numpy as np

        labels: list[str] = []
        vectors: list[Any] = []
        characters = "N*HMBAZGCHX0123456789#+./"
        fonts = (
            cv2.FONT_HERSHEY_SIMPLEX,
            cv2.FONT_HERSHEY_PLAIN,
            cv2.FONT_HERSHEY_DUPLEX,
            cv2.FONT_HERSHEY_COMPLEX,
            cv2.FONT_HERSHEY_TRIPLEX,
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
            cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
        )
        for character in characters:
            for font in fonts:
                for font_scale in (0.72, 0.98):
                    for thickness in (1, 2):
                        image = np.full((80, 80), 255, dtype=np.uint8)
                        (width, height), _baseline = cv2.getTextSize(
                            character, font, font_scale, thickness
                        )
                        cv2.putText(
                            image,
                            character,
                            ((80 - width) // 2, (80 + height) // 2),
                            font,
                            font_scale,
                            0,
                            thickness,
                            cv2.LINE_AA,
                        )
                        template = _normalize_static_reference_glyph(image, np=np, cv2=cv2)
                        vector = _static_status_feature_vector(template, np=np, cv2=cv2)
                        if vector is None:
                            continue
                        labels.append(character)
                        vectors.append(vector)
        if not vectors:
            return None, ()
        return np.stack(vectors), tuple(labels)
    except Exception:
        return None, ()


def _normalize_static_reference_glyph(image: Any, *, np: Any, cv2: Any) -> Any | None:
    ys, xs = np.where(image < 128)
    if not len(xs):
        return None
    glyph = image[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    ink_full = (glyph < 128).astype(np.float32)
    dense_cols = np.where(
        ink_full.mean(axis=0) >= max(0.12, 2.0 / max(ink_full.shape[0], 1))
    )[0]
    dense_rows = np.where(
        ink_full.mean(axis=1) >= max(0.12, 2.0 / max(ink_full.shape[1], 1))
    )[0]
    if len(dense_cols) and len(dense_rows):
        x0 = max(0, int(dense_cols.min()) - 1)
        x1 = min(glyph.shape[1], int(dense_cols.max()) + 2)
        y0 = max(0, int(dense_rows.min()) - 1)
        y1 = min(glyph.shape[0], int(dense_rows.max()) + 2)
        glyph = glyph[y0:y1, x0:x1]
    height, width = glyph.shape[:2]
    scale = min(22.0 / max(width, 1), 22.0 / max(height, 1))
    resized = cv2.resize(
        glyph,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_NEAREST,
    )
    canvas = np.zeros((32, 32), dtype=np.float32)
    ink = (resized < 128).astype(np.float32)
    y0 = (32 - ink.shape[0]) // 2
    x0 = (32 - ink.shape[1]) // 2
    canvas[y0 : y0 + ink.shape[0], x0 : x0 + ink.shape[1]] = ink
    return canvas


def _static_status_feature_vector(template: Any, *, np: Any, cv2: Any) -> Any | None:
    if template is None:
        return None
    normalized = np.asarray(template, dtype=np.float32)
    if normalized.size == 0:
        return None
    blurred = cv2.GaussianBlur(normalized, (3, 3), 0.7)
    compact = cv2.resize(blurred, (16, 16), interpolation=cv2.INTER_AREA)
    vector = compact.reshape(-1)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else None


def _static_status_reference_scores(template: Any, *, np: Any) -> dict[str, float]:
    import cv2

    bank, labels = _static_status_reference_bank()
    if bank is None or not labels:
        return {}
    vector = _static_status_feature_vector(template, np=np, cv2=cv2)
    if vector is None or vector.size != int(bank.shape[1]):
        return {}
    similarities = bank @ vector
    scores: dict[str, float] = {}
    for index, label in enumerate(labels):
        scores[label] = max(scores.get(label, 0.0), float(similarities[index]))
    return scores


def _static_n_star_glyph_classification(
    template: Any,
) -> tuple[str, float, dict[str, Any]] | None:
    """Classify the two OCR-confusable zero-status glyphs without invoking OCR.

    PBOC monthly grids use a tall two-stem ``N`` and a centre-heavy asterisk.
    The normalized bitmap is produced from the already rendered page crop, so
    this check is deterministic, field-specific, and CPU-cheap.  Deliberately
    ambiguous shapes are not guessed.
    """

    try:
        import cv2
        import numpy as np

        ink = np.asarray(template, dtype=np.float32) >= 0.5
        ys, xs = np.where(ink)
        if not len(xs):
            return None
        compact = ink[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        height, width = compact.shape[:2]
        if height < 6 or width < 5:
            return None
        ink_density = float(compact.mean())
        reference_scores = _static_status_reference_scores(template, np=np)
        ranked = sorted(
            ((score, label) for label, score in reference_scores.items()),
            reverse=True,
        )
        if not ranked:
            return None
        winning_score, winning_label = ranked[0]
        competitor_score = ranked[1][0] if len(ranked) > 1 else 0.0
        margin = winning_score - competitor_score
        contours = cv2.findContours(
            compact.astype(np.uint8) * 255,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )[-2]
        solidity = 1.0
        if contours:
            contour = max(contours, key=cv2.contourArea)
            contour_area = float(cv2.contourArea(contour))
            hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
            if hull_area > 0:
                solidity = contour_area / hull_area
        audit: dict[str, Any] = {
            "ink_density": round(ink_density, 4),
            "reference_winner": winning_label,
            "reference_similarity": round(winning_score, 4),
            "reference_margin": round(margin, 4),
            "solidity": round(solidity, 4),
            "classification_basis": "conservative_reference_bank",
        }
        if winning_label == "N" and winning_score >= 0.90 and margin >= 0.020:
            confidence = min(0.98, 0.68 + winning_score * 0.3)
            return "N", confidence, audit
        if (
            winning_label == "*"
            and winning_score >= 0.90
            and margin >= 0.030
            and solidity <= 0.75
        ):
            confidence = min(0.98, 0.68 + winning_score * 0.3)
            return "*", confidence, audit
    except Exception:
        return None
    return None


def _neighbor_status_fallback(
    statuses: dict[int, str],
    month: int,
    *,
    zero_overdue_statuses: set[str] | frozenset[str] = frozenset(_ZERO_OVERDUE_STATUSES),
) -> str:
    """Repair one unreadable cell only when adjacent row context is decisive."""
    left = [statuses.get(month - offset, "") for offset in (1, 2)]
    right = [statuses.get(month + offset, "") for offset in (1, 2)]
    if left[0] and right[0] and left[0] == right[0] and left[0] in zero_overdue_statuses:
        return left[0]
    if not left[0] and right[0] and right[0] == right[1] and right[0] in zero_overdue_statuses:
        return right[0]
    if not right[0] and left[0] and left[0] == left[1] and left[0] in zero_overdue_statuses:
        return left[0]
    return ""


def _template_visual_status(recognition: Any, *, min_confidence: float) -> bool:
    text = str(getattr(recognition, "text", "") or "")
    for vote in (getattr(recognition, "audit", {}) or {}).get("votes") or []:
        if (
            isinstance(vote, dict)
            and vote.get("variant") == "document_glyph_template"
            and str(vote.get("text") or "") == text
            and float(vote.get("confidence") or 0.0) >= min_confidence
        ):
            return True
    return False


def _find_anchor(lines: list[dict[str, Any]]) -> tuple[dict[str, Any], tuple[int, int, int, int]] | None:
    for line in lines:
        normalized = line["text"].replace(" ", "")
        m = _RANGE_RE.search(normalized)
        if m:
            sy, sm, ey, em = map(int, m.groups())
            return line, (sy, sm, ey, em)
    return None


def _nearest_year_lines(
    lines: list[dict[str, Any]],
    anchor: dict[str, Any],
    *,
    start_year: int,
    end_year: int,
) -> list[dict[str, Any]]:
    ax0, ay0, ax1, ay1 = anchor["bbox"]
    expected_years = set(range(start_year, end_year + 1))
    candidates = sorted(
        [
        line
        for line in lines
        if (match := _YEAR_RE.match(line["text"].strip()))
        and int(match.group(0)) in expected_years
        and line["bbox"][1] > ay1
        and line["bbox"][1] < ay1 + 300
        ],
        key=lambda line: (float(line["bbox"][1]), float(line["bbox"][0])),
    )
    # Select one physical row for each year in the printed date range.  The
    # former four-row slice could silently include an unrelated year or omit a
    # valid row when OCR duplicated a year label.
    selected: list[dict[str, Any]] = []
    seen: set[int] = set()
    for line in candidates:
        match = _YEAR_RE.match(line["text"].strip())
        if match is None:
            continue
        year = int(match.group(0))
        if year in seen:
            continue
        seen.add(year)
        selected.append(line)
    return selected


def _month_col_bands(header_line: dict[str, Any], *, n_months: int = 12) -> list[dict[str, Any]]:
    return cast(
        list[dict[str, Any]],
        equal_col_bands(header_line["bbox"], count=n_months, start_index=1, role="month"),
    )


def _visual_page_context(
    *,
    source_line: dict[str, Any],
    bbox: BBox,
    base_page: int,
    base_page_width: float | None,
    base_page_height: float | None,
    page_image: Any | None,
    page_image_resolver: Any | None,
) -> tuple[Any, BBox, float, float, int] | None:
    """Resolve a local-page image and undo cross-page evidence y shifting."""
    logical_page = int(source_line.get("source_logical_page") or base_page)
    context = page_image_resolver(logical_page) if page_image_resolver is not None else None
    if isinstance(context, dict):
        image = context.get("image")
        width = float(context.get("page_width") or base_page_width or 0.0)
        height = float(context.get("page_height") or base_page_height or 0.0)
    else:
        image = page_image if logical_page == base_page else None
        width = float(base_page_width or 0.0)
        height = float(base_page_height or 0.0)
    if image is None or width <= 0 or height <= 0:
        return None
    x0, y0, x1, y1 = bbox
    if logical_page != base_page:
        shift = float(base_page_height or 0.0)
        y0 -= shift
        y1 -= shift
    return image, (x0, y0, x1, y1), width, height, logical_page


def _local_page_bbox(
    bbox: BBox,
    *,
    logical_page: int,
    base_page: int,
    base_page_height: float | None,
) -> list[float]:
    """Undo the one-page continuation shift used by joined evidence rows."""
    x0, y0, x1, y1 = (float(value) for value in bbox)
    if logical_page != base_page and base_page_height:
        y0 -= float(base_page_height)
        y1 -= float(base_page_height)
    return [x0, y0, x1, y1]


def _visual_month_col_bands(
    month_cols: list[dict[str, Any]],
    *,
    page_image: Any | None,
    page_width: float | None,
    page_height: float | None,
    y0: float,
    y1: float,
    max_left_shift_months: float = 1.10,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Align month bands to vertical table rules while preserving legacy text geometry."""
    if (
        page_image is None
        or page_width is None
        or page_height is None
        or not month_cols
        or getattr(page_image, "size", 0) == 0
    ):
        return month_cols, {"source": "text_bbox", "offset": 0.0}
    try:
        import cv2
        import numpy as np

        shape = page_image.shape
        image_height, image_width = int(shape[0]), int(shape[1])
        gray = cv2.cvtColor(page_image, cv2.COLOR_RGB2GRAY) if len(shape) == 3 else page_image
        sx = image_width / max(float(page_width), 1.0)
        sy = image_height / max(float(page_height), 1.0)
        start = float(month_cols[0]["bbox"][0])
        end = float(month_cols[-1]["bbox"][2])
        step = (end - start) / len(month_cols)
        py0 = max(0, int(round(max(0.0, y0) * sy)))
        py1 = min(image_height, int(round(min(float(page_height), y1) * sy)))
        if py1 - py0 < 12 or step <= 1.0:
            return month_cols, {"source": "text_bbox", "offset": 0.0}
        ink = (gray[py0:py1] < 115).astype(np.float32)
        projection = ink.mean(axis=0)
        projection = np.convolve(projection, np.ones(5, dtype=np.float32) / 5.0, mode="same")
        # Month glyph boxes are narrower than the table: find both outside
        # rules, allowing the column pitch to expand instead of applying a
        # single offset that becomes wrong near month 12.
        start_offsets = np.linspace(-max(0.2, max_left_shift_months) * step, -0.10 * step, 113)
        end_offsets = np.linspace(-0.90 * step, 0.55 * step, 89)
        best_score = -1.0
        best_start, best_end = start, end
        for start_offset in start_offsets:
            candidate_start = start + float(start_offset)
            for end_offset in end_offsets:
                candidate_end = end + float(end_offset)
                positions = np.linspace(candidate_start * sx, candidate_end * sx, 13)
                indices = np.clip(np.rint(positions).astype(int), 0, image_width - 1)
                score = float(projection[indices].sum())
                if score > best_score:
                    best_score = score
                    best_start, best_end = candidate_start, candidate_end
        best_offset = best_start - start
        # Reject offsets without sustained vertical-rule evidence.
        baseline = float(np.median(projection)) * 13.0
        if best_score < max(0.5, baseline * 1.05):
            return month_cols, {"source": "text_bbox", "offset": 0.0}
        refined = equal_col_bands(
            (best_start, float(month_cols[0]["bbox"][1]), best_end, float(month_cols[0]["bbox"][3])),
            count=12,
            start_index=1,
            role="month",
        )
        return refined, {
            "source": "vertical_rule_projection",
            "offset": round(best_offset, 4),
            "right_offset": round(best_end - end, 4),
            "score": round(best_score, 4),
        }
    except Exception:
        return month_cols, {"source": "text_bbox", "offset": 0.0}


def _coerce_token(obj: Any, *, page: int, idx: int) -> OCRToken | None:
    if isinstance(obj, OCRToken):
        return obj
    b = _bbox(obj)
    text = _text(obj)
    if not b or not text:
        return None
    token_id = obj.get("token_id") if isinstance(obj, dict) else getattr(obj, "token_id", None)
    raw_bbox = obj.get("raw_bbox") if isinstance(obj, dict) else getattr(obj, "raw_bbox", None)
    raw: BBox | None = None
    if raw_bbox and len(raw_bbox) == 4:
        raw = (float(raw_bbox[0]), float(raw_bbox[1]), float(raw_bbox[2]), float(raw_bbox[3]))
    return OCRToken(
        token_id=str(token_id or f"ocr_p{page}_t{idx}"),
        text=text,
        bbox=b,
        confidence=_confidence(obj),
        page=page,
        source=str(obj.get("source", "rapidocr") if isinstance(obj, dict) else getattr(obj, "source", "rapidocr")),
        coordinate_system=str(
            obj.get("coordinate_system", "pdf_points_top_left")
            if isinstance(obj, dict)
            else getattr(obj, "coordinate_system", "pdf_points_top_left")
        ),
        raw_bbox=raw,
        raw_coordinate_system=str(
            obj.get("raw_coordinate_system", "image_pixels")
            if isinstance(obj, dict)
            else getattr(obj, "raw_coordinate_system", "image_pixels")
        ),
    )


def _coerce_tokens(tokens: Iterable[Any] | None, *, page: int) -> list[OCRToken]:
    out: list[OCRToken] = []
    for idx, token in enumerate(tokens or []):
        coerced = _coerce_token(token, page=page, idx=idx)
        if coerced is not None:
            out.append(coerced)
    return out


def _row_band(line: dict[str, Any], role: str, *, x0: float, x1: float, pad_y: float = 3.0) -> dict[str, Any]:
    _lx0, y0, _lx1, y1 = line["bbox"]
    return {
        "index": -1,
        "role": role,
        "bbox": [x0, y0 - pad_y, x1, y1 + pad_y],
        "geometry_status": "estimated",
        "source_line_index": line["idx"],
    }


def _line_after(
    lines: list[dict[str, Any]], y: float, *, x_min: float, x_max: float, max_gap: float = 55.0
) -> dict[str, Any] | None:
    candidates = []
    for line in lines:
        lx0, ly0, lx1, ly1 = line["bbox"]
        if ly0 <= y or ly0 - y > max_gap:
            continue
        if lx1 < x_min or lx0 > x_max:
            continue
        candidates.append(line)
    return candidates[0] if candidates else None


def _find_month_header(lines: list[dict[str, Any]], anchor: dict[str, Any]) -> dict[str, Any] | None:
    _ax0, ay0, _ax1, ay1 = anchor["bbox"]
    candidates: list[tuple[int, dict[str, Any]]] = []
    for line in lines:
        _lx0, ly0, _lx1, ly1 = line["bbox"]
        if ly1 < ay0 or ly0 > ay1 + 40.0:
            continue
        months = [int(value) for value in re.findall(r"(?<!\d)(?:1[0-2]|[1-9])(?!\d)", line["text"])]
        score = len(set(months) & set(range(1, 13)))
        if score >= 8:
            candidates.append((score, line))
    return max(candidates, key=lambda item: (item[0], -float(item[1]["bbox"][1])))[1] if candidates else None


def _line_before(
    lines: list[dict[str, Any]], y: float, *, x_min: float, x_max: float, max_gap: float = 55.0
) -> dict[str, Any] | None:
    candidates = []
    for line in lines:
        lx0, ly0, lx1, ly1 = line["bbox"]
        if ly0 >= y or y - ly0 > max_gap:
            continue
        if lx1 < x_min or lx0 > x_max:
            continue
        candidates.append(line)
    return candidates[-1] if candidates else None


def reconstruct_repayment_micro_grid_from_lines(
    lines: Iterable[Any],
    *,
    page: int,
    tokens: Iterable[Any] | None = None,
    page_width: float | None = None,
    page_height: float | None = None,
    page_image: Any | None = None,
    page_image_resolver: Any | None = None,
    enable_cell_ocr: bool = False,
    enable_static_status_validation: bool = False,
    extra_status_chars: Iterable[str] = (),
    grid_index: int = 0,
) -> RepaymentExtraction:
    """Reconstruct a credit repayment grid from OCR line-level geometry."""
    line_items = _line_items(lines)
    status_charset = _STATUS_CHARS | {str(item) for item in extra_status_chars if str(item)}
    zero_overdue_statuses = _ZERO_OVERDUE_STATUSES | ({"A"} if "A" in status_charset else set())
    evidence_tokens = _coerce_tokens(tokens, page=page)
    candidates = detect_micro_grid_candidates(
        evidence_tokens,
        lines=line_items,
        page=page,
        page_width=page_width,
        page_height=page_height,
    )
    found = _find_anchor(line_items)
    if not found:
        return RepaymentExtraction(
            None,
            [],
            {
                "reason": "anchor_not_found",
                "micro_grid_candidates": [candidate.to_dict() for candidate in candidates],
            },
        )

    anchor, (start_year, start_month, end_year, end_month) = found
    ax0, ay0, ax1, ay1 = anchor["bbox"]
    header_line = _find_month_header(line_items, anchor) or _line_after(
        line_items, ay1, x_min=ax0 - 220, x_max=ax1 + 260, max_gap=35.0
    )
    if header_line is None:
        return RepaymentExtraction(None, [], {"reason": "month_header_not_found", "anchor": anchor})

    years = _nearest_year_lines(
        line_items,
        anchor,
        start_year=start_year,
        end_year=end_year,
    )
    if not years:
        return RepaymentExtraction(None, [], {"reason": "year_rows_not_found", "anchor": anchor})

    month_cols = _month_col_bands(header_line)
    base_visual_context = page_image_resolver(page) if page_image_resolver is not None else None
    base_visual_image = base_visual_context.get("image") if isinstance(base_visual_context, dict) else page_image
    base_visual_width = (
        float(base_visual_context.get("page_width") or page_width or 0.0)
        if isinstance(base_visual_context, dict)
        else page_width
    )
    base_visual_height = (
        float(base_visual_context.get("page_height") or page_height or 0.0)
        if isinstance(base_visual_context, dict)
        else page_height
    )
    visual_month_cols, visual_geometry_audit = _visual_month_col_bands(
        month_cols,
        page_image=base_visual_image,
        page_width=base_visual_width,
        page_height=base_visual_height,
        y0=float(header_line["bbox"][1]) - 5.0,
        y1=min(
            float(base_visual_height or page_height or 0.0),
            max(float(year_line["bbox"][3]) for year_line in years) + 35.0,
        ),
    )
    month_x0 = min(col["bbox"][0] for col in month_cols)
    grid_x1 = max(col["bbox"][2] for col in month_cols)
    year_x0 = min(float(year_line["bbox"][0]) for year_line in years)
    year_x1 = max(float(year_line["bbox"][2]) for year_line in years)
    year_y1 = max(float(year_line["bbox"][3]) for year_line in years)
    year_col_band: dict[str, Any] = {
        "index": 0,
        "header": "year",
        "role": "year",
        "bbox": [year_x0, header_line["bbox"][1], max(month_x0, year_x1), year_y1],
        "geometry_status": "estimated",
    }
    grid_x0 = min(float(year_col_band["bbox"][0]), month_x0)

    synthetic_tokens: list[OCRToken] = []
    if evidence_tokens:
        roi_tokens = [token for token in evidence_tokens if ay1 <= token.center[1] <= ay1 + 170]
        synthetic_tokens = expand_tokens_to_char_tokens(roi_tokens)
        synthetic_tokens.extend(_line_split_tokens_under_anchor(line_items, ay1=ay1, page=page))
        synthetic_tokens = dedupe_visual_tokens(synthetic_tokens)
        token_source = "ocr_tokens+line_bbox_fallback+char_split"
    else:
        synthetic_tokens.extend(_line_split_tokens_under_anchor(line_items, ay1=ay1, page=page))
        synthetic_tokens = dedupe_visual_tokens(synthetic_tokens)
        token_source = "ocr_line_bbox+char_split"

    row_bands: list[dict[str, Any]] = []
    cell_rows: list[list[MicroGridCell]] = []

    anchor_band = _row_band(anchor, "anchor", x0=grid_x0, x1=grid_x1)
    anchor_band["index"] = 0
    row_bands.append(anchor_band)
    anchor_col = {
        "index": 0,
        "header": "anchor",
        "role": "anchor",
        "bbox": list(anchor["bbox"]),
        "geometry_status": "exact",
    }
    cell_rows.append(
        [
            build_cell(
                row_band=anchor_band,
                col_band=anchor_col,
                tokens=[],
                text=anchor["text"],
                role="anchor",
            )
        ]
    )

    header_band = _row_band(header_line, "month_header", x0=grid_x0, x1=grid_x1)
    header_band["index"] = 1
    row_bands.append(header_band)
    header_col = {
        "index": 0,
        "header": "months",
        "role": "month_header",
        "bbox": list(header_line["bbox"]),
        "geometry_status": "exact",
    }
    cell_rows.append(
        [
            build_cell(
                row_band=header_band,
                col_band=header_col,
                tokens=[],
                text=header_line["text"],
                role="month_header",
            )
        ]
    )

    records: list[dict[str, Any]] = []
    record_months = set(_months_between(start_year, start_month, end_year, end_month))
    crop_ocr_attempts = 0
    crop_ocr_hits = 0
    static_status_attempts = 0
    static_status_resolved = 0
    static_status_corrections = 0
    static_status_unresolved = 0
    static_status_unavailable = 0
    status_templates: dict[str, list[Any]] = {}

    for year_line in years:
        year_match = _YEAR_RE.match(year_line["text"].strip())
        if year_match is None:
            continue
        year = int(year_match.group(0))
        status_line = _line_before(
            line_items, year_line["bbox"][1], x_min=grid_x0 - 10, x_max=grid_x1 + 10, max_gap=55.0
        )
        if status_line is None or status_line is header_line or "还款记录" in status_line["text"]:
            status_line = _line_after(
                line_items, year_line["bbox"][1], x_min=grid_x0 - 10, x_max=grid_x1 + 10, max_gap=55.0
            )
        if status_line is None:
            continue
        # Credit-report grids render the calendar year in the first cell of
        # the overdue-amount row.  The immediately preceding row is status.
        amount_line = year_line

        status_band = _row_band(status_line, "status", x0=grid_x0, x1=grid_x1)
        status_band["index"] = len(row_bands)
        row_bands.append(status_band)
        amount_band = None
        if amount_line is not None:
            amount_band = _row_band(amount_line, "overdue_amount", x0=grid_x0, x1=grid_x1)
            amount_band["index"] = len(row_bands)
            row_bands.append(amount_band)

        year_visual_cols = visual_month_cols
        row_visual_context = _visual_page_context(
            source_line=status_line,
            bbox=(grid_x0, status_band["bbox"][1], grid_x1, (amount_band or status_band)["bbox"][3]),
            base_page=page,
            base_page_width=page_width,
            base_page_height=page_height,
            page_image=page_image,
            page_image_resolver=page_image_resolver,
        )
        if row_visual_context is not None:
            row_image, row_local_bbox, row_width, row_height, row_page = row_visual_context
            year_visual_cols, _row_geometry_audit = _visual_month_col_bands(
                month_cols,
                page_image=row_image,
                page_width=row_width,
                page_height=row_height,
                y0=max(0.0, row_local_bbox[1] - 5.0),
                y1=min(row_height, row_local_bbox[3] + 5.0),
                max_left_shift_months=1.85 if row_page != page else 1.10,
            )
        year_visual_cols_by_month = {int(col["header"]): col for col in year_visual_cols}

        year_col = {
            "index": 0,
            "header": str(year),
            "role": "year",
            "bbox": list(year_line["bbox"]),
            "geometry_status": "exact",
        }
        status_cells: list[MicroGridCell] = [
            build_cell(
                row_band=status_band,
                col_band=year_col,
                tokens=[],
                text=str(year),
                role="year",
            )
        ]
        amount_cells: list[MicroGridCell] = []
        normalized_status_text = status_line["text"].replace("★", "*").replace("☆", "*").replace("※", "*")
        raw_status_text = normalized_status_text.replace(" ", "")
        # Preserve an OCR uncertainty marker for positional alignment. It is
        # never emitted as a credit status, but dropping it here would shift
        # every later month one cell to the left.
        status_chars = [ch for ch in raw_status_text if ch in status_charset or ch == "#"]
        active_months = [month for yy, month in sorted(record_months) if yy == year]
        if year == end_year and len(status_chars) == 2 and set(status_chars) == {"N", "C"}:
            status_by_month = {active_months[0]: "N", active_months[1]: "C"} if len(active_months) == 2 else {}
            row_alignment_exact = False
        elif len(status_chars) == 12:
            status_by_month = dict(zip(range(1, 13), status_chars))
            row_alignment_exact = "#" not in status_chars
        elif len(status_chars) == len(active_months):
            status_by_month = dict(zip(active_months, status_chars))
            row_alignment_exact = "#" not in status_chars
        # Domain correction: closed credit-report rows commonly render the final
        # month as C. If OCR collapsed a two-month N/C row into "CN", use the
        # anchor date range and visual convention to place C on the final month.
        else:
            status_by_month = {}
            row_alignment_exact = False

        status_row_tokens = _expand_line_to_char_tokens(
            {**status_line, "text": normalized_status_text},
            page=page,
            prefix=f"ocr_p{page}_repay_status_{year}",
        )
        amount_row_tokens = _expand_line_to_char_tokens(
            amount_line,
            page=page,
            prefix=f"ocr_p{page}_repay_amount_{year}",
        )
        status_assignments = _assign_row(status_row_tokens, status_band, month_cols)
        amount_assignments = _assign_row(amount_row_tokens, amount_band, month_cols) if amount_band is not None else {}

        for col in month_cols:
            month = int(col["header"])
            st_tokens = status_assignments.get(month, [])
            status_center_y = (float(status_line["bbox"][1]) + float(status_line["bbox"][3])) / 2.0
            amount_center_y = (float(amount_line["bbox"][1]) + float(amount_line["bbox"][3])) / 2.0
            st_tokens = [
                token
                for token in st_tokens
                if abs(token.center[1] - status_center_y) <= abs(token.center[1] - amount_center_y)
            ]
            status = normalize_allowlist_text(
                status_by_month.get(month) or _token_text(st_tokens), status_charset, max_chars=1
            )
            status_crop = None
            status_recognition_source = "tokens"
            status_recognition_audit: dict[str, Any] = {}
            static_status_failed = False
            if row_alignment_exact and month in status_by_month:
                status_recognition_source = "canonical_row_sequence"
                status_recognition_audit = {
                    "alignment_status": "exact",
                    "expected_cell_count": len(active_months),
                    "observed_status_count": len(status_chars),
                    "active_months": list(active_months),
                    "logical_page": int(status_line.get("source_logical_page") or page),
                }
            neighbor_status = (
                _neighbor_status_fallback(
                    status_by_month,
                    month,
                    zero_overdue_statuses=zero_overdue_statuses,
                )
                if not status
                else ""
            )
            if neighbor_status:
                status = neighbor_status
                status_recognition_source = "row_neighbor_consensus"
                status_recognition_audit = {
                    "reason": "unreadable_cell_with_matching_adjacent_statuses",
                    "logical_page": int(status_line.get("source_logical_page") or page),
                }
            visual_status_bbox = _cell_bbox(status_band, year_visual_cols_by_month.get(month, col))
            visual = None
            if (
                (enable_cell_ocr or (enable_static_status_validation and status in {"N", "*"}))
                and (year, month) in record_months
            ):
                visual = _visual_page_context(
                    source_line=status_line,
                    bbox=visual_status_bbox,
                    base_page=page,
                    base_page_width=page_width,
                    base_page_height=page_height,
                    page_image=page_image,
                    page_image_resolver=page_image_resolver,
                )
            if enable_cell_ocr and (year, month) in record_months:
                if visual is not None:
                    crop_ocr_attempts += 1
                    visual_image, visual_bbox, visual_width, visual_height, visual_page = visual
                    rec = recognize_micro_cell_from_image(
                        visual_image,
                        visual_bbox,
                        page_width=visual_width,
                        page_height=visual_height,
                        allowed_charset=status_charset,
                        max_chars=1,
                        reference_templates=status_templates,
                    )
                    status_crop = rec.raw_text
                    status_recognition_source = rec.source
                    status_recognition_audit = {**rec.audit, "logical_page": visual_page}
                    if rec.text:
                        crop_ocr_hits += 1
                        consensus_count = int((rec.audit or {}).get("consensus_count") or 0)
                        weak_cell = not status or status.isdigit()
                        template_confirmed = _template_visual_status(
                            rec,
                            min_confidence=0.9 if status.isdigit() else 0.8,
                        )
                        if (weak_cell and (_strong_visual_status(rec) or template_confirmed)) or consensus_count >= 2:
                            status = rec.text
            static_template = None
            if (
                enable_static_status_validation
                and status in {"N", "*"}
                and (year, month) in record_months
            ):
                static_status_attempts += 1
                observed_row_status = status
                if visual is None:
                    status = ""
                    static_status_failed = True
                    static_status_unresolved += 1
                    static_status_unavailable += 1
                    status_recognition_source = "static_glyph_shape_unavailable"
                    status_recognition_audit = {
                        "reason": "page_image_or_cell_crop_unavailable",
                        "observed_row_status": observed_row_status,
                        "logical_page": int(status_line.get("source_logical_page") or page),
                    }
                else:
                    visual_image, visual_bbox, visual_width, visual_height, visual_page = visual
                    static_template = extract_micro_cell_glyph_template(
                        visual_image,
                        visual_bbox,
                        page_width=visual_width,
                        page_height=visual_height,
                    )
                    classification = (
                        _static_n_star_glyph_classification(static_template)
                        if static_template is not None
                        else None
                    )
                if visual is not None and classification is None:
                    # Keep the month as a typed uncertainty candidate so the
                    # final schema gate can report the exact affected month.
                    status = ""
                    static_status_failed = True
                    static_status_unresolved += 1
                    status_recognition_source = "static_glyph_shape_unresolved"
                    status_recognition_audit = {
                        "reason": "n_star_glyph_not_decisive",
                        "observed_row_status": observed_row_status,
                        "logical_page": visual_page,
                    }
                elif visual is not None and classification is not None:
                    visual_status, visual_confidence, visual_audit = classification
                    static_status_resolved += 1
                    if visual_status != observed_row_status:
                        static_status_corrections += 1
                    status = visual_status
                    status_recognition_source = "static_glyph_shape_validation"
                    status_recognition_audit = {
                        "reason": "field_specific_n_star_shape_validation",
                        "observed_row_status": observed_row_status,
                        "visual_status": visual_status,
                        "visual_confidence": round(float(visual_confidence), 4),
                        "logical_page": visual_page,
                        **visual_audit,
                    }
            if not status and not static_status_failed:
                neighbor_status = _neighbor_status_fallback(
                    status_by_month,
                    month,
                    zero_overdue_statuses=zero_overdue_statuses,
                )
                if neighbor_status:
                    status = neighbor_status
                    status_recognition_source = "row_neighbor_consensus"
                    status_recognition_audit = {
                        "reason": "unreadable_cell_with_matching_adjacent_statuses",
                        "logical_page": int(status_line.get("source_logical_page") or page),
                    }
            if status and status != "#" and not status.isdigit() and visual is not None:
                visual_image, visual_bbox, visual_width, visual_height, _visual_page = visual
                template = static_template
                if template is None:
                    template = extract_micro_cell_glyph_template(
                        visual_image,
                        visual_bbox,
                        page_width=visual_width,
                        page_height=visual_height,
                    )
                if template is not None:
                    status_templates.setdefault(status, []).append(template)
            if len(status) > 1:
                status = status[0]
            status_logical_page = int(status_line.get("source_logical_page") or page)
            status_evidence_bbox = (
                visual_status_bbox if visual is not None else _cell_bbox(status_band, col)
            )
            status_recognition_audit["source_ref"] = {
                "page": status_logical_page,
                "logical_page": status_logical_page,
                "bbox": _local_page_bbox(
                    status_evidence_bbox,
                    logical_page=status_logical_page,
                    base_page=page,
                    base_page_height=page_height,
                ),
                "geometry_scope": "cell",
                "coordinate_system": "pdf_points_top_left",
                "grid_id": f"mg_p{page}_repayment_{grid_index}",
                "row": status_band["index"],
                "col": month,
                "field_name": "status",
            }
            st_cell = build_cell(
                row_band=status_band,
                col_band=col,
                tokens=st_tokens,
                text=status,
                role="status",
                crop_ocr_text=status_crop,
                recognition_source=status_recognition_source,
                recognition_audit=status_recognition_audit,
            )
            status_cells.append(st_cell)

            amount = ""
            status_amount_conflict = False
            amount_bbox = None
            if amount_band is not None:
                amt_tokens = amount_assignments.get(month, [])
                amount = _normalize_amount_text(_token_text(amt_tokens))
                status_implies_zero = status in zero_overdue_statuses
                status_amount_conflict = status_implies_zero and amount not in {"", "0"}
                if status_implies_zero and not status_amount_conflict:
                    # Infer semantic zero only when the amount cell is empty or
                    # already zero. A nonzero token is retained below because
                    # this layer cannot prove whether it is noise or business data.
                    amount = "0"
                amount_bbox = _cell_bbox(amount_band, col)
                visual_amount_bbox = _cell_bbox(amount_band, year_visual_cols_by_month.get(month, col))
                amount_crop = None
                if status_amount_conflict:
                    amount_recognition_source = "status_amount_conflict"
                    amount_recognition_audit = {
                        "reason": "zero_status_conflicts_with_observed_nonzero_amount",
                        "status": status,
                        "observed_amount": amount,
                    }
                elif status_implies_zero:
                    amount_recognition_source = "status_semantic_zero"
                    amount_recognition_audit = {"reason": "non_overdue_status_semantics"}
                else:
                    amount_recognition_source = "tokens"
                    amount_recognition_audit = {}
                if (
                    (not amount or status_recognition_source == "cell_crop_consensus")
                    and not status_implies_zero
                    and enable_cell_ocr
                    and (year, month) in record_months
                ):
                    visual = _visual_page_context(
                        source_line=amount_line,
                        bbox=visual_amount_bbox,
                        base_page=page,
                        base_page_width=page_width,
                        base_page_height=page_height,
                        page_image=page_image,
                        page_image_resolver=page_image_resolver,
                    )
                    if visual is not None:
                        crop_ocr_attempts += 1
                        visual_image, visual_bbox, visual_width, visual_height, visual_page = visual
                        rec = recognize_micro_cell_from_image(
                            visual_image,
                            visual_bbox,
                            page_width=visual_width,
                            page_height=visual_height,
                            allowed_charset=set("0123456789.,"),
                            max_chars=16,
                        )
                        amount_crop = rec.raw_text
                        amount_recognition_source = rec.source
                        amount_recognition_audit = {**rec.audit, "logical_page": visual_page}
                        if rec.text:
                            crop_ocr_hits += 1
                            amount = _normalize_amount_text(rec.text)
                amount_logical_page = int(amount_line.get("source_logical_page") or page)
                amount_recognition_audit["source_ref"] = {
                    "page": amount_logical_page,
                    "logical_page": amount_logical_page,
                    "bbox": _local_page_bbox(
                        amount_bbox,
                        logical_page=amount_logical_page,
                        base_page=page,
                        base_page_height=page_height,
                    ),
                    "geometry_scope": "cell",
                    "coordinate_system": "pdf_points_top_left",
                    "grid_id": f"mg_p{page}_repayment_{grid_index}",
                    "row": amount_band["index"],
                    "col": month,
                    "field_name": "overdue_amount",
                }
                amount_cells.append(
                    build_cell(
                        row_band=amount_band,
                        col_band=col,
                        tokens=amt_tokens,
                        text=amount,
                        role="overdue_amount",
                        crop_ocr_text=amount_crop,
                        recognition_source=amount_recognition_source,
                        recognition_audit=amount_recognition_audit,
                    )
                )

            if (year, month) in record_months and status:
                status_ref = {
                    "page": page,
                    "grid_id": f"mg_p{page}_repayment_{grid_index}",
                    "row": st_cell.row_index,
                    "col": month,
                }
                refs = [status_ref]
                if amount_band is not None:
                    refs.append(
                        {
                            "page": page,
                            "grid_id": f"mg_p{page}_repayment_{grid_index}",
                            "row": amount_band["index"],
                            "col": month,
                        }
                    )
                records.append(
                    {
                        "year": year,
                        "month": month,
                        "status": status,
                        "overdue_amount": amount if amount else ("0" if status in zero_overdue_statuses else None),
                        "status_bbox": list(st_cell.bbox),
                        **({"amount_bbox": list(amount_bbox)} if amount_bbox else {}),
                        "source_cell_refs": refs,
                        "confidence": st_cell.confidence or 0.7,
                        **(
                            {
                                "extraction_status": "review",
                                "audit": {
                                    "reason": "zero_status_conflicts_with_observed_nonzero_amount",
                                    "status": status,
                                    "observed_amount": amount,
                                },
                            }
                            if status_amount_conflict
                            else {}
                        ),
                    }
                )
        cell_rows.append(status_cells)
        if amount_cells:
            cell_rows.append(amount_cells)

    all_y = [anchor["bbox"][1], header_line["bbox"][1]]
    all_y.extend(b["bbox"][1] for b in row_bands)
    all_y.extend(b["bbox"][3] for b in row_bands)
    all_y.extend(float(year_line["bbox"][1]) for year_line in years)
    all_y.extend(float(year_line["bbox"][3]) for year_line in years)
    grid_bbox = (grid_x0, min(all_y), grid_x1, max(all_y))
    micro_grid = MicroGrid(
        grid_id=f"mg_p{page}_repayment_{grid_index}",
        page=page,
        bbox=grid_bbox,
        anchor_text=anchor["text"],
        row_bands=row_bands,
        col_bands=[year_col_band, *month_cols],
        cells=cell_rows,
        grid_type_hint="credit_repayment_record",
        geometry_source=f"{token_source}+estimated_month_bands",
        confidence=0.82,
        audit={
            "anchor_line_index": anchor["idx"],
            "header_line_index": header_line["idx"],
            "micro_grid_candidates": [candidate.to_dict() for candidate in candidates],
            "date_range": {
                "start_year": start_year,
                "start_month": start_month,
                "end_year": end_year,
                "end_month": end_month,
            },
            "zero_overdue_statuses": sorted(zero_overdue_statuses),
            "token_count": len(synthetic_tokens),
            "source_token_count": len(evidence_tokens),
            "cell_crop_ocr": {
                "enabled": bool(enable_cell_ocr),
                "attempts": crop_ocr_attempts,
                "hits": crop_ocr_hits,
            },
            "static_status_validation": {
                "enabled": bool(enable_static_status_validation),
                "attempts": static_status_attempts,
                "resolved": static_status_resolved,
                "corrections": static_status_corrections,
                "unresolved": static_status_unresolved,
                "unavailable": static_status_unavailable,
            },
            "visual_month_geometry": visual_geometry_audit,
        },
    )
    return RepaymentExtraction(
        micro_grid,
        records,
        {"reason": "ok" if records else "grid_materialized_without_status_cells", "record_count": len(records)},
    )


def extract_credit_repayment_records(
    lines: Iterable[Any],
    *,
    page: int,
    tokens: Iterable[Any] | None = None,
    page_width: float | None = None,
    page_height: float | None = None,
    page_image: Any | None = None,
    page_image_resolver: Any | None = None,
    enable_cell_ocr: bool = False,
    enable_static_status_validation: bool = False,
    extra_status_chars: Iterable[str] = (),
    grid_index: int = 0,
) -> dict[str, Any]:
    extraction = reconstruct_repayment_micro_grid_from_lines(
        lines,
        page=page,
        tokens=tokens,
        page_width=page_width,
        page_height=page_height,
        page_image=page_image,
        page_image_resolver=page_image_resolver,
        enable_cell_ocr=enable_cell_ocr,
        enable_static_status_validation=enable_static_status_validation,
        extra_status_chars=extra_status_chars,
        grid_index=grid_index,
    )
    return {
        "micro_grid": extraction.micro_grid.to_dict() if extraction.micro_grid else None,
        "repayment_records": extraction.records,
        "audit": extraction.audit,
    }


def _date_range_from_grid(grid: dict[str, Any]) -> tuple[int, int, int, int] | None:
    audit_range = (grid.get("audit") or {}).get("date_range") or {}
    if audit_range.get("start_year") and audit_range.get("end_year"):
        return (
            int(audit_range["start_year"]),
            int(audit_range.get("start_month") or 1),
            int(audit_range["end_year"]),
            int(audit_range.get("end_month") or 12),
        )
    match = _RANGE_RE.search(str(grid.get("anchor_text") or ""))
    if not match:
        return None
    return tuple(int(v) for v in match.groups())  # type: ignore[return-value]


def _years_by_status_row_index(grid: dict[str, Any]) -> dict[int, int]:
    """Map status row_band index to calendar year from structure year cells."""
    out: dict[int, int] = {}
    for row in grid.get("cells") or []:
        if not isinstance(row, list):
            continue
        year_cell = next(
            (cell for cell in row if isinstance(cell, dict) and cell.get("role") == "year"),
            None,
        )
        status_cell = next(
            (cell for cell in row if isinstance(cell, dict) and cell.get("role") == "status"),
            None,
        )
        if year_cell is None or status_cell is None:
            continue
        text = str(year_cell.get("text") or "").strip()
        match = _YEAR_RE.match(text)
        if match is None:
            continue
        row_idx = int(status_cell.get("row_index") or year_cell.get("row_index") or 0)
        out[row_idx] = int(match.group(0))
    return out


def records_from_micro_grid_dict(
    grid: dict[str, Any],
    *,
    accept_exact_row_numeric_status: bool = False,
) -> list[dict[str, Any]]:
    """Project finance repayment records from a persisted micro_grid structure."""
    if not isinstance(grid, dict):
        return []
    page = int(grid.get("page") or 0)
    grid_id = str(grid.get("grid_id") or f"mg_p{page}_repayment_0")
    date_range = _date_range_from_grid(grid)
    if date_range is None:
        return []
    start_year, start_month, end_year, end_month = date_range
    valid_months = set(_months_between(start_year, start_month, end_year, end_month))
    zero_overdue_statuses = {
        str(value)
        for value in ((grid.get("audit") or {}).get("zero_overdue_statuses") or _ZERO_OVERDUE_STATUSES)
    }

    col_map: dict[int, int] = {}
    for band in grid.get("col_bands") or []:
        if not isinstance(band, dict):
            continue
        header = str(band.get("header") or "").strip()
        if header.isdigit():
            col_map[int(band.get("index") or 0)] = int(header)

    status_rows: list[list[dict[str, Any]]] = []
    amount_rows: list[list[dict[str, Any]]] = []
    for row in grid.get("cells") or []:
        if not isinstance(row, list) or not row:
            continue
        roles = {str(cell.get("role") or "") for cell in row if isinstance(cell, dict)}
        if "status" in roles:
            status_rows.append([cell for cell in row if isinstance(cell, dict)])
        elif "overdue_amount" in roles:
            amount_rows.append([cell for cell in row if isinstance(cell, dict)])

    amount_by_row_col: dict[tuple[int, int], str] = {}
    amount_cell_by_row_col: dict[tuple[int, int], dict[str, Any]] = {}
    amount_row_indices: list[int] = []
    for row in amount_rows:
        row_idx = next(
            (int(cell.get("row_index") or 0) for cell in row if str(cell.get("role") or "") == "overdue_amount"),
            0,
        )
        amount_row_indices.append(row_idx)
        for cell in row:
            text = str(cell.get("text") or "").strip()
            amount_cell_by_row_col[(row_idx, int(cell.get("col_index") or 0))] = cell
            if text:
                amount_by_row_col[(row_idx, int(cell.get("col_index") or 0))] = text

    years_by_row = _years_by_status_row_index(grid)
    status_cell_by_year_month: dict[tuple[int, int], dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    for status_row in status_rows:
        row_idx = next(
            (int(cell.get("row_index") or 0) for cell in status_row if str(cell.get("role") or "") == "status"),
            0,
        )
        row_year = years_by_row.get(row_idx)
        amount_row_idx = next((index for index in sorted(amount_row_indices) if index > row_idx), None)
        for cell in status_row:
            if str(cell.get("role") or "") != "status":
                continue
            col_idx = int(cell.get("col_index") or 0)
            month = col_map.get(col_idx)
            if row_year is not None and month and (row_year, month) in valid_months:
                status_cell_by_year_month[(row_year, month)] = cell
            status = str(cell.get("text") or "").strip()
            if not month or not status:
                continue
            year = row_year
            if year is None:
                year = next((y for y, m in valid_months if m == month), None)
            if year is None or (year, month) not in valid_months:
                continue
            amount_cell = (
                amount_cell_by_row_col.get((amount_row_idx, col_idx))
                if amount_row_idx is not None
                else None
            )
            amount = (
                amount_by_row_col.get((amount_row_idx, col_idx))
                if amount_row_idx is not None
                else None
            )
            if amount in (None, "") and status in zero_overdue_statuses:
                amount = "0"
            bbox = cell.get("bbox")
            recognition_audit = dict(cell.get("recognition_audit") or {})
            persisted_status_ref = recognition_audit.get("source_ref")
            logical_page = int(recognition_audit.get("logical_page") or page)
            source_refs: list[dict[str, Any]] = [
                dict(persisted_status_ref)
                if isinstance(persisted_status_ref, dict)
                else {
                    "page": logical_page,
                    "logical_page": logical_page,
                    "grid_id": grid_id,
                    "row": cell.get("row_index"),
                    "col": month,
                    "field_name": "status",
                    "geometry_scope": "cell",
                    **({"bbox": list(bbox)} if isinstance(bbox, list) and len(bbox) == 4 else {}),
                }
            ]
            amount_bbox = amount_cell.get("bbox") if isinstance(amount_cell, dict) else None
            amount_audit: dict[str, Any] = {}
            if isinstance(amount_cell, dict):
                amount_audit = dict(amount_cell.get("recognition_audit") or {})
                persisted_amount_ref = amount_audit.get("source_ref")
                amount_page = int(amount_audit.get("logical_page") or logical_page)
                source_refs.append(
                    dict(persisted_amount_ref)
                    if isinstance(persisted_amount_ref, dict)
                    else {
                        "page": amount_page,
                        "logical_page": amount_page,
                        "grid_id": grid_id,
                        "row": amount_cell.get("row_index"),
                        "col": month,
                        "field_name": "overdue_amount",
                        "geometry_scope": "cell",
                        **(
                            {"bbox": list(amount_bbox)}
                            if isinstance(amount_bbox, list) and len(amount_bbox) == 4
                            else {}
                        ),
                    }
                )
            record: dict[str, Any] = {
                "repayment_id": f"{grid_id}:{year:04d}-{month:02d}",
                "grid_id": grid_id,
                "year": year,
                "month": month,
                "status": status,
                "overdue_amount": amount,
                "source_cell_refs": source_refs,
                "confidence": float(cell.get("confidence") or 0.7),
            }
            recognition_source = str(cell.get("recognition_source") or "tokens")
            if recognition_source != "tokens":
                record["recognition_source"] = recognition_source
            if recognition_audit:
                record["audit"] = recognition_audit
            if amount_audit.get("reason") == "zero_status_conflicts_with_observed_nonzero_amount":
                record["extraction_status"] = "review"
                record["audit"] = {
                    **recognition_audit,
                    "reason": "zero_status_conflicts_with_observed_nonzero_amount",
                    "status": status,
                    "observed_amount": amount,
                }
            if cell.get("crop_ocr_text") is not None:
                record["raw_status"] = str(cell.get("crop_ocr_text") or "")
            if isinstance(bbox, list) and len(bbox) == 4:
                record["status_bbox"] = list(bbox)
            records.append(record)
    bbox_years: dict[tuple[float, ...], set[int]] = {}
    for record in records:
        bbox = record.get("status_bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            bbox_years.setdefault(tuple(float(value) for value in bbox), set()).add(int(record["year"]))
    reused_bboxes = {bbox for bbox, years in bbox_years.items() if len(years) > 1}
    for record in records:
        bbox = record.get("status_bbox")
        key = tuple(float(value) for value in bbox) if isinstance(bbox, list) and len(bbox) == 4 else None
        if key not in reused_bboxes:
            continue
        record["raw_status"] = record.get("status")
        record["status"] = "unknown"
        record["overdue_amount"] = None
        record["confidence"] = 0.0
        record["extraction_status"] = "review"
        record["audit"] = {"reason": "status_geometry_reused_across_years"}
    for record in records:
        status = str(record.get("status") or "")
        visually_confirmed = (
            str(record.get("recognition_source") or "") == "cell_crop_consensus"
            and int((record.get("audit") or {}).get("consensus_count") or 0) >= 2
        ) or (
            accept_exact_row_numeric_status
            and str(record.get("recognition_source") or "") == "canonical_row_sequence"
            and str((record.get("audit") or {}).get("alignment_status") or "") == "exact"
        )
        if status.isdigit() and not visually_confirmed:
            record["raw_status"] = status
            record["status"] = "unknown"
            record["overdue_amount"] = None
            record["confidence"] = 0.0
            record["extraction_status"] = "review"
            record["audit"] = {"reason": "numeric_status_requires_cell_ocr_confirmation"}
    existing_months = {(int(record.get("year") or 0), int(record.get("month") or 0)) for record in records}
    for year, month in sorted(valid_months - existing_months):
        empty_cell = status_cell_by_year_month.get((year, month))
        empty_audit = dict(empty_cell.get("recognition_audit") or {}) if isinstance(empty_cell, dict) else {}
        empty_ref = empty_audit.get("source_ref")
        records.append(
            {
                "repayment_id": f"{grid_id}:{year:04d}-{month:02d}",
                "grid_id": grid_id,
                "year": year,
                "month": month,
                "status": "unknown",
                "overdue_amount": None,
                "source": "repayment_grid_date_range_placeholder",
                "source_cell_refs": [
                    dict(empty_ref)
                    if isinstance(empty_ref, dict)
                    else {
                        "page": page,
                        "logical_page": page,
                        "grid_id": grid_id,
                        "row": 0,
                        "col": month,
                        "field_name": "status",
                        "geometry_scope": "logical_page",
                        "geometry_status": "unresolved",
                    }
                ],
                "confidence": 0.0,
                "extraction_status": "review",
            }
        )
    return records


def dedupe_repayment_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse only business-equivalent detector replays.

    Conflicting observations must reach the Candidate B relationship layer,
    where the selected value can be accompanied by a structured issue.
    """
    output: list[dict[str, Any]] = []
    equivalent_positions: dict[tuple[str, int, int, str, str | None], int] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        refs = record.get("source_cell_refs") or [{}]
        grid_id = str(record.get("grid_id") or (refs[0] or {}).get("grid_id") or "")
        status, amount = _repayment_business_signature(record)
        key = (
            grid_id,
            int(record.get("year") or 0),
            int(record.get("month") or 0),
            status,
            amount,
        )
        existing = equivalent_positions.get(key)
        if existing is None:
            equivalent_positions[key] = len(output)
            output.append(record)
            continue
        current = output[existing]
        selected = record if float(record.get("confidence") or 0.0) > float(current.get("confidence") or 0.0) else current
        selected = dict(selected)
        merged_refs = _merged_source_cell_refs(current, record)
        if merged_refs:
            selected["source_cell_refs"] = merged_refs
        output[existing] = selected
    return output
