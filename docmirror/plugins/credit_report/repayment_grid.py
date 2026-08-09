# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Credit-report repayment micro-grid reconstruction."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, replace
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
_MIN_MONTH_GRID_PAGE_COVERAGE = 0.40


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


def _explicit_amount_value(text: Any) -> str | None:
    """Return an amount only when the persisted cell contains an explicit number."""
    raw = str(text or "").strip()
    if not raw:
        return None
    compact = re.sub(r"[,，\s]", "", raw)
    if not re.fullmatch(r"\d+(?:\.\d+)?", compact):
        return None
    try:
        value = Decimal(compact)
    except InvalidOperation:
        return None
    normalized = format(value.normalize(), "f")
    return "0" if normalized in {"-0", "0"} else normalized


def _is_positive_amount(value: str | None) -> bool:
    if value is None:
        return False
    try:
        return Decimal(value) > 0
    except InvalidOperation:
        return False


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


def _static_candidate_b_zero_status_glyph_classification(
    template: Any,
) -> tuple[str, float, dict[str, Any]] | None:
    """Validate Candidate-B's OCR-confusable ``N``/``*``/``C`` cells.

    The shared extractor deliberately keeps its historical N/asterisk
    classifier.  Personal detailed reports additionally need a conservative C
    class because an OCR ``N`` in the final active month can otherwise be
    silently corroborated even though the printed glyph is ``C``.
    """

    try:
        import numpy as np

        reference_scores = _static_status_reference_scores(template, np=np)
        classification = _static_n_star_glyph_classification(template)
        if classification is not None:
            label = classification[0]
            # An N-shaped crop that remains close to the closed-account C class
            # is not decisive.  This is the exact adjacent-row/terminal-month
            # confusion Candidate B must report instead of silently accepting.
            if label != "N" or float(reference_scores.get("N") or 0.0) - float(
                reference_scores.get("C") or 0.0
            ) >= 0.06:
                return classification
        ranked = sorted(
            ((score, label) for label, score in reference_scores.items()),
            reverse=True,
        )
        if not ranked:
            return None
        winning_score, winning_label = ranked[0]
        competitor_score = ranked[1][0] if len(ranked) > 1 else 0.0
        margin = winning_score - competitor_score
        if winning_label != "C" or winning_score < 0.94 or margin < 0.045:
            return None
        confidence = min(0.98, 0.68 + winning_score * 0.3)
        return (
            "C",
            confidence,
            {
                "reference_winner": winning_label,
                "reference_similarity": round(winning_score, 4),
                "reference_margin": round(margin, 4),
                "classification_basis": "candidate_b_conservative_reference_bank",
            },
        )
    except Exception:
        return None


def _scan_specific_status_prototypes(
    seeds: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """Build conservative per-grid glyph medoids from independent static seeds."""

    try:
        import cv2
        import numpy as np

        prototypes: dict[str, dict[str, Any]] = {}
        for status, observations in seeds.items():
            distinct: dict[tuple[Any, ...], dict[str, Any]] = {}
            for observation in observations:
                source_key = tuple(observation.get("source_key") or ())
                template = observation.get("template")
                if not source_key or template is None:
                    continue
                vector = _static_status_feature_vector(template, np=np, cv2=cv2)
                if vector is not None:
                    distinct[source_key] = {**observation, "vector": vector}
            values = list(distinct.values())
            if len(values) < 2:
                continue
            vectors = np.stack([item["vector"] for item in values])
            similarities = vectors @ vectors.T
            # Two independently classified glyphs must also agree with each
            # other; the synthetic-font decision alone is not a scan-specific
            # template bank.
            off_diagonal = similarities[~np.eye(len(values), dtype=bool)]
            if not len(off_diagonal) or float(off_diagonal.min()) < 0.88:
                continue
            medoid_index = int(np.argmax(similarities.mean(axis=1)))
            prototypes[status] = {
                "vector": vectors[medoid_index],
                "seed_count": len(values),
                "minimum_seed_similarity": float(off_diagonal.min()),
            }
        return prototypes
    except Exception:
        return {}


def _scan_specific_status_match(
    template: Any,
    *,
    expected_status: str,
    prototypes: dict[str, dict[str, Any]],
) -> tuple[float, float] | None:
    """Return same-symbol similarity/margin only for decisive scan evidence."""

    try:
        import cv2
        import numpy as np

        expected = prototypes.get(expected_status)
        vector = _static_status_feature_vector(template, np=np, cv2=cv2)
        if expected is None or vector is None:
            return None
        expected_score = float(vector @ expected["vector"])
        competitor_scores = [
            float(vector @ prototype["vector"])
            for status, prototype in prototypes.items()
            if status != expected_status
        ]
        competitor = max(competitor_scores, default=0.0)
        margin = expected_score - competitor
        if expected_score < 0.90 or (competitor_scores and margin < 0.04):
            return None
        return expected_score, margin
    except Exception:
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
    """Build month cells from header centres when word OCR preserved them.

    A one-shot page OCR result is word-level evidence.  Treating the first
    post-anchor word as a complete month header can otherwise divide one glyph
    box into twelve microscopic cells.  Canonical month centres let us recover
    the real lattice without another OCR pass.
    """

    raw_centres = header_line.get("month_centers")
    if isinstance(raw_centres, (list, tuple)) and len(raw_centres) == n_months:
        try:
            centres = [float(value) for value in raw_centres]
        except (TypeError, ValueError):
            centres = []
        if centres and all(right > left for left, right in zip(centres, centres[1:])):
            boundaries = [centres[0] - (centres[1] - centres[0]) / 2.0]
            boundaries.extend((left + right) / 2.0 for left, right in zip(centres, centres[1:]))
            boundaries.append(centres[-1] + (centres[-1] - centres[-2]) / 2.0)
            y0, y1 = float(header_line["bbox"][1]), float(header_line["bbox"][3])
            geometry_status = (
                "exact"
                if header_line.get("month_header_geometry") == "word_center_sequence_exact"
                else "estimated"
            )
            return [
                {
                    "index": month,
                    "header": str(month),
                    "role": "month",
                    "bbox": [boundaries[month - 1], y0, boundaries[month], y1],
                    "geometry_status": geometry_status,
                    "geometry_source": "month_header_word_centers",
                }
                for month in range(1, n_months + 1)
            ]
    return cast(
        list[dict[str, Any]],
        equal_col_bands(header_line["bbox"], count=n_months, start_index=1, role="month"),
    )


def _month_band_plausibility(
    month_cols: list[dict[str, Any]],
    *,
    page_width: float,
) -> tuple[bool, dict[str, Any]]:
    """Validate that twelve month cells occupy a plausible page-wide lattice."""

    if len(month_cols) != 12 or page_width <= 0:
        return False, {"reason": "month_band_count_or_page_width_invalid"}
    try:
        boxes = [tuple(float(value) for value in col["bbox"]) for col in month_cols]
    except (KeyError, TypeError, ValueError):
        return False, {"reason": "month_band_bbox_invalid"}
    widths = [box[2] - box[0] for box in boxes]
    heights = [box[3] - box[1] for box in boxes]
    start, end = boxes[0][0], boxes[-1][2]
    total_width = end - start
    coverage = total_width / page_width
    sorted_widths = sorted(widths)
    median_width = sorted_widths[len(sorted_widths) // 2]
    sorted_heights = sorted(height for height in heights if height > 0)
    median_height = sorted_heights[len(sorted_heights) // 2] if sorted_heights else 0.0
    cell_aspect = median_width / max(median_height, 1e-6)
    uniformity = min(widths) / max(max(widths), 1e-6)
    metrics = {
        "coverage": round(coverage, 4),
        "total_width": round(total_width, 4),
        "median_cell_width": round(median_width, 4),
        "median_header_height": round(median_height, 4),
        "cell_aspect": round(cell_aspect, 4),
        "width_uniformity": round(uniformity, 4),
    }
    if not all(width > 0 for width in widths):
        return False, {**metrics, "reason": "nonpositive_month_band"}
    if any(boxes[index][0] < boxes[index - 1][2] - 1e-4 for index in range(1, 12)):
        return False, {**metrics, "reason": "overlapping_month_bands"}
    if not _MIN_MONTH_GRID_PAGE_COVERAGE <= coverage <= 0.95:
        return False, {**metrics, "reason": "implausible_page_coverage"}
    if median_width < max(2.5, page_width * 0.01) or not 0.45 <= cell_aspect <= 30.0:
        return False, {**metrics, "reason": "implausible_cell_aspect"}
    if uniformity < 0.72:
        return False, {**metrics, "reason": "nonuniform_month_pitch"}
    if start < -page_width * 0.03 or end > page_width * 1.03:
        return False, {**metrics, "reason": "month_lattice_outside_page"}
    return True, metrics


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
    """Align plausible month bands to table rules; never crop a collapsed box."""
    if page_width is not None:
        plausible, geometry = _month_band_plausibility(month_cols, page_width=float(page_width))
        if not plausible:
            return [], {
                "source": "rejected_month_geometry",
                "usable": False,
                "offset": 0.0,
                **geometry,
            }
    else:
        geometry = {}
    if (
        page_image is None
        or page_width is None
        or page_height is None
        or not month_cols
        or getattr(page_image, "size", 0) == 0
    ):
        return month_cols, {"source": "header_geometry", "usable": True, "offset": 0.0, **geometry}
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
            return month_cols, {
                "source": "header_geometry",
                "usable": True,
                "offset": 0.0,
                "reason": "vertical_projection_band_too_short",
                **geometry,
            }
        ink = (gray[py0:py1] < 115).astype(np.float32)
        projection = ink.mean(axis=0)
        projection = np.convolve(projection, np.ones(5, dtype=np.float32) / 5.0, mode="same")
        # Month glyph boxes are narrower than the table: find both outside
        # rules, allowing the column pitch to expand instead of applying a
        # single offset that becomes wrong near month 12.
        start_offsets = np.linspace(-max(0.2, max_left_shift_months) * step, 0.45 * step, 129)
        end_offsets = np.linspace(-1.10 * step, 0.55 * step, 113)
        best_score = -1.0
        best_start, best_end = start, end
        best_strengths: Any = None
        for start_offset in start_offsets:
            candidate_start = start + float(start_offset)
            for end_offset in end_offsets:
                candidate_end = end + float(end_offset)
                if candidate_end <= candidate_start:
                    continue
                positions = np.linspace(candidate_start * sx, candidate_end * sx, 13)
                indices = np.clip(np.rint(positions).astype(int), 0, image_width - 1)
                strengths = projection[indices]
                score = float(strengths.sum())
                if score > best_score:
                    best_score = score
                    best_start, best_end = candidate_start, candidate_end
                    best_strengths = strengths
        best_offset = best_start - start
        # A few glyph strokes are not a table lattice.  Require sustained
        # evidence at most of the thirteen equally spaced vertical rules.
        baseline_per_rule = float(np.median(projection))
        rule_floor = max(0.04, baseline_per_rule * 1.5)
        rule_hits = int((best_strengths >= rule_floor).sum()) if best_strengths is not None else 0
        baseline = baseline_per_rule * 13.0
        if best_score < max(0.5, baseline * 1.05) or rule_hits < 10:
            return month_cols, {
                "source": "header_geometry",
                "usable": True,
                "offset": 0.0,
                "reason": "vertical_rule_lattice_not_decisive",
                "rule_hits": rule_hits,
                **geometry,
            }
        refined = equal_col_bands(
            (best_start, float(month_cols[0]["bbox"][1]), best_end, float(month_cols[0]["bbox"][3])),
            count=12,
            start_index=1,
            role="month",
        )
        refined_plausible, refined_geometry = _month_band_plausibility(
            refined,
            page_width=float(page_width),
        )
        if not refined_plausible:
            return month_cols, {
                "source": "header_geometry",
                "usable": True,
                "offset": 0.0,
                "reason": "refined_rule_lattice_rejected",
                **geometry,
            }
        return refined, {
            "source": "vertical_rule_projection",
            "usable": True,
            "offset": round(best_offset, 4),
            "right_offset": round(best_end - end, 4),
            "score": round(best_score, 4),
            "rule_hits": rule_hits,
            **refined_geometry,
        }
    except Exception as exc:
        return month_cols, {
            "source": "header_geometry",
            "usable": True,
            "offset": 0.0,
            "reason": "vertical_rule_projection_failed",
            "error_type": type(exc).__name__,
            **geometry,
        }


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


def _explicit_amount_line_after_year(
    lines: list[dict[str, Any]],
    year_line: dict[str, Any],
    *,
    x_min: float,
    x_max: float,
) -> dict[str, Any] | None:
    """Find the unique numeric amount line immediately beneath a year label."""
    _x0, y0, _x1, y1 = year_line["bbox"]
    candidates: list[dict[str, Any]] = []
    for line in lines:
        lx0, ly0, lx1, _ly1 = line["bbox"]
        if line is year_line or ly0 <= y0 or ly0 - y1 > 24.0:
            continue
        if lx1 < x_min or lx0 > x_max:
            continue
        compact = re.sub(r"[,，.\s]", "", str(line.get("text") or ""))
        if not compact or not compact.isdigit() or _YEAR_RE.match(str(line.get("text") or "").strip()):
            continue
        candidates.append(line)
    return candidates[0] if len(candidates) == 1 else None


def _candidate_b_status_chars(text: Any, status_charset: set[str]) -> list[str]:
    normalized = str(text or "").translate(str.maketrans({"☆": "*", "★": "*", "＊": "*"}))
    compact = re.sub(r"[\s.,，。·:：;；|]+", "", normalized)
    if not compact or any(character not in status_charset and character != "#" for character in compact):
        return []
    return list(compact)


def _candidate_b_status_row(
    lines: list[dict[str, Any]],
    year_line: dict[str, Any],
    *,
    month_cols: list[dict[str, Any]],
    status_charset: set[str],
    page: int,
    excluded_line_indices: set[int],
) -> dict[str, Any] | None:
    """Assemble one canonical status y-band without shifting adjacent cells.

    Page OCR may expose a printed status row as one merged word or as one word
    per month.  Candidate B clusters only the canonical band immediately above
    the year/amount row and preserves every word's horizontal geometry.
    """

    year_y0 = float(year_line["bbox"][1])
    year_y1 = float(year_line["bbox"][3])
    year_center = (year_y0 + year_y1) / 2.0
    year_page = int(year_line.get("source_logical_page") or page)
    month_x0 = min(float(col["bbox"][0]) for col in month_cols)
    month_x1 = max(float(col["bbox"][2]) for col in month_cols)
    candidates: list[dict[str, Any]] = []
    for line in lines:
        line_index = int(line.get("idx") if line.get("idx") is not None else -1)
        if (
            line is year_line
            or line_index in excluded_line_indices
            or int(line.get("source_logical_page") or page) != year_page
        ):
            continue
        lx0, ly0, lx1, ly1 = (float(value) for value in line["bbox"])
        center_y = (ly0 + ly1) / 2.0
        if lx1 <= month_x0 or lx0 >= month_x1 or center_y >= year_center:
            continue
        # Month headers are normally the next OCR band above the status row.
        # A tight canonical offset prevents an isolated header digit from being
        # promoted when the actual status band was missed.
        if not (-2.0 <= year_y0 - center_y <= 32.0):
            continue
        chars = _candidate_b_status_chars(line.get("text"), status_charset)
        if chars:
            candidates.append({**line, "candidate_b_status_chars": chars})
    if not candidates:
        return None

    clusters: list[list[dict[str, Any]]] = []
    for line in sorted(candidates, key=lambda item: (float(item["bbox"][1]), float(item["bbox"][0]))):
        center = (float(line["bbox"][1]) + float(line["bbox"][3])) / 2.0
        height = max(1.0, float(line["bbox"][3]) - float(line["bbox"][1]))
        selected: list[dict[str, Any]] | None = None
        for cluster in clusters:
            cluster_center = sum(
                (float(item["bbox"][1]) + float(item["bbox"][3])) / 2.0
                for item in cluster
            ) / len(cluster)
            cluster_height = max(
                max(1.0, float(item["bbox"][3]) - float(item["bbox"][1]))
                for item in cluster
            )
            if abs(center - cluster_center) <= max(3.0, 0.55 * max(height, cluster_height)):
                selected = cluster
                break
        if selected is None:
            selected = []
            clusters.append(selected)
        selected.append(line)

    # The band nearest the year row is canonical.  More distant bands belong
    # to another year or the month header; they are never concatenated.
    row_lines = max(
        clusters,
        key=lambda cluster: sum(
            (float(item["bbox"][1]) + float(item["bbox"][3])) / 2.0
            for item in cluster
        )
        / len(cluster),
    )
    row_lines = sorted(row_lines, key=lambda item: float(item["bbox"][0]))
    status_tokens: list[OCRToken] = []
    for line in row_lines:
        chars = list(line.get("candidate_b_status_chars") or ())
        x0, y0, x1, y1 = (float(value) for value in line["bbox"])
        step = max(x1 - x0, 1.0) / len(chars)
        for index, character in enumerate(chars):
            status_tokens.append(
                OCRToken(
                    token_id=f"ocr_p{page}_repay_status_cell_{int(line['idx'])}_{index}",
                    text=character,
                    bbox=(x0 + step * index, y0, x0 + step * (index + 1), y1),
                    confidence=float(line.get("confidence") or 0.0),
                    page=page,
                    source="ocr_status_cell_observation",
                    source_token_id=f"line_{int(line['idx'])}",
                )
            )
    status_tokens.sort(key=lambda token: token.center[0])
    return {
        "idx": min(int(line["idx"]) for line in row_lines),
        "text": "".join(token.text for token in status_tokens),
        "bbox": [
            min(float(line["bbox"][0]) for line in row_lines),
            min(float(line["bbox"][1]) for line in row_lines),
            max(float(line["bbox"][2]) for line in row_lines),
            max(float(line["bbox"][3]) for line in row_lines),
        ],
        "confidence": min(float(line.get("confidence") or 0.0) for line in row_lines),
        "source_logical_page": year_page,
        "candidate_b_status_tokens": status_tokens,
        "status_source_line_indices": [int(line["idx"]) for line in row_lines],
    }


def _numeric_amount_groups(text: Any) -> list[str]:
    """Return explicit numeric groups only; OCR prose is never an amount row."""

    value = str(text or "").strip()
    if not value or re.fullmatch(r"[0-9.,，\s]+", value) is None:
        return []
    return re.findall(r"\d+(?:[.,，]\d+)?", value)


def _amount_sequence_tokens(
    text: Any,
    *,
    months: list[int],
    cols_by_month: dict[int, dict[str, Any]],
    y0: float,
    y1: float,
    page: int,
    prefix: str,
    confidence: float,
    source_line_index: int,
) -> list[OCRToken] | None:
    """Bind one merged numeric sequence only when its cardinality is exact."""

    groups = _numeric_amount_groups(text)
    if not groups or not months:
        return None
    values: list[str]
    if len(groups) == len(months):
        values = groups
    else:
        digits = [character for character in str(text or "") if character.isdigit()]
        if len(digits) != len(months):
            return None
        values = digits
    tokens: list[OCRToken] = []
    for month, value in zip(months, values, strict=True):
        col = cols_by_month.get(month)
        if col is None:
            return None
        x0, _cy0, x1, _cy1 = (float(item) for item in col["bbox"])
        tokens.append(
            OCRToken(
                token_id=f"{prefix}_{source_line_index}_{month}",
                text=value,
                bbox=(x0, y0, x1, y1),
                confidence=confidence,
                page=page,
                source="ocr_amount_sequence_split",
                source_token_id=f"line_{source_line_index}",
            )
        )
    return tokens


def _candidate_b_amount_row_pair(
    lines: list[dict[str, Any]],
    year_line: dict[str, Any],
    *,
    month_cols: list[dict[str, Any]],
    active_months: list[int],
    page: int,
    excluded_line_indices: set[int],
) -> dict[str, Any]:
    """Resolve a unique immediate amount row from canonical static geometry.

    Candidate B accepts the year-line remainder or exactly one aligned/adjacent
    numeric y-band.  Every OCR word remains tied to its observed cell geometry;
    a second row, a non-immediate row, or two words in one month is unresolved.
    """

    cols_by_month = {int(col["header"]): col for col in month_cols}
    year_match = _YEAR_RE.match(str(year_line.get("text") or "").strip())
    year_remainder = (
        _YEAR_RE.sub("", str(year_line.get("text") or "").strip(), count=1)
        if year_match is not None
        else ""
    )
    year_y0 = float(year_line["bbox"][1])
    year_y1 = float(year_line["bbox"][3])
    if _numeric_amount_groups(year_remainder):
        remainder_tokens = _amount_sequence_tokens(
            year_remainder,
            months=active_months,
            cols_by_month=cols_by_month,
            y0=year_y0,
            y1=year_y1,
            page=page,
            prefix=f"ocr_p{page}_repay_amount_remainder",
            confidence=float(year_line.get("confidence") or 0.0),
            source_line_index=int(year_line["idx"]),
        )
        if remainder_tokens is None:
            return {
                "status": "ambiguous_year_remainder",
                "row_relation": "year_line_remainder",
                "source_line_indices": [int(year_line["idx"])],
                "observed_texts": [year_remainder],
                "tokens": [],
                "cell_status_by_month": {
                    str(month): "ambiguous_sequence_cardinality" for month in active_months
                },
            }
        active_cols = [cols_by_month[month] for month in active_months if month in cols_by_month]
        amount_line = {
            **year_line,
            "text": year_remainder,
            "bbox": [
                min(float(col["bbox"][0]) for col in active_cols),
                year_y0,
                max(float(col["bbox"][2]) for col in active_cols),
                year_y1,
            ],
            "amount_source_line_indices": [int(year_line["idx"])],
        }
        return {
            "status": "exact",
            "row_relation": "year_line_remainder",
            "line": amount_line,
            "tokens": remainder_tokens,
            "source_line_indices": [int(year_line["idx"])],
            "observed_texts": [year_remainder],
            "cell_status_by_month": {str(month): "exact" for month in active_months},
        }

    month_x0 = min(float(col["bbox"][0]) for col in month_cols)
    month_x1 = max(float(col["bbox"][2]) for col in month_cols)
    year_center = (year_y0 + year_y1) / 2.0
    year_logical_page = int(year_line.get("source_logical_page") or page)
    immediate: list[dict[str, Any]] = []
    non_immediate: list[dict[str, Any]] = []
    for line in lines:
        if line is year_line or int(line.get("idx") or -1) in excluded_line_indices:
            continue
        if int(line.get("source_logical_page") or page) != year_logical_page:
            continue
        lx0, ly0, lx1, ly1 = (float(value) for value in line["bbox"])
        if lx1 <= month_x0 or lx0 >= month_x1 or not _numeric_amount_groups(line.get("text")):
            continue
        center_y = (ly0 + ly1) / 2.0
        aligned_or_below = center_y >= year_center - 2.0 and ly0 >= year_y0 - 2.0
        gap = ly0 - year_y1
        if aligned_or_below and gap <= 24.0:
            immediate.append(line)
        elif 24.0 < gap <= 55.0:
            non_immediate.append(line)

    clusters: list[list[dict[str, Any]]] = []
    for line in sorted(immediate, key=lambda item: (float(item["bbox"][1]), float(item["bbox"][0]))):
        center = (float(line["bbox"][1]) + float(line["bbox"][3])) / 2.0
        height = max(1.0, float(line["bbox"][3]) - float(line["bbox"][1]))
        selected: list[dict[str, Any]] | None = None
        for cluster in clusters:
            cluster_centers = [
                (float(item["bbox"][1]) + float(item["bbox"][3])) / 2.0 for item in cluster
            ]
            cluster_heights = [
                max(1.0, float(item["bbox"][3]) - float(item["bbox"][1])) for item in cluster
            ]
            tolerance = max(3.0, 0.6 * max(height, sum(cluster_heights) / len(cluster_heights)))
            if abs(center - sum(cluster_centers) / len(cluster_centers)) <= tolerance:
                selected = cluster
                break
        if selected is None:
            selected = []
            clusters.append(selected)
        selected.append(line)

    if len(clusters) != 1:
        status = "ambiguous_immediate_rows" if clusters else (
            "non_immediate_amount_row" if non_immediate else "missing_amount_row"
        )
        candidates = [item for cluster in clusters for item in cluster] or non_immediate
        return {
            "status": status,
            "row_relation": "unresolved",
            "source_line_indices": [int(item["idx"]) for item in candidates],
            "observed_texts": [str(item.get("text") or "") for item in candidates],
            "tokens": [],
            "cell_status_by_month": {str(month): status for month in active_months},
        }

    row_lines = sorted(clusters[0], key=lambda item: float(item["bbox"][0]))
    row_y0 = min(float(line["bbox"][1]) for line in row_lines)
    row_y1 = max(float(line["bbox"][3]) for line in row_lines)
    row_line = {
        "idx": min(int(line["idx"]) for line in row_lines),
        "text": " ".join(str(line.get("text") or "") for line in row_lines),
        "bbox": [
            min(float(line["bbox"][0]) for line in row_lines),
            row_y0,
            max(float(line["bbox"][2]) for line in row_lines),
            row_y1,
        ],
        "confidence": min(float(line.get("confidence") or 0.0) for line in row_lines),
        "source_logical_page": year_logical_page,
        "amount_source_line_indices": [int(line["idx"]) for line in row_lines],
    }
    tokens: list[OCRToken] = []
    ambiguous_months: set[int] = set()
    month_widths = [
        max(1.0, float(col["bbox"][2]) - float(col["bbox"][0]))
        for col in month_cols
        if int(col["header"]) in active_months
    ]
    median_month_width = sorted(month_widths)[len(month_widths) // 2]
    for line in row_lines:
        lx0, _ly0, lx1, _ly1 = (float(value) for value in line["bbox"])
        line_width = max(1.0, lx1 - lx0)
        center_x = (lx0 + lx1) / 2.0
        if line_width <= median_month_width * 1.35:
            nearest = min(
                (int(col["header"]) for col in month_cols),
                key=lambda month: abs(
                    center_x
                    - (
                        float(cols_by_month[month]["bbox"][0])
                        + float(cols_by_month[month]["bbox"][2])
                    )
                    / 2.0
                ),
                default=None,
            )
            nearest_bbox = cols_by_month[nearest]["bbox"] if nearest is not None else None
            spanned = (
                [nearest]
                if nearest is not None
                and nearest in active_months
                and nearest_bbox is not None
                and float(nearest_bbox[0]) <= center_x <= float(nearest_bbox[2])
                else []
            )
        else:
            spanned = [
                int(col["header"])
                for col in month_cols
                if int(col["header"]) in active_months
                and max(
                    0.0,
                    min(lx1, float(col["bbox"][2]))
                    - max(lx0, float(col["bbox"][0])),
                )
                / min(
                    line_width,
                    max(1.0, float(col["bbox"][2]) - float(col["bbox"][0])),
                )
                >= 0.2
            ]
        line_tokens = _amount_sequence_tokens(
            line.get("text"),
            months=spanned,
            cols_by_month=cols_by_month,
            y0=row_y0,
            y1=row_y1,
            page=page,
            prefix=f"ocr_p{page}_repay_amount_cell",
            confidence=float(line.get("confidence") or 0.0),
            source_line_index=int(line["idx"]),
        )
        if line_tokens is None:
            ambiguous_months.update(spanned)
        else:
            tokens.extend(line_tokens)

    tokens_by_month: dict[int, list[OCRToken]] = {}
    for token in tokens:
        month = min(
            active_months,
            key=lambda value: abs(
                token.center[0]
                - (
                    float(cols_by_month[value]["bbox"][0])
                    + float(cols_by_month[value]["bbox"][2])
                )
                / 2.0
            ),
        )
        tokens_by_month.setdefault(month, []).append(token)
    for month, month_tokens in tokens_by_month.items():
        source_ids = {token.source_token_id or token.token_id for token in month_tokens}
        if len(source_ids) > 1:
            ambiguous_months.add(month)
    cell_status_by_month = {
        str(month): (
            "duplicate_or_ambiguous_cell"
            if month in ambiguous_months
            else "exact"
            if tokens_by_month.get(month)
            else "blank_amount_cell"
        )
        for month in active_months
    }
    return {
        "status": "exact",
        "row_relation": "aligned_or_immediate_after_year",
        "line": row_line,
        "tokens": tokens,
        "source_line_indices": [int(line["idx"]) for line in row_lines],
        "observed_texts": [str(line.get("text") or "") for line in row_lines],
        "cell_status_by_month": cell_status_by_month,
    }


def _month_observations(line: dict[str, Any]) -> list[tuple[int, float]]:
    """Return month labels and approximate glyph centres from one OCR word."""

    text = str(line.get("text") or "")
    if re.search(r"20\d{2}", text):
        return []
    matches = list(re.finditer(r"(?<!\d)(?:1[0-2]|[1-9])(?!\d)", text))
    if not matches:
        return []
    x0, _y0, x1, _y1 = (float(value) for value in line["bbox"])
    width = x1 - x0
    if len(matches) == 1:
        return [(int(matches[0].group(0)), (x0 + x1) / 2.0)]
    return [
        (
            int(match.group(0)),
            x0 + width * ((index + 0.5) / len(matches)),
        )
        for index, match in enumerate(matches)
    ]


def _word_center_month_header(
    lines: list[dict[str, Any]],
    anchor: dict[str, Any],
    *,
    page_width: float | None,
) -> dict[str, Any] | None:
    """Recover a canonical 1..12 header from aligned word-level OCR boxes."""

    _ax0, ay0, _ax1, ay1 = anchor["bbox"]
    observations: list[tuple[dict[str, Any], list[tuple[int, float]]]] = []
    for line in lines:
        if line is anchor:
            continue
        _lx0, ly0, _lx1, ly1 = line["bbox"]
        if ly1 < ay0 or ly0 > ay1 + 40.0:
            continue
        month_values = _month_observations(line)
        if month_values:
            observations.append((line, month_values))
    if not observations:
        return None

    clusters: list[list[tuple[dict[str, Any], list[tuple[int, float]]]]] = []
    for line, values in sorted(
        observations,
        key=lambda item: (
            (float(item[0]["bbox"][1]) + float(item[0]["bbox"][3])) / 2.0,
            float(item[0]["bbox"][0]),
        ),
    ):
        y0, y1 = float(line["bbox"][1]), float(line["bbox"][3])
        center = (y0 + y1) / 2.0
        height = y1 - y0
        selected: list[tuple[dict[str, Any], list[tuple[int, float]]]] | None = None
        for cluster in clusters:
            cluster_centers = [
                (float(item[0]["bbox"][1]) + float(item[0]["bbox"][3])) / 2.0
                for item in cluster
            ]
            cluster_heights = [float(item[0]["bbox"][3]) - float(item[0]["bbox"][1]) for item in cluster]
            tolerance = max(3.0, 0.65 * max(height, sum(cluster_heights) / len(cluster_heights)))
            if abs(center - sum(cluster_centers) / len(cluster_centers)) <= tolerance:
                selected = cluster
                break
        if selected is None:
            selected = []
            clusters.append(selected)
        selected.append((line, values))

    candidates: list[tuple[tuple[int, float], dict[str, Any]]] = []
    for cluster in clusters:
        by_month: dict[int, list[float]] = {}
        for _line, values in cluster:
            for month, center in values:
                by_month.setdefault(month, []).append(center)
        if len(by_month) < 8 or min(by_month) > 2 or max(by_month) < 11:
            continue
        centres_by_month = {
            month: sorted(values)[len(values) // 2]
            for month, values in by_month.items()
            if month in range(1, 13)
        }
        ordered = sorted(centres_by_month.items())
        if any(right[1] <= left[1] for left, right in zip(ordered, ordered[1:])):
            continue
        mean_month = sum(month for month, _center in ordered) / len(ordered)
        mean_x = sum(center for _month, center in ordered) / len(ordered)
        denominator = sum((month - mean_month) ** 2 for month, _center in ordered)
        if denominator <= 0:
            continue
        pitch = sum((month - mean_month) * (center - mean_x) for month, center in ordered) / denominator
        if pitch <= 0:
            continue
        intercept = mean_x - pitch * (mean_month - 1.0)
        fitted_centres = [intercept + pitch * index for index in range(12)]
        residuals = [abs(center - fitted_centres[month - 1]) for month, center in ordered]
        max_residual = max(residuals, default=0.0)
        if max_residual > pitch * 0.38:
            continue
        left = fitted_centres[0] - pitch / 2.0
        right = fitted_centres[-1] + pitch / 2.0
        if page_width and not _MIN_MONTH_GRID_PAGE_COVERAGE <= (right - left) / float(page_width) <= 0.95:
            continue
        cluster_lines = [line for line, _values in cluster]
        header = {
            "idx": min(int(line.get("idx") or 0) for line in cluster_lines),
            "text": "1 2 3 4 5 6 7 8 9 10 11 12",
            "bbox": (
                left,
                min(float(line["bbox"][1]) for line in cluster_lines),
                right,
                max(float(line["bbox"][3]) for line in cluster_lines),
            ),
            "confidence": min(float(line.get("confidence") or 0.0) for line in cluster_lines),
            "month_centers": fitted_centres,
            "month_header_geometry": (
                "word_center_sequence_exact" if len(centres_by_month) == 12 else "word_center_sequence_estimated"
            ),
            "month_header_observed_count": len(centres_by_month),
            "month_header_max_residual": round(max_residual, 4),
            "month_header_source_line_indices": [
                int(line["idx"]) for line in cluster_lines
            ],
        }
        logical_pages = {
            int(line["source_logical_page"])
            for line in cluster_lines
            if line.get("source_logical_page")
        }
        if len(logical_pages) == 1:
            header["source_logical_page"] = next(iter(logical_pages))
        candidates.append(((len(centres_by_month), -max_residual), header))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def _find_month_header(
    lines: list[dict[str, Any]],
    anchor: dict[str, Any],
    *,
    page_width: float | None = None,
) -> dict[str, Any] | None:
    _ax0, ay0, _ax1, ay1 = anchor["bbox"]
    candidates: list[tuple[int, dict[str, Any]]] = []
    for line in lines:
        _lx0, ly0, _lx1, ly1 = line["bbox"]
        if ly1 < ay0 or ly0 > ay1 + 40.0:
            continue
        months = [int(value) for value in re.findall(r"(?<!\d)(?:1[0-2]|[1-9])(?!\d)", line["text"])]
        score = len(set(months) & set(range(1, 13)))
        width = float(line["bbox"][2]) - float(line["bbox"][0])
        coverage = width / float(page_width) if page_width else 1.0
        if score >= 8 and coverage >= _MIN_MONTH_GRID_PAGE_COVERAGE:
            candidates.append((score, {**line, "month_header_geometry": "merged_line_exact"}))
    if candidates:
        return max(candidates, key=lambda item: (item[0], -float(item[1]["bbox"][1])))[1]
    return _word_center_month_header(lines, anchor, page_width=page_width)


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
    enable_candidate_b_amount_pairing: bool = False,
    grid_index: int = 0,
) -> RepaymentExtraction:
    """Reconstruct a credit repayment grid from OCR line-level geometry."""
    line_items = _line_items(lines)
    status_charset = _STATUS_CHARS | {str(item) for item in extra_status_chars if str(item)}
    zero_overdue_statuses = _ZERO_OVERDUE_STATUSES | ({"A"} if "A" in status_charset else set())
    static_sensitive_statuses = {"N", "*"} | (
        {"C"} if enable_candidate_b_amount_pairing else set()
    )
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
    exact_header_line = _find_month_header(line_items, anchor, page_width=page_width)
    header_line = exact_header_line or _line_after(
        line_items, ay1, x_min=ax0 - 220, x_max=ax1 + 260, max_gap=35.0
    )
    header_alignment_exact = bool(
        exact_header_line is not None
        and exact_header_line.get("month_header_geometry")
        in {"merged_line_exact", "word_center_sequence_exact"}
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
    if visual_month_cols:
        month_cols = visual_month_cols
    if visual_geometry_audit.get("source") == "vertical_rule_projection":
        header_alignment_exact = True
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
    header_source_indices = {
        int(value) for value in header_line.get("month_header_source_line_indices") or ()
    }
    header_source_indices.add(int(header_line["idx"]))
    status_lines_by_year_index: dict[int, dict[str, Any]] = {}
    for candidate_year_line in years:
        candidate_status_line = (
            _candidate_b_status_row(
                line_items,
                candidate_year_line,
                month_cols=month_cols,
                status_charset=status_charset,
                page=page,
                excluded_line_indices=header_source_indices,
            )
            if enable_candidate_b_amount_pairing
            else _line_before(
                line_items,
                candidate_year_line["bbox"][1],
                x_min=grid_x0 - 10,
                x_max=grid_x1 + 10,
                max_gap=55.0,
            )
        )
        if (
            candidate_status_line is None
            or candidate_status_line is header_line
            or "还款记录" in candidate_status_line["text"]
        ) and not enable_candidate_b_amount_pairing:
            candidate_status_line = _line_after(
                line_items,
                candidate_year_line["bbox"][1],
                x_min=grid_x0 - 10,
                x_max=grid_x1 + 10,
                max_gap=55.0,
            )
        if candidate_status_line is not None:
            status_lines_by_year_index[int(candidate_year_line["idx"])] = candidate_status_line
    excluded_amount_line_indices = set()
    for status_line in status_lines_by_year_index.values():
        source_indices = status_line.get("status_source_line_indices")
        if isinstance(source_indices, list):
            excluded_amount_line_indices.update(int(value) for value in source_indices)
        else:
            excluded_amount_line_indices.add(int(status_line["idx"]))
    amount_pairing_by_year: dict[str, dict[str, Any]] = {}
    crop_ocr_attempts = 0
    crop_ocr_hits = 0
    static_status_attempts = 0
    static_status_resolved = 0
    static_status_corrections = 0
    static_status_unresolved = 0
    static_status_unavailable = 0
    static_template_consensus_resolved = 0
    status_templates: dict[str, list[Any]] = {}
    static_status_seeds: dict[str, list[dict[str, Any]]] = {}
    static_status_contradictions: set[str] = set()
    pending_static_statuses: list[dict[str, Any]] = []
    static_status_geometry: list[tuple[tuple[float, ...], int]] = []
    continuation_visual_cols_cache: dict[
        tuple[int, int, float, float, tuple[int, ...]],
        tuple[list[dict[str, Any]], dict[str, Any]],
    ] = {}

    for year_line in years:
        year_match = _YEAR_RE.match(year_line["text"].strip())
        if year_match is None:
            continue
        year = int(year_match.group(0))
        status_line = status_lines_by_year_index.get(int(year_line["idx"]))
        if (
            status_line is None
            or status_line is header_line
            or "还款记录" in status_line["text"]
        ) and not enable_candidate_b_amount_pairing:
            status_line = _line_after(
                line_items, year_line["bbox"][1], x_min=grid_x0 - 10, x_max=grid_x1 + 10, max_gap=55.0
            )
        if status_line is None:
            continue
        active_months = [month for yy, month in sorted(record_months) if yy == year]
        amount_pairing: dict[str, Any] | None = None
        amount_row_tokens: list[OCRToken] | None = None
        if enable_candidate_b_amount_pairing:
            amount_pairing = _candidate_b_amount_row_pair(
                line_items,
                year_line,
                month_cols=month_cols,
                active_months=active_months,
                page=page,
                excluded_line_indices=excluded_amount_line_indices,
            )
            amount_line = amount_pairing.get("line")
            raw_amount_tokens = amount_pairing.get("tokens")
            amount_row_tokens = (
                list(raw_amount_tokens) if isinstance(raw_amount_tokens, list) else []
            )
        else:
            # Shared callers retain their existing materialization. Candidate B
            # never falls back to treating the printed year as an amount row.
            year_remainder = _YEAR_RE.sub("", year_line["text"].strip(), count=1)
            amount_line = year_line
            if not re.search(r"\d", year_remainder):
                amount_line = _explicit_amount_line_after_year(
                    line_items,
                    year_line,
                    x_min=grid_x0 - 10,
                    x_max=grid_x1 + 10,
                ) or year_line

        row_pad = 0.0 if enable_candidate_b_amount_pairing else 3.0
        status_band = _row_band(
            status_line,
            "status",
            x0=grid_x0,
            x1=grid_x1,
            pad_y=row_pad,
        )
        amount_band = (
            _row_band(
                amount_line,
                "overdue_amount",
                x0=grid_x0,
                x1=grid_x1,
                pad_y=row_pad,
            )
            if amount_line is not None
            else None
        )
        if enable_candidate_b_amount_pairing and amount_band is not None:
            status_center = (
                float(status_band["bbox"][1]) + float(status_band["bbox"][3])
            ) / 2.0
            amount_center = (
                float(amount_band["bbox"][1]) + float(amount_band["bbox"][3])
            ) / 2.0
            if amount_center <= status_center + 0.5:
                amount_pairing = {
                    **(amount_pairing or {}),
                    "status": "ambiguous_vertical_row_geometry",
                    "row_relation": "unresolved",
                    "tokens": [],
                    "cell_status_by_month": {
                        str(month): "ambiguous_vertical_row_geometry"
                        for month in active_months
                    },
                }
                amount_row_tokens = []
                amount_line = None
                amount_band = None
            elif float(status_band["bbox"][3]) > float(amount_band["bbox"][1]):
                boundary = (status_center + amount_center) / 2.0
                status_band["bbox"][3] = min(float(status_band["bbox"][3]), boundary)
                amount_band["bbox"][1] = max(float(amount_band["bbox"][1]), boundary)
        if enable_candidate_b_amount_pairing and amount_pairing is not None:
            amount_pairing_by_year[str(year)] = {
                key: value
                for key, value in amount_pairing.items()
                if key not in {"line", "tokens"}
            }
        status_band["index"] = len(row_bands)
        row_bands.append(status_band)
        if amount_band is not None:
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
            if row_page == page:
                year_visual_cols = visual_month_cols
            else:
                image_shape = tuple(int(value) for value in getattr(row_image, "shape", ()))
                cache_key = (
                    int(row_page),
                    id(row_image),
                    float(row_width),
                    float(row_height),
                    image_shape,
                )
                cached_geometry = continuation_visual_cols_cache.get(cache_key)
                if cached_geometry is None:
                    cached_geometry = _visual_month_col_bands(
                        month_cols,
                        page_image=row_image,
                        page_width=row_width,
                        page_height=row_height,
                        y0=max(0.0, row_local_bbox[1] - 5.0),
                        y1=min(row_height, row_local_bbox[3] + 5.0),
                        max_left_shift_months=1.85,
                    )
                    continuation_visual_cols_cache[cache_key] = cached_geometry
                year_visual_cols, _row_geometry_audit = cached_geometry
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
        # ``#`` has two deliberately distinct contracts.  Document-specific
        # callers may opt it into ``status_charset`` because PBOC detailed
        # reports print it as the business state "account opened, monthly
        # status unknown".  Other callers still retain an unapproved ``#``
        # only as an alignment witness so later months are not shifted.
        hash_is_business_status = "#" in status_charset
        status_chars = [ch for ch in raw_status_text if ch in status_charset or ch == "#"]
        candidate_status_tokens = status_line.get("candidate_b_status_tokens")
        status_row_tokens = (
            list(candidate_status_tokens)
            if enable_candidate_b_amount_pairing
            and isinstance(candidate_status_tokens, list)
            else _expand_line_to_char_tokens(
                {**status_line, "text": normalized_status_text},
                page=page,
                prefix=f"ocr_p{page}_repay_status_{year}",
            )
        )
        status_assignments = _assign_row(status_row_tokens, status_band, month_cols)
        if enable_candidate_b_amount_pairing:
            status_by_month = {}
            observed_status_token_count = 0
            for candidate_month in range(1, 13):
                cell_tokens = [
                    token
                    for token in status_assignments.get(candidate_month, [])
                    if token.text in status_charset or token.text == "#"
                ]
                observed_status_token_count += len(cell_tokens)
                if len(cell_tokens) == 1:
                    status_by_month[candidate_month] = cell_tokens[0].text
            observed_months = set(status_by_month)
            exact_full_row = (
                observed_months == set(range(1, 13))
                and observed_status_token_count == 12
            )
            exact_active_row = (
                observed_months == set(active_months)
                and observed_status_token_count == len(active_months)
            )
            row_alignment_exact = bool(
                (exact_full_row or exact_active_row)
                and ("#" not in status_by_month.values() or hash_is_business_status)
            )
        elif year == end_year and len(status_chars) == 2 and set(status_chars) == {"N", "C"}:
            status_by_month = (
                {active_months[0]: "N", active_months[1]: "C"}
                if len(active_months) == 2
                else {}
            )
            row_alignment_exact = False
        elif len(status_chars) == 12:
            status_by_month = dict(zip(range(1, 13), status_chars))
            row_alignment_exact = "#" not in status_chars or hash_is_business_status
        elif len(status_chars) == len(active_months):
            status_by_month = dict(zip(active_months, status_chars))
            row_alignment_exact = "#" not in status_chars or hash_is_business_status
        else:
            status_by_month = {}
            row_alignment_exact = False

        if amount_row_tokens is None:
            amount_row_tokens = (
                _expand_line_to_char_tokens(
                    amount_line,
                    page=page,
                    prefix=f"ocr_p{page}_repay_amount_{year}",
                )
                if amount_line is not None
                else []
            )
        amount_assignments = _assign_row(amount_row_tokens, amount_band, month_cols) if amount_band is not None else {}

        for col in month_cols:
            month = int(col["header"])
            st_tokens = status_assignments.get(month, [])
            status_center_y = (float(status_line["bbox"][1]) + float(status_line["bbox"][3])) / 2.0
            if amount_line is not None:
                amount_center_y = (
                    float(amount_line["bbox"][1]) + float(amount_line["bbox"][3])
                ) / 2.0
                st_tokens = [
                    token
                    for token in st_tokens
                    if abs(token.center[1] - status_center_y)
                    <= abs(token.center[1] - amount_center_y)
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
            visual_col = year_visual_cols_by_month.get(month)
            cell_col = visual_col or col
            visual_status_bbox = _cell_bbox(status_band, visual_col) if visual_col is not None else None
            visual = None
            if (
                (
                    enable_cell_ocr
                    or (
                        enable_static_status_validation
                        and status in static_sensitive_statuses
                    )
                )
                and (year, month) in record_months
                and visual_status_bbox is not None
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
            observed_row_status = ""
            if (
                enable_static_status_validation
                and status in static_sensitive_statuses
                and (year, month) in record_months
            ):
                static_status_attempts += 1
                observed_row_status = status
                if visual is None:
                    static_status_failed = True
                    static_status_unresolved += 1
                    static_status_unavailable += 1
                    status_recognition_source = "static_glyph_shape_unavailable"
                    status_recognition_audit = {
                        **status_recognition_audit,
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
                    classification = None
                    if static_template is not None:
                        classification = (
                            _static_candidate_b_zero_status_glyph_classification(
                                static_template
                            )
                            if enable_candidate_b_amount_pairing
                            else _static_n_star_glyph_classification(static_template)
                        )
                if visual is not None and classification is None:
                    # Static shape evidence only corroborates or corrects the
                    # canonical row observation; an indecisive crop must not
                    # erase that independently observed business candidate.
                    static_status_failed = True
                    static_status_unresolved += 1
                    status_recognition_source = "static_glyph_shape_unresolved"
                    status_recognition_audit = {
                        **status_recognition_audit,
                        "reason": "zero_status_glyph_not_decisive",
                        "observed_row_status": observed_row_status,
                        "logical_page": visual_page,
                    }
                elif visual is not None and classification is not None:
                    visual_status, visual_confidence, visual_audit = classification
                    static_status_resolved += 1
                    if visual_status != observed_row_status:
                        static_status_corrections += 1
                        static_status_contradictions.add(observed_row_status)
                    elif visual_confidence >= 0.95 and static_template is not None:
                        static_status_seeds.setdefault(visual_status, []).append(
                            {
                                "template": static_template,
                                "source_key": (
                                    int(visual_page),
                                    int(status_band["index"]),
                                    int(month),
                                ),
                                "confidence": float(visual_confidence),
                            }
                        )
                    status = visual_status
                    status_recognition_source = "static_glyph_shape_validation"
                    status_recognition_audit = {
                        **status_recognition_audit,
                        "reason": "field_specific_zero_status_shape_validation",
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
            status_ref_payload: dict[str, Any] = {
                "page": status_logical_page,
                "logical_page": status_logical_page,
                "geometry_scope": "cell" if visual_col is not None else "logical_page",
                **({"geometry_status": "unresolved"} if visual_col is None else {}),
                "coordinate_system": "pdf_points_top_left",
                "grid_id": f"mg_p{page}_repayment_{grid_index}",
                "row": status_band["index"],
                "col": month,
                "field_name": "status",
            }
            if visual_col is not None:
                status_ref_payload["bbox"] = _local_page_bbox(
                    _cell_bbox(status_band, cell_col),
                    logical_page=status_logical_page,
                    base_page=page,
                    base_page_height=page_height,
                )
            status_recognition_audit["source_ref"] = status_ref_payload
            st_cell = build_cell(
                row_band=status_band,
                col_band=cell_col,
                tokens=st_tokens,
                text=status,
                role="status",
                crop_ocr_text=status_crop,
                recognition_source=status_recognition_source,
                recognition_audit=status_recognition_audit,
            )
            status_cells.append(st_cell)
            if observed_row_status in static_sensitive_statuses and (year, month) in record_months:
                static_status_geometry.append(
                    (tuple(round(float(value), 4) for value in st_cell.bbox), year)
                )

            amount = ""
            status_amount_conflict = False
            amount_bbox = None
            if amount_band is not None:
                amt_tokens = amount_assignments.get(month, [])
                amount_pair_status = "exact"
                if enable_candidate_b_amount_pairing and amount_pairing is not None:
                    amount_pair_status = str(
                        (amount_pairing.get("cell_status_by_month") or {}).get(
                            str(month),
                            "not_applicable" if month not in active_months else "blank_amount_cell",
                        )
                    )
                    if amount_pair_status != "exact":
                        # Structure ambiguity is not a cue for cell OCR. Keep
                        # the observed candidates in audit and withhold value.
                        amt_tokens = []
                amount = _normalize_amount_text(_token_text(amt_tokens))
                status_implies_zero = status in zero_overdue_statuses
                status_amount_conflict = status_implies_zero and amount not in {"", "0"}
                amount_bbox = _cell_bbox(amount_band, cell_col) if visual_col is not None else None
                visual_amount_bbox = amount_bbox
                amount_crop = None
                if amount_pair_status not in {"exact", "not_applicable"}:
                    amount_recognition_source = "candidate_b_amount_pair_unresolved"
                    amount_recognition_audit = {
                        "reason": amount_pair_status,
                        "amount_pair_status": amount_pair_status,
                        "row_pair_status": str(amount_pairing.get("status") or "unresolved"),
                        "source_line_indices": list(
                            amount_pairing.get("source_line_indices") or ()
                        ),
                        "observed_texts": list(amount_pairing.get("observed_texts") or ()),
                    }
                elif status_amount_conflict:
                    amount_recognition_source = "status_amount_conflict"
                    amount_recognition_audit = {
                        "reason": "zero_status_conflicts_with_observed_nonzero_amount",
                        "status": status,
                        "observed_amount": amount,
                    }
                elif status_implies_zero and amount == "0":
                    amount_recognition_source = "explicit_paired_zero"
                    amount_recognition_audit = {"reason": "explicit_zero_in_paired_amount_cell"}
                else:
                    amount_recognition_source = "tokens"
                    amount_recognition_audit = {}
                if (
                    (not amount or status_recognition_source == "cell_crop_consensus")
                    and not status_implies_zero
                    and enable_cell_ocr
                    and (year, month) in record_months
                    and visual_amount_bbox is not None
                    and amount_pair_status == "exact"
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
                amount_ref_payload: dict[str, Any] = {
                    "page": amount_logical_page,
                    "logical_page": amount_logical_page,
                    "geometry_scope": "cell" if amount_bbox is not None else "logical_page",
                    **({"geometry_status": "unresolved"} if amount_bbox is None else {}),
                    "coordinate_system": "pdf_points_top_left",
                    "grid_id": f"mg_p{page}_repayment_{grid_index}",
                    "row": amount_band["index"],
                    "col": month,
                    "field_name": "overdue_amount",
                }
                if amount_bbox is not None:
                    amount_ref_payload["bbox"] = _local_page_bbox(
                        amount_bbox,
                        logical_page=amount_logical_page,
                        base_page=page,
                        base_page_height=page_height,
                    )
                amount_recognition_audit["source_ref"] = amount_ref_payload
                amount_cells.append(
                    build_cell(
                        row_band=amount_band,
                        col_band=cell_col,
                        tokens=amt_tokens,
                        text=amount,
                        role="overdue_amount",
                        crop_ocr_text=amount_crop,
                        recognition_source=amount_recognition_source,
                        recognition_audit=amount_recognition_audit,
                    )
                )

            if static_status_failed and observed_row_status in static_sensitive_statuses:
                pending_static_statuses.append(
                    {
                        "status_cells": status_cells,
                        "cell_index": len(status_cells) - 1,
                        "cell": st_cell,
                        "template": static_template,
                        "observed_status": observed_row_status,
                        "year": year,
                        "month": month,
                        "amount": amount,
                        "amount_bbox": amount_bbox,
                        "amount_band_index": amount_band["index"] if amount_band is not None else None,
                        "status_amount_conflict": bool(
                            observed_row_status in zero_overdue_statuses
                            and amount not in {"", "0"}
                        ),
                        "alignment_exact": bool(
                            row_alignment_exact
                            and header_alignment_exact
                            and (year, month) in record_months
                        ),
                    }
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
                        "overdue_amount": amount or None,
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

    # Static glyph validation is a correction plane, not a destructive veto.
    # Resolve indecisive exact-row zero-status cells only after the full grid has
    # supplied independent, scan-specific visual seeds. Later rows can thus
    # corroborate earlier cells without another OCR pass.
    prototypes = _scan_specific_status_prototypes(
        {
            status: observations
            for status, observations in static_status_seeds.items()
            if status not in static_status_contradictions
        }
    )
    geometry_years: dict[tuple[float, ...], set[int]] = {}
    for bbox_key, geometry_year in static_status_geometry:
        geometry_years.setdefault(bbox_key, set()).add(geometry_year)
    reused_status_geometry = {
        bbox_key for bbox_key, geometry_year_values in geometry_years.items() if len(geometry_year_values) > 1
    }
    for pending in pending_static_statuses:
        expected_status = str(pending.get("observed_status") or "")
        cell = pending.get("cell")
        template = pending.get("template")
        if (
            not isinstance(cell, MicroGridCell)
            or expected_status in static_status_contradictions
            or not pending.get("alignment_exact")
            or pending.get("status_amount_conflict")
            or template is None
            or tuple(round(float(value), 4) for value in cell.bbox) in reused_status_geometry
        ):
            continue
        match = _scan_specific_status_match(
            template,
            expected_status=expected_status,
            prototypes=prototypes,
        )
        if match is None:
            continue
        similarity, margin = match
        prior_audit = dict(cell.recognition_audit or {})
        repaired_cell = replace(
            cell,
            text=expected_status,
            recognition_source="static_grid_template_consensus",
            recognition_audit={
                **prior_audit,
                "reason": "scan_specific_static_template_consensus",
                "observed_row_status": expected_status,
                "visual_status": expected_status,
                "template_similarity": round(similarity, 4),
                "template_margin": round(margin, 4),
                "template_seed_count": int(
                    (prototypes.get(expected_status) or {}).get("seed_count") or 0
                ),
            },
        )
        status_cells = pending.get("status_cells")
        cell_index = int(pending.get("cell_index") or 0)
        if not isinstance(status_cells, list) or not (0 <= cell_index < len(status_cells)):
            continue
        status_cells[cell_index] = repaired_cell
        year = int(pending.get("year") or 0)
        month = int(pending.get("month") or 0)
        amount = str(pending.get("amount") or "")
        amount_bbox = pending.get("amount_bbox")
        amount_band_index = pending.get("amount_band_index")
        refs = [
            {
                "page": page,
                "grid_id": f"mg_p{page}_repayment_{grid_index}",
                "row": repaired_cell.row_index,
                "col": month,
            }
        ]
        if amount_band_index is not None:
            refs.append(
                {
                    "page": page,
                    "grid_id": f"mg_p{page}_repayment_{grid_index}",
                    "row": int(amount_band_index),
                    "col": month,
                }
            )
        records.append(
            {
                "year": year,
                "month": month,
                "status": expected_status,
                "overdue_amount": amount or None,
                "status_bbox": list(repaired_cell.bbox),
                **(
                    {"amount_bbox": list(amount_bbox)}
                    if isinstance(amount_bbox, (list, tuple)) and len(amount_bbox) == 4
                    else {}
                ),
                "source_cell_refs": refs,
                "confidence": repaired_cell.confidence or 0.7,
                "recognition_source": repaired_cell.recognition_source,
                "audit": repaired_cell.recognition_audit,
            }
        )
        static_template_consensus_resolved += 1

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
            "month_header_geometry": str(header_line.get("month_header_geometry") or "fallback_word"),
            "month_header_observed_count": int(header_line.get("month_header_observed_count") or 0),
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
                "required_statuses": sorted(static_sensitive_statuses),
                "attempts": static_status_attempts,
                "resolved": static_status_resolved,
                "corrections": static_status_corrections,
                "template_consensus_resolved": static_template_consensus_resolved,
                "unresolved": max(
                    0,
                    static_status_unresolved - static_template_consensus_resolved,
                ),
                "unavailable": static_status_unavailable,
                "contradicted_observed_symbols": sorted(static_status_contradictions),
            },
            **(
                {"candidate_b_amount_pairing": amount_pairing_by_year}
                if enable_candidate_b_amount_pairing
                else {}
            ),
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
    enable_candidate_b_amount_pairing: bool = False,
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
        enable_candidate_b_amount_pairing=enable_candidate_b_amount_pairing,
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
    grid_audit = grid.get("audit") if isinstance(grid.get("audit"), dict) else {}
    candidate_b_amount_pairing = (
        grid_audit.get("candidate_b_amount_pairing")
        if isinstance(grid_audit.get("candidate_b_amount_pairing"), dict)
        else {}
    )
    static_validation_audit = grid_audit.get("static_status_validation") or {}
    static_validation_enabled = bool(static_validation_audit.get("enabled"))
    static_sensitive_statuses = {
        str(value)
        for value in static_validation_audit.get("required_statuses") or ("N", "*")
        if value
    }
    visual_month_geometry = grid_audit.get("visual_month_geometry") or {}
    month_geometry_usable = not isinstance(visual_month_geometry, dict) or (
        visual_month_geometry.get("usable") is not False
    )

    months_by_col: dict[int, set[int]] = {}
    for band in grid.get("col_bands") or []:
        if not isinstance(band, dict):
            continue
        header = str(band.get("header") or "").strip()
        if header.isdigit():
            months_by_col.setdefault(int(band.get("index") or 0), set()).add(int(header))
    col_map = {
        col_idx: next(iter(months))
        for col_idx, months in months_by_col.items()
        if len(months) == 1 and next(iter(months)) in range(1, 13)
    }
    unique_month_columns = (
        month_geometry_usable
        and len(col_map) == len(set(col_map.values()))
        and all(len(months) == 1 for months in months_by_col.values())
    )

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

    amount_cells_by_row_col: dict[tuple[int, int], list[dict[str, Any]]] = {}
    amount_rows_by_index: dict[int, list[list[dict[str, Any]]]] = {}
    for row in amount_rows:
        row_indices = {
            int(cell.get("row_index") or 0)
            for cell in row
            if str(cell.get("role") or "") == "overdue_amount"
        }
        if len(row_indices) != 1:
            continue
        row_idx = next(iter(row_indices))
        amount_rows_by_index.setdefault(row_idx, []).append(row)
        for cell in row:
            if str(cell.get("role") or "") != "overdue_amount":
                continue
            amount_cells_by_row_col.setdefault(
                (row_idx, int(cell.get("col_index") or 0)), []
            ).append(cell)

    status_year_counts: dict[int, int] = {}
    for row in status_rows:
        years = {
            int(match.group(0))
            for cell in row
            if str(cell.get("role") or "") == "year"
            and (match := _YEAR_RE.match(str(cell.get("text") or "").strip()))
        }
        if len(years) == 1:
            year = next(iter(years))
            status_year_counts[year] = status_year_counts.get(year, 0) + 1
    status_cell_by_year_month: dict[tuple[int, int], dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    for status_row in status_rows:
        status_row_indices = {
            int(cell.get("row_index") or 0)
            for cell in status_row
            if str(cell.get("role") or "") == "status"
        }
        status_years = {
            int(match.group(0))
            for cell in status_row
            if str(cell.get("role") or "") == "year"
            and (match := _YEAR_RE.match(str(cell.get("text") or "").strip()))
        }
        row_idx = next(iter(status_row_indices)) if len(status_row_indices) == 1 else -1
        row_year = next(iter(status_years)) if len(status_years) == 1 else None
        status_cells_by_col: dict[int, list[dict[str, Any]]] = {}
        for candidate_cell in status_row:
            if str(candidate_cell.get("role") or "") == "status":
                status_cells_by_col.setdefault(
                    int(candidate_cell.get("col_index") or 0), []
                ).append(candidate_cell)
        amount_row_idx = row_idx + 1
        paired_amount_rows = amount_rows_by_index.get(amount_row_idx, [])
        unique_status_row_role = bool(
            row_idx >= 0
            and row_year is not None
            and status_year_counts.get(row_year) == 1
        )
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
            paired_amount_cells = amount_cells_by_row_col.get((amount_row_idx, col_idx), [])
            amount_cell = paired_amount_cells[0] if len(paired_amount_cells) == 1 else None
            amount_audit = (
                dict(amount_cell.get("recognition_audit") or {})
                if isinstance(amount_cell, dict)
                else {}
            )
            year_pairing = (
                candidate_b_amount_pairing.get(str(year))
                if isinstance(candidate_b_amount_pairing.get(str(year)), dict)
                else {}
            )
            declared_pair_status = str(
                amount_audit.get("amount_pair_status")
                or (year_pairing.get("cell_status_by_month") or {}).get(str(month))
                or ""
            )
            amount = _explicit_amount_value(amount_cell.get("text")) if amount_cell else None
            unique_status_cell_geometry = bool(
                unique_month_columns
                and len(status_cells_by_col.get(col_idx, [])) == 1
            )
            exact_status_geometry = unique_status_row_role and unique_status_cell_geometry
            unique_amount_geometry = bool(
                len(paired_amount_rows) == 1
                and len(paired_amount_cells) == 1
                and declared_pair_status in {"", "exact"}
            )
            exact_row_pair = exact_status_geometry and unique_amount_geometry
            if not exact_row_pair:
                amount = None
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
            if isinstance(amount_cell, dict):
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
            amount_pair_status = declared_pair_status
            if not amount_pair_status:
                if len(paired_amount_rows) != 1:
                    amount_pair_status = (
                        "missing_amount_row"
                        if not paired_amount_rows
                        else "ambiguous_immediate_rows"
                    )
                elif len(paired_amount_cells) != 1:
                    amount_pair_status = (
                        "blank_amount_cell"
                        if not paired_amount_cells
                        else "duplicate_or_ambiguous_cell"
                    )
                elif amount is None:
                    amount_pair_status = "blank_amount_cell"
                else:
                    amount_pair_status = "exact"
            amount_pair_unresolved = bool(
                accept_exact_row_numeric_status
                and status
                and (amount_pair_status != "exact" or amount is None)
            )
            if amount_pair_unresolved and not isinstance(amount_cell, dict):
                source_refs.append(
                    {
                        "page": logical_page,
                        "logical_page": logical_page,
                        "grid_id": grid_id,
                        "row": amount_row_idx,
                        "col": month,
                        "field_name": "overdue_amount",
                        "geometry_scope": "logical_page",
                        "geometry_status": "unresolved",
                        "amount_pair_status": amount_pair_status,
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
                "_exact_row_pair": exact_row_pair,
                "_exact_status_geometry": exact_status_geometry,
                "_paired_amount_explicit": amount is not None,
                "_paired_amount_positive": _is_positive_amount(amount),
            }
            if amount_pair_unresolved:
                record["_amount_pairing"] = {
                    "status": amount_pair_status,
                    "row_pair_status": str(year_pairing.get("status") or amount_pair_status),
                    "source_line_indices": list(year_pairing.get("source_line_indices") or ()),
                    "observed_texts": list(year_pairing.get("observed_texts") or ()),
                }
                record["extraction_status"] = "review"
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
        record["confidence"] = 0.0
        record["extraction_status"] = "review"
        record["audit"] = {"reason": "status_geometry_reused_across_years"}
    for record in records:
        status = str(record.get("status") or "")
        crop_confirmed = (
            str(record.get("recognition_source") or "") == "cell_crop_consensus"
            and int((record.get("audit") or {}).get("consensus_count") or 0) >= 2
        )
        exact_row_sequence = (
            accept_exact_row_numeric_status
            and str(record.get("recognition_source") or "") == "canonical_row_sequence"
            and str((record.get("audit") or {}).get("alignment_status") or "") == "exact"
        )
        exact_row_pair = bool(record.get("_exact_row_pair"))
        exact_status_geometry = bool(record.get("_exact_status_geometry"))
        paired_amount_positive = bool(record.get("_paired_amount_positive"))
        static_status_corroborated = not static_validation_enabled or str(
            record.get("recognition_source") or ""
        ) in {"static_glyph_shape_validation", "static_grid_template_consensus"}
        static_sensitive_exact_candidate = bool(
            status in static_sensitive_statuses
            and exact_status_geometry
        )
        structurally_typed_status = status in (_STATUS_CHARS | {"A", "#"})
        if (
            accept_exact_row_numeric_status
            and structurally_typed_status
            and not status.isdigit()
            and not (
                exact_status_geometry
                and (
                    status not in static_sensitive_statuses
                    or static_status_corroborated
                    or static_sensitive_exact_candidate
                )
            )
        ):
            record["raw_status"] = status
            record["status"] = "unknown"
            record["confidence"] = 0.0
            record["extraction_status"] = "review"
            unresolved_fields = ["status_code"]
            if record.get("overdue_amount") is None:
                unresolved_fields.append("overdue_amount")
            record["audit"] = {
                "reason": "symbolic_status_row_role_or_month_geometry_unresolved",
                "field_name": "status_code",
                "unresolved_fields": unresolved_fields,
            }
        elif status.isdigit() and not (
            status in {"1", "2", "3", "4", "5", "6", "7"}
            and exact_row_pair
            and exact_status_geometry
            and paired_amount_positive
            and (crop_confirmed or exact_row_sequence)
        ):
            record["raw_status"] = status
            record["status"] = "unknown"
            record["confidence"] = 0.0
            record["extraction_status"] = "review"
            unresolved_fields = ["status_code"]
            if record.get("overdue_amount") is None:
                unresolved_fields.append("overdue_amount")
            record["audit"] = {
                "reason": (
                    "numeric_status_requires_positive_paired_amount"
                    if status in {"1", "2", "3", "4", "5", "6", "7"}
                    and exact_row_pair
                    and exact_status_geometry
                    else "numeric_status_row_role_or_month_geometry_unresolved"
                ),
                "field_name": "status_code",
                "unresolved_fields": unresolved_fields,
            }
        elif static_sensitive_exact_candidate and not static_status_corroborated:
            prior_audit = dict(record.get("audit") or {})
            record["extraction_status"] = "review"
            record["audit"] = {
                **prior_audit,
                "reason": "zero_status_static_corroboration_unavailable",
                "field_name": "status_code",
                "observed_status": status,
                "reported_value_retained": True,
                **(
                    {"static_reason": prior_audit.get("reason")}
                    if prior_audit.get("reason")
                    else {}
                ),
            }
    for record in records:
        for internal_key in (
            "_exact_row_pair",
            "_exact_status_geometry",
            "_paired_amount_explicit",
            "_paired_amount_positive",
        ):
            record.pop(internal_key, None)
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
