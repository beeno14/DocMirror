# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Credit-report repayment micro-grid reconstruction."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from math import isfinite
from typing import Any, cast

from docmirror.ocr.micro_grid.cell_recognition import (
    extract_micro_cell_glyph_template,
    normalize_allowlist_text,
    pdf_bbox_to_image_region,
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
from docmirror.plugins.credit_report.source_table_month_lattice import (
    SourceTableMonthLattice,
    resolve_unique_source_table_year_plus_twelve_ownership,
    resolve_unique_source_table_year_plus_twelve_ownership_from_year,
)

_RANGE_RE = re.compile(r"(20\d{2})年\s*(\d{1,2})月\s*[-—一至~～]\s*(20\d{2})年\s*(\d{1,2})月.*还款记录")
_YEAR_RE = re.compile(r"^20\d{2}(?=\s|$)")
_STATUS_CHARS = {"*", "/", "N", "C", "1", "2", "3", "4", "5", "6", "7", "B", "M", "D", "Z", "G"}
_ZERO_OVERDUE_STATUSES = {"*", "/", "N", "C"}
_MIN_MONTH_GRID_PAGE_COVERAGE = 0.40
_EXACT_SOURCE_STATUS_CELL_SOURCES = frozenset(
    {
        "exact_native_source_table_status_cell",
        "exact_corrected_source_table_status_cell",
    }
)
_EXACT_SOURCE_AMOUNT_CELL_SOURCES = frozenset(
    {
        "exact_native_source_table_amount_cell",
        "exact_corrected_source_table_amount_cell",
    }
)


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
    def positive_page(value: Any) -> int | None:
        return (
            value
            if isinstance(value, int) and not isinstance(value, bool) and value > 0
            else None
        )

    out: list[dict[str, Any]] = []
    for idx, line in enumerate(lines):
        b = _bbox(line)
        t = _text(line)
        if not b or not t:
            continue
        raw_source_logical_page = (
            line.get("source_logical_page") if isinstance(line, dict) else getattr(line, "source_logical_page", None)
        )
        coordinate_logical_page = positive_page(
            line.get("coordinate_logical_page")
            if isinstance(line, dict)
            else getattr(line, "coordinate_logical_page", None)
        )
        explicit_source_origin_logical_page = positive_page(
            line.get("source_origin_logical_page")
            if isinstance(line, dict)
            else getattr(line, "source_origin_logical_page", None)
        )
        raw_coordinate_status = (
            line.get("coordinate_status")
            if isinstance(line, dict)
            else getattr(line, "coordinate_status", None)
        )
        coordinate_status = str(raw_coordinate_status or "").strip()
        raw_source_logical_page = positive_page(raw_source_logical_page)
        effective_logical_page = coordinate_logical_page or raw_source_logical_page
        source_origin_logical_page = (
            explicit_source_origin_logical_page
            or raw_source_logical_page
        )
        raw_source_bbox = (
            line.get("source_bbox")
            if isinstance(line, dict)
            else getattr(line, "source_bbox", None)
        )
        printed_anchor_identity = (
            line.get("printed_anchor_identity")
            if isinstance(line, dict)
            else getattr(line, "printed_anchor_identity", None)
        )
        source_bbox: tuple[float, float, float, float] | None = None
        if isinstance(raw_source_bbox, (list, tuple)) and len(raw_source_bbox) == 4:
            try:
                candidate_source_bbox = tuple(float(value) for value in raw_source_bbox)
            except (TypeError, ValueError):
                candidate_source_bbox = ()
            if (
                len(candidate_source_bbox) == 4
                and candidate_source_bbox[2] > candidate_source_bbox[0]
                and candidate_source_bbox[3] > candidate_source_bbox[1]
            ):
                source_bbox = cast(
                    tuple[float, float, float, float],
                    candidate_source_bbox,
                )
        out.append(
            {
                "idx": idx,
                "text": t,
                "bbox": b,
                "confidence": _confidence(line),
                **(
                    {"source_logical_page": effective_logical_page}
                    if effective_logical_page is not None
                    else {}
                ),
                **(
                    {"coordinate_logical_page": coordinate_logical_page}
                    if coordinate_logical_page is not None
                    else {}
                ),
                **(
                    {"source_origin_logical_page": source_origin_logical_page}
                    if source_origin_logical_page is not None
                    else {}
                ),
                **({"coordinate_status": coordinate_status} if coordinate_status else {}),
                **({"source_bbox": source_bbox} if source_bbox is not None else {}),
                **(
                    {"printed_anchor_identity": deepcopy(printed_anchor_identity)}
                    if isinstance(printed_anchor_identity, Mapping)
                    else {}
                ),
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


def _candidate_b_owned_status_token(
    cell_tokens: list[OCRToken],
    *,
    row_tokens: list[OCRToken],
    visual_col: Mapping[str, Any] | None,
    geometry_audit: Mapping[str, Any],
    status_charset: set[str],
) -> tuple[str, OCRToken] | None:
    """Return one independently positioned status token from an owned month cell.

    An incomplete merged row is not a trustworthy positional sequence: splitting
    eleven glyphs across twelve months can shift every later value.  Candidate B
    therefore uses this witness only when the physical year-plus-twelve-rule
    lattice owns the month, exactly one token falls in that cell, and the token
    came from a one-character OCR word rather than a synthetic row split.
    """

    if not (
        visual_col is not None
        and geometry_audit.get("usable") is not False
        and geometry_audit.get("source") == "vertical_rule_projection"
        and geometry_audit.get("selection_basis") == "year_plus_twelve_rule_ownership"
        and len(cell_tokens) == 1
    ):
        return None
    token = cell_tokens[0]
    chars = _candidate_b_status_chars(token.text, status_charset)
    if len(chars) != 1 or chars[0] not in status_charset:
        return None
    source_key = token.source_token_id or token.token_id
    if sum(1 for row_token in row_tokens if (row_token.source_token_id or row_token.token_id) == source_key) != 1:
        return None
    try:
        token_x0, _token_y0, token_x1, _token_y1 = (float(value) for value in token.bbox)
        col_x0, _col_y0, col_x1, _col_y1 = (float(value) for value in visual_col["bbox"])
    except (KeyError, TypeError, ValueError):
        return None
    token_width = token_x1 - token_x0
    overlap = max(0.0, min(token_x1, col_x1) - max(token_x0, col_x0))
    token_center = (token_x0 + token_x1) / 2.0
    if token_width <= 0.0 or overlap / token_width < 0.80 or not col_x0 <= token_center <= col_x1:
        return None
    return chars[0], token


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
                        (width, height), _baseline = cv2.getTextSize(character, font, font_scale, thickness)
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
    dense_cols = np.where(ink_full.mean(axis=0) >= max(0.12, 2.0 / max(ink_full.shape[0], 1)))[0]
    dense_rows = np.where(ink_full.mean(axis=1) >= max(0.12, 2.0 / max(ink_full.shape[1], 1)))[0]
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
        if winning_label == "*" and winning_score >= 0.90 and margin >= 0.030 and solidity <= 0.75:
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
            if (
                label != "N"
                or float(reference_scores.get("N") or 0.0) - float(reference_scores.get("C") or 0.0) >= 0.06
            ):
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


def _static_amount_zero_glyph_classification(
    page_image: Any,
    bbox: BBox,
    *,
    page_width: float,
    page_height: float,
) -> tuple[float, dict[str, Any]] | None:
    """Recognize one printed zero without invoking OCR or using status semantics.

    Watermarks crossing a small monthly-amount cell can make OCR report ``o``,
    ``10``, or ``20`` for a single printed zero.  The dark report glyph remains
    a closed loop, while the watermark is materially lighter.  Require that
    topology and a conservative zero-shape vote at several fixed thresholds;
    any second digit-like component vetoes the repair.
    """

    shape = getattr(page_image, "shape", None)
    if not shape or len(shape) < 2:
        return None
    try:
        import cv2
        import numpy as np

        region = pdf_bbox_to_image_region(
            bbox,
            page_width=page_width,
            page_height=page_height,
            image_width=int(shape[1]),
            image_height=int(shape[0]),
            pad_px=0,
        )
        crop = page_image[region[1] : region[3], region[0] : region[2]]
        if crop is None or getattr(crop, "size", 0) == 0:
            return None
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if len(crop.shape) == 3 else crop.copy()
        crop_height, crop_width = gray.shape[:2]
        if crop_height < 10 or crop_width < 10:
            return None
        border = max(1, int(round(min(crop_height, crop_width) * 0.07)))
        gray[:border, :] = 255
        gray[-border:, :] = 255
        gray[:, :border] = 255
        gray[:, -border:] = 255

        votes: list[dict[str, Any]] = []
        for threshold in (96, 112, 128, 144, 160):
            binary = (gray < threshold).astype(np.uint8)
            count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
                binary,
                8,
            )
            components: list[dict[str, Any]] = []
            for label in range(1, count):
                x, y, width, height, area = (int(value) for value in stats[label])
                center_x = x + width / 2.0
                center_y = y + height / 2.0
                if not (
                    area >= 6
                    and height >= max(6, int(round(crop_height * 0.20)))
                    and height <= crop_height * 0.78
                    and width >= 1
                    and width <= crop_width * 0.45
                    and crop_width * 0.15 <= center_x <= crop_width * 0.85
                    and crop_height * 0.10 <= center_y <= crop_height * 0.90
                ):
                    continue
                component_mask = labels[y : y + height, x : x + width] == label
                contours, hierarchy = cv2.findContours(
                    component_mask.astype(np.uint8) * 255,
                    cv2.RETR_CCOMP,
                    cv2.CHAIN_APPROX_SIMPLE,
                )[-2:]
                holes = 0 if hierarchy is None else sum(1 for item in hierarchy[0] if int(item[3]) >= 0)
                white_glyph = np.where(component_mask, 0, 255).astype(np.uint8)
                template = _normalize_static_reference_glyph(
                    white_glyph,
                    np=np,
                    cv2=cv2,
                )
                scores = _static_status_reference_scores(template, np=np)
                ranked = sorted(
                    ((float(score), str(label)) for label, score in scores.items()),
                    reverse=True,
                )
                components.append(
                    {
                        "area": area,
                        "bbox": (x, y, width, height),
                        "center": (center_x, center_y),
                        "holes": holes,
                        "ranked": ranked,
                    }
                )

            zero_candidates: list[dict[str, Any]] = []
            for component in components:
                ranked = component["ranked"]
                if not ranked:
                    continue
                winning_score, winning_label = ranked[0]
                competitor_score = ranked[1][0] if len(ranked) > 1 else 0.0
                _x, _y, width, height = component["bbox"]
                if (
                    component["holes"] == 1
                    and 0.30 <= width / max(height, 1) <= 0.90
                    and winning_label == "0"
                    and winning_score >= 0.82
                    and winning_score - competitor_score >= 0.02
                ):
                    zero_candidates.append(component)
            if len(zero_candidates) != 1:
                continue
            zero = zero_candidates[0]
            _zero_x, _zero_y, _zero_width, zero_height = zero["bbox"]
            blockers = [
                component
                for component in components
                if component is not zero
                and component["area"] >= max(6, int(round(zero["area"] * 0.18)))
                and component["bbox"][3] >= zero_height * 0.60
                and component["ranked"]
                and component["ranked"][0][1] in set("0123456789")
                and component["ranked"][0][0] >= 0.72
            ]
            if blockers:
                continue
            winning_score, _winning_label = zero["ranked"][0]
            competitor_score = zero["ranked"][1][0]
            votes.append(
                {
                    "threshold": threshold,
                    "score": winning_score,
                    "margin": winning_score - competitor_score,
                    "center": zero["center"],
                }
            )

        if len(votes) < 2:
            return None
        normalized_x = [vote["center"][0] / crop_width for vote in votes]
        normalized_y = [vote["center"][1] / crop_height for vote in votes]
        if max(normalized_x) - min(normalized_x) > 0.08 or max(normalized_y) - min(normalized_y) > 0.08:
            return None
        confidence = min(
            0.99,
            0.72 + 0.04 * len(votes) + 0.08 * min(vote["score"] for vote in votes),
        )
        return (
            confidence,
            {
                "classification_basis": "dark_ink_closed_zero_multithreshold",
                "threshold_vote_count": len(votes),
                "thresholds": [vote["threshold"] for vote in votes],
                "minimum_reference_similarity": round(
                    min(vote["score"] for vote in votes),
                    4,
                ),
                "minimum_reference_margin": round(
                    min(vote["margin"] for vote in votes),
                    4,
                ),
                "ocr_invoked": False,
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
            float(vector @ prototype["vector"]) for status, prototype in prototypes.items() if status != expected_status
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
                "exact" if header_line.get("month_header_geometry") == "word_center_sequence_exact" else "estimated"
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
    """Resolve an origin-page crop without mixing coordinate planes."""

    explicit_coordinate_page = source_line.get("coordinate_logical_page")
    coordinate_page = int(
        explicit_coordinate_page
        or source_line.get("source_logical_page")
        or base_page
    )
    source_origin_page = int(
        source_line.get("source_origin_logical_page")
        or source_line.get("source_logical_page")
        or coordinate_page
    )
    context = (
        page_image_resolver(source_origin_page)
        if page_image_resolver is not None
        else None
    )
    if isinstance(context, dict):
        image = context.get("image")
        width = float(context.get("page_width") or base_page_width or 0.0)
        height = float(context.get("page_height") or base_page_height or 0.0)
    else:
        image = page_image if source_origin_page == base_page else None
        width = float(base_page_width or 0.0)
        height = float(base_page_height or 0.0)
    if image is None or width <= 0 or height <= 0:
        return None
    x0, y0, x1, y1 = (float(value) for value in bbox)
    if explicit_coordinate_page is not None:
        registered_line_bbox = _bbox(source_line)
        raw_source_bbox = source_line.get("source_bbox")
        if (
            registered_line_bbox is None
            or not isinstance(raw_source_bbox, (list, tuple))
            or len(raw_source_bbox) != 4
        ):
            return None
        try:
            sx0, sy0, sx1, sy1 = (float(value) for value in raw_source_bbox)
        except (TypeError, ValueError):
            return None
        cx0, cy0, cx1, cy1 = registered_line_bbox
        if (
            not all(
                isfinite(value)
                for value in (sx0, sy0, sx1, sy1, cx0, cy0, cx1, cy1)
            )
            or sx1 <= sx0
            or sy1 <= sy0
            or cx1 <= cx0
            or cy1 <= cy0
        ):
            return None
        scale_x = (cx1 - cx0) / (sx1 - sx0)
        scale_y = (cy1 - cy0) / (sy1 - sy0)
        if not isfinite(scale_x) or not isfinite(scale_y) or scale_x <= 0.0 or scale_y <= 0.0:
            return None
        x0, x1 = sx0 + (x0 - cx0) / scale_x, sx0 + (x1 - cx0) / scale_x
        y0, y1 = sy0 + (y0 - cy0) / scale_y, sy0 + (y1 - cy0) / scale_y
    elif source_origin_page != base_page:
        # Legacy joined evidence uses a one-page vertical stack.
        shift = float(base_page_height or 0.0)
        y0 -= shift
        y1 -= shift
    if not all(isfinite(value) for value in (x0, y0, x1, y1)) or x1 <= x0 or y1 <= y0:
        return None
    return image, (x0, y0, x1, y1), width, height, source_origin_page


def _local_page_bbox(
    bbox: BBox,
    *,
    logical_page: int,
    base_page: int,
    base_page_height: float | None,
    coordinates_already_registered: bool = False,
    coordinate_status: str | None = None,
) -> list[float] | None:
    """Return a continuation-local canonical box for tables and persisted refs.

    Canonical registration and cross-page stacking are independent transforms.
    A registered continuation line can therefore still require the explicit
    stack shift to be removed before it is compared with page-local source-table
    geometry.  If that shift cannot be proved, fail closed instead of returning
    a box in the wrong coordinate plane.
    """
    x0, y0, x1, y1 = (float(value) for value in bbox)
    must_remove_stack_shift = bool(
        logical_page != base_page
        and (
            str(coordinate_status or "") == "cross_page_y_shift"
            or not coordinates_already_registered
        )
    )
    if must_remove_stack_shift:
        if not base_page_height or float(base_page_height) <= 0.0:
            return None
        y0 -= float(base_page_height)
        y1 -= float(base_page_height)
    return [x0, y0, x1, y1]


def _stacked_page_bbox(
    bbox: Iterable[Any],
    *,
    logical_page: int,
    base_page: int,
    base_page_height: float | None,
) -> tuple[float, float, float, float] | None:
    """Register one page-local canonical box in the base-page stack."""

    try:
        parsed = tuple(float(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    if (
        len(parsed) != 4
        or not all(isfinite(value) for value in parsed)
        or parsed[2] <= parsed[0]
        or parsed[3] <= parsed[1]
    ):
        return None
    x0, y0, x1, y1 = parsed
    if logical_page != base_page:
        if not base_page_height or float(base_page_height) <= 0.0:
            return None
        y0 += float(base_page_height)
        y1 += float(base_page_height)
    return x0, y0, x1, y1


def _source_lattice_row_provenance(
    lattice: SourceTableMonthLattice,
    *,
    bbox: Iterable[Any],
    base_page: int,
    base_page_height: float | None,
) -> dict[str, Any]:
    """Carry the table's proved raw inverse onto a synthesized stacked row."""

    logical_page = int(lattice.logical_page)
    provenance: dict[str, Any] = {
        "source_logical_page": logical_page,
        "coordinate_logical_page": logical_page,
        "source_origin_logical_page": int(lattice.source_logical_page),
    }
    if logical_page != base_page:
        provenance["coordinate_status"] = "cross_page_y_shift"
    affine = lattice.source_to_canonical_affine
    if affine is None:
        return provenance
    try:
        x0, y0, x1, y1 = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return provenance
    if logical_page != base_page:
        if not base_page_height or float(base_page_height) <= 0.0:
            return provenance
        y0 -= float(base_page_height)
        y1 -= float(base_page_height)
    scale_x, scale_y, offset_x, offset_y = affine
    if (
        not all(isfinite(value) for value in (*affine, x0, y0, x1, y1))
        or scale_x <= 0.0
        or scale_y <= 0.0
        or x1 <= x0
        or y1 <= y0
    ):
        return provenance
    provenance["source_bbox"] = [
        (x0 - offset_x) / scale_x,
        (y0 - offset_y) / scale_y,
        (x1 - offset_x) / scale_x,
        (y1 - offset_y) / scale_y,
    ]
    return provenance


def _synthesized_row_provenance(
    contributors: Iterable[Mapping[str, Any]],
    *,
    bbox: Iterable[Any],
    page: int,
) -> dict[str, Any]:
    """Preserve a synthesized row's coordinate plane without inventing it.

    The coordinate marker is retained independently of crop availability.  A
    raw source box is emitted only when every contributing line proves the same
    coordinate page, origin page, and affine transform; otherwise later visual
    OCR fails closed while structural reconstruction may continue.
    """

    rows = list(contributors)
    if not rows:
        return {"source_logical_page": int(page)}

    def positive_page(value: Any) -> int | None:
        return (
            int(value)
            if isinstance(value, int) and not isinstance(value, bool) and int(value) > 0
            else None
        )

    source_pages = {
        positive_page(row.get("source_logical_page"))
        or positive_page(row.get("coordinate_logical_page"))
        or int(page)
        for row in rows
    }
    source_page = next(iter(source_pages)) if len(source_pages) == 1 else int(page)
    provenance: dict[str, Any] = {"source_logical_page": source_page}

    coordinate_pages = [positive_page(row.get("coordinate_logical_page")) for row in rows]
    explicit_coordinate_pages = {value for value in coordinate_pages if value is not None}
    if explicit_coordinate_pages:
        coordinate_page = (
            next(iter(explicit_coordinate_pages))
            if len(explicit_coordinate_pages) == 1
            else source_page
        )
        provenance["coordinate_logical_page"] = coordinate_page

    statuses = {str(row.get("coordinate_status") or "").strip() for row in rows}
    if len(statuses) == 1 and (coordinate_status := next(iter(statuses))):
        provenance["coordinate_status"] = coordinate_status

    if not explicit_coordinate_pages or any(value is None for value in coordinate_pages):
        return provenance
    coordinate_page = int(provenance["coordinate_logical_page"])
    if any(value != coordinate_page for value in coordinate_pages):
        return provenance

    origins = [positive_page(row.get("source_origin_logical_page")) for row in rows]
    if any(value is None for value in origins) or len(set(origins)) != 1:
        return provenance

    transforms: list[tuple[float, float, float, float]] = []
    for row in rows:
        canonical_box = _bbox(row)
        raw_source_box = row.get("source_bbox")
        if (
            canonical_box is None
            or not isinstance(raw_source_box, (list, tuple))
            or len(raw_source_box) != 4
        ):
            return provenance
        try:
            sx0, sy0, sx1, sy1 = (float(value) for value in raw_source_box)
        except (TypeError, ValueError):
            return provenance
        cx0, cy0, cx1, cy1 = canonical_box
        if (
            not all(isfinite(value) for value in (sx0, sy0, sx1, sy1, cx0, cy0, cx1, cy1))
            or sx1 <= sx0
            or sy1 <= sy0
            or cx1 <= cx0
            or cy1 <= cy0
        ):
            return provenance
        scale_x = (cx1 - cx0) / (sx1 - sx0)
        scale_y = (cy1 - cy0) / (sy1 - sy0)
        offset_x = cx0 - scale_x * sx0
        offset_y = cy0 - scale_y * sy0
        if scale_x <= 0.0 or scale_y <= 0.0:
            return provenance
        transforms.append((scale_x, scale_y, offset_x, offset_y))

    reference = transforms[0]
    for transform in transforms[1:]:
        if any(
            abs(value - expected) > 1e-4 * max(1.0, abs(expected))
            for value, expected in zip(transform, reference, strict=True)
        ):
            return provenance
    try:
        x0, y0, x1, y1 = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return provenance
    if (
        not all(isfinite(value) for value in (x0, y0, x1, y1))
        or x1 <= x0
        or y1 <= y0
    ):
        return provenance
    scale_x, scale_y, offset_x, offset_y = reference
    source_bbox = [
        (x0 - offset_x) / scale_x,
        (y0 - offset_y) / scale_y,
        (x1 - offset_x) / scale_x,
        (y1 - offset_y) / scale_y,
    ]
    if not all(isfinite(value) for value in source_bbox):
        return provenance
    provenance["source_origin_logical_page"] = int(origins[0])
    provenance["source_bbox"] = source_bbox
    return provenance


def _representative_year_column_bbox(
    year_lines: Iterable[Mapping[str, Any]],
    *,
    logical_page: int,
) -> list[float] | None:
    """Return the median printed-year box on one logical repayment page."""

    boxes: list[tuple[float, float, float, float]] = []
    for line in year_lines:
        line_page = int(line.get("source_logical_page") or logical_page)
        if line_page != logical_page:
            continue
        bbox = tuple(line.get("bbox") or ())
        if len(bbox) != 4:
            continue
        try:
            x0, y0, x1, y1 = (float(value) for value in bbox)
        except (TypeError, ValueError):
            continue
        if x1 <= x0 or y1 <= y0:
            continue
        boxes.append((x0, y0, x1, y1))
    if not boxes:
        return None
    ordered_x0 = sorted(box[0] for box in boxes)
    ordered_x1 = sorted(box[2] for box in boxes)
    middle = len(boxes) // 2
    return [
        ordered_x0[middle],
        min(box[1] for box in boxes),
        ordered_x1[middle],
        max(box[3] for box in boxes),
    ]


def _visual_month_col_bands(
    month_cols: list[dict[str, Any]],
    *,
    page_image: Any | None,
    page_width: float | None,
    page_height: float | None,
    y0: float,
    y1: float,
    year_column_bbox: Iterable[Any] | None = None,
    require_physical_month_ownership: bool = False,
    max_left_shift_months: float = 1.10,
    max_right_shift_months: float = 0.55,
    prefer_validated_header_lattice: bool = False,
    retain_validated_header_on_residual: bool = False,
    allow_unowned_header_fallback: bool = True,
    max_residual_shift_months: float | None = None,
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
        if require_physical_month_ownership and not allow_unowned_header_fallback:
            return [], {
                "source": "rejected_month_geometry",
                "usable": False,
                "offset": 0.0,
                "reason": "physical_month_column_ownership_unavailable",
                **geometry,
            }
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
            if require_physical_month_ownership and not allow_unowned_header_fallback:
                return [], {
                    "source": "rejected_month_geometry",
                    "usable": False,
                    "offset": 0.0,
                    "reason": "physical_month_column_ownership_projection_too_short",
                    **geometry,
                }
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
        start_offsets = np.linspace(
            -max(0.2, max_left_shift_months) * step,
            max(0.2, max_right_shift_months) * step,
            145,
        )
        end_offsets = np.linspace(
            -1.10 * step,
            max(0.2, max_right_shift_months) * step,
            129,
        )
        best_score = -1.0
        best_start, best_end = start, end
        best_strengths: Any = None
        candidates: list[tuple[float, float, float, float]] = []
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
                header_distance = (abs(candidate_start - start) + abs(candidate_end - end)) / max(step, 1e-6)
                if prefer_validated_header_lattice or require_physical_month_ownership:
                    candidates.append(
                        (
                            score,
                            candidate_start,
                            candidate_end,
                            header_distance,
                        )
                    )
                if score > best_score:
                    best_score = score
                    best_start, best_end = candidate_start, candidate_end
                    best_strengths = strengths
        ownership_selected = False
        ownership_audit: dict[str, Any] = {}
        year_bbox = tuple(year_column_bbox or ())
        if require_physical_month_ownership and len(year_bbox) == 4 and candidates:
            try:
                year_x0, year_x1 = float(year_bbox[0]), float(year_bbox[2])
            except (TypeError, ValueError):
                year_x0 = year_x1 = 0.0
            rule_floor = max(0.04, float(np.median(projection)) * 1.5)
            owned_candidates: list[tuple[float, float, float, float, int, float, float, float]] = []
            for score, candidate_start, candidate_end, header_distance in candidates:
                candidate_step = (candidate_end - candidate_start) / 12.0
                if not 0.80 * step <= candidate_step <= 1.20 * step:
                    continue
                predicted_year_left = candidate_start - candidate_step
                if predicted_year_left < 0.0:
                    continue
                # OCR text may sit against (or slightly outside) either edge
                # of the printed year cell.  Ownership needs the year glyph
                # box to touch that physical interval, not to be centred in it.
                if year_x1 < predicted_year_left:
                    ownership_error = (predicted_year_left - year_x1) / max(candidate_step, 1e-6)
                elif year_x0 > candidate_start:
                    ownership_error = (year_x0 - candidate_start) / max(candidate_step, 1e-6)
                else:
                    ownership_error = 0.0
                if ownership_error > 0.40:
                    continue
                year_width = max(1e-6, year_x1 - year_x0)
                year_glyph_left_of_month_coverage = (
                    max(
                        0.0,
                        min(candidate_start, year_x1) - year_x0,
                    )
                    / year_width
                )
                # Merely touching the predicted year interval is insufficient:
                # the classic year+months-1..11 lattice places its first
                # "month" rule through the printed year glyph.  A physical
                # month lattice must leave most of that glyph on the year side.
                if year_glyph_left_of_month_coverage < 0.72:
                    continue
                month_positions = np.linspace(
                    candidate_start * sx,
                    candidate_end * sx,
                    13,
                )
                month_indices = np.clip(
                    np.rint(month_positions).astype(int),
                    0,
                    image_width - 1,
                )
                month_strengths = projection[month_indices]
                year_left_index = int(np.clip(round(predicted_year_left * sx), 0, image_width - 1))
                year_left_strength = float(projection[year_left_index])
                month_rule_hits = int((month_strengths >= rule_floor).sum())
                if month_rule_hits < 11 or year_left_strength < rule_floor:
                    continue
                owned_candidates.append(
                    (
                        score + year_left_strength,
                        candidate_start,
                        candidate_end,
                        header_distance,
                        month_rule_hits,
                        ownership_error,
                        year_left_strength,
                        year_glyph_left_of_month_coverage,
                    )
                )
            if owned_candidates:
                (
                    _owned_score,
                    best_start,
                    best_end,
                    _best_header_distance,
                    owned_rule_hits,
                    owned_error,
                    owned_year_rule_strength,
                    owned_year_glyph_coverage,
                ) = max(
                    owned_candidates,
                    key=lambda candidate: (
                        candidate[4],
                        candidate[0],
                        -candidate[5],
                        candidate[7],
                        -candidate[3],
                    ),
                )
                best_positions = np.linspace(best_start * sx, best_end * sx, 13)
                best_indices = np.clip(np.rint(best_positions).astype(int), 0, image_width - 1)
                best_strengths = projection[best_indices]
                best_score = float(best_strengths.sum())
                ownership_selected = True
                ownership_audit = {
                    "selection_basis": "year_plus_twelve_rule_ownership",
                    "year_column_ownership_error": round(owned_error, 4),
                    "owned_month_rule_hits": owned_rule_hits,
                    "owned_year_left_rule_strength": round(
                        owned_year_rule_strength,
                        4,
                    ),
                    "year_glyph_left_of_month_coverage": round(
                        owned_year_glyph_coverage,
                        4,
                    ),
                }
        # Ownership constrains a *visual* replacement.  If the projection does
        # not establish a table lattice at all, keep the pre-existing header
        # geometry; there is no generic visual window to accept or reject.
        preliminary_baseline_per_rule = float(np.median(projection))
        preliminary_rule_floor = max(0.04, preliminary_baseline_per_rule * 1.5)
        preliminary_rule_hits = (
            int((best_strengths >= preliminary_rule_floor).sum()) if best_strengths is not None else 0
        )
        preliminary_baseline = preliminary_baseline_per_rule * 13.0
        if best_score < max(0.5, preliminary_baseline * 1.05) or preliminary_rule_hits < 10:
            if require_physical_month_ownership and not allow_unowned_header_fallback:
                return [], {
                    "source": "rejected_month_geometry",
                    "usable": False,
                    "offset": 0.0,
                    "reason": "physical_month_column_ownership_unproven",
                    "rule_hits": preliminary_rule_hits,
                    **geometry,
                }
            return month_cols, {
                "source": "header_geometry",
                "usable": True,
                "offset": 0.0,
                "reason": "vertical_rule_lattice_not_decisive",
                "rule_hits": preliminary_rule_hits,
                **geometry,
            }
        if require_physical_month_ownership and not ownership_selected and retain_validated_header_on_residual:
            return month_cols, {
                "source": "header_geometry",
                "usable": True,
                "offset": 0.0,
                "reason": "physical_rule_ownership_unavailable_exact_header_retained",
                **geometry,
            }
        if require_physical_month_ownership and not ownership_selected:
            return [], {
                "source": "rejected_month_geometry",
                "usable": False,
                "offset": 0.0,
                "reason": "physical_month_column_ownership_unproven",
                **geometry,
            }
        if prefer_validated_header_lattice and candidates and not ownership_selected:
            # A repayment table has fourteen vertical rules when its year column
            # precedes the twelve month cells.  Two thirteen-rule lattices can
            # then have the same (or nearly the same) projection score.  The
            # left-to-right search used to choose the year rule plus months
            # 1-11, shifting every source crop one cell left.  When the month
            # header was independently validated, use it as the tie-breaker.
            near_tie_tolerance = max(0.02, best_score * 0.02)
            near_ties = [candidate for candidate in candidates if candidate[0] >= best_score - near_tie_tolerance]
            (
                best_score,
                best_start,
                best_end,
                _best_header_distance,
            ) = min(
                near_ties,
                key=lambda candidate: (candidate[3], -candidate[0]),
            )
            best_positions = np.linspace(best_start * sx, best_end * sx, 13)
            best_indices = np.clip(np.rint(best_positions).astype(int), 0, image_width - 1)
            best_strengths = projection[best_indices]
        best_offset = best_start - start
        right_offset = best_end - end
        residual_shift_months = max(abs(best_offset), abs(right_offset)) / max(step, 1e-6)
        if (
            max_residual_shift_months is not None
            and residual_shift_months > max_residual_shift_months
            and not ownership_selected
        ):
            if retain_validated_header_on_residual:
                # The visual projection is only a refinement of an already
                # validated 1..12 header lattice.  A missing outside rule can
                # make the strongest visual lattice sit exactly one month to
                # the left or right; rejecting the refinement must not also
                # discard the independently exact header cells.  Retain the
                # header geometry and let the existing row/cell contracts
                # decide whether individual business values are usable.
                return month_cols, {
                    "source": "header_geometry",
                    "usable": True,
                    "offset": 0.0,
                    "rejected_visual_offset": round(best_offset, 4),
                    "rejected_visual_right_offset": round(right_offset, 4),
                    "residual_shift_months": round(residual_shift_months, 4),
                    "reason": "visual_lattice_residual_rejected_header_retained",
                    **geometry,
                }
            return [], {
                "source": "rejected_month_geometry",
                "usable": False,
                "offset": round(best_offset, 4),
                "right_offset": round(right_offset, 4),
                "residual_shift_months": round(residual_shift_months, 4),
                "reason": "month_lattice_residual_shift_exceeds_bound",
                **geometry,
            }
        # A few glyph strokes are not a table lattice.  Require sustained
        # evidence at most of the thirteen equally spaced vertical rules.
        baseline_per_rule = float(np.median(projection))
        rule_floor = max(0.04, baseline_per_rule * 1.5)
        rule_hits = int((best_strengths >= rule_floor).sum()) if best_strengths is not None else 0
        baseline = baseline_per_rule * 13.0
        if best_score < max(0.5, baseline * 1.05) or rule_hits < 10:
            if require_physical_month_ownership and not allow_unowned_header_fallback:
                return [], {
                    "source": "rejected_month_geometry",
                    "usable": False,
                    "offset": 0.0,
                    "reason": "physical_month_column_ownership_unproven",
                    "rule_hits": rule_hits,
                    **geometry,
                }
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
            if require_physical_month_ownership and not allow_unowned_header_fallback:
                return [], {
                    "source": "rejected_month_geometry",
                    "usable": False,
                    "offset": 0.0,
                    "reason": "owned_month_lattice_refinement_rejected",
                    **geometry,
                }
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
            "right_offset": round(right_offset, 4),
            "residual_shift_months": round(residual_shift_months, 4),
            **(
                ownership_audit
                if ownership_selected
                else {"selection_basis": "validated_header_near_tie"}
                if prefer_validated_header_lattice
                else {}
            ),
            "score": round(best_score, 4),
            "rule_hits": rule_hits,
            **refined_geometry,
        }
    except Exception as exc:
        if require_physical_month_ownership and not allow_unowned_header_fallback:
            return [], {
                "source": "rejected_month_geometry",
                "usable": False,
                "offset": 0.0,
                "reason": "physical_month_column_ownership_projection_failed",
                "error_type": type(exc).__name__,
                **geometry,
            }
        return month_cols, {
            "source": "header_geometry",
            "usable": True,
            "offset": 0.0,
            "reason": "vertical_rule_projection_failed",
            "error_type": type(exc).__name__,
            **geometry,
        }


def _visual_month_col_bands_in_registered_plane(
    month_cols: list[dict[str, Any]],
    *,
    source_lines: Iterable[Mapping[str, Any]],
    base_page: int,
    base_page_width: float | None,
    base_page_height: float | None,
    page_image: Any | None,
    page_image_resolver: Any | None,
    y0: float,
    y1: float,
    year_column_bbox: Iterable[Any] | None,
    cache: dict[Any, tuple[list[dict[str, Any]], dict[str, Any]]] | None = None,
    **strategy_options: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the unchanged visual strategy in raw coordinates, then register it.

    All row/header contributors must prove one origin and affine.  An unknown
    inverse disables visual evidence, but retains the strategy's existing
    header-only fallback policy.
    """

    def unavailable(reason: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        cols, audit = _visual_month_col_bands(
            month_cols,
            page_image=None,
            page_width=base_page_width,
            page_height=base_page_height,
            y0=y0,
            y1=y1,
            year_column_bbox=year_column_bbox,
            **strategy_options,
        )
        return cols, {**audit, "visual_coordinate_unavailable_reason": reason}

    rows = [dict(line) for line in source_lines]
    if not rows:
        return unavailable("source_coordinate_owner_missing")
    logical_pages = {
        int(line.get("source_logical_page") or line.get("coordinate_logical_page") or base_page)
        for line in rows
    }
    if len(logical_pages) != 1:
        return unavailable("source_coordinate_pages_disagree")
    logical_page = next(iter(logical_pages))
    for line in rows:
        if line.get("coordinate_logical_page") is not None:
            continue
        legacy_source_bbox = _local_page_bbox(
            tuple(float(value) for value in line["bbox"]),
            logical_page=logical_page,
            base_page=base_page,
            base_page_height=base_page_height,
            coordinate_status=str(line.get("coordinate_status") or ""),
        )
        if legacy_source_bbox is None:
            return unavailable("source_coordinate_stack_unproven")
        line["source_logical_page"] = logical_page
        line["coordinate_logical_page"] = logical_page
        line["source_origin_logical_page"] = logical_page
        line["source_bbox"] = legacy_source_bbox
        if logical_page != base_page:
            line["coordinate_status"] = "cross_page_y_shift"
    probe_bbox = [
        min(float(line["bbox"][0]) for line in rows),
        min(float(line["bbox"][1]) for line in rows),
        max(float(line["bbox"][2]) for line in rows),
        max(float(line["bbox"][3]) for line in rows),
    ]
    provenance = _synthesized_row_provenance(rows, bbox=probe_bbox, page=logical_page)
    if provenance.get("coordinate_logical_page") is not None:
        raw_probe_bbox = provenance.get("source_bbox")
        origin_page = provenance.get("source_origin_logical_page")
        if not isinstance(raw_probe_bbox, list) or origin_page is None:
            return unavailable("source_coordinate_affine_unproven")
    else:
        origin_page = logical_page
        raw_probe_bbox = _local_page_bbox(
            tuple(probe_bbox),
            logical_page=logical_page,
            base_page=base_page,
            base_page_height=base_page_height,
            coordinate_status=str(provenance.get("coordinate_status") or ""),
        )
        if raw_probe_bbox is None:
            return unavailable("source_coordinate_stack_unproven")
    sx0, sy0, sx1, sy1 = (float(value) for value in raw_probe_bbox)
    cx0, cy0, cx1, cy1 = probe_bbox
    if sx1 <= sx0 or sy1 <= sy0:
        return unavailable("source_coordinate_affine_degenerate")
    scale_x = (cx1 - cx0) / (sx1 - sx0)
    scale_y = (cy1 - cy0) / (sy1 - sy0)
    offset_x = cx0 - scale_x * sx0
    offset_y = cy0 - scale_y * sy0
    if (
        not all(isfinite(value) for value in (scale_x, scale_y, offset_x, offset_y))
        or scale_x <= 0.0
        or scale_y <= 0.0
    ):
        return unavailable("source_coordinate_affine_invalid")

    def raw_box(value: Iterable[Any]) -> list[float]:
        x0, box_y0, x1, box_y1 = (float(item) for item in value)
        return [
            (x0 - offset_x) / scale_x,
            (box_y0 - offset_y) / scale_y,
            (x1 - offset_x) / scale_x,
            (box_y1 - offset_y) / scale_y,
        ]

    raw_cols = [{**col, "bbox": raw_box(col["bbox"])} for col in month_cols]
    raw_year_bbox = raw_box(year_column_bbox) if year_column_bbox is not None else None
    context = page_image_resolver(int(origin_page)) if page_image_resolver is not None else None
    if isinstance(context, dict):
        raw_image = context.get("image")
        raw_width = float(context.get("page_width") or base_page_width or 0.0)
        raw_height = float(context.get("page_height") or base_page_height or 0.0)
    else:
        raw_image = page_image if int(origin_page) == base_page else None
        raw_width = base_page_width
        raw_height = base_page_height
    raw_y0 = max(0.0, (y0 - offset_y) / scale_y)
    raw_y1 = min(float(raw_height or 0.0), (y1 - offset_y) / scale_y)
    image_shape = tuple(int(value) for value in getattr(raw_image, "shape", ()))
    cache_key = (
        int(origin_page),
        logical_page,
        id(raw_image),
        raw_width,
        raw_height,
        image_shape,
        (round(raw_y0, 6), round(raw_y1, 6)),
        tuple(round(value, 6) for value in (scale_x, scale_y, offset_x, offset_y)),
        tuple(round(float(value), 4) for col in raw_cols for value in col["bbox"]),
        tuple(round(value, 4) for value in raw_year_bbox) if raw_year_bbox else (),
        tuple(sorted(strategy_options.items())),
    )
    if cache is not None and cache_key in cache:
        return deepcopy(cache[cache_key])
    raw_bands, audit = _visual_month_col_bands(
        raw_cols,
        page_image=raw_image,
        page_width=raw_width,
        page_height=raw_height,
        y0=raw_y0,
        y1=raw_y1,
        year_column_bbox=raw_year_bbox,
        **strategy_options,
    )
    registered_bands = []
    for col in raw_bands:
        x0, box_y0, x1, box_y1 = (float(value) for value in col["bbox"])
        registered_bands.append(
            {
                **col,
                "bbox": [
                    offset_x + x0 * scale_x,
                    offset_y + box_y0 * scale_y,
                    offset_x + x1 * scale_x,
                    offset_y + box_y1 * scale_y,
                ],
            }
        )
    audit = dict(audit)
    for key in (
        "offset", "right_offset", "rejected_visual_offset", "rejected_visual_right_offset",
        "total_width", "median_cell_width",
    ):
        if isinstance(audit.get(key), (int, float)):
            audit[key] = float(audit[key]) * scale_x
    if isinstance(audit.get("median_header_height"), (int, float)):
        audit["median_header_height"] = float(audit["median_header_height"]) * scale_y
    if isinstance(audit.get("cell_aspect"), (int, float)):
        audit["cell_aspect"] = float(audit["cell_aspect"]) * scale_x / scale_y
    audit.update(
        {
            "logical_page": logical_page,
            "source_logical_page": int(origin_page),
            "coordinate_system": "pdf_points_top_left",
            "visual_source_to_coordinate_affine": {
                "scale_x": scale_x,
                "scale_y": scale_y,
                "offset_x": offset_x,
                "offset_y": offset_y,
            },
        }
    )
    result = (registered_bands, audit)
    if cache is not None:
        cache[cache_key] = deepcopy(result)
    return result


def _accepted_month_geometry_provenance(audit: Mapping[str, Any]) -> dict[str, Any]:
    """Return compact, serializable proof for one accepted month lattice."""

    if audit.get("usable") is False:
        return {}
    retained_keys = (
        "selection_basis",
        "source",
        "rule_hits",
        "owned_month_rule_hits",
        "owned_year_left_rule_strength",
        "year_column_ownership_error",
        "year_glyph_left_of_month_coverage",
        "offset",
        "right_offset",
        "residual_shift_months",
        "reason",
        "table_id",
        "source_table_id",
        "continuation_logical_page",
        "vertical_rule_count",
        "rule_count",
        "horizontal_rule_count",
        "column_count",
        "month_column_count",
        "status_row_index",
        "amount_row_index",
        "year_anchor_row_index",
        "year_anchor_mode",
        "year_row_span",
        "active_cell_geometry_exact",
        "active_cell_rule_derived_count",
        "coordinate_system",
        "value_inputs_used",
        "corroborated_by_source_table_geometry",
        "ambiguous_visual_geometry_superseded",
        "conflicting_visual_geometry_superseded",
        "exact_source_atom_geometry_months",
        "exact_source_status_atom_geometry_months",
        "exact_source_amount_atom_geometry_months",
        "source_table_comparison",
        "calibrated_from_source_table_geometry",
        "visual_selection_basis",
        "visual_owned_month_rule_hits",
        "visual_residual_shift_months",
        "logical_page",
        "source_logical_page",
        "source_page",
        "visual_source_to_coordinate_affine",
    )
    provenance = {key: audit[key] for key in retained_keys if key in audit and audit[key] is not None}
    return {"geometry_provenance": provenance} if provenance else {}


def _rejected_month_geometry_provenance(audit: Mapping[str, Any]) -> dict[str, Any]:
    """Return compact, value-free evidence for a rejected month lattice."""

    if audit.get("usable") is not False:
        return {}
    retained_keys = (
        "source",
        "reason",
        "source_table_id",
        "source_table_comparison",
        "logical_page",
        "value_inputs_used",
        "visual_selection_basis",
        "visual_owned_month_rule_hits",
        "visual_residual_shift_months",
    )
    provenance = {key: audit[key] for key in retained_keys if key in audit and audit[key] is not None}
    return {"geometry_rejection": provenance} if provenance else {}


def _month_geometry_edges(
    month_cols: Iterable[Mapping[str, Any]],
) -> tuple[float, ...] | None:
    """Return the thirteen ordered x edges of one 12-month band set."""

    by_month: dict[int, tuple[float, float]] = {}
    for col in month_cols:
        try:
            month = int(col.get("header") or col.get("index") or 0)
            bbox = tuple(float(value) for value in col.get("bbox") or ())
        except (TypeError, ValueError):
            return None
        if month not in range(1, 13) or len(bbox) != 4 or bbox[2] <= bbox[0]:
            return None
        by_month[month] = (bbox[0], bbox[2])
    if set(by_month) != set(range(1, 13)):
        return None
    ordered = [by_month[month] for month in range(1, 13)]
    if any(abs(left[1] - right[0]) > 2.0 for left, right in zip(ordered, ordered[1:])):
        return None
    return (ordered[0][0], *(band[1] for band in ordered))


def _source_lattice_month_cols(
    lattice: SourceTableMonthLattice,
) -> list[dict[str, Any]]:
    """Adapt a value-free source lattice to repayment-grid month bands."""

    return [
        {
            **band,
            "header": str(month),
            "geometry_status": "exact",
            "geometry_source": "source_table_geometry",
        }
        for month, band in enumerate(lattice.month_col_bands(), start=1)
    ]


def _month_geometry_planes_agree(
    visual_cols: Iterable[Mapping[str, Any]],
    source_cols: Iterable[Mapping[str, Any]],
) -> bool:
    """Require tight x-edge agreement before corroborating two geometry planes."""

    visual_edges = _month_geometry_edges(visual_cols)
    source_edges = _month_geometry_edges(source_cols)
    if visual_edges is None or source_edges is None:
        return False
    source_widths = [right - left for left, right in zip(source_edges, source_edges[1:])]
    # The visual-rule detector and the sealed physical-table reconstructor can
    # differ by a small page-local calibration scale even when they identify
    # the same thirteen month edges.  Keep agreement far below a half-cell
    # (which could change calendar ownership), while allowing the bounded
    # sub-cell drift observed across a full page-width lattice.
    tolerance = min(3.0, max(0.75, sorted(source_widths)[6] * 0.10))
    return all(abs(visual - source) <= tolerance for visual, source in zip(visual_edges, source_edges, strict=True))


def _candidate_b_visual_lattice_needs_source_table(
    audit: Mapping[str, Any],
) -> bool:
    """Reject underdetermined visual ownership unless a source table binds it.

    A year-plus-twelve table has thirteen month-column boundary rules.  Seeing
    only twelve leaves an adjacent-lattice interpretation open.  A displacement
    near half a month is similarly non-decisive.  Neither condition may remain
    an exact Candidate-B geometry claim on visual evidence alone.
    """

    if (
        audit.get("usable") is False
        or audit.get("source") != "vertical_rule_projection"
        or audit.get("selection_basis") != "year_plus_twelve_rule_ownership"
    ):
        return False
    try:
        owned_rule_hits = int(audit.get("owned_month_rule_hits") or 0)
        residual_shift = abs(float(audit.get("residual_shift_months") or 0.0))
    except (TypeError, ValueError):
        return True
    return owned_rule_hits < 13 or residual_shift >= 0.45


def _joined_continuation_row_y(
    boxes: Iterable[Iterable[float]],
    *,
    logical_page: int,
    base_page: int,
    base_page_height: float | None,
) -> tuple[float, float] | None:
    """Register a source-table row band in the joined base-page stack."""

    parsed: list[tuple[float, float, float, float]] = []
    for raw in boxes:
        try:
            bbox = tuple(float(value) for value in raw)
        except (TypeError, ValueError):
            return None
        if len(bbox) != 4 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            return None
        parsed.append(bbox)
    if not parsed:
        return None
    shift = 0.0
    if logical_page != base_page:
        if not base_page_height or float(base_page_height) <= 0.0:
            return None
        shift = float(base_page_height)
    return (
        min(box[1] for box in parsed) + shift,
        max(box[3] for box in parsed) + shift,
    )


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
            cluster_center = sum((float(item["bbox"][1]) + float(item["bbox"][3])) / 2.0 for item in cluster) / len(
                cluster
            )
            cluster_height = max(max(1.0, float(item["bbox"][3]) - float(item["bbox"][1])) for item in cluster)
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
        key=lambda cluster: (
            sum((float(item["bbox"][1]) + float(item["bbox"][3])) / 2.0 for item in cluster) / len(cluster)
        ),
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
    row_bbox = [
        min(float(line["bbox"][0]) for line in row_lines),
        min(float(line["bbox"][1]) for line in row_lines),
        max(float(line["bbox"][2]) for line in row_lines),
        max(float(line["bbox"][3]) for line in row_lines),
    ]
    return {
        "idx": min(int(line["idx"]) for line in row_lines),
        "text": "".join(token.text for token in status_tokens),
        "bbox": row_bbox,
        "confidence": min(float(line.get("confidence") or 0.0) for line in row_lines),
        **_synthesized_row_provenance(row_lines, bbox=row_bbox, page=year_page),
        "candidate_b_status_tokens": status_tokens,
        "status_source_line_indices": [int(line["idx"]) for line in row_lines],
    }


def _candidate_b_exact_active_source_status_row(
    tokens: list[OCRToken],
    year_line: dict[str, Any],
    *,
    source_lattice: SourceTableMonthLattice,
    active_months: list[int],
    status_charset: set[str],
    page: int,
    base_page_height: float | None,
    fallback_status_tokens: list[OCRToken] | None = None,
) -> dict[str, Any] | None:
    """Recover independently positioned words one active source cell at a time.

    A noisy merged OCR line may be unusable even though each active month still
    has a clean one-word status observation.  The detached source table proves
    the exact year-plus-twelve lattice without contributing values.  A missing,
    duplicate, or source-reused token withholds only its affected month; it no
    longer discards independently proved sibling cells.
    """

    if not active_months or len(set(active_months)) != len(active_months):
        return None
    logical_page = int(source_lattice.logical_page)
    year_logical_page = int(year_line.get("source_logical_page") or page)
    if logical_page != year_logical_page:
        return None
    source_cells: dict[int, tuple[float, float, float, float]] = {}
    for month in active_months:
        if not 1 <= month <= 12:
            continue
        registered = _stacked_page_bbox(
            source_lattice.month_bboxes[month - 1],
            logical_page=logical_page,
            base_page=page,
            base_page_height=base_page_height,
        )
        if registered is None:
            return None
        source_cells[month] = registered
    if set(source_cells) != set(active_months):
        return None

    def owned_tokens_by_month(
        observations: Iterable[OCRToken],
    ) -> dict[int, list[OCRToken]]:
        owned: dict[int, list[OCRToken]] = {month: [] for month in active_months}
        for token in observations:
            if str(token.coordinate_system or "") != "pdf_points_top_left":
                continue
            chars = _candidate_b_status_chars(token.text, status_charset)
            if len(chars) != 1 or chars[0] not in status_charset:
                continue
            tx0, ty0, tx1, ty1 = (float(value) for value in token.bbox)
            token_width = tx1 - tx0
            token_height = ty1 - ty0
            if token_width <= 0.0 or token_height <= 0.0:
                continue
            owners: list[int] = []
            for month, (cx0, cy0, cx1, cy1) in source_cells.items():
                x_overlap = max(0.0, min(tx1, cx1) - max(tx0, cx0))
                y_overlap = max(0.0, min(ty1, cy1) - max(ty0, cy0))
                cell_height = cy1 - cy0
                if (
                    x_overlap / token_width >= 0.80
                    and y_overlap / min(token_height, cell_height) >= 0.70
                    and cx0 <= token.center[0] <= cx1
                    and cy0 <= token.center[1] <= cy1
                ):
                    owners.append(month)
            if len(owners) == 1:
                owned[owners[0]].append(replace(token, text=chars[0]))
        return owned

    tokens_by_month = owned_tokens_by_month(tokens)
    fallback_by_month = owned_tokens_by_month(fallback_status_tokens or ())

    def unique_candidates(
        observations_by_month: Mapping[int, list[OCRToken]],
        *,
        source_key: Callable[[OCRToken], str],
    ) -> tuple[dict[int, OCRToken], set[int]]:
        candidates = {
            month: month_tokens[0]
            for month, month_tokens in observations_by_month.items()
            if len(month_tokens) == 1
        }
        ambiguous_months = {
            month
            for month, month_tokens in observations_by_month.items()
            if month_tokens and len(month_tokens) != 1
        }
        key_months: dict[str, list[int]] = {}
        for month, token in candidates.items():
            key_months.setdefault(source_key(token), []).append(month)
        ambiguous_keys = {
            key for key, months in key_months.items() if not key or len(months) != 1
        }
        ambiguous_months.update(
            month for key in ambiguous_keys for month in key_months.get(key, ())
        )
        return (
            {
                month: token
                for month, token in candidates.items()
                if source_key(token) not in ambiguous_keys
            },
            ambiguous_months,
        )

    fallback_candidates, fallback_ambiguous_months = unique_candidates(
        fallback_by_month,
        # Split characters from one corrected row have distinct token IDs even
        # though they intentionally share the row's source token ID.
        source_key=lambda token: str(token.token_id or ""),
    )
    exact_tokens_by_month = {
        month: [
            token
            for token in month_tokens
            if token.source in _EXACT_SOURCE_STATUS_CELL_SOURCES
        ]
        for month, month_tokens in tokens_by_month.items()
    }
    exact_candidates, exact_ambiguous_months = unique_candidates(
        exact_tokens_by_month,
        source_key=lambda token: str(token.source_token_id or token.token_id or ""),
    )
    ordinary_tokens_by_month = {
        month: [
            token
            for token in month_tokens
            if token.source not in _EXACT_SOURCE_STATUS_CELL_SOURCES
        ]
        for month, month_tokens in tokens_by_month.items()
    }
    ordinary_candidates, ordinary_ambiguous_months = unique_candidates(
        ordinary_tokens_by_month,
        source_key=lambda token: str(token.source_token_id or token.token_id or ""),
    )

    # Repair is fill-only for a missing cell. Two provenance-complete corrected
    # planes must agree before either can populate a nonempty month.
    conflict_months = {
        month
        for month in active_months
        if month in fallback_ambiguous_months
        or month in exact_ambiguous_months
        or month in ordinary_ambiguous_months
        or (
            month in fallback_candidates
            and month in exact_candidates
            and fallback_candidates[month].text != exact_candidates[month].text
        )
    }
    selected_by_month: dict[int, OCRToken] = {}
    for month in active_months:
        if month in conflict_months:
            continue
        selected = (
            fallback_candidates.get(month)
            or exact_candidates.get(month)
            or ordinary_candidates.get(month)
        )
        if selected is not None:
            selected_by_month[month] = selected
    unresolved_months = set(active_months).difference(selected_by_month)
    selected_months = sorted(selected_by_month)
    selected = [selected_by_month[month] for month in selected_months]
    # Structural ownership is independent of status-value availability.  The
    # full lattice remains available to sibling amount fields even when every
    # status atom is absent or conflicted.
    active_cells = [source_cells[month] for month in active_months]
    conflict_observations = {
        str(month): {
            "fallback": sorted({token.text for token in fallback_by_month.get(month, ())}),
            "exact_source_cell": sorted(
                {token.text for token in exact_tokens_by_month.get(month, ())}
            ),
            "ordinary": sorted(
                {token.text for token in ordinary_tokens_by_month.get(month, ())}
            ),
        }
        for month in sorted(conflict_months)
    }
    row_bbox = [
        min(cell[0] for cell in active_cells),
        min(cell[1] for cell in active_cells),
        max(cell[2] for cell in active_cells),
        max(cell[3] for cell in active_cells),
    ]
    return {
        "idx": int(year_line["idx"]),
        "text": "".join(token.text for token in selected),
        "bbox": row_bbox,
        "confidence": (
            min(float(token.confidence) for token in selected)
            if selected
            else 0.0
        ),
        **_source_lattice_row_provenance(
            source_lattice,
            bbox=row_bbox,
            base_page=page,
            base_page_height=base_page_height,
        ),
        "candidate_b_status_tokens": selected,
        "candidate_b_status_token_months": selected_months,
        "candidate_b_exact_source_status_months": sorted(exact_candidates),
        "status_source_line_indices": [],
        "candidate_b_status_unresolved_months": sorted(unresolved_months),
        "candidate_b_status_conflict_months": sorted(conflict_months),
        "candidate_b_status_conflict_observations": conflict_observations,
        "candidate_b_status_row_repair": ("exact_active_source_lattice_token_ownership"),
    }


def _candidate_b_exact_source_zero_amount_tokens(
    tokens: list[OCRToken],
    *,
    source_lattice: SourceTableMonthLattice,
    active_months: Iterable[int],
    page: int,
    base_page_height: float | None,
) -> dict[int, OCRToken]:
    """Bind exact singleton ``0`` atoms to exact source amount cells."""

    months = tuple(dict.fromkeys(int(month) for month in active_months))
    if not months or any(month < 1 or month > 12 for month in months):
        return {}
    logical_page = int(source_lattice.logical_page)
    source_cells: dict[int, tuple[float, float, float, float]] = {}
    for month in months:
        registered = _stacked_page_bbox(
            source_lattice.amount_bboxes[month - 1],
            logical_page=logical_page,
            base_page=page,
            base_page_height=base_page_height,
        )
        if registered is None:
            return {}
        source_cells[month] = registered
    owned: dict[int, list[OCRToken]] = {month: [] for month in months}
    for token in tokens:
        if (
            token.source not in _EXACT_SOURCE_AMOUNT_CELL_SOURCES
            or token.text.strip() != "0"
            or token.coordinate_system != "pdf_points_top_left"
        ):
            continue
        tx0, ty0, tx1, ty1 = (float(value) for value in token.bbox)
        token_width = tx1 - tx0
        token_height = ty1 - ty0
        if token_width <= 0.0 or token_height <= 0.0:
            continue
        owners: list[int] = []
        for month, (cx0, cy0, cx1, cy1) in source_cells.items():
            x_overlap = max(0.0, min(tx1, cx1) - max(tx0, cx0))
            y_overlap = max(0.0, min(ty1, cy1) - max(ty0, cy0))
            cell_height = cy1 - cy0
            if (
                x_overlap / token_width >= 0.80
                and y_overlap / min(token_height, cell_height) >= 0.70
                and cx0 <= token.center[0] <= cx1
                and cy0 <= token.center[1] <= cy1
            ):
                owners.append(month)
        if len(owners) == 1:
            owned[owners[0]].append(token)

    selected = {month: month_tokens[0] for month, month_tokens in owned.items() if len(month_tokens) == 1}
    source_key_months: dict[str, list[int]] = {}
    for month, token in selected.items():
        source_key = str(token.source_token_id or token.token_id or "")
        source_key_months.setdefault(source_key, []).append(month)
    reused_source_keys = {
        source_key
        for source_key, source_months in source_key_months.items()
        if not source_key or len(source_months) != 1
    }
    return {
        month: token
        for month, token in selected.items()
        if str(token.source_token_id or token.token_id or "") not in reused_source_keys
    }


def _candidate_b_merge_exact_source_zero_amounts(
    amount_pairing: dict[str, Any],
    tokens: list[OCRToken],
    year_line: dict[str, Any],
    *,
    source_lattice: SourceTableMonthLattice,
    active_months: list[int],
    source_geometry_months: set[int],
    page: int,
    base_page_height: float | None,
) -> dict[str, Any]:
    """Repair only missing/blank amount fields proved by exact zero atoms."""

    eligible_months = [month for month in active_months if month in source_geometry_months]
    exact_zeros = _candidate_b_exact_source_zero_amount_tokens(
        tokens,
        source_lattice=source_lattice,
        active_months=eligible_months,
        page=page,
        base_page_height=base_page_height,
    )
    cell_status = dict(amount_pairing.get("cell_status_by_month") or {})
    repairable_statuses = {
        "blank_amount_cell",
        "missing_amount_row",
        "non_immediate_amount_row",
    }
    repaired = {
        month: token
        for month, token in exact_zeros.items()
        if str(cell_status.get(str(month)) or "missing_amount_row") in repairable_statuses
    }
    if not repaired:
        return amount_pairing

    for month in repaired:
        cell_status[str(month)] = "exact"
    merged_tokens = list(amount_pairing.get("tokens") or ()) + [repaired[month] for month in sorted(repaired)]
    amount_line = amount_pairing.get("line")
    if not isinstance(amount_line, dict):
        logical_page = int(source_lattice.logical_page)
        boxes = [
            registered
            for month in sorted(repaired)
            if (
                registered := _stacked_page_bbox(
                    source_lattice.amount_bboxes[month - 1],
                    logical_page=logical_page,
                    base_page=page,
                    base_page_height=base_page_height,
                )
            )
            is not None
        ]
        if len(boxes) != len(repaired):
            return amount_pairing
        row_bbox = [
            min(float(box[0]) for box in boxes),
            min(float(box[1]) for box in boxes),
            max(float(box[2]) for box in boxes),
            max(float(box[3]) for box in boxes),
        ]
        amount_line = {
            "idx": int(year_line["idx"]),
            "text": " ".join("0" for _month in sorted(repaired)),
            "bbox": row_bbox,
            "confidence": min(float(token.confidence) for token in repaired.values()),
            **_source_lattice_row_provenance(
                source_lattice,
                bbox=row_bbox,
                base_page=page,
                base_page_height=base_page_height,
            ),
            "amount_source_line_indices": [],
        }

    complete = all(str(cell_status.get(str(month)) or "") == "exact" for month in active_months)
    return {
        **amount_pairing,
        "status": ("exact" if complete else "field_local_exact_source_zero_repair"),
        "row_relation": (
            "exact_source_zero_cells" if complete else str(amount_pairing.get("row_relation") or "unresolved")
        ),
        "line": amount_line,
        "tokens": merged_tokens,
        "cell_status_by_month": cell_status,
        "exact_source_zero_months": sorted(repaired),
    }


def _candidate_b_owned_source_lattice(
    source_tables: list[Mapping[str, Any]],
    *,
    logical_page: int,
    expected_year: int,
    active_months: Iterable[int],
    year_bbox: BBox,
    status_bbox: BBox | None,
) -> SourceTableMonthLattice | None:
    """Resolve the row before values, including one-sided singleton years.

    A damaged status cell can eliminate the correct pair from a year-only
    search and make the adjacent amount/status pair appear uniquely valid.
    A singleton year therefore needs the observed status band when available;
    a genuinely spanning year can still recover a noisy or absent text band.
    """

    months = tuple(active_months)
    year_lattice = resolve_unique_source_table_year_plus_twelve_ownership_from_year(
        source_tables,
        logical_page=logical_page,
        expected_year=expected_year,
        active_months=months,
        year_bbox=year_bbox,
    )
    if status_bbox is None:
        return year_lattice
    status_lattice = resolve_unique_source_table_year_plus_twelve_ownership(
        source_tables,
        logical_page=logical_page,
        expected_year=expected_year,
        active_months=months,
        year_bbox=year_bbox,
        status_bbox=status_bbox,
    )
    if status_lattice is not None:
        return status_lattice
    if (
        year_lattice is not None
        and year_lattice.provenance_dict().get("year_anchor_mode")
        == "target_bound_singleton_year_cell"
    ):
        return None
    return year_lattice


def _candidate_b_sparse_exact_source_status_row(
    tokens: list[OCRToken],
    year_line: dict[str, Any],
    *,
    source_tables: list[Mapping[str, Any]],
    logical_page: int,
    expected_year: int,
    active_months: list[int],
    year_bbox: BBox,
    status_charset: set[str],
    page: int,
    base_page_height: float | None,
    fallback_status_tokens: list[OCRToken] | None = None,
    status_bbox: BBox | None = None,
) -> dict[str, Any] | None:
    """Recover exact sibling cells when one damaged cell blocks a whole row.

    Each month must independently resolve to the same value-free native table
    row.  Only the jointly re-resolved subset is exposed to token ownership;
    ambiguous or merged cells remain unresolved and cannot lend geometry to a
    sibling.
    """

    def resolve(months: Iterable[int]) -> SourceTableMonthLattice | None:
        return _candidate_b_owned_source_lattice(
            source_tables,
            logical_page=logical_page,
            expected_year=expected_year,
            active_months=months,
            year_bbox=year_bbox,
            status_bbox=status_bbox,
        )

    lattices_by_month: dict[int, SourceTableMonthLattice] = {}
    identities: set[tuple[str, int, int, int, int]] = set()
    for month in active_months:
        lattice = resolve((month,))
        if lattice is None:
            continue
        identity = (
            lattice.table_id,
            lattice.logical_page,
            lattice.year_anchor_row_index,
            lattice.status_row_index,
            lattice.amount_row_index,
        )
        identities.add(identity)
        lattices_by_month[month] = lattice
    if not lattices_by_month or len(identities) != 1:
        return None

    geometry_months = sorted(lattices_by_month)
    aggregate_lattice = resolve(geometry_months)
    if aggregate_lattice is None:
        return None
    aggregate_identity = (
        aggregate_lattice.table_id,
        aggregate_lattice.logical_page,
        aggregate_lattice.year_anchor_row_index,
        aggregate_lattice.status_row_index,
        aggregate_lattice.amount_row_index,
    )
    if aggregate_identity not in identities:
        return None

    repaired = _candidate_b_exact_active_source_status_row(
        tokens,
        year_line,
        source_lattice=aggregate_lattice,
        active_months=geometry_months,
        status_charset=status_charset,
        page=page,
        base_page_height=base_page_height,
        fallback_status_tokens=fallback_status_tokens,
    )
    if repaired is None:
        return None
    unresolved = {
        int(value)
        for value in repaired.get("candidate_b_status_unresolved_months", ())
        if isinstance(value, int) and not isinstance(value, bool)
    }
    repaired["candidate_b_status_unresolved_months"] = sorted(
        set(active_months).difference(geometry_months) | unresolved
    )
    repaired["candidate_b_sparse_source_lattice"] = aggregate_lattice
    repaired["candidate_b_sparse_geometry_months"] = geometry_months
    repaired["candidate_b_status_row_repair"] = "exact_sparse_source_lattice_token_ownership"
    return repaired


def _numeric_amount_groups(text: Any) -> list[str]:
    """Return explicit numeric groups only; OCR prose is never an amount row."""

    value = str(text or "").strip()
    if not value or re.fullmatch(r"[0-9.,，\s]+", value) is None:
        return []
    return re.findall(r"\d+(?:[.,，]\d+)?", value)


def _undelimited_zero_run_matches_month_geometry(
    text: Any,
    *,
    months: list[int],
    cols_by_month: dict[int, dict[str, Any]],
    source_bbox: Iterable[Any] | None,
) -> bool:
    """Accept a merged zero run only when each glyph owns one month band."""

    compact = re.sub(r"\s+", "", str(text or ""))
    if len(months) < 2 or re.fullmatch(r"0+", compact) is None:
        return False
    if len(compact) != len(months) or months != sorted(months):
        return False
    if any(right != left + 1 for left, right in zip(months, months[1:])):
        return False
    bbox = list(source_bbox or ())
    if len(bbox) != 4:
        return False
    try:
        source_x0, source_x1 = float(bbox[0]), float(bbox[2])
    except (TypeError, ValueError):
        return False
    if source_x1 <= source_x0:
        return False

    step = (source_x1 - source_x0) / len(compact)
    resolved_months: list[int] = []
    for index in range(len(compact)):
        center = source_x0 + step * (index + 0.5)
        owners = [
            month
            for month in months
            if (col := cols_by_month.get(month)) is not None
            and float(col["bbox"][0]) <= center <= float(col["bbox"][2])
        ]
        if len(owners) != 1:
            return False
        resolved_months.append(owners[0])
    return resolved_months == months and len(set(resolved_months)) == len(months)


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
    source_bbox: Iterable[Any] | None = None,
    allow_geometry_proven_zero_run: bool = False,
) -> list[OCRToken] | None:
    """Bind one merged numeric sequence only when its cardinality is exact."""

    groups = _numeric_amount_groups(text)
    if not groups or not months:
        return None
    values: list[str]
    if len(groups) == len(months):
        values = groups
    elif allow_geometry_proven_zero_run and _undelimited_zero_run_matches_month_geometry(
        text,
        months=months,
        cols_by_month=cols_by_month,
        source_bbox=source_bbox,
    ):
        values = ["0"] * len(months)
    else:
        return None
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


def _candidate_b_year_block_pitch(
    year_line: dict[str, Any],
    *,
    year_lines: list[dict[str, Any]] | None,
    status_line: dict[str, Any] | None,
    page: int,
) -> float | None:
    """Estimate one canonical two-row year block without consulting OCR again."""

    year_text = str(year_line.get("text") or "").strip()
    year_match = _YEAR_RE.match(year_text)
    year_center = (float(year_line["bbox"][1]) + float(year_line["bbox"][3])) / 2.0
    year_page = int(year_line.get("source_logical_page") or page)
    pitch_candidates: list[float] = []
    if year_match is not None:
        year_value = int(year_match.group(0))
        for other in year_lines or ():
            if other is year_line or int(other.get("source_logical_page") or page) != year_page:
                continue
            other_match = _YEAR_RE.match(str(other.get("text") or "").strip())
            if other_match is None:
                continue
            year_delta = abs(int(other_match.group(0)) - year_value)
            if year_delta == 0:
                continue
            other_center = (float(other["bbox"][1]) + float(other["bbox"][3])) / 2.0
            per_year_pitch = abs(other_center - year_center) / year_delta
            if 10.0 <= per_year_pitch <= 90.0:
                pitch_candidates.append(per_year_pitch)

    status_gap: float | None = None
    if status_line is not None:
        status_center = (float(status_line["bbox"][1]) + float(status_line["bbox"][3])) / 2.0
        if status_center < year_center:
            status_gap = year_center - status_center

    if pitch_candidates:
        ordered = sorted(pitch_candidates)
        pitch = ordered[len(ordered) // 2]
        if status_gap is not None:
            # A distant same-year label from another account must not widen the
            # local amount slot past the adjacent canonical two-row block.
            pitch = min(pitch, max(16.0, 4.0 * status_gap))
        return pitch
    if status_gap is not None:
        # The status row occupies the half-row immediately above the printed
        # year/amount boundary.  This fallback is deliberately tighter than a
        # page-global pixel allowance.
        return min(90.0, max(12.0, 3.0 * status_gap))
    return None


def _candidate_b_fragment_months(
    line: dict[str, Any],
    *,
    month_cols: list[dict[str, Any]],
    active_months: list[int],
    median_month_width: float,
) -> list[int] | None:
    """Return a unique contiguous month interval owned by one numeric fragment."""

    lx0, _ly0, lx1, _ly1 = (float(value) for value in line["bbox"])
    line_width = max(1.0, lx1 - lx0)
    center_x = (lx0 + lx1) / 2.0
    if line_width <= median_month_width * 1.35:
        owners = [
            int(col["header"])
            for col in month_cols
            if int(col["header"]) in active_months and float(col["bbox"][0]) <= center_x <= float(col["bbox"][2])
        ]
        if len(owners) != 1:
            return None
        return owners

    spanned = [
        int(col["header"])
        for col in month_cols
        if int(col["header"]) in active_months
        and max(
            0.0,
            min(lx1, float(col["bbox"][2])) - max(lx0, float(col["bbox"][0])),
        )
        / min(
            line_width,
            max(1.0, float(col["bbox"][2]) - float(col["bbox"][0])),
        )
        >= 0.2
    ]
    if not spanned or any(right != left + 1 for left, right in zip(spanned, spanned[1:])):
        return None
    return spanned


def _candidate_b_unique_amount_fragment_cover(
    candidates: list[dict[str, Any]],
    year_line: dict[str, Any],
    *,
    year_lines: list[dict[str, Any]] | None,
    status_line: dict[str, Any] | None,
    month_cols: list[dict[str, Any]],
    active_months: list[int],
    page: int,
) -> dict[str, Any] | None:
    """Join split glyph boxes only inside one uniquely owned amount slot.

    OCR words from one printed row can have disjoint vertical glyph boxes.  A
    raw y-cluster count therefore is not sufficient evidence for two business
    rows.  This repair is intentionally closed-world: the canonical year pitch
    bounds the amount slot, while exact x ownership and cardinality prove each
    emitted value.  Uncovered cells remain blank and no zero is manufactured.
    """

    pitch = _candidate_b_year_block_pitch(
        year_line,
        year_lines=year_lines,
        status_line=status_line,
        page=page,
    )
    if pitch is None:
        return None
    year_center = (float(year_line["bbox"][1]) + float(year_line["bbox"][3])) / 2.0
    slot_y0 = year_center - max(2.0, 0.08 * pitch)
    slot_y1 = year_center + 0.48 * pitch
    slot_candidates: list[dict[str, Any]] = []
    for line in candidates:
        text = str(line.get("text") or "").strip()
        if re.fullmatch(r"20\d{2}", text):
            continue
        center_y = (float(line["bbox"][1]) + float(line["bbox"][3])) / 2.0
        if slot_y0 <= center_y <= slot_y1:
            slot_candidates.append(line)
    if not slot_candidates:
        return None

    centers = [(float(line["bbox"][1]) + float(line["bbox"][3])) / 2.0 for line in slot_candidates]
    if max(centers) - min(centers) > max(4.0, 0.35 * pitch):
        return None
    row_y0 = min(float(line["bbox"][1]) for line in slot_candidates)
    row_y1 = max(float(line["bbox"][3]) for line in slot_candidates)
    if row_y1 - row_y0 > max(10.0, 0.65 * pitch):
        return None

    cols_by_month = {int(col["header"]): col for col in month_cols}
    month_widths = [
        max(1.0, float(col["bbox"][2]) - float(col["bbox"][0]))
        for col in month_cols
        if int(col["header"]) in active_months
    ]
    if not month_widths:
        return None
    median_month_width = sorted(month_widths)[len(month_widths) // 2]
    tokens_by_month: dict[int, OCRToken] = {}
    previous_last_month: int | None = None
    row_lines = sorted(slot_candidates, key=lambda item: float(item["bbox"][0]))
    for line in row_lines:
        spanned = _candidate_b_fragment_months(
            line,
            month_cols=month_cols,
            active_months=active_months,
            median_month_width=median_month_width,
        )
        if spanned is None or previous_last_month is not None and spanned[0] <= previous_last_month:
            return None
        line_tokens = _amount_sequence_tokens(
            line.get("text"),
            months=spanned,
            cols_by_month=cols_by_month,
            y0=row_y0,
            y1=row_y1,
            page=page,
            prefix=f"ocr_p{page}_repay_amount_fragment_cover",
            confidence=float(line.get("confidence") or 0.0),
            source_line_index=int(line["idx"]),
            source_bbox=line.get("bbox"),
            allow_geometry_proven_zero_run=True,
        )
        if line_tokens is None or len(line_tokens) != len(spanned):
            return None
        for month, token in zip(spanned, line_tokens, strict=True):
            if month in tokens_by_month:
                return None
            tokens_by_month[month] = token
        previous_last_month = spanned[-1]

    if not tokens_by_month:
        return None
    row_bbox = [
        min(float(line["bbox"][0]) for line in row_lines),
        row_y0,
        max(float(line["bbox"][2]) for line in row_lines),
        row_y1,
    ]
    row_line = {
        "idx": min(int(line["idx"]) for line in row_lines),
        "text": " ".join(str(line.get("text") or "") for line in row_lines),
        "bbox": row_bbox,
        "confidence": min(float(line.get("confidence") or 0.0) for line in row_lines),
        **_synthesized_row_provenance(row_lines, bbox=row_bbox, page=page),
        "amount_source_line_indices": [int(line["idx"]) for line in row_lines],
    }
    return {
        "status": "exact",
        "row_relation": "aligned_or_immediate_after_year",
        "line": row_line,
        "tokens": [tokens_by_month[month] for month in sorted(tokens_by_month)],
        "source_line_indices": [int(line["idx"]) for line in row_lines],
        "observed_texts": [str(line.get("text") or "") for line in row_lines],
        "cell_status_by_month": {
            str(month): "exact" if month in tokens_by_month else "blank_amount_cell" for month in active_months
        },
    }


def _candidate_b_amount_row_pair(
    lines: list[dict[str, Any]],
    year_line: dict[str, Any],
    *,
    month_cols: list[dict[str, Any]],
    active_months: list[int],
    page: int,
    excluded_line_indices: set[int],
    year_lines: list[dict[str, Any]] | None = None,
    status_line: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve a unique immediate amount row from canonical static geometry.

    Candidate B accepts the year-line remainder or exactly one aligned/adjacent
    numeric y-band.  Every OCR word remains tied to its observed cell geometry;
    a second row, a non-immediate row, or two words in one month is unresolved.
    """

    cols_by_month = {int(col["header"]): col for col in month_cols}
    owned_active_months = [month for month in active_months if month in cols_by_month]
    unowned_active_months = sorted(set(active_months).difference(owned_active_months))

    def retain_month_geometry_ownership(pairing: dict[str, Any]) -> dict[str, Any]:
        if not unowned_active_months:
            return pairing
        cell_status_by_month = dict(pairing.get("cell_status_by_month") or {})
        cell_status_by_month.update(
            {str(month): "month_geometry_unowned" for month in unowned_active_months}
        )
        return {
            **pairing,
            "cell_status_by_month": cell_status_by_month,
            "unowned_geometry_months": unowned_active_months,
        }

    if not owned_active_months:
        return {
            "status": "month_geometry_unowned",
            "row_relation": "unresolved",
            "source_line_indices": [],
            "observed_texts": [],
            "tokens": [],
            "cell_status_by_month": {
                str(month): "month_geometry_unowned" for month in active_months
            },
            "unowned_geometry_months": unowned_active_months,
        }
    year_match = _YEAR_RE.match(str(year_line.get("text") or "").strip())
    year_remainder = (
        _YEAR_RE.sub("", str(year_line.get("text") or "").strip(), count=1) if year_match is not None else ""
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
            return retain_month_geometry_ownership({
                "status": "ambiguous_year_remainder",
                "row_relation": "year_line_remainder",
                "source_line_indices": [int(year_line["idx"])],
                "observed_texts": [year_remainder],
                "tokens": [],
                "cell_status_by_month": {str(month): "ambiguous_sequence_cardinality" for month in active_months},
            })
        active_cols = [cols_by_month[month] for month in active_months if month in cols_by_month]
        remainder_bbox = [
            min(float(col["bbox"][0]) for col in active_cols),
            year_y0,
            max(float(col["bbox"][2]) for col in active_cols),
            year_y1,
        ]
        amount_line = {
            **year_line,
            "text": year_remainder,
            "bbox": remainder_bbox,
            "amount_source_line_indices": [int(year_line["idx"])],
        }
        for key in (
            "source_logical_page",
            "coordinate_logical_page",
            "source_origin_logical_page",
            "coordinate_status",
            "source_bbox",
        ):
            amount_line.pop(key, None)
        amount_line.update(
            _synthesized_row_provenance(
                [year_line],
                bbox=remainder_bbox,
                page=page,
            )
        )
        return retain_month_geometry_ownership({
            "status": "exact",
            "row_relation": "year_line_remainder",
            "line": amount_line,
            "tokens": remainder_tokens,
            "source_line_indices": [int(year_line["idx"])],
            "observed_texts": [year_remainder],
            "cell_status_by_month": {str(month): "exact" for month in active_months},
        })

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
            cluster_centers = [(float(item["bbox"][1]) + float(item["bbox"][3])) / 2.0 for item in cluster]
            cluster_heights = [max(1.0, float(item["bbox"][3]) - float(item["bbox"][1])) for item in cluster]
            tolerance = max(3.0, 0.6 * max(height, sum(cluster_heights) / len(cluster_heights)))
            if abs(center - sum(cluster_centers) / len(cluster_centers)) <= tolerance:
                selected = cluster
                break
        if selected is None:
            selected = []
            clusters.append(selected)
        selected.append(line)

    if len(clusters) != 1:
        fragment_cover = _candidate_b_unique_amount_fragment_cover(
            [item for cluster in clusters for item in cluster],
            year_line,
            year_lines=year_lines,
            status_line=status_line,
            month_cols=month_cols,
            active_months=active_months,
            page=page,
        )
        if fragment_cover is not None:
            return retain_month_geometry_ownership(fragment_cover)
        status = (
            "ambiguous_immediate_rows"
            if clusters
            else ("non_immediate_amount_row" if non_immediate else "missing_amount_row")
        )
        candidates = [item for cluster in clusters for item in cluster] or non_immediate
        return retain_month_geometry_ownership({
            "status": status,
            "row_relation": "unresolved",
            "source_line_indices": [int(item["idx"]) for item in candidates],
            "observed_texts": [str(item.get("text") or "") for item in candidates],
            "tokens": [],
            "cell_status_by_month": {str(month): status for month in active_months},
        })

    row_lines = sorted(clusters[0], key=lambda item: float(item["bbox"][0]))
    row_y0 = min(float(line["bbox"][1]) for line in row_lines)
    row_y1 = max(float(line["bbox"][3]) for line in row_lines)
    row_bbox = [
        min(float(line["bbox"][0]) for line in row_lines),
        row_y0,
        max(float(line["bbox"][2]) for line in row_lines),
        row_y1,
    ]
    row_line = {
        "idx": min(int(line["idx"]) for line in row_lines),
        "text": " ".join(str(line.get("text") or "") for line in row_lines),
        "bbox": row_bbox,
        "confidence": min(float(line.get("confidence") or 0.0) for line in row_lines),
        **_synthesized_row_provenance(row_lines, bbox=row_bbox, page=page),
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
                    center_x - (float(cols_by_month[month]["bbox"][0]) + float(cols_by_month[month]["bbox"][2])) / 2.0
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
                    min(lx1, float(col["bbox"][2])) - max(lx0, float(col["bbox"][0])),
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
            source_bbox=line.get("bbox"),
            allow_geometry_proven_zero_run=True,
        )
        if line_tokens is None:
            ambiguous_months.update(spanned)
        else:
            tokens.extend(line_tokens)

    tokens_by_month: dict[int, list[OCRToken]] = {}
    for token in tokens:
        month = min(
            owned_active_months,
            key=lambda value: abs(
                token.center[0]
                - (float(cols_by_month[value]["bbox"][0]) + float(cols_by_month[value]["bbox"][2])) / 2.0
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
    return retain_month_geometry_ownership({
        "status": "exact",
        "row_relation": "aligned_or_immediate_after_year",
        "line": row_line,
        "tokens": tokens,
        "source_line_indices": [int(line["idx"]) for line in row_lines],
        "observed_texts": [str(line.get("text") or "") for line in row_lines],
        "cell_status_by_month": cell_status_by_month,
    })


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
            cluster_centers = [(float(item[0]["bbox"][1]) + float(item[0]["bbox"][3])) / 2.0 for item in cluster]
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
            month: sorted(values)[len(values) // 2] for month, values in by_month.items() if month in range(1, 13)
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
            "month_header_source_line_indices": [int(line["idx"]) for line in cluster_lines],
        }
        logical_pages = {int(line["source_logical_page"]) for line in cluster_lines if line.get("source_logical_page")}
        if len(logical_pages) == 1:
            header.update(
                _synthesized_row_provenance(
                    cluster_lines,
                    bbox=header["bbox"],
                    page=next(iter(logical_pages)),
                )
            )
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
    candidate_b_status_glyph_observations: list[dict[str, Any]] | None = None,
    continuation_logical_pages: Iterable[int] = (),
    source_table_geometry_by_page: Mapping[
        str | int,
        Iterable[Mapping[str, Any]],
    ]
    | None = None,
    grid_index: int = 0,
) -> RepaymentExtraction:
    """Reconstruct a credit repayment grid from OCR line-level geometry."""
    line_items = _line_items(lines)
    status_charset = _STATUS_CHARS | {str(item) for item in extra_status_chars if str(item)}
    zero_overdue_statuses = _ZERO_OVERDUE_STATUSES | ({"A"} if "A" in status_charset else set())
    static_sensitive_statuses = {"N", "*"} | ({"C"} if enable_candidate_b_amount_pairing else set())
    affirmative_continuation_pages = {int(value) for value in continuation_logical_pages if int(value) > 0}
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
    header_line = exact_header_line or _line_after(line_items, ay1, x_min=ax0 - 220, x_max=ax1 + 260, max_gap=35.0)
    header_alignment_exact = bool(
        exact_header_line is not None
        and exact_header_line.get("month_header_geometry") in {"merged_line_exact", "word_center_sequence_exact"}
    )
    header_spatial_lattice_exact = bool(
        exact_header_line is not None and exact_header_line.get("month_header_geometry") == "word_center_sequence_exact"
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
    base_year_lines = [
        year_line
        for year_line in years
        if int(year_line.get("source_logical_page") or page) == page
    ]
    base_year_column_bbox = _representative_year_column_bbox(
        base_year_lines,
        logical_page=page,
    )
    visual_month_cols, visual_geometry_audit = _visual_month_col_bands_in_registered_plane(
        month_cols,
        source_lines=[header_line, *base_year_lines],
        base_page=page,
        base_page_width=page_width,
        base_page_height=page_height,
        page_image=page_image,
        page_image_resolver=page_image_resolver,
        y0=float(header_line["bbox"][1]) - 5.0,
        y1=max(
            (float(year_line["bbox"][3]) for year_line in base_year_lines),
            default=float(header_line["bbox"][3]),
        ) + 35.0,
        year_column_bbox=base_year_column_bbox,
        require_physical_month_ownership=enable_candidate_b_amount_pairing,
        max_right_shift_months=(1.10 if enable_candidate_b_amount_pairing else 0.55),
        prefer_validated_header_lattice=bool(enable_candidate_b_amount_pairing and header_alignment_exact),
        retain_validated_header_on_residual=bool(enable_candidate_b_amount_pairing and header_spatial_lattice_exact),
        max_residual_shift_months=(0.5 if enable_candidate_b_amount_pairing and header_alignment_exact else None),
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
    header_source_indices = {int(value) for value in header_line.get("month_header_source_line_indices") or ()}
    header_source_indices.add(int(header_line["idx"]))
    source_tables_cache: dict[int, list[Mapping[str, Any]]] = {}

    def source_tables_for(logical_page: int) -> list[Mapping[str, Any]]:
        cached = source_tables_cache.get(logical_page)
        if cached is not None:
            return cached
        raw_source_tables = (
            source_table_geometry_by_page.get(logical_page) if source_table_geometry_by_page is not None else None
        )
        if raw_source_tables is None and source_table_geometry_by_page is not None:
            raw_source_tables = source_table_geometry_by_page.get(str(logical_page))
        normalized = (
            [table for table in raw_source_tables if isinstance(table, Mapping)]
            if isinstance(raw_source_tables, Iterable) and not isinstance(raw_source_tables, (str, bytes, Mapping))
            else []
        )
        source_tables_cache[logical_page] = normalized
        return normalized

    status_lines_by_year_index: dict[int, dict[str, Any]] = {}
    for candidate_year_line in years:
        candidate_year_match = _YEAR_RE.match(str(candidate_year_line.get("text") or "").strip())
        candidate_year = int(candidate_year_match.group(0)) if candidate_year_match is not None else None
        candidate_active_months = (
            [month for observed_year, month in sorted(record_months) if observed_year == candidate_year]
            if candidate_year is not None
            else []
        )
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
            enable_candidate_b_amount_pairing
            and candidate_year is not None
            and candidate_active_months
            and source_table_geometry_by_page is not None
        ):
            candidate_logical_page = int(candidate_year_line.get("source_logical_page") or page)
            if candidate_logical_page == page or candidate_logical_page in affirmative_continuation_pages:
                candidate_source_tables = source_tables_for(candidate_logical_page)
                candidate_year_bbox = _local_page_bbox(
                    tuple(float(value) for value in candidate_year_line["bbox"]),
                    logical_page=candidate_logical_page,
                    base_page=page,
                    base_page_height=page_height,
                    coordinates_already_registered=(
                        candidate_year_line.get("coordinate_logical_page") is not None
                    ),
                    coordinate_status=str(candidate_year_line.get("coordinate_status") or ""),
                )
                candidate_status_bbox: BBox | None = None
                if (
                    candidate_status_line is not None
                    and int(candidate_status_line.get("source_logical_page") or page)
                    == candidate_logical_page
                ):
                    candidate_status_bbox = _local_page_bbox(
                        tuple(float(value) for value in candidate_status_line["bbox"]),
                        logical_page=candidate_logical_page,
                        base_page=page,
                        base_page_height=page_height,
                        coordinates_already_registered=(
                            candidate_status_line.get("coordinate_logical_page") is not None
                        ),
                        coordinate_status=str(candidate_status_line.get("coordinate_status") or ""),
                    )
                # Resolve status-band ambiguity before comparing the raw and
                # canonical values, never for the first time at materialization.
                candidate_source_lattice = (
                    _candidate_b_owned_source_lattice(
                        candidate_source_tables,
                        logical_page=candidate_logical_page,
                        expected_year=candidate_year,
                        active_months=candidate_active_months,
                        year_bbox=candidate_year_bbox,
                        status_bbox=candidate_status_bbox,
                    )
                    if candidate_source_tables and candidate_year_bbox is not None
                    else None
                )
                fallback_status_tokens = (
                    list(candidate_status_line.get("candidate_b_status_tokens"))
                    if isinstance(candidate_status_line, Mapping)
                    and isinstance(
                        candidate_status_line.get("candidate_b_status_tokens"),
                        list,
                    )
                    else None
                )
                exact_source_status_line = None
                if candidate_source_lattice is not None:
                    exact_source_status_line = _candidate_b_exact_active_source_status_row(
                        evidence_tokens,
                        candidate_year_line,
                        source_lattice=candidate_source_lattice,
                        active_months=candidate_active_months,
                        status_charset=status_charset,
                        page=page,
                        base_page_height=page_height,
                        fallback_status_tokens=fallback_status_tokens,
                    )
                    if exact_source_status_line is not None:
                        exact_source_status_line["candidate_b_sparse_source_lattice"] = candidate_source_lattice
                        exact_source_status_line["candidate_b_sparse_geometry_months"] = list(
                            candidate_active_months
                        )
                        exact_source_status_line["candidate_b_complete_source_lattice"] = True
                elif candidate_source_tables and candidate_year_bbox is not None:
                    exact_source_status_line = _candidate_b_sparse_exact_source_status_row(
                        evidence_tokens,
                        candidate_year_line,
                        source_tables=candidate_source_tables,
                        logical_page=candidate_logical_page,
                        expected_year=candidate_year,
                        active_months=candidate_active_months,
                        year_bbox=candidate_year_bbox,
                        status_charset=status_charset,
                        page=page,
                        base_page_height=page_height,
                        fallback_status_tokens=fallback_status_tokens,
                        status_bbox=candidate_status_bbox,
                    )
                if exact_source_status_line is not None:
                    # A line-level candidate can be only a surviving fragment
                    # of the printed row. Independently source-owned cells are
                    # stronger evidence and remain field-local when a sibling
                    # cell's geometry is damaged.
                    candidate_status_line = exact_source_status_line
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
    static_amount_zero_attempts = 0
    static_amount_zero_resolved = 0
    static_amount_zero_corrections = 0
    static_amount_zero_unresolved = 0
    static_amount_zero_unavailable = 0
    status_templates: dict[str, list[Any]] = {}
    static_status_seeds: dict[str, list[dict[str, Any]]] = {}
    static_status_contradictions: set[str] = set()
    pending_static_statuses: list[dict[str, Any]] = []
    document_status_glyph_observations: list[dict[str, Any]] = []
    static_status_geometry: list[tuple[tuple[float, ...], int]] = []
    continuation_visual_cols_cache: dict[
        Any, tuple[list[dict[str, Any]], dict[str, Any]]
    ] = {}
    visual_geometry_audit_by_page: dict[str, dict[str, Any]] = {str(page): dict(visual_geometry_audit)}

    for year_line in years:
        year_match = _YEAR_RE.match(year_line["text"].strip())
        if year_match is None:
            continue
        year = int(year_match.group(0))
        status_line = status_lines_by_year_index.get(int(year_line["idx"]))
        if (
            status_line is None or status_line is header_line or "还款记录" in status_line["text"]
        ) and not enable_candidate_b_amount_pairing:
            status_line = _line_after(
                line_items, year_line["bbox"][1], x_min=grid_x0 - 10, x_max=grid_x1 + 10, max_gap=55.0
            )
        if status_line is None:
            continue
        active_months = [month for yy, month in sorted(record_months) if yy == year]
        status_logical_page = int(status_line.get("source_logical_page") or page)
        year_visual_cols = visual_month_cols
        row_geometry_audit = dict(visual_geometry_audit)
        if status_logical_page != page:
            year_visual_cols, row_geometry_audit = _visual_month_col_bands_in_registered_plane(
                month_cols,
                source_lines=[status_line, year_line],
                base_page=page,
                base_page_width=page_width,
                base_page_height=page_height,
                page_image=page_image,
                page_image_resolver=page_image_resolver,
                y0=min(float(year_line["bbox"][1]), float(status_line["bbox"][1])) - 5.0,
                y1=max(float(year_line["bbox"][3]), float(status_line["bbox"][3])) + 40.0,
                year_column_bbox=year_line["bbox"],
                cache=continuation_visual_cols_cache,
                require_physical_month_ownership=enable_candidate_b_amount_pairing,
                max_left_shift_months=1.85,
                max_right_shift_months=(1.85 if enable_candidate_b_amount_pairing else 0.55),
                prefer_validated_header_lattice=bool(
                    enable_candidate_b_amount_pairing and header_alignment_exact
                ),
                # A base-page header is not a continuation-page physical witness.
                retain_validated_header_on_residual=False,
                allow_unowned_header_fallback=(not enable_candidate_b_amount_pairing),
                max_residual_shift_months=(
                    0.5 if enable_candidate_b_amount_pairing and header_alignment_exact else None
                ),
            )
        source_lattice: SourceTableMonthLattice | None = None
        complete_source_lattice = False
        source_geometry_months = set(active_months)
        source_amount_geometry_months = set(active_months)
        source_amount_month_cols: list[dict[str, Any]] = []
        source_status_row_y: tuple[float, float] | None = None
        source_amount_row_y: tuple[float, float] | None = None
        if (
            enable_candidate_b_amount_pairing
            and (status_logical_page == page or status_logical_page in affirmative_continuation_pages)
            and source_table_geometry_by_page is not None
        ):
            source_tables = source_tables_for(status_logical_page)
            local_year_bbox = _local_page_bbox(
                tuple(float(value) for value in year_line["bbox"]),
                logical_page=status_logical_page,
                base_page=page,
                base_page_height=page_height,
                coordinates_already_registered=(
                    year_line.get("coordinate_logical_page") is not None
                ),
                coordinate_status=str(year_line.get("coordinate_status") or ""),
            )
            local_status_bbox = _local_page_bbox(
                tuple(float(value) for value in status_line["bbox"]),
                logical_page=status_logical_page,
                base_page=page,
                base_page_height=page_height,
                coordinates_already_registered=(
                    status_line.get("coordinate_logical_page") is not None
                ),
                coordinate_status=str(status_line.get("coordinate_status") or ""),
            )
            source_lattice = (
                resolve_unique_source_table_year_plus_twelve_ownership(
                    source_tables,
                    logical_page=status_logical_page,
                    expected_year=year,
                    active_months=active_months,
                    year_bbox=local_year_bbox,
                    status_bbox=local_status_bbox,
                )
                if source_tables
                and local_year_bbox is not None
                and local_status_bbox is not None
                else None
            )
            complete_source_lattice = source_lattice is not None
            sparse_lattice = status_line.get("candidate_b_sparse_source_lattice")
            sparse_months = status_line.get("candidate_b_sparse_geometry_months")
            normalized_sparse_months = {
                int(value)
                for value in sparse_months or ()
                if isinstance(value, int) and not isinstance(value, bool) and value in active_months
            }
            if (
                isinstance(sparse_lattice, SourceTableMonthLattice)
                and sparse_lattice.logical_page == status_logical_page
                and sparse_lattice.expected_year == year
                and normalized_sparse_months
            ):
                if source_lattice is None:
                    source_lattice = sparse_lattice
                if status_line.get("candidate_b_complete_source_lattice") is True:
                    complete_source_lattice = True
                source_geometry_months = normalized_sparse_months
                source_amount_geometry_months = set(normalized_sparse_months)
        if source_lattice is not None:
            full_source_month_cols = [
                col
                for col in _source_lattice_month_cols(source_lattice)
                if int(col["header"]) in active_months
            ]
            source_month_cols = [
                col for col in full_source_month_cols if int(col["header"]) in source_geometry_months
            ]
            source_amount_month_cols = [
                col
                for col in full_source_month_cols
                if int(col["header"]) in source_amount_geometry_months
            ]
            source_comparison_cols = (
                full_source_month_cols if complete_source_lattice else source_month_cols
            )
            visual_physical_lattice = bool(
                year_visual_cols
                and row_geometry_audit.get("usable") is not False
                and row_geometry_audit.get("source") == "vertical_rule_projection"
            )
            visual_requires_source = bool(
                visual_physical_lattice and _candidate_b_visual_lattice_needs_source_table(row_geometry_audit)
            )
            visual_source_agree = bool(
                visual_physical_lattice
                and _month_geometry_planes_agree(
                    year_visual_cols,
                    source_comparison_cols,
                )
            )
            repaired_status_tokens = status_line.get("candidate_b_status_tokens")
            repaired_status_months = status_line.get("candidate_b_status_token_months")
            declared_exact_status_months = status_line.get(
                "candidate_b_exact_source_status_months"
            )
            exact_status_atom_months = (
                {
                    int(month)
                    for month in declared_exact_status_months
                    if isinstance(month, int)
                    and not isinstance(month, bool)
                    and month in active_months
                }
                if isinstance(declared_exact_status_months, (list, tuple, set))
                else {
                    int(month)
                    for month, token in zip(
                        repaired_status_months or (),
                        repaired_status_tokens or (),
                        strict=False,
                    )
                    if isinstance(month, int)
                    and not isinstance(month, bool)
                    and isinstance(token, OCRToken)
                    and token.source in _EXACT_SOURCE_STATUS_CELL_SOURCES
                }
            )
            exact_amount_atom_months = set(
                _candidate_b_exact_source_zero_amount_tokens(
                    evidence_tokens,
                    source_lattice=source_lattice,
                    active_months=active_months,
                    page=page,
                    base_page_height=page_height,
                )
            )
            strong_visual_source_conflict = bool(
                visual_physical_lattice and not visual_source_agree and not visual_requires_source
            )
            exact_atom_source_override = False
            exact_amount_atom_source_override = False
            if strong_visual_source_conflict:
                source_geometry_months &= exact_status_atom_months
                source_amount_geometry_months &= exact_amount_atom_months
                source_month_cols = [
                    col
                    for col in source_comparison_cols
                    if int(col["header"]) in source_geometry_months
                ]
                source_amount_month_cols = [
                    col
                    for col in source_comparison_cols
                    if int(col["header"]) in source_amount_geometry_months
                ]
                exact_atom_source_override = bool(source_month_cols)
                exact_amount_atom_source_override = bool(source_amount_month_cols)
            elif complete_source_lattice:
                # A complete source table is compared as a complete plane.  A
                # partial set of field-repair atoms must not make corroborated
                # or ambiguity-resolving sibling columns disappear.
                source_geometry_months = set(active_months)
                source_amount_geometry_months = set(active_months)
                source_month_cols = full_source_month_cols
                source_amount_month_cols = full_source_month_cols
            if (
                strong_visual_source_conflict
                and not exact_atom_source_override
                and not exact_amount_atom_source_override
            ):
                row_geometry_audit = {
                    "source": "rejected_month_geometry",
                    "usable": False,
                    "reason": (
                        "continuation_month_geometry_plane_conflict"
                        if status_logical_page != page
                        else "source_table_month_geometry_plane_conflict"
                    ),
                    "visual_edges": _month_geometry_edges(year_visual_cols),
                    "source_table_edges": _month_geometry_edges(source_comparison_cols),
                    "source_table_id": source_lattice.table_id,
                    "source_table_comparison": "disagree",
                    "value_inputs_used": False,
                    "logical_page": status_logical_page,
                    "source_logical_page": source_lattice.source_logical_page,
                    "source_page": source_lattice.source_page,
                }
                year_visual_cols = []
                source_lattice = None
            else:
                source_status_row_y = _joined_continuation_row_y(
                    source_lattice.month_bboxes,
                    logical_page=status_logical_page,
                    base_page=page,
                    base_page_height=page_height,
                )
                source_amount_row_y = _joined_continuation_row_y(
                    source_lattice.amount_bboxes,
                    logical_page=status_logical_page,
                    base_page=page,
                    base_page_height=page_height,
                )
                source_provenance = source_lattice.provenance_dict()
                # The detached source table is the exact physical coordinate
                # plane.  Even when the visual plane agrees, snap every month m
                # to source column m so refs and crops cannot retain sub-cell
                # drift.  No source-table text or cell value participates.
                year_visual_cols = source_month_cols
                row_geometry_audit = {
                    **source_provenance,
                    "source": "source_table_geometry",
                    "usable": True,
                    "selection_basis": ("source_table_year_plus_twelve_ownership"),
                    "reason": "exact_source_table_month_lattice_calibration",
                    "table_id": source_lattice.table_id,
                    "continuation_logical_page": (status_logical_page if status_logical_page != page else None),
                    "logical_page": status_logical_page,
                    "vertical_rule_count": int(source_provenance.get("rule_count") or 14),
                    "column_count": 13,
                    "status_row_index": source_lattice.status_row_index,
                    "amount_row_index": source_lattice.amount_row_index,
                    "coordinate_system": source_lattice.coordinate_system,
                    "source_table_comparison": (
                        "agree"
                        if visual_source_agree
                        else (
                            "source_over_conflicting_visual_exact_cell_atoms"
                            if exact_atom_source_override or exact_amount_atom_source_override
                            else ("source_over_ambiguous_visual" if visual_requires_source else "source_only")
                        )
                    ),
                    "calibrated_from_source_table_geometry": True,
                    "corroborated_by_source_table_geometry": bool(visual_source_agree),
                    "ambiguous_visual_geometry_superseded": bool(visual_requires_source and not visual_source_agree),
                    "conflicting_visual_geometry_superseded": bool(
                        (exact_atom_source_override or exact_amount_atom_source_override)
                        and not visual_source_agree
                    ),
                    "exact_source_atom_geometry_months": sorted(
                        source_geometry_months if exact_atom_source_override else ()
                    ),
                    "exact_source_status_atom_geometry_months": sorted(
                        source_geometry_months if exact_atom_source_override else ()
                    ),
                    "exact_source_amount_atom_geometry_months": sorted(
                        source_amount_geometry_months
                        if exact_amount_atom_source_override
                        else ()
                    ),
                    "visual_selection_basis": row_geometry_audit.get("selection_basis"),
                    "visual_owned_month_rule_hits": row_geometry_audit.get("owned_month_rule_hits"),
                    "visual_residual_shift_months": row_geometry_audit.get("residual_shift_months"),
                }
        elif enable_candidate_b_amount_pairing and _candidate_b_visual_lattice_needs_source_table(row_geometry_audit):
            rejected_visual_audit = dict(row_geometry_audit)
            year_visual_cols = []
            row_geometry_audit = {
                "source": "rejected_month_geometry",
                "usable": False,
                "reason": "source_table_month_ownership_required",
                "logical_page": status_logical_page,
                "value_inputs_used": False,
                "visual_selection_basis": rejected_visual_audit.get("selection_basis"),
                "visual_owned_month_rule_hits": rejected_visual_audit.get("owned_month_rule_hits"),
                "visual_residual_shift_months": rejected_visual_audit.get("residual_shift_months"),
            }
        if enable_candidate_b_amount_pairing:
            visual_geometry_audit_by_page[str(status_logical_page)] = dict(row_geometry_audit)
        row_month_geometry_exact = bool(header_alignment_exact or source_lattice is not None)
        year_assignment_cols = year_visual_cols if enable_candidate_b_amount_pairing else month_cols
        amount_assignment_cols = (
            source_amount_month_cols
            if enable_candidate_b_amount_pairing and source_lattice is not None
            else year_assignment_cols
        )
        year_visual_cols_by_month = {int(col["header"]): col for col in year_visual_cols}
        amount_visual_cols_by_month = {
            int(col["header"]): col for col in amount_assignment_cols
        }
        row_geometry_provenance = (
            _accepted_month_geometry_provenance(row_geometry_audit) if enable_candidate_b_amount_pairing else {}
        )
        row_geometry_rejection = (
            _rejected_month_geometry_provenance(row_geometry_audit) if enable_candidate_b_amount_pairing else {}
        )
        amount_pairing: dict[str, Any] | None = None
        amount_row_tokens: list[OCRToken] | None = None
        if enable_candidate_b_amount_pairing:
            if amount_assignment_cols:
                amount_pairing = _candidate_b_amount_row_pair(
                    line_items,
                    year_line,
                    month_cols=amount_assignment_cols,
                    active_months=active_months,
                    page=page,
                    excluded_line_indices=excluded_amount_line_indices,
                    year_lines=years,
                    status_line=status_line,
                )
            else:
                amount_pairing = {
                    "status": "month_geometry_unowned",
                    "row_relation": "unresolved",
                    "source_line_indices": [],
                    "observed_texts": [],
                    "tokens": [],
                    "cell_status_by_month": {str(month): "month_geometry_unowned" for month in active_months},
                    "unowned_geometry_months": list(active_months),
                }
            if source_lattice is not None:
                amount_pairing = _candidate_b_merge_exact_source_zero_amounts(
                    amount_pairing,
                    evidence_tokens,
                    year_line,
                    source_lattice=source_lattice,
                    active_months=active_months,
                    source_geometry_months=source_amount_geometry_months,
                    page=page,
                    base_page_height=page_height,
                )
            amount_line = amount_pairing.get("line")
            raw_amount_tokens = amount_pairing.get("tokens")
            amount_row_tokens = list(raw_amount_tokens) if isinstance(raw_amount_tokens, list) else []
        else:
            # Shared callers retain their existing materialization. Candidate B
            # never falls back to treating the printed year as an amount row.
            year_remainder = _YEAR_RE.sub("", year_line["text"].strip(), count=1)
            amount_line = year_line
            if not re.search(r"\d", year_remainder):
                amount_line = (
                    _explicit_amount_line_after_year(
                        line_items,
                        year_line,
                        x_min=grid_x0 - 10,
                        x_max=grid_x1 + 10,
                    )
                    or year_line
                )

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
        if source_status_row_y is not None:
            status_band["bbox"][1] = source_status_row_y[0]
            status_band["bbox"][3] = source_status_row_y[1]
        if source_amount_row_y is not None and amount_band is not None:
            amount_band["bbox"][1] = source_amount_row_y[0]
            amount_band["bbox"][3] = source_amount_row_y[1]
        if enable_candidate_b_amount_pairing and amount_band is not None:
            status_center = (float(status_band["bbox"][1]) + float(status_band["bbox"][3])) / 2.0
            amount_center = (float(amount_band["bbox"][1]) + float(amount_band["bbox"][3])) / 2.0
            if amount_center <= status_center + 0.5:
                amount_pairing = {
                    **(amount_pairing or {}),
                    "status": "ambiguous_vertical_row_geometry",
                    "row_relation": "unresolved",
                    "tokens": [],
                    "cell_status_by_month": {str(month): "ambiguous_vertical_row_geometry" for month in active_months},
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
                key: value for key, value in amount_pairing.items() if key not in {"line", "tokens"}
            }
        status_band["index"] = len(row_bands)
        row_bands.append(status_band)
        if amount_band is not None:
            amount_band["index"] = len(row_bands)
            row_bands.append(amount_band)

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
        source_repair_unresolved_months = {
            int(value)
            for value in status_line.get(
                "candidate_b_status_unresolved_months",
                (),
            )
            if isinstance(value, int) and not isinstance(value, bool)
        }
        source_repair_conflict_months = {
            int(value)
            for value in status_line.get(
                "candidate_b_status_conflict_months",
                (),
            )
            if isinstance(value, int) and not isinstance(value, bool)
        }
        source_repair_conflict_observations = status_line.get(
            "candidate_b_status_conflict_observations",
            {},
        )
        source_repair_conflict_observations = (
            source_repair_conflict_observations
            if isinstance(source_repair_conflict_observations, Mapping)
            else {}
        )
        status_row_tokens = (
            list(candidate_status_tokens)
            if enable_candidate_b_amount_pairing and isinstance(candidate_status_tokens, list)
            else _expand_line_to_char_tokens(
                {**status_line, "text": normalized_status_text},
                page=page,
                prefix=f"ocr_p{page}_repay_status_{year}",
            )
        )
        status_assignments = _assign_row(
            status_row_tokens,
            status_band,
            year_assignment_cols,
        )
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
            exact_full_row = observed_months == set(range(1, 13)) and observed_status_token_count == 12
            exact_active_row = observed_months == set(active_months) and observed_status_token_count == len(
                active_months
            )
            row_alignment_exact = bool(
                (exact_full_row or exact_active_row)
                and ("#" not in status_by_month.values() or hash_is_business_status)
            )
        elif year == end_year and len(status_chars) == 2 and set(status_chars) == {"N", "C"}:
            status_by_month = {active_months[0]: "N", active_months[1]: "C"} if len(active_months) == 2 else {}
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
        amount_assignments = (
            _assign_row(amount_row_tokens, amount_band, amount_assignment_cols)
            if amount_band is not None
            else {}
        )

        for col in month_cols:
            month = int(col["header"])
            st_tokens = status_assignments.get(month, [])
            status_center_y = (float(status_line["bbox"][1]) + float(status_line["bbox"][3])) / 2.0
            if amount_line is not None:
                amount_center_y = (float(amount_line["bbox"][1]) + float(amount_line["bbox"][3])) / 2.0
                st_tokens = [
                    token
                    for token in st_tokens
                    if abs(token.center[1] - status_center_y) <= abs(token.center[1] - amount_center_y)
                ]
            visual_col = year_visual_cols_by_month.get(month)
            amount_visual_col = amount_visual_cols_by_month.get(month)
            owned_token_witness = (
                _candidate_b_owned_status_token(
                    st_tokens,
                    row_tokens=status_row_tokens,
                    visual_col=visual_col,
                    geometry_audit=row_geometry_audit,
                    status_charset=status_charset,
                )
                if enable_candidate_b_amount_pairing
                else None
            )
            status = (
                ""
                if month in source_repair_conflict_months
                else normalize_allowlist_text(
                    status_by_month.get(month) or _token_text(st_tokens),
                    status_charset,
                    max_chars=1,
                )
            )
            exact_row_status = (
                normalize_allowlist_text(
                    status_by_month.get(month) or "",
                    status_charset,
                    max_chars=1,
                )
                if row_alignment_exact and month in status_by_month
                else ""
            )
            status_crop = None
            status_recognition_source = "tokens"
            status_recognition_audit: dict[str, Any] = {}
            static_status_failed = False
            status_row_cell_conflict = False
            if month in source_repair_conflict_months:
                status_recognition_source = "corrected_source_planes_conflict"
                status_recognition_audit = {
                    "reason": "corrected_status_planes_disagree",
                    "logical_page": int(status_line.get("source_logical_page") or page),
                    "observations": deepcopy(
                        source_repair_conflict_observations.get(str(month), {})
                    ),
                }
                status_row_cell_conflict = True
            elif row_alignment_exact and month in status_by_month:
                status_recognition_source = "canonical_row_sequence"
                status_recognition_audit = {
                    "alignment_status": "exact",
                    "expected_cell_count": len(active_months),
                    "observed_status_count": len(status_chars),
                    "active_months": list(active_months),
                    "logical_page": int(status_line.get("source_logical_page") or page),
                    **(
                        {"row_repair": status_line["candidate_b_status_row_repair"]}
                        if status_line.get("candidate_b_status_row_repair")
                        else {}
                    ),
                }
            elif (
                month in status_by_month
                and month not in source_repair_unresolved_months
                and status_line.get("candidate_b_status_row_repair")
            ):
                status_recognition_source = "candidate_b_source_cell_repair"
                status_recognition_audit = {
                    "alignment_status": "exact_source_cell",
                    "active_months": list(active_months),
                    "logical_page": int(status_line.get("source_logical_page") or page),
                    "row_repair": status_line["candidate_b_status_row_repair"],
                    "unresolved_months": sorted(source_repair_unresolved_months),
                }
            neighbor_status = (
                _neighbor_status_fallback(
                    status_by_month,
                    month,
                    zero_overdue_statuses=zero_overdue_statuses,
                )
                if not status
                and month not in source_repair_unresolved_months
                and month not in source_repair_conflict_months
                else ""
            )
            if neighbor_status:
                status = neighbor_status
                status_recognition_source = "row_neighbor_consensus"
                status_recognition_audit = {
                    "reason": "unreadable_cell_with_matching_adjacent_statuses",
                    "logical_page": int(status_line.get("source_logical_page") or page),
                }
            cell_col = visual_col or col
            visual_status_bbox = _cell_bbox(status_band, visual_col) if visual_col is not None else None
            visual = None
            if (
                (enable_cell_ocr or (enable_static_status_validation and status in static_sensitive_statuses))
                and (year, month) in record_months
                and visual_status_bbox is not None
                and month not in source_repair_conflict_months
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
                        cell_candidate_accepted = bool(
                            (weak_cell and (_strong_visual_status(rec) or template_confirmed)) or consensus_count >= 2
                        )
                        if cell_candidate_accepted:
                            if enable_candidate_b_amount_pairing and exact_row_status and rec.text != exact_row_status:
                                # Exact row parsing and independent cell OCR are
                                # two evidence planes.  A disagreement is not a
                                # licence for either plane to silently win.
                                status = ""
                                status_row_cell_conflict = True
                                status_recognition_source = "candidate_b_exact_row_cell_status_conflict"
                                status_recognition_audit = {
                                    **rec.audit,
                                    "reason": "exact_row_cell_status_disagreement",
                                    "row_status": exact_row_status,
                                    "cell_status": rec.text,
                                    "consensus_count": consensus_count,
                                    "resolution": "withheld_unknown_review",
                                    "logical_page": visual_page,
                                }
                            elif (
                                enable_candidate_b_amount_pairing
                                and owned_token_witness is not None
                                and rec.text != owned_token_witness[0]
                            ):
                                token_status, token_witness = owned_token_witness
                                # A uniquely positioned source token and an
                                # accepted crop OCR result are independent
                                # observations.  A disagreement localizes this
                                # month; neither value is allowed to win.
                                status = ""
                                status_row_cell_conflict = True
                                status_recognition_source = "candidate_b_owned_token_cell_status_conflict"
                                status_recognition_audit = {
                                    **rec.audit,
                                    "reason": ("owned_month_token_cell_status_disagreement"),
                                    "token_status": token_status,
                                    "cell_status": rec.text,
                                    "token_source": token_witness.source,
                                    "token_source_id": (token_witness.source_token_id or token_witness.token_id),
                                    "consensus_count": consensus_count,
                                    "month_geometry_selection_basis": (row_geometry_audit.get("selection_basis")),
                                    "resolution": "withheld_unknown_review",
                                    "logical_page": visual_page,
                                }
                            else:
                                status = rec.text
            static_template = None
            observed_row_status = ""
            visual_status = ""
            visual_confidence = 0.0
            visual_audit: dict[str, Any] = {}
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
                            _static_candidate_b_zero_status_glyph_classification(static_template)
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
            if (
                not status
                and not static_status_failed
                and not status_row_cell_conflict
                and month not in source_repair_unresolved_months
                and month not in source_repair_conflict_months
            ):
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
                **row_geometry_provenance,
                **row_geometry_rejection,
            }
            if visual_col is not None:
                localized_status_bbox = _local_page_bbox(
                    _cell_bbox(status_band, cell_col),
                    logical_page=status_logical_page,
                    base_page=page,
                    base_page_height=page_height,
                    coordinates_already_registered=(
                        status_line.get("coordinate_logical_page") is not None
                    ),
                    coordinate_status=str(status_line.get("coordinate_status") or ""),
                )
                if localized_status_bbox is not None:
                    status_ref_payload["bbox"] = localized_status_bbox
                else:
                    status_ref_payload["geometry_scope"] = "logical_page"
                    status_ref_payload["geometry_status"] = "unresolved"
            if enable_candidate_b_amount_pairing:
                status_recognition_audit["field_geometry_exact"] = bool(
                    row_month_geometry_exact
                    and visual_col is not None
                    and status_ref_payload.get("geometry_scope") == "cell"
                    and isinstance(status_ref_payload.get("bbox"), list)
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
                static_status_geometry.append((tuple(round(float(value), 4) for value in st_cell.bbox), year))

            amount = ""
            status_amount_conflict = False
            amount_bbox = None
            amount_ref_payload: dict[str, Any] | None = None
            amount_pair_status = "missing_amount_row"
            if amount_band is not None:
                amt_tokens = amount_assignments.get(month, [])
                observed_amount_text = _token_text(amt_tokens)
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
                amount_cell_col = amount_visual_col or col
                amount_bbox = (
                    _cell_bbox(amount_band, amount_visual_col)
                    if amount_visual_col is not None
                    else None
                )
                visual_amount_bbox = amount_bbox
                amount_crop = None
                static_amount_zero: tuple[float, dict[str, Any]] | None = None
                if (
                    enable_candidate_b_amount_pairing
                    and enable_static_status_validation
                    and status_implies_zero
                    and amount != "0"
                    and amount_pair_status in {"exact", "blank_amount_cell"}
                    # A uniquely owned source-table lattice proves this target
                    # cell's geometry even when noise elsewhere makes the
                    # whole status row non-exact.
                    and (row_alignment_exact or source_lattice is not None)
                    and row_month_geometry_exact
                    and (year, month) in record_months
                    and visual_amount_bbox is not None
                ):
                    static_amount_zero_attempts += 1
                    amount_visual = _visual_page_context(
                        source_line=(amount_line if isinstance(amount_line, dict) else status_line),
                        bbox=visual_amount_bbox,
                        base_page=page,
                        base_page_width=page_width,
                        base_page_height=page_height,
                        page_image=page_image,
                        page_image_resolver=page_image_resolver,
                    )
                    if amount_visual is None:
                        static_amount_zero_unresolved += 1
                        static_amount_zero_unavailable += 1
                    else:
                        (
                            amount_visual_image,
                            amount_visual_bbox,
                            amount_visual_width,
                            amount_visual_height,
                            _amount_visual_page,
                        ) = amount_visual
                        static_amount_zero = _static_amount_zero_glyph_classification(
                            amount_visual_image,
                            amount_visual_bbox,
                            page_width=amount_visual_width,
                            page_height=amount_visual_height,
                        )
                        if static_amount_zero is None:
                            static_amount_zero_unresolved += 1
                        else:
                            static_amount_zero_resolved += 1
                            static_amount_zero_corrections += 1
                            amount = "0"
                            amount_pair_status = "exact"
                status_amount_conflict = status_implies_zero and amount not in {"", "0"}
                if static_amount_zero is not None:
                    amount_recognition_source = "static_amount_zero_glyph_validation"
                    amount_recognition_audit = {
                        "reason": "printed_zero_glyph_independently_validated",
                        "amount_pair_status": "exact",
                        "observed_amount_text": observed_amount_text,
                        "published_amount": "0",
                        "confidence": static_amount_zero[0],
                        **static_amount_zero[1],
                    }
                elif amount_pair_status not in {"exact", "not_applicable"}:
                    amount_recognition_source = "candidate_b_amount_pair_unresolved"
                    amount_recognition_audit = {
                        "reason": amount_pair_status,
                        "amount_pair_status": amount_pair_status,
                        "row_pair_status": str(amount_pairing.get("status") or "unresolved"),
                        "source_line_indices": list(amount_pairing.get("source_line_indices") or ()),
                        "observed_texts": list(amount_pairing.get("observed_texts") or ()),
                    }
                elif status_amount_conflict:
                    amount_recognition_source = "status_amount_conflict"
                    amount_recognition_audit = {
                        "reason": "zero_status_conflicts_with_observed_nonzero_amount",
                        "status": status,
                        "observed_amount": amount,
                    }
                elif (
                    status_implies_zero
                    and amount == "0"
                    and len(amt_tokens) == 1
                    and amt_tokens[0].source in _EXACT_SOURCE_AMOUNT_CELL_SOURCES
                ):
                    amount_recognition_source = "exact_source_zero_cell"
                    amount_recognition_audit = {
                        "reason": "exact_zero_atom_in_exact_source_amount_cell",
                        "source_token_id": (amt_tokens[0].source_token_id or amt_tokens[0].token_id),
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
                amount_ref_payload = {
                    "page": amount_logical_page,
                    "logical_page": amount_logical_page,
                    "geometry_scope": "cell" if amount_bbox is not None else "logical_page",
                    **({"geometry_status": "unresolved"} if amount_bbox is None else {}),
                    "coordinate_system": "pdf_points_top_left",
                    "grid_id": f"mg_p{page}_repayment_{grid_index}",
                    "row": amount_band["index"],
                    "col": month,
                    "field_name": "overdue_amount",
                    **row_geometry_provenance,
                    **row_geometry_rejection,
                }
                if amount_bbox is not None:
                    localized_amount_bbox = _local_page_bbox(
                        amount_bbox,
                        logical_page=amount_logical_page,
                        base_page=page,
                        base_page_height=page_height,
                        coordinates_already_registered=(
                            isinstance(amount_line, Mapping)
                            and amount_line.get("coordinate_logical_page") is not None
                        ),
                        coordinate_status=(
                            str(amount_line.get("coordinate_status") or "")
                            if isinstance(amount_line, Mapping)
                            else ""
                        ),
                    )
                    if localized_amount_bbox is not None:
                        amount_ref_payload["bbox"] = localized_amount_bbox
                    else:
                        amount_ref_payload["geometry_scope"] = "logical_page"
                        amount_ref_payload["geometry_status"] = "unresolved"
                if enable_candidate_b_amount_pairing:
                    amount_recognition_audit["field_geometry_exact"] = bool(
                        row_month_geometry_exact
                        and amount_visual_col is not None
                        and amount_pair_status == "exact"
                        and amount_ref_payload.get("geometry_scope") == "cell"
                        and isinstance(amount_ref_payload.get("bbox"), list)
                    )
                amount_recognition_audit["source_ref"] = amount_ref_payload
                amount_cells.append(
                    build_cell(
                        row_band=amount_band,
                        col_band=amount_cell_col,
                        tokens=amt_tokens,
                        text=amount,
                        role="overdue_amount",
                        crop_ocr_text=amount_crop,
                        recognition_source=amount_recognition_source,
                        recognition_audit=amount_recognition_audit,
                    )
                )

            if (
                enable_candidate_b_amount_pairing
                and static_template is not None
                and observed_row_status in {"N", "*"}
                and (year, month) in record_months
            ):
                exact_status_geometry = bool(
                    row_alignment_exact and row_month_geometry_exact and visual_col is not None
                )
                document_status_glyph_observations.append(
                    {
                        "repayment_id": (f"mg_p{page}_repayment_{grid_index}:{year:04d}-{month:02d}"),
                        "grid_id": f"mg_p{page}_repayment_{grid_index}",
                        "page": status_logical_page,
                        "year": year,
                        "month": month,
                        "observed_status": observed_row_status,
                        "resolved_status": status,
                        "template": static_template,
                        "decisive_label": (
                            visual_status
                            if visual_status == observed_row_status
                            and visual_status in {"N", "*"}
                            and visual_confidence >= 0.95
                            else ""
                        ),
                        "decisive_confidence": float(visual_confidence),
                        "classifier_conflict": bool(visual_status and visual_status != observed_row_status),
                        "alignment_exact": bool(
                            row_alignment_exact and row_month_geometry_exact and (year, month) in record_months
                        ),
                        "exact_status_geometry": exact_status_geometry,
                        "status_bbox_key": tuple(round(float(value), 4) for value in st_cell.bbox),
                        "amount": amount or None,
                        "amount_pair_exact": bool(amount_pair_status == "exact" and amount not in {"", None}),
                        "status_amount_conflict": bool(status_amount_conflict),
                        "source_ref": dict(status_ref_payload),
                    }
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
                            observed_row_status in zero_overdue_statuses and amount not in {"", "0"}
                        ),
                        "alignment_exact": bool(
                            row_alignment_exact and row_month_geometry_exact and (year, month) in record_months
                        ),
                    }
                )

            independently_exact_amount = bool(
                enable_candidate_b_amount_pairing
                and amount not in {"", None}
                and amount_pair_status == "exact"
                and isinstance(amount_ref_payload, dict)
                and amount_ref_payload.get("geometry_scope") == "cell"
                and isinstance(amount_ref_payload.get("bbox"), list)
                and amount_recognition_audit.get("field_geometry_exact") is True
            )
            if (year, month) in record_months and (status or independently_exact_amount):
                refs = [dict(status_ref_payload)]
                if isinstance(amount_ref_payload, dict):
                    refs.append(dict(amount_ref_payload))
                records.append(
                    {
                        "year": year,
                        "month": month,
                        "status": status or "unknown",
                        "overdue_amount": amount or None,
                        "status_bbox": list(st_cell.bbox),
                        **({"amount_bbox": list(amount_bbox)} if amount_bbox else {}),
                        "source_cell_refs": refs,
                        "confidence": st_cell.confidence or 0.7,
                        **(
                            {
                                "extraction_status": "review",
                                "audit": {
                                    "reason": "status_value_withheld",
                                    "field_name": "status_code",
                                    "unresolved_fields": ["status_code"],
                                    "reported_amount_retained": independently_exact_amount,
                                },
                            }
                            if not status
                            else
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
    # Candidate B validates unresolved glyphs only after every repayment grid
    # in the document has been materialized.  Its stricter document-local bank
    # must not inherit the historical two-seed, same-grid consensus.
    prototypes = (
        {}
        if enable_candidate_b_amount_pairing
        else _scan_specific_status_prototypes(
            {
                status: observations
                for status, observations in static_status_seeds.items()
                if status not in static_status_contradictions
            }
        )
    )
    geometry_years: dict[tuple[float, ...], set[int]] = {}
    for bbox_key, geometry_year in static_status_geometry:
        geometry_years.setdefault(bbox_key, set()).add(geometry_year)
    reused_status_geometry = {
        bbox_key for bbox_key, geometry_year_values in geometry_years.items() if len(geometry_year_values) > 1
    }
    for observation in document_status_glyph_observations:
        if tuple(observation.get("status_bbox_key") or ()) in reused_status_geometry:
            observation["exact_status_geometry"] = False
            observation["geometry_reused_across_years"] = True
        observation.pop("status_bbox_key", None)
    if candidate_b_status_glyph_observations is not None:
        candidate_b_status_glyph_observations.extend(document_status_glyph_observations)
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
                "template_seed_count": int((prototypes.get(expected_status) or {}).get("seed_count") or 0),
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
            **(
                {"printed_anchor_provenance": deepcopy(anchor["printed_anchor_identity"])}
                if isinstance(anchor.get("printed_anchor_identity"), Mapping)
                else {}
            ),
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
            "static_amount_zero_validation": {
                "enabled": bool(enable_candidate_b_amount_pairing and enable_static_status_validation),
                "attempts": static_amount_zero_attempts,
                "resolved": static_amount_zero_resolved,
                "corrections": static_amount_zero_corrections,
                "unresolved": static_amount_zero_unresolved,
                "unavailable": static_amount_zero_unavailable,
                "ocr_invoked": False,
            },
            **({"candidate_b_amount_pairing": amount_pairing_by_year} if enable_candidate_b_amount_pairing else {}),
            "visual_month_geometry": visual_geometry_audit,
            **(
                {"visual_month_geometry_by_page": visual_geometry_audit_by_page}
                if enable_candidate_b_amount_pairing
                else {}
            ),
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
    candidate_b_status_glyph_observations: list[dict[str, Any]] | None = None,
    continuation_logical_pages: Iterable[int] = (),
    source_table_geometry_by_page: Mapping[
        str | int,
        Iterable[Mapping[str, Any]],
    ]
    | None = None,
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
        candidate_b_status_glyph_observations=(candidate_b_status_glyph_observations),
        continuation_logical_pages=continuation_logical_pages,
        source_table_geometry_by_page=source_table_geometry_by_page,
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


def _persisted_exact_field_ref(
    ref: Any,
    *,
    field_name: str,
    grid_id: str,
    row: int,
    col: int,
) -> bool:
    """Validate one field-local, page-local cell reference fail closed."""

    if not isinstance(ref, Mapping):
        return False
    if str(ref.get("field_name") or "") != field_name:
        return False
    if str(ref.get("grid_id") or "") != grid_id:
        return False
    if ref.get("geometry_scope") != "cell":
        return False
    geometry_status = ref.get("geometry_status")
    if geometry_status is not None and (
        not isinstance(geometry_status, str)
        or geometry_status not in {"", "exact", "accepted"}
    ):
        return False
    if ref.get("geometry_rejection"):
        return False
    if str(ref.get("coordinate_system") or "") != "pdf_points_top_left":
        return False
    if any(
        not isinstance(ref.get(key), int) or isinstance(ref.get(key), bool)
        for key in ("row", "col", "page", "logical_page")
    ):
        return False
    raw_bbox = ref.get("bbox")
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        return False
    try:
        ref_row = int(ref["row"])
        ref_col = int(ref["col"])
        ref_page = int(ref["page"])
        logical_page = int(ref["logical_page"])
        bbox = tuple(float(value) for value in raw_bbox)
    except (TypeError, ValueError):
        return False
    return bool(
        ref_row == row
        and ref_col == col
        and ref_page > 0
        and logical_page == ref_page
        and len(bbox) == 4
        and all(isfinite(value) for value in bbox)
        and bbox[2] > bbox[0]
        and bbox[3] > bbox[1]
    )


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
        str(value) for value in static_validation_audit.get("required_statuses") or ("N", "*") if value
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
        len(col_map) == len(set(col_map.values()))
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
            int(cell.get("row_index") or 0) for cell in row if str(cell.get("role") or "") == "overdue_amount"
        }
        if len(row_indices) != 1:
            continue
        row_idx = next(iter(row_indices))
        amount_rows_by_index.setdefault(row_idx, []).append(row)
        for cell in row:
            if str(cell.get("role") or "") != "overdue_amount":
                continue
            amount_cells_by_row_col.setdefault((row_idx, int(cell.get("col_index") or 0)), []).append(cell)

    status_year_counts: dict[int, int] = {}
    for row in status_rows:
        years = {
            int(match.group(0))
            for cell in row
            if str(cell.get("role") or "") == "year" and (match := _YEAR_RE.match(str(cell.get("text") or "").strip()))
        }
        if len(years) == 1:
            year = next(iter(years))
            status_year_counts[year] = status_year_counts.get(year, 0) + 1
    status_cell_by_year_month: dict[tuple[int, int], dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    for status_row in status_rows:
        status_row_indices = {
            int(cell.get("row_index") or 0) for cell in status_row if str(cell.get("role") or "") == "status"
        }
        status_years = {
            int(match.group(0))
            for cell in status_row
            if str(cell.get("role") or "") == "year" and (match := _YEAR_RE.match(str(cell.get("text") or "").strip()))
        }
        row_idx = next(iter(status_row_indices)) if len(status_row_indices) == 1 else -1
        row_year = next(iter(status_years)) if len(status_years) == 1 else None
        status_cells_by_col: dict[int, list[dict[str, Any]]] = {}
        for candidate_cell in status_row:
            if str(candidate_cell.get("role") or "") == "status":
                status_cells_by_col.setdefault(int(candidate_cell.get("col_index") or 0), []).append(candidate_cell)
        amount_row_idx = row_idx + 1
        paired_amount_rows = amount_rows_by_index.get(amount_row_idx, [])
        unique_status_row_role = bool(row_idx >= 0 and row_year is not None and status_year_counts.get(row_year) == 1)
        for cell in status_row:
            if str(cell.get("role") or "") != "status":
                continue
            col_idx = int(cell.get("col_index") or 0)
            month = col_map.get(col_idx)
            if row_year is not None and month and (row_year, month) in valid_months:
                status_cell_by_year_month[(row_year, month)] = cell
            status = str(cell.get("text") or "").strip()
            if not month:
                continue
            year = row_year
            if year is None:
                year = next((y for y, m in valid_months if m == month), None)
            if year is None or (year, month) not in valid_months:
                continue
            paired_amount_cells = amount_cells_by_row_col.get((amount_row_idx, col_idx), [])
            amount_cell = paired_amount_cells[0] if len(paired_amount_cells) == 1 else None
            amount_audit = dict(amount_cell.get("recognition_audit") or {}) if isinstance(amount_cell, dict) else {}
            recognition_audit = dict(cell.get("recognition_audit") or {})
            independent_field_geometry = bool(
                accept_exact_row_numeric_status
                or candidate_b_amount_pairing
                or "field_geometry_exact" in recognition_audit
                or "field_geometry_exact" in amount_audit
            )
            if not status and not independent_field_geometry:
                continue
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
                unique_month_columns and len(status_cells_by_col.get(col_idx, [])) == 1
            )
            structurally_unique_status_geometry = bool(
                unique_status_row_role and unique_status_cell_geometry
            )
            structurally_unique_amount_geometry = bool(
                unique_status_row_role
                and unique_month_columns
                and len(paired_amount_rows) == 1
                and len(paired_amount_cells) == 1
                and declared_pair_status in {"", "exact"}
            )
            bbox = cell.get("bbox")
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
                    "coordinate_system": "pdf_points_top_left",
                    **({"bbox": list(bbox)} if isinstance(bbox, list) and len(bbox) == 4 else {}),
                }
            ]
            amount_bbox = amount_cell.get("bbox") if isinstance(amount_cell, dict) else None
            persisted_amount_ref: Any = None
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
                        "coordinate_system": "pdf_points_top_left",
                        **(
                            {"bbox": list(amount_bbox)}
                            if isinstance(amount_bbox, list) and len(amount_bbox) == 4
                            else {}
                        ),
                    }
                )
            status_ref = source_refs[0]
            amount_ref = source_refs[-1] if isinstance(amount_cell, dict) else None
            status_field_mask = recognition_audit.get("field_geometry_exact")
            amount_field_mask = amount_audit.get("field_geometry_exact")
            if independent_field_geometry:
                exact_status_geometry = bool(
                    structurally_unique_status_geometry
                    and (status_field_mask is None or status_field_mask is True)
                    and (
                        status_field_mask is True and isinstance(persisted_status_ref, Mapping)
                        or status_field_mask is None and month_geometry_usable
                    )
                    and _persisted_exact_field_ref(
                        status_ref,
                        field_name="status",
                        grid_id=grid_id,
                        row=row_idx,
                        col=month,
                    )
                )
                exact_amount_geometry = bool(
                    structurally_unique_amount_geometry
                    and (amount_field_mask is None or amount_field_mask is True)
                    and (
                        amount_field_mask is True and isinstance(persisted_amount_ref, Mapping)
                        or amount_field_mask is None and month_geometry_usable
                    )
                    and _persisted_exact_field_ref(
                        amount_ref,
                        field_name="overdue_amount",
                        grid_id=grid_id,
                        row=amount_row_idx,
                        col=month,
                    )
                )
            else:
                # Other credit-report variants retain their established
                # structural row-pair contract. The stricter field-local
                # source-ref policy is deployed only by Candidate B (or an
                # explicitly persisted field mask), not by shared callers.
                exact_status_geometry = bool(
                    structurally_unique_status_geometry and month_geometry_usable
                )
                exact_amount_geometry = bool(
                    exact_status_geometry and structurally_unique_amount_geometry
                )
            exact_row_pair = exact_status_geometry and exact_amount_geometry
            if not exact_amount_geometry:
                amount = None
            amount_pair_status = declared_pair_status
            if not amount_pair_status:
                if len(paired_amount_rows) != 1:
                    amount_pair_status = "missing_amount_row" if not paired_amount_rows else "ambiguous_immediate_rows"
                elif len(paired_amount_cells) != 1:
                    amount_pair_status = (
                        "blank_amount_cell" if not paired_amount_cells else "duplicate_or_ambiguous_cell"
                    )
                elif amount is None:
                    amount_pair_status = "blank_amount_cell"
                else:
                    amount_pair_status = "exact"
            amount_pair_unresolved = bool(
                accept_exact_row_numeric_status and status and (amount_pair_status != "exact" or amount is None)
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
                "status": status or "unknown",
                "overdue_amount": amount,
                "source_cell_refs": source_refs,
                "confidence": (
                    float(cell.get("confidence") or 0.7)
                    if status
                    else 0.0
                ),
                "_exact_row_pair": exact_row_pair,
                "_exact_status_geometry": exact_status_geometry,
                "_exact_amount_geometry": exact_amount_geometry,
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
            if not status:
                record["raw_status"] = ""
                record["extraction_status"] = "review"
                record["audit"] = {
                    **recognition_audit,
                    "reason": str(
                        recognition_audit.get("reason")
                        or "status_value_withheld"
                    ),
                    "field_name": "status_code",
                    "unresolved_fields": (
                        ["status_code", "overdue_amount"]
                        if amount is None
                        else ["status_code"]
                    ),
                    "reported_amount_retained": amount is not None,
                }
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
        static_status_corroborated = not static_validation_enabled or str(record.get("recognition_source") or "") in {
            "static_glyph_shape_validation",
            "static_grid_template_consensus",
        }
        static_sensitive_exact_candidate = bool(status in static_sensitive_statuses and exact_status_geometry)
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
                    if status in {"1", "2", "3", "4", "5", "6", "7"} and exact_row_pair and exact_status_geometry
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
                **({"static_reason": prior_audit.get("reason")} if prior_audit.get("reason") else {}),
            }
    for record in records:
        for internal_key in (
            "_exact_row_pair",
            "_exact_status_geometry",
            "_exact_amount_geometry",
            "_paired_amount_explicit",
            "_paired_amount_positive",
        ):
            record.pop(internal_key, None)
    existing_months = {(int(record.get("year") or 0), int(record.get("month") or 0)) for record in records}
    for year, month in sorted(valid_months - existing_months):
        empty_cell = status_cell_by_year_month.get((year, month))
        empty_audit = dict(empty_cell.get("recognition_audit") or {}) if isinstance(empty_cell, dict) else {}
        empty_ref = empty_audit.get("source_ref")
        placeholder: dict[str, Any] = {
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
        if isinstance(empty_cell, dict):
            recognition_source = str(empty_cell.get("recognition_source") or "")
            if recognition_source and recognition_source != "tokens":
                placeholder["recognition_source"] = recognition_source
        if empty_audit:
            placeholder["audit"] = empty_audit
        records.append(placeholder)
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
        selected = (
            record if float(record.get("confidence") or 0.0) > float(current.get("confidence") or 0.0) else current
        )
        selected = dict(selected)
        merged_refs = _merged_source_cell_refs(current, record)
        if merged_refs:
            selected["source_cell_refs"] = merged_refs
        output[existing] = selected
    return output
