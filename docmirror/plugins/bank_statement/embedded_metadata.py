# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lazy, source-bound recovery for metadata printed inside bank seals.

Digital statements remain native-text first. OCR is restricted to plausible
embedded seal images and, for very small curved identifiers, to a clipped
render of that image's source-page rectangle. No full page is OCRed here.
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from docmirror.plugins.bank_statement.work_cache import memoize_bank_document_work

_SEAL_MARKERS = (
    "零售业务电子凭证专用章",
    "电子回单业务专用章",
    "账户明细专用章",
    "明细回单专用章",
    "对账单专用章",
    "回单专用章",
    "业务专用章",
    "对账专用章",
    "电子专用章",
    "会计业务章",
)
_SEAL_MARKER_RE = re.compile("|".join(sorted(map(re.escape, _SEAL_MARKERS), key=len, reverse=True)))
_MIXED_SEAL_CODE_RE = re.compile(r"(?=[A-Z0-9]{8,32}\Z)(?=.*[A-Z])(?=.*\d)[A-Z0-9]+")
_NUMERIC_SEAL_CODE_RE = re.compile(r"\d{8,20}")
_ASCII_RUN_RE = re.compile(r"[A-Z0-9]{8,32}")
_BRANCH_RE = re.compile(r"[\u4e00-\u9fff·]{2,40}(?:支行|分行|营业部|信用社|信用联社)")
_BANK_RE = re.compile(
    r"[\u4e00-\u9fff·]{2,50}?(?:农村商业银行股份有限公司|银行股份有限公司|银行|信用联社)"
)


@dataclass(frozen=True)
class EmbeddedMetadataFact:
    field_key: str
    source_label: str
    value: str
    page: int
    page_id: str
    bbox: tuple[float, float, float, float] | None
    evidence_id: str
    confidence: float
    supporting_evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class EmbeddedMetadataResult:
    facts: tuple[EmbeddedMetadataFact, ...] = ()
    candidate_images: int = 0
    ocr_images: int = 0
    status: str = "not_needed"


@dataclass(frozen=True)
class _Observation:
    field_key: str
    value: str
    confidence: float
    variant: str
    evidence_ids: tuple[str, ...] = ()


def _value(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _page_number(atom: Any) -> int:
    page_id = str(_value(atom, "page_id", "") or "")
    match = re.search(r"(\d+)$", page_id)
    return int(match.group(1)) if match else 0


def _bbox(atom: Any) -> tuple[float, float, float, float] | None:
    for name in ("bbox", "source_bbox"):
        raw = _value(atom, name)
        if not isinstance(raw, (list, tuple)) or len(raw) < 4:
            continue
        try:
            return tuple(float(item) for item in raw[:4])  # type: ignore[return-value]
        except (TypeError, ValueError):
            continue
    return None


def _candidate_image_atoms(parse_result: Any) -> list[Any]:
    plane = _value(parse_result, "evidence_plane")
    evidence = _value(plane, "evidence")
    candidates: list[Any] = []
    for atom in list(_value(evidence, "image_atoms", []) or []):
        if str(_value(atom, "kind", "") or "") != "embedded_image":
            continue
        metadata = _value(atom, "metadata", {}) or {}
        try:
            width = int(_value(metadata, "width", 0) or 0)
            height = int(_value(metadata, "height", 0) or 0)
        except (TypeError, ValueError):
            continue
        if width < 80 or height < 40 or width * height > 8_000_000 or _bbox(atom) is None:
            continue
        aspect = max(width, height) / max(min(width, height), 1)
        if aspect <= 4.0:
            candidates.append(atom)
    return candidates


def _source_path(parse_result: Any) -> Path | None:
    provenance = _value(parse_result, "provenance")
    plane = _value(parse_result, "evidence_plane")
    source = _value(plane, "source")
    for raw_path in (
        _value(parse_result, "file_path", ""),
        _value(provenance, "file_path", ""),
        _value(source, "filename", ""),
    ):
        if not raw_path:
            continue
        try:
            path = Path(str(raw_path)).expanduser().resolve()
        except (OSError, RuntimeError, ValueError):
            continue
        if path.is_file() and path.suffix.lower() == ".pdf":
            return path
    return None


def _plausible_seal_image(image: Any) -> bool:
    """Reject monochrome square QR images before loading OCR work."""

    try:
        height, width = image.shape[:2]
        aspect = max(width, height) / max(min(width, height), 1)
        if aspect >= 1.15:
            return True
        import cv2

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturated_ratio = float((hsv[:, :, 1] >= 60).mean())
        return saturated_ratio >= 0.002
    except Exception:
        return False


def _compact(value: Any) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(value or ""))).upper()


def _word_parts(word: Any) -> tuple[str, float]:
    try:
        if isinstance(word, dict):
            return str(word.get("text") or "").strip(), float(word.get("confidence", 1.0) or 0.0)
        text = str(word[4]).strip()
        confidence_index = 8 if len(word) > 8 else 5
        return text, float(word[confidence_index])
    except (IndexError, TypeError, ValueError):
        return "", 0.0


def _valid_seal_code(value: str) -> bool:
    return bool(_MIXED_SEAL_CODE_RE.fullmatch(value) or _NUMERIC_SEAL_CODE_RE.fullmatch(value))


def _code_matches(tokens: Sequence[str]) -> list[str]:
    matches: list[str] = []
    for text in (*tokens, "".join(tokens)):
        for candidate in _ASCII_RUN_RE.findall(_compact(text)):
            if _valid_seal_code(candidate):
                matches.append(candidate)
    return list(dict.fromkeys(matches))


def _deduplicated_matches(pattern: re.Pattern[str], tokens: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(match for token in tokens for match in pattern.findall(token) if match))


def _branch_matches(tokens: Sequence[str]) -> list[str]:
    """Prefer an exact branch token; remove only an explicit legal-bank prefix."""

    direct = _deduplicated_matches(_BRANCH_RE, tokens)
    if direct:
        cleaned: list[str] = []
        for value in direct:
            branch = value
            for bank_suffix in ("农村商业银行股份有限公司", "银行股份有限公司"):
                marker = branch.rfind(bank_suffix)
                if marker >= 0 and branch[marker + len(bank_suffix) :]:
                    branch = branch[marker + len(bank_suffix) :]
                    break
            cleaned.append(branch)
        return list(dict.fromkeys(cleaned))
    joined = "".join(tokens)
    stripped: list[str] = []
    for value in _deduplicated_matches(_BRANCH_RE, (joined,)):
        branch = value
        for bank_suffix in ("农村商业银行股份有限公司", "银行股份有限公司", "银行", "信用联社"):
            marker = branch.rfind(bank_suffix)
            if marker >= 0 and branch[marker + len(bank_suffix) :]:
                branch = branch[marker + len(bank_suffix) :]
                break
        if _BRANCH_RE.fullmatch(branch):
            stripped.append(branch)
    return list(dict.fromkeys(stripped))


def _bank_matches(tokens: Sequence[str]) -> list[str]:
    direct = _deduplicated_matches(_BANK_RE, tokens)
    joined = _deduplicated_matches(_BANK_RE, ("".join(tokens),))
    # Curved legal names are sometimes detected as ``...有限公`` + ``司``
    # with low-confidence digits between the two Chinese runs.  Dropping only
    # non-Chinese OCR noise lets the complete printed legal name remain
    # recoverable without inventing an unprinted suffix.
    chinese_joined = "".join(re.sub(r"[^\u4e00-\u9fff·]", "", token) for token in tokens)
    stitched = _deduplicated_matches(_BANK_RE, (chinese_joined,))
    matches = list(dict.fromkeys([*direct, *joined, *stitched]))
    registered_legal_names: list[str] = []
    try:
        from docmirror.plugins.bank_statement.institution import detect_registered_institution

        for value in matches:
            registered = detect_registered_institution(value)
            if not registered:
                continue
            legal_name = (
                registered
                if registered.endswith("股份有限公司")
                else f"{registered}股份有限公司"
            )
            if legal_name in value:
                registered_legal_names.append(legal_name)
    except Exception:
        pass
    return list(dict.fromkeys([*matches, *registered_legal_names]))


def _observations_from_words(
    words: Iterable[Any],
    *,
    variant: str,
    evidence_ids: tuple[str, ...] = (),
) -> list[_Observation]:
    raw_tokens: list[tuple[str, float]] = []
    for word in words:
        text, confidence = _word_parts(word)
        if text:
            raw_tokens.append((text, confidence))
    tokens = [(text, confidence) for text, confidence in raw_tokens if confidence >= 0.70]
    if not tokens:
        return []
    texts = [unicodedata.normalize("NFKC", text).strip() for text, _confidence in tokens]
    compact_texts = [_compact(text) for text in texts]
    joined = "".join(compact_texts)
    joined_confidence = min((confidence for _text, confidence in tokens), default=0.0)
    observations: list[_Observation] = []

    for marker in dict.fromkeys(_SEAL_MARKER_RE.findall(joined)):
        marker_confidence = max(
            (confidence for text, confidence in tokens if marker in _compact(text)),
            default=joined_confidence,
        )
        observations.append(_Observation("seal_type", marker, marker_confidence, variant, evidence_ids))

    for code in _code_matches(compact_texts):
        code_confidence = max(
            (confidence for text, confidence in tokens if code in _compact(text)),
            default=joined_confidence,
        )
        observations.append(_Observation("seal_code", code, code_confidence, variant, evidence_ids))

    for branch in _branch_matches(texts):
        confidence = max(
            (item_confidence for text, item_confidence in tokens if branch in text),
            default=joined_confidence,
        )
        observations.append(_Observation("issuing_branch", branch, confidence, variant, evidence_ids))

    branch_values = {observation.value for observation in observations if observation.field_key == "issuing_branch"}
    has_legal_prefix = any(_compact(text).endswith("银行股份有限公") for text, _confidence in tokens)
    bank_tokens = [
        (text, confidence)
        for text, confidence in raw_tokens
        if confidence >= 0.70
        or (has_legal_prefix and confidence >= 0.55 and _compact(text) == "司")
    ]
    bank_texts = [unicodedata.normalize("NFKC", text).strip() for text, _confidence in bank_tokens]
    bank_joined_confidence = min((confidence for _text, confidence in bank_tokens), default=0.0)
    for bank in _bank_matches(bank_texts):
        if bank in branch_values:
            continue
        confidence = max(
            (item_confidence for text, item_confidence in bank_tokens if bank in text),
            default=bank_joined_confidence,
        )
        observations.append(_Observation("issuing_bank", bank, confidence, variant, evidence_ids))
    return observations


def _facts_from_ocr_words(
    words: Iterable[Any],
    *,
    page: int,
    page_id: str,
    bbox: tuple[float, float, float, float] | None,
    evidence_id: str,
) -> list[EmbeddedMetadataFact]:
    """Compatibility helper used by focused tests for one OCR observation."""

    observations = _observations_from_words(words, variant="single", evidence_ids=(evidence_id,))
    fields = {observation.field_key for observation in observations}
    if "seal_type" not in fields and not {"issuing_branch", "seal_code"} <= fields:
        return []
    labels = {
        "seal_type": "印章类型",
        "issuing_branch": "业务印章签发机构",
        "seal_code": "业务印章编码",
        "issuing_bank": "业务印章签发银行",
    }
    unique: dict[tuple[str, str], _Observation] = {}
    for observation in observations:
        key = (observation.field_key, observation.value)
        if key not in unique or observation.confidence > unique[key].confidence:
            unique[key] = observation
    return [
        EmbeddedMetadataFact(
            field_key=observation.field_key,
            source_label=labels[observation.field_key],
            value=observation.value,
            page=page,
            page_id=page_id,
            bbox=bbox,
            evidence_id=evidence_id,
            confidence=observation.confidence,
            supporting_evidence_ids=observation.evidence_ids,
        )
        for observation in unique.values()
    ]


def _bbox_distance(left: tuple[float, float, float, float], right: Sequence[Any]) -> float:
    try:
        values = tuple(float(value) for value in right[:4])
    except (TypeError, ValueError):
        return float("inf")
    return sum(abs(a - b) for a, b in zip(left, values, strict=True))


def _inline_image_bytes(page: Any, atom: Any) -> bytes:
    target_bbox = _bbox(atom)
    if target_bbox is None:
        return b""
    metadata = _value(atom, "metadata", {}) or {}
    try:
        target_width = int(_value(metadata, "width", 0) or 0)
        target_height = int(_value(metadata, "height", 0) or 0)
        blocks = (page.get_text("dict") or {}).get("blocks") or ()
    except Exception:
        return b""
    matches: list[tuple[float, bytes]] = []
    for block in blocks:
        if not isinstance(block, dict) or int(block.get("type", -1)) != 1:
            continue
        image_bytes = block.get("image") or b""
        if not image_bytes:
            continue
        width = int(block.get("width") or 0)
        height = int(block.get("height") or 0)
        dimension_penalty = 0.0 if (width, height) == (target_width, target_height) else 20.0
        distance = _bbox_distance(target_bbox, block.get("bbox") or ()) + dimension_penalty
        matches.append((distance, image_bytes))
    if not matches:
        return b""
    distance, image_bytes = min(matches, key=lambda item: item[0])
    return image_bytes if distance <= 25.0 else b""


def _image_bytes(document: Any, page: Any, atom: Any) -> bytes:
    metadata = _value(atom, "metadata", {}) or {}
    try:
        xref = int(_value(metadata, "xref", 0) or 0)
    except (TypeError, ValueError):
        xref = 0
    if xref > 0:
        try:
            image_bytes = document.extract_image(xref).get("image") or b""
            if image_bytes:
                return image_bytes
        except Exception:
            pass
    return _inline_image_bytes(page, atom)


def _native_observations(
    parse_result: Any,
    *,
    page: int,
    image_bbox: tuple[float, float, float, float],
) -> list[_Observation]:
    try:
        from docmirror.plugins._runtime.evidence_access import text_atoms

        atoms = text_atoms(parse_result)
    except Exception:
        return []
    words: list[dict[str, Any]] = []
    evidence_ids: list[str] = []
    x0, y0, x1, y1 = image_bbox
    for atom in atoms:
        if _page_number(atom) != page:
            continue
        atom_bbox = _bbox(atom)
        if atom_bbox is None:
            continue
        ax0, ay0, ax1, ay1 = atom_bbox
        overlap_x = min(x1, ax1) - max(x0, ax0)
        overlap_y = min(y1, ay1) - max(y0, ay0)
        center_x = (ax0 + ax1) / 2.0
        center_y = (ay0 + ay1) / 2.0
        if not (
            (overlap_x > 0.0 and overlap_y > 0.0)
            or (x0 - 2 <= center_x <= x1 + 2 and y0 - 2 <= center_y <= y1 + 2)
        ):
            continue
        text = str(_value(atom, "text", "") or "").strip()
        if not text:
            continue
        evidence_id = str(_value(atom, "id", "") or "")
        if evidence_id:
            evidence_ids.append(evidence_id)
        words.append({"text": text, "confidence": 1.0})
    return _observations_from_words(
        words,
        variant="native_text_in_image_bbox",
        evidence_ids=tuple(dict.fromkeys(evidence_ids)),
    )


def _ordinary_ocr_observations(
    image: Any,
    engine: Any,
    cv2: Any,
    *,
    variant_prefix: str = "fixed_scale",
) -> list[_Observation]:
    observations: list[_Observation] = []
    for scale in (1.0, 1.5, 2.0):
        scaled = image if scale == 1.0 else cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )
        try:
            words = list(engine.detect_image_words(scaled, multi_scale=False) or [])
        except Exception:
            words = []
        observations.extend(_observations_from_words(words, variant=f"{variant_prefix}_{scale:g}"))
    return observations


def _forced_multiline_code_observations(image: Any, engine: Any) -> list[_Observation]:
    """Recover one logical identifier printed across aligned seal text lines."""

    try:
        detected = list(engine.detect_image_words(image, multi_scale=False) or [])
    except Exception:
        return []
    code_lines: list[tuple[float, float, float, float, str, float]] = []
    for word in detected:
        text, confidence = _word_parts(word)
        normalized = re.sub(r"[^A-Z0-9]", "", _compact(text))
        try:
            x0, y0, x1, y1 = (float(word[index]) for index in range(4))
        except (IndexError, TypeError, ValueError):
            continue
        if confidence >= 0.80 and _MIXED_SEAL_CODE_RE.fullmatch(normalized):
            code_lines.append((x0, y0, x1, y1, normalized, confidence))
    code_lines.sort(key=lambda item: (item[1], item[0]))
    if len(code_lines) < 2:
        return []

    groups: list[list[tuple[float, float, float, float, str, float]]] = []
    for line in code_lines:
        if not groups:
            groups.append([line])
            continue
        previous = groups[-1][-1]
        line_height = max(line[3] - line[1], 1.0)
        previous_height = max(previous[3] - previous[1], 1.0)
        overlap = min(line[2], previous[2]) - max(line[0], previous[0])
        narrower_width = max(min(line[2] - line[0], previous[2] - previous[0]), 1.0)
        vertical_gap = line[1] - previous[3]
        if overlap >= narrower_width * 0.65 and -2.0 <= vertical_gap <= max(line_height, previous_height) * 1.5:
            groups[-1].append(line)
        else:
            groups.append([line])

    observations: list[_Observation] = []
    for group_index, group in enumerate(groups, start=1):
        if len(group) < 2:
            continue
        regions: list[tuple[int, int, int, int]] = []
        for x0, y0, x1, y1, _text, _confidence in group:
            width = max(x1 - x0, 1.0)
            height = max(y1 - y0, 1.0)
            regions.append(
                (
                    max(0, int(math.floor(x0 - width * 0.18))),
                    max(0, int(math.floor(y0 - height))),
                    min(int(image.shape[1]), int(math.ceil(x1 + width * 0.18))),
                    min(int(image.shape[0]), int(math.ceil(y1 + height * 0.45))),
                )
            )
        try:
            forced = list(engine.force_recognize_regions(image, regions) or [])
        except Exception:
            forced = []
        forced_by_region = {
            tuple(int(round(float(value))) for value in word[:4]): _word_parts(word)
            for word in forced
        }
        selected_tokens: list[str] = []
        selected_confidences: list[float] = []
        for region, line in zip(regions, group, strict=True):
            forced_text, forced_confidence = forced_by_region.get(region, ("", 0.0))
            normalized = re.sub(r"[^A-Z0-9]", "", _compact(forced_text))
            if forced_confidence >= 0.80 and _MIXED_SEAL_CODE_RE.fullmatch(normalized):
                selected_tokens.append(normalized)
                selected_confidences.append(forced_confidence)
            else:
                selected_tokens.append(line[4])
                selected_confidences.append(line[5])
        value = "".join(selected_tokens)
        if _MIXED_SEAL_CODE_RE.fullmatch(value):
            observations.append(
                _Observation(
                    "seal_code",
                    value,
                    min(selected_confidences),
                    f"forced_multiline_code_{group_index}",
                )
            )
    return observations


def _polar_ocr_observations(image: Any, engine: Any, cv2: Any, np: Any) -> list[_Observation]:
    """Unwrap the outer ellipse so curved legal issuer text becomes one line."""

    del np  # Kept explicit in the call signature with other optional CV dependencies.
    try:
        height, width = image.shape[:2]
        side = max(height, width)
        target = max(300, side)
        square = cv2.resize(image, (target, target), interpolation=cv2.INTER_CUBIC)
        radius = target / 2.0
        angle_samples = max(1440, int(2.0 * math.pi * radius * 1.5))
        polar = cv2.warpPolar(
            square,
            (int(radius), angle_samples),
            (target / 2.0, target / 2.0),
            radius,
            cv2.WARP_POLAR_LINEAR | cv2.WARP_FILL_OUTLIERS,
        )
        unwrapped = cv2.rotate(polar, cv2.ROTATE_90_COUNTERCLOCKWISE)
    except Exception:
        return []
    observations: list[_Observation] = []
    for index, (start, end) in enumerate(((0.04, 0.36), (0.06, 0.42), (0.0, 0.45)), start=1):
        band = unwrapped[int(radius * start) : max(int(radius * end), int(radius * start) + 1), :]
        if not getattr(band, "size", 0):
            continue
        band = cv2.resize(band, None, fx=0.5, fy=1.0, interpolation=cv2.INTER_AREA)
        try:
            words = list(engine.detect_image_words(band, multi_scale=False) or [])
        except Exception:
            words = []
        observations.extend(_observations_from_words(words, variant=f"polar_outer_band_{index}"))
    return observations


def _render_bbox_image(
    page: Any,
    bbox: tuple[float, float, float, float],
    fitz: Any,
    np: Any,
    cv2: Any,
) -> Any:
    try:
        pixmap = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), clip=fitz.Rect(bbox), alpha=False)
        channels = int(pixmap.n)
        image = np.frombuffer(pixmap.samples, np.uint8).reshape(pixmap.height, pixmap.width, channels)
        if channels >= 3:
            return cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2BGR)
    except Exception:
        pass
    return None


def _bottom_code_observations(
    page: Any,
    bbox: tuple[float, float, float, float],
    engine: Any,
    fitz: Any,
    np: Any,
    cv2: Any,
) -> list[_Observation]:
    """Recover tiny curved codes from a source-page render, never raw upscaling."""

    image = _render_bbox_image(page, bbox, fitz, np, cv2)
    if image is None:
        return []
    height, width = image.shape[:2]
    regions: list[tuple[int, int, int, int]] = []
    for y0, y1 in ((0.82, 0.97), (0.84, 0.97), (0.85, 0.98), (0.86, 0.98)):
        for x0, x1 in ((0.24, 0.76), (0.25, 0.75), (0.26, 0.74)):
            regions.append((int(width * x0), int(height * y0), int(width * x1), int(height * y1)))
    try:
        recognized = list(engine.force_recognize_regions(image, regions) or [])
    except Exception:
        return []
    observations: list[_Observation] = []
    for index, word in enumerate(recognized, start=1):
        text, confidence = _word_parts(word)
        normalized = re.sub(r"[^A-Z0-9]", "", _compact(text))
        if confidence >= 0.80 and _valid_seal_code(normalized):
            observations.append(_Observation("seal_code", normalized, confidence, f"page_render_bottom_{index}"))
    return observations


def _has_seal_context(observations: Sequence[_Observation]) -> bool:
    fields = {observation.field_key for observation in observations}
    return bool(
        "seal_type" in fields
        or "issuing_bank" in fields
        or {"issuing_branch", "seal_code"} <= fields
    )


def _select_observation(
    field_key: str,
    observations: Sequence[_Observation],
) -> tuple[str, float, tuple[str, ...]] | None:
    grouped: dict[str, list[_Observation]] = defaultdict(list)
    for observation in observations:
        if observation.field_key == field_key and observation.value:
            grouped[observation.value].append(observation)
    qualified: list[tuple[int, int, float, int, str, list[_Observation]]] = []
    all_bottom_count = sum(
        observation.field_key == "seal_code"
        and observation.variant.startswith("page_render_bottom_")
        for observation in observations
    )
    for value, items in grouped.items():
        variants = {item.variant for item in items}
        maximum = max(item.confidence for item in items)
        native = any(item.variant == "native_text_in_image_bbox" for item in items)
        bottom_count = sum(item.variant.startswith("page_render_bottom_") for item in items)
        forced_multiline = any(item.variant.startswith("forced_multiline_code_") for item in items)
        numeric_code = field_key == "seal_code" and bool(_NUMERIC_SEAL_CODE_RE.fullmatch(value))
        legal_bank = field_key == "issuing_bank" and value.endswith("银行股份有限公司")
        if field_key == "seal_type":
            accepted = maximum >= 0.70
        elif numeric_code:
            # A long account number inside a broad image rectangle is not a
            # seal identifier. Numeric-only codes require repeated recognition
            # from tightly clipped bottom-of-seal regions.
            accepted = (
                bottom_count >= 6
                and bottom_count * 2 > all_bottom_count
                and maximum >= 0.90
            )
        elif native:
            accepted = True
        elif field_key == "issuing_bank":
            accepted = (
                maximum >= 0.97
                or (len(variants) >= 2 and maximum >= 0.82)
                or (legal_bank and len(variants) >= 2 and maximum >= 0.55)
            )
        elif field_key == "seal_code" and forced_multiline:
            accepted = maximum >= 0.85
        else:
            accepted = maximum >= 0.96 or (len(variants) >= 2 and maximum >= 0.75)
        if accepted:
            priority = 3 if numeric_code else 2 if forced_multiline or legal_bank else 1
            qualified.append((priority, len(variants), maximum, len(value), value, items))
    if not qualified:
        return None
    qualified.sort(reverse=True)
    best = qualified[0]
    if len(qualified) > 1 and best[:3] == qualified[1][:3] and best[4] != qualified[1][4]:
        return None
    evidence_ids = tuple(
        dict.fromkeys(evidence_id for item in best[5] for evidence_id in item.evidence_ids if evidence_id)
    )
    return best[4], best[2], evidence_ids


@memoize_bank_document_work
def extract_embedded_business_metadata(parse_result: Any) -> EmbeddedMetadataResult:
    """Recover only consensus-backed facts from plausible embedded seal images."""

    candidates = _candidate_image_atoms(parse_result)
    if not candidates:
        return EmbeddedMetadataResult()
    path = _source_path(parse_result)
    if path is None:
        return EmbeddedMetadataResult(candidate_images=len(candidates), status="source_unavailable")
    try:
        import cv2
        import fitz
        import numpy as np

        from docmirror.ocr.vision.rapidocr_engine import get_ocr_engine

        engine = get_ocr_engine()
    except Exception:
        return EmbeddedMetadataResult(candidate_images=len(candidates), status="ocr_unavailable")

    try:
        document = fitz.open(path)
    except Exception:
        return EmbeddedMetadataResult(candidate_images=len(candidates), status="source_unavailable")

    facts: list[EmbeddedMetadataFact] = []
    ocr_images = 0
    digest_cache: dict[str, tuple[_Observation, ...]] = {}
    labels = {
        "seal_type": "印章类型",
        "issuing_branch": "业务印章签发机构",
        "seal_code": "业务印章编码",
        "issuing_bank": "业务印章签发银行",
    }
    try:
        for atom in candidates:
            page_number = _page_number(atom)
            image_bbox = _bbox(atom)
            if page_number <= 0 or image_bbox is None or page_number > len(document):
                continue
            page = document[page_number - 1]
            image_bytes = _image_bytes(document, page, atom)
            if not image_bytes:
                continue
            try:
                image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            except Exception:
                image = None
            if image is None or not _plausible_seal_image(image):
                continue
            digest = hashlib.sha256(image_bytes).hexdigest()
            if digest not in digest_cache:
                observations = _ordinary_ocr_observations(image, engine, cv2)
                observations.extend(_forced_multiline_code_observations(image, engine))
                observations.extend(_polar_ocr_observations(image, engine, cv2, np))
                digest_cache[digest] = tuple(observations)
                ocr_images += 1

            evidence_id = str(_value(atom, "id", "") or f"embedded_image:{page_number}:{digest[:12]}")
            observations = [
                _Observation(
                    item.field_key,
                    item.value,
                    item.confidence,
                    item.variant,
                    item.evidence_ids or (evidence_id,),
                )
                for item in digest_cache[digest]
            ]
            observations.extend(_native_observations(parse_result, page=page_number, image_bbox=image_bbox))
            if _has_seal_context(observations):
                rendered = _render_bbox_image(page, image_bbox, fitz, np, cv2)
                if rendered is not None:
                    observations.extend(
                        _ordinary_ocr_observations(
                            rendered,
                            engine,
                            cv2,
                            variant_prefix="source_page_render_scale",
                        )
                    )
                observations.extend(_bottom_code_observations(page, image_bbox, engine, fitz, np, cv2))
            if not _has_seal_context(observations):
                continue

            page_id = str(_value(atom, "page_id", "") or f"page:{page_number:04d}")
            for field_key in ("seal_type", "issuing_branch", "seal_code", "issuing_bank"):
                selected = _select_observation(field_key, observations)
                if selected is None:
                    continue
                value, confidence, supporting_ids = selected
                facts.append(
                    EmbeddedMetadataFact(
                        field_key=field_key,
                        source_label=labels[field_key],
                        value=value,
                        page=page_number,
                        page_id=page_id,
                        bbox=image_bbox,
                        evidence_id=evidence_id,
                        confidence=confidence,
                        supporting_evidence_ids=tuple(dict.fromkeys((evidence_id, *supporting_ids))),
                    )
                )
    finally:
        document.close()

    unique = {
        (fact.field_key, fact.value, fact.page, fact.evidence_id): fact
        for fact in facts
        if fact.page > 0 and fact.value
    }
    return EmbeddedMetadataResult(
        facts=tuple(unique.values()),
        candidate_images=len(candidates),
        ocr_images=ocr_images,
        status="complete" if unique else "no_business_metadata_found",
    )


__all__ = [
    "EmbeddedMetadataFact",
    "EmbeddedMetadataResult",
    "extract_embedded_business_metadata",
]
