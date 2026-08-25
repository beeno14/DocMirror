# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed token lookup across the two sealed personal-detail stores."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

ExactTokenAtom = tuple[str, tuple[float, float, float, float], str]


def _value(item: Any, key: str, default: Any = None) -> Any:
    return item.get(key, default) if isinstance(item, Mapping) else getattr(item, key, default)


def _bbox(item: Any) -> tuple[float, float, float, float] | None:
    raw = _value(item, "bbox")
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        box = tuple(float(value) for value in raw)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in box) or box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def _text(item: Any) -> str:
    text = str(_value(item, "text", "") or "").strip()
    content = str(_value(item, "content", "") or "").strip()
    if text and content and text != content:
        return ""
    return text or content


def _resolved_atom(item: Any, token_id: str) -> ExactTokenAtom | None:
    text = _text(item)
    bbox = _bbox(item)
    if not text or bbox is None:
        return None
    return text, bbox, token_id


def resolve_exact_page_token_atoms(
    owner: Any,
    token_ids: Iterable[str],
    *,
    logical_page: int | None = None,
) -> tuple[ExactTokenAtom, ...] | None:
    """Resolve one exact closed token-ID set from immutable source evidence.

    The canonical evidence plane remains authoritative when it owns any of the
    requested IDs.  A page bundle is considered only when the plane owns none,
    and only with an explicit logical page.  Duplicate IDs, partial ownership,
    wrong-page tokens, non-singleton evidence ownership, or malformed token
    geometry all fail closed.
    """

    requested = tuple(str(value) for value in token_ids if str(value or ""))
    if not requested or len(requested) != len(set(requested)):
        return None

    plane = getattr(owner, "evidence_plane", None)
    if plane is None:
        plane = getattr(getattr(owner, "parse_result", None), "evidence_plane", None)
    evidence_store = getattr(plane, "evidence", None)
    text_atoms = getattr(evidence_store, "text_atoms", None)
    plane_matches: dict[str, Any] = {}
    if isinstance(text_atoms, list):
        for atom in text_atoms:
            atom_id = str(_value(atom, "id", "") or "")
            if atom_id not in requested:
                continue
            if atom_id in plane_matches:
                return None
            plane_matches[atom_id] = atom
    if plane_matches:
        if set(plane_matches) != set(requested):
            return None
        resolved = tuple(_resolved_atom(plane_matches[token_id], token_id) for token_id in requested)
        return tuple(atom for atom in resolved if atom is not None) if all(resolved) else None

    if not isinstance(logical_page, int) or isinstance(logical_page, bool) or logical_page <= 0:
        return None
    parse_result = getattr(owner, "parse_result", owner)
    domain_specific = getattr(getattr(parse_result, "entities", None), "domain_specific", None)
    bundles = domain_specific.get("_page_evidence_bundles") if isinstance(domain_specific, Mapping) else None
    if not isinstance(bundles, list):
        return None

    bundle_matches: dict[str, Any] = {}
    for bundle in bundles:
        if not isinstance(bundle, Mapping):
            continue
        try:
            bundle_page = int(bundle.get("page") or 0)
        except (TypeError, ValueError):
            bundle_page = 0
        tokens = bundle.get("tokens")
        if not isinstance(tokens, list):
            continue
        for token in tokens:
            if not isinstance(token, Mapping):
                continue
            token_id = str(token.get("token_id") or "")
            if token_id not in requested:
                continue
            try:
                token_page = int(token.get("page") or 0)
            except (TypeError, ValueError):
                token_page = 0
            evidence_ids = tuple(
                str(value)
                for value in token.get("evidence_ids") or ()
                if str(value or "")
            )
            if (
                token_id in bundle_matches
                or bundle_page != logical_page
                or token_page != logical_page
                or evidence_ids != (token_id,)
            ):
                return None
            bundle_matches[token_id] = token
    if set(bundle_matches) != set(requested):
        return None
    resolved = tuple(_resolved_atom(bundle_matches[token_id], token_id) for token_id in requested)
    return tuple(atom for atom in resolved if atom is not None) if all(resolved) else None


def exact_cell_visually_contains_dash_pair(
    owner: Any,
    cell: Any,
    *,
    logical_page: int,
    horizontal_fraction: tuple[float, float] = (0.0, 1.0),
) -> bool:
    """Prove that one evidence-sealed exact cell visibly contains only ``--``.

    This is a shape verifier, never OCR or a semantic correction. It accepts no
    page/template identity: callers must supply one exact cell and its logical
    page. Pale watermark pixels and tiny specks are ignored, while any third
    substantial component or malformed/uncentered pair fails closed.
    """

    if (
        str(_value(cell, "geometry_status", "") or "") != "exact"
        or not isinstance(logical_page, int)
        or isinstance(logical_page, bool)
        or logical_page <= 0
    ):
        return False
    try:
        fraction_left, fraction_right = (float(value) for value in horizontal_fraction)
    except (TypeError, ValueError):
        return False
    if not (0.0 <= fraction_left < fraction_right <= 1.0):
        return False
    evidence_ids = tuple(str(value) for value in _value(cell, "evidence_ids", ()) or () if str(value or ""))
    if len(evidence_ids) != 1:
        return False
    cell_bbox = _bbox(cell)
    resolver = getattr(owner, "_page_image_resolver", None)
    if cell_bbox is None or not callable(resolver):
        return False
    rendered = resolver(logical_page)
    if not isinstance(rendered, Mapping):
        return False
    image = rendered.get("image")
    shape = getattr(image, "shape", None)
    page_width = float(rendered.get("page_width") or 0.0)
    page_height = float(rendered.get("page_height") or 0.0)
    if not shape or len(shape) < 2 or page_width <= 0 or page_height <= 0:
        return False
    try:
        import cv2
        import numpy as np

        scale_x = float(shape[1]) / page_width
        scale_y = float(shape[0]) / page_height
        left = max(0, min(int(round(cell_bbox[0] * scale_x)), int(shape[1]) - 1))
        top = max(0, min(int(round(cell_bbox[1] * scale_y)), int(shape[0]) - 1))
        right = max(left + 1, min(int(round(cell_bbox[2] * scale_x)), int(shape[1])))
        bottom = max(top + 1, min(int(round(cell_bbox[3] * scale_y)), int(shape[0])))
        full_crop = np.asarray(image)[top:bottom, left:right]
        crop_left = int(round(full_crop.shape[1] * fraction_left))
        crop_right = int(round(full_crop.shape[1] * fraction_right))
        crop = full_crop[:, crop_left:crop_right]
        if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 16:
            return False
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        margin_x = max(2, int(round(gray.shape[1] * 0.04)))
        margin_y = max(2, int(round(gray.shape[0] * 0.14)))
        inner = gray[margin_y : gray.shape[0] - margin_y, margin_x : gray.shape[1] - margin_x]
        if inner.size == 0:
            return False

        def is_dash_pair(binary: Any) -> bool:
            count, _labels, stats, centers = cv2.connectedComponentsWithStats(binary, 8)
            components: list[tuple[int, int, int, int, int, float, float]] = []
            for index in range(1, count):
                x, y, width, height, area = (int(value) for value in stats[index])
                if area < 3 or width < 3:
                    continue
                components.append(
                    (x, y, width, height, area, float(centers[index][0]), float(centers[index][1]))
                )
            if len(components) != 2:
                return False
            first, second = sorted(components, key=lambda item: item[5])
            max_height = max(3, int(round(inner.shape[0] * 0.28)))
            if any(
                width < 1.8 * height or height > max_height
                for _x, _y, width, height, _area, _center_x, _center_y in components
            ):
                return False
            if abs(first[6] - second[6]) > max(2.0, inner.shape[0] * 0.12):
                return False
            gap = second[0] - (first[0] + first[2])
            if not (0 <= gap <= max(first[2], second[2]) * 2.5):
                return False
            pair_center_x = (first[5] + second[5]) / 2.0
            pair_center_y = (first[6] + second[6]) / 2.0
            if abs(pair_center_x - inner.shape[1] / 2.0) > max(6.0, inner.shape[1] * 0.08):
                return False
            return inner.shape[0] * 0.25 <= pair_center_y <= inner.shape[0] * 0.82

        _fixed_threshold, dark_binary = cv2.threshold(inner, 128, 255, cv2.THRESH_BINARY_INV)
        if is_dash_pair(dark_binary):
            return True
        _threshold, binary = cv2.threshold(inner, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return is_dash_pair(binary)
    except Exception:
        return False


__all__ = [
    "ExactTokenAtom",
    "exact_cell_visually_contains_dash_pair",
    "resolve_exact_page_token_atoms",
]
