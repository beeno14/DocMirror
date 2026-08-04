# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validated logical-page geometry for scanned personal detailed reports.

The core extraction pipeline remains the authority for page splitting.  This
module verifies that the resulting one-page or two-page topology is safe to
reuse for plugin-owned OCR.  A bounded recovery reruns the core splitter only
when a logical page's stored transform cannot be rendered, and only when the
recovered slice can be matched to the existing logical page without changing
its coordinate system.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from docmirror.plugins.credit_report.page_image_resolver import LogicalPageImageResolver

_PRINTED_FOOTER_RE = re.compile(r"第\s*(?P<page>\d+)\s*页\s*[,，]?\s*共\s*(?P<total>\d+)\s*页")


@dataclass(frozen=True)
class LogicalPageGeometry:
    logical_page: int
    source_page: int
    width: float
    height: float
    split_kind: str
    segment_index: int | None
    selected_rotation: int
    source_crop_bbox: tuple[float, float, float, float] | None
    transform_usable: bool
    issues: tuple[str, ...] = ()


class PersonalDetailPageTopology:
    """Validate the canonical single-page/two-page layouts owned by core."""

    def __init__(self, parse_result: Any) -> None:
        self._pages: dict[int, Any] = {}
        self._geometry: dict[int, LogicalPageGeometry] = {}
        self._logicals_by_source: dict[int, tuple[int, ...]] = {}
        self._invalid_sources: set[int] = set()
        self._issues: list[str] = []
        self._build(parse_result)

    def _build(self, parse_result: Any) -> None:
        grouped: dict[int, list[int]] = {}
        for fallback, page in enumerate(getattr(parse_result, "pages", None) or [], start=1):
            logical = int(getattr(page, "page_number", 0) or fallback)
            if logical <= 0 or logical in self._pages:
                self._issues.append(f"invalid_or_duplicate_logical_page:{logical}")
                continue
            geometry = _page_geometry(page, logical)
            self._pages[logical] = page
            self._geometry[logical] = geometry
            grouped.setdefault(geometry.source_page, []).append(logical)
            self._issues.extend(f"logical_page:{logical}:{issue}" for issue in geometry.issues)

        for source, logicals in grouped.items():
            ordered = self._validate_source_group(source, logicals)
            self._logicals_by_source[source] = ordered

    def _validate_source_group(self, source: int, logicals: list[int]) -> tuple[int, ...]:
        geometries = [self._geometry[logical] for logical in logicals]
        if source <= 0 or not (1 <= len(geometries) <= 2):
            self._invalidate(source, f"unsupported_logical_page_count:{len(geometries)}")
            return tuple(sorted(logicals))
        if any(not geometry.transform_usable for geometry in geometries):
            self._invalidate(source, "unusable_coordinate_transform")

        if len(geometries) == 1:
            geometry = geometries[0]
            if geometry.split_kind not in {"none", "two_page_spread"}:
                self._invalidate(source, f"unsupported_split_kind:{geometry.split_kind or 'missing'}")
            if geometry.split_kind == "two_page_spread" and geometry.segment_index not in {0, 1}:
                self._invalidate(source, "spread_segment_missing")
            return (geometry.logical_page,)

        if any(geometry.split_kind != "two_page_spread" for geometry in geometries):
            self._invalidate(source, "two_logical_pages_without_spread_metadata")
        segments = {geometry.segment_index for geometry in geometries}
        if segments != {0, 1}:
            self._invalidate(source, "spread_segments_not_zero_and_one")
        rotations = {geometry.selected_rotation for geometry in geometries}
        if len(rotations) != 1:
            self._invalidate(source, "spread_rotation_mismatch")
        first_crop, second_crop = (geometry.source_crop_bbox for geometry in geometries)
        if first_crop is None or second_crop is None or _overlap_ratio(first_crop, second_crop) > 0.02:
            self._invalidate(source, "spread_crops_overlap_or_missing")

        return tuple(
            geometry.logical_page
            for geometry in sorted(
                geometries,
                key=lambda item: (
                    item.segment_index if item.segment_index is not None else 99,
                    item.logical_page,
                ),
            )
        )

    def _invalidate(self, source: int, issue: str) -> None:
        self._invalid_sources.add(source)
        self._issues.append(f"source_page:{source}:{issue}")

    def geometry(self, logical_page: int) -> LogicalPageGeometry | None:
        return self._geometry.get(int(logical_page or 0))

    def page(self, logical_page: int) -> Any | None:
        return self._pages.get(int(logical_page or 0))

    def logicals_for_source(self, source_page: int) -> tuple[int, ...]:
        return self._logicals_by_source.get(int(source_page or 0), ())

    def ordered_pair(self, logical_pages: Iterable[int]) -> tuple[int, int] | None:
        requested = {int(page) for page in logical_pages if int(page) > 0}
        if len(requested) != 2:
            return None
        sources = {
            geometry.source_page
            for logical in requested
            if (geometry := self._geometry.get(logical)) is not None
        }
        if len(sources) != 1:
            return None
        source = next(iter(sources))
        ordered = self._logicals_by_source.get(source, ())
        if source in self._invalid_sources or len(ordered) != 2 or set(ordered) != requested:
            return None
        return (ordered[0], ordered[1])

    def transform_usable(self, logical_page: int) -> bool:
        geometry = self.geometry(logical_page)
        return bool(
            geometry is not None
            and geometry.transform_usable
            and geometry.source_page not in self._invalid_sources
        )

    def audit(self) -> dict[str, Any]:
        single_sources = 0
        double_sources = 0
        partial_spread_sources = 0
        for logicals in self._logicals_by_source.values():
            if len(logicals) == 2:
                double_sources += 1
            elif logicals:
                geometry = self._geometry[logicals[0]]
                if geometry.split_kind == "two_page_spread":
                    partial_spread_sources += 1
                else:
                    single_sources += 1
        return {
            "valid": not self._invalid_sources and not any(
                issue.startswith("invalid_or_duplicate_logical_page:") for issue in self._issues
            ),
            "logical_page_count": len(self._geometry),
            "source_page_count": len(self._logicals_by_source),
            "single_page_sources": single_sources,
            "double_page_sources": double_sources,
            "partial_spread_sources": partial_spread_sources,
            "invalid_source_pages": sorted(self._invalid_sources),
            "issues": list(self._issues),
            "logical_pages_by_source": {
                str(source): list(logicals) for source, logicals in sorted(self._logicals_by_source.items())
            },
        }


class PersonalDetailLogicalPageImageResolver:
    """Render only topology-validated logical pages for plugin OCR repair."""

    def __init__(
        self,
        parse_result: Any,
        *,
        zoom: float = 3.0,
        topology: PersonalDetailPageTopology | None = None,
    ) -> None:
        self._parse_result = parse_result
        self._file_path = Path(str(getattr(parse_result, "file_path", "") or ""))
        self._zoom = max(1.0, float(zoom))
        self.topology = topology or PersonalDetailPageTopology(parse_result)
        self._base = LogicalPageImageResolver(parse_result, zoom=self._zoom)
        self._cache: dict[int, dict[str, Any] | None] = {}
        self._recoveries: set[int] = set()
        self._supplemental_cache: dict[int, tuple[dict[str, Any], ...]] = {}
        self._supplemental_recoveries: list[dict[str, Any]] = []

    def __call__(self, logical_page: int) -> dict[str, Any] | None:
        logical = int(logical_page or 0)
        if logical in self._cache:
            return self._cache[logical]
        rendered = self._base(logical) if self.topology.transform_usable(logical) else None
        if rendered is None:
            rendered = self._recover_with_core_splitter(logical)
            if rendered is not None:
                self._recoveries.add(logical)
        self._cache[logical] = rendered
        return rendered

    def clear(self) -> None:
        self._base.clear()
        self._cache.clear()
        self._supplemental_cache.clear()

    def audit(self) -> dict[str, Any]:
        return {
            **self.topology.audit(),
            "recovered_logical_pages": sorted(self._recoveries),
            "supplemental_spread_recoveries": list(self._supplemental_recoveries),
        }

    def supplemental_spread_slices(self, source_pages: Iterable[int]) -> list[dict[str, Any]]:
        """Recover footer-confirmed missing halves without inventing logical pages."""
        result: list[dict[str, Any]] = []
        for source_page in sorted({int(page) for page in source_pages if int(page) > 0}):
            if source_page not in self._supplemental_cache:
                self._supplemental_cache[source_page] = tuple(
                    self._recover_missing_spread_slice(source_page)
                )
            result.extend(dict(item) for item in self._supplemental_cache[source_page])
        return result

    def _recover_missing_spread_slice(self, source_page: int) -> list[dict[str, Any]]:
        siblings = self.topology.logicals_for_source(source_page)
        if len(siblings) != 1:
            return []
        geometry = self.topology.geometry(siblings[0])
        if (
            geometry is None
            or geometry.split_kind != "two_page_spread"
            or geometry.segment_index not in {0, 1}
            or not self._file_path.is_file()
            or self._file_path.suffix.lower() != ".pdf"
        ):
            return []
        try:
            import cv2
            import fitz
            import numpy as np

            from docmirror.input.extraction.page_splitter import (
                analyze_spread_candidates,
                decision_from_analyses,
                split_or_passthrough,
            )
            from docmirror.ocr.repair.recognizers import rapidocr_recognize

            with fitz.open(self._file_path) as document:
                if source_page > len(document):
                    return []
                source = document[source_page - 1]
                source_width = float(source.rect.width)
                source_height = float(source.rect.height)
                pix = source.get_pixmap(matrix=fitz.Matrix(self._zoom, self._zoom), alpha=False)
            image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            image = image[:, :, :3] if pix.n >= 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            decision = decision_from_analyses(analyze_spread_candidates(image), mode="auto")
            if not decision.should_split or not decision.analyses:
                return []
            rotation = int(decision.analyses[0].rotation) % 360
            slices = split_or_passthrough(
                _rotate_image(image, rotation),
                source_width=source_width,
                source_height=source_height,
                selected_rotation=rotation,
                zoom=self._zoom,
                decision=decision,
                mode="auto",
            )
            missing_segment = 1 - int(geometry.segment_index)
            candidate = next(
                (item for item in slices if int(item.segment_index) == missing_segment),
                None,
            )
            if candidate is None:
                return []
            candidate_image = candidate.image
            height = int(candidate_image.shape[0])
            footer_words = rapidocr_recognize(
                candidate_image[int(height * 0.80) : height, :],
                source="personal_detail_supplemental_footer_ocr",
            )
            footer_text = " ".join(
                str(word.get("text") or "")
                for word in footer_words
                if float(word.get("confidence") or 0.0) >= 0.65
            )
            matches = {
                (int(match.group("page")), int(match.group("total")))
                for match in _PRINTED_FOOTER_RE.finditer(footer_text)
            }
            if len(matches) != 1:
                return []
            printed_page, printed_total = next(iter(matches))
            if not 1 <= printed_page <= printed_total:
                return []
            recovery = {
                "image": candidate_image,
                "page_width": float(candidate.width),
                "page_height": float(candidate.height),
                "source_page": source_page,
                "segment_index": missing_segment,
                "printed_page": printed_page,
                "printed_total": printed_total,
                "zoom": self._zoom,
                "source_crop_bbox": list(candidate.source_crop_bbox),
                "supplemental_page_id": f"source:{source_page}:segment:{missing_segment}:printed:{printed_page}",
            }
            self._supplemental_recoveries.append(
                {key: value for key, value in recovery.items() if key != "image"}
            )
            return [recovery]
        except Exception:
            return []

    def _recover_with_core_splitter(self, logical_page: int) -> dict[str, Any] | None:
        """Recover an existing logical slice; never invent a new logical page."""
        page = self.topology.page(logical_page)
        geometry = self.topology.geometry(logical_page)
        if page is None or geometry is None or not self._file_path.is_file() or self._file_path.suffix.lower() != ".pdf":
            return None
        if geometry.source_page <= 0:
            return None

        try:
            import cv2
            import fitz
            import numpy as np

            from docmirror.input.extraction.page_splitter import (
                analyze_spread_candidates,
                decision_from_analyses,
                split_or_passthrough,
            )

            with fitz.open(self._file_path) as document:
                if geometry.source_page > len(document):
                    return None
                source = document[geometry.source_page - 1]
                source_width = float(source.rect.width)
                source_height = float(source.rect.height)
                pix = source.get_pixmap(matrix=fitz.Matrix(self._zoom, self._zoom), alpha=False)
            image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            image = image[:, :, :3] if pix.n >= 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            analyses = analyze_spread_candidates(image)
            decision = decision_from_analyses(analyses, mode="auto")
            if not decision.analyses:
                return None
            selected_rotation = int(decision.analyses[0].rotation) % 360
            if not decision.should_split and selected_rotation != 0:
                return None
            oriented = _rotate_image(image, selected_rotation)
            slices = split_or_passthrough(
                oriented,
                source_width=source_width,
                source_height=source_height,
                selected_rotation=selected_rotation,
                zoom=self._zoom,
                decision=decision,
                mode="auto",
            )
            siblings = self.topology.logicals_for_source(geometry.source_page)
            if len(siblings) == 2:
                sibling_segments = {
                    sibling_geometry.segment_index
                    for sibling in siblings
                    if (sibling_geometry := self.topology.geometry(sibling)) is not None
                }
                if sibling_segments != {0, 1} or len(slices) != 2:
                    return None
            page_slice = _select_recovered_slice(geometry, slices)
            if page_slice is None or not _dimensions_match(
                geometry.width,
                geometry.height,
                float(page_slice.width),
                float(page_slice.height),
            ):
                return None
            return {
                "image": page_slice.image,
                "page_width": float(page_slice.width),
                "page_height": float(page_slice.height),
                "logical_page": logical_page,
                "source_page": geometry.source_page,
                "zoom": self._zoom,
                "coordinate_transform": {
                    "source_page_number": geometry.source_page,
                    "source_crop_bbox": list(page_slice.source_crop_bbox),
                    "matrix": page_slice.source_to_logical,
                    "inverse_matrix": page_slice.logical_to_source,
                    "display_width": float(page_slice.width),
                    "display_height": float(page_slice.height),
                    "decomposition": {
                        "kind": "two_page_spread" if decision.should_split else "none",
                        "segment_index": int(page_slice.segment_index),
                        "selected_rotation": selected_rotation,
                        "confidence": float(page_slice.split_confidence),
                    },
                },
                "recovered_with_core_splitter": True,
            }
        except Exception:
            return None


def _page_geometry(page: Any, logical_page: int) -> LogicalPageGeometry:
    transform = dict(getattr(page, "coordinate_transform", None) or {})
    decomposition = dict(transform.get("decomposition") or {})
    source_page = int(
        transform.get("source_page_number")
        or getattr(page, "source_page_number", 0)
        or logical_page
    )
    width = _finite(getattr(page, "width", 0) or transform.get("display_width"))
    height = _finite(getattr(page, "height", 0) or transform.get("display_height"))
    split_kind = str(decomposition.get("kind") or "")
    segment_raw = decomposition.get("segment_index")
    segment_index = int(segment_raw) if isinstance(segment_raw, int) and not isinstance(segment_raw, bool) else None
    rotation = int(decomposition.get("selected_rotation") or transform.get("content_rotation_applied") or 0) % 360
    crop = _bbox4(transform.get("source_crop_bbox"))
    issues: list[str] = []
    if source_page <= 0:
        issues.append("invalid_source_page")
    if width <= 0 or height <= 0:
        issues.append("invalid_logical_dimensions")
    if not _is_matrix3(transform.get("matrix")):
        issues.append("invalid_source_to_logical_matrix")
    if not _is_matrix3(transform.get("inverse_matrix")):
        issues.append("invalid_logical_to_source_matrix")
    if crop is None:
        issues.append("invalid_source_crop_bbox")
    return LogicalPageGeometry(
        logical_page=logical_page,
        source_page=source_page,
        width=width,
        height=height,
        split_kind=split_kind,
        segment_index=segment_index,
        selected_rotation=rotation,
        source_crop_bbox=crop,
        transform_usable=not issues,
        issues=tuple(issues),
    )


def _select_recovered_slice(geometry: LogicalPageGeometry, slices: list[Any]) -> Any | None:
    if not slices:
        return None
    if len(slices) == 1:
        only = slices[0]
        if geometry.split_kind == "two_page_spread" and geometry.segment_index not in {None, int(only.segment_index)}:
            return None
        return only
    if len(slices) != 2 or geometry.segment_index not in {0, 1}:
        return None
    return next((item for item in slices if int(item.segment_index) == geometry.segment_index), None)


def _dimensions_match(expected_width: float, expected_height: float, width: float, height: float) -> bool:
    if expected_width <= 0 or expected_height <= 0:
        return False
    width_tolerance = max(5.0, expected_width * 0.03)
    height_tolerance = max(5.0, expected_height * 0.03)
    return abs(expected_width - width) <= width_tolerance and abs(expected_height - height) <= height_tolerance


def _rotate_image(image: Any, rotation: int) -> Any:
    import cv2

    if rotation == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def _finite(value: Any) -> float:
    try:
        number = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _bbox4(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    x0, y0, x1, y1 = (_finite(item) for item in value)
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def _is_matrix3(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return False
    try:
        return all(
            isinstance(row, (list, tuple))
            and len(row) == 3
            and all(math.isfinite(float(item)) for item in row)
            for row in value
        )
    except (TypeError, ValueError):
        return False


def _overlap_ratio(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    intersection = max(0.0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0.0, min(left[3], right[3]) - max(left[1], right[1])
    )
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    return intersection / max(1.0, min(left_area, right_area))


__all__ = [
    "LogicalPageGeometry",
    "PersonalDetailLogicalPageImageResolver",
    "PersonalDetailPageTopology",
]
