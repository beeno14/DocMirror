# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validated source-fragment geometry for scanned personal detailed reports.

Core logical pages are immutable evidence fragments, not canonical PBOC page
identities.  The plugin validates their transforms and may register any number
of non-overlapping fragments onto one canonical page. Missing spread halves
are recovered with static image geometry only; topology construction never
runs OCR and never mutates the sealed ParseResult.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from docmirror.plugins.credit_report.page_image_resolver import LogicalPageImageResolver


@dataclass(frozen=True)
class LogicalPageGeometry:
    logical_page: int
    source_page: int
    width: float
    height: float
    split_kind: str
    segment_index: int | None
    selected_rotation: int
    split_confidence: float
    source_crop_bbox: tuple[float, float, float, float] | None
    transform_usable: bool
    issues: tuple[str, ...] = ()


class PersonalDetailPageTopology:
    """Validate arbitrary core-produced source fragments without re-splitting them."""

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
        if source <= 0 or not geometries:
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

        if len(geometries) > 2:
            crops = [geometry.source_crop_bbox for geometry in geometries]
            if any(crop is None for crop in crops):
                self._invalidate(source, "multi_fragment_source_crop_missing")
            else:
                for index, left in enumerate(crops):
                    for right in crops[index + 1 :]:
                        if left is not None and right is not None and _overlap_ratio(left, right) > 0.08:
                            self._invalidate(source, "multi_fragment_crops_overlap")
                            break
            return tuple(
                geometry.logical_page
                for geometry in sorted(
                    geometries,
                    key=lambda item: (
                        (item.source_crop_bbox or (0, 0, 0, 0))[1],
                        (item.source_crop_bbox or (0, 0, 0, 0))[0],
                        item.logical_page,
                    ),
                )
            )

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

    def ordered_fragments(self, source_page: int) -> tuple[int, ...]:
        """Return every validated fragment for a physical source surface."""
        source = int(source_page or 0)
        if source in self._invalid_sources:
            return ()
        return self._logicals_by_source.get(source, ())

    def ordered_pair(self, logical_pages: Iterable[int]) -> tuple[int, int] | None:
        requested = {int(page) for page in logical_pages if int(page) > 0}
        if len(requested) != 2:
            return None
        sources = {
            geometry.source_page for logical in requested if (geometry := self._geometry.get(logical)) is not None
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
            geometry is not None and geometry.transform_usable and geometry.source_page not in self._invalid_sources
        )

    def audit(self) -> dict[str, Any]:
        single_sources = 0
        double_sources = 0
        partial_spread_sources = 0
        fragmented_sources = 0
        potential_split_sources: list[int] = []
        for source, logicals in self._logicals_by_source.items():
            if len(logicals) > 2:
                fragmented_sources += 1
            elif len(logicals) == 2:
                double_sources += 1
            elif logicals:
                geometry = self._geometry[logicals[0]]
                if geometry.split_kind == "two_page_spread":
                    partial_spread_sources += 1
                else:
                    single_sources += 1
                if geometry.split_kind == "two_page_spread" or geometry.split_confidence >= 0.55:
                    potential_split_sources.append(source)
        return {
            "valid": not self._invalid_sources
            and not any(issue.startswith("invalid_or_duplicate_logical_page:") for issue in self._issues),
            "logical_page_count": len(self._geometry),
            "source_page_count": len(self._logicals_by_source),
            "single_page_sources": single_sources,
            "double_page_sources": double_sources,
            "partial_spread_sources": partial_spread_sources,
            "fragmented_sources": fragmented_sources,
            "potential_split_sources": sorted(potential_split_sources),
            "invalid_source_pages": sorted(self._invalid_sources),
            "issues": list(self._issues),
            "logical_pages_by_source": {
                str(source): list(logicals) for source, logicals in sorted(self._logicals_by_source.items())
            },
        }


class PersonalDetailLogicalPageImageResolver:
    """Render frozen logical pages and statically recover missing spread slices."""

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
        self._registered_static_splits: dict[int, dict[str, Any]] = {}
        self._static_split_cache: dict[int, tuple[dict[str, Any], ...]] = {}
        self._static_split_recoveries: list[dict[str, Any]] = []
        self._static_split_decisions: list[dict[str, Any]] = []

    def __call__(self, logical_page: int) -> dict[str, Any] | None:
        logical = int(logical_page or 0)
        if logical in self._registered_static_splits:
            return self._registered_static_splits[logical]
        if logical in self._cache:
            return self._cache[logical]
        rendered = self._base(logical) if self.topology.transform_usable(logical) else None
        self._cache[logical] = rendered
        return rendered

    def clear(self) -> None:
        self._base.clear()
        self._cache.clear()
        self._registered_static_splits.clear()
        self._static_split_cache.clear()

    def register_static_logical_page(self, logical_page: int, recovered: dict[str, Any]) -> None:
        """Freeze one statically split image under its final logical-page id."""

        logical = int(logical_page or 0)
        if logical <= 0:
            raise ValueError("static logical page must be positive")
        page = dict(recovered)
        page["logical_page"] = logical
        page["page"] = logical
        self._registered_static_splits[logical] = page
        self._cache.pop(logical, None)

    def page_key(self, logical_page: int) -> str:
        """Return the immutable image identity for one-shot page re-OCR."""

        logical = int(logical_page or 0)
        registered = self._registered_static_splits.get(logical)
        if registered is not None:
            return str(registered.get("page_key") or registered.get("static_page_id") or "")
        rendered = self._cache.get(logical)
        if isinstance(rendered, dict):
            transform = dict(rendered.get("coordinate_transform") or {})
            decomposition = dict(transform.get("decomposition") or {})
            source = int(transform.get("source_page_number") or rendered.get("source_page") or 0)
            crop = transform.get("source_crop_bbox")
            if source > 0 and isinstance(crop, (list, tuple)) and len(crop) == 4:
                crop_key = ":".join(f"{float(value):.4f}" for value in crop)
                return (
                    f"source:{source}:crop:{crop_key}:"
                    f"rotation:{int(decomposition.get('selected_rotation') or 0) % 360}"
                )
        geometry = self.topology.geometry(logical)
        if geometry is None:
            return ""
        crop = geometry.source_crop_bbox or (0.0, 0.0, geometry.width, geometry.height)
        crop_key = ":".join(f"{float(value):.4f}" for value in crop)
        return (
            f"source:{geometry.source_page}:crop:{crop_key}:"
            f"rotation:{int(geometry.selected_rotation) % 360}"
        )

    def audit(self) -> dict[str, Any]:
        return {
            **self.topology.audit(),
            "static_split_recoveries": list(self._static_split_recoveries),
            "static_split_decisions": list(self._static_split_decisions),
        }

    def static_split_slices(self, source_pages: Iterable[int]) -> list[dict[str, Any]]:
        """Construct statically confirmed subpages for potential-split sources."""
        result: list[dict[str, Any]] = []
        for source_page in sorted({int(page) for page in source_pages if int(page) > 0}):
            if source_page not in self._static_split_cache:
                self._static_split_cache[source_page] = tuple(self._construct_split_slices(source_page))
            result.extend(dict(item) for item in self._static_split_cache[source_page])
        return result

    def _construct_split_slices(self, source_page: int) -> list[dict[str, Any]]:
        siblings = self.topology.logicals_for_source(source_page)
        if not siblings or len(siblings) > 2:
            self._static_split_decisions.append(
                {
                    "source_page": source_page,
                    "status": "failed",
                    "reason": "invalid_source_fragment_count",
                    "observed_fragment_count": len(siblings),
                }
            )
            return []
        geometries = [geometry for logical in siblings if (geometry := self.topology.geometry(logical)) is not None]
        if not geometries or not self._file_path.is_file() or self._file_path.suffix.lower() != ".pdf":
            self._static_split_decisions.append(
                {
                    "source_page": source_page,
                    "status": "failed",
                    "reason": "source_page_not_renderable",
                }
            )
            return []
        existing_segments = {
            int(geometry.segment_index)
            for geometry in geometries
            if geometry.split_kind == "two_page_spread" and geometry.segment_index in {0, 1}
        }
        if existing_segments == {0, 1}:
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

            analysis_zoom = min(1.25, self._zoom)
            with fitz.open(self._file_path) as document:
                if source_page > len(document):
                    self._static_split_decisions.append(
                        {
                            "source_page": source_page,
                            "status": "failed",
                            "reason": "source_page_out_of_range",
                        }
                    )
                    return []
                source = document[source_page - 1]
                source_width = float(source.rect.width)
                source_height = float(source.rect.height)
                pix = source.get_pixmap(matrix=fitz.Matrix(analysis_zoom, analysis_zoom), alpha=False)
            image: Any = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            image = image[:, :, :3] if pix.n >= 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            consensus_boost = self._split_consensus_boost()
            geometry_rotation = int(geometries[0].selected_rotation) % 360
            analyses = analyze_spread_candidates(image)
            selected_analyses = tuple(
                item for item in analyses if int(item.rotation) % 360 == geometry_rotation
            )
            decision = decision_from_analyses(
                selected_analyses,
                mode="auto",
                consensus_boost=consensus_boost,
            )
            if not decision.should_split or not decision.analyses:
                self._static_split_decisions.append(
                    {
                        "source_page": source_page,
                        "status": "uncertain" if decision.confidence >= 0.65 else "no_split",
                        "confidence": float(decision.confidence),
                        "core_confidence": float(geometries[0].split_confidence),
                        "selected_rotation": geometry_rotation,
                    }
                )
                return []
            # The core transform has already selected page orientation. Static
            # topology validation must not invoke OCR to second-guess it.
            rotation = geometry_rotation
            if self._zoom > analysis_zoom:
                with fitz.open(self._file_path) as document:
                    source = document[source_page - 1]
                    pix = source.get_pixmap(matrix=fitz.Matrix(self._zoom, self._zoom), alpha=False)
                image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                image = image[:, :, :3] if pix.n >= 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            slices = split_or_passthrough(
                _rotate_image(image, rotation),
                source_width=source_width,
                source_height=source_height,
                selected_rotation=rotation,
                zoom=self._zoom,
                decision=decision,
                mode="auto",
            )
            slices_by_segment = {int(item.segment_index): item for item in slices}
            if set(slices_by_segment) != {0, 1}:
                self._static_split_decisions.append(
                    {
                        "source_page": source_page,
                        "status": "uncertain",
                        "confidence": float(decision.confidence),
                        "core_confidence": float(geometries[0].split_confidence),
                        "selected_rotation": rotation,
                        "observed_segments": sorted(slices_by_segment),
                    }
                )
                return []
            final_width = float(slices_by_segment[0].width) + float(slices_by_segment[1].width)
            final_split_ratio = float(slices_by_segment[0].width) / final_width if final_width > 0 else 0.5
            self._static_split_decisions.append(
                {
                    "source_page": source_page,
                    "status": "split",
                    "confidence": float(decision.confidence),
                    "core_confidence": float(geometries[0].split_confidence),
                    "selected_rotation": rotation,
                    "split_ratio": final_split_ratio,
                }
            )
            # A single unsplit core page represents neither half reliably, so
            # both split slices become static topology candidates. For a partial
            # core spread, only the segment absent from the sealed topology is
            # constructed.
            missing_segments = ({0, 1} - existing_segments) if existing_segments else {0, 1}
            recoveries: list[dict[str, Any]] = []
            for missing_segment in sorted(missing_segments):
                candidate = slices_by_segment[missing_segment]
                candidate_image = candidate.image
                crop_key = ":".join(f"{float(value):.4f}" for value in candidate.source_crop_bbox)
                page_key = (
                    f"source:{source_page}:crop:{crop_key}:"
                    f"rotation:{int(candidate.selected_rotation) % 360}"
                )
                recovery = {
                    "image": candidate_image,
                    "page_width": float(candidate.width),
                    "page_height": float(candidate.height),
                    "source_page": source_page,
                    "segment_index": missing_segment,
                    "zoom": self._zoom,
                    "source_crop_bbox": list(candidate.source_crop_bbox),
                    "crop_bbox_oriented": list(candidate.crop_bbox_oriented),
                    "source_to_logical": candidate.source_to_logical,
                    "logical_to_source": candidate.logical_to_source,
                    "selected_rotation": rotation,
                    "split_confidence": float(candidate.split_confidence),
                    "split_position": float(candidate.crop_bbox_oriented[2])
                    if missing_segment == 0
                    else float(candidate.crop_bbox_oriented[0]),
                    "split_ratio": final_split_ratio,
                    "split_consensus_boost": consensus_boost,
                    "subpage_basis": "static_split_validator",
                    "static_page_id": f"source:{source_page}:segment:{missing_segment}:static-split",
                    "page_key": page_key,
                    "coordinate_transform": {
                        "source_page_number": source_page,
                        "source_crop_bbox": list(candidate.source_crop_bbox),
                        "display_width": float(candidate.width),
                        "display_height": float(candidate.height),
                        "matrix": candidate.source_to_logical,
                        "inverse_matrix": candidate.logical_to_source,
                        "decomposition": {
                            "kind": "two_page_spread",
                            "segment_index": missing_segment,
                            "selected_rotation": rotation,
                            "confidence": float(candidate.split_confidence),
                        },
                    },
                }
                self._static_split_recoveries.append({key: value for key, value in recovery.items() if key != "image"})
                recoveries.append(recovery)
            return recoveries
        except Exception as exc:
            self._static_split_decisions.append(
                {
                    "source_page": source_page,
                    "status": "failed",
                    "reason": "static_validator_exception",
                    "error_type": type(exc).__name__,
                }
            )
            return []

    def _split_consensus_boost(self) -> float:
        """Reuse strong document topology evidence without consulting footers."""
        audit = self.topology.audit()
        return 0.05 if int(audit.get("double_page_sources") or 0) >= 2 else 0.0

def _page_geometry(page: Any, logical_page: int) -> LogicalPageGeometry:
    transform = dict(getattr(page, "coordinate_transform", None) or {})
    decomposition = dict(transform.get("decomposition") or {})
    source_page = int(transform.get("source_page_number") or getattr(page, "source_page_number", 0) or logical_page)
    width = _finite(getattr(page, "width", 0) or transform.get("display_width"))
    height = _finite(getattr(page, "height", 0) or transform.get("display_height"))
    split_kind = str(decomposition.get("kind") or "")
    segment_raw = decomposition.get("segment_index")
    segment_index = int(segment_raw) if isinstance(segment_raw, int) and not isinstance(segment_raw, bool) else None
    rotation = int(decomposition.get("selected_rotation") or transform.get("content_rotation_applied") or 0) % 360
    split_confidence = max(0.0, min(1.0, _finite(decomposition.get("confidence"))))
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
        split_confidence=split_confidence,
        source_crop_bbox=crop,
        transform_usable=not issues,
        issues=tuple(issues),
    )


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
            isinstance(row, (list, tuple)) and len(row) == 3 and all(math.isfinite(float(item)) for item in row)
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
