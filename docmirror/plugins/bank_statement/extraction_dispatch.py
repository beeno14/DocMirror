# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Acquisition-mode routing for the Community bank-statement projector.

The sealed ``ParseResult`` already records how the document was acquired.  The
bank plugin treats that metadata as the routing authority and resolves the
route before it builds any table/style context.  This keeps native-PDF
strategies from competing with OCR strategies later in the parser registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from docmirror.models.entities.parse_result import ExtractionMethod


class BankExtractionRoute(str, Enum):
    """The two supported bank decoding pipelines."""

    DIGITAL = "digital"
    SCANNED = "scanned"


@dataclass(frozen=True)
class BankExtractionPolicy:
    """Immutable source/parser capabilities for one decoding route."""

    route: BankExtractionRoute
    allowed_parser_ids: frozenset[str]
    allow_semantic_text: bool = False
    allow_schema_guided_page_text: bool = False
    allow_physical_tables: bool = False
    allow_positioned_records: bool = False
    allow_evidence_atoms: bool = False
    allow_native_wide_tables: bool = False
    allow_ocr_implicit_tables: bool = False

    def allows_parser(self, parser_id: str) -> bool:
        return parser_id == "kv_identity" or parser_id in self.allowed_parser_ids


DIGITAL_POLICY = BankExtractionPolicy(
    route=BankExtractionRoute.DIGITAL,
    allowed_parser_ids=frozenset({"compact_merged", "grid_standard", "signed_amount"}),
    allow_semantic_text=True,
    allow_schema_guided_page_text=True,
    allow_physical_tables=True,
    allow_positioned_records=True,
    allow_evidence_atoms=True,
    allow_native_wide_tables=True,
)

SCANNED_POLICY = BankExtractionPolicy(
    route=BankExtractionRoute.SCANNED,
    allowed_parser_ids=frozenset({"grid_standard", "signed_amount", "borderless_ocr"}),
    allow_evidence_atoms=True,
    allow_ocr_implicit_tables=True,
)


_NATIVE_PAGE_MODES = frozenset({"digital", "native", "native_text", "text"})
_SCANNED_PAGE_MODES = frozenset({"image", "ocr", "scanned", "scanned_ocr"})


def _page_has_content(page: Any, *, evidence_content_pages: set[int]) -> bool:
    if list(getattr(page, "tables", None) or []):
        return True
    if any(str(getattr(item, "content", "") or "").strip() for item in getattr(page, "texts", None) or []):
        return True
    if any(
        str(getattr(item, "key", "") or "").strip() or str(getattr(item, "value", "") or "").strip()
        for item in getattr(page, "key_values", None) or []
    ):
        return True
    try:
        page_number = int(getattr(page, "page_number", 0) or 0)
    except (TypeError, ValueError):
        return False
    return page_number in evidence_content_pages


def _evidence_content_pages(parse_result: Any) -> set[int]:
    plane = getattr(parse_result, "evidence_plane", None)
    if plane is None:
        return set()
    evidence = getattr(plane, "evidence", None)
    content_page_ids: set[str] = set()
    for collection_name in ("text_atoms", "image_atoms"):
        for atom in getattr(evidence, collection_name, None) or []:
            if collection_name == "text_atoms" and not str(getattr(atom, "text", "") or "").strip():
                continue
            page_id = str(getattr(atom, "page_id", "") or "")
            if page_id:
                content_page_ids.add(page_id)

    content_pages: set[int] = set()
    indexes = getattr(evidence, "indexes", None) or {}
    for candidate in indexes.get("table_candidates", []) if isinstance(indexes, dict) else []:
        if not isinstance(candidate, dict):
            continue
        try:
            page_number = int(candidate.get("page_number") or 0)
        except (TypeError, ValueError):
            continue
        if page_number > 0:
            content_pages.add(page_number)
    for snapshot in getattr(plane, "pages", None) or []:
        if str(getattr(snapshot, "page_id", "") or "") not in content_page_ids:
            continue
        try:
            page_number = int(getattr(snapshot, "page_number", 0) or 0)
        except (TypeError, ValueError):
            continue
        if page_number > 0:
            content_pages.add(page_number)
    return content_pages


def _content_page_route(parse_result: Any) -> BankExtractionRoute | None:
    """Resolve a formerly hybrid result when one side is content-empty.

    Cached ``ParseResult`` objects can retain ``HYBRID`` metadata produced by
    older extractors. Reclassify only when every content-bearing page proves
    the same acquisition family. Unknown or genuinely mixed content remains
    fail-closed.
    """

    pages = list(getattr(parse_result, "pages", None) or [])
    if not pages:
        return None
    evidence_content_pages = _evidence_content_pages(parse_result)
    snapshots: dict[int, str] = {}
    for item in getattr(getattr(parse_result, "evidence_plane", None), "pages", None) or []:
        try:
            page_number = int(getattr(item, "page_number", 0) or 0)
        except (TypeError, ValueError):
            continue
        if page_number > 0:
            snapshots[page_number] = str(getattr(item, "content_mode", "") or "").strip().lower()

    route_flags: set[BankExtractionRoute] = set()
    saw_unknown = False
    for page in pages:
        if not _page_has_content(page, evidence_content_pages=evidence_content_pages):
            continue
        try:
            page_number = int(getattr(page, "page_number", 0) or 0)
        except (TypeError, ValueError):
            page_number = 0
        mode = str(getattr(page, "page_mode", "") or snapshots.get(page_number, "")).strip().lower()
        if mode in _NATIVE_PAGE_MODES:
            route_flags.add(BankExtractionRoute.DIGITAL)
        elif mode in _SCANNED_PAGE_MODES:
            route_flags.add(BankExtractionRoute.SCANNED)
        elif mode in {"hybrid", "mixed"}:
            route_flags.update({BankExtractionRoute.DIGITAL, BankExtractionRoute.SCANNED})
        else:
            saw_unknown = True
    if not saw_unknown and len(route_flags) == 1:
        return next(iter(route_flags))
    return None


def resolve_bank_extraction_route(parse_result: Any) -> BankExtractionRoute:
    """Resolve the route only from sealed parser metadata.

    ``IMAGE`` is OCR acquisition and therefore shares the scanned route.
    ``HYBRID`` remains unsupported when both acquisition modes contain source
    content. Empty pages are neutral, including for cached results produced
    before the extractor learned that distinction.

    ``None`` remains a small compatibility seam for text-only SDK/tests where
    no PDF ``ParseResult`` exists.  Real ``ParseResult`` objects must expose a
    valid extraction method.
    """

    if parse_result is None:
        return BankExtractionRoute.DIGITAL
    parser_info = getattr(parse_result, "parser_info", None)
    if parser_info is None:
        # Compatibility for old extension/test stubs. Canonical ParseResult
        # instances always provide ParserInfo, so production routing remains
        # metadata-driven.
        return BankExtractionRoute.DIGITAL
    raw_method = getattr(parser_info, "extraction_method", None)
    value = raw_method.value if hasattr(raw_method, "value") else str(raw_method or "").strip().lower()
    try:
        method = ExtractionMethod(value)
    except ValueError as exc:
        raise ValueError(f"unsupported bank extraction method: {value or '<missing>'}") from exc
    if method is ExtractionMethod.DIGITAL:
        return BankExtractionRoute.DIGITAL
    if method in {ExtractionMethod.OCR, ExtractionMethod.IMAGE}:
        return BankExtractionRoute.SCANNED
    if method is ExtractionMethod.HYBRID:
        content_route = _content_page_route(parse_result)
        if content_route is not None:
            return content_route
    raise ValueError("hybrid bank statements are not supported by the split Community pipeline")


__all__ = [
    "BankExtractionPolicy",
    "BankExtractionRoute",
    "DIGITAL_POLICY",
    "SCANNED_POLICY",
    "resolve_bank_extraction_route",
]
