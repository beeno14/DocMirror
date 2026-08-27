# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Community v3/v4 and business-facing v5 JSON, Markdown, CSV, and audit projection."""

from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import mimetypes
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from docmirror.models.community_semantic import CommunitySemanticResult
from docmirror.output.bank_business_view import (
    BUSINESS_VIEW_VERSION,
    business_view,
    is_business_view,
    render_business_markdown,
    restore_business_records,
)
from docmirror.output.markdown_renderer import render_markdown
from docmirror.output.normalized_records import enrich_normalized_dataset, strip_source_value_pools

_SYSTEM_COLUMNS = ("record_id", "_page_start", "_page_end")
_AUDIT_COLUMNS = (
    "dataset_id",
    "record_id",
    "field_key",
    "value",
    "raw",
    "value_type",
    "unit",
    "page_start",
    "page_end",
    "bbox",
    "confidence",
    "evidence_ref",
    "csv_escape_applied",
)

_READING_COLUMN_PREFERENCES: dict[str, tuple[str, ...]] = {
    "credit_accounts": (
        "sequence",
        "account_type",
        "institution",
        "business_type",
        "open_date",
        "credit_limit",
        "loan_amount",
        "balance",
        "status",
    ),
    "credit_lines": (
        "facility_type",
        "institution",
        "total_limit",
        "used_limit",
        "status",
    ),
    "repayment_liability_records": (
        "sequence",
        "liability_date",
        "related_party_name",
        "institution",
        "business_type",
        "responsibility_amount",
        "balance",
    ),
    "inquiry_records": (
        "sequence",
        "inquiry_date",
        "institution",
        "reason",
        "inquiry_type",
    ),
}

_NON_DATASET_KEYS = frozenset(
    {
        "fields",
        "field_details",
        "field_metadata",
        "field_schema",
        "normalized_fields",
        "columns",
        "summary",
        "sections",
        "tables",
        "notes",
        "document_flow",
        "datasets",
        "data_dictionary",
        "source_content",
        "extraction_audit",
    }
)

_INTERNAL_RECORD_KEYS = frozenset(
    {
        "source",
        "source_refs",
        "source_cell_refs",
        "source_fact_ids",
        "source_anchor",
        "evidence_ids",
        "confidence",
        "review",
        "normalizer",
        "extraction_method",
        "canonical_raw",
        "record_id",
        "row_id",
    }
)

_TYPE_MAP = {
    "number": "decimal",
    "float": "decimal",
    "double": "decimal",
    "int": "integer",
    "currency": "money",
    "amount": "money",
    "percentage": "decimal",
    "phone": "string",
    "identifier": "string",
    "id_number": "string",
    "account_number": "string",
    "long_id": "string",
    "email": "string",
    "array": "text",
    "object": "text",
    "json": "text",
    "unknown": "string",
}


def _slug(value: Any, fallback: str = "item") -> str:
    text = re.sub(r"[^0-9A-Za-z_]+", "_", str(value or "").strip()).strip("_").lower()
    if text:
        return text
    digest = hashlib.sha1(str(value or fallback).encode("utf-8")).hexdigest()[:10]
    return f"{fallback}_{digest}"


def _plain(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    for key in ("normalized_value", "value", "normalized", "raw_value", "raw"):
        if value.get(key) not in (None, ""):
            return value[key]
    return value


def _raw_value(value: Any, detail: dict[str, Any] | None = None) -> Any:
    detail = detail or {}
    for candidate in (
        detail.get("raw"),
        value.get("raw_value") if isinstance(value, dict) else None,
        value.get("raw") if isinstance(value, dict) else None,
    ):
        if candidate not in (None, ""):
            return candidate
    plain = _plain(value)
    return plain if not isinstance(plain, (dict, list)) else _canonical_json(plain)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _scalar(value: Any) -> Any:
    value = _plain(value)
    if isinstance(value, (dict, list)):
        return _canonical_json(value)
    return value


def _type_of(value: Any, declared: Any = "") -> str:
    declared_text = str(declared or "").lower()
    if declared_text:
        return _TYPE_MAP.get(declared_text, declared_text)
    value = _plain(value)
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "decimal"
    if isinstance(value, (dict, list)):
        return "text"
    return "string"


def _json_value(value: Any, value_type: str) -> Any:
    if value_type in {"array", "object", "json"}:
        return _json_safe(value)
    value = _scalar(value)
    if value_type in {"money", "decimal"} and value not in (None, ""):
        return str(value)
    return value


_SOURCE_PAGE_KEY_PRIORITY = (
    "logical_page",
    "page",
    "page_id",
    "page_number",
    "source_page",
    "source_page_number",
)


def _positive_page_number(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float):
        return int(value) if value.is_integer() and value > 0 else None
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit():
            page = int(text)
            return page if page > 0 else None
        if len(text) > 1 and text[0].casefold() == "p" and text[1:].isdigit():
            page = int(text[1:])
            return page if page > 0 else None
    return None


def _source_pages(value: Any) -> list[int]:
    """Collect logical source pages without mixing coordinate systems per ref."""

    def collect(item: Any) -> list[int]:
        if isinstance(item, dict):
            normalized_items = {
                str(key).casefold(): nested for key, nested in item.items()
            }
            selected_page = next(
                (
                    page
                    for key in _SOURCE_PAGE_KEY_PRIORITY
                    if (
                        page := _positive_page_number(normalized_items.get(key))
                    )
                    is not None
                ),
                None,
            )
            nested_pages: list[int] = []
            for key in ("source", "source_refs", "source_cell_refs"):
                nested_refs = normalized_items.get(key)
                if isinstance(nested_refs, dict):
                    nested_pages.extend(collect(nested_refs))
                elif isinstance(nested_refs, (list, tuple)):
                    for nested_ref in nested_refs:
                        nested_pages.extend(collect(nested_ref))

            if selected_page is not None:
                return [selected_page, *nested_pages]
            if nested_pages:
                return nested_pages
            page_range = normalized_items.get("page_range")
            if isinstance(page_range, (list, tuple)):
                return [
                    page
                    for candidate in page_range
                    if (page := _positive_page_number(candidate)) is not None
                ]
            return []
        elif isinstance(item, (list, tuple)):
            pages: list[int] = []
            for nested_item in item:
                pages.extend(collect(nested_item))
            return pages
        return []

    return sorted(set(collect(value)))


def _page_range(value: Any, fallback: list[int] | None = None) -> list[int]:
    pages = _source_pages(value)
    return [min(pages), max(pages)] if pages else list(fallback or [])


def _source_hash(file_path: str) -> str:
    path = Path(file_path) if file_path else None
    if path is None or not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _domain(domain_view: dict[str, Any], projection: dict[str, Any]) -> str:
    document = domain_view.get("document") if isinstance(domain_view.get("document"), dict) else {}
    base = str(document.get("document_type") or document.get("domain") or "generic")
    variant = projection.get("document_variants") or {}
    properties = document.get("properties") if isinstance(document.get("properties"), dict) else {}
    field_name = str(variant.get("field") or "")
    value = str(properties.get(field_name) or "").lower()
    mapped = (variant.get("values") or {}).get(value)
    if mapped:
        return str(mapped)
    return base or "generic"


def _support_level(domain_view: dict[str, Any], domain: str) -> str:
    metadata = domain_view.get("metadata") if isinstance(domain_view.get("metadata"), dict) else {}
    route = str(metadata.get("route_type") or metadata.get("community_tier") or "")
    if domain == "generic" or route == "generic_fallback":
        return "generic"
    if route in {"enterprise_only", "mirror_only"}:
        return "unsupported"
    status = str(metadata.get("domain_status") or "").lower()
    return "ga" if status in {"ga", "ready", "pass", "core_domain"} else "beta"


def _field_label(key: str, dictionary: dict[str, Any]) -> str:
    fields = dictionary.get("fields") if isinstance(dictionary.get("fields"), dict) else {}
    descriptor = fields.get(key) if isinstance(fields.get(key), dict) else {}
    return str(descriptor.get("label") or key.replace("_", " "))


def _field_descriptor(key: str, dictionary: dict[str, Any], value: Any) -> tuple[str, str | None]:
    fields = dictionary.get("fields") if isinstance(dictionary.get("fields"), dict) else {}
    descriptor = fields.get(key) if isinstance(fields.get(key), dict) else {}
    value_type = _type_of(value, descriptor.get("format") or descriptor.get("type"))
    unit = descriptor.get("unit")
    return value_type, str(unit) if unit not in (None, "") else None


def _section_type(title: str, projection: dict[str, Any], raw_type: Any = "") -> str:
    if raw_type:
        return _slug(raw_type, "section")
    markers = projection.get("section_type_markers") or {}
    return next((str(kind) for marker, kind in markers.items() if str(marker) in title), _slug(title, "section"))


def _normalize_section(
    raw: dict[str, Any],
    index: int,
    page_count: int,
    projection: dict[str, Any],
) -> dict[str, Any]:
    title = str(raw.get("title") or raw.get("name") or f"章节 {index}")
    section_id = str(raw.get("id") or f"sec_{_slug(raw.get('type') or title, 'section')}_{index}")
    start = raw.get("source_page_start") or raw.get("page_start") or raw.get("logical_page_start")
    end = raw.get("source_page_end") or raw.get("page_end") or raw.get("logical_page_end") or start
    try:
        start_i = max(1, int(start or 1))
    except (TypeError, ValueError):
        start_i = 1
    try:
        end_i = max(start_i, int(end or start_i))
    except (TypeError, ValueError):
        end_i = start_i
    if page_count:
        start_i, end_i = min(start_i, page_count), min(end_i, page_count)
    return {
        "id": section_id,
        "title": title,
        "type": _section_type(title, projection, raw.get("type")),
        "page_range": [start_i, end_i],
        "items": [],
        "groups": [],
        "dataset_refs": [],
    }


def _record_pools(row: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(row, dict):
        return {"value": row}, {"value": row}
    normalized = row.get("normalized") if isinstance(row.get("normalized"), dict) else {}
    raw = row.get("canonical_raw") if isinstance(row.get("canonical_raw"), dict) else {}
    if not raw:
        raw = row.get("raw") if isinstance(row.get("raw"), dict) else {}
    if normalized or raw:
        keys = list(dict.fromkeys([*normalized.keys(), *raw.keys()]))
        source = row.get("source") if isinstance(row.get("source"), dict) else {}
        field_sources = source.get("field_sources") if isinstance(source.get("field_sources"), dict) else {}
        canonical_keys = []
        for key in keys:
            detail = field_sources.get(key)
            normalized_only = bool(isinstance(detail, dict) and detail.get("normalized_only") is True)
            explicitly_derived = bool(
                isinstance(detail, dict)
                and str(detail.get("source") or "").startswith("derived.")
                and str(detail.get("derivation") or "").strip()
            )
            if key in raw or not (normalized_only or explicitly_derived):
                canonical_keys.append(key)
        return (
            {str(key): normalized.get(key, raw.get(key)) for key in keys},
            {str(key): raw.get(key, normalized.get(key)) for key in canonical_keys},
        )
    public = {str(key): value for key, value in row.items() if key not in _INTERNAL_RECORD_KEYS}
    return public, public


def _json_safe(value: Any) -> Any:
    """Return a deterministic JSON-compatible representation."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        try:
            return isoformat()
        except (TypeError, ValueError):
            pass
    return str(value)


def _source_absent_normalized_fields(
    rows: list[Any],
    columns: list[dict[str, Any]],
    *,
    source_aliases: dict[str, Any],
    foreign_keys: list[dict[str, Any]],
) -> list[str]:
    """Find optional schema placeholders, using records before public expansion.

    Absence is dataset-wide: a source column on any row keeps that field on
    every row. Explicit nulls, canonical source keys, and field provenance are
    retained. An unrecognized blank source column makes absence uncertain.
    """
    if not rows or not all(isinstance(row, dict) for row in rows):
        return []

    def header_key(value: Any) -> str:
        return re.sub(r"[\s:：]+", "", unicodedata.normalize("NFKC", str(value))).casefold()

    names: dict[str, set[str]] = {}
    supported = {
        str(column["key"])
        for column in columns
        if column.get("nullable") is False or column.get("required") is True
    }
    for foreign_key in foreign_keys:
        supported.update(str(key) for key in foreign_key.get("columns") or [])
    for column in columns:
        key = str(column["key"])
        aliases = source_aliases.get(key)
        aliases = aliases if isinstance(aliases, (list, tuple)) else []
        for name in (key, column.get("label") or key, *aliases):
            normalized_name = header_key(name)
            if normalized_name:
                names.setdefault(normalized_name, set()).add(key)

    source_headers: dict[str, bool] = {}
    for row in rows:
        normalized = row.get("normalized") if isinstance(row.get("normalized"), dict) else {}
        raw = row.get("raw") if isinstance(row.get("raw"), dict) else {}
        canonical_raw = row.get("canonical_raw") if isinstance(row.get("canonical_raw"), dict) else {}
        source = row.get("source") if isinstance(row.get("source"), dict) else {}
        field_sources = source.get("field_sources") if isinstance(source.get("field_sources"), dict) else {}
        supported.update(canonical_raw)
        supported.update(field_sources)
        supported.update(key for key, value in normalized.items() if value != "")
        if not raw and not canonical_raw and not field_sources:
            # Without a source plane, an explicit empty value is not proof of
            # source absence. Only fields inserted by the schema may disappear.
            supported.update(normalized)
        for key, value in raw.items():
            if not str(key).startswith("_"):
                name = header_key(key)
                source_headers[name] = source_headers.get(name, False) or value in (None, "")

    for header, has_blank in source_headers.items():
        matched = set(names.get(header, ()))
        if not matched:
            # Conservative containment preserves stacked/bilingual/compound
            # source headings; false matches merely retain an extra column.
            for name, fields in names.items():
                if len(name) > 1 and name in header:
                    matched.update(fields)
        if not matched and has_blank:
            return []
        supported.update(matched)
    return [str(column["key"]) for column in columns if str(column["key"]) not in supported]


def _public_record(
    row: Any,
    *,
    dataset_id: str,
    row_index: int,
    columns: list[dict[str, Any]],
    fallback_page_range: list[int],
    preserve_canonical_raw: bool = False,
) -> dict[str, Any]:
    """Project one complete, stable Community API record."""
    record_id = _canonical_record_id(row, dataset_id, row_index)
    normalized, canonical_raw = _record_pools(row)
    if preserve_canonical_raw and isinstance(row, dict):
        # A public replay must not turn schema-inserted normalized nulls into
        # purported source values. The exported source pool is authoritative.
        canonical_raw = row.get("canonical_raw") or {}
    source_raw = row.get("raw") if isinstance(row, dict) and isinstance(row.get("raw"), dict) else canonical_raw
    column_types = {str(column["key"]): str(column.get("type") or "string") for column in columns}

    normalized_public: dict[str, Any] = {}
    for key in dict.fromkeys([*column_types.keys(), *normalized.keys()]):
        value = normalized.get(key)
        normalized_public[key] = _json_safe(_json_value(value, column_types.get(key, _type_of(value))))

    canonical_raw_public = {str(key): _json_safe(_scalar(value)) for key, value in canonical_raw.items()}
    raw_public = {str(key): _json_safe(_scalar(value)) for key, value in source_raw.items()}

    row_mapping = row if isinstance(row, dict) else {}
    source_value = row_mapping.get("source") if isinstance(row_mapping.get("source"), dict) else {}
    source = {str(key): _json_safe(value) for key, value in source_value.items() if value not in (None, "", [])}
    for key in ("source_refs", "source_cell_refs", "source_fact_ids", "evidence_ids"):
        if key not in source and row_mapping.get(key) not in (None, "", []):
            source[key] = _json_safe(row_mapping[key])
    if "confidence" not in source and row_mapping.get("confidence") not in (None, ""):
        source["confidence"] = _json_safe(row_mapping["confidence"])
    page_range = _page_range(source_value, _page_range(row_mapping, fallback_page_range))
    if page_range:
        source["page_range"] = page_range

    public: dict[str, Any] = {
        "record_id": record_id,
        "normalized": normalized_public,
        "canonical_raw": canonical_raw_public,
        "raw": raw_public,
        "source": source,
    }
    if row_mapping.get("confidence") not in (None, ""):
        public["confidence"] = _json_safe(row_mapping["confidence"])
    if row_mapping.get("review") not in (None, "", {}):
        public["review"] = _json_safe(row_mapping["review"])
    return public


def _dataset_columns(rows: list[Any], dictionary: dict[str, Any], dataset_id: str) -> list[dict[str, Any]]:
    datasets = dictionary.get("datasets") if isinstance(dictionary.get("datasets"), dict) else {}
    ds_schema = datasets.get(dataset_id) if isinstance(datasets.get(dataset_id), dict) else {}
    declared = ds_schema.get("columns") if isinstance(ds_schema.get("columns"), dict) else {}
    record_columns = dictionary.get("record_columns") if dataset_id == "records" else {}
    if isinstance(record_columns, dict):
        declared = {**record_columns, **declared}
    values: dict[str, list[Any]] = {}
    raw_available: set[str] = set()
    evidence_available: set[str] = set()
    present_count: dict[str, int] = {}
    for row in rows:
        normalized, raw = _record_pools(row)
        for key in normalized:
            value = normalized.get(key, raw.get(key))
            values.setdefault(key, []).append(value)
            if value not in (None, ""):
                present_count[key] = present_count.get(key, 0) + 1
            if key in raw and raw.get(key) not in (None, ""):
                raw_available.add(key)
            if _has_evidence(value) or _has_evidence(row):
                evidence_available.add(key)
    columns: list[dict[str, Any]] = []
    for key in sorted(set(declared) | set(values)):
        info = declared.get(key) if isinstance(declared.get(key), dict) else {}
        sample = next((value for value in values.get(key, []) if value not in (None, "")), "")
        col_type = _type_of(sample, info.get("format") or info.get("type"))
        column: dict[str, Any] = {
            "key": str(key),
            "label": str(info.get("label") or str(key).replace("_", " ")),
            "type": col_type,
            "nullable": present_count.get(key, 0) < len(rows),
            "raw_available": key in raw_available,
            "evidence_available": key in evidence_available,
        }
        if info.get("unit") not in (None, ""):
            column["unit"] = str(info["unit"])
        if str(info.get("format") or info.get("type") or "").lower() == "long_id":
            column["format"] = "long_id"
        if str(info.get("display_format") or info.get("type") or "").lower() == "percentage":
            column["display_format"] = "percentage"
        for metadata_key in (
            "definition",
            "sensitive",
            "display",
            "display_format",
            "aggregation",
            "logical_type",
            "json_type",
            "enum_ref",
        ):
            if info.get(metadata_key) not in (None, ""):
                column[metadata_key] = _json_safe(info[metadata_key])
        columns.append(column)
    return columns


def _has_evidence(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    return any(
        value.get(key) not in (None, "", [])
        for key in (
            "source_page",
            "source_page_number",
            "page",
            "bbox",
            "source_refs",
            "source_cell_refs",
            "source_fact_ids",
            "evidence_ids",
            "evidence_ref",
        )
    )


def _dataset_section_id(
    data: dict[str, Any],
    key: str,
    sections: list[dict[str, Any]],
    projection: dict[str, Any],
) -> str:
    path = f"/data/{key}"
    for table in data.get("tables") or []:
        if not isinstance(table, dict):
            continue
        ref = table.get("data_ref")
        ref_path = ref.get("path") if isinstance(ref, dict) else ref
        if str(ref_path or "") == path and table.get("section_id"):
            return str(table["section_id"])
    if sections:
        preferred = tuple(str(marker) for marker in (projection.get("section_markers") or {}).get(key, ()))
        for marker in preferred:
            for section in sections:
                if marker in str(section.get("type") or ""):
                    return str(section["id"])
        source_pages = (
            sorted(
                {
                    page
                    for row in (data.get(key) or [])
                    if isinstance(row, dict)
                    for page in _page_range(row, [])
                }
            )
            if key.startswith("enterprise_public_")
            else []
        )
        if source_pages:
            anchor_page = source_pages[0]
            preceding_sections: list[tuple[int, int, dict[str, Any]]] = []
            for index, section in enumerate(sections):
                page_range = section.get("page_range") or []
                try:
                    start_page = int(page_range[0])
                except (IndexError, TypeError, ValueError):
                    continue
                if start_page <= anchor_page:
                    preceding_sections.append((start_page, index, section))
            if preceding_sections:
                return str(max(preceding_sections, key=lambda item: (item[0], item[1]))[2]["id"])
        return str(sections[0]["id"])
    return ""


def _physical_marker_row_count(result: Any, markers: set[str]) -> int:
    """Count source rows whose first cell matches a plugin-declared marker."""
    count = 0
    for page in getattr(result, "pages", None) or []:
        for table in getattr(page, "tables", None) or []:
            header = list(getattr(table, "headers", None) or [])
            if header and re.sub(r"\s+", "", str(header[0] or "")) in markers:
                count += 1
            for row in getattr(table, "rows", None) or []:
                cells = list(getattr(row, "cells", None) or [])
                first = re.sub(r"\s+", "", str(getattr(cells[0], "text", "") or "")) if cells else ""
                if first in markers:
                    count += 1
    return count


def _dataset_completeness(
    result: Any,
    key: str,
    rows: list[Any],
    projection: dict[str, Any],
    data: dict[str, Any],
) -> dict[str, Any]:
    """Resolve an independent expected count where the physical contract permits it."""
    emitted = len(rows)
    summary = data.get("credit_summary") if isinstance(data.get("credit_summary"), dict) else {}
    independent_count_keys = {
        "credit_accounts": ("reported_account_count", "source_account_count"),
        "credit_lines": ("reported_credit_line_count",),
    }
    count_keys = independent_count_keys.get(key, ())
    if key == "credit_accounts" and summary.get("account_population_comparable") is False:
        count_keys = ()
    expected = next(
        (
            int(summary[count_key])
            for count_key in count_keys
            if isinstance(summary.get(count_key), int)
            and not isinstance(summary.get(count_key), bool)
            and int(summary[count_key]) >= 0
        ),
        None,
    )
    if expected is not None:
        return {
            "expected_row_count": expected,
            "emitted_row_count": emitted,
            "omitted_row_count": max(0, expected - emitted),
            "verified": expected == emitted,
            "basis": "source_report_summary",
        }
    if key == "inquiry_records":
        sequences: dict[str, set[int]] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            normalized, raw = _record_pools(row)
            inquiry_type = str(normalized.get("inquiry_type", raw.get("inquiry_type", "")) or "unknown")
            try:
                sequence = int(normalized.get("sequence", raw.get("sequence", 0)) or 0)
            except (TypeError, ValueError):
                sequence = 0
            if sequence > 0:
                sequences.setdefault(inquiry_type, set()).add(sequence)
        if sequences:
            expected = sum(max(values) for values in sequences.values())
            return {
                "expected_row_count": expected,
                "emitted_row_count": emitted,
                "omitted_row_count": max(0, expected - emitted),
                "verified": expected == emitted
                and all(values == set(range(1, max(values) + 1)) for values in sequences.values()),
                "basis": "source_sequence_ledger",
            }
    policy = (projection.get("completeness") or {}).get(key) or {}
    if policy.get("basis") == "domain_fact_count":
        configured_candidates = policy.get("count_candidates") or []
        count_candidates = [candidate for candidate in configured_candidates if isinstance(candidate, dict)]
        if not count_candidates:
            count_candidates = [
                {
                    "key": policy.get("count_key"),
                    "public_basis": policy.get("public_basis"),
                }
            ]
        selected_count: tuple[int, str] | None = None
        for candidate in count_candidates:
            count_key = str(candidate.get("key") or "")
            count_value = data.get(count_key)
            try:
                minimum = max(0, int(candidate.get("minimum", 0) or 0))
            except (TypeError, ValueError):
                minimum = 0
            if isinstance(count_value, int) and not isinstance(count_value, bool) and count_value >= minimum:
                selected_count = (
                    int(count_value),
                    str(candidate.get("public_basis") or policy.get("public_basis") or "domain_fact_count"),
                )
                break
        if selected_count is not None:
            expected, public_basis = selected_count
            verified = expected == emitted
            status_key = str(policy.get("verification_status_key") or "")
            if status_key and data.get(status_key) not in (None, ""):
                allowed_statuses = {str(value) for value in policy.get("verified_statuses") or ["success"]}
                verified = verified and str(data.get(status_key) or "") in allowed_statuses
            return {
                "expected_row_count": expected,
                "emitted_row_count": emitted,
                "omitted_row_count": max(0, expected - emitted),
                "verified": verified,
                "basis": public_basis,
            }
    if policy.get("basis") == "physical_marker_rows":
        markers = {re.sub(r"\s+", "", str(value)) for value in policy.get("first_column_values") or []}
        expected = _physical_marker_row_count(result, markers)
        if expected > 0:
            return {
                "expected_row_count": expected,
                "emitted_row_count": emitted,
                "omitted_row_count": max(0, expected - emitted),
                "verified": expected == emitted,
                "basis": str(policy.get("public_basis") or "physical_marker_rows"),
            }
    return {
        "expected_row_count": emitted,
        "emitted_row_count": emitted,
        "omitted_row_count": 0,
        "verified": False,
        "basis": "emitted_records_only",
    }


def _warning_code(raw: str) -> str:
    base = raw.split(":", 1)[0]
    code = re.sub(r"[^0-9A-Za-z]+", "_", base).strip("_").upper()
    return code or "PARTIAL_PARSE"


def _effective_omitted_normalized_fields(dataset: dict[str, Any]) -> set[str]:
    """Honor saved omission declarations without hiding subsequently added data."""
    omitted = set(dataset.get("omitted_normalized_fields") or [])
    if not omitted:
        return omitted
    columns = list(dataset.get("columns") or [])
    omitted.intersection_update(str(column["key"]) for column in columns)
    omitted.difference_update(
        str(column["key"])
        for column in columns
        if column.get("nullable") is False or column.get("required") is True
    )
    for foreign_key in dataset.get("foreign_keys") or []:
        omitted.difference_update(foreign_key.get("columns") or [])
    for row in dataset.get("rows") or []:
        if not isinstance(row, dict):
            continue
        source = row.get("source") if isinstance(row.get("source"), dict) else {}
        field_sources = source.get("field_sources") if isinstance(source.get("field_sources"), dict) else {}
        omitted.difference_update(field_sources)
        for pool_name in ("normalized", "canonical_raw"):
            pool = row.get(pool_name) if isinstance(row.get(pool_name), dict) else {}
            omitted.difference_update(key for key, value in pool.items() if value not in (None, ""))
    return omitted


def _reading_column_keys(dataset: dict[str, Any]) -> list[str]:
    available = [str(column.get("key") or "") for column in dataset.get("columns") or [] if column.get("key")]
    preferred = tuple(dataset.get("reading_columns") or ()) or _READING_COLUMN_PREFERENCES.get(
        str(dataset.get("name") or ""), ()
    )
    selected = [key for key in preferred if key in available]
    omitted = _effective_omitted_normalized_fields(dataset)
    return [key for key in (selected or available) if key not in omitted]


def _build_public_reading_model(
    document: dict[str, Any],
    sections: list[dict[str, Any]],
    datasets: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a public, replayable reading plan over Community data."""
    datasets_by_id = {str(dataset.get("id") or ""): dataset for dataset in datasets}
    tables: list[dict[str, Any]] = []
    flow: list[dict[str, Any]] = [{"order": 1, "kind": "document", "ref_id": str(document.get("id") or "")}]
    attached: set[str] = set()

    def append(kind: str, ref_id: str) -> None:
        flow.append({"order": len(flow) + 1, "kind": kind, "ref_id": ref_id})

    for section in sections:
        section_id = str(section.get("id") or "")
        append("section", section_id)
        for dataset_id in section.get("dataset_refs") or []:
            dataset_id = str(dataset_id)
            dataset = datasets_by_id.get(dataset_id)
            if dataset is None or dataset_id in attached:
                continue
            tables.append(
                {
                    "id": f"reading:{dataset_id}",
                    "dataset_id": dataset_id,
                    "section_id": section_id,
                    "title": str(dataset.get("label") or dataset.get("name") or dataset_id),
                    "column_keys": _reading_column_keys(dataset),
                    "row_count": int(dataset.get("row_count") or 0),
                }
            )
            append("dataset", dataset_id)
            attached.add(dataset_id)

    for dataset in datasets:
        dataset_id = str(dataset.get("id") or "")
        if not dataset_id or dataset_id in attached:
            continue
        tables.append(
            {
                "id": f"reading:{dataset_id}",
                "dataset_id": dataset_id,
                "section_id": str(dataset.get("section_id") or ""),
                "title": str(dataset.get("label") or dataset.get("name") or dataset_id),
                "column_keys": _reading_column_keys(dataset),
                "row_count": int(dataset.get("row_count") or 0),
            }
        )
        append("dataset", dataset_id)

    return {
        "version": "1.0",
        "profile": "community",
        "document_flow": flow,
        "tables": tables,
    }


def _markdown_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\r", " ").replace("\n", "<br>")


def _masked_display(value: Any) -> str:
    text = str(value or "")
    if len(text) <= 4:
        return "*" * len(text)
    if len(text) <= 8:
        return f"{text[:1]}{'*' * (len(text) - 3)}{text[-2:]}"
    return f"{text[:4]}{'*' * max(4, len(text) - 8)}{text[-4:]}"


def _markdown_display(
    value: Any,
    *,
    key: str,
    descriptor: dict[str, Any],
    dictionary: dict[str, Any],
    privacy_mode: str = "masked",
) -> str:
    if privacy_mode != "full" and (descriptor.get("sensitive") is True or descriptor.get("display") == "masked"):
        return _markdown_text(_masked_display(value))
    if descriptor.get("display_format") == "percentage" or descriptor.get("type") == "percentage":
        try:
            return _markdown_text(f"{float(value) * 100:.2f}%")
        except (TypeError, ValueError):
            pass
    enums = dictionary.get("enums") if isinstance(dictionary.get("enums"), dict) else {}
    enum_values = enums.get(key) if isinstance(enums.get(key), dict) else {}
    if not enum_values and key.endswith("_status"):
        enum_values = enums.get("reporting_status") if isinstance(enums.get("reporting_status"), dict) else {}
    if isinstance(value, (str, int, float, bool)):
        enum_value = enum_values.get(value)
        if enum_value is None and isinstance(value, bool):
            enum_value = enum_values.get(str(value).lower())
        if enum_value is not None:
            return _markdown_text(enum_value)
    return _markdown_text(value)


def render_community_reading_markdown(payload: dict[str, Any]) -> str:
    """Transcribe a public Community semantic result without accessing ParseResult."""
    semantic_domain = payload.get("domain") if isinstance(payload.get("domain"), dict) else {}
    dictionary = (
        semantic_domain.get("data_dictionary") if isinstance(semantic_domain.get("data_dictionary"), dict) else {}
    )
    extensions = semantic_domain.get("extensions") if isinstance(semantic_domain.get("extensions"), dict) else {}
    presentation = extensions.get("enhanced_markdown") if isinstance(extensions.get("enhanced_markdown"), dict) else {}
    privacy_mode = str(presentation.get("privacy_mode") or (payload.get("reading") or {}).get("privacy_mode") or "masked")
    if (payload.get("schema") or {}).get("name") == "docmirror.community.semantic":
        payload = _community_view_from_semantic(payload)
    if is_business_view(payload):
        return render_business_markdown(payload)
    document = payload.get("document") or {}
    if document.get("type") == "personal_credit_report_detailed":
        privacy_mode = "full"
    sections = {str(section.get("id") or ""): section for section in payload.get("sections") or []}
    datasets = {str(dataset.get("id") or ""): dataset for dataset in payload.get("datasets") or []}
    datasets_by_name = {
        str(dataset.get("name") or ""): dataset
        for dataset in datasets.values()
        if dataset.get("name")
    }
    reading = payload.get("reading") or {}
    tables = {str(table.get("dataset_id") or ""): table for table in reading.get("tables") or []}
    parts = [
        '<!-- docmirror:markdown-profile version="1.0" -->',
        '<!-- docmirror:reading-profile version="2.0" mode="enhanced" source="community-semantic" -->',
        f"# {_markdown_text(document.get('title') or document.get('type') or '文档')}",
    ]
    dictionary_enums = dictionary.get("enums") if isinstance(dictionary.get("enums"), dict) else {}
    document_type_labels = (
        dictionary_enums.get("document_type")
        if isinstance(dictionary_enums.get("document_type"), dict)
        else {}
    )
    document_type = document.get("type")
    metadata = [
        ("文档类型", document_type_labels.get(document_type, document_type)),
        ("页数", document.get("page_count")),
    ]
    if presentation.get("show_top_document_metadata", True):
        for label, value in metadata:
            if value not in (None, ""):
                parts.append(f"**{label}:** {_markdown_text(value)}")

    dictionary_fields = dictionary.get("fields") if isinstance(dictionary.get("fields"), dict) else {}

    def document_value(path: str) -> Any:
        value: Any = document
        for part in path.split("."):
            if not isinstance(value, dict):
                return None
            value = value.get(part)
        return value

    def item_line(item: dict[str, Any], *, label: str = "") -> str:
        key = str(item.get("key") or "")
        line = (
            f"**{_markdown_text(label or item.get('label') or key)}:** "
            f"{_markdown_display(item.get('value'), key=key, descriptor=item, dictionary=dictionary, privacy_mode=privacy_mode)}"
        )
        for value in item.get("additional_values") or []:
            line += " · " + _markdown_display(
                value, key=key, descriptor=item, dictionary=dictionary, privacy_mode=privacy_mode
            )
        return line

    def document_item(spec: dict[str, Any]) -> dict[str, Any] | None:
        path = str(spec.get("path") or "")
        key = str(spec.get("key") or path.replace(".", "_"))
        value = document_value(path)
        if not path or value in (None, "", [], {}):
            return None
        descriptor = dictionary_fields.get(key) if isinstance(dictionary_fields.get(key), dict) else {}
        return {
            "key": key,
            "label": str(spec.get("label") or descriptor.get("label") or key.replace("_", " ")),
            "value": value,
            **{
                metadata_key: descriptor[metadata_key]
                for metadata_key in ("sensitive", "display")
                if descriptor.get(metadata_key) not in (None, "")
            },
        }

    def section_pools(section: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        items = {
            str(item.get("key") or ""): item
            for item in section.get("items") or []
            if isinstance(item, dict) and item.get("key")
        }
        groups = {
            str(group.get("key") or ""): group
            for group in section.get("groups") or []
            if isinstance(group, dict) and group.get("key")
        }
        return items, groups

    def configured_item(
        reference: Any,
        items: dict[str, dict[str, Any]],
    ) -> dict[str, Any] | None:
        spec = reference if isinstance(reference, dict) else {"key": str(reference)}
        key = str(spec.get("key") or "")
        item = items.get(key)
        if item is None and spec.get("fallback"):
            item = items.get(str(spec["fallback"]))
        return item

    def section_has_renderable_content(
        section: dict[str, Any],
        layout: dict[str, Any] | None,
    ) -> bool:
        if any(
            not (
                isinstance(dataset_layouts.get(str(datasets.get(str(ref_id), {}).get("name") or "")), dict)
                and dataset_layouts[str(datasets.get(str(ref_id), {}).get("name") or "")].get("hidden", False)
            )
            for ref_id in section.get("dataset_refs") or []
            if str(ref_id) in datasets
        ):
            return True
        if not isinstance(layout, dict):
            return bool(section.get("items") or section.get("groups"))
        items, groups = section_pools(section)
        hidden_fields = {str(key) for key in layout.get("hidden_fields") or []}
        hidden_groups = {str(key) for key in layout.get("hidden_groups") or []}
        if not layout.get("omit_unlisted", False):
            if any(key not in hidden_fields for key in items):
                return True
            if any(key not in hidden_groups and group.get("items") for key, group in groups.items()):
                return True
        if any(
            document_item(spec) is not None
            for group_spec in layout.get("groups") or []
            if isinstance(group_spec, dict)
            for spec in group_spec.get("document_fields") or []
            if isinstance(spec, dict)
        ):
            return True
        for group_spec in layout.get("groups") or []:
            if not isinstance(group_spec, dict):
                continue
            if any(configured_item(reference, items) is not None for reference in group_spec.get("fields") or []):
                return True
            if any(
                str(group_key) in groups
                and str(group_key) not in hidden_groups
                and groups[str(group_key)].get("items")
                for group_key in group_spec.get("nested_groups") or []
            ):
                return True
        return False

    rendered_sections: set[str] = set()
    section_layouts = (
        presentation.get("section_layouts") if isinstance(presentation.get("section_layouts"), dict) else {}
    )
    suppress_empty_sections = bool(presentation.get("suppress_empty_sections", False))
    dataset_layouts = (
        presentation.get("dataset_layouts") if isinstance(presentation.get("dataset_layouts"), dict) else {}
    )
    deferred_appendix_datasets: list[
        tuple[dict[str, Any], dict[str, Any], dict[str, Any]]
    ] = []
    for entry in reading.get("document_flow") or []:
        kind = str(entry.get("kind") or "")
        ref_id = str(entry.get("ref_id") or "")
        if kind == "section":
            section = sections.get(ref_id)
            if section is None:
                continue
            rendered_sections.add(ref_id)
            layout = section_layouts.get(str(section.get("type") or ""))
            if suppress_empty_sections and not section_has_renderable_content(
                section, layout if isinstance(layout, dict) else None
            ):
                continue
            parts.append(f"## {_markdown_text(section.get('title') or section.get('type') or ref_id)}")
            if isinstance(layout, dict):
                items, groups = section_pools(section)
                rendered_item_keys: set[str] = {
                    str(key) for key in layout.get("hidden_fields") or []
                }
                rendered_group_keys: set[str] = {
                    str(key) for key in layout.get("hidden_groups") or []
                }
                for group_spec in layout.get("groups") or []:
                    if not isinstance(group_spec, dict):
                        continue
                    group_parts: list[str] = []
                    for reference in group_spec.get("fields") or []:
                        item = configured_item(reference, items)
                        if item is None:
                            continue
                        rendered_item_keys.add(str(item.get("key") or ""))
                        group_parts.append(item_line(item))
                    for spec in group_spec.get("document_fields") or []:
                        item = document_item(spec) if isinstance(spec, dict) else None
                        if item is not None:
                            group_parts.append(item_line(item))
                    for group_key in group_spec.get("nested_groups") or []:
                        group = groups.get(str(group_key))
                        if group is None or not group.get("items"):
                            continue
                        rendered_group_keys.add(str(group_key))
                        group_parts.append(f"#### {_markdown_text(group.get('label') or group.get('key'))}")
                        group_parts.extend(
                            item_line(item) for item in group.get("items") or [] if isinstance(item, dict)
                        )
                    if group_parts:
                        if not group_spec.get("hide_title", False):
                            parts.append(f"### {_markdown_text(group_spec.get('title') or '概要')}")
                        parts.extend(group_parts)
                if not layout.get("omit_unlisted", False):
                    parts.extend(item_line(item) for key, item in items.items() if key not in rendered_item_keys)
                    for key, group in groups.items():
                        if key in rendered_group_keys:
                            continue
                        parts.append(f"### {_markdown_text(group.get('label') or group.get('key'))}")
                        parts.extend(item_line(item) for item in group.get("items") or [] if isinstance(item, dict))
            else:
                for item in section.get("items") or []:
                    parts.append(item_line(item))
                for group in section.get("groups") or []:
                    parts.append(f"### {_markdown_text(group.get('label') or group.get('key'))}")
                    parts.extend(item_line(item) for item in group.get("items") or [] if isinstance(item, dict))
            continue
        if kind != "dataset":
            continue
        dataset = datasets.get(ref_id)
        table = tables.get(ref_id)
        if dataset is None or table is None:
            continue
        dataset_layout = dataset_layouts.get(str(dataset.get("name") or ""))
        if not isinstance(dataset_layout, dict):
            dataset_layout = {}
        if dataset_layout.get("hidden", False):
            continue
        if dataset_layout.get("placement") == "appendix":
            deferred_appendix_datasets.append((dataset, table, dataset_layout))
            continue
        if not dataset_layout.get("hide_title", False):
            parts.append(f"### {_markdown_text(table.get('title') or dataset.get('label') or ref_id)}")
        column_by_key = {
            str(column.get("key") or ""): column for column in dataset.get("columns") or [] if column.get("key")
        }
        def row_value(row: dict[str, Any], key: str) -> Any:
            normalized = row.get("normalized") if isinstance(row.get("normalized"), dict) else {}
            canonical_raw = row.get("canonical_raw") if isinstance(row.get("canonical_raw"), dict) else {}
            return normalized.get(key, canonical_raw.get(key))

        dataset_rows = [row for row in (dataset.get("rows") or []) if isinstance(row, dict)]
        configured_keys = dataset_layout.get("columns") or table.get("column_keys") or []
        keys = [str(key) for key in configured_keys if str(key) in column_by_key]
        omitted = _effective_omitted_normalized_fields(dataset)
        keys = [key for key in keys if key not in omitted]
        if dataset_layout.get(
            "suppress_empty_columns",
            presentation.get("suppress_empty_columns", False),
        ):
            keys = [
                key
                for key in keys
                if any(row_value(row, key) not in (None, "", [], {}) for row in dataset_rows)
            ]
        if not keys:
            parts.append("_暂无可展示字段。_")
            continue

        def display_value(row: dict[str, Any], key: str) -> str:
            return _markdown_display(
                row_value(row, key),
                key=key,
                descriptor=column_by_key[key],
                dictionary=dictionary,
                privacy_mode=privacy_mode,
            )

        def render_rows(rows: list[dict[str, Any]], row_keys: list[str] | None = None) -> str:
            active_keys = row_keys or keys
            active_labels = [
                _markdown_text(column_by_key[key].get("label") or key)
                for key in active_keys
            ]
            lines = [
                "| " + " | ".join(active_labels) + " |",
                "| " + " | ".join("---" for _ in active_keys) + " |",
            ]
            for row in rows:
                values = [display_value(row, key) for key in active_keys]
                lines.append("| " + " | ".join(values) + " |")
            return "\n".join(lines)

        if dataset_layout.get("mode") == "partitioned_tables":
            partition_by = str(dataset_layout.get("partition_by") or "")
            partition_specs = [
                spec
                for spec in dataset_layout.get("partitions") or []
                if isinstance(spec, dict) and spec.get("value") not in (None, "")
            ]
            if partition_by not in column_by_key or not partition_specs:
                parts.append(render_rows(dataset_rows))
                continue
            grouped: dict[str, list[dict[str, Any]]] = {}
            for row in dataset_rows:
                partition_value = str(row_value(row, partition_by) or "unknown")
                grouped.setdefault(partition_value, []).append(row)

            def prepend_partition(
                spec: dict[str, Any],
                primary_rows: list[dict[str, Any]],
            ) -> None:
                config = spec.get("prepend_partition")
                if not isinstance(config, dict):
                    return
                companion = datasets_by_name.get(str(config.get("dataset") or ""))
                join_on = str(config.get("join_on") or "")
                if companion is None or not join_on:
                    return
                join_values = {
                    str(value)
                    for row in primary_rows
                    if (value := row_value(row, join_on)) not in (None, "")
                }
                if not join_values:
                    return

                companion_columns = {
                    str(column.get("key") or ""): column
                    for column in companion.get("columns") or []
                    if column.get("key")
                }

                def companion_value(row: dict[str, Any], key: str) -> Any:
                    normalized = (
                        row.get("normalized")
                        if isinstance(row.get("normalized"), dict)
                        else {}
                    )
                    canonical_raw = (
                        row.get("canonical_raw")
                        if isinstance(row.get("canonical_raw"), dict)
                        else {}
                    )
                    return normalized.get(key, canonical_raw.get(key))

                companion_rows = [
                    row
                    for row in companion.get("rows") or []
                    if isinstance(row, dict)
                    and str(companion_value(row, join_on)) in join_values
                ]
                if not companion_rows:
                    return
                companion_table = tables.get(str(companion.get("id") or "")) or {}
                configured_columns = (
                    config.get("columns")
                    or companion_table.get("column_keys")
                    or []
                )
                companion_keys = [
                    str(key)
                    for key in configured_columns
                    if str(key) in companion_columns
                ]
                if not companion_keys:
                    return
                parts.append(
                    f"#### {_markdown_text(config.get('title') or companion_table.get('title') or companion.get('label'))}"
                )
                parts.append(
                    "\n".join(
                        [
                            "| "
                            + " | ".join(
                                _markdown_text(
                                    companion_columns[key].get("label") or key
                                )
                                for key in companion_keys
                            )
                            + " |",
                            "| " + " | ".join("---" for _ in companion_keys) + " |",
                            *[
                                "| "
                                + " | ".join(
                                    _markdown_display(
                                        companion_value(row, key),
                                        key=key,
                                        descriptor=companion_columns[key],
                                        dictionary=dictionary,
                                        privacy_mode=privacy_mode,
                                    )
                                    for key in companion_keys
                                )
                                + " |"
                                for row in companion_rows
                            ],
                        ]
                    )
                )

            rendered_values: set[str] = set()
            for spec in partition_specs:
                partition_value = str(spec["value"])
                rows = grouped.get(partition_value)
                if not rows or partition_value in rendered_values:
                    continue
                partition_keys = [
                    str(key)
                    for key in spec.get("columns") or []
                    if str(key) in column_by_key
                ]
                prepend_partition(spec, rows)
                parts.append(
                    f"#### {_markdown_text(spec.get('title') or partition_value)}"
                )
                parts.append(render_rows(rows, partition_keys or keys))
                rendered_values.add(partition_value)
            for partition_value in sorted(set(grouped) - rendered_values):
                shown_partition = _markdown_display(
                    partition_value,
                    key=partition_by,
                    descriptor=column_by_key[partition_by],
                    dictionary=dictionary,
                    privacy_mode=privacy_mode,
                )
                parts.append(f"#### {_markdown_text(shown_partition)}")
                parts.append(render_rows(grouped[partition_value]))
        elif dataset_layout.get("mode") == "record_cards":
            configured_title_separator = dataset_layout.get("title_separator")
            title_separator = (
                " · "
                if configured_title_separator is None
                else str(configured_title_separator)
            )
            title_fields = [
                str(key)
                for key in dataset_layout.get("title_fields") or []
                if str(key) in column_by_key
            ]
            for index, row in enumerate(dataset_rows, start=1):
                title_values = [
                    display_value(row, key)
                    for key in title_fields
                    if row_value(row, key) not in (None, "", [], {})
                ]
                if not dataset_layout.get("hide_record_titles", False):
                    parts.append(
                        f"#### {_markdown_text(title_separator.join(title_values) or f'记录 {index}')}"
                    )
                for key in keys:
                    value = row_value(row, key)
                    if value in (None, "", [], {}):
                        continue
                    parts.append(
                        f"**{_markdown_text(column_by_key[key].get('label') or key)}:** "
                        f"{display_value(row, key)}"
                    )
        elif dataset_layout.get("mode") == "grouped_table":
            group_by = str(dataset_layout.get("group_by") or "")
            metadata_keys = [
                str(key)
                for key in dataset_layout.get("group_metadata") or []
                if str(key) in column_by_key
            ]
            grouped: dict[str, list[dict[str, Any]]] = {}
            for row in dataset_rows:
                group_value = str(row_value(row, group_by) or "未关联账户")
                grouped.setdefault(group_value, []).append(row)
            for group_value, rows in grouped.items():
                descriptor = column_by_key.get(group_by, {})
                shown_group = _markdown_display(
                    group_value,
                    key=group_by,
                    descriptor=descriptor,
                    dictionary=dictionary,
                    privacy_mode=privacy_mode,
                )
                group_title_prefix = str(
                    dataset_layout.get("group_title_prefix", "账户 ")
                )
                parts.append(f"#### {group_title_prefix}{shown_group}")
                first = rows[0]
                for key in metadata_keys:
                    value = row_value(first, key)
                    if value in (None, "", [], {}):
                        continue
                    parts.append(
                        f"**{_markdown_text(column_by_key[key].get('label') or key)}:** "
                        f"{display_value(first, key)}"
                    )
                column_groups = [
                    group
                    for group in dataset_layout.get("column_groups") or []
                    if isinstance(group, dict)
                ]
                if column_groups:
                    for column_group in column_groups:
                        group_keys = [
                            str(key)
                            for key in column_group.get("columns") or []
                            if str(key) in column_by_key
                        ]
                        if not group_keys:
                            continue
                        parts.append(
                            f"##### {_markdown_text(column_group.get('title') or '明细')}"
                        )
                        parts.append(render_rows(rows, group_keys))
                else:
                    parts.append(render_rows(rows))
        elif dataset.get("name") == "inquiry_records" and "inquiry_type" in keys:
            grouped: dict[str, list[dict[str, Any]]] = {}
            for row in dataset_rows:
                normalized = row.get("normalized") if isinstance(row.get("normalized"), dict) else {}
                canonical_raw = row.get("canonical_raw") if isinstance(row.get("canonical_raw"), dict) else {}
                group_key = str(normalized.get("inquiry_type", canonical_raw.get("inquiry_type")) or "unknown")
                grouped.setdefault(group_key, []).append(row)
            enum_labels = (dictionary.get("enums") or {}).get("inquiry_type") or {}
            for group_key in ("institution", "personal", *sorted(set(grouped) - {"institution", "personal"})):
                rows = grouped.get(group_key)
                if not rows:
                    continue
                parts.append(f"#### {_markdown_text(enum_labels.get(group_key) or group_key)}")
                parts.append(render_rows(rows))
        else:
            parts.append(render_rows(dataset_rows))

    for section_id, section in sections.items():
        if section_id not in rendered_sections:
            layout = section_layouts.get(str(section.get("type") or ""))
            if not suppress_empty_sections or section_has_renderable_content(
                section, layout if isinstance(layout, dict) else None
            ):
                parts.append(f"## {_markdown_text(section.get('title') or section.get('type') or section_id)}")

    appendix = presentation.get("appendix") if isinstance(presentation.get("appendix"), dict) else {}
    if appendix or deferred_appendix_datasets:
        all_items: dict[str, dict[str, Any]] = {}
        for section in sections.values():
            section_items, _section_groups = section_pools(section)
            all_items.update(section_items)
        appendix_parts: list[str] = []
        for dataset, table, dataset_layout in deferred_appendix_datasets:
            column_by_key = {
                str(column.get("key") or ""): column
                for column in dataset.get("columns") or []
                if column.get("key")
            }
            configured_keys = (
                dataset_layout.get("columns")
                or table.get("column_keys")
                or []
            )
            keys = [
                str(key)
                for key in configured_keys
                if str(key) in column_by_key
            ]
            if not keys:
                continue
            appendix_parts.append(
                f"### {_markdown_text(table.get('title') or dataset.get('label') or dataset.get('name'))}"
            )
            labels = [
                _markdown_text(column_by_key[key].get("label") or key)
                for key in keys
            ]
            lines = [
                "| " + " | ".join(labels) + " |",
                "| " + " | ".join("---" for _ in keys) + " |",
            ]
            for row in dataset.get("rows") or []:
                if not isinstance(row, dict):
                    continue
                normalized = (
                    row.get("normalized")
                    if isinstance(row.get("normalized"), dict)
                    else {}
                )
                canonical_raw = (
                    row.get("canonical_raw")
                    if isinstance(row.get("canonical_raw"), dict)
                    else {}
                )
                values = [
                    _markdown_display(
                        normalized.get(key, canonical_raw.get(key)),
                        key=key,
                        descriptor=column_by_key[key],
                        dictionary=dictionary,
                        privacy_mode=privacy_mode,
                    )
                    for key in keys
                ]
                lines.append("| " + " | ".join(values) + " |")
            appendix_parts.append("\n".join(lines))
        for reference in appendix.get("fields") or []:
            item = configured_item(reference, all_items)
            if item is not None:
                appendix_parts.append(item_line(item))
        for spec in appendix.get("document_fields") or []:
            item = document_item(spec) if isinstance(spec, dict) else None
            if item is not None:
                appendix_parts.append(item_line(item))
        facts = semantic_domain.get("facts") if isinstance(semantic_domain.get("facts"), dict) else {}
        audit = facts.get("credit_extraction_audit") if isinstance(facts.get("credit_extraction_audit"), dict) else {}
        reconciliations = {
            str(item.get("name") or ""): item
            for item in audit.get("reconciliations") or []
            if isinstance(item, dict) and item.get("name")
        }
        for spec in appendix.get("audit_reconciliations") or []:
            if not isinstance(spec, dict):
                continue
            reconciliation = reconciliations.get(str(spec.get("name") or ""))
            if reconciliation is None:
                continue
            reconciliation_parts: list[str] = []
            for field_spec in spec.get("fields") or []:
                if not isinstance(field_spec, dict):
                    continue
                key = str(field_spec.get("key") or "")
                value = reconciliation.get(key)
                if not key or value in (None, "", [], {}):
                    continue
                value_labels = (
                    field_spec.get("value_labels")
                    if isinstance(field_spec.get("value_labels"), dict)
                    else {}
                )
                shown = value_labels.get(value)
                if shown is None:
                    shown = _markdown_display(
                        value,
                        key=key,
                        descriptor=field_spec,
                        dictionary=dictionary,
                        privacy_mode=privacy_mode,
                    )
                reconciliation_parts.append(
                    f"**{_markdown_text(field_spec.get('label') or key)}:** "
                    f"{_markdown_text(shown)}"
                )
            if reconciliation_parts:
                appendix_parts.append(
                    f"### {_markdown_text(spec.get('title') or reconciliation.get('name'))}"
                )
                appendix_parts.extend(reconciliation_parts)
                if spec.get("note"):
                    appendix_parts.append(f"**处理原则:** {_markdown_text(spec['note'])}")
        if appendix_parts:
            parts.append(f"## {_markdown_text(appendix.get('title') or '附录')}")
            parts.extend(appendix_parts)
    return "\n\n".join(part for part in parts if part).rstrip() + "\n"


def _semantic_source_structure(
    result: Any,
    sections: list[dict[str, Any]],
    datasets: list[dict[str, Any]],
) -> dict[str, Any]:
    """Publish source-order blocks plus auditable logical reconstructions."""
    graph = getattr(result, "document_flow", None)

    def graph_values(name: str) -> list[Any]:
        values: list[Any] = []
        for value in list(getattr(graph, name, None) or []):
            if hasattr(value, "model_dump"):
                values.append(value.model_dump(mode="json", exclude_none=True))
            elif isinstance(value, dict):
                values.append(copy.deepcopy(value))
        return _json_safe(values)

    graph_nodes = list(getattr(graph, "nodes", None) or [])
    nodes_by_id = {str(getattr(node, "node_id", "") or ""): node for node in graph_nodes}
    ordered_nodes: list[Any] = []
    seen: set[str] = set()
    for flow in list(getattr(graph, "reading_flow", None) or []):
        for node_id in list(getattr(flow, "node_ids", None) or []):
            node_id = str(node_id)
            node = nodes_by_id.get(node_id)
            if node is not None and node_id not in seen:
                ordered_nodes.append(node)
                seen.add(node_id)
    ordered_nodes.extend(
        sorted(
            (node for node_id, node in nodes_by_id.items() if node_id not in seen),
            key=lambda node: (
                int(getattr(node, "page", 1) or 1),
                int(getattr(node, "reading_order", 0) or 0),
                str(getattr(node, "node_id", "") or ""),
            ),
        )
    )

    blocks: list[dict[str, Any]] = []
    section_titles = {
        re.sub(r"\s+", "", str(section.get("title") or "")): str(section.get("id") or "")
        for section in sections
        if section.get("title")
    }
    for order, node in enumerate(ordered_nodes, start=1):
        node_text = str(getattr(node, "text", "") or "")
        compact_node_text = re.sub(r"\s+", "", node_text)
        role = str(getattr(node, "role", "") or "body")
        if compact_node_text in section_titles:
            role = "heading"
        elif order == 1 and any(marker in compact_node_text for marker in ("信用报告", "征信报告")):
            role = "title"
        elif re.fullmatch(r"\d+[.、]?", compact_node_text):
            role = "list_marker"
        block: dict[str, Any] = {
            "id": str(getattr(node, "node_id", "") or f"source:block:{order:06d}"),
            "kind": str(getattr(node, "type", "") or "paragraph"),
            "role": role,
            "order": order,
            "page": max(1, int(getattr(node, "page", 1) or 1)),
            "text": str(getattr(node, "text", "") or ""),
            "fact_refs": [str(value) for value in (getattr(node, "fact_refs", None) or [])],
            "evidence_refs": [str(value) for value in (getattr(node, "evidence_refs", None) or [])],
        }
        bbox = getattr(node, "bbox", None)
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            block["bbox"] = [_json_safe(value) for value in bbox]
        confidence = getattr(node, "confidence", None)
        if confidence not in (None, ""):
            block["confidence"] = _json_safe(confidence)
        metadata = getattr(node, "metadata", None)
        if isinstance(metadata, dict) and metadata:
            block["extensions"] = _json_safe(metadata)
            if metadata.get("table_id"):
                block["source_table_ref"] = str(metadata["table_id"])
        blocks.append(block)

    if not blocks:
        fallback_candidates: list[tuple[int, float, int, dict[str, Any]]] = []
        serial = 0
        for page in list(getattr(result, "pages", None) or []):
            page_number = max(1, int(getattr(page, "page_number", 1) or 1))
            for text_index, text in enumerate(list(getattr(page, "texts", None) or []), start=1):
                serial += 1
                bbox = getattr(text, "bbox", None)
                level = str(getattr(text, "level", "") or "")
                kind = "heading" if any(marker in level.lower() for marker in ("title", "heading")) else "paragraph"
                block = {
                    "id": f"source:text:p{page_number}:{text_index}",
                    "kind": kind,
                    "role": "title" if kind == "heading" else "body",
                    "page": page_number,
                    "text": str(getattr(text, "content", "") or ""),
                    "fact_refs": [],
                    "evidence_refs": [str(value) for value in (getattr(text, "evidence_ids", None) or [])],
                }
                if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    block["bbox"] = [_json_safe(value) for value in bbox]
                top = float(bbox[1]) if isinstance(bbox, (list, tuple)) and len(bbox) == 4 else float(serial)
                fallback_candidates.append((page_number, top, serial, block))
            for pair_index, pair in enumerate(list(getattr(page, "key_values", None) or []), start=1):
                serial += 1
                bbox = getattr(pair, "bbox", None)
                block = {
                    "id": f"source:key_value:p{page_number}:{pair_index}",
                    "kind": "key_value",
                    "role": "body",
                    "page": page_number,
                    "text": f"{getattr(pair, 'key', '')}: {getattr(pair, 'value', '')}".strip(),
                    "key": str(getattr(pair, "key", "") or ""),
                    "value": str(getattr(pair, "value", "") or ""),
                    "fact_refs": [],
                    "evidence_refs": [],
                }
                if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    block["bbox"] = [_json_safe(value) for value in bbox]
                top = float(bbox[1]) if isinstance(bbox, (list, tuple)) and len(bbox) == 4 else float(serial)
                fallback_candidates.append((page_number, top, serial, block))
            for table_index, table in enumerate(list(getattr(page, "tables", None) or []), start=1):
                serial += 1
                bbox = getattr(table, "bbox", None)
                table_id = str(getattr(table, "table_id", "") or f"source:table:p{page_number}:{table_index}")
                block = {
                    "id": f"source:table_block:p{page_number}:{table_index}",
                    "kind": "physical_table",
                    "role": "body",
                    "page": page_number,
                    "text": "",
                    "source_table_ref": table_id,
                    "fact_refs": [],
                    "evidence_refs": [str(value) for value in (getattr(table, "evidence_ids", None) or [])],
                }
                if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    block["bbox"] = [_json_safe(value) for value in bbox]
                top = float(bbox[1]) if isinstance(bbox, (list, tuple)) and len(bbox) == 4 else float(serial)
                fallback_candidates.append((page_number, top, serial, block))
        blocks = [
            {**block, "order": order}
            for order, (_page, _top, _serial, block) in enumerate(sorted(fallback_candidates), start=1)
        ]

    source_tables: list[dict[str, Any]] = []
    table_order = 0
    for page in list(getattr(result, "pages", None) or []):
        page_number = max(1, int(getattr(page, "page_number", 1) or 1))
        for table_index, table in enumerate(list(getattr(page, "tables", None) or []), start=1):
            table_order += 1
            table_id = str(getattr(table, "table_id", "") or f"source:table:p{page_number}:{table_index}")
            source_table: dict[str, Any] = {
                "id": table_id,
                "page": page_number,
                "order": table_order,
                "headers": [str(value or "") for value in (getattr(table, "headers", None) or [])],
                "rows": [
                    [str(getattr(cell, "text", "") or "") for cell in (getattr(row, "cells", None) or [])]
                    for row in (getattr(table, "rows", None) or [])
                ],
                "row_models": [
                    _json_safe(row.model_dump(mode="json", exclude_none=True))
                    if hasattr(row, "model_dump")
                    else _json_safe(row)
                    for row in (getattr(table, "rows", None) or [])
                ],
            }
            bbox = getattr(table, "bbox", None)
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                source_table["bbox"] = [_json_safe(value) for value in bbox]
            metadata = getattr(table, "metadata", None)
            if isinstance(metadata, dict) and metadata:
                source_table["extensions"] = _json_safe(metadata)
            extraction_layer = getattr(table, "extraction_layer", None)
            if extraction_layer not in (None, ""):
                source_table["extraction_layer"] = str(extraction_layer)
            source_tables.append(source_table)

    synthesized_flows: list[dict[str, Any]] = []
    for dataset in datasets:
        if str(dataset.get("name") or "") != "inquiry_records":
            continue
        column_keys = [
            key
            for key in ("sequence", "inquiry_date", "institution", "reason", "inquiry_type")
            if any(str(column.get("key") or "") == key for column in dataset.get("columns") or [])
        ]
        logical_rows: list[list[str]] = []
        row_models: list[dict[str, Any]] = []
        pages: list[int] = []
        evidence_ids: list[str] = []
        for row in dataset.get("rows") or []:
            if not isinstance(row, dict):
                continue
            normalized = row.get("normalized") if isinstance(row.get("normalized"), dict) else {}
            raw = row.get("canonical_raw") if isinstance(row.get("canonical_raw"), dict) else {}
            source = row.get("source") if isinstance(row.get("source"), dict) else {}
            page_range = [int(value) for value in (source.get("page_range") or []) if int(value or 0) > 0]
            pages.extend(page_range)
            row_evidence = [str(value) for value in (source.get("evidence_ids") or []) if value]
            evidence_ids.extend(row_evidence)
            logical_rows.append([str(normalized.get(key, raw.get(key)) or "") for key in column_keys])
            row_models.append(
                {
                    "record_id": str(row.get("record_id") or ""),
                    "page_range": page_range,
                    "source_refs": _json_safe(source.get("source_refs") or []),
                    "evidence_ids": row_evidence,
                }
            )
        if not logical_rows:
            continue
        logical_id = f"logical:{dataset['id']}"
        page_start = min(pages) if pages else 1
        page_end = max(pages) if pages else page_start
        segments = [
            {
                "id": f"{logical_id}:p{page}",
                "page": page,
                "record_ids": [model["record_id"] for model in row_models if page in (model.get("page_range") or [])],
                "repeated_header": page != page_start,
            }
            for page in sorted(set(pages))
        ]
        source_tables.append(
            {
                "id": logical_id,
                "kind": "logical_reconstruction",
                "dataset_id": str(dataset.get("id") or ""),
                "section_id": str(dataset.get("section_id") or ""),
                "page": page_start,
                "page_start": page_start,
                "page_end": page_end,
                "order": len(source_tables) + 1,
                "headers": column_keys,
                "rows": logical_rows,
                "row_models": row_models,
                "record_ids": [model["record_id"] for model in row_models],
                "segments": segments,
                "evidence_ids": list(dict.fromkeys(evidence_ids)),
                "provenance": "plugin_reconstruction_from_source_blocks",
            }
        )
        if page_end > page_start:
            synthesized_flows.append(
                {
                    "id": f"flow:{logical_id}",
                    "type": "table_continuation",
                    "source_table_ref": logical_id,
                    "page_start": page_start,
                    "page_end": page_end,
                    "segment_refs": [segment["id"] for segment in segments],
                    "repeated_headers": [segment["id"] for segment in segments if segment["repeated_header"]],
                    "evidence_refs": list(dict.fromkeys(evidence_ids)),
                }
            )

    heading_indexes: dict[str, int] = {}
    for section in sections:
        section_id = str(section.get("id") or "")
        title = re.sub(r"\s+", "", str(section.get("title") or ""))
        matched = next(
            (
                index
                for index in range(len(blocks))
                if re.sub(r"\s+", "", str(blocks[index].get("text") or "")) == title
            ),
            None,
        )
        if matched is not None:
            heading_indexes[section_id] = matched
            blocks[matched]["role"] = "heading"

    ordered_heading_indexes = sorted(set(heading_indexes.values()))

    def next_heading(index: int) -> int:
        return next((value for value in ordered_heading_indexes if value > index), len(blocks))

    explicit_heading_claims = {
        section_id: set(range(index, next_heading(index)))
        for section_id, index in heading_indexes.items()
    }
    all_explicit_heading_claims = {
        index for claimed in explicit_heading_claims.values() for index in claimed
    }

    dataset_indexes_by_section: dict[str, list[int]] = {}
    for dataset in datasets:
        section_id = str(dataset.get("section_id") or "")
        for row in dataset.get("rows") or []:
            if not isinstance(row, dict):
                continue
            source = row.get("source") if isinstance(row.get("source"), dict) else {}
            for ref in source.get("source_refs") or []:
                if not isinstance(ref, dict):
                    continue
                for node_id in [ref.get("node_id"), *(ref.get("node_ids") or [])]:
                    node_id = str(node_id or "")
                    if node_id:
                        index = next(
                            (
                                position
                                for position, block in enumerate(blocks)
                                if str(block.get("id") or "") == node_id
                            ),
                            None,
                        )
                        if index is not None:
                            dataset_indexes_by_section.setdefault(section_id, []).append(index)

    claimed_by_dataset_sections: dict[str, set[int]] = {}
    for section_id, indexes in dataset_indexes_by_section.items():
        if not indexes:
            continue
        first_index = min(indexes)
        data_start = first_index
        while data_start > 0:
            previous = blocks[data_start - 1]
            if previous.get("role") == "heading" or previous.get("kind") == "physical_table":
                break
            data_start -= 1
        data_end = next_heading(max(indexes))
        claimed_by_dataset_sections[section_id] = set(range(data_start, data_end))
    all_dataset_claims = {index for claimed in claimed_by_dataset_sections.values() for index in claimed}

    semantic_sections: list[dict[str, Any]] = []
    for section in sections:
        public = {key: _json_safe(value) for key, value in section.items() if not key.startswith("_")}
        section_id = str(public.get("id") or "")
        if section_id in heading_indexes:
            start_index = heading_indexes[section_id]
            section_indexes = set(range(start_index, next_heading(start_index)))
            if section_id in claimed_by_dataset_sections:
                section_indexes.update(claimed_by_dataset_sections[section_id])
            else:
                section_indexes.difference_update(all_dataset_claims)
            section_blocks = [blocks[index] for index in sorted(section_indexes)]
            public["block_refs"] = [str(block["id"]) for block in section_blocks]
            public["source_table_refs"] = list(
                dict.fromkeys(
                    [
                        str(block.get("source_table_ref") or "")
                        for block in section_blocks
                        if block.get("source_table_ref")
                    ]
                    + [
                        str(table.get("id") or "")
                        for table in source_tables
                        if str(table.get("section_id") or "") == section_id
                    ]
                )
            )
        else:
            page_range = list(public.get("page_range") or [])
            if len(page_range) != 2:
                semantic_sections.append(public)
                continue
            start, end = int(page_range[0]), int(page_range[1])
            section_indexes = {
                index
                for index, block in enumerate(blocks)
                if start <= int(block.get("page") or 1) <= end
            }
            section_indexes.update(claimed_by_dataset_sections.get(section_id, set()))
            section_indexes.difference_update(all_explicit_heading_claims)
            public["block_refs"] = [
                str(blocks[index]["id"]) for index in sorted(section_indexes)
            ]
            public["source_table_refs"] = [
                str(table["id"]) for table in source_tables if start <= int(table.get("page") or 1) <= end
            ]
        semantic_sections.append(public)
    outline = graph_values("outline")
    if not outline:
        outline = [
            {
                "id": f"outline:{section.get('id')}",
                "level": 1,
                "title": str(section.get("title") or ""),
                "section_id": str(section.get("id") or ""),
                "block_ref": (
                    str(blocks[heading_indexes[str(section.get("id") or "")]]["id"])
                    if str(section.get("id") or "") in heading_indexes
                    else ""
                ),
            }
            for section in semantic_sections
        ]
    return {
        "sections": semantic_sections,
        "blocks": blocks,
        "source_tables": source_tables,
        "reading_flows": graph_values("reading_flow"),
        "outline": outline,
        "edges": graph_values("edges"),
        "relations": graph_values("relations"),
        "cross_page_flows": [*graph_values("cross_page_flows"), *synthesized_flows],
        "suppressed_noise": graph_values("suppressed_noise"),
    }


def _semantic_bindings(
    datasets: list[dict[str, Any]],
    structure: dict[str, Any],
) -> list[dict[str, Any]]:
    """Bind each public record to available public source blocks and tables."""
    block_ids = {str(block.get("id") or "") for block in structure.get("blocks") or [] if isinstance(block, dict)}
    table_ids = {
        str(table.get("id") or "") for table in structure.get("source_tables") or [] if isinstance(table, dict)
    }
    logical_tables_by_record: dict[str, list[str]] = {}
    for table in structure.get("source_tables") or []:
        if not isinstance(table, dict) or table.get("kind") != "logical_reconstruction":
            continue
        table_id = str(table.get("id") or "")
        for record_id in table.get("record_ids") or []:
            logical_tables_by_record.setdefault(str(record_id), []).append(table_id)
    bindings: list[dict[str, Any]] = []
    for dataset in datasets:
        dataset_id = str(dataset.get("id") or "")
        for row in dataset.get("rows") or []:
            if not isinstance(row, dict):
                continue
            record_id = str(row.get("record_id") or "")
            source = row.get("source") if isinstance(row.get("source"), dict) else {}
            source_block_refs: list[str] = []
            source_table_refs: list[str] = []
            evidence_refs: list[str] = []
            refs = [
                *list(source.get("source_refs") or []),
                *list(source.get("source_cell_refs") or []),
            ]
            for ref in refs:
                if not isinstance(ref, dict):
                    continue
                for node_id in [ref.get("node_id"), *(ref.get("node_ids") or [])]:
                    node_id = str(node_id or "")
                    if node_id in block_ids and node_id not in source_block_refs:
                        source_block_refs.append(node_id)
                for table_id in (ref.get("table_id"), ref.get("source_table_id")):
                    table_id = str(table_id or "")
                    if table_id in table_ids and table_id not in source_table_refs:
                        source_table_refs.append(table_id)
                for evidence in [ref.get("evidence_ref"), *(ref.get("evidence_ids") or [])]:
                    evidence = str(evidence or "")
                    if evidence and evidence not in evidence_refs:
                        evidence_refs.append(evidence)
            for evidence in source.get("evidence_ids") or []:
                evidence = str(evidence or "")
                if evidence and evidence not in evidence_refs:
                    evidence_refs.append(evidence)
            for table_id in logical_tables_by_record.get(record_id, []):
                if table_id not in source_table_refs:
                    source_table_refs.append(table_id)
            binding: dict[str, Any] = {
                "id": f"binding:{dataset_id}:{record_id}",
                "dataset_id": dataset_id,
                "record_id": record_id,
                "source_block_refs": source_block_refs,
                "source_table_refs": source_table_refs,
                "evidence_refs": evidence_refs,
            }
            page_range = list(source.get("page_range") or [])
            if len(page_range) == 2:
                binding["page_range"] = [int(page_range[0]), int(page_range[1])]
            bindings.append(binding)
    return bindings


def _community_view_from_semantic(semantic: dict[str, Any]) -> dict[str, Any]:
    """Return the stable Community JSON view of one semantic result."""
    structure = semantic.get("structure") if isinstance(semantic.get("structure"), dict) else {}
    payload = {
        "schema": {
            "name": "docmirror.community",
            "version": "3.0.0",
            "edition": "community",
            "domain": str((semantic.get("classification") or {}).get("document_type") or "generic"),
            "support_level": str((semantic.get("classification") or {}).get("support_level") or "generic"),
        },
        "document": copy.deepcopy(dict(semantic.get("document") or {})),
        "sections": [
            {
                key: copy.deepcopy(value)
                for key, value in section.items()
                if key not in {"block_refs", "source_table_refs"} and not key.startswith("_")
            }
            for section in (structure.get("sections") or [])
            if isinstance(section, dict)
        ],
        "datasets": copy.deepcopy(list(semantic.get("datasets") or [])),
        "reading": copy.deepcopy(dict(semantic.get("reading") or {})),
        "files": copy.deepcopy(dict(semantic.get("files") or {})),
        "warnings": copy.deepcopy(list(semantic.get("warnings") or [])),
    }
    extensions = (semantic.get("domain") or {}).get("extensions") or {}
    policy = extensions.get("compact_output") or {}
    if policy.get("normalized_only") is True:
        strip_source_value_pools(payload)
        # Persist the digital bank's explicit presentation choice for JSON replay.
        privacy_mode = (extensions.get("enhanced_markdown") or {}).get("privacy_mode")
        if privacy_mode in {"full", "masked"}:
            payload["reading"]["privacy_mode"] = privacy_mode
        if policy.get("business_view") is True:
            return business_view(payload)
    return payload


@dataclass
class CommunityDataset:
    public: dict[str, Any]
    rows: list[Any] = field(default_factory=list)

    def to_payload(
        self,
        *,
        fallback_page_range: list[int] | None = None,
        compact: bool = False,
        source_aliases: dict[str, Any] | None = None,
        preserve_canonical_raw: bool = False,
        normalized_only: bool = False,
        preserve_normalized_keys: bool = False,
    ) -> dict[str, Any]:
        """Return the self-contained public dataset, including every record."""
        metadata = {key: _json_safe(value) for key, value in self.public.items() if not key.startswith("_")}
        columns = list(metadata.get("columns") or [])
        projected_rows = [
            _public_record(
                row,
                dataset_id=str(metadata.get("id") or "dataset"),
                row_index=index,
                columns=columns,
                fallback_page_range=list(fallback_page_range or []),
                preserve_canonical_raw=preserve_canonical_raw,
            )
            for index, row in enumerate(self.rows, start=1)
        ]
        if preserve_normalized_keys:
            for projected, original in zip(projected_rows, self.rows, strict=True):
                projected["normalized"] = {key: value for key, value in projected["normalized"].items()
                                           if key in original["normalized"]}
        if normalized_only:
            enrich_normalized_dataset(projected_rows, self.rows, columns, source_aliases or {})
            metadata["columns"] = columns
        omitted = list(metadata.get("omitted_normalized_fields") or [])
        if compact:
            omitted = list(
                dict.fromkeys(
                    [
                        *omitted,
                        *_source_absent_normalized_fields(
                            self.rows,
                            columns,
                            source_aliases=source_aliases or {},
                            foreign_keys=list(metadata.get("foreign_keys") or []),
                        ),
                    ]
                )
            )
        if omitted:
            # Materialized Community JSON already carries source-absence
            # decisions; do not re-expand its sparse rows on a replay.
            omitted_keys = _effective_omitted_normalized_fields(
                {**metadata, "rows": projected_rows, "omitted_normalized_fields": omitted}
            )
            if omitted_keys:
                for row in projected_rows:
                    row["normalized"] = {
                        key: value
                        for key, value in row["normalized"].items()
                        if key not in omitted_keys or value not in (None, "")
                    }
                metadata["omitted_normalized_fields"] = [key for key in omitted if key in omitted_keys]
            else:
                metadata.pop("omitted_normalized_fields", None)
        record_ids = [str(row["record_id"]) for row in projected_rows]
        if len(record_ids) != len(set(record_ids)):
            raise ValueError(f"duplicate record_id in dataset {metadata.get('id')}")

        emitted = len(projected_rows)
        completeness = dict(metadata.get("completeness") or {})
        expected = int(completeness.get("expected_row_count", emitted) or 0)
        completeness.update(
            {
                "expected_row_count": expected,
                "emitted_row_count": emitted,
                "omitted_row_count": max(0, expected - emitted),
                "verified": bool(completeness.get("verified", expected == emitted)) and expected == emitted,
                "basis": str(completeness.get("basis") or "canonical_dataset"),
            }
        )
        metadata["row_count"] = emitted
        metadata["primary_key"] = "record_id"
        metadata["storage_role"] = "canonical"
        metadata["record_path"] = "rows"
        metadata["status"] = (
            "empty"
            if emitted == 0 and expected == 0
            else ("complete" if completeness["verified"] and expected == emitted else "partial")
        )
        metadata["completeness"] = completeness
        metadata["rows"] = projected_rows
        return metadata


@dataclass
class CommunityBundle:
    schema: dict[str, Any]
    document: dict[str, Any]
    sections: list[dict[str, Any]]
    datasets: list[CommunityDataset]
    files: dict[str, str]
    warnings: list[dict[str, Any]]
    result: Any
    source_fingerprint: str = ""
    parse_result_schema: str = "docmirror.sealed_parse_result.v1"
    classification: dict[str, Any] = field(default_factory=dict)
    domain: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    content_markdown_override: str = ""

    @property
    def compact_output(self) -> dict[str, Any]:
        """Return provider-opted-in formatting policy; defaults stay unchanged."""
        extensions = self.domain.get("extensions") or {}
        policy = extensions.get("compact_output") if isinstance(extensions, dict) else None
        if isinstance(policy, dict):
            return policy
        if self.schema.get("version") == "4.0.0":
            return {"normalized_only": True, "minify_json": True}
        if self.schema.get("version") == BUSINESS_VIEW_VERSION:
            return {"normalized_only": True, "minify_json": True, "business_view": True}
        if any(dataset.public.get("omitted_normalized_fields") for dataset in self.datasets):
            return {"minify_json": True}
        return {}

    def semantic_payload(self) -> dict[str, Any]:
        """Build and validate the public semantic source for all renderers."""
        sections_by_id = {str(section["id"]): section for section in self.sections}
        public_sections = [self._public_section(section) for section in self.sections]
        compact = self.compact_output
        source_aliases = compact.get("source_aliases") or {}
        public_datasets = [
            dataset.to_payload(
                fallback_page_range=list(
                    sections_by_id.get(str(dataset.public.get("section_id") or ""), {}).get("page_range") or []
                ),
                compact=compact.get("omit_absent_fields") is True,
                source_aliases=source_aliases.get(str(dataset.public.get("name") or ""), {}),
                preserve_canonical_raw=(
                    bool(compact) and self.diagnostics.get("materialized_from_community_json") is True
                ),
                normalized_only=compact.get("normalized_only") is True,
                preserve_normalized_keys=(
                    compact.get("business_view") is True and self.diagnostics.get("materialized_from_community_json") is True
                ),
            )
            for dataset in self.datasets
        ]
        reading = _build_public_reading_model(self.document, public_sections, public_datasets)
        fingerprint = self.source_fingerprint
        if not fingerprint and callable(getattr(self.result, "fact_fingerprint", None)):
            fingerprint = str(self.result.fact_fingerprint())
        classification = {
            "document_type": str(self.classification.get("document_type") or self.schema.get("domain") or "generic"),
            "projector_id": str(self.classification.get("projector_id") or "community.generic"),
            "support_level": str(
                self.classification.get("support_level") or self.schema.get("support_level") or "generic"
            ),
            **{
                str(key): _json_safe(value)
                for key, value in self.classification.items()
                if key not in {"document_type", "projector_id", "support_level"}
            },
        }
        structure = _semantic_source_structure(self.result, public_sections, public_datasets)
        domain = copy.deepcopy(self.domain)
        if compact.get("normalized_only") is True:
            domain.setdefault("extensions", {})["compact_output"] = copy.deepcopy(compact)
        payload = {
            "schema": {
                "name": "docmirror.community.semantic",
                "version": "1.0.0",
                "edition": "community",
                "document_type": classification["document_type"],
            },
            "source": {
                "parse_result_schema": self.parse_result_schema,
                "fingerprint": fingerprint or "unavailable",
                "file": copy.deepcopy(self.document.get("source_file") or {}),
            },
            "classification": classification,
            "document": copy.deepcopy(self.document),
            "structure": structure,
            "datasets": public_datasets,
            "bindings": _semantic_bindings(public_datasets, structure),
            "domain": _json_safe(domain),
            "reading": reading,
            "files": copy.deepcopy(self.files),
            "warnings": copy.deepcopy(self.warnings),
            "diagnostics": _json_safe(self.diagnostics),
        }
        return CommunitySemanticResult.model_validate(payload).model_dump(mode="json", by_alias=True)

    def json_payload(self, semantic: dict[str, Any] | None = None) -> dict[str, Any]:
        """Render the Community JSON API from the public semantic source."""
        return _community_view_from_semantic(semantic or self.semantic_payload())

    @staticmethod
    def _public_section(section: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in section.items() if not key.startswith("_")}

    def render_markdown(self) -> str:
        """Render the source-complete reading projection using DMP 1.0."""
        if self.content_markdown_override.strip():
            return self.content_markdown_override.rstrip() + "\n"
        markdown = render_markdown(self.result)
        if 'docmirror:nontext type="image" disposition="omitted"' in markdown and not any(
            warning.get("code") == "MARKDOWN_IMAGE_OMITTED" for warning in self.warnings
        ):
            self.warnings.append(
                {
                    "code": "MARKDOWN_IMAGE_OMITTED",
                    "level": "info",
                    "message": "Unmaterialized source images were omitted from content Markdown.",
                }
            )
        return markdown

    def render_enhanced_markdown(self, semantic: dict[str, Any] | None = None) -> str:
        """Transcribe the public Community reading model."""
        return render_community_reading_markdown(semantic or self.semantic_payload())

    def render_dataset_csvs(self, semantic: dict[str, Any] | None = None) -> dict[str, str]:
        """Render dataset CSVs directly from the public semantic source."""
        rendered: dict[str, str] = {}
        payload = semantic or self.semantic_payload()
        if self.compact_output.get("business_view") is True:
            payload = restore_business_records(self.json_payload(payload))
        for public in payload.get("datasets") or []:
            relative_path = str(public["csv"])
            columns = list(public.get("columns") or [])
            fieldnames = [*_SYSTEM_COLUMNS, *(str(column["key"]) for column in columns)]
            output = io.StringIO(newline="")
            writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
            writer.writeheader()
            for row in public.get("rows") or []:
                normalized = row.get("normalized") if isinstance(row.get("normalized"), dict) else {}
                raw = row.get("canonical_raw") if isinstance(row.get("canonical_raw"), dict) else {}
                source = row.get("source") if isinstance(row.get("source"), dict) else {}
                page_range = list(source.get("page_range") or [])
                output_row: dict[str, Any] = {
                    "record_id": str(row.get("record_id") or ""),
                    "_page_start": page_range[0] if page_range else "",
                    "_page_end": page_range[-1] if page_range else "",
                }
                for column in columns:
                    key = str(column["key"])
                    value = normalized.get(key, raw.get(key))
                    csv_type = (
                        "long_id"
                        if str(column.get("format") or "").lower() == "long_id"
                        else str(column.get("type") or "string")
                    )
                    output_row[key] = _csv_safe(
                        _json_value(value, csv_type),
                        csv_type,
                    )
                writer.writerow(output_row)
            rendered[relative_path] = "\ufeff" + output.getvalue()
        return rendered

    def conservation_issues(
        self,
        *,
        payload: dict[str, Any] | None = None,
        dataset_csvs: dict[str, str] | None = None,
    ) -> list[str]:
        """Return JSON/internal/CSV record conservation violations."""
        public_payload = payload or self.json_payload()
        if is_business_view(public_payload):
            public_payload = restore_business_records(public_payload)
        internal = {str(dataset.public.get("id") or ""): dataset for dataset in self.datasets}
        issues: list[str] = []
        for dataset_payload in public_payload.get("datasets") or []:
            dataset_id = str(dataset_payload.get("id") or "")
            rows = list(dataset_payload.get("rows") or [])
            row_count = int(dataset_payload.get("row_count") or 0)
            if row_count != len(rows):
                issues.append(f"{dataset_id}:row_count={row_count}:rows={len(rows)}")
            source = internal.get(dataset_id)
            if source is None:
                issues.append(f"{dataset_id}:missing_internal_dataset")
            elif len(source.rows) != len(rows):
                issues.append(f"{dataset_id}:internal={len(source.rows)}:json={len(rows)}")
            record_ids = [str(row.get("record_id") or "") for row in rows if isinstance(row, dict)]
            if len(record_ids) != len(rows) or any(not value for value in record_ids):
                issues.append(f"{dataset_id}:missing_record_id")
            if len(record_ids) != len(set(record_ids)):
                issues.append(f"{dataset_id}:duplicate_record_id")

            completeness = dataset_payload.get("completeness") or {}
            if int(completeness.get("emitted_row_count") or 0) != len(rows):
                issues.append(f"{dataset_id}:completeness_emitted_mismatch")

            if dataset_csvs is None:
                continue
            relative_path = str(dataset_payload.get("csv") or "")
            csv_content = dataset_csvs.get(relative_path)
            if csv_content is None:
                issues.append(f"{dataset_id}:missing_csv:{relative_path}")
                continue
            csv_rows = list(csv.DictReader(io.StringIO(csv_content.lstrip("\ufeff"))))
            csv_ids = [str(row.get("record_id") or "") for row in csv_rows]
            if len(csv_rows) != len(rows):
                issues.append(f"{dataset_id}:csv={len(csv_rows)}:json={len(rows)}")
            if csv_ids != record_ids:
                issues.append(f"{dataset_id}:csv_json_record_id_divergence")
        return issues

    def render_audit_csv(self, semantic: dict[str, Any] | None = None) -> str:
        """Render the audit CSV directly from the public semantic datasets."""
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=list(_AUDIT_COLUMNS), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        semantic_payload = semantic or self.semantic_payload()
        structure = semantic_payload.get("structure") if isinstance(semantic_payload.get("structure"), dict) else {}
        source_tables = {
            str(table["id"]): table
            for table in structure.get("source_tables") or []
            if isinstance(table, dict) and table.get("id")
        }
        sections_by_id = {
            str(section["id"]): section
            for section in (structure.get("sections") or [])
            if isinstance(section, dict) and section.get("id")
        }
        for public in semantic_payload.get("datasets") or []:
            if not isinstance(public, dict):
                continue
            section = sections_by_id.get(str(public.get("section_id") or ""), {})
            columns = {str(column["key"]): column for column in public.get("columns") or []}
            for row_index, row in enumerate(public.get("rows") or [], start=1):
                if not isinstance(row, dict):
                    continue
                normalized, raw = _record_pools(row)
                source = row.get("source") if isinstance(row.get("source"), dict) else {}
                page_range = _page_range(source, section.get("page_range") or [])
                record_id = _canonical_record_id(row, str(public["id"]), row_index)
                for key in columns:
                    value = normalized.get(key, raw.get(key))
                    raw_value = raw.get(key, value)
                    if self.compact_output.get("normalized_only") is True and (
                        key == "additional_fields"
                        or (
                            self.diagnostics.get("materialized_from_community_json") is True
                            and not row.get("canonical_raw")
                            and not row.get("raw")
                        )
                    ):
                        # Supplemental normalized fields have no single raw
                        # cell; a lean JSON replay has no raw evidence at all.
                        raw_value = ""
                    if value in (None, "") and raw_value in (None, ""):
                        continue
                    column = columns.get(key) or {"key": key, "label": key, "type": _type_of(value)}
                    value_type = (
                        "long_id"
                        if str(column.get("format") or "").lower() == "long_id"
                        else str(column.get("type") or "string")
                    )
                    safe_value, value_escaped = _csv_safe_with_flag(_json_value(value, value_type), value_type)
                    safe_raw, raw_escaped = _csv_safe_with_flag(_scalar(raw_value), value_type)
                    field_evidence = _field_evidence(
                        value,
                        source,
                        page_range,
                        raw_value=raw_value,
                        source_tables=source_tables,
                    )
                    writer.writerow(
                        {
                            "dataset_id": public.get("id", ""),
                            "record_id": record_id,
                            "field_key": key,
                            "value": safe_value,
                            "raw": safe_raw,
                            "value_type": value_type,
                            "unit": column.get("unit", ""),
                            **field_evidence,
                            "csv_escape_applied": "true" if value_escaped or raw_escaped else "false",
                        }
                    )
        domain = semantic_payload.get("domain") if isinstance(semantic_payload.get("domain"), dict) else {}
        extensions = domain.get("extensions") if isinstance(domain.get("extensions"), dict) else {}
        audit_csv = extensions.get("audit_csv") if isinstance(extensions.get("audit_csv"), dict) else {}
        facts = domain.get("facts") if isinstance(domain.get("facts"), dict) else {}
        audit = facts.get("credit_extraction_audit") if isinstance(facts.get("credit_extraction_audit"), dict) else {}
        reconciliations = {
            str(item.get("name") or ""): item
            for item in audit.get("reconciliations") or []
            if isinstance(item, dict) and item.get("name")
        }
        for spec in audit_csv.get("reconciliations") or []:
            if not isinstance(spec, dict):
                continue
            name = str(spec.get("name") or "")
            reconciliation = reconciliations.get(name)
            if reconciliation is None:
                continue
            for key in (str(value) for value in spec.get("fields") or []):
                value = reconciliation.get(key)
                if value in (None, "", [], {}):
                    continue
                value_type = _type_of(value)
                safe_value, value_escaped = _csv_safe_with_flag(value, value_type)
                unit = (
                    reconciliation.get("amount_unit", "")
                    if key in {"expected", "actual", "difference", "tolerance"}
                    else ""
                )
                writer.writerow(
                    {
                        "dataset_id": "_audit_reconciliations",
                        "record_id": f"audit:{name}",
                        "field_key": key,
                        "value": safe_value,
                        "raw": safe_value,
                        "value_type": value_type,
                        "unit": unit,
                        "csv_escape_applied": "true" if value_escaped else "false",
                    }
                )
        return "\ufeff" + output.getvalue()


def _csv_safe(value: Any, value_type: str = "string") -> Any:
    return _csv_safe_with_flag(value, value_type)[0]


def _csv_safe_with_flag(value: Any, value_type: str = "string") -> tuple[Any, bool]:
    if value is None:
        return "", False
    if value_type in {"array", "object", "json"} and isinstance(value, (list, dict)):
        return _canonical_json(value), False
    if not isinstance(value, str):
        return value, False
    # Prevent spreadsheet formula execution for textual cells without changing
    # legitimate signed numbers such as -10.25. JSON remains untouched.
    if value_type == "long_id" and value:
        return ("'" + value if not value.startswith("'") else value), not value.startswith("'")
    textual_types = {"string", "text", "enum", "date", "datetime", "json"}
    escaped = value_type in textual_types and value.startswith(("=", "+", "-", "@"))
    return ("'" + value if escaped else value), escaped


def _canonical_record_id(row: Any, dataset_id: str, row_index: int) -> str:
    if isinstance(row, dict) and row.get("record_id"):
        return str(row["record_id"])
    prefix = _slug(str(dataset_id).removeprefix("ds_"), "records")
    return f"{prefix}:r{row_index:06d}"


def _audit_match_key(value: Any) -> str:
    """Return a conservative comparison key for source-table cell matching."""
    scalar = _scalar(value)
    if scalar in (None, ""):
        return ""
    return re.sub(r"[\s,，]+", "", str(scalar)).casefold()


def _source_table_row(table: dict[str, Any], row_index: Any) -> dict[str, Any] | None:
    try:
        expected = int(row_index)
    except (TypeError, ValueError):
        return None
    rows = [row for row in table.get("row_models") or [] if isinstance(row, dict)]
    for row in rows:
        try:
            source_index = int(row.get("source_row_index"))
        except (TypeError, ValueError):
            continue
        if source_index == expected:
            return row
    return rows[expected] if 0 <= expected < len(rows) else None


def _bbox_union(cells: list[dict[str, Any]]) -> list[float] | None:
    boxes: list[list[float]] = []
    for cell in cells:
        bbox = cell.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            boxes.append([float(value) for value in bbox])
        except (TypeError, ValueError):
            continue
    if not boxes:
        return None
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


def _source_ref_token(ref: dict[str, Any]) -> str:
    """Serialize an existing structural source reference as stable audit evidence."""
    preferred = ("source", "page", "table_id", "row", "col", "node_id")
    keys = [key for key in preferred if ref.get(key) not in (None, "")]
    keys.extend(
        key
        for key in sorted(ref)
        if key not in preferred and ref.get(key) not in (None, "") and not isinstance(ref.get(key), (dict, list))
    )
    return "source-ref:" + "|".join(f"{key}={ref[key]}" for key in keys)


def _structural_field_evidence(
    value: Any,
    raw_value: Any,
    row_source: dict[str, Any],
    source_tables: dict[str, dict[str, Any]],
) -> tuple[list[str], list[float] | None, float | None, bool]:
    refs = [
        ref
        for key in ("source_cell_refs", "source_refs")
        for ref in (row_source.get(key) or [])
        if isinstance(ref, dict)
    ]
    if not refs:
        return [], None, None, False

    match_keys = {key for key in (_audit_match_key(value), _audit_match_key(raw_value)) if key}
    all_cells: list[dict[str, Any]] = []
    matched_cells: list[dict[str, Any]] = []
    for ref in refs:
        table = source_tables.get(str(ref.get("table_id") or ""))
        if table is None:
            continue
        row = _source_table_row(table, ref.get("row"))
        if row is None:
            continue
        cells = [cell for cell in row.get("cells") or [] if isinstance(cell, dict)]
        if ref.get("col") not in (None, ""):
            try:
                column = int(ref["col"])
            except (TypeError, ValueError):
                column = -1
            column_cells: list[dict[str, Any]] = []
            for cell in cells:
                try:
                    cell_column = int(cell.get("col_index", -1))
                except (TypeError, ValueError):
                    continue
                if cell_column == column:
                    column_cells.append(cell)
            cells = column_cells
        all_cells.extend(cells)
        matched_cells.extend(cell for cell in cells if _audit_match_key(cell.get("text")) in match_keys)

    selected_cells = matched_cells or all_cells
    evidence_ids = list(
        dict.fromkeys(
            str(evidence_id)
            for cell in selected_cells
            for evidence_id in (cell.get("evidence_ids") or cell.get("token_ids") or [])
            if evidence_id
        )
    )
    if not evidence_ids:
        evidence_ids = list(dict.fromkeys(token for ref in refs if (token := _source_ref_token(ref))))
    confidences: list[float] = []
    for cell in selected_cells:
        if cell.get("confidence") in (None, ""):
            continue
        try:
            confidences.append(float(cell["confidence"]))
        except (TypeError, ValueError):
            continue
    confidence = min(confidences) if confidences else None
    return evidence_ids, _bbox_union(selected_cells), confidence, bool(matched_cells)


def _field_evidence(
    value: Any,
    row: Any,
    fallback_page_range: list[int],
    *,
    raw_value: Any = None,
    source_tables: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    source = value if isinstance(value, dict) else {}
    row_source = row if isinstance(row, dict) else {}
    page_range = _page_range(source, _page_range(row_source, fallback_page_range))
    bbox = source.get("bbox") or _first_ref_value(source, "bbox") or _first_ref_value(row_source, "bbox")
    confidence = source.get("confidence", row_source.get("confidence", ""))
    evidence = (
        source.get("evidence_ref")
        or source.get("evidence_ids")
        or row_source.get("evidence_ids")
        or row_source.get("source_fact_ids")
        or ""
    )
    if source_tables:
        structural_evidence, structural_bbox, structural_confidence, exact_match = _structural_field_evidence(
            value,
            raw_value,
            row_source,
            source_tables,
        )
        if structural_evidence and (exact_match or not evidence):
            evidence = structural_evidence
        if structural_bbox and (exact_match or not bbox):
            bbox = structural_bbox
        if structural_confidence is not None and confidence in (None, ""):
            confidence = structural_confidence
    return {
        "page_start": page_range[0] if page_range else "",
        "page_end": page_range[-1] if page_range else "",
        "bbox": _canonical_json(bbox) if isinstance(bbox, (dict, list)) else bbox or "",
        "confidence": confidence,
        "evidence_ref": _canonical_json(evidence) if isinstance(evidence, (dict, list)) else evidence,
    }


def _first_ref_value(value: dict[str, Any], key: str) -> Any:
    for ref_key in ("source_refs", "source_cell_refs"):
        refs = value.get(ref_key) if isinstance(value.get(ref_key), list) else []
        for ref in refs:
            if isinstance(ref, dict) and ref.get(key) not in (None, ""):
                return ref[key]
    return ""


def project_community_bundle(
    result: Any,
    *,
    file_path: str = "",
    file_id: str = "001",
    document_id: str = "",
    projection_data: dict[str, Any] | None = None,
    projection_policy: dict[str, Any] | None = None,
) -> CommunityBundle:
    """Assemble Community Bundle v3 from Seal and post-seal plugin derivation."""
    from docmirror.models.sealed import SealedParseResult

    if not isinstance(result, SealedParseResult):
        raise TypeError(f"project_community_bundle expects SealedParseResult; got {type(result).__name__}")
    source_fingerprint = result.integrity_fingerprint
    parse_result_schema = result.schema_version
    result = result.to_read_view()
    derived = copy.deepcopy(dict(projection_data or {}))
    entities = getattr(result, "entities", None)
    extension = dict(getattr(entities, "domain_specific", None) or {})
    domain_facts = derived.get("domain_facts") if isinstance(derived.get("domain_facts"), dict) else {}
    extension.update(domain_facts)
    field_details = extension.get("field_details") if isinstance(extension.get("field_details"), dict) else {}
    dictionary = extension.get("data_dictionary") if isinstance(extension.get("data_dictionary"), dict) else {}
    fields: dict[str, Any] = {}
    for key in (
        "organization",
        "subject_name",
        "subject_id",
        "document_date",
        "period_start",
        "period_end",
    ):
        value = getattr(entities, key, None)
        if value not in (None, ""):
            fields[key] = value
    internal_fields = {str(key) for key in (projection_policy or {}).get("internal_fields") or ()}
    for key, value in extension.items():
        if (
            key.startswith("_")
            or key in {"field_details", "data_dictionary", "community_support_level"}
            or key in internal_fields
        ):
            continue
        if not isinstance(value, (dict, list)):
            fields[key] = value
    if isinstance(derived.get("entity_fields"), dict):
        fields.update({str(key): value for key, value in derived["entity_fields"].items() if value not in (None, "")})

    raw_sections = [
        section.model_dump(mode="json", exclude_none=True) if hasattr(section, "model_dump") else dict(section)
        for section in (getattr(result, "sections", None) or [])
        if hasattr(section, "model_dump") or isinstance(section, dict)
    ]
    if isinstance(derived.get("sections"), (list, tuple)) and derived["sections"]:
        raw_sections = [copy.deepcopy(section) for section in derived["sections"] if isinstance(section, dict)]
    data = {
        key: value
        for key, value in extension.items()
        if not key.startswith("_") and key not in {"field_details", "data_dictionary", "community_support_level"}
    }
    data.update(
        {
            "fields": fields,
            "field_details": field_details,
            "sections": raw_sections,
            "tables": [],
            "data_dictionary": dictionary,
        }
    )
    if isinstance(derived.get("datasets"), dict):
        data.update(
            {str(key): copy.deepcopy(value) for key, value in derived["datasets"].items() if isinstance(value, list)}
        )
    detected_type = str(derived.get("document_type") or getattr(entities, "document_type", None) or "generic")
    properties = {
        key: extension[key]
        for key in ("report_subtype", "content_mode", "units")
        if extension.get(key) not in (None, "")
    }
    source_name = Path(file_path or getattr(result, "file_path", "")).name
    parser_warnings = list(getattr(getattr(result, "parser_info", None), "warnings", None) or [])
    parser_warnings.extend(str(item) for item in (derived.get("warnings") or ()) if str(item))
    errors = list(getattr(result, "errors", None) or [])
    support_level = str(extension.get("community_support_level") or "")
    if not support_level and detected_type not in {"", "generic", "unknown"}:
        try:
            from docmirror.configs.ga_readiness import dgc_status_for_domain

            support_level = str(dgc_status_for_domain(detected_type) or "")
        except Exception:
            support_level = ""
    domain_view = {
        "document": {
            "document_type": detected_type,
            "document_name": source_name,
            "page_count": len(getattr(result, "pages", None) or []),
            "language": str(extension.get("language") or "zh"),
            "properties": properties,
        },
        "business": {"document_label": str(extension.get("document_label") or "")},
        "data": data,
        "quality": {"issues": []},
        "validation": getattr(getattr(result, "trust", None), "details", None) or {},
        "status": {
            "success": bool(getattr(result, "success", True)),
            "warnings": parser_warnings,
            "errors": errors,
        },
        "metadata": {"domain_status": support_level},
    }
    domain_document = domain_view.get("document") if isinstance(domain_view.get("document"), dict) else {}
    data = domain_view.get("data") if isinstance(domain_view.get("data"), dict) else {}
    dictionary = data.get("data_dictionary") if isinstance(data.get("data_dictionary"), dict) else {}
    projection = copy.deepcopy(dict(projection_policy or {}))
    semantic_extensions = (
        derived.get("semantic") if isinstance(derived.get("semantic"), dict) else {}
    )
    dataset_reading_columns = (
        semantic_extensions.get("dataset_reading_columns")
        if isinstance(semantic_extensions.get("dataset_reading_columns"), dict)
        else {}
    )
    domain = _domain(domain_view, projection)
    page_count = int(domain_document.get("page_count") or len(getattr(result, "pages", []) or []))
    path = Path(file_path) if file_path else None
    file_name = path.name if path else str(domain_document.get("document_name") or "")
    mime_type = mimetypes.guess_type(file_name)[0] or "application/octet-stream"
    title = str((domain_view.get("business") or {}).get("document_label") or "")
    if not title:
        title = path.stem if path else str(domain_document.get("document_name") or domain)
    language = str(domain_document.get("language") or "zh")
    language = "zh-CN" if language.lower() in {"zh", "cn", "zh_cn"} else language.replace("_", "-")
    properties = domain_document.get("properties") if isinstance(domain_document.get("properties"), dict) else {}
    units = properties.get("units") if isinstance(properties.get("units"), dict) else {}
    document = {
        "id": document_id or f"doc_{hashlib.sha1((file_name or domain).encode('utf-8')).hexdigest()[:16]}",
        "type": domain,
        "title": title,
        "page_count": page_count,
        "language": [language],
        "source_file": {
            "name": file_name,
            "mime_type": mime_type,
            "sha256": _source_hash(file_path),
        },
        "units": dict(units),
    }
    domain_schema = semantic_extensions.get("domain_schema")
    if isinstance(domain_schema, dict):
        document["domain_schema"] = _json_safe(domain_schema)
    raw_sections = [section for section in (data.get("sections") or []) if isinstance(section, dict)]
    sections = [
        _normalize_section(raw, index, page_count, projection) for index, raw in enumerate(raw_sections, start=1)
    ]
    if not sections:
        sections = [
            {
                "id": "sec_document",
                "title": title,
                "type": "document",
                "page_range": [1, max(1, page_count)],
                "items": [],
                "groups": [],
                "dataset_refs": [],
            }
        ]

    fields = data.get("fields") if isinstance(data.get("fields"), dict) else {}
    details = data.get("field_details") if isinstance(data.get("field_details"), dict) else {}
    field_section = next(
        (section for section in sections if section["type"] in {"basic_information", "identity"}),
        sections[0],
    )
    for key, value in fields.items():
        if value in (None, "", [], {}):
            continue
        detail = details.get(key) if isinstance(details.get(key), dict) else {}
        value_type, unit = _field_descriptor(str(key), dictionary, value)
        item: dict[str, Any] = {
            "key": str(key),
            "label": _field_label(str(key), dictionary),
            "value": _json_value(value, value_type),
            "raw": str(_raw_value(value, detail)),
            "type": value_type,
        }
        if unit:
            item["unit"] = unit
        field_descriptors = dictionary.get("fields") if isinstance(dictionary.get("fields"), dict) else {}
        descriptor = field_descriptors.get(str(key)) if isinstance(field_descriptors.get(str(key)), dict) else {}
        for metadata_key in ("definition", "sensitive", "display"):
            if descriptor.get(metadata_key) not in (None, ""):
                item[metadata_key] = _json_safe(descriptor[metadata_key])
        field_section["items"].append(item)

    for fact_key, section_type in (projection.get("summary_facts") or {}).items():
        summary_fact = data.get(fact_key) if isinstance(data.get(fact_key), dict) else {}
        if not summary_fact:
            continue
        summary_section = next((section for section in sections if section["type"] == section_type), sections[0])
        for key, value in summary_fact.items():
            if value in (None, "", [], {}):
                continue
            if isinstance(value, dict):
                group = {"key": str(key), "label": _field_label(str(key), dictionary), "items": []}
                field_descriptors = (
                    dictionary.get("fields")
                    if isinstance(dictionary.get("fields"), dict)
                    else {}
                )
                parent_descriptor = (
                    field_descriptors.get(str(key))
                    if isinstance(field_descriptors.get(str(key)), dict)
                    else {}
                )
                map_key_enum = str(parent_descriptor.get("map_key_enum") or "")
                enums = (
                    dictionary.get("enums")
                    if isinstance(dictionary.get("enums"), dict)
                    else {}
                )
                map_key_labels = (
                    enums.get(map_key_enum)
                    if map_key_enum and isinstance(enums.get(map_key_enum), dict)
                    else {}
                )
                for child_key, child_value in value.items():
                    if child_value in (None, "", [], {}):
                        continue
                    child_type = _type_of(child_value)
                    group["items"].append(
                        {
                            "key": str(child_key),
                            "label": str(
                                map_key_labels.get(child_key)
                                or _field_label(str(child_key), dictionary)
                            ),
                            "value": _json_value(child_value, child_type),
                            "raw": str(_scalar(child_value)),
                            "type": child_type,
                        }
                    )
                if group["items"]:
                    summary_section["groups"].append(group)
            elif not isinstance(value, list):
                value_type = _type_of(value)
                summary_section["items"].append(
                    {
                        "key": str(key),
                        "label": _field_label(str(key), dictionary),
                        "value": _json_value(value, value_type),
                        "raw": str(_scalar(value)),
                        "type": value_type,
                    }
                )

    dataset_candidates: list[tuple[str, list[Any]]] = []
    internal_facts = {str(key) for key in (projection.get("internal_facts") or ())}
    publish_empty_datasets = {
        str(key) for key in (projection.get("publish_empty_datasets") or ()) if str(key)
    }
    for key, value in data.items():
        if (
            key.startswith("_")
            or key in _NON_DATASET_KEYS
            or key in internal_facts
            or not isinstance(value, list)
            or (not value and key not in publish_empty_datasets)
        ):
            continue
        if not value or all(isinstance(item, dict) for item in value):
            dataset_candidates.append((str(key), value))
    configured_dataset_order = semantic_extensions.get("dataset_document_order")
    if isinstance(configured_dataset_order, list):
        dataset_rank = {
            str(name): index
            for index, name in enumerate(configured_dataset_order)
            if str(name)
        }
        fallback_rank = len(dataset_rank)
        dataset_candidates = [
            candidate
            for _original_index, candidate in sorted(
                enumerate(dataset_candidates),
                key=lambda item: (
                    dataset_rank.get(
                        str(
                            (projection.get("dataset_aliases") or {}).get(
                                item[1][0]
                            )
                            or item[1][0]
                        ),
                        fallback_rank,
                    ),
                    item[0],
                ),
            )
        ]
    datasets: list[CommunityDataset] = []
    csv_paths: set[str] = set()
    for key, rows in dataset_candidates:
        public_name = str((projection.get("dataset_aliases") or {}).get(key) or key)
        dataset_id = f"ds_{_slug(public_name, 'dataset')}"
        section_id = _dataset_section_id(data, key, sections, projection)
        label = str((projection.get("dataset_labels") or {}).get(public_name) or public_name.replace("_", " "))
        csv_path = f"{file_id}_datasets/{_slug(public_name, 'dataset')}.csv"
        if csv_path in csv_paths:
            raise ValueError(f"dataset CSV filename collision: {csv_path}")
        csv_paths.add(csv_path)
        dataset_type = str((projection.get("dataset_types") or {}).get(public_name) or "")
        if not dataset_type:
            dataset_type = _slug(public_name.removesuffix("_records") or "records", "dataset")
        public = {
            "id": dataset_id,
            "name": public_name,
            "label": label,
            "type": dataset_type,
            "section_id": section_id,
            "csv": csv_path,
            "row_count": len(rows),
            "grain": str(
                (projection.get("dataset_grains") or {}).get(public_name)
                or f"one row per {dataset_type}"
            ),
            "primary_key": "record_id",
            "schema_version": "1.0",
            "status": "complete" if rows else "empty",
            "columns": _dataset_columns(rows, dictionary, key),
            "completeness": _dataset_completeness(result, key, rows, projection, data),
        }
        representation_role = (projection.get("dataset_representation_roles") or {}).get(public_name)
        if representation_role:
            public["representation_role"] = str(representation_role)
        derived_from = (projection.get("dataset_derived_from") or {}).get(public_name)
        if isinstance(derived_from, (list, tuple)) and derived_from:
            public["derived_from"] = [str(value) for value in derived_from if str(value)]
        foreign_keys = (projection.get("dataset_foreign_keys") or {}).get(public_name)
        if isinstance(foreign_keys, (list, tuple)) and foreign_keys:
            public["foreign_keys"] = _json_safe(list(foreign_keys))
        configured_reading_columns = list(
            dataset_reading_columns.get(public_name)
            or (projection.get("reading_columns") or {}).get(public_name)
            or ()
        )
        if configured_reading_columns:
            public["reading_columns"] = [str(value) for value in configured_reading_columns]
        datasets.append(CommunityDataset(public=public, rows=rows))
        for section in sections:
            if section["id"] == section_id and dataset_id not in section["dataset_refs"]:
                section["dataset_refs"].append(dataset_id)

    datasets_by_section: dict[str, list[CommunityDataset]] = {}
    for dataset in datasets:
        datasets_by_section.setdefault(str(dataset.public.get("section_id") or ""), []).append(dataset)
    for section in sections:
        pages = list(section.get("page_range") or [])
        for dataset in datasets_by_section.get(str(section.get("id") or ""), []):
            for row in dataset.rows:
                pages.extend(_page_range(row, []))
        positive_pages = [int(page) for page in pages if isinstance(page, (int, float)) and int(page) > 0]
        if positive_pages:
            section["page_range"] = [min(positive_pages), max(positive_pages)]

    warnings: list[dict[str, Any]] = []
    seen_warnings: set[tuple[str, str]] = set()
    status = domain_view.get("status") if isinstance(domain_view.get("status"), dict) else {}
    warning_sources = [
        *(("error", str(value)) for value in (status.get("errors") or [])),
        *(("warning", str(value)) for value in (status.get("warnings") or [])),
    ]
    quality = domain_view.get("quality") if isinstance(domain_view.get("quality"), dict) else {}
    for issue in quality.get("issues") or []:
        if isinstance(issue, dict):
            warning_sources.append(
                (str(issue.get("severity") or "warning"), str(issue.get("source_code") or issue.get("message") or ""))
            )
    for level, raw in warning_sources:
        marker = (_warning_code(raw), raw)
        if not raw or marker in seen_warnings:
            continue
        seen_warnings.add(marker)
        if raw == "community_generic_fallback":
            level = "info"
        warnings.append(
            {"code": marker[0], "level": level if level in {"info", "warning", "error"} else "warning", "message": raw}
        )
    for dataset in datasets:
        completeness = dataset.public.get("completeness") or {}
        if completeness.get("verified") is False:
            emitted = completeness.get("emitted_row_count")
            expected = completeness.get("expected_row_count")
            independently_counted = completeness.get("basis") != "emitted_records_only"
            if independently_counted:
                if emitted == expected:
                    warnings.append(
                        {
                            "code": "DATASET_VERIFICATION_BLOCKED",
                            "level": "error",
                            "message": (
                                f"dataset {dataset.public.get('id')} emitted the expected {expected} records "
                                "but domain quality did not permit verification"
                            ),
                            "dataset_id": str(dataset.public.get("id") or ""),
                        }
                    )
                else:
                    warnings.append(
                        {
                            "code": "DATASET_ROW_COUNT_MISMATCH",
                            "level": "error",
                            "message": (
                                f"dataset {dataset.public.get('id')} emitted {emitted} of {expected} expected records"
                            ),
                            "dataset_id": str(dataset.public.get("id") or ""),
                        }
                    )
            else:
                warnings.append(
                    {
                        "code": "DATASET_COMPLETENESS_UNVERIFIED",
                        "level": "warning",
                        "message": (
                            f"dataset {dataset.public.get('id')} has {emitted} emitted records "
                            "but no independent source count"
                        ),
                        "dataset_id": str(dataset.public.get("id") or ""),
                    }
                )

    return CommunityBundle(
        schema={
            "name": "docmirror.community",
            "version": "3.0.0",
            "edition": "community",
            "domain": domain,
            "support_level": _support_level(domain_view, domain),
        },
        document=document,
        sections=sections,
        datasets=datasets,
        files={
            "content_md": f"{file_id}_content.md",
            "enhanced_reading_md": f"{file_id}_enhanced_reading.md",
            "datasets_dir": f"{file_id}_datasets",
            "dataset_audit_csv": f"{file_id}_datasets/_audit_cells.csv",
        },
        warnings=warnings,
        result=result,
        source_fingerprint=source_fingerprint,
        parse_result_schema=parse_result_schema,
        classification={
            "document_type": domain,
            "projector_id": str(derived.get("projector_id") or "community.generic"),
            "support_level": _support_level(domain_view, domain),
            "confidence": _json_safe(derived.get("confidence", 1.0)),
            "reason": str(derived.get("reason") or "post-seal domain projection"),
        },
        domain={
            "entity_fields": _json_safe(derived.get("entity_fields") or {}),
            "facts": _json_safe({key: value for key, value in domain_facts.items() if key != "data_dictionary"}),
            "data_dictionary": _json_safe(dictionary),
            "extensions": _json_safe(derived.get("semantic") or {}),
        },
        diagnostics={
            "evidence_ids": [str(value) for value in (derived.get("evidence_ids") or ())],
            "parser_warnings": parser_warnings,
            "parser_errors": [str(value) for value in errors],
        },
        content_markdown_override=str(derived.get("content_markdown_override") or ""),
    )


__all__ = ["CommunityBundle", "CommunityDataset", "project_community_bundle"]
