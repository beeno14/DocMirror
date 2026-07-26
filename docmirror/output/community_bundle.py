# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Community Bundle v3 JSON, Markdown, dataset CSV, and audit CSV projection."""

from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import mimetypes
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from docmirror.models.community_semantic import CommunitySemanticResult
from docmirror.output.markdown_renderer import render_markdown

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
    value = _scalar(value)
    if value_type in {"money", "decimal"} and value not in (None, ""):
        return str(value)
    return value


def _source_pages(value: Any) -> list[int]:
    pages: list[int] = []
    if isinstance(value, dict):
        direct = value.get("source_page") or value.get("source_page_number") or value.get("page")
        try:
            if int(direct or 0) > 0:
                pages.append(int(direct))
        except (TypeError, ValueError):
            pass
        for page in value.get("page_range") or []:
            try:
                if int(page or 0) > 0:
                    pages.append(int(page))
            except (TypeError, ValueError):
                pass
        for key in ("source", "source_refs", "source_cell_refs"):
            refs = value.get(key)
            refs = [refs] if isinstance(refs, dict) else refs
            if not isinstance(refs, list):
                continue
            for ref in refs:
                pages.extend(_source_pages(ref))
    return sorted(set(pages))


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
        return (
            {str(key): normalized.get(key, raw.get(key)) for key in keys},
            {str(key): raw.get(key, normalized.get(key)) for key in keys},
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


def _public_record(
    row: Any,
    *,
    dataset_id: str,
    row_index: int,
    columns: list[dict[str, Any]],
    fallback_page_range: list[int],
) -> dict[str, Any]:
    """Project one complete, stable Community API record."""
    record_id = _canonical_record_id(row, dataset_id, row_index)
    normalized, canonical_raw = _record_pools(row)
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
) -> dict[str, Any]:
    """Resolve an independent expected count where the physical contract permits it."""
    emitted = len(rows)
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
        "verified": True,
        "basis": "canonical_dataset",
    }


def _warning_code(raw: str) -> str:
    base = raw.split(":", 1)[0]
    code = re.sub(r"[^0-9A-Za-z]+", "_", base).strip("_").upper()
    return code or "PARTIAL_PARSE"


def _reading_column_keys(dataset: dict[str, Any]) -> list[str]:
    available = [str(column.get("key") or "") for column in dataset.get("columns") or [] if column.get("key")]
    preferred = _READING_COLUMN_PREFERENCES.get(str(dataset.get("name") or ""), ())
    selected = [key for key in preferred if key in available]
    return selected or available


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


def render_community_reading_markdown(payload: dict[str, Any]) -> str:
    """Transcribe a public Community semantic result without accessing ParseResult."""
    if (payload.get("schema") or {}).get("name") == "docmirror.community.semantic":
        payload = _community_view_from_semantic(payload)
    document = payload.get("document") or {}
    sections = {str(section.get("id") or ""): section for section in payload.get("sections") or []}
    datasets = {str(dataset.get("id") or ""): dataset for dataset in payload.get("datasets") or []}
    reading = payload.get("reading") or {}
    tables = {str(table.get("dataset_id") or ""): table for table in reading.get("tables") or []}
    parts = [
        '<!-- docmirror:markdown-profile version="1.0" -->',
        '<!-- docmirror:reading-profile version="2.0" mode="enhanced" source="community-semantic" -->',
        f"# {_markdown_text(document.get('title') or document.get('type') or 'Document')}",
    ]
    metadata = [
        ("Document type", document.get("type")),
        ("Pages", document.get("page_count")),
    ]
    for label, value in metadata:
        if value not in (None, ""):
            parts.append(f"**{label}:** {_markdown_text(value)}")

    rendered_sections: set[str] = set()
    for entry in reading.get("document_flow") or []:
        kind = str(entry.get("kind") or "")
        ref_id = str(entry.get("ref_id") or "")
        if kind == "section":
            section = sections.get(ref_id)
            if section is None:
                continue
            rendered_sections.add(ref_id)
            parts.append(f"## {_markdown_text(section.get('title') or section.get('type') or ref_id)}")
            for item in section.get("items") or []:
                parts.append(f"**{_markdown_text(item.get('label') or item.get('key'))}:** {_markdown_text(item.get('value'))}")
            for group in section.get("groups") or []:
                parts.append(f"### {_markdown_text(group.get('label') or group.get('key'))}")
                for item in group.get("items") or []:
                    parts.append(
                        f"**{_markdown_text(item.get('label') or item.get('key'))}:** "
                        f"{_markdown_text(item.get('value'))}"
                    )
            continue
        if kind != "dataset":
            continue
        dataset = datasets.get(ref_id)
        table = tables.get(ref_id)
        if dataset is None or table is None:
            continue
        parts.append(f"### {_markdown_text(table.get('title') or dataset.get('label') or ref_id)}")
        column_by_key = {
            str(column.get("key") or ""): column for column in dataset.get("columns") or [] if column.get("key")
        }
        keys = [str(key) for key in table.get("column_keys") or [] if str(key) in column_by_key]
        if not keys:
            parts.append("_No displayable columns._")
            continue
        labels = [_markdown_text(column_by_key[key].get("label") or key) for key in keys]
        lines = [
            "| " + " | ".join(labels) + " |",
            "| " + " | ".join("---" for _ in keys) + " |",
        ]
        for row in dataset.get("rows") or []:
            normalized = row.get("normalized") if isinstance(row.get("normalized"), dict) else {}
            canonical_raw = row.get("canonical_raw") if isinstance(row.get("canonical_raw"), dict) else {}
            values = [_markdown_text(normalized.get(key, canonical_raw.get(key))) for key in keys]
            lines.append("| " + " | ".join(values) + " |")
        parts.append("\n".join(lines))

    for section_id, section in sections.items():
        if section_id not in rendered_sections:
            parts.append(f"## {_markdown_text(section.get('title') or section.get('type') or section_id)}")
    return "\n\n".join(part for part in parts if part).rstrip() + "\n"


def _semantic_source_structure(result: Any, sections: list[dict[str, Any]]) -> dict[str, Any]:
    """Publish source-order blocks and physical tables without mutating ParseResult."""
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
    for order, node in enumerate(ordered_nodes, start=1):
        block: dict[str, Any] = {
            "id": str(getattr(node, "node_id", "") or f"source:block:{order:06d}"),
            "kind": str(getattr(node, "type", "") or "paragraph"),
            "role": str(getattr(node, "role", "") or "body"),
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

    semantic_sections: list[dict[str, Any]] = []
    for section in sections:
        public = {key: _json_safe(value) for key, value in section.items() if not key.startswith("_")}
        page_range = list(public.get("page_range") or [])
        if len(page_range) == 2:
            start, end = int(page_range[0]), int(page_range[1])
            public["block_refs"] = [
                str(block["id"]) for block in blocks if start <= int(block.get("page") or 1) <= end
            ]
            public["source_table_refs"] = [
                str(table["id"]) for table in source_tables if start <= int(table.get("page") or 1) <= end
            ]
        semantic_sections.append(public)
    return {
        "sections": semantic_sections,
        "blocks": blocks,
        "source_tables": source_tables,
        "reading_flows": graph_values("reading_flow"),
        "outline": graph_values("outline"),
        "edges": graph_values("edges"),
        "relations": graph_values("relations"),
        "cross_page_flows": graph_values("cross_page_flows"),
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
    return {
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


@dataclass
class CommunityDataset:
    public: dict[str, Any]
    rows: list[Any] = field(default_factory=list)

    def to_payload(self, *, fallback_page_range: list[int] | None = None) -> dict[str, Any]:
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
            )
            for index, row in enumerate(self.rows, start=1)
        ]
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
        metadata["status"] = (
            "empty" if emitted == 0 and expected == 0 else ("complete" if expected == emitted else "partial")
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

    def semantic_payload(self) -> dict[str, Any]:
        """Build and validate the public semantic source for all renderers."""
        sections_by_id = {str(section["id"]): section for section in self.sections}
        public_sections = [self._public_section(section) for section in self.sections]
        public_datasets = [
            dataset.to_payload(
                fallback_page_range=list(
                    sections_by_id.get(str(dataset.public.get("section_id") or ""), {}).get("page_range") or []
                )
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
        structure = _semantic_source_structure(self.result, public_sections)
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
            "domain": _json_safe(self.domain),
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
        for public in (semantic or self.semantic_payload()).get("datasets") or []:
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
                    output_row[key] = _csv_safe(
                        _json_value(value, str(column.get("type") or "string")),
                        str(column.get("type") or "string"),
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
                    if value in (None, "") and raw_value in (None, ""):
                        continue
                    column = columns.get(key) or {"key": key, "label": key, "type": _type_of(value)}
                    value_type = str(column.get("type") or "string")
                    safe_value, value_escaped = _csv_safe_with_flag(_json_value(value, value_type), value_type)
                    safe_raw, raw_escaped = _csv_safe_with_flag(_scalar(raw_value), value_type)
                    field_evidence = _field_evidence(value, source, page_range)
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
        return "\ufeff" + output.getvalue()


def _csv_safe(value: Any, value_type: str = "string") -> Any:
    return _csv_safe_with_flag(value, value_type)[0]


def _csv_safe_with_flag(value: Any, value_type: str = "string") -> tuple[Any, bool]:
    if value is None:
        return "", False
    if not isinstance(value, str):
        return value, False
    # Prevent spreadsheet formula execution for textual cells without changing
    # legitimate signed numbers such as -10.25. JSON remains untouched.
    textual_types = {"string", "text", "enum", "date", "datetime"}
    escaped = value_type in textual_types and value.startswith(("=", "+", "-", "@"))
    return ("'" + value if escaped else value), escaped


def _canonical_record_id(row: Any, dataset_id: str, row_index: int) -> str:
    if isinstance(row, dict) and row.get("record_id"):
        return str(row["record_id"])
    prefix = _slug(str(dataset_id).removeprefix("ds_"), "records")
    return f"{prefix}:r{row_index:06d}"


def _field_evidence(value: Any, row: Any, fallback_page_range: list[int]) -> dict[str, Any]:
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
    for key, value in extension.items():
        if key.startswith("_") or key in {"field_details", "data_dictionary", "community_support_level"}:
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
                group = {"key": str(key), "label": str(key).replace("_", " "), "items": []}
                for child_key, child_value in value.items():
                    if child_value in (None, "", [], {}):
                        continue
                    child_type = _type_of(child_value)
                    group["items"].append(
                        {
                            "key": str(child_key),
                            "label": str(child_key).replace("_", " "),
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
                        "label": str(key).replace("_", " "),
                        "value": _json_value(value, value_type),
                        "raw": str(_scalar(value)),
                        "type": value_type,
                    }
                )

    dataset_candidates: list[tuple[str, list[Any]]] = []
    internal_facts = {str(key) for key in (projection.get("internal_facts") or ())}
    for key, value in data.items():
        if (
            key.startswith("_")
            or key in _NON_DATASET_KEYS
            or key in internal_facts
            or not isinstance(value, list)
            or not value
        ):
            continue
        if all(isinstance(item, dict) for item in value):
            dataset_candidates.append((str(key), value))
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
            "grain": f"one row per {dataset_type}",
            "primary_key": "record_id",
            "schema_version": "1.0",
            "status": "complete" if rows else "empty",
            "columns": _dataset_columns(rows, dictionary, key),
            "completeness": _dataset_completeness(result, key, rows, projection),
        }
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
            warnings.append(
                {
                    "code": "DATASET_ROW_COUNT_MISMATCH",
                    "level": "error",
                    "message": (
                        f"dataset {dataset.public.get('id')} emitted {completeness.get('emitted_row_count')} "
                        f"of {completeness.get('expected_row_count')} expected records"
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
            "semantic_json": f"{file_id}_community_semantic.json",
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
            "facts": _json_safe(domain_facts),
            "data_dictionary": _json_safe(dictionary),
            "extensions": _json_safe(derived.get("semantic") or {}),
        },
        diagnostics={
            "evidence_ids": [str(value) for value in (derived.get("evidence_ids") or ())],
            "parser_warnings": parser_warnings,
            "parser_errors": [str(value) for value in errors],
        },
    )


__all__ = ["CommunityBundle", "CommunityDataset", "project_community_bundle"]
