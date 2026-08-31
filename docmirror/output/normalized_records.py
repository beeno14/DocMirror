"""Lossless, opt-in normalized business views of evidence-backed records.

This is an output-schema adapter, not an extractor. Existing normalized values
are never repaired or overwritten. Source fields whose representation cannot
be proved are kept as explicitly named supplemental normalized business fields.
"""

from __future__ import annotations

import copy
import re
import unicodedata
from datetime import datetime
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from typing import Any

ADDITIONAL_FIELDS_COLUMN = {
    "key": "additional_fields",
    "label": "补充业务字段",
    "type": "array",
    "nullable": True,
    "raw_available": False,
    "evidence_available": True,
    "definition": "Source-labeled business values not fully represented by the standard normalized fields.",
}


@lru_cache(maxsize=8192)
def _label(value: Any) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(value))).strip(":").casefold()


def _same(left: Any, right: Any) -> bool:
    """JSON equality that does not conflate false with zero."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_same(value, right[key]) for key, value in left.items())
    if isinstance(left, list):
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def value_is_represented(value: Any, normalized: Any, descriptor: dict[str, Any]) -> bool:
    """Recognize only lossless type/format conversions; no business inference."""
    if _same(value, normalized):
        return True
    if value in (None, "") and normalized in (None, ""):
        return True
    if isinstance(value, list) and value:
        # Header evidence can repeat the same value once per source page.
        return all(value_is_represented(item, normalized, descriptor) for item in value)
    if isinstance(value, bool) or isinstance(normalized, bool):
        return False
    if isinstance(value, str) and isinstance(normalized, str):
        if " ".join(value.split()) == " ".join(normalized.split()):
            return True
    value_type = str(descriptor.get("format") or descriptor.get("type") or "")
    if value_type in {"money", "decimal", "integer", "number"}:
        def decimal(item: Any) -> Decimal | None:
            text = str(item).strip()
            if not re.fullmatch(r"[+-]?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?", text):
                return None
            try:
                return Decimal(text.replace(",", ""))
            except InvalidOperation:
                return None

        left, right = decimal(value), decimal(normalized)
        return left is not None and right is not None and left == right
    if value_type in {"date", "datetime"} and isinstance(value, str) and isinstance(normalized, str):
        formats = ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%Y%m%d", "%Y年%m月%d日")
        if value_type == "datetime":
            formats = tuple(
                date + time
                for date in formats
                for time in (" %H:%M:%S", "T%H:%M:%S", " %H:%M", "%H%M%S")
            )

        def date_value(text: str) -> datetime | None:
            for fmt in formats:
                try:
                    return datetime.strptime(text.strip(), fmt)
                except ValueError:
                    continue
            return None

        left, right = date_value(value), date_value(normalized)
        return left is not None and right is not None and left == right
    return False


def source_fields(
    name: str,
    row: dict[str, Any],
    columns: list[dict[str, Any]],
    aliases: dict[str, Any],
) -> list[str]:
    """Resolve source roles only through declared names or field provenance."""
    source_label = _label(name)
    field_sources = (row.get("source") or {}).get("field_sources") or {}
    matches = []
    for column in columns:
        key = str(column["key"])
        if key == "additional_fields":
            continue
        detail = field_sources.get(key) or {}
        names = [key, column.get("label", key), *list(aliases.get(key) or [])]
        if isinstance(detail, dict) and detail.get("raw_name"):
            names.append(detail["raw_name"])
        for candidate in names:
            label = _label(candidate)
            # A Chinese heading may be followed by its English translation.
            bilingual = bool(
                label
                and re.search(r"[\u3400-\u9fff]", label)
                and source_label.startswith(label)
                and re.fullmatch(r"[a-z0-9/()._-]+", source_label[len(label):])
            )
            if label and (source_label == label or bilingual):
                matches.append(key)
                break
    return matches


def _delivery_value_excluded(source: dict[str, Any], pool: str, name: str, value: Any) -> bool:
    """Match one exact producer-proved exclusion retained with internal evidence."""

    source_detail = source.get("source") if isinstance(source.get("source"), dict) else {}
    exclusions = source_detail.get("_delivery_value_exclusions") or []
    return any(
        isinstance(item, dict)
        and item.get("pool") == pool
        and item.get("name") == name
        and "value" in item
        and _same(item["value"], value)
        for item in exclusions
    )


def additional_business_fields(
    row: dict[str, Any],
    columns: list[dict[str, Any]],
    aliases: dict[str, Any],
    *,
    source_row: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Preserve every source cell not demonstrably represented in its role.

    Unknown labels are never discarded just because an unrelated normalized
    field happens to have the same value. Ambiguous/compound cells and source
    placeholders are retained rather than guessed into standard fields.
    """
    normalized = row.get("normalized") or {}
    descriptors = {str(column["key"]): column for column in columns}
    source = source_row if source_row is not None else row
    raw = source.get("raw") or {}
    result: list[dict[str, Any]] = []

    for name, value in raw.items():
        if _delivery_value_excluded(source, "raw", str(name), value):
            continue
        fields = source_fields(str(name), row, columns, aliases)
        if any(
            key in normalized and value_is_represented(value, normalized[key], descriptors[key])
            for key in fields
        ):
            continue
        result.append({"name": str(name), "value": copy.deepcopy(value)})
    for key, value in (source.get("canonical_raw") or {}).items():
        if _delivery_value_excluded(source, "canonical_raw", str(key), value):
            continue
        if key in normalized and value_is_represented(value, normalized[key], descriptors.get(key, {})):
            continue
        if value in (None, ""):
            continue
        # Merge only when the original header explicitly identifies this role.
        matching = next(
            (
                item
                for item in result
                if "field" not in item
                and _same(item["value"], value)
                and key in source_fields(item["name"], row, columns, aliases)
            ),
            None,
        )
        if matching is not None:
            matching["field"] = str(key)
        else:
            result.append({"name": str(key), "field": str(key), "value": copy.deepcopy(value)})
    return result


def enrich_normalized_dataset(
    rows: list[dict[str, Any]],
    source_rows: list[Any],
    columns: list[dict[str, Any]],
    aliases: dict[str, Any],
) -> None:
    """Upgrade a fresh projection only; never mutate extracted source rows."""
    for row, source_row in zip(rows, source_rows, strict=True):
        normalized = row["normalized"]
        existing = normalized.get("additional_fields")
        if existing is not None and not isinstance(existing, list):
            raise ValueError("normalized.additional_fields must be an array")
        additions = additional_business_fields(
            row, columns, aliases, source_row=source_row if isinstance(source_row, dict) else None
        )
        if additions:
            values = copy.deepcopy(existing or [])
            for addition in additions:
                if not any(_same(addition, value) for value in values):
                    values.append(addition)
            normalized["additional_fields"] = values
        elif isinstance(source_row, dict) and "additional_fields" not in (source_row.get("normalized") or {}):
            normalized.pop("additional_fields", None)
    if any(row["normalized"].get("additional_fields") for row in rows):
        if not any(column["key"] == "additional_fields" for column in columns):
            columns.append(copy.deepcopy(ADDITIONAL_FIELDS_COLUMN))


def strip_source_value_pools(payload: dict[str, Any]) -> None:
    """Strip source-only value planes from a fresh v4 delivery projection."""
    payload["schema"]["version"] = "4.0.0"
    for dataset in payload.get("datasets") or []:
        for row in dataset.get("rows") or []:
            row.pop("canonical_raw", None)
            row.pop("raw", None)
    for section in payload.get("sections") or []:
        items = list(section.get("items") or [])
        items.extend(item for group in section.get("groups") or [] for item in group.get("items") or [])
        for item in items:
            if "raw" not in item:
                continue
            value = item.pop("raw")
            if not value_is_represented(value, item.get("value"), item):
                additional = item.setdefault("additional_values", [])
                if not any(_same(value, prior) for prior in additional):
                    additional.append(value)
