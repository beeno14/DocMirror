# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ProjectionData construction for the generic post-seal projector."""

from __future__ import annotations

from typing import Any

from docmirror.plugins._base.projector import ProjectionData

_RECORD_METADATA_KEYS = {
    "record_id",
    "normalized",
    "canonical_raw",
    "raw",
    "source",
    "confidence",
    "review",
}
def _source_columns(rows: list[dict[str, Any]]) -> list[str]:
    """Return first-seen business columns from source records."""
    ordered: list[str] = []
    for row in rows:
        raw = row.get("canonical_raw") if isinstance(row.get("canonical_raw"), dict) else row.get("raw")
        normalized = row.get("normalized") if isinstance(row.get("normalized"), dict) else None
        record_pools = (raw, normalized) if isinstance(raw, dict) or isinstance(normalized, dict) else (row,)
        pools = [pool for pool in record_pools if isinstance(pool, dict)]
        for pool in pools:
            for key in pool:
                name = str(key)
                if name not in _RECORD_METADATA_KEYS and name not in ordered:
                    ordered.append(name)
    return ordered


def _dataset_column_semantic(datasets: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    orders = {dataset_id: _source_columns(rows) for dataset_id, rows in datasets.items()}
    orders = {dataset_id: columns for dataset_id, columns in orders.items() if columns}
    return {
        "dataset_column_order": orders,
        "dataset_reading_columns": orders,
    }


def make_generic_projection(
    detected_type: str,
    fields: dict[str, Any],
    structured_data: dict[str, Any],
    warnings: list[str],
) -> ProjectionData:
    records = structured_data["records"]
    normalized_entity_fields = {
        str(key): descriptor.get("value", descriptor.get("normalized_value"))
        for key, descriptor in structured_data["normalized_fields"].items()
        if isinstance(descriptor, dict)
        and descriptor.get("value", descriptor.get("normalized_value")) not in (None, "")
    }
    raw_datasets = structured_data.get("datasets")
    if not isinstance(raw_datasets, dict):
        raw_datasets = {"records": records} if records else {}
    canonical_datasets: dict[str, list[dict[str, Any]]] = {}
    for dataset_id, dataset_rows in raw_datasets.items():
        if not isinstance(dataset_rows, list):
            continue
        canonical_rows = [
            {
                **dict(record),
                "record_id": str(record.get("record_id") or f"{dataset_id}:r{index:06d}"),
            }
            for index, record in enumerate(dataset_rows, start=1)
            if isinstance(record, dict)
        ]
        if canonical_rows:
            canonical_datasets[str(dataset_id)] = canonical_rows
    return ProjectionData(
        projector_id="generic",
        document_type=detected_type,
        entity_fields={
            **normalized_entity_fields,
            **{
                key: fields[key]
                for key in ("subject_name", "subject_id", "organization", "document_date", "period_start", "period_end")
                if fields.get(key) not in (None, "")
            },
        },
        domain_facts={
            **fields,
            "field_details": structured_data["field_metadata"],
            "summary": structured_data["summary"],
            "normalized_fields": structured_data["normalized_fields"],
            "field_schema": structured_data["field_schema"],
            "columns": structured_data.get("columns", {}),
            "identities": structured_data.get("identities", {}),
        },
        semantic=_dataset_column_semantic(canonical_datasets),
        datasets=canonical_datasets,
        sections=tuple(dict(section) for section in structured_data["sections"]),
        warnings=tuple(warnings),
        reason="post-seal generic projection",
    )


def build_generic_projection(parse_result: Any, detected_type: str, full_text: str = "") -> ProjectionData:
    """Derive generic projection data from a sealed read view."""
    from docmirror.plugins._base.generic_community_adapter import derive_generic_projection

    projection = derive_generic_projection(parse_result, detected_type, full_text)
    if not isinstance(projection, ProjectionData):
        raise TypeError("generic projector did not return ProjectionData")
    return projection
