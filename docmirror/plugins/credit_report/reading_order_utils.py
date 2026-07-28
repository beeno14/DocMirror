# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared read-only helpers for credit-report reading-order reconstruction."""

from __future__ import annotations

from typing import Any


def valid_bbox(value: Any) -> tuple[float, float, float, float] | None:
    raw = getattr(value, "bbox", None)
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(item) for item in raw)
    except (TypeError, ValueError):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def ordered_document_nodes(parse_result: Any) -> list[Any]:
    flow = getattr(parse_result, "document_flow", None)
    nodes = list(getattr(flow, "nodes", None) or [])
    reading_flows = list(getattr(flow, "reading_flow", None) or [])
    if not nodes or not reading_flows:
        return []
    node_by_id = {str(getattr(node, "node_id", "") or ""): node for node in nodes}
    return [
        node_by_id[str(node_id)]
        for node_id in list(getattr(reading_flows[0], "node_ids", None) or [])
        if str(node_id) in node_by_id
    ]


__all__ = ["ordered_document_nodes", "valid_bbox"]
