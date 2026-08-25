# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
KV identity enrichment parser for bank statement header rows.

Scans table header and preamble rows for embedded ``key: value`` cells (account
holder, account number, query period) and merges them into identity field maps.

Pipeline role: auxiliary parser in ``style_registry`` chain; runs before or alongside
grid parsers to populate identity metadata missing from Mirror entities.

Key exports: ``PARSER_ID``, ``enrich_identity_fields``.

Dependencies: ``bank_statement.context.StyleContext``.
"""

from __future__ import annotations

import re

from docmirror.plugins.bank_statement.context import StyleContext

PARSER_ID = "kv_identity"
_KV_IN_CELL_RE = re.compile(r"^([^:：]+)[:：]\s*(.+)$")


def _matches_identity_key(key: str, candidate: str) -> bool:
    """Match a leading identity label without accepting transaction prose."""
    normalized_key = re.sub(r"\s+", " ", key).strip()
    normalized_candidate = re.sub(r"\s+", " ", candidate).strip()
    if normalized_key.casefold() == normalized_candidate.casefold():
        return True
    return bool(
        re.match(
            rf"^{re.escape(normalized_candidate)}(?:\s|[/／]|[（(])",
            normalized_key,
            re.I,
        )
    )


def enrich_identity_fields(
    ctx: StyleContext,
    identity_fields: dict[str, dict],
    identity_config: tuple[tuple[str, tuple[str, ...]], ...],
) -> dict[str, dict]:
    fields = dict(identity_fields)
    parse_result = ctx.parse_result
    if not parse_result or not hasattr(parse_result, "pages"):
        return fields

    for page in getattr(parse_result, "pages", []):
        for table in getattr(page, "tables", []):
            for row in getattr(table, "rows", []):
                for cell in getattr(row, "cells", []):
                    text = getattr(cell, "text", "").strip()
                    if not text:
                        continue
                    kv = _KV_IN_CELL_RE.match(text)
                    if not kv:
                        continue
                    key, val = kv.group(1).strip(), kv.group(2).strip()
                    for field_name, candidate_keys in identity_config:
                        if field_name in fields:
                            continue
                        for ck in candidate_keys:
                            if _matches_identity_key(key, ck):
                                source_ref = {
                                    "source": "page.table_kv",
                                    "page": int(getattr(page, "page_number", 0) or 0),
                                }
                                if getattr(cell, "bbox", None):
                                    source_ref["bbox"] = list(cell.bbox)
                                fields[field_name] = {
                                    "raw_name": key,
                                    "raw_value": val,
                                    "normalized_value": val,
                                    "data_type": "string",
                                    "source": "page.table_kv",
                                    "source_refs": [source_ref],
                                    "evidence_ids": list(getattr(cell, "evidence_ids", []) or []),
                                }
                                break

    return fields
