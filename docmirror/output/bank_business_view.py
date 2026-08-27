"""Business-facing digital-bank delivery; never extracts or repairs a value.

The full semantic result remains the evidence source. This view names existing
source-labelled values, and separates small delivery metadata from business data.
"""

from __future__ import annotations

import copy
import html
import re
from collections import defaultdict
from typing import Any

from docmirror.output.normalized_records import _label, _same

BUSINESS_VIEW_VERSION = "5.0.0"

# Explicit producer metadata, not a heuristic applied to source column labels.
EXTRACTION_FIELDS = frozenset({
    "style_id", "style_confidence", "parser_chain", "institution_hint", "secondary_styles",
    "reconstruction_source", "expected_primary_rows", "extracted_rows", "coverage_ratio",
    "institution_authority", "pipe_parse_failed", "canonical_expected", "canonical_extracted",
    "canonical_ratio", "extract_status", "blo_tables_parsed", "blo_tables_skipped", "extraction_route",
    "document_scene_refined", "layout_profile_id_refined", "layout_profile_refine_confidence",
    "classification_source", "doc_type_hint_source", "user_doc_type_hint", "user_doc_type_hint_strength",
    "mirror_expected_data_rows", "mirror_ltqg_enabled", "mirror_ltqg_export_tables",
    "mirror_ltqg_passed_tables", "mirror_ltqg_raw_max_rows", "mirror_ltqg_skipped_tables",
    "mirror_quarantined_logical_count", "mirror_quarantined_physical_count",
    "counterparty_status", "source_header_page_label",
})
INTERNAL_ROW_FIELDS = EXTRACTION_FIELDS | {"statement_header_id", "additional_fields"}
INTERNAL_COLUMN_FIELDS = frozenset({"raw_available", "evidence_available"})
INTERNAL_DATASET_FIELDS = frozenset({"storage_role", "record_path"})

# Page-local header summaries are audit evidence, not statement totals. Match
# complete labels only, and only in statement headers; never inspect cell text
# or discard transaction income/expense columns because of their names.
PAGE_SUMMARY_LABELS = frozenset(_label(name) for name in (
    "本页支出笔数", "本页收入笔数", "本页借方笔数", "本页贷方笔数", "本页交易笔数",
    "本页支出算数合计", "本页支出算术合计", "本页收入算数合计", "本页收入算术合计",
    "本页支出合计", "本页收入合计", "本页借方合计", "本页贷方合计",
    "本页支出金额合计", "本页收入金额合计", "本页借方金额合计", "本页贷方金额合计",
    "Page Debit Total", "Page Credit Total", "Page Expense Total", "Page Income Total",
    "Page Debit Count", "Page Credit Count", "Page Expense Count", "Page Income Count",
    "Page Transaction Count", "页码", "页号", "Page Number", "source_header_page_label",
))
CONTEXT_FIELDS = frozenset({
    "account_holder", "own_account", "bank_name", "statement_title", "currency",
    "query_period", "period_start", "period_end", "print_date", "document_date",
})
TRANSACTION_ORDER = (
    "sequence_no", "date", "timestamp", "direction", "amount", "balance", "currency",
    "counter_party", "counter_account", "counter_bank_name", "summary", "purpose", "note",
)


def is_business_view(payload: dict[str, Any]) -> bool:
    return (payload.get("schema") or {}).get("version") == BUSINESS_VIEW_VERSION


def _source_entries(row: dict[str, Any]):
    occurrences: dict[str, int] = defaultdict(int)
    for item in (row.get("normalized") or {}).get("additional_fields") or []:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str) or "value" not in item:
            raise ValueError("invalid source-labelled business field")
        name = item["name"]
        occurrences[name] += 1
        yield (name, occurrences[name]), item


def _business_columns(dataset: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[tuple[str, int], str]]:
    rows = dataset.get("rows") or []
    declared = {column["key"]: column for column in dataset.get("columns") or []}
    active = {key for row in rows for key in row["normalized"] if key not in INTERNAL_ROW_FIELDS}
    columns = [copy.deepcopy(column) for key, column in declared.items() if key in active]
    # A source value is never discarded just because its key was not catalogued.
    for key in sorted(active - declared.keys()):
        columns.append({"key": key, "label": key, "type": "json", "nullable": True,
                        "raw_available": False, "evidence_available": False})
    used_keys = set(declared) | active | INTERNAL_ROW_FIELDS | {
        "record_id", "extraction", "normalized", "source", "_page_start", "_page_end",
    }
    used_labels = {str(column.get("label") or column["key"]) for column in columns}
    roles: dict[tuple[str, int], set[str]] = {}
    for row in rows:
        for identity, item in _source_entries(row):
            roles.setdefault(identity, set())
            if item.get("field"):
                roles[identity].add(str(item["field"]))
    promoted = {}
    for (name, occurrence), fields in roles.items():
        canonical_field = next(iter(fields)) if len(fields) == 1 else ""
        descriptor = declared.get(canonical_field, {})
        heading = re.sub(r"\s+", " ", name).strip().strip(":：")
        synthetic_label = name == canonical_field or name == "unlabelled_header_period"
        label = str(descriptor.get("label") or heading or "未命名字段") if synthetic_label else heading or "未命名字段"
        if synthetic_label or label in used_labels:
            label += "（原文）"
        base_label = label
        number = 2
        while label in used_labels:
            label = f"{base_label}（{number}）"
            number += 1
        used_labels.add(label)
        base_key = f"{canonical_field}_original" if canonical_field else heading or "未命名字段"
        # Source headings become CSV column keys as well as JSON keys.
        # Keep the original label, but never emit a formula as a CSV header.
        if base_key.startswith(("=", "+", "-", "@")):
            base_key = "field_" + base_key
        key = base_key
        number = 2
        while key in used_keys:
            key = f"{base_key}_{number}"
            number += 1
        used_keys.add(key)
        promoted[(name, occurrence)] = key
        column = {"key": key, "label": label, "type": "json", "nullable": True,
                  "raw_available": False, "evidence_available": True, "source_header": name,
                  "source_occurrence": occurrence}
        if canonical_field:
            column["canonical_field"] = canonical_field
        columns.append(column)
    return columns, promoted


def compact_row_extraction(row: dict[str, Any]) -> dict[str, Any]:
    source = row.get("source") or {}
    metadata: dict[str, Any] = {"record_id": row["record_id"]}
    if source.get("page_range"):
        metadata["page_range"] = copy.deepcopy(source["page_range"])
    header_id = (row.get("normalized") or {}).get("statement_header_id")
    if header_id not in (None, ""):
        metadata["statement_header_id"] = copy.deepcopy(header_id)
    if row.get("confidence") not in (None, ""):
        metadata["confidence"] = copy.deepcopy(row["confidence"])
    if row.get("review") not in (None, "", {}):
        metadata["review"] = copy.deepcopy(row["review"])
    return metadata


def _hidden_field(dataset_name: str, key: str, descriptor: dict[str, Any]) -> bool:
    if key in INTERNAL_ROW_FIELDS:
        return True
    return dataset_name == "statement_header" and any(
        _label(name) in PAGE_SUMMARY_LABELS
        for name in (key, descriptor.get("source_header", ""), descriptor.get("label", ""))
    )


def _clean_business_view(result: dict[str, Any]) -> dict[str, Any]:
    """Clean a fresh v5 view, including previously saved v5 deliveries."""
    for dataset in result.get("datasets") or []:
        name = dataset.get("name", "")
        declared = {column["key"]: column for column in dataset.get("columns") or []}
        hidden = {key for key, column in declared.items() if _hidden_field(name, key, column)}
        hidden.update(key for row in dataset.get("rows") or [] for key in row["normalized"]
                      if _hidden_field(name, key, declared.get(key, {})))
        dataset["columns"] = [
            {key: value for key, value in column.items() if key not in INTERNAL_COLUMN_FIELDS}
            for column in declared.values() if column["key"] not in hidden
        ]
        for row in dataset.get("rows") or []:
            row["normalized"] = {key: value for key, value in row["normalized"].items() if key not in hidden}
        for key in INTERNAL_DATASET_FIELDS:
            dataset.pop(key, None)
    for section in result.get("sections") or []:
        section["items"] = [item for item in section.get("items") or []
                            if not _hidden_field("statement_header", item["key"], item)]
        for group in section.get("groups") or []:
            group["items"] = [item for item in group.get("items") or []
                              if not _hidden_field("statement_header", item["key"], item)]
        section["groups"] = [group for group in section.get("groups") or [] if group["items"]]
    by_id = {dataset["id"]: dataset for dataset in result.get("datasets") or []}
    for table in result.get("reading", {}).get("tables", []):
        table["column_keys"] = [column["key"] for column in by_id[table["dataset_id"]]["columns"]]
    return result


def business_view(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a fresh v5 delivery projection of an evidence-accounted v4 view."""
    if (payload.get("schema") or {}).get("domain") != "bank_statement":
        raise ValueError("business bank view must not be applied to another provider")
    if is_business_view(payload):
        return _clean_business_view(copy.deepcopy(payload))
    if (payload.get("schema") or {}).get("version") != "4.0.0":
        raise ValueError("business bank view requires an evidence-accounted normalized v4 source")
    result = copy.deepcopy(payload)
    result["schema"]["version"] = BUSINESS_VIEW_VERSION
    for dataset in result.get("datasets") or []:
        columns, promoted = _business_columns(dataset)
        rows = []
        for row in dataset.get("rows") or []:
            normalized = {key: copy.deepcopy(value) for key, value in row["normalized"].items()
                          if key not in INTERNAL_ROW_FIELDS}
            for identity, item in _source_entries(row):
                normalized[promoted[identity]] = copy.deepcopy(item["value"])
            rows.append({"normalized": normalized, "extraction": compact_row_extraction(row)})
        dataset["rows"] = rows
        dataset["columns"] = columns
        dataset["primary_key"] = "extraction.record_id"
        dataset.pop("omitted_normalized_fields", None)
        for relation in dataset.get("foreign_keys") or []:
            relation["columns"] = ["extraction.statement_header_id" if key == "statement_header_id" else key
                                   for key in relation.get("columns") or []]
            relation["reference_columns"] = ["extraction.record_id" if key == "record_id" else key
                                             for key in relation.get("reference_columns") or []]
    result["reading"]["privacy_mode"] = "full"
    result["reading"]["presentation"] = "bank_business"
    result["extraction"] = {"route": "digital", "warnings": result.pop("warnings", [])}
    return _clean_business_view(result)


def restore_business_records(payload: dict[str, Any]) -> dict[str, Any]:
    """Restore renderer-compatible identities, not deliberately withheld evidence."""
    if not is_business_view(payload):
        return copy.deepcopy(payload)
    result = business_view(payload)
    for dataset in result.get("datasets") or []:
        dataset["primary_key"] = "record_id"
        for row in dataset.get("rows") or []:
            metadata = row.pop("extraction")
            row["record_id"] = metadata["record_id"]
            row["source"] = {"page_range": copy.deepcopy(metadata["page_range"])} if "page_range" in metadata else {}
            if "statement_header_id" in metadata:
                row["normalized"]["statement_header_id"] = metadata["statement_header_id"]
            for key in ("confidence", "review"):
                if key in metadata:
                    row[key] = copy.deepcopy(metadata[key])
        for relation in dataset.get("foreign_keys") or []:
            relation["columns"] = [key.removeprefix("extraction.") for key in relation.get("columns") or []]
            relation["reference_columns"] = [key.removeprefix("extraction.") for key in relation.get("reference_columns") or []]
    result["warnings"] = result.pop("extraction", {}).get("warnings") or []
    return result


def _business_text(value: Any) -> str:
    """Display source text literally in HTML-free Markdown, without masking it."""
    if value is None:
        return ""
    text = str(value).lower() if isinstance(value, bool) else str(value)
    text = html.escape(text, quote=False).replace("\\", "\\\\")
    text = re.sub(r"([`*_\[\]~|])", r"\\\1", text)
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", " ↵ ")


def render_business_markdown(payload: dict[str, Any]) -> str:
    """Readable business tables plus a short, trailing extraction appendix."""

    def shown(value: Any) -> str:
        if isinstance(value, list):
            return "；".join(shown(item) for item in value)
        if isinstance(value, dict):
            if "value" in value and set(value) <= {"page", "value"}:
                prefix = f"第 {_business_text(value['page'])} 页：" if "page" in value else ""
                return prefix + shown(value["value"])
            return "；".join(f"{_business_text(key)}：{shown(item)}" for key, item in value.items())
        return _business_text(value)

    document = payload.get("document") or {}
    parts = ['<!-- docmirror:markdown-profile version="1.0" -->',
             '<!-- docmirror:reading-profile version="2.0" mode="enhanced" source="community-semantic" -->',
             f"# {shown(document.get('title') or '银行流水')}"]
    # Scalar business facts remain visible, including ones not repeated in datasets.
    scalar_items = []
    for section in payload.get("sections") or []:
        scalar_items.extend(section.get("items") or [])
        for group in section.get("groups") or []:
            scalar_items.extend(group.get("items") or [])
    for item in scalar_items:
        value = item.get("value")
        values = [value, *(item.get("additional_values") or [])]
        if any(value not in (None, "") for value in values):
            parts.append(f"**{shown(item.get('label') or item['key'])}:** " + " · ".join(shown(v) for v in values))
    for dataset in payload.get("datasets") or []:
        columns = {column["key"]: column for column in dataset.get("columns") or []}
        rows = dataset.get("rows") or []
        if dataset["name"] == "statement_header":
            parts.append("## 账户信息")
            for index, row in enumerate(rows, 1):
                if len(rows) > 1:
                    parts.append(f"### 账户 {index}")
                lines = ["| 项目 | 内容 |", "| --- | --- |"]
                lines.extend(f"| {shown(column.get('label') or key)} | {shown(row['normalized'][key])} |"
                             for key, column in columns.items() if key in row["normalized"])
                parts.append("\n".join(lines))
            continue
        parts.append(f"## {shown(dataset.get('label') or dataset['name'])}")
        keys = [key for key in TRANSACTION_ORDER if key in columns]
        keys.extend(key for key in columns if key not in keys)
        for key in list(keys):
            column = columns[key]
            context_key = column.get("canonical_field") or key
            if rows and len(keys) > 1 and context_key in CONTEXT_FIELDS:
                value = rows[0]["normalized"].get(key)
                if value not in (None, "") and all(key in row["normalized"] and _same(value, row["normalized"][key]) for row in rows):
                    parts.append(f"**{shown(column.get('label') or key)}:** {shown(value)}")
                    keys.remove(key)
        if keys:
            lines = ["| " + " | ".join(shown(columns[key].get("label") or key) for key in keys) + " |",
                     "| " + " | ".join("---" for _ in keys) + " |"]
            lines.extend("| " + " | ".join(shown(row["normalized"].get(key)) for key in keys) + " |" for row in rows)
            parts.append("\n".join(lines))
        if not rows:
            parts.append("_无交易记录。_")
    parts.append("## 提取说明")
    parts.append(f"来源：{shown((document.get('source_file') or {}).get('name') or '数字 PDF')}；页数：{shown(document.get('page_count'))}。")
    for dataset in payload.get("datasets") or []:
        status = "已核验" if (dataset.get("completeness") or {}).get("verified") is True else "尚未核验"
        parts.append(f"{shown(dataset.get('label') or dataset['name'])}：{dataset['row_count']} 条；完整性：{status}。")
    warnings = (payload.get("extraction") or {}).get("warnings") or []
    seen = set()
    for warning in warnings:
        signature = (warning.get("code"), warning.get("message"))
        if signature not in seen:
            seen.add(signature)
            parts.append(f"提示（{shown(warning.get('code'))}）：{shown(warning.get('message'))}")
    return "\n\n".join(parts).rstrip() + "\n"
