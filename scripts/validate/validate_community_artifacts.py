#!/usr/bin/env python3
"""Validate a persisted Community JSON/Markdown/Dataset Bundle as one contract."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from docmirror.models.schemas.registry import validate_projection_payload
from docmirror.output.markdown_renderer import MARKDOWN_PROFILE_MARKER, validate_markdown

_TOP_LEVEL_BLOCKS = {"schema", "document", "sections", "datasets", "reading", "files", "warnings"}
_RECORD_BLOCKS = {"record_id", "normalized", "canonical_raw", "raw", "source"}
_COMMUNITY_READING_MARKER = (
    '<!-- docmirror:reading-profile version="2.0" mode="enhanced" source="community-semantic" -->'
)
_AUDIT_COLUMNS = {
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
}
PAYMENT_DIRECTIONS = ("收入", "支出", "其他", "不计收支")


class _FirstTableCellParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.td = Counter()
        self.th = Counter()
        self._in_row = False
        self._capturing = False
        self._captured = False
        self._tag = ""
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        tag = tag.lower()
        if tag == "tr":
            self._in_row = True
            self._captured = False
        elif self._in_row and not self._captured and tag in {"td", "th"}:
            self._capturing = True
            self._captured = True
            self._tag = tag
            self._parts = []

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if self._capturing and tag == self._tag:
            value = "".join(self._parts)
            value = "".join(value.split())
            getattr(self, self._tag)[value] += 1
            self._capturing = False
        if tag == "tr":
            self._in_row = False
            self._capturing = False

    def handle_data(self, data: str) -> None:
        if self._capturing:
            self._parts.append(data)


def payment_direction_cells(markdown: str) -> tuple[Counter[str], Counter[str]]:
    """Return payment-direction counts in ordinary and header cells."""
    parser = _FirstTableCellParser()
    parser.feed(markdown)
    parser.close()

    lines = markdown.splitlines()
    for index, line in enumerate(lines):
        cells = _gfm_table_cells(line)
        if cells is None or _is_gfm_separator(cells):
            continue
        next_cells = _gfm_table_cells(lines[index + 1]) if index + 1 < len(lines) else None
        target = parser.th if next_cells is not None and _is_gfm_separator(next_cells) else parser.td
        target["".join(cells[0].split())] += 1
    return parser.td, parser.th


def _gfm_table_cells(line: str) -> list[str] | None:
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        return None
    cells: list[str] = []
    current: list[str] = []
    escaped = False
    for char in stripped[1:-1]:
        if char == "|" and not escaped:
            cells.append("".join(current).strip())
            current = []
            continue
        current.append(char)
        if char == "\\" and not escaped:
            escaped = True
        else:
            escaped = False
    cells.append("".join(current).strip())
    return cells


def _is_gfm_separator(cells: list[str]) -> bool:
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def _companion_path(root: Path, relative: Any, label: str, issues: list[str]) -> Path | None:
    if not isinstance(relative, str) or not relative:
        issues.append(f"{label}: missing relative path")
        return None
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root.resolve()):
        issues.append(f"{label}: path escapes artifact directory: {relative}")
        return None
    if not candidate.is_file():
        issues.append(f"{label}: file not found: {relative}")
        return None
    return candidate


def validate_community_artifacts(community_path: str | Path) -> list[str]:
    """Return all violations across Community JSON, Markdown, wide CSVs and audit CSV."""
    path = Path(community_path).resolve()
    issues: list[str] = []
    if not path.is_file():
        return [f"community: file not found: {path}"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"community: invalid JSON: {exc}"]
    if not isinstance(payload, dict):
        return ["community: top level must be an object"]

    validation = validate_projection_payload("community", payload)
    issues.extend(f"schema: {error}" for error in validation.errors)
    if set(payload) != _TOP_LEVEL_BLOCKS:
        issues.append(f"community: top-level blocks={sorted(payload)}")

    root = path.parent
    files = payload.get("files") if isinstance(payload.get("files"), dict) else {}
    semantic_path = _companion_path(root, files.get("semantic_json"), "semantic", issues)
    content_path = _companion_path(root, files.get("content_md"), "content", issues)
    enhanced_path = _companion_path(root, files.get("enhanced_reading_md"), "enhanced_reading", issues)
    audit_path = _companion_path(root, files.get("dataset_audit_csv"), "audit", issues)

    semantic: dict[str, Any] = {}
    if semantic_path is not None:
        try:
            loaded_semantic = json.loads(semantic_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            issues.append(f"semantic: invalid JSON: {exc}")
        else:
            if not isinstance(loaded_semantic, dict):
                issues.append("semantic: top level must be an object")
            else:
                semantic = loaded_semantic
                semantic_validation = validate_projection_payload("community_semantic", semantic)
                issues.extend(f"semantic schema: {error}" for error in semantic_validation.errors)
                semantic_structure = (
                    semantic.get("structure") if isinstance(semantic.get("structure"), dict) else {}
                )
                if semantic.get("document") != payload.get("document"):
                    issues.append("semantic: document diverges from Community JSON")
                if semantic_structure.get("sections") is not None:
                    projected_sections = [
                        {
                            key: value
                            for key, value in section.items()
                            if key not in {"block_refs", "source_table_refs"} and not key.startswith("_")
                        }
                        for section in semantic_structure.get("sections") or []
                        if isinstance(section, dict)
                    ]
                    if projected_sections != payload.get("sections"):
                        issues.append("semantic: sections diverge from Community JSON")
                if semantic.get("datasets") != payload.get("datasets"):
                    issues.append("semantic: datasets diverge from Community JSON")
                if semantic.get("reading") != payload.get("reading"):
                    issues.append("semantic: reading plan diverges from Community JSON")
                semantic_dataset_records = {
                    (str(dataset.get("id") or ""), str(row.get("record_id") or ""))
                    for dataset in semantic.get("datasets") or []
                    if isinstance(dataset, dict)
                    for row in dataset.get("rows") or []
                    if isinstance(row, dict)
                }
                semantic_block_ids = {
                    str(block.get("id") or "")
                    for block in semantic_structure.get("blocks") or []
                    if isinstance(block, dict)
                }
                semantic_table_ids = {
                    str(table.get("id") or "")
                    for table in semantic_structure.get("source_tables") or []
                    if isinstance(table, dict)
                }
                bound_records: list[tuple[str, str]] = []
                for index, binding in enumerate(semantic.get("bindings") or []):
                    if not isinstance(binding, dict):
                        issues.append(f"semantic.bindings[{index}]: must be an object")
                        continue
                    record_key = (
                        str(binding.get("dataset_id") or ""),
                        str(binding.get("record_id") or ""),
                    )
                    bound_records.append(record_key)
                    if record_key not in semantic_dataset_records:
                        issues.append(f"semantic.bindings[{index}]: unknown dataset/record={record_key}")
                    unknown_blocks = set(binding.get("source_block_refs") or []) - semantic_block_ids
                    if unknown_blocks:
                        issues.append(
                            f"semantic.bindings[{index}]: unknown source_block_refs={sorted(unknown_blocks)}"
                        )
                    unknown_tables = set(binding.get("source_table_refs") or []) - semantic_table_ids
                    if unknown_tables:
                        issues.append(
                            f"semantic.bindings[{index}]: unknown source_table_refs={sorted(unknown_tables)}"
                        )
                if len(bound_records) != len(set(bound_records)):
                    issues.append("semantic.bindings: duplicate dataset/record binding")
                missing_bindings = semantic_dataset_records - set(bound_records)
                if missing_bindings:
                    issues.append(f"semantic.bindings: {len(missing_bindings)} records are unbound")

    datasets = payload.get("datasets") if isinstance(payload.get("datasets"), list) else []
    dataset_ids: set[str] = set()
    dataset_record_ids: dict[str, set[str]] = {}
    expected_audited_ids: dict[str, set[str]] = {}
    dataset_row_counts: dict[str, int] = {}
    dataset_column_keys: dict[str, set[str]] = {}
    for index, dataset in enumerate(datasets):
        if not isinstance(dataset, dict):
            issues.append(f"dataset[{index}]: must be an object")
            continue
        dataset_id = str(dataset.get("id") or f"dataset[{index}]")
        if dataset_id in dataset_ids:
            issues.append(f"{dataset_id}: duplicate dataset id")
        dataset_ids.add(dataset_id)

        rows = dataset.get("rows") if isinstance(dataset.get("rows"), list) else []
        row_count = dataset.get("row_count")
        completeness = dataset.get("completeness") if isinstance(dataset.get("completeness"), dict) else {}
        emitted = completeness.get("emitted_row_count")
        expected = completeness.get("expected_row_count")
        omitted = completeness.get("omitted_row_count")
        if row_count != len(rows) or emitted != len(rows):
            issues.append(f"{dataset_id}: JSON count mismatch row_count={row_count} emitted={emitted} rows={len(rows)}")
        if isinstance(expected, int) and isinstance(emitted, int):
            expected_omitted = max(0, expected - emitted)
            if omitted != expected_omitted:
                issues.append(f"{dataset_id}: omitted_row_count={omitted}, expected={expected_omitted}")
            verified = completeness.get("verified")
            if verified is not (expected == emitted):
                issues.append(f"{dataset_id}: completeness.verified contradicts expected/emitted counts")

        record_ids: list[str] = []
        audited_ids: set[str] = set()
        for row_index, row in enumerate(rows):
            if not isinstance(row, dict):
                issues.append(f"{dataset_id}.rows[{row_index}]: must be an object")
                continue
            missing = _RECORD_BLOCKS - set(row)
            if missing:
                issues.append(f"{dataset_id}.rows[{row_index}]: missing {sorted(missing)}")
            record_id = str(row.get("record_id") or "")
            record_ids.append(record_id)
            if not record_id:
                issues.append(f"{dataset_id}.rows[{row_index}]: empty record_id")
            for block in ("normalized", "canonical_raw", "raw", "source"):
                if not isinstance(row.get(block), dict):
                    issues.append(f"{dataset_id}.rows[{row_index}].{block}: must be an object")
            normalized = row.get("normalized") if isinstance(row.get("normalized"), dict) else {}
            if any(value not in (None, "", [], {}) for value in normalized.values()) and record_id:
                audited_ids.add(record_id)
        if len(record_ids) != len(set(record_ids)):
            issues.append(f"{dataset_id}: duplicate record_id")
        dataset_record_ids[dataset_id] = set(record_ids)
        expected_audited_ids[dataset_id] = audited_ids
        dataset_row_counts[dataset_id] = len(rows)
        dataset_column_keys[dataset_id] = {
            str(column.get("key") or "")
            for column in dataset.get("columns") or []
            if isinstance(column, dict) and column.get("key")
        }

        csv_path = _companion_path(root, dataset.get("csv"), f"{dataset_id}.csv", issues)
        if csv_path is None:
            continue
        with csv_path.open(encoding="utf-8-sig", newline="") as stream:
            csv_rows = list(csv.DictReader(stream))
        csv_ids = [str(row.get("record_id") or "") for row in csv_rows]
        if len(csv_rows) != len(rows):
            issues.append(f"{dataset_id}: CSV rows={len(csv_rows)}, JSON rows={len(rows)}")
        if csv_ids != record_ids:
            issues.append(f"{dataset_id}: ordered record_id mismatch between JSON and CSV")

    reading = payload.get("reading") if isinstance(payload.get("reading"), dict) else {}
    reading_tables = reading.get("tables") if isinstance(reading.get("tables"), list) else []
    table_dataset_ids: list[str] = []
    for index, table in enumerate(reading_tables):
        if not isinstance(table, dict):
            issues.append(f"reading.tables[{index}]: must be an object")
            continue
        dataset_id = str(table.get("dataset_id") or "")
        table_dataset_ids.append(dataset_id)
        if dataset_id not in dataset_ids:
            issues.append(f"reading.tables[{index}]: unknown dataset_id={dataset_id}")
            continue
        if table.get("row_count") != dataset_row_counts[dataset_id]:
            issues.append(
                f"reading.tables[{index}]: row_count={table.get('row_count')}, "
                f"JSON rows={dataset_row_counts[dataset_id]}"
            )
        column_keys = [str(key) for key in table.get("column_keys") or []]
        unknown_keys = set(column_keys) - dataset_column_keys[dataset_id]
        if unknown_keys:
            issues.append(f"reading.tables[{index}]: unknown column_keys={sorted(unknown_keys)}")
    if len(table_dataset_ids) != len(set(table_dataset_ids)):
        issues.append("reading.tables: duplicate dataset_id")
    missing_table_ids = dataset_ids - set(table_dataset_ids)
    if missing_table_ids:
        issues.append(f"reading.tables: missing datasets={sorted(missing_table_ids)}")

    sections = payload.get("sections") if isinstance(payload.get("sections"), list) else []
    section_ids = {
        str(section.get("id") or "")
        for section in sections
        if isinstance(section, dict) and section.get("id")
    }
    document = payload.get("document") if isinstance(payload.get("document"), dict) else {}
    document_id = str(document.get("id") or "")
    flow = reading.get("document_flow") if isinstance(reading.get("document_flow"), list) else []
    flow_orders: list[int] = []
    flow_dataset_ids: list[str] = []
    for index, entry in enumerate(flow):
        if not isinstance(entry, dict):
            issues.append(f"reading.document_flow[{index}]: must be an object")
            continue
        order = entry.get("order")
        if isinstance(order, int):
            flow_orders.append(order)
        kind = str(entry.get("kind") or "")
        ref_id = str(entry.get("ref_id") or "")
        if kind == "document" and ref_id != document_id:
            issues.append(f"reading.document_flow[{index}]: unknown document ref_id={ref_id}")
        elif kind == "section" and ref_id not in section_ids:
            issues.append(f"reading.document_flow[{index}]: unknown section ref_id={ref_id}")
        elif kind == "dataset":
            flow_dataset_ids.append(ref_id)
            if ref_id not in dataset_ids:
                issues.append(f"reading.document_flow[{index}]: unknown dataset ref_id={ref_id}")
    if flow_orders != list(range(1, len(flow) + 1)):
        issues.append("reading.document_flow: order must be contiguous and match array order")
    if len(flow_dataset_ids) != len(set(flow_dataset_ids)):
        issues.append("reading.document_flow: duplicate dataset ref_id")
    missing_flow_ids = dataset_ids - set(flow_dataset_ids)
    if missing_flow_ids:
        issues.append(f"reading.document_flow: missing datasets={sorted(missing_flow_ids)}")

    if content_path is not None:
        markdown = content_path.read_text(encoding="utf-8")
        if not markdown.strip():
            issues.append("content: Markdown is empty")
        if MARKDOWN_PROFILE_MARKER not in markdown:
            issues.append("content: DMP profile marker missing")
        issues.extend(f"content: {issue}" for issue in validate_markdown(markdown))

    if enhanced_path is not None:
        enhanced = enhanced_path.read_text(encoding="utf-8")
        if not enhanced.strip():
            issues.append("enhanced_reading: Markdown is empty")
        if MARKDOWN_PROFILE_MARKER not in enhanced:
            issues.append("enhanced_reading: DMP profile marker missing")
        if _COMMUNITY_READING_MARKER not in enhanced:
            issues.append("enhanced_reading: Community reading profile marker missing")
        issues.extend(f"enhanced_reading: {issue}" for issue in validate_markdown(enhanced))

    if audit_path is not None:
        with audit_path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            audit_rows = list(reader)
            audit_columns = set(reader.fieldnames or [])
        if audit_columns != _AUDIT_COLUMNS:
            issues.append(f"audit: columns={sorted(audit_columns)}")
        seen_audited_ids: dict[str, set[str]] = {dataset_id: set() for dataset_id in dataset_ids}
        for row_index, row in enumerate(audit_rows):
            dataset_id = str(row.get("dataset_id") or "")
            record_id = str(row.get("record_id") or "")
            if dataset_id not in dataset_record_ids:
                issues.append(f"audit[{row_index}]: unknown dataset_id={dataset_id}")
                continue
            if record_id not in dataset_record_ids[dataset_id]:
                issues.append(f"audit[{row_index}]: unknown record_id={record_id}")
            seen_audited_ids[dataset_id].add(record_id)
        for dataset_id, expected_ids in expected_audited_ids.items():
            missing_ids = expected_ids - seen_audited_ids.get(dataset_id, set())
            if missing_ids:
                issues.append(f"{dataset_id}: {len(missing_ids)} records missing from audit CSV")

    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("community_json", type=Path, help="Path to <file_id>_community.json")
    parser.add_argument(
        "--payment-markdown-parity",
        action="store_true",
        help="Require every payment JSON record to appear as a Markdown data row, not a header row.",
    )
    args = parser.parse_args()
    issues = validate_community_artifacts(args.community_json)
    if args.payment_markdown_parity and not issues:
        payload = json.loads(args.community_json.read_text(encoding="utf-8"))
        markdown_path = args.community_json.parent / payload["files"]["content_md"]
        td, th = payment_direction_cells(markdown_path.read_text(encoding="utf-8"))
        transaction_rows = sum(td[direction] for direction in PAYMENT_DIRECTIONS)
        header_rows = sum(th[direction] for direction in PAYMENT_DIRECTIONS)
        payment_datasets = [dataset for dataset in payload["datasets"] if dataset.get("type") == "transaction"]
        expected_rows = sum(int(dataset.get("row_count") or 0) for dataset in payment_datasets)
        if transaction_rows != expected_rows:
            issues.append(f"content: payment data rows={transaction_rows}, JSON rows={expected_rows}")
        if header_rows:
            issues.append(f"content: {header_rows} payment records rendered as header cells")
    if issues:
        for issue in issues:
            print(f"ERROR {issue}")
        return 1
    print(f"OK {args.community_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
