"""Independent field/row conservation checks for the business-facing bank view."""

from __future__ import annotations

import copy
import csv
import io
from collections import defaultdict
from typing import Any

from docmirror.output.bank_business_view import EXTRACTION_FIELDS, INTERNAL_ROW_FIELDS


def evidence_delivery(evidence: dict) -> dict:
    """The existing v4 source-accounted view, before v5 presentation changes."""
    from docmirror.output.community_bundle import _community_view_from_semantic

    legacy = copy.deepcopy(evidence)
    legacy["domain"]["extensions"]["compact_output"].pop("business_view", None)
    return _community_view_from_semantic(legacy)


def assert_business_value_conservation(original: dict, cleaned: dict) -> None:
    from scripts.validate.bank_compact_exports import _first_difference

    def equal(expected: Any, actual: Any, subject: str) -> None:
        if difference := _first_difference(expected, actual):
            raise AssertionError(f"business export changed {subject}: {difference}")

    if cleaned["schema"]["version"] != "5.0.0":
        raise AssertionError("business export must declare v5")
    for key in ("document", "files"):
        equal(original[key], cleaned[key], key)
    equal(original.get("warnings") or [], cleaned.get("extraction", {}).get("warnings"), "warnings")
    expected_sections = copy.deepcopy(original["sections"])
    for section in expected_sections:
        section["items"] = [item for item in section["items"] if item["key"] not in EXTRACTION_FIELDS]
        for group in section["groups"]:
            group["items"] = [item for item in group["items"] if item["key"] not in EXTRACTION_FIELDS]
        section["groups"] = [group for group in section["groups"] if group["items"]]
    equal(expected_sections, cleaned["sections"], "scalar business facts")
    if len(original["datasets"]) != len(cleaned["datasets"]):
        raise AssertionError("business export changed dataset count")
    for before, after in zip(original["datasets"], cleaned["datasets"], strict=True):
        for key in ("id", "name", "label", "row_count", "completeness", "status", "csv", "section_id"):
            equal(before.get(key), after.get(key), f"dataset {key}")
        if len(before["rows"]) != len(after["rows"]):
            raise AssertionError("business export changed row count")
        standard = {column["key"]: column for column in before["columns"] if column["key"] not in INTERNAL_ROW_FIELDS}
        promoted = {}
        actual_columns = {}
        for column in after["columns"]:
            key = column["key"]
            if key in actual_columns or key in INTERNAL_ROW_FIELDS:
                raise AssertionError("duplicate or intermediate business column")
            actual_columns[key] = column
            if key in standard:
                equal(standard[key], column, "existing business column")
            elif "source_header" in column:
                identity = (column["source_header"], column.get("source_occurrence", 1))
                if identity in promoted:
                    raise AssertionError("source column was promoted twice")
                promoted[identity] = key
            else:
                # Uncatalogued normalized values must also be retained.
                if not any(key in row["normalized"] for row in before["rows"]):
                    raise AssertionError("new column has no business source")
        used_promotions = set()
        record_ids = []
        for before_row, after_row in zip(before["rows"], after["rows"], strict=True):
            expected_values = {key: copy.deepcopy(value) for key, value in before_row["normalized"].items()
                               if key not in INTERNAL_ROW_FIELDS}
            occurrences: dict[str, int] = defaultdict(int)
            for item in before_row["normalized"].get("additional_fields") or []:
                name = item["name"]
                occurrences[name] += 1
                identity = (name, occurrences[name])
                if identity not in promoted:
                    raise AssertionError("source business field is missing from named columns")
                key = promoted[identity]
                if key in expected_values:
                    raise AssertionError("source promotion overwrote a normalized business value")
                expected_values[key] = copy.deepcopy(item["value"])
                used_promotions.add(identity)
            equal(expected_values, after_row["normalized"], "business row values")
            if set(expected_values) - actual_columns.keys():
                raise AssertionError("business field is missing from the column catalog")
            expected_metadata = {"record_id": before_row["record_id"]}
            if (before_row.get("source") or {}).get("page_range"):
                expected_metadata["page_range"] = before_row["source"]["page_range"]
            if before_row["normalized"].get("statement_header_id") not in (None, ""):
                expected_metadata["statement_header_id"] = before_row["normalized"]["statement_header_id"]
            for key in ("confidence", "review"):
                if before_row.get(key) not in (None, "", {}):
                    expected_metadata[key] = before_row[key]
            equal(expected_metadata, after_row["extraction"], "compact extraction metadata")
            if list(after_row) != ["normalized", "extraction"]:
                raise AssertionError("business rows must put only compact extraction metadata last")
            record_ids.append(after_row["extraction"]["record_id"])
        if len(record_ids) != len(set(record_ids)):
            raise AssertionError("business export duplicated record identities")
        if set(promoted) != used_promotions:
            raise AssertionError("unused promoted business columns")
        expected_relations = copy.deepcopy(before.get("foreign_keys") or [])
        for relation in expected_relations:
            relation["columns"] = ["extraction.statement_header_id" if key == "statement_header_id" else key
                                   for key in relation.get("columns") or []]
            relation["reference_columns"] = ["extraction.record_id" if key == "record_id" else key
                                             for key in relation.get("reference_columns") or []]
        equal(expected_relations, after.get("foreign_keys") or [], "record relationships")
    if next(reversed(cleaned)) != "extraction":
        raise AssertionError("extraction summary must be last")
    by_id = {dataset["id"]: dataset for dataset in cleaned["datasets"]}
    for table in cleaned["reading"]["tables"]:
        equal([column["key"] for column in by_id[table["dataset_id"]]["columns"]], table["column_keys"], "reading columns")


def assert_business_csv_conservation(original: str, cleaned: str, dataset: dict) -> None:
    """Only absent/technical columns disappear; source-labelled columns replace the wrapper."""
    from scripts.validate.bank_compact_exports import _first_difference

    before = list(csv.DictReader(io.StringIO(original.lstrip("\ufeff"))))
    after = list(csv.DictReader(io.StringIO(cleaned.lstrip("\ufeff"))))
    if len(before) != len(after):
        raise AssertionError("business CSV changed row count")
    source_columns = {column["key"] for column in dataset["columns"] if "source_header" in column}
    for old_row, new_row in zip(before, after, strict=True):
        for key, value in old_row.items():
            if key in INTERNAL_ROW_FIELDS:
                continue
            if key not in new_row:
                if value != "":
                    raise AssertionError("business CSV dropped a populated field")
            elif _first_difference(value, new_row[key]):
                raise AssertionError(f"business CSV changed original field {key}")
        if set(new_row) - set(old_row) - source_columns:
            raise AssertionError("business CSV contains unexplained columns")


def assert_business_markdown_values(payload: dict, markdown: str) -> None:
    """Check every rendered business leaf before the extraction appendix."""
    import html

    if "\n\n## 提取说明\n\n" not in markdown:
        raise AssertionError("business Markdown is missing its trailing extraction appendix")
    body = markdown.split("\n\n## 提取说明\n\n", 1)[0]

    def check(value: Any) -> None:
        if isinstance(value, dict):
            for item in value.values():
                check(item)
        elif isinstance(value, list):
            for item in value:
                check(item)
        elif value not in (None, ""):
            expected = html.escape(str(value).lower() if isinstance(value, bool) else str(value), quote=False)
            expected = expected.replace("\\", "\\\\")
            for char in "`*_[]~|":
                expected = expected.replace(char, "\\" + char)
            expected = expected.replace("\r\n", "\n").replace("\r", "\n").replace("\n", " ↵ ")
            if expected not in body:
                raise AssertionError("a business value is absent from the Markdown body")

    for dataset in payload["datasets"]:
        for row in dataset["rows"]:
            for value in row["normalized"].values():
                check(value)
        if any(column["key"] in INTERNAL_ROW_FIELDS for column in dataset["columns"]):
            raise AssertionError("intermediate field survived into the reading catalog")
    for section in payload["sections"]:
        items = [*section["items"], *(item for group in section["groups"] for item in group["items"])]
        for item in items:
            check(item.get("value"))
            check(item.get("additional_values"))
