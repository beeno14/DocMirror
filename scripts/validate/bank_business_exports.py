"""Independent field/row conservation checks for the business-facing bank view."""

from __future__ import annotations

import copy
import csv
import io
import re
import unicodedata
from collections import defaultdict
from typing import Any

from docmirror.output.bank_business_view import INTERNAL_ROW_FIELDS

# Deliberately separate from the production filter: the auditor permits only
# these named page-local header fields to disappear, never arbitrary amounts.
_PAGE_ONLY_LABELS = {
    "本页支出笔数", "本页收入笔数", "本页借方笔数", "本页贷方笔数", "本页交易笔数",
    "本页支出算数合计", "本页支出算术合计", "本页收入算数合计", "本页收入算术合计",
    "本页支出合计", "本页收入合计", "本页借方合计", "本页贷方合计",
    "本页支出金额合计", "本页收入金额合计", "本页借方金额合计", "本页贷方金额合计",
    "pagedebittotal", "pagecredittotal", "pageexpensetotal", "pageincometotal",
    "pagedebitcount", "pagecreditcount", "pageexpensecount", "pageincomecount",
    "pagetransactioncount", "页码", "页号", "pagenumber", "source_header_page_label",
}

# Independent presentation expectations; never reuse the renderer's dictionary
# to validate its own translations. Source variants must bypass these mappings.
_MARKDOWN_ENUM_LABELS = {
    "direction": {"income": "收入", "expense": "支出"},
    "currency": {"CNY": "人民币", "RMB": "人民币", "USD": "美元", "HKD": "港元", "EUR": "欧元", "JPY": "日元"},
    "direction_filter": {"income": "收入", "expense": "支出", "all": "全部"},
    "sort_order": {"asc": "升序", "ascending": "升序", "desc": "降序", "descending": "降序"},
    "document_type": {"bank_statement": "银行流水", "bank_reconciliation": "银行对账单"},
}


def _allowed_omission(dataset_name: str, key: str, descriptor: dict) -> bool:
    if key in INTERNAL_ROW_FIELDS:
        return True
    if dataset_name != "statement_header":
        return False
    names = (key, descriptor.get("source_header", ""), descriptor.get("label", ""))
    return any(re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(name))).strip(":").casefold()
               in _PAGE_ONLY_LABELS for name in names)


def _business_descriptor(column: dict) -> dict:
    return {key: value for key, value in column.items() if key not in {"raw_available", "evidence_available"}}


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
        section["items"] = [item for item in section["items"]
                            if not _allowed_omission("statement_header", item["key"], item)]
        for group in section["groups"]:
            group["items"] = [item for item in group["items"]
                              if not _allowed_omission("statement_header", item["key"], item)]
        section["groups"] = [group for group in section["groups"] if group["items"]]
    equal(expected_sections, cleaned["sections"], "scalar business facts")
    if len(original["datasets"]) != len(cleaned["datasets"]):
        raise AssertionError("business export changed dataset count")
    for before, after in zip(original["datasets"], cleaned["datasets"], strict=True):
        for key in ("id", "name", "label", "row_count", "completeness", "status", "csv", "section_id"):
            equal(before.get(key), after.get(key), f"dataset {key}")
        if len(before["rows"]) != len(after["rows"]):
            raise AssertionError("business export changed row count")
        if {"storage_role", "record_path"}.intersection(after):
            raise AssertionError("internal storage metadata survived into the business dataset")
        declared = {column["key"]: column for column in before["columns"]}
        standard = {key: _business_descriptor(column) for key, column in declared.items()
                    if not _allowed_omission(before["name"], key, column)}
        promoted = {}
        actual_columns = {}
        for column in after["columns"]:
            key = column["key"]
            if key in actual_columns or _allowed_omission(after["name"], key, column):
                raise AssertionError("duplicate or intermediate business column")
            if {"raw_available", "evidence_available"}.intersection(column):
                raise AssertionError("internal availability flags survived into the business catalog")
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
                               if not _allowed_omission(before["name"], key, declared.get(key, {}))}
            occurrences: dict[str, int] = defaultdict(int)
            for item in before_row["normalized"].get("additional_fields") or []:
                name = item["name"]
                occurrences[name] += 1
                identity = (name, occurrences[name])
                # A literal source label matching a reserved implementation key
                # is still a business field. Only explicit page summaries in
                # statement headers may be omitted from source promotions.
                if before["name"] == "statement_header" and _allowed_omission(
                    "statement_header", "", {"source_header": name}
                ):
                    continue
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


def assert_business_csv_conservation(
    original: str, cleaned: str, dataset: dict, *, original_dataset: dict | None = None
) -> None:
    """Only absent/technical columns disappear; source-labelled columns replace the wrapper."""
    from scripts.validate.bank_compact_exports import _first_difference

    before = list(csv.DictReader(io.StringIO(original.lstrip("\ufeff"))))
    after = list(csv.DictReader(io.StringIO(cleaned.lstrip("\ufeff"))))
    if len(before) != len(after):
        raise AssertionError("business CSV changed row count")
    source_columns = {column["key"] for column in dataset["columns"] if "source_header" in column}
    declared = {column["key"]: column for column in (original_dataset or dataset)["columns"]}
    for old_row, new_row in zip(before, after, strict=True):
        for key, value in old_row.items():
            if _allowed_omission(dataset["name"], key, declared.get(key, {})):
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

    def check(value: Any, enum_key: str = "") -> None:
        if isinstance(value, dict):
            if "value" in value and set(value) <= {"page", "value"}:
                check(value.get("page"))
                check(value["value"], enum_key)
            else:
                for item in value.values():
                    check(item)
        elif isinstance(value, list):
            for item in value:
                check(item, enum_key)
        elif value not in (None, ""):
            value = _MARKDOWN_ENUM_LABELS.get(enum_key, {}).get(value, value)
            expected = html.escape(str(value).lower() if isinstance(value, bool) else str(value), quote=False)
            expected = expected.replace("\\", "\\\\")
            for char in "`*_[]~|":
                expected = expected.replace(char, "\\" + char)
            expected = expected.replace("\r\n", "\n").replace("\r", "\n").replace("\n", " ↵ ")
            if expected not in body:
                raise AssertionError("a business value is absent from the Markdown body")

    for dataset in payload["datasets"]:
        columns = {column["key"]: column for column in dataset["columns"]}
        for row in dataset["rows"]:
            for key, value in row["normalized"].items():
                descriptor = columns.get(key, {})
                check(value, "" if "source_header" in descriptor else descriptor.get("enum_ref") or key)
        if any(_allowed_omission(dataset["name"], column["key"], column) for column in dataset["columns"]):
            raise AssertionError("intermediate field survived into the reading catalog")
    for section in payload["sections"]:
        items = [*section["items"], *(item for group in section["groups"] for item in group["items"])]
        for item in items:
            enum_key = "" if "source_header" in item else item.get("enum_ref") or item["key"]
            check(item.get("value"), enum_key)
            check(item.get("additional_values"), enum_key)

    # Presence alone could miss swapped directions (both Chinese words might
    # occur elsewhere). Check the actual direction cells in transaction order.
    tables = [block.splitlines() for block in body.split("\n\n") if block.startswith("| ")]
    for dataset in payload["datasets"]:
        column = next((column for column in dataset["columns"] if column["key"] == "direction"), None)
        if dataset["name"] == "statement_header" or column is None:
            continue
        labels = {column.get("label"), "收支方向", "direction"}
        for table in tables:
            headings = [cell.strip() for cell in re.split(r"(?<!\\)\|", table[0])[1:-1]]
            if not any(heading in labels for heading in headings):
                continue
            index = next(i for i, heading in enumerate(headings) if heading in labels)
            actual = [re.split(r"(?<!\\)\|", line)[1:-1][index].strip() for line in table[2:]]
            translations = {} if "source_header" in column else _MARKDOWN_ENUM_LABELS["direction"]
            expected = [translations.get(row["normalized"].get("direction"), row["normalized"].get("direction"))
                        for row in dataset["rows"]]
            # Known enums have no escaping; unknown/source values were checked
            # literally above and must not be silently assigned a direction.
            if len(actual) != len(expected) or any(value in {"收入", "支出"} and value != shown
                                                    for value, shown in zip(expected, actual)):
                raise AssertionError("Markdown transaction directions differ from business rows")
            tables.remove(table)
            break
        else:
            raise AssertionError("Markdown transaction direction column is absent")
