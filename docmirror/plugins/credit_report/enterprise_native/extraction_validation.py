# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schema-owned extraction failure protocol for digital enterprise reports."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable

from docmirror.plugins.credit_report.enterprise_native.ir import CanonicalEnterpriseDocumentIR

_SPACE = re.compile(r"\s+")
_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?")
_DATE_DIGITS = re.compile(r"\d")
_LABEL_SUFFIX = re.compile(r"^[：:]+")
_MISSING_MARKERS = frozenset(
    {
        "",
        "-",
        "--",
        "\u2014",
        "\uff0d",
        "/",
        "\u4e0d\u8be6",
        "\u672a\u77e5",
    }
)
_TECHNICAL_FIELDS = frozenset(
    {
        "sequence",
        "source_page",
        "source_page_end",
        "source_table_id",
        "page",
        "confidence",
        "is_total",
        "account_state",
        "activation_state",
        "payoff_state",
        "current_overdue",
        "revolving_flag",
        "summary_scope",
    }
)
_SINGLETON_DATASETS = frozenset(
    {
        "enterprise_report_identity",
        "enterprise_profile",
        "enterprise_exchange_rates",
        "enterprise_credit_overview",
        "enterprise_dispute_overview",
    }
)
_PUBLIC_COUNT_DATASETS = {
    "tax_arrears": "enterprise_public_tax_arrears_records",
    "civil_judgment": "enterprise_public_civil_judgment_records",
    "enforcement": "enterprise_public_enforcement_records",
    "administrative_penalty": "enterprise_public_administrative_penalty_records",
}
_CONTINUATION_DATASETS = {
    "current_credit_summary": "enterprise_current_credit_summary",
    "closed_credit_summary": "enterprise_closed_credit_summary",
    "repayment_responsibility_summary": "enterprise_repayment_responsibility_summary",
    "repayment_liability": "enterprise_repayment_responsibility_accounts",
    "attachment_account": "enterprise_attachment_accounts",
    "attachment_credit_detail": "enterprise_attachment_credit_details",
}


def _compact(value: Any) -> str:
    return _SPACE.sub("", "" if value is None else str(value)).strip()


def _is_missing(value: Any) -> bool:
    return _compact(value) in _MISSING_MARKERS


def _record_value(record: dict[str, Any], field_name: str) -> Any:
    normalized = record.get("normalized")
    if isinstance(normalized, dict) and field_name in normalized:
        return normalized.get(field_name)
    return record.get(field_name)


def _record_id(dataset: str, record: dict[str, Any], index: int) -> str:
    normalized = record.get("normalized") if isinstance(record.get("normalized"), dict) else {}
    for field_name in (
        "record_id",
        "account_id",
        "credit_line_id",
        "liability_id",
        "enterprise_identity_id",
        "enterprise_profile_id",
        "current_summary_id",
        "closed_summary_id",
        "displayed_summary_id",
        "responsibility_summary_id",
        "attachment_account_id",
        "attachment_detail_id",
        "supplement_id",
        "public_record_id",
        "note_id",
    ):
        value = record.get(field_name) or normalized.get(field_name)
        if value not in (None, ""):
            return str(value)
    return f"{dataset}:r{index:06d}"


def _source_pages(record: dict[str, Any]) -> tuple[int, ...]:
    pages: set[int] = set()
    for key in ("source_page", "source_page_end", "page"):
        value = record.get(key)
        if isinstance(value, int) and value > 0:
            pages.add(value)
    for ref in record.get("source_refs") or ():
        if not isinstance(ref, dict):
            continue
        value = ref.get("page", ref.get("source_page"))
        if isinstance(value, int) and value > 0:
            pages.add(value)
    return tuple(sorted(pages))


def _source_refs(record: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    return tuple(dict(ref) for ref in (record.get("source_refs") or ()) if isinstance(ref, dict))


@dataclass(frozen=True)
class EnterpriseExtractionFailure:
    """One deterministic extraction defect with source evidence."""

    code: str
    message: str
    stage: str
    severity: str = "error"
    path: str = ""
    dataset: str = ""
    record_id: str = ""
    field_name: str = ""
    source_pages: tuple[int, ...] = ()
    source_refs: tuple[dict[str, Any], ...] = ()
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": self.code,
            "severity": self.severity,
            "stage": self.stage,
            "message": self.message,
        }
        if self.path:
            payload["path"] = self.path
        if self.dataset:
            payload["dataset"] = self.dataset
        if self.record_id:
            payload["record_id"] = self.record_id
        if self.field_name:
            payload["field"] = self.field_name
        if self.source_pages:
            payload["source_pages"] = list(self.source_pages)
        if self.source_refs:
            payload["source_refs"] = [dict(ref) for ref in self.source_refs]
        if self.evidence:
            payload["evidence"] = dict(self.evidence)
        return payload


@dataclass(frozen=True)
class EnterpriseExtractionReport:
    """Machine-readable extraction outcome embedded in semantic JSON."""

    status: str
    failures: tuple[EnterpriseExtractionFailure, ...]
    checks: dict[str, Any]
    protocol: str = "pboc-enterprise-extraction-failure"
    version: str = "1.0.0"

    def to_payload(self) -> dict[str, Any]:
        failures = [failure.to_payload() for failure in self.failures]
        return {
            "protocol": self.protocol,
            "version": self.version,
            "status": self.status,
            "summary": {
                "failure_count": sum(item["severity"] == "error" for item in failures),
                "warning_count": sum(item["severity"] == "warning" for item in failures),
                **dict(self.checks),
            },
            "failures": failures,
        }


def _table_rows(
    document: CanonicalEnterpriseDocumentIR,
) -> dict[tuple[int, str, int], tuple[str, ...]]:
    rows: dict[tuple[int, str, int], tuple[str, ...]] = {}
    for page, table_id, table_rows in document.table_rows:
        for row_index, row in enumerate(table_rows):
            rows[(int(page), str(table_id), row_index)] = tuple(str(value or "") for value in row)
    return rows


def _dictionary_datasets(data_dictionary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    datasets = data_dictionary.get("datasets")
    return datasets if isinstance(datasets, dict) else {}


def _field_columns(data_dictionary: dict[str, Any], dataset: str) -> dict[str, dict[str, Any]]:
    spec = _dictionary_datasets(data_dictionary).get(dataset)
    columns = spec.get("columns") if isinstance(spec, dict) else {}
    return columns if isinstance(columns, dict) else {}


def _all_labels(data_dictionary: dict[str, Any]) -> frozenset[str]:
    labels: set[str] = set()
    for spec in _dictionary_datasets(data_dictionary).values():
        columns = spec.get("columns") if isinstance(spec, dict) else {}
        if not isinstance(columns, dict):
            continue
        for metadata in columns.values():
            label = _compact(metadata.get("label")) if isinstance(metadata, dict) else ""
            if label:
                labels.add(label)
    return frozenset(labels)


def _business_field(field_name: str, metadata: dict[str, Any]) -> bool:
    label = _compact(metadata.get("label"))
    if not label or field_name in _TECHNICAL_FIELDS:
        return False
    if metadata.get("deprecated") or metadata.get("canonical_field"):
        return False
    if label.upper().endswith("ID"):
        return False
    if field_name.endswith("_status") and label.endswith("\u62a5\u544a\u72b6\u6001"):
        return False
    return not field_name.startswith("source_")


def _referenced_rows(
    record: dict[str, Any],
    table_rows: dict[tuple[int, str, int], tuple[str, ...]],
) -> list[tuple[int, str, int, tuple[str, ...]]]:
    selected: list[tuple[int, str, int, tuple[str, ...]]] = []
    seen: set[tuple[int, str, int]] = set()
    for ref in record.get("source_refs") or ():
        if not isinstance(ref, dict):
            continue
        page = ref.get("page", ref.get("source_page"))
        table_id = ref.get("table_id", ref.get("source_table_id"))
        row = ref.get("row")
        if not isinstance(page, int) or not isinstance(row, int) or not table_id:
            continue
        key = (page, str(table_id), row)
        if key in seen or key not in table_rows:
            continue
        seen.add(key)
        selected.append((*key, table_rows[key]))
    return sorted(selected, key=lambda item: (item[0], item[1], item[2]))


def _values_for_label(
    rows: Iterable[tuple[int, str, int, tuple[str, ...]]],
    label: str,
    labels: frozenset[str],
) -> tuple[list[str], tuple[int, ...], tuple[dict[str, Any], ...]]:
    compact_label = _compact(label)
    ordered = list(rows)
    values: list[str] = []
    pages: set[int] = set()
    refs: list[dict[str, Any]] = []
    for position, (page, table_id, row_index, row) in enumerate(ordered):
        compact_cells = [_compact(cell) for cell in row]
        for column, cell in enumerate(compact_cells):
            if not cell or compact_label not in cell:
                continue
            pages.add(page)
            ref = {"source": "canonical_physical_table", "page": page, "table_id": table_id, "row": row_index}
            if ref not in refs:
                refs.append(ref)
            suffix = _LABEL_SUFFIX.sub("", cell.split(compact_label, 1)[1])
            if suffix and suffix not in labels and not _is_missing(suffix):
                values.append(suffix)
            if cell == compact_label and column + 1 < len(compact_cells):
                adjacent = compact_cells[column + 1]
                if adjacent and adjacent not in labels and not _is_missing(adjacent):
                    values.append(adjacent)
            if cell != compact_label:
                continue
            for next_page, next_table, next_index, next_row in ordered[position + 1 :]:
                if next_page != page or next_table != table_id or next_index <= row_index:
                    continue
                next_cells = [_compact(value) for value in next_row]
                if column >= len(next_cells):
                    continue
                candidate = next_cells[column]
                if candidate and candidate not in labels and not _is_missing(candidate):
                    values.append(candidate)
                break
    return list(dict.fromkeys(values)), tuple(sorted(pages)), tuple(refs)


def _text_values_for_label(
    document: CanonicalEnterpriseDocumentIR,
    label: str,
) -> tuple[list[str], tuple[int, ...], tuple[dict[str, Any], ...]]:
    compact_label = _compact(label)
    values: list[str] = []
    pages: set[int] = set()
    refs: list[dict[str, Any]] = []
    for page, text in document.page_texts.items():
        lines = [_compact(line) for line in str(text or "").splitlines() if _compact(line)]
        for index, line in enumerate(lines):
            if compact_label not in line:
                continue
            pages.add(int(page))
            ref = {"source": "canonical_native_text", "page": int(page)}
            if ref not in refs:
                refs.append(ref)
            suffix = _LABEL_SUFFIX.sub("", line.split(compact_label, 1)[1])
            if suffix and not _is_missing(suffix):
                values.append(suffix)
            elif line == compact_label and index + 1 < len(lines) and not _is_missing(lines[index + 1]):
                values.append(lines[index + 1])
    return list(dict.fromkeys(values)), tuple(sorted(pages)), tuple(refs)


def _document_values_for_label(
    document: CanonicalEnterpriseDocumentIR,
    label: str,
    labels: frozenset[str],
    table_rows: dict[tuple[int, str, int], tuple[str, ...]],
) -> tuple[list[str], tuple[int, ...], tuple[dict[str, Any], ...]]:
    rows = [(*key, row) for key, row in sorted(table_rows.items())]
    table_values, table_pages, table_refs = _values_for_label(rows, label, labels)
    text_values, text_pages, text_refs = _text_values_for_label(document, label)
    return (
        list(dict.fromkeys([*table_values, *text_values])),
        tuple(sorted({*table_pages, *text_pages})),
        tuple([*table_refs, *(ref for ref in text_refs if ref not in table_refs)]),
    )


def _numeric(value: Any) -> Decimal | None:
    match = _NUMBER.search(_compact(value).replace(",", "").replace("\uff0c", ""))
    if match is None:
        return None
    try:
        return Decimal(match.group(0))
    except InvalidOperation:
        return None


def _equivalent(source: Any, extracted: Any, metadata: dict[str, Any]) -> bool | None:
    field_type = str(metadata.get("type") or "string")
    source_value = _compact(source)
    target_value = _compact(extracted)
    if not source_value or not target_value:
        return False
    if field_type in {"money", "number", "integer", "percentage"}:
        left = _numeric(source_value)
        right = _numeric(target_value)
        return None if left is None or right is None else left == right
    if field_type in {"date", "datetime"}:
        left = "".join(_DATE_DIGITS.findall(source_value))
        right = "".join(_DATE_DIGITS.findall(target_value))
        if not left or not right:
            return None
        return left.startswith(right) or right.startswith(left)
    if field_type == "long_id" or metadata.get("format") == "long_id":
        left = re.sub(r"[^0-9A-Z]", "", source_value.upper())
        right = re.sub(r"[^0-9A-Z]", "", target_value.upper())
        return bool(left and right and left == right)
    return None


def _field_failure(
    *,
    code: str,
    dataset: str,
    record_id: str,
    field_name: str,
    label: str,
    values: list[str],
    pages: tuple[int, ...],
    refs: tuple[dict[str, Any], ...],
    extracted_value: Any = None,
) -> EnterpriseExtractionFailure:
    path = f"/data/{dataset}/{record_id}/{field_name}"
    if code == "EXPECTED_FIELD_NOT_EXTRACTED":
        message = f"Source field {label} is populated but {dataset}.{field_name} is absent."
    else:
        message = f"Extracted {dataset}.{field_name} does not match the populated source field {label}."
    evidence: dict[str, Any] = {"source_label": label, "source_values": values}
    if extracted_value not in (None, ""):
        evidence["extracted_value"] = extracted_value
    return EnterpriseExtractionFailure(
        code=code,
        message=message,
        stage="canonical_extraction",
        path=path,
        dataset=dataset,
        record_id=record_id,
        field_name=field_name,
        source_pages=pages,
        source_refs=refs,
        evidence=evidence,
    )


def _validate_record_fields(
    document: CanonicalEnterpriseDocumentIR,
    datasets: dict[str, list[dict[str, Any]]],
    data_dictionary: dict[str, Any],
    table_rows: dict[tuple[int, str, int], tuple[str, ...]],
    labels: frozenset[str],
) -> tuple[list[EnterpriseExtractionFailure], int, int]:
    failures: list[EnterpriseExtractionFailure] = []
    checked = 0
    satisfied = 0
    enum_fields = set((data_dictionary.get("enums") or {}).keys())
    for dataset, records in datasets.items():
        if dataset in _SINGLETON_DATASETS:
            continue
        columns = _field_columns(data_dictionary, dataset)
        if not columns:
            continue
        for index, record in enumerate(records, start=1):
            if not isinstance(record, dict):
                continue
            record_id = _record_id(dataset, record, index)
            source_rows = _referenced_rows(record, table_rows)
            if not source_rows:
                continue
            for field_name, metadata in columns.items():
                if not isinstance(metadata, dict) or not _business_field(field_name, metadata):
                    continue
                label = str(metadata.get("label") or "")
                values, pages, refs = _values_for_label(source_rows, label, labels)
                if not values:
                    continue
                checked += 1
                extracted = _record_value(record, field_name)
                if extracted in (None, ""):
                    failures.append(
                        _field_failure(
                            code="EXPECTED_FIELD_NOT_EXTRACTED",
                            dataset=dataset,
                            record_id=record_id,
                            field_name=field_name,
                            label=label,
                            values=values,
                            pages=pages or _source_pages(record),
                            refs=refs or _source_refs(record),
                        )
                    )
                    continue
                comparable = None if field_name in enum_fields else [_equivalent(value, extracted, metadata) for value in values]
                if comparable and all(result is False for result in comparable):
                    failures.append(
                        _field_failure(
                            code="EXTRACTED_FIELD_VALUE_MISMATCH",
                            dataset=dataset,
                            record_id=record_id,
                            field_name=field_name,
                            label=label,
                            values=values,
                            pages=pages or _source_pages(record),
                            refs=refs or _source_refs(record),
                            extracted_value=extracted,
                        )
                    )
                    continue
                satisfied += 1
    return failures, checked, satisfied


def _validate_singleton_fields(
    document: CanonicalEnterpriseDocumentIR,
    datasets: dict[str, list[dict[str, Any]]],
    data_dictionary: dict[str, Any],
    table_rows: dict[tuple[int, str, int], tuple[str, ...]],
    labels: frozenset[str],
) -> tuple[list[EnterpriseExtractionFailure], int, int]:
    failures: list[EnterpriseExtractionFailure] = []
    checked = 0
    satisfied = 0
    enum_fields = set((data_dictionary.get("enums") or {}).keys())
    for dataset in _SINGLETON_DATASETS:
        columns = _field_columns(data_dictionary, dataset)
        records = datasets.get(dataset) or []
        record = records[0] if records else {}
        record_id = _record_id(dataset, record, 1)
        for field_name, metadata in columns.items():
            if not isinstance(metadata, dict) or not _business_field(field_name, metadata):
                continue
            label = str(metadata.get("label") or "")
            values, pages, refs = _document_values_for_label(document, label, labels, table_rows)
            if not values:
                continue
            checked += 1
            extracted = _record_value(record, field_name)
            if extracted in (None, ""):
                failures.append(
                    _field_failure(
                        code="EXPECTED_FIELD_NOT_EXTRACTED",
                        dataset=dataset,
                        record_id=record_id,
                        field_name=field_name,
                        label=label,
                        values=values,
                        pages=pages,
                        refs=refs,
                    )
                )
                continue
            comparable = None if field_name in enum_fields else [_equivalent(value, extracted, metadata) for value in values]
            if comparable and all(result is False for result in comparable):
                failures.append(
                    _field_failure(
                        code="EXTRACTED_FIELD_VALUE_MISMATCH",
                        dataset=dataset,
                        record_id=record_id,
                        field_name=field_name,
                        label=label,
                        values=values,
                        pages=pages,
                        refs=refs,
                        extracted_value=extracted,
                    )
                )
                continue
            satisfied += 1
    return failures, checked, satisfied


def _record_contract_failures(
    datasets: dict[str, list[dict[str, Any]]],
    continuation_audit: Iterable[dict[str, Any]],
    dataset_completeness: dict[str, dict[str, Any]],
) -> tuple[list[EnterpriseExtractionFailure], int]:
    failures: list[EnterpriseExtractionFailure] = []
    checked = 0
    failed_datasets: set[str] = set()
    for audit in continuation_audit:
        checked += 1
        unresolved = int(audit.get("unresolved_record_count") or 0)
        unexpected = int(audit.get("unexpected_record_count") or 0)
        if not unresolved and not unexpected:
            continue
        family = str(audit.get("continuation_family") or "unknown")
        dataset = _CONTINUATION_DATASETS.get(family, "")
        if dataset:
            failed_datasets.add(dataset)
        failures.append(
            EnterpriseExtractionFailure(
                code="RECORD_RECONSTRUCTION_MISMATCH",
                message=(
                    f"Continuation family {family} expected {int(audit.get('expected_record_count') or 0)} "
                    f"records but extracted {int(audit.get('extracted_record_count') or 0)}."
                ),
                stage="continuation_reconstruction",
                path=f"/data/{dataset}" if dataset else "/data",
                dataset=dataset,
                evidence={
                    key: audit.get(key)
                    for key in (
                        "continuation_family",
                        "business_category",
                        "expected_record_count",
                        "extracted_record_count",
                        "unresolved_record_count",
                        "unexpected_record_count",
                    )
                    if audit.get(key) not in (None, "")
                },
            )
        )
    for dataset, details in dataset_completeness.items():
        if dataset in failed_datasets or details.get("verified") is not False:
            continue
        checked += 1
        failures.append(
            EnterpriseExtractionFailure(
                code="RECORD_RECONSTRUCTION_MISMATCH",
                message=(
                    f"Dataset {dataset} emitted {int(details.get('emitted_row_count') or 0)} "
                    f"of {int(details.get('expected_row_count') or 0)} expected records."
                ),
                stage="canonical_extraction",
                path=f"/data/{dataset}",
                dataset=dataset,
                evidence=dict(details),
            )
        )
    counts = datasets.get("enterprise_public_record_counts") or []
    for count_record in counts:
        record_type = str(_record_value(count_record, "record_type") or "")
        target_dataset = _PUBLIC_COUNT_DATASETS.get(record_type)
        if not target_dataset:
            continue
        checked += 1
        expected = int(_record_value(count_record, "record_count") or 0)
        extracted = len(datasets.get(target_dataset) or [])
        if expected == extracted:
            continue
        failures.append(
            EnterpriseExtractionFailure(
                code="RECORD_RECONSTRUCTION_MISMATCH",
                message=f"Public-record summary reports {expected} {record_type} records but {extracted} were extracted.",
                stage="canonical_extraction",
                path=f"/data/{target_dataset}",
                dataset=target_dataset,
                source_pages=_source_pages(count_record),
                source_refs=_source_refs(count_record),
                evidence={
                    "record_type": record_type,
                    "expected_record_count": expected,
                    "extracted_record_count": extracted,
                },
            )
        )
    return failures, checked


def _input_and_component_failures(
    document: CanonicalEnterpriseDocumentIR,
) -> list[EnterpriseExtractionFailure]:
    failures: list[EnterpriseExtractionFailure] = []
    for flag in document.input_quality_flags:
        if flag.get("severity") != "error":
            continue
        failures.append(
            EnterpriseExtractionFailure(
                code="INPUT_INTEGRITY_VIOLATION",
                message=str(flag.get("message") or "Enterprise ParseResult failed an input-integrity check."),
                stage="parseresult_input",
                source_pages=tuple(int(page) for page in (flag.get("source_pages") or ()) if int(page) > 0),
                evidence={"source_code": str(flag.get("code") or "")},
            )
        )
    unassigned = tuple(str(value) for value in document.entity_context.unassigned_unit_ids if str(value))
    if not document.entity_context.content_conserved or unassigned:
        failures.append(
            EnterpriseExtractionFailure(
                code="UNCONSUMED_BUSINESS_TEXT",
                message="One or more source units were not assigned to the canonical enterprise component graph.",
                stage="continuation_reconstruction",
                path="/canonical_ir/components",
                evidence={"unassigned_component_ids": list(unassigned)},
            )
        )
    return failures


def _unstructured_content_failures(
    datasets: dict[str, list[dict[str, Any]]],
    labels: frozenset[str],
) -> list[EnterpriseExtractionFailure]:
    failures: list[EnterpriseExtractionFailure] = []
    for dataset, records in datasets.items():
        if dataset == "report_notes" or "statement" in dataset:
            continue
        for index, record in enumerate(records, start=1):
            normalized = record.get("normalized") if isinstance(record.get("normalized"), dict) else record
            content = normalized.get("content") if isinstance(normalized, dict) else None
            if not isinstance(content, str) or not content.strip():
                continue
            compact_content = _compact(content)
            matched = sorted(label for label in labels if len(label) >= 3 and label in compact_content)
            if len(matched) < 2:
                continue
            record_id = _record_id(dataset, record, index)
            failures.append(
                EnterpriseExtractionFailure(
                    code="UNSTRUCTURED_BUSINESS_CONTENT",
                    message=f"{dataset}.content contains multiple canonical business fields instead of typed values.",
                    stage="canonical_extraction",
                    path=f"/data/{dataset}/{record_id}/content",
                    dataset=dataset,
                    record_id=record_id,
                    field_name="content",
                    source_pages=_source_pages(record),
                    source_refs=_source_refs(record),
                    evidence={"matched_labels": matched},
                )
            )
    return failures


def build_enterprise_extraction_report(
    document: CanonicalEnterpriseDocumentIR,
    datasets: dict[str, list[dict[str, Any]]],
    *,
    continuation_audit: Iterable[dict[str, Any]],
    dataset_completeness: dict[str, dict[str, Any]],
    data_dictionary: dict[str, Any],
) -> EnterpriseExtractionReport:
    """Validate the canonical extraction without attempting a second extraction."""
    table_rows = _table_rows(document)
    labels = _all_labels(data_dictionary)
    failures = _input_and_component_failures(document)
    singleton_failures, singleton_checked, singleton_satisfied = _validate_singleton_fields(
        document,
        datasets,
        data_dictionary,
        table_rows,
        labels,
    )
    row_failures, row_checked, row_satisfied = _validate_record_fields(
        document,
        datasets,
        data_dictionary,
        table_rows,
        labels,
    )
    record_failures, record_checked = _record_contract_failures(
        datasets,
        continuation_audit,
        dataset_completeness,
    )
    failures.extend(singleton_failures)
    failures.extend(row_failures)
    failures.extend(record_failures)
    failures.extend(_unstructured_content_failures(datasets, labels))

    deduplicated: list[EnterpriseExtractionFailure] = []
    seen: set[tuple[str, str, str, str]] = set()
    for failure in failures:
        marker = (failure.code, failure.dataset, failure.record_id, failure.field_name)
        if marker in seen:
            continue
        seen.add(marker)
        deduplicated.append(failure)
    fatal = any(failure.code == "INPUT_INTEGRITY_VIOLATION" for failure in deduplicated)
    status = "failed" if fatal else "partial" if deduplicated else "complete"
    checked_fields = singleton_checked + row_checked
    satisfied_fields = singleton_satisfied + row_satisfied
    return EnterpriseExtractionReport(
        status=status,
        failures=tuple(deduplicated),
        checks={
            "checked_field_count": checked_fields,
            "satisfied_field_count": satisfied_fields,
            "failed_field_count": max(0, checked_fields - satisfied_fields),
            "record_contract_count": record_checked,
            "source_component_count": len(document.components),
            "source_components_conserved": bool(document.entity_context.content_conserved),
        },
    )


__all__ = [
    "EnterpriseExtractionFailure",
    "EnterpriseExtractionReport",
    "build_enterprise_extraction_report",
]
