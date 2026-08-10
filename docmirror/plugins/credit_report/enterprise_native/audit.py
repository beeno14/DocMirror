# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observational audit pass for canonical digital enterprise reports.

The audit consumes the finalized canonical datasets.  It never repairs,
suppresses, or rewrites extracted business values.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from docmirror.plugins.credit_report.enterprise_native.ir import (
    CanonicalEnterpriseDocumentIR,
)

_SPACE = re.compile(r"\s+")
_REPORT_UNIT = re.compile(
    r"金额类(?:汇总)?数据项单位(?:均)?为(?:人民币)?(万元|元)"
)
_SEVERITY_RANK = {"error": 0, "warning": 1, "info": 2}
_MISSING_TEXT = frozenset({"", "-", "--", "—", "－", "/", "不详", "未知"})
_REPORT_DEFAULT_UNIT_DATASETS = frozenset(
    {
        "enterprise_public_utility_payment_records",
        "enterprise_public_tax_arrears_records",
        "enterprise_public_civil_judgment_records",
        "enterprise_public_enforcement_records",
        "enterprise_public_administrative_penalty_records",
        "enterprise_public_housing_fund_payment_records",
        "enterprise_utility_payment_history",
        "enterprise_housing_fund_history",
    }
)
_DATASET_PRIMARY_IDS = {
    "enterprise_credit_accounts": ("account_id",),
    "enterprise_credit_facilities": ("credit_line_id",),
    "enterprise_repayment_responsibility_accounts": ("liability_id",),
    "enterprise_attachment_accounts": ("attachment_account_id",),
    "enterprise_credit_supplement": ("supplement_id",),
    "enterprise_attachment_credit_details": ("attachment_detail_id",),
    "enterprise_special_transactions": ("special_transaction_id",),
    "enterprise_utility_payment_history": ("utility_history_id",),
    "enterprise_housing_fund_history": ("housing_fund_history_id",),
}


def _compact(value: Any) -> str:
    return _SPACE.sub("", "" if value is None else str(value)).strip()


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() in _MISSING_TEXT)


def _normalized(record: Mapping[str, Any]) -> Mapping[str, Any]:
    value = record.get("normalized")
    return value if isinstance(value, Mapping) else record


def _record_value(record: Mapping[str, Any], field_name: str) -> Any:
    normalized = _normalized(record)
    return normalized.get(field_name, record.get(field_name))


def _record_id(dataset: str, record: Mapping[str, Any], index: int) -> str:
    normalized = _normalized(record)
    candidates = (
        *_DATASET_PRIMARY_IDS.get(dataset, ()),
        "record_id",
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
        "utility_history_id",
        "housing_fund_history_id",
        "account_id",
        "attachment_account_id",
    )
    for field_name in dict.fromkeys(candidates):
        value = record.get(field_name) or normalized.get(field_name)
        if value not in (None, ""):
            return str(value)
    return f"{dataset}:r{index:06d}"


def _source_refs(record: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    return tuple(
        dict(ref)
        for ref in (record.get("source_refs") or ())
        if isinstance(ref, Mapping)
    )


def _source_pages(record: Mapping[str, Any]) -> tuple[int, ...]:
    pages: set[int] = set()
    normalized = _normalized(record)
    for source in (record, normalized):
        for key in ("source_page", "source_page_end", "page"):
            value = source.get(key)
            if isinstance(value, int) and value > 0:
                pages.add(value)
    for ref in _source_refs(record):
        value = ref.get("page", ref.get("source_page"))
        if isinstance(value, int) and value > 0:
            pages.add(value)
    return tuple(sorted(pages))


def _path(dataset: str = "", record_id: str = "", field_name: str = "") -> str:
    values = ["data"]
    values.extend(value for value in (dataset, record_id, field_name) if value)
    return "/" + "/".join(values)


@dataclass(frozen=True)
class EnterpriseAuditFinding:
    """One actionable observation that does not change extracted values."""

    code: str
    severity: str
    category: str
    message: str
    path: str = ""
    dataset: str = ""
    record_id: str = ""
    field_name: str = ""
    source_pages: tuple[int, ...] = ()
    source_refs: tuple[dict[str, Any], ...] = ()
    evidence: dict[str, Any] = field(default_factory=dict)
    recommended_action: str = "Review the cited source area before relying on this value."

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": self.code,
            "severity": self.severity,
            "category": self.category,
            "message": self.message,
            "recommended_action": self.recommended_action,
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
class EnterpriseAuditReport:
    """Machine-readable, warning-only review protocol."""

    findings: tuple[EnterpriseAuditFinding, ...]
    checks: dict[str, Any]
    protocol: str = "pboc-enterprise-observational-audit"
    version: str = "1.0.0"
    mode: str = "observational"

    def to_payload(self) -> dict[str, Any]:
        findings = [finding.to_payload() for finding in self.findings]
        warning_count = sum(
            item["severity"] in {"warning", "error"} for item in findings
        )
        return {
            "protocol": self.protocol,
            "version": self.version,
            "mode": self.mode,
            "mutates_extraction": False,
            "status": "review_required" if warning_count else "clear",
            "summary": {
                "finding_count": len(findings),
                "info_count": sum(item["severity"] == "info" for item in findings),
                "warning_count": sum(
                    item["severity"] == "warning" for item in findings
                ),
                "error_count": sum(item["severity"] == "error" for item in findings),
                **dict(self.checks),
            },
            "findings": findings,
        }


def _finding_from_extraction(value: Mapping[str, Any]) -> EnterpriseAuditFinding:
    return EnterpriseAuditFinding(
        code=str(value.get("code") or "ENTERPRISE_EXTRACTION_REVIEW"),
        severity=str(value.get("severity") or "error"),
        category="extraction_validation",
        message=str(value.get("message") or "Enterprise extraction requires review."),
        path=str(value.get("path") or ""),
        dataset=str(value.get("dataset") or ""),
        record_id=str(value.get("record_id") or ""),
        field_name=str(value.get("field") or ""),
        source_pages=tuple(
            sorted(
                {
                    int(page)
                    for page in (value.get("source_pages") or ())
                    if str(page).isdigit() and int(page) > 0
                }
            )
        ),
        source_refs=tuple(
            dict(ref)
            for ref in (value.get("source_refs") or ())
            if isinstance(ref, Mapping)
        ),
        evidence=dict(value.get("evidence") or {}),
    )


def _finding_from_quality(
    value: Mapping[str, Any],
) -> EnterpriseAuditFinding:
    code = str(value.get("code") or "ENTERPRISE_SOURCE_REVIEW")
    severity = str(value.get("severity") or "warning")
    dataset = str(value.get("dataset") or "")
    field_name = str(value.get("field") or "")
    path = (
        f"/data/{dataset}/*/{field_name}"
        if dataset and field_name
        else f"/data/{dataset}/*"
        if dataset
        else f"/data/*/*/{field_name}"
        if field_name
        else "/source"
    )
    details = dict(value.get("details") or {})
    for key in ("status", "scope", "category"):
        if value.get(key) not in (None, "", []):
            details[key] = value.get(key)
    return EnterpriseAuditFinding(
        code=code,
        severity=severity,
        category=str(value.get("category") or "source_information"),
        message=str(value.get("message") or "Enterprise source information requires review."),
        path=path,
        dataset=dataset,
        field_name=field_name,
        source_pages=tuple(
            sorted(
                {
                    int(page)
                    for page in (value.get("source_pages") or ())
                    if str(page).isdigit() and int(page) > 0
                }
            )
        ),
        evidence=details,
        recommended_action=(
            "No action is required; this entry records source-reported absence or context."
            if severity == "info"
            else "Review the cited source limitation when interpreting affected values."
        ),
    )


def _field_resolution_findings(
    datasets: Mapping[str, Iterable[Mapping[str, Any]]],
) -> tuple[list[EnterpriseAuditFinding], Counter[str]]:
    findings: list[EnterpriseAuditFinding] = []
    states: Counter[str] = Counter()
    inherited_context: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "record_ids": set(),
            "fields": set(),
            "source_pages": set(),
            "source_refs": [],
        }
    )
    for dataset, raw_records in datasets.items():
        for index, record in enumerate(raw_records, start=1):
            if not isinstance(record, Mapping):
                continue
            record_id = _record_id(dataset, record, index)
            field_info = record.get("field_info")
            if not isinstance(field_info, Mapping):
                continue
            for field_name, raw_info in field_info.items():
                if not isinstance(raw_info, Mapping):
                    continue
                info = dict(raw_info)
                state = str(info.get("source_state") or "unclassified")
                states[state] += 1
                refs = tuple(
                    dict(ref)
                    for ref in (info.get("source_refs") or ())
                    if isinstance(ref, Mapping)
                ) or _source_refs(record)
                pages = tuple(
                    sorted(
                        {
                            int(ref.get("page", ref.get("source_page")))
                            for ref in refs
                            if str(ref.get("page", ref.get("source_page", ""))).isdigit()
                            and int(ref.get("page", ref.get("source_page"))) > 0
                        }
                    )
                ) or _source_pages(record)
                conflicts = [
                    str(value)
                    for value in (info.get("conflicts") or ())
                    if str(value)
                ]
                if state == "unresolved":
                    findings.append(
                        EnterpriseAuditFinding(
                            code="ENTERPRISE_AUDIT_FIELD_UNRESOLVED",
                            severity="warning",
                            category="field_provenance",
                            message=(
                                f"{dataset}.{field_name} has no unique authoritative source binding; "
                                "the extraction result was not altered."
                            ),
                            path=_path(dataset, record_id, str(field_name)),
                            dataset=dataset,
                            record_id=record_id,
                            field_name=str(field_name),
                            source_pages=pages,
                            source_refs=refs,
                            evidence={
                                key: info[key]
                                for key in ("basis", "source_value", "source_label")
                                if info.get(key) not in (None, "")
                            },
                        )
                    )
                if conflicts:
                    findings.append(
                        EnterpriseAuditFinding(
                            code="ENTERPRISE_AUDIT_FIELD_EVIDENCE_CONFLICT",
                            severity="warning",
                            category="field_provenance",
                            message=(
                                f"Competing source evidence exists for {dataset}.{field_name}; "
                                "the selected value was retained."
                            ),
                            path=_path(dataset, record_id, str(field_name)),
                            dataset=dataset,
                            record_id=record_id,
                            field_name=str(field_name),
                            source_pages=pages,
                            source_refs=refs,
                            evidence={"conflicts": conflicts, "basis": info.get("basis")},
                        )
                    )
                if state == "derived" and (
                    not str(info.get("basis") or "") or not refs
                ):
                    findings.append(
                        EnterpriseAuditFinding(
                            code="ENTERPRISE_AUDIT_DERIVED_PROVENANCE_MISSING",
                            severity="warning",
                            category="field_provenance",
                            message=(
                                f"Derived field {dataset}.{field_name} lacks a complete rule/evidence reference; "
                                "the value was retained."
                            ),
                            path=_path(dataset, record_id, str(field_name)),
                            dataset=dataset,
                            record_id=record_id,
                            field_name=str(field_name),
                            source_pages=pages,
                            source_refs=refs,
                            evidence={"basis": info.get("basis")},
                        )
                    )
                if (
                    state == "derived"
                    and info.get("basis") == "adjacent_continuation_footer"
                    and refs
                ):
                    bucket = inherited_context[(dataset, str(info.get("basis")))]
                    bucket["record_ids"].add(record_id)
                    bucket["fields"].add(str(field_name))
                    bucket["source_pages"].update(pages)
                    for ref in refs:
                        if ref not in bucket["source_refs"]:
                            bucket["source_refs"].append(dict(ref))
    for (dataset, basis), details in sorted(inherited_context.items()):
        record_ids = sorted(details["record_ids"])
        fields = sorted(details["fields"])
        findings.append(
            EnterpriseAuditFinding(
                code="ENTERPRISE_AUDIT_CONTINUATION_CONTEXT_INHERITED",
                severity="info",
                category="continuation_provenance",
                message=(
                    f"{len(record_ids)} {dataset} record(s) inherit {', '.join(fields)} "
                    "from one authoritative continuation footer."
                ),
                path=f"/data/{dataset}/*",
                dataset=dataset,
                source_pages=tuple(sorted(details["source_pages"])),
                source_refs=tuple(details["source_refs"]),
                evidence={
                    "basis": basis,
                    "record_ids": record_ids,
                    "fields": fields,
                },
                recommended_action="No action is required; this entry records shared continuation evidence.",
            )
        )
    return findings, states


def _explicit_amount_units(labels: Iterable[Any]) -> frozenset[str]:
    text = "".join(_compact(label) for label in labels if _compact(label))
    units: set[str] = set()
    if re.search(r"[（(](?:人民币)?万元[）)]", text) or re.search(
        r"单位[:：]?(?:人民币)?万元", text
    ):
        units.add("CNY_10K")
    if re.search(r"[（(](?:人民币)?元[）)]", text) or re.search(
        r"单位[:：]?(?:人民币)?元(?:整)?(?:$|[，。；;])", text
    ):
        units.add("CNY_1")
    return frozenset(units)


def _report_default_unit(
    document: CanonicalEnterpriseDocumentIR,
) -> tuple[str, tuple[int, ...]]:
    matches: list[tuple[str, int]] = []
    for page, text in document.page_texts.items():
        match = _REPORT_UNIT.search(_compact(text))
        if match:
            matches.append(("CNY_10K" if match.group(1) == "万元" else "CNY_1", int(page)))
    units = {unit for unit, _ in matches}
    if len(units) != 1:
        return "", tuple(sorted(page for _, page in matches))
    return next(iter(units)), tuple(sorted(page for _, page in matches))


def _table_rows(
    document: CanonicalEnterpriseDocumentIR,
) -> dict[tuple[int, str], tuple[tuple[str, ...], ...]]:
    return {
        (int(page), str(table_id)): tuple(tuple(str(value or "") for value in row) for row in rows)
        for page, table_id, rows in document.table_rows
    }


def _record_unit_evidence(
    record: Mapping[str, Any],
    rows_by_table: Mapping[tuple[int, str], tuple[tuple[str, ...], ...]],
) -> tuple[frozenset[str], tuple[str, ...], bool]:
    """Resolve only row-local or nearest preceding unit evidence.

    A physical table may contain several independent subregions. Flattening the
    whole table would let a unit from an unrelated subregion affect this record.
    """

    units: set[str] = set()
    table_ids: list[str] = []
    row_evidence_available = False
    for ref in _source_refs(record):
        page = ref.get("page", ref.get("source_page"))
        table_id = ref.get("table_id", ref.get("source_table_id"))
        row_index = ref.get("row", ref.get("source_row"))
        if not (
            str(page).isdigit()
            and int(page) > 0
            and str(table_id or "")
            and str(row_index).isdigit()
        ):
            continue
        key = (int(page), str(table_id))
        rows = rows_by_table.get(key) or ()
        index = int(row_index)
        if not 0 <= index < len(rows):
            continue
        row_evidence_available = True
        if str(table_id) not in table_ids:
            table_ids.append(str(table_id))
        units.update(_explicit_amount_units(rows[index]))
    return frozenset(units), tuple(table_ids), row_evidence_available


def _money_field_groups(
    data_dictionary: Mapping[str, Any],
) -> dict[str, tuple[tuple[str, str, tuple[str, ...]], ...]]:
    """Map money fields to the currency/unit pair that governs them."""

    output: dict[str, tuple[tuple[str, str, tuple[str, ...]], ...]] = {}
    datasets = data_dictionary.get("datasets")
    if not isinstance(datasets, Mapping):
        return output
    for dataset, raw_spec in datasets.items():
        spec = raw_spec if isinstance(raw_spec, Mapping) else {}
        columns = spec.get("columns") if isinstance(spec.get("columns"), Mapping) else {}
        money_fields = tuple(
            str(field_name)
            for field_name, metadata in columns.items()
            if isinstance(metadata, Mapping) and metadata.get("type") == "money"
        )
        if not money_fields:
            continue
        special: dict[str, tuple[str, str]] = {}
        if str(dataset) == "enterprise_repayment_responsibility_accounts":
            special = {
                "responsibility_amount": (
                    "responsibility_amount_unit",
                    "responsibility_currency",
                ),
                "loan_or_credit_amount": (
                    "obligation_amount_unit",
                    "obligation_currency",
                ),
            }
        groups: list[tuple[str, str, tuple[str, ...]]] = []
        generic = tuple(field for field in money_fields if field not in special)
        if generic:
            groups.append(("amount_unit", "currency", generic))
        for field_name, (unit_field, currency_field) in special.items():
            if field_name in money_fields:
                groups.append((unit_field, currency_field, (field_name,)))
        output[str(dataset)] = tuple(groups)
    return output


def _currency_specific_unit(
    magnitude_unit: str,
    *,
    currency: str,
    emitted: str,
) -> str:
    if not magnitude_unit:
        return ""
    suffix = "10K" if magnitude_unit.endswith("_10K") else "1"
    code = currency.strip().upper()
    if not code and "_" in emitted:
        code = emitted.split("_", 1)[0].strip().upper()
    return f"{code or 'CNY'}_{suffix}"


def _amount_unit_findings(
    document: CanonicalEnterpriseDocumentIR,
    datasets: Mapping[str, Iterable[Mapping[str, Any]]],
    data_dictionary: Mapping[str, Any],
) -> tuple[list[EnterpriseAuditFinding], int]:
    findings: list[EnterpriseAuditFinding] = []
    provenance: dict[tuple[str, str, str, str], dict[str, Any]] = defaultdict(
        lambda: {"record_ids": [], "source_pages": set(), "table_ids": set()}
    )
    default_unit, default_pages = _report_default_unit(document)
    rows_by_table = _table_rows(document)
    checked = 0
    for dataset, groups in _money_field_groups(data_dictionary).items():
        for index, record in enumerate(datasets.get(dataset) or (), start=1):
            record_units, table_ids, row_evidence_available = _record_unit_evidence(
                record,
                rows_by_table,
            )
            for unit_field, currency_field, fields in groups:
                populated = [
                    field_name
                    for field_name in fields
                    if not _is_missing(_record_value(record, field_name))
                ]
                if not populated:
                    continue
                raw_field_info = record.get("field_info")
                field_info = (
                    raw_field_info.get(unit_field)
                    if isinstance(raw_field_info, Mapping)
                    and isinstance(raw_field_info.get(unit_field), Mapping)
                    else {}
                )
                if field_info.get("source_state") == "unresolved":
                    # The generic field-provenance audit already emits the
                    # addressable unresolved/conflict warning for this field.
                    continue
                basis = str(field_info.get("basis") or "")
                conflicts = {
                    str(value)
                    for value in (field_info.get("conflicts") or ())
                    if str(value)
                }
                explicit_units = (
                    frozenset(conflicts)
                    if conflicts
                    else frozenset({str(field_info.get("source_value"))})
                    if basis == "local_explicit_amount_unit"
                    and field_info.get("source_value")
                    else record_units
                )
                default_scope_verified = bool(
                    basis == "report_default_amount_unit"
                    or (
                        dataset in _REPORT_DEFAULT_UNIT_DATASETS
                        and row_evidence_available
                    )
                )
                if not explicit_units and not default_scope_verified:
                    # Without field-local unit evidence, a report-wide default is
                    # only authoritative for the table families whose extractor
                    # implements that precedence explicitly.
                    continue
                checked += 1
                record_id = _record_id(dataset, record, index)
                emitted = str(_record_value(record, unit_field) or "")
                currency = str(_record_value(record, currency_field) or "")
                refs = tuple(
                    dict(ref)
                    for ref in (field_info.get("source_refs") or ())
                    if isinstance(ref, Mapping)
                ) or _source_refs(record)
                pages = tuple(
                    sorted(
                        {
                            *_source_pages(record),
                            *default_pages,
                            *(
                                int(ref.get("page", ref.get("source_page")))
                                for ref in refs
                                if str(
                                    ref.get("page", ref.get("source_page", ""))
                                ).isdigit()
                            ),
                        }
                    )
                )
                evidence = {
                    "money_fields": populated,
                    "unit_field": unit_field,
                    "currency_field": currency_field,
                    "currency": currency,
                    "emitted_amount_unit": emitted,
                    "explicit_units": sorted(explicit_units),
                    "report_default_amount_unit": default_unit,
                    "source_table_ids": list(table_ids),
                }
                if len(explicit_units) > 1:
                    findings.append(
                        EnterpriseAuditFinding(
                            code="ENTERPRISE_AUDIT_AMOUNT_UNIT_EVIDENCE_CONFLICT",
                            severity="warning",
                            category="amount_unit",
                            message=(
                                f"Money fields in {dataset} have conflicting local unit evidence; "
                                "the extracted values were retained."
                            ),
                            path=_path(dataset, record_id, unit_field),
                            dataset=dataset,
                            record_id=record_id,
                            field_name=unit_field,
                            source_pages=pages,
                            source_refs=refs,
                            evidence=evidence,
                        )
                    )
                    continue
                magnitude = next(iter(explicit_units)) if explicit_units else default_unit
                expected = _currency_specific_unit(
                    magnitude,
                    currency=currency,
                    emitted=emitted,
                )
                if not expected:
                    findings.append(
                        EnterpriseAuditFinding(
                            code="ENTERPRISE_AUDIT_AMOUNT_UNIT_UNRESOLVED",
                            severity="warning",
                            category="amount_unit",
                            message=(
                                f"Money fields in {dataset} have neither a unique local unit nor a report default; "
                                "the numeric values were retained."
                            ),
                            path=_path(dataset, record_id, unit_field),
                            dataset=dataset,
                            record_id=record_id,
                            field_name=unit_field,
                            source_pages=pages,
                            source_refs=refs,
                            evidence=evidence,
                        )
                    )
                    continue
                if emitted != expected:
                    findings.append(
                        EnterpriseAuditFinding(
                            code="ENTERPRISE_AUDIT_AMOUNT_UNIT_MISMATCH",
                            severity="warning",
                            category="amount_unit",
                            message=(
                                f"{dataset}.{unit_field} is {emitted or 'missing'} but source precedence resolves to {expected}; "
                                "the extracted value was retained."
                            ),
                            path=_path(dataset, record_id, unit_field),
                            dataset=dataset,
                            record_id=record_id,
                            field_name=unit_field,
                            source_pages=pages,
                            source_refs=refs,
                            evidence={**evidence, "expected_amount_unit": expected},
                        )
                    )
                    continue
                provenance_basis = (
                    "local_explicit" if explicit_units else "report_default"
                )
                bucket = provenance[
                    (dataset, unit_field, provenance_basis, expected)
                ]
                bucket["record_ids"].append(record_id)
                bucket["source_pages"].update(_source_pages(record))
                bucket["table_ids"].update(table_ids)

    for (dataset, unit_field, basis, amount_unit), details in sorted(provenance.items()):
        record_ids = list(details["record_ids"])
        source_pages = tuple(sorted({*details["source_pages"], *default_pages}))
        findings.append(
            EnterpriseAuditFinding(
                code=(
                    "ENTERPRISE_AUDIT_AMOUNT_UNIT_LOCAL_EVIDENCE"
                    if basis == "local_explicit"
                    else "ENTERPRISE_AUDIT_AMOUNT_UNIT_REPORT_DEFAULT"
                ),
                severity="info",
                category="amount_unit",
                message=(
                    f"{len(record_ids)} {dataset} record(s) use {amount_unit} from "
                    f"{'local table labels' if basis == 'local_explicit' else 'the report-wide default'}."
                ),
                path=f"/data/{dataset}/*/{unit_field}",
                dataset=dataset,
                field_name=unit_field,
                source_pages=source_pages,
                evidence={
                    "basis": basis,
                    "amount_unit": amount_unit,
                    "record_count": len(record_ids),
                    "sample_record_ids": record_ids[:20],
                    "source_table_ids": sorted(details["table_ids"]),
                    "report_default_source_pages": list(default_pages),
                },
                recommended_action="No action is required; this entry records the applied unit precedence.",
            )
        )
    return findings, checked


def _schema_value_count(
    datasets: Mapping[str, Iterable[Mapping[str, Any]]],
    data_dictionary: Mapping[str, Any],
) -> int:
    count = 0
    dictionary_datasets = data_dictionary.get("datasets")
    if not isinstance(dictionary_datasets, Mapping):
        return count
    for dataset, records in datasets.items():
        raw_spec = dictionary_datasets.get(dataset)
        spec = raw_spec if isinstance(raw_spec, Mapping) else {}
        columns = spec.get("columns") if isinstance(spec.get("columns"), Mapping) else {}
        if not columns:
            continue
        for record in records:
            if not isinstance(record, Mapping):
                continue
            for field_name in columns:
                if not _is_missing(_record_value(record, str(field_name))):
                    count += 1
    return count


def build_enterprise_audit_report(
    document: CanonicalEnterpriseDocumentIR,
    datasets: Mapping[str, Iterable[Mapping[str, Any]]],
    *,
    extraction_report: Mapping[str, Any],
    quality_flags: Iterable[Mapping[str, Any]],
    data_dictionary: Mapping[str, Any],
) -> EnterpriseAuditReport:
    """Inspect finalized extraction results without modifying them."""

    dataset_records = {
        str(dataset): tuple(records) for dataset, records in datasets.items()
    }

    extraction_failures = [
        value
        for value in (extraction_report.get("failures") or ())
        if isinstance(value, Mapping)
    ]
    findings = [_finding_from_extraction(value) for value in extraction_failures]
    findings.extend(
        _finding_from_quality(value)
        for value in quality_flags
        if isinstance(value, Mapping) and value.get("severity") in {"info", "warning", "error"}
    )
    field_findings, field_states = _field_resolution_findings(dataset_records)
    findings.extend(field_findings)
    unit_findings, checked_money_records = _amount_unit_findings(
        document,
        dataset_records,
        data_dictionary,
    )
    findings.extend(unit_findings)

    deduplicated: list[EnterpriseAuditFinding] = []
    seen: set[tuple[Any, ...]] = set()
    for finding in sorted(
        findings,
        key=lambda item: (
            _SEVERITY_RANK.get(item.severity, 9),
            item.code,
            item.dataset,
            item.record_id,
            item.field_name,
            item.path,
        ),
    ):
        marker = (
            finding.code,
            finding.severity,
            finding.category,
            finding.dataset,
            finding.record_id,
            finding.field_name,
            finding.path,
            finding.message,
            finding.source_pages,
            repr(finding.source_refs),
            repr(finding.evidence),
        )
        if marker in seen:
            continue
        seen.add(marker)
        deduplicated.append(finding)

    summary = extraction_report.get("summary")
    extraction_summary = summary if isinstance(summary, Mapping) else {}
    schema_value_count = _schema_value_count(dataset_records, data_dictionary)
    checked = int(extraction_summary.get("checked_field_count") or 0)
    verified_equal = int(
        extraction_summary.get("verified_equal_field_count") or 0
    )
    present_unverified = int(
        extraction_summary.get("present_unverified_field_count") or 0
    )
    return EnterpriseAuditReport(
        findings=tuple(deduplicated),
        checks={
            "semantic_dataset_count": sum(bool(records) for records in dataset_records.values()),
            "semantic_record_count": sum(len(records) for records in dataset_records.values()),
            "populated_schema_field_count": schema_value_count,
            "label_linked_field_check_count": checked,
            "verified_equal_field_count": verified_equal,
            "present_unverified_field_count": present_unverified,
            "label_linked_schema_field_ratio": (
                round(checked / schema_value_count, 6) if schema_value_count else 1.0
            ),
            "verified_equal_schema_field_ratio": (
                round(verified_equal / schema_value_count, 6)
                if schema_value_count
                else 1.0
            ),
            "amount_unit_group_check_count": checked_money_records,
            "field_source_state_counts": dict(sorted(field_states.items())),
            "source_components_conserved": bool(document.entity_context.content_conserved),
            "extraction_values_changed": 0,
            "coverage_scope": (
                "Counts cover populated dictionary fields; equality checks cover only "
                "fields with uniquely addressable source labels and comparable types."
            ),
        },
    )


def safely_build_enterprise_audit_report(
    document: CanonicalEnterpriseDocumentIR,
    datasets: Mapping[str, Iterable[Mapping[str, Any]]],
    *,
    extraction_report: Mapping[str, Any],
    quality_flags: Iterable[Mapping[str, Any]],
    data_dictionary: Mapping[str, Any],
) -> EnterpriseAuditReport:
    """Keep audit implementation faults from suppressing extracted business data."""

    try:
        return build_enterprise_audit_report(
            document,
            datasets,
            extraction_report=extraction_report,
            quality_flags=quality_flags,
            data_dictionary=data_dictionary,
        )
    except Exception as exc:  # pragma: no cover - branch is fault-injection tested
        return EnterpriseAuditReport(
            findings=(
                EnterpriseAuditFinding(
                    code="ENTERPRISE_AUDIT_INTERNAL_ERROR",
                    severity="warning",
                    category="audit_runtime",
                    message=(
                        "The observational audit could not complete; extracted business "
                        "values were retained unchanged."
                    ),
                    path="/audit",
                    evidence={"error_type": type(exc).__name__},
                    recommended_action=(
                        "Review the enterprise semantic result manually and inspect the audit runtime."
                    ),
                ),
            ),
            checks={
                "extraction_values_changed": 0,
                "coverage_scope": "Audit execution failed before coverage could be calculated.",
            },
        )


def audit_warning_strings(audit_report: Mapping[str, Any]) -> tuple[str, ...]:
    """Render warning/error findings for the existing Community warning channel."""

    values: list[str] = []
    for finding in audit_report.get("findings") or ():
        if not isinstance(finding, Mapping) or finding.get("severity") not in {
            "warning",
            "error",
        }:
            continue
        path = str(finding.get("path") or "")
        suffix = f" Review {path}." if path else ""
        values.append(
            f"{finding.get('code', 'ENTERPRISE_AUDIT_REVIEW')}: "
            f"{str(finding.get('message') or '').strip()}{suffix}"
        )
    return tuple(dict.fromkeys(values))


__all__ = [
    "EnterpriseAuditFinding",
    "EnterpriseAuditReport",
    "audit_warning_strings",
    "build_enterprise_audit_report",
    "safely_build_enterprise_audit_report",
]
