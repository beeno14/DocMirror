# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Warning-only business audit for digital PBOC personal-brief reports.

The auditor observes finalized semantic and Community datasets.  It never
repairs, suppresses, reorders, or otherwise changes extracted business data.
Only compact findings are appended to the Community ``warnings`` array.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal

from docmirror.plugins.credit_report.personal_brief_native.contracts import (
    PERSONAL_BRIEF_ENUM_CONTRACT,
    PERSONAL_BRIEF_MONEY_FIELDS,
    PERSONAL_BRIEF_REPORTING_AMOUNT_PRECISION,
    PERSONAL_BRIEF_REPORTING_AMOUNT_UNIT,
    PERSONAL_BRIEF_REPORTING_CURRENCY,
)

AuditLevel = Literal["info", "warning", "error"]

_MONEY = re.compile(r"(?:0|[1-9][0-9]*)\Z")
_ISO_DATE_OR_MONTH = re.compile(r"(?P<year>[0-9]{4})-(?P<month>[0-9]{2})(?:-(?P<day>[0-9]{2}))?\Z")
_SEVERITY_RANK = {"error": 0, "warning": 1, "info": 2}
_DATASET_ORDER = (
    "personal_report_metadata",
    "identity_documents",
    "personal_credit_summary_records",
    "asset_disposition_records",
    "guarantor_compensation_records",
    "credit_accounts",
    "overdue_records",
    "repayment_liability_records",
    "postpaid_records",
    "tax_arrears_records",
    "civil_judgment_records",
    "enforcement_records",
    "administrative_penalty_records",
    "institution_statement_records",
    "inquiry_records",
)
_DATASET_RANK = {name: index for index, name in enumerate(_DATASET_ORDER)}
_ACCOUNT_CATEGORY_BY_TYPE = {
    "credit_card": "credit_cards",
    "loan": "loans",
    "credit_line": "loans",
    "other_business": "other_business",
}
_TERMINATION_BY_LIFECYCLE = {
    "closed": "account_closed",
    "settled": "debt_settled",
    "transferred_out": "transferred_out",
}
_ACCOUNT_AMOUNT_STATUS_FIELDS = (
    ("credit_limit", "credit_limit_status"),
    ("used_amount", "used_amount_status"),
    ("loan_amount", "loan_amount_status"),
    ("balance", "balance_status"),
)
_ACCOUNT_SUMMARY_METRICS = frozenset(
    {
        "account_count",
        "unclosed_account_count",
        "ever_overdue_account_count",
        "over_90_days_account_count",
    }
)
_SINGLETON_SUMMARY_METRICS = frozenset(
    {
        "asset_disposition_count",
        "guarantor_compensation_count",
        "personal_repayment_liability_count",
        "enterprise_repayment_liability_count",
    }
)
_ACCOUNT_SUMMARY_CATEGORIES = frozenset(
    {"credit_card", "housing_loan", "other_loan", "other_business"}
)
_OVERDUE_IDENTITY_FIELDS = (
    "account_type",
    "institution",
    "business_type",
    "card_tail",
    "open_date",
)
_PUBLIC_RECORD_DATASETS = {
    "tax_arrears": "tax_arrears_records",
    "civil_judgment": "civil_judgment_records",
    "enforcement": "enforcement_records",
    "administrative_penalty": "administrative_penalty_records",
}
_SECTION_DATASETS_BY_TYPE = {
    "report_header": ("personal_report_metadata", "identity_documents"),
    "credit_details": (
        "personal_credit_summary_records",
        "asset_disposition_records",
        "guarantor_compensation_records",
        "credit_accounts",
        "overdue_records",
        "repayment_liability_records",
    ),
    "non_credit_transactions": ("postpaid_records",),
    "public_records": tuple(_PUBLIC_RECORD_DATASETS.values()),
    "institution_statements": ("institution_statement_records",),
    "inquiries": ("inquiry_records",),
}


def _normalized(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("normalized")
    return value if isinstance(value, Mapping) else {}


def _rows(dataset: Mapping[str, Any] | None) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(dataset, Mapping):
        return ()
    return tuple(row for row in (dataset.get("rows") or ()) if isinstance(row, Mapping))


def _datasets(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(dataset.get("name") or ""): dataset
        for dataset in (payload.get("datasets") or ())
        if isinstance(dataset, Mapping) and dataset.get("name")
    }


def _has_value(values: Mapping[str, Any], field_name: str) -> bool:
    value = values.get(field_name)
    return field_name in values and value is not None and value != ""


def _record_id(row: Mapping[str, Any], index: int, dataset_name: str) -> str:
    value = row.get("record_id")
    return str(value) if value not in (None, "") else f"{dataset_name}:r{index:06d}"


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, str) and value.isdigit() and int(value) > 0:
        return int(value)
    return None


def _nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _date_precedes(value: Any, reference: Any) -> bool:
    """Compare ISO dates without inventing a day for month-precision values."""

    left = _ISO_DATE_OR_MONTH.fullmatch(str(value or ""))
    right = _ISO_DATE_OR_MONTH.fullmatch(str(reference or ""))
    if left is None or right is None:
        return False
    left_month = (int(left.group("year")), int(left.group("month")))
    right_month = (int(right.group("year")), int(right.group("month")))
    if left_month != right_month:
        return left_month < right_month
    left_day = left.group("day")
    right_day = right.group("day")
    return bool(left_day and right_day and int(left_day) < int(right_day))


def _source_pages(row: Mapping[str, Any]) -> tuple[int, ...]:
    pages: set[int] = set()
    source = row.get("source") if isinstance(row.get("source"), Mapping) else {}
    page_range = source.get("page_range") if isinstance(source, Mapping) else None
    if isinstance(page_range, Sequence) and not isinstance(page_range, (str, bytes)):
        for value in page_range:
            page = _positive_int(value)
            if page is not None:
                pages.add(page)
    for ref in source.get("source_refs") or ():
        if not isinstance(ref, Mapping):
            continue
        for key in ("page", "source_page", "page_number"):
            page = _positive_int(ref.get(key))
            if page is not None:
                pages.add(page)
    for key in ("source_page", "source_page_end", "page"):
        page = _positive_int(row.get(key))
        if page is not None:
            pages.add(page)
    return tuple(sorted(pages))


def _rich_row_lookup(
    semantic_datasets: Mapping[str, Mapping[str, Any]],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    output: dict[tuple[str, str], Mapping[str, Any]] = {}
    for dataset_name, dataset in semantic_datasets.items():
        for index, row in enumerate(_rows(dataset), start=1):
            output[(dataset_name, _record_id(row, index, dataset_name))] = row
    return output


@dataclass(frozen=True)
class PersonalBriefAuditFinding:
    """One deterministic observation against finalized personal-brief data."""

    code: str
    level: AuditLevel
    message: str
    dataset: str = ""
    section_id: str = ""
    record_id: str = ""
    fields: tuple[str, ...] = ()
    source_pages: tuple[int, ...] = ()

    def contextual_message(self) -> str:
        context: list[str] = []
        if self.dataset:
            context.append(f"dataset={self.dataset}")
        if self.record_id:
            context.append(f"record_id={self.record_id}")
        if self.fields:
            context.append(f"fields={','.join(self.fields)}")
        prefix = f"[{'; '.join(context)}] " if context else ""
        return f"{prefix}{self.message}"


def _finding(
    code: str,
    message: str,
    *,
    level: AuditLevel = "error",
    dataset: str = "",
    section_id: str = "",
    record_id: str = "",
    fields: Sequence[str] = (),
    source_pages: Sequence[int] = (),
) -> PersonalBriefAuditFinding:
    pages: set[int] = set()
    for value in source_pages:
        page = _positive_int(value)
        if page is not None:
            pages.add(page)
    return PersonalBriefAuditFinding(
        code=code,
        level=level,
        message=message,
        dataset=dataset,
        section_id=section_id,
        record_id=record_id,
        fields=tuple(str(field) for field in fields if field),
        source_pages=tuple(sorted(pages)),
    )


def _finding_sort_key(finding: PersonalBriefAuditFinding) -> tuple[Any, ...]:
    return (
        _SEVERITY_RANK.get(finding.level, 99),
        _DATASET_RANK.get(finding.dataset, len(_DATASET_ORDER)),
        finding.dataset,
        finding.section_id,
        finding.source_pages or (10**9,),
        finding.record_id,
        finding.fields,
        finding.code,
        finding.message,
    )


def _deduplicate_findings(
    findings: Sequence[PersonalBriefAuditFinding],
) -> tuple[PersonalBriefAuditFinding, ...]:
    unique: dict[tuple[Any, ...], PersonalBriefAuditFinding] = {}
    for finding in findings:
        key = (
            finding.code,
            finding.level,
            finding.dataset,
            finding.section_id,
            finding.record_id,
            finding.fields,
            finding.source_pages,
            finding.message,
        )
        unique.setdefault(key, finding)
    return tuple(sorted(unique.values(), key=_finding_sort_key))


def _collect_projection_conservation(
    semantic_datasets: Mapping[str, Mapping[str, Any]],
    public_datasets: Mapping[str, Mapping[str, Any]],
) -> list[PersonalBriefAuditFinding]:
    from docmirror.plugins.credit_report.personal_brief_native.projector import (
        _personal_brief_public_dataset_policy_template,
    )

    findings: list[PersonalBriefAuditFinding] = []
    policy = _personal_brief_public_dataset_policy_template()
    retained = set(_DATASET_ORDER)
    for dataset_name in _DATASET_ORDER:
        semantic_dataset = semantic_datasets.get(dataset_name)
        public_dataset = public_datasets.get(dataset_name)
        semantic_rows = _rows(semantic_dataset)
        if semantic_rows and public_dataset is None:
            findings.append(
                _finding(
                    "PERSONAL_BRIEF_AUDIT_PROJECTION_CONSERVATION_CONFLICT",
                    "A non-empty semantic business dataset is absent from Community JSON.",
                    dataset=dataset_name,
                )
            )
            continue
        if public_dataset is not None and semantic_dataset is None:
            findings.append(
                _finding(
                    "PERSONAL_BRIEF_AUDIT_PROJECTION_CONSERVATION_CONFLICT",
                    "A Community business dataset has no semantic counterpart.",
                    dataset=dataset_name,
                )
            )
            continue
        if public_dataset is None or semantic_dataset is None:
            continue
        public_rows = _rows(public_dataset)
        semantic_ids = [
            _record_id(row, index, dataset_name)
            for index, row in enumerate(semantic_rows, start=1)
        ]
        public_ids = [
            _record_id(row, index, dataset_name)
            for index, row in enumerate(public_rows, start=1)
        ]
        if public_ids != semantic_ids:
            findings.append(
                _finding(
                    "PERSONAL_BRIEF_AUDIT_PROJECTION_CONSERVATION_CONFLICT",
                    "Community row identities or source order differ from the semantic dataset.",
                    dataset=dataset_name,
                )
            )
            continue
        semantic_by_id = {
            _record_id(row, index, dataset_name): row
            for index, row in enumerate(semantic_rows, start=1)
        }
        published_columns = tuple(
            str(column.get("key") or "")
            for column in (public_dataset.get("columns") or ())
            if isinstance(column, Mapping) and column.get("key")
        )
        public_field_contract = tuple(policy.get(dataset_name) or ())
        if published_columns != public_field_contract:
            findings.append(
                _finding(
                    "PERSONAL_BRIEF_AUDIT_PROJECTION_CONSERVATION_CONFLICT",
                    "Published columns differ from the plugin-owned personal-brief field contract.",
                    dataset=dataset_name,
                    fields=("columns",),
                )
            )
        for index, public_row in enumerate(public_rows, start=1):
            record_id = _record_id(public_row, index, dataset_name)
            semantic_values = _normalized(semantic_by_id[record_id])
            public_values = _normalized(public_row)
            changed_fields = [
                field_name
                for field_name in public_field_contract
                if _has_value(semantic_values, field_name)
                != _has_value(public_values, field_name)
                or (
                    _has_value(semantic_values, field_name)
                    and semantic_values.get(field_name) != public_values.get(field_name)
                )
            ]
            if not changed_fields:
                continue
            findings.append(
                _finding(
                    "PERSONAL_BRIEF_AUDIT_PROJECTION_CONSERVATION_CONFLICT",
                    "Published business values differ from their semantic source values.",
                    dataset=dataset_name,
                    record_id=record_id,
                    fields=changed_fields,
                    source_pages=_source_pages(semantic_by_id[record_id]),
                )
            )
    unexpected_public = sorted(set(public_datasets) - retained)
    for dataset_name in unexpected_public:
        findings.append(
            _finding(
                "PERSONAL_BRIEF_AUDIT_PROJECTION_CONSERVATION_CONFLICT",
                "Community JSON contains a dataset outside the personal-brief public contract.",
                dataset=dataset_name,
            )
        )
    return findings


def _collect_dataset_envelopes(
    public_datasets: Mapping[str, Mapping[str, Any]],
) -> list[PersonalBriefAuditFinding]:
    findings: list[PersonalBriefAuditFinding] = []
    for dataset_name, dataset in public_datasets.items():
        rows = _rows(dataset)
        actual = len(rows)
        row_count = dataset.get("row_count")
        if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count != actual:
            findings.append(
                _finding(
                    "PERSONAL_BRIEF_AUDIT_DATASET_ENVELOPE_CONFLICT",
                    f"row_count={row_count!r} does not equal the emitted row count {actual}.",
                    dataset=dataset_name,
                    fields=("row_count",),
                )
            )
        ids = [
            _record_id(row, index, dataset_name)
            for index, row in enumerate(rows, start=1)
        ]
        for record_id, count in sorted(Counter(ids).items()):
            if count <= 1:
                continue
            findings.append(
                _finding(
                    "PERSONAL_BRIEF_AUDIT_DUPLICATE_RECORD_ID",
                    f"The record identifier occurs {count} times.",
                    dataset=dataset_name,
                    record_id=record_id,
                    fields=("record_id",),
                )
            )
        completeness = (
            dataset.get("completeness")
            if isinstance(dataset.get("completeness"), Mapping)
            else {}
        )
        emitted = completeness.get("emitted_row_count")
        expected = completeness.get("expected_row_count")
        omitted = completeness.get("omitted_row_count")
        verified = completeness.get("verified")
        valid_expected = isinstance(expected, int) and not isinstance(expected, bool) and expected >= 0
        valid_omitted = isinstance(omitted, int) and not isinstance(omitted, bool) and omitted >= 0
        conflict = emitted != actual or not valid_expected or not valid_omitted
        if valid_expected and valid_omitted:
            conflict = conflict or omitted != max(expected - actual, 0)
        if not isinstance(verified, bool):
            conflict = True
        if verified is True and valid_expected:
            conflict = conflict or expected != actual or omitted != 0
        if conflict:
            findings.append(
                _finding(
                    "PERSONAL_BRIEF_AUDIT_DATASET_ENVELOPE_CONFLICT",
                    "Dataset completeness counters are internally inconsistent.",
                    dataset=dataset_name,
                    fields=(
                        "completeness.expected_row_count",
                        "completeness.emitted_row_count",
                        "completeness.omitted_row_count",
                        "completeness.verified",
                    ),
                )
            )
        if verified is False:
            issue_fields = [
                key
                for key in (
                    "omitted_row_count",
                    "unexpected_row_count",
                    "unresolved_row_count",
                    "missing_required_field_record_ids",
                    "invalid_provenance_record_ids",
                    "uncovered_boundary_ids",
                    "duplicate_boundary_ids",
                    "missing_source_fields",
                    "present_but_unobserved",
                )
                if completeness.get(key) not in (None, 0, [], (), {})
            ]
            findings.append(
                _finding(
                    "PERSONAL_BRIEF_AUDIT_DATASET_INCOMPLETE",
                    "The canonical dataset completeness contract requires review"
                    f"{': ' + ', '.join(issue_fields) if issue_fields else ''}.",
                    level="warning",
                    dataset=dataset_name,
                    fields=tuple(f"completeness.{key}" for key in issue_fields)
                    or ("completeness.verified",),
                )
            )
    return findings


def _collect_provenance(
    public_datasets: Mapping[str, Mapping[str, Any]],
    rich_rows: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[PersonalBriefAuditFinding]:
    findings: list[PersonalBriefAuditFinding] = []
    for dataset_name, dataset in public_datasets.items():
        for index, row in enumerate(_rows(dataset), start=1):
            record_id = _record_id(row, index, dataset_name)
            rich = rich_rows.get((dataset_name, record_id))
            if rich is None:
                continue
            source = rich.get("source") if isinstance(rich.get("source"), Mapping) else {}
            has_provenance = bool(
                source.get("page_range")
                or source.get("source_refs")
                or source.get("evidence_ids")
                or rich.get("source_refs")
                or rich.get("evidence_ids")
            )
            if has_provenance:
                continue
            findings.append(
                _finding(
                    "PERSONAL_BRIEF_AUDIT_PROVENANCE_MISSING",
                    "The emitted business row has no semantic source provenance.",
                    level="warning",
                    dataset=dataset_name,
                    record_id=record_id,
                )
            )
    return findings


def _collect_extraction_report_consistency(
    semantic_payload: Mapping[str, Any],
    public_datasets: Mapping[str, Mapping[str, Any]],
) -> list[PersonalBriefAuditFinding]:
    domain = semantic_payload.get("domain")
    facts = domain.get("facts") if isinstance(domain, Mapping) else None
    report = facts.get("personal_brief_extraction_report") if isinstance(facts, Mapping) else None
    if not isinstance(report, Mapping):
        return []
    status = str(report.get("status") or "")
    failures = [failure for failure in (report.get("failures") or ()) if isinstance(failure, Mapping)]
    content_conserved = report.get("content_conserved")
    completeness = (
        report.get("dataset_completeness")
        if isinstance(report.get("dataset_completeness"), Mapping)
        else {}
    )
    unverified = sorted(
        str(dataset_name)
        for dataset_name, details in completeness.items()
        if isinstance(details, Mapping) and details.get("verified") is not True
    )
    contradictions: list[str] = []
    if status == "complete" and failures:
        contradictions.append("status is complete while failures are present")
    if status == "complete" and content_conserved is not True:
        contradictions.append("status is complete while source content is not conserved")
    if status == "complete" and unverified:
        contradictions.append(f"status is complete while datasets are unverified: {unverified}")
    findings: list[PersonalBriefAuditFinding] = []
    if contradictions:
        findings.append(
            _finding(
                "PERSONAL_BRIEF_AUDIT_EXTRACTION_REPORT_CONFLICT",
                "; ".join(contradictions) + ".",
                fields=("personal_brief_extraction_report",),
            )
        )
    for dataset_name in unverified:
        if dataset_name in public_datasets:
            continue
        findings.append(
            _finding(
                "PERSONAL_BRIEF_AUDIT_DATASET_INCOMPLETE",
                "An unverified canonical dataset is absent from Community JSON.",
                level="warning",
                dataset=dataset_name,
                fields=("personal_brief_extraction_report.dataset_completeness",),
            )
        )
    if content_conserved is False and not failures:
        findings.append(
            _finding(
                "PERSONAL_BRIEF_AUDIT_SOURCE_CONSERVATION_FAILURE",
                "Canonical source content is not conserved and no extraction failure explains it.",
                level="warning",
                fields=("personal_brief_extraction_report.content_conserved",),
            )
        )
    return findings


def _collect_enum_contract(
    public_datasets: Mapping[str, Mapping[str, Any]],
    rich_rows: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[PersonalBriefAuditFinding]:
    findings: list[PersonalBriefAuditFinding] = []
    for (dataset_name, field_name), labels in PERSONAL_BRIEF_ENUM_CONTRACT.items():
        dataset = public_datasets.get(dataset_name)
        if dataset is None:
            continue
        columns = {
            str(column.get("key") or ""): column
            for column in (dataset.get("columns") or ())
            if isinstance(column, Mapping)
        }
        descriptor = columns.get(field_name)
        if (
            descriptor is None
            or descriptor.get("type") != "enum"
            or descriptor.get("enum") != labels
        ):
            findings.append(
                _finding(
                    "PERSONAL_BRIEF_AUDIT_ENUM_CONTRACT_CONFLICT",
                    "Published enum metadata differs from the plugin-owned allowed values.",
                    dataset=dataset_name,
                    fields=(field_name,),
                )
            )
        for index, row in enumerate(_rows(dataset), start=1):
            values = _normalized(row)
            if not _has_value(values, field_name):
                continue
            value = values.get(field_name)
            if not isinstance(value, str) or value not in labels:
                record_id = _record_id(row, index, dataset_name)
                rich = rich_rows.get((dataset_name, record_id), {})
                findings.append(
                    _finding(
                        "PERSONAL_BRIEF_AUDIT_ENUM_CONTRACT_CONFLICT",
                        f"The value {value!r} is outside the declared allowed-value set.",
                        dataset=dataset_name,
                        record_id=record_id,
                        fields=(field_name,),
                        source_pages=_source_pages(rich),
                    )
                )
    return findings


def _collect_money_contract(
    public_datasets: Mapping[str, Mapping[str, Any]],
    rich_rows: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[PersonalBriefAuditFinding]:
    findings: list[PersonalBriefAuditFinding] = []
    metadata_rows = _rows(public_datasets.get("personal_report_metadata"))
    metadata = _normalized(metadata_rows[0]) if metadata_rows else {}
    metadata_policy = (
        metadata.get("reporting_currency"),
        metadata.get("reporting_amount_unit"),
        metadata.get("reporting_amount_precision"),
    )
    expected_policy = (
        PERSONAL_BRIEF_REPORTING_CURRENCY,
        PERSONAL_BRIEF_REPORTING_AMOUNT_UNIT,
        PERSONAL_BRIEF_REPORTING_AMOUNT_PRECISION,
    )
    if metadata_rows and metadata_policy != expected_policy:
        record_id = _record_id(metadata_rows[0], 1, "personal_report_metadata")
        findings.append(
            _finding(
                "PERSONAL_BRIEF_AUDIT_MONEY_POLICY_CONFLICT",
                f"Report amount policy {metadata_policy!r} does not equal {expected_policy!r}.",
                dataset="personal_report_metadata",
                record_id=record_id,
                fields=(
                    "reporting_currency",
                    "reporting_amount_unit",
                    "reporting_amount_precision",
                ),
                source_pages=_source_pages(
                    rich_rows.get(("personal_report_metadata", record_id), {})
                ),
            )
        )
    for dataset_name, money_fields in PERSONAL_BRIEF_MONEY_FIELDS.items():
        dataset = public_datasets.get(dataset_name)
        if dataset is None:
            continue
        columns = {
            str(column.get("key") or ""): column
            for column in (dataset.get("columns") or ())
            if isinstance(column, Mapping)
        }
        for field_name in money_fields:
            descriptor = columns.get(field_name) or {}
            if (
                descriptor.get("type") != "money"
                or descriptor.get("unit") != PERSONAL_BRIEF_REPORTING_AMOUNT_UNIT
            ):
                findings.append(
                    _finding(
                        "PERSONAL_BRIEF_AUDIT_MONEY_POLICY_CONFLICT",
                        "Money-column unit metadata does not equal the report amount unit.",
                        dataset=dataset_name,
                        fields=(field_name,),
                    )
                )
        for index, row in enumerate(_rows(dataset), start=1):
            values = _normalized(row)
            record_id = _record_id(row, index, dataset_name)
            pages = _source_pages(rich_rows.get((dataset_name, record_id), {}))
            row_policy = (
                values.get("reporting_amount_currency"),
                values.get("reporting_amount_unit"),
            )
            if row_policy != expected_policy[:2]:
                findings.append(
                    _finding(
                        "PERSONAL_BRIEF_AUDIT_MONEY_POLICY_CONFLICT",
                        f"Row amount policy {row_policy!r} does not equal {expected_policy[:2]!r}.",
                        dataset=dataset_name,
                        record_id=record_id,
                        fields=("reporting_amount_currency", "reporting_amount_unit"),
                        source_pages=pages,
                    )
                )
            precision = values.get("reporting_amount_precision")
            if _has_value(values, "reporting_amount_precision") and (
                isinstance(precision, bool)
                or precision != PERSONAL_BRIEF_REPORTING_AMOUNT_PRECISION
            ):
                findings.append(
                    _finding(
                        "PERSONAL_BRIEF_AUDIT_MONEY_POLICY_CONFLICT",
                        "Row amount precision conflicts with CNY_1.",
                        dataset=dataset_name,
                        record_id=record_id,
                        fields=("reporting_amount_precision",),
                        source_pages=pages,
                    )
                )
            for field_name in money_fields:
                if not _has_value(values, field_name):
                    continue
                value = values.get(field_name)
                if not isinstance(value, str) or _MONEY.fullmatch(value) is None:
                    findings.append(
                        _finding(
                            "PERSONAL_BRIEF_AUDIT_MONEY_VALUE_CONFLICT",
                            "The amount is not a canonical nonnegative integral CNY_1 string.",
                            dataset=dataset_name,
                            record_id=record_id,
                            fields=(field_name,),
                            source_pages=pages,
                        )
                    )
    return findings


def _collect_identity(
    public_datasets: Mapping[str, Mapping[str, Any]],
    rich_rows: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[PersonalBriefAuditFinding]:
    metadata_rows = _rows(public_datasets.get("personal_report_metadata"))
    identity_rows = _rows(public_datasets.get("identity_documents"))
    findings: list[PersonalBriefAuditFinding] = []
    if len(metadata_rows) != 1:
        findings.append(
            _finding(
                "PERSONAL_BRIEF_AUDIT_IDENTITY_CONFLICT",
                f"Expected exactly one report metadata row, found {len(metadata_rows)}.",
                dataset="personal_report_metadata",
            )
        )
    primary_rows = [row for row in identity_rows if _normalized(row).get("is_primary") is True]
    if len(primary_rows) != 1:
        findings.append(
            _finding(
                "PERSONAL_BRIEF_AUDIT_IDENTITY_CONFLICT",
                f"Expected exactly one primary identity document, found {len(primary_rows)}.",
                dataset="identity_documents",
                fields=("is_primary",),
            )
        )
    if len(metadata_rows) != 1 or len(primary_rows) != 1:
        return findings
    metadata = _normalized(metadata_rows[0])
    primary = _normalized(primary_rows[0])
    mapping = (
        ("subject_name", "holder_name"),
        ("primary_id_type", "document_type"),
        ("primary_id_number", "document_number"),
    )
    mismatches = [left for left, right in mapping if metadata.get(left) != primary.get(right)]
    if mismatches:
        record_id = _record_id(primary_rows[0], 1, "identity_documents")
        findings.append(
            _finding(
                "PERSONAL_BRIEF_AUDIT_IDENTITY_CONFLICT",
                "Primary identity values do not agree with report metadata.",
                dataset="identity_documents",
                record_id=record_id,
                fields=tuple(mismatches),
                source_pages=_source_pages(
                    rich_rows.get(("identity_documents", record_id), {})
                ),
            )
        )
    return findings


def _collect_summary(
    public_datasets: Mapping[str, Mapping[str, Any]],
    rich_rows: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[PersonalBriefAuditFinding]:
    dataset_name = "personal_credit_summary_records"
    rows = _rows(public_datasets.get(dataset_name))
    findings: list[PersonalBriefAuditFinding] = []
    cells: dict[tuple[str, str], tuple[int, str, tuple[int, ...]]] = {}
    seen: Counter[tuple[str, str]] = Counter()
    for index, row in enumerate(rows, start=1):
        values = _normalized(row)
        record_id = _record_id(row, index, dataset_name)
        pages = _source_pages(rich_rows.get((dataset_name, record_id), {}))
        metric = str(values.get("metric") or "")
        category = str(values.get("business_category") or "")
        status = str(values.get("reporting_status") or "")
        key = (metric, category)
        seen[key] += 1
        valid_pair = (
            metric in _ACCOUNT_SUMMARY_METRICS
            and category in _ACCOUNT_SUMMARY_CATEGORIES
        ) or (metric in _SINGLETON_SUMMARY_METRICS and category == "all")
        if not valid_pair:
            findings.append(
                _finding(
                    "PERSONAL_BRIEF_AUDIT_SUMMARY_CONFLICT",
                    "The summary metric/business-category pair is outside the canonical layout.",
                    dataset=dataset_name,
                    record_id=record_id,
                    fields=("metric", "business_category"),
                    source_pages=pages,
                )
            )
        value_present = _has_value(values, "value")
        value = _nonnegative_int(values.get("value")) if value_present else None
        if status in {"reported", "derived"}:
            if value is None:
                findings.append(
                    _finding(
                        "PERSONAL_BRIEF_AUDIT_SUMMARY_CONFLICT",
                        "A reported summary cell does not contain a nonnegative integer value.",
                        dataset=dataset_name,
                        record_id=record_id,
                        fields=("value", "reporting_status"),
                        source_pages=pages,
                    )
                )
            else:
                cells[key] = (value, record_id, pages)
        elif status in {"not_reported", "not_applicable"} and value_present:
            findings.append(
                _finding(
                    "PERSONAL_BRIEF_AUDIT_SUMMARY_CONFLICT",
                    "An unreported summary cell contains a numeric value.",
                    dataset=dataset_name,
                    record_id=record_id,
                    fields=("value", "reporting_status"),
                    source_pages=pages,
                )
            )
    for (metric, category), count in sorted(seen.items()):
        if count <= 1:
            continue
        findings.append(
            _finding(
                "PERSONAL_BRIEF_AUDIT_SUMMARY_CONFLICT",
                f"The summary cell ({metric}, {category}) occurs {count} times.",
                dataset=dataset_name,
                fields=("metric", "business_category"),
            )
        )
    categories = {category for _, category in cells}
    for category in sorted(categories):
        values = {
            metric: cell
            for (metric, cell_category), cell in cells.items()
            if cell_category == category
        }
        account_count = values.get("account_count")
        unclosed = values.get("unclosed_account_count")
        ever_overdue = values.get("ever_overdue_account_count")
        over_90 = values.get("over_90_days_account_count")
        contradictions: list[str] = []
        if account_count and unclosed and unclosed[0] > account_count[0]:
            contradictions.append("unclosed_account_count > account_count")
        if account_count and ever_overdue and ever_overdue[0] > account_count[0]:
            contradictions.append("ever_overdue_account_count > account_count")
        if ever_overdue and over_90 and over_90[0] > ever_overdue[0]:
            contradictions.append("over_90_days_account_count > ever_overdue_account_count")
        if contradictions:
            pages = sorted({page for cell in values.values() for page in cell[2]})
            findings.append(
                _finding(
                    "PERSONAL_BRIEF_AUDIT_SUMMARY_CONFLICT",
                    f"Summary count lattice is impossible for {category}: {'; '.join(contradictions)}.",
                    dataset=dataset_name,
                    fields=("value",),
                    source_pages=pages,
                )
            )
    singleton_targets = {
        "asset_disposition_count": "asset_disposition_records",
        "guarantor_compensation_count": "guarantor_compensation_records",
    }
    for metric, target_dataset_name in singleton_targets.items():
        cell = cells.get((metric, "all"))
        target_dataset = public_datasets.get(target_dataset_name)
        completeness = (
            target_dataset.get("completeness")
            if isinstance(target_dataset, Mapping)
            and isinstance(target_dataset.get("completeness"), Mapping)
            else {}
        )
        if cell is None or completeness.get("verified") is not True:
            continue
        detail_count = len(_rows(target_dataset))
        if cell[0] == detail_count:
            continue
        findings.append(
            _finding(
                "PERSONAL_BRIEF_AUDIT_SUMMARY_CONFLICT",
                f"Reported {metric}={cell[0]} does not equal verified detail count {detail_count}.",
                dataset=dataset_name,
                record_id=cell[1],
                fields=("metric", "value"),
                source_pages=cell[2],
            )
        )
    personal_liabilities = cells.get(("personal_repayment_liability_count", "all"))
    enterprise_liabilities = cells.get(("enterprise_repayment_liability_count", "all"))
    liability_dataset = public_datasets.get("repayment_liability_records")
    liability_completeness = (
        liability_dataset.get("completeness")
        if isinstance(liability_dataset, Mapping)
        and isinstance(liability_dataset.get("completeness"), Mapping)
        else {}
    )
    if (
        personal_liabilities is not None
        and enterprise_liabilities is not None
        and liability_completeness.get("verified") is True
    ):
        reported_total = personal_liabilities[0] + enterprise_liabilities[0]
        detail_count = len(_rows(liability_dataset))
        if reported_total != detail_count:
            findings.append(
                _finding(
                    "PERSONAL_BRIEF_AUDIT_SUMMARY_CONFLICT",
                    "Reported personal/enterprise repayment-liability total "
                    f"{reported_total} does not equal verified detail count {detail_count}.",
                    dataset=dataset_name,
                    fields=("metric", "value"),
                    source_pages=sorted(
                        {*personal_liabilities[2], *enterprise_liabilities[2]}
                    ),
                )
            )
    return findings


def _collect_account_relations(
    public_datasets: Mapping[str, Mapping[str, Any]],
    rich_rows: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[PersonalBriefAuditFinding]:
    dataset_name = "credit_accounts"
    findings: list[PersonalBriefAuditFinding] = []
    for index, row in enumerate(_rows(public_datasets.get(dataset_name)), start=1):
        values = _normalized(row)
        record_id = _record_id(row, index, dataset_name)
        pages = _source_pages(rich_rows.get((dataset_name, record_id), {}))
        account_type = str(values.get("account_type") or "")
        category = str(values.get("business_category") or "")
        expected_category = _ACCOUNT_CATEGORY_BY_TYPE.get(account_type)
        if expected_category is not None and category != expected_category:
            findings.append(
                _finding(
                    "PERSONAL_BRIEF_AUDIT_ACCOUNT_RELATION_CONFLICT",
                    f"business_category={category!r} does not match account_type={account_type!r}.",
                    dataset=dataset_name,
                    record_id=record_id,
                    fields=("account_type", "business_category"),
                    source_pages=pages,
                )
            )
        lifecycle = str(values.get("account_lifecycle_state") or "")
        termination = str(values.get("termination_event_type") or "")
        expected_termination = _TERMINATION_BY_LIFECYCLE.get(lifecycle)
        if lifecycle == "open" and (termination or _has_value(values, "termination_event_date")):
            findings.append(
                _finding(
                    "PERSONAL_BRIEF_AUDIT_ACCOUNT_RELATION_CONFLICT",
                    "An open account contains a termination event.",
                    dataset=dataset_name,
                    record_id=record_id,
                    fields=(
                        "account_lifecycle_state",
                        "termination_event_type",
                        "termination_event_date",
                    ),
                    source_pages=pages,
                )
            )
        elif expected_termination is not None and (
            termination != expected_termination
            or not _has_value(values, "termination_event_date")
        ):
            findings.append(
                _finding(
                    "PERSONAL_BRIEF_AUDIT_ACCOUNT_RELATION_CONFLICT",
                    "Termination type/date do not match the terminal account lifecycle "
                    f"{lifecycle!r}.",
                    dataset=dataset_name,
                    record_id=record_id,
                    fields=(
                        "account_lifecycle_state",
                        "termination_event_type",
                        "termination_event_date",
                    ),
                    source_pages=pages,
                )
            )
        open_date = values.get("open_date")
        invalid_date_fields = [
            field_name
            for field_name in (
                "termination_event_date",
                "contract_maturity_date",
                "credit_line_expiry_date",
            )
            if _has_value(values, field_name)
            and _date_precedes(values.get(field_name), open_date)
        ]
        if invalid_date_fields:
            findings.append(
                _finding(
                    "PERSONAL_BRIEF_AUDIT_ACCOUNT_RELATION_CONFLICT",
                    "An account event or expiry date precedes the account open date.",
                    dataset=dataset_name,
                    record_id=record_id,
                    fields=("open_date", *invalid_date_fields),
                    source_pages=pages,
                )
            )
        payoff = str(values.get("payoff_state") or "")
        expected_payoff = None
        if account_type == "credit_card":
            expected_payoff = "not_applicable"
        elif account_type in {"loan", "credit_line", "other_business"}:
            expected_payoff = {
                "open": "outstanding",
                "settled": "settled",
                "transferred_out": "unknown",
            }.get(lifecycle)
        if expected_payoff is not None and payoff != expected_payoff:
            findings.append(
                _finding(
                    "PERSONAL_BRIEF_AUDIT_ACCOUNT_RELATION_CONFLICT",
                    f"payoff_state={payoff!r} does not match account type/lifecycle.",
                    dataset=dataset_name,
                    record_id=record_id,
                    fields=("account_type", "account_lifecycle_state", "payoff_state"),
                    source_pages=pages,
                )
            )
        if account_type != "credit_card":
            invalid_card_fields = [
                field_name
                for field_name in ("credit_card_type", "card_tail")
                if _has_value(values, field_name)
            ]
            activation = values.get("card_activation_state")
            if _has_value(values, "card_activation_state") and activation != "not_applicable":
                invalid_card_fields.append("card_activation_state")
            if invalid_card_fields:
                findings.append(
                    _finding(
                        "PERSONAL_BRIEF_AUDIT_ACCOUNT_RELATION_CONFLICT",
                        "A non-card account contains card-only business fields.",
                        dataset=dataset_name,
                        record_id=record_id,
                        fields=invalid_card_fields,
                        source_pages=pages,
                    )
                )
        if values.get("current_overdue") is True and values.get("ever_overdue") is not True:
            findings.append(
                _finding(
                    "PERSONAL_BRIEF_AUDIT_ACCOUNT_RELATION_CONFLICT",
                    "A currently overdue account is not marked as ever overdue.",
                    dataset=dataset_name,
                    record_id=record_id,
                    fields=("current_overdue", "ever_overdue"),
                    source_pages=pages,
                )
            )
        if values.get("over_90_days") is True and values.get("ever_overdue") is not True:
            findings.append(
                _finding(
                    "PERSONAL_BRIEF_AUDIT_ACCOUNT_RELATION_CONFLICT",
                    "An account with over-90-day delinquency is not marked as ever overdue.",
                    dataset=dataset_name,
                    record_id=record_id,
                    fields=("over_90_days", "ever_overdue"),
                    source_pages=pages,
                )
            )
        for amount_field, status_field in _ACCOUNT_AMOUNT_STATUS_FIELDS:
            amount_present = _has_value(values, amount_field)
            status = values.get(status_field)
            consistent = (status == "reported" and amount_present) or (
                status in {"not_reported", "not_applicable"} and not amount_present
            )
            if not consistent:
                findings.append(
                    _finding(
                        "PERSONAL_BRIEF_AUDIT_AMOUNT_STATUS_CONFLICT",
                        "Amount presence disagrees with its reporting-status field.",
                        dataset=dataset_name,
                        record_id=record_id,
                        fields=(amount_field, status_field),
                        source_pages=pages,
                    )
                )
    return findings


def _collect_status_value_relations(
    public_datasets: Mapping[str, Mapping[str, Any]],
    rich_rows: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[PersonalBriefAuditFinding]:
    rules = (
        ("civil_judgment_records", "cause", "cause_status"),
        ("enforcement_records", "cause", "cause_status"),
        (
            "administrative_penalty_records",
            "administrative_review_result",
            "administrative_review_result_status",
        ),
    )
    findings: list[PersonalBriefAuditFinding] = []
    for dataset_name, value_field, status_field in rules:
        for index, row in enumerate(_rows(public_datasets.get(dataset_name)), start=1):
            values = _normalized(row)
            if not _has_value(values, value_field) and not _has_value(values, status_field):
                continue
            record_id = _record_id(row, index, dataset_name)
            value_present = _has_value(values, value_field)
            expected = "reported" if value_present else "not_reported"
            if values.get(status_field) == expected:
                continue
            findings.append(
                _finding(
                    "PERSONAL_BRIEF_AUDIT_STATUS_VALUE_CONFLICT",
                    "Business value presence disagrees with its reporting-status field.",
                    dataset=dataset_name,
                    record_id=record_id,
                    fields=(value_field, status_field),
                    source_pages=_source_pages(
                        rich_rows.get((dataset_name, record_id), {})
                    ),
                )
            )
    dataset_name = "repayment_liability_records"
    for index, row in enumerate(_rows(public_datasets.get(dataset_name)), start=1):
        values = _normalized(row)
        if not _has_value(values, "responsibility_amount") and not _has_value(
            values, "responsibility_amount_reported"
        ):
            continue
        record_id = _record_id(row, index, dataset_name)
        expected = _has_value(values, "responsibility_amount")
        if values.get("responsibility_amount_reported") is expected:
            continue
        findings.append(
            _finding(
                "PERSONAL_BRIEF_AUDIT_STATUS_VALUE_CONFLICT",
                "Responsibility-amount presence disagrees with its reported flag.",
                dataset=dataset_name,
                record_id=record_id,
                fields=("responsibility_amount", "responsibility_amount_reported"),
                source_pages=_source_pages(rich_rows.get((dataset_name, record_id), {})),
            )
        )
    return findings


def _collect_overdue_relations(
    public_datasets: Mapping[str, Mapping[str, Any]],
    rich_rows: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[PersonalBriefAuditFinding]:
    account_dataset = "credit_accounts"
    overdue_dataset = "overdue_records"
    account_rows = _rows(public_datasets.get(account_dataset))
    overdue_rows = _rows(public_datasets.get(overdue_dataset))
    findings: list[PersonalBriefAuditFinding] = []
    account_candidates: defaultdict[
        str,
        list[tuple[Mapping[str, Any], str, tuple[int, ...]]],
    ] = defaultdict(list)
    for index, row in enumerate(account_rows, start=1):
        values = _normalized(row)
        account_id = str(values.get("account_id") or "")
        record_id = _record_id(row, index, account_dataset)
        pages = _source_pages(rich_rows.get((account_dataset, record_id), {}))
        if not account_id:
            findings.append(
                _finding(
                    "PERSONAL_BRIEF_AUDIT_ACCOUNT_ID_CONFLICT",
                    "The credit account has no relational account_id.",
                    dataset=account_dataset,
                    record_id=record_id,
                    fields=("account_id",),
                    source_pages=pages,
                )
            )
            continue
        account_candidates[account_id].append(
            (
                values,
                record_id,
                pages,
            )
        )
    for account_id, candidates in sorted(account_candidates.items()):
        if len(candidates) <= 1:
            continue
        findings.append(
            _finding(
                "PERSONAL_BRIEF_AUDIT_ACCOUNT_ID_CONFLICT",
                f"The relational account_id occurs {len(candidates)} times.",
                dataset=account_dataset,
                record_id=candidates[0][1],
                fields=("account_id",),
                source_pages=sorted(
                    {page for _, _, pages in candidates for page in pages}
                ),
            )
        )
    accounts = {
        account_id: candidates[0]
        for account_id, candidates in account_candidates.items()
        if len(candidates) == 1
    }
    overdue_by_account: defaultdict[
        str,
        list[tuple[Mapping[str, Any], str, tuple[int, ...]]],
    ] = defaultdict(list)
    for index, row in enumerate(overdue_rows, start=1):
        values = _normalized(row)
        account_id = str(values.get("account_id") or "")
        record_id = _record_id(row, index, overdue_dataset)
        pages = _source_pages(rich_rows.get((overdue_dataset, record_id), {}))
        overdue_by_account[account_id].append((values, record_id, pages))
        account = accounts.get(account_id)
        if account is None:
            findings.append(
                _finding(
                    "PERSONAL_BRIEF_AUDIT_ORPHAN_OVERDUE_RECORD",
                    "The overdue record does not resolve to exactly one credit account.",
                    dataset=overdue_dataset,
                    record_id=record_id,
                    fields=("account_id",),
                    source_pages=pages,
                )
            )
            continue
        account_values = account[0]
        mismatches = [
            field_name
            for field_name in _OVERDUE_IDENTITY_FIELDS
            if values.get(field_name) != account_values.get(field_name)
        ]
        if values.get("currency") != account_values.get("account_currency"):
            mismatches.append("currency")
        if mismatches:
            findings.append(
                _finding(
                    "PERSONAL_BRIEF_AUDIT_OVERDUE_STATE_CONFLICT",
                    "Overdue account identity does not agree with the linked credit account.",
                    dataset=overdue_dataset,
                    record_id=record_id,
                    fields=mismatches,
                    source_pages=pages,
                )
            )
        if account_values.get("ever_overdue") is not True:
            findings.append(
                _finding(
                    "PERSONAL_BRIEF_AUDIT_OVERDUE_STATE_CONFLICT",
                    "An overdue row links to an account not marked as ever overdue.",
                    dataset=overdue_dataset,
                    record_id=record_id,
                    fields=("ever_overdue",),
                    source_pages=pages,
                )
            )
        for field_name in ("overdue_months", "over_90_days", "current_overdue"):
            overdue_present = _has_value(values, field_name)
            account_present = _has_value(account_values, field_name)
            if overdue_present != account_present or (
                overdue_present and values.get(field_name) != account_values.get(field_name)
            ):
                findings.append(
                    _finding(
                        "PERSONAL_BRIEF_AUDIT_OVERDUE_STATE_CONFLICT",
                        "Overdue facts disagree with the linked credit account.",
                        dataset=overdue_dataset,
                        record_id=record_id,
                        fields=(field_name,),
                        source_pages=pages,
                    )
                )
        status = values.get("current_overdue_status")
        current_present = _has_value(values, "current_overdue")
        current = values.get("current_overdue")
        status_consistent = (
            (status == "overdue" and current is True)
            or (status == "not_overdue" and current is False)
            or (status == "not_reported" and not current_present)
        )
        if not status_consistent:
            findings.append(
                _finding(
                    "PERSONAL_BRIEF_AUDIT_OVERDUE_STATE_CONFLICT",
                    "Current-overdue status disagrees with the current-overdue flag.",
                    dataset=overdue_dataset,
                    record_id=record_id,
                    fields=("current_overdue", "current_overdue_status"),
                    source_pages=pages,
                )
            )
        overdue_months = (
            _nonnegative_int(values.get("overdue_months"))
            if _has_value(values, "overdue_months")
            else None
        )
        over_90_months = (
            _nonnegative_int(values.get("over_90_days_months"))
            if _has_value(values, "over_90_days_months")
            else None
        )
        invalid_count_fields = [
            field_name
            for field_name, parsed in (
                ("overdue_months", overdue_months),
                ("over_90_days_months", over_90_months),
            )
            if _has_value(values, field_name) and parsed is None
        ]
        if invalid_count_fields:
            findings.append(
                _finding(
                    "PERSONAL_BRIEF_AUDIT_OVERDUE_STATE_CONFLICT",
                    "An overdue month count is not a nonnegative integer.",
                    dataset=overdue_dataset,
                    record_id=record_id,
                    fields=invalid_count_fields,
                    source_pages=pages,
                )
            )
        if (
            overdue_months is not None
            and over_90_months is not None
            and over_90_months > overdue_months
        ):
            findings.append(
                _finding(
                    "PERSONAL_BRIEF_AUDIT_OVERDUE_STATE_CONFLICT",
                    "Over-90-day months exceed total overdue months.",
                    dataset=overdue_dataset,
                    record_id=record_id,
                    fields=("overdue_months", "over_90_days_months"),
                    source_pages=pages,
                )
            )
        if over_90_months is not None and values.get("over_90_days") is not (over_90_months > 0):
            findings.append(
                _finding(
                    "PERSONAL_BRIEF_AUDIT_OVERDUE_STATE_CONFLICT",
                    "Over-90-day flag disagrees with the reported month count.",
                    dataset=overdue_dataset,
                    record_id=record_id,
                    fields=("over_90_days", "over_90_days_months"),
                    source_pages=pages,
                )
            )
    for account_id, rows in sorted(overdue_by_account.items()):
        if account_id and len(rows) > 1:
            findings.append(
                _finding(
                    "PERSONAL_BRIEF_AUDIT_DUPLICATE_OVERDUE_RECORD",
                    f"The account has {len(rows)} overdue rows; the canonical grain permits one.",
                    dataset=overdue_dataset,
                    record_id=rows[0][1],
                    fields=("account_id",),
                    source_pages=sorted({page for _, _, pages in rows for page in pages}),
                )
            )
    for account_id, (values, record_id, pages) in accounts.items():
        if values.get("ever_overdue") is True and len(overdue_by_account.get(account_id, ())) != 1:
            findings.append(
                _finding(
                    "PERSONAL_BRIEF_AUDIT_OVERDUE_STATE_CONFLICT",
                    "An account marked as ever overdue does not have exactly one overdue row.",
                    dataset=account_dataset,
                    record_id=record_id,
                    fields=("account_id", "ever_overdue"),
                    source_pages=pages,
                )
            )
    return findings


def _collect_section_status(
    public_payload: Mapping[str, Any],
    public_datasets: Mapping[str, Mapping[str, Any]],
    semantic_datasets: Mapping[str, Mapping[str, Any]],
) -> list[PersonalBriefAuditFinding]:
    findings: list[PersonalBriefAuditFinding] = []
    for section in public_payload.get("sections") or ():
        if not isinstance(section, Mapping):
            continue
        section_id = str(section.get("id") or "")
        section_type = str(section.get("type") or "")
        owned_names = _SECTION_DATASETS_BY_TYPE.get(section_type, ())
        owned_datasets = [
            public_datasets[name] for name in owned_names if name in public_datasets
        ]
        owned_refs = {
            str(dataset.get("id") or "")
            for dataset in owned_datasets
            if dataset.get("id")
        }
        actual_refs = {
            str(dataset_id) for dataset_id in (section.get("dataset_refs") or ())
        }
        if owned_names and actual_refs != owned_refs:
            findings.append(
                _finding(
                    "PERSONAL_BRIEF_AUDIT_SECTION_STATUS_CONFLICT",
                    "Section dataset references do not match the plugin-owned section mapping.",
                    section_id=section_id,
                    fields=("dataset_refs",),
                    source_pages=section.get("page_range") or (),
                )
            )
        statuses = [
            item.get("value")
            for item in (section.get("items") or ())
            if isinstance(item, Mapping) and item.get("key") == "record_status"
        ]
        if not statuses:
            continue
        status = statuses[0]
        row_count = sum(len(_rows(dataset)) for dataset in owned_datasets)
        if status in {"no_records", "absent_from_report"} and row_count:
            findings.append(
                _finding(
                    "PERSONAL_BRIEF_AUDIT_SECTION_STATUS_CONFLICT",
                    f"record_status={status!r} conflicts with {row_count} emitted rows.",
                    section_id=section_id,
                    fields=("record_status",),
                    source_pages=section.get("page_range") or (),
                )
            )
        if status == "reported" and row_count == 0:
            findings.append(
                _finding(
                    "PERSONAL_BRIEF_AUDIT_SECTION_STATUS_CONFLICT",
                    "record_status='reported' has no emitted business rows.",
                    section_id=section_id,
                    fields=("record_status",),
                    source_pages=section.get("page_range") or (),
                )
            )
    aggregate = semantic_datasets.get("public_records")
    if aggregate is not None:
        aggregate_counts: Counter[str] = Counter(
            str(_normalized(row).get("record_type") or "") for row in _rows(aggregate)
        )
        for record_type, dataset_name in _PUBLIC_RECORD_DATASETS.items():
            typed_count = len(_rows(public_datasets.get(dataset_name)))
            if aggregate_counts.get(record_type, 0) == typed_count:
                continue
            findings.append(
                _finding(
                    "PERSONAL_BRIEF_AUDIT_PUBLIC_RECORD_CONSERVATION_CONFLICT",
                    f"Aggregate {record_type} count {aggregate_counts.get(record_type, 0)} "
                    f"does not equal typed count {typed_count}.",
                    dataset=dataset_name,
                )
            )
    return findings


def _collect_source_field_coverage(
    semantic_payload: Mapping[str, Any],
    public_payload: Mapping[str, Any],
) -> list[PersonalBriefAuditFinding]:
    """Observe source-to-public field obligations, including pre-projection loss."""

    domain = semantic_payload.get("domain") or {}
    facts = domain.get("facts") or {}
    report = facts.get("personal_brief_extraction_report") or {}
    coverage = report.get("source_field_coverage") or {}
    semantic_datasets = _datasets(semantic_payload)
    public_datasets = _datasets(public_payload)
    source_accounts = {
        (values.get("source_section"), _positive_int(values.get("source_sequence"))): row
        for row in _rows(semantic_datasets.get("credit_accounts"))
        for values in (_normalized(row),)
    }
    public_accounts = {
        str(row.get("record_id") or ""): row
        for row in _rows(public_datasets.get("credit_accounts"))
    }
    public_overdue = {
        str(_normalized(row).get("account_id") or ""): row
        for row in _rows(public_datasets.get("overdue_records"))
    }
    findings = []
    for requirement in coverage.get("accounts") or ():
        key = (requirement.get("source_section"), requirement.get("source_sequence"))
        rich = source_accounts.get(key, {})
        record_id = str(rich.get("record_id") or f"{key[0]}:{key[1]}")
        account = public_accounts.get(record_id, {})
        account_id = str(_normalized(rich).get("account_id") or "")
        for dataset_name, field_map in (requirement.get("fields") or {}).items():
            row = account if dataset_name == "credit_accounts" else public_overdue.get(account_id, {})
            values = _normalized(row)
            missing = [field for field in field_map if not _has_value(values, field)]
            if missing:
                findings.append(_finding(
                    "PERSONAL_BRIEF_AUDIT_SOURCE_FIELD_MISSING",
                    "A business clause printed in the source has no structured public value.",
                    dataset=dataset_name,
                    record_id=str(row.get("record_id") or record_id),
                    fields=missing,
                    source_pages=requirement.get("source_pages") or (),
                ))
    sections = {
        section.get("type"): section
        for section in public_payload.get("sections") or ()
        if isinstance(section, Mapping)
    }
    for requirement in coverage.get("sections") or ():
        section = sections.get(requirement.get("section_type"), {})
        values = {
            item.get("key"): item.get("value")
            for item in section.get("items") or ()
            if isinstance(item, Mapping)
        }
        changed = [
            field for field, expected in (requirement.get("fields") or {}).items()
            if values.get(field) != expected
        ]
        if changed:
            findings.append(_finding(
                "PERSONAL_BRIEF_AUDIT_SOURCE_SECTION_FIELD_MISSING",
                "A source section qualifier is missing or changed in Community JSON.",
                section_id=str(section.get("id") or ""),
                fields=changed,
                source_pages=requirement.get("source_pages") or (),
            ))
    return findings


def build_personal_brief_observational_findings(
    semantic_payload: Mapping[str, Any],
    public_payload: Mapping[str, Any],
) -> tuple[PersonalBriefAuditFinding, ...]:
    """Return deterministic findings without mutating either input payload."""

    semantic_datasets = _datasets(semantic_payload)
    public_datasets = _datasets(public_payload)
    rich_rows = _rich_row_lookup(semantic_datasets)
    findings = [
        *_collect_source_field_coverage(semantic_payload, public_payload),
        *_collect_projection_conservation(semantic_datasets, public_datasets),
        *_collect_dataset_envelopes(public_datasets),
        *_collect_provenance(public_datasets, rich_rows),
        *_collect_extraction_report_consistency(semantic_payload, public_datasets),
        *_collect_enum_contract(public_datasets, rich_rows),
        *_collect_money_contract(public_datasets, rich_rows),
        *_collect_identity(public_datasets, rich_rows),
        *_collect_summary(public_datasets, rich_rows),
        *_collect_account_relations(public_datasets, rich_rows),
        *_collect_status_value_relations(public_datasets, rich_rows),
        *_collect_overdue_relations(public_datasets, rich_rows),
        *_collect_section_status(public_payload, public_datasets, semantic_datasets),
    ]
    return _deduplicate_findings(findings)


def _warning_from_finding(
    finding: PersonalBriefAuditFinding,
    public_datasets: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    warning: dict[str, Any] = {
        "code": finding.code,
        "level": finding.level,
        "message": finding.contextual_message(),
    }
    dataset = public_datasets.get(finding.dataset)
    if dataset is not None:
        dataset_id = str(dataset.get("id") or "")
        section_id = str(dataset.get("section_id") or "")
        if dataset_id:
            warning["dataset_id"] = dataset_id
        if section_id:
            warning["section_id"] = section_id
    elif finding.section_id:
        warning["section_id"] = finding.section_id
    if finding.source_pages:
        warning["page_range"] = [
            min(finding.source_pages),
            max(finding.source_pages),
        ]
    return warning


def _warning_key(warning: Mapping[str, Any]) -> tuple[Any, ...]:
    page_range = warning.get("page_range")
    pages = (
        tuple(page_range)
        if isinstance(page_range, Sequence) and not isinstance(page_range, (str, bytes))
        else ()
    )
    return (
        warning.get("code"),
        warning.get("level"),
        warning.get("message"),
        warning.get("section_id"),
        warning.get("dataset_id"),
        pages,
    )


def append_personal_brief_observational_warnings(
    semantic_payload: Mapping[str, Any],
    public_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a copy of ``public_payload`` changed only by appended warnings.

    Audit runtime errors are fail-open: business data are retained and an
    explicit internal-error warning is appended for downstream review.
    """

    output = deepcopy(dict(public_payload))
    try:
        findings = build_personal_brief_observational_findings(
            semantic_payload,
            public_payload,
        )
        public_datasets = _datasets(output)
        warnings = [
            warning
            for warning in (output.get("warnings") or ())
            if isinstance(warning, Mapping)
        ]
        seen = {_warning_key(warning) for warning in warnings}
        for finding in findings:
            if finding.level not in {"warning", "error"}:
                continue
            warning = _warning_from_finding(finding, public_datasets)
            key = _warning_key(warning)
            if key in seen:
                continue
            warnings.append(warning)
            seen.add(key)
        output["warnings"] = warnings
        return output
    except Exception as exc:  # pragma: no cover - behavior covered via fault injection
        warnings = [
            warning
            for warning in (output.get("warnings") or ())
            if isinstance(warning, Mapping)
        ]
        message = (
            "The observational auditor failed internally; extracted business data "
            f"were retained unchanged ({type(exc).__name__})."
        )
        if not any(
            warning.get("code") == "PERSONAL_BRIEF_AUDIT_INTERNAL_ERROR"
            and warning.get("message") == message
            for warning in warnings
        ):
            warnings.append(
                {
                    "code": "PERSONAL_BRIEF_AUDIT_INTERNAL_ERROR",
                    "level": "warning",
                    "message": message,
                }
            )
        output["warnings"] = warnings
        return output


__all__ = [
    "PersonalBriefAuditFinding",
    "append_personal_brief_observational_warnings",
    "build_personal_brief_observational_findings",
]
