# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Document-local consistency checks for the final Candidate-B source plane.

The ledger runs after the one permitted page-level OCR correction and before
v2 projection.  It never consults a registry and never changes the sealed
ParseResult.  Its comparisons are deliberately narrow: independently bound
rows in this one report can expose a suspicious value, but a majority is not
treated as business truth.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, MutableMapping
from statistics import median
from typing import Any

_INSTITUTION_FIELDS: dict[str, tuple[str, ...]] = {
    "personal_report_metadata": ("query_institution",),
    "credit_accounts": ("management_institution",),
    "credit_lines": ("institution",),
    "repayment_liability_records": ("institution",),
    "inquiry_records": ("institution",),
    "residence_records": ("data_provider",),
    "employment_records": ("data_provider",),
    "mobile_phone_records": ("data_provider",),
    "spouse_records": ("data_provider",),
}

_IDENTITY_FIELDS: dict[str, tuple[str, ...]] = {
    "personal_report_metadata": ("personal_report_metadata_id",),
    "credit_accounts": ("account_id",),
    "credit_lines": ("credit_line_id", "credit_agreement_id"),
    "repayment_liability_records": ("liability_id", "repayment_responsibility_id"),
    "inquiry_records": ("inquiry_id",),
    "residence_records": ("residence_record_id",),
    "employment_records": ("employment_record_id",),
    "mobile_phone_records": ("mobile_phone_record_id",),
    "spouse_records": ("spouse_record_id",),
    "personal_detail_summary_cells": ("summary_cell_id",),
}

_DATASET_TARGET_ALIASES: dict[str, frozenset[str]] = {
    "personal_report_metadata": frozenset({"personal_report_metadata", "report_metadata", "report_query"}),
    "credit_accounts": frozenset({"credit_accounts"}),
    "credit_lines": frozenset({"credit_lines", "credit_agreements"}),
    "repayment_liability_records": frozenset(
        {"repayment_liability_records", "repayment_responsibilities"}
    ),
    "inquiry_records": frozenset({"inquiry_records", "inquiries"}),
    "residence_records": frozenset({"residence_records", "subject_residences"}),
    "employment_records": frozenset({"employment_records", "subject_employment"}),
    "mobile_phone_records": frozenset({"mobile_phone_records", "subject_mobile_phones"}),
    "spouse_records": frozenset({"spouse_records", "subject_spouse"}),
    "personal_detail_summary_cells": frozenset(
        {"personal_detail_summary_cells", "credit_business_overview"}
    ),
}

_LEGAL_ROOT_MARKERS = (
    "银行股份有限公司",
    "银行有限责任公司",
    "小额贷款有限公司",
    "消费金融有限公司",
    "汽车金融有限公司",
    "金融租赁有限公司",
    "保险股份有限公司",
    "证券股份有限公司",
    "股份有限公司",
    "有限责任公司",
    "农村商业银行",
    "农村合作银行",
    "商业银行",
    "信用合作联社",
    "农村信用合作社",
    "住房公积金管理中心",
    "征信中心",
    "有限公司",
    "信用社",
    "联社",
    "信托",
    "银行",
    "公司",
)
_BRANCH_SUFFIXES = ("分行", "支行", "中心")
_STALE_RESOLVED_FIELD_ISSUES = frozenset(
    {
        "pboc_cell_contract_unresolved",
        "candidate_b_exact_slot_value_invalid",
        "candidate_b_inquiry_required_cell_unresolved",
        "candidate_b_inquiry_institution_unresolved",
    }
)
_FAMILY_ACCOUNT_TYPES: dict[str, frozenset[str]] = {
    "非循环贷账户": frozenset({"non_revolving_loan"}),
    "循环贷账户一": frozenset({"revolving_loan", "revolving_loan_subaccount"}),
    "循环贷账户二": frozenset({"revolving_loan_account"}),
    "贷记卡账户": frozenset({"credit_card"}),
    "准贷记卡账户": frozenset({"quasi_credit_card"}),
}
_ADDRESS_DIRECTION_GLYPHS = frozenset("东西南北前后上下内外左右甲乙丙丁")


def _values(record: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    normalized = record.get("normalized")
    return normalized if isinstance(normalized, MutableMapping) else record


def _compact(value: Any) -> str:
    return "".join(str(value or "").split())


def _record_id(dataset: str, record: Mapping[str, Any], index: int) -> str:
    values = record.get("normalized") if isinstance(record.get("normalized"), Mapping) else record
    for field_name in ("record_id", *_IDENTITY_FIELDS.get(dataset, ())):
        value = record.get(field_name)
        if value in (None, "") and isinstance(values, Mapping):
            value = values.get(field_name)
        if value not in (None, ""):
            return str(value)
    if isinstance(values, Mapping):
        for field_name, value in values.items():
            if field_name.endswith("_id") and value not in (None, ""):
                return str(value)
    return f"{dataset}:{index}"


def _field_refs(record: Mapping[str, Any], field_name: str) -> list[dict[str, Any]]:
    values = record.get("normalized") if isinstance(record.get("normalized"), Mapping) else record
    refs: Any = None
    for owner in (values, record):
        if not isinstance(owner, Mapping):
            continue
        refs_by_field = owner.get("source_refs_by_field")
        if isinstance(refs_by_field, Mapping) and refs_by_field.get(field_name):
            refs = refs_by_field[field_name]
            break
    if not refs:
        for owner in (values, record):
            if not isinstance(owner, Mapping):
                continue
            refs = owner.get("source_refs") or owner.get("source_cell_refs")
            if refs:
                break
    return [dict(ref) for ref in refs or () if isinstance(ref, Mapping)]


def _raw_field_values(record: Mapping[str, Any], field_name: str) -> list[str]:
    observed: list[str] = []
    values = record.get("normalized") if isinstance(record.get("normalized"), Mapping) else record
    for owner in (record, values):
        if not isinstance(owner, Mapping):
            continue
        for pool_name in ("canonical_raw", "raw"):
            pool = owner.get(pool_name)
            value = pool.get(field_name) if isinstance(pool, Mapping) else None
            if isinstance(value, str) and value.strip():
                observed.append(value.strip())
            elif isinstance(value, (list, tuple)):
                observed.extend(str(item).strip() for item in value if str(item).strip())
    return list(dict.fromkeys(observed))


def _preserve_raw(record: MutableMapping[str, Any], field_name: str, value: Any) -> None:
    pool = record.get("canonical_raw")
    if not isinstance(pool, MutableMapping):
        pool = {}
        record["canonical_raw"] = pool
    pool.setdefault(field_name, value)


def _mark_uncertain(record: MutableMapping[str, Any], field_name: str) -> None:
    values = _values(record)
    unresolved = values.get("_unresolved_fields")
    if not isinstance(unresolved, list):
        unresolved = list(unresolved or ())
    if field_name not in unresolved:
        unresolved.append(field_name)
    values["_unresolved_fields"] = unresolved
    values["extraction_status"] = "review"


def _mark_reported_conflict(record: MutableMapping[str, Any], field_name: str) -> None:
    values = _values(record)
    conflicts = values.get("_reported_field_conflicts")
    if not isinstance(conflicts, list):
        conflicts = list(conflicts or ())
    if field_name not in conflicts:
        conflicts.append(field_name)
    values["_reported_field_conflicts"] = conflicts
    values["extraction_status"] = "review"


def _clear_resolved_marker(record: MutableMapping[str, Any], field_name: str) -> None:
    values = _values(record)
    unresolved = [
        str(value)
        for value in values.get("_unresolved_fields") or ()
        if str(value) != field_name
    ]
    if unresolved:
        values["_unresolved_fields"] = unresolved
    else:
        values.pop("_unresolved_fields", None)


def _legal_root(value: Any) -> str | None:
    text = _compact(value)
    if not text:
        return None
    candidates: list[tuple[int, int, str]] = []
    for marker in _LEGAL_ROOT_MARKERS:
        start = text.find(marker)
        if start >= 0:
            end = start + len(marker)
            candidates.append((end, len(marker), text[:end]))
    if not candidates:
        return None
    # The furthest legal ending owns any following branch name.  Marker length
    # breaks ties so 股份有限公司 outranks the contained 有限公司/公司.
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def _valid_complete_institution(value: Any) -> bool:
    text = _compact(value)
    if len(text) < 4 or len(text) > 96:
        return False
    if re.search(r"\d{4}[./-]\d", text) or any(char in text for char in "，,;；|[]{}"):
        return False
    return _legal_root(text) is not None


def _branch_fragment_without_root(value: Any) -> bool:
    text = _compact(value)
    return bool(text and text.endswith(_BRANCH_SUFFIXES) and _legal_root(text) is None)


def _separated_prefix_candidate(value: Any) -> tuple[str, str] | None:
    match = re.fullmatch(r"\s*([\u3400-\u9fff])\s+(.+?)\s*", str(value or ""))
    if match is None:
        return None
    remainder = _compact(match.group(2))
    if not _valid_complete_institution(remainder):
        return None
    return match.group(1), remainder


def _one_han_substitution(left: str, right: str) -> tuple[int, str, str] | None:
    if len(left) != len(right) or left == right:
        return None
    differences = [index for index, pair in enumerate(zip(left, right, strict=True)) if pair[0] != pair[1]]
    if len(differences) != 1:
        return None
    index = differences[0]
    left_char, right_char = left[index], right[index]
    if not ("\u3400" <= left_char <= "\u9fff" and "\u3400" <= right_char <= "\u9fff"):
        return None
    return index, left_char, right_char


def _institution_pair_is_structural(left: str, right: str) -> bool:
    difference = _one_han_substitution(left, right)
    if difference is None or not (_valid_complete_institution(left) and _valid_complete_institution(right)):
        return False
    index, _left_char, _right_char = difference
    return index >= 3 and len(left) - index - 1 >= 6


def _address_pair_is_structural(left: str, right: str) -> bool:
    difference = _one_han_substitution(left, right)
    if difference is None:
        return False
    index, left_char, right_char = difference
    if left_char in _ADDRESS_DIRECTION_GLYPHS or right_char in _ADDRESS_DIRECTION_GLYPHS:
        return False
    if index < 6 or len(left) - index - 1 < 8:
        return False
    left_numbers = re.findall(r"\d+", left)
    right_numbers = re.findall(r"\d+", right)
    return bool(left_numbers and left_numbers == right_numbers)


def _field_confidence(record: Mapping[str, Any], field_name: str) -> float | None:
    values = record.get("normalized") if isinstance(record.get("normalized"), Mapping) else record
    for owner in (values, record):
        if not isinstance(owner, Mapping):
            continue
        by_field = owner.get("confidence_by_field") or owner.get("field_confidence")
        value = by_field.get(field_name) if isinstance(by_field, Mapping) else None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return max(0.0, min(1.0, float(value)))
    for owner in (values, record):
        value = owner.get("confidence") if isinstance(owner, Mapping) else None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return max(0.0, min(1.0, float(value)))
    return None


def _materially_weaker(
    outlier: Mapping[str, Any],
    majority: Iterable[Mapping[str, Any]],
    *,
    field_name: str,
) -> bool:
    outlier_confidence = _field_confidence(outlier, field_name)
    majority_confidences = [
        confidence
        for record in majority
        if (confidence := _field_confidence(record, field_name)) is not None
    ]
    return bool(
        outlier_confidence is not None
        and len(majority_confidences) >= 2
        and median(majority_confidences) >= 0.80
        and median(majority_confidences) - outlier_confidence >= 0.20
    )


def _record_issue(
    context: Any,
    *,
    category: str,
    issue_code: str,
    message: str,
    dataset: str,
    record_id: str,
    field_name: str,
    observed_value: Any,
    candidate_value: Any | None = None,
    source_refs: Iterable[Mapping[str, Any]] = (),
    reason_codes: Iterable[str] = (),
    severity: str = "warning",
    status: str = "requires_review",
) -> None:
    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
        make_issue,
        record_issue,
    )

    record_issue(
        context,
        make_issue(
            category=category,
            issue_code=issue_code,
            message=message,
            severity=severity,
            status=status,
            parser_stage="candidate_b_document_consistency_ledger",
            target_dataset=dataset,
            target_record_id=record_id,
            field_name=field_name,
            observed_value=observed_value,
            candidate_value=candidate_value,
            source_refs=source_refs,
            reason_codes=reason_codes,
        ),
    )


def _prune_stale_resolved_field_issues(
    context: Any,
    *,
    dataset: str,
    record_id: str,
    field_name: str,
) -> None:
    issues = getattr(context, "_personal_detail_extraction_issues", None)
    if not isinstance(issues, list):
        return
    aliases = _DATASET_TARGET_ALIASES.get(dataset, frozenset({dataset}))
    context._personal_detail_extraction_issues = [
        issue
        for issue in issues
        if not (
            isinstance(issue, Mapping)
            and str(issue.get("target_dataset") or "") in aliases
            and str(issue.get("target_record_id") or "") == record_id
            and str(issue.get("field_name") or "") == field_name
            and str(issue.get("issue_code") or "") in _STALE_RESOLVED_FIELD_ISSUES
        )
    ]


def _institution_observations(
    datasets: Mapping[str, Any],
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for dataset, field_names in _INSTITUTION_FIELDS.items():
        for index, raw_record in enumerate(datasets.get(dataset) or (), start=1):
            if not isinstance(raw_record, MutableMapping):
                continue
            values = _values(raw_record)
            for field_name in field_names:
                value = values.get(field_name)
                refs = _field_refs(raw_record, field_name)
                if value in (None, "") or not refs:
                    continue
                observations.append(
                    {
                        "dataset": dataset,
                        "record": raw_record,
                        "record_id": _record_id(dataset, raw_record, index),
                        "field_name": field_name,
                        "value": str(value),
                        "key": _compact(value),
                        "refs": refs,
                    }
                )
    return observations


def _institution_root_witnesses(datasets: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    witnesses: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str, str]] = set()
    for dataset, field_names in _INSTITUTION_FIELDS.items():
        for index, raw_record in enumerate(datasets.get(dataset) or (), start=1):
            if not isinstance(raw_record, MutableMapping):
                continue
            record_id = _record_id(dataset, raw_record, index)
            for field_name in field_names:
                refs = _field_refs(raw_record, field_name)
                if not refs:
                    continue
                values = _values(raw_record)
                candidates = [values.get(field_name), *_raw_field_values(raw_record, field_name)]
                for candidate in candidates:
                    if candidate in (None, "") or _separated_prefix_candidate(candidate) is not None:
                        continue
                    compact = _compact(candidate)
                    if not _valid_complete_institution(compact):
                        continue
                    root = _legal_root(compact)
                    marker = (root or "", dataset, record_id)
                    if not root or marker in seen:
                        continue
                    seen.add(marker)
                    witnesses[root].append(
                        {
                            "dataset": dataset,
                            "record_id": record_id,
                            "field_name": field_name,
                            "value": compact,
                            "refs": refs,
                        }
                    )
    return witnesses


def _repair_separated_institution_prefixes(
    context: Any,
    datasets: MutableMapping[str, Any],
    audit: Counter[str],
) -> None:
    witnesses_by_root = _institution_root_witnesses(datasets)
    for dataset, field_names in _INSTITUTION_FIELDS.items():
        for index, raw_record in enumerate(datasets.get(dataset) or (), start=1):
            if not isinstance(raw_record, MutableMapping):
                continue
            values = _values(raw_record)
            record_id = _record_id(dataset, raw_record, index)
            for field_name in field_names:
                if values.get(field_name) not in (None, ""):
                    continue
                candidates = [
                    (raw, parsed)
                    for raw in _raw_field_values(raw_record, field_name)
                    if (parsed := _separated_prefix_candidate(raw)) is not None
                ]
                distinct = {(prefix, remainder) for _raw, (prefix, remainder) in candidates}
                if len(distinct) != 1:
                    continue
                prefix, remainder = next(iter(distinct))
                root = _legal_root(remainder)
                witnesses = [
                    witness
                    for witness in witnesses_by_root.get(root or "", ())
                    if not (
                        witness["dataset"] == dataset and witness["record_id"] == record_id
                    )
                ]
                refs = _field_refs(raw_record, field_name)
                raw = candidates[0][0]
                if root and len({(item["dataset"], item["record_id"]) for item in witnesses}) >= 2:
                    values[field_name] = remainder
                    _preserve_raw(raw_record, field_name, raw)
                    _clear_resolved_marker(raw_record, field_name)
                    _prune_stale_resolved_field_issues(
                        context,
                        dataset=dataset,
                        record_id=record_id,
                        field_name=field_name,
                    )
                    _record_issue(
                        context,
                        category="ocr_cell_level_error",
                        issue_code="candidate_b_document_local_institution_prefix_resolved",
                        message=(
                            "One separated OCR glyph was removed only because the remaining complete "
                            "institution legal root had two independent source-bound witnesses in this report."
                        ),
                        dataset=dataset,
                        record_id=record_id,
                        field_name=field_name,
                        observed_value={"raw": raw, "separated_glyph": prefix},
                        candidate_value={
                            "normalized_institution": remainder,
                            "legal_root": root,
                            "independent_witness_count": len(witnesses),
                        },
                        source_refs=[
                            *refs,
                            *[
                                {**ref, "consistency_witness_record_id": witness["record_id"]}
                                for witness in witnesses[:2]
                                for ref in witness["refs"][:1]
                            ],
                        ],
                        reason_codes=(
                            "single_separated_han_glyph_residue",
                            "complete_institution_remainder",
                            "document_local_legal_root_corroborated_twice",
                            "professional_field_correction",
                        ),
                        severity="info",
                        status="resolved",
                    )
                    audit["institution_prefix_resolved"] += 1
                    continue

                values.pop(field_name, None)
                _preserve_raw(raw_record, field_name, raw)
                _mark_uncertain(raw_record, field_name)
                _record_issue(
                    context,
                    category="ocr_cell_level_error",
                    issue_code="candidate_b_document_local_institution_prefix_unresolved",
                    message=(
                        "An institution field had a separated leading glyph, but the complete remainder "
                        "lacked two independent document-local legal-root witnesses."
                    ),
                    dataset=dataset,
                    record_id=record_id,
                    field_name=field_name,
                    observed_value={"raw": raw, "separated_glyph": prefix},
                    candidate_value={
                        "withheld_remainder": remainder,
                        "legal_root": root,
                        "independent_witness_count": len(witnesses),
                    },
                    source_refs=refs,
                    reason_codes=(
                        "single_separated_han_glyph_residue",
                        "document_local_legal_root_not_sufficiently_corroborated",
                        "normalized_value_withheld",
                    ),
                )
                audit["institution_prefix_unresolved"] += 1


def _reject_rootless_institution_branches(
    context: Any,
    datasets: MutableMapping[str, Any],
    audit: Counter[str],
) -> None:
    for dataset, field_names in _INSTITUTION_FIELDS.items():
        for index, raw_record in enumerate(datasets.get(dataset) or (), start=1):
            if not isinstance(raw_record, MutableMapping):
                continue
            values = _values(raw_record)
            record_id = _record_id(dataset, raw_record, index)
            for field_name in field_names:
                value = values.get(field_name)
                refs = _field_refs(raw_record, field_name)
                if not refs or not _branch_fragment_without_root(value):
                    continue
                _preserve_raw(raw_record, field_name, value)
                values.pop(field_name, None)
                _mark_uncertain(raw_record, field_name)
                _record_issue(
                    context,
                    category="ocr_structure_correction",
                    issue_code="candidate_b_institution_branch_without_legal_root",
                    message=(
                        "A branch-only institution fragment ended in 分行/支行/中心 without a bank, "
                        "company, or other legal organization root; the fragment was withheld."
                    ),
                    dataset=dataset,
                    record_id=record_id,
                    field_name=field_name,
                    observed_value=value,
                    source_refs=refs,
                    reason_codes=(
                        "institution_branch_suffix_observed",
                        "legal_organization_root_missing",
                        "normalized_value_withheld",
                    ),
                )
                audit["rootless_institution_branch_withheld"] += 1


def _apply_one_glyph_conflicts(
    context: Any,
    observations: list[dict[str, Any]],
    *,
    pair_is_structural: Any,
    issue_code: str,
    audit_key: str,
    audit: Counter[str],
) -> None:
    by_value: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for observation in observations:
        by_value[observation["key"]].append(observation)
    counts = Counter({value: len({item["record_id"] for item in rows}) for value, rows in by_value.items()})
    majority_values = [value for value, count in counts.items() if count >= 2]
    for outlier_value, outlier_rows in by_value.items():
        if counts[outlier_value] != 1:
            continue
        matching_majorities = [
            value
            for value in majority_values
            if pair_is_structural(outlier_value, value)
        ]
        if len(matching_majorities) != 1:
            continue
        majority_value = matching_majorities[0]
        outlier = outlier_rows[0]
        majority = by_value[majority_value]
        weaker = _materially_weaker(
            outlier["record"],
            (item["record"] for item in majority),
            field_name=outlier["field_name"],
        )
        raw_record = outlier["record"]
        values = _values(raw_record)
        if weaker:
            _preserve_raw(raw_record, outlier["field_name"], outlier["value"])
            values.pop(outlier["field_name"], None)
            _mark_uncertain(raw_record, outlier["field_name"])
            action = "withheld_materially_weaker_outlier"
            reasons = (
                "one_han_glyph_document_local_conflict",
                "unique_singleton_variant",
                "materially_weaker_source_evidence",
                "normalized_value_withheld",
            )
            audit[f"{audit_key}_withheld"] += 1
        else:
            _mark_reported_conflict(raw_record, outlier["field_name"])
            action = "retained_with_localized_uncertainty"
            reasons = (
                "one_han_glyph_document_local_conflict",
                "unique_singleton_variant",
                "majority_not_used_as_correction_authority",
                "candidate_value_retained_with_uncertainty",
            )
            audit[f"{audit_key}_retained_with_issue"] += 1
        _record_issue(
            context,
            category="ocr_cell_level_error",
            issue_code=issue_code,
            message=(
                "One independently bound individualized value differed by one Han glyph from a repeated "
                "document-local variant; no registry or majority correction was applied."
            ),
            dataset=outlier["dataset"],
            record_id=outlier["record_id"],
            field_name=outlier["field_name"],
            observed_value={
                "outlier_value": outlier["value"],
                "repeated_variant": majority[0]["value"],
                "outlier_row_count": 1,
                "repeated_variant_row_count": counts[majority_value],
            },
            candidate_value={"action": action},
            source_refs=[
                *outlier["refs"],
                *[
                    {**ref, "consistency_witness_record_id": item["record_id"]}
                    for item in majority[:2]
                    for ref in item["refs"][:1]
                ],
            ],
            reason_codes=reasons,
        )


def _check_individualized_conflicts(
    context: Any,
    datasets: MutableMapping[str, Any],
    audit: Counter[str],
) -> None:
    institutions = [
        observation
        for observation in _institution_observations(datasets)
        if _valid_complete_institution(observation["key"])
    ]
    _apply_one_glyph_conflicts(
        context,
        institutions,
        pair_is_structural=_institution_pair_is_structural,
        issue_code="candidate_b_document_local_institution_glyph_conflict",
        audit_key="institution_glyph_conflict",
        audit=audit,
    )

    addresses: list[dict[str, Any]] = []
    for index, raw_record in enumerate(datasets.get("residence_records") or (), start=1):
        if not isinstance(raw_record, MutableMapping):
            continue
        values = _values(raw_record)
        value = values.get("address")
        refs = _field_refs(raw_record, "address")
        if value in (None, "") or not refs:
            continue
        addresses.append(
            {
                "dataset": "residence_records",
                "record": raw_record,
                "record_id": _record_id("residence_records", raw_record, index),
                "field_name": "address",
                "value": str(value),
                "key": _compact(value),
                "refs": refs,
            }
        )
    _apply_one_glyph_conflicts(
        context,
        addresses,
        pair_is_structural=_address_pair_is_structural,
        issue_code="candidate_b_document_local_address_glyph_conflict",
        audit_key="address_glyph_conflict",
        audit=audit,
    )


def _summary_family(value: Any) -> str | None:
    compact = _compact(value)
    compact = re.sub(r"信息?汇总$", "", compact)
    return compact if compact in _FAMILY_ACCOUNT_TYPES else None


def _check_summary_account_counts(
    context: Any,
    datasets: MutableMapping[str, Any],
    audit: Counter[str],
) -> None:
    account_rows = [
        _values(record)
        for record in datasets.get("credit_accounts") or ()
        if isinstance(record, MutableMapping)
    ]
    account_ids = {
        str(row.get("account_id") or index)
        for index, row in enumerate(account_rows, start=1)
    }
    total_accounts = len(account_ids)
    family_counts = {
        family: sum(str(row.get("account_type") or "") in types for row in account_rows)
        for family, types in _FAMILY_ACCOUNT_TYPES.items()
    }
    for index, raw_record in enumerate(
        datasets.get("personal_detail_summary_cells") or (), start=1
    ):
        if not isinstance(raw_record, MutableMapping):
            continue
        values = _values(raw_record)
        if _compact(values.get("column_label")) not in {"账户数", "账户数量"}:
            continue
        raw_count = _compact(values.get("value"))
        if re.fullmatch(r"\d+", raw_count) is None:
            continue
        count = int(raw_count)
        family = _summary_family(values.get("summary_type")) or _summary_family(
            values.get("title")
        )
        family_count = family_counts.get(family) if family else None
        if count <= total_accounts and (family_count is None or count <= family_count):
            continue
        record_id = _record_id("personal_detail_summary_cells", raw_record, index)
        refs = _field_refs(raw_record, "value")
        observed = {
            "reported_account_count": count,
            "final_document_account_count": total_accounts,
            "summary_family": family,
            "final_family_account_count": family_count,
        }
        if count > total_accounts:
            values["value_status"] = "unreadable"
            _mark_uncertain(raw_record, "value")
            _record_issue(
                context,
                category="schema_incompleteness",
                issue_code="candidate_b_summary_account_count_exceeds_document_population",
                message=(
                    "A summary account count exceeded the complete final document account population; "
                    "the impossible scalar was withheld without substituting a family count."
                ),
                dataset="personal_detail_summary_cells",
                record_id=record_id,
                field_name="value",
                observed_value=observed,
                source_refs=refs,
                reason_codes=(
                    "summary_account_count_metric",
                    "count_exceeds_final_document_population",
                    "no_cross_dataset_value_invented",
                    "normalized_value_withheld",
                ),
            )
            audit["summary_count_withheld"] += 1
            continue

        _mark_reported_conflict(raw_record, "value")
        _record_issue(
            context,
            category="schema_incompleteness",
            issue_code="candidate_b_summary_account_count_exceeds_family_population",
            message=(
                "A summary family account count exceeded the final family population but not the document "
                "total; the source value was retained with localized uncertainty."
            ),
            dataset="personal_detail_summary_cells",
            record_id=record_id,
            field_name="value",
            observed_value=observed,
            source_refs=refs,
            reason_codes=(
                "summary_account_count_metric",
                "count_exceeds_final_family_population",
                "possible_family_extraction_gap",
                "candidate_value_retained_with_uncertainty",
            ),
        )
        audit["summary_count_retained_with_issue"] += 1


def apply_document_consistency_ledger(
    context: Any,
    datasets: MutableMapping[str, Any],
) -> dict[str, Any]:
    """Apply bounded, document-local checks to final corrected source rows."""

    audit: Counter[str] = Counter()
    _repair_separated_institution_prefixes(context, datasets, audit)
    _reject_rootless_institution_branches(context, datasets, audit)
    _check_individualized_conflicts(context, datasets, audit)
    _check_summary_account_counts(context, datasets, audit)
    return {
        "stage": "after_final_ocr_correction_before_v2_projection",
        "registry_used": False,
        "parse_result_mutated": False,
        **dict(sorted(audit.items())),
    }


__all__ = ["apply_document_consistency_ledger"]
