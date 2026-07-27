# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native credit-report facts without constructing an edition envelope."""

from __future__ import annotations

from typing import Any

from docmirror.plugins._base.projector import ProjectionData

_DEFAULT_RECORD_ID_KEYS = (
    "record_id",
    "account_id",
    "liability_id",
    "inquiry_id",
    "public_record_id",
    "note_id",
)
_REPAYMENT_RECORD_ID_KEYS = ("record_id", "repayment_id")
_DATASET_RECORD_ID_KEYS = {
    "credit_lines": ("record_id", "credit_line_id"),
    "enterprise_facility_summary": ("record_id", "credit_line_id"),
    "enterprise_closed_credit_summary": ("record_id", "closed_summary_id"),
    "enterprise_repayment_responsibility_summary": (
        "record_id",
        "responsibility_summary_id",
    ),
    "enterprise_attachment_accounts": ("record_id", "attachment_account_id"),
    "enterprise_credit_supplement": ("record_id", "supplement_id"),
    "enterprise_attachment_credit_details": ("record_id", "attachment_detail_id"),
    "enterprise_special_transactions": ("record_id", "special_transaction_id"),
}


def _records(dataset_id: str, values: Any) -> list[dict[str, Any]]:
    """Give projected business records stable canonical record identities."""
    rows: list[dict[str, Any]] = []
    id_keys = (
        _REPAYMENT_RECORD_ID_KEYS
        if dataset_id == "repayment_records"
        else _DATASET_RECORD_ID_KEYS.get(dataset_id, _DEFAULT_RECORD_ID_KEYS)
    )
    for index, value in enumerate(values or (), start=1):
        if not isinstance(value, dict):
            continue
        row = dict(value)
        identity = next((row.get(key) for key in id_keys if row.get(key)), None)
        row["record_id"] = str(identity or f"{dataset_id}:r{index:06d}")
        rows.append(row)
    return rows


def _account_structure_warnings(accounts: list[dict[str, Any]]) -> tuple[str, ...]:
    """Keep credit-account completeness policy inside the credit plugin."""
    if not accounts:
        return ()
    collapsed = 0
    for account in accounts:
        required = (
            bool(account.get("open_date")),
            bool(account.get("loan_amount") or account.get("credit_limit")),
            bool(account.get("management_institution")),
        )
        if sum(required) <= 1:
            collapsed += 1
    failure_rate = collapsed / len(accounts)
    if failure_rate <= 0.3:
        return ()
    return (f"credit:account_structure_collapse:failure_rate={failure_rate:.3f}",)


def _enterprise_scope_warnings(
    report_subtype: str,
    summary: dict[str, Any],
) -> tuple[str, ...]:
    if report_subtype != "enterprise" or not summary.get("source_display_limited"):
        return ()
    return (
        "credit:enterprise_source_display_limited:"
        "the source report states that only part of the credit records are shown; "
        "attachment records are exported in separate enterprise datasets",
    )


def derive_credit_report_projection(plugin: Any, parse_result: Any, full_text: str = "") -> ProjectionData:
    """Return identity, profile, section, and business datasets as one ProjectionData."""
    from docmirror.plugins._base.kv_community_enrich import (
        _canonicalize_credit_accounts,
        _domain_specific,
        _ensure_credit_repayment_records,
        _extract_credit_accounts_from_local_structure_evidence,
        _has_credit_repayment_structures,
        _recover_credit_subject_identity,
    )
    from docmirror.plugins._base.kv_projection import extract_kv_projection
    from docmirror.plugins.credit_report.business_assembly import assemble_credit_report_business
    from docmirror.plugins.credit_report.report_profile import (
        detect_credit_report_content_mode,
        detect_credit_report_subtype,
        recover_credit_report_header_fields,
    )
    from docmirror.plugins.credit_report.scanned_business import (
        link_repayment_records_to_accounts,
    )
    from docmirror.plugins.credit_report.semantic_enrichment import (
        enrich_credit_report_record_evidence,
    )
    from docmirror.plugins.credit_report.variant_router import (
        resolve_credit_report_variant,
    )

    base = extract_kv_projection(
        plugin,
        parse_result,
        identity_specs=plugin.identity_fields,
        full_text=full_text,
        include_block_kv=False,
        include_generic_records=False,
    )
    domain_facts = dict(base.domain_facts)
    field_details = dict(domain_facts.get("field_details") or {})

    for field_name, recovered in _recover_credit_subject_identity(parse_result).items():
        domain_facts.setdefault(field_name, recovered["value"])
        field_details.setdefault(
            field_name,
            {
                "source": "canonical_evidence_atoms",
                "page_id": recovered["page_id"],
                "evidence_ids": recovered["evidence_ids"],
            },
        )

    report_subtype = detect_credit_report_subtype(parse_result, full_text)
    content_mode = detect_credit_report_content_mode(parse_result)
    variant = resolve_credit_report_variant(report_subtype, content_mode)
    if not variant.keep_query_institution:
        domain_facts.pop("query_institution", None)
        field_details.pop("query_institution", None)
    recovered_header = recover_credit_report_header_fields(
        parse_result,
        full_text,
        report_subtype=report_subtype,
    )
    if report_subtype != "unknown":
        recovered_header.setdefault("report_subtype", report_subtype)
    if content_mode != "unknown":
        recovered_header.setdefault("content_mode", content_mode)
    for field_name, value in recovered_header.items():
        domain_facts[field_name] = value
        field_details[field_name] = {
            "source": "credit_report_header",
            "confidence": 0.95 if field_name not in {"report_subtype", "content_mode"} else 1.0,
        }
    variant.refine_domain_facts(domain_facts, field_details)
    domain_facts["field_details"] = field_details

    source_domain = _domain_specific(parse_result)
    scanned_business = variant.extract_auxiliary_business(
        parse_result,
        full_text,
        content_mode=content_mode,
    )

    repayment_records = list(source_domain.get("credit_repayment_records") or [])
    if not repayment_records and (
        content_mode in {"scanned_ocr", "mixed"} or _has_credit_repayment_structures(parse_result)
    ):
        repayment_records = _ensure_credit_repayment_records(parse_result)

    credit_accounts: list[dict[str, Any]] = []
    if variant.use_generic_credit_accounts():
        credit_accounts = _canonicalize_credit_accounts(list(scanned_business.get("credit_accounts") or []))
        if not credit_accounts:
            credit_accounts = _canonicalize_credit_accounts(list(source_domain.get("credit_accounts") or []))
        if not credit_accounts:
            credit_accounts = _canonicalize_credit_accounts(
                _extract_credit_accounts_from_local_structure_evidence(parse_result)
            )

    from docmirror.models.mirror.domain_access import micro_grid_structures_from_domain_specific

    repayment_records = link_repayment_records_to_accounts(
        repayment_records,
        credit_accounts,
        micro_grid_structures_from_domain_specific(source_domain),
    )
    assembled = assemble_credit_report_business(
        parse_result,
        full_text,
        report_subtype=report_subtype,
        content_mode=content_mode,
        existing_collections={
            "credit_accounts": credit_accounts,
            "credit_lines": [],
            "repayment_liability_records": [],
            "repayment_records": repayment_records,
            "overdue_records": [],
            "inquiry_records": list(scanned_business.get("inquiry_records") or []),
            "public_records": [],
        },
        existing_summary=dict(scanned_business.get("credit_summary") or {}),
    )
    dataset_names = variant.dataset_names()
    datasets = {name: rows for name in dataset_names if (rows := _records(name, assembled.get(name)))}
    if assembled.get("credit_summary"):
        domain_facts["credit_summary"] = dict(assembled["credit_summary"])
    if assembled.get("credit_extraction_audit"):
        domain_facts["credit_extraction_audit"] = dict(assembled["credit_extraction_audit"])
    section_content = variant.build_section_content(parse_result, full_text)
    if section_content:
        supplemental_facts = section_content.get("facts")
        if isinstance(supplemental_facts, dict):
            domain_facts.update(supplemental_facts)
        for fact_name in ("non_credit_transaction_summary", "public_record_summary"):
            value = section_content.get(fact_name)
            if isinstance(value, dict) and value.get("source_statement"):
                domain_facts[fact_name] = value
        report_notes = list(section_content.get("report_notes") or [])
        if report_notes:
            enrich_credit_report_record_evidence(parse_result, {"report_notes": report_notes})
            datasets["report_notes"] = _records("report_notes", report_notes)
        supplemental_datasets = section_content.get("datasets")
        if isinstance(supplemental_datasets, dict):
            for dataset_name, records in supplemental_datasets.items():
                typed_records = [record for record in (records or []) if isinstance(record, dict)]
                if not typed_records:
                    continue
                enrich_credit_report_record_evidence(parse_result, {str(dataset_name): typed_records})
                datasets[str(dataset_name)] = _records(str(dataset_name), typed_records)
    domain_facts["data_dictionary"] = variant.data_dictionary()
    evidence_ids = tuple(
        dict.fromkeys(
            [
                *base.evidence_ids,
                *[
                    str(evidence_id)
                    for rows in datasets.values()
                    for row in rows
                    for evidence_id in (row.get("evidence_ids") or [])
                    if evidence_id
                ],
            ]
        )
    )

    entity_fields = dict(base.entity_fields)
    if domain_facts.get("subject_name"):
        entity_fields["subject_name"] = domain_facts["subject_name"]
    if domain_facts.get("id_number"):
        entity_fields["subject_id"] = domain_facts["id_number"]
    if domain_facts.get("marital_status"):
        entity_fields["marital_status"] = domain_facts["marital_status"]
    variant.refine_entity_fields(entity_fields)
    assembled_accounts = list(assembled.get("credit_accounts") or [])
    assembled_summary = dict(assembled.get("credit_summary") or {})
    warnings = tuple(
        dict.fromkeys(
            (
                *base.warnings,
                *_account_structure_warnings(assembled_accounts),
                *_enterprise_scope_warnings(report_subtype, assembled_summary),
            )
        )
    )
    return ProjectionData(
        projector_id=base.projector_id,
        document_type=base.document_type,
        entity_fields=entity_fields,
        domain_facts=domain_facts,
        semantic=variant.semantic_extensions(),
        datasets=datasets,
        sections=variant.build_sections(parse_result, full_text),
        warnings=warnings,
        evidence_ids=evidence_ids,
        confidence=base.confidence,
        reason="post-seal credit-report projection",
    )


__all__ = ["derive_credit_report_projection"]
