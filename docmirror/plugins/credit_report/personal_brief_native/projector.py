# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Post-seal projection for the canonical personal-brief pipeline."""

from __future__ import annotations

from typing import Any

from docmirror.plugins._base.projector import ProjectionData
from docmirror.plugins.credit_report.personal_brief_native.pipeline import (
    run_personal_brief_pipeline,
)
from docmirror.plugins.credit_report.projection import (
    _account_structure_warnings,
    _records,
)
from docmirror.plugins.credit_report.report_profile import (
    detect_credit_report_content_mode,
)


def derive_personal_brief_projection(
    plugin: Any,
    parse_result: Any,
    full_text: str = "",
) -> ProjectionData:
    """Run ParseResult -> canonical IR -> one rigid schema -> projection."""
    content_mode = detect_credit_report_content_mode(parse_result)
    artifacts = run_personal_brief_pipeline(parse_result, content_mode=content_mode)
    document = artifacts.document_ir
    semantic_document = artifacts.semantic_document
    from docmirror.plugins.credit_report.personal_brief_native.variant import variant

    raw_datasets = semantic_document.datasets
    # A source node may own one presentation section only.  Supplemental
    # datasets retain canonical unit/bbox/evidence provenance, while node
    # ownership is reserved for the authoritative presentation datasets.
    ownership_datasets = {
        "credit_accounts",
        "repayment_liability_records",
        "repayment_records",
        "overdue_records",
        "public_records",
        "inquiry_records",
        "report_notes",
    }
    for dataset_name, records in raw_datasets.items():
        if dataset_name in ownership_datasets:
            continue
        for record in records:
            for ref in record.get("source_refs") or []:
                if isinstance(ref, dict):
                    ref.pop("node_id", None)
                    ref.pop("node_ids", None)
    from docmirror.plugins.credit_report.business_assembly import (
        _NORMALIZED_FIELDS,
        _build_audit,
        _normalize_record,
    )

    datasets: dict[str, list[dict[str, Any]]] = {}
    for name, rows in raw_datasets.items():
        if not rows:
            continue
        if name in _NORMALIZED_FIELDS:
            normalized_rows: list[dict[str, Any]] = []
            for index, row in enumerate(rows):
                normalized = _normalize_record(name, row, index)
                normalized_payload = normalized.setdefault("normalized", {})
                if name == "public_records" and row.get("content") not in (None, ""):
                    # ``content`` is a typed nested business object.  The
                    # generic scalar normalizer intentionally unwraps dicts,
                    # so preserve this personal-brief contract explicitly.
                    normalized_payload["content"] = row["content"]
                if name == "credit_accounts":
                    for field in ("source_section", "source_sequence", "business_category"):
                        if row.get(field) not in (None, ""):
                            normalized_payload[field] = row[field]
                normalized_rows.append(normalized)
            datasets[name] = normalized_rows
        else:
            datasets[name] = _records(name, rows)
    audit = _build_audit(
        parse_result=document,
        full_text=document.full_text,
        report_subtype="personal_brief",
        content_mode=content_mode,
        collections={
            "credit_accounts": datasets.get("credit_accounts", []),
            "credit_lines": [],
            "repayment_liability_records": datasets.get("repayment_liability_records", []),
            "repayment_records": datasets.get("repayment_records", []),
            "overdue_records": datasets.get("overdue_records", []),
            "inquiry_records": datasets.get("inquiry_records", []),
            "public_records": datasets.get("public_records", []),
        },
        conflicts=[],
        credit_summary=semantic_document.credit_summary,
    )
    failures = semantic_document.extraction_report.get("failures") or []
    if failures:
        audit["issues"] = list(
            dict.fromkeys(
                [
                    *list(audit.get("issues") or []),
                    *[
                        f"canonical_extraction:{failure.get('code')}"
                        for failure in failures
                        if isinstance(failure, dict)
                    ],
                ]
            )
        )
        audit["status"] = "review"

    domain_facts = dict(semantic_document.facts)
    domain_facts["credit_summary"] = semantic_document.credit_summary
    domain_facts["credit_extraction_audit"] = audit
    domain_facts["personal_brief_extraction_report"] = semantic_document.extraction_report
    domain_facts["field_details"] = {
        key: {
            "source": "canonical_personal_brief_document_ir",
            "confidence": document.confidence,
        }
        for key, value in domain_facts.items()
        if key
        not in {
            "credit_summary",
            "credit_extraction_audit",
            "personal_brief_extraction_report",
            "canonical_section_presence",
        }
        and value not in (None, "")
    }
    domain_facts["data_dictionary"] = variant.data_dictionary()

    evidence_ids = tuple(
        dict.fromkeys(
            str(evidence_id)
            for rows in raw_datasets.values()
            for record in rows
            for evidence_id in (
                *list(record.get("evidence_ids") or []),
                *[
                    value
                    for ref in record.get("source_refs") or []
                    if isinstance(ref, dict)
                    for value in ref.get("evidence_ids") or []
                ],
            )
            if evidence_id
        )
    )
    accounts = datasets.get("credit_accounts") or []
    warnings = tuple(
        dict.fromkeys(
            [
                *list(getattr(getattr(parse_result, "parser_info", None), "warnings", None) or []),
                *_account_structure_warnings(accounts),
                *[
                    f"{failure.get('code', 'PERSONAL_BRIEF_EXTRACTION_FAILURE')}: "
                    f"{failure.get('message', '')}".strip()
                    for failure in failures
                    if isinstance(failure, dict)
                ],
            ]
        )
    )
    entity_fields: dict[str, Any] = {}
    for target, source in (
        ("subject_name", "subject_name"),
        ("subject_id", "id_number"),
        ("marital_status", "marital_status"),
    ):
        if domain_facts.get(source) not in (None, ""):
            entity_fields[target] = domain_facts[source]

    semantic = variant.semantic_extensions()
    overrides = semantic.setdefault("community_projection_overrides", {})
    completeness_policy = overrides.setdefault("completeness", {})
    internal_fields = overrides.setdefault("internal_fields", [])
    internal_facts = overrides.setdefault("internal_facts", [])
    for dataset_name, details in semantic_document.dataset_completeness.items():
        count_key = f"personal_brief_expected_{dataset_name}_count"
        domain_facts[count_key] = int(details.get("expected_row_count") or 0)
        if count_key not in internal_fields:
            internal_fields.append(count_key)
        if count_key not in internal_facts:
            internal_facts.append(count_key)
        completeness_policy[dataset_name] = {
            "basis": "domain_fact_count",
            "count_key": count_key,
            "public_basis": str(details.get("basis") or "canonical_source_component_count"),
        }
    semantic["personal_brief_dataset_completeness"] = semantic_document.dataset_completeness

    return ProjectionData(
        projector_id=plugin.projector_id,
        document_type=plugin.domain_name,
        entity_fields=entity_fields,
        domain_facts=domain_facts,
        semantic=semantic,
        datasets=datasets,
        sections=variant.build_sections(document, document.full_text),
        warnings=warnings,
        evidence_ids=tuple(evidence_ids),
        confidence=document.confidence,
        reason="canonical personal brief IR schema projection",
    )


__all__ = ["derive_personal_brief_projection"]
