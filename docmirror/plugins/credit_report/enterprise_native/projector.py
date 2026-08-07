# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Post-seal projection for the canonical digital-enterprise pipeline."""

from __future__ import annotations

from typing import Any

from docmirror.plugins._base.projector import ProjectionData
from docmirror.plugins.credit_report.enterprise_native.pipeline import run_enterprise_pipeline
from docmirror.plugins.credit_report.enterprise_native.quality import quality_warning
from docmirror.plugins.credit_report.projection import (
    _account_structure_warnings,
    _records,
)
from docmirror.plugins.credit_report.report_profile import (
    detect_credit_report_content_mode,
    recover_credit_report_header_fields,
)
from docmirror.plugins.credit_report.semantic_enrichment import enrich_credit_report_record_evidence


def derive_enterprise_projection(plugin: Any, parse_result: Any, full_text: str = "") -> ProjectionData:
    """Run ParseResult -> canonical IR -> one enterprise schema -> projection."""
    content_mode = detect_credit_report_content_mode(parse_result)
    artifacts = run_enterprise_pipeline(parse_result, content_mode=content_mode)
    document = artifacts.document_ir
    semantic_document = artifacts.semantic_document
    from docmirror.plugins.credit_report.enterprise_native.variant import variant

    recovered_header = recover_credit_report_header_fields(
        document,
        document.full_text,
        report_subtype="enterprise",
    )
    domain_facts = {**semantic_document.facts, **recovered_header}
    for field_name in (
        "company_name",
        "id_type",
        "id_number",
        "subject_id",
        "marital_status",
        "report_edition",
        "exchange_rate_usd_cny",
        "exchange_rate_effective_period",
    ):
        domain_facts.pop(field_name, None)
    domain_facts["report_subtype"] = "enterprise"
    domain_facts["content_mode"] = content_mode
    domain_facts["credit_summary"] = semantic_document.credit_summary
    domain_facts["extraction_report"] = semantic_document.extraction_report
    domain_facts["source_information_quality"] = {
        "status": (
            "bad_input"
            if any(flag.get("severity") == "error" for flag in semantic_document.quality_flags)
            else "limited"
            if any(flag.get("status") in {"source_limited", "source_truncated", "estimated"} for flag in semantic_document.quality_flags)
            else "complete_as_reported"
        ),
        "flags": list(semantic_document.quality_flags),
    }
    domain_facts["field_details"] = {
        key: {
            "source": "canonical_enterprise_document_ir",
            "confidence": document.confidence,
        }
        for key, value in domain_facts.items()
        if key not in {"credit_summary", "extraction_report", "source_information_quality"}
        and value not in (None, "")
    }
    domain_facts["data_dictionary"] = variant.data_dictionary()

    datasets = {
        name: _records(name, rows)
        for name, rows in semantic_document.datasets.items()
        if rows
    }
    evidence_ids = enrich_credit_report_record_evidence(parse_result, datasets)
    accounts = datasets.get("enterprise_credit_accounts") or []
    warnings = tuple(
        dict.fromkeys(
            [
                *list(getattr(getattr(parse_result, "parser_info", None), "warnings", None) or []),
                *_account_structure_warnings(accounts),
                *[
                    quality_warning(flag)
                    for flag in semantic_document.quality_flags
                    if flag.get("severity") in {"warning", "error"}
                ],
                *[
                    f"{failure.get('code', 'ENTERPRISE_EXTRACTION_FAILURE')}: "
                    f"{failure.get('message', '')}".strip()
                    for failure in semantic_document.extraction_report.get("failures") or []
                    if isinstance(failure, dict)
                ],
            ]
        )
    )
    entity_fields = {
        "subject_name": domain_facts["subject_name"]
    } if domain_facts.get("subject_name") else {}
    semantic = variant.semantic_extensions()
    overrides = semantic.setdefault("community_projection_overrides", {})
    completeness = overrides.setdefault("completeness", {})
    internal_fields = overrides.setdefault("internal_fields", [])
    internal_facts = overrides.setdefault("internal_facts", [])
    for dataset_name, details in semantic_document.dataset_completeness.items():
        count_key = f"enterprise_expected_{dataset_name}_count"
        domain_facts[count_key] = int(details.get("expected_row_count") or 0)
        internal_fields.append(count_key)
        internal_facts.append(count_key)
        completeness[dataset_name] = {
            "basis": "domain_fact_count",
            "count_key": count_key,
            "public_basis": str(details.get("basis") or "canonical_source_component_count"),
        }
    semantic["enterprise_dataset_completeness"] = semantic_document.dataset_completeness
    return ProjectionData(
        projector_id=plugin.projector_id,
        document_type=plugin.domain_name,
        entity_fields=entity_fields,
        domain_facts=domain_facts,
        semantic=semantic,
        datasets=datasets,
        sections=semantic_document.sections,
        warnings=warnings,
        evidence_ids=tuple(evidence_ids),
        confidence=float(getattr(parse_result, "confidence", 1.0) or 0.0),
        reason="canonical enterprise IR schema projection",
    )


__all__ = ["derive_enterprise_projection"]
