# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authoritative Candidate B pipeline for scanned personal detailed reports.

There is deliberately one extraction result.  Compatibility entry points may
expose different slices of it to the generic credit-report projector, but they
must never invoke another OCR/business extractor or merge another population.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_CORE_BUSINESS_DATASETS = (
    "credit_accounts",
    "credit_lines",
    "repayment_liability_records",
    "repayment_records",
    "overdue_records",
    "inquiry_records",
    "public_records",
)


@dataclass(frozen=True)
class CandidateBExtraction:
    """One immutable-by-convention result shared by every variant adapter hook."""

    business: dict[str, Any]
    section_content: dict[str, Any]
    audit: dict[str, Any]


class CandidateBPipeline:
    """Extract registered canonical pages into the PBOC source schema once."""

    def __init__(self, context: Any, full_text: str) -> None:
        self.context = context
        self.full_text = str(full_text or "")

    def run(self) -> CandidateBExtraction:
        from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
            collect_extraction_issues,
            dataset_states_from_issues,
        )
        from docmirror.plugins.credit_report.personal_detail_scanned.native_extraction import (
            _extract_credit_lines,
            _extract_employment_records,
            _extract_header_datasets,
            _extract_inquiries,
            _extract_liabilities,
            _extract_personal_notes,
            _extract_postpaid_payment_history,
            _extract_postpaid_records,
            _extract_profile_detail_records,
            _extract_public_records,
            _extract_recovery_records,
            _extract_residence_records,
            _extract_source_rows,
            _extract_summary_datasets,
            _source_completeness_ledger,
            reconcile_candidate_b_credit_lines,
        )
        from docmirror.plugins.credit_report.personal_detail_scanned.profile_extraction import (
            extract_candidate_b_profile,
        )
        from docmirror.plugins.credit_report.personal_detail_scanned.relations import (
            derive_candidate_b_overdue_records,
            link_candidate_b_repayments,
        )
        from docmirror.plugins.credit_report.personal_detail_scanned.source_projection import (
            prepare_personal_detail_source_collections,
        )

        def extract_source_pass() -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
            # Registration and fragment joining use static ParseResult evidence
            # only. No OCR can be started before business candidates exist.
            canonical_pages = self.context.pages
            del canonical_pages

            accounts, _discarded_parallel_monthly_rows, account_events = self.context.account_collections()
            repayments = self.context.corrected_repayment_records()
            repayments = link_candidate_b_repayments(
                repayments,
                accounts,
                self.context.corrected_repayment_micro_grids(),
                reading_order_by_logical=dict(self.context.reading_order_by_logical),
                issue_context=self.context,
            )
            business: dict[str, Any] = {
                "credit_accounts": accounts,
                "credit_lines": _extract_credit_lines(self.context),
                "repayment_liability_records": _extract_liabilities(self.context),
                "repayment_records": repayments,
                "overdue_records": derive_candidate_b_overdue_records(accounts, repayments),
                "inquiry_records": _extract_inquiries(self.context),
                "public_records": _extract_public_records(self.context),
            }
            business["credit_summary"] = {
                "source": "candidate_b_canonical_templates",
                "reported_account_count": len(business["credit_accounts"]),
                "projected_account_count": len(business["credit_accounts"]),
                "repayment_liability_count": len(business["repayment_liability_records"]),
                "inquiry_count": len(business["inquiry_records"]),
                "account_population_comparable": False,
            }
            annotations, statements = _extract_personal_notes(self.context)
            summary_records, summary_cells = _extract_summary_datasets(self.context)
            datasets: dict[str, list[dict[str, Any]]] = {
                **_extract_header_datasets(self.context, self.full_text),
                **{name: list(business.get(name) or ()) for name in _CORE_BUSINESS_DATASETS},
                "recovery_records": _extract_recovery_records(self.context),
                "postpaid_records": _extract_postpaid_records(self.context),
                "postpaid_payment_history": _extract_postpaid_payment_history(self.context),
                "personal_detail_account_events": account_events,
                "personal_detail_summary_records": summary_records,
                "personal_detail_summary_cells": summary_cells,
                "residence_records": _extract_residence_records(self.context),
                "employment_records": _extract_employment_records(self.context),
                "annotations": annotations,
                "statements": statements,
                "personal_detail_source_rows": _extract_source_rows(self.context),
                **_extract_profile_detail_records(self.context),
            }
            return business, datasets

        first_business, first_datasets = extract_source_pass()
        repair_payload = {
            "credit_summary": dict(first_business.get("credit_summary") or {}),
            **first_datasets,
        }
        if self.context.prepare_candidate_b_business_repair(repair_payload):
            source_business, source_datasets = extract_source_pass()
        else:
            source_business, source_datasets = first_business, first_datasets

        # The final correction plane covers every source dataset, including
        # monthly grids and profile/detail tables. It consumes only evidence
        # selected by the document-wide repair coordinator.
        corrected_payload = self.context.correct_candidate_b_datasets(
            {
                "credit_summary": dict(source_business.get("credit_summary") or {}),
                **source_datasets,
            }
        )
        all_datasets: dict[str, list[dict[str, Any]]] = {
            name: list(corrected_payload.get(name) or ())
            for name in source_datasets
        }
        all_datasets["credit_lines"] = reconcile_candidate_b_credit_lines(
            self.context,
            list(all_datasets.get("credit_lines") or ()),
        )
        business: dict[str, Any] = {
            name: list(all_datasets.get(name) or ())
            for name in _CORE_BUSINESS_DATASETS
        }
        business["credit_summary"] = dict(corrected_payload.get("credit_summary") or {})
        all_datasets = {name: rows for name, rows in all_datasets.items() if rows}

        profile = extract_candidate_b_profile(self.context)
        issues = collect_extraction_issues(self.context)
        if issues:
            all_datasets["personal_detail_extraction_issues"] = issues
        final_counts = {
            name: sum(isinstance(row, dict) for row in rows)
            for name, rows in all_datasets.items()
            if isinstance(rows, list)
        }
        facts: dict[str, Any] = {
            "subject_profile": profile,
            "credit_summary": dict(business.get("credit_summary") or {}),
            "canonical_dataset_schema": "personal_credit_report_detailed.v2",
            "personal_detail_source_completeness_ledger": _source_completeness_ledger(self.context),
            "personal_detail_dataset_states": dataset_states_from_issues(issues),
            **{f"personal_detail_expected_{name}_count": count for name, count in final_counts.items()},
        }
        content = prepare_personal_detail_source_collections(
            {"facts": facts, "datasets": all_datasets},
            business,
            final_dataset_counts=final_counts,
        )
        supplemental = {
            name: rows
            for name, rows in (content.get("datasets") or {}).items()
            if name not in _CORE_BUSINESS_DATASETS
        }
        section_content = {
            "facts": dict(content.get("facts") or {}),
            "datasets": supplemental,
        }
        audit = {
            "architecture": "candidate_b_clean",
            "source_of_truth": "static_canonical_pages_then_schema_triggered_repair",
            "candidate_population_count": 1,
            "schema_extraction_pass_count": 2
            if self.context.ocr_correction_audit().get("business_repair", {}).get("second_schema_pass_required")
            else 1,
            "parse_result_mutated": False,
            "canonical_layout": self.context.canonical_layout_audit(),
            "page_topology": self.context.page_topology_audit(),
            "ocr_correction": self.context.ocr_correction_audit(),
            "source_dataset_counts": final_counts,
        }
        business["credit_extraction_audit"] = audit
        return CandidateBExtraction(
            business=business,
            section_content=section_content,
            audit=audit,
        )


__all__ = ["CandidateBExtraction", "CandidateBPipeline"]
