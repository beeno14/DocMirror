# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Orchestration adapter for scanned personal detailed credit reports."""

import re
from copy import deepcopy
from typing import Any, cast

from docmirror.plugins.credit_report.shared.variant import CreditReportVariantAdapter


class PersonalDetailScannedVariant(CreditReportVariantAdapter):
    """Keep scanned-detail extraction behind a dedicated variant boundary."""

    def __init__(self) -> None:
        super().__init__(
            variant_id="personal_detail_scanned",
            report_subtype="personal_detail",
            expected_content_modes=frozenset({"native_text", "scanned_ocr", "mixed"}),
            include_credit_lines=True,
        )

    def prepare_extraction(self, parse_result: Any, full_text: str) -> Any:
        """Build one logical-page graph and cache for the detailed report."""
        del full_text
        from docmirror.plugins.credit_report.personal_detail_scanned.context import (
            build_personal_detail_extraction_context,
        )

        return build_personal_detail_extraction_context(parse_result)

    def extract_native_business(
        self,
        parse_result: Any,
        full_text: str,
        *,
        content_mode: str,
    ) -> dict[str, Any]:
        """Project the labelled native tables into canonical datasets."""
        if content_mode not in {"native_text", "mixed"}:
            return {}
        from docmirror.plugins.credit_report.personal_detail_scanned.native_extraction import (
            extract_personal_detail_native_business,
        )

        cached = getattr(parse_result, "native_business", None)
        if callable(cached):
            return cast(dict[str, Any], cached(full_text))
        return extract_personal_detail_native_business(parse_result, full_text)

    def extract_auxiliary_business(
        self,
        parse_result: Any,
        full_text: str,
        *,
        content_mode: str,
    ) -> dict[str, Any]:
        """Reuse the detailed context's scanned evidence pass."""
        if content_mode not in {"scanned_ocr", "mixed"}:
            return {}
        cached = getattr(parse_result, "scanned_business", None)
        if callable(cached):
            return cast(dict[str, Any], cached(full_text))
        return super().extract_auxiliary_business(
            parse_result,
            full_text,
            content_mode=content_mode,
        )

    def assemble_business(
        self,
        parse_result: Any,
        full_text: str,
        *,
        content_mode: str,
        existing_collections: dict[str, list[Any]] | None,
        existing_summary: dict[str, Any] | None,
        variant_input: Any,
    ) -> dict[str, Any]:
        """Complete detailed-report links after the shared canonical merge."""
        assembled = super().assemble_business(
            parse_result,
            full_text,
            content_mode=content_mode,
            existing_collections=existing_collections,
            existing_summary=existing_summary,
            variant_input=variant_input,
        )
        accounts = [account for account in assembled.get("credit_accounts") or [] if isinstance(account, dict)]
        valid_account_ids = {str(account.get("account_id") or "") for account in accounts if account.get("account_id")}
        repayment_records = [
            dict(record) for record in assembled.get("repayment_records") or [] if isinstance(record, dict)
        ]
        needs_relink = False
        for record in repayment_records:
            account_id = str(record.get("account_id") or "")
            if account_id and account_id not in valid_account_ids:
                needs_relink = True
                record.pop("account_id", None)
                record.pop("account_identifier", None)
                normalized = record.get("normalized")
                if isinstance(normalized, dict):
                    normalized = dict(normalized)
                    normalized.pop("account_id", None)
                    normalized.pop("account_identifier", None)
                    record["normalized"] = normalized
        if needs_relink:
            from docmirror.models.mirror.domain_access import micro_grid_structures_from_domain_specific
            from docmirror.plugins.credit_report.scanned_business import link_repayment_records_to_accounts

            domain_specific = getattr(getattr(parse_result, "entities", None), "domain_specific", {})
            repayment_records = link_repayment_records_to_accounts(
                repayment_records,
                accounts,
                micro_grid_structures_from_domain_specific(
                    domain_specific if isinstance(domain_specific, dict) else {}
                ),
                reading_order_by_logical=dict(
                    getattr(variant_input, "reading_order_by_logical", {}) or {}
                ),
            )
            assembled["repayment_records"] = repayment_records
        account_identifiers = {
            str(account.get("account_id") or ""): account.get("account_identifier")
            for account in accounts
            if account.get("account_id")
        }
        for record in assembled.get("repayment_records") or []:
            if not record.get("account_identifier"):
                record["account_identifier"] = account_identifiers.get(str(record.get("account_id") or ""))
            normalized = record.get("normalized")
            if isinstance(normalized, dict) and not normalized.get("account_identifier"):
                normalized["account_identifier"] = record.get("account_identifier")
        from docmirror.plugins.credit_report.personal_detail_scanned.contract_projection import (
            project_typed_public_records,
        )

        reporting_currency: str | None = None
        reporting_amount_unit: str | None = None
        content_loader = getattr(variant_input, "section_content", None)
        if callable(content_loader):
            source_content = cast(dict[str, Any], content_loader(full_text))
            metadata_rows = (source_content.get("datasets") or {}).get("personal_report_metadata") or []
            if metadata_rows:
                metadata = metadata_rows[0].get("normalized", metadata_rows[0])
                if isinstance(metadata, dict):
                    reporting_currency = str(metadata.get("reporting_currency") or "").strip() or None
                    reporting_amount_unit = str(metadata.get("reporting_amount_unit") or "").strip() or None
        for dataset_name, records in project_typed_public_records(
            assembled.get("public_records"),
            reporting_currency=reporting_currency,
            reporting_amount_unit=reporting_amount_unit,
        ).items():
            if records:
                assembled[dataset_name] = records
        setattr(
            variant_input,
            "_personal_detail_final_dataset_counts",
            {
                name: sum(isinstance(record, dict) for record in records)
                for name, records in assembled.items()
                if isinstance(records, list)
            },
        )
        return assembled

    def business_dataset_copies(
        self,
        assembled: dict[str, Any],
    ) -> dict[str, list[dict[str, Any]]]:
        """Publish typed public-record views derived from the common envelope."""
        typed_names = (
            "tax_arrears_records",
            "civil_judgment_records",
            "enforcement_records",
            "administrative_penalty_records",
            "personal_housing_fund_records",
            "professional_qualification_records",
            "award_records",
        )
        return {
            name: list(assembled.get(name) or [])
            for name in typed_names
            if assembled.get(name)
        }

    def finalize_datasets(self, datasets: dict[str, list[dict[str, Any]]]) -> None:
        """Reconcile the absence ledger against the rows that will be published."""
        for status_row in datasets.get("personal_detail_dataset_status") or []:
            dataset_name = str(status_row.get("dataset_name") or "")
            if not dataset_name:
                continue
            observed_count = len(datasets.get(dataset_name) or [])
            status_row["observed_row_count"] = observed_count
            if observed_count:
                status_row["presence_status"] = "observed_nonempty"
                status_row["reason"] = "records_projected"
            elif status_row.get("presence_status") == "observed_nonempty":
                status_row["presence_status"] = "not_observed"
                status_row["reason"] = "no_explicit_absence_evidence"

    def refine_domain_facts(
        self,
        domain_facts: dict[str, Any],
        field_details: dict[str, Any],
    ) -> None:
        """Reject generic enterprise identity leakage for personal reports."""
        for field_name in (
            "unified_social_credit_code",
            "zhongzheng_code",
            "organization_code",
            "national_tax_id",
        ):
            domain_facts.pop(field_name, None)
            field_details.pop(field_name, None)
        id_number = re.sub(r"\s+", "", str(domain_facts.get("id_number") or ""))
        id_type = str(domain_facts.get("id_type") or "").strip()
        if re.fullmatch(r"\d{17}[0-9Xx]", id_number) and id_type not in {
            "身份证",
            "居民身份证",
        }:
            domain_facts["id_type"] = "身份证"
            field_details["id_type"] = {
                "source": "personal_detail_identity_validation",
                "confidence": 1.0,
            }

    def build_section_content(
        self,
        parse_result: Any,
        full_text: str,
        *,
        auxiliary_business: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Use the complete personal-detail parser for every detailed report."""
        from docmirror.plugins.credit_report.personal_detail_scanned.contract_projection import (
            apply_personal_detail_contract,
        )

        auxiliary = auxiliary_business or {}
        if not getattr(parse_result, "pages", None):
            facts: dict[str, Any] = {}
            if isinstance(auxiliary.get("subject_profile"), dict):
                facts["subject_profile"] = deepcopy(auxiliary["subject_profile"])
            datasets = {
                name: list(auxiliary.get(name) or [])
                for name in ("residence_records", "employment_records", "statements", "annotations")
                if auxiliary.get(name)
            }
            return apply_personal_detail_contract({
                **({"facts": facts} if facts else {}),
                **({"datasets": datasets} if datasets else {}),
            }, auxiliary, final_dataset_counts=getattr(
                parse_result, "_personal_detail_final_dataset_counts", {}
            ))
        from docmirror.plugins.credit_report.personal_detail_scanned.native_extraction import (
            extract_personal_detail_section_content,
        )

        cached = getattr(parse_result, "section_content", None)
        content: dict[str, Any] = (
            cast(dict[str, Any], cached(full_text))
            if callable(cached)
            else extract_personal_detail_section_content(parse_result, full_text)
        )
        facts = content.setdefault("facts", {})
        if not facts.get("subject_profile") and isinstance(auxiliary.get("subject_profile"), dict):
            facts["subject_profile"] = deepcopy(auxiliary["subject_profile"])
        datasets = content.setdefault("datasets", {})
        for name in ("residence_records", "employment_records", "statements", "annotations"):
            if not datasets.get(name) and auxiliary.get(name):
                datasets[name] = list(auxiliary[name])
        return apply_personal_detail_contract(
            content,
            auxiliary,
            final_dataset_counts=getattr(
                parse_result, "_personal_detail_final_dataset_counts", {}
            ),
        )

    def data_dictionary(self) -> dict[str, Any]:
        """Describe the datasets exposed only by the personal detailed variant."""
        from docmirror.plugins.credit_report.personal_detail_scanned.schema import (
            personal_detail_data_dictionary,
        )

        return personal_detail_data_dictionary()

    def build_sections(self, parse_result: Any, full_text: str) -> tuple[dict[str, Any], ...]:
        """Return the source-faithful detailed-report section hierarchy."""
        del full_text
        page_count = len(getattr(parse_result, "pages", None) or [])
        if page_count < 13:
            return super().build_sections(parse_result, "")
        sections = (
            {
                "id": "sec_personal_basic",
                "title": "个人基本信息",
                "type": "basic_information",
                "page_start": 1,
                "page_end": 2,
            },
            {
                "id": "sec_credit_summary",
                "title": "信息概要",
                "type": "credit_summary",
                "page_start": 2,
                "page_end": 4,
            },
            {
                "id": "sec_credit_details",
                "title": "信贷交易信息明细",
                "type": "credit_details",
                "page_start": 4,
                "page_end": 12,
            },
            {
                "id": "sec_non_credit_transactions",
                "title": "非信贷交易信息明细",
                "type": "non_credit_transactions",
                "page_start": 12,
                "page_end": 12,
            },
            {
                "id": "sec_public_records",
                "title": "公共信息明细",
                "type": "public_records",
                "page_start": 12,
                "page_end": 13,
            },
            {
                "id": "sec_statements",
                "title": "机构说明与本人声明",
                "type": "statements",
                "page_start": 6,
                "page_end": 6,
            },
            {
                "id": "sec_annotations",
                "title": "异议标注",
                "type": "annotations",
                "page_start": 2,
                "page_end": 13,
            },
            {
                "id": "sec_inquiries",
                "title": "查询记录",
                "type": "inquiries",
                "page_start": 13,
                "page_end": 13,
            },
        )
        if page_count >= 15:
            sections += (
                {
                    "id": "sec_report_explanation",
                    "title": "报告说明与编制说明",
                    "type": "report_explanation",
                    "page_start": 14,
                    "page_end": 15,
                },
            )
        return sections

    def semantic_extensions(self) -> dict[str, Any]:
        """Declare datasets as the detailed report's canonical storage."""
        from docmirror.plugins.credit_report.personal_detail_scanned.schema import (
            personal_detail_semantic_extensions,
        )

        return personal_detail_semantic_extensions()


variant = PersonalDetailScannedVariant()

__all__ = ["PersonalDetailScannedVariant", "variant"]
