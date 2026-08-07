# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Orchestration adapter for native-text personal brief credit reports."""

from __future__ import annotations

from typing import Any

from docmirror.plugins.credit_report.shared.variant import CreditReportVariantAdapter


class PersonalBriefNativeVariant(CreditReportVariantAdapter):
    """Keep personal-brief extraction and reading behavior isolated."""

    def __init__(self) -> None:
        super().__init__(
            variant_id="personal_brief_native",
            report_subtype="personal_brief",
            expected_content_modes=frozenset({"native_text", "mixed"}),
            include_credit_lines=False,
            keep_query_institution=False,
        )

    def prepare_extraction(self, parse_result: Any, full_text: str) -> Any:
        """Build the read-only Personal Brief entity index once."""
        del full_text
        from docmirror.plugins.credit_report.personal_brief_native.context import (
            build_personal_brief_extraction_context,
        )

        return build_personal_brief_extraction_context(parse_result)

    def build_section_content(
        self,
        parse_result: Any,
        full_text: str,
        *,
        auxiliary_business: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del auxiliary_business
        from docmirror.plugins.credit_report.personal_brief_native.extraction import (
            extract_personal_brief_section_content,
        )

        return extract_personal_brief_section_content(parse_result, full_text)

    def extract_native_business(
        self,
        parse_result: Any,
        full_text: str,
        *,
        content_mode: str,
    ) -> dict[str, Any]:
        """Extract only native personal-brief business records."""
        if content_mode not in self.expected_content_modes:
            return {}
        from docmirror.plugins.credit_report.personal_brief_native.extraction import (
            extract_personal_brief_native_business,
        )

        return extract_personal_brief_native_business(parse_result, full_text)

    def data_dictionary(self) -> dict[str, Any]:
        """Return the personal-brief-owned public data dictionary."""
        from docmirror.plugins.credit_report.personal_brief_native.schema import (
            personal_brief_data_dictionary,
        )

        return personal_brief_data_dictionary()

    def semantic_extensions(self) -> dict[str, Any]:
        """Return personal document-order and rendering policy."""
        from docmirror.plugins.credit_report.personal_brief_native.schema import (
            personal_brief_semantic_extensions,
        )

        return personal_brief_semantic_extensions()

    def build_sections(self, parse_result: Any, full_text: str) -> tuple[dict[str, Any], ...]:
        """Project the canonical source order instead of parser page headings."""
        from docmirror.plugins.credit_report.personal_brief_native.ir import (
            CanonicalPersonalBriefDocumentIR,
        )

        if not isinstance(parse_result, CanonicalPersonalBriefDocumentIR):
            return super().build_sections(parse_result, full_text)

        section_specs = (
            ("report_header", "报告信息", "report_header", ("report_header",)),
            (
                "credit_details",
                "信贷记录",
                "credit_details",
                (
                    "credit_summary",
                    "asset_disposition",
                    "guarantor_compensation",
                    "credit_cards",
                    "loans",
                    "other_business",
                    "repayment_liability",
                ),
            ),
            (
                "non_credit_transactions",
                "非信贷交易记录",
                "non_credit_transactions",
                ("non_credit_transactions",),
            ),
            (
                "public_records",
                "公共记录",
                "public_records",
                (
                    "public_records",
                    "tax_arrears",
                    "civil_judgments",
                    "enforcements",
                    "administrative_penalties",
                ),
            ),
            (
                "institution_statements",
                "机构说明",
                "institution_statements",
                ("institution_statements",),
            ),
            (
                "inquiries",
                "查询记录",
                "inquiries",
                ("institution_inquiries", "personal_inquiries"),
            ),
            ("notes", "说明", "notes", ("report_notes",)),
        )
        sections: list[dict[str, Any]] = []
        for section_id, title, section_type, canonical_keys in section_specs:
            components = [
                component
                for key in canonical_keys
                for component in parse_result.components_for(key)
            ]
            if not components:
                continue
            source_pages = sorted(
                {
                    page
                    for component in components
                    for page in component.source_pages
                    if page > 0
                }
            )
            sections.append(
                {
                    "id": f"sec_personal_brief_{section_id}",
                    "title": title,
                    "name": title,
                    "type": section_type,
                    "page_start": source_pages[0] if source_pages else 1,
                    "page_end": source_pages[-1] if source_pages else 1,
                }
            )
        return tuple(sections)

    def strip_supplemental_node_bindings(self) -> bool:
        """Personal supplemental views copy source rows into several datasets."""
        return True

    def build_reading_projection(
        self,
        parse_result: Any,
        *,
        content_mode: str,
    ) -> Any | None:
        if content_mode not in {"native_text", "mixed"}:
            return None
        from docmirror.output.reading_projection import ReadingProjection
        from docmirror.plugins.credit_report.account_reading_order import (
            build_account_reading_projection,
        )
        from docmirror.plugins.credit_report.inquiry_reading_order import (
            build_institution_inquiry_reading_projection,
        )

        projections = (
            build_account_reading_projection(parse_result),
            build_institution_inquiry_reading_projection(parse_result),
        )
        transforms = tuple(
            transform for projection in projections if projection is not None for transform in projection.transforms
        )
        return ReadingProjection(plugin_id="credit_report", transforms=transforms) if transforms else None


variant = PersonalBriefNativeVariant()

__all__ = ["PersonalBriefNativeVariant", "variant"]
