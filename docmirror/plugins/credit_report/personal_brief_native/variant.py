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
