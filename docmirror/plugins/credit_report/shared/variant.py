# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavior-preserving adapter contract for credit-report variants.

The public credit-report plugin remains a single registered domain plugin.
Variant adapters isolate subtype-specific orchestration while the existing
extractors are moved behind these boundaries incrementally.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from docmirror.plugins.credit_report.contracts import (
    BASE_DATASET_NAMES,
    CONTENT_MODES,
    SCANNED_CONTENT_MODES,
)


@dataclass(frozen=True)
class CreditReportVariantAdapter:
    """Orchestration boundary for one credit-report document family."""

    variant_id: str
    report_subtype: str
    expected_content_modes: frozenset[str]
    include_credit_lines: bool
    keep_query_institution: bool = True

    def dataset_names(self) -> tuple[str, ...]:
        """Return public dataset order for this variant."""
        names = list(BASE_DATASET_NAMES)
        if self.include_credit_lines:
            names.insert(1, "credit_lines")
        return tuple(names)

    def content_mode_is_expected(self, content_mode: str) -> bool:
        """Report whether routing received the variant's normal source mode."""
        return content_mode in self.expected_content_modes

    def prepare_extraction(self, parse_result: Any, full_text: str) -> Any:
        """Build optional variant-owned indexes once for the current projection."""
        del full_text
        return parse_result

    def use_generic_credit_accounts(self) -> bool:
        """Return whether shared pre-assembled account candidates are trusted."""
        return True

    def extract_native_business(
        self,
        parse_result: Any,
        full_text: str,
        *,
        content_mode: str,
    ) -> dict[str, Any]:
        """Extract native business records owned by this document variant."""
        return {}

    def refine_domain_facts(
        self,
        domain_facts: dict[str, Any],
        field_details: dict[str, Any],
    ) -> None:
        """Apply subtype-specific identity and metadata cleanup in place."""

    def refine_entity_fields(self, entity_fields: dict[str, Any]) -> None:
        """Apply subtype-specific cleanup to the public entity view in place."""

    def data_dictionary(self) -> dict[str, Any]:
        """Return a subtype-owned copy of the public credit dictionary."""
        from docmirror.plugins.credit_report.semantic_enrichment import (
            credit_report_data_dictionary,
        )

        return deepcopy(credit_report_data_dictionary())

    def extract_auxiliary_business(
        self,
        parse_result: Any,
        full_text: str,
        *,
        content_mode: str,
    ) -> dict[str, Any]:
        """Preserve the existing scanned/mixed auxiliary extraction behavior."""
        if content_mode not in SCANNED_CONTENT_MODES:
            return {}
        from docmirror.plugins.credit_report.scanned_business import (
            extract_scanned_credit_business,
        )

        return extract_scanned_credit_business(parse_result, full_text)

    def build_section_content(
        self,
        parse_result: Any,
        full_text: str,
        *,
        auxiliary_business: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return variant-only supplemental facts and records."""
        del auxiliary_business
        return {}

    def build_sections(self, parse_result: Any, full_text: str) -> tuple[dict[str, Any], ...]:
        """Build the current lightweight section view behind the variant seam."""
        from docmirror.plugins._base.kv_community_enrich import (
            build_credit_sections_light,
        )

        return tuple(build_credit_sections_light(parse_result, full_text))

    def semantic_extensions(self) -> dict[str, Any]:
        """Return the existing semantic/output policy for this subtype."""
        from docmirror.plugins.credit_report.semantic_enrichment import (
            credit_report_semantic_extensions,
        )

        return credit_report_semantic_extensions(report_subtype=self.report_subtype)

    def build_reading_projection(
        self,
        parse_result: Any,
        *,
        content_mode: str,
    ) -> Any | None:
        """Return an optional source-reading transform."""
        return None


class UnknownCreditReportVariant(CreditReportVariantAdapter):
    """Compatibility fallback for unclassified credit reports."""

    def __init__(self) -> None:
        super().__init__(
            variant_id="unknown",
            report_subtype="unknown",
            expected_content_modes=CONTENT_MODES,
            include_credit_lines=True,
        )


__all__ = [
    "CreditReportVariantAdapter",
    "UnknownCreditReportVariant",
]
