# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve credit-report subtype/content mode to an internal variant adapter."""

from __future__ import annotations

from docmirror.plugins.credit_report.enterprise_native.variant import (
    variant as enterprise_native,
)
from docmirror.plugins.credit_report.personal_brief_native.variant import (
    variant as personal_brief_native,
)
from docmirror.plugins.credit_report.personal_detail_scanned.variant import (
    variant as personal_detail_scanned,
)
from docmirror.plugins.credit_report.shared.variant import (
    CreditReportVariantAdapter,
    UnknownCreditReportVariant,
)

_VARIANTS: dict[str, CreditReportVariantAdapter] = {
    personal_brief_native.report_subtype: personal_brief_native,
    enterprise_native.report_subtype: enterprise_native,
    personal_detail_scanned.report_subtype: personal_detail_scanned,
}
_UNKNOWN_VARIANT = UnknownCreditReportVariant()


def resolve_credit_report_variant(
    report_subtype: str,
    content_mode: str = "unknown",
) -> CreditReportVariantAdapter:
    """Return the subtype adapter while preserving behavior for mode mismatches."""
    del content_mode  # Mode expectations are diagnostics, not a routing override.
    return _VARIANTS.get(str(report_subtype or "").strip(), _UNKNOWN_VARIANT)


def registered_credit_report_variants() -> tuple[CreditReportVariantAdapter, ...]:
    """Return the three public credit-report variant adapters."""
    return tuple(_VARIANTS.values())


__all__ = [
    "registered_credit_report_variants",
    "resolve_credit_report_variant",
]
