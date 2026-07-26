# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Orchestration adapter for native-text enterprise credit reports."""

from docmirror.plugins.credit_report.shared.variant import CreditReportVariantAdapter


class EnterpriseNativeVariant(CreditReportVariantAdapter):
    """Keep enterprise extraction behind a dedicated variant boundary."""

    def __init__(self) -> None:
        super().__init__(
            variant_id="enterprise_native",
            report_subtype="enterprise",
            expected_content_modes=frozenset({"native_text", "mixed"}),
            include_credit_lines=True,
        )


variant = EnterpriseNativeVariant()

__all__ = ["EnterpriseNativeVariant", "variant"]
