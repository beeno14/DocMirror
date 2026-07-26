# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Orchestration adapter for scanned personal detailed credit reports."""

from docmirror.plugins.credit_report.shared.variant import CreditReportVariantAdapter


class PersonalDetailScannedVariant(CreditReportVariantAdapter):
    """Keep scanned-detail extraction behind a dedicated variant boundary."""

    def __init__(self) -> None:
        super().__init__(
            variant_id="personal_detail_scanned",
            report_subtype="personal_detail",
            expected_content_modes=frozenset({"scanned_ocr", "mixed"}),
            include_credit_lines=True,
        )


variant = PersonalDetailScannedVariant()

__all__ = ["PersonalDetailScannedVariant", "variant"]
