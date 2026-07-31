# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Personal-brief schema boundary.

The current personal schema remains behavior-compatible, but enterprise code
no longer derives from it.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def personal_brief_data_dictionary() -> dict[str, Any]:
    from docmirror.plugins.credit_report.semantic_enrichment import (
        credit_report_data_dictionary,
    )

    dictionary = deepcopy(credit_report_data_dictionary())
    dictionary["schema_id"] = "personal_brief_credit_report"
    return dictionary


def personal_brief_semantic_extensions() -> dict[str, Any]:
    from docmirror.plugins.credit_report.semantic_enrichment import (
        credit_report_semantic_extensions,
    )

    semantic = credit_report_semantic_extensions(report_subtype="personal_brief")
    semantic["dataset_document_order"] = [
        "personal_report_metadata",
        "report_notes",
        "identity_documents",
        "personal_credit_summary_records",
        "asset_disposition_records",
        "guarantor_compensation_records",
        "credit_accounts",
        "repayment_liability_records",
        "repayment_records",
        "overdue_records",
        "postpaid_records",
        "tax_arrears_records",
        "civil_judgment_records",
        "enforcement_records",
        "administrative_penalty_records",
        "public_records",
        "inquiry_records",
    ]
    return semantic


__all__ = [
    "personal_brief_data_dictionary",
    "personal_brief_semantic_extensions",
]
