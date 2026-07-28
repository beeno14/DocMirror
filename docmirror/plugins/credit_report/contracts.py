# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical public vocabulary shared by all credit-report variants."""

from __future__ import annotations

CONTENT_MODE_NATIVE = "native_text"
CONTENT_MODE_SCANNED = "scanned_ocr"
CONTENT_MODE_MIXED = "mixed"
CONTENT_MODE_UNKNOWN = "unknown"

CONTENT_MODES = frozenset(
    {
        CONTENT_MODE_NATIVE,
        CONTENT_MODE_SCANNED,
        CONTENT_MODE_MIXED,
        CONTENT_MODE_UNKNOWN,
    }
)
SCANNED_CONTENT_MODES = frozenset({CONTENT_MODE_SCANNED, CONTENT_MODE_MIXED})

BASE_DATASET_NAMES = (
    "credit_accounts",
    "repayment_liability_records",
    "repayment_records",
    "overdue_records",
    "inquiry_records",
    "public_records",
)

__all__ = [
    "BASE_DATASET_NAMES",
    "CONTENT_MODES",
    "CONTENT_MODE_MIXED",
    "CONTENT_MODE_NATIVE",
    "CONTENT_MODE_SCANNED",
    "CONTENT_MODE_UNKNOWN",
    "SCANNED_CONTENT_MODES",
]
