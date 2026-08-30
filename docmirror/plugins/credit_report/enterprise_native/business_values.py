# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lossless value rules for enterprise business identifiers."""

from typing import Any

from docmirror.plugins.credit_report.value_utils import compact_text


def opaque_identifier(value: Any) -> str:
    """Join PDF line wraps, but never case-fold or strip identifier punctuation."""
    text = compact_text(value)
    return "" if text in {"", "-", "--", "—", "－", "/"} else text
