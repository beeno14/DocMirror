# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavior-locked scalar helpers shared by credit-report extractors."""

from __future__ import annotations

import hashlib
import re
from decimal import Decimal, InvalidOperation
from typing import Any


def compact_text(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or ""))


def linear_text(value: Any) -> str:
    text = str(value or "").replace("**", "").replace("|", " ")
    return re.sub(r"\s+", " ", text).strip()


def parse_number(value: Any) -> int | float | None:
    raw = re.sub(r"[^0-9.-]", "", str(value or "").replace(",", ""))
    if not raw or raw in {"-", ".", "-."}:
        return None
    try:
        number = Decimal(raw)
    except InvalidOperation:
        return None
    return int(number) if number == number.to_integral_value() else float(number)


def stable_record_id(prefix: str, *parts: Any) -> str:
    identity = "|".join(compact_text(part).upper() for part in parts)
    digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"


__all__ = [
    "compact_text",
    "linear_text",
    "parse_number",
    "stable_record_id",
]
