# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only extraction context for native Digital Personal Brief reports."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from docmirror.plugins.credit_report.shared.entity_decoder import (
    CreditReportEntityContext,
    decode_credit_report_entities,
)


@dataclass(frozen=True)
class PersonalBriefExtractionContext:
    """Variant-owned indexes without changing the sealed ParseResult."""

    parse_result: Any
    entity_context: CreditReportEntityContext

    def __getattr__(self, name: str) -> Any:
        return getattr(self.parse_result, name)


_LOOKBACK_YEARS_RE = re.compile(r"最近\s*(?P<years>\d+)\s*年")


def extract_personal_brief_lookback_years(value: Any) -> int | None:
    """Decode a printed lookback even when PDF layout inserts whitespace."""

    match = _LOOKBACK_YEARS_RE.search(str(value or ""))
    return int(match.group("years")) if match is not None else None


def build_personal_brief_extraction_context(parse_result: Any) -> PersonalBriefExtractionContext:
    """Decode page entities once for the Personal Brief projection."""
    if isinstance(parse_result, PersonalBriefExtractionContext):
        return parse_result
    return PersonalBriefExtractionContext(
        parse_result=parse_result,
        entity_context=decode_credit_report_entities(
            parse_result,
            report_family="personal_brief",
        ),
    )


__all__ = [
    "PersonalBriefExtractionContext",
    "build_personal_brief_extraction_context",
    "extract_personal_brief_lookback_years",
]
