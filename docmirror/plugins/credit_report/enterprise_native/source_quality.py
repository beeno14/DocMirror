# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""F4 source-information caveats for canonical enterprise reports."""

from __future__ import annotations

import re
from typing import Any

from docmirror.plugins.credit_report.enterprise_native.ir import CanonicalEnterpriseDocumentIR
from docmirror.plugins.credit_report.enterprise_native.quality import EnterpriseQualityFlag

_SPACE = re.compile(r"\s+")
_LIMITED = re.compile(r"(?:受?篇幅所限|受篇幅限制|仅展示|只展示|不展示全部)")
_ESTIMATED_AVAILABLE = re.compile(r"(?:剩余)?可用额度.{0,32}(?:无法准确计算|不能准确计算|估算)")
_PLACEHOLDERS = frozenset({"", "-", "--", "—", "－", "/", "不详", "未知"})


def _compact(value: Any) -> str:
    return _SPACE.sub("", str(value or ""))


def _scope(text: str) -> tuple[str, ...]:
    scopes: list[str] = []
    for marker, scope in (
        ("信贷", "credit_records"),
        ("已结清", "settled_credit_records"),
        ("非信贷", "non_credit_records"),
        ("公共", "public_records"),
    ):
        if marker in text:
            scopes.append(scope)
    return tuple(scopes or ("document",))


def _source_pages(document: CanonicalEnterpriseDocumentIR, pattern: re.Pattern[str]) -> tuple[int, ...]:
    return tuple(page for page, text in document.page_texts.items() if pattern.search(_compact(text)))


def _profile_truncation_flag(datasets: dict[str, list[dict[str, Any]]]) -> EnterpriseQualityFlag | None:
    records = datasets.get("enterprise_profile") or []
    if not records:
        return None
    record = records[0]
    value = str(record.get("operating_address") or "").strip()
    pages = sorted(
        {
            int(ref.get("source_page"))
            for ref in record.get("source_refs") or ()
            if isinstance(ref, dict) and str(ref.get("source_page") or "").isdigit()
        }
    )
    # An address ending in a connective character is a source-side truncation
    # signal. It is flagged, never repaired from conjecture.
    if len(value) >= 8 and value[-1:] in "中与及和至于向从由":
        return EnterpriseQualityFlag(
            code="ENTERPRISE_SOURCE_FIELD_TRUNCATED",
            severity="warning",
            category="source_information",
            status="source_truncated",
            message="The operating address appears truncated in the source report; no value was inferred.",
            source_pages=tuple(pages),
            scope=("basic_information",),
            dataset="enterprise_profile",
            field_name="operating_address",
            details={"source_value": value},
        )
    return None


def assess_enterprise_source_information(
    document: CanonicalEnterpriseDocumentIR,
    datasets: dict[str, list[dict[str, Any]]],
) -> tuple[EnterpriseQualityFlag, ...]:
    """Flag limitations stated by the source and suspicious source omissions."""
    flags: list[EnterpriseQualityFlag] = []
    for page, text in document.page_texts.items():
        compact = _compact(text)
        if _LIMITED.search(compact):
            flags.append(
                EnterpriseQualityFlag(
                    code="ENTERPRISE_SOURCE_SCOPE_LIMITED",
                    severity="warning",
                    category="source_information",
                    status="source_limited",
                    message="The report explicitly limits the records displayed; absence outside that scope is not proof of no record.",
                    source_pages=(page,),
                    scope=_scope(compact),
                )
            )
        if _ESTIMATED_AVAILABLE.search(compact):
            flags.append(
                EnterpriseQualityFlag(
                    code="ENTERPRISE_SOURCE_VALUE_ESTIMATED",
                    severity="warning",
                    category="source_information",
                    status="estimated",
                    message="The report states that available credit cannot be calculated accurately and may be estimated.",
                    source_pages=(page,),
                    scope=("credit_summary", "credit_records"),
                    field_name="available_limit",
                )
            )

    placeholder_hits: dict[tuple[str, str], set[int]] = {}
    for dataset, rows in datasets.items():
        for row in rows:
            page = row.get("source_page")
            for field_name, value in row.items():
                if field_name.startswith("source_") or field_name in {"normalized", "source_refs", "confidence"}:
                    continue
                if isinstance(value, str) and value.strip() in _PLACEHOLDERS:
                    key = (dataset, field_name)
                    placeholder_hits.setdefault(key, set())
                    if isinstance(page, int) and page > 0:
                        placeholder_hits[key].add(page)
    for (dataset, field_name), pages in sorted(placeholder_hits.items()):
        flags.append(
            EnterpriseQualityFlag(
                code="ENTERPRISE_SOURCE_VALUE_NOT_REPORTED",
                severity="info",
                category="source_information",
                status="not_reported",
                message=f"The source uses a missing-value marker for {dataset}.{field_name}.",
                source_pages=tuple(sorted(pages)),
                dataset=dataset,
                field_name=field_name,
            )
        )
    truncation = _profile_truncation_flag(datasets)
    if truncation is not None:
        flags.append(truncation)
    return tuple(flags)


__all__ = ["assess_enterprise_source_information"]
