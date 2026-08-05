# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Orchestration adapter for scanned personal detailed credit reports."""

import re
from copy import deepcopy
from typing import Any, cast

from docmirror.plugins.credit_report.shared.variant import CreditReportVariantAdapter

_SECTION_CLASSIFIERS = (
    ("sec_personal_basic", "个人基本信息", "basic_information", ("个人基本信息",)),
    ("sec_credit_summary", "信息概要", "credit_summary", ("信息概要",)),
    ("sec_credit_details", "信贷交易信息明细", "credit_details", ("信贷交易信息明细",)),
    ("sec_non_credit_transactions", "非信贷交易信息明细", "non_credit_transactions", ("非信贷交易信息明细",)),
    (
        "sec_public_records",
        "公共信息明细",
        "public_records",
        ("公共信息明细", "欠税记录", "民事判决记录", "强制执行记录"),
    ),
    ("sec_inquiries", "查询记录", "inquiries", ("查询记录", "机构查询记录明细", "本人查询记录明细")),
    ("sec_report_explanation", "报告说明与编制说明", "report_explanation", ("报告说明", "编制说明")),
)


def _page_source_text(page: Any) -> str:
    values = [str(getattr(block, "content", "") or "") for block in getattr(page, "texts", None) or ()]
    for table in getattr(page, "tables", None) or ():
        metadata = getattr(table, "metadata", None) or {}
        rows = metadata.get("raw_rows") if isinstance(metadata, dict) else None
        if isinstance(rows, list):
            values.extend(str(cell or "") for row in rows if isinstance(row, list) for cell in row)
    return re.sub(r"\s+", "", " ".join(values))


def _classified_sections(parse_result: Any) -> tuple[dict[str, Any], ...]:
    pages = list(getattr(parse_result, "pages", None) or ())
    if not pages:
        return ()
    page_text = {
        int(getattr(page, "page_number", 0) or index): _page_source_text(page)
        for index, page in enumerate(pages, start=1)
    }
    starts: list[tuple[int, str, str, str]] = []
    for section_id, title, section_type, markers in _SECTION_CLASSIFIERS:
        matches = []
        for page, text in page_text.items():
            if not any(marker in text for marker in markers):
                continue
            if section_id == "sec_inquiries" and "查询记录概要" in text:
                if not any(marker in text for marker in ("机构查询记录明细", "本人查询记录明细")) and not re.search(
                    r"(?:^|[一二三四五六七八九十])[、.．)）]?查询记录(?!概要)", text
                ):
                    continue
            matches.append(page)
        if matches:
            starts.append((min(matches), section_id, title, section_type))
    # One damaged marker must not replace the full fallback hierarchy. Two or
    # more ordered anchors are sufficient to derive source-relative ranges.
    if len(starts) < 2:
        return ()
    ordered = sorted(starts, key=lambda item: (item[0], item[1]))
    page_count = max(page_text)
    sections: list[dict[str, Any]] = []
    for page_start, section_id, title, section_type in ordered:
        sections.append(
            {
                "id": section_id,
                "title": title,
                "type": section_type,
                "page_start": page_start,
                # Dataset provenance extends populated sections below. Starting
                # at the root anchor avoids absorbing a summary or the first
                # page of the following section merely because both share a
                # physical page in the canonical layout.
                "page_end": page_count if section_id == "sec_report_explanation" else page_start,
            }
        )
    report_start = next(
        (page for page, section_id, _title, _type in ordered if section_id == "sec_report_explanation"),
        page_count + 1,
    )
    for section_id, title, section_type, markers in (
        ("sec_statements", "机构说明与本人声明", "statements", ("机构说明", "本人声明")),
        ("sec_annotations", "异议标注", "annotations", ("异议标注", "异议处理中")),
    ):
        matches = [
            page
            for page, text in page_text.items()
            if page < report_start and any(marker in text for marker in markers)
        ]
        if matches:
            sections.append(
                {
                    "id": section_id,
                    "title": title,
                    "type": section_type,
                    "page_start": min(matches),
                    "page_end": max(matches),
                }
            )
    return tuple(sections)


class PersonalDetailScannedVariant(CreditReportVariantAdapter):
    """Keep scanned-detail extraction behind a dedicated variant boundary."""

    def __init__(self) -> None:
        super().__init__(
            variant_id="personal_detail_scanned",
            report_subtype="personal_detail",
            expected_content_modes=frozenset({"native_text", "scanned_ocr", "mixed"}),
            include_credit_lines=True,
        )

    def dataset_names(self) -> tuple[str, ...]:
        """Return the only public dataset vocabulary: canonical PBOC v2."""
        from docmirror.plugins.credit_report.personal_detail_scanned.schema import (
            PBOC_DATASET_ORDER,
        )

        return PBOC_DATASET_ORDER

    def source_dataset_names(self) -> tuple[str, ...]:
        """Return private collection names emitted by the extraction assembly."""
        return super().dataset_names()

    def use_generic_credit_accounts(self) -> bool:
        """Candidate B never trusts projector-level account candidates."""
        return False

    def prepare_extraction(self, parse_result: Any, full_text: str) -> Any:
        """Build one logical-page graph and cache for the detailed report."""
        del full_text
        from docmirror.plugins.credit_report.personal_detail_scanned.context import (
            build_personal_detail_extraction_context,
        )

        return build_personal_detail_extraction_context(parse_result)

    def extract_native_business(
        self,
        parse_result: Any,
        full_text: str,
        *,
        content_mode: str,
    ) -> dict[str, Any]:
        """Disable the shared assembly's second candidate population."""
        del parse_result, full_text, content_mode
        return {}

    def extract_auxiliary_business(
        self,
        parse_result: Any,
        full_text: str,
        *,
        content_mode: str,
    ) -> dict[str, Any]:
        """Expose Candidate B to the generic routing envelope without merging."""
        del content_mode
        loader = getattr(parse_result, "candidate_b_extraction", None)
        if not callable(loader):
            return {}
        return deepcopy(cast(Any, loader(full_text)).business)

    def assemble_business(
        self,
        parse_result: Any,
        full_text: str,
        *,
        content_mode: str,
        existing_collections: dict[str, list[Any]] | None,
        existing_summary: dict[str, Any] | None,
        variant_input: Any,
    ) -> dict[str, Any]:
        """Return Candidate B verbatim; discard every projector candidate."""
        del parse_result, content_mode, existing_collections, existing_summary
        loader = getattr(variant_input, "candidate_b_extraction", None)
        if not callable(loader):
            return {}
        result = loader(full_text)
        assembled = deepcopy(cast(Any, result).business)
        setattr(
            variant_input,
            "_personal_detail_final_dataset_counts",
            {
                name: sum(isinstance(record, dict) for record in records)
                for name, records in assembled.items()
                if isinstance(records, list)
            },
        )
        return assembled

    def business_dataset_copies(
        self,
        assembled: dict[str, Any],
    ) -> dict[str, list[dict[str, Any]]]:
        """Typed views are emitted once by Candidate B section projection."""
        del assembled
        return {}

    def finalize_datasets(self, datasets: dict[str, list[dict[str, Any]]]) -> None:
        """Reconcile the absence ledger against the rows that will be published."""
        for status_row in datasets.get("personal_detail_dataset_status") or []:
            dataset_name = str(status_row.get("dataset_name") or "")
            if not dataset_name:
                continue
            observed_count = len(datasets.get(dataset_name) or [])
            status_row["observed_row_count"] = observed_count
            if observed_count:
                if status_row.get("presence_status") not in {"partial", "extraction_failed", "unknown"}:
                    status_row["presence_status"] = "observed_nonempty"
                    status_row["reason"] = "records_projected"
            elif status_row.get("presence_status") == "observed_nonempty":
                status_row["presence_status"] = "not_observed"
                status_row["reason"] = "no_explicit_absence_evidence"

    def refine_domain_facts(
        self,
        domain_facts: dict[str, Any],
        field_details: dict[str, Any],
    ) -> None:
        """Reject generic enterprise identity leakage for personal reports."""
        for field_name in (
            "unified_social_credit_code",
            "zhongzheng_code",
            "organization_code",
            "national_tax_id",
        ):
            domain_facts.pop(field_name, None)
            field_details.pop(field_name, None)
        id_number = re.sub(r"\s+", "", str(domain_facts.get("id_number") or ""))
        id_type = str(domain_facts.get("id_type") or "").strip()
        if re.fullmatch(r"\d{17}[0-9Xx]", id_number) and id_type not in {
            "身份证",
            "居民身份证",
        }:
            domain_facts["id_type"] = "身份证"
            field_details["id_type"] = {
                "source": "personal_detail_identity_validation",
                "confidence": 1.0,
            }

    def build_section_content(
        self,
        parse_result: Any,
        full_text: str,
        *,
        auxiliary_business: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return Candidate B supplemental datasets without a fallback merge."""
        del auxiliary_business
        loader = getattr(parse_result, "candidate_b_extraction", None)
        if not callable(loader):
            return {}
        return deepcopy(cast(Any, loader(full_text)).section_content)

    def data_dictionary(self) -> dict[str, Any]:
        """Return the canonical PBOC v2 dictionary for detailed reports."""
        from docmirror.plugins.credit_report.personal_detail_scanned.schema import (
            personal_detail_data_dictionary,
        )

        return personal_detail_data_dictionary()

    def build_sections(self, parse_result: Any, full_text: str) -> tuple[dict[str, Any], ...]:
        """Return the source-faithful detailed-report section hierarchy."""
        del full_text
        pages = list(getattr(parse_result, "pages", None) or [])
        page_count = max(
            (int(getattr(page, "page_number", 0) or index) for index, page in enumerate(pages, start=1)),
            default=0,
        )
        classified = _classified_sections(parse_result)
        if classified:
            sections = classified
        elif page_count < 13:
            return super().build_sections(parse_result, "")
        else:
            sections = (
                {
                    "id": "sec_personal_basic",
                    "title": "个人基本信息",
                    "type": "basic_information",
                    "page_start": 1,
                    "page_end": 2,
                },
                {
                    "id": "sec_credit_summary",
                    "title": "信息概要",
                    "type": "credit_summary",
                    "page_start": 2,
                    "page_end": 4,
                },
                {
                    "id": "sec_credit_details",
                    "title": "信贷交易信息明细",
                    "type": "credit_details",
                    "page_start": 4,
                    "page_end": 12,
                },
                {
                    "id": "sec_non_credit_transactions",
                    "title": "非信贷交易信息明细",
                    "type": "non_credit_transactions",
                    "page_start": 12,
                    "page_end": 12,
                },
                {
                    "id": "sec_public_records",
                    "title": "公共信息明细",
                    "type": "public_records",
                    "page_start": 12,
                    "page_end": 13,
                },
                {
                    "id": "sec_statements",
                    "title": "机构说明与本人声明",
                    "type": "statements",
                    "page_start": 6,
                    "page_end": 6,
                },
                {
                    "id": "sec_annotations",
                    "title": "异议标注",
                    "type": "annotations",
                    "page_start": 2,
                    "page_end": 13,
                },
                {
                    "id": "sec_inquiries",
                    "title": "查询记录",
                    "type": "inquiries",
                    "page_start": 13,
                    "page_end": 13,
                },
            )
        if not classified and page_count >= 15:
            sections += (
                {
                    "id": "sec_report_explanation",
                    "title": "报告说明与编制说明",
                    "type": "report_explanation",
                    "page_start": 14,
                    "page_end": 15,
                },
            )
        sections += (
            {
                "id": "sec_extraction_review",
                "title": "提取问题与人工复核",
                "type": "extraction_review",
                "page_start": 1,
                "page_end": page_count,
            },
        )
        return sections

    def semantic_extensions(self) -> dict[str, Any]:
        """Declare PBOC v2 datasets as the detailed report's canonical storage."""
        from docmirror.plugins.credit_report.personal_detail_scanned.schema import (
            personal_detail_semantic_extensions,
        )

        return personal_detail_semantic_extensions()


variant = PersonalDetailScannedVariant()

__all__ = ["PersonalDetailScannedVariant", "variant"]
