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
        """Project the labelled native tables into canonical datasets."""
        if content_mode not in {"native_text", "mixed"}:
            return {}
        from docmirror.plugins.credit_report.personal_detail_scanned.native_extraction import (
            extract_personal_detail_native_business,
        )

        cached = getattr(parse_result, "native_business", None)
        if callable(cached):
            return cast(dict[str, Any], cached(full_text))
        return extract_personal_detail_native_business(parse_result, full_text)

    def extract_auxiliary_business(
        self,
        parse_result: Any,
        full_text: str,
        *,
        content_mode: str,
    ) -> dict[str, Any]:
        """Reuse the detailed context's scanned evidence pass."""
        if content_mode not in {"scanned_ocr", "mixed"}:
            return {}
        cached = getattr(parse_result, "scanned_business", None)
        if callable(cached):
            auxiliary = cast(dict[str, Any], cached(full_text))
        else:
            auxiliary = super().extract_auxiliary_business(
                parse_result,
                full_text,
                content_mode=content_mode,
            )
        if content_mode == "scanned_ocr":
            native_loader = getattr(parse_result, "native_business", None)
            if callable(native_loader):
                native = cast(dict[str, Any], native_loader(full_text))
                for dataset_name in ("credit_lines", "repayment_liability_records"):
                    if native.get(dataset_name):
                        auxiliary[dataset_name] = list(native[dataset_name])
        return auxiliary

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
        """Complete detailed-report links after the shared canonical merge."""
        assembled = super().assemble_business(
            parse_result,
            full_text,
            content_mode=content_mode,
            existing_collections=existing_collections,
            existing_summary=existing_summary,
            variant_input=variant_input,
        )
        post_merge_corrector = getattr(variant_input, "correct_assembled_business", None)
        if callable(post_merge_corrector):
            assembled = post_merge_corrector(assembled, stage="business_assembly")
        from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
            liability_record_is_substantive,
            make_issue,
            record_issue,
        )

        liability_rows: list[dict[str, Any]] = []
        for index, record in enumerate(assembled.get("repayment_liability_records") or [], start=1):
            if not isinstance(record, dict):
                continue
            if liability_record_is_substantive(record):
                liability_rows.append(record)
                continue
            record_issue(
                variant_input,
                make_issue(
                    category="schema_incompleteness",
                    issue_code="identifier_only_liability_row_suppressed",
                    message=(
                        "A compatibility extractor produced a repayment-responsibility row with no "
                        "business fields; the redundant row was suppressed."
                    ),
                    severity="info",
                    status="suppressed_redundant",
                    parser_stage="business_assembly",
                    target_dataset="repayment_liability_records",
                    target_record_id=str(record.get("record_id") or f"row:{index}"),
                    source_refs=record.get("source_refs") or (),
                    reason_codes=("no_substantive_liability_field", "canonical_row_guard"),
                ),
            )
        if liability_rows or assembled.get("repayment_liability_records"):
            assembled["repayment_liability_records"] = liability_rows
        accounts = [account for account in assembled.get("credit_accounts") or [] if isinstance(account, dict)]
        valid_account_ids = {str(account.get("account_id") or "") for account in accounts if account.get("account_id")}
        repayment_records = [
            dict(record) for record in assembled.get("repayment_records") or [] if isinstance(record, dict)
        ]
        needs_relink = False
        for record in repayment_records:
            account_id = str(record.get("account_id") or "")
            if account_id and account_id not in valid_account_ids:
                needs_relink = True
                record.pop("account_id", None)
                record.pop("account_identifier", None)
                normalized = record.get("normalized")
                if isinstance(normalized, dict):
                    normalized = dict(normalized)
                    normalized.pop("account_id", None)
                    normalized.pop("account_identifier", None)
                    record["normalized"] = normalized
        if needs_relink:
            from docmirror.models.mirror.domain_access import micro_grid_structures_from_domain_specific
            from docmirror.plugins.credit_report.scanned_business import link_repayment_records_to_accounts

            domain_specific = getattr(getattr(parse_result, "entities", None), "domain_specific", {})
            repayment_records = link_repayment_records_to_accounts(
                repayment_records,
                accounts,
                micro_grid_structures_from_domain_specific(
                    domain_specific if isinstance(domain_specific, dict) else {}
                ),
                reading_order_by_logical=dict(getattr(variant_input, "reading_order_by_logical", {}) or {}),
            )
            assembled["repayment_records"] = repayment_records
        account_identifiers = {
            str(account.get("account_id") or ""): account.get("account_identifier")
            for account in accounts
            if account.get("account_id")
        }
        for record in assembled.get("repayment_records") or []:
            if not record.get("account_identifier"):
                record["account_identifier"] = account_identifiers.get(str(record.get("account_id") or ""))
            normalized = record.get("normalized")
            if isinstance(normalized, dict) and not normalized.get("account_identifier"):
                normalized["account_identifier"] = record.get("account_identifier")
        from docmirror.plugins.credit_report.personal_detail_scanned.source_projection import (
            project_typed_public_records,
        )

        reporting_currency: str | None = None
        reporting_amount_unit: str | None = None
        content_loader = getattr(variant_input, "section_content", None)
        if callable(content_loader):
            source_content = cast(dict[str, Any], content_loader(full_text))
            metadata_rows = (source_content.get("datasets") or {}).get("personal_report_metadata") or []
            if metadata_rows:
                metadata = metadata_rows[0].get("normalized", metadata_rows[0])
                if isinstance(metadata, dict):
                    reporting_currency = str(metadata.get("reporting_currency") or "").strip() or None
                    reporting_amount_unit = str(metadata.get("reporting_amount_unit") or "").strip() or None
        for dataset_name, records in project_typed_public_records(
            assembled.get("public_records"),
            reporting_currency=reporting_currency,
            reporting_amount_unit=reporting_amount_unit,
        ).items():
            if records:
                assembled[dataset_name] = records
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
        """Publish typed public-record views derived from the common envelope."""
        typed_names = (
            "tax_arrears_records",
            "civil_judgment_records",
            "enforcement_records",
            "administrative_penalty_records",
            "personal_housing_fund_records",
            "professional_qualification_records",
            "award_records",
        )
        return {name: list(assembled.get(name) or []) for name in typed_names if assembled.get(name)}

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
        """Use the complete personal-detail parser for every detailed report."""
        from docmirror.plugins.credit_report.personal_detail_scanned.source_projection import (
            prepare_personal_detail_source_collections,
        )

        auxiliary = auxiliary_business or {}
        if not getattr(parse_result, "pages", None):
            facts: dict[str, Any] = {}
            if isinstance(auxiliary.get("subject_profile"), dict):
                facts["subject_profile"] = deepcopy(auxiliary["subject_profile"])
            datasets = {
                name: list(auxiliary.get(name) or [])
                for name in ("residence_records", "employment_records", "statements", "annotations")
                if auxiliary.get(name)
            }
            return prepare_personal_detail_source_collections(
                {
                    **({"facts": facts} if facts else {}),
                    **({"datasets": datasets} if datasets else {}),
                },
                auxiliary,
                final_dataset_counts=getattr(parse_result, "_personal_detail_final_dataset_counts", {}),
            )
        from docmirror.plugins.credit_report.personal_detail_scanned.native_extraction import (
            extract_personal_detail_section_content,
        )

        cached = getattr(parse_result, "section_content", None)
        content: dict[str, Any] = (
            cast(dict[str, Any], cached(full_text))
            if callable(cached)
            else extract_personal_detail_section_content(parse_result, full_text)
        )
        from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
            ISSUE_DATASET,
            collect_extraction_issues,
            dataset_states_from_issues,
        )

        issues = collect_extraction_issues(parse_result)
        facts = content.setdefault("facts", {})
        states = facts.setdefault("personal_detail_dataset_states", {})
        if not isinstance(states, dict):
            states = {}
            facts["personal_detail_dataset_states"] = states
        for dataset_name, state in dataset_states_from_issues(issues).items():
            states.setdefault(dataset_name, state)
        if not facts.get("subject_profile") and isinstance(auxiliary.get("subject_profile"), dict):
            facts["subject_profile"] = deepcopy(auxiliary["subject_profile"])
        datasets = content.setdefault("datasets", {})
        if issues:
            datasets[ISSUE_DATASET] = issues
        for name in ("residence_records", "employment_records", "statements", "annotations"):
            if not datasets.get(name) and auxiliary.get(name):
                datasets[name] = list(auxiliary[name])
        return prepare_personal_detail_source_collections(
            content,
            auxiliary,
            final_dataset_counts=getattr(parse_result, "_personal_detail_final_dataset_counts", {}),
        )

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
