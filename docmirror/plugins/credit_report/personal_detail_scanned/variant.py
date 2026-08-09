# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Orchestration adapter for scanned personal detailed credit reports."""

import re
from copy import deepcopy
from typing import Any, cast

from docmirror.plugins.credit_report.shared.variant import CreditReportVariantAdapter

_CANONICAL_SECTION_ROLES = (
    (("report_header_and_identity",), "sec_personal_basic", "个人基本信息", "basic_information"),
    (("information_summary",), "sec_credit_summary", "信息概要", "credit_summary"),
    (
        ("credit_account_detail", "repayment_responsibility", "credit_agreement"),
        "sec_credit_details",
        "信贷交易信息明细",
        "credit_details",
    ),
    (("postpaid_detail",), "sec_non_credit_transactions", "非信贷交易信息明细", "non_credit_transactions"),
    (("public_information",), "sec_public_records", "公共信息明细", "public_records"),
    (("report_explanation",), "sec_report_explanation", "报告说明与编制说明", "report_explanation"),
)
_CANONICAL_TEMPLATE_IDS = frozenset(
    template_id
    for template_ids, _section_id, _title, _section_type in _CANONICAL_SECTION_ROLES
    for template_id in template_ids
) | {"annotations_and_inquiries"}

_FINAL_V2_SECTION_FIELD_SOURCES: dict[str, tuple[tuple[str, str], ...]] = {
    "subject_name": (
        ("report_metadata", "subject_name"),
        ("subject_profile", "subject_name"),
        ("subject_identity_documents", "holder_name"),
    ),
    "id_number": (
        ("subject_identity_documents", "document_number"),
        ("report_metadata", "primary_id_number"),
        ("subject_profile", "primary_id_number"),
    ),
    "id_type": (
        ("subject_identity_documents", "document_type"),
        ("report_metadata", "primary_id_type"),
        ("subject_profile", "primary_id_type"),
    ),
    "marital_status": (("subject_profile", "marital_status"),),
    "query_institution": (("report_query", "query_institution"),),
    "report_time": (("report_metadata", "report_time"),),
    "report_number": (("report_metadata", "report_number"),),
}


def _page_source_text(page: Any) -> str:
    values = [str(getattr(block, "content", "") or "") for block in getattr(page, "texts", None) or ()]
    for table in getattr(page, "tables", None) or ():
        metadata = getattr(table, "metadata", None) or {}
        rows = metadata.get("raw_rows") if isinstance(metadata, dict) else None
        if isinstance(rows, list):
            values.extend(str(cell or "") for row in rows if isinstance(row, list) for cell in row)
    return re.sub(r"\s+", "", " ".join(values))


def _classified_sections(parse_result: Any) -> tuple[dict[str, Any], ...]:
    """Advertise only sections backed by a registered canonical page role.

    Raw page-count ranges and generic section reconstruction are intentionally
    excluded.  The annotations/inquiries template contains three optional
    source sections, so those are additionally gated by an observed canonical
    heading inside a page already registered to that template.
    """

    pages = list(getattr(parse_result, "pages", None) or ())
    template_text: dict[str, list[tuple[int, str]]] = {}
    registered: dict[str, set[int]] = {}
    for index, page in enumerate(pages, start=1):
        page_number = int(getattr(page, "page_number", 0) or index)
        text = _page_source_text(page)
        template_id = str(getattr(page, "canonical_template_id", "") or "")
        if template_id not in _CANONICAL_TEMPLATE_IDS:
            continue
        registered.setdefault(template_id, set()).add(page_number)
        template_text.setdefault(template_id, []).append((page_number, text))

    audit_loader = getattr(parse_result, "canonical_layout_audit", None)
    if callable(audit_loader):
        audit = audit_loader()
        registrations = audit.get("registrations") if isinstance(audit, dict) else ()
        for registration in registrations or ():
            if not isinstance(registration, dict) or registration.get("status") != "registered":
                continue
            template_id = str(registration.get("template_id") or "")
            logical_page = int(registration.get("logical_page") or 0)
            if template_id in _CANONICAL_TEMPLATE_IDS and logical_page > 0:
                registered.setdefault(template_id, set()).add(logical_page)

    if not registered:
        return ()

    sections: list[dict[str, Any]] = []
    for template_ids, section_id, title, section_type in _CANONICAL_SECTION_ROLES:
        matches = sorted(
            page_number
            for template_id in template_ids
            for page_number in registered.get(template_id, ())
        )
        if not matches:
            continue
        sections.append(
            {
                "id": section_id,
                "title": title,
                "type": section_type,
                "page_start": min(matches),
                "page_end": max(matches),
            }
        )

    optional_roles = (
        (
            "sec_inquiries",
            "查询记录",
            "inquiries",
            ("机构查询记录明细", "本人查询记录明细"),
            frozenset({"annotations_and_inquiries"}),
        ),
        (
            "sec_statements",
            "机构说明与本人声明",
            "statements",
            ("机构说明", "本人声明"),
            _CANONICAL_TEMPLATE_IDS - {"report_explanation"},
        ),
        (
            "sec_annotations",
            "异议标注",
            "annotations",
            ("异议标注", "异议处理中"),
            _CANONICAL_TEMPLATE_IDS - {"report_explanation"},
        ),
    )
    for section_id, title, section_type, markers, allowed_templates in optional_roles:
        matches = [
            page_number
            for template_id in allowed_templates
            for page_number, text in template_text.get(template_id, ())
            if any(marker in text for marker in markers)
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
    return tuple(sorted(sections, key=lambda section: (section["page_start"], section["id"])))


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
                status_row["presence_status"] = "unknown"
                status_row["reason"] = "source_presence_not_established"

    def reconcile_final_v2_section_fields(
        self,
        domain_facts: dict[str, Any],
        field_details: dict[str, Any],
        datasets: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any]:
        """Make Community section scalars a view of the finalized PBOC v2 rows."""
        resolved: dict[str, Any] = {}
        resolved_details: dict[str, dict[str, Any]] = {}
        for target_field, candidates in _FINAL_V2_SECTION_FIELD_SOURCES.items():
            value: Any = None
            detail: dict[str, Any] | None = None
            for dataset_name, source_field in candidates:
                for record in datasets.get(dataset_name) or []:
                    values = record.get("normalized") if isinstance(record.get("normalized"), dict) else record
                    candidate = values.get(source_field)
                    if candidate in (None, ""):
                        continue
                    value = deepcopy(candidate)
                    raw_values = (
                        record.get("canonical_raw")
                        if isinstance(record.get("canonical_raw"), dict)
                        else record.get("raw") if isinstance(record.get("raw"), dict) else {}
                    )
                    detail = {
                        "source": "personal_detail_final_v2",
                        "source_dataset": dataset_name,
                        "source_record_id": record.get("record_id"),
                        "raw": deepcopy(raw_values.get(source_field, candidate)),
                    }
                    if record.get("confidence") is not None:
                        detail["confidence"] = record["confidence"]
                    break
                if detail is not None:
                    break
            resolved[target_field] = value
            domain_facts[target_field] = value
            if detail is None:
                field_details.pop(target_field, None)
            else:
                field_details[target_field] = detail
                resolved_details[target_field] = detail

        # Community's entity vocabulary calls the same finalized identifier
        # ``subject_id``. Keep the alias authoritative too, including None,
        # so a stale sealed/generic identity cannot repopulate the section.
        resolved["subject_id"] = resolved.get("id_number")
        domain_facts["subject_id"] = resolved["subject_id"]
        if "id_number" in resolved_details:
            field_details["subject_id"] = {
                **deepcopy(resolved_details["id_number"]),
                "source": "personal_detail_final_v2_alias",
            }
        else:
            field_details.pop("subject_id", None)
        domain_facts["field_details"] = field_details
        return resolved

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
        """Return only source-observed sections from canonical registrations."""
        del full_text
        return _classified_sections(parse_result)

    def semantic_extensions(self) -> dict[str, Any]:
        """Declare PBOC v2 datasets as the detailed report's canonical storage."""
        from docmirror.plugins.credit_report.personal_detail_scanned.schema import (
            personal_detail_semantic_extensions,
        )

        return personal_detail_semantic_extensions()


variant = PersonalDetailScannedVariant()

__all__ = ["PersonalDetailScannedVariant", "variant"]
