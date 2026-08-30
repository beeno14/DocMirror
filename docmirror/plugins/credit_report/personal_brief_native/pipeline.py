# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single canonical pipeline for PBOC personal brief credit reports."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import partial
from types import SimpleNamespace
from typing import Any, Callable

from docmirror.plugins.credit_report.personal_brief_native.account_rules import (
    ACCOUNT_ACTION_PATTERN as _ACCOUNT_ACTION_PATTERN,
)
from docmirror.plugins.credit_report.personal_brief_native.account_rules import (
    ACCOUNT_DATE_PATTERN as _ACCOUNT_DATE_PATTERN,
)
from docmirror.plugins.credit_report.personal_brief_native.account_rules import (
    OTHER_BUSINESS_TYPES as _OTHER_BUSINESS_TYPES,
)
from docmirror.plugins.credit_report.personal_brief_native.account_rules import (
    account_narratives,
    account_source_fields,
)
from docmirror.plugins.credit_report.personal_brief_native.context import (
    extract_personal_brief_lookback_years,
)
from docmirror.plugins.credit_report.personal_brief_native.contracts import (
    PERSONAL_BRIEF_REPORTING_AMOUNT_UNIT,
    canonicalize_personal_brief_reporting_units,
)
from docmirror.plugins.credit_report.personal_brief_native.date_rules import (
    PERSONAL_BRIEF_DATE_PATTERN,
    PERSONAL_BRIEF_MONTH_OR_DATE_PATTERN,
    PERSONAL_BRIEF_MONTH_PATTERN,
    PERSONAL_BRIEF_YEAR_PATTERN,
)
from docmirror.plugins.credit_report.personal_brief_native.ir import (
    CanonicalPersonalBriefComponent,
    CanonicalPersonalBriefDocumentIR,
    PersonalBriefSourceRef,
    build_canonical_personal_brief_document,
)


@dataclass(frozen=True, init=False)
class PersonalBriefSemanticDocument:
    facts: dict[str, Any]
    datasets: dict[str, list[dict[str, Any]]]
    credit_summary: dict[str, Any]
    extraction_report: dict[str, Any]
    dataset_completeness: dict[str, dict[str, Any]]
    _credit_extraction_audit: dict[str, Any] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _credit_extraction_audit_factory: Callable[[], dict[str, Any]] | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __init__(
        self,
        facts: dict[str, Any],
        datasets: dict[str, list[dict[str, Any]]],
        credit_summary: dict[str, Any],
        credit_extraction_audit: dict[str, Any] | None,
        extraction_report: dict[str, Any],
        dataset_completeness: dict[str, dict[str, Any]],
        *,
        credit_extraction_audit_factory: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        object.__setattr__(self, "facts", facts)
        object.__setattr__(self, "datasets", datasets)
        object.__setattr__(self, "credit_summary", credit_summary)
        object.__setattr__(self, "extraction_report", extraction_report)
        object.__setattr__(self, "dataset_completeness", dataset_completeness)
        object.__setattr__(self, "_credit_extraction_audit", credit_extraction_audit)
        object.__setattr__(
            self,
            "_credit_extraction_audit_factory",
            credit_extraction_audit_factory,
        )

    @property
    def credit_extraction_audit(self) -> dict[str, Any]:
        """Materialize the legacy diagnostic audit only when it is requested."""
        audit = self._credit_extraction_audit
        if audit is None:
            factory = self._credit_extraction_audit_factory
            audit = factory() if factory is not None else {}
            object.__setattr__(self, "_credit_extraction_audit", audit)
            object.__setattr__(self, "_credit_extraction_audit_factory", None)
        return audit

    def to_debug_payload(self) -> dict[str, Any]:
        return {
            "schema": {"id": "personal_brief_credit_report", "version": "2.0.0"},
            "facts": self.facts,
            "credit_summary": self.credit_summary,
            "credit_extraction_audit": self.credit_extraction_audit,
            "extraction": self.extraction_report,
            "dataset_completeness": self.dataset_completeness,
            "datasets": self.datasets,
        }

    def to_debug_json(self) -> str:
        return json.dumps(self.to_debug_payload(), ensure_ascii=False, sort_keys=True, default=str)


@dataclass(frozen=True, init=False)
class PersonalBriefPipelineArtifacts:
    document_ir: CanonicalPersonalBriefDocumentIR
    semantic_document: PersonalBriefSemanticDocument
    _ir_debug_json: str | None = field(default=None, repr=False, compare=False)
    _semantic_debug_json: str | None = field(default=None, repr=False, compare=False)

    def __init__(
        self,
        document_ir: CanonicalPersonalBriefDocumentIR,
        semantic_document: PersonalBriefSemanticDocument,
        ir_debug_json: str | None = None,
        semantic_debug_json: str | None = None,
    ) -> None:
        object.__setattr__(self, "document_ir", document_ir)
        object.__setattr__(self, "semantic_document", semantic_document)
        object.__setattr__(self, "_ir_debug_json", ir_debug_json)
        object.__setattr__(self, "_semantic_debug_json", semantic_debug_json)

    @property
    def ir_debug_json(self) -> str:
        value = self._ir_debug_json
        if value is None:
            value = self.document_ir.to_debug_json()
            object.__setattr__(self, "_ir_debug_json", value)
        return value

    @property
    def semantic_debug_json(self) -> str:
        value = self._semantic_debug_json
        if value is None:
            value = self.semantic_document.to_debug_json()
            object.__setattr__(self, "_semantic_debug_json", value)
        return value


def _source_ref(
    ref: PersonalBriefSourceRef,
    source: str,
    *,
    component_id: str = "",
) -> dict[str, Any]:
    payload: dict[str, Any] = {"source": source, "page": ref.source_page}
    if component_id:
        payload["component_id"] = component_id
    if ref.unit_id:
        payload["unit_id"] = ref.unit_id
    if ref.bbox is not None:
        payload["bbox"] = list(ref.bbox)
    if ref.node_ids:
        payload["node_ids"] = list(ref.node_ids)
    if ref.evidence_ids:
        payload["evidence_ids"] = list(ref.evidence_ids)
    if ref.table_id:
        payload["table_id"] = ref.table_id
    if ref.row_index is not None:
        payload["row_index"] = ref.row_index
    return payload


def _component_refs(
    components: list[CanonicalPersonalBriefComponent],
    source: str,
) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int | None]] = set()
    for component in components:
        for ref in component.source_refs:
            key = (component.component_id, ref.unit_id, ref.row_index)
            if key in seen:
                continue
            seen.add(key)
            refs.append(
                _source_ref(
                    ref,
                    source,
                    component_id=component.component_id,
                )
            )
    return refs


def _compact(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or ""))


def _record_tokens(value: Any) -> tuple[str, ...]:
    """Return stable source-search tokens from one structured business row."""
    tokens: list[str] = []

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for key, nested in item.items():
                if key not in {
                    "source",
                    "source_refs",
                    "source_cell_refs",
                    "evidence_ids",
                    "confidence",
                    "normalized",
                } and not key.endswith("_status"):
                    visit(nested)
            return
        if isinstance(item, (list, tuple, set)):
            for nested in item:
                visit(nested)
            return
        text = _compact(item)
        if len(text) >= 4 and text not in {"reported", "not_reported", "unresolved"}:
            tokens.append(text)
            if len(text) > 24:
                tokens.extend(
                    text[start : start + 16]
                    for start in range(0, len(text), 16)
                    if len(text[start : start + 16]) >= 6
                )

    visit(value)
    return tuple(dict.fromkeys(sorted(tokens, key=len, reverse=True)))


def _attach_canonical_provenance(
    document: CanonicalPersonalBriefDocumentIR,
    records: list[dict[str, Any]],
    section_keys: tuple[str, ...],
    source: str,
) -> None:
    """Bind every emitted row to exact canonical units, never page labels alone."""
    components = [
        component
        for section_key in section_keys
        for component in document.components_for(section_key)
        if component.source_refs
    ]
    if not components:
        return
    for record in records:
        if _valid_record_provenance(document, record):
            continue
        page_hints = {
            int(ref.get("page") or 0)
            for ref in record.get("source_refs") or []
            if isinstance(ref, dict) and ref.get("unit_id") and ref.get("page")
        }
        page_candidates = [
            component
            for component in components
            if not page_hints or page_hints.intersection(component.source_pages)
        ] or components
        tokens = _record_tokens(record)
        scored = [
            (
                sum(len(token) for token in tokens if token in _compact(component.text)),
                -component.global_order,
                component,
            )
            for component in page_candidates
        ]
        best_score = max((item[0] for item in scored), default=0)
        if best_score <= 0:
            continue
        selected = [item[2] for item in scored if item[0] == best_score][:3]
        refs = _component_refs(selected, source)
        if not refs:
            continue
        record["source_refs"] = refs
        record["evidence_ids"] = list(
            dict.fromkeys(
                evidence_id
                for ref in refs
                for evidence_id in ref.get("evidence_ids") or []
            )
        )


def _account_section_text(
    document: CanonicalPersonalBriefDocumentIR,
    section_key: str,
) -> str:
    """Linearize one account section without leaking printed ordinals into amounts."""
    parts: list[str] = []
    ordinal_only = re.compile(r"^\s*(?:\d{1,4}[.、]?\s*)+$")
    for component in document.components_for(section_key):
        if component.kind == "logical_table":
            continue
        value = component.text
        if component.kind == "numbered_record":
            if ordinal_only.fullmatch(value):
                continue
            value = re.sub(
                rf"^\s*\d{{1,4}}[.、]\s*(?={PERSONAL_BRIEF_YEAR_PATTERN}年)",
                "",
                value,
            )
        if value.strip():
            parts.append(value)
    text = re.sub(r"\s+", " ", "\n".join(parts)).strip()
    # Personal Brief explicitly reports monetary fields to whole yuan. OCR may
    # substitute a full stop for the thousands comma (for example 119.652).
    return re.sub(r"(?<=\d)\.(?=\d{3}(?!\d))", ",", text)


def _inquiry_records(document: CanonicalPersonalBriefDocumentIR) -> list[dict[str, Any]]:
    from docmirror.plugins.credit_report.value_utils import stable_record_id

    records: list[dict[str, Any]] = []
    for inquiry_type, role in (
        ("institution", "institution_inquiries"),
        ("personal", "personal_inquiries"),
    ):
        table = document.logical_table(role)
        if table is None:
            continue
        for row in table.rows:
            sequence_text, query_date, institution, source_reason = (*row.values, "", "", "", "")[:4]
            try:
                sequence = int(sequence_text)
            except (TypeError, ValueError):
                continue
            normalized_reason = "本人查询" if inquiry_type == "personal" else source_reason
            refs = [_source_ref(ref, "canonical_personal_brief_inquiry_table") for ref in row.source_refs]
            evidence_ids = list(
                dict.fromkeys(
                    evidence_id
                    for ref in row.source_refs
                    for evidence_id in ref.evidence_ids
                )
            )
            query_channel_match = re.search(r"[（(]([^）)]+)[）)]", source_reason)
            record = {
                "inquiry_id": stable_record_id(
                    "credit_inquiry",
                    inquiry_type,
                    sequence,
                    query_date,
                    institution,
                    source_reason,
                ),
                "sequence": sequence,
                "inquiry_type": inquiry_type,
                "inquiry_date": query_date or None,
                "inquiry_date_status": "reported" if query_date else "unresolved",
                "institution": institution or None,
                "institution_status": "reported" if institution else "unresolved",
                "reason": normalized_reason or None,
                "reason_status": "reported" if normalized_reason else "unresolved",
                "source_reason": source_reason or None,
                "source_reason_status": "reported" if source_reason else "unresolved",
                "source": "canonical_personal_brief_inquiry_table",
                "source_refs": refs,
                "evidence_ids": evidence_ids,
                "extraction_status": row.status,
                "confidence": 0.99 if row.status == "reported" else 0.55,
            }
            if query_channel_match:
                record["query_channel"] = query_channel_match.group(1)
            records.append(record)
    return sorted(
        records,
        key=lambda item: (
            0 if item.get("inquiry_type") == "institution" else 1,
            int(item.get("sequence") or 0),
        ),
    )


def _inquiry_scope(document: CanonicalPersonalBriefDocumentIR) -> dict[str, Any]:
    """Read the source's inquiry scope even when its sentence crosses units/pages."""

    text = ""
    ranges: list[tuple[int, int, CanonicalPersonalBriefComponent]] = []
    for component in document.components:
        if component.kind == "logical_table" or component.section_key not in {
            "inquiries_container", "institution_inquiries", "personal_inquiries"
        }:
            continue
        start = len(text)
        text += _compact(component.text)
        ranges.append((start, len(text), component))
    match = re.search(r"这部分包含您的信用报告最近(?P<years>\d+)年内被查询的记录[。.]?", text)
    if match is None:
        return {}
    pages = sorted({
        page
        for start, end, component in ranges
        if start < match.end() and end > match.start()
        for page in component.source_pages
    })
    return {
        "lookback_years": int(match.group("years")),
        "source_statement": match.group(0),
        "source_pages": pages,
    }


def _numbered_chunks(
    document: CanonicalPersonalBriefDocumentIR,
    section_key: str,
) -> list[tuple[int, str, list[CanonicalPersonalBriefComponent]]]:
    chunks: list[tuple[int, str, list[CanonicalPersonalBriefComponent]]] = []
    sequence = 0
    texts: list[str] = []
    components: list[CanonicalPersonalBriefComponent] = []
    for component in document.components_for(section_key):
        if component.kind == "logical_table" or not component.text.strip():
            continue
        lines = [line.strip() for line in component.text.splitlines() if line.strip()]
        if len(lines) > 1 and all(re.fullmatch(r"\d{1,4}[.、]", line) for line in lines):
            # Native PDFs may expose the visual ordinal column as one tall text
            # atom (``7.\n8.\n...``). It is not a record body; the independent
            # narrative-boundary observer handles the corresponding rows.
            continue
        match = re.match(r"^\s*(\d{1,4})[.、]\s*(.*)$", component.text, re.DOTALL)
        if match:
            if sequence:
                chunks.append((sequence, "\n".join(texts), components))
            sequence = int(match.group(1))
            texts = [match.group(2)]
            components = [component]
        elif sequence:
            texts.append(component.text)
            components.append(component)
    if sequence:
        chunks.append((sequence, "\n".join(texts), components))
    return chunks


def _valid_account_boundary_count(text: str) -> int:
    """Count record starts independently from the business row decoder."""
    return len(account_narratives(text))


def _account_boundary_expectations(
    document: CanonicalPersonalBriefDocumentIR,
) -> dict[str, int]:
    expected: dict[str, int] = {}
    for section_key in ("credit_cards", "loans", "other_business"):
        numbered = [sequence for sequence, _text, _components in _numbered_chunks(document, section_key)]
        numbered_count = max(numbered, default=0)
        detected_count = _valid_account_boundary_count(_account_section_text(document, section_key))
        expected[section_key] = max(numbered_count, detected_count)
    return expected


def _partial_account_record(
    section_key: str,
    source_sequence: int,
    text: str,
    components: list[CanonicalPersonalBriefComponent],
) -> dict[str, Any]:
    from docmirror.plugins.credit_report.business_records import (
        _iso_date,
        _iso_month,
        _personal_brief_account_currency,
    )
    from docmirror.plugins.credit_report.value_utils import parse_number, stable_record_id

    compact = _compact(text)
    date_match = re.search(_ACCOUNT_DATE_PATTERN, compact)
    snapshot_match = re.search(rf"({PERSONAL_BRIEF_MONTH_PATTERN})", compact)
    institution_match = re.search(
        rf"{_ACCOUNT_DATE_PATTERN}(.{{4,80}}?){_ACCOUNT_ACTION_PATTERN}",
        compact,
    )
    limit_match = re.search(r"信用额度([\d,.]+)", compact)
    balance_match = re.search(r"余额(?:为)?([\d,.]+)", compact)
    loan_amount_match = re.search(r"发放(?:的)?([\d,.]+)元", compact)
    business_type = next((value for value in _OTHER_BUSINESS_TYPES if value in compact), None)
    if section_key == "credit_cards":
        account_type = "credit_card"
        business_type = business_type or ("准贷记卡" if "准贷记卡" in compact else "贷记卡" if "贷记卡" in compact else None)
    elif section_key == "other_business":
        account_type = "other_business"
    else:
        account_type = "credit_line" if "授信" in compact or "信用额度" in compact else "loan"
        if business_type is None:
            loan_type = re.search(r"([\u3400-\u9fff（）()]{1,24}贷款)", compact)
            business_type = loan_type.group(1) if loan_type else None
    refs = _component_refs(components, "canonical_numbered_account_fragment")
    return {
        "account_id": stable_record_id(
            "credit_account_partial",
            section_key,
            source_sequence,
            compact,
        ),
        "sequence": source_sequence,
        "source_sequence": source_sequence,
        "source_section": section_key,
        "account_type": account_type,
        "account_type_status": "inferred_from_canonical_section",
        "business_category": "other_business" if section_key == "other_business" else section_key,
        "business_type": business_type,
        "business_type_status": "reported" if business_type else "unresolved",
        "management_institution": institution_match.group(1) if institution_match else None,
        "management_institution_status": "reported" if institution_match else "unresolved",
        "open_date": _iso_date(date_match.group(0)) if date_match else None,
        "open_date_status": "reported" if date_match else "unresolved",
        "currency": _personal_brief_account_currency(compact),
        "account_currency": _personal_brief_account_currency(compact),
        "reporting_amount_currency": "CNY",
        "amount_unit": PERSONAL_BRIEF_REPORTING_AMOUNT_UNIT,
        "reporting_amount_unit": PERSONAL_BRIEF_REPORTING_AMOUNT_UNIT,
        "credit_limit": parse_number(limit_match.group(1)) if limit_match else None,
        "loan_amount": parse_number(loan_amount_match.group(1)) if loan_amount_match else None,
        "balance": parse_number(balance_match.group(1)) if balance_match else None,
        "information_as_of": _iso_month(snapshot_match.group(1)) if snapshot_match else None,
        "current_overdue": False if "当前无逾期" in compact else True if "当前有逾期" in compact else None,
        "account_status": "active" if "当前无逾期" in compact else "unknown",
        "account_lifecycle_state": "open" if "当前无逾期" in compact else "unknown",
        "card_activation_state": "not_reported" if section_key == "credit_cards" else "not_applicable",
        "credit_quality_status": "unresolved",
        "source": "canonical_numbered_account_fragment",
        "source_refs": refs,
        "confidence": 0.58,
        "extraction_status": "unresolved",
    }


def _reconcile_numbered_accounts(
    document: CanonicalPersonalBriefDocumentIR,
    section_key: str,
    existing: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Use a complete printed ordinal lattice when the IR exposes one."""
    from docmirror.plugins.credit_report.business_records import (
        _personal_brief_account_from_chunk,
    )
    from docmirror.plugins.credit_report.value_utils import stable_record_id

    chunks = _numbered_chunks(document, section_key)
    sequences = [sequence for sequence, _text, _components in chunks]
    if not sequences:
        return existing
    expected = max(sequences)
    # Some native parsers bundle several numbered narratives into one text
    # component.  In that case the ordinal lattice is not granular enough to
    # replace decoded rows; its maximum still remains an independent count.
    if (
        len(existing) >= expected
        or
        sorted(set(sequences)) != list(range(1, expected + 1))
        or expected < _valid_account_boundary_count(_account_section_text(document, section_key))
    ):
        return existing
    page_texts = [
        (
            page.page_number,
            re.sub(r"\s+", " ", "\n".join(block.content for block in page.texts)).strip(),
            re.sub(r"\s+", "", "\n".join(block.content for block in page.texts)),
        )
        for page in document.pages
    ]
    reconciled: list[dict[str, Any]] = []
    for source_sequence, text, components in chunks:
        parsed = _personal_brief_account_from_chunk(
            re.sub(r"\s+", " ", text).strip(),
            page_texts,
        )
        if parsed is None:
            parsed = _partial_account_record(
                section_key,
                source_sequence,
                text,
                components,
            )
        else:
            parsed["account_id"] = stable_record_id(
                "credit_account",
                parsed.get("account_id"),
                section_key,
                source_sequence,
            )
            parsed["sequence"] = source_sequence
            parsed["source_sequence"] = source_sequence
            parsed["source_section"] = section_key
            parsed["business_category"] = (
                "other_business" if section_key == "other_business" else section_key
            )
            if section_key == "other_business":
                compact = _compact(text)
                parsed["account_type"] = "other_business"
                parsed["business_type"] = next(
                    (value for value in _OTHER_BUSINESS_TYPES if value in compact),
                    parsed.get("business_type"),
                )
            parsed["source_refs"] = _component_refs(
                components,
                "canonical_numbered_account_record",
            )
        reconciled.append(parsed)
    return reconciled


def _extract_account_section(
    document: CanonicalPersonalBriefDocumentIR,
    section_key: str,
    page_texts: list[tuple[int, str, str]],
) -> list[dict[str, Any]]:
    from docmirror.plugins.credit_report.business_records import (
        _personal_brief_account_from_chunk,
        _personal_brief_accounts,
    )

    section_text = _account_section_text(document, section_key)
    if section_key != "other_business":
        records = _personal_brief_accounts(section_text, page_texts)
    else:
        records = []
        for narrative in account_narratives(section_text):
            record = _personal_brief_account_from_chunk(
                narrative.text,
                page_texts,
            )
            if record is None:
                continue
            compact = _compact(narrative.text)
            record["account_type"] = "other_business"
            record["business_type"] = next(
                (value for value in _OTHER_BUSINESS_TYPES if value in compact),
                record.get("business_type"),
            )
            records.append(record)
        for sequence, record in enumerate(records, start=1):
            record["sequence"] = sequence
    for record in records:
        record["source_section"] = section_key
        record["source_sequence"] = int(record.get("sequence") or 0)
        record["business_category"] = (
            "other_business" if section_key == "other_business" else section_key
        )
    return _reconcile_numbered_accounts(document, section_key, records)


def _repayment_liabilities_from_canonical_records(
    document: CanonicalPersonalBriefDocumentIR,
) -> list[dict[str, Any]]:
    from docmirror.plugins.credit_report.business_records import _iso_date, _iso_month
    from docmirror.plugins.credit_report.value_utils import parse_number, stable_record_id

    chunks: list[tuple[int, str, list[CanonicalPersonalBriefComponent]]] = []
    for component in document.components_for("repayment_liability"):
        if component.semantic_role != "repayment_liability_record" or not component.rows:
            continue
        row = component.rows[0]
        if len(row.values) < 2 or not str(row.values[0]).isdigit():
            continue
        chunks.append((int(row.values[0]), str(row.values[1]), [component]))
    records: list[dict[str, Any]] = []
    for source_sequence, text, components in chunks:
        compact = re.sub(r"\s+", "", text)
        core = re.search(
            rf"^({PERSONAL_BRIEF_DATE_PATTERN})[，,]?为"
            r"(.+?)[（(]证件类型[:：](.+?)[，,]证件号码[:：](.+?)[）)]"
            r"在(.+?)(?:办理|办.{1,20}?理)(?:的)?(.+?)承担相关还款责任[，,]"
            r"责任人类型为(.+?)[，,]相关还款责任金额([\d,.]+|--)",
            compact,
        )
        if not core:
            continue
        contract_match = re.search(r"保证合同编号[:：]([^）)]+)", compact)
        snapshot_match = re.search(
            rf"截至({PERSONAL_BRIEF_MONTH_OR_DATE_PATTERN})[，,]",
            compact,
        )
        snapshot_business = re.search(
            rf"截至{PERSONAL_BRIEF_MONTH_OR_DATE_PATTERN}[，,](.+?)余额",
            compact,
        )
        balance_match = re.search(r"余额(?:为)?([\d,.]+)", compact)
        amount_text = core.group(8)
        refs = _component_refs(components, "canonical_repayment_liability_record")
        if (
            len(components) == 1
            and components[0].semantic_role == "repayment_liability_record"
            and components[0].rows
        ):
            refs = [
                _source_ref(
                    ref,
                    "canonical_repayment_liability_record",
                    component_id=components[0].component_id,
                )
                for ref in components[0].rows[0].source_refs
            ]
        extraction_status = (
            components[0].rows[0].status
            if len(components) == 1 and components[0].rows
            else "unresolved"
        )
        records.append(
            {
                "liability_id": stable_record_id(
                    "repayment_liability",
                    core.group(1),
                    core.group(4),
                    core.group(5),
                    contract_match.group(1) if contract_match else "",
                    source_sequence,
                ),
                "sequence": source_sequence,
                "liability_date": _iso_date(core.group(1)),
                "related_party_name": core.group(2),
                "related_party_id_type": core.group(3),
                "related_party_id_number": core.group(4),
                "management_institution": core.group(5),
                "business_type": core.group(6),
                "underlying_business_type": core.group(6),
                "snapshot_balance_business_type": (
                    snapshot_business.group(1) if snapshot_business else core.group(6)
                ),
                "responsibility_type": core.group(7),
                "responsibility_amount": parse_number(amount_text),
                "responsibility_amount_reported": amount_text != "--",
                **(
                    {"contract_number": contract_match.group(1)}
                    if contract_match
                    else {}
                ),
                "snapshot_date": (
                    _iso_date(snapshot_match.group(1)) or _iso_month(snapshot_match.group(1))
                    if snapshot_match
                    else None
                ),
                "balance": parse_number(balance_match.group(1)) if balance_match else None,
                "currency": "CNY",
                "reporting_amount_currency": "CNY",
                "amount_unit": PERSONAL_BRIEF_REPORTING_AMOUNT_UNIT,
                "reporting_amount_unit": PERSONAL_BRIEF_REPORTING_AMOUNT_UNIT,
                "source": "canonical_repayment_liability_record",
                "source_refs": refs,
                "extraction_status": extraction_status,
                "confidence": min(
                    0.96,
                    float(components[0].confidence) if len(components) == 1 else 0.55,
                ),
            }
        )
    return records


def _statement_after(
    blocks: list[tuple[int, str]],
    heading: str,
) -> tuple[int, str]:
    from docmirror.plugins.credit_report.value_utils import compact_text

    for index, (page, content) in enumerate(blocks):
        if compact_text(content) != heading:
            continue
        for next_page, candidate in blocks[index + 1 :]:
            if next_page != page:
                break
            compact = compact_text(candidate)
            if compact and not _PAGE_NUMBER_RE.fullmatch(compact):
                return page, re.sub(r"\s+", " ", candidate).strip()
    return 0, ""


_PAGE_NUMBER_RE = re.compile(r"第\s*\d+\s*页\s*[,，]\s*共\s*\d+\s*页")


def _report_notes(blocks: list[tuple[int, str]]) -> list[dict[str, Any]]:
    from docmirror.plugins.credit_report.value_utils import compact_text, stable_record_id

    notes: list[dict[str, Any]] = []
    for index, (page, content) in enumerate(blocks):
        if compact_text(content) != "说明":
            continue
        note_text = "\n".join(
            candidate
            for next_page, candidate in blocks[index + 1 :]
            if next_page == page and not _PAGE_NUMBER_RE.fullmatch(compact_text(candidate))
        )
        for match in re.finditer(r"(?ms)(\d+)\.\s*(.*?)(?=^\d+\.|\Z)", note_text):
            sequence = int(match.group(1))
            value = re.sub(r"\s+", " ", match.group(2)).strip()
            if not value:
                continue
            notes.append(
                {
                    "note_id": stable_record_id("credit_report_note", sequence, value),
                    "sequence": sequence,
                    "content": value,
                    "source": "canonical_personal_brief_notes",
                    "source_refs": [{"source": "canonical_personal_brief_notes", "page": page}],
                    "confidence": 1.0,
                }
            )
        break
    return notes


def _institution_statement_records(
    document: CanonicalPersonalBriefDocumentIR,
) -> list[dict[str, Any]]:
    """Decode the optional report-wide ``机构说明`` table into two rigid fields."""
    from docmirror.plugins.credit_report.business_records import _iso_date
    from docmirror.plugins.credit_report.value_utils import stable_record_id

    components = list(document.components_for("institution_statements"))
    records: list[dict[str, Any]] = []

    def append(content: str, added_date: str, sources: list[CanonicalPersonalBriefComponent]) -> None:
        normalized_content = re.sub(r"\s+", " ", content).strip(" ：:，,。")
        normalized_date = _iso_date(added_date)
        if not normalized_content or not normalized_date:
            return
        identity = stable_record_id("institution_statement", normalized_content, normalized_date)
        if any(record.get("institution_statement_id") == identity for record in records):
            return
        records.append(
            {
                "institution_statement_id": identity,
                "sequence": len(records) + 1,
                "statement_content": normalized_content,
                "added_date": normalized_date,
                "source": "canonical_institution_statement",
                "source_refs": _component_refs(sources, "canonical_institution_statement"),
                "confidence": 0.98,
            }
        )

    for component in components:
        rows = [list(row.values) for row in component.rows]
        header_index = next(
            (
                index
                for index, row in enumerate(rows)
                if any("说明内容" in _compact(cell) for cell in row)
                and any("添加日期" in _compact(cell) for cell in row)
            ),
            None,
        )
        if header_index is None:
            continue
        header = [_compact(value) for value in rows[header_index]]
        content_column = next(index for index, value in enumerate(header) if "说明内容" in value)
        date_column = next(index for index, value in enumerate(header) if "添加日期" in value)
        for row in rows[header_index + 1 :]:
            if max(content_column, date_column) >= len(row):
                continue
            append(row[content_column], row[date_column], [component])

    section_text = "\n".join(
        component.text
        for component in components
        if component.kind != "logical_table" and _compact(component.text) != "机构说明"
    )
    for match in re.finditer(
        rf"说明内容\s*[:：]\s*(.+?)\s*添加日期\s*[:：]\s*({_ACCOUNT_DATE_PATTERN})",
        section_text,
        re.DOTALL,
    ):
        append(match.group(1), match.group(2), components)
    for _sequence, text, sources in _numbered_chunks(document, "institution_statements"):
        date_match = re.search(_ACCOUNT_DATE_PATTERN, text)
        if date_match is None:
            continue
        content = re.sub(r"(?:说明内容|添加日期)\s*[:：]?", "", text)
        content = content.replace(date_match.group(0), "")
        append(content, date_match.group(0), sources)
    return records


_DATASET_SECTIONS: dict[str, tuple[str, ...]] = {
    "personal_report_metadata": ("report_header",),
    "identity_documents": ("report_header",),
    "personal_credit_summary_records": ("credit_summary",),
    "asset_disposition_records": ("asset_disposition",),
    "guarantor_compensation_records": ("guarantor_compensation",),
    "credit_accounts": ("credit_cards", "loans", "other_business"),
    "repayment_liability_records": ("repayment_liability",),
    "postpaid_records": ("non_credit_transactions",),
    "tax_arrears_records": ("tax_arrears",),
    "civil_judgment_records": ("civil_judgments",),
    "enforcement_records": ("enforcements",),
    "administrative_penalty_records": ("administrative_penalties",),
    "public_records": ("tax_arrears", "civil_judgments", "enforcements", "administrative_penalties"),
    "institution_statement_records": ("institution_statements",),
    "inquiry_records": ("institution_inquiries", "personal_inquiries"),
    "report_notes": ("report_notes",),
    "overdue_records": ("credit_cards", "loans", "other_business"),
}

_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "personal_report_metadata": ("report_number", "report_time", "subject_name", "primary_id_type", "primary_id_number"),
    "identity_documents": ("document_type", "document_number"),
    "personal_credit_summary_records": ("metric", "business_category"),
    "asset_disposition_records": ("disposition_date", "asset_management_company"),
    "guarantor_compensation_records": ("compensation_start_date", "guarantor"),
    "credit_accounts": (
        "account_id",
        "sequence",
        "source_sequence",
        "source_section",
        "account_type",
        "business_category",
        "management_institution",
        "business_type",
        "open_date",
    ),
    "repayment_liability_records": ("liability_id", "sequence", "liability_date", "related_party_name", "management_institution"),
    "postpaid_records": ("postpaid_record_id", "institution", "business_type"),
    "tax_arrears_records": ("tax_arrears_id", "tax_authority"),
    "civil_judgment_records": ("civil_judgment_id", "filing_court", "case_number"),
    "enforcement_records": ("enforcement_record_id", "court", "case_number"),
    "administrative_penalty_records": ("administrative_penalty_id", "authority", "document_number"),
    "public_records": ("public_record_id", "record_type", "content"),
    "institution_statement_records": ("institution_statement_id", "statement_content", "added_date"),
    "inquiry_records": ("inquiry_id", "sequence", "inquiry_date", "institution", "source_reason"),
    "report_notes": ("note_id", "sequence", "content"),
}


def _section_text(document: CanonicalPersonalBriefDocumentIR, section_key: str) -> str:
    return "\n".join(
        component.text
        for component in document.components_for(section_key)
        if component.kind != "logical_table" and component.text.strip()
    )


def _canonical_long_blocks(
    document: CanonicalPersonalBriefDocumentIR,
) -> list[tuple[int, str]]:
    """Expose canonical order with no physical-page stop semantics."""
    return [
        (1, component.text)
        for component in sorted(document.components, key=lambda item: item.global_order)
        if component.kind != "logical_table" and component.text.strip()
    ]


def _canonical_long_table_view(document: CanonicalPersonalBriefDocumentIR) -> Any:
    """Adapt each continuous IR logical table onto one virtual page."""
    tables: list[Any] = []
    for component in sorted(document.components, key=lambda item: item.global_order):
        if component.kind != "logical_table" or not component.rows:
            continue
        rows = [row for row in component.rows if row.status != "repeated_header"]
        if component.semantic_role in {"institution_inquiries", "personal_inquiries"}:
            headers = ("编号", "查询日期", "查询机构", "查询原因")
            body = rows
            raw_rows = [list(headers), *[list(row.values) for row in rows]]
        else:
            headers = tuple(rows[0].values)
            body = rows[1:]
            raw_rows = [list(row.values) for row in rows]
        refs = [ref for row in rows for ref in row.source_refs]
        boxes = [ref.bbox for ref in refs if ref.bbox is not None]
        bbox = (
            [
                min(box[0] for box in boxes),
                min(box[1] for box in boxes),
                max(box[2] for box in boxes),
                max(box[3] for box in boxes),
            ]
            if boxes
            else None
        )
        tables.append(
            SimpleNamespace(
                table_id=component.component_id,
                headers=headers,
                rows=tuple(
                    SimpleNamespace(
                        cells=tuple(SimpleNamespace(text=value) for value in row.values)
                    )
                    for row in body
                ),
                page=1,
                bbox=bbox,
                evidence_ids=list(
                    dict.fromkeys(
                        evidence_id
                        for ref in refs
                        for evidence_id in ref.evidence_ids
                    )
                ),
                metadata={
                    "raw_rows": raw_rows,
                    "canonical_component_id": component.component_id,
                },
            )
        )
    return SimpleNamespace(pages=[SimpleNamespace(page_number=1, tables=tables)])


def _explicit_no_records(document: CanonicalPersonalBriefDocumentIR, section_key: str) -> bool:
    text = _compact(_section_text(document, section_key))
    return bool(
        text
        and (
            "没有" in text
            or "暂无" in text
            or re.search(r"(?:无|未发现)(?:相关)?(?:信息|记录|账户)", text)
        )
    )


def _summary_cell_count(document: CanonicalPersonalBriefDocumentIR) -> int:
    grouped_labels = {
        "账户数",
        "未结清/未销户账户数",
        "发生过逾期的账户数",
        "发生过90天以上逾期的账户数",
    }
    count = 0
    for component in document.components_for("credit_summary"):
        if component.kind != "logical_table" or not component.rows:
            continue
        headers = {_compact(value) for value in component.rows[0].values}
        for row in component.rows:
            label = _compact(row.values[0]) if row.values else ""
            if label == "账户数" and {"资产处置信息", "垫款信息"} <= headers:
                count += 2
            elif label in grouped_labels:
                # These four canonical business columns exist even when OCR
                # loses a printed ``--`` and the source cell becomes empty.
                count += 4
            elif label == "相关还款责任账户数":
                count += 2
    return count


def _text_anchor_count(
    document: CanonicalPersonalBriefDocumentIR,
    section_key: str,
    pattern: str,
) -> int:
    return len(re.findall(pattern, _section_text(document, section_key), re.DOTALL))


def _institution_statement_table_count(document: CanonicalPersonalBriefDocumentIR) -> int:
    count = 0
    for component in document.components_for("institution_statements"):
        if component.kind != "logical_table":
            continue
        rows = [list(row.values) for row in component.rows]
        header_index = next(
            (
                index
                for index, row in enumerate(rows)
                if any("说明内容" in _compact(value) for value in row)
                and any("添加日期" in _compact(value) for value in row)
            ),
            None,
        )
        if header_index is None:
            continue
        header = [_compact(value) for value in rows[header_index]]
        content_column = next(index for index, value in enumerate(header) if "说明内容" in value)
        date_column = next(index for index, value in enumerate(header) if "添加日期" in value)
        count += sum(
            component.rows[row_index].status != "repeated_header"
            and max(content_column, date_column) < len(row)
            and bool(_compact(row[content_column]) or _compact(row[date_column]))
            for row_index, row in enumerate(
                rows[header_index + 1 :],
                start=header_index + 1,
            )
        )
    return count


def _inquiry_ordinal_count(document: CanonicalPersonalBriefDocumentIR) -> int:
    total = 0
    for role in ("institution_inquiries", "personal_inquiries"):
        table = document.logical_table(role)
        sequences = [
            int(row.values[0])
            for row in table.rows if row.values and str(row.values[0]).isdigit()
        ] if table is not None else []
        total += max(sequences, default=0)
    return total


def _expected_record_counts(
    document: CanonicalPersonalBriefDocumentIR,
    datasets: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, int], dict[str, str]]:
    account_expectations = _account_boundary_expectations(document)
    expected: dict[str, int] = {
        "personal_report_metadata": 1,
        "identity_documents": 1
        + len(
            re.findall(
                r"[\u3400-\u9fff（）()]+?[A-Za-z0-9][A-Za-z0-9*.-]+",
                _compact(_section_text(document, "report_header")).partition("其他证件信息：")[2],
            )
        ),
        "personal_credit_summary_records": _summary_cell_count(document),
        "asset_disposition_records": max(
            [sequence for sequence, _text, _parts in _numbered_chunks(document, "asset_disposition")],
            default=0,
        ),
        "guarantor_compensation_records": max(
            [sequence for sequence, _text, _parts in _numbered_chunks(document, "guarantor_compensation")],
            default=0,
        ),
        "credit_accounts": sum(account_expectations.values()),
        "repayment_liability_records": _text_anchor_count(
            document,
            "repayment_liability",
            r"承担相关还款责任\s*[，,]\s*责任人类型",
        ),
        "postpaid_records": _text_anchor_count(document, "non_credit_transactions", r"机构名称\s*[:：]"),
        "tax_arrears_records": _text_anchor_count(document, "tax_arrears", r"主管税务机关\s*[:：]"),
        "civil_judgment_records": _text_anchor_count(document, "civil_judgments", r"立案法院\s*[:：]"),
        "enforcement_records": _text_anchor_count(document, "enforcements", r"执行法院\s*[:：]"),
        "administrative_penalty_records": _text_anchor_count(document, "administrative_penalties", r"处罚机构\s*[:：]"),
        "institution_statement_records": max(
            max(
                [
                    sequence
                    for sequence, _text, _parts in _numbered_chunks(
                        document, "institution_statements"
                    )
                ],
                default=0,
            ),
            _text_anchor_count(document, "institution_statements", r"说明内容\s*[:：]"),
            _institution_statement_table_count(document),
        ),
        "inquiry_records": _inquiry_ordinal_count(document),
        "report_notes": max(
            [
                sequence
                for sequence, _text, _parts in _numbered_chunks(document, "report_notes")
            ]
            + [
                int(match.group(1))
                for match in re.finditer(
                    r"(?m)^\s*(\d+)\s*[.、．]\s*",
                    _section_text(document, "report_notes"),
                )
            ],
            default=0,
        ),
    }
    expected["public_records"] = sum(
        expected[name]
        for name in (
            "tax_arrears_records",
            "civil_judgment_records",
            "enforcement_records",
            "administrative_penalty_records",
        )
    )
    category_sections = {
        "credit_card": "credit_cards",
        "housing_loan": "loans",
        "other_loan": "loans",
        "other_business": "other_business",
    }
    source_overdue_values = [
        row.get("value")
        for row in datasets.get("personal_credit_summary_records", [])
        if row.get("metric") == "ever_overdue_account_count"
        and row.get("reporting_status") == "reported"
        and isinstance(row.get("value"), int | float)
        and document.section_presence.get(category_sections.get(row.get("business_category"), ""))
        == "present"
    ]
    expected["overdue_records"] = (
        int(sum(source_overdue_values))
        if source_overdue_values
        else sum(row.get("ever_overdue") is True for row in datasets.get("credit_accounts", []))
    )
    expected["repayment_records"] = 0
    basis = {name: "canonical_source_record_boundaries" for name in expected}
    basis.update(
        {
            "personal_report_metadata": "canonical_report_header",
            "identity_documents": "canonical_identity_boundaries",
            "personal_credit_summary_records": "canonical_summary_business_cells",
            "credit_accounts": "canonical_account_record_boundaries",
            "repayment_liability_records": "canonical_liability_narrative_anchors",
            "inquiry_records": "canonical_printed_inquiry_ordinal_maxima",
            "overdue_records": (
                "canonical_source_summary_overdue_count"
                if source_overdue_values
                else "derived_from_observed_credit_accounts"
            ),
            "public_records": "derived_from_typed_public_record_boundaries",
            "repayment_records": "not_applicable_to_personal_brief",
        }
    )
    return expected, basis


def _record_identifier(dataset_name: str, row: dict[str, Any], index: int) -> str:
    preferred = next(
        (
            value
            for key, value in row.items()
            if key.endswith("_id") and value not in (None, "")
        ),
        None,
    )
    return str(preferred or row.get("record_id") or f"{dataset_name}:row:{index}")


def _valid_record_provenance(
    document: CanonicalPersonalBriefDocumentIR,
    row: dict[str, Any],
) -> bool:
    canonical: dict[str, list[tuple[str, PersonalBriefSourceRef]]] = {}
    for component in document.components:
        refs = [*component.source_refs, *(ref for record in component.rows for ref in record.source_refs)]
        for ref in refs:
            if ref.unit_id:
                canonical.setdefault(ref.unit_id, []).append((component.component_id, ref))
    for source_ref in row.get("source_refs") or []:
        if not isinstance(source_ref, dict):
            continue
        unit_id = str(source_ref.get("unit_id") or "")
        if not unit_id:
            continue
        for component_id, ref in canonical.get(unit_id, []):
            if int(source_ref.get("page") or 0) != ref.source_page:
                continue
            bbox = source_ref.get("bbox")
            bbox_matches = bool(
                bbox
                and ref.bbox is not None
                and len(bbox) >= 4
                and all(abs(float(left) - float(right)) < 0.01 for left, right in zip(bbox[:4], ref.bbox))
            )
            evidence_matches = bool(
                set(str(value) for value in source_ref.get("evidence_ids") or [])
                .intersection(ref.evidence_ids)
            )
            table_matches = bool(source_ref.get("table_id") and source_ref.get("table_id") == ref.table_id)
            if bbox_matches or evidence_matches or table_matches:
                return True
    return False


def _account_source_requirements(
    document: CanonicalPersonalBriefDocumentIR,
    accounts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Observe printed fields before judging decoder or projection completeness."""

    by_source = {
        (row.get("source_section"), int(row.get("source_sequence") or row.get("sequence") or 0)): row
        for row in accounts
    }
    requirements = []
    for section_key in ("credit_cards", "loans", "other_business"):
        for sequence, narrative in enumerate(account_narratives(_account_section_text(document, section_key)), 1):
            account = by_source.get((section_key, sequence), {})
            fields: dict[str, dict[str, str]] = {}
            for semantic_field, dataset_name, public_field in account_source_fields(narrative.text):
                if dataset_name == "overdue_records" and account.get("ever_overdue") is False:
                    continue  # a never-overdue account has no separate overdue view
                fields.setdefault(dataset_name, {})[public_field] = semantic_field
            pages = sorted({
                page
                for component in document.components_for(section_key)
                if _compact(narrative.text)[:40] in _compact(component.text)
                for page in component.source_pages
            })
            pages = sorted(set(pages) | {
                int(ref.get("page") or ref.get("source_page") or 0)
                for ref in account.get("source_refs") or []
                if ref.get("page") or ref.get("source_page")
            })
            requirements.append({
                "source_section": section_key,
                "source_sequence": sequence,
                "source_pages": pages,
                "fields": fields,
                "missing_fields": sorted({
                    public_field
                    for field_map in fields.values()
                    for public_field, semantic_field in field_map.items()
                    if account.get(semantic_field) in (None, "")
                }),
            })
    return requirements


def _dataset_completeness(
    document: CanonicalPersonalBriefDocumentIR,
    datasets: dict[str, list[dict[str, Any]]],
    credit_summary: dict[str, Any],
    *,
    account_requirements: list[dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    del credit_summary  # summary totals and displayed detail have intentionally different scopes
    if account_requirements is None:
        account_requirements = _account_source_requirements(document, datasets.get("credit_accounts", []))
    expected, basis = _expected_record_counts(document, datasets)
    output: dict[str, dict[str, Any]] = {}
    for dataset_name, rows in datasets.items():
        expected_count = expected.get(dataset_name, 0)
        sections = _DATASET_SECTIONS.get(dataset_name, ())
        present_sections = [
            section
            for section in sections
            if document.section_presence.get(section) == "present"
        ]
        no_records = bool(present_sections) and all(
            _explicit_no_records(document, section) for section in present_sections
        )
        source_dataset = dataset_name not in {"overdue_records", "public_records", "repayment_records"}
        present_but_unobserved = bool(
            source_dataset
            and present_sections
            and expected_count == 0
            and not no_records
            and dataset_name not in {"personal_report_metadata", "identity_documents"}
        )
        unresolved = sum(
            str(row.get("extraction_status") or "") == "unresolved"
            or any(
                str(value) == "unresolved"
                for key, value in row.items()
                if key.endswith("_status")
            )
            for row in rows
        )
        missing_required_ids: list[str] = []
        invalid_provenance_ids: list[str] = []
        uncovered_boundary_ids: list[str] = []
        duplicate_boundary_ids: list[str] = []
        missing_source_fields: list[dict[str, Any]] = []
        required = _REQUIRED_FIELDS.get(dataset_name, ())
        for index, row in enumerate(rows, start=1):
            row_id = _record_identifier(dataset_name, row, index)
            if any(row.get(field) in (None, "") for field in required):
                missing_required_ids.append(row_id)
            if not _valid_record_provenance(document, row):
                invalid_provenance_ids.append(row_id)
        if dataset_name == "credit_accounts":
            missing_source_fields = [
                {
                    "source_section": requirement["source_section"],
                    "source_sequence": requirement["source_sequence"],
                    "fields": list(requirement["missing_fields"]),
                    "source_pages": list(requirement["source_pages"]),
                }
                for requirement in account_requirements
                if requirement["missing_fields"]
            ]
            account_expectations = _account_boundary_expectations(document)
            for section_key, section_expected in account_expectations.items():
                section_rows = [
                    row
                    for row in rows
                    if row.get("source_section") == section_key
                    or (
                        not row.get("source_section")
                        and section_key == "credit_cards"
                        and row.get("account_type") == "credit_card"
                    )
                    or (
                        not row.get("source_section")
                        and section_key == "other_business"
                        and row.get("account_type") == "other_business"
                    )
                    or (
                        not row.get("source_section")
                        and section_key == "loans"
                        and row.get("account_type") not in {"credit_card", "other_business"}
                    )
                ]
                counts: dict[int, int] = {}
                for row in section_rows:
                    sequence = int(row.get("source_sequence") or row.get("sequence") or 0)
                    if sequence > 0:
                        counts[sequence] = counts.get(sequence, 0) + 1
                uncovered_boundary_ids.extend(
                    f"{section_key}:{sequence}"
                    for sequence in range(1, section_expected + 1)
                    if counts.get(sequence, 0) == 0
                )
                duplicate_boundary_ids.extend(
                    f"{section_key}:{sequence}"
                    for sequence, count in sorted(counts.items())
                    if count > 1
                )
        verified = bool(
            document.content_conserved
            and expected_count == len(rows)
            and unresolved == 0
            and not missing_required_ids
            and not invalid_provenance_ids
            and not uncovered_boundary_ids
            and not duplicate_boundary_ids
            and not missing_source_fields
            and not present_but_unobserved
        )
        if verified and not present_sections and not rows:
            status = "absent_from_report"
        elif verified and no_records and not rows:
            status = "no_records"
        elif verified:
            status = "complete"
        else:
            status = "incomplete"
        output[dataset_name] = {
            "expected_row_count": expected_count,
            "emitted_row_count": len(rows),
            "omitted_row_count": max(0, expected_count - len(rows)),
            "unexpected_row_count": max(0, len(rows) - expected_count),
            "unresolved_row_count": unresolved,
            "missing_required_field_record_count": len(missing_required_ids),
            "missing_required_field_record_ids": missing_required_ids,
            "invalid_provenance_record_count": len(invalid_provenance_ids),
            "invalid_provenance_record_ids": invalid_provenance_ids,
            "uncovered_boundary_ids": uncovered_boundary_ids,
            "duplicate_boundary_ids": duplicate_boundary_ids,
            "missing_source_fields": missing_source_fields,
            "present_but_unobserved": present_but_unobserved,
            "verified": verified,
            "basis": basis.get(dataset_name, "canonical_source_record_boundaries"),
            "status": status,
        }
    return output


def _extraction_report(
    document: CanonicalPersonalBriefDocumentIR,
    datasets: dict[str, list[dict[str, Any]]],
    completeness: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    if not document.content_conserved:
        failures.append(
            {
                "code": "PERSONAL_BRIEF_SOURCE_NOT_CONSERVED",
                "severity": "error",
                "message": "Canonical reconstruction did not assign every source unit exactly once.",
                "source_unit_ids": list(document.unassigned_source_unit_ids),
            }
        )
    for dataset_name, details in completeness.items():
        if details.get("verified"):
            continue
        failures.append(
            {
                "code": f"PERSONAL_BRIEF_{dataset_name.upper()}_INCOMPLETE",
                "severity": (
                    "error"
                    if any(
                        details.get(key)
                        for key in (
                            "omitted_row_count",
                            "unexpected_row_count",
                            "missing_required_field_record_count",
                            "invalid_provenance_record_count",
                            "present_but_unobserved",
                            "uncovered_boundary_ids",
                            "duplicate_boundary_ids",
                            "missing_source_fields",
                        )
                    )
                    else "warning"
                ),
                "message": (
                    f"{dataset_name}: expected {details.get('expected_row_count')}, "
                    f"emitted {details.get('emitted_row_count')}, "
                    f"unresolved {details.get('unresolved_row_count')}, "
                    f"invalid provenance {details.get('invalid_provenance_record_count')}."
                ),
                "dataset": dataset_name,
                "details": details,
            }
        )
    return {
        "protocol": "pboc-personal-brief-extraction",
        "schema_version": "2.0.0",
        "status": "complete" if not failures else "incomplete",
        "failures": failures,
        "source_unit_count": document.source_unit_count,
        "content_conserved": document.content_conserved,
        "canonical_section_presence": dict(document.section_presence),
        "dataset_completeness": completeness,
        "business_record_count": sum(len(rows) for rows in datasets.values()),
    }


def _is_personal_brief_document(
    document: CanonicalPersonalBriefDocumentIR,
    report_metadata: list[dict[str, Any]],
) -> tuple[bool, dict[str, Any]]:
    compact = _compact(document.full_text)
    metadata = report_metadata[0] if report_metadata else {}
    identity_signals = sum(
        metadata.get(field) not in (None, "")
        for field in ("report_number", "report_time", "subject_name", "primary_id_number")
    )
    title_present = "个人信用报告" in compact
    recognized = bool(document.source_page_count > 0 and title_present and identity_signals >= 2)
    return recognized, {
        "title_present": title_present,
        "identity_signal_count": identity_signals,
        "source_page_count": document.source_page_count,
    }


def _rejected_semantic_document(
    document: CanonicalPersonalBriefDocumentIR,
    *,
    content_mode: str,
    recognition: dict[str, Any],
) -> PersonalBriefSemanticDocument:
    datasets: dict[str, list[dict[str, Any]]] = {
        name: []
        for name in (
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
            "institution_statement_records",
            "inquiry_records",
        )
    }
    failure = {
        "code": "PERSONAL_BRIEF_DOCUMENT_NOT_RECOGNIZED",
        "severity": "error",
        "message": "Input does not satisfy the canonical PBOC personal-brief report header contract.",
        "details": recognition,
    }
    extraction = {
        "protocol": "pboc-personal-brief-extraction",
        "schema_version": "2.0.0",
        "status": "incomplete",
        "failures": [failure],
        "source_unit_count": document.source_unit_count,
        "content_conserved": document.content_conserved,
        "canonical_section_presence": dict(document.section_presence),
        "dataset_completeness": {},
        "business_record_count": 0,
    }
    return PersonalBriefSemanticDocument(
        facts={
            "report_subtype": "personal_brief",
            "content_mode": content_mode,
            "canonical_section_presence": dict(document.section_presence),
            "canonical_ir_schema_version": document.schema_version,
        },
        datasets=datasets,
        credit_summary={},
        credit_extraction_audit={
            "status": "review",
            "issues": ["canonical_extraction:PERSONAL_BRIEF_DOCUMENT_NOT_RECOGNIZED"],
            "source_page_complete": False,
            "source_page_count": document.source_page_count,
        },
        extraction_report=extraction,
        dataset_completeness={},
    )


def _build_personal_brief_compatibility_audit(
    document: CanonicalPersonalBriefDocumentIR,
    *,
    content_mode: str,
    datasets: dict[str, list[dict[str, Any]]],
    credit_summary: dict[str, Any],
    extraction_report: dict[str, Any],
) -> dict[str, Any]:
    """Build the pre-projector diagnostic audit for explicit debug consumers."""
    from docmirror.plugins.credit_report.business_assembly import _build_audit
    from docmirror.plugins.credit_report.business_records import _personal_brief_credit_lines

    accounts = datasets.get("credit_accounts", [])
    audit = _build_audit(
        parse_result=document,
        full_text=document.full_text,
        report_subtype="personal_brief",
        content_mode=content_mode,
        collections={
            "credit_accounts": accounts,
            "credit_lines": _personal_brief_credit_lines(accounts),
            "repayment_liability_records": datasets.get("repayment_liability_records", []),
            "repayment_records": datasets.get("repayment_records", []),
            "overdue_records": datasets.get("overdue_records", []),
            "inquiry_records": datasets.get("inquiry_records", []),
            "public_records": datasets.get("public_records", []),
        },
        conflicts=[],
        credit_summary=credit_summary,
    )
    failures = extraction_report.get("failures") or []
    if failures:
        audit["issues"] = list(
            dict.fromkeys(
                [
                    *list(audit.get("issues") or []),
                    *[
                        f"canonical_extraction:{failure['code']}"
                        for failure in failures
                    ],
                ]
            )
        )
        audit["status"] = "review"
    return audit


def extract_personal_brief_semantic_document(
    document: CanonicalPersonalBriefDocumentIR,
    *,
    content_mode: str = "native_text",
) -> PersonalBriefSemanticDocument:
    """Apply the rigid personal-brief schema exactly once to the canonical IR."""
    if not isinstance(document, CanonicalPersonalBriefDocumentIR):
        raise TypeError("personal brief extraction requires CanonicalPersonalBriefDocumentIR")
    from docmirror.plugins.credit_report.business_records import (
        _overdue_from_personal_brief_accounts,
        _page_texts,
        _personal_brief_summary_from_canonical_tables,
    )
    from docmirror.plugins.credit_report.personal_brief_native.extraction import (
        _asset_and_compensation_records,
        _personal_header_datasets,
        _personal_public_records,
        _personal_summary_records,
        _postpaid_records,
        _summary_source_text,
    )

    blocks = _canonical_long_blocks(document)
    table_view = _canonical_long_table_view(document)
    identity_documents, report_metadata, amount_policy = _personal_header_datasets(document, blocks)
    recognized, recognition = _is_personal_brief_document(document, report_metadata)
    if not recognized:
        return _rejected_semantic_document(
            document,
            content_mode=content_mode,
            recognition=recognition,
        )

    text = re.sub(r"\s+", " ", document.full_text).strip()
    page_texts = _page_texts(document)
    accounts: list[dict[str, Any]] = []
    for section_key in ("credit_cards", "loans", "other_business"):
        accounts.extend(_extract_account_section(document, section_key, page_texts))
    liabilities = _repayment_liabilities_from_canonical_records(document)
    inquiries = _inquiry_records(document)
    inquiry_scope = _inquiry_scope(document)
    overdue = _overdue_from_personal_brief_accounts(accounts)
    summary_text = _summary_source_text(document, text, document.full_text)
    source_summary = _personal_brief_summary_from_canonical_tables(
        table_view,
        summary_text,
        expected_account_count=None,
    )
    credit_summary = {
        "source": "canonical_personal_brief_document_ir",
        "account_count": len(accounts),
        "active_account_count": sum(account.get("account_status") == "active" for account in accounts),
        "active_account_count_basis": "legacy_compatibility_status_active",
        "unclosed_account_count": sum(
            account.get("account_lifecycle_state") == "open" for account in accounts
        ),
        "activated_credit_card_account_count": sum(
            account.get("card_activation_state") == "activated" for account in accounts
        ),
        "inactive_credit_card_account_count": sum(
            account.get("card_activation_state") == "not_activated" for account in accounts
        ),
        "settled_account_count": sum(
            account.get("termination_event_type") == "debt_settled" for account in accounts
        ),
        "closed_credit_card_account_count": sum(
            account.get("termination_event_type") == "account_closed" for account in accounts
        ),
        "transferred_out_account_count": sum(
            account.get("termination_event_type") == "transferred_out" for account in accounts
        ),
        "derived_ever_overdue_account_count": len(overdue),
        "repayment_liability_count": len(liabilities),
        "inquiry_count": len(inquiries),
        "institution_inquiry_count": sum(item.get("inquiry_type") == "institution" for item in inquiries),
        "personal_inquiry_count": sum(item.get("inquiry_type") == "personal" for item in inquiries),
        "projected_account_count": len(accounts),
        **source_summary,
    }

    asset_dispositions, guarantor_compensations = _asset_and_compensation_records(document, text)
    public_datasets = _personal_public_records(table_view, blocks)
    summary_records = _personal_summary_records(
        table_view,
        summary_text,
        expected_account_count=None,
        parsed_summary=source_summary,
    )
    summary_components = [
        component
        for component in document.components_for("credit_summary")
        if component.kind == "logical_table"
    ]
    if summary_components and summary_records:
        summary_refs = _component_refs(
            summary_components,
            "canonical_personal_brief_summary_table",
        )
        summary_evidence = list(
            dict.fromkeys(
                evidence_id
                for ref in summary_refs
                for evidence_id in ref.get("evidence_ids") or []
            )
        )
        for record in summary_records:
            record["source_refs"] = [dict(ref) for ref in summary_refs]
            record["evidence_ids"] = list(summary_evidence)
    non_credit_page, non_credit_statement = _statement_after(blocks, "非信贷交易记录")
    public_page, public_statement = _statement_after(blocks, "公共记录")
    non_credit_lookback_years = extract_personal_brief_lookback_years(
        non_credit_statement
    )
    public_lookback_years = extract_personal_brief_lookback_years(public_statement)
    notes = _report_notes(blocks)
    institution_statements = _institution_statement_records(document)

    datasets: dict[str, list[dict[str, Any]]] = {
        "personal_report_metadata": report_metadata,
        "report_notes": notes,
        "identity_documents": identity_documents,
        "personal_credit_summary_records": summary_records,
        "asset_disposition_records": asset_dispositions,
        "guarantor_compensation_records": guarantor_compensations,
        "credit_accounts": accounts,
        "repayment_liability_records": liabilities,
        "repayment_records": [],
        "overdue_records": overdue,
        "postpaid_records": _postpaid_records(blocks),
        **public_datasets,
        "institution_statement_records": institution_statements,
        "inquiry_records": inquiries,
    }
    canonicalize_personal_brief_reporting_units(
        datasets,
        amount_policy=amount_policy,
    )
    for dataset_name, rows in datasets.items():
        sections = _DATASET_SECTIONS.get(dataset_name)
        if sections and rows:
            _attach_canonical_provenance(
                document,
                rows,
                sections,
                f"canonical_personal_brief_{dataset_name}",
            )
    metadata = report_metadata[0] if report_metadata else {}
    facts = {
        "report_subtype": "personal_brief",
        "content_mode": content_mode,
        "document_label": "个人信用报告",
        "subject_name": metadata.get("subject_name"),
        "id_type": metadata.get("primary_id_type"),
        "id_number": metadata.get("primary_id_number"),
        "subject_id": metadata.get("primary_id_number"),
        "report_number": metadata.get("report_number"),
        "report_time": metadata.get("report_time"),
        "marital_status": metadata.get("marital_status"),
        "marital_status_raw": metadata.get("marital_status_raw"),
        "reporting_context": amount_policy,
        "non_credit_transaction_summary": {
            "record_status": (
                "absent_from_report"
                if document.section_presence.get("non_credit_transactions") != "present"
                else "no_records"
                if _explicit_no_records(document, "non_credit_transactions")
                else "reported"
                if datasets.get("postpaid_records") or non_credit_statement
                else "unresolved"
            ),
            "lookback_years": non_credit_lookback_years,
            "source_statement": non_credit_statement,
            "source_page": non_credit_page or None,
        },
        "public_record_summary": {
            "record_status": (
                "absent_from_report"
                if document.section_presence.get("public_records") != "present"
                else "no_records"
                if _explicit_no_records(document, "public_records")
                else "reported"
                if public_datasets.get("public_records") or public_statement
                else "unresolved"
            ),
            "lookback_years": public_lookback_years,
            "source_statement": public_statement,
            "source_page": public_page or None,
        },
        "canonical_section_presence": dict(document.section_presence),
        "canonical_ir_schema_version": document.schema_version,
    }
    facts = {key: value for key, value in facts.items() if value not in (None, "")}
    if inquiry_scope:
        facts["inquiry_record_summary"] = inquiry_scope
    account_requirements = _account_source_requirements(document, accounts)
    completeness = _dataset_completeness(
        document, datasets, credit_summary, account_requirements=account_requirements
    )
    extraction = _extraction_report(document, datasets, completeness)
    section_requirements = []
    for section_type, years, page in (
        ("non_credit_transactions", non_credit_lookback_years, non_credit_page),
        ("public_records", public_lookback_years, public_page),
    ):
        if years is not None:
            section_requirements.append(
                {
                    "section_type": section_type,
                    "fields": {"lookback_years": years},
                    "source_pages": [page] if page else [],
                }
            )
    if inquiry_scope:
        section_requirements.append(
            {
                "section_type": "inquiries",
                "fields": {"lookback_years": inquiry_scope["lookback_years"]},
                "source_pages": inquiry_scope["source_pages"],
            }
        )
    extraction["source_field_coverage"] = {
        "accounts": account_requirements,
        "sections": section_requirements,
    }
    return PersonalBriefSemanticDocument(
        facts=facts,
        datasets=datasets,
        credit_summary=credit_summary,
        credit_extraction_audit=None,
        extraction_report=extraction,
        dataset_completeness=completeness,
        credit_extraction_audit_factory=partial(
            _build_personal_brief_compatibility_audit,
            document,
            content_mode=content_mode,
            datasets=datasets,
            credit_summary=credit_summary,
            extraction_report=extraction,
        ),
    )


def run_personal_brief_pipeline(
    parse_result: Any,
    *,
    content_mode: str = "native_text",
) -> PersonalBriefPipelineArtifacts:
    document = build_canonical_personal_brief_document(parse_result)
    semantic = extract_personal_brief_semantic_document(
        document,
        content_mode=content_mode,
    )
    return PersonalBriefPipelineArtifacts(
        document_ir=document,
        semantic_document=semantic,
    )


__all__ = [
    "PersonalBriefPipelineArtifacts",
    "PersonalBriefSemanticDocument",
    "extract_personal_brief_semantic_document",
    "run_personal_brief_pipeline",
]
