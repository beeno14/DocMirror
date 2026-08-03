# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Post-seal OCR correction overlay for personal detailed credit reports.

The overlay never mutates ``ParseResult``.  It produces corrected copies of
OCR evidence and extraction candidates, keyed by stable semantic roles rather
than Community JSON paths.  Domain extractors can therefore consume corrected
evidence while the sealed source remains auditable and schema redesigns remain
independent from OCR policy.
"""

from __future__ import annotations

import hashlib
import os
import re
import unicodedata
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
from functools import lru_cache
from importlib.resources import files
from typing import Any, Iterable, Mapping

import yaml

_DATE_TOKEN_RE = re.compile(r"(?<!\d)(20\d{2})[./-](\d{1,2})[./-](\d{1,2})(?!\d)")
_DATE_LOOSE_RE = re.compile(r"(?<!\d)(20\d{2})[.,/-]?(\d{2})[.,/-](\d{2})(?!\d)")
_DATETIME_DIGITS_RE = re.compile(r"(?<!\d)(20\d{2})\D*(\d{2})\D*(\d{2})\D*(\d{2})\D*(\d{2})\D*(\d{2})(?!\d)")
_CN_ID_RE = re.compile(r"^\d{17}[0-9X]$")
_MOBILE_RE = re.compile(r"^1[3-9]\d{9}$")
_ACCOUNT_IDENTIFIER_RE = re.compile(r"^[A-Z0-9]{8,64}$")
_AMOUNT_RE = re.compile(r"^-?(?:0|[1-9]\d{0,14})(?:\.\d{1,2})?$")
_INQUIRY_DATE_LIKE_RE = re.compile(r"20\d{2}(?:[.,/:;-]?\d{1,2}[.,/:;-]\d{1,2}|[.]\d{4})")
_VALID_INQUIRY_REASONS = (
    "本人查询",
    "贷后管理",
    "贷款审批",
    "信用卡审批",
    "担保资格审查",
    "融资审批",
    "保前审查",
    "客户准入资格审查",
    "资信审查",
    "法人代表、负责人、高管",
)
_INQUIRY_REASON_MARKERS = _VALID_INQUIRY_REASONS + (
    "费后管理",
    "货后管理",
    "货款审批",
    "资款审批",
    "信用卡审抵",
    "担保资格申查",
    "贷后智理",
)
_INSTITUTION_SUFFIX_RE = re.compile(
    r"[\u3400-\u9fff（）()·]{2,80}?(?:"
    r"银行卡业务部[（(]牡丹卡中心[）)]|信用卡中心|个人信贷部|"
    r"(?:福州|厦门|福建省|温泉|鼓楼|晋安|华林)[\u3400-\u9fff]{0,6}支行|"
    r"(?:福州|厦门市|福建省)[\u3400-\u9fff]{0,4}分行|"
    r"农村信用合作联社|股份有限公司|有限责任公司|有限公司|管理中心"
    r")"
)
_LEADING_ROW_NOISE_RE = re.compile(r"^[\s\W_]*(?:[A-Za-z\u3400-\u9fff]{1,2}\s+)?(?=\d{0,3}\s*20\d{2})")
_TRAILING_INSTITUTION_NOISE_RE = re.compile(r"(?:\s+[A-Za-z0-9￥¥?$]{1,3})+$")

_FIELD_ROLES: dict[str, str] = {
    "report_time": "report_datetime",
    "query_time": "report_datetime",
    "primary_id_number": "identity_document_number",
    "id_number": "identity_document_number",
    "document_number": "identity_document_number",
    "mobile_phone": "mobile_phone",
    "phone": "phone",
    "work_phone": "phone",
    "residence_phone": "phone",
    "residential_phone": "phone",
    "employer_phone": "phone",
    "account_identifier": "account_identifier",
    "credit_agreement_identifier": "account_identifier",
    "institution": "institution_name",
    "management_institution": "institution_name",
    "query_institution": "institution_name",
    "data_provider": "institution_name",
    "数据发生机构名称": "institution_name",
    "信息更新日期": "date",
    "open_date": "date",
    "due_date": "date",
    "snapshot_date": "date",
    "close_date": "date",
    "inquiry_date": "date",
    "birth_date": "date",
    "event_date": "date",
    "repayment_date": "date",
    "balance": "amount",
    "loan_amount": "amount",
    "credit_limit": "amount",
    "used_amount": "amount",
    "overdue_amount": "amount",
    "status": "repayment_status",
    "repayment_status": "repayment_status",
}
_RAW_OR_PROVENANCE_KEYS = frozenset(
    {
        "raw",
        "canonical_raw",
        "raw_value",
        "raw_values",
        "raw_text",
        "raw_detail_text",
        "raw_detail_lines",
        "source_refs",
        "source_cell_refs",
        "audit",
        "ocr_correction",
        "ocr_corrections",
    }
)


@dataclass(frozen=True)
class PersonalDetailCorrectionDecision:
    correction_id: str
    role: str
    original: str
    corrected: str
    action: str
    method: str
    reason_codes: tuple[str, ...]
    confidence: float
    source_refs: tuple[dict[str, Any], ...] = ()
    candidates: tuple[str, ...] = ()
    pack_id: str = "pboc.personal_detail.zh-CN"
    pack_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@lru_cache(maxsize=1)
def _pack() -> dict[str, Any]:
    payload = yaml.safe_load(
        files("docmirror.plugins.credit_report.personal_detail_scanned")
        .joinpath("ocr_corrections.yaml")
        .read_text(encoding="utf-8")
    ) or {}
    if not isinstance(payload, dict) or int(payload.get("version") or 0) < 1:
        raise ValueError("invalid personal-detail OCR correction pack")
    return payload


def _plain_text(value: Any) -> str:
    return unicodedata.normalize("NFKC", str(value or "")).replace("\u200b", "").strip()


def _valid_date(value: str) -> bool:
    try:
        datetime.strptime(value, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def _normalize_date(value: str) -> str:
    text = _plain_text(value).replace(",", ".")
    match = _DATE_TOKEN_RE.search(text) or _DATE_LOOSE_RE.search(text)
    if not match:
        return text
    candidate = f"{int(match.group(1)):04d}-{int(match.group(2)):02d}-{int(match.group(3)):02d}"
    return candidate if _valid_date(candidate) else text


def _normalize_datetime(value: str) -> str:
    text = _plain_text(value).replace(",", ".")
    match = _DATETIME_DIGITS_RE.search(text)
    if not match:
        digits = re.sub(r"\D", "", text)
        if len(digits) == 14 and digits.startswith("20"):
            parts = (digits[:4], digits[4:6], digits[6:8], digits[8:10], digits[10:12], digits[12:14])
        else:
            return text
    else:
        parts = match.groups()
    candidate = f"{parts[0]}-{parts[1]}-{parts[2]}T{parts[3]}:{parts[4]}:{parts[5]}+08:00"
    try:
        datetime.fromisoformat(candidate)
    except ValueError:
        return text
    return candidate


def _cn_id_checksum_valid(value: str) -> bool:
    if not _CN_ID_RE.fullmatch(value):
        return False
    weights = (7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)
    checks = "10X98765432"
    return checks[sum(int(ch) * weight for ch, weight in zip(value[:17], weights, strict=True)) % 11] == value[-1]


def _normalize_identity(value: str) -> str:
    text = re.sub(r"[\s-]+", "", _plain_text(value)).upper()
    if _cn_id_checksum_valid(text):
        return text
    substitutions = str.maketrans({"O": "0", "Q": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8"})
    candidate = text.translate(substitutions)
    return candidate if _cn_id_checksum_valid(candidate) else text


def _normalize_phone(value: str, *, mobile: bool) -> str:
    text = _plain_text(value)
    digits = re.sub(r"\D", "", text)
    if mobile and _MOBILE_RE.fullmatch(digits):
        return digits
    if not mobile and 5 <= len(digits) <= 16 and not re.search(r"[A-Za-z\u3400-\u9fff]", text):
        return digits
    return text


def _normalize_identifier(value: str) -> str:
    text = re.sub(r"[\s:：,，]+", "", _plain_text(value)).upper()
    return text if _ACCOUNT_IDENTIFIER_RE.fullmatch(text) else _plain_text(value)


def _normalize_amount(value: str) -> str:
    text = _plain_text(value).replace("，", ",").replace(",", "")
    if not _AMOUNT_RE.fullmatch(text):
        substitutions = str.maketrans({"O": "0", "Q": "0", "I": "1", "L": "1", "S": "5", "B": "8"})
        candidate = text.translate(substitutions)
        if not _AMOUNT_RE.fullmatch(candidate):
            return _plain_text(value)
        text = candidate
    try:
        Decimal(text)
    except InvalidOperation:
        return _plain_text(value)
    return text


def normalize_institution_name(value: str) -> str:
    """Return a conservative institution-name correction."""
    text = re.sub(r"\s+", " ", _plain_text(value)).strip(" -_:：,，;；")
    for original, corrected in dict(_pack().get("institution_substitutions") or {}).items():
        text = text.replace(str(original), str(corrected))
    text = re.sub(r"^[中福装R$证芬心多离囍版真苏德会食守]\s+(?=.{2,})", "", text)
    text = _TRAILING_INSTITUTION_NOISE_RE.sub("", text).strip()
    matches = list(_INSTITUTION_SUFFIX_RE.finditer(text))
    if matches:
        # OCR debris tends to occur before or after the legal-name span. Prefer
        # the longest suffix-terminated Chinese span and preserve branch names.
        selected_match = max(matches, key=lambda match: len(match.group(0)))
        selected = selected_match.group(0)
        trailing = text[selected_match.end() :]
        specialized_tail = re.match(
            r"(?:信用卡中心|个人信贷部|银行卡业务部[（(]牡丹卡中心[）)]|"
            r"(?:福州|厦门|福建省|温泉|鼓楼|晋安|华林)[\u3400-\u9fff]{0,6}支行|"
            r"(?:福州|厦门市|福建省)[\u3400-\u9fff]{0,4}分行)",
            trailing,
        )
        if specialized_tail:
            selected += specialized_tail.group(0)
        selected = re.sub(r"^[证芬心多离囍版真苏德会食守](?=[\u3400-\u9fff]{4,})", "", selected)
        text = selected
    return text


def _normalize_inquiry_line(value: str) -> str:
    text = re.sub(r"\s+", " ", _plain_text(value)).strip()
    text = re.sub(r"(?<=\d)[,:;](?=\d)", ".", text)
    text = re.sub(r"(?<!\d)(20\d{2})[.](\d{2})(\d{2})(?!\d)", r"\1.\2.\3", text)
    text = re.sub(r"(?<!\d)(20\d{2})(\d{2})[.](\d{2})(?!\d)", r"\1.\2.\3", text)
    for original, corrected in dict(_pack().get("inquiry_reason_substitutions") or {}).items():
        text = text.replace(str(original), str(corrected))
    return text


def _normalize_account_line(value: str) -> str:
    text = _plain_text(value)
    for original, corrected in dict(_pack().get("account_label_substitutions") or {}).items():
        text = text.replace(str(original), str(corrected))
    text = re.sub(r"^[^\u3400-\u9fff0-9]*[账联][户广尸]\s*(?=\d{1,3}\b)", "账户 ", text)
    return text


def _is_valid_for_role(value: str, role: str) -> bool:
    if role == "date":
        return _valid_date(value)
    if role == "report_datetime":
        try:
            datetime.fromisoformat(value)
            return True
        except ValueError:
            return False
    if role == "identity_document_number":
        return _cn_id_checksum_valid(value)
    if role == "mobile_phone":
        return bool(_MOBILE_RE.fullmatch(value))
    if role == "phone":
        return 5 <= len(re.sub(r"\D", "", value)) <= 16
    if role == "account_identifier":
        return bool(_ACCOUNT_IDENTIFIER_RE.fullmatch(value))
    if role == "amount":
        return bool(_AMOUNT_RE.fullmatch(value))
    if role == "institution_name":
        return bool(_INSTITUTION_SUFFIX_RE.fullmatch(value) or value == "本人")
    if role == "repayment_status":
        return value in {"N", "C", "M", "B", "D", "G", "#", "*", "1", "2", "3", "4", "5", "6", "7"}
    if role == "inquiry_row":
        return bool(_DATE_TOKEN_RE.search(value) and any(marker in value for marker in _VALID_INQUIRY_REASONS))
    return bool(value)


def _normalize_role(value: str, role: str) -> str:
    if role == "date":
        return _normalize_date(value)
    if role == "report_datetime":
        return _normalize_datetime(value)
    if role == "identity_document_number":
        return _normalize_identity(value)
    if role == "mobile_phone":
        return _normalize_phone(value, mobile=True)
    if role == "phone":
        return _normalize_phone(value, mobile=False)
    if role == "account_identifier":
        return _normalize_identifier(value)
    if role == "amount":
        return _normalize_amount(value)
    if role == "institution_name":
        return normalize_institution_name(value)
    if role == "repayment_status":
        return _plain_text(value).upper()
    if role == "inquiry_row":
        return _normalize_inquiry_line(value)
    if role == "account_line":
        return _normalize_account_line(value)
    return _plain_text(value)


def _source_refs(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes, Mapping)):
        return ()
    return tuple(dict(item) for item in value if isinstance(item, dict))


class PersonalDetailOCRCorrectionOverlay:
    """Schema-independent corrected view over one sealed personal report."""

    def __init__(
        self,
        parse_result: Any,
        *,
        page_image_resolver: Any | None = None,
        repair_engine: Any | None = None,
        enable_targeted_ocr: bool | None = None,
        max_targeted_requests: int = 8,
    ) -> None:
        self.parse_result = parse_result
        self._page_image_resolver = page_image_resolver
        self._repair_engine = repair_engine
        self.enable_targeted_ocr = (
            os.environ.get("DOCMIRROR_PERSONAL_DETAIL_TARGETED_OCR", "1") != "0"
            if enable_targeted_ocr is None
            else bool(enable_targeted_ocr)
        )
        self.max_targeted_requests = max(0, int(max_targeted_requests))
        self._targeted_requests = 0
        self._decisions: list[PersonalDetailCorrectionDecision] = []
        self._decision_keys: set[tuple[str, str, str, str]] = set()

    @property
    def decisions(self) -> tuple[PersonalDetailCorrectionDecision, ...]:
        return tuple(self._decisions)

    def audit(self) -> dict[str, Any]:
        counts = Counter(decision.action for decision in self._decisions)
        return {
            "pack_id": str(_pack().get("pack_id") or "pboc.personal_detail.zh-CN"),
            "pack_version": int(_pack().get("version") or 1),
            "decision_count": len(self._decisions),
            "applied_count": counts.get("applied", 0),
            "suggested_count": counts.get("suggested", 0),
            "targeted_ocr_requests": self._targeted_requests,
            "decisions": [decision.to_dict() for decision in self._decisions],
        }

    def _record(
        self,
        *,
        role: str,
        original: str,
        corrected: str,
        method: str,
        reason_codes: tuple[str, ...],
        confidence: float,
        refs: tuple[dict[str, Any], ...],
        candidates: tuple[str, ...] = (),
        action: str = "applied",
    ) -> PersonalDetailCorrectionDecision:
        marker = (role, original, corrected, repr(refs))
        digest = hashlib.sha256("\x1f".join(marker).encode("utf-8")).hexdigest()[:16]
        decision = PersonalDetailCorrectionDecision(
            correction_id=f"personal_detail_ocr:{digest}",
            role=role,
            original=original,
            corrected=corrected,
            action=action,
            method=method,
            reason_codes=reason_codes,
            confidence=max(0.0, min(1.0, float(confidence))),
            source_refs=refs,
            candidates=candidates,
            pack_id=str(_pack().get("pack_id") or "pboc.personal_detail.zh-CN"),
            pack_version=int(_pack().get("version") or 1),
        )
        if marker not in self._decision_keys:
            self._decision_keys.add(marker)
            self._decisions.append(decision)
        return decision

    def correct_text(
        self,
        value: Any,
        *,
        role: str,
        source_refs: Iterable[dict[str, Any]] = (),
        confidence: float | None = None,
        allow_targeted_ocr: bool = False,
    ) -> tuple[str, PersonalDetailCorrectionDecision | None]:
        original = str(value or "")
        refs = _source_refs(source_refs)
        corrected = _normalize_role(original, role)
        if corrected != original and _is_valid_for_role(corrected, role):
            return corrected, self._record(
                role=role,
                original=original,
                corrected=corrected,
                method="typed_deterministic",
                reason_codes=("role_scoped_normalization", "typed_validation"),
                confidence=max(0.98, float(confidence or 0.0)),
                refs=refs,
            )
        if allow_targeted_ocr and not _is_valid_for_role(corrected, role):
            repaired = self._targeted_repair(original, role=role, refs=refs)
            if repaired is not None:
                return repaired
        return original, None

    def _targeted_repair(
        self,
        original: str,
        *,
        role: str,
        refs: tuple[dict[str, Any], ...],
    ) -> tuple[str, PersonalDetailCorrectionDecision] | None:
        if not self.enable_targeted_ocr or self._targeted_requests >= self.max_targeted_requests:
            return None
        ref = next(
            (
                item
                for item in refs
                if isinstance(item.get("bbox"), (list, tuple))
                and len(item["bbox"]) == 4
                and int(item.get("logical_page") or item.get("page") or 0) > 0
            ),
            None,
        )
        if ref is None:
            return None
        if self._page_image_resolver is None:
            from docmirror.plugins.credit_report.page_image_resolver import LogicalPageImageResolver

            self._page_image_resolver = LogicalPageImageResolver(self.parse_result, zoom=3.0)
        logical_page = int(ref.get("logical_page") or ref.get("page") or 0)
        rendered = self._page_image_resolver(logical_page)
        if not rendered:
            return None
        if self._repair_engine is None:
            from docmirror.ocr.repair import LocalOCRRepairEngine

            self._repair_engine = LocalOCRRepairEngine()
        from docmirror.evidence.repair import RepairRequest

        self._targeted_requests += 1
        request_id = f"personal_detail:{logical_page}:{self._targeted_requests:04d}"
        request = RepairRequest(
            request_id=request_id,
            domain="credit_report.personal_detail",
            kind=role,
            page_number=logical_page,
            bbox=tuple(float(item) for item in ref["bbox"]),
            constraints=(role, "typed_validation", "preserve_raw"),
            evidence_ids=tuple(str(item) for item in ref.get("evidence_ids") or ()),
            reason="invalid_or_ambiguous_typed_ocr",
        )
        candidates = self._repair_engine.repair_from_image(
            request,
            rendered["image"],
            page_width=float(rendered["page_width"]),
            page_height=float(rendered["page_height"]),
            max_variants=8,
            min_confidence=0.45,
        )
        valid: list[tuple[float, str]] = []
        for candidate in candidates:
            normalized = _normalize_role(candidate.text, role)
            if _is_valid_for_role(normalized, role):
                valid.append((float(candidate.confidence), normalized))
        valid.sort(reverse=True)
        if not valid or valid[0][0] < 0.72:
            return None
        if len(valid) > 1 and valid[0][1] != valid[1][1] and valid[0][0] - valid[1][0] < 0.08:
            return None
        selected = valid[0][1]
        decision = self._record(
            role=role,
            original=original,
            corrected=selected,
            method="targeted_crop_ocr_consensus",
            reason_codes=("source_region_rerendered", "typed_validation", "candidate_margin"),
            confidence=valid[0][0],
            refs=refs,
            candidates=tuple(dict.fromkeys(value for _score, value in valid)),
        )
        return selected, decision

    @staticmethod
    def _line_role(text: str) -> str | None:
        compact = re.sub(r"\s+", "", text)
        has_inquiry_reason = any(marker in compact for marker in _INQUIRY_REASON_MARKERS)
        has_inquiry_date = bool(_INQUIRY_DATE_LIKE_RE.search(text))
        has_damaged_inquiry_date = bool(re.search(r"(?:^|\D)\d{0,3}\D{0,3}20\d{2}", text))
        if has_inquiry_reason and (has_inquiry_date or has_damaged_inquiry_date):
            return "inquiry_row"
        if re.search(r"^[^\u3400-\u9fff0-9]*[账联][户广尸]\s*\d{1,3}\b", text):
            return "account_line"
        return None

    def corrected_evidence_pages(self, pages: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        corrected_pages: list[dict[str, Any]] = []
        for raw_page in pages:
            page = deepcopy(raw_page)
            corrected_lines: list[dict[str, Any]] = []
            for raw_line in page.get("lines") or []:
                line = dict(raw_line)
                original = str(line.get("text") or line.get("content") or "")
                role = self._line_role(original)
                if role:
                    corrected, decision = self.correct_text(
                        original,
                        role=role,
                        source_refs=(
                            {
                                "source": "scanned_ocr_line",
                                "logical_page": int(page.get("page") or 0),
                                "source_page": int(page.get("source_page") or 0),
                                "bbox": list(line.get("bbox") or []),
                                "evidence_ids": list(line.get("evidence_ids") or []),
                            },
                        ),
                        confidence=float(line.get("confidence") or 0.0),
                        allow_targeted_ocr=role == "inquiry_row",
                    )
                    if decision is not None:
                        line["ocr_original_text"] = original
                        line["ocr_correction"] = decision.to_dict()
                        line["text"] = corrected
                        if "content" in line:
                            line["content"] = corrected
                corrected_lines.append(line)
            page["lines"] = corrected_lines
            corrected_pages.append(page)
        return corrected_pages

    def correct_business_candidates(self, payload: Mapping[str, Any], *, stage: str) -> dict[str, Any]:
        corrected = deepcopy(dict(payload))
        self._walk(corrected, parent=None, refs=(), stage=stage)
        self._apply_institution_consensus(corrected)
        self._promote_account_identifier_candidates(corrected)
        return corrected

    def _walk(
        self,
        value: Any,
        *,
        parent: str | None,
        refs: tuple[dict[str, Any], ...],
        stage: str,
    ) -> None:
        if isinstance(value, list):
            for item in value:
                self._walk(item, parent=parent, refs=refs, stage=stage)
            return
        if not isinstance(value, dict):
            return
        local_refs = _source_refs(value.get("source_refs")) or refs
        confidence = value.get("confidence")
        for key, item in list(value.items()):
            role = _FIELD_ROLES.get(str(key))
            if role and item not in (None, "") and not isinstance(item, (dict, list)):
                updated, decision = self.correct_text(
                    item,
                    role=role,
                    source_refs=local_refs,
                    confidence=float(confidence or 0.0),
                    allow_targeted_ocr=len(local_refs) == 1 and role in {
                        "identity_document_number",
                        "report_datetime",
                        "date",
                    },
                )
                if decision is not None:
                    value[key] = updated
            elif isinstance(item, dict) and role and item.get("value") not in (None, ""):
                nested_refs = _source_refs(item.get("source_refs")) or local_refs
                updated, decision = self.correct_text(
                    item["value"],
                    role=role,
                    source_refs=nested_refs,
                    confidence=float(item.get("confidence") or confidence or 0.0),
                    allow_targeted_ocr=role in {"identity_document_number", "report_datetime", "date"},
                )
                if decision is not None:
                    item.setdefault("raw", item["value"])
                    item["value"] = updated
            if isinstance(item, (dict, list)) and str(key) not in _RAW_OR_PROVENANCE_KEYS:
                self._walk(item, parent=str(key), refs=local_refs, stage=stage)

    def _apply_institution_consensus(self, payload: dict[str, Any]) -> None:
        fields: list[tuple[dict[str, Any], str, str]] = []

        def collect(value: Any) -> None:
            if isinstance(value, list):
                for item in value:
                    collect(item)
            elif isinstance(value, dict):
                for key, item in value.items():
                    if _FIELD_ROLES.get(str(key)) == "institution_name" and isinstance(item, str) and item:
                        fields.append((value, str(key), item))
                    collect(item)

        collect(payload)
        counts = Counter(normalize_institution_name(item) for _owner, _key, item in fields)
        references = [name for name, count in counts.items() if count >= 1 and _is_valid_for_role(name, "institution_name")]
        for owner, key, original in fields:
            current = normalize_institution_name(original)
            scored = sorted(
                ((SequenceMatcher(None, current, candidate).ratio(), counts[candidate], candidate) for candidate in references),
                reverse=True,
            )
            if not scored:
                continue
            best_score, best_count, best = scored[0]
            runner_score = scored[1][0] if len(scored) > 1 else 0.0
            if best != current and best_score >= 0.94 and best_score - runner_score >= 0.015 and best_count >= counts[current]:
                owner[key] = best
                self._record(
                    role="institution_name",
                    original=original,
                    corrected=best,
                    method="document_internal_consensus",
                    reason_codes=("typed_legal_suffix", "document_candidate_match", "candidate_margin"),
                    confidence=best_score,
                    refs=(),
                    candidates=tuple(item[2] for item in scored[:3]),
                )

    def _promote_account_identifier_candidates(self, payload: dict[str, Any]) -> None:
        accounts = payload.get("credit_accounts")
        if not isinstance(accounts, list):
            return
        for account in accounts:
            if not isinstance(account, dict) or account.get("account_identifier"):
                continue
            raw_candidates = account.get("account_identifier_candidates")
            if not isinstance(raw_candidates, list):
                continue
            candidates = tuple(
                dict.fromkeys(
                    normalized
                    for item in raw_candidates
                    if (normalized := _normalize_identifier(str(item or "")))
                    and _is_valid_for_role(normalized, "account_identifier")
                )
            )
            if len(candidates) != 1:
                continue
            account["account_identifier"] = candidates[0]
            self._record(
                role="account_identifier",
                original="",
                corrected=candidates[0],
                method="unique_typed_candidate",
                reason_codes=("missing_typed_field", "single_valid_candidate", "identifier_character_set"),
                confidence=float(account.get("confidence") or 0.9),
                refs=_source_refs(account.get("source_refs")),
                candidates=candidates,
            )


__all__ = [
    "PersonalDetailCorrectionDecision",
    "PersonalDetailOCRCorrectionOverlay",
    "normalize_institution_name",
]
