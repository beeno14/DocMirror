import json
import os
import re
from pathlib import Path
from typing import Any

import pytest

_INACTIVE_STATUSES = {"resolved", "suppressed_redundant", "informational"}
_UNRESOLVED_SEQUENCE_CODES = {
    "candidate_b_inquiry_multiple_missing_sequences_unresolved",
    "candidate_b_inquiry_boundary_sequence_unresolved",
    "candidate_b_inquiry_sequence_unresolved",
}
_CANONICAL_REASONS = {
    "本人查询",
    "贷后管理",
    "贷款审批",
    "信用卡审批",
    "担保资格审查",
    "融资审批",
    "保前审查",
    "保后管理",
    "客户准入资格审查",
    "资信审查",
    "法人代表、负责人、高管等资信审查",
    "特约商户实名审查",
    "异议处理",
    "司法调查",
}


def _dataset_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(dataset.get("name") or ""): dataset
        for dataset in payload.get("datasets") or ()
        if isinstance(dataset, dict) and dataset.get("name")
    }


def _evidence_value(row: dict[str, Any]) -> Any:
    values = row.get("normalized") or {}
    for key in (
        "string_value",
        "integer_value",
        "number_value",
        "boolean_value",
    ):
        if values.get(key) is not None:
            return values[key]
    return None


def _issue_evidence(
    evidence_rows: list[dict[str, Any]],
) -> dict[str, dict[tuple[str, str], Any]]:
    grouped: dict[str, dict[tuple[str, str], Any]] = {}
    for wrapper in evidence_rows:
        values = wrapper.get("normalized") or {}
        issue_id = str(values.get("extraction_issue_id") or "")
        if not issue_id:
            continue
        grouped.setdefault(issue_id, {})[
            (
                str(values.get("evidence_kind") or ""),
                str(values.get("evidence_path") or ""),
            )
        ] = _evidence_value(wrapper)
    return grouped


def _normalized_date(value: Any) -> str:
    text = str(value or "").strip()
    match = re.search(r"((?:19|20)\d{2})[./-]?(\d{2})[./-]?(\d{2})", text)
    return f"{match.group(1)}-{match.group(2)}-{match.group(3)}" if match else text


def _business_key(*, date: Any, institution: Any, reason: Any) -> tuple[str, str, str]:
    normalized_reason = re.sub(r"\s+", "", str(reason or ""))
    reason_matches = [
        candidate for candidate in _CANONICAL_REASONS if candidate in normalized_reason
    ]
    if reason_matches:
        normalized_reason = max(reason_matches, key=len)
    return (
        _normalized_date(date),
        re.sub(r"\s+", "", str(institution or "")),
        normalized_reason,
    )


def _omission_business_key(
    evidence: dict[tuple[str, str], Any],
) -> tuple[str, str, str] | None:
    date = evidence.get(("observed", "row.inquiry_date"))
    if date is None:
        date = evidence.get(("observed", "row.raw_inquiry_date"))
    institution = evidence.get(("observed", "row.institution"))
    reason = evidence.get(("observed", "row.reason"))
    if date is None:
        date = evidence.get(("observed", "row[1]"))
        institution = evidence.get(("observed", "row[2]"))
        reason = evidence.get(("observed", "row[3]"))
    if date is None or institution is None or reason is None:
        return None
    return _business_key(date=date, institution=institution, reason=reason)


def test_saved_lin_community_inquiry_lifecycle_and_normalized_fields() -> None:
    """Audit the real saved Community datasets, not an in-memory business view."""

    audit_dir = os.environ.get("DOCMIRROR_PERSONAL_DETAIL_SAVED_LIN_AUDIT_DIR")
    if not audit_dir:
        pytest.skip("set DOCMIRROR_PERSONAL_DETAIL_SAVED_LIN_AUDIT_DIR")
    directory = Path(audit_dir)
    community_path = directory / "林岚挺征信.community.json"
    assert community_path.is_file(), f"missing saved Lin artifact: {community_path}"
    payload = json.loads(community_path.read_text(encoding="utf-8"))
    datasets = _dataset_map(payload)
    assert {"inquiries", "extraction_issues", "extraction_issue_evidence"} <= set(
        datasets
    )

    inquiries = [
        wrapper.get("normalized") or {}
        for wrapper in datasets["inquiries"].get("rows") or ()
    ]
    issues = [
        wrapper.get("normalized") or {}
        for wrapper in datasets["extraction_issues"].get("rows") or ()
    ]
    evidence = _issue_evidence(
        list(datasets["extraction_issue_evidence"].get("rows") or ())
    )

    inquiry_dataset = datasets["inquiries"]
    assert inquiry_dataset.get("row_count") == 87
    assert inquiry_dataset.get("status") == "partial"
    assert inquiry_dataset.get("completeness") == {
        "expected_row_count": 90,
        "emitted_row_count": 87,
        "omitted_row_count": 3,
        "verified": False,
        "basis": "personal_detail_dataset_status:partial",
    }

    institutional = [
        row
        for row in inquiries
        if str(row.get("query_channel") or row.get("inquiry_type") or "")
        == "institution"
    ]
    institutional_sequences = {int(row["sequence"]) for row in institutional}
    assert institutional_sequences == set(range(1, 90)) - {4, 66, 67}
    assert len(inquiries) == 87

    emitted_business = {
        _business_key(
            date=row.get("inquiry_date"),
            institution=row.get("institution"),
            reason=row.get("reason"),
        )
        for row in institutional
        if row.get("institution") and row.get("reason")
    }
    active_omissions = [
        issue
        for issue in issues
        if str(issue.get("issue_code") or "") in _UNRESOLVED_SEQUENCE_CODES
        and str(issue.get("status") or "requires_review") not in _INACTIVE_STATUSES
    ]
    assert len(active_omissions) == 3
    omission_keys = []
    for issue in active_omissions:
        issue_id = str(issue.get("extraction_issue_id") or "")
        issue_evidence = evidence[issue_id]
        assert (
            "reason",
            next(
                path
                for kind, path in issue_evidence
                if kind == "reason"
                and issue_evidence[(kind, path)] == "record_not_emitted"
            ),
        ) in issue_evidence
        business_key = _omission_business_key(issue_evidence)
        assert business_key is not None
        assert issue_evidence.get(("observed", "row.raw_inquiry_date"))
        assert business_key not in emitted_business
        omission_keys.append(business_key)
    assert {key[0] for key in omission_keys} == {
        "2022-12-14",
        "2021-09-30",
        "2021-09-03",
    }

    gap = next(
        issue
        for issue in issues
        if issue.get("issue_code") == "canonical_inquiry_sequence_gap"
        and str(issue.get("status") or "requires_review") not in _INACTIVE_STATUSES
    )
    gap_evidence = evidence[str(gap["extraction_issue_id"])]
    missing_sequences = sorted(
        int(value)
        for (kind, path), value in gap_evidence.items()
        if kind == "candidate" and re.fullmatch(r"missing_sequences\[\d+\]", path)
    )
    assert missing_sequences == [4, 66, 67]

    inquiries_by_id = {
        str(row.get("inquiry_id") or ""): row for row in inquiries
    }
    for issue in issues:
        if (
            str(issue.get("status") or "requires_review") in _INACTIVE_STATUSES
            or issue.get("target_dataset") != "inquiries"
            or issue.get("field_name") != "institution"
        ):
            continue
        issue_evidence = evidence.get(str(issue.get("extraction_issue_id") or ""), {})
        if "normalized_value_withheld" not in set(issue_evidence.values()):
            continue
        target = inquiries_by_id.get(str(issue.get("target_record_id") or ""))
        assert target is not None
        assert target.get("institution") is None

    by_sequence = {int(row["sequence"]): row for row in institutional}
    for sequence in (1, 38, 39, 62):
        assert by_sequence[sequence].get("institution") is None
    assert by_sequence[87].get("reason") == "贷后管理"
    assert by_sequence[87].get("source_reason") == "如 贷后管理"
    for row in institutional:
        reason = row.get("reason")
        assert reason is None or reason in _CANONICAL_REASONS
        if reason is None:
            assert any(
                issue.get("target_record_id") == row.get("inquiry_id")
                and issue.get("target_dataset") == "inquiries"
                and issue.get("field_name") == "reason"
                and str(issue.get("status") or "requires_review")
                not in _INACTIVE_STATUSES
                for issue in issues
            )
