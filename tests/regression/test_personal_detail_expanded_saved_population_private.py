# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Saved-artifact population acceptance for the expanded detailed-report set.

This module deliberately stays separate from the large live-OCR private regression.
The source-audited identity universes below are acceptance oracles, not values inferred
from parser output.  A source row is conserved only when it is either emitted once or
represented by one exact, active, evidence-backed omission finding.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

import pytest

from docmirror.models.schemas.registry import validate_projection_payload
from docmirror.plugins.credit_report.value_utils import stable_record_id

_SAVED_AUDIT_ENV = "DOCMIRROR_PERSONAL_DETAIL_EXPANDED_SAVED_AUDIT_DIR"
_FIXTURE_DIR = Path(
    os.environ.get(
        "DOCMIRROR_PERSONAL_DETAIL_FIXTURE_DIR",
        "tests/fixtures-private/credit_report/Scanned Personal Detailed",
    )
)

_ACCOUNT_ID = re.compile(r"^credit_account:(?P<family>[a-z][a-z0-9_]*):(?P<ordinal>[1-9]\d*)$")
_AGREEMENT_OMISSION_ID = re.compile(r"^(?:credit_agreement|credit_line)(?::sequence)?:(?P<ordinal>[1-9]\d*)$")
_INQUIRY_OMISSION_ID = re.compile(
    r"^(?:credit_)?inquiry:(?P<inquiry_type>institution|personal):"
    r"(?P<sequence>[1-9]\d*)$"
)
_MONTH_ID = re.compile(r"^(?P<grid_id>.+):(?P<month>\d{4}-(?:0[1-9]|1[0-2]))$")
_SOURCE_ACCOUNT_MONTH_ID = re.compile(
    r"^source_account_month:(?P<owner_hash>[0-9a-f]{16}):"
    r"(?P<month>\d{4}-(?:0[1-9]|1[0-2]))$"
)
_MONTH_OMISSION_FIELDS = frozenset({"performance_month", "status_code"})
_ACTIVE_ISSUE_STATUSES = frozenset(
    {"active", "open", "requires_review", "review", "warning", "error"}
)
_RECORD_OMISSION_ISSUE_CODES = {
    "credit_accounts": frozenset({"source_account_record_omitted"}),
    "credit_agreements": frozenset({"source_credit_agreement_record_omitted"}),
    "inquiries": frozenset({"source_inquiry_record_omitted"}),
}
_MONTH_OMISSION_ISSUE_CODES = frozenset(
    {
        "candidate_b_monthly_account_range_missing_month",
        "candidate_b_monthly_grid_owner_unresolved_field",
        "candidate_b_monthly_grid_contract_missing_field",
        "candidate_b_monthly_owned_grid_missing_field",
        "canonical_monthly_source_structure_missing_field",
    }
)
_RECORD_IDENTITY_EVIDENCE = {
    "credit_accounts": (
        ("account_type", 0),
        ("category_sequence", 1),
    ),
    "credit_agreements": (("credit_agreement_sequence", None),),
    "inquiries": (
        ("inquiry_type", 0),
        ("sequence", 1),
    ),
}
_RECORD_IDENTITY_REF_FIELDS = {
    "credit_accounts": frozenset({"account_id", "category_sequence", "sequence"}),
    "credit_agreements": frozenset(
        {"credit_agreement_id", "credit_line_id", "sequence"}
    ),
    "inquiries": frozenset({"inquiry_id", "sequence"}),
}
_RECORD_IDENTITY_BINDINGS = {
    "credit_accounts": frozenset(
        {
            "canonical_sequence_row",
            "printed_account_anchor",
            "printed_account_anchor_heading",
        }
    ),
    "credit_agreements": frozenset(
        {
            "canonical_header_row",
            "canonical_sequence_row",
            "printed_credit_agreement_ordinal",
        }
    ),
    "inquiries": frozenset(
        {"canonical_header_column", "canonical_header_row", "canonical_sequence_row"}
    ),
}
_EXACT_FIELD_BINDINGS = frozenset(
    {
        "canonical_field_slot",
        "canonical_header_column",
        "canonical_label_slot",
        "grid_month_cell",
        "monthly_grid_cell",
        "source_monthly_field_cell",
    }
)


@dataclass(frozen=True)
class PopulationOracle:
    fixture_name: str
    sha256: str
    account_families: dict[str, int]
    month_positions: int
    agreements: int
    inquiry_types: dict[str, int]
    month_identity_ledger: str | None = None
    month_identity_ledger_sha256: str | None = None
    month_identity_set_sha256: str | None = None

    @property
    def stem(self) -> str:
        return Path(self.fixture_name).stem

    @property
    def accounts(self) -> int:
        return sum(self.account_families.values())

    @property
    def inquiries(self) -> int:
        return sum(self.inquiry_types.values())


# SHA-256 values bind every manually audited population to the exact PDF revision.
_ORACLES = (
    PopulationOracle(
        fixture_name="王根镇征信.pdf",
        sha256="eb6e963fefab972c1d74147be4943741233d912c06f57e9d60d2316dd62aecd2",
        account_families={
            "non_revolving_loan": 28,
            "revolving_loan_subaccount": 3,
            "revolving_loan_account": 17,
            "credit_card": 12,
        },
        month_positions=798,
        agreements=23,
        inquiry_types={"institution": 143},
    ),
    PopulationOracle(
        fixture_name="黄圣辉_个人详版征信报告.pdf",
        sha256="8d5482f9e354d2d9b02614942283c0c62fce56b49228e8c5b3069e5aaf85f0a5",
        account_families={
            "non_revolving_loan": 18,
            "revolving_loan_subaccount": 55,
            "revolving_loan_account": 8,
            "credit_card": 34,
            "quasi_credit_card": 1,
        },
        month_positions=875,
        agreements=19,
        inquiry_types={"institution": 145, "personal": 6},
        month_identity_ledger=(
            "artifacts/personal_detail_100pct_iteration_20260813/"
            "huang_month_truth/ledger.jsonl"
        ),
        month_identity_ledger_sha256=(
            "ced81f694f18f47a226fd7143a93c88a8f90516189ae9f3c2b1c5813ca1abc0c"
        ),
        month_identity_set_sha256=(
            "4b680476155015013a48caf5e6072b3f12410d103faa7e8b50f6cbc870542f73"
        ),
    ),
    PopulationOracle(
        fixture_name="曹末艳-征信.pdf",
        sha256="1a33a80b4b818640105e94488db00f1656d8f3e86004caa92f0e98163d523eee",
        account_families={
            "non_revolving_loan": 115,
            "revolving_loan_subaccount": 6,
            "revolving_loan_account": 2,
        },
        month_positions=666,
        agreements=2,
        inquiry_types={"institution": 18},
    ),
)


@dataclass(frozen=True)
class OmissionClaim:
    identity: Hashable
    field_name: str | None
    issue_id: str


def _dataset_map(payload: dict) -> dict[str, dict]:
    return {
        str(dataset.get("name") or ""): dataset
        for dataset in payload.get("datasets") or ()
        if isinstance(dataset, dict) and dataset.get("name")
    }


def _normalized_rows(dataset: dict) -> list[dict]:
    return [wrapper.get("normalized") or {} for wrapper in dataset.get("rows") or () if isinstance(wrapper, dict)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_account_identity(value: str) -> tuple[str, int] | None:
    match = _ACCOUNT_ID.fullmatch(value)
    if match is None:
        return None
    return match.group("family"), int(match.group("ordinal"))


def _parse_agreement_omission(value: str) -> int | None:
    match = _AGREEMENT_OMISSION_ID.fullmatch(value)
    return int(match.group("ordinal")) if match is not None else None


def _parse_inquiry_omission(value: str) -> tuple[str, int] | None:
    match = _INQUIRY_OMISSION_ID.fullmatch(value)
    if match is None:
        return None
    return match.group("inquiry_type"), int(match.group("sequence"))


def _inquiry_omission_parser(
    oracle: PopulationOracle,
) -> Callable[[str], tuple[str, int] | None]:
    """Accept readable IDs and the canonical deterministic inquiry IDs."""

    stable_targets = {
        stable_record_id("credit_inquiry", inquiry_type, sequence): (
            inquiry_type,
            sequence,
        )
        for inquiry_type, count in oracle.inquiry_types.items()
        for sequence in range(1, count + 1)
    }

    def parse(value: str) -> tuple[str, int] | None:
        return _parse_inquiry_omission(value) or stable_targets.get(value)

    return parse


def _parse_month_identity(value: str) -> str | None:
    return value if _MONTH_ID.fullmatch(value) is not None else None


def _parse_source_account_month_identity(value: str) -> tuple[str, str] | None:
    match = _SOURCE_ACCOUNT_MONTH_ID.fullmatch(value)
    if match is None:
        return None
    return match.group("owner_hash"), match.group("month")


def _source_account_month_owner_hash(account_id: str) -> str:
    return stable_record_id("source_account_month_owner", account_id).split(":", 1)[-1]


def _source_account_month_claim_identity(
    *,
    target: str,
    field_name: str | None,
    issue_code: str,
    evidence_rows: list[Mapping[str, object]],
    refs: list[dict],
    issue_page_range: tuple[int, int],
) -> tuple[str, str] | None:
    """Validate a stable exact account/month omission contract.

    This credits only the performance-month identity.  A paired status
    diagnostic is deliberately excluded because it is not a second identity.
    """

    parsed = _parse_source_account_month_identity(target)
    if (
        parsed is None
        or field_name != "performance_month"
        or issue_code
        not in {
            "candidate_b_monthly_account_range_missing_month",
            "candidate_b_monthly_owned_grid_missing_field",
        }
    ):
        return None
    observed: dict[str, list[object]] = {}
    for row in evidence_rows:
        if str(row.get("evidence_kind") or "") != "observed":
            continue
        observed.setdefault(str(row.get("evidence_path") or ""), []).append(
            _evidence_scalar(row)
        )
    account_values = observed.get("account_id")
    month_values = observed.get("performance_month")
    if (
        not account_values
        or len(account_values) != 1
        or not month_values
        or month_values != [parsed[1]]
    ):
        return None
    account_id = str(account_values[0] or "")
    if not account_id or parsed[0] != _source_account_month_owner_hash(account_id):
        return None
    if issue_code == "candidate_b_monthly_account_range_missing_month":
        valid_refs = [
            ref
            for ref in refs
            if _positive_page(ref.get("logical_page")) is not None
            and _positive_page(ref.get("source_page")) is not None
            and issue_page_range[0]
            <= int(ref["logical_page"])
            <= issue_page_range[1]
            and _finite_nondegenerate_bbox(ref.get("bbox")) is not None
            and _nonempty_evidence_ids(ref)
            and ref.get("geometry_scope") == "line"
            and ref.get("binding") == "source_account_month_range"
            and ref.get("binding_quality") == "source_account_month_range"
            and ref.get("field_name") == "performance_month"
            and ref.get("account_id") == account_id
            and ref.get("performance_month") == parsed[1]
        ]
    else:
        if set(observed) != {"account_id", "performance_month"}:
            return None
        valid_refs = []
        for ref in refs:
            binding = str(ref.get("binding") or "")
            binding_quality = str(ref.get("binding_quality") or "")
            if (
                str(ref.get("source") or "")
                == "candidate_b_monthly_owned_grid_cell"
                and _positive_page(ref.get("logical_page")) is not None
                and _positive_page(ref.get("source_page")) is not None
                and issue_page_range[0]
                <= int(ref["logical_page"])
                <= issue_page_range[1]
                and _finite_nondegenerate_bbox(ref.get("bbox")) is not None
                and _nonempty_evidence_ids(ref)
                and _row_cell_ref(ref)
                and str(ref.get("geometry_scope") or "") == "cell"
                and str(ref.get("field_name") or "") == "performance_month"
                and str(ref.get("grid_id") or "")
                and str(ref.get("performance_month") or "") == parsed[1]
                and str(ref.get("account_id") or "") == account_id
                and binding == "source_account_month_identity"
                and binding_quality == "source_account_month_identity"
            ):
                valid_refs.append(ref)
    return (
        (account_id, parsed[1])
        if (
            len(refs) == len(valid_refs) == 1
            if issue_code == "candidate_b_monthly_account_range_missing_month"
            else len(refs) == len(valid_refs) == 1
        )
        else None
    )


def _monthly_grid_account_claim_identity(
    identity: Hashable,
    evidence_rows: list[Mapping[str, object]],
) -> tuple[str, str] | None:
    """Project a grid/month claim onto its exact public account/month owner."""

    match = _MONTH_ID.fullmatch(str(identity))
    if match is None:
        return None
    account_values = [
        _evidence_scalar(row)
        for row in evidence_rows
        if str(row.get("evidence_kind") or "") == "observed"
        and str(row.get("evidence_path") or "") == "account_id"
    ]
    if len(account_values) != 1:
        return None
    account_id = str(account_values[0] or "")
    if _ACCOUNT_ID.fullmatch(account_id) is None:
        return None
    return account_id, match.group("month")


def _positive_page(value: object) -> int | None:
    return value if type(value) is int and value > 0 else None


def _nonnegative_index(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _finite_nondegenerate_bbox(value: object) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    if any(type(coordinate) not in {int, float} for coordinate in value):
        return None
    bbox = tuple(float(coordinate) for coordinate in value)
    if (
        not all(isfinite(coordinate) for coordinate in bbox)
        or bbox[2] <= bbox[0]
        or bbox[3] <= bbox[1]
    ):
        return None
    return bbox


def _source_page_range(wrapper: Mapping[str, object]) -> tuple[int, int] | None:
    source = wrapper.get("source")
    if not isinstance(source, Mapping):
        return None
    page_range = source.get("page_range")
    if not isinstance(page_range, (list, tuple)) or len(page_range) != 2:
        return None
    first = _positive_page(page_range[0])
    last = _positive_page(page_range[1])
    if first is None or last is None or last < first:
        return None
    return first, last


def _nonempty_evidence_ids(ref: Mapping[str, object]) -> bool:
    values = ref.get("evidence_ids")
    return isinstance(values, (list, tuple)) and bool(
        values
        and all(isinstance(value, str) and value.strip() for value in values)
        and len(set(values)) == len(values)
    )


def _row_cell_ref(ref: Mapping[str, object]) -> bool:
    return bool(
        str(ref.get("table_id") or "").strip()
        and _nonnegative_index(ref.get("row")) is not None
        and _nonnegative_index(ref.get("column")) is not None
    )


def _exact_source_ref(
    ref: Mapping[str, object],
    *,
    issue_page_range: tuple[int, int],
    dataset_name: str,
    field_name: str | None,
    identity: Hashable,
) -> bool:
    logical_page = _positive_page(ref.get("logical_page"))
    source_page = _positive_page(ref.get("source_page"))
    if (
        logical_page is None
        or source_page is None
        or not issue_page_range[0] <= logical_page <= issue_page_range[1]
        or not _nonempty_evidence_ids(ref)
    ):
        return False

    bbox = _finite_nondegenerate_bbox(ref.get("bbox"))
    cell = _row_cell_ref(ref)
    if bbox is None and not cell:
        return False

    geometry_scope = str(ref.get("geometry_scope") or "")
    binding = str(ref.get("binding") or "")
    binding_quality = str(ref.get("binding_quality") or "")
    ref_field = str(ref.get("field_name") or "")
    if dataset_name == "credit_account_monthly_performance":
        # A month-position omission is a field claim, not a row-count claim.
        # It must survive Community serialization with the exact source cell
        # and the same canonical field binding; page-only provenance cannot
        # distinguish neighbouring months or status cells.
        match = _MONTH_ID.fullmatch(str(identity))
        return bool(
            match is not None
            and field_name in _MONTH_OMISSION_FIELDS
            and geometry_scope == "cell"
            and cell
            and ref_field == field_name
            and str(ref.get("grid_id") or "") == match.group("grid_id")
            and str(ref.get("performance_month") or "") == match.group("month")
            and binding in _EXACT_FIELD_BINDINGS
            and (
                not binding_quality
                or binding_quality in _EXACT_FIELD_BINDINGS
            )
        )

    if field_name not in _RECORD_IDENTITY_REF_FIELDS.get(dataset_name, frozenset()):
        return False
    expected_ordinal = (
        identity[1]
        if isinstance(identity, tuple) and len(identity) == 2
        else identity
    )
    ref_ordinal = ref.get("sequence", ref.get("category_sequence"))
    exact_ordinal = type(ref_ordinal) is int and ref_ordinal == expected_ordinal
    field_bound = ref_field == field_name or ref_field in {
        "sequence",
        "category_sequence",
    }
    exact_binding = binding in _RECORD_IDENTITY_BINDINGS.get(
        dataset_name, frozenset()
    ) or binding_quality in _RECORD_IDENTITY_BINDINGS.get(
        dataset_name, frozenset()
    )
    return bool(
        geometry_scope in {"cell", "row", "line", "canonical_sequence_cell"}
        and (cell or bbox is not None)
        and exact_ordinal
        and field_bound
        and exact_binding
    )


def _evidence_scalar(row: Mapping[str, object]) -> object:
    value_type = str(row.get("value_type") or "")
    value_key = {
        "string": "string_value",
        "integer": "integer_value",
        "number": "number_value",
        "boolean": "boolean_value",
    }.get(value_type)
    return row.get(value_key) if value_key is not None else None


def _identity_evidence_matches(
    rows: list[Mapping[str, object]],
    *,
    dataset_name: str,
    identity: Hashable,
    field_name: str | None,
) -> bool:
    if dataset_name == "credit_account_monthly_performance":
        match = _MONTH_ID.fullmatch(str(identity))
        if match is None or field_name not in _MONTH_OMISSION_FIELDS:
            return False
        expected = {
            "grid_id": match.group("grid_id"),
            "performance_month": match.group("month"),
        }
    else:
        spec = _RECORD_IDENTITY_EVIDENCE.get(dataset_name)
        if spec is None:
            return False
        expected = {
            path: identity[index] if index is not None else identity
            for path, index in spec
        }

    observed: dict[str, list[object]] = {}
    for row in rows:
        if str(row.get("evidence_kind") or "") != "observed":
            continue
        path = str(row.get("evidence_path") or "")
        observed.setdefault(path, []).append(_evidence_scalar(row))
    return all(observed.get(path) == [value] for path, value in expected.items())


def _issue_evidence_by_id(datasets: dict[str, dict]) -> dict[str, list[dict]]:
    evidence = datasets.get("extraction_issue_evidence") or {}
    rows: dict[str, list[dict]] = {}
    for wrapper in evidence.get("rows") or ():
        if not isinstance(wrapper, dict):
            continue
        values = wrapper.get("normalized") or {}
        if not isinstance(values, dict):
            continue
        issue_id = str(values.get("extraction_issue_id") or "")
        evidence_id = str(values.get("extraction_issue_evidence_id") or "")
        if not issue_id or not evidence_id:
            continue
        rows.setdefault(issue_id, []).append(wrapper)
    return rows


def _issue_source_refs(wrapper: Mapping[str, object]) -> list[dict]:
    source = wrapper.get("source")
    if not isinstance(source, Mapping):
        return []
    refs = source.get("source_refs") or source.get("source_cell_refs") or ()
    return [dict(ref) for ref in refs if isinstance(ref, Mapping)]


def _exact_omission_claims(
    payload: dict,
    *,
    dataset_name: str,
    parse_identity: Callable[[str], Hashable | None],
    allowed_fields: frozenset[str] | None = None,
) -> list[OmissionClaim]:
    """Return only exact, active, evidence-backed record-level omissions.

    Aggregate sequence/count gaps and grid-only findings intentionally do not count.
    They can describe risk, but cannot prove which source row was withheld.
    """

    datasets = _dataset_map(payload)
    evidence_by_issue = _issue_evidence_by_id(datasets)
    issues = datasets.get("extraction_issues") or {}
    claims: list[OmissionClaim] = []
    for wrapper in issues.get("rows") or ():
        if not isinstance(wrapper, dict):
            continue
        issue = wrapper.get("normalized") or {}
        if not isinstance(issue, dict):
            continue
        if issue.get("target_dataset") != dataset_name:
            continue
        issue_code = str(issue.get("issue_code") or "")
        allowed_codes = (
            _MONTH_OMISSION_ISSUE_CODES
            if dataset_name == "credit_account_monthly_performance"
            else _RECORD_OMISSION_ISSUE_CODES.get(dataset_name, frozenset())
        )
        if issue_code not in allowed_codes:
            continue
        if str(issue.get("status") or "").strip().lower() not in _ACTIVE_ISSUE_STATUSES:
            continue
        issue_id = str(issue.get("extraction_issue_id") or "")
        evidence_wrappers = evidence_by_issue.get(issue_id, [])
        if not issue_id or not evidence_wrappers:
            continue
        target = str(issue.get("target_record_id") or "")
        identity = parse_identity(target)
        field_name_value = issue.get("field_name")
        field_name = str(field_name_value) if field_name_value is not None else None
        if allowed_fields is not None and field_name not in allowed_fields:
            continue
        evidence_rows = [
            evidence.get("normalized") or {}
            for evidence in evidence_wrappers
            if isinstance(evidence.get("normalized"), dict)
        ]
        issue_page_range = _source_page_range(wrapper)
        source_refs = _issue_source_refs(wrapper)
        source_account_month_identity = (
            _source_account_month_claim_identity(
                target=target,
                field_name=field_name,
                issue_code=issue_code,
                evidence_rows=evidence_rows,
                refs=source_refs,
                issue_page_range=issue_page_range,
            )
            if dataset_name == "credit_account_monthly_performance"
            and issue_page_range is not None
            else None
        )
        if source_account_month_identity is not None:
            identity = source_account_month_identity
        if identity is None:
            continue
        canonical_monthly_identity = (
            _monthly_grid_account_claim_identity(identity, evidence_rows)
            if dataset_name == "credit_account_monthly_performance"
            and source_account_month_identity is None
            else None
        )
        if (
            issue_page_range is None
            or any(_source_page_range(evidence) != issue_page_range for evidence in evidence_wrappers)
            or (
                source_account_month_identity is None
                and not _identity_evidence_matches(
                    evidence_rows,
                    dataset_name=dataset_name,
                    identity=identity,
                    field_name=field_name,
                )
            )
            or (
                source_account_month_identity is None
                and not any(
                    _exact_source_ref(
                        ref,
                        issue_page_range=issue_page_range,
                        dataset_name=dataset_name,
                        field_name=field_name,
                        identity=identity,
                    )
                    for ref in source_refs
                )
            )
        ):
            continue
        claims.append(
            OmissionClaim(
                identity=canonical_monthly_identity or identity,
                field_name=field_name,
                issue_id=issue_id,
            )
        )
    return claims


def _assert_no_duplicate_omission_findings(
    label: str,
    claims: list[OmissionClaim],
    *,
    distinguish_fields: bool,
) -> None:
    # Two different fields on one absent month are distinct useful findings.  Two
    # findings for the same identity/field are duplicate error reports.
    keys = (
        [(claim.identity, claim.field_name) for claim in claims]
        if distinguish_fields
        else [claim.identity for claim in claims]
    )
    duplicate_keys = {key: count for key, count in Counter(keys).items() if count > 1}
    assert not duplicate_keys, f"{label}: duplicate omission findings={duplicate_keys!r}"


def _assert_exact_partition(
    label: str,
    *,
    expected: set[Hashable],
    emitted: list[Hashable],
    claims: list[OmissionClaim],
) -> None:
    _assert_no_duplicate_omission_findings(label, claims, distinguish_fields=False)
    emitted_counts = Counter(emitted)
    duplicate_emitted = {identity: count for identity, count in emitted_counts.items() if count > 1}
    assert not duplicate_emitted, f"{label}: duplicate emitted identities={duplicate_emitted!r}"

    emitted_set = set(emitted)
    omitted_set = {claim.identity for claim in claims}
    assert not emitted_set - expected, f"{label}: extra emitted={emitted_set - expected!r}"
    assert not omitted_set - expected, f"{label}: extra reported omissions={omitted_set - expected!r}"
    assert not emitted_set & omitted_set, (
        f"{label}: identities both emitted and reported omitted={emitted_set & omitted_set!r}"
    )
    missing = expected - emitted_set - omitted_set
    assert not missing, f"{label}: silent omissions={missing!r}"
    assert emitted_set | omitted_set == expected


def _assert_count_partition(
    label: str,
    *,
    expected_count: int,
    emitted: list[Hashable],
    claims: list[OmissionClaim],
) -> None:
    """Reject count-only conservation without a source-audited identity universe.

    A total of N month positions cannot prove which N identities exist.  Passing
    merely because ``len(emitted) + len(reported) == N`` would let arbitrary
    syntactic grid/month IDs manufacture a perfect score.
    """

    _assert_no_duplicate_omission_findings(label, claims, distinguish_fields=True)
    emitted_counts = Counter(emitted)
    duplicate_emitted = {identity: count for identity, count in emitted_counts.items() if count > 1}
    assert not duplicate_emitted, f"{label}: duplicate emitted identities={duplicate_emitted!r}"
    emitted_set = set(emitted)
    omitted_set = {claim.identity for claim in claims}
    assert not emitted_set & omitted_set, (
        f"{label}: identities both emitted and reported omitted={emitted_set & omitted_set!r}"
    )
    assert not omitted_set, (
        f"{label}: count-only source total cannot validate omission identities; "
        "add a source-audited exact identity universe"
    )
    assert len(emitted_set) == expected_count, (
        f"{label}: emitted={len(emitted_set)} != source={expected_count}; "
        "an aggregate source count cannot localize the missing month identities"
    )


def _assert_completeness(
    dataset_name: str,
    dataset: dict,
    *,
    expected_count: int,
    emitted_count: int,
    omitted_count: int,
) -> None:
    rows = dataset.get("rows") or []
    assert dataset.get("row_count") == len(rows), f"{dataset_name}: row_count disagrees with public rows"
    assert emitted_count == len(rows), f"{dataset_name}: extracted emitted identity count disagrees with public rows"
    completeness = dataset.get("completeness") or {}
    assert completeness.get("expected_row_count") == expected_count, (
        f"{dataset_name}: expected-row report={completeness.get('expected_row_count')!r}, source={expected_count}"
    )
    assert completeness.get("emitted_row_count") == emitted_count, (
        f"{dataset_name}: emitted-row report={completeness.get('emitted_row_count')!r}, public={emitted_count}"
    )
    assert completeness.get("omitted_row_count") == omitted_count, (
        f"{dataset_name}: omitted-row report={completeness.get('omitted_row_count')!r}, localized={omitted_count}"
    )
    assert expected_count == emitted_count + omitted_count


def _account_identities(dataset: dict) -> list[tuple[str, int]]:
    identities: list[tuple[str, int]] = []
    for wrapper in dataset.get("rows") or ():
        row = wrapper.get("normalized") or {}
        record_id = str(wrapper.get("record_id") or "")
        identity = _parse_account_identity(record_id)
        assert identity is not None, f"credit_accounts: noncanonical record_id={record_id!r}"
        family, _ordinal = identity
        assert row.get("account_id") == record_id
        assert row.get("account_type") == family
        identities.append(identity)
    return identities


def _agreement_identities(dataset: dict) -> list[int]:
    identities: list[int] = []
    for row in _normalized_rows(dataset):
        sequence = row.get("sequence")
        assert type(sequence) is int and sequence > 0, f"credit_agreements: invalid source ordinal={sequence!r}"
        identities.append(sequence)
    return identities


def _inquiry_identities(dataset: dict) -> list[tuple[str, int]]:
    identities: list[tuple[str, int]] = []
    for row in _normalized_rows(dataset):
        inquiry_type = row.get("inquiry_type")
        channel = row.get("query_channel")
        assert inquiry_type in {"institution", "personal"}, f"inquiries: invalid type={inquiry_type!r}"
        if channel is not None:
            assert channel == inquiry_type, f"inquiries: inquiry_type={inquiry_type!r} != query_channel={channel!r}"
        sequence = row.get("sequence")
        assert type(sequence) is int and sequence > 0, f"inquiries: invalid source sequence={sequence!r}"
        identities.append((str(inquiry_type), sequence))
    return identities


def _month_identities(dataset: dict) -> list[tuple[str, str]]:
    identities: list[tuple[str, str]] = []
    for wrapper in dataset.get("rows") or ():
        row = wrapper.get("normalized") or {}
        record_id = str(wrapper.get("record_id") or "")
        match = _MONTH_ID.fullmatch(record_id)
        assert match is not None, f"credit_account_monthly_performance: invalid identity={record_id!r}"
        month = row.get("performance_month")
        grid_id = row.get("grid_id")
        assert month == match.group("month")
        assert grid_id == match.group("grid_id")
        assert row.get("monthly_performance_id") == record_id
        account_id = row.get("account_id")
        assert isinstance(account_id, str) and account_id, (
            f"{record_id}: emitted structural month has no account_id"
        )
        assert isinstance(row.get("status_code"), str) and row["status_code"], (
            f"{record_id}: emitted structural month has no status_code"
        )
        identities.append((account_id, str(month)))
    return identities


def _month_identity_ledger(
    oracle: PopulationOracle,
) -> set[tuple[str, str]] | None:
    """Load an optional source-audited exact account/month universe.

    The Huang package is independently signed off for account/month identity
    only.  Binding it by source, file, and normalized-set hashes prevents an
    edited audit file from silently changing acceptance semantics.
    """

    if oracle.month_identity_ledger is None:
        return None
    ledger_path = Path(oracle.month_identity_ledger)
    assert ledger_path.is_file(), ledger_path
    assert oracle.month_identity_ledger_sha256 is not None
    assert _sha256(ledger_path) == oracle.month_identity_ledger_sha256
    manifest_path = ledger_path.parent / "source_manifest.json"
    assert manifest_path.is_file(), manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest.get("status") == "source_identity_ledger_human_render_signed_off"
    signoff = manifest.get("human_source_render_signoff") or {}
    assert signoff.get("scope") == "account_id_and_performance_month_identity_only"
    assert signoff.get("normalized_identity_count") == oracle.month_positions
    assert signoff.get("normalized_identity_sha256") == (
        oracle.month_identity_set_sha256
    )
    assert signoff.get("certifies_monthly_values") is False
    signoff_path = Path(str(signoff.get("path") or ""))
    assert signoff_path.is_file(), signoff_path
    source_pdfs = [
        item
        for item in manifest.get("inputs") or ()
        if isinstance(item, dict) and item.get("role") == "source_pdf"
    ]
    assert len(source_pdfs) == 1
    assert source_pdfs[0].get("sha256") == oracle.sha256
    assert Path(str(source_pdfs[0].get("path") or "")).name == oracle.fixture_name
    rows = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    identities = {
        (str(row.get("account_id") or ""), str(row.get("performance_month") or ""))
        for row in rows
    }
    assert len(rows) == len(identities) == oracle.month_positions
    assert all(
        _ACCOUNT_ID.fullmatch(str(row.get("account_id") or "")) is not None
        and re.fullmatch(
            r"\d{4}-(?:0[1-9]|1[0-2])",
            str(row.get("performance_month") or ""),
        )
        is not None
        for row in rows
    )
    identity_body = "\n".join(
        f"{account_id}|{month}" for account_id, month in sorted(identities)
    )
    assert oracle.month_identity_set_sha256 is not None
    assert hashlib.sha256(identity_body.encode("utf-8")).hexdigest() == (
        oracle.month_identity_set_sha256
    )
    return identities


def _expected_account_identities(oracle: PopulationOracle) -> set[tuple[str, int]]:
    return {(family, ordinal) for family, count in oracle.account_families.items() for ordinal in range(1, count + 1)}


def _expected_inquiry_identities(oracle: PopulationOracle) -> set[tuple[str, int]]:
    return {
        (inquiry_type, sequence)
        for inquiry_type, count in oracle.inquiry_types.items()
        for sequence in range(1, count + 1)
    }


def _assert_source_metadata(oracle: PopulationOracle, community: dict, semantic: dict) -> None:
    fixture = _FIXTURE_DIR / oracle.fixture_name
    assert fixture.is_file(), fixture
    assert _sha256(fixture) == oracle.sha256, f"fixture hash changed: {fixture}"

    expected_hash = f"sha256:{oracle.sha256}"
    community_file = (community.get("document") or {}).get("source_file") or {}
    semantic_file = (semantic.get("source") or {}).get("file") or {}
    for label, source_file in (("Community", community_file), ("Semantic", semantic_file)):
        assert source_file.get("name") == oracle.fixture_name, f"{label}: source name={source_file.get('name')!r}"
        assert source_file.get("sha256") == expected_hash, f"{label}: source sha256={source_file.get('sha256')!r}"


def _assert_semantic_source_oracle(oracle: PopulationOracle, semantic: dict) -> None:
    ledger = semantic["domain"]["facts"]["personal_detail_source_completeness_ledger"]
    assert ledger.get("credit_accounts") == oracle.accounts
    assert ledger.get("account_family_source_populations") == oracle.account_families
    assert ledger.get("credit_agreements") == oracle.agreements
    assert ledger.get("credit_agreement_sequence_endpoint") == oracle.agreements
    assert ledger.get("inquiry_records") == oracle.inquiries
    assert ledger.get("inquiry_sequence_endpoints") == oracle.inquiry_types


def _account_month_closure(semantic: dict) -> dict:
    closure = semantic["domain"]["facts"].get(
        "personal_detail_account_month_closure"
    )
    assert isinstance(closure, dict), "missing account/month source-position ledger"
    raw_positions = closure.get("raw_source_month_positions")
    owner_bound = closure.get("owner_bound_account_months")
    owner_unresolved = closure.get("owner_unresolved_positions")
    aliases = closure.get("alias_source_month_positions")
    expected_identities = closure.get("expected_identity_count")
    assert all(
        type(value) is int and value >= 0
        for value in (
            raw_positions,
            owner_bound,
            owner_unresolved,
            aliases,
            expected_identities,
        )
    )
    assert raw_positions == owner_bound + owner_unresolved
    assert aliases <= owner_bound
    assert closure.get("source_position_balance_valid") is True
    assert closure.get("unlocalized_owner_unresolved_positions") == 0
    return closure


def _assert_saved_population_oracle(oracle: PopulationOracle, community: dict, semantic: dict) -> None:
    community_validation = validate_projection_payload("community", community)
    assert community_validation.valid, community_validation.errors
    semantic_validation = validate_projection_payload("community_semantic", semantic)
    assert semantic_validation.valid, semantic_validation.errors
    _assert_source_metadata(oracle, community, semantic)
    _assert_semantic_source_oracle(oracle, semantic)

    datasets = _dataset_map(community)
    required = {
        "credit_accounts",
        "credit_account_monthly_performance",
        "credit_agreements",
        "inquiries",
        "extraction_issues",
        "extraction_issue_evidence",
    }
    assert not required - set(datasets), f"missing datasets={required - set(datasets)!r}"

    accounts = _account_identities(datasets["credit_accounts"])
    account_claims = _exact_omission_claims(
        community,
        dataset_name="credit_accounts",
        parse_identity=_parse_account_identity,
        allowed_fields=frozenset({"account_id"}),
    )
    _assert_exact_partition(
        "credit_accounts",
        expected=set(_expected_account_identities(oracle)),
        emitted=list(accounts),
        claims=account_claims,
    )
    _assert_completeness(
        "credit_accounts",
        datasets["credit_accounts"],
        expected_count=oracle.accounts,
        emitted_count=len(accounts),
        omitted_count=len({claim.identity for claim in account_claims}),
    )

    months = _month_identities(datasets["credit_account_monthly_performance"])
    month_claims = _exact_omission_claims(
        community,
        dataset_name="credit_account_monthly_performance",
        parse_identity=_parse_month_identity,
        allowed_fields=_MONTH_OMISSION_FIELDS,
    )
    closure = _account_month_closure(semantic)
    expected_identity_count = closure["expected_identity_count"]
    expected_months = _month_identity_ledger(oracle)
    if expected_months is None:
        assert closure["raw_source_month_positions"] == oracle.month_positions
        _assert_count_partition(
            "credit_account_monthly_performance",
            expected_count=expected_identity_count,
            emitted=list(months),
            claims=month_claims,
        )
    else:
        assert expected_identity_count == len(expected_months)
        _assert_exact_partition(
            "credit_account_monthly_performance",
            expected=expected_months,
            emitted=list(months),
            claims=month_claims,
        )
    _assert_completeness(
        "credit_account_monthly_performance",
        datasets["credit_account_monthly_performance"],
        expected_count=expected_identity_count,
        emitted_count=len(months),
        omitted_count=len({claim.identity for claim in month_claims}),
    )

    agreements = _agreement_identities(datasets["credit_agreements"])
    agreement_claims = _exact_omission_claims(
        community,
        dataset_name="credit_agreements",
        parse_identity=_parse_agreement_omission,
        allowed_fields=frozenset({"credit_agreement_id", "credit_line_id", "sequence"}),
    )
    _assert_exact_partition(
        "credit_agreements",
        expected=set(range(1, oracle.agreements + 1)),
        emitted=list(agreements),
        claims=agreement_claims,
    )
    _assert_completeness(
        "credit_agreements",
        datasets["credit_agreements"],
        expected_count=oracle.agreements,
        emitted_count=len(agreements),
        omitted_count=len({claim.identity for claim in agreement_claims}),
    )

    inquiries = _inquiry_identities(datasets["inquiries"])
    inquiry_claims = _exact_omission_claims(
        community,
        dataset_name="inquiries",
        parse_identity=_inquiry_omission_parser(oracle),
        allowed_fields=frozenset({"inquiry_id", "sequence"}),
    )
    _assert_exact_partition(
        "inquiries",
        expected=set(_expected_inquiry_identities(oracle)),
        emitted=list(inquiries),
        claims=inquiry_claims,
    )
    _assert_completeness(
        "inquiries",
        datasets["inquiries"],
        expected_count=oracle.inquiries,
        emitted_count=len(inquiries),
        omitted_count=len({claim.identity for claim in inquiry_claims}),
    )


@pytest.fixture(scope="module")
def expanded_saved_audit_dir() -> Path:
    value = os.environ.get(_SAVED_AUDIT_ENV)
    if not value:
        pytest.skip(f"set {_SAVED_AUDIT_ENV}")
    directory = Path(value)
    assert directory.is_dir(), directory
    return directory


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.tier_slow
def test_saved_expanded_personal_detail_pair_catalog(
    expanded_saved_audit_dir: Path,
) -> None:
    expected = {oracle.stem for oracle in _ORACLES}
    community = {
        path.name.removesuffix(".community.json") for path in expanded_saved_audit_dir.glob("*.community.json")
    }
    semantic = {path.name.removesuffix(".semantic.json") for path in expanded_saved_audit_dir.glob("*.semantic.json")}
    assert community == expected
    assert semantic == expected


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.tier_slow
@pytest.mark.parametrize("oracle", _ORACLES, ids=lambda oracle: oracle.stem)
def test_saved_expanded_personal_detail_population_acceptance(
    expanded_saved_audit_dir: Path,
    oracle: PopulationOracle,
) -> None:
    community_path = expanded_saved_audit_dir / f"{oracle.stem}.community.json"
    semantic_path = expanded_saved_audit_dir / f"{oracle.stem}.semantic.json"
    assert community_path.is_file(), community_path
    assert semantic_path.is_file(), semantic_path
    community = json.loads(community_path.read_text(encoding="utf-8"))
    semantic = json.loads(semantic_path.read_text(encoding="utf-8"))
    _assert_saved_population_oracle(oracle, community, semantic)


def _synthetic_issue_payload(
    *issues: dict,
    evidence_by_issue: dict[str, list[dict]],
    source_refs_by_issue: dict[str, list[dict]] | None = None,
    page_by_issue: dict[str, int] | None = None,
) -> dict:
    source_refs_by_issue = source_refs_by_issue or {}
    page_by_issue = page_by_issue or {}

    def issue_wrapper(issue: dict) -> dict:
        issue_id = str(issue.get("extraction_issue_id") or "")
        page = page_by_issue.get(issue_id, 1)
        source = {"page_range": [page, page]}
        refs = source_refs_by_issue.get(issue_id)
        if refs:
            source["source_refs"] = refs
        return {"normalized": issue, "source": source}

    return {
        "datasets": [
            {
                "name": "extraction_issues",
                "rows": [issue_wrapper(issue) for issue in issues],
            },
            {
                "name": "extraction_issue_evidence",
                "rows": [
                    {
                        "normalized": {
                            "extraction_issue_id": issue_id,
                            "extraction_issue_evidence_id": (
                                f"extraction_issue_evidence:{issue_id}:{index}"
                            ),
                            **evidence,
                        },
                        "source": {
                            "page_range": [
                                page_by_issue.get(issue_id, 1),
                                page_by_issue.get(issue_id, 1),
                            ]
                        },
                    }
                    for issue_id, evidence_rows in sorted(evidence_by_issue.items())
                    for index, evidence in enumerate(evidence_rows, start=1)
                ],
            },
        ]
    }


def _observed_evidence(path: str, value: object) -> dict:
    if type(value) is int:
        return {
            "evidence_kind": "observed",
            "evidence_path": path,
            "value_type": "integer",
            "integer_value": value,
        }
    return {
        "evidence_kind": "observed",
        "evidence_path": path,
        "value_type": "string",
        "string_value": str(value),
    }


def _exact_ref(
    *,
    field_name: str,
    sequence: int,
    binding: str = "canonical_sequence_row",
    page: int = 1,
) -> dict:
    return {
        "logical_page": page,
        "source_page": page,
        "table_id": "pt_1_0",
        "row": sequence,
        "column": 0,
        "bbox": [10.0, 20.0, 30.0, 40.0],
        "geometry_scope": "cell",
        "binding": binding,
        "binding_quality": binding,
        "field_name": field_name,
        "sequence": sequence,
        "evidence_ids": [f"ocr:p{page}:row{sequence}:c0"],
    }


def _source_account_month_ref(
    account_id: str,
    month: str,
    *,
    page: int = 1,
) -> dict:
    return {
        "source": "candidate_b_monthly_anchor_ledger",
        "logical_page": page,
        "source_page": page,
        "bbox": [10.0, 20.0, 300.0, 40.0],
        "geometry_scope": "line",
        "binding": "source_account_month_range",
        "binding_quality": "source_account_month_range",
        "field_name": "performance_month",
        "account_id": account_id,
        "performance_month": month,
        "evidence_ids": [f"ocr:p{page}:monthly-range"],
    }


def _source_account_month_payload(
    *,
    account_id: str = "credit_account:credit_card:3",
    month: str = "2024-03",
    issue_code: str = "candidate_b_monthly_account_range_missing_month",
    field_name: str = "performance_month",
    target: str | None = None,
    evidence: list[dict] | None = None,
    refs: list[dict] | None = None,
) -> dict:
    issue_id = "source-account-month"
    owner_hash = _source_account_month_owner_hash(account_id)
    return _synthetic_issue_payload(
        {
            "extraction_issue_id": issue_id,
            "status": "requires_review",
            "target_dataset": "credit_account_monthly_performance",
            "target_record_id": target
            or f"source_account_month:{owner_hash}:{month}",
            "field_name": field_name,
            "issue_code": issue_code,
        },
        evidence_by_issue={
            issue_id: evidence
            or [
                _observed_evidence("account_id", account_id),
                _observed_evidence("performance_month", month),
            ]
        },
        source_refs_by_issue={
            issue_id: refs or [_source_account_month_ref(account_id, month)]
        },
    )


def test_exact_omission_helper_normalizes_source_account_month_identity() -> None:
    claims = _exact_omission_claims(
        _source_account_month_payload(),
        dataset_name="credit_account_monthly_performance",
        parse_identity=_parse_month_identity,
        allowed_fields=_MONTH_OMISSION_FIELDS,
    )

    assert [(claim.identity, claim.field_name) for claim in claims] == [
        (("credit_account:credit_card:3", "2024-03"), "performance_month")
    ]


def test_exact_omission_helper_accepts_stable_owned_grid_identity() -> None:
    account_id = "credit_account:credit_card:3"
    month = "2024-03"
    grid_id = "grid:owned"
    issue_id = "owned-grid-month"
    owner_hash = _source_account_month_owner_hash(account_id)
    payload = _synthetic_issue_payload(
        {
            "extraction_issue_id": issue_id,
            "status": "requires_review",
            "target_dataset": "credit_account_monthly_performance",
            "target_record_id": f"source_account_month:{owner_hash}:{month}",
            "field_name": "performance_month",
            "issue_code": "candidate_b_monthly_owned_grid_missing_field",
        },
        evidence_by_issue={
            issue_id: [
                _observed_evidence("account_id", account_id),
                _observed_evidence("performance_month", month),
            ]
        },
        source_refs_by_issue={
            issue_id: [
                {
                    "source": "candidate_b_monthly_owned_grid_cell",
                    "logical_page": 1,
                    "source_page": 1,
                    "table_id": "monthly-table:1",
                    "row": 4,
                    "column": 3,
                    "bbox": [10.0, 20.0, 30.0, 40.0],
                    "geometry_scope": "cell",
                    "binding": "source_account_month_identity",
                    "binding_quality": "source_account_month_identity",
                    "account_id": account_id,
                    "field_name": "performance_month",
                    "grid_id": grid_id,
                    "performance_month": month,
                    "evidence_ids": ["ocr:p1:monthly:r4:c3"],
                }
            ]
        },
    )

    claims = _exact_omission_claims(
        payload,
        dataset_name="credit_account_monthly_performance",
        parse_identity=_parse_month_identity,
        allowed_fields=_MONTH_OMISSION_FIELDS,
    )

    assert [(claim.identity, claim.field_name) for claim in claims] == [
        ((account_id, month), "performance_month")
    ]


@pytest.mark.parametrize(
    "mutation",
    (
        "status_diagnostic",
        "malformed_target",
        "wrong_hash",
        "missing_account_evidence",
        "duplicate_account_evidence",
        "wrong_month_evidence",
        "ref_page_mismatch",
        "wrong_ref_account",
        "wrong_ref_month",
        "missing_evidence_ids",
        "duplicate_ref",
        "mixed_ref",
    ),
)
def test_source_account_month_claim_rejects_ambiguous_or_unbound_identity(
    mutation: str,
) -> None:
    account_id = "credit_account:credit_card:3"
    month = "2024-03"
    issue_code = "candidate_b_monthly_account_range_missing_month"
    field_name = "performance_month"
    target = None
    evidence = [
        _observed_evidence("account_id", account_id),
        _observed_evidence("performance_month", month),
    ]
    ref = _source_account_month_ref(account_id, month)
    refs = [ref]
    if mutation == "status_diagnostic":
        issue_code = "candidate_b_monthly_account_range_status_grid_unavailable"
        field_name = "status_code"
        ref["field_name"] = "status_code"
        ref["binding"] = "source_account_month_identity"
        ref["binding_quality"] = "source_account_month_identity"
    elif mutation == "malformed_target":
        target = "source_account_month:123:2024-03"
    elif mutation == "wrong_hash":
        target = "source_account_month:0000000000000000:2024-03"
    elif mutation == "missing_account_evidence":
        evidence.pop(0)
    elif mutation == "duplicate_account_evidence":
        evidence.insert(0, _observed_evidence("account_id", account_id))
    elif mutation == "wrong_month_evidence":
        evidence[-1] = _observed_evidence("performance_month", "2024-02")
    elif mutation == "ref_page_mismatch":
        ref["logical_page"] = 2
    elif mutation == "wrong_ref_account":
        ref["account_id"] = "credit_account:credit_card:2"
    elif mutation == "wrong_ref_month":
        ref["performance_month"] = "2024-02"
    elif mutation == "missing_evidence_ids":
        ref["evidence_ids"] = []
    elif mutation == "duplicate_ref":
        refs.append(dict(ref))
    elif mutation == "mixed_ref":
        refs.append({**ref, "geometry_scope": "cell"})

    claims = _exact_omission_claims(
        _source_account_month_payload(
            issue_code=issue_code,
            field_name=field_name,
            target=target,
            evidence=evidence,
            refs=refs,
        ),
        dataset_name="credit_account_monthly_performance",
        parse_identity=_parse_month_identity,
        allowed_fields=_MONTH_OMISSION_FIELDS,
    )

    assert claims == []


def test_signed_huang_month_identity_ledger_is_hash_and_source_bound() -> None:
    oracle = next(item for item in _ORACLES if item.month_identity_ledger)

    identities = _month_identity_ledger(oracle)

    assert identities is not None
    assert len(identities) == oracle.month_positions


def test_exact_omission_helper_accepts_only_localized_month_fields() -> None:
    payload = _synthetic_issue_payload(
        {
            "extraction_issue_id": "exact-status",
            "status": "requires_review",
            "target_dataset": "credit_account_monthly_performance",
            "target_record_id": "grid-7:2024-02",
            "field_name": "status_code",
            "issue_code": "candidate_b_monthly_grid_owner_unresolved_field",
        },
        {
            "extraction_issue_id": "exact-month",
            "status": "requires_review",
            "target_dataset": "credit_account_monthly_performance",
            "target_record_id": "grid-8:2024-03",
            "field_name": "performance_month",
            "issue_code": "candidate_b_monthly_grid_contract_missing_field",
        },
        {
            "extraction_issue_id": "aggregate-only",
            "status": "requires_review",
            "target_dataset": "credit_account_monthly_performance",
            "target_record_id": "grid-9",
            "field_name": "status_code",
            "issue_code": "candidate_b_monthly_grid_contract_unresolved",
        },
        {
            "extraction_issue_id": "unsupported-field",
            "status": "requires_review",
            "target_dataset": "credit_account_monthly_performance",
            "target_record_id": "grid-10:2024-04",
            "field_name": "status_amount",
            "issue_code": "candidate_b_monthly_grid_owner_unresolved_field",
        },
        {
            "extraction_issue_id": "no-evidence",
            "status": "requires_review",
            "target_dataset": "credit_account_monthly_performance",
            "target_record_id": "grid-11:2024-05",
            "field_name": "status_code",
            "issue_code": "candidate_b_monthly_grid_owner_unresolved_field",
        },
        evidence_by_issue={
            "exact-status": [
                _observed_evidence("grid_id", "grid-7"),
                _observed_evidence("performance_month", "2024-02"),
            ],
            "exact-month": [
                _observed_evidence("grid_id", "grid-8"),
                _observed_evidence("performance_month", "2024-03"),
            ],
            "aggregate-only": [_observed_evidence("grid_id", "grid-9")],
            "unsupported-field": [
                _observed_evidence("grid_id", "grid-10"),
                _observed_evidence("performance_month", "2024-04"),
            ],
        },
        source_refs_by_issue={
            "exact-status": [
                    {
                        **_exact_ref(
                        field_name="status_code",
                        sequence=2,
                        binding="monthly_grid_cell",
                        ),
                        "geometry_scope": "cell",
                        "grid_id": "grid-7",
                        "performance_month": "2024-02",
                    }
            ],
            "exact-month": [
                    {
                        **_exact_ref(
                        field_name="performance_month",
                        sequence=3,
                        binding="monthly_grid_cell",
                        ),
                        "geometry_scope": "cell",
                        "grid_id": "grid-8",
                        "performance_month": "2024-03",
                    }
            ],
            "unsupported-field": [
                    {
                        **_exact_ref(
                        field_name="status_amount",
                        sequence=4,
                        binding="monthly_grid_cell",
                        ),
                        "geometry_scope": "cell",
                        "grid_id": "grid-10",
                        "performance_month": "2024-04",
                    }
            ],
        },
    )
    claims = _exact_omission_claims(
        payload,
        dataset_name="credit_account_monthly_performance",
        parse_identity=_parse_month_identity,
        allowed_fields=_MONTH_OMISSION_FIELDS,
    )
    assert {(claim.identity, claim.field_name) for claim in claims} == {
        ("grid-7:2024-02", "status_code"),
        ("grid-8:2024-03", "performance_month"),
    }


def test_exact_omission_helper_rejects_review_issue_on_emitted_identity() -> None:
    payload = _synthetic_issue_payload(
        {
            "extraction_issue_id": "review-only",
            "issue_code": "candidate_b_inquiry_sequence_inferred_from_row_order",
            "status": "requires_review",
            "target_dataset": "inquiries",
            "target_record_id": "credit_inquiry:institution:2",
            "field_name": "sequence",
        },
        {
            "extraction_issue_id": "real-omission",
            "issue_code": "source_inquiry_record_omitted",
            "status": "requires_review",
            "target_dataset": "inquiries",
            "target_record_id": "credit_inquiry:institution:3",
            "field_name": "inquiry_id",
        },
        evidence_by_issue={
            "review-only": [
                _observed_evidence("inquiry_type", "institution"),
                _observed_evidence("sequence", 2),
            ],
            "real-omission": [
                _observed_evidence("inquiry_type", "institution"),
                _observed_evidence("sequence", 3),
            ],
        },
        source_refs_by_issue={
            "review-only": [_exact_ref(field_name="sequence", sequence=2)],
            "real-omission": [_exact_ref(field_name="inquiry_id", sequence=3)],
        },
    )
    oracle = _ORACLES[0]

    claims = _exact_omission_claims(
        payload,
        dataset_name="inquiries",
        parse_identity=_inquiry_omission_parser(oracle),
        allowed_fields=frozenset({"inquiry_id", "sequence"}),
    )

    assert [(claim.identity, claim.field_name) for claim in claims] == [
        (("institution", 3), "inquiry_id")
    ]


def test_inquiry_omission_parser_accepts_canonical_stable_identity() -> None:
    oracle = _ORACLES[1]
    parse = _inquiry_omission_parser(oracle)
    assert parse(stable_record_id("credit_inquiry", "institution", 143)) == (
        "institution",
        143,
    )
    assert parse(stable_record_id("credit_inquiry", "personal", 6)) == (
        "personal",
        6,
    )
    assert parse(stable_record_id("credit_inquiry", "personal", 7)) is None


def test_population_partition_helper_accepts_exact_emitted_reported_split() -> None:
    claims = [OmissionClaim(("credit_card", 2), "account_id", "issue-2")]
    _assert_exact_partition(
        "synthetic accounts",
        expected={("credit_card", 1), ("credit_card", 2)},
        emitted=[("credit_card", 1)],
        claims=claims,
    )
    _assert_count_partition(
        "synthetic complete months",
        expected_count=2,
        emitted=["grid-1:2024-01", "grid-1:2024-02"],
        claims=[],
    )


def test_count_only_month_partition_rejects_arbitrary_omission_identity() -> None:
    with pytest.raises(AssertionError, match="count-only source total"):
        _assert_count_partition(
            "synthetic months",
            expected_count=2,
            emitted=["grid-1:2024-01"],
            claims=[OmissionClaim("invented-grid:2024-02", "status_code", "month-2")],
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "unrelated_issue_id_only",
        "identity_value_mismatch",
        "page_mismatch",
        "boolean_page",
        "missing_bbox_and_cell",
        "empty_evidence_ids",
        "wrong_field_binding",
    ),
)
def test_exact_omission_helper_rejects_nonlocal_or_unbound_evidence(
    mutation: str,
) -> None:
    issue_id = "omitted-inquiry-3"
    evidence_by_issue = {
        issue_id: [
            _observed_evidence("inquiry_type", "institution"),
            _observed_evidence("sequence", 3),
        ]
    }
    ref = _exact_ref(field_name="inquiry_id", sequence=3)
    page_by_issue = {issue_id: 1}
    if mutation == "unrelated_issue_id_only":
        evidence_by_issue = {
            "some-other-issue": [
                _observed_evidence("inquiry_type", "institution"),
                _observed_evidence("sequence", 3),
            ]
        }
    elif mutation == "identity_value_mismatch":
        evidence_by_issue[issue_id][-1] = _observed_evidence("sequence", 2)
    elif mutation == "page_mismatch":
        ref["logical_page"] = 2
    elif mutation == "boolean_page":
        ref["logical_page"] = True
    elif mutation == "missing_bbox_and_cell":
        ref.pop("bbox")
        ref.pop("table_id")
        ref.pop("row")
        ref.pop("column")
    elif mutation == "empty_evidence_ids":
        ref["evidence_ids"] = []
    elif mutation == "wrong_field_binding":
        ref["field_name"] = "reason"

    payload = _synthetic_issue_payload(
        {
            "extraction_issue_id": issue_id,
            "issue_code": "source_inquiry_record_omitted",
            "status": "requires_review",
            "target_dataset": "inquiries",
            "target_record_id": "credit_inquiry:institution:3",
            "field_name": "inquiry_id",
        },
        evidence_by_issue=evidence_by_issue,
        source_refs_by_issue={issue_id: [ref]},
        page_by_issue=page_by_issue,
    )

    assert not _exact_omission_claims(
        payload,
        dataset_name="inquiries",
        parse_identity=lambda value: _parse_inquiry_omission(value),
        allowed_fields=frozenset({"inquiry_id", "sequence"}),
    )


def test_month_claim_rejects_row_local_but_not_field_local_ref() -> None:
    issue_id = "month-status"
    ref = _exact_ref(field_name="status_code", sequence=2)
    ref.update({"geometry_scope": "row", "binding": "canonical_header_row"})
    payload = _synthetic_issue_payload(
        {
            "extraction_issue_id": issue_id,
            "issue_code": "canonical_monthly_source_structure_missing_field",
            "status": "requires_review",
            "target_dataset": "credit_account_monthly_performance",
            "target_record_id": "grid-7:2024-02",
            "field_name": "status_code",
        },
        evidence_by_issue={
            issue_id: [
                _observed_evidence("grid_id", "grid-7"),
                _observed_evidence("performance_month", "2024-02"),
            ]
        },
        source_refs_by_issue={issue_id: [ref]},
    )

    assert not _exact_omission_claims(
        payload,
        dataset_name="credit_account_monthly_performance",
        parse_identity=_parse_month_identity,
        allowed_fields=_MONTH_OMISSION_FIELDS,
    )


@pytest.mark.parametrize(
    ("emitted", "claims", "message"),
    (
        ([1, 1], [], "duplicate emitted"),
        ([1], [], "silent omissions"),
        ([1, 3], [], "extra emitted"),
        ([1], [OmissionClaim(1, "sequence", "issue")], "both emitted"),
        (
            [1],
            [
                OmissionClaim(2, "sequence", "issue-a"),
                OmissionClaim(2, "sequence", "issue-b"),
            ],
            "duplicate omission findings",
        ),
    ),
)
def test_population_partition_helper_rejects_nonconservation(
    emitted: list[int], claims: list[OmissionClaim], message: str
) -> None:
    with pytest.raises(AssertionError, match=message):
        _assert_exact_partition(
            "synthetic",
            expected={1, 2},
            emitted=emitted,
            claims=claims,
        )
