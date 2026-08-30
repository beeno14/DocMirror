"""Source-authored Personal Brief goldens, independent of the product parser.

Facts count explicit source values once. Exact row contracts additionally check
inferences, duplicates, absence, types, ownership and unanticipated output. No
DocMirror extraction, normalization or audit function is used as an oracle.
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Any

_MISSING = {"$missing": True}
_MONEY = {
    "balance", "credit_limit", "used_amount", "loan_amount",
    "unbilled_installment_balance", "responsibility_amount",
    "received_debt_amount", "cumulative_compensation_amount",
    "current_arrears_amount", "arrears_amount", "claim_amount",
    "requested_amount", "executed_amount", "penalty_amount",
}


def same_value(expected: Any, actual: Any, field: str) -> bool:
    if actual == _MISSING:
        return False
    if isinstance(expected, bool):
        return type(actual) is bool and expected == actual
    if field in _MONEY:
        try:
            return not isinstance(actual, bool) and Decimal(str(expected)) == Decimal(str(actual))
        except (InvalidOperation, ValueError):
            return False
    if type(expected) is int:
        return type(actual) is int and expected == actual
    if field == "report_time":
        try:
            values = [datetime.fromisoformat(str(value)) for value in (expected, actual)]
            china = timezone(timedelta(hours=8))
            return values[0].replace(tzinfo=values[0].tzinfo or china) == values[1].replace(
                tzinfo=values[1].tzinfo or china
            )
        except ValueError:
            return False
    if isinstance(expected, str):
        if not isinstance(actual, str):
            return False
        def compact(value):
            return (
                re.sub(r"\s+", "", value)
                .translate(str.maketrans("（）", "()"))
                .removesuffix("。")
            )
        return compact(expected) == compact(actual)
    return type(expected) is type(actual) and expected == actual


def audit_business_truth(standard: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Audit a persisted public payload against an immutable source standard."""

    issues = []
    datasets = {}
    for di, dataset in enumerate(payload.get("datasets", [])):
        name = dataset.get("name")
        if name in datasets:
            issues.append({"kind": "duplicate_dataset", "dataset": name})
        datasets[name] = (di, dataset.get("rows", []))
    sections = {}
    for si, section in enumerate(payload.get("sections", [])):
        name = section.get("type")
        if name in sections:
            issues.append({"kind": "duplicate_section", "section": name})
        sections[name] = (si, section.get("items", []))

    def actual_value(target):
        if "dataset" in target:
            di, rows = datasets.get(target["dataset"], (-1, []))
            index, field = target["row"], target["field"]
            pointer = f"/datasets/{di}/rows/{index}/normalized/{field}"
            value = rows[index].get("normalized", {}).get(field, _MISSING) if index < len(rows) else _MISSING
            return value, pointer
        if "section_property" in target:
            si, _items = sections.get(target["section"], (-1, []))
            section = payload.get("sections", [])[si] if si >= 0 else {}
            field = target["section_property"]
            return section.get(field, _MISSING), f"/sections/{si}/{field}"
        si, items = sections.get(target["section"], (-1, []))
        matches = [(i, item) for i, item in enumerate(items) if item.get("key") == target["field"]]
        if len(matches) != 1:
            return _MISSING, f"/sections/{si}/items/{target['field']}"
        ii, item = matches[0]
        return item.get("value", _MISSING), f"/sections/{si}/items/{ii}/value"

    def check(target, expected):
        actual, pointer = actual_value(target)
        if isinstance(expected, dict) and "$account_ref" in expected:
            account_id, _ = actual_value({"dataset": "credit_accounts", "row": expected["$account_ref"], "field": "account_id"})
            correct = isinstance(actual, str) and bool(actual) and actual == account_id
        else:
            correct = same_value(
                expected,
                actual,
                target.get("field", target.get("section_property", "")),
            )
        return {"json_pointer": pointer, "expected": expected, "actual": actual, "correct": correct}

    ledger = []
    for fact in standard["facts"]:
        checks = [check(target, target["expected"]) for target in fact["checks"]]
        correct = bool(checks) and all(value["correct"] for value in checks)
        ledger.append({**{k: v for k, v in fact.items() if k != "checks"}, "checks": checks,
                       "verdict": "correct" if correct else "incorrect_or_missing"})

    reviews = []
    if set(datasets) != set(standard["datasets"]):
        issues.append({"kind": "dataset_set", "expected": sorted(standard["datasets"]), "actual": sorted(datasets)})
    envelope_ids = []
    account_ids = []
    for name, expected_rows in standard["datasets"].items():
        di, rows = datasets.get(name, (-1, []))
        if len(rows) != len(expected_rows):
            issues.append({"kind": "row_count", "dataset": name, "expected": len(expected_rows), "actual": len(rows)})
        for ri, expected in enumerate(expected_rows):
            envelope = rows[ri] if ri < len(rows) else {}
            values = envelope.get("normalized", {})
            pointer = f"/datasets/{di}/rows/{ri}"
            if set(envelope) != {"record_id", "normalized"}:
                issues.append({"kind": "row_envelope", "json_pointer": pointer, "actual": sorted(envelope)})
            envelope_ids.append(envelope.get("record_id"))
            if name == "credit_accounts":
                account_ids.append(values.get("account_id"))
            for field, value in expected.items():
                reviews.append(check({"dataset": name, "row": ri, "field": field}, value))
            for field in sorted(set(values) - set(expected)):
                issues.append({"kind": "unsupported_field", "json_pointer": pointer + "/normalized/" + field,
                               "actual": values[field]})
    for kind, identifiers in [("record_id", envelope_ids), ("account_id", account_ids)]:
        if any(not isinstance(value, str) or not value for value in identifiers) or len(set(identifiers)) != len(identifiers):
            issues.append({"kind": "invalid_or_duplicate_identifier", "field": kind})

    if set(sections) != set(standard["sections"]):
        issues.append({"kind": "section_set", "expected": sorted(standard["sections"]), "actual": sorted(sections)})
    for name, expected_items in standard["sections"].items():
        si, items = sections.get(name, (-1, []))
        keys = [item.get("key") for item in items]
        if Counter(keys) != Counter(expected_items.keys()):
            issues.append({"kind": "section_item_set", "section": name, "expected": list(expected_items), "actual": keys})
        for field, value in expected_items.items():
            reviews.append(check({"section": name, "field": field}, value))

    if payload.get("schema", {}).get("version") != "4.0.0" or payload.get("document", {}).get("type") != "personal_credit_report_brief":
        issues.append({"kind": "wrong_public_contract"})
    failures = [value for value in reviews if not value["correct"]]
    correct = sum(fact["verdict"] == "correct" for fact in ledger)
    return {
        "case_id": standard["case_id"], "source_facts": len(ledger), "correct": correct,
        "correctness_percent": round(100 * correct / len(ledger), 6) if ledger else 0,
        "passed": bool(ledger) and correct == len(ledger) and not failures and not issues,
        "business_values_reviewed": len(reviews), "value_failures": failures,
        "structure_issues": issues, "field_audit": ledger, "output_field_review": reviews,
        "product_audit_used_as_oracle": False,
    }
