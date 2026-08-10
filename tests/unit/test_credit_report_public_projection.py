# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pytest

from docmirror.plugins.credit_report.projection import (
    _compact_public_datasets,
    _source_page_range,
)


def _private_keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {
                "bbox",
                "evidence_ids",
                "node_id",
                "node_ids",
                "source_anchor",
                "source_cell_refs",
                "source_fact_ids",
                "source_refs",
            }:
                found.add(key)
            found.update(_private_keys(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_private_keys(item))
    return found


@pytest.mark.parametrize(
    ("dataset_name", "record", "expected_page_range"),
    [
        (
            "enterprise_credit_accounts",
            {
                "record_id": "account:1",
                "account_id": "account:1",
                "institution": "示例银行",
                "confidence": 0.98,
                "source_refs": [
                    {
                        "source": "native_table",
                        "page": 3,
                        "table_id": "table:p3:2",
                        "node_id": "node:p3:table:2",
                        "bbox": [10, 20, 30, 40],
                        "evidence_ids": ["ev:3:1"],
                    }
                ],
                "evidence_ids": ["ev:3:1"],
            },
            [3, 3],
        ),
        (
            "personal_report_metadata",
            {
                "record_id": "metadata:1",
                "report_number": "202608040001",
                "source": {
                    "source_refs": [
                        {
                            "source": "native_text_header",
                            "page": "1",
                            "bbox": [1, 2, 3, 4],
                            "evidence_ids": ["ev:1:1"],
                        }
                    ],
                    "evidence_ids": ["ev:1:1"],
                },
            },
            [1, 1],
        ),
        (
            "repayment_records",
            {
                "record_id": "repayment:1",
                "account_id": "account:1",
                "month": "2026-01",
                "status": "N",
                "source_cell_refs": [
                    {
                        "page_id": "p8",
                        "cell_id": "cell:p8:4:2",
                        "node_ids": ["node:p8:grid:4"],
                        "bbox": [11, 12, 13, 14],
                        "evidence_ids": ["ev:8:2"],
                    },
                    {"page": 9, "evidence_ids": ["ev:9:2"]},
                ],
                "source_anchor": {
                    "page": 8,
                    "line_ids": ["line:p8:12"],
                },
                "evidence_ids": ["ev:8:2", "ev:9:2"],
            },
            [8, 9],
        ),
    ],
)
def test_public_credit_datasets_hide_private_provenance(
    dataset_name: str,
    record: dict[str, Any],
    expected_page_range: list[int],
) -> None:
    original_private_keys = _private_keys(record)

    compacted = _compact_public_datasets({dataset_name: [record]})
    public_record = compacted[dataset_name][0]

    assert original_private_keys
    assert _private_keys(record) == original_private_keys
    assert _private_keys(public_record) == set()
    assert public_record["source"] == {"page_range": expected_page_range}
    assert public_record["record_id"] == record["record_id"]
    for business_key in ("account_id", "institution", "report_number", "month", "status"):
        if business_key in record:
            assert public_record[business_key] == record[business_key]


def test_public_credit_datasets_keep_required_empty_source_object() -> None:
    compacted = _compact_public_datasets(
        {"report_notes": [{"record_id": "note:1", "text": "无来源页的业务说明"}]}
    )

    assert compacted["report_notes"][0] == {
        "record_id": "note:1",
        "text": "无来源页的业务说明",
        "source": {},
    }


def test_source_page_range_prefers_logical_page_within_each_ref() -> None:
    assert _source_page_range({"logical_page": 19, "source_page": 10}) == [19, 19]
    assert _source_page_range(
        {
            "source_refs": [
                {
                    "logical_page": 19,
                    "page": 19,
                    "page_id": "p19",
                    "page_number": 19,
                    "source_page": 10,
                }
            ]
        }
    ) == [19, 19]


def test_source_page_range_keeps_physical_source_page_as_fallback() -> None:
    assert _source_page_range(
        {
            "source_refs": [
                {
                    "logical_page": None,
                    "page": "invalid",
                    "page_id": None,
                    "page_number": 0,
                    "source_page": "p10",
                }
            ]
        }
    ) == [10, 10]


def test_source_page_range_traverses_nested_refs_and_explicit_range() -> None:
    assert _source_page_range(
        {
            "source": {
                "page_range": [4, "p6"],
                "provenance": {
                    "refs": [
                        {"logical_page": 3, "source_page": 2},
                        {"page_id": "p7", "source_page": 4},
                    ]
                },
            }
        }
    ) == [3, 7]


def test_source_page_range_preserves_default_page_key_behavior() -> None:
    assert _source_page_range(
        {
            "source_refs": [
                {"page": 8, "page_id": "p9", "page_number": 10, "source_page": 7}
            ]
        }
    ) == [8, 8]
    assert _source_page_range(
        {
            "source_cell_refs": [
                {"page": 8},
                {"page_id": "p9"},
                {"page_number": 10},
            ],
            "source_anchor": {"page": 8},
        }
    ) == [8, 10]
