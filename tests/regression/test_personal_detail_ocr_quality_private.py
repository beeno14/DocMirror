# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import math
import os
import re
from collections import Counter
from copy import deepcopy
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pytest

from docmirror.input.entry.factory import PerceiveOptions, perceive_document
from docmirror.input.entry.options import normalize_parse_policy
from docmirror.models.schemas.registry import validate_projection_payload
from docmirror.plugins.credit_report.community_plugin import CreditReportPlugin
from docmirror.plugins.credit_report.personal_detail_scanned.context import (
    build_personal_detail_extraction_context,
)
from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
    collect_extraction_issues,
)
from docmirror.plugins.credit_report.personal_detail_scanned.schema import (
    PBOC_DATASET_ORDER,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.tier_slow]

_FIXTURE_DIR = Path(
    os.environ.get(
        "DOCMIRROR_PERSONAL_DETAIL_FIXTURE_DIR",
        "tests/fixtures-private/credit_report/Scanned Personal Detailed",
    )
)
_FIXTURES = sorted(_FIXTURE_DIR.glob("*.pdf"))
_EXPECTED_SCHEMA_INPUT_COUNTS = {
    "余泽熙7.15征信.pdf": (27, 641),
    "杨松林个人征信24.7.29.pdf": (38, 615),
    # Source-structure audit proves 40 printed repayment grids: 39 exact
    # physical-table date ranges contribute 942 months and the separate
    # 2023-06..2024-02 block contributes 9.  The former 884 value measured a
    # retired cell-crop coverage plane, not the document population.
    "叶永燕征信.pdf": (42, 951),
    "征信.pdf": (43, 408),
    # Source-page audit confirms 45 business accounts.  In addition to the
    # three responsibility-table false cards removed from the former 48-row
    # oracle, logical page 12 proves that D10053310... is the sole type-R2
    # account: the former 46-row result emitted its table again as an R1 row.
    # A source-grid audit counts all 40 printed repayment grids and their
    # bounded date ranges: 944 printed month positions.  The former 801 oracle
    # omitted valid grids and could make a silent population loss look healthy.
    "林岚挺征信.pdf": (45, 944),
    "洪晓鑫征信报告2025.11.05.pdf": (8, 176),
    "王根镇征信.pdf": (61, 757),
}
_EXPECTED_AGREEMENT_COUNTS = {
    "叶永燕征信.pdf": 16,
    "林岚挺征信.pdf": 11,
    "余泽熙7.15征信.pdf": 8,
    "杨松林个人征信24.7.29.pdf": 7,
    "洪晓鑫征信报告2025.11.05.pdf": 7,
}
_EXPECTED_INQUIRY_COUNTS = {
    "叶永燕征信.pdf": 112,
    "林岚挺征信.pdf": 90,
    "余泽熙7.15征信.pdf": 26,
    "杨松林个人征信24.7.29.pdf": 117,
    "洪晓鑫征信报告2025.11.05.pdf": 20,
}

_LIN_EXPECTED_ACCOUNTS = 45
_LIN_EXPECTED_SOURCE_VETTED_ACCOUNT_CARDS = 42
_LIN_EXPECTED_SOURCE_VETTED_ACCOUNT_TYPES = {
    "non_revolving_loan": 22,
    "revolving_loan_account": 1,
    "credit_card": 19,
}
_LIN_EXPECTED_MONTH_POSITIONS = 944
_LIN_EXPECTED_INQUIRIES = 90
_LIN_EXPECTED_LIABILITIES = 3
_YE_RECOVERED_CARD_IDS = {
    "B10911000H000115603050013394541",
    "B11313900H000115603090424251222",
    "D10123910H000115604050032149",
    "B10411000H000115602800002159651279117266",
    "B10611000H00016226880219191368607",
    "B11911000H000115661000042356833",
}
_LIN_ACCOUNT_REQUIRED_FIELDS = {
    "management_institution": ("management_institution", {"management_institution"}),
    "account_identifier": ("account_identifier", {"account_identifier"}),
    "open_date": ("open_date", {"open_date"}),
    "currency": ("account_currency", {"currency", "account_currency"}),
}
_LIN_ACCOUNT_HEADER_LABELS = frozenset({"账户标识", "开立日期", "币种"})
_LIN_ACCOUNT_INSTITUTION_LABELS = frozenset({"管理机构", "发卡机构"})
_LIN_ACCOUNT_BUSINESS_TYPES = (
    "个人住房商业贷款",
    "其他个人消费贷款",
    "融资租赁业务",
    "个人经营性贷款",
    "个人汽车消费贷款",
    "其他贷款",
    "大额专项分期卡",
    "贷记卡",
)
_LIN_ACCOUNT_GUARANTEE_TYPES = ("信用/免担保", "抵押", "保证")
_LIN_MONTH_GEOMETRY_GUARDS = {
    "mg_p5_repayment_2:2022-06": {"status", "overdue_amount"},
    "mg_p16_repayment_1:2019-06": {"status", "overdue_amount"},
    "mg_p16_repayment_1:2021-11": {"status", "overdue_amount"},
}


def _dataset_map(payload: dict) -> dict[str, dict]:
    return {
        str(dataset.get("name") or ""): dataset
        for dataset in payload.get("datasets") or ()
        if isinstance(dataset, dict) and dataset.get("name")
    }


def _compact_source_table(table: dict) -> str:
    return re.sub(
        r"\s+",
        "",
        "".join(
            str(cell or "")
            for row in (table.get("rows") or ())[:4]
            if isinstance(row, list)
            for cell in row
        ),
    )


def _unique_visible_finite_value(source_texts: list[str], values: tuple[str, ...]) -> str | None:
    observed = {
        candidate
        for candidate in values
        if any(re.sub(r"\s+", "", candidate) in source_text for source_text in source_texts)
    }
    if not observed:
        return None
    longest = max(len(value) for value in observed)
    maximal = {value for value in observed if len(value) == longest}
    return next(iter(maximal)) if len(maximal) == 1 else None


def _assert_lin_semantic_account_oracle(semantic: dict, community: dict) -> None:
    """Audit saved/live Lin output against source-vetted canonical card evidence."""

    semantic_datasets = _dataset_map(semantic)
    community_datasets = _dataset_map(community)
    required_dataset_names = {
        "credit_accounts",
        "credit_account_monthly_performance",
        "repayment_responsibilities",
        "inquiries",
        "extraction_issues",
    }
    assert required_dataset_names <= set(semantic_datasets)
    assert required_dataset_names <= set(community_datasets)

    defects: list[str] = []
    for dataset_name in required_dataset_names - {"extraction_issues"}:
        semantic_records = {
            str(row.get("record_id") or "")
            for row in semantic_datasets[dataset_name].get("rows") or ()
        }
        community_records = {
            str(row.get("record_id") or "")
            for row in community_datasets[dataset_name].get("rows") or ()
        }
        if semantic_records != community_records:
            defects.append(f"{dataset_name}: semantic/community record identities diverged")

    account_wrappers = community_datasets["credit_accounts"].get("rows") or []
    accounts = {
        str(wrapper.get("record_id") or ""): wrapper.get("normalized") or {}
        for wrapper in account_wrappers
    }
    if len(accounts) != _LIN_EXPECTED_ACCOUNTS:
        defects.append(
            f"credit_accounts: expected {_LIN_EXPECTED_ACCOUNTS}, observed {len(accounts)}"
        )

    issue_rows = [
        wrapper.get("normalized") or {}
        for wrapper in community_datasets["extraction_issues"].get("rows") or []
    ]

    def has_actionable_field_issue(record_id: str, field_names: set[str]) -> bool:
        return any(
            row.get("target_dataset") == "credit_accounts"
            and str(row.get("target_record_id") or "") == record_id
            and str(row.get("field_name") or "") in field_names
            and bool(row.get("issue_code"))
            and str(row.get("status") or "requires_review")
            not in {"resolved", "suppressed_redundant", "informational"}
            for row in issue_rows
        )

    source_tables = {
        str(table.get("id") or ""): table
        for table in (semantic.get("structure") or {}).get("source_tables") or ()
        if isinstance(table, dict) and table.get("id")
    }
    account_dataset_id = str(semantic_datasets["credit_accounts"].get("id") or "")
    source_texts_by_record: dict[str, list[str]] = {}
    for binding in semantic.get("bindings") or ():
        if not isinstance(binding, dict) or binding.get("dataset_id") != account_dataset_id:
            continue
        record_id = str(binding.get("record_id") or "")
        source_texts = []
        for table_id in binding.get("source_table_refs") or ():
            table = source_tables.get(str(table_id))
            if table is None:
                continue
            compact = _compact_source_table(table)
            if not all(label in compact for label in _LIN_ACCOUNT_HEADER_LABELS):
                continue
            if not any(label in compact for label in _LIN_ACCOUNT_INSTITUTION_LABELS):
                continue
            source_texts.append(compact)
        if source_texts:
            source_texts_by_record[record_id] = source_texts

    if len(source_texts_by_record) != _LIN_EXPECTED_SOURCE_VETTED_ACCOUNT_CARDS:
        defects.append(
            "source-vetted account cards: expected "
            f"{_LIN_EXPECTED_SOURCE_VETTED_ACCOUNT_CARDS}, observed "
            f"{len(source_texts_by_record)}"
        )
    source_vetted_account_types = Counter(
        str((accounts.get(record_id) or {}).get("account_type") or "")
        for record_id in source_texts_by_record
    )
    if source_vetted_account_types != Counter(
        _LIN_EXPECTED_SOURCE_VETTED_ACCOUNT_TYPES
    ):
        defects.append(
            "source-vetted account cards: expected family distribution "
            f"{_LIN_EXPECTED_SOURCE_VETTED_ACCOUNT_TYPES}, observed "
            f"{dict(source_vetted_account_types)}"
        )

    for record_id, source_texts in sorted(source_texts_by_record.items()):
        account = accounts.get(record_id)
        if account is None:
            defects.append(f"{record_id}: source-vetted account card was silently omitted")
            continue
        for canonical_field, (output_field, issue_fields) in _LIN_ACCOUNT_REQUIRED_FIELDS.items():
            if account.get(output_field) not in (None, ""):
                continue
            if not has_actionable_field_issue(record_id, issue_fields):
                defects.append(
                    f"{record_id}.{canonical_field}: visible canonical slot missing without "
                    "a field-local actionable issue"
                )

        for field_name, vocabulary in (
            ("business_type", _LIN_ACCOUNT_BUSINESS_TYPES),
            ("guarantee_type", _LIN_ACCOUNT_GUARANTEE_TYPES),
        ):
            expected = _unique_visible_finite_value(source_texts, vocabulary)
            if expected is None or account.get(field_name) == expected:
                continue
            if not has_actionable_field_issue(record_id, {field_name}):
                defects.append(
                    f"{record_id}.{field_name}: source visibly contains {expected!r}, "
                    f"published {account.get(field_name)!r} without a field-local issue"
                )

    table_missing_record_ids = {
        str(row.get("target_record_id") or "")
        for row in issue_rows
        if row.get("target_dataset") == "credit_accounts"
        and row.get("issue_code") == "candidate_b_account_table_missing"
    }
    for record_id in sorted(table_missing_record_ids):
        account = accounts.get(record_id)
        if account is None:
            defects.append(f"{record_id}: account_table_missing target has no retained account row")
            continue
        for canonical_field, (output_field, issue_fields) in _LIN_ACCOUNT_REQUIRED_FIELDS.items():
            if account.get(output_field) not in (None, ""):
                continue
            if not has_actionable_field_issue(record_id, issue_fields):
                defects.append(
                    f"{record_id}.{canonical_field}: aggregate account_table_missing is not "
                    "a field-local report"
                )

    forbidden_account_keys = {
        "content",
        "fields",
        "fragments",
        "raw_detail_lines",
        "raw_detail_text",
        "value",
        "values",
    }
    canonical_labels = (
        "管理机构",
        "发卡机构",
        "账户标识",
        "开立日期",
        "到期日期",
        "业务种类",
        "担保方式",
        "还款频率",
        "还款方式",
    )
    for record_id, account in accounts.items():
        leaked_keys = forbidden_account_keys & set(account)
        if leaked_keys:
            defects.append(
                f"{record_id}: unstructured account keys published {sorted(leaked_keys)}"
            )
        for field_name, value in account.items():
            if not isinstance(value, str):
                continue
            visible_labels = [label for label in canonical_labels if label in value]
            if len(visible_labels) >= 2:
                defects.append(
                    f"{record_id}.{field_name}: unstructured multi-field blob contains "
                    f"{visible_labels}"
                )

    monthly_rows = community_datasets["credit_account_monthly_performance"].get("rows") or []
    monthly_values = [wrapper.get("normalized") or {} for wrapper in monthly_rows]
    account_ids = set(accounts)
    if len(monthly_values) > _LIN_EXPECTED_MONTH_POSITIONS:
        defects.append(
            "credit_account_monthly_performance: emitted more rows than the canonical "
            f"{_LIN_EXPECTED_MONTH_POSITIONS} source positions"
        )
    monthly_keys = {
        (str(row.get("grid_id") or ""), str(row.get("performance_month") or ""))
        for row in monthly_values
    }
    if len(monthly_keys) != len(monthly_values):
        defects.append("credit_account_monthly_performance: duplicate grid/month rows")
    if any(row.get("account_id") not in account_ids for row in monthly_values):
        defects.append("credit_account_monthly_performance: rows reference unknown accounts")
    if len(monthly_values) < _LIN_EXPECTED_MONTH_POSITIONS and not any(
        row.get("target_dataset") == "credit_account_monthly_performance"
        and row.get("issue_code")
        in {
            "canonical_monthly_reconstruction_incomplete",
            "candidate_b_monthly_status_grid_unresolved",
            "monthly_linkage_collision_from_account_gap",
            "monthly_population_incomplete_from_account_gap",
        }
        for row in issue_rows
    ):
        defects.append("credit_account_monthly_performance: source shortfall is silent")

    inquiry_rows = community_datasets["inquiries"].get("rows") or []
    inquiry_values = [wrapper.get("normalized") or {} for wrapper in inquiry_rows]
    if len(inquiry_values) > _LIN_EXPECTED_INQUIRIES:
        defects.append(
            f"inquiries: expected at most {_LIN_EXPECTED_INQUIRIES}, observed "
            f"{len(inquiry_values)}"
        )
    if len(inquiry_values) < _LIN_EXPECTED_INQUIRIES and not any(
        row.get("target_dataset") == "inquiries"
        and row.get("issue_code")
        in {"canonical_inquiry_sequence_gap", "source_sequence_or_count_gap"}
        for row in issue_rows
    ):
        defects.append("inquiries: source shortfall is silent")
    inquiry_keys = {
        (
            str(row.get("query_channel") or row.get("inquiry_type") or ""),
            row.get("sequence"),
        )
        for row in inquiry_values
    }
    if len(inquiry_keys) != len(inquiry_values):
        defects.append("inquiries: duplicate channel/sequence rows")

    liability_rows = community_datasets["repayment_responsibilities"].get("rows") or []
    if len(liability_rows) != _LIN_EXPECTED_LIABILITIES:
        defects.append(
            f"repayment_responsibilities: expected {_LIN_EXPECTED_LIABILITIES}, "
            f"observed {len(liability_rows)}"
        )
    liability_ids = {str(row.get("record_id") or "") for row in liability_rows}
    if len(liability_ids) != len(liability_rows) or "" in liability_ids:
        defects.append("repayment_responsibilities: record identities are not conserved")

    assert not defects, "Lin semantic oracle failures:\n- " + "\n- ".join(defects)


def _assert_lin_february_2020_status_oracle(community: dict) -> None:
    """Require the source-vetted terminal status to be correct or explicitly reported."""

    datasets = _dataset_map(community)
    target_record_id = "mg_p8_repayment_1:2020-02"
    monthly_rows = datasets["credit_account_monthly_performance"].get("rows") or []
    target_rows = [
        wrapper
        for wrapper in monthly_rows
        if str(wrapper.get("record_id") or "") == target_record_id
        or (
            (wrapper.get("normalized") or {}).get("grid_id") == "mg_p8_repayment_1"
            and (wrapper.get("normalized") or {}).get("performance_month") == "2020-02"
        )
    ]
    assert len(target_rows) <= 1

    active_status_issues = [
        wrapper.get("normalized") or {}
        for wrapper in datasets["extraction_issues"].get("rows") or []
        if (wrapper.get("normalized") or {}).get("target_dataset")
        == "credit_account_monthly_performance"
        and str((wrapper.get("normalized") or {}).get("target_record_id") or "")
        == target_record_id
        and (wrapper.get("normalized") or {}).get("field_name") == "status_code"
        and bool((wrapper.get("normalized") or {}).get("issue_code"))
        and str((wrapper.get("normalized") or {}).get("status") or "requires_review")
        not in {"resolved", "suppressed_redundant", "informational"}
    ]
    status_code = (
        (target_rows[0].get("normalized") or {}).get("status_code")
        if target_rows
        else None
    )
    if status_code != "C" and not active_status_issues:
        raise AssertionError(
            f"{target_record_id}: source table pt_8_2 shows C, but Community published "
            f"{status_code!r} without an exact active status_code issue"
        )


def _numeric_bbox(value: object) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        bbox = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in bbox):
        return None
    x0, y0, x1, y1 = bbox
    return bbox if x1 > x0 and y1 > y0 else None


def _year_plus_twelve_month_bands(
    source_table: dict,
) -> tuple[tuple[float, float], ...] | None:
    """Return physical year+12 column bands only when source geometry proves them."""

    extensions = source_table.get("extensions") or {}
    geometry = extensions.get("geometry") or {}
    raw_bands = geometry.get("col_bands") or []
    if len(raw_bands) != 13:
        return None
    try:
        ordered_bands = sorted(raw_bands, key=lambda band: int(band["index"]))
        if [int(band["index"]) for band in ordered_bands] != list(range(13)):
            return None
        bands = tuple((float(band["x0"]), float(band["x1"])) for band in ordered_bands)
    except (KeyError, TypeError, ValueError):
        return None
    if not all(
        math.isfinite(x0) and math.isfinite(x1) and x1 > x0
        for x0, x1 in bands
    ):
        return None

    raw_rows = extensions.get("raw_rows") or source_table.get("rows") or []
    if not any(
        isinstance(row, list)
        and len(row) >= 13
        and re.search(r"(?:19|20)\d{2}", str(row[0] or ""))
        for row in raw_rows
    ):
        return None
    return bands


def _dominant_physical_column(
    bbox: tuple[float, float, float, float],
    bands: tuple[tuple[float, float], ...],
) -> int | None:
    """Resolve a cell bbox to its unique maximum-overlap physical column."""

    x0, _, x1, _ = bbox
    overlaps = [max(0.0, min(x1, band_x1) - max(x0, band_x0)) for band_x0, band_x1 in bands]
    maximum = max(overlaps, default=0.0)
    if maximum <= 0.0:
        return None
    owners = [
        index
        for index, overlap in enumerate(overlaps)
        if math.isclose(overlap, maximum, abs_tol=1e-6)
    ]
    return owners[0] if len(owners) == 1 else None


def _lin_month_ref_physical_ownership(
    semantic: dict,
) -> tuple[list[str], set[tuple[str, str]]]:
    """Compare exact monthly refs only to unambiguous year+12 source tables."""

    tables_by_page: dict[
        int,
        list[tuple[str, tuple[float, float, float, float], tuple[tuple[float, float], ...]]],
    ] = {}
    for source_table in (semantic.get("structure") or {}).get("source_tables") or []:
        if not isinstance(source_table, dict):
            continue
        bands = _year_plus_twelve_month_bands(source_table)
        table_bbox = _numeric_bbox(source_table.get("bbox"))
        if bands is None or table_bbox is None:
            continue
        page = int(source_table.get("page") or 0)
        if page <= 0:
            continue
        tables_by_page.setdefault(page, []).append(
            (str(source_table.get("id") or ""), table_bbox, bands)
        )

    datasets = _dataset_map(semantic)
    monthly_rows = datasets["credit_account_monthly_performance"].get("rows") or []
    defects: list[str] = []
    compared_refs: set[tuple[str, str]] = set()
    exact_field_names = {"status", "status_amount", "overdue_amount"}
    for wrapper in monthly_rows:
        values = wrapper.get("normalized") or {}
        record_id = str(
            wrapper.get("record_id") or values.get("monthly_performance_id") or ""
        )
        month_match = re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", str(values.get("performance_month") or ""))
        if not record_id or month_match is None:
            continue
        expected_column = int(month_match.group(1))
        source = wrapper.get("source") or {}
        for ref in source.get("source_cell_refs") or []:
            if (
                not isinstance(ref, dict)
                or ref.get("field_name") not in exact_field_names
                or ref.get("geometry_scope") != "cell"
                or ref.get("coordinate_system") != "pdf_points_top_left"
            ):
                continue
            bbox = _numeric_bbox(ref.get("bbox"))
            if bbox is None:
                continue
            page = int(ref.get("logical_page") or ref.get("page") or 0)
            center_x = (bbox[0] + bbox[2]) / 2.0
            center_y = (bbox[1] + bbox[3]) / 2.0
            comparable_tables = [
                (table_id, bands)
                for table_id, table_bbox, bands in tables_by_page.get(page, [])
                if table_bbox[0] - 1.0 <= center_x <= table_bbox[2] + 1.0
                and table_bbox[1] - 1.0 <= center_y <= table_bbox[3] + 1.0
            ]
            if len(comparable_tables) != 1:
                continue
            table_id, bands = comparable_tables[0]
            field_name = str(ref["field_name"])
            compared_refs.add((record_id, field_name))
            observed_column = _dominant_physical_column(bbox, bands)
            if observed_column == expected_column:
                continue
            observed_label = "ambiguous" if observed_column is None else str(observed_column)
            defects.append(
                f"{record_id}.{field_name}: {table_id} exact bbox belongs to physical "
                f"column {observed_label}, expected month column {expected_column}"
            )
    return defects, compared_refs


def _assert_lin_month_ref_physical_ownership_oracle(semantic: dict) -> None:
    defects, compared_refs = _lin_month_ref_physical_ownership(semantic)
    missing_guards = sorted(
        (record_id, field_name)
        for record_id, field_names in _LIN_MONTH_GEOMETRY_GUARDS.items()
        for field_name in field_names
        if (record_id, field_name) not in compared_refs
    )
    if missing_guards:
        defects.append(
            "source-specific year+12 geometry guards were not compared: "
            + ", ".join(f"{record_id}.{field_name}" for record_id, field_name in missing_guards)
        )
    if defects:
        raise AssertionError(
            "Lin physical month-column ownership failures:\n- " + "\n- ".join(defects)
        )


def _active_monthly_field_issues(
    community: dict,
    record_id: str,
    field_name: str,
) -> list[dict]:
    datasets = _dataset_map(community)
    return [
        wrapper.get("normalized") or {}
        for wrapper in datasets["extraction_issues"].get("rows") or []
        if (wrapper.get("normalized") or {}).get("target_dataset")
        == "credit_account_monthly_performance"
        and str((wrapper.get("normalized") or {}).get("target_record_id") or "")
        == record_id
        and (wrapper.get("normalized") or {}).get("field_name") == field_name
        and bool((wrapper.get("normalized") or {}).get("issue_code"))
        and str((wrapper.get("normalized") or {}).get("status") or "requires_review")
        not in {"resolved", "suppressed_redundant", "informational"}
    ]


def _monthly_wrappers_for_record(community: dict, record_id: str) -> list[dict]:
    datasets = _dataset_map(community)
    return [
        wrapper
        for wrapper in datasets["credit_account_monthly_performance"].get("rows") or []
        if str(
            wrapper.get("record_id")
            or (wrapper.get("normalized") or {}).get("monthly_performance_id")
            or ""
        )
        == record_id
    ]


def _assert_monthly_status_value_or_reported(
    community: dict,
    record_id: str,
    expected_status: str,
) -> None:
    """Require a source-vetted status or exact field-local withholding."""

    wrappers = _monthly_wrappers_for_record(community, record_id)
    assert len(wrappers) <= 1, f"{record_id}: duplicate monthly rows"
    if wrappers:
        status = (wrappers[0].get("normalized") or {}).get("status_code")
        if str(status or "").strip().upper() == expected_status:
            return
        assert status is None, (
            f"{record_id}: published {status!r}; source-vetted status is "
            f"{expected_status!r}"
        )
    assert _active_monthly_field_issues(community, record_id, "status_code"), (
        f"{record_id}: source-vetted status {expected_status!r} was withheld "
        "without an exact active status_code issue"
    )


def _assert_exact_native_status_conflict(
    community: dict,
    record_id: str,
    *,
    corrected_status: str,
    native_status: str,
    logical_page: int | None = None,
) -> None:
    datasets = _dataset_map(community)
    issue_wrappers = [
        wrapper
        for wrapper in datasets["extraction_issues"].get("rows") or []
        if (wrapper.get("normalized") or {}).get("target_dataset")
        == "credit_account_monthly_performance"
        and str((wrapper.get("normalized") or {}).get("target_record_id") or "")
        == record_id
        and (wrapper.get("normalized") or {}).get("field_name") == "status_code"
        and (wrapper.get("normalized") or {}).get("issue_code")
        == "candidate_b_native_source_cell_repayment_status_conflict"
        and str((wrapper.get("normalized") or {}).get("status") or "requires_review")
        not in {"resolved", "suppressed_redundant", "informational"}
    ]
    assert len(issue_wrappers) == 1, (
        f"{record_id}: withheld status requires one exact active native-source-cell "
        f"conflict, observed {len(issue_wrappers)}"
    )
    issue_wrapper = issue_wrappers[0]
    issue = issue_wrapper.get("normalized") or {}
    issue_id = str(issue.get("extraction_issue_id") or issue_wrapper.get("record_id") or "")
    evidence_wrappers = [
        wrapper
        for wrapper in datasets["extraction_issue_evidence"].get("rows") or []
        if str((wrapper.get("normalized") or {}).get("extraction_issue_id") or "")
        == issue_id
    ]
    observed_values = {
        str((wrapper.get("normalized") or {}).get("evidence_path") or ""): str(
            (wrapper.get("normalized") or {}).get("string_value") or ""
        )
        .strip()
        .upper()
        for wrapper in evidence_wrappers
        if (wrapper.get("normalized") or {}).get("evidence_kind") == "observed"
    }
    assert observed_values.get("corrected_final") == corrected_status, (
        f"{record_id}: native conflict did not preserve corrected status "
        f"{corrected_status!r}: {observed_values!r}"
    )
    assert observed_values.get("sealed_native_source_cell") == native_status, (
        f"{record_id}: native conflict did not preserve sealed source status "
        f"{native_status!r}: {observed_values!r}"
    )
    reason_codes = {
        str((wrapper.get("normalized") or {}).get("string_value") or "")
        for wrapper in evidence_wrappers
        if (wrapper.get("normalized") or {}).get("evidence_kind") == "reason"
    }
    assert "normalized_value_withheld" in reason_codes, (
        f"{record_id}: exact native status conflict lacks normalized_value_withheld"
    )
    if logical_page is not None:
        expected_range = [logical_page, logical_page]
        assert (issue_wrapper.get("source") or {}).get("page_range") == expected_range, (
            f"{record_id}: conflict issue does not project logical page {logical_page}"
        )
        assert evidence_wrappers and all(
            (wrapper.get("source") or {}).get("page_range") == expected_range
            for wrapper in evidence_wrappers
        ), f"{record_id}: conflict evidence does not stay on logical page {logical_page}"


def _assert_lin_august_2022_status_oracle(community: dict) -> None:
    """Retain p19 August's amount while exactly localizing the M/N conflict."""

    target_record_id = "mg_p19_repayment_0:2022-08"
    target_rows = _monthly_wrappers_for_record(community, target_record_id)
    if len(target_rows) != 1:
        raise AssertionError(
            f"{target_record_id}: conflict row and agreed amount must be retained exactly once"
        )
    values = target_rows[0].get("normalized") or {}
    assert values.get("status_code") is None, (
        f"{target_record_id}: conflicting status must be null, observed "
        f"{values.get('status_code')!r}"
    )
    assert _decimal_amount(values.get("status_amount")) == Decimal(0), (
        f"{target_record_id}: exact agreed amount 0 must be retained"
    )
    _assert_exact_native_status_conflict(
        community,
        target_record_id,
        corrected_status="M",
        native_status="N",
        logical_page=19,
    )


def _assert_lin_p20_continuation_month_binding_oracle(community: dict) -> None:
    """Keep p20 continuation glyphs bound to their physical August/September cells."""

    august_id = "mg_p20_repayment_1:2021-08"
    august_rows = _monthly_wrappers_for_record(community, august_id)
    assert len(august_rows) == 1, f"{august_id}: continuation month must be retained once"
    august = august_rows[0].get("normalized") or {}
    assert str(august.get("status_code") or "").strip().upper() == "*", (
        f"{august_id}: physical August cell is '*', never the September '#' glyph"
    )
    assert _decimal_amount(august.get("status_amount")) == Decimal(0), (
        f"{august_id}: physical August amount must be 0"
    )

    september_id = "mg_p20_repayment_1:2021-09"
    september_rows = _monthly_wrappers_for_record(community, september_id)
    assert len(september_rows) == 1, (
        f"{september_id}: continuation month and amount must be retained exactly once"
    )
    september = september_rows[0].get("normalized") or {}
    assert _decimal_amount(september.get("status_amount")) == Decimal(0), (
        f"{september_id}: physical September amount must be 0"
    )
    september_status = september.get("status_code")
    if str(september_status or "").strip().upper() == "#":
        return
    if september_status is not None:
        raise AssertionError(
            f"{september_id}: expected exact '#'/0 or null status with an exact */# "
            f"native-source-cell conflict, observed {september_status!r}; generic review is insufficient"
        )
    _assert_exact_native_status_conflict(
        community,
        september_id,
        corrected_status="*",
        native_status="#",
    )


def _assert_lin_monthly_position_conservation_oracle(community: dict) -> None:
    datasets = _dataset_map(community)
    monthly_dataset = datasets["credit_account_monthly_performance"]
    rows = monthly_dataset.get("rows") or []
    completeness = monthly_dataset.get("completeness") or {}
    emitted = int(completeness.get("emitted_row_count") or 0)
    omitted = int(completeness.get("omitted_row_count") or 0)
    expected = int(completeness.get("expected_row_count") or 0)
    assert monthly_dataset.get("row_count") == len(rows) == emitted
    assert expected == emitted + omitted == _LIN_EXPECTED_MONTH_POSITIONS

    issue_rows = [
        wrapper.get("normalized") or {}
        for wrapper in datasets["extraction_issues"].get("rows") or []
    ]
    status_grid_issue_ids = {
        str(issue.get("extraction_issue_id") or "")
        for issue in issue_rows
        if issue.get("issue_code") == "candidate_b_monthly_status_grid_unresolved"
        and issue.get("target_dataset") == "credit_account_monthly_performance"
    }
    evidence_rows = [
        wrapper.get("normalized") or {}
        for wrapper in datasets["extraction_issue_evidence"].get("rows") or []
    ]
    reported_withheld = sum(
        int(evidence.get("integer_value") or 0)
        for evidence in evidence_rows
        if str(evidence.get("extraction_issue_id") or "") in status_grid_issue_ids
        and evidence.get("evidence_kind") == "observed"
        and evidence.get("evidence_path") == "withheld_month_count"
    )
    assert reported_withheld == omitted


def _assert_lin_risky_zero_amount_cells_oracle(community: dict) -> None:
    """Keep source-vetted p13/p14/p15 zero-status conflicts localized."""

    datasets = _dataset_map(community)
    monthly_by_id = {
        str(
            wrapper.get("record_id")
            or (wrapper.get("normalized") or {}).get("monthly_performance_id")
            or ""
        ): wrapper.get("normalized") or {}
        for wrapper in datasets["credit_account_monthly_performance"].get("rows") or []
    }
    issue_rows = [
        wrapper.get("normalized") or {}
        for wrapper in datasets["extraction_issues"].get("rows") or []
    ]
    evidence_rows = [
        wrapper.get("normalized") or {}
        for wrapper in datasets["extraction_issue_evidence"].get("rows") or []
    ]
    reasons_by_issue: dict[str, set[str]] = {}
    for evidence in evidence_rows:
        if evidence.get("evidence_kind") == "reason" and evidence.get("string_value"):
            reasons_by_issue.setdefault(
                str(evidence.get("extraction_issue_id") or ""), set()
            ).add(str(evidence["string_value"]))

    conflict_targets = {
        "mg_p13_repayment_1:2022-07": ("N", Decimal("10")),
        "mg_p14_repayment_1:2019-12": ("N", Decimal("10")),
        "mg_p15_repayment_0:2021-03": ("*", Decimal("20")),
    }
    for record_id, (expected_status, observed_amount) in conflict_targets.items():
        if record_id not in monthly_by_id:
            raise AssertionError(
                f"{record_id}: zero-status row must be retained while its raw nonzero "
                "amount is withheld and localized"
            )
        values = monthly_by_id[record_id]
        assert str(values.get("status_code") or "").strip().upper() == expected_status
        assert values.get("status_amount") in (None, "")
        matching_issues = [
            issue
            for issue in issue_rows
            if issue.get("target_dataset") == "credit_account_monthly_performance"
            and str(issue.get("target_record_id") or "") == record_id
            and issue.get("field_name") == "status_amount"
            and issue.get("issue_code") == "candidate_b_monthly_zero_status_amount_conflict"
            and str(issue.get("status") or "requires_review")
            not in {"resolved", "suppressed_redundant", "informational"}
        ]
        assert len(matching_issues) == 1, record_id
        issue = matching_issues[0]
        assert _decimal_amount(issue.get("observed_value")) == observed_amount
        assert issue.get("candidate_value") is None
        assert "normalized_value_withheld" in reasons_by_issue.get(
            str(issue.get("extraction_issue_id") or ""), set()
        )

    p15_april = monthly_by_id["mg_p15_repayment_0:2021-04"]
    assert str(p15_april.get("status_code") or "").strip().upper() == "*"
    assert _decimal_amount(p15_april.get("status_amount")) == Decimal(0)


def _decimal_amount(value: object) -> Decimal | None:
    if value in (None, ""):
        return None
    try:
        amount = Decimal(str(value))
    except InvalidOperation:
        return None
    return amount if amount.is_finite() else None


def _assert_zero_overdue_status_amount_oracle(community: dict) -> None:
    """Reject nonzero business amounts for statuses that mean zero overdue."""

    datasets = _dataset_map(community)
    defects: list[str] = []
    zero_overdue_statuses = {"N", "*", "/", "C"}
    monthly_wrappers = (
        datasets["credit_account_monthly_performance"].get("rows") or []
    )
    monthly_by_id = {
        str(
            wrapper.get("record_id")
            or (wrapper.get("normalized") or {}).get("monthly_performance_id")
            or ""
        ): wrapper
        for wrapper in monthly_wrappers
    }
    for wrapper in monthly_wrappers:
        values = wrapper.get("normalized") or {}
        status_code = str(values.get("status_code") or "").strip().upper()
        if status_code not in zero_overdue_statuses:
            continue
        raw_amount = values.get("status_amount")
        if raw_amount in (None, ""):
            continue
        amount = _decimal_amount(raw_amount)
        if amount is not None and amount == 0:
            continue
        record_id = str(
            wrapper.get("record_id")
            or values.get("monthly_performance_id")
            or ""
        )
        defects.append(
            f"{record_id}: status {status_code} published nonzero/invalid "
            f"status_amount {raw_amount!r}; review metadata cannot make that "
            "normalized business value valid"
        )

    issue_rows = [
        wrapper.get("normalized") or {}
        for wrapper in datasets["extraction_issues"].get("rows") or []
    ]
    evidence_rows = [
        wrapper.get("normalized") or {}
        for wrapper in datasets["extraction_issue_evidence"].get("rows") or []
    ]
    reasons_by_issue: dict[str, set[str]] = {}
    for evidence in evidence_rows:
        if evidence.get("evidence_kind") == "reason" and evidence.get("string_value"):
            reasons_by_issue.setdefault(
                str(evidence.get("extraction_issue_id") or ""), set()
            ).add(str(evidence["string_value"]))
    for wrapper in datasets["field_observations"].get("rows") or []:
        observation = wrapper.get("normalized") or {}
        if (
            observation.get("dataset_name")
            != "credit_account_monthly_performance"
            or observation.get("field_name") != "status_amount"
        ):
            continue
        source_amount = _decimal_amount(observation.get("raw_value"))
        if source_amount is None or source_amount == 0:
            continue
        record_id = str(observation.get("business_record_id") or "")
        monthly = monthly_by_id.get(record_id)
        if monthly is None:
            continue
        values = monthly.get("normalized") or {}
        status_code = str(values.get("status_code") or "").strip().upper()
        if status_code not in zero_overdue_statuses:
            continue
        if values.get("status_amount") not in (None, ""):
            continue  # Already reported above as an invalid normalized value.
        matching_issues = [
            issue
            for issue in issue_rows
            if issue.get("target_dataset")
            == "credit_account_monthly_performance"
            and str(issue.get("target_record_id") or "") == record_id
            and issue.get("field_name") == "status_amount"
            and issue.get("issue_code")
            == "candidate_b_monthly_zero_status_amount_conflict"
            and str(issue.get("status") or "requires_review")
            not in {"resolved", "suppressed_redundant", "informational"}
        ]
        if len(matching_issues) != 1:
            defects.append(
                f"{record_id}: source-observed nonzero status_amount was withheld "
                "without one exact active field-local conflict issue"
            )
            continue
        issue = matching_issues[0]
        issue_amount = _decimal_amount(issue.get("observed_value"))
        issue_id = str(issue.get("extraction_issue_id") or "")
        if (
            observation.get("normalized_value") is not None
            or observation.get("observation_status") not in {"ambiguous", "unreadable"}
            or
            issue_amount != source_amount
            or issue.get("candidate_value") is not None
            or "normalized_value_withheld"
            not in reasons_by_issue.get(issue_id, set())
        ):
            defects.append(
                f"{record_id}: field-local status_amount conflict does not conserve "
                "the raw nonzero value and explicit withholding reason"
            )

    if defects:
        raise AssertionError(
            "Zero-overdue status amount failures:\n- " + "\n- ".join(defects)
        )


def _perceive(fixture: Path):
    return asyncio.run(
        perceive_document(
            fixture,
            PerceiveOptions(
                policy=normalize_parse_policy(
                    enhance_mode="standard",
                    doc_type_hint="credit_report:force",
                )
            ),
        )
    )


@pytest.mark.parametrize("fixture", _FIXTURES, ids=lambda path: path.name)
def test_personal_detail_ocr_correction_invariants(
    fixture: Path,
) -> None:
    """Exercise source correction and the schema boundary on every private report."""
    sealed = _perceive(fixture)
    result = sealed.to_read_view()
    context = build_personal_detail_extraction_context(result)
    domain_specific = result.entities.domain_specific
    raw_bundles = domain_specific.get("_page_evidence_bundles") or []
    raw_snapshot = deepcopy(raw_bundles)
    topology_audit = context.page_topology_audit()

    corrected_pages = context.corrected_evidence_pages()
    business = context.scanned_business(result.full_text or "")
    native_business = context.native_business(result.full_text or "")
    repayment_records = context.corrected_repayment_records()
    audit = context.ocr_correction_audit()
    ocr_bundles = [
        bundle
        for bundle in raw_bundles
        if isinstance(bundle, dict)
        and isinstance(bundle.get("local_structure_evidence"), dict)
        and bundle["local_structure_evidence"].get("lines")
    ]

    # Persist the JSON audit artifact before diagnostic assertions so a
    # topology/test-harness failure cannot discard an otherwise usable output.
    bundle = _project_personal_detail_bundle(sealed, fixture)
    semantic = bundle.semantic_payload()
    payload = bundle.json_payload(semantic)
    audit_dir = os.environ.get("DOCMIRROR_PERSONAL_DETAIL_AUDIT_DIR")
    if audit_dir:
        destination = Path(audit_dir)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / f"{fixture.stem}.community.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (destination / f"{fixture.stem}.semantic.json").write_text(
            json.dumps(semantic, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    assert raw_bundles == raw_snapshot
    assert topology_audit["valid"] is True
    assert topology_audit["logical_page_count"] == len(result.pages)
    assert all(len(logical_pages) >= 1 for logical_pages in topology_audit["logical_pages_by_source"].values())
    assert sorted(context.reading_order_by_logical.values()) == list(
        range(1, len(context.reading_order_by_logical) + 1)
    )
    assert context.entity_context.content_conserved is True
    static_pages = [page for page in corrected_pages if page.get("plugin_static_subpage")]
    static_sources = {int(page.get("source_page") or 0) for page in static_pages}
    if static_pages:
        raw_counts = Counter(
            int(
                bundle.get("source_page_number")
                or (bundle.get("local_structure_evidence") or {}).get("source_page")
                or 0
            )
            for bundle in ocr_bundles
        )
        corrected_counts = Counter(int(page.get("source_page") or 0) for page in corrected_pages)
        for source in static_sources:
            corrected_segments = []
            for page in corrected_pages:
                if int(page.get("source_page") or 0) != source:
                    continue
                if page.get("plugin_static_subpage"):
                    corrected_segments.append(int(page.get("segment_index") or 0))
                    continue
                geometry = context.page_topology.geometry(int(page.get("page") or 0))
                if geometry is not None and geometry.segment_index in {0, 1}:
                    corrected_segments.append(int(geometry.segment_index))
            assert sorted(corrected_segments) == [0, 1]
        assert all(corrected_counts[source] == 2 and raw_counts[source] == 1 for source in static_sources)
        assert all(
            corrected_counts[source] == count for source, count in raw_counts.items() if source not in static_sources
        )
    else:
        assert len(corrected_pages) == len(ocr_bundles)
    assert audit["ocr_started_by_correction_overlay"] is False
    repair_decisions = audit["business_repair"]["page_decisions"]
    assert audit["business_repair"]["field_triggered_ocr_requests"] == sum(
        int(decision.get("ocr_invocations") or 0) for decision in repair_decisions
    )

    assert all(int(decision.get("ocr_invocations") or 0) <= 1 for decision in repair_decisions)
    repaired_pages = [
        int(decision["logical_page"])
        for decision in repair_decisions
        if int(decision.get("ocr_invocations") or 0) == 1
    ]
    assert len(repaired_pages) == len(set(repaired_pages))
    assert all(
        decision["original"] != decision["corrected"]
        and decision["action"] in {"applied", "suggested"}
        and decision["role"]
        for decision in audit["decisions"]
    )

    accounts = business.get("credit_accounts") or []
    expected_counts = _EXPECTED_SCHEMA_INPUT_COUNTS.get(fixture.name)
    if expected_counts is not None:
        expected_accounts, expected_repayments = expected_counts
        if len(accounts) != expected_accounts:
            # A withheld/suppressed record is acceptable only when the final
            # community JSON exposes enough structured account-level issues to
            # cover the complete source-backed shortfall. Silent omissions are
            # never accepted as a count tolerance.
            assert len(accounts) < expected_accounts
            account_issues = [
                issue
                for issue in collect_extraction_issues(context)
                if issue.get("target_dataset") == "credit_accounts"
            ]
            suppressed = {
                str(issue.get("extraction_issue_id") or issue.get("target_record_id") or "")
                for issue in account_issues
                if issue.get("issue_code") == "candidate_b_unmatched_account_table_suppressed"
            }
            sequence_gap = max(
                (
                    len((issue.get("candidate_value") or {}).get("missing_category_sequences") or ())
                    for issue in account_issues
                    if issue.get("issue_code") == "candidate_b_account_sequence_gap"
                ),
                default=0,
            )
            assert len(accounts) + max(len(suppressed), sequence_gap) >= expected_accounts
    # Candidate B owns the final account/month relation. Re-running the retired
    # shared linker against sealed pre-repair grids would compare two different
    # evidence planes and can discard valid corrected-grid rows.
    linked_repayments = list(business.get("repayment_records") or ())
    source_issues = collect_extraction_issues(context)
    if expected_counts is not None:
        # Candidate-B deliberately removed typed cell-level OCR.  The former
        # cell-crop row count is retained only as a coverage regression guard,
        # not as a completeness oracle: a difference is reportable only when
        # canonical schema/source structure independently demonstrates a gap.
        canonical_gaps = [
            issue
            for issue in collect_extraction_issues(context)
            if issue.get("issue_code") == "canonical_monthly_reconstruction_incomplete"
        ]
        population_gaps = [
            issue
            for issue in collect_extraction_issues(context)
            if issue.get("issue_code") == "monthly_population_incomplete_from_account_gap"
        ]
        linkage_gaps = [
            issue
            for issue in collect_extraction_issues(context)
            if issue.get("issue_code") == "monthly_linkage_collision_from_account_gap"
        ]
        status_gaps = [
            issue
            for issue in collect_extraction_issues(context)
            if issue.get("issue_code") == "candidate_b_monthly_status_grid_unresolved"
        ]
        if len(linked_repayments) < int(expected_repayments * 0.90):
            assert canonical_gaps or population_gaps or linkage_gaps or status_gaps
        for canonical_gap in canonical_gaps:
            canonical_count = canonical_gap["observed_value"]["canonical_row_count"]
            assert canonical_count >= len(linked_repayments)
            if canonical_count > len(linked_repayments):
                assert population_gaps or linkage_gaps or status_gaps
            assert canonical_gap["candidate_value"]["structural_expected_row_count"] > len(linked_repayments)
            assert canonical_gap["candidate_value"]["missing_month_count"] > 0
        for population_gap in population_gaps:
            assert population_gap["observed_value"]["canonical_grid_row_count"] >= len(linked_repayments)
            missing_sequences = population_gap["candidate_value"]["missing_account_category_sequences"]
            unresolved_printed_ordinals = any(
                int((issue.get("candidate_value") or {}).get("unresolved_printed_ordinal_count") or 0) > 0
                for issue in collect_extraction_issues(context)
                if issue.get("issue_code") == "candidate_b_account_sequence_gap"
            )
            assert any(missing_sequences.values()) or unresolved_printed_ordinals
        for linkage_gap in linkage_gaps:
            assert linkage_gap["observed_value"]["final_linked_row_count"] == len(linked_repayments)
            assert linkage_gap["candidate_value"]["pre_deduplication_row_count"] > len(linked_repayments)
    unresolved_monthly_ids = {
        str(issue.get("target_record_id") or "")
        for issue in source_issues
        if issue.get("issue_code") == "candidate_b_monthly_grid_owner_unresolved"
    }
    unresolved_monthly_grid_ids = {
        target.rsplit(":", 1)[0] for target in unresolved_monthly_ids if ":" in target
    }
    for record in linked_repayments:
        if record.get("account_id"):
            continue
        repayment_id = str(record.get("repayment_id") or record.get("record_id") or "")
        assert record.get("extraction_status") == "review"
        assert repayment_id in unresolved_monthly_ids or str(record.get("grid_id") or "") in unresolved_monthly_grid_ids
    account_ids = [account.get("account_id") for account in accounts]
    assert len(account_ids) == len(set(account_ids))
    logical_pages = set(context.source_page_by_logical)
    assert all(
        int(ref.get("logical_page") or ref.get("page") or 0) in logical_pages
        for account in accounts
        for ref in account.get("source_refs") or []
        if isinstance(ref, dict)
    )
    assert all(
        int(ref.get("page") or ref.get("logical_page") or 0) in logical_pages
        for record in repayment_records
        for ref in record.get("source_cell_refs") or []
        if isinstance(ref, dict) and (ref.get("page") or ref.get("logical_page"))
    )
    assert isinstance(native_business.get("credit_accounts") or [], list)

    inquiries = business.get("inquiry_records") or []
    for inquiry_type in {row.get("inquiry_type") for row in inquiries}:
        sequences = [row.get("sequence") for row in inquiries if row.get("inquiry_type") == inquiry_type]
        # OCR-visible gaps remain explicit evidence of a missed source row, but
        # duplicate/backward row numbers must not leak into reconstruction.
        assert sequences == sorted(set(sequences))

    assert validate_projection_payload("community", payload).valid
    v2_validation = validate_projection_payload("personal_credit_report_detailed", payload)
    assert v2_validation.valid, v2_validation.errors
    assert payload["document"]["domain_schema"]["version"] == "2.0.0"
    v2_datasets = {dataset["name"]: dataset for dataset in payload["datasets"]}
    for dataset in v2_datasets.values():
        assert dataset["row_count"] == len(dataset.get("rows") or [])
        record_ids = [str(row.get("record_id") or "") for row in dataset.get("rows") or []]
        assert all(record_ids)
        assert len(record_ids) == len(set(record_ids))
    assert v2_datasets["credit_accounts"]["row_count"] == len(accounts)
    canonical_monthly_statuses = {
        "*", "/", "#", "N", "1", "2", "3", "4", "5", "6", "7",
        "A", "B", "C", "D", "G", "M", "Z",
    }
    typed_linked_repayments = [
        record
        for record in linked_repayments
        if record.get("account_id")
        and str(record.get("status_code") or record.get("status") or "").strip().upper()
        in canonical_monthly_statuses
    ]
    v2_statuses = {
        row["normalized"]["dataset_name"]: row["normalized"] for row in v2_datasets["dataset_status"]["rows"]
    }
    assert set(v2_statuses) <= set(PBOC_DATASET_ORDER) - {
        "field_observations",
        "extraction_issues",
        "extraction_issue_evidence",
        "pboc_extension_fields",
        "dataset_status",
    }
    assert all(
        status["presence_status"] in {"not_observed", "partial", "extraction_failed", "unknown"}
        for status in v2_statuses.values()
    )
    assert all(
        status["observed_row_count"] == v2_datasets.get(name, {}).get("row_count", 0)
        for name, status in v2_statuses.items()
    )
    account_rows = [row["normalized"] for row in v2_datasets["credit_accounts"]["rows"]]
    monthly_record_rows = v2_datasets["credit_account_monthly_performance"]["rows"]
    monthly_rows = [row["normalized"] for row in monthly_record_rows]
    assert len(
        {
            (str(row.get("grid_id") or ""), str(row.get("performance_month") or ""))
            for row in monthly_rows
        }
    ) == len(monthly_rows)
    assert all("raw_detail_lines" not in row and "raw_detail_text" not in row for row in account_rows)
    account_id_set = {item["account_id"] for item in account_rows}
    assert all(
        not row.get("account_identifier")
        or str(row["account_identifier"]).replace("-", "").isalnum()
        and str(row["account_identifier"]).isascii()
        for row in monthly_rows
    )
    issue_rows = [row["normalized"] for row in v2_datasets["extraction_issues"]["rows"]]
    active_native_status_conflict_ids = {
        str(row.get("target_record_id") or "")
        for row in issue_rows
        if row.get("target_dataset") == "credit_account_monthly_performance"
        and row.get("field_name") == "status_code"
        and row.get("issue_code")
        == "candidate_b_native_source_cell_repayment_status_conflict"
        and str(row.get("status") or "requires_review")
        not in {"resolved", "suppressed_redundant", "informational"}
    }
    canonical_monthly_status_values = {
        "*", "/", "#", "N", "1", "2", "3", "4", "5", "6", "7",
        "A", "B", "C", "D", "G", "M", "Z",
    }
    for wrapper, row in zip(monthly_record_rows, monthly_rows, strict=True):
        status_code = row.get("status_code")
        if status_code in canonical_monthly_status_values:
            continue
        monthly_id = str(
            wrapper.get("record_id") or row.get("monthly_performance_id") or ""
        )
        assert "status_code" in row and status_code is None, (
            f"{monthly_id}: monthly status must be canonical or explicitly null, "
            f"observed {status_code!r}"
        )
        assert monthly_id in active_native_status_conflict_ids, (
            f"{monthly_id}: null monthly status lacks an exact active native-source-cell conflict"
        )
    field_observation_rows = [
        row["normalized"]
        for row in v2_datasets.get("field_observations", {}).get("rows", [])
    ]
    assert all(
        not row.get("target_dataset") or row["target_dataset"] in set(PBOC_DATASET_ORDER)
        for row in issue_rows
    )
    assert all(row.get("target_dataset") != "unknown" for row in issue_rows)
    final_unresolved_monthly_ids = {
        str(row.get("target_record_id") or "")
        for row in issue_rows
        if row.get("target_dataset") == "credit_account_monthly_performance"
        and row.get("issue_code") == "candidate_b_monthly_grid_owner_unresolved"
    }
    final_unresolved_monthly_grid_ids = {
        target.rsplit(":", 1)[0]
        for target in final_unresolved_monthly_ids
        if ":" in target
    }
    for wrapper, row in zip(monthly_record_rows, monthly_rows, strict=True):
        if row.get("account_id"):
            assert row["account_id"] in account_id_set
            continue
        monthly_id = str(wrapper.get("record_id") or row.get("monthly_performance_id") or "")
        assert row.get("extraction_status") == "review"
        assert (
            monthly_id in final_unresolved_monthly_ids
            or str(row.get("grid_id") or "") in final_unresolved_monthly_grid_ids
        )
    missing_identifier_ids = {
        str(account.get("account_id") or "")
        for account in account_rows
        if not account.get("account_identifier")
    }
    explicitly_reported_identifier_ids = {
        str(row.get("target_record_id") or "")
        for row in issue_rows
        if row.get("target_dataset") == "credit_accounts"
        and (
            row.get("field_name") == "account_identifier"
            or row.get("issue_code") == "candidate_b_account_table_missing"
        )
    }
    assert missing_identifier_ids <= explicitly_reported_identifier_ids
    forbidden_business_metadata = {
        "audit",
        "amount_bbox",
        "bbox",
        "raw_status",
        "recognition_source",
        "status_bbox",
    }
    for dataset_name, dataset in v2_datasets.items():
        if dataset_name in {
            "field_observations",
            "extraction_issues",
            "extraction_issue_evidence",
            "pboc_extension_fields",
            "dataset_status",
        }:
            continue
        for wrapper in dataset.get("rows", []):
            for value in (wrapper.get("normalized") or {}).values():
                assert not isinstance(value, (dict, list, tuple, set))
                if isinstance(value, str) and value[:1] in "[{" and value[-1:] in "]}":
                    try:
                        decoded = json.loads(value)
                    except (TypeError, ValueError):
                        decoded = None
                    assert not isinstance(decoded, (dict, list))
        assert all(
            not (forbidden_business_metadata & set(row.get(pool_name) or {}))
            for row in dataset.get("rows", [])
            for pool_name in ("normalized", "canonical_raw", "raw")
        )
    for dataset_name, dataset in v2_datasets.items():
        if dataset_name in {
            "field_observations",
            "extraction_issues",
            "extraction_issue_evidence",
            "pboc_extension_fields",
            "dataset_status",
        }:
            continue
        for wrapper in dataset.get("rows", []):
            values = wrapper.get("normalized") or {}
            has_review = values.get("extraction_status") == "review" or isinstance(
                wrapper.get("review"), dict
            )
            if not has_review:
                assert "confidence" not in wrapper
    months_by_grid: dict[str, list[str]] = {}
    for row in monthly_rows:
        grid_key = str(row.get("grid_id") or row.get("account_id") or "unresolved")
        months_by_grid.setdefault(grid_key, []).append(str(row.get("performance_month") or ""))
        if str(row.get("status_code") or "") in {"1", "2", "3", "4", "5", "6", "7"}:
            try:
                amount = Decimal(str(row.get("status_amount") or ""))
            except InvalidOperation:
                amount = Decimal(0)
            assert amount > 0
    assert all(months == sorted(months) for months in months_by_grid.values())
    assert not any(
        row.get("field_name") == "housing_fund_record_id"
        for row in issue_rows
    )
    unresolved_source_months = len(linked_repayments) - len(typed_linked_repayments)
    final_status_grid_issues = [
        row
        for row in issue_rows
        if row.get("issue_code") == "candidate_b_monthly_status_grid_unresolved"
        and row.get("target_dataset") == "credit_account_monthly_performance"
    ]
    if unresolved_source_months:
        assert final_status_grid_issues
        assert v2_statuses["credit_account_monthly_performance"]["presence_status"] == "partial"
    monthly_status_contract_issues = [
        row
        for row in issue_rows
        if row.get("issue_code") == "pboc_cell_contract_unresolved"
        and row.get("target_dataset") == "credit_account_monthly_performance"
        and row.get("field_name") in {"status", "status_code"}
    ]
    monthly_status_contract_targets = {
        str(row.get("target_record_id") or "")
        for row in monthly_status_contract_issues
    }
    emitted_monthly_record_ids = {
        str(wrapper.get("record_id") or (wrapper.get("normalized") or {}).get("monthly_performance_id") or "")
        for wrapper in monthly_record_rows
    }
    assert all(row.get("field_name") == "status_code" for row in monthly_status_contract_issues)
    assert "" not in monthly_status_contract_targets
    assert len(monthly_status_contract_targets) == len(monthly_status_contract_issues)
    assert monthly_status_contract_targets.isdisjoint(emitted_monthly_record_ids)
    localized_status_observation_targets = {
        str(row.get("business_record_id") or "")
        for row in field_observation_rows
        if row.get("dataset_name") == "credit_account_monthly_performance"
        and row.get("field_name") == "status_code"
        and row.get("normalized_value") is None
        and row.get("observation_status") in {"ambiguous", "unreadable"}
    }
    assert monthly_status_contract_targets <= localized_status_observation_targets
    for row in v2_datasets.get("pboc_extension_fields", {}).get("rows", []):
        values = row.get("normalized") or {}
        if values.get("source_dataset") != "personal_detail_summary_cells":
            continue
        value = values.get("value")
        assert not isinstance(value, (dict, list))
        assert not (
            isinstance(value, str)
            and len(value) > 1
            and value[0] in "[{"
            and value[-1] in "]}"
        )
    assert all(not any(key.startswith(("observed__", "candidate__", "reason__")) for key in row) for row in issue_rows)
    evidence_rows = [
        row["normalized"] for row in v2_datasets.get("extraction_issue_evidence", {}).get("rows", [])
    ]
    issue_ids = {row["extraction_issue_id"] for row in issue_rows}
    assert evidence_rows
    assert all(row["extraction_issue_id"] in issue_ids for row in evidence_rows)
    assert all(
        set(row)
        <= {
            "extraction_issue_evidence_id",
            "extraction_issue_id",
            "evidence_kind",
            "evidence_path",
            "value_type",
            "string_value",
            "integer_value",
            "number_value",
            "boolean_value",
        }
        for row in evidence_rows
    )
    emitted_ids_by_dataset = {
        name: {
            str(value)
            for wrapper in dataset.get("rows", [])
            for value in (
                wrapper.get("record_id"),
                *((wrapper.get("normalized") or {}).get(key) for key in (wrapper.get("normalized") or {}) if key.endswith("_id")),
            )
            if value not in (None, "")
        }
        for name, dataset in v2_datasets.items()
    }
    reasons_by_issue: dict[str, set[str]] = {}
    for row in evidence_rows:
        if row.get("evidence_kind") == "reason" and row.get("string_value"):
            reasons_by_issue.setdefault(str(row["extraction_issue_id"]), set()).add(str(row["string_value"]))

    typed_linked_ids = [
        str(
            record.get("repayment_id")
            or record.get("monthly_performance_id")
            or record.get("record_id")
            or ""
        )
        for record in typed_linked_repayments
    ]
    assert all(typed_linked_ids)
    assert len(typed_linked_ids) == len(set(typed_linked_ids))
    typed_linked_id_set = set(typed_linked_ids)
    typed_linked_by_id = dict(zip(typed_linked_ids, typed_linked_repayments, strict=True))
    emitted_monthly_ids = {
        str(
            wrapper.get("record_id")
            or (wrapper.get("normalized") or {}).get("monthly_performance_id")
            or ""
        )
        for wrapper in monthly_record_rows
    }
    assert "" not in emitted_monthly_ids
    assert emitted_monthly_ids <= typed_linked_id_set
    _assert_zero_overdue_status_amount_oracle(payload)

    monthly_wrapper_by_id = {
        str(
            wrapper.get("record_id")
            or (wrapper.get("normalized") or {}).get("monthly_performance_id")
            or ""
        ): wrapper
        for wrapper in monthly_record_rows
    }
    for target_record_id, source_record in typed_linked_by_id.items():
        source_status = str(
            source_record.get("status_code") or source_record.get("status") or ""
        ).strip().upper()
        source_amount = _decimal_amount(source_record.get("overdue_amount"))
        if source_status not in {"N", "*", "/", "C"} or source_amount in {
            None,
            Decimal(0),
        }:
            continue
        emitted_wrapper = monthly_wrapper_by_id.get(target_record_id)
        if emitted_wrapper is not None:
            assert (emitted_wrapper.get("normalized") or {}).get("status_amount") in {
                None,
                "",
            }
        matching_issues = [
            row
            for row in issue_rows
            if row.get("target_dataset") == "credit_account_monthly_performance"
            and str(row.get("target_record_id") or "") == target_record_id
            and row.get("field_name") == "status_amount"
            and row.get("issue_code")
            == "candidate_b_monthly_zero_status_amount_conflict"
            and str(row.get("status") or "requires_review")
            not in {"resolved", "suppressed_redundant", "informational"}
        ]
        assert len(matching_issues) == 1, target_record_id
        issue = matching_issues[0]
        issue_id = str(issue["extraction_issue_id"])
        assert _decimal_amount(issue.get("observed_value")) == source_amount
        assert issue.get("candidate_value") is None
        assert "normalized_value_withheld" in reasons_by_issue.get(issue_id, set())
        matching_observations = [
            row
            for row in field_observation_rows
            if row.get("dataset_name") == "credit_account_monthly_performance"
            and str(row.get("business_record_id") or "") == target_record_id
            and row.get("field_name") == "status_amount"
            and _decimal_amount(row.get("raw_value")) == source_amount
            and row.get("normalized_value") is None
            and row.get("observation_status") in {"ambiguous", "unreadable"}
        ]
        assert matching_observations, target_record_id

    actionable_status_withheld_ids: set[str] = set()
    for target_record_id in sorted(typed_linked_id_set - emitted_monthly_ids):
        matching_issues = [
            row
            for row in issue_rows
            if row.get("target_dataset") == "credit_account_monthly_performance"
            and str(row.get("target_record_id") or "") == target_record_id
            and row.get("field_name") == "status_code"
            and row.get("issue_code") == "candidate_b_monthly_terminal_status_conflict"
            and str(row.get("status") or "requires_review")
            not in {"resolved", "suppressed_redundant", "informational"}
        ]
        assert len(matching_issues) == 1, target_record_id
        issue = matching_issues[0]
        issue_id = str(issue["extraction_issue_id"])
        raw_status = str(
            typed_linked_by_id[target_record_id].get("status_code")
            or typed_linked_by_id[target_record_id].get("status")
            or ""
        ).strip().upper()
        assert str(issue.get("observed_value") or "").strip().upper() == raw_status
        assert issue.get("candidate_value") is None
        assert {
            "normalized_value_withheld",
            "terminal_status_not_inferred",
        } <= reasons_by_issue.get(issue_id, set())
        matching_observations = [
            row
            for row in field_observation_rows
            if row.get("dataset_name") == "credit_account_monthly_performance"
            and str(row.get("business_record_id") or "") == target_record_id
            and row.get("field_name") == "status_code"
            and row.get("normalized_value") is None
            and str(row.get("raw_value") or "").strip().upper() == raw_status
            and row.get("observation_status") in {"ambiguous", "unreadable"}
        ]
        assert matching_observations, target_record_id
        actionable_status_withheld_ids.add(target_record_id)

    assert (
        emitted_monthly_ids
        == typed_linked_id_set - actionable_status_withheld_ids
    )
    assert (
        v2_datasets["credit_account_monthly_performance"]["row_count"]
        == len(emitted_monthly_ids)
    )
    status_grid_issue_ids = {
        str(row["extraction_issue_id"])
        for row in final_status_grid_issues
    }
    reported_withheld_months = sum(
        int(row.get("integer_value") or 0)
        for row in evidence_rows
        if str(row.get("extraction_issue_id") or "") in status_grid_issue_ids
        and row.get("evidence_kind") == "observed"
        and row.get("evidence_path") == "withheld_month_count"
    )
    assert reported_withheld_months == (
        unresolved_source_months + len(actionable_status_withheld_ids)
    )
    monthly_completeness = v2_datasets["credit_account_monthly_performance"][
        "completeness"
    ]
    assert monthly_completeness["emitted_row_count"] == len(emitted_monthly_ids)
    assert monthly_completeness["expected_row_count"] == (
        monthly_completeness["emitted_row_count"]
        + monthly_completeness["omitted_row_count"]
    )
    linked_emission_gap = len(linked_repayments) - len(emitted_monthly_ids)
    assert linked_emission_gap == reported_withheld_months
    assert monthly_completeness["omitted_row_count"] >= linked_emission_gap
    if monthly_completeness["expected_row_count"] == len(linked_repayments):
        assert monthly_completeness["omitted_row_count"] == linked_emission_gap
    for issue_id in status_grid_issue_ids:
        issue_evidence = [
            row
            for row in evidence_rows
            if str(row.get("extraction_issue_id") or "") == issue_id
            and row.get("evidence_kind") == "observed"
        ]
        withheld_count = next(
            int(row.get("integer_value") or 0)
            for row in issue_evidence
            if row.get("evidence_path") == "withheld_month_count"
        )
        withheld_months = sorted(
            str(row.get("string_value") or "")
            for row in issue_evidence
            if re.fullmatch(r"withheld_months\[\d+\]", str(row.get("evidence_path") or ""))
        )
        assert len(withheld_months) == withheld_count
        assert len(withheld_months) == len(set(withheld_months))
    non_emission_markers = (
        "withheld",
        "suppressed",
        "not_invented",
        "not_emitted",
        "unresolved",
        "record_not_silently_dropped",
        "silent_drop_prevented",
    )
    for row in issue_rows:
        dataset_name = str(row.get("target_dataset") or "")
        target_record_id = str(row.get("target_record_id") or "")
        if not dataset_name or not target_record_id:
            continue
        if target_record_id in emitted_ids_by_dataset.get(dataset_name, set()):
            continue
        assert any(
            marker in reason
            for reason in reasons_by_issue.get(str(row["extraction_issue_id"]), set())
            for marker in non_emission_markers
        )

    control_datasets = {
        "field_observations",
        "extraction_issues",
        "extraction_issue_evidence",
        "pboc_extension_fields",
        "dataset_status",
    }
    dash_only = re.compile(r"[-‐‑‒–—―－﹘﹣]+")
    assert not any(
        isinstance(value, str) and dash_only.fullmatch(value.strip())
        for dataset_name, dataset in v2_datasets.items()
        if dataset_name not in control_datasets
        for wrapper in dataset.get("rows", [])
        for value in (wrapper.get("normalized") or {}).values()
    )

    if expected_counts == (45, 944):
        _assert_lin_semantic_account_oracle(semantic, payload)
        _assert_lin_month_ref_physical_ownership_oracle(semantic)
        _assert_lin_monthly_position_conservation_oracle(payload)
        _assert_lin_august_2022_status_oracle(payload)
        _assert_lin_p20_continuation_month_binding_oracle(payload)
        _assert_lin_risky_zero_amount_cells_oracle(payload)

        amount_issue_targets = {
            str(row.get("target_record_id") or "")
            for row in issue_rows
            if row.get("target_dataset") == "credit_account_monthly_performance"
            and row.get("field_name") == "status_amount"
        }
        assert all(
            row.get("status_amount") not in (None, "")
            or str(wrapper.get("record_id") or row.get("monthly_performance_id") or "")
            in amount_issue_targets
            for wrapper, row in zip(monthly_record_rows, monthly_rows, strict=True)
        )

        overview_rows = [
            wrapper.get("normalized") or {}
            for wrapper in v2_datasets["credit_business_overview"]["rows"]
        ]
        for row in overview_rows:
            if row.get("metric_code") != "account_count" or row.get("numeric_value") in (None, ""):
                continue
            assert Decimal(str(row["numeric_value"])) <= Decimal(len(account_rows))

        def has_field_issue(dataset_name: str, record_id: str, field_name: str) -> bool:
            return any(
                row.get("target_dataset") == dataset_name
                and str(row.get("target_record_id") or "") == record_id
                and row.get("field_name") == field_name
                for row in issue_rows
            )

        _assert_lin_february_2020_status_oracle(payload)

        residence_wrapper = next(
            wrapper
            for wrapper in v2_datasets["subject_residences"]["rows"]
            if (wrapper.get("normalized") or {}).get("sequence") == 5
        )
        residence = residence_wrapper.get("normalized") or {}
        assert "卢滨路" in str(residence.get("address") or "") or has_field_issue(
            "subject_residences",
            str(residence_wrapper.get("record_id") or ""),
            "address",
        )

        account_22_wrapper = next(
            wrapper
            for wrapper in v2_datasets["credit_accounts"]["rows"]
            if (wrapper.get("normalized") or {}).get("account_id")
            == "credit_account:non_revolving_loan:22"
        )
        account_22 = account_22_wrapper.get("normalized") or {}
        assert "蚂蚁商诚" in str(account_22.get("management_institution") or "") or has_field_issue(
            "credit_accounts",
            str(account_22_wrapper.get("record_id") or ""),
            "management_institution",
        )

        inquiry_1_wrapper = next(
            wrapper
            for wrapper in v2_datasets["inquiries"]["rows"]
            if (wrapper.get("normalized") or {}).get("query_channel") == "institution"
            and (wrapper.get("normalized") or {}).get("sequence") == 1
        )
        inquiry_1 = inquiry_1_wrapper.get("normalized") or {}
        assert "中国建设银行股份有限公司北京市分行" in str(
            inquiry_1.get("institution") or ""
        ) or has_field_issue(
            "inquiries",
            str(inquiry_1_wrapper.get("record_id") or ""),
            "institution",
        )

    agreement_rows = [row["normalized"] for row in v2_datasets["credit_agreements"]["rows"]]
    agreement_ids = {
        str(row.get("credit_agreement_id") or "")
        for row in agreement_rows
        if row.get("credit_agreement_id")
    }
    assert all(
        not row.get("target_record_id") or str(row["target_record_id"]) in agreement_ids
        for row in issue_rows
        if row.get("target_dataset") == "credit_agreements"
    )
    for required_field in ("institution", "facility_type", "effective_date"):
        missing_required_ids = {
            str(row.get("credit_agreement_id") or "")
            for row in agreement_rows
            if row.get(required_field) in (None, "")
        }
        explicitly_reported_required_ids = {
            str(row.get("target_record_id") or "")
            for row in issue_rows
            if row.get("target_dataset") == "credit_agreements"
            and row.get("field_name") == required_field
        }
        assert missing_required_ids <= explicitly_reported_required_ids
    expected_agreements = _EXPECTED_AGREEMENT_COUNTS.get(fixture.name)
    if expected_agreements is not None:
        assert len(agreement_rows) <= expected_agreements
        assert len({row.get("account_identifier") for row in agreement_rows}) == len(agreement_rows)
        printed_sequences = [row["sequence"] for row in agreement_rows if row.get("sequence") is not None]
        assert len(printed_sequences) == len(set(printed_sequences))
        assert all(1 <= sequence <= expected_agreements for sequence in printed_sequences)
        unresolved_sequences = len(agreement_rows) - len(printed_sequences)
        reported_sequences = sum(
            row.get("target_dataset") == "credit_agreements"
            and row.get("field_name") == "sequence"
            and row.get("issue_code") == "candidate_b_credit_agreement_sequence_unresolved"
            for row in issue_rows
        )
        assert reported_sequences >= unresolved_sequences
        if len(agreement_rows) < expected_agreements:
            assert v2_statuses["credit_agreements"]["presence_status"] == "partial"
            assert any(
                row.get("target_dataset") == "credit_agreements"
                and row.get("issue_code")
                in {
                    "source_sequence_or_count_gap",
                    "candidate_b_credit_agreement_population_gap",
                }
                for row in issue_rows
            )
    if fixture.name == "余泽熙7.15征信.pdf":
        assert [row.get("sequence") for row in agreement_rows] == list(range(1, 9))
        agreement_two = next(row for row in agreement_rows if row.get("sequence") == 2)
        assert agreement_two["account_identifier"] == (
            "B10711000H0001100000111111112446567900000"
        )
        assert agreement_two["institution"] == "中国光大银行股份有限公司"
        assert agreement_two["facility_type"] == "信用卡共享额度"
        assert agreement_two["effective_date"] == "2019-12-01"
        assert agreement_two["validity_type"] == "perpetual"
        assert agreement_two["facility_limit"] == "0"
        assert agreement_two["used_limit"] == "0"
        assert agreement_two["currency"] == "CNY"
        assert not any(
            row.get("issue_code") == "candidate_b_credit_agreement_identity_ambiguous"
            for row in issue_rows
        )
    expected_inquiries = _EXPECTED_INQUIRY_COUNTS.get(fixture.name)
    if expected_inquiries is not None and v2_datasets["inquiries"]["row_count"] != expected_inquiries:
        assert v2_datasets["inquiries"]["row_count"] < expected_inquiries
        assert v2_statuses["inquiries"]["presence_status"] == "partial"
        assert any(
            row.get("target_dataset") == "inquiries"
            and row.get("issue_code") in {"canonical_inquiry_sequence_gap", "source_sequence_or_count_gap"}
            for row in issue_rows
        )

    if fixture.name.startswith("叶永燕"):
        institutional = [row for row in inquiries if row.get("inquiry_type") == "institution"]
        personal = [row for row in inquiries if row.get("inquiry_type") == "personal"]
        source_ledger = semantic["domain"]["facts"][
            "personal_detail_source_completeness_ledger"
        ]
        assert source_ledger["credit_accounts"] == 42
        assert source_ledger["account_family_source_populations"] == {
            "non_revolving_loan": 18,
            "revolving_loan_subaccount": 6,
            "revolving_loan_account": 6,
            "credit_card": 12,
        }
        assert source_ledger["account_family_endpoints"] == {
            "non_revolving_loan": 18,
            "revolving_loan_subaccount": 6,
            "revolving_loan_account": 6,
            "credit_card": 12,
        }
        assert source_ledger["inquiry_records"] == 112
        assert source_ledger["inquiry_sequence_endpoints"] == {
            "institution": 96,
            "personal": 16,
        }

        account_completeness = v2_datasets["credit_accounts"]["completeness"]
        assert Counter(row.get("account_type") for row in account_rows) == {
            "non_revolving_loan": 18,
            "revolving_loan_subaccount": 6,
            "revolving_loan_account": 6,
            "credit_card": 12,
        }
        recovered_card_types = {
            str(row.get("account_identifier") or ""): row.get("account_type")
            for row in account_rows
            if str(row.get("account_identifier") or "") in _YE_RECOVERED_CARD_IDS
        }
        assert set(recovered_card_types) == _YE_RECOVERED_CARD_IDS
        assert set(recovered_card_types.values()) == {"credit_card"}
        assert account_completeness["expected_row_count"] == 42
        assert account_completeness["emitted_row_count"] == len(account_rows)
        assert account_completeness["expected_row_count"] == (
            account_completeness["emitted_row_count"]
            + account_completeness["omitted_row_count"]
        )
        assert monthly_completeness["expected_row_count"] == 951
        inquiry_completeness = v2_datasets["inquiries"]["completeness"]
        assert inquiry_completeness["expected_row_count"] == 112
        assert inquiry_completeness["emitted_row_count"] == len(inquiries)
        assert inquiry_completeness["expected_row_count"] == (
            inquiry_completeness["emitted_row_count"]
            + inquiry_completeness["omitted_row_count"]
        )
        geometry_defects, compared_month_refs = _lin_month_ref_physical_ownership(semantic)
        assert compared_month_refs, "Ye physical month-column oracle compared no exact refs"
        if geometry_defects:
            raise AssertionError(
                "Ye physical month-column ownership failures:\n- "
                + "\n- ".join(geometry_defects)
            )
        for record_id in (
            "mg_p10_repayment_0:2022-06",
            "mg_p10_repayment_0:2020-08",
            "mg_p17_repayment_0:2020-09",
        ):
            _assert_monthly_status_value_or_reported(payload, record_id, "N")

        # Population shortfalls are governed by the structured checks above;
        # this block pins the quality of values that were safely emitted.
        assert len(institutional) >= 94
        if len(institutional) < 96:
            gap = next(
                issue
                for issue in collect_extraction_issues(context)
                if issue.get("issue_code") == "canonical_inquiry_sequence_gap"
                and (issue.get("observed_value") or {}).get("inquiry_type") == "institution"
            )
            assert gap["candidate_value"]["missing_sequences"]
            assert "dataset_incomplete" in gap["reason_codes"]
        assert len(personal) == 16
        if len(institutional) == 96:
            assert [row["sequence"] for row in institutional] == list(range(1, 97))
        assert [row["sequence"] for row in personal] == list(range(1, 17))
        assert audit["applied_count"] >= 70


def _project_personal_detail_bundle(sealed, fixture: Path):
    """Project this plugin while an unrelated enterprise-only semantic contract is global."""
    return CreditReportPlugin().project_bundle(sealed, file_path=str(fixture))


def test_saved_lin_semantic_account_fragment_oracle() -> None:
    """Audit a saved Lin pair without invoking document perception or OCR."""

    audit_dir = os.environ.get("DOCMIRROR_PERSONAL_DETAIL_SAVED_LIN_AUDIT_DIR")
    if not audit_dir:
        pytest.skip("set DOCMIRROR_PERSONAL_DETAIL_SAVED_LIN_AUDIT_DIR")

    directory = Path(audit_dir)
    semantic_path = directory / "林岚挺征信.semantic.json"
    community_path = directory / "林岚挺征信.community.json"
    assert semantic_path.is_file(), semantic_path
    assert community_path.is_file(), community_path

    semantic = json.loads(semantic_path.read_text(encoding="utf-8"))
    community = json.loads(community_path.read_text(encoding="utf-8"))
    _assert_lin_semantic_account_oracle(semantic, community)
    saved_oracle_failures: list[str] = []
    for oracle, payload in (
        (_assert_lin_month_ref_physical_ownership_oracle, semantic),
        (_assert_lin_monthly_position_conservation_oracle, community),
        (_assert_lin_february_2020_status_oracle, community),
        (_assert_lin_august_2022_status_oracle, community),
        (_assert_lin_p20_continuation_month_binding_oracle, community),
        (_assert_lin_risky_zero_amount_cells_oracle, community),
        (_assert_zero_overdue_status_amount_oracle, community),
    ):
        try:
            oracle(payload)
        except AssertionError as exc:
            saved_oracle_failures.append(str(exc))
    assert not saved_oracle_failures, "Saved Lin oracle failures:\n- " + "\n- ".join(
        saved_oracle_failures
    )


def test_saved_ye_population_and_month_geometry_oracle() -> None:
    """Audit a saved Ye pair without invoking document perception or OCR."""

    audit_dir = os.environ.get("DOCMIRROR_PERSONAL_DETAIL_SAVED_YE_AUDIT_DIR")
    if not audit_dir:
        pytest.skip("set DOCMIRROR_PERSONAL_DETAIL_SAVED_YE_AUDIT_DIR")

    directory = Path(audit_dir)
    semantic_paths = list(directory.glob("*.semantic.json"))
    community_paths = list(directory.glob("*.community.json"))
    assert len(semantic_paths) == len(community_paths) == 1, directory
    semantic = json.loads(semantic_paths[0].read_text(encoding="utf-8"))
    community = json.loads(community_paths[0].read_text(encoding="utf-8"))
    datasets = _dataset_map(community)
    ledger = semantic["domain"]["facts"]["personal_detail_source_completeness_ledger"]

    failures: list[str] = []

    expected_families = {
        "non_revolving_loan": 18,
        "revolving_loan_subaccount": 6,
        "revolving_loan_account": 6,
        "credit_card": 12,
    }
    if ledger.get("credit_accounts") != 42:
        failures.append(f"source credit_accounts={ledger.get('credit_accounts')!r}, expected 42")
    if ledger.get("account_family_source_populations") != expected_families:
        failures.append(
            "source account families="
            f"{ledger.get('account_family_source_populations')!r}, expected {expected_families!r}"
        )
    if ledger.get("account_family_endpoints") != expected_families:
        failures.append(
            "account family endpoints="
            f"{ledger.get('account_family_endpoints')!r}, expected {expected_families!r}"
        )
    if ledger.get("inquiry_records") != 112:
        failures.append(f"source inquiry_records={ledger.get('inquiry_records')!r}, expected 112")
    expected_inquiry_endpoints = {"institution": 96, "personal": 16}
    if ledger.get("inquiry_sequence_endpoints") != expected_inquiry_endpoints:
        failures.append(
            "inquiry endpoints="
            f"{ledger.get('inquiry_sequence_endpoints')!r}, expected {expected_inquiry_endpoints!r}"
        )

    expected_counts = {
        "credit_accounts": 42,
        "credit_account_monthly_performance": 951,
        "inquiries": 112,
    }
    for dataset_name, expected_count in expected_counts.items():
        dataset = datasets[dataset_name]
        completeness = dataset["completeness"]
        if completeness["expected_row_count"] != expected_count:
            failures.append(
                f"{dataset_name} expected={completeness['expected_row_count']!r}, "
                f"source oracle={expected_count}"
            )
        if completeness["emitted_row_count"] != dataset["row_count"]:
            failures.append(f"{dataset_name} emitted count disagrees with row_count")
        if completeness["expected_row_count"] != (
            completeness["emitted_row_count"] + completeness["omitted_row_count"]
        ):
            failures.append(f"{dataset_name} emitted+omitted conservation failed")
    account_rows = [wrapper.get("normalized") or {} for wrapper in datasets["credit_accounts"]["rows"]]
    observed_families = Counter(row.get("account_type") for row in account_rows)
    if observed_families != expected_families:
        failures.append(f"emitted account families={dict(observed_families)!r}")
    recovered_card_types = {
        str(row.get("account_identifier") or ""): row.get("account_type")
        for row in account_rows
        if str(row.get("account_identifier") or "") in _YE_RECOVERED_CARD_IDS
    }
    if set(recovered_card_types) != _YE_RECOVERED_CARD_IDS:
        failures.append(
            "missing recovered card IDs="
            f"{sorted(_YE_RECOVERED_CARD_IDS - set(recovered_card_types))!r}"
        )
    if set(recovered_card_types.values()) - {"credit_card"}:
        failures.append(f"recovered cards have wrong families={recovered_card_types!r}")

    inquiry_rows = [wrapper.get("normalized") or {} for wrapper in datasets["inquiries"]["rows"]]
    institutional_sequences = sorted(
        int(row["sequence"])
        for row in inquiry_rows
        if row.get("query_channel") == "institution"
    )
    personal_sequences = sorted(
        int(row["sequence"])
        for row in inquiry_rows
        if row.get("query_channel") == "personal"
    )
    expected_institutional = [*range(1, 27), *range(28, 97)]
    if institutional_sequences != expected_institutional:
        failures.append(
            f"institution inquiry sequences={institutional_sequences!r}, "
            f"expected={expected_institutional!r}"
        )
    expected_personal = list(range(1, 17))
    if personal_sequences != expected_personal:
        failures.append(
            f"personal inquiry sequences={personal_sequences!r}, expected={expected_personal!r}"
        )

    geometry_defects, compared_month_refs = _lin_month_ref_physical_ownership(semantic)
    if not compared_month_refs:
        failures.append("Ye physical month-column oracle compared no exact refs")
    failures.extend(geometry_defects)
    for record_id in (
        "mg_p10_repayment_0:2022-06",
        "mg_p10_repayment_0:2020-08",
        "mg_p17_repayment_0:2020-09",
    ):
        try:
            _assert_monthly_status_value_or_reported(community, record_id, "N")
        except AssertionError as exc:
            failures.append(str(exc))
    assert not failures, "Saved Ye oracle failures:\n- " + "\n- ".join(failures)


def test_physical_month_column_helper_is_fail_closed() -> None:
    bands = tuple((float(index), float(index + 1)) for index in range(13))

    assert _dominant_physical_column((6.1, 0.0, 6.9, 1.0), bands) == 6
    assert _dominant_physical_column((5.2, 0.0, 6.1, 1.0), bands) == 5
    assert _dominant_physical_column((5.5, 0.0, 6.5, 1.0), bands) is None


def test_year_plus_twelve_month_bands_require_a_source_year_row() -> None:
    source_table = {
        "rows": [["2022", *("N" for _ in range(12))]],
        "extensions": {
            "geometry": {
                "col_bands": [
                    {"index": index, "x0": float(index), "x1": float(index + 1)}
                    for index in range(13)
                ]
            }
        },
    }

    assert _year_plus_twelve_month_bands(source_table) is not None
    source_table["rows"] = [["year", *("N" for _ in range(12))]]
    assert _year_plus_twelve_month_bands(source_table) is None
