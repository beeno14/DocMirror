from __future__ import annotations

from types import SimpleNamespace

import pytest

from docmirror.plugins.credit_report.personal_detail_scanned.context import (
    _printed_reading_order,
)
from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
    collect_extraction_issues,
    dataset_states_from_issues,
    liability_record_is_substantive,
    make_issue,
    register_issue_target_remap,
)
from docmirror.plugins.credit_report.personal_detail_scanned.field_contracts import validate_pboc_field
from docmirror.plugins.credit_report.personal_detail_scanned.native_extraction import (
    _apply_account_facts,
    _credible_sequence_endpoint,
    _dedupe_liability_records,
    _extract_credit_lines,
    _extract_header_datasets,
    _extract_liabilities,
    _inquiry_sequence_endpoint,
    _record_pre_repair_source_gaps,
    _source_completeness_ledger,
    reconcile_candidate_b_credit_lines,
    reconcile_candidate_b_liabilities,
)
from docmirror.plugins.credit_report.personal_detail_scanned.native_parser import (
    PBOCPersonalDetailNativeParser,
)
from docmirror.plugins.credit_report.personal_detail_scanned.schema import (
    project_personal_detail_datasets,
)
from docmirror.plugins.credit_report.personal_detail_scanned.source_projection import (
    prepare_personal_detail_source_collections,
)
from docmirror.plugins.credit_report.personal_detail_scanned.variant import (
    PersonalDetailScannedVariant,
)
from docmirror.plugins.credit_report.value_utils import stable_record_id


def _table(table_id: str, rows: list[list[str]]) -> SimpleNamespace:
    return SimpleNamespace(
        table_id=table_id,
        metadata={"raw_rows": rows},
        headers=[],
        rows=[],
        bbox=[20, 20, 580, 300],
    )


def _liability_page(
    *tables: SimpleNamespace,
    page_number: int = 4,
    source_page_number: int = 2,
) -> SimpleNamespace:
    for table in tables:
        table.metadata["canonical_template_id"] = "repayment_responsibility"
    return SimpleNamespace(
        page_number=page_number,
        source_page_number=source_page_number,
        canonical_template_id="repayment_responsibility",
        tables=list(tables),
        texts=[],
    )


def _exact_table(
    table_id: str,
    rows: list[list[str]],
    *,
    derived_cells: set[tuple[int, int]] = frozenset(),
    canonical_template_id: str = "report_header_and_identity",
) -> SimpleNamespace:
    width = max((len(row) for row in rows), default=0)
    cell_bboxes = [
        [
            [20.0 + column * 100.0, 20.0 + row * 20.0, 120.0 + column * 100.0, 40.0 + row * 20.0]
            for column in range(width)
        ]
        for row in range(len(rows))
    ]
    statuses = [
        ["derived" if (row, column) in derived_cells else "exact" for column in range(width)]
        for row in range(len(rows))
    ]
    evidence = [
        [
            [] if (row, column) in derived_cells else [f"native:{table_id}:{row}:{column}"]
            for column in range(width)
        ]
        for row in range(len(rows))
    ]
    return SimpleNamespace(
        table_id=table_id,
        metadata={
            "raw_rows": rows,
            "canonical_template_id": canonical_template_id,
            "geometry": {
                "coordinate_system": "logical_page_pixels",
                "row_bands": [
                    {"index": row, "y0": 20.0 + row * 20.0, "y1": 40.0 + row * 20.0}
                    for row in range(len(rows))
                ],
                "col_bands": [
                    {"index": column, "x0": 20.0 + column * 100.0, "x1": 120.0 + column * 100.0}
                    for column in range(width)
                ],
                "cell_bboxes": cell_bboxes,
                "cell_geometry_status": statuses,
                "cell_evidence_ids": evidence,
                "cell_spans": [],
            },
        },
        headers=[],
        rows=[],
        bbox=[20.0, 20.0, 20.0 + width * 100.0, 20.0 + len(rows) * 20.0],
    )


def test_uncertain_cell_is_preserved_and_published_for_review() -> None:
    context = SimpleNamespace(
        ocr_correction_audit=lambda: {
            "cell_anomalies": [
                {
                    "stage": "native_business",
                    "path": "credit_accounts[0].balance",
                    "role": "amount",
                    "value": "12O0",
                    "reason_codes": ["role_validation_failed", "preserved_unresolved_value"],
                    "source_refs": [{"logical_page": 3, "source_page": 2}],
                }
            ]
        },
        page_topology_audit=lambda: {"issues": []},
    )

    issues = collect_extraction_issues(context)

    assert len(issues) == 1
    assert issues[0]["observed_value"] == "12O0"
    assert issues[0]["status"] == "requires_review"
    assert dataset_states_from_issues(issues) == {}


def test_pboc_controlled_vocabulary_is_diagnostic_and_broad() -> None:
    assert validate_pboc_field("贷后管理", "inquiry_reason").valid is True
    assert validate_pboc_field("司法调查", "inquiry_reason").valid is True
    assert validate_pboc_field("保后管理", "inquiry_reason").valid is True
    assert validate_pboc_field("公积金提取复核", "inquiry_reason").valid is True
    assert validate_pboc_field("本人查询（临柜）", "inquiry_reason").valid is True
    assert validate_pboc_field("人民币元", "currency").valid is True
    for currency in ("澳元", "加拿大元", "瑞士法郎", "新加坡元", "澳门元"):
        assert validate_pboc_field(currency, "currency").valid is True
    for currency in ("AUD", "CAD", "CHF", "SGD", "MOP"):
        assert validate_pboc_field(currency, "currency").valid is True
    assert validate_pboc_field("ZZZ", "currency").valid is False
    assert validate_pboc_field("专业技术人员", "employment_status").valid is True
    assert validate_pboc_field("中专、职高、技校", "education_level").valid is True
    assert validate_pboc_field("本人查询 (商业银行网上 银行)", "inquiry_reason").valid is True
    invalid = validate_pboc_field("贷后管埋", "inquiry_reason")
    assert invalid.assessed is True
    assert invalid.valid is False
    assert invalid.reason_code == "controlled_vocabulary_mismatch"


def test_only_row_blocking_structure_failures_degrade_dataset_presence() -> None:
    issue = make_issue(
        category="ocr_structure_correction",
        issue_code="recognized_native_section_missing_required_value",
        message="required value missing",
        target_dataset="credit_lines",
    )

    assert dataset_states_from_issues([issue])["credit_lines"]["presence_status"] == "extraction_failed"


def test_default_liability_currency_is_not_substantive_source_evidence() -> None:
    assert liability_record_is_substantive({"liability_id": "synthetic", "currency": "CNY"}) is False
    assert liability_record_is_substantive({"currency": "CNY", "responsibility_amount": 4000000}) is True


def test_native_liability_extraction_drops_header_only_tables_before_counting() -> None:
    header_only = _table(
        "header-only",
        [["责任人类型", "保证合同编号"], ["", ""], ["报告日期", "2024.07.22"]],
    )
    substantive = _table(
        "liability-1",
        [
            ["责任人类型", "保证合同编号", "还款责任金额"],
            ["保证人", "B10512900H0001C240320GR3596590", "4,000,000"],
        ],
    )
    result = SimpleNamespace(
        pages=[
            _liability_page(
                header_only,
                substantive,
                page_number=1,
                source_page_number=1,
            )
        ]
    )

    rows = _extract_liabilities(result)

    assert len(rows) == 1
    assert rows[0]["sequence"] == 1
    assert rows[0]["responsibility_amount"] == 4000000


def test_liability_dedupe_does_not_merge_transposed_contract_identifiers() -> None:
    native = {
        "contract_number": "J10257010H00012016DB20221130NS000003202",
        "related_party_id_number": "5309020000053763",
        "responsibility_amount": 700210,
        "balance": 116702,
        "due_date": "2024-11-20",
        "sequence": 4,
    }
    replay = {
        "contract_number": "00012016DBJ10257010H20221130NS000003202",
        "related_party_id_number": "5309020000053763",
        "responsibility_amount": 700210,
        "balance": 116702,
        "due_date": "2024-11-20",
        "sequence": 14,
    }
    distinct = {
        "contract_number": None,
        "related_party_id_number": "5329010002043257",
        "responsibility_amount": 700210,
        "balance": 3500000,
        "due_date": "2024-09-13",
        "sequence": 10,
    }

    rows = _dedupe_liability_records([native, replay, distinct])

    # Similar borrower/amount/date fields cannot prove that two distinct
    # guarantee-contract cells represent one business record.
    assert rows == [native, replay, distinct]
    assert [row["sequence"] for row in rows] == [1, 2, 3]


def test_complete_liability_card_preserves_exact_slots_and_snapshot_date() -> None:
    table = _complete_liability_table("liability-complete")
    result = SimpleNamespace(pages=[_liability_page(table)])

    rows = _extract_liabilities(result)

    assert len(rows) == 1
    row = rows[0]
    assert row["institution"] == "样例银行某分行"
    assert row["responsibility_amount"] == 200000
    assert row["snapshot_date"] == "2025-04-25"
    assert row["currency"] == "CNY"
    assert row["repayment_status_code"] == "N"
    assert row["overdue_months"] is None
    assert row["source_refs_by_field"]["institution"][0]["geometry_scope"] == "cell"
    assert getattr(result, "_personal_detail_extraction_issues", []) == []


def test_liability_overdue_month_label_is_not_reinterpreted_as_status_code() -> None:
    table = _complete_liability_table("liability-overdue-months", status="0")
    table.metadata["raw_rows"][-2][7] = "逾期月数"
    result = SimpleNamespace(pages=[_liability_page(table)])

    row = _extract_liabilities(result)[0]

    assert row["overdue_months"] == 0
    assert row["repayment_status_code"] is None
    assert getattr(result, "_personal_detail_extraction_issues", []) == []


def test_liability_currency_is_unknown_and_reported_instead_of_defaulted() -> None:
    table = _complete_liability_table("liability-bad-currency", currency="人民币无")
    result = SimpleNamespace(pages=[_liability_page(table)])

    rows = _extract_liabilities(result)

    assert len(rows) == 1
    assert rows[0]["currency"] is None
    assert rows[0]["reporting_amount_currency"] is None
    issues = getattr(result, "_personal_detail_extraction_issues", [])
    currency_issues = [issue for issue in issues if issue.get("field_name") == "currency"]
    assert len(currency_issues) == 1
    assert currency_issues[0]["issue_code"] == "candidate_b_repayment_responsibility_field_invalid"
    assert currency_issues[0]["target_record_id"] == rows[0]["liability_id"]


def test_liability_extended_exact_currency_is_silent() -> None:
    for index, (raw, expected) in enumerate(
        (("新加坡元", "SGD"), ("MOP", "MOP")),
        start=1,
    ):
        table = _complete_liability_table(f"liability-currency-{index}", currency=raw)
        result = SimpleNamespace(pages=[_liability_page(table)])

        rows = _extract_liabilities(result)

        assert rows[0]["currency"] == expected
        assert rows[0]["reporting_amount_currency"] == expected
        assert not any(
            issue.get("field_name") == "currency"
            for issue in getattr(result, "_personal_detail_extraction_issues", [])
        )


def test_explicit_liability_placeholder_is_known_absence_not_failure() -> None:
    table = _complete_liability_table("liability-explicit-absence", currency="--")
    result = SimpleNamespace(pages=[_liability_page(table)])

    rows = _extract_liabilities(result)

    assert rows[0]["currency"] is None
    assert not any(
        issue.get("field_name") == "currency"
        for issue in getattr(result, "_personal_detail_extraction_issues", [])
    )


def test_equal_provenance_liability_conflict_is_withheld_and_linked() -> None:
    first = _complete_liability_table("liability-conflict-a", responsibility_amount="200,000", top=20)
    second = _complete_liability_table("liability-conflict-b", responsibility_amount="300,000", top=260)
    result = SimpleNamespace(pages=[_liability_page(first, second)])

    rows = _extract_liabilities(result)

    assert len(rows) == 1
    assert rows[0]["responsibility_amount"] is None
    conflicts = [
        issue
        for issue in getattr(result, "_personal_detail_extraction_issues", [])
        if issue.get("issue_code") == "candidate_b_repayment_responsibility_field_conflict"
        and issue.get("field_name") == "responsibility_amount"
    ]
    assert len(conflicts) == 1
    assert conflicts[0]["target_record_id"] == rows[0]["liability_id"]


def test_corrected_cell_slot_outranks_native_liability_observation() -> None:
    cell_ref = {
        "source": "corrected",
        "logical_page": 4,
        "source_page": 2,
        "bbox": [10, 10, 30, 20],
        "geometry_scope": "cell",
    }
    native = {
        "contract_number": "B31015210H0001AHIH0001",
        "responsibility_amount": 200000,
        "source_refs": [cell_ref],
        "source_refs_by_field": {"responsibility_amount": [cell_ref]},
        "_field_binding_quality": {"responsibility_amount": "native_label_column"},
        "confidence": 0.99,
    }
    corrected = {
        "contract_number": "B31015210H0001AHIH0001",
        "responsibility_amount": 300000,
        "source_refs": [cell_ref],
        "source_refs_by_field": {"responsibility_amount": [cell_ref]},
        "_field_binding_quality": {"responsibility_amount": "canonical_cell_slot"},
        "confidence": 0.90,
    }
    result = SimpleNamespace()

    rows = reconcile_candidate_b_liabilities(result, [native, corrected])

    assert rows[0]["responsibility_amount"] == 300000
    assert not any(
        issue.get("field_name") == "responsibility_amount"
        for issue in getattr(result, "_personal_detail_extraction_issues", [])
    )


def test_huaneng_complementary_liability_observations_merge_by_account_identity() -> None:
    common = {
        "institution": "华能贵诚信托有限公司",
        "open_date": "2022-09-02",
        "due_date": "2024-09-07",
        "related_party_name": "厦门奕翔祥商贸有限公司",
        "related_party_id_type": "中征码",
        "related_party_id_number": "3502030011973425",
        "balance": 46667,
        "_party_category": "organization",
        "source_refs": [],
        "confidence": 0.98,
    }
    first = {**common, "responsibility_type": "保证人", "responsibility_amount": 56000}
    second = {**common, "business_type": "企业经营贷款", "five_tier_class": "正常", "overdue_months": 0}
    result = SimpleNamespace()

    rows = reconcile_candidate_b_liabilities(result, [first, second])

    assert len(rows) == 1
    assert rows[0]["responsibility_amount"] == 56000
    assert rows[0]["five_tier_class"] == "正常"
    assert rows[0]["overdue_months"] == 0
    assert rows[0]["related_party_category"] == "organization"


def test_cross_plane_liability_complements_merge_only_inside_native_card_geometry() -> None:
    shared = {
        "business_type": "个人经营性贷款",
        "open_date": "2023-12-04",
        "due_date": "2024-10-04",
        "responsibility_type": "保证人",
        "responsibility_amount": 3500000,
        "currency": "CNY",
        "_party_category": "person",
        "confidence": 0.98,
    }
    native = {
        **shared,
        "institution": "云南芒市农村商业银行股份有限公司",
        "contract_number": "G10117300H03991101011630211202230000011",
        "source_refs": [
            {
                "source": "native_detail_table",
                "source_page": 9,
                "table_id": "pt_18_2",
                "bbox": [20, 100, 400, 300],
            }
        ],
    }
    corrected = {
        **shared,
        "related_party_name": "杨天云",
        "snapshot_date": "2024-07-21",
        "balance": 2000000,
        "repayment_status_code": "N",
        "source_refs": [
            {
                "source": "personal_detail_corrected_page_cell",
                "source_page": 9,
                "bbox": [120, 160, 220, 190],
            }
        ],
    }

    rows = reconcile_candidate_b_liabilities(SimpleNamespace(), [native, corrected])

    assert len(rows) == 1
    assert rows[0]["contract_number"] == native["contract_number"]
    assert rows[0]["institution"] == native["institution"]
    assert rows[0]["snapshot_date"] == "2024-07-21"
    assert rows[0]["balance"] == 2000000


def test_cross_plane_liability_complement_does_not_bridge_distinct_card_geometry() -> None:
    shared = {
        "business_type": "企业经营贷款",
        "open_date": "2021-06-30",
        "due_date": "2026-06-30",
        "responsibility_type": "保证人",
        "responsibility_amount": 5500000,
        "currency": "CNY",
        "_party_category": "organization",
        "confidence": 0.98,
    }
    native_a = {
        **shared,
        "institution": "中国银行股份有限公司大理州分行",
        "contract_number": "CONTRACT-A",
        "source_refs": [
            {
                "source": "native_detail_table",
                "source_page": 10,
                "table_id": "pt_19_3",
                "bbox": [20, 100, 400, 220],
            }
        ],
    }
    native_b = {
        **shared,
        "institution": "中国银行股份有限公司大理州分行",
        "contract_number": "CONTRACT-B",
        "source_refs": [
            {
                "source": "native_detail_table",
                "source_page": 10,
                "table_id": "pt_19_4",
                "bbox": [20, 260, 400, 380],
            }
        ],
    }
    corrected = {
        **shared,
        "related_party_name": "大理某企业有限公司",
        "balance": 5500000,
        "overdue_months": 0,
        "source_refs": [
            {
                "source": "personal_detail_corrected_page_cell",
                "source_page": 10,
                "bbox": [100, 145, 220, 175],
            }
        ],
    }

    rows = reconcile_candidate_b_liabilities(
        SimpleNamespace(), [native_b, native_a, corrected]
    )

    assert len(rows) == 2
    assert {row.get("contract_number") for row in rows} == {
        "CONTRACT-A",
        "CONTRACT-B",
    }
    completed = next(row for row in rows if row.get("contract_number") == "CONTRACT-A")
    assert completed["balance"] == 5500000


def test_same_liability_ordinal_never_overrides_distinct_contracts() -> None:
    common = {
        "_printed_sequence": 1,
        "_party_category": "organization",
        "source_refs": [],
        "confidence": 0.98,
    }

    rows = reconcile_candidate_b_liabilities(
        SimpleNamespace(),
        [
            {**common, "contract_number": "CONTRACT-ONE"},
            {**common, "contract_number": "CONTRACT-TWO"},
        ],
    )

    assert len(rows) == 2


def _lin_liability_candidate(
    sequence: int,
    *,
    institution: str,
    contract_number: str,
    party_id_type: str,
    party_id_number: str,
    balance: str,
    status_label: str = "逾期月数",
    status: str = "0",
) -> SimpleNamespace:
    fields = {
        "__printed_sequence": str(sequence),
        "__party_category": "organization",
        "管理机构": institution,
        "业务种类": "企业经营贷款",
        "开立日期": "2022.09.02",
        "到期日期": "2024.09.07",
        "责任人类型": "保证人",
        "还款责任金额": "56000",
        "币种": "人民币元",
        "保证合同编号": contract_number,
        "主业务借款人": "厦门奕翔祥商贸有限公司",
        "主业务借款人证件类型": party_id_type,
        "主业务借款人证件号码": party_id_number,
        "__snapshot_date": "截至2023年1月13日",
        "余额": balance,
        "五级分类": "正常",
        status_label: status,
    }
    field_ref = {
        "logical_page": 23,
        "source_page": 12,
        "bbox": [20.0, float(sequence * 100), 240.0, float(sequence * 100 + 18)],
        "geometry_scope": "cell",
    }
    printed_labels = {key for key in fields if not key.startswith("__")}
    return SimpleNamespace(
        fields=fields,
        observed_labels=frozenset(printed_labels),
        unresolved_labels=frozenset(),
        source_refs=(field_ref,),
        source_refs_by_field={label: (field_ref,) for label in printed_labels},
        binding_quality_by_field={label: "canonical_cell_slot" for label in printed_labels},
        confidence=0.98,
    )


def test_lin_party_id_contracts_corroborate_contaminated_zhongzheng_code_and_preserve_absence(
    monkeypatch,
) -> None:
    candidates = [
        _lin_liability_candidate(
            1,
            institution="梅赛德斯-奔驰汽车金融有限公司",
            contract_number="LINCONTRACT0001",
            party_id_type="中征码",
            party_id_number="3502030011973425 2",
            balance="348791",
        ),
        _lin_liability_candidate(
            2,
            institution="深圳前海微众银行股份有限公司",
            contract_number="LINCONTRACT0002",
            party_id_type="统一社会信用代码",
            party_id_number="91350203MA33H1DP8L",
            balance="258666",
        ),
        _lin_liability_candidate(
            3,
            institution="华能贵诚信托有限公司",
            contract_number="LINCONTRACT0003",
            party_id_type="中征码",
            party_id_number="3502030011973425",
            balance="46667",
            status="--",
        ),
    ]
    monkeypatch.setattr(PBOCPersonalDetailNativeParser, "records", lambda _self, _dataset: candidates)
    result = SimpleNamespace()

    rows = _extract_liabilities(result)

    assert len(rows) == 3
    assert rows[0]["related_party_id_number"] == "3502030011973425"
    assert rows[0]["canonical_raw"]["related_party_id_number"] == "3502030011973425 2"
    assert rows[1]["related_party_id_type"] == "统一社会信用代码"
    assert rows[1]["related_party_id_number"] == "91350203MA33H1DP8L"
    assert rows[2]["overdue_months"] is None
    assert "overdue_months" in rows[2]["_source_absent_fields"]
    assert rows[2]["canonical_raw"]["overdue_months"] == "--"
    issues = getattr(result, "_personal_detail_extraction_issues", [])
    correction = next(
        issue for issue in issues if issue["issue_code"] == "candidate_b_liability_party_id_corroborated"
    )
    assert correction["status"] == "resolved"
    assert correction["observed_value"] == ["3502030011973425 2"]
    assert correction["candidate_value"] == "3502030011973425"
    assert not any(
        issue.get("field_name") in {"overdue_months", "repayment_status_code"} for issue in issues
    )


@pytest.mark.parametrize(
    "party_id_number",
    ("91350203MA33H1DP8", "91350203ma33h1dp8l"),
)
def test_unified_social_credit_code_requires_exact_18_uppercase_alphanumeric(
    monkeypatch,
    party_id_number: str,
) -> None:
    candidate = _lin_liability_candidate(
        1,
        institution="深圳前海微众银行股份有限公司",
        contract_number="INVALIDUSCC0001",
        party_id_type="统一社会信用代码",
        party_id_number=party_id_number,
        balance="258666",
    )
    monkeypatch.setattr(PBOCPersonalDetailNativeParser, "records", lambda _self, _dataset: [candidate])
    result = SimpleNamespace()

    row = _extract_liabilities(result)[0]

    assert row["related_party_id_number"] is None
    issue = next(
        issue
        for issue in result._personal_detail_extraction_issues
        if issue.get("field_name") == "related_party_id_number"
    )
    assert issue["issue_code"] == "candidate_b_liability_party_id_corroboration_unresolved"
    assert issue["observed_value"] == [party_id_number]


def test_liability_projection_withholds_type_invalid_party_identifier() -> None:
    projected = project_personal_detail_datasets(
        {
            "repayment_liability_records": [
                {
                    "liability_id": "liability:invalid-id",
                    "related_party_category": "organization",
                    "related_party_id_type": "中征码",
                    "related_party_id_number": "35020300119734252",
                }
            ]
        }
    )
    values = projected["repayment_responsibilities"][0].get(
        "normalized", projected["repayment_responsibilities"][0]
    )

    assert "related_party_id_number" not in values
    assert values["source_related_party_id_number"] == "35020300119734252"
    assert values["extraction_status"] == "review"


def test_contaminated_zhongzheng_code_is_withheld_without_corroboration(monkeypatch) -> None:
    candidate = _lin_liability_candidate(
        1,
        institution="梅赛德斯-奔驰汽车金融有限公司",
        contract_number="NOCORROBORATION0001",
        party_id_type="中征码",
        party_id_number="3502030011973425 2",
        balance="348791",
    )
    monkeypatch.setattr(PBOCPersonalDetailNativeParser, "records", lambda _self, _dataset: [candidate])
    result = SimpleNamespace()

    row = _extract_liabilities(result)[0]

    assert row["related_party_id_number"] is None
    issue = next(
        issue
        for issue in result._personal_detail_extraction_issues
        if issue.get("field_name") == "related_party_id_number"
    )
    assert "candidate_value" not in issue
    assert "independent_party_identifier_missing" in issue["reason_codes"]


def test_contaminated_zhongzheng_code_is_withheld_when_corroborators_disagree(monkeypatch) -> None:
    candidates = [
        _lin_liability_candidate(
            1,
            institution="梅赛德斯-奔驰汽车金融有限公司",
            contract_number="AMBIGUOUS0001",
            party_id_type="中征码",
            party_id_number="3502030011973425 2",
            balance="348791",
        ),
        _lin_liability_candidate(
            2,
            institution="某信托有限公司",
            contract_number="AMBIGUOUS0002",
            party_id_type="中征码",
            party_id_number="3502030011973425",
            balance="46667",
        ),
        _lin_liability_candidate(
            3,
            institution="某银行股份有限公司",
            contract_number="AMBIGUOUS0003",
            party_id_type="中征码",
            party_id_number="3502030011973426",
            balance="56667",
        ),
    ]
    monkeypatch.setattr(PBOCPersonalDetailNativeParser, "records", lambda _self, _dataset: candidates)
    result = SimpleNamespace()

    rows = _extract_liabilities(result)

    assert rows[0]["related_party_id_number"] is None
    issue = next(
        issue
        for issue in result._personal_detail_extraction_issues
        if issue.get("target_record_id") == rows[0]["liability_id"]
        and issue.get("field_name") == "related_party_id_number"
    )
    assert issue["candidate_value"] == ["3502030011973425", "3502030011973426"]
    assert "ambiguous_independent_party_identifiers" in issue["reason_codes"]


def test_similar_liabilities_with_distinct_source_identity_remain_distinct() -> None:
    common = {
        "institution": "某银行股份有限公司",
        "open_date": "2020-01-02",
        "due_date": "2030-01-02",
        "related_party_name": "华能某能源有限公司",
        "balance": 800000,
        "_party_category": "organization",
        "source_refs": [],
        "confidence": 0.98,
    }
    first = {**common, "_printed_sequence": 1, "contract_number": "CONTRACT0001"}
    second = {**common, "_printed_sequence": 2, "contract_number": "CONTRACT0002"}

    rows = reconcile_candidate_b_liabilities(SimpleNamespace(), [first, second])

    assert len(rows) == 2
    assert [row["_printed_sequence"] for row in rows] == [1, 2]


def test_liability_projection_preserves_heading_category_and_overdue_month_variant() -> None:
    projected = project_personal_detail_datasets(
        {
            "repayment_liability_records": [
                {
                    "liability_id": "liability:enterprise:1",
                    "related_party_category": "organization",
                    "overdue_months": 0,
                }
            ]
        }
    )
    values = projected["repayment_responsibilities"][0].get(
        "normalized",
        projected["repayment_responsibilities"][0],
    )

    assert values["related_party_category"] == "organization"
    assert values["overdue_months"] == 0
    assert values.get("repayment_status_code") in (None, "")


def test_credit_line_limits_are_schema_typed_amounts() -> None:
    from docmirror.plugins.credit_report.personal_detail_scanned.ocr_correction import (
        _is_valid_for_role,
        _mapping_role,
        _normalize_amount,
        normalize_institution_name,
    )

    assert _mapping_role({}, "total_limit") == "amount"
    assert _mapping_role({}, "used_limit") == "amount"
    assert _normalize_amount("00") == "0"
    assert _is_valid_for_role("M10255810H0001GALC-HL-2107233906-1", "account_identifier")
    assert normalize_institution_name("河南中原消费金融股份 有限公司") == "河南中原消费金融股份有限公司"


def test_credit_agreement_does_not_invent_active_status(monkeypatch) -> None:
    from docmirror.plugins.credit_report.personal_detail_scanned.native_parser import (
        PBOCPersonalDetailNativeParser,
    )

    candidate = SimpleNamespace(
        fields={
            "授信协议标识": "T10151210H0001ABC12345",
            "管理机构": "示例银行",
            "授信额度用途": "循环额度",
            "生效日期": "2020.01.01",
            "到期日期": "2024.01.01",
            "授信额度": "100000",
            "授信限额": "100000",
            "已用额度": "0",
            "授信限额编号": "L10151210H0001ABC12345",
            "币种": "人民币元",
        },
        source_refs=(),
        source_refs_by_field={},
        binding_quality_by_field={},
        observed_labels=frozenset(),
        unresolved_labels=frozenset(),
        confidence=0.98,
    )
    monkeypatch.setattr(
        PBOCPersonalDetailNativeParser,
        "records",
        lambda _self, dataset_name: [candidate] if dataset_name == "credit_lines" else [],
    )

    rows = _extract_credit_lines(SimpleNamespace())

    assert len(rows) == 1
    assert "status" not in rows[0]


@pytest.mark.parametrize(
    ("raw", "expected_status", "expected_lifecycle", "expected_resolution"),
    (
        ("正 常", "active", "open", "resolved"),
        ("结清", "settled", "settled", "resolved"),
        ("结 请", "settled", "settled", "ocr_noise_normalized"),
        ("结 消", "settled", "settled", "ocr_noise_normalized"),
        ("未 激 活", "inactive", "open", "resolved"),
        ("银行止付", "suspended", None, "resolved"),
    ),
)
def test_account_status_publishes_only_a_complete_registered_cell(
    raw: str,
    expected_status: str,
    expected_lifecycle: str | None,
    expected_resolution: str,
) -> None:
    context = SimpleNamespace()
    account = {"account_id": f"credit_account:status:{raw}"}
    table = _table("account-status-exact", [["账户状态"], [raw]])

    _apply_account_facts(
        context,
        account,
        table.metadata["raw_rows"],
        page=SimpleNamespace(page_number=6, source_page_number=3),
        table=table,
    )

    assert account["account_status"] == expected_status
    assert account.get("account_lifecycle_state") == expected_lifecycle
    assert account["account_status_raw"] == raw.replace(" ", "")
    assert account["account_status_resolution"] == expected_resolution
    assert collect_extraction_issues(context) == []


@pytest.mark.parametrize(
    "raw",
    (
        "非正常",
        "未销户",
        "正常1",
        "正常附",
        "X正常",
        "正常。",
        "结清销户",
        "R结清",
    ),
)
def test_account_status_negation_or_residue_is_withheld_with_field_issue(raw: str) -> None:
    context = SimpleNamespace()
    account = {"account_id": f"credit_account:status-invalid:{raw}"}
    table = _table("account-status-invalid", [["账户状态"], [raw]])

    _apply_account_facts(
        context,
        account,
        table.metadata["raw_rows"],
        page=SimpleNamespace(page_number=6, source_page_number=3),
        table=table,
    )

    for field_name in (
        "account_status",
        "account_status_raw",
        "account_status_resolution",
        "account_lifecycle_state",
        "card_activation_state",
        "credit_quality_status",
        "current_overdue",
        "current_overdue_status",
    ):
        assert field_name not in account
    assert account["canonical_raw"]["account_status"] == [raw]
    assert "account_status" in account["_unresolved_fields"]
    issues = collect_extraction_issues(context)
    assert len(issues) == 1
    assert issues[0]["issue_code"] == "candidate_b_exact_slot_value_invalid"
    assert issues[0]["target_record_id"] == account["account_id"]
    assert issues[0]["field_name"] == "account_status"
    assert issues[0]["observed_value"] == [raw]
    assert issues[0]["status"] == "requires_review"
    assert "normalized_value_withheld" in issues[0]["reason_codes"]


def test_account_status_sentence_does_not_bypass_the_whole_cell_contract() -> None:
    context = SimpleNamespace()
    account = {"account_id": "credit_account:status-sentence-invalid"}
    raw = "非正常"
    table = _table("account-status-sentence-invalid", [[f"截至2024年1月1日，账户状态为“{raw}”"]])

    _apply_account_facts(
        context,
        account,
        table.metadata["raw_rows"],
        page=SimpleNamespace(page_number=6, source_page_number=3),
        table=table,
    )

    assert "account_status" not in account
    assert "account_lifecycle_state" not in account
    assert account["canonical_raw"]["account_status"] == [raw]
    issues = collect_extraction_issues(context)
    assert len(issues) == 1
    assert issues[0]["issue_code"] == "candidate_b_exact_slot_value_invalid"
    assert issues[0]["field_name"] == "account_status"
    assert issues[0]["source_refs"][0]["binding"] == "canonical_account_status_sentence"
    assert issues[0]["status"] == "requires_review"


def test_account_currency_alias_with_any_substantive_residue_is_withheld_and_active() -> None:
    examples = (
        "人民币元 共同借款标志",
        "美元X",
        "非人民币",
        "美元。",
        "美元1",
        "福 澳元 第",
    )
    for index, raw in enumerate(examples, start=1):
        context = SimpleNamespace()
        account = {"account_id": f"credit_account:loan:{index}"}
        table = _table(f"account-currency-{index}", [["账户币种"], [raw]])
        page = SimpleNamespace(page_number=6, source_page_number=3)

        _apply_account_facts(
            context,
            account,
            table.metadata["raw_rows"],
            page=page,
            table=table,
        )

        assert "currency" not in account
        assert "account_currency" not in account
        assert account["canonical_raw"]["currency"] == [raw]
        issues = collect_extraction_issues(context)
        assert len(issues) == 1
        assert issues[0]["issue_code"] == "candidate_b_exact_slot_value_invalid"
        assert issues[0]["target_record_id"] == account["account_id"]
        assert issues[0]["field_name"] == "currency"
        assert issues[0]["observed_value"] == [raw]
        assert issues[0]["status"] == "requires_review"
        assert "normalized_value_withheld" in issues[0]["reason_codes"]


def test_account_exact_finite_currency_token_is_silent() -> None:
    examples = (
        ("人 民 币 元", "CNY"),
        ("人民币", "CNY"),
        ("RMB", "CNY"),
        ("rmb", "CNY"),
        ("美元", "USD"),
        ("usd", "USD"),
        ("港元", "HKD"),
        ("澳元", "AUD"),
        ("加拿大元", "CAD"),
        ("瑞士法郎", "CHF"),
        ("新加坡元", "SGD"),
        ("澳门元", "MOP"),
        ("AUD", "AUD"),
        ("CAD", "CAD"),
        ("CHF", "CHF"),
        ("SGD", "SGD"),
        ("MOP", "MOP"),
        ("银", "XAG"),
    )
    for index, (raw, expected) in enumerate(examples, start=1):
        context = SimpleNamespace()
        account = {"account_id": f"credit_account:loan:{index}"}
        table = _table(f"account-currency-{index}", [["账户币种"], [raw]])
        page = SimpleNamespace(page_number=6, source_page_number=3)

        _apply_account_facts(
            context,
            account,
            table.metadata["raw_rows"],
            page=page,
            table=table,
        )

        assert account["currency"] == expected
        assert account["account_currency"] == expected
        assert collect_extraction_issues(context) == []


def test_account_single_han_currency_alias_does_not_match_institution_prose() -> None:
    context = SimpleNamespace()
    account = {"account_id": "credit_account:credit_card:20"}
    raw = "中国工商银行 B10111000H 豫"
    table = _table("account-currency-embedded-silver", [["账户币种"], [raw]])
    page = SimpleNamespace(page_number=23, source_page_number=12)

    _apply_account_facts(
        context,
        account,
        table.metadata["raw_rows"],
        page=page,
        table=table,
    )

    assert "currency" not in account
    assert "account_currency" not in account
    issues = collect_extraction_issues(context)
    assert len(issues) == 1
    assert issues[0]["issue_code"] == "candidate_b_exact_slot_value_invalid"
    assert issues[0]["observed_value"] == [raw]
    assert "normalized_value_withheld" in issues[0]["reason_codes"]


def test_account_currency_unknown_or_multiple_tokens_are_withheld_and_reported() -> None:
    for index, raw in enumerate(("ZZZ", "人民币元美元", "澳元美元"), start=1):
        context = SimpleNamespace()
        account = {"account_id": f"credit_account:loan:{index}"}
        table = _table(f"account-currency-{index}", [["账户币种"], [raw]])
        page = SimpleNamespace(page_number=6, source_page_number=3)

        _apply_account_facts(
            context,
            account,
            table.metadata["raw_rows"],
            page=page,
            table=table,
        )

        assert "currency" not in account
        assert "account_currency" not in account
        issues = collect_extraction_issues(context)
        assert len(issues) == 1
        assert issues[0]["issue_code"] == "candidate_b_exact_slot_value_invalid"
        assert issues[0]["target_record_id"] == account["account_id"]


def test_credit_agreement_currency_residue_is_withheld_and_field_reported(monkeypatch) -> None:
    candidate = SimpleNamespace(
        fields={
            "授信协议标识": "T10151210H0001ABC12345",
            "管理机构": "示例银行",
            "授信额度用途": "循环额度",
            "生效日期": "2020.01.01",
            "到期日期": "2024.01.01",
            "授信额度": "100000",
            "授信限额": "100000",
            "已用额度": "0",
            "授信限额编号": "L10151210H0001ABC12345",
            "币种": "美元 江",
        },
        source_refs=({"logical_page": 7},),
        source_refs_by_field={"币种": ({"logical_page": 7, "column": 4},)},
        binding_quality_by_field={"币种": "canonical_header_column"},
        observed_labels=frozenset({"币种"}),
        unresolved_labels=frozenset(),
        confidence=0.98,
    )
    monkeypatch.setattr(
        PBOCPersonalDetailNativeParser,
        "records",
        lambda _self, dataset_name: [candidate] if dataset_name == "credit_lines" else [],
    )
    context = SimpleNamespace()

    rows = _extract_credit_lines(context)
    rows = reconcile_candidate_b_credit_lines(context, rows)

    assert rows[0]["currency"] is None
    assert rows[0]["account_currency"] is None
    assert rows[0]["reporting_amount_currency"] is None
    assert "currency" in rows[0]["_unresolved_fields"]
    issues = [
        issue
        for issue in collect_extraction_issues(context)
        if issue["issue_code"] == "candidate_b_credit_agreement_currency_unresolved"
    ]
    assert len(issues) == 1
    assert issues[0]["target_record_id"] == rows[0]["credit_line_id"]
    assert issues[0]["observed_value"] == "美元 江"
    assert issues[0]["status"] == "requires_review"
    assert "normalized_value_withheld" in issues[0]["reason_codes"]


def test_credit_agreement_exact_extended_currency_is_silent_and_final(monkeypatch) -> None:
    candidate = SimpleNamespace(
        fields={
            "授信协议标识": "T10151210H0001MOP12345",
            "币种": "澳门元",
        },
        source_refs=({"logical_page": 7},),
        source_refs_by_field={"币种": ({"logical_page": 7, "column": 4},)},
        binding_quality_by_field={"币种": "canonical_header_column"},
        observed_labels=frozenset({"币种"}),
        unresolved_labels=frozenset(),
        confidence=0.98,
    )
    monkeypatch.setattr(
        PBOCPersonalDetailNativeParser,
        "records",
        lambda _self, dataset_name: [candidate] if dataset_name == "credit_lines" else [],
    )
    context = SimpleNamespace()

    rows = reconcile_candidate_b_credit_lines(context, _extract_credit_lines(context))

    assert rows[0]["currency"] == "MOP"
    assert rows[0]["account_currency"] == "MOP"
    assert rows[0]["reporting_amount_currency"] == "MOP"
    assert not any(
        issue.get("field_name") in {"currency", "account_currency", "reporting_amount_currency"}
        for issue in collect_extraction_issues(context)
    )


def test_credit_agreement_multiple_or_arbitrary_currency_codes_are_withheld(monkeypatch) -> None:
    for raw in ("人民币元美元", "澳元美元", "非人民币", "美元X", "美元。", "美元1", "ZZZ"):
        candidate = SimpleNamespace(
            fields={
                "授信协议标识": f"T10151210H0001{raw.encode().hex().upper()}",
                "币种": raw,
            },
            source_refs=({"logical_page": 7},),
            source_refs_by_field={"币种": ({"logical_page": 7, "column": 4},)},
            binding_quality_by_field={"币种": "canonical_header_column"},
            observed_labels=frozenset({"币种"}),
            unresolved_labels=frozenset(),
            confidence=0.98,
        )
        monkeypatch.setattr(
            PBOCPersonalDetailNativeParser,
            "records",
            lambda _self, dataset_name, row=candidate: [row]
            if dataset_name == "credit_lines"
            else [],
        )
        context = SimpleNamespace()

        rows = _extract_credit_lines(context)
        rows = reconcile_candidate_b_credit_lines(context, rows)

        assert rows[0]["currency"] is None
        assert rows[0]["account_currency"] is None
        assert rows[0]["reporting_amount_currency"] is None
        assert "currency" in rows[0]["_unresolved_fields"]
        issues = [
            issue
            for issue in collect_extraction_issues(context)
            if issue["issue_code"] == "candidate_b_credit_agreement_currency_unresolved"
        ]
        assert len(issues) == 1
        assert issues[0]["target_record_id"] == rows[0]["credit_line_id"]
        assert not issues[0].get("source_refs")


def test_final_grid_count_repairs_only_zero_expected_count() -> None:
    content = prepare_personal_detail_source_collections(
        {
            "facts": {
                "personal_detail_expected_repayment_records_count": 0,
                "personal_detail_expected_credit_lines_count": 5,
            },
            "datasets": {},
        },
        final_dataset_counts={"repayment_records": 583, "credit_lines": 4},
    )

    assert content["facts"]["personal_detail_expected_repayment_records_count"] == 583
    assert content["facts"]["personal_detail_expected_credit_lines_count"] == 5


def test_source_sequence_ledger_reports_partial_datasets_without_inventing_rows() -> None:
    content = prepare_personal_detail_source_collections(
        {
            "facts": {
                "personal_detail_source_completeness_ledger": {
                    "sequence_endpoints": {"employment_records": 5},
                    "credit_accounts": 2,
                }
            },
            "datasets": {
                "employment_records": [
                    {"record_id": f"employment:{sequence}", "sequence": sequence}
                    for sequence in (1, 2, 3, 5)
                ],
                "inquiry_records": [
                    {
                        "record_id": f"inquiry:{sequence}",
                        "sequence": sequence,
                        "inquiry_type": "institution",
                    }
                    for sequence in (1, 3)
                ],
                "credit_accounts": [{"record_id": "account:1", "account_id": "account:1"}],
            },
        },
        final_dataset_counts={"credit_accounts": 1},
    )

    statuses = {
        row["dataset_name"]: row
        for row in content["datasets"]["personal_detail_dataset_status"]
    }
    assert statuses["employment_records"]["presence_status"] == "partial"
    assert statuses["employment_records"]["expected_row_count"] == 5
    assert statuses["inquiry_records"]["presence_status"] == "partial"
    assert statuses["inquiry_records"]["expected_row_count"] == 3
    assert statuses["credit_accounts"]["presence_status"] == "partial"
    assert statuses["credit_accounts"]["expected_row_count"] == 2
    assert len(content["datasets"]["employment_records"]) == 4
    assert len(content["datasets"]["inquiry_records"]) == 2
    assert {
        issue["target_dataset"]
        for issue in content["datasets"]["personal_detail_extraction_issues"]
        if issue["issue_code"] == "source_sequence_or_count_gap"
    } == {"employment_records", "inquiry_records", "credit_accounts"}


def test_source_ledger_treats_spaced_duplicate_sequence_digits_as_one_number() -> None:
    table = _table(
        "employment",
        [
            ["编号", "工作单位", "单位性质", "进入本单位年份"],
            ["子 2 2", "示例公司", "民营企业", "2024"],
        ],
    )
    result = SimpleNamespace(
        pages=[SimpleNamespace(page_number=1, source_page_number=1, tables=[table])],
        corrected_evidence_pages=lambda: [],
    )

    ledger = _source_completeness_ledger(result)

    assert ledger["sequence_endpoints"]["employment_records"] == 2


def test_inquiry_source_ledger_counts_withheld_rows_and_headerless_continuations() -> None:
    _institutional = _table(
        "institutional-1",
        [
            ["编号", "查询日期", "查询机构", "查询原因"],
            ["1", "2024.01.03", "示例银行", "贷款审批"],
            ["3", "", "", ""],
        ],
    )


def _complete_liability_table(
    table_id: str,
    *,
    responsibility_amount: str = "200,000",
    currency: str = "人民币元",
    status: str = "N",
    contract_number: str = "B31015210H0001AHIH0001",
    top: float = 20.0,
) -> SimpleNamespace:
    rows = [
        ["账户1", "", "", "", "", "", "", "", "", ""],
        ["管理机构", "业务种类", "开立日期", "", "到期日期", "责任人类型", "还款责任金额", "", "币种", "保证合同编号"],
        ["样例银行某分行", "个人经营性贷款", "2020.05.15", "", "2035.05.14", "保证", responsibility_amount, "", currency, contract_number],
        ["主业务借款人", "", "", "主业务借款人证件类型", "", "", "", "主业务借款人证件号码", "", ""],
        ["张三", "", "", "身份证", "", "", "", "110109198312030000", "", ""],
        ["截至2025 年4 月25 日", "", "", "", "", "", "", "", "", ""],
        ["余额", "", "", "五级分类", "", "", "", "还款状态", "", ""],
        ["100,000", "", "", "正常", "", "", "", status, "", ""],
    ]
    cell_bboxes = [
        [[20 + column * 55, top + row * 24, 70 + column * 55, top + row * 24 + 18] for column in range(10)]
        for row in range(len(rows))
    ]
    return SimpleNamespace(
        table_id=table_id,
        metadata={"raw_rows": rows, "source_cell_bboxes": cell_bboxes},
        headers=[],
        rows=[],
        bbox=[20, top, 570, top + len(rows) * 24],
    )
    continuation = _table(
        "institutional-2",
        [["4", "2024.01.01", "另一银行", "贷后管理"]],
    )
    personal = _table(
        "personal",
        [
            ["编号", "查询日期", "查询机构", "查询原因"],
            ["1", "2023.12.03", "本人", "本人查询"],
            ["2", "", "", ""],
        ],
    )
    pages = [
        SimpleNamespace(
            page_number=20,
            source_page_number=10,
            canonical_template_id="annotations_and_inquiries",
            tables=[institutional],
        ),
        SimpleNamespace(
            page_number=21,
            source_page_number=11,
            canonical_template_id="annotations_and_inquiries",
            tables=[continuation],
        ),
        SimpleNamespace(
            page_number=22,
            source_page_number=12,
            canonical_template_id="annotations_and_inquiries",
            tables=[personal],
        ),
    ]
    result = SimpleNamespace(pages=pages, corrected_evidence_pages=lambda: [])

    ledger = _source_completeness_ledger(result)

    assert ledger["inquiry_records"] == 6
    assert ledger["inquiry_sequence_endpoints"] == {"institution": 4, "personal": 2}
    assert ledger["inquiry_observed_sequences"] == {
        "institution": [1, 3, 4],
        "personal": [1, 2],
    }
    assert {ref["logical_page"] for ref in ledger["source_refs"]["inquiry_records"]} == {
        20,
        21,
        22,
    }


def test_inquiry_endpoint_trusts_sparse_exact_ordinals_but_rejects_proven_prefix_bleed() -> None:
    endpoint, outliers = _inquiry_sequence_endpoint(
        {1, 88, 89, 90, 117, 789},
        {789},
    )

    assert endpoint == 117
    assert outliers == [789]
    assert _inquiry_sequence_endpoint({17, 117}, ()) == (117, [])
    assert _inquiry_sequence_endpoint({89, 789}, ()) == (789, [])
    assert _inquiry_sequence_endpoint({788, 789, 790}, ()) == (790, [])


def test_inquiry_source_gap_is_repair_eligible_without_orphan_record_target() -> None:
    table = _exact_table(
        "institutional",
        [
            ["编号", "查询日期", "查询机构", "查询原因"],
            ["1", "2024.01.03", "示例银行", "贷款审批"],
            ["3", "", "", ""],
        ],
        canonical_template_id="annotations_and_inquiries",
    )
    result = SimpleNamespace(
        pages=[
            SimpleNamespace(
                page_number=20,
                source_page_number=10,
                canonical_template_id="annotations_and_inquiries",
                tables=[table],
            )
        ],
        corrected_evidence_pages=lambda: [],
    )

    _record_pre_repair_source_gaps(
        result,
        {"inquiry_records": [{"inquiry_id": "credit_inquiry:institution:1"}]},
    )

    issue = next(
        row
        for row in result._personal_detail_extraction_issues
        if row.get("target_dataset") == "inquiry_records"
    )
    assert issue["candidate_value"]["source_expected_row_count"] == 3
    assert issue["candidate_value"]["source_sequence_endpoints"] == {"institution": 3}
    assert "target_record_id" not in issue
    assert "schema_triggered_page_repair_eligible" in issue["reason_codes"]
    assert issue["source_refs"][0]["logical_page"] == 20


def test_projection_prefers_independent_inquiry_endpoints_over_emitted_rows() -> None:
    content = prepare_personal_detail_source_collections(
        {
            "facts": {
                "personal_detail_source_completeness_ledger": {
                    "inquiry_records": 6,
                    "inquiry_sequence_endpoints": {"institution": 4, "personal": 2},
                    "source_refs": {
                        "inquiry_records": [
                            {
                                "logical_page": 20,
                                "source_page": 10,
                                "geometry_scope": "table",
                            }
                        ]
                    },
                }
            },
            "datasets": {
                "inquiry_records": [
                    {
                        "record_id": "inquiry:institution:1",
                        "sequence": 1,
                        "inquiry_type": "institution",
                    },
                    {
                        "record_id": "inquiry:personal:1",
                        "sequence": 1,
                        "inquiry_type": "personal",
                    },
                ]
            },
        }
    )

    status = next(
        row
        for row in content["datasets"]["personal_detail_dataset_status"]
        if row["dataset_name"] == "inquiry_records"
    )
    issue = next(
        row
        for row in content["datasets"]["personal_detail_extraction_issues"]
        if row.get("target_dataset") == "inquiry_records"
    )
    assert status["presence_status"] == "partial"
    assert status["expected_row_count"] == 6
    assert issue["candidate_value"]["source_sequence_endpoints"] == {
        "institution": 4,
        "personal": 2,
    }
    assert issue["source_refs"][0]["logical_page"] == 20
    assert "target_record_id" not in issue


def test_source_ledger_projects_exact_missing_account_and_printed_field_issues() -> None:
    ref = {
        "source": "candidate_b_account_anchor",
        "logical_page": 8,
        "source_page": 4,
        "geometry_scope": "line",
        "binding": "printed_account_ordinal",
        "binding_quality": "printed_account_ordinal",
        "account_type": "credit_card",
        "category_sequence": 2,
        "bbox": [20.0, 40.0, 200.0, 60.0],
        "evidence_ids": ["account-anchor-2"],
    }
    field_ref = {
        "source": "native_detail_table_cell",
        "logical_page": 8,
        "source_page": 4,
        "table_id": "account-table-2",
        "row": 1,
        "column": 1,
        "geometry_scope": "cell",
        "binding": "canonical_field_slot",
        "binding_quality": "canonical_header_column",
        "field_name": "account_identifier",
        "bbox": [40.0, 80.0, 220.0, 100.0],
        "evidence_ids": ["account-identifier-2"],
    }
    content = prepare_personal_detail_source_collections(
        {
            "facts": {
                "personal_detail_source_completeness_ledger": {
                    "credit_accounts": 2,
                    "account_family_endpoints": {"credit_card": 2},
                    "account_family_ordinal_observations": {
                        "credit_card": {
                            "2": {
                                "account_id": "credit_account:credit_card:2",
                                "printed_fields": ["account_identifier"],
                                "field_source_refs": {
                                    "account_identifier": [field_ref]
                                },
                                "source_refs": [ref],
                            }
                        }
                    },
                }
            },
            "datasets": {
                "credit_accounts": [
                    {
                        "account_id": "credit_account:credit_card:1",
                        "account_type": "credit_card",
                        "category_sequence": 1,
                    }
                ]
            },
        },
        final_dataset_counts={"credit_accounts": 1},
    )

    issues = content["datasets"]["personal_detail_extraction_issues"]
    identity_issue = next(
        issue for issue in issues if issue.get("issue_code") == "source_account_record_omitted"
    )
    field_issue = next(
        issue for issue in issues if issue.get("issue_code") == "source_account_field_omitted"
    )
    assert identity_issue["target_record_id"] == "credit_account:credit_card:2"
    assert identity_issue["field_name"] == "account_id"
    assert field_issue["target_record_id"] == "credit_account:credit_card:2"
    assert field_issue["field_name"] == "account_identifier"
    assert field_issue["source_refs"][0]["logical_page"] == 8


def test_source_ledger_rejects_duplicate_or_foreign_account_anchor_omission() -> None:
    exact = {
        "source": "candidate_b_account_anchor",
        "logical_page": 8,
        "source_page": 4,
        "geometry_scope": "line",
        "binding": "printed_account_ordinal",
        "binding_quality": "printed_account_ordinal",
        "account_type": "credit_card",
        "category_sequence": 2,
        "bbox": [20.0, 40.0, 200.0, 60.0],
        "evidence_ids": ["account-anchor-2"],
    }
    for observation in (
        {
            "account_id": "credit_account:credit_card:2",
            "source_refs": [exact, {**exact, "evidence_ids": ["duplicate-owner"]}],
        },
        {
            "account_id": "credit_account:non_revolving_loan:2",
            "source_refs": [exact],
        },
    ):
        content = prepare_personal_detail_source_collections(
            {
                "facts": {
                    "personal_detail_source_completeness_ledger": {
                        "credit_accounts": 2,
                        "account_family_endpoints": {"credit_card": 2},
                        "account_family_ordinal_observations": {
                            "credit_card": {"2": observation}
                        },
                    }
                },
                "datasets": {
                    "credit_accounts": [
                        {
                            "account_id": "credit_account:credit_card:1",
                            "account_type": "credit_card",
                            "category_sequence": 1,
                        }
                    ]
                },
            },
            final_dataset_counts={"credit_accounts": 1},
        )

        issues = content["datasets"]["personal_detail_extraction_issues"]
        assert not any(
            issue.get("issue_code") == "source_account_record_omitted"
            for issue in issues
        )


def test_source_ledger_projects_exact_missing_agreement_issue_with_evidence() -> None:
    content = prepare_personal_detail_source_collections(
        {
            "facts": {
                "personal_detail_source_completeness_ledger": {
                    "credit_agreements": 3,
                    "credit_agreement_sequence_endpoint": 3,
                    "credit_agreement_observed_sequences": [1, 2, 3],
                    "credit_agreement_ordinal_observations": {
                        "3": {
                            "sequence": 3,
                            "source_refs": [
                                {
                                    "source": "candidate_b_source_coverage_ledger",
                                    "logical_page": 12,
                                    "source_page": 6,
                                    "geometry_scope": "line",
                                    "binding": "printed_credit_agreement_ordinal",
                                    "binding_quality": "printed_credit_agreement_ordinal",
                                    "sequence": 3,
                                    "bbox": [20.0, 40.0, 120.0, 60.0],
                                    "evidence_ids": ["ocr:sp0006:lp0012:0003"],
                                }
                            ],
                            "printed_fields": ["institution", "total_limit"],
                            "field_source_refs": {
                                "institution": [
                                    {
                                        "source": "personal_detail_corrected_page_cell",
                                        "logical_page": 12,
                                        "source_page": 6,
                                        "geometry_scope": "cell",
                                        "binding": "canonical_label_slot",
                                        "field_name": "institution",
                                        "bbox": [20.0, 80.0, 120.0, 96.0],
                                        "evidence_ids": ["ocr:sp0006:lp0012:institution"],
                                    }
                                ],
                                "total_limit": [
                                    {
                                        "source": "personal_detail_corrected_page_cell",
                                        "logical_page": 12,
                                        "source_page": 6,
                                        "geometry_scope": "cell",
                                        "binding": "canonical_label_slot",
                                        "field_name": "total_limit",
                                        "bbox": [140.0, 80.0, 220.0, 96.0],
                                        "evidence_ids": ["ocr:sp0006:lp0012:total_limit"],
                                    }
                                ],
                            },
                        }
                    },
                }
            },
            "datasets": {
                "credit_lines": [
                    {
                        "credit_line_id": "credit_agreement:1",
                        "_printed_sequence": 1,
                        "_canonical_card_key": "credit_agreement:1",
                    },
                    {
                        "credit_line_id": "credit_agreement:2",
                        "_printed_sequence": 2,
                        "_canonical_card_key": "credit_agreement:2",
                    },
                ]
            },
        }
    )

    issues = content["datasets"]["personal_detail_extraction_issues"]
    issue = next(
        issue
        for issue in issues
        if issue.get("issue_code") == "source_credit_agreement_record_omitted"
    )
    assert issue["target_record_id"] == "credit_agreement:3"
    assert issue["field_name"] == "credit_line_id"
    assert issue["source_refs"][0]["logical_page"] == 12
    field_issues = {
        item["field_name"]: item
        for item in issues
        if item.get("issue_code") == "source_credit_agreement_field_omitted"
    }
    assert set(field_issues) == {"institution", "total_limit"}
    assert all(
        item["source_refs"][0]["field_name"] == field_name
        for field_name, item in field_issues.items()
    )


def _agreement_lifecycle_content(
    emitted_rows: list[dict[str, object]],
) -> dict[str, object]:
    return prepare_personal_detail_source_collections(
        {
            "facts": {
                "personal_detail_source_completeness_ledger": {
                    "credit_agreements": 1,
                    "credit_agreement_sequence_endpoint": 1,
                    "credit_agreement_observed_sequences": [1],
                    "credit_agreement_ordinal_observations": {
                        "1": {
                            "sequence": 1,
                            "source_refs": [
                                {
                                    "source": "candidate_b_source_coverage_ledger",
                                    "logical_page": 12,
                                    "source_page": 6,
                                    "geometry_scope": "line",
                                    "binding": "printed_credit_agreement_ordinal",
                                    "binding_quality": "printed_credit_agreement_ordinal",
                                    "sequence": 1,
                                    "bbox": [20.0, 40.0, 120.0, 60.0],
                                    "evidence_ids": ["ocr:sp0006:lp0012:agreement-1"],
                                }
                            ],
                        }
                    },
                }
            },
            "datasets": {"credit_lines": emitted_rows},
        }
    )


def _hashed_agreement_with_source_identity(
    record_suffix: str,
    *,
    bbox: list[float] | None = None,
) -> dict[str, object]:
    return {
        "credit_line_id": f"credit_line:hashed-{record_suffix}",
        "sequence": 1,
        "_source_agreement_identity": {
            "sequence": 1,
            "source_refs": [
                {
                    "source": "native_detail_canonical_anchor_text",
                    "logical_page": 12,
                    "source_page": 6,
                    "geometry_scope": "text",
                    "binding": "canonical_card_anchor",
                    "bbox": bbox or [22.0, 42.0, 118.0, 58.0],
                }
            ],
        },
    }


def test_source_ledger_reconciles_hashed_agreement_with_exact_card_owner() -> None:
    content = _agreement_lifecycle_content(
        [_hashed_agreement_with_source_identity("one")]
    )

    assert not any(
        issue.get("issue_code")
        in {
            "source_credit_agreement_record_omitted",
            "source_credit_agreement_field_omitted",
        }
        for issue in content["datasets"]["personal_detail_extraction_issues"]
    )


@pytest.mark.parametrize("owner_defect", ("nonoverlap", "duplicate"))
def test_source_ledger_does_not_reconcile_ambiguous_hashed_agreement_owner(
    owner_defect: str,
) -> None:
    rows = [_hashed_agreement_with_source_identity("one")]
    if owner_defect == "nonoverlap":
        rows[0] = _hashed_agreement_with_source_identity(
            "one",
            bbox=[220.0, 240.0, 320.0, 260.0],
        )
    else:
        rows.append(_hashed_agreement_with_source_identity("two"))
    content = _agreement_lifecycle_content(rows)

    assert any(
        issue.get("issue_code") == "source_credit_agreement_record_omitted"
        and issue.get("target_record_id") == "credit_agreement:1"
        for issue in content["datasets"]["personal_detail_extraction_issues"]
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_observation_sequence",
        "wrong_observation_sequence",
        "wrong_source",
        "broad_scope",
        "wrong_binding",
        "wrong_binding_quality",
        "missing_logical_page",
        "missing_source_page",
        "wrong_ref_sequence",
        "missing_bbox",
        "nonfinite_bbox",
        "degenerate_bbox",
        "missing_evidence",
    ),
)
def test_source_ledger_rejects_forged_agreement_omission_refs(
    mutation: str,
) -> None:
    observation = {
        "sequence": 3,
        "source_refs": [
            {
                "source": "candidate_b_source_coverage_ledger",
                "logical_page": 12,
                "source_page": 6,
                "geometry_scope": "line",
                "binding": "printed_credit_agreement_ordinal",
                "binding_quality": "printed_credit_agreement_ordinal",
                "sequence": 3,
                "bbox": [20.0, 40.0, 120.0, 60.0],
                "evidence_ids": ["ocr:sp0006:lp0012:0003"],
            }
        ],
    }
    ref = observation["source_refs"][0]
    if mutation == "missing_observation_sequence":
        observation.pop("sequence")
    elif mutation == "wrong_observation_sequence":
        observation["sequence"] = 2
    elif mutation == "wrong_source":
        ref["source"] = "native_detail_canonical_anchor_text"
    elif mutation == "broad_scope":
        ref["geometry_scope"] = "logical_page"
    elif mutation == "wrong_binding":
        ref["binding"] = "canonical_card_anchor"
    elif mutation == "wrong_binding_quality":
        ref["binding_quality"] = "canonical_card_anchor"
    elif mutation == "missing_logical_page":
        ref.pop("logical_page")
    elif mutation == "missing_source_page":
        ref.pop("source_page")
    elif mutation == "wrong_ref_sequence":
        ref["sequence"] = 2
    elif mutation == "missing_bbox":
        ref.pop("bbox")
    elif mutation == "nonfinite_bbox":
        ref["bbox"] = [20.0, 40.0, float("inf"), 60.0]
    elif mutation == "degenerate_bbox":
        ref["bbox"] = [20.0, 40.0, 20.0, 60.0]
    elif mutation == "missing_evidence":
        ref["evidence_ids"] = []

    content = prepare_personal_detail_source_collections(
        {
            "facts": {
                "personal_detail_source_completeness_ledger": {
                    "credit_agreements": 3,
                    "credit_agreement_sequence_endpoint": 3,
                    "credit_agreement_observed_sequences": [1, 2, 3],
                    "credit_agreement_ordinal_observations": {
                        "3": observation,
                    },
                }
            },
            "datasets": {
                "credit_lines": [
                    {
                        "credit_line_id": "credit_agreement:1",
                        "_printed_sequence": 1,
                        "_canonical_card_key": "credit_agreement:1",
                    },
                    {
                        "credit_line_id": "credit_agreement:2",
                        "_printed_sequence": 2,
                        "_canonical_card_key": "credit_agreement:2",
                    },
                ]
            },
        }
    )

    assert not any(
        issue.get("issue_code") == "source_credit_agreement_record_omitted"
        for issue in content["datasets"]["personal_detail_extraction_issues"]
    )


def test_source_ledger_inquiry_identity_expansion_fails_closed_on_sparse_outlier() -> None:
    content = prepare_personal_detail_source_collections(
        {
            "facts": {
                "personal_detail_source_completeness_ledger": {
                    "inquiry_records": 117,
                    "inquiry_sequence_endpoints": {"institution": 117},
                    "inquiry_observed_sequences": {"institution": [1, 117]},
                }
            },
            "datasets": {
                "inquiry_records": [
                    {
                        "inquiry_id": "credit_inquiry:institution:1",
                        "inquiry_type": "institution",
                        "sequence": 1,
                    }
                ]
            },
        }
    )

    issues = content["datasets"]["personal_detail_extraction_issues"]
    assert any(issue.get("issue_code") == "source_sequence_or_count_gap" for issue in issues)
    assert not any(issue.get("issue_code") == "source_inquiry_record_omitted" for issue in issues)


def test_exact_population_issues_are_idempotent_and_project_structured_evidence() -> None:
    ref = {"logical_page": 20, "source_page": 10, "geometry_scope": "table"}
    content = {
        "facts": {
            "personal_detail_source_completeness_ledger": {
                "inquiry_records": 3,
                "inquiry_sequence_endpoints": {"institution": 3},
                "inquiry_observed_sequences": {"institution": [1, 2, 3]},
                "inquiry_ordinal_observations": {
                    "institution": {"3": {"source_refs": [ref]}}
                },
            }
        },
        "datasets": {
            "inquiry_records": [
                {
                    "inquiry_id": "credit_inquiry:institution:1",
                    "inquiry_type": "institution",
                    "sequence": 1,
                },
                {
                    "inquiry_id": "credit_inquiry:institution:2",
                    "inquiry_type": "institution",
                    "sequence": 2,
                },
            ]
        },
    }
    prepare_personal_detail_source_collections(content)
    prepare_personal_detail_source_collections(content)

    source_issues = [
        issue
        for issue in content["datasets"]["personal_detail_extraction_issues"]
        if issue.get("issue_code") == "source_inquiry_record_omitted"
    ]
    assert len(source_issues) == 1
    assert source_issues[0]["candidate_value"] == {"source_sequence_endpoint": 3}
    projected = project_personal_detail_datasets(content["datasets"])
    public_issue = next(
        row
        for row in projected["extraction_issues"]
        if row.get("issue_code") == "source_inquiry_record_omitted"
    )
    assert public_issue["candidate_value_type"] == "object"
    assert any(
        evidence.get("extraction_issue_id") == public_issue["extraction_issue_id"]
        and evidence.get("evidence_kind") == "candidate"
        and evidence.get("evidence_path") == "source_sequence_endpoint"
        and evidence.get("integer_value") == 3
        for evidence in projected["extraction_issue_evidence"]
    )


def test_agreement_text_without_sealed_source_structure_is_not_population() -> None:
    evidence_pages = [
        {
            "page": 12,
            "source_page": 6,
            "lines": [
                {"text": "授信协议信息"},
                {"text": "授信协议1 授信协议标识A"},
                {"text": "授信协议2 授信协议标识B"},
                {"text": "授信协议3 授信协议标识C"},
            ],
            "tables": [],
        }
    ]
    result = SimpleNamespace(
        pages=[],
        corrected_evidence_pages=lambda: evidence_pages,
    )
    ledger = _source_completeness_ledger(result)
    assert "credit_agreement_sequence_endpoint" not in ledger
    assert "credit_agreement_ordinal_observations" not in ledger


def test_agreement_ledger_uses_frozen_geometry_not_grouped_representative_line() -> None:
    evidence_pages = [
        {
            "page": 20,
            "source_page": 10,
            "lines": [
                {
                    "text": "授信协议信息",
                    "bbox": [10.0, 20.0, 120.0, 36.0],
                    "source_bbox": [10.0, 20.0, 120.0, 36.0],
                    "source_logical_page": 20,
                    "source_page": 10,
                    "evidence_ids": ["ocr:sp0010:lp0020:heading"],
                },
                {
                    "text": "授信协议1",
                    # Canonical-layout grouping moved the line onto the
                    # representative page and transformed its working bbox.
                    "bbox": [210.0, 240.0, 310.0, 260.0],
                    "page": 20,
                    "source_bbox": [12.0, 40.0, 112.0, 60.0],
                    "source_logical_page": 21,
                    "source_page": 11,
                    "evidence_ids": ["ocr:sp0011:lp0021:ordinal"],
                },
            ],
            "tables": [],
        }
    ]
    frozen_page = SimpleNamespace(
        page_number=21,
        source_page_number=11,
        tables=[],
        texts=[
            SimpleNamespace(
                content="授信协议信息",
                bbox=[10.0, 20.0, 120.0, 36.0],
                evidence_ids=["ocr:sp0011:lp0021:heading"],
            ),
            SimpleNamespace(
                content="授信协议1",
                bbox=[12.0, 40.0, 112.0, 60.0],
                evidence_ids=["ocr:sp0011:lp0021:ordinal"],
            ),
            SimpleNamespace(
                content="机构查询记录明细",
                bbox=[12.0, 70.0, 140.0, 86.0],
                evidence_ids=["ocr:sp0011:lp0021:boundary"],
            ),
        ],
    )
    result = SimpleNamespace(
        parse_result=SimpleNamespace(pages=[frozen_page]),
        _frozen_logical_pages={21: frozen_page},
        pages=[],
        corrected_evidence_pages=lambda: evidence_pages,
        reading_order_by_logical={21: 1},
        reading_order_resolution={"resolved": True, "authoritative": True},
    )

    ledger = _source_completeness_ledger(result)
    ref = ledger["credit_agreement_ordinal_observations"]["1"]["source_refs"][0]

    assert ref["logical_page"] == 21
    assert ref["source_page"] == 11
    assert ref["bbox"] == [12.0, 40.0, 112.0, 60.0]


def test_exact_population_does_not_localize_dataset_wide_endpoint_refs() -> None:
    content = prepare_personal_detail_source_collections(
        {
            "facts": {
                "personal_detail_source_completeness_ledger": {
                    "credit_accounts": 2,
                    "account_family_endpoints": {"credit_card": 2},
                    "source_refs": {
                        "credit_accounts": [
                            {
                                "logical_page": 8,
                                "source_page": 4,
                                "geometry_scope": "logical_page",
                            }
                        ]
                    },
                }
            },
            "datasets": {
                "credit_accounts": [
                    {
                        "account_id": "credit_account:credit_card:1",
                        "account_type": "credit_card",
                        "category_sequence": 1,
                    }
                ]
            },
        },
        final_dataset_counts={"credit_accounts": 1},
    )

    issues = content["datasets"]["personal_detail_extraction_issues"]
    assert any(issue.get("issue_code") == "source_sequence_or_count_gap" for issue in issues)
    assert not any(issue.get("issue_code") == "source_account_record_omitted" for issue in issues)


def test_exact_residence_and_employment_rows_project_stable_local_issues() -> None:
    contracts = {
        "residence_records": (
            "residence_record_id",
            "credit_residence",
            "residence",
            (
                "sequence",
                "address",
                "residential_phone",
                "residence_status",
                "information_updated_date",
            ),
        ),
        "employment_records": (
            "employment_record_id",
            "credit_employment",
            "basic",
            (
                "sequence",
                "employer",
                "employer_type",
                "employer_address",
                "employer_phone",
            ),
        ),
    }

    def exact_ref(
        dataset: str,
        component: str,
        ordinal: int,
        field_name: str,
        column: int,
    ) -> dict[str, object]:
        sequence_ref = field_name == "sequence"
        binding = "printed_profile_sequence" if sequence_ref else "printed_profile_field"
        return {
            "source": (
                "candidate_b_raw_profile_sequence_cell"
                if sequence_ref
                else "candidate_b_raw_profile_field_cell"
            ),
            "logical_page": 37,
            "source_page": 19,
            "table_id": f"{dataset}:{component}",
            "row": ordinal,
            "column": column,
            "bbox": [column * 20.0, ordinal * 20.0, column * 20.0 + 10.0, ordinal * 20.0 + 10.0],
            "geometry_scope": "cell",
            "evidence_ids": [f"{dataset}:{component}:{ordinal}:{field_name}"],
            "binding": binding,
            "binding_quality": binding,
            "canonical_template_id": "report_header_and_identity",
            "dataset_name": dataset,
            "component": component,
            "sequence": ordinal,
            "field_name": field_name,
        }

    ordinal_observations: dict[str, dict[str, object]] = {}
    for dataset, (id_field, prefix, component, fields) in contracts.items():
        ordinal_observations[dataset] = {}
        for ordinal in (1, 2):
            field_refs = {
                field_name: [
                    exact_ref(dataset, component, ordinal, field_name, column)
                ]
                for column, field_name in enumerate(fields[1:], 1)
            }
            ordinal_observations[dataset][str(ordinal)] = {
                "sequence": ordinal,
                id_field: stable_record_id(prefix, ordinal),
                "canonical_template_id": "report_header_and_identity",
                "canonical_header_fields_by_component": {
                    component: list(fields)
                },
                "printed_fields": list(fields[1:]),
                "field_source_refs": field_refs,
                "source_refs": [
                    exact_ref(dataset, component, ordinal, "sequence", 0)
                ],
            }
    ledger = {
        "sequence_endpoints": {
            "residence_records": 2,
            "employment_records": 2,
        },
        "sequence_observed_sequences": {
            "residence_records": [1, 2],
            "employment_records": [1, 2],
        },
        "sequence_ordinal_observations": ordinal_observations,
    }
    content = prepare_personal_detail_source_collections(
        {
            "facts": {"personal_detail_source_completeness_ledger": ledger},
            "datasets": {
                "residence_records": [
                    {
                        "record_id": stable_record_id("credit_residence", 1),
                        "residence_record_id": stable_record_id("credit_residence", 1),
                        "sequence": 1,
                    }
                ],
                "employment_records": [
                    {
                        "record_id": stable_record_id("credit_employment", 1),
                        "employment_record_id": stable_record_id("credit_employment", 1),
                        "sequence": 1,
                    }
                ],
            },
        }
    )

    issues = content["datasets"]["personal_detail_extraction_issues"]
    residence_issue = next(
        issue for issue in issues if issue.get("issue_code") == "source_residence_record_omitted"
    )
    employment_issue = next(
        issue for issue in issues if issue.get("issue_code") == "source_employment_record_omitted"
    )
    assert residence_issue["target_record_id"] == stable_record_id("credit_residence", 2)
    assert residence_issue["observed_value"]["sequence"] == 2
    assert residence_issue["source_refs"][0]["sequence"] == 2
    assert employment_issue["target_record_id"] == stable_record_id("credit_employment", 2)
    assert employment_issue["observed_value"]["sequence"] == 2
    assert any(
        issue.get("target_record_id") == stable_record_id("credit_residence", 2)
        and issue.get("field_name") == "address"
        for issue in issues
    )
    assert any(
        issue.get("target_record_id") == stable_record_id("credit_employment", 2)
        and issue.get("field_name") == "employer"
        for issue in issues
    )
    assert len(content["datasets"]["residence_records"]) == 1
    assert len(content["datasets"]["employment_records"]) == 1
    projected = project_personal_detail_datasets(content["datasets"])
    projected_ids = {
        row["target_record_id"]
        for row in projected["extraction_issues"]
        if row.get("target_record_id")
    }
    assert stable_record_id("credit_residence", 2) in projected_ids
    assert stable_record_id("credit_employment", 2) in projected_ids


def test_duplicate_exact_sequence_rows_fail_closed_for_local_omission() -> None:
    residence = _exact_table(
        "residence-duplicate",
        [
            ["编号", "居住地址", "住宅电话", "居住状况", "信息更新日期"],
            ["1", "北京市一号", "01012345678", "自置", "2024.01.01"],
            ["2", "北京市二号", "01087654321", "租房", "2024.02.01"],
            ["2", "北京市三号", "01011112222", "租房", "2024.03.01"],
        ],
    )
    ledger = _source_completeness_ledger(
        SimpleNamespace(
            pages=[
                SimpleNamespace(
                    page_number=2,
                    source_page_number=1,
                    tables=[residence],
                )
            ],
            corrected_evidence_pages=lambda: [],
        )
    )
    content = prepare_personal_detail_source_collections(
        {
            "facts": {"personal_detail_source_completeness_ledger": ledger},
            "datasets": {
                "residence_records": [
                    {
                        "record_id": stable_record_id("credit_residence", 1),
                        "residence_record_id": stable_record_id("credit_residence", 1),
                        "sequence": 1,
                    }
                ]
            },
        }
    )

    assert "2" not in ledger.get("sequence_ordinal_observations", {}).get(
        "residence_records", {}
    )
    assert not any(
        issue.get("issue_code") == "source_residence_record_omitted"
        for issue in content["datasets"]["personal_detail_extraction_issues"]
    )


def test_sequence_ledger_keeps_mixed_table_components_in_header_bounded_regions() -> None:
    mixed = _exact_table(
        "mixed-residence-employment",
        [
            ["\u7f16\u53f7", "\u5c45\u4f4f\u5730\u5740", "\u4f4f\u5b85\u7535\u8bdd", "\u5c45\u4f4f\u72b6\u51b5", "\u4fe1\u606f\u66f4\u65b0\u65e5\u671f"],
            ["1", "RESIDENCE ONE", "01012345678", "STATUS ONE", "2024.01.01"],
            ["\u7f16\u53f7", "\u5de5\u4f5c\u5355\u4f4d", "\u5355\u4f4d\u6027\u8d28", "\u5355\u4f4d\u5730\u5740", "\u5355\u4f4d\u7535\u8bdd"],
            ["2", "EMPLOYER TWO", "TYPE TWO", "ADDRESS TWO", "01087654321"],
        ],
    )
    ledger = _source_completeness_ledger(
        SimpleNamespace(
            pages=[
                SimpleNamespace(
                    page_number=2,
                    source_page_number=1,
                    tables=[mixed],
                )
            ],
            corrected_evidence_pages=lambda: [],
        )
    )

    assert ledger["sequence_endpoints"] == {
        "residence_records": 1,
        "employment_records": 2,
    }
    observations = ledger["sequence_ordinal_observations"]
    assert set(observations["residence_records"]) == {"1"}
    assert set(observations["employment_records"]) == {"2"}

    content = prepare_personal_detail_source_collections(
        {
            "facts": {"personal_detail_source_completeness_ledger": ledger},
            "datasets": {
                "residence_records": [
                    {
                        "record_id": stable_record_id("credit_residence", 1),
                        "residence_record_id": stable_record_id("credit_residence", 1),
                        "sequence": 1,
                    }
                ],
                "employment_records": [
                    {
                        "record_id": stable_record_id("credit_employment", 2),
                        "employment_record_id": stable_record_id("credit_employment", 2),
                        "sequence": 2,
                    }
                ],
            },
        }
    )
    exact_issue_codes = {
        issue.get("issue_code")
        for issue in content["datasets"]["personal_detail_extraction_issues"]
        if issue.get("target_record_id")
    }
    assert "source_residence_record_omitted" not in exact_issue_codes
    assert "source_employment_record_omitted" not in exact_issue_codes


def test_mobile_source_ledger_requires_exact_unique_four_slot_rows() -> None:
    mobile = _exact_table(
        "mobile",
        [
            ["编号", "手机号码", "信息更新日期", "数据发生机构名称"],
            ["1", "13800138000", "2024.01.01", "样例银行"],
            ["2", "13900139000", "2024.02.01", "另一银行"],
        ],
    )
    ledger = _source_completeness_ledger(
        SimpleNamespace(
            pages=[
                SimpleNamespace(
                    page_number=1,
                    source_page_number=1,
                    tables=[mobile],
                )
            ],
            corrected_evidence_pages=lambda: [],
        )
    )

    assert ledger["sequence_endpoints"]["mobile_phone_records"] == 2
    observation = ledger["sequence_ordinal_observations"]["mobile_phone_records"]["2"]
    assert observation["sequence"] == 2
    assert observation["canonical_header_fields"] == [
        "data_provider",
        "information_updated_date",
        "mobile_phone",
        "sequence",
    ]
    assert observation["printed_fields"] == [
        "data_provider",
        "information_updated_date",
        "mobile_phone",
    ]
    assert all(ref["sequence"] == 2 for ref in observation["source_refs"])


def test_inquiry_source_ledger_accepts_only_exact_dense_headerless_carry() -> None:
    header = _exact_table(
        "inquiry-header",
        [
            ["编号", "查询日期", "查询机构", "查询原因"],
            ["1", "2024.01.01", "样例银行", "贷后管理"],
            ["2", "2024.01.02", "样例银行", "贷后管理"],
            ["3", "2024.01.03", "样例银行", "贷后管理"],
        ],
        canonical_template_id="annotations_and_inquiries",
    )
    continuation_rows = [
        [str(sequence), "2024.01.04", "样例银行", "贷后管理"]
        for sequence in range(4, 144)
    ]
    continuation = _exact_table(
        "inquiry-continuation",
        continuation_rows,
        canonical_template_id="annotations_and_inquiries",
    )
    result = SimpleNamespace(
        pages=[
            SimpleNamespace(
                page_number=33,
                source_page_number=17,
                canonical_template_id="annotations_and_inquiries",
                tables=[header],
            ),
            SimpleNamespace(
                page_number=37,
                source_page_number=19,
                canonical_template_id="annotations_and_inquiries",
                tables=[continuation],
            ),
        ],
        corrected_evidence_pages=lambda: [],
        reading_order_by_logical={33: 1, 37: 2},
        reading_order_resolution={
            "status": "resolved",
            "resolved": True,
            "authoritative": True,
        },
    )

    ledger = _source_completeness_ledger(result)

    assert ledger["inquiry_sequence_endpoints"] == {"institution": 143}
    assert ledger["inquiry_observed_sequences"]["institution"] == list(range(1, 144))
    last = ledger["inquiry_ordinal_observations"]["institution"]["143"]
    assert last["source_refs"][0]["logical_page"] == 37
    assert last["source_refs"][0]["geometry_scope"] == "row"


def test_inquiry_source_ledger_keeps_duplicate_ordinal_owners_aggregate_only() -> None:
    inquiry_table = _exact_table(
        "inquiry-duplicate-ordinal",
        [
            ["编号", "查询日期", "查询机构", "查询原因"],
            ["1", "2024.01.01", "样例银行", "贷后管理"],
            ["2", "2024.01.02", "样例银行", "贷后管理"],
            ["2", "2024.01.02", "样例银行", "贷后管理"],
            ["3", "2024.01.03", "样例银行", "贷后管理"],
        ],
        canonical_template_id="annotations_and_inquiries",
    )
    ledger = _source_completeness_ledger(
        SimpleNamespace(
            pages=[
                SimpleNamespace(
                    page_number=20,
                    source_page_number=10,
                    canonical_template_id="annotations_and_inquiries",
                    tables=[inquiry_table],
                )
            ],
            corrected_evidence_pages=lambda: [],
        )
    )

    assert ledger["inquiry_sequence_endpoints"] == {"institution": 3}
    observations = ledger["inquiry_ordinal_observations"]["institution"]
    assert set(observations) == {"1", "3"}

    content = {
        "facts": {"personal_detail_source_completeness_ledger": ledger},
        "datasets": {
            "inquiry_records": [
                {
                    "inquiry_id": "credit_inquiry:institution:1",
                    "inquiry_type": "institution",
                    "sequence": 1,
                },
                {
                    "inquiry_id": "credit_inquiry:institution:3",
                    "inquiry_type": "institution",
                    "sequence": 3,
                },
            ]
        },
    }
    prepare_personal_detail_source_collections(content)

    issues = content["datasets"]["personal_detail_extraction_issues"]
    assert any(
        issue.get("issue_code") == "source_sequence_or_count_gap"
        and issue.get("target_dataset") == "inquiry_records"
        for issue in issues
    )
    assert not any(
        issue.get("issue_code") in {
            "source_inquiry_record_omitted",
            "source_inquiry_field_omitted",
        }
        and issue.get("target_record_id") == "credit_inquiry:institution:2"
        for issue in issues
    )


@pytest.mark.parametrize("defect", ["wrong_page", "competing_group", "derived_geometry"])
def test_inquiry_source_ledger_rejects_unbounded_headerless_carry(defect: str) -> None:
    header = _exact_table(
        "inquiry-negative-header",
        [
            ["编号", "查询日期", "查询机构", "查询原因"],
            ["1", "2024.01.01", "样例银行", "贷后管理"],
            ["2", "2024.01.02", "样例银行", "贷后管理"],
            ["3", "2024.01.03", "样例银行", "贷后管理"],
        ],
        canonical_template_id="annotations_and_inquiries",
    )
    rows = [["4", "2024.01.04", "样例银行", "贷后管理"]]
    if defect == "competing_group":
        rows[0][2:] = ["本人", "本人查询"]
    continuation = _exact_table(
        "inquiry-negative-continuation",
        rows,
        derived_cells={(0, 2)} if defect == "derived_geometry" else frozenset(),
        canonical_template_id="annotations_and_inquiries",
    )
    result = SimpleNamespace(
        pages=[
            SimpleNamespace(
                page_number=33,
                source_page_number=17,
                canonical_template_id="annotations_and_inquiries",
                tables=[header],
            ),
            SimpleNamespace(
                page_number=37,
                source_page_number=19,
                canonical_template_id=(
                    "public_records" if defect == "wrong_page" else "annotations_and_inquiries"
                ),
                tables=[continuation],
            ),
        ],
        corrected_evidence_pages=lambda: [],
        reading_order_resolution={"status": "ambiguous"},
    )

    ledger = _source_completeness_ledger(result)

    assert ledger["inquiry_sequence_endpoints"] == {"institution": 3}
    assert "4" not in ledger.get("inquiry_ordinal_observations", {}).get(
        "institution", {}
    )


def test_source_ledger_rejects_isolated_account_sequence_outlier() -> None:
    endpoint, outliers = _credible_sequence_endpoint(set(range(1, 28)) | {115})

    assert endpoint == 27
    assert outliers == [115]


def test_unsealed_sample_account_lines_cannot_establish_population() -> None:
    lines = [
        {"text": "（一）非循环贷账户"},
        *({"text": f"账户{sequence}"} for sequence in range(1, 23)),
        {"text": "（二）循环贷账户二"},
        {
            "text": (
                "账户（授信协议标识："
                "D10053310H00011022661000153960931220220529）"
            )
        },
        {"text": "（三）贷记卡账户"},
        *({"text": f"账户{sequence}"} for sequence in range(1, 23)),
        {"text": "（五）授信协议信息"},
    ]
    result = SimpleNamespace(
        pages=[],
        corrected_evidence_pages=lambda: [
            {"page": 1, "source_page": 1, "lines": lines}
        ],
    )

    ledger = _source_completeness_ledger(result)

    assert "credit_accounts" not in ledger
    assert "account_family_endpoints" not in ledger
    assert "account_family_source_populations" not in ledger


def test_unsealed_sparse_and_weak_account_lines_cannot_establish_population() -> None:
    result = SimpleNamespace(
        pages=[],
        corrected_evidence_pages=lambda: [
            {
                "page": 1,
                "source_page": 1,
                "lines": [
                    {"text": "（一）非循环贷账户"},
                    {"text": "账户1"},
                    {"text": "账户3"},
                    {"text": "账户115"},
                    {"text": "（二）循环贷账户二"},
                    {"text": "账户（授信协议标识：R2ACCOUNT0001）"},
                    # Repeated source text with the same strong identity is one account.
                    {"text": "账户（授信协议标识：R2ACCOUNT0001）"},
                    {"text": "账户（授信协议标识：R2ACCOUNT0002）"},
                    {"text": "（三）贷记卡账户"},
                    # Multiple weak, unnumbered anchors could be overlapping OCR
                    # evidence and therefore cannot establish a population of two.
                    {"text": "账户"},
                    {"text": "账户"},
                    {"text": "（五）授信协议信息"},
                    {"text": "账户99"},
                ],
            }
        ],
    )

    ledger = _source_completeness_ledger(result)

    assert "credit_accounts" not in ledger
    assert "account_family_endpoints" not in ledger
    assert "account_family_source_populations" not in ledger


def test_unregistered_single_weak_account_anchor_is_not_a_source_witness() -> None:
    result = SimpleNamespace(
        pages=[],
        corrected_evidence_pages=lambda: [
            {
                "page": 7,
                "source_page": 4,
                "lines": [
                    {"text": "（二）循环贷账户二"},
                    {
                        "text": "账户",
                        "bbox": [20.0, 40.0, 60.0, 52.0],
                        "evidence_ids": ["ocr:sp0004:lp0007:0001"],
                    },
                ],
            }
        ],
    )

    ledger = _source_completeness_ledger(result)

    assert "credit_accounts" not in ledger
    assert "account_family_source_populations" not in ledger
    assert "account_family_unnumbered_anchor_counts" not in ledger


def test_account_source_ledger_does_not_mutate_final_business_rows() -> None:
    source_rows = [
        {"record_id": f"account:{sequence}", "account_id": f"account:{sequence}"}
        for sequence in range(1, 46)
    ]
    content = prepare_personal_detail_source_collections(
        {
            "facts": {
                "personal_detail_source_completeness_ledger": {
                    "credit_accounts": 45,
                    "account_family_source_populations": {
                        "non_revolving_loan": 22,
                        "revolving_loan_account": 1,
                        "credit_card": 22,
                    },
                }
            },
            "datasets": {"credit_accounts": list(source_rows)},
        },
        final_dataset_counts={"credit_accounts": 45},
    )

    assert content["datasets"]["credit_accounts"] == source_rows
    assert not any(
        issue.get("issue_code") == "source_sequence_or_count_gap"
        and issue.get("target_dataset") == "credit_accounts"
        for issue in content["datasets"].get("personal_detail_extraction_issues", [])
    )


def test_agreement_ledger_does_not_count_account_heading_identifiers() -> None:
    raw_pages = [
        SimpleNamespace(
            page_number=1,
            source_page_number=1,
            tables=[],
            texts=[
                SimpleNamespace(
                    content="账户1（授信协议标识：ACCOUNT0001）",
                    bbox=[10.0, 20.0, 200.0, 30.0],
                    evidence_ids=["account-heading"],
                )
            ],
        ),
        SimpleNamespace(
            page_number=2,
            source_page_number=2,
            tables=[],
            texts=[
                SimpleNamespace(
                    content=text,
                    bbox=[10.0, 20.0 + index * 20.0, 180.0, 30.0 + index * 20.0],
                    evidence_ids=[f"agreement-structure:{index}"],
                )
                for index, text in enumerate(
                    ("（五）授信协议信息", "授信协议1", "授信协议2", "公共信息明细")
                )
            ],
        ),
    ]
    result = SimpleNamespace(
        parse_result=SimpleNamespace(pages=raw_pages),
        _frozen_logical_pages={page.page_number: page for page in raw_pages},
        pages=[],
        corrected_evidence_pages=lambda: [],
        reading_order_by_logical={1: 1, 2: 2},
        reading_order_resolution={"resolved": True, "authoritative": True},
    )

    ledger = _source_completeness_ledger(result)

    assert ledger["credit_agreements"] == 2


def test_agreement_ledger_prefers_printed_endpoint_over_duplicate_primary_labels() -> None:
    raw_page = SimpleNamespace(
        page_number=24,
        source_page_number=12,
        tables=[],
        texts=[
            SimpleNamespace(
                content=text,
                bbox=[10.0, 20.0 + index * 20.0, 180.0, 30.0 + index * 20.0],
                evidence_ids=[f"agreement-structure:{index}"],
            )
            for index, text in enumerate(
                (
                    "（五）授信协议信息",
                    "授信协议1",
                    "授信协议标识",
                    "授信协议标识",
                    "授信协议2",
                    "授信协议标识",
                    "公共信息明细",
                )
            )
        ],
    )
    result = SimpleNamespace(
        parse_result=SimpleNamespace(pages=[raw_page]),
        _frozen_logical_pages={24: raw_page},
        pages=[],
        corrected_evidence_pages=lambda: [],
        reading_order_by_logical={24: 1},
        reading_order_resolution={"resolved": True, "authoritative": True},
    )

    ledger = _source_completeness_ledger(result)

    assert ledger["credit_agreements"] == 2


def test_duplicate_issue_stages_merge_evidence_instead_of_repeating() -> None:
    first = make_issue(
        category="ocr_cell_level_error",
        issue_code="pboc_cell_contract_unresolved",
        message="first stage",
        parser_stage="native",
        target_dataset="credit_accounts",
        target_record_id="account:1",
        field_name="balance",
        observed_value="12O0",
        source_refs=({"logical_page": 4, "evidence_id": "native"},),
    )
    context = SimpleNamespace(
        _personal_detail_extraction_issues=[first],
        ocr_correction_audit=lambda: {
            "cell_anomalies": [
                {
                    "stage": "projection",
                    "path": "credit_accounts[0].balance",
                    "dataset_name": "credit_accounts",
                    "record_id": "account:1",
                    "field_name": "balance",
                    "role": "amount",
                    "value": "12O0",
                    "source_refs": [{"logical_page": 4, "evidence_id": "projection"}],
                }
            ]
        },
    )

    issues = collect_extraction_issues(context)

    assert len(issues) == 1
    assert {ref["evidence_id"] for ref in issues[0]["source_refs"]} == {"native", "projection"}


def test_precise_agreement_field_issue_supersedes_only_redundant_required_warning() -> None:
    precise = make_issue(
        category="ocr_cell_level_error",
        issue_code="pboc_cell_contract_unresolved",
        message="typed cell failed",
        target_dataset="credit_lines",
        target_record_id="credit_line:1",
        field_name="credit_limit",
        observed_value='".',
        source_refs=({"logical_page": 8, "evidence_id": "precise"},),
        reason_codes=("role_validation_failed", "normalized_value_withheld"),
    )
    generic = make_issue(
        category="schema_incompleteness",
        issue_code="candidate_b_credit_agreement_required_field_unresolved",
        message="required field unavailable",
        target_dataset="credit_lines",
        target_record_id="credit_line:1",
        field_name="credit_limit",
        source_refs=({"logical_page": 8, "evidence_id": "generic"},),
        reason_codes=("required_field_missing",),
    )
    context = SimpleNamespace(_personal_detail_extraction_issues=[precise, generic])

    issues = collect_extraction_issues(context)

    assert len(issues) == 1
    assert issues[0]["issue_code"] == "pboc_cell_contract_unresolved"
    assert issues[0]["observed_value"] == '".'
    assert {ref["evidence_id"] for ref in issues[0]["source_refs"]} == {
        "precise",
        "generic",
    }
    assert "required_field_missing" in issues[0]["reason_codes"]

    generic_only = collect_extraction_issues(
        SimpleNamespace(_personal_detail_extraction_issues=[generic])
    )
    assert len(generic_only) == 1
    assert generic_only[0]["issue_code"] == (
        "candidate_b_credit_agreement_required_field_unresolved"
    )


def test_issue_targets_follow_chained_agreement_identity_reconciliation() -> None:
    first = make_issue(
        category="ocr_cell_level_error",
        issue_code="pboc_cell_contract_unresolved",
        message="native observation",
        target_dataset="credit_lines",
        target_record_id="credit_line:old",
        field_name="institution",
        observed_value="damaged",
        source_refs=({"logical_page": 7, "evidence_id": "native"},),
    )
    context = SimpleNamespace(
        _personal_detail_extraction_issues=[first],
        ocr_correction_audit=lambda: {
            "cell_anomalies": [
                {
                    "stage": "projection",
                    "dataset_name": "credit_lines",
                    "record_id": "credit_line:corrected",
                    "field_name": "institution",
                    "role": "institution_name",
                    "value": "damaged",
                    "source_refs": [{"logical_page": 7, "evidence_id": "projection"}],
                }
            ]
        },
    )
    register_issue_target_remap(context, "credit_line:old", "credit_line:corrected")
    register_issue_target_remap(context, "credit_line:corrected", "credit_line:final")

    issues = collect_extraction_issues(context)

    assert len(issues) == 1
    assert issues[0]["target_record_id"] == "credit_line:final"
    assert issues[0]["record_id"] == issues[0]["extraction_issue_id"]
    assert issues[0]["extraction_issue_id"] != first["extraction_issue_id"]
    assert {ref["evidence_id"] for ref in issues[0]["source_refs"]} == {"native", "projection"}


def test_ambiguous_issue_target_is_reported_without_guessing_a_record() -> None:
    issue = make_issue(
        category="ocr_cell_level_error",
        issue_code="pboc_cell_contract_unresolved",
        message="ambiguous identity",
        target_dataset="credit_lines",
        target_record_id="credit_line:old",
        field_name="institution",
        observed_value="damaged",
    )
    context = SimpleNamespace(_personal_detail_extraction_issues=[issue])
    register_issue_target_remap(context, "credit_line:old", "credit_line:first")
    register_issue_target_remap(context, "credit_line:old", "credit_line:second")

    remapped = collect_extraction_issues(context)[0]

    assert "target_record_id" not in remapped
    assert "issue_target_identity_ambiguous" in remapped["reason_codes"]
    assert "diagnostic_left_unlinked" in remapped["reason_codes"]


def test_header_consensus_reports_missing_fields_for_coordinated_page_repair(monkeypatch) -> None:
    monkeypatch.setattr(PBOCPersonalDetailNativeParser, "records", lambda self, name: [])
    result = SimpleNamespace(
        pages=[SimpleNamespace(page_number=1, source_page_number=1, tables=[])],
        _personal_detail_extraction_issues=[],
    )

    datasets = _extract_header_datasets(result, "")

    metadata = datasets["personal_report_metadata"][0]
    assert metadata["subject_name"] is None
    assert metadata["primary_id_type"] is None
    assert metadata["primary_id_number"] is None
    assert metadata["report_time"] is None
    assert len(result._personal_detail_extraction_issues) == 7
    assert all(
        issue["source_refs"][0].get("source")
        == "candidate_b_report_header_business_fields"
        for issue in result._personal_detail_extraction_issues
    )
    assert all(
        issue["source_refs"][0].get("canonical_template_id")
        == "report_header_and_identity"
        for issue in result._personal_detail_extraction_issues
    )
    assert all(
        "logical_page" not in issue["source_refs"][0]
        for issue in result._personal_detail_extraction_issues
    )
    targets = {
        issue["field_name"]: issue["target_dataset"]
        for issue in result._personal_detail_extraction_issues
    }
    assert targets["query_institution"] == "report_query"
    assert targets["query_reason"] == "report_query"
    assert {
        targets[field]
        for field in (
            "report_number",
            "report_time",
            "subject_name",
            "primary_id_type",
            "primary_id_number",
        )
    } == {"report_metadata"}
    metadata_id = metadata["personal_report_metadata_id"]
    assert all(
        issue["target_record_id"]
        == (
            f"report_query:{metadata_id}"
            if issue["target_dataset"] == "report_query"
            else metadata_id
        )
        for issue in result._personal_detail_extraction_issues
    )
    assert all(
        issue["record_id"] == issue["extraction_issue_id"]
        for issue in result._personal_detail_extraction_issues
    )


def _exact_header_token_context(
    *,
    value_tokens: tuple[tuple[str, tuple[float, float, float, float], str], ...],
    raw_name: str,
) -> SimpleNamespace:
    rows = [
        ["被查询者姓名", "被查询者证件类型", "被查询者证件号码", "查询机构 查询原因"],
        [raw_name, "身份证", "350121199101285219", "本人 本人查询(自助查询机)"],
    ]
    cell_bboxes = [
        [[0, 0, 60, 10], [60, 0, 120, 10], [120, 0, 220, 10], [220, 0, 360, 10]],
        [[0, 10, 60, 30], [60, 10, 120, 30], [120, 10, 220, 30], [220, 10, 360, 30]],
    ]
    header_ids = ["header-name", "header-type", "header-number", "header-query"]
    value_ids = [token[2] for token in value_tokens]
    table = SimpleNamespace(
        table_id="header-token-residue",
        metadata={
            "canonical_template_id": "report_header_and_identity",
            "raw_rows": rows,
            "geometry": {
                "cell_bboxes": cell_bboxes,
                "cell_geometry_status": [["exact"] * 4, ["exact"] * 4],
                "row_bands": [
                    {"index": 0, "y0": 0, "y1": 10},
                    {"index": 1, "y0": 10, "y1": 30},
                ],
                "col_bands": [
                    {"index": 0, "x0": 0, "x1": 60},
                    {"index": 1, "x0": 60, "x1": 120},
                    {"index": 2, "x0": 120, "x1": 220},
                    {"index": 3, "x0": 220, "x1": 360},
                ],
                "cell_spans": [],
                "cell_evidence_ids": [
                    [[header_ids[column]] for column in range(4)],
                    [value_ids, ["value-type"], ["value-number"], ["value-query"]],
                ],
                "cell_token_ids": [
                    [[header_ids[column]] for column in range(4)],
                    [value_ids, ["value-type"], ["value-number"], ["value-query"]],
                ],
            },
        },
        headers=[],
        rows=[],
        bbox=[0, 0, 360, 30],
        confidence=0.99,
    )
    atoms = [
        {"id": "header-name", "text": "被查询者姓名", "bbox": [0, 0, 60, 10]},
        {"id": "header-type", "text": "被查询者证件类型", "bbox": [60, 0, 120, 10]},
        {"id": "header-number", "text": "被查询者证件号码", "bbox": [120, 0, 220, 10]},
        {"id": "header-query", "text": "查询机构 查询原因", "bbox": [220, 0, 360, 10]},
        *[
            {"id": token_id, "text": text, "bbox": list(bbox)}
            for text, bbox, token_id in value_tokens
        ],
        {"id": "value-type", "text": "身份证", "bbox": [60, 10, 120, 30]},
        {"id": "value-number", "text": "350121199101285219", "bbox": [120, 10, 220, 30]},
        {"id": "value-query", "text": "本人 本人查询(自助查询机)", "bbox": [220, 10, 360, 30]},
    ]
    page = SimpleNamespace(
        page_number=1,
        source_page_number=1,
        canonical_template_id="report_header_and_identity",
        tables=[table],
    )
    return SimpleNamespace(
        pages=[page],
        evidence_plane=SimpleNamespace(evidence=SimpleNamespace(text_atoms=atoms)),
        _personal_detail_extraction_issues=[],
    )


def test_header_exact_tokens_remove_one_tiny_watermark_without_editing_name(
    monkeypatch,
) -> None:
    monkeypatch.setattr(PBOCPersonalDetailNativeParser, "records", lambda self, name: [])
    context = _exact_header_token_context(
        raw_name="P 黄圣辉",
        value_tokens=(
            ("P", (2.0, 15.0, 5.0, 23.0), "name-watermark"),
            ("黄圣辉", (10.0, 14.0, 42.0, 24.0), "name-value"),
        ),
    )

    datasets = _extract_header_datasets(context, "报告编号:2025080514473622586957 报告时间:2025.08.05 14:47:36")

    assert datasets["personal_report_metadata"][0]["subject_name"] == "黄圣辉"
    assert not any(
        issue.get("field_name") == "subject_name"
        for issue in context._personal_detail_extraction_issues
    )


@pytest.mark.parametrize(
    ("raw_name", "value_tokens"),
    [
        (
            "P 黄圣辉 王",
            (
                ("P", (2.0, 15.0, 5.0, 23.0), "name-watermark"),
                ("黄圣辉", (10.0, 14.0, 42.0, 24.0), "name-value"),
                ("王", (45.0, 14.0, 55.0, 24.0), "second-name"),
            ),
        ),
        (
            "黄圣辉 王小明",
            (
                ("黄圣辉", (2.0, 14.0, 28.0, 24.0), "name-value"),
                ("王小明", (30.0, 14.0, 55.0, 24.0), "second-name"),
            ),
        ),
        (
            "银行 黄圣辉",
            (
                ("银行", (2.0, 14.0, 18.0, 24.0), "wide-residue"),
                ("黄圣辉", (22.0, 14.0, 48.0, 24.0), "name-value"),
            ),
        ),
    ],
)
def test_header_subject_name_token_recovery_fails_closed_for_competing_text(
    monkeypatch,
    raw_name,
    value_tokens,
) -> None:
    monkeypatch.setattr(PBOCPersonalDetailNativeParser, "records", lambda self, name: [])
    context = _exact_header_token_context(raw_name=raw_name, value_tokens=value_tokens)

    datasets = _extract_header_datasets(context, "报告编号:2025080514473622586957 报告时间:2025.08.05 14:47:36")

    assert datasets["personal_report_metadata"][0]["subject_name"] is None
    assert any(
        issue.get("field_name") == "subject_name"
        and issue.get("issue_code") == "page_one_consensus_unresolved"
        for issue in context._personal_detail_extraction_issues
    )


def test_page_one_issues_target_final_rows_without_duplicate_gate_issues(monkeypatch) -> None:
    monkeypatch.setattr(PBOCPersonalDetailNativeParser, "records", lambda self, name: [])
    result = SimpleNamespace(
        pages=[SimpleNamespace(page_number=1, source_page_number=1, tables=[])],
        _personal_detail_extraction_issues=[],
    )
    datasets = _extract_header_datasets(result, "")
    datasets["personal_detail_extraction_issues"] = list(
        result._personal_detail_extraction_issues
    )

    projected = project_personal_detail_datasets(datasets)

    def values(record):
        return record.get("normalized", record)

    final_ids = {
        dataset_name: {
            str(values(record)[identity_field])
            for record in projected[dataset_name]
        }
        for dataset_name, identity_field in (
            ("report_metadata", "report_metadata_id"),
            ("report_query", "report_query_id"),
        )
    }
    issue_values = [values(issue) for issue in projected["extraction_issues"]]
    page_one_keys: dict[tuple[str, str, str], int] = {}
    for issue in issue_values:
        target_dataset = str(issue["target_dataset"])
        target_record_id = str(issue["target_record_id"])
        field_name = str(issue["field_name"])
        assert target_record_id in final_ids[target_dataset]
        key = (target_dataset, target_record_id, field_name)
        page_one_keys[key] = page_one_keys.get(key, 0) + 1

    assert len(issue_values) == 12
    assert set(page_one_keys.values()) == {1}
    assert sum(
        issue.get("issue_code") == "page_one_consensus_unresolved"
        for issue in issue_values
    ) == 7


def test_withheld_typed_value_is_partial_and_has_field_observation() -> None:
    issue = make_issue(
        category="ocr_cell_level_error",
        issue_code="pboc_cell_contract_unresolved",
        message="withheld",
        target_dataset="credit_accounts",
        target_record_id="account:1",
        field_name="balance",
        observed_value="12O0",
        reason_codes=("role_validation_failed", "normalized_value_withheld"),
    )

    assert dataset_states_from_issues([issue])["credit_accounts"]["presence_status"] == "partial"
    content = prepare_personal_detail_source_collections(
        {
            "facts": {
                "personal_detail_dataset_states": dataset_states_from_issues([issue]),
            },
            "datasets": {
                "credit_accounts": [{"record_id": "account:1"}],
                "personal_detail_extraction_issues": [issue],
            },
        }
    )
    observations = content["datasets"]["personal_detail_field_observations"]
    balance = next(row for row in observations if row["field_name"] == "balance")
    assert balance["business_record_id"] == "account:1"
    assert balance["raw_value"] == "12O0"
    assert balance["observation_status"] == "unreadable"
    statuses = {row["dataset_name"]: row for row in content["datasets"]["personal_detail_dataset_status"]}
    assert statuses["credit_accounts"]["presence_status"] == "partial"


def test_native_parser_rejects_similarity_only_label_damage() -> None:
    table = _table(
        "credit-line",
        [
            ["授信协议标", "授信额度用途", "管理机构"],
            ["AGREEMENT0001", "循环额度", "示例银行股份有限公司"],
        ],
    )
    table.bbox = [20, 100, 580, 300]
    table.metadata["canonical_template_id"] = "credit_agreement"
    page = SimpleNamespace(
        page_number=1,
        source_page_number=1,
        canonical_template_id="credit_agreement",
        tables=[table],
        texts=[SimpleNamespace(content="授信协议1", bbox=[20, 78, 120, 98])],
    )
    context = SimpleNamespace(
        pages=[page],
        reading_order_by_logical={1: 1},
        tables_continue=lambda _left, _right: None,
    )

    records = PBOCPersonalDetailNativeParser(context).records("credit_lines")

    assert records == []
    issue = context._personal_detail_extraction_issues[0]
    assert issue["issue_code"] == "canonical_label_authorization_unresolved"
    assert issue["observed_value"]["missing_labels"] == ["授信协议标识"]
    assert "schema_triggered_page_repair_eligible" in issue["reason_codes"]


def test_native_agreement_parser_uses_template_and_heading_sequence() -> None:
    agreement_table = _table(
        "credit-line",
        [
            ["授信协议标识", "授信额度用途", "管理机构"],
            ["AGREEMENT0007", "循环贷款额度", "示例银行股份有限公司"],
        ],
    )
    agreement_table.bbox = [20, 100, 580, 300]
    agreement_table.metadata["canonical_template_id"] = "credit_agreement"
    heading = SimpleNamespace(content="授信协议7", bbox=[20, 60, 120, 80])
    agreement_page = SimpleNamespace(
        page_number=7,
        source_page_number=4,
        canonical_template_id="credit_agreement",
        tables=[agreement_table],
        texts=[heading],
    )
    account_table = _table(
        "account-card",
        [
            ["授信协议标识", "授信额度用途"],
            ["ACCOUNT0001", "循环贷款额度"],
        ],
    )
    account_table.metadata["canonical_template_id"] = "credit_account_detail"
    account_page = SimpleNamespace(
        page_number=6,
        source_page_number=3,
        canonical_template_id="credit_account_detail",
        tables=[account_table],
        texts=[],
    )
    context = SimpleNamespace(
        pages=[account_page, agreement_page],
        reading_order_by_logical={6: 6, 7: 7},
        tables_continue=lambda _left, _right: None,
    )

    records = PBOCPersonalDetailNativeParser(context).records("credit_lines")

    assert len(records) == 1
    assert records[0].fields["授信协议标识"] == "AGREEMENT0007"
    assert records[0].fields["__printed_sequence"] == "7"
    assert records[0].binding_quality_by_field["__printed_sequence"] == (
        "canonical_card_anchor"
    )
    sequence_ref = records[0].source_refs_by_field["__printed_sequence"][0]
    assert sequence_ref["source"] == "native_detail_canonical_anchor_text"
    assert sequence_ref["binding"] == "canonical_card_anchor"
    assert sequence_ref["bbox"] == [20, 60, 120, 80]


def test_native_agreement_parser_assigns_each_heading_to_one_card_in_merged_table() -> None:
    rows = [
        ["管理机构", "授信协议标识", "生效日期", "到期日期", "授信额度用途"],
        ["机构甲", "AGREEMENT0004", "2025.07.19", "2026.07.19", "循环贷款额度"],
        ["授信额度", "授信限额", "授信限额编号", "已用额度", "币种"],
        ["3,000", "--", "--", "--", "人民币元"],
        ["管理机构", "授信协议标识", "生效日期", "到期日期", "授信额度用途"],
        ["机构乙", "AGREEMENT0005", "2024.07.17", "长期", "信用卡共享额度"],
        ["授信额度", "授信限额", "授信限额编号", "已用额度", "币种"],
        ["12,000", "--", "--", "0", "人民币元"],
    ]
    table = _table("merged-agreement-cards", rows)
    table.bbox = [20, 80, 580, 360]
    table.metadata["canonical_template_id"] = "credit_agreement"
    table.metadata["source_cell_bboxes"] = [
        [[20 + column * 110, 100 + row * 28, 120 + column * 110, 120 + row * 28] for column in range(5)]
        for row in range(len(rows))
    ]
    page = SimpleNamespace(
        page_number=8,
        source_page_number=4,
        canonical_template_id="credit_agreement",
        tables=[table],
        texts=[
            SimpleNamespace(content="授信协议4", bbox=[20, 70, 120, 90]),
            SimpleNamespace(content="授信协议5", bbox=[20, 180, 120, 200]),
        ],
    )
    context = SimpleNamespace(
        pages=[page],
        reading_order_by_logical={8: 8},
        tables_continue=lambda _left, _right: None,
    )

    records = PBOCPersonalDetailNativeParser(context).records("credit_lines")

    assert [record.fields["授信协议标识"] for record in records] == [
        "AGREEMENT0004",
        "AGREEMENT0005",
    ]
    assert [record.fields["__printed_sequence"] for record in records] == ["4", "5"]
    refs = [record.source_refs_by_field["__printed_sequence"][0] for record in records]
    assert [ref["bbox"] for ref in refs] == [[20, 70, 120, 90], [20, 180, 120, 200]]


def test_native_agreement_parser_recovers_closed_schema_values_collapsed_with_labels() -> None:
    def line(text: str, x: float, y: float, width: float = 520) -> dict[str, object]:
        return {"text": text, "bbox": [x, y, x + width, y + 18], "confidence": 0.98}

    context = SimpleNamespace(
        pages=[],
        corrected_evidence_pages=lambda: [
            {
                "page": 8,
                "source_page": 4,
                "lines": [
                    line("（四）授信协议信息", 20, 10),
                    line("授信协议6", 20, 40, 120),
                    line(
                        "管理机构 中信消费金融有限公司 授信协议标识 "
                        "T10151024H0002HT11202211122200008302277 "
                        "生效日期 2022.11.12 到期日期 2031.02.12 授信额度用途 循环贷款额度",
                        20,
                        70,
                    ),
                    line(
                        "授信额度 23,600 授信限额 -- 授信限额编号 -- 已用额度 0 币种 人民币元",
                        20,
                        100,
                    ),
                    line("查询记录", 20, 140, 120),
                ],
            }
        ],
    )

    records = PBOCPersonalDetailNativeParser(context).records("credit_lines")

    assert len(records) == 1
    assert records[0].fields == {
        "管理机构": "中信消费金融有限公司",
        "授信协议标识": "T10151024H0002HT11202211122200008302277",
        "生效日期": "2022.11.12",
        "到期日期": "2031.02.12",
        "授信额度用途": "循环贷款额度",
        "授信额度": "23,600",
        "授信限额": "--",
        "授信限额编号": "--",
        "已用额度": "0",
        "币种": "人民币元",
        "__printed_sequence": "6",
    }


def test_native_agreement_table_recovers_values_embedded_in_exact_label_cells() -> None:
    table = _table(
        "collapsed-native-agreement",
        [
            [
                "管理机构 中信消费金融有限公司 授信额度 23,600",
                "授信协议标识",
                "生效日期",
                "到期日期 长期",
                "授信额度用途",
            ],
            [
                "",
                "T10151024H0002HT11202211122200008302277",
                "2022.11.12",
                "",
                "信用卡共享额度",
            ],
            ["", "授信限额", "授信限额编号", "已用额度", "币种"],
            ["", "--", "--", "0", "人民币元"],
        ],
    )
    table.bbox = [20, 100, 580, 300]
    table.metadata["canonical_template_id"] = "credit_agreement"
    page = SimpleNamespace(
        page_number=8,
        source_page_number=4,
        canonical_template_id="credit_agreement",
        tables=[table],
        texts=[SimpleNamespace(content="授信协议6", bbox=[20, 60, 120, 80])],
    )
    context = SimpleNamespace(
        pages=[page],
        reading_order_by_logical={8: 8},
        tables_continue=lambda _left, _right: None,
    )

    records = PBOCPersonalDetailNativeParser(context).records("credit_lines")

    assert len(records) == 1
    assert records[0].fields["管理机构"] == "中信消费金融有限公司"
    assert records[0].fields["授信额度"] == "23,600"
    assert records[0].fields["到期日期"] == "长期"
    assert records[0].fields["已用额度"] == "0"
    assert records[0].fields["__printed_sequence"] == "6"


def test_credit_agreement_institution_disagreement_is_withheld_across_ocr_planes() -> None:
    identifier = "X3501010000133CT6009127971477147648"
    common = {
        "credit_line_id": f"credit_line:{identifier}",
        "account_identifier": identifier,
        "_printed_sequence": 4,
        "facility_type": "循环贷款额度",
        "effective_date": "2025-07-19",
        "due_date": "2026-07-19",
        "total_limit": 3000,
        "currency": "CNY",
        "source_refs": [{"logical_page": 8, "source_page": 4}],
    }
    native = {
        **common,
        "institution": "福州奇高网络小额贷款有限公司",
        "source_refs_by_field": {
            "institution": [
                {
                    "logical_page": 8,
                    "source_page": 4,
                    "geometry_scope": "cell",
                    "binding": "label_column",
                }
            ]
        },
        "_field_binding_quality": {"institution": "native_label_column"},
        "confidence": 0.99,
    }
    corrected = {
        **common,
        "institution": "福州奇富网络小额贷款有限公司",
        "source_refs_by_field": {
            "institution": [
                {
                    "logical_page": 8,
                    "source_page": 4,
                    "geometry_scope": "cell",
                    "binding": "canonical_label_slot",
                }
            ]
        },
        "_field_binding_quality": {"institution": "canonical_cell_slot"},
        "confidence": 0.98,
    }
    context = SimpleNamespace()

    rows = reconcile_candidate_b_credit_lines(context, [native, corrected])

    assert len(rows) == 1
    assert rows[0]["institution"] is None
    assert "institution" in rows[0]["_unresolved_fields"]
    issue = next(
        issue
        for issue in context._personal_detail_extraction_issues
        if issue["issue_code"] == "candidate_b_credit_agreement_observation_conflict"
    )
    assert issue["observed_value"]["conflicting_fields"] == ["institution"]


def test_native_parser_defers_incomplete_cells_to_coordinated_page_evidence() -> None:
    table = _table(
        "credit-line-incomplete",
        [
            ["授信协议标识", "授信额度用途"],
            ["", "循环额度"],
        ],
    )
    page = SimpleNamespace(page_number=1, source_page_number=1, tables=[table])
    recorded: list[dict[str, object]] = []

    context = SimpleNamespace(
        pages=[page],
        reading_order_by_logical={1: 1},
        tables_continue=lambda _left, _right: None,
        corrected_evidence_pages=lambda: [],
        _personal_detail_extraction_issues=recorded,
    )

    records = PBOCPersonalDetailNativeParser(context).records("credit_lines")

    assert records == []
    # The native pass only records observations.  Missing-slot reporting is
    # deferred until the corrected page evidence has been reconciled, so a
    # transient first-pass miss cannot survive as a false-positive issue.
    assert context._personal_detail_extraction_issues == []
    assert not hasattr(context, "full_page_ocr_evidence")


def test_native_parser_segments_repeated_liabilities_from_corrected_page_rows() -> None:
    def line(text: str, x: float, y: float) -> dict[str, object]:
        return {"text": text, "bbox": [x, y, x + 120, y + 18], "confidence": 0.98}

    context = SimpleNamespace(
        pages=[],
        corrected_evidence_pages=lambda: [
            {
                "page": 7,
                "source_page": 4,
                "lines": [
                    line("(六)相关还款责任信息", 20, 10),
                    line("账户1", 20, 40),
                    line("责任人类型", 20, 70),
                    line("还款责任金额", 180, 70),
                    line("保证合同编号", 340, 70),
                    line("保证人", 20, 100),
                    line("4,000,000", 180, 100),
                    line("G-001", 340, 100),
                    line("账户2", 20, 140),
                    line("责任人类型", 20, 170),
                    line("还款责任金额", 180, 170),
                    line("保证合同编号", 340, 170),
                    line("保证人", 20, 200),
                    line("2,400,000", 180, 200),
                    line("G-002", 340, 200),
                    line("(七)授信协议信息", 20, 240),
                ],
            }
        ],
    )

    records = PBOCPersonalDetailNativeParser(context).records("repayment_liability_records")

    assert [record.fields["保证合同编号"] for record in records] == ["G-001", "G-002"]
    assert [record.fields["还款责任金额"] for record in records] == ["4,000,000", "2,400,000"]


def test_native_parser_preserves_corrected_liability_snapshot_slot_and_family() -> None:
    def line(text: str, x: float, y: float, width: float = 120) -> dict[str, object]:
        return {
            "text": text,
            "bbox": [x, y, x + width, y + 18],
            "source_bbox": [x + 5, y + 5, x + width + 5, y + 23],
            "confidence": 0.98,
        }

    context = SimpleNamespace(
        pages=[],
        corrected_evidence_pages=lambda: [
            {
                "page": 7,
                "source_page": 4,
                "lines": [
                    line("(六)相关还款责任信息", 20, 10),
                    line("有相关还款责任的个人借款", 20, 35, 240),
                    line("账户1", 20, 60),
                    line("责任人类型", 20, 90),
                    line("还款责任金额", 180, 90),
                    line("保证合同编号", 340, 90),
                    line("保证", 20, 120),
                    line("200,000", 180, 120),
                    line("B31015210H0001AHIH0001", 340, 120),
                    line("截至2025 年4 月25 日", 20, 150, 220),
                    line("(七)授信协议信息", 20, 190),
                ],
            }
        ],
    )

    records = PBOCPersonalDetailNativeParser(context).records("repayment_liability_records")

    assert len(records) == 1
    assert records[0].fields["__printed_sequence"] == "1"
    assert records[0].fields["__party_category"] == "person"
    assert records[0].fields["__snapshot_date"] == "截至2025 年4 月25 日"
    snapshot_ref = records[0].source_refs_by_field["__snapshot_date"][0]
    assert snapshot_ref["geometry_scope"] == "cell"
    assert snapshot_ref["bbox"] == [25.0, 155.0, 245.0, 173.0]


def test_enterprise_liability_heading_sets_organization_category() -> None:
    def line(text: str, x: float, y: float) -> dict[str, object]:
        return {"text": text, "bbox": [x, y, x + 160, y + 18], "confidence": 0.98}

    context = SimpleNamespace(
        pages=[],
        corrected_evidence_pages=lambda: [
            {
                "page": 7,
                "source_page": 4,
                "lines": [
                    line("(六)相关还款责任信息", 20, 10),
                    line("有相关还款责任的企业借款", 20, 35),
                    line("账户1", 20, 60),
                    line("责任人类型", 20, 90),
                    line("还款责任金额", 180, 90),
                    line("保证合同编号", 340, 90),
                    line("保证人", 20, 120),
                    line("1,000,000", 180, 120),
                    line("ORGCONTRACT0001", 340, 120),
                    line("(七)授信协议信息", 20, 160),
                ],
            }
        ],
    )

    records = PBOCPersonalDetailNativeParser(context).records("repayment_liability_records")

    assert len(records) == 1
    assert records[0].fields["__party_category"] == "organization"


def test_native_parser_carries_credit_agreement_card_across_corrected_pages() -> None:
    def line(text: str, x: float, y: float) -> dict[str, object]:
        return {"text": text, "bbox": [x, y, x + 150, y + 18], "confidence": 0.98}

    context = SimpleNamespace(
        pages=[],
        reading_order_by_logical={8: 1, 9: 2, 10: 3},
        reading_order_resolution={"resolved": True, "authoritative": True},
        corrected_evidence_pages=lambda: [
            {
                "page": 8,
                "source_page": 4,
                "lines": [
                    line("(七)授信协议信息", 20, 10),
                    line("授信协议1", 20, 40),
                    line("授信协议标识", 20, 70),
                    line("授信额度用途", 220, 70),
                ],
            },
            {
                "page": 9,
                "source_page": 5,
                "lines": [
                    line("AGREEMENT0001", 20, 20),
                    line("循环贷款额度", 220, 20),
                ],
            },
            {
                "page": 10,
                "source_page": 6,
                "lines": [
                    line("账户7（授信协议标识：ACCOUNT0007）", 20, 20),
                    line("授信协议标识", 20, 50),
                    line("授信额度用途", 220, 50),
                    line("ACCOUNT0007", 20, 80),
                    line("循环贷款额度", 220, 80),
                    line("非信贷交易信息", 20, 120),
                ],
            },
        ],
    )

    records = PBOCPersonalDetailNativeParser(context).records("credit_lines")

    assert len(records) == 1
    assert records[0].fields["授信协议标识"] == "AGREEMENT0001"
    assert records[0].fields["授信额度用途"] == "循环贷款额度"
    assert records[0].fields["__printed_sequence"] == "1"
    assert records[0].binding_quality_by_field["__printed_sequence"] == (
        "canonical_card_anchor"
    )
    sequence_ref = records[0].source_refs_by_field["__printed_sequence"][0]
    assert sequence_ref["source"] == "personal_detail_corrected_page_cell"
    assert sequence_ref["binding"] == "canonical_card_anchor"


def test_native_parser_accepts_only_declared_ocr_label_aliases() -> None:
    table = _table(
        "credit-line",
        [
            ["授信协议标识", "授信额度用途", "营理机构"],
            ["AGREEMENT0001", "循环额度", "示例银行股份有限公司"],
        ],
    )
    table.bbox = [20, 100, 580, 300]
    table.metadata["canonical_template_id"] = "credit_agreement"
    page = SimpleNamespace(
        page_number=1,
        source_page_number=1,
        canonical_template_id="credit_agreement",
        tables=[table],
        texts=[SimpleNamespace(content="授信协议1", bbox=[20, 78, 120, 98])],
    )
    context = SimpleNamespace(
        pages=[page],
        reading_order_by_logical={1: 1},
        tables_continue=lambda _left, _right: None,
    )

    records = PBOCPersonalDetailNativeParser(context).records("credit_lines")

    assert len(records) == 1
    assert records[0].fields["管理机构"] == "示例银行股份有限公司"
    assert records[0].confidence == 0.96
    assert records[0].binding_quality_by_field["管理机构"] == "native_label_column"


def test_scanned_auxiliary_exposes_only_the_single_candidate_b_result() -> None:
    expected = {
        "credit_accounts": [{"account_id": "account:1"}],
        "credit_lines": [{"credit_line_id": "line:1"}],
        "repayment_liability_records": [{"liability_id": "liability:1"}],
    }
    context = SimpleNamespace(
        candidate_b_extraction=lambda _text: SimpleNamespace(business=expected),
        scanned_business=lambda _text: (_ for _ in ()).throw(AssertionError("legacy scanned path called")),
        native_business=lambda _text: (_ for _ in ()).throw(AssertionError("legacy native path called")),
    )

    result = PersonalDetailScannedVariant().extract_auxiliary_business(
        context,
        "",
        content_mode="scanned_ocr",
    )

    assert result == expected
    assert result is not expected


def test_printed_reading_order_tolerates_large_observed_page_gaps() -> None:
    pages = [
        SimpleNamespace(
            page_number=30,
            source_page_number=3,
            width=600,
            height=800,
            texts=[SimpleNamespace(content="第 8 页，共 8 页", bbox=[220, 770, 380, 790])],
        ),
        SimpleNamespace(
            page_number=10,
            source_page_number=1,
            width=600,
            height=800,
            texts=[SimpleNamespace(content="第 1 页，共 8 页", bbox=[220, 770, 380, 790])],
        ),
        SimpleNamespace(
            page_number=20,
            source_page_number=2,
            width=600,
            height=800,
            texts=[SimpleNamespace(content="第 4 页，共 8 页", bbox=[220, 770, 380, 790])],
        ),
    ]
    result = SimpleNamespace(pages=pages, entities=SimpleNamespace(domain_specific={}))

    assert _printed_reading_order(result) == {10: 1, 20: 2, 30: 3}


def test_sections_are_derived_from_registered_canonical_roles() -> None:
    def page(number: int, template_id: str, text: str) -> SimpleNamespace:
        return SimpleNamespace(
            page_number=number,
            canonical_template_id=template_id,
            texts=[SimpleNamespace(content=text)],
            tables=[],
        )

    result = SimpleNamespace(
        pages=[
            page(1, "report_header_and_identity", "个人基本信息"),
            page(3, "information_summary", "信息概要"),
            page(8, "credit_account_detail", "信贷交易信息明细"),
            page(45, "public_information", "公共信息明细"),
            page(50, "annotations_and_inquiries", "机构查询记录明细"),
            page(55, "report_explanation", "报告说明"),
        ]
    )

    sections = PersonalDetailScannedVariant().build_sections(result, "")
    by_type = {section["type"]: section for section in sections}

    assert by_type["credit_details"]["page_start"] == 8
    assert by_type["public_records"]["page_start"] == 45
    assert by_type["inquiries"]["page_start"] == 50
    assert by_type["report_explanation"]["page_start"] == 55


def test_section_roots_do_not_promote_summary_labels_or_report_explanations() -> None:
    def page(number: int, template_id: str, text: str) -> SimpleNamespace:
        return SimpleNamespace(
            page_number=number,
            canonical_template_id=template_id,
            texts=[SimpleNamespace(content=text)],
            tables=[],
        )

    result = SimpleNamespace(
        pages=[
            page(1, "report_header_and_identity", "一 个人基本信息"),
            page(2, "information_summary", "二 信息概要 非循环贷账户 （六）查询记录概要"),
            page(4, "credit_account_detail", "三 信贷交易信息明细"),
            page(6, "credit_account_detail", "机构说明 添加日期"),
            page(11, "postpaid_detail", "四 非信贷交易信息明细"),
            page(12, "public_information", "五 公共信息明细"),
            page(13, "annotations_and_inquiries", "异议标注 六 查询记录 机构查询记录明细"),
            page(14, "report_explanation", "报告说明 本人声明与异议标注的含义"),
            page(15, "report_explanation", "编制说明"),
        ]
    )

    sections = PersonalDetailScannedVariant().build_sections(result, "")
    by_type = {section["type"]: section for section in sections}

    assert by_type["credit_details"]["page_start"] == 4
    assert by_type["inquiries"]["page_start"] == 13
    assert by_type["public_records"]["page_start"] == 12
    assert by_type["statements"]["page_end"] == 6
    assert by_type["annotations"]["page_end"] == 13
    assert by_type["report_explanation"]["page_end"] == 15


def test_identifier_only_liability_rows_are_redundant_not_business_records() -> None:
    assert liability_record_is_substantive({"liability_id": "row:1", "source_refs": []}) is False
    assert liability_record_is_substantive({"liability_id": "row:2", "responsibility_type": "保证人"}) is True


def test_v2_projection_keeps_extraction_issues_as_typed_control_rows() -> None:
    issue = make_issue(
        category="ocr_cell_level_error",
        issue_code="pboc_cell_contract_unresolved",
        message="review",
        target_dataset="credit_accounts",
        field_name="balance",
        observed_value="12O0",
    )

    projected = project_personal_detail_datasets({"personal_detail_extraction_issues": [issue]})
    issue_row = projected["extraction_issues"][0]
    values = issue_row.get("normalized", issue_row)

    assert values["target_dataset"] == "credit_accounts"
    assert values["field_name"] == "balance"
    assert values["observed_value"] == "12O0"
