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


def _table(table_id: str, rows: list[list[str]]) -> SimpleNamespace:
    return SimpleNamespace(
        table_id=table_id,
        metadata={"raw_rows": rows},
        headers=[],
        rows=[],
        bbox=[20, 20, 580, 300],
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
        pages=[SimpleNamespace(page_number=1, source_page_number=1, tables=[header_only, substantive])]
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
    result = SimpleNamespace(
        pages=[SimpleNamespace(page_number=4, source_page_number=2, tables=[table], texts=[])]
    )

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
    result = SimpleNamespace(
        pages=[SimpleNamespace(page_number=4, source_page_number=2, tables=[table], texts=[])]
    )

    row = _extract_liabilities(result)[0]

    assert row["overdue_months"] == 0
    assert row["repayment_status_code"] is None
    assert getattr(result, "_personal_detail_extraction_issues", []) == []


def test_liability_currency_is_unknown_and_reported_instead_of_defaulted() -> None:
    table = _complete_liability_table("liability-bad-currency", currency="人民币无")
    result = SimpleNamespace(
        pages=[SimpleNamespace(page_number=4, source_page_number=2, tables=[table], texts=[])]
    )

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
        result = SimpleNamespace(
            pages=[SimpleNamespace(page_number=4, source_page_number=2, tables=[table], texts=[])]
        )

        rows = _extract_liabilities(result)

        assert rows[0]["currency"] == expected
        assert rows[0]["reporting_amount_currency"] == expected
        assert not any(
            issue.get("field_name") == "currency"
            for issue in getattr(result, "_personal_detail_extraction_issues", [])
        )


def test_explicit_liability_placeholder_is_known_absence_not_failure() -> None:
    table = _complete_liability_table("liability-explicit-absence", currency="--")
    result = SimpleNamespace(
        pages=[SimpleNamespace(page_number=4, source_page_number=2, tables=[table], texts=[])]
    )

    rows = _extract_liabilities(result)

    assert rows[0]["currency"] is None
    assert not any(
        issue.get("field_name") == "currency"
        for issue in getattr(result, "_personal_detail_extraction_issues", [])
    )


def test_equal_provenance_liability_conflict_is_withheld_and_linked() -> None:
    first = _complete_liability_table("liability-conflict-a", responsibility_amount="200,000", top=20)
    second = _complete_liability_table("liability-conflict-b", responsibility_amount="300,000", top=260)
    result = SimpleNamespace(
        pages=[SimpleNamespace(page_number=4, source_page_number=2, tables=[first, second], texts=[])]
    )

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


def test_account_currency_unique_token_with_residue_is_retained_and_reported() -> None:
    examples = (
        ("人民币元 共同借款标志", "CNY", "共同借款标志"),
        ("美元 江", "USD", "江"),
        ("福 人民币元 第", "CNY", "福第"),
        ("福 澳元 第", "AUD", "福第"),
    )
    for index, (raw, expected, residue) in enumerate(examples, start=1):
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
        issues = collect_extraction_issues(context)
        assert len(issues) == 1
        assert issues[0]["issue_code"] == "candidate_b_currency_token_residue_corrected"
        assert issues[0]["target_record_id"] == account["account_id"]
        assert issues[0]["field_name"] == "currency"
        assert issues[0]["observed_value"] == {"raw": raw, "residue": residue}
        assert issues[0]["severity"] == "info"
        assert issues[0]["status"] == "resolved"


def test_account_exact_finite_currency_token_is_silent() -> None:
    examples = (
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


def test_credit_agreement_currency_residue_is_retained_and_field_reported(monkeypatch) -> None:
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

    assert rows[0]["currency"] == "USD"
    assert rows[0]["account_currency"] == "USD"
    assert rows[0]["reporting_amount_currency"] == "USD"
    issues = [
        issue
        for issue in collect_extraction_issues(context)
        if issue["issue_code"] == "candidate_b_currency_token_residue_corrected"
    ]
    assert len(issues) == 1
    assert issues[0]["target_record_id"] == rows[0]["credit_line_id"]
    assert issues[0]["observed_value"] == {"raw": "美元 江", "residue": "江"}
    assert issues[0]["severity"] == "info"
    assert issues[0]["status"] == "resolved"


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
    for raw in ("人民币元美元", "澳元美元", "ZZZ"):
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
    table = _table(
        "institutional",
        [
            ["编号", "查询日期", "查询机构", "查询原因"],
            ["1", "2024.01.03", "示例银行", "贷款审批"],
            ["3", "", "", ""],
        ],
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


def test_source_ledger_rejects_isolated_account_sequence_outlier() -> None:
    endpoint, outliers = _credible_sequence_endpoint(set(range(1, 28)) | {115})

    assert endpoint == 27
    assert outliers == [115]


def test_account_source_ledger_counts_lin_numbered_and_unnumbered_families() -> None:
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

    assert ledger["credit_accounts"] == 45
    assert ledger["account_family_endpoints"] == {
        "non_revolving_loan": 22,
        "credit_card": 22,
    }
    assert ledger["account_family_source_populations"] == {
        "credit_card": 22,
        "non_revolving_loan": 22,
        "revolving_loan_account": 1,
    }
    assert ledger["account_family_unnumbered_anchor_counts"] == {
        "revolving_loan_account": 1
    }


def test_account_source_ledger_is_conservative_for_sparse_and_weak_anchors() -> None:
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

    assert ledger["credit_accounts"] == 5
    assert ledger["account_family_endpoints"] == {"non_revolving_loan": 3}
    assert ledger["account_family_sequence_outliers"] == {
        "non_revolving_loan": [115]
    }
    assert ledger["account_family_source_populations"] == {
        "non_revolving_loan": 3,
        "revolving_loan_account": 2,
    }
    assert "credit_card" not in ledger["account_family_source_populations"]


def test_exact_single_weak_account_anchor_is_a_bounded_source_witness() -> None:
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

    assert ledger["credit_accounts"] == 1
    assert ledger["account_family_source_populations"] == {
        "revolving_loan_account": 1
    }
    assert ledger["account_family_unnumbered_anchor_counts"] == {
        "revolving_loan_account": 1
    }


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
    result = SimpleNamespace(
        pages=[],
        corrected_evidence_pages=lambda: [
            {
                "page": 1,
                "source_page": 1,
                "lines": [
                    {"text": "账户1（授信协议标识：ACCOUNT0001）"},
                    {"text": "授信协议标识"},
                ],
            },
            {
                "page": 2,
                "source_page": 2,
                "lines": [
                    {"text": "（五）授信协议信息"},
                    {"text": "授信协议1"},
                    {"text": "授信协议标识"},
                    {"text": "授信协议2"},
                    {"text": "授信协议标识"},
                    {"text": "公共信息明细"},
                ],
            },
        ],
    )

    ledger = _source_completeness_ledger(result)

    assert ledger["credit_agreements"] == 2


def test_agreement_ledger_prefers_printed_endpoint_over_duplicate_primary_labels() -> None:
    result = SimpleNamespace(
        pages=[],
        corrected_evidence_pages=lambda: [
            {
                "page": 24,
                "source_page": 12,
                "lines": [
                    {"text": "（五）授信协议信息"},
                    {"text": "授信协议1"},
                    {"text": "授信协议标识"},
                    {"text": "授信协议标识"},
                    {"text": "授信协议2"},
                    {"text": "授信协议标识"},
                    {"text": "公共信息明细"},
                ],
            }
        ],
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
        issue["source_refs"][0]["logical_page"] == 1
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
    page = SimpleNamespace(page_number=1, source_page_number=1, tables=[table])
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
    page = SimpleNamespace(page_number=1, source_page_number=1, tables=[table])
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
