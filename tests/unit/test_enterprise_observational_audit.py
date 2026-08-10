# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

from pytest import MonkeyPatch

from docmirror.models.entities.parse_result import PageContent, TableBlock, TextBlock
from docmirror.plugins.credit_report.community_plugin import _merge_warning_page_range
from docmirror.plugins.credit_report.enterprise_native import audit as audit_module
from docmirror.plugins.credit_report.enterprise_native.audit import (
    audit_warning_strings,
    build_enterprise_audit_report,
    safely_build_enterprise_audit_report,
)
from docmirror.plugins.credit_report.enterprise_native.extraction_validation import (
    build_enterprise_extraction_report,
)
from docmirror.plugins.credit_report.enterprise_native.ir import (
    build_canonical_enterprise_document,
)
from docmirror.plugins.credit_report.enterprise_native.pipeline import run_enterprise_pipeline


def _table(table_id: str, rows: list[list[str]]) -> TableBlock:
    return TableBlock(table_id=table_id, metadata={"raw_rows": rows})


def _result(pages: list[PageContent]) -> SimpleNamespace:
    return SimpleNamespace(pages=pages, confidence=1.0)


def _dictionary() -> dict[str, object]:
    return {
        "datasets": {
            "enterprise_public_utility_payment_records": {
                "columns": {
                    "cumulative_arrears": {
                        "label": "累计欠费金额",
                        "type": "money",
                    },
                    "currency": {"label": "币种", "type": "string"},
                    "amount_unit": {"label": "金额单位", "type": "string"},
                }
            },
            "enterprise_key_personnel": {
                "columns": {
                    "name": {"label": "姓名", "type": "string"},
                    "source_institution": {
                        "label": "信息来源机构",
                        "type": "string",
                    },
                    "update_date": {"label": "更新日期", "type": "date"},
                }
            },
        },
        "enums": {},
    }


def _empty_extraction_report() -> dict[str, object]:
    return {
        "protocol": "pboc-enterprise-extraction-failure",
        "version": "1.0.0",
        "status": "complete",
        "summary": {
            "failure_count": 0,
            "warning_count": 0,
            "checked_field_count": 0,
            "satisfied_field_count": 0,
            "failed_field_count": 0,
            "verified_equal_field_count": 0,
            "present_unverified_field_count": 0,
        },
        "failures": [],
    }


def _unit_document():
    return build_canonical_enterprise_document(
        _result(
            [
                PageContent(
                    page_number=1,
                    texts=[
                        TextBlock(
                            content=(
                                "企业信用报告（自主查询版）\n"
                                "如无特别说明，本报告中的金额类数据项单位均为万元。"
                            )
                        )
                    ],
                ),
                PageContent(
                    page_number=2,
                    tables=[
                        _table(
                            "utility",
                            [
                                ["公用事业单位名称", "累计欠费金额"],
                                ["中国移动", "0.30"],
                            ],
                        )
                    ],
                ),
            ]
        )
    )


def _utility_datasets(amount_unit: str) -> dict[str, list[dict[str, object]]]:
    return {
        "enterprise_public_utility_payment_records": [
            {
                "public_record_id": "utility:1",
                "normalized": {
                    "cumulative_arrears": "0.3",
                    "currency": "CNY",
                    "amount_unit": amount_unit,
                },
                "source_page": 2,
                "source_table_id": "utility",
                "source_refs": [
                    {
                        "source": "canonical_physical_table",
                        "page": 2,
                        "table_id": "utility",
                        "row": 1,
                    }
                ],
            }
        ]
    }


def test_amount_unit_mismatch_warns_without_mutating_extraction() -> None:
    document = _unit_document()
    datasets = _utility_datasets("CNY_1")
    before = deepcopy(datasets)

    report = build_enterprise_audit_report(
        document,
        datasets,
        extraction_report=_empty_extraction_report(),
        quality_flags=(),
        data_dictionary=_dictionary(),
    ).to_payload()

    assert datasets == before
    finding = next(
        item
        for item in report["findings"]
        if item["code"] == "ENTERPRISE_AUDIT_AMOUNT_UNIT_MISMATCH"
    )
    assert finding["severity"] == "warning"
    assert finding["dataset"] == "enterprise_public_utility_payment_records"
    assert finding["record_id"] == "utility:1"
    assert finding["field"] == "amount_unit"
    assert finding["path"].endswith("/utility:1/amount_unit")
    assert finding["evidence"]["expected_amount_unit"] == "CNY_10K"
    assert datasets["enterprise_public_utility_payment_records"][0]["normalized"][
        "amount_unit"
    ] == "CNY_1"


def test_report_default_unit_is_provenance_information_not_a_warning() -> None:
    datasets = _utility_datasets("CNY_10K")
    report = build_enterprise_audit_report(
        _unit_document(),
        datasets,
        extraction_report=_empty_extraction_report(),
        quality_flags=(),
        data_dictionary=_dictionary(),
    ).to_payload()

    finding = next(
        item
        for item in report["findings"]
        if item["code"] == "ENTERPRISE_AUDIT_AMOUNT_UNIT_REPORT_DEFAULT"
    )
    assert finding["severity"] == "info"
    assert report["status"] == "clear"
    assert audit_warning_strings(report) == ()


def test_amount_unit_audit_respects_field_specific_foreign_currencies() -> None:
    document = _unit_document()
    source_refs = [{"page": 2, "table_id": "utility", "row": 1}]
    datasets = {
        "enterprise_repayment_responsibility_accounts": [
            {
                "normalized": {
                    "responsibility_amount": "1",
                    "responsibility_currency": "USD",
                    "responsibility_amount_unit": "USD_10K",
                    "loan_or_credit_amount": "2",
                    "obligation_currency": "EUR",
                    "obligation_amount_unit": "EUR_10K",
                    "balance": "3",
                    "currency": "CNY",
                    "amount_unit": "CNY_10K",
                },
                "source_refs": source_refs,
                "field_info": {
                    field_name: {
                        "source_state": "reported",
                        "basis": "local_explicit_amount_unit",
                        "source_value": "CNY_10K",
                        "source_refs": source_refs,
                    }
                    for field_name in (
                        "responsibility_amount_unit",
                        "obligation_amount_unit",
                        "amount_unit",
                    )
                },
            }
        ]
    }
    dictionary = {
        "datasets": {
            "enterprise_repayment_responsibility_accounts": {
                "columns": {
                    "responsibility_amount": {
                        "label": "还款责任金额",
                        "type": "money",
                    },
                    "responsibility_currency": {
                        "label": "还款责任金额币种",
                        "type": "string",
                    },
                    "responsibility_amount_unit": {
                        "label": "还款责任金额单位",
                        "type": "string",
                    },
                    "loan_or_credit_amount": {
                        "label": "借款金额/信用额度",
                        "type": "money",
                    },
                    "obligation_currency": {
                        "label": "借款/授信金额币种",
                        "type": "string",
                    },
                    "obligation_amount_unit": {
                        "label": "借款/授信金额单位",
                        "type": "string",
                    },
                    "balance": {"label": "余额", "type": "money"},
                    "currency": {"label": "币种", "type": "string"},
                    "amount_unit": {"label": "金额单位", "type": "string"},
                }
            }
        },
        "enums": {},
    }

    report = build_enterprise_audit_report(
        document,
        datasets,
        extraction_report=_empty_extraction_report(),
        quality_flags=(),
        data_dictionary=dictionary,
    ).to_payload()

    assert not [
        item
        for item in report["findings"]
        if item["severity"] in {"warning", "error"}
    ]
    local_units = {
        item["field"]: item["evidence"]["amount_unit"]
        for item in report["findings"]
        if item["code"] == "ENTERPRISE_AUDIT_AMOUNT_UNIT_LOCAL_EVIDENCE"
    }
    assert local_units == {
        "amount_unit": "CNY_10K",
        "obligation_amount_unit": "EUR_10K",
        "responsibility_amount_unit": "USD_10K",
    }


def test_unresolved_field_is_addressable_and_does_not_remove_selected_value() -> None:
    document = build_canonical_enterprise_document(_result([PageContent(page_number=1)]))
    datasets = {
        "enterprise_key_personnel": [
            {
                "normalized": {"name": "甲", "source_institution": "甲银行"},
                "field_info": {
                    "source_institution": {
                        "source_state": "unresolved",
                        "basis": "conflicting_footer",
                    }
                },
                "source_refs": [{"page": 1, "table_id": "personnel", "row": 1}],
            }
        ]
    }
    before = deepcopy(datasets)

    report = build_enterprise_audit_report(
        document,
        datasets,
        extraction_report=_empty_extraction_report(),
        quality_flags=(),
        data_dictionary=_dictionary(),
    ).to_payload()

    assert datasets == before
    finding = next(
        item
        for item in report["findings"]
        if item["code"] == "ENTERPRISE_AUDIT_FIELD_UNRESOLVED"
    )
    assert finding["record_id"] == "enterprise_key_personnel:r000001"
    assert finding["field"] == "source_institution"
    assert datasets["enterprise_key_personnel"][0]["normalized"][
        "source_institution"
    ] == "甲银行"


def test_audit_runtime_failure_cannot_suppress_business_values(
    monkeypatch: MonkeyPatch,
) -> None:
    datasets = _utility_datasets("CNY_10K")
    before = deepcopy(datasets)

    def fail_audit(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("injected audit failure")

    monkeypatch.setattr(audit_module, "_amount_unit_findings", fail_audit)
    report = safely_build_enterprise_audit_report(
        _unit_document(),
        datasets,
        extraction_report=_empty_extraction_report(),
        quality_flags=(),
        data_dictionary=_dictionary(),
    ).to_payload()

    assert datasets == before
    assert report["status"] == "review_required"
    assert report["findings"][0]["code"] == "ENTERPRISE_AUDIT_INTERNAL_ERROR"
    assert report["summary"]["extraction_values_changed"] == 0


def test_compact_warning_conserves_repeated_source_page_envelope() -> None:
    warning: dict[str, object] = {"page_range": [2, 2]}

    _merge_warning_page_range(warning, [9])

    assert warning["page_range"] == [2, 9]


def test_observational_audit_does_not_change_legacy_failure_status() -> None:
    document = build_canonical_enterprise_document(_result([PageContent(page_number=1)]))
    report = build_enterprise_extraction_report(
        document,
        {},
        continuation_audit=(),
        dataset_completeness={},
        data_dictionary={"datasets": {}, "enums": {}},
        public_records=(
            {
                "public_record_id": "public:1",
                "record_type": "license",
                "field_info": {
                    "record_type": {
                        "source_state": "reported",
                        "conflicts": ["certification"],
                    }
                },
                "source_refs": [{"page": 1, "table_id": "public", "row": 1}],
            },
        ),
    ).to_payload()

    assert report["status"] == "partial"
    assert report["summary"]["failure_count"] == 0
    assert report["summary"]["warning_count"] == 1
    assert report["failures"][0]["severity"] == "warning"


def test_personnel_continuation_records_shared_footer_provenance() -> None:
    result = _result(
        [
            PageContent(
                page_number=1,
                texts=[TextBlock(content="企业信用报告（自主查询版）\n基本信息")],
                tables=[
                    _table(
                        "personnel-start",
                        [
                            ["职位", "姓名", "身份标识类型", "证件号码"],
                            ["董事长", "甲", "身份证", "110101198001010011"],
                        ],
                    )
                ],
            ),
            PageContent(
                page_number=2,
                tables=[
                    _table(
                        "personnel-continuation",
                        [
                            ["董事", "乙", "身份证", "110101198001010022"],
                            [
                                "信息来源机构：中国银行 更新日期：2014-8-12",
                                "",
                                "",
                                "",
                            ],
                        ],
                    )
                ],
            ),
        ]
    )

    artifacts = run_enterprise_pipeline(result)
    rows = artifacts.semantic_document.datasets["enterprise_key_personnel"]
    assert [row["normalized"]["name"] for row in rows] == ["甲", "乙"]
    assert all(row["normalized"]["source_institution"] == "中国银行" for row in rows)
    assert {ref["page"] for ref in rows[0]["source_refs"]} == {1}
    assert rows[0]["field_info"]["source_institution"]["basis"] == (
        "adjacent_continuation_footer"
    )
    assert {
        ref["page"]
        for ref in rows[0]["field_info"]["source_institution"]["source_refs"]
    } == {2}

    audit = artifacts.semantic_document.audit_report
    inherited = next(
        item
        for item in audit["findings"]
        if item["code"] == "ENTERPRISE_AUDIT_CONTINUATION_CONTEXT_INHERITED"
    )
    assert inherited["severity"] == "info"
    assert audit_warning_strings(audit) == ()


def test_validation_distinguishes_verified_numeric_from_present_string() -> None:
    document = build_canonical_enterprise_document(
        _result(
            [
                PageContent(
                    page_number=1,
                    tables=[
                        _table(
                            "profile",
                            [["发生信贷交易的机构数", "2"], ["姓名", "甲"]],
                        )
                    ],
                )
            ]
        )
    )
    dictionary = {
        "datasets": {
            "enterprise_credit_overview": {
                "columns": {
                    "credit_institution_count": {
                        "label": "发生信贷交易的机构数",
                        "type": "integer",
                    }
                }
            },
            "enterprise_key_personnel": {
                "columns": {"name": {"label": "姓名", "type": "string"}}
            },
        },
        "enums": {},
    }
    report = build_enterprise_extraction_report(
        document,
        {
            "enterprise_credit_overview": [
                {"normalized": {"credit_institution_count": 2}}
            ],
            "enterprise_key_personnel": [
                {
                    "normalized": {"name": "甲"},
                    "source_refs": [
                        {"page": 1, "table_id": "profile", "row": 1}
                    ],
                }
            ],
        },
        continuation_audit=(),
        dataset_completeness={},
        data_dictionary=dictionary,
    ).to_payload()

    assert report["summary"]["verified_equal_field_count"] == 1
    assert report["summary"]["present_unverified_field_count"] == 1
    assert report["status"] == "complete"
