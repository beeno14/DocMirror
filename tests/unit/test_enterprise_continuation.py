from __future__ import annotations

from types import SimpleNamespace

from docmirror.plugins.credit_report.enterprise_native.continuation import (
    CLOSED_SUMMARY_BODY_CONTRACT,
    FACILITY_VALUE_CONTRACT,
    EnterpriseContinuationResolver,
)
from docmirror.plugins.credit_report.enterprise_native.extraction import (
    extract_enterprise_continuation_audit,
)


def _table(table_id: str, rows: list[list[str]]) -> SimpleNamespace:
    return SimpleNamespace(
        table_id=table_id,
        headers=[],
        rows=[],
        metadata={"raw_rows": rows},
    )


def _result(*pages: tuple[int, list[SimpleNamespace]]) -> SimpleNamespace:
    return SimpleNamespace(
        pages=[SimpleNamespace(page_number=page_number, tables=tables) for page_number, tables in pages]
    )


def test_facility_continuation_accepts_only_valid_next_value_row() -> None:
    result = _result(
        (
            1,
            [
                _table(
                    "header",
                    [
                        ["非循环信用额度", "", "", "循环信用额度", "", ""],
                        ["总额", "已用额度", "剩余可用额度", "总额", "已用额度", "剩余可用额度"],
                    ],
                )
            ],
        ),
        (2, [_table("values", [["3000", "3000", "0", "4900", "4900", "0"]])]),
    )
    resolver = EnterpriseContinuationResolver(result)

    match = resolver.following_row(
        resolver.fragments[0],
        FACILITY_VALUE_CONTRACT,
    )

    assert match is not None
    assert match.fragment.table_id == "values"
    assert list(match.row) == ["3000", "3000", "0", "4900", "4900", "0"]


def test_same_column_count_does_not_authorize_unrelated_table_merge() -> None:
    result = _result(
        (
            1,
            [
                _table(
                    "closed_header",
                    [["", "正常类账户数", "关注类账户数", "不良类账户数", "合计"]],
                )
            ],
        ),
        (
            2,
            [
                _table(
                    "unrelated_header",
                    [["类型", "正常类账户数", "关注类账户数", "不良类账户数", "合计"]],
                )
            ],
        ),
    )
    resolver = EnterpriseContinuationResolver(result)

    assert (
        resolver.following_row(
            resolver.fragments[0],
            CLOSED_SUMMARY_BODY_CONTRACT,
        )
        is None
    )
    assert resolver.audit_rows() == [
        {
            "contract": "closed_credit_summary_body",
            "source_table_id": "closed_header",
            "candidate_table_id": "unrelated_header",
            "reason": "new_header",
        }
    ]


def test_nonadjacent_and_distant_tables_are_never_skipped_into_a_merge() -> None:
    result = _result(
        (
            1,
            [
                _table(
                    "closed_header",
                    [["", "正常类账户数", "关注类账户数", "不良类账户数", "合计"]],
                ),
                _table("intervening", [["说明", "值"]]),
            ],
        ),
        (3, [_table("plausible_but_distant", [["合计", "1", "0", "0", "1"]])]),
    )
    resolver = EnterpriseContinuationResolver(result)

    assert (
        resolver.following_row(
            resolver.fragments[0],
            CLOSED_SUMMARY_BODY_CONTRACT,
        )
        is None
    )
    assert resolver.audit_rows()[0]["candidate_table_id"] == "intervening"
    assert resolver.audit_rows()[0]["reason"] == "column_shape"


def test_continuation_audit_distinguishes_unexpected_records_without_mutating_input() -> None:
    result = _result((1, []))
    datasets = {
        "enterprise_current_credit_summary": [{"current_summary_id": "unexpected"}],
        "enterprise_closed_credit_summary": [],
        "enterprise_repayment_responsibility_summary": [],
        "repayment_liability_records": [],
        "enterprise_attachment_accounts": [],
    }

    audits = extract_enterprise_continuation_audit(result, datasets=datasets)

    assert datasets == {
        "enterprise_current_credit_summary": [{"current_summary_id": "unexpected"}],
        "enterprise_closed_credit_summary": [],
        "enterprise_repayment_responsibility_summary": [],
        "repayment_liability_records": [],
        "enterprise_attachment_accounts": [],
    }
    assert audits[0]["unresolved_record_count"] == 0
    assert audits[0]["unexpected_record_count"] == 1
    assert audits[0]["reconciliation_status"] == "unresolved"
