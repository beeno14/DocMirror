# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One record-boundary grammar and source-field vocabulary for Personal Brief.

Boundary candidates are validated before they can end the preceding record.
Field probes observe printed clauses, independently of whether their value
decoder succeeds; they never supply replacement business values.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from docmirror.plugins.credit_report.personal_brief_native.date_rules import (
    PERSONAL_BRIEF_DATE_PATTERN,
    PERSONAL_BRIEF_MONTH_OR_DATE_PATTERN,
    PERSONAL_BRIEF_YEAR_PATTERN,
)

ACCOUNT_DATE_PATTERN = PERSONAL_BRIEF_DATE_PATTERN
ACCOUNT_MONTH_OR_DATE_PATTERN = PERSONAL_BRIEF_MONTH_OR_DATE_PATTERN
ACCOUNT_ACTION_PATTERN = (
    r"(?:(?:发\s*放|办\s*理|开\s*立|提\s*供)(?:\s*了)?(?:\s*的)?"
    r"|为(?=[\u3400-\u9fff（）()\s]{0,40}贷\s*款\s*授\s*信))"
)
_ACCOUNT_START_RE = re.compile(
    rf"{ACCOUNT_DATE_PATTERN}"
    rf"(?=(?P<institution>(?:(?!{ACCOUNT_DATE_PATTERN})[^，,。；;]){{4,180}}?)"
    rf"{ACCOUNT_ACTION_PATTERN})"
)
OTHER_BUSINESS_TYPES = (
    "约定购回式证券交易",
    "股票质押式回购交易",
    "融资融券交易",
    "融资租赁",
)
_BUSINESS_MARKERS = ("贷记卡", "准贷记卡", "贷款", "授信", *OTHER_BUSINESS_TYPES)

# Personal-brief loan labels may place a canonical scope qualifier either
# before or after ``贷款``.  The value must be captured as one business label;
# stopping at the first ``贷款`` is lossy for labels such as
# ``个人商用房贷款（包括商住两用房）``.
_LOAN_LABEL_CORE = (
    r"[\u3400-\u9fff]{1,30}"
    r"(?:[（(][^（）()]{1,30}[）)])?"
    r"贷款"
    r"(?:[（(][^（）()]{1,30}[）)])?"
)
_LOAN_LABEL_AFTER_CURRENCY_RE = re.compile(
    rf"[）)](?P<business_type>{_LOAN_LABEL_CORE})"
    rf"(?=[，,。；;]|{PERSONAL_BRIEF_YEAR_PATTERN}年|$)"
)
_LOAN_LABEL_RE = re.compile(
    rf"(?P<business_type>{_LOAN_LABEL_CORE})"
    rf"(?=[，,。；;]|{PERSONAL_BRIEF_YEAR_PATTERN}年|已|$)"
)
_RECORD_END = re.compile(
    r"相关还款责任信息|非信贷交易记录|公共记录|查询记录|资产处置信息|保证人代偿信息"
    r"|(?:从未发生过逾期|从未逾期过|发生过逾期)[^。]{0,100}?(?:明细如下|明细为|明细：)"
)


@dataclass(frozen=True)
class AccountNarrative:
    start: int
    end: int
    text: str


def account_narratives(text: str) -> tuple[AccountNarrative, ...]:
    """Return complete source records, never splitting on an unvalidated date."""

    starts = []
    for match in _ACCOUNT_START_RE.finditer(text):
        institution = re.sub(r"\s+", "", match.group("institution"))
        if not 4 <= len(institution) <= 80:
            continue
        # The product/action must belong to this opening clause, not to a
        # later dated record. In particular, 余额为0 is not 为…贷款授信.
        opening = re.split(
            rf"{ACCOUNT_DATE_PATTERN}|[。；;]", text[match.end() :], maxsplit=1
        )[0]
        opening = re.sub(r"\s+", "", opening)
        if "承担相关还款责任" in opening or not any(value in opening for value in _BUSINESS_MARKERS):
            continue
        starts.append(match.start())
    records = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(text)
        boundary = _RECORD_END.search(text, start, end)
        if boundary is not None:
            end = boundary.start()
        records.append(AccountNarrative(start, end, text[start:end].strip()))
    return tuple(records)


def personal_brief_loan_business_type(compact_text: str) -> str | None:
    """Return the complete printed loan label, including its qualifier."""

    for pattern in (_LOAN_LABEL_AFTER_CURRENCY_RE, _LOAN_LABEL_RE):
        match = pattern.search(compact_text)
        if match is not None:
            return match.group("business_type")
    return None


# (source-clause probe, semantic field, public dataset, public field).
# These intentionally probe labels rather than reusing value-decoding regexes.
# A printed but undecodable number/date must remain an extraction failure, not
# become "not_reported" merely because the value is absent from the result.
_FIELD_PROBES = (
    (r"信用额度(?![-—])", "credit_limit", "credit_accounts", "credit_limit"),
    (r"已使用额度(?![-—])", "used_amount", "credit_accounts", "used_amount"),
    (r"(?:发放(?:了)?(?:的)?)[\d,]+元", "loan_amount", "credit_accounts", "loan_amount"),
    (r"(?<!分期)余额(?![-—])", "balance", "credit_accounts", "balance"),
    (r"大额专项分期余额", "unbilled_installment_balance", "credit_accounts", "unbilled_installment_balance"),
    (r"卡片尾号", "card_tail", "credit_accounts", "card_tail"),
    (r"额度有效期至", "credit_line_expiry_date", "credit_accounts", "credit_line_expiry_date"),
    (r"额度(?:有效期至|长期有效)", "credit_line_validity_type", "credit_accounts", "credit_line_validity_type"),
    (rf"{ACCOUNT_MONTH_OR_DATE_PATTERN}到期", "contract_maturity_date", "credit_accounts", "contract_maturity_date"),
    (r"(?:不可|可)循环使用", "is_revolving", "credit_accounts", "is_revolving"),
    (r"截至", "information_as_of", "credit_accounts", "snapshot_date"),
    (r"当前[无有]逾期", "current_overdue", "credit_accounts", "current_overdue"),
    (r"(?:尚未|已)激活", "card_activation_state", "credit_accounts", "card_activation_state"),
    (r"(?:已结清|销户|已转出)", "termination_event_date", "credit_accounts", "termination_event_date"),
    (r"(?:已结清|销户|已转出)", "termination_event_type", "credit_accounts", "termination_event_type"),
    (r"最近5年内有", "overdue_months_last_5y", "credit_accounts", "overdue_months"),
    (r"最近5年内有", "overdue_months_last_5y", "overdue_records", "overdue_months"),
    (r"(?:逾期超过90天|发生过90天以上)", "over_90_days", "credit_accounts", "over_90_days"),
    (r"(?:其中\d+个月逾期超过90天|没有发生过90天以上)", "over_90_days_months", "overdue_records", "over_90_days_months"),
    (r"(?:贷记卡|准贷记卡)[（(][^（）()]+账户|元[（(][^（）()]+[）)]", "account_currency", "credit_accounts", "account_currency"),
)
_COMPILED_FIELD_PROBES = tuple(
    (re.compile(pattern), semantic_field, dataset, public_field)
    for pattern, semantic_field, dataset, public_field in _FIELD_PROBES
)


def account_source_fields(text: str) -> tuple[tuple[str, str, str], ...]:
    """Observed (semantic field, public dataset, public field) obligations."""

    compact = re.sub(r"\s+", "", text)
    fields = [
        ("management_institution", "credit_accounts", "institution"),
        ("business_type", "credit_accounts", "business_type"),
        ("open_date", "credit_accounts", "open_date"),
        ("account_type", "credit_accounts", "account_type"),
    ]
    fields.extend(
        (semantic_field, dataset, public_field)
        for pattern, semantic_field, dataset, public_field in _COMPILED_FIELD_PROBES
        if pattern.search(compact)
    )
    return tuple(dict.fromkeys(fields))
