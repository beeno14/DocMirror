# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Closed-world contracts for printed detailed-report section headings.

Section numbering is presentation metadata: different report revisions can
insert or omit sections without changing a section's semantic title.  Callers
therefore register the exact canonical title and accept an optional, complete
parenthesized Chinese numeral, but never infer a title from a numeral or fuzzy
text.
"""

from __future__ import annotations

import re
from typing import Any

REGISTERED_SECTION_TEMPLATE_BY_TITLE = {
    "个人基本信息": "report_header_and_identity",
    "信息概要": "information_summary",
    "信贷交易信息明细": "credit_account_detail",
    "相关还款责任信息": "repayment_responsibility",
    "授信协议信息": "credit_agreement",
    "非信贷交易信息明细": "postpaid_detail",
    "公共信息明细": "public_information",
    "查询记录明细": "annotations_and_inquiries",
    "报告说明": "report_explanation",
}

ACCOUNT_FAMILY_BY_TITLE = {
    "非循环贷账户": "non_revolving_loan",
    "循环贷账户一": "revolving_loan_subaccount",
    "循环贷账户二": "revolving_loan_account",
    "循环贷账户（一）": "revolving_loan_subaccount",
    "循环贷账户（二）": "revolving_loan_account",
    "贷记卡账户": "credit_card",
    "准贷记卡账户": "quasi_credit_card",
}

# Subsection titles may share one physical page with the tail of the preceding
# PBOC section.  They identify only the semantic owner of a following table;
# they never classify a whole page or authorize a table whose own exact header
# schema disagrees.  Keep this finite catalog separate from the top-level
# section graph because official PBOC revisions can renumber or regroup these
# subsections without changing their field semantics.
REGISTERED_SUBSECTION_TEMPLATE_BY_TITLE = {
    "后付费记录": "postpaid_detail",
    "后付费记录账户": "postpaid_detail",
    "欠税记录": "public_information",
    "民事判决记录": "public_information",
    "强制执行记录": "public_information",
    "行政处罚记录": "public_information",
    "住房公积金参缴记录": "public_information",
    "执业资格记录": "public_information",
    "行政奖励记录": "public_information",
    "异议标注": "annotations_and_inquiries",
    "查询记录": "annotations_and_inquiries",
    "查询记录机构查询记录明细": "annotations_and_inquiries",
    "机构查询记录明细": "annotations_and_inquiries",
    "本人查询记录明细": "annotations_and_inquiries",
}

_CHINESE_SECTION_ORDINAL = r"[〇零一二三四五六七八九十百]{1,5}"
_OPTIONAL_SECTION_ORDINAL = (
    rf"(?:(?:[（(]{_CHINESE_SECTION_ORDINAL}[）)])|{_CHINESE_SECTION_ORDINAL})?"
)
_REGISTERED_SECTION_HEADING_RE = re.compile(
    rf"^{_OPTIONAL_SECTION_ORDINAL}"
    rf"(?P<title>{'|'.join(re.escape(title) for title in REGISTERED_SECTION_TEMPLATE_BY_TITLE)})$"
)
_ACCOUNT_FAMILY_HEADING_RE = re.compile(
    rf"^{_OPTIONAL_SECTION_ORDINAL}"
    rf"(?P<title>{'|'.join(re.escape(title) for title in ACCOUNT_FAMILY_BY_TITLE)})$"
)
_REGISTERED_SUBSECTION_HEADING_RE = re.compile(
    rf"^{_OPTIONAL_SECTION_ORDINAL}"
    rf"(?P<title>{'|'.join(re.escape(title) for title in sorted(REGISTERED_SUBSECTION_TEMPLATE_BY_TITLE, key=len, reverse=True))})$"
)


def canonical_registered_section_heading(value: Any) -> str | None:
    """Return one exact registered title, ignoring only its printed numeral.

    PBOC revisions print top-level section ordinals both bare (``一``) and in
    parentheses (``（一）``).  The complete semantic title remains mandatory;
    an ordinal never selects a section role by itself.
    """

    compact = re.sub(r"\s+", "", str(value or "")).rstrip("：:")
    match = _REGISTERED_SECTION_HEADING_RE.fullmatch(compact)
    return match.group("title") if match is not None else None


def canonical_account_family_heading(value: Any) -> str | None:
    """Return one exact PBOC account-family type, ignoring its outer numeral."""

    compact = re.sub(r"\s+", "", str(value or "")).rstrip("：:")
    match = _ACCOUNT_FAMILY_HEADING_RE.fullmatch(compact)
    return ACCOUNT_FAMILY_BY_TITLE[match.group("title")] if match is not None else None


def canonical_registered_subsection_heading(value: Any) -> tuple[str, str] | None:
    """Return ``(template_id, title)`` for one exact PBOC subsection title.

    Whitespace and an optional complete Chinese section numeral are
    presentation details.  Partial labels, surrounding prose, and substring
    matches are deliberately rejected so contents pages cannot own business
    tables.
    """

    compact = re.sub(r"\s+", "", str(value or "")).rstrip("：:")
    match = _REGISTERED_SUBSECTION_HEADING_RE.fullmatch(compact)
    if match is None:
        return None
    title = match.group("title")
    return REGISTERED_SUBSECTION_TEMPLATE_BY_TITLE[title], title
