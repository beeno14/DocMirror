# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from docmirror.plugins.credit_report.pboc_vocabularies import (
    PBOC_INQUIRY_REASON_FORMS,
    PBOC_PERSONAL_INQUIRY_REASON_CONTRACT_ID,
    PBOC_PERSONAL_INQUIRY_REASON_CONTRACT_VERSION,
    canonical_pboc_inquiry_reason,
    longest_pboc_inquiry_reason_suffix,
    pboc_inquiry_reason_root,
    pboc_inquiry_reason_suffix,
)


def test_pboc_inquiry_reason_registry_is_versioned_and_cross_revision() -> None:
    assert PBOC_PERSONAL_INQUIRY_REASON_CONTRACT_ID == "pboc.personal_credit_report.inquiry_reason"
    assert PBOC_PERSONAL_INQUIRY_REASON_CONTRACT_VERSION == "1.0.0"
    assert {
        "贷后管理",
        "司法调查",
        "公积金提取复核",
        "特约商户实名审查",
        "本人查询(临柜)",
        "本人查询(征信中心柜台)",
    } <= PBOC_INQUIRY_REASON_FORMS


@pytest.mark.parametrize(
    ("observed", "canonical", "root"),
    [
        ("贷后管理", "贷后管理", "贷后管理"),
        (" 本人查询（商业银行网上 银行） ", "本人查询(商业银行网上银行)", "本人查询"),
        ("本人查询(互联网个人信用信息服务平台)", "本人查询(互联网个人信用信息服务平台)", "本人查询"),
    ],
)
def test_pboc_inquiry_reason_typography_does_not_change_semantics(
    observed: str,
    canonical: str,
    root: str,
) -> None:
    assert canonical_pboc_inquiry_reason(observed) == canonical
    assert pboc_inquiry_reason_root(observed) == root


@pytest.mark.parametrize(
    "observed",
    [
        "货后管理",
        "贷后智理",
        "某货后管理服务有限公司",
        "贷后管理备注",
        "法人代表、负责人、高管",
        "",
    ],
)
def test_ocr_scars_substrings_and_truncated_reasons_do_not_enter_pboc_vocabulary(
    observed: str,
) -> None:
    assert canonical_pboc_inquiry_reason(observed) is None
    assert pboc_inquiry_reason_root(observed) is None


def test_reason_suffix_is_longest_registered_complete_suffix_only() -> None:
    source = "18 2024.06.01 示例银行 法人代表、负责人、高管等资信审查"
    assert longest_pboc_inquiry_reason_suffix(source) == "法人代表、负责人、高管等资信审查"
    assert pboc_inquiry_reason_suffix(source) == (
        "法人代表、负责人、高管等资信审查",
        "法人代表、负责人、高管等资信审查",
        source.index("法人代表"),
    )
    assert longest_pboc_inquiry_reason_suffix("某货后管理服务有限公司") is None
    assert longest_pboc_inquiry_reason_suffix("贷后管理备注") is None
    assert longest_pboc_inquiry_reason_suffix("示例征信服务其他") is None
    assert longest_pboc_inquiry_reason_suffix("示例征信服务 其他") == "其他"
    self_source = "1 2024.01.01 本人 本人查询（临柜）"
    assert pboc_inquiry_reason_suffix(self_source) == (
        "本人查询(临柜)",
        "本人查询",
        self_source.index("本人查询"),
    )
