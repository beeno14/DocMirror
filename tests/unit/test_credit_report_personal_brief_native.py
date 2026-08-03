# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from docmirror.plugins.credit_report.personal_brief_native.extraction import (
    _personal_header_datasets,
)
from docmirror.plugins.credit_report.personal_brief_native.schema import (
    personal_brief_data_dictionary,
)


def test_personal_brief_metadata_emits_marital_status() -> None:
    blocks = [
        (
            1,
            "个人信用报告 报告编号：2026071900012345678901 "
            "报告时间：2026-07-19 09:08:07 姓名：张三 "
            "证件类型：身份证 证件号码：11010519491231002X 已婚",
        ),
        (1, "信贷记录"),
    ]

    _identities, metadata, _amount_policy = _personal_header_datasets(
        SimpleNamespace(),
        blocks,
    )

    assert metadata[0]["marital_status"] == "married"
    assert (
        personal_brief_data_dictionary()["datasets"]["personal_report_metadata"]
        ["columns"]["marital_status"]["type"]
        == "enum"
    )
