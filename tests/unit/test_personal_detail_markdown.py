# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

from docmirror.plugins.credit_report.community_plugin import _CreditReportCommunityBundle
from docmirror.plugins.credit_report.personal_detail_scanned.markdown import (
    render_personal_detail_business_markdown,
)


def test_personal_detail_markdown_hides_semantic_ids_and_diagnostics() -> None:
    semantic = {
        "domain": {
            "facts": {
                "report_subtype": "personal_detail",
                "content_mode": "scanned_ocr",
            }
        },
        "datasets": [
            {
                "name": "subject_profile",
                "columns": [
                    {"key": "subject_profile_id", "label": "个人基本资料ID"},
                    {"key": "subject_name", "label": "姓名"},
                    {"key": "gender", "label": "性别"},
                    {"key": "extraction_status", "label": "extraction status"},
                ],
                "rows": [
                    {
                        "record_id": "personal_profile:1",
                        "normalized": {
                            "subject_profile_id": "personal_profile:1",
                            "subject_name": "洪晓鑫",
                            "gender": "男",
                            "extraction_status": "review",
                        },
                    }
                ],
            },
            {
                "name": "credit_accounts",
                "columns": [
                    {"key": "account_id", "label": "账户ID"},
                    {"key": "account_identifier", "label": "账户标识"},
                    {"key": "management_institution", "label": "管理机构"},
                ],
                "rows": [
                    {
                        "record_id": "credit_account:credit_card:1",
                        "normalized": {
                            "account_id": "credit_account:credit_card:1",
                            "account_identifier": "621234",
                            "management_institution": "示例银行",
                        },
                    }
                ],
            },
            {
                "name": "extraction_issues",
                "columns": [{"key": "extraction_issue_id", "label": "问题ID"}],
                "rows": [
                    {
                        "record_id": "personal_detail_extraction_issue:abc",
                        "normalized": {"extraction_issue_id": "personal_detail_extraction_issue:abc"},
                    }
                ],
            },
        ]
    }
    snapshot = deepcopy(semantic)

    markdown = render_personal_detail_business_markdown(semantic)

    assert semantic == snapshot
    assert markdown.startswith("# 个人信用报告\n")
    assert "## 个人基本资料" in markdown
    assert "洪晓鑫" in markdown
    assert "性别" in markdown
    assert "男" in markdown
    assert "621234" in markdown
    assert "示例银行" in markdown
    assert "personal_profile:1" not in markdown
    assert "credit_account:credit_card:1" not in markdown
    assert "personal_detail_extraction_issue" not in markdown
    assert "extraction status" not in markdown
    assert "字段观测" not in markdown
    assert "<!--" not in markdown

    bundle = object.__new__(_CreditReportCommunityBundle)
    bundle.semantic_payload = lambda: semantic
    assert bundle.render_markdown() == markdown
    assert bundle.render_enhanced_markdown(semantic) == markdown
