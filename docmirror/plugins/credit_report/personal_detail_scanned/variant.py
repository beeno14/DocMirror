# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Orchestration adapter for scanned personal detailed credit reports."""

from copy import deepcopy
from typing import Any

from docmirror.plugins.credit_report.shared.variant import CreditReportVariantAdapter


class PersonalDetailScannedVariant(CreditReportVariantAdapter):
    """Keep scanned-detail extraction behind a dedicated variant boundary."""

    def __init__(self) -> None:
        super().__init__(
            variant_id="personal_detail_scanned",
            report_subtype="personal_detail",
            expected_content_modes=frozenset({"scanned_ocr", "mixed"}),
            include_credit_lines=True,
        )

    def build_section_content(
        self,
        parse_result: Any,
        full_text: str,
        *,
        auxiliary_business: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Forward scanned-only records already recovered by the auxiliary pass."""
        del parse_result, full_text
        auxiliary = auxiliary_business or {}
        datasets = {
            name: list(auxiliary.get(name) or [])
            for name in (
                "residence_records",
                "employment_records",
                "statements",
                "annotations",
            )
            if auxiliary.get(name)
        }
        facts: dict[str, Any] = {}
        subject_profile = auxiliary.get("subject_profile")
        if isinstance(subject_profile, dict) and subject_profile:
            facts["subject_profile"] = deepcopy(subject_profile)
        return {
            **({"facts": facts} if facts else {}),
            **({"datasets": datasets} if datasets else {}),
        }

    def data_dictionary(self) -> dict[str, Any]:
        """Describe the scanned-only datasets exposed by this variant."""
        dictionary = super().data_dictionary()
        datasets = dictionary.setdefault("datasets", {})
        datasets.update(
            {
                "residence_records": {
                    "definition": "一行对应一条居住信息记录。",
                    "columns": {
                        "sequence": {"label": "序号", "type": "integer"},
                        "values": {"label": "居住信息", "type": "object"},
                        "page": {"label": "逻辑页码", "type": "integer"},
                        "source_page": {"label": "源页码", "type": "integer"},
                    },
                },
                "employment_records": {
                    "definition": "一行对应一条职业信息记录。",
                    "columns": {
                        "sequence": {"label": "序号", "type": "integer"},
                        "values": {"label": "职业信息", "type": "object"},
                        "page": {"label": "逻辑页码", "type": "integer"},
                        "source_page": {"label": "源页码", "type": "integer"},
                    },
                },
                "statements": {
                    "definition": "一行对应一项信息主体声明。",
                    "columns": {
                        "id": {"label": "声明记录ID", "type": "string"},
                        "text": {"label": "声明内容", "type": "string"},
                        "logical_page": {"label": "逻辑页码", "type": "integer"},
                        "source_page": {"label": "源页码", "type": "integer"},
                    },
                },
                "annotations": {
                    "definition": "一行对应一项异议标注。",
                    "columns": {
                        "id": {"label": "标注记录ID", "type": "string"},
                        "text": {"label": "标注内容", "type": "string"},
                        "logical_page": {"label": "逻辑页码", "type": "integer"},
                        "source_page": {"label": "源页码", "type": "integer"},
                    },
                },
            }
        )
        return dictionary


variant = PersonalDetailScannedVariant()

__all__ = ["PersonalDetailScannedVariant", "variant"]
