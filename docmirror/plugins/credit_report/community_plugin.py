# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# Author: Adam Lin <adamlin@valuemapglobal.com>
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
Credit report community domain plugin.

Premium community plugin for personal brief, personal detail, and enterprise
credit reports. Extracts identity fields, report subtype/content mode, optional
lightweight section hints, and table records via shared KV extract helpers.

Pipeline role: post-seal domain derivation and Community JSON projection.

Key exports: ``CreditReportPlugin``, ``plugin``.

Dependencies: ``ProjectionData`` and the credit-report projection orchestrator.
"""

from __future__ import annotations

from collections.abc import Sequence

from docmirror.plugins._base.projector import CommunityProjector, ProjectionData


class CreditReportPlugin(CommunityProjector):
    """Community edition plugin for credit report document processing."""

    @property
    def domain_name(self) -> str:
        return "credit_report"

    @property
    def display_name(self) -> str:
        return "Credit Report (Community)"

    @property
    def projector_id(self) -> str:
        return self.domain_name

    @property
    def identity_fields(self) -> Sequence[tuple[str, Sequence[str]]]:
        return (
            ("subject_name", ("被查询者姓名", "企业名称", "姓名", "Name", "报告主体")),
            ("id_number", ("被查询者证件号码", "身份证号", "证件号码", "ID Number")),
            ("id_type", ("被查询者证件类型", "证件类型", "ID Type")),
            ("marital_status", ("婚姻状况",)),
            ("unified_social_credit_code", ("统一社会信用代码",)),
            ("zhongzheng_code", ("中征码", "贷款卡编码", "贷款卡号")),
            ("query_institution", ("查询机构",)),
            ("report_time", ("报告时间", "查询时间", "Report Time")),
            ("report_number", ("报告编号", "Report No", "NO.")),
        )

    def derive(self, parse_result, text: str = "") -> ProjectionData:
        from docmirror.plugins.credit_report.projection import derive_credit_report_projection

        return derive_credit_report_projection(self, parse_result, text)

    def project_bundle(
        self,
        sealed,
        *,
        file_path: str = "",
        file_id: str = "001",
        document_id: str = "",
    ):
        """Apply document-variant presentation overrides inside this plugin."""
        from docmirror.models.sealed import SealedParseResult
        from docmirror.output.community_bundle import project_community_bundle
        from docmirror.plugins._base.projector import load_projection_policy

        if not isinstance(sealed, SealedParseResult):
            raise TypeError(f"{type(self).__name__}.project expects SealedParseResult")
        if not self.supports(sealed):
            return None
        before = sealed.integrity_fingerprint
        read_view = sealed.to_read_view()
        derived = self.derive(
            read_view,
            str(read_view.full_text or read_view.raw_text or ""),
        )
        policy = load_projection_policy(type(self).__module__.rsplit(".", 1)[0])
        overrides = derived.semantic.get("community_projection_overrides")
        if isinstance(overrides, dict):
            for key, values in overrides.items():
                if isinstance(values, dict):
                    policy[key] = {**dict(policy.get(key) or {}), **values}
                elif key in {"internal_fields", "internal_facts"} and isinstance(
                    values, (list, tuple)
                ):
                    policy[key] = list(
                        dict.fromkeys([*(policy.get(key) or ()), *map(str, values)])
                    )
        bundle = project_community_bundle(
            sealed,
            file_path=file_path,
            file_id=file_id,
            document_id=document_id,
            projection_data=derived.model_dump(mode="python"),
            projection_policy=policy,
        )
        bundle.render_markdown()
        if sealed.integrity_fingerprint != before or not sealed.verify_integrity():
            raise RuntimeError("Post-seal projector changed the sealed snapshot")
        return bundle

    def reading_projection(self, parse_result):
        from docmirror.plugins.credit_report.report_profile import (
            detect_credit_report_content_mode,
            detect_credit_report_subtype,
        )
        from docmirror.plugins.credit_report.variant_router import (
            resolve_credit_report_variant,
        )

        text = str(parse_result.full_text or parse_result.raw_text or "")
        report_subtype = detect_credit_report_subtype(parse_result, text)
        content_mode = detect_credit_report_content_mode(parse_result)
        variant = resolve_credit_report_variant(report_subtype, content_mode)
        return variant.build_reading_projection(
            parse_result,
            content_mode=content_mode,
        )


plugin = CreditReportPlugin()
