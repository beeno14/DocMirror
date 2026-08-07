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
from typing import Any

from docmirror.output.community_bundle import CommunityBundle
from docmirror.plugins._base.projector import CommunityProjector, ProjectionData

_PERSONAL_DETAIL_SOURCE_COMPLETE = frozenset(
    {"observed_nonempty", "explicitly_empty", "not_applicable"}
)


def _apply_personal_detail_dataset_status(payload: dict[str, Any]) -> None:
    """Make public dataset envelopes agree with the v2 source-status ledger.

    Community's ordinary completeness calculation proves only that projected
    rows were conserved during serialization.  It must not turn a source-
    partial scanned dataset into ``complete`` merely because all rows that
    reached the projector were written successfully.
    """

    datasets = [item for item in payload.get("datasets") or () if isinstance(item, dict)]
    status_dataset = next(
        (item for item in datasets if str(item.get("name") or "") == "dataset_status"),
        None,
    )
    if status_dataset is None:
        return
    status_by_name: dict[str, dict[str, Any]] = {}
    for wrapper in status_dataset.get("rows") or ():
        if not isinstance(wrapper, dict):
            continue
        values = wrapper.get("normalized") if isinstance(wrapper.get("normalized"), dict) else wrapper
        name = str(values.get("dataset_name") or "")
        if name:
            status_by_name[name] = values

    for dataset in datasets:
        name = str(dataset.get("name") or "")
        control = status_by_name.get(name)
        if control is None:
            continue
        presence = str(control.get("presence_status") or "unknown")
        source_complete = presence in _PERSONAL_DETAIL_SOURCE_COMPLETE
        dataset["status"] = "complete" if source_complete else "partial"
        emitted = int(dataset.get("row_count") or len(dataset.get("rows") or ()))
        expected_raw = control.get("expected_row_count")
        expected = (
            int(expected_raw)
            if isinstance(expected_raw, (int, float)) and not isinstance(expected_raw, bool)
            else None
        )
        completeness = dict(dataset.get("completeness") or {})
        completeness.update(
            {
                "expected_row_count": expected,
                "emitted_row_count": emitted,
                "omitted_row_count": max(expected - emitted, 0) if expected is not None else None,
                "verified": bool(source_complete),
                "basis": f"personal_detail_dataset_status:{presence}",
            }
        )
        dataset["completeness"] = completeness


class _CreditReportCommunityBundle(CommunityBundle):
    """Publish compact Community JSON without weakening rich semantic bindings."""

    @staticmethod
    def _is_enterprise_semantic(payload: dict[str, Any]) -> bool:
        domain = payload.get("domain") if isinstance(payload.get("domain"), dict) else {}
        facts = domain.get("facts") if isinstance(domain.get("facts"), dict) else {}
        return facts.get("report_subtype") == "enterprise"

    def semantic_payload(self) -> dict[str, Any]:
        payload = super().semantic_payload()
        domain = payload.get("domain") if isinstance(payload.get("domain"), dict) else {}
        facts = domain.get("facts") if isinstance(domain.get("facts"), dict) else {}
        if facts.get("report_subtype") != "enterprise":
            return payload
        extraction = facts.pop("extraction_report", None)
        if isinstance(extraction, dict):
            payload["extraction"] = extraction
        for key in tuple(facts):
            if key.startswith("enterprise_expected_"):
                facts.pop(key, None)
        extensions = domain.get("extensions") if isinstance(domain.get("extensions"), dict) else {}
        extensions.pop("enterprise_dataset_completeness", None)
        overrides = (
            extensions.get("community_projection_overrides")
            if isinstance(extensions.get("community_projection_overrides"), dict)
            else {}
        )
        for key in ("internal_fields", "internal_facts"):
            values = overrides.get(key)
            if not isinstance(values, list):
                continue
            overrides[key] = [
                value for value in values if not str(value).startswith("enterprise_expected_")
            ]
        return payload

    def json_payload(self, semantic: dict[str, Any] | None = None) -> dict[str, Any]:
        from docmirror.plugins.credit_report.projection import _compact_public_datasets

        semantic_payload = semantic or self.semantic_payload()
        payload = super().json_payload(semantic_payload)
        domain = (
            semantic_payload.get("domain")
            if isinstance(semantic_payload.get("domain"), dict)
            else {}
        )
        facts = domain.get("facts") if isinstance(domain.get("facts"), dict) else {}
        enterprise = facts.get("report_subtype") == "enterprise"
        scanned_personal_detail = bool(
            facts.get("report_subtype") == "personal_detailed"
            and facts.get("content_mode") in {"scanned_ocr", "mixed"}
        )
        extensions = domain.get("extensions") if isinstance(domain.get("extensions"), dict) else {}
        enterprise_completeness = (
            extensions.get("enterprise_dataset_completeness", {}) if enterprise else {}
        )
        for dataset in payload.get("datasets") or []:
            if not isinstance(dataset, dict):
                continue
            dataset_id = str(dataset.get("id") or dataset.get("name") or "dataset")
            records = [record for record in (dataset.get("rows") or []) if isinstance(record, dict)]
            dataset["rows"] = _compact_public_datasets({dataset_id: records})[dataset_id]
            if not enterprise:
                continue
            details = enterprise_completeness.get(str(dataset.get("name") or ""))
            if not isinstance(details, dict):
                continue
            dataset["completeness"] = {
                key: details[key]
                for key in (
                    "expected_row_count",
                    "emitted_row_count",
                    "omitted_row_count",
                    "verified",
                    "basis",
                )
                if key in details
            }
        if enterprise:
            from docmirror.plugins.credit_report.enterprise_native.projector import (
                project_enterprise_community_json,
            )

            payload = project_enterprise_community_json(payload)
        elif scanned_personal_detail:
            _apply_personal_detail_dataset_status(payload)
        return payload

    def _enterprise_artifact_semantic(
        self,
        semantic: dict[str, Any],
    ) -> dict[str, Any]:
        from docmirror.plugins.credit_report.enterprise_native.projector import (
            project_enterprise_artifact_semantic,
        )

        return project_enterprise_artifact_semantic(
            semantic,
            self.json_payload(semantic),
        )

    def render_dataset_csvs(self, semantic: dict[str, Any] | None = None) -> dict[str, str]:
        semantic_payload = semantic or self.semantic_payload()
        if not self._is_enterprise_semantic(semantic_payload):
            return super().render_dataset_csvs(semantic_payload)
        return super().render_dataset_csvs(
            self._enterprise_artifact_semantic(semantic_payload)
        )

    def render_audit_csv(self, semantic: dict[str, Any] | None = None) -> str:
        semantic_payload = semantic or self.semantic_payload()
        if not self._is_enterprise_semantic(semantic_payload):
            return super().render_audit_csv(semantic_payload)
        return super().render_audit_csv(
            self._enterprise_artifact_semantic(semantic_payload)
        )


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
        from docmirror.plugins.credit_report.report_profile import (
            detect_credit_report_subtype,
        )

        report_subtype = detect_credit_report_subtype(parse_result, text)
        if report_subtype == "enterprise":
            from docmirror.plugins.credit_report.enterprise_native.projector import (
                derive_enterprise_projection,
            )

            return derive_enterprise_projection(self, parse_result, text)
        if report_subtype == "personal_brief":
            from docmirror.plugins.credit_report.personal_brief_native.projector import (
                derive_personal_brief_projection,
            )

            return derive_personal_brief_projection(self, parse_result, text)
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
        projected = project_community_bundle(
            sealed,
            file_path=file_path,
            file_id=file_id,
            document_id=document_id,
            projection_data=derived.model_dump(mode="python"),
            projection_policy=policy,
        )
        bundle = _CreditReportCommunityBundle(
            schema=projected.schema,
            document=projected.document,
            sections=projected.sections,
            datasets=projected.datasets,
            files=projected.files,
            warnings=projected.warnings,
            result=projected.result,
            source_fingerprint=projected.source_fingerprint,
            parse_result_schema=projected.parse_result_schema,
            classification=projected.classification,
            domain=projected.domain,
            diagnostics=projected.diagnostics,
            content_markdown_override=projected.content_markdown_override,
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
