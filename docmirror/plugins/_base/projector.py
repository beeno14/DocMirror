# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Post-seal projector contracts shared by bundled plugins."""

from __future__ import annotations

from importlib.resources import files
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ProjectionData(BaseModel):
    """Domain-derived data used only to assemble one plugin JSON projection."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    projector_id: str = Field(min_length=1)
    document_type: str | None = None
    entity_fields: dict[str, Any] = Field(default_factory=dict)
    domain_facts: dict[str, Any] = Field(default_factory=dict)
    semantic: dict[str, Any] = Field(
        default_factory=dict,
        description="Document-type-specific public semantic extensions for Community renderers.",
    )
    content_markdown_override: str = Field(
        default="",
        description="Optional plugin-rendered source-complete Markdown for documents with broken native text layers.",
    )
    datasets: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    sections: tuple[dict[str, Any], ...] = ()
    warnings: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    reason: str = "post-seal domain projection"


def load_projection_policy(package: str) -> dict[str, Any]:
    """Load one bundled plugin's Seal-after projection policy."""
    manifest_path = files(package).joinpath("plugin.yaml")
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    return dict(payload.get("projection") or {})


class CommunityProjector:
    """Community-edition projector registered in the shared PluginRegistry."""

    edition = "community"
    requires_license = False

    def derive(self, parse_result: Any, text: str = "") -> ProjectionData:
        raise NotImplementedError

    def reading_projection(self, parse_result: Any) -> Any | None:
        """Return an optional declarative projection for enhanced Markdown."""
        return None

    def supports(self, sealed: Any) -> bool:
        from docmirror.models.sealed import SealedParseResult

        if not isinstance(sealed, SealedParseResult):
            return False
        document_type = str(sealed.to_read_view().entities.document_type or "")
        if self.domain_name == "generic":
            return document_type in {"", "generic", "unknown"}
        return document_type == self.domain_name

    def project(self, sealed: Any) -> dict[str, Any] | None:
        bundle = self.project_bundle(sealed)
        return bundle.json_payload() if bundle is not None else None

    def project_bundle(
        self,
        sealed: Any,
        *,
        file_path: str = "",
        file_id: str = "001",
        document_id: str = "",
    ) -> Any | None:
        """Build the public semantic Community bundle from one sealed result."""
        from docmirror.models.sealed import SealedParseResult
        from docmirror.output.community_bundle import project_community_bundle

        if not isinstance(sealed, SealedParseResult):
            raise TypeError(f"{type(self).__name__}.project expects SealedParseResult")
        default_support = getattr(self.supports, "__func__", None) is CommunityProjector.supports
        if not default_support and not self.supports(sealed) and self.domain_name != "generic":
            return None
        before = sealed.integrity_fingerprint
        read_view = sealed.to_read_view()
        if default_support:
            document_type = str(read_view.entities.document_type or "")
            supported = (
                document_type in {"", "generic", "unknown"}
                if self.domain_name == "generic"
                else document_type == self.domain_name
            )
            # Extension/instance support hooks retain their original decision
            # above. The default path inspects this view instead of creating
            # another complete view just to read one scalar routing field.
            if not supported and self.domain_name != "generic":
                return None
        derived = self.derive(
            read_view,
            str(read_view.full_text or read_view.raw_text or ""),
        )
        bundle = project_community_bundle(
            sealed,
            file_path=file_path,
            file_id=file_id,
            document_id=document_id,
            projection_data=derived.model_dump(mode="python"),
            projection_policy=load_projection_policy(type(self).__module__.rsplit(".", 1)[0]),
        )
        bundle.render_markdown()
        if sealed.integrity_fingerprint != before or not sealed.verify_integrity():
            raise RuntimeError("Post-seal projector changed the sealed snapshot")
        return bundle


__all__ = ["CommunityProjector", "ProjectionData", "load_projection_policy"]
