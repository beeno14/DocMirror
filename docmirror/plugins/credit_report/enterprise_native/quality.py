# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structured quality findings for the canonical enterprise pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EnterpriseQualityFlag:
    """One actionable finding without changing or inventing source facts."""

    code: str
    severity: str
    category: str
    message: str
    status: str
    source_pages: tuple[int, ...] = ()
    scope: tuple[str, ...] = ()
    dataset: str = ""
    field_name: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": self.code,
            "severity": self.severity,
            "category": self.category,
            "status": self.status,
            "message": self.message,
        }
        if self.source_pages:
            payload["source_pages"] = list(self.source_pages)
        if self.scope:
            payload["scope"] = list(self.scope)
        if self.dataset:
            payload["dataset"] = self.dataset
        if self.field_name:
            payload["field"] = self.field_name
        if self.details:
            payload["details"] = dict(self.details)
        return payload


def quality_warning(flag: EnterpriseQualityFlag | dict[str, Any]) -> str:
    """Render a stable warning string for the Community warning contract."""
    payload = flag.to_payload() if isinstance(flag, EnterpriseQualityFlag) else flag
    return f"{payload.get('code', 'ENTERPRISE_QUALITY')}: {payload.get('message', '')}".strip()


__all__ = ["EnterpriseQualityFlag", "quality_warning"]
