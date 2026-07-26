# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public post-seal semantic document produced by Community plugins.

The semantic result is intentionally separate from ``ParseResult``.  It is a
document-type-specific, public interpretation of one immutable parser result
and is the sole input to Community JSON, dataset CSV, and enhanced Markdown
renderers.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CommunitySemanticResult(BaseModel):
    """Flexible public semantic contract shared by Community renderers."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    schema_: dict[str, Any] = Field(alias="schema")
    source: dict[str, Any]
    classification: dict[str, Any]
    document: dict[str, Any]
    structure: dict[str, Any]
    datasets: list[dict[str, Any]] = Field(default_factory=list)
    bindings: list[dict[str, Any]] = Field(default_factory=list)
    domain: dict[str, Any] = Field(default_factory=dict)
    reading: dict[str, Any] = Field(default_factory=dict)
    files: dict[str, str] = Field(default_factory=dict)
    warnings: list[dict[str, Any]] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)


__all__ = ["CommunitySemanticResult"]
