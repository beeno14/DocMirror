# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validated post-seal reading transformations for enhanced Markdown."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FragmentJoin(BaseModel):
    """Join existing source fragments onto one anchor without inventing text."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["join_fragments"] = "join_fragments"
    scope: str = Field(min_length=1)
    source_node_ids: tuple[str, ...] = Field(min_length=2)
    anchor_node_id: str = Field(min_length=1)
    fragment_node_ids: tuple[str, ...] = Field(min_length=1)
    reason: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _validate_node_membership(self) -> FragmentJoin:
        if len(set(self.source_node_ids)) != len(self.source_node_ids):
            raise ValueError("source_node_ids must be unique")
        if len(set(self.fragment_node_ids)) != len(self.fragment_node_ids):
            raise ValueError("fragment_node_ids must be unique")
        if self.anchor_node_id not in self.source_node_ids:
            raise ValueError("anchor_node_id must be included in source_node_ids")
        if self.anchor_node_id in self.fragment_node_ids:
            raise ValueError("anchor_node_id cannot also be a fragment")
        if not set(self.fragment_node_ids).issubset(self.source_node_ids):
            raise ValueError("all fragment_node_ids must be included in source_node_ids")
        return self


class SourceTextSlice(BaseModel):
    """One exact, bounded slice of an existing source node's text."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: str = Field(min_length=1)
    start: int = Field(ge=0)
    end: int = Field(gt=0)

    @model_validator(mode="after")
    def _validate_range(self) -> SourceTextSlice:
        if self.end <= self.start:
            raise ValueError("source text slice end must be greater than start")
        return self


class SliceReflow(BaseModel):
    """Reorder exact source-text slices without inventing or duplicating text."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["reflow_slices"] = "reflow_slices"
    scope: str = Field(min_length=1)
    source_node_ids: tuple[str, ...] = Field(min_length=2)
    anchor_node_id: str = Field(min_length=1)
    output_slices: tuple[SourceTextSlice, ...] = Field(min_length=2)
    reason: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _validate_node_membership(self) -> SliceReflow:
        if len(set(self.source_node_ids)) != len(self.source_node_ids):
            raise ValueError("source_node_ids must be unique")
        if self.anchor_node_id not in self.source_node_ids:
            raise ValueError("anchor_node_id must be included in source_node_ids")
        if self.anchor_node_id != self.source_node_ids[0]:
            raise ValueError("anchor_node_id must be the first source node")
        unknown = {item.node_id for item in self.output_slices}.difference(self.source_node_ids)
        if unknown:
            raise ValueError(f"all output slice node IDs must be included in source_node_ids: {sorted(unknown)}")
        return self


class ReadingProjection(BaseModel):
    """One plugin's declarative enhanced-reading projection."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    plugin_id: str = Field(min_length=1)
    profile: Literal["enhanced"] = "enhanced"
    transforms: tuple[FragmentJoin | SliceReflow, ...] = ()

    @model_validator(mode="after")
    def _validate_non_overlapping_transforms(self) -> ReadingProjection:
        claimed: set[str] = set()
        for transform in self.transforms:
            if isinstance(transform, SliceReflow):
                active = set(transform.source_node_ids)
            else:
                active = {transform.anchor_node_id, *transform.fragment_node_ids}
            overlap = claimed.intersection(active)
            if overlap:
                raise ValueError(f"reading transforms overlap on node IDs: {sorted(overlap)}")
            claimed.update(active)
        return self


__all__ = [
    "FragmentJoin",
    "ReadingProjection",
    "SliceReflow",
    "SourceTextSlice",
]
