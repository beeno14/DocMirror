# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Materialization planning for the personal-detail Candidate B pipeline.

The types in this module deliberately know nothing about extractor callbacks or
dataset payloads.  Candidate B owns those details.  This layer only answers
which registered stages must execute, which existing stage snapshots can be
reused, and when uncertainty requires the existing eager materialization mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, Protocol


class ExtractionPlanError(ValueError):
    """Raised when a stage graph or materialization request is inconsistent."""


class SectionState(str, Enum):
    """Evidence-backed state assigned by the lightweight section census."""

    OBSERVED = "observed"
    EXPLICITLY_EMPTY = "explicitly_empty"
    ABSENT_PROVEN = "absent_proven"
    UNRESOLVED = "unresolved"


class PlanPurpose(str, Enum):
    """Reason for constructing a materialization plan."""

    INITIAL = "initial"
    REPAIR = "repair"


class MaterializationMode(str, Enum):
    """How a plan will materialize registered extraction stages."""

    EAGER = "eager"
    LAZY = "lazy"
    EAGER_FALLBACK = "eager_fallback"


def _normalized_name(value: object, *, field_name: str) -> str:
    name = str(value or "").strip()
    if not name:
        raise ExtractionPlanError(f"{field_name} cannot contain an empty name")
    return name


def _normalized_names(values: Iterable[object], *, field_name: str) -> frozenset[str]:
    return frozenset(
        _normalized_name(value, field_name=field_name)
        for value in values
    )


@dataclass(frozen=True, slots=True)
class ExtractionStage:
    """Immutable description of one Candidate B extraction stage.

    ``optional`` controls initial materialization only.  A stage selected as a
    dependency always runs, even when its own section is optional.  Extractor
    functions are intentionally not stored here so behavior remains co-located
    with Candidate B.
    """

    name: str
    dependencies: frozenset[str] = frozenset()
    section: str | None = None
    optional: bool = False
    output_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        name = _normalized_name(self.name, field_name="stage")
        dependencies = _normalized_names(
            self.dependencies,
            field_name=f"dependencies for {name!r}",
        )
        if name in dependencies:
            raise ExtractionPlanError(f"stage {name!r} cannot depend on itself")
        section = str(self.section or "").strip() or None
        output_names = tuple(
            _normalized_name(value, field_name=f"outputs for {name!r}")
            for value in self.output_names
        )
        if len(output_names) != len(set(output_names)):
            raise ExtractionPlanError(f"stage {name!r} contains duplicate outputs")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "dependencies", dependencies)
        object.__setattr__(self, "section", section)
        object.__setattr__(self, "output_names", output_names)


@dataclass(frozen=True, slots=True)
class StageRegistry:
    """Validated immutable DAG of Candidate B extraction stages."""

    stages: tuple[ExtractionStage, ...]

    def __post_init__(self) -> None:
        stages = tuple(self.stages)
        names = tuple(stage.name for stage in stages)
        if not stages:
            raise ExtractionPlanError("stage registry cannot be empty")
        if len(names) != len(set(names)):
            duplicates = sorted(
                name for name in set(names) if names.count(name) > 1
            )
            raise ExtractionPlanError(
                "stage registry contains duplicate stages: " + ", ".join(duplicates)
            )
        known_names = set(names)
        missing = sorted(
            {
                dependency
                for stage in stages
                for dependency in stage.dependencies
                if dependency not in known_names
            }
        )
        if missing:
            raise ExtractionPlanError(
                "stage registry contains unknown dependencies: " + ", ".join(missing)
            )
        object.__setattr__(self, "stages", stages)
        # Resolve the complete order during construction so cyclic registries
        # fail before any extraction callback can run.
        self._topological_names()

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(stage.name for stage in self.stages)

    @property
    def sections(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                stage.section
                for stage in self.stages
                if stage.section is not None
            )
        )

    def stage(self, name: str) -> ExtractionStage:
        normalized = _normalized_name(name, field_name="stage")
        for stage in self.stages:
            if stage.name == normalized:
                return stage
        raise ExtractionPlanError(f"unknown extraction stage: {normalized}")

    def validate_names(self, names: Iterable[str], *, field_name: str) -> frozenset[str]:
        normalized = _normalized_names(names, field_name=field_name)
        unknown = sorted(normalized.difference(self.names))
        if unknown:
            raise ExtractionPlanError(
                f"{field_name} contains unknown stages: " + ", ".join(unknown)
            )
        return normalized

    def stages_for_sections(self, sections: Iterable[str]) -> frozenset[str]:
        normalized = _normalized_names(sections, field_name="sections")
        return frozenset(
            stage.name for stage in self.stages if stage.section in normalized
        )

    def unknown_sections(self, sections: Iterable[str]) -> frozenset[str]:
        normalized = _normalized_names(sections, field_name="sections")
        return normalized.difference(self.sections)

    def dependency_closure(self, names: Iterable[str]) -> frozenset[str]:
        """Return the supplied stages and every transitive prerequisite."""

        closure = set(self.validate_names(names, field_name="stage selection"))
        changed = True
        while changed:
            changed = False
            for stage in self.stages:
                if stage.name not in closure:
                    continue
                additions = stage.dependencies.difference(closure)
                if additions:
                    closure.update(additions)
                    changed = True
        return frozenset(closure)

    def dependent_closure(self, names: Iterable[str]) -> frozenset[str]:
        """Return the supplied stages and every transitive dependent."""

        closure = set(self.validate_names(names, field_name="dirty stages"))
        changed = True
        while changed:
            changed = False
            for stage in self.stages:
                if stage.name in closure:
                    continue
                if stage.dependencies.intersection(closure):
                    closure.add(stage.name)
                    changed = True
        return frozenset(closure)

    def ordered(self, names: Iterable[str] | None = None) -> tuple[str, ...]:
        """Return a deterministic topological ordering of selected stages."""

        ordered = self._topological_names()
        if names is None:
            return ordered
        selected = self.validate_names(names, field_name="stage selection")
        return tuple(name for name in ordered if name in selected)

    def _topological_names(self) -> tuple[str, ...]:
        emitted: list[str] = []
        emitted_set: set[str] = set()
        while len(emitted) < len(self.stages):
            added = False
            for stage in self.stages:
                if stage.name in emitted_set:
                    continue
                if stage.dependencies.issubset(emitted_set):
                    emitted.append(stage.name)
                    emitted_set.add(stage.name)
                    added = True
                    # Reconsider the registry from the beginning after every
                    # emission. This gives declaration order a stable and
                    # unsurprising tie-break among newly ready stages.
                    break
            if not added:
                cyclic = sorted(set(self.names).difference(emitted_set))
                raise ExtractionPlanError(
                    "stage registry contains a dependency cycle involving: "
                    + ", ".join(cyclic)
                )
        return tuple(emitted)


@dataclass(frozen=True, slots=True)
class SectionCensus:
    """Immutable, deterministic view of section-presence decisions."""

    states: tuple[tuple[str, SectionState], ...]
    complete: bool = True
    incomplete_reason: str = ""

    def __post_init__(self) -> None:
        normalized: list[tuple[str, SectionState]] = []
        seen: set[str] = set()
        for raw_name, raw_state in self.states:
            name = _normalized_name(raw_name, field_name="census section")
            if name in seen:
                raise ExtractionPlanError(f"duplicate census section: {name}")
            seen.add(name)
            try:
                state = raw_state if isinstance(raw_state, SectionState) else SectionState(str(raw_state))
            except ValueError as exc:
                raise ExtractionPlanError(
                    f"invalid census state for section {name!r}: {raw_state!r}"
                ) from exc
            normalized.append((name, state))
        normalized.sort(key=lambda item: item[0])
        object.__setattr__(self, "states", tuple(normalized))
        object.__setattr__(self, "incomplete_reason", str(self.incomplete_reason or "").strip())

    @classmethod
    def from_mapping(
        cls,
        states: Mapping[str, SectionState | str],
        *,
        complete: bool = True,
        incomplete_reason: str = "",
    ) -> SectionCensus:
        return cls(
            states=tuple(states.items()),
            complete=complete,
            incomplete_reason=incomplete_reason,
        )

    def state_for(self, section: str) -> SectionState | None:
        name = _normalized_name(section, field_name="section")
        for section_name, state in self.states:
            if section_name == name:
                return state
        return None

    @property
    def unresolved_sections(self) -> tuple[str, ...]:
        return tuple(
            name for name, state in self.states if state is SectionState.UNRESOLVED
        )


@dataclass(frozen=True, slots=True)
class ExtractionRequest:
    """Immutable inputs used by an extraction materialization strategy."""

    purpose: PlanPurpose = PlanPurpose.INITIAL
    census: SectionCensus | None = None
    requested_stage_names: frozenset[str] = frozenset()
    available_stage_names: frozenset[str] = frozenset()
    dirty_stage_names: frozenset[str] = frozenset()
    dirty_section_names: frozenset[str] = frozenset()
    dependency_closure_known: bool = True
    force_eager_reason: str = ""

    def __post_init__(self) -> None:
        try:
            purpose = self.purpose if isinstance(self.purpose, PlanPurpose) else PlanPurpose(str(self.purpose))
        except ValueError as exc:
            raise ExtractionPlanError(f"invalid plan purpose: {self.purpose!r}") from exc
        object.__setattr__(self, "purpose", purpose)
        for field_name in (
            "requested_stage_names",
            "available_stage_names",
            "dirty_stage_names",
            "dirty_section_names",
        ):
            object.__setattr__(
                self,
                field_name,
                _normalized_names(getattr(self, field_name), field_name=field_name),
            )
        object.__setattr__(
            self,
            "force_eager_reason",
            str(self.force_eager_reason or "").strip(),
        )
        if purpose is PlanPurpose.INITIAL and (
            self.available_stage_names
            or self.dirty_stage_names
            or self.dirty_section_names
        ):
            raise ExtractionPlanError(
                "initial extraction request cannot contain available or dirty stages"
            )

    @classmethod
    def initial(
        cls,
        census: SectionCensus,
        *,
        requested_stage_names: Iterable[str] = (),
        force_eager_reason: str = "",
    ) -> ExtractionRequest:
        return cls(
            purpose=PlanPurpose.INITIAL,
            census=census,
            requested_stage_names=frozenset(requested_stage_names),
            force_eager_reason=force_eager_reason,
        )

    @classmethod
    def repair(
        cls,
        census: SectionCensus,
        *,
        available_stage_names: Iterable[str],
        dirty_stage_names: Iterable[str] = (),
        dirty_section_names: Iterable[str] = (),
        requested_stage_names: Iterable[str] = (),
        dependency_closure_known: bool = True,
        force_eager_reason: str = "",
    ) -> ExtractionRequest:
        return cls(
            purpose=PlanPurpose.REPAIR,
            census=census,
            requested_stage_names=frozenset(requested_stage_names),
            available_stage_names=frozenset(available_stage_names),
            dirty_stage_names=frozenset(dirty_stage_names),
            dirty_section_names=frozenset(dirty_section_names),
            dependency_closure_known=dependency_closure_known,
            force_eager_reason=force_eager_reason,
        )


@dataclass(frozen=True, slots=True)
class ExtractionPlan:
    """Deterministic immutable materialization plan."""

    strategy: str
    mode: MaterializationMode
    purpose: PlanPurpose
    requested_stage_names: tuple[str, ...]
    ordered_stage_names: tuple[str, ...]
    reused_stage_names: tuple[str, ...]
    skipped_stage_names: tuple[str, ...]
    dirty_stage_names: tuple[str, ...] = ()
    dirty_section_names: tuple[str, ...] = ()
    fallback_reason: str = ""

    def __post_init__(self) -> None:
        strategy = _normalized_name(self.strategy, field_name="strategy")
        try:
            mode = self.mode if isinstance(self.mode, MaterializationMode) else MaterializationMode(str(self.mode))
            purpose = self.purpose if isinstance(self.purpose, PlanPurpose) else PlanPurpose(str(self.purpose))
        except ValueError as exc:
            raise ExtractionPlanError("invalid plan mode or purpose") from exc
        tuple_fields = (
            "requested_stage_names",
            "ordered_stage_names",
            "reused_stage_names",
            "skipped_stage_names",
            "dirty_stage_names",
            "dirty_section_names",
        )
        for field_name in tuple_fields:
            values = tuple(
                _normalized_name(value, field_name=field_name)
                for value in getattr(self, field_name)
            )
            if len(values) != len(set(values)):
                raise ExtractionPlanError(f"{field_name} contains duplicate names")
            object.__setattr__(self, field_name, values)
        execute = set(self.ordered_stage_names)
        reuse = set(self.reused_stage_names)
        skipped = set(self.skipped_stage_names)
        if execute.intersection(reuse) or execute.intersection(skipped) or reuse.intersection(skipped):
            raise ExtractionPlanError(
                "executed, reused, and skipped stage sets must be disjoint"
            )
        object.__setattr__(self, "strategy", strategy)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "purpose", purpose)
        object.__setattr__(self, "fallback_reason", str(self.fallback_reason or "").strip())

    @property
    def fallback_used(self) -> bool:
        return self.mode is MaterializationMode.EAGER_FALLBACK

    def audit(self) -> ExtractionRunAudit:
        return ExtractionRunAudit.from_plan(self)


@dataclass(frozen=True, slots=True)
class StageSnapshot:
    """Immutable provenance descriptor for one materialized stage payload.

    The payload remains owned by Candidate B.  This descriptor records the
    generation and dependency generations needed to decide whether that payload
    can be reused without changing any extractor's input or output contract.
    """

    stage_name: str
    generation: int = 1
    dependency_generations: tuple[tuple[str, int], ...] = ()
    output_names: tuple[str, ...] = ()
    record_counts: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        stage_name = _normalized_name(self.stage_name, field_name="snapshot stage")
        if self.generation < 1:
            raise ExtractionPlanError("snapshot generation must be positive")
        dependencies = tuple(
            sorted(
                (
                    _normalized_name(name, field_name="snapshot dependency"),
                    int(generation),
                )
                for name, generation in self.dependency_generations
            )
        )
        if any(generation < 1 for _name, generation in dependencies):
            raise ExtractionPlanError(
                "snapshot dependency generations must be positive"
            )
        if len(dependencies) != len({name for name, _generation in dependencies}):
            raise ExtractionPlanError("snapshot contains duplicate dependencies")
        outputs = tuple(
            _normalized_name(name, field_name="snapshot output")
            for name in self.output_names
        )
        if len(outputs) != len(set(outputs)):
            raise ExtractionPlanError("snapshot contains duplicate outputs")
        counts = tuple(
            sorted(
                (
                    _normalized_name(name, field_name="snapshot count"),
                    int(count),
                )
                for name, count in self.record_counts
            )
        )
        if any(count < 0 for _name, count in counts):
            raise ExtractionPlanError("snapshot record counts cannot be negative")
        if len(counts) != len({name for name, _count in counts}):
            raise ExtractionPlanError("snapshot contains duplicate record counts")
        object.__setattr__(self, "stage_name", stage_name)
        object.__setattr__(self, "dependency_generations", dependencies)
        object.__setattr__(self, "output_names", outputs)
        object.__setattr__(self, "record_counts", counts)

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage_name,
            "generation": self.generation,
            "dependency_generations": dict(self.dependency_generations),
            "outputs": list(self.output_names),
            "record_counts": dict(self.record_counts),
        }


@dataclass(frozen=True, slots=True)
class ExtractionRunAudit:
    """Stable serializable audit of a strategy decision."""

    strategy: str
    mode: MaterializationMode
    purpose: PlanPurpose
    requested_stage_names: tuple[str, ...]
    ordered_stage_names: tuple[str, ...]
    reused_stage_names: tuple[str, ...]
    skipped_stage_names: tuple[str, ...]
    dirty_stage_names: tuple[str, ...]
    dirty_section_names: tuple[str, ...]
    fallback_reason: str

    @classmethod
    def from_plan(cls, plan: ExtractionPlan) -> ExtractionRunAudit:
        return cls(
            strategy=plan.strategy,
            mode=plan.mode,
            purpose=plan.purpose,
            requested_stage_names=plan.requested_stage_names,
            ordered_stage_names=plan.ordered_stage_names,
            reused_stage_names=plan.reused_stage_names,
            skipped_stage_names=plan.skipped_stage_names,
            dirty_stage_names=plan.dirty_stage_names,
            dirty_section_names=plan.dirty_section_names,
            fallback_reason=plan.fallback_reason,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "mode": self.mode.value,
            "purpose": self.purpose.value,
            "requested_stages": list(self.requested_stage_names),
            "ordered_stages": list(self.ordered_stage_names),
            "reused_stages": list(self.reused_stage_names),
            "skipped_stages": list(self.skipped_stage_names),
            "dirty_stages": list(self.dirty_stage_names),
            "dirty_sections": list(self.dirty_section_names),
            "fallback_reason": self.fallback_reason or None,
        }


class ExtractionStrategy(Protocol):
    """Interface implemented by Candidate B materialization strategies."""

    name: str

    def plan(
        self,
        registry: StageRegistry,
        request: ExtractionRequest | None = None,
    ) -> ExtractionPlan:
        """Build a deterministic stage plan without invoking extractors."""


@dataclass(frozen=True, slots=True)
class EagerExtractionStrategy:
    """Existing Candidate B behavior expressed through the stage graph."""

    name: str = "eager"

    def plan(
        self,
        registry: StageRegistry,
        request: ExtractionRequest | None = None,
    ) -> ExtractionPlan:
        request = request or ExtractionRequest()
        requested = registry.validate_names(
            request.requested_stage_names,
            field_name="requested_stage_names",
        )
        dirty = registry.validate_names(
            request.dirty_stage_names,
            field_name="dirty_stage_names",
        )
        dirty_sections = request.dirty_section_names
        unknown_sections = registry.unknown_sections(dirty_sections) if dirty_sections else frozenset()
        if unknown_sections:
            raise ExtractionPlanError(
                "dirty_section_names contains unknown sections: "
                + ", ".join(sorted(unknown_sections))
            )
        dirty = dirty.union(registry.stages_for_sections(dirty_sections))
        return ExtractionPlan(
            strategy=self.name,
            mode=MaterializationMode.EAGER,
            purpose=request.purpose,
            requested_stage_names=registry.ordered(requested),
            ordered_stage_names=registry.ordered(),
            reused_stage_names=(),
            skipped_stage_names=(),
            dirty_stage_names=registry.ordered(dirty),
            dirty_section_names=tuple(sorted(dirty_sections)),
        )


@dataclass(frozen=True, slots=True)
class LazyExtractionStrategy:
    """Conservative section-lazy and repair-incremental planner."""

    name: str = "lazy"

    def plan(
        self,
        registry: StageRegistry,
        request: ExtractionRequest | None = None,
    ) -> ExtractionPlan:
        request = request or ExtractionRequest()
        requested = registry.validate_names(
            request.requested_stage_names,
            field_name="requested_stage_names",
        )
        available = registry.validate_names(
            request.available_stage_names,
            field_name="available_stage_names",
        )
        dirty = registry.validate_names(
            request.dirty_stage_names,
            field_name="dirty_stage_names",
        )
        unknown_dirty_sections = (
            registry.unknown_sections(request.dirty_section_names)
            if request.dirty_section_names
            else frozenset()
        )
        dirty = dirty.union(registry.stages_for_sections(request.dirty_section_names))

        fallback_reason = self._fallback_reason(
            registry,
            request,
            unknown_dirty_sections=unknown_dirty_sections,
        )
        if fallback_reason:
            return self._fallback_plan(
                registry,
                request,
                requested=requested,
                dirty=dirty,
                fallback_reason=fallback_reason,
            )

        desired = self._desired_stages(
            registry,
            request.census,
            requested=requested.union(dirty),
            available=(
                available
                if request.purpose is PlanPurpose.REPAIR
                else frozenset()
            ),
        )
        if request.purpose is PlanPurpose.INITIAL:
            execute = desired
            reused = frozenset()
        else:
            affected = registry.dependent_closure(dirty).intersection(desired)
            # A partial snapshot set is valid: missing desired prerequisites are
            # simply materialized now.  Clean available stages are reused.
            execute = affected.union(desired.difference(available))
            reused = desired.intersection(available).difference(affected)

        skipped = set(registry.names).difference(execute).difference(reused)
        return ExtractionPlan(
            strategy=self.name,
            mode=MaterializationMode.LAZY,
            purpose=request.purpose,
            requested_stage_names=registry.ordered(requested),
            ordered_stage_names=registry.ordered(execute),
            reused_stage_names=registry.ordered(reused),
            skipped_stage_names=registry.ordered(skipped),
            dirty_stage_names=registry.ordered(dirty),
            dirty_section_names=tuple(sorted(request.dirty_section_names)),
        )

    @staticmethod
    def _desired_stages(
        registry: StageRegistry,
        census: SectionCensus | None,
        *,
        requested: frozenset[str],
        available: frozenset[str],
    ) -> frozenset[str]:
        roots = {
            stage.name
            for stage in registry.stages
            if not stage.optional
        }
        roots.update(requested)
        if census is not None:
            for stage in registry.stages:
                if not stage.optional or stage.section is None:
                    continue
                if census.state_for(stage.section) is SectionState.OBSERVED:
                    roots.add(stage.name)
            # A repair must keep previously materialized optional downstream
            # stages in its dependency universe.  Otherwise a sectionless
            # optional stage can disappear before ``dependent_closure`` gets a
            # chance to invalidate it.  Section-bound snapshots are retained
            # only while the repaired census still proves their section is
            # present; stale rows from a now-absent section are not reused.
            for stage_name in available:
                stage = registry.stage(stage_name)
                section_state = (
                    census.state_for(stage.section)
                    if stage.section is not None
                    else None
                )
                if (
                    not stage.optional
                    or stage.section is None
                    or section_state
                    in {SectionState.OBSERVED, SectionState.UNRESOLVED}
                ):
                    roots.add(stage_name)
        return registry.dependency_closure(roots)

    @staticmethod
    def _fallback_reason(
        registry: StageRegistry,
        request: ExtractionRequest,
        *,
        unknown_dirty_sections: frozenset[str],
    ) -> str:
        if request.force_eager_reason:
            return request.force_eager_reason
        if not request.dependency_closure_known:
            return "dependency_closure_unknown"
        if unknown_dirty_sections:
            return "unknown_dirty_sections:" + ",".join(sorted(unknown_dirty_sections))
        census = request.census
        if census is None:
            return "section_census_missing"
        partial_repair = (
            request.purpose is PlanPurpose.REPAIR
            and request.dependency_closure_known
        )
        if not census.complete and not partial_repair:
            return census.incomplete_reason or "section_census_incomplete"
        if census.unresolved_sections and not partial_repair:
            return "section_census_unresolved:" + ",".join(census.unresolved_sections)
        missing_sections = tuple(
            section
            for section in registry.sections
            if any(
                stage.optional and stage.section == section
                for stage in registry.stages
            )
            and census.state_for(section) is None
        )
        if missing_sections:
            return "section_census_missing_sections:" + ",".join(missing_sections)
        return ""

    def _fallback_plan(
        self,
        registry: StageRegistry,
        request: ExtractionRequest,
        *,
        requested: frozenset[str],
        dirty: frozenset[str],
        fallback_reason: str,
    ) -> ExtractionPlan:
        return ExtractionPlan(
            strategy=self.name,
            mode=MaterializationMode.EAGER_FALLBACK,
            purpose=request.purpose,
            requested_stage_names=registry.ordered(requested),
            ordered_stage_names=registry.ordered(),
            reused_stage_names=(),
            skipped_stage_names=(),
            dirty_stage_names=registry.ordered(dirty),
            dirty_section_names=tuple(sorted(request.dirty_section_names)),
            fallback_reason=fallback_reason,
        )


__all__ = [
    "EagerExtractionStrategy",
    "ExtractionPlan",
    "ExtractionPlanError",
    "ExtractionRequest",
    "ExtractionRunAudit",
    "ExtractionStage",
    "ExtractionStrategy",
    "LazyExtractionStrategy",
    "MaterializationMode",
    "PlanPurpose",
    "SectionCensus",
    "SectionState",
    "StageRegistry",
    "StageSnapshot",
]
