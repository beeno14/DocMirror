# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request-local reuse of equivalent bank extraction work.

The cache never lives in sealed evidence or in a process-global document map.
One extraction owns it, including its bounded forced-eager retry. Callers must
key all non-document inputs and explicitly replay any derivation side effects.
Cached mutable values are private snapshots; every consumer receives its own
copy. This changes computation reuse, not candidate eligibility or selection.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Hashable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, TypeVar

_T = TypeVar("_T")


@dataclass
class BankWorkCache:
    parse_result: Any
    enabled: bool = True
    entries: dict[Hashable, tuple[Any, Any]] = field(default_factory=dict)
    hits: Counter[str] = field(default_factory=Counter)
    misses: Counter[str] = field(default_factory=Counter)


_current: ContextVar[BankWorkCache | None] = ContextVar("bank_work_cache", default=None)


@contextmanager
def bank_work_session(parse_result: Any, *, enabled: bool = True) -> Iterator[BankWorkCache]:
    """Reuse nested sessions for the same read view; discard work at the boundary."""
    existing = _current.get()
    if existing is not None and existing.parse_result is parse_result:
        yield existing
        return
    cache = BankWorkCache(parse_result, enabled=enabled and parse_result is not None)
    token = _current.set(cache)
    try:
        yield cache
    finally:
        _current.reset(token)
        cache.entries.clear()


def active_bank_cache(parse_result: Any) -> bool:
    cache = _current.get()
    return cache is not None and cache.enabled and cache.parse_result is parse_result


def reuse_bank_work(
    parse_result: Any,
    operation: Hashable,
    arguments: Hashable,
    compute: Callable[[], _T],
    *,
    capture: Callable[[], Any] | None = None,
    restore: Callable[[Any], None] | None = None,
    cache_empty: bool = True,
) -> _T:
    """Compute an identical job once, preserving its result and explicit effects.

    Exceptions are never cached. Source-PDF readers also disable empty-result
    caching because their existing error handling can return an empty result
    after a transient I/O/backend failure.
    """
    cache = _current.get()
    if cache is None or not cache.enabled or cache.parse_result is not parse_result:
        return compute()
    key = (operation, arguments)
    try:
        hash(key)
    except TypeError:
        # Compatibility with extension/stub inputs outside the string-matrix
        # contract: computation remains available even when reuse is not safe.
        return compute()
    label = getattr(operation, "__name__", str(operation))
    if key in cache.entries:
        cache.hits[label] += 1
        value, effects = cache.entries[key]
        if restore is not None:
            restore(deepcopy(effects))
        return deepcopy(value)
    cache.misses[label] += 1
    value = compute()
    if cache_empty or value:
        cache.entries[key] = (deepcopy(value), deepcopy(capture()) if capture is not None else None)
    return value


def memoize_bank_document_work(function: Callable[..., _T]) -> Callable[..., _T]:
    """Reuse a pure derivation of the session's unchanged document evidence."""

    @wraps(function)
    def wrapped(parse_result: Any, *args: Any, **kwargs: Any) -> _T:
        return reuse_bank_work(
            parse_result,
            function,
            (args, tuple(sorted(kwargs.items()))),
            lambda: function(parse_result, *args, **kwargs),
        )

    return wrapped
