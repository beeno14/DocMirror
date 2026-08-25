# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-shot page re-OCR ownership for scanned personal detailed reports.

The final logical-page topology is frozen before this registry is used.  A
stable page key therefore names one immutable subpage image, and the producer
is allowed to run at most once for that key.  Failed or empty attempts are
terminal just like successful attempts; later consumers receive the cached
outcome instead of causing another OCR pass.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class _PageAttempt:
    page_key: str
    logical_page: int
    status: str = "requested"
    reasons: list[str] = field(default_factory=list)
    evidence: dict[str, Any] | None = None
    ocr_invocations: int = 0
    details: dict[str, Any] = field(default_factory=dict)

    def audit_row(self) -> dict[str, Any]:
        return {
            "page_key": self.page_key,
            "logical_page": self.logical_page,
            "status": self.status,
            "reasons": list(self.reasons),
            "ocr_invocations": self.ocr_invocations,
            **deepcopy(self.details),
        }


class OneShotPageReOCRRegistry:
    """Run one producer at most once for each frozen logical subpage."""

    def __init__(self, *, max_pages: int) -> None:
        self.max_pages = max(0, int(max_pages))
        self._attempts: dict[str, _PageAttempt] = {}

    def resolve(
        self,
        *,
        page_key: str,
        logical_page: int,
        reason: str,
        producer: Callable[[], tuple[dict[str, Any] | None, str, dict[str, Any]]],
    ) -> dict[str, Any] | None:
        """Return cached evidence or execute ``producer`` exactly once.

        The producer returns ``(evidence, status, details)``.  Every outcome is
        cached, including render failures and empty OCR results.
        """

        stable_key = str(page_key or "").strip()
        if not stable_key:
            raise ValueError("one-shot page re-OCR requires a stable page key")
        reason_text = str(reason or "unspecified")
        existing = self._attempts.get(stable_key)
        if existing is not None:
            if reason_text not in existing.reasons:
                existing.reasons.append(reason_text)
            evidence = deepcopy(existing.evidence)
            if evidence is not None:
                evidence["page"] = int(logical_page)
                evidence["logical_page"] = int(logical_page)
            return evidence

        if len(self._attempts) >= self.max_pages:
            attempt = _PageAttempt(
                page_key=stable_key,
                logical_page=int(logical_page),
                status="document_budget_exhausted",
                reasons=[reason_text],
            )
            self._attempts[stable_key] = attempt
            return None

        attempt = _PageAttempt(
            page_key=stable_key,
            logical_page=int(logical_page),
            reasons=[reason_text],
        )
        self._attempts[stable_key] = attempt
        try:
            evidence, status, details = producer()
        except Exception as exc:
            attempt.status = "producer_failed"
            attempt.ocr_invocations = 1
            attempt.details = {"error_type": type(exc).__name__}
            return None
        attempt.ocr_invocations = int((details or {}).get("ocr_invocations") or 0)
        if attempt.ocr_invocations > 1:
            raise RuntimeError(f"page re-OCR producer violated one-shot contract for {stable_key}")
        attempt.status = str(status or ("completed" if evidence else "ocr_empty"))
        attempt.details = {
            str(key): deepcopy(value)
            for key, value in dict(details or {}).items()
            if key != "ocr_invocations"
        }
        attempt.evidence = deepcopy(evidence)
        result = deepcopy(evidence)
        if result is not None:
            result["page"] = int(logical_page)
            result["logical_page"] = int(logical_page)
        return result

    def audit(self) -> dict[str, Any]:
        rows = [attempt.audit_row() for attempt in self._attempts.values()]
        return {
            "page_reocr_requests": rows,
            "page_reocr_page_count": len(rows),
            "page_reocr_engine_invocation_count": sum(row["ocr_invocations"] for row in rows),
            "one_shot_per_page_enforced": True,
            "max_ocr_invocations_per_page": max((row["ocr_invocations"] for row in rows), default=0),
        }


__all__ = ["OneShotPageReOCRRegistry"]
