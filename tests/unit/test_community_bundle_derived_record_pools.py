from __future__ import annotations

from docmirror.output.community_bundle import _record_pools


def test_record_pools_do_not_invent_canonical_raw_for_explicitly_derived_field() -> None:
    normalized, canonical = _record_pools(
        {
            "normalized": {
                "source_unitemized_debit_count": 1,
                "ordinary_normalized_only": "normalized value",
                "source_backed": "normalized source value",
            },
            "canonical_raw": {
                "source_backed": "issuer source value",
                "source_unitemized_credit_count": 2,
            },
            "source": {
                "field_sources": {
                    "source_unitemized_debit_count": {
                        "source": "derived.bank_statement.source_unitemized",
                        "derivation": "source_unitemized_reconciliation",
                    },
                    "source_unitemized_credit_count": {
                        "source": "derived.bank_statement.source_unitemized",
                        "derivation": "source_unitemized_reconciliation",
                    },
                }
            },
        }
    )

    assert normalized == {
        "source_unitemized_debit_count": 1,
        "ordinary_normalized_only": "normalized value",
        "source_backed": "normalized source value",
        "source_unitemized_credit_count": 2,
    }
    assert canonical == {
        "ordinary_normalized_only": "normalized value",
        "source_backed": "issuer source value",
        "source_unitemized_credit_count": 2,
    }


def test_record_pools_require_both_derived_source_and_derivation_to_suppress_backfill() -> None:
    normalized, canonical = _record_pools(
        {
            "normalized": {"source_unitemized_debit_count": 1},
            "canonical_raw": {},
            "source": {
                "field_sources": {
                    "source_unitemized_debit_count": {
                        "source": "derived.bank_statement.source_unitemized",
                    }
                }
            },
        }
    )

    assert normalized == {"source_unitemized_debit_count": 1}
    assert canonical == {"source_unitemized_debit_count": 1}


def test_record_pools_respect_normalized_only_for_direct_source_derivation() -> None:
    normalized, canonical = _record_pools(
        {
            "normalized": {
                "query_period": "2024-01-01 ~ 2024-02-29",
                "period_start": "2024-01-01",
                "period_end": "2024-02-29",
            },
            "canonical_raw": {
                "period_start": "20240101",
                "period_end": "20240229",
            },
            "source": {
                "field_sources": {
                    "query_period": {
                        "source": "canonical_evidence_atoms",
                        "derivation": "source_period_envelope",
                        "normalized_only": True,
                    }
                }
            },
        }
    )

    assert normalized["query_period"] == "2024-01-01 ~ 2024-02-29"
    assert canonical == {
        "period_start": "20240101",
        "period_end": "20240229",
    }
