# Personal detailed credit report schema 2.0 rollout

Schema `2.0.0` is a PBOC-native, breaking successor to the frozen `1.2.0`
domain contract. It retains the Community JSON `3.0.0` envelope and keeps all
canonical business records in `datasets[*].rows[*].normalized`.

## Shadow and canary phase

`1.2.0` remains the default. Enable the v2 projection for a process with:

```text
DOCMIRROR_PERSONAL_DETAIL_SCHEMA_VERSION=2.0.0
```

The selector is evaluated only for the `personal_detail_scanned` variant.
Enterprise and personal-brief credit-report variants do not enter the v2
projection path.

During this phase, run the same detailed-report corpus once with the default
and once with v2 enabled. Compare source business-field coverage, record
relationships, dataset counts, and consumer behavior. Do not compare auditing
or presentation-only fields as business records.

## Cutover

After consumers accept v2, change deployment configuration to select `2.0.0`.
Keep the `1.2.0` schema, registry entry, and default-compatible projection
available for rollback through the agreed support window.

Remove the opt-in gate or retire `1.2.0` only in a separately reviewed release.
That release must update consumer fixtures and the migration documentation; it
must not change the Community major version unless the envelope itself changes.

