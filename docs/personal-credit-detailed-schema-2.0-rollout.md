# Personal detailed credit report schema 2.0

Schema `2.0.0` is the sole PBOC-native contract for scanned personal detailed
credit reports. It retains the Community JSON `3.0.0` envelope and keeps all
canonical business records in `datasets[*].rows[*].normalized`.

There is no schema-selection environment variable or alternate projection.
Extractor-owned source collections are private inputs and are converted to the
canonical PBOC datasets before Community JSON is assembled.

The cutover applies only to the `personal_detail_scanned` variant. Digital
enterprise and digital personal-brief credit reports retain their independent
variant dictionaries and projection paths.

Potentially flawed field observations and extraction issues are exposed as
typed control datasets. Successful observation metadata remains omitted.

