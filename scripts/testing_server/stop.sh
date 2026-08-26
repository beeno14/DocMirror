#!/usr/bin/env bash
# Backward-compatible entry point: stop both services by default.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec "$SCRIPT_DIR/docmirror-services.sh" stop "${1:-all}"
