#!/bin/sh

# DocMirror versioned testing-server deployment using a persistent runtime.
#
# Intended usage:
#   1. Upload docmirror-complete.tar.gz to /home/docmirror/
#   2. Run: cd /home/docmirror && sh deploy-no-deps.sh
#
# Optional:
#   sh deploy-no-deps.sh /absolute/path/to/docmirror-complete.tar.gz
#
# This script never installs or upgrades dependencies. It reuses the pinned
# runtime in /home/docmirror/venv, validates it before cutover, and runs each
# immutable source release with --app-dir. Provision the shared venv separately.

set -eu
umask 022

APP_HOME="/home/docmirror"
ARCHIVE="${1:-$APP_HOME/docmirror-complete.tar.gz}"
RELEASES_DIR="$APP_HOME/releases"
CURRENT_LINK="$APP_HOME/current"
LOCK_DIR="$APP_HOME/.deploy.lock"
PID_FILE="$APP_HOME/api.pid"
LOG_FILE="$APP_HOME/api.log"
HEALTH_HOST="${DOCMIRROR_HOST:-0.0.0.0}"
HEALTH_PORT="${DOCMIRROR_PORT:-8000}"
UVICORN_WORKERS="${DOCMIRROR_UVICORN_WORKERS:-1}"
HEALTH_URL="${DOCMIRROR_HEALTH_URL:-http://127.0.0.1:$HEALTH_PORT/health}"
RUNTIME_LINK="$APP_HOME/runtime-venv"
RUNTIME_LINK_TARGET=""
if [ -n "${DOCMIRROR_RUNTIME_PYTHON:-}" ]; then
    BOOTSTRAP_PYTHON="$DOCMIRROR_RUNTIME_PYTHON"
elif [ -x "$RUNTIME_LINK/bin/python" ]; then
    BOOTSTRAP_PYTHON="$RUNTIME_LINK/bin/python"
elif [ -x "$CURRENT_LINK/venv/bin/python" ]; then
    RUNTIME_LINK_TARGET="$(readlink -f "$CURRENT_LINK/venv")"
    BOOTSTRAP_PYTHON="$RUNTIME_LINK_TARGET/bin/python"
else
    BOOTSTRAP_PYTHON="$APP_HOME/venv/bin/python"
fi
PERSISTENT_ENV_FILE="${DOCMIRROR_ENV_FILE:-$APP_HOME/.env}"
PYMUPDF_VERSION="${DOCMIRROR_PYMUPDF_VERSION:-1.28.0}"
PDFPLUMBER_VERSION="${DOCMIRROR_PDFPLUMBER_VERSION:-0.11.10}"
PYPDF_VERSION="${DOCMIRROR_PYPDF_VERSION:-6.14.2}"
PYDANTIC_VERSION="${DOCMIRROR_PYDANTIC_VERSION:-2.13.4}"
LEGACY_START="$APP_HOME/start.sh"
RELEASE_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
RELEASE_DIR="$RELEASES_DIR/$RELEASE_ID"
HEALTH_FILE="$APP_HOME/.deploy-health.$RELEASE_ID.json"
CURRENT_TEMP="$APP_HOME/.current.$RELEASE_ID"

LOCK_HELD=0
CUTOVER_ACTIVE=0
CURRENT_SWITCHED=0
PREVIOUS_RELEASE=""
PREVIOUS_SERVICE_RUNNING=0

log() {
    printf '[deploy] %s\n' "$*"
}

warn() {
    printf '[deploy] WARNING: %s\n' "$*" >&2
}

fail() {
    printf '[deploy] ERROR: %s\n' "$*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || fail "Required command not found: $1"
}

unlock_deployment() {
    if [ "$LOCK_HELD" -eq 1 ] && [ -d "$LOCK_DIR" ]; then
        if [ -f "$LOCK_DIR/pid" ] && [ "$(cat "$LOCK_DIR/pid" 2>/dev/null || true)" = "$$" ]; then
            rm -f -- "$LOCK_DIR/pid"
            rmdir -- "$LOCK_DIR" 2>/dev/null || true
        fi
    fi
}

read_project_version() {
    "$BOOTSTRAP_PYTHON" - "$1/pyproject.toml" <<'PY'
from pathlib import Path
import re
import sys

text = Path(sys.argv[1]).read_text(encoding="utf-8")
match = re.search(
    r'^version\s*=\s*["\x27]([^"\x27]+)["\x27]\s*$',
    text,
    re.MULTILINE,
)
if not match:
    raise SystemExit("Unable to read project version")
print(match.group(1))
PY
}

process_is_docmirror_api() {
    CHECK_PID="$1"
    case "$CHECK_PID" in
        ''|*[!0-9]*) return 1 ;;
    esac
    [ -r "/proc/$CHECK_PID/cmdline" ] || return 1
    CHECK_CMD="$(tr '\000' ' ' < "/proc/$CHECK_PID/cmdline" 2>/dev/null || true)"
    CHECK_CWD="$(readlink -f "/proc/$CHECK_PID/cwd" 2>/dev/null || true)"
    [ "$CHECK_CWD" = "$APP_HOME" ] || return 1
    case "$CHECK_CMD" in
        *uvicorn*docmirror.server.api:app*) return 0 ;;
        *) return 1 ;;
    esac
}

find_service_pids() {
    {
        if [ -f "$PID_FILE" ]; then
            cat "$PID_FILE" 2>/dev/null || true
        fi
        pgrep -f 'uvicorn.*docmirror\.server\.api:app' 2>/dev/null || true
    } | awk '/^[0-9]+$/ && !seen[$0]++ { print $0 }'
}

service_is_running() {
    RUNNING_PIDS="$(find_service_pids)"
    [ -n "$RUNNING_PIDS" ] || return 1
    for RUNNING_PID in $RUNNING_PIDS; do
        if process_is_docmirror_api "$RUNNING_PID" && kill -0 "$RUNNING_PID" 2>/dev/null; then
            return 0
        fi
    done
    return 1
}

stop_service() {
    STOP_PIDS="$(find_service_pids)"
    if [ -z "$STOP_PIDS" ]; then
        rm -f -- "$PID_FILE"
        log "No running DocMirror API process found"
        return 0
    fi

    VALID_STOP_PIDS=""
    for STOP_PID in $STOP_PIDS; do
        if process_is_docmirror_api "$STOP_PID" && kill -0 "$STOP_PID" 2>/dev/null; then
            VALID_STOP_PIDS="$VALID_STOP_PIDS $STOP_PID"
        fi
    done

    if [ -z "$VALID_STOP_PIDS" ]; then
        rm -f -- "$PID_FILE"
        log "No valid DocMirror API process found"
        return 0
    fi

    log "Stopping DocMirror API process(es):$VALID_STOP_PIDS"
    for STOP_PID in $VALID_STOP_PIDS; do
        kill -TERM "$STOP_PID"
    done

    STOP_WAIT=0
    while [ "$STOP_WAIT" -lt 30 ]; do
        STOP_ALIVE=0
        for STOP_PID in $VALID_STOP_PIDS; do
            if kill -0 "$STOP_PID" 2>/dev/null; then
                STOP_ALIVE=1
            fi
        done
        [ "$STOP_ALIVE" -eq 1 ] || break
        sleep 1
        STOP_WAIT=$((STOP_WAIT + 1))
    done

    for STOP_PID in $VALID_STOP_PIDS; do
        if kill -0 "$STOP_PID" 2>/dev/null; then
            warn "Process $STOP_PID did not stop after 30 seconds"
            return 1
        fi
    done

    rm -f -- "$PID_FILE"
    return 0
}

start_release() {
    START_RELEASE="$1"
    START_PYTHON="$BOOTSTRAP_PYTHON"

    [ -x "$START_PYTHON" ] || {
        warn "Shared runtime Python is unavailable: $START_PYTHON"
        return 1
    }
    [ -f "$START_RELEASE/docmirror/server/api.py" ] || {
        warn "Release API source is unavailable: $START_RELEASE/docmirror/server/api.py"
        return 1
    }

    log "Starting DocMirror from $START_RELEASE"
    {
        printf '\n===== start %s release=%s =====\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$START_RELEASE"
    } >> "$LOG_FILE"

    (
        cd "$APP_HOME"
        nohup "$START_PYTHON" -m uvicorn docmirror.server.api:app \
            --app-dir "$START_RELEASE" \
            --host "$HEALTH_HOST" \
            --port "$HEALTH_PORT" \
            --workers "$UVICORN_WORKERS" \
            >> "$LOG_FILE" 2>&1 &
        printf '%s\n' "$!" > "$PID_FILE"
    )
    START_PID="$(cat "$PID_FILE")"

    sleep 2
    if ! kill -0 "$START_PID" 2>/dev/null; then
        warn "DocMirror process exited during startup; inspect $LOG_FILE"
        return 1
    fi
    return 0
}

wait_for_health() {
    HEALTH_RELEASE="$1"
    HEALTH_VERSION="$2"
    HEALTH_PYTHON="$BOOTSTRAP_PYTHON"
    HEALTH_ATTEMPT=1

    while [ "$HEALTH_ATTEMPT" -le 24 ]; do
        if curl --connect-timeout 3 --max-time 10 -fsS "$HEALTH_URL" > "$HEALTH_FILE" 2>/dev/null; then
            if "$HEALTH_PYTHON" - "$HEALTH_FILE" "$HEALTH_VERSION" <<'PY'
import json
from pathlib import Path
import sys

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected_version = sys.argv[2]
if payload.get("status") != "ok":
    raise SystemExit("health status is not ok")
if payload.get("version") != expected_version:
    raise SystemExit(
        f"health version {payload.get('version')!r} != {expected_version!r}"
    )
PY
            then
                return 0
            fi
        fi

        if [ -f "$PID_FILE" ]; then
            HEALTH_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
            if [ -n "$HEALTH_PID" ] && ! kill -0 "$HEALTH_PID" 2>/dev/null; then
                warn "DocMirror process exited before health check succeeded"
                return 1
            fi
        fi

        sleep 5
        HEALTH_ATTEMPT=$((HEALTH_ATTEMPT + 1))
    done

    warn "Health endpoint did not report status=ok and version=$HEALTH_VERSION within 120 seconds"
    return 1
}

wait_for_legacy_health() {
    LEGACY_ATTEMPT=1
    while [ "$LEGACY_ATTEMPT" -le 24 ]; do
        if curl --connect-timeout 3 --max-time 10 -fsS "$HEALTH_URL" > "$HEALTH_FILE" 2>/dev/null; then
            if "$BOOTSTRAP_PYTHON" - "$HEALTH_FILE" <<'PY'
import json
from pathlib import Path
import sys

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload.get("status") != "ok":
    raise SystemExit("legacy health status is not ok")
PY
            then
                return 0
            fi
        fi
        sleep 5
        LEGACY_ATTEMPT=$((LEGACY_ATTEMPT + 1))
    done
    return 1
}

switch_current_link() {
    SWITCH_TARGET="$1"
    rm -f -- "$CURRENT_TEMP"
    ln -s "$SWITCH_TARGET" "$CURRENT_TEMP"
    mv -Tf -- "$CURRENT_TEMP" "$CURRENT_LINK"
    CURRENT_SWITCHED=1
}

restore_previous_release() {
    warn "Rolling back the failed deployment"
    stop_service || warn "Unable to stop failed candidate cleanly"

    if [ "$CURRENT_SWITCHED" -eq 1 ]; then
        if [ -n "$PREVIOUS_RELEASE" ]; then
            ROLLBACK_LINK="$APP_HOME/.current.rollback.$RELEASE_ID"
            rm -f -- "$ROLLBACK_LINK"
            ln -s "$PREVIOUS_RELEASE" "$ROLLBACK_LINK"
            mv -Tf -- "$ROLLBACK_LINK" "$CURRENT_LINK"
        else
            if [ -L "$CURRENT_LINK" ]; then
                rm -f -- "$CURRENT_LINK"
            fi
        fi
    fi

    [ "$PREVIOUS_SERVICE_RUNNING" -eq 1 ] || {
        warn "Previous service was not running; rollback will not start it"
        return 0
    }

    if [ -n "$PREVIOUS_RELEASE" ] \
        && [ -f "$PREVIOUS_RELEASE/docmirror/server/api.py" ]; then
        PREVIOUS_VERSION="$(read_project_version "$PREVIOUS_RELEASE" 2>/dev/null || true)"
        if start_release "$PREVIOUS_RELEASE"; then
            if [ -n "$PREVIOUS_VERSION" ] && wait_for_health "$PREVIOUS_RELEASE" "$PREVIOUS_VERSION"; then
                warn "Previous version was restored successfully"
                return 0
            fi
        fi
        warn "Previous version could not be restored automatically"
        return 1
    fi

    if [ -f "$LEGACY_START" ] && command -v bash >/dev/null 2>&1; then
        warn "Restoring the legacy service with $LEGACY_START"
        if ! (
            cd "$APP_HOME"
            bash "$LEGACY_START"
        ); then
            warn "Legacy start.sh returned an error"
            return 1
        fi
        if wait_for_legacy_health; then
            warn "Legacy service was restored successfully"
            return 0
        fi
        warn "Legacy service restart did not pass its health check"
        return 1
    fi

    warn "No managed previous release or legacy start.sh is available"
    return 1
}

on_exit() {
    EXIT_STATUS=$?
    trap - 0 1 2 15
    set +e

    if [ "$EXIT_STATUS" -ne 0 ] && [ "$CUTOVER_ACTIVE" -eq 1 ]; then
        restore_previous_release || warn "ROLLBACK FAILED; manual recovery is required"
    fi

    unlock_deployment
    exit "$EXIT_STATUS"
}

trap on_exit 0
trap 'exit 129' 1
trap 'exit 130' 2
trap 'exit 143' 15

[ "$#" -le 1 ] || fail "Usage: sh deploy-no-deps.sh [ARCHIVE]"
[ -d "$APP_HOME" ] || fail "Application home does not exist: $APP_HOME"
[ -f "$ARCHIVE" ] || fail "Release archive not found: $ARCHIVE"
[ -x "$BOOTSTRAP_PYTHON" ] || fail "Bootstrap Python not found: $BOOTSTRAP_PYTHON"

for REQUIRED_COMMAND in tar curl pgrep awk tr kill nohup sleep date mkdir ln mv readlink; do
    require_command "$REQUIRED_COMMAND"
done

case "$HEALTH_PORT" in
    ''|*[!0-9]*) fail "DOCMIRROR_PORT must be an integer" ;;
esac
case "$UVICORN_WORKERS" in
    ''|*[!0-9]*|0) fail "DOCMIRROR_UVICORN_WORKERS must be a positive integer" ;;
esac

if ! mkdir -- "$LOCK_DIR" 2>/dev/null; then
    LOCK_OWNER="$(cat "$LOCK_DIR/pid" 2>/dev/null || true)"
    fail "Another deployment may be active (lock owner: ${LOCK_OWNER:-unknown}); inspect $LOCK_DIR"
fi
LOCK_HELD=1
printf '%s\n' "$$" > "$LOCK_DIR/pid"

if [ -n "$RUNTIME_LINK_TARGET" ]; then
    if [ -e "$RUNTIME_LINK" ] || [ -L "$RUNTIME_LINK" ]; then
        fail "Runtime pointer already exists but was not usable: $RUNTIME_LINK"
    fi
    ln -s "$RUNTIME_LINK_TARGET" "$RUNTIME_LINK"
    BOOTSTRAP_PYTHON="$RUNTIME_LINK/bin/python"
    log "Pinned persistent runtime pointer: $RUNTIME_LINK -> $RUNTIME_LINK_TARGET"
fi
log "Using persistent runtime: $(readlink -f "$BOOTSTRAP_PYTHON")"

log "Validating archive structure and member types"
"$BOOTSTRAP_PYTHON" - "$ARCHIVE" <<'PY'
from pathlib import PurePosixPath
import sys
import tarfile

archive = sys.argv[1]
required = {
    "docmirror/pyproject.toml",
    "docmirror/Dockerfile",
    "docmirror/docker-compose.yml",
    "docmirror/docmirror/server/api.py",
}

with tarfile.open(archive, "r:gz") as handle:
    members = handle.getmembers()

if not members:
    raise SystemExit("archive is empty")

normalized = []
total_size = 0
for member in members:
    name = member.name.rstrip("/")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise SystemExit(f"unsafe archive path: {member.name}")
    if not path.parts or path.parts[0] != "docmirror":
        raise SystemExit(f"member is outside docmirror/ wrapper: {member.name}")
    if not (member.isfile() or member.isdir()):
        raise SystemExit(f"links and special archive members are forbidden: {member.name}")
    normalized.append(name)
    total_size += member.size

if len(normalized) != len(set(normalized)):
    raise SystemExit("archive contains duplicate member names")
if total_size > 10 * 1024 * 1024 * 1024:
    raise SystemExit("archive expands beyond the 10 GiB safety limit")

missing = sorted(required.difference(normalized))
if missing:
    raise SystemExit("archive is missing: " + ", ".join(missing))
PY

if command -v sha256sum >/dev/null 2>&1; then
    ARCHIVE_SHA256="$(sha256sum "$ARCHIVE" | awk '{print $1}')"
    log "Archive SHA-256: $ARCHIVE_SHA256"
    if [ -f "$ARCHIVE.sha256" ]; then
        EXPECTED_SHA256="$(awk 'NF {print $1; exit}' "$ARCHIVE.sha256")"
        [ "$ARCHIVE_SHA256" = "$EXPECTED_SHA256" ] \
            || fail "Archive checksum does not match $ARCHIVE.sha256"
        log "Archive checksum verified"
    else
        warn "No checksum file was uploaded; continuing with the computed digest"
    fi
fi

mkdir -p -- "$RELEASES_DIR"
[ ! -e "$RELEASE_DIR" ] || fail "Release directory already exists: $RELEASE_DIR"
mkdir -- "$RELEASE_DIR"

log "Extracting candidate release to $RELEASE_DIR"
tar -xzf "$ARCHIVE" \
    -C "$RELEASE_DIR" \
    --strip-components=1 \
    --no-same-owner \
    --no-same-permissions

[ -f "$RELEASE_DIR/pyproject.toml" ] || fail "Candidate has no pyproject.toml"
[ -f "$RELEASE_DIR/docmirror/server/api.py" ] || fail "Candidate has no docmirror/server/api.py"

if [ -f "$PERSISTENT_ENV_FILE" ]; then
    ln -s "$PERSISTENT_ENV_FILE" "$RELEASE_DIR/.env"
    log "Linked persistent runtime configuration: $PERSISTENT_ENV_FILE"
else
    warn "Persistent runtime configuration is absent: $PERSISTENT_ENV_FILE"
    warn "The service will use only DOCMIRROR_* variables exported to this deployment process"
fi

CANDIDATE_VERSION="$(read_project_version "$RELEASE_DIR")"
log "Candidate version: $CANDIDATE_VERSION"
if [ -n "${DOCMIRROR_EXPECTED_VERSION:-}" ]; then
    [ "$CANDIDATE_VERSION" = "$DOCMIRROR_EXPECTED_VERSION" ] \
        || fail "Expected $DOCMIRROR_EXPECTED_VERSION, got $CANDIDATE_VERSION"
fi

log "Validating the persistent runtime; no packages will be installed"
"$BOOTSTRAP_PYTHON" - \
    "$PYMUPDF_VERSION" \
    "$PDFPLUMBER_VERSION" \
    "$PYPDF_VERSION" \
    "$PYDANTIC_VERSION" <<'PY'
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
import sys

expected = {
    "PyMuPDF": sys.argv[1],
    "pdfplumber": sys.argv[2],
    "pypdf": sys.argv[3],
    "pydantic": sys.argv[4],
}
problems = []
for distribution, wanted in expected.items():
    try:
        actual = version(distribution)
    except PackageNotFoundError:
        problems.append(f"{distribution} is not installed (expected {wanted})")
        continue
    if actual != wanted:
        problems.append(f"{distribution}=={actual}, expected {wanted}")

required_modules = (
    "fitz",
    "pdfplumber",
    "pypdf",
    "pydantic",
    "fastapi",
    "uvicorn",
    "multipart",
    "dotenv",
    "filetype",
    "yaml",
    "rich",
    "pluggy",
)
for module in required_modules:
    try:
        import_module(module)
    except Exception as exc:
        problems.append(f"cannot import {module}: {type(exc).__name__}: {exc}")

if problems:
    raise SystemExit(
        "Persistent runtime is not ready:\n- " + "\n- ".join(problems)
    )
PY
"$BOOTSTRAP_PYTHON" -m pip check
"$BOOTSTRAP_PYTHON" -m pip freeze > "$RELEASE_DIR/shared-runtime-freeze.txt"

log "Verifying candidate import path and source version"
(
    cd "$RELEASE_DIR"
    PYTHONPATH= "$BOOTSTRAP_PYTHON" - "$RELEASE_DIR" "$CANDIDATE_VERSION" <<'PY'
from pathlib import Path
import sys

import docmirror
import docmirror.server.api as api

release = Path(sys.argv[1]).resolve()
expected_version = sys.argv[2]
expected_api = (release / "docmirror" / "server" / "api.py").resolve()
actual_api = Path(api.__file__).resolve()
source_version = docmirror.__version__

print(f"docmirror package: {Path(docmirror.__file__).resolve()}")
print(f"docmirror API:     {actual_api}")
print(f"source version:    {source_version}")

if actual_api != expected_api:
    raise SystemExit(f"API imported from {actual_api}, expected {expected_api}")
if source_version != expected_version:
    raise SystemExit(
        f"source version {source_version}, expected {expected_version}"
    )
PY
)

if [ -e "$CURRENT_LINK" ] || [ -L "$CURRENT_LINK" ]; then
    [ -L "$CURRENT_LINK" ] || fail "$CURRENT_LINK exists but is not a symbolic link"
    PREVIOUS_RELEASE="$(readlink -f "$CURRENT_LINK")"
    case "$PREVIOUS_RELEASE" in
        "$RELEASES_DIR"/*) ;;
        *) fail "Current release points outside $RELEASES_DIR: $PREVIOUS_RELEASE" ;;
    esac
fi
if service_is_running; then
    PREVIOUS_SERVICE_RUNNING=1
fi
if [ -z "$PREVIOUS_RELEASE" ] && [ "$PREVIOUS_SERVICE_RUNNING" -eq 0 ]; then
    warn "No running legacy service or managed previous release was found"
    warn "This first managed deployment has no automatic service rollback target"
fi

CUTOVER_ACTIVE=1
stop_service || fail "Unable to stop the existing DocMirror API cleanly"
switch_current_link "$RELEASE_DIR"
start_release "$RELEASE_DIR" || fail "Candidate process failed during startup"

if ! wait_for_health "$RELEASE_DIR" "$CANDIDATE_VERSION"; then
    if [ -f "$LOG_FILE" ]; then
        warn "Last 100 server log lines:"
        tail -n 100 "$LOG_FILE" >&2 || true
    fi
    fail "Candidate failed its health check"
fi

CUTOVER_ACTIVE=0

log "Deployment completed successfully"
log "Active release: $RELEASE_DIR"
log "Version: $CANDIDATE_VERSION"
log "Health response: $HEALTH_FILE"
log "Shared runtime snapshot: $RELEASE_DIR/shared-runtime-freeze.txt"
log "Archive retained: $ARCHIVE"
log "Previous releases were retained for rollback"
