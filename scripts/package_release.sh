#!/usr/bin/env bash
# Create the complete DocMirror source archive used by the testing server.

set -Eeuo pipefail
umask 022

log() {
    printf '[package] %s\n' "$*"
}

fail() {
    printf '[package] ERROR: %s\n' "$*" >&2
    exit 1
}

for command_name in tar sha256sum mktemp cp mkdir mv chmod dirname basename grep awk rm diff cmp find; do
    command -v "$command_name" >/dev/null 2>&1 \
        || fail "Required command not found: $command_name"
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
OUTPUT_INPUT="${1:-$REPO_ROOT/dist/docmirror-complete.tar.gz}"

if [[ "$OUTPUT_INPUT" = /* ]]; then
    OUTPUT="$OUTPUT_INPUT"
else
    OUTPUT="$(pwd -P)/$OUTPUT_INPUT"
fi

[[ "$OUTPUT" == *.tar.gz ]] || fail "Output filename must end in .tar.gz: $OUTPUT"

OUTPUT_DIR="$(dirname -- "$OUTPUT")"
OUTPUT_NAME="$(basename -- "$OUTPUT")"
mkdir -p -- "$OUTPUT_DIR"
OUTPUT_DIR="$(cd -- "$OUTPUT_DIR" && pwd -P)"
OUTPUT="$OUTPUT_DIR/$OUTPUT_NAME"
CHECKSUM="$OUTPUT.sha256"

# The server build expects these paths, and deploy.sh expects the archive root
# to be named "docmirror".
required_entries=(
    pyproject.toml
    README.md
    LICENSE
    Dockerfile
    docker-compose.yml
    docmirror
    docmirror_enterprise
    docmirror_finance
)

optional_entries=(
    .dockerignore
    .env.example
    AUTHORS.md
    CHANGELOG.md
    README_zh-CN.md
    SECURITY.md
    docs
    scripts
)

for entry in "${required_entries[@]}"; do
    [[ -e "$REPO_ROOT/$entry" ]] || fail "Required release entry is missing: $entry"
done

TEMP_PARENT="${TMPDIR:-/tmp}"
[[ -d "$TEMP_PARENT" ]] || fail "Temporary directory does not exist: $TEMP_PARENT"
TEMP_PARENT="$(cd -- "$TEMP_PARENT" && pwd -P)"
TEMP_DIR="$(mktemp -d "$TEMP_PARENT/docmirror-package.XXXXXXXX")"
TEMP_DIR="$(cd -- "$TEMP_DIR" && pwd -P)"
STAGE_DIR="$TEMP_DIR/docmirror"
ARCHIVE_TMP=""
CHECKSUM_TMP=""

cleanup() {
    if [[ -n "${ARCHIVE_TMP:-}" && -f "$ARCHIVE_TMP" ]]; then
        rm -f -- "$ARCHIVE_TMP"
    fi
    if [[ -n "${CHECKSUM_TMP:-}" && -f "$CHECKSUM_TMP" ]]; then
        rm -f -- "$CHECKSUM_TMP"
    fi
    if [[ -n "${TEMP_DIR:-}" \
        && -d "$TEMP_DIR" \
        && "$TEMP_DIR" == "$TEMP_PARENT"/docmirror-package.* \
        && "$TEMP_DIR" != "/" ]]; then
        rm -rf -- "$TEMP_DIR"
    fi
}
trap cleanup EXIT INT TERM

mkdir -p -- "$STAGE_DIR"

copy_entry() {
    local entry="$1"
    cp -a -- "$REPO_ROOT/$entry" "$STAGE_DIR/$entry"
}

log "Staging required release files"
for entry in "${required_entries[@]}"; do
    copy_entry "$entry"
done

for entry in "${optional_entries[@]}"; do
    if [[ -e "$REPO_ROOT/$entry" ]]; then
        copy_entry "$entry"
    fi
done

# Detect edits that land while the staging copy is being made. Ignored runtime
# files are excluded because they are intentionally absent from the archive.
log "Checking staged files against the working tree"
for entry in "${required_entries[@]}" "${optional_entries[@]}"; do
    [[ -e "$REPO_ROOT/$entry" ]] || continue
    if [[ -d "$REPO_ROOT/$entry" ]]; then
        diff -qr --no-dereference \
            --exclude='__pycache__' \
            --exclude='*.pyc' \
            --exclude='*.pyo' \
            --exclude='.DS_Store' \
            --exclude='.plugin_state.json' \
            --exclude='.pytest_cache' \
            --exclude='.mypy_cache' \
            --exclude='.ruff_cache' \
            --exclude='.git' \
            --exclude='.env' \
            --exclude='credentials' \
            --exclude='secrets' \
            --exclude='*.log' \
            "$REPO_ROOT/$entry" "$STAGE_DIR/$entry" \
            || fail "Working tree changed while packaging: $entry; run the command again"
    else
        cmp -s -- "$REPO_ROOT/$entry" "$STAGE_DIR/$entry" \
            || fail "Working tree changed while packaging: $entry; run the command again"
    fi
done

special_entry="$(find "$STAGE_DIR" ! -type d ! -type f -print -quit)"
[[ -z "$special_entry" ]] \
    || fail "Links and special files are not allowed in the release: $special_entry"

ARCHIVE_TMP="$(mktemp "$OUTPUT_DIR/.docmirror-archive.XXXXXXXX")"
CHECKSUM_TMP="$(mktemp "$OUTPUT_DIR/.docmirror-checksum.XXXXXXXX")"

log "Creating $OUTPUT"
tar -czf "$ARCHIVE_TMP" \
    --exclude='*/__pycache__' \
    --exclude='*.pyc' \
    --exclude='*.pyo' \
    --exclude='*/.DS_Store' \
    --exclude='*/.plugin_state.json' \
    --exclude='*/.pytest_cache' \
    --exclude='*/.mypy_cache' \
    --exclude='*/.ruff_cache' \
    --exclude='*/.git' \
    --exclude='*/.git/*' \
    --exclude='*/.env' \
    --exclude='*/credentials' \
    --exclude='*/credentials/*' \
    --exclude='*/secrets' \
    --exclude='*/secrets/*' \
    --exclude='*.log' \
    -C "$TEMP_DIR" \
    docmirror

tar -tzf "$ARCHIVE_TMP" >/dev/null \
    || fail "Created archive failed gzip/tar validation"

archive_listing="$(tar -tzf "$ARCHIVE_TMP")"
for required_member in \
    docmirror/pyproject.toml \
    docmirror/docmirror/server/api.py \
    docmirror/docmirror_enterprise/ \
    docmirror/docmirror_finance/; do
    grep -Fqx "$required_member" <<<"$archive_listing" \
        || fail "Created archive is missing: $required_member"
done

chmod 0644 "$ARCHIVE_TMP"
mv -f -- "$ARCHIVE_TMP" "$OUTPUT"
ARCHIVE_TMP=""

(
    cd -- "$OUTPUT_DIR"
    sha256sum "$OUTPUT_NAME"
) > "$CHECKSUM_TMP"
chmod 0644 "$CHECKSUM_TMP"
mv -f -- "$CHECKSUM_TMP" "$CHECKSUM"
CHECKSUM_TMP=""

log "Package completed successfully"
printf 'Archive:  %s\n' "$OUTPUT"
printf 'Checksum: %s\n' "$CHECKSUM"
printf 'SHA-256:  %s\n' "$(sha256sum "$OUTPUT" | awk '{print $1}')"
