#!/usr/bin/env bash
# Manage the vNext and legacy DocMirror processes on the testing server.

set -Eeuo pipefail
umask 022

APP_HOME="${DOCMIRROR_APP_HOME:-/home/docmirror}"
ACTION="${1:-status}"
TARGET="${2:-all}"
STARTUP_TIMEOUT_SECONDS="${DOCMIRROR_STARTUP_TIMEOUT_SECONDS:-120}"
STOP_TIMEOUT_SECONDS="${DOCMIRROR_STOP_TIMEOUT_SECONDS:-30}"

log() {
    printf '[docmirror-services] %s\n' "$*"
}

fail() {
    printf '[docmirror-services] ERROR: %s\n' "$*" >&2
    exit 1
}

usage() {
    cat <<'EOF'
Usage: docmirror-services.sh <start|stop|restart|status|logs> [all|vnext|legacy]

Examples:
  ./docmirror-services.sh start all
  ./docmirror-services.sh stop legacy
  ./docmirror-services.sh restart vnext
  ./docmirror-services.sh status all
  ./docmirror-services.sh logs legacy
EOF
}

case "$ACTION" in
    start|stop|restart|status|logs) ;;
    -h|--help|help)
        usage
        exit 0
        ;;
    *)
        usage >&2
        fail "Unsupported action: $ACTION"
        ;;
esac

case "$TARGET" in
    all|vnext|legacy) ;;
    *)
        usage >&2
        fail "Unsupported target: $TARGET"
        ;;
esac

for command_name in curl date kill mkdir nohup readlink sleep tr; do
    command -v "$command_name" >/dev/null 2>&1 \
        || fail "Required command not found: $command_name"
done

service_value() {
    local service="$1"
    local key="$2"

    case "$service:$key" in
        vnext:root) printf '%s' "${DOCMIRROR_VNEXT_ROOT:-$APP_HOME/current}" ;;
        vnext:python) printf '%s' "${DOCMIRROR_VNEXT_PYTHON:-$APP_HOME/current/venv/bin/python}" ;;
        vnext:port) printf '%s' "${DOCMIRROR_VNEXT_PORT:-8000}" ;;
        vnext:pid) printf '%s' "${DOCMIRROR_VNEXT_PID_FILE:-$APP_HOME/api.pid}" ;;
        vnext:log) printf '%s' "${DOCMIRROR_VNEXT_LOG_FILE:-$APP_HOME/api.log}" ;;
        vnext:env) printf '%s' "${DOCMIRROR_VNEXT_ENV_FILE:-$APP_HOME/.env}" ;;
        legacy:root) printf '%s' "${DOCMIRROR_LEGACY_ROOT:-$APP_HOME/docmirror_old}" ;;
        legacy:python) printf '%s' "${DOCMIRROR_LEGACY_PYTHON:-$APP_HOME/docmirror_old/venv/bin/python}" ;;
        legacy:port) printf '%s' "${DOCMIRROR_LEGACY_PORT:-8002}" ;;
        legacy:pid) printf '%s' "${DOCMIRROR_LEGACY_PID_FILE:-$APP_HOME/docmirror_old/api.pid}" ;;
        legacy:log) printf '%s' "${DOCMIRROR_LEGACY_LOG_FILE:-$APP_HOME/docmirror_old/api.log}" ;;
        legacy:env) printf '%s' "${DOCMIRROR_LEGACY_ENV_FILE:-$APP_HOME/docmirror_old/.env.test}" ;;
        *) fail "Unknown service setting: $service:$key" ;;
    esac
}

selected_services() {
    case "$TARGET" in
        all) printf '%s\n' vnext legacy ;;
        *) printf '%s\n' "$TARGET" ;;
    esac
}

resolve_path() {
    local path="$1"
    readlink -f -- "$path" 2>/dev/null || printf '%s' "$path"
}

read_pid() {
    local pid_file="$1"
    local pid=""

    if [[ -f "$pid_file" ]]; then
        read -r pid < "$pid_file" || true
    fi
    case "$pid" in
        ''|*[!0-9]*) return 1 ;;
        *) printf '%s' "$pid" ;;
    esac
}

process_matches_service() {
    local service="$1"
    local pid="$2"
    local service_root expected_root command_line

    [[ -r "/proc/$pid/cmdline" ]] || return 1
    service_root="$(service_value "$service" root)"
    expected_root="$(resolve_path "$service_root")"
    command_line="$(tr '\000' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"

    [[ "$command_line" == *"uvicorn docmirror.server.api:app"* ]] || return 1
    [[ "$command_line" == *"--app-dir $expected_root"* ]] || return 1
    [[ "$command_line" == *"--port $(service_value "$service" port)"* ]]
}

running_pid() {
    local service="$1"
    local pid_file pid

    pid_file="$(service_value "$service" pid)"
    pid="$(read_pid "$pid_file" 2>/dev/null || true)"
    [[ -n "$pid" ]] || return 1
    kill -0 "$pid" 2>/dev/null || return 1
    process_matches_service "$service" "$pid" || return 1
    printf '%s' "$pid"
}

port_is_listening() {
    local port="$1"

    if command -v ss >/dev/null 2>&1; then
        ss -lnt 2>/dev/null | awk -v suffix=":$port" '$1 == "LISTEN" && $4 ~ suffix "$" { found=1 } END { exit !found }'
        return
    fi
    if command -v lsof >/dev/null 2>&1; then
        lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1
        return
    fi
    return 1
}

load_service_env() {
    local service="$1"
    local env_file

    env_file="$(service_value "$service" env)"
    [[ -f "$env_file" ]] || return 0

    log "Loading $service environment from $env_file"
    set -a
    # shellcheck disable=SC1090
    source "$env_file"
    set +a
}

health_url() {
    printf 'http://127.0.0.1:%s/health' "$(service_value "$1" port)"
}

wait_for_health() {
    local service="$1"
    local pid="$2"
    local elapsed=0
    local url

    url="$(health_url "$service")"
    while (( elapsed < STARTUP_TIMEOUT_SECONDS )); do
        if curl --connect-timeout 2 --max-time 5 -fsS "$url" >/dev/null 2>&1; then
            log "$service is healthy at $url"
            return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            log "$service exited during startup; inspect $(service_value "$service" log)"
            return 1
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done

    log "$service did not become healthy within ${STARTUP_TIMEOUT_SECONDS}s"
    return 1
}

start_service() {
    local service="$1"
    local service_root python_bin port pid_file log_file pid existing_pid

    existing_pid="$(running_pid "$service" 2>/dev/null || true)"
    if [[ -n "$existing_pid" ]]; then
        log "$service is already running (pid=$existing_pid, port=$(service_value "$service" port))"
        return 0
    fi

    service_root="$(resolve_path "$(service_value "$service" root)")"
    python_bin="$(resolve_path "$(service_value "$service" python)")"
    port="$(service_value "$service" port)"
    pid_file="$(service_value "$service" pid)"
    log_file="$(service_value "$service" log)"

    [[ -d "$service_root" ]] || fail "$service root does not exist: $service_root"
    [[ -x "$python_bin" ]] || fail "$service Python is not executable: $python_bin"
    [[ -f "$service_root/docmirror/server/api.py" ]] \
        || fail "$service API source is missing: $service_root/docmirror/server/api.py"
    if port_is_listening "$port"; then
        fail "Port $port is already in use, but no valid $service PID was found"
    fi

    mkdir -p -- "$(dirname -- "$pid_file")" "$(dirname -- "$log_file")"
    rm -f -- "$pid_file"
    load_service_env "$service"
    export ENVIRONMENT="${ENVIRONMENT:-test}"

    {
        printf '\n===== start %s service=%s root=%s port=%s =====\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$service" "$service_root" "$port"
    } >> "$log_file"

    log "Starting $service from $service_root on port $port"
    (
        cd -- "$service_root"
        export PYTHONPATH="$service_root${PYTHONPATH:+:$PYTHONPATH}"
        nohup "$python_bin" -m uvicorn docmirror.server.api:app \
            --app-dir "$service_root" \
            --host 0.0.0.0 \
            --port "$port" \
            --workers "${DOCMIRROR_UVICORN_WORKERS:-1}" \
            >> "$log_file" 2>&1 &
        printf '%s\n' "$!" > "$pid_file"
    )
    pid="$(read_pid "$pid_file")"

    if ! wait_for_health "$service" "$pid"; then
        kill -TERM "$pid" 2>/dev/null || true
        rm -f -- "$pid_file"
        return 1
    fi
}

stop_service() {
    local service="$1"
    local pid_file pid elapsed=0

    pid_file="$(service_value "$service" pid)"
    pid="$(running_pid "$service" 2>/dev/null || true)"
    if [[ -z "$pid" ]]; then
        if [[ -f "$pid_file" ]]; then
            log "$service PID file is stale or does not identify the configured service; removing only the PID file"
            rm -f -- "$pid_file"
        else
            log "$service is not running"
        fi
        return 0
    fi

    log "Stopping $service (pid=$pid)"
    kill -TERM "$pid"
    while kill -0 "$pid" 2>/dev/null && (( elapsed < STOP_TIMEOUT_SECONDS )); do
        sleep 1
        elapsed=$((elapsed + 1))
    done
    if kill -0 "$pid" 2>/dev/null; then
        fail "$service did not stop within ${STOP_TIMEOUT_SECONDS}s; process was not force-killed"
    fi
    rm -f -- "$pid_file"
    log "$service stopped"
}

status_service() {
    local service="$1"
    local pid

    pid="$(running_pid "$service" 2>/dev/null || true)"
    if [[ -n "$pid" ]]; then
        if curl --connect-timeout 2 --max-time 5 -fsS "$(health_url "$service")" >/dev/null 2>&1; then
            printf '%-7s RUNNING pid=%s port=%s health=ok root=%s\n' \
                "$service" "$pid" "$(service_value "$service" port)" \
                "$(resolve_path "$(service_value "$service" root)")"
            return 0
        fi
        printf '%-7s DEGRADED pid=%s port=%s health=failed root=%s\n' \
            "$service" "$pid" "$(service_value "$service" port)" \
            "$(resolve_path "$(service_value "$service" root)")"
        return 1
    fi

    printf '%-7s STOPPED port=%s root=%s\n' \
        "$service" "$(service_value "$service" port)" \
        "$(resolve_path "$(service_value "$service" root)")"
    return 1
}

logs_service() {
    local service="$1"
    local log_file

    log_file="$(service_value "$service" log)"
    [[ -f "$log_file" ]] || fail "$service log file does not exist: $log_file"
    tail -n "${DOCMIRROR_LOG_LINES:-200}" -- "$log_file"
}

run_for_selected_services() {
    local operation="$1"
    local service
    local failed=0

    while IFS= read -r service; do
        if ! "$operation" "$service"; then
            failed=1
        fi
    done < <(selected_services)
    return "$failed"
}

case "$ACTION" in
    start)
        run_for_selected_services start_service
        ;;
    stop)
        # Stop the legacy service first so a partial stop keeps vNext available.
        if [[ "$TARGET" == "all" ]]; then
            stop_service legacy
            stop_service vnext
        else
            run_for_selected_services stop_service
        fi
        ;;
    restart)
        if [[ "$TARGET" == "all" ]]; then
            stop_service legacy
            stop_service vnext
            start_service vnext
            start_service legacy
        else
            run_for_selected_services stop_service
            run_for_selected_services start_service
        fi
        ;;
    status)
        run_for_selected_services status_service
        ;;
    logs)
        run_for_selected_services logs_service
        ;;
esac
