#!/usr/bin/env bash
set -euo pipefail

runtime_root="${GLMRT_WIP_RUNTIME_ROOT:-/wip/run}"
mkdir -p "$runtime_root"

valid_name() {
  [[ "$1" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]]
}

process_alive() {
  local pid="$1"
  [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null
}

process_matches() {
  local name="$1" pid="$2" command_line
  process_alive "$pid" || return 1
  command_line="$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null || true)"
  [[ "$command_line" == *wip-process.sh*run*"$name"* ]]
}

process_start_ticks() {
  local pid="$1"
  awk '{print $22}' "/proc/$pid/stat" 2>/dev/null
}

bound_identity() {
  local name="$1" pid_file identity_file pid start_ticks
  local schema recorded_pid recorded_start fingerprint
  pid_file="$runtime_root/$name.pid"
  identity_file="$runtime_root/$name.identity"
  [[ -s "$pid_file" && -s "$identity_file" ]] || return 1
  pid="$(<"$pid_file")"
  process_matches "$name" "$pid" || return 1
  start_ticks="$(process_start_ticks "$pid")"
  [[ -n "$start_ticks" ]] || return 1
  schema="$(sed -n 's/^schema=//p' "$identity_file")"
  recorded_pid="$(sed -n 's/^pid=//p' "$identity_file")"
  recorded_start="$(sed -n 's/^start_ticks=//p' "$identity_file")"
  fingerprint="$(sed -n 's/^fingerprint=//p' "$identity_file")"
  [[ "$schema" == 1 && "$recorded_pid" == "$pid" && "$recorded_start" == "$start_ticks" ]] ||
    return 1
  [[ "$fingerprint" =~ ^[0-9a-f]{64}$ ]] || return 1
  printf '%s\n' "$fingerprint"
}

case "${1:-}" in
  run)
    name="${2:-}"
    shift 2
    valid_name "$name" || { echo "invalid WIP process name: $name" >&2; exit 2; }
    [[ $# -gt 0 ]] || { echo "WIP process command is missing" >&2; exit 2; }
    pid_file="$runtime_root/$name.pid"
    log_file="$runtime_root/$name.log"
    identity_file="$runtime_root/$name.identity"
    if [[ -f "$pid_file" ]] && process_matches "$name" "$(<"$pid_file")"; then
      echo "WIP process is already running: $name" >&2
      exit 2
    fi
    rm -f "$pid_file" "$identity_file"
    echo "$$" >"$pid_file"
    trap 'rm -f "$pid_file" "$identity_file"' EXIT
    exec >>"$log_file" 2>&1
    echo "== $(date --iso-8601=seconds) starting $name: $* =="
    "$@" &
    child=$!
    trap 'kill -TERM "$child" 2>/dev/null || true' TERM INT
    set +e
    wait "$child"
    status=$?
    set -e
    echo "== $(date --iso-8601=seconds) $name exited status=$status =="
    exit "$status"
    ;;
  stop)
    name="${2:-}"
    valid_name "$name" || { echo "invalid WIP process name: $name" >&2; exit 2; }
    pid_file="$runtime_root/$name.pid"
    identity_file="$runtime_root/$name.identity"
    [[ -f "$pid_file" ]] || exit 0
    pid="$(<"$pid_file")"
    if ! process_matches "$name" "$pid"; then
      rm -f "$pid_file" "$identity_file"
      exit 0
    fi
    kill -TERM "$pid"
    for _ in $(seq 1 300); do
      process_alive "$pid" || { rm -f "$pid_file" "$identity_file"; exit 0; }
      sleep 0.1
    done
    echo "WIP process did not stop within 30 seconds: $name pid=$pid" >&2
    exit 2
    ;;
  status)
    name="${2:-}"
    valid_name "$name" || { echo "invalid WIP process name: $name" >&2; exit 2; }
    pid_file="$runtime_root/$name.pid"
    if [[ -f "$pid_file" ]] && process_matches "$name" "$(<"$pid_file")"; then
      echo "running $(<"$pid_file")"
    else
      echo stopped
    fi
    ;;
  bind-identity)
    name="${2:-}"
    fingerprint="${3:-}"
    valid_name "$name" || { echo "invalid WIP process name: $name" >&2; exit 2; }
    [[ "$fingerprint" =~ ^[0-9a-f]{64}$ ]] || {
      echo "invalid WIP process fingerprint for $name" >&2
      exit 2
    }
    pid_file="$runtime_root/$name.pid"
    identity_file="$runtime_root/$name.identity"
    [[ -s "$pid_file" ]] || { echo "WIP process is not running: $name" >&2; exit 2; }
    pid="$(<"$pid_file")"
    process_matches "$name" "$pid" || {
      echo "WIP process identity does not match pid $pid: $name" >&2
      exit 2
    }
    start_ticks="$(process_start_ticks "$pid")"
    [[ -n "$start_ticks" ]] || { echo "cannot read process start time: $name" >&2; exit 2; }
    identity_tmp="$(mktemp "$runtime_root/.${name}.identity.XXXXXX")"
    trap 'rm -f "$identity_tmp"' EXIT
    {
      printf 'schema=1\n'
      printf 'pid=%s\n' "$pid"
      printf 'start_ticks=%s\n' "$start_ticks"
      printf 'fingerprint=%s\n' "$fingerprint"
    } >"$identity_tmp"
    chmod 0600 "$identity_tmp"
    mv -f "$identity_tmp" "$identity_file"
    trap - EXIT
    ;;
  identity)
    name="${2:-}"
    valid_name "$name" || { echo "invalid WIP process name: $name" >&2; exit 2; }
    bound_identity "$name"
    ;;
  log)
    name="${2:-}"
    valid_name "$name" || { echo "invalid WIP process name: $name" >&2; exit 2; }
    tail -n "${3:-200}" "$runtime_root/$name.log" 2>/dev/null || true
    ;;
  *)
    echo "usage: wip-process.sh run|stop|status|bind-identity|identity|log NAME [ARGS...]" >&2
    exit 2
    ;;
esac
