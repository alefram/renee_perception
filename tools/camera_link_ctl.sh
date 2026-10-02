#!/bin/bash
# Start/stop/restart/status for the ZED camera link service on the Jetson, without
# systemd or sudo (nohup + pidfile). Idempotent. Run it on the Jetson, from anywhere:
#     tools/camera_link_ctl.sh start|stop|restart|status
#
# Contract (the host's jetson-code-sync relies on it):
#   - prints exactly one status line on stdout, prefixed "camera-link:";
#   - exit 0: start/restart -> service is running; stop -> service is not running
#     (also when it already was not); status -> running;
#   - exit 1: failure (start/restart: process died or never appeared; stop: still alive
#     after SIGKILL) and, for `status`, "not running";
#   - stop/restart wait until the old process has exited, i.e. the ZED is released, before
#     returning (restart then waits RESTART_SETTLE seconds for the camera's USB to settle).
#
# Environment (all optional):
#   CAMERA_LINK_ARGS     extra arguments for camera_link_service.py (default: none; the
#                        service defaults to the ZED on 127.0.0.1:7788)
#   CAMERA_LINK_HOME     state directory (default ~/.camera_link): service.pid, service.log
#   CAMERA_LINK_START_WAIT   seconds start waits to see the process stay alive (default 4)
#   CAMERA_LINK_STOP_WAIT    seconds stop waits after SIGTERM before SIGKILL (default 10)
#   CAMERA_LINK_RESTART_SETTLE  seconds between stop and start on restart (default 2)
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
SERVICE="$SCRIPT_DIR/camera_link_service.py"
STATE_DIR="${CAMERA_LINK_HOME:-$HOME/.camera_link}"
PIDFILE="$STATE_DIR/service.pid"
LOGFILE="$STATE_DIR/service.log"
START_WAIT="${CAMERA_LINK_START_WAIT:-4}"
STOP_WAIT="${CAMERA_LINK_STOP_WAIT:-10}"
RESTART_SETTLE="${CAMERA_LINK_RESTART_SETTLE:-2}"
MAX_LOG_BYTES=5242880

say() { echo "camera-link: $*"; }

# pid of our running service, or empty. Checks the command line so a recycled pid is not
# mistaken for it.
running_pid() {
    local pid
    [ -f "$PIDFILE" ] || return 1
    pid="$(cat "$PIDFILE" 2>/dev/null)"
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null || return 1
    ps -p "$pid" -o args= 2>/dev/null | grep -q "camera_link_service.py" || return 1
    echo "$pid"
}

# Best-effort camera state from the service log (the service logs every open attempt).
camera_state() {
    local line
    line="$(grep -E "camera ready|camera open failed|camera failure|ZED opened" "$LOGFILE" 2>/dev/null | tail -n 1)"
    case "$line" in
        *"camera ready"*|*"ZED opened"*) echo ready ;;
        *"open failed"*|*"failure"*) echo "error ($(echo "$line" | sed -E 's/^[0-9:]+ //; s/^camera open failed: //' | cut -c1-200))" ;;
        *) echo opening ;;
    esac
}

status_line() {
    local pid
    if pid="$(running_pid)"; then
        say "running pid=$pid camera=$(camera_state) log=$LOGFILE"
        return 0
    fi
    say "not running"
    return 1
}

do_start() {
    local pid i
    if pid="$(running_pid)"; then
        say "already running pid=$pid camera=$(camera_state) log=$LOGFILE"
        return 0
    fi
    mkdir -p "$STATE_DIR" || { say "cannot create $STATE_DIR"; return 1; }
    rm -f "$PIDFILE"
    if [ -f "$LOGFILE" ] && [ "$(stat -c %s "$LOGFILE" 2>/dev/null || echo 0)" -gt "$MAX_LOG_BYTES" ]; then
        mv -f "$LOGFILE" "$LOGFILE.1"
    fi
    # shellcheck disable=SC2086  # CAMERA_LINK_ARGS is a word list on purpose
    # The whole launcher is detached from our stdout/stderr/stdin, otherwise a caller such as
    # `ssh jetson ctl start` never returns while the service holds its pipes.
    (cd "$REPO_DIR" && setsid nohup bash -c 'echo $$ > "$0"; exec python3 -u "$@"' \
        "$PIDFILE" "$SERVICE" ${CAMERA_LINK_ARGS:-} >> "$LOGFILE" 2>&1 < /dev/null &) \
        > /dev/null 2>&1 < /dev/null
    # The service must stay alive for START_WAIT seconds (catches port-in-use, syntax errors).
    for ((i = 0; i < START_WAIT * 5; i++)); do
        sleep 0.2
        if [ -f "$PIDFILE" ] && ! running_pid > /dev/null; then
            say "failed to start (exited immediately), see $LOGFILE: $(tail -n 1 "$LOGFILE" 2>/dev/null | cut -c1-120)"
            rm -f "$PIDFILE"
            return 1
        fi
    done
    if pid="$(running_pid)"; then
        say "started pid=$pid camera=$(camera_state) log=$LOGFILE"
        return 0
    fi
    say "failed to start, see $LOGFILE"
    return 1
}

do_stop() {
    local pid i
    if ! pid="$(running_pid)"; then
        rm -f "$PIDFILE"
        say "not running"
        return 0
    fi
    kill -TERM "$pid" 2>/dev/null
    for ((i = 0; i < STOP_WAIT * 5; i++)); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.2
    done
    if kill -0 "$pid" 2>/dev/null; then
        kill -KILL "$pid" 2>/dev/null
        for ((i = 0; i < 25; i++)); do
            kill -0 "$pid" 2>/dev/null || break
            sleep 0.2
        done
    fi
    if kill -0 "$pid" 2>/dev/null; then
        say "failed to stop pid=$pid"
        return 1
    fi
    rm -f "$PIDFILE"
    say "stopped (camera released)"
    return 0
}

case "${1:-}" in
    start)   do_start ;;
    stop)    do_stop ;;
    restart) do_stop || exit 1; sleep "$RESTART_SETTLE"; do_start ;;
    status)  status_line ;;
    *)       echo "usage: $0 start|stop|restart|status" >&2; exit 2 ;;
esac
