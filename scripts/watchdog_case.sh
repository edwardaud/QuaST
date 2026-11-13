#!/usr/bin/env bash
# =============================================================================
# scripts/watchdog_case.sh — keep the realistic-case spin-up alive.
#
# The long realistic-case runs (out_julia/case_realistic_B2, ..._C) have died
# mid-run more than once: the julia process disappears, the log ends at an
# AMR / step record and there is no DONE.  This script is the unattended restart
# for that situation.
#
#   scripts/watchdog_case.sh [CASE_DIR]
#   scripts/watchdog_case.sh --help
#
# CASE_DIR is relative to the repository root (or absolute) and defaults to
# $CASE_DIR from the environment.  The log is $CASE_DIR.log when that file
# exists (the B2 runs redirect there), else $CASE_DIR/run.log; override with
# CASE_LOG.  Every INTERVAL seconds the watchdog checks, in this order:
#
#   1. a live `julia ... case_realistic_2d.jl` process for CASE_DIR?  -> running
#   2. the log ends with the driver's completion record ([case] DONE, possibly
#      followed by [case] final: — see "DONE detection" below)         -> done
#   3. the newest readable *.h5 checkpoint                          -> wait-no-ckpt
#   4. that checkpoint more than STALE_TOL steps behind the last logged step
#      (log and frames/frame_meta.csv)                              -> wait-stale-ckpt
#   5. otherwise: take the launch lock, re-check 1/3/4, back the checkpoint up
#      to <ckpt>.bak and start
#          CASE_OUT=... julia --project=julia case_realistic_2d.jl \
#              resume <ckpt>            (or freeze <ckpt> if amr_every=0)
#      with setsid nohup, appending to the same log                  -> restart
#
# One heartbeat line is appended to $CASE_DIR/watchdog.log per cycle, with the
# timestamp, action, newest step, checkpoint name/step/mtime and the last log
# and frame steps.  The file is never rotated.
#
# Safety properties
#   * a per-case instance lock ($CASE_DIR/.watchdog.instance, pid + liveness
#     check) refuses a second watchdog for the same case;
#   * the resume itself is taken under flock($CASE_DIR/.watchdog.lock), with a
#     full re-check inside the lock and a 1 s settle, so two watchdogs (or a
#     watchdog and a manual resume) cannot both launch;
#   * a launch stamp + LAUNCH_GRACE stops a restart loop after a recent launch;
#   * the checkpoint is copied to <ckpt>.bak before it is used;
#   * stale checkpoints are never resumed: if log/frames are ahead of the
#     checkpoint by more than STALE_TOL steps the watchdog logs a warning
#     heartbeat and waits (a lost checkpoint is better than silently repeating
#     hundreds of steps).
#
# DONE detection: for a normal finish the driver prints
#
#     [case] DONE mode=... steps=N ... 
#     [case] final: Nu_top=...
#
# so the log does not literally *end* with the DONE line.  Completion is
# therefore recognised when the last non-empty line is DONE or the second to
# last is DONE and the last is [case] final:.  A DONE older than the newest
# frame step is treated as stale (a later run got further) and ignored.
#
# Configuration (environment; all optional)
#   JULIA, JULIA_PROJECT, CASE_SCRIPT, PY   julia / project / script / h5 python
#   INTERVAL=120        seconds between checks
#   CASE_STEPS=20000    step budget passed to the resumed run
#   CASE_MAXLEAVES=8192 CASE_COARSEN=0.05 CASE_THETATOL=1e-2
#   CASE_LINEAR=direct  CASE_LATENT=per_phase  CASE_MAXLEV=7
#   CASE_WALLMIN=5 CASE_WALLSTRIP=0.1
#                       wall refinement forwarded to the restart; the checkpoint
#                       stores the same two keys and the driver compares them on
#                       restart, so a case whose checkpoints were made with the
#                       wall unrefined needs CASE_WALLMIN=0 (or a `wallrefine`
#                       seed) or the resume is refused by the restart guard
#   CASE_AMR=classic CASE_SYMMETRY=x CASE_AMR_GATE=0
#                       AMR policy forwarded to the restart.  These three keep
#                       the PRE-2026-11-13 card values on purpose: the watchdog
#                       only resumes existing checkpoints, while the card itself
#                       now defaults to balanced / none / 1, so a run started
#                       under the new defaults must export all three (the
#                       restart guard compares them, and a pre-gate checkpoint
#                       has no amr_coarsen_gate key to compare against).
#   STALE_TOL=400       max (last logged step - checkpoint step) allowed
#   LAUNCH_GRACE=90     seconds after a launch during which no second launch
#   FAIL_BACKOFF=600    sleep after a launch that did not survive 3 s
#   CASE_LOG=...        log path override
#   ONCE=1              one cycle then exit (for tests)
#
# Exit status: 0 on [case] DONE (or ONCE), 1 on configuration/instance errors.
# =============================================================================
set -u

# ------------------------------- configuration ------------------------------
PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
JULIA=${JULIA:-/home/szh/opt/julia-1.12.7/bin/julia}
JULIA_PROJECT=${JULIA_PROJECT:-julia}
CASE_SCRIPT=${CASE_SCRIPT:-julia/scripts/case_realistic_2d.jl}
PY=${PY:-/home/szh/dedalus-env/bin/python}

INTERVAL=${INTERVAL:-120}
CASE_STEPS=${CASE_STEPS:-20000}
CASE_MAXLEAVES=${CASE_MAXLEAVES:-8192}
CASE_COARSEN=${CASE_COARSEN:-0.05}
CASE_THETATOL=${CASE_THETATOL:-1e-2}
CASE_LINEAR=${CASE_LINEAR:-direct}
CASE_LATENT=${CASE_LATENT:-per_phase}
CASE_MAXLEV=${CASE_MAXLEV:-7}
CASE_WALLMIN=${CASE_WALLMIN:-5}
CASE_WALLSTRIP=${CASE_WALLSTRIP:-0.1}
# The three AMR keys keep the old card values: this script resumes a checkpoint
# written under them, while the card now defaults to balanced / none / 1.
CASE_AMR=${CASE_AMR:-classic}
CASE_SYMMETRY=${CASE_SYMMETRY:-x}
CASE_AMR_GATE=${CASE_AMR_GATE:-0}

STALE_TOL=${STALE_TOL:-400}
LAUNCH_GRACE=${LAUNCH_GRACE:-90}
FAIL_BACKOFF=${FAIL_BACKOFF:-600}
LOCK_WAIT=${LOCK_WAIT:-10}
ONCE=${ONCE:-0}

usage() {
    # the header block above (the sed range must be its last line)
    sed -n '2,83p' "$0" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

case "${1:-}" in
    -h|--help) usage 0 ;;
esac
CASE_DIR=${1:-${CASE_DIR:-}}
if [ -z "$CASE_DIR" ]; then
    echo "usage: $0 CASE_DIR   (e.g. out_julia/case_realistic_B2)" >&2
    exit 2
fi
case "$CASE_DIR" in
    */) [ "$CASE_DIR" = "/" ] || CASE_DIR=${CASE_DIR%/} ;;
esac

cd -- "$PROJECT_DIR" || { echo "cannot cd to $PROJECT_DIR" >&2; exit 1; }

case "$CASE_DIR" in
    /*) CASE_ABS=$CASE_DIR ;;
    *)  CASE_ABS=$PROJECT_DIR/$CASE_DIR ;;
esac
CASE_BASE=$(basename -- "$CASE_DIR")

[ -d "$CASE_DIR" ] || { echo "no such case directory: $CASE_DIR" >&2; exit 2; }
[ -x "$JULIA" ] || { echo "julia not executable: $JULIA" >&2; exit 2; }
[ -f "$CASE_SCRIPT" ] || { echo "no case script: $CASE_SCRIPT" >&2; exit 2; }
[ -f "$JULIA_PROJECT/Project.toml" ] || { echo "not a julia project: $JULIA_PROJECT" >&2; exit 2; }

# h5py is used read-only, to get the checkpoint step (the HDF5 attribute written
# by julia/src/semquad_checkpoint.jl) and its amr_every config entry.
if ! "$PY" -c 'import h5py' >/dev/null 2>&1; then
    if command -v python3 >/dev/null 2>&1 && python3 -c 'import h5py' >/dev/null 2>&1; then
        PY=python3
    else
        echo "no python with h5py (tried $PY and python3); set PY=..." >&2
        exit 2
    fi
fi

if [ -n "${CASE_LOG:-}" ]; then
    LOG=$CASE_LOG
elif [ -f "${CASE_DIR}.log" ]; then
    LOG=${CASE_DIR}.log
else
    LOG=$CASE_DIR/run.log
fi

WLOG=$CASE_DIR/watchdog.log
LOCKFILE=$CASE_DIR/.watchdog.lock
LOCKDIR=$CASE_DIR/.watchdog.lock.d
INST_DIR=$CASE_DIR/.watchdog.instance
LAUNCH_STAMP=$CASE_DIR/.watchdog.last_launch

# --------------------------------- helpers ----------------------------------
now_iso() { date '+%Y-%m-%dT%H:%M:%S%z'; }

mtime_iso() {
    if [ -e "$1" ]; then date -r "$1" '+%Y-%m-%dT%H:%M:%S%z' 2>/dev/null; else printf 'none'; fi
}

# hb ACTION PID STEP CKPT CKPT_STEP LOG_STEP FRAME_STEP
hb() {
    printf '%s action=%s pid=%s step=%s ckpt=%s ckpt_step=%s ckpt_mtime=%s log_step=%s frame_step=%s\n' \
        "$(now_iso)" "${1:--}" "${2:--}" "${3:--}" "$(basename -- "${4:--}")" \
        "${5:--}" "$(mtime_iso "${4:--}")" "${6:--}" "${7:--}" >>"$WLOG"
}

proc_alive() {
    local pid=${1:-}
    [ -n "$pid" ] || return 1
    [ -d "/proc/$pid" ] || return 1
    local st
    st=$(awk '{print $3}' "/proc/$pid/stat" 2>/dev/null) || return 1
    [ "$st" != "Z" ]
}

# Live julia processes of this case: comm == julia, cmdline contains the case
# script, and the case directory appears as a path component (or CASE_OUT is in
# the environment, for a `spinup` launch whose argv has no case path).
find_run_pids() {
    local d pid comm cmd matched p env_line v
    for d in /proc/[0-9]*; do
        pid=${d#/proc/}
        [ "$pid" = "$$" ] && continue
        [ -r "$d/comm" ] || continue
        comm=$(cat -- "$d/comm" 2>/dev/null) || continue
        [ "$comm" = "julia" ] || continue
        proc_alive "$pid" || continue
        cmd=$(tr '\0' ' ' <"$d/cmdline" 2>/dev/null) || continue
        case "$cmd" in
            *case_realistic_2d.jl*) ;;
            *) continue ;;
        esac
        matched=0
        for p in "$CASE_DIR" "$CASE_ABS" "$CASE_BASE"; do
            case " $cmd " in
                *" $p "*)  matched=1; break ;;
                *" $p/"*) matched=1; break ;;
            esac
        done
        if [ "$matched" -eq 0 ] && [ -r "$d/environ" ]; then
            while IFS= read -r env_line; do
                case "$env_line" in
                    CASE_OUT=*)
                        v=${env_line#CASE_OUT=}
                        if [ "$v" = "$CASE_DIR" ] || [ "$v" = "$CASE_ABS" ]; then
                            matched=1
                        fi
                        ;;
                esac
            done < <(tr '\0' '\n' <"$d/environ" 2>/dev/null)
        fi
        [ "$matched" -eq 1 ] && printf '%s\n' "$pid"
    done
}

# ckpt_info PATH -> "STEP NLEAVES AMR_EVERY" (AMR_EVERY is "-" when absent).
ckpt_info() {
    "$PY" - "$1" <<'PY' 2>/dev/null
import sys
try:
    import h5py
    with h5py.File(sys.argv[1], "r") as f:
        step = int(f.attrs["step"])
        n = int(f["theta"].shape[0])
        leaves = int(f["leaves"].shape[0])
        if n <= 0 or leaves <= 0:
            sys.exit(1)
        cfg = f.attrs.get("config", b"")
        if isinstance(cfg, bytes):
            cfg = cfg.decode("utf-8", "replace")
        amr = "-"
        for line in str(cfg).splitlines():
            if line.startswith("amr_every="):
                amr = line.split("=", 1)[1].strip() or "-"
        print("%d %d %s" % (step, n, amr))
except Exception:
    sys.exit(1)
PY
}

# Sets CKPT_PATH / CKPT_STEP / CKPT_AMR / CKPT_MTIME / CKPT_UNREADABLE.
# Newest readable checkpoint by step then mtime; *.h5.bak is a candidate too, so
# a checkpoint corrupted by a kill during its h5 write is recoverable from the
# copy the watchdog made before the previous resume.
choose_checkpoint() {
    CKPT_PATH=""; CKPT_STEP=""; CKPT_AMR=""; CKPT_MTIME=0; CKPT_UNREADABLE=0
    local f info step rest amr mt best_step=-1 best_mtime=0
    for f in "$CASE_DIR"/*.h5 "$CASE_DIR"/*.h5.bak; do
        [ -f "$f" ] || continue
        if ! info=$(ckpt_info "$f"); then
            CKPT_UNREADABLE=$((CKPT_UNREADABLE + 1))
            continue
        fi
        step=${info%% *}
        rest=${info#* }
        rest=${rest#* }
        amr=$rest
        mt=$(stat -c %Y -- "$f" 2>/dev/null || echo 0)
        if [ "$step" -gt "$best_step" ] || \
           { [ "$step" -eq "$best_step" ] && [ "$mt" -gt "$best_mtime" ]; }; then
            best_step=$step; best_mtime=$mt
            CKPT_PATH=$f; CKPT_STEP=$step; CKPT_AMR=$amr; CKPT_MTIME=$mt
        fi
    done
}

last_step_in_log() {
    local s
    [ -r "$LOG" ] || { printf '0'; return; }
    s=$(grep -a -oE '(^|[[:space:]])step[ =][0-9]+' "$LOG" 2>/dev/null |
        tail -n 1 | grep -oE '[0-9]+')
    printf '%s' "${s:-0}"
}

last_step_in_frames() {
    local f=$CASE_DIR/frames/frame_meta.csv s
    [ -r "$f" ] || { printf '0'; return; }
    s=$(tail -n 5 -- "$f" | awk -F, 'NF>=2 && $2 ~ /^[0-9]+$/ {s=$2} END {print (s==""?0:s)}')
    printf '%s' "${s:-0}"
}

# Sets DONE_LINE/DONE_STEP and returns 0 when the log finishes a run.
log_completed() {
    DONE_LINE=""; DONE_STEP=0
    [ -r "$LOG" ] || return 1
    local tail6 l1 l2
    tail6=$(tail -n 8 -- "$LOG" | grep -a -v '^[[:space:]]*$')
    l1=$(printf '%s\n' "$tail6" | tail -n 1)
    l2=$(printf '%s\n' "$tail6" | tail -n 2 | head -n 1)
    case "$l1" in
        "[case] DONE"*) DONE_LINE=$l1 ;;
        "[case] final:"*)
            case "$l2" in "[case] DONE"*) DONE_LINE=$l2 ;; esac ;;
    esac
    [ -n "$DONE_LINE" ] || return 1
    DONE_STEP=$(printf '%s\n' "$DONE_LINE" | sed -n 's/.*steps=\([0-9][0-9]*\).*/\1/p')
    DONE_STEP=${DONE_STEP:-0}
    return 0
}

# --------------------------- instance lock (pid) ----------------------------
if ! mkdir -- "$INST_DIR" 2>/dev/null; then
    old=$(cat "$INST_DIR/pid" 2>/dev/null || true)
    if proc_alive "$old"; then
        oldcmd=$(tr '\0' ' ' <"/proc/$old/cmdline" 2>/dev/null || true)
        case "$oldcmd" in
            *watchdog_case.sh*)
                echo "watchdog already running for $CASE_DIR (pid $old)" >&2
                exit 1
                ;;
        esac
    fi
    rm -rf -- "$INST_DIR" 2>/dev/null || true
    mkdir -- "$INST_DIR" || { echo "cannot create $INST_DIR" >&2; exit 1; }
fi
printf '%s\n' "$$" >"$INST_DIR/pid"

# --------------------------- launch lock (flock) -----------------------------
HAVE_FLOCK=0
if command -v flock >/dev/null 2>&1; then
    if ( exec 9>>"$LOCKFILE.probe" && flock -n 9 ) 2>/dev/null; then
        HAVE_FLOCK=1
    fi
    rm -f -- "$LOCKFILE.probe" 2>/dev/null || true
fi
LOCK_HELD=0

lock_acquire() {
    local tries owner
    [ "$LOCK_HELD" -eq 1 ] && return 0
    if [ "$HAVE_FLOCK" -eq 1 ]; then
        exec 9>>"$LOCKFILE" || return 1
        if flock -w "$LOCK_WAIT" 9; then
            LOCK_HELD=1
            return 0
        fi
        exec 9>&- 2>/dev/null || true
        return 1
    fi
    tries=0
    while [ "$tries" -lt 50 ]; do
        if mkdir -- "$LOCKDIR" 2>/dev/null; then
            printf '%s\n' "$$" >"$LOCKDIR/pid"
            LOCK_HELD=1
            return 0
        fi
        owner=$(cat "$LOCKDIR/pid" 2>/dev/null || true)
        if [ -n "$owner" ] && ! proc_alive "$owner"; then
            rm -rf -- "$LOCKDIR" 2>/dev/null || true
            continue
        fi
        tries=$((tries + 1))
        sleep 0.2
    done
    return 1
}

lock_release() {
    [ "$LOCK_HELD" -eq 1 ] || return 0
    if [ "$HAVE_FLOCK" -eq 1 ]; then
        flock -u 9 2>/dev/null || true
        exec 9>&- 2>/dev/null || true
    else
        rm -rf -- "$LOCKDIR" 2>/dev/null || true
    fi
    LOCK_HELD=0
}

SLEEP_PID=""
pause_cycle() {
    if [ "$ONCE" -eq 1 ]; then
        STOP=1
        return 0
    fi
    sleep "$INTERVAL" &
    SLEEP_PID=$!
    wait "$SLEEP_PID" 2>/dev/null
    SLEEP_PID=""
}

cleanup() {
    lock_release 2>/dev/null || true
    if [ -f "$INST_DIR/pid" ] && [ "$(cat "$INST_DIR/pid" 2>/dev/null || true)" = "$$" ]; then
        rm -rf -- "$INST_DIR" 2>/dev/null || true
    fi
}

STOP=0
trap 'STOP=1; [ -n "${SLEEP_PID:-}" ] && kill "$SLEEP_PID" 2>/dev/null; cleanup; exit 0' INT TERM
trap cleanup EXIT

# ---------------------------------- startup ---------------------------------
: >>"$WLOG" || { echo "cannot write $WLOG" >&2; exit 2; }
{
    printf '%s action=start pid=%s step=0 ckpt=- ckpt_step=- ckpt_mtime=none log_step=- frame_step=-\n' "$(now_iso)" "$$"
    printf '# watchdog: case=%s log=%s julia=%s script=%s interval=%ss steps=%s maxleaves=%s linear=%s latent=%s wall_min=%s wall_strip=%s amr=%s symmetry=%s amr_gate=%s flock=%s\n' \
        "$CASE_DIR" "$LOG" "$JULIA" "$CASE_SCRIPT" "$INTERVAL" "$CASE_STEPS" \
        "$CASE_MAXLEAVES" "$CASE_LINEAR" "$CASE_LATENT" \
        "$CASE_WALLMIN" "$CASE_WALLSTRIP" \
        "$CASE_AMR" "$CASE_SYMMETRY" "$CASE_AMR_GATE" "$HAVE_FLOCK"
} >>"$WLOG" 2>/dev/null || true

# --------------------------------- main loop --------------------------------
while [ "$STOP" -eq 0 ]; do
    choose_checkpoint
    log_step=$(last_step_in_log)
    frame_step=$(last_step_in_frames)
    newest=$log_step
    [ "$frame_step" -gt "$newest" ] && newest=$frame_step
    if [ -n "$CKPT_STEP" ] && [ "$CKPT_STEP" -gt "$newest" ]; then
        newest=$CKPT_STEP
    fi

    pids=$(find_run_pids | tr '\n' ' ' | sed 's/ *$//')

    # 1. the run is alive: nothing to do
    if [ -n "$pids" ]; then
        hb running "$pids" "$newest" "${CKPT_PATH:--}" "${CKPT_STEP:--}" "$log_step" "$frame_step"
        pause_cycle
        continue
    fi

    # 2. finished normally
    if log_completed; then
        if [ "$DONE_STEP" -ge "$frame_step" ]; then
            hb done - "$newest" "${CKPT_PATH:--}" "${CKPT_STEP:--}" "$log_step" "$frame_step"
            STOP=1
            break
        fi
        # a later run has progressed past this DONE: it is stale, keep watching
    fi

    # 3. a readable checkpoint must exist
    if [ -z "$CKPT_PATH" ]; then
        hb wait-no-ckpt - "$newest" - - "$log_step" "$frame_step"
        pause_cycle
        continue
    fi

    # 4. never go back hundreds of steps
    logged=$log_step
    [ "$frame_step" -gt "$logged" ] && logged=$frame_step
    if [ $((logged - CKPT_STEP)) -gt "$STALE_TOL" ]; then
        hb wait-stale-ckpt - "$newest" "$CKPT_PATH" "$CKPT_STEP" "$log_step" "$frame_step"
        pause_cycle
        continue
    fi

    # 5. restart, under the launch lock and re-checked inside it
    if ! lock_acquire; then
        hb wait-lock - "$newest" "$CKPT_PATH" "$CKPT_STEP" "$log_step" "$frame_step"
        pause_cycle
        continue
    fi

    sleep 1                      # let a just-forked/just-exiting run settle
    pids=$(find_run_pids | tr '\n' ' ' | sed 's/ *$//')
    if [ -n "$pids" ]; then
        lock_release
        hb running-race "$pids" "$newest" "$CKPT_PATH" "$CKPT_STEP" "$log_step" "$frame_step"
        pause_cycle
        continue
    fi

    now_s=$(date +%s)
    if [ -f "$LAUNCH_STAMP" ]; then
        last_s=$(cat "$LAUNCH_STAMP" 2>/dev/null || true)
        case "${last_s:-}" in ''|*[!0-9]*) last_s=0 ;; esac
        if [ $((now_s - last_s)) -lt "$LAUNCH_GRACE" ]; then
            lock_release
            hb wait-recent-launch - "$newest" "$CKPT_PATH" "$CKPT_STEP" "$log_step" "$frame_step"
            pause_cycle
            continue
        fi
    fi

    case "$CKPT_PATH" in
        *.bak) : ;;   # already the backup copy made before the previous resume
        *)
            if ! cp -p -- "$CKPT_PATH" "$CKPT_PATH.bak"; then
                lock_release
                hb wait-backup-failed - "$newest" "$CKPT_PATH" "$CKPT_STEP" "$log_step" "$frame_step"
                pause_cycle
                continue
            fi
            ;;
    esac
    printf '%s\n' "$now_s" >"$LAUNCH_STAMP"

    if [ "$CKPT_AMR" = "0.0" ] || [ "$CKPT_AMR" = "0" ]; then
        RESTART_MODE=freeze
    else
        RESTART_MODE=resume
    fi

    CASE_OUT="$CASE_DIR" \
    CASE_STEPS="$CASE_STEPS" \
    CASE_MAXLEAVES="$CASE_MAXLEAVES" \
    CASE_COARSEN="$CASE_COARSEN" \
    CASE_THETATOL="$CASE_THETATOL" \
    CASE_LINEAR="$CASE_LINEAR" \
    CASE_LATENT="$CASE_LATENT" \
    CASE_MAXLEV="$CASE_MAXLEV" \
    CASE_WALLMIN="$CASE_WALLMIN" \
    CASE_WALLSTRIP="$CASE_WALLSTRIP" \
    CASE_AMR="$CASE_AMR" \
    CASE_SYMMETRY="$CASE_SYMMETRY" \
    CASE_AMR_GATE="$CASE_AMR_GATE" \
        setsid nohup "$JULIA" --project="$JULIA_PROJECT" "$CASE_SCRIPT" \
            "$RESTART_MODE" "$CKPT_PATH" >>"$LOG" 2>&1 </dev/null 9>&- &
    child=$!
    printf '%s\n' "$child" >"$CASE_DIR/.last_child"
    lock_release

    sleep 3
    pids=$(find_run_pids | tr '\n' ' ' | sed 's/ *$//')
    if [ -n "$pids" ]; then
        hb restart "$pids" "$newest" "$CKPT_PATH" "$CKPT_STEP" "$log_step" "$frame_step"
    else
        hb restart-failed "$child" "$newest" "$CKPT_PATH" "$CKPT_STEP" "$log_step" "$frame_step"
        sleep "$FAIL_BACKOFF" &
        SLEEP_PID=$!
        wait "$SLEEP_PID" 2>/dev/null
        SLEEP_PID=""
        pause_cycle
        continue
    fi
    pause_cycle
done

exit 0
