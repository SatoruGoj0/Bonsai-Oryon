#!/usr/bin/env bash
# Bonsai Oryon benchmark - run on the target laptop (no root needed).
#
#   bonsai-benchmark.sh [-m MODEL] [-b BASELINE_LLAMA_BENCH] [-r REPS] [-o OUTDIR] [--quick] [--no-mem]
#
# Every experiment uses CPU only (-ngl 0), the same model file and -r 3 repetitions (tokens/s +- stddev):
#   E0  bonsai-diag: DRAM read bandwidth vs tg-shaped PQ2_0 GEMV throughput (is decoding memory-bound?)
#   E1  tg128 + pp256 thread sweep 4/6/7/8 with the default (v0.2) kernels
#   E2  v0.2 4-row PQ2_0 layout vs v0.1 kernel path (BONSAI_PQ2_REPACK=0) at 8 threads and the best E1 count
#   E3  OpenMP thread placement: default vs OMP_PROC_BIND=close/OMP_PLACES=cores vs OMP_WAIT_POLICY=active
#   E4  optional (-b): the same settings with your previous llama-bench build
# --quick: fewer threads/variants (about 5 minutes instead of about 20). Results: <OUTDIR>/report.md + raw JSON.
set -uo pipefail
MODEL="${HOME}/bonsai-models/Ternary-Bonsai-2-9B-PQ2_0.gguf"
BASELINE=""; MEM=1; QUICK=0; OUT="./bonsai-bench-$(date +%Y%m%d-%H%M%S)"
BENCH="${BONSAI_BENCH:-bonsai-bench}"
DIAG="${BONSAI_DIAG:-/usr/lib/bonsai-oryon/bin/bonsai-diag}"
REPS=3; PP=256; TG=128
while [ $# -gt 0 ]; do
    case "$1" in
        -m) MODEL="$2"; shift 2;;
        -b) BASELINE="$2"; shift 2;;
        -o) OUT="$2"; shift 2;;
        -r) REPS="$2"; shift 2;;
        --quick) QUICK=1; shift;;
        --sweep) shift;;                      # v0.1 option, the sweep is always on now
        --no-mem) MEM=0; shift;;
        -h|--help) sed -n '2,14p' "$0"; exit 0;;
        *) echo "unknown option $1" >&2; exit 2;;
    esac
done
[ -f "$MODEL" ] || { echo "model not found: $MODEL" >&2; exit 1; }
command -v "$BENCH" >/dev/null || { echo "$BENCH not found (is bonsai-oryon installed?)" >&2; exit 1; }
mkdir -p "$OUT"; R="$OUT/report.md"
NCPU=$(nproc)
if [ $QUICK = 1 ]; then SWEEP="6,8"; else SWEEP="4,6,7,8"; fi

snapshot_freqs() { # $1 = label
    {
        echo "#### clocks ($1)"
        echo '```'
        for c in /sys/devices/system/cpu/cpufreq/policy*; do
            [ -d "$c" ] && echo "$(basename "$c"): cpus=$(cat "$c/related_cpus" 2>/dev/null) cur=$(cat "$c/scaling_cur_freq" 2>/dev/null) max=$(cat "$c/scaling_max_freq" 2>/dev/null) hw_max=$(cat "$c/cpuinfo_max_freq" 2>/dev/null) gov=$(cat "$c/scaling_governor" 2>/dev/null)"
        done
        for d in /sys/class/devfreq/*; do
            [ -d "$d" ] && echo "devfreq $(basename "$d") ($(cat "$d/name" 2>/dev/null)): cur=$(cat "$d/cur_freq" 2>/dev/null) max=$(cat "$d/max_freq" 2>/dev/null) gov=$(cat "$d/governor" 2>/dev/null)"
        done
        echo "thermal: $(for z in /sys/class/thermal/thermal_zone*; do printf '%s=%s ' "$(cat "$z/type" 2>/dev/null)" "$(cat "$z/temp" 2>/dev/null)"; done | head -c 600)"
        echo '```'
    } >> "$R"
}

{
echo "# Bonsai Oryon benchmark report"
echo
echo "- date: $(date -Iseconds)"
echo "- machine: $(cat /sys/devices/virtual/dmi/id/sys_vendor 2>/dev/null) $(cat /sys/devices/virtual/dmi/id/product_name 2>/dev/null) $(cat /sys/devices/virtual/dmi/id/product_version 2>/dev/null)"
echo "- cpu: $(lscpu | sed -n 's/^Model name: *//p' | sort -u | tr '\n' ' ')($NCPU cpus)"
echo "- kernel: $(uname -sr), $(. /etc/os-release; echo "$PRETTY_NAME")"
echo "- cpu features: $(grep -m1 Features /proc/cpuinfo | cut -d: -f2)"
echo "- power: $(cat /sys/class/power_supply/*/status 2>/dev/null | tr '\n' ' ') profile: $(cat /sys/firmware/acpi/platform_profile 2>/dev/null || echo n/a)"
echo "- memory: $(free -h | awk '/Mem:/{print $2" total, "$7" available"}')"
echo "- load average at start: $(cut -d' ' -f1-3 /proc/loadavg)"
echo "- model: $MODEL ($(du -h "$MODEL" | cut -f1))"
echo "- bonsai-bench: $(command -v "$BENCH") / $("$BENCH" --version 2>&1 | grep -m1 -i version)"
echo
echo "All runs: \`-ngl 0 -r $REPS\`, pp$PP and tg$TG, same model file, llama-bench defaults otherwise (batch 2048, ubatch 512, mmap)."
echo
} > "$R"
snapshot_freqs "idle, before the runs"

# ---- E0: diagnostics
if [ -x "$DIAG" ]; then
    echo ">> E0 bonsai-diag" >&2
    { echo "## E0. Memory bandwidth vs PQ2_0 GEMV (bonsai-diag)"; echo; echo '```'; } >> "$R"
    "$DIAG" "$NCPU" $([ $QUICK = 1 ] && echo --quick) 2>&1 | tee "$OUT/diag.txt" >> "$R"
    { echo '```'; echo; } >> "$R"
else
    echo "(bonsai-diag not found at $DIAG; skipping E0)" >> "$R"
fi

# run_bench <label> <binary> <threads-list> <pp> <tg> [ENV=VAL ...]
run_bench() {
    local label="$1" bin="$2" ts="$3" pp="$4" tg="$5"; shift 5
    local j="$OUT/${label}.json"
    echo ">> $label: $* $bin -t $ts -p $pp -n $tg" >&2
    if ! env "$@" "$bin" -m "$MODEL" -ngl 0 -t "$ts" -p "$pp" -n "$tg" -r "$REPS" -o json > "$j" 2> "$OUT/${label}.log"; then
        echo "| $label | $ts | failed (see ${label}.log) | | |" >> "$R"; return 1
    fi
    python3 - "$j" "$label" "$*" >> "$R" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); label, env = sys.argv[2], sys.argv[3] or "-"
rows = {}
for r in d:
    t = r["n_threads"]; key = "pp" if r["n_prompt"] and not r["n_gen"] else "tg"
    rows.setdefault(t, {})[key] = "%.2f ± %.2f" % (r["avg_ts"], r["stddev_ts"])
for t in sorted(rows):
    print(f"| {label} | `{env}` | {t} | {rows[t].get('pp', '-')} | {rows[t].get('tg', '-')} |")
PY
}
best_tg_threads() { # prints the thread count with the highest tg in the given json
    python3 - "$1" <<'PY'
import json, sys
d = [r for r in json.load(open(sys.argv[1])) if r["n_gen"] and not r["n_prompt"]]
print(max(d, key=lambda r: r["avg_ts"])["n_threads"] if d else 8)
PY
}
table_head() { echo "| run | env | threads | pp$PP tok/s | tg$TG tok/s |" >> "$R"; echo "|---|---|---|---|---|" >> "$R"; }

# ---- E1: thread sweep with the default kernels
{ echo "## E1. Thread sweep (default = v0.2 kernels)"; echo; } >> "$R"; table_head
run_bench e1-sweep "$BENCH" "$SWEEP" "$PP" "$TG"
BEST=$(best_tg_threads "$OUT/e1-sweep.json" 2>/dev/null || echo 8)
echo >> "$R"; echo "Best tg thread count: **$BEST**" >> "$R"; echo >> "$R"
snapshot_freqs "after E1"

# ---- E2: 4-row layout on/off
{ echo "## E2. v0.2 4-row PQ2_0 layout vs v0.1 kernel path"; echo; } >> "$R"; table_head
TS2=$(printf '%s\n' 8 "$BEST" | sort -un | paste -sd,)
run_bench e2-v02-4row  "$BENCH" "$TS2" "$PP" "$TG"
run_bench e2-v01-plain "$BENCH" "$TS2" "$PP" "$TG" BONSAI_PQ2_REPACK=0
echo >> "$R"

# ---- E3: OpenMP placement / waiting
if [ $QUICK = 0 ]; then
    { echo "## E3. OpenMP thread placement (tg only, threads=$BEST)"; echo; } >> "$R"; table_head
    run_bench e3-default "$BENCH" "$BEST" 0 "$TG"
    run_bench e3-bind    "$BENCH" "$BEST" 0 "$TG" OMP_PROC_BIND=close OMP_PLACES=cores
    run_bench e3-active  "$BENCH" "$BEST" 0 "$TG" OMP_WAIT_POLICY=active GOMP_SPINCOUNT=infinite
    echo >> "$R"
fi

# ---- E4: previous build
if [ -n "$BASELINE" ]; then
    { echo "## E4. Your previous build ($BASELINE)"; echo; } >> "$R"; table_head
    run_bench e4-baseline "$BASELINE" "$TS2" "$PP" "$TG"
    echo >> "$R"
fi

# ---- peak memory
if [ $MEM = 1 ] && [ -x /usr/bin/time ]; then
    { echo "## Peak memory (max RSS, one tg$TG run, threads=$BEST)"; echo; } >> "$R"
    for v in "v0.2-4row:" "v0.1-plain:BONSAI_PQ2_REPACK=0"; do
        l=${v%%:*}; e=${v#*:}
        env $e /usr/bin/time -v "$BENCH" -m "$MODEL" -ngl 0 -t "$BEST" -p 0 -n "$TG" -r 1 > /dev/null 2> "$OUT/mem-$l.log"
        echo "- $l: $(sed -n 's/.*Maximum resident set size (kbytes): //p' "$OUT/mem-$l.log" | awk '{printf "%.0f MiB", $1/1024}')" >> "$R"
    done
    echo "  (the 4-row layout copies the weights into anonymous memory; the mmapped file pages become reclaimable cache)" >> "$R"
    echo >> "$R"
fi
snapshot_freqs "end"
echo "Raw data: $OUT" >> "$R"
echo; cat "$R"
