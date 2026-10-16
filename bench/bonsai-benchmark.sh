#!/usr/bin/env bash
# Bonsai Oryon benchmark - run on the target laptop (no root needed).
#
#   bonsai-benchmark.sh [-m MODEL] [-b BASELINE_LLAMA_BENCH] [--sweep] [--no-mem] [-o OUTDIR]
#
# Default run mirrors the reference measurement: CPU only (-ngl 0), 8 threads, pp256 + tg128,
# 3 repetitions, same model file. With -b it runs the SAME settings with your existing llama.cpp
# llama-bench (e.g. ~/llama.cpp/build/bin/llama-bench) back to back for an A/B comparison.
# --sweep adds 2/4/6/8 threads. Results: <OUTDIR>/report.md plus raw JSON/logs.
set -uo pipefail
MODEL="${HOME}/bonsai-models/Ternary-Bonsai-2-9B-PQ2_0.gguf"
BASELINE=""; SWEEP=0; MEM=1; OUT="./bonsai-bench-$(date +%Y%m%d-%H%M%S)"
BENCH="${BONSAI_BENCH:-bonsai-bench}"
REPS=3; PP=256; TG=128; THREADS=8
while [ $# -gt 0 ]; do
    case "$1" in
        -m) MODEL="$2"; shift 2;;
        -b) BASELINE="$2"; shift 2;;
        -o) OUT="$2"; shift 2;;
        -r) REPS="$2"; shift 2;;
        --sweep) SWEEP=1; shift;;
        --no-mem) MEM=0; shift;;
        -h|--help) sed -n '2,12p' "$0"; exit 0;;
        *) echo "unknown option $1" >&2; exit 2;;
    esac
done
[ -f "$MODEL" ] || { echo "model not found: $MODEL" >&2; exit 1; }
command -v "$BENCH" >/dev/null || { echo "$BENCH not found (is bonsai-oryon installed?)" >&2; exit 1; }
mkdir -p "$OUT"; R="$OUT/report.md"

{
echo "# Bonsai Oryon benchmark report"
echo
echo "- date: $(date -Iseconds)"
echo "- host: $(hostnamectl 2>/dev/null | sed -n 's/.*Hardware Model: //p' | head -1) $(cat /sys/devices/virtual/dmi/id/product_name 2>/dev/null) $(cat /sys/devices/virtual/dmi/id/product_version 2>/dev/null)"
echo "- cpu: $(lscpu | sed -n 's/^Model name: *//p' | sort -u | tr '\n' ' ')($(nproc) cpus)"
echo "- kernel: $(uname -sr), $(. /etc/os-release; echo "$PRETTY_NAME")"
echo "- cpu features: $(grep -m1 Features /proc/cpuinfo | cut -d: -f2)"
echo "- governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo n/a)"
echo "- power: $(cat /sys/class/power_supply/*/status 2>/dev/null | tr '\n' ' ')"
echo "- memory: $(free -h | awk '/Mem:/{print $2" total, "$7" available"}')"
echo "- model: $MODEL ($(du -h "$MODEL" | cut -f1))"
echo "- bonsai-bench: $(command -v "$BENCH") / $("$BENCH" --version 2>&1 | grep -m1 -i version)"
echo "- loaded CPU backend: $("$BENCH" --version 2>&1 | grep -oE 'libggml-cpu-[^ ]+\.so' | head -1)"
echo
echo "Settings: -ngl 0 -t $THREADS -p $PP -n $TG -r $REPS (llama-bench defaults otherwise: batch 2048, ubatch 512, mmap on)."
echo
} > "$R"

run_bench() { # label binary threads
    local label="$1" bin="$2" t="$3" j="$OUT/${label}-t$3.json"
    echo ">> $label: $bin -t $t" >&2
    "$bin" -m "$MODEL" -ngl 0 -t "$t" -p "$PP" -n "$TG" -r "$REPS" -o json > "$j" 2> "$OUT/${label}-t$t.log" || { echo "| $label | $t | failed (see ${label}-t$t.log) | | |" >> "$R"; return; }
    python3 - "$j" "$label" "$t" >> "$R" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); label, t = sys.argv[2], sys.argv[3]
pp = next((r for r in d if r["n_prompt"] and not r["n_gen"]), None)
tg = next((r for r in d if r["n_gen"] and not r["n_prompt"]), None)
f = lambda r: "%.2f ± %.2f" % (r["avg_ts"], r["stddev_ts"]) if r else "n/a"
print(f"| {label} | {t} | {f(pp)} | {f(tg)} | {d[0].get('cpu_info','')} |")
PY
}

echo "## Throughput (tokens/s)" >> "$R"; echo >> "$R"
echo "| build | threads | pp$PP | tg$TG | cpu |" >> "$R"; echo "|---|---|---|---|---|" >> "$R"
run_bench bonsai-oryon "$BENCH" "$THREADS"
[ -n "$BASELINE" ] && run_bench baseline "$BASELINE" "$THREADS"
if [ $SWEEP = 1 ]; then
    for t in 2 4 6; do run_bench bonsai-oryon "$BENCH" "$t"; [ -n "$BASELINE" ] && run_bench baseline "$BASELINE" "$t"; done
fi
echo >> "$R"

if [ $MEM = 1 ] && [ -x /usr/bin/time ]; then
    echo "## Peak memory (max RSS, one tg$TG run)" >> "$R"; echo >> "$R"
    for pair in "bonsai-oryon:$BENCH" ${BASELINE:+"baseline:$BASELINE"}; do
        l=${pair%%:*}; b=${pair#*:}
        /usr/bin/time -v "$b" -m "$MODEL" -ngl 0 -t "$THREADS" -p 0 -n "$TG" -r 1 > /dev/null 2> "$OUT/$l-mem.log"
        echo "- $l: $(sed -n 's/.*Maximum resident set size (kbytes): //p' "$OUT/$l-mem.log" | awk '{printf "%.0f MiB", $1/1024}') (mmap'd weights count once resident)" >> "$R"
    done
    echo >> "$R"
elif [ $MEM = 1 ]; then
    echo "(install the 'time' package for peak-memory measurement)" >> "$R"
fi

echo "Raw data: $OUT" >> "$R"
echo; cat "$R"
