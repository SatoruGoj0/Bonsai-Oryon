#!/usr/bin/env bash
# Bonsai Oryon benchmark (v0.3) - interleaved, condition-tracked A/B benchmark. No root needed.
#
#   bonsai-benchmark.sh [--short | --full] [-m MODEL] [-o OUTDIR] [-r ROUNDS] [--cooldown SEC]
#                       [-b PREVIOUS_LLAMA_BENCH] [--release-deb OLD.deb]...
#
# Method
#   * every configuration is run once per round, rounds are repeated (-r, default 5 short / 7 full) and the
#     order of configurations is shuffled in every round, so slow drifts (heat, battery, background load) hit
#     all configurations alike instead of biasing whichever ran last;
#   * llama-bench excludes model loading from its timing and does its own warm-up run; one extra discarded
#     warm-up round fills the page cache; a cooldown (default 10 s) separates runs;
#   * before every run: max thermal-zone temperature, CPU frequency caps, 1-min load, available memory and
#     AC/battery state are recorded; runs taken while the machine was clearly hotter/throttled/busier than at
#     the start, or after an AC<->battery change, are flagged and reported separately;
#   * results: mean, standard deviation and 95% confidence interval per configuration and test, plus the
#     ratio against the reference configuration (v0.2-equivalent);
#   * model load time and peak RSS are measured separately (repacked vs plain layout, mmap vs no-mmap).
#
# Configurations (all CPU only, -ngl 0, same model file):
#   short: v0.3 balanced profile, v0.3 at 8 threads, v0.2-equivalent (OpenMP modules, 4-row kernel, 4 thr),
#          v0.1-equivalent (OpenMP modules, plain kernel, 4 thr)
#   full : + threadpool at 2/3/5/6 threads, barrier spin variants, poll 0/100, per-cluster pinning,
#          bounded OpenMP active wait, ternary layout, throughput profile
#   --release-deb: runs the bonsai-bench of an older release .deb extracted to a cache directory (nothing
#          installed is touched); -b: your previous llama.cpp llama-bench.
set -uo pipefail
MODE=short; MODEL="${HOME}/bonsai-models/Ternary-Bonsai-2-9B-PQ2_0.gguf"; OUT="./bonsai-bench-$(date +%Y%m%d-%H%M%S)"
ROUNDS=""; COOLDOWN=10; BASELINE=""; RELEASE_DEBS=()
BENCH="${BONSAI_BENCH:-bonsai-bench}"
LIBDIR="${BONSAI_LIBDIR:-/usr/lib/bonsai-oryon}"
SHARE="$(cd "$(dirname "$0")" && pwd)"
while [ $# -gt 0 ]; do
    case "$1" in
        --short) MODE=short; shift;;
        --full) MODE=full; shift;;
        --quick) MODE=short; shift;;                 # v0.2 option name
        -m) MODEL="$2"; shift 2;;
        -o) OUT="$2"; shift 2;;
        -r) ROUNDS="$2"; shift 2;;
        --cooldown) COOLDOWN="$2"; shift 2;;
        -b) BASELINE="$2"; shift 2;;
        --release-deb) RELEASE_DEBS+=("$2"); shift 2;;
        --sweep|--no-mem) shift;;
        -h|--help) sed -n '2,30p' "$0"; exit 0;;
        *) echo "unknown option $1" >&2; exit 2;;
    esac
done
[ -n "$ROUNDS" ] || { [ $MODE = full ] && ROUNDS=7 || ROUNDS=5; }
[ -f "$MODEL" ] || { echo "model not found: $MODEL" >&2; exit 1; }
command -v "$BENCH" >/dev/null || { echo "$BENCH not found (is bonsai-oryon installed?)" >&2; exit 1; }
command -v python3 >/dev/null || { echo "python3 is required" >&2; exit 1; }
mkdir -p "$OUT"; R="$OUT/report.md"; CSV="$OUT/runs.csv"
NCPU=$(nproc)
if [ $MODE = full ]; then PP=256; TG=128; else PP=128; TG=64; fi

# ---------------------------------------------------------------------------------------------------------
# system state
max_temp_c() { local m=0 t; for z in /sys/class/thermal/thermal_zone*/temp; do t=$(cat "$z" 2>/dev/null || echo 0); [ "${t:-0}" -gt "$m" ] && m=$t; done; echo $((m / 1000)); }
freq_cap_mhz() { local s=0 n=0 f; for c in /sys/devices/system/cpu/cpufreq/policy*/scaling_max_freq; do f=$(cat "$c" 2>/dev/null) || continue; s=$((s + f)); n=$((n + 1)); done; [ $n -gt 0 ] && echo $((s / n / 1000)) || echo 0; }
freq_cur_mhz() { local s=0 n=0 f; for c in /sys/devices/system/cpu/cpufreq/policy*/scaling_cur_freq; do f=$(cat "$c" 2>/dev/null) || continue; s=$((s + f)); n=$((n + 1)); done; [ $n -gt 0 ] && echo $((s / n / 1000)) || echo 0; }
loadavg1() { cut -d' ' -f1 /proc/loadavg; }
mem_avail_mb() { awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }
ac_state() {
    local s=""
    for p in /sys/class/power_supply/*; do
        [ -f "$p/type" ] || continue
        case "$(cat "$p/type")" in
            Mains|USB) [ "$(cat "$p/online" 2>/dev/null)" = 1 ] && s="AC";;
            Battery) [ -z "$s" ] && s="battery($(cat "$p/capacity" 2>/dev/null)%)";;
        esac
    done
    echo "${s:-unknown}"
}
# clusters = cpufreq policies (one per cluster on Snapdragon X); prints "hexmask cpulist" per cluster
clusters() {
    for p in /sys/devices/system/cpu/cpufreq/policy*; do
        [ -f "$p/related_cpus" ] || continue
        local m=0; for c in $(cat "$p/related_cpus"); do m=$((m | (1 << c))); done
        printf '%x %s\n' "$m" "$(cat "$p/related_cpus" | tr ' ' ',')"
    done
}

# ---------------------------------------------------------------------------------------------------------
# configurations: name|env (space separated VAR=VAL, or -)|bench args (without -m/-p/-n/-r)|binary (default $BENCH)
CONF=()
add() { CONF+=("$1|$2|$3|${4:-$BENCH}"); }
add "v0.3-balanced"   "-" "--oryon-profile balanced"
add "v0.3-t8"         "-" "-t 8"
add "v0.2-equiv-t4"   "BONSAI_BACKEND_DIR=$LIBDIR/omp" "-t 4"
add "v0.1-equiv-t4"   "BONSAI_BACKEND_DIR=$LIBDIR/omp BONSAI_PQ2_REPACK=0" "-t 4"
REF="v0.2-equiv-t4"
if [ $MODE = full ]; then
    for t in 2 3 5 6; do [ $t -le $NCPU ] && add "v0.3-t$t" "-" "-t $t"; done
    add "v0.3-t4-purespin"   "BONSAI_BARRIER_SPIN=-1" "-t 4"
    add "v0.3-t4-yield"      "BONSAI_BARRIER_SPIN=0"  "-t 4"
    add "v0.3-t4-poll0"      "-" "-t 4 --poll 0"
    add "v0.3-t4-poll100"    "-" "-t 4 --poll 100"
    add "v0.3-ternary-t4"    "BONSAI_PQ2_TERNARY=1" "-t 4"
    add "v0.2-equiv-t4-activewait" "BONSAI_BACKEND_DIR=$LIBDIR/omp OMP_WAIT_POLICY=active GOMP_SPINCOUNT=20000000" "-t 4"
    i=0
    while read -r mask list; do
        n=$(echo "$list" | tr ',' '\n' | wc -l)
        add "v0.3-cluster$i-t$n" "-" "-t $n -C $mask --cpu-strict 1"
        i=$((i + 1))
    done < <(clusters)
fi
for deb in "${RELEASE_DEBS[@]:-}"; do
    [ -n "$deb" ] || continue
    ver=$(dpkg-deb -f "$deb" Version 2>/dev/null) || { echo "cannot read $deb" >&2; continue; }
    add "release-$ver-t4" "-" "-t 4" "$SHARE/bonsai-release-run.sh $deb"
done
[ -n "$BASELINE" ] && add "previous-build-t4" "-" "-t 4" "$BASELINE"

# ---------------------------------------------------------------------------------------------------------
{
echo "# Bonsai Oryon benchmark report ($MODE)"
echo
echo "- date: $(date -Iseconds)"
echo "- machine: $(cat /sys/devices/virtual/dmi/id/sys_vendor 2>/dev/null) $(cat /sys/devices/virtual/dmi/id/product_name 2>/dev/null) $(cat /sys/devices/virtual/dmi/id/product_version 2>/dev/null)"
echo "- cpu: $(lscpu | sed -n 's/^Model name: *//p' | sort -u | tr '\n' ' ')($NCPU cpus); clusters: $(clusters | awk '{printf "[%s] ", $2}')"
echo "- cpu features: $(grep -m1 Features /proc/cpuinfo | cut -d: -f2)"
echo "- kernel: $(uname -r); os: $(. /etc/os-release; echo "$PRETTY_NAME")"
echo "- package: bonsai-oryon $(dpkg-query -W -f='${Version}' bonsai-oryon 2>/dev/null || echo '(not installed via dpkg)'); bench: $("$BENCH" --version 2>&1 | grep -m1 -i version)"
echo "- power: $(ac_state); platform profile: $(cat /sys/firmware/acpi/platform_profile 2>/dev/null || echo n/a); governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo n/a)"
echo "- memory available: $(mem_avail_mb) MiB of $(free -m | awk '/Mem:/{print $2}') MiB; load average: $(cut -d' ' -f1-3 /proc/loadavg)"
echo "- start temperature $(max_temp_c) C, CPU frequency cap avg $(freq_cap_mhz) MHz (hardware max $(cat /sys/devices/system/cpu/cpufreq/policy*/cpuinfo_max_freq 2>/dev/null | sort -n | tail -1 | awk '{print int($1/1000)}') MHz)"
echo "- model: $MODEL ($(du -h "$MODEL" | cut -f1))"
echo "- method: $ROUNDS rounds x ${#CONF[@]} configurations, shuffled order per round, pp$PP + tg$TG per run, ${COOLDOWN}s cooldown, 1 discarded warm-up round"
echo
} > "$R"

# load time and peak memory (separate from inference timing)
if [ -x /usr/bin/time ]; then
    { echo "## Model load time and peak RSS (one 1-token run each)"; echo; echo "| layout | load mode | wall s (load + 1 token) | peak RSS MiB |"; echo "|---|---|---|---|"; } >> "$R"
    for v in "4-row|-|mmap" "4-row|-|no-mmap" "plain|BONSAI_PQ2_REPACK=0|mmap"; do
        IFS='|' read -r lay e lm <<< "$v"
        envs=(); [ "$e" != - ] && envs=($e)
        mm=1; [ "$lm" = no-mmap ] && mm=0
        env "${envs[@]}" /usr/bin/time -f '%e %M' -o "$OUT/load.tmp" "$BENCH" -m "$MODEL" -ngl 0 -t 4 -p 0 -n 1 -r 1 --no-warmup -mmp $mm > /dev/null 2>&1
        read -r wall rss < "$OUT/load.tmp" || true
        echo "| $lay | $lm | ${wall:-?} | $(( ${rss:-0} / 1024 )) |" >> "$R"
        sleep 2
    done
    echo >> "$R"
fi

echo "round,order,config,test,tps,temp_c,freq_cap_mhz,freq_cur_mhz,load1,mem_avail_mb,power" > "$CSV"
run_one() { # round order idx
    local round=$1 order=$2 idx=$3
    IFS='|' read -r name envs args bin <<< "${CONF[$idx]}"
    local e=(); [ "$envs" != "-" ] && e=($envs)
    local st_temp st_cap st_cur st_load st_mem st_pw
    st_temp=$(max_temp_c); st_cap=$(freq_cap_mhz); st_cur=$(freq_cur_mhz); st_load=$(loadavg1); st_mem=$(mem_avail_mb); st_pw=$(ac_state)
    local j="$OUT/raw/${name}-r${round}.json"
    # shellcheck disable=SC2086
    if env "${e[@]}" $bin -m "$MODEL" -ngl 0 $args -p "$PP" -n "$TG" -r 1 -o json > "$j" 2> "$OUT/raw/${name}-r${round}.log"; then
        python3 - "$j" <<'PY' | while read -r test tps; do echo "$round,$order,$name,$test,$tps,$st_temp,$st_cap,$st_cur,$st_load,$st_mem,$st_pw" >> "$CSV"; done
import json, sys
for r in json.load(open(sys.argv[1])):
    print("pp" if r["n_prompt"] and not r["n_gen"] else "tg", "%.4f" % r["avg_ts"])
PY
    else
        echo "  ! $name failed (see raw/${name}-r${round}.log)" >&2
    fi
}

mkdir -p "$OUT/raw"
echo ">> warm-up round (discarded)" >&2
for idx in "${!CONF[@]}"; do run_one 0 0 "$idx" > /dev/null; sleep 2; done
sed -i '2,$d' "$CSV"
for round in $(seq 1 "$ROUNDS"); do
    order=$(for idx in "${!CONF[@]}"; do echo "$RANDOM $idx"; done | sort -n | cut -d' ' -f2)
    k=0
    for idx in $order; do
        k=$((k + 1))
        echo ">> round $round/$ROUNDS [$k/${#CONF[@]}] ${CONF[$idx]%%|*}" >&2
        run_one "$round" "$k" "$idx"
        sleep "$COOLDOWN"
    done
done

python3 "$SHARE/bonsai-bench-report.py" "$CSV" "$REF" "$NCPU" >> "$R"
echo; cat "$R"
echo "Raw data: $OUT (runs.csv, raw/)"
