#!/usr/bin/env bash
# bonsai-tune: measure this machine once and write the tuned profile used by `--oryon-profile auto` (the default
# of bonsai-cli / bonsai-server). No root needed; nothing system-wide is changed.
#
#   bonsai-tune [-m MODEL] [--rounds N] [--dry-run] [--print]
#
# It runs short llama-bench measurements (tg32 for generation, pp64 for prompt processing) for a set of thread
# counts and per-cluster CPU masks, interleaved over N rounds (default 3) so heat/load drift does not favour one
# candidate, and picks
#   * generation threads: the fastest median tg; on a tie (within 3%) the configuration with fewer threads, which
#     leaves more cores to the desktop;
#   * batch threads: the fastest median pp among the same candidates;
# then writes ${XDG_CONFIG_HOME:-~/.config}/bonsai-oryon/profile (key=value). Delete that file to go back to the
# built-in heuristic. Run it on AC power, with the machine in the state you normally use it in. Takes ~3-6 min.
set -uo pipefail
MODEL="${HOME}/bonsai-models/Ternary-Bonsai-2-9B-PQ2_0.gguf"; ROUNDS=3; DRY=0
BENCH="${BONSAI_BENCH:-bonsai-bench}"
while [ $# -gt 0 ]; do
    case "$1" in
        -m) MODEL="$2"; shift 2;;
        --rounds) ROUNDS="$2"; shift 2;;
        --dry-run) DRY=1; shift;;
        --print) cat "${XDG_CONFIG_HOME:-$HOME/.config}/bonsai-oryon/profile" 2>/dev/null || echo "(no tuned profile)"; exit 0;;
        -h|--help) sed -n '2,17p' "$0"; exit 0;;
        *) echo "unknown option $1" >&2; exit 2;;
    esac
done
[ -f "$MODEL" ] || { echo "model not found: $MODEL" >&2; exit 1; }
command -v "$BENCH" >/dev/null || { echo "$BENCH not found" >&2; exit 1; }
NCPU=$(python3 -c 'import os; print(len(os.sched_getaffinity(0)))' 2>/dev/null || nproc)
CFG="${XDG_CONFIG_HOME:-$HOME/.config}/bonsai-oryon/profile"
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

# candidates: name|bench args
CAND=()
for t in 2 3 4 5 6 8 10 12; do [ $t -le "$NCPU" ] && CAND+=("t$t|-t $t"); done
[ "$NCPU" -gt 12 ] && CAND+=("t$NCPU|-t $NCPU")
# one cluster pinned (cpufreq policies = clusters on Snapdragon X), if there is more than one cluster
nclu=$(ls -d /sys/devices/system/cpu/cpufreq/policy* 2>/dev/null | wc -l)
if [ "$nclu" -gt 1 ]; then
    for p in /sys/devices/system/cpu/cpufreq/policy*; do
        m=0; n=0; for c in $(cat "$p/related_cpus"); do m=$((m | (1 << c))); n=$((n + 1)); done
        [ $n -ge 2 ] && CAND+=("cluster-${p##*policy}|-t $n -C $(printf '%x' $m) --cpu-strict 1")
    done
fi
echo "bonsai-tune: $NCPU CPUs, ${#CAND[@]} candidates x $ROUNDS rounds" >&2
"$BENCH" -m "$MODEL" -ngl 0 -t 4 -p 0 -n 4 -r 1 > /dev/null 2>&1    # warm the page cache

for r in $(seq 1 "$ROUNDS"); do
    for i in $(for i in "${!CAND[@]}"; do echo "$RANDOM $i"; done | sort -n | cut -d' ' -f2); do
        IFS='|' read -r name args <<< "${CAND[$i]}"
        echo "  round $r: $name" >&2
        # shellcheck disable=SC2086
        "$BENCH" -m "$MODEL" -ngl 0 $args -p 64 -n 32 -r 1 -o csv 2>/dev/null | python3 -c '
import csv, sys
for row in csv.DictReader(sys.stdin):
    print(("pp" if row["n_gen"] == "0" else "tg"), row["avg_ts"])' | sed "s/^/$name /" >> "$TMP/res"
        sleep 3
    done
done

python3 - "$TMP/res" "$NCPU" "$CFG" "$DRY" "${CAND[@]}" <<'PY'
import os, statistics, sys
res, ncpu, cfg, dry, cands = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4] == "1", sys.argv[5:]
args = dict(c.split("|", 1) for c in cands)
d = {}
for line in open(res):
    name, test, v = line.split()
    d.setdefault(name, {}).setdefault(test, []).append(float(v))
if not d:
    sys.exit("no successful measurement")
def nthreads(n):
    a = args[n].split()
    return int(a[a.index("-t") + 1])
def med(n, t):
    return statistics.median(d[n][t]) if t in d.get(n, {}) else 0.0
print("\n| candidate | threads | tg32 median | pp64 median |\n|---|---|---|---|")
for n in sorted(d, key=lambda n: -med(n, "tg")):
    print(f"| {n} | {nthreads(n)} | {med(n, 'tg'):.2f} | {med(n, 'pp'):.2f} |")
best_tg = max(med(n, "tg") for n in d)
gen = min((n for n in d if med(n, "tg") >= 0.97 * best_tg), key=lambda n: (nthreads(n), -med(n, "tg")))
bat = max(d, key=lambda n: med(n, "pp"))
a = args[gen].split()
lines = ["# written by bonsai-tune; delete this file to use the built-in heuristic",
         f"ncpu={ncpu}", f"threads={nthreads(gen)}", f"threads_batch={nthreads(bat)}", "poll=50"]
if "-C" in a:
    lines += [f"cpu_mask={a[a.index('-C') + 1]}", "cpu_strict=1"]
print(f"\nselected: generation {gen} ({med(gen, 'tg'):.2f} tok/s), batch {nthreads(bat)} threads ({med(bat, 'pp'):.2f} tok/s)")
if dry:
    print("(dry run) would write", cfg, ":\n" + "\n".join(lines))
else:
    os.makedirs(os.path.dirname(cfg), exist_ok=True)
    open(cfg, "w").write("\n".join(lines) + "\n")
    print("wrote", cfg)
PY
