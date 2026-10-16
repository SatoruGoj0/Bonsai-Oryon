#!/usr/bin/env bash
# Multithreading correctness: the full-vocabulary logits must be BIT-IDENTICAL for every thread count, for the
# threadpool and the OpenMP CPU modules, for every barrier policy and for the opt-in barrier-free decode path.
#   run-threading.sh <llama-perplexity> <model> <text> [results.md]
# Env: LIBDIR  directory with the libggml-cpu-*.so modules (OpenMP modules in $LIBDIR/omp); default
#              /usr/lib/bonsai-oryon. Build trees: also put LIBDIR on LD_LIBRARY_PATH.
#      CTX (128), CHUNKS (2)
# Two groups: batched prompt evaluation (-b CTX) and token-by-token decoding (-b 1, the tg code path).
set -uo pipefail
PPL="$1"; MODEL="$2"; TEXT="$3"; RESULTS="${4:-/dev/null}"
LIBDIR="${LIBDIR:-/usr/lib/bonsai-oryon}"; CTX=${CTX:-128}; CHUNKS=${CHUNKS:-2}
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
NCPU=$(nproc)
pass=0; fail=0
echo "| group | configuration | result |" >> "$RESULTS"; echo "|---|---|---|" >> "$RESULTS"

run() { # file batch threads env...
    local out=$1 b=$2 t=$3; shift 3
    env BONSAI_BACKEND_DIR="$LIBDIR" "$@" "$PPL" -m "$MODEL" -f "$TEXT" -c "$CTX" --chunks "$CHUNKS" -b "$b" -ub "$b" \
        -t "$t" -ngl 0 --kl-divergence-base "$out" > "$out.log" 2>&1
}

for group in "batched:$CTX" "decode:1"; do
    g=${group%%:*}; b=${group#*:}
    run "$W/$g-ref" "$b" 1 || { echo "FAIL $g reference run"; tail -20 "$W/$g-ref.log"; fail=$((fail+1)); continue; }
    ppl=$(grep -oE 'Final estimate: PPL = [0-9.]+' "$W/$g-ref.log" | grep -oE '[0-9.]+$')
    echo "| $g | reference: threadpool, 1 thread (PPL $ppl) | - |" >> "$RESULTS"
    confs=()
    for t in 2 3 4 "$NCPU"; do confs+=("threadpool t=$t|$t|-"); done
    confs+=("threadpool t=4 BONSAI_BARRIER_SPIN=0 (yield)|4|BONSAI_BARRIER_SPIN=0"
            "threadpool t=4 BONSAI_BARRIER_SPIN=-1 (pure spin)|4|BONSAI_BARRIER_SPIN=-1"
            "threadpool t=3 BONSAI_TG_NOBARRIER=1|3|BONSAI_TG_NOBARRIER=1"
            "threadpool t=4 BONSAI_TG_NOBARRIER=1|4|BONSAI_TG_NOBARRIER=1"
            "threadpool t=4 BONSAI_PQ2_TERNARY=1|4|BONSAI_PQ2_TERNARY=1")
    if ls "$LIBDIR"/omp/libggml-cpu-*.so > /dev/null 2>&1; then
        confs+=("OpenMP t=1|1|BONSAI_BACKEND_DIR=$LIBDIR/omp" "OpenMP t=3|3|BONSAI_BACKEND_DIR=$LIBDIR/omp"
                "OpenMP t=4|4|BONSAI_BACKEND_DIR=$LIBDIR/omp")
    else
        echo "| $g | OpenMP modules | not present in $LIBDIR/omp, skipped |" >> "$RESULTS"
    fi
    i=0
    for c in "${confs[@]}"; do
        IFS='|' read -r name t e <<< "$c"
        envs=(); [ "$e" != - ] && envs=($e)
        i=$((i + 1))
        if run "$W/$g-$i" "$b" "$t" "${envs[@]}" && cmp -s "$W/$g-ref" "$W/$g-$i"; then
            echo "PASS  $g: $name"; echo "| $g | $name | bit-identical |" >> "$RESULTS"; pass=$((pass+1))
        else
            echo "FAIL  $g: $name"; tail -5 "$W/$g-$i.log"
            echo "| $g | $name | **differs / failed** |" >> "$RESULTS"; fail=$((fail+1))
        fi
    done
done
echo "threading: $pass bit-identical, $fail failed"
echo; echo "**threading: $pass bit-identical, $fail failed**" >> "$RESULTS"
[ $fail -eq 0 ]
