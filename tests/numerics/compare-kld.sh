#!/usr/bin/env bash
# Numerical output compatibility: full-vocabulary logits of the optimized build vs the unpatched
# pinned upstream build, on the same model and text, via llama-perplexity's KL-divergence mode.
#   compare-kld.sh <baseline llama-perplexity> <optimized llama-perplexity> <model> <text> [results.md]
# Env: BASE_BACKEND_DIR / OPT_BACKEND_DIR (build bin dirs; also used as each binary's LD_LIBRARY_PATH), CTX (128), CHUNKS (8), THREADS (2),
#      MAX_KLD (1e-4), MIN_TOP1 (99.0)
set -euo pipefail
BASE="$1"; OPT="$2"; MODEL="$3"; TEXT="$4"; RESULTS="${5:-/dev/null}"
CTX=${CTX:-128}; CHUNKS=${CHUNKS:-8}; T=${THREADS:-2}; MAX_KLD=${MAX_KLD:-1e-4}; MIN_TOP1=${MIN_TOP1:-99.0}
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT

LD_LIBRARY_PATH="${BASE_BACKEND_DIR:-${LD_LIBRARY_PATH:-}}" BONSAI_BACKEND_DIR="${BASE_BACKEND_DIR:-}" "$BASE" -m "$MODEL" -f "$TEXT" -c "$CTX" --chunks "$CHUNKS" -t "$T" -ngl 0 -b "$CTX" \
    --kl-divergence-base "$W/base.kld" > "$W/base.log" 2>&1 || { tail -20 "$W/base.log"; exit 1; }
LD_LIBRARY_PATH="${OPT_BACKEND_DIR:-${LD_LIBRARY_PATH:-}}" BONSAI_BACKEND_DIR="${OPT_BACKEND_DIR:-}" "$OPT" -m "$MODEL" -c "$CTX" --chunks "$CHUNKS" -t "$T" -ngl 0 -b "$CTX" \
    --kl-divergence-base "$W/base.kld" --kl-divergence > "$W/opt.log" 2>&1 || { tail -20 "$W/opt.log"; exit 1; }

ppl_base=$(grep -oE 'Final estimate: PPL = [0-9.]+' "$W/base.log" | grep -oE '[0-9.]+$' || true)
kld=$(grep -E '^Mean +KLD:' "$W/opt.log" | awk '{print $3}')
top1=$(grep -E '^Same top p:' "$W/opt.log" | awk '{print $4}')
ppl_opt=$(grep -E '^Mean PPL\(Q\) +:' "$W/opt.log" | awk '{print $4}')
ppl_ref=$(grep -E '^Mean PPL\(base\) +:' "$W/opt.log" | awk '{print $4}')
echo "baseline PPL: ${ppl_base:-$ppl_ref}   optimized PPL: $ppl_opt   mean KLD: $kld   same top-1: $top1 %"
sed -n '/====== Perplexity statistics/,$p' "$W/opt.log" | head -40

{
  echo "| metric | value | threshold |"; echo "|---|---|---|"
  echo "| PPL baseline (unpatched e3de715) | ${ppl_ref:-$ppl_base} | |"
  echo "| PPL bonsai-oryon | $ppl_opt | |"
  echo "| mean KL divergence | $kld | < $MAX_KLD |"
  echo "| same top-1 token | $top1 % | >= $MIN_TOP1 % |"
} >> "$RESULTS"

python3 - "$kld" "$top1" "$MAX_KLD" "$MIN_TOP1" <<'PY'
import sys
kld, top1, mk, mt = (float(v) for v in sys.argv[1:5])
ok = kld < mk and top1 >= mt
print("numerical compatibility:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
