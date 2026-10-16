#!/usr/bin/env bash
# Numerical output compatibility on the real model, run on the laptop:
# KL divergence of bonsai-oryon's full-vocabulary logits against your existing llama.cpp build.
#   compare-output.sh <your llama-perplexity> [model] [text-file]
# Uses /usr/lib/bonsai-oryon/bin/bonsai-perplexity from the package. ~2-5 minutes on 8 threads.
set -euo pipefail
BASE="$1"; MODEL="${2:-$HOME/bonsai-models/Ternary-Bonsai-2-9B-PQ2_0.gguf}"
TEXT="${3:-/usr/share/doc/bonsai-oryon/licenses/llama.cpp-AUTHORS}"
OPT=/usr/lib/bonsai-oryon/bin/bonsai-perplexity
CTX=${CTX:-256} CHUNKS=${CHUNKS:-4} THREADS=${THREADS:-8} MAX_KLD=${MAX_KLD:-0.001} MIN_TOP1=${MIN_TOP1:-98} \
    /usr/share/bonsai-oryon/compare-kld.sh "$BASE" "$OPT" "$MODEL" "$TEXT"
