# bonsai-oryon (Debian package)

CPU-only ARM64 build of PrismML llama.cpp (e3de715) with Bonsai-Oryon PQ2_0 kernels.
Project page: https://github.com/SatoruGoj0/Bonsai-Oryon

## Commands

    bonsai-cli    -m ~/bonsai-models/Ternary-Bonsai-2-9B-PQ2_0.gguf -t 8 -ngl 0 -c 4096
    bonsai-server -m ~/bonsai-models/Ternary-Bonsai-2-9B-PQ2_0.gguf --alias bonsai-9b \
                  --host 127.0.0.1 --port 8080 -t 8 -ngl 0 -c 8192
    bonsai-bench  -m ~/bonsai-models/Ternary-Bonsai-2-9B-PQ2_0.gguf -t 8 -ngl 0 -p 256 -n 128 -r 3

OpenAI-compatible API base URL: http://127.0.0.1:8080/v1 (model id = --alias).
The server binds to 127.0.0.1 unless --host says otherwise. Don't expose it publicly without
--api-key and a firewall.

## Benchmarks and comparisons

    /usr/share/bonsai-oryon/bonsai-benchmark.sh [--sweep] [-b <your llama-bench>]
    /usr/share/bonsai-oryon/compare-output.sh <your llama-perplexity>

## Files

    /usr/bin/bonsai-{cli,server,bench}
    /usr/lib/bonsai-oryon/        private libraries + CPU variant modules (chosen at run time)
    /usr/share/bonsai-oryon/      scripts
    /usr/share/doc/bonsai-oryon/  this file, changelog, copyright, licenses, PROVENANCE.txt

The package doesn't conflict with other llama.cpp installations, ships no services and runs no
maintainer scripts.

## Removal

    sudo apt remove bonsai-oryon

## Known limitations

Experimental. CPU only (no Vulkan/GPU). No embedded web UI. No HTTPS model download. Model weights
are not included.
