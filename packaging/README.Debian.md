# bonsai-oryon (Debian package)

CPU-only ARM64 build of PrismML llama.cpp (e3de715) with Bonsai-Oryon PQ2_0 kernels.
Project page: https://github.com/SatoruGoj0/Bonsai-Oryon

## Commands

    bonsai-cli    -m ~/bonsai-models/Ternary-Bonsai-2-9B-PQ2_0.gguf -ngl 0 -c 4096
    bonsai-server -m ~/bonsai-models/Ternary-Bonsai-2-9B-PQ2_0.gguf --alias bonsai-9b \
                  --host 127.0.0.1 --port 8080 -ngl 0 -c 8192
    bonsai-bench  -m ~/bonsai-models/Ternary-Bonsai-2-9B-PQ2_0.gguf -t 4 -ngl 0 -p 256 -n 128 -r 3
    bonsai-tune   [-m MODEL]      # measure once, write ~/.config/bonsai-oryon/profile

Without -t, cli and server take threads from --oryon-profile (default auto: the bonsai-tune
profile if present, else balanced = 4 threads on an 8-core Snapdragon X). The choice is logged.
Profiles: balanced, throughput, low-power, auto, none (env BONSAI_ORYON_PROFILE).

OpenAI-compatible API base URL: http://127.0.0.1:8080/v1 (model id = --alias).
The server binds to 127.0.0.1 unless --host says otherwise. Don't expose it publicly without
--api-key and a firewall.

## Switches

    BONSAI_BACKEND_DIR=/usr/lib/bonsai-oryon/omp   OpenMP threading as in v0.2
    BONSAI_BARRIER_SPIN=N                          spin iterations before yielding (default 4096)
    BONSAI_TG_NOBARRIER=1                          experimental barrier-free single-token matmul
    BONSAI_PQ2_REPACK=0 / BONSAI_PQ2_TERNARY=1     v0.1 kernels / ternary layout
    BONSAI_PROFILE=1                               print where graph time goes at exit

## Benchmarks and comparisons

    /usr/share/bonsai-oryon/bonsai-benchmark.sh --short|--full [-m MODEL] [--release-deb OLD.deb] [-b <llama-bench>]
    /usr/share/bonsai-oryon/bonsai-release-run.sh OLD.deb [--tool bonsai-bench] ARGS   # run an old release, not installed
    /usr/share/bonsai-oryon/compare-output.sh <your llama-perplexity>
    /usr/lib/bonsai-oryon/bin/bonsai-diag

## Files

    /usr/bin/bonsai-{cli,server,bench,tune}
    /usr/lib/bonsai-oryon/        private libraries + CPU variant modules (chosen at run time)
    /usr/lib/bonsai-oryon/omp/    the same CPU modules built with OpenMP
    /usr/share/bonsai-oryon/      scripts
    /usr/share/doc/bonsai-oryon/  this file, changelog, copyright, licenses, PROVENANCE.txt

The package doesn't conflict with other llama.cpp installations, ships no services and runs no
maintainer scripts. bonsai-tune writes only to the calling user's ~/.config.

## Removal

    sudo apt remove bonsai-oryon
    rm -f ~/.config/bonsai-oryon/profile

## Known limitations

Experimental, not yet measured on Snapdragon X. CPU only (no Vulkan/GPU). No embedded web UI.
No HTTPS model download. Model weights are not included. With mmap, peak RSS includes the mapped
model file in addition to the repacked weights; --no-mmap avoids that.
