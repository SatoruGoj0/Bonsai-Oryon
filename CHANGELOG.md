# Changelog

## 0.3.0 (2026-10-16), experimental

Focus: threading and synchronization of single-token decoding on Snapdragon X, measurement tooling.

- **Threadpool instead of OpenMP (default).** The packaged CPU modules are built with `GGML_OPENMP=OFF`, so
  ggml's persistent threadpool is used: workers spin inside a graph and sleep (after polling, `--poll`) between
  graphs, instead of libgomp's spin-then-futex barrier on every node. The v0.2 OpenMP modules (same kernels)
  are still shipped in `/usr/lib/bonsai-oryon/omp` (`BONSAI_BACKEND_DIR=/usr/lib/bonsai-oryon/omp`).
- **0007** graph profiler (`BONSAI_PROFILE=1`: per-op time, compute vs barrier wait, wake-up latency) and a
  bounded barrier spin (`BONSAI_BARRIER_SPIN`, default 65536 spins then `sched_yield()`; `-1` = upstream pure
  spin). No unbounded busy-wait is enabled by default.
- **0008** `--oryon-profile balanced|throughput|low-power|auto|none` for cli, server and bench. `auto` (default
  for cli/server) uses the per-user profile written by the new `bonsai-tune`, else `balanced`
  (4 threads on 8-core Snapdragon X). Explicit `-t/-tb/--poll/-C` always win.
- **0009** opt-in barrier-free single-token matmul (`BONSAI_TG_NOBARRIER=1`), bit-identical results.
- `bonsai-benchmark.sh` rewritten: shuffled interleaved rounds, warm-up, cooldowns, per-run temperature /
  frequency cap / load / power-source recording with drift flags, mean/sd/95% CI, separate load-time/RSS
  measurement, `--short`/`--full`, `--release-deb` to compare older releases without installing them.
- New tests: bit-identical logits across thread counts, threading backends, barrier policies and the
  barrier-free path; 2048-token context vs upstream; profiles; profiler; real-model server tool calling.

## 0.2.0 (2026-10-16), experimental

Focus: single-token decoding (tg). Upstream is unchanged (PrismML `prism` @ `e3de715`); patches 0001–0004
are unchanged from 0.1.0.

- `0005` 4-row interleaved PQ2_0 layout for AArch64 (repacked at load time; the GGUF file is untouched).
  Single-token GEMV shares the activation loads across 4 rows, reduces the 4 row sums into vector lanes, and
  decodes codes with 4 ops per 16 bytes instead of 6. Each thread streams one contiguous array. Prompt
  processing uses a 4-row × 2-token tile. Batched and single-token results are bit-identical.
  `BONSAI_PQ2_REPACK=0` restores the 0.1.0 kernel path.
- `0006` opt-in lossless ternary 4-row layout (`BONSAI_PQ2_TERNARY=1`): 1.75 bpw instead of 2.125 for
  ternary tensors (all of Ternary-Bonsai-2-9B). It moves 17.6% fewer bytes per token but costs more decode
  ALU, so it helps only where decoding is memory-bound.
- `bonsai-diag` (`/usr/lib/bonsai-oryon/bin/bonsai-diag`): DRAM bandwidth vs tg-shaped GEMV per thread count.
- `bonsai-benchmark.sh`: diagnostics, thread sweep, layout A/B (v0.1 path / 4-row / ternary), OpenMP
  placement variants, CPU/devfreq/thermal snapshots.

## 0.1.0 (2026-10-16), experimental

First release. Upstream is PrismML llama.cpp `prism` @ `e3de715222b2667c54a267441159a5729f11155e`.

- `0001` AArch64 NEON PQ2_0 × Q8_K kernel: de-interleaved decode (VLD4 + shift/mask), the −1 offset
  folded out through Q8_K `bsums`, 4 blocks per float vector, and a new bit-consistent 2×2 tile for
  prompt processing.
- `0002` NEON `quantize_row_q8_K`, bit-identical to the reference, 1.26× faster on Neoverse N2.
- `0003` `armv8.6_nosve` CPU variant (DOTPROD+FP16+I8MM, no SVE), which Snapdragon X now selects.
- `0004` backends load only from `/usr/lib/bonsai-oryon`.
- Debian package `bonsai-oryon_0.1.0_arm64.deb`: `bonsai-cli`, `bonsai-server`, `bonsai-bench`, with
  private libraries, no services and no maintainer scripts.
- Test suite: kernel unit tests on three ISA levels, upstream ggml PQ2_0 tests, CLI/bench/server
  OpenAI-API functional tests, KL-divergence comparison against unpatched kernels, and package
  install/remove in a clean Debian 13 container.
- Not yet measured on Snapdragon X hardware.
