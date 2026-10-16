# Bonsai Oryon v0.2.0: ARM64 CPU inference (experimental)

CPU-only ARM64 build of the [PrismML llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp) (`prism` @ `e3de715`) with AArch64 kernels for **Ternary-Bonsai-2-9B PQ2_0**, packaged for **Debian 13 (trixie) arm64**. This release focuses on single-token decoding (tg).

> **Experimental, not yet measured on Snapdragon X.** Functional and numerical validation passed on generic ARM64 CI runners (Arm Neoverse N2). Every v0.2 performance number below comes from those runners. They do **not** predict Snapdragon X results (v0.1.0 already behaved very differently there), so please run the benchmark commands below on the laptop.

## Install / upgrade

```bash
sha256sum -c bonsai-oryon_0.2.0_arm64.deb.sha256
sudo apt install ./bonsai-oryon_0.2.0_arm64.deb          # upgrades 0.1.0 in place
```

Roll back to 0.1.0 (still available from the [v0.1.0 release](https://github.com/SatoruGoj0/Bonsai-Oryon/releases/tag/v0.1.0)):

```bash
sudo apt install --allow-downgrades ./bonsai-oryon_0.1.0_arm64.deb     # or: sudo apt remove bonsai-oryon
```

## Measure on the laptop (exact commands)

```bash
# full run, about 20 min, all at -ngl 0, pp256 + tg128, -r 3, same model file:
/usr/share/bonsai-oryon/bonsai-benchmark.sh -b <path to your previous llama-bench>
# or step by step:
/usr/lib/bonsai-oryon/bin/bonsai-diag                                                  # bandwidth vs GEMV
bonsai-bench -m ~/bonsai-models/Ternary-Bonsai-2-9B-PQ2_0.gguf -ngl 0 -t 4,6,7,8 -p 256 -n 128 -r 3
BONSAI_PQ2_REPACK=0  bonsai-bench -m ~/bonsai-models/Ternary-Bonsai-2-9B-PQ2_0.gguf -ngl 0 -t 8 -p 256 -n 128 -r 3   # v0.1 path
BONSAI_PQ2_TERNARY=1 bonsai-bench -m ~/bonsai-models/Ternary-Bonsai-2-9B-PQ2_0.gguf -ngl 0 -t 8 -p 256 -n 128 -r 3   # ternary layout
```

## What's new

- **`0005` 4-row interleaved PQ2_0 layout (default).** It is repacked at load time and the GGUF is unchanged. For single-token decoding, 4 rows share the activation loads, the row sums are reduced in vector lanes, codes decode in 4 ops per 16 bytes instead of 6, and each thread reads one contiguous weight stream. Prompt processing uses a 4×2 tile. Batched and single-token results are bit-identical.
- **`0006` lossless ternary layout (opt-in, `BONSAI_PQ2_TERNARY=1`).** The checkpoint turned out to be strictly ternary (no +2 codes), so it can be streamed at 1.75 bpw instead of 2.125, 17.6% fewer bytes per token. Decoding costs more, so this only helps where decoding is memory-bound.
- **`bonsai-diag`.** Compares DRAM bandwidth with tg-shaped GEMV throughput per thread count, which shows whether this machine is memory- or compute-bound.
- **Benchmark script.** Runs the diagnostics, a thread sweep, layout A/B, OpenMP placement variants, a comparison against your previous build, and records cpufreq/devfreq/thermal state.
- **I8MM evaluation.** I8MM was evaluated for single-token decoding and not used: with one activation column, half of every SMMLA result is wasted, so it does no better than SDOT.

## Measured on CI hardware (Neoverse N2, 4 vCPU, real 9B model; not Snapdragon X)

| build (Neoverse N2 runner, real 9B model, `-r 3`) | pp256 t=4 | tg128 t=1 | tg128 t=2 | tg128 t=4 |
|---|---|---|---|---|
| upstream kernels (e3de715 + 0003/0004) | 8.98 | 1.91 | 3.65 | 6.92 |
| v0.1 path (`BONSAI_PQ2_REPACK=0`) | 13.91 | 2.22 | 4.21 | 8.04 |
| **v0.2 4-row layout (default)** | **18.44** | **3.01** | **5.74** | **10.47** |
| v0.2 ternary layout (`BONSAI_PQ2_TERNARY=1`) | 11.88 | 1.83 | 3.56 | 6.81 |

Output compatibility against the unpatched kernels on the real model: mean KL divergence 0.000000 (max 5e-5), same top-1 token 100%, PPL 10.537824 vs 10.537794 (both 4-row layouts).

| Snapdragon X, 8 threads (user measurements) | pp256 tok/s | tg128 tok/s |
|---|---|---|
| PrismML e3de715 | 15.57 ± 3.02 | 5.02 ± 0.71 |
| Bonsai Oryon v0.1.0 | 34.82 ± 4.19 | 6.01 ± 0.57 |
| Bonsai Oryon v0.2.0 | not measured | not measured |

## Known limitations

- Not measured on Snapdragon X. Whether tg exceeds 7 tok/s there is unknown until the laptop benchmark is run.
- On Oryon, v0.1.0's pattern (fast pp, slow tg) suggests tg may be limited by memory bandwidth or synchronization rather than the kernel. In that case the default 4-row layout helps less than on N2, and `BONSAI_PQ2_TERNARY=1` may be the better choice. `bonsai-diag` tells which case applies.
- The 4-row layout copies the weights into anonymous memory at load time, and loading takes longer.
- CPU only (no Vulkan, no MTP). No embedded web UI. No HTTPS model download. Model weights are not included.

Full details are in [docs/OPTIMIZATIONS.md](https://github.com/SatoruGoj0/Bonsai-Oryon/blob/claude/wonderful-curie-my0ukq/docs/OPTIMIZATIONS.md).
