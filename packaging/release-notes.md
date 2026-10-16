# Bonsai Oryon v0.3.0: ARM64 CPU inference (experimental)

CPU-only ARM64 build of the [PrismML llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp) (`prism` @ `e3de715`) with AArch64 kernels for **Ternary-Bonsai-2-9B PQ2_0**, packaged for **Debian 13 (trixie) arm64**. This release is about threading and synchronization (Snapdragon X decoded at 8.40 tok/s with 4 threads but 3.04 with 8 in v0.2), runtime profiles and measurement tooling. The kernels are those of v0.2.

> **Experimental, not yet measured on Snapdragon X.** Functional, numerical and multithreading validation passed on generic ARM64 CI runners (Arm Neoverse N2). Every v0.3 performance number below comes from those runners and does **not** predict Snapdragon X. Please run the commands below on the laptop.

## Install / upgrade

```bash
sha256sum -c bonsai-oryon_0.3.0_arm64.deb.sha256
sudo apt install ./bonsai-oryon_0.3.0_arm64.deb          # upgrades 0.1.0 / 0.2.0 in place
bonsai-tune                                              # optional (~5 min): writes ~/.config/bonsai-oryon/profile
```

Roll back: `sudo apt install --allow-downgrades ./bonsai-oryon_0.2.0_arm64.deb` ([v0.2.0](https://github.com/SatoruGoj0/Bonsai-Oryon/releases/tag/v0.2.0), [v0.1.0](https://github.com/SatoruGoj0/Bonsai-Oryon/releases/tag/v0.1.0) stay available), or keep 0.3.0 and use v0.2's OpenMP threading with `BONSAI_BACKEND_DIR=/usr/lib/bonsai-oryon/omp`. Remove the tuned profile with `rm ~/.config/bonsai-oryon/profile`.

## Measure on the laptop (exact commands)

```bash
M=~/bonsai-models/Ternary-Bonsai-2-9B-PQ2_0.gguf
/usr/share/bonsai-oryon/bonsai-benchmark.sh --short -m $M      # ~10 min: v0.3 vs v0.2-/v0.1-equivalent, interleaved, with CIs
/usr/share/bonsai-oryon/bonsai-benchmark.sh --full  -m $M      # ~45 min: + 2..6 threads, barrier/poll variants, per-cluster pinning
/usr/share/bonsai-oryon/bonsai-benchmark.sh --short -m $M --release-deb bonsai-oryon_0.2.0_arm64.deb   # the real v0.2 package, not installed
bonsai-tune -m $M                                              # pick and store the best thread setup
BONSAI_PROFILE=1 bonsai-bench -m $M -ngl 0 -t 4 -p 0 -n 64     # where decode time goes (compute vs barrier wait)
BONSAI_PROFILE=1 bonsai-bench -m $M -ngl 0 -t 8 -p 0 -n 64
bonsai-bench -m $M -ngl 0 -t 4,8 -p 256 -n 128 -r 3                                       # v0.3 threadpool
BONSAI_BACKEND_DIR=/usr/lib/bonsai-oryon/omp bonsai-bench -m $M -ngl 0 -t 4,8 -p 256 -n 128 -r 3   # v0.2 OpenMP
```

Plug in AC power for stable numbers; the report records power source, temperature, frequency caps and load for every run and flags unstable ones.

## What's new

- **ggml threadpool instead of OpenMP.** Workers spin inside a graph, poll briefly between graphs and then sleep. The v0.2 OpenMP modules are kept in `/usr/lib/bonsai-oryon/omp`.
- **`0007` bounded barrier spin + profiler.** Waiters spin at most 4096 iterations, then `sched_yield()` (`BONSAI_BARRIER_SPIN`; `-1` = upstream unbounded spin, `0` = yield at once). No unbounded busy-wait is enabled by default. `BONSAI_PROFILE=1` prints per-op time, compute vs barrier wait and wake-up latency.
- **`0008` `--oryon-profile balanced|throughput|low-power|auto|none`** for cli, server and bench. Default `auto`: the profile written by `bonsai-tune`, else `balanced` (4 threads on 8-core Snapdragon X). `-t/-tb/--poll/-C` always win. No start-up measurement.
- **`0009` opt-in barrier-free single-token matmul** (`BONSAI_TG_NOBARRIER=1`), bit-identical.
- **`bonsai-tune`**, **rewritten `bonsai-benchmark.sh`** (shuffled interleaved rounds, warm-up, cooldowns, condition tracking, drift flags, mean/sd/95% CI, separate load/RSS, `--short`/`--full`, `--release-deb`).
- **Tests:** logits byte-identical across 1/2/3/4 threads, threadpool vs OpenMP, barrier policies, barrier-free path and ternary layout (batched and token-by-token); 2048-token context vs upstream; profiles; profiler; real-model server tool calling.

## Measured on CI hardware (Neoverse N2, 4 vCPU, real 9B model; not Snapdragon X)

[model-validation run 38946610464](https://github.com/SatoruGoj0/Bonsai-Oryon/actions/runs/38946610464):

| build | pp256 t=4 | tg128 t=4 | tg64 t=8 on 4 vCPU (oversubscribed) |
|---|---|---|---|
| upstream kernels + OpenMP | 8.97 | 6.87 | 6.00 |
| v0.2 equivalent (OpenMP) | 18.39 | 10.56 | 8.56 |
| **v0.3 default** | **18.44** | **10.65** | **9.76** |
| unbounded spin (upstream threadpool behaviour) | 18.48 | 10.63 | 0.23 |

Idle machine: v0.3 = v0.2 (same kernels; threading within 1%). Oversubscribed: v0.3 keeps 94% of its speed, v0.2 82%. Numerics vs unpatched kernels: mean KLD 0.000000, top-1 100%, PPL 10.537824 vs 10.537794. Tool calling on the real model: `finish_reason=tool_calls`, `get_weather({"city":"Paris"})`.

## Known limitations

- Not measured on Snapdragon X. The 10 tok/s sustained target is reached on the N2 runner (10.65 tok/s at 4 threads), not yet shown on the laptop.
- With mmap, peak RSS is 4.4 GB (repacked copy + mapped file pages); `--no-mmap` gives 2.7 GB.
- With small `max_tokens` the model may answer only in `reasoning_content`.
- CPU only, no GPU backends, no web UI, no `-hf` downloads.
