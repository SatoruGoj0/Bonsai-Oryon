# Bonsai Oryon

**ARM64 CPU inference engine for PrismML Ternary-Bonsai (PQ2_0) models, packaged for Debian 13 on
Snapdragon X (Qualcomm Oryon) laptops.**

Bonsai Oryon is the [PrismML fork of llama.cpp](https://github.com/PrismML-Eng/llama.cpp) (branch
`prism`, pinned at commit `e3de715`), built CPU-only, plus a small set of auditable patches that speed
up the PQ2_0 path on AArch64. It ships as one `.deb` with `bonsai-cli`, `bonsai-server` and
`bonsai-bench`, so you don't have to compile anything on the laptop.

> **Status: experimental (v0.3.0).** CI checks functional and numerical correctness on generic ARM64
> servers (Arm Neoverse N2, GitHub `ubuntu-24.04-arm`). On Snapdragon X the user measured v0.1.0 and
> v0.2.0 (best tg128 8.40 tok/s with 4 threads, collapsing to 3.04 with 8 threads). v0.3.0 changes the
> threading (ggml threadpool instead of OpenMP, bounded barrier spin, `--oryon-profile` defaulting to
> 4 threads on 8-core Snapdragon X). **v0.3.0 has not been measured on Snapdragon X yet**: run
> `bonsai-tune` and [`bonsai-benchmark.sh`](bench/bonsai-benchmark.sh) on the laptop.

## Install

```bash
# download the .deb and checksum from the GitHub Release, then:
sha256sum -c bonsai-oryon_0.3.0_arm64.deb.sha256
sudo apt install ./bonsai-oryon_0.3.0_arm64.deb      # upgrades 0.1.0 / 0.2.0 in place
bonsai-tune                                          # optional, ~5 min: measure and store the best thread setup
```

The package installs only these paths:

| path | contents |
|---|---|
| `/usr/bin/bonsai-cli`, `/usr/bin/bonsai-server`, `/usr/bin/bonsai-bench` | executables (renamed `llama-cli`, `llama-server`, `llama-bench`) |
| `/usr/bin/bonsai-tune` | measures thread setups once and writes `~/.config/bonsai-oryon/profile` (per user, no root) |
| `/usr/lib/bonsai-oryon/` | private `libllama`, `libggml*`, `libmtmd` and one `libggml-cpu-<variant>.so` per ARM feature level (ggml threadpool) |
| `/usr/lib/bonsai-oryon/omp/` | the same CPU modules built with OpenMP (v0.2 threading), for A/B tests and fallback |
| `/usr/lib/bonsai-oryon/bin/bonsai-perplexity` | helper for output-compatibility checks (not on `$PATH`) |
| `/usr/lib/bonsai-oryon/bin/bonsai-diag` | memory-bandwidth vs PQ2_0 GEMV diagnostics (not on `$PATH`) |
| `/usr/share/bonsai-oryon/` | benchmark and comparison scripts |
| `/usr/share/doc/bonsai-oryon/` | README, changelog, copyright, upstream licences, provenance |

It never touches an existing llama.cpp installation. Binaries use `DT_RPATH=/usr/lib/bonsai-oryon`,
which takes precedence even over a `LD_LIBRARY_PATH` that points at another build. Backends load
**only** from `/usr/lib/bonsai-oryon`, never from the current directory. There are no maintainer
scripts and no systemd units, and nothing starts on its own.

### Uninstall / rollback

```bash
sudo apt remove bonsai-oryon        # removes every file the package installed
# your previous llama.cpp build was never modified; keep using it as before
rm -f ~/.config/bonsai-oryon/profile   # tuned profile written by bonsai-tune (if you ran it)
# back to 0.2.0 (or 0.1.0): download it from its release and
sudo apt install --allow-downgrades ./bonsai-oryon_0.2.0_arm64.deb
# or keep 0.3.0 and switch only the threading back to v0.2's OpenMP modules:
BONSAI_BACKEND_DIR=/usr/lib/bonsai-oryon/omp bonsai-cli ...
```

## Use

The model is **not** bundled. Point the tools at the GGUF you already downloaded:

```bash
bonsai-cli \
  -m ~/bonsai-models/Ternary-Bonsai-2-9B-PQ2_0.gguf \
  -ngl 0 -c 4096
```

```bash
bonsai-server \
  -m ~/bonsai-models/Ternary-Bonsai-2-9B-PQ2_0.gguf \
  --alias bonsai-9b \
  --host 127.0.0.1 \
  --port 8080 \
  -ngl 0 -c 8192
```

Without `-t`, the thread count comes from the runtime profile (next section); the chosen setting is logged
at start-up. `-t`, `-tb`, `--poll` and `-C` always override the profile.

### Runtime profiles (`--oryon-profile`, environment `BONSAI_ORYON_PROFILE`)

| profile | generation threads | prompt (batch) threads | other |
|---|---|---|---|
| `auto` (default for cli/server) | from `bonsai-tune`'s profile if present, else as `balanced` | idem | reads one small file, no start-up measurement |
| `balanced` | all CPUs up to 4, else max(4, CPUs/2): **4 on an 8-core Snapdragon X** | same | `--poll 50` |
| `throughput` | as balanced | all CPUs | for long prompts on an otherwise idle machine |
| `low-power` | half of balanced | same | no polling, barrier yields immediately |
| `none` | upstream defaults | | |

Why not 8 threads: on the Snapdragon X laptop, v0.2 decoded at 8.40 tok/s with 4 threads and 3.04 tok/s
with 8 (desktop session). When every core is busy, any background task delays one worker and all others
wait for it at the ~thousands of synchronization points per token. `bonsai-tune` checks this on your
machine (2..8 threads and each CPU cluster pinned, interleaved rounds) and keeps the fastest setup, preferring
fewer threads when within 3%.
The server listens on `127.0.0.1` by default (also when `--host` is omitted). Don't bind it to
`0.0.0.0` unless you add `--api-key` and a firewall.

### OpenAI-compatible API (OpenCode, SYNAPSE, any OpenAI client)

| endpoint | purpose |
|---|---|
| `GET  /v1/models` | lists `bonsai-9b` (the `--alias`) |
| `POST /v1/chat/completions` | chat, streaming (`"stream": true`, SSE), tools/function calling (`--jinja` is on by default) |
| `POST /v1/completions` | plain completion |
| `POST /v1/embeddings` | needs `--embeddings` |
| `GET  /health` | readiness probe |

Base URL: `http://127.0.0.1:8080/v1`. The API key can be any string unless you start the server with
`--api-key`.

OpenCode (`~/.config/opencode/opencode.json`):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "bonsai": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Bonsai Oryon (local)",
      "options": { "baseURL": "http://127.0.0.1:8080/v1", "apiKey": "local" },
      "models": { "bonsai-9b": { "name": "Ternary Bonsai 2 9B" } }
    }
  }
}
```

For SYNAPSE or any other OpenAI-compatible client, use the provider type "OpenAI compatible", base URL
`http://127.0.0.1:8080/v1`, model `bonsai-9b`.

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"bonsai-9b","messages":[{"role":"user","content":"Hello"}]}'
```

## Switches (environment variables, for A/B tests)

| variable | effect |
|---|---|
| *(none)* | default: 4-row interleaved PQ2_0 layout (v0.2), ggml threadpool, barrier spin 4096 then yield |
| `BONSAI_PQ2_REPACK=0` | the v0.1.0 kernel path (plain layout) |
| `BONSAI_PQ2_TERNARY=1` | lossless ternary 4-row layout: 17.6% fewer bytes per token, more decode work. Helps only if decoding is memory-bound on your machine (see `bonsai-diag`) |
| `BONSAI_BACKEND_DIR=/usr/lib/bonsai-oryon/omp` | OpenMP threading as in v0.2 (same kernels) |
| `BONSAI_BARRIER_SPIN=N` | barrier spin iterations before `sched_yield()` (default 4096; `0` yield at once; `-1` unbounded spin as upstream: fastest on an idle machine, collapses when threads > free cores) |
| `BONSAI_TG_NOBARRIER=1` | experimental: single-token matmuls without the quantize→GEMV barrier (static row split) |
| `BONSAI_PROFILE=1` | print where graph time goes at exit (per op, compute vs barrier wait, worker wake-up) |

All switches produce bit-identical or numerically equivalent output (CI compares the logits byte for byte
across thread counts, threading backends and barrier policies).

## Benchmark on your laptop

```bash
/usr/share/bonsai-oryon/bonsai-benchmark.sh --short          # ~10 min: v0.3 vs v0.2- and v0.1-equivalent
/usr/share/bonsai-oryon/bonsai-benchmark.sh --full           # ~45 min: + thread counts, barrier/poll variants, clusters
/usr/share/bonsai-oryon/bonsai-benchmark.sh --short --release-deb bonsai-oryon_0.2.0_arm64.deb   # real old release
BONSAI_PROFILE=1 bonsai-bench -m MODEL -p 0 -n 64            # where decode time goes
/usr/lib/bonsai-oryon/bin/bonsai-diag                        # DRAM bandwidth vs GEMV
```

Method: every configuration runs once per round, the order is shuffled every round (so heat, battery and
background load affect all configurations alike), with a discarded warm-up round and a cooldown between
runs. Before each run the script records temperature, CPU frequency cap, load average, free memory and
AC/battery; runs taken while hotter/throttled/busier or after a power change are flagged. The report gives
mean, standard deviation and 95% confidence interval per configuration plus a stable-runs-only mean, and
measures model load time and peak RSS separately. `v0.2-equiv` / `v0.1-equiv` use the v0.2 OpenMP modules
(with the v0.1 kernel switch) inside the 0.3.0 package; `--release-deb` runs an older `.deb` from a cache
directory without installing it, `-b` your own llama-bench. Plug the laptop in for stable numbers.

| Snapdragon X, Lenovo IdeaPad Slim 3 15Q8X10, 8 threads | pp256 tok/s | tg128 tok/s |
|---|---|---|
| PrismML e3de715 (user measurement) | 15.57 ± 3.02 | 5.02 ± 0.71 |
| Bonsai Oryon v0.1.0 (user measurement) | 34.82 ± 4.19 | 6.01 ± 0.57 |
| Snapdragon X, same laptop, v0.2.0 (user measurement, on battery, background load) | pp256 | tg128 |
| 4 threads | 26.03 | 8.40 |
| 6 / 7 / 8 threads | 19.30 / 17.69 / 19.71 | 5.15 / 4.20 / 3.04 |
| Bonsai Oryon v0.3.0 | *not measured yet* | *not measured yet* |

## Measured results on CI hardware (Neoverse N2, not Snapdragon X)

Real model `Ternary-Bonsai-2-9B-PQ2_0.gguf` on a GitHub `ubuntu-24.04-arm` runner (Arm Neoverse N2,
4 vCPU), all configurations back to back on the same machine
([model-validation run 38946610464](https://github.com/SatoruGoj0/Bonsai-Oryon/actions/runs/38946610464)):

| build (N2 runner, real 9B model, `-r 3`) | pp256 t=4 | tg128 t=1 | tg128 t=4 | tg64 t=8 on 4 vCPU (oversubscribed) |
|---|---|---|---|---|
| upstream kernels + OpenMP (e3de715) | 8.97 | 1.89 | 6.87 | 6.00 |
| v0.1 equivalent | 13.87 | 2.21 | 8.00 | |
| v0.2 equivalent (4-row, OpenMP) | 18.39 | 3.00 | 10.56 | 8.56 |
| **v0.3 default (4-row, threadpool, bounded spin)** | **18.44** | **2.99** | **10.65** | **9.76** |
| upstream threadpool behaviour (unbounded spin) | 18.48 | 2.97 | 10.63 | 0.23 |

On an idle machine v0.3 equals v0.2 (the kernels are the same; the threading backends are within 1%).
With more runnable threads than cores, v0.3 keeps 94% of its throughput where v0.2's OpenMP build keeps
82% and an unbounded spin collapses to 2%. Output compatibility with the unpatched kernels (full-vocabulary
logits, 1024 tokens): mean KL divergence **0.000000**, same top-1 100%, PPL 10.537824 vs 10.537794.
Logits are byte-identical across thread counts, threadpool vs OpenMP and all barrier policies (CI).
Details, profiler output and memory numbers: [docs/OPTIMIZATIONS.md](docs/OPTIMIZATIONS.md).
**None of this predicts Snapdragon X**; the laptop has to be measured.

## What was changed and why

See [docs/OPTIMIZATIONS.md](docs/OPTIMIZATIONS.md) for the full analysis. Summary of
[the patches](patches/):

1. **`0001` PQ2_0 × Q8_K NEON kernel.** The upstream kernel unpacked each group of 16 weights with
   TBL + SHL + AND + SUB before every SDOT. The new kernel shifts and masks a 16-byte load into four
   code vectors and pairs them with VLD4-deinterleaved activations. It removes the code offset (−1)
   using the Q8_K block sums, and it scales four blocks per float vector instead of running a scalar
   multiply-add chain. A new 2×2 tile (`nrc == 2`) serves prompt processing: each weight vector is
   decoded once and used for two tokens. Batched and single-token results stay bit-identical.
   Measured single-thread GEMV speedup on a Neoverse N2 CI runner: 1.33×.
2. **`0002` NEON `quantize_row_q8_K`.** On ARM this was the scalar reference. The new version is
   bit-identical to it (1.26× faster on Neoverse N2).
3. **`0003` `armv8.6_nosve` CPU variant.** Every upstream I8MM variant also requires SVE. Snapdragon
   X has I8MM but no SVE, so it was dispatched to `armv8.2_2` and lost the I8MM kernels for
   non-PQ2_0 tensors (Q8_0/Q4_K repacks).
4. **`0004` backend isolation.** Backends load only from the package directory, so a stray Vulkan
   backend from another build can never be picked up.
5. **`0005` 4-row interleaved PQ2_0 layout (v0.2, default).** Repacked when the model loads; the
   GGUF is unchanged. For single-token decoding, 4 rows share the activation loads, the row sums
   land in vector lanes (no horizontal reductions) and the codes decode with 4 ops per 16 bytes
   instead of 6, all from one contiguous stream per thread. Prompt processing uses a 4-row × 2-token
   tile.
6. **`0006` lossless ternary 4-row layout (v0.2, opt-in).** Every PQ2_0 tensor of
   Ternary-Bonsai-2-9B uses only −1/0/+1, so it can be streamed at 1.75 bpw (PTQ1_0 trit packing)
   instead of 2.125, 17.6% fewer bytes per token. The decode costs more ALU, so it only helps
   memory-bound machines (`BONSAI_PQ2_TERNARY=1`).

7. **`0007`–`0009` threading (v0.3).** The CPU modules use ggml's persistent threadpool instead of
   OpenMP; the barrier spins at most `BONSAI_BARRIER_SPIN` iterations and then yields, so an
   oversubscribed or busy machine degrades gracefully instead of collapsing; a profiler
   (`BONSAI_PROFILE=1`); `--oryon-profile`; an opt-in barrier-free single-token matmul.

I8MM is not used for single-token decoding: with one activation column half of every SMMLA result
is wasted, so it does no better than SDOT (details in [docs/OPTIMIZATIONS.md](docs/OPTIMIZATIONS.md)).

The runtime picks the CPU variant from HWCAP/HWCAP2. Nothing assumes dotprod, i8mm or SVE: an
ARMv8.0 machine gets the `armv8.0_1` module, which emulates SDOT with SMULL/SADDLP, and CI tests
that path.

## Build from source (reproducible)

```bash
git clone --recursive https://github.com/SatoruGoj0/Bonsai-Oryon && cd Bonsai-Oryon
scripts/prepare-source.sh        # exports pinned e3de715 to build/src and applies patches/
scripts/build.sh                 # CPU-only, GGML_CPU_ALL_VARIANTS, -> build/release/bin
packaging/build-deb.sh           # -> build/dist/bonsai-oryon_0.3.0_arm64.deb
```

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) does exactly this inside `debian:trixie`
on `ubuntu-24.04-arm`. It then runs the kernel, functional, numerical and package tests. A release
is published only by a manual run with `publish=true` or by pushing a `v*` tag, and only if every test
job passed.
[`.github/workflows/model-validation.yml`](.github/workflows/model-validation.yml) is a manual,
opt-in job. It downloads the real 9B model from Hugging Face (nothing gets redistributed) and checks
it against the unpatched build on a CI runner.

## System requirements

- ARM64 (AArch64) Linux, Debian 13 "trixie" or newer (glibc ≥ 2.41, libstdc++ from GCC 14).
  Snapdragon X / Oryon is the target, and any ARMv8.0+ CPU works.
- About 3 GB of free RAM for the 9B PQ2_0 model at `-c 4096`. The KV cache grows with `-c`.
- No root needed to run. Installing the `.deb` needs `sudo`.

## Known limitations

- Experimental: v0.3.0 has not been measured on Snapdragon X. All v0.3 performance numbers in this
  repository come from GitHub's Neoverse N2 runners, and N2 results do not predict Oryon.
- The barrier spin limit and the default thread counts are chosen from N2 measurements and the user's
  v0.2 Snapdragon numbers; `bonsai-tune` exists because the best setting depends on the machine and on
  what else runs on it.
- The 4-row layout copies the weights into anonymous memory at load time. With mmap (default) the
  mapped file pages are counted in RSS too: peak RSS 4.4 GB vs 2.7 GB with `--no-mmap` (N2 runner, +0.35 s
  load). The file pages are reclaimable cache, but on a 16 GB laptop `--no-mmap` is the leaner choice.
- With short `max_tokens` the model may return only `reasoning_content` (thinking) and an empty `content`.
- CPU only. GPU backends are not built (Vulkan is excluded on purpose). No MTP or speculative decoding.
- The server has no embedded web UI (`LLAMA_BUILD_UI=OFF`). The API endpoints are unaffected.
- Downloading models over HTTPS (`-hf`) is disabled (`LLAMA_OPENSSL=OFF`). Use a local GGUF.
- The faster kernels cover PQ2_0 tensors whose rows are a multiple of 256 wide (all of Bonsai-2-9B).
  Other tensor types use the upstream kernels (with I8MM on Oryon thanks to `0003`).

## Attribution and licence

- [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp): MIT, © the ggml authors (Georgi
  Gerganov and contributors).
- [PrismML llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp): MIT. PQ2_0/PTQ1_0 formats and
  Bonsai model support © PrismML and fork contributors.
- Ternary-Bonsai models © PrismML, distributed separately under their own licence. They are not
  included or redistributed here.
- Bonsai Oryon patches, scripts and packaging: MIT ([LICENSE](LICENSE)).

Bundled third-party code (cpp-httplib, nlohmann/json, xxHash, …) is listed in
[packaging/copyright](packaging/copyright). Its licence files are installed under
`/usr/share/doc/bonsai-oryon/licenses/`.
