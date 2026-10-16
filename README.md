# Bonsai Oryon

**ARM64 CPU inference engine for PrismML Ternary-Bonsai (PQ2_0) models, packaged for Debian 13 on
Snapdragon X (Qualcomm Oryon) laptops.**

Bonsai Oryon is the [PrismML fork of llama.cpp](https://github.com/PrismML-Eng/llama.cpp) (branch
`prism`, pinned at commit `e3de715`), built CPU-only, plus a small set of auditable patches that speed
up the PQ2_0 path on AArch64. It ships as one `.deb` with `bonsai-cli`, `bonsai-server` and
`bonsai-bench`, so you don't have to compile anything on the laptop.

> **Status: experimental (v0.1.0).** CI has checked functional and numerical correctness on generic
> ARM64 servers (Arm Neoverse, GitHub `ubuntu-24.04-arm`). Nobody has measured performance on
> Snapdragon X yet. Use [`bench/bonsai-benchmark.sh`](bench/bonsai-benchmark.sh) to measure it on your
> own machine.

## Install

```bash
# download the .deb and checksum from the GitHub Release, then:
sha256sum -c bonsai-oryon_0.1.0_arm64.deb.sha256
sudo apt install ./bonsai-oryon_0.1.0_arm64.deb
```

The package installs only these paths:

| path | contents |
|---|---|
| `/usr/bin/bonsai-cli`, `/usr/bin/bonsai-server`, `/usr/bin/bonsai-bench` | executables (renamed `llama-cli`, `llama-server`, `llama-bench`) |
| `/usr/lib/bonsai-oryon/` | private `libllama`, `libggml*`, `libmtmd` and one `libggml-cpu-<variant>.so` per ARM feature level |
| `/usr/lib/bonsai-oryon/bin/bonsai-perplexity` | helper for output-compatibility checks (not on `$PATH`) |
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
```

## Use

The model is **not** bundled. Point the tools at the GGUF you already downloaded:

```bash
bonsai-cli \
  -m ~/bonsai-models/Ternary-Bonsai-2-9B-PQ2_0.gguf \
  -t 8 -ngl 0 -c 4096
```

```bash
bonsai-server \
  -m ~/bonsai-models/Ternary-Bonsai-2-9B-PQ2_0.gguf \
  --alias bonsai-9b \
  --host 127.0.0.1 \
  --port 8080 \
  -t 8 -ngl 0 -c 8192
```

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

## Benchmark on your laptop

```bash
/usr/share/bonsai-oryon/bonsai-benchmark.sh                       # pp256 + tg128, 8 threads, 3 reps
/usr/share/bonsai-oryon/bonsai-benchmark.sh --sweep               # also 2/4/6 threads
/usr/share/bonsai-oryon/bonsai-benchmark.sh -b ~/llama.cpp/build/bin/llama-bench   # A/B vs your build
/usr/share/bonsai-oryon/compare-output.sh ~/llama.cpp/build/bin/llama-perplexity   # logits KL divergence
```

The default settings match the reference measurement (CPU only, `-t 8`, `pp256`, `tg128`, 3
repetitions, same model file). The script writes `report.md` with system details (CPU features,
governor, AC/battery state), tokens/s ± stddev and peak RSS. For stable numbers, plug the laptop in and
close other applications.

| reference (PrismML e3de715, your Lenovo, before this project) | tok/s |
|---|---|
| CPU pp256 | 15.57 |
| CPU tg128 | 5.02 |
| Vulkan tg128 (excluded: desktop lag) | 4.22 |

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

The runtime picks the CPU variant from HWCAP/HWCAP2. Nothing assumes dotprod, i8mm or SVE: an
ARMv8.0 machine gets the `armv8.0_1` module, which emulates SDOT with SMULL/SADDLP, and CI tests
that path.

## Build from source (reproducible)

```bash
git clone --recursive https://github.com/SatoruGoj0/Bonsai-Oryon && cd Bonsai-Oryon
scripts/prepare-source.sh        # exports pinned e3de715 to build/src and applies patches/
scripts/build.sh                 # CPU-only, GGML_CPU_ALL_VARIANTS, -> build/release/bin
packaging/build-deb.sh           # -> build/dist/bonsai-oryon_0.1.0_arm64.deb
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

- Experimental: performance on Snapdragon X has not been measured by this project yet. The
  reference numbers above are from before the patches.
- CPU only. GPU backends are not built (Vulkan is excluded on purpose).
- The server has no embedded web UI (`LLAMA_BUILD_UI=OFF`). The API endpoints are unaffected.
- Downloading models over HTTPS (`-hf`) is disabled (`LLAMA_OPENSSL=OFF`). Use a local GGUF.
- The faster kernels cover PQ2_0. Other tensor types use the upstream kernels (with I8MM on Oryon
  thanks to `0003`).

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
