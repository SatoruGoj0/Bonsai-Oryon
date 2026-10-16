# Bonsai Oryon v0.1.0: ARM64 CPU inference (experimental)

CPU-only ARM64 build of the [PrismML llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp) (`prism` @ `e3de715`), with AArch64 kernels tuned for **Ternary-Bonsai-2-9B PQ2_0**. Packaged for **Debian 13 (trixie) arm64**.

> **Experimental.** This build passed functional and numerical validation on generic ARM64 CI runners. Nobody has measured its performance on Snapdragon X yet. Please run the bundled benchmark on your machine (see below). The release is marked *pre-release* for that reason.

## Install

```bash
sha256sum -c bonsai-oryon_0.1.0_arm64.deb.sha256
sudo apt install ./bonsai-oryon_0.1.0_arm64.deb
```

## Run

```bash
bonsai-cli    -m ~/bonsai-models/Ternary-Bonsai-2-9B-PQ2_0.gguf -t 8 -ngl 0 -c 4096
bonsai-server -m ~/bonsai-models/Ternary-Bonsai-2-9B-PQ2_0.gguf --alias bonsai-9b \
              --host 127.0.0.1 --port 8080 -t 8 -ngl 0 -c 8192
# OpenAI-compatible API: http://127.0.0.1:8080/v1  (model: bonsai-9b)
```

## Measure on your laptop

```bash
/usr/share/bonsai-oryon/bonsai-benchmark.sh --sweep -b <path to your current llama-bench>
/usr/share/bonsai-oryon/compare-output.sh <path to your current llama-perplexity>
```

## Rollback

```bash
sudo apt remove bonsai-oryon     # your existing llama.cpp build is untouched
```

## What's inside

- `0001` NEON PQ2_0×Q8_K kernel: de-interleaved decode, offset folded out through the Q8_K bsums, a 2×2 tile for prompt processing. Batched and single-token results are bit-identical.
- `0002` NEON `quantize_row_q8_K`, bit-identical to the reference.
- `0003` `armv8.6_nosve` CPU variant. Snapdragon X has I8MM but no SVE; it now gets the I8MM kernels.
- `0004` backends load only from `/usr/lib/bonsai-oryon`, never from the CWD.

Full details are in [docs/OPTIMIZATIONS.md](https://github.com/SatoruGoj0/Bonsai-Oryon/blob/main/docs/OPTIMIZATIONS.md).

## Known limitations

CPU only (no Vulkan). No embedded web UI. No HTTPS model download. Model weights are not included. Not yet benchmarked on Snapdragon X.
