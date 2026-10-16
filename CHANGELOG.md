# Changelog

## 0.1.0 (2026-10-16), experimental

First release. Upstream is PrismML llama.cpp `prism` @ `e3de715222b2667c54a267441159a5729f11155e`.

- `0001` AArch64 NEON PQ2_0 × Q8_K kernel: de-interleaved decode (VLD4 + shift/mask), the −1 offset
  folded out through Q8_K `bsums`, 4 blocks per float vector, and a new bit-consistent 2×2 tile for
  prompt processing.
- `0002` NEON `quantize_row_q8_K`, bit-identical to the reference.
- `0003` `armv8.6_nosve` CPU variant (DOTPROD+FP16+I8MM, no SVE), which Snapdragon X now selects.
- `0004` backends load only from `/usr/lib/bonsai-oryon`.
- Debian package `bonsai-oryon_0.1.0_arm64.deb`: `bonsai-cli`, `bonsai-server`, `bonsai-bench`, with
  private libraries, no services and no maintainer scripts.
- Test suite: kernel unit tests on three ISA levels, upstream ggml PQ2_0 tests, CLI/bench/server
  OpenAI-API functional tests, KL-divergence comparison against unpatched kernels, and package
  install/remove in a clean Debian 13 container.
- Not yet measured on Snapdragon X hardware.
