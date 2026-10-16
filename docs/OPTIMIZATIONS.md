# PQ2_0 on AArch64: analysis and optimizations

Target: Ternary-Bonsai-2-9B-PQ2_0 on Snapdragon X (8 Oryon cores, LPDDR5x, Debian 13).
Upstream: PrismML llama.cpp `prism` @ `e3de715`.

## 1. How PQ2_0 runs on the CPU (upstream e3de715)

* Format (`ggml-common.h`): 128 weights per block, one fp16 scale plus 32 bytes of 2-bit codes, so
  34 bytes per block, 2.125 bpw. Byte *m* holds the codes of weights 4m..4m+3, least significant pair
  first. Each weight is `d · (code − 1)`, with codes {0,1,2,3} mapping to {−1, 0, +1, +2}.
* Matmul (`ggml-cpu.c`): the type traits set `vec_dot = ggml_vec_dot_pq2_0_q8_K`,
  `vec_dot_type = Q8_K`, `nrows = 1`. Every matmul input row is quantized to Q8_K (256 int8 values
  per block plus an fp32 scale plus 16 int16 partial sums `bsums`), and then each output element is
  one `vec_dot` call. Rows whose width is not a multiple of 256 fall back to Q8_0 activations and
  `ggml_vec_dot_pq2_0_q8_0`.
* On ARM there is no repack and no llamafile/tinyBLAS path for PQ2_0. Those exist only for
  AVX2/AVX-512. Token generation (tg) and prompt processing (pp) both go through `vec_dot`, one row
  and one column at a time.
* `quantize_row_q8_K` on ARM was the scalar reference. The source comment calls it a "placeholder
  implementation for Apple targets".

### Static analysis of the upstream NEON kernel

For each block of 128 weights, the upstream kernel runs this four times, once per group of 32
weights: `ld1` (8 B), `combine`, then twice `tbl`, `ushl`, `and`, `sub` and `sdot`. After the four
groups it does a horizontal `addv`, an int→float conversion, and a scalar `fmadd` into a single
running `sumf`. GCC 14 at `-O3` (`armv8.6-a+dotprod+i8mm`) emits **≈ 84 instructions per
128-weight block**, 8 of which are SDOTs. Every block also ends in a loop-carried scalar FMA chain
(`addv` → `scvtf` → `fmadd`).

### Is tg compute-bound or memory-bound on Snapdragon X?

A 9B model at 2.125 bpw is about 2.4 GB of weights per generated token. The measured 5.02 tok/s
moves only about 12 GB/s, a small fraction of the roughly 135 GB/s LPDDR5x-8448 peak. At 84
instructions per 34-byte block, the eight cores would need about 70 M × 84 ≈ 5.9 G instructions per
token just for PQ2_0 dots. The kernel's instruction count is therefore the first bottleneck, well
before DRAM bandwidth. A bandwidth roofline (about 2.4 GB at an assumed ~100 GB/s achievable)
would allow roughly 40 tok/s. That ceiling is theoretical, and other ops (attention / delta-net,
norms, the output head) add time.

## 2. Implemented changes (see `patches/`)

### 0001: de-interleaved PQ2_0 × Q8_K kernel + 2×2 tile

```
q    = ld1(x.qs[0..15])                       ; codes of weights 0..63, byte m = weights 4m..4m+3
c0   = q & 3        ; lane m = code(4m+0)
c1   = (q >> 2) & 3 ; lane m = code(4m+1)
c2   = (q >> 4) & 3 ; lane m = code(4m+2)
c3   = q >> 6       ; lane m = code(4m+3)
y    = ld4(q8[0..63]) ; y.val[j][m] = q8[4m+j]   <- VLD4 de-interleaves to match
acc  = sdot(sdot(sdot(sdot(acc, c0, y0), c1, y1), c2, y2), c3, y3)
```

* No TBL and no SUB. The −1 offset comes out algebraically:
  `Σ (c−1)·y = Σ c·y − Σ y`, and `Σ y` over a 128-half is eight of the Q8_K `bsums`, which the
  activation quantizer computes anyway. Codes 0..3 are valid signed int8 SDOT operands.
  Per 16 weights the decode costs 1.5 ALU ops instead of 4.
* Four blocks are processed per iteration on independent accumulators. Pairwise `addp` puts block *k*
  in lane *k*, the `bsums` are subtracted as a vector, and a single `fmla` with four `d_x·d_y` scales
  replaces four scalar `addv`/`fmadd` chains.
* Result: **≈ 42 instructions per 128-weight block** (GCC 14 `-O3`, counted from the loop body), about
  half of upstream, with no loop-carried scalar FP dependency.
* **Measured** (CI runner, Arm Neoverse N2, one thread, `-march=armv8.6-a+dotprod+fp16+i8mm`, 4096-wide
  rows): upstream 14.3 Gweights/s, Bonsai 19.1 Gweights/s, a **1.33×** speedup. The result is the same
  for a 35 KB cache-resident matrix and an 8.9 MB matrix, so the kernel is compute-bound on that core.
  N2 has two 128-bit SIMD pipes, while Oryon is reported to have four, so the gain on Snapdragon X
  may differ in either direction. It has **not been measured** there.
* **2×2 tile** (`nrc == 2`, `nrows = 2` on AArch64 NEON): for prompt processing, each weight vector
  is decoded once and each activation vector loaded once for two rows × two tokens. The float
  operations for a given (row, column) are identical to the single-row path. Batched pp output is
  therefore **bit-identical** to single-token output, and the test suite checks this.
* Integer block sums are exact. Only the float summation order across blocks changes relative to the
  scalar reference: the measured relative error against an exact double reference is ≤ 1e-7.
* ARMv8.0 CPUs without dotprod get `ggml_vdotq_s32`, upstream's SMULL/SADDLP emulation. Only the
  total of the four lanes is used, so the emulation is exact.
* The Q8_0 fallback (`ggml_vec_dot_pq2_0_q8_0`) serves `nrc == 2` as four single dots, so rows of
  128·odd width stay correct with `nrows = 2`.

### 0002: NEON `quantize_row_q8_K`

The new code vectorizes the max-abs pass (`fmaxnm`, so NaNs are skipped exactly like the reference's
`>` comparison), the scaling and rounding, the saturating narrow to int8 and the 16-element `bsums`.
It is **bit-identical** to `quantize_row_q8_K_ref` as compiled by GCC. The reference's
`nearest_int(iscale*x)` uses the 1.5·2²³ magic-number trick, and GCC's default `-ffp-contract=fast`
fuses `iscale*x + magic` into one FMA (confirmed in the disassembly). A plain multiply followed by
FCVTNS gave 39 mismatches in 1668 blocks, all on rounding ties. The patch therefore uses FMLA with the
same magic constant, and the result has 0 mismatches.

**Measured** (CI runner, Neoverse N2, single thread, 12288 floats): reference 8.85 µs, NEON 7.04 µs,
a 1.26× speedup. Activation quantization is a small share of a token in any case.

### 0003: `armv8.6_nosve` CPU variant

`GGML_CPU_ALL_VARIANTS` on Linux builds `armv8.0_1`, `armv8.2_{1,2,3}`, `armv8.6_{1,2}` and
`armv9.2_{1,2}`, and every I8MM variant there also requires SVE. Oryon has `asimddp`, `asimdhp`,
`i8mm` and `bf16`, but **no SVE**, so the loader scored it onto `armv8.2_2`. That variant lacks I8MM,
so the Q8_0 and Q4_K repacked i8mm GEMM/GEMV kernels and the i8mm `vec_dot`s for non-PQ2_0 tensors
(embeddings, output head and so on) were never used. The new variant is `DOTPROD FP16 MATMUL_INT8
NOSVE` with a score of 23, compared with 7 for `armv8.2_2`, so the loader prefers it.

### 0004: backend isolation

When `GGML_BACKEND_DIR` is set (the package sets it to `/usr/lib/bonsai-oryon`), backends are searched
**only** there, not in the executable directory or the CWD. Starting `bonsai-cli` from inside another
llama.cpp build directory can therefore never load that build's `libggml-vulkan.so`. Tests can set
`BONSAI_BACKEND_DIR` to point a build tree at its own modules.

## 3. Investigated, not changed (yet)

| area | finding | decision |
|---|---|---|
| threading | ggml's mul_mat splits PQ2_0 rows across threads in 16-row chunks with a dynamic chunk counter. tg uses one activation column, so the Q8_K quantize of that column is single-threaded, which is cheap now thanks to 0002. | No change. Measure `-t 6/8` on Oryon with the sweep, since mixed cluster clocks may favour 6 threads. |
| intermediate buffers | Activations are quantized once per matmul into `wdata`, sized by `ggml_graph_plan`. There is no per-row allocation. | Already optimal for this design. |
| KV cache | Allocated once at context creation: `-c` × layers × kv-heads × head-dim in F16 by default. Bonsai-2-9B is a hybrid (delta-net plus attention), so only the attention layers hold a KV cache. | `-ctk q8_0 -ctv q8_0` (with `-fa on`) halves it if RAM is tight. Left as a user option. |
| memory bandwidth | PQ2_0 rows are streamed linearly, 34 B per block, and the activation row stays in L1 (4–48 KiB). | The kernel's instruction count, not DRAM, was the limiter (see §1). |
| repacked 4-row PQ2_0 layout on ARM | Upstream has the `pq2_0_4x8` repack only for AVX-512. On NEON, interleaving four rows would let one VLD4 activation load feed four rows during tg. | A follow-up candidate. Needs hardware profiling first, since it is a large change. |
| `usdot` (I8MM) decode | Masking without shifting (`q & 0xC0`, …) yields codes ×4ʲ as unsigned bytes, which feed `usdot` and save two ALU ops per 16 bytes. | Gain ≤ 10%, more complexity. Revisit after measuring on Oryon. |
| handwritten assembly | The intrinsic loop already schedules well (no spills in the 1-row path). | Not justified without hardware profiles. |
