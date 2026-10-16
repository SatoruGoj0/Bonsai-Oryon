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

## 2b. End-to-end measurement (CI hardware, not Snapdragon X)

Real model, `Ternary-Bonsai-2-9B-PQ2_0.gguf` (arch `qwen35`, 8.95 B params, every weight tensor PQ2_0
including the 248k-vocabulary embedding and output head). GitHub `ubuntu-24.04-arm` runner: Arm
Neoverse N2, 4 vCPU, `-t 4 -ngl 0 -p 256 -n 128 -r 3`, same machine, same CPU module (`armv8.6_2`),
run back to back
([model-validation run](https://github.com/SatoruGoj0/Bonsai-Oryon/actions/runs/38889641006)):

| build | pp256 tok/s | tg128 tok/s |
|---|---|---|
| pinned upstream kernels (patches 0003+0004 only) | 8.95 ± 0.01 | 6.73 ± 0.06 |
| Bonsai Oryon (all patches) | **13.80 ± 0.01** (+54%) | **7.82 ± 0.01** (+16%) |

Output compatibility on the same model, full-vocabulary logits against the unpatched kernels over
1024 tokens: mean KL divergence 5.2e-4 (threshold 1e-3), PPL ratio 1.0018 ± 0.0020, same top-1 token
98.6%.

These numbers come from a 4-vCPU server core with two 128-bit SIMD pipes. **They do not predict
Snapdragon X results.** Nobody has measured the package on Snapdragon X yet, so whether it beats the
5.02 tok/s tg128 baseline there is still open. Run `bonsai-benchmark.sh` on the laptop to find out.

The KL divergence is not zero even though the integer dot products are exact. The patched kernel sums
the per-block float terms in a different order, so each dot changes by about 1e-7 relative. Across 32
recurrent and attention layers that perturbation grows to the KLD above. A residual risk remains:
nobody has yet tested whether a pure float reordering accounts for all of it.

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

---

# v0.2: single-token decoding (tg)

## 4. What the Snapdragon X numbers say

Your measurements of v0.1.0 (8 threads): pp256 34.82 ± 4.19 tok/s (2.2× upstream), tg128 6.01 ± 0.57
(1.2× upstream). On the 4-vCPU Neoverse N2 CI runner the same v0.1 kernels run tg at about 7.7 tok/s. So
with **twice the cores**, Oryon is ~25% **slower** in tg while ~2.5× **faster** in pp. In pp each weight
is reused for 256 tokens, so that number tracks core compute; in tg every weight is streamed once per
token (≈1.97 GB of PQ2_0 weights per token; the 270 MB embedding is only gathered). At 6 tok/s Oryon
streams ≈12 GB/s, a small fraction of LPDDR5x-8448's ≈135 GB/s peak. That pattern means Oryon tg is
**not** limited by the PQ2_0 kernel's compute: the candidates are achievable memory bandwidth under Linux
(memory-controller/interconnect frequency scaling), synchronization stragglers (8 threads with a desktop
running) or CPU clocks. These cannot be told apart without running on the laptop, so v0.2 ships
`bonsai-diag` and an extended benchmark that measure exactly this (§7).

## 5. Profile of tg on the CI runner (N2, real model, 4 threads, `perf`)

| v0.1 path | share | v0.2 4-row layout | share |
|---|---|---|---|
| `ggml_vec_dot_pq2_0_q8_K` | 85.5% | `gemv<block_pq2_0x4r>` | 77.5% |
| libgomp (OpenMP barrier waits) | 6.1% | libgomp (OpenMP barrier waits) | 11.0% |
| gated delta-net | 0.8% | gated delta-net | 1.0% |
| everything else | < 1% each | everything else | < 1% each |

On N2, tg is dominated by the PQ2_0 GEMV; attention/delta-net/norms are negligible. As the GEMV gets
faster, barrier waiting (load imbalance and serial bits such as the single-threaded activation
quantization) becomes the next item.

## 6. Implemented in v0.2

### 0005: 4-row interleaved PQ2_0 layout (default on AArch64)

Repacked at load time through ggml's CPU_REPACK extra buffer (the GGUF is not modified; `-nr`/
`BONSAI_PQ2_REPACK=0` keep the old layout). `block_pq2_0x4r` = 4 fp16 scales + codes ordered
[half][row][16 B] for the same 128-wide block of 4 consecutive rows (136 B, same size as 4 plain blocks).

Per 4 rows × 128 weights, single-token GEMV:
* 2 VLD4 activation loads shared by 4 rows (was 8),
* decode 4 ops per 16 code bytes instead of 6: `q & 0x03` and `q >> 6` are dotted into accumulator A,
  `q & 0x0C` (= code1·4) into B, `q & 0x30` (= code2·16) into C; `A + (B >> 2) + (C >> 4)` via two SSRA
  is exact because every term of B is a multiple of 4 and of C a multiple of 16; all operands are valid
  signed int8 for SDOT (no I8MM needed),
* 3 ADDP put the 4 row sums into the 4 lanes of one vector; the −1 offset (Q8_K bsums), the 4 fp16 scales
  (one FCVTL) and the float accumulation are applied to 4 rows at once, with no horizontal reductions,
* one contiguous weight stream per thread instead of 4 interleaved row streams.

Prompt processing uses a 4-row × 2-token tile (decode shared by both tokens). For every (row, token)
the GEMV and GEMM perform identical float operations, so batched and single-token outputs are
bit-identical (tested). The per-row float accumulation is sequential over blocks like upstream's scalar
order, which is why the real-model KL divergence against the unpatched kernels dropped from 5.2e-4
(v0.1) to **0.000000** (max 5e-5, top-1 100%, PPL 10.537824 vs 10.537794).

Measured on the N2 runner (not Snapdragon X):
* tg-shaped multi-threaded GEMV through the backend (4096×32768, 35.7 MB): **1.39–1.44×** the v0.1 path
  at 1/2/4 threads (25.3 GB/s at 4 threads);
* real model tg64 at 4 threads: **9.35–10.11 tok/s vs 7.73–7.78** for the v0.1 path (+20–31% over two runs).

### 0006: opt-in lossless ternary layout (`BONSAI_PQ2_TERNARY=1`)

A scan of every PQ2_0 tensor of Ternary-Bonsai-2-9B found **no +2 codes**: the checkpoint is strictly
ternary, so PrismML's PTQ1_0 trit packing (5 trits per byte, 1.75 bpw) represents it losslessly. The
opt-in layout streams **17.6% fewer bytes per token**; a tensor that contains a +2 code is detected at
load time and keeps the 2-bit layout. Results are bit-identical to the 2-bit 4-row kernels (same integer
sums, same float sequence). The price is decode ALU (≈4.5 vector ops per 16 trits vs 1 per 16 codes), so
on the compute-bound N2 it is slower (backend GEMV 0.83–0.87× of v0.1, real-model tg64 6.60 tok/s). It
can only win where decoding is memory-bandwidth-bound — which is exactly the open question for Oryon
under Linux. It is therefore off by default and measured by the benchmark script (E2).

### Evaluated and not used

| idea | finding |
|---|---|
| I8MM (SMMLA) for single-token decoding | SMMLA computes a 2×2 tile over 8-deep K. With one activation column, half of every SMMLA result is wasted: 16 useful MACs per instruction, the same as SDOT, plus ZIP shuffles to form the 2×8 operands. No gain for GEMV; I8MM stays useful for prompt processing and for non-PQ2_0 tensors (selected by the `armv8.6_nosve` variant from v0.1). |
| USDOT decode (`q & mask` without shifts, unsigned codes) | saves 2 ops per 16 bytes but needs 4 accumulators per row and recombination; the signed SDOT/SSRA variant above gets the same saving with no I8MM requirement. |
| larger row groups (8 rows) | 8 rows × (A,B,C) accumulators + 8 activation registers exceed the 32 NEON registers; 4 rows keep everything in registers. |
| software prefetch distance tuning | a single `PRFM` 512 B ahead is kept; with one contiguous stream per thread the hardware prefetcher already tracks it on N2. To revisit with Oryon numbers. |

### End-to-end A/B on the CI runner

[model-validation run 38898480980](https://github.com/SatoruGoj0/Bonsai-Oryon/actions/runs/38898480980),
real 9B model, Neoverse N2 4 vCPU, same machine, back to back:

| build (Neoverse N2 runner, real 9B model, `-r 3`) | pp256 t=4 | tg128 t=1 | tg128 t=2 | tg128 t=4 |
|---|---|---|---|---|
| upstream kernels (e3de715 + 0003/0004) | 8.98 | 1.91 | 3.65 | 6.92 |
| v0.1 path (`BONSAI_PQ2_REPACK=0`) | 13.91 | 2.22 | 4.21 | 8.04 |
| **v0.2 4-row layout (default)** | **18.44** | **3.01** | **5.74** | **10.47** |
| v0.2 ternary layout (`BONSAI_PQ2_TERNARY=1`) | 11.88 | 1.83 | 3.56 | 6.81 |

pp is preserved and improved (+33% over v0.1). KL divergence against the unpatched kernels: 0.000000
(max 5e-5, top-1 100%) for both 4-row layouts.

## 7. How to find Oryon's actual limit (run on the laptop)

```bash
/usr/share/bonsai-oryon/bonsai-benchmark.sh -b <your previous llama-bench>
```

* E0 `bonsai-diag`: DRAM read GB/s vs tg-shaped GEMV GB/s per thread count. If the 4-row GEMV reaches
  the DRAM figure, decoding is memory-bound → try E2's ternary row; if it is far below, compute or
  threading is the limit → E1/E3 tell which thread count/placement works best.
* E1 thread sweep 4/6/7/8, E2 layouts (v0.1 path / 4-row / ternary), E3 OpenMP placement, all with
  `-r 3`, plus CPU-frequency, devfreq (memory-controller) and thermal snapshots before/after.

# v0.3: threading and synchronization

## 8. What the Snapdragon X v0.2 numbers say

User measurement on the laptop, v0.2.0 (OpenMP build), on battery, with background load:

| threads | pp256 | tg128 |
|---|---|---|
| 4 | 26.03 | 8.40 |
| 6 | 19.30 | 5.15 |
| 7 | 17.69 | 4.20 |
| 8 | 19.71 | 3.04 |

`bonsai-diag` on the same machine: DRAM read 103.5 GB/s and 4-row GEMV 110.9 GB/s at 8 threads, i.e. the
matmuls alone would allow roughly 45-50 tok/s at 8 threads. At 4 threads tg128 = 8.40 tok/s is ~120 ms per
token while the GEMV share is ~30-40 ms. The remaining time is not in the kernels. A token is 1185 graph
nodes (measured with the v0.3 profiler); every node ends with a barrier, so a decode step has ~1200
synchronization points. On 8 threads
with a desktop running, one preempted worker stalls the other seven at the next barrier. libgomp's barrier
spins for a while and then sleeps on a futex, so each stall costs a wake-up. That explains why throughput
falls monotonically from 4 to 8 threads, and why `OMP_WAIT_POLICY=active` helped at 4 threads (8.94 vs 7.56).

## 9. Implemented in v0.3

* **ggml threadpool instead of OpenMP (build change).** The default CPU modules are built with
  `GGML_OPENMP=OFF`, so the persistent ggml threadpool (created once per context by llama.cpp's common code)
  runs graphs: workers spin on an atomic barrier inside a graph and, between graphs, poll for `--poll`
  (default 50) and then sleep on a condition variable, so an idle server costs nothing. The OpenMP modules
  are still built from the same sources and shipped in `/usr/lib/bonsai-oryon/omp` for A/B and fallback.
* **0007 bounded spin, then yield.** Upstream's threadpool barrier spins without bound. When there are more
  runnable threads than free cores, the spinning waiters take the cores the late worker needs. That is the
  failure the laptop numbers show. After `BONSAI_BARRIER_SPIN` iterations (default 4096) the waiters call
  `sched_yield()` between checks. In qemu with 8 threads on 4 cores (only indicative): unbounded spin 4 tok/s,
  65536 → 46, 1024 → 175, 0 → 193, OpenMP 115, and with 4 threads all policies were equal (≈410-430).
  This is not an unbounded busy-wait: after the bound, the CPU is handed to whoever needs it.
* **0007 profiler.** `BONSAI_PROFILE=1` prints, at exit, thread-0 compute time per op, per-thread compute
  vs barrier/imbalance wait, worker wake-up latency per graph and the graph wall time. The disabled cost is
  one predictable branch per node. Use `bonsai-bench -p 0 -n 64` to profile decoding only.
* **0008 `--oryon-profile`.** Default `auto`: per-user file from `bonsai-tune`, else `balanced`
  (4 threads on 8 cores). Explicit flags always win. No start-up measurement.
* **0009 barrier-free single-token matmul (opt-in, `BONSAI_TG_NOBARRIER=1`).** For <4 activation rows each
  thread quantizes the activation itself and multiplies a static share of the rows, removing the
  quantize→GEMV barrier (one of the two per matmul). Bit-identical results.

All variants are checked in CI for bit-identical logits (thread counts 1/2/3/4, threadpool vs OpenMP,
barrier policies, barrier-free path, ternary layout; batched and token-by-token).

## 10. Measurements of v0.3 on the CI runner (Neoverse N2, 4 vCPU, real 9B model; not Snapdragon X)

[model-validation run 38946610464](https://github.com/SatoruGoj0/Bonsai-Oryon/actions/runs/38946610464),
artifacts of [CI run 38945705618](https://github.com/SatoruGoj0/Bonsai-Oryon/actions/runs/38945705618), all
configurations back to back on one runner, `-r 3`:

| build | pp256 t=4 | tg128 t=1 | tg128 t=2 | tg128 t=4 |
|---|---|---|---|---|
| upstream (e3de715 kernels, OpenMP) | 8.97 | 1.89 | 3.65 | 6.87 |
| v0.1 equivalent (plain kernels, OpenMP modules) | 13.87 | 2.21 | 4.26 | 8.00 |
| v0.2 equivalent (4-row, OpenMP modules) | 18.39 | 3.00 | 5.73 | 10.56 |
| **v0.3 default (4-row, threadpool, spin 4096)** | **18.44** | **2.99** | **5.76** | **10.65** |
| v0.3 + `BONSAI_TG_NOBARRIER=1` | 18.40 | 2.98 | 5.74 | 10.74 |
| v0.3, pure spin (`BONSAI_BARRIER_SPIN=-1`) | 18.48 | 2.97 | 5.73 | 10.63 |
| v0.3, yield at once (`BONSAI_BARRIER_SPIN=0`) | 18.38 | 2.97 | 5.62 | 10.31 |
| v0.3, ternary layout | 11.87 | 1.84 | 3.57 | 6.77 |

On an idle machine the threading backend does not matter (all within ~1%): the profiler shows 92.9% compute
and 7.1% barrier/imbalance wait per thread, 88% of thread-0 time in the PQ2_0 GEMV (400 µs per call), and a
worker wake-up of 0.6 µs per graph for the threadpool vs 30.6 µs for OpenMP.

The difference shows when the machine is busy. tg64 with as many threads as vCPUs, and with twice as many
(every worker competes with another runnable thread, like a desktop under load):

| configuration | t=4 | t=8 (oversubscribed) |
|---|---|---|
| **v0.3 default (spin 4096, then yield)** | 10.37 | **9.76** (−6%) |
| spin 0 (yield at once) | 10.44 | 9.93 |
| spin 65536 | 10.44 | 7.79 |
| unbounded spin (upstream threadpool behaviour) | 10.60 | **0.23** (−98%) |
| v0.2 OpenMP modules | 10.49 | 8.56 (−18%) |
| upstream OpenMP build | 6.92 | 6.00 |

So the bound on the spin is what makes the threadpool usable at all on a loaded machine (upstream's unbounded
spin is 45× slower), and with it v0.3 keeps 94% of its throughput under 2× oversubscription where v0.2's
OpenMP keeps 82%. Whether this closes the gap on the laptop (8.40 at 4 threads vs 3.04 at 8) has to be
measured there; the N2 runner has no desktop, no heterogeneous clusters and no thermal limit.

The barrier-free matmul (0009) is within noise on N2 (+0.8% tg at t=4); it stays opt-in until measured on Oryon.

**Memory / load** (warm page cache, one 1-token run):

| layout | load | wall s | peak RSS MiB |
|---|---|---|---|
| 4-row (default) | mmap | 1.43 | 4416 |
| 4-row | `--no-mmap` | 1.78 | 2655 |
| plain (`BONSAI_PQ2_REPACK=0`) | mmap | 0.77 | 2525 |
| plain | `--no-mmap` | 1.41 | 2525 |

With mmap the repacked copy (anonymous memory) and the mapped file pages are both counted in RSS. The file
pages are clean page cache and the kernel can drop them, but they occupy memory until it does. On a 16 GB
laptop `--no-mmap` (or `-mmp 0` for bonsai-bench) saves ~1.7 GB RSS for ~0.3 s more load time.

**Server** (real model, `--jinja`): `/v1/chat/completions` with tools returned
`finish_reason=tool_calls` and `get_weather({"city":"Paris"})`; streaming delivered 34 SSE chunks and `[DONE]`.
With 32-64 `max_tokens` the visible `content` was empty because the model was still in its thinking
section (returned as `reasoning_content`), so use a larger `max_tokens` or disable thinking in the chat template
for short answers.

## 11. Remaining bottlenecks and v0.4 directions

* On idle hardware tg is the PQ2_0 GEMV (88%). N2 is compute-bound at ~25 GB/s per 4 threads; Oryon reached
  110.9 GB/s at 8 threads in `bonsai-diag`, so on the laptop the kernel is not the limit but scheduling is.
* Synchronization: ~1200 barriers per token. v0.4 candidates: fuse elementwise chains (norm·mul, GLU, add) into
  their producer so they need no barrier; let threads skip the barrier for ops whose rows they own
  (thread-affine row partition from one matmul to the next).
* Heterogeneous clusters: Snapdragon X has several clusters; pinning one cluster (`bonsai-tune` tries this)
  vs spreading.
* Repack memory: drop the mmapped source pages after repacking (`madvise(MADV_DONTNEED)` / unmap the
  repacked ranges) to get no-mmap RSS with mmap load times.
* The ternary layout loses on N2 (compute-bound); only worth it if Oryon is memory-bound at the chosen
  thread count, which `bonsai-diag` and `bonsai-benchmark.sh --full` show.
