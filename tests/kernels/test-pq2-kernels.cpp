// Bonsai-Oryon kernel tests: PQ2_0 decoding, PQ2_0 x Q8_K GEMV accuracy, 2x2 tile consistency,
// Q8_K activation quantization bit-exactness, Q8_0 fallback rows, ARM64 runtime feature detection,
// full ggml mul_mat consistency (single-token vs batched) and an old-vs-new kernel microbenchmark.
//
// Uses only the public ggml / ggml-cpu API plus the exported reference quantizers, so it tests exactly
// what the inference engine calls. Exit code 0 = all tests passed.
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"

#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#if defined(__linux__) && defined(__aarch64__)
#include <sys/auxv.h>
#include <asm/hwcap.h>
#endif
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

static int g_failures = 0;
static int g_checks   = 0;

#define CHECK(cond, ...)                                                    \
    do {                                                                    \
        ++g_checks;                                                         \
        if (!(cond)) {                                                      \
            ++g_failures;                                                   \
            std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);      \
            std::fprintf(stderr, __VA_ARGS__);                              \
            std::fprintf(stderr, "\n");                                     \
        }                                                                   \
    } while (0)

static std::mt19937 rng(1234567);

static float frand(float lo, float hi) { return std::uniform_real_distribution<float>(lo, hi)(rng); }

static std::vector<block_pq2_0> random_pq2(int64_t n) {
    std::vector<block_pq2_0> x(n / QK_PQ2_0);
    for (auto & b : x) {
        b.d = ggml_fp32_to_fp16(frand(0.001f, 0.1f));
        for (auto & q : b.qs) q = (uint8_t) (rng() & 0xff);
    }
    return x;
}

static std::vector<float> random_act(int64_t n, float scale = 1.0f) {
    std::vector<float> v(n);
    std::normal_distribution<float> nd(0.0f, scale);
    for (auto & f : v) f = nd(rng);
    return v;
}

static int pq2_code(const block_pq2_0 & b, int j) { return (b.qs[j / 4] >> (2 * (j % 4))) & 3; }

// exact reference: integer dot per 128-block, then double accumulation
static double ref_dot_q8K(int64_t n, const block_pq2_0 * x, const block_q8_K * y, double * abs_sum) {
    double s = 0.0, as = 0.0;
    for (int64_t i = 0; i < n / QK_PQ2_0; ++i) {
        const block_q8_K & yb = y[i / 2];
        const int8_t * q8 = yb.qs + 128 * (i % 2);
        int64_t isum = 0;
        for (int j = 0; j < QK_PQ2_0; ++j) isum += (int64_t) (pq2_code(x[i], j) - 1) * q8[j];
        const double t = (double) ggml_fp16_to_fp32(x[i].d) * (double) yb.d * (double) isum;
        s += t; as += std::fabs(t);
    }
    if (abs_sum) *abs_sum = as;
    return s;
}

// ---------------------------------------------------------------------------------------------
// Copy of the upstream (PrismML e3de715) NEON PQ2_0 x Q8_K kernel, used only as the microbenchmark
// baseline. MIT License, Copyright (c) 2023-2026 The ggml authors.
#if defined(__ARM_NEON) && defined(__aarch64__)
static inline int32x4_t vdot_compat(int32x4_t acc, int8x16_t a, int8x16_t b) {
#if defined(__ARM_FEATURE_DOTPROD)
    return vdotq_s32(acc, a, b);
#else
    const int16x8_t p0 = vmull_s8(vget_low_s8(a), vget_low_s8(b));
    const int16x8_t p1 = vmull_s8(vget_high_s8(a), vget_high_s8(b));
    return vaddq_s32(acc, vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1)));
#endif
}
__attribute__((noinline)) static void upstream_vec_dot_pq2_0_q8_K(int n, float * s, const void * vx, const void * vy) {
    const block_pq2_0 * x = (const block_pq2_0 *) vx;
    const block_q8_K  * y = (const block_q8_K *) vy;
    const int nb = n / QK_PQ2_0;
    static const uint8_t tbl_idx_lo[16] = {0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3};
    static const uint8_t tbl_idx_hi[16] = {4,4,4,4, 5,5,5,5, 6,6,6,6, 7,7,7,7};
    static const int8_t  shift_vals[16] = {0,-2,-4,-6, 0,-2,-4,-6, 0,-2,-4,-6, 0,-2,-4,-6};
    const uint8x16_t idx_lo = vld1q_u8(tbl_idx_lo);
    const uint8x16_t idx_hi = vld1q_u8(tbl_idx_hi);
    const int8x16_t  shifts = vld1q_s8(shift_vals);
    const uint8x16_t mask2  = vdupq_n_u8(0x03);
    const int8x16_t  one    = vdupq_n_s8(1);
    float sumf = 0.0f;
    for (int i = 0; i < nb; i++) {
        const block_q8_K * yb = &y[i >> 1];
        const int8_t     * q8 = yb->qs + 128 * (i & 1);
        int32x4_t acc = vdupq_n_s32(0);
        for (int k = 0; k < 4; k++) {
            const uint8x8_t  raw   = vld1_u8(&x[i].qs[8 * k]);
            const uint8x16_t raw16 = vcombine_u8(raw, raw);
            const int8x16_t qv0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshlq_u8(vqtbl1q_u8(raw16, idx_lo), shifts), mask2)), one);
            const int8x16_t qv1 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshlq_u8(vqtbl1q_u8(raw16, idx_hi), shifts), mask2)), one);
            acc = vdot_compat(acc, qv0, vld1q_s8(q8 + 32 * k));
            acc = vdot_compat(acc, qv1, vld1q_s8(q8 + 32 * k + 16));
        }
        sumf += (ggml_fp16_to_fp32(x[i].d) * yb->d) * (float) vaddvq_s32(acc);
    }
    *s = sumf;
}
#endif
// ---------------------------------------------------------------------------------------------

static void test_runtime_detection() {
    std::printf("[detect] compiled: NEON=%d DOTPROD=%d I8MM=%d SVE=%d FP16VA=%d\n",
#if defined(__ARM_NEON)
        1,
#else
        0,
#endif
#if defined(__ARM_FEATURE_DOTPROD)
        1,
#else
        0,
#endif
#if defined(__ARM_FEATURE_MATMUL_INT8)
        1,
#else
        0,
#endif
#if defined(__ARM_FEATURE_SVE)
        1,
#else
        0,
#endif
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
        1
#else
        0
#endif
    );
    std::printf("[detect] ggml_cpu_has: neon=%d dotprod=%d matmul_int8=%d sve=%d fp16_va=%d\n",
        ggml_cpu_has_neon(), ggml_cpu_has_dotprod(), ggml_cpu_has_matmul_int8(), ggml_cpu_has_sve(),
        ggml_cpu_has_fp16_va());
#if defined(__linux__) && defined(__aarch64__)
    const unsigned long hw  = getauxval(AT_HWCAP);
    const unsigned long hw2 = getauxval(AT_HWCAP2);
    const int hw_dot  = (hw & HWCAP_ASIMDDP) != 0;
    const int hw_sve  = (hw & HWCAP_SVE) != 0;
    const int hw_fp16 = (hw & HWCAP_ASIMDHP) != 0;
#ifdef HWCAP2_I8MM
    const int hw_i8mm = (hw2 & HWCAP2_I8MM) != 0;
#else
    const int hw_i8mm = (hw2 & (1UL << 13)) != 0;
#endif
    std::printf("[detect] kernel HWCAP: asimddp=%d i8mm=%d sve=%d asimdhp=%d\n", hw_dot, hw_i8mm, hw_sve, hw_fp16);
    // ggml reports a feature only if it was compiled in AND the CPU has it at runtime
#if defined(__ARM_FEATURE_DOTPROD)
    CHECK(ggml_cpu_has_dotprod() == hw_dot, "dotprod detection mismatch (ggml=%d hwcap=%d)", ggml_cpu_has_dotprod(), hw_dot);
#else
    CHECK(ggml_cpu_has_dotprod() == 0, "dotprod reported although not compiled in");
#endif
#if defined(__ARM_FEATURE_MATMUL_INT8)
    CHECK(ggml_cpu_has_matmul_int8() == hw_i8mm, "i8mm detection mismatch (ggml=%d hwcap=%d)", ggml_cpu_has_matmul_int8(), hw_i8mm);
#else
    CHECK(ggml_cpu_has_matmul_int8() == 0, "i8mm reported although not compiled in");
#endif
#if defined(__ARM_FEATURE_SVE)
    CHECK(ggml_cpu_has_sve() == hw_sve, "sve detection mismatch");
#else
    CHECK(ggml_cpu_has_sve() == 0, "sve reported although not compiled in");
#endif
    CHECK(ggml_cpu_has_neon() == 1, "NEON must be available on AArch64");
#endif
}

static void test_decode() {
    const ggml_type_traits * tt = ggml_get_type_traits(GGML_TYPE_PQ2_0);
    CHECK(tt->blck_size == 128 && tt->type_size == 34, "unexpected PQ2_0 block geometry %" PRId64 "/%zu", tt->blck_size, tt->type_size);
    const int64_t n = 128 * 64;
    auto x = random_pq2(n);
    std::vector<float> out(n);
    tt->to_float(x.data(), out.data(), n);
    int bad = 0;
    for (int64_t i = 0; i < n; ++i) {
        const float want = (float) (pq2_code(x[i / 128], (int) (i % 128)) - 1) * ggml_fp16_to_fp32(x[i / 128].d);
        if (out[i] != want) ++bad;
    }
    CHECK(bad == 0, "dequantize_row_pq2_0 mismatches: %d", bad);

    // ternary weights with an fp16-exact scale round-trip losslessly through the reference quantizer
    std::vector<float> w(n), back(n);
    for (int64_t i = 0; i < n; ++i) w[i] = 0.5f * (float) ((int) (rng() % 3) - 1);
    for (int64_t b = 0; b < n / 128; ++b) w[b * 128] = 0.5f; // pin amax
    std::vector<block_pq2_0> q(n / 128);
    quantize_row_pq2_0_ref(w.data(), q.data(), n);
    tt->to_float(q.data(), back.data(), n);
    bad = 0;
    for (int64_t i = 0; i < n; ++i) if (back[i] != w[i]) ++bad;
    CHECK(bad == 0, "ternary PQ2_0 round trip mismatches: %d", bad);
    std::printf("[decode] PQ2_0 decode + ternary round-trip: %s\n", bad == 0 ? "ok" : "FAILED");
}

static bool same_q8K(const block_q8_K & a, const block_q8_K & b) {
    if (std::memcmp(&a.d, &b.d, sizeof(float)) != 0) return false;
    if (std::memcmp(a.qs, b.qs, sizeof(a.qs)) != 0) return false;
    if (a.d == 0.0f) return true; // the reference leaves bsums untouched for all-zero blocks
    return std::memcmp(a.bsums, b.bsums, sizeof(a.bsums)) == 0;
}

static void test_q8K_quantize() {
    const auto * cpu = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    int bad = 0, total = 0;
    for (int trial = 0; trial < 200; ++trial) {
        const int64_t n = 256 * (1 + trial % 16);
        auto v = random_act(n, trial % 3 == 0 ? 100.0f : 1.0f);
        if (trial % 5 == 1) for (int64_t i = 0; i < n; i += 3) v[i] = std::round(v[i] * 4.0f) / 4.0f; // exact ties
        if (trial % 7 == 2) { v[5] = 3.0f; v[9] = -3.0f; for (int64_t i = 10; i < 256; ++i) v[i] = std::fmod(v[i], 2.9f); } // +/- equal amax
        if (trial % 11 == 3) std::fill(v.begin(), v.begin() + 256, 0.0f); // zero block
        if (trial % 13 == 4) v[0] = -1e-30f; // tiny negative max in otherwise-zero-ish data
        std::vector<block_q8_K> a(n / 256), b(n / 256);
        cpu->from_float(v.data(), a.data(), n);
        quantize_row_q8_K_ref(v.data(), b.data(), n);
        for (size_t i = 0; i < a.size(); ++i) { ++total; if (!same_q8K(a[i], b[i])) ++bad; }
    }
    CHECK(bad == 0, "quantize_row_q8_K differs from reference in %d/%d blocks", bad, total);
    std::printf("[q8_K] NEON quantize vs reference: %d/%d blocks bit-identical\n", total - bad, total);
}

static void test_vec_dot_q8K() {
    const auto * cpu = ggml_get_type_traits_cpu(GGML_TYPE_PQ2_0);
    const auto * q8k = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    CHECK(cpu->vec_dot_type == GGML_TYPE_Q8_K, "PQ2_0 vec_dot_type is not Q8_K");
    std::printf("[gemv] PQ2_0 vec_dot nrows=%" PRId64 "\n", cpu->nrows);
    double worst = 0.0;
    int bad2x2 = 0;
    const int64_t sizes[] = {256, 512, 768, 1280, 2048, 4096, 5120, 12288};
    for (int64_t n : sizes) {
        for (int trial = 0; trial < 20; ++trial) {
            auto x0 = random_pq2(n), x1 = random_pq2(n);
            auto a0 = random_act(n), a1 = random_act(n, 3.0f);
            std::vector<block_q8_K> y0(n / 256), y1(n / 256);
            q8k->from_float(a0.data(), y0.data(), n);
            q8k->from_float(a1.data(), y1.data(), n);
            float s1 = 0.0f;
            cpu->vec_dot((int) n, &s1, 0, x0.data(), 0, y0.data(), 0, 1);
            double as = 0.0;
            const double r = ref_dot_q8K(n, x0.data(), y0.data(), &as);
            const double err = std::fabs((double) s1 - r) / (as + 1e-30);
            if (err > worst) worst = err;
            CHECK(err < 1e-5, "vec_dot n=%" PRId64 " rel err %.3g (got %.8g want %.8g)", n, err, s1, r);

            if (cpu->nrows == 2) {
                // 2x2 tile: x rows contiguous at stride bx, y columns at stride by, outputs s[0],s[1],s[bs],s[bs+1]
                const size_t bx = (n / 128) * sizeof(block_pq2_0), by = (n / 256) * sizeof(block_q8_K);
                std::vector<uint8_t> X(2 * bx), Y(2 * by);
                std::memcpy(X.data(), x0.data(), bx); std::memcpy(X.data() + bx, x1.data(), bx);
                std::memcpy(Y.data(), y0.data(), by); std::memcpy(Y.data() + by, y1.data(), by);
                float t[32] = {0};
                cpu->vec_dot((int) n, t, 16, X.data(), bx, Y.data(), by, 2);
                float e[4];
                cpu->vec_dot((int) n, &e[0], 0, x0.data(), 0, y0.data(), 0, 1);
                cpu->vec_dot((int) n, &e[1], 0, x1.data(), 0, y0.data(), 0, 1);
                cpu->vec_dot((int) n, &e[2], 0, x0.data(), 0, y1.data(), 0, 1);
                cpu->vec_dot((int) n, &e[3], 0, x1.data(), 0, y1.data(), 0, 1);
                if (std::memcmp(&t[0], &e[0], 4) || std::memcmp(&t[1], &e[1], 4) ||
                    std::memcmp(&t[16], &e[2], 4) || std::memcmp(&t[17], &e[3], 4)) ++bad2x2;
            }
        }
    }
    CHECK(bad2x2 == 0, "2x2 tile is not bit-identical to single dots in %d cases", bad2x2);
    std::printf("[gemv] PQ2_0 x Q8_K worst relative error vs exact reference: %.3g; 2x2 tile mismatches: %d\n", worst, bad2x2);
}

// ggml graph mul_mat: weights [K x M] PQ2_0, activations [K x B] f32 -> [M x B]
static std::vector<float> run_mul_mat(const std::vector<block_pq2_0> & w, int64_t K, int64_t M,
                                      const std::vector<float> & act, int64_t B, int nthreads) {
    const size_t mem = ggml_tensor_overhead() * 8 + ggml_graph_overhead() + w.size() * sizeof(block_pq2_0) +
                       act.size() * 4 + M * B * 4 + 8 * 1024 * 1024;
    ggml_init_params ip = { mem, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * tw = ggml_new_tensor_2d(ctx, GGML_TYPE_PQ2_0, K, M);
    ggml_tensor * ta = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, B);
    std::memcpy(tw->data, w.data(), ggml_nbytes(tw));
    std::memcpy(ta->data, act.data(), ggml_nbytes(ta));
    ggml_tensor * out = ggml_mul_mat(ctx, tw, ta);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, nthreads);
    std::vector<float> res(M * B);
    std::memcpy(res.data(), out->data, res.size() * 4);
    ggml_free(ctx);
    return res;
}

static void test_mul_mat() {
    const ggml_type_traits * tt = ggml_get_type_traits(GGML_TYPE_PQ2_0);
    struct Case { int64_t K, M; } cases[] = { {4096, 64}, {640, 32}, {12288, 16}, {384, 18} };
    for (auto c : cases) {
        auto w = random_pq2(c.K * c.M);
        std::vector<float> wf(c.K * c.M);
        tt->to_float(w.data(), wf.data(), c.K * c.M);
        for (int64_t B : {1, 2, 7, 8, 32}) {
            auto act = random_act(c.K * B);
            auto res = run_mul_mat(w, c.K, c.M, act, B, 4);
            // accuracy vs float reference (activation quantization error only)
            double num = 0.0, den = 0.0;
            for (int64_t b = 0; b < B; ++b) for (int64_t m = 0; m < c.M; ++m) {
                double r = 0.0;
                for (int64_t k = 0; k < c.K; ++k) r += (double) wf[m * c.K + k] * act[b * c.K + k];
                const double d = res[b * c.M + m] - r;
                num += d * d; den += r * r;
            }
            const double nmse = num / (den + 1e-30);
            CHECK(nmse < 2e-4, "mul_mat K=%" PRId64 " M=%" PRId64 " B=%" PRId64 " NMSE %.3g", c.K, c.M, B, nmse);
            // every column of a batched product must equal the single-token product bit-for-bit
            int mism = 0;
            for (int64_t b = 0; b < B; ++b) {
                std::vector<float> col(act.begin() + b * c.K, act.begin() + (b + 1) * c.K);
                auto r1 = run_mul_mat(w, c.K, c.M, col, 1, 3);
                if (std::memcmp(r1.data(), res.data() + b * c.M, c.M * 4) != 0) ++mism;
            }
            CHECK(mism == 0, "mul_mat K=%" PRId64 " B=%" PRId64 ": %d columns differ from single-token result", c.K, B, mism);
            std::printf("[mul_mat] K=%-5" PRId64 " M=%-3" PRId64 " B=%-2" PRId64 " NMSE=%.2e batched==single: %s\n",
                        c.K, c.M, B, nmse, mism == 0 ? "yes" : "NO");
        }
    }
}

static void bench() {
#if defined(__ARM_NEON) && defined(__aarch64__)
    const auto * cpu = ggml_get_type_traits_cpu(GGML_TYPE_PQ2_0);
    const auto * q8k = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    const int64_t K = 4096, M = 1024; // 1024 rows of a 4096-wide matrix ~ 4.4 MB of PQ2_0 weights
    auto w = random_pq2(K * M);
    auto a = random_act(K);
    std::vector<block_q8_K> y(K / 256);
    q8k->from_float(a.data(), y.data(), K);
    std::vector<float> out(M);
    const size_t row = (K / 128) * sizeof(block_pq2_0);
    auto time_it = [&](auto fn) {
        double best = 1e30;
        for (int rep = 0; rep < 15; ++rep) {
            auto t0 = std::chrono::steady_clock::now();
            for (int64_t m = 0; m < M; ++m) fn(m);
            auto t1 = std::chrono::steady_clock::now();
            best = std::min(best, std::chrono::duration<double>(t1 - t0).count());
        }
        return best;
    };
    const double t_old = time_it([&](int64_t m) { upstream_vec_dot_pq2_0_q8_K((int) K, &out[m], (const char *) w.data() + m * row, y.data()); });
    const double t_new = time_it([&](int64_t m) { cpu->vec_dot((int) K, &out[m], 0, (const char *) w.data() + m * row, 0, y.data(), 0, 1); });
    const double gw_old = (double) (K * M) / t_old / 1e9, gw_new = (double) (K * M) / t_new / 1e9;
    std::printf("[bench] single-thread GEMV 4096x1024 PQ2_0 (this CPU, not Snapdragon X unless stated):\n");
    std::printf("[bench]   upstream kernel: %8.3f ms  (%.2f Gweights/s)\n", t_old * 1e3, gw_old);
    std::printf("[bench]   bonsai kernel  : %8.3f ms  (%.2f Gweights/s)  speedup x%.2f\n", t_new * 1e3, gw_new, t_old / t_new);

    // quantize_row_q8_K
    auto big = random_act(12288);
    std::vector<block_q8_K> qb(12288 / 256);
    auto tq = [&](auto fn) { double best = 1e30; for (int r = 0; r < 50; ++r) { auto t0 = std::chrono::steady_clock::now(); fn(); auto t1 = std::chrono::steady_clock::now(); best = std::min(best, std::chrono::duration<double>(t1 - t0).count()); } return best; };
    const double q_ref = tq([&] { quantize_row_q8_K_ref(big.data(), qb.data(), 12288); });
    const double q_new = tq([&] { q8k->from_float(big.data(), qb.data(), 12288); });
    std::printf("[bench] quantize_row_q8_K 12288: ref %.2f us, NEON %.2f us, speedup x%.2f\n", q_ref * 1e6, q_new * 1e6, q_ref / q_new);
#endif
}

int main(int argc, char ** argv) {
    const bool do_bench = argc > 1 && std::string(argv[1]) == "--bench";
    ggml_cpu_init();
    test_runtime_detection();
    test_decode();
    test_q8K_quantize();
    test_vec_dot_q8K();
    test_mul_mat();
    if (do_bench) bench();
    std::printf("\n%d/%d checks passed\n", g_checks - g_failures, g_checks);
    return g_failures == 0 ? 0 : 1;
}
