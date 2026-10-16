// bonsai-diag: decide whether single-token decoding on this machine is limited by memory bandwidth or by the
// PQ2_0 GEMV kernel, without the model. Uses only the public ggml backend API, so the CPU module is chosen at
// run time exactly like in bonsai-cli/bonsai-server (no machine-specific instructions in this binary).
//
//   bonsai-diag [max_threads] [--quick]
//
// 1. raw DRAM read bandwidth with 1..N threads (1 GiB buffer, streaming 64-bit sums)
// 2. tg-shaped PQ2_0 mul_mat (one activation column) on a 4096 x 65536 matrix (71 MB, beyond any cache),
//    plain layout vs Bonsai-Oryon 4-row layout (ternary layout with BONSAI_PQ2_TERNARY=1), 1..N threads,
//    reported as GB/s of PQ2_0-equivalent weights (2.125 bpw) streamed
// 3. what that implies for Ternary-Bonsai-2-9B (about 1.97 GB of PQ2_0 weights read per generated token)
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

using clk = std::chrono::steady_clock;
static double secs(clk::time_point a, clk::time_point b) { return std::chrono::duration<double>(b - a).count(); }

static constexpr double kModelBytesPerToken = 1.97e9;   // PQ2_0 weights read per token (all but token_embd)

static double mem_read_bw(const uint64_t * buf, size_t n, int nt) {
    std::vector<uint64_t> sink(nt * 8);
    double best = 0;
    for (int rep = 0; rep < 5; rep++) {
        std::atomic<int> ready{0};
        std::vector<std::thread> th;
        auto t0 = clk::now();
        for (int t = 0; t < nt; t++) {
            th.emplace_back([&, t] {
                const size_t a = n * t / nt, b = n * (t + 1) / nt;
                uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                for (size_t i = a; i + 4 <= b; i += 4) { s0 += buf[i]; s1 += buf[i + 1]; s2 += buf[i + 2]; s3 += buf[i + 3]; }
                sink[t * 8] = s0 + s1 + s2 + s3;
            });
        }
        for (auto & x : th) x.join();
        const double dt = secs(t0, clk::now());
        best = std::max(best, n * 8.0 / dt / 1e9);
    }
    volatile uint64_t keep = sink[0]; (void) keep;
    return best;
}

int main(int argc, char ** argv) {
    int max_threads = (int) std::thread::hardware_concurrency();
    bool quick = false;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--quick") quick = true; else max_threads = std::max(1, atoi(argv[i]));
    }
    ggml_backend_load_all();
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("CPU");
    if (!reg) { fprintf(stderr, "CPU backend not found\n"); return 1; }
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
    printf("bonsai-diag: CPU backend '%s', %d hardware threads\n", ggml_backend_dev_description(dev), (int) std::thread::hardware_concurrency());

    // --- 1. memory bandwidth
    const size_t n64 = (quick ? 256u : 1024u) * 1024 * 1024 / 8;
    std::vector<uint64_t> buf(n64);
    for (size_t i = 0; i < n64; i++) buf[i] = i * 0x9E3779B97F4A7C15ull;
    std::vector<int> tcounts;
    for (int t : {1, 2, 4, 6, 7, 8, 12, 16}) if (t <= max_threads) tcounts.push_back(t);
    if (std::find(tcounts.begin(), tcounts.end(), max_threads) == tcounts.end()) tcounts.push_back(max_threads);
    printf("\n[memory] streaming read bandwidth (%zu MiB buffer)\n", n64 * 8 >> 20);
    std::vector<double> bw(tcounts.size());
    for (size_t k = 0; k < tcounts.size(); k++) {
        bw[k] = mem_read_bw(buf.data(), n64, tcounts[k]);
        printf("[memory]   threads=%-2d %7.1f GB/s   -> tg ceiling for the 9B PQ2_0 model if purely bandwidth-bound: %5.1f tok/s\n",
               tcounts[k], bw[k], bw[k] * 1e9 / kModelBytesPerToken);
    }
    std::vector<uint64_t>().swap(buf);

    // --- 2. PQ2_0 GEMV through the backend
    auto get_extra = (ggml_backend_dev_get_extra_bufts_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");
    auto set_threads = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    ggml_backend_buffer_type_t repack = nullptr;
    if (get_extra) for (auto * b = get_extra(dev); b && *b; ++b) if (std::string(ggml_backend_buft_name(*b)) == "CPU_REPACK") repack = *b;

    const int64_t K = 4096, M = quick ? 16384 : 65536;
    const size_t wbytes = (size_t) K * M / 128 * 34;
    std::vector<uint8_t> wq(wbytes);
    std::mt19937 rng(7);
    for (size_t i = 0; i < wbytes; i += 34) {
        const uint16_t d = 0x2000 + (rng() & 0x3ff);           // fp16 scale ~0.008
        memcpy(&wq[i], &d, 2);
        for (int j = 2; j < 34; j++) {   // ternary codes {0,1,2}, like Ternary-Bonsai, so either 4-row layout applies
            uint8_t q = 0; for (int k = 0; k < 4; k++) q |= (uint8_t) ((rng() % 3) << (2 * k));
            wq[i + j] = q;
        }
    }
    struct W { ggml_context * ctx; ggml_backend_buffer_t buf; ggml_tensor * t; };
    auto make_w = [&](ggml_backend_buffer_type_t bt) {
        W w; ggml_init_params ip = { ggml_tensor_overhead() * 2, nullptr, true };
        w.ctx = ggml_init(ip); w.t = ggml_new_tensor_2d(w.ctx, GGML_TYPE_PQ2_0, K, M);
        w.buf = ggml_backend_alloc_ctx_tensors_from_buft(w.ctx, bt);
        // a repack buffer without a matching layout leaves extra == NULL and must not receive data
        if (bt == ggml_backend_dev_buffer_type(dev) || w.t->extra) ggml_backend_tensor_set(w.t, wq.data(), 0, wbytes);
        return w;
    };
    std::vector<std::pair<std::string, W>> layouts;
    layouts.push_back({"plain (v0.1)", make_w(ggml_backend_dev_buffer_type(dev))});
    if (repack) {
        W r = make_w(repack);
        const char * te = getenv("BONSAI_PQ2_TERNARY");
        if (r.t->extra) layouts.push_back({te && te[0] == '1' ? "4-row ternary (opt-in)" : "4-row (v0.2)", r});
        else { printf("\n(4-row layout not available on this CPU/build)\n"); ggml_backend_buffer_free(r.buf); ggml_free(r.ctx); }
    }
    std::vector<float> act(K);
    for (auto & a : act) a = std::normal_distribution<float>(0, 1)(rng);

    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
    printf("\n[gemv] tg-shaped PQ2_0 mul_mat, %lld x %lld (%.0f MB of weights), best of %d\n", (long long) K, (long long) M, wbytes / 1e6, quick ? 5 : 15);
    printf("[gemv]   %-8s", "threads");
    for (auto & l : layouts) printf(" | %-28s", l.first.c_str());
    printf(" | DRAM read\n");
    for (size_t k = 0; k < tcounts.size(); k++) {
        if (set_threads) set_threads(be, tcounts[k]);
        printf("[gemv]   %-8d", tcounts[k]);
        for (auto & l : layouts) {
            ggml_init_params ip = { ggml_tensor_overhead() * 4 + ggml_graph_overhead() + K * 4 + M * 4 + 65536, nullptr, false };
            ggml_context * ctx = ggml_init(ip);
            ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, K);
            memcpy(a->data, act.data(), K * 4);
            ggml_tensor * out = ggml_mul_mat(ctx, l.second.t, a);
            ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, out);
            ggml_backend_graph_compute(be, gf);
            double best = 1e30;
            for (int r = 0; r < (quick ? 5 : 15); r++) {
                auto t0 = clk::now(); ggml_backend_graph_compute(be, gf); best = std::min(best, secs(t0, clk::now()));
            }
            const double gbs = wbytes / best / 1e9;
            printf(" | %6.1f GB/s = %5.1f tok/s eq.", gbs, gbs * 1e9 / kModelBytesPerToken);
            ggml_free(ctx);
        }
        printf(" | %6.1f GB/s\n", bw[k]);
    }
    printf("\nInterpretation: if the 4-row GEMV GB/s is close to the DRAM read GB/s at the same thread count, decoding is\n"
           "memory-bound on this machine and only fewer bytes per token can speed it up. If it is well below, the kernel\n"
           "or threading is the limit. 'tok/s eq.' is the matmul-only ceiling for the 9B model; real tg is lower because\n"
           "attention/delta-net, norms and synchronization add time.\n");
    ggml_backend_free(be);
    for (auto & l : layouts) { ggml_backend_buffer_free(l.second.buf); ggml_free(l.second.ctx); }
    return 0;
}
