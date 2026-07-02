// CK ck_tile FMHA-forward (bf16) vendor floor for the svod-tk `vendor` fa bench arm.
//
// Wraps Composable Kernel's host `fmha_fwd` dispatcher (runtime device/CU/tile selection
// over the codegen'd instances) behind a tiny C ABI, timed with HIP events (on-device ns,
// comparable to svod's PM4 stamps). Like the hipBLASLt shim, this SELF-ALLOCATES its q/k/v/o
// device buffers (svod is KFD-direct, so its VAs aren't valid in HIP; the vendor bench passes
// only shapes). Supports **GQA** (`nhead_kv < nhead`), the tk `[B,T,H,d]` (iperm=0) layout,
// and the 1/sqrt(d) softmax scale. `create` returns null when CK has no instance for the shape
// (probe `fmha_fwd` < 0) → the bench skips that row.
//
// Built by the flake's `ckFmhaShim` derivation (compiles + links the gfx942/gfx1151 instances);
// loaded via libloading from benches/vendor.rs. Not cargo-built.

#include "ck_tile/host.hpp"
#include "fmha_fwd.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

extern "C" {
void* ck_fmha_create(int batch, int nhead, int nhead_kv, int seqlen, int hdim, int causal);
double ck_fmha_run_ns(void* plan, int warmup, int iters);
void ck_fmha_destroy(void* plan);
}

namespace {
inline bool ok(hipError_t s) { return s == hipSuccess; }

struct FaPlan {
    void* q = nullptr;
    void* k = nullptr;
    void* v = nullptr;
    void* o = nullptr;
    fmha_fwd_traits traits{};
    fmha_fwd_args args{};
    bool valid = false;
};
} // namespace

// Allocate + configure a batched FMHA-forward. `nhead` = query heads; `nhead_kv` = key/value
// heads (GQA: nhead_kv ≤ nhead, nhead % nhead_kv == 0). bf16 in/out, `[B,T,H,d]` layout.
void* ck_fmha_create(int batch, int nhead, int nhead_kv, int seqlen, int hdim, int causal) {
    auto* p = new FaPlan();
    const size_t qo_bytes = static_cast<size_t>(batch) * seqlen * nhead * hdim * 2;    // bf16
    const size_t kv_bytes = static_cast<size_t>(batch) * seqlen * nhead_kv * hdim * 2; // bf16
    if (!ok(hipMalloc(&p->q, qo_bytes)) || !ok(hipMalloc(&p->o, qo_bytes)) || !ok(hipMalloc(&p->k, kv_bytes))
        || !ok(hipMalloc(&p->v, kv_bytes))) {
        ck_fmha_destroy(p);
        return nullptr;
    }
    // Fill inputs with a small finite bf16 (byte 0x3c → ~1.6e-2) so softmax is well-behaved.
    hipMemset(p->q, 0x3c, qo_bytes);
    hipMemset(p->k, 0x3c, kv_bytes);
    hipMemset(p->v, 0x3c, kv_bytes);

    fmha_fwd_traits& traits = p->traits;
    traits.hdim_q = hdim;
    traits.hdim_v = hdim;
    traits.data_type = "bf16";
    traits.is_group_mode = false;
    traits.is_v_rowmajor = true;
    traits.has_logits_soft_cap = false;
    traits.mask_type = causal ? mask_enum::mask_top_left : mask_enum::no_mask;
    traits.bias_type = bias_enum::no_bias;
    traits.has_lse = false;
    traits.has_dropout = false;
    traits.do_fp8_static_quant = false;
    traits.skip_min_seqlen_q = false;

    fmha_fwd_args& args = p->args;
    args.q_ptr = p->q;
    args.k_ptr = p->k;
    args.v_ptr = p->v;
    args.o_ptr = p->o;
    args.seqlen_q = seqlen;
    args.seqlen_k = seqlen;
    args.batch = batch;
    args.max_seqlen_q = seqlen;
    args.hdim_q = hdim;
    args.hdim_v = hdim;
    args.nhead_q = nhead;    // GQA: query heads
    args.nhead_k = nhead_kv; // GQA: key/value heads (CK maps q-head → kv-head by the ratio)
    args.scale_s = 1.0f / std::sqrt(static_cast<float>(hdim));
    args.logits_soft_cap = 0.0f;

    // [B,T,H,d] strides (element (b,s,h,e) at ((b*S + s)*H + h)*d + e). Seqlen step = H·d
    // (H differs for q/o vs k/v under GQA); head step = d; batch step = H·S·d.
    args.stride_q = nhead * hdim;
    args.stride_k = nhead_kv * hdim;
    args.stride_v = nhead_kv * hdim; // is_v_rowmajor
    args.stride_o = nhead * hdim;
    args.nhead_stride_q = hdim;
    args.nhead_stride_k = hdim;
    args.nhead_stride_v = hdim;
    args.nhead_stride_o = hdim;
    args.batch_stride_q = static_cast<ck_tile::index_t>(nhead) * seqlen * hdim;
    args.batch_stride_k = static_cast<ck_tile::index_t>(nhead_kv) * seqlen * hdim;
    args.batch_stride_v = static_cast<ck_tile::index_t>(nhead_kv) * seqlen * hdim;
    args.batch_stride_o = static_cast<ck_tile::index_t>(nhead) * seqlen * hdim;

    args.window_size_left = -1;
    args.window_size_right = causal ? 0 : -1;
    args.mask_type = static_cast<ck_tile::index_t>(traits.mask_type);
    args.min_seqlen_q = 0;
    args.p_drop = 0.0f;
    args.s_randval = false;

    // Probe: one fwd (default stream, timing off). <0 == no matching instance for this shape.
    ck_tile::stream_config sc{nullptr};
    if (fmha_fwd(p->traits, p->args, sc) < 0.0f || !ok(hipDeviceSynchronize())) {
        ck_fmha_destroy(p);
        return nullptr;
    }
    p->valid = true;
    return p;
}

// Median device ns over `iters` fmha_fwd launches (after `warmup`), HIP-event timed.
double ck_fmha_run_ns(void* plan, int warmup, int iters) {
    auto* p = static_cast<FaPlan*>(plan);
    if (!p || !p->valid) return -1.0;

    hipStream_t stream = nullptr;
    hipEvent_t start = nullptr, stop = nullptr;
    double result = -1.0;
    if (ok(hipStreamCreate(&stream)) && ok(hipEventCreate(&start)) && ok(hipEventCreate(&stop))) {
        ck_tile::stream_config sc{stream};
        bool good = true;
        for (int i = 0; i < warmup && good; ++i) {
            good = fmha_fwd(p->traits, p->args, sc) >= 0.0f;
        }
        good = good && ok(hipStreamSynchronize(stream));

        std::vector<double> samples;
        samples.reserve(iters > 0 ? static_cast<size_t>(iters) : 0);
        for (int i = 0; i < iters && good; ++i) {
            float ms = 0.0f;
            good = ok(hipEventRecord(start, stream)) && fmha_fwd(p->traits, p->args, sc) >= 0.0f
                   && ok(hipEventRecord(stop, stream)) && ok(hipEventSynchronize(stop))
                   && ok(hipEventElapsedTime(&ms, start, stop));
            if (good) samples.push_back(static_cast<double>(ms) * 1.0e6); // ms -> ns
        }
        if (good) {
            if (samples.empty()) {
                result = 0.0;
            } else {
                std::sort(samples.begin(), samples.end());
                result = samples[samples.size() / 2];
            }
        }
    }
    if (stop) hipEventDestroy(stop);
    if (start) hipEventDestroy(start);
    if (stream) hipStreamDestroy(stream);
    return result;
}

void ck_fmha_destroy(void* plan) {
    auto* p = static_cast<FaPlan*>(plan);
    if (!p) return;
    if (p->q) hipFree(p->q);
    if (p->k) hipFree(p->k);
    if (p->v) hipFree(p->v);
    if (p->o) hipFree(p->o);
    delete p;
}
