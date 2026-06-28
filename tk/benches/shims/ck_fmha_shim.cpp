// CK ck_tile FMHA-forward (bf16) shim for the svod-tk `vendor` fa bench arm.
//
// Wraps Composable Kernel's host `fmha_fwd` dispatcher (which does runtime
// device-name / CU / tile selection across the codegen'd instances) behind a
// tiny C ABI, timed with HIP events so it returns on-device nanoseconds —
// comparable to svod's per-kernel PM4 stamps. q/k/v/o are svod device VAs in
// tk's `[B,T,H,d]` (iperm=0) layout; the shim fills the iperm=0 strides
// (mirroring example/ck_tile/01_fmha/fmha_fwd_runner.hpp) and the 1/sqrt(d)
// softmax scale. A `<0` return from `fmha_fwd` means "no instance for this
// shape" → the shim returns -1 and the bench skips the row.
//
// Built by the flake's `ckFmhaShim` derivation (compiles + links the gfx1151
// instances); loaded via libloading from benches/vendor.rs. Not cargo-built.

#include "ck_tile/host.hpp"
#include "fmha_fwd.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <vector>

extern "C" double ck_fmha_fwd_bf16_run_ns(const void* q, const void* k, const void* v, void* o, int batch, int nhead,
                                          int seqlen, int hdim, int causal, int warmup, int iters);

namespace {
inline bool ok(hipError_t s) { return s == hipSuccess; }
} // namespace

double ck_fmha_fwd_bf16_run_ns(const void* q, const void* k, const void* v, void* o, int batch, int nhead, int seqlen,
                               int hdim, int causal, int warmup, int iters) {
    // traits: bf16, batch mode, no bias / lse / dropout / quant / sink.
    fmha_fwd_traits traits{};
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
    traits.qscale_type = quant_scale_enum::no_scale;
    traits.skip_min_seqlen_q = false;
    traits.has_sink = false;

    // args: contiguous [B,T,H,d] (iperm=0), nhead_q == nhead_k, seqlen_q == seqlen_k.
    fmha_fwd_args args{};
    args.q_ptr = q;
    args.k_ptr = k;
    args.v_ptr = v;
    args.o_ptr = o;
    args.seqlen_q = seqlen;
    args.seqlen_k = seqlen;
    args.batch = batch;
    args.max_seqlen_q = seqlen;
    args.hdim_q = hdim;
    args.hdim_v = hdim;
    args.nhead_q = nhead;
    args.nhead_k = nhead;
    args.num_head_q_total = nhead;
    args.head_start = 0;
    args.scale_s = 1.0f / std::sqrt(static_cast<float>(hdim));
    args.logits_soft_cap = 0.0f;

    // [B,T,H,d] strides: element (b,s,h,e) at ((b*S + s)*H + h)*d + e.
    args.stride_q = nhead * hdim; // step over seqlen
    args.stride_k = nhead * hdim;
    args.stride_v = nhead * hdim; // is_v_rowmajor
    args.stride_o = nhead * hdim;
    args.nhead_stride_q = hdim;
    args.nhead_stride_k = hdim;
    args.nhead_stride_v = hdim;
    args.nhead_stride_o = hdim;
    args.batch_stride_q = nhead * seqlen * hdim;
    args.batch_stride_k = nhead * seqlen * hdim;
    args.batch_stride_v = nhead * seqlen * hdim;
    args.batch_stride_o = nhead * seqlen * hdim;

    args.window_size_left = -1; // full attention (no SWA)
    args.window_size_right = causal ? 0 : -1;
    args.sink_size = 0;
    args.mask_type = static_cast<ck_tile::index_t>(traits.mask_type);
    args.min_seqlen_q = 0;
    args.p_drop = 0.0f;
    args.s_randval = false;

    // Own HIP stream + events; CK's stream_config timing stays off (time_kernel_=false).
    hipStream_t stream = nullptr;
    hipEvent_t start = nullptr, stop = nullptr;
    double result = -1.0;
    if (ok(hipStreamCreate(&stream)) && ok(hipEventCreate(&start)) && ok(hipEventCreate(&stop))) {
        ck_tile::stream_config sc{stream};
        bool good = true;
        for (int i = 0; i < warmup && good; ++i) {
            good = fmha_fwd(traits, args, sc) >= 0.0f; // <0 == no matching instance
        }
        good = good && ok(hipStreamSynchronize(stream));

        std::vector<double> samples;
        samples.reserve(iters > 0 ? static_cast<size_t>(iters) : 0);
        for (int i = 0; i < iters && good; ++i) {
            float ms = 0.0f;
            good = ok(hipEventRecord(start, stream)) && fmha_fwd(traits, args, sc) >= 0.0f
                   && ok(hipEventRecord(stop, stream)) && ok(hipEventSynchronize(stop))
                   && ok(hipEventElapsedTime(&ms, start, stop));
            if (good) {
                samples.push_back(static_cast<double>(ms) * 1.0e6); // ms -> ns
            }
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
