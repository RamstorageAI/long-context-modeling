import triton
import torch
from ltriton.lsa import attention as lsa_attn
from native_sparse_attention.ops.parallel import parallel_nsa
from einops import rearrange

try:
    from flash_attn.flash_attn_interface import \
        flash_attn_func as flash_attn_func
    HAS_FLASH = True
except BaseException:
    HAS_FLASH = False

BATCH, N_HEADS, HEAD_DIM = 8, 16, 64
# vary seq length for fixed head and batch=4
configs = []
for mode in ["fwd", "bwd"]:
    for causal in [False]:
        configs.append(
            triton.testing.Benchmark(
                x_names=["N_CTX"],
                x_vals=[2**i for i in range(10, 17)],
                line_arg="provider",
                line_vals=["gca-fp16"] + ["nsa-fp16"] + ["flash"],
                line_names=["gca-fp16"] + ["nsa-fp16"] + ["flash"],
                styles=[("red", "-"), ("blue", "-"), ("green", "-")],
                ylabel="TFLOPS",
                plot_name=f"fused-attention-batch{BATCH}-head{N_HEADS}-d{HEAD_DIM}-{mode}-causal={causal}",
                args={
                    "H": N_HEADS,
                    "BATCH": BATCH,
                    "HEAD_DIM": HEAD_DIM,
                    "mode": mode,
                    "causal": causal,
                },
            ))

@triton.testing.perf_report(configs)
def bench_flash_attention(BATCH, H, N_CTX, HEAD_DIM, causal, mode, provider, device="cuda"):
    import torch.nn.functional as F
    assert mode in ["fwd", "bwd"]
    warmup = 25
    rep = 500
    dtype = torch.bfloat16
    K = 8
    S = 64
    # Q_CTX, K_CTX = N_CTX, N_CTX // K
    ms = 1
    times = 3
    if "gca" in provider:
        KV_CTX = (N_CTX // S) * S
        q = torch.randn((BATCH, N_CTX, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((BATCH, KV_CTX, H // 16, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((BATCH, KV_CTX, H // 16, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        weights = F.softmax(torch.randn((BATCH, N_CTX, 1, min(K, N_CTX // S)), dtype=dtype, device=device, requires_grad=True), dim=-1)
        k_mean = torch.randn((BATCH, N_CTX // S, H * HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        _q = rearrange(q, 'B L h d->B L (h d)')
        # indices = torch.randint(low=0, high=N_CTX // S, size=(BATCH, N_CTX, K), device=device)
        # if mode == "fwd" and "fp8" in provider:
        #     q = q.to(torch.float8_e5m2)
        #     k = k.to(torch.float8_e5m2)
        #     v = v.permute(0, 1, 2, 4, 3).contiguous()
        #     v = v.permute(0, 1, 2, 4, 3)
        #     v = v.to(torch.float8_e5m2)
        sm_scale = 1.3
        def fn():
            attn_scores = torch.einsum('B L d, B C d->B L C', _q, k_mean)
            _, indices = torch.topk(attn_scores, k=min(K, N_CTX // S), dim=-1)
            indices = indices.unsqueeze(-2)
            # print(indices.shape)
            # indices = indices.contiguous()
            h = q
            for _ in range(times):
                h = lsa_attn(h, k, v, weights, indices, 1.0, S, sm_scale)
            return h
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    if "nsa" in provider:
        KV_CTX = (N_CTX // S) * S
        q = torch.randn((BATCH, N_CTX, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((BATCH, KV_CTX, H // 16, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((BATCH, KV_CTX, H // 16, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        def fn():
            h = q
            for _ in range(times):
                h =  parallel_nsa(
                            q=h.to(q.dtype),
                            k=k,
                            v=v,
                            g_cmp=torch.ones((BATCH, N_CTX, H), device=device),
                            g_slc=torch.ones((BATCH, N_CTX, H), device=device),
                            g_swa=torch.ones((BATCH, N_CTX, H), device=device),
                            block_size=S,
                            block_counts=min(K, N_CTX // S),
                            window_size=0,
                            cu_seqlens=None,
                            head_first=False)
            return h
        if mode == 'bwd':
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    if provider == "flash":
        # qkv = torch.randn((BATCH, N_CTX, 3, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        q = torch.randn((BATCH, N_CTX, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((BATCH, N_CTX, H // 16, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((BATCH, N_CTX, H // 16, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        def fn():
            h = q
            for _ in range(times):
                h = flash_attn_func(h, k, v, causal=causal)
            return h
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    # flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
    # total_flops = 2 * flops_per_matmul
    # if causal:
    #     total_flops *= 0.5
    # if mode == "bwd":
    #     total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
    # return total_flops * 1e-12 / (ms * 1e-3)
    return ms
  
if __name__ == "__main__":
    # only works on post-Ampere GPUs right now
    bench_flash_attention.run(save_path=".", print_data=True)