# ruff: noqa
import torch
from ops.hsa_triton import HSA
import tilelang
from tilelang import language as T
import tilelang.testing

tilelang.testing.set_random_seed(0)


@tilelang.jit(
    out_idx=[-1], pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    })
def hierarchical_sparse_attention(batch,
                                  heads,
                                  q_len,
                                  kv_len,
                                  head_dim,
                                  scale=None,
                                  block_size=64,
                                  groups=16,
                                  selected_blocks=16):
    if scale is None:
        scale = (1.0 / head_dim)**0.5 * 1.44269504  # log2(e)
    else:
        scale = scale * 1.44269504  # log2(e)

    head_kv = heads // groups
    q_shape = [batch, q_len, heads, head_dim]
    kv_shape = [batch, kv_len, head_kv, head_dim]
    weight_shape = [batch, q_len, head_kv, selected_blocks]
    block_indices_shape = [batch, q_len, head_kv, selected_blocks]
    block_indices_dtype = "int32"
    dtype = "bfloat16"
    accum_dtype = "float"
    block_S = block_size
    block_T = min(128, tilelang.math.next_power_of_2(head_dim))

    NK = tilelang.cdiv(head_dim, block_T)
    NV = tilelang.cdiv(head_dim, block_T)
    assert NK == 1, "The key dimension can not be larger than 256"

    S = selected_blocks
    G = groups
    BS = block_S
    BK = BV = block_T
    num_stages = 2
    threads = 32

    @T.prim_func
    def hsa(
            Q: T.Tensor(q_shape, dtype),
            K: T.Tensor(kv_shape, dtype),
            V: T.Tensor(kv_shape, dtype),
            W: T.Tensor[weight_shape, dtype],
            BlockIndices: T.Tensor(block_indices_shape, block_indices_dtype),
            Output: T.Tensor(q_shape, dtype),
    ):
        with T.Kernel(q_len, NV, batch * head_kv, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([G, BK], dtype)
            K_shared = T.alloc_shared([BS, BK], dtype)
            V_shared = T.alloc_shared([BS, BV], dtype)
            O_shared = T.alloc_shared([G, BV], dtype)

            acc_s = T.alloc_fragment([G, BS], accum_dtype)
            acc_s_cast = T.alloc_fragment([G, BS], dtype)
            acc_o = T.alloc_fragment([G, BV], accum_dtype)
            scores_max = T.alloc_fragment([G], accum_dtype)
            # scores_max_prev = T.alloc_fragment([G], accum_dtype)
            # scores_scale = T.alloc_fragment([G], accum_dtype)            
            scores_sum = T.alloc_fragment([G], accum_dtype)
            # logsum = T.alloc_fragment([G], accum_dtype)

            i_t, i_v, i_bh = bx, by, bz
            i_b, i_h = i_bh // head_kv, i_bh % head_kv

            NS = S
            T.copy(Q[i_b, i_t, i_h * G:(i_h + 1) * G, :], Q_shared)


            T.fill(acc_o, 0)
            # T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            for i in T.Pipelined(NS, num_stages=num_stages):
                blk_idx = BlockIndices[i_b, i_t, i_h, i]
                i_s = blk_idx * BS
                if i_s >= 0:
                    # [BS, BK]
                    T.copy(K[i_b, i_s:i_s + BS, i_h, :], K_shared)
                    # T.copy(W[i_b, i_t, i_h, blk_idx], W_shared)
                    chunk_weight = W[i_b, i_t, i_h, i]

                    T.clear(acc_s)

                    T.gemm(
                        Q_shared,
                        K_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow)

                    # Softmax
                    # T.copy(scores_max, scores_max_prev)
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.reduce_max(acc_s, scores_max, dim=1, clear=True)

                    for i, j in T.Parallel(G, BS):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for i, j in T.Parallel(G, BS):
                        acc_s[i, j] = chunk_weight * acc_s[i, j] / scores_sum[i]
                    T.copy(acc_s, acc_s_cast)

                    # V * softmax(Q * K)
                    T.copy(V[i_b, i_s:i_s + BS, i_h, i_v * BV:(i_v + 1) * BV], V_shared)
                    T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output[i_b, i_t, i_h * G:(i_h + 1) * G, i_v * BV:(i_v + 1) * BV])

    return hsa


def main():
    B, SEQ_LEN, H, HQ, D, S, block_size, dtype, scale = 4, 4096, 2, 32, 64, 16, 64, torch.bfloat16, None

    kernel = hierarchical_sparse_attention(
        batch=B,
        heads=HQ,
        q_len=SEQ_LEN,
        kv_len=SEQ_LEN,
        head_dim=D,
        block_size=block_size,
        groups=HQ // H,
        selected_blocks=S,
        scale=scale,
    )
    # print(kernel.get_kernel_source())
    torch.random.manual_seed(0)
    Q = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device='cuda').requires_grad_(True)
    K = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device='cuda').requires_grad_(True)
    V = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device='cuda').requires_grad_(True)
    W = torch.randn((B, SEQ_LEN, H, S), dtype=dtype, device='cuda').requires_grad_(True)
    import torch.nn.functional as F
    W = F.softmax(W, dim=-1)
    DO = torch.randn((B, SEQ_LEN, H, S), dtype=dtype, device='cuda')

    block_indices = torch.full((B, SEQ_LEN, H, S), 0, dtype=torch.long, device='cuda')
    for b in range(B):
        for t in range(SEQ_LEN):
            for h in range(H):
                i_i = torch.randperm(max(1, (t // block_size)))[:S]
                block_indices[b, t, h, :len(i_i)] = i_i
    block_indices = block_indices.sort(-1)[0]
    # print(block_indices)
    block_counts = torch.randint(1, S + 1, (B, SEQ_LEN, H), device='cuda')
    block_indices = block_indices.to(torch.int32)

    import time
    for _ in range(10):
        out = kernel(Q, K, V, W, block_indices)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(1000):
        out = kernel(Q, K, V, W, block_indices)
    torch.cuda.synchronize()
    end = time.time()
    tilelang_time = end - start


    for _ in range(10):
        ref = HSA(
            Q, K, V, W, block_indices, 0.0, block_size, scale, 0.0, 0.0
        )
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(1000):
        ref = HSA(
            Q, K, V, W, block_indices, 0.0, block_size, scale, 0.0, 0.0
        )
    torch.cuda.synchronize()
    end = time.time()
    triton_time = end - start
    print(f'tilelang: {tilelang_time} vs triton: {triton_time}')

    # print("out", out)
    # print("ref", ref)
    torch.testing.assert_close(ref, out, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    main()