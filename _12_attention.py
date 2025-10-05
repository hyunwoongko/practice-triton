import math
import time

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"block_size_q": 64, "tile_size_kv": 64, "num_warps": 4, "num_stages": 2}),
        triton.Config({"block_size_q": 128, "tile_size_kv": 64, "num_warps": 8, "num_stages": 2}),
        triton.Config({"block_size_q": 64, "tile_size_kv": 128, "num_warps": 8, "num_stages": 2}),
        triton.Config({"block_size_q": 32, "tile_size_kv": 64, "num_warps": 4, "num_stages": 2}),
        triton.Config({"block_size_q": 64, "tile_size_kv": 32, "num_warps": 4, "num_stages": 2}),
    ],
    key=["seq_len_kv", "dim"]
)
@triton.jit
def flash_attn_kernel_fwd(
    q_ptr, k_ptr, v_ptr, o_ptr,
    bh, seq_len_q, seq_len_kv,
    stride_q_bh, stride_q_seq, stride_q_dim,
    stride_k_bh, stride_k_seq, stride_k_dim,
    stride_v_bh, stride_v_seq, stride_v_dim,
    stride_o_bh, stride_o_seq, stride_o_dim,
    causal: tl.constexpr,
    softmax_scale,
    dim: tl.constexpr,
    block_size_q: tl.constexpr,
    tile_size_kv: tl.constexpr,
):
    """
    Flash-attention kernel은 각 텐서를 어떻게 타일링 할까?
        Attention kernel은 query의 sequence length를 block 레벨로 쪼개서 처리한다.
        즉, 하나의 block이 block_size_q개의 query token을 처리하는 것이다.
        그리고 key/value는 블록 레벨로 쪼개지 않고, 하나의 블록에서 루프로 key/value 전체를 처리한다.

        요약하자면:
            - Query: seq_len 차원에서의 블록 타일링
            - Key/Value: seq_len 차원에서의 루프 타일링

    왜 sequence length 차원으로만 타일링을 할까?
        sequence length는 짧으면 수백, 길면 수만까지 커질 수 있고 이는 하나의 블록에서 처리하기 어렵다.
        그러나 dim 차원은 head 개수로 나뉘어 있기 때문에 그다지 크지 않다.
        Qwen3-32B 기준으로 dim 차원은 5120인데, head 개수 64로 나누면 128밖에 되지 않는다.

    Query original:
        ------------------- dim -------------------
        |------|------|------|------|------|------|  |
        | Q_00 | Q_01 | Q_02 | Q_03 | Q_04 | Q_05 |  | → token0
        |------|------|------|------|------|------|  |
        | Q_10 | Q_11 | Q_12 | Q_13 | Q_14 | Q_15 |  s → token1
        |------|------|------|------|------|------|  e
        | Q_20 | Q_21 | Q_22 | Q_23 | Q_24 | Q_25 |  q → token2
        |------|------|------|------|------|------|  |
        | Q_30 | Q_31 | Q_32 | Q_33 | Q_34 | Q_35 |  l → token3
        |------|------|------|------|------|------|  e
        | Q_40 | Q_41 | Q_42 | Q_43 | Q_44 | Q_45 |  n → token4
        |------|------|------|------|------|------|  |
        | Q_50 | Q_51 | Q_52 | Q_53 | Q_54 | Q_55 |  | → token5
        |------|------|------|------|------|------|  |

    Query blocked (block_size_q=2):
        ------------------- dim -------------------  b
        |------|------|------|------|------|------|  l
        | Q_00 | Q_01 | Q_02 | Q_03 | Q_04 | Q_05 |  o → token0
        |------|------|------|------|------|------|  c
        | Q_10 | Q_11 | Q_12 | Q_13 | Q_14 | Q_15 |  k → token1
        |------|------|------|------|------|------|  1

        ------------------- dim -------------------  b
        |------|------|------|------|------|------|  l
        | Q_20 | Q_21 | Q_22 | Q_23 | Q_24 | Q_25 |  o → token2
        |------|------|------|------|------|------|  c
        | Q_30 | Q_31 | Q_32 | Q_33 | Q_34 | Q_35 |  k → token3
        |------|------|------|------|------|------|  2
                            ...

    Query blocked * Key^T:
        ------------------- dim -------------------  b         |  ---- loop1 ----     ---- loop2 ----
        |------|------|------|------|------|------|  l         |  |------|------|     |------|------|
        | Q_00 | Q_01 | Q_02 | Q_03 | Q_04 | Q_05 |  o         |  | K_00 | K_10 |     | K_20 | K_30 |
        |------|------|------|------|------|------|  c     *   |  |------|------|     |------|------|
        | Q_10 | Q_11 | Q_12 | Q_13 | Q_14 | Q_15 |  k         |  | K_01 | K_11 |     | K_21 | K_31 |
        |------|------|------|------|------|------|  1         |  |------|------|     |------|------|
                                                               d  | K_02 | K_12 |     | K_22 | K_32 |
                                                               i  |------|------|  →  |------|------|  ...
                                                               m  | K_03 | K_13 |     | K_23 | K_33 |
                                                               |  |------|------|     |------|------|
                                                               |  | K_04 | K_14 |     | K_24 | K_34 |
                                                               |  |------|------|     |------|------|
                                                               |  | K_05 | K_15 |     | K_25 | K_35 |
                                                               |  |------|------|     |------|------|
                                                                     ↓      ↓            ↓      ↓
                                                                  token0  token1      token2  token3
    Streaming softmax (Online softmax):
        본래 softmax를 계산하기 위해서는 하나의 Query 토큰에 대한 Key/Value 전체가 필요하다.
        그러나 우리는 Key/Value를 loop로 쪼개서 연산하고 있기 때문에 일반적인 softmax를 사용할 수 없다.
        그래서 동적으로 softmax를 재계산하는 Streaming softmax (Online softmax) 알고리즘을 사용한다.

        일반적인 softmax 알고리즘은 다음과 같다.
            >>> import numpy as np
            >>>
            >>> def standard_softmax(x):
            ...     x_max = np.max(x)
            ...     ez = np.exp(x - x_max)
            ...     return ez / ez.sum()

        Streaming softmax는 다음과 같이 연산한다.
            >>> def streaming_softmax(x, tile_size):
            ...    x_max = -np.inf
            ...    ez_sum = 0.0
            ...    for idx in range(0, x.size, tile_size):
            ...        current_x = x[idx:idx + tile_size]
            ...        current_x_max = np.max(current_x)
            ...        new_x_max = np.maximum(current_x_max, x_max)
            ...        rescale = np.exp(x_max - new_x_max)
            ...        ez_sum *= rescale
            ...        ez_sum += np.exp(current_x - new_x_max).sum()
            ...        x_max = new_x_max
            ...    return np.exp(x - x_max) / ez_sum

        두 함수의 연산 결과는 동일하다.
            >>> x = np.random.randn(1024)
            >>> y_standard = standard_softmax(x)
            >>> y_streaming = streaming_softmax(x, tile_size=64)
            >>> print(np.allclose(y_standard, y_streaming))  # True

        핵심 아이디어는 다음과 같다.
            - x_max, ez_sum과 같은 값을 한번에 구할 수 없으니 루프를 돌면서 업데이트 한다.
            - 최종 업데이트 된 x_max, ez_sum을 활용하여 최종 softmax 값을 구한다.
    """
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)

    # batch * head 차원을 고려해야 하므로 베이스 포인터 위치를 bh 차원의 block에 stride를 곱해서 계산한다.
    # batch와 head는 실제로 커널에 들어올때는 reshape 되어서 들어오기 때문에 그냥 차원 하나로 봐도 무방하다.
    q_bh = q_ptr + pid_bh * stride_q_bh
    k_bh = k_ptr + pid_bh * stride_k_bh
    v_bh = v_ptr + pid_bh * stride_v_bh
    o_bh = o_ptr + pid_bh * stride_o_bh

    # query가 블록 레벨로 쪼개지기 때문에 query의 포인터는 block id에 따라 달라져야 한다.
    # key/value는 루프로 연산되기 때문에 로컬 포인터를 만들어두고 루프 내에서 조정한다.
    q_start = pid_q * block_size_q
    offs_q = q_start + tl.arange(0, block_size_q)
    offs_kv = tl.arange(0, tile_size_kv)

    q_block_ptr = tl.make_block_ptr(
        base=q_bh,
        shape=(seq_len_q, dim),
        # dim 축은 block 레벨로 쪼개지지 않으므로 모든 블록에서 0부터 시작함.
        offsets=(q_start, 0),
        block_shape=(block_size_q, dim),
        strides=(stride_q_seq, stride_q_dim),
        order=(1, 0),
    )
    o_block_ptr = tl.make_block_ptr(
        base=o_bh,
        shape=(seq_len_q, dim),
        # dim 축은 block 레벨로 쪼개지지 않으므로 모든 블록에서 0부터 시작함.
        offsets=(q_start, 0),
        block_shape=(block_size_q, dim),
        strides=(stride_o_seq, stride_o_dim),
        order=(1, 0),
    )

    q = tl.load(
        q_block_ptr,
        boundary_check=(0, 1),
        padding_option="zero",
    )

    max_q = tl.full((block_size_q,), -float("inf"), dtype=tl.float32)
    ez_sum = tl.zeros((block_size_q,), dtype=tl.float32)
    ez_dot_v = tl.zeros((block_size_q, dim), dtype=tl.float32)

    for kv_start in range(0, seq_len_kv, tile_size_kv):
        # k, v 블록 포인터 생성
        k_block_ptr = tl.make_block_ptr(
            base=k_bh,
            shape=(seq_len_kv, dim),
            offsets=(kv_start, 0),
            block_shape=(tile_size_kv, dim),
            strides=(stride_k_seq, stride_k_dim),
            order=(1, 0),
        )
        v_block_ptr = tl.make_block_ptr(
            base=v_bh,
            shape=(seq_len_kv, dim),
            offsets=(kv_start, 0),
            block_shape=(tile_size_kv, dim),
            strides=(stride_v_seq, stride_v_dim),
            order=(1, 0),
        )

        # k, v 데이터 로드
        k = tl.load(
            k_block_ptr,
            boundary_check=(0, 1),
            padding_option="zero",
        )
        v = tl.load(
            v_block_ptr,
            boundary_check=(0, 1),
            padding_option="zero",
        )

        # (Q * K^T) / sqrt(d)
        scores = tl.dot(q, tl.trans(k)) * softmax_scale

        # 필요한 경우 causal mask 적용
        if causal:
            q_pos = offs_q[:, None]
            kv_pos = (kv_start + offs_kv)[None, :]
            mask = kv_pos > q_pos
            scores = tl.where(mask, -float("inf"), scores)

        # streaming softmax 업데이트
        current_max_q = tl.max(scores, axis=1)
        new_max_q = tl.maximum(max_q, current_max_q)

        rescale = tl.exp(max_q - new_max_q)
        current_ez = tl.exp(scores - new_max_q[:, None])
        ez_sum = ez_sum * rescale + tl.sum(current_ez, axis=1)
        ez_dot_v = ez_dot_v * rescale[:, None] + tl.dot(current_ez, v)
        max_q = new_max_q

    o = ez_dot_v / ez_sum[:, None]

    tl.store(
        o_block_ptr, o,
        boundary_check=(0, 1),
    )


def flash_attn_fwd(q, k, v, causal=True, softmax_scale=None):
    bsz, num_heads, seq_len_q, dim_head = q.shape
    seq_len_kv = k.shape[2]
    assert k.shape == v.shape == (bsz, num_heads, seq_len_kv, dim_head)

    bh = bsz * num_heads

    def merge_heads(x):
        return x.contiguous().view(bh, x.shape[2], dim_head)

    def grid(meta):
        return triton.cdiv(seq_len_q, meta["block_size_q"]), bh

    q_merged = merge_heads(q)
    k_merged = merge_heads(k)
    v_merged = merge_heads(v)
    o = torch.empty_like(q_merged)

    stride_q_bh, stride_q_seq, stride_q_dim = q_merged.stride()
    stride_k_bh, stride_k_seq, stride_k_dim = k_merged.stride()
    stride_v_bh, stride_v_seq, stride_v_dim = v_merged.stride()
    stride_o_bh, stride_o_seq, stride_o_dim = o.stride()

    if softmax_scale is None:
        softmax_scale = 1.0 / (dim_head ** 0.5)

    flash_attn_kernel_fwd[grid](
        q_merged, k_merged, v_merged, o,
        bh, seq_len_q, seq_len_kv,
        stride_q_bh, stride_q_seq, stride_q_dim,
        stride_k_bh, stride_k_seq, stride_k_dim,
        stride_v_bh, stride_v_seq, stride_v_dim,
        stride_o_bh, stride_o_seq, stride_o_dim,
        causal=causal,
        softmax_scale=softmax_scale,
        dim=dim_head,
    )

    return o.view(bsz, num_heads, seq_len_q, dim_head)


def naive_attention_torch(q, k, v, causal=True, softmax_scale=None):
    bsz, num_heads, q_len, d_head = q.shape
    kv_len = k.shape[2]
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d_head)

    scores = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale

    if causal:
        mask = torch.triu(torch.ones(q_len, kv_len, device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask[None, None, :, :], float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)
    return out


def sdpa_torch(q, k, v, causal=True, softmax_scale=None):
    bsz, num_heads, seq_len_q, dim_head = q.shape
    seq_len_kv = k.shape[2]
    q_merged = q.reshape(bsz * num_heads, seq_len_q, dim_head)
    k_merged = k.reshape(bsz * num_heads, seq_len_kv, dim_head)
    v_merged = v.reshape(bsz * num_heads, seq_len_kv, dim_head)

    if softmax_scale is None:
        softmax_scale = 1.0 / (dim_head ** 0.5)

    out = torch.nn.functional.scaled_dot_product_attention(
        q_merged, k_merged, v_merged,
        attn_mask=None, dropout_p=0.0, is_causal=causal, scale=softmax_scale,
    )

    return out.reshape(bsz, num_heads, seq_len_q, dim_head)


@torch.inference_mode()
def check_correctness():
    torch.manual_seed(7)
    cases = [
        # (bsz, num_heads, seq_len_q, seq_len_kv, dim_head, causal)
        (2, 8, 5, 5, 64, True),
        (2, 8, 5, 128, 64, True),
        (2, 8, 64, 64, 128, False),
        (1, 16, 1, 2048, 64, True),
    ]

    for (bsz, num_heads, seq_len_q, seq_len_kv, dim_head, causal) in cases:
        q = torch.randn(bsz, num_heads, seq_len_q, dim_head, device="cuda", dtype=torch.float32)
        k = torch.randn(bsz, num_heads, seq_len_kv, dim_head, device="cuda", dtype=torch.float32)
        v = torch.randn(bsz, num_heads, seq_len_kv, dim_head, device="cuda", dtype=torch.float32)

        out_triton = flash_attn_fwd(q, k, v, causal=causal)
        out_sdpa = sdpa_torch(q, k, v, causal=causal)
        out_naive = naive_attention_torch(q, k, v, causal=causal)

        def stats(a, b):
            diff = (a - b).float().abs()
            return diff.max().item(), diff.mean().item(), (diff / (b.float().abs().clamp_min(1e-6))).mean().item()

        m_ts, mae_ts, rel_ts = stats(out_triton, out_sdpa)
        m_ns, mae_ns, rel_ns = stats(out_naive,  out_sdpa)
        m_tn, mae_tn, rel_tn = stats(out_triton, out_naive)

        print(f"[bsz={bsz} heads={num_heads} q_len={seq_len_q} kv_len={seq_len_kv} d_head={dim_head} causal={causal}]")
        print(f"  Triton vs SDPA : max_abs={m_ts:.3e}  mae={mae_ts:.3e}  rel_mean={rel_ts:.3e}")
        print(f"  Naive  vs SDPA : max_abs={m_ns:.3e}  mae={mae_ns:.3e}  rel_mean={rel_ns:.3e}")
        print(f"  Triton vs Naive: max_abs={m_tn:.3e}  mae={mae_tn:.3e}  rel_mean={rel_tn:.3e}")


@torch.inference_mode()
def benchmark(warmup=10, iters=50):
    torch.manual_seed(7)
    configs = [
        ("inference-like", (16, 64, 1,    4096, 128, True)),
        ("training-like",  (16, 64, 1024, 1024, 128, True)),
    ]

    for name, (bsz, num_heads, q_len, kv_len, d_head, causal) in configs:
        q = torch.randn(bsz, num_heads, q_len, d_head, device="cuda", dtype=torch.float32)
        k = torch.randn(bsz, num_heads, kv_len, d_head, device="cuda", dtype=torch.float32)
        v = torch.randn(bsz, num_heads, kv_len, d_head, device="cuda", dtype=torch.float32)

        for _ in range(warmup):
            _ = flash_attn_fwd(q, k, v, causal=causal)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = flash_attn_fwd(q, k, v, causal=causal)
        torch.cuda.synchronize()
        triton_ms = (time.perf_counter() - t0) * 1000 / iters

        for _ in range(warmup):
            _ = sdpa_torch(q, k, v, causal=causal)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = sdpa_torch(q, k, v, causal=causal)
        torch.cuda.synchronize()
        sdpa_ms = (time.perf_counter() - t0) * 1000 / iters

        for _ in range(warmup):
            _ = naive_attention_torch(q, k, v, causal=causal)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = naive_attention_torch(q, k, v, causal=causal)
        torch.cuda.synchronize()
        naive_ms = (time.perf_counter() - t0) * 1000 / iters

        print(
            f"[{name}] bsz={bsz} heads={num_heads} q_len={q_len} kv_len={kv_len} d_head={d_head} causal={causal} "
            f"=> Triton: {triton_ms:.2f} ms | SDPA: {sdpa_ms:.2f} ms | Naive: {naive_ms:.2f} ms"
        )


if __name__ == "__main__":
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True

    print("== Correctness ==")
    check_correctness()

    print("\n== Benchmark ==")
    benchmark(warmup=10, iters=50)
