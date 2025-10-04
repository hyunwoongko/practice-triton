import torch
import triton
import triton.language as tl


@triton.jit
def fused_matmul_bias_relu(
    A, B, C, bias,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    M_BLOCK_SIZE: tl.constexpr,
    N_BLOCK_SIZE: tl.constexpr,
    K_BLOCK_SIZE: tl.constexpr,
):
    m_pid = tl.program_id(0)
    n_pid = tl.program_id(1)

    m_block_start = m_pid * M_BLOCK_SIZE
    n_block_start = n_pid * N_BLOCK_SIZE

    m_offsets = m_block_start + tl.arange(0, M_BLOCK_SIZE)
    n_offsets = n_block_start + tl.arange(0, N_BLOCK_SIZE)
    k_offsets = tl.arange(0, K_BLOCK_SIZE)

    A_ptrs = A + (
        m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
    )
    B_ptrs = B + (
        k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn
    )

    C_output = tl.zeros((M_BLOCK_SIZE, N_BLOCK_SIZE), dtype=tl.float32)

    for k0 in range(0, K, K_BLOCK_SIZE):
        _A = tl.load(
            A_ptrs,
            mask=(m_offsets[:, None] < M) & ((k0 + k_offsets[None, :]) < K),
            other=0.0,
        )
        _B = tl.load(
            B_ptrs,
            mask=((k0 + k_offsets[:, None]) < K) & (n_offsets[None, :] < N),
            other=0.0,
        )
        C_output += tl.dot(_A, _B, allow_tf32=False)
        A_ptrs += K_BLOCK_SIZE * stride_ak
        B_ptrs += K_BLOCK_SIZE * stride_bk

    bias_offs = n_offsets
    bias_mask = n_offsets < N
    _bias = tl.load(
        bias + bias_offs,
        mask=bias_mask,
        other=0.0,
    )
    C_output += _bias[None, :]
    C_output = tl.maximum(C_output, 0)

    C_offsets = (
        m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    )
    tl.store(
        C + C_offsets, C_output,
        mask=(m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    )


def fused_matmul_bias_relu_run(A, B, bias,
                               BM=128, BN=128, BK=64,
                               num_warps=4, num_stages=3):
    M, K = A.shape
    K2, N = B.shape
    assert K == K2 and bias.shape == (N,)
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    fused_matmul_bias_relu[grid](
        A, B, C, bias,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        M_BLOCK_SIZE=BM, N_BLOCK_SIZE=BN, K_BLOCK_SIZE=BK,
        # 커널에 ALLOW_TF32 추가했다면 아래도 넘기세요:
        # ALLOW_TF32=allow_tf32,
        num_warps=num_warps, num_stages=num_stages,
    )
    return C


# --- 빠른 검증 ---
torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

for (M, K, N) in [(32, 64, 128), (128, 128, 128), (513, 257, 511)]:
    A = torch.randn(M, K, device='cuda', dtype=torch.float32)
    B = torch.randn(K, N, device='cuda', dtype=torch.float32)
    bias = torch.randn(N, device='cuda', dtype=torch.float32)

    ref = torch.relu(A @ B + bias)
    tri = fused_matmul_bias_relu_run(A, B, bias, BM=128, BN=128, BK=64)

    ok = torch.allclose(ref, tri, rtol=1e-5, atol=1e-5)
    print((M, K, N), "allclose:", ok)
