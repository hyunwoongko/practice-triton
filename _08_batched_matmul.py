import torch
import triton
import triton.language as tl


@triton.jit
def batched_matmul_kernel(
    A, B, C,
    M, N, K,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    A_batch = A + pid_b * stride_ab
    B_batch = B + pid_b * stride_bb
    C_batch = C + pid_b * stride_cb

    m_block_start = pid_m * BLOCK_SIZE_M
    n_block_start = pid_n * BLOCK_SIZE_N

    offs_m = m_block_start + tl.arange(0, BLOCK_SIZE_M)
    offs_n = n_block_start + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    A_ptrs = A_batch + (
        offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    )
    B_ptrs = B_batch + (
        offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    )
    C_ptrs = C_batch + (
        offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    )

    C_output = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_SIZE_K):
        A_loaded = tl.load(
            A_ptrs,
            mask=(offs_m[:, None] < M) & ((k0 + offs_k[None, :]) < K),
            other=0.0,
        )
        B_loaded = tl.load(
            B_ptrs,
            mask=((k0 + offs_k[:, None]) < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        C_output += tl.dot(A_loaded, B_loaded, allow_tf32=False)
        A_ptrs += stride_ak * BLOCK_SIZE_K
        B_ptrs += stride_bk * BLOCK_SIZE_K

    tl.store(
        C_ptrs,
        C_output,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N)
    )


def bmm_triton(A, B, BM=128, BN=128, BK=64, num_warps=4, num_stages=3):
    BATCH, M, K = A.shape
    _, _K, N = B.shape
    assert K == _K
    C = torch.empty((BATCH, M, N), device=A.device, dtype=torch.float32)
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN), BATCH)
    batched_matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1), A.stride(2),
        B.stride(0), B.stride(1), B.stride(2),
        C.stride(0), C.stride(1), C.stride(2),
        BLOCK_SIZE_M=BM, BLOCK_SIZE_N=BN, BLOCK_SIZE_K=BK,
        num_warps=num_warps, num_stages=num_stages,
    )
    return C


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda"

    shapes = [
        (2, 37, 61, 113),
        (4, 128, 128, 128),
        (3, 257, 257, 255),
    ]
    for (BATCH, M, K, N) in shapes:
        A = torch.randn(BATCH, M, K, device=device, dtype=torch.float32)
        B = torch.randn(BATCH, K, N, device=device, dtype=torch.float32)

        ref = torch.bmm(A, B)
        out = bmm_triton(A, B, BM=128, BN=128, BK=64)

        ok = torch.allclose(ref, out, rtol=1e-5, atol=1e-5)
        diff = (ref - out).abs()
        print(f"[{BATCH},{M},{K},{N}] allclose={ok}  max={float(diff.max()):.3e}  mean={float(diff.mean()):.3e}")
