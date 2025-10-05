import torch
import triton
import triton.language as tl

BLOCK_SIZE_M = 128
BLOCK_SIZE_N = 128
BLOCK_SIZE_K = 32


@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N

    A_ptr = tl.make_block_ptr(
        A,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(m_start, 0),  # K축은 로컬이라 0부터
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        order=(1, 0)
    )

    B_ptr = tl.make_block_ptr(
        B,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, n_start),  # K축은 로컬이라 0부터
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
        order=(1, 0)
    )

    C_output = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(A_ptr, boundary_check=(0, 1), padding_option="zero")
        b = tl.load(B_ptr, boundary_check=(0, 1), padding_option="zero")

        C_output += tl.dot(a, b, allow_tf32=False)
        A_ptr = tl.advance(A_ptr, (0, BLOCK_SIZE_K))
        B_ptr = tl.advance(B_ptr, (BLOCK_SIZE_K, 0))

    C_ptr = tl.make_block_ptr(
        C,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(m_start, n_start),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0)
    )
    tl.store(C_ptr, C_output, boundary_check=(0, 1))


def matmul(A, B):
    M, K = A.shape
    _K, N = B.shape
    assert K == _K, "AK and BK must be same."

    C = torch.empty((M, N), device="cuda", dtype=torch.float32)

    def grid(meta):
        return (
            triton.cdiv(meta["M"], meta["BLOCK_SIZE_M"]),
            triton.cdiv(meta["N"], meta["BLOCK_SIZE_N"]),
        )

    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
    )
    return C


if __name__ == "__main__":
    torch.manual_seed(0)
    A = torch.randn(4096, 4096, device="cuda", dtype=torch.float32)
    B = torch.randn(4096, 4096, device="cuda", dtype=torch.float32)

    out_ref = A @ B
    out_tri = matmul(A, B)

    print(torch.allclose(out_ref, out_tri, atol=1e-5, rtol=1e-5))
