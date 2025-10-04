import torch
import triton
import triton.language as tl


BLOCK_SIZE_M = 128
BLOCK_SIZE_N = 128
BLOCK_SIZE_K = 32


@triton.jit
def matmul_kernel(
    A, B, C,  # 입력 행렬 A, B와 출력 행렬 C의 포인터
    M, N, K,  # 행렬의 크기 A(M,K) · B(K, N) = C(M, N)
    stride_am, stride_ak,  # A의 m_stride, k_stride
    stride_bk, stride_bn,  # B의 k_stride, n_stride
    stride_cm, stride_cn,  # C의 m_stride, n_stride
    BLOCK_SIZE_M: tl.constexpr,  # M에 할당할 블록 사이즈
    BLOCK_SIZE_N: tl.constexpr,  # N에 할당할 블록 사이즈
    BLOCK_SIZE_K: tl.constexpr,  # K에 할당할 블록 사이즈
):
    pid_m = tl.program_id(axis=0)  # m방향 block index
    pid_n = tl.program_id(axis=1)  # n방향 block index

    m_start_idx_by_block = pid_m * BLOCK_SIZE_M  # m방향 현재 블록의 start idx
    n_start_idx_by_block = pid_n * BLOCK_SIZE_N  # n방향 현재 블록의 start idx

    offset_m = m_start_idx_by_block + tl.arange(0, BLOCK_SIZE_M)  # m방향 글로벌 오프셋
    offset_n = n_start_idx_by_block + tl.arange(0, BLOCK_SIZE_N)  # n방향 글로벌 오프셋
    offset_k = tl.arange(0, BLOCK_SIZE_K)  # k방향 로컬 오프셋

    a_ptrs = A + (offset_m[:, None] * stride_am + offset_k[None, :] * stride_ak)
    # offset_m을 전치하고 스트라이드 곱하고 + offset_k를 unsqueeze(0)하고 스트라이드 곱하고 더해서 2차원으로 만듦
    b_ptrs = B + (offset_k[:, None] * stride_bk + offset_n[None, :] * stride_bn)
    # offset_k을 전치하고 스트라이드 곱하고 + offset_n를 unsqueeze(0)하고 스트라이드 곱하고 더해서 2차원으로 만듦

    # 연산 결과를 누적하여 저장할 텐서
    c_output = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(
            a_ptrs,
            mask=(offset_m[:, None] < M) & ((k + offset_k[None, :]) < K),
            # offset_m은 M보다 작아야 하고 로컬 k + offset_k는 K보다 작아야 함
            other=0,
        )

        b = tl.load(
            b_ptrs,
            mask=((k + offset_k[:, None]) < K) & (offset_n[None, :] < N),
            # 로컬 k + offset_k는 K보다 작아야 하고 offset_n은 N보다 작아야 함.
            other=0,
        )

        c_output += tl.dot(a, b, allow_tf32=False)
        # dot product, 정합성 검증을 위해 allow_tf32 사용.

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c_ptrs = C + (offset_m[:, None] * stride_cm + offset_n[None, :] * stride_cn)
    tl.store(c_ptrs, c_output, mask=(offset_m[:, None] < M) & (offset_n[None, :] < N))


def matmul(A, B):
    M, K = A.shape
    _K, N = B.shape

    assert K == _K, "AK and BK must be same."
    C = torch.empty(M, N, device="cuda", dtype=torch.float32)

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
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
    )

    return C


if __name__ == '__main__':
    A = torch.randn(4096, 4096, device="cuda", dtype=torch.float32)
    B = torch.randn(4096, 4096, device="cuda", dtype=torch.float32)

    output_torch = A @ B
    output_triton = matmul(A, B)

    print(torch.allclose(output_torch, output_triton))


