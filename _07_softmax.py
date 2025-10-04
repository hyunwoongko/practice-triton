import torch
import triton
import triton.language as tl


def softmax_1d_torch(x):
    x_max = x.max()
    z = x - x_max
    ez = torch.exp(z)
    return ez / ez.sum()


@triton.jit
def softmax_1d_kernel(
    X, Y,
    SIZE,
    X_BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * X_BLOCK_SIZE
    offsets = block_start + tl.arange(0, X_BLOCK_SIZE)

    X_ptrs = X + offsets
    X_mask = offsets < SIZE
    X_loaded = tl.load(
        X_ptrs,
        mask=X_mask,
        other=-float("inf"),
    )

    X_max = tl.max(X_loaded, axis=0)
    Z = X_loaded - X_max
    EZ = tl.exp(Z)
    OUTPUT = EZ / tl.sum(EZ, axis=0)

    tl.store(
        Y + offsets,
        OUTPUT,
        mask=offsets < SIZE
    )


def softmax_2d_torch(X):
    # X: [M, N]
    x_max = X.max(dim=1, keepdim=True).values  # [M,1]
    Z = X - x_max
    EZ = torch.exp(Z)
    S = EZ.sum(dim=1, keepdim=True)  # [M,1]
    return EZ / S


@triton.jit
def softmax_2d_kernel(
    X, Y,
    M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    M_BLOCK_SIZE: tl.constexpr,
    N_BLOCK_SIZE: tl.constexpr,
):
    m_pid = tl.program_id(0)
    m_block_start = M_BLOCK_SIZE * m_pid

    m_offsets = m_block_start + tl.arange(0, M_BLOCK_SIZE)
    n_offsets = tl.arange(0, N_BLOCK_SIZE)
    # N차원을 axis로 계산을 하려면 N차원이 전부 있어야 함.
    mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)

    X_load = tl.load(
        X + m_offsets[:, None] * stride_xm + n_offsets[None, :] * stride_xn,
        mask=mask,
        other=-float("inf"),
    )

    X_max = tl.max(X_load, axis=1)[:, None]
    Z = X_load - X_max
    EZ = tl.exp(Z)
    S = tl.sum(EZ, axis=1)[:, None]
    Y_output = EZ / S

    tl.store(
        Y + m_offsets[:, None] * stride_ym + n_offsets[None, :] * stride_yn,
        Y_output,
        mask=mask,
    )


def softmax_rowwise(X, BM=128):
    M, N = X.shape
    Y = torch.empty_like(X)
    BN = triton.next_power_of_2(N)  # BN >= N
    grid = (triton.cdiv(M, BM), 1)
    softmax_2d_kernel[grid](
        X, Y, M, N,
        X.stride(0), X.stride(1),
        Y.stride(0), Y.stride(1),
        M_BLOCK_SIZE=BM, N_BLOCK_SIZE=BN,
        num_warps=4, num_stages=2,
    )
    return Y


# quick check
torch.manual_seed(0)
X = torch.randn(512, 512, device="cuda", dtype=torch.float32)
ref = torch.softmax(X, dim=1)  # PyTorch 기준
tri = softmax_rowwise(X)
print("allclose:", torch.allclose(ref, tri, rtol=1e-5, atol=1e-5))
print("row sums:", (tri.sum(dim=1).min().item(), tri.sum(dim=1).max().item()))
