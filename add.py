import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(ptr_a, ptr_b, ptr_z, size, block_size: tl.constexpr):
    """
    65536개의 엘리멘트를 1024 사이즈의 블록 64개로 나눠서 실행함.
        a[0, ..., 1023], a[1024, ..., 2047], ... 64개
    +   b[0, ..., 1023], b[1024, ..., 2047], ... 64개
    -----------------------------------------------
        z[0, ..., 1023], z[1024, ..., 2047], ... 64개
    """
    pid = tl.program_id(0)
    current_block_start = pid * block_size
    offsets = current_block_start + tl.arange(0, block_size)
    mask = offsets < size

    a = tl.load(ptr_a + offsets, mask=mask)
    b = tl.load(ptr_b + offsets, mask=mask)
    z = a + b
    tl.store(ptr_z + offsets, z, mask=mask)


def add(a, b, block_size):
    z = torch.empty_like(a)
    grid = lambda meta: (triton.cdiv(meta["size"], meta["block_size"]),)
    add_kernel[grid](a, b, z, size=a.numel(), block_size=block_size)
    return z


if __name__ == '__main__':
    a = torch.randn(65536, device="cuda")
    b = torch.randn(65536, device="cuda")
    z = add(a, b, block_size=1024)
    print(torch.allclose(z, a + b))