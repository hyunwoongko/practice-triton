import torch
import triton
import triton.language as tl


@triton.jit
def combine(a, b):
    return a + b


@triton.jit
def sum_kernel(ptr_x, ptr_y, size, block_size: tl.constexpr):
    offsets = tl.arange(0, block_size)
    mask = offsets < size

    x = tl.load(ptr_x + offsets, mask=mask)
    y = tl.reduce(x, 0, combine)

    tl.store(ptr_y, y)


def _sum(x):
    size = x.numel()
    y = torch.empty(1, device="cuda")
    grid = lambda m: (1,)
    block_size = triton.next_power_of_2(size)
    sum_kernel[grid](x, y, size=size, block_size=block_size)
    return y


if __name__ == '__main__':
    x = torch.randn(4096, device="cuda")
    torch_sum = torch.sum(x)
    triton_sum = _sum(x)
    print(torch.allclose(torch_sum, triton_sum))
