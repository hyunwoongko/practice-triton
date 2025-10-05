import torch
import triton
import triton.language as tl


@triton.jit
def atomic_kernel(x_ptr, increment):
    tl.atomic_add(x_ptr, increment)


@triton.jit
def non_atomic_kernel(x_ptrs, increment):
    x = tl.load(x_ptrs)
    x += increment
    tl.store(x_ptrs, x)


def atomic(increment):
    x = torch.zeros(1, device="cuda")
    grid = (1024,)
    atomic_kernel[grid](x, increment)
    return x


def non_atomic(increment):
    x = torch.zeros(1, device="cuda")
    grid = (1024,)
    non_atomic_kernel[grid](x, increment)
    return x


x = atomic(2)
print(x)
y = non_atomic(2)
print(y)
