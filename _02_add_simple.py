import torch
import triton
import triton.language as tl


@triton.jit
def add_simple_kernel(ptr_a, ptr_b, ptr_z, size):
    """
    index별로 a, b를 더해서 z에 저장
    각 block에서 1개의 인덱스만 처리함. -> 비효율적
    """
    pid = tl.program_id(0)
    if pid < size:
        a = tl.load(ptr_a + pid)
        b = tl.load(ptr_b + pid)
        tl.store(ptr_z + pid, a + b)


def add_simple(a, b):
    z = torch.empty_like(a)
    grid = lambda m: (a.numel(),)
    add_simple_kernel[grid](a, b, z, a.numel())
    return z


if __name__ == '__main__':
    a = torch.tensor([1, 2, 3], device="cuda")
    b = torch.tensor([10, 20, 30], device="cuda")
    z = add_simple(a, b)
    print(z)  # tensor([11, 22, 33], device='cuda:0')
