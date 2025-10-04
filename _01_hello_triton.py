import torch
import triton
import triton.language as tl


@triton.jit
def hello_kernel(ptr_output):
    tl.device_print("Hello triton!")
    tl.store(ptr_output, 123.45)


output = torch.empty(1, dtype=torch.float32, device='cuda')
hello_kernel[(1,)](output)
print(output)