import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel_w_block_pointer(
    x_ptr, y_ptr, z_ptr, size, block_size: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * block_size

    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(size,),
        strides=(1,),
        offsets=(offset,),
        block_shape=(block_size,),
        order=(0,),
    )
    y_block_ptr = tl.make_block_ptr(
        y_ptr,
        shape=(size,),
        strides=(1,),
        offsets=(offset,),
        block_shape=(block_size,),
        order=(0,),
    )

    x = tl.load(x_block_ptr, boundary_check=(0,))
    y = tl.load(y_block_ptr, boundary_check=(0,))

    z_block_ptr = tl.make_block_ptr(
        z_ptr,
        shape=(size,),
        strides=(1,),
        offsets=(offset,),
        block_shape=(block_size,),
        order=(0,),
    )
    tl.store(
        z_block_ptr, x+y, boundary_check=(0,)
    )


def add(a, b, block_size):
    z = torch.empty_like(a)
    grid = lambda meta: (triton.cdiv(meta["size"], meta["block_size"]),)
    add_kernel_w_block_pointer[grid](a, b, z, size=a.numel(), block_size=block_size)
    return z


if __name__ == '__main__':
    a = torch.randn(65536, device="cuda")
    b = torch.randn(65536, device="cuda")
    z = add(a, b, block_size=1024)
    print(torch.allclose(z, a + b))
