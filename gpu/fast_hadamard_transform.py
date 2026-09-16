"""Pure-PyTorch stand-in for the fast_hadamard_transform CUDA package.

DeepSeek V4 calls hadamard_transform() to spread information across dimensions before FP8
quantization. The CUDA package needs nvcc, which this image lacks, so this implements the same
fast Walsh-Hadamard transform with butterfly steps in float32.
"""

import torch


def hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    n = x.shape[-1]
    assert n & (n - 1) == 0, f"last dim must be a power of two, got {n}"
    out = x.float().reshape(-1, n)
    step = 1
    while step < n:
        out = out.view(-1, n // (2 * step), 2, step)
        a, b = out[:, :, 0, :], out[:, :, 1, :]
        out = torch.stack((a + b, a - b), dim=2)
        step *= 2
    return (out.reshape(x.shape) * scale).to(x.dtype)
