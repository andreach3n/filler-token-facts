"""Replace DeepSeek's tilelang FP4 GEMM with a PyTorch implementation.

On this GPU (RTX PRO 6000, sm120) the tilelang fp4_gemm kernel returns garbage: relative error
1.04 against a dequantized reference, while fp8_gemm (0.0016) and every other kernel are fine.
All MoE experts are FP4, so the model degenerated to copying its context.

Same maths as the kernel: C = dequant(A_fp8) @ dequant(B_fp4)^T with per-128 activation scales
and per-32 weight scales. Every dequantized value is exact in bf16 (fp8/fp4 mantissas times a
power-of-two scale), and bf16 matmul accumulates in fp32, matching the kernel's accumulation.
"""

from pathlib import Path

p = Path("/dev/shm/models/DeepSeek-V4-Flash/inference/kernel.py")
s = p.read_text()
marker = "\ndef fp4_gemm(\n"
assert s.count(marker) == 1, "fp4_gemm definition not found exactly once"
head, tail = s.split(marker)

replacement = '''

_FP4_LUT = None


def fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float32):
    """PyTorch replacement for the tilelang FP4 GEMM (broken on sm120). See patch_fp4.py."""
    global _FP4_LUT
    if _FP4_LUT is None or _FP4_LUT.device != b.device:
        _FP4_LUT = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
            device=b.device,
        )
    K = a.size(-1)
    a2 = a.reshape(-1, K)
    s2 = a_s.reshape(a2.size(0), -1)
    a_deq = (a2.float() * s2.float().repeat_interleave(K // s2.size(-1), dim=-1)).to(torch.bfloat16)
    u8 = b.view(torch.uint8)
    nib = torch.stack([u8 & 0x0F, (u8 >> 4) & 0x0F], dim=-1).flatten(-2)
    b_deq = (_FP4_LUT[nib.long()] * b_s.float().repeat_interleave(K // b_s.size(-1), dim=-1)).to(torch.bfloat16)
    out = a_deq @ b_deq.T
    return out.to(torch.get_default_dtype()).reshape(*a.shape[:-1], b.size(0))


def _fp4_gemm_tilelang(
'''
p.write_text(head + replacement + tail)
print("patched kernel.py: fp4_gemm -> PyTorch implementation (tilelang version kept as _fp4_gemm_tilelang)")
