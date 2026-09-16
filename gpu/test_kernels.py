"""Check each tilelang kernel against plain PyTorch math on real checkpoint weights.

Runs on one GPU with no model build: tensors come straight from the rank-0 converted file.
Quantization itself costs a few percent, so compare the kernel against the dequantized
reference (should be ~1e-3) and separately report how far the reference sits from bf16 math.
"""

import sys

import torch
from safetensors import safe_open

sys.path.insert(0, "/dev/shm/models/DeepSeek-V4-Flash/inference")
from kernel import act_quant, fp4_act_quant, fp4_gemm, fp8_gemm, hc_split_sinkhorn  # noqa: E402

CKPT = "/dev/shm/models/V4F-mp2/model0-mp2.safetensors"
E8M0 = torch.float8_e8m0fnu
LUT = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], device="cuda")
torch.manual_seed(0)


def rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm())


def dequant_act(x):
    """Quantize like the model does and give back both the kernel inputs and a float copy."""
    xq, xs = act_quant(x, 128, "ue8m0", E8M0)
    x_deq = xq.float() * xs.float().repeat_interleave(128, dim=-1)
    return xq, xs, x_deq


def unpack_fp4(w):
    u8 = w.view(torch.uint8)
    low, high = u8 & 0x0F, (u8 >> 4) & 0x0F
    return torch.stack([LUT[low.long()], LUT[high.long()]], dim=-1).flatten(-2)


with safe_open(CKPT, framework="pt", device="cuda:0") as f:
    keys = list(f.keys())
    w8 = f.get_tensor("layers.5.attn.wq_a.weight")
    s8 = f.get_tensor("layers.5.attn.wq_a.scale")
    w4 = f.get_tensor("layers.5.ffn.experts.0.w1.weight")
    s4 = f.get_tensor("layers.5.ffn.experts.0.w1.scale")
    hc_fn = f.get_tensor("layers.5.hc_attn_fn")
    hc_scale = f.get_tensor("layers.5.hc_attn_scale")
    hc_base = f.get_tensor("layers.5.hc_attn_base")
print("fp8 weight", tuple(w8.shape), w8.dtype, "scale", tuple(s8.shape), s8.dtype)
print("fp4 weight", tuple(w4.shape), w4.dtype, "scale", tuple(s4.shape), s4.dtype)

# --- act_quant round trip ------------------------------------------------------------------
x = torch.randn(64, w8.shape[1], dtype=torch.bfloat16, device="cuda")
xq, xs, x_deq = dequant_act(x)
print(f"act_quant   round-trip rel err vs x: {rel(x_deq, x):.4f}   (fp8 quantization noise; expect ~0.02-0.05)")

# --- fp8_gemm ------------------------------------------------------------------------------
w8_deq = w8.float() * s8.float().repeat_interleave(128, 0).repeat_interleave(128, 1)[: w8.shape[0], : w8.shape[1]]
c_ref = x_deq @ w8_deq.T
c = fp8_gemm(xq, xs, w8, s8, E8M0)
print(f"fp8_gemm    rel err vs dequantized reference: {rel(c, c_ref):.4f}   (expect ~0.001-0.01)")
print(f"            reference vs bf16 math:            {rel(c_ref, x.float() @ w8_deq.T):.4f}")

# --- fp4_gemm (expert weight) ---------------------------------------------------------------
x4 = torch.randn(64, w4.shape[1] * 2, dtype=torch.bfloat16, device="cuda")
x4q, x4s, x4_deq = dequant_act(x4)
w4_deq = unpack_fp4(w4) * s4.float().repeat_interleave(32, dim=-1)
c4_ref = x4_deq @ w4_deq.T
c4 = fp4_gemm(x4q, x4s, w4, s4, E8M0)
print(f"fp4_gemm    rel err vs dequantized reference: {rel(c4, c4_ref):.4f}   (expect ~0.001-0.01)")
print(f"            reference vs bf16 math:            {rel(c4_ref, x4.float() @ w4_deq.T):.4f}")

# --- fp4_act_quant round trip ----------------------------------------------------------------
y = torch.randn(64, 512, dtype=torch.bfloat16, device="cuda")
yq, ys = fp4_act_quant(y, 32)
y_deq = unpack_fp4(yq) * ys.float().repeat_interleave(32, dim=-1)
print(f"fp4_act_quant round-trip rel err: {rel(y_deq, y):.4f}   (fp4 noise; expect ~0.1-0.2)")

# --- hc_split_sinkhorn -----------------------------------------------------------------------
hc, eps, iters = 4, 1e-6, 20
mixes = torch.randn(1, 8, (2 + hc) * hc, dtype=torch.float32, device="cuda")
pre, post, comb = hc_split_sinkhorn(mixes, hc_scale.float(), hc_base.float(), hc, iters, eps)
m, b_ = mixes[0], hc_base.float()
pre_ref = torch.sigmoid(m[:, :hc] * hc_scale[0].float() + b_[:hc]) + eps
post_ref = 2 * torch.sigmoid(m[:, hc:2 * hc] * hc_scale[1].float() + b_[hc:2 * hc])
comb_ref = (m[:, 2 * hc:].view(-1, hc, hc) * hc_scale[2].float() + b_[2 * hc:].view(hc, hc)).softmax(-1) + eps
comb_ref = comb_ref / (comb_ref.sum(-2, keepdim=True) + eps)
for _ in range(iters - 1):
    comb_ref = comb_ref / (comb_ref.sum(-1, keepdim=True) + eps)
    comb_ref = comb_ref / (comb_ref.sum(-2, keepdim=True) + eps)
print(f"hc_sinkhorn pre {rel(pre[0], pre_ref):.5f}  post {rel(post[0], post_ref):.5f}  comb {rel(comb[0], comb_ref):.5f}   (expect ~1e-6)")
