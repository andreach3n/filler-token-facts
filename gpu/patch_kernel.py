"""Patch DeepSeek's kernel.py for this GPU.

Two changes to sparse_attn:
1. Head chunking. The kernel holds all q heads in shared memory ((64, 512) bf16 = 64 KB), which
   pushes a block past this card's ~100 KB limit. Heads are independent in attention, so running
   chunks and concatenating is exactly equivalent with a smaller footprint.
2. Contiguous inputs. The kernel requires exact strides, and the model passes head-slice views
   during decoding.
"""

from pathlib import Path

src = Path("/root/kernel_orig.py").read_text()

old = """def sparse_attn(
    q: torch.Tensor, kv: torch.Tensor, attn_sink: torch.Tensor, topk_idxs: torch.Tensor, softmax_scale: float
) -> torch.Tensor:
    b, s, h, d = q.size()"""

new = '''SPARSE_ATTN_HEAD_CHUNK = 16


def sparse_attn(
    q: torch.Tensor, kv: torch.Tensor, attn_sink: torch.Tensor, topk_idxs: torch.Tensor, softmax_scale: float
) -> torch.Tensor:
    """Head-chunked wrapper: identical maths, smaller shared-memory footprint per block."""
    h_total = q.size(2)
    if h_total > SPARSE_ATTN_HEAD_CHUNK:
        outs = [
            _sparse_attn(q[:, :, i:i + SPARSE_ATTN_HEAD_CHUNK], kv,
                         attn_sink[i:i + SPARSE_ATTN_HEAD_CHUNK], topk_idxs, softmax_scale)
            for i in range(0, h_total, SPARSE_ATTN_HEAD_CHUNK)
        ]
        return torch.cat(outs, dim=2)
    return _sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale)


def _sparse_attn(
    q: torch.Tensor, kv: torch.Tensor, attn_sink: torch.Tensor, topk_idxs: torch.Tensor, softmax_scale: float
) -> torch.Tensor:
    # The kernel checks strides exactly. During decoding q has a size-1 sequence dim whose
    # stride PyTorch leaves at the parent's value, so .contiguous() is not enough: flatten and
    # re-view to force canonical strides.
    q = q.contiguous().flatten().view(q.shape)
    attn_sink, kv, topk_idxs = attn_sink.contiguous(), kv.contiguous(), topk_idxs.contiguous()
    b, s, h, d = q.size()'''

assert src.count(old) == 1
Path("/dev/shm/models/DeepSeek-V4-Flash/inference/kernel.py").write_text(src.replace(old, new))
print("patched kernel.py: head chunking + contiguous inputs")
