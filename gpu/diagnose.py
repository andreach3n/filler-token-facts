"""Is the local model producing sensible text at all? Greedy-decode a few short prompts."""

import json
import os
import sys

import torch
import torch.distributed as dist

INF = "/dev/shm/models/DeepSeek-V4-Flash/inference"
ENC = "/dev/shm/models/DeepSeek-V4-Flash/encoding"
CKPT = "/dev/shm/models/V4F-mp2"
sys.path[:0] = [INF, ENC]
from model import Transformer, ModelArgs  # noqa: E402
from encoding_dsv4 import encode_messages  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from safetensors.torch import load_model  # noqa: E402

world_size = int(os.getenv("WORLD_SIZE", "1"))
rank = int(os.getenv("RANK", "0"))
if world_size > 1:
    dist.init_process_group("nccl")
torch.cuda.set_device(int(os.getenv("LOCAL_RANK", "0")))
torch.cuda.memory._set_allocator_settings("expandable_segments:True")
torch.set_default_dtype(torch.bfloat16)
torch.set_num_threads(8)
log = print if rank == 0 else (lambda *a, **k: None)

args = ModelArgs(**json.load(open(f"{INF}/config.json")))
args.max_batch_size = 1
args.max_seq_len = 4096
with torch.device("cuda"):
    model = Transformer(args)
load_model(model, f"{CKPT}/model{rank}-mp{world_size}.safetensors", strict=False)
torch.set_default_device("cuda")
tok = AutoTokenizer.from_pretrained(CKPT)


@torch.inference_mode()
def greedy(ids, n=8):
    """Prefill then decode one token at a time, as DeepSeek's own generate() does."""
    cur = torch.tensor([ids], device="cuda")
    out, prev = [], 0
    for _ in range(n):
        logits = model.forward(cur[:, prev:], prev)
        prev = cur.shape[1]
        nxt = int(logits[0].argmax())
        out.append(nxt)
        cur = torch.cat([cur, torch.tensor([[nxt]], device="cuda")], dim=1)
    return tok.decode(out)


spec = json.load(open("/root/smoke_prompt_full.json"))
chat = lambda msgs: tok.encode(encode_messages(msgs, thinking_mode="chat"))  # noqa: E731
cases = {
    "raw text": tok.encode("The capital of Italy is"),
    "chat capital": chat([{"role": "user", "content": "What is the capital of Italy? One word."}]),
    "chat 2+2": chat([{"role": "user", "content": "What is 2+2? Answer with just the number."}]),
    "soe prompt": chat([{"role": "system", "content": spec["system"]}] + spec["messages"]),
    "soe no system": chat(spec["messages"]),
}
for name, ids in cases.items():
    log("%-14s (%5d tokens) -> %r" % (name, len(ids), greedy(ids)))
log("(our prompt: gold %s, API returned %r)" % (spec["answer"], spec["api_reply"]))

if world_size > 1:
    dist.destroy_process_group()
