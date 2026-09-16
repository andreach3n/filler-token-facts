"""Phase 1: run every prompt through DeepSeek V4 Flash and store merged hidden states.

For each prompt we keep, at every layer, the residual at the filler-token positions plus the
last few positions (where the answer is produced), merged from the 4 hyper-connection streams
with the model's own final merge (hc_head): the smoke test showed that merge reads out best.
Phase 2 (lens_readout.py) applies the logit lens, J-lens and R-lens to these states without
the model loaded, so analysis can iterate cheaply.

Also records the model's own answer at the last position (greedy token, probability of the
correct answer), which doubles as the accuracy check on these exact weights.

Run on the pod:  torchrun --nproc-per-node 2 extract_states.py
Resumable: prompts with an existing output file are skipped.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

INF = "/dev/shm/models/DeepSeek-V4-Flash/inference"
ENC = "/dev/shm/models/DeepSeek-V4-Flash/encoding"
CKPT = "/dev/shm/models/V4F-mp2"
PROMPTS = Path("/workspace/gpu_prompts_soe300.jsonl")
OUT = Path("/workspace/states")
KEEP_LAST = 4          # positions at the end of the prompt (the answer is predicted at the last one)
FIRST_LAYER = 0        # store block outputs from this layer on (all 43)
MAX_WORKSPACE_GB = 44  # the volume quota is ~50 GB; stop before hitting it
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
n_layers = len(model.layers)
log(f"model loaded; {n_layers} layers")

# --- capture: merge the 4 streams at the chosen positions inside the hook ----------------------
capture = {"positions": None, "states": {}}


def make_hook(layer):
    def hook(module, inputs, out):
        if layer < FIRST_LAYER or capture["positions"] is None:
            return
        h = out[:, capture["positions"]]  # [1, P, 4, d]
        merged = model.head.hc_head(h, model.hc_head_fn, model.hc_head_scale, model.hc_head_base)
        capture["states"][layer] = merged[0].to(torch.float16).cpu()
    return hook


for i, block in enumerate(model.layers):
    block.register_forward_hook(make_hook(i))


def token_positions(prompt, offsets, filler_text):
    """Token indices covering the last occurrence of the filler text (the target turn)."""
    if not filler_text:
        return []
    start = prompt.rfind(filler_text)
    assert start >= 0, "filler text not found in prompt"
    end = start + len(filler_text)
    return [i for i, (a, b) in enumerate(offsets) if b > a and a < end and b > start]


def answer_token_id(answer):
    ids = tok.encode(str(answer), add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None


prompts = [json.loads(line) for line in PROMPTS.read_text().splitlines()]
order = {"none": 0, "counting": 1, "false": 2, "true": 3}
prompts.sort(key=lambda p: (order[p["condition"]], p["set"], p["id"]))
OUT.mkdir(parents=True, exist_ok=True)

done = skipped = 0
t_start = time.time()
for n, spec in enumerate(prompts):
    out_path = OUT / f"{spec['id']}.npz"
    if out_path.exists():
        skipped += 1
        continue
    # disk_usage() reports the whole shared cluster; the quota applies to our own directory.
    used_gb = int(subprocess.check_output(["du", "-sb", "/workspace"]).split()[0]) / 1e9
    if used_gb > MAX_WORKSPACE_GB:  # both ranks see the same number, so both stop
        log(f"STOP: /workspace usage {used_gb:.1f} GB is above {MAX_WORKSPACE_GB} GB")
        break

    messages = [{"role": "system", "content": spec["system"]}] + spec["messages"]
    prompt = encode_messages(messages, thinking_mode="chat")
    enc = tok(prompt, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    filler_pos = token_positions(prompt, offsets, spec["filler_text"])
    last_pos = list(range(len(ids) - KEEP_LAST, len(ids)))
    positions = filler_pos + [p for p in last_pos if p not in filler_pos]
    capture["positions"] = positions
    capture["states"] = {}

    t0 = time.time()
    with torch.inference_mode():
        logits = model.forward(torch.tensor([ids], device="cuda"), 0)[0].float()
    probs = logits.softmax(-1)
    top = probs.topk(10)
    a_id = answer_token_id(spec["answer"])
    greedy = tok.decode([int(top.indices[0])])
    if rank == 0:
        states = torch.stack([capture["states"][l] for l in range(FIRST_LAYER, n_layers)])  # [L, P, d]
        np.savez(
            out_path,
            states=states.numpy(),
            positions=np.array(positions, dtype=np.int32),
            n_filler=np.int32(len(filler_pos)),
            n_tokens=np.int32(len(ids)),
            layers=np.arange(FIRST_LAYER, n_layers, dtype=np.int32),
            answer=np.int32(spec["answer"]),
            answer_token_id=np.int32(-1 if a_id is None else a_id),
            answer_prob=np.float32(probs[a_id].item() if a_id is not None else float("nan")),
            greedy_token_id=np.int32(int(top.indices[0])),
            top10_ids=top.indices.cpu().numpy().astype(np.int32),
            top10_probs=top.values.cpu().numpy().astype(np.float32),
            intermediates=np.array(json.dumps(spec["intermediates"])),
            meta=np.array(json.dumps({k: spec[k] for k in ("id", "condition", "set", "problem_id", "api_reply")})),
        )
    done += 1
    elapsed = time.time() - t_start
    log(f"[{n + 1}/{len(prompts)}] {spec['id']:22s} tokens={len(ids):5d} filler={len(filler_pos):3d} "
        f"greedy={greedy!r:8s} gold={spec['answer']:<4d} api={spec['api_reply']!r:8s} "
        f"p(ans)={probs[a_id].item() if a_id is not None else float('nan'):.3f} "
        f"fwd={time.time() - t0:5.1f}s  avg={elapsed / done:5.1f}s/prompt")

log(f"finished: {done} new, {skipped} skipped, {time.time() - t_start:.0f}s")
if world_size > 1:
    dist.destroy_process_group()
