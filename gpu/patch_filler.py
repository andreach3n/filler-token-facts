"""Causal test: activation patching of the filler positions (paper 1's KV-transplant analog).

For a prompt A (answered correctly), pick another problem B from the same condition and
statement set. Run B and capture its residual stream (all 4 hyper-connection streams) at every
filler position for every layer; then run A with its filler-position residuals replaced by B's
at a chosen set of layers, everything downstream recomputed. If the answer position reads the
computation stored at the filler positions, A's answer should move toward B's.

Layer sets: all (0-42), late (30-42, where y / 2y / the answer were decodable), early (0-20).
Sanity: the first few prompts per condition are also patched with themselves (must reproduce
the unpatched answer).

Resumable (one JSON line per record in RESULTS; done records are skipped); heartbeat file
for the monitor; DONE marker at the end.

  torchrun --nproc-per-node 2 patch_filler.py
"""

import json
import os
import random
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
STATE_DIRS = [Path("/workspace/states"), Path("/root/states2")]
RESULTS = Path("/root/results/patch_results.jsonl")
HEARTBEAT = Path("/root/heartbeat_patch")
DONE = Path("/root/patch.done")
CONDITIONS = ("false", "counting")
PER_CONDITION = 100
N_SANITY = 5
LAYER_SETS = {"all": range(0, 43), "late": range(30, 43), "early": range(0, 21)}
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
log("model loaded")
if rank == 0:
    HEARTBEAT.write_text(json.dumps({"time": time.time(), "status": "model loaded"}))

# --- hooks: capture at filler positions, or overwrite them from a stored source ------------------
mode = {"capture_positions": None, "captured": {}, "patch_layers": set(), "patch_positions": None, "source": None}


def make_hook(layer):
    def hook(module, inputs, out):
        if mode["capture_positions"] is not None:
            mode["captured"][layer] = out[:, mode["capture_positions"]].clone()
        if layer in mode["patch_layers"]:
            out[:, mode["patch_positions"]] = mode["source"][layer]
        return out
    return hook


for i, block in enumerate(model.layers):
    block.register_forward_hook(make_hook(i))


def encode(spec):
    messages = [{"role": "system", "content": spec["system"]}] + spec["messages"]
    prompt = encode_messages(messages, thinking_mode="chat")
    enc = tok(prompt, return_offsets_mapping=True)
    start = prompt.rfind(spec["filler_text"])
    end = start + len(spec["filler_text"])
    positions = [i for i, (a, b) in enumerate(enc["offset_mapping"]) if b > a and a < end and b > start]
    return enc["input_ids"], positions


def answer_id(answer):
    ids = tok.encode(str(answer), add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None


@torch.inference_mode()
def forward(ids):
    logits = model.forward(torch.tensor([ids], device="cuda"), 0)[0].float()
    return logits.softmax(-1)


def run_capture(spec):
    ids, positions = encode(spec)
    mode.update(capture_positions=positions, captured={}, patch_layers=set())
    forward(ids)
    captured = mode["captured"]
    mode.update(capture_positions=None, captured={})
    return captured, len(positions)


def run_patched(spec, source, layers):
    ids, positions = encode(spec)
    mode.update(capture_positions=None, patch_layers=set(layers), patch_positions=positions, source=source)
    probs = forward(ids)
    mode.update(patch_layers=set(), source=None)
    return probs


# --- pairs: correct prompts within (condition, set), each patched from the next one -----------------
specs = {}
for line in PROMPTS.read_text().splitlines():
    s = json.loads(line)
    if s["condition"] in CONDITIONS:
        specs[s["id"]] = s


def phase1(id_):
    for d in STATE_DIRS:
        f = d / f"{id_}.npz"
        if f.exists():
            z = np.load(f)
            return int(z["greedy_token_id"]) == int(z["answer_token_id"]), float(z["answer_prob"])
    return None, None


groups = {}
for id_, s in specs.items():
    correct, p = phase1(id_)
    if correct:
        groups.setdefault((s["condition"], s["set"]), []).append(id_)
rng = random.Random(0)
jobs = []
for cond in CONDITIONS:
    per_set = []
    for key in sorted(k for k in groups if k[0] == cond):
        ids = sorted(groups[key])
        rng.shuffle(ids)
        per_set.append([(a, ids[(i + 1) % len(ids)]) for i, a in enumerate(ids)])
    picked, i = [], 0
    while len(picked) < PER_CONDITION and any(per_set):
        for lst in per_set:
            if lst and len(picked) < PER_CONDITION:
                picked.append(lst.pop(0))
    for n, (a, b) in enumerate(picked):
        if n < N_SANITY:
            jobs.append((a, a, "self", "all"))
        for name in LAYER_SETS:
            jobs.append((a, b, "cross", name))
log(f"{len(jobs)} patch jobs over {sum(len(v) for v in groups.values())} correct prompts")

done = set()
if RESULTS.exists():
    for line in RESULTS.read_text().splitlines():
        r = json.loads(line)
        done.add((r["a"], r["b"], r["kind"], r["layers"]))
RESULTS.parent.mkdir(parents=True, exist_ok=True)

cache = {"b": None, "captured": None}
t_start = time.time()
n_done = 0
for n, (a, b, kind, layers_name) in enumerate(jobs):
    if (a, b, kind, layers_name) in done:
        continue
    if cache["b"] != b:
        cache["captured"], n_fill_b = run_capture(specs[b])
        cache["b"] = b
    spec_a = specs[a]
    _, base_p = phase1(a)
    probs = run_patched(spec_a, cache["captured"], LAYER_SETS[layers_name])
    ans_a, ans_b = answer_id(spec_a["answer"]), answer_id(specs[b]["answer"])
    greedy = int(probs.argmax())
    rec = {
        "a": a, "b": b, "kind": kind, "layers": layers_name, "condition": spec_a["condition"], "set": spec_a["set"],
        "answer_a": spec_a["answer"], "answer_b": specs[b]["answer"],
        "base_p_a": base_p, "p_a": float(probs[ans_a]) if ans_a is not None else None,
        "p_b": float(probs[ans_b]) if ans_b is not None else None,
        "greedy": tok.decode([greedy]), "greedy_is_a": greedy == ans_a, "greedy_is_b": greedy == ans_b,
    }
    if rank == 0:
        with RESULTS.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        HEARTBEAT.write_text(json.dumps({"time": time.time(), "status": "running", "job": n + 1, "of": len(jobs)}))
    n_done += 1
    log(f"[{n + 1}/{len(jobs)}] {kind:5s} {layers_name:5s} {a} <- {b}: base p(A)={base_p:.2f} now p(A)={rec['p_a']:.2f} "
        f"p(B)={rec['p_b']:.2f} greedy={rec['greedy']!r} ({'A' if rec['greedy_is_a'] else 'B' if rec['greedy_is_b'] else 'other'}) "
        f"avg {(time.time() - t_start) / n_done:.1f}s/job")

if rank == 0:
    DONE.write_text("done\n")
log("finished")
if world_size > 1:
    dist.destroy_process_group()
