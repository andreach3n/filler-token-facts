"""Smoke test: load DeepSeek V4 Flash on 2 GPUs, capture hidden states, and work out how the
4 residual streams must be merged for a lens readout.

The model keeps a [batch, seq, 4, dim] residual and only merges the streams at the output head.
The J-lens expects a single [dim] vector per position, and its provenance does not say which
merge was used, so we compare candidates against the model own predictions.
"""
import json, os, sys, time
import torch
import torch.distributed as dist
import torch.nn.functional as F

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
local_rank = int(os.getenv("LOCAL_RANK", "0"))
if world_size > 1:
    dist.init_process_group("nccl")
torch.cuda.set_device(local_rank)
torch.cuda.memory._set_allocator_settings("expandable_segments:True")
torch.set_default_dtype(torch.bfloat16)
torch.set_num_threads(8)
torch.manual_seed(0)
log = print if rank == 0 else (lambda *a, **k: None)

t0 = time.time()
args = ModelArgs(**json.load(open(f"{INF}/config.json")))
args.max_batch_size = 1
args.max_seq_len = 4096
with torch.device("cuda"):
    model = Transformer(args)
load_model(model, f"{CKPT}/model{rank}-mp{world_size}.safetensors", strict=False)
torch.set_default_device("cuda")
tok = AutoTokenizer.from_pretrained(CKPT)
log(f"loaded in {time.time() - t0:.0f}s; GPU mem {torch.cuda.memory_allocated() / 1e9:.1f} GB")

acts = {}
for i, blk in enumerate(model.layers):
    blk.register_forward_hook(lambda m, inp, out, i=i: acts.__setitem__(i, out.detach()))

spec = json.load(open("/root/smoke_prompt_full.json"))
messages = [{"role": "system", "content": spec["system"]}] + spec["messages"]
prompt = encode_messages(messages, thinking_mode="chat")
log("prompt %s: gold answer %s, the API returned %r" % (spec["id"], spec["answer"], spec["api_reply"]))
ids = torch.tensor([tok.encode(prompt)], device="cuda")
log(f"prompt tokens: {ids.shape[1]}")

t0 = time.time()
model_logits = model.forward(ids, 0)
log(f"forward in {time.time() - t0:.1f}s; captured layers: {len(acts)}")
log("model next token: " + repr(tok.decode([model_logits[0].argmax().item()])))

def lens_logits(h, merge, positions):
    """Read out token logits from a [b, s, hc, d] residual using one stream merge."""
    h = h[:, positions]
    if merge == "hc_head":
        x = model.head.hc_head(h, model.hc_head_fn, model.hc_head_scale, model.hc_head_base)
    elif merge == "mean":
        x = h.mean(dim=2)
    elif merge == "sum":
        x = h.sum(dim=2)
    elif merge == "stream0":
        x = h[:, :, 0]
    local = F.linear(model.norm(x).float(), model.head.weight)
    if world_size > 1:
        parts = [torch.empty_like(local) for _ in range(world_size)]
        dist.all_gather(parts, local)
        local = torch.cat(parts, dim=-1)
    return local

n_layers = len(acts)
positions = list(range(-min(64, ids.shape[1]), 0))
truth = lens_logits(acts[n_layers - 1], "hc_head", positions)  # the model own output path
agree_last = int(truth[0, -1].argmax().item() == model_logits[0].argmax().item())
log(f"sanity: my readout of the last block matches model.forward at the last position: {bool(agree_last)}")

results = {}
for layer in (n_layers - 1, n_layers - 2, n_layers - 3, 30, 20):
    for merge in ("hc_head", "mean", "sum", "stream0"):
        got = lens_logits(acts[layer], merge, positions)
        top1 = (got.argmax(-1) == truth.argmax(-1)).float().mean().item()
        kl = F.kl_div(got.log_softmax(-1), truth.log_softmax(-1), log_target=True, reduction="batchmean").item()
        results[f"layer{layer}_{merge}"] = {"top1_agreement": round(top1, 3), "kl": round(kl, 3)}
        log(f"  layer {layer:2d} merge {merge:8s} top-1 agreement with model {top1:5.1%}  KL {kl:7.3f}")

if rank == 0:
    json.dump({"n_layers": n_layers, "results": results}, open("/workspace/logs/smoke_results.json", "w"), indent=1)
if world_size > 1:
    dist.destroy_process_group()
