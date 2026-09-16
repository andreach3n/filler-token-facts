"""Print what the lenses read at the last positions of one readout file (sanity check)."""
import json
import sys

import numpy as np
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("/dev/shm/models/V4F-mp2")
d = np.load(sys.argv[1])
meta = json.loads(str(d["meta"]))
print("file:", sys.argv[1], "| condition", meta["condition"], "| set", meta["set"])
print("targets", list(d["targets"]), "ids", d["target_ids"], "| answer prob", float(d["answer_prob"]), "greedy", tok.decode([int(d["greedy_token_id"])]))
layers = list(d["layers"])
for lens in ("logit", "jlens", "rlens"):
    for li, l in enumerate(layers):
        ids, ps = d[f"{lens}_raw_top_ids"][li, -1, :5], d[f"{lens}_raw_top_p"][li, -1, :5]
        top = [(tok.decode([int(i)]), round(float(p), 3)) for i, p in zip(ids, ps)]
        print(f"{lens:5s} layer {l:2d} last pos: top5 {top}")
        print(f"      target raw p {np.round(d[f'{lens}_target_raw_p'][li, -1], 4)}  residual rank {d[f'{lens}_target_res_rank'][li, -1]}")
