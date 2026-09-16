"""Summarize the activation-patching results by condition, kind (self / cross) and layer set."""

import collections
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/root/results/patch_results.jsonl"
rows = [json.loads(line) for line in open(path)]
groups = collections.defaultdict(list)
for r in rows:
    groups[(r["condition"], r["kind"], r["layers"])].append(r)

print(f"{len(rows)} records")
print(f"{'condition':9s} {'kind':5s} {'layers':6s} {'n':>3s}  {'base p(A)':>9s}  {'p(A)':>6s}  {'p(B)':>6s}  {'greedy=A':>8s}  {'greedy=B':>8s}  {'other':>6s}")
for key in sorted(groups):
    v = groups[key]
    n = len(v)

    def mean(f):
        return sum(f(r) for r in v) / n

    print(f"{key[0]:9s} {key[1]:5s} {key[2]:6s} {n:3d}  {mean(lambda r: r['base_p_a']):9.3f}  "
          f"{mean(lambda r: r['p_a']):6.3f}  {mean(lambda r: r['p_b']):6.3f}  "
          f"{mean(lambda r: r['greedy_is_a']):8.2f}  {mean(lambda r: r['greedy_is_b']):8.2f}  "
          f"{mean(lambda r: not r['greedy_is_a'] and not r['greedy_is_b']):6.2f}")
