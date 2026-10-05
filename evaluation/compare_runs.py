import json
import sys
from scipy.stats import binomtest


def load(path):
    # read a .jsonl results file into {question id: record}
    with open(path, encoding="utf-8") as f:
        return {r["id"]: r for r in map(json.loads, f)}


a_path, b_path = sys.argv[1], sys.argv[2]
a, b = load(a_path), load(b_path)
ids = sorted(set(a) & set(b))

a_only = [i for i in ids if a[i]["predicted"] == a[i]["expert"] and b[i]["predicted"] != b[i]["expert"]]
b_only = [i for i in ids if b[i]["predicted"] == b[i]["expert"] and a[i]["predicted"] != a[i]["expert"]]
changed = [i for i in ids if a[i]["predicted"] != b[i]["predicted"]]

print(f"compared {len(ids)} questions")
print(f"answers that changed between runs: {len(changed)}")
print(f"right only in A ({a_path}): {len(a_only)}")
print(f"right only in B ({b_path}): {len(b_only)}")
if a_only or b_only:
    p = binomtest(len(a_only), len(a_only) + len(b_only), 0.5).pvalue
    print(f"McNemar exact p = {p:.3f}")