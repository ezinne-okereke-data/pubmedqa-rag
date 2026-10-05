import numpy as np
from src.data import load_pubmedqa
from src.embeddings import embed
from src.index import build_index, search


def recall_at_k(hits, k):
    # question i's correct abstract is abstract i, so check whether i is in its top k
    return np.mean([i in hits[i, :k] for i in range(len(hits))])


def mean_reciprocal_rank(hits):
    reciprocal_ranks = []
    for i, row in enumerate(hits):
        position = np.where(row == i)[0]
        reciprocal_ranks.append(1 / (position[0] + 1) if len(position) else 0.0)
    return np.mean(reciprocal_ranks)


questions, abstracts = load_pubmedqa()
index = build_index(embed(abstracts))
scores, hits = search(index, embed(questions), k=10)

for k in (1, 3, 5, 10):
    print(f"recall@{k}: {recall_at_k(hits, k):.3f}")
print(f"MRR@10: {mean_reciprocal_rank(hits):.3f}")

# the failures are where the learning is, so look at a few by hand
misses = [i for i in range(len(questions)) if i not in hits[i, :5]]
print(f"\n{len(misses)} questions missed at k=5")
for i in misses[:3]:
    print("\nQUESTION:", questions[i])
    print("TOP HIT:  ", abstracts[hits[i, 0]][:200])
    print("CORRECT:  ", abstracts[i][:200])
    