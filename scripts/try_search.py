from src.data import load_pubmedqa
from src.embeddings import embed
from src.index import build_index, search

questions, abstracts = load_pubmedqa()
print(len(questions), ', questions', len(abstracts), ', abstracts')

abstract_vectors = embed(abstracts)
index = build_index(abstract_vectors)

q= 0
scores, hits = search(index, embed([questions[q]]), k=3)

print("\nQUESTION:", questions[q])
print("CORRECT ABSTRACT: #", q)
for rank, (i, s) in enumerate(zip(hits[0], scores[0]), start=1):
    print(f"{rank}. abstract #{i} (score {s:.2f}): {abstracts[i][:150]}...")