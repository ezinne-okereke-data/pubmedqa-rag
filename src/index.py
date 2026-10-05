import faiss
import numpy as np

def build_index(vectors):
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors.astype(np.float32))
    return index

def search(index, query_vectors, k =5):
    scores, ids = index.search(query_vectors.astype(np.float32), k)
    return scores, ids

