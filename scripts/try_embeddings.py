import numpy as np
from src.embeddings import embed

sentences = [
    "The patient was prescribed metformin for type 2 diabetes.",
    "Metformin is a common first-line drug for diabetes.",
    "The football match ended in a draw.",
    "Insulin resistance is a feature of type 2 diabetes.",
]

vectors = embed(sentences)
print(vectors.shape)
print(np.round(vectors @ vectors.T, 2))