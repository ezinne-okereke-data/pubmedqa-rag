import os
from fastapi import FastAPI
from pydantic import BaseModel
from src.data import load_pubmedqa
from src.retrieval import build_retriever
from src.generate import build_chain, parse_label, PROMPT_V1

MODEL = os.getenv("RAG_MODEL", "llama3.2:3b")

app = FastAPI(title="PubMed RAG")

# build everything once when the server starts, not on every request
questions, abstracts = load_pubmedqa()
retriever = build_retriever(abstracts, k=1)
chain = build_chain(retriever, model=MODEL, prompt=PROMPT_V1)


class Question(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL}


@app.post("/ask")
def ask(q: Question):
    docs = retriever.invoke(q.question)
    answer = chain.invoke(q.question)
    return {
        "question": q.question,
        "label": parse_label(answer),
        "answer": answer,
        "sources": [
            {"abstract_id": d.metadata["abstract_id"], "snippet": d.page_content[:300]}
            for d in docs
        ],
    }