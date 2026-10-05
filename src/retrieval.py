from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def build_retriever(abstracts, k=3):
    docs = [Document(page_content=text, metadata={"abstract_id": i}) for i, text in enumerate(abstracts)]
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True},
    )
    store = FAISS.from_documents(docs, embeddings)
    return store.as_retriever(search_kwargs={"k": k})