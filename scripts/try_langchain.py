from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from src.data import load_pubmedqa

questions, abstracts = load_pubmedqa()


docs = [Document(page_content=text, metadata={"abstract_id": i}) for i, text in enumerate(abstracts)]


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True},
)

store = FAISS.from_documents(docs, embeddings)            # embeds everything and builds the index
retriever = store.as_retriever(search_kwargs={"k": 3})    # "give me the top 3"

results = retriever.invoke(questions[0])
print("QUESTION:", questions[0])
for d in results:
    print(d.metadata["abstract_id"], "|", d.page_content[:120], "...")