from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from src.data import load_pubmedqa

questions, abstracts = load_pubmedqa()

docs = [Document(page_content=text, metadata={"abstract_id": i}) for i, text in enumerate(abstracts)]
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True},
)
store = FAISS.from_documents(docs, embeddings)
retriever = store.as_retriever(search_kwargs={"k": 3})

prompt = ChatPromptTemplate.from_template(
    "Answer the question using only the abstracts below.\n"
    "Start your answer with yes, no, or maybe, then explain in one or two sentences.\n"
    "If the abstracts do not contain the answer, say so.\n\n"
    "Abstracts:\n{context}\n\n"
    "Question: {question}"
)


def format_docs(docs):
    return "\n\n".join(f"[{d.metadata['abstract_id']}] {d.page_content}" for d in docs)


llm = ChatOllama(model="llama3.2:3b", temperature=0, num_ctx=4096)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("QUESTION:", questions[0])
print("ANSWER:", chain.invoke(questions[0]))