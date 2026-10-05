import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama

# V1: simple instructions. Best accuracy with the 8B model (68.5% on held-out questions). Default.
PROMPT_V1 = ChatPromptTemplate.from_template(
    "Answer the question using only the abstracts below.\n"
    "Start your answer with yes, no, or maybe, then explain in one or two sentences.\n"
    "If the abstracts do not contain the answer, say so.\n\n"
    "Abstracts:\n{context}\n\n"
    "Question: {question}"
)

# V2: adds question restatement, label definitions and an anti-yes warning.
# Overcorrected into a no-bias (52% on dev) and never predicted "maybe".
PROMPT_V2 = ChatPromptTemplate.from_template(
    "You are checking what a medical research abstract shows.\n\n"
    "Abstract:\n{context}\n\n"
    "Question: {question}\n\n"
    "Instructions:\n"
    "1. Restate exactly what the question asks. Keep its specific wording "
    "(for example, 'equal to' is not the same as 'similar to').\n"
    "2. Decide the answer using only the abstract:\n"
    "   - yes: the results directly support what the question asks\n"
    "   - no: the results show no effect, no difference, or the opposite\n"
    "   - maybe: the results are mixed or inconclusive, or the abstract does not settle the question\n"
    "   Do not answer yes just because the abstract discusses the same topic.\n\n"
    "Reply in exactly this format:\n"
    "Question asks: <one sentence>\n"
    "Answer: <yes, no, or maybe>\n"
    "Reason: <one or two sentences>"
)

# V3: V2 with a looser "yes" definition (conclusions are removed from PubMedQA abstracts).
# Small improvement over V2 (55% on dev, not significant).
PROMPT_V3 = ChatPromptTemplate.from_template(
    "You are checking what a medical research abstract shows.\n\n"
    "Abstract:\n{context}\n\n"
    "Question: {question}\n\n"
    "Instructions:\n"
    "1. Restate exactly what the question asks. Keep its specific wording "
    "(for example, 'equal to' is not the same as 'similar to').\n"
    "2. Decide the answer using only the abstract:\n"
    "   - yes: the results support what the question asks. The abstract's conclusion has been "
    "removed, so you will usually need to infer the answer from the results rather than "
    "find it stated directly.\n"
    "   - no: the results show no effect, no difference, or the opposite\n"
    "   - maybe: the results are mixed or inconclusive, or the abstract does not settle the question\n"
    "   Do not answer yes just because the abstract discusses the same topic.\n\n"
    "Reply in exactly this format:\n"
    "Question asks: <one sentence>\n"
    "Answer: <yes, no, or maybe>\n"
    "Reason: <one or two sentences>"
)

# V4: V3 without the anti-yes warning. Significantly better than V3 (p = 0.019) and the
# most balanced prompt (60.5% on dev), but not more accurate than V1.
PROMPT_V4 = ChatPromptTemplate.from_template(
    "You are checking what a medical research abstract shows.\n\n"
    "Abstract:\n{context}\n\n"
    "Question: {question}\n\n"
    "Instructions:\n"
    "1. Restate exactly what the question asks. Keep its specific wording "
    "(for example, 'equal to' is not the same as 'similar to').\n"
    "2. Decide the answer using only the abstract:\n"
    "   - yes: the results support what the question asks. The abstract's conclusion has been "
    "removed, so you will usually need to infer the answer from the results rather than "
    "find it stated directly.\n"
    "   - no: the results show no effect, no difference, or the opposite\n"
    "   - maybe: the results are mixed or inconclusive, or the abstract does not settle the question\n\n"
    "Reply in exactly this format:\n"
    "Question asks: <one sentence>\n"
    "Answer: <yes, no, or maybe>\n"
    "Reason: <one or two sentences>"
)

def format_docs(docs):
    return "\n\n".join(f"[{d.metadata['abstract_id']}] {d.page_content}" for d in docs)


def build_chain(retriever, model="llama3.2:3b", prompt=PROMPT_V1):
    llm = ChatOllama(model=model, temperature=0, num_ctx=4096)
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


def parse_label(answer):
    text = answer.strip().lower()
    # V2 format: look for a line like "Answer: no"
    match = re.search(r"answer:\s*\**\s*(yes|no|maybe)\b", text)
    if match:
        return match.group(1)
    # V1 format: the first word is the label
    words = text.split()
    if not words:
        return "unparsed"
    first = words[0].strip(".,:;!*")
    return first if first in {"yes", "no", "maybe"} else "unparsed"