# PubMedQA RAG

A question answering pipeline for biomedical yes/no/maybe questions. It finds the most relevant PubMed abstract for a question and has a local language model answer from that abstract. I built it to learn retrieval-augmented generation (RAG) from the ground up, and I evaluated each stage against the 1,000 expert-labeled questions in PubMedQA.

On 200 held-out questions, Llama 3.1 8B answering from one retrieved abstract was correct 68.5% of the time (95% CI 61.8 to 74.5%). Always answering "yes" scores 52.0% on the same questions (McNemar p < 0.001).

## How it works

1. Each of the 1,000 abstracts is embedded with `all-MiniLM-L6-v2` (sentence-transformers) and stored in a FAISS index.
2. A question is embedded the same way, and the closest abstract is retrieved.
3. The abstract and the question go into a prompt. A model running locally through Ollama answers yes, no or maybe, followed by a one or two sentence reason.
4. LangChain connects the steps. A FastAPI endpoint, `/ask`, returns the answer together with the abstract it was based on, so every answer can be checked against its source.

The data is the `pqa_labeled` split of [PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA) ([Jin et al., 2019](https://arxiv.org/abs/1909.06146)). Each question was written from one abstract, the abstract's conclusion section was removed, and experts labeled the answer yes, no or maybe. Because the conclusion is missing, the model has to infer the answer from the methods and results.

## Results

All runs used a laptop CPU with 16 GB of RAM and temperature 0.

### Retrieval

Measured on all 1,000 questions. For each question, the correct abstract is the one it was written from.

| Metric | Score |
| --- | --- |
| Recall@1 | 0.971 |
| Recall@3 | 0.989 |
| Recall@5 | 0.991 |
| Recall@10 | 0.997 |
| MRR@10 | 0.981 |

These numbers are high because the task is easy: there are only 1,000 candidates, and every question was written from its own abstract. The 9 misses at k=5 fell into three groups: near-duplicate papers on the same procedure, an underspecified question ("Is it Crohn's disease?"), and a figurative title.

### Answer accuracy on held-out questions

Questions 0 to 199 were used for every design decision. Questions 200 to 399 were run once at the end, with the final setup (one retrieved abstract, prompt V1).

| | Accuracy | 95% CI | "yes" correct | "no" correct | "maybe" correct | Seconds per question |
| --- | --- | --- | --- | --- | --- | --- |
| Always "yes" | 52.0% | | 104 / 104 | 0 / 69 | 0 / 27 | |
| Llama 3.2 3B | 58.5% | 51.6 to 65.1% | 95 / 104 | 21 / 69 | 1 / 27 | 7.4 |
| Llama 3.1 8B | 68.5% | 61.8 to 74.5% | 92 / 104 | 45 / 69 | 0 / 27 | 17.3 |

- 3B vs always "yes": 22 questions right only for the model, 9 right only for the baseline, p = 0.029.
- 8B vs always "yes": 45 against 12, p < 0.001.
- 8B vs 3B: the models disagreed on 36 questions. The 8B model was right on 28 of them and the 3B model on 8 (p = 0.001).

Confidence intervals are Wilson score intervals. p-values are exact McNemar tests (a binomial test on the questions where the two methods disagree).

### Development experiments

Run on questions 0 to 199, where always answering "yes" scores 56.0%. Each row changes one thing from the row it is compared with.

| Run | What changed | Accuracy | Result |
| --- | --- | --- | --- |
| 3B, 3 abstracts, prompt V1 | Starting point | 63.0% | Answered "yes" 170 times out of 200 |
| 3B, 1 abstract, prompt V1 | Pass one abstract instead of three | 63.0% | Same accuracy (8 gained, 8 lost, p = 1.0) and 2.6 times faster |
| 3B, 1 abstract, prompt V2 | Restate the question, define each label, warn against answering "yes" by default | 52.0% | Answered "no" 130 times; never answered "maybe" |
| 3B, 1 abstract, prompt V3 | V2 with a looser definition of "yes" | 55.0% | 8 gained, 2 lost compared with V2 (p = 0.11) |
| 3B, 1 abstract, prompt V4 | V3 without the warning sentence | 60.5% | 15 gained, 4 lost compared with V3 (p = 0.019) |
| 8B, 1 abstract, prompt V1 | Larger model | 67.0% | Caught 30 of 53 "no" questions, against 16 for the 3B model |

I kept one abstract and prompt V1 for the final setup. V4 was the most balanced 3B prompt (it caught 45 of 53 "no" questions) but was not more accurate than V1 (p = 0.63 head to head).

## What I found

Most errors happened after retrieval had worked. In 73 of the 74 errors from the first full run, the correct abstract had been retrieved. The model had the right source and still answered wrong. The usual advice is that RAG failures are mostly retrieval failures; in this setup it was the reverse, partly because retrieval here is easy.

The 3B model leaned toward "yes". Reading the errors by hand, the model often answered a slightly easier question than the one asked. One question asked whether two values were "equal"; the model answered that they were "similar". Another asked about "sleep and energy-related problems" and got an answer about sleep disorders. The easier question was usually one the abstract did support, so the answer came out "yes".

Prompt changes moved the bias around more than they removed it. Prompt V2 turned the yes-bias into a no-bias. Removing one sentence from it ("Do not answer yes just because the abstract discusses the same topic") recovered 15 questions and lost 4. Prompts V1 and V4 scored almost the same but disagreed on 79 of 200 answers: one leaned "yes", the other "no". The larger model was the only change that caught more "no" answers while keeping almost all of the "yes" answers (104 of 112, against 106 for the 3B model).

No model handled "maybe". Across both held-out runs, 1 of 62 "maybe" questions was answered correctly. The structured prompts (V2 to V4) produced 0 "maybe" answers in 600 tries. Many "maybe" questions probably depend on the removed conclusion section.

The model refused one legitimate question. Under prompt V2, Llama declined to answer a question about a cause of accidental death, apparently on safety grounds. The same question was answered under prompt V1. For a healthcare system, refusals like this matter as much as wrong answers.

A small pilot was misleading. A 20-question pilot scored 75%. On 200 questions the same setup scored 63%. The pilot happened to contain only one "maybe" question.

## Limitations

- Retrieval is tested on 1,000 abstracts, each the source of its own question. A larger corpus with unrelated abstracts would be harder; I have not tested that yet.
- `all-MiniLM-L6-v2` reads at most 256 word-pieces, so the ends of longer abstracts are cut off when they are embedded. The language model still reads the full abstract.
- With 200 questions per set, confidence intervals are about 13 points wide. The 8B vs 3B difference was not significant on the development set (p = 0.256); I estimate that a difference of 4 points would need roughly 700 questions to detect reliably.
- Several comparisons were run on the same development set. With a Holm correction, the V3 vs V4 result (p = 0.019) would not count as significant.
- Faithfulness (whether each claim in an answer is supported by the abstract) was checked by hand on a handful of answers, not measured across the full set.
- Temperature 0 makes answers close to repeatable, but wording can still change slightly between runs.

## Repository layout

```
src/
  data.py          load PubMedQA questions, abstracts and expert labels
  embeddings.py    sentence-transformers embedding (hand-built version)
  index.py         FAISS index and search (hand-built version)
  retrieval.py     LangChain retriever over FAISS
  generate.py      prompts V1 to V4, the LangChain chain, label parsing
  api.py           FastAPI service (/health, /ask)
evaluation/
  evaluate.py              retrieval recall@k and MRR
  evaluate_generation.py   answer accuracy, confidence interval, McNemar, confusion matrix
  compare_runs.py          McNemar test between two results files
  show_errors.py           print example errors of a chosen type
  results/                 one JSON Lines file per run
scripts/                   small step-by-step experiments from the build
Dockerfile
requirements.txt
```

Results files are named `set_model_k_prompt`. For example, `test_8b_k1_v1.jsonl` is the held-out set, the 8B model, one abstract and prompt V1. Each line holds the question, the expert label, the predicted label, the full answer and the retrieved abstract ids.

## Running it

You need Python 3.12 and [Ollama](https://ollama.com).

```
python -m venv ragvenv
ragvenv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
ollama pull llama3.2:3b
```

On macOS or Linux, activate with `source ragvenv/bin/activate`. The evaluation scripts also use scipy and scikit-learn (`pip install scipy scikit-learn` if they are missing).

### Evaluation

```
python -m evaluation.evaluate
python -m evaluation.evaluate_generation
python -m evaluation.compare_runs evaluation/results/test_3b_k1_v1.jsonl evaluation/results/test_8b_k1_v1.jsonl
```

The question range, model, prompt and output file for `evaluate_generation.py` are set at the top of the script.

### API

```
uvicorn src.api:app
```

The model defaults to `llama3.2:3b`. Set the `RAG_MODEL` environment variable to use another one. Then:

```
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d "{\"question\": \"Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?\"}"
```

The response holds `label` (yes, no or maybe), `answer` (the model's full reply) and `sources` (the abstract id and the start of its text). Interactive docs are at `http://127.0.0.1:8000/docs`.

### Docker

Ollama runs on the host machine, and the container connects to it.

```
docker build -t pubmed-rag .
docker run --rm -p 8000:8000 -e OLLAMA_HOST=http://host.docker.internal:11434 pubmed-rag
```

The image uses a CPU-only build of PyTorch, which keeps it at about 560 MB compressed.

## Planned next

- Add unrelated PubMed abstracts to the index and measure how retrieval recall changes as the corpus grows.
- Split long abstracts into chunks to get around the 256 word-piece limit, and compare BM25, dense and hybrid retrieval.
- Add a retrieval step that checks whether the abstract answers the question and searches again if not (LangGraph).
