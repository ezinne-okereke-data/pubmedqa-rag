import os
import json
import math
import time
from collections import Counter
from scipy.stats import binomtest
from src.data import load_pubmedqa, load_labels
from src.retrieval import build_retriever
from src.generate import build_chain, parse_label, PROMPT_V1
from sklearn.metrics import confusion_matrix

START, END = 200, 400
OUT = "evaluation/results/test_3b_k1.jsonl"
if os.path.exists(OUT):
    raise SystemExit(f"{OUT} already exists. Choose a new OUT name so earlier results are not erased.")

def wilson_interval(correct, n, z=1.96):
    # 95% confidence interval for a proportion; behaves better than the textbook formula for small n
    p = correct / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - margin, centre + margin


questions, abstracts = load_pubmedqa()
labels = load_labels()
retriever = build_retriever(abstracts, k=1)
chain = build_chain(retriever, model="llama3.2:3b", prompt =PROMPT_V1)

results = []
start = time.time()
with open(OUT, "w", encoding="utf-8") as f:
    for i in range(START, END):
        retrieved_ids = [d.metadata["abstract_id"] for d in retriever.invoke(questions[i])]
        answer = chain.invoke(questions[i])
        predicted = parse_label(answer)
        r = {"id": i, "expert": labels[i], "predicted": predicted,
             "retrieved_ids": retrieved_ids, "right_abstract_found": i in retrieved_ids,
             "question": questions[i], "answer": answer}
        results.append(r)
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        mark = "OK" if predicted == labels[i] else "XX"
        print(f"{mark} q{i}: predicted={predicted}, expert={labels[i]}, found={i in retrieved_ids}")
elapsed = time.time() - start

n = len(results)
correct = sum(r["predicted"] == r["expert"] for r in results)
low, high = wilson_interval(correct, n)
print(f"\naccuracy: {correct}/{n} = {correct / n:.3f}  (95% CI {low:.3f} to {high:.3f})")
print(f"time: {elapsed / 60:.0f} min total, {elapsed / n:.1f}s per question")

expert_counts = Counter(r["expert"] for r in results)
print("expert labels:", expert_counts)
print("predictions:  ", Counter(r["predicted"] for r in results))

top_label, top_count = expert_counts.most_common(1)[0]
print(f"baseline (always '{top_label}'): {top_count / n:.3f}")

# McNemar's test only uses the questions where the model and the baseline disagree
model_only = sum(r["predicted"] == r["expert"] and r["expert"] != top_label for r in results)
baseline_only = sum(r["predicted"] != r["expert"] and r["expert"] == top_label for r in results)
if model_only + baseline_only > 0:
    p = binomtest(model_only, model_only + baseline_only, 0.5).pvalue
    print(f"McNemar exact: model-only right = {model_only}, baseline-only right = {baseline_only}, p = {p:.3f}")

errors = [r for r in results if r["predicted"] != r["expert"]]
found = sum(r["right_abstract_found"] for r in errors)
print(f"\n{len(errors)} errors; the right abstract was retrieved in {found} of them")
print("unparsed answers:", sum(r["predicted"] == "unparsed" for r in results))

order = ["yes", "no", "maybe"]
cm = confusion_matrix([r["expert"] for r in results], [r["predicted"] for r in results], labels=order)
print("\nconfusion matrix (rows = expert, columns = predicted):")
print("        " + "  ".join(f"{label:>5}" for label in order))
for label, row in zip(order, cm):
    print(f"{label:>6}  " + "  ".join(f"{count:>5}" for count in row))