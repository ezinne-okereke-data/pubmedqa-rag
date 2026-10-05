import json
import sys

path, expert, predicted = sys.argv[1], sys.argv[2], sys.argv[3]
n = int(sys.argv[4]) if len(sys.argv) > 4 else 5

with open(path, encoding="utf-8") as f:
    rows = [json.loads(line) for line in f]

matches = [r for r in rows
           if (expert == "any" or r["expert"] == expert) and r["predicted"] == predicted]
print(f"{len(matches)} questions with expert={expert}, predicted={predicted}; showing {min(n, len(matches))}\n")
for r in matches[:n]:
    print(f"--- q{r['id']} (expert: {r['expert']}) ---")
    print("QUESTION:", r["question"])
    print("ANSWER:", r["answer"])
    print()