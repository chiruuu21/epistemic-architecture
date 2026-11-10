"""
TriviaQA helpers: sampling, closed-book prompt, answer normalisation and
matching. No model code here, so it is unit-testable offline.

Data: mandarjoshi/trivia_qa, config rc.nocontext, split validation
(17,944 questions; the official test split has no public answers). A
fixed seed picks the sample, so every model sees the same questions.

Correctness is exact match after the official TriviaQA normalisation
(lowercase, strip punctuation and articles, collapse whitespace) against
any of the question's aliases -- the standard closed-book metric. A
lenient flag (an alias appears inside the prediction) is recorded too,
for reporting how sensitive accuracy is to the matching rule.
"""

from __future__ import annotations

import re
import string

DATASET = ("mandarjoshi/trivia_qa", "rc.nocontext")
SPLIT = "validation"

# Hand-written exemplars, deliberately not drawn from the evaluation split.
FEW_SHOT = [
    ("Which planet is known as the Red Planet?", "Mars"),
    ("Who wrote the play Romeo and Juliet?", "William Shakespeare"),
    ("What is the chemical symbol for gold?", "Au"),
    ("In which country is the city of Kyoto?", "Japan"),
    ("How many sides does a hexagon have?", "Six"),
]

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = set(string.punctuation + "‘’´`_")


def normalize(s: str) -> str:
    """Official TriviaQA / SQuAD-style answer normalisation."""
    s = s.lower()
    s = "".join(" " if ch in _PUNCT else ch for ch in s)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def is_correct(pred: str, aliases: list[str]) -> bool:
    p = normalize(pred)
    return bool(p) and any(p == normalize(a) for a in aliases)


def is_correct_lenient(pred: str, aliases: list[str]) -> bool:
    p = f" {normalize(pred)} "
    return any(normalize(a) and f" {normalize(a)} " in p for a in aliases)


def prompt(question: str) -> str:
    shots = "".join(f"Q: {q}\nA: {a}\n\n" for q, a in FEW_SHOT)
    return f"Answer each question with a short answer.\n\n{shots}Q: {question.strip()}\nA:"


def belief_text(question: str, answer: str) -> str:
    """How a stored belief is phrased. The key is computed from this text,
    so it shares the question's wording but not its exact token sequence."""
    return f"{question.strip()} Answer: {answer.strip()}."


def load_sample(n: int, seed: int = 0) -> list[dict]:
    """n questions from the validation split, fixed by seed. Each item:
    {id, question, value, aliases}. Few-shot exemplar questions are
    excluded so a model cannot be scored on its own prompt."""
    import numpy as np
    from datasets import load_dataset

    ds = load_dataset(*DATASET, split=SPLIT)
    shot_qs = {normalize(q) for q, _ in FEW_SHOT}
    order = np.random.default_rng(seed).permutation(len(ds))
    out = []
    for i in order:
        row = ds[int(i)]
        if normalize(row["question"]) in shot_qs:
            continue
        ans = row["answer"]
        aliases = list(dict.fromkeys([ans["value"], *ans["aliases"], *ans.get("normalized_aliases", [])]))
        out.append({"id": row["question_id"], "question": row["question"],
                    "value": ans["value"], "aliases": aliases})
        if len(out) == n:
            break
    return out
