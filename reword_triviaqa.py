"""
Reworded TriviaQA: make retrieval hard on purpose. Run once on Colab (T4):

    python reword_triviaqa.py            # ~15 min, Qwen2.5-3B-Instruct

WHY
  In the first calibration study a stored belief is "<question> Answer:
  <answer>." Its key shares the question's wording, so looking it up with
  the same question is easy (top-1 0.95) and the store-path confidence has
  an easy job. A real store holds facts as statements, and people ask in
  their own words. This script produces, for every question of an existing
  collection (same ids, so every model sees the same rewordings):

    paraphrase   the question asked in different words, answer unchanged
    statement    the fact as one declarative sentence containing the answer

  collect_triviaqa.py --reworded then computes keys and answers for them,
  and reworded_study.py compares easy against hard retrieval.

HOW
  An instruction-tuned model rewrites with greedy decoding (deterministic)
  after three worked examples. It is a different checkpoint from the base
  models under test, and only its text is used.

CHECKS (recorded per item; the analysis keeps items that pass both)
  para_ok   non-empty, a question, not the original, and does not leak
            any answer alias that the original question did not contain
  stmt_ok   non-empty, not a question, and contains the answer string
            (so a poisoned version can be made by swapping the answer)
  overlap   Jaccard overlap of content words between original and
            paraphrase -- reported, not filtered: it says how different
            the rewording really is.
  Whether a paraphrase still asks the same thing is checked downstream:
  collect_triviaqa.py has the model under test answer both wordings, and
  a large accuracy drop on the paraphrases would show meaning drift.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time

import torch

from triviaqa_data import normalize

PARA_SYS = ("You rewrite trivia questions. Ask for the same fact in different words and a "
            "different sentence structure. Keep every detail needed to identify the answer. "
            "Never include the answer. Reply with the rewritten question only.")
PARA_SHOTS = [
    ("Which planet is known as the Red Planet?",
     "What is the name of the planet that people often call the Red Planet?"),
    ("Who wrote the play Romeo and Juliet?",
     "The play Romeo and Juliet was the work of which writer?"),
    ("In which country is the city of Kyoto?",
     "Kyoto is a city located in what country?"),
]
STMT_SYS = ("You turn a trivia question and its answer into one declarative sentence that "
            "states the fact. Use the answer exactly as it is given. Do not write a question. "
            "Reply with the sentence only.")
STMT_SHOTS = [
    ("Which planet is known as the Red Planet?", "Mars", "Mars is the planet known as the Red Planet."),
    ("Who wrote the play Romeo and Juliet?", "William Shakespeare",
     "The play Romeo and Juliet was written by William Shakespeare."),
    ("In which country is the city of Kyoto?", "Japan", "The city of Kyoto is in Japan."),
]

_STOP = set("a an the of in on at to for by with and or is are was were be been which what who whom "
            "whose when where why how that this these those it its as from did do does has have had "
            "name named called known".split())


def content_words(s: str) -> set[str]:
    return {w for w in normalize(s).split() if w not in _STOP}


def overlap(a: str, b: str) -> float:
    A, B = content_words(a), content_words(b)
    return len(A & B) / max(1, len(A | B))


def clean(text: str) -> str:
    """First non-empty line, without wrapping quotes or a leading label."""
    line = next((l.strip() for l in text.splitlines() if l.strip()), "")
    line = re.sub(r"^(rewritten question|question|statement|sentence|answer)\s*:\s*", "", line, flags=re.I)
    return line.strip().strip('"“”').strip()


def leaks(text: str, aliases: list[str], original: str) -> bool:
    """True if `text` contains an answer alias the original question did not."""
    t, o = f" {normalize(text)} ", f" {normalize(original)} "
    for a in aliases:
        a = normalize(a)
        if len(a) >= 3 and f" {a} " in t and f" {a} " not in o:
            return True
    return False


def check_paraphrase(para: str, question: str, aliases: list[str]) -> bool:
    return bool(para) and para.endswith("?") and normalize(para) != normalize(question) \
        and not leaks(para, aliases, question)


def check_statement(stmt: str, value: str) -> bool:
    return bool(stmt) and "?" not in stmt and value.strip().lower() in stmt.lower()


def swap_answer(stmt: str, value: str, new: str) -> str:
    """The statement with its answer replaced (for the poisoned-store run)."""
    return re.sub(re.escape(value.strip()), lambda _: new.strip(), stmt, flags=re.I)


class Rewriter:
    def __init__(self, name: str, dtype: str, device):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device
        self.tok = AutoTokenizer.from_pretrained(name)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        dt = {"float16": torch.float16, "float32": torch.float32}[dtype]
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=dt).to(device).eval()
        if device.type != "cuda":
            for p in self.model.parameters():  # lm_hook.py Fix 10 (mmap SIGBUS on macOS)
                p.data = p.data.clone()

    def _chat(self, system, shots, user):
        msgs = [{"role": "system", "content": system}]
        for u, a in shots:
            msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
        msgs.append({"role": "user", "content": user})
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def run(self, prompts: list[str], bs: int, max_new: int, label: str) -> list[str]:
        out, t0 = [], time.time()
        # longest first inside the loop would reorder; keep order, pad left
        for s in range(0, len(prompts), bs):
            enc = self.tok(prompts[s:s + bs], return_tensors="pt", padding=True).to(self.device)
            gen = self.model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id)
            new = gen[:, enc["input_ids"].shape[1]:]
            out += [clean(t) for t in self.tok.batch_decode(new, skip_special_tokens=True)]
            if (s // bs) % 20 == 0:
                print(f"  {label}: {len(out)}/{len(prompts)} ({time.time() - t0:.0f}s)", flush=True)
        return out

    def paraphrase(self, questions, bs, max_new):
        return self.run([self._chat(PARA_SYS, PARA_SHOTS, q.strip()) for q in questions],
                        bs, max_new, "paraphrase")

    def statement(self, questions, values, bs, max_new):
        fmt = lambda q, a: f"Question: {q.strip()}\nAnswer: {a.strip()}"
        shots = [(fmt(q, a), s) for q, a, s in STMT_SHOTS]
        return self.run([self._chat(STMT_SYS, shots, fmt(q, a)) for q, a in zip(questions, values)],
                        bs, max_new, "statement")


def main():
    from calibration_study import first_occurrence
    from collect_triviaqa import pick_device

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--src", default="results/triviaqa_gpt2.jsonl",
                    help="an existing collection; its questions (and ids) are reworded")
    ap.add_argument("--n", type=int, default=None, help="only the first n (for a dry run)")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    device = pick_device()
    dtype = args.dtype if device.type == "cuda" else "float32"
    items = [json.loads(l) for l in open(args.src)]
    # the sample lists some questions twice (calibration_study.load); reword each once
    items = [items[i] for i in first_occurrence(it["id"] for it in items)][: args.n]
    print(f"{len(items)} questions from {args.src}; loading {args.model} ({dtype}) on {device}", flush=True)
    t0 = time.time()
    rw = Rewriter(args.model, dtype, device)
    qs, vals = [it["question"] for it in items], [it["value"] for it in items]
    paras = rw.paraphrase(qs, args.bs, args.max_new)
    stmts = rw.statement(qs, vals, args.bs, args.max_new)

    os.makedirs(args.out, exist_ok=True)
    rows = []
    for it, pa, st in zip(items, paras, stmts):
        rows.append({"id": it["id"], "question": it["question"], "value": it["value"],
                     "paraphrase": pa, "statement": st,
                     "para_ok": check_paraphrase(pa, it["question"], it["aliases"]),
                     "stmt_ok": check_statement(st, it["value"]),
                     "overlap": round(overlap(it["question"], pa), 3)})
    with open(f"{args.out}/triviaqa_reworded.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n = len(rows)
    both = [r for r in rows if r["para_ok"] and r["stmt_ok"]]
    import transformers
    meta = {"model": args.model, "dtype": dtype, "device": str(device), "n": n, "src": args.src,
            "para_ok": sum(r["para_ok"] for r in rows) / n, "stmt_ok": sum(r["stmt_ok"] for r in rows) / n,
            "both_ok": len(both) / n,
            "mean_overlap_ok": sum(r["overlap"] for r in both) / max(1, len(both)),
            "seconds": round(time.time() - t0, 1), "transformers": transformers.__version__}
    with open(f"{args.out}/triviaqa_reworded_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))
    print("\nexamples:")
    for r in rows[:8]:
        print(f"  Q  {r['question']}\n  P  {r['paraphrase']}  [{'ok' if r['para_ok'] else 'REJECTED'}, "
              f"overlap {r['overlap']}]\n  S  {r['statement']}  [{'ok' if r['stmt_ok'] else 'REJECTED'}]\n")


if __name__ == "__main__":
    main()
