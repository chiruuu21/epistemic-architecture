"""
Offline checks for the reworded TriviaQA pipeline: the rewording checks
(reword_triviaqa.py), duplicate-question removal (calibration_study.load)
and the per-condition views (reworded_study.py). No model, no downloads.
"""

import json
import os
import tempfile

import numpy as np

from calibration_study import first_occurrence, load
from reword_triviaqa import check_paraphrase, check_statement, clean, overlap, swap_answer
from reworded_study import CONDITIONS, view

Q = "Who wrote the play Romeo and Juliet?"
ALIASES = ["William Shakespeare", "Shakespeare"]


def test_paraphrase_checks():
    assert check_paraphrase("Romeo and Juliet was the work of which writer?", Q, ALIASES)
    assert not check_paraphrase("", Q, ALIASES)
    assert not check_paraphrase("who wrote the play Romeo and Juliet", Q, ALIASES)        # not a question
    assert not check_paraphrase("Who wrote the play, Romeo and Juliet?", Q, ALIASES)      # same after normalising
    assert not check_paraphrase("Which play did Shakespeare write about Verona?", Q, ALIASES)  # leaks the answer
    # an alias the ORIGINAL already contains is not a leak
    q2 = "In which country is Japan's old capital Kyoto?"
    assert check_paraphrase("Kyoto, once the capital of Japan, lies in what country?", q2, ["Japan"])
    assert overlap(Q, Q) == 1.0 and overlap(Q, "How tall is Everest?") == 0.0
    assert clean('Rewritten question: "What is it?"\nextra line') == "What is it?"
    print("[PASS] paraphrase checks: empty / not a question / unchanged / answer leak rejected")


def test_statement_checks_and_poison_swap():
    s = "The play Romeo and Juliet was written by William Shakespeare."
    assert check_statement(s, "william shakespeare")          # case-insensitive
    assert not check_statement("Who wrote Romeo and Juliet? William Shakespeare.", "William Shakespeare")
    assert not check_statement("The play was written by the Bard.", "William Shakespeare")
    assert swap_answer(s, "william shakespeare", "India") == "The play Romeo and Juliet was written by India."
    # regex metacharacters in an answer, and a backslash in the replacement, are taken literally
    assert swap_answer("It costs $5 (five).", "$5 (five)", r"\1") == r"It costs \1."
    print("[PASS] statement must contain the answer and not be a question; answer swap is literal")


def test_duplicate_questions_are_dropped_at_load():
    assert first_occurrence(["a", "b", "a", "c", "b"]) == [0, 1, 3]
    with tempfile.TemporaryDirectory() as d:
        ids = ["a", "b", "a", "c"]
        with open(os.path.join(d, "triviaqa_t.jsonl"), "w") as f:
            for n, i in enumerate(ids):
                f.write(json.dumps({"id": i, "row": n}) + "\n")
        np.savez(os.path.join(d, "triviaqa_t_keys.npz"), q=np.arange(8, dtype=np.float32).reshape(4, 2))
        json.dump({"n": 4}, open(os.path.join(d, "triviaqa_t_meta.json"), "w"))
        recs, keys, meta = load("t", d)
    assert [r["row"] for r in recs] == [0, 1, 3]
    assert keys["q"].tolist() == [[0, 1], [2, 3], [6, 7]]  # keys stay aligned with records
    assert meta["n_collected"] == 4 and meta["n_unique"] == 3
    print("[PASS] duplicate question ids dropped at load; keys stay aligned")


def test_condition_views():
    recs = [{"correct": True, "conf_seq": 0.9, "correct_para": False, "conf_seq_para": 0.2}]
    keys = {k: np.full((1, 2), v, dtype=np.float32) for v, k in
            enumerate(["q", "q_para", "b_true", "b_wrong", "b_stmt", "b_stmt_wrong"])}
    r1, k1 = view(recs, keys, *CONDITIONS[0][1:])
    r4, k4 = view(recs, keys, *CONDITIONS[3][1:])
    assert r1[0]["correct"] is True and r1[0]["conf_seq"] == 0.9
    assert r4[0]["correct"] is False and r4[0]["conf_seq"] == 0.2  # scored on the paraphrase it was asked
    assert (k1["q"][0, 0], k1["b_true"][0, 0], k1["b_wrong"][0, 0]) == (0, 2, 3)
    assert (k4["q"][0, 0], k4["b_true"][0, 0], k4["b_wrong"][0, 0]) == (1, 4, 5)
    assert recs[0]["correct"] is True  # the input is not modified
    print("[PASS] condition views pick the right keys and the answer to the wording asked")


if __name__ == "__main__":
    test_paraphrase_checks()
    test_statement_checks_and_poison_swap()
    test_duplicate_questions_are_dropped_at_load()
    test_condition_views()
