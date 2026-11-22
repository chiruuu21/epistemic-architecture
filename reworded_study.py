"""
Reworded TriviaQA study: does the calibration result survive hard retrieval?
CPU only; reads what reword_triviaqa.py and collect_triviaqa.py --reworded
saved on the GPU.

    python reworded_study.py --tag gpt2
    python reworded_study.py --tag qwen2.5-3b

The first study (calibration_study.py) stored each belief as "<question>
Answer: <answer>." and queried it with the same question. This one crosses
two wordings of the query with two wordings of the belief:

    condition                  query                 stored belief
    1 same Q  -> Q+A belief    original question     question + answer  (the first study)
    2 new Q   -> Q+A belief    paraphrased question  question + answer
    3 same Q  -> statement     original question     declarative statement
    4 new Q   -> statement     paraphrased question  declarative statement  (hardest)

Everything else is calibration_study.run_split, unchanged: 50/50
calibration/test split, half the questions stored, theta_rel and every
Platt map fit on the calibration half, all numbers on the test half. In
conditions 2 and 4 the fallback answer is the model's answer to the
paraphrased question, so the system is scored on what it was asked.

Only questions whose paraphrase AND statement passed the checks in
reword_triviaqa.py are used, in all four conditions, so the rows are
comparable. Condition 1 is therefore recomputed on that subset and will
differ slightly from calibration_study.py.

FIXED BEFORE THE FIRST RUN
  Success for the architecture is unchanged: ECE < 0.05 for the per-path
  P(IK) with AUROC well above 0.5, now in condition 4.
  Expected: retrieval (top-1, relevance AUROC) falls from 1 to 4; FOK
  fires more often on stored questions; system accuracy moves toward the
  LM's own. Whether calibration holds is the open question -- the Platt
  maps are refit per condition, so it can hold even when retrieval is
  poor, as long as the match cosine still ranks right above wrong.
  Section C asks the stricter question: thresholds and maps fit on
  condition 1, used unchanged on condition 4 (a store tuned on easy
  wording, then deployed on hard wording).
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from calibration import auroc, ece
from calibration_study import HEADER, W, evaluate, load, row, run_split

CONDITIONS = [
    ("1 same Q -> Q+A belief", "q", "b_true", "b_wrong", False),
    ("2 new Q  -> Q+A belief", "q_para", "b_true", "b_wrong", True),
    ("3 same Q -> statement", "q", "b_stmt", "b_stmt_wrong", False),
    ("4 new Q  -> statement", "q_para", "b_stmt", "b_stmt_wrong", True),
]


def load_all(tag, root="results"):
    recs, keys, meta = load(tag, root)
    rows = [json.loads(l) for l in open(f"{root}/triviaqa_{tag}_reworded.jsonl")]
    k2 = np.load(f"{root}/triviaqa_{tag}_reworded_keys.npz")
    text = {r["id"]: r for r in map(json.loads, open(f"{root}/triviaqa_reworded.jsonl"))}
    pos = {}  # id -> its row in the reworded key arrays
    for n, r in enumerate(rows):
        pos.setdefault(r["id"], n)
    rw = {i: rows[n] for i, n in pos.items()}
    keep = [n for n, r in enumerate(recs) if r["id"] in pos
            and rw[r["id"]]["para_ok"] and rw[r["id"]]["stmt_ok"]]
    sel2 = [pos[recs[n]["id"]] for n in keep]
    out_recs = [{**recs[n], **rw[recs[n]["id"]], "overlap": text[recs[n]["id"]]["overlap"]} for n in keep]
    out_keys = {name: keys[name][keep] for name in keys}
    out_keys.update({name: k2[name].astype(np.float32)[sel2] for name in k2.files})
    return out_recs, out_keys, meta, len(rw)


def view(recs, keys, q, b_true, b_wrong, para):
    """recs/keys as calibration_study.run_split expects, for one condition."""
    k = {"q": keys[q], "b_true": keys[b_true], "b_wrong": keys[b_wrong]}
    if not para:
        return recs, k
    return [{**r, "correct": r["correct_para"], "conf_seq": r["conf_seq_para"]} for r in recs], k


def measure(r, seed=0):
    test, stored, sim = r["test"], r["stored"], r["sim"]
    t_st, t_un = test[stored[test]], test[~stored[test]]
    own = np.array([r["store"]["ids"][b] == i for i, b in enumerate(r["best"])])
    return {"top1": float(own[t_st].mean()),
            "own_cos": float(sim[t_st][own[t_st]].mean()) if own[t_st].any() else float("nan"),
            "rel_auroc": float(auroc(sim[t_st], sim[t_un])),
            "theta": r["theta"],
            "fires_stored": float(1 - r["use_store"][t_st].mean()),
            "fires_unstored": float(1 - r["use_store"][t_un].mean()),
            "sys_acc": float(r["sys_ok"][test].mean()), "lm_acc": float(r["lm_ok"][test].mean()),
            "binary": evaluate(r["pik_binary"], r["sys_ok"], test, seed),
            "store_only": evaluate(r["pik_store_only"], r["sys_ok"], test, seed),
            "graded": evaluate(r["pik_graded"], r["sys_ok"], test, seed)}


def transfer(r_fit, r_new):
    """theta_rel and the per-path P(IK) fit on one condition's calibration
    half, applied unchanged to another condition's test half."""
    test = r_new["test"]
    w = r_new["store"]["weff"][r_new["best"]]
    use = r_new["sim"] >= r_fit["theta"]
    ok = np.where(use, r_new["store_ok"], r_new["lm_ok"]).astype(float)
    pik = r_fit["pik_model"].predict(r_new["sim"], w, use, r_new["lm_raw"])
    stored = r_new["stored"]
    s = evaluate(pik, ok, test, 0)
    s["fires_stored"] = float(1 - use[test][stored[test]].mean())
    s["sys_acc"] = float(ok[test].mean())
    return s


def report(tag, root, splits):
    recs, keys, meta, n_rew = load_all(tag, root)
    rmeta = json.load(open(f"{root}/triviaqa_reworded_meta.json"))
    out = []
    p = out.append
    p("=" * W)
    p(f"Reworded TriviaQA study: {meta['model']} (layer {meta['layer']})")
    p(f"rewordings by {rmeta['model']}; {len(recs)} of {n_rew} questions passed both checks "
      f"({len(recs) / n_rew:.1%})")
    ov = np.array([r["overlap"] for r in recs])
    p(f"content-word overlap, original vs paraphrased question: mean {ov.mean():.2f} "
      f"(quartiles {np.quantile(ov, .25):.2f} / {np.quantile(ov, .5):.2f} / {np.quantile(ov, .75):.2f})")
    a0 = np.mean([r["correct"] for r in recs])
    a1 = np.mean([r["correct_para"] for r in recs])
    p(f"meaning check: model accuracy on the original wording {a0:.3f}, on the paraphrase {a1:.3f}")
    p("=" * W)

    runs, res = {}, {}
    for name, q, bt, bw, para in CONDITIONS:
        rc, kc = view(recs, keys, q, bt, bw, para)
        runs[name] = run_split(rc, kc, split_seed=0, store_seed=1)
        res[name] = measure(runs[name])

    p("\nA. Retrieval and the FOK decision (test half)")
    p(f"  {'condition':<26}{'top-1':>7}{'own cos':>9}{'rel AUROC':>11}{'theta':>8}"
      f"{'FOK stored':>12}{'FOK unstored':>14}")
    for name, *_ in CONDITIONS:
        m = res[name]
        p(f"  {name:<26}{m['top1']:>7.3f}{m['own_cos']:>9.3f}{m['rel_auroc']:>11.3f}{m['theta']:>8.3f}"
          f"{m['fires_stored']:>12.3f}{m['fires_unstored']:>14.3f}")
    p("  top-1: the stored question retrieves its own belief. own cos: its mean match cosine.")
    p("  rel AUROC: best-match cosine separates stored from unstored questions.")
    p("  FOK stored / unstored: how often the system falls back to the LM (ideal 0 / 1).")

    p("\nB. System accuracy and calibration (test half; maps refit per condition)")
    for name, *_ in CONDITIONS:
        m = res[name]
        p(f"  {name}: system accuracy {m['sys_acc']:.3f} (LM alone {m['lm_acc']:.3f})")
        p(HEADER)
        p(row("binary PIK (paper: weff or 0)", m["binary"]))
        p(row("graded, store signal only", m["store_only"]))
        p(row("graded, per path (store | LM)", m["graded"]))

    hard = CONDITIONS[-1][0]
    p("\nC. Transfer: theta_rel and P(IK) fit on condition 1, used unchanged on harder wording")
    p(f"  {'tested on':<26}{'sys acc':>8}{'FOK stored':>12}{'ECE [95% CI]':>26}{'AUROC':>8}")
    tr = {}
    for name, *_ in CONDITIONS[1:]:
        tr[name] = t = transfer(runs[CONDITIONS[0][0]], runs[name])
        ci = t["ece_ci"]
        p(f"  {name:<26}{t['sys_acc']:>8.3f}{t['fires_stored']:>12.3f}"
          f"{f'{t['ece']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]':>26}{t['auroc']:>8.3f}")

    p(f"\nD. Poisoned store in condition 4 (20% of stored statements carry a wrong answer)")
    p(HEADER)
    _, q, bt, bw, para = CONDITIONS[-1]
    rc, kc = view(recs, keys, q, bt, bw, para)
    poison = {}
    for informative in (True, False):
        rp = run_split(rc, kc, 0, 1, poison=0.2, informative=informative)
        nm = "graded, provenance informative" if informative else "graded, provenance shuffled"
        poison[nm] = evaluate(rp["pik_graded"], rp["sys_ok"], rp["test"], 0)
        p(row(nm, poison[nm]))

    p(f"\nRobustness: per-path P(IK) test ECE across {splits} calibration/test splits (mean, min-max)")
    rob = {}
    for name, q, bt, bw, para in CONDITIONS:
        rc, kc = view(recs, keys, q, bt, bw, para)
        es, us = [], []
        for s in range(splits):
            rs = run_split(rc, kc, split_seed=100 + s, store_seed=1)
            t = rs["test"]
            es.append(ece(rs["pik_graded"][t], rs["sys_ok"][t]))
            us.append(auroc(rs["pik_graded"][t][rs["sys_ok"][t] == 1], rs["pik_graded"][t][rs["sys_ok"][t] == 0]))
        rob[name] = {"ece": es, "auroc": us}
        p(f"  {name:<26} ECE {np.mean(es):.3f} ({min(es):.3f}-{max(es):.3f})   AUROC {np.nanmean(us):.3f}")
    p(f"\nTarget: ECE < 0.05 with AUROC well above 0.5, in condition 4 ({hard.strip()}).")

    text = "\n".join(out)
    js = {"meta": meta, "reword_meta": rmeta, "n_used": len(recs), "n_reworded": n_rew,
          "accuracy_original": float(a0), "accuracy_paraphrase": float(a1),
          "conditions": res, "transfer_from_condition_1": tr, "poison_condition_4": poison,
          "robustness": rob}
    return text, js


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--root", default="results")
    ap.add_argument("--splits", type=int, default=5)
    args = ap.parse_args()
    text, js = report(args.tag, args.root, args.splits)
    print(text)
    with open(f"{args.root}/reworded_{args.tag}.txt", "w") as f:
        f.write(text + "\n")
    with open(f"{args.root}/reworded_{args.tag}.json", "w") as f:
        json.dump(js, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


if __name__ == "__main__":
    main()
