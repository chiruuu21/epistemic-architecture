"""
Calibration study on TriviaQA -- the proposal's success metric (ECE < 0.05),
measured so that a pass would actually mean something. CPU only; reads
what collect_triviaqa.py saved on the GPU.

    python calibration_study.py --tag gpt2
    python calibration_study.py --tag qwen2.5-3b

DESIGN (all choices fixed before looking at test numbers)
  Split. Questions are split 50/50 into CALIBRATION and TEST by seed. Every
    fitted quantity (Platt maps, theta_rel) is fit on calibration only;
    every reported number is on test only, with a 95% bootstrap interval
    over test questions. --splits K repeats with K split seeds and reports
    the spread, so one lucky split cannot carry a result.
  Store. Independently, 50% of all questions get a stored belief: the
    question with its true answer (triviaqa_data.belief_text), VERIFIED,
    weight 0.9. The rest have nothing stored. Keys are pooled hidden
    states at the model's layer, centred on the mean of the stored
    beliefs' keys (belief-side data only; no query is used to fit it).
  System. For each question, context retrieval's last-token query is the
    pooled question key (collect_triviaqa.py verifies this equals what
    EpTLoop computes). If its best cosine >= theta_rel the system answers
    with the retrieved belief's answer (store path); otherwise FOK fires
    and the system falls back to the LM's own closed-book answer.

WHAT IS REPORTED
  A. LM alone: raw answer probability, Platt-scaled probability, and the
     constant "always predict the base rate" baseline. The baseline is the
     sanity check that ECE alone cannot be the success criterion.
  B. The loop on TriviaQA: relevance AUROC (stored vs not stored), FOK
     fire rates at the held-out theta_rel, store-path accuracy.
  C. EpT system calibration: the paper's binary PIK (0 / weff) versus
     graded PIK (a Platt map of match cosine and weff, fit on calibration).
  D. Poisoned-store simulation: 20% of stored beliefs carry another
     question's answer. Run twice -- provenance informative (poisoned
     beliefs are STATED, weight 0.6; clean ones VERIFIED, 0.9) and
     provenance shuffled (same tiers, assigned at random). Shows how much
     of EpT's calibration depends on trust tiers carrying information.
     It is a simulation: in it the tiers are informative by construction.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from belief_store import SOURCE_TRUST, SourceTier
from calibration import PlattScaler, bootstrap, ece, logit, reliability, summary, auroc
from triviaqa_data import is_correct

W = 78


def load(tag, root="results"):
    recs = [json.loads(l) for l in open(f"{root}/triviaqa_{tag}.jsonl")]
    k = np.load(f"{root}/triviaqa_{tag}_keys.npz")
    meta = json.load(open(f"{root}/triviaqa_{tag}_meta.json"))
    return recs, {n: k[n].astype(np.float32) for n in k.files}, meta


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def fmt_ci(point, lo_hi):
    return f"{point:.3f} [{lo_hi[0]:.3f}, {lo_hi[1]:.3f}]"


def evaluate(conf, correct, idx, seed):
    """summary() on idx, with bootstrap CIs for ECE, AUROC and accuracy."""
    c, y = conf[idx], correct[idx]
    s = summary(c, y)
    n = idx.size
    s["ece_ci"] = bootstrap(lambda b: ece(c[b], y[b]), n, seed=seed)
    s["auroc_ci"] = bootstrap(lambda b: auroc(c[b][y[b] == 1], c[b][y[b] == 0]), n, seed=seed)
    s["acc_ci"] = bootstrap(lambda b: y[b].mean(), n, seed=seed)
    return s


def row(name, s):
    return (f"  {name:<34}{s['accuracy']:>6.3f} {s['mean_conf']:>6.3f} "
            f"{fmt_ci(s['ece'], s['ece_ci']):>24} {s['brier']:>6.3f} "
            f"{fmt_ci(s['auroc'], s['auroc_ci']):>24}")


HEADER = (f"  {'confidence':<34}{'acc':>6} {'conf':>6} {'ECE [95% CI]':>24} {'Brier':>6} "
          f"{'AUROC right>wrong [CI]':>24}")


def build_store(recs, keys, stored, poison=None, informative=True, seed=0):
    """Returns per-store-entry arrays: key (centred, unit), answer text,
    weff, and whether the entry is poisoned."""
    ids = np.flatnonzero(stored)
    pois = np.zeros(len(recs), dtype=bool)
    if poison:
        rng = np.random.default_rng(seed + 7)
        pois[rng.choice(ids, int(round(poison * ids.size)), replace=False)] = True
    raw = np.where(pois[:, None], keys["b_wrong"], keys["b_true"])[ids]
    center = raw.mean(0)
    answers = [recs[i]["wrong_value"] if pois[i] else recs[i]["value"] for i in ids]
    clean_w = 0.9 * SOURCE_TRUST[SourceTier.VERIFIED]
    bad_w = 0.6 * SOURCE_TRUST[SourceTier.STATED]
    tier_bad = pois[ids].copy()
    if poison and not informative:
        tier_bad = np.random.default_rng(seed + 11).permutation(tier_bad)
    weff = np.where(tier_bad, bad_w, clean_w)
    return {"ids": ids, "K": unit(raw - center), "center": center,
            "answers": answers, "weff": weff, "poisoned": pois[ids]}


def retrieve(recs, keys, store):
    Q = unit(keys["q"] - store["center"])
    sim = Q @ store["K"].T
    best = sim.argmax(1)
    best_sim = sim[np.arange(len(recs)), best]
    store_correct = np.array([is_correct(store["answers"][b], r["aliases"])
                              for b, r in zip(best, recs)])
    return best, best_sim, store_correct


def pick_theta(sim, store_ok, lm_ok, idx):
    """theta_rel maximising system accuracy on the calibration split."""
    cands = np.unique(np.quantile(sim[idx], np.linspace(0, 1, 201)))
    accs = [np.where(sim[idx] >= t, store_ok[idx], lm_ok[idx]).mean() for t in cands]
    return float(cands[int(np.argmax(accs))])


def run_split(recs, keys, split_seed, store_seed, poison=None, informative=True):
    n = len(recs)
    rng = np.random.default_rng(split_seed)
    calib = np.zeros(n, dtype=bool)
    calib[rng.permutation(n)[: n // 2]] = True
    cal, test = np.flatnonzero(calib), np.flatnonzero(~calib)
    stored = np.zeros(n, dtype=bool)
    stored[np.random.default_rng(store_seed).permutation(n)[: n // 2]] = True

    lm_ok = np.array([r["correct"] for r in recs], dtype=float)
    lm_raw = np.array([r["conf_seq"] for r in recs])

    # A. LM alone
    platt_lm = PlattScaler().fit(logit(lm_raw[cal]), lm_ok[cal])
    lm_cal = platt_lm.predict(logit(lm_raw))
    const = np.full(n, lm_ok[cal].mean())

    # B/C. the loop
    st = build_store(recs, keys, stored, poison, informative, seed=store_seed)
    best, sim, store_ok = retrieve(recs, keys, st)
    theta = pick_theta(sim, store_ok, lm_ok, cal)
    use_store = sim >= theta
    sys_ok = np.where(use_store, store_ok, lm_ok).astype(float)
    w = st["weff"][best]

    # paper's binary PIK: weff on the store path, 0 when FOK fires (the
    # fallback answer then has no confidence from the loop at all)
    pik_binary = np.where(use_store, w, 0.0)
    # graded PIK: Platt map of (cosine, log weff) on the store path, fit on
    # calibration store-path rows; the FOK path uses the LM's Platt map
    feats = np.stack([sim, np.log(w)], 1)
    sp_cal = cal[use_store[cal]]
    platt_store = PlattScaler().fit(feats[sp_cal], store_ok[sp_cal].astype(float))
    pik_graded = np.where(use_store, platt_store.predict(feats), lm_cal)

    return {"cal": cal, "test": test, "stored": stored, "lm_ok": lm_ok, "lm_raw": lm_raw,
            "lm_cal": lm_cal, "const": const, "sim": sim, "best": best, "store_ok": store_ok,
            "theta": theta, "use_store": use_store, "sys_ok": sys_ok,
            "pik_binary": pik_binary, "pik_graded": pik_graded, "store": st,
            "platt_store": platt_store.coef_}


def report(tag, recs, keys, meta, splits):
    out = []
    p = out.append
    p("=" * W)
    p(f"Calibration study: {meta['model']} (layer {meta['layer']}, {meta['dtype']}, "
      f"{meta.get('gpu') or meta['device']})")
    p(f"TriviaQA rc.nocontext validation, n={len(recs)} (seed {meta['seed']}); "
      f"calibration/test 50/50; 95% bootstrap CIs over test questions")
    p(f"loop check: |EpTLoop last-token cosine - pooled cosine| <= "
      f"{meta['loop_check']['max_abs_diff']:.1e} over {meta['loop_check']['n']} questions")
    p("=" * W)

    r = run_split(recs, keys, split_seed=0, store_seed=1)
    test, seed = r["test"], 0
    lenient = np.mean([x["correct_lenient"] for x in recs])
    p(f"\nA. LM alone (closed-book, 5-shot, greedy). Accuracy (EM) {r['lm_ok'].mean():.3f}; "
      f"lenient match {lenient:.3f}")
    p(HEADER)
    sA = {k: evaluate(r[k], r["lm_ok"], test, seed) for k in ("lm_raw", "lm_cal", "const")}
    p(row("answer probability (raw)", sA["lm_raw"]))
    p(row("answer probability, Platt-scaled", sA["lm_cal"]))
    p(row("constant = calib accuracy", sA["const"]))
    p("  The constant row is the sanity check: near-zero ECE with AUROC 0.5. A low ECE")
    p("  only counts if AUROC is clearly above 0.5 as well.")

    stored, sim = r["stored"], r["sim"]
    t_st, t_un = test[stored[test]], test[~stored[test]]
    au = auroc(sim[t_st], sim[t_un])
    au_ci = bootstrap(lambda b: auroc(sim[test[b]][stored[test[b]]], sim[test[b]][~stored[test[b]]]),
                      test.size, seed=seed)
    own = np.array([r["store"]["ids"][b] == i for i, b in enumerate(r["best"])])
    p(f"\nB. The loop on TriviaQA ({stored.sum()} beliefs stored, half the questions)")
    p(f"  relevance AUROC, stored vs not stored (best-match cosine): {fmt_ci(au, au_ci)}")
    p(f"  top-1 retrieval of the question's own belief (stored, test): {own[t_st].mean():.3f}")
    p(f"  theta_rel (fit on calibration to maximise system accuracy): {r['theta']:.3f}")
    fires_st = 1 - r["use_store"][t_st].mean()
    fires_un = 1 - r["use_store"][t_un].mean()
    p(f"  FOK fires (test): stored {fires_st:.3f}   not stored {fires_un:.3f}")
    sp = test[r["use_store"][test]]
    p(f"  store path: {sp.size}/{test.size} test questions, accuracy {r['store_ok'][sp].mean():.3f}; "
      f"wrongly used on unstored: {(~stored[sp]).sum()}")

    p(f"\nC. EpT system (store answer if relevant, else LM): accuracy "
      f"{r['sys_ok'][test].mean():.3f} vs LM alone {r['lm_ok'][test].mean():.3f}")
    p(HEADER)
    sC = {k: evaluate(r[k], r["sys_ok"], test, seed) for k in ("pik_binary", "pik_graded")}
    p(row("binary PIK (paper: weff or 0)", sC["pik_binary"]))
    p(row("graded PIK (held-out Platt)", sC["pik_graded"]))
    a, b = r["platt_store"][:2], r["platt_store"][2]
    p(f"  graded store-path map: sigmoid({a[0]:.1f}*cos {a[1]:+.2f}*log weff {b:+.2f})")
    p("  reliability, graded PIK on test (equal-width bins):")
    p(f"    {'bin':<13}{'n':>6}{'conf':>8}{'acc':>8}")
    for lo, hi, cnt, mc, acc in reliability(r["pik_graded"][test], r["sys_ok"][test]):
        p(f"    {lo:.1f}-{hi:.1f}{'':<6}{cnt:>6}{mc:>8.3f}{acc:>8.3f}")

    p("\nD. Poisoned-store simulation (20% of stored beliefs carry a wrong answer)")
    p(HEADER)
    sD = {}
    for informative in (True, False):
        rp = run_split(recs, keys, 0, 1, poison=0.2, informative=informative)
        name = "graded, provenance informative" if informative else "graded, provenance shuffled"
        sD[name] = evaluate(rp["pik_graded"], rp["sys_ok"], test, seed)
        p(row(name, sD[name]))
    p("  Informative provenance is built in by the simulation; the gap between the two")
    p("  rows is how much calibration the trust tiers can contribute, not evidence that")
    p("  real-world tiers are that informative.")

    p(f"\nRobustness: test ECE across {splits} calibration/test splits (mean, min-max)")
    rows = {"LM Platt": [], "constant": [], "EpT binary PIK": [], "EpT graded PIK": []}
    aus = {k: [] for k in rows}
    for s in range(splits):
        rs = run_split(recs, keys, split_seed=100 + s, store_seed=1)
        t = rs["test"]
        for name, conf, ok in (("LM Platt", rs["lm_cal"], rs["lm_ok"]),
                               ("constant", rs["const"], rs["lm_ok"]),
                               ("EpT binary PIK", rs["pik_binary"], rs["sys_ok"]),
                               ("EpT graded PIK", rs["pik_graded"], rs["sys_ok"])):
            rows[name].append(ece(conf[t], ok[t]))
            aus[name].append(auroc(conf[t][ok[t] == 1], conf[t][ok[t] == 0]))
    for name in rows:
        e, u = np.array(rows[name]), np.array(aus[name])
        p(f"  {name:<18} ECE {e.mean():.3f} ({e.min():.3f}-{e.max():.3f})   "
          f"AUROC {np.nanmean(u):.3f}")
    p(f"\nTarget from the proposal: ECE < 0.05 (with AUROC well above 0.5).")

    text = "\n".join(out)
    js = {"meta": meta, "A": sA, "C": sC, "D": sD, "theta_rel": r["theta"],
          "relevance_auroc": [au, au_ci], "fok_fires": {"stored": fires_st, "not_stored": fires_un}}
    return text, js


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--root", default="results")
    ap.add_argument("--splits", type=int, default=5)
    args = ap.parse_args()
    recs, keys, meta = load(args.tag, args.root)
    text, js = report(args.tag, recs, keys, meta, args.splits)
    print(text)
    with open(f"{args.root}/calibration_{args.tag}.txt", "w") as f:
        f.write(text + "\n")
    with open(f"{args.root}/calibration_{args.tag}.json", "w") as f:
        json.dump(js, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


if __name__ == "__main__":
    main()
