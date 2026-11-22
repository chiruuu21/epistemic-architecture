"""
Trained addresses: a small projection on the concept keys, so a reworded
question lands near the statement that answers it. CPU only, from the keys
collect_triviaqa.py --reworded saved.

    python train_key_projection.py --tag qwen2.5-3b
    python train_key_projection.py --tag gpt2 --ood     # + out-of-domain check

WHY
  reworded_study.py: with a reworded question against a statement belief,
  the right belief is still ranked first 95% of the time (Qwen2.5-3B), but
  the match is weak, so the FOK gate falls back to the LM on 22% of the
  facts the store holds. The keys are raw mean hidden states -- nothing in
  them was ever asked to put a question next to its answer. This is the
  first trained component of the project.

WHAT IS TRAINED
  One shared linear map P on the keys (the models stay frozen). Shared,
  because the loop cannot know whether the current context is a question
  or a statement: evidence sentences and probe questions go through the
  same encoder. Variants, all starting exactly at the untrained baseline:
    diag      x * w                    (d parameters)
    lowrank   x + (x U) V^T, rank r    (2 d r parameters)
    full      x W                      (d^2 parameters)
  Loss: symmetric InfoNCE (temperature 0.05) between each training
  question (both wordings) and its belief (both wordings), against the
  other training beliefs. Keys are centred on the projected belief mean
  and unit-normalised inside the loss, exactly as the store does.

DESIGN (fixed before the first run)
  Split by question, by seed: 1/3 TRAIN (of which 20% is a validation
  slice for early stopping and for choosing the variant), 2/3 EVALUATION.
  The evaluation pool then goes through calibration_study.run_split
  unchanged (its own 50/50 calibration/test split, half the questions
  stored), once with the raw keys and once with projected keys. No
  evaluation question is ever seen in training, and theta_rel / Platt
  maps are fit on projected calibration keys, never on training ones.
  The variant is chosen on validation loss only; every variant's test
  row is printed anyway, so the choice can be audited.

CRITERIA (condition 4: reworded question -> statement belief, test half)
  1. coverage:    fallback on stored facts falls by at least a third,
                  and fallback on unstored questions does not drop by
                  more than 0.03 (no trading one error for the other)
  2. accuracy:    system accuracy rises by at least 2 points
  3. calibration: per-path P(IK) ECE stays < 0.05
  4. no harm:     condition 1 system accuracy within 1 point of baseline
  5. transfer:    (--ood, GPT-2) on the hand-written convergence facts,
                  which share nothing with TriviaQA, probe top-1 and
                  related-vs-unrelated AUROC are not below the baseline.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from calibration import auroc, ece
from calibration_study import W, run_split, unit
from reworded_study import CONDITIONS, load_all, measure, view

TAU = 0.05
KEYS = ("q", "q_para", "b_true", "b_wrong", "b_stmt", "b_stmt_wrong")
VARIANTS = [("diag", 0), ("lowrank", 16), ("lowrank", 64), ("full", 0)]


class Projection(torch.nn.Module):
    def __init__(self, d: int, kind: str, rank: int = 0, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.kind, self.rank = kind, rank
        if kind == "diag":
            self.w = torch.nn.Parameter(torch.ones(d))
        elif kind == "lowrank":
            self.U = torch.nn.Parameter(torch.randn(d, rank, generator=g) / d ** 0.5)
            self.V = torch.nn.Parameter(torch.zeros(d, rank))
        elif kind == "full":
            self.Wm = torch.nn.Parameter(torch.eye(d))
        else:
            raise ValueError(kind)

    def forward(self, x):
        if self.kind == "diag":
            return x * self.w
        if self.kind == "lowrank":
            return x + (x @ self.U) @ self.V.T
        return x @ self.Wm

    def penalty(self):
        """Distance from the untrained baseline (the identity map)."""
        if self.kind == "diag":
            return ((self.w - 1) ** 2).sum()
        if self.kind == "lowrank":
            return (self.V ** 2).sum()
        return ((self.Wm - torch.eye(self.Wm.shape[0])) ** 2).sum()

    @property
    def name(self):
        return f"lowrank r={self.rank}" if self.kind == "lowrank" else self.kind

    def matrix(self) -> np.ndarray:
        """The map as one (d, d) matrix M, with projected = x @ M."""
        d = next(self.parameters()).shape[0]
        with torch.no_grad():
            return self(torch.eye(d)).numpy()


def info_nce(P, Q, B):
    """Symmetric InfoNCE over (query form, belief form) pairs. Q, B: lists
    of (n, d) tensors, row i of every tensor belongs to question i."""
    loss, n = 0.0, 0
    for b in B:
        pb = P(b)
        c = pb.mean(0, keepdim=True)  # the store centres on its beliefs
        pb = torch.nn.functional.normalize(pb - c, dim=-1)
        for q in Q:
            pq = torch.nn.functional.normalize(P(q) - c, dim=-1)
            logits = pq @ pb.T / TAU
            y = torch.arange(len(q))
            loss = loss + torch.nn.functional.cross_entropy(logits, y) \
                + torch.nn.functional.cross_entropy(logits.T, y)
            n += 2
    return loss / n


def train(keys, fit, val, kind, rank, steps=400, lr=1e-3, lam=1e-3, seed=0):
    """Returns (best projection, best validation loss, step it was reached)."""
    torch.manual_seed(seed)
    T = lambda name, idx: torch.from_numpy(keys[name][idx])
    Qf, Bf = [T("q", fit), T("q_para", fit)], [T("b_stmt", fit), T("b_true", fit)]
    Qv, Bv = [T("q", val), T("q_para", val)], [T("b_stmt", val), T("b_true", val)]
    P = Projection(keys["q"].shape[1], kind, rank, seed)
    opt = torch.optim.Adam(P.parameters(), lr=lr)
    best = (float("inf"), 0, None)
    with torch.no_grad():
        best = (float(info_nce(P, Qv, Bv)), 0, {k: v.clone() for k, v in P.state_dict().items()})
    base = best[0]
    for step in range(1, steps + 1):
        opt.zero_grad()
        loss = info_nce(P, Qf, Bf) + lam * P.penalty()
        loss.backward()
        opt.step()
        if step % 10 == 0:
            with torch.no_grad():
                v = float(info_nce(P, Qv, Bv))
            if v < best[0]:
                best = (v, step, {k: t.clone() for k, t in P.state_dict().items()})
    P.load_state_dict(best[2])
    return P, best[0], best[1], base


def project(keys, idx, M, scale):
    return {name: (keys[name][idx] / scale) @ M for name in KEYS}


def evaluate_pool(recs, keys, split_seed=0):
    out = {}
    for name, q, bt, bw, para in CONDITIONS:
        rc, kc = view(recs, keys, q, bt, bw, para)
        out[name] = measure(run_split(rc, kc, split_seed=split_seed, store_seed=1))
    return out


def line(label, m):
    g = m["graded"]
    return (f"  {label:<22}{m['top1']:>7.3f}{m['own_cos']:>9.3f}{m['rel_auroc']:>10.3f}"
            f"{m['fires_stored']:>10.3f}{m['fires_unstored']:>12.3f}{m['sys_acc']:>9.3f}"
            f"{g['ece']:>8.3f}{g['auroc']:>8.3f}")


HEAD = (f"  {'keys':<22}{'top-1':>7}{'own cos':>9}{'rel AUROC':>10}{'FOK stor':>10}"
        f"{'FOK unstor':>12}{'sys acc':>9}{'ECE':>8}{'AUROC':>8}")


def ood_check(meta, M, scale):
    """Hand-written convergence facts (both sets): nothing in common with
    TriviaQA. Keys are computed exactly like the TriviaQA keys."""
    import convergence_data as d0
    import convergence_data_fresh as d1
    from collect_triviaqa import LM, pick_device

    lm = LM(meta["model"], meta["layer"], "float32", pick_device())
    items = [it for d in (d0, d1) for it in d.CONTRADICTED + d.CONSISTENT]
    unrelated = [t for d in (d0, d1) for ctxs in d.UNRELATED for t in ctxs]
    beliefs = lm.pooled([it["A"] for it in items], 16)
    probes = lm.pooled([it["probe"] for it in items], 16)
    evid = lm.pooled([b for it in items for b in it["B"]], 16)
    evid_owner = np.repeat(np.arange(len(items)), 3)
    unrel = lm.pooled(unrelated, 16)
    res = {}
    for label, mat in (("raw keys", None), ("projected", M)):
        f = (lambda x: x) if mat is None else (lambda x: (x / scale) @ mat)
        Bk = f(beliefs)
        c = Bk.mean(0)
        Bk = unit(Bk - c)
        sim = lambda x: unit(f(x) - c) @ Bk.T
        sp, se, su = sim(probes), sim(evid), sim(unrel)
        own = sp[np.arange(len(items)), np.arange(len(items))]
        res[label] = {
            "probe_top1": float((sp.argmax(1) == np.arange(len(items))).mean()),
            "probe_own_cos": float(own.mean()),
            "evidence_top1": float((se.argmax(1) == evid_owner).mean()),
            "related_vs_unrelated_auroc": float(auroc(sp.max(1), su.max(1))),
            "unrelated_best_cos": float(su.max(1).mean()),
            "n_topics": len(items), "n_unrelated": len(unrelated)}
    return res


def report(tag, root, seeds, ood):
    recs, keys, meta, _ = load_all(tag, root)
    n, d = keys["q"].shape
    out = []
    p = out.append
    p("=" * W)
    p(f"Trained key projection: {meta['model']} (layer {meta['layer']}, d = {d})")
    p(f"{n} reworded questions; 1/3 train (20% of it validation), 2/3 evaluation pool")
    p("=" * W)

    results, chosen_M, chosen_scale = [], None, None
    for seed in range(seeds):
        order = np.random.default_rng(1000 + seed).permutation(n)
        n_tr = n // 3
        tr, pool = order[:n_tr], np.sort(order[n_tr:])
        fit, val = tr[: int(0.8 * n_tr)], tr[int(0.8 * n_tr):]
        scale = float(np.linalg.norm(keys["b_stmt"][fit] - keys["b_stmt"][fit].mean(0), axis=1).mean())
        sk = {name: keys[name] / scale for name in KEYS}
        pool_recs = [recs[i] for i in pool]
        base = evaluate_pool(pool_recs, {name: keys[name][pool] for name in KEYS})
        rows, best = {}, None
        for kind, rank in VARIANTS:
            P, vloss, step, vbase = train(sk, fit, val, kind, rank, seed=seed)
            M = P.matrix()
            rows[P.name] = {"val_loss": vloss, "val_loss_untrained": vbase, "step": step,
                            "eval": evaluate_pool(pool_recs, project(keys, pool, M, scale))}
            if best is None or vloss < best[0]:
                best = (vloss, P.name, M)
        results.append({"seed": seed, "n_train": int(n_tr), "n_pool": int(pool.size),
                        "baseline": base, "variants": rows, "chosen": best[1]})
        if seed == 0:
            chosen_M, chosen_scale = best[2], scale

    r0 = results[0]
    hard, easy = CONDITIONS[3][0], CONDITIONS[0][0]
    p(f"\nA. Training (seed 0): validation loss, untrained {next(iter(r0['variants'].values()))['val_loss_untrained']:.3f}")
    for name, v in r0["variants"].items():
        mark = "  <- chosen (lowest validation loss)" if name == r0["chosen"] else ""
        p(f"  {name:<14} validation loss {v['val_loss']:.3f} at step {v['step']:>3}{mark}")

    for cond in (hard, easy):
        p(f"\nB. Test half of the evaluation pool, condition {cond.strip()} (seed 0)")
        p(HEAD)
        p(line("raw keys (baseline)", r0["baseline"][cond]))
        for name, v in r0["variants"].items():
            p(line(name + (" *" if name == r0["chosen"] else ""), v["eval"][cond]))
    p("  * chosen on validation. FOK stor / unstor: how often the system falls back to the")
    p("  LM on stored / unstored questions (ideal 0 / 1). ECE, AUROC: per-path P(IK).")

    p(f"\nC. All four conditions, raw keys -> chosen projection (seed 0)")
    p(f"  {'condition':<26}{'top-1':>14}{'FOK stored':>16}{'sys acc':>16}{'ECE':>16}")
    ch0 = r0["variants"][r0["chosen"]]["eval"]
    for cond, *_ in CONDITIONS:
        b, c = r0["baseline"][cond], ch0[cond]
        f = lambda k, g=None: (f"{(b[g][k] if g else b[k]):.3f} -> {(c[g][k] if g else c[k]):.3f}")
        p(f"  {cond:<26}{f('top1'):>14}{f('fires_stored'):>16}{f('sys_acc'):>16}{f('ece', 'graded'):>16}")

    p(f"\nD. Robustness over {seeds} train/evaluation splits, condition {hard.strip()} (mean, min-max)")
    stats = {}
    for label, pick in (("raw keys", lambda r: r["baseline"][hard]),
                        ("chosen projection", lambda r: r["variants"][r["chosen"]]["eval"][hard])):
        ms = [pick(r) for r in results]
        stats[label] = {k: [float(g(m)) for m in ms] for k, g in (
            ("fires_stored", lambda m: m["fires_stored"]), ("fires_unstored", lambda m: m["fires_unstored"]),
            ("sys_acc", lambda m: m["sys_acc"]), ("top1", lambda m: m["top1"]),
            ("ece", lambda m: m["graded"]["ece"]))}
        s = stats[label]
        fm = lambda k: f"{np.mean(s[k]):.3f} ({min(s[k]):.3f}-{max(s[k]):.3f})"
        p(f"  {label:<18} FOK stored {fm('fires_stored')}  FOK unstored {fm('fires_unstored')}")
        p(f"  {'':<18} sys acc {fm('sys_acc')}  top-1 {fm('top1')}  ECE {fm('ece')}")
    p(f"  variant chosen per split: {', '.join(r['chosen'] for r in results)}")

    b, c = r0["baseline"], ch0
    drop = 1 - c[hard]["fires_stored"] / max(b[hard]["fires_stored"], 1e-9)
    crit = {
        "1 coverage (fallback on stored down >= 1/3, unstored not down > 0.03)":
            drop >= 1 / 3 and c[hard]["fires_unstored"] >= b[hard]["fires_unstored"] - 0.03,
        "2 accuracy (system accuracy up >= 2 points)": c[hard]["sys_acc"] - b[hard]["sys_acc"] >= 0.02,
        "3 calibration (per-path ECE < 0.05)": c[hard]["graded"]["ece"] < 0.05,
        "4 no harm (condition 1 accuracy within 1 point)": c[easy]["sys_acc"] >= b[easy]["sys_acc"] - 0.01,
    }
    ood_res = None
    if ood:
        ood_res = ood_check(meta, chosen_M, chosen_scale)
        p(f"\nE. Out of domain: hand-written convergence facts ({ood_res['raw keys']['n_topics']} topics, "
          f"{ood_res['raw keys']['n_unrelated']} unrelated contexts), seed-0 projection")
        p(f"  {'keys':<12}{'probe top-1':>12}{'probe own cos':>15}{'evidence top-1':>16}"
          f"{'related/unrelated AUROC':>25}{'unrelated best cos':>20}")
        for label, m in ood_res.items():
            p(f"  {label:<12}{m['probe_top1']:>12.3f}{m['probe_own_cos']:>15.3f}{m['evidence_top1']:>16.3f}"
              f"{m['related_vs_unrelated_auroc']:>25.3f}{m['unrelated_best_cos']:>20.3f}")
        p("  probe: a question about a stored belief. evidence: a statement about it.")
        r, pj = ood_res["raw keys"], ood_res["projected"]
        crit["5 transfer (probe top-1 and AUROC not below raw keys)"] = (
            pj["probe_top1"] >= r["probe_top1"]
            and pj["related_vs_unrelated_auroc"] >= r["related_vs_unrelated_auroc"])

    p("\nCriteria (fixed before the first run; seed 0)")
    for k, ok in crit.items():
        p(f"  [{'PASS' if ok else 'FAIL'}] {k}")

    js = {"meta": meta, "tau": TAU, "results": results, "robustness": stats,
          "criteria": {k: bool(v) for k, v in crit.items()}, "ood": ood_res}
    return "\n".join(out), js, chosen_M, chosen_scale, r0["chosen"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--root", default="results")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--ood", action="store_true", help="out-of-domain check (runs the LM; GPT-2 locally)")
    args = ap.parse_args()
    text, js, M, scale, name = report(args.tag, args.root, args.seeds, args.ood)
    print(text)
    with open(f"{args.root}/key_projection_{args.tag}.txt", "w") as f:
        f.write(text + "\n")
    with open(f"{args.root}/key_projection_{args.tag}.json", "w") as f:
        json.dump(js, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    np.savez_compressed(f"{args.root}/key_projection_{args.tag}.npz", M=M.astype(np.float16),
                        scale=np.float32(scale), variant=name)


if __name__ == "__main__":
    main()
