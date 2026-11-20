"""
Experiment 4.2 -- belief convergence under contradicting evidence.

Paper: "Pre-load belief A (weight=0.8, INFERRED). Three turns of context
with belief B (VERIFIED). Record weight trajectory and retrieval switch
point." Here with 20 such pairs at once in one shared store, plus 10
consistent beliefs (evidence agrees: should be reinforced, never contested)
and 10 unrelated contexts (nothing stored: should change nothing), so
cross-talk, false contests and RIF side-effects are measured, not assumed.

    python exp_convergence.py            # ~2 min on a laptop CPU

PIPELINE (real components end to end, nothing mocked)
  keys      GPT-2 small layer 8, residual - b_dec (ResidualKeyEncoder;
            Report 3: retrieves better than SAE codes). A belief's key is
            the mean over its tokens.
  retrieval EpTLoop in context mode, theta_rel = 0.608 -- the held-out
            threshold from analysis_heldout.py, fit on the 50+50 prompt
            set, never on this one. min_ctx = 3.
  update    each context is observed once per turn; every belief it
            retrieves is reconsolidated against that context with the
            real DeBERTa-v3 NLI backend (contradiction >= 0.70,
            entailment >= 0.50). The evidence sentence is the claim:
            Experiment 4.2 is about updating from observed context, so no
            generation or claim extraction is involved (that would mix
            extractor errors into the result).
  clock     one turn = one day (decay constants are per day; see
            belief_store.py UNITS).
  probe     after each turn, a neutral question per topic is run through
            store.peek() -- what the store would answer, read-only.

CONFIGURATIONS (one ablation each; all values fixed before the first run)
  paper            the spec as written: cosine retrieval, successor
                   INFERRED w=0.5, contradiction only flips state, RIF on
  + trust          trust-aware retrieval (BeliefStore Fix 3)
  + tier           successor takes the evidence's tier (VERIFIED)
  revised          trust + tier + weaken (eta_contest 0.1) + dedup +
                   supersede (reconsolidation.py docstring)
  revised - X      drop one revision at a time; "- RIF" sets eta_inhibit 0
  no recon         retrieval only, no NLI update (the floor)
"""

from __future__ import annotations

import json
import random
import time
from collections import Counter

import numpy as np
import torch

from belief_store import BeliefState, ContentType, SourceTier
from convergence_data import CONSISTENT, CONTRADICTED, UNRELATED
from ept_loop import EpTLoop
from nli_backend import HFNLIBackend, NLIResult

THETA_REL = 0.608     # analysis_heldout.py, resid - b_dec, Report 4 Table 1
MIN_CTX = 3
TURNS = 3
THETA_CONTEST = 0.70  # post hoc, see POSTHOC
W = 78

BASE = dict(rank="cosine", tiers=False, eta_contest=0.0, dedup=False, supersede=False,
            eta_inhibit=0.05, recon=True, rif_min_sim=0.0, contest_min_sim=None)
REVISED = dict(BASE, rank="trust", tiers=True, eta_contest=0.1, dedup=True, supersede=True)
CONFIGS = [
    ("paper", BASE),
    ("paper + trust", dict(BASE, rank="trust")),
    ("paper + tier", dict(BASE, tiers=True)),
    ("revised", REVISED),
    ("revised - trust", dict(REVISED, rank="cosine")),
    ("revised - tier", dict(REVISED, tiers=False)),
    ("revised - weaken", dict(REVISED, eta_contest=0.0)),
    ("revised - dedup", dict(REVISED, dedup=False)),
    ("revised - supersede", dict(REVISED, supersede=False)),
    ("revised - RIF", dict(REVISED, eta_inhibit=0.0)),
    ("no recon", dict(REVISED, recon=False)),
]
# Added AFTER the first run, to address the two failures its section B
# exposed (trust ranking crossing topics; RIF taxing bystanders at cos ~0.6).
# Reported separately and labelled post hoc.
POSTHOC = [
    ("revised, lineage", dict(REVISED, rank="lineage")),
    ("revised, RIF floor", dict(REVISED, rif_min_sim=THETA_REL)),
    ("revised, both", dict(REVISED, rank="lineage", rif_min_sim=THETA_REL)),
    ("paper, both", dict(BASE, rank="lineage", rif_min_sim=THETA_REL)),
    # added after the second run: NLI calls ~4% of unrelated pairs
    # CONTRADICTION, while key cosine separates same-topic from cross-topic
    # evidence at AUROC 0.99 -- so retrieval must vouch for the topic before
    # NLI may contest. 0.70 was chosen ON THIS SET (sweep in section E).
    ("revised, both + gate", dict(REVISED, rank="lineage", rif_min_sim=THETA_REL,
                                  contest_min_sim=THETA_CONTEST)),
]


# -- features, computed once ------------------------------------------------

def encode_all(lm, enc):
    texts = set()
    for item in CONTRADICTED + CONSISTENT:
        texts |= {item["A"], item["probe"], *item["B"]}
    for ctxs in UNRELATED:
        texts |= set(ctxs)
    codes = {}
    for t in sorted(texts):
        codes[t] = enc(lm.resid(t).to(enc.device)).float()  # (S, D), BOS dropped
    return codes


def key(codes, text):
    return codes[text].mean(0).cpu().numpy()


# -- one run --------------------------------------------------------------

def run_config(cfg, codes, enc, nli, seed=0, theta=THETA_REL):
    loop = EpTLoop(d_model=enc.d_in, encoder=enc, retrieval="context",
                   theta_rel=theta, min_ctx=MIN_CTX, nli_fn=nli,
                   rank=cfg["rank"], eta_inhibit=cfg["eta_inhibit"],
                   eta_contest=cfg["eta_contest"], dedup_successors=cfg["dedup"],
                   supersede=cfg["supersede"], rif_min_sim=cfg["rif_min_sim"],
                   contest_min_sim=cfg["contest_min_sim"])
    A_ids, C_ids = [], []
    for item in CONTRADICTED:
        A_ids.append(loop.add_belief(key(codes, item["A"]), item["A"], 0.8, SourceTier.INFERRED,
                                     ContentType[item["type"]], t_now=0).id)
    for item in CONSISTENT:
        C_ids.append(loop.add_belief(key(codes, item["A"]), item["A"], 0.8, SourceTier.VERIFIED,
                                     ContentType[item["type"]], t_now=0).id)
    store = loop.store
    owner = {eid: ("A", i) for i, eid in enumerate(A_ids)}
    owner.update({eid: ("C", i) for i, eid in enumerate(C_ids)})

    def lineage_root(e):
        while e.supersedes is not None:
            e = store._entries[e.supersedes]
        return e.id

    def probe(day):
        out = {"A": [], "C": []}
        for kind, items, ids in (("A", CONTRADICTED, A_ids), ("C", CONSISTENT, C_ids)):
            for item, root in zip(items, ids):
                hit = store.peek(key(codes, item["probe"]), day, min_sim=theta, rank=cfg["rank"])
                if hit is None or hit[2] <= 0.0:
                    out[kind].append("none")  # nothing relevant, or only a dead (SUPERSEDED) entry
                    continue
                e = hit[0]
                if lineage_root(e) != root:
                    out[kind].append("other")
                elif e.id == root:
                    out[kind].append("orig")
                else:
                    out[kind].append("succ")
        return out

    def snapshot(day):
        a = [store._entries[i] for i in A_ids]
        c = [store._entries[i] for i in C_ids]
        return {
            "A_weight": [e.weight for e in a], "A_weff": [e.effective_weight(day) for e in a],
            "A_state": [e.state.value for e in a],
            "succ_weff": [max([s.effective_weight(day) for s in store.successors(i)] or [0.0])
                          for i in A_ids],
            "n_succ": [len(store.successors(i)) for i in A_ids],
            "C_weight": [e.weight for e in c], "C_state": [e.state.value for e in c],
            "probe": probe(day),
        }

    traj = [snapshot(0)]
    events = []  # one row per (context, retrieved belief)
    rng = random.Random(seed)
    for turn in range(1, TURNS + 1):
        stream = ([("A", i, it["B"][turn - 1]) for i, it in enumerate(CONTRADICTED)]
                  + [("C", i, it["B"][turn - 1]) for i, it in enumerate(CONSISTENT)]
                  + [("U", i, ctxs[turn - 1]) for i, ctxs in enumerate(UNRELATED)])
        rng.shuffle(stream)
        for kind, idx, text in stream:
            loop.reset_generation()
            loop.step(codes[text][None].to(loop.device) + enc.center, t_now=turn)
            flagged = loop.pending_flagged
            if cfg["recon"]:
                summ = loop.finish_generation(
                    text, t_now=turn, claims={e.id: text for e in flagged},
                    claim_tier=SourceTier.VERIFIED if cfg["tiers"] else None)
                verdicts = {eid: v.label.value for eid, v in summ["verdicts"].items()}
                # record what HAPPENED, not what NLI said: a contradiction the
                # contest gate held back changed nothing
                for eid in summ.get("contest_gated", []):
                    verdicts[eid] = "GATED"
            else:
                loop.reset_generation()
                verdicts = {}
            for e in flagged:
                root = lineage_root(e)
                events.append({"turn": turn, "src": kind, "src_idx": idx,
                               "hit": owner.get(root, ("?", -1)), "is_root": e.id == root,
                               "verdict": verdicts.get(e.id, "-")})
        traj.append(snapshot(turn))
    return {"traj": traj, "events": events, "n_entries": len(store.entries)}


# -- summarising ----------------------------------------------------------

def summarise(res):
    tr, ev = res["traj"], res["events"]
    n = len(CONTRADICTED)
    s = {}
    # retrieval: did each turn's evidence retrieve its own belief (or lineage)?
    own = lambda e: e["hit"] == (e["src"], e["src_idx"])
    hits = {(e["turn"], e["src"], e["src_idx"]) for e in ev if own(e)}
    s["A_detect"] = np.mean([(t, "A", i) in hits for t in range(1, TURNS + 1) for i in range(n)])
    s["C_detect"] = np.mean([(t, "C", i) in hits for t in range(1, TURNS + 1)
                             for i in range(len(CONSISTENT))])
    a_own = [e for e in ev if e["src"] == "A" and own(e)]
    s["A_contra_rate"] = (np.mean([e["verdict"] == "CONTRADICTION" for e in a_own if e["is_root"]])
                          if any(e["is_root"] for e in a_own) else float("nan"))
    s["off_target_flags"] = sum(1 for e in ev if not own(e) and e["src"] != "U")
    s["unrelated_flags"] = sum(1 for e in ev if e["src"] == "U")
    s["false_contests"] = sum(1 for e in ev if e["verdict"] == "CONTRADICTION"
                              and (e["src"] != "A" or not own(e)))
    s["C_contested"] = sum(st != "ACTIVE" for st in tr[-1]["C_state"])
    s["gated"] = sum(1 for e in ev if e["verdict"] == "GATED")
    s["gated_own"] = sum(1 for e in ev if e["verdict"] == "GATED" and e["src"] == "A" and own(e))
    # convergence
    s["switch"] = [np.mean([p == "succ" for p in t["probe"]["A"]]) for t in tr]
    # topics whose probe reaches the belief at all before any update (t0):
    # a probe below theta_rel can never show a switch, whatever the update does
    reach = [i for i, p in enumerate(tr[0]["probe"]["A"]) if p == "orig"]
    s["n_reach"] = len(reach)
    s["switch_reach"] = [float(np.mean([t["probe"]["A"][i] == "succ" for i in reach]))
                         for t in tr]
    s["probe_final"] = Counter(tr[-1]["probe"]["A"])
    first = []
    for i in range(n):
        ts = [k for k, t in enumerate(tr) if t["probe"]["A"][i] == "succ"]
        first.append(ts[0] if ts else None)
    s["switch_turn"] = first
    s["A_weight"] = [float(np.mean(t["A_weight"])) for t in tr]
    s["A_weff"] = [float(np.mean(t["A_weff"])) for t in tr]
    s["succ_weff"] = [float(np.mean(t["succ_weff"])) for t in tr]
    s["A_states"] = Counter(tr[-1]["A_state"])
    s["n_succ_mean"] = float(np.mean(tr[-1]["n_succ"]))
    s["n_succ_max"] = int(np.max(tr[-1]["n_succ"]))
    # of the consistent topics whose probe reached their belief at t0, the
    # share still answered by it at t3 (each config against its own t0, so
    # ranking differences before any update don't count as damage)
    c_reach = [i for i, p in enumerate(tr[0]["probe"]["C"]) if p == "orig"]
    s["C_reach"] = len(c_reach)
    s["C_keep"] = f"{sum(tr[-1]['probe']['C'][i] == 'orig' for i in c_reach)}/{len(c_reach)}"
    s["C_weight"] = float(np.mean(tr[-1]["C_weight"]))
    return s


def nli_ceiling(nli):
    """NLI alone on the right (belief, evidence) pairs -- what reconsolidation
    could reach with perfect retrieval."""
    con = [nli.classify(it["A"], b).label for it in CONTRADICTED for b in it["B"]]
    ent = [nli.classify(it["A"], b).label for it in CONSISTENT for b in it["B"]]
    return (np.mean([l == NLIResult.CONTRADICTION for l in con]),
            np.mean([l == NLIResult.ENTAILMENT for l in ent]),
            np.mean([l == NLIResult.CONTRADICTION for l in ent]),
            [(it["A"], b, l.value) for it in CONTRADICTED for b, l in
             zip(it["B"], con[CONTRADICTED.index(it) * 3: CONTRADICTED.index(it) * 3 + 3])
             if l != NLIResult.CONTRADICTION])


def main():
    from lm_hook import GPT2Activations
    from sae_loader import load_residual_key_encoder

    t0 = time.time()
    torch.manual_seed(0)
    lm = GPT2Activations(layer=8)
    enc = load_residual_key_encoder()
    nli = HFNLIBackend()
    codes = encode_all(lm, enc)

    out, p = [], None
    p = out.append
    p("=" * W)
    p("Experiment 4.2: belief convergence under contradicting evidence".center(W))
    p("=" * W)
    p(f"  GPT-2 small layer 8, residual - b_dec keys, context retrieval, theta_rel {THETA_REL}")
    p(f"  NLI {nli.model_id.split('/')[-1]} (contradiction >= {nli.contradiction_threshold}, "
      f"entailment >= {nli.entailment_threshold})")
    p(f"  store: {len(CONTRADICTED)} contradicted beliefs A (INFERRED, w 0.8), "
      f"{len(CONSISTENT)} consistent (VERIFIED, w 0.8)")
    p(f"  {TURNS} turns, each: {len(CONTRADICTED)} B-evidence + {len(CONSISTENT)} agreeing + "
      f"{len(UNRELATED)} unrelated contexts, shuffled; 1 turn = 1 day")

    c_rate, e_rate, e_false, misses = nli_ceiling(nli)
    p("\nNLI ceiling (right belief, right evidence -- no retrieval involved)")
    p(f"  A vs its B evidence judged CONTRADICTION: {c_rate:.2f} of {3 * len(CONTRADICTED)}")
    p(f"  consistent belief vs agreeing evidence:   ENTAILMENT {e_rate:.2f}, "
      f"CONTRADICTION {e_false:.2f} of {3 * len(CONSISTENT)}")
    for a, b, l in misses:
        p(f"    missed: {a!r} vs {b!r} -> {l}")

    results, summaries = {}, {}
    for name, cfg in CONFIGS + POSTHOC:
        results[name] = run_config(cfg, codes, enc, nli)
        summaries[name] = summarise(results[name])

    s0 = summaries["paper"]
    p("\nRetrieval during evidence (same for every config with the same ranking)")
    p(f"  evidence retrieved its own belief: contradicted {s0['A_detect']:.2f}, "
      f"consistent {s0['C_detect']:.2f} (per context, over 3 turns)")
    p(f"  unrelated contexts that retrieved something: {s0['unrelated_flags']} flags "
      f"over {3 * len(UNRELATED)} contexts")

    p(f"\nA. Convergence (20 contradicted beliefs). switch = probe answers with the correction,")
    p(f"  over the {s0['n_reach']} topics whose probe question reaches the belief at all (t0)")
    p(f"  {'config':<20}{'switch t1/t2/t3':>17}{'w(A) t0->t3':>14}{'weff(A) t3':>11}"
      f"{'weff(B) t3':>11}{'succ/A':>8}{'A final states':>22}")
    for name, _ in CONFIGS + POSTHOC:
        if name == POSTHOC[0][0]:
            p("  -- post hoc " + "-" * 60)
        s = summaries[name]
        sw = "/".join(f"{x:.2f}" for x in s["switch_reach"][1:])
        st = " ".join(f"{k[:3]}{v}" for k, v in sorted(s["A_states"].items()))
        p(f"  {name:<20}{sw:>17}{s['A_weight'][0]:>7.2f}->{s['A_weight'][-1]:.2f}"
          f"{s['A_weff'][-1]:>11.3f}{s['succ_weff'][-1]:>11.3f}"
          f"{s['n_succ_mean']:>5.1f}/{s['n_succ_max']}{st:>22}")
    p("  succ/A = successors per contradicted belief, mean/max (1 is ideal; the paper rule")
    p("  creates one per contradiction). States: ACT(ive) CON(tested) SUP(erseded).")

    p("\nB. Side effects. A config is only good if it converges WITHOUT these.")
    p(f"  {'config':<20}{'false contests':>15}{'consistent contested':>21}{'consistent kept':>16}"
      f"{'w(consistent) t3':>17}{'off-target flags':>17}")
    for name, _ in CONFIGS + POSTHOC:
        if name == POSTHOC[0][0]:
            p("  -- post hoc " + "-" * 60)
        s = summaries[name]
        p(f"  {name:<20}{s['false_contests']:>15}{s['C_contested']:>21}{s['C_keep']:>16}"
          f"{s['C_weight']:>17.3f}{s['off_target_flags']:>17}")
    p("  false contests: contests APPLIED to anything but a contradicted belief's own")
    p("  evidence. consistent kept: of the consistent topics the probe reached at t0, how many")
    p("  it still answers with the original belief at t3.")
    g = summaries["revised, both + gate"]
    p(f"  contest gate held back {g['gated']} contradictions, {g['gated_own']} of them true ones")

    p("\nC. Final probe answers for the 20 contradicted topics")
    p(f"  {'config':<20}{'correction':>11}{'original A':>11}{'nothing':>9}{'other topic':>12}")
    for name, _ in CONFIGS + POSTHOC:
        if name == POSTHOC[0][0]:
            p("  -- post hoc " + "-" * 60)
        c = summaries[name]["probe_final"]
        p(f"  {name:<20}{c.get('succ', 0):>11}{c.get('orig', 0):>11}{c.get('none', 0):>9}"
          f"{c.get('other', 0):>12}")

    p("\nD. Weight trajectories, mean over the 20 contradicted beliefs (t0, t1, t2, t3)")
    for name in ("paper", "paper + trust", "revised", "revised - RIF", "revised, both + gate"):
        s = summaries[name]
        p(f"  {name:<20} w(A)    " + "  ".join(f"{x:.3f}" for x in s["A_weight"]))
        p(f"  {'':<20} weff(A) " + "  ".join(f"{x:.3f}" for x in s["A_weff"]))
        p(f"  {'':<20} weff(B) " + "  ".join(f"{x:.3f}" for x in s["succ_weff"]))

    p("\nE. Threshold sensitivity (theta_rel also sets the RIF floor where used)")
    gated = dict(POSTHOC)["revised, both + gate"]
    sweep = [("revised", REVISED, th, None) for th in (0.55, THETA_REL, 0.66)]
    sweep += [("both + gate", gated, th, THETA_CONTEST) for th in (0.55, THETA_REL, 0.66)]
    sweep += [("both + gate", gated, THETA_REL, g) for g in (0.65, 0.75, 0.80)]
    p(f"  {'config':<16}{'contest':>8}" + f"{'theta_rel':>10}{'detect A':>10}{'switch t3':>10}"
      f"{'false contests':>15}{'unrelated flags':>16}{'off-target':>11}")
    for cname, cfg, th, g in sweep:
        c = dict(cfg, rif_min_sim=th) if cfg["rif_min_sim"] else dict(cfg)
        c["contest_min_sim"] = g
        s = summarise(run_config(c, codes, enc, nli, theta=th))
        p(f"  {cname:<16}{('-' if g is None else f'{g:.2f}'):>8}{th:>10.3f}{s['A_detect']:>10.2f}{s['switch_reach'][-1]:>10.2f}{s['false_contests']:>15}"
          f"{s['unrelated_flags']:>16}{s['off_target_flags']:>11}")

    p(f"\n  {time.time() - t0:.0f}s")
    text = "\n".join(out)
    print(text)
    with open("results/exp_convergence.txt", "w") as f:
        f.write(text + "\n")
    js = {name: {k: (dict(v) if isinstance(v, Counter) else v) for k, v in s.items()}
          for name, s in summaries.items()}
    with open("results/exp_convergence.json", "w") as f:
        json.dump({"theta_rel": THETA_REL, "configs": dict(CONFIGS + POSTHOC), "summaries": js,
                   "trajectories": {n: r["traj"] for n, r in results.items()}},
                  f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


if __name__ == "__main__":
    main()
