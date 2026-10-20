"""
Retrieval and relevance on real GPT-2 hidden states, with confidence
intervals. Model-dependent analysis, not a test.

    python analysis_relevance.py

Data: eval_data.py -- 50 beliefs + 50 same-domain distractors stored
(100 entries, chance top-1 = 1%), 50 low-overlap paraphrase queries whose
belief IS stored, 50 unrelated prompts from domains with nothing stored.
Every headline number has a 95% bootstrap interval, resampling prompts
(not tokens: tokens of one prompt are not independent).

1. RETRIEVAL. Does the pretrained SAE retrieve the right belief better
   than the random encoder it replaced? Re-checks the 12-query result of
   smoke_test_sae_semantics.py at a size where one query is 2 points, not 8.

2. RELEVANCE. Does best-match similarity separate "a relevant belief is
   stored" from "nothing relevant is stored"? Why it matters: in the
   paper's per-token retrieval every token receives its nearest belief's
   full weff (~0.8) however far away that belief is, so FOK never fires.

3. THE LOOP. Token retrieval (paper spec) vs context retrieval
   (BeliefGate retrieval="context") across theta_rel.
"""

from __future__ import annotations

import numpy as np
import torch

from belief_gate import TopKSAEEncoder
from eval_data import DISTRACTORS, PAIRS, UNRELATED
from lm_hook import GPT2Activations
from sae_loader import DEFAULT_HOOK, hook_layer, load_pretrained_sae
from smoke_test_sae_semantics import jaccard_matrix

W = 78
B = 2000
rng = np.random.default_rng(0)


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def auroc(pos, neg):
    """P(random related score > random unrelated score), ties = 1/2."""
    pos, neg = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return float((pos > neg).mean() + 0.5 * (pos == neg).mean())


def ci(samples):
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return f"[{lo:.2f}, {hi:.2f}]"


def boot_auroc(pos, neg):
    """pos/neg: per-prompt arrays (or lists of per-prompt token arrays)."""
    ragged = isinstance(pos, list)
    flat = (lambda xs, idx: np.concatenate([xs[i] for i in idx])) if ragged else (lambda xs, idx: xs[idx])
    stats = [auroc(flat(pos, rng.integers(0, len(pos), len(pos))),
                   flat(neg, rng.integers(0, len(neg), len(neg)))) for _ in range(B)]
    point = auroc(flat(pos, range(len(pos))), flat(neg, range(len(neg))))
    return point, ci(stats)


def boot_mean(per_prompt):
    """per_prompt: list of token arrays (or 1-d values); mean over tokens."""
    arrs = [np.atleast_1d(a) for a in per_prompt]
    stats = [np.concatenate([arrs[i] for i in rng.integers(0, len(arrs), len(arrs))]).mean()
             for _ in range(B)]
    return float(np.concatenate(arrs).mean()), ci(stats)


def retrieval(lm, encoders, beliefs, queries):
    print("=" * W)
    print(f"1. Retrieval: {len(queries)} paraphrase queries vs {len(beliefs)} stored beliefs "
          f"(chance top-1 {1 / len(beliefs):.0%})")
    print("=" * W)
    n = len(queries)

    def key_fn(name):
        if name == "residual, no SAE":
            return lambda t: lm.resid(t).mean(0).numpy()
        return lambda t: lm.concept_key(encoders[name], t)

    sims = {}
    for name in [*encoders, "residual, no SAE"]:
        f = key_fn(name)
        K = unit(np.stack([f(b) for b in beliefs]))
        Q = unit(np.stack([f(q) for q in queries]))
        sims[name] = Q @ K.T
    sims["lexical (Jaccard)"] = jaccard_matrix(queries, beliefs)

    print(f"  {'method':<26}{'top-1':>7}{'95% CI':>15}{'MRR':>7}{'matched':>9}{'mismatch':>10}")
    hits = {}
    for name, sim in sims.items():
        hit = (sim.argmax(1) == np.arange(n)).astype(float)
        hits[name] = hit
        ranks = np.array([(np.delete(sim[i], i) >= sim[i, i]).sum() + 1 for i in range(n)])
        off = np.ones_like(sim, dtype=bool)
        off[np.arange(n), np.arange(n)] = False
        stats = [hit[rng.integers(0, n, n)].mean() for _ in range(B)]
        print(f"  {name:<26}{hit.mean():>7.2f}{ci(stats):>15}{np.mean(1 / ranks):>7.2f}"
              f"{np.diag(sim).mean():>9.3f}{sim[off].mean():>10.3f}")

    # paired comparisons: same queries, so bootstrap the difference directly
    for x, y in (("pretrained SAE, top-32", "random SAE (old)"),
                 ("residual, no SAE", "pretrained SAE, top-32")):
        a, b = hits[x], hits[y]
        diffs = [(a[i] - b[i]).mean() for i in (rng.integers(0, n, n) for _ in range(B))]
        print(f"  {x} - {y}: {np.mean(a - b):+.2f}  95% CI {ci(diffs)}  (paired)")
    return sims


def relevance(lm, encoder, beliefs, related):
    print("\n" + "=" * W)
    print(f"2. Relevance: {len(related)} related vs {len(UNRELATED)} unrelated prompts")
    print("=" * W)

    def codes(text):
        with torch.no_grad():
            return encoder(lm.resid(text)).numpy()

    K = unit(np.stack([codes(b).mean(0) for b in beliefs]))
    K_res = unit(np.stack([lm.resid(b).mean(0).numpy() for b in beliefs]))

    def signals(text):
        c = codes(text)
        tok = (unit(c) @ K.T).max(1)
        return {"token max": tok.max(), "token mean": tok.mean(),
                "pooled SAE": (unit(c.mean(0)) @ K.T).max(),
                "pooled resid": (unit(lm.resid(text).mean(0).numpy()) @ K_res.T).max()}, tok

    rel = [signals(q) for q in related]
    unr = [signals(q) for q in UNRELATED]
    print(f"  {'signal':<16}{'related':>10}{'unrelated':>11}{'AUROC':>8}{'95% CI':>15}")
    for name in rel[0][0]:
        p = np.array([s[name] for s, _ in rel])
        n = np.array([s[name] for s, _ in unr])
        point, interval = boot_auroc(p, n)
        print(f"  {name:<16}{p.mean():>10.3f}{n.mean():>11.3f}{point:>8.2f}{interval:>15}")

    tok_rel = np.concatenate([t for _, t in rel])
    print(f"  per-token best cos (related): "
          + "  ".join(f"p{q}={np.percentile(tok_rel, q):.2f}" for q in (50, 90, 99))
          + "   -> theta_recon=0.15 (cos 0.85) flags every token")


def compare_modes(lm, encoder, beliefs, related):
    """All beliefs VERIFIED at weight 0.9 (weff 0.81) so trust tiers do not
    confound PIK. Warm-up tokens (min_ctx) are excluded in both modes."""
    from belief_store import ContentType, SourceTier
    from ept_loop import EpTLoop

    MIN_CTX = 3

    def run(retrieval_mode, theta_rel):
        loop = EpTLoop(d_model=lm.d_model, dict_size=encoder.W_enc.shape[1], encoder=encoder,
                       retrieval=retrieval_mode, theta_rel=theta_rel, min_ctx=MIN_CTX,
                       theta_fok=0.5)
        keys = {b: lm.concept_key(encoder, b) for b in beliefs}
        entries = [loop.add_belief(keys[b], b, weight=0.9, source_tier=SourceTier.VERIFIED,
                                   content_type=ContentType.ARCHITECTURE, t_now=0)
                   for b in beliefs]
        res = {"rel": [], "unr": []}
        with lm.attached(loop):
            for kind, prompts in (("rel", related), ("unr", UNRELATED)):
                for i, p in enumerate(prompts):
                    step = lm.observe(p)
                    flagged = {e.id for e in loop.pending_flagged}
                    res[kind].append({
                        "trig": step["trigger_mask"][0].numpy()[MIN_CTX - 1:].astype(float),
                        "pik": step["PIK"][0].numpy()[MIN_CTX - 1:],
                        "n_flags": len(flagged),
                        "right": kind == "rel" and entries[i].id in flagged,
                    })
                    loop.reset_generation()
        return res

    print("\n" + "=" * W)
    print("3. Full loop: token vs context retrieval (FOK fire rate per token, 95% CI)")
    print("=" * W)
    print(f"  {'mode':<20}{'fires: related':>21}{'fires: unrelated':>21}{'PIK AUROC':>17}"
          f"{'right':>7}{'flags r/u':>11}")
    configs = [("token", 0.15)] + [("context", t) for t in (0.10, 0.15, 0.20, 0.25)]
    for mode, theta_rel in configs:
        r = run(mode, theta_rel)
        fr, fr_ci = boot_mean([x["trig"] for x in r["rel"]])
        fu, fu_ci = boot_mean([x["trig"] for x in r["unr"]])
        au, au_ci = boot_auroc([x["pik"] for x in r["rel"]], [x["pik"] for x in r["unr"]])
        right = sum(x["right"] for x in r["rel"])
        flags = (np.mean([x["n_flags"] for x in r["rel"]]), np.mean([x["n_flags"] for x in r["unr"]]))
        name = "token (paper)" if mode == "token" else f"context >={theta_rel:.2f}"
        print(f"  {name:<20}{fr:>7.0%} {fr_ci:<13}{fu:>7.0%} {fu_ci:<13}{au:>5.2f} {au_ci:<11}"
              f"{right:>3}/{len(related)}{flags[0]:>5.1f}/{flags[1]:.1f}")
    print("  want: fires rarely on related, often on unrelated; AUROC near 1; the right")
    print("        belief flagged; few flags on unrelated prompts. 'right' in token mode is")
    print("        inflated: it flags several beliefs per prompt.")


def main():
    encoder = load_pretrained_sae(k=32)
    lm = GPT2Activations(layer=hook_layer(DEFAULT_HOOK))
    beliefs = [b for b, _ in PAIRS] + DISTRACTORS
    related = [q for _, q in PAIRS]

    from sae_loader import encoder_from_tensors, load_sae_tensors
    cfg, tensors = load_sae_tensors()
    torch.manual_seed(0)
    encoders = {
        "pretrained SAE, top-32": encoder,
        "pretrained SAE, native": encoder_from_tensors(cfg, tensors, DEFAULT_HOOK, k=None),
        "random SAE (old)": TopKSAEEncoder(cfg["d_in"], cfg["d_sae"], k=32),
    }
    del tensors

    retrieval(lm, encoders, beliefs, related)
    relevance(lm, encoder, beliefs, related)
    compare_modes(lm, encoder, beliefs, related)


if __name__ == "__main__":
    main()
