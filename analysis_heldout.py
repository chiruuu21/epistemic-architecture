"""
Held-out re-check of Report 3's best configuration on the 50+50 set:
context retrieval with centred residual keys (residual - b_dec) on GPT-2
layer 8. Report 3's caveat was that theta_rel = 0.608 was chosen on the
same prompts it was scored on. Runs on CPU or MPS in a few minutes:

    python analysis_heldout.py

1. HELD-OUT THRESHOLD. 200 random splits of the prompts into halves
   (25 related + 25 unrelated each). On the calibration half theta_rel is
   chosen by the Report 3 rule (75th percentile of unrelated prompts'
   pooled best-match cosine); FOK firing and PIK AUROC are then measured
   on the other half only. Reported: median and 2.5-97.5% range over
   splits, next to the in-sample numbers.

2. WHERE FOK FIRES ON RELATED PROMPTS. Fire rate by token position.
   Report 3 suspected the 46% comes from the start of each prompt, where
   the running mean has seen only a few tokens. Also reports the
   prompt-level decision (last token), which is what a downstream
   answer would actually use.

3. GRADED vs BINARY PIK. Token-level PIK AUROC (related vs unrelated)
   with the paper's binary weff and with confidence="graded".
"""

from __future__ import annotations

import numpy as np
import torch

from belief_store import ContentType, SourceTier
from calibration import auroc
from ept_loop import EpTLoop
from eval_data import DISTRACTORS, PAIRS, UNRELATED
from lm_hook import GPT2Activations
from sae_loader import DEFAULT_HOOK, hook_layer, load_residual_key_encoder

W = 78
MIN_CTX = 3
N_SPLITS = 200


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def run_loop(enc, keys, beliefs, states, theta, confidence="binary", tau=0.05):
    """Per prompt: PIK and trigger arrays over post-warm-up tokens."""
    loop = EpTLoop(d_model=enc.d_in, dict_size=enc.d_out, encoder=enc, retrieval="context",
                   theta_rel=theta, min_ctx=MIN_CTX, theta_fok=0.5,
                   confidence=confidence, tau=tau)
    for k, b in zip(keys, beliefs):
        loop.add_belief(k, b, weight=0.9, source_tier=SourceTier.VERIFIED,
                        content_type=ContentType.ARCHITECTURE, t_now=0)
    out = []
    for z in states:
        s = loop.step(z[None], t_now=1)
        loop.reset_generation()
        out.append((s["PIK"][0].numpy()[MIN_CTX - 1:], s["trigger_mask"][0].numpy()[MIN_CTX - 1:]))
    return out


def main():
    lm = GPT2Activations(layer=hook_layer(DEFAULT_HOOK))
    enc = load_residual_key_encoder()
    beliefs = [b for b, _ in PAIRS] + DISTRACTORS
    related = [q for _, q in PAIRS]
    prompts = related + UNRELATED
    is_rel = np.array([1] * len(related) + [0] * len(UNRELATED), dtype=bool)

    keys = [lm.concept_key(enc, b) for b in beliefs]
    states = [lm.resid(p) for p in prompts]
    K = unit(np.stack(keys))
    pooled = np.array([(unit(lm.concept_key(enc, p)) @ K.T).max() for p in prompts])

    # one loop run per candidate theta is cheap; cache by theta
    cache = {}

    def per_prompt(theta, confidence="binary", tau=0.05):
        key = (round(theta, 6), confidence, tau)
        if key not in cache:
            cache[key] = run_loop(enc, keys, beliefs, states, theta, confidence, tau)
        return cache[key]

    def metrics(res, idx):
        rel = [res[i] for i in idx if is_rel[i]]
        unr = [res[i] for i in idx if not is_rel[i]]
        fr = np.concatenate([t for _, t in rel]).mean()
        fu = np.concatenate([t for _, t in unr]).mean()
        au = auroc(np.concatenate([p for p, _ in rel]), np.concatenate([p for p, _ in unr]))
        last_r = np.mean([t[-1] for _, t in rel])
        last_u = np.mean([t[-1] for _, t in unr])
        return fr, fu, au, last_r, last_u

    print("=" * W)
    print("1. Held-out threshold (Report 3 rule: 75th pct of unrelated pooled cosine)")
    print("=" * W)
    theta_in = float(np.percentile(pooled[~is_rel], 75))
    m_in = metrics(per_prompt(theta_in), range(len(prompts)))
    rng = np.random.default_rng(0)
    rel_idx, unr_idx = np.flatnonzero(is_rel), np.flatnonzero(~is_rel)
    thetas, held = [], []
    for _ in range(N_SPLITS):
        r, u = rng.permutation(rel_idx), rng.permutation(unr_idx)
        cal_u = u[: len(u) // 2]
        test = np.concatenate([r[len(r) // 2:], u[len(u) // 2:]])
        th = float(np.percentile(pooled[cal_u], 75))
        thetas.append(th)
        held.append(metrics(per_prompt(th), test))
    held = np.array(held)
    names = ["FOK fires, related tokens", "FOK fires, unrelated tokens", "PIK AUROC (tokens)",
             "FOK fires at last token, related", "FOK fires at last token, unrelated"]
    print(f"  theta_rel: in-sample {theta_in:.3f}; held-out median {np.median(thetas):.3f} "
          f"[{np.percentile(thetas, 2.5):.3f}, {np.percentile(thetas, 97.5):.3f}]")
    print(f"  {'metric':<36}{'in-sample':>10}{'held-out median [2.5-97.5%]':>32}")
    for j, nm in enumerate(names):
        lo, md, hi = np.percentile(held[:, j], [2.5, 50, 97.5])
        print(f"  {nm:<36}{m_in[j]:>10.2f}{md:>12.2f} [{lo:.2f}, {hi:.2f}]")

    print("\n" + "=" * W)
    print(f"2. FOK fire rate by token position (theta_rel {theta_in:.3f}, related prompts)")
    print("=" * W)
    res = per_prompt(theta_in)
    rel_trig = [t for (_, t), r in zip(res, is_rel) if r]
    unr_trig = [t for (_, t), r in zip(res, is_rel) if not r]
    maxlen = max(map(len, rel_trig))
    print(f"  {'position (from first scored token)':<38}{'related':>9}{'unrelated':>11}{'n':>5}")
    for pos in range(min(maxlen, 12)):
        rr = [t[pos] for t in rel_trig if len(t) > pos]
        uu = [t[pos] for t in unr_trig if len(t) > pos]
        print(f"  {pos + MIN_CTX:<38}{np.mean(rr):>9.2f}{np.mean(uu) if uu else float('nan'):>11.2f}"
              f"{len(rr):>5}")
    first3 = np.concatenate([t[:3] for t in rel_trig]).mean()
    rest = np.concatenate([t[3:] for t in rel_trig if len(t) > 3]).mean()
    print(f"  related: first 3 scored tokens {first3:.2f} vs later tokens {rest:.2f}")

    print("\n" + "=" * W)
    print("3. Graded vs binary PIK (token-level AUROC related vs unrelated, held-out theta)")
    print("=" * W)
    for conf, taus in (("binary", [0.05]), ("graded", [0.02, 0.05, 0.1])):
        for tau in taus:
            res = per_prompt(theta_in, conf, tau)
            au = auroc(np.concatenate([p for (p, _), r in zip(res, is_rel) if r]),
                       np.concatenate([p for (p, _), r in zip(res, is_rel) if not r]))
            last = auroc(np.array([p[-1] for (p, _), r in zip(res, is_rel) if r]),
                         np.array([p[-1] for (p, _), r in zip(res, is_rel) if not r]))
            name = conf if conf == "binary" else f"graded, tau={tau}"
            print(f"  {name:<22} token AUROC {au:.3f}   last-token (prompt) AUROC {last:.3f}")
    print("  Pooled-cosine AUROC (upper bound for any monotone map of it): "
          f"{auroc(pooled[is_rel], pooled[~is_rel]):.3f}")


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
