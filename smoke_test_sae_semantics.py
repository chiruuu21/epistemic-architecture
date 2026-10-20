"""
Does the pretrained SAE make belief lookup semantic? Model-dependent
(downloads GPT-2 small + one SAE layer, ~700 MB), so not in
run_all_tests.py.

    python smoke_test_sae_semantics.py

Three sections:

1. LOAD FIDELITY. The SAE's own reconstruction of real GPT-2 activations.
   A correctly loaded SAE explains most of the variance; a mis-mapped one
   (wrong layer, missing b_dec, transposed W_enc) does not. This is the
   check that the weights are what we think they are, independent of
   any retrieval result.

2. RETRIEVAL. Each belief sentence has a paraphrase with little word
   overlap. For every paraphrase, retrieve the nearest belief by cosine
   over sentence-level concept keys. Compared against:
     random SAE  -- the encoder the project used until now
     raw resid   -- mean residual stream, no SAE
     lexical     -- word-overlap (Jaccard), what MockNLI-style matching sees
   If the pretrained SAE does not beat random, Week 8-9 has not
   delivered meaning.

3. SEPARATION. Mean cosine between matched pairs vs mismatched pairs. Top-1
   accuracy says whether the right belief wins; the gap says by how much.
"""

from __future__ import annotations

import re
import sys
import time

import numpy as np
import torch

from belief_gate import TopKSAEEncoder
from lm_hook import GPT2Activations
from sae_loader import DEFAULT_HOOK, encoder_from_tensors, hook_layer, load_sae_tensors

# (belief, paraphrase). Paraphrases deliberately avoid the belief's
# content words so lexical overlap cannot carry the result.
PAIRS = [
    ("The belief store keeps all entries on the CPU.",
     "Every record in the memory lives in main processor RAM, not graphics hardware."),
    ("The encoder is frozen during training.",
     "Its weights stay fixed and the optimizer never updates them."),
    ("Paris is the capital of France.",
     "The French government is seated in the city on the Seine."),
    ("Water boils at one hundred degrees Celsius at sea level.",
     "At normal atmospheric pressure H2O turns to steam once it reaches 100C."),
    ("The Pacific is the largest ocean on Earth.",
     "No other sea on the planet covers as much area as this one."),
    ("Shakespeare wrote Hamlet.",
     "The tragedy about the Danish prince came from the Bard of Avon."),
    ("Dogs are loyal pets that many families keep.",
     "Lots of households own a faithful canine companion."),
    ("The stock market fell sharply on Monday.",
     "Share prices dropped hard at the start of the week."),
    ("She plays the violin in an orchestra.",
     "She is a string musician in a symphony ensemble."),
    ("The patient was given antibiotics for the infection.",
     "Doctors prescribed medicine to fight his bacterial illness."),
    ("Heavy rain caused flooding in the town.",
     "Storms left streets underwater after the downpour."),
    ("The team won the championship game last night.",
     "Yesterday evening the squad clinched the title."),
]

SEED = 0


def _words(s):
    return set(re.findall(r"[a-z0-9]+", s.lower())) - {"the", "a", "an", "of", "on", "in", "is", "at", "to", "and"}


def jaccard_matrix(queries, beliefs):
    return np.array([[len(_words(q) & _words(b)) / max(1, len(_words(q) | _words(b)))
                      for b in beliefs] for q in queries])


def cosine_matrix(Q, K):
    Q = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8)
    K = K / (np.linalg.norm(K, axis=1, keepdims=True) + 1e-8)
    return Q @ K.T


def score(sim):
    n = sim.shape[0]
    top1 = float((sim.argmax(axis=1) == np.arange(n)).mean())
    matched = float(np.diag(sim).mean())
    off = sim[~np.eye(n, dtype=bool)]
    mismatched = float(off.mean())
    # mean reciprocal rank: how far down the right belief sits when it misses
    ranks = [(sim[i] > sim[i, i]).sum() + 1 for i in range(n)]
    mrr = float(np.mean([1 / r for r in ranks]))
    return {"top1": top1, "mrr": mrr, "matched": matched, "mismatched": mismatched,
            "gap": matched - mismatched}


def main():
    device = "cpu"
    print("=" * 78)
    print("Loading GPT-2 small + pretrained SAE".center(78))
    print("=" * 78)
    t0 = time.perf_counter()
    cfg, tensors = load_sae_tensors(hook_point=DEFAULT_HOOK)
    lm = GPT2Activations(layer=hook_layer(DEFAULT_HOOK), device=device)
    print(f"  loaded in {time.perf_counter() - t0:.1f}s   SAE {cfg['d_in']}->{cfg['d_sae']} "
          f"@ {DEFAULT_HOOK}   model layer {lm.layer}")

    W_dec = tensors["W_dec"].float()
    native = encoder_from_tensors(cfg, tensors, DEFAULT_HOOK, k=None)
    topk = encoder_from_tensors(cfg, tensors, DEFAULT_HOOK, k=32)
    torch.manual_seed(SEED)
    random_enc = TopKSAEEncoder(cfg["d_in"], cfg["d_sae"], k=32)  # the pre-Week-8 encoder

    # ---------------------------------------------------------------- 1
    print("\n" + "=" * 78)
    print("1. Load fidelity: SAE reconstruction of real layer-8 activations")
    print("=" * 78)
    texts = [s for pair in PAIRS for s in pair]
    with torch.no_grad():
        ids = [lm.encode_ids(t) for t in texts]
        full = [lm.model(i, output_hidden_states=True).hidden_states[lm.layer][0] for i in ids]
    bos_norm = float(np.mean([h[0].norm() for h in full]))
    tok_norm = float(torch.cat([h[1:] for h in full]).norm(dim=-1).mean())
    X = torch.cat([h[1:] for h in full])  # (N_tokens, 768), BOS dropped

    def fvu(enc, W_dec_, b_dec_):
        with torch.no_grad():
            x_hat = enc(X) @ W_dec_ + b_dec_
        return float(((X - x_hat) ** 2).sum() / ((X - X.mean(0)) ** 2).sum())

    fvu_native = fvu(native, W_dec, tensors["b_dec"])
    fvu_no_bdec = fvu(encoder_from_tensors(cfg, {**tensors, "b_dec": torch.zeros_like(tensors["b_dec"])},
                                           DEFAULT_HOOK, k=None), W_dec, tensors["b_dec"])
    l0 = float((native(X) > 0).sum(-1).float().mean())
    print(f"  tokens: {X.shape[0]}   mean L0 (active features/token, native): {l0:.1f}")
    print(f"  BOS residual norm {bos_norm:.1f} vs other tokens {tok_norm:.1f} "
          f"({bos_norm / tok_norm:.0f}x) -> why position 0 is dropped")
    print(f"  fraction of variance unexplained (lower is better):")
    print(f"    pretrained SAE, correct mapping   {fvu_native:.3f}")
    print(f"    same weights, b_dec left at zero  {fvu_no_bdec:.3f}   <- mapping bug (1) in sae_loader")
    ok_fidelity = fvu_native < 0.3

    # ---------------------------------------------------------------- 2/3
    print("\n" + "=" * 78)
    print(f"2/3. Retrieval of {len(PAIRS)} beliefs from low-overlap paraphrases")
    print("=" * 78)
    beliefs = [b for b, _ in PAIRS]
    queries = [q for _, q in PAIRS]

    def keys(fn):
        return np.stack([fn(t) for t in beliefs]), np.stack([fn(t) for t in queries])

    variants = {
        "pretrained SAE, top-32": keys(lambda t: lm.concept_key(topk, t)),
        "pretrained SAE, native": keys(lambda t: lm.concept_key(native, t)),
        "random SAE (old)": keys(lambda t: lm.concept_key(random_enc, t)),
        "raw residual, no SAE": keys(lambda t: lm.resid(t).mean(0).numpy()),
    }
    results = {name: score(cosine_matrix(Q, K)) for name, (K, Q) in variants.items()}
    results["lexical (Jaccard)"] = score(jaccard_matrix(queries, beliefs))

    print(f"  {'method':<26}{'top-1':>8}{'MRR':>8}{'matched':>10}{'mismatch':>10}{'gap':>8}")
    for name, r in results.items():
        print(f"  {name:<26}{r['top1']:>8.2f}{r['mrr']:>8.2f}{r['matched']:>10.3f}"
              f"{r['mismatched']:>10.3f}{r['gap']:>8.3f}")
    print(f"  (chance top-1 = {1 / len(PAIRS):.2f})")

    K, Q = variants["pretrained SAE, top-32"]
    sim = cosine_matrix(Q, K)
    print("\n  per-pair, pretrained SAE top-32:")
    for i, (b, q) in enumerate(PAIRS):
        j = int(sim[i].argmax())
        mark = "ok " if j == i else "MISS"
        print(f"   [{mark}] {q[:52]:<52} -> {beliefs[j][:34]}")

    sae, rnd = results["pretrained SAE, top-32"], results["random SAE (old)"]
    beats_random = sae["top1"] > rnd["top1"] and sae["gap"] > rnd["gap"]

    print("\n" + "=" * 78)
    verdict = ok_fidelity and beats_random
    print(f"  load fidelity FVU<0.3: {ok_fidelity}   pretrained beats random on top-1 and gap: "
          f"{beats_random}")
    print(("PASSED" if verdict else "FAILED") + " -- see table above for the numbers")
    sys.exit(0 if verdict else 1)


if __name__ == "__main__":
    main()
