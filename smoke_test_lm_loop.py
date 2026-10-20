"""
EpTLoop on a real language model: GPT-2 small's layer-8 residual stream,
pretrained SAE concept keys, beliefs populated from real sentences.
Model-dependent, so not in run_all_tests.py.

    python smoke_test_lm_loop.py              # MockNLI for reconsolidation
    python smoke_test_lm_loop.py --real-nli   # DeBERTa-v3 (another ~370 MB in RAM)

Four sections:

1. OBSERVE MODE IS A NO-OP ON THE MODEL. The hook runs every forward pass
   through the loop but must not change what GPT-2 generates. Checked by
   generating with and without the hook attached.

2. RETRIEVAL ON REAL HIDDEN STATES. For each prompt, which stored belief
   do its tokens retrieve, and how similar is the best match? This is the
   per-token path BeliefGate actually uses (not the sentence-level keys
   of smoke_test_sae_semantics.py).

3. GATE MODE. z_out written back into the residual stream. Reports how
   far the generation moves -- the gates were never trained with GPT-2,
   so this measures the effect rather than expecting good text.

4. RECONSOLIDATION FROM A REAL GENERATION. Flags accumulated across a
   real prefill + decode, then finish_generation on the generated text
   plus a contradicting claim.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import torch

from belief_store import ContentType, SourceTier
from ept_loop import EpTLoop
from lm_hook import GPT2Activations
from sae_loader import DEFAULT_HOOK, hook_layer, load_pretrained_sae

BELIEFS = [
    ("Paris is the capital of France.", SourceTier.VERIFIED, ContentType.ARCHITECTURE, 0.9),
    ("The belief store keeps all entries on the CPU.", SourceTier.OBSERVED, ContentType.IMPLEMENTATION, 0.8),
    ("Water boils at one hundred degrees Celsius.", SourceTier.VERIFIED, ContentType.ARCHITECTURE, 0.9),
    ("The stock market fell sharply on Monday.", SourceTier.STATED, ContentType.STATUS, 0.6),
    ("Shakespeare wrote Hamlet.", SourceTier.INFERRED, ContentType.ARCHITECTURE, 0.7),
]

# prompt -> index of the belief it should retrieve
PROMPTS = [
    ("The French capital city is", 0),
    ("Where does the memory store its records? On the", 1),
    ("When you heat water on the stove it", 2),
    ("Investors lost money when shares", 3),
    ("The playwright who wrote the Danish tragedy was", 4),
]

W = 78


def make_loop(encoder, nli_fn=None):
    loop = EpTLoop(d_model=768, dict_size=encoder.W_enc.shape[1], encoder=encoder,
                   nli_fn=nli_fn, theta_fok=0.5)
    return loop


def populate(loop, lm, encoder):
    entries = []
    for text, tier, ctype, w in BELIEFS:
        key = lm.concept_key(encoder, text)
        entries.append(loop.add_belief(key, text, weight=w, source_tier=tier,
                                       content_type=ctype, t_now=0))
    return entries


def token_matches(lm, encoder, entries, text):
    """Per-token nearest belief + cosine, mirroring BeliefStore.query_batch."""
    with torch.no_grad():
        c = encoder(lm.resid(text)).numpy()
    K = np.stack([e.concept_key for e in entries])
    K = K / (np.linalg.norm(K, axis=1, keepdims=True) + 1e-8)
    Q = c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-8)
    sim = Q @ K.T
    return sim.argmax(1), sim.max(1)


def main():
    real_nli = "--real-nli" in sys.argv
    print("=" * W)
    print("EpTLoop on GPT-2 small, layer 8".center(W))
    print("=" * W)
    t0 = time.perf_counter()
    encoder = load_pretrained_sae(k=32)
    lm = GPT2Activations(layer=hook_layer(DEFAULT_HOOK))
    nli_fn = None
    if real_nli:
        from nli_backend import HFNLIBackend
        nli_fn = HFNLIBackend()
    print(f"  loaded in {time.perf_counter() - t0:.1f}s   encoder {encoder.source}   "
          f"NLI {'DeBERTa-v3' if real_nli else 'MockNLI'}")
    failures = []

    # ---------------------------------------------------------------- 1
    print("\n" + "=" * W)
    print("1. Observe mode leaves GPT-2's output unchanged")
    print("=" * W)
    loop = make_loop(encoder, nli_fn)
    entries = populate(loop, lm, encoder)
    prompt = PROMPTS[0][0]
    plain = lm.generate(prompt, max_new_tokens=12)
    t1 = time.perf_counter()
    with lm.attached(loop):
        observed = lm.generate(prompt, max_new_tokens=12)
    dt = time.perf_counter() - t1
    same = plain == observed
    print(f"  prompt:   {prompt!r}")
    print(f"  plain:    {plain!r}")
    print(f"  observed: {observed!r}")
    print(f"  identical: {same}   loop steps: {loop.stats['steps']} "
          f"(1 prefill + {loop.stats['steps'] - 1} decode)   tokens through loop: "
          f"{loop.stats['tokens']}   {dt * 1000:.0f} ms")
    if not same:
        failures.append("observe mode changed the generation")

    # ---------------------------------------------------------------- 2
    print("\n" + "=" * W)
    print("2. Per-token retrieval on real hidden states")
    print("=" * W)
    correct = 0
    for p, want in PROMPTS:
        idx, sim = token_matches(lm, encoder, entries, p)
        votes = np.bincount(idx, minlength=len(entries))
        winner = int(votes.argmax())
        ok = winner == want
        correct += ok
        print(f"  [{'ok ' if ok else 'MISS'}] {p[:44]:<44} -> {BELIEFS[winner][0][:30]:<30} "
              f"{votes[winner]}/{len(idx)} tok, best cos {sim.max():.2f}")
    print(f"  majority-vote retrieval: {correct}/{len(PROMPTS)} (chance {1 / len(BELIEFS):.2f})")

    # similarity scale vs theta_recon: what fraction of tokens would flag
    _, sims = zip(*[token_matches(lm, encoder, entries, p) for p, _ in PROMPTS])
    sims = np.concatenate(sims)
    print(f"  per-token best cosine: median {np.median(sims):.2f}, max {sims.max():.2f}; "
          f"theta_recon=0.15 flags any token below cos 0.85 -> "
          f"{(1 - sims > 0.15).mean():.0%} of tokens")

    # ---------------------------------------------------------------- 3
    print("\n" + "=" * W)
    print("3. Gate mode: z_out written back into the residual stream")
    print("=" * W)
    for p, _ in PROMPTS[:3]:
        plain = lm.generate(p, max_new_tokens=10)
        loop = make_loop(encoder, nli_fn)
        populate(loop, lm, encoder)
        with lm.attached(loop, gate=True):
            gated = lm.generate(p, max_new_tokens=10)
        weff = lm.last_step["weff"]
        frac = loop.stats["triggered"] / max(1, loop.stats["tokens"])
        print(f"  {p!r}")
        print(f"    plain: {plain!r}")
        print(f"    gated: {gated!r}   FOK triggered on {frac:.0%} of tokens, "
              f"last weff {float(weff.mean()):.3f}")

    # ---------------------------------------------------------------- 4
    print("\n" + "=" * W)
    print("4. Reconsolidation after a real generation")
    print("=" * W)
    loop = make_loop(encoder, nli_fn)
    entries = populate(loop, lm, encoder)
    prompt = "The belief store is a memory system. It keeps all of its entries on"
    with lm.attached(loop):
        generated = lm.generate(prompt, max_new_tokens=15)
        t_end = lm.t_now
    print(f"  generated: {generated!r}")
    print(f"  flagged across {loop.stats['steps']} passes: {len(loop.pending_flagged)} "
          f"unique entries from {loop.stats['flag_events']} flag events")

    # GPT-2's own continuation rarely states a clean claim, so append one
    # that contradicts belief 1. This is the case the loop exists for.
    output_text = prompt + generated + " In fact, the belief store runs entirely on the GPU."
    target = entries[1]
    before = (target.state.value, target.weight)
    summary = loop.finish_generation(output_text, t_now=t_end)
    after = (target.state.value, target.weight)
    print(f"  claim attached to target: {summary['claims'].get(target.id)!r}")
    print(f"  target belief {target.content_text!r}")
    print(f"    {before[0]} w={before[1]:.3f} -> {after[0]} w={after[1]:.3f}   "
          f"weff now {target.effective_weight(t_end):.4f}")
    print(f"  summary: contested={len(summary['contested'])} reinforced={len(summary['reinforced'])} "
          f"neutral={len(summary['neutral'])}")
    if target.id not in loop.store._entries:
        failures.append("target belief missing after reconsolidation")

    print("\n" + "=" * W)
    if failures:
        print("FAILED: " + "; ".join(failures))
        sys.exit(1)
    print("PASSED -- real GPT-2 hidden states drove the full loop")


if __name__ == "__main__":
    main()
