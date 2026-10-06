"""
Smoke test for the real NLI backend. DOWNLOADS A MODEL on first run
(~370MB for the default deberta-v3-base, cached in ~/.cache/huggingface).

    python smoke_test_nli_backend.py                  # default deberta-v3-base
    python smoke_test_nli_backend.py distilroberta    # ~80MB, faster to try

Three things are checked:
  1. The model agrees with hand-labelled NLI cases (i.e. pair encoding
     and label mapping are actually correct, not just running).
  2. Cases where MockNLI's word-overlap heuristic is WRONG -- the
     concrete argument for using a real model in the reconsolidation loop.
  3. An end-to-end run_reconsolidation_pass driven by the real backend.
"""

import sys
import time

import numpy as np

from belief_store import BeliefStore, SourceTier, ContentType, BeliefState
from nli_backend import HFNLIBackend, NLIResult, MODEL_REGISTRY
from reconsolidation import MockNLI, run_reconsolidation_pass


# (premise, hypothesis, expected) -- premise is the "stored belief",
# hypothesis the "new claim", matching the reconsolidation call order.
CASES = [
    ("The belief store keeps all entries on the CPU.",
     "The belief store runs entirely on the GPU.",
     NLIResult.CONTRADICTION),

    ("Attention weights are computed with a softmax over the logits.",
     "A softmax normalizes the attention weights.",
     NLIResult.ENTAILMENT),

    ("The encoder is frozen during training.",
     "The encoder weights are not updated by the optimizer.",
     NLIResult.ENTAILMENT),

    ("The model was trained on 8 A100 GPUs.",
     "The model was trained on a single laptop CPU.",
     NLIResult.CONTRADICTION),

    ("The belief store uses cosine similarity for retrieval.",
     "The project began in March.",
     NLIResult.NEUTRAL),

    ("Reconsolidation runs after generation completes.",
     "Reconsolidation runs before the forward pass begins.",
     NLIResult.CONTRADICTION),
]

# Cases chosen because MockNLI gets them wrong. Each is a real failure
# mode of word-overlap + negation matching, and each would corrupt the
# belief store: a false CONTRADICTION contests a correct belief, a missed
# one lets a stale belief keep full weight.
MOCK_FAILURES = [
    ("The model uses attention layers.",
     "The model uses no fewer than twelve attention layers.",
     NLIResult.ENTAILMENT,
     "'no fewer than' trips the negation heuristic -> Mock cries CONTRADICTION "
     "and would contest a belief the claim actually supports"),

    ("The belief store keeps all entries on the CPU.",
     "The belief store runs entirely on the GPU.",
     NLIResult.CONTRADICTION,
     "a real contradiction with no negation word in it -> Mock sees only "
     "moderate overlap and returns NEUTRAL, missing the conflict entirely"),
]


def check_labelled_cases(nli):
    print("\n" + "=" * 72)
    print("1. Hand-labelled NLI cases")
    print("=" * 72)
    wrong = 0
    for premise, hypothesis, expected in CASES:
        verdict = nli.classify(premise, hypothesis)
        ok = verdict.label == expected
        wrong += not ok
        print(f"  [{'ok ' if ok else 'MISS'}] expected {expected.value:<13} got {verdict}")
        print(f"         premise:    {premise}")
        print(f"         hypothesis: {hypothesis}")
        if not ok:
            print(f"         scores: "
                  f"{ {k.value: round(v, 3) for k, v in verdict.scores.items()} }")
    print(f"\n  {len(CASES) - wrong}/{len(CASES)} correct")
    return wrong


def compare_against_mock(nli):
    print("\n" + "=" * 72)
    print("2. Cases MockNLI gets wrong")
    print("=" * 72)
    mock = MockNLI()
    fixed = 0
    for premise, hypothesis, expected, why in MOCK_FAILURES:
        mock_label = mock(premise, hypothesis)
        real = nli.classify(premise, hypothesis)
        fixed += (real.label == expected and mock_label != expected)
        print(f"\n  premise:    {premise}")
        print(f"  hypothesis: {hypothesis}")
        print(f"  expected:   {expected.value}")
        print(f"  MockNLI:    {mock_label.value}   <-- {'WRONG' if mock_label != expected else 'ok'}")
        print(f"  real model: {real}   <-- {'ok' if real.label == expected else 'WRONG'}")
        print(f"  why it matters: {why}")
    print(f"\n  real backend corrected {fixed}/{len(MOCK_FAILURES)} of Mock's failures")
    return fixed


def end_to_end_pass(nli):
    print("\n" + "=" * 72)
    print("3. End-to-end reconsolidation pass with the real backend")
    print("=" * 72)
    rng = np.random.default_rng(0)
    store = BeliefStore()

    beliefs = [
        ("The belief store keeps all entries on the CPU.", SourceTier.VERIFIED),
        ("The encoder is frozen during training.", SourceTier.OBSERVED),
        ("The project uses cosine similarity for retrieval.", SourceTier.STATED),
    ]
    entries = []
    for text, tier in beliefs:
        e = store.create(rng.normal(size=16), content_kv=None, weight=0.8,
                         source_tier=tier, content_type=ContentType.IMPLEMENTATION,
                         t_now=0, content_text=text)
        entries.append(e)
        store._flagged_this_pass.append(e.id)

    claims = {
        entries[0].id: "The belief store runs entirely on the GPU.",       # contradiction
        entries[1].id: "The encoder weights are not updated by the optimizer.",  # entailment
        entries[2].id: "The project began in March.",                      # neutral
    }

    before = [(e.id, e.state, e.weight) for e in entries]
    t0 = time.perf_counter()
    summary = run_reconsolidation_pass(store, claims, t_now=5, nli_fn=nli)
    elapsed = time.perf_counter() - t0

    print(f"  pass completed in {elapsed * 1000:.0f}ms\n")
    for (eid, old_state, old_weight), e in zip(before, entries):
        verdict = summary["verdicts"].get(eid)
        print(f"  {e.content_text[:46]:<48}")
        print(f"      NLI: {verdict}")
        print(f"      state  {old_state.value} -> {e.state.value}")
        print(f"      weight {old_weight:.3f} -> {e.weight:.3f}  "
              f"(weff={e.effective_weight(t_now=5):.4f})")

    print(f"\n  contested:  {summary['contested']}")
    print(f"  reinforced: {summary['reinforced']}")
    print(f"  neutral:    {summary['neutral']}")
    print(f"  abstained:  {summary['abstained']}")

    problems = []
    if len(summary["contested"]) != 1:
        problems.append(f"expected 1 contested, got {len(summary['contested'])}")
    if entries[0].state != BeliefState.CONTESTED:
        problems.append(f"GPU/CPU conflict should have contested entry 0, "
                        f"state is {entries[0].state.value}")
    if entries[1].weight <= 0.8:
        problems.append(f"entailment should have reinforced entry 1, "
                        f"weight is {entries[1].weight}")

    # the contested entry must still retrieve at reduced weight, not vanish
    if entries[0].effective_weight(t_now=5) <= 0:
        problems.append("contested entry dropped to zero weff (Fix 1 regression)")

    return problems


def main():
    alias = sys.argv[1] if len(sys.argv) > 1 else "deberta-v3-base"
    entry = MODEL_REGISTRY.get(alias)
    if entry:
        print(f"Model: {alias} -> {entry['hf_id']} ({entry['params']})")
        print(f"  {entry['note']}")
    else:
        print(f"Model: {alias} (not in registry, treating as a raw HF id)")

    print("\nLoading (first run downloads weights, this can take a few minutes)...")
    t0 = time.perf_counter()
    nli = HFNLIBackend(alias)
    print(f"Loaded in {time.perf_counter() - t0:.1f}s")
    print(f"  {nli!r}")
    print(f"  label map: { {i: l.value for i, l in sorted(nli.label_map.items())} }")

    wrong = check_labelled_cases(nli)
    compare_against_mock(nli)
    problems = end_to_end_pass(nli)

    print("\n" + "=" * 72)
    print(f"stats: {nli.stats}")

    if problems:
        print("\nFAILURES:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)

    if wrong:
        # A miss on a hand-labelled case is worth seeing but isn't
        # necessarily a code fault -- NLI models genuinely disagree on
        # borderline pairs. The end-to-end assertions above are the hard
        # gate; this is a soft warning.
        print(f"\nPASSED with {wrong} hand-labelled case(s) missed "
              f"(soft warning -- check whether the pair is genuinely borderline)")
    else:
        print("\nAll smoke checks passed.")


if __name__ == "__main__":
    main()
