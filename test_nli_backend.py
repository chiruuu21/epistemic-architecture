"""
Tests for nli_backend that run WITHOUT downloading a model.

Everything here exercises the parts that were silently wrong in the old
HFNLIClassifier -- label ordering and unthresholded argmax -- plus the
batching path added to run_reconsolidation_pass. The model-dependent
half lives in smoke_test_nli_backend.py.
"""

import numpy as np

from belief_store import BeliefStore, SourceTier, ContentType, BeliefState
from nli_backend import (
    NLIResult, NLIVerdict, ScriptedNLI,
    canonical_label, _build_label_map, make_verdict, resolve_model_id,
    MODEL_REGISTRY,
)
from reconsolidation import run_reconsolidation_pass

rng = np.random.default_rng(0)

# Real id2label configs, copied from the actual model cards. The whole
# point of the label-map code is that these disagree with each other.
REAL_CONFIGS = {
    "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli":
        {0: "entailment", 1: "neutral", 2: "contradiction"},
    "facebook/bart-large-mnli":
        {0: "contradiction", 1: "neutral", 2: "entailment"},
    "roberta-large-mnli":
        {0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"},
    "cross-encoder/nli-deberta-v3-base":
        {0: "contradiction", 1: "entailment", 2: "neutral"},
}


# --------------------------------------------------------------------------
# Label canonicalization -- bug 2 in the old implementation
# --------------------------------------------------------------------------

def test_label_map_differs_across_real_models():
    """The old code assumed one fixed label order. Confirm that assumption
    is false for checkpoints people actually use, i.e. that this code has
    a job to do."""
    maps = {name: _build_label_map(cfg) for name, cfg in REAL_CONFIGS.items()}

    deberta = maps["MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"]
    bart = maps["facebook/bart-large-mnli"]
    assert deberta[0] == NLIResult.ENTAILMENT
    assert bart[0] == NLIResult.CONTRADICTION
    assert deberta[0] != bart[0], "index 0 must not be assumed to mean one thing"

    for name, m in maps.items():
        assert set(m.values()) == set(NLIResult), f"{name} lost a label"
    print("[PASS] label maps resolved per-model; index 0 means ENTAILMENT for "
          "DeBERTa and CONTRADICTION for BART -- the old hardcoded map inverted one of them")


def test_case_and_separator_insensitive():
    assert canonical_label("ENTAILMENT") == NLIResult.ENTAILMENT
    assert canonical_label("  neutral  ") == NLIResult.NEUTRAL
    assert canonical_label("Contradiction") == NLIResult.CONTRADICTION
    print("[PASS] label parsing tolerates case/whitespace variation")


def test_generic_labels_raise_rather_than_guess():
    """LABEL_0/1/2 carries no order information. Guessing would invert
    belief updates, so this must fail loudly at load time."""
    try:
        _build_label_map({0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"})
    except ValueError as e:
        assert "label_map" in str(e), "error should tell the caller how to fix it"
        print("[PASS] generic LABEL_n config raises with a remediation hint instead of guessing")
        return
    raise AssertionError("expected ValueError for uninterpretable labels")


def test_two_way_model_rejected():
    """entailment/not_entailment models can't distinguish contradiction
    from neutral -- accepting one would contest every unrelated claim."""
    try:
        _build_label_map({0: "entailment", 1: "neutral"})
    except ValueError as e:
        assert "three" in str(e).lower() or "two-way" in str(e).lower()
        print("[PASS] two-way NLI model rejected at load time")
        return
    raise AssertionError("expected ValueError for a 2-label model")


def test_explicit_override_accepted():
    m = _build_label_map(
        {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"},
        override={0: NLIResult.CONTRADICTION, 1: NLIResult.NEUTRAL, 2: NLIResult.ENTAILMENT},
    )
    assert m[0] == NLIResult.CONTRADICTION and m[2] == NLIResult.ENTAILMENT
    print("[PASS] explicit label_map override bypasses name inference")


# --------------------------------------------------------------------------
# Thresholding / abstention -- bug 3 in the old implementation
# --------------------------------------------------------------------------

DEBERTA_MAP = _build_label_map(REAL_CONFIGS["MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"])


def test_low_confidence_contradiction_abstains():
    """The scenario the old bare-argmax would have gotten wrong: a nearly
    uniform softmax contesting a VERIFIED belief."""
    # [entailment, neutral, contradiction]
    v = make_verdict([0.33, 0.33, 0.34], DEBERTA_MAP,
                     entailment_threshold=0.5, contradiction_threshold=0.7)
    assert v.argmax_label == NLIResult.CONTRADICTION, "model did pick contradiction"
    assert v.label == NLIResult.NEUTRAL, "but 0.34 must not trigger the destructive branch"
    assert v.abstained is True
    print(f"[PASS] 0.34 contradiction abstains -> {v} (argmax preserved for logging)")


def test_confident_contradiction_passes():
    v = make_verdict([0.02, 0.08, 0.90], DEBERTA_MAP,
                     entailment_threshold=0.5, contradiction_threshold=0.7)
    assert v.label == NLIResult.CONTRADICTION and not v.abstained
    print(f"[PASS] 0.90 contradiction acted on -> {v}")


def test_thresholds_are_asymmetric():
    """0.60 clears entailment but not contradiction: the destructive
    branch is deliberately harder to trigger than the additive one."""
    ent = make_verdict([0.60, 0.25, 0.15], DEBERTA_MAP, 0.5, 0.7)
    con = make_verdict([0.15, 0.25, 0.60], DEBERTA_MAP, 0.5, 0.7)
    assert ent.label == NLIResult.ENTAILMENT and not ent.abstained
    assert con.label == NLIResult.NEUTRAL and con.abstained
    print("[PASS] same 0.60 confidence: entailment acted on, contradiction abstained")


def test_neutral_never_abstains_into_itself():
    v = make_verdict([0.30, 0.40, 0.30], DEBERTA_MAP, 0.5, 0.7)
    assert v.label == NLIResult.NEUTRAL and v.abstained is False
    print("[PASS] a genuine low-confidence NEUTRAL is not marked as an abstention")


def test_scores_use_canonical_labels_not_indices():
    v = make_verdict([0.7, 0.2, 0.1], DEBERTA_MAP, 0.5, 0.7)
    assert set(v.scores) == set(NLIResult)
    assert abs(v.scores[NLIResult.ENTAILMENT] - 0.7) < 1e-6
    assert abs(sum(v.scores.values()) - 1.0) < 1e-6
    print("[PASS] verdict.scores keyed by NLIResult and sums to 1")


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

def test_registry_aliases_and_passthrough():
    assert resolve_model_id("deberta-v3-base") == MODEL_REGISTRY["deberta-v3-base"]["hf_id"]
    assert resolve_model_id("some/custom-model") == "some/custom-model"
    for name, entry in MODEL_REGISTRY.items():
        assert "/" in entry["hf_id"], f"{name} hf_id should be org-qualified"
    print("[PASS] registry aliases resolve, raw HF ids pass through untouched")


# --------------------------------------------------------------------------
# Batched integration with run_reconsolidation_pass
# --------------------------------------------------------------------------

def _store_with(texts):
    store = BeliefStore()
    entries = []
    for t in texts:
        key = rng.normal(size=16)
        e = store.create(key, content_kv=None, weight=0.8,
                         source_tier=SourceTier.VERIFIED,
                         content_type=ContentType.IMPLEMENTATION,
                         t_now=0, content_text=t)
        entries.append(e)
        store._flagged_this_pass.append(e.id)
    return store, entries


def test_batched_backend_gets_one_call_for_all_entries():
    store, entries = _store_with(["belief one", "belief two", "belief three"])
    claims = {e.id: f"claim {i}" for i, e in enumerate(entries)}

    nli = ScriptedNLI({("belief one", "claim 0"): NLIResult.CONTRADICTION,
                       ("belief two", "claim 1"): NLIResult.ENTAILMENT})
    summary = run_reconsolidation_pass(store, claims, t_now=5, nli_fn=nli)

    assert len(nli.seen) == 3, f"expected all 3 pairs in one batch, saw {len(nli.seen)}"
    assert len(summary["contested"]) == 1
    assert len(summary["reinforced"]) == 1
    assert len(summary["neutral"]) == 1
    print(f"[PASS] 3 flagged entries -> one batched classify_batch call, "
          f"verdicts routed correctly {nli.seen}")


def test_premise_is_stored_belief_hypothesis_is_claim():
    """Direction matters: NLI is not symmetric (docstring note 3)."""
    store, entries = _store_with(["the model uses attention layers"])
    claims = {entries[0].id: "the model uses layers"}
    nli = ScriptedNLI({})
    run_reconsolidation_pass(store, claims, t_now=5, nli_fn=nli)

    premise, hypothesis = nli.seen[0]
    assert premise == "the model uses attention layers", "premise must be the stored belief"
    assert hypothesis == "the model uses layers", "hypothesis must be the new claim"
    print("[PASS] pair ordering is (stored_belief, new_claim), not reversed")


def test_abstention_recorded_separately_from_neutral():
    store, entries = _store_with(["belief a", "belief b"])
    claims = {e.id: f"claim {i}" for i, e in enumerate(entries)}

    class AbstainingNLI(ScriptedNLI):
        def classify_batch(self, pairs):
            out = []
            for i, (p, h) in enumerate(pairs):
                out.append(NLIVerdict(
                    label=NLIResult.NEUTRAL, confidence=0.4,
                    scores={l: 0.33 for l in NLIResult},
                    argmax_label=NLIResult.CONTRADICTION,
                    abstained=(i == 0),
                ))
            return out

    summary = run_reconsolidation_pass(store, claims, t_now=5, nli_fn=AbstainingNLI({}))

    assert len(summary["neutral"]) == 2, "both land in neutral (that's the acted-on label)"
    assert len(summary["abstained"]) == 1, "only one was an abstention"
    assert summary["abstained"][0] == entries[0].id
    assert summary["verdicts"][entries[0].id].argmax_label == NLIResult.CONTRADICTION
    print("[PASS] abstention tracked distinctly from true neutral, argmax kept in verdicts")


def test_plain_callable_backend_still_works():
    """MockNLI and any bare (premise, hypothesis) -> NLIResult lambda must
    keep working now that the pass prefers classify_batch."""
    store, entries = _store_with(["the model uses attention layers"])
    claims = {entries[0].id: "the model does not use attention layers"}

    calls = []

    def plain_nli(premise, hypothesis):
        calls.append((premise, hypothesis))
        return NLIResult.CONTRADICTION

    summary = run_reconsolidation_pass(store, claims, t_now=5, nli_fn=plain_nli)

    assert len(calls) == 1
    assert entries[0].state == BeliefState.CONTESTED
    assert len(summary["contested"]) == 1
    assert summary["verdicts"][entries[0].id].confidence == 1.0
    print("[PASS] plain callable backend falls back to the per-pair path")


def test_entries_without_claims_are_not_sent_to_the_model():
    """Don't pay a forward pass for entries the pass is going to skip."""
    store, entries = _store_with(["belief a", "belief b"])
    claims = {entries[1].id: "only b has a claim"}
    nli = ScriptedNLI({})

    summary = run_reconsolidation_pass(store, claims, t_now=5, nli_fn=nli)

    assert len(nli.seen) == 1, f"only the claimable entry should be classified, saw {nli.seen}"
    assert entries[0].id in summary["skipped_no_claim"]
    print("[PASS] unclaimable entries excluded from the batch, not classified then discarded")


if __name__ == "__main__":
    test_label_map_differs_across_real_models()
    test_case_and_separator_insensitive()
    test_generic_labels_raise_rather_than_guess()
    test_two_way_model_rejected()
    test_explicit_override_accepted()
    test_low_confidence_contradiction_abstains()
    test_confident_contradiction_passes()
    test_thresholds_are_asymmetric()
    test_neutral_never_abstains_into_itself()
    test_scores_use_canonical_labels_not_indices()
    test_registry_aliases_and_passthrough()
    test_batched_backend_gets_one_call_for_all_entries()
    test_premise_is_stored_belief_hypothesis_is_claim()
    test_abstention_recorded_separately_from_neutral()
    test_plain_callable_backend_still_works()
    test_entries_without_claims_are_not_sent_to_the_model()
    print("\nAll nli_backend tests passed (no model download required).")
