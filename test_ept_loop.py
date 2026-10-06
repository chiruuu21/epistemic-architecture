"""
Tests for the end-to-end loop and claim extraction.

Uses MockNLI/ScriptedNLI throughout -- no model download. The point here
is control flow and state handling, not NLI quality.
"""

import numpy as np
import torch

from belief_store import BeliefState, ContentType, SourceTier
from claim_extraction import (
    LexicalClaimExtractor, NLIRelevanceClaimExtractor, split_sentences,
)
from ept_loop import EpTLoop
from nli_backend import NLIResult, ScriptedNLI
from reconsolidation import MockNLI

rng = np.random.default_rng(0)
D = 32


# --------------------------------------------------------------------------
# Sentence splitting
# --------------------------------------------------------------------------

def test_sentence_split_basic():
    s = split_sentences("The store is on CPU. The gate runs on MPS! Is it fast? Yes.")
    assert len(s) == 4, s
    assert s[0] == "The store is on CPU."
    assert s[2] == "Is it fast?"
    print(f"[PASS] basic sentence split -> {len(s)} sentences")


def test_sentence_split_respects_abbreviations():
    """'e.g.' must not split a claim in half -- the fragment would then
    be matched and NLI'd as if it were a complete claim."""
    s = split_sentences("The store uses a floor, e.g. 0.02, to avoid sign flips. That is Fix 2.")
    assert len(s) == 2, f"expected 2 sentences, got {len(s)}: {s}"
    assert "0.02" in s[0] and "sign flips" in s[0]
    print(f"[PASS] abbreviation guard kept 'e.g. 0.02' inline -> {s[0]!r}")


def test_sentence_split_empty():
    assert split_sentences("") == []
    assert split_sentences("   ") == []
    print("[PASS] empty/whitespace input returns no sentences")


# --------------------------------------------------------------------------
# Lexical claim extraction
# --------------------------------------------------------------------------

def _entries(store, texts):
    out = []
    for t in texts:
        e = store.create(rng.normal(size=16), content_kv=None, weight=0.8,
                         source_tier=SourceTier.VERIFIED,
                         content_type=ContentType.IMPLEMENTATION,
                         t_now=0, content_text=t)
        out.append(e)
    return out


def test_lexical_matches_right_sentence():
    from belief_store import BeliefStore
    store = BeliefStore()
    entries = _entries(store, [
        "the belief store keeps entries on the cpu",
        "the encoder is frozen during training",
    ])
    output = ("The encoder is frozen during training and never updated. "
              "Separately, the belief store keeps entries on the CPU for lookups.")

    claims = LexicalClaimExtractor()(output, entries)

    assert "cpu" in claims[entries[0].id].lower()
    assert "encoder" in claims[entries[1].id].lower()
    print(f"[PASS] lexical extractor attached the topically matching sentence to each entry")


def test_lexical_attaches_nothing_below_threshold():
    """Unrelated output must produce no claim -- the entry then lands in
    skipped_no_claim rather than being compared against a wrong sentence."""
    from belief_store import BeliefStore
    store = BeliefStore()
    entries = _entries(store, ["the belief store keeps entries on the cpu"])
    claims = LexicalClaimExtractor(min_score=0.15)("Pineapples grow in tropical climates.", entries)
    assert claims == {}, f"expected no claim, got {claims}"
    print("[PASS] unrelated output attaches no claim instead of guessing")


def test_one_sentence_not_reused_across_entries():
    """A single vague sentence must not reinforce every belief at once."""
    from belief_store import BeliefStore
    store = BeliefStore()
    entries = _entries(store, [
        "the model uses attention layers",
        "the model uses attention layers heavily",
    ])
    claims = LexicalClaimExtractor()("The model uses attention layers.", entries)
    assert len(claims) == 1, f"one sentence should serve one entry, got {claims}"
    print("[PASS] sentence consumed by its best match, not shared across entries")


def test_entries_without_content_text_ignored():
    from belief_store import BeliefStore
    store = BeliefStore()
    e = store.create(rng.normal(size=16), content_kv=None, weight=0.8,
                     source_tier=SourceTier.VERIFIED,
                     content_type=ContentType.IMPLEMENTATION, t_now=0)
    claims = LexicalClaimExtractor()("Anything at all here.", [e])
    assert claims == {}
    print("[PASS] entries with no content_text are skipped by the extractor")


def test_nli_relevance_extractor_beats_lexical_on_paraphrase():
    """Lexical overlap misses paraphrase; NLI relevance is supposed to
    catch it. Scripted so the test needs no model."""
    from belief_store import BeliefStore
    from nli_backend import NLIVerdict

    store = BeliefStore()
    entries = _entries(store, ["the encoder is frozen during training"])
    output = "Gradient updates never reach those weights. Unrelated filler sentence here."

    assert LexicalClaimExtractor(min_score=0.15)(output, entries) == {}, \
        "precondition: lexical should fail on this paraphrase"

    class Relevant(ScriptedNLI):
        def classify_batch(self, pairs):
            out = []
            for premise, hypothesis in pairs:
                related = "Gradient updates" in hypothesis
                neutral = 0.1 if related else 0.9
                out.append(NLIVerdict(
                    label=NLIResult.ENTAILMENT if related else NLIResult.NEUTRAL,
                    confidence=1 - neutral,
                    scores={NLIResult.ENTAILMENT: 1 - neutral,
                            NLIResult.NEUTRAL: neutral,
                            NLIResult.CONTRADICTION: 0.0},
                    argmax_label=NLIResult.ENTAILMENT if related else NLIResult.NEUTRAL,
                ))
            return out

    claims = NLIRelevanceClaimExtractor(Relevant({}))(output, entries)
    assert claims.get(entries[0].id, "").startswith("Gradient updates"), claims
    print("[PASS] NLI-relevance extractor caught the paraphrase lexical overlap missed")


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------

def _loop(**kw):
    return EpTLoop(d_model=D, dict_size=256, sae_k=8, **kw)


def test_step_returns_all_signals():
    loop = _loop()
    loop.add_belief(rng.normal(size=256), "the store keeps entries on the cpu",
                    weight=0.9, source_tier=SourceTier.VERIFIED,
                    content_type=ContentType.IMPLEMENTATION, t_now=0)

    out = loop.step(torch.randn(2, 5, D), t_now=1)

    assert out["z_out"].shape == (2, 5, D)
    assert out["weff"].shape == (2, 5)
    assert out["PIK"].shape == (2, 5)
    assert out["trigger_mask"].shape == (2, 5)
    assert out["trigger_mask"].dtype == torch.bool
    print(f"[PASS] step() returns z_out/weff/PIK/trigger_mask with correct shapes; "
          f"{int(out['trigger_mask'].sum())}/10 tokens triggered reasoning")


def test_flagged_survives_belief_gate_drain():
    """THE bug this driver exists to prevent. BeliefGate.forward() drains
    the store's flagged buffer, so a naive wiring reconsolidates nothing.
    The loop must still have the entries afterwards."""
    loop = _loop()
    key = rng.normal(size=256)
    entry = loop.add_belief(key, "the store keeps entries on the cpu",
                            weight=0.9, source_tier=SourceTier.VERIFIED,
                            content_type=ContentType.IMPLEMENTATION, t_now=0)

    loop.step(torch.randn(1, 4, D), t_now=1)

    # the store's own buffer is empty -- BeliefGate consumed it
    assert loop.store.flagged_this_pass() == [], \
        "precondition: BeliefGate should have drained the store buffer"
    # but the loop kept them
    assert entry.id in {e.id for e in loop.pending_flagged}, \
        "loop lost the flagged entry -- reconsolidation would silently no-op"
    print(f"[PASS] store buffer drained by BeliefGate, but loop retained "
          f"{len(loop.pending_flagged)} flagged entry/entries")


def test_flags_accumulate_and_dedup_across_steps():
    loop = _loop()
    loop.add_belief(rng.normal(size=256), "the store keeps entries on the cpu",
                    weight=0.9, source_tier=SourceTier.VERIFIED,
                    content_type=ContentType.IMPLEMENTATION, t_now=0)

    for t in range(1, 6):
        loop.step(torch.randn(1, 4, D), t_now=t)

    pending = loop.pending_flagged
    ids = [e.id for e in pending]
    assert len(ids) == len(set(ids)), "an entry flagged on many passes must appear once"
    assert loop.stats["steps"] == 5
    print(f"[PASS] 5 steps, {loop.stats['flag_events']} flag events -> "
          f"{len(pending)} unique pending entr(ies)")


def test_end_to_end_contradiction_updates_store():
    """The full path: step -> accumulate -> extract -> NLI -> mutate."""
    loop = _loop()
    entry = loop.add_belief(rng.normal(size=256),
                            "the model uses attention layers",
                            weight=0.8, source_tier=SourceTier.VERIFIED,
                            content_type=ContentType.IMPLEMENTATION, t_now=0)
    loop.step(torch.randn(1, 4, D), t_now=1)
    assert loop.pending_flagged, "precondition: entry should be flagged"

    summary = loop.finish_generation(
        "The model does not use attention layers.", t_now=5
    )

    assert summary["n_flagged"] >= 1
    assert summary["n_claims_matched"] == 1, summary["claims"]
    assert entry.state == BeliefState.CONTESTED, f"state is {entry.state}"
    assert len(summary["contested"]) == 1
    assert entry.effective_weight(t_now=5) > 0, "contested must retrieve at a penalty, not vanish"
    print(f"[PASS] end-to-end: flagged -> claim {summary['claims'][entry.id]!r} -> "
          f"CONTRADICTION -> state {entry.state.value}, weff="
          f"{entry.effective_weight(t_now=5):.4f}")


def test_finish_generation_clears_pending():
    loop = _loop()
    loop.add_belief(rng.normal(size=256), "the model uses attention layers",
                    weight=0.8, source_tier=SourceTier.VERIFIED,
                    content_type=ContentType.IMPLEMENTATION, t_now=0)
    loop.step(torch.randn(1, 4, D), t_now=1)
    loop.finish_generation("The model uses attention layers.", t_now=5)

    assert loop.pending_flagged == [], "pending must reset between generations"

    second = loop.finish_generation("Anything.", t_now=6)
    assert second["n_flagged"] == 0
    assert second["contested"] == [] and second["reinforced"] == []
    print("[PASS] finish_generation clears pending; a second call is a clean no-op")


def test_explicit_claims_bypass_extractor():
    loop = _loop()
    entry = loop.add_belief(rng.normal(size=256), "the model uses attention layers",
                            weight=0.8, source_tier=SourceTier.VERIFIED,
                            content_type=ContentType.IMPLEMENTATION, t_now=0)
    loop.step(torch.randn(1, 4, D), t_now=1)

    summary = loop.finish_generation(
        "totally unrelated text about pineapples", t_now=5,
        claims={entry.id: "the model does not use attention layers"},
    )
    assert entry.state == BeliefState.CONTESTED
    print("[PASS] explicitly supplied claims bypass the heuristic extractor")


def test_reset_generation_discards_without_mutating():
    loop = _loop()
    entry = loop.add_belief(rng.normal(size=256), "the model uses attention layers",
                            weight=0.8, source_tier=SourceTier.VERIFIED,
                            content_type=ContentType.IMPLEMENTATION, t_now=0)
    loop.step(torch.randn(1, 4, D), t_now=1)
    loop.reset_generation()

    assert loop.pending_flagged == []
    assert entry.state == BeliefState.ACTIVE and entry.weight == 0.8
    print("[PASS] reset_generation drops flags without touching belief state")


def test_add_belief_requires_content_text():
    loop = _loop()
    try:
        loop.add_belief(rng.normal(size=256), "", weight=0.8,
                        source_tier=SourceTier.VERIFIED,
                        content_type=ContentType.IMPLEMENTATION, t_now=0)
    except ValueError as e:
        assert "content_text" in str(e)
        print("[PASS] add_belief rejects empty content_text up front "
              "instead of creating a permanently un-comparable entry")
        return
    raise AssertionError("expected ValueError for missing content_text")


def test_custom_ffn_is_applied():
    """The gates sandwich an FFN in a real block; make sure it's used."""
    called = []

    def ffn(z):
        called.append(z.shape)
        return z * 2.0

    loop = _loop(ffn=ffn)
    loop.add_belief(rng.normal(size=256), "a belief", weight=0.9,
                    source_tier=SourceTier.VERIFIED,
                    content_type=ContentType.IMPLEMENTATION, t_now=0)
    loop.step(torch.randn(1, 3, D), t_now=1)
    assert called, "ffn was never called -- gates are not sandwiching it"
    print(f"[PASS] custom ffn invoked between the gates on {called[0]}")


def test_empty_store_is_safe():
    """No beliefs yet: weff is 0 everywhere, every token triggers FOK,
    and reconsolidation is a clean no-op."""
    loop = _loop()
    out = loop.step(torch.randn(1, 4, D), t_now=1)
    assert torch.all(out["weff"] == 0.0)
    assert bool(out["trigger_mask"].all()), "zero confidence should trigger reasoning everywhere"
    summary = loop.finish_generation("Some output.", t_now=2)
    assert summary["n_flagged"] == 0
    print("[PASS] empty store: weff=0, all tokens trigger FOK, reconsolidation no-ops")


if __name__ == "__main__":
    test_sentence_split_basic()
    test_sentence_split_respects_abbreviations()
    test_sentence_split_empty()
    test_lexical_matches_right_sentence()
    test_lexical_attaches_nothing_below_threshold()
    test_one_sentence_not_reused_across_entries()
    test_entries_without_content_text_ignored()
    test_nli_relevance_extractor_beats_lexical_on_paraphrase()
    test_step_returns_all_signals()
    test_flagged_survives_belief_gate_drain()
    test_flags_accumulate_and_dedup_across_steps()
    test_end_to_end_contradiction_updates_store()
    test_finish_generation_clears_pending()
    test_explicit_claims_bypass_extractor()
    test_reset_generation_discards_without_mutating()
    test_add_belief_requires_content_text()
    test_custom_ffn_is_applied()
    test_empty_store_is_safe()
    print("\nAll ept_loop tests passed.")
