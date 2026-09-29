import numpy as np
from belief_store import BeliefStore, SourceTier, ContentType, BeliefState
from reconsolidation import run_reconsolidation_pass, MockNLI, NLIResult

rng = np.random.default_rng(0)
nli = MockNLI()


def make_store_with_entry(content_text, weight=0.8, source_tier=SourceTier.VERIFIED):
    store = BeliefStore()
    key = rng.normal(size=16)
    entry = store.create(key, content_kv=None, weight=weight,
                          source_tier=source_tier, content_type=ContentType.IMPLEMENTATION,
                          t_now=0, content_text=content_text)
    return store, entry, key


def force_flag(store, entry):
    """Directly flag an entry, bypassing the query()-based flagging path
    (already tested elsewhere) so this test focuses only on what
    run_reconsolidation_pass does once something IS flagged."""
    store._flagged_this_pass.append(entry.id)


def test_contradiction_sets_contested_not_superseded():
    """This is the specific bug scenario: old create() would have
    silently overwritten CONTESTED -> SUPERSEDED here. Confirms the fix."""
    store, entry, key = make_store_with_entry("the model uses attention layers")
    force_flag(store, entry)
    claims = {entry.id: "the model does not use attention layers"}

    summary = run_reconsolidation_pass(store, claims, t_now=5, nli_fn=nli)

    assert entry.state == BeliefState.CONTESTED, (
        f"expected CONTESTED, got {entry.state} -- the create()-overwrite bug is back"
    )
    assert len(summary["contested"]) == 1
    old_id, new_id = summary["contested"][0]
    assert old_id == entry.id
    new_entry = store._entries[new_id]
    assert new_entry.supersedes == entry.id
    assert new_entry.source_tier == SourceTier.INFERRED
    assert new_entry.weight == 0.5
    assert new_entry.state == BeliefState.ACTIVE

    # the contested entry should still retrieve at reduced (not zero) weight
    weff = entry.effective_weight(t_now=5)
    assert weff > 0, "CONTESTED entry should retrieve at a penalty, not vanish"
    print(f"[PASS] contradiction -> old entry CONTESTED (weff={weff:.4f}, not 0), "
          f"new INFERRED entry created with supersedes link")


def test_entailment_reinforces_weight():
    store, entry, key = make_store_with_entry(
        "the model uses attention layers", weight=0.6, source_tier=SourceTier.STATED
    )
    force_flag(store, entry)
    claims = {entry.id: "the model uses attention layers"}  # near-identical -> high overlap

    summary = run_reconsolidation_pass(store, claims, t_now=5, nli_fn=nli, eta_reinforce=0.1)

    assert entry.weight > 0.6, f"expected reinforcement, weight is still {entry.weight}"
    assert abs(entry.weight - 0.7) < 1e-6
    assert entry.id in summary["reinforced"]
    assert entry.state == BeliefState.ACTIVE
    print(f"[PASS] entailment -> weight reinforced 0.6 -> {entry.weight:.4f}")


def test_reinforcement_caps_at_one():
    store, entry, key = make_store_with_entry(
        "the model uses attention layers", weight=0.97
    )
    force_flag(store, entry)
    claims = {entry.id: "the model uses attention layers"}

    run_reconsolidation_pass(store, claims, t_now=5, nli_fn=nli, eta_reinforce=0.1)
    assert entry.weight <= 1.0, f"weight exceeded cap: {entry.weight}"
    print(f"[PASS] reinforcement capped at 1.0 (got {entry.weight:.4f})")


def test_missing_claim_is_skipped_not_dropped_silently():
    store, entry, key = make_store_with_entry("some belief")
    force_flag(store, entry)
    claims = {}  # no claim supplied for the flagged entry

    summary = run_reconsolidation_pass(store, claims, t_now=5, nli_fn=nli)

    assert entry.id in summary["skipped_no_claim"]
    assert entry.state == BeliefState.ACTIVE  # untouched
    assert entry.weight == 0.8  # untouched
    print("[PASS] flagged entry with no supplied claim is skipped and reported, not silently ignored")


def test_missing_content_text_still_runs_rif():
    """Entry created without content_text (e.g. legacy entries, or ones
    only ever populated via KV tensor) can't be NLI-compared, but RIF
    should still fire since retrieval happened regardless."""
    store = BeliefStore()
    base_key = rng.normal(size=16)
    base_key /= np.linalg.norm(base_key)
    entry = store.create(base_key, content_kv=None, weight=0.8,
                          source_tier=SourceTier.VERIFIED,
                          content_type=ContentType.IMPLEMENTATION, t_now=0)
    # content_text intentionally omitted

    neighbor_key = 0.7 * base_key + 0.3 * rng.normal(size=16)
    neighbor = store.create(neighbor_key, content_kv=None, weight=0.5,
                             source_tier=SourceTier.INFERRED,
                             content_type=ContentType.STATUS, t_now=0)

    force_flag(store, entry)
    claims = {entry.id: "some claim"}  # claim supplied, but entry has no content_text to compare

    summary = run_reconsolidation_pass(store, claims, t_now=5, nli_fn=nli, eta_inhibit=0.2)

    assert entry.id in summary["skipped_no_text"]
    assert neighbor.weight < 0.5, "RIF should still have fired against the neighbor"
    print(f"[PASS] missing content_text -> skipped for NLI but RIF still applied "
          f"(neighbor weight 0.5 -> {neighbor.weight:.4f})")


def test_rif_fires_on_both_outcomes():
    """RIF should suppress competing beliefs whether the flagged entry
    was contested or reinforced -- the paper's loop runs it
    unconditionally, not gated on the NLI branch."""
    for claim, label in [("the model does not use attention layers", "contradiction"),
                          ("the model uses attention layers", "entailment")]:
        store, entry, base_key = make_store_with_entry(
            "the model uses attention layers", weight=0.8
        )
        base_key = base_key / np.linalg.norm(base_key)
        neighbor_key = 0.7 * base_key + 0.3 * rng.normal(size=16)
        neighbor = store.create(neighbor_key, content_kv=None, weight=0.5,
                                 source_tier=SourceTier.INFERRED,
                                 content_type=ContentType.STATUS, t_now=0)
        force_flag(store, entry)

        run_reconsolidation_pass(store, {entry.id: claim}, t_now=5, nli_fn=nli, eta_inhibit=0.2)
        assert neighbor.weight < 0.5, f"RIF didn't fire on {label} branch"
        print(f"[PASS] RIF fired on {label} branch (neighbor weight 0.5 -> {neighbor.weight:.4f})")


if __name__ == "__main__":
    test_contradiction_sets_contested_not_superseded()
    test_entailment_reinforces_weight()
    test_reinforcement_caps_at_one()
    test_missing_claim_is_skipped_not_dropped_silently()
    test_missing_content_text_still_runs_rif()
    test_rif_fires_on_both_outcomes()
    print("\nAll reconsolidation tests passed.")
