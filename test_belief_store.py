import numpy as np
from belief_store import (
    BeliefStore, SourceTier, ContentType, BeliefState, RIF_FLOOR, CONTESTED_PENALTY
)

rng = np.random.default_rng(0)


def test_contested_is_penalized_not_zeroed():
    store = BeliefStore()
    key = rng.normal(size=16)
    e = store.create(key, content_kv=None, weight=0.8,
                      source_tier=SourceTier.VERIFIED,
                      content_type=ContentType.IMPLEMENTATION, t_now=0)

    weff_active = e.effective_weight(t_now=0)
    e.state = BeliefState.CONTESTED
    weff_contested = e.effective_weight(t_now=0)

    assert weff_active > 0, "sanity check: active weight should be > 0"
    assert weff_contested > 0, "BUG (old spec): CONTESTED collapsed to 0"
    assert abs(weff_contested - weff_active * CONTESTED_PENALTY) < 1e-6
    print(f"[PASS] CONTESTED weff={weff_contested:.4f} "
          f"(penalized from {weff_active:.4f}, not zeroed)")


def test_superseded_is_hard_excluded():
    store = BeliefStore()
    key = rng.normal(size=16)
    e = store.create(key, content_kv=None, weight=0.9,
                      source_tier=SourceTier.OBSERVED,
                      content_type=ContentType.STATUS, t_now=0)
    e.state = BeliefState.SUPERSEDED

    assert e.effective_weight(t_now=0) == 0.0
    print("[PASS] SUPERSEDED weff is hard 0.0, as intended")


def test_rif_never_goes_negative():
    store = BeliefStore()
    base_key = rng.normal(size=16)
    base_key /= np.linalg.norm(base_key)

    # a semantically related but distinct neighbor (cos ~0.7, well under
    # the 0.999 "is this literally the same entry" exclusion threshold)
    # that will get hammered by many RIF applications
    neighbor_key = 0.7 * base_key + 0.3 * rng.normal(size=16)
    neighbor = store.create(neighbor_key, content_kv=None, weight=0.5,
                             source_tier=SourceTier.INFERRED,
                             content_type=ContentType.STATUS, t_now=0)
    sim_check = store._cosine(base_key, neighbor_key)
    assert 0.3 < sim_check < 0.99, f"test setup issue: sim={sim_check:.3f} not in expected range"

    # simulate 50 retrievals of the base concept, each triggering RIF
    # against its neighbors -- enough applications that the OLD unbounded
    # formula would drive weight well below zero
    for _ in range(50):
        store.apply_rif(base_key, eta_inhibit=0.1, k=3)

    assert neighbor.weight >= RIF_FLOOR - 1e-9, (
        f"BUG (old spec): weight went to {neighbor.weight:.4f}, "
        f"should floor at {RIF_FLOOR}"
    )
    assert neighbor.weight <= 1.0
    print(f"[PASS] After 50 rounds of RIF, neighbor.weight={neighbor.weight:.4f} "
          f"(floored at {RIF_FLOOR}, never negative)")


def test_rif_floor_vs_unbounded_comparison():
    """Directly show what the paper's unbounded formula would have done."""
    weight = 0.5
    eta_inhibit, sim = 0.1, 0.95
    for _ in range(50):
        weight -= eta_inhibit * sim  # old, unbounded spec
    print(f"[INFO] Old unbounded spec after 50 rounds: weight={weight:.4f} "
          f"(negative -> would flip zgated sign in Belief Gate)")
    assert weight < 0, "expected the unfixed formula to go negative, confirming the bug existed"


def test_batch_matches_isolated_query_semantics():
    """query_batch() must agree with query() when there's no cross-token
    interference -- i.e. each query hits a distinct entry, so mid-loop
    mutation of last_accessed can't contaminate a later query in the
    same pass. This isolates 'is the math the same' from the separate,
    real bug documented in test_batch_fixes_intra_pass_decay_bug below."""
    store_loop = BeliefStore()
    store_batch = BeliefStore()
    keys = [rng.normal(size=32) for _ in range(20)]
    for i, k in enumerate(keys):
        for s in (store_loop, store_batch):
            s.create(k, content_kv=None, weight=0.6 + 0.02 * i,
                      source_tier=SourceTier.STATED,
                      content_type=ContentType.ARCHITECTURE, t_now=0)

    # queries built to be near-duplicates of *distinct* stored keys, so
    # each one has a clearly separate nearest entry -- no shared hits.
    queries = np.stack([keys[i] + 0.001 * rng.normal(size=32) for i in range(10)])

    loop_weffs = np.array([store_loop.query(q, t_now=5) for q in queries])
    batch_weffs, _ = store_batch.query_batch(queries, t_now=5)

    assert np.allclose(loop_weffs, batch_weffs, atol=1e-5), (
        f"Mismatch: loop={loop_weffs}, batch={batch_weffs}"
    )
    print(f"[PASS] query_batch matches query() when no entry is hit twice in-pass "
          f"(max diff={np.max(np.abs(loop_weffs - batch_weffs)):.2e})")


def test_batch_fixes_intra_pass_decay_bug():
    """DOCUMENTED BUG in the original per-token query() design (found while
    validating the batched version, not introduced by it): if two tokens
    in the *same* forward pass retrieve the same belief entry, the first
    call mutates entry.last_accessed to t_now as a side effect -- so the
    second call computes decay as if zero time had passed, understating
    how stale the belief actually was at the start of this pass.

    query_batch() computes weff for all entries from a pre-pass snapshot
    and only applies bookkeeping updates afterward, so it doesn't have
    this problem. This test locks in the correct (batched) behavior and
    demonstrates the old loop's bug numerically."""
    store_loop = BeliefStore()
    store_batch = BeliefStore()
    key = rng.normal(size=32)
    for s in (store_loop, store_batch):
        s.create(key, content_kv=None, weight=0.8,
                  source_tier=SourceTier.STATED,
                  content_type=ContentType.STATUS,  # fast decay (14-day half-life) to make the bug visible
                  t_now=0)

    # two queries in the "same pass", both hitting the one entry
    same_entry_queries = np.stack([key, key])

    loop_weffs = np.array([store_loop.query(q, t_now=5) for q in same_entry_queries])
    batch_weffs, _ = store_batch.query_batch(same_entry_queries, t_now=5)

    # old loop: second call sees artificially fresh last_accessed -> higher weff than first
    assert loop_weffs[1] > loop_weffs[0], "expected the old loop's decay-reset bug to reproduce"
    # batched: both queries in the pass see the same pre-pass state -> identical weff
    assert abs(batch_weffs[0] - batch_weffs[1]) < 1e-6, "batched queries in one pass should agree"
    print(f"[PASS] Old loop bug reproduced: weff[0]={loop_weffs[0]:.4f} -> weff[1]={loop_weffs[1]:.4f} "
          f"(2nd query wrongly 'fresher'). Batched: weff[0]={batch_weffs[0]:.4f} == weff[1]={batch_weffs[1]:.4f}")


if __name__ == "__main__":
    test_contested_is_penalized_not_zeroed()
    test_superseded_is_hard_excluded()
    test_rif_never_goes_negative()
    test_rif_floor_vs_unbounded_comparison()
    test_batch_matches_isolated_query_semantics()
    test_batch_fixes_intra_pass_decay_bug()
    print("\nAll tests passed.")