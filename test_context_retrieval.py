"""
Offline checks for context-mode retrieval (BeliefGate retrieval="context")
and BeliefStore.query_batch(min_sim=...). Uses an identity encoder
(c = relu(z)) so expected codes, means and cosines can be written by hand.
"""

import numpy as np
import torch

from belief_gate import BeliefGate, TopKSAEEncoder
from belief_store import BeliefStore, ContentType, SourceTier
from ept_loop import EpTLoop

D = 8


def _identity_encoder():
    enc = TopKSAEEncoder(D, D, k=D)
    with torch.no_grad():
        enc.W_enc.copy_(torch.eye(D))
    return enc


def _e(i):
    v = np.zeros(D, dtype=np.float32)
    v[i] = 1.0
    return v


def _gate(**kw):
    kw.setdefault("theta_rel", 0.5)
    kw.setdefault("min_ctx", 1)
    gate = BeliefGate(d_model=D, dict_size=D, encoder=_identity_encoder(),
                      retrieval="context", **kw)
    return gate


def _add(store, key, w=0.8):
    return store.create(key, None, w, SourceTier.OBSERVED, ContentType.ARCHITECTURE, t_now=0)


def _raises(fn, needle):
    try:
        fn()
    except ValueError as e:
        assert needle in str(e), f"wrong error: {e}"
        return
    raise AssertionError(f"expected ValueError mentioning {needle!r}")


def test_min_sim_leaves_irrelevant_entries_untouched():
    store = BeliefStore()
    a = _add(store, _e(0))
    weff, flagged, sim = store.query_batch(np.stack([_e(1), _e(0)]), t_now=50,
                                           theta_recon=-1.0, min_sim=0.5, return_sim=True)
    assert weff[0] == 0.0 and weff[1] > 0, weff
    assert np.allclose(sim, [0.0, 1.0])
    assert a.access_count == 1 and a.last_accessed == 50  # only the relevant row touched it
    assert flagged == [a.id]
    # irrelevant-only query: no touch, no flag, no decay-clock reset
    store.flagged_this_pass()
    weff, flagged = store.query_batch(_e(1)[None], t_now=90, theta_recon=-1.0, min_sim=0.5)
    assert weff[0] == 0.0 and flagged == [] and a.last_accessed == 50
    print("[PASS] min_sim: irrelevant query gets weff 0, entry not touched or flagged; "
          "default 2-tuple return unchanged")


def test_running_mean_continues_across_calls():
    """prefill + decode steps must give the same queries as one pass."""
    torch.manual_seed(0)
    z = torch.rand(1, 6, D)
    one = _gate()
    _add(one.store, _e(0))
    w_one, _ = one(z, t_now=0)[1], None
    sim_one = one.last_sim.copy()

    split = _gate()
    _add(split.store, _e(0))
    parts = [split(z[:, :3], t_now=0)[1]]
    sims = [split.last_sim]
    for t in range(3, 6):
        parts.append(split(z[:, t:t + 1], t_now=0)[1])
        sims.append(split.last_sim)
    assert torch.allclose(torch.cat(parts, dim=1), w_one)
    assert np.allclose(np.concatenate(sims, axis=1), sim_one, atol=1e-6)
    print("[PASS] context mean: prefill(3) + 3 decode steps == one 6-token pass")


def test_query_is_running_mean_not_current_token():
    gate = _gate(theta_rel=0.6)
    _add(gate.store, _e(0))
    # token 1 is e0, tokens 2-3 are e1: running means e0, (e0+e1)/2, (e0+2e1)/3
    z = torch.from_numpy(np.stack([_e(0), _e(1), _e(1)]))[None]
    gate(z, t_now=0)
    expected = [1.0, 1 / np.sqrt(2), 1 / np.sqrt(5)]
    assert np.allclose(gate.last_sim[0], expected, atol=1e-6), gate.last_sim
    assert gate.last_relevant[0].tolist() == [True, True, False]
    print(f"[PASS] query at t is the running mean of codes 1..t "
          f"(cos {', '.join(f'{s:.3f}' for s in expected)})")


def test_irrelevant_passes_through_and_weff_is_zero():
    gate = _gate(theta_rel=0.5)
    _add(gate.store, _e(0), w=0.8)
    z = torch.from_numpy(np.stack([_e(0), _e(0), _e(5)]) * 3.0)[None]
    gate.reset_context()
    z_rel, w_rel, _ = gate(z[:, :2], t_now=0)
    assert torch.allclose(z_rel, z[:, :2] * w_rel[..., None])  # gated by weff
    assert (w_rel > 0).all()

    gate.reset_context()
    z_irr = torch.from_numpy(_e(5) * 3.0)[None, None].repeat(1, 2, 1)
    out, w_irr, flagged = gate(z_irr, t_now=0)
    assert torch.equal(out, z_irr), "irrelevant tokens must pass through unscaled"
    assert (w_irr == 0).all() and flagged == []
    print("[PASS] relevant tokens gated by weff; irrelevant pass through with weff 0, no flag")


def test_warmup_positions_never_flag():
    gate = _gate(theta_rel=0.5, min_ctx=3)
    a = _add(gate.store, _e(0))
    z = torch.from_numpy(np.stack([_e(0)] * 4))[None]
    gate(z[:, :2], t_now=7)   # both positions inside warm-up
    assert gate.last_relevant.sum() == 0
    assert a.access_count == 0 and gate.store.flagged_this_pass() == []
    gate(z[:, 2:], t_now=8)   # context now 3, 4 tokens
    assert gate.last_relevant[0].tolist() == [True, True]
    print("[PASS] warm-up (< min_ctx tokens) is irrelevant and never touches or flags an entry")


def test_reset_context_isolates_prompts():
    gate = _gate(theta_rel=0.9)
    _add(gate.store, _e(0))
    gate(torch.from_numpy(_e(1))[None, None], t_now=0)        # prompt A: e1
    gate(torch.from_numpy(_e(0))[None, None], t_now=0)        # still prompt A: mean(e1,e0)
    assert not gate.last_relevant.any()
    gate.reset_context()
    gate(torch.from_numpy(_e(0))[None, None], t_now=0)        # prompt B: e0 alone
    assert gate.last_relevant.all()
    print("[PASS] reset_context: a new prompt does not pool the previous prompt's codes")


def test_loop_fok_fires_when_nothing_relevant():
    loop = EpTLoop(d_model=D, dict_size=D, encoder=_identity_encoder(),
                   retrieval="context", theta_rel=0.5, min_ctx=1, theta_fok=0.5)
    loop.add_belief(_e(0), "a stored belief", weight=0.9, source_tier=SourceTier.OBSERVED,
                    content_type=ContentType.ARCHITECTURE, t_now=0)
    known = loop.step(torch.from_numpy(_e(0))[None, None].repeat(1, 3, 1), t_now=1)
    loop.reset_generation()
    unknown = loop.step(torch.from_numpy(_e(4))[None, None].repeat(1, 3, 1), t_now=2)
    assert not known["trigger_mask"].any() and unknown["trigger_mask"].all()
    assert known["relevant"].all() and not unknown["relevant"].any()
    print("[PASS] loop: FOK silent on a stored topic, fires on every token of an unstored one")


def test_bad_config_refused():
    _raises(lambda: _gate(theta_rel=0.0), "theta_rel must be > 0")
    _raises(lambda: BeliefGate(d_model=D, retrieval="sentence"), "retrieval must be")
    print("[PASS] theta_rel <= 0 and unknown retrieval modes refused")


if __name__ == "__main__":
    test_min_sim_leaves_irrelevant_entries_untouched()
    test_running_mean_continues_across_calls()
    test_query_is_running_mean_not_current_token()
    test_irrelevant_passes_through_and_weff_is_zero()
    test_warmup_positions_never_flag()
    test_reset_context_isolates_prompts()
    test_loop_fok_fires_when_nothing_relevant()
    test_bad_config_refused()
