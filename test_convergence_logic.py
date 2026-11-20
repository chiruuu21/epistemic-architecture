"""
Offline checks for the Report 4 plan items: trust-aware retrieval,
the reconsolidation revisions used by exp_convergence.py (evidence tier,
weakening, successor dedup, supersession, RIF by identity), and the
store-plus-model P(IK) wired into EpTLoop. No model downloads; NLI is
ScriptedNLI, encoders are identity maps so cosines can be written by hand.
"""

import numpy as np
import torch

from belief_gate import TopKSAEEncoder
from belief_store import (BeliefState, BeliefStore, ContentType, SourceTier,
                          RIF_FLOOR)
from calibration import CombinedPIK, PlattScaler, logit
from ept_loop import EpTLoop
from nli_backend import NLIResult, ScriptedNLI
from reconsolidation import run_reconsolidation_pass

D = 8
A_TXT, B_TXT, B2_TXT = "the tower is in Berlin", "the tower is in Paris", "Paris has the tower"


def _e(i, j=None, a=1.0):
    v = np.zeros(D, dtype=np.float32)
    v[i] = 1.0
    if j is not None:
        v[j] = a
    return v


def _add(store, key, w, tier=SourceTier.INFERRED, text=None):
    return store.create(key, None, w, tier, ContentType.ARCHITECTURE, t_now=0, content_text=text)


# -- trust-aware retrieval ------------------------------------------------

def test_cosine_tie_keeps_contested_original():
    """The failure Fix 3 exists for: a successor at the same address ties
    on cosine, and creation order hands retrieval to the contested belief."""
    s = BeliefStore()
    a = _add(s, _e(0), 0.8)
    a.state = BeliefState.CONTESTED
    b = _add(s, _e(0), 0.5, SourceTier.VERIFIED)
    assert b.effective_weight(0) > a.effective_weight(0)
    cos_pick = s.peek(_e(0), 0, rank="cosine")[0]
    trust_pick = s.peek(_e(0), 0, rank="trust")[0]
    assert cos_pick is a and trust_pick is b
    print(f"[PASS] same-address tie: cosine returns the CONTESTED original "
          f"(weff {a.effective_weight(0):.2f}), trust returns its successor "
          f"(weff {b.effective_weight(0):.2f})")


def test_trust_rank_respects_margin_and_relevance():
    s = BeliefStore()
    near = _add(s, _e(0), 0.3)                         # cos 1.0 to the query
    far = _add(s, _e(0, 1, 1.0), 1.0, SourceTier.OBSERVED)  # cos 0.707, strong
    assert s.peek(_e(0), 0, rank="trust", rank_margin=0.05)[0] is near
    assert s.peek(_e(0), 0, rank="trust", rank_margin=0.5)[0] is far
    # relevance is decided by the BEST cosine, not by the trusted pick
    w, _, best = s.query_batch(_e(0)[None], 0, min_sim=0.9, return_sim=True,
                               rank="trust", rank_margin=0.5)
    assert best[0] > 0.99 and w[0] == np.float32(far.effective_weight(0))
    print("[PASS] trust ranking only chooses among beliefs within rank_margin of the "
          "best cosine; relevance still uses the best cosine")


def test_peek_is_read_only():
    s = BeliefStore()
    a = _add(s, _e(0), 0.8)
    s.peek(_e(0), t_now=50, rank="trust")
    assert a.last_accessed == 0 and a.access_count == 0 and s.flagged_this_pass() == []
    print("[PASS] peek() leaves last_accessed, access_count and the flag buffer untouched")


# -- RIF by identity -------------------------------------------------------

def test_rif_by_id_reaches_same_address_competitor():
    s = BeliefStore()
    a = _add(s, _e(0), 0.8)
    b = _add(s, _e(0), 0.5)
    s.apply_rif(_e(0), eta_inhibit=0.1)          # legacy: cos>=0.999 means "self"
    assert a.weight == 0.8 and b.weight == 0.5
    s.apply_rif(_e(0), eta_inhibit=0.1, exclude_ids={b.id})
    assert abs(a.weight - 0.7) < 1e-6 and b.weight == 0.5
    print("[PASS] RIF excluding the retrieved entry by id suppresses a same-address "
          "competitor that the cosine rule silently exempted")


def test_rif_does_not_boost_anticorrelated():
    s = BeliefStore()
    anti = _add(s, -_e(0), 0.5)
    s.apply_rif(_e(0), eta_inhibit=0.1, exclude_ids=set())
    assert anti.weight == 0.5
    print("[PASS] RIF leaves anti-correlated keys alone (old rule raised their weight)")


# -- reconsolidation revisions --------------------------------------------

def _contest_setup():
    s = BeliefStore()
    a = _add(s, _e(0), 0.8, text=A_TXT)
    nli = ScriptedNLI({(A_TXT, B_TXT): NLIResult.CONTRADICTION,
                       (A_TXT, B2_TXT): NLIResult.CONTRADICTION,
                       (B_TXT, B2_TXT): NLIResult.ENTAILMENT,
                       (B_TXT, B_TXT): NLIResult.ENTAILMENT})
    return s, a, nli


def test_paper_rule_duplicates_and_never_supersedes():
    s, a, nli = _contest_setup()
    for t, claim in enumerate([B_TXT, B2_TXT, B_TXT], start=1):
        run_reconsolidation_pass(s, {a.id: claim}, t, nli, flagged=[a])
    succ = s.successors(a.id)
    assert len(succ) == 3 and all(x.source_tier == SourceTier.INFERRED for x in succ)
    assert a.state == BeliefState.CONTESTED and a.weight == 0.8
    print("[PASS] paper rule: 3 contradictions -> 3 INFERRED duplicate successors, "
          "weight(A) unchanged at 0.8, A never SUPERSEDED")


def test_revised_rule_tier_weaken_dedup_supersede():
    s, a, nli = _contest_setup()
    kw = dict(claim_tier=SourceTier.VERIFIED, eta_contest=0.2,
              dedup_successors=True, supersede=True)
    r1 = run_reconsolidation_pass(s, {a.id: B_TXT}, 1, nli, flagged=[a], **kw)
    (b,) = s.successors(a.id)
    assert b.source_tier == SourceTier.VERIFIED and abs(a.weight - 0.6) < 1e-6
    assert a.state == BeliefState.CONTESTED and not r1.get("superseded")
    # the same evidence again, retrieved through A: reinforce b, no duplicate
    r2 = run_reconsolidation_pass(s, {a.id: B2_TXT}, 2, nli, flagged=[a], **kw)
    assert len(s.successors(a.id)) == 1 and r2["deduplicated"] == [(a.id, b.id)]
    assert a.state == BeliefState.SUPERSEDED and r2["superseded"] == [(a.id, b.id)]
    assert a.effective_weight(2) == 0.0
    print(f"[PASS] revised rule: successor VERIFIED, A weakened 0.8->0.6, second "
          f"contradiction reinforces the existing successor (w {b.weight:.2f}) and "
          f"A becomes SUPERSEDED")


def test_entailment_promotes_tier_only_when_given():
    s = BeliefStore()
    e = _add(s, _e(0), 0.5, SourceTier.INFERRED, text=B_TXT)
    nli = ScriptedNLI({(B_TXT, B_TXT): NLIResult.ENTAILMENT})
    run_reconsolidation_pass(s, {e.id: B_TXT}, 1, nli, flagged=[e])
    assert e.source_tier == SourceTier.INFERRED
    run_reconsolidation_pass(s, {e.id: B_TXT}, 2, nli, flagged=[e],
                             claim_tier=SourceTier.VERIFIED)
    assert e.source_tier == SourceTier.VERIFIED
    run_reconsolidation_pass(s, {e.id: B_TXT}, 3, nli, flagged=[e],
                             claim_tier=SourceTier.INFERRED)
    assert e.source_tier == SourceTier.VERIFIED  # never demoted by weaker evidence
    print("[PASS] entailment by VERIFIED evidence promotes an INFERRED belief; weaker "
          "evidence never demotes")


# -- store-plus-model P(IK) in the loop ------------------------------------

def _identity_encoder():
    enc = TopKSAEEncoder(D, D, k=D)
    with torch.no_grad():
        enc.W_enc.copy_(torch.eye(D))
    return enc


def test_combined_pik_matches_per_path_platt():
    rng = np.random.default_rng(0)
    n = 400
    sim = rng.uniform(0.4, 1.0, n)
    weff = rng.uniform(0.3, 0.9, n)
    use = sim > 0.7
    s_ok = (rng.uniform(size=n) < sim).astype(float)
    lm_p = rng.uniform(0.01, 0.99, n)
    l_ok = (rng.uniform(size=n) < lm_p).astype(float)
    m = CombinedPIK().fit(sim, weff, use, s_ok, lm_p, l_ok)
    feats = np.stack([sim, np.log(weff)], 1)
    ref_store = PlattScaler().fit(feats[use], s_ok[use]).predict(feats)
    ref_lm = PlattScaler().fit(logit(lm_p), l_ok).predict(logit(lm_p))
    assert np.allclose(m.predict(sim, weff, use, lm_p), np.where(use, ref_store, ref_lm))
    m2 = CombinedPIK.from_dict(m.to_dict())
    assert np.allclose(m2.predict(sim, weff, use, lm_p), m.predict(sim, weff, use, lm_p))
    print("[PASS] CombinedPIK = store-path Platt(cos, log weff) | LM-path Platt(logit p); "
          "to_dict/from_dict round-trips")


def test_loop_answer_confidence_uses_prefill_decision():
    rng = np.random.default_rng(1)
    n = 200
    pik = CombinedPIK().fit(rng.uniform(0.5, 1, n), rng.uniform(0.3, 0.9, n),
                            rng.uniform(size=n) < 0.5, rng.uniform(size=n) < 0.7,
                            rng.uniform(0.01, 0.99, n), rng.uniform(size=n) < 0.4)
    loop = EpTLoop(d_model=D, dict_size=D, encoder=_identity_encoder(),
                   retrieval="context", theta_rel=0.5, min_ctx=1, pik_model=pik)
    e = loop.add_belief(_e(0), "belief zero", 0.9, SourceTier.VERIFIED,
                        ContentType.ARCHITECTURE, t_now=0)
    prompt = torch.tensor(np.stack([_e(0), _e(0, 1, 0.5)]))[None]
    loop.step(prompt, t_now=0)
    d = dict(loop.decision)
    loop.step(torch.tensor(_e(3))[None, None], t_now=0)  # a decode step drifts off-topic
    assert loop.decision["sim"] == d["sim"]
    got = loop.answer_confidence(0.3)
    want = pik.predict(d["sim"], [e.effective_weight(0)], [True], [0.3])
    assert d["use_store"][0] and np.allclose(got, want)
    loop.finish_generation("", t_now=0)
    loop.step(torch.tensor(_e(5))[None, None].repeat(1, 2, 1), t_now=0)  # unrelated prompt
    assert not loop.decision["use_store"][0]
    assert np.allclose(loop.answer_confidence(0.3), pik.lm_map.predict(logit([0.3])))
    print("[PASS] EpTLoop.answer_confidence scores the last PROMPT token (decode steps "
          "don't overwrite it): store map when relevant, LM map when FOK fired")


# -- post-hoc fixes from exp_convergence.py section B ----------------------

def test_lineage_rank_never_crosses_topics():
    s = BeliefStore()
    topic = _add(s, _e(0), 0.8)                            # INFERRED, weff 0.56
    other = _add(s, _e(0, 1, 0.2), 0.9, SourceTier.OBSERVED)  # cos 0.98, weff 0.9
    q = _e(0)
    assert s.peek(q, 0, rank="trust", rank_margin=0.05)[0] is other   # crosses topics
    assert s.peek(q, 0, rank="lineage")[0] is topic
    topic.state = BeliefState.CONTESTED
    succ = s.create(_e(0), None, 0.5, SourceTier.VERIFIED, ContentType.ARCHITECTURE,
                    t_now=0, supersedes=topic.id)
    assert s.peek(q, 0, rank="lineage")[0] is succ          # still prefers the correction
    print("[PASS] lineage ranking: a trusted neighbouring topic cannot take retrieval, "
          "but a belief's own correction still wins over it")


def test_rif_floor_spares_bystanders():
    s = BeliefStore()
    rival = _add(s, _e(0, 1, 0.2), 0.8)   # cos 0.98 to the query: a competitor
    bystander = _add(s, _e(0, 1, 1.0), 0.8)  # cos 0.71: merely nearby
    s.apply_rif(_e(0), eta_inhibit=0.1, exclude_ids=set(), min_sim=0.9)
    assert rival.weight < 0.8 and bystander.weight == 0.8
    print("[PASS] RIF with a competitor floor inhibits the rival (cos 0.98) and spares "
          "the bystander (cos 0.71)")


def test_contest_gate_needs_strong_match():
    s, a, nli = _contest_setup()
    r = run_reconsolidation_pass(s, {a.id: B_TXT}, 1, nli, flagged=[a],
                                 match_sims={a.id: 0.65}, contest_min_sim=0.7)
    assert a.state == BeliefState.ACTIVE and r["contest_gated"] == [a.id]
    assert not s.successors(a.id)
    r = run_reconsolidation_pass(s, {a.id: B_TXT}, 1, nli, flagged=[a],
                                 match_sims={a.id: 0.85}, contest_min_sim=0.7)
    assert a.state == BeliefState.CONTESTED and len(s.successors(a.id)) == 1
    print("[PASS] contest gate: CONTRADICTION at match cos 0.65 is held as NEUTRAL; "
          "at 0.85 it contests")


if __name__ == "__main__":
    test_cosine_tie_keeps_contested_original()
    test_trust_rank_respects_margin_and_relevance()
    test_peek_is_read_only()
    test_rif_by_id_reaches_same_address_competitor()
    test_rif_does_not_boost_anticorrelated()
    test_paper_rule_duplicates_and_never_supersedes()
    test_revised_rule_tier_weaken_dedup_supersede()
    test_entailment_promotes_tier_only_when_given()
    test_combined_pik_matches_per_path_platt()
    test_loop_answer_confidence_uses_prefill_decision()
    test_lineage_rank_never_crosses_topics()
    test_rif_floor_spares_bystanders()
    test_contest_gate_needs_strong_match()
