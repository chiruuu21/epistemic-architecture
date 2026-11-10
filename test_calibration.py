"""
Offline checks for the Week 10-12 additions: calibration metrics
(calibration.py), TriviaQA answer matching (triviaqa_data.py) and graded
confidence in context retrieval (BeliefGate confidence="graded").
"""

import numpy as np
import torch

from belief_gate import BeliefGate, TopKSAEEncoder
from belief_store import ContentType, SourceTier
from calibration import PlattScaler, auroc, ece, reliability, sigmoid, summary
from ept_loop import EpTLoop
from triviaqa_data import is_correct, is_correct_lenient, normalize, prompt

D = 8


def test_ece_by_hand():
    # bin 0.2: conf .2,.2 acc 0/2 -> gap .2 ; bin 0.9: conf .9 x3, acc 2/3 -> gap .2333
    conf = [0.2, 0.2, 0.9, 0.9, 0.9]
    y = [0, 0, 1, 1, 0]
    expect = (2 * 0.2 + 3 * abs(2 / 3 - 0.9)) / 5
    assert abs(ece(conf, y) - expect) < 1e-12, ece(conf, y)
    assert ece([1.0, 1.0], [1, 1]) == 0.0  # conf 1.0 lands in the top bin, not out of range
    rows = reliability(conf, y)
    assert [r[2] for r in rows] == [2, 3]
    # equal-mass bins put the same number of items in each bin
    rows_a = reliability(np.linspace(0, 1, 100), np.zeros(100), n_bins=10, adaptive=True)
    assert [r[2] for r in rows_a] == [10] * 10
    print("[PASS] ECE matches a hand computation; conf=1.0 binned; adaptive bins equal-mass")


def test_constant_baseline_has_low_ece_and_no_skill():
    rng = np.random.default_rng(0)
    y = (rng.random(4000) < 0.05).astype(float)
    s = summary(np.full(4000, y.mean()), y)
    assert s["ece"] < 1e-9 and abs(s["auroc"] - 0.5) < 1e-9
    print("[PASS] constant base-rate confidence: ECE 0 with AUROC 0.5 (why ECE alone is not enough)")


def test_auroc_matches_brute_force_with_ties():
    rng = np.random.default_rng(1)
    p = rng.integers(0, 5, 40).astype(float)
    n = rng.integers(0, 5, 30).astype(float)
    brute = ((p[:, None] > n[None]).mean() + 0.5 * (p[:, None] == n[None]).mean())
    assert abs(auroc(p, n) - brute) < 1e-12
    assert np.isnan(auroc([], [1.0]))
    print("[PASS] rank AUROC equals the pairwise definition, ties counted 1/2")


def test_platt_recovers_known_map():
    rng = np.random.default_rng(2)
    x = rng.normal(size=20000)
    y = (rng.random(x.size) < sigmoid(2.0 * x - 1.0)).astype(float)
    w = PlattScaler(l2=0).fit(x, y).coef_
    assert abs(w[0] - 2.0) < 0.1 and abs(w[1] + 1.0) < 0.1, w
    # perfectly separable data stays finite thanks to the ridge
    w2 = PlattScaler().fit(np.array([-1.0, -0.5, 0.5, 1.0]), np.array([0, 0, 1, 1.0])).coef_
    assert np.all(np.isfinite(w2))
    # all-correct calibration data: target smoothing keeps p < 1
    p = PlattScaler().fit(np.zeros(9), np.ones(9)).predict(np.zeros(1))[0]
    assert abs(p - 10 / 11) < 1e-3, p
    print("[PASS] Platt scaling recovers a known map; separable/all-correct data stay finite")


def test_answer_matching():
    assert normalize("The  Beatles!") == "beatles"
    aliases = ["The Beatles", "Beatles"]
    assert is_correct("beatles", aliases) and is_correct("The Beatles.", aliases)
    assert not is_correct("", aliases) and not is_correct("The Rolling Stones", aliases)
    assert is_correct_lenient("It was the Beatles, of course", aliases)
    assert not is_correct_lenient("Beatlesque", aliases)  # word boundary, not substring
    assert prompt("Who?").endswith("Q: Who?\nA:")
    print("[PASS] TriviaQA normalisation and exact/lenient matching")


def _identity_encoder():
    enc = TopKSAEEncoder(D, D, k=D)
    with torch.no_grad():
        enc.W_enc.copy_(torch.eye(D))
    return enc


def _unit(*idx_w):
    v = np.zeros(D, dtype=np.float32)
    for i, w in idx_w:
        v[i] = w
    return v


def _graded_gate(**kw):
    g = BeliefGate(d_model=D, dict_size=D, encoder=_identity_encoder(), retrieval="context",
                   theta_rel=0.5, min_ctx=1, confidence="graded", tau=0.05, **kw)
    g.store.create(_unit((0, 1.0)), None, 0.9, SourceTier.OBSERVED, ContentType.ARCHITECTURE, t_now=0)
    return g


def test_graded_confidence_is_monotone_in_match():
    pik = []
    for c in (0.2, 0.45, 0.55, 0.9):        # cosine to the stored belief
        g = _graded_gate()
        z = torch.from_numpy(_unit((0, c), (1, float(np.sqrt(1 - c * c)))))[None, None]
        _, w, _ = g(z, t_now=0)
        pik.append(float(w[0, 0]))
        assert abs(g.last_sim[0, 0] - c) < 1e-5
    assert all(a < b for a, b in zip(pik, pik[1:])), pik
    assert pik[0] < 0.01 and pik[-1] > 0.89 and 0.1 < pik[1] < pik[2] < 0.8, pik
    print(f"[PASS] graded PIK rises smoothly with match cosine: "
          + ", ".join(f"{p:.3f}" for p in pik))


def test_graded_keeps_gating_and_warmup_rules():
    g = BeliefGate(d_model=D, dict_size=D, encoder=_identity_encoder(), retrieval="context",
                   theta_rel=0.5, min_ctx=3, confidence="graded")
    g.store.create(_unit((0, 1.0)), None, 0.9, SourceTier.OBSERVED, ContentType.ARCHITECTURE, t_now=0)
    z = torch.from_numpy(np.stack([_unit((0, 1.0))] * 4))[None]
    out, w, _ = g(z, t_now=0)
    assert (w[0, :2] == 0).all(), "warm-up positions must stay at 0"
    assert (w[0, 2:] > 0.8).all()
    # below theta_rel: graded weff may be > 0 but the token still passes through unscaled
    g.reset_context()
    zi = torch.from_numpy(_unit((0, 0.4), (3, float(np.sqrt(1 - 0.16)))))[None, None].repeat(1, 4, 1)
    out, w, flagged = g(zi, t_now=0)
    assert torch.equal(out, zi) and flagged == [] and (w[0, 2:] < 0.5).all()
    print("[PASS] graded mode: warm-up stays 0; irrelevant tokens pass through unflagged")


def test_binary_default_unchanged_and_config_refused():
    loop = EpTLoop(d_model=D, dict_size=D, encoder=_identity_encoder(), retrieval="context",
                   theta_rel=0.5, min_ctx=1)
    assert loop.belief_gate.confidence == "binary"
    loop.add_belief(_unit((0, 1.0)), "b", weight=0.9, source_tier=SourceTier.OBSERVED,
                    content_type=ContentType.ARCHITECTURE, t_now=0)
    z = torch.from_numpy(_unit((0, 0.8), (1, 0.6)))[None, None]
    out = loop.step(z, t_now=0)
    assert abs(float(out["PIK"][0, 0]) - 0.9) < 1e-6  # full weff, no grading
    for kw, needle in (({"confidence": "soft", "retrieval": "context"}, "confidence must be"),
                       ({"confidence": "graded"}, "needs retrieval='context'"),
                       ({"confidence": "graded", "retrieval": "context", "tau": 0}, "tau must be")):
        try:
            BeliefGate(d_model=D, **kw)
        except ValueError as e:
            assert needle in str(e), e
        else:
            raise AssertionError(kw)
    print("[PASS] binary confidence is still the default; bad graded configs refused")


if __name__ == "__main__":
    test_ece_by_hand()
    test_constant_baseline_has_low_ece_and_no_skill()
    test_auroc_matches_brute_force_with_ties()
    test_platt_recovers_known_map()
    test_answer_matching()
    test_graded_confidence_is_monotone_in_match()
    test_graded_keeps_gating_and_warmup_rules()
    test_binary_default_unchanged_and_config_refused()
