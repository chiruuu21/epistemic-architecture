"""
Offline checks for sae_loader: the mapping from a pretrained SAE's
tensors into TopKSAEEncoder, and the refusals. Synthetic tensors only --
no download, so this belongs in run_all_tests.py. The real weights are
exercised by smoke_test_sae_semantics.py.
"""

import numpy as np
import torch

from belief_gate import BeliefGate
from ept_loop import EpTLoop
from sae_loader import encoder_from_tensors, hook_layer, validate

HOOK = "blocks.8.hook_resid_pre"
D_IN, D_SAE = 16, 64

g = torch.Generator().manual_seed(0)


def _sae():
    cfg = {"hook_point": HOOK, "d_in": D_IN, "d_sae": D_SAE, "model_name": "toy"}
    tensors = {
        "W_enc": torch.randn(D_IN, D_SAE, generator=g),
        "b_enc": torch.randn(D_SAE, generator=g) * 0.1,
        "b_dec": torch.randn(D_IN, generator=g),
        "W_dec": torch.randn(D_SAE, D_IN, generator=g),
    }
    return cfg, tensors


def _raises(fn, needle):
    try:
        fn()
    except ValueError as e:
        assert needle in str(e), f"wrong error: {e}"
        return str(e)
    raise AssertionError(f"expected ValueError mentioning {needle!r}")


def test_hook_layer_parsing():
    assert hook_layer("blocks.8.hook_resid_pre") == 8
    assert hook_layer("blocks.11.hook_resid_pre") == 11
    _raises(lambda: hook_layer("blocks.11.hook_resid_post"), "unsupported")
    _raises(lambda: hook_layer("layer8"), "unsupported")
    print("[PASS] hook point -> layer index; resid_post and malformed names refused")


def test_native_relu_codes_reproduced_exactly():
    """k=None must equal the SAE's own encode: relu((x - b_dec) W_enc + b_enc)."""
    cfg, t = _sae()
    enc = encoder_from_tensors(cfg, t, HOOK, k=None)
    x = torch.randn(5, D_IN, generator=g)
    expected = torch.relu((x - t["b_dec"]) @ t["W_enc"] + t["b_enc"])
    got = enc(x)
    assert torch.allclose(got, expected, atol=1e-5)
    print(f"[PASS] k=None reproduces the SAE's native ReLU encode exactly "
          f"(max err {float((got - expected).abs().max()):.1e})")


def test_b_dec_is_subtracted_before_encoding():
    """Fail mode (1) in sae_loader: b_pre left at zero still gives sparse
    codes, just different ones."""
    cfg, t = _sae()
    enc = encoder_from_tensors(cfg, t, HOOK, k=None)
    assert torch.equal(enc.b_pre, t["b_dec"].float())
    x = torch.randn(5, D_IN, generator=g)
    wrong = torch.relu(x @ t["W_enc"] + t["b_enc"])
    assert not torch.allclose(enc(x), wrong, atol=1e-3)
    print("[PASS] b_dec mapped to b_pre (the no-bias encode gives different codes)")


def test_topk_keeps_strongest_k():
    cfg, t = _sae()
    native = encoder_from_tensors(cfg, t, HOOK, k=None)
    topk = encoder_from_tensors(cfg, t, HOOK, k=8)
    x = torch.randn(3, D_IN, generator=g)
    c_native, c_topk = native(x), topk(x)
    assert ((c_topk > 0).sum(-1) <= 8).all()
    # every kept value is the native value at that position
    kept = c_topk > 0
    assert torch.allclose(c_topk[kept], c_native[kept])
    print(f"[PASS] k=8 keeps <=8 features per token, values unchanged from native encode")


def test_validate_refuses_bad_layouts():
    cfg, t = _sae()
    validate(cfg, t, HOOK)  # sanity: the good one passes

    msgs = []
    msgs.append(_raises(lambda: validate(cfg, {k: v for k, v in t.items() if k != "b_dec"}, HOOK),
                        "missing"))
    msgs.append(_raises(lambda: validate(cfg, {**t, "W_enc": t["W_enc"].T.contiguous()}, HOOK),
                        "W_enc has shape"))
    msgs.append(_raises(lambda: validate(cfg, t, "blocks.6.hook_resid_pre"), "hook_point"))
    bad = t["b_enc"].clone()
    bad[3] = float("nan")
    msgs.append(_raises(lambda: validate(cfg, {**t, "b_enc": bad}, HOOK), "non-finite"))
    print(f"[PASS] validate refuses: missing key, transposed W_enc, wrong layer, NaN "
          f"({len(msgs)} cases)")


def test_gate_rejects_encoder_of_wrong_width():
    cfg, t = _sae()
    enc = encoder_from_tensors(cfg, t, HOOK)
    _raises(lambda: BeliefGate(d_model=D_IN * 2, dict_size=D_SAE, encoder=enc), "encoder expects")
    gate = BeliefGate(d_model=D_IN, dict_size=D_SAE, encoder=enc)
    assert gate.encoder is enc
    print("[PASS] BeliefGate accepts a matching encoder, refuses a mismatched width")


def test_loop_runs_with_loaded_encoder():
    cfg, t = _sae()
    enc = encoder_from_tensors(cfg, t, HOOK, k=8)
    loop = EpTLoop(d_model=D_IN, dict_size=D_SAE, encoder=enc)
    from belief_store import ContentType, SourceTier
    key = enc(torch.randn(1, D_IN, generator=g))[0].numpy()
    loop.add_belief(key, "a toy belief", weight=0.8, source_tier=SourceTier.VERIFIED,
                    content_type=ContentType.ARCHITECTURE, t_now=0)
    out = loop.step(torch.randn(1, 4, D_IN, generator=g), t_now=1)
    assert out["weff"].shape == (1, 4)
    assert np.isfinite(out["weff"].numpy()).all()
    print("[PASS] EpTLoop runs end to end with a loaded encoder threaded through")


if __name__ == "__main__":
    test_hook_layer_parsing()
    test_native_relu_codes_reproduced_exactly()
    test_b_dec_is_subtracted_before_encoding()
    test_topk_keeps_strongest_k()
    test_validate_refuses_bad_layouts()
    test_gate_rejects_encoder_of_wrong_width()
    test_loop_runs_with_loaded_encoder()
