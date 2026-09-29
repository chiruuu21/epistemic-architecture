import torch
from fok_gate import FOKGate
from check_device import get_device


def test_basic_shapes_and_trigger(device):
    d_model = 64
    gate = FOKGate(d_model=d_model, theta_fok=0.5).to(device)

    zffn = torch.randn(2, 8, d_model, device=device)
    # half low confidence (should trigger), half high (should not)
    weff = torch.cat([
        torch.full((2, 4), 0.2, device=device),  # < 0.5 -> triggers
        torch.full((2, 4), 0.9, device=device),  # >= 0.5 -> passes through
    ], dim=1)

    zout, PIK, trigger_mask = gate(zffn, weff)

    assert zout.shape == zffn.shape
    assert trigger_mask.sum().item() == 8, f"expected 8 triggered tokens, got {trigger_mask.sum().item()}"
    # untouched tokens should be numerically identical to input
    untouched = ~trigger_mask
    assert torch.allclose(zout[untouched], zffn[untouched]), "non-triggered tokens should pass through unchanged"
    # triggered tokens should differ from raw zffn (mixing_head changed them)
    assert not torch.allclose(zout[trigger_mask], zffn[trigger_mask]), "triggered tokens should be modified"
    print(f"[PASS] basic shapes/trigger: {trigger_mask.sum().item()}/16 tokens triggered, "
          f"untouched tokens pass through exactly, triggered tokens modified")


def test_no_triggers_short_circuits(device):
    """All-confident batch: thought_fn should never even be called."""
    d_model = 32
    gate = FOKGate(d_model=d_model, theta_fok=0.1).to(device)

    call_count = {"n": 0}
    orig_forward = gate.thought_fn.forward
    def counting_forward(x):
        call_count["n"] += 1
        return orig_forward(x)
    gate.thought_fn.forward = counting_forward

    zffn = torch.randn(2, 8, d_model, device=device)
    weff = torch.full((2, 8), 0.99, device=device)  # all confident, nothing should trigger

    zout, PIK, trigger_mask = gate(zffn, weff)

    assert trigger_mask.sum().item() == 0
    assert call_count["n"] == 0, "thought_fn was called despite zero triggers -- compute saving broken"
    assert torch.equal(zout, zffn), "output should be exactly the input when nothing triggers"
    print("[PASS] zero triggers: thought_fn never called, output identical to input")


def test_all_triggered(device):
    d_model = 32
    gate = FOKGate(d_model=d_model, theta_fok=0.99).to(device)
    zffn = torch.randn(2, 8, d_model, device=device)
    weff = torch.full((2, 8), 0.1, device=device)  # everything triggers

    zout, PIK, trigger_mask = gate(zffn, weff)
    assert trigger_mask.all()
    assert zout.shape == zffn.shape
    print("[PASS] all triggered: shapes correct, no crash on full-batch trigger")


def test_gradients_flow(device):
    """Mixing head has learnable params -- must receive gradients for
    triggered tokens. This is the part most likely to break: index_copy
    + torch.where needs to preserve the autograd graph correctly."""
    d_model = 32
    gate = FOKGate(d_model=d_model, theta_fok=0.5).to(device)

    zffn = torch.randn(2, 8, d_model, device=device, requires_grad=True)
    weff = torch.cat([
        torch.full((2, 4), 0.1, device=device),
        torch.full((2, 4), 0.9, device=device),
    ], dim=1)

    zout, PIK, trigger_mask = gate(zffn, weff)
    loss = zout.sum()
    loss.backward()

    assert zffn.grad is not None, "no gradient reached zffn at all"
    # triggered rows should have nonzero grad (went through mixing_head)
    triggered_grad_norm = zffn.grad[trigger_mask].abs().sum().item()
    assert triggered_grad_norm > 0, "triggered tokens got zero gradient -- autograd graph likely broken"

    mixing_head_grad = gate.mixing_head[0].weight.grad
    assert mixing_head_grad is not None and mixing_head_grad.abs().sum().item() > 0, (
        "mixing_head received no gradient -- it wouldn't be trainable"
    )
    print(f"[PASS] gradients flow correctly: zffn.grad populated, "
          f"mixing_head.grad norm={mixing_head_grad.abs().sum().item():.4f}")


def main():
    device = get_device()
    print(f"Using device: {device}\n")
    test_basic_shapes_and_trigger(device)
    test_no_triggers_short_circuits(device)
    test_all_triggered(device)
    test_gradients_flow(device)
    print("\nAll FOK Gate tests passed.")


if __name__ == "__main__":
    main()
