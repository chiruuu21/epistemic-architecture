
import torch
from belief_gate import BeliefGate
from belief_store import SourceTier, ContentType
from check_device import get_device


def main():
    device = get_device()
    print(f"Using device: {device}")

    d_model, dict_size = 64, 512
    gate = BeliefGate(d_model=d_model, dict_size=dict_size, sae_k=16, device=device)

    # seed the store with a couple of beliefs so query() has something to find
    dummy_hidden = torch.randn(d_model, device=device)
    concept = gate.encoder(dummy_hidden)
    gate.add_belief(concept, content_kv=None, weight=0.9,
                     source_tier=SourceTier.VERIFIED,
                     content_type=ContentType.IMPLEMENTATION, t_now=0)

    # fake attention output: (batch=2, seq=8, d_model)
    z_attn = torch.randn(2, 8, d_model, device=device)

    z_gated, weff, flagged = gate(z_attn, t_now=1)

    print(f"z_gated shape: {tuple(z_gated.shape)}, device: {z_gated.device}")
    print(f"weff shape: {tuple(weff.shape)}, range: [{weff.min():.4f}, {weff.max():.4f}]")
    print(f"flagged for reconsolidation: {len(flagged)} entries")

    assert z_gated.shape == z_attn.shape
    assert z_gated.device.type == device.type
    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()
