"""
EpT Belief Gate (Section 3.3) — wires BeliefStore into attention output.

    c = TopK(Wenc(zattn - bpre) + benc)          # SAE concept extraction
    weff = BeliefStore.query(c)                  # epistemic weight lookup
    zgated = zattn (elementwise*) weff            # gate attention output
    err = ||zgated - BeliefStore.stored_value(c)||^2
    if err > theta_recon: flag entry for reconsolidation

Design note on devices:
    The BeliefStore (Python dict + numpy cosine search over concept keys)
    is inherently a CPU-side symbolic structure -- there's no meaningful
    GPU speedup for dict lookups and a k=5 nearest-neighbor search over a
    few thousand entries. Only the SAE encode/decode and the zgated
    elementwise multiply are GPU-bound tensor math. So this module keeps
    BeliefStore on CPU and moves only the tensor ops onto `device`
    (MPS on Apple Silicon, CUDA on NVIDIA, or CPU fallback) -- forcing
    the whole store onto GPU would add data-transfer overhead for no
    benefit.

"""

from __future__ import annotations

import torch
import torch.nn as nn

from belief_store import BeliefStore, SourceTier, ContentType


class TopKSAEEncoder(nn.Module):
    """Frozen SAE-style sparse encoder: c = TopK(Wenc(z - bpre) + benc).

    This is a structural placeholder -- in the real pipeline you'd load
    pretrained SAE weights (e.g. an Eleuther/GemmaScope-style SAE trained
    on the target model's residual stream), not train this from scratch.
    Frozen by construction (Section 2.2 / Fig. in paper: "SAE Feature
    Decoder -- frozen").
    """

    def __init__(self, d_model: int, dict_size: int, k: int = 32):
        super().__init__()
        self.k = k
        self.b_pre = nn.Parameter(torch.zeros(d_model), requires_grad=False)
        self.W_enc = nn.Parameter(torch.randn(d_model, dict_size) * 0.02, requires_grad=False)
        self.b_enc = nn.Parameter(torch.zeros(dict_size), requires_grad=False)

    @torch.no_grad()
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (..., d_model) -> c: (..., dict_size), top-k sparse."""
        pre_act = (z - self.b_pre) @ self.W_enc + self.b_enc
        topk_vals, topk_idx = torch.topk(pre_act, self.k, dim=-1)
        topk_vals = torch.relu(topk_vals)
        c = torch.zeros_like(pre_act)
        c.scatter_(-1, topk_idx, topk_vals)
        return c


class BeliefGate(nn.Module):
    """Gates attention output by source-weighted, decaying belief confidence.

    forward() returns (z_gated, weff, flagged) so the caller can inspect
    per-token effective weights and which entries got flagged for
    reconsolidation this pass, without the gate owning the update logic
    itself (that stays in BeliefStore.reconsolidate / apply_rif, called
    separately post-generation per Section 3.5).
    """

    def __init__(self, d_model: int, dict_size: int = 4096, sae_k: int = 32,
                 theta_recon: float = 0.15, device: str | torch.device = "cpu"):
        super().__init__()
        # Resolve to the concrete device (e.g. "mps:0", "cuda:0") by
        # querying a real tensor, rather than trusting the bare
        # torch.device(device) string -- see .type comparison note below.
        self.device = torch.zeros(1, device=device).device
        self.encoder = TopKSAEEncoder(d_model, dict_size, k=sae_k).to(self.device)
        self.store = BeliefStore()
        self.theta_recon = theta_recon

    @torch.no_grad()
    def forward(self, z_attn: torch.Tensor, t_now: int):
        """
        z_attn: (batch, seq, d_model), already on self.device
        Returns:
            z_gated: (batch, seq, d_model)
            weff:    (batch, seq) -- effective weight applied per token
            flagged: list[BeliefEntry] flagged for reconsolidation this pass
        """
        # NOTE: torch.device("mps") != tensor.device (which resolves to
        # "mps:0"), even though they're the same physical device -- so we
        # compare by .type, not equality, and normalize by moving instead
        # of asserting. Same issue would occur with "cuda" vs "cuda:0".
        if z_attn.device.type != self.device.type:
            raise ValueError(
                f"z_attn on {z_attn.device}, expected device type {self.device.type} "
                f"-- move it with .to(device) first"
            )
        z_attn = z_attn.to(self.device)

        c = self.encoder(z_attn)  # (B, S, dict_size), stays on self.device

        B, S, _ = z_attn.shape

        # Single batched store query instead of B*S individual ones (was
        # the throughput bottleneck -- see BeliefStore.query_batch
        # docstring). One off-device transfer for the whole block, one
        # vectorized numpy matmul, one on-device transfer back.
        c_flat = c.reshape(B * S, -1).detach().cpu().numpy()
        weff_np, _flagged_ids = self.store.query_batch(c_flat, t_now=t_now, theta_recon=self.theta_recon)
        weff = torch.from_numpy(weff_np).to(self.device).reshape(B, S)

        z_gated = z_attn * weff.unsqueeze(-1)  # broadcast scalar weight over d_model
        flagged = self.store.flagged_this_pass()
        return z_gated, weff, flagged

    # -- convenience for populating the store from outside -----------------

    def add_belief(self, concept_vec, content_kv, weight, source_tier: SourceTier,
                    content_type: ContentType, t_now: int):
        concept_np = concept_vec.detach().cpu().numpy() if torch.is_tensor(concept_vec) else concept_vec
        return self.store.create(concept_np, content_kv, weight, source_tier, content_type, t_now=t_now)