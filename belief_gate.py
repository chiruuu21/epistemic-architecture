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

import numpy as np
import torch
import torch.nn as nn

from belief_store import BeliefStore, SourceTier, ContentType


class TopKSAEEncoder(nn.Module):
    """Frozen SAE-style sparse encoder: c = TopK(Wenc(z - bpre) + benc).

    Constructed directly, the weights are random -- a structural
    placeholder only, concept keys are random projections. For real
    concept extraction use sae_loader.load_pretrained_sae(), which fills
    the same three tensors from a pretrained SAE trained on the target
    model's residual stream. Frozen by construction (Section 2.2 / Fig.
    in paper: "SAE Feature Decoder -- frozen").
    """

    def __init__(self, d_model: int, dict_size: int, k: int = 32):
        super().__init__()
        self.k = k
        self.b_pre = nn.Parameter(torch.zeros(d_model), requires_grad=False)
        self.W_enc = nn.Parameter(torch.randn(d_model, dict_size) * 0.02, requires_grad=False)
        self.b_enc = nn.Parameter(torch.zeros(dict_size), requires_grad=False)

    @property
    def d_in(self) -> int:
        return self.W_enc.shape[0]

    @property
    def d_out(self) -> int:
        return self.W_enc.shape[1]

    @property
    def device(self) -> torch.device:
        return self.W_enc.device

    @torch.no_grad()
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (..., d_model) -> c: (..., dict_size), top-k sparse."""
        pre_act = (z - self.b_pre) @ self.W_enc + self.b_enc
        topk_vals, topk_idx = torch.topk(pre_act, self.k, dim=-1)
        topk_vals = torch.relu(topk_vals)
        c = torch.zeros_like(pre_act)
        c.scatter_(-1, topk_idx, topk_vals)
        return c


class ResidualKeyEncoder(nn.Module):
    """Concept key = the residual vector itself, minus a fixed center.

    A drop-in for TopKSAEEncoder (same d_in / d_out / device / forward
    contract) that skips the SAE. On GPT-2 layer 8, mean-pooled residuals
    retrieve the right belief more often than mean-pooled SAE codes
    (top-1 0.72 vs 0.50 on 50 queries, paired diff +0.22, 95% CI
    [0.08, 0.36]; analysis_relevance.py) -- top-k sparse codes discard
    part of what the dense vector carries.

    Raw residual cosines are squeezed into ~0.85-0.92 for any two
    sentences (the space is anisotropic: everything shares a large common
    direction), so the center matters. sae_loader.load_residual_key_encoder
    uses the SAE's b_dec, initialised as the geometric median of this
    layer's activations over OpenWebText -- an outside-data estimate of
    that common direction, nothing fit on our own prompts.
    """

    def __init__(self, d_model: int, center: torch.Tensor | None = None):
        super().__init__()
        c = torch.zeros(d_model) if center is None else center.detach().float().clone()
        if c.shape != (d_model,):
            raise ValueError(f"center has shape {tuple(c.shape)}, expected ({d_model},)")
        self.register_buffer("center", c)

    @property
    def d_in(self) -> int:
        return self.center.shape[0]

    @property
    def d_out(self) -> int:
        return self.center.shape[0]

    @property
    def device(self) -> torch.device:
        return self.center.device

    @torch.no_grad()
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z - self.center


class BeliefGate(nn.Module):
    """Gates attention output by source-weighted, decaying belief confidence.

    forward() returns (z_gated, weff, flagged) so the caller can inspect
    per-token effective weights and which entries got flagged for
    reconsolidation this pass, without the gate owning the update logic
    itself (that stays in BeliefStore.reconsolidate / apply_rif, called
    separately post-generation per Section 3.5).

    RETRIEVAL MODES
    "token" (default, the paper's spec): each token's own SAE code is the
        query and its nearest belief always answers. On real GPT-2 codes
        this carries no relevance signal -- per-token best-match cosine
        separates related from unrelated prompts at AUROC 0.42, and the
        best-matching token is often a function word (analysis_relevance.py).
    "context": the query at token t is the running mean of SAE codes over
        tokens 1..t of the current prompt (causal, so prefill and each
        decode step continue the same mean; reset_context() starts a new
        prompt). Below theta_rel, or before min_ctx tokens of context,
        nothing relevant is stored:
          - weff = 0, so FOK's PIK is low and reasoning triggers;
          - z passes through unscaled -- there is no belief to gate by;
          - the entry is not touched or flagged (BeliefStore min_sim).
        theta_rel=0.15 is the best of {0.15, 0.20, 0.25, 0.30} on 12+12
        prompts with GPT-2 layer 8 (analysis_relevance.py section 5) --
        chosen on the same small set it was measured on, so provisional.
        A relevant match is flagged on every retrieval (theta_recon is
        not consulted; context codes never get near cos 0.85), so each
        belief actually used is checked against the output by NLI.
    last_sim / last_relevant hold the most recent pass's match cosine and
    relevance mask for inspection.
    """

    def __init__(self, d_model: int, dict_size: int = 4096, sae_k: int = 32,
                 theta_recon: float = 0.15, device: str | torch.device = "cpu",
                 encoder: TopKSAEEncoder | ResidualKeyEncoder | None = None,
                 retrieval: str = "token",
                 theta_rel: float = 0.15, min_ctx: int = 3):
        super().__init__()
        if retrieval not in ("token", "context"):
            raise ValueError(f"retrieval must be 'token' or 'context', got {retrieval!r}")
        if retrieval == "context" and not theta_rel > 0:
            # warm-up masking relies on a zero query (cos 0) being irrelevant
            raise ValueError(f"theta_rel must be > 0 in context mode, got {theta_rel}")
        self.retrieval = retrieval
        self.theta_rel = theta_rel
        self.min_ctx = min_ctx
        self.last_sim = None
        self.last_relevant = None
        self.reset_context()
        # Resolve to the concrete device (e.g. "mps:0", "cuda:0") by
        # querying a real tensor, rather than trusting the bare
        # torch.device(device) string -- see .type comparison note below.
        self.device = torch.zeros(1, device=device).device
        if encoder is None:
            encoder = TopKSAEEncoder(d_model, dict_size, k=sae_k)
        elif encoder.d_in != d_model:
            # a pretrained SAE is tied to one model's residual width; a
            # mismatch here means the wrong SAE for this model, not a
            # shape to broadcast around
            raise ValueError(
                f"encoder expects d_model={encoder.d_in}, gate built for {d_model}"
            )
        self.encoder = encoder.to(self.device)
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
        if self.retrieval == "context":
            return self._forward_context(z_attn, c, t_now)

        # Single batched store query instead of B*S individual ones (was
        # the throughput bottleneck -- see BeliefStore.query_batch
        # docstring). One off-device transfer for the whole block, one
        # vectorized numpy matmul, one on-device transfer back.
        c_flat = c.reshape(B * S, -1).detach().cpu().numpy()
        weff_np, _flagged_ids, sim = self.store.query_batch(
            c_flat, t_now=t_now, theta_recon=self.theta_recon, return_sim=True)
        weff = torch.from_numpy(weff_np).to(self.device).reshape(B, S)
        self.last_sim = sim.reshape(B, S)
        self.last_relevant = np.ones((B, S), dtype=bool)

        z_gated = z_attn * weff.unsqueeze(-1)  # broadcast scalar weight over d_model
        flagged = self.store.flagged_this_pass()
        return z_gated, weff, flagged

    def reset_context(self) -> None:
        """Start a new prompt: drop the running code sum (context mode)."""
        self._ctx_sum = None
        self._ctx_n = 0

    def _forward_context(self, z_attn, c, t_now):
        B, S, _ = z_attn.shape
        # running mean over the prompt so far, continued across calls so a
        # prefill pass followed by one-token decode passes gives the same
        # queries as a single pass over the whole sequence
        csum = torch.cumsum(c, dim=1)
        if self._ctx_sum is not None:
            if self._ctx_sum.shape[0] != B:
                raise ValueError("batch size changed mid-prompt; call reset_context() first")
            csum = csum + self._ctx_sum
        n = torch.arange(1, S + 1, device=c.device, dtype=c.dtype) + self._ctx_n  # (S,)
        q = csum / n[None, :, None]
        self._ctx_sum = csum[:, -1:].detach()
        self._ctx_n += S

        # warm-up positions (< min_ctx tokens of context) become zero
        # queries: cosine 0 is below theta_rel, so query_batch neither
        # touches nor flags their nearest entry -- otherwise a lone
        # prompt-initial "The" (cos 0.35 to beliefs starting with "The")
        # would leak flags into reconsolidation
        warm = (n >= self.min_ctx)
        q = q * warm[None, :, None]
        q_flat = q.reshape(B * S, -1).detach().cpu().numpy()
        weff_np, _flagged_ids, sim = self.store.query_batch(
            q_flat, t_now=t_now, theta_recon=0.0, min_sim=self.theta_rel, return_sim=True)
        relevant = sim.reshape(B, S) >= self.theta_rel
        self.last_sim = sim.reshape(B, S)
        self.last_relevant = relevant

        weff = torch.from_numpy(weff_np).to(self.device).reshape(B, S)
        rel_t = torch.from_numpy(relevant).to(self.device)
        gate_w = torch.where(rel_t, weff, torch.ones_like(weff))  # pass-through if nothing relevant
        z_gated = z_attn * gate_w.unsqueeze(-1)
        flagged = self.store.flagged_this_pass()
        return z_gated, weff, flagged

    # -- convenience for populating the store from outside -----------------

    def add_belief(self, concept_vec, content_kv, weight, source_tier: SourceTier,
                    content_type: ContentType, t_now: int):
        concept_np = concept_vec.detach().cpu().numpy() if torch.is_tensor(concept_vec) else concept_vec
        return self.store.create(concept_np, content_kv, weight, source_tier, content_type, t_now=t_now)