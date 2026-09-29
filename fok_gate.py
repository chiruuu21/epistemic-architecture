"""
EpT FOK Gate (Section 3.4) -- conditionally triggers internal reasoning
only when belief confidence falls below a threshold.


Design goal: make Quiet-STaR-style
reasoning overhead proportional to epistemic uncertainty, not uniform
across every token. That only holds if the expensive "thought" step is
actually skipped for confident tokens -- not computed for everyone and
then masked away. So this gate gathers only the triggered (low-PIK)
tokens, runs thought_fn on that subset, and scatters the result back.

SCOPING: thought_fn is a pluggable placeholder here, not a real
Quiet-STaR implementation. Actual rationale generation (parallel token
sampling per Zelikman et al. 2024, training the mixing head via
straight-through/RL since the trigger itself is a hard threshold and
therefore non-differentiable) is a substantial separate component and
out of scope for this pass. The default thought_fn is a
small MLP so the gate's control flow and compute-saving property can be
validated structurally; swap it for a real reasoning module later
without touching this class.

"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn


class _PlaceholderThought(nn.Module):
    """Structural stand-in for real Quiet-STaR-style rationale generation.
    NOT a real reasoning module -- see module docstring."""

    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class FOKGate(nn.Module):
    """Feeling-of-Knowing gate: conditional reasoning trigger.

    forward() returns (zout, PIK, trigger_mask) so callers can inspect
    what fraction of tokens triggered reasoning this pass, useful for
    both debugging and for Experiment 4.3 (FOK calibration / ECE).
    """

    def __init__(self, d_model: int, theta_fok: float = 0.5,
                 f: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                 thought_fn: Optional[nn.Module] = None):
        super().__init__()
        self.theta_fok = theta_fok
        self.f = f if f is not None else (lambda x: x)  # identity default; weff is already ~[0,1]
        self.thought_fn = thought_fn if thought_fn is not None else _PlaceholderThought(d_model)
        self.mixing_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, zffn: torch.Tensor, weff: torch.Tensor):
        """
        zffn: (B, S, D) -- feed-forward network output
        weff: (B, S) or (B, S, C) -- effective belief weight(s) per token.
              If (B, S, C), averaged over C (multiple concepts per token)
              per the paper's PIK = f(mean_c weff(c)) definition. The
              current BeliefGate produces (B, S) (single nearest-entry
              retrieval per token), which is the common case.
        Returns:
            zout: (B, S, D)
            PIK: (B, S) -- the computed feeling-of-knowing signal
            trigger_mask: (B, S) bool -- which tokens triggered reasoning
        """
        pik_input = weff.mean(dim=-1) if weff.dim() == 3 else weff
        PIK = self.f(pik_input)  # (B, S)
        trigger_mask = PIK < self.theta_fok  # (B, S) bool

        B, S, D = zffn.shape
        zffn_flat = zffn.reshape(B * S, D)
        mask_flat = trigger_mask.reshape(B * S)

        idx = mask_flat.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            # nothing triggered this pass -- skip thought_fn/mixing_head
            # entirely rather than running them on zero rows
            return zffn, PIK, trigger_mask

        selected = zffn_flat.index_select(0, idx)          # (T, D) -- gather: only the uncertain tokens
        thought = self.thought_fn(selected)                 # (T, D)
        combined = selected + thought                        # zffn + thought, per paper notation
        mixed = self.mixing_head(torch.cat([selected, combined], dim=-1))  # (T, D)

        # scatter back via torch.where (autograd-safe, avoids in-place
        # index assignment pitfalls): build a full-size update tensor
        # with mixed values in triggered rows, zero elsewhere, then
        # select per-row between update and the untouched original.
        # scatter (not index_copy) because aten::index_copy.out has no MPS
        # kernel; scatter is implemented on CPU/CUDA/MPS alike and is
        # likewise out-of-place, so the grad graph stays intact.
        scatter_idx = idx.unsqueeze(-1).expand(-1, D)
        full_update = torch.zeros_like(zffn_flat).scatter(0, scatter_idx, mixed)

        mask_expanded = mask_flat.unsqueeze(-1).expand(-1, D)
        zout_flat = torch.where(mask_expanded, full_update, zffn_flat)
        zout = zout_flat.reshape(B, S, D)

        return zout, PIK, trigger_mask
