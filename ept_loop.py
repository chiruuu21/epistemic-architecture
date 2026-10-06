"""
EpT end-to-end loop -- the driver that connects the four components.

Until now BeliefStore, BeliefGate, FOKGate and run_reconsolidation_pass
each worked and were each tested, but nothing ran them in sequence. This
is that sequence, following the paper's block diagram (Section 3.2):

    per forward pass
        z_attn  --BeliefGate-->  z_gated, weff, flagged
        z_gated --FFN-------->   z_ffn
        z_ffn   --FOKGate---->   z_out, PIK, trigger_mask   (uses weff)

    once, after generation completes (Section 3.5)
        output_text --extract_claims--> claims
        claims + accumulated flagged --> run_reconsolidation_pass

TWO THINGS THE WIRING HAS TO GET RIGHT, both of which fail silently:

1. THE FLAGGED BUFFER IS DRAINED BY WHOEVER READS IT FIRST.
   BeliefStore.flagged_this_pass() returns the flagged entries AND clears
   the list. BeliefGate.forward() calls it every pass to build its return
   value. So calling run_reconsolidation_pass afterwards finds an empty
   buffer and does nothing -- no error, just an empty summary that reads
   exactly like "no beliefs needed updating". This loop accumulates the
   entries BeliefGate hands back and passes them in explicitly via the
   `flagged=` argument.

2. FLAGGING ACCUMULATES ACROSS THE WHOLE GENERATION, NOT PER TOKEN.
   Reconsolidation is specified as post-generation. A belief flagged at
   token 3 must still be reconsolidated after token 200, so `step()`
   merges into a dict keyed by entry id (dedup: an entry flagged in
   twenty passes is reconsolidated once) and `finish_generation()`
   consumes the accumulation.

SCOPE. This owns control flow and state, not the placeholder internals it
drives. Known placeholders, all documented at their definitions and none
of them fixed here:
  - TopKSAEEncoder uses random frozen weights, so concept keys are random
    projections and "semantically related" is not yet semantic.
  - FOKGate.thought_fn is an MLP, not Quiet-STaR rationale generation.
  - `ffn` defaults to identity; supply a real FFN to place the gates in a
    genuine transformer block.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

from belief_gate import BeliefGate
from belief_store import BeliefEntry, ContentType, SourceTier
from claim_extraction import LexicalClaimExtractor
from fok_gate import FOKGate
from reconsolidation import MockNLI, run_reconsolidation_pass


class EpTLoop:
    """Owns the gates and the per-generation flagged accumulation.

        loop = EpTLoop(d_model=64, nli_fn=HFNLIBackend())
        loop.add_belief(key, "the store keeps entries on the CPU",
                        weight=0.9, source_tier=SourceTier.VERIFIED,
                        content_type=ContentType.IMPLEMENTATION, t_now=0)

        for t, z_attn in enumerate(stream):
            out = loop.step(z_attn, t_now=t)

        summary = loop.finish_generation(generated_text, t_now=t)
    """

    def __init__(
        self,
        d_model: int,
        dict_size: int = 4096,
        sae_k: int = 32,
        theta_recon: float = 0.15,
        theta_fok: float = 0.5,
        device: str | torch.device = "cpu",
        nli_fn=None,
        claim_extractor=None,
        ffn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        eta_reinforce: float = 0.05,
        eta_inhibit: float = 0.05,
        rif_k: int = 5,
    ):
        self.belief_gate = BeliefGate(
            d_model=d_model, dict_size=dict_size, sae_k=sae_k,
            theta_recon=theta_recon, device=device,
        )
        self.device = self.belief_gate.device
        self.fok_gate = FOKGate(d_model=d_model, theta_fok=theta_fok).to(self.device)

        # MockNLI is a deliberate default: it keeps the loop runnable with
        # no model download, and it is loudly documented as non-semantic.
        # Pass HFNLIBackend() for anything real.
        self.nli_fn = nli_fn if nli_fn is not None else MockNLI()
        self.claim_extractor = claim_extractor or LexicalClaimExtractor()
        self.ffn = ffn

        self.eta_reinforce = eta_reinforce
        self.eta_inhibit = eta_inhibit
        self.rif_k = rif_k

        # entry_id -> BeliefEntry, accumulated across the generation.
        # A dict rather than a list so an entry flagged on many passes is
        # reconsolidated exactly once.
        self._pending: dict[str, BeliefEntry] = {}
        self.stats = {"steps": 0, "tokens": 0, "triggered": 0, "flag_events": 0}

    # -- store passthrough --------------------------------------------------

    @property
    def store(self):
        return self.belief_gate.store

    def add_belief(self, concept_vec, content_text: str, weight: float,
                   source_tier: SourceTier, content_type: ContentType,
                   t_now: int, content_kv=None):
        """Populate the store. Unlike BeliefGate.add_belief this requires
        content_text -- an entry without it can never be NLI-compared and
        would silently land in summary["skipped_no_text"] forever."""
        if not content_text:
            raise ValueError(
                "content_text is required: entries without it cannot be "
                "NLI-compared and are skipped by every reconsolidation pass"
            )
        concept_np = (concept_vec.detach().cpu().numpy()
                      if torch.is_tensor(concept_vec) else concept_vec)
        return self.store.create(
            concept_key=concept_np, content_kv=content_kv,
            content_text=content_text, weight=weight,
            source_tier=source_tier, content_type=content_type, t_now=t_now,
        )

    # -- per-pass -----------------------------------------------------------

    @torch.no_grad()
    def step(self, z_attn: torch.Tensor, t_now: int) -> dict:
        """One forward pass through both gates.

        z_attn: (B, S, D) attention output, on this loop's device.
        Returns z_out / weff / PIK / trigger_mask plus the entries flagged
        on THIS pass (already accumulated internally -- the caller does
        not need to hold on to them).
        """
        z_gated, weff, flagged = self.belief_gate(z_attn, t_now=t_now)

        # (1) from the module docstring: BeliefGate just drained the
        # store's flagged buffer. Stash the entries now or they are gone.
        for entry in flagged:
            self._pending[entry.id] = entry

        z_ffn = self.ffn(z_gated) if self.ffn is not None else z_gated
        z_out, PIK, trigger_mask = self.fok_gate(z_ffn, weff)

        self.stats["steps"] += 1
        self.stats["tokens"] += int(z_attn.shape[0] * z_attn.shape[1])
        self.stats["triggered"] += int(trigger_mask.sum().item())
        self.stats["flag_events"] += len(flagged)

        return {
            "z_out": z_out,
            "z_gated": z_gated,
            "weff": weff,
            "PIK": PIK,
            "trigger_mask": trigger_mask,
            "flagged": flagged,
        }

    # -- post-generation ----------------------------------------------------

    @property
    def pending_flagged(self) -> list[BeliefEntry]:
        """Entries flagged so far this generation, in first-flagged order."""
        return list(self._pending.values())

    def finish_generation(self, output_text: str, t_now: int,
                          claims: Optional[dict[str, str]] = None) -> dict:
        """Extract claims from the generated text and reconsolidate.

        claims: supply explicitly to bypass extraction (useful in tests
                and when an upstream pipeline already has attributions).

        Returns run_reconsolidation_pass's summary, plus:
            n_flagged        -- entries accumulated this generation
            n_claims_matched -- how many got a claim attached
            claims           -- the attributed claim text, for auditing
        The extractor is heuristic, so keeping the attributions in the
        summary is what makes a wrong update traceable afterwards rather
        than mysterious.
        """
        flagged = self.pending_flagged
        if claims is None:
            claims = self.claim_extractor(output_text, flagged)

        summary = run_reconsolidation_pass(
            store=self.store,
            claims=claims,
            t_now=t_now,
            nli_fn=self.nli_fn,
            eta_reinforce=self.eta_reinforce,
            eta_inhibit=self.eta_inhibit,
            rif_k=self.rif_k,
            flagged=flagged,  # (1) explicit -- the buffer is already empty
        )

        summary["n_flagged"] = len(flagged)
        summary["n_claims_matched"] = len(claims)
        summary["claims"] = claims

        self._pending.clear()
        return summary

    def reset_generation(self) -> None:
        """Drop accumulated flags without reconsolidating (aborted run)."""
        self._pending.clear()

    def __repr__(self) -> str:
        return (
            f"EpTLoop(device={self.device}, entries={len(self.store._entries)}, "
            f"pending_flagged={len(self._pending)}, "
            f"nli={type(self.nli_fn).__name__}, "
            f"extractor={type(self.claim_extractor).__name__})"
        )
