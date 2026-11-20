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
  - TopKSAEEncoder uses random frozen weights by default, so concept keys
    are random projections. Pass encoder=load_pretrained_sae() (and real
    activations via lm_hook.GPT2Activations) for meaningful keys.
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
        encoder=None,
        retrieval: str = "token",
        theta_rel: float = 0.15,
        min_ctx: int = 3,
        confidence: str = "binary",
        tau: float = 0.05,
        rank: str = "cosine",
        rank_margin: float = 0.05,
        eta_contest: float = 0.0,
        dedup_successors: bool = False,
        supersede: bool = False,
        rif_min_sim: float = 0.0,
        contest_min_sim: Optional[float] = None,
        pik_model=None,
    ):
        self.belief_gate = BeliefGate(
            d_model=d_model, dict_size=dict_size, sae_k=sae_k,
            theta_recon=theta_recon, device=device, encoder=encoder,
            retrieval=retrieval, theta_rel=theta_rel, min_ctx=min_ctx,
            confidence=confidence, tau=tau, rank=rank, rank_margin=rank_margin,
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
        # reconsolidation revisions (reconsolidation.run_reconsolidation_pass
        # docstring); all off = the paper's update rule
        self.eta_contest = eta_contest
        self.dedup_successors = dedup_successors
        self.supersede = supersede
        self.rif_min_sim = rif_min_sim
        self.contest_min_sim = contest_min_sim
        # calibrated store-plus-model confidence (calibration.CombinedPIK);
        # see answer_confidence()
        self.pik_model = pik_model
        self.decision = None

        # entry_id -> BeliefEntry, accumulated across the generation.
        # A dict rather than a list so an entry flagged on many passes is
        # reconsolidated exactly once.
        self._pending: dict[str, BeliefEntry] = {}
        self._pending_sim: dict[str, float] = {}  # best match per pending entry
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
        prefill = self.belief_gate._ctx_n == 0
        z_gated, weff, flagged = self.belief_gate(z_attn, t_now=t_now)
        if prefill and self.belief_gate.retrieval == "context":
            # the FOK decision point: last prompt token, before any
            # generated token enters the running mean. This is what
            # answer_confidence() scores (and what calibration_study.py's
            # pooled question key reproduces).
            g = self.belief_gate
            self.decision = {
                "sim": g.last_sim[:, -1].copy(),
                "weff": g.last_nearest_weff[:, -1].copy(),
                "use_store": g.last_relevant[:, -1].copy(),
            }

        # (1) from the module docstring: BeliefGate just drained the
        # store's flagged buffer. Stash the entries now or they are gone.
        sims = self.belief_gate.last_flag_sims
        for entry in flagged:
            self._pending[entry.id] = entry
            s = sims.get(entry.id)
            if s is not None:
                self._pending_sim[entry.id] = max(s, self._pending_sim.get(entry.id, -1.0))

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
            "match_sim": self.belief_gate.last_sim,
            "relevant": self.belief_gate.last_relevant,
        }

    # -- post-generation ----------------------------------------------------

    @property
    def pending_flagged(self) -> list[BeliefEntry]:
        """Entries flagged so far this generation, in first-flagged order."""
        return list(self._pending.values())

    def answer_confidence(self, lm_prob) -> "np.ndarray":
        """Calibrated P(IK) for the answer just generated -- store-plus-model
        confidence (calibration.CombinedPIK; Report 4's revision of the
        paper's Section 3.4).

        lm_prob: the LM's own probability of its answer (product of the
            generated tokens' probabilities), shape (B,) or scalar.

        The FOK gate's TRIGGER stays store-based (is anything relevant
        stored?) because it fires mid-network, before the LM's answer
        probability exists. The CONFIDENCE reported for the answer is
        computed afterwards and uses both signals: the store's match
        quality on the store path, the LM's own probability on the FOK
        path. Store-only PIK cannot see what the model knows from its
        weights (ECE 0.085 on Qwen2.5-3B vs 0.012 for this).
        """
        import numpy as np
        if self.pik_model is None:
            raise RuntimeError("no pik_model: pass EpTLoop(pik_model=CombinedPIK...)")
        if self.decision is None:
            raise RuntimeError("no decision recorded: run a prompt in context mode first")
        d = self.decision
        lm = np.broadcast_to(np.asarray(lm_prob, dtype=np.float64), d["sim"].shape)
        return self.pik_model.predict(d["sim"], d["weff"], d["use_store"], lm)

    def finish_generation(self, output_text: str, t_now: int,
                          claims: Optional[dict[str, str]] = None,
                          claim_tier: Optional[SourceTier] = None) -> dict:
        """Extract claims from the generated text and reconsolidate.

        claims: supply explicitly to bypass extraction (useful in tests
                and when an upstream pipeline already has attributions).
        claim_tier: where output_text came from. None (default) = the
                model's own generation, so corrections are INFERRED as in
                the paper. Pass the source's tier when the text is observed
                context (Experiment 4.2's evidence is VERIFIED).

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
            claim_tier=claim_tier,
            eta_contest=self.eta_contest,
            dedup_successors=self.dedup_successors,
            supersede=self.supersede,
            rif_min_sim=self.rif_min_sim,
            match_sims=dict(self._pending_sim),
            contest_min_sim=self.contest_min_sim,
        )

        summary["n_flagged"] = len(flagged)
        summary["n_claims_matched"] = len(claims)
        summary["claims"] = claims

        self._pending.clear()
        self._pending_sim.clear()
        self.belief_gate.reset_context()
        return summary

    def reset_generation(self) -> None:
        """Drop accumulated flags without reconsolidating (aborted run)."""
        self._pending.clear()
        self._pending_sim.clear()
        self.belief_gate.reset_context()

    def __repr__(self) -> str:
        return (
            f"EpTLoop(device={self.device}, entries={len(self.store._entries)}, "
            f"pending_flagged={len(self._pending)}, "
            f"nli={type(self.nli_fn).__name__}, "
            f"extractor={type(self.claim_extractor).__name__})"
        )
