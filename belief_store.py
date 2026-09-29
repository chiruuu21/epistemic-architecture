"""
EpT Belief Store — core data structure and epistemic weighting logic.

Implements Section 3.1 (Belief Store) and the retrieval-triggered portion
of Section 3.5 (Selective Reconsolidation Update), with two corrections
to the paper's spec:

FIX 1 (weff state handling):
    Paper: weff includes 1[state == ACTIVE], which silently sends
    CONTESTED entries to zero — identical treatment to SUPERSEDED.
    That's wrong: a CONTESTED belief is exactly the case where the
    model most needs to retrieve it (with reduced confidence) so
    downstream components can reason about the conflict. SUPERSEDED
    entries, by contrast, should be excluded entirely — they have a
    successor entry that should be retrieved instead.

    Fix: SUPERSEDED -> hard exclude (weff = 0.0, not just "small").
         CONTESTED  -> multiply by a configurable penalty (default 0.5)
                       instead of zeroing.
         ACTIVE     -> full weight, no penalty.

FIX 2 (RIF floor):
    Paper: entry.weight -= eta_inhibit * cos(c, n.concept_key), unbounded
    below. Repeated retrieval of semantically-related concepts can drive
    weight negative, which then propagates a *sign flip* into
    zgated = zattn (dot) weff — i.e. RIF wouldn't just suppress a
    competing belief, it could invert it. That's a silent correctness
    bug, not just an edge case.

    Fix: clamp to [RIF_FLOOR, 1.0]. RIF_FLOOR is a small positive value
    (default 0.02), not 0 — this keeps "heavily suppressed" distinguishable
    from "entry does not exist" (weff == 0.0 only via SUPERSEDED or
    decay-to-negligible). Full removal is an explicit act (supersession),
    never an emergent side effect of gradual inhibition.
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


# --------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------

class SourceTier(Enum):
    OBSERVED = "OBSERVED"
    VERIFIED = "VERIFIED"
    STATED = "STATED"
    INFERRED = "INFERRED"


class ContentType(Enum):
    ARCHITECTURE = "ARCHITECTURE"
    IMPLEMENTATION = "IMPLEMENTATION"
    STATUS = "STATUS"


class BeliefState(Enum):
    ACTIVE = "ACTIVE"
    CONTESTED = "CONTESTED"
    SUPERSEDED = "SUPERSEDED"


# Trust lookup (Section 3.1 table)
SOURCE_TRUST: dict[SourceTier, float] = {
    SourceTier.OBSERVED: 1.00,
    SourceTier.VERIFIED: 0.90,
    SourceTier.STATED: 0.85,
    SourceTier.INFERRED: 0.70,
}

# Decay constants lambda = ln(2) / half_life_days (Section 3.1 table)
CONTENT_DECAY: dict[ContentType, float] = {
    ContentType.ARCHITECTURE: math.log(2) / 730,
    ContentType.IMPLEMENTATION: math.log(2) / 120,
    ContentType.STATUS: math.log(2) / 14,
}

# --- Fix constants ---
CONTESTED_PENALTY = 0.5   # weff multiplier for CONTESTED entries (Fix 1)
RIF_FLOOR = 0.02          # minimum weight after inhibition (Fix 2)


# --------------------------------------------------------------------------
# Belief entry
# --------------------------------------------------------------------------

@dataclass
class BeliefEntry:
    concept_key: np.ndarray            # SAE feature vector (the address)
    content_kv: object                 # sparse KV tensor (opaque here)
    weight: float                      # belief strength in [0, 1]
    source_tier: SourceTier
    content_type: ContentType
    state: BeliefState = BeliefState.ACTIVE
    created_at: int = 0                 # token step
    last_accessed: int = 0
    access_count: int = 0
    supersedes: Optional[str] = None
    content_text: Optional[str] = None  # human-readable claim text, for NLI comparison
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def effective_weight(self, t_now: int) -> float:
        """weff(e, t) with Fix 1 applied for state handling."""
        if self.state == BeliefState.SUPERSEDED:
            return 0.0  # hard exclude — successor entry should be retrieved instead

        base = (
            self.weight
            * SOURCE_TRUST[self.source_tier]
            * math.exp(-CONTENT_DECAY[self.content_type] * (t_now - self.last_accessed))
        )

        if self.state == BeliefState.CONTESTED:
            return base * CONTESTED_PENALTY  # reduced, not zero
        return base  # ACTIVE


# --------------------------------------------------------------------------
# Belief Store
# --------------------------------------------------------------------------

class BeliefStore:
    def __init__(self):
        self._entries: dict[str, BeliefEntry] = {}
        self._flagged_this_pass: list[str] = []

    # -- creation / lookup -------------------------------------------------

    def create(self, concept_key, content_kv, weight, source_tier,
               content_type, t_now=None, supersedes=None, content_text=None) -> BeliefEntry:
        t_now = t_now if t_now is not None else int(time.time())
        entry = BeliefEntry(
            concept_key=np.asarray(concept_key, dtype=np.float32),
            content_kv=content_kv,
            content_text=content_text,
            weight=weight,
            source_tier=source_tier,
            content_type=content_type,
            created_at=t_now,
            last_accessed=t_now,
            supersedes=supersedes,
        )
        # NOTE (bug fix): does NOT auto-transition the superseded entry's
        # state. Section 3.5's reconsolidation loop explicitly sets the
        # old entry to CONTESTED *before* creating its successor -- if
        # create() force-overwrote that to SUPERSEDED, the CONTESTED
        # penalty semantics (Fix 1: contested beliefs retrieve at reduced
        # confidence, not zero) would never actually be observable. The
        # `supersedes` field is provenance only; state transitions are
        # the caller's explicit responsibility (see reconsolidation.py).
        self._entries[entry.id] = entry
        return entry

    def query(self, concept_key, t_now, theta_recon=0.15) -> float:
        """Retrieve weff for the nearest entry to concept_key; flag for
        reconsolidation if prediction error exceeds theta_recon."""
        entry = self._nearest_entry(concept_key)
        if entry is None:
            return 0.0

        # BUG (found via query_batch cross-check): must compute weff
        # BEFORE mutating last_accessed, or decay is computed against
        # itself (t_now - t_now == 0) and silently never applies --
        # every query would look like it happened at t_now regardless
        # of how stale the entry actually was.
        weff = entry.effective_weight(t_now)
        entry.last_accessed = t_now
        entry.access_count += 1

        # prediction error proxy: cosine distance between query and stored key
        err = 1.0 - self._cosine(concept_key, entry.concept_key)
        if err > theta_recon and entry.id not in self._flagged_this_pass:
            self._flagged_this_pass.append(entry.id)

        return weff

    def query_batch(self, concept_keys: np.ndarray, t_now: int, theta_recon: float = 0.15):
        """Vectorized version of query() for many concept vectors at once.

        Replaces a per-token Python loop (each iteration doing its own
        dict traversal + cosine search) with a single matrix multiply
        against all stored entries. This is the fix for the throughput
        bottleneck flagged in BeliefGate: at seq=2048 the old approach
        meant 2048+ individual store.query() calls per forward pass;
        this does the equivalent work as one (M, D) x (D, N) matmul.

        concept_keys: (M, D) array -- e.g. a whole (batch*seq, dict_size)
                      block flattened from a forward pass.
        Returns:
            weff: (M,) float32 array of effective weights, one per query.
            flagged_ids: list of entry ids flagged for reconsolidation
                         this call (also merged into the internal
                         flagged_this_pass() buffer).

        Complexity: O(M * N) for the similarity matrix, same asymptotic
        work as the loop version, but vectorized in numpy rather than
        paying Python-level overhead N times. For very large belief
        stores (tens of thousands+ of entries) this still becomes the
        bottleneck eventually -- at that scale, swap the dense matmul
        for an approximate nearest-neighbor index (e.g. FAISS); that's
        a drop-in replacement for the similarity-matrix step only, the
        weighting/state/flagging logic below is unaffected.
        """
        entries = list(self._entries.values())
        M = concept_keys.shape[0]
        if not entries:
            return np.zeros(M, dtype=np.float32), []

        K = np.stack([e.concept_key for e in entries]).astype(np.float32)  # (N, D)
        Q = np.asarray(concept_keys, dtype=np.float32)                    # (M, D)

        K_norm = K / (np.linalg.norm(K, axis=1, keepdims=True) + 1e-8)
        Q_norm = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8)
        sim = Q_norm @ K_norm.T  # (M, N) -- the one matmul replacing M separate searches

        best_idx = np.argmax(sim, axis=1)          # (M,)
        best_sim = sim[np.arange(M), best_idx]      # (M,)

        weight_arr = np.array([e.weight for e in entries], dtype=np.float32)
        trust_arr = np.array([SOURCE_TRUST[e.source_tier] for e in entries], dtype=np.float32)
        decay_arr = np.array([CONTENT_DECAY[e.content_type] for e in entries], dtype=np.float32)
        last_acc_arr = np.array([e.last_accessed for e in entries], dtype=np.float32)
        state_arr = np.array(
            [0 if e.state == BeliefState.ACTIVE else (1 if e.state == BeliefState.CONTESTED else 2)
             for e in entries]
        )

        decay_factor = np.exp(-decay_arr * (t_now - last_acc_arr))
        # same state handling as BeliefEntry.effective_weight (Fix 1):
        # ACTIVE -> 1.0, CONTESTED -> penalty (not zero), SUPERSEDED -> hard 0
        state_factor = np.where(state_arr == 2, 0.0, np.where(state_arr == 1, CONTESTED_PENALTY, 1.0))
        weff_all = weight_arr * trust_arr * decay_factor * state_factor  # (N,)
        weff = weff_all[best_idx]  # (M,)

        # bookkeeping: update last_accessed / access_count for matched entries.
        # np.unique so an entry matched by many tokens gets one last_accessed
        # write (idempotent) and a correct summed access_count.
        unique_idx, counts = np.unique(best_idx, return_counts=True)
        for ui, cnt in zip(unique_idx, counts):
            entries[ui].last_accessed = t_now
            entries[ui].access_count += int(cnt)

        # reconsolidation flagging, vectorized (Section 3.3: err > theta_recon)
        err = 1.0 - best_sim
        flagged_idx = np.unique(best_idx[err > theta_recon])
        flagged_ids = []
        for fi in flagged_idx:
            eid = entries[fi].id
            if eid not in self._flagged_this_pass:
                self._flagged_this_pass.append(eid)
            flagged_ids.append(eid)

        return weff.astype(np.float32), flagged_ids

    def flagged_this_pass(self) -> list[BeliefEntry]:
        out = [self._entries[eid] for eid in self._flagged_this_pass if eid in self._entries]
        self._flagged_this_pass = []
        return out

    # -- similarity search ---------------------------------------------------

    def k_nearest(self, concept_key, k=5) -> list[BeliefEntry]:
        scored = [
            (self._cosine(concept_key, e.concept_key), e)
            for e in self._entries.values()
        ]
        scored.sort(key=lambda x: -x[0])
        return [e for _, e in scored[:k]]

    def _nearest_entry(self, concept_key) -> Optional[BeliefEntry]:
        nn = self.k_nearest(concept_key, k=1)
        return nn[0] if nn else None

    @staticmethod
    def _cosine(a, b) -> float:
        a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    # -- reconsolidation + RIF (Section 3.5, with Fix 2) --------------------

    def apply_rif(self, concept_key, eta_inhibit=0.05, k=5):
        """Retrieval-Induced Forgetting with a floored inhibition (Fix 2)."""
        neighbors = self.k_nearest(concept_key, k=k + 1)  # +1 since query itself may be included
        for n in neighbors:
            sim = self._cosine(concept_key, n.concept_key)
            if sim >= 0.999:
                continue  # this is the retrieved entry itself, not a competitor
            inhibited = n.weight - eta_inhibit * sim
            n.weight = max(RIF_FLOOR, min(1.0, inhibited))  # <-- the fix

    def reconsolidate(self, entry_id, nli_result, eta_reinforce=0.05):
        """Apply CONTESTED/reinforcement transition per NLI verdict."""
        entry = self._entries.get(entry_id)
        if entry is None:
            return None
        if nli_result == "CONTRADICTION":
            entry.state = BeliefState.CONTESTED
        elif nli_result == "ENTAILMENT":
            entry.weight = min(1.0, entry.weight + eta_reinforce)
        return entry