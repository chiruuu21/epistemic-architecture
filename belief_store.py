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

FIX 3 (trust-aware retrieval, optional: rank="trust"):
    Paper: retrieval is by cosine alone, so the nearest key answers
    whatever its weff. A reconsolidated belief's successor is created at
    the SAME concept address as the belief it contests, so under cosine
    ranking the two tie exactly and creation order decides -- the
    contested original keeps winning retrieval forever and the store can
    never switch to the correction (exp_convergence.py measures this).
    Fix: among entries whose cosine is within rank_margin of the best
    match, return the one with the highest weff. Relevance (min_sim) is
    still decided by the best cosine, so trust can only choose BETWEEN
    relevant beliefs, never make an irrelevant one relevant.

UNITS. CONTENT_DECAY is per DAY (Section 3.1 half-lives), but the paper
also says created_at is a "token step". Feed t_now in days. lm_hook
advances its clock once per forward pass, which is fine for single-turn
runs and wrong for anything multi-turn -- set t_start explicitly there.
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
        self.last_flag_sims: dict[str, float] = {}  # set by query_batch

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

    # -- vectorized helpers ---------------------------------------------------

    def _weff_all(self, entries, t_now) -> np.ndarray:
        """(N,) weff for every entry, same rules as BeliefEntry.effective_weight."""
        weight_arr = np.array([e.weight for e in entries], dtype=np.float32)
        trust_arr = np.array([SOURCE_TRUST[e.source_tier] for e in entries], dtype=np.float32)
        decay_arr = np.array([CONTENT_DECAY[e.content_type] for e in entries], dtype=np.float32)
        last_acc_arr = np.array([e.last_accessed for e in entries], dtype=np.float32)
        state_arr = np.array(
            [0 if e.state == BeliefState.ACTIVE else (1 if e.state == BeliefState.CONTESTED else 2)
             for e in entries]
        )
        decay_factor = np.exp(-decay_arr * (t_now - last_acc_arr))
        # ACTIVE -> 1.0, CONTESTED -> penalty (not zero), SUPERSEDED -> hard 0 (Fix 1)
        state_factor = np.where(state_arr == 2, 0.0, np.where(state_arr == 1, CONTESTED_PENALTY, 1.0))
        return (weight_arr * trust_arr * decay_factor * state_factor).astype(np.float32)

    @staticmethod
    def _similarity(Q, entries) -> np.ndarray:
        K = np.stack([e.concept_key for e in entries]).astype(np.float32)  # (N, D)
        Q = np.asarray(Q, dtype=np.float32)                                # (M, D)
        K_norm = K / (np.linalg.norm(K, axis=1, keepdims=True) + 1e-8)
        Q_norm = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8)
        return Q_norm @ K_norm.T  # (M, N)

    def _roots(self, entries) -> np.ndarray:
        """(N,) index of each entry's lineage root (follow `supersedes`)."""
        pos = {e.id: i for i, e in enumerate(entries)}
        out = np.empty(len(entries), dtype=np.int64)
        for i, e in enumerate(entries):
            r = e
            while r.supersedes is not None and r.supersedes in self._entries:
                r = self._entries[r.supersedes]
            out[i] = pos.get(r.id, i)
        return out

    def _select(self, sim, weff_all, rank: str, rank_margin: float, entries=None):
        """Pick one entry per query row. Returns (chosen_idx, best_sim):
        best_sim is the top cosine (decides relevance); chosen_idx is the
        entry that answers (Fix 3 -- differs from argmax only for
        'trust' / 'lineage')."""
        M = sim.shape[0]
        best_idx = np.argmax(sim, axis=1)
        best_sim = sim[np.arange(M), best_idx]
        if rank == "cosine":
            return best_idx, best_sim
        if rank == "lineage":
            # trust arbitrates only between VERSIONS OF THE SAME BELIEF (the
            # best match's root and everything that supersedes it), never
            # across topics. exp_convergence.py: plain 'trust' with a 0.05
            # margin let VERIFIED beliefs on neighbouring topics win
            # retrieval from the belief the context was actually about.
            roots = self._roots(entries)
            near = roots[None, :] == roots[best_idx][:, None]
        elif rank == "trust":
            near = sim >= (best_sim[:, None] - rank_margin)
        else:
            raise ValueError(f"rank must be 'cosine', 'trust' or 'lineage', got {rank!r}")
        # weff first; cosine only breaks exact weff ties
        score = np.where(near, weff_all[None, :] + 1e-6 * sim, -np.inf)
        return np.argmax(score, axis=1), best_sim

    def peek(self, concept_key, t_now, min_sim: Optional[float] = None,
             rank: str = "cosine", rank_margin: float = 0.05):
        """Read-only retrieval: (entry, cosine_to_entry, weff), or None if
        the store is empty or nothing clears min_sim. Touches no
        bookkeeping (no last_accessed, no access_count, no flag) -- for
        probing what the store WOULD answer, e.g. in exp_convergence.py."""
        entries = list(self._entries.values())
        if not entries:
            return None
        sim = self._similarity(np.asarray(concept_key)[None, :], entries)
        weff_all = self._weff_all(entries, t_now)
        idx, best_sim = self._select(sim, weff_all, rank, rank_margin, entries)
        if min_sim is not None and best_sim[0] < min_sim:
            return None
        i = int(idx[0])
        return entries[i], float(sim[0, i]), float(weff_all[i])

    def query_batch(self, concept_keys: np.ndarray, t_now: int, theta_recon: float = 0.15,
                    min_sim: Optional[float] = None, return_sim: bool = False,
                    rank: str = "cosine", rank_margin: float = 0.05):
        """Vectorized version of query() for many concept vectors at once.

        Replaces a per-token Python loop (each iteration doing its own
        dict traversal + cosine search) with a single matrix multiply
        against all stored entries. This is the fix for the throughput
        bottleneck flagged in BeliefGate: at seq=2048 the old approach
        meant 2048+ individual store.query() calls per forward pass;
        this does the equivalent work as one (M, D) x (D, N) matmul.

        concept_keys: (M, D) array -- e.g. a whole (batch*seq, dict_size)
                      block flattened from a forward pass.
        min_sim: relevance cutoff. Without it the nearest entry always
                 answers, however far away it is -- on real GPT-2 codes the
                 median best-match cosine is ~0.06 and every query still got
                 its nearest entry's full weight (analysis_relevance.py).
                 With it, a query whose best cosine is below min_sim gets
                 weff 0.0 ("nothing relevant stored") and does NOT touch the
                 entry: no last_accessed refresh (an irrelevant match would
                 otherwise reset the entry's decay clock) and no flag.
        return_sim: also return the (M,) best-match cosines.
        Returns:
            weff: (M,) float32 array of effective weights, one per query.
            flagged_ids: list of entry ids flagged for reconsolidation
                         this call (also merged into the internal
                         flagged_this_pass() buffer).
            best_sim: only if return_sim.

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
            self.last_nearest_weff = np.zeros(M, dtype=np.float32)
            self.last_flag_sims = {}
            empty = (np.zeros(M, dtype=np.float32), [])
            return (*empty, np.zeros(M, dtype=np.float32)) if return_sim else empty

        # one (M, D) x (D, N) matmul replacing M separate searches
        sim = self._similarity(concept_keys, entries)
        weff_all = self._weff_all(entries, t_now)  # (N,)
        best_idx, best_sim = self._select(sim, weff_all, rank, rank_margin, entries)
        # cosine to the entry that actually answers; equals best_sim under
        # rank="cosine". Prediction error (flagging) is measured against it.
        chosen_sim = sim[np.arange(M), best_idx]
        relevant = best_sim >= min_sim if min_sim is not None else np.ones(M, dtype=bool)
        weff = np.where(relevant, weff_all[best_idx], 0.0)  # (M,)
        # nearest entry's weff before the relevance mask, for graded
        # confidence (BeliefGate confidence="graded"); read-only, no
        # bookkeeping depends on it
        self.last_nearest_weff = weff_all[best_idx].astype(np.float32)

        # bookkeeping: update last_accessed / access_count for matched entries.
        # np.unique so an entry matched by many tokens gets one last_accessed
        # write (idempotent) and a correct summed access_count.
        unique_idx, counts = np.unique(best_idx[relevant], return_counts=True)
        for ui, cnt in zip(unique_idx, counts):
            entries[ui].last_accessed = t_now
            entries[ui].access_count += int(cnt)

        # reconsolidation flagging, vectorized (Section 3.3: err > theta_recon)
        err = 1.0 - chosen_sim
        flagged_idx = np.unique(best_idx[(err > theta_recon) & relevant])
        flagged_ids = []
        # strongest match each flagged entry got this call -- how sure
        # retrieval was that the context is ABOUT this belief (the contest
        # gate in reconsolidation reads it)
        flag_mask = (err > theta_recon) & relevant
        self.last_flag_sims = {}
        for fi in flagged_idx:
            eid = entries[fi].id
            if eid not in self._flagged_this_pass:
                self._flagged_this_pass.append(eid)
            flagged_ids.append(eid)
            self.last_flag_sims[eid] = float(chosen_sim[flag_mask & (best_idx == fi)].max())

        if return_sim:
            return weff.astype(np.float32), flagged_ids, best_sim.astype(np.float32)
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

    def apply_rif(self, concept_key, eta_inhibit=0.05, k=5, exclude_ids=None,
                  min_sim: float = 0.0):
        """Retrieval-Induced Forgetting with a floored inhibition (Fix 2).

        exclude_ids: the retrieved entry (paper: k_nearest(c) - {c}) --
            excluded by IDENTITY. Without it, falls back to skipping any
            neighbour at cosine >= 0.999 as "the query itself". That
            fallback is wrong once reconsolidation creates successors: a
            successor shares its predecessor's concept key, so the
            similarity rule exempts exactly the competitor RIF exists to
            suppress (and the successor itself).
        min_sim: competitor floor. Paper: the k nearest are inhibited
            whatever their similarity. In an anisotropic key space every
            belief's 5 nearest sit near cos 0.6 (exp_convergence.py), so
            RIF taxes unrelated bystanders on every retrieval. Setting this
            to the relevance threshold limits RIF to beliefs that would
            themselves be retrieved for this concept -- actual competitors.
        """
        if exclude_ids is None:
            neighbors = self.k_nearest(concept_key, k=k + 1)  # +1 since query itself may be included
            neighbors = [n for n in neighbors if self._cosine(concept_key, n.concept_key) < 0.999][:k]
        else:
            neighbors = [n for n in self.k_nearest(concept_key, k=k + len(exclude_ids))
                         if n.id not in exclude_ids][:k]
        for n in neighbors:
            sim = self._cosine(concept_key, n.concept_key)
            if sim <= 0 or sim < min_sim:
                continue  # anti-correlated / unrelated keys are not competitors
            inhibited = n.weight - eta_inhibit * sim
            n.weight = max(RIF_FLOOR, min(1.0, inhibited))  # <-- the fix

    def successors(self, entry_id) -> list[BeliefEntry]:
        """Entries whose `supersedes` points at entry_id, oldest first."""
        return [e for e in self._entries.values() if e.supersedes == entry_id]

    @property
    def entries(self) -> list[BeliefEntry]:
        return list(self._entries.values())

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