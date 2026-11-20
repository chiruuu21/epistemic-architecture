"""
EpT Selective Reconsolidation Update (Section 3.5) -- NLI wiring and the
pass that ties BeliefGate's flagged entries to actual store mutations.

    for entry in BeliefStore.flagged_this_pass():
        claim = extract_claims(output, entry.concept_key)
        if NLI(claim, entry.content_kv) == CONTRADICTION:
            entry.state = CONTESTED
            BeliefStore.create(content=claim, source_tier=INFERRED,
                                weight=0.5, supersedes=entry.id)
        elif NLI(claim, entry.content_kv) == ENTAILMENT:
            entry.weight = min(1.0, entry.weight + eta_reinforce)
            ...
        # RIF
        for n in BeliefStore.k_nearest(c, k=5) - {c}:
            n.weight -= eta_inhibit * cos(c, n.concept_key)

SCOPING NOTES (gaps in the paper's spec, resolved here):

1. extract_claims(output, concept_key) -- the paper doesn't specify how
   generation output becomes a textual claim tied to a specific belief
   entry; that's a nontrivial NLP problem in its own right (span
   attribution from hidden states back to natural language). This module
   does NOT implement claim extraction -- it takes claim text as an input
   (`claims: dict[entry_id, str]`), supplied by whatever generation loop
   calls this. That keeps this module's scope to "given a claim, run the
   epistemic update correctly," which is the part actually specified.

2. NLI(claim, entry.content_kv) -- content_kv is an opaque sparse KV
   tensor (Memory3-format), and you cannot run textual NLI against a KV
   tensor directly. BeliefEntry now has a `content_text` field (see
   belief_store.py) for exactly this purpose; NLI runs against that.
   Entries created without content_text can't be compared and are
   skipped with a warning rather than silently mishandled.

3. DIRECTION. NLI is not symmetric, so which text is the premise
   matters. We call nli_fn(entry.content_text, claim_text): the STORED
   BELIEF is the premise, the NEW CLAIM is the hypothesis. Read that as
   "given what I already believe, does this new claim follow?" -- which
   is the question the reinforce/contest branch is actually asking.
   Reversing the arguments changes the verdict on any entailment that
   only holds one way (a specific belief entails a general claim, not
   the reverse).

Requires: numpy only for MockNLI + the pass logic (fully testable
without torch). The real backend lives in nli_backend.HFNLIBackend and
pulls in torch + transformers; this module imports only the enum from
there, so the pass logic stays dependency-light.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Union

from belief_store import (BeliefStore, BeliefEntry, BeliefState, SourceTier,
                          SOURCE_TRUST, RIF_FLOOR)
# NLIResult is defined in nli_backend (which owns the NLI contract) and
# re-exported here so existing `from reconsolidation import NLIResult`
# imports keep working. nli_backend has no heavy module-scope imports.
from nli_backend import NLIResult, NLIVerdict

__all__ = [
    "NLIResult", "NLIVerdict", "NLIFn", "MockNLI",
    "HFNLIClassifier", "run_reconsolidation_pass",
]

NLIFn = Callable[[str, str], NLIResult]  # (premise, hypothesis) -> result


class MockNLI:
    """Deterministic, non-ML heuristic NLI for testing the reconsolidation
    PLUMBING without a real model. Word-overlap + negation-mismatch only.

    NOT semantically meaningful -- do not use for anything beyond
    validating that flagged entries flow through the pass correctly.
    Swap for HFNLIClassifier (or your own) for anything real.
    """

    NEGATION_WORDS = {"not", "no", "never", "isn't", "doesn't", "won't", "false", "cannot"}

    def __call__(self, premise: str, hypothesis: str) -> NLIResult:
        p_words = set(premise.lower().split())
        h_words = set(hypothesis.lower().split())
        overlap = p_words & h_words
        overlap_ratio = len(overlap) / max(len(p_words), 1)

        h_neg = bool(h_words & self.NEGATION_WORDS)
        p_neg = bool(p_words & self.NEGATION_WORDS)

        if overlap_ratio > 0.3 and (h_neg != p_neg):
            return NLIResult.CONTRADICTION
        if overlap_ratio > 0.6:
            return NLIResult.ENTAILMENT
        return NLIResult.NEUTRAL


def HFNLIClassifier(*args, **kwargs):
    """Deprecated alias for nli_backend.HFNLIBackend.

    The previous implementation here was silently wrong in two ways: it
    passed f"{premise} [SEP] {hypothesis}" as a single string to a
    sentence-PAIR model, and it assumed a fixed label ordering that only
    some MNLI checkpoints use. Both produced confident, plausible,
    incorrect verdicts rather than errors. See nli_backend's module
    docstring for the full write-up.
    """
    import warnings
    from nli_backend import HFNLIBackend

    warnings.warn(
        "HFNLIClassifier is deprecated and its old implementation was "
        "incorrect (single-string pair encoding + hardcoded label order). "
        "Use nli_backend.HFNLIBackend instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return HFNLIBackend(*args, **kwargs)


def _classify_all(nli_fn, pairs: Sequence[tuple[str, str]]) -> list[NLIVerdict]:
    """Run every (premise, hypothesis) pair through `nli_fn`, batching when
    the backend supports it.

    Backends exposing `classify_batch` (HFNLIBackend, ScriptedNLI) get one
    padded forward pass for the whole flagged set instead of one per
    belief -- the difference between ~40 sequential GPU round-trips and 3
    batches on a typical pass. Plain callables (MockNLI, any user lambda)
    still work through the one-at-a-time path.

    Returns NLIVerdicts either way, so downstream code has a single shape
    to handle; bare-callable results are wrapped with confidence 1.0 since
    they carry no score.
    """
    if not pairs:
        return []

    batch_fn = getattr(nli_fn, "classify_batch", None)
    if callable(batch_fn):
        return list(batch_fn(pairs))

    out = []
    for premise, hypothesis in pairs:
        result = nli_fn(premise, hypothesis)
        if isinstance(result, NLIVerdict):
            out.append(result)
        else:
            out.append(NLIVerdict(
                label=result, confidence=1.0,
                scores={result: 1.0}, argmax_label=result,
            ))
    return out


def run_reconsolidation_pass(
    store: BeliefStore,
    claims: dict[str, str],
    t_now: int,
    nli_fn: NLIFn,
    eta_reinforce: float = 0.05,
    eta_inhibit: float = 0.05,
    rif_k: int = 5,
    flagged: Optional[list[BeliefEntry]] = None,
    claim_tier: Optional[SourceTier] = None,
    successor_weight: float = 0.5,
    eta_contest: float = 0.0,
    dedup_successors: bool = False,
    supersede: bool = False,
    rif_min_sim: float = 0.0,
    match_sims: Optional[dict[str, float]] = None,
    contest_min_sim: Optional[float] = None,
) -> dict:
    """Consume BeliefStore.flagged_this_pass() and apply the full
    reconsolidation + RIF update. Call this once per forward pass,
    post-generation, per the paper's block diagram (Section 3.2).

    claims: entry_id -> claim text extracted by the caller's generation
            loop (see module docstring, note 1). Flagged entries with no
            corresponding claim are skipped, not silently dropped --
            check summary["skipped_no_claim"].

    flagged: entries to reconsolidate. Defaults to draining the store's
            own buffer.

            PASS THIS EXPLICITLY IF ANYTHING ELSE ALREADY READ THE BUFFER.
            BeliefStore.flagged_this_pass() is destructive -- it returns
            the flagged entries and clears the list in the same call. And
            BeliefGate.forward() calls it on every forward pass to build
            its own return value. So the obvious wiring,

                _, _, flagged = gate(z_attn, t_now)     # drains the buffer
                run_reconsolidation_pass(store, claims, t_now, nli)  # sees []

            produces a reconsolidation pass that silently does nothing:
            no error, no warning, an empty summary that looks like "no
            beliefs needed updating". Generation accumulates flagged
            entries over many forward passes anyway, so the driver in
            ept_loop.py collects them across the whole generation and
            hands them over here.

    REVISIONS TO THE PAPER'S UPDATE (all off by default = paper spec;
    exp_convergence.py ablates each one):

    claim_tier: where the claim came from. The paper hardcodes
        source_tier=INFERRED for every successor, which is right for the
        model's own generated output and wrong for Experiment 4.2, whose
        evidence is VERIFIED context. None keeps the paper (successor is
        INFERRED, no promotion). Given, the successor takes this tier, and
        an entailed belief is promoted to it if it is more trusted
        ("confirmed against an authoritative source").
    eta_contest: contradiction also WEAKENS the contested belief
        (weight -= eta_contest, floored). Paper: contradiction only flips
        state, so weight(A) -- the paper's own 4.2 metric -- never moves
        except through RIF. Reconsolidation is "strengthen, update, or
        weaken" (Section 1); the paper implements only the first two.
    dedup_successors: if the contradicted belief already has a successor
        that the new claim entails, reinforce that successor instead of
        creating another. Paper: every contradiction of the same belief
        creates a fresh INFERRED entry, so three turns of the same
        evidence leave three duplicate corrections.
    rif_min_sim: only neighbours at least this similar are inhibited
        (BeliefStore.apply_rif). 0.0 = the paper (any of the k nearest).
    contest_min_sim: topic gate for the destructive branch. A
        CONTRADICTION only contests a belief if retrieval matched it at
        cosine >= contest_min_sim (match_sims[entry.id], the best match
        the generation produced); below that it is treated as NEUTRAL and
        listed in summary["contest_gated"]. Why: DeBERTa-v3-base calls
        ~4% of completely unrelated pairs CONTRADICTION at >= 0.70
        ("The Eiffel Tower is in Berlin" vs "The Sun is a star"), so NLI
        cannot be the judge of whether the claim is about the belief.
        Key cosine can (own vs cross-topic AUROC 0.99, exp_convergence.py).
        Same logic as the asymmetric NLI thresholds: the destructive
        action needs the stronger evidence. None = the paper (no gate).
    supersede: the paper defines SUPERSEDED but never enters it. Here a
        CONTESTED belief becomes SUPERSEDED once one of its successors is
        confirmed (entailed) and outweighs it (successor weff > its weff).

    Returns a summary dict for logging/debugging/Experiment 4.2
    (belief convergence tracking).
    """
    new_tier = claim_tier if claim_tier is not None else SourceTier.INFERRED
    summary = {
        "contested": [],       # list of (old_entry_id, new_entry_id)
        "reinforced": [],      # list of entry_id
        "neutral": [],         # list of entry_id -- retrieved, claim compared, no state change
        "skipped_no_claim": [],  # list of entry_id -- flagged but no claim supplied
        "skipped_no_text": [],   # list of entry_id -- flagged but entry has no content_text to compare against
        "abstained": [],       # list of entry_id -- model picked a label but under threshold, treated as neutral
        "verdicts": {},        # entry_id -> NLIVerdict, for Experiment 4.2 logging
    }

    if flagged is None:
        flagged = store.flagged_this_pass()

    # -- phase 1: classify (read-only, so hoisting it out of the mutation
    # loop below cannot change semantics) ---------------------------------
    #
    # Batching is the whole point of splitting the pass in two: a flagged
    # set of N beliefs becomes ceil(N / batch_size) forward passes instead
    # of N. The mutation loop that follows still walks `flagged` in its
    # original order, so RIF interleaving -- which is order-dependent,
    # since apply_rif mutates neighbour weights -- is unchanged.
    comparable = [
        entry for entry in flagged
        if claims.get(entry.id) is not None and entry.content_text is not None
    ]
    # premise = stored belief, hypothesis = new claim (see docstring note 3)
    pairs = [(entry.content_text, claims[entry.id]) for entry in comparable]
    verdicts = {entry.id: v for entry, v in zip(comparable, _classify_all(nli_fn, pairs))}

    # -- phase 2: mutate, in flagged order ---------------------------------
    for entry in flagged:
        claim_text = claims.get(entry.id)
        if claim_text is None:
            summary["skipped_no_claim"].append(entry.id)
            continue

        if entry.content_text is None:
            summary["skipped_no_text"].append(entry.id)
            # still run RIF -- retrieval happened regardless of whether
            # we could NLI-compare it, and RIF is about suppressing
            # competitors of the retrieved concept, not about the
            # comparison outcome
            store.apply_rif(entry.concept_key, eta_inhibit=eta_inhibit, k=rif_k, min_sim=rif_min_sim)
            continue

        verdict = verdicts[entry.id]
        result = verdict.label
        summary["verdicts"][entry.id] = verdict
        if verdict.abstained:
            # the model's argmax was CONTRADICTION or ENTAILMENT but below
            # its threshold, so it was downgraded to NEUTRAL. Recorded
            # separately -- an abstention means "the model had an opinion
            # we declined to act on", which is a different signal from
            # "the model saw no relation" when tuning thresholds.
            summary["abstained"].append(entry.id)

        if (result == NLIResult.CONTRADICTION and contest_min_sim is not None
                and (match_sims or {}).get(entry.id, 1.0) < contest_min_sim):
            summary.setdefault("contest_gated", []).append(entry.id)
            result = NLIResult.NEUTRAL

        winner = entry  # the belief this retrieval ends up supporting (RIF exempts it)
        if result == NLIResult.CONTRADICTION:
            if entry.state == BeliefState.SUPERSEDED:
                # already replaced; contradicting it again is not news
                summary["neutral"].append(entry.id)
                store.apply_rif(entry.concept_key, eta_inhibit=eta_inhibit, k=rif_k, min_sim=rif_min_sim,
                                exclude_ids={entry.id})
                continue
            entry.state = BeliefState.CONTESTED  # penalized, not excluded (Fix 1) -- and
            # NOT overwritten to SUPERSEDED by the create() call below,
            # since create() no longer has that side effect (bug fix
            # applied to belief_store.py this session)
            if eta_contest:
                entry.weight = max(RIF_FLOOR, entry.weight - eta_contest)
            existing = None
            if dedup_successors:
                live = [s for s in store.successors(entry.id)
                        if s.state != BeliefState.SUPERSEDED and s.content_text]
                if live:
                    checks = _classify_all(nli_fn, [(s.content_text, claim_text) for s in live])
                    for s, v in zip(live, checks):
                        if v.label == NLIResult.ENTAILMENT:
                            existing = s
                            break
            if existing is not None:
                _reinforce(existing, eta_reinforce, t_now, claim_tier)
                summary["reinforced"].append(existing.id)
                summary.setdefault("deduplicated", []).append((entry.id, existing.id))
                winner = existing
            else:
                new_entry = store.create(
                    concept_key=entry.concept_key,  # same concept address, new content
                    content_kv=None,  # would be re-encoded from claim_text in a full pipeline
                    content_text=claim_text,
                    weight=successor_weight,
                    source_tier=new_tier,
                    content_type=entry.content_type,
                    t_now=t_now,
                    supersedes=entry.id,
                )
                summary["contested"].append((entry.id, new_entry.id))
                winner = new_entry

        elif result == NLIResult.ENTAILMENT:
            _reinforce(entry, eta_reinforce, t_now, claim_tier)
            summary["reinforced"].append(entry.id)

        else:
            summary["neutral"].append(entry.id)

        if supersede and result != NLIResult.NEUTRAL:
            _maybe_supersede(store, winner, t_now, summary)

        # RIF fires regardless of NLI outcome -- retrieval happened, so
        # semantically competing beliefs get suppressed either way
        # (Section 3.5's loop runs this unconditionally, not gated on
        # the CONTRADICTION/ENTAILMENT branch). The retrieved entry is
        # excluded by identity (paper: k_nearest(c) - {c}); when the
        # retrieval ended up supporting a successor, that successor is
        # exempt too -- it is what was "practiced".
        store.apply_rif(entry.concept_key, eta_inhibit=eta_inhibit, k=rif_k, min_sim=rif_min_sim,
                        exclude_ids={entry.id, winner.id})

    return summary


def _reinforce(entry: BeliefEntry, eta: float, t_now: int,
               claim_tier: Optional[SourceTier]) -> None:
    entry.weight = min(1.0, entry.weight + eta)
    entry.access_count += 1
    entry.last_accessed = t_now
    if claim_tier is not None and SOURCE_TRUST[claim_tier] > SOURCE_TRUST[entry.source_tier]:
        entry.source_tier = claim_tier  # confirmed by more trusted evidence


def _maybe_supersede(store: BeliefStore, confirmed: BeliefEntry, t_now: int,
                     summary: dict) -> None:
    """A CONTESTED predecessor is retired once its successor is confirmed
    and carries more effective weight than it does."""
    if confirmed.supersedes is None or confirmed.access_count == 0:
        return  # never confirmed: a single contradiction is not enough
    prev = store._entries.get(confirmed.supersedes)
    if prev is None or prev.state != BeliefState.CONTESTED:
        return
    if confirmed.effective_weight(t_now) > prev.effective_weight(t_now):
        prev.state = BeliefState.SUPERSEDED
        summary.setdefault("superseded", []).append((prev.id, confirmed.id))
