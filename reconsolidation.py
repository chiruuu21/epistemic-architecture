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

Requires: numpy only for MockNLI + the pass logic (fully testable
without torch). HFNLIClassifier additionally requires `transformers`
and network access to download a model -- not available in this
sandbox, use locally.
"""

from __future__ import annotations

from enum import Enum
from typing import Callable, Optional

from belief_store import BeliefStore, BeliefEntry, BeliefState, SourceTier


class NLIResult(Enum):
    ENTAILMENT = "ENTAILMENT"
    CONTRADICTION = "CONTRADICTION"
    NEUTRAL = "NEUTRAL"


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


class HFNLIClassifier:
    """Real NLI backed by a HuggingFace model (e.g. DeBERTa-v3 MNLI, per
    the paper's own Section 5 recommendation).

    NOT importable/runnable in this sandbox -- no network to download
    weights. Use on your own machine. Label names vary by model card;
    verify self._label_map matches whatever model you pass before
    trusting the output.
    """

    def __init__(self, model_name: str = "microsoft/deberta-v3-base-mnli-fever-anli",
                 device: Optional[str] = None):
        try:
            from transformers import pipeline
        except ImportError as e:
            raise ImportError(
                "HFNLIClassifier requires `transformers`: pip install transformers"
            ) from e
        self._pipe = pipeline("text-classification", model=model_name, device=device)
        self._label_map = {
            "ENTAILMENT": NLIResult.ENTAILMENT,
            "CONTRADICTION": NLIResult.CONTRADICTION,
            "NEUTRAL": NLIResult.NEUTRAL,
        }

    def __call__(self, premise: str, hypothesis: str) -> NLIResult:
        result = self._pipe(f"{premise} [SEP] {hypothesis}")[0]
        label = result["label"].upper()
        if label not in self._label_map:
            raise ValueError(
                f"Unrecognized NLI label '{label}' from model -- check "
                f"self._label_map matches this model's actual label set"
            )
        return self._label_map[label]


def run_reconsolidation_pass(
    store: BeliefStore,
    claims: dict[str, str],
    t_now: int,
    nli_fn: NLIFn,
    eta_reinforce: float = 0.05,
    eta_inhibit: float = 0.05,
    rif_k: int = 5,
) -> dict:
    """Consume BeliefStore.flagged_this_pass() and apply the full
    reconsolidation + RIF update. Call this once per forward pass,
    post-generation, per the paper's block diagram (Section 3.2).

    claims: entry_id -> claim text extracted by the caller's generation
            loop (see module docstring, note 1). Flagged entries with no
            corresponding claim are skipped, not silently dropped --
            check summary["skipped_no_claim"].

    Returns a summary dict for logging/debugging/Experiment 4.2
    (belief convergence tracking).
    """
    summary = {
        "contested": [],       # list of (old_entry_id, new_entry_id)
        "reinforced": [],      # list of entry_id
        "neutral": [],         # list of entry_id -- retrieved, claim compared, no state change
        "skipped_no_claim": [],  # list of entry_id -- flagged but no claim supplied
        "skipped_no_text": [],   # list of entry_id -- flagged but entry has no content_text to compare against
    }

    flagged = store.flagged_this_pass()

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
            store.apply_rif(entry.concept_key, eta_inhibit=eta_inhibit, k=rif_k)
            continue

        result = nli_fn(entry.content_text, claim_text)

        if result == NLIResult.CONTRADICTION:
            entry.state = BeliefState.CONTESTED  # penalized, not excluded (Fix 1) -- and
            # NOT overwritten to SUPERSEDED by the create() call below,
            # since create() no longer has that side effect (bug fix
            # applied to belief_store.py this session)
            new_entry = store.create(
                concept_key=entry.concept_key,
                content_kv=None,  # would be re-encoded from claim_text in a full pipeline
                content_text=claim_text,
                weight=0.5,
                source_tier=SourceTier.INFERRED,
                content_type=entry.content_type,
                t_now=t_now,
                supersedes=entry.id,
            )
            summary["contested"].append((entry.id, new_entry.id))

        elif result == NLIResult.ENTAILMENT:
            entry.weight = min(1.0, entry.weight + eta_reinforce)
            entry.access_count += 1
            entry.last_accessed = t_now
            summary["reinforced"].append(entry.id)

        else:
            summary["neutral"].append(entry.id)

        # RIF fires regardless of NLI outcome -- retrieval happened, so
        # semantically competing beliefs get suppressed either way
        # (Section 3.5's loop runs this unconditionally, not gated on
        # the CONTRADICTION/ENTAILMENT branch)
        store.apply_rif(entry.concept_key, eta_inhibit=eta_inhibit, k=rif_k)

    return summary
