"""
EpT Claim Extraction -- turning generation output into per-entry claims.

This fills the gap flagged in reconsolidation.py's docstring (note 1):
run_reconsolidation_pass needs `claims: dict[entry_id, str]`, and nothing
produced it.

WHAT THIS IS NOT. The paper's extract_claims(output, concept_key) implies
span attribution from hidden states back to natural language -- tracing
which tokens a given belief entry actually influenced, via attention
attribution or an SAE feature's activation footprint over the sequence.
That is a research problem, not a utility function, and it is NOT what
this module does.

WHAT THIS IS. A text-side approximation: split the output into sentences,
then for each flagged belief pick the sentence most plausibly *about*
that belief. Two strategies, both heuristic:

  LexicalClaimExtractor  -- IDF-weighted token overlap. No model, no
                            dependencies, fast. Fails on paraphrase, the
                            same weakness that makes MockNLI unusable.

  NLIRelevanceClaimExtractor -- prefilters candidates lexically, then
                            asks the NLI model which sentence is least
                            NEUTRAL with respect to the belief. Slower,
                            but "relatedness" is exactly what a 3-way NLI
                            head already measures, so it catches
                            paraphrase and negation that overlap misses.

BOTH CAN ATTACH THE WRONG SENTENCE. That risk is bounded on purpose: a
wrongly-attached claim can only reinforce or contest ONE entry, and
contesting requires the NLI backend to also clear its 0.70 confidence
threshold. Entries with no candidate above `min_score` get no claim at
all and land in summary["skipped_no_claim"] -- silence is the default,
not a guess. Tune min_score up if you would rather miss updates than
make wrong ones; for a VERIFIED-heavy store that is usually the right
trade.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Optional, Protocol, Sequence

from belief_store import BeliefEntry

# Sentence splitter: break on .!? followed by whitespace + a capital or
# digit. Deliberately dependency-free (no nltk/spacy download). The
# abbreviation guard covers the cases that actually show up in technical
# output -- "e.g." mid-sentence would otherwise split a claim in half.
_ABBREVIATIONS = {
    "e.g.", "i.e.", "etc.", "cf.", "vs.", "fig.", "eq.", "sec.", "approx.",
    "Dr.", "Mr.", "Ms.", "Prof.", "St.", "no.", "al.",
}
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_TOKEN = re.compile(r"[a-z0-9_]+")

# Words too common to carry topical signal. Kept small on purpose --
# IDF already down-weights whatever is frequent in the actual output.
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "and", "or", "but", "if", "then", "than", "that", "this", "these",
    "those", "of", "to", "in", "on", "at", "for", "with", "by", "from",
    "as", "it", "its", "we", "you", "they", "he", "she", "i", "not",
    "no", "do", "does", "did", "can", "could", "will", "would", "should",
    "has", "have", "had", "there", "their", "which", "who", "what",
}


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, rejoining false breaks on abbreviations."""
    if not text or not text.strip():
        return []

    raw = _SENTENCE_BOUNDARY.split(text.strip())
    out: list[str] = []
    for piece in raw:
        piece = piece.strip()
        if not piece:
            continue
        # if the previous sentence ended on a known abbreviation, this
        # was not a real boundary -- glue it back on
        if out and out[-1].split()[-1].lower() in _ABBREVIATIONS:
            out[-1] = f"{out[-1]} {piece}"
        else:
            out.append(piece)
    return out


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


class ClaimExtractor(Protocol):
    def __call__(self, output_text: str,
                 entries: Sequence[BeliefEntry]) -> dict[str, str]: ...


# --------------------------------------------------------------------------
# Lexical
# --------------------------------------------------------------------------

class LexicalClaimExtractor:
    """Attach the sentence with the highest IDF-weighted token overlap.

    IDF is computed over the output's own sentences, so words repeated
    throughout the generation (the topic itself, usually) stop dominating
    the match and the discriminative terms decide it.
    """

    def __init__(self, min_score: float = 0.15, allow_reuse: bool = False):
        """
        min_score: normalized overlap below which no claim is attached.
        allow_reuse: if False, each sentence is claimed by at most one
            entry -- the best match wins and the sentence is consumed.
            Prevents one vague sentence from reinforcing every belief in
            the store at once.
        """
        self.min_score = min_score
        self.allow_reuse = allow_reuse

    def __call__(self, output_text: str,
                 entries: Sequence[BeliefEntry]) -> dict[str, str]:
        sentences = split_sentences(output_text)
        comparable = [e for e in entries if e.content_text]
        if not sentences or not comparable:
            return {}

        sent_tokens = [set(_tokens(s)) for s in sentences]
        idf = self._idf(sent_tokens)

        # score every (entry, sentence) pair, then assign greedily by
        # descending score so the strongest match claims its sentence first
        scored = []
        for entry in comparable:
            e_tokens = set(_tokens(entry.content_text))
            if not e_tokens:
                continue
            for si, s_tokens in enumerate(sent_tokens):
                score = self._score(e_tokens, s_tokens, idf)
                if score >= self.min_score:
                    scored.append((score, entry.id, si))

        scored.sort(key=lambda x: -x[0])
        claims: dict[str, str] = {}
        used: set[int] = set()
        for score, entry_id, si in scored:
            if entry_id in claims:
                continue
            if si in used and not self.allow_reuse:
                continue
            claims[entry_id] = sentences[si]
            used.add(si)
        return claims

    @staticmethod
    def _idf(sent_tokens: list[set[str]]) -> dict[str, float]:
        n = len(sent_tokens)
        df = Counter()
        for toks in sent_tokens:
            df.update(toks)
        # smoothed IDF; +1 keeps a term appearing in every sentence at a
        # small positive weight rather than exactly zero
        return {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}

    @staticmethod
    def _score(e_tokens: set[str], s_tokens: set[str], idf: dict[str, float]) -> float:
        shared = e_tokens & s_tokens
        if not shared:
            return 0.0
        num = sum(idf.get(t, 1.0) for t in shared)
        denom = sum(idf.get(t, 1.0) for t in e_tokens)
        return num / denom if denom else 0.0


# --------------------------------------------------------------------------
# NLI-relevance
# --------------------------------------------------------------------------

class NLIRelevanceClaimExtractor:
    """Pick the sentence the NLI model finds least NEUTRAL w.r.t. the belief.

    A 3-way NLI head already answers "are these two texts related, and
    how" -- P(entailment) + P(contradiction) is a direct relatedness
    score, and unlike token overlap it survives paraphrase. Crucially it
    ranks a contradicting sentence just as high as a supporting one,
    which is what reconsolidation needs: the whole point is to notice
    when generation conflicts with a stored belief.

    Cost is |entries| x |candidates| forward passes, so candidates are
    prefiltered lexically first. With prefilter_k=3 that is a handful of
    pairs per entry, batched into one call.
    """

    def __init__(self, nli, prefilter_k: int = 3, min_relevance: float = 0.35,
                 lexical_min_score: float = 0.0):
        """
        nli: an NLI backend exposing classify_batch (HFNLIBackend).
        prefilter_k: lexical candidates per entry to score with the model.
        min_relevance: minimum P(entailment) + P(contradiction) to attach.
        """
        self.nli = nli
        self.prefilter_k = prefilter_k
        self.min_relevance = min_relevance
        self._lexical = LexicalClaimExtractor(min_score=lexical_min_score)

    def __call__(self, output_text: str,
                 entries: Sequence[BeliefEntry]) -> dict[str, str]:
        from nli_backend import NLIResult

        sentences = split_sentences(output_text)
        comparable = [e for e in entries if e.content_text]
        if not sentences or not comparable:
            return {}

        sent_tokens = [set(_tokens(s)) for s in sentences]
        idf = LexicalClaimExtractor._idf(sent_tokens)

        # 1. lexical prefilter -> candidate sentence indices per entry
        pairs: list[tuple[str, str]] = []
        owners: list[tuple[str, int]] = []
        for entry in comparable:
            e_tokens = set(_tokens(entry.content_text))
            ranked = sorted(
                range(len(sentences)),
                key=lambda si: -LexicalClaimExtractor._score(e_tokens, sent_tokens[si], idf),
            )[: self.prefilter_k]
            for si in ranked:
                pairs.append((entry.content_text, sentences[si]))
                owners.append((entry.id, si))

        if not pairs:
            return {}

        verdicts = self.nli.classify_batch(pairs)

        # 2. relatedness = 1 - P(neutral). Uses the full score
        # distribution rather than the thresholded label, because a
        # low-confidence contradiction still signals "these are about the
        # same thing" even when it is not actionable as a verdict.
        best: dict[str, tuple[float, int]] = {}
        for (entry_id, si), verdict in zip(owners, verdicts):
            relevance = 1.0 - verdict.scores.get(NLIResult.NEUTRAL, 0.0)
            if relevance < self.min_relevance:
                continue
            if entry_id not in best or relevance > best[entry_id][0]:
                best[entry_id] = (relevance, si)

        # 3. one sentence per entry, no sentence claimed twice
        claims: dict[str, str] = {}
        used: set[int] = set()
        for entry_id, (relevance, si) in sorted(best.items(), key=lambda kv: -kv[1][0]):
            if si in used:
                continue
            claims[entry_id] = sentences[si]
            used.add(si)
        return claims
