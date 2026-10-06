"""
EpT NLI Backend (Section 3.5) -- real cross-encoder NLI for the
reconsolidation update.

This module owns the NLI contract (`NLIResult`, `NLIVerdict`, `NLIBackend`)
and provides a production HuggingFace implementation. `reconsolidation.py`
re-exports `NLIResult` from here, so existing imports keep working.

WHY THIS MODULE EXISTS (what was wrong with the old HFNLIClassifier):

1. PAIR ENCODING. The old code called the pipeline with a single string,
   f"{premise} [SEP] {hypothesis}". A cross-encoder NLI model is trained
   on a *sentence pair* -- two segments with distinct token_type_ids and a
   real [SEP] boundary inserted by the tokenizer. Passing one flat string
   means the literal characters "[SEP]" get tokenized as ordinary text and
   the whole thing lands in segment 0. The model still returns a confident
   softmax; it is just answering a different question than the one asked.
   This is the worst failure mode available -- silently wrong, never
   raises. Fixed here by tokenizing (premise, hypothesis) as a true pair.

2. LABEL ORDER. The old code mapped by uppercased label name against a
   hardcoded dict. Label *order* is model-specific and not inferable:

       MoritzLaurer/DeBERTa-v3-*-mnli-fever-anli -> [entailment, neutral, contradiction]
       facebook/bart-large-mnli                  -> [contradiction, neutral, entailment]
       roberta-large-mnli                        -> [CONTRADICTION, NEUTRAL, ENTAILMENT]
       cross-encoder/nli-deberta-v3-base         -> [contradiction, entailment, neutral]

   Swap two of those and ENTAILMENT reads as CONTRADICTION -- which in
   this system does not just mislabel, it drives a destructive state
   transition (ACTIVE -> CONTESTED plus a superseding entry). We now read
   `model.config.id2label` and canonicalize by name, and refuse to run at
   all against a model whose labels are uninterpretable (LABEL_0/1/2).

3. ABSTENTION. The old code took a bare argmax. A 0.34/0.33/0.33 softmax
   would contest a VERIFIED belief. Because CONTRADICTION is destructive
   and ENTAILMENT is merely additive, thresholds here are ASYMMETRIC:
   contradiction must clear a higher bar than entailment. Anything below
   its threshold falls back to NEUTRAL (recorded as `abstained=True`, so
   an abstention stays distinguishable from a genuine neutral in the
   Experiment 4.2 logs).

DEVICE NOTE: unlike BeliefStore (CPU-bound dict/numpy work -- see the
belief_gate.py docstring), this IS real tensor math and belongs on the
accelerator. Device is auto-selected via check_device.get_device().

Requires: torch, transformers. DeBERTa-v3 tokenizers additionally need
sentencepiece (+ protobuf for the slow->fast conversion). Install:

    pip install "transformers>=4.40" sentencepiece protobuf
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Protocol, Sequence

logger = logging.getLogger(__name__)

# NOTE: no torch/transformers import at module scope -- reconsolidation.py
# imports NLIResult from here and its docstring promises the pass logic is
# testable with numpy alone. Heavy deps are imported inside
# HFNLIBackend.__init__ so that promise survives.


# --------------------------------------------------------------------------
# Contract
# --------------------------------------------------------------------------

class NLIResult(Enum):
    ENTAILMENT = "ENTAILMENT"
    CONTRADICTION = "CONTRADICTION"
    NEUTRAL = "NEUTRAL"


@dataclass(frozen=True)
class NLIVerdict:
    """A single premise/hypothesis judgement.

    `label` is what the caller should act on (post-threshold).
    `argmax_label` is the model's raw pick before abstention was applied --
    keeping both is what makes "the model said CONTRADICTION but not
    confidently enough" visible in logs instead of indistinguishable from
    a true NEUTRAL.
    """
    label: NLIResult
    confidence: float
    scores: dict[NLIResult, float]
    argmax_label: NLIResult
    abstained: bool = False

    def __str__(self) -> str:
        tag = " (abstained)" if self.abstained else ""
        return f"{self.label.value} p={self.confidence:.3f}{tag}"


class NLIBackend(Protocol):
    """Structural type for anything usable as `nli_fn`.

    `__call__` keeps drop-in compatibility with the existing NLIFn
    signature. `classify_batch` is optional; run_reconsolidation_pass
    detects it and batches when present.
    """
    def __call__(self, premise: str, hypothesis: str) -> NLIResult: ...


# --------------------------------------------------------------------------
# Label canonicalization
# --------------------------------------------------------------------------

_LABEL_ALIASES: dict[str, NLIResult] = {
    "entailment": NLIResult.ENTAILMENT,
    "entail": NLIResult.ENTAILMENT,
    "entailed": NLIResult.ENTAILMENT,
    "contradiction": NLIResult.CONTRADICTION,
    "contradict": NLIResult.CONTRADICTION,
    "contradictory": NLIResult.CONTRADICTION,
    "neutral": NLIResult.NEUTRAL,
    "neutral_or_unrelated": NLIResult.NEUTRAL,
}


def canonical_label(raw: str) -> NLIResult:
    """Map a model's own label string onto NLIResult.

    Raises on anything uninterpretable rather than guessing. A wrong guess
    here inverts belief updates (see module docstring, point 2), so a hard
    failure at load time is strictly better than a plausible default.
    """
    key = raw.strip().lower().replace("-", "_").replace(" ", "_")
    if key in _LABEL_ALIASES:
        return _LABEL_ALIASES[key]
    raise ValueError(
        f"Cannot interpret NLI label {raw!r}. This usually means the model "
        f"config has generic labels (LABEL_0/LABEL_1/...) and the true label "
        f"order is unknown. Pass an explicit label_map=... to HFNLIBackend, "
        f"e.g. {{0: NLIResult.CONTRADICTION, 1: NLIResult.NEUTRAL, "
        f"2: NLIResult.ENTAILMENT}} -- check the model card for the order."
    )


def make_verdict(
    probs: Sequence[float],
    label_map: dict[int, NLIResult],
    entailment_threshold: float,
    contradiction_threshold: float,
) -> NLIVerdict:
    """Turn a softmax row into a thresholded verdict.

    Module-level and pure so the abstention policy -- the part most
    likely to be retuned -- is testable without downloading a model.

    NEUTRAL has no threshold: it is the fallback that abstention falls
    *into*, so gating it would leave nowhere to land.
    """
    scores = {label_map[i]: float(p) for i, p in enumerate(probs)}
    argmax = max(scores, key=lambda k: scores[k])
    confidence = scores[argmax]

    threshold = {
        NLIResult.CONTRADICTION: contradiction_threshold,
        NLIResult.ENTAILMENT: entailment_threshold,
        NLIResult.NEUTRAL: 0.0,
    }[argmax]

    if confidence < threshold:
        return NLIVerdict(
            label=NLIResult.NEUTRAL,
            confidence=confidence,
            scores=scores,
            argmax_label=argmax,
            abstained=True,
        )
    return NLIVerdict(label=argmax, confidence=confidence, scores=scores, argmax_label=argmax)


def _build_label_map(id2label: dict, override: Optional[dict] = None) -> dict[int, NLIResult]:
    if override:
        resolved = {int(k): v for k, v in override.items()}
        missing = set(id2label) - set(resolved)
        if missing:
            raise ValueError(f"label_map override is missing ids {sorted(missing)}")
        return resolved

    mapping = {int(idx): canonical_label(name) for idx, name in id2label.items()}

    covered = set(mapping.values())
    if covered != set(NLIResult):
        raise ValueError(
            f"Model exposes labels {sorted(l.value for l in covered)} but NLI "
            f"needs all three of ENTAILMENT/NEUTRAL/CONTRADICTION. Two-way "
            f"models (entailment / not_entailment) are not usable here: the "
            f"reconsolidation pass distinguishes contradiction from neutral, "
            f"and collapsing them would contest every unrelated claim."
        )
    return mapping


# --------------------------------------------------------------------------
# Known-good models
# --------------------------------------------------------------------------

#: Vetted defaults. `deps` flags tokenizers needing sentencepiece.
MODEL_REGISTRY: dict[str, dict] = {
    "deberta-v3-base": {
        "hf_id": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        "params": "184M",
        "deps": ("sentencepiece", "protobuf"),
        "note": "Default. Best accuracy/size tradeoff; strong on ANLI (adversarial).",
    },
    "deberta-v3-large": {
        "hf_id": "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        "params": "435M",
        "deps": ("sentencepiece", "protobuf"),
        "note": "Highest accuracy. ~3x slower; use when precision matters more than latency.",
    },
    "distilroberta": {
        "hf_id": "cross-encoder/nli-distilroberta-base",
        "params": "82M",
        "deps": (),
        "note": "Fastest, no sentencepiece needed. Noticeably weaker on negation.",
    },
    "bart-large": {
        "hf_id": "facebook/bart-large-mnli",
        "params": "407M",
        "deps": (),
        "note": "MNLI-only (no FEVER/ANLI). Widely used baseline.",
    },
}

DEFAULT_MODEL = MODEL_REGISTRY["deberta-v3-base"]["hf_id"]


def resolve_model_id(name: str) -> str:
    """Accept either a registry alias ('deberta-v3-base') or a raw HF id."""
    entry = MODEL_REGISTRY.get(name)
    return entry["hf_id"] if entry else name


# --------------------------------------------------------------------------
# HuggingFace backend
# --------------------------------------------------------------------------

class HFNLIBackend:
    """Batched, cached, threshold-aware cross-encoder NLI.

    Typical use:

        from nli_backend import HFNLIBackend
        nli = HFNLIBackend()                    # downloads on first use
        summary = run_reconsolidation_pass(store, claims, t_now, nli_fn=nli)

    The reconsolidation pass detects `classify_batch` and sends every
    flagged entry through the model in one padded batch rather than one
    forward pass per belief.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        batch_size: int = 16,
        max_length: int = 256,
        entailment_threshold: float = 0.50,
        contradiction_threshold: float = 0.70,
        cache_size: int = 4096,
        label_map: Optional[dict] = None,
        torch_dtype=None,
        warmup: bool = True,
    ):
        """
        entailment_threshold / contradiction_threshold: minimum softmax
            probability before the corresponding verdict is acted on.
            Contradiction defaults higher because it is the destructive
            branch -- it flips an entry to CONTESTED and forks a
            superseding entry, whereas entailment only nudges a weight by
            eta_reinforce. Asymmetric costs, asymmetric thresholds.

        cache_size: (premise, hypothesis) -> NLIVerdict LRU. Reconsolidation
            re-compares the same stored belief text across passes, so the
            hit rate is high in practice and each miss is a full forward
            pass.
        """
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "HFNLIBackend requires torch + transformers:\n"
                "    pip install \"transformers>=4.40\" sentencepiece protobuf"
            ) from e

        self._torch = torch
        self.model_id = resolve_model_id(model_name)
        self.batch_size = batch_size
        self.max_length = max_length
        self.entailment_threshold = entailment_threshold
        self.contradiction_threshold = contradiction_threshold

        self.device = self._resolve_device(device)

        # fp16 is a real speedup on CUDA and a known source of NaNs on MPS,
        # where several reduction kernels still fall back to fp32 anyway --
        # so autoselect only on CUDA and leave MPS/CPU in fp32.
        if torch_dtype is None:
            torch_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.dtype = torch_dtype

        logger.info("Loading NLI model %s onto %s (%s)", self.model_id, self.device, self.dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        # transformers renamed torch_dtype -> dtype in v5; the old name
        # still works but warns. Try the new spelling first so this is
        # quiet on v5 and correct on v4.
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_id, dtype=self.dtype
            )
        except TypeError:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_id, torch_dtype=self.dtype
            )
        self.model.eval()
        self.model.to(self.device)

        self.label_map = _build_label_map(self.model.config.id2label, label_map)
        logger.info("Label map resolved: %s", {k: v.value for k, v in self.label_map.items()})

        self._cache: OrderedDict[tuple[str, str], NLIVerdict] = OrderedDict()
        self._cache_size = cache_size
        self._lock = threading.Lock()
        self.stats = {"calls": 0, "cache_hits": 0, "forward_passes": 0, "abstentions": 0}

        if warmup:
            # First forward pass pays lazy-kernel-compile cost (especially on
            # MPS). Do it here so it doesn't land inside a timed pass.
            self.classify_batch([("warmup premise.", "warmup hypothesis.")])
            self._cache.clear()
            self.stats = {"calls": 0, "cache_hits": 0, "forward_passes": 0, "abstentions": 0}

    # -- device ------------------------------------------------------------

    def _resolve_device(self, device):
        torch = self._torch
        if device is not None:
            return torch.zeros(1, device=device).device
        try:
            from check_device import get_device
            return torch.zeros(1, device=get_device()).device
        except Exception:  # pragma: no cover - fallback path
            if torch.cuda.is_available():
                return torch.device("cuda:0")
            if torch.backends.mps.is_available():
                return torch.zeros(1, device="mps").device
            return torch.device("cpu")

    # -- inference ---------------------------------------------------------

    def classify_batch(self, pairs: Sequence[tuple[str, str]]) -> list[NLIVerdict]:
        """Classify many (premise, hypothesis) pairs in padded batches.

        Returns verdicts in the same order as `pairs`. Pairs already in the
        LRU are served from cache and never reach the model.
        """
        if not pairs:
            return []

        self.stats["calls"] += len(pairs)
        out: list[Optional[NLIVerdict]] = [None] * len(pairs)

        # 1. serve cache hits, collect the misses
        misses: list[int] = []
        with self._lock:
            for i, pair in enumerate(pairs):
                hit = self._cache.get(pair)
                if hit is not None:
                    self._cache.move_to_end(pair)
                    out[i] = hit
                    self.stats["cache_hits"] += 1
                else:
                    misses.append(i)

        if not misses:
            return [v for v in out]  # type: ignore[misc]

        # 2. length-sort the misses so each padded batch is roughly uniform.
        #    Padding is charged per batch at its longest member, so grouping
        #    similar lengths cuts wasted compute on mixed-length claim sets.
        order = sorted(misses, key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))

        for chunk_start in range(0, len(order), self.batch_size):
            chunk = order[chunk_start:chunk_start + self.batch_size]
            verdicts = self._forward([pairs[i] for i in chunk])
            for i, verdict in zip(chunk, verdicts):
                out[i] = verdict
                self._cache_put(pairs[i], verdict)

        return [v for v in out]  # type: ignore[misc]

    def _forward(self, batch: list[tuple[str, str]]) -> list[NLIVerdict]:
        torch = self._torch
        premises = [p for p, _ in batch]
        hypotheses = [h for _, h in batch]

        # THE FIX (module docstring, point 1): two positional arguments, so
        # the tokenizer builds a genuine sentence pair with its own [SEP]
        # and token_type_ids -- not one concatenated string.
        encoded = self.tokenizer(
            premises,
            hypotheses,
            padding=True,
            truncation="only_first",  # keep the hypothesis (the new claim) intact
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**encoded).logits
        # softmax in fp32 regardless of model dtype -- fp16 softmax over
        # three logits is numerically fine but the probabilities get
        # compared against thresholds, so keep full precision here.
        probs = torch.softmax(logits.float(), dim=-1).cpu()

        self.stats["forward_passes"] += 1
        return [self._to_verdict(row) for row in probs]

    def _to_verdict(self, prob_row) -> NLIVerdict:
        verdict = make_verdict(
            prob_row.tolist(),
            self.label_map,
            self.entailment_threshold,
            self.contradiction_threshold,
        )
        if verdict.abstained:
            self.stats["abstentions"] += 1
        return verdict

    def classify(self, premise: str, hypothesis: str) -> NLIVerdict:
        """Single pair, full verdict (scores + abstention flag)."""
        return self.classify_batch([(premise, hypothesis)])[0]

    def __call__(self, premise: str, hypothesis: str) -> NLIResult:
        """Drop-in NLIFn compatibility -- bare label, no scores."""
        return self.classify(premise, hypothesis).label

    # -- cache -------------------------------------------------------------

    def _cache_put(self, pair: tuple[str, str], verdict: NLIVerdict) -> None:
        with self._lock:
            self._cache[pair] = verdict
            self._cache.move_to_end(pair)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()

    def __repr__(self) -> str:
        return (
            f"HFNLIBackend(model={self.model_id!r}, device={self.device}, "
            f"dtype={self.dtype}, batch_size={self.batch_size}, "
            f"thresholds=(ent={self.entailment_threshold}, "
            f"con={self.contradiction_threshold}))"
        )


# --------------------------------------------------------------------------
# Test double
# --------------------------------------------------------------------------

class ScriptedNLI:
    """Returns pre-programmed verdicts. For testing the reconsolidation
    pass (including the batched path and abstention bookkeeping) without
    loading a model.

        nli = ScriptedNLI({("premise a", "claim a"): NLIResult.CONTRADICTION})
    """

    def __init__(self, table: dict[tuple[str, str], NLIResult],
                 default: NLIResult = NLIResult.NEUTRAL,
                 confidence: float = 0.95):
        self.table = table
        self.default = default
        self.confidence = confidence
        self.seen: list[tuple[str, str]] = []

    def classify_batch(self, pairs: Sequence[tuple[str, str]]) -> list[NLIVerdict]:
        self.seen.extend(pairs)
        out = []
        for pair in pairs:
            label = self.table.get(pair, self.default)
            out.append(NLIVerdict(
                label=label,
                confidence=self.confidence,
                scores={l: (self.confidence if l is label else (1 - self.confidence) / 2)
                        for l in NLIResult},
                argmax_label=label,
            ))
        return out

    def classify(self, premise: str, hypothesis: str) -> NLIVerdict:
        return self.classify_batch([(premise, hypothesis)])[0]

    def __call__(self, premise: str, hypothesis: str) -> NLIResult:
        return self.classify(premise, hypothesis).label
