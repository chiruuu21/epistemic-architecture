# EpstLoop

An implementation and test of the **EpistemicTransformer (EpT)** proposal
("Closing the Epistemic Loop"). EpT adds three mechanisms to a language
model:

1. a **Belief Store**: facts with trust tiers, decay and state, gated into the model's activations;
2. a **Feeling-of-Knowing (FOK) gate**: confidence decides whether the model answers from the store or falls back;
3. **selective reconsolidation**: a retrieved belief is checked against new evidence and strengthened, contested or replaced, while competing beliefs are suppressed (RIF).

The models stay frozen. The loop reads (and can optionally write) the
residual stream of GPT-2 small (layer 8) or Qwen2.5-3B (layer 24).

## Status against the paper

| Paper | Status |
|---|---|
| §3.1 Belief Store | Built. Paper fixes: CONTESTED is penalised instead of zeroed, RIF has a floor, decay units are days |
| §3.3 Belief Gate | Retrieval works with residual keys (SAE keys are worse, see Results). Gates are untrained |
| §3.4 FOK gate | Trigger built. Confidence revised to **store + model** (`calibration.CombinedPIK`). The Quiet-STaR thought step is a placeholder |
| §3.5 Reconsolidation + RIF | Built with real DeBERTa-v3 NLI, plus revisions (below) |
| §4.2 Belief convergence | **Done**: [`exp_convergence.py`](exp_convergence.py) |
| §4.3 FOK calibration | **Done** on TriviaQA: [`calibration_study.py`](calibration_study.py) |
| §4.1 Context poisoning, §4.4 Drift, §4.5 full ablation | Not started. §4.1 needs trained gates |

## Results

**§4.3 Calibration** (TriviaQA, 1,500 held-out questions per model;
[results/calibration_*.txt](results/))

| Confidence | GPT-2 small ECE (AUROC) | Qwen2.5-3B ECE (AUROC) |
|---|---|---|
| Paper's binary P(IK) | 0.044 (0.90) | 0.322 (0.80) |
| Store-only, graded | 0.030 (0.98) | 0.085 (0.82) |
| **Store + model, per path** | **0.021 (0.99)** | **0.012 (0.95)** |

The target was ECE < 0.05. It is only met once P(IK) also takes the
model's own answer probability, because the store cannot see what the
model already knows from its weights.

**§4.2 Convergence** (20 contradicted beliefs, 10 consistent beliefs, 10
unrelated contexts, 3 turns; [results/exp_convergence.txt](results/exp_convergence.txt))

| Config | Switch to correction (t3)\* | Successors per belief | False contests |
|---|---|---|---|
| Paper as written | **0 / 12** | 3.0 (duplicates) | 3 |
| + trust-aware retrieval | 11 / 12 | 1.0 | 8 |
| Revised (pre-planned) | 11 / 12 | 1.0 | 8 |
| Revised + lineage ranking + RIF floor + contest gate *(post hoc)* | **12 / 12** | 1.0 | **1** |

\* Of the 12 topics whose probe question reaches the belief at all.

- **As written, the paper's loop never converges.** A correction is stored at the same concept address as the belief it contests. Cosine retrieval then ties, and the old belief keeps winning. Trust-aware retrieval is the fix that matters most.
- **NLI cannot judge whether two sentences are on the same topic.** DeBERTa-v3-base labels about 4% of completely unrelated pairs CONTRADICTION. Key cosine separates same-topic from cross-topic evidence at AUROC 0.99, so retrieval has to vouch for the topic before NLI may contest anything (`contest_min_sim`).
- **RIF taxes bystanders.** In this key space every belief's 5 nearest neighbours sit near cosine 0.6, so the paper's RIF lowers unrelated beliefs on every retrieval. A competitor floor (`rif_min_sim`) limits it to real competitors.
- The post-hoc settings were chosen **on this fact set**. They need confirming on a fresh one.

**Retrieval** (GPT-2 layer 8, 50 paraphrase queries): residual−b_dec keys
get 0.78 top-1, versus 0.50 for pretrained SAE codes.

## Revisions to the paper

All of them are options. The defaults reproduce the paper, so every
revision can be ablated.

| Option | Where | What it changes |
|---|---|---|
| `rank="trust"` / `"lineage"` | `BeliefStore`, `BeliefGate`, `EpTLoop` | Retrieval prefers the higher-weff version of a belief; `lineage` never crosses topics |
| `claim_tier` | `finish_generation` | A correction inherits the evidence's tier (the paper always uses INFERRED) |
| `eta_contest` | `EpTLoop` | Contradiction also weakens the contested belief |
| `dedup_successors` | `EpTLoop` | Repeated contradiction reinforces the existing correction instead of duplicating it |
| `supersede` | `EpTLoop` | CONTESTED → SUPERSEDED once the correction is confirmed and outweighs it (the paper never enters this state) |
| `rif_min_sim` | `EpTLoop` | RIF only inhibits neighbours above this similarity |
| `contest_min_sim` | `EpTLoop` | A contradiction only contests when the retrieval match is this strong |
| `pik_model` + `answer_confidence()` | `EpTLoop` | Calibrated store + model P(IK) |
| `confidence="graded"` | `BeliefGate` | P(IK) rises smoothly with match quality |

## Running

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python run_all_tests.py            # 83 offline checks, ~4 s, no downloads
python exp_convergence.py          # §4.2, ~30 s on a laptop CPU (GPT-2 + DeBERTa)
python calibration_study.py --tag qwen2.5-3b   # §4.3, CPU, from saved keys
python smoke_test_lm_loop.py       # full loop on real GPT-2 hidden states
```

`collect_triviaqa.py` is the only GPU step. It runs on a Colab T4 through
[`EpstLoop_colab.ipynb`](EpstLoop_colab.ipynb) and saves per-question
records and hidden-state keys to `results/`. Everything else runs on a
laptop CPU.

## Layout

| File | Role |
|---|---|
| `belief_store.py` | Entries, trust, decay, states, retrieval ranking, RIF |
| `belief_gate.py` | Concept keys (SAE or residual), token/context retrieval, gating |
| `fok_gate.py` | FOK trigger and placeholder thought step |
| `reconsolidation.py` | NLI-driven update, revisions, RIF call |
| `ept_loop.py` | Driver: per-pass gating, post-generation reconsolidation, P(IK) |
| `nli_backend.py` | DeBERTa NLI with label-map checks and asymmetric abstention |
| `claim_extraction.py` | Heuristic generation → per-belief claims |
| `lm_hook.py`, `sae_loader.py` | GPT-2 residual hooks; pretrained SAE / b_dec centre |
| `calibration.py`, `calibration_study.py` | ECE/AUROC/Platt, `CombinedPIK`, §4.3 |
| `convergence_data.py`, `exp_convergence.py` | §4.2 fact set and experiment |
| `analysis_*.py` | Retrieval and FOK analyses from Reports 3–4 |

## Known limitations

- Gates are untrained. Gate mode changes the model's output, but not usefully yet.
- The thought step is an MLP placeholder, not Quiet-STaR.
- On TriviaQA each stored belief contains its question's wording, so retrieval is easy (top-1 ≈ 0.95). Reworded queries are still to be tested.
- In §4.2, probe questions reach only 12 of the 20 topics at θ_rel 0.608. Paraphrased evidence is often judged NEUTRAL by NLI, so corrections are reinforced about half the time.
