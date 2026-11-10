"""
GPU stage of the calibration study: run a language model on a TriviaQA
sample once and save everything the analysis needs. Run on Colab (T4):

    python collect_triviaqa.py --model openai-community/gpt2 --n 3000
    python collect_triviaqa.py --model Qwen/Qwen2.5-3B --n 3000

Writes results/triviaqa_<tag>.jsonl (one record per question) and
results/triviaqa_<tag>_keys.npz (pooled hidden-state keys), plus a
_meta.json. calibration_study.py then does all analysis on CPU from these
files, so thresholds, splits and plots can be revisited without a GPU.

PER QUESTION
  1. Closed-book answer: greedy decoding after a 5-shot prompt
     (triviaqa_data.FEW_SHOT), stopped at the first newline. Confidence
     is the probability of the generated answer (product of its token
     probabilities) and of its first token. Scored by exact match against
     TriviaQA's aliases after official normalisation.
  2. Keys at layer L (default: 2/3 of the depth -- GPT-2's 8 of 12, the
     layer used all project): mean hidden state over the question's
     tokens, and over two belief texts -- the question with its true
     answer, and with another question's answer (for a poisoned-store
     ablation). A fixed "Q:" prefix is prepended and excluded from the
     mean, so the first position (an attention sink with a huge norm in
     both GPT-2 and Qwen) never enters a key.
  3. Loop check (first 32 questions): the real EpTLoop is run in context
     mode on the question's per-token hidden states, and its last-token
     match cosine is compared to the pooled-key cosine the analysis uses.
     They are the same quantity -- the context query at the last token is
     the running mean of (z - center), i.e. the pooled key -- and this
     check verifies it, so the offline analysis provably describes what
     the loop would do.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from triviaqa_data import belief_text, is_correct, is_correct_lenient, load_sample, prompt

PREFIX = "Q:"


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class LM:
    def __init__(self, name: str, layer: int | None, dtype: str, device):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device
        self.tok = AutoTokenizer.from_pretrained(name)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        dt = {"float16": torch.float16, "float32": torch.float32}[dtype]
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=dt).to(device).eval()
        if device.type != "cuda":
            # lm_hook.py Fix 10: owned copies, not mmap'd weights (SIGBUS on macOS)
            for p in self.model.parameters():
                p.data = p.data.clone()
        n_layers = self.model.config.num_hidden_layers
        self.layer = round(2 * n_layers / 3) if layer is None else layer
        if not 0 < self.layer < n_layers:
            raise ValueError(f"layer {self.layer} out of range for {n_layers} layers")
        self.d_model = self.model.config.hidden_size
        bos = self.tok.bos_token_id
        self.lead = ([bos] if bos is not None else []) + self.tok.encode(PREFIX, add_special_tokens=False)

    # -- keys -------------------------------------------------------------

    def _ids(self, text: str) -> list[int]:
        return self.lead + self.tok.encode(" " + text.strip(), add_special_tokens=False)

    @torch.no_grad()
    def token_states(self, text: str) -> torch.Tensor:
        """(S, d) float32 hidden states entering block `layer`, prefix dropped."""
        ids = torch.tensor([self._ids(text)], device=self.device)
        h = self.model(ids, output_hidden_states=True).hidden_states[self.layer][0]
        self._check(h)
        return h[len(self.lead):].float()

    @torch.no_grad()
    def pooled(self, texts: list[str], bs: int) -> np.ndarray:
        """(n, d) mean hidden state over each text's non-prefix tokens.
        Right padding: the model is causal, so pads cannot affect real
        positions, and no position-id shift is introduced."""
        out = np.zeros((len(texts), self.d_model), dtype=np.float32)
        L = len(self.lead)
        for s in range(0, len(texts), bs):
            seqs = [self._ids(t) for t in texts[s:s + bs]]
            m = max(map(len, seqs))
            ids = torch.full((len(seqs), m), self.tok.pad_token_id, device=self.device)
            mask = torch.zeros((len(seqs), m), dtype=torch.long, device=self.device)
            for i, q in enumerate(seqs):
                ids[i, :len(q)] = torch.tensor(q, device=self.device)
                mask[i, :len(q)] = 1
            h = self.model(ids, attention_mask=mask, output_hidden_states=True).hidden_states[self.layer]
            keep = mask.clone()
            keep[:, :L] = 0
            self._check(h[keep.bool()])
            h = h.float() * keep[..., None]
            out[s:s + len(seqs)] = (h.sum(1) / keep.sum(1, keepdim=True)).cpu().numpy()
        return out

    @staticmethod
    def _check(h):
        if not torch.isfinite(h).all():
            raise FloatingPointError(
                "non-finite hidden states -- fp16 overflow on this model; rerun with --dtype float32")

    # -- answers ----------------------------------------------------------

    @torch.no_grad()
    def answer(self, questions: list[str], bs: int, max_new: int = 16) -> list[dict]:
        self.tok.padding_side = "left"
        res = []
        for s in range(0, len(questions), bs):
            batch = [prompt(q) for q in questions[s:s + bs]]
            enc = self.tok(batch, return_tensors="pt", padding=True).to(self.device)
            gen = self.model.generate(
                **enc, max_new_tokens=max_new, do_sample=False,
                output_scores=True, return_dict_in_generate=True,
                pad_token_id=self.tok.pad_token_id)
            new = gen.sequences[:, enc["input_ids"].shape[1]:]
            logp = torch.stack(gen.scores, 1).float().log_softmax(-1)  # (b, T, V)
            tok_lp = logp.gather(-1, new[..., None])[..., 0].cpu().numpy()
            stop_ids = {self.tok.eos_token_id, self.tok.pad_token_id}
            for i in range(len(batch)):
                # answer = generated text up to the first newline (or EOS);
                # its confidence covers exactly the tokens that produced it
                text, lps = "", []
                for t, tid in enumerate(new[i].tolist()):
                    if tid in stop_ids:
                        break
                    piece = self.tok.decode([tid])
                    head = piece.split("\n")[0]
                    if head.strip():
                        lps.append(float(tok_lp[i, t]))
                    text += head
                    if "\n" in piece:
                        break
                ans = text.strip()
                res.append({"pred": ans, "n_tok": len(lps),
                            "conf_seq": float(np.exp(sum(lps))) if lps and ans else 0.0,
                            "conf_first": float(np.exp(lps[0])) if lps and ans else 0.0})
        self.tok.padding_side = "right"
        return res


def loop_check(lm: LM, items, q_keys, b_keys, n: int = 32) -> dict:
    """Run EpTLoop (context retrieval, residual keys) on per-token states
    and compare its last-token match cosine to the pooled-key cosine."""
    from belief_gate import ResidualKeyEncoder
    from belief_store import ContentType, SourceTier
    from ept_loop import EpTLoop

    center = torch.from_numpy(b_keys.mean(0))
    enc = ResidualKeyEncoder(lm.d_model, center=center)
    loop = EpTLoop(d_model=lm.d_model, dict_size=lm.d_model, encoder=enc,
                   retrieval="context", theta_rel=1e-6, min_ctx=1)
    for k, it in zip(b_keys, items):
        loop.add_belief(k - center.numpy(), belief_text(it["question"], it["value"]), weight=0.9,
                        source_tier=SourceTier.VERIFIED, content_type=ContentType.ARCHITECTURE, t_now=0)
    K = b_keys - center.numpy()
    K /= np.linalg.norm(K, axis=1, keepdims=True)
    diffs = []
    for i in range(min(n, len(items))):
        z = lm.token_states(items[i]["question"]).to(loop.device)[None]
        out = loop.step(z, t_now=1)
        loop.reset_generation()
        q = q_keys[i] - center.numpy()
        pooled_sim = float((K @ (q / np.linalg.norm(q))).max())
        diffs.append(abs(float(out["match_sim"][0, -1]) - pooled_sim))
    return {"n": len(diffs), "max_abs_diff": float(max(diffs))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai-community/gpt2")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--out", default="results")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    device = pick_device()
    dtype = args.dtype if device.type == "cuda" else "float32"
    tag = args.tag or args.model.split("/")[-1].lower()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    items = load_sample(args.n, args.seed)
    print(f"{len(items)} questions; loading {args.model} ({dtype}) on {device}", flush=True)
    lm = LM(args.model, args.layer, dtype, device)
    print(f"layer {lm.layer}, d_model {lm.d_model}", flush=True)

    answers = lm.answer([it["question"] for it in items], args.bs)
    print(f"answers done ({time.time() - t0:.0f}s)", flush=True)

    # a poisoned belief takes another question's answer (fixed derangement)
    perm = np.random.default_rng(args.seed + 1).permutation(len(items))
    perm = np.where(perm == np.arange(len(items)), np.roll(perm, 1), perm)
    wrong = [items[j]["value"] for j in perm]

    q_keys = lm.pooled([it["question"] for it in items], args.bs * 2)
    b_true = lm.pooled([belief_text(it["question"], it["value"]) for it in items], args.bs * 2)
    b_wrong = lm.pooled([belief_text(it["question"], w) for it, w in zip(items, wrong)], args.bs * 2)
    print(f"keys done ({time.time() - t0:.0f}s)", flush=True)

    check = loop_check(lm, items, q_keys, b_true)
    print(f"loop check: max |loop cos - pooled cos| over {check['n']} questions = "
          f"{check['max_abs_diff']:.2e}", flush=True)

    with open(f"{args.out}/triviaqa_{tag}.jsonl", "w") as f:
        for it, a, w in zip(items, answers, wrong):
            f.write(json.dumps({**it, **a, "wrong_value": w,
                                "correct": is_correct(a["pred"], it["aliases"]),
                                "correct_lenient": is_correct_lenient(a["pred"], it["aliases"])}) + "\n")
    np.savez_compressed(f"{args.out}/triviaqa_{tag}_keys.npz", q=q_keys.astype(np.float16),
                        b_true=b_true.astype(np.float16), b_wrong=b_wrong.astype(np.float16))
    import transformers
    acc = float(np.mean([is_correct(a["pred"], it["aliases"]) for it, a in zip(items, answers)]))
    meta = {"model": args.model, "layer": lm.layer, "d_model": lm.d_model, "dtype": dtype,
            "device": str(device), "n": len(items), "seed": args.seed, "accuracy_em": acc,
            "loop_check": check, "seconds": round(time.time() - t0, 1),
            "torch": torch.__version__, "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None}
    with open(f"{args.out}/triviaqa_{tag}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
