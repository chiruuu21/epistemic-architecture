"""
Attach EpTLoop to a real language model's hidden states.

Until now every z_attn handed to EpTLoop.step() was torch.randn. This
module runs GPT-2 small and feeds the loop the actual residual stream at
the layer the SAE was trained on (sae_loader.DEFAULT_HOOK, layer 8).

    lm = GPT2Activations(layer=8)
    key = lm.concept_key(encoder, "The belief store keeps entries on the CPU.")
    with lm.attached(loop):
        text = lm.generate("The belief store", max_new_tokens=20)

TWO MODES
    observe (default): the hook reads the residual stream, runs it
        through loop.step(), and leaves the model's computation untouched.
        Generation is identical to plain GPT-2; we are measuring what the
        gates *would* do.
    gate=True: the hook writes loop.step()'s z_out back into the
        residual stream, so belief confidence actually changes the
        model's output. The gates were never trained against this model,
        so this mode is for measuring the effect, not for producing good
        text.

RESIDUALS ARE MEAN-CENTERED BEFORE THE LOOP SEES THEM.
The SAE was trained on TransformerLens activations, and TransformerLens
loads GPT-2 with center_writing_weights=True: every weight that writes to
the residual stream has its mean over d_model removed, so TL residual
vectors have zero mean across their 768 dims. HuggingFace's do not. Since
the residual stream is a sum of those writes, x_TL == x_HF - mean(x_HF)
exactly. Skipping this does not crash -- the SAE still returns sparse
codes -- it just reconstructs worse than predicting the mean (FVU ~11
measured) with ~500 features firing per token instead of tens. In gate
mode the removed mean is added back when z_out is written into the
stream, so the model only sees the change the gates made.

POSITION 0 IS EXCLUDED, BOTH FOR KEYS AND FOR THE LOOP.
GPT-2's first position (we prepend <|endoftext|> as BOS, as the SAE's
training did) carries a residual norm ~29x the others and fires the same
handful of SAE features regardless of the text. Left in, it dominates
every mean-pooled concept key and makes all sentences look alike. On the
prefill pass the hook passes positions 1: to the loop; incremental
decoding steps (one new token each, via the KV cache) have no BOS and
pass through whole.
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import torch

DEFAULT_MODEL = "openai-community/gpt2"


class GPT2Activations:
    def __init__(self, model_name: str = DEFAULT_MODEL, layer: int = 8,
                 device: str | torch.device = "cpu"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = torch.zeros(1, device=device).device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float32).to(self.device).eval()
        # Copy weights out of the memory-mapped safetensors file. On this
        # machine (torch 2.13 / Accelerate, macOS 14.1) a single-row matmul
        # against the mmap'd tied embedding -- exactly what every decode
        # step's lm_head does -- dies with SIGBUS; the same multiply on an
        # owned copy is fine. parameters() yields the tied wte/lm_head
        # Parameter once, so the tie survives.
        for p in self.model.parameters():
            p.data = p.data.clone()
        self.layer = layer
        self.d_model = self.model.config.n_embd
        n_layer = self.model.config.n_layer
        if not 0 <= layer < n_layer:
            raise ValueError(f"layer {layer} out of range for {n_layer}-layer model")
        self.bos_id = self.tokenizer.bos_token_id

        self._loop = None
        self._gate = False
        self._t = 0
        self._prefill = True

    # -- reading activations ------------------------------------------------

    def encode_ids(self, text: str) -> torch.Tensor:
        ids = [self.bos_id] + self.tokenizer.encode(text)
        return torch.tensor([ids], device=self.device)

    @staticmethod
    def center(x: torch.Tensor) -> torch.Tensor:
        """HF residual -> TransformerLens basis the SAE was trained in."""
        return x - x.mean(dim=-1, keepdim=True)

    @torch.no_grad()
    def resid(self, text: str, keep_bos: bool = False) -> torch.Tensor:
        """(S, d_model) centered residual stream entering block `layer`,
        BOS dropped unless keep_bos. hidden_states[i] is the input to
        transformer.h[i]: index 0 is the embeddings, so hidden_states[8]
        == blocks.8.hook_resid_pre."""
        out = self.model(self.encode_ids(text), output_hidden_states=True)
        h = self.center(out.hidden_states[self.layer][0])
        return h if keep_bos else h[1:]

    @torch.no_grad()
    def concept_key(self, encoder, text: str) -> np.ndarray:
        """Sentence-level concept key: mean of per-token SAE codes."""
        c = encoder(self.resid(text).to(encoder.W_enc.device))
        return c.mean(dim=0).cpu().numpy()

    # -- attaching the loop -------------------------------------------------

    def _hook(self, module, args, kwargs):
        hidden = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
        start = 1 if self._prefill else 0
        if self._prefill:
            # a new prompt: context-mode retrieval must not pool codes
            # across prompts (no-op in token mode)
            self._loop.belief_gate.reset_context()
        self._prefill = False

        raw = hidden[:, start:]
        if raw.shape[1] == 0:
            return None
        mean = raw.mean(dim=-1, keepdim=True)
        out = self._loop.step((raw - mean).to(self._loop.device), t_now=self._t)
        self._t += 1
        self.last_step = out

        if not self._gate:
            return None
        z_out = out["z_out"].to(hidden.device, hidden.dtype) + mean
        new_hidden = torch.cat([hidden[:, :start], z_out], dim=1)
        if "hidden_states" in kwargs:
            kwargs = {**kwargs, "hidden_states": new_hidden}
            return args, kwargs
        return (new_hidden, *args[1:]), kwargs

    @contextmanager
    def attached(self, loop, gate: bool = False, t_start: int = 0):
        if loop.belief_gate.encoder.W_enc.shape[0] != self.d_model:
            raise ValueError("loop's encoder width does not match this model")
        self._loop, self._gate, self._t, self._prefill = loop, gate, t_start, True
        self.last_step = None
        handle = self.model.transformer.h[self.layer].register_forward_pre_hook(
            self._hook, with_kwargs=True)
        try:
            yield self
        finally:
            handle.remove()
            self._loop = None

    @property
    def t_now(self) -> int:
        return self._t

    @torch.no_grad()
    def observe(self, text: str) -> dict:
        """One forward pass over text inside attached(); returns that
        pass's loop.step() output. A new prompt, so context is reset."""
        self._prefill = True
        self.model(self.encode_ids(text))
        return self.last_step

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 20) -> str:
        """Greedy continuation. Inside attached(), every forward pass goes
        through the loop: one prefill pass, then one per new token."""
        self._prefill = True
        ids = self.encode_ids(prompt)
        out = self.model.generate(
            ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
