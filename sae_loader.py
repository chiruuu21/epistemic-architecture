"""
Pretrained SAE weights for the Belief Gate's concept encoder.

Until now TopKSAEEncoder held torch.randn weights, so a belief's concept
key was a random projection of the activation: two sentences about the
same thing landed no closer together than two unrelated ones, and
"retrieve the relevant belief" was structurally correct but semantically
empty. This module fills the same three tensors (W_enc, b_enc, b_pre)
from an SAE trained on GPT-2 small's residual stream.

DEFAULT SAE
    jbloom/GPT2-Small-SAEs-Reformatted, blocks.8.hook_resid_pre
    d_in=768, d_sae=24576, 300M OpenWebText tokens. In HuggingFace GPT-2
    terms that is hidden_states[8] == the input to transformer.h[8],
    which is exactly where lm_hook.GPT2Activations attaches.

THREE THINGS THE MAPPING HAS TO GET RIGHT, all of which fail silently
(a wrong SAE still produces sparse, plausible-looking codes):

1. b_pre IS THE SAE's b_dec.
   This SAE encodes as relu((x - b_dec) @ W_enc + b_enc): the decoder
   bias is subtracted before encoding. Our encoder calls that tensor
   b_pre. Leaving it at zero shifts every pre-activation by a constant
   and quietly changes which features win the top-k.

2. W_enc ORIENTATION.
   Ours is (d_model, dict_size). A transposed matrix of the same total
   size would still multiply if d_model == dict_size, so shapes are
   checked against cfg.json explicitly, not inferred.

3. THE HOOK POINT MUST MATCH THE MODEL LAYER.
   An SAE trained on layer 8 applied to layer 6 activations gives codes
   that look sparse and normal and mean nothing. The hook point is kept
   on the returned encoder (encoder.hook_point) so lm_hook can read the
   layer from it instead of the caller repeating it.

Anything the loader cannot verify, it refuses -- same policy as
nli_backend's label map check.

ACTIVATION. This SAE was trained with ReLU + L1, not TopK. The paper's
encoder is TopK(...); we keep TopK on top of the ReLU pre-activations,
which keeps the strongest k features and drops the weak tail. With
k=None the encoder reproduces the SAE's native ReLU codes exactly.
"""

from __future__ import annotations

import json

import torch

from belief_gate import TopKSAEEncoder

DEFAULT_REPO = "jbloom/GPT2-Small-SAEs-Reformatted"
DEFAULT_HOOK = "blocks.8.hook_resid_pre"

REQUIRED_KEYS = ("W_enc", "b_enc", "b_dec")


def hook_layer(hook_point: str) -> int:
    """'blocks.8.hook_resid_pre' -> 8. Only resid_pre is supported: it is
    the one hook point that maps to a single HF tensor (the input to
    transformer.h[layer]) with no ambiguity about LayerNorm placement."""
    parts = hook_point.split(".")
    if len(parts) != 3 or parts[0] != "blocks" or parts[2] != "hook_resid_pre":
        raise ValueError(f"unsupported hook point {hook_point!r}: expected blocks.<n>.hook_resid_pre")
    return int(parts[1])


def load_sae_tensors(repo_id: str = DEFAULT_REPO, hook_point: str = DEFAULT_HOOK):
    """Download (or read from the HF cache) and validate. Returns
    (cfg, tensors). tensors includes W_dec, which the encoder does not
    need but reconstruction checks do."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    cfg_path = hf_hub_download(repo_id, f"{hook_point}/cfg.json")
    weights_path = hf_hub_download(repo_id, f"{hook_point}/sae_weights.safetensors")
    with open(cfg_path) as f:
        cfg = json.load(f)
    tensors = load_file(weights_path)
    validate(cfg, tensors, hook_point)
    return cfg, tensors


def validate(cfg: dict, tensors: dict, hook_point: str) -> None:
    """Refuse anything whose layout we cannot confirm. Split out from the
    download so the tests can exercise it on synthetic files."""
    missing = [k for k in REQUIRED_KEYS if k not in tensors]
    if missing:
        raise ValueError(f"SAE weights missing {missing}; found {sorted(tensors)}")

    if cfg.get("hook_point") != hook_point:
        raise ValueError(f"cfg hook_point {cfg.get('hook_point')!r} != requested {hook_point!r}")

    d_in, d_sae = cfg["d_in"], cfg["d_sae"]
    expected = {"W_enc": (d_in, d_sae), "b_enc": (d_sae,), "b_dec": (d_in,)}
    if "W_dec" in tensors:
        expected["W_dec"] = (d_sae, d_in)
    for name, shape in expected.items():
        got = tuple(tensors[name].shape)
        if got != shape:
            raise ValueError(f"{name} has shape {got}, cfg implies {shape}")

    for name, t in tensors.items():
        if not torch.isfinite(t).all():
            raise ValueError(f"{name} contains non-finite values")


def encoder_from_tensors(cfg: dict, tensors: dict, hook_point: str,
                         k: int | None = 32) -> TopKSAEEncoder:
    d_in, d_sae = cfg["d_in"], cfg["d_sae"]
    enc = TopKSAEEncoder(d_model=d_in, dict_size=d_sae, k=k if k is not None else d_sae)
    with torch.no_grad():
        enc.W_enc.copy_(tensors["W_enc"].float())
        enc.b_enc.copy_(tensors["b_enc"].float())
        enc.b_pre.copy_(tensors["b_dec"].float())  # (1) from the module docstring
    enc.hook_point = hook_point
    enc.source = f"{cfg.get('model_name', '?')}:{hook_point}"
    return enc


def load_pretrained_sae(repo_id: str = DEFAULT_REPO, hook_point: str = DEFAULT_HOOK,
                        k: int | None = 32, device: str | torch.device = "cpu") -> TopKSAEEncoder:
    """The one call most code needs:

        encoder = load_pretrained_sae()
        loop = EpTLoop(d_model=768, dict_size=24576, encoder=encoder, ...)

    k=32 matches the paper's TopK; k=None keeps every active ReLU feature.
    """
    hook_layer(hook_point)  # fail on an unsupported hook before downloading
    cfg, tensors = load_sae_tensors(repo_id, hook_point)
    tensors.pop("W_dec", None)  # ~75 MB the encoder never reads
    return encoder_from_tensors(cfg, tensors, hook_point, k=k).to(device)
