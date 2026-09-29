"""
Run this locally (not in this sandbox) to confirm your device.
Requires: pip install torch

On Mac (Apple Silicon): should print mps
On NVIDIA machine:      should print cuda
Otherwise:               falls back to cpu
"""

import torch


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


if __name__ == "__main__":
    device = get_device()
    print(f"Selected device: {device}")

    if device.type == "mps":
        print(f"MPS built: {torch.backends.mps.is_built()}")
    elif device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")

    # sanity check: run a small op on the device
    x = torch.randn(1024, 1024, device=device)
    y = torch.randn(1024, 1024, device=device)
    z = x @ y
    print(f"Matmul sanity check ok, result device: {z.device}, shape: {tuple(z.shape)}")
