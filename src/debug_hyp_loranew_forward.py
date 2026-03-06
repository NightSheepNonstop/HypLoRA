"""Minimal forward sanity check for O-LoRA hyperbolic loranew_ branch.

This script doesn't load any HF models. It only instantiates the LoRA Linear module
and runs a forward pass for two configurations:
- std (legacy behavior)
- hyperbolic / hyperbolic_rot for the *loranew_* branch

Run from repo root:
    python -m src.debug_hyp_loranew_forward

Note: do NOT run as `python -m debug_hyp_loranew_forward.py`.
"""

import torch

from peft.tuners.lora import Linear


def _run_case(lora_type: str):
    torch.manual_seed(0)
    layer = Linear(
        adapter_name="default",
        in_features=16,
        out_features=8,
        r=4,
        r_sum=4,
        lora_alpha=8,
        lora_dropout=0.0,
        lora_type=lora_type,
        bias=False,
    )
    x = torch.randn(2, 16)
    y = layer(x)
    assert y.shape == (2, 8)
    assert torch.isfinite(y).all(), f"Non-finite output for lora_type={lora_type}"
    return y


def main():
    _run_case("std")
    _run_case("hyperbolic-1.0")
    _run_case("hyperbolic_rot-1.0")
    print("OK: forward passed for std / hyperbolic / hyperbolic_rot")


if __name__ == "__main__":
    main()
