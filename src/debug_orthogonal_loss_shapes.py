"""Sanity check for orthogonal regularization shape alignment.

This mimics the relevant part of `UIETrainer.training_step()` that computes
orthogonal_loss between `lora_A` and `loranew_A` weights.

It validates we can handle:
- std: same input dim
- hyperbolic/hyperbolic_rot: loranew_A has +1 input dim (padded time-like coord)
- edge case: r_sum == 0 produces empty lora_A weights

Run from repo root:
    python -m src.debug_orthogonal_loss_shapes
"""

import torch


def _compute_orthogonal(param: torch.Tensor, param_new: torch.Tensor) -> torch.Tensor:
    # minimal reproduction of current logic in `uie_trainer_lora.py`
    if param.numel() == 0 or param_new.numel() == 0:
        return torch.tensor(0.0)

    a = param
    anew = param_new
    if anew.shape[1] == a.shape[1] + 1:
        anew = anew[:, 1:]
    elif a.shape[1] == anew.shape[1] + 1:
        a = a[:, 1:]

    if a.shape[1] != anew.shape[1]:
        return torch.tensor(0.0)

    return torch.abs(torch.mm(a, anew.T)).sum()


def main():
    torch.manual_seed(0)

    # std
    a = torch.randn(8, 1024)
    anew = torch.randn(4, 1024)
    v = _compute_orthogonal(a, anew)
    assert torch.isfinite(v)

    # hyperbolic (loranew has +1)
    a = torch.randn(8, 1024)
    anew = torch.randn(4, 1025)
    v = _compute_orthogonal(a, anew)
    assert torch.isfinite(v)

    # edge: r_sum == 0 => empty old lora_A
    a = torch.randn(0, 1024)
    anew = torch.randn(4, 1025)
    v = _compute_orthogonal(a, anew)
    assert v.item() == 0.0

    print("OK: orthogonal_loss shapes aligned")


if __name__ == "__main__":
    main()
