"""Muon optimizer used as an optional transformer optimizer ablation."""

from __future__ import annotations

from paper2._torch import Tensor, require_torch, torch


def orthogonalise_via_newtonschulz5(
    grad: Tensor,
    steps: int = 5,
    eps: float = 1e-7,
) -> Tensor:
    require_torch()
    if grad.ndim < 2:
        raise ValueError("Muon requires matrix-like gradients with ndim >= 2")
    a, b, c = 3.4445, -4.7750, 2.0315
    x = grad.bfloat16()
    x = x / (x.norm(dim=(-1, -2), keepdim=True) + eps)
    transposed = grad.size(-2) > grad.size(-1)
    if transposed:
        x = x.mT
    for _ in range(steps):
        gram = x @ x.mT
        x = a * x + b * gram @ x + c * gram @ gram @ x
    if transposed:
        x = x.mT
    return x


class Muon(torch.optim.Optimizer):
    """Minimal Muon optimizer for hidden matrix weights.

    This follows the previous transformer training code in
    `src/transformer/train/muon.py`, but keeps it local to Paper 2 so the new
    training runner does not depend on the older reconstruction package layout.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
    ) -> None:
        require_torch()
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires positive momentum.")
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                state = self.state[param]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                momentum_buffer = state["momentum_buffer"]
                momentum_buffer.lerp_(grad, 1.0 - group["momentum"])
                grad = (
                    grad.lerp_(momentum_buffer, group["momentum"])
                    if group["nesterov"]
                    else momentum_buffer
                )
                param.mul_(1.0 - group["lr"] * group["weight_decay"])
                grad = orthogonalise_via_newtonschulz5(grad, steps=group["ns_steps"])
                scale = max(1.0, (param.size(-2) / param.size(-1)) ** 0.5)
                param.sub_(grad, alpha=group["lr"] * scale)
        return loss
