"""PyTorch autograd operators backed by the Alelyon Compute Kit (milestone 4a).

`AckLinear` is a drop-in for `torch.nn.Linear` whose three matmuls run on the
kit's BF16 cooperative-matrix kernels through the C ABI, with the rest of the
model on CPU tensors. It exists to prove the kernel chain under real autograd
(forward, input gradient, weight gradient) on the AKV model before the
`PrivateUse1` device backend exists; every call pays host-device copies, so it
measures correctness, not speed.

Layout mapping (no transposes anywhere, the kit reads operands as stored):

    y  = x @ W.T        x [T, K] as stored, W [N, K] as stored        -> NT
    dx = dy @ W         dy [T, N] as stored, W [N, K] as B [K'=N, N'=K] -> NN
    dW = dy.T @ x       dy [T, N] as stored A^T (A_T), x [T, K] as B  -> TN

Shapes the kernels cannot tile (any of T, N, K off the 128 grid, or K % 32)
fall back to torch on CPU and are counted in `AckLinear.fallbacks`, so a run
can say which layers the kit actually carried.
"""
from __future__ import annotations

import numpy as np
import torch

from . import ack

TILE = ack.TILE
K_MULTIPLE = ack.K_MULTIPLE


def _tileable(m: int, n: int, k: int) -> bool:
    """Whether the kit will take this product rather than the host fallback.

    The tile is no longer part of it: the edge variants carry any remainder, so
    only a zero extent falls back here.
    """
    return m > 0 and n > 0 and k > 0


def _to_bf16_np(t: torch.Tensor) -> np.ndarray:
    """A CPU tensor (any float dtype) to bf16 bit patterns, contiguous."""
    return ack.to_bf16(t.detach().to(torch.float32).contiguous().numpy())


class _Device:
    """Process-wide lazily opened device; None when the kit is unavailable."""

    instance: ack.Device | None = None
    tried = False

    @classmethod
    def get(cls) -> ack.Device | None:
        if not cls.tried:
            cls.tried = True
            try:
                cls.instance = ack.Device()
            except ack.AckUnavailable:
                cls.instance = None
        return cls.instance


class AckMatmulFunction(torch.autograd.Function):
    """y = x @ W.T with the kit; gradients through the kit's NN and TN layouts."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, counter) -> torch.Tensor:  # noqa: D401
        dev = _Device.get()
        t, k = x.shape
        n = weight.shape[0]
        ctx.save_for_backward(x, weight)
        ctx.counter = counter
        if dev is None or not _tileable(t, n, k):
            counter["fallback_forward"] += 1
            return x.to(torch.float32) @ weight.to(torch.float32).T
        counter["kit_forward"] += 1
        y = dev.matmul(_to_bf16_np(x), _to_bf16_np(weight), a_t=False, b_t=True)
        return torch.from_numpy(y)

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x, weight = ctx.saved_tensors
        counter = ctx.counter
        dev = _Device.get()
        t, k = x.shape
        n = weight.shape[0]
        dx = dw = None
        if dev is None or not _tileable(t, k, n) or not _tileable(n, k, t):
            counter["fallback_backward"] += 1
            dy32 = dy.to(torch.float32)
            if ctx.needs_input_grad[0]:
                dx = dy32 @ weight.to(torch.float32)
            if ctx.needs_input_grad[1]:
                dw = dy32.T @ x.to(torch.float32)
            return dx, dw, None
        counter["kit_backward"] += 1
        dy16 = _to_bf16_np(dy)
        if ctx.needs_input_grad[0]:
            # dx[T,K] = dy[T,N] . W[N,K]: B is W as stored (rows of K), K' = N
            dx = torch.from_numpy(dev.matmul(dy16, _to_bf16_np(weight), a_t=False, b_t=False))
        if ctx.needs_input_grad[1]:
            # dW[N,K] = dy.T[N,T] . x[T,K]: A is dy stored as T rows of N (A_T), B is x as stored
            dw = torch.from_numpy(dev.matmul(dy16, _to_bf16_np(x), a_t=True, b_t=False))
        return dx, dw, None


class AckLinear(torch.nn.Module):
    """`nn.Linear` whose matmuls run on the kit when the shape tiles.

    Weights live on CPU as float32 masters; the kit sees BF16 copies, the way
    the trainer's BF16 autocast path does. Bias is added on CPU in f32.
    """

    fallbacks: dict[str, int] = {"kit_forward": 0, "kit_backward": 0, "fallback_forward": 0, "fallback_backward": 0}

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.zeros(out_features)) if bias else None
        torch.nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    @classmethod
    def from_linear(cls, linear: torch.nn.Linear) -> "AckLinear":
        """Wrap a Linear SHARING its parameters, so a weight tied elsewhere (the
        AKV head is the embedding table) stays tied and its gradient still
        accumulates into the one parameter."""
        out = cls(linear.in_features, linear.out_features, bias=linear.bias is not None)
        out.weight = linear.weight
        out.bias = linear.bias
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lead = x.shape[:-1]
        flat = x.reshape(-1, x.shape[-1])
        y = AckMatmulFunction.apply(flat, self.weight, AckLinear.fallbacks)
        if self.bias is not None:
            y = y + self.bias.to(y.dtype)
        return y.reshape(*lead, self.out_features)

    @classmethod
    def reset_counters(cls) -> None:
        for key in cls.fallbacks:
            cls.fallbacks[key] = 0


def replace_linears(module: torch.nn.Module) -> int:
    """Swap every `nn.Linear` under `module` for an `AckLinear`; returns the count."""
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.Linear):
            setattr(module, name, AckLinear.from_linear(child))
            count += 1
        else:
            count += replace_linears(child)
    return count
