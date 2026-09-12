"""A small, explicit Vulkan harness for running and updating VQ experts.

The harness is intentionally model-agnostic at its boundary: a caller supplies
one expert's input activations and targets as finite float32 NumPy arrays. The
expert weights, momentum and variance are read from an :class:`ExpertBank`;
packed matmul and the AdamW state update run through :class:`VQDevice`.

This is a linear-expert adapter, not a complete transformer runtime. Model
authors can use the same ``forward`` and ``train_step`` boundary from their
own router/graph and keep routing, tokenisation and checkpoint policy in their
model code. Codebook calibration and diagnostic reductions are CPU work; the
packed matmul and optimizer update are Vulkan work.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np

from . import ack, expert_bank, vq


class HarnessError(ValueError):
    """The supplied model state, arrays or job boundary is not admissible."""


@dataclass(frozen=True)
class ExpertStep:
    """Durable evidence for one successful expert update."""

    layer: int
    expert: int
    optimizer_step: int
    revision: int
    loss: float
    gradient_norm_before_clip: float
    clipped: bool
    elapsed_seconds: float
    persistent_state_bytes: int
    input_layout: dict
    activation_layout: dict
    gradient_layout: dict

    def to_dict(self) -> dict:
        return {
            "layer": self.layer,
            "expert": self.expert,
            "optimizer_step": self.optimizer_step,
            "revision": self.revision,
            "loss": self.loss,
            "gradient_norm_before_clip": self.gradient_norm_before_clip,
            "clipped": self.clipped,
            "elapsed_seconds": self.elapsed_seconds,
            "persistent_state_bytes": self.persistent_state_bytes,
            "input_layout": self.input_layout,
            "activation_layout": self.activation_layout,
            "gradient_layout": self.gradient_layout,
        }


def _array(value, name: str) -> np.ndarray:
    result = np.asarray(value)
    if result.dtype != np.dtype("float32") or result.ndim != 2:
        raise HarnessError(f"{name} must be a rank-2 float32 array")
    if not result.size or not np.isfinite(result).all():
        raise HarnessError(f"{name} must be nonempty and finite")
    return np.ascontiguousarray(result)


def _positive_float(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise HarnessError(f"{name} must be finite and positive")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise HarnessError(f"{name} must be finite and positive") from exc
    if not math.isfinite(result) or result <= 0:
        raise HarnessError(f"{name} must be finite and positive")
    return result


class VulkanExpertHarness:
    """Run one durable linear expert through the native Vulkan VQ path.

    ``backend`` is injectable for host-only tests. When omitted, the harness
    explicitly opens one :class:`ack.Device`; importing this module never opens
    a driver or allocates GPU memory.
    """

    def __init__(self, bank: expert_bank.ExpertBank, *, backend=None, device=None):
        if not isinstance(bank, expert_bank.ExpertBank):
            raise HarnessError("bank must be an ExpertBank")
        if backend is not None and device is not None:
            raise HarnessError("supply backend or device, not both")
        self.bank = bank
        self._device = device
        self._owned_device = False
        if backend is None:
            self._device = device or ack.Device()
            self._owned_device = device is None
            backend = vq.VQDevice(self._device)
        self.backend = backend

    @classmethod
    def open(cls, root, *, device=None, backend=None) -> "VulkanExpertHarness":
        """Open an existing bank without creating or initializing payloads."""
        return cls(expert_bank.ExpertBank.open(root), backend=backend, device=device)

    def __enter__(self) -> "VulkanExpertHarness":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._owned_device and self._device is not None:
            self._device.close()
            self._device = None

    def _state(self, layer: int, expert: int) -> expert_bank.ExpertState:
        try:
            state = self.bank.read_expert_state(layer, expert)
        except expert_bank.ExpertBankError as exc:
            raise HarnessError(str(exc)) from exc
        if len(state.tensors) < 3:
            raise HarnessError("expert state must contain weight, momentum and variance")
        weight, momentum, variance = state.tensors[:3]
        if not (weight.shape == momentum.shape == variance.shape
                and weight.group == momentum.group == variance.group):
            raise HarnessError("expert optimizer state geometry disagrees")
        return state

    def _encode(self, values: np.ndarray, group: int):
        # Calibration is explicit CPU work. The returned image is owned and
        # immutable; the backend then transfers it for device arithmetic.
        calibration = values
        if values.size < group:
            # ``fit_codebook`` needs one complete vector. Padding only the
            # calibration sample keeps the wire geometry exact for tiny
            # batches and the encoder still trims the padded tail.
            calibration = np.pad(values.reshape(-1), (0, group - values.size))
        return self.backend.encode(values, vq.fit_codebook(calibration, group=group), group=group)

    def forward(self, inputs, *, layer: int = 0, expert: int = 0) -> np.ndarray:
        """Run one expert matrix ``inputs @ weight`` on Vulkan."""
        x = _array(inputs, "inputs")
        state = self._state(layer, expert)
        weight = state.tensors[0]
        if x.shape[1] != weight.shape[0]:
            raise HarnessError(
                f"inputs have width {x.shape[1]}, expert expects {weight.shape[0]}"
            )
        return self.backend.matmul(self._encode(x, weight.group), weight)

    def train_step(
        self,
        inputs,
        targets,
        *,
        layer: int = 0,
        expert: int = 0,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.95,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        max_grad_norm: float = 1.0,
    ) -> ExpertStep:
        """Run one MSE expert update and atomically commit its new state."""
        x, target = _array(inputs, "inputs"), _array(targets, "targets")
        if x.shape[0] != target.shape[0]:
            raise HarnessError("inputs and targets must have the same row count")
        state = self._state(layer, expert)
        weight, momentum, variance = state.tensors[:3]
        if x.shape[1] != weight.shape[0] or target.shape[1] != weight.shape[1]:
            raise HarnessError("inputs/targets do not match expert matrix geometry")
        max_grad_norm = _positive_float(max_grad_norm, "max_grad_norm")
        started = time.perf_counter()
        x_image = self._encode(x, weight.group)
        target_image = self._encode(target, weight.group)
        prediction = self.backend.matmul(x_image, weight)
        residual = prediction - self.backend.decode(target_image)
        loss = float(np.mean(np.square(residual, dtype=np.float64)))
        if not math.isfinite(loss):
            raise HarnessError("expert loss is nonfinite")
        residual *= np.float32(2.0 / residual.size)
        residual_image = self._encode(residual, weight.group)
        gradient = self.backend.matmul(x_image, residual_image, transpose_a=True)
        norm = float(np.linalg.norm(gradient.astype(np.float64)))
        if not math.isfinite(norm):
            raise HarnessError("expert gradient norm is nonfinite")
        scale = min(1.0, max_grad_norm / (norm + 1e-6))
        gradient *= np.float32(scale)
        gradient_image = self._encode(gradient, weight.group)
        try:
            optimizer = vq.VQAdamW(
                self.backend,
                weight,
                lr=lr,
                betas=(beta1, beta2),
                eps=eps,
                weight_decay=weight_decay,
            )
            optimizer.momentum, optimizer.variance = momentum, variance
            optimizer.steps = state.optimizer_step
            update = optimizer.step(gradient_image)
        except (TypeError, ValueError, ack.AckError) as exc:
            raise HarnessError(f"expert optimizer refused the update: {exc}") from exc
        try:
            receipt = self.bank.write_expert(
                layer,
                expert,
                (optimizer.weight, optimizer.momentum, optimizer.variance),
                updated_count=weight.layout["elements"],
                optimizer_step=optimizer.steps,
            )
        except expert_bank.ExpertBankError as exc:
            raise HarnessError(f"expert update was not committed: {exc}") from exc
        elapsed = time.perf_counter() - started
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise HarnessError("expert step elapsed time is invalid")
        return ExpertStep(
            layer=layer,
            expert=expert,
            optimizer_step=optimizer.steps,
            revision=receipt["revision"],
            loss=loss,
            gradient_norm_before_clip=norm,
            clipped=scale < 1.0,
            elapsed_seconds=elapsed,
            persistent_state_bytes=update["persistent_state_bytes"],
            input_layout=x_image.layout,
            activation_layout=target_image.layout,
            gradient_layout=gradient_image.layout,
        )

    def train(self, inputs, targets, *, steps: int = 1, **kwargs) -> tuple[ExpertStep, ...]:
        """Run bounded sequential steps, committing after each successful step."""
        if type(steps) is not int or not 1 <= steps <= 1_000_000:
            raise HarnessError("steps must be an integer in 1..1000000")
        return tuple(self.train_step(inputs, targets, **kwargs) for _ in range(steps))


def _load_array(path: Path, name: str) -> np.ndarray:
    if path.is_symlink() or not path.is_file():
        raise HarnessError(f"{name} must be a regular file")
    try:
        value = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise HarnessError(f"{name} is not a readable NumPy array") from exc
    if not isinstance(value, np.ndarray):
        raise HarnessError(f"{name} must be a single .npy array, not an archive")
    return _array(value, name)


def _save_array(path: Path, value: np.ndarray, *, force: bool) -> None:
    if path.exists() and not force:
        raise HarnessError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(raw)
    try:
        with temporary.open("wb") as stream:
            np.save(stream, value, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--device", action="store_true", help="explicitly open Vulkan")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if not args.device:
        parser.error("--device is required; execution is never implicit")
    inputs = _load_array(args.input, "input")
    with VulkanExpertHarness.open(args.bank) as harness:
        if args.target is None:
            result = harness.forward(inputs, layer=args.layer, expert=args.expert)
            _save_array(args.output, result, force=args.force)
            report = {"mode": "forward", "shape": list(result.shape),
                      "device": harness._device.capability_record(),
                      "model_quality_validated": False}
        else:
            targets = _load_array(args.target, "target")
            steps = harness.train(inputs, targets, steps=args.steps, layer=args.layer,
                                  expert=args.expert, lr=args.lr,
                                  max_grad_norm=args.max_grad_norm)
            result = harness.forward(inputs, layer=args.layer, expert=args.expert)
            _save_array(args.output, result, force=args.force)
            report = {"mode": "train", "steps": [step.to_dict() for step in steps],
                      "device": harness._device.capability_record(),
                      "model_quality_validated": False}
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
