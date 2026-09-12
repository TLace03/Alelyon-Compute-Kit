"""Experimental sparse-on-disk expert generations and measured-gain scheduling.

Trusted local single writer, concurrent readers. Hashes detect changed bytes;
they do not authenticate an author or prove that declared updates ran. An expert
becomes visible only when its complete generation pointer is atomically replaced.
Interrupted writes may leave unreferenced shards, which are counted as disk bytes
and never as materialized parameters. No automatic deletion or dense allocation.
File fsync plus replace covers process interruption; power-loss durability of the
directory entry is platform/filesystem dependent (especially on Windows).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .vq import ENTRIES, GROUPS, VQArray, storage_layout

MAX_SHARD_BYTES = 128 * 1024 * 1024
MAX_METADATA_BYTES = 128 * 1024
_SCHEMA = "ack-expert-bank/v1"
_POINTER = re.compile(r"expert-([0-9]+)-([0-9]+)\.json\Z")
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_GENERATION = re.compile(r"[0-9a-f]{32}\Z")


class ExpertBankError(ValueError):
    """A bank, payload, declaration or boundary is not admissible."""


def _integer(value, name: str, low: int = 0, high: int = (1 << 63) - 1) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ExpertBankError(f"{name} must be an integer in {low}..{high}")
    return value


def _keys(value, keys: set[str], name: str):
    if not isinstance(value, dict) or set(value) != keys:
        raise ExpertBankError(f"invalid {name} fields")


def _json(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ExpertBankError("duplicate metadata key")
        result[key] = value
    return result


def _nonfinite(_value):
    raise ExpertBankError("nonfinite metadata")


def _parse(data: bytes):
    try:
        value = json.loads(data, object_pairs_hook=_pairs, parse_constant=_nonfinite)
        canonical = _json(value)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ExpertBankError("invalid bank metadata") from exc
    except ValueError as exc:
        raise ExpertBankError(str(exc)) from exc
    if canonical != data:
        raise ExpertBankError("noncanonical bank metadata")
    return value


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class BankConfig:
    layers: int
    experts_per_layer: int
    matrix_shapes: tuple[tuple[int, int], ...]
    group: int = 16

    def __post_init__(self):
        _integer(self.layers, "layers", 1, 1_000_000)
        _integer(self.experts_per_layer, "experts_per_layer", 1, 1_000_000)
        if type(self.group) is not int or self.group not in GROUPS:
            raise ExpertBankError("group must be 8, 16 or 32")
        if not isinstance(self.matrix_shapes, (tuple, list)) or not 1 <= len(self.matrix_shapes) <= 16:
            raise ExpertBankError("matrix_shapes must contain 1..16 matrices")
        shapes = []
        for shape in self.matrix_shapes:
            if not isinstance(shape, (tuple, list)) or len(shape) != 2:
                raise ExpertBankError("each matrix must have two dimensions")
            shape = tuple(_integer(dim, "matrix dimension", 1) for dim in shape)
            try:
                layout = storage_layout(shape, self.group)
            except ValueError as exc:
                raise ExpertBankError(str(exc)) from exc
            if layout["serialized_bytes"] > MAX_SHARD_BYTES:
                raise ExpertBankError("matrix exceeds shard byte limit")
            shapes.append(shape)
        object.__setattr__(self, "matrix_shapes", tuple(shapes))
        _integer(self.potential_parameters, "potential_parameters")

    @property
    def parameters_per_expert(self) -> int:
        return sum(math.prod(shape) for shape in self.matrix_shapes)

    @property
    def potential_parameters(self) -> int:
        return self.layers * self.experts_per_layer * self.parameters_per_expert

    def to_dict(self) -> dict:
        return {"layers": self.layers, "experts_per_layer": self.experts_per_layer,
                "matrix_shapes": [list(shape) for shape in self.matrix_shapes], "group": self.group}

    @classmethod
    def from_dict(cls, value) -> "BankConfig":
        _keys(value, {"layers", "experts_per_layer", "matrix_shapes", "group"}, "config")
        return cls(**value)


@dataclass(frozen=True)
class ExpertState:
    tensors: tuple[VQArray, ...]
    optimizer_step: int
    revision: int
    parameter_update_applications: int


class ExpertBank:
    """A fixed logical bank; create/open never initialize expert payloads."""

    def __init__(self, root: Path, config: BankConfig, config_hash: str):
        self.root, self.config, self._config_hash = root, config, config_hash

    @staticmethod
    def _read(path: Path, limit: int) -> bytes:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or path.is_symlink() or info.st_size > limit:
            raise ExpertBankError("file is not regular or exceeds byte limit")
        with path.open("rb") as stream:
            data = stream.read(limit + 1)
        if len(data) > limit or len(data) != info.st_size:
            raise ExpertBankError("file changed size or exceeds byte limit")
        return data

    @staticmethod
    def _write(path: Path, data: bytes):
        with path.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())

    @classmethod
    def create(cls, root, config: BankConfig) -> "ExpertBank":
        if not isinstance(config, BankConfig):
            raise ExpertBankError("config must be BankConfig")
        path = Path(root)
        # No reuse of an existing directory, even an empty one.
        path.mkdir(parents=False, exist_ok=False)
        if path.is_symlink():
            raise ExpertBankError("bank root must not be a symlink")
        path = path.resolve(strict=True)
        data = _json({"schema": _SCHEMA, "config": config.to_dict()})
        cls._write(path / "bank.json", data)
        return cls(path, config, _digest(data))

    @classmethod
    def open(cls, root) -> "ExpertBank":
        path = Path(root)
        if path.is_symlink() or not path.is_dir():
            raise ExpertBankError("bank root must be a regular directory")
        path = path.resolve(strict=True)
        data = cls._read(path / "bank.json", MAX_METADATA_BYTES)
        value = _parse(data)
        _keys(value, {"schema", "config"}, "bank")
        if value["schema"] != _SCHEMA:
            raise ExpertBankError("unsupported bank schema")
        return cls(path, BankConfig.from_dict(value["config"]), _digest(data))

    def _check(self):
        if self.root.is_symlink() or not self.root.is_dir() \
                or _digest(self._read(self.root / "bank.json", MAX_METADATA_BYTES)) != self._config_hash:
            raise ExpertBankError("bank configuration changed")

    def _id(self, layer: int, expert: int) -> str:
        _integer(layer, "layer", 0, self.config.layers - 1)
        _integer(expert, "expert", 0, self.config.experts_per_layer - 1)
        return f"expert-{layer}-{expert}"

    def _pointer(self, layer: int, expert: int):
        base = self._id(layer, expert)
        path = self.root / f"{base}.json"
        try:
            data = self._read(path, MAX_METADATA_BYTES)
        except FileNotFoundError:
            return None
        value = _parse(data)
        _keys(value, {"schema", "config_sha256", "layer", "expert", "generation", "revision",
                      "shards", "parameter_update_applications", "update_commits", "optimizer_step"}, "expert")
        if value["schema"] != _SCHEMA or value["config_sha256"] != self._config_hash \
                or type(value["layer"]) is not int or value["layer"] != layer \
                or type(value["expert"]) is not int or value["expert"] != expert \
                or not isinstance(value["generation"], str) or not _GENERATION.fullmatch(value["generation"]):
            raise ExpertBankError("expert identity mismatch")
        _integer(value["revision"], "revision", 1)
        _integer(value["optimizer_step"], "optimizer_step", 0, 1 << 53)
        updates = _integer(value["parameter_update_applications"], "parameter_update_applications")
        commits = _integer(value["update_commits"], "update_commits", 0, value["revision"])
        if (updates == 0) != (commits == 0) or updates < commits \
                or updates > commits * self.config.parameters_per_expert:
            raise ExpertBankError("inconsistent update counters")
        if not isinstance(value["shards"], list) or len(value["shards"]) != len(self.config.matrix_shapes):
            raise ExpertBankError("expert shard count mismatch")
        for shard, shape in zip(value["shards"], self.config.matrix_shapes):
            _keys(shard, {"bytes", "sha256"}, "shard")
            _integer(shard["bytes"], "shard bytes", 1, MAX_SHARD_BYTES)
            if shard["bytes"] != storage_layout(shape, self.config.group)["serialized_bytes"] \
                    or not isinstance(shard["sha256"], str) or not _HASH.fullmatch(shard["sha256"]):
                raise ExpertBankError("shard identity or layout mismatch")
        return value

    def _shard_path(self, base: str, generation: str, index: int) -> Path:
        return self.root / f"{base}-g{generation}-t{index}.vq"

    def read_expert(self, layer: int, expert: int) -> tuple[VQArray, ...]:
        return self.read_expert_state(layer, expert).tensors

    def read_expert_state(self, layer: int, expert: int) -> ExpertState:
        """Read tensors and scalar optimizer bookkeeping from one pointer snapshot."""
        self._check()
        pointer = self._pointer(layer, expert)
        if pointer is None:
            raise ExpertBankError("expert is not materialized")
        return ExpertState(self._read_generation(pointer), pointer["optimizer_step"],
                           pointer["revision"], pointer["parameter_update_applications"])

    def _read_generation(self, pointer: dict, *, retain: bool = True) -> tuple[VQArray, ...]:
        base = self._id(pointer["layer"], pointer["expert"])
        result = []
        for index, (shard, shape) in enumerate(zip(pointer["shards"], self.config.matrix_shapes)):
            data = self._read(self._shard_path(base, pointer["generation"], index), MAX_SHARD_BYTES)
            if len(data) != shard["bytes"] or _digest(data) != shard["sha256"]:
                raise ExpertBankError("shard checksum mismatch")
            try:
                array = VQArray.from_bytes(data, max_bytes=MAX_SHARD_BYTES)
            except ValueError as exc:
                raise ExpertBankError(f"invalid VQ shard: {exc}") from exc
            if array.shape != shape or array.group != self.config.group:
                raise ExpertBankError("decoded shard geometry mismatch")
            if retain:
                result.append(array)
        return tuple(result)

    def write_expert(self, layer: int, expert: int, tensors: Sequence[VQArray], *,
                     updated_count: int = 0, optimizer_step: int | None = None) -> dict:
        self._check()
        base = self._id(layer, expert)
        _integer(updated_count, "updated_count", 0, self.config.parameters_per_expert)
        if optimizer_step is not None:
            _integer(optimizer_step, "optimizer_step", 0, 1 << 53)
        if not isinstance(tensors, (tuple, list)) or len(tensors) != len(self.config.matrix_shapes):
            raise ExpertBankError("one VQArray per configured matrix is required")
        # Validate every geometry before writing; each payload is then checked
        # before its own write. Failure leaves no newly committed generation.
        for array, shape in zip(tensors, self.config.matrix_shapes):
            if not isinstance(array, VQArray) or array.shape != shape or array.group != self.config.group:
                raise ExpertBankError("VQArray geometry mismatch")
            if array.layout["serialized_bytes"] > MAX_SHARD_BYTES:
                raise ExpertBankError("shard byte limit exceeded")
        previous = self._pointer(layer, expert)
        prior_step = 0 if previous is None else previous["optimizer_step"]
        if optimizer_step is None:
            optimizer_step = prior_step
        if optimizer_step < prior_step:
            raise ExpertBankError("optimizer_step must not regress")
        if previous is not None:
            self._read_generation(previous, retain=False)  # refuse corrupt committed state
        generation = uuid.uuid4().hex
        pointer = {"schema": _SCHEMA, "config_sha256": self._config_hash,
                   "layer": layer, "expert": expert, "generation": generation,
                   "optimizer_step": optimizer_step,
                   "revision": 1 if previous is None else previous["revision"] + 1,
                   "shards": [], "parameter_update_applications": updated_count + (
                       0 if previous is None else previous["parameter_update_applications"]),
                   "update_commits": int(updated_count > 0) + (
                       0 if previous is None else previous["update_commits"])}
        _integer(pointer["parameter_update_applications"], "parameter_update_applications")
        for index, array in enumerate(tensors):
            data = array.to_bytes()
            # A caller may have mutated exposed NumPy storage despite its flag.
            VQArray.from_bytes(data, max_bytes=MAX_SHARD_BYTES)
            self._write(self._shard_path(base, generation, index), data)
            pointer["shards"].append({"bytes": len(data), "sha256": _digest(data)})
        pending = self.root / f"pending-{generation}.json"
        self._write(pending, _json(pointer))
        self._check()
        if self._pointer(layer, expert) != previous:
            raise ExpertBankError("another writer changed this expert")
        os.replace(pending, self.root / f"{base}.json")
        return {"layer": layer, "expert": expert, "revision": pointer["revision"],
                "optimizer_step": optimizer_step,
                "materialized_parameters": self.config.parameters_per_expert,
                "updated_count": updated_count,
                "parameter_update_applications": pointer["parameter_update_applications"],
                "generation_bytes": sum(shard["bytes"] for shard in pointer["shards"])}

    def initialize_expert(self, layer: int, expert: int, *, seed: int = 0,
                          zero_optimizer_state: bool = False) -> dict:
        """Materialize deterministic payloads for an expert.

        ``zero_optimizer_state=True`` is the initialization contract for the
        VQ AdamW harness: the first matrix is seeded as a weight and the next
        two matrices are zero momentum and variance. The default keeps the
        generic bank initializer's historical deterministic random payloads.
        """
        self._check()
        self._id(layer, expert)
        _integer(seed, "seed", 0, (1 << 64) - 1)
        if not isinstance(zero_optimizer_state, bool):
            raise ExpertBankError("zero_optimizer_state must be bool")
        if zero_optimizer_state and len(self.config.matrix_shapes) < 3:
            raise ExpertBankError("zero optimizer state needs three matrix roles")
        if self._pointer(layer, expert) is not None:
            raise ExpertBankError("expert is already materialized")
        tensors = []
        for index, shape in enumerate(self.config.matrix_shapes):
            layout = storage_layout(shape, self.config.group)
            if zero_optimizer_state and index in (1, 2):
                codes = np.zeros(layout["code_bytes"] // 4, dtype=np.uint32)
                book = np.zeros((ENTRIES, self.config.group), dtype=np.float32)
            else:
                rng = np.random.default_rng(np.random.SeedSequence([seed, layer, expert, index]))
                codes = rng.integers(0, 1 << 32, layout["code_bytes"] // 4, dtype=np.uint32)
                tail = layout["vectors"] % 8
                if tail:
                    codes[-1] &= np.uint32((1 << (tail * 4)) - 1)
                book = rng.standard_normal((ENTRIES, self.config.group), dtype=np.float32)
                book *= np.float32(1.0 / math.sqrt(shape[0]))
            tensors.append(VQArray(shape, self.config.group, codes, book))
        return self.write_expert(layer, expert, tensors)

    def stats(self) -> dict:
        """Validate committed payloads; sum logical file lengths, including history.

        Update applications are caller-declared cumulative counts, not unique
        parameter coverage. Stats scans only existing files, not potential IDs.
        `disk_bytes` is not filesystem allocated-block usage. A coherent aggregate
        census requires a quiescent writer; per-expert reads use atomic snapshots.
        """
        self._check()
        materialized = updates = updated_experts = committed_bytes = disk_bytes = 0
        for path in self.root.iterdir():
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or path.is_symlink():
                raise ExpertBankError("unexpected nonregular bank entry")
            disk_bytes += info.st_size
            match = _POINTER.fullmatch(path.name)
            if match is None:
                continue
            layer, expert = map(int, match.groups())
            if path.name != f"{self._id(layer, expert)}.json":
                raise ExpertBankError("noncanonical expert filename")
            pointer = self._pointer(layer, expert)
            if pointer is None:
                raise ExpertBankError("expert pointer disappeared")
            self._read_generation(pointer, retain=False)
            materialized += 1
            updates += pointer["parameter_update_applications"]
            updated_experts += int(pointer["update_commits"] > 0)
            committed_bytes += sum(shard["bytes"] for shard in pointer["shards"])
        return {"potential_experts": self.config.layers * self.config.experts_per_layer,
                "potential_parameters": self.config.potential_parameters,
                "materialized_experts": materialized,
                "materialized_parameters": materialized * self.config.parameters_per_expert,
                "updated_experts": updated_experts,
                "parameter_update_applications": updates,
                "unique_updated_parameters": None,
                "committed_payload_bytes": committed_bytes, "disk_bytes": disk_bytes}


@dataclass(frozen=True)
class GainObservation:
    loss_before: float
    loss_after: float
    elapsed_s: float
    samples: int
    updated_count: int

    @property
    def gain_per_second(self) -> float:
        return max(0.0, self.loss_before - self.loss_after) / self.elapsed_s if self.updated_count else 0.0


class GainScheduler:
    """Pure CPU scheduling from latest declared held-out observations.

    A positive score is observed improvement per second, not proof of future
    utility. An exploration fraction is reserved across calls (including
    single-expert requests); ties use the lower flat expert ID.
    """

    def __init__(self, total_experts: int, exploration_fraction: float = 0.2):
        self.total_experts = _integer(total_experts, "total_experts", 1)
        if isinstance(exploration_fraction, bool) or not isinstance(exploration_fraction, (int, float)) \
                or not math.isfinite(exploration_fraction) or not 0 < exploration_fraction <= 1:
            raise ExpertBankError("exploration_fraction must be in (0, 1]")
        self.exploration_fraction = float(exploration_fraction)
        self.observations: dict[int, GainObservation] = {}
        self._cursor = 0
        self._exploration_credit = 0.0

    def observe(self, expert: int, loss_before: float, loss_after: float, elapsed_s: float,
                samples: int, updated_count: int, *, heldout: bool = True):
        _integer(expert, "expert", 0, self.total_experts - 1)
        if heldout is not True:
            raise ExpertBankError("scheduler requires held-out loss observations")
        for name, value in (("loss_before", loss_before), ("loss_after", loss_after), ("elapsed_s", elapsed_s)):
            if isinstance(value, bool) or not isinstance(value, (float, int)) \
                    or not math.isfinite(value) or value < 0:
                raise ExpertBankError(f"{name} must be finite and nonnegative")
        if elapsed_s == 0:
            raise ExpertBankError("elapsed_s must be positive")
        _integer(samples, "samples", 1)
        _integer(updated_count, "updated_count")
        observation = GainObservation(float(loss_before), float(loss_after), float(elapsed_s), samples, updated_count)
        if not math.isfinite(observation.gain_per_second):
            raise ExpertBankError("gain per second is not finite")
        self.observations[expert] = observation

    def select(self, count: int) -> tuple[int, ...]:
        _integer(count, "count", 1, min(self.total_experts, 4096))
        self._exploration_credit += count * self.exploration_fraction
        reserve = min(count, int(self._exploration_credit + 1e-12))
        self._exploration_credit = max(0.0, self._exploration_credit - reserve)
        selected: list[int] = []

        def explore():
            while self._cursor in selected:
                self._cursor = (self._cursor + 1) % self.total_experts
            selected.append(self._cursor)
            self._cursor = (self._cursor + 1) % self.total_experts

        for _ in range(reserve):
            explore()
        ranked = sorted((expert for expert, obs in self.observations.items() if obs.gain_per_second > 0),
                        key=lambda expert: (-self.observations[expert].gain_per_second, expert))
        for expert in ranked:
            if len(selected) == count:
                break
            if expert not in selected:
                selected.append(expert)
        while len(selected) < count:
            explore()
        return tuple(selected)
