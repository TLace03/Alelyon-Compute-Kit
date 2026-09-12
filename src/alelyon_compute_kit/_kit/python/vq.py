"""Experimental vector-coded tensors and Vulkan operations.

Each nibble selects a vector of 8, 16 or 32 values from a 16-entry FP32
codebook. Payload rates include the codebook and padded codewords. This is
lossy storage; arithmetic and temporary outputs use FP32. No model quality
or training-time guarantee follows from a storage rate.
"""
from __future__ import annotations

import ctypes
import math
import operator
import struct
from contextlib import ExitStack
from dataclasses import dataclass

import numpy as np

from . import ack

SCHEMA = 1
GROUPS = (8, 16, 32)
ENTRIES = 16
MAX_ELEMENTS = 268_435_456
MAX_MATMUL_MACS = 268_435_456
_HEADER = struct.Struct("<8sIIIIQQ")
_MAGIC = b"ACKVQ4\0\0"


def _integer(value, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} must be in {minimum}..{maximum}")
    return result


def _shape(shape) -> tuple[int, ...]:
    result = tuple(_integer(x, "dimension", 1, MAX_ELEMENTS) for x in shape)
    if not 1 <= len(result) <= 6 or math.prod(result) > MAX_ELEMENTS:
        raise ValueError("vq shape requires rank 1..6 and a bounded element count")
    return result


def _group(group) -> int:
    group = _integer(group, "group", 8, 32)
    if group not in GROUPS:
        raise ValueError("vq group must be 8, 16 or 32")
    return group


def _float_array(value, name: str) -> np.ndarray:
    result = np.asarray(value)
    if result.dtype != np.dtype("float32") or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite float32 values")
    return np.ascontiguousarray(result)


def _book(value, group: int) -> np.ndarray:
    result = _float_array(value, "codebook")
    if result.shape != (ENTRIES, group):
        raise ValueError(f"codebook must have shape (16, {group})")
    return result


def storage_layout(shape, group: int = 16) -> dict:
    shape, group = _shape(shape), _group(group)
    elements = math.prod(shape)
    vectors = (elements + group - 1) // group
    codes = ((vectors + 7) // 8) * 4
    book = ENTRIES * group * 4
    metadata = _HEADER.size + 8 * len(shape)
    total = codes + book + metadata
    return {"elements": elements, "vectors": vectors, "code_bytes": codes,
            "codebook_bytes": book, "metadata_bytes": metadata,
            "serialized_bytes": total, "payload_bytes": codes + book,
            "bits_per_value": 8 * total / elements,
            "sub_bit": 8 * total < elements}


@dataclass(frozen=True, init=False)
class VQArray:
    """Owned CPU wire image, without a retained dense master tensor."""

    shape: tuple[int, ...]
    group: int
    codes: np.ndarray
    codebook: np.ndarray

    def __init__(self, shape, group, codes, codebook):
        shape, group = _shape(shape), _group(group)
        layout = storage_layout(shape, group)
        raw = np.asarray(codes)
        if raw.dtype != np.dtype("uint32") or raw.shape != (layout["code_bytes"] // 4,):
            raise ValueError("vq codes must be the exact uint32 word image")
        book = _book(codebook, group)
        # Canonical tail bits avoid different wire images for the same tensor.
        tail = layout["vectors"] % 8
        if tail and int(raw[-1]) >> (4 * tail):
            raise ValueError("vq unused codeword bits must be zero")
        owned_codes, owned_book = raw.copy(), book.copy()
        owned_codes.flags.writeable = False
        owned_book.flags.writeable = False
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "group", group)
        object.__setattr__(self, "codes", owned_codes)
        object.__setattr__(self, "codebook", owned_book)

    @property
    def layout(self) -> dict:
        return storage_layout(self.shape, self.group)

    def decode(self) -> np.ndarray:
        layout = self.layout
        positions = np.arange(layout["vectors"], dtype=np.int64)
        indices = (self.codes[positions // 8] >> ((positions % 8) * 4).astype(np.uint32)) & 15
        return self.codebook[indices].reshape(-1)[:layout["elements"]].copy().reshape(self.shape)

    def to_bytes(self) -> bytes:
        layout = self.layout
        header = _HEADER.pack(_MAGIC, SCHEMA, self.group, len(self.shape), 0,
                              layout["elements"], layout["code_bytes"])
        dims = struct.pack("<" + "Q" * len(self.shape), *self.shape)
        return header + dims + self.codebook.astype("<f4").tobytes() + self.codes.astype("<u4").tobytes()

    @classmethod
    def from_bytes(cls, data, *, max_bytes: int = 128 * 1024 * 1024) -> "VQArray":
        limit = _integer(max_bytes, "max_bytes", _HEADER.size, 1 << 32)
        view = memoryview(data).cast("B")
        if not _HEADER.size <= view.nbytes <= limit:
            raise ValueError("vq artifact byte limit or truncated header")
        magic, schema, group, rank, reserved, elements, code_bytes = _HEADER.unpack_from(view)
        if magic != _MAGIC or schema != SCHEMA or reserved or not 1 <= rank <= 6:
            raise ValueError("unsupported vq artifact header")
        dims_end = _HEADER.size + rank * 8
        if view.nbytes < dims_end:
            raise ValueError("truncated vq dimensions")
        shape = struct.unpack_from("<" + "Q" * rank, view, _HEADER.size)
        layout = storage_layout(shape, group)
        if elements != layout["elements"] or code_bytes != layout["code_bytes"]:
            raise ValueError("vq header disagrees with its dimensions")
        if view.nbytes != layout["serialized_bytes"]:
            raise ValueError("vq artifact length mismatch")
        book_end = dims_end + layout["codebook_bytes"]
        book = np.frombuffer(view[dims_end:book_end], dtype="<f4").reshape(ENTRIES, group)
        codes = np.frombuffer(view[book_end:], dtype="<u4")
        return cls(shape, group, codes.astype(np.uint32), book.astype(np.float32))


def encode(values, codebook, *, group: int = 16) -> VQArray:
    """CPU reference: nearest code vector with lowest-index tie breaking."""
    group = _group(group)
    x = _float_array(values, "values")
    shape = _shape(x.shape)
    book = _book(codebook, group)
    layout = storage_layout(shape, group)
    padded = np.zeros(layout["vectors"] * group, dtype=np.float32)
    padded[:x.size] = x.reshape(-1)
    vectors = padded.reshape(-1, group)
    chosen = np.empty(layout["vectors"], dtype=np.uint32)
    # Chunking bounds temporary distances rather than allocating N*16*group.
    for start in range(0, len(vectors), 4096):
        v = vectors[start:start + 4096].astype(np.float64)
        distances = np.zeros((len(v), ENTRIES), dtype=np.float64)
        for j in range(group):
            delta = v[:, j, None] - book[None, :, j].astype(np.float64)
            if start + len(v) == len(vectors) and x.size % group and j >= x.size % group:
                delta[-1] = 0  # tail values do not exist and cannot affect selection
            distances += delta * delta
        chosen[start:start + len(v)] = np.argmin(distances, axis=1).astype(np.uint32)
    words = np.zeros(layout["code_bytes"] // 4, dtype=np.uint32)
    for lane in range(8):
        part = chosen[lane::8]
        words[:len(part)] |= part << np.uint32(lane * 4)
    return VQArray(shape, group, words, book)


def fit_codebook(values, *, group: int = 16, iterations: int = 4,
                 sample_vectors: int = 4096) -> np.ndarray:
    """Bounded deterministic calibration; does not establish task quality."""
    group = _group(group)
    iterations = _integer(iterations, "iterations", 0, 32)
    sample_vectors = _integer(sample_vectors, "sample_vectors", ENTRIES, 65536)
    x = _float_array(values, "values").reshape(-1)
    if x.size < group:
        raise ValueError("codebook calibration needs at least one complete vector")
    complete = x[:x.size // group * group].reshape(-1, group)
    sample = complete[np.linspace(0, len(complete) - 1, min(len(complete), sample_vectors), dtype=np.int64)]
    book = sample[np.linspace(0, len(sample) - 1, ENTRIES, dtype=np.int64)].copy()
    for _ in range(iterations):
        assigned = encode(sample, book, group=group)
        pos = np.arange(len(sample))
        idx = (assigned.codes[pos // 8] >> ((pos % 8) * 4).astype(np.uint32)) & 15
        for entry in range(ENTRIES):
            selected = sample[idx == entry]
            if len(selected):
                book[entry] = selected.mean(axis=0, dtype=np.float64).astype(np.float32)
    return book


class VQDevice:
    """Synchronous status-checked VQ extension on an already owned ACK device.

    Calls transfer packed images and temporary outputs. This convenience API
    is not a claim that an entire training graph stays resident on the GPU.
    """

    def __init__(self, device: ack.Device):
        self.device = device
        with device._lock:
            device._require_open()
            lib = device._lib.cdll
            try:
                schema = lib.ack_vq_schema
                self._call = lib.ack_vq
            except AttributeError as exc:
                raise ack.AckUnavailable("library lacks the optional vq4/v1 extension") from exc
            schema.argtypes, schema.restype = [], ctypes.c_uint32
            if schema() != SCHEMA:
                raise ack.AckUnavailable("incompatible vq4 extension schema")
            self._call.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_void_p, ctypes.c_size_t]
            self._call.restype = ctypes.c_int

    def _run(self, words, inputs, output_bytes: int, dtype):
        device = self.device
        with device._lock, ExitStack() as cleanup:
            device._require_open()
            buffers = []
            for array in inputs:
                b = device.buffer(array.nbytes)
                cleanup.callback(b.free)
                b.upload(array)
                buffers.append(b)
            out = device.buffer(output_bytes)
            cleanup.callback(out.free)
            status = device.buffer(4)
            cleanup.callback(status.free)
            status.upload(np.zeros(1, dtype=np.uint32))
            while len(buffers) < 4:
                buffers.append(buffers[0])  # unread slots still hold a live input
            plan = struct.pack("<8I", *words)
            rc = self._call(device._handle, *(b._handle for b in buffers),
                            out._handle, status._handle, plan, len(plan))
            if rc != ack.ACK_OK:
                raise ack._kit_error("vq", rc, device._lib.last_error())
            flags = int(status.download(np.uint32)[0])
            if flags:
                raise ack.AckError(f"vq device arithmetic refused: status={flags}")
            return out.download(dtype)

    def encode(self, values, codebook, *, group: int = 16) -> VQArray:
        group = _group(group)
        x = _float_array(values, "values")
        layout = storage_layout(x.shape, group)
        book = _book(codebook, group)
        result = self._run((0, x.size, 1, 1, group, ENTRIES, 0, 0),
                           (x, book), layout["code_bytes"], np.uint32)
        return VQArray(x.shape, group, result, book)

    def decode(self, value: VQArray) -> np.ndarray:
        layout = value.layout
        return self._run((1, layout["elements"], 1, 1, value.group, ENTRIES, 0, 0),
                         (value.codes, value.codebook), layout["elements"] * 4,
                         np.float32).reshape(value.shape)

    def matmul(self, a: VQArray, b: VQArray, *, transpose_a=False, transpose_b=False) -> np.ndarray:
        if type(transpose_a) is not bool or type(transpose_b) is not bool:
            raise ValueError("transpose flags must be boolean")
        if len(a.shape) != 2 or len(b.shape) != 2 or a.group != b.group:
            raise ValueError("vq matmul requires two matrices using the same vector group")
        m, k = a.shape[::-1] if transpose_a else a.shape
        k2, n = b.shape[::-1] if transpose_b else b.shape
        if k != k2 or max(m, n, k) > 65536 or m * n > MAX_ELEMENTS:
            raise ValueError("vq matmul dimensions are incompatible or exceed bounds")
        if m * n * k > MAX_MATMUL_MACS:
            raise ValueError("vq-matmul-work-limit: split the product into bounded calls")
        flags = int(transpose_a) | (int(transpose_b) << 1)
        return self._run((2, m, n, k, a.group, ENTRIES, flags, 0),
                         (a.codes, a.codebook, b.codes, b.codebook), m * n * 4,
                         np.float32).reshape(m, n)

    def adamw(self, weight: VQArray, momentum: VQArray, variance: VQArray,
              gradient: VQArray, *, lr: float, beta1: float, beta2: float,
              eps: float, weight_decay: float, step: int) -> tuple[np.ndarray, ...]:
        """Native update from four coded states into bounded FP32 scratch."""
        states = (weight, momentum, variance, gradient)
        if any(s.shape != weight.shape or s.group != weight.group for s in states):
            raise ValueError("vq AdamW states must have identical shapes and groups")
        step = _integer(step, "step", 1, 1 << 53)
        values = (lr, beta1, beta2, eps, weight_decay)
        if any(isinstance(x, (bool, np.bool_)) or not math.isfinite(float(x)) for x in values):
            raise ValueError("vq AdamW coefficients must be finite numbers")
        if lr < 0 or not 0 <= beta1 < 1 or not 0 <= beta2 < 1 or eps <= 0 or weight_decay < 0:
            raise ValueError("vq AdamW coefficients outside their domains")
        count = weight.layout["elements"]
        if count * 3 > MAX_ELEMENTS:
            raise ValueError("vq AdamW active scratch exceeds element bound")
        coefficients = np.array([lr, beta1, beta2, eps, weight_decay,
                                 1 - beta1 ** step, 1 - beta2 ** step], dtype=np.float32)
        if not np.isfinite(coefficients).all() or coefficients[3] <= 0 or (coefficients[5:] <= 0).any():
            raise ValueError("vq AdamW coefficients are not representable in float32")
        codes = np.concatenate([s.codes for s in states])
        books = np.concatenate([s.codebook.reshape(-1) for s in states])
        result = self._run((3, count, 1, 1, weight.group, ENTRIES, 0, 0),
                           (codes, books, coefficients), count * 3 * 4, np.float32)
        return tuple(x.reshape(weight.shape) for x in np.split(result, 3))


class VQAdamW:
    """Experimental optimizer with coded persistent weights and both moments.

    GPU update scratch and CPU codebook calibration are FP32. Re-encoding is
    lossy; this implements an experiment, not ordinary AdamW equivalence. A
    failed update/re-encode leaves the three previous state images and step
    counter unchanged. The caller retains responsibility for durable commits.
    """

    def __init__(self, backend: VQDevice, weight: VQArray, *, lr=1e-3,
                 betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0):
        self.backend = backend
        self.weight = weight
        zeros = np.zeros_like(weight.codes)
        zero_book = np.zeros((ENTRIES, weight.group), dtype=np.float32)
        self.momentum = VQArray(weight.shape, weight.group, zeros, zero_book)
        self.variance = VQArray(weight.shape, weight.group, zeros, zero_book)
        self.lr, self.betas, self.eps, self.weight_decay = lr, betas, eps, weight_decay
        self.steps = 0

    def step(self, gradient: VQArray) -> dict:
        next_step = self.steps + 1
        scratch = self.backend.adamw(self.weight, self.momentum, self.variance, gradient,
                                    lr=self.lr, beta1=self.betas[0], beta2=self.betas[1],
                                    eps=self.eps, weight_decay=self.weight_decay, step=next_step)
        encoded = []
        for value in scratch:
            book = fit_codebook(value, group=self.weight.group)
            encoded.append(self.backend.encode(value, book, group=self.weight.group))
        changed = int(np.count_nonzero(encoded[0].decode() != self.weight.decode()))
        self.weight, self.momentum, self.variance = encoded
        self.steps = next_step
        return {"step": self.steps, "changed_weight_values": changed,
                "persistent_state_bytes": sum(s.layout["serialized_bytes"] for s in encoded),
                "gradient_bytes": gradient.layout["serialized_bytes"],
                "native_update_output_bytes": sum(s.nbytes for s in scratch),
                "calibration": "CPU float32/float64 temporary arrays",
                "model_quality_validated": False}
