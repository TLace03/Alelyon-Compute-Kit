"""Python surface of the Alelyon Compute Kit: ctypes over the crate's C ABI.

    >>> from alelyon.languages.compute_kit.python import ack
    >>> dev = ack.Device()                     # raises ack.AckUnavailable when no device
    >>> c = dev.matmul(a, b)                   # a: (M, K) float32 or bf16 bits, b: (K, N); c: (M, N) float32

BF16 is carried as uint16 bit patterns (numpy has no bf16 dtype); `to_bf16`
rounds float32 to nearest-even and `from_bf16` widens back. The matmul is the
kit's BF16 cooperative-matrix kernel with f32 accumulation and f32 output, so
`matmul(a, b)` equals `from_bf16(to_bf16(a)) @ from_bf16(to_bf16(b))` in f32 up
to summation order. Shapes must be multiples of the 128x128 tile with K a
multiple of 32; the kit refuses others and this wrapper raises `AckShapeError`
rather than padding.

The shared library is looked up next to the crate (`target/release`, then
`target/debug`) or at `ACK_LIBRARY`; when it is absent, `Device()` raises
`AckUnavailable` so a test can report UNMEASURED instead of failing.

Buffer transfers accept plain fixed-width NumPy bool, integer, float and complex
dtypes only (1-byte bool, 1/2/4/8-byte integers, 2/4/8-byte floats, 8/16-byte
complex). Structured, subarray, string and object dtypes are refused. Transfers
copy bytes without changing byte order. A Device refuses close while any of its
buffers remain live; call Buffer.free() first. Calls through one Python Device
are serialized, including close/free, because ctypes releases the GIL.

The library announces its ABI (`ack_abi_info`): the loader refuses, with
`AckUnavailable`, a library whose ABI version is not the one this module was
written for, before binding any other symbol. At the C ABI every handle is an
id that names its owner, so a closed device, a foreign buffer or a freed buffer
is refused by code (ACK_ERR_CLOSED, ACK_ERR_FOREIGN, ACK_ERR_FREED), never
dereferenced; the Python guards above are the first layer, the kit's registry
the second. Ids are sequential, not secrets. A device whose state is unknown
after a failed fence wait refuses every later transfer, and a free on it, with
ACK_ERR_POISONED, raised here as `AckPoisoned`: close it and start a new
process (a reopen in the same process is UNMEASURED beyond a healthy card).
"""

from __future__ import annotations

import ctypes
import struct
import operator
import os
import sys
import threading
import warnings
from contextlib import ExitStack
from pathlib import Path

# NumPy is OPTIONAL and must stay that way. The published package declares
# `dependencies = []` and its distribution verifier pins that zero in several
# places, and the direction matters: moving numpy from required to optional
# AFTER a release breaks every install that relied on it, while adding a
# required dependency later does not break anyone. So the irreversible choice is
# the one taken deliberately here -- the ctypes surface works with bytes and
# memoryviews, and numpy buys convenience, not capability.
try:  # pragma: no cover - the presence branch is what every test exercises
    import numpy as np
except ImportError:  # pragma: no cover - exercised by test_ack_without_numpy
    np = None

_HERE = Path(__file__).resolve().parent
_CRATE = _HERE.parent


def _require_numpy(what: str):
    """The numpy module, or a refusal naming what needed it and how to get it."""
    if np is None:
        raise AckError(
            f"{what} needs numpy, which is not installed. The kit's core surface "
            "does not: use upload_bytes/download_bytes, or install the extra "
            "(pip install alelyon-ai[numpy])."
        )
    return np


def _is_bool(value) -> bool:
    """True for a Python bool or a numpy bool, without requiring numpy."""
    return isinstance(value, bool) or (np is not None and isinstance(value, np.bool_))

ACK_OK = 0
ACK_ERR_NULL = -1
ACK_ERR_VULKAN = -2
ACK_ERR_SHAPE = -3
ACK_ERR_SIZE = -4
ACK_ERR_LIVE_BUFFERS = -5
ACK_ERR_CLOSED = -6      # device handle is not open (closed, never opened, or not a kit handle)
ACK_ERR_FOREIGN = -7     # buffer belongs to another device
ACK_ERR_FREED = -8       # buffer handle is unknown to its device (freed, or never allocated)
ACK_ERR_ABI = -9         # the caller's AckAbiInfo is smaller than the library's
ACK_ERR_POISONED = -10   # the device's state is unknown after a failed wait, or it is lost; close and reopen
ACK_ERR_UNSUPPORTED_OP = -11   # the operator's kernels could not be built on this device (BF16 cooperative matrices absent)
ACK_ERR_TIMING = -12     # declared, and unreachable since deferred submission: no C entry times a dispatch any more, so every product entry reports NaN in ms_out

ACK_ABI_VERSION = 10     # the C ABI this module is written for; ack_abi_info must agree exactly
ACK_OP_MATMUL_BF16 = 1
ACK_OP_CAST = 2
ACK_OP_MATMUL_F32 = 4    # matmul-f32/v1: ack_matmul_f32 with the 64-byte plan block
ACK_OP_POINTWISE = 8     # pointwise-f32/v1: ack_pointwise with the 120-byte plan block (ABI 6)
ACK_OP_REDUCE = 16       # reduce-f32/v1: ack_reduce with the 80-byte request block (ABI 7)
ACK_OP_ROWWISE = 32      # rowwise-f32/v2: ack_rowwise with the kernel's 16-byte push block (ABI 8)
ACK_OP_LOSS = 64         # indexed-cross-entropy-f32/v2: ack_loss, which validates and uploads its own targets (ABI 9)
ACK_OP_EMBEDDING = 128   # embedding-f32/v2: ack_embedding, which derives and uploads its index and CSR images (ABI 9)
# the operator schemas this module is written for; the handshake requires each
# exactly (0 means the library does not declare that operator's signature)
ACK_OP_SCHEMA_MATMUL_BF16 = 2  # 2: ack_matmul_bf16_z, the z-grid product beside ack_matmul_bf16 (ABI 10, 2026-09-09)
ACK_OP_SCHEMA_CAST = 1
ACK_OP_SCHEMA_MATMUL_F32 = 2  # 2: kinds 2-3, the plain-fp32 v2 kernel through the same block (2026-09-09)
ACK_OP_SCHEMA_POINTWISE = 3  # 3: bf16 storage switches in flags bits 2 and 3, same block, 2026-09-10;
                             # 2 was operations 6-23 (reciprocal .. triu) in the same block, 2026-09-07
# 2 since 2026-09-10: `ack_reduce` also accepts the 80-byte block followed
# by one STORAGE FLAGS word (bit 0 the input holds bf16, bit 1 the output
# does). The 80-byte block still means f32 on both sides, so a caller that
# never sends the longer one is bit-for-bit the caller it was.
ACK_OP_SCHEMA_REDUCE = 2
ACK_OP_SCHEMA_ROWWISE = 1
ACK_OP_SCHEMA_LOSS = 1
ACK_OP_SCHEMA_EMBEDDING = 1
POINTWISE_PUSH_BYTES = 120   # the pointwise family's plan block, thirty little-endian u32
REDUCE_PUSH_BYTES = 80       # the reduce family's request block, twenty little-endian u32 (splits and chunk left 0)
ROWWISE_PUSH_BYTES = 16      # the row family's request block: rows, cols, operation, epsilon bits
LOSS_PUSH_BYTES = 20         # the loss family's push block: rows, cols, operation, reduction, normalizer
EMBEDDING_PUSH_BYTES = 16    # the embedding family's push block: n, vocab, dim, operation
# the kind word of ack_matmul_f32: 0/1 are v1 (near-exact, compensated), 2/3 are
# v2 (plain fp32 accumulation, 64x64 tiles); odd kinds are batched
MATMUL_F32_KIND_MM = 0
MATMUL_F32_KIND_BMM = 1
MATMUL_F32_KIND_MM_PLAIN = 2
MATMUL_F32_KIND_BMM_PLAIN = 3
MATMUL_F32_MAX_BATCH = 4096          # the family's batch ceiling (MAX_BATCH * MAX_DIM is exactly MAX_ELEMENTS)
MATMUL_F32_MAX_DIM = 65536           # the family's bound on a view's rows and columns (m, n): addressing
MATMUL_F32_MAX_K = 65536             # and on the reduction length k, separately: measured accuracy, not addressing
MATMUL_F32_MAX_ELEMENTS = 268_435_456  # and at most this many addressed elements per operand view
# The row and embedding capacity bounds, mirroring src/row_ops.rs and
# src/embedding_ops.rs. ADDED 2026-09-08 WITH THE VOCABULARY RAISE, and the
# reason is worth keeping: tests/languages/test_compute_kit_device.py already
# read `ack.__dict__.get("ROWWISE_MAX_COLS", 1 << 16)`, and because no such name
# existed the DEFAULT was always taken -- a hand-copied 65,536 wearing the
# costume of a derived bound, which would have kept pinning the old value
# straight through this raise. Defining the names for real is the fix; bumping
# the literal would have preserved the costume.
# tests/languages/test_compute_kit_row_bounds.py and
# test_compute_kit_embedding_bounds.py hold these equal to the Rust.
ROWWISE_MAX_ROWS = 1 << 24
ROWWISE_MAX_COLS = 1 << 18       # the vocabulary axis: log-softmax reduces over it as COLUMNS
ROWWISE_MAX_ELEMENTS = 1 << 30
EMBEDDING_MAX_N = 1 << 16
EMBEDDING_MAX_VOCAB = 1 << 18
EMBEDDING_MAX_DIM = 1 << 14
EMBEDDING_MAX_ELEMENTS = 1 << 30
ACK_DEVICE_INFO_VERSION = 1
# `AckDispatchStats`'s own version. IT IS NOT THE ABI VERSION AND DOES NOT MOVE
# WITH IT. `ack_dispatch_stats` is absent from every pointer table and changes
# no existing signature, code or struct, so ACK_ABI_VERSION deliberately did
# not move for it; this number plus the struct's byte size ARE the handshake,
# checked the way `AckDeviceInfo`'s are. A library that later changes the
# layout is refused BY NAME by a reader written for version 1.
ACK_DISPATCH_STATS_VERSION = 1

ACK_FEATURE_BITS = {
    "cooperative_matrix": 1,
    "bf16_type": 2,
    "bf16_dot_product": 4,
    "bf16_cooperative_matrix": 8,
    "subgroup_size_control": 16,
    "coopmat_bf16_16x16x16_subgroup": 32,
    "shader_int16": 64,
}
ACK_DTYPE_NAMES = {1: "bf16", 2: "f32"}

# The bf16 kernels' BLOCKING FACTOR, which ack_abi_info still reports. Since
# the -DEDGE=1 variants these are no longer a shape law: an off-tile product
# runs the edge kernel rather than being refused, so nothing here gates on
# them. They are kept because they describe the tile, and because the live
# ABI is pinned against them.
TILE = 128
K_MULTIPLE = 32
_MAX_NBYTES = min((1 << 64) - 1, sys.maxsize)
_POD_SIZES = {
    "b": (1,), "i": (1, 2, 4, 8), "u": (1, 2, 4, 8),
    "f": (2, 4, 8), "c": (8, 16),
}
# Built on demand, never at import: evaluating np.dtype here would make numpy
# a hard import-time dependency again, which is exactly what this file avoids.
def _matmul_dtypes():
    n = _require_numpy("the matmul convenience API")
    return (n.dtype(n.float32), n.dtype(n.uint16))


class AckAbiInfo(ctypes.Structure):
    """What `ack_abi_info` fills in: the library's struct size, ABI version,
    operator bits, shader and build identities, each operator's schema version,
    and the capability versions a training SDK asks for (0 = absent)."""

    _fields_ = [
        ("size", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("operators", ctypes.c_uint64),
        ("shaders", ctypes.c_uint64),
        ("build", ctypes.c_uint64),
        ("op_schema_matmul_bf16", ctypes.c_uint32),
        ("op_schema_cast", ctypes.c_uint32),
        ("capability_streams", ctypes.c_uint32),
        ("capability_rng", ctypes.c_uint32),
        ("capability_optimizer", ctypes.c_uint32),
        ("capability_serialization", ctypes.c_uint32),
        ("op_schema_matmul_f32", ctypes.c_uint32),   # appended in ABI 4
        ("op_schema_pointwise", ctypes.c_uint32),    # appended in ABI 6
        ("op_schema_reduce", ctypes.c_uint32),       # reserved in ABI 6, set in ABI 7
        ("op_schema_rowwise", ctypes.c_uint32),      # reserved in ABI 6, set in ABI 8
        # reserved in ABI 6 for the next operator families, reported as 0 (absent)
        # until each one's C entry exists, so adding one moves no other field
        ("op_schema_loss", ctypes.c_uint32),        # reserved in ABI 6, set in ABI 9
        ("op_schema_embedding", ctypes.c_uint32),
    ]


class AckDeviceInfo(ctypes.Structure):
    """What `ack_device_info` fills in for an open device: PCI ids, device and
    driver UUIDs, driver and API versions, feature bits, the compute limits the
    safe API checks against, what the matmul accepts, and one identity, FNV-1a
    over every numeric field of the struct and the library's shaders (the
    names are display text and are not hashed)."""

    _fields_ = [
        ("size", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("api_version", ctypes.c_uint32),
        ("driver_version_raw", ctypes.c_uint32),
        ("driver_id", ctypes.c_uint32),
        ("vendor_id", ctypes.c_uint32),
        ("device_id", ctypes.c_uint32),
        ("device_type", ctypes.c_uint32),
        ("device_uuid", ctypes.c_uint8 * 16),
        ("driver_uuid", ctypes.c_uint8 * 16),
        ("features", ctypes.c_uint64),
        ("subgroup_size", ctypes.c_uint32),
        ("timestamp_valid_bits", ctypes.c_uint32),
        ("timestamp_period_ns", ctypes.c_float),
        ("max_push_constant_bytes", ctypes.c_uint32),
        ("device_local_bytes", ctypes.c_uint64),
        ("max_storage_buffer_bytes", ctypes.c_uint64),
        ("max_workgroup_invocations", ctypes.c_uint32),
        ("max_workgroup_count", ctypes.c_uint32 * 3),
        ("max_workgroup_size", ctypes.c_uint32 * 3),
        ("matmul_tile", ctypes.c_uint32),
        ("matmul_k_multiple", ctypes.c_uint32),
        ("matmul_layouts", ctypes.c_uint32),
        ("matmul_operand_dtype", ctypes.c_uint32),
        ("matmul_result_dtype", ctypes.c_uint32),
        ("cast_modes", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("identity", ctypes.c_uint64),
        ("device_name", ctypes.c_char * 256),
        ("driver_name", ctypes.c_char * 64),
        ("driver_info", ctypes.c_char * 128),
    ]


class AckDispatchStats(ctypes.Structure):
    """What `ack_dispatch_stats` fills in: what the kit's dispatch instrument
    has counted and, when it was switched on, timed.

    THE COUNTS ALWAYS RUN. `items` is batch items recorded by
    `Context::dispatch`; `commands` is `vkCmdDispatch` commands, which is
    `>= items` because one item with `repeats > 1` records one command per
    repeat; `barriers` is what reached the command buffer.

    `device_ns` IS DEVICE-ELAPSED TIME summed over `timed_items`: barrier
    drain, wave launch, kernel arithmetic and memory traffic together. Wall time outside this span also includes device transfers and queue waits. IT IS NOT a utilisation or occupancy
    figure and not a split of arithmetic from memory traffic. It is 0, with
    `timestamps_enabled` 0, unless `ACK_DISPATCH_TIMESTAMPS` was set when the
    device was opened -- so an absent figure is distinguishable from a measured
    zero, which a bare 0 would not be. `unresolved_flushes` nonzero means
    `device_ns` covers LESS than the run did.
    """

    _fields_ = [
        ("size", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("items", ctypes.c_uint64),
        ("commands", ctypes.c_uint64),
        ("barriers", ctypes.c_uint64),
        ("flushes", ctypes.c_uint64),
        ("timed_flushes", ctypes.c_uint64),
        ("timed_items", ctypes.c_uint64),
        ("device_ns", ctypes.c_uint64),
        ("min_item_ns", ctypes.c_uint64),
        ("max_item_ns", ctypes.c_uint64),
        ("unresolved_flushes", ctypes.c_uint64),
        ("timestamps_enabled", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


def _api_version_string(packed: int) -> str:
    return f"{(packed >> 22) & 0x7F}.{(packed >> 12) & 0x3FF}.{packed & 0xFFF}"


def capability_record(abi: AckAbiInfo, info: AckDeviceInfo) -> dict:
    """A JSON-serialisable record of what a run ran on, for a training
    contract or checkpoint receipt to pin. Everything here is DECLARED by the
    library and the driver. `device.identity` summarises the numeric fields and
    the shaders; a resume gate should compare the whole record and treat a
    differing identity as a refusal, not the identity alone as an acceptance.
    No such gate exists yet; the trainer adapter is where it belongs."""
    return {
        "schema": 1,
        "abi": {
            "version": abi.version,
            "operators": abi.operators,
            "op_schema_matmul_bf16": abi.op_schema_matmul_bf16,
            "op_schema_cast": abi.op_schema_cast,
            "op_schema_matmul_f32": abi.op_schema_matmul_f32,
            "op_schema_pointwise": abi.op_schema_pointwise,
            "reserved_op_schemas": {
                "reduce": abi.op_schema_reduce,
                "rowwise": abi.op_schema_rowwise,
                "loss": abi.op_schema_loss,
                "embedding": abi.op_schema_embedding,
            },
            "shaders": f"{abi.shaders:016x}",
            "build": f"{abi.build:016x}",
            "capabilities": {
                "streams": abi.capability_streams,
                "rng": abi.capability_rng,
                "optimizer": abi.capability_optimizer,
                "serialization": abi.capability_serialization,
            },
        },
        "device": {
            "name": info.device_name.decode("utf-8", errors="replace"),
            "vendor_id": info.vendor_id,
            "device_id": info.device_id,
            "device_type": info.device_type,
            "device_uuid": bytes(info.device_uuid).hex(),
            "driver_uuid": bytes(info.driver_uuid).hex(),
            "driver_id": info.driver_id,
            "driver_version_raw": info.driver_version_raw,
            "driver_name": info.driver_name.decode("utf-8", errors="replace"),
            "driver_info": info.driver_info.decode("utf-8", errors="replace"),
            "api_version": _api_version_string(info.api_version),
            "features": sorted(name for name, bit in ACK_FEATURE_BITS.items() if info.features & bit),
            "subgroup_size": info.subgroup_size,
            "timestamp_valid_bits": info.timestamp_valid_bits,
            "timestamp_period_ns": info.timestamp_period_ns,
            "device_local_bytes": info.device_local_bytes,
            "limits": {
                "max_storage_buffer_bytes": info.max_storage_buffer_bytes,
                "max_push_constant_bytes": info.max_push_constant_bytes,
                "max_workgroup_invocations": info.max_workgroup_invocations,
                "max_workgroup_count": list(info.max_workgroup_count),
                "max_workgroup_size": list(info.max_workgroup_size),
            },
            "matmul": {
                "tile": info.matmul_tile,
                "k_multiple": info.matmul_k_multiple,
                "layouts": [n for i, n in enumerate(("NN", "NT", "TN", "TT")) if info.matmul_layouts & (1 << i)],
                "operand_dtype": ACK_DTYPE_NAMES.get(info.matmul_operand_dtype, info.matmul_operand_dtype),
                "result_dtype": ACK_DTYPE_NAMES.get(info.matmul_result_dtype, info.matmul_result_dtype),
            },
            "cast_modes": info.cast_modes,
            "identity": f"{info.identity:016x}",
        },
    }


class AckError(RuntimeError):
    """A kit call returned an error code; the message is the kit's."""


class AckUnavailable(AckError):
    """No shared library, or no usable Vulkan device: the kit is UNMEASURED here."""


class AckShapeError(AckError):
    """The shape is not a tile multiple; the kit refuses rather than pads."""


class AckPoisoned(AckError):
    """The device's state is unknown after a failed fence wait (ACK-L0-03): every
    later call on it refuses, a free leaves its buffer allocated, and the remedy
    is to close it and start a new process."""


def _kit_error(what: str, rc: int, message: str) -> AckError:
    cls = AckPoisoned if rc == ACK_ERR_POISONED else AckError
    return cls(f"{what}: {message}")


def _checked_nbytes(nbytes: int) -> int:
    if _is_bool(nbytes):
        raise AckError("nbytes must be a positive integer, not a boolean")
    try:
        count = operator.index(nbytes)
    except TypeError as exc:
        raise AckError("nbytes must be a positive integer") from exc
    if not 0 < count <= _MAX_NBYTES:
        raise AckError(f"nbytes must be in 1..{_MAX_NBYTES}")
    return count


def _pod_dtype(dtype) -> np.dtype:
    try:
        result = np.dtype(dtype)
    except (TypeError, ValueError) as exc:
        raise AckError("unsupported buffer dtype") from exc
    # hasobject is recursive: an object field inside a nested record is unsafe
    # too. Keep the remaining accepted layouts explicit and fixed-width.
    if (
        result.hasobject or result.fields is not None or result.subdtype is not None
        or result.itemsize not in _POD_SIZES.get(result.kind, ())
    ):
        raise AckError(f"unsupported buffer dtype {result} is refused; use a plain fixed-width numeric or bool dtype")
    return result


def _library_candidates() -> list[Path]:
    names = {
        "win32": ["alelyon_compute_kit.dll"],
        "darwin": ["libalelyon_compute_kit.dylib"],
    }.get(sys.platform, ["libalelyon_compute_kit.so"])
    out: list[Path] = []
    if os.environ.get("ACK_LIBRARY"):
        out.append(Path(os.environ["ACK_LIBRARY"]))
    # INSTALLED PACKAGE FIRST. A wheel carries the shared library beside this
    # module (or in a `_lib/` subdirectory), and before 2026-09-08 this function
    # searched only the crate's `target/` build tree -- so an installed package
    # could never find its own library, and the SDK export was blocked on it.
    # Both locations are inside the package: the lookup never reaches out of it.
    for name in names:
        out.append(_HERE / name)
        out.append(_HERE / "_lib" / name)
    # then the development checkout, where the library lives in the build tree
    for profile in ("release", "debug"):
        for name in names:
            out.append(_CRATE / "target" / profile / name)
    return out


def library_path() -> Path | None:
    for candidate in _library_candidates():
        if candidate.is_file():
            return candidate
    return None


def to_bf16(x: np.ndarray) -> np.ndarray:
    """float32 -> bf16 bits, nearest-even; preserve special classes, quiet NaNs."""
    np = _require_numpy("to_bf16")
    bits = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    lsb = (bits >> 16) & 1
    rounded = bits + np.uint32(0x7FFF) + lsb
    is_nan = ((bits & 0x7F800000) == 0x7F800000) & ((bits & 0x007FFFFF) != 0)
    # Rounding a NaN payload can carry into the sign or erase its mantissa.
    # Keep the sign/high payload and set the BF16 quiet bit instead.
    return np.where(is_nan, (bits >> 16) | 0x0040, rounded >> 16).astype(np.uint16)


def from_bf16(h: np.ndarray) -> np.ndarray:
    """bf16 bit patterns (uint16) -> float32, exactly."""
    np = _require_numpy("from_bf16")
    return (np.ascontiguousarray(h, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


class _Lib:
    def __init__(self, path: Path) -> None:
        self.cdll = ctypes.CDLL(str(path))
        self.info: AckAbiInfo | None = None
        # Whether THIS library carries the dispatch instrument's entry. False
        # is a supported state, not an error: see `_declare`.
        self.has_dispatch_stats = False
        try:
            self._declare()
        except AttributeError as exc:
            # a library without one of these symbols is a different ABI: refuse
            # by name, as UNMEASURED-with-reason for callers, not a bare error
            raise AckUnavailable(
                f"library {path} lacks a symbol of ABI version {ACK_ABI_VERSION} ({exc}); "
                "rebuild the library or update the binding"
            ) from exc

    def _declare(self) -> None:
        c = self.cdll
        c.ack_abi_info.restype = ctypes.c_int32
        c.ack_abi_info.argtypes = [ctypes.POINTER(AckAbiInfo), ctypes.c_size_t]
        c.ack_device_info.restype = ctypes.c_int32
        c.ack_device_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(AckDeviceInfo), ctypes.c_size_t]
        c.ack_open.restype = ctypes.c_void_p
        c.ack_open.argtypes = []
        c.ack_close.restype = ctypes.c_int32
        c.ack_close.argtypes = [ctypes.c_void_p]
        c.ack_device_name.restype = ctypes.c_int32
        c.ack_device_name.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
        c.ack_live_objects.restype = ctypes.c_int32
        c.ack_live_objects.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64)]
        c.ack_leaked_on_poison.restype = ctypes.c_int32
        c.ack_leaked_on_poison.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)]
        c.ack_last_error.restype = ctypes.c_int32
        c.ack_last_error.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
        c.ack_buffer_alloc.restype = ctypes.c_void_p
        c.ack_buffer_alloc.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        c.ack_buffer_free.restype = ctypes.c_int32
        c.ack_buffer_free.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        c.ack_upload.restype = ctypes.c_int32
        c.ack_upload.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        c.ack_download.restype = ctypes.c_int32
        c.ack_download.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        c.ack_cast.restype = ctypes.c_int32
        c.ack_cast.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int32]
        c.ack_matmul_f32.restype = ctypes.c_int32
        c.ack_matmul_f32.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_uint32, ctypes.c_char_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_double),
        ]
        c.ack_matmul_bf16.restype = ctypes.c_int32
        c.ack_matmul_bf16.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int32, ctypes.c_int32,
            ctypes.POINTER(ctypes.c_double),
        ]
        # ABI 10: the same product with a z grid: device, a, b, c, m, n, k, a_t, b_t,
        # z, zmode (0 batch, 1 K-split), za, zb, zc (elements per z), kz (k per split),
        # lda, ldb (stored row pitches in elements, 0 = the logical width), ms_out
        c.ack_matmul_bf16_z.restype = ctypes.c_int32
        c.ack_matmul_bf16_z.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int32, ctypes.c_int32,
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_uint32, ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_double),
        ]
        # pointwise-f32/v1 (ABI 6): device, x, y, z, out, the 120-byte plan block, its length, ms_out
        c.ack_pointwise.restype = ctypes.c_int32
        c.ack_pointwise.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_char_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_double),
        ]
        # reduce-f32/v1 (ABI 7): device, x, partials (null for a one-pass plan), out, the 80-byte request
        # block, its length, ms_out
        c.ack_reduce.restype = ctypes.c_int32
        c.ack_reduce.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_char_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_double),
        ]
        # rowwise-f32/v2 (ABI 8): device, primary, upstream, gamma, out, the 16-byte push block, its length, ms_out
        c.ack_rowwise.restype = ctypes.c_int32
        c.ack_rowwise.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_char_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_double),
        ]
        # indexed-cross-entropy-f32/v2 (ABI 9). Unlike every other entry this one takes the
        # PARAMETERS and the caller's host targets rather than an opaque push block, because
        # the plan derives host data: it canonicalises the targets and counts the non-ignored
        # ones for the mean normalizer. It validates them, uploads its own canonical image
        # into the targets buffer, and reports the bytes it wrote.
        c.ack_loss.restype = ctypes.c_int32
        # embedding-f32/v2 (ABI 9): device, primary, indices, offsets, out, n, vocab, dim, op,
        # the host ids, their count, uploaded_bytes out, ms_out
        c.ack_embedding.restype = ctypes.c_int32
        c.ack_embedding.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_int64), ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_double),
        ]
        c.ack_loss.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int64,
            ctypes.POINTER(ctypes.c_int64), ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_double),
        ]
        # THE ONE OPTIONAL SYMBOL, and the reason it is optional is a decision
        # rather than a convenience: `ack_dispatch_stats` is in no pointer
        # table, changes no signature and did NOT move ACK_ABI_VERSION, so a
        # library of the SAME declared ABI can legitimately lack it -- an
        # instrument that is missing, not an ABI that is wrong. Caught HERE and
        # not by __init__, so every symbol above stays required: their
        # AttributeError still escapes into the refusal-by-name that
        # `AckUnavailable` contract is. A caller reads `has_dispatch_stats`, or
        # takes the `AckUnavailable` that `Device.dispatch_stats` raises.
        try:
            stats = c.ack_dispatch_stats
        except AttributeError:
            self.has_dispatch_stats = False
            return
        stats.restype = ctypes.c_int32
        stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(AckDispatchStats), ctypes.c_size_t]
        self.has_dispatch_stats = True

    def last_error(self) -> str:
        buf = ctypes.create_string_buffer(1024)
        self.cdll.ack_last_error(buf, len(buf))
        return buf.value.decode("utf-8", errors="replace")

    def abi_info(self) -> AckAbiInfo:
        """The library's ABI description, or AckUnavailable when it has none."""
        try:
            probe = self.cdll.ack_abi_info
        except AttributeError as exc:
            raise AckUnavailable("library has no ack_abi_info: it predates the ABI handshake") from exc
        info = AckAbiInfo()
        rc = probe(ctypes.byref(info), ctypes.sizeof(info))
        if rc != ACK_OK:
            raise AckUnavailable(f"ack_abi_info refused ({rc}): {self.last_error()}")
        return info

    def handshake(self) -> AckAbiInfo:
        """Refuse a library whose ABI version is not this module's, or which
        does not carry the operators this module binds."""
        info = self.abi_info()
        if info.size != ctypes.sizeof(AckAbiInfo):
            raise AckUnavailable(
                f"library reports an AckAbiInfo of size {info.size} bytes, this module's size is {ctypes.sizeof(AckAbiInfo)}; "
                "the layouts differ, so nothing else it says can be read"
            )
        if info.version != ACK_ABI_VERSION:
            raise AckUnavailable(
                f"library ABI version {info.version} is not the {ACK_ABI_VERSION} this module is written for; "
                "rebuild the library or update the binding"
            )
        for name, reported, written_for in (
            ("matmul_bf16", info.op_schema_matmul_bf16, ACK_OP_SCHEMA_MATMUL_BF16),
            ("cast", info.op_schema_cast, ACK_OP_SCHEMA_CAST),
            ("matmul_f32", info.op_schema_matmul_f32, ACK_OP_SCHEMA_MATMUL_F32),
            ("pointwise", info.op_schema_pointwise, ACK_OP_SCHEMA_POINTWISE),
            ("reduce", info.op_schema_reduce, ACK_OP_SCHEMA_REDUCE),
            ("rowwise", info.op_schema_rowwise, ACK_OP_SCHEMA_ROWWISE),
            ("loss", info.op_schema_loss, ACK_OP_SCHEMA_LOSS),
            ("embedding", info.op_schema_embedding, ACK_OP_SCHEMA_EMBEDDING),
        ):
            if reported != written_for:
                raise AckUnavailable(
                    f"library reports {name} schema {reported}; this module is written for schema {written_for} "
                    "(0 means the operator's signature is not declared): rebuild the library or update the binding"
                )
        needed = (ACK_OP_MATMUL_BF16 | ACK_OP_CAST | ACK_OP_MATMUL_F32 | ACK_OP_POINTWISE | ACK_OP_REDUCE
                  | ACK_OP_ROWWISE | ACK_OP_LOSS | ACK_OP_EMBEDDING)
        if info.operators & needed != needed:
            raise AckUnavailable(
                f"library operators {info.operators:#x} lack the matmul and cast, or the pointwise, reduce, "
                f"rowwise, loss or embedding family, this module binds ({needed:#x})"
            )
        self.info = info
        return info


_LIB: _Lib | None = None
_LIB_LOCK = threading.Lock()


def abi_info() -> AckAbiInfo:
    """The loaded library's ABI description (version, operators, shader identity)."""
    lib = _lib()
    return lib.info if lib.info is not None else lib.handshake()


def dispatch_stats_available() -> bool:
    """Whether the loaded library carries `ack_dispatch_stats`.

    False is not a failure and not an ABI mismatch: the ABI version does not
    move for that entry, so a library of the same declared ABI may simply
    predate the instrument. A caller that finds False reports UNMEASURED with
    that reason instead of a number."""
    return bool(getattr(_lib(), "has_dispatch_stats", False))


def _lib() -> _Lib:
    global _LIB
    with _LIB_LOCK:
        if _LIB is None:
            path = library_path()
            if path is None:
                raise AckUnavailable(
                    "alelyon_compute_kit shared library not built; run "
                    "`cargo build --release --manifest-path alelyon/languages/compute_kit/Cargo.toml`"
                )
            lib = _Lib(path)
            lib.handshake()   # raises AckUnavailable; nothing is published on a mismatch
            _LIB = lib
        return _LIB


class Buffer:
    """A device-local byte buffer; explicit free() releases its Device lease."""

    def __init__(self, device: "Device", nbytes: int) -> None:
        self._device = device
        self._handle = None
        self._nbytes = _checked_nbytes(nbytes)
        with device._lock:
            device._require_open()
            handle = device._lib.cdll.ack_buffer_alloc(device._handle, self.nbytes)
            if not handle:
                raise AckError(f"buffer alloc: {device._lib.last_error()}")
            try:
                device._buffers.add(handle)
            except BaseException:
                device._lib.cdll.ack_buffer_free(device._handle, handle)
                raise
            self._handle = handle

    @property
    def nbytes(self) -> int:
        return self._nbytes

    def _require_live(self) -> None:
        if not self._handle:
            raise AckError("buffer has been freed")
        self._device._require_open()

    def upload_bytes(self, data) -> None:
        """Upload raw bytes. The numpy-free core of `upload`.

        `data` is anything supporting the buffer protocol -- bytes, bytearray,
        memoryview, array.array, or a numpy array -- and must be contiguous and
        exactly this buffer's size. This is the entry the kit's own surface is
        built on, so a caller who does not want numpy never needs it.
        """
        with self._device._lock:
            self._require_live()
            view = memoryview(data).cast("B")
            if not view.contiguous:
                raise AckError("upload needs a contiguous buffer")
            if view.nbytes != self.nbytes:
                raise AckError(f"upload of {view.nbytes} bytes into a {self.nbytes}-byte buffer")
            lib = self._device._lib
            source = (ctypes.c_char * view.nbytes).from_buffer_copy(view)
            rc = lib.cdll.ack_upload(self._device._handle, self._handle,
                                     ctypes.cast(source, ctypes.c_void_p), view.nbytes)
            if rc != ACK_OK:
                raise _kit_error("upload", rc, lib.last_error())

    def download_bytes(self) -> bytes:
        """Download this buffer's raw bytes. The numpy-free core of `download`."""
        with self._device._lock:
            self._require_live()
            lib = self._device._lib
            out = (ctypes.c_char * self.nbytes)()
            rc = lib.cdll.ack_download(self._device._handle, self._handle,
                                       ctypes.cast(out, ctypes.c_void_p), self.nbytes)
            if rc != ACK_OK:
                raise _kit_error("download", rc, lib.last_error())
            return bytes(out)

    def upload(self, data: np.ndarray) -> None:
        np = _require_numpy("Buffer.upload")
        with self._device._lock:
            self._require_live()
            array = np.asarray(data)
            _pod_dtype(array.dtype)
            if array.nbytes != self.nbytes:
                raise AckError(f"upload of {array.nbytes} bytes into a {self.nbytes}-byte buffer")
            raw = np.ascontiguousarray(array)
            lib = self._device._lib
            rc = lib.cdll.ack_upload(self._device._handle, self._handle, raw.ctypes.data, raw.nbytes)
            if rc != ACK_OK:
                raise _kit_error("upload", rc, lib.last_error())

    def download(self, dtype) -> np.ndarray:
        np = _require_numpy("Buffer.download")
        with self._device._lock:
            self._require_live()
            dtype = _pod_dtype(dtype)
            if self.nbytes % dtype.itemsize:
                raise AckError(
                    f"buffer size {self.nbytes} is not divisible by dtype itemsize {dtype.itemsize}; "
                    "a transfer requires a whole number of elements"
                )
            out = np.empty(self.nbytes // dtype.itemsize, dtype=dtype)
            lib = self._device._lib
            rc = lib.cdll.ack_download(self._device._handle, self._handle, out.ctypes.data, out.nbytes)
            if rc != ACK_OK:
                raise _kit_error("download", rc, lib.last_error())
            return out

    def free(self) -> None:
        with self._device._lock:
            if self._handle:
                self._device._require_open()
                lib = self._device._lib
                rc = lib.cdll.ack_buffer_free(self._device._handle, self._handle)
                # whatever the kit answered, this object no longer names a live buffer
                self._device._buffers.discard(self._handle)
                self._handle = None
                if rc != ACK_OK:
                    raise _kit_error("free", rc, lib.last_error())

    def __del__(self) -> None:  # best effort; explicit free() is the contract
        try:
            self.free()
        except AckError as exc:
            # a refusal here is a disagreement between this object and the kit;
            # a finaliser cannot raise, so it is reported the way Python reports
            # leaked resources, not swallowed
            try:
                warnings.warn(f"ack.Buffer finaliser: {exc}", ResourceWarning, stacklevel=2)
            except Exception:
                pass
        except Exception:
            pass


class Device:
    """One Vulkan device. Free its buffers before close(); calls are serialized."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._buffers: set[int] = set()
        self._handle = None
        self._lib = lib = _lib()
        self._handle = lib.cdll.ack_open()
        if not self._handle:
            raise AckUnavailable(f"ack_open: {lib.last_error()}")
        try:
            buf = ctypes.create_string_buffer(256)
            rc = lib.cdll.ack_device_name(self._handle, buf, len(buf))
            if rc != ACK_OK:
                raise AckError(f"device name: {lib.last_error()}")
            self.name = buf.value.decode("utf-8", errors="replace")
            info = AckDeviceInfo()
            rc = lib.cdll.ack_device_info(self._handle, ctypes.byref(info), ctypes.sizeof(info))
            if rc != ACK_OK:
                raise AckError(f"device info: {lib.last_error()}")
            if info.size != ctypes.sizeof(AckDeviceInfo) or info.version != ACK_DEVICE_INFO_VERSION:
                raise AckUnavailable(
                    f"library reports an AckDeviceInfo of size {info.size} bytes version {info.version}; "
                    f"this module's size is {ctypes.sizeof(AckDeviceInfo)} bytes version {ACK_DEVICE_INFO_VERSION}"
                )
            self.info = info
            self.identity = info.identity
        except BaseException:
            self.close()
            raise
        # What the last product call reported through the ABI's `ms_out`.
        #
        # `None` means no call has reported yet, or the last one RAISED: every
        # entry clears it before validating, so a stale duration can never be
        # read as this call's.
        #
        # SINCE DEFERRED SUBMISSION (increment 5) A SUCCESSFUL PRODUCT CALL
        # LEAVES IT NaN, NOT A DURATION. `matmul` and `matmul_f32` RECORD their
        # dispatch; it is submitted at the kit's next flush (the `download`
        # each of them makes right afterwards is that flush), so at the moment
        # the entry returns there is no interval to report. NaN is the ABI's
        # existing "accepted, duration unknown" value (`ACK_ERR_TIMING`'s
        # sibling), and `ack_matmul_bf16`/`ack_matmul_f32` write it
        # unconditionally on `ACK_OK`.
        #
        # SO IT IS NOT A TIMING SOURCE. A caller that wants a device duration
        # measures the wall clock around a call that ends in a readback, or
        # uses the Rust `Context::dispatch_timed`, which still times its own
        # submission. `math.isnan(dev.last_ms)` is the pinned contract
        # (tests/languages/test_compute_kit_ffi.py); `> 0` was the contract
        # before increment 5 and is now false for every successful call.
        self.last_ms: float | None = None

    def capability_record(self) -> dict:
        """What this device and library declare, JSON-serialisable (see `capability_record`)."""
        with self._lock:
            self._require_open()
            return capability_record(self._lib.info if self._lib.info is not None else self._lib.handshake(), self.info)

    def _require_open(self) -> None:
        if not self._handle:
            raise AckError("device is closed")

    def close(self) -> None:
        """Free the device only on ACK_OK; refusal keeps the handle valid."""
        with self._lock:
            if self._handle:
                if self._buffers:
                    raise AckError("device has live buffers still allocated; free them before close")
                rc = self._lib.cdll.ack_close(self._handle)
                if rc == ACK_ERR_CLOSED:
                    # the native device is already gone (closed underneath this
                    # object); converge to closed rather than wedge, and say so
                    self._handle = None
                    raise AckError(f"close: {self._lib.last_error()}")
                if rc != ACK_OK:
                    raise AckError(f"close: {self._lib.last_error()}")
                self._handle = None

    def __enter__(self) -> "Device":
        with self._lock:
            self._require_open()
            return self

    def __exit__(self, *exc) -> None:
        self.close()

    def buffer(self, nbytes: int) -> Buffer:
        return Buffer(self, nbytes)

    def matmul(self, a: np.ndarray, b: np.ndarray, *, a_t: bool = False, b_t: bool = False) -> np.ndarray:
        """C[M,N] = op(A) . op(B) in BF16 with f32 accumulation; returns float32.

        `a` is (M, K) as stored, or (K, M) when `a_t`; `b` is (K, N), or (N, K)
        when `b_t`. float32 inputs are rounded to BF16 first; uint16 inputs are
        taken as BF16 bit patterns. Other dtypes are refused: callers must
        explicitly choose any narrowing conversion. Layout flags must be bool.

        `last_ms` IS NaN ON SUCCESS, NOT A DURATION: since deferred submission
        the entry records its dispatch and the kit submits it at the next
        flush, so the ABI has no interval to report (see `Device.last_ms`).
        """
        with self._lock:
            self._require_open()
            self.last_ms = None
            if not isinstance(a_t, (bool, np.bool_)) or not isinstance(b_t, (bool, np.bool_)):
                raise AckShapeError("matmul layout flags must be bool")
            a, b = np.asarray(a), np.asarray(b)
            if any(operand.dtype not in _matmul_dtypes() for operand in (a, b)):
                raise AckError("matmul operand dtype must be native float32 or uint16 BF16 bits")
            if a.ndim != 2 or b.ndim != 2:
                raise AckShapeError("matmul takes 2-D operands")
            m, k = (a.shape[1], a.shape[0]) if a_t else a.shape
            k2, n = (b.shape[1], b.shape[0]) if b_t else b.shape
            if k != k2:
                raise AckShapeError(f"reduction mismatch: A gives K={k}, B gives K={k2}")
            if not all(0 < dimension <= 0xFFFFFFFF for dimension in (m, n, k)):
                raise AckShapeError("matmul dimensions must be positive uint32 values")
            a16 = a if a.dtype == np.uint16 else to_bf16(a)
            b16 = b if b.dtype == np.uint16 else to_bf16(b)
            # Register each allocation as soon as it succeeds. Cleanup cannot
            # depend on CPython refcounting or on reaching the final allocation.
            with ExitStack() as cleanup:
                ba = self.buffer(a16.nbytes)
                cleanup.callback(ba.free)
                bb = self.buffer(b16.nbytes)
                cleanup.callback(bb.free)
                bc = self.buffer(m * n * 4)
                cleanup.callback(bc.free)
                ba.upload(a16)
                bb.upload(b16)
                ms = ctypes.c_double(0.0)
                rc = self._lib.cdll.ack_matmul_bf16(
                    self._handle, ba._handle, bb._handle, bc._handle, m, n, k, int(a_t), int(b_t), ctypes.byref(ms)
                )
                if rc == ACK_ERR_SHAPE:
                    raise AckShapeError(self._lib.last_error())
                if rc != ACK_OK:
                    raise _kit_error("matmul", rc, self._lib.last_error())
                result = bc.download(np.float32).reshape(m, n)
                self.last_ms = ms.value
                return result

    def live_objects(self) -> int:
        """Vulkan objects the device's context has created and not destroyed
        (ACK-L0-05), counted by the runtime's own create and destroy wrappers;
        the kernels built at open count four each. A poisoned device keeps its
        leaked objects counted."""
        with self._lock:
            self._require_open()
            out = ctypes.c_int64(0)
            rc = self._lib.cdll.ack_live_objects(self._handle, ctypes.byref(out))
            if rc != ACK_OK:
                raise _kit_error("live_objects", rc, self._lib.last_error())
            return int(out.value)

    def dispatch_stats(self) -> AckDispatchStats:
        """What the dispatch instrument has counted and, if it was switched on,
        timed for THIS device (`AckDispatchStats`).

        RAISES `AckUnavailable` WHEN THE LIBRARY HAS NO SUCH ENTRY, and that is
        a first-class answer rather than a failure: the ABI version did not move
        for this entry, so a library of the same declared ABI may predate it.
        Check `ack.dispatch_stats_available()` first, or catch this and report
        UNMEASURED with the reason.

        The struct's own `size` and `version` are the handshake -- there is no
        ABI bump behind them -- so a mismatch is refused by name here, exactly
        as `AckDeviceInfo`'s is at open.

        IT DOES NOT FLUSH. Work recorded and unsubmitted at the moment of the
        call is in `items` and `commands` but not yet in `device_ns`; take the
        reading after a readback has drained the batch."""
        with self._lock:
            self._require_open()
            if not getattr(self._lib, "has_dispatch_stats", False):
                raise AckUnavailable(
                    "library has no ack_dispatch_stats: it predates the dispatch instrument "
                    "(the ABI version does not move for that entry, so this is not an ABI mismatch)"
                )
            stats = AckDispatchStats()
            rc = self._lib.cdll.ack_dispatch_stats(
                self._handle, ctypes.byref(stats), ctypes.sizeof(stats)
            )
            if rc != ACK_OK:
                raise _kit_error("dispatch_stats", rc, self._lib.last_error())
            if stats.size != ctypes.sizeof(AckDispatchStats) or stats.version != ACK_DISPATCH_STATS_VERSION:
                raise AckUnavailable(
                    f"library reports an AckDispatchStats of size {stats.size} bytes version {stats.version}; "
                    f"this module's size is {ctypes.sizeof(AckDispatchStats)} bytes version "
                    f"{ACK_DISPATCH_STATS_VERSION}"
                )
            return stats

    def leaked_on_poison(self) -> int:
        """Objects left allocated on purpose because the device is poisoned
        (ACK-L0-03); 0 on a healthy device."""
        with self._lock:
            self._require_open()
            out = ctypes.c_uint64(0)
            rc = self._lib.cdll.ack_leaked_on_poison(self._handle, ctypes.byref(out))
            if rc != ACK_OK:
                raise _kit_error("leaked_on_poison", rc, self._lib.last_error())
            return int(out.value)

    def matmul_f32(self, a, b, *, plain: bool = False) -> np.ndarray:
        """C = A . B in FP32 on the device (matmul-f32/v1): 2-D float32 operands
        in C or Fortran order (a transposed view is read in its stored layout),
        a row-major float32 result, no BF16 rounding anywhere. The family's
        bounds apply (m and n at most 65,536, the reduction length k at most
        65,536, at most 268,435,456 elements per operand) and the kit refuses
        beyond them by name (`AckShapeError`).
        Fixed-fixture numerical evidence only: see MATMUL_KERNELS.md.

        `last_ms` IS NaN ON SUCCESS, NOT A DURATION: since deferred submission
        the entry records its dispatch and the kit submits it at the next
        flush, so the ABI has no interval to report (see `Device.last_ms`)."""
        with self._lock:
            self._require_open()
            self.last_ms = None
            a, b = np.asarray(a), np.asarray(b)
            if a.dtype != np.float32 or b.dtype != np.float32:
                raise AckError("matmul_f32 operands must be float32")
            if a.ndim != 2 or b.ndim != 2:
                raise AckShapeError("matmul_f32 takes 2-D operands")
            m, k = a.shape
            k2, n = b.shape
            if k != k2:
                raise AckShapeError(f"reduction mismatch: A gives K={k}, B gives K={k2}")
            if not all(0 < dimension <= MATMUL_F32_MAX_DIM for dimension in (m, n)):
                raise AckShapeError(f"matmul_f32 dimensions must be in 1..{MATMUL_F32_MAX_DIM}")
            # k is bounded separately: it is the reduction length, and its ceiling
            # is an accuracy measurement rather than the addressing arithmetic
            if not 0 < k <= MATMUL_F32_MAX_K:
                raise AckShapeError(f"matmul_f32 reduction length must be in 1..{MATMUL_F32_MAX_K}")
            if max(m * k, k * n, m * n) > MATMUL_F32_MAX_ELEMENTS:
                raise AckShapeError(f"matmul_f32 addresses at most {MATMUL_F32_MAX_ELEMENTS} elements per operand")
            for name, operand in (("A", a), ("B", b)):
                if not (operand.flags.c_contiguous or operand.flags.f_contiguous):
                    raise AckShapeError(f"matmul_f32 operand {name} must be C- or Fortran-contiguous")
            # the stored bytes go up as they are, and the plan carries each operand's element
            # strides from its LAYOUT, never from numpy's strides: numpy reports a size-1 axis
            # (x[None, :], x[:, None]) as contiguous with an arbitrary, often zero, stride
            def layout(operand, rows, cols):
                if operand.flags.c_contiguous:
                    return np.ascontiguousarray(operand), cols, 1
                return np.ascontiguousarray(operand.T), 1, rows

            stored_a, a_rs, a_cs = layout(a, m, k)
            stored_b, b_rs, b_cs = layout(b, k, n)
            words = [1, m, n, k, 0, 1, a_rs, a_cs, 0, 1, b_rs, b_cs, 0, 1, n, 1]
            plan = struct.pack("<16I", *words)
            with ExitStack() as cleanup:
                ba = self.buffer(a.nbytes)
                cleanup.callback(ba.free)
                bb = self.buffer(b.nbytes)
                cleanup.callback(bb.free)
                bc = self.buffer(m * n * 4)
                cleanup.callback(bc.free)
                ba.upload(stored_a)
                bb.upload(stored_b)
                ms = ctypes.c_double(0.0)
                kind = MATMUL_F32_KIND_MM_PLAIN if plain else MATMUL_F32_KIND_MM
                rc = self._lib.cdll.ack_matmul_f32(
                    self._handle, ba._handle, bb._handle, bc._handle, kind, plan, len(plan), ctypes.byref(ms)
                )
                if rc in (ACK_ERR_SHAPE, ACK_ERR_SIZE):
                    raise AckShapeError(self._lib.last_error())
                if rc != ACK_OK:
                    raise _kit_error("matmul_f32", rc, self._lib.last_error())
                result = bc.download(np.float32).reshape(m, n)
                self.last_ms = ms.value
                return result
