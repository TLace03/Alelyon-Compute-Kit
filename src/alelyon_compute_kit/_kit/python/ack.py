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
ACK_ERR_UNSUPPORTED_OP = -11   # the operator's kernels could not be built on this device (BF16, or for ack_matmul_f16 fp16, cooperative matrices absent)
ACK_ERR_TIMING = -12     # declared, and unreachable since deferred submission: no C entry times a dispatch any more, so every product entry reports NaN in ms_out

ACK_ABI_VERSION = 14     # the C ABI this module is written for; ack_abi_info must agree exactly
ACK_OP_MATMUL_BF16 = 1
ACK_OP_CAST = 2
ACK_OP_MATMUL_F32 = 4    # matmul-f32/v1: ack_matmul_f32 with the 64-byte plan block
ACK_OP_POINTWISE = 8     # pointwise-f32/v1: ack_pointwise with the 120-byte plan block (ABI 6)
ACK_OP_REDUCE = 16       # reduce-f32/v1: ack_reduce with the 80-byte request block (ABI 7)
ACK_OP_ROWWISE = 32      # rowwise-f32/v2: ack_rowwise with the kernel's 16-byte push block (ABI 8)
ACK_OP_LOSS = 64         # indexed-cross-entropy-f32/v2: ack_loss, which validates and uploads its own targets (ABI 9)
ACK_OP_EMBEDDING = 128   # embedding-f32/v2: ack_embedding, which derives and uploads its index and CSR images (ABI 9)
ACK_OP_MATMUL_F16 = 256  # ack_matmul_f16 and ack_matmul_f16_z: the bf16 pair with fp16 operands (ABI 11)
ACK_OP_INDEX = 512       # index/v1: ack_index, gather/scatter/scatter-add with the indices kept on the device (ABI 12)
ACK_OP_CONVERT = 1024    # convert/v1: ack_convert, a dtype conversion over strided views, rounded as c10 rounds (ABI 13)
ACK_OP_INTWISE = 2048    # intwise/v1: ack_intwise, torch's integer and bool pointwise arithmetic (ABI 14)
# the operator schemas this module is written for; the handshake requires each
# exactly (0 means the library does not declare that operator's signature)
ACK_OP_SCHEMA_MATMUL_BF16 = 2  # 2: ack_matmul_bf16_z, the z-grid product beside ack_matmul_bf16 (ABI 10, 2026-09-09)
ACK_OP_SCHEMA_CAST = 1
ACK_OP_SCHEMA_MATMUL_F32 = 2  # 2: kinds 2-3, the plain-fp32 v2 kernel through the same block (2026-09-09)
ACK_OP_SCHEMA_POINTWISE = 8  # 8: operations 43-46 (minimum, maximum, relu, threshold_backward),
                             # selections decided on the bits, same block, 2026-10-16;
                             # 7 was binary16 storage in flags bits 10-13, same operations and block,
                             # 2026-09-29;
                             # 6 was operations 40-42 (tanh, log, xlogy), in correctly rounded adds
                             # and multiplies only, same block, 2026-09-28;
                             # 5 was operations 37-39 (silu_backward, x*silu(y), the SiLU gate's
                             # gradient), same block, 2026-09-28;
                             # 4 was operations 24-36 (exp, round, comparisons, logical, where)
                             # and bool storage in flags bits 6-9, same block, 2026-09-22;
                             # 3 was the bf16 storage switches in flags bits 2-5, 2026-09-10;
                             # 2 was operations 6-23 (reciprocal .. triu) in the same block, 2026-09-07
# 2 since 2026-09-10: `ack_reduce` also accepts the 80-byte block followed
# by one STORAGE FLAGS word (bit 0 the input holds bf16, bit 1 the output
# does). The 80-byte block still means f32 on both sides, so a caller that
# never sends the longer one is bit-for-bit the caller it was.
# 3 since 2026-09-27: operation 3, AbsMax (the largest |x|, the norm of order
# infinity), in the same block.
# 4 since 2026-09-29: storage bit 2, a float64 sum (input, partials and output
# float64) on v1's plan with a binary64 add in integer words.
ACK_OP_SCHEMA_REDUCE = 4
ACK_OP_SCHEMA_ROWWISE = 1
ACK_OP_SCHEMA_LOSS = 1
ACK_OP_SCHEMA_EMBEDDING = 1
ACK_OP_SCHEMA_MATMUL_F16 = 1  # 1: both entries, the z grid included (ABI 11, 2026-09-26)
ACK_OP_SCHEMA_INDEX = 3  # 3: a long fiber's scatter (elected) or int32/int64 scatter-add
                         # (integer atomics) an element an invocation, same request and results
                         # (2026-09-29);
                         # 2: a 1- or 2-byte scatter output in any layout, written with integer
                         # atomics where fibers share words, not refused, and the float64
                         # scatter-add in binary64 (2026-09-28);
                         # 1: gather, scatter, scatter-add over strided views (ABI 12, 2026-09-26)
ACK_OP_SCHEMA_CONVERT = 2  # 2: a strided 1- or 2-byte output written with atomics, not refused;
                           # 1: every storage pair but float -> integer (ABI 13, 2026-09-27)
ACK_OP_SCHEMA_INTWISE = 6  # 6: a rank row longer than 1,024 with an int64 or int32 output ranked
                           # by the radix module (radix/v1), past the walk's row and work limits
                           # (2026-09-29);
                           # 5: operations 27 rank_asc and 28 rank_desc, each element's stable rank
                           # along the last dimension, x of any storage code (2026-09-28);
                           # 4: operations 25 tril and 26 triu over the last two dimensions, the
                           # diagonal in the scalar word (2026-09-28); 3: operation 24, cumsum along
                           # the last dimension, an integer C (2026-09-28); 2: C = float64, the binary64 module: add, sub (alpha
                           # +-1.0), mul, div, reciprocal, neg, abs, fill and the comparisons, operands
                           # of any storage code (2026-09-28); 1: add .. arange on
                           # i64/i32/i16/i8/u8/bool (ABI 14)
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
    "shader_float16": 128,                      # ABI 11
    "coopmat_f16_16x16x16_subgroup": 256,       # ABI 11
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
def _coopmat_shape(a: np.ndarray, b: np.ndarray, a_t, b_t) -> tuple[int, int, int]:
    """(m, n, k) of a 16-bit cooperative-matrix product, or a refusal by name:
    bool layout flags, 2-D operands, agreeing reductions and positive uint32
    dimensions. Shared by `Device.matmul` and `Device.matmul_f16`."""
    if not isinstance(a_t, (bool, np.bool_)) or not isinstance(b_t, (bool, np.bool_)):
        raise AckShapeError("matmul layout flags must be bool")
    if a.ndim != 2 or b.ndim != 2:
        raise AckShapeError("matmul takes 2-D operands")
    m, k = (a.shape[1], a.shape[0]) if a_t else a.shape
    k2, n = (b.shape[1], b.shape[0]) if b_t else b.shape
    if k != k2:
        raise AckShapeError(f"reduction mismatch: A gives K={k}, B gives K={k2}")
    if not all(0 < dimension <= 0xFFFFFFFF for dimension in (m, n, k)):
        raise AckShapeError("matmul dimensions must be positive uint32 values")
    return m, n, k


def _matmul_dtypes():
    n = _require_numpy("the matmul convenience API")
    return (n.dtype(n.float32), n.dtype(n.uint16))


def _index_storage(dtype, bf16: bool) -> int:
    """index/v1's storage code for a numpy dtype (src/index_ops.rs `Dtype`):
    bf16 is uint16 bits named by `bf16=True`; bool is uint8 storage; uint16 is
    int16 storage, whose wrapping add is the same bits."""
    n = _require_numpy("the index convenience API")
    dtype = n.dtype(dtype)
    if bf16:
        if dtype != n.uint16:
            raise AckError("bf16=True names uint16 arrays of bf16 bits")
        return 1
    codes = {
        n.dtype(n.float32): 0, n.dtype(n.float16): 2, n.dtype(n.int64): 3, n.dtype(n.int32): 4,
        n.dtype(n.int16): 5, n.dtype(n.uint16): 5, n.dtype(n.int8): 6, n.dtype(n.uint8): 7,
        n.dtype(n.bool_): 7, n.dtype(n.float64): 8,
    }
    if dtype not in codes or dtype.byteorder == ">":
        raise AckError(f"the index family does not carry {dtype}")
    return codes[dtype]


#: convert/v1's storage codes (src/convert_ops.rs `Code`), by name
CONVERT_CODES = {"float32": 0, "bfloat16": 1, "float16": 2, "int64": 3, "int32": 4, "int16": 5,
                 "int8": 6, "uint8": 7, "float64": 8, "bool": 9}


#: intwise/v1's operations (src/intwise_ops.rs `IntOp`), by name
INTWISE_OPS = {"add": 0, "sub": 1, "mul": 2, "and": 3, "or": 4, "xor": 5, "shl": 6, "shr": 7,
               "eq": 8, "ne": 9, "lt": 10, "le": 11, "gt": 12, "ge": 13, "neg": 14, "abs": 15,
               "fill": 16, "arange": 17, "div_trunc": 18, "div_floor": 19, "remainder": 20, "fmod": 21,
               "div": 22, "reciprocal": 23, "cumsum": 24, "tril": 25, "triu": 26, "rank_asc": 27,
               "rank_desc": 28}
_INTWISE_DTYPES = ("int64", "int32", "int16", "int8", "uint8", "bool")
# with C = float64 (schema 2) an operand may also be a float; bf16 has no numpy dtype
_INTWISE_F64_OPERANDS = ("float64", "float32", "float16") + _INTWISE_DTYPES


def _intwise_code(dtype, binary64: bool = False) -> int:
    n = _require_numpy("the intwise convenience API")
    dtype = n.dtype(dtype)
    if dtype.byteorder == ">" or dtype.name not in (_INTWISE_F64_OPERANDS if binary64 else _INTWISE_DTYPES):
        raise AckError(f"the intwise family does not carry {dtype}"
                       + ("" if binary64 else " where C is an integer type"))
    return CONVERT_CODES[dtype.name]


def _f64_bits(value) -> int:
    """An f64's bits as the signed 64-bit word the request's scalar and alpha carry."""
    return struct.unpack("<q", struct.pack("<d", float(value)))[0]


def _convert_code(dtype, bf16: bool) -> int:
    """convert/v1's storage code for a numpy dtype: bf16 is uint16 bits named
    by `bf16=True`; uint16 otherwise has no code (int16 storage would convert
    its large values wrongly)."""
    n = _require_numpy("the convert convenience API")
    dtype = n.dtype(dtype)
    if bf16:
        if dtype != n.uint16:
            raise AckError("bf16=True names uint16 arrays of bf16 bits")
        return CONVERT_CODES["bfloat16"]
    if dtype.byteorder == ">" or dtype.name not in CONVERT_CODES:
        raise AckError(f"the convert family does not carry {dtype}")
    return CONVERT_CODES[dtype.name]


def _index_dim(dim: int, ndim: int, name: str) -> int:
    if not 1 <= ndim <= 4:
        raise AckShapeError(f"{name} takes arrays of 1 to 4 dimensions, not {ndim}")
    if not -ndim <= dim < ndim:
        raise AckShapeError(f"{name} dim {dim} is out of range for {ndim} dimensions")
    return dim % ndim


def _index_shapes(index_shape, tensor_shape, dim: int, name: str, tensor: str) -> None:
    """torch's rule: the index has the tensor's rank and is no larger off `dim`."""
    if len(index_shape) != len(tensor_shape):
        raise AckShapeError(f"{name} index rank {len(index_shape)} is not {tensor}'s {len(tensor_shape)}")
    for d, (i, t) in enumerate(zip(index_shape, tensor_shape)):
        if d != dim and i > t:
            raise AckShapeError(f"{name} index {tuple(index_shape)} exceeds {tensor} {tuple(tensor_shape)} off dim {dim}")


def _c_strides(shape) -> list:
    """Row-major element strides."""
    strides, step = [0] * len(shape), 1
    for k in range(len(shape) - 1, -1, -1):
        strides[k] = step
        step *= shape[k]
    return strides


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
        ("op_schema_matmul_f16", ctypes.c_uint32),   # appended in ABI 11
        ("op_schema_index", ctypes.c_uint32),        # ABI 11's reserved word, set in ABI 12; no field moved
        ("op_schema_convert", ctypes.c_uint32),      # appended in ABI 13
        ("op_schema_intwise", ctypes.c_uint32),      # ABI 13's zero reserved word, set in ABI 14
    ]


class AckIndexRequest(ctypes.Structure):
    """One `ack_index` call's request (index/v1, ABI 12): the operation, the
    storage code, the iteration shape with its indexed dimension, and each
    operand's element offset and strides; the library plans the dispatch
    itself. 200 bytes, the layout src/ffi.rs pins."""

    _fields_ = [
        ("op", ctypes.c_uint32),
        ("dtype", ctypes.c_uint32),
        ("ndim", ctypes.c_uint32),
        ("dim", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("status_slot", ctypes.c_uint32),
        ("shape", ctypes.c_uint64 * 4),
        ("index_offset", ctypes.c_uint64),
        ("index_strides", ctypes.c_uint64 * 4),
        ("src_offset", ctypes.c_uint64),
        ("src_strides", ctypes.c_uint64 * 4),
        ("out_offset", ctypes.c_uint64),
        ("out_strides", ctypes.c_uint64 * 4),
        ("bound", ctypes.c_uint64),
        ("value", ctypes.c_uint64),
        ("alpha_bits", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


class AckConvertRequest(ctypes.Structure):
    """One `ack_convert` call's request (convert/v1, ABI 13): the input and
    output storage codes (`CONVERT_CODES`), the shape, and each operand's
    element offset and strides; the library plans the dispatch itself. 128
    bytes, the layout src/ffi.rs pins."""

    _fields_ = [
        ("src_code", ctypes.c_uint32),
        ("out_code", ctypes.c_uint32),
        ("ndim", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("shape", ctypes.c_uint64 * 4),
        ("in_offset", ctypes.c_uint64),
        ("in_strides", ctypes.c_uint64 * 4),
        ("out_offset", ctypes.c_uint64),
        ("out_strides", ctypes.c_uint64 * 4),
    ]


class AckIntwiseRequest(ctypes.Structure):
    """One `ack_intwise` call's request (intwise/v1, ABI 14): the operation
    (`INTWISE_OPS`), the storage codes of x, y, the output and C (the common
    dtype torch computes in), flags (bit 0: y is `scalar`), the shape, each
    operand's element offset and strides, the scalar and alpha. 200 bytes,
    the layout src/ffi.rs pins."""

    _fields_ = [
        ("op", ctypes.c_uint32),
        ("x_code", ctypes.c_uint32),
        ("y_code", ctypes.c_uint32),
        ("out_code", ctypes.c_uint32),
        ("compute_code", ctypes.c_uint32),
        ("ndim", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("shape", ctypes.c_uint64 * 4),
        ("x_offset", ctypes.c_uint64),
        ("x_strides", ctypes.c_uint64 * 4),
        ("y_offset", ctypes.c_uint64),
        ("y_strides", ctypes.c_uint64 * 4),
        ("out_offset", ctypes.c_uint64),
        ("out_strides", ctypes.c_uint64 * 4),
        ("scalar", ctypes.c_int64),
        ("alpha", ctypes.c_int64),
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
            "op_schema_matmul_f16": abi.op_schema_matmul_f16,
            "op_schema_index": abi.op_schema_index,
            "op_schema_convert": abi.op_schema_convert,
            "op_schema_intwise": abi.op_schema_intwise,
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
        # ABI 11: the fp16 pair, the bf16 pair's exact signatures
        c.ack_matmul_f16.restype = ctypes.c_int32
        c.ack_matmul_f16.argtypes = list(c.ack_matmul_bf16.argtypes)
        c.ack_matmul_f16_z.restype = ctypes.c_int32
        c.ack_matmul_f16_z.argtypes = list(c.ack_matmul_bf16_z.argtypes)
        # ABI 12: index/v1: device, src, index, out, status, the request, its size, ms_out
        c.ack_index.restype = ctypes.c_int32
        c.ack_index.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.POINTER(AckIndexRequest), ctypes.c_size_t, ctypes.POINTER(ctypes.c_double),
        ]
        # ABI 13: convert/v1: device, src, out, the request, its size, ms_out
        c.ack_convert.restype = ctypes.c_int32
        c.ack_convert.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.POINTER(AckConvertRequest), ctypes.c_size_t, ctypes.POINTER(ctypes.c_double),
        ]
        # ABI 14: intwise/v1: device, x, y, out, the request, its size, ms_out
        c.ack_intwise.restype = ctypes.c_int32
        c.ack_intwise.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.POINTER(AckIntwiseRequest), ctypes.c_size_t, ctypes.POINTER(ctypes.c_double),
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
            ("matmul_f16", info.op_schema_matmul_f16, ACK_OP_SCHEMA_MATMUL_F16),
            ("index", info.op_schema_index, ACK_OP_SCHEMA_INDEX),
            ("convert", info.op_schema_convert, ACK_OP_SCHEMA_CONVERT),
            ("intwise", info.op_schema_intwise, ACK_OP_SCHEMA_INTWISE),
        ):
            if reported != written_for:
                raise AckUnavailable(
                    f"library reports {name} schema {reported}; this module is written for schema {written_for} "
                    "(0 means the operator's signature is not declared): rebuild the library or update the binding"
                )
        needed = (ACK_OP_MATMUL_BF16 | ACK_OP_CAST | ACK_OP_MATMUL_F32 | ACK_OP_POINTWISE | ACK_OP_REDUCE
                  | ACK_OP_ROWWISE | ACK_OP_LOSS | ACK_OP_EMBEDDING | ACK_OP_MATMUL_F16 | ACK_OP_INDEX
                  | ACK_OP_CONVERT | ACK_OP_INTWISE)
        if info.operators & needed != needed:
            raise AckUnavailable(
                f"library operators {info.operators:#x} lack the matmul and cast, or the pointwise, reduce, "
                f"rowwise, loss, embedding, fp16 matmul, index, convert or intwise family, this module binds "
                f"({needed:#x})"
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
            a, b = np.asarray(a), np.asarray(b)
            if any(operand.dtype not in _matmul_dtypes() for operand in (a, b)):
                raise AckError("matmul operand dtype must be native float32 or uint16 BF16 bits")
            m, n, k = _coopmat_shape(a, b, a_t, b_t)
            a16 = a if a.dtype == np.uint16 else to_bf16(a)
            b16 = b if b.dtype == np.uint16 else to_bf16(b)
            return self._coopmat_product("ack_matmul_bf16", "matmul", a16, b16, m, n, k, a_t, b_t)

    def matmul_f16(self, a: np.ndarray, b: np.ndarray, *, a_t: bool = False, b_t: bool = False) -> np.ndarray:
        """`matmul` through the fp16 cooperative-matrix modules (ABI 11):
        the same shapes, layout flags and f32 result, with fp16 operands, for
        a device whose tensor cores take fp16 and not bf16 (NVIDIA Turing).

        A native float16 array is sent as it is. A float32 array is rounded to
        fp16 by numpy (round to nearest even), and REFUSED if a finite value
        falls outside fp16's range (|x| > 65504): that value would reach the
        device as an infinity, and the product would be wrong without an error.
        A device without fp16 cooperative matrices refuses by code
        (ACK_ERR_UNSUPPORTED_OP). `last_ms` is NaN on success, as for `matmul`.
        """
        with self._lock:
            self._require_open()
            self.last_ms = None
            a, b = np.asarray(a), np.asarray(b)
            halves = []
            for label, operand in (("A", a), ("B", b)):
                if operand.dtype == np.float16:
                    halves.append(np.ascontiguousarray(operand))
                elif operand.dtype == np.float32:
                    # an overflow is detected and refused just below, so numpy's
                    # own warning about the same cast would only repeat it
                    with np.errstate(over="ignore"):
                        rounded = operand.astype(np.float16)
                    overflow = np.isfinite(operand) & ~np.isfinite(rounded)
                    if overflow.any():
                        raise AckError(
                            f"matmul_f16 operand {label} holds {int(overflow.sum())} finite value(s) outside "
                            "fp16's range (|x| > 65504), which would reach the device as infinity"
                        )
                    halves.append(np.ascontiguousarray(rounded))
                else:
                    raise AckError("matmul_f16 operand dtype must be native float16 or float32")
            m, n, k = _coopmat_shape(a, b, a_t, b_t)
            return self._coopmat_product("ack_matmul_f16", "matmul_f16", halves[0], halves[1], m, n, k, a_t, b_t)

    # ---- index/v1 (ABI 12) ------------------------------------------------------

    def gather(self, src, dim: int, index, *, bf16: bool = False) -> np.ndarray:
        """`torch.gather(src, dim, index)` on the device (index/v1): arrays of
        rank 1-4, `index` int64 or int32 with `index.shape[d] <= src.shape[d]`
        off `dim`. The result has `index`'s shape and `src`'s dtype, copied bit
        for bit. An index out of range raises IndexError (the kit never reads
        with it). `bf16=True` names a uint16 array as bf16 bits."""
        np = _require_numpy("Device.gather")
        src, index = np.asarray(src), np.asarray(index)
        dim = _index_dim(dim, src.ndim, "gather")
        _index_shapes(index.shape, src.shape, dim, "gather", "src")
        out = np.empty(index.shape, src.dtype)
        return self._index("gather", 0, src, index, out, index.shape, dim, src.shape[dim],
                           _c_strides(index.shape), _c_strides(src.shape), _c_strides(index.shape), bf16=bf16)

    def index_select(self, src, dim: int, index, *, bf16: bool = False) -> np.ndarray:
        """`torch.index_select(src, dim, index)` on the device: `index` 1-D int64
        or int32, the result `src`'s shape with `len(index)` along `dim`."""
        np = _require_numpy("Device.index_select")
        src, index = np.asarray(src), np.asarray(index)
        dim = _index_dim(dim, src.ndim, "index_select")
        if index.ndim != 1:
            raise AckShapeError("index_select takes a 1-D index")
        shape = src.shape[:dim] + (index.shape[0],) + src.shape[dim + 1:]
        out = np.empty(shape, src.dtype)
        idx_st = [0] * len(shape)
        idx_st[dim] = 1
        return self._index("index_select", 0, src, index, out, shape, dim, src.shape[dim],
                           idx_st, _c_strides(src.shape), _c_strides(shape), bf16=bf16)

    def scatter(self, out, dim: int, index, src=None, *, value=None, bf16: bool = False) -> np.ndarray:
        """`out.scatter(dim, index, src)` (or `value=`) on the device, returned
        as a new array: each index writes `src` at the same coordinates, and a
        repeated index leaves the LAST value in index order, as torch's CPU
        kernel does. `index.shape[d] <= src.shape[d]` everywhere and
        `<= out.shape[d]` off `dim`."""
        np = _require_numpy("Device.scatter")
        out, index = np.asarray(out), np.asarray(index)
        dim = _index_dim(dim, out.ndim, "scatter")
        _index_shapes(index.shape, out.shape, dim, "scatter", "out")
        if (src is None) == (value is None):
            raise AckShapeError("scatter takes exactly one of src and value")
        src_st = [0] * out.ndim
        if src is not None:
            src = np.asarray(src)
            if src.dtype != out.dtype:
                raise AckError(f"scatter src dtype {src.dtype} is not out's {out.dtype}")
            if src.ndim != index.ndim or any(i > s for i, s in zip(index.shape, src.shape)):
                raise AckShapeError(f"scatter index {index.shape} exceeds src {src.shape}")
            src_st = _c_strides(src.shape)
        return self._index("scatter", 1, src, index, out.copy(), index.shape, dim, out.shape[dim],
                           _c_strides(index.shape), src_st, _c_strides(out.shape), bf16=bf16, value=value)

    def scatter_add(self, out, dim: int, index, src, *, bf16: bool = False) -> np.ndarray:
        """`out.scatter_add(dim, index, src)` on the device, returned as a new
        array. Each destination adds its sources in index order and rounds each
        step in its own dtype: numpy's order and rounding, and torch's CPU
        result for float32 and integers. For float16 (and bf16 bits) torch's
        CPU kernels instead accumulate in f32 and round once (measured), so a
        16-bit result can differ from torch's in the last place. float64 adds
        in binary64 as torch's CPU does, NaN rule included (index schema 2).
        Not for bool."""
        np = _require_numpy("Device.scatter_add")
        out, index, src = np.asarray(out), np.asarray(index), np.asarray(src)
        dim = _index_dim(dim, out.ndim, "scatter_add")
        _index_shapes(index.shape, out.shape, dim, "scatter_add", "out")
        if src.dtype != out.dtype:
            raise AckError(f"scatter_add src dtype {src.dtype} is not out's {out.dtype}")
        if src.ndim != index.ndim or any(i > s for i, s in zip(index.shape, src.shape)):
            raise AckShapeError(f"scatter_add index {index.shape} exceeds src {src.shape}")
        return self._index("scatter_add", 2, src, index, out.copy(), index.shape, dim, out.shape[dim],
                           _c_strides(index.shape), _c_strides(src.shape), _c_strides(out.shape), bf16=bf16)

    def index_add(self, out, dim: int, index, src, *, alpha: float = 1.0, bf16: bool = False) -> np.ndarray:
        """`out.index_add(dim, index, src, alpha=alpha)` on the device, returned
        as a new array: `src` is `out`'s shape with `len(index)` along `dim`.
        The same order and rounding as `scatter_add`; an alpha other than 1 is
        taken for float32 only (`src * alpha` in f32, then the add)."""
        return self._index_along("index_add", 2, out, dim, index, src, alpha=alpha, bf16=bf16)

    def index_copy(self, out, dim: int, index, src, *, bf16: bool = False) -> np.ndarray:
        """`out.index_copy(dim, index, src)` on the device, returned as a new
        array; a repeated index leaves the last row in index order."""
        return self._index_along("index_copy", 1, out, dim, index, src, alpha=1.0, bf16=bf16)

    def _index_along(self, name, op, out, dim, index, src, *, alpha, bf16) -> np.ndarray:
        np = _require_numpy(f"Device.{name}")
        out, index, src = np.asarray(out), np.asarray(index), np.asarray(src)
        dim = _index_dim(dim, out.ndim, name)
        if index.ndim != 1:
            raise AckShapeError(f"{name} takes a 1-D index")
        want = out.shape[:dim] + (index.shape[0],) + out.shape[dim + 1:]
        if src.shape != want:
            raise AckShapeError(f"{name} src {src.shape} is not {want}")
        if src.dtype != out.dtype:
            raise AckError(f"{name} src dtype {src.dtype} is not out's {out.dtype}")
        idx_st = [0] * out.ndim
        idx_st[dim] = 1
        return self._index(name, op, src, index, out.copy(), src.shape, dim, out.shape[dim],
                           idx_st, _c_strides(src.shape), _c_strides(out.shape), bf16=bf16, alpha=alpha)

    def _index(self, name, op, src, index, out, shape, dim, bound, idx_st, src_st, out_st,
               *, bf16=False, value=None, alpha=1.0) -> np.ndarray:
        """One `ack_index` call: upload, dispatch, check the status word,
        download `out`. `src` is None for a value scatter (its buffer is then
        `out`'s, which the kit allows for an unread source)."""
        np = _require_numpy(f"Device.{name}")
        code = _index_storage(out.dtype, bf16)
        if index.dtype not in (np.int64, np.int32):
            raise AckError(f"{name} index dtype must be int64 or int32, not {index.dtype}")
        if not 1 <= len(shape) <= 4:
            raise AckShapeError(f"{name} takes 1 to 4 dimensions, not {len(shape)}")
        if any(extent == 0 for extent in shape) or out.size == 0:
            return out
        if op == 2 and out.dtype == np.bool_:
            raise AckError(f"{name} does not add {out.dtype} (a bool add is a logical or)")
        flags = 0
        if index.dtype == np.int32:
            flags |= 1
        request = AckIndexRequest()
        request.op, request.dtype, request.ndim, request.dim = op, code, len(shape), dim
        if value is not None:
            flags |= 4
            bits = np.asarray(value, dtype=out.dtype).reshape(1).view(np.uint8)
            request.value = int.from_bytes(bits.tobytes(), "little")
        if alpha != 1.0:
            if op != 2 or out.dtype != np.float32:
                raise AckError(f"{name} takes an alpha other than 1 for float32 only")
            flags |= 8
            request.alpha_bits = int(np.asarray(alpha, np.float32).view(np.uint32))
        request.flags = flags
        for k in range(len(shape)):
            request.shape[k] = shape[k]
            request.index_strides[k] = idx_st[k]
            request.src_strides[k] = src_st[k]
            request.out_strides[k] = out_st[k]
        request.bound = bound
        with self._lock:
            self._require_open()
            self.last_ms = None
            with ExitStack() as cleanup:
                def upload(array):
                    raw = np.ascontiguousarray(array).reshape(-1).view(np.uint8)
                    # whole words: the kit addresses storage as 32-bit words
                    padded = np.zeros(max(4, -(-raw.nbytes // 4) * 4), np.uint8)
                    padded[:raw.nbytes] = raw
                    buffer = self.buffer(padded.nbytes)
                    cleanup.callback(buffer.free)
                    buffer.upload(padded)
                    return buffer
                b_index = upload(index)
                b_out = upload(out)
                b_src = upload(src) if src is not None else b_out
                b_status = upload(np.zeros(4, np.uint32))
                ms = ctypes.c_double(0.0)
                rc = self._lib.cdll.ack_index(self._handle, b_src._handle, b_index._handle, b_out._handle,
                                              b_status._handle, ctypes.byref(request),
                                              ctypes.sizeof(request), ctypes.byref(ms))
                if rc == ACK_ERR_SHAPE:
                    raise AckShapeError(self._lib.last_error())
                if rc != ACK_OK:
                    raise _kit_error(name, rc, self._lib.last_error())
                status = b_status.download(np.uint32)
                if status[request.status_slot] & 1:
                    raise IndexError(f"{name}: an index is out of range for dimension {dim} of extent {bound}")
                raw = b_out.download(np.uint8)[:out.nbytes]
                self.last_ms = ms.value
                return raw.view(out.dtype).reshape(out.shape).copy()

    # ---- convert/v1 (ABI 13) ------------------------------------------------------

    def convert(self, src, dtype, *, bf16: bool = False, out_bf16: bool = False) -> np.ndarray:
        """`src` converted to `dtype` on the device (convert/v1), rounded as c10
        rounds: float -> float through float32 to nearest even, int -> float to
        nearest even from the exact value, int -> int by two's complement
        truncation, anything -> bool as nonzero. A float -> integer conversion
        raises AckShapeError: c10's cast is undefined out of range, so the kit
        refuses it by name. `bf16=True` names a uint16 `src` as bf16 bits;
        `out_bf16=True` returns bf16 bits as uint16 (and ignores `dtype`)."""
        np = _require_numpy("Device.convert")
        src = np.asarray(src)
        out_dtype = np.dtype(np.uint16) if out_bf16 else np.dtype(dtype)
        return self.convert_view(src, out_dtype, src.shape, 0, _c_strides(src.shape), bf16=bf16, out_bf16=out_bf16)

    # ---- intwise/v1 (ABI 14) ------------------------------------------------------

    def intwise(self, op: str, x=None, y=None, *, scalar=0, alpha=1, compute=None,
                out_dtype=None, shape=None) -> np.ndarray:
        """torch's integer or bool pointwise operation `op` (`INTWISE_OPS`) on
        the device (intwise/v1), over dense arrays: x and y of one shape, or y
        None and `scalar` in its place. `compute` is C, the dtype torch computes
        in (default numpy's result_type of the operands, which agrees with
        torch's for these types); the result is `out_dtype` (default bool for a
        comparison, else C). fill and arange take no operands: `shape` and
        `out_dtype` name the output, `scalar` is the value or the start and
        `alpha` arange's step. The divisions (div_trunc, div_floor, remainder,
        fmod) take y None and `scalar` as the divisor, which must be nonzero in
        C and not -1 in an int32 or int64 C (the kit refuses those by name).
        Operands are wrapped to C and the result truncated to the output as c10
        casts; see kernels/intwise.comp.

        With `compute=np.float64` (schema 2) the binary64 module computes
        torch's float64 arithmetic bit for bit: add and sub (alpha 1 or -1),
        mul, div, reciprocal, neg, abs, fill and the comparisons, x and y of
        float64, float32, float16 or an integer or bool dtype, widened as torch
        casts them; `scalar` is a float. With a float operand name `compute`:
        numpy promotes float32 with int64 to float64 where torch keeps float32.

        cumsum (schema 3) takes x alone and sums along its LAST axis in an
        integer C, which is also the output's dtype by default, as torch's
        cumsum(x, -1, dtype=C) computes it (torch's own default C for an
        integer or bool x is int64: name it).

        tril and triu (schema 4) take x alone and keep it over its last two
        axes where column - row <= `scalar` (tril) or >= `scalar` (triu), the
        diagonal, else 0. They move bits and compute none, so a float array
        goes as the integer view of its width (x.view(np.int32) for float32).

        rank_asc and rank_desc (schema 5) take x alone, of any dtype (float64,
        float32, float16, an integer or bool), and give each element its
        position in the stable ascending or descending sort of its row (the
        LAST axis), as int64 by default: numpy's argsort(kind="stable") inverted.
        Floats compare as torch's sort compares them: -0.0 equals +0.0, and
        every NaN is equal to every other and larger than +inf."""
        np = _require_numpy("Device.intwise")
        if op not in INTWISE_OPS:
            raise AckError(f"intwise has no operation {op!r}")
        code = INTWISE_OPS[op]
        if op in ("fill", "arange"):
            if shape is None or out_dtype is None:
                raise AckError(f"intwise {op} takes shape= and out_dtype=")
            xs = None
            out_shape = tuple(int(e) for e in shape)
            compute = np.dtype(out_dtype) if compute is None else np.dtype(compute)
        else:
            xs = np.ascontiguousarray(x)
            out_shape = xs.shape
            if y is not None:
                y = np.ascontiguousarray(y)
                if y.shape != xs.shape:
                    raise AckShapeError(f"intwise takes x and y of one shape, not {xs.shape} and {y.shape}")
            if compute is None and op in ("rank_asc", "rank_desc"):
                compute = np.int64   # a rank is an integer whatever x is
            if compute is None:
                operands = (xs.dtype,) if y is None else (xs.dtype, y.dtype)
                if any(d.kind == "f" for d in operands) and np.dtype(np.float64) not in operands:
                    raise AckError("intwise with a float operand takes compute= (numpy's float promotion is "
                                   "not torch's: float32 with int64 is float64 in numpy, float32 in torch)")
                compute = np.result_type(*operands)
            compute = np.dtype(compute)
        binary64 = compute == np.dtype(np.float64)
        compare = op in ("eq", "ne", "lt", "le", "gt", "ge")
        out_dtype = np.dtype(out_dtype) if out_dtype is not None else np.dtype(bool if compare else compute)
        if len(out_shape) == 0:
            flat = self.intwise(op, None if xs is None else xs.reshape(1), None if y is None else y.reshape(1),
                                scalar=scalar, alpha=alpha, compute=compute, out_dtype=out_dtype,
                                shape=None if shape is None else (1,))
            return flat.reshape(())
        if not 1 <= len(out_shape) <= 4:
            raise AckShapeError(f"intwise takes 1 to 4 dimensions, not {out_shape}")
        out = np.empty(out_shape, out_dtype)
        if out.size == 0:
            return out
        request = AckIntwiseRequest()
        request.op = code
        request.compute_code = _intwise_code(compute, binary64)
        request.out_code = _intwise_code(out_dtype, binary64)
        # a rank's x may be any dtype, whatever C is
        x_any = binary64 or op in ("rank_asc", "rank_desc")
        request.x_code = _intwise_code(xs.dtype, x_any) if xs is not None else request.compute_code
        request.y_code = _intwise_code(y.dtype, binary64) if y is not None else request.compute_code
        request.ndim = len(out_shape)
        request.flags = 1 if (xs is not None and y is None
                              and op not in ("neg", "abs", "reciprocal", "cumsum", "tril", "triu",
                                             "rank_asc", "rank_desc")) else 0
        strides = _c_strides(out_shape)
        for k in range(len(out_shape)):
            request.shape[k] = out_shape[k]
            request.x_strides[k] = request.y_strides[k] = request.out_strides[k] = strides[k]
        if binary64:
            request.scalar, request.alpha = _f64_bits(scalar), _f64_bits(alpha)
        else:
            request.scalar, request.alpha = int(scalar), int(alpha)
        with self._lock:
            self._require_open()
            self.last_ms = None
            with ExitStack() as cleanup:
                def buffer_with(array):
                    # whole words: the kit addresses storage as 32-bit words
                    raw = np.zeros(max(4, -(-array.nbytes // 4) * 4), np.uint8)
                    raw[:array.nbytes] = array.reshape(-1).view(np.uint8)
                    buffer = self.buffer(raw.nbytes)
                    cleanup.callback(buffer.free)
                    buffer.upload(raw)
                    return buffer
                b_out = buffer_with(np.zeros(out.nbytes, np.uint8))
                b_x = buffer_with(xs) if xs is not None else b_out
                b_y = buffer_with(y) if y is not None else b_out
                ms = ctypes.c_double(0.0)
                rc = self._lib.cdll.ack_intwise(self._handle, b_x._handle, b_y._handle, b_out._handle,
                                                ctypes.byref(request), ctypes.sizeof(request), ctypes.byref(ms))
                if rc == ACK_ERR_SHAPE:
                    raise AckShapeError(self._lib.last_error())
                if rc != ACK_OK:
                    raise _kit_error("intwise", rc, self._lib.last_error())
                got = b_out.download(np.uint8)[:out.nbytes]
                self.last_ms = ms.value
                return got.view(out_dtype).reshape(out_shape).copy()

    def convert_view(self, base, dtype, shape, offset: int, strides, *, bf16: bool = False,
                     out_bf16: bool = False) -> np.ndarray:
        """`convert` over a strided view of `base` (uploaded whole, in C order):
        `shape` of rank 1-4, an element `offset` and element `strides` into
        `base`'s flat storage. The result is a new dense array of `dtype`."""
        np = _require_numpy("Device.convert_view")
        base = np.ascontiguousarray(base)
        out_dtype = np.dtype(np.uint16) if out_bf16 else np.dtype(dtype)
        src_code = _convert_code(base.dtype, bf16)
        out_code = _convert_code(out_dtype, out_bf16)
        shape = tuple(int(extent) for extent in shape)
        if len(shape) == 0:
            return self.convert_view(base, dtype, (1,), offset, (0,), bf16=bf16, out_bf16=out_bf16).reshape(())
        if not 1 <= len(shape) <= 4 or len(strides) != len(shape):
            raise AckShapeError(f"convert takes 1 to 4 dimensions with one stride each, not {shape} / {tuple(strides)}")
        out = np.empty(shape, out_dtype)
        if out.size == 0:
            return out
        request = AckConvertRequest()
        request.src_code, request.out_code, request.ndim, request.flags = src_code, out_code, len(shape), 0
        out_strides = _c_strides(shape)
        for k in range(len(shape)):
            request.shape[k] = shape[k]
            request.in_strides[k] = int(strides[k])
            request.out_strides[k] = out_strides[k]
        request.in_offset, request.out_offset = int(offset), 0
        with self._lock:
            self._require_open()
            self.last_ms = None
            with ExitStack() as cleanup:
                def buffer_of(nbytes):
                    # whole words: the kit addresses storage as 32-bit words
                    buffer = self.buffer(max(4, -(-nbytes // 4) * 4))
                    cleanup.callback(buffer.free)
                    return buffer
                raw = base.reshape(-1).view(np.uint8)
                b_src = buffer_of(raw.nbytes)
                padded = np.zeros(b_src.nbytes, np.uint8)
                padded[:raw.nbytes] = raw
                b_src.upload(padded)
                b_out = buffer_of(out.nbytes)
                ms = ctypes.c_double(0.0)
                rc = self._lib.cdll.ack_convert(self._handle, b_src._handle, b_out._handle, ctypes.byref(request),
                                                ctypes.sizeof(request), ctypes.byref(ms))
                if rc == ACK_ERR_SHAPE:
                    raise AckShapeError(self._lib.last_error())
                if rc != ACK_OK:
                    raise _kit_error("convert", rc, self._lib.last_error())
                got = b_out.download(np.uint8)[:out.nbytes]
                self.last_ms = ms.value
                return got.view(out_dtype).reshape(shape).copy()

    def _coopmat_product(self, entry: str, name: str, a16, b16, m: int, n: int, k: int, a_t, b_t) -> np.ndarray:
        """One product through a 16-bit cooperative-matrix entry, from operands
        already in their 16-bit storage. The caller holds the lock."""
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
            rc = getattr(self._lib.cdll, entry)(
                self._handle, ba._handle, bb._handle, bc._handle, m, n, k, int(a_t), int(b_t), ctypes.byref(ms)
            )
            if rc == ACK_ERR_SHAPE:
                raise AckShapeError(self._lib.last_error())
            if rc != ACK_OK:
                raise _kit_error(name, rc, self._lib.last_error())
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
