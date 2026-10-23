//! Pure geometry, capacity, aliasing and dispatch planning for `index/v1`
//! (operation schema 2): gather, scatter and scatter-add over strided views of
//! at most four dimensions, for elements of 1, 2, 4 or 8 bytes and int32 or
//! int64 indices. `kernels/index.comp` is the shader; its header states the
//! arithmetic, the order of accumulation and what an index out of range does.
//!
//! One ITERATION SPACE (`shape`) with an indexed dimension `dim`. At each
//! coordinate the kernel reads the index view there and checks it against
//! `bound`, the extent of the indexed tensor's dimension `dim`:
//! - [`IndexOp::Gather`]: `out[c] = src[c with c[dim] := t]` (torch's gather,
//!   index_select, and index.Tensor with one index tensor);
//! - [`IndexOp::Scatter`]: `out[c with c[dim] := t] = src[c]`, or the scalar
//!   value (scatter, index_copy_);
//! - [`IndexOp::ScatterAdd`]: `out[c with c[dim] := t] += src[c]`, times an f32
//!   alpha for f32 only (scatter_add, index_add).
//!
//! A 1-D index along `dim` reaches every other dimension through stride 0.
//!
//! SCHEMA 2 (2026-09-28): a scatter's 1- or 2-byte output whose fibers would
//! share 32-bit words, which schema 1 refused as
//! [`IndexPlanError::ScatterOutputMisaligned`], is planned one invocation a
//! fiber with [`FLAG_ATOMIC_OUT`]: the shader writes each element with an
//! integer atomicAnd and atomicOr (`kernels/index.comp`, SUB-WORD ELEMENTS).
//! MEASURED 2026-09-28: the toy stage-E step's last 8 `scatter.value` host
//! fallbacks were bool masks scattered along rows whose width is not a
//! multiple of 4. The name is kept for decoding and is no longer produced.
//! In the same schema a scatter-add over [`Dtype::F64`] is planned too, which
//! schema 1 refused as [`IndexPlanError::AddUnsupportedDtype`]: the shader adds
//! in IEEE binary64 with integer instructions only (`kernels/index.comp`,
//! FLOAT64 SCATTER-ADD), torch's CPU NaN rule included. MEASURED 2026-09-28:
//! the toy stage-E step's 16 `scatter_add_` host fallbacks (`index-add-dtype`).
//!
//! SCHEMA 3 (2026-09-29): LONG FIBERS IN PARALLEL. A scatter walks each fiber
//! in one invocation, so a 1-D scatter is one invocation: the sub-bit
//! student's MoE routing scatters, counts and inverts its 65,536 (token, slot)
//! pairs that way, 23-37 ms a call on an L4 (MEASURED 2026-09-29, b162b930's
//! stage-E step anatomy: 216 such calls, 6.2 s of the step's 24.1 device s).
//! Where a fiber has at least [`LONG_FIBER`] elements the plan takes an
//! element-parallel [`IndexStrategy`] instead, bit for bit the CPU's result:
//! - [`IndexStrategy::AtomicAdd`]: an int32 or int64 scatter-add, one
//!   invocation an element, added with integer atomics (exact in any order);
//! - [`IndexStrategy::Elected`]: a scatter in three passes over a scratch of
//!   one word a destination that the native entry allocates: clear, elect
//!   each destination's last writer along its fiber (the CPU's survivor), and
//!   let only the winners write. A scratch past [`ELECT_SCRATCH_LIMIT`] words
//!   keeps the fiber walk.
//! - [`IndexStrategy::Grouped`]: any other scatter-add (a float's, whose order
//!   is part of its sum, or a narrow integer's), whose sources are grouped by
//!   destination with the radix module's stable sort (`radix_ops`) and added
//!   one group an invocation in the fiber's order, which is the CPU's walk for
//!   that destination: a KEYS pass, the sort's digit passes, a RUNS pass.
//!   MEASURED 2026-09-29: the load-balancing loss's float64 expert counts, 24
//!   calls and 0.9 s of the L4 step.
//!
//! The request, the refusals and the result are unchanged; only the passes
//! differ ([`IndexPlan::passes`], [`IndexPlan::radix`]).
//!
//! This module has no native `Context` or `Buffer` dependency: storage
//! identities and capacities are the caller's declaration, and the native
//! entry takes them from the real buffers immediately before binding.
//!
//! Refusals are by name, in this order: operation, dtype and flags (at
//! decode), number of dimensions, the indexed dimension, zero elements, the
//! bound, the value and alpha rules, the status slot, then address arithmetic
//! and capacity for the index, the source and the output in turn, the gather
//! output's contiguity, the scatter output's distinct addresses, and last the
//! aliasing of the output and the status word.

use std::fmt;

use crate::convert_ops::Code;
use crate::radix_ops::{self, RadixPass, RadixPlan};

pub const ABI: &str = "index/v1";
/// The operation schema the C ABI reports for the family
/// (`ffi::ACK_OP_SCHEMA_INDEX`): 2 since a sub-word scatter output of any
/// layout is written with atomics rather than refused, and a float64
/// scatter-add is computed rather than refused (2026-09-28); 3 since a long
/// fiber's scatter or integer scatter-add runs an element an invocation
/// (2026-09-29).
pub const SCHEMA: u32 = 3;
pub const MAX_NDIM: usize = 4;
/// `src`, `index`, `out`, `status`, in binding order: the Vulkan baseline
/// guarantees four storage buffers per stage, and this family needs no more.
pub const BINDINGS: u32 = 4;
pub const WORKGROUP_INVOCATIONS: u32 = 64;
/// Every device offers at least this many workgroups on each dispatch axis.
pub const MAX_GROUPS_X: u32 = 65_535;
/// Every element and word index the shader forms is below this, so its u32
/// arithmetic cannot wrap and an index fits a signed 32-bit int.
pub const WORD_LIMIT: u64 = 1 << 31;
pub const PUSH_WORDS: usize = 28;
pub const PUSH_BYTES: usize = PUSH_WORDS * 4;

/// Flags word bit 0: the index is int32 (else int64 as little-endian word pairs).
pub const FLAG_I32_INDEX: u32 = 1;
/// Bit 1: a negative index in `[-bound, 0)` counts from the end (index.Tensor's rule).
pub const FLAG_WRAP: u32 = 2;
/// Bit 2: a scatter writes the scalar in words 24-25 and does not read `src`.
pub const FLAG_VALUE: u32 = 4;
/// Bit 3: an f32 scatter-add scales each source value by the f32 in word 24 first.
pub const FLAG_ALPHA: u32 = 8;
const KNOWN_FLAGS: u32 = FLAG_I32_INDEX | FLAG_WRAP | FLAG_VALUE | FLAG_ALPHA;
/// Push word 3, bit 4 (schema 2): a 1- or 2-byte scatter output whose fibers
/// share words, written one invocation a fiber with integer atomics. Set by
/// the planner, never by a request (it is not in the request's known flags).
pub const FLAG_ATOMIC_OUT: u32 = 16;
/// Push word 3, bit 5 (schema 3): an int32 or int64 scatter-add planned one
/// invocation an element, added with integer atomics
/// ([`IndexStrategy::AtomicAdd`]). The planner's, never a request's.
pub const FLAG_ATOMIC_ADD: u32 = 32;

/// Push word 0's operations of the elected scatter's passes (schema 3): the
/// planner's, never a request's ([`IndexOp`] decodes 0-2 only).
pub const OP_CLEAR: u32 = 3;
pub const OP_ELECT: u32 = 4;
pub const OP_WRITE: u32 = 5;
/// The grouped scatter-add's first and last passes (schema 3): the planner's.
pub const OP_KEYS: u32 = 6;
pub const OP_RUNS: u32 = 7;

/// The grouped scatter-add's key storage for `slots` destinations: the
/// narrowest of 1, 2 or 4 bytes whose largest value (the SENTINEL an index out
/// of range takes, which sorts after every slot) is above every slot; as
/// `index.comp`'s `group_sentinel`.
pub fn group_key_code(slots: u64) -> Code {
    if slots < 256 {
        Code::U8
    } else if slots < 65_536 {
        Code::I16
    } else {
        Code::I32
    }
}

/// Schema 3: whether a request's words alone allow a grouped scatter-add (an
/// operation 2 other than int32 or int64 whose indexed extent is long), so
/// the native entry can build the radix module before it borrows any buffer.
/// The planner decides; this never says no where the planner could say yes.
pub fn may_plan_grouped(op: u32, dtype: u32, ndim: u32, dim: u32, shape: [u64; MAX_NDIM]) -> bool {
    let fiber = match (ndim, dim) {
        (1..=4, d) if d < ndim => shape[d as usize],
        _ => 0,
    };
    op == IndexOp::ScatterAdd as u32
        && dtype != Dtype::I32 as u32
        && dtype != Dtype::I64 as u32
        && fiber >= LONG_FIBER
}

/// The fiber length from which a scatter or an integer scatter-add runs an
/// element an invocation (schema 3). Below it, one invocation walking a fiber
/// costs a few microseconds at most, and the walk is the proven path.
pub const LONG_FIBER: u64 = 256;
/// The most scratch words an elected scatter takes (64 MiB): one word a
/// destination, fibers x bound. Above it the scatter keeps the fiber walk.
pub const ELECT_SCRATCH_LIMIT: u64 = 1 << 24;

/// A request's flags decoded by name, before any buffer is looked at:
/// [`IndexPlanError::FlagsOutOfRange`] for a bit outside the four a request
/// may carry, the planner's own [`FLAG_ATOMIC_OUT`] included.
pub fn request_flags(flags: u32) -> Result<u32, IndexPlanError> {
    if flags & !KNOWN_FLAGS != 0 {
        return Err(IndexPlanError::FlagsOutOfRange);
    }
    Ok(flags)
}

/// Status word bit 0: an index was out of range (the kernel never addressed with it).
pub const STATUS_INDEX: u32 = 1;
/// Status word bit 1: the block failed the shader's own guards (raw ABI misuse).
pub const STATUS_PLAN: u32 = 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IndexOp {
    Gather = 0,
    Scatter = 1,
    ScatterAdd = 2,
}

impl TryFrom<u32> for IndexOp {
    type Error = IndexPlanError;

    fn try_from(word: u32) -> Result<Self, Self::Error> {
        match word {
            0 => Ok(Self::Gather),
            1 => Ok(Self::Scatter),
            2 => Ok(Self::ScatterAdd),
            _ => Err(IndexPlanError::OperationOutOfRange),
        }
    }
}

/// Element storage. Gather and scatter copy bits, so every storage takes them;
/// a scatter-add computes in the storage's own type and rounds each step as
/// one element's `a += b` does in c10: `F64` too since schema 2, in binary64
/// words with integer instructions (no shaderFloat64, which is optional in
/// Vulkan). For f32, f64 and integers that is torch's CPU result; for bf16 and
/// f16 it is not (torch's
/// CPU kernels accumulate those in f32 and round once), so the torch adapter
/// refuses 16-bit float adds. torch's bool is `U8` storage here; its adapter
/// refuses a bool scatter-add, whose meaning is a logical or, before it
/// reaches this family.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Dtype {
    F32 = 0,
    Bf16 = 1,
    F16 = 2,
    I64 = 3,
    I32 = 4,
    I16 = 5,
    I8 = 6,
    U8 = 7,
    F64 = 8,
}

impl Dtype {
    pub fn bytes(self) -> u64 {
        match self {
            Self::I64 | Self::F64 => 8,
            Self::F32 | Self::I32 => 4,
            Self::Bf16 | Self::F16 | Self::I16 => 2,
            Self::I8 | Self::U8 => 1,
        }
    }
}

impl TryFrom<u32> for Dtype {
    type Error = IndexPlanError;

    fn try_from(word: u32) -> Result<Self, Self::Error> {
        Ok(match word {
            0 => Self::F32,
            1 => Self::Bf16,
            2 => Self::F16,
            3 => Self::I64,
            4 => Self::I32,
            5 => Self::I16,
            6 => Self::I8,
            7 => Self::U8,
            8 => Self::F64,
            _ => return Err(IndexPlanError::DtypeOutOfRange),
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IndexPlanError {
    OperationOutOfRange,
    DtypeOutOfRange,
    FlagsOutOfRange,
    NdimOutOfRange,
    DimOutOfRange,
    ZeroElements,
    BoundOutOfRange,
    /// `FLAG_VALUE` on an operation other than a scatter.
    ValueNeedsScatter,
    /// `FLAG_ALPHA` other than on an f32 scatter-add.
    AlphaNeedsF32ScatterAdd,
    /// Schema 1's refusal of a scatter-add over `F64` storage. Schema 2 adds
    /// in binary64, so this is no longer produced; it stays for decoding an
    /// older library's codes.
    AddUnsupportedDtype,
    StatusSlotOutOfRange,
    AddressOverflow,
    /// A gather's output is not row-major contiguous over the shape, or not
    /// word-aligned for elements under four bytes.
    GatherOutputNotContiguous,
    /// Schema 1's refusal of a scatter of 1- or 2-byte elements whose output
    /// words an invocation would share with another. Schema 2 plans such a
    /// scatter with [`FLAG_ATOMIC_OUT`] instead, so this is no longer
    /// produced; it stays for decoding an older library's codes.
    ScatterOutputMisaligned,
    OutputSelfOverlap,
    OutputOverlapsInput,
    StatusOverlapsOperand,
}

impl fmt::Display for IndexPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::OperationOutOfRange => "index-operation-out-of-range",
            Self::DtypeOutOfRange => "index-dtype-out-of-range",
            Self::FlagsOutOfRange => "index-flags-out-of-range",
            Self::NdimOutOfRange => "index-ndim-out-of-range",
            Self::DimOutOfRange => "index-dim-out-of-range",
            Self::ZeroElements => "index-zero-elements",
            Self::BoundOutOfRange => "index-bound-out-of-range",
            Self::ValueNeedsScatter => "index-value-needs-scatter",
            Self::AlphaNeedsF32ScatterAdd => "index-alpha-needs-f32-scatter-add",
            Self::AddUnsupportedDtype => "index-add-unsupported-dtype",
            Self::StatusSlotOutOfRange => "index-status-slot-out-of-range",
            Self::AddressOverflow => "index-address-overflow",
            Self::GatherOutputNotContiguous => "index-gather-output-not-contiguous",
            Self::ScatterOutputMisaligned => "index-scatter-output-misaligned",
            Self::OutputSelfOverlap => "index-output-self-overlap",
            Self::OutputOverlapsInput => "index-output-overlaps-input",
            Self::StatusOverlapsOperand => "index-status-overlaps-operand",
        })
    }
}

impl std::error::Error for IndexPlanError {}

/// One operand's element offset and per-dimension element strides, in its own
/// element size, indexed like the plan's shape.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct View {
    pub offset: u64,
    pub strides: [u64; MAX_NDIM],
}

impl View {
    pub const fn new(offset: u64, strides: [u64; MAX_NDIM]) -> Self {
        Self { offset, strides }
    }

    /// Row-major strides for `shape` at `offset`.
    pub fn contiguous(offset: u64, shape: &[u64]) -> Self {
        let mut strides = [0u64; MAX_NDIM];
        let mut stride = 1u64;
        for (dimension, &extent) in shape.iter().enumerate().rev() {
            strides[dimension] = stride;
            stride = stride.saturating_mul(extent);
        }
        Self { offset, strides }
    }
}

/// The buffer an operand is bound to, as the caller declares it: an identity
/// equal exactly when two operands share one allocation, and its size in bytes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Storage {
    pub identity: u64,
    pub capacity_bytes: u64,
}

impl Storage {
    pub const fn new(identity: u64, capacity_bytes: u64) -> Self {
        Self {
            identity,
            capacity_bytes,
        }
    }
}

/// What a caller asks for. `index` and `src` are addressed over `shape`; `out`
/// over `shape` for a gather and over `shape` with extent `bound` along `dim`
/// for a scatter; `src` of a gather over `shape` with extent `bound` along
/// `dim`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IndexRequest {
    pub op: IndexOp,
    pub dtype: Dtype,
    pub ndim: usize,
    pub shape: [u64; MAX_NDIM],
    pub dim: usize,
    pub index: View,
    pub index_i32: bool,
    pub src: View,
    pub out: View,
    pub bound: u64,
    pub wrap: bool,
    /// A scatter's scalar source, in the storage's own bits (low bytes).
    pub value: Option<u64>,
    /// An f32 scatter-add's alpha, as its bits.
    pub alpha: Option<u32>,
    pub status_slot: u32,
}

/// Storage in binding order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Bindings {
    pub src: Storage,
    pub index: Storage,
    pub out: Storage,
    pub status: Storage,
}

/// How a plan runs (schema 3; module docs, LONG FIBERS IN PARALLEL).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IndexStrategy {
    /// One pass: a gather's output units, or a scatter's fiber groups, each
    /// walking its fibers in index order (schemas 1 and 2).
    Fibers,
    /// One pass, one invocation an element: an int32 or int64 scatter-add of
    /// long fibers, added with integer atomics.
    AtomicAdd,
    /// Three passes (clear, elect, write) over the native entry's scratch of
    /// [`IndexPlan::scratch_words`] words: a scatter of long fibers.
    Elected,
    /// A KEYS pass, the radix module's digit passes ([`IndexPlan::radix`]) and
    /// a RUNS pass over the native entry's scratch of
    /// [`IndexPlan::scratch_words`] words: any other scatter-add of long fibers.
    Grouped,
}

/// One dispatch of a plan: its push block and its workgroups.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IndexPass {
    words: [u32; PUSH_WORDS],
    groups: [u32; 3],
}

impl IndexPass {
    fn new(words: [u32; PUSH_WORDS]) -> Self {
        let units = u64::from(words[27]);
        let total_groups = units.div_ceil(u64::from(WORKGROUP_INVOCATIONS));
        let x = total_groups.min(u64::from(MAX_GROUPS_X)).max(1);
        let y = total_groups.div_ceil(x).max(1);
        Self {
            words,
            groups: [x as u32, y as u32, 1],
        }
    }

    /// Push word 0: 0-2 the request's operation, or the elected scatter's
    /// [`OP_CLEAR`], [`OP_ELECT`] and [`OP_WRITE`].
    pub fn op(&self) -> u32 {
        self.words[0]
    }

    /// Invocations with work.
    pub fn units(&self) -> u32 {
        self.words[27]
    }

    pub fn words(&self) -> [u32; PUSH_WORDS] {
        self.words
    }

    pub fn push_constants(&self) -> [u8; PUSH_BYTES] {
        let mut bytes = [0u8; PUSH_BYTES];
        for (chunk, word) in bytes.chunks_exact_mut(4).zip(self.words) {
            chunk.copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }

    pub fn dispatch_groups(&self) -> [u32; 3] {
        self.groups
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IndexPlan {
    request: IndexRequest,
    strategy: IndexStrategy,
    passes: [IndexPass; 3],
    pass_count: usize,
    scratch_words: u64,
    /// The grouped scatter-add's sort: its digit passes run between KEYS and RUNS.
    radix: Option<RadixPlan>,
}

/// The highest element `view` addresses over `extents`, checked against the
/// limit; `None` when an address would overflow.
fn max_element(view: &View, extents: &[u64; MAX_NDIM]) -> Option<u64> {
    let mut max_index = view.offset;
    for (k, &extent) in extents.iter().enumerate() {
        if extent > 1 {
            let span = (extent - 1).checked_mul(view.strides[k])?;
            max_index = max_index.checked_add(span)?;
        }
    }
    Some(max_index)
}

/// Every coordinate reaches a distinct address: ordered by stride, each stride
/// is at least the span of the dimensions inside it (the pointwise family's rule).
fn addresses_are_distinct(view: &View, extents: &[u64; MAX_NDIM]) -> bool {
    let mut dimensions: Vec<(u64, u64)> = extents
        .iter()
        .zip(view.strides)
        .filter(|(&extent, _)| extent > 1)
        .map(|(&extent, stride)| (stride, extent))
        .collect();
    dimensions.sort_unstable();
    let mut span = 1u64;
    for (stride, extent) in dimensions {
        if stride < span {
            return false;
        }
        span = span.saturating_add(stride.saturating_mul(extent - 1));
    }
    true
}

/// Every fiber's address interval `[base, base + (bound - 1) * stride_d]` is
/// disjoint from every other's: the distinctness rule with the fiber's own span
/// as the innermost block. With word-aligned bases this means no two fibers
/// write into one word.
fn fibers_are_disjoint(
    out_st: &[u64; MAX_NDIM],
    extents: &[u64; MAX_NDIM],
    d: usize,
    bound: u64,
) -> bool {
    let mut dimensions: Vec<(u64, u64)> = (0..MAX_NDIM)
        .filter(|&k| k != d && extents[k] > 1)
        .map(|k| (out_st[k], extents[k]))
        .collect();
    dimensions.sort_unstable();
    let mut span = (bound - 1).saturating_mul(out_st[d]).saturating_add(1);
    for (stride, extent) in dimensions {
        if stride < span {
            return false;
        }
        span = span.saturating_add(stride.saturating_mul(extent - 1));
    }
    true
}

/// Checks `view` over `extents` against the word limit (in the element's own
/// word arithmetic) and `storage`'s LOGICAL size: every element the view
/// reaches must end inside it. The shader moves whole words, and the word that
/// holds a final sub-word element may run past the logical size; the kit
/// creates every VkBuffer a whole word long for exactly that
/// (`Context::buffer`), and those padding bytes are kept, never used.
fn check_capacity(
    view: &View,
    extents: &[u64; MAX_NDIM],
    element_bytes: u64,
    storage: &Storage,
) -> Result<(), IndexPlanError> {
    let max = max_element(view, extents).ok_or(IndexPlanError::AddressOverflow)?;
    let end_bytes = max
        .checked_add(1)
        .and_then(|elements| elements.checked_mul(element_bytes))
        .ok_or(IndexPlanError::AddressOverflow)?;
    // the element index and the last word it reaches (an 8-byte element's
    // second word) are formed in u32 by the shader
    let last_word = (end_bytes - 1) / 4;
    if max >= WORD_LIMIT || last_word >= WORD_LIMIT {
        return Err(IndexPlanError::AddressOverflow);
    }
    if end_bytes > storage.capacity_bytes {
        return Err(IndexPlanError::AddressOverflow);
    }
    Ok(())
}

impl IndexPlan {
    pub fn new(request: IndexRequest, bindings: Bindings) -> Result<Self, IndexPlanError> {
        let IndexRequest {
            op,
            dtype,
            ndim,
            shape,
            dim,
            index,
            index_i32,
            src,
            out,
            bound,
            wrap,
            value,
            alpha,
            status_slot,
        } = request;
        if ndim == 0 || ndim > MAX_NDIM {
            return Err(IndexPlanError::NdimOutOfRange);
        }
        if dim >= ndim {
            return Err(IndexPlanError::DimOutOfRange);
        }
        // pad at the front, so the row-major order of the coordinates is unchanged
        let pad = MAX_NDIM - ndim;
        let mut extents = [1u64; MAX_NDIM];
        let mut idx_st = [0u64; MAX_NDIM];
        let mut src_st = [0u64; MAX_NDIM];
        let mut out_st = [0u64; MAX_NDIM];
        extents[pad..].copy_from_slice(&shape[..ndim]);
        idx_st[pad..].copy_from_slice(&index.strides[..ndim]);
        src_st[pad..].copy_from_slice(&src.strides[..ndim]);
        out_st[pad..].copy_from_slice(&out.strides[..ndim]);
        let d = pad + dim;
        if extents.contains(&0) {
            return Err(IndexPlanError::ZeroElements);
        }
        let n = extents
            .iter()
            .try_fold(1u64, |acc, &extent| acc.checked_mul(extent))
            .filter(|&n| n < WORD_LIMIT)
            .ok_or(IndexPlanError::AddressOverflow)?;
        if bound == 0 || bound >= WORD_LIMIT {
            return Err(IndexPlanError::BoundOutOfRange);
        }
        if value.is_some() && op != IndexOp::Scatter {
            return Err(IndexPlanError::ValueNeedsScatter);
        }
        if alpha.is_some() && !(op == IndexOp::ScatterAdd && dtype == Dtype::F32) {
            return Err(IndexPlanError::AlphaNeedsF32ScatterAdd);
        }
        if u64::from(status_slot) >= bindings.status.capacity_bytes / 4 {
            return Err(IndexPlanError::StatusSlotOutOfRange);
        }
        let es = dtype.bytes();
        // the index view, over the iteration space
        let index_view = View::new(index.offset, idx_st);
        check_capacity(
            &index_view,
            &extents,
            if index_i32 { 4 } else { 8 },
            &bindings.index,
        )?;
        // the indexed tensor spans `bound` along `dim`
        let mut indexed = extents;
        indexed[d] = bound;
        let src_view = View::new(src.offset, src_st);
        let out_view = View::new(out.offset, out_st);
        if value.is_none() {
            let src_extents = if op == IndexOp::Gather {
                &indexed
            } else {
                &extents
            };
            check_capacity(&src_view, src_extents, es, &bindings.src)?;
        }
        let out_extents = if op == IndexOp::Gather {
            &extents
        } else {
            &indexed
        };
        check_capacity(&out_view, out_extents, es, &bindings.out)?;
        let mut atomic_out = false;
        let (inner, log2g, units) = if op == IndexOp::Gather {
            // row-major contiguous over the shape, word-aligned under four bytes
            let contiguous = View::contiguous(0, &extents);
            let strided_ok =
                (0..MAX_NDIM).all(|k| extents[k] == 1 || out_st[k] == contiguous.strides[k]);
            if !strided_ok || (es < 4 && out.offset % (4 / es) != 0) {
                return Err(IndexPlanError::GatherOutputNotContiguous);
            }
            let units = if es == 8 { n } else { (n * es).div_ceil(4) };
            (0usize, 0u32, units)
        } else {
            if !addresses_are_distinct(&out_view, &indexed) {
                return Err(IndexPlanError::OutputSelfOverlap);
            }
            let fibers = n / extents[d];
            let inner = (0..MAX_NDIM).rev().find(|&k| k != d && extents[k] > 1);
            match (es < 4, inner) {
                (true, Some(inner)) => {
                    let g = 4 / es;
                    // (a) g adjacent fibers along an inner dimension of output
                    // stride 1 share each word, so one invocation takes all g
                    let grouped = out_st[inner] == 1
                        && extents[inner] % g == 0
                        && out.offset % g == 0
                        && (0..MAX_NDIM)
                            .all(|k| k == inner || indexed[k] == 1 || out_st[k] % g == 0);
                    // (b) each fiber alone owns whole words: its base is
                    // word-aligned and no other fiber's addresses fall inside
                    // its span (a bool mask scattered along its last dimension)
                    let alone = out.offset % g == 0
                        && (0..MAX_NDIM).all(|k| k == d || extents[k] == 1 || out_st[k] % g == 0)
                        && fibers_are_disjoint(&out_st, &extents, d, bound);
                    if grouped {
                        (inner, g.trailing_zeros(), fibers / g)
                    } else {
                        // (c) otherwise fibers share words (schema 2): one fiber
                        // an invocation still, each element written with atomics
                        atomic_out = !alone;
                        (inner, 0u32, fibers)
                    }
                }
                // one fiber (every other extent 1), or whole-word elements
                _ => (inner.unwrap_or(0), 0u32, fibers),
            }
        };
        // the output is written while the other bindings are read
        let src_read = value.is_none();
        if bindings.out.identity == bindings.index.identity
            || (src_read && bindings.out.identity == bindings.src.identity)
        {
            return Err(IndexPlanError::OutputOverlapsInput);
        }
        let status_id = bindings.status.identity;
        if status_id == bindings.out.identity
            || status_id == bindings.index.identity
            || (src_read && status_id == bindings.src.identity)
        {
            return Err(IndexPlanError::StatusOverlapsOperand);
        }
        let mut flags = 0u32;
        if index_i32 {
            flags |= FLAG_I32_INDEX;
        }
        if wrap {
            flags |= FLAG_WRAP;
        }
        if value.is_some() {
            flags |= FLAG_VALUE;
        }
        if alpha.is_some() {
            flags |= FLAG_ALPHA;
        }
        if atomic_out {
            flags |= FLAG_ATOMIC_OUT;
        }
        let (value_lo, value_hi) = match (value, alpha) {
            (Some(bits), _) => (bits as u32, (bits >> 32) as u32),
            (None, Some(bits)) => (bits, 0),
            (None, None) => (0, 0),
        };
        // a stride on an extent-1 dimension never contributes; zeroing it
        // makes a decoded block re-encode to the same words
        let canon = |strides: [u64; MAX_NDIM], over: &[u64; MAX_NDIM]| {
            let mut words = [0u32; MAX_NDIM];
            for k in 0..MAX_NDIM {
                if over[k] > 1 {
                    words[k] = strides[k] as u32;
                }
            }
            words
        };
        let idx_words = canon(idx_st, &extents);
        let src_words = canon(
            src_st,
            if op == IndexOp::Gather {
                &indexed
            } else {
                &extents
            },
        );
        let out_words = canon(out_st, out_extents);
        let mut words = [0u32; PUSH_WORDS];
        words[0] = op as u32;
        words[1] = dtype as u32;
        words[2] = d as u32 | (inner as u32) << 4 | log2g << 8;
        words[3] = flags;
        for k in 0..MAX_NDIM {
            words[4 + k] = extents[k] as u32;
            words[9 + k] = idx_words[k];
            words[14 + k] = if value.is_some() { 0 } else { src_words[k] };
            words[19 + k] = out_words[k];
        }
        words[8] = index.offset as u32;
        words[13] = if value.is_some() {
            0
        } else {
            src.offset as u32
        };
        words[18] = out.offset as u32;
        words[23] = bound as u32;
        words[24] = value_lo;
        words[25] = value_hi;
        words[26] = status_slot;
        words[27] = units as u32;
        // Schema 3: a long fiber runs an element an invocation. Both paths
        // address exactly what the walk addresses, so every check above holds.
        let fiber_len = extents[d];
        let fibers = n / fiber_len;
        let strategy = if op == IndexOp::ScatterAdd
            && fiber_len >= LONG_FIBER
            && matches!(dtype, Dtype::I32 | Dtype::I64)
        {
            IndexStrategy::AtomicAdd
        } else if op == IndexOp::Scatter
            && fiber_len >= LONG_FIBER
            && fibers
                .checked_mul(bound)
                .is_some_and(|slots| slots <= ELECT_SCRATCH_LIMIT)
        {
            IndexStrategy::Elected
        } else if op == IndexOp::ScatterAdd
            && fiber_len >= LONG_FIBER
            && n <= radix_ops::MAX_ELEMENTS
            && fibers
                .checked_mul(bound)
                .is_some_and(|slots| slots < WORD_LIMIT - 1)
        {
            IndexStrategy::Grouped
        } else {
            IndexStrategy::Fibers
        };
        // a pass: the block above with its operation, dims word, flags, status
        // slot and unit count replaced
        let pass = |op_word: u32, dims: u32, flags: u32, slot: u32, units: u64| {
            let mut w = words;
            w[0] = op_word;
            w[2] = dims;
            w[3] = flags;
            w[26] = slot;
            w[27] = units as u32;
            IndexPass::new(w)
        };
        let index_flags = flags & (FLAG_I32_INDEX | FLAG_WRAP);
        let only = pass(words[0], words[2], words[3], status_slot, units);
        let mut radix = None;
        let (passes, pass_count, scratch_words) = match strategy {
            IndexStrategy::Fibers => ([only; 3], 1, 0),
            // an integer add: no alpha, and whole-word elements (no atomic_out)
            IndexStrategy::AtomicAdd => {
                let add = pass(op as u32, d as u32, flags | FLAG_ATOMIC_ADD, status_slot, n);
                ([add; 3], 1, 0)
            }
            IndexStrategy::Elected => {
                // clear and write report nothing: their binding 3 is no status word
                let slots = fibers * bound;
                (
                    [
                        pass(OP_CLEAR, d as u32, 0, 0, slots),
                        pass(OP_ELECT, d as u32, index_flags, status_slot, n),
                        pass(OP_WRITE, d as u32, index_flags | (flags & FLAG_VALUE), 0, n),
                    ],
                    3,
                    slots,
                )
            }
            IndexStrategy::Grouped => {
                // one row of n keys, each the destination slot as the
                // narrowest storage that holds the sentinel; the output code
                // is the module's own requirement and is never written
                let line = crate::convert_ops::View::new(0, [0, 0, 0, 1]);
                let sort = RadixPlan::new(
                    [1, 1, 1, n],
                    group_key_code(fibers * bound),
                    line,
                    Code::I32,
                    line,
                    false,
                )
                .expect("n is within the radix module's limit");
                let half = sort.digit_passes() & 1;
                let keys = pass(OP_KEYS, d as u32, index_flags, status_slot, n);
                // RUNS adds with add_elem, alpha included; its status slot
                // word names the half the sort's last pass wrote
                let runs = pass(
                    OP_RUNS,
                    d as u32,
                    flags & (index_flags | FLAG_ALPHA),
                    half,
                    n,
                );
                let words = sort.scratch_words();
                radix = Some(sort);
                ([keys, runs, runs], 2, words)
            }
        };
        Ok(Self {
            request,
            strategy,
            passes,
            pass_count,
            scratch_words,
            radix,
        })
    }

    /// A request in the C ABI's words (`ffi::AckIndexRequest`): the operation,
    /// storage and flags decoded by name, then planned against the bindings.
    /// `flags` takes `FLAG_I32_INDEX`, `FLAG_WRAP`, `FLAG_VALUE` (then `value`
    /// is the scalar source) and `FLAG_ALPHA` (then `alpha_bits` is the f32).
    #[allow(clippy::too_many_arguments)]
    pub fn from_abi(
        op: u32,
        dtype: u32,
        ndim: u32,
        dim: u32,
        flags: u32,
        status_slot: u32,
        shape: [u64; MAX_NDIM],
        index: View,
        src: View,
        out: View,
        bound: u64,
        value: u64,
        alpha_bits: u32,
        bindings: Bindings,
    ) -> Result<Self, IndexPlanError> {
        let op = IndexOp::try_from(op)?;
        let dtype = Dtype::try_from(dtype)?;
        let flags = request_flags(flags)?;
        let request = IndexRequest {
            op,
            dtype,
            ndim: usize::try_from(ndim).map_err(|_| IndexPlanError::NdimOutOfRange)?,
            shape,
            dim: usize::try_from(dim).map_err(|_| IndexPlanError::DimOutOfRange)?,
            index,
            index_i32: flags & FLAG_I32_INDEX != 0,
            src,
            out,
            bound,
            wrap: flags & FLAG_WRAP != 0,
            value: (flags & FLAG_VALUE != 0).then_some(value),
            alpha: (flags & FLAG_ALPHA != 0).then_some(alpha_bits),
            status_slot,
        };
        Self::new(request, bindings)
    }

    pub fn request(&self) -> &IndexRequest {
        &self.request
    }

    pub fn op(&self) -> IndexOp {
        self.request.op
    }

    pub fn strategy(&self) -> IndexStrategy {
        self.strategy
    }

    /// The passes to record, in order: one, or the elected scatter's three
    /// (clear, elect, write).
    pub fn passes(&self) -> &[IndexPass] {
        &self.passes[..self.pass_count]
    }

    /// The 32-bit scratch words the native entry allocates: fibers x bound
    /// for an [`IndexStrategy::Elected`] plan, the radix module's scratch for
    /// an [`IndexStrategy::Grouped`] one; 0 for the others.
    pub fn scratch_words(&self) -> u64 {
        self.scratch_words
    }

    /// A grouped scatter-add's sort: the digit passes the native entry records
    /// between the plan's KEYS and RUNS passes, with the radix module.
    pub fn radix(&self) -> Option<&RadixPlan> {
        self.radix.as_ref()
    }

    /// The radix module's digit passes of a grouped scatter-add, in order.
    pub fn radix_passes(&self) -> Vec<RadixPass> {
        self.radix.map(|r| r.digit_pass_list()).unwrap_or_default()
    }

    /// The pass that writes the output (the only pass, or the elected
    /// scatter's write).
    fn last(&self) -> &IndexPass {
        &self.passes[self.pass_count - 1]
    }

    /// Invocations with work in the pass that writes the output: output words
    /// (elements, for 8-byte storage) for a gather, fiber groups for a
    /// scatter's walk, elements for the element-parallel strategies.
    pub fn units(&self) -> u32 {
        self.last().units()
    }

    pub fn words(&self) -> [u32; PUSH_WORDS] {
        self.last().words()
    }

    pub fn push_constants(&self) -> [u8; PUSH_BYTES] {
        self.last().push_constants()
    }

    pub fn dispatch_groups(&self) -> [u32; 3] {
        self.last().dispatch_groups()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SRC: Storage = Storage::new(1, 1 << 20);
    const IDX: Storage = Storage::new(2, 1 << 20);
    const OUT: Storage = Storage::new(3, 1 << 20);
    const STATUS: Storage = Storage::new(4, 64);

    fn bindings() -> Bindings {
        Bindings {
            src: SRC,
            index: IDX,
            out: OUT,
            status: STATUS,
        }
    }

    /// torch.gather(src [4, 6], 1, index [4, 3]) into a contiguous out [4, 3].
    fn gather_request(dtype: Dtype) -> IndexRequest {
        IndexRequest {
            op: IndexOp::Gather,
            dtype,
            ndim: 2,
            shape: [4, 3, 0, 0],
            dim: 1,
            index: View::new(0, [3, 1, 0, 0]),
            index_i32: false,
            src: View::new(0, [6, 1, 0, 0]),
            out: View::new(0, [3, 1, 0, 0]),
            bound: 6,
            wrap: false,
            value: None,
            alpha: None,
            status_slot: 0,
        }
    }

    /// x.index_add_(0, index [5], src [5, 8]) into out [10, 8].
    fn index_add_request(dtype: Dtype) -> IndexRequest {
        IndexRequest {
            op: IndexOp::ScatterAdd,
            dtype,
            ndim: 2,
            shape: [5, 8, 0, 0],
            dim: 0,
            index: View::new(0, [1, 0, 0, 0]),
            index_i32: false,
            src: View::new(0, [8, 1, 0, 0]),
            out: View::new(0, [8, 1, 0, 0]),
            bound: 10,
            wrap: false,
            value: None,
            alpha: None,
            status_slot: 3,
        }
    }

    #[test]
    fn a_gather_pads_at_the_front_and_counts_output_words() {
        let plan = IndexPlan::new(gather_request(Dtype::F32), bindings()).unwrap();
        let w = plan.words();
        assert_eq!(&w[4..8], &[1, 1, 4, 3]);
        assert_eq!(w[2] & 0xF, 3, "dim 1 of a 2-D shape is dimension 3 of four");
        assert_eq!(plan.units(), 12);
        assert_eq!(plan.dispatch_groups(), [1, 1, 1]);
        // bf16: two elements a word, twelve elements in six words
        let plan = IndexPlan::new(gather_request(Dtype::Bf16), bindings()).unwrap();
        assert_eq!(plan.units(), 6);
        // uint8: four a word
        let plan = IndexPlan::new(gather_request(Dtype::U8), bindings()).unwrap();
        assert_eq!(plan.units(), 3);
        // int64: one element per invocation, two words each
        let plan = IndexPlan::new(gather_request(Dtype::I64), bindings()).unwrap();
        assert_eq!(plan.units(), 12);
    }

    #[test]
    fn the_abi_words_decode_by_name_and_plan_as_the_request_does() {
        let r = index_add_request(Dtype::F32);
        let from_abi = |op, dtype, flags| {
            IndexPlan::from_abi(
                op,
                dtype,
                r.ndim as u32,
                r.dim as u32,
                flags,
                r.status_slot,
                r.shape,
                r.index,
                r.src,
                r.out,
                r.bound,
                0,
                0,
                bindings(),
            )
        };
        let direct = IndexPlan::new(r, bindings()).unwrap();
        assert_eq!(from_abi(2, 0, 0).unwrap().words(), direct.words());
        assert_eq!(from_abi(3, 0, 0), Err(IndexPlanError::OperationOutOfRange));
        assert_eq!(from_abi(2, 9, 0), Err(IndexPlanError::DtypeOutOfRange));
        // the atomic output is the planner's to choose, never a request's
        assert_eq!(
            from_abi(2, 0, FLAG_ATOMIC_OUT),
            Err(IndexPlanError::FlagsOutOfRange)
        );
        assert_eq!(from_abi(2, 0, 32), Err(IndexPlanError::FlagsOutOfRange));
        // FLAG_ALPHA carries its f32 into word 24
        let with_alpha = IndexPlan::from_abi(
            2,
            0,
            r.ndim as u32,
            r.dim as u32,
            FLAG_ALPHA,
            r.status_slot,
            r.shape,
            r.index,
            r.src,
            r.out,
            r.bound,
            0,
            0.5f32.to_bits(),
            bindings(),
        )
        .unwrap();
        assert_eq!(with_alpha.words()[24], 0.5f32.to_bits());
        assert_eq!(with_alpha.words()[3], FLAG_ALPHA);
    }

    #[test]
    fn a_scatter_walks_fibers_and_groups_sub_word_ones() {
        // f32: one fiber per column, eight of them
        let plan = IndexPlan::new(index_add_request(Dtype::F32), bindings()).unwrap();
        assert_eq!(plan.units(), 8);
        assert_eq!((plan.words()[2] >> 8) & 0xF, 0);
        // bf16: two adjacent columns share a word, so four groups of two
        let plan = IndexPlan::new(index_add_request(Dtype::Bf16), bindings()).unwrap();
        assert_eq!(plan.units(), 4);
        assert_eq!(
            (plan.words()[2] >> 4) & 0xF,
            3,
            "inner is the column dimension"
        );
        assert_eq!((plan.words()[2] >> 8) & 0xF, 1, "log2 of the group of two");
        // int8: four columns a word, two groups
        let plan = IndexPlan::new(index_add_request(Dtype::I8), bindings()).unwrap();
        assert_eq!(plan.units(), 2);
    }

    /// The plan's (units, log2 G, atomic) for a request.
    fn scatter_mode(request: IndexRequest) -> (u32, u32, bool) {
        let plan = IndexPlan::new(request, bindings()).unwrap();
        let w = plan.words();
        (plan.units(), (w[2] >> 8) & 0xF, w[3] & FLAG_ATOMIC_OUT != 0)
    }

    #[test]
    fn a_sub_word_scatter_whose_fibers_share_words_is_written_with_atomics() {
        // schema 1 refused each of these as index-scatter-output-misaligned
        // six columns of bf16 at offset 1: the first word would be split
        let mut request = index_add_request(Dtype::Bf16);
        request.out.offset = 1;
        assert_eq!(
            scatter_mode(request),
            (8, 0, true),
            "a column an invocation"
        );
        // an odd column count along the inner dimension
        let mut request = index_add_request(Dtype::Bf16);
        request.shape = [5, 7, 0, 0];
        request.src.strides = [7, 1, 0, 0];
        request.out.strides = [7, 1, 0, 0];
        assert_eq!(scatter_mode(request), (7, 0, true));
        // a column-strided output
        let mut request = index_add_request(Dtype::I8);
        request.out.strides = [16, 2, 0, 0];
        assert_eq!(scatter_mode(request), (8, 0, true));
        // neither grouping nor a whole-word fiber is ever made atomic
        assert_eq!(scatter_mode(index_add_request(Dtype::Bf16)), (4, 1, false));
        assert_eq!(scatter_mode(index_add_request(Dtype::I8)), (2, 2, false));
        // f32, int64 and f64 (a scatter-add since schema 2) own their words
        // whatever the layout
        for dtype in [Dtype::F32, Dtype::I64, Dtype::F64] {
            let mut request = index_add_request(dtype);
            request.out.offset = 1;
            request.out.strides = [16, 2, 0, 0];
            assert_eq!(scatter_mode(request), (8, 0, false));
        }
        // a gather never writes with atomics: its output words are its own
        let plan = IndexPlan::new(gather_request(Dtype::U8), bindings()).unwrap();
        assert_eq!(plan.words()[3] & FLAG_ATOMIC_OUT, 0);
    }

    #[test]
    fn a_one_dimensional_scatter_is_one_fiber_whatever_its_storage() {
        // counts.scatter_add_(0, flat [7], ones [7]) into counts [5]: int64
        let request = IndexRequest {
            op: IndexOp::ScatterAdd,
            dtype: Dtype::I64,
            ndim: 1,
            shape: [7, 0, 0, 0],
            dim: 0,
            index: View::new(0, [1, 0, 0, 0]),
            index_i32: false,
            src: View::new(0, [1, 0, 0, 0]),
            out: View::new(0, [1, 0, 0, 0]),
            bound: 5,
            wrap: false,
            value: None,
            alpha: None,
            status_slot: 0,
        };
        let plan = IndexPlan::new(request, bindings()).unwrap();
        assert_eq!(plan.units(), 1);
        // a bool scatter along one dimension, at an odd offset: one fiber owns every word
        let mut request = request;
        request.op = IndexOp::Scatter;
        request.dtype = Dtype::U8;
        request.out.offset = 3;
        request.value = Some(1);
        let plan = IndexPlan::new(request, bindings()).unwrap();
        assert_eq!(plan.units(), 1);
    }

    #[test]
    fn a_bool_mask_scattered_along_its_rows_takes_a_row_an_invocation_with_atomics_if_rows_share_words(
    ) {
        // torch.zeros(3, W, dtype=bool).scatter(-1, idx [3, 2], True)
        let mask = |width: u64| IndexRequest {
            op: IndexOp::Scatter,
            dtype: Dtype::U8,
            ndim: 2,
            shape: [3, 2, 0, 0],
            dim: 1,
            index: View::new(0, [2, 1, 0, 0]),
            index_i32: false,
            src: View::new(0, [0; 4]),
            out: View::new(0, [width, 1, 0, 0]),
            bound: width,
            wrap: false,
            value: Some(1),
            alpha: None,
            status_slot: 0,
        };
        // twelve-byte rows own whole words: one row per invocation, no grouping
        assert_eq!(scatter_mode(mask(12)), (3, 0, false));
        // ten-byte rows: row 1 starts inside row 0's last word (the toy
        // stage-E step's last `scatter.value` refusals, MEASURED 2026-09-28)
        assert_eq!(scatter_mode(mask(10)), (3, 0, true));
    }

    #[test]
    fn refusals_are_by_name_in_order() {
        let mut r = gather_request(Dtype::F32);
        r.ndim = 5;
        assert_eq!(
            IndexPlan::new(r, bindings()),
            Err(IndexPlanError::NdimOutOfRange)
        );
        let mut r = gather_request(Dtype::F32);
        r.dim = 2;
        assert_eq!(
            IndexPlan::new(r, bindings()),
            Err(IndexPlanError::DimOutOfRange)
        );
        let mut r = gather_request(Dtype::F32);
        r.shape[0] = 0;
        assert_eq!(
            IndexPlan::new(r, bindings()),
            Err(IndexPlanError::ZeroElements)
        );
        let mut r = gather_request(Dtype::F32);
        r.bound = 0;
        assert_eq!(
            IndexPlan::new(r, bindings()),
            Err(IndexPlanError::BoundOutOfRange)
        );
        let mut r = gather_request(Dtype::F32);
        r.value = Some(0);
        assert_eq!(
            IndexPlan::new(r, bindings()),
            Err(IndexPlanError::ValueNeedsScatter)
        );
        let mut r = index_add_request(Dtype::Bf16);
        r.alpha = Some(2.0f32.to_bits());
        assert_eq!(
            IndexPlan::new(r, bindings()),
            Err(IndexPlanError::AlphaNeedsF32ScatterAdd)
        );
        // a float64 scatter-add plans since schema 2 (schema 1 refused it as
        // index-add-unsupported-dtype); an alpha on it is still refused
        assert!(IndexPlan::new(index_add_request(Dtype::F64), bindings()).is_ok());
        let mut r = index_add_request(Dtype::F64);
        r.alpha = Some(2.0f32.to_bits());
        assert_eq!(
            IndexPlan::new(r, bindings()),
            Err(IndexPlanError::AlphaNeedsF32ScatterAdd)
        );
        let mut r = gather_request(Dtype::F32);
        r.status_slot = 16;
        assert_eq!(
            IndexPlan::new(r, bindings()),
            Err(IndexPlanError::StatusSlotOutOfRange)
        );
        // a source view past its buffer: the gather reads `bound` rows along dim
        let mut b = bindings();
        b.src = Storage::new(1, 4 * 4 * 6 - 4);
        assert_eq!(
            IndexPlan::new(gather_request(Dtype::F32), b),
            Err(IndexPlanError::AddressOverflow)
        );
        let mut r = gather_request(Dtype::F32);
        r.out.strides = [1, 4, 0, 0];
        assert_eq!(
            IndexPlan::new(r, bindings()),
            Err(IndexPlanError::GatherOutputNotContiguous)
        );
        let mut r = index_add_request(Dtype::F32);
        r.out.strides = [0, 1, 0, 0];
        assert_eq!(
            IndexPlan::new(r, bindings()),
            Err(IndexPlanError::OutputSelfOverlap)
        );
        let mut b = bindings();
        b.out = IDX;
        assert_eq!(
            IndexPlan::new(gather_request(Dtype::F32), b),
            Err(IndexPlanError::OutputOverlapsInput)
        );
        let mut b = bindings();
        b.status = Storage::new(1, 64);
        assert_eq!(
            IndexPlan::new(gather_request(Dtype::F32), b),
            Err(IndexPlanError::StatusOverlapsOperand)
        );
    }

    #[test]
    fn capacity_is_the_logical_size_to_the_byte() {
        // a three-byte uint8 tensor: three elements fit, a fourth does not,
        // although the buffer's word holds four bytes
        let request = |n: u64| IndexRequest {
            op: IndexOp::Gather,
            dtype: Dtype::U8,
            ndim: 1,
            shape: [n, 0, 0, 0],
            dim: 0,
            index: View::new(0, [1, 0, 0, 0]),
            index_i32: false,
            src: View::new(0, [1, 0, 0, 0]),
            out: View::new(0, [1, 0, 0, 0]),
            bound: n,
            wrap: false,
            value: None,
            alpha: None,
            status_slot: 0,
        };
        let b = |bytes| Bindings {
            src: Storage::new(1, bytes),
            index: Storage::new(2, 64),
            out: Storage::new(3, bytes),
            status: STATUS,
        };
        assert!(IndexPlan::new(request(3), b(3)).is_ok());
        assert_eq!(
            IndexPlan::new(request(4), b(3)),
            Err(IndexPlanError::AddressOverflow)
        );
        // an int64 index of 8 elements needs 64 bytes, not 60
        let mut r = request(8);
        r.dtype = Dtype::F32;
        let mut short = b(32);
        short.index = Storage::new(2, 60);
        assert_eq!(
            IndexPlan::new(r, short),
            Err(IndexPlanError::AddressOverflow)
        );
    }

    #[test]
    fn a_value_scatter_may_bind_its_unread_source_to_the_output() {
        let mut request = index_add_request(Dtype::F32);
        request.op = IndexOp::Scatter;
        request.value = Some(u64::from(1.5f32.to_bits()));
        let mut b = bindings();
        b.src = OUT;
        let plan = IndexPlan::new(request, b).unwrap();
        assert_eq!(plan.words()[3] & FLAG_VALUE, FLAG_VALUE);
        assert_eq!(plan.words()[24], 1.5f32.to_bits());
    }

    #[test]
    fn a_large_gather_spreads_over_a_second_grid_axis() {
        let mut request = gather_request(Dtype::F32);
        request.shape = [1 << 13, 1 << 10, 0, 0];
        request.index.strides = [1 << 10, 1, 0, 0];
        request.out.strides = [1 << 10, 1, 0, 0];
        request.src.strides = [1 << 10, 1, 0, 0];
        request.bound = 1 << 10;
        let b = Bindings {
            src: Storage::new(1, 1 << 26),
            index: Storage::new(2, 1 << 27),
            out: Storage::new(3, 1 << 26),
            status: Storage::new(4, 64),
        };
        let plan = IndexPlan::new(request, b).unwrap();
        let [x, y, z] = plan.dispatch_groups();
        assert_eq!(z, 1);
        assert!(x <= MAX_GROUPS_X);
        assert!(u64::from(x) * u64::from(y) * 64 >= u64::from(plan.units()));
        assert!(y > 1);
    }

    /// The MoE routing's 1-D form: `n` (token, slot) pairs along one fiber
    /// into `bound` destinations, contiguous, int64 indices.
    fn routing_request(op: IndexOp, dtype: Dtype, n: u64, bound: u64) -> IndexRequest {
        IndexRequest {
            op,
            dtype,
            ndim: 1,
            shape: [n, 0, 0, 0],
            dim: 0,
            index: View::new(0, [1, 0, 0, 0]),
            index_i32: false,
            src: View::new(0, [1, 0, 0, 0]),
            out: View::new(0, [1, 0, 0, 0]),
            bound,
            wrap: false,
            value: None,
            alpha: None,
            status_slot: 3,
        }
    }

    fn big_bindings() -> Bindings {
        Bindings {
            src: Storage::new(1, 1 << 26),
            index: Storage::new(2, 1 << 26),
            out: Storage::new(3, 1 << 26),
            status: Storage::new(4, 64),
        }
    }

    #[test]
    fn a_long_integer_scatter_add_runs_an_element_an_invocation_with_atomics() {
        // counts.scatter_add_(0, flat [65536], ones) into counts [128]
        for dtype in [Dtype::I64, Dtype::I32] {
            let plan = IndexPlan::new(
                routing_request(IndexOp::ScatterAdd, dtype, 65_536, 128),
                big_bindings(),
            )
            .unwrap();
            assert_eq!(plan.strategy(), IndexStrategy::AtomicAdd);
            assert_eq!(plan.passes().len(), 1);
            assert_eq!(plan.scratch_words(), 0);
            let w = plan.words();
            assert_eq!(w[0], IndexOp::ScatterAdd as u32);
            assert_eq!(w[3], FLAG_ATOMIC_ADD);
            assert_eq!(
                w[2], 3,
                "dim 0 padded to 3, no inner dimension, no grouping"
            );
            assert_eq!(w[26], 3, "the request's status slot");
            assert_eq!(plan.units(), 65_536);
            let [x, y, _] = plan.dispatch_groups();
            assert!(u64::from(x) * u64::from(y) * 64 >= 65_536);
        }
    }

    #[test]
    fn a_float_scatter_add_is_grouped_by_destination_in_its_fibers_order() {
        // the load-balancing loss's counts: float64 ones into 128 experts
        for dtype in [
            Dtype::F64,
            Dtype::F32,
            Dtype::Bf16,
            Dtype::F16,
            Dtype::I16,
            Dtype::I8,
            Dtype::U8,
        ] {
            let plan = IndexPlan::new(
                routing_request(IndexOp::ScatterAdd, dtype, 65_536, 128),
                big_bindings(),
            )
            .unwrap();
            assert_eq!(plan.strategy(), IndexStrategy::Grouped, "{dtype:?}");
            let passes = plan.passes();
            assert_eq!(passes.len(), 2);
            assert_eq!((passes[0].op(), passes[1].op()), (OP_KEYS, OP_RUNS));
            assert_eq!((passes[0].units(), passes[1].units()), (65_536, 65_536));
            // 128 slots: one-byte keys, one digit pass, so RUNS reads half 1
            let sort = plan.radix().expect("a grouped plan sorts");
            assert_eq!(sort.digit_passes(), 1);
            assert_eq!(plan.radix_passes().len(), 3, "histogram, scan, scatter");
            assert_eq!(
                passes[0].words()[26],
                3,
                "KEYS reports in the request's slot"
            );
            assert_eq!(passes[1].words()[26], 1, "RUNS names the last pass's half");
            assert_eq!(plan.scratch_words(), sort.scratch_words());
        }
        // the key widens with the slots, and alpha reaches the RUNS pass only
        let key_passes = |bound: u64| {
            IndexPlan::new(
                routing_request(IndexOp::ScatterAdd, Dtype::F32, 4_096, bound),
                big_bindings(),
            )
            .unwrap()
            .radix()
            .map(|r| r.digit_passes())
        };
        assert_eq!(key_passes(255), Some(1));
        assert_eq!(key_passes(256), Some(2));
        assert_eq!(key_passes(65_535), Some(2));
        assert_eq!(key_passes(65_536), Some(4));
        let mut r = routing_request(IndexOp::ScatterAdd, Dtype::F32, 4_096, 300);
        r.alpha = Some(0.5f32.to_bits());
        let plan = IndexPlan::new(r, big_bindings()).unwrap();
        assert_eq!(plan.passes()[0].words()[3] & FLAG_ALPHA, 0);
        assert_eq!(plan.passes()[1].words()[3] & FLAG_ALPHA, FLAG_ALPHA);
        // a short fiber keeps its walk
        let plan = IndexPlan::new(
            routing_request(IndexOp::ScatterAdd, Dtype::F64, LONG_FIBER - 1, 128),
            big_bindings(),
        )
        .unwrap();
        assert_eq!(plan.strategy(), IndexStrategy::Fibers);
        assert!(plan.radix().is_none());
    }

    #[test]
    fn the_native_entry_can_tell_a_grouped_plan_from_the_words_alone() {
        let shape = [65_536, 0, 0, 0];
        for dtype in [Dtype::F32, Dtype::F64, Dtype::U8] {
            assert!(may_plan_grouped(2, dtype as u32, 1, 0, shape));
        }
        assert!(
            !may_plan_grouped(2, Dtype::I64 as u32, 1, 0, shape),
            "atomics"
        );
        assert!(
            !may_plan_grouped(1, Dtype::F32 as u32, 1, 0, shape),
            "a scatter"
        );
        assert!(
            !may_plan_grouped(2, Dtype::F32 as u32, 1, 0, [255, 0, 0, 0]),
            "short"
        );
        assert!(
            !may_plan_grouped(2, Dtype::F32 as u32, 1, 1, shape),
            "dim past ndim"
        );
        // wherever the planner groups, the words said it might
        for (dtype, n) in [
            (Dtype::F64, 65_536u64),
            (Dtype::F32, 256),
            (Dtype::I8, 1_000),
        ] {
            let plan = IndexPlan::new(
                routing_request(IndexOp::ScatterAdd, dtype, n, 128),
                big_bindings(),
            )
            .unwrap();
            assert_eq!(plan.strategy(), IndexStrategy::Grouped);
            assert!(may_plan_grouped(2, dtype as u32, 1, 0, [n, 0, 0, 0]));
        }
    }

    #[test]
    fn a_long_scatter_is_elected_in_three_passes_over_one_word_a_destination() {
        // padded_src.scatter_(0, padded_of_sorted [65536], rows) into [70000]
        let plan = IndexPlan::new(
            routing_request(IndexOp::Scatter, Dtype::I64, 65_536, 70_000),
            big_bindings(),
        )
        .unwrap();
        assert_eq!(plan.strategy(), IndexStrategy::Elected);
        assert_eq!(plan.scratch_words(), 70_000);
        let passes = plan.passes();
        assert_eq!(passes.len(), 3);
        assert_eq!(
            passes.iter().map(|p| p.op()).collect::<Vec<_>>(),
            vec![OP_CLEAR, OP_ELECT, OP_WRITE]
        );
        assert_eq!(
            passes.iter().map(|p| p.units()).collect::<Vec<_>>(),
            vec![70_000, 65_536, 65_536]
        );
        // only the elect pass reports: the others' binding 3 is no status word
        assert_eq!(
            passes.iter().map(|p| p.words()[26]).collect::<Vec<_>>(),
            vec![0, 3, 0]
        );
        assert!(passes.iter().all(|p| p.words()[3] == 0));
        // the plan's own words are the pass that writes the output
        assert_eq!(plan.words(), passes[2].words());
        for p in passes {
            let [x, y, _] = p.dispatch_groups();
            assert!(u64::from(x) * u64::from(y) * 64 >= u64::from(p.units()));
        }
    }

    #[test]
    fn the_elected_passes_carry_the_index_flags_and_only_the_write_carries_the_value() {
        let mut request = routing_request(IndexOp::Scatter, Dtype::U8, 1_000, 5_000);
        request.index_i32 = true;
        request.wrap = true;
        request.value = Some(1);
        let plan = IndexPlan::new(request, big_bindings()).unwrap();
        assert_eq!(plan.strategy(), IndexStrategy::Elected);
        let flags: Vec<u32> = plan.passes().iter().map(|p| p.words()[3]).collect();
        let index = FLAG_I32_INDEX | FLAG_WRAP;
        assert_eq!(flags, vec![0, index, index | FLAG_VALUE]);
        assert_eq!(plan.passes()[2].words()[24], 1, "the scalar");
    }

    #[test]
    fn the_long_fiber_threshold_and_the_scratch_limit_choose_the_walk() {
        let strategy = |n: u64, bound: u64| {
            IndexPlan::new(
                routing_request(IndexOp::Scatter, Dtype::F32, n, bound),
                big_bindings(),
            )
            .unwrap()
            .strategy()
        };
        assert_eq!(strategy(LONG_FIBER - 1, 1_000), IndexStrategy::Fibers);
        assert_eq!(strategy(LONG_FIBER, 1_000), IndexStrategy::Elected);
        assert_eq!(strategy(1_000, ELECT_SCRATCH_LIMIT), IndexStrategy::Elected);
        let mut request =
            routing_request(IndexOp::Scatter, Dtype::F32, 1_000, ELECT_SCRATCH_LIMIT + 1);
        let huge = Bindings {
            out: Storage::new(3, 1 << 30),
            ..big_bindings()
        };
        assert_eq!(
            IndexPlan::new(request, huge).unwrap().strategy(),
            IndexStrategy::Fibers
        );
        // a gather is never elected, however long
        request.op = IndexOp::Gather;
        request.bound = 4;
        assert_eq!(
            IndexPlan::new(request, big_bindings()).unwrap().strategy(),
            IndexStrategy::Fibers
        );
    }

    #[test]
    fn several_long_fibers_share_one_scratch_a_bound_apart() {
        // out [4, 3000].scatter_(1, idx [4, 2048], src [4, 2048])
        let request = IndexRequest {
            op: IndexOp::Scatter,
            dtype: Dtype::F32,
            ndim: 2,
            shape: [4, 2_048, 0, 0],
            dim: 1,
            index: View::new(0, [2_048, 1, 0, 0]),
            index_i32: false,
            src: View::new(0, [2_048, 1, 0, 0]),
            out: View::new(0, [3_000, 1, 0, 0]),
            bound: 3_000,
            wrap: false,
            value: None,
            alpha: None,
            status_slot: 0,
        };
        let plan = IndexPlan::new(request, big_bindings()).unwrap();
        assert_eq!(plan.strategy(), IndexStrategy::Elected);
        assert_eq!(plan.scratch_words(), 4 * 3_000);
        assert_eq!(plan.passes()[1].units(), 4 * 2_048);
        assert_eq!(plan.passes()[1].words()[2], 3, "dim 1 of 2, padded to 3");
    }
}
