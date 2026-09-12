//! Pure geometry, capacity, aliasing and dispatch planning for `pointwise-f32/v1`
//! (operation schema 2).
//!
//! The shader evaluates `out[i] = f(x[i], y[i], z[i])` for every linear index
//! `i` of one row-major shape of at most four dimensions. Each operand
//! addresses its storage through its own element offset and per-dimension
//! element strides, so a view, a transposed view, or a broadcast (stride 0)
//! all reach the kernel without a copy. The four operands are `x`, `y`, `z`
//! and `out`; which of the inputs an operation reads is [`PointwiseOp`]'s
//! word (`z` only by `addcmul` and `addcdiv`), and the scalar-`z` flag is
//! reserved: never set here and refused in a block.
//!
//! Schema 2 (2026-09-07) appended the operations 6-23 to schema 1's 0-5 with
//! the same 120-byte block, buffers and entry: a new operation is a new value
//! of the operation word, so the family's schema pin moved and its layout did
//! not. Each operation's operand slots and scalar words are documented on its
//! variant and in the shader header.
//!
//! This module has no native `Context` or `Buffer` dependency. Its storage
//! declarations (identity and capacity) are the caller's, so a successful plan
//! does not prove buffer ownership, liveness, actual capacity, or which buffers
//! are really the same allocation. The native integration must establish those
//! properties from real buffer objects immediately before binding and dispatch
//! and hand this module what it found.
//!
//! Refusals are by name, checked in this order: operation (at decode), number
//! of dimensions (and `triu`'s need for two), zero elements, address
//! arithmetic and capacity for `x`, `y`, `z`, `out` in turn, the output's own
//! overlap, then each input sharing the output's storage.

use std::fmt;

pub const ABI: &str = "pointwise-f32/v1";
/// The operation schema the C ABI reports for the family
/// (`ffi::ACK_OP_SCHEMA_POINTWISE`): 1 was operations 0-5, 2 adds 6-23,
/// 3 (2026-09-10) adds flags bits 2 and 3, the bf16 storage switches, in
/// the same block through the same entry.
pub const SCHEMA: u32 = 3;
/// The last operation word this schema decodes; `try_from` refuses beyond it.
pub const LAST_OP: u32 = 23;
/// Dimensions the push block carries; fewer are padded with extent 1, stride 0.
pub const MAX_NDIM: usize = 4;
/// `x`, `y`, `z`, `out`, in binding order.
pub const OPERANDS: usize = 4;
pub const WORKGROUP_INVOCATIONS: u32 = 256;
pub const ELEMENTS_PER_INVOCATION: u32 = 4;
/// Elements one workgroup handles; the dispatch is `ceil(n / 1024)` groups.
pub const ELEMENTS_PER_GROUP: u32 = WORKGROUP_INVOCATIONS * ELEMENTS_PER_INVOCATION;
/// Every address a view touches is below this, so the shader's u32 extent
/// arithmetic cannot wrap and a buffer index always fits a signed 32-bit int.
pub const ADDRESS_LIMIT: u64 = 1 << 31;
pub const PUSH_BYTES: usize = 120;
pub const PUSH_WORDS: usize = PUSH_BYTES / 4;
/// Flags word bit 0: `y` is the scalar in word 29, and the `y` buffer is not read.
pub const FLAG_Y_SCALAR: u32 = 1;
/// Flags bits 2-5 (schema 3, 2026-09-10): buffer `x`, `y`, `z` and `out`
/// hold bf16 rather than f32. ONE BIT PER BUFFER, not one for the inputs
/// together, because a slot the operation does not read is bound to whatever
/// the caller had to hand -- the output for `zero_`, the source for a copy --
/// and a guessed dtype makes that slot's capacity check too generous, which is
/// a weakened guard.
///
/// They start at bit 2 because bit 1 is `FLAG_Z_SCALAR`, reserved and named
/// though not yet implemented: taking it would have made a future z-scalar
/// block read as a bf16 one.
///
/// The SHADER has one input switch, because its three input buffers share a
/// storage type there. So the slots an operation READS must agree, and a block
/// whose read inputs disagree is refused by name. The output is independent,
/// which is what makes f32 in with bf16 out a strided cast through any view --
/// the thing the adapter's copy path needs and a whole-block `ack_cast` cannot
/// express.
///
/// The arithmetic is f32 in every combination: a value is widened at the load
/// and narrowed at the store, and nothing else changes.
pub const FLAG_BF16: [u32; OPERANDS] = [4, 8, 16, 32];
const STORAGE_FLAGS: u32 = FLAG_BF16[0] | FLAG_BF16[1] | FLAG_BF16[2] | FLAG_BF16[3];

/// Bytes one element of slot `index` occupies under a flags word.
pub fn slot_element_bytes(flags: u32, index: usize) -> u64 {
    if flags & FLAG_BF16[index] != 0 {
        2
    } else {
        4
    }
}

/// The module a flags word selects: the four bits, x in bit 0 through out in
/// bit 3, as an index into the sixteen the context can build. 0 is the f32
/// module every device builds at open.
pub fn storage_module(flags: u32) -> usize {
    (0..OPERANDS)
        .map(|i| usize::from(flags & FLAG_BF16[i] != 0) << i)
        .sum()
}
/// Flags word bit 1: reserved for a scalar `z`; this version never sets it
/// and refuses a block that does.
pub const FLAG_Z_SCALAR: u32 = 2;
const KNOWN_FLAGS: u32 = FLAG_Y_SCALAR | STORAGE_FLAGS;

/// The operation word. `a` and `b` are the two scalar words ([`Scalars`]);
/// every operation that reads `y` reads the scalar `b` instead when the plan
/// carries a scalar `y` (flags bit 0).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum PointwiseOp {
    /// `x + a * y`
    AddScaled = 0,
    /// `x - a * y`
    SubScaled = 1,
    /// `x * y`
    Mul = 2,
    /// `x / y`
    Div = 3,
    /// `a`
    Fill = 4,
    /// `x`
    Copy = 5,
    /// `1 / x`
    Reciprocal = 6,
    /// `|x|`
    Abs = 7,
    /// `sqrt(x)`, NaN below zero as on the CPU
    Sqrt = 8,
    /// `1 / sqrt(x)`
    Rsqrt = 9,
    /// `-x`
    Neg = 10,
    /// `x ^ y`, `y` the exponent (a buffer, or the scalar `b`): the CPU pow
    /// kernel's own forms for 2, 3, -2, 0.5, -0.5 and -1, otherwise
    /// `std::pow`'s semantics (NaN for a negative base and a non-integer
    /// exponent)
    Pow = 11,
    /// `a ^ x`, `a` the base (torch's `pow.Scalar`)
    PowScalarBase = 12,
    /// `min(max(x, a), b)` with a NaN `x` propagated and `a > b` giving `b`
    /// everywhere, as torch does; an absent bound is `-inf` / `+inf`
    /// ([`Scalars::clamp`]), so `clamp_min` and `clamp_max` are this operation
    Clamp = 13,
    /// `x / (1 + exp(-x))`
    Silu = 14,
    /// `1 / (1 + exp(-x))`
    Sigmoid = 15,
    /// `x * (1 - y) * y`: `x` the gradient, `y` the sigmoid output
    SigmoidBackward = 16,
    /// `x + a * y * z`
    Addcmul = 17,
    /// `x + a * y / z`
    Addcdiv = 18,
    /// torch's two-branch `lerp(x, y, a)`: `x` the start, `y` the end, `a`
    /// the weight
    Lerp = 19,
    /// `cos(x)`, quadrant-reduced in the shader; NaN outside the reduction's
    /// proven domain `|x| <= 411774.03125` (derived and measured in the
    /// shader header), so a wrong answer is loud rather than plausible
    Cos = 20,
    /// `sin(x)`, likewise
    Sin = 21,
    /// `a + i * b` over the row-major linear index `i` ([`Scalars::iota`]);
    /// reads no input. Evaluated entirely in f32, so the error is absolute in
    /// the magnitudes the range travels through (`<= 2^-21 * max(|a|, |a + n*b|)`,
    /// derived; measured 2^-22.06), not relative to the element: this is not
    /// the family tolerance, and the adapter documents the difference
    Iota = 22,
    /// `x` where `column - row >= diagonal` over the last two dimensions,
    /// else 0; the diagonal is the `b` word as an `i32` ([`Scalars::triu`])
    /// and the shape needs at least two dimensions
    Triu = 23,
}

impl PointwiseOp {
    pub fn reads_x(self) -> bool {
        !matches!(self, Self::Fill | Self::Iota)
    }

    pub fn reads_y(self) -> bool {
        matches!(
            self,
            Self::AddScaled
                | Self::SubScaled
                | Self::Mul
                | Self::Div
                | Self::Pow
                | Self::SigmoidBackward
                | Self::Addcmul
                | Self::Addcdiv
                | Self::Lerp
        )
    }

    pub fn reads_z(self) -> bool {
        matches!(self, Self::Addcmul | Self::Addcdiv)
    }
}

impl TryFrom<u32> for PointwiseOp {
    type Error = PointwisePlanError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::AddScaled),
            1 => Ok(Self::SubScaled),
            2 => Ok(Self::Mul),
            3 => Ok(Self::Div),
            4 => Ok(Self::Fill),
            5 => Ok(Self::Copy),
            6 => Ok(Self::Reciprocal),
            7 => Ok(Self::Abs),
            8 => Ok(Self::Sqrt),
            9 => Ok(Self::Rsqrt),
            10 => Ok(Self::Neg),
            11 => Ok(Self::Pow),
            12 => Ok(Self::PowScalarBase),
            13 => Ok(Self::Clamp),
            14 => Ok(Self::Silu),
            15 => Ok(Self::Sigmoid),
            16 => Ok(Self::SigmoidBackward),
            17 => Ok(Self::Addcmul),
            18 => Ok(Self::Addcdiv),
            19 => Ok(Self::Lerp),
            20 => Ok(Self::Cos),
            21 => Ok(Self::Sin),
            22 => Ok(Self::Iota),
            23 => Ok(Self::Triu),
            _ => Err(PointwisePlanError::OperationOutOfRange),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(usize)]
pub enum Operand {
    X = 0,
    Y = 1,
    Z = 2,
    Out = 3,
}

impl Operand {
    pub const ALL: [Operand; OPERANDS] = [Self::X, Self::Y, Self::Z, Self::Out];
    pub const INPUTS: [Operand; OPERANDS - 1] = [Self::X, Self::Y, Self::Z];

    pub fn index(self) -> usize {
        self as usize
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PointwisePlanError {
    OperationOutOfRange,
    NdimOutOfRange,
    /// `triu` over a shape of fewer than two dimensions (schema 2).
    TriuNeedsTwoDimensions,
    ZeroElements,
    AddressOverflow,
    OutputOverlapsInput,
    OutputSelfOverlap,
    FlagsOutOfRange,
    ElementCountMismatch,
}

impl fmt::Display for PointwisePlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::OperationOutOfRange => "pointwise-operation-out-of-range",
            Self::NdimOutOfRange => "pointwise-ndim-out-of-range",
            Self::TriuNeedsTwoDimensions => "pointwise-triu-needs-two-dimensions",
            Self::ZeroElements => "pointwise-zero-elements",
            Self::AddressOverflow => "pointwise-address-overflow",
            Self::OutputOverlapsInput => "pointwise-output-overlaps-input",
            Self::OutputSelfOverlap => "pointwise-output-self-overlap",
            Self::FlagsOutOfRange => "pointwise-flags-out-of-range",
            Self::ElementCountMismatch => "pointwise-element-count-mismatch",
        })
    }
}

impl std::error::Error for PointwisePlanError {}

/// One operand's element offset and per-dimension element strides, indexed
/// like the plan's shape. Entries beyond the shape's length are ignored, and a
/// stride on a dimension of extent 1 never contributes to an address, so two
/// views that address the same elements compare equal once planned.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StridedView {
    offset: u64,
    strides: [u64; MAX_NDIM],
}

impl StridedView {
    pub const fn new(offset: u64, strides: [u64; MAX_NDIM]) -> Self {
        Self { offset, strides }
    }

    /// Row-major strides for `shape` at offset zero.
    pub fn contiguous(shape: &[u64]) -> Result<Self, PointwisePlanError> {
        if shape.len() > MAX_NDIM {
            return Err(PointwisePlanError::NdimOutOfRange);
        }
        let mut strides = [0u64; MAX_NDIM];
        let mut stride = 1u64;
        for (dimension, &extent) in shape.iter().enumerate().rev() {
            strides[dimension] = stride;
            stride = stride
                .checked_mul(extent)
                .ok_or(PointwisePlanError::AddressOverflow)?;
        }
        Ok(Self { offset: 0, strides })
    }

    pub fn offset(&self) -> u64 {
        self.offset
    }

    pub fn strides(&self) -> [u64; MAX_NDIM] {
        self.strides
    }

    /// The same addresses with the irrelevant strides zeroed, after the highest
    /// address the view touches is checked against the limit and the capacity.
    fn canonical(
        &self,
        shape: &[u64],
        capacity_elements: u64,
    ) -> Result<StridedView, PointwisePlanError> {
        let mut strides = [0u64; MAX_NDIM];
        let mut max_index = self.offset;
        for (dimension, &extent) in shape.iter().enumerate() {
            if extent > 1 {
                strides[dimension] = self.strides[dimension];
                let span = (extent - 1)
                    .checked_mul(self.strides[dimension])
                    .ok_or(PointwisePlanError::AddressOverflow)?;
                max_index = max_index
                    .checked_add(span)
                    .ok_or(PointwisePlanError::AddressOverflow)?;
            }
        }
        if max_index >= ADDRESS_LIMIT || max_index >= capacity_elements {
            return Err(PointwisePlanError::AddressOverflow);
        }
        Ok(StridedView {
            offset: self.offset,
            strides,
        })
    }

    /// Sufficient condition for every coordinate to reach a distinct address:
    /// ordered by stride, each stride is at least the span of the dimensions
    /// inside it. Every slice, step, transpose or permutation of a compact
    /// tensor satisfies it; a zero stride or two equal strides on dimensions
    /// of extent above one do not. Call on a canonical view.
    fn addresses_are_distinct(&self, shape: &[u64]) -> bool {
        let mut dimensions: Vec<(u64, u64)> = shape
            .iter()
            .zip(self.strides)
            .filter(|(&extent, _)| extent > 1)
            .map(|(&extent, stride)| (stride, extent))
            .collect();
        dimensions.sort_unstable();
        let mut span = 1u64;
        for (stride, extent) in dimensions {
            if stride < span {
                return false;
            }
            // Bounded by the maximum address, which `canonical` proved is below
            // ADDRESS_LIMIT, so this arithmetic cannot overflow.
            span += stride * (extent - 1);
        }
        true
    }

    fn push_words(&self) -> [u32; MAX_NDIM + 1] {
        // A canonical view's offset and contributing strides are below
        // ADDRESS_LIMIT; the others are zero.
        let mut words = [0u32; MAX_NDIM + 1];
        for (word, stride) in words.iter_mut().zip(self.strides) {
            *word = stride as u32;
        }
        words[MAX_NDIM] = self.offset as u32;
        words
    }
}

/// The storage an operand is bound to, as the caller declares it: an identity
/// that is equal exactly when two operands share one allocation, and that
/// allocation's capacity in FP32 elements. The native integration takes both
/// from the real buffer objects.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Storage {
    identity: u64,
    capacity_elements: u64,
}

impl Storage {
    pub const fn new(identity: u64, capacity_elements: u64) -> Self {
        Self {
            identity,
            capacity_elements,
        }
    }

    pub fn identity(&self) -> u64 {
        self.identity
    }

    pub fn capacity_elements(&self) -> u64 {
        self.capacity_elements
    }
}

/// The two scalar words of the block, `a` (word 28) and `b` (word 29), kept
/// as their bits so a plan is comparable and re-encodes exactly. `a` is the
/// alpha of the scaled operations, the fill value, the base of
/// `PowScalarBase`, the lower bound of `Clamp`, the weight of `Lerp`, the
/// start of `Iota`; `b` is the scalar `y` when flags bit 0 is set (every
/// operation that reads `y` then reads it here instead of the `y` buffer),
/// the upper bound of `Clamp`, the step of `Iota`, and the diagonal of `Triu`
/// as an `i32`. An operation reads the words it names and ignores the rest.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Scalars {
    a_bits: u32,
    b_bits: u32,
    y_is_scalar: bool,
}

impl Scalars {
    /// `a` alone; `b` is zero and `y` comes from its buffer.
    pub const fn new(a: f32) -> Self {
        Self::from_words(a.to_bits(), 0, false)
    }

    /// `a`, and the scalar `y` in `b` (flags bit 0).
    pub const fn with_scalar_y(a: f32, y: f32) -> Self {
        Self::from_words(a.to_bits(), y.to_bits(), true)
    }

    /// Both words as floats with `y` from its buffer: `Clamp` (bounds) and
    /// `Iota` (start, step) name them.
    pub const fn pair(a: f32, b: f32) -> Self {
        Self::from_words(a.to_bits(), b.to_bits(), false)
    }

    /// `Clamp`'s bounds: an absent lower bound is `-inf`, an absent upper
    /// bound `+inf`, so `clamp_min` and `clamp_max` need no operation of
    /// their own and every value compares as torch's `std::min(std::max(x,
    /// low), high)` does.
    pub const fn clamp(low: Option<f32>, high: Option<f32>) -> Self {
        let low = match low {
            Some(value) => value,
            None => f32::NEG_INFINITY,
        };
        let high = match high {
            Some(value) => value,
            None => f32::INFINITY,
        };
        Self::pair(low, high)
    }

    /// `Iota`'s `start` and `step`.
    pub const fn iota(start: f32, step: f32) -> Self {
        Self::pair(start, step)
    }

    /// `Triu`'s diagonal offset, an `i32` in the `b` word (`a` is zero).
    pub const fn triu(diagonal: i32) -> Self {
        Self::from_words(0, diagonal as u32, false)
    }

    const fn from_words(a_bits: u32, b_bits: u32, y_is_scalar: bool) -> Self {
        Self {
            a_bits,
            b_bits,
            y_is_scalar,
        }
    }

    pub fn a(&self) -> f32 {
        f32::from_bits(self.a_bits)
    }

    /// The `b` word as a float (the scalar `y`, the upper bound, the step).
    pub fn b(&self) -> f32 {
        f32::from_bits(self.b_bits)
    }

    /// The `b` word as `Triu` reads it.
    pub fn diagonal(&self) -> i32 {
        self.b_bits as i32
    }

    /// The scalar `y`, when the plan carries one.
    pub fn y(&self) -> Option<f32> {
        if self.y_is_scalar {
            Some(self.b())
        } else {
            None
        }
    }

    fn flags(&self) -> u32 {
        if self.y_is_scalar {
            FLAG_Y_SCALAR
        } else {
            0
        }
    }
}

/// Immutable geometry, aliasing verdict and dispatch metadata for one
/// four-buffer pointwise operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PointwisePlan {
    op: PointwiseOp,
    ndim: u32,
    n: u32,
    shape: [u32; MAX_NDIM],
    views: [StridedView; OPERANDS],
    storage: [Storage; OPERANDS],
    scalars: Scalars,
    /// FLAG_IN_BF16 | FLAG_OUT_BF16 as the block carried them, kept so a
    /// decoded plan re-encodes to the same bytes.
    storage_flags: u32,
    dispatch_groups: [u32; 3],
}

impl PointwisePlan {
    /// `shape` is the output's row-major shape (at most four dimensions; an
    /// empty shape is one element; `Triu` needs at least two); every view is
    /// indexed by it, with stride 0 on a broadcast dimension. An input may
    /// share the output's storage only with a view that addresses exactly the
    /// output's elements (in place); an operand this operation does not read
    /// is held to the same rule, so bind an unused slot to the output's own
    /// view or to other storage.
    pub fn new(
        op: PointwiseOp,
        shape: &[u64],
        views: [StridedView; OPERANDS],
        scalars: Scalars,
        storage: [Storage; OPERANDS],
    ) -> Result<Self, PointwisePlanError> {
        Self::new_with_storage_flags(op, shape, views, scalars, storage, 0)
    }

    /// `new`, with the bf16 storage bits a decoded block carried.
    pub fn new_with_storage_flags(
        op: PointwiseOp,
        shape: &[u64],
        views: [StridedView; OPERANDS],
        scalars: Scalars,
        storage: [Storage; OPERANDS],
        storage_flags: u32,
    ) -> Result<Self, PointwisePlanError> {
        if storage_flags & !STORAGE_FLAGS != 0 {
            return Err(PointwisePlanError::FlagsOutOfRange);
        }
        if shape.len() > MAX_NDIM {
            return Err(PointwisePlanError::NdimOutOfRange);
        }
        if op == PointwiseOp::Triu && shape.len() < 2 {
            return Err(PointwisePlanError::TriuNeedsTwoDimensions);
        }
        if shape.contains(&0) {
            return Err(PointwisePlanError::ZeroElements);
        }
        let mut canonical = [StridedView::new(0, [0; MAX_NDIM]); OPERANDS];
        for operand in Operand::ALL {
            let index = operand.index();
            canonical[index] = views[index].canonical(shape, storage[index].capacity_elements())?;
        }
        let output = canonical[Operand::Out.index()];
        // Two lanes writing one address is a race for every operation except
        // a fill, where both write the same constant (torch admits
        // `expand(...).fill_()` and `zero_()` on the same grounds).
        if op != PointwiseOp::Fill && !output.addresses_are_distinct(shape) {
            return Err(PointwisePlanError::OutputSelfOverlap);
        }
        for operand in Operand::INPUTS {
            let index = operand.index();
            if storage[index].identity() == storage[Operand::Out.index()].identity()
                && canonical[index] != output
            {
                return Err(PointwisePlanError::OutputOverlapsInput);
            }
        }
        // Distinct output addresses below ADDRESS_LIMIT bound the element count.
        let elements = shape
            .iter()
            .try_fold(1u64, |count, &extent| count.checked_mul(extent))
            .ok_or(PointwisePlanError::AddressOverflow)?;
        let n = u32::try_from(elements).map_err(|_| PointwisePlanError::AddressOverflow)?;
        let mut padded = [1u32; MAX_NDIM];
        for (word, &extent) in padded.iter_mut().zip(shape) {
            *word = u32::try_from(extent).map_err(|_| PointwisePlanError::AddressOverflow)?;
        }
        Ok(Self {
            op,
            ndim: shape.len() as u32,
            n,
            shape: padded,
            views: canonical,
            storage,
            scalars,
            storage_flags,
            dispatch_groups: [n.div_ceil(ELEMENTS_PER_GROUP), 1, 1],
        })
    }

    /// Re-validate a caller's push block against the storage the native
    /// integration resolved. The block's `n` must be its shape's element
    /// count and its flags must be ones this version implements; shape and
    /// stride words beyond `ndim` are padding and are not read.
    pub fn from_push_constants(
        bytes: &[u8; PUSH_BYTES],
        storage: [Storage; OPERANDS],
    ) -> Result<Self, PointwisePlanError> {
        let mut words = [0u32; PUSH_WORDS];
        for (word, chunk) in words.iter_mut().zip(bytes.chunks_exact(4)) {
            *word = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        let op = PointwiseOp::try_from(words[0])?;
        let ndim = usize::try_from(words[1]).map_err(|_| PointwisePlanError::NdimOutOfRange)?;
        if ndim > MAX_NDIM {
            return Err(PointwisePlanError::NdimOutOfRange);
        }
        if words[3] & !KNOWN_FLAGS != 0 {
            return Err(PointwisePlanError::FlagsOutOfRange);
        }
        let mut shape = [0u64; MAX_NDIM];
        for (extent, &word) in shape.iter_mut().zip(&words[4..4 + MAX_NDIM]) {
            *extent = u64::from(word);
        }
        let mut views = [StridedView::new(0, [0; MAX_NDIM]); OPERANDS];
        for operand in Operand::ALL {
            let base = 8 + operand.index() * (MAX_NDIM + 1);
            let mut strides = [0u64; MAX_NDIM];
            for (stride, &word) in strides.iter_mut().zip(&words[base..base + MAX_NDIM]) {
                *stride = u64::from(word);
            }
            views[operand.index()] = StridedView::new(u64::from(words[base + MAX_NDIM]), strides);
        }
        // both words are carried whatever the flag: `b` is a bound, a step or
        // a diagonal for the operations that do not read `y`
        let scalars = Scalars::from_words(words[28], words[29], words[3] & FLAG_Y_SCALAR != 0);
        let plan = Self::new_with_storage_flags(
            op,
            &shape[..ndim],
            views,
            scalars,
            storage,
            words[3] & STORAGE_FLAGS,
        )?;
        if plan.n != words[2] {
            return Err(PointwisePlanError::ElementCountMismatch);
        }
        Ok(plan)
    }

    pub fn op(&self) -> PointwiseOp {
        self.op
    }
    pub fn ndim(&self) -> u32 {
        self.ndim
    }
    pub fn n(&self) -> u32 {
        self.n
    }
    pub fn flags(&self) -> u32 {
        self.scalars.flags() | self.storage_flags
    }

    /// The module index this plan's storage selects; see `storage_module`.
    pub fn storage_module(&self) -> usize {
        storage_module(self.storage_flags)
    }
    /// The shape padded to `MAX_NDIM` with extent 1.
    pub fn shape(&self) -> [u32; MAX_NDIM] {
        self.shape
    }
    /// The operand's view as planned: strides of extent-1 and padded
    /// dimensions are zero.
    pub fn view(&self, operand: Operand) -> StridedView {
        self.views[operand.index()]
    }
    pub fn storage(&self, operand: Operand) -> Storage {
        self.storage[operand.index()]
    }
    pub fn scalars(&self) -> Scalars {
        self.scalars
    }
    pub fn dispatch_groups(&self) -> [u32; 3] {
        self.dispatch_groups
    }

    /// The shader's push block: `op, ndim, n, flags, shape[4]`, then
    /// `stride[4], offset` for `x`, `y`, `z`, `out`, then the `a` and `b`
    /// scalar words, as thirty little-endian u32 words.
    pub fn push_constants(&self) -> [u8; PUSH_BYTES] {
        let mut words = [0u32; PUSH_WORDS];
        words[..4].copy_from_slice(&[self.op as u32, self.ndim, self.n, self.flags()]);
        words[4..4 + MAX_NDIM].copy_from_slice(&self.shape);
        for operand in Operand::ALL {
            let base = 8 + operand.index() * (MAX_NDIM + 1);
            words[base..base + MAX_NDIM + 1]
                .copy_from_slice(&self.views[operand.index()].push_words());
        }
        words[28] = self.scalars.a_bits;
        words[29] = self.scalars.b_bits;

        let mut bytes = [0u8; PUSH_BYTES];
        for (index, word) in words.iter().enumerate() {
            bytes[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contiguous(shape: &[u64]) -> StridedView {
        StridedView::contiguous(shape).unwrap()
    }

    fn elements(shape: &[u64]) -> u64 {
        shape.iter().product()
    }

    /// Four distinct storages, each exactly the compact size of `shape`.
    fn distinct_storage(shape: &[u64]) -> [Storage; OPERANDS] {
        [
            Storage::new(1, elements(shape)),
            Storage::new(2, elements(shape)),
            Storage::new(3, elements(shape)),
            Storage::new(4, elements(shape)),
        ]
    }

    fn words_of(plan: &PointwisePlan) -> Vec<u32> {
        plan.push_constants()
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
            .collect()
    }

    fn compact_plan(op: PointwiseOp, shape: &[u64]) -> Result<PointwisePlan, PointwisePlanError> {
        PointwisePlan::new(
            op,
            shape,
            [contiguous(shape); OPERANDS],
            Scalars::new(1.0),
            distinct_storage(shape),
        )
    }

    #[test]
    fn operation_tags_and_named_errors_are_closed() {
        use PointwiseOp::*;
        let table = [
            (0, AddScaled),
            (1, SubScaled),
            (2, Mul),
            (3, Div),
            (4, Fill),
            (5, Copy),
            (6, Reciprocal),
            (7, Abs),
            (8, Sqrt),
            (9, Rsqrt),
            (10, Neg),
            (11, Pow),
            (12, PowScalarBase),
            (13, Clamp),
            (14, Silu),
            (15, Sigmoid),
            (16, SigmoidBackward),
            (17, Addcmul),
            (18, Addcdiv),
            (19, Lerp),
            (20, Cos),
            (21, Sin),
            (22, Iota),
            (23, Triu),
        ];
        for (word, op) in table {
            assert_eq!(PointwiseOp::try_from(word), Ok(op));
            assert_eq!(op as u32, word);
        }
        assert_eq!(table.len() as u32, LAST_OP + 1);
        assert_eq!((SCHEMA, LAST_OP), (3, 23));
        assert_eq!(
            PointwiseOp::try_from(LAST_OP + 1),
            Err(PointwisePlanError::OperationOutOfRange)
        );
        assert_eq!(
            PointwiseOp::try_from(u32::MAX),
            Err(PointwisePlanError::OperationOutOfRange)
        );
        // the operand slots each operation reads, as the shader header lists them
        for op in [Fill, Iota] {
            assert!(!op.reads_x() && !op.reads_y() && !op.reads_z(), "{op:?}");
        }
        for op in [
            Copy,
            Reciprocal,
            Abs,
            Sqrt,
            Rsqrt,
            Neg,
            PowScalarBase,
            Clamp,
            Silu,
            Sigmoid,
            Cos,
            Sin,
            Triu,
        ] {
            assert!(op.reads_x() && !op.reads_y() && !op.reads_z(), "{op:?}");
        }
        for op in [AddScaled, SubScaled, Mul, Div, Pow, SigmoidBackward, Lerp] {
            assert!(op.reads_x() && op.reads_y() && !op.reads_z(), "{op:?}");
        }
        for op in [Addcmul, Addcdiv] {
            assert!(op.reads_x() && op.reads_y() && op.reads_z(), "{op:?}");
        }
        for (error, name) in [
            (
                PointwisePlanError::OperationOutOfRange,
                "pointwise-operation-out-of-range",
            ),
            (
                PointwisePlanError::NdimOutOfRange,
                "pointwise-ndim-out-of-range",
            ),
            (
                PointwisePlanError::TriuNeedsTwoDimensions,
                "pointwise-triu-needs-two-dimensions",
            ),
            (PointwisePlanError::ZeroElements, "pointwise-zero-elements"),
            (
                PointwisePlanError::AddressOverflow,
                "pointwise-address-overflow",
            ),
            (
                PointwisePlanError::OutputOverlapsInput,
                "pointwise-output-overlaps-input",
            ),
            (
                PointwisePlanError::OutputSelfOverlap,
                "pointwise-output-self-overlap",
            ),
            (
                PointwisePlanError::FlagsOutOfRange,
                "pointwise-flags-out-of-range",
            ),
            (
                PointwisePlanError::ElementCountMismatch,
                "pointwise-element-count-mismatch",
            ),
        ] {
            assert_eq!(error.to_string(), name);
        }
        assert_eq!(
            (
                ABI,
                PUSH_BYTES,
                PUSH_WORDS,
                ELEMENTS_PER_GROUP,
                ADDRESS_LIMIT
            ),
            ("pointwise-f32/v1", 120, 30, 1024, 2_147_483_648)
        );
    }

    #[test]
    fn ndim_above_four_and_zero_elements_refuse_by_name() {
        // Five dimensions refuse before any view is inspected, whatever the
        // views and storages declare.
        let five = [1u64, 1, 1, 1, 1];
        assert_eq!(
            PointwisePlan::new(
                PointwiseOp::Mul,
                &five,
                [StridedView::new(0, [0; MAX_NDIM]); OPERANDS],
                Scalars::new(1.0),
                distinct_storage(&five),
            ),
            Err(PointwisePlanError::NdimOutOfRange)
        );
        assert_eq!(
            StridedView::contiguous(&five),
            Err(PointwisePlanError::NdimOutOfRange)
        );
        assert_eq!(
            compact_plan(PointwiseOp::Mul, &[3, 0, 5]),
            Err(PointwisePlanError::ZeroElements)
        );
        assert_eq!(
            compact_plan(PointwiseOp::Mul, &[0]),
            Err(PointwisePlanError::ZeroElements)
        );
        // Four dimensions and a zero-dimensional shape are inside the contract.
        let four = compact_plan(PointwiseOp::Mul, &[2, 3, 4, 5]).unwrap();
        assert_eq!(
            (four.ndim(), four.n(), four.shape()),
            (4, 120, [2, 3, 4, 5])
        );
        let scalar = compact_plan(PointwiseOp::Copy, &[]).unwrap();
        assert_eq!((scalar.ndim(), scalar.n(), scalar.shape()), (0, 1, [1; 4]));
        assert_eq!(scalar.dispatch_groups(), [1, 1, 1]);
    }

    #[test]
    fn address_arithmetic_refuses_the_limit_the_capacity_and_wraparound() {
        let shape = [4u64];
        let storage = distinct_storage(&shape);
        // The compact view needs exactly four elements.
        let short = [storage[0], storage[1], storage[2], Storage::new(4, 3)];
        assert_eq!(
            PointwisePlan::new(
                PointwiseOp::Copy,
                &shape,
                [contiguous(&shape); OPERANDS],
                Scalars::new(0.0),
                short,
            ),
            Err(PointwisePlanError::AddressOverflow)
        );
        // An unread operand's declaration is held to the same capacity rule.
        let short_z = [storage[0], storage[1], Storage::new(3, 3), storage[3]];
        assert_eq!(
            PointwisePlan::new(
                PointwiseOp::Fill,
                &shape,
                [contiguous(&shape); OPERANDS],
                Scalars::new(0.0),
                short_z,
            ),
            Err(PointwisePlanError::AddressOverflow)
        );
        // Maximum address 2^31 - 1 is the last admitted; 2^31 is refused even
        // when the declared capacity would hold it.
        let at_limit = |offset: u64| {
            PointwisePlan::new(
                PointwiseOp::Copy,
                &[1],
                [StridedView::new(offset, [0; MAX_NDIM]); OPERANDS],
                Scalars::new(0.0),
                [Storage::new(9, ADDRESS_LIMIT + 1); OPERANDS],
            )
        };
        let admitted = at_limit(ADDRESS_LIMIT - 1).unwrap();
        assert_eq!(admitted.view(Operand::X).offset(), ADDRESS_LIMIT - 1);
        assert_eq!(
            words_of(&admitted)[12],
            u32::try_from(ADDRESS_LIMIT - 1).unwrap()
        );
        assert_eq!(
            at_limit(ADDRESS_LIMIT),
            Err(PointwisePlanError::AddressOverflow)
        );
        // A stride reaching the limit through a wide dimension is refused the
        // same way, and checked arithmetic refuses a wraparound by name.
        let wide = [2u64, 3];
        let unbounded = [
            Storage::new(1, u64::MAX),
            Storage::new(2, u64::MAX),
            Storage::new(3, u64::MAX),
            Storage::new(4, u64::MAX),
        ];
        let with_x = |x: StridedView| {
            PointwisePlan::new(
                PointwiseOp::Mul,
                &wide,
                [x, contiguous(&wide), contiguous(&wide), contiguous(&wide)],
                Scalars::new(1.0),
                unbounded,
            )
        };
        // One step of the outer dimension reaches 2^31 + 2.
        assert_eq!(
            with_x(StridedView::new(0, [ADDRESS_LIMIT, 1, 0, 0])),
            Err(PointwisePlanError::AddressOverflow)
        );
        // Half that stride stays below the limit and is admitted.
        assert!(with_x(StridedView::new(0, [ADDRESS_LIMIT / 2, 1, 0, 0])).is_ok());
        assert_eq!(
            with_x(StridedView::new(u64::MAX - 1, [u64::MAX, 1, 0, 0])),
            Err(PointwisePlanError::AddressOverflow)
        );
        assert_eq!(
            StridedView::contiguous(&[u64::MAX, 2]),
            Err(PointwisePlanError::AddressOverflow)
        );
    }

    #[test]
    fn broadcast_strides_are_zero_and_address_only_the_broadcast_source() {
        // w[512] * x[1, 128, 512]: the weight row is read for every token row.
        let shape = [1u64, 128, 512];
        let weight = StridedView::new(0, [0, 0, 1, 0]);
        let storage = [
            Storage::new(1, elements(&shape)),
            Storage::new(2, 512),
            Storage::new(1, elements(&shape)),
            Storage::new(4, elements(&shape)),
        ];
        let plan = PointwisePlan::new(
            PointwiseOp::Mul,
            &shape,
            [
                contiguous(&shape),
                weight,
                contiguous(&shape),
                contiguous(&shape),
            ],
            Scalars::new(1.0),
            storage,
        )
        .unwrap();
        assert_eq!(plan.n(), 65_536);
        assert_eq!(plan.dispatch_groups(), [64, 1, 1]);
        // The leading extent-1 dimension's stride is irrelevant and is zeroed.
        assert_eq!(plan.view(Operand::X).strides(), [0, 512, 1, 0]);
        assert_eq!(plan.view(Operand::Y).strides(), [0, 0, 1, 0]);
        // One weight element short refuses: the broadcast still reads 512.
        let short = [storage[0], Storage::new(2, 511), storage[2], storage[3]];
        assert_eq!(
            PointwisePlan::new(
                PointwiseOp::Mul,
                &shape,
                [
                    contiguous(&shape),
                    weight,
                    contiguous(&shape),
                    contiguous(&shape)
                ],
                Scalars::new(1.0),
                short,
            ),
            Err(PointwisePlanError::AddressOverflow)
        );
        // x / s[..., None] with s of shape [1, 128, 1]: stride 0 along the
        // last dimension, and only 128 elements are ever addressed.
        let scale = StridedView::new(0, [128, 1, 0, 0]);
        let scaled = PointwisePlan::new(
            PointwiseOp::Div,
            &shape,
            [
                contiguous(&shape),
                scale,
                contiguous(&shape),
                contiguous(&shape),
            ],
            Scalars::new(1.0),
            [storage[0], Storage::new(2, 128), storage[2], storage[3]],
        )
        .unwrap();
        assert_eq!(scaled.view(Operand::Y).strides(), [0, 1, 0, 0]);
        assert_eq!(
            PointwisePlan::new(
                PointwiseOp::Div,
                &shape,
                [
                    contiguous(&shape),
                    scale,
                    contiguous(&shape),
                    contiguous(&shape)
                ],
                Scalars::new(1.0),
                [storage[0], Storage::new(2, 127), storage[2], storage[3]],
            ),
            Err(PointwisePlanError::AddressOverflow)
        );
    }

    #[test]
    fn output_self_overlap_is_refused_and_every_torch_view_layout_is_admitted() {
        let shape = [2u64, 2];
        let with_output = |output: StridedView, capacity: u64| {
            PointwisePlan::new(
                PointwiseOp::Copy,
                &shape,
                [
                    contiguous(&shape),
                    contiguous(&shape),
                    contiguous(&shape),
                    output,
                ],
                Scalars::new(0.0),
                [
                    Storage::new(1, 4),
                    Storage::new(2, 4),
                    Storage::new(3, 4),
                    Storage::new(4, capacity),
                ],
            )
        };
        // A broadcast output would write one address from two lanes.
        assert_eq!(
            with_output(StridedView::new(0, [0, 1, 0, 0]), 4),
            Err(PointwisePlanError::OutputSelfOverlap)
        );
        // Equal strides on two dimensions of extent above one coincide.
        assert_eq!(
            with_output(StridedView::new(0, [1, 1, 0, 0]), 4),
            Err(PointwisePlanError::OutputSelfOverlap)
        );
        // Transposed, stepped, padded and offset outputs are distinct.
        assert!(with_output(StridedView::new(0, [1, 2, 0, 0]), 4).is_ok());
        assert!(with_output(StridedView::new(0, [4, 2, 0, 0]), 7).is_ok());
        assert!(with_output(StridedView::new(0, [3, 1, 0, 0]), 5).is_ok());
        assert!(with_output(StridedView::new(5, [2, 1, 0, 0]), 9).is_ok());
        // A broadcast input is fine: only the output must not overlap itself.
        let broadcast_input = PointwisePlan::new(
            PointwiseOp::Mul,
            &shape,
            [
                StridedView::new(0, [0, 1, 0, 0]),
                contiguous(&shape),
                contiguous(&shape),
                contiguous(&shape),
            ],
            Scalars::new(1.0),
            [
                Storage::new(1, 2),
                Storage::new(2, 4),
                Storage::new(3, 4),
                Storage::new(4, 4),
            ],
        );
        assert!(broadcast_input.is_ok());
    }

    #[test]
    fn in_place_is_accepted_only_through_a_view_identical_to_the_output() {
        let shape = [3u64, 4];
        let x = contiguous(&shape);
        let transposed = StridedView::new(0, [1, 3, 0, 0]);
        let shared = Storage::new(7, 12);
        let other = Storage::new(8, 12);
        // x.mul_(y): out is x's storage through x's view.
        let in_place = PointwisePlan::new(
            PointwiseOp::Mul,
            &shape,
            [x, contiguous(&shape), x, x],
            Scalars::new(1.0),
            [shared, other, shared, shared],
        )
        .unwrap();
        assert_eq!(in_place.view(Operand::Out), in_place.view(Operand::X));
        assert_eq!(in_place.storage(Operand::Out), shared);
        // The same storage through another view is a race, not an in-place op.
        assert_eq!(
            PointwisePlan::new(
                PointwiseOp::Mul,
                &shape,
                [x, contiguous(&shape), x, transposed],
                Scalars::new(1.0),
                [shared, other, shared, shared],
            ),
            Err(PointwisePlanError::OutputOverlapsInput)
        );
        // x.mul_(x.t()): y reads x's storage through a view out does not match.
        assert_eq!(
            PointwisePlan::new(
                PointwiseOp::Mul,
                &shape,
                [x, transposed, x, x],
                Scalars::new(1.0),
                [shared, shared, shared, shared],
            ),
            Err(PointwisePlanError::OutputOverlapsInput)
        );
        // An unread slot bound to the output's storage must also match it.
        assert_eq!(
            PointwisePlan::new(
                PointwiseOp::Fill,
                &shape,
                [x, x, transposed, x],
                Scalars::new(0.0),
                [shared, shared, shared, shared],
            ),
            Err(PointwisePlanError::OutputOverlapsInput)
        );
        assert!(PointwisePlan::new(
            PointwiseOp::Fill,
            &shape,
            [x, x, x, x],
            Scalars::new(0.0),
            [shared, shared, shared, shared],
        )
        .is_ok());
        // x.mul_(2.0): the y slot is bound to x with x's view and never read.
        let scaled = PointwisePlan::new(
            PointwiseOp::Mul,
            &shape,
            [x, x, x, x],
            Scalars::with_scalar_y(1.0, 2.0),
            [shared, shared, shared, shared],
        )
        .unwrap();
        assert_eq!(scaled.flags(), FLAG_Y_SCALAR);
        assert_eq!(scaled.scalars().y(), Some(2.0));
        // Strides of extent-1 dimensions do not distinguish views: torch
        // reports arbitrary ones and both address the same elements.
        let column_shape = [3u64, 1];
        let a = StridedView::new(2, [1, 1, 0, 0]);
        let b = StridedView::new(2, [1, 99, 0, 0]);
        let same = PointwisePlan::new(
            PointwiseOp::Copy,
            &column_shape,
            [a, a, a, b],
            Scalars::new(0.0),
            [shared, shared, shared, shared],
        )
        .unwrap();
        assert_eq!(same.view(Operand::Out), same.view(Operand::X));
        assert_eq!(same.view(Operand::Out).strides(), [1, 0, 0, 0]);
        // Inputs may share storage with each other freely: x * x.t().
        assert!(PointwisePlan::new(
            PointwiseOp::Mul,
            &shape,
            [x, transposed, x, contiguous(&shape)],
            Scalars::new(1.0),
            [shared, shared, shared, other],
        )
        .is_ok());
    }

    #[test]
    fn push_layout_is_exact_and_little_endian() {
        let shape = [2u64, 3, 4];
        let plan = PointwisePlan::new(
            PointwiseOp::SubScaled,
            &shape,
            [
                StridedView::new(5, [12, 4, 1, 77]),
                StridedView::new(6, [0, 4, 1, 0]),
                StridedView::new(7, [24, 8, 2, 0]),
                StridedView::new(8, [13, 4, 1, 0]),
            ],
            Scalars::with_scalar_y(0.5, -2.0),
            [
                Storage::new(1, 5 + 12 + 8 + 3 + 1),
                Storage::new(2, 6 + 8 + 3 + 1),
                Storage::new(3, 7 + 24 + 16 + 6 + 1),
                Storage::new(4, 8 + 13 + 8 + 3 + 1),
            ],
        )
        .unwrap();
        let bytes = plan.push_constants();
        assert_eq!(bytes.len(), PUSH_BYTES);
        assert_eq!(bytes[..4], 1u32.to_le_bytes());
        assert_eq!(bytes[112..116], 0.5f32.to_bits().to_le_bytes());
        assert_eq!(bytes[116..120], (-2.0f32).to_bits().to_le_bytes());
        let words = words_of(&plan);
        assert_eq!(words.len(), PUSH_WORDS);
        // op, ndim, n, flags; then the shape padded with extent 1
        assert_eq!(&words[..4], &[1, 3, 24, FLAG_Y_SCALAR]);
        assert_eq!(&words[4..8], &[2, 3, 4, 1]);
        // stride[4] then offset for x (its padded stride zeroed), y, z, out
        assert_eq!(&words[8..13], &[12, 4, 1, 0, 5]);
        assert_eq!(&words[13..18], &[0, 4, 1, 0, 6]);
        assert_eq!(&words[18..23], &[24, 8, 2, 0, 7]);
        assert_eq!(&words[23..28], &[13, 4, 1, 0, 8]);
        // a = 0.5 and the scalar y = -2.0 as float bits
        assert_eq!(&words[28..], &[0x3f00_0000, 0xc000_0000]);
        assert_eq!(plan.dispatch_groups(), [1, 1, 1]);
        let without_scalar = PointwisePlan::new(
            PointwiseOp::Fill,
            &[1],
            [StridedView::new(0, [0; MAX_NDIM]); OPERANDS],
            Scalars::new(f32::NEG_INFINITY),
            [Storage::new(1, 1); OPERANDS],
        )
        .unwrap();
        let words = words_of(&without_scalar);
        assert_eq!(&words[..8], &[4, 1, 1, 0, 1, 1, 1, 1]);
        assert_eq!(&words[8..28], &[0; 20]);
        assert_eq!(&words[28..], &[f32::NEG_INFINITY.to_bits(), 0]);
    }

    #[test]
    fn the_head_logit_tensor_fits_one_dispatch_axis() {
        // 128 tokens by the 32,768-entry vocabulary: 4,194,304 elements, four
        // per invocation, 1,024 per workgroup.
        let shape = [128u64, 32_768];
        let plan = compact_plan(PointwiseOp::AddScaled, &shape).unwrap();
        assert_eq!(plan.n(), 4_194_304);
        assert_eq!(plan.dispatch_groups(), [4_096, 1, 1]);
        assert!(plan.dispatch_groups()[0] < 65_535);
        assert_eq!(plan.view(Operand::Out).strides(), [32_768, 1, 0, 0]);
        for (n, groups) in [(1u64, 1u32), (1_024, 1), (1_025, 2), (4_194_305, 4_097)] {
            assert_eq!(
                compact_plan(PointwiseOp::Copy, &[n])
                    .unwrap()
                    .dispatch_groups(),
                [groups, 1, 1]
            );
        }
    }

    #[test]
    fn a_push_block_re_validates_to_the_same_plan_and_refuses_by_name() {
        let shape = [2u64, 5, 3];
        let storage = distinct_storage(&shape);
        let plan = PointwisePlan::new(
            PointwiseOp::AddScaled,
            &shape,
            [
                contiguous(&shape),
                StridedView::new(0, [0, 3, 1, 0]),
                contiguous(&shape),
                contiguous(&shape),
            ],
            Scalars::new(0.25),
            [storage[0], Storage::new(2, 15), storage[2], storage[3]],
        )
        .unwrap();
        let bytes = plan.push_constants();
        let decoded = PointwisePlan::from_push_constants(
            &bytes,
            [storage[0], Storage::new(2, 15), storage[2], storage[3]],
        )
        .unwrap();
        assert_eq!(decoded, plan);
        assert_eq!(decoded.push_constants(), bytes);

        let scalar_y = PointwisePlan::new(
            PointwiseOp::Div,
            &shape,
            [contiguous(&shape); OPERANDS],
            Scalars::with_scalar_y(1.0, 4.0),
            storage,
        )
        .unwrap();
        let decoded =
            PointwisePlan::from_push_constants(&scalar_y.push_constants(), storage).unwrap();
        assert_eq!(decoded, scalar_y);
        assert_eq!(decoded.scalars(), Scalars::with_scalar_y(1.0, 4.0));

        let word = |index: usize, value: u32, storage: [Storage; OPERANDS]| {
            let mut altered = bytes;
            altered[index * 4..index * 4 + 4].copy_from_slice(&value.to_le_bytes());
            PointwisePlan::from_push_constants(&altered, storage)
        };
        let expected_storage = [storage[0], Storage::new(2, 15), storage[2], storage[3]];
        assert_eq!(
            word(0, LAST_OP + 1, expected_storage),
            Err(PointwisePlanError::OperationOutOfRange)
        );
        assert_eq!(
            word(1, 5, expected_storage),
            Err(PointwisePlanError::NdimOutOfRange)
        );
        assert_eq!(
            word(2, 31, expected_storage),
            Err(PointwisePlanError::ElementCountMismatch)
        );
        assert_eq!(
            word(3, FLAG_Z_SCALAR, expected_storage),
            Err(PointwisePlanError::FlagsOutOfRange)
        );
        // the lowest bit no version implements; 4 through 32 were that bit
        // until schema 3 took them for the four storage switches
        assert_eq!(
            word(3, 64, expected_storage),
            Err(PointwisePlanError::FlagsOutOfRange)
        );
        // and the four it DID take are accepted and survive a round trip,
        // which is what makes the assertion above a statement about unknown
        // bits rather than about every bit above zero. The block's operation
        // is `AddScaled`, which reads x and y, so those two must agree; z
        // and out are free, and z is not read at all.
        for flags in [
            0,
            FLAG_BF16[3],
            FLAG_BF16[2],
            FLAG_BF16[0] | FLAG_BF16[1],
            STORAGE_FLAGS,
        ] {
            let plan = word(3, flags, expected_storage)
                .unwrap_or_else(|e| panic!("flags {flags} refused: {e}"));
            assert_eq!(plan.flags() & STORAGE_FLAGS, flags);
            assert_eq!(plan.storage_module(), storage_module(flags));
            let bytes = plan.push_constants();
            assert_eq!(
                u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) & !FLAG_Y_SCALAR,
                flags,
                "a decoded plan must re-encode its storage bits"
            );
        }
        // and every one of the sixteen combinations is a module, including
        // one operand of each dtype, which torch really does hand the backend
        for flags in 0..16u32 {
            let bits = (0..OPERANDS)
                .map(|i| if flags >> i & 1 == 1 { FLAG_BF16[i] } else { 0 })
                .fold(0, |a, b| a | b);
            let plan = word(3, bits, expected_storage)
                .unwrap_or_else(|e| panic!("flags {bits} refused: {e}"));
            assert_eq!(plan.storage_module(), flags as usize);
        }
        assert_eq!(
            word(5, 0, expected_storage),
            Err(PointwisePlanError::ZeroElements)
        );
        // The real capacities decide: the same block against a smaller buffer.
        assert_eq!(
            word(
                27,
                0,
                [
                    storage[0],
                    Storage::new(2, 15),
                    storage[2],
                    Storage::new(4, 29)
                ]
            ),
            Err(PointwisePlanError::AddressOverflow)
        );
        // A block whose output is an input's storage through another view.
        assert_eq!(
            word(
                27,
                1,
                [
                    storage[0],
                    Storage::new(2, 15),
                    storage[2],
                    Storage::new(1, 31)
                ]
            ),
            Err(PointwisePlanError::OutputOverlapsInput)
        );
    }

    #[test]
    fn schema_two_scalar_words_are_encoded_and_carried_without_the_y_flag() {
        // clamp: an absent bound is the infinity that compares as no bound
        let shape = [3u64, 4];
        let storage = distinct_storage(&shape);
        let low_only = PointwisePlan::new(
            PointwiseOp::Clamp,
            &shape,
            [contiguous(&shape); OPERANDS],
            Scalars::clamp(Some(-0.5), None),
            storage,
        )
        .unwrap();
        let words = words_of(&low_only);
        assert_eq!(&words[..4], &[13, 2, 12, 0], "no y flag: b is a bound");
        assert_eq!(
            &words[28..],
            &[(-0.5f32).to_bits(), f32::INFINITY.to_bits()]
        );
        assert_eq!(low_only.scalars().y(), None);
        assert_eq!(
            (low_only.scalars().a(), low_only.scalars().b()),
            (-0.5, f32::INFINITY)
        );
        let high_only = Scalars::clamp(None, Some(0.25));
        assert_eq!((high_only.a(), high_only.b()), (f32::NEG_INFINITY, 0.25));
        // min above max is admitted, as torch admits it (every element becomes max)
        assert!(PointwisePlan::new(
            PointwiseOp::Clamp,
            &shape,
            [contiguous(&shape); OPERANDS],
            Scalars::clamp(Some(0.3), Some(0.1)),
            storage,
        )
        .is_ok());
        // the b word survives a decode without the flag (schema 1 dropped it)
        let bytes = low_only.push_constants();
        let decoded = PointwisePlan::from_push_constants(&bytes, storage).unwrap();
        assert_eq!(decoded, low_only);
        assert_eq!(decoded.push_constants(), bytes);
        assert_eq!(decoded.scalars().b(), f32::INFINITY);

        // pow with a scalar exponent: y is the scalar b, flagged, and z is not read
        let squared = PointwisePlan::new(
            PointwiseOp::Pow,
            &shape,
            [contiguous(&shape); OPERANDS],
            Scalars::with_scalar_y(0.0, 2.0),
            storage,
        )
        .unwrap();
        let words = words_of(&squared);
        assert_eq!(&words[..4], &[11, 2, 12, FLAG_Y_SCALAR]);
        assert_eq!(&words[28..], &[0, 2.0f32.to_bits()]);
        assert_eq!(squared.scalars().y(), Some(2.0));
        assert!(!PointwiseOp::Pow.reads_z());
        // pow with a scalar base: the base is a, y is not read
        let base = PointwisePlan::new(
            PointwiseOp::PowScalarBase,
            &shape,
            [contiguous(&shape); OPERANDS],
            Scalars::new(2.0),
            storage,
        )
        .unwrap();
        assert_eq!(&words_of(&base)[28..], &[2.0f32.to_bits(), 0]);

        // iota: start and step, no input read, over the linear index of any shape
        let ramp = PointwisePlan::new(
            PointwiseOp::Iota,
            &[1000],
            [contiguous(&[1000]); OPERANDS],
            Scalars::iota(0.5, 0.25),
            distinct_storage(&[1000]),
        )
        .unwrap();
        let words = words_of(&ramp);
        assert_eq!(&words[..4], &[22, 1, 1000, 0]);
        assert_eq!(&words[28..], &[0.5f32.to_bits(), 0.25f32.to_bits()]);
        assert!(!PointwiseOp::Iota.reads_x());

        // triu: the diagonal is an i32 in the b word, and two dimensions are needed
        let upper = PointwisePlan::new(
            PointwiseOp::Triu,
            &shape,
            [contiguous(&shape); OPERANDS],
            Scalars::triu(-1),
            storage,
        )
        .unwrap();
        let words = words_of(&upper);
        assert_eq!(&words[28..], &[0, 0xffff_ffff]);
        assert_eq!(upper.scalars().diagonal(), -1);
        assert_eq!(Scalars::triu(2).diagonal(), 2);
        for shape in [&[][..], &[7][..]] {
            assert_eq!(
                PointwisePlan::new(
                    PointwiseOp::Triu,
                    shape,
                    [contiguous(shape); OPERANDS],
                    Scalars::triu(0),
                    distinct_storage(shape),
                ),
                Err(PointwisePlanError::TriuNeedsTwoDimensions),
                "{shape:?}"
            );
        }
        // a batched triu is admitted: the last two dimensions are the matrix
        assert!(compact_plan(PointwiseOp::Triu, &[2, 3, 4]).is_ok());
        assert!(compact_plan(PointwiseOp::Triu, &[2, 2, 3, 4]).is_ok());
        // the refusal is named before the views are inspected, like ndim
        let mut block = ramp.push_constants();
        block[..4].copy_from_slice(&(PointwiseOp::Triu as u32).to_le_bytes());
        assert_eq!(
            PointwisePlan::from_push_constants(&block, distinct_storage(&[1000])),
            Err(PointwisePlanError::TriuNeedsTwoDimensions)
        );

        // the ternary operations read z; the plan validates z's view for every
        // operation, so admission does not depend on it
        for op in [PointwiseOp::Addcmul, PointwiseOp::Addcdiv] {
            let plan = PointwisePlan::new(
                op,
                &shape,
                [
                    contiguous(&shape),
                    StridedView::new(0, [0, 1, 0, 0]),
                    StridedView::new(2, [4, 1, 0, 0]),
                    contiguous(&shape),
                ],
                Scalars::new(0.7),
                [
                    storage[0],
                    Storage::new(2, 4),
                    Storage::new(3, 14),
                    storage[3],
                ],
            )
            .unwrap();
            assert_eq!(plan.view(Operand::Z).offset(), 2);
            assert_eq!(words_of(&plan)[28], 0.7f32.to_bits());
        }
        // pair and the schema-1 constructors agree on the words they share
        assert_eq!(Scalars::pair(1.0, 0.0), Scalars::new(1.0));
        assert_ne!(Scalars::pair(1.0, 4.0), Scalars::with_scalar_y(1.0, 4.0));
    }
}
