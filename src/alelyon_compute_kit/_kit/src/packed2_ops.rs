//! Geometry, capacity and dispatch planning for `matmul-packed2/v1`.
//!
//! The shader evaluates `C[m, n] = sum_l A[m, l] * dequantise(B[l, n])`, where
//! B arrives as 2-bit codewords -- 16 per `uint32`, three levels `{-1, 0, +1}`
//! with code 3 reserved -- and one fp32 scale per `block` consecutive weights
//! in B's flattened `(k, n)` order. It is the first kernel in this crate whose
//! weight operand is not a dtype the device knows.
//!
//! WHY THIS MODULE EXISTS AT ALL, and it is not symmetry with `matmul_ops`.
//! The shader's own bounds check `return`s 0.0 for an address it cannot reach.
//! For an fp32 kernel that is belt and braces, because the caller passes a
//! buffer whose length the runtime knows in the same units as the addressing.
//! Here it is not: the codes buffer is addressed in WORDS while the plan is
//! written in WEIGHTS, so a caller who sizes it as though it held one weight
//! per element is short by 16x and the shader answers 0.0 for
//! fifteen-sixteenths of the matrix. That is a wrong answer at full speed with
//! a success code on it. Refusing here is what makes it nameable.
//!
//! This module has no `Context` or `Buffer` dependency. Its capacities are
//! caller declarations, so a successful plan does not prove buffer ownership,
//! liveness, actual capacity, or output/input non-aliasing. The native
//! integration must establish those from real buffer objects immediately
//! before binding and dispatch.

use std::fmt;

pub const ABI: &str = "matmul-packed2/v1";

/// The control built from the SAME source with `-DPACKED_B=0`: `decode_b`
/// reads a plain fp32 weight from binding 2 instead of decoding a codeword,
/// and everything else -- tile, loop, bounds, push block, bindings, grid --
/// is identical. It exists so step 3's gate ("decode must cost less than the
/// memory it saves") can be measured against an arm that differs in exactly
/// one thing. It is NOT a shipped operator.
pub const ABI_CONTROL: &str = "matmul-unpacked-ref/v1";

/// Weights per `uint32` of the codes array. Mirrored in the shader as
/// `WEIGHTS_PER_WORD` and in `research/kv_precision/packed2.py`; a drift
/// between the three is a silent wrong answer, so it is asserted in
/// `tests/packed2_plan.rs` rather than merely written down three times.
pub const WEIGHTS_PER_WORD: u32 = 16;
/// Bits per weight in the codes array.
pub const BITS_PER_WEIGHT: u32 = 2;
/// The codeword that decodes to 0.0 rather than to a fourth level. The 2-bit
/// grid `quantize_rowwise` produces is TERNARY -- rho = 2^(b-1) - 1 = 1 -- so
/// one of the four codes is unused, and it is reserved rather than repurposed
/// so this kernel decodes exactly what the measured optimizer produced.
/// Reclaiming the 0.415 bits/weight it leaves is a format change and a new
/// measurement, not a free win.
pub const RESERVED_CODE: u32 = 3;

/// Ceiling on `m` and `n`. The same addressing law and the same number as
/// `matmul_ops::MAX_DIM`; stated separately because this kernel's B operand is
/// addressed in a different unit and a shared constant would hide that.
pub const MAX_DIM: u32 = 65_536;
/// Ceiling on the reduction length. Equal to `MAX_DIM`, and -- unlike
/// `matmul_ops::MAX_K`, which is bounded by an ACCURACY sweep against `fsum`
/// because that kernel compensates -- this one is bounded by addressing alone.
/// This kernel accumulates in plain fp32 with no compensation term, so no
/// accuracy sweep licenses it and none is claimed: the arithmetic bound here
/// is UNMEASURED and `k` is limited only by what can be addressed.
pub const MAX_K: u32 = 65_536;
/// Ceiling on the weights one operand view may address, logical and spanned.
/// Mirrored in the shader as `LIMIT`, a 32-bit uint, so it must fit u32.
pub const MAX_ADDRESSED_ELEMENTS: u64 = 268_435_456;
/// Ceiling on the scale count, so `n_scales` fits the push word.
pub const MAX_SCALES: u64 = MAX_ADDRESSED_ELEMENTS;
pub const TILE: u32 = 8;
pub const WORKGROUP_INVOCATIONS: u32 = TILE * TILE;

/// `matmul-packed2/v2`: one codeword decoded ONCE into registers and spent on
/// sixteen multiply-accumulates, and its control. v1 LOSES to its own fp32
/// control at every measured shape -- 0.74-0.89x from 4 MiB of weights to
/// 256 MiB -- because a 2-bit weight is not a 2-bit memory transaction: v1's
/// invocation loads a whole word to use two bits of it, so the sixteen
/// invocations covering one word load that word sixteen times and v1 issues
/// exactly as many loads as fp32 while additionally paying the decode.
pub const ABI_V2: &str = "matmul-packed2/v2";
pub const ABI_CONTROL_V2: &str = "matmul-unpacked-ref/v2";
/// Columns one v2 invocation owns: exactly one codeword.
pub const COLS_PER_INVOCATION: u32 = WEIGHTS_PER_WORD;

/// Which kernel a plan is for. The two take the SAME fifteen-word block and
/// differ only in the dispatch grid, which is why this lives on the plan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum Packed2Kind {
    /// v1: one invocation, one output element, one weight decoded per MAC.
    Scalar = 0,
    /// v2: one invocation, sixteen output columns, one word decoded per
    /// sixteen MACs. Requires a word-aligned, unit-column-stride B.
    Wide16 = 1,
}
/// Fifteen little-endian u32 words: `m, n, k`, A's three, B's three (its
/// offset and strides are in WEIGHTS), C's three, then `block`, `n_scales`
/// and `block_shift`.
pub const PUSH_BYTES: usize = 60;

/// The `block_shift` value that tells the shader to take the DIVIDE path.
/// `block` is a push constant, so `addr / p.block` is a runtime 32-bit integer
/// division performed once per multiply-accumulate; a power-of-two block is a
/// shift instead. This sentinel is not dead code and not a fallback nobody
/// reaches: it is how the two paths are A/B'd through one push word of the
/// same module, which is what established that the divide -- not the decode --
/// was what failed step 3's gate.
pub const BLOCK_SHIFT_DIVIDE: u32 = 32;
/// Buffers the kernel binds: A values, B codes, B scales, C values. The
/// control binds the same four and leaves B codes unread.
pub const OPERANDS: usize = 4;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Packed2PlanError {
    DimensionOutOfRange,
    DimensionMismatch,
    ZeroStride,
    ZeroBlock,
    AddressOverflow,
    AddressLimitExceeded,
    /// The fp32 side (A, C, or the scale table) is shorter than the plan reaches.
    CapacityTooSmall,
    /// The CODES buffer is shorter than the plan reaches. Separate from
    /// `CapacityTooSmall` on purpose: this is the failure a caller who sized
    /// the buffer in weights rather than words gets, and it must not be
    /// confused with an ordinary short buffer.
    CodesCapacityTooSmall,
    /// `n_scales` does not cover every weight the plan addresses. The shader
    /// would return 0.0 past the end of the table, which is a wrong answer
    /// and not a refusal.
    ScaleTableTooShort,
    OutputOverlap,
    /// v2 only: B's column stride is not 1, so a codeword's sixteen weights
    /// are not sixteen consecutive columns and the kernel would decode
    /// sixteen unrelated weights into one register tile.
    ColumnStrideNotOne,
    /// v2 only: B's offset is not a multiple of 16, so an invocation's
    /// sixteen columns straddle two codewords.
    OffsetNotWordAligned,
    /// v2 only: a later B row would start inside a codeword. A one-row
    /// operand never uses its row stride, so it needs no such restriction.
    RowStrideNotWordAligned,
    /// v2 only: `block` is not a multiple of 16, so one codeword can straddle
    /// two scales and the single per-word scale load would be wrong for part
    /// of the word.
    BlockNotWordAligned,
}

impl fmt::Display for Packed2PlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::DimensionOutOfRange => "packed2-dimension-out-of-range",
            Self::DimensionMismatch => "packed2-dimension-mismatch",
            Self::ZeroStride => "packed2-zero-stride",
            Self::ZeroBlock => "packed2-zero-block",
            Self::AddressOverflow => "packed2-address-overflow",
            Self::AddressLimitExceeded => "packed2-address-limit-exceeded",
            Self::CapacityTooSmall => "packed2-capacity-too-small",
            Self::CodesCapacityTooSmall => "packed2-codes-capacity-too-small",
            Self::ScaleTableTooShort => "packed2-scale-table-too-short",
            Self::OutputOverlap => "packed2-output-overlap",
            Self::ColumnStrideNotOne => "packed2-v2-column-stride-not-one",
            Self::OffsetNotWordAligned => "packed2-v2-offset-not-word-aligned",
            Self::RowStrideNotWordAligned => "packed2-v2-row-stride-not-word-aligned",
            Self::BlockNotWordAligned => "packed2-v2-block-not-word-aligned",
        })
    }
}

impl std::error::Error for Packed2PlanError {}

/// How many `uint32` words hold `weights` codes.
pub fn words_for(weights: u64) -> u64 {
    weights.div_ceil(u64::from(WEIGHTS_PER_WORD))
}

/// How many scales cover `weights` at this block size.
pub fn scales_for(weights: u64, block: u32) -> u64 {
    if block == 0 {
        return 0;
    }
    weights.div_ceil(u64::from(block))
}

/// Bytes the packed form of `weights` occupies: codes plus scales. The scales
/// are part of the answer. Omitting them reports 2.000 bits/weight for every
/// block size, which is the one arithmetic mistake this format invites.
pub fn packed_bytes(weights: u64, block: u32) -> u64 {
    words_for(weights) * 4 + scales_for(weights, block) * 4
}

/// Bits per weight of the packed form, scales included.
pub fn bits_per_weight(weights: u64, block: u32) -> f64 {
    if weights == 0 {
        return 0.0;
    }
    packed_bytes(weights, block) as f64 * 8.0 / weights as f64
}

/// An immutable logical `[rows, cols]` FP32 view into one storage.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FloatView {
    rows: u32,
    cols: u32,
    offset: u32,
    row_stride: u32,
    col_stride: u32,
    capacity_elements: u64,
    required_elements: u64,
}

impl FloatView {
    pub fn new(
        rows: u32,
        cols: u32,
        offset: u64,
        row_stride: u64,
        col_stride: u64,
        capacity_elements: u64,
    ) -> Result<Self, Packed2PlanError> {
        let (offset, row_stride, col_stride, required_elements) =
            span(rows, cols, offset, row_stride, col_stride)?;
        if capacity_elements < required_elements {
            return Err(Packed2PlanError::CapacityTooSmall);
        }
        Ok(Self {
            rows,
            cols,
            offset,
            row_stride,
            col_stride,
            capacity_elements,
            required_elements,
        })
    }

    /// A compact row-major view. `capacity_elements` includes any prefix at
    /// `offset`.
    pub fn contiguous(
        rows: u32,
        cols: u32,
        offset: u64,
        capacity_elements: u64,
    ) -> Result<Self, Packed2PlanError> {
        Self::new(rows, cols, offset, u64::from(cols), 1, capacity_elements)
    }

    pub fn rows(&self) -> u32 {
        self.rows
    }
    pub fn cols(&self) -> u32 {
        self.cols
    }
    pub fn offset(&self) -> u32 {
        self.offset
    }
    pub fn row_stride(&self) -> u32 {
        self.row_stride
    }
    pub fn col_stride(&self) -> u32 {
        self.col_stride
    }
    pub fn capacity_elements(&self) -> u64 {
        self.capacity_elements
    }
    pub fn required_elements(&self) -> u64 {
        self.required_elements
    }

    fn is_non_overlapping_output(&self) -> bool {
        (self.col_stride == 1 && self.row_stride >= self.cols)
            || (self.row_stride == 1 && self.col_stride >= self.rows)
    }

    fn push_words(&self) -> [u32; 3] {
        [self.offset, self.row_stride, self.col_stride]
    }
}

/// An immutable logical `[rows, cols]` view of PACKED weights.
///
/// The offset and both strides are in WEIGHTS, because that is the unit the
/// format's addressing is written in and the unit the shader computes in. The
/// two capacities are in the units their buffers really have -- codes in
/// `uint32` WORDS, scales in fp32 ELEMENTS -- so that a caller cannot satisfy
/// the check by declaring a plausible number in the wrong unit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PackedView {
    rows: u32,
    cols: u32,
    offset: u32,
    row_stride: u32,
    col_stride: u32,
    block: u32,
    n_scales: u32,
    block_shift: u32,
    codes_capacity_words: u64,
    scales_capacity_elements: u64,
    required_weights: u64,
}

impl PackedView {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        rows: u32,
        cols: u32,
        offset: u64,
        row_stride: u64,
        col_stride: u64,
        block: u32,
        n_scales: u64,
        codes_capacity_words: u64,
        scales_capacity_elements: u64,
    ) -> Result<Self, Packed2PlanError> {
        if block == 0 {
            return Err(Packed2PlanError::ZeroBlock);
        }
        let (offset, row_stride, col_stride, required_weights) =
            span(rows, cols, offset, row_stride, col_stride)?;

        // The codes buffer is addressed in words and declared in words. A
        // caller who declared it in weights lands here rather than reading
        // zeros for fifteen-sixteenths of the matrix.
        if codes_capacity_words < words_for(required_weights) {
            return Err(Packed2PlanError::CodesCapacityTooSmall);
        }
        // The scale table must cover the LAST weight the plan addresses, not
        // merely the logical element count: a strided or offset view reaches
        // further than rows*cols.
        if n_scales > MAX_SCALES {
            return Err(Packed2PlanError::AddressLimitExceeded);
        }
        if n_scales < scales_for(required_weights, block) {
            return Err(Packed2PlanError::ScaleTableTooShort);
        }
        if scales_capacity_elements < n_scales {
            return Err(Packed2PlanError::CapacityTooSmall);
        }

        Ok(Self {
            rows,
            cols,
            offset,
            row_stride,
            col_stride,
            block,
            n_scales: u32::try_from(n_scales)
                .map_err(|_| Packed2PlanError::AddressLimitExceeded)?,
            // A power-of-two block gets the shift; anything else keeps the
            // divide, which is correct and slow rather than refused.
            block_shift: if block.is_power_of_two() {
                block.trailing_zeros()
            } else {
                BLOCK_SHIFT_DIVIDE
            },
            codes_capacity_words,
            scales_capacity_elements,
            required_weights,
        })
    }

    /// A compact row-major packed view over exactly `rows * cols` weights,
    /// with the scale table `pack()` produces for them.
    pub fn contiguous(
        rows: u32,
        cols: u32,
        block: u32,
        codes_capacity_words: u64,
        scales_capacity_elements: u64,
    ) -> Result<Self, Packed2PlanError> {
        let weights = u64::from(rows)
            .checked_mul(u64::from(cols))
            .ok_or(Packed2PlanError::AddressOverflow)?;
        Self::new(
            rows,
            cols,
            0,
            u64::from(cols),
            1,
            block,
            scales_for(weights, block),
            codes_capacity_words,
            scales_capacity_elements,
        )
    }

    pub fn rows(&self) -> u32 {
        self.rows
    }
    pub fn cols(&self) -> u32 {
        self.cols
    }
    pub fn offset(&self) -> u32 {
        self.offset
    }
    pub fn row_stride(&self) -> u32 {
        self.row_stride
    }
    pub fn col_stride(&self) -> u32 {
        self.col_stride
    }
    pub fn block(&self) -> u32 {
        self.block
    }
    pub fn n_scales(&self) -> u32 {
        self.n_scales
    }
    pub fn block_shift(&self) -> u32 {
        self.block_shift
    }
    /// Force the shader's divide path, for the A/B that measures what the
    /// shift is worth. Not a configuration a caller should reach for.
    pub fn with_divide_path(mut self) -> Self {
        self.block_shift = BLOCK_SHIFT_DIVIDE;
        self
    }
    pub fn codes_capacity_words(&self) -> u64 {
        self.codes_capacity_words
    }
    pub fn scales_capacity_elements(&self) -> u64 {
        self.scales_capacity_elements
    }
    pub fn required_weights(&self) -> u64 {
        self.required_weights
    }

    /// Bytes this view's operand occupies in its packed form -- the figure the
    /// residency arithmetic is actually about.
    pub fn packed_bytes(&self) -> u64 {
        words_for(self.required_weights) * 4 + u64::from(self.n_scales) * 4
    }

    /// What the same weights would occupy as fp32, for the ratio.
    pub fn fp32_bytes(&self) -> u64 {
        self.required_weights * 4
    }

    fn push_words(&self) -> [u32; 3] {
        [self.offset, self.row_stride, self.col_stride]
    }
}

/// Shared extent and addressing validation for both view kinds.
fn span(
    rows: u32,
    cols: u32,
    offset: u64,
    row_stride: u64,
    col_stride: u64,
) -> Result<(u32, u32, u32, u64), Packed2PlanError> {
    if rows == 0 || rows > MAX_DIM || cols == 0 || cols > MAX_DIM {
        return Err(Packed2PlanError::DimensionOutOfRange);
    }
    if row_stride == 0 || col_stride == 0 {
        return Err(Packed2PlanError::ZeroStride);
    }
    let logical = u64::from(rows)
        .checked_mul(u64::from(cols))
        .ok_or(Packed2PlanError::AddressOverflow)?;
    if logical > MAX_ADDRESSED_ELEMENTS {
        return Err(Packed2PlanError::AddressLimitExceeded);
    }
    let row_term = u64::from(rows - 1)
        .checked_mul(row_stride)
        .ok_or(Packed2PlanError::AddressOverflow)?;
    let col_term = u64::from(cols - 1)
        .checked_mul(col_stride)
        .ok_or(Packed2PlanError::AddressOverflow)?;
    let required = offset
        .checked_add(row_term)
        .and_then(|v| v.checked_add(col_term))
        .and_then(|v| v.checked_add(1))
        .ok_or(Packed2PlanError::AddressOverflow)?;
    if required > MAX_ADDRESSED_ELEMENTS {
        return Err(Packed2PlanError::AddressLimitExceeded);
    }
    Ok((
        u32::try_from(offset).map_err(|_| Packed2PlanError::AddressLimitExceeded)?,
        u32::try_from(row_stride).map_err(|_| Packed2PlanError::AddressLimitExceeded)?,
        u32::try_from(col_stride).map_err(|_| Packed2PlanError::AddressLimitExceeded)?,
        required,
    ))
}

/// Immutable geometry and dispatch metadata for one packed product.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Packed2Plan {
    kind: Packed2Kind,
    m: u32,
    n: u32,
    k: u32,
    a: FloatView,
    b: PackedView,
    output: FloatView,
    dispatch_groups: [u32; 3],
}

impl Packed2Plan {
    pub fn new(a: FloatView, b: PackedView, output: FloatView) -> Result<Self, Packed2PlanError> {
        Self::with_kind(Packed2Kind::Scalar, a, b, output)
    }

    /// A v2 plan: the same views and the same block, a different grid, and
    /// extra properties of B that v2's register tile depends on. Each is
    /// refused by name HERE because each produces a wrong answer rather than a
    /// fault: the shader's own copies of these checks `return` without writing
    /// and the host call still reports success.
    pub fn wide16(
        a: FloatView,
        b: PackedView,
        output: FloatView,
    ) -> Result<Self, Packed2PlanError> {
        Self::with_kind(Packed2Kind::Wide16, a, b, output)
    }

    pub fn with_kind(
        kind: Packed2Kind,
        a: FloatView,
        b: PackedView,
        output: FloatView,
    ) -> Result<Self, Packed2PlanError> {
        if kind == Packed2Kind::Wide16 {
            if b.col_stride != 1 {
                return Err(Packed2PlanError::ColumnStrideNotOne);
            }
            if b.offset % WEIGHTS_PER_WORD != 0 {
                return Err(Packed2PlanError::OffsetNotWordAligned);
            }
            if b.rows > 1 && b.row_stride % WEIGHTS_PER_WORD != 0 {
                return Err(Packed2PlanError::RowStrideNotWordAligned);
            }
            if b.block % WEIGHTS_PER_WORD != 0 {
                return Err(Packed2PlanError::BlockNotWordAligned);
            }
        }
        if a.cols != b.rows || output.rows != a.rows || output.cols != b.cols {
            return Err(Packed2PlanError::DimensionMismatch);
        }
        if a.cols > MAX_K {
            return Err(Packed2PlanError::DimensionOutOfRange);
        }
        if !output.is_non_overlapping_output() {
            return Err(Packed2PlanError::OutputOverlap);
        }
        Ok(Self {
            kind,
            m: a.rows,
            n: b.cols,
            k: a.cols,
            a,
            b,
            output,
            // v1 cuts the grid to one output element per invocation; v2 to
            // sixteen columns per invocation, so a workgroup covers
            // TILE x (TILE * 16).
            dispatch_groups: match kind {
                Packed2Kind::Scalar => [b.cols.div_ceil(TILE), a.rows.div_ceil(TILE), 1],
                Packed2Kind::Wide16 => [
                    b.cols.div_ceil(TILE * COLS_PER_INVOCATION),
                    a.rows.div_ceil(TILE),
                    1,
                ],
            },
        })
    }

    pub fn kind(&self) -> Packed2Kind {
        self.kind
    }

    /// Elements a v2 control's fp32 weight buffer must hold: the plan's reach
    /// rounded UP to a whole sixteen-column group. v2's dead lanes read the
    /// group past the last live column, and the shader `return`s rather than
    /// storing if that read would be out of range -- which is a missing
    /// output, not an error. Allocating this much makes the guard unreachable
    /// instead of leaving it as a silent partial write.
    pub fn control_capacity_elements(&self) -> u64 {
        self.b
            .required_weights
            .div_ceil(u64::from(COLS_PER_INVOCATION))
            * u64::from(COLS_PER_INVOCATION)
    }

    pub fn m(&self) -> u32 {
        self.m
    }
    pub fn n(&self) -> u32 {
        self.n
    }
    pub fn k(&self) -> u32 {
        self.k
    }
    pub fn a(&self) -> FloatView {
        self.a
    }
    pub fn b(&self) -> PackedView {
        self.b
    }
    pub fn output(&self) -> FloatView {
        self.output
    }
    /// The same plan with the shader's DIVIDE path selected.
    pub fn with_divide_path(mut self) -> Self {
        self.b = self.b.with_divide_path();
        self
    }
    pub fn dispatch_groups(&self) -> [u32; 3] {
        self.dispatch_groups
    }

    /// Multiply-accumulates this product performs: `m * n * k`. The denominator
    /// of any decode-cost-per-MAC figure, so it is derived here rather than
    /// recomputed at each call site.
    pub fn macs(&self) -> u64 {
        u64::from(self.m) * u64::from(self.n) * u64::from(self.k)
    }

    /// Weight bytes a fully-cold pass over B must read, packed and as fp32.
    /// A LOWER BOUND on traffic, not an estimate of it: it counts each weight
    /// once and this kernel re-reads B once per output row.
    pub fn weight_bytes(&self) -> (u64, u64) {
        (self.b.packed_bytes(), self.b.fp32_bytes())
    }

    pub fn push_constants(&self) -> [u8; PUSH_BYTES] {
        let mut words = [0u32; PUSH_BYTES / 4];
        words[0] = self.m;
        words[1] = self.n;
        words[2] = self.k;
        words[3..6].copy_from_slice(&self.a.push_words());
        words[6..9].copy_from_slice(&self.b.push_words());
        words[9..12].copy_from_slice(&self.output.push_words());
        words[12] = self.b.block;
        words[13] = self.b.n_scales;
        words[14] = self.b.block_shift;

        let mut bytes = [0u8; PUSH_BYTES];
        for (index, word) in words.iter().enumerate() {
            bytes[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }
}
