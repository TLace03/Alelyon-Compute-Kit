//! Pure geometry, capacity, and dispatch planning for `matmul-f32/v1`.
//!
//! The shader evaluates `C[b, i, j] = sum_l A[b, i, l] * B[b, l, j]`.
//! Batch one is the `mm` form; `bmm` requires equal, explicit batches and does
//! not broadcast. Input views may have arbitrary positive strides. Output
//! views must be non-overlapping padded row-major or column-major matrices.
//!
//! This module has no native `Context` or `Buffer` dependency. Its capacities
//! are caller declarations, so a successful plan does not prove buffer
//! ownership, liveness, actual capacity, or output/input non-aliasing. The
//! native integration must establish those properties from real buffer
//! objects immediately before binding and dispatch.

use std::fmt;

pub const ABI: &str = "matmul-f32/v1";

/// Batch count ceiling. Unraised; `MAX_BATCH * MAX_DIM` is exactly
/// `MAX_ADDRESSED_ELEMENTS`, so the shader's `p.batch * p.m` (a 32-bit
/// multiply) has no headroom left and neither of the two may rise alone.
pub const MAX_BATCH: u32 = 4_096;

/// Ceiling on a view's rows and columns, i.e. on `m` and `n`. Justified by
/// addressing arithmetic alone: one dimension of `MAX_ADDRESSED_ELEMENTS`
/// f32 is the largest a single extent can be under the element cap.
pub const MAX_DIM: u32 = 65_536;

/// Ceiling on the reduction length `k`. Equal to `MAX_DIM` today, but it is a
/// SEPARATE claim with a SEPARATE justification: `m` and `n` are bounded by
/// addressing, `k` is bounded by ACCURACY, because the shader's Neumaier
/// compensated summation with a product-residual correction accumulates over
/// exactly `k` terms. 65,536 is the largest `k` actually swept against an
/// exactly-rounded `fsum` reference (see `results/capacity_raise_probe/`), not
/// an extrapolation, and raising `MAX_DIM` past it must not silently raise it.
pub const MAX_K: u32 = 65_536;

/// Ceiling on the elements one operand view may address, logical and spanned.
/// Mirrored in the shader as `LIMIT`, a 32-bit uint, so it must fit u32.
pub const MAX_ADDRESSED_ELEMENTS: u64 = 268_435_456;
pub const TILE: u32 = 8;
pub const WORKGROUP_INVOCATIONS: u32 = TILE * TILE;
pub const PUSH_BYTES: usize = 64;
/// matmul-f32/v2 (2026-09-09): the PLAIN-fp32 tiled kernel behind kinds 2 and
/// 3 of `ack_matmul_f32`. Same three buffers, same 64-byte block, same
/// capacity guards; fp32 accumulation with no compensation term, a 64 x 64
/// output tile per workgroup walked in k steps of 16, a 4 x 4 register
/// micro-tile per invocation. v1 stays the kernel of kinds 0 and 1 and of the
/// strict fp32 diagnostic mode.
pub const ABI_V2: &str = "matmul-f32/v2";
/// The operation schema `ack_matmul_f32` reports. 1 was kinds 0-1 on v1; 2
/// adds kinds 2-3 on v2 with the SAME block and entry, so the layout did not
/// move and only this pin did, exactly as pointwise schema 2 was done.
pub const SCHEMA: u32 = 2;
pub const V2_TILE_M: u32 = 64;
pub const V2_TILE_N: u32 = 64;
pub const V2_TILE_K: u32 = 16;
pub const V2_WORKGROUP_INVOCATIONS: u32 = 256;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum MatmulKind {
    /// One product on the near-exact compensated kernel (matmul-f32/v1).
    Mm = 0,
    /// A batch of products on v1, through the batch strides.
    Bmm = 1,
    /// One product in PLAIN fp32 on matmul-f32/v2 (schema 2).
    MmPlain = 2,
    /// A batch of products on v2.
    BmmPlain = 3,
}

impl MatmulKind {
    /// v2's arithmetic contract (fp32 accumulation, no compensation) rather
    /// than v1's; decides which kernel `ack_matmul_f32` binds and which tile
    /// the dispatch grid is cut to.
    pub fn is_plain(self) -> bool {
        matches!(self, Self::MmPlain | Self::BmmPlain)
    }
    /// A batched product, whose batch word may exceed one.
    pub fn is_batched(self) -> bool {
        matches!(self, Self::Bmm | Self::BmmPlain)
    }
}

impl TryFrom<u32> for MatmulKind {
    type Error = MatmulPlanError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Mm),
            1 => Ok(Self::Bmm),
            2 => Ok(Self::MmPlain),
            3 => Ok(Self::BmmPlain),
            _ => Err(MatmulPlanError::OperationOutOfRange),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MatmulPlanError {
    OperationOutOfRange,
    DimensionOutOfRange,
    DimensionMismatch,
    BatchMismatch,
    MmRequiresSingleBatch,
    ZeroStride,
    AddressOverflow,
    AddressLimitExceeded,
    CapacityTooSmall,
    OutputOverlap,
}

impl fmt::Display for MatmulPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::OperationOutOfRange => "matmul-operation-out-of-range",
            Self::DimensionOutOfRange => "matmul-dimension-out-of-range",
            Self::DimensionMismatch => "matmul-dimension-mismatch",
            Self::BatchMismatch => "matmul-batch-mismatch",
            Self::MmRequiresSingleBatch => "matmul-mm-requires-single-batch",
            Self::ZeroStride => "matmul-zero-stride",
            Self::AddressOverflow => "matmul-address-overflow",
            Self::AddressLimitExceeded => "matmul-address-limit-exceeded",
            Self::CapacityTooSmall => "matmul-capacity-too-small",
            Self::OutputOverlap => "matmul-output-overlap",
        })
    }
}

impl std::error::Error for MatmulPlanError {}

/// An immutable logical `[batch, rows, cols]` FP32 view into one storage.
///
/// `capacity_elements` is deliberately part of the caller's declaration. It
/// bounds address arithmetic here, but only native buffer inspection can prove
/// the declaration matches the allocation used for dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MatrixView {
    batch: u32,
    rows: u32,
    cols: u32,
    offset: u32,
    batch_stride: u32,
    row_stride: u32,
    col_stride: u32,
    capacity_elements: u64,
    required_elements: u64,
    logical_elements: u64,
}

impl MatrixView {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        batch: u32,
        rows: u32,
        cols: u32,
        offset: u64,
        batch_stride: u64,
        row_stride: u64,
        col_stride: u64,
        capacity_elements: u64,
    ) -> Result<Self, MatmulPlanError> {
        if batch == 0
            || batch > MAX_BATCH
            || rows == 0
            || rows > MAX_DIM
            || cols == 0
            || cols > MAX_DIM
        {
            return Err(MatmulPlanError::DimensionOutOfRange);
        }
        if batch_stride == 0 || row_stride == 0 || col_stride == 0 {
            return Err(MatmulPlanError::ZeroStride);
        }

        let logical_elements = u64::from(batch)
            .checked_mul(u64::from(rows))
            .and_then(|value| value.checked_mul(u64::from(cols)))
            .ok_or(MatmulPlanError::AddressOverflow)?;
        if logical_elements > MAX_ADDRESSED_ELEMENTS {
            return Err(MatmulPlanError::AddressLimitExceeded);
        }

        let batch_term = u64::from(batch - 1)
            .checked_mul(batch_stride)
            .ok_or(MatmulPlanError::AddressOverflow)?;
        let row_term = u64::from(rows - 1)
            .checked_mul(row_stride)
            .ok_or(MatmulPlanError::AddressOverflow)?;
        let col_term = u64::from(cols - 1)
            .checked_mul(col_stride)
            .ok_or(MatmulPlanError::AddressOverflow)?;
        let max_index = offset
            .checked_add(batch_term)
            .and_then(|value| value.checked_add(row_term))
            .and_then(|value| value.checked_add(col_term))
            .ok_or(MatmulPlanError::AddressOverflow)?;
        let required_elements = max_index
            .checked_add(1)
            .ok_or(MatmulPlanError::AddressOverflow)?;
        if required_elements > MAX_ADDRESSED_ELEMENTS {
            return Err(MatmulPlanError::AddressLimitExceeded);
        }
        if capacity_elements < required_elements {
            return Err(MatmulPlanError::CapacityTooSmall);
        }

        Ok(Self {
            batch,
            rows,
            cols,
            offset: u32::try_from(offset).map_err(|_| MatmulPlanError::AddressLimitExceeded)?,
            batch_stride: u32::try_from(batch_stride)
                .map_err(|_| MatmulPlanError::AddressLimitExceeded)?,
            row_stride: u32::try_from(row_stride)
                .map_err(|_| MatmulPlanError::AddressLimitExceeded)?,
            col_stride: u32::try_from(col_stride)
                .map_err(|_| MatmulPlanError::AddressLimitExceeded)?,
            capacity_elements,
            required_elements,
            logical_elements,
        })
    }

    /// A compact row-major view. `capacity_elements` includes any prefix at
    /// `offset`; an exact full-size view therefore cannot also have an offset.
    pub fn contiguous(
        batch: u32,
        rows: u32,
        cols: u32,
        offset: u64,
        capacity_elements: u64,
    ) -> Result<Self, MatmulPlanError> {
        let row_stride = u64::from(cols);
        let batch_stride = u64::from(rows)
            .checked_mul(row_stride)
            .ok_or(MatmulPlanError::AddressOverflow)?;
        Self::new(
            batch,
            rows,
            cols,
            offset,
            batch_stride,
            row_stride,
            1,
            capacity_elements,
        )
    }

    /// A logical `[rows, cols]` view of compact `[cols, rows]` storage.
    pub fn transposed_storage(
        batch: u32,
        rows: u32,
        cols: u32,
        offset: u64,
        capacity_elements: u64,
    ) -> Result<Self, MatmulPlanError> {
        let col_stride = u64::from(rows);
        let batch_stride = u64::from(cols)
            .checked_mul(col_stride)
            .ok_or(MatmulPlanError::AddressOverflow)?;
        Self::new(
            batch,
            rows,
            cols,
            offset,
            batch_stride,
            1,
            col_stride,
            capacity_elements,
        )
    }

    pub fn batch(&self) -> u32 {
        self.batch
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
    pub fn batch_stride(&self) -> u32 {
        self.batch_stride
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
    pub fn logical_elements(&self) -> u64 {
        self.logical_elements
    }

    fn matrix_span(&self) -> Result<u64, MatmulPlanError> {
        u64::from(self.rows - 1)
            .checked_mul(u64::from(self.row_stride))
            .and_then(|value| {
                value.checked_add(u64::from(self.cols - 1) * u64::from(self.col_stride))
            })
            .and_then(|value| value.checked_add(1))
            .ok_or(MatmulPlanError::AddressOverflow)
    }

    fn is_non_overlapping_output(&self) -> Result<bool, MatmulPlanError> {
        let row_major = self.col_stride == 1 && self.row_stride >= self.cols;
        let column_major = self.row_stride == 1 && self.col_stride >= self.rows;
        Ok((row_major || column_major)
            && (self.batch == 1 || u64::from(self.batch_stride) >= self.matrix_span()?))
    }

    fn push_words(&self) -> [u32; 4] {
        [
            self.offset,
            self.batch_stride,
            self.row_stride,
            self.col_stride,
        ]
    }
}

/// Immutable geometry and dispatch metadata for one three-buffer product.
///
/// The plan validates declared extents and output non-overlap. Native code must
/// still prove that all buffers belong to the same live context, their real
/// capacities cover these declarations, and the output buffer is distinct
/// from both inputs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MatmulPlan {
    kind: MatmulKind,
    batch: u32,
    m: u32,
    n: u32,
    k: u32,
    a: MatrixView,
    b: MatrixView,
    output: MatrixView,
    dispatch_groups: [u32; 3],
}

impl MatmulPlan {
    pub fn new(
        kind: MatmulKind,
        a: MatrixView,
        b: MatrixView,
        output: MatrixView,
    ) -> Result<Self, MatmulPlanError> {
        if a.batch != b.batch || a.batch != output.batch {
            return Err(MatmulPlanError::BatchMismatch);
        }
        if !kind.is_batched() && a.batch != 1 {
            return Err(MatmulPlanError::MmRequiresSingleBatch);
        }
        if a.cols != b.rows || output.rows != a.rows || output.cols != b.cols {
            return Err(MatmulPlanError::DimensionMismatch);
        }
        // The reduction length is bounded separately from `m` and `n`: a
        // `MatrixView` cannot see which of its extents is `k`, and only here
        // is that known. Redundant while MAX_K == MAX_DIM, live the moment
        // addressing licenses a wider extent than the accuracy sweep did.
        if a.cols > MAX_K {
            return Err(MatmulPlanError::DimensionOutOfRange);
        }
        if !output.is_non_overlapping_output()? {
            return Err(MatmulPlanError::OutputOverlap);
        }

        Ok(Self {
            kind,
            batch: a.batch,
            m: a.rows,
            n: b.cols,
            k: a.cols,
            a,
            b,
            output,
            // the grid is cut to the kernel the kind binds: v1's 8 x 8 tile or
            // v2's 64 x 64 one, div_ceil so the partial edge tile is launched
            dispatch_groups: if kind.is_plain() {
                [
                    b.cols.div_ceil(V2_TILE_N),
                    a.rows.div_ceil(V2_TILE_M),
                    a.batch,
                ]
            } else {
                [b.cols.div_ceil(TILE), a.rows.div_ceil(TILE), a.batch]
            },
        })
    }

    pub fn mm(a: MatrixView, b: MatrixView, output: MatrixView) -> Result<Self, MatmulPlanError> {
        Self::new(MatmulKind::Mm, a, b, output)
    }

    pub fn bmm(a: MatrixView, b: MatrixView, output: MatrixView) -> Result<Self, MatmulPlanError> {
        Self::new(MatmulKind::Bmm, a, b, output)
    }

    pub fn mm_plain(
        a: MatrixView,
        b: MatrixView,
        output: MatrixView,
    ) -> Result<Self, MatmulPlanError> {
        Self::new(MatmulKind::MmPlain, a, b, output)
    }

    pub fn bmm_plain(
        a: MatrixView,
        b: MatrixView,
        output: MatrixView,
    ) -> Result<Self, MatmulPlanError> {
        Self::new(MatmulKind::BmmPlain, a, b, output)
    }

    pub fn kind(&self) -> MatmulKind {
        self.kind
    }
    pub fn batch(&self) -> u32 {
        self.batch
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
    pub fn a(&self) -> MatrixView {
        self.a
    }
    pub fn b(&self) -> MatrixView {
        self.b
    }
    pub fn output(&self) -> MatrixView {
        self.output
    }
    pub fn dispatch_groups(&self) -> [u32; 3] {
        self.dispatch_groups
    }

    pub fn push_constants(&self) -> [u8; PUSH_BYTES] {
        let mut words = [0u32; PUSH_BYTES / 4];
        words[..4].copy_from_slice(&[self.batch, self.m, self.n, self.k]);
        words[4..8].copy_from_slice(&self.a.push_words());
        words[8..12].copy_from_slice(&self.b.push_words());
        words[12..16].copy_from_slice(&self.output.push_words());

        let mut bytes = [0u8; PUSH_BYTES];
        for (index, word) in words.iter().enumerate() {
            bytes[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }
}
