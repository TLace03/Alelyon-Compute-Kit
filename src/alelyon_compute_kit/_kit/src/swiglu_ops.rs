//! Pure validation and dispatch planning for swiglu/v1 (2026-09-29): the
//! SiLU-gated activation of one packed projection, `out = u * silu(g)` with
//! `u, g = packed.chunk(2, -1)`, and its backward, over FP32 or bf16 storage
//! (`kernels/swiglu.comp`).
//!
//! The packed operand holds `rows` rows of `2 * cols`. The forward writes
//! `rows` rows of `cols`. The backward reads the upstream gradient (`rows` rows
//! of `cols`) and writes the packed operand's gradient (`rows` rows of
//! `2 * cols`), both halves in one dispatch. The law is the decomposition
//! torch runs for the storage dtype through the kit's pointwise operations 14,
//! 2 and 37, with every intermediate tensor rounded to that dtype. The kernel's
//! header states it step by step.
//!
//! These bounds limit this implementation's contract; they are not limits of
//! Vulkan or of any selected GPU.
//!
//! swiglu-scaled/v1 (2026-09-30, [`SwigluScaledPlan`], `kernels/swiglu_scaled.comp`)
//! is the sub-bit student's experts' activation, `out = silu(gate) * (up * g)`
//! (`research/subbit_expert/moe/formats.py` `_swiglu_f`), with gate and up
//! separate `rows` x `cols` operands and g one `cols` row for every `width`
//! consecutive rows, over FP32, bf16 or fp16 storage. Its backward writes three
//! outputs of `rows` x `cols`, each to its own buffer: gate's gradient, up's
//! gradient, and the product whose sum over each group's rows is g's gradient,
//! a sum the caller leaves to torch. The law is again the kit's own decomposition in that dtype (the
//! kernel's header states it), so every value that is not a NaN is that
//! decomposition's bit for bit.

use std::fmt;

pub const ABI: &str = "swiglu/v1";
/// Operations 0 (forward) and 1 (backward) over the push block below.
pub const SCHEMA: u32 = 1;
pub const WORKGROUP: u32 = 256;
/// Every buffer the kernel addresses holds at most this many elements, so an
/// element index stays below 2^31.
pub const MAX_ELEMENTS: usize = 1 << 30;
/// Vulkan's guaranteed minimum for one dispatch dimension.
pub const MAX_GROUPS_X: u32 = 65_535;
/// rows, cols, op, elements: four u32 words.
pub const PUSH_BYTES: usize = 16;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum SwigluOp {
    Forward = 0,
    Backward = 1,
}

impl TryFrom<u32> for SwigluOp {
    type Error = SwigluPlanError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Forward),
            1 => Ok(Self::Backward),
            _ => Err(SwigluPlanError::OperationOutOfRange),
        }
    }
}

/// The storage of every operand of one call: all FP32, or all bf16.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum SwigluStorage {
    F32 = 0,
    Bf16 = 1,
}

impl SwigluStorage {
    pub fn bytes(self) -> u64 {
        match self {
            Self::F32 => 4,
            Self::Bf16 => 2,
        }
    }
}

impl TryFrom<u32> for SwigluStorage {
    type Error = SwigluPlanError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::F32),
            1 => Ok(Self::Bf16),
            _ => Err(SwigluPlanError::StorageOutOfRange),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SwigluPlanError {
    DimensionOutOfRange,
    ElementLimitExceeded,
    OperationOutOfRange,
    StorageOutOfRange,
}

impl fmt::Display for SwigluPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::DimensionOutOfRange => "swiglu-dimension-out-of-range",
            Self::ElementLimitExceeded => "swiglu-element-limit-exceeded",
            Self::OperationOutOfRange => "swiglu-operation-out-of-range",
            Self::StorageOutOfRange => "swiglu-storage-out-of-range",
        })
    }
}

impl std::error::Error for SwigluPlanError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SwigluPlan {
    rows: u32,
    cols: u32,
    op: SwigluOp,
    storage: SwigluStorage,
    elements: usize,
    dispatch_groups: [u32; 3],
}

impl SwigluPlan {
    /// `rows` rows; each half of the packed operand is `cols` wide. A zero
    /// dimension and a packed operand past the element bound are refused by name.
    pub fn new(
        rows: u32,
        cols: u32,
        op: SwigluOp,
        storage: SwigluStorage,
    ) -> Result<Self, SwigluPlanError> {
        if rows == 0 || cols == 0 {
            return Err(SwigluPlanError::DimensionOutOfRange);
        }
        let elements = rows as usize * cols as usize;
        if 2 * elements > MAX_ELEMENTS {
            return Err(SwigluPlanError::ElementLimitExceeded);
        }
        let groups = (elements as u32).div_ceil(WORKGROUP);
        let dispatch_groups = if groups <= MAX_GROUPS_X {
            [groups, 1, 1]
        } else {
            [MAX_GROUPS_X, groups.div_ceil(MAX_GROUPS_X), 1]
        };
        Ok(Self {
            rows,
            cols,
            op,
            storage,
            elements,
            dispatch_groups,
        })
    }

    pub fn rows(&self) -> u32 {
        self.rows
    }
    pub fn cols(&self) -> u32 {
        self.cols
    }
    pub fn op(&self) -> SwigluOp {
        self.op
    }
    pub fn storage(&self) -> SwigluStorage {
        self.storage
    }
    /// Elements of the forward's output and of the upstream gradient.
    pub fn elements(&self) -> usize {
        self.elements
    }
    /// Elements of the packed operand, and of the backward's output.
    pub fn packed_len(&self) -> usize {
        2 * self.elements
    }
    /// Elements the output buffer must hold.
    pub fn output_len(&self) -> usize {
        match self.op {
            SwigluOp::Forward => self.elements,
            SwigluOp::Backward => 2 * self.elements,
        }
    }
    pub fn dispatch_groups(&self) -> [u32; 3] {
        self.dispatch_groups
    }

    pub fn push_constants(&self) -> [u8; PUSH_BYTES] {
        let mut bytes = [0u8; PUSH_BYTES];
        let words = [self.rows, self.cols, self.op as u32, self.elements as u32];
        for (index, word) in words.iter().enumerate() {
            bytes[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }
}

/// swiglu-scaled/v1's name and its push block: rows, cols, width, op, elements.
pub const SCALED_ABI: &str = "swiglu-scaled/v1";
pub const SCALED_PUSH_BYTES: usize = 20;

/// swiglu-scaled/v1's storage for every operand of one call: all FP32, bf16 or
/// fp16 (the student's experts run under fp16 autocast). A separate type from
/// [`SwigluStorage`], which stays FP32 and bf16: `ack_swiglu` has no fp16 module.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum ScaledStorage {
    F32 = 0,
    Bf16 = 1,
    F16 = 2,
}

impl ScaledStorage {
    pub fn bytes(self) -> u64 {
        match self {
            Self::F32 => 4,
            Self::Bf16 | Self::F16 => 2,
        }
    }
}

impl TryFrom<u32> for ScaledStorage {
    type Error = SwigluPlanError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::F32),
            1 => Ok(Self::Bf16),
            2 => Ok(Self::F16),
            _ => Err(SwigluPlanError::StorageOutOfRange),
        }
    }
}

/// One swiglu-scaled/v1 dispatch: `rows` rows of `cols` for gate, up, the
/// upstream gradient and each output, and one scale row a `width` rows.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SwigluScaledPlan {
    rows: u32,
    cols: u32,
    width: u32,
    op: SwigluOp,
    storage: ScaledStorage,
    elements: usize,
    dispatch_groups: [u32; 3],
}

impl SwigluScaledPlan {
    /// A zero dimension, a `width` that does not divide `rows`, and an operand
    /// past the element bound are refused by name.
    pub fn new(
        rows: u32,
        cols: u32,
        width: u32,
        op: SwigluOp,
        storage: ScaledStorage,
    ) -> Result<Self, SwigluPlanError> {
        if rows == 0 || cols == 0 || width == 0 || rows % width != 0 {
            return Err(SwigluPlanError::DimensionOutOfRange);
        }
        let elements = rows as usize * cols as usize;
        if elements > MAX_ELEMENTS {
            return Err(SwigluPlanError::ElementLimitExceeded);
        }
        let groups = (elements as u32).div_ceil(WORKGROUP);
        let dispatch_groups = if groups <= MAX_GROUPS_X {
            [groups, 1, 1]
        } else {
            [MAX_GROUPS_X, groups.div_ceil(MAX_GROUPS_X), 1]
        };
        Ok(Self {
            rows,
            cols,
            width,
            op,
            storage,
            elements,
            dispatch_groups,
        })
    }

    pub fn op(&self) -> SwigluOp {
        self.op
    }
    pub fn storage(&self) -> ScaledStorage {
        self.storage
    }
    /// Elements of gate, of up, of the upstream gradient and of each output.
    pub fn elements(&self) -> usize {
        self.elements
    }
    /// Elements of the scale operand: one row of `cols` a `width` rows.
    pub fn scale_len(&self) -> usize {
        (self.rows / self.width) as usize * self.cols as usize
    }
    pub fn dispatch_groups(&self) -> [u32; 3] {
        self.dispatch_groups
    }

    pub fn push_constants(&self) -> [u8; SCALED_PUSH_BYTES] {
        let mut bytes = [0u8; SCALED_PUSH_BYTES];
        let words = [
            self.rows,
            self.cols,
            self.width,
            self.op as u32,
            self.elements as u32,
        ];
        for (index, word) in words.iter().enumerate() {
            bytes[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_scaled_plan_takes_fp16_refuses_by_name_and_carries_its_push_block() {
        assert_eq!(ScaledStorage::try_from(2), Ok(ScaledStorage::F16));
        assert_eq!(
            ScaledStorage::try_from(3),
            Err(SwigluPlanError::StorageOutOfRange)
        );
        assert_eq!(
            SwigluStorage::try_from(2),
            Err(SwigluPlanError::StorageOutOfRange),
            "ack_swiglu keeps its two storages"
        );
        let (b, f16) = (SwigluOp::Backward, ScaledStorage::F16);
        let named = |r: Result<SwigluScaledPlan, SwigluPlanError>| r.unwrap_err().to_string();
        assert_eq!(
            named(SwigluScaledPlan::new(0, 8, 1, b, f16)),
            "swiglu-dimension-out-of-range"
        );
        assert_eq!(
            named(SwigluScaledPlan::new(8, 8, 0, b, f16)),
            "swiglu-dimension-out-of-range"
        );
        assert_eq!(
            named(SwigluScaledPlan::new(9, 8, 2, b, f16)),
            "swiglu-dimension-out-of-range",
            "a width that does not divide the rows"
        );
        assert_eq!(
            named(SwigluScaledPlan::new((1 << 19) + 1, 1 << 11, 1, b, f16)),
            "swiglu-element-limit-exceeded",
            "one row past 2^30 elements"
        );
        assert!(SwigluScaledPlan::new(1 << 19, 1 << 11, 1, b, f16).is_ok());
        // a stage-E group: 12 experts of 640 rows, FF 768
        let plan = SwigluScaledPlan::new(12 * 640, 768, 640, b, f16).unwrap();
        assert_eq!((plan.elements(), plan.scale_len()), (5_898_240, 12 * 768));
        let push = plan.push_constants();
        let word = |i: usize| u32::from_le_bytes(push[i * 4..i * 4 + 4].try_into().unwrap());
        assert_eq!(
            (0..5).map(word).collect::<Vec<_>>(),
            vec![7_680, 768, 640, 1, 5_898_240]
        );
        let forward = SwigluScaledPlan::new(12 * 640, 768, 640, SwigluOp::Forward, f16).unwrap();
        assert_eq!(forward.op(), SwigluOp::Forward);
        assert_eq!(ScaledStorage::F16.bytes(), 2);
    }

    #[test]
    fn the_words_decode_and_nothing_past_them_does() {
        assert_eq!(SwigluOp::try_from(0), Ok(SwigluOp::Forward));
        assert_eq!(SwigluOp::try_from(1), Ok(SwigluOp::Backward));
        assert_eq!(
            SwigluOp::try_from(2),
            Err(SwigluPlanError::OperationOutOfRange)
        );
        assert_eq!(SwigluStorage::try_from(1), Ok(SwigluStorage::Bf16));
        assert_eq!(
            SwigluStorage::try_from(2),
            Err(SwigluPlanError::StorageOutOfRange)
        );
        assert_eq!(
            (SwigluStorage::F32.bytes(), SwigluStorage::Bf16.bytes()),
            (4, 2)
        );
        assert_eq!(SCHEMA, 1);
    }

    #[test]
    fn refusals_are_named() {
        let (f, s) = (SwigluOp::Forward, SwigluStorage::Bf16);
        let named = |r: Result<SwigluPlan, SwigluPlanError>| r.unwrap_err().to_string();
        assert_eq!(
            named(SwigluPlan::new(0, 8, f, s)),
            "swiglu-dimension-out-of-range"
        );
        assert_eq!(
            named(SwigluPlan::new(8, 0, f, s)),
            "swiglu-dimension-out-of-range"
        );
        assert_eq!(
            named(SwigluPlan::new(1 << 20, 1 << 10, f, s)),
            "swiglu-element-limit-exceeded"
        );
    }

    #[test]
    fn the_push_block_the_lengths_and_the_grid_carry_the_plan() {
        // the A3r-X2 FFN: 4 x 1024 tokens, halves of 1,628
        let plan = SwigluPlan::new(4096, 1628, SwigluOp::Backward, SwigluStorage::Bf16).unwrap();
        assert_eq!(
            (plan.elements(), plan.packed_len(), plan.output_len()),
            (6_668_288, 13_336_576, 13_336_576)
        );
        assert_eq!(plan.dispatch_groups(), [26_048, 1, 1]);
        let push = plan.push_constants();
        let word = |i: usize| u32::from_le_bytes(push[i * 4..i * 4 + 4].try_into().unwrap());
        assert_eq!(
            (0..4).map(word).collect::<Vec<_>>(),
            vec![4096, 1628, 1, 6_668_288]
        );
        let forward = SwigluPlan::new(4096, 1628, SwigluOp::Forward, SwigluStorage::F32).unwrap();
        assert_eq!(forward.output_len(), 6_668_288);
        // past one dimension's guarantee the grid folds into a second
        let big = SwigluPlan::new(1 << 16, 1 << 12, SwigluOp::Forward, SwigluStorage::F32).unwrap();
        let [x, y, z] = big.dispatch_groups();
        assert_eq!((x, z), (MAX_GROUPS_X, 1));
        assert!(x as u64 * y as u64 * WORKGROUP as u64 >= big.elements() as u64);
    }
}
