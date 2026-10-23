//! Pure admission for the optional vq4/v1 extension. Four-bit indices select
//! one of 16 FP32 vectors; groups have 8, 16 or 32 logical values. Costs
//! include padded codewords and every codebook value. This is lossy storage,
//! not a training-quality, throughput or fractional scalar-type claim.
//!
//! The FFI supplies actual live allocation capacities/identities immediately
//! before binding. Caller-declared capacities alone do not establish safety.
//! Geometry is deliberately bounded per active tensor; larger expert banks
//! must be sharded. Op3 emits active FP32 scratch, not a persistent master.

use std::fmt;

pub const SCHEMA: u32 = 1;
pub const OPERANDS: usize = 6;
pub const PUSH_BYTES: usize = 32;
pub const ENTRIES: u32 = 16;
pub const INDICES_PER_WORD: u32 = 8;
pub const WORKGROUP_SIZE: u32 = 64;
pub const MAX_VALUES: u32 = 1 << 28;
pub const MAX_DIM: u32 = 65536;
/// Bound one scalar-kernel dispatch independently of storage capacity.
/// Larger active products need explicit chunking before admission.
pub const MAX_MATMUL_MACS: u64 = 1 << 28;
const GRID_X: u32 = 65535;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum VqOp {
    Encode = 0,
    Decode = 1,
    Matmul = 2,
    AdamW = 3,
}

impl TryFrom<u32> for VqOp {
    type Error = VqPlanError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Encode),
            1 => Ok(Self::Decode),
            2 => Ok(Self::Matmul),
            3 => Ok(Self::AdamW),
            _ => Err(VqPlanError::Operation),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VqPlanError {
    Operation,
    Group,
    Entries,
    Flags,
    Reserved,
    Geometry,
    AddressLimit,
    WorkLimit,
    Capacity(usize),
    NullIdentity,
    OutputAlias,
    StatusAlias,
    Dispatch,
}

impl fmt::Display for VqPlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Operation => f.write_str("vq-operation-out-of-range"),
            Self::Group => f.write_str("vq-unsupported-group"),
            Self::Entries => f.write_str("vq-codebook-entries-not-16"),
            Self::Flags => f.write_str("vq-unsupported-flags"),
            Self::Reserved => f.write_str("vq-nonzero-reserved-word"),
            Self::Geometry => f.write_str("vq-invalid-geometry"),
            Self::AddressLimit => f.write_str("vq-address-limit-exceeded"),
            Self::WorkLimit => f.write_str("vq-matmul-work-limit"),
            Self::Capacity(slot) => write!(f, "vq-binding-{slot}-too-small"),
            Self::NullIdentity => f.write_str("vq-null-allocation-identity"),
            Self::OutputAlias => f.write_str("vq-output-aliases-input"),
            Self::StatusAlias => f.write_str("vq-status-aliases-operand"),
            Self::Dispatch => f.write_str("vq-dispatch-grid-unsupported"),
        }
    }
}

impl std::error::Error for VqPlanError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VqPlan {
    words: [u32; 8],
    op: VqOp,
    required_bytes: [u64; OPERANDS],
    work_items: u32,
    dispatch: [u32; 3],
}

fn packed_bytes(values: u32, group: u32) -> u64 {
    u64::from(values.div_ceil(group).div_ceil(INDICES_PER_WORD)) * 4
}

fn product(a: u32, b: u32) -> Result<u32, VqPlanError> {
    let values = u64::from(a) * u64::from(b);
    if values > u64::from(MAX_VALUES) {
        Err(VqPlanError::AddressLimit)
    } else {
        Ok(values as u32)
    }
}

impl VqPlan {
    pub fn from_words(words: [u32; 8]) -> Result<Self, VqPlanError> {
        let [op, m, n, k, group, entries, flags, reserved] = words;
        let op = VqOp::try_from(op)?;
        if !matches!(group, 8 | 16 | 32) {
            return Err(VqPlanError::Group);
        }
        if entries != ENTRIES {
            return Err(VqPlanError::Entries);
        }
        if reserved != 0 {
            return Err(VqPlanError::Reserved);
        }
        if (op == VqOp::Matmul && flags > 3) || (op != VqOp::Matmul && flags != 0) {
            return Err(VqPlanError::Flags);
        }
        if m == 0 || n == 0 || k == 0 {
            return Err(VqPlanError::Geometry);
        }
        let book = u64::from(ENTRIES * group) * 4;
        let (required_bytes, work_items) = match op {
            VqOp::Encode | VqOp::Decode | VqOp::AdamW => {
                if n != 1 || k != 1 {
                    return Err(VqPlanError::Geometry);
                }
                if m > MAX_VALUES {
                    return Err(VqPlanError::AddressLimit);
                }
                let packed = packed_bytes(m, group);
                let dense = u64::from(m) * 4;
                match op {
                    VqOp::Encode => ([dense, book, 0, 0, packed, 4], (packed / 4) as u32),
                    VqOp::Decode => ([packed, book, 0, 0, dense, 4], m),
                    VqOp::AdamW => {
                        product(m, 3)?;
                        ([4 * packed, 4 * book, 7 * 4, 0, 3 * dense, 4], m)
                    }
                    VqOp::Matmul => unreachable!(),
                }
            }
            VqOp::Matmul => {
                if m > MAX_DIM || n > MAX_DIM || k > MAX_DIM {
                    return Err(VqPlanError::Geometry);
                }
                let a = product(m, k)?;
                let b = product(k, n)?;
                let c = product(m, n)?;
                if u64::from(c) * u64::from(k) > MAX_MATMUL_MACS {
                    return Err(VqPlanError::WorkLimit);
                }
                (
                    [
                        packed_bytes(a, group),
                        book,
                        packed_bytes(b, group),
                        book,
                        u64::from(c) * 4,
                        4,
                    ],
                    c,
                )
            }
        };
        let groups = work_items.div_ceil(WORKGROUP_SIZE);
        let x = groups.min(GRID_X);
        let dispatch = [x, groups.div_ceil(x), 1];
        Ok(Self {
            words,
            op,
            required_bytes,
            work_items,
            dispatch,
        })
    }

    /// Allocation ids are obtained from the device registry under its lock.
    /// Read slots may share each other; both write slots must be distinct
    /// from all read slots, including the ones this operation does not read.
    pub fn validate_buffers(
        &self,
        capacities: [u64; OPERANDS],
        ids: [u64; OPERANDS],
    ) -> Result<(), VqPlanError> {
        if ids.contains(&0) {
            return Err(VqPlanError::NullIdentity);
        }
        for (slot, (&actual, &required)) in capacities.iter().zip(&self.required_bytes).enumerate()
        {
            if actual < required {
                return Err(VqPlanError::Capacity(slot));
            }
        }
        if ids[..4].contains(&ids[4]) {
            return Err(VqPlanError::OutputAlias);
        }
        if ids[..5].contains(&ids[5]) {
            return Err(VqPlanError::StatusAlias);
        }
        Ok(())
    }

    pub fn validate_dispatch(&self, limits: [u32; 3]) -> Result<(), VqPlanError> {
        if self
            .dispatch
            .iter()
            .zip(limits)
            .any(|(&count, limit)| count == 0 || count > limit)
        {
            Err(VqPlanError::Dispatch)
        } else {
            Ok(())
        }
    }

    pub fn op(&self) -> VqOp {
        self.op
    }
    pub fn required_bytes(&self) -> [u64; OPERANDS] {
        self.required_bytes
    }
    pub fn work_items(&self) -> u32 {
        self.work_items
    }
    pub fn dispatch_groups(&self) -> [u32; 3] {
        self.dispatch
    }
    pub fn push_constants(&self) -> [u8; PUSH_BYTES] {
        let mut block = [0u8; PUSH_BYTES];
        for (chunk, word) in block.chunks_exact_mut(4).zip(self.words) {
            chunk.copy_from_slice(&word.to_le_bytes());
        }
        block
    }
}
