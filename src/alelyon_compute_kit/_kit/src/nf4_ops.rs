//! Pure validation and dispatch planning for nf4-decode/v1 (2026-09-30):
//! bitsandbytes' NF4 with double quantisation (QLoRA's 4-bit NormalFloat,
//! `research/subbit_expert/moe/nf4.py`), decoded from its stored form by
//! `kernels/nf4_decode.comp`, whose header states the law.
//!
//! The stored form: the packed 4-bit indices (two a byte, the first value in
//! the high nibble), one 8-bit absmax index a block of [`BLOCK`] values, one
//! FP32 `absmax2` a nested block of [`NESTED`] blocks, one FP32 offset a
//! member, and a table holding the 16 code values then the 256 map values.
//! Members (weights quantized alone and stored back to back) start on whole
//! nested blocks. The call decodes either the whole source, member by member,
//! or a selection of members of one size (`ids`), output member `j` being
//! member `ids[j]`.
//!
//! One invocation decodes one packed word of 8 values under one absmax. These
//! bounds limit this implementation's contract; they are not limits of Vulkan
//! or of any selected GPU.
//!
//! nf4-matmul/v1 (2026-09-30, [`Nf4MatmulPlan`]) is the kit's own
//! cooperative-matrix product with B read in that stored form:
//! `kernels/coopmat_bf16_tiled_v4.comp` built with `-DNF4_B=1`, whose staging
//! decodes each group of eight weights into exactly nf4-decode/v1's 16-bit words
//! and changes nothing else, so each product is, bit for bit, nf4-decode/v1
//! followed by the kit's dense product. A call is GROUPED: z is a routed expert
//! with its own rows of A and C and its own weight, from a table of three words
//! a group (first row, rows, weight base) the caller writes and the kernel
//! guards. Since 2026-09-30 its staging caches each slot's member and unrolls a
//! whole group of eight, and its fp16 modules come in two builds with the same
//! words wherever no NaN is involved: convert/v1's integer rounding, and the
//! device's own conversion under declared float controls (NF4_HW).
//!
//! nf4-gemv/v1 (2026-09-30, [`Nf4GemvPlan`], `kernels/nf4_gemv.comp`) is the
//! same product at one to 16 rows a group, as decoding runs it: one workgroup
//! of 64 invocations a column of a group, the weight decoded to its 16-bit
//! value and each product exact in f32, summed in a documented order (each
//! invocation's words ascending, then a fixed pairwise tree over the 64). The
//! order is the kernel header's and a host reproduces it word for word; it is
//! not the cooperative-matrix product's. Its fp16 modules come in two builds
//! with the same words wherever no NaN is involved: convert/v1's integer
//! conversions, and the device's own under declared float controls (F16_HW,
//! 2026-09-30), which a device reporting `shaderFloat16` and the controls takes.

use std::fmt;

pub const ABI: &str = "nf4-decode/v1";
/// Values sharing one absmax (bitsandbytes' and QLoRA's block).
pub const BLOCK: u32 = 64;
/// Blocks sharing one `absmax2` (bitsandbytes' nested block).
pub const NESTED: u32 = 256;
/// Values under one `absmax2`: a member starts on a multiple of this.
pub const NESTED_VALUES: u32 = BLOCK * NESTED;
/// The table's entries: the 16 code values, then the 256 map values.
pub const TABLE_ENTRIES: usize = 16 + 256;
pub const WORKGROUP: u32 = 256;
pub const VALUES_PER_WORD: u32 = 8;
/// Values one call decodes at most, so a value index stays below 2^31.
pub const MAX_VALUES: u32 = 1 << 31;
/// Vulkan's guaranteed minimum for one dispatch dimension.
pub const MAX_GROUPS_X: u32 = 65_535;
/// n, words, members, selected, member_values, src_base, src_limit.
pub const PUSH_BYTES: usize = 28;

/// The output's storage: FP32, bf16 or fp16.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum Nf4Storage {
    F32 = 0,
    Bf16 = 1,
    F16 = 2,
}

impl Nf4Storage {
    pub fn bytes(self) -> u64 {
        match self {
            Self::F32 => 4,
            Self::Bf16 | Self::F16 => 2,
        }
    }
}

impl TryFrom<u32> for Nf4Storage {
    type Error = Nf4PlanError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::F32),
            1 => Ok(Self::Bf16),
            2 => Ok(Self::F16),
            _ => Err(Nf4PlanError::StorageOutOfRange),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Nf4PlanError {
    ZeroValues,
    ValueLimitExceeded,
    StorageOutOfRange,
    MembersOutOfRange,
    SelectionOutOfRange,
    BaseOutOfRange,
    SourceTooSmall,
}

impl fmt::Display for Nf4PlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::ZeroValues => "nf4-zero-values",
            Self::ValueLimitExceeded => "nf4-value-limit-exceeded",
            Self::StorageOutOfRange => "nf4-storage-out-of-range",
            Self::MembersOutOfRange => "nf4-members-out-of-range",
            Self::SelectionOutOfRange => "nf4-selection-out-of-range",
            Self::BaseOutOfRange => "nf4-base-out-of-range",
            Self::SourceTooSmall => "nf4-source-too-small",
        })
    }
}

impl std::error::Error for Nf4PlanError {}

/// Values the three stored buffers cover from their fronts, each read in whole
/// 32-bit words: the packed indices (8 a word), the absmax indices (4 blocks a
/// word) and the `absmax2` values (one nested block each).
pub fn values_covered(packed_bytes: u64, absmax_q_bytes: u64, absmax2_bytes: u64) -> u64 {
    let packed = 8 * (packed_bytes / 4);
    let blocks = 4 * (absmax_q_bytes / 4);
    let nested = absmax2_bytes / 4;
    packed
        .min(blocks * u64::from(BLOCK))
        .min(nested * u64::from(NESTED_VALUES))
}

/// One nf4-decode/v1 dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Nf4DecodePlan {
    n: u32,
    words: u32,
    members: u32,
    selected: u32,
    member_values: u32,
    src_base: u32,
    src_limit: u32,
    storage: Nf4Storage,
    dispatch_groups: [u32; 3],
}

impl Nf4DecodePlan {
    /// `n` values out; `members` entries of starts and offsets; `selected`
    /// entries of ids (0: the whole source), each member then `member_values`
    /// long; the source `src_base` values into buffers that cover `covered`
    /// values ([`values_covered`]).
    ///
    /// Refused by name: no values, more than [`MAX_VALUES`], no members, a
    /// selection whose member size is not a positive multiple of [`BLOCK`] or
    /// whose output is not `selected * member_values`, a base that is not a
    /// whole nested block or lies past the buffers, and a whole-source decode
    /// longer than the buffers hold.
    pub fn new(
        n: u32,
        members: u32,
        selected: u32,
        member_values: u32,
        src_base: u32,
        covered: u64,
        storage: Nf4Storage,
    ) -> Result<Self, Nf4PlanError> {
        if n == 0 {
            return Err(Nf4PlanError::ZeroValues);
        }
        if n > MAX_VALUES {
            return Err(Nf4PlanError::ValueLimitExceeded);
        }
        if members == 0 {
            return Err(Nf4PlanError::MembersOutOfRange);
        }
        let member_values = if selected == 0 { 0 } else { member_values };
        if selected != 0
            && (member_values == 0
                || member_values % BLOCK != 0
                || u64::from(selected) * u64::from(member_values) != u64::from(n))
        {
            return Err(Nf4PlanError::SelectionOutOfRange);
        }
        if src_base % NESTED_VALUES != 0 || u64::from(src_base) >= covered {
            return Err(Nf4PlanError::BaseOutOfRange);
        }
        let src_limit = (covered - u64::from(src_base)).min(u64::from(u32::MAX)) as u32;
        if selected == 0 && n > src_limit {
            return Err(Nf4PlanError::SourceTooSmall);
        }
        let words = n.div_ceil(VALUES_PER_WORD);
        let groups = words.div_ceil(WORKGROUP);
        let dispatch_groups = if groups <= MAX_GROUPS_X {
            [groups, 1, 1]
        } else {
            [MAX_GROUPS_X, groups.div_ceil(MAX_GROUPS_X), 1]
        };
        Ok(Self {
            n,
            words,
            members,
            selected,
            member_values,
            src_base,
            src_limit,
            storage,
            dispatch_groups,
        })
    }

    pub fn values(&self) -> u32 {
        self.n
    }
    pub fn members(&self) -> u32 {
        self.members
    }
    pub fn selected(&self) -> u32 {
        self.selected
    }
    pub fn storage(&self) -> Nf4Storage {
        self.storage
    }
    pub fn output_bytes(&self) -> u64 {
        u64::from(self.n) * self.storage.bytes()
    }
    pub fn dispatch_groups(&self) -> [u32; 3] {
        self.dispatch_groups
    }

    pub fn push_constants(&self) -> [u8; PUSH_BYTES] {
        let mut bytes = [0u8; PUSH_BYTES];
        let words = [
            self.n,
            self.words,
            self.members,
            self.selected,
            self.member_values,
            self.src_base,
            self.src_limit,
        ];
        for (index, word) in words.iter().enumerate() {
            bytes[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }
}

/// nf4-matmul/v1's name and push block: the coopmat modules' ten words (M as
/// the largest group's rows, N, K, four zero z words, lda, ldb 0), then the
/// source's base, its member count, the rows of A and C, and the source's
/// values past its base.
pub const MATMUL_ABI: &str = "nf4-matmul/v1";
pub const MATMUL_PUSH_BYTES: usize = 56;
/// The coopmat modules' output tile, rows and columns.
pub const MATMUL_TILE: u32 = 128;
/// Words a group takes in the groups table: its first row, its rows, and its
/// weight's first value in the source.
pub const GROUP_WORDS: u32 = 3;
/// Vulkan's guaranteed minimum for a z grid, and for each other dimension.
pub const MAX_GRID: u32 = 65_535;

/// The 16-bit element type of A and of the decoded weight: bf16 or fp16.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum Nf4Elem {
    Bf16 = 0,
    F16 = 1,
}

impl TryFrom<u32> for Nf4Elem {
    type Error = Nf4MatmulError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Bf16),
            1 => Ok(Self::F16),
            _ => Err(Nf4MatmulError::ElemOutOfRange),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Nf4MatmulError {
    ZeroExtent,
    GroupsOutOfRange,
    RowsOutOfRange,
    RowNotWholeWords,
    PitchBelowWidth,
    MembersOutOfRange,
    BaseOutOfRange,
    SourceTooSmall,
    AddressLimitExceeded,
    ElemOutOfRange,
    GemvRowsAbove16,
    PitchNotWholeWords,
}

impl fmt::Display for Nf4MatmulError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::ZeroExtent => "nf4-matmul-zero-extent",
            Self::GroupsOutOfRange => "nf4-matmul-groups-out-of-range",
            Self::RowsOutOfRange => "nf4-matmul-rows-out-of-range",
            Self::RowNotWholeWords => "nf4-matmul-row-not-whole-words",
            Self::PitchBelowWidth => "nf4-matmul-pitch-below-width",
            Self::MembersOutOfRange => "nf4-members-out-of-range",
            Self::BaseOutOfRange => "nf4-base-out-of-range",
            Self::SourceTooSmall => "nf4-source-too-small",
            Self::AddressLimitExceeded => "nf4-matmul-address-limit-exceeded",
            Self::ElemOutOfRange => "nf4-matmul-elem-out-of-range",
            Self::GemvRowsAbove16 => "nf4-gemv-rows-above-16",
            Self::PitchNotWholeWords => "nf4-gemv-pitch-not-whole-words",
        })
    }
}

impl std::error::Error for Nf4MatmulError {}

/// One nf4-matmul/v1 dispatch: `rows` rows of A (K wide) and of C (N wide) in
/// `groups` groups, none longer than `max_rows`; each group's weight is N x K
/// stored (`b_t`: C = A . W^T) or K x N (C = A . W), in a source `src_base`
/// values into its buffers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Nf4MatmulPlan {
    rows: u32,
    max_rows: u32,
    n: u32,
    k: u32,
    b_t: bool,
    lda: u32,
    groups: u32,
    members: u32,
    src_base: u32,
    src_limit: u32,
    elem: Nf4Elem,
}

impl Nf4MatmulPlan {
    /// `lda` is A's row pitch in elements, 0 for K; `covered` is what the three
    /// stored buffers cover ([`values_covered`]); `members` the entries of the
    /// starts and offsets the kernel reads.
    ///
    /// Refused by name: a zero extent; no groups or more than [`MAX_GRID`]; a
    /// largest group longer than the rows; a weight row (K stored, or N) that is
    /// not whole words of 8; a pitch below K; no members; a base off a nested
    /// block or past the buffers; buffers that cannot hold one weight; and an
    /// index of A, C or the source past a u32, or a grid past [`MAX_GRID`].
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        rows: u32,
        max_rows: u32,
        n: u32,
        k: u32,
        b_t: bool,
        lda: u32,
        groups: u32,
        members: u32,
        src_base: u32,
        covered: u64,
        elem: Nf4Elem,
    ) -> Result<Self, Nf4MatmulError> {
        if rows == 0 || max_rows == 0 || n == 0 || k == 0 {
            return Err(Nf4MatmulError::ZeroExtent);
        }
        if groups == 0 || groups > MAX_GRID {
            return Err(Nf4MatmulError::GroupsOutOfRange);
        }
        if max_rows > rows {
            return Err(Nf4MatmulError::RowsOutOfRange);
        }
        let row_len = if b_t { k } else { n };
        if row_len % VALUES_PER_WORD != 0 {
            return Err(Nf4MatmulError::RowNotWholeWords);
        }
        let lda = if lda == 0 { k } else { lda };
        if lda < k {
            return Err(Nf4MatmulError::PitchBelowWidth);
        }
        if members == 0 {
            return Err(Nf4MatmulError::MembersOutOfRange);
        }
        if src_base % NESTED_VALUES != 0 || u64::from(src_base) >= covered {
            return Err(Nf4MatmulError::BaseOutOfRange);
        }
        let weight = u64::from(n) * u64::from(k);
        let past_base = covered - u64::from(src_base);
        if weight > past_base {
            return Err(Nf4MatmulError::SourceTooSmall);
        }
        let limit = u64::from(u32::MAX);
        let a_span = u64::from(rows - 1) * u64::from(lda) + u64::from(k);
        let grid_ok =
            n.div_ceil(MATMUL_TILE) <= MAX_GRID && max_rows.div_ceil(MATMUL_TILE) <= MAX_GRID;
        if a_span > limit || u64::from(rows) * u64::from(n) > limit || weight > limit || !grid_ok {
            return Err(Nf4MatmulError::AddressLimitExceeded);
        }
        // every source index the kernel forms, base plus value, stays a u32
        let src_limit = past_base.min(limit - u64::from(src_base)) as u32;
        Ok(Self {
            rows,
            max_rows,
            n,
            k,
            b_t,
            lda,
            groups,
            members,
            src_base,
            src_limit,
            elem,
        })
    }

    pub fn b_t(&self) -> bool {
        self.b_t
    }
    pub fn elem(&self) -> Nf4Elem {
        self.elem
    }
    pub fn groups(&self) -> u32 {
        self.groups
    }
    pub fn members(&self) -> u32 {
        self.members
    }
    /// Elements of A the call may read: its last row's start plus K.
    pub fn a_elements(&self) -> u64 {
        u64::from(self.rows - 1) * u64::from(self.lda) + u64::from(self.k)
    }
    /// Elements of C (f32) the call may write.
    pub fn c_elements(&self) -> u64 {
        u64::from(self.rows) * u64::from(self.n)
    }
    /// Values one group's weight holds, N x K.
    pub fn weight_values(&self) -> u64 {
        u64::from(self.n) * u64::from(self.k)
    }
    pub fn dispatch_groups(&self) -> [u32; 3] {
        [
            self.n.div_ceil(MATMUL_TILE),
            self.max_rows.div_ceil(MATMUL_TILE),
            self.groups,
        ]
    }

    pub fn push_constants(&self) -> [u8; MATMUL_PUSH_BYTES] {
        let mut bytes = [0u8; MATMUL_PUSH_BYTES];
        let words = [
            self.max_rows,
            self.n,
            self.k,
            0,
            0,
            0,
            0,
            0,
            self.lda,
            0,
            self.src_base,
            self.members,
            self.rows,
            self.src_limit,
        ];
        for (index, word) in words.iter().enumerate() {
            bytes[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }
}

/// nf4-gemv/v1's name and push block: n, k, lda, rows, src_base, src_limit,
/// members.
pub const GEMV_ABI: &str = "nf4-gemv/v1";
pub const GEMV_PUSH_BYTES: usize = 28;
/// The most rows a group may have: the kernel's accumulators a row.
pub const GEMV_MAX_ROWS: u32 = 16;
/// Invocations a workgroup, one column of one group: the tree's width.
pub const GEMV_LANES: u32 = 64;
/// The rows a group the kernel's modules hold (2026-09-30): a call runs the
/// smallest that holds its largest group. The words do not depend on the module.
pub const GEMV_ROW_CAPS: [u32; 3] = [1, 4, 16];

/// The module (an index into [`GEMV_ROW_CAPS`]) a call whose largest group has
/// `max_rows` rows runs, or None when no module holds it (0, or more than 16).
pub fn gemv_module(max_rows: u32) -> Option<usize> {
    if max_rows == 0 {
        return None;
    }
    GEMV_ROW_CAPS.iter().position(|&cap| max_rows <= cap)
}

/// One nf4-gemv/v1 dispatch: `rows` rows of A (K wide, pitch `lda`) and of C
/// (N wide) in `groups` groups of at most [`GEMV_MAX_ROWS`] rows; each group's
/// weight is N x K stored (C = A . W^T), in a source `src_base` values into its
/// buffers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Nf4GemvPlan {
    rows: u32,
    max_rows: u32,
    n: u32,
    k: u32,
    lda: u32,
    groups: u32,
    members: u32,
    src_base: u32,
    src_limit: u32,
    elem: Nf4Elem,
}

impl Nf4GemvPlan {
    /// Refused by name: nf4-matmul/v1's refusals for its forward form, a
    /// largest group of more than [`GEMV_MAX_ROWS`] rows, and a row pitch of A
    /// that is not whole words of 8 (the kernel reads A as uvec4).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        rows: u32,
        max_rows: u32,
        n: u32,
        k: u32,
        lda: u32,
        groups: u32,
        members: u32,
        src_base: u32,
        covered: u64,
        elem: Nf4Elem,
    ) -> Result<Self, Nf4MatmulError> {
        if rows == 0 || max_rows == 0 || n == 0 || k == 0 {
            return Err(Nf4MatmulError::ZeroExtent);
        }
        if groups == 0 || groups > MAX_GRID {
            return Err(Nf4MatmulError::GroupsOutOfRange);
        }
        if max_rows > rows {
            return Err(Nf4MatmulError::RowsOutOfRange);
        }
        if max_rows > GEMV_MAX_ROWS {
            return Err(Nf4MatmulError::GemvRowsAbove16);
        }
        if k % VALUES_PER_WORD != 0 {
            return Err(Nf4MatmulError::RowNotWholeWords);
        }
        let lda = if lda == 0 { k } else { lda };
        if lda < k {
            return Err(Nf4MatmulError::PitchBelowWidth);
        }
        if lda % VALUES_PER_WORD != 0 {
            return Err(Nf4MatmulError::PitchNotWholeWords);
        }
        if members == 0 {
            return Err(Nf4MatmulError::MembersOutOfRange);
        }
        if src_base % NESTED_VALUES != 0 || u64::from(src_base) >= covered {
            return Err(Nf4MatmulError::BaseOutOfRange);
        }
        let weight = u64::from(n) * u64::from(k);
        let past_base = covered - u64::from(src_base);
        if weight > past_base {
            return Err(Nf4MatmulError::SourceTooSmall);
        }
        let limit = u64::from(u32::MAX);
        let a_span = u64::from(rows - 1) * u64::from(lda) + u64::from(k);
        if a_span > limit || u64::from(rows) * u64::from(n) > limit || weight > limit {
            return Err(Nf4MatmulError::AddressLimitExceeded);
        }
        let src_limit = past_base.min(limit - u64::from(src_base)) as u32;
        Ok(Self {
            rows,
            max_rows,
            n,
            k,
            lda,
            groups,
            members,
            src_base,
            src_limit,
            elem,
        })
    }

    pub fn elem(&self) -> Nf4Elem {
        self.elem
    }
    /// The module this call runs: the smallest of [`GEMV_ROW_CAPS`] that holds
    /// its largest group.
    pub fn module(&self) -> usize {
        gemv_module(self.max_rows).expect("the plan holds at most 16 rows a group")
    }
    /// Elements of A the call may read.
    pub fn a_elements(&self) -> u64 {
        u64::from(self.rows - 1) * u64::from(self.lda) + u64::from(self.k)
    }
    /// Elements of C (f32) the call may write.
    pub fn c_elements(&self) -> u64 {
        u64::from(self.rows) * u64::from(self.n)
    }
    /// One workgroup a column (folded past [`MAX_GRID`] into y) a group.
    pub fn dispatch_groups(&self) -> [u32; 3] {
        [self.n.min(MAX_GRID), self.n.div_ceil(MAX_GRID), self.groups]
    }

    pub fn push_constants(&self) -> [u8; GEMV_PUSH_BYTES] {
        let mut bytes = [0u8; GEMV_PUSH_BYTES];
        let words = [
            self.n,
            self.k,
            self.lda,
            self.rows,
            self.src_base,
            self.src_limit,
            self.members,
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
    fn the_plan_refuses_by_name_and_carries_its_push_block() {
        let f16 = Nf4Storage::F16;
        let covered = 1u64 << 24;
        let named = |r: Result<Nf4DecodePlan, Nf4PlanError>| r.unwrap_err().to_string();
        assert_eq!(
            named(Nf4DecodePlan::new(0, 1, 0, 0, 0, covered, f16)),
            "nf4-zero-values"
        );
        assert_eq!(
            named(Nf4DecodePlan::new(
                MAX_VALUES + 1,
                1,
                0,
                0,
                0,
                u64::MAX,
                f16
            )),
            "nf4-value-limit-exceeded"
        );
        assert_eq!(
            named(Nf4DecodePlan::new(64, 0, 0, 0, 0, covered, f16)),
            "nf4-members-out-of-range"
        );
        assert_eq!(
            named(Nf4DecodePlan::new(96, 4, 1, 96, 0, covered, f16)),
            "nf4-selection-out-of-range",
            "a member size off the block"
        );
        assert_eq!(
            named(Nf4DecodePlan::new(192, 4, 2, 64, 0, covered, f16)),
            "nf4-selection-out-of-range",
            "an output that is not the selection's"
        );
        assert_eq!(
            named(Nf4DecodePlan::new(64, 1, 0, 0, 64, covered, f16)),
            "nf4-base-out-of-range",
            "a base off the nested block"
        );
        assert_eq!(
            named(Nf4DecodePlan::new(
                64,
                1,
                0,
                0,
                NESTED_VALUES,
                u64::from(NESTED_VALUES),
                f16
            )),
            "nf4-base-out-of-range",
            "a base past the buffers"
        );
        assert_eq!(
            named(Nf4DecodePlan::new(
                NESTED_VALUES + 8,
                1,
                0,
                0,
                0,
                u64::from(NESTED_VALUES),
                f16
            )),
            "nf4-source-too-small"
        );
        assert_eq!(
            Nf4Storage::try_from(3).unwrap_err().to_string(),
            "nf4-storage-out-of-range"
        );
        // 12 of 128 experts selected, each 2,048 x 768, from the second nested block on
        let ev = 2048 * 768;
        let plan = Nf4DecodePlan::new(
            12 * ev,
            128,
            12,
            ev,
            NESTED_VALUES,
            128 * u64::from(ev) + 16_384,
            f16,
        )
        .unwrap();
        assert_eq!(plan.output_bytes(), 2 * 12 * u64::from(ev));
        assert_eq!(plan.dispatch_groups(), [9_216, 1, 1]);
        let push = plan.push_constants();
        let word = |i: usize| u32::from_le_bytes(push[i * 4..i * 4 + 4].try_into().unwrap());
        assert_eq!(
            (0..7).map(word).collect::<Vec<_>>(),
            vec![12 * ev, 12 * ev / 8, 128, 12, ev, NESTED_VALUES, 128 * ev]
        );
        // the whole source ignores a member size; an odd count reads a partial last word
        let whole = Nf4DecodePlan::new(13, 1, 0, 777, 0, 16, Nf4Storage::F32).unwrap();
        let push = whole.push_constants();
        assert_eq!(u32::from_le_bytes(push[16..20].try_into().unwrap()), 0);
        assert_eq!(u32::from_le_bytes(push[4..8].try_into().unwrap()), 2);
    }

    #[test]
    fn the_buffers_cover_what_their_whole_words_hold_and_a_long_decode_folds() {
        // 1,024 packed bytes hold 2,048 values; 30 absmax bytes, 7 whole words, 28 blocks; one absmax2
        assert_eq!(values_covered(1024, 30, 4), 28 * 64);
        assert_eq!(values_covered(1024, 1024, 4), 2048);
        assert_eq!(values_covered(1 << 20, 1 << 20, 0), 0);
        let plan = Nf4DecodePlan::new(
            MAX_VALUES,
            1,
            0,
            0,
            0,
            u64::from(MAX_VALUES),
            Nf4Storage::Bf16,
        )
        .unwrap();
        let [x, y, z] = plan.dispatch_groups();
        assert_eq!((x, z), (MAX_GROUPS_X, 1));
        assert!(
            u64::from(x) * u64::from(y) * u64::from(WORKGROUP)
                >= u64::from(MAX_VALUES / VALUES_PER_WORD)
        );
    }

    #[test]
    fn the_matmul_plan_refuses_by_name_and_carries_its_push_block() {
        let f16 = Nf4Elem::F16;
        let nv = u64::from(NESTED_VALUES);
        // 128 experts' stacked gate and up weights, 1,536 x 2,048 each, from the group's front
        let covered = 128 * 1536 * 2048;
        let named = |r: Result<Nf4MatmulPlan, Nf4MatmulError>| r.unwrap_err().to_string();
        let ok = |rows, max_rows, n, k, b_t, lda, groups, base: u32, cov: u64| {
            Nf4MatmulPlan::new(rows, max_rows, n, k, b_t, lda, groups, 256, base, cov, f16)
        };
        assert_eq!(
            named(ok(0, 1, 1536, 2048, true, 0, 1, 0, covered)),
            "nf4-matmul-zero-extent"
        );
        assert_eq!(
            named(ok(8, 8, 1536, 2048, true, 0, 0, 0, covered)),
            "nf4-matmul-groups-out-of-range"
        );
        assert_eq!(
            named(ok(8, 8, 1536, 2048, true, 0, 65_536, 0, covered)),
            "nf4-matmul-groups-out-of-range"
        );
        assert_eq!(
            named(ok(8, 9, 1536, 2048, true, 0, 1, 0, covered)),
            "nf4-matmul-rows-out-of-range"
        );
        assert_eq!(
            named(ok(8, 8, 1536, 2044, true, 0, 1, 0, covered)),
            "nf4-matmul-row-not-whole-words",
            "rows @ W^T reads W's rows of K"
        );
        assert_eq!(
            named(ok(8, 8, 2044, 1536, false, 0, 1, 0, covered)),
            "nf4-matmul-row-not-whole-words",
            "rows @ W reads W's rows of N"
        );
        assert!(
            ok(8, 8, 2044, 1536, true, 0, 1, 0, covered).is_ok(),
            "N itself may be any width"
        );
        assert_eq!(
            named(ok(8, 8, 1536, 2048, true, 2040, 1, 0, covered)),
            "nf4-matmul-pitch-below-width"
        );
        assert_eq!(
            named(Nf4MatmulPlan::new(
                8, 8, 1536, 2048, true, 0, 1, 0, 0, covered, f16
            )),
            "nf4-members-out-of-range"
        );
        assert_eq!(
            named(ok(8, 8, 1536, 2048, true, 0, 1, 64, covered)),
            "nf4-base-out-of-range"
        );
        assert_eq!(
            named(ok(8, 8, 1536, 2048, true, 0, 1, 0, 1536 * 2048 - 8)),
            "nf4-source-too-small"
        );
        assert_eq!(
            named(ok(1 << 20, 1 << 20, 1 << 13, 64, true, 0, 1, 0, covered)),
            "nf4-matmul-address-limit-exceeded",
            "C's 2^33 elements"
        );
        assert_eq!(
            Nf4Elem::try_from(2).unwrap_err().to_string(),
            "nf4-matmul-elem-out-of-range"
        );
        // A''s grouped gate_up call: 4,096 routed rows over 60 experts, the largest 131,
        // the weight base one nested block in
        let plan = ok(
            4096,
            131,
            1536,
            2048,
            true,
            0,
            60,
            NESTED_VALUES,
            covered + nv,
        )
        .unwrap();
        assert_eq!(plan.dispatch_groups(), [12, 2, 60]);
        assert_eq!(
            (plan.a_elements(), plan.c_elements()),
            (4095 * 2048 + 2048, 4096 * 1536)
        );
        assert_eq!(plan.weight_values(), 1536 * 2048);
        let push = plan.push_constants();
        let word = |i: usize| u32::from_le_bytes(push[i * 4..i * 4 + 4].try_into().unwrap());
        assert_eq!(
            (0..14).map(word).collect::<Vec<_>>(),
            vec![
                131,
                1536,
                2048,
                0,
                0,
                0,
                0,
                0,
                2048,
                0,
                NESTED_VALUES,
                256,
                4096,
                covered as u32
            ]
        );
        // the backward's rows @ W over the down weights, 2,048 x 768 stored: N 768 wide
        let back = ok(512, 512, 768, 2048, false, 0, 8, 0, covered).unwrap();
        assert!(!back.b_t());
        assert_eq!(back.dispatch_groups(), [6, 4, 8]);
    }

    #[test]
    fn the_gemv_plan_refuses_by_name_folds_wide_weights_and_carries_its_push_block() {
        let bf16 = Nf4Elem::Bf16;
        let covered = 128 * 1536 * 2048;
        let named = |r: Result<Nf4GemvPlan, Nf4MatmulError>| r.unwrap_err().to_string();
        let plan = |rows, max_rows, n, k, lda| {
            Nf4GemvPlan::new(rows, max_rows, n, k, lda, 8, 256, 0, covered, bf16)
        };
        assert_eq!(named(plan(17, 17, 1536, 2048, 0)), "nf4-gemv-rows-above-16");
        assert_eq!(
            named(plan(8, 8, 1536, 2044, 0)),
            "nf4-matmul-row-not-whole-words"
        );
        assert_eq!(
            named(plan(8, 8, 1536, 2048, 2052)),
            "nf4-gemv-pitch-not-whole-words"
        );
        assert_eq!(
            named(plan(8, 8, 1536, 2048, 2040)),
            "nf4-matmul-pitch-below-width"
        );
        assert_eq!(
            named(plan(8, 9, 1536, 2048, 0)),
            "nf4-matmul-rows-out-of-range"
        );
        assert_eq!(named(plan(0, 1, 1536, 2048, 0)), "nf4-matmul-zero-extent");
        // decoding one token through 8 routed experts' gate_up weights: a workgroup a column a group
        let decode = plan(8, 1, 1536, 2048, 0).unwrap();
        assert_eq!(decode.dispatch_groups(), [1536, 1, 8]);
        let push = decode.push_constants();
        let word = |i: usize| u32::from_le_bytes(push[i * 4..i * 4 + 4].try_into().unwrap());
        assert_eq!(
            (0..7).map(word).collect::<Vec<_>>(),
            vec![1536, 2048, 2048, 8, 0, covered as u32, 256]
        );
        // a weight wider than one grid dimension folds into y
        let wide = Nf4GemvPlan::new(1, 1, 70_000, 64, 0, 1, 1, 0, 70_000 * 64, bf16).unwrap();
        assert_eq!(wide.dispatch_groups(), [65_535, 2, 1]);
        // the smallest module that holds the call's largest group
        assert_eq!(decode.module(), 0);
        assert_eq!(plan(8, 4, 1536, 2048, 0).unwrap().module(), 1);
        assert_eq!(plan(8, 5, 1536, 2048, 0).unwrap().module(), 2);
        assert_eq!(plan(16, 16, 1536, 2048, 0).unwrap().module(), 2);
        assert_eq!((gemv_module(0), gemv_module(17)), (None, None));
    }
}
