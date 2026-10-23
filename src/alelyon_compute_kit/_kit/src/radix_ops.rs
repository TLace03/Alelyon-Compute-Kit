//! Pure planning for `radix/v1` (intwise schema 6, 2026-09-29): the stable
//! ranks of long rows -- the intwise family's rank_asc and rank_desc -- by a
//! least-significant-digit radix sort. `kernels/radix.comp` is the shader; its
//! header states the order, the passes and the scratch layout.
//!
//! [`crate::intwise_ops::IntwisePlan`] validates the request (codes, capacity,
//! overlap) and asks for a radix plan for a rank row longer than
//! [`crate::intwise_ops::RADIX_MIN_ROW`] whose output is int64 or int32. This
//! module lays out the passes: KEYS, then for every digit pass a histogram, a
//! scan and a scatter, then RANKS. A key is as wide as its storage (1, 2, 4 or
//! 8 bytes: that many value passes), and a row index takes as many more
//! passes as its bytes need when there is more than one row.
//!
//! This module has no native `Context` or `Buffer` dependency; the native
//! entry allocates the scratch ([`RadixPlan::scratch_words`]) and records
//! [`RadixPlan::passes`] in order.

use crate::convert_ops::{Code, View};

pub const ABI: &str = "radix/v1";
pub const MAX_NDIM: usize = 4;
/// `x`, the scratch, `out`, in binding order.
pub const BINDINGS: u32 = 3;
pub const WORKGROUP_INVOCATIONS: u32 = 64;
/// Every device offers at least this many workgroups on each dispatch axis.
pub const MAX_GROUPS_X: u32 = 65_535;
/// Elements a block: a histogram's and a scatter's workgroup, 16 rounds of 64.
pub const BLOCK: u64 = 1024;
/// The values of one 8-bit digit.
pub const DIGITS: u64 = 256;
/// Every word index the shader forms is below this.
pub const WORD_LIMIT: u64 = 1 << 31;
/// The most elements a plan takes: its scratch, 6n + 256 B + 256 words for
/// B = ceil(n / BLOCK), stays below [`WORD_LIMIT`] (the shader's own guard,
/// `n <= (2^31 - 1 - 256) / 7`).
pub const MAX_ELEMENTS: u64 = (WORD_LIMIT - 1 - DIGITS) / 7;
pub const PUSH_WORDS: usize = 20;
pub const PUSH_BYTES: usize = PUSH_WORDS * 4;
/// Push word 5, bit 8: the rank is descending (the keys complemented).
pub const FLAG_DESCENDING: u32 = 0x100;

/// Push word 0: what a dispatch does (radix.comp's header).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RadixMode {
    Keys = 0,
    Histogram = 1,
    Scan = 2,
    Scatter = 3,
    Ranks = 4,
}

/// One dispatch: its mode, push block and workgroups.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RadixPass {
    mode: RadixMode,
    words: [u32; PUSH_WORDS],
    groups: [u32; 3],
}

impl RadixPass {
    pub fn mode(&self) -> RadixMode {
        self.mode
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

/// The key's width in bytes, and so the number of value passes: a float or
/// integer key is as wide as its storage (radix.comp's `key_bytes`).
pub fn key_bytes(code: Code) -> u32 {
    match code {
        Code::F64 | Code::I64 => 8,
        Code::F32 | Code::I32 => 4,
        Code::Bf16 | Code::F16 | Code::I16 => 2,
        Code::I8 | Code::U8 | Code::Bool => 1,
    }
}

/// The digit passes a row index needs: none for one row, else one a byte of
/// the largest row index.
pub fn row_passes(rows: u64) -> u32 {
    if rows <= 1 {
        return 0;
    }
    let bits = 64 - (rows - 1).leading_zeros();
    bits.div_ceil(8)
}

fn groups_for(workgroups: u64) -> [u32; 3] {
    let x = workgroups.min(u64::from(MAX_GROUPS_X)).max(1);
    let y = workgroups.div_ceil(x).max(1);
    [x as u32, y as u32, 1]
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RadixPlan {
    n: u64,
    blocks: u64,
    value_passes: u32,
    row_passes: u32,
    /// The push block every pass shares; each pass sets its mode and pass.
    base: [u32; PUSH_WORDS],
}

impl RadixPlan {
    /// A plan for the ranks of `x` (storage `x_code`) along the LAST of the
    /// padded `extents`, written to `out` as `out_code` (int64 or int32).
    /// Views are in elements of their own storage, strides indexed like the
    /// padded extents, a stride on an extent-1 dimension ignored. `None` for
    /// an output code other than int64 or int32, or more than
    /// [`MAX_ELEMENTS`] elements; the caller has checked everything else.
    pub fn new(
        extents: [u64; MAX_NDIM],
        x_code: Code,
        x: View,
        out_code: Code,
        out: View,
        descending: bool,
    ) -> Option<Self> {
        if !matches!(out_code, Code::I64 | Code::I32) {
            return None;
        }
        let n = extents
            .iter()
            .try_fold(1u64, |acc, &extent| acc.checked_mul(extent))?;
        let row = extents[MAX_NDIM - 1];
        if n == 0 || row == 0 || n > MAX_ELEMENTS {
            return None;
        }
        let blocks = n.div_ceil(BLOCK);
        let value_passes = key_bytes(x_code);
        let row_passes = row_passes(n / row);
        let stride = |view: &View, k: usize| {
            if extents[k] > 1 {
                view.strides[k] as u32
            } else {
                0
            }
        };
        let mut base = [0u32; PUSH_WORDS];
        base[1] = n as u32;
        base[2] = row as u32;
        base[3] = blocks as u32;
        base[5] = x_code as u32
            | (out_code as u32) << 4
            | if descending { FLAG_DESCENDING } else { 0 }
            | value_passes << 12
            | row_passes << 16;
        for k in 0..MAX_NDIM {
            base[6 + k] = extents[k] as u32;
            base[11 + k] = stride(&x, k);
            base[16 + k] = stride(&out, k);
        }
        base[10] = x.offset as u32;
        base[15] = out.offset as u32;
        Some(Self {
            n,
            blocks,
            value_passes,
            row_passes,
            base,
        })
    }

    /// The digit passes: the key's bytes, then the row index's.
    pub fn digit_passes(&self) -> u32 {
        self.value_passes + self.row_passes
    }

    /// The scratch the native entry allocates, in 32-bit words: two halves of
    /// keys (two words an element) and positions, the digit-major histogram
    /// and the digit totals (radix.comp's SCRATCH).
    pub fn scratch_words(&self) -> u64 {
        6 * self.n + DIGITS * self.blocks + DIGITS
    }

    fn pass(&self, mode: RadixMode, pass: u32, workgroups: u64) -> RadixPass {
        let mut words = self.base;
        words[0] = mode as u32;
        words[4] = pass;
        RadixPass {
            mode,
            words,
            groups: groups_for(workgroups),
        }
    }

    /// The digit passes alone, in order (a histogram, a scan and a scatter
    /// for each): a caller that writes the keys itself (the index family's
    /// grouped scatter-add) and reads the sorted pairs itself records these.
    pub fn digit_pass_list(&self) -> Vec<RadixPass> {
        let mut out = Vec::new();
        for j in 0..self.digit_passes() {
            out.push(self.pass(RadixMode::Histogram, j, self.blocks));
            out.push(self.pass(RadixMode::Scan, j, DIGITS));
            out.push(self.pass(RadixMode::Scatter, j, self.blocks));
        }
        out
    }

    /// Every dispatch, in recording order: KEYS; a histogram, a scan and a
    /// scatter for each digit pass; RANKS (whose pass word is the number of
    /// digit passes, which names the half the last pass wrote).
    pub fn passes(&self) -> Vec<RadixPass> {
        let per_element = self.n.div_ceil(u64::from(WORKGROUP_INVOCATIONS));
        let mut out = vec![self.pass(RadixMode::Keys, 0, per_element)];
        for j in 0..self.digit_passes() {
            out.push(self.pass(RadixMode::Histogram, j, self.blocks));
            out.push(self.pass(RadixMode::Scan, j, DIGITS));
            out.push(self.pass(RadixMode::Scatter, j, self.blocks));
        }
        out.push(self.pass(RadixMode::Ranks, self.digit_passes(), per_element));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A contiguous row: stride 1 along the last dimension.
    fn row_view() -> View {
        View::new(0, [0, 0, 0, 1])
    }

    #[test]
    fn the_moe_routing_argsort_is_eight_digit_passes_of_one_row() {
        // torch.argsort(flat [65536] int64, stable=True): one row, 64-bit keys
        let n = 65_536;
        let plan = RadixPlan::new(
            [1, 1, 1, n],
            Code::I64,
            row_view(),
            Code::I64,
            row_view(),
            false,
        )
        .unwrap();
        assert_eq!(plan.digit_passes(), 8);
        assert_eq!(plan.scratch_words(), 6 * n + 256 * 64 + 256);
        let passes = plan.passes();
        assert_eq!(passes.len(), 1 + 3 * 8 + 1);
        assert_eq!(passes[0].mode(), RadixMode::Keys);
        assert_eq!(passes[0].dispatch_groups(), [1024, 1, 1]);
        assert_eq!(passes[1].mode(), RadixMode::Histogram);
        assert_eq!(
            passes[1].dispatch_groups(),
            [64, 1, 1],
            "one workgroup a block"
        );
        assert_eq!(passes[2].mode(), RadixMode::Scan);
        assert_eq!(
            passes[2].dispatch_groups(),
            [256, 1, 1],
            "one workgroup a digit"
        );
        assert_eq!(passes[3].mode(), RadixMode::Scatter);
        let last = passes.last().unwrap();
        assert_eq!(last.mode(), RadixMode::Ranks);
        assert_eq!(
            last.words()[4],
            8,
            "RANKS reads the half the eighth pass wrote"
        );
        // the digit passes count up; every pass shares the geometry
        for (j, triple) in passes[1..25].chunks(3).enumerate() {
            assert!(triple.iter().all(|p| p.words()[4] == j as u32));
        }
        let codes = passes[0].words()[5];
        assert_eq!(codes & 0xF, Code::I64 as u32);
        assert_eq!((codes >> 4) & 0xF, Code::I64 as u32);
        assert_eq!(codes & FLAG_DESCENDING, 0);
        assert_eq!((codes >> 12) & 0xF, 8);
        assert_eq!((codes >> 16) & 0xF, 0);
        assert_eq!(&passes[0].words()[6..10], &[1, 1, 1, n as u32]);
        assert_eq!(passes[0].words()[14], 1, "the row's stride");
        assert_eq!(
            passes[0].words()[11],
            0,
            "a stride on an extent-1 dimension is ignored"
        );
    }

    #[test]
    fn a_key_is_as_wide_as_its_storage() {
        let widths: Vec<u32> = Code::ALL.iter().map(|&c| key_bytes(c)).collect();
        // f32 bf16 f16 i64 i32 i16 i8 u8 f64 bool
        assert_eq!(widths, vec![4, 2, 2, 8, 4, 2, 1, 1, 8, 1]);
        for code in Code::ALL {
            let plan = RadixPlan::new(
                [1, 1, 1, 5000],
                code,
                row_view(),
                Code::I32,
                row_view(),
                true,
            )
            .unwrap();
            assert_eq!(plan.digit_passes(), key_bytes(code));
            assert_eq!(
                plan.passes()[0].words()[5] & FLAG_DESCENDING,
                FLAG_DESCENDING
            );
        }
    }

    #[test]
    fn several_rows_add_the_row_index_bytes_as_the_last_passes() {
        assert_eq!(row_passes(1), 0);
        assert_eq!(row_passes(2), 1);
        assert_eq!(row_passes(256), 1);
        assert_eq!(row_passes(257), 2);
        assert_eq!(row_passes(65_536), 2);
        assert_eq!(row_passes(65_537), 3);
        // 300 rows of 2,048 f32: four value passes, two row passes
        let x = View::new(0, [0, 0, 2048, 1]);
        let plan = RadixPlan::new([1, 1, 300, 2048], Code::F32, x, Code::I64, x, false).unwrap();
        assert_eq!(plan.digit_passes(), 6);
        assert_eq!((plan.passes()[0].words()[5] >> 16) & 0xF, 2);
    }

    #[test]
    fn a_narrow_output_or_too_many_elements_is_not_planned() {
        for out in [
            Code::I16,
            Code::I8,
            Code::U8,
            Code::Bool,
            Code::F32,
            Code::F64,
        ] {
            assert!(RadixPlan::new(
                [1, 1, 1, 4096],
                Code::F32,
                row_view(),
                out,
                row_view(),
                false
            )
            .is_none());
        }
        let big = MAX_ELEMENTS + 1;
        assert!(RadixPlan::new(
            [1, 1, 1, big],
            Code::I32,
            row_view(),
            Code::I64,
            row_view(),
            false
        )
        .is_none());
        let at = MAX_ELEMENTS;
        let plan = RadixPlan::new(
            [1, 1, 1, at],
            Code::I32,
            row_view(),
            Code::I64,
            row_view(),
            false,
        )
        .unwrap();
        assert!(plan.scratch_words() < WORD_LIMIT);
    }

    #[test]
    fn a_large_row_spreads_its_per_element_passes_over_a_second_grid_axis() {
        let n = 1u64 << 24;
        let plan = RadixPlan::new(
            [1, 1, 1, n],
            Code::F32,
            row_view(),
            Code::I64,
            row_view(),
            false,
        )
        .unwrap();
        let passes = plan.passes();
        let [x, y, z] = passes[0].dispatch_groups();
        assert_eq!(z, 1);
        assert!(x <= MAX_GROUPS_X && y > 1);
        assert!(u64::from(x) * u64::from(y) * 64 >= n);
        let [bx, by, _] = passes[1].dispatch_groups();
        assert!(u64::from(bx) * u64::from(by) >= n / BLOCK);
    }
}
