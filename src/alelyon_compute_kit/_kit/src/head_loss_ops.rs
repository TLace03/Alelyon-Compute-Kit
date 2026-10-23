//! Pure validation and dispatch planning for head-kl-ce-f16/v1 (2026-09-29):
//! the sub-bit student's head loss, its top-k KL against a teacher plus CE, over
//! an fp16 head output `yp` (rows x cols) and an fp32 per-column scale `h`.
//! The logits are z = f32(yp) * h, never materialised.
//!
//! Two kernels, one pass over `yp` each:
//! - `kernels/head_lse_f16.comp` (the forward): per row, the log-sum-exp of z
//!   over every column and over the columns not among the row's top-k;
//! - `kernels/head_grad_f16.comp` (the backward): every column's final
//!   gradient at fp16, the top-k and target columns included, and each row
//!   group's partial of the gradient with respect to `h`.
//!
//! The per-row values between them are the reference's own expressions on
//! (rows, k) values, which the caller computes (research/subbit_expert/moe/
//! student.py `_kl_ce_block_fwd_f` / `_kl_ce_block_bwd_f` are the law).
//!
//! These bounds limit this implementation's contract; they are not limits of
//! Vulkan or of any selected GPU.

use std::fmt;

pub const ABI: &str = "head-kl-ce-f16/v1";
pub const SCHEMA: u32 = 1;
pub const WORKGROUP: u32 = 256;
/// The widest row this contract takes. The forward walks a row in segments of
/// 32,768 columns, each segment's top-k set a 4 KiB bit mask in shared memory,
/// so the width is bounded by the addressing, not by shared memory.
pub const MAX_COLS: u32 = 1 << 24;
/// A top-k wider than a workgroup is refused.
pub const MAX_K: u32 = 256;
/// The backward's column block: 256 lanes, two fp16 columns a lane.
pub const BLOCK_COLS: u32 = 512;
/// The backward's rows a workgroup: its shared mask holds 16 words a row.
pub const MAX_GROUP_ROWS: u32 = 64;
/// Every buffer the kernels address holds at most this many elements, so
/// every element and word offset stays below 2^31.
pub const MAX_ELEMENTS: usize = 1 << 31;
/// The top-k ids are int64, addressed as two words each: rows x k at most 2^28.
pub const MAX_TOP_ENTRIES: usize = 1 << 28;
/// Vulkan's guaranteed minimum for one dispatch dimension.
pub const MAX_GROUPS_X: u32 = 65_535;
/// rows, cols, k, groups_x.
pub const LSE_PUSH_BYTES: usize = 16;
/// rows, cols, k, rows_per_group.
pub const GRAD_PUSH_BYTES: usize = 16;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HeadLossPlanError {
    DimensionOutOfRange,
    OddColumns,
    ColumnsAboveLimit,
    TopKAboveLimit,
    RowsPerGroupOutOfRange,
    ElementLimitExceeded,
    GridAboveLimit,
}

impl fmt::Display for HeadLossPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::DimensionOutOfRange => "head-loss-dimension-out-of-range",
            Self::OddColumns => "head-loss-odd-columns",
            Self::ColumnsAboveLimit => "head-loss-columns-above-limit",
            Self::TopKAboveLimit => "head-loss-top-k-above-limit",
            Self::RowsPerGroupOutOfRange => "head-loss-rows-per-group-out-of-range",
            Self::ElementLimitExceeded => "head-loss-element-limit-exceeded",
            Self::GridAboveLimit => "head-loss-grid-above-limit",
        })
    }
}

impl std::error::Error for HeadLossPlanError {}

fn check_shape(rows: u32, cols: u32, k: u32) -> Result<(), HeadLossPlanError> {
    if rows == 0 || cols == 0 || k == 0 {
        return Err(HeadLossPlanError::DimensionOutOfRange);
    }
    if cols % 2 != 0 {
        return Err(HeadLossPlanError::OddColumns);
    }
    if cols > MAX_COLS {
        return Err(HeadLossPlanError::ColumnsAboveLimit);
    }
    if k > MAX_K {
        return Err(HeadLossPlanError::TopKAboveLimit);
    }
    if rows as usize * cols as usize > MAX_ELEMENTS || rows as usize * k as usize > MAX_TOP_ENTRIES
    {
        return Err(HeadLossPlanError::ElementLimitExceeded);
    }
    Ok(())
}

/// The forward: one workgroup a row.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HeadLsePlan {
    rows: u32,
    cols: u32,
    k: u32,
    dispatch_groups: [u32; 3],
}

impl HeadLsePlan {
    pub fn new(rows: u32, cols: u32, k: u32) -> Result<Self, HeadLossPlanError> {
        check_shape(rows, cols, k)?;
        let x = rows.min(MAX_GROUPS_X);
        let y = rows.div_ceil(x);
        if y > MAX_GROUPS_X {
            return Err(HeadLossPlanError::GridAboveLimit);
        }
        Ok(Self {
            rows,
            cols,
            k,
            dispatch_groups: [x, y, 1],
        })
    }

    pub fn rows(&self) -> u32 {
        self.rows
    }
    pub fn cols(&self) -> u32 {
        self.cols
    }
    pub fn k(&self) -> u32 {
        self.k
    }
    /// fp16 elements of `yp`.
    pub fn yp_elements(&self) -> usize {
        self.rows as usize * self.cols as usize
    }
    /// int64 entries of the top-k ids.
    pub fn top_entries(&self) -> usize {
        self.rows as usize * self.k as usize
    }
    /// f32 elements of the output: every row's two log-sum-exps.
    pub fn out_elements(&self) -> usize {
        2 * self.rows as usize
    }
    pub fn dispatch_groups(&self) -> [u32; 3] {
        self.dispatch_groups
    }

    pub fn push_constants(&self) -> [u8; LSE_PUSH_BYTES] {
        words_le([self.rows, self.cols, self.k, self.dispatch_groups[0]])
    }
}

/// The backward: one workgroup a (512-column block, row group).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HeadGradPlan {
    rows: u32,
    cols: u32,
    k: u32,
    rows_per_group: u32,
    groups: u32,
    dispatch_groups: [u32; 3],
}

impl HeadGradPlan {
    pub fn new(
        rows: u32,
        cols: u32,
        k: u32,
        rows_per_group: u32,
    ) -> Result<Self, HeadLossPlanError> {
        check_shape(rows, cols, k)?;
        if rows_per_group == 0 || rows_per_group > MAX_GROUP_ROWS {
            return Err(HeadLossPlanError::RowsPerGroupOutOfRange);
        }
        let groups = rows.div_ceil(rows_per_group);
        let blocks = cols.div_ceil(BLOCK_COLS);
        if groups > MAX_GROUPS_X || blocks > MAX_GROUPS_X {
            return Err(HeadLossPlanError::GridAboveLimit);
        }
        if groups as usize * cols as usize > MAX_ELEMENTS {
            return Err(HeadLossPlanError::ElementLimitExceeded);
        }
        Ok(Self {
            rows,
            cols,
            k,
            rows_per_group,
            groups,
            dispatch_groups: [blocks, groups, 1],
        })
    }

    pub fn rows(&self) -> u32 {
        self.rows
    }
    pub fn cols(&self) -> u32 {
        self.cols
    }
    pub fn k(&self) -> u32 {
        self.k
    }
    pub fn rows_per_group(&self) -> u32 {
        self.rows_per_group
    }
    /// Row groups, each one row of `partial`.
    pub fn groups(&self) -> u32 {
        self.groups
    }
    pub fn yp_elements(&self) -> usize {
        self.rows as usize * self.cols as usize
    }
    pub fn top_entries(&self) -> usize {
        self.rows as usize * self.k as usize
    }
    /// f32 elements of the row values: lse, c_top, c_nt (rows each), then g_ce.
    pub fn row_value_elements(&self) -> usize {
        3 * self.rows as usize + 1
    }
    /// f32 elements of the d/dh partials: groups x cols.
    pub fn partial_elements(&self) -> usize {
        self.groups as usize * self.cols as usize
    }
    pub fn dispatch_groups(&self) -> [u32; 3] {
        self.dispatch_groups
    }

    pub fn push_constants(&self) -> [u8; GRAD_PUSH_BYTES] {
        words_le([self.rows, self.cols, self.k, self.rows_per_group])
    }
}

fn words_le<const W: usize, const B: usize>(words: [u32; W]) -> [u8; B] {
    let mut bytes = [0u8; B];
    for (index, word) in words.iter().enumerate() {
        bytes[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
    }
    bytes
}

/// fp16 bits -> f32, exact (every binary16 value is a binary32 value).
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = u32::from(h & 0x8000) << 16;
    let e = u32::from((h >> 10) & 0x1F);
    let m = u32::from(h & 0x3FF);
    let bits = if e == 0x1F {
        sign | 0x7F80_0000 | (m << 13)
    } else if e == 0 {
        if m == 0 {
            sign
        } else {
            let t = 31 - m.leading_zeros();
            sign | ((t + 103) << 23) | ((m << (23 - t)) & 0x7F_FFFF)
        }
    } else {
        sign | ((e + 112) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}

/// f32 -> fp16 bits, round to nearest even; NaN -> sign | 0x7E00, as c10 writes it.
pub fn f32_to_f16(v: f32) -> u16 {
    let w = v.to_bits();
    let sign = (w >> 16) & 0x8000;
    let e = (w >> 23) & 0xFF;
    let m = w & 0x7F_FFFF;
    if e == 0xFF {
        return (sign | if m != 0 { 0x7E00 } else { 0x7C00 }) as u16;
    }
    let he = e as i32 - 112;
    if he >= 31 {
        return (sign | 0x7C00) as u16;
    }
    if he <= 0 {
        if he < -10 {
            return sign as u16;
        }
        let mf = m | 0x80_0000;
        let shift = (14 - he) as u32;
        let mut q = mf >> shift;
        let rem = mf & ((1 << shift) - 1);
        let half = 1 << (shift - 1);
        if rem > half || (rem == half && q & 1 != 0) {
            q += 1;
        }
        return (sign | q) as u16;
    }
    let mut h = ((he as u32) << 10) | (m >> 13);
    let rem = m & 0x1FFF;
    if rem > 0x1000 || (rem == 0x1000 && h & 1 != 0) {
        h += 1;
    }
    (sign | h) as u16
}

fn top_set(top: &[i64], row: usize, k: usize, cols: usize) -> Option<Vec<usize>> {
    let ids = &top[row * k..(row + 1) * k];
    ids.iter()
        .map(|&id| {
            if id >= 0 && (id as usize) < cols {
                Some(id as usize)
            } else {
                None
            }
        })
        .collect()
}

fn logsumexp(values: impl Iterator<Item = f64>) -> f64 {
    let v: Vec<f64> = values.collect();
    if v.iter().any(|x| x.is_nan()) {
        return f64::NAN;
    }
    let big = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if big.is_infinite() {
        return big;
    }
    big + v.iter().map(|x| (x - big).exp()).sum::<f64>().ln()
}

/// The forward's law in f64, for tests: `2 rows` values, every row's
/// log-sum-exp over all columns, then over the non-top columns, of
/// z = f32(yp) * h (the f32 product, widened). A row with a top-k id out of
/// range is NaN in both.
pub fn reference_lse(yp: &[u16], h: &[f32], top: &[i64], plan: &HeadLsePlan) -> Vec<f64> {
    let (rows, cols, k) = (plan.rows as usize, plan.cols as usize, plan.k as usize);
    let mut out = vec![0.0f64; 2 * rows];
    for r in 0..rows {
        let z: Vec<f64> = (0..cols)
            .map(|c| f64::from(f16_to_f32(yp[r * cols + c]) * h[c]))
            .collect();
        match top_set(top, r, k, cols) {
            None => {
                out[r] = f64::NAN;
                out[rows + r] = f64::NAN;
            }
            Some(ids) => {
                let mut is_top = vec![false; cols];
                for id in ids {
                    is_top[id] = true;
                }
                out[r] = logsumexp(z.iter().copied());
                out[rows + r] = logsumexp((0..cols).filter(|&c| !is_top[c]).map(|c| z[c]));
            }
        }
    }
    out
}

/// The backward's law in f64, for tests: every element's final gradient
/// before its fp16 rounding (times h, as the kernel stores it), and d/dh
/// summed over every row. `row_values` and `gkp` as the kernel takes them.
pub fn reference_grad(
    yp: &[u16],
    h: &[f32],
    top: &[i64],
    targets: &[i64],
    row_values: &[f32],
    gkp: &[f32],
    plan: &HeadGradPlan,
) -> (Vec<f64>, Vec<f64>) {
    let (rows, cols, k) = (plan.rows as usize, plan.cols as usize, plan.k as usize);
    let g_ce = f64::from(row_values[3 * rows]);
    let mut dyp = vec![0.0f64; rows * cols];
    let mut dh = vec![0.0f64; cols];
    for r in 0..rows {
        let lse = f64::from(row_values[r]);
        let c_top = f64::from(row_values[rows + r]);
        let c_nt = f64::from(row_values[2 * rows + r]);
        // the last of the row's top-k entries naming a column wins, as torch's scatter keeps
        let mut last: Vec<Option<usize>> = vec![None; cols];
        for i in 0..k {
            let id = top[r * k + i];
            if id >= 0 && (id as usize) < cols {
                last[id as usize] = Some(i);
            }
        }
        let target = targets[r];
        for c in 0..cols {
            let y = f64::from(f16_to_f32(yp[r * cols + c]));
            let z = f64::from(f16_to_f32(yp[r * cols + c]) * h[c]);
            let q = (z - lse).exp();
            let mut g = match last[c] {
                Some(i) => q * c_top - f64::from(gkp[r * k + i]),
                None => q * c_nt,
            };
            if target >= 0 && target as usize == c {
                g -= g_ce;
            }
            dyp[r * cols + c] = g * f64::from(h[c]);
            dh[c] += y * g;
        }
    }
    (dyp, dh)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refusals_are_named() {
        let named = |r: Result<HeadLsePlan, HeadLossPlanError>| r.unwrap_err().to_string();
        assert_eq!(
            named(HeadLsePlan::new(0, 16, 4)),
            "head-loss-dimension-out-of-range"
        );
        assert_eq!(
            named(HeadLsePlan::new(4, 0, 4)),
            "head-loss-dimension-out-of-range"
        );
        assert_eq!(
            named(HeadLsePlan::new(4, 16, 0)),
            "head-loss-dimension-out-of-range"
        );
        assert_eq!(named(HeadLsePlan::new(4, 15, 4)), "head-loss-odd-columns");
        assert_eq!(
            named(HeadLsePlan::new(4, MAX_COLS + 2, 4)),
            "head-loss-columns-above-limit"
        );
        assert_eq!(
            named(HeadLsePlan::new(4, 16, MAX_K + 1)),
            "head-loss-top-k-above-limit"
        );
        assert_eq!(
            named(HeadLsePlan::new(200, MAX_COLS, 4)),
            "head-loss-element-limit-exceeded"
        );
        let grad = |r: Result<HeadGradPlan, HeadLossPlanError>| r.unwrap_err().to_string();
        assert_eq!(
            grad(HeadGradPlan::new(4, 16, 4, 0)),
            "head-loss-rows-per-group-out-of-range"
        );
        assert_eq!(
            grad(HeadGradPlan::new(4, 16, 4, MAX_GROUP_ROWS + 1)),
            "head-loss-rows-per-group-out-of-range"
        );
        assert_eq!(
            grad(HeadGradPlan::new(70_000, 16, 4, 1)),
            "head-loss-grid-above-limit"
        );
    }

    #[test]
    fn the_push_blocks_and_the_grids_carry_the_plans() {
        let word = |b: &[u8], i: usize| u32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap());
        let lse = HeadLsePlan::new(2048, 151_936, 32).unwrap();
        assert_eq!(lse.dispatch_groups(), [2048, 1, 1]);
        let push = lse.push_constants();
        assert_eq!(
            (0..4).map(|i| word(&push, i)).collect::<Vec<_>>(),
            vec![2048, 151_936, 32, 2048]
        );
        // past one dimension's guarantee the rows fold into y
        let many = HeadLsePlan::new(70_000, 16, 2).unwrap();
        let [x, y, _] = many.dispatch_groups();
        assert_eq!(x, MAX_GROUPS_X);
        assert!(x as u64 * y as u64 >= 70_000);
        let back = HeadGradPlan::new(2048, 151_936, 32, 64).unwrap();
        assert_eq!(back.groups(), 32);
        assert_eq!(back.dispatch_groups(), [297, 32, 1]);
        assert_eq!(back.partial_elements(), 32 * 151_936);
        assert_eq!(back.row_value_elements(), 3 * 2048 + 1);
        let push = back.push_constants();
        assert_eq!(
            (0..4).map(|i| word(&push, i)).collect::<Vec<_>>(),
            vec![2048, 151_936, 32, 64]
        );
    }

    #[test]
    fn the_fp16_conversions_are_exact_and_round_to_nearest_even() {
        for bits in 0u16..=0xFFFF {
            let v = f16_to_f32(bits);
            if v.is_nan() {
                assert_eq!(f32_to_f16(v) & 0x7E00, 0x7E00);
            } else {
                assert_eq!(f32_to_f16(v), bits, "{bits:#06x}");
            }
        }
        // halfway between 1 and the next fp16 (1 + 2^-10) rounds to even: 1
        assert_eq!(f32_to_f16(1.0 + 2f32.powi(-11)), 0x3C00);
        // just above halfway rounds up
        assert_eq!(f32_to_f16(1.0 + 2f32.powi(-11) + 2f32.powi(-20)), 0x3C01);
        // past the largest finite fp16 (65504) at the halfway point: infinity
        assert_eq!(f32_to_f16(65_520.0), 0x7C00);
    }

    #[test]
    fn the_references_hold_the_law_on_a_small_row() {
        // one row of four columns, top-k {1, 3}, target 3 (a top column)
        let plan = HeadLsePlan::new(1, 4, 2).unwrap();
        let yp: Vec<u16> = [0.5f32, 1.0, -2.0, 0.25]
            .iter()
            .map(|&v| f32_to_f16(v))
            .collect();
        let h = [2.0f32, 1.0, 0.5, 4.0];
        let top = [1i64, 3];
        let out = reference_lse(&yp, &h, &top, &plan);
        let z = [1.0f64, 1.0, -1.0, 1.0];
        let all = (z.iter().map(|v| v.exp()).sum::<f64>()).ln();
        let rest = (z[0].exp() + z[2].exp()).ln();
        assert!((out[0] - all).abs() < 1e-12 && (out[1] - rest).abs() < 1e-12);
        // an index out of range makes the row NaN
        assert!(reference_lse(&yp, &h, &[1, 4], &plan)
            .iter()
            .all(|v| v.is_nan()));
        let back = HeadGradPlan::new(1, 4, 2, 1).unwrap();
        let lse = all as f32;
        let row_values = [lse, 1.5, 1.25, 0.5];
        let gkp = [0.1f32, 0.2];
        let (dyp, dh) = reference_grad(&yp, &h, &top, &[3], &row_values, &gkp, &back);
        let q = |c: usize| (z[c] - f64::from(lse)).exp();
        let g = [
            q(0) * 1.25,
            q(1) * 1.5 - 0.1f32 as f64,
            q(2) * 1.25,
            q(3) * 1.5 - 0.2f32 as f64 - 0.5,
        ];
        for c in 0..4 {
            assert!(
                (dyp[c] - g[c] * f64::from(h[c])).abs() < 1e-12,
                "column {c}"
            );
            let y = f64::from(f16_to_f32(yp[c]));
            assert!((dh[c] - y * g[c]).abs() < 1e-12, "d/dh {c}");
        }
    }
}
