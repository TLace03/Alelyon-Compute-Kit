//! Pure validation and dispatch planning for rotary-f32/v1 (2026-09-29): the
//! rotate-half rotary position embedding over dense FP32 rows, and its backward
//! (`kernels/rotary_f32.comp`).
//!
//! Row `r` of the primary sits at position `(r / pos_div) % positions`, and the
//! two tables (cos, sin) hold `positions` rows of `cols`. One layout covers the
//! two a model hands over. A `(B, H, T, D)` tensor stored in that order has
//! `pos_div` 1. The transposed view of a dense `(B, T, H, D)` projection, which
//! is what attention's queries and keys are, has `pos_div` H.
//!
//! These bounds limit this implementation's contract; they are not limits of
//! Vulkan or of any selected GPU.

use std::fmt;

pub const ABI: &str = "rotary-f32/v1";
/// Operations 0 (forward) and 1 (backward) over the push block below.
pub const SCHEMA: u32 = 1;
pub const WORKGROUP: u32 = 256;
/// Every buffer the kernel addresses holds at most this many elements, so a
/// pair index and every element offset stay below 2^31.
pub const MAX_ELEMENTS: usize = 1 << 30;
/// Vulkan's guaranteed minimum for one dispatch dimension.
pub const MAX_GROUPS_X: u32 = 65_535;
/// rows, cols, op, pos_div, positions, pairs: six u32 words.
pub const PUSH_BYTES: usize = 24;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum RotaryOp {
    Forward = 0,
    Backward = 1,
}

impl TryFrom<u32> for RotaryOp {
    type Error = RotaryPlanError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Forward),
            1 => Ok(Self::Backward),
            _ => Err(RotaryPlanError::OperationOutOfRange),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RotaryPlanError {
    DimensionOutOfRange,
    OddColumns,
    ElementLimitExceeded,
    OperationOutOfRange,
}

impl fmt::Display for RotaryPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::DimensionOutOfRange => "rotary-dimension-out-of-range",
            Self::OddColumns => "rotary-odd-columns",
            Self::ElementLimitExceeded => "rotary-element-limit-exceeded",
            Self::OperationOutOfRange => "rotary-operation-out-of-range",
        })
    }
}

impl std::error::Error for RotaryPlanError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RotaryPlan {
    rows: u32,
    cols: u32,
    op: RotaryOp,
    pos_div: u32,
    positions: u32,
    elements: usize,
    table_len: usize,
    pairs: u32,
    dispatch_groups: [u32; 3],
}

impl RotaryPlan {
    /// `rows` rows of `cols` elements; `cols` even. A zero dimension, a zero
    /// `pos_div` or `positions`, and an odd `cols` are refused by name.
    pub fn new(
        rows: u32,
        cols: u32,
        op: RotaryOp,
        pos_div: u32,
        positions: u32,
    ) -> Result<Self, RotaryPlanError> {
        if rows == 0 || cols == 0 || pos_div == 0 || positions == 0 {
            return Err(RotaryPlanError::DimensionOutOfRange);
        }
        if cols % 2 != 0 {
            return Err(RotaryPlanError::OddColumns);
        }
        let elements = rows as usize * cols as usize;
        let table_len = positions as usize * cols as usize;
        if elements > MAX_ELEMENTS || table_len > MAX_ELEMENTS {
            return Err(RotaryPlanError::ElementLimitExceeded);
        }
        let pairs = (elements / 2) as u32;
        let groups = pairs.div_ceil(WORKGROUP);
        let dispatch_groups = if groups <= MAX_GROUPS_X {
            [groups, 1, 1]
        } else {
            [MAX_GROUPS_X, groups.div_ceil(MAX_GROUPS_X), 1]
        };
        Ok(Self {
            rows,
            cols,
            op,
            pos_div,
            positions,
            elements,
            table_len,
            pairs,
            dispatch_groups,
        })
    }

    pub fn rows(&self) -> u32 {
        self.rows
    }
    pub fn cols(&self) -> u32 {
        self.cols
    }
    pub fn op(&self) -> RotaryOp {
        self.op
    }
    /// Elements of the primary and of the output.
    pub fn elements(&self) -> usize {
        self.elements
    }
    /// Elements of each table.
    pub fn table_len(&self) -> usize {
        self.table_len
    }
    pub fn pairs(&self) -> u32 {
        self.pairs
    }
    pub fn dispatch_groups(&self) -> [u32; 3] {
        self.dispatch_groups
    }

    pub fn push_constants(&self) -> [u8; PUSH_BYTES] {
        let mut bytes = [0u8; PUSH_BYTES];
        let words = [
            self.rows,
            self.cols,
            self.op as u32,
            self.pos_div,
            self.positions,
            self.pairs,
        ];
        for (index, word) in words.iter().enumerate() {
            bytes[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }
}

/// The law the kernel computes, on the host, for tests and for a caller's own
/// reference: each product rounded, then their sum, in the kernel's order.
pub fn reference(
    primary: &[f32],
    cos: &[f32],
    sin: &[f32],
    plan: &RotaryPlan,
) -> Result<Vec<f32>, RotaryPlanError> {
    if primary.len() != plan.elements || cos.len() != plan.table_len || sin.len() != plan.table_len
    {
        return Err(RotaryPlanError::DimensionOutOfRange);
    }
    let cols = plan.cols as usize;
    let half = cols / 2;
    let mut out = vec![0.0f32; plan.elements];
    for r in 0..plan.rows as usize {
        let t = (r / plan.pos_div as usize) % plan.positions as usize;
        for i in 0..half {
            let (lo, hi) = (r * cols + i, r * cols + i + half);
            let (tlo, thi) = (t * cols + i, t * cols + i + half);
            let (a, b) = (primary[lo], primary[hi]);
            match plan.op {
                RotaryOp::Forward => {
                    out[lo] = a * cos[tlo] + (-b) * sin[tlo];
                    out[hi] = b * cos[thi] + a * sin[thi];
                }
                RotaryOp::Backward => {
                    out[lo] = a * cos[tlo] + b * sin[thi];
                    out[hi] = b * cos[thi] - a * sin[tlo];
                }
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_operation_words_decode_and_nothing_past_them_does() {
        assert_eq!(RotaryOp::try_from(0), Ok(RotaryOp::Forward));
        assert_eq!(RotaryOp::try_from(1), Ok(RotaryOp::Backward));
        assert_eq!(
            RotaryOp::try_from(2),
            Err(RotaryPlanError::OperationOutOfRange)
        );
        assert_eq!(SCHEMA, 1);
    }

    #[test]
    fn refusals_are_named() {
        let f = RotaryOp::Forward;
        let named = |r: Result<RotaryPlan, RotaryPlanError>| r.unwrap_err().to_string();
        assert_eq!(
            named(RotaryPlan::new(0, 16, f, 1, 8)),
            "rotary-dimension-out-of-range"
        );
        assert_eq!(
            named(RotaryPlan::new(8, 16, f, 0, 8)),
            "rotary-dimension-out-of-range"
        );
        assert_eq!(
            named(RotaryPlan::new(8, 16, f, 1, 0)),
            "rotary-dimension-out-of-range"
        );
        assert_eq!(named(RotaryPlan::new(8, 15, f, 1, 8)), "rotary-odd-columns");
        assert_eq!(
            named(RotaryPlan::new(1 << 20, 1 << 11, f, 1, 8)),
            "rotary-element-limit-exceeded"
        );
    }

    #[test]
    fn the_push_block_and_the_grid_carry_the_plan() {
        let plan = RotaryPlan::new(32_768, 16, RotaryOp::Backward, 8, 1024).unwrap();
        assert_eq!(plan.pairs(), 262_144);
        assert_eq!(plan.dispatch_groups(), [1024, 1, 1]);
        let push = plan.push_constants();
        let word = |i: usize| u32::from_le_bytes(push[i * 4..i * 4 + 4].try_into().unwrap());
        assert_eq!(
            (0..6).map(word).collect::<Vec<_>>(),
            vec![32_768, 16, 1, 8, 1024, 262_144]
        );
        // past one dimension's guarantee the grid folds into a second
        let big = RotaryPlan::new(1 << 19, 128, RotaryOp::Forward, 1, 1).unwrap();
        let [x, y, z] = big.dispatch_groups();
        assert_eq!((x, z), (MAX_GROUPS_X, 1));
        assert!(x as u64 * y as u64 * WORKGROUP as u64 >= big.pairs() as u64);
    }

    #[test]
    fn the_reference_is_the_rotate_half_law_and_its_backward() {
        // two rows at positions 0 and 1 (pos_div 1), cols 4: h = 2
        let plan = RotaryPlan::new(2, 4, RotaryOp::Forward, 1, 2).unwrap();
        let x = [1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.0, -3.0];
        let cos = [1.0, 0.5, 1.0, 0.5, 0.0, 0.25, 0.0, 0.25];
        let sin = [0.0, 0.5, 0.0, 0.5, 1.0, 0.75, 1.0, 0.75];
        let out = reference(&x, &cos, &sin, &plan).unwrap();
        // row 0: [1*1 + -3*0, 2*.5 + -4*.5, 3*1 + 1*0, 4*.5 + 2*.5]
        assert_eq!(&out[..4], &[1.0, -1.0, 3.0, 3.0]);
        // row 1: [-1*0 + -2*1, .5*.25 + 3*.75, 2*0 + -1*1, -3*.25 + .5*.75]
        assert_eq!(&out[4..], &[-2.0, 2.375, -1.0, -0.375]);
        // the backward is the transpose: <out(x), g> == <x, back(g)> exactly here
        let back = RotaryPlan::new(2, 4, RotaryOp::Backward, 1, 2).unwrap();
        let g = [0.5, -1.0, 2.0, 1.0, 1.0, 1.0, -1.0, 2.0];
        let gx = reference(&g, &cos, &sin, &back).unwrap();
        let lhs: f32 = out.iter().zip(&g).map(|(a, b)| a * b).sum();
        let rhs: f32 = x.iter().zip(&gx).map(|(a, b)| a * b).sum();
        assert_eq!(lhs, rhs);
    }
}
