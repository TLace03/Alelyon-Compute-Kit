//! Pure validation and dispatch planning for expert-factor-f16/v1 (2026-09-29):
//! the sub-bit student's expert factors, each built from its signs in one pass,
//! and their backward (`kernels/expert_factor_f16.comp`).
//!
//! For each routed expert `q < P` (its id `ids[q]`, int64, below `E`) and each
//! factor row `m < M`, the forward writes columns `[OFF, OFF + R8)` of the fp16
//! factor `(P, M, LD)`:
//!
//!   `out[q, m, OFF + c] = c < R ? sign(e, m, c) * f16(row[q, m] * col[q, c]) : 0`
//!
//! with an absent scale taken as 1, and the sign from the fp32 latent
//! `W[e, m, c] >= 0` (op 0) or from bit `c % 8` of byte `c / 8` of the packed
//! cache `S[e, m]` (op 1), or, with [`INT8_SIGNS`] (2026-09-30), from the
//! signed byte `S[e, m, c]` of an int8 cache, +1 or -1 (the student's
//! `_sign_cache` form; any other byte makes its entry and its sums NaN).
//! The backward (ops 2-4) reads the fp16 upstream
//! gradient at the same columns. The student's reference is
//! research/subbit_expert/moe/formats.py `ExpertsIndependent._fused_factors`
//! (`scaled_sign`, `_pad_rank8`); its CUDA counterpart is
//! `student_kernels.latent_factors` / `packed_factors`.
//!
//! These bounds limit this implementation's contract; they are not limits of
//! Vulkan or of any selected GPU.

use std::fmt;

pub const ABI: &str = "expert-factor-f16/v1";
pub const SCHEMA: u32 = 1;
pub const WORKGROUP: u32 = 64;
/// Every buffer the kernel addresses holds at most this many elements, so
/// every element and word offset stays below 2^31.
pub const MAX_ELEMENTS: usize = 1 << 31;
/// Vulkan's guaranteed minimum for one dispatch dimension.
pub const MAX_GROUPS_X: u32 = 65_535;
/// A factor's rank: the kernel walks a row's columns in one invocation.
pub const MAX_RANK: u32 = 1024;
/// op, flags, P, M, R, R8, LD, OFF, NB, k_bits, E, groups_x: twelve u32 words.
pub const PUSH_BYTES: usize = 48;
pub const HAS_ROW: u32 = 1;
pub const HAS_COL: u32 = 2;
/// The cache ops (1, 3, 4) read an int8 cache, one signed byte a sign, in
/// place of packed bits: the student's frozen attention factors (2026-09-30).
pub const INT8_SIGNS: u32 = 4;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum ExpertFactorOp {
    /// The factor from the fp32 latent's signs.
    LatentForward = 0,
    /// The factor from the packed sign cache.
    PackedForward = 1,
    /// The latent's gradient (the SmoothSign surrogate) and the scales' partials.
    LatentBackward = 2,
    /// The row scale's gradient, from the packed cache's signs.
    PackedBackwardRow = 3,
    /// The column scale's gradient, from the packed cache's signs.
    PackedBackwardCol = 4,
}

impl TryFrom<u32> for ExpertFactorOp {
    type Error = ExpertFactorPlanError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::LatentForward),
            1 => Ok(Self::PackedForward),
            2 => Ok(Self::LatentBackward),
            3 => Ok(Self::PackedBackwardRow),
            4 => Ok(Self::PackedBackwardCol),
            _ => Err(ExpertFactorPlanError::OperationOutOfRange),
        }
    }
}

impl ExpertFactorOp {
    pub fn is_latent(self) -> bool {
        matches!(self, Self::LatentForward | Self::LatentBackward)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExpertFactorPlanError {
    DimensionOutOfRange,
    OperationOutOfRange,
    FlagsOutOfRange,
    RankAboveLimit,
    PaddedRankInvalid,
    LayoutInvalid,
    ScaleMissing,
    Int8SignsWithLatent,
    ElementLimitExceeded,
    GridAboveLimit,
}

impl fmt::Display for ExpertFactorPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::DimensionOutOfRange => "expert-factor-dimension-out-of-range",
            Self::OperationOutOfRange => "expert-factor-operation-out-of-range",
            Self::FlagsOutOfRange => "expert-factor-flags-out-of-range",
            Self::RankAboveLimit => "expert-factor-rank-above-limit",
            Self::PaddedRankInvalid => "expert-factor-padded-rank-invalid",
            Self::LayoutInvalid => "expert-factor-layout-invalid",
            Self::ScaleMissing => "expert-factor-scale-missing",
            Self::Int8SignsWithLatent => "expert-factor-int8-signs-with-latent",
            Self::ElementLimitExceeded => "expert-factor-element-limit-exceeded",
            Self::GridAboveLimit => "expert-factor-grid-above-limit",
        })
    }
}

impl std::error::Error for ExpertFactorPlanError {}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExpertFactorPlan {
    op: ExpertFactorOp,
    flags: u32,
    experts: u32,
    p: u32,
    m: u32,
    r: u32,
    r8: u32,
    ld: u32,
    off: u32,
    nb: u32,
    k: f32,
    dispatch_groups: [u32; 3],
}

impl ExpertFactorPlan {
    /// `experts` (E) latent or cache rows; `p` routed experts; `m` factor rows;
    /// rank `r`, padded to `r8` (a multiple of 8, at least `r`, less than
    /// `r + 8`); the factor's row length `ld` and this part's first column `off`
    /// (both even, `off + r8 <= ld`); the SmoothSign slope `k` (op 2).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        op: ExpertFactorOp,
        flags: u32,
        experts: u32,
        p: u32,
        m: u32,
        r: u32,
        r8: u32,
        ld: u32,
        off: u32,
        k: f32,
    ) -> Result<Self, ExpertFactorPlanError> {
        if experts == 0 || p == 0 || m == 0 || r == 0 {
            return Err(ExpertFactorPlanError::DimensionOutOfRange);
        }
        if flags > (HAS_ROW | HAS_COL | INT8_SIGNS) {
            return Err(ExpertFactorPlanError::FlagsOutOfRange);
        }
        if flags & INT8_SIGNS != 0 && op.is_latent() {
            return Err(ExpertFactorPlanError::Int8SignsWithLatent);
        }
        if r > MAX_RANK {
            return Err(ExpertFactorPlanError::RankAboveLimit);
        }
        if r8 % 8 != 0 || r8 < r || r8 >= r + 8 {
            return Err(ExpertFactorPlanError::PaddedRankInvalid);
        }
        if ld % 2 != 0 || off % 2 != 0 || off as u64 + r8 as u64 > ld as u64 {
            return Err(ExpertFactorPlanError::LayoutInvalid);
        }
        // a column scale comes with a row scale (the student's (h l^T) products), and op 4
        // reduces against the row scale
        if flags & (HAS_ROW | HAS_COL) == HAS_COL
            || (op == ExpertFactorOp::PackedBackwardCol && flags & HAS_ROW == 0)
        {
            return Err(ExpertFactorPlanError::ScaleMissing);
        }
        let (p_, m_, r_) = (p as usize, m as usize, r as usize);
        let nb = r.div_ceil(8);
        let too_big = [
            p_ * m_ * ld as usize,
            experts as usize * m_ * r_,
            experts as usize * m_ * nb as usize,
            p_ * m_ * r_,
        ]
        .iter()
        .any(|&n| n > MAX_ELEMENTS);
        if too_big {
            return Err(ExpertFactorPlanError::ElementLimitExceeded);
        }
        let invocations = if op == ExpertFactorOp::PackedBackwardCol {
            p_ * r_
        } else {
            p_ * m_
        };
        let groups = invocations.div_ceil(WORKGROUP as usize);
        let x = groups.min(MAX_GROUPS_X as usize);
        let y = groups.div_ceil(x);
        if y > MAX_GROUPS_X as usize {
            return Err(ExpertFactorPlanError::GridAboveLimit);
        }
        Ok(Self {
            op,
            flags,
            experts,
            p,
            m,
            r,
            r8,
            ld,
            off,
            nb,
            k,
            dispatch_groups: [x as u32, y as u32, 1],
        })
    }

    pub fn op(&self) -> ExpertFactorOp {
        self.op
    }
    pub fn flags(&self) -> u32 {
        self.flags
    }
    pub fn p(&self) -> u32 {
        self.p
    }
    pub fn m(&self) -> u32 {
        self.m
    }
    pub fn r(&self) -> u32 {
        self.r
    }
    pub fn r8(&self) -> u32 {
        self.r8
    }
    pub fn nb(&self) -> u32 {
        self.nb
    }
    /// Bytes of the source: the latent (E, M, R) f32, the packed cache (E, M, NB)
    /// bytes, or the int8 cache (E, M, R) bytes.
    pub fn src_bytes(&self) -> u64 {
        let rows = self.experts as u64 * self.m as u64;
        if self.op.is_latent() {
            rows * self.r as u64 * 4
        } else if self.flags & INT8_SIGNS != 0 {
            rows * self.r as u64
        } else {
            rows * self.nb as u64
        }
    }
    /// Bytes of the fp16 factor, (P, M, LD).
    pub fn factor_bytes(&self) -> u64 {
        self.p as u64 * self.m as u64 * self.ld as u64 * 2
    }
    /// Bytes of the row scale (P, M), or 0 without one.
    pub fn row_bytes(&self) -> u64 {
        if self.flags & HAS_ROW != 0 {
            self.p as u64 * self.m as u64 * 4
        } else {
            0
        }
    }
    /// Bytes of the column scale (P, R), or 0 without one.
    pub fn col_bytes(&self) -> u64 {
        if self.flags & HAS_COL != 0 {
            self.p as u64 * self.r as u64 * 4
        } else {
            0
        }
    }
    /// Bytes of output A: the latent's gradient (op 2), g_row (op 3), g_col (op 4).
    pub fn out_a_bytes(&self) -> u64 {
        match self.op {
            ExpertFactorOp::LatentBackward => {
                self.experts as u64 * self.m as u64 * self.r as u64 * 4
            }
            ExpertFactorOp::PackedBackwardRow => self.p as u64 * self.m as u64 * 4,
            ExpertFactorOp::PackedBackwardCol => self.p as u64 * self.r as u64 * 4,
            _ => 0,
        }
    }
    /// Bytes of output B (op 2 only): gs (P, M, R) with both scales, g_row (P, M)
    /// with a row scale only, else nothing.
    pub fn out_b_bytes(&self) -> u64 {
        if self.op != ExpertFactorOp::LatentBackward {
            return 0;
        }
        match (self.flags & HAS_ROW != 0, self.flags & HAS_COL != 0) {
            (true, true) => self.p as u64 * self.m as u64 * self.r as u64 * 4,
            (true, false) => self.p as u64 * self.m as u64 * 4,
            _ => 0,
        }
    }
    pub fn dispatch_groups(&self) -> [u32; 3] {
        self.dispatch_groups
    }

    pub fn push_constants(&self) -> [u8; PUSH_BYTES] {
        let words = [
            self.op as u32,
            self.flags,
            self.p,
            self.m,
            self.r,
            self.r8,
            self.ld,
            self.off,
            self.nb,
            self.k.to_bits(),
            self.experts,
            self.dispatch_groups[0],
        ];
        let mut bytes = [0u8; PUSH_BYTES];
        for (index, word) in words.iter().enumerate() {
            bytes[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }
}

/// The operations' law on the host, for tests: the inputs as the kernel reads
/// them (the latent's f32 values, the packed cache's bytes or the int8 cache's
/// signs, the ids, the scales, the
/// fp16 factor's bits) and its outputs in the kernel's own arithmetic order.
pub struct Reference<'a> {
    pub plan: &'a ExpertFactorPlan,
    pub latent: &'a [f32],
    pub packed: &'a [u8],
    pub int8: &'a [i8],
    pub ids: &'a [i64],
    pub row: &'a [f32],
    pub col: &'a [f32],
}

impl Reference<'_> {
    fn sign(&self, e: usize, m: usize, c: usize) -> f32 {
        let (mm, r, nb) = (
            self.plan.m as usize,
            self.plan.r as usize,
            self.plan.nb as usize,
        );
        if self.plan.op.is_latent() {
            if self.latent[(e * mm + m) * r + c] >= 0.0 {
                1.0
            } else {
                -1.0
            }
        } else if self.plan.flags & INT8_SIGNS != 0 {
            match self.int8[(e * mm + m) * r + c] {
                1 => 1.0,
                -1 => -1.0,
                _ => f32::NAN,
            }
        } else if (self.packed[(e * mm + m) * nb + c / 8] >> (c % 8)) & 1 != 0 {
            1.0
        } else {
            -1.0
        }
    }

    fn scale(&self, q: usize, m: usize, c: usize) -> f32 {
        let (mm, r) = (self.plan.m as usize, self.plan.r as usize);
        let row = if self.plan.flags & HAS_ROW != 0 {
            Some(self.row[q * mm + m])
        } else {
            None
        };
        let col = if self.plan.flags & HAS_COL != 0 {
            Some(self.col[q * r + c])
        } else {
            None
        };
        match (row, col) {
            (Some(a), Some(b)) => a * b,
            (Some(a), None) => a,
            (None, Some(b)) => b,
            (None, None) => 1.0,
        }
    }

    /// The forward's fp16 factor part: (P, M, R8) bits, `sign * f16(scale)` below the rank.
    pub fn forward(&self) -> Vec<u16> {
        let (p, mm, r, r8) = (
            self.plan.p as usize,
            self.plan.m as usize,
            self.plan.r as usize,
            self.plan.r8 as usize,
        );
        let mut out = vec![0u16; p * mm * r8];
        for q in 0..p {
            let e = self.ids[q] as usize;
            for m in 0..mm {
                for c in 0..r {
                    let s16 = crate::head_loss_ops::f32_to_f16(self.scale(q, m, c));
                    let sign = self.sign(e, m, c);
                    out[(q * mm + m) * r8 + c] = if sign.is_nan() {
                        0x7E00
                    } else if sign < 0.0 {
                        s16 ^ 0x8000
                    } else {
                        s16
                    };
                }
            }
        }
        out
    }

    /// The backward's outputs in f64 from the fp16 gradient part `grad` (P, M, R8):
    /// (the latent's gradient rows by (q, m, c), `g * sign` (P, M, R), its row sums
    /// against the column scale (P, M), its column sums against the row scale (P, R)).
    pub fn backward(&self, grad: &[u16], k: f32) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let (p, mm, r, r8) = (
            self.plan.p as usize,
            self.plan.m as usize,
            self.plan.r as usize,
            self.plan.r8 as usize,
        );
        let (mut gw, mut gs) = (vec![0.0f64; p * mm * r], vec![0.0f64; p * mm * r]);
        let (mut g_row, mut g_col) = (vec![0.0f64; p * mm], vec![0.0f64; p * r]);
        for q in 0..p {
            let e = self.ids[q] as usize;
            for m in 0..mm {
                for c in 0..r {
                    let g = f64::from(crate::head_loss_ops::f16_to_f32(
                        grad[(q * mm + m) * r8 + c],
                    ));
                    let sg = f64::from(self.sign(e, m, c)) * g;
                    gs[(q * mm + m) * r + c] = sg;
                    if self.plan.op.is_latent() {
                        let w = f64::from(self.latent[(e * mm + m) * r + c]);
                        let t = (w * f64::from(k)).tanh();
                        gw[(q * mm + m) * r + c] =
                            g * f64::from(self.scale(q, m, c)) * ((t * t - 1.0) * -f64::from(k));
                    }
                    let col = if self.plan.flags & HAS_COL != 0 {
                        f64::from(self.col[q * r + c])
                    } else {
                        1.0
                    };
                    let row = if self.plan.flags & HAS_ROW != 0 {
                        f64::from(self.row[q * mm + m])
                    } else {
                        1.0
                    };
                    g_row[q * mm + m] += sg * col;
                    g_col[q * r + c] += sg * row;
                }
            }
        }
        (gw, gs, g_row, g_col)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(op: ExpertFactorOp, flags: u32) -> Result<ExpertFactorPlan, ExpertFactorPlanError> {
        ExpertFactorPlan::new(op, flags, 128, 120, 768, 41, 48, 96, 48, 30.0)
    }

    #[test]
    fn refusals_are_named() {
        let named = |r: Result<ExpertFactorPlan, ExpertFactorPlanError>| r.unwrap_err().to_string();
        let f = ExpertFactorOp::LatentForward;
        assert_eq!(
            named(ExpertFactorPlan::new(f, 0, 0, 1, 1, 1, 8, 8, 0, 1.0)),
            "expert-factor-dimension-out-of-range"
        );
        assert_eq!(named(plan(f, 8)), "expert-factor-flags-out-of-range");
        assert_eq!(
            named(plan(f, HAS_ROW | INT8_SIGNS)),
            "expert-factor-int8-signs-with-latent"
        );
        assert_eq!(
            named(plan(ExpertFactorOp::LatentBackward, INT8_SIGNS)),
            "expert-factor-int8-signs-with-latent"
        );
        assert_eq!(
            named(plan(ExpertFactorOp::PackedForward, HAS_COL | INT8_SIGNS)),
            "expert-factor-scale-missing"
        );
        assert_eq!(
            named(ExpertFactorPlan::new(
                f, 0, 4, 1, 1, 2000, 2000, 2000, 0, 1.0
            )),
            "expert-factor-rank-above-limit"
        );
        assert_eq!(
            named(ExpertFactorPlan::new(f, 0, 4, 1, 1, 41, 44, 96, 0, 1.0)),
            "expert-factor-padded-rank-invalid"
        );
        assert_eq!(
            named(ExpertFactorPlan::new(f, 0, 4, 1, 1, 41, 56, 96, 0, 1.0)),
            "expert-factor-padded-rank-invalid"
        );
        assert_eq!(
            named(ExpertFactorPlan::new(f, 0, 4, 1, 1, 41, 48, 96, 49, 1.0)),
            "expert-factor-layout-invalid"
        );
        assert_eq!(
            named(ExpertFactorPlan::new(f, 0, 4, 1, 1, 41, 48, 94, 48, 1.0)),
            "expert-factor-layout-invalid"
        );
        assert_eq!(
            named(plan(ExpertFactorOp::PackedBackwardCol, HAS_COL)),
            "expert-factor-scale-missing"
        );
        assert_eq!(
            named(plan(ExpertFactorOp::LatentForward, HAS_COL)),
            "expert-factor-scale-missing"
        );
        assert_eq!(
            named(plan(ExpertFactorOp::PackedBackwardCol, 0)),
            "expert-factor-scale-missing"
        );
        assert_eq!(
            ExpertFactorOp::try_from(5),
            Err(ExpertFactorPlanError::OperationOutOfRange)
        );
    }

    #[test]
    fn the_sizes_and_the_grid_follow_the_operation() {
        let fwd = plan(ExpertFactorOp::LatentForward, HAS_ROW | HAS_COL).unwrap();
        assert_eq!(fwd.src_bytes(), 128 * 768 * 41 * 4);
        assert_eq!(fwd.factor_bytes(), 120 * 768 * 96 * 2);
        assert_eq!(
            fwd.dispatch_groups(),
            [(120 * 768usize).div_ceil(64) as u32, 1, 1]
        );
        let int8 = plan(ExpertFactorOp::PackedForward, HAS_ROW | INT8_SIGNS).unwrap();
        assert_eq!(int8.src_bytes(), 128 * 768 * 41);
        let int8_col = plan(
            ExpertFactorOp::PackedBackwardCol,
            HAS_ROW | HAS_COL | INT8_SIGNS,
        )
        .unwrap();
        assert_eq!(int8_col.src_bytes(), 128 * 768 * 41);
        assert_eq!(int8_col.out_a_bytes(), 120 * 41 * 4);
        let packed = plan(ExpertFactorOp::PackedForward, HAS_ROW).unwrap();
        assert_eq!(packed.nb(), 6);
        assert_eq!(packed.src_bytes(), 128 * 768 * 6);
        assert_eq!(packed.col_bytes(), 0);
        let back = plan(ExpertFactorOp::LatentBackward, HAS_ROW | HAS_COL).unwrap();
        assert_eq!(back.out_a_bytes(), 128 * 768 * 41 * 4);
        assert_eq!(back.out_b_bytes(), 120 * 768 * 41 * 4);
        let row_only = plan(ExpertFactorOp::LatentBackward, HAS_ROW).unwrap();
        assert_eq!(row_only.out_b_bytes(), 120 * 768 * 4);
        let col = plan(ExpertFactorOp::PackedBackwardCol, HAS_ROW | HAS_COL).unwrap();
        assert_eq!(col.out_a_bytes(), 120 * 41 * 4);
        assert_eq!(
            col.dispatch_groups(),
            [(120 * 41usize).div_ceil(64) as u32, 1, 1]
        );
        let push = back.push_constants();
        let word = |i: usize| u32::from_le_bytes(push[i * 4..i * 4 + 4].try_into().unwrap());
        assert_eq!(
            (0..9).map(word).collect::<Vec<_>>(),
            vec![2, 3, 120, 768, 41, 48, 96, 48, 6]
        );
        assert_eq!(f32::from_bits(word(9)), 30.0);
        assert_eq!(word(10), 128);
    }
}
