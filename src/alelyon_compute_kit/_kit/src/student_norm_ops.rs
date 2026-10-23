//! Pure validation and dispatch planning for student-norm-f16/v1 (2026-09-29):
//! the sub-bit student's attention-block RMSNorms, the residual add before the
//! post-attention one, and its q/k norms with RoPE, forward and backward
//! (`kernels/student_norm_f16.comp`; research/subbit_expert/moe/student.py's O12
//! Fusables are the law).
//!
//! These bounds limit this implementation's contract; they are not limits of
//! Vulkan or of any selected GPU.

use std::fmt;

pub const ABI: &str = "student-norm-f16/v1";
pub const SCHEMA: u32 = 1;
/// A norm row (ops 0, 1): 64 lanes of at most 32 element pairs.
pub const MAX_NORM_D: u32 = 4096;
/// A q/k row (ops 2, 3) is staged in 1 KiB of shared memory.
pub const MAX_QK_D: u32 = 256;
pub const MAX_GROUP_ROWS: u32 = 4096;
pub const MAX_ELEMENTS: usize = 1 << 31;
pub const MAX_GROUPS_X: u32 = 65_535;
/// op, flags, rows, D, eps bits, T, H, rows_per_group, groups_x.
pub const PUSH_BYTES: usize = 36;

pub const X16: u32 = 1;
pub const AUX16: u32 = 2;
pub const ADD: u32 = 4;
pub const Y32: u32 = 8;
pub const Y16: u32 = 16;
pub const DY16: u32 = 32;
pub const DH: u32 = 64;
pub const SAME_LAYOUT: u32 = 128;
/// Op 1 with DY16: no f32 upstream gradient (the fp16 copy's alone); `dy` is not read.
pub const DY_ABSENT: u32 = 256;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum NormOp {
    Norm = 0,
    NormBackward = 1,
    QkNormRope = 2,
    QkNormRopeBackward = 3,
}

impl TryFrom<u32> for NormOp {
    type Error = NormPlanError;
    fn try_from(v: u32) -> Result<Self, Self::Error> {
        match v {
            0 => Ok(Self::Norm),
            1 => Ok(Self::NormBackward),
            2 => Ok(Self::QkNormRope),
            3 => Ok(Self::QkNormRopeBackward),
            _ => Err(NormPlanError::OperationOutOfRange),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NormPlanError {
    DimensionOutOfRange,
    OddWidth,
    WidthAboveLimit,
    OperationOutOfRange,
    FlagsInvalid,
    PositionsInvalid,
    RowsPerGroupOutOfRange,
    EpsilonInvalid,
    ElementLimitExceeded,
    GridAboveLimit,
}

impl fmt::Display for NormPlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::DimensionOutOfRange => "student-norm-dimension-out-of-range",
            Self::OddWidth => "student-norm-odd-width",
            Self::WidthAboveLimit => "student-norm-width-above-limit",
            Self::OperationOutOfRange => "student-norm-operation-out-of-range",
            Self::FlagsInvalid => "student-norm-flags-invalid",
            Self::PositionsInvalid => "student-norm-positions-invalid",
            Self::RowsPerGroupOutOfRange => "student-norm-rows-per-group-out-of-range",
            Self::EpsilonInvalid => "student-norm-epsilon-invalid",
            Self::ElementLimitExceeded => "student-norm-element-limit-exceeded",
            Self::GridAboveLimit => "student-norm-grid-above-limit",
        })
    }
}

impl std::error::Error for NormPlanError {}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NormPlan {
    op: NormOp,
    flags: u32,
    rows: u32,
    d: u32,
    eps: f32,
    t: u32,
    h: u32,
    rows_per_group: u32,
    groups: u32,
    dispatch_groups: [u32; 3],
}

impl NormPlan {
    /// `rows` rows of `d`; for the q/k ops `rows` = B * T * H with `t` positions and
    /// `h` heads (`rows` a multiple of `t * h`); `rows_per_group` rows a backward
    /// workgroup sums the weight's gradient over.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        op: NormOp,
        flags: u32,
        rows: u32,
        d: u32,
        eps: f32,
        t: u32,
        h: u32,
        rows_per_group: u32,
    ) -> Result<Self, NormPlanError> {
        if rows == 0 || d == 0 {
            return Err(NormPlanError::DimensionOutOfRange);
        }
        if d % 2 != 0 {
            return Err(NormPlanError::OddWidth);
        }
        let qk = matches!(op, NormOp::QkNormRope | NormOp::QkNormRopeBackward);
        if d > if qk { MAX_QK_D } else { MAX_NORM_D } {
            return Err(NormPlanError::WidthAboveLimit);
        }
        if !(eps.is_finite() && eps > 0.0) {
            return Err(NormPlanError::EpsilonInvalid);
        }
        let allowed = match op {
            NormOp::Norm => X16 | AUX16 | ADD | Y32 | Y16,
            NormOp::NormBackward => DY16 | DH | DY_ABSENT,
            NormOp::QkNormRope | NormOp::QkNormRopeBackward => SAME_LAYOUT,
        };
        if flags & !allowed != 0
            || (op == NormOp::Norm && flags & (Y32 | Y16) == 0)
            || (flags & DY_ABSENT != 0 && flags & DY16 == 0)
        {
            return Err(NormPlanError::FlagsInvalid);
        }
        if qk && (t == 0 || h == 0 || rows % (t * h) != 0) {
            return Err(NormPlanError::PositionsInvalid);
        }
        let backward = matches!(op, NormOp::NormBackward | NormOp::QkNormRopeBackward);
        if backward && (rows_per_group == 0 || rows_per_group > MAX_GROUP_ROWS) {
            return Err(NormPlanError::RowsPerGroupOutOfRange);
        }
        if rows as usize * d as usize > MAX_ELEMENTS {
            return Err(NormPlanError::ElementLimitExceeded);
        }
        let groups = if backward {
            rows.div_ceil(rows_per_group)
        } else {
            rows
        };
        let x = groups.min(MAX_GROUPS_X);
        let y = groups.div_ceil(x);
        if y > MAX_GROUPS_X {
            return Err(NormPlanError::GridAboveLimit);
        }
        Ok(Self {
            op,
            flags,
            rows,
            d,
            eps,
            t,
            h,
            rows_per_group,
            groups,
            dispatch_groups: [x, y, 1],
        })
    }

    pub fn op(&self) -> NormOp {
        self.op
    }
    pub fn flags(&self) -> u32 {
        self.flags
    }
    pub fn groups(&self) -> u32 {
        self.groups
    }
    fn cells(&self) -> u64 {
        self.rows as u64 * self.d as u64
    }
    fn has(&self, flag: u32) -> bool {
        self.flags & flag != 0
    }
    /// Bytes each role needs: x, aux, w, tab, out0, out1, dy, io2 (0 where the
    /// operation does not use the role).
    pub fn role_bytes(&self) -> [u64; 8] {
        let n = self.cells();
        let d4 = self.d as u64 * 4;
        match self.op {
            NormOp::Norm => [
                n * if self.has(X16) { 2 } else { 4 },
                if self.has(ADD) {
                    n * if self.has(AUX16) { 2 } else { 4 }
                } else {
                    0
                },
                d4,
                0,
                if self.has(Y32) { n * 4 } else { 0 },
                if self.has(Y16) { n * 2 } else { 0 },
                0,
                if self.has(ADD) { n * 4 } else { 0 },
            ],
            NormOp::NormBackward => [
                n * 4,
                if self.has(DY16) { n * 2 } else { 0 },
                d4,
                0,
                n * 4,
                self.groups as u64 * d4,
                if self.has(DY_ABSENT) { 0 } else { n * 4 },
                if self.has(DH) { n * 4 } else { 0 },
            ],
            NormOp::QkNormRope => [n * 2, 0, d4, 2 * self.t as u64 * d4, n * 2, 0, 0, 0],
            NormOp::QkNormRopeBackward => [
                n * 2,
                n * 2,
                d4,
                2 * self.t as u64 * d4,
                n * 2,
                self.groups as u64 * d4,
                0,
                0,
            ],
        }
    }
    /// Which roles the operation writes (out0, out1, io2 by index 4, 5, 7).
    pub fn writes(&self) -> [bool; 8] {
        let mut w = [false; 8];
        match self.op {
            NormOp::Norm => {
                w[4] = self.has(Y32);
                w[5] = self.has(Y16);
                w[7] = self.has(ADD);
            }
            NormOp::NormBackward | NormOp::QkNormRopeBackward => {
                w[4] = true;
                w[5] = true;
            }
            NormOp::QkNormRope => w[4] = true,
        }
        w
    }
    pub fn dispatch_groups(&self) -> [u32; 3] {
        self.dispatch_groups
    }
    pub fn push_constants(&self) -> [u8; PUSH_BYTES] {
        let words = [
            self.op as u32,
            self.flags,
            self.rows,
            self.d,
            self.eps.to_bits(),
            self.t,
            self.h,
            self.rows_per_group,
            self.dispatch_groups[0],
        ];
        let mut bytes = [0u8; PUSH_BYTES];
        for (i, w) in words.iter().enumerate() {
            bytes[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
        }
        bytes
    }
}

/// The law in f64, for the tests: each operation's arithmetic with the kernel's
/// fp16 roundings at the kernel's sites. Beside each output it gives a
/// magnitude, the sum of the absolute values of the terms that make it, which
/// the kernel's f32 arithmetic is held to relative to (its sums run in another
/// order, and Vulkan allows `inversesqrt` 2 ULP and a division 2.5 ULP). Where
/// the kernel rounds to fp16, its f32 value and this f64 one can fall either
/// side of a rounding boundary, so a rounded value may differ by one fp16 ulp.
pub mod reference {
    use crate::head_loss_ops::{f16_to_f32, f32_to_f16};

    fn round16(v: f64) -> f64 {
        f64::from(f16_to_f32(f32_to_f16(v as f32)))
    }

    fn rsqrt(row: &[f32], eps: f32) -> f64 {
        let sum_sq: f64 = row.iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
        1.0 / (sum_sq / row.len() as f64 + f64::from(eps)).sqrt()
    }

    /// The row of the q/k output that holds input row `row` = (b, t, h) of
    /// (B, T, H, D): (B, H, T, D), or the input's own with `same_layout`.
    pub fn out_row(row: usize, t: usize, h: usize, same_layout: bool) -> usize {
        if same_layout {
            return row;
        }
        let (hh, bt) = (row % h, row / h);
        let (tt, b) = (bt % t, bt / t);
        (b * h + hh) * t + tt
    }

    /// Op 0: `(h, y)`. `h = x + delta` is one f32 add (the kernel's, bit for
    /// bit), or `x`. `y = w * T(h * r)` with `T` the fp16 rounding when `narrow`
    /// (X16). Its magnitude is `|y|`: no term cancels.
    pub fn norm(
        x: &[f32],
        delta: Option<&[f32]>,
        w: &[f32],
        d: usize,
        eps: f32,
        narrow: bool,
    ) -> (Vec<f32>, Vec<f64>) {
        let h: Vec<f32> = match delta {
            Some(dl) => x.iter().zip(dl).map(|(a, b)| a + b).collect(),
            None => x.to_vec(),
        };
        let mut y = vec![0.0; h.len()];
        for (row, hr) in h.chunks_exact(d).enumerate() {
            let r = rsqrt(hr, eps);
            for c in 0..d {
                let n = f64::from(hr[c]) * r;
                y[row * d + c] = f64::from(w[c]) * if narrow { round16(n) } else { n };
            }
        }
        (h, y)
    }

    /// Op 1: `(dx, dx_magnitude, dw, dw_magnitude)`, `dw` one row of `d` a
    /// group of `rows_per_group` rows. The upstream gradient is `dy (+ dy16)`,
    /// one f32 add as the kernel adds it, or either alone (DY_ABSENT: `dy`
    /// None).
    #[allow(clippy::too_many_arguments)]
    pub fn norm_backward(
        h: &[f32],
        dy: Option<&[f32]>,
        dy16: Option<&[f32]>,
        dh: Option<&[f32]>,
        w: &[f32],
        d: usize,
        eps: f32,
        rows_per_group: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let rows = h.len() / d;
        let groups = rows.div_ceil(rows_per_group);
        let (mut dx, mut sx) = (vec![0.0; h.len()], vec![0.0; h.len()]);
        let (mut dw, mut sw) = (vec![0.0; groups * d], vec![0.0; groups * d]);
        for row in 0..rows {
            let base = row * d;
            let hr = &h[base..base + d];
            let up: Vec<f64> = (0..d)
                .map(|c| {
                    let i = base + c;
                    f64::from(match (dy, dy16) {
                        (Some(a), Some(e)) => a[i] + e[i],
                        (Some(a), None) => a[i],
                        (None, Some(e)) => e[i],
                        (None, None) => 0.0,
                    })
                })
                .collect();
            let g: Vec<f64> = (0..d).map(|c| f64::from(w[c]) * up[c]).collect();
            let r = rsqrt(hr, eps);
            let r3 = r * r * r;
            let hg: f64 = (0..d).map(|c| f64::from(hr[c]) * g[c]).sum();
            let hg_abs: f64 = (0..d).map(|c| (f64::from(hr[c]) * g[c]).abs()).sum();
            let group = row / rows_per_group;
            for c in 0..d {
                let (i, hv) = (base + c, f64::from(hr[c]));
                let e = dh.map_or(0.0, |v| f64::from(v[i]));
                dx[i] = e + (r * g[c] - hv * (r3 * (hg / d as f64)));
                sx[i] = e.abs() + (r * g[c]).abs() + hv.abs() * r3 * (hg_abs / d as f64);
                dw[group * d + c] += up[c] * (hv * r);
                sw[group * d + c] += (up[c] * (hv * r)).abs();
            }
        }
        (dx, sx, dw, sw)
    }

    /// Op 2: `(out, magnitude)` in the output's layout: `out = rope(w *
    /// f16(x * r))` before the kernel's final fp16 rounding. `cos` and `sin`
    /// are (T, D).
    #[allow(clippy::too_many_arguments)]
    pub fn qk_norm_rope(
        x: &[f32],
        w: &[f32],
        cos: &[f32],
        sin: &[f32],
        t: usize,
        h: usize,
        d: usize,
        eps: f32,
        same_layout: bool,
    ) -> (Vec<f64>, Vec<f64>) {
        let half = d / 2;
        let (mut out, mut mag) = (vec![0.0; x.len()], vec![0.0; x.len()]);
        for (row, xr) in x.chunks_exact(d).enumerate() {
            let r = rsqrt(xr, eps);
            let y: Vec<f64> = (0..d)
                .map(|c| f64::from(w[c]) * round16(f64::from(xr[c]) * r))
                .collect();
            let tt = (row / h) % t;
            let o = out_row(row, t, h, same_layout) * d;
            for c in 0..d {
                let rot = if c < half { -y[c + half] } else { y[c - half] };
                let m0 = y[c] * f64::from(cos[tt * d + c]);
                let m1 = rot * f64::from(sin[tt * d + c]);
                out[o + c] = m0 + m1;
                mag[o + c] = m0.abs() + m1.abs();
            }
        }
        (out, mag)
    }

    /// Op 3: `(dx, dx_magnitude, dw, dw_magnitude)`, `dx` before the kernel's
    /// final fp16 rounding, from `dout` in the output's layout. The rope's
    /// backward can cancel, so each magnitude carries its terms' absolute
    /// values through `dn = f16(w * gy)` and the norm's backward.
    #[allow(clippy::too_many_arguments)]
    pub fn qk_norm_rope_backward(
        x: &[f32],
        dout: &[f32],
        w: &[f32],
        cos: &[f32],
        sin: &[f32],
        t: usize,
        h: usize,
        d: usize,
        eps: f32,
        same_layout: bool,
        rows_per_group: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let half = d / 2;
        let rows = x.len() / d;
        let groups = rows.div_ceil(rows_per_group);
        let (mut dx, mut sx) = (vec![0.0; x.len()], vec![0.0; x.len()]);
        let (mut dw, mut sw) = (vec![0.0; groups * d], vec![0.0; groups * d]);
        for row in 0..rows {
            let base = row * d;
            let xr = &x[base..base + d];
            let o = out_row(row, t, h, same_layout) * d;
            let g = |c: usize| f64::from(dout[o + c]);
            let tt = (row / h) % t;
            let (cs, sn) = (&cos[tt * d..tt * d + d], &sin[tt * d..tt * d + d]);
            let mut gy = vec![0.0; d];
            let mut gy_mag = vec![0.0; d];
            for c in 0..d {
                let m0 = g(c) * f64::from(cs[c]);
                let m1 = if c < half {
                    g(c + half) * f64::from(sn[c + half])
                } else {
                    -g(c - half) * f64::from(sn[c - half])
                };
                gy[c] = m0 + m1;
                gy_mag[c] = m0.abs() + m1.abs();
            }
            let r = rsqrt(xr, eps);
            let r3 = r * r * r;
            let dn: Vec<f64> = (0..d).map(|c| round16(f64::from(w[c]) * gy[c])).collect();
            let dn_mag: Vec<f64> = (0..d)
                .map(|c| f64::from(w[c]).abs() * gy_mag[c] + dn[c].abs())
                .collect();
            let xd: f64 = (0..d).map(|c| f64::from(xr[c]) * dn[c]).sum();
            let xd_mag: f64 = (0..d).map(|c| f64::from(xr[c]).abs() * dn_mag[c]).sum();
            let group = row / rows_per_group;
            for c in 0..d {
                let xv = f64::from(xr[c]);
                dx[base + c] = r * dn[c] - xv * (r3 * (xd / d as f64));
                sx[base + c] = r * dn_mag[c] + xv.abs() * r3 * (xd_mag / d as f64);
                let n16 = round16(xv * r);
                dw[group * d + c] += gy[c] * n16;
                sw[group * d + c] += gy_mag[c] * n16.abs();
            }
        }
        (dx, sx, dw, sw)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refusals_are_named() {
        let named = |r: Result<NormPlan, NormPlanError>| r.unwrap_err().to_string();
        let n = NormOp::Norm;
        assert_eq!(
            named(NormPlan::new(n, Y32, 0, 16, 1e-6, 1, 1, 1)),
            "student-norm-dimension-out-of-range"
        );
        assert_eq!(
            named(NormPlan::new(n, Y32, 4, 15, 1e-6, 1, 1, 1)),
            "student-norm-odd-width"
        );
        assert_eq!(
            named(NormPlan::new(n, Y32, 4, 8192, 1e-6, 1, 1, 1)),
            "student-norm-width-above-limit"
        );
        assert_eq!(
            named(NormPlan::new(NormOp::QkNormRope, 0, 4, 512, 1e-6, 2, 2, 1)),
            "student-norm-width-above-limit"
        );
        assert_eq!(
            named(NormPlan::new(n, 0, 4, 16, 1e-6, 1, 1, 1)),
            "student-norm-flags-invalid"
        );
        assert_eq!(
            named(NormPlan::new(n, Y32 | DH, 4, 16, 1e-6, 1, 1, 1)),
            "student-norm-flags-invalid"
        );
        assert_eq!(
            named(NormPlan::new(n, Y32, 4, 16, 0.0, 1, 1, 1)),
            "student-norm-epsilon-invalid"
        );
        assert_eq!(
            named(NormPlan::new(NormOp::QkNormRope, 0, 6, 128, 1e-6, 4, 1, 1)),
            "student-norm-positions-invalid"
        );
        assert_eq!(
            named(NormPlan::new(NormOp::NormBackward, 0, 4, 16, 1e-6, 1, 1, 0)),
            "student-norm-rows-per-group-out-of-range"
        );
        // no upstream gradient at all: DY_ABSENT needs the fp16 one
        assert_eq!(
            named(NormPlan::new(
                NormOp::NormBackward,
                DY_ABSENT,
                4,
                16,
                1e-6,
                1,
                1,
                1
            )),
            "student-norm-flags-invalid"
        );
        assert_eq!(
            named(NormPlan::new(n, Y32 | DY_ABSENT, 4, 16, 1e-6, 1, 1, 1)),
            "student-norm-flags-invalid"
        );
        assert_eq!(NormOp::try_from(4), Err(NormPlanError::OperationOutOfRange));
    }

    #[test]
    fn the_roles_the_grid_and_the_push_block_follow_the_plan() {
        let add = NormPlan::new(
            NormOp::Norm,
            ADD | AUX16 | Y32 | Y16,
            8192,
            2048,
            1e-6,
            1,
            1,
            1,
        )
        .unwrap();
        assert_eq!(
            add.role_bytes(),
            [
                8192 * 2048 * 4,
                8192 * 2048 * 2,
                2048 * 4,
                0,
                8192 * 2048 * 4,
                8192 * 2048 * 2,
                0,
                8192 * 2048 * 4
            ]
        );
        assert_eq!(
            add.writes(),
            [false, false, false, false, true, true, false, true]
        );
        assert_eq!(add.dispatch_groups(), [8192, 1, 1]);
        let back =
            NormPlan::new(NormOp::NormBackward, DY16 | DH, 8192, 2048, 1e-6, 1, 1, 64).unwrap();
        assert_eq!(back.groups(), 128);
        assert_eq!(back.role_bytes()[5], 128 * 2048 * 4);
        assert_eq!(back.role_bytes()[6], 8192 * 2048 * 4);
        let fp16_only = NormPlan::new(
            NormOp::NormBackward,
            DY16 | DY_ABSENT,
            8192,
            2048,
            1e-6,
            1,
            1,
            64,
        )
        .unwrap();
        assert_eq!(fp16_only.role_bytes()[6], 0);
        assert_eq!(fp16_only.role_bytes()[1], 8192 * 2048 * 2);
        let qk =
            NormPlan::new(NormOp::QkNormRope, 0, 4 * 2048 * 32, 128, 1e-6, 2048, 32, 1).unwrap();
        assert_eq!(qk.role_bytes()[3], 2 * 2048 * 128 * 4);
        let [x, y, _] = qk.dispatch_groups();
        assert_eq!(x, MAX_GROUPS_X);
        assert!(x as u64 * y as u64 >= 4 * 2048 * 32);
        let push = back.push_constants();
        let word = |i: usize| u32::from_le_bytes(push[i * 4..i * 4 + 4].try_into().unwrap());
        assert_eq!(
            (0..4).map(word).collect::<Vec<_>>(),
            vec![1, DY16 | DH, 8192, 2048]
        );
        assert_eq!(f32::from_bits(word(4)), 1e-6);
        assert_eq!(word(7), 64);
    }
}
