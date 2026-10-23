//! weighted-bag/v1: the weighted sum of gathered table rows and its two gradients, in FP32
//! (kernels/weighted_bag.comp).
//!
//! `out[r, c] = sum_{s < k} w[r, s] * table[idx[r, s], c]`: a product-key memory's read, an `embedding_bag`
//! with per-sample weights and mode "sum", without materialising the `[rows, k, d]` gather.
//!
//! - The forward is one dispatch.
//! - The weights' gradient (`grad_w[r, s] = up[r] . table[idx[r, s]]`) is one dispatch.
//! - The table's gradient (`grad_t[v] = sum over pairs naming v of w[p] * up[p / k]`) is six: zero, count, scan,
//!   place, sort, sum. It builds a CSR of the indices on the device and sums each row's pairs in position order, so
//!   it needs no index readback and no floating-point atomics. Every table row is written; a row no pair names
//!   gets zeros.

use crate::{Access, Buffer, Context, Kernel, Result};

pub const ABI: &str = "weighted-bag/v1";
pub const WORKGROUP: u32 = 64;
pub const PUSH_BYTES: usize = 32;
/// Pairs (`rows * k`) a call takes.
pub const MAX_PAIRS: u64 = 1 << 24;
pub const MAX_VOCAB: u64 = 1 << 22;
pub const MAX_DIM: u64 = 16_384;
/// Elements of the output and of the table.
pub const MAX_ELEMENTS: u64 = 1 << 30;
pub const MAX_GROUPS_X: u32 = 65_535;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightedBagOp {
    Forward = 0,
    GradWeights = 1,
    GradTable = 2,
}

impl TryFrom<u32> for WeightedBagOp {
    type Error = String;
    fn try_from(v: u32) -> std::result::Result<Self, String> {
        match v {
            0 => Ok(Self::Forward),
            1 => Ok(Self::GradWeights),
            2 => Ok(Self::GradTable),
            _ => Err(format!("weighted-bag-operation-out-of-range ({v})")),
        }
    }
}

/// One validated call: its geometry and the buffer sizes each role needs (in 4-byte words).
#[derive(Clone, Copy, Debug)]
pub struct WeightedBagPlan {
    pub rows: u32,
    pub k: u32,
    pub vocab: u32,
    pub d: u32,
    pub op: WeightedBagOp,
}

impl WeightedBagPlan {
    pub fn new(
        rows: u32,
        k: u32,
        vocab: u32,
        d: u32,
        op: WeightedBagOp,
    ) -> std::result::Result<Self, String> {
        let (r, kk, v, dd) = (
            u64::from(rows),
            u64::from(k),
            u64::from(vocab),
            u64::from(d),
        );
        if r == 0 || kk == 0 || v == 0 || dd == 0 {
            return Err("weighted-bag-dimension-out-of-range".into());
        }
        if r * kk > MAX_PAIRS
            || v > MAX_VOCAB
            || dd > MAX_DIM
            || r * dd > MAX_ELEMENTS
            || v * dd > MAX_ELEMENTS
        {
            return Err("weighted-bag-element-limit-exceeded".into());
        }
        Ok(Self {
            rows,
            k,
            vocab,
            d,
            op,
        })
    }

    pub fn pairs(&self) -> u64 {
        u64::from(self.rows) * u64::from(self.k)
    }

    pub fn table_words(&self) -> u64 {
        u64::from(self.vocab) * u64::from(self.d)
    }

    pub fn rows_words(&self) -> u64 {
        u64::from(self.rows) * u64::from(self.d)
    }

    /// Words the output role needs: the forward's rows, the weights' gradient (one a pair), the table's gradient.
    pub fn output_words(&self) -> u64 {
        match self.op {
            WeightedBagOp::Forward => self.rows_words(),
            WeightedBagOp::GradWeights => self.pairs(),
            WeightedBagOp::GradTable => self.table_words(),
        }
    }

    /// Words of scratch the table's gradient needs (counts, offsets, cursor: vocab + 1 each; the pair list).
    pub fn scratch_words(&self) -> u64 {
        3 * (u64::from(self.vocab) + 1) + self.pairs()
    }

    fn push(&self, op: u32, groups_x: u32) -> [u8; PUSH_BYTES] {
        let mut out = [0u8; PUSH_BYTES];
        for (i, v) in [self.rows, self.k, self.vocab, self.d, op, groups_x, 0, 0]
            .iter()
            .enumerate()
        {
            out[4 * i..4 * i + 4].copy_from_slice(&v.to_le_bytes());
        }
        out
    }
}

/// A 2-D grid of `groups` workgroups: x at most `MAX_GROUPS_X`.
fn grid(groups: u64) -> [u32; 3] {
    let g = groups.max(1);
    let x = g.min(u64::from(MAX_GROUPS_X));
    [x as u32, g.div_ceil(x) as u32, 1]
}

/// The buffers of one call. `up` is read by the gradients only and `scratch` (u32 words) by the table's gradient
/// only: another call may pass any live buffer in either slot.
pub struct WeightedBagBuffers<'a> {
    pub table: &'a Buffer,
    pub idx: &'a Buffer,
    pub w: &'a Buffer,
    pub up: &'a Buffer,
    pub out: &'a Buffer,
    pub scratch: &'a Buffer,
}

/// Record the plan's dispatches on `kernel` (weighted_bag.spv, 6 storage buffers, `PUSH_BYTES`). The caller has
/// checked every buffer's size against the plan and that `out` aliases no input.
pub fn record(
    ctx: &Context,
    kernel: &Kernel,
    plan: &WeightedBagPlan,
    b: &WeightedBagBuffers<'_>,
) -> Result<()> {
    ctx.bind(kernel, &[b.table, b.idx, b.w, b.up, b.out, b.scratch])?;
    let wg = u64::from(WORKGROUP);
    // one declaration a bound slot, in binding order
    let run = |op: u32, groups: u64, out_written: bool, scratch_written: bool| -> Result<()> {
        let g = grid(groups);
        let rw = |id: u64, written: bool| {
            if written {
                Access::write(id)
            } else {
                Access::read(id)
            }
        };
        let access = [
            Access::read(b.table.id()),
            Access::read(b.idx.id()),
            Access::read(b.w.id()),
            Access::read(b.up.id()),
            rw(b.out.id(), out_written),
            rw(b.scratch.id(), scratch_written),
        ];
        ctx.dispatch(kernel, &plan.push(op, g[0]), g, 1, &access)
    };
    match plan.op {
        WeightedBagOp::Forward => run(0, plan.rows_words().div_ceil(wg), true, false),
        WeightedBagOp::GradWeights => run(1, plan.pairs(), true, false),
        WeightedBagOp::GradTable => {
            let v1 = u64::from(plan.vocab) + 1;
            run(2, v1.div_ceil(wg), false, true)?;
            run(3, plan.pairs().div_ceil(wg), false, true)?;
            run(4, 1, false, true)?;
            run(5, plan.pairs().div_ceil(wg), false, true)?;
            run(6, u64::from(plan.vocab).div_ceil(wg), false, true)?;
            run(7, plan.table_words().div_ceil(wg), true, false)
        }
    }
}

/// The host law, in f64 then rounded: (out, grad_w, grad_table) for `up`.
pub fn reference(
    table: &[f32],
    idx: &[u32],
    w: &[f32],
    up: &[f32],
    rows: usize,
    k: usize,
    d: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let vocab = table.len() / d;
    let mut out = vec![0f32; rows * d];
    let mut gw = vec![0f32; rows * k];
    let mut gt = vec![0f64; vocab * d];
    for r in 0..rows {
        for c in 0..d {
            let mut acc = 0f64;
            for s in 0..k {
                let p = r * k + s;
                acc += f64::from(w[p]) * f64::from(table[idx[p] as usize * d + c]);
            }
            out[r * d + c] = acc as f32;
        }
        for s in 0..k {
            let p = r * k + s;
            let v = idx[p] as usize;
            let mut acc = 0f64;
            for c in 0..d {
                acc += f64::from(up[r * d + c]) * f64::from(table[v * d + c]);
                gt[v * d + c] += f64::from(w[p]) * f64::from(up[r * d + c]);
            }
            gw[p] = acc as f32;
        }
    }
    (out, gw, gt.into_iter().map(|x| x as f32).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_plan_refuses_by_name() {
        assert!(
            WeightedBagPlan::new(0, 32, 65_536, 512, WeightedBagOp::Forward)
                .unwrap_err()
                .contains("dimension")
        );
        assert!(
            WeightedBagPlan::new(1 << 20, 32, 65_536, 512, WeightedBagOp::Forward)
                .unwrap_err()
                .contains("limit")
        );
        let p = WeightedBagPlan::new(32_768, 32, 65_536, 512, WeightedBagOp::GradTable).unwrap();
        assert_eq!(p.pairs(), 1 << 20);
        assert_eq!(p.output_words(), 65_536 * 512);
        assert_eq!(p.scratch_words(), 3 * 65_537 + (1 << 20));
        assert!(WeightedBagOp::try_from(3).is_err());
    }

    #[test]
    fn the_grid_covers_every_workgroup_with_x_bounded() {
        for g in [1u64, 64, 65_535, 65_536, 1 << 24] {
            let [x, y, _] = grid(g);
            assert!(x <= MAX_GROUPS_X && u64::from(x) * u64::from(y) >= g, "{g}");
        }
    }
}
