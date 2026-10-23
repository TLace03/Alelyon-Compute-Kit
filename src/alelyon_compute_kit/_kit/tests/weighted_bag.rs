//! weighted-bag/v1 on the device against its f64 host law (`weighted_bag_ops::reference`), and the table's
//! gradient twice for the same bits (its CSR is built with integer atomics, then sorted, so its order is fixed).

use alelyon_compute_kit::weighted_bag_ops::{
    self as wb, WeightedBagBuffers, WeightedBagOp, WeightedBagPlan,
};
use alelyon_compute_kit::{Buffer, Context};

const SPV: &[u8] = include_bytes!("../kernels/weighted_bag.spv");

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn unit(&mut self) -> f32 {
        (self.next() % 1_000_000) as f32 / 1_000_000.0
    }
}

fn open_or_unmeasured(what: &str) -> Option<Context> {
    match Context::open() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("UNMEASURED: {what}: no device ({e})");
            None
        }
    }
}

fn upload_words(ctx: &Context, words: &[u32]) -> Buffer {
    let b = ctx.buffer((words.len().max(1) * 4) as u64).unwrap();
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    ctx.upload(&b, &bytes).unwrap();
    b
}

fn f32s(ctx: &Context, b: &Buffer, n: usize) -> Vec<f32> {
    ctx.download(b).unwrap()[..n * 4]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn close(got: f32, want: f32, scale: f32) -> bool {
    (got - want).abs() <= 2e-6 * scale + 1e-30
}

#[test]
fn weighted_bag_matches_its_host_law_and_repeats_its_table_gradient_bit_for_bit() {
    let Some(ctx) = open_or_unmeasured("weighted bag") else {
        return;
    };
    let kernel = ctx.kernel(SPV, 6, wb::PUSH_BYTES as u32).unwrap();
    // a skewed index distribution: repeated rows, rows no pair names, and one row many pairs name
    for &(rows, k, vocab, d) in &[
        (37usize, 5usize, 64usize, 24usize),
        (300, 32, 4096, 128),
        (64, 32, 65_536, 512),
    ] {
        let mut rng = Lcg(rows as u64 * 31 + d as u64);
        let table: Vec<f32> = (0..vocab * d).map(|_| rng.unit() * 2.0 - 1.0).collect();
        let idx: Vec<u32> = (0..rows * k)
            .map(|i| {
                if i % 7 == 0 {
                    3
                } else {
                    (rng.next() % (vocab as u64 / 2)) as u32
                }
            })
            .collect();
        let w: Vec<f32> = (0..rows * k).map(|_| rng.unit()).collect();
        let up: Vec<f32> = (0..rows * d).map(|_| rng.unit() * 2.0 - 1.0).collect();
        let (want_out, want_gw, want_gt) = wb::reference(&table, &idx, &w, &up, rows, k, d);
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
        let b_table = upload_words(&ctx, &bits(&table));
        let b_idx = upload_words(&ctx, &idx);
        let b_w = upload_words(&ctx, &bits(&w));
        let b_up = upload_words(&ctx, &bits(&up));
        let mut results = Vec::new();
        for (op, n) in [
            (WeightedBagOp::Forward, rows * d),
            (WeightedBagOp::GradWeights, rows * k),
            (WeightedBagOp::GradTable, vocab * d),
            (WeightedBagOp::GradTable, vocab * d),
        ] {
            let plan =
                WeightedBagPlan::new(rows as u32, k as u32, vocab as u32, d as u32, op).unwrap();
            let out = ctx.buffer(plan.output_words() * 4).unwrap();
            let scratch = ctx.buffer(plan.scratch_words() * 4).unwrap();
            let bufs = WeightedBagBuffers {
                table: &b_table,
                idx: &b_idx,
                w: &b_w,
                up: &b_up,
                out: &out,
                scratch: &scratch,
            };
            wb::record(&ctx, &kernel, &plan, &bufs).unwrap();
            ctx.sync().unwrap();
            results.push(f32s(&ctx, &out, n));
            ctx.destroy_buffer(out);
            ctx.destroy_buffer(scratch);
        }
        let check = |name: &str, got: &[f32], want: &[f32], scale: f32| {
            let bad = got
                .iter()
                .zip(want)
                .filter(|(g, w)| !close(**g, **w, scale))
                .count();
            assert_eq!(
                bad,
                0,
                "{name} at rows {rows} k {k} vocab {vocab} d {d}: {bad} of {} off",
                want.len()
            );
        };
        check("forward", &results[0], &want_out, k as f32);
        check("weights' gradient", &results[1], &want_gw, d as f32);
        // the busiest row sums about rows * k / 7 terms
        check("table's gradient", &results[2], &want_gt, (rows * k) as f32);
        assert_eq!(
            bits(&results[2]),
            bits(&results[3]),
            "the table's gradient repeats bit for bit"
        );
        // a row no pair names is exactly zero
        let named: std::collections::HashSet<u32> = idx.iter().copied().collect();
        let unnamed = (0..vocab as u32).find(|v| !named.contains(v)).unwrap();
        assert!(results[2][unnamed as usize * d..(unnamed as usize + 1) * d]
            .iter()
            .all(|x| x.to_bits() == 0));
        for b in [b_table, b_idx, b_w, b_up] {
            ctx.destroy_buffer(b);
        }
    }
}
