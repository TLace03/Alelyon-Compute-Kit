//! attention-v2 forward on the device (2026-09-27), against an fp64 reference
//! computed from the SAME bf16-rounded inputs, at head dimensions 64 and 128.
//!
//! The kernel keeps the running output in cooperative-matrix accumulators and
//! rescales it by a component-wise product with a loaded per-row matrix, so the
//! first thing these tests establish is that the product lands on the right
//! rows on this vendor: a wrong mapping would scale row r by another row's
//! alpha, which the fp64 comparison sees at once.
//!
//! Shapes: one block; tail query rows and keys; grouped kv heads; causal off; a
//! `[B, T, H, D]` view; and operands whose row strides are not multiples of 8
//! elements, which takes the kernel's 4-byte staging path instead of its
//! 16-byte one. Then the AKV cell's geometry (B4 H8 KV4 T1024 D64) and the
//! stage-E student's (B4 H32 KV4 T2048 D128), checked on sampled heads and
//! timed. (v1 was timed beside v2 here until it was retired on 2026-09-27: at
//! the AKV shape on an RX 9070 XT, v1 0.562 ms forward, 0.520 dQ, 0.675 dK/dV;
//! v2 0.161, 0.150, 0.232.)
//!
//! The binary16 modules (attention-f16, 2026-09-29: the same sources built
//! with -DELT_F16=1) run every test here too, against the fp64 reference
//! computed from the fp16-rounded inputs, under the same laws scaled by the
//! element's unit roundoff relative to bf16's (2^-11 / 2^-8 = 1/8) for mean
//! errors and by 1/4 for the worst element (see `Elt::worst_scale`).
//!
//! The binary16 modules with the device's own conversions (-DF16_HW=1,
//! 2026-09-29: round to nearest even and denormal preservation declared for
//! 16-bit floats) run the same laws as `Elt::F16Hw`, and are held BIT FOR BIT
//! to the integer modules on the same inputs, including scores wide enough
//! that most probabilities are binary16 subnormals, outputs in the subnormal
//! range and near binary16's largest finite value, and the stage-E step's own
//! geometry. A device that does not report both float controls prints
//! UNMEASURED for those. Both sets carry NoContraction on every float result
//! (kernels/no_contraction.py, 2026-09-30). Without it this comparison FAILED
//! on its first device run, D64's O differing in 1 of 102,400 elements by one
//! ULP: the driver contracted the two sets' identical f32 arithmetic
//! differently.
//!
//! Needs a Vulkan device with BF16 (and, for the f16 tests, FP16) cooperative
//! matrices and subgroup size control; without one every test prints UNMEASURED and returns (set
//! `ACK_REQUIRE_DEVICE=1` to make that a failure). A GREEN RUN WITHOUT THAT
//! VARIABLE IS NOT DEVICE EVIDENCE.

use alelyon_compute_kit::attention_fused::{
    AttentionPlan, AttentionShape, Strides, OPERANDS, PUSH_BYTES,
};
use alelyon_compute_kit::{Access, Buffer, Context, Kernel};

const V2: [(u32, u32, &[u8]); 4] = [
    (
        64,
        64,
        include_bytes!("../kernels/attention_fwd_v2_d64_s64.spv"),
    ),
    (
        64,
        32,
        include_bytes!("../kernels/attention_fwd_v2_d64_s32.spv"),
    ),
    (
        128,
        64,
        include_bytes!("../kernels/attention_fwd_v2_d128_s64.spv"),
    ),
    (
        128,
        32,
        include_bytes!("../kernels/attention_fwd_v2_d128_s32.spv"),
    ),
];

fn open_or_unmeasured(what: &str) -> Option<Context> {
    match Context::open() {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            if std::env::var_os("ACK_REQUIRE_DEVICE").is_some() {
                panic!("{what}: ACK_REQUIRE_DEVICE is set and no device opened: {e}");
            }
            eprintln!("UNMEASURED here ({what}): {e}");
            None
        }
    }
}

/// Every width the device can require, widest first (a device that takes both
/// runs both modules: the 32-lane one is NVIDIA's, and is checked here too).
fn widths(ctx: &Context) -> Vec<u32> {
    let w: Vec<u32> = [64u32, 32]
        .into_iter()
        .filter(|&w| ctx.can_require_subgroup_size(w))
        .collect();
    if w.is_empty() {
        if std::env::var_os("ACK_REQUIRE_DEVICE").is_some() {
            panic!("ACK_REQUIRE_DEVICE is set and the device can require neither 64 nor 32 lanes");
        }
        eprintln!("UNMEASURED here: the device can require neither 64 nor 32 lanes");
    }
    w
}

fn width(ctx: &Context) -> Option<u32> {
    widths(ctx).first().copied()
}

/// Whether this device may run the `-DF16_HW=1` modules: it reports
/// shaderFloat16, shaderRoundingModeRTEFloat16 and shaderDenormPreserveFloat16,
/// the rule `ack_attention_f16_fwd` builds them by. UNMEASURED otherwise, and
/// not a failure: the integer modules are that device's fp16 attention.
fn hardware_f16(ctx: &Context, what: &str) -> bool {
    let r = &ctx.report;
    let ok = r.shader_float16 && r.f16_rounding_rte && r.f16_denorm_preserve;
    if !ok {
        eprintln!(
            "UNMEASURED here ({what}): the device does not report fp16 round-to-nearest-even and denormal preservation"
        );
    }
    ok
}

const V2_F16: [(u32, u32, &[u8]); 4] = [
    (
        64,
        64,
        include_bytes!("../kernels/attention_fwd_v2_f16_d64_s64.spv"),
    ),
    (
        64,
        32,
        include_bytes!("../kernels/attention_fwd_v2_f16_d64_s32.spv"),
    ),
    (
        128,
        64,
        include_bytes!("../kernels/attention_fwd_v2_f16_d128_s64.spv"),
    ),
    (
        128,
        32,
        include_bytes!("../kernels/attention_fwd_v2_f16_d128_s32.spv"),
    ),
];

const V2_F16HW: [(u32, u32, &[u8]); 4] = [
    (
        64,
        64,
        include_bytes!("../kernels/attention_fwd_v2_f16hw_d64_s64.spv"),
    ),
    (
        64,
        32,
        include_bytes!("../kernels/attention_fwd_v2_f16hw_d64_s32.spv"),
    ),
    (
        128,
        64,
        include_bytes!("../kernels/attention_fwd_v2_f16hw_d128_s64.spv"),
    ),
    (
        128,
        32,
        include_bytes!("../kernels/attention_fwd_v2_f16hw_d128_s32.spv"),
    ),
];

/// The element types the modules are built for; `F16Hw` is binary16 with the
/// device's own conversions, the same numbers as `F16` by construction.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Elt {
    Bf16,
    F16,
    F16Hw,
}

impl Elt {
    fn round(self, x: f32) -> u16 {
        match self {
            Elt::Bf16 => bf16(x),
            Elt::F16 | Elt::F16Hw => f16(x),
        }
    }
    fn value(self, h: u16) -> f32 {
        match self {
            Elt::Bf16 => from_bf16(h),
            Elt::F16 | Elt::F16Hw => from_f16(h),
        }
    }
    /// The mean-error laws below scale by the element's unit roundoff over
    /// bf16's (2^-11 / 2^-8 = 1/8; the stage-E backward's mean errors measured
    /// 8.0x below bf16's). The WORST-element laws scale by 1/4: a probability or
    /// dS below 2^-14 is a binary16 subnormal whose precision is absolute
    /// (2^-25), not relative, so the worst element of a long sequence improves
    /// by less than the ratio (measured 5.9-8.2x; the first run's 1/8 failed at
    /// the stage-E backward's dV, 0.0342 of the mean magnitude against 0.03125).
    fn law_scale(self) -> f64 {
        match self {
            Elt::Bf16 => 1.0,
            Elt::F16 | Elt::F16Hw => 0.125,
        }
    }
    fn worst_scale(self) -> f64 {
        match self {
            Elt::Bf16 => 1.0,
            Elt::F16 | Elt::F16Hw => 0.25,
        }
    }
}

fn v2_kernel_elt(ctx: &Context, d: u32, w: u32, elt: Elt) -> Kernel {
    let table = match elt {
        Elt::Bf16 => &V2,
        Elt::F16 => &V2_F16,
        Elt::F16Hw => &V2_F16HW,
    };
    let spv = table
        .iter()
        .find(|(dd, ww, _)| *dd == d && *ww == w)
        .expect("module")
        .2;
    ctx.kernel_with_subgroup(spv, OPERANDS as u32, PUSH_BYTES as u32, Some(w))
        .expect("attention-v2 kernel")
}

fn bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    let lsb = (bits >> 16) & 1;
    ((bits + 0x7FFF + lsb) >> 16) as u16
}

fn from_bf16(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

/// binary32 -> binary16, round to nearest even, subnormals kept, an overflow to
/// infinity (tests/attention_abi.rs holds this routine to every binary16).
fn f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let a = bits & 0x7FFF_FFFF;
    if a >= 0x7F80_0000 {
        let nan = if a > 0x7F80_0000 {
            0x0200 | ((a >> 13) & 0x03FF) as u16
        } else {
            0
        };
        return sign | 0x7C00 | nan;
    }
    if a >= 0x477F_F000 {
        return sign | 0x7C00;
    }
    if a >= 0x3880_0000 {
        let r = a - 0x3800_0000;
        let mut h = r >> 13;
        let rest = r & 0x1FFF;
        if rest > 0x1000 || (rest == 0x1000 && h & 1 == 1) {
            h += 1;
        }
        return sign | h as u16;
    }
    if a < 0x3300_0000 {
        return sign;
    }
    let e = a >> 23;
    let m = (a & 0x7F_FFFF) | 0x80_0000;
    let shift = 126 - e;
    let mut h = m >> shift;
    let rest = m & ((1u32 << shift) - 1);
    let half = 1u32 << (shift - 1);
    if rest > half || (rest == half && h & 1 == 1) {
        h += 1;
    }
    sign | h as u16
}

/// binary16 -> binary32, exactly.
fn from_f16(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0f64 } else { 1.0 };
    let e = i32::from((h >> 10) & 0x1F);
    let m = f64::from(h & 0x03FF);
    let v = match e {
        0x1F if m == 0.0 => sign * f64::INFINITY,
        0x1F => f64::NAN,
        0 => sign * m * 2f64.powi(-24),
        _ => sign * (1.0 + m / 1024.0) * 2f64.powi(e - 15),
    };
    v as f32
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f32 {
        let mut s = 0.0f32;
        for _ in 0..4 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            s += ((self.0 >> 40) as f32) / (1u64 << 24) as f32;
        }
        (s - 2.0) * 1.7
    }
}

#[derive(Clone, Copy)]
enum Layout {
    Bhtd,
    Bthd,
    /// `[B, H, T, D + 2]`: row strides that are not multiples of 8 elements
    Padded,
}

#[derive(Clone, Copy)]
struct Case {
    b: u32,
    h: u32,
    hkv: u32,
    t: u32,
    d: u32,
    causal: bool,
    scale: f32,
    q_layout: Layout,
    kv_layout: Layout,
}

struct Operands {
    q: Vec<u16>,
    k: Vec<u16>,
    v: Vec<u16>,
    shape: AttentionShape,
    elt: Elt,
}

fn strides(l: Layout, heads: u64, seq: u64, d: u64) -> (Strides, usize) {
    match l {
        Layout::Bhtd => (
            Strides::dense_bhtd_d(heads, seq, d),
            (heads * seq * d) as usize,
        ),
        Layout::Bthd => (
            Strides::dense_bthd_d(heads, seq, d),
            (heads * seq * d) as usize,
        ),
        Layout::Padded => (
            Strides::dense_bhtd_d(heads, seq, d + 2),
            (heads * seq * (d + 2)) as usize,
        ),
    }
}

fn operands_elt(c: &Case, seed: u64, elt: Elt) -> Operands {
    let d = c.d as u64;
    let (qs, nq) = strides(c.q_layout, c.h as u64, c.t as u64, d);
    let (kvs, nkv) = strides(c.kv_layout, c.hkv as u64, c.t as u64, d);
    let mut rng = Lcg(seed);
    let mut fill = |n: usize| {
        (0..n * c.b as usize)
            .map(|_| elt.round(rng.next()))
            .collect::<Vec<u16>>()
    };
    let q = fill(nq);
    let k = fill(nkv);
    let v = fill(nkv);
    Operands {
        q,
        k,
        v,
        shape: AttentionShape {
            batch: c.b,
            heads: c.h,
            kv_heads: c.hkv,
            seq: c.t,
            head_dim: c.d,
            causal: c.causal,
            scale: c.scale,
            q: qs,
            k: kvs,
            v: kvs,
            o: qs,
        },
        elt,
    }
}

fn at(s: Strides, b: u32, h: u32, r: u32) -> usize {
    (b as u64 * s.batch + h as u64 * s.head + r as u64 * s.row) as usize
}

/// fp64 attention for one (batch, head): (o rows, lse).
fn reference(ops: &Operands, b: u32, h: u32) -> (Vec<Vec<f64>>, Vec<f64>) {
    let s = ops.shape;
    let d = s.head_dim as usize;
    let hk = h / (s.heads / s.kv_heads);
    let mut out = Vec::with_capacity(s.seq as usize);
    let mut lse = Vec::with_capacity(s.seq as usize);
    let kf: Vec<f64> = ops.k.iter().map(|&x| ops.elt.value(x) as f64).collect();
    let vf: Vec<f64> = ops.v.iter().map(|&x| ops.elt.value(x) as f64).collect();
    for i in 0..s.seq {
        let qi = at(s.q, b, h, i);
        let qrow: Vec<f64> = (0..d)
            .map(|e| ops.elt.value(ops.q[qi + e]) as f64)
            .collect();
        let last = if s.causal { i } else { s.seq - 1 };
        let mut scores = Vec::with_capacity(last as usize + 1);
        for j in 0..=last {
            let kj = at(s.k, b, hk, j);
            let dot: f64 = (0..d).map(|e| qrow[e] * kf[kj + e]).sum();
            scores.push(dot * s.scale as f64);
        }
        let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = scores.iter().map(|x| (x - m).exp()).collect();
        let l: f64 = weights.iter().sum();
        let mut row = vec![0.0f64; d];
        for (j, w) in weights.iter().enumerate() {
            let vj = at(s.v, b, hk, j as u32);
            for (e, r) in row.iter_mut().enumerate() {
                *r += w * vf[vj + e];
            }
        }
        for x in row.iter_mut() {
            *x /= l;
        }
        out.push(row);
        lse.push(m + l.ln());
    }
    (out, lse)
}

fn bytes_u16(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

struct Bound {
    q: Buffer,
    k: Buffer,
    v: Buffer,
    o: Buffer,
    lse: Buffer,
    plan: AttentionPlan,
    n_o: usize,
}

fn bind(ctx: &Context, kernel: &Kernel, ops: &Operands) -> Bound {
    let s = ops.shape;
    let n_o = ops.q.len();
    let n_lse = (s.batch * s.heads * s.seq) as usize;
    let plan = AttentionPlan::new(
        s,
        [
            ops.q.len() as u64,
            ops.k.len() as u64,
            ops.v.len() as u64,
            n_o as u64,
            n_lse as u64,
        ],
    )
    .expect("plan");
    let buf = |bytes: usize| ctx.buffer(bytes as u64).expect("buffer");
    let bound = Bound {
        q: buf(ops.q.len() * 2),
        k: buf(ops.k.len() * 2),
        v: buf(ops.v.len() * 2),
        o: buf(n_o * 2),
        lse: buf(n_lse * 4),
        plan,
        n_o,
    };
    ctx.upload(&bound.q, &bytes_u16(&ops.q)).expect("q");
    ctx.upload(&bound.k, &bytes_u16(&ops.k)).expect("k");
    ctx.upload(&bound.v, &bytes_u16(&ops.v)).expect("v");
    // poison o, so a row the kernel never writes cannot pass as zero
    ctx.upload(&bound.o, &vec![0xFFu8; n_o * 2])
        .expect("o poison");
    ctx.bind(
        kernel,
        &[&bound.q, &bound.k, &bound.v, &bound.o, &bound.lse],
    )
    .expect("bind");
    bound
}

fn run(ctx: &Context, kernel: &Kernel, bound: &Bound) -> (Vec<u16>, Vec<f32>) {
    let access = [
        Access::read(bound.q.id()),
        Access::read(bound.k.id()),
        Access::read(bound.v.id()),
        Access::write(bound.o.id()),
        Access::write(bound.lse.id()),
    ];
    ctx.dispatch(
        kernel,
        &bound.plan.push_constants(),
        bound.plan.dispatch_groups(),
        1,
        &access,
    )
    .expect("dispatch");
    let o16 = ctx
        .download(&bound.o)
        .expect("o")
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect::<Vec<u16>>();
    let lse = ctx
        .download(&bound.lse)
        .expect("lse")
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(o16.len(), bound.n_o);
    (o16, lse)
}

fn release(ctx: &Context, bound: Bound) {
    for b in [bound.q, bound.k, bound.v, bound.o, bound.lse] {
        ctx.destroy_buffer(b);
    }
}

/// Worst and mean |O error| and worst |LSE error| over the (b, h) pairs.
fn check(ops: &Operands, o: &[u16], lse: &[f32], pairs: &[(u32, u32)]) -> (f64, f64, f64) {
    let s = ops.shape;
    let d = s.head_dim as usize;
    let (mut worst_o, mut sum_o, mut n, mut worst_l) = (0.0f64, 0.0f64, 0usize, 0.0f64);
    for &(b, h) in pairs {
        let (want, want_lse) = reference(ops, b, h);
        for i in 0..s.seq {
            let oi = at(s.o, b, h, i);
            for e in 0..d {
                let got = ops.elt.value(o[oi + e]) as f64;
                let err = if got.is_finite() {
                    (got - want[i as usize][e]).abs()
                } else {
                    f64::INFINITY
                };
                worst_o = worst_o.max(err);
                sum_o += err;
                n += 1;
            }
            let li = ((b * s.heads + h) * s.seq + i) as usize;
            worst_l = worst_l.max((lse[li] as f64 - want_lse[i as usize]).abs());
        }
    }
    (worst_o, sum_o / n as f64, worst_l)
}

#[test]
fn the_v2_forward_matches_fp64_attention_at_both_head_dimensions() {
    forward_matches_fp64(Elt::Bf16);
}

#[test]
fn the_f16_forward_matches_fp64_attention_at_both_head_dimensions() {
    forward_matches_fp64(Elt::F16);
}

#[test]
fn the_f16hw_forward_matches_fp64_attention_at_both_head_dimensions() {
    forward_matches_fp64(Elt::F16Hw);
}

fn forward_matches_fp64(elt: Elt) {
    let Some(ctx) = open_or_unmeasured("attention-v2 forward") else {
        return;
    };
    if elt == Elt::F16Hw && !hardware_f16(&ctx, "attention-v2 f16hw") {
        return;
    }
    let (law, worst_law) = (elt.law_scale(), elt.worst_scale());
    for (w, d) in widths(&ctx)
        .into_iter()
        .flat_map(|w| [64u32, 128].map(move |d| (w, d)))
    {
        let kernel = v2_kernel_elt(&ctx, d, w, elt);
        let base = Case {
            b: 1,
            h: 2,
            hkv: 1,
            t: 64,
            d,
            causal: true,
            scale: 1.0 / (d as f32).sqrt(),
            q_layout: Layout::Bhtd,
            kv_layout: Layout::Bhtd,
        };
        let cases = [
            ("one block", base),
            (
                "tail rows and keys",
                Case {
                    b: 2,
                    h: 4,
                    hkv: 2,
                    t: 200,
                    ..base
                },
            ),
            (
                "causal off",
                Case {
                    h: 2,
                    hkv: 2,
                    t: 130,
                    causal: false,
                    scale: 0.2,
                    ..base
                },
            ),
            (
                "[B, T, H, D] view",
                Case {
                    b: 2,
                    h: 8,
                    hkv: 4,
                    t: 96,
                    scale: 1.0,
                    q_layout: Layout::Bthd,
                    ..base
                },
            ),
            (
                "4-byte staging",
                Case {
                    b: 2,
                    h: 4,
                    hkv: 2,
                    t: 150,
                    q_layout: Layout::Padded,
                    kv_layout: Layout::Padded,
                    ..base
                },
            ),
            (
                "4-byte k/v only",
                Case {
                    b: 1,
                    h: 4,
                    hkv: 1,
                    t: 77,
                    causal: false,
                    kv_layout: Layout::Padded,
                    ..base
                },
            ),
        ];
        for (i, (name, case)) in cases.iter().enumerate() {
            let ops = operands_elt(case, 2000 + 10 * d as u64 + i as u64, elt);
            let bound = bind(&ctx, &kernel, &ops);
            let (o, lse) = run(&ctx, &kernel, &bound);
            release(&ctx, bound);
            let pairs: Vec<(u32, u32)> = (0..case.b)
                .flat_map(|b| (0..case.h).map(move |h| (b, h)))
                .collect();
            let (worst, mean, worst_lse) = check(&ops, &o, &lse, &pairs);
            eprintln!(
                "{elt:?} D{d} {name} ({w} lanes): max |O err| {worst:.2e}, mean {mean:.2e}, max |LSE err| {worst_lse:.2e}"
            );
            assert!(
                worst < 2e-2 * worst_law,
                "{elt:?} D{d} {name}: max |O err| {worst}"
            );
            assert!(
                mean < 3e-3 * law,
                "{elt:?} D{d} {name}: mean |O err| {mean}"
            );
            assert!(
                worst_lse < 2e-3 * worst_law,
                "{elt:?} D{d} {name}: max |LSE err| {worst_lse}"
            );
        }
        ctx.destroy_kernel(kernel);
    }
}

fn timed(ctx: &Context, kernel: &Kernel, bound: &Bound) -> Option<f64> {
    let mut ms = Vec::new();
    for _ in 0..7 {
        match ctx.dispatch_timed(
            kernel,
            &bound.plan.push_constants(),
            bound.plan.dispatch_groups(),
            10,
        ) {
            Ok(t) => ms.push(t / 10.0),
            Err(e) => {
                eprintln!("UNMEASURED here (timing): {e}");
                return None;
            }
        }
    }
    ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Some(ms[ms.len() / 2])
}

#[test]
fn the_registered_geometries_are_correct_on_sampled_heads_and_timed() {
    registered_forward(Elt::Bf16);
}

#[test]
fn the_f16_registered_geometries_are_correct_on_sampled_heads_and_timed() {
    registered_forward(Elt::F16);
}

#[test]
fn the_f16hw_registered_geometries_are_correct_on_sampled_heads_and_timed() {
    registered_forward(Elt::F16Hw);
}

fn registered_forward(elt: Elt) {
    let Some(ctx) = open_or_unmeasured("attention-v2, registered geometries") else {
        return;
    };
    if elt == Elt::F16Hw && !hardware_f16(&ctx, "attention-v2 f16hw") {
        return;
    }
    let (law, worst_law) = (elt.law_scale(), elt.worst_scale());
    let Some(w) = width(&ctx) else {
        return;
    };
    let geometries = [
        (
            "AKV B4 H8 KV4 T1024 D64",
            Case {
                b: 4,
                h: 8,
                hkv: 4,
                t: 1024,
                d: 64,
                causal: true,
                scale: 0.125,
                q_layout: Layout::Bthd,
                kv_layout: Layout::Bthd,
            },
            [(0u32, 0u32), (3, 7)],
        ),
        (
            "stage-E B4 H32 KV4 T2048 D128",
            Case {
                b: 4,
                h: 32,
                hkv: 4,
                t: 2048,
                d: 128,
                causal: true,
                scale: 1.0 / 128f32.sqrt(),
                q_layout: Layout::Bthd,
                kv_layout: Layout::Bthd,
            },
            [(0, 0), (2, 31)],
        ),
    ];
    for (name, case, pairs) in geometries {
        let ops = operands_elt(&case, 17 + case.d as u64, elt);
        let kernel = v2_kernel_elt(&ctx, case.d, w, elt);
        let bound = bind(&ctx, &kernel, &ops);
        let (o, lse) = run(&ctx, &kernel, &bound);
        let (worst, mean, worst_lse) = check(&ops, &o, &lse, &pairs);
        eprintln!("{elt:?} {name} ({w} lanes), sampled heads: max |O err| {worst:.2e}, mean {mean:.2e}, max |LSE err| {worst_lse:.2e}");
        assert!(
            worst < 2e-2 * worst_law && mean < 3e-3 * law && worst_lse < 2e-3 * worst_law,
            "{elt:?} {name}"
        );
        if let Some(ms) = timed(&ctx, &kernel, &bound) {
            eprintln!(
                "TIMING {elt:?} {name} causal ({w} lanes): v2 {ms:.3} ms per forward, {:.1} TFLOPS (causal half)",
                bound.plan.flops() / (ms * 1e-3) / 1e12
            );
        }
        release(&ctx, bound);
        ctx.destroy_kernel(kernel);
    }
}

// ---------------------------------------------------------------------------
// the backward: dQ over query blocks, dK and dV over key blocks
// ---------------------------------------------------------------------------

use alelyon_compute_kit::attention_fused::{
    AttentionBackward, BackwardStrides, DKDV_OPERANDS, DKDV_PUSH_BYTES, DQ_OPERANDS, DQ_PUSH_BYTES,
};

const DQ_V2: [(u32, u32, &[u8]); 4] = [
    (
        64,
        64,
        include_bytes!("../kernels/attention_bwd_dq_v2_d64_s64.spv"),
    ),
    (
        64,
        32,
        include_bytes!("../kernels/attention_bwd_dq_v2_d64_s32.spv"),
    ),
    (
        128,
        64,
        include_bytes!("../kernels/attention_bwd_dq_v2_d128_s64.spv"),
    ),
    (
        128,
        32,
        include_bytes!("../kernels/attention_bwd_dq_v2_d128_s32.spv"),
    ),
];
const DKDV_V2: [(u32, u32, &[u8]); 4] = [
    (
        64,
        64,
        include_bytes!("../kernels/attention_bwd_dkdv_v2_d64_s64.spv"),
    ),
    (
        64,
        32,
        include_bytes!("../kernels/attention_bwd_dkdv_v2_d64_s32.spv"),
    ),
    (
        128,
        64,
        include_bytes!("../kernels/attention_bwd_dkdv_v2_d128_s64.spv"),
    ),
    (
        128,
        32,
        include_bytes!("../kernels/attention_bwd_dkdv_v2_d128_s32.spv"),
    ),
];

const DQ_V2_F16: [(u32, u32, &[u8]); 4] = [
    (
        64,
        64,
        include_bytes!("../kernels/attention_bwd_dq_v2_f16_d64_s64.spv"),
    ),
    (
        64,
        32,
        include_bytes!("../kernels/attention_bwd_dq_v2_f16_d64_s32.spv"),
    ),
    (
        128,
        64,
        include_bytes!("../kernels/attention_bwd_dq_v2_f16_d128_s64.spv"),
    ),
    (
        128,
        32,
        include_bytes!("../kernels/attention_bwd_dq_v2_f16_d128_s32.spv"),
    ),
];
const DKDV_V2_F16: [(u32, u32, &[u8]); 4] = [
    (
        64,
        64,
        include_bytes!("../kernels/attention_bwd_dkdv_v2_f16_d64_s64.spv"),
    ),
    (
        64,
        32,
        include_bytes!("../kernels/attention_bwd_dkdv_v2_f16_d64_s32.spv"),
    ),
    (
        128,
        64,
        include_bytes!("../kernels/attention_bwd_dkdv_v2_f16_d128_s64.spv"),
    ),
    (
        128,
        32,
        include_bytes!("../kernels/attention_bwd_dkdv_v2_f16_d128_s32.spv"),
    ),
];

const DQ_V2_F16HW: [(u32, u32, &[u8]); 4] = [
    (
        64,
        64,
        include_bytes!("../kernels/attention_bwd_dq_v2_f16hw_d64_s64.spv"),
    ),
    (
        64,
        32,
        include_bytes!("../kernels/attention_bwd_dq_v2_f16hw_d64_s32.spv"),
    ),
    (
        128,
        64,
        include_bytes!("../kernels/attention_bwd_dq_v2_f16hw_d128_s64.spv"),
    ),
    (
        128,
        32,
        include_bytes!("../kernels/attention_bwd_dq_v2_f16hw_d128_s32.spv"),
    ),
];
const DKDV_V2_F16HW: [(u32, u32, &[u8]); 4] = [
    (
        64,
        64,
        include_bytes!("../kernels/attention_bwd_dkdv_v2_f16hw_d64_s64.spv"),
    ),
    (
        64,
        32,
        include_bytes!("../kernels/attention_bwd_dkdv_v2_f16hw_d64_s32.spv"),
    ),
    (
        128,
        64,
        include_bytes!("../kernels/attention_bwd_dkdv_v2_f16hw_d128_s64.spv"),
    ),
    (
        128,
        32,
        include_bytes!("../kernels/attention_bwd_dkdv_v2_f16hw_d128_s32.spv"),
    ),
];

struct Family {
    fwd: Kernel,
    dq: Kernel,
    dkdv: Kernel,
}

fn v2_family(ctx: &Context, d: u32, w: u32, elt: Elt) -> Family {
    let pick = |t: &[(u32, u32, &'static [u8])]| {
        t.iter()
            .find(|(dd, ww, _)| *dd == d && *ww == w)
            .expect("module")
            .2
    };
    let (dq_table, dkdv_table) = match elt {
        Elt::Bf16 => (&DQ_V2, &DKDV_V2),
        Elt::F16 => (&DQ_V2_F16, &DKDV_V2_F16),
        Elt::F16Hw => (&DQ_V2_F16HW, &DKDV_V2_F16HW),
    };
    Family {
        fwd: v2_kernel_elt(ctx, d, w, elt),
        dq: ctx
            .kernel_with_subgroup(
                pick(dq_table),
                DQ_OPERANDS as u32,
                DQ_PUSH_BYTES as u32,
                Some(w),
            )
            .expect("dq v2"),
        dkdv: ctx
            .kernel_with_subgroup(
                pick(dkdv_table),
                DKDV_OPERANDS as u32,
                DKDV_PUSH_BYTES as u32,
                Some(w),
            )
            .expect("dkdv v2"),
    }
}

fn destroy(ctx: &Context, f: Family) {
    for k in [f.fwd, f.dq, f.dkdv] {
        ctx.destroy_kernel(k);
    }
}

struct Grads {
    dq: Vec<u16>,
    dk: Vec<u16>,
    dv: Vec<u16>,
}

struct Bufs {
    all: Vec<Buffer>,
    plan: AttentionPlan,
    bw: AttentionBackward,
}

/// q, k, v, o, lse, dO, dQ, dK, dV: inputs uploaded, gradients poisoned (a row
/// a pass never writes cannot pass as zero).
fn backward_bufs(ctx: &Context, ops: &Operands, dout: &[u16]) -> Bufs {
    let s = ops.shape;
    let (n_q, n_kv) = (ops.q.len(), ops.k.len());
    let n_lse = (s.batch * s.heads * s.seq) as usize;
    let plan = AttentionPlan::new(
        s,
        [
            n_q as u64,
            n_kv as u64,
            n_kv as u64,
            n_q as u64,
            n_lse as u64,
        ],
    )
    .expect("plan");
    let bs = BackwardStrides {
        dout: s.o,
        dq: s.q,
        dk: s.k,
        dv: s.v,
    };
    let bw = AttentionBackward::new(plan, bs, [n_q as u64, n_q as u64, n_kv as u64, n_kv as u64])
        .expect("backward plan");
    let sizes = [
        n_q * 2,
        n_kv * 2,
        n_kv * 2,
        n_q * 2,
        n_lse * 4,
        n_q * 2,
        n_q * 2,
        n_kv * 2,
        n_kv * 2,
    ];
    let all: Vec<Buffer> = sizes
        .iter()
        .map(|&n| ctx.buffer(n as u64).expect("buffer"))
        .collect();
    ctx.upload(&all[0], &bytes_u16(&ops.q)).expect("q");
    ctx.upload(&all[1], &bytes_u16(&ops.k)).expect("k");
    ctx.upload(&all[2], &bytes_u16(&ops.v)).expect("v");
    ctx.upload(&all[5], &bytes_u16(dout)).expect("dO");
    for i in [6usize, 7, 8] {
        ctx.upload(&all[i], &vec![0xFFu8; sizes[i]])
            .expect("poison");
    }
    Bufs { all, plan, bw }
}

fn dispatch_all(ctx: &Context, f: &Family, b: &Bufs) {
    let a = &b.all;
    let (r, w) = (Access::read, Access::write);
    ctx.bind(&f.fwd, &[&a[0], &a[1], &a[2], &a[3], &a[4]])
        .expect("bind fwd");
    ctx.dispatch(
        &f.fwd,
        &b.plan.push_constants(),
        b.plan.dispatch_groups(),
        1,
        &[
            r(a[0].id()),
            r(a[1].id()),
            r(a[2].id()),
            w(a[3].id()),
            w(a[4].id()),
        ],
    )
    .expect("fwd");
    ctx.bind(&f.dq, &[&a[0], &a[1], &a[2], &a[3], &a[5], &a[4], &a[6]])
        .expect("bind dq");
    ctx.dispatch(
        &f.dq,
        &b.bw.dq_push_constants(),
        b.bw.dq_groups(),
        1,
        &[
            r(a[0].id()),
            r(a[1].id()),
            r(a[2].id()),
            r(a[3].id()),
            r(a[5].id()),
            r(a[4].id()),
            w(a[6].id()),
        ],
    )
    .expect("dq");
    ctx.bind(
        &f.dkdv,
        &[&a[0], &a[1], &a[2], &a[3], &a[5], &a[4], &a[7], &a[8]],
    )
    .expect("bind dkdv");
    ctx.dispatch(
        &f.dkdv,
        &b.bw.dkdv_push_constants(),
        b.bw.dkdv_groups(),
        1,
        &[
            r(a[0].id()),
            r(a[1].id()),
            r(a[2].id()),
            r(a[3].id()),
            r(a[5].id()),
            r(a[4].id()),
            w(a[7].id()),
            w(a[8].id()),
        ],
    )
    .expect("dkdv");
}

fn to_u16(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn grads(ctx: &Context, b: &Bufs) -> Grads {
    Grads {
        dq: to_u16(&ctx.download(&b.all[6]).expect("dq")),
        dk: to_u16(&ctx.download(&b.all[7]).expect("dk")),
        dv: to_u16(&ctx.download(&b.all[8]).expect("dv")),
    }
}

fn release_bufs(ctx: &Context, b: Bufs) {
    for x in b.all {
        ctx.destroy_buffer(x);
    }
}

type Rows = Vec<Vec<f64>>;

/// fp64 gradients for batch b and kv head hk over every query head of its
/// group: (dQ rows per query head, dK rows, dV rows).
fn reference_backward(ops: &Operands, dout: &[u16], b: u32, hk: u32) -> (Vec<Rows>, Rows, Rows) {
    let s = ops.shape;
    let d = s.head_dim as usize;
    let group = s.heads / s.kv_heads;
    let t = s.seq as usize;
    let f = |v: &[u16], i: usize| ops.elt.value(v[i]) as f64;
    let scale = s.scale as f64;
    let mut dk = vec![vec![0.0f64; d]; t];
    let mut dv = vec![vec![0.0f64; d]; t];
    let mut dqs = Vec::new();
    for gi in 0..group {
        let h = hk * group + gi;
        let (o_ref, _) = reference(ops, b, h);
        let mut dq = vec![vec![0.0f64; d]; t];
        for i in 0..s.seq {
            let qi = at(s.q, b, h, i);
            let oi = at(s.o, b, h, i);
            let last = if s.causal { i } else { s.seq - 1 };
            let scores: Vec<f64> = (0..=last)
                .map(|j| {
                    let kj = at(s.k, b, hk, j);
                    (0..d)
                        .map(|e| f(&ops.q, qi + e) * f(&ops.k, kj + e))
                        .sum::<f64>()
                        * scale
                })
                .collect();
            let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let l: f64 = scores.iter().map(|x| (x - m).exp()).sum();
            let delta: f64 = (0..d).map(|e| f(dout, oi + e) * o_ref[i as usize][e]).sum();
            for (j, sc) in scores.iter().enumerate() {
                let pij = (sc - m).exp() / l;
                let vj = at(s.v, b, hk, j as u32);
                let kj = at(s.k, b, hk, j as u32);
                let dpij: f64 = (0..d).map(|e| f(dout, oi + e) * f(&ops.v, vj + e)).sum();
                let dsij = pij * (dpij - delta);
                for e in 0..d {
                    dq[i as usize][e] += scale * dsij * f(&ops.k, kj + e);
                    dk[j][e] += scale * dsij * f(&ops.q, qi + e);
                    dv[j][e] += pij * f(dout, oi + e);
                }
            }
        }
        dqs.push(dq);
    }
    (dqs, dk, dv)
}

/// Worst and mean |error| of bf16 rows against fp64 rows, relative to the
/// mean |reference|; a non-finite value (a poisoned row) is an infinite error.
fn rel_errors(got: &[u16], want: &Rows, rows: impl Fn(usize) -> usize, elt: Elt) -> (f64, f64) {
    let (mut worst, mut sum, mut mag, mut n) = (0.0f64, 0.0f64, 0.0f64, 0usize);
    for (i, row) in want.iter().enumerate() {
        let base = rows(i);
        for (e, w) in row.iter().enumerate() {
            let g = elt.value(got[base + e]) as f64;
            let err = if g.is_finite() {
                (g - w).abs()
            } else {
                f64::INFINITY
            };
            worst = worst.max(err);
            sum += err;
            mag += w.abs();
            n += 1;
        }
    }
    let mean_mag = (mag / n as f64).max(1e-12);
    (worst / mean_mag, sum / n as f64 / mean_mag)
}

fn check_backward(
    ops: &Operands,
    dout: &[u16],
    g: &Grads,
    pairs: &[(u32, u32)],
) -> [(f64, f64); 3] {
    let s = ops.shape;
    let group = s.heads / s.kv_heads;
    let mut out = [(0.0f64, 0.0f64); 3];
    for &(b, hk) in pairs {
        let (dqs, dk, dv) = reference_backward(ops, dout, b, hk);
        for (gi, dq) in dqs.iter().enumerate() {
            let h = hk * group + gi as u32;
            let e = rel_errors(&g.dq, dq, |i| at(s.q, b, h, i as u32), ops.elt);
            out[0] = (out[0].0.max(e.0), out[0].1.max(e.1));
        }
        let e = rel_errors(&g.dk, &dk, |i| at(s.k, b, hk, i as u32), ops.elt);
        out[1] = (out[1].0.max(e.0), out[1].1.max(e.1));
        let e = rel_errors(&g.dv, &dv, |i| at(s.v, b, hk, i as u32), ops.elt);
        out[2] = (out[2].0.max(e.0), out[2].1.max(e.1));
    }
    out
}

fn random_dout(n: usize, seed: u64, elt: Elt) -> Vec<u16> {
    let mut rng = Lcg(seed);
    (0..n).map(|_| elt.round(rng.next())).collect()
}

#[test]
fn the_v2_backward_matches_fp64_gradients_at_both_head_dimensions() {
    backward_matches_fp64(Elt::Bf16);
}

#[test]
fn the_f16_backward_matches_fp64_gradients_at_both_head_dimensions() {
    backward_matches_fp64(Elt::F16);
}

#[test]
fn the_f16hw_backward_matches_fp64_gradients_at_both_head_dimensions() {
    backward_matches_fp64(Elt::F16Hw);
}

fn backward_matches_fp64(elt: Elt) {
    let Some(ctx) = open_or_unmeasured("attention-v2 backward") else {
        return;
    };
    if elt == Elt::F16Hw && !hardware_f16(&ctx, "attention-v2 f16hw") {
        return;
    }
    let (law, worst_law) = (elt.law_scale(), elt.worst_scale());
    for (w, d) in widths(&ctx)
        .into_iter()
        .flat_map(|w| [64u32, 128].map(move |d| (w, d)))
    {
        let fam = v2_family(&ctx, d, w, elt);
        let base = Case {
            b: 1,
            h: 2,
            hkv: 1,
            t: 64,
            d,
            causal: true,
            scale: 1.0 / (d as f32).sqrt(),
            q_layout: Layout::Bhtd,
            kv_layout: Layout::Bhtd,
        };
        let cases = [
            ("one block", base),
            (
                "tail rows and keys",
                Case {
                    b: 2,
                    h: 4,
                    hkv: 2,
                    t: 200,
                    ..base
                },
            ),
            (
                "causal off",
                Case {
                    h: 2,
                    hkv: 2,
                    t: 130,
                    causal: false,
                    scale: 0.2,
                    ..base
                },
            ),
            (
                "[B, T, H, D] view, 4 query heads a kv head",
                Case {
                    b: 2,
                    h: 8,
                    hkv: 2,
                    t: 96,
                    q_layout: Layout::Bthd,
                    kv_layout: Layout::Bthd,
                    ..base
                },
            ),
            (
                "4-byte staging",
                Case {
                    b: 1,
                    h: 4,
                    hkv: 2,
                    t: 150,
                    q_layout: Layout::Padded,
                    kv_layout: Layout::Padded,
                    ..base
                },
            ),
        ];
        for (i, (name, case)) in cases.iter().enumerate() {
            let ops = operands_elt(case, 5000 + 10 * d as u64 + i as u64, elt);
            let dout = random_dout(ops.q.len(), 6000 + i as u64, elt);
            let bufs = backward_bufs(&ctx, &ops, &dout);
            dispatch_all(&ctx, &fam, &bufs);
            let g = grads(&ctx, &bufs);
            release_bufs(&ctx, bufs);
            let pairs: Vec<(u32, u32)> = (0..case.b)
                .flat_map(|b| (0..case.hkv).map(move |hk| (b, hk)))
                .collect();
            let [dq, dk, dv] = check_backward(&ops, &dout, &g, &pairs);
            eprintln!(
                "{elt:?} D{d} {name} ({w} lanes), error / mean |ref|: dQ max {:.2e} mean {:.2e}; dK max {:.2e} mean {:.2e}; dV max {:.2e} mean {:.2e}",
                dq.0, dq.1, dk.0, dk.1, dv.0, dv.1
            );
            for (what, (worst, mean)) in [("dQ", dq), ("dK", dk), ("dV", dv)] {
                assert!(
                    worst < 0.25 * worst_law,
                    "{elt:?} D{d} {name}: {what} worst error {worst} of the mean magnitude"
                );
                assert!(
                    mean < 0.02 * law,
                    "{elt:?} D{d} {name}: {what} mean error {mean} of the mean magnitude"
                );
            }
        }
        destroy(&ctx, fam);
    }
}

fn median_ms(ctx: &Context, kernel: &Kernel, push: &[u8], groups: [u32; 3]) -> Option<f64> {
    let mut ms = Vec::new();
    for _ in 0..7 {
        match ctx.dispatch_timed(kernel, push, groups, 10) {
            Ok(t) => ms.push(t / 10.0),
            Err(e) => {
                eprintln!("UNMEASURED here (timing): {e}");
                return None;
            }
        }
    }
    ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Some(ms[ms.len() / 2])
}

/// (forward, dQ, dK/dV) in ms, each pass alone from timestamp queries.
fn time_family(ctx: &Context, f: &Family, b: &Bufs) -> Option<(f64, f64, f64)> {
    let a = &b.all;
    ctx.bind(&f.fwd, &[&a[0], &a[1], &a[2], &a[3], &a[4]])
        .expect("bind");
    let tf = median_ms(
        ctx,
        &f.fwd,
        &b.plan.push_constants(),
        b.plan.dispatch_groups(),
    )?;
    ctx.bind(&f.dq, &[&a[0], &a[1], &a[2], &a[3], &a[5], &a[4], &a[6]])
        .expect("bind");
    let tq = median_ms(ctx, &f.dq, &b.bw.dq_push_constants(), b.bw.dq_groups())?;
    ctx.bind(
        &f.dkdv,
        &[&a[0], &a[1], &a[2], &a[3], &a[5], &a[4], &a[7], &a[8]],
    )
    .expect("bind");
    let tkv = median_ms(
        ctx,
        &f.dkdv,
        &b.bw.dkdv_push_constants(),
        b.bw.dkdv_groups(),
    )?;
    Some((tf, tq, tkv))
}

#[test]
fn the_v2_backward_at_the_registered_geometries_is_correct_on_sampled_heads_and_timed() {
    registered_backward(Elt::Bf16);
}

#[test]
fn the_f16_backward_at_the_registered_geometries_is_correct_on_sampled_heads_and_timed() {
    registered_backward(Elt::F16);
}

#[test]
fn the_f16hw_backward_at_the_registered_geometries_is_correct_on_sampled_heads_and_timed() {
    registered_backward(Elt::F16Hw);
}

fn registered_backward(elt: Elt) {
    let Some(ctx) = open_or_unmeasured("attention-v2 backward, registered geometries") else {
        return;
    };
    if elt == Elt::F16Hw && !hardware_f16(&ctx, "attention-v2 f16hw") {
        return;
    }
    let (law, worst_law) = (elt.law_scale(), elt.worst_scale());
    let Some(w) = width(&ctx) else {
        return;
    };
    let geometries = [
        (
            "AKV B4 H8 KV4 T1024 D64",
            Case {
                b: 4,
                h: 8,
                hkv: 4,
                t: 1024,
                d: 64,
                causal: true,
                scale: 0.125,
                q_layout: Layout::Bthd,
                kv_layout: Layout::Bthd,
            },
            vec![(0u32, 0u32), (3, 3)],
        ),
        (
            "stage-E B4 H32 KV4 T2048 D128",
            Case {
                b: 4,
                h: 32,
                hkv: 4,
                t: 2048,
                d: 128,
                causal: true,
                scale: 1.0 / 128f32.sqrt(),
                q_layout: Layout::Bthd,
                kv_layout: Layout::Bthd,
            },
            vec![(1, 2)],
        ),
    ];
    for (name, case, pairs) in geometries {
        let ops = operands_elt(&case, 41 + case.d as u64, elt);
        let dout = random_dout(ops.q.len(), 43, elt);
        let fam = v2_family(&ctx, case.d, w, elt);
        let bufs = backward_bufs(&ctx, &ops, &dout);
        dispatch_all(&ctx, &fam, &bufs);
        let g = grads(&ctx, &bufs);
        let [dq, dk, dv] = check_backward(&ops, &dout, &g, &pairs);
        eprintln!(
            "{elt:?} {name} backward ({w} lanes), sampled kv heads, error / mean |ref|: dQ max {:.2e} mean {:.2e}; dK max {:.2e} mean {:.2e}; dV max {:.2e} mean {:.2e}",
            dq.0, dq.1, dk.0, dk.1, dv.0, dv.1
        );
        for (what, (worst, mean)) in [("dQ", dq), ("dK", dk), ("dV", dv)] {
            assert!(
                worst < 0.25 * worst_law && mean < 0.02 * law,
                "{elt:?} {name} {what}: worst {worst}, mean {mean}"
            );
        }
        if let Some((tf, tq, tkv)) = time_family(&ctx, &fam, &bufs) {
            eprintln!(
                "TIMING {elt:?} {name} causal ({w} lanes): v2 forward {tf:.3} ms, dQ {tq:.3} ms, dK/dV {tkv:.3} ms; backward {:.3} ms ({:.1} TFLOPS); forward + backward {:.3} ms",
                tq + tkv,
                bufs.bw.flops() / ((tq + tkv) * 1e-3) / 1e12,
                tf + tq + tkv
            );
        }
        release_bufs(&ctx, bufs);
        destroy(&ctx, fam);
    }
}

/// Every output of one forward and backward: O, LSE (as bits), dQ, dK, dV.
fn outputs(ctx: &Context, b: &Bufs) -> Vec<Vec<u32>> {
    let words16 = |i: usize| {
        to_u16(&ctx.download(&b.all[i]).expect("download"))
            .into_iter()
            .map(u32::from)
            .collect::<Vec<u32>>()
    };
    let lse = ctx
        .download(&b.all[4])
        .expect("lse")
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    vec![words16(3), lse, words16(6), words16(7), words16(8)]
}

fn is_f16_subnormal(h: u32) -> bool {
    (h & 0x7C00) == 0 && (h & 0x03FF) != 0
}

#[test]
fn the_hardware_f16_modules_are_the_integer_modules_bit_for_bit() {
    let Some(ctx) = open_or_unmeasured("attention-v2 f16hw against f16") else {
        return;
    };
    if !hardware_f16(&ctx, "attention-v2 f16hw against f16") {
        return;
    }
    let names = ["O", "LSE", "dQ", "dK", "dV"];
    let (mut compared, mut subnormal_out, mut near_max_out) = (0usize, 0usize, 0usize);
    for (w, d) in widths(&ctx)
        .into_iter()
        .flat_map(|w| [64u32, 128].map(move |d| (w, d)))
    {
        let int = v2_family(&ctx, d, w, Elt::F16);
        let hw = v2_family(&ctx, d, w, Elt::F16Hw);
        let base = Case {
            b: 2,
            h: 4,
            hkv: 2,
            t: 200,
            d,
            causal: true,
            scale: 1.0 / (d as f32).sqrt(),
            q_layout: Layout::Bhtd,
            kv_layout: Layout::Bhtd,
        };
        // (name, case, factor on v): scale 4 spreads a row's scores over
        // hundreds of units, so most probabilities are binary16 subnormals or
        // zero; v * 2^-18 puts O and dV below binary16's normal range; v * 2^14
        // puts O within a binade of 65504
        let mut cases = vec![
            ("default", base, 1.0f32),
            ("wide scores", Case { scale: 4.0, ..base }, 1.0),
            (
                "wide scores, causal off",
                Case {
                    scale: 4.0,
                    causal: false,
                    t: 130,
                    ..base
                },
                1.0,
            ),
            ("subnormal outputs", base, 1.0 / 262_144.0),
            (
                "outputs near the largest finite",
                Case { t: 96, ..base },
                16_384.0,
            ),
            (
                "[B, T, H, D] view",
                Case {
                    q_layout: Layout::Bthd,
                    kv_layout: Layout::Bthd,
                    ..base
                },
                1.0,
            ),
        ];
        // the stage-E step's own geometry, at the width the kit builds for
        // this device: where the first comparison in situ (2026-09-29, before
        // NoContraction) found O differing in 539 of 16.8M elements
        if d == 128 && Some(w) == width(&ctx) {
            cases.push((
                "stage-E B2 H32 KV4 T2048",
                Case {
                    h: 32,
                    hkv: 4,
                    t: 2048,
                    ..base
                },
                1.0,
            ));
        }
        for (i, (name, case, factor)) in cases.iter().enumerate() {
            let mut ops = operands_elt(case, 9000 + 10 * d as u64 + i as u64, Elt::F16);
            if *factor != 1.0 {
                ops.v = ops.v.iter().map(|&h| f16(from_f16(h) * factor)).collect();
            }
            let dout = random_dout(ops.q.len(), 9100 + i as u64, Elt::F16);
            let mut got = Vec::new();
            for fam in [&int, &hw] {
                let bufs = backward_bufs(&ctx, &ops, &dout);
                dispatch_all(&ctx, fam, &bufs);
                got.push(outputs(&ctx, &bufs));
                release_bufs(&ctx, bufs);
            }
            for (k, name_k) in names.iter().enumerate() {
                let (a, b) = (&got[0][k], &got[1][k]);
                assert_eq!(a.len(), b.len());
                let differ: Vec<usize> = (0..a.len()).filter(|&j| a[j] != b[j]).collect();
                assert!(
                    differ.is_empty(),
                    "D{d} {name} ({w} lanes): {name_k} differs at {} of {} elements, first at {} (integer {:#06x}, hardware {:#06x})",
                    differ.len(),
                    a.len(),
                    differ[0],
                    a[differ[0]],
                    b[differ[0]]
                );
                compared += a.len();
                if k != 1 {
                    subnormal_out += a.iter().filter(|&&h| is_f16_subnormal(h)).count();
                    near_max_out += a
                        .iter()
                        .filter(|&&h| (h & 0x7FFF) >= 0x7800 && (h & 0x7C00) != 0x7C00)
                        .count();
                }
            }
        }
        destroy(&ctx, int);
        destroy(&ctx, hw);
    }
    eprintln!(
        "f16hw against f16: {compared} outputs equal bit for bit, {subnormal_out} of them binary16 subnormals, {near_max_out} at or above 32768"
    );
    assert!(
        subnormal_out > 0,
        "no output was a binary16 subnormal: the comparison did not reach that range"
    );
    assert!(
        near_max_out > 0,
        "no output reached the top binade: the comparison did not reach that range"
    );
}
