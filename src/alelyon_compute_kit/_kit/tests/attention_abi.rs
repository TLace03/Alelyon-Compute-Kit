//! `ack_attention_fwd` and `ack_attention_bwd` (attention-bf16/v1): the kit's
//! OPTIONAL fused attention entries, through the C ABI the torch adapter binds
//! (torch_backend/ack_backend.cpp), on a real device.
//!
//! One small geometry carries the argument order and the stride words end to
//! end: grouped kv heads (two query heads read one kv head), a tail block (80
//! rows), and q, o, dO and dQ in a `[B, T, H, D]` layout viewed as
//! `[B, H, T, D]`, which is what the model hands SDPA. Every output is poisoned
//! with NaN first and checked finite, and the values are held to the laws of
//! tests/attention_v2.rs against fp64 attention and its gradients computed
//! from the same bf16-rounded inputs; this file does not re-derive them. Then
//! each refusal, by code and by name, before anything is recorded.
//!
//! The binary16 entries (`ack_attention_f16_fwd` / `ack_attention_f16_bwd`,
//! attention-f16, 2026-09-29) run the same geometry through the same argument
//! order, against the same fp64 reference computed from the fp16-rounded
//! inputs, under the same laws scaled by the element's unit roundoff relative
//! to bf16's (2^-11 / 2^-8 = 1/8; the worst element's by 1/4, see
//! `Elt::worst_scale`), and refuse by the same names.
//!
//! Skips as UNMEASURED without a device unless ACK_REQUIRE_DEVICE is set. A
//! GREEN RUN WITHOUT THAT VARIABLE IS NOT DEVICE EVIDENCE.
use alelyon_compute_kit::{ffi, Buffer};
use std::ffi::{c_char, CStr};
use std::mem::{offset_of, size_of};

// The adapter's C mirror static_asserts the two sizes; the offsets pin the
// field order it assumes.
const _: () = assert!(size_of::<ffi::AckAttentionShape>() == 128);
const _: () = assert!(size_of::<ffi::AckAttentionBackwardShape>() == 224);
const _: () = assert!(offset_of!(ffi::AckAttentionShape, scale) == 24);
const _: () = assert!(offset_of!(ffi::AckAttentionShape, q) == 32);
const _: () = assert!(offset_of!(ffi::AckAttentionShape, o) == 104);
const _: () = assert!(offset_of!(ffi::AckAttentionBackwardShape, dout) == 128);
const _: () = assert!(offset_of!(ffi::AckAttentionBackwardShape, dv) == 200);

const D: usize = 64;
const B: usize = 1;
const H: usize = 2;
const KV: usize = 1;
const T: usize = 80;
const SCALE: f32 = 0.125;
const N_Q: usize = B * T * H * D;
const N_KV: usize = B * KV * T * D;
const N_LSE: usize = B * H * T;
/// `[B, T, H, D]` storage addressed as `[B, H, T, D]`: q, o, dO and dQ.
const QS: [u64; 3] = [(T * H * D) as u64, D as u64, (H * D) as u64];
/// Dense `[B, KV, T, D]`: k, v, dK and dV.
const KS: [u64; 3] = [(KV * T * D) as u64, (T * D) as u64, D as u64];
const NAN_BF16: u16 = 0x7FC1;
const NAN_F16: u16 = 0x7E01;

fn last_error() -> String {
    let mut buf = vec![0 as c_char; 1024];
    let rc = unsafe { ffi::ack_last_error(buf.as_mut_ptr(), buf.len()) };
    assert!(rc >= 0, "ack_last_error returned {rc}");
    unsafe { CStr::from_ptr(buf.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

fn open_or_unmeasured(what: &str) -> Option<*mut ffi::AckDevice> {
    let dev = ffi::ack_open();
    if dev.is_null() {
        if std::env::var_os("ACK_REQUIRE_DEVICE").is_some() {
            panic!(
                "{what}: ACK_REQUIRE_DEVICE is set and ack_open failed: {}",
                last_error()
            );
        }
        eprintln!("UNMEASURED here ({what}): {}", last_error());
        return None;
    }
    Some(dev)
}

fn bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    let lsb = (bits >> 16) & 1;
    ((bits + 0x7FFF + lsb) >> 16) as u16
}

fn from_bf16(h: u16) -> f64 {
    f64::from(f32::from_bits(u32::from(h) << 16))
}

/// binary32 -> binary16, round to nearest even, subnormals kept, an overflow to
/// infinity, a NaN quieted with its payload's top bits: the routine of the
/// kernels' f32_to_f16, written again here.
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

/// binary16 -> binary64, exactly.
fn from_f16(h: u16) -> f64 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = i32::from((h >> 10) & 0x1F);
    let m = f64::from(h & 0x03FF);
    match e {
        0x1F if m == 0.0 => sign * f64::INFINITY,
        0x1F => f64::NAN,
        0 => sign * m * 2f64.powi(-24),
        _ => sign * (1.0 + m / 1024.0) * 2f64.powi(e - 15),
    }
}

/// The two element types the entries take.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Elt {
    Bf16,
    F16,
}

type FwdEntry = unsafe extern "C" fn(
    *const ffi::AckDevice,
    *const Buffer,
    *const Buffer,
    *const Buffer,
    *const Buffer,
    *const Buffer,
    *const ffi::AckAttentionShape,
    *mut f64,
) -> i32;
type BwdEntry = unsafe extern "C" fn(
    *const ffi::AckDevice,
    *const Buffer,
    *const Buffer,
    *const Buffer,
    *const Buffer,
    *const Buffer,
    *const Buffer,
    *const Buffer,
    *const Buffer,
    *const Buffer,
    *const ffi::AckAttentionBackwardShape,
    *mut f64,
) -> i32;

impl Elt {
    fn round(self, x: f32) -> u16 {
        match self {
            Elt::Bf16 => bf16(x),
            Elt::F16 => f16(x),
        }
    }
    fn value(self, h: u16) -> f64 {
        match self {
            Elt::Bf16 => from_bf16(h),
            Elt::F16 => from_f16(h),
        }
    }
    fn nan(self) -> u16 {
        match self {
            Elt::Bf16 => NAN_BF16,
            Elt::F16 => NAN_F16,
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
            Elt::F16 => 0.125,
        }
    }
    fn worst_scale(self) -> f64 {
        match self {
            Elt::Bf16 => 1.0,
            Elt::F16 => 0.25,
        }
    }
    fn entries(self) -> (FwdEntry, BwdEntry) {
        match self {
            Elt::Bf16 => (ffi::ack_attention_fwd, ffi::ack_attention_bwd),
            Elt::F16 => (ffi::ack_attention_f16_fwd, ffi::ack_attention_f16_bwd),
        }
    }
}

#[test]
fn the_f16_rounding_is_round_to_nearest_even_and_exact_both_ways() {
    // every finite binary16 round-trips; the boundaries are the ones the kernels use
    for h in 0..=u16::MAX {
        let v = from_f16(h);
        if v.is_nan() {
            continue;
        }
        assert_eq!(f16(v as f32), h, "{h:#06x} = {v}");
    }
    assert_eq!(f16(1.0), 0x3C00);
    assert_eq!(f16(65504.0), 0x7BFF);
    assert_eq!(f16(65519.99), 0x7BFF, "below the halfway point to 2^16");
    assert_eq!(
        f16(65520.0),
        0x7C00,
        "the halfway point rounds to even: infinity"
    );
    assert_eq!(f16(2f32.powi(-24)), 0x0001);
    assert_eq!(
        f16(2f32.powi(-25)),
        0x0000,
        "half the smallest subnormal ties to even: zero"
    );
    assert_eq!(f16(1.5 * 2f32.powi(-25)), 0x0001);
    assert_eq!(
        f16(3.0 * 2f32.powi(-25)),
        0x0002,
        "1.5 subnormal units tie to even"
    );
    assert_eq!(f16(-0.0), 0x8000);
    assert_eq!(f16(f32::NEG_INFINITY), 0xFC00);
    assert_eq!(
        f16(f32::from_bits(0x7F81_2345)) & 0x7E00,
        0x7E00,
        "a NaN comes back quiet"
    );
    assert_eq!(
        f16(1.0 + 2f32.powi(-11)),
        0x3C00,
        "a tie at 1 rounds to even"
    );
    assert_eq!(
        f16(1.0 + 3.0 * 2f32.powi(-11)),
        0x3C02,
        "and the next tie rounds up to even"
    );
}

/// The same deterministic pseudo-normal draw as tests/attention_v2.rs.
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

fn draws(n: usize, seed: u64, elt: Elt) -> Vec<u16> {
    let mut rng = Lcg(seed);
    (0..n).map(|_| elt.round(rng.next())).collect()
}

fn at(s: [u64; 3], b: usize, h: usize, r: usize) -> usize {
    (b as u64 * s[0] + h as u64 * s[1] + r as u64 * s[2]) as usize
}

fn buffer(dev: *mut ffi::AckDevice, bytes: &[u8]) -> *mut Buffer {
    let b = ffi::ack_buffer_alloc(dev, bytes.len() as u64);
    assert!(!b.is_null(), "{}", last_error());
    let rc = unsafe { ffi::ack_upload(dev, b, bytes.as_ptr(), bytes.len()) };
    assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
    b
}

fn u16_bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn download(dev: *mut ffi::AckDevice, buf: *const Buffer, bytes: usize) -> Vec<u8> {
    let mut out = vec![0u8; bytes];
    let rc = unsafe { ffi::ack_download(dev, buf, out.as_mut_ptr(), bytes) };
    assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
    out
}

fn as_u16(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn as_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn shape() -> ffi::AckAttentionShape {
    ffi::AckAttentionShape {
        batch: B as u32,
        heads: H as u32,
        kv_heads: KV as u32,
        seq: T as u32,
        head_dim: D as u32,
        causal: 1,
        scale: SCALE,
        reserved: 0,
        q: QS,
        k: KS,
        v: KS,
        o: QS,
    }
}

/// fp64 attention and its gradients from the rounded inputs: (o, lse, dq, dk,
/// dv) in the operands' own layouts.
#[allow(clippy::type_complexity)]
fn reference(
    q: &[u16],
    k: &[u16],
    v: &[u16],
    g: &[u16],
    elt: Elt,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let from_bf16 = |h: u16| elt.value(h);
    let (mut o, mut lse) = (vec![0.0f64; N_Q], vec![0.0f64; N_LSE]);
    let (mut dq, mut dk, mut dv) = (vec![0.0f64; N_Q], vec![0.0f64; N_KV], vec![0.0f64; N_KV]);
    let scale = f64::from(SCALE);
    for b in 0..B {
        for h in 0..H {
            let hk = h / (H / KV);
            for i in 0..T {
                let qi = at(QS, b, h, i);
                let scores: Vec<f64> = (0..=i)
                    .map(|j| {
                        let kj = at(KS, b, hk, j);
                        (0..D)
                            .map(|d| from_bf16(q[qi + d]) * from_bf16(k[kj + d]))
                            .sum::<f64>()
                            * scale
                    })
                    .collect();
                let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let l: f64 = scores.iter().map(|x| (x - m).exp()).sum();
                lse[(b * H + h) * T + i] = m + l.ln();
                let p: Vec<f64> = scores.iter().map(|x| (x - m).exp() / l).collect();
                for (j, pj) in p.iter().enumerate() {
                    let vj = at(KS, b, hk, j);
                    for d in 0..D {
                        o[qi + d] += pj * from_bf16(v[vj + d]);
                    }
                }
                let delta: f64 = (0..D).map(|d| from_bf16(g[qi + d]) * o[qi + d]).sum();
                for (j, pj) in p.iter().enumerate() {
                    let (kj, vj) = (at(KS, b, hk, j), at(KS, b, hk, j));
                    let dp: f64 = (0..D)
                        .map(|d| from_bf16(g[qi + d]) * from_bf16(v[vj + d]))
                        .sum();
                    let ds = pj * (dp - delta);
                    for d in 0..D {
                        dq[qi + d] += scale * ds * from_bf16(k[kj + d]);
                        dk[kj + d] += scale * ds * from_bf16(q[qi + d]);
                        dv[vj + d] += pj * from_bf16(g[qi + d]);
                    }
                }
            }
        }
    }
    (o, lse, dq, dk, dv)
}

/// Worst and mean absolute error of bf16 `got` against `want`, each over the
/// mean magnitude of `want` (tests/attention_v2.rs's `rel_errors`), after
/// asserting every element was written.
fn rel_errors(what: &str, got: &[u16], want: &[f64], elt: Elt) -> (f64, f64) {
    let (mut worst, mut sum, mut mag) = (0.0f64, 0.0f64, 0.0f64);
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let g = elt.value(*g);
        assert!(
            g.is_finite(),
            "{what}[{i}] is not finite: the poison survived"
        );
        let e = (g - w).abs();
        worst = worst.max(e);
        sum += e;
        mag += w.abs();
    }
    let mean_mag = (mag / want.len() as f64).max(1e-12);
    (worst / mean_mag, sum / want.len() as f64 / mean_mag)
}

#[test]
fn the_attention_entries_match_fp64_through_the_c_abi() {
    matches_fp64_through_the_c_abi(Elt::Bf16);
}

#[test]
fn the_f16_attention_entries_match_fp64_through_the_c_abi() {
    matches_fp64_through_the_c_abi(Elt::F16);
}

fn matches_fp64_through_the_c_abi(elt: Elt) {
    let Some(dev) = open_or_unmeasured("attention C ABI") else {
        return;
    };
    let (fwd_entry, bwd_entry) = elt.entries();
    let (q, k, v, g) = (
        draws(N_Q, 41, elt),
        draws(N_KV, 42, elt),
        draws(N_KV, 43, elt),
        draws(N_Q, 44, elt),
    );
    let (want_o, want_lse, want_dq, want_dk, want_dv) = reference(&q, &k, &v, &g, elt);
    let poison_q = u16_bytes(&vec![elt.nan(); N_Q]);
    let poison_kv = u16_bytes(&vec![elt.nan(); N_KV]);
    let poison_lse: Vec<u8> = (0..N_LSE).flat_map(|_| f32::NAN.to_le_bytes()).collect();
    let qb = buffer(dev, &u16_bytes(&q));
    let kb = buffer(dev, &u16_bytes(&k));
    let vb = buffer(dev, &u16_bytes(&v));
    let gb = buffer(dev, &u16_bytes(&g));
    let ob = buffer(dev, &poison_q);
    let lb = buffer(dev, &poison_lse);
    let dqb = buffer(dev, &poison_q);
    let dkb = buffer(dev, &poison_kv);
    let dvb = buffer(dev, &poison_kv);

    let fwd = shape();
    let mut ms = 0.0f64;
    let rc = unsafe { fwd_entry(dev, qb, kb, vb, ob, lb, &fwd, &mut ms) };
    assert_eq!(rc, ffi::ACK_OK, "{elt:?} forward: {}", last_error());
    assert!(
        ms.is_nan(),
        "a recorded entry reports no duration, got {ms}"
    );
    let block = ffi::AckAttentionBackwardShape {
        forward: fwd,
        dout: QS,
        dq: QS,
        dk: KS,
        dv: KS,
    };
    let mut ms = 0.0f64;
    let rc = unsafe { bwd_entry(dev, qb, kb, vb, ob, gb, lb, dqb, dkb, dvb, &block, &mut ms) };
    assert_eq!(rc, ffi::ACK_OK, "{elt:?} backward: {}", last_error());
    assert!(
        ms.is_nan(),
        "a recorded entry reports no duration, got {ms}"
    );

    let o = as_u16(&download(dev, ob, N_Q * 2));
    let lse = as_f32(&download(dev, lb, N_LSE * 4));
    let dq = as_u16(&download(dev, dqb, N_Q * 2));
    let dk = as_u16(&download(dev, dkb, N_KV * 2));
    let dv = as_u16(&download(dev, dvb, N_KV * 2));

    // the forward's laws are absolute (O is a convex mix of v rows)
    let (mut worst_o, mut sum_o) = (0.0f64, 0.0f64);
    for (i, (got, want)) in o.iter().zip(&want_o).enumerate() {
        let got = elt.value(*got);
        assert!(got.is_finite(), "o[{i}] is not finite: the poison survived");
        worst_o = worst_o.max((got - want).abs());
        sum_o += (got - want).abs();
    }
    let mean_o = sum_o / N_Q as f64;
    let mut worst_lse = 0.0f64;
    for (i, (got, want)) in lse.iter().zip(&want_lse).enumerate() {
        assert!(
            got.is_finite(),
            "lse[{i}] is not finite: the poison survived"
        );
        worst_lse = worst_lse.max((f64::from(*got) - want).abs());
    }
    let e_dq = rel_errors("dq", &dq, &want_dq, elt);
    let e_dk = rel_errors("dk", &dk, &want_dk, elt);
    let e_dv = rel_errors("dv", &dv, &want_dv, elt);
    eprintln!(
        "C ABI {elt:?}, B1 H2 KV1 T80 causal, [B, T, H, D] q/o/dO/dQ: O max {worst_o:.2e} mean {mean_o:.2e}; \
         LSE max {worst_lse:.2e}; error / mean |ref|: dQ {:.2e}/{:.2e}, dK {:.2e}/{:.2e}, dV {:.2e}/{:.2e}",
        e_dq.0, e_dq.1, e_dk.0, e_dk.1, e_dv.0, e_dv.1
    );
    let (law, worst_law) = (elt.law_scale(), elt.worst_scale());
    assert!(worst_o < 2e-2 * worst_law, "{elt:?} max |O err| {worst_o}");
    assert!(mean_o < 3e-3 * law, "{elt:?} mean |O err| {mean_o}");
    assert!(
        worst_lse < 2e-3 * worst_law,
        "{elt:?} max |LSE err| {worst_lse}"
    );
    for (what, (worst, mean)) in [("dQ", e_dq), ("dK", e_dk), ("dV", e_dv)] {
        assert!(
            worst < 0.25 * worst_law,
            "{elt:?} {what} worst error {worst} of the mean magnitude"
        );
        assert!(
            mean < 0.02 * law,
            "{elt:?} {what} mean error {mean} of the mean magnitude"
        );
    }
    for b in [qb, kb, vb, gb, ob, lb, dqb, dkb, dvb] {
        assert_eq!(
            ffi::ack_buffer_free(dev, b),
            ffi::ACK_OK,
            "{}",
            last_error()
        );
    }
    assert_eq!(ffi::ack_close(dev), ffi::ACK_OK, "{}", last_error());
}

fn refused(what: &str, rc: i32, code: i32, name: &str) {
    assert_eq!(
        rc,
        code,
        "{what}: expected {code}, got {rc} ({})",
        last_error()
    );
    let e = last_error();
    assert!(
        e.contains(name),
        "{what}: the refusal does not name {name}: {e}"
    );
}

#[test]
fn the_attention_entries_refuse_by_name_before_recording() {
    refuse_by_name_before_recording(Elt::Bf16);
}

#[test]
fn the_f16_attention_entries_refuse_by_name_before_recording() {
    refuse_by_name_before_recording(Elt::F16);
}

fn refuse_by_name_before_recording(elt: Elt) {
    let Some(dev) = open_or_unmeasured("attention C ABI refusals") else {
        return;
    };
    let (fwd_entry, bwd_entry) = elt.entries();
    let zeros_q = vec![0u8; N_Q * 2];
    let zeros_kv = vec![0u8; N_KV * 2];
    let qb = buffer(dev, &zeros_q);
    let kb = buffer(dev, &zeros_kv);
    let vb = buffer(dev, &zeros_kv);
    let gb = buffer(dev, &zeros_q);
    let ob = buffer(dev, &zeros_q);
    let lb = buffer(dev, &vec![0u8; N_LSE * 4]);
    let short = buffer(dev, &vec![0u8; N_LSE * 4 - 4]);
    let dqb = buffer(dev, &zeros_q);
    let dkb = buffer(dev, &zeros_kv);
    let dvb = buffer(dev, &zeros_kv);
    let fwd = |s: &ffi::AckAttentionShape, o: *mut Buffer, l: *mut Buffer| {
        let mut ms = 0.0f64;
        let rc = unsafe { fwd_entry(dev, qb, kb, vb, o, l, s, &mut ms) };
        assert!(ms.is_nan(), "every call writes NaN first, got {ms}");
        rc
    };

    let rc = unsafe {
        fwd_entry(
            dev,
            qb,
            kb,
            vb,
            ob,
            lb,
            std::ptr::null(),
            std::ptr::null_mut(),
        )
    };
    assert_eq!(rc, ffi::ACK_ERR_NULL, "a null shape block");
    let mut s = shape();
    s.reserved = 1;
    refused(
        "reserved word",
        fwd(&s, ob, lb),
        ffi::ACK_ERR_SHAPE,
        "reserved word must be 0",
    );
    let mut s = shape();
    s.causal = 2;
    refused(
        "causal 2",
        fwd(&s, ob, lb),
        ffi::ACK_ERR_SHAPE,
        "causal 0 or 1",
    );
    let mut s = shape();
    s.head_dim = 32;
    refused(
        "head dim 32",
        fwd(&s, ob, lb),
        ffi::ACK_ERR_SHAPE,
        "attention-head-dim-unsupported",
    );
    let mut s = shape();
    s.kv_heads = 3;
    refused(
        "3 kv heads",
        fwd(&s, ob, lb),
        ffi::ACK_ERR_SHAPE,
        "attention-kv-heads-do-not-divide",
    );
    let mut s = shape();
    s.q = [QS[0], QS[1] + 1, QS[2]];
    refused(
        "odd q stride",
        fwd(&s, ob, lb),
        ffi::ACK_ERR_SHAPE,
        "attention-odd-stride",
    );
    refused(
        "short lse",
        fwd(&shape(), ob, short),
        ffi::ACK_ERR_SHAPE,
        "attention-operand-too-small",
    );
    refused(
        "o is q",
        fwd(&shape(), qb, lb),
        ffi::ACK_ERR_SHAPE,
        "attention-output-aliases-input",
    );
    refused(
        "o is lse",
        fwd(&shape(), ob, ob),
        ffi::ACK_ERR_SHAPE,
        "attention-outputs-alias",
    );

    let block = ffi::AckAttentionBackwardShape {
        forward: shape(),
        dout: QS,
        dq: QS,
        dk: KS,
        dv: KS,
    };
    let bwd = |dq: *mut Buffer, dk: *mut Buffer, dv: *mut Buffer| {
        let mut ms = 0.0f64;
        unsafe { bwd_entry(dev, qb, kb, vb, ob, gb, lb, dq, dk, dv, &block, &mut ms) }
    };
    refused(
        "dK is dV",
        bwd(dqb, dkb, dkb),
        ffi::ACK_ERR_SHAPE,
        "attention-outputs-alias",
    );
    refused(
        "dQ is dO",
        bwd(gb, dkb, dvb),
        ffi::ACK_ERR_SHAPE,
        "attention-output-aliases-input",
    );
    let rc = unsafe {
        bwd_entry(
            dev,
            qb,
            kb,
            vb,
            ob,
            gb,
            lb,
            dqb,
            dkb,
            dvb,
            std::ptr::null(),
            std::ptr::null_mut(),
        )
    };
    assert_eq!(rc, ffi::ACK_ERR_NULL, "a null backward block");

    // nothing above was recorded: the same device still runs both entries
    assert_eq!(fwd(&shape(), ob, lb), ffi::ACK_OK, "{}", last_error());
    assert_eq!(bwd(dqb, dkb, dvb), ffi::ACK_OK, "{}", last_error());
    let _ = download(dev, dvb, N_KV * 2);
    for b in [qb, kb, vb, gb, ob, lb, short, dqb, dkb, dvb] {
        assert_eq!(
            ffi::ack_buffer_free(dev, b),
            ffi::ACK_OK,
            "{}",
            last_error()
        );
    }
    assert_eq!(ffi::ack_close(dev), ffi::ACK_OK, "{}", last_error());
}
