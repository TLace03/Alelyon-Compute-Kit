//! `ack_intwise` schema 2 (2026-09-28), the binary64 module: torch's float64
//! pointwise arithmetic (C = f64) on a real device, through the C ABI, bit for
//! bit against a host reference.
//!
//! THE REFERENCE SHARES NO ARITHMETIC WITH THE SHADER. The shader computes IEEE
//! 754 binary64 in 32-bit integer words (`kernels/intwise.comp` built with
//! `-DF64=1`). The reference is this host's own binary64 arithmetic (x86-64
//! SSE2: round to nearest even, subnormals kept), with the NaN rules written out
//! rather than left to the compiler, which may commute `a + b`. They are torch's
//! CPU's, MEASURED 2026-09-28 on torch 2.14 (its vector body, scalar tail and
//! strided loop agree): a NaN operand comes out quieted; where both operands
//! are NaN, mul and div return x's (x86 SSE's first operand) and add and sub
//! return y's, because torch computes x + alpha*y and forms the product first;
//! an invalid operation (inf - inf, 0 * inf, 0/0, inf/inf) returns x86's
//! default NaN, 0xFFF8000000000000; neg and abs flip or clear the sign and do
//! not quiet. Operands are widened as torch casts them to f64: f32 exactly (a
//! NaN quieted, as cvtss2sd does), bf16 and f16 through f32 (f16's subnormals
//! by a float multiply, not the shader's bit shuffle), the integers with Rust's
//! `as f64` (int64 rounds to nearest even), bool as 0 or 1.
//!
//! INPUTS: every pair drawn from a list of edge values (both zeros, the
//! smallest and largest subnormals, the smallest normal, one and its
//! neighbours, half and quarter ulps of one, the largest finite values, both
//! infinities, quiet and signalling NaNs of both signs with payloads), then a
//! fixed pseudo-random set: raw bit patterns over every exponent, operands a
//! few exponents apart (alignment and cancellation), constructed exact ties in
//! add and mul, products and quotients that land among the subnormals, and
//! ones that overflow. Outputs are poisoned first, so an element the kernel
//! never wrote cannot pass, and the bytes past a bool output's last element
//! must come back as they were.
//!
//! Without a device every test prints UNMEASURED (`ACK_REQUIRE_DEVICE=1`
//! turns that into a failure).
use alelyon_compute_kit::convert_ops::Code;
use alelyon_compute_kit::ffi;
use alelyon_compute_kit::intwise_ops::{IntOp, F64_MINUS_ONE, F64_PLUS_ONE};
use std::ffi::{c_char, CStr};

fn last_error() -> String {
    let mut buf = vec![0 as c_char; 1024];
    let rc = unsafe { ffi::ack_last_error(buf.as_mut_ptr(), buf.len()) };
    assert!(rc >= 0, "ack_last_error returned {rc}");
    unsafe { CStr::from_ptr(buf.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

fn open_or_unmeasured(what: &str) -> Option<Device> {
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
    Some(Device { dev, live: vec![] })
}

type Buf = *mut alelyon_compute_kit::Buffer;

struct Device {
    dev: *mut ffi::AckDevice,
    live: Vec<Buf>,
}

const POISON: u8 = 0xA5;

impl Device {
    /// A buffer holding `bytes`, padded with `pad` to a whole word.
    fn upload(&mut self, bytes: &[u8], pad: u8) -> Buf {
        let mut data = bytes.to_vec();
        while data.len() % 4 != 0 || data.is_empty() {
            data.push(pad);
        }
        let buf = ffi::ack_buffer_alloc(self.dev, data.len() as u64);
        assert!(!buf.is_null(), "{}", last_error());
        let rc = unsafe { ffi::ack_upload(self.dev, buf, data.as_ptr(), data.len()) };
        assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
        self.live.push(buf);
        buf
    }

    fn poisoned(&mut self, len: usize) -> Buf {
        self.upload(&vec![POISON; len], POISON)
    }

    fn download(&self, buf: Buf, len: usize) -> Vec<u8> {
        let mut data = vec![0u8; len.div_ceil(4) * 4];
        let rc = unsafe { ffi::ack_download(self.dev, buf, data.as_mut_ptr(), data.len()) };
        assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
        data
    }

    fn intwise(&self, x: Buf, y: Buf, out: Buf, r: &ffi::AckIntwiseRequest) -> i32 {
        let mut ms = 0.0f64;
        unsafe {
            ffi::ack_intwise(
                self.dev,
                x,
                y,
                out,
                r,
                std::mem::size_of::<ffi::AckIntwiseRequest>(),
                &mut ms,
            )
        }
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        for buf in self.live.drain(..) {
            assert_eq!(ffi::ack_buffer_free(self.dev, buf), ffi::ACK_OK);
        }
        assert_eq!(ffi::ack_close(self.dev), ffi::ACK_OK, "{}", last_error());
    }
}

// ---- the reference -----------------------------------------------------------------

const SIGN: u64 = 1 << 63;
const QUIET: u64 = 1 << 51;
const ONE: u64 = 0x3FF0_0000_0000_0000;
/// x86's "QNaN floating-point indefinite", the result of an invalid operation
const DEFAULT_NAN: u64 = 0xFFF8_0000_0000_0000;

fn is_nan(v: u64) -> bool {
    (v >> 52) & 0x7FF == 0x7FF && v & 0x000F_FFFF_FFFF_FFFF != 0
}

/// A binary operation as x86 SSE computes it: the first NaN operand, quieted;
/// an invalid operation, the default NaN; otherwise the IEEE result.
fn sse(a: u64, b: u64, f: fn(f64, f64) -> f64) -> u64 {
    if is_nan(a) {
        return a | QUIET;
    }
    if is_nan(b) {
        return b | QUIET;
    }
    let r = f(f64::from_bits(a), f64::from_bits(b));
    if r.is_nan() {
        DEFAULT_NAN
    } else {
        r.to_bits()
    }
}

/// add and sub as torch's CPU computes them, x + alpha*y: the product is
/// formed first, so y's NaN (quieted, its sign kept) wins over x's.
fn plus(a: u64, b: u64, f: fn(f64, f64) -> f64) -> u64 {
    if is_nan(b) {
        return b | QUIET;
    }
    sse(a, b, f)
}

/// torch's float64 answer for one element; `minus` is alpha = -1.0.
fn reference(op: IntOp, a: u64, b: u64, minus: bool) -> u64 {
    let (fa, fb) = (f64::from_bits(a), f64::from_bits(b));
    let flag = |c: bool| u64::from(c);
    match op {
        IntOp::Add if minus => plus(a, b, |x, y| x - y),
        IntOp::Add => plus(a, b, |x, y| x + y),
        IntOp::Sub if minus => plus(a, b, |x, y| x + y),
        IntOp::Sub => plus(a, b, |x, y| x - y),
        IntOp::Mul => sse(a, b, |x, y| x * y),
        IntOp::Div => sse(a, b, |x, y| x / y),
        IntOp::Reciprocal => sse(ONE, a, |x, y| x / y),
        IntOp::Neg => a ^ SIGN,
        IntOp::Abs => a & !SIGN,
        IntOp::Eq => flag(fa == fb),
        IntOp::Ne => flag(fa != fb),
        IntOp::Lt => flag(fa < fb),
        IntOp::Le => flag(fa <= fb),
        IntOp::Gt => flag(fa > fb),
        IntOp::Ge => flag(fa >= fb),
        IntOp::Fill => b,
        other => unreachable!("{other:?} is not a binary64 operation"),
    }
}

/// f32 bits -> f64 bits: exact, and a NaN quieted with its payload moved up
/// (x86's cvtss2sd), written out rather than left to `as`
fn f32_to_f64(w: u32) -> u64 {
    let w = u64::from(w);
    if w & 0x7F80_0000 == 0x7F80_0000 && w & 0x7F_FFFF != 0 {
        return ((w & 0x8000_0000) << 32) | 0x7FF8_0000_0000_0000 | ((w & 0x3F_FFFF) << 29);
    }
    f64::from(f32::from_bits(w as u32)).to_bits()
}

/// f16 bits -> f32 bits (c10's Half to float): a subnormal by a float multiply
fn f16_to_f32(h: u16) -> u32 {
    let sign = u32::from(h & 0x8000) << 16;
    let e = u32::from((h >> 10) & 0x1F);
    let m = u32::from(h & 0x3FF);
    if e == 0x1F {
        return sign | 0x7F80_0000 | (m << 13);
    }
    if e == 0 {
        return sign | (m as f32 * 2f32.powi(-24)).to_bits();
    }
    sign | ((e + 112) << 23) | (m << 13)
}

/// A storage type of an operand, its raw bits per element held in a u64.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Src {
    F64,
    F32,
    Bf16,
    F16,
    I64,
    I32,
    I16,
    I8,
    U8,
    Bool,
}

impl Src {
    fn code(self) -> Code {
        match self {
            Src::F64 => Code::F64,
            Src::F32 => Code::F32,
            Src::Bf16 => Code::Bf16,
            Src::F16 => Code::F16,
            Src::I64 => Code::I64,
            Src::I32 => Code::I32,
            Src::I16 => Code::I16,
            Src::I8 => Code::I8,
            Src::U8 => Code::U8,
            Src::Bool => Code::Bool,
        }
    }

    fn bytes(self) -> usize {
        match self {
            Src::F64 | Src::I64 => 8,
            Src::F32 | Src::I32 => 4,
            Src::Bf16 | Src::F16 | Src::I16 => 2,
            Src::I8 | Src::U8 | Src::Bool => 1,
        }
    }

    fn encode(self, raw: &[u64]) -> Vec<u8> {
        raw.iter()
            .flat_map(|v| v.to_le_bytes()[..self.bytes()].to_vec())
            .collect()
    }

    /// torch's cast of one element to f64, as bits
    fn widen(self, raw: u64) -> u64 {
        match self {
            Src::F64 => raw,
            Src::F32 => f32_to_f64(raw as u32),
            Src::Bf16 => f32_to_f64((raw as u32 & 0xFFFF) << 16),
            Src::F16 => f32_to_f64(f16_to_f32(raw as u16)),
            Src::I64 => (raw as i64 as f64).to_bits(),
            Src::I32 => (raw as u32 as i32 as f64).to_bits(),
            Src::I16 => (raw as u16 as i16 as f64).to_bits(),
            Src::I8 => (raw as u8 as i8 as f64).to_bits(),
            Src::U8 => (raw as u8 as f64).to_bits(),
            Src::Bool => {
                if raw & 0xFF != 0 {
                    ONE
                } else {
                    0
                }
            }
        }
    }
}

// ---- inputs ------------------------------------------------------------------------

fn edges() -> Vec<u64> {
    vec![
        0x0000_0000_0000_0000, // +0
        0x8000_0000_0000_0000, // -0
        0x0000_0000_0000_0001, // the smallest subnormal
        0x8000_0000_0000_0001,
        0x000F_FFFF_FFFF_FFFF, // the largest subnormal
        0x800F_FFFF_FFFF_FFFF,
        0x0008_0000_0000_0000, // a subnormal, half the smallest normal
        0x0010_0000_0000_0000, // the smallest normal
        0x8010_0000_0000_0000,
        0x0010_0000_0000_0001,
        0x001F_FFFF_FFFF_FFFF,
        0x3FF0_0000_0000_0000, // 1
        0xBFF0_0000_0000_0000, // -1
        0x3FF0_0000_0000_0001, // 1 + ulp
        0x3FEF_FFFF_FFFF_FFFF, // 1 - ulp/2
        0x4000_0000_0000_0000, // 2
        0x3FE0_0000_0000_0000, // 0.5
        0x3FF8_0000_0000_0000, // 1.5
        0x4008_0000_0000_0000, // 3
        0x3CA0_0000_0000_0000, // 2^-53, half an ulp of 1
        0x3CB0_0000_0000_0000, // 2^-52, an ulp of 1
        0x3C90_0000_0000_0000, // 2^-54
        0x3CA8_0000_0000_0000, // 3 * 2^-54
        0x7FEF_FFFF_FFFF_FFFF, // the largest finite
        0xFFEF_FFFF_FFFF_FFFF,
        0x7FE0_0000_0000_0000, // 2^1023
        0x7FF0_0000_0000_0000, // +inf
        0xFFF0_0000_0000_0000, // -inf
        0x7FF8_0000_0000_0000, // a quiet NaN
        0xFFF8_0000_0000_0000, // -qNaN, x86's default NaN
        0x7FF0_0000_0000_0001, // a signalling NaN, payload 1
        0xFFF4_0000_0000_0123, // -sNaN with a payload
        0x7FFC_DEAD_BEEF_0001, // a quiet NaN with a payload
        0x4059_0000_0000_0000, // 100
        0x3FB9_9999_9999_999A, // 0.1
        0xC00C_0000_0000_0000, // -3.5
        0x4340_0000_0000_0000, // 2^53
        0x4340_0000_0000_0001, // 2^53 + 2
        0x0360_0000_0000_0000, // 2^-969
        0x7C90_0000_0000_0000, // 2^969
    ]
}

/// xorshift64*: a fixed stream, so a failure reproduces
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// a finite normal with biased exponent in `lo..=hi`, either sign
    fn normal(&mut self, lo: u64, hi: u64) -> u64 {
        let r = self.next();
        let e = lo + (r >> 11) % (hi - lo + 1);
        (r & SIGN) | (e << 52) | (self.next() & 0x000F_FFFF_FFFF_FFFF)
    }
}

/// Every pair of the edge values, then the random pairs.
fn operands(seed: u64) -> (Vec<u64>, Vec<u64>) {
    let e = edges();
    let (mut a, mut b) = (vec![], vec![]);
    for &x in &e {
        for &y in &e {
            a.push(x);
            b.push(y);
        }
    }
    let mut rng = Rng(seed);
    for _ in 0..16_384 {
        // raw bit patterns: every exponent, NaNs and infinities among them
        a.push(rng.next());
        b.push(rng.next());
    }
    for _ in 0..16_384 {
        // a few exponents apart: alignment, and cancellation when the signs differ
        let x = rng.normal(0x3F0, 0x40F);
        let delta = (rng.next() % 7) as i64 - 3;
        let ey = (((x >> 52) & 0x7FF) as i64 + delta) as u64;
        a.push(x);
        b.push((rng.next() & SIGN) | (ey << 52) | (rng.next() & 0x000F_FFFF_FFFF_FFFF));
    }
    for _ in 0..4_096 {
        // exact ties: x plus half (or 3/2) of its ulp; 1 + k ulp times 1.5 with k odd
        let x = rng.normal(0x3F0, 0x40F);
        let ex = (x >> 52) & 0x7FF;
        let half = (ex - 53) << 52;
        a.push(x);
        b.push(if rng.next() & 1 == 0 {
            half
        } else {
            half | 0x0008_0000_0000_0000
        });
        a.push(ONE | ((rng.next() & 0xFFFF) << 1 | 1));
        b.push(0x3FF8_0000_0000_0000);
    }
    for _ in 0..8_192 {
        // tiny times tiny and tiny over huge: results among the subnormals
        a.push(rng.normal(0x1C0, 0x220));
        b.push(rng.normal(0x1C0, 0x220));
        a.push(rng.normal(0x001, 0x040));
        b.push(rng.normal(0x3FF, 0x440));
    }
    for _ in 0..4_096 {
        // huge times huge, huge plus huge: results that overflow
        a.push(rng.normal(0x5F0, 0x7FE));
        b.push(rng.normal(0x5F0, 0x7FE));
    }
    (a, b)
}

// ---- requests ------------------------------------------------------------------------

/// A contiguous 1-D request of `n` elements with C = f64.
fn flat(op: IntOp, x: Src, y: Src, out: Code, n: u64) -> ffi::AckIntwiseRequest {
    ffi::AckIntwiseRequest {
        op: op as u32,
        x_code: x.code() as u32,
        y_code: y.code() as u32,
        out_code: out as u32,
        compute_code: Code::F64 as u32,
        ndim: 1,
        flags: 0,
        reserved: 0,
        shape: [n, 0, 0, 0],
        x_offset: 0,
        x_strides: [1, 0, 0, 0],
        y_offset: 0,
        y_strides: [1, 0, 0, 0],
        out_offset: 0,
        out_strides: [1, 0, 0, 0],
        scalar: 0,
        alpha: F64_PLUS_ONE,
    }
}

fn out_code(op: IntOp) -> Code {
    if op.is_comparison() {
        Code::Bool
    } else {
        Code::F64
    }
}

/// Runs one contiguous request (y a tensor, or the scalar when `ys` is None)
/// and checks every element bit for bit and the bytes past the last one.
#[allow(clippy::too_many_arguments)]
fn check(
    dev: &mut Device,
    op: IntOp,
    (x, y): (Src, Src),
    xs: &[u64],
    ys: Option<&[u64]>,
    scalar: u64,
    minus: bool,
) {
    let n = xs.len();
    let out = out_code(op);
    let xb = dev.upload(&x.encode(xs), 0);
    let yb = match ys {
        Some(ys) => dev.upload(&y.encode(ys), 0),
        None => xb,
    };
    let es = if out == Code::F64 { 8 } else { 1 };
    let ob = dev.poisoned(n * es + 3);
    let mut r = flat(op, x, y, out, n as u64);
    r.scalar = scalar as i64;
    r.alpha = if minus { F64_MINUS_ONE } else { F64_PLUS_ONE };
    if ys.is_none() && op.has_y() {
        r.flags = 1;
    }
    let rc = dev.intwise(xb, yb, ob, &r);
    assert_eq!(rc, ffi::ACK_OK, "{op:?} {x:?},{y:?}: {}", last_error());
    let got = dev.download(ob, n * es + 3);
    for (i, &raw) in xs.iter().enumerate() {
        let a = x.widen(raw);
        let b = ys.map_or(scalar, |ys| y.widen(ys[i]));
        let want = reference(op, a, b, minus);
        let have = if es == 8 {
            u64::from_le_bytes(got[8 * i..8 * i + 8].try_into().unwrap())
        } else {
            u64::from(got[i])
        };
        assert_eq!(
            have, want,
            "{op:?} alpha {} {x:?}({raw:#018x} = {a:#018x}) , {y:?}({b:#018x}): got {have:#018x}, want {want:#018x}, element {i}",
            if minus { "-1" } else { "+1" }
        );
    }
    assert!(
        got[n * es..n * es + 3].iter().all(|&byte| byte == POISON),
        "{op:?}: the bytes past the last element changed"
    );
}

const BINARY: [IntOp; 10] = [
    IntOp::Add,
    IntOp::Sub,
    IntOp::Mul,
    IntOp::Div,
    IntOp::Eq,
    IntOp::Ne,
    IntOp::Lt,
    IntOp::Le,
    IntOp::Gt,
    IntOp::Ge,
];

#[test]
fn every_binary64_operation_matches_the_host_bit_for_bit() {
    let Some(mut dev) = open_or_unmeasured("intwise binary64 operations") else {
        return;
    };
    let (a, b) = operands(0x9E37_79B9_7F4A_7C15);
    for op in BINARY {
        check(&mut dev, op, (Src::F64, Src::F64), &a, Some(&b), 0, false);
        if matches!(op, IntOp::Add | IntOp::Sub) {
            check(&mut dev, op, (Src::F64, Src::F64), &a, Some(&b), 0, true);
        }
    }
    for op in [IntOp::Neg, IntOp::Abs, IntOp::Reciprocal] {
        check(&mut dev, op, (Src::F64, Src::F64), &a, None, 0, false);
        check(&mut dev, op, (Src::F64, Src::F64), &b, None, 0, false);
    }
}

#[test]
fn a_scalar_y_is_the_same_f64_in_every_element() {
    let Some(mut dev) = open_or_unmeasured("intwise binary64 scalar y") else {
        return;
    };
    let (a, _) = operands(0xD1B5_4A32_D192_ED03);
    let a = &a[..4_096];
    for scalar in edges() {
        for op in BINARY {
            check(&mut dev, op, (Src::F64, Src::F64), a, None, scalar, false);
        }
    }
}

#[test]
fn operands_of_every_storage_code_are_widened_as_torch_casts_them() {
    let Some(mut dev) = open_or_unmeasured("intwise binary64 widening") else {
        return;
    };
    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    let (_, ys) = operands(0xA076_1D64_78BD_642F);
    for src in [
        Src::F32,
        Src::Bf16,
        Src::F16,
        Src::I64,
        Src::I32,
        Src::I16,
        Src::I8,
        Src::U8,
        Src::Bool,
    ] {
        let mut xs: Vec<u64> = match src {
            // every f32 class, NaNs with payloads among them
            Src::F32 => vec![
                0,
                0x8000_0000,
                1,
                0x007F_FFFF,
                0x0080_0000,
                0x3F80_0000,
                0x7F7F_FFFF,
                0x7F80_0000,
                0xFF80_0000,
                0x7FC0_0000,
                0x7F80_0001,
                0xFFA0_0123,
            ],
            Src::Bf16 | Src::F16 => vec![
                0, 0x8000, 1, 0x03FF, 0x0400, 0x3C00, 0x7BFF, 0x7C00, 0xFC00, 0x7E00, 0x7C01,
                0xFD23, 0x7F80, 0x7FC1, 0xFF81,
            ],
            // around 2^53, where int64 -> f64 starts to round, and the extremes
            Src::I64 => vec![
                0,
                1,
                u64::MAX,
                (1 << 53) + 1,
                (1 << 53) + 3,
                (1 << 54) + 2,
                (1 << 54) + 6,
                i64::MAX as u64,
                i64::MIN as u64,
                (-(1i64 << 53) - 1) as u64,
            ],
            _ => vec![
                0,
                1,
                0x7F,
                0x80,
                0xFF,
                0x7FFF,
                0x8000,
                0xFFFF,
                0x7FFF_FFFF,
                0x8000_0000,
                0xFFFF_FFFF,
            ],
        };
        while xs.len() < 4_096 {
            let r = rng.next();
            xs.push(if src == Src::Bool { r & 1 } else { r });
        }
        let y = &ys[..xs.len()];
        for op in [IntOp::Add, IntOp::Mul, IntOp::Lt, IntOp::Eq] {
            check(&mut dev, op, (src, Src::F64), &xs, Some(y), 0, false);
            check(&mut dev, op, (Src::F64, src), y, Some(&xs), 0, false);
        }
    }
}

#[test]
fn fill_writes_the_scalars_bits_in_every_element() {
    let Some(mut dev) = open_or_unmeasured("intwise binary64 fill") else {
        return;
    };
    for scalar in edges() {
        for n in [1usize, 7, 1000] {
            let xs = vec![0u64; n];
            check(
                &mut dev,
                IntOp::Fill,
                (Src::F64, Src::F64),
                &xs,
                None,
                scalar,
                false,
            );
        }
    }
}

#[test]
fn what_the_binary64_module_does_not_have_is_refused_by_name_and_writes_nothing() {
    let Some(mut dev) = open_or_unmeasured("intwise binary64 refusals") else {
        return;
    };
    let xb = dev.upload(&Src::F64.encode(&[ONE; 8]), 0);
    let ob = dev.poisoned(64);
    let cases: [(ffi::AckIntwiseRequest, &str); 6] = [
        (
            flat(IntOp::Add, Src::F64, Src::F64, Code::F32, 8),
            "intwise-float-output",
        ),
        (
            flat(IntOp::Lt, Src::F64, Src::F64, Code::F64, 8),
            "intwise-float-output",
        ),
        (
            ffi::AckIntwiseRequest {
                alpha: 0x3FE0_0000_0000_0000, // 0.5
                ..flat(IntOp::Add, Src::F64, Src::F64, Code::F64, 8)
            },
            "intwise-float-alpha",
        ),
        (
            ffi::AckIntwiseRequest {
                alpha: 1, // the integer 1, not 1.0's bits
                ..flat(IntOp::Sub, Src::F64, Src::F64, Code::F64, 8)
            },
            "intwise-float-alpha",
        ),
        (
            flat(IntOp::And, Src::F64, Src::F64, Code::F64, 8),
            "intwise-op-not-for-compute",
        ),
        (
            ffi::AckIntwiseRequest {
                compute_code: Code::I64 as u32,
                x_code: Code::I64 as u32,
                y_code: Code::I64 as u32,
                out_code: Code::I64 as u32,
                ..flat(IntOp::Div, Src::I64, Src::I64, Code::I64, 8)
            },
            "intwise-op-not-for-compute",
        ),
    ];
    for (r, name) in cases {
        let rc = dev.intwise(xb, xb, ob, &r);
        assert_eq!(rc, ffi::ACK_ERR_SHAPE, "{name}");
        assert!(last_error().contains(name), "{name}: {}", last_error());
    }
    assert!(
        dev.download(ob, 64).iter().all(|&byte| byte == POISON),
        "a refused request wrote to its output"
    );
}
