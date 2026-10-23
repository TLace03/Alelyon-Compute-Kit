//! `ack_intwise` (ABI 14): every operation on every integer compute type, on a
//! real device, through the C ABI, bit for bit against host references.
//!
//! THE REFERENCE SHARES NO ARITHMETIC WITH THE SHADER. The shader carries
//! every value as a 64-bit integer in two 32-bit words and wraps it to C; the
//! reference computes in Rust's own type for C (i64, i32, i16, i8, u8, bool)
//! with its wrapping operations, converts each operand to that type with `as`
//! (c10's static_cast), and writes the result to the output's type with `as`
//! (bool: nonzero). The shifts are c10's BinaryOpsKernel rules written out
//! per type: a right shift by a negative amount (read as C) or by at least
//! width - 1 (signed) or width (u8) is a shift by that bound, and a left shift
//! by a negative amount or at least the width is 0. bool's add is a logical
//! or (with alpha) and its mul a logical and, as C++'s bool arithmetic gives.
//! The divisions are c10's integral kernels written out with Rust's own
//! division and remainder: div_floor_integer (subtract one from the quotient
//! where the signs differ and the remainder is nonzero), Python's remainder
//! (add the divisor where the remainder's sign differs from it), fmod (the
//! remainder). An i8 or i16 MIN / -1 wraps, as C++'s promoted division
//! truncated to the type does; the plan refuses -1 in i32 and i64.
//!
//! The inputs are each type's boundary values (0, +-1, MIN, MAX, their
//! neighbours, the shift boundaries) paired with each other, and a fixed
//! pseudo-random set. Outputs are poisoned first, so an element the kernel
//! never wrote cannot pass, and the bytes past a partly filled last output
//! word must come back as they were.
//!
//! Without a device every test prints UNMEASURED (`ACK_REQUIRE_DEVICE=1`
//! turns that into a failure).
use alelyon_compute_kit::convert_ops::Code;
use alelyon_compute_kit::ffi;
use alelyon_compute_kit::intwise_ops::IntOp;
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

// ---- types, values and storage ---------------------------------------------------

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum T {
    I64,
    I32,
    I16,
    I8,
    U8,
    Bool,
}

const TYPES: [T; 6] = [T::I64, T::I32, T::I16, T::I8, T::U8, T::Bool];

impl T {
    fn code(self) -> Code {
        match self {
            T::I64 => Code::I64,
            T::I32 => Code::I32,
            T::I16 => Code::I16,
            T::I8 => Code::I8,
            T::U8 => Code::U8,
            T::Bool => Code::Bool,
        }
    }

    fn bytes(self) -> usize {
        match self {
            T::I64 => 8,
            T::I32 => 4,
            T::I16 => 2,
            _ => 1,
        }
    }

    /// `v` stored as this type (c10's static_cast; bool is nonzero), then read
    /// back as an i64 by sign or zero extension.
    fn cast(self, v: i64) -> i64 {
        match self {
            T::I64 => v,
            T::I32 => v as i32 as i64,
            T::I16 => v as i16 as i64,
            T::I8 => v as i8 as i64,
            T::U8 => v as u8 as i64,
            T::Bool => (v != 0) as i64,
        }
    }

    fn encode(self, values: &[i64]) -> Vec<u8> {
        let mut out = Vec::with_capacity(values.len() * self.bytes());
        for &v in values {
            match self {
                T::I64 => out.extend_from_slice(&v.to_le_bytes()),
                T::I32 => out.extend_from_slice(&(v as i32).to_le_bytes()),
                T::I16 => out.extend_from_slice(&(v as i16).to_le_bytes()),
                T::I8 | T::U8 => out.push(v as u8),
                T::Bool => out.push((v != 0) as u8),
            }
        }
        out
    }

    fn read(self, bytes: &[u8], i: usize) -> i64 {
        let b = &bytes[i * self.bytes()..(i + 1) * self.bytes()];
        match self {
            T::I64 => i64::from_le_bytes(b.try_into().unwrap()),
            T::I32 => i32::from_le_bytes(b.try_into().unwrap()) as i64,
            T::I16 => i16::from_le_bytes(b.try_into().unwrap()) as i64,
            T::I8 => b[0] as i8 as i64,
            T::U8 | T::Bool => b[0] as i64,
        }
    }

    /// Boundary values of this type, and a pseudo-random set.
    fn inputs(self, random: usize, seed: u64) -> Vec<i64> {
        let mut v: Vec<i64> = match self {
            T::Bool => vec![0, 1],
            T::U8 => vec![0, 1, 2, 7, 8, 9, 127, 128, 200, 254, 255],
            _ => {
                let bits = self.bytes() as u32 * 8;
                let min = if bits == 64 {
                    i64::MIN
                } else {
                    -(1i64 << (bits - 1))
                };
                let max = if bits == 64 {
                    i64::MAX
                } else {
                    (1i64 << (bits - 1)) - 1
                };
                let mut v = vec![0, 1, -1, 2, -2, 5, -7, min, min + 1, max, max - 1];
                for s in [
                    bits as i64 - 2,
                    bits as i64 - 1,
                    bits as i64,
                    bits as i64 + 1,
                ] {
                    v.push(self.cast(s));
                    v.push(self.cast(-s));
                }
                v.push(self.cast(0x5555_5555_5555_5555));
                v
            }
        };
        let mut state = seed | 1;
        for _ in 0..random {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            v.push(self.cast(state as i64));
        }
        v
    }
}

// ---- the reference: Rust's own arithmetic in C -------------------------------------

macro_rules! signed_reference {
    ($name:ident, $t:ty, $ut:ty, $bits:expr) => {
        fn $name(op: IntOp, a: i64, b: i64, alpha: i64) -> i64 {
            let (a, b, alpha) = (a as $t, b as $t, alpha as $t);
            let r: $t = match op {
                IntOp::Add => a.wrapping_add(alpha.wrapping_mul(b)),
                IntOp::Sub => a.wrapping_sub(alpha.wrapping_mul(b)),
                IntOp::Mul => a.wrapping_mul(b),
                IntOp::And => a & b,
                IntOp::Or => a | b,
                IntOp::Xor => a ^ b,
                IntOp::Shl => {
                    if !(0..$bits).contains(&b) {
                        0
                    } else {
                        ((a as $ut) << b) as $t
                    }
                }
                IntOp::Shr => {
                    if !(0..$bits - 1).contains(&b) {
                        a >> ($bits - 1)
                    } else {
                        a >> b
                    }
                }
                IntOp::Eq => (a == b) as $t,
                IntOp::Ne => (a != b) as $t,
                IntOp::Lt => (a < b) as $t,
                IntOp::Le => (a <= b) as $t,
                IntOp::Gt => (a > b) as $t,
                IntOp::Ge => (a >= b) as $t,
                IntOp::Neg => a.wrapping_neg(),
                IntOp::Abs => a.wrapping_abs(),
                IntOp::DivTrunc => a.wrapping_div(b),
                IntOp::DivFloor => {
                    let q = a.wrapping_div(b);
                    if (a < 0) != (b < 0) && a.wrapping_rem(b) != 0 {
                        q.wrapping_sub(1)
                    } else {
                        q
                    }
                }
                IntOp::Remainder => {
                    let r = a.wrapping_rem(b);
                    if r != 0 && (r < 0) != (b < 0) {
                        r.wrapping_add(b)
                    } else {
                        r
                    }
                }
                IntOp::Fmod => a.wrapping_rem(b),
                IntOp::Fill | IntOp::Arange | IntOp::Cumsum => {
                    unreachable!("not an elementwise operation")
                }
                IntOp::Tril | IntOp::Triu => {
                    unreachable!("a triangle is decided by coordinates: kept()")
                }
                IntOp::RankAsc | IntOp::RankDesc => {
                    unreachable!("a rank is decided by its row: ref_ranks()")
                }
                IntOp::Div | IntOp::Reciprocal => {
                    unreachable!("the binary64 module's: tests/intwise_f64_abi.rs")
                }
            };
            r as i64
        }
    };
}

signed_reference!(ref_i64, i64, u64, 64);
signed_reference!(ref_i32, i32, u32, 32);
signed_reference!(ref_i16, i16, u16, 16);
signed_reference!(ref_i8, i8, u8, 8);

fn ref_u8(op: IntOp, a: i64, b: i64, alpha: i64) -> i64 {
    let (a, b, alpha) = (a as u8, b as u8, alpha as u8);
    let r: u8 = match op {
        IntOp::Add => a.wrapping_add(alpha.wrapping_mul(b)),
        IntOp::Sub => a.wrapping_sub(alpha.wrapping_mul(b)),
        IntOp::Mul => a.wrapping_mul(b),
        IntOp::And => a & b,
        IntOp::Or => a | b,
        IntOp::Xor => a ^ b,
        // (int8_t)b < 0 or b >= 8: 0 either way for a u8
        IntOp::Shl => {
            if (b as i8) < 0 || b >= 8 {
                0
            } else {
                a << b
            }
        }
        IntOp::Shr => {
            if (b as i8) < 0 || b >= 8 {
                0
            } else {
                a >> b
            }
        }
        IntOp::Eq => (a == b) as u8,
        IntOp::Ne => (a != b) as u8,
        IntOp::Lt => (a < b) as u8,
        IntOp::Le => (a <= b) as u8,
        IntOp::Gt => (a > b) as u8,
        IntOp::Ge => (a >= b) as u8,
        IntOp::Neg => a.wrapping_neg(),
        IntOp::Abs => a,
        IntOp::DivTrunc | IntOp::DivFloor => a / b,
        IntOp::Remainder | IntOp::Fmod => a % b,
        IntOp::Fill | IntOp::Arange | IntOp::Cumsum => unreachable!("not an elementwise operation"),
        IntOp::Tril | IntOp::Triu => unreachable!("a triangle is decided by coordinates: kept()"),
        IntOp::RankAsc | IntOp::RankDesc => {
            unreachable!("a rank is decided by its row: ref_ranks()")
        }
        IntOp::Div | IntOp::Reciprocal => {
            unreachable!("the binary64 module's: tests/intwise_f64_abi.rs")
        }
    };
    r as i64
}

fn ref_bool(op: IntOp, a: i64, b: i64, alpha: i64) -> i64 {
    let (a, b, alpha) = (a != 0, b != 0, alpha != 0);
    let r = match op {
        IntOp::Add => a || (alpha && b),
        IntOp::Mul => a && b,
        IntOp::And => a & b,
        IntOp::Or => a | b,
        IntOp::Xor => a ^ b,
        IntOp::Eq => a == b,
        IntOp::Ne => a != b,
        IntOp::Lt => !a & b,
        IntOp::Le => a <= b,
        IntOp::Gt => a & !b,
        IntOp::Ge => a >= b,
        _ => unreachable!("torch refuses {op:?} on bool"),
    };
    r as i64
}

/// torch's answer for one element: operands converted to C, computed in C.
fn reference(op: IntOp, c: T, a: i64, b: i64, alpha: i64) -> i64 {
    match c {
        T::I64 => ref_i64(op, a, b, alpha),
        T::I32 => ref_i32(op, a, b, alpha),
        T::I16 => ref_i16(op, a, b, alpha),
        T::I8 => ref_i8(op, a, b, alpha),
        T::U8 => ref_u8(op, a, b, alpha),
        T::Bool => ref_bool(op, a, b, alpha),
    }
}

const BINARY: [IntOp; 14] = [
    IntOp::Add,
    IntOp::Sub,
    IntOp::Mul,
    IntOp::And,
    IntOp::Or,
    IntOp::Xor,
    IntOp::Shl,
    IntOp::Shr,
    IntOp::Eq,
    IntOp::Ne,
    IntOp::Lt,
    IntOp::Le,
    IntOp::Gt,
    IntOp::Ge,
];

fn is_compare(op: IntOp) -> bool {
    matches!(
        op,
        IntOp::Eq | IntOp::Ne | IntOp::Lt | IntOp::Le | IntOp::Gt | IntOp::Ge
    )
}

fn refused_on_bool(op: IntOp) -> bool {
    matches!(
        op,
        IntOp::Sub
            | IntOp::Neg
            | IntOp::Abs
            | IntOp::Shl
            | IntOp::Shr
            | IntOp::Arange
            | IntOp::DivTrunc
            | IntOp::DivFloor
            | IntOp::Remainder
            | IntOp::Fmod
    )
}

/// A contiguous 1-D request of `n` elements.
fn flat(op: IntOp, x: T, y: T, out: T, c: T, n: u64) -> ffi::AckIntwiseRequest {
    ffi::AckIntwiseRequest {
        op: op as u32,
        x_code: x.code() as u32,
        y_code: y.code() as u32,
        out_code: out.code() as u32,
        compute_code: c.code() as u32,
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
        alpha: 1,
    }
}

/// Every pair of `xs` and `ys` values, then the rest of the longer list.
fn pairs(xs: &[i64], ys: &[i64]) -> (Vec<i64>, Vec<i64>) {
    let (mut a, mut b) = (vec![], vec![]);
    for &x in xs.iter().take(24) {
        for &y in ys.iter().take(24) {
            a.push(x);
            b.push(y);
        }
    }
    for (i, &x) in xs.iter().enumerate() {
        a.push(x);
        b.push(ys[(i * 7 + 3) % ys.len()]);
    }
    (a, b)
}

/// Runs one contiguous request and checks every element and the bytes past it.
#[allow(clippy::too_many_arguments)]
fn check_flat(
    dev: &mut Device,
    op: IntOp,
    (x, y, out, c): (T, T, T, T),
    a: &[i64],
    b: Option<&[i64]>,
    scalar: i64,
    alpha: i64,
) {
    let n = a.len();
    let xb = dev.upload(&x.encode(a), 0);
    let yb = match b {
        Some(b) => dev.upload(&y.encode(b), 0),
        None => xb,
    };
    let out_len = n * out.bytes();
    let ob = dev.poisoned(out_len + 3);
    let mut r = flat(op, x, y, out, c, n as u64);
    r.scalar = scalar;
    r.alpha = alpha;
    if b.is_none() && op != IntOp::Neg && op != IntOp::Abs {
        r.flags = 1;
    }
    let rc = dev.intwise(xb, yb, ob, &r);
    assert_eq!(
        rc,
        ffi::ACK_OK,
        "{op:?} {x:?},{y:?}->{out:?} in {c:?}: {}",
        last_error()
    );
    let got = dev.download(ob, out_len + 3);
    for (i, &av) in a.iter().enumerate() {
        let bv = b.map_or(scalar, |b| b[i]);
        let want = out.cast(reference(op, c, c.cast(av), c.cast(bv), alpha));
        let have = out.read(&got, i);
        assert_eq!(
            have, want,
            "{op:?} {x:?}({av}) , {y:?}({bv}) alpha {alpha} in {c:?} -> {out:?}, element {i}"
        );
    }
    assert!(
        got[out_len..out_len + 3].iter().all(|&byte| byte == POISON),
        "{op:?} {c:?}: the bytes past the last element changed"
    );
}

#[test]
fn every_operation_on_every_compute_type_matches_the_host() {
    let Some(mut dev) = open_or_unmeasured("intwise every operation") else {
        return;
    };
    for c in TYPES {
        let (a, b) = pairs(&c.inputs(40, 0x9E37_79B9), &c.inputs(40, 0x85EB_CA6B));
        for op in BINARY {
            if c == T::Bool && refused_on_bool(op) {
                continue;
            }
            let out = if is_compare(op) { T::Bool } else { c };
            check_flat(&mut dev, op, (c, c, out, c), &a, Some(&b), 0, 1);
        }
        for alpha in [-3i64, 2] {
            if c != T::Bool {
                check_flat(&mut dev, IntOp::Add, (c, c, c, c), &a, Some(&b), 0, alpha);
                check_flat(&mut dev, IntOp::Sub, (c, c, c, c), &a, Some(&b), 0, alpha);
            }
        }
        if c != T::Bool {
            check_flat(&mut dev, IntOp::Neg, (c, c, c, c), &a, None, 0, 1);
            check_flat(&mut dev, IntOp::Abs, (c, c, c, c), &a, None, 0, 1);
        }
    }
}

#[test]
fn a_scalar_y_is_the_same_value_in_every_element() {
    let Some(mut dev) = open_or_unmeasured("intwise scalar y") else {
        return;
    };
    for c in [T::I64, T::I32, T::I16, T::U8, T::Bool] {
        let a = c.inputs(30, 0x2545_F491);
        for op in [
            IntOp::Add,
            IntOp::Mul,
            IntOp::And,
            IntOp::Shr,
            IntOp::Shl,
            IntOp::Lt,
            IntOp::Eq,
        ] {
            if c == T::Bool && refused_on_bool(op) {
                continue;
            }
            let out = if is_compare(op) { T::Bool } else { c };
            for s in [0i64, 1, 3, -1, 7] {
                let s = c.cast(s);
                check_flat(&mut dev, op, (c, c, out, c), &a, None, s, 1);
            }
        }
    }
}

#[test]
fn division_by_a_scalar_matches_the_host() {
    let Some(mut dev) = open_or_unmeasured("intwise division") else {
        return;
    };
    const DIVISIONS: [IntOp; 4] = [
        IntOp::DivTrunc,
        IntOp::DivFloor,
        IntOp::Remainder,
        IntOp::Fmod,
    ];
    for c in TYPES {
        if c == T::Bool {
            continue; // torch refuses a bool division
        }
        let a = c.inputs(60, 0x51ED_2701);
        let mut divisors: Vec<i64> = vec![];
        for d in [
            1i64,
            2,
            3,
            7,
            -2,
            -3,
            -7,
            127,
            -128,
            255,
            1 << 20,
            -(1 << 31),
            (1 << 31) - 1,
            i64::MAX,
            i64::MIN,
            i64::MIN + 1,
            -1,
            0x5555_5555_5555_5555,
        ] {
            let d = c.cast(d);
            // what the plan refuses: 0 in C, and -1 in i32 and i64
            if d != 0 && !(d == -1 && matches!(c, T::I32 | T::I64)) && !divisors.contains(&d) {
                divisors.push(d);
            }
        }
        for op in DIVISIONS {
            for &d in &divisors {
                check_flat(&mut dev, op, (c, c, c, c), &a, None, d, 1);
            }
        }
    }
    // an i8 tensor divided in an i32 C into an i64 output, and an i64
    // quotient narrowed to an i16 output
    let a = T::I8.inputs(40, 5);
    for op in DIVISIONS {
        check_flat(
            &mut dev,
            op,
            (T::I8, T::I32, T::I64, T::I32),
            &a,
            None,
            -3,
            1,
        );
    }
    let a = T::I64.inputs(40, 9);
    for op in DIVISIONS {
        check_flat(
            &mut dev,
            op,
            (T::I64, T::I64, T::I16, T::I64),
            &a,
            None,
            1000,
            1,
        );
    }
}

#[test]
fn mixed_operand_types_are_converted_to_the_common_type_first() {
    let Some(mut dev) = open_or_unmeasured("intwise mixed types") else {
        return;
    };
    // (x, y, C): torch's result_type of the two
    for (x, y, c) in [
        (T::I16, T::U8, T::I16),
        (T::U8, T::I8, T::I16),
        (T::Bool, T::I64, T::I64),
        (T::I32, T::I16, T::I32),
        (T::U8, T::Bool, T::U8),
    ] {
        let (a, b) = pairs(&x.inputs(20, 7), &y.inputs(20, 11));
        for op in [IntOp::Add, IntOp::Mul, IntOp::Sub, IntOp::Gt, IntOp::Or] {
            if op == IntOp::Sub && (x == T::Bool || y == T::Bool) {
                continue; // torch refuses a bool operand of sub
            }
            let out = if is_compare(op) { T::Bool } else { c };
            check_flat(&mut dev, op, (x, y, out, c), &a, Some(&b), 0, 1);
        }
    }
    // the output's own type after C: an i32 sum wraps at 32 bits before it
    // widens to an i64 output, and an i64 product narrows to an i16 one
    let (a, b) = pairs(&T::I32.inputs(20, 3), &T::I32.inputs(20, 5));
    check_flat(
        &mut dev,
        IntOp::Add,
        (T::I32, T::I32, T::I64, T::I32),
        &a,
        Some(&b),
        0,
        1,
    );
    let (a, b) = pairs(&T::I64.inputs(20, 3), &T::I64.inputs(20, 5));
    check_flat(
        &mut dev,
        IntOp::Mul,
        (T::I64, T::I64, T::I16, T::I64),
        &a,
        Some(&b),
        0,
        1,
    );
    check_flat(
        &mut dev,
        IntOp::Eq,
        (T::I64, T::I64, T::I32, T::I64),
        &a,
        Some(&b),
        0,
        1,
    );
}

#[test]
fn fill_and_arange_write_every_element() {
    let Some(mut dev) = open_or_unmeasured("intwise fill and arange") else {
        return;
    };
    for out in TYPES {
        for &(n, value) in &[(1usize, 0i64), (7, -1), (13, 3), (64, 0x7FFF_FFFF_FFFF)] {
            let ob = dev.poisoned(n * out.bytes() + 3);
            let mut r = flat(IntOp::Fill, out, out, out, out, n as u64);
            r.scalar = value;
            assert_eq!(dev.intwise(ob, ob, ob, &r), ffi::ACK_OK, "{}", last_error());
            let got = dev.download(ob, n * out.bytes() + 3);
            for i in 0..n {
                assert_eq!(
                    out.read(&got, i),
                    out.cast(value),
                    "fill {out:?} {value} [{i}]"
                );
            }
            assert!(got[n * out.bytes()..].iter().all(|&b| b == POISON));
        }
        if out == T::Bool {
            continue;
        }
        for &(n, start, step) in &[
            (10usize, 0i64, 1i64),
            (300, -7, 3),
            (33, 100, -9),
            (5, i64::MAX - 2, 1),
        ] {
            let ob = dev.poisoned(n * out.bytes() + 3);
            let mut r = flat(IntOp::Arange, out, out, out, out, n as u64);
            r.scalar = start;
            r.alpha = step;
            assert_eq!(dev.intwise(ob, ob, ob, &r), ffi::ACK_OK, "{}", last_error());
            let got = dev.download(ob, n * out.bytes() + 3);
            for i in 0..n {
                // accumulated in int64, then cast to the output (c10's arange)
                let want = out.cast(start.wrapping_add((i as i64).wrapping_mul(step)));
                assert_eq!(
                    out.read(&got, i),
                    want,
                    "arange {out:?} {start} {step} [{i}]"
                );
            }
        }
    }
}

#[test]
#[allow(clippy::needless_range_loop)]
fn strided_broadcast_and_in_place_views_address_what_they_name() {
    let Some(mut dev) = open_or_unmeasured("intwise strided views") else {
        return;
    };
    // x: [3, 4, 5] read transposed from a [5, 4, 3] block at offset 2; y: a
    // [4, 1] column broadcast over dims 0 and 2; out: every other element of a
    // [3, 4, 10] block (a strided sub-word output: atomics), then a dense one.
    let shape = [3u64, 4, 5];
    let n = 60usize;
    for c in [T::I16, T::U8, T::I64] {
        let base = c.inputs(80, 0xDEAD_BEEF);
        let xs: Vec<i64> = (0..62).map(|i| base[i % base.len()]).collect();
        let ys: Vec<i64> = (0..4).map(|i| base[(i * 13 + 5) % base.len()]).collect();
        let xb = dev.upload(&c.encode(&xs), 0);
        let yb = dev.upload(&c.encode(&ys), 0);
        let mut r = flat(IntOp::Mul, c, c, c, c, 0);
        r.ndim = 3;
        r.shape = [shape[0], shape[1], shape[2], 0];
        r.x_offset = 2;
        r.x_strides = [1, 3, 12, 0];
        r.y_offset = 0;
        r.y_strides = [0, 1, 0, 0];
        for (out_offset, out_strides, out_len) in
            [(1u64, [40u64, 10, 2, 0], 121usize), (0, [20, 5, 1, 0], 60)]
        {
            r.out_offset = out_offset;
            r.out_strides = out_strides;
            let ob = dev.poisoned(out_len * c.bytes());
            assert_eq!(dev.intwise(xb, yb, ob, &r), ffi::ACK_OK, "{}", last_error());
            let got = dev.download(ob, out_len * c.bytes());
            let mut written = vec![false; out_len];
            for i in 0..3 {
                for j in 0..4 {
                    for k in 0..5 {
                        let a = xs[2 + i + 3 * j + 12 * k];
                        let b = ys[j];
                        let e = (out_offset
                            + out_strides[0] * i as u64
                            + out_strides[1] * j as u64
                            + out_strides[2] * k as u64) as usize;
                        written[e] = true;
                        assert_eq!(
                            c.read(&got, e),
                            reference(IntOp::Mul, c, a, b, 1),
                            "{c:?} [{i},{j},{k}]"
                        );
                    }
                }
            }
            // every element the view does not name keeps its poison
            let poison = c.read(&[POISON; 8], 0);
            for (e, &w) in written.iter().enumerate() {
                if !w {
                    assert_eq!(
                        c.read(&got, e),
                        poison,
                        "{c:?}: element {e} was not in the view"
                    );
                }
            }
        }
        // in place: out is x's own view, x += 3 * y
        let own: Vec<i64> = (0..n).map(|i| base[(i * 5 + 1) % base.len()]).collect();
        let ob = dev.upload(&c.encode(&own), 0);
        let mut r = flat(IntOp::Add, c, c, c, c, n as u64);
        r.alpha = 3;
        r.y_strides = [0, 0, 0, 0];
        r.y_offset = 2;
        assert_eq!(dev.intwise(ob, yb, ob, &r), ffi::ACK_OK, "{}", last_error());
        let got = dev.download(ob, n * c.bytes());
        for (i, &v) in own.iter().enumerate() {
            assert_eq!(
                c.read(&got, i),
                reference(IntOp::Add, c, v, ys[2], 3),
                "{c:?} in place [{i}]"
            );
        }
    }
}

// ---- the scan: cumsum (schema 3) ---------------------------------------------------

// torch's cumsum for one row, computed in Rust's own type for C: each element
// converted to C with `as` (c10's static_cast of x to the output's dtype),
// summed with C's own wrapping add, every running sum kept. torch accumulates
// in int64 and casts each sum to C, which is the same value modulo C's width.
macro_rules! scan_reference {
    ($name:ident, $t:ty) => {
        fn $name(row: &[i64]) -> Vec<i64> {
            let mut acc: $t = 0;
            row.iter()
                .map(|&v| {
                    acc = acc.wrapping_add(v as $t);
                    acc as i64
                })
                .collect()
        }
    };
}

scan_reference!(scan_i64, i64);
scan_reference!(scan_i32, i32);
scan_reference!(scan_i16, i16);
scan_reference!(scan_i8, i8);
scan_reference!(scan_u8, u8);

fn scan_reference(c: T, row: &[i64]) -> Vec<i64> {
    match c {
        T::I64 => scan_i64(row),
        T::I32 => scan_i32(row),
        T::I16 => scan_i16(row),
        T::I8 => scan_i8(row),
        T::U8 => scan_u8(row),
        T::Bool => unreachable!("torch refuses cumsum on bool"),
    }
}

/// A contiguous [rows, len] cumsum of x (type `x`) in C, into `out`, checked
/// against the reference row by row, with the bytes past the output intact.
fn check_scan(dev: &mut Device, (x, out, c): (T, T, T), rows: usize, len: usize, seed: u64) {
    let base = x.inputs(97, seed);
    let values: Vec<i64> = (0..rows * len)
        .map(|i| base[(i * 7 + i / 5) % base.len()])
        .collect();
    let xb = dev.upload(&x.encode(&values), 0);
    let out_len = rows * len * out.bytes();
    let ob = dev.poisoned(out_len + 3);
    let mut r = flat(IntOp::Cumsum, x, c, out, c, 0);
    r.ndim = 2;
    r.shape = [rows as u64, len as u64, 0, 0];
    r.x_strides = [len as u64, 1, 0, 0];
    r.out_strides = [len as u64, 1, 0, 0];
    assert_eq!(
        dev.intwise(xb, ob, ob, &r),
        ffi::ACK_OK,
        "cumsum {x:?} in {c:?} -> {out:?} [{rows}, {len}]: {}",
        last_error()
    );
    let got = dev.download(ob, out_len + 3);
    for row in 0..rows {
        let want = scan_reference(c, &values[row * len..(row + 1) * len]);
        for (k, &w) in want.iter().enumerate() {
            let i = row * len + k;
            assert_eq!(
                out.read(&got, i),
                out.cast(w),
                "cumsum {x:?} in {c:?} -> {out:?} [{rows}, {len}]: row {row} element {k}"
            );
        }
    }
    assert!(
        got[out_len..].iter().all(|&byte| byte == POISON),
        "cumsum {c:?} [{rows}, {len}]: the bytes past the last element changed"
    );
}

#[test]
fn cumsum_along_the_last_dimension_matches_the_host() {
    let Some(mut dev) = open_or_unmeasured("intwise cumsum") else {
        return;
    };
    // row lengths on each side of the chunking: one element, fewer than the
    // 64 invocations, exactly 64, one chunk of 2 and a lone last element,
    // 128 and 129, and long rows with every invocation's chunk full but the last
    for c in [T::I64, T::I32, T::I16, T::I8, T::U8] {
        for (i, &(rows, len)) in [
            (1usize, 1usize),
            (3, 2),
            (2, 63),
            (2, 64),
            (3, 65),
            (1, 127),
            (2, 128),
            (2, 129),
            (1, 1000),
            (2, 4099),
            (5, 3),
        ]
        .iter()
        .enumerate()
        {
            check_scan(&mut dev, (c, c, c), rows, len, 0xC0FF_EE00 + i as u64);
        }
    }
}

#[test]
fn cumsum_converts_x_to_c_first_and_wraps_every_running_sum() {
    let Some(mut dev) = open_or_unmeasured("intwise cumsum conversions") else {
        return;
    };
    // (x, C): torch's default output for a bool or narrower integer x is
    // int64; dtype= narrows C, and x is cast to it before the sum (MEASURED
    // 2026-09-28, torch 2.14 CPU: int64 [2^40, 2^40, -2^41 + 5] in int32 gives
    // [0, 0, 5])
    for (x, c) in [
        (T::Bool, T::I64),
        (T::U8, T::I64),
        (T::I8, T::I64),
        (T::I16, T::I64),
        (T::I32, T::I64),
        (T::I64, T::I32),
        (T::I64, T::I8),
        (T::U8, T::I8),
        (T::I8, T::U8),
        (T::Bool, T::I16),
    ] {
        check_scan(&mut dev, (x, c, c), 3, 77, 0x5EED);
    }
    // the measured case itself
    let xb = dev.upload(&T::I64.encode(&[1 << 40, 1 << 40, -(1i64 << 41) + 5]), 0);
    let ob = dev.poisoned(12);
    let r = flat(IntOp::Cumsum, T::I64, T::I32, T::I32, T::I32, 3);
    assert_eq!(dev.intwise(xb, ob, ob, &r), ffi::ACK_OK, "{}", last_error());
    let got = dev.download(ob, 12);
    let have: Vec<i64> = (0..3).map(|i| T::I32.read(&got, i)).collect();
    assert_eq!(have, vec![0, 0, 5]);
}

#[test]
#[allow(clippy::needless_range_loop)]
fn cumsum_addresses_strided_views_many_rows_and_its_own_input_in_place() {
    let Some(mut dev) = open_or_unmeasured("intwise cumsum views") else {
        return;
    };
    // x: [2, 3, 4, 70] read with the scan's dimension strided (every 3rd of a
    // wider block) from an offset; out: a transposed block with a gap, so a
    // sub-word output is written with atomics on words it shares with the gap
    let shape = [2usize, 3, 4, 70];
    let n: usize = shape.iter().product();
    for c in [T::I64, T::I16, T::U8] {
        let base = c.inputs(120, 0xFACE);
        let x_st = [3 * 4 * 70 * 3, 4 * 70 * 3, 70 * 3, 3];
        let x_off = 5usize;
        let x_len = x_off + (0..4).map(|d| (shape[d] - 1) * x_st[d]).sum::<usize>() + 1;
        let xs: Vec<i64> = (0..x_len)
            .map(|i| base[(i * 11 + 3) % base.len()])
            .collect();
        let xb = dev.upload(&c.encode(&xs), 0);
        // out: dims (0, 1, 2, 3) laid out as [70, 4, 3, 2] with a stride-2 gap
        let o_st = [2usize, 4, 12, 48 * 2];
        let o_off = 1usize;
        let o_len = o_off + (0..4).map(|d| (shape[d] - 1) * o_st[d]).sum::<usize>() + 1;
        let ob = dev.poisoned(o_len * c.bytes());
        let mut r = flat(IntOp::Cumsum, c, c, c, c, 0);
        r.ndim = 4;
        r.shape = [2, 3, 4, 70];
        r.x_offset = x_off as u64;
        r.x_strides = x_st.map(|s| s as u64);
        r.out_offset = o_off as u64;
        r.out_strides = o_st.map(|s| s as u64);
        assert_eq!(
            dev.intwise(xb, ob, ob, &r),
            ffi::ACK_OK,
            "{c:?}: {}",
            last_error()
        );
        let got = dev.download(ob, o_len * c.bytes());
        let mut written = vec![false; o_len];
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    let row: Vec<i64> = (0..70)
                        .map(|l| xs[x_off + i * x_st[0] + j * x_st[1] + k * x_st[2] + l * x_st[3]])
                        .collect();
                    let want = scan_reference(c, &row);
                    for l in 0..70 {
                        let e = o_off + i * o_st[0] + j * o_st[1] + k * o_st[2] + l * o_st[3];
                        written[e] = true;
                        assert_eq!(c.read(&got, e), want[l], "{c:?} [{i},{j},{k},{l}]");
                    }
                }
            }
        }
        let poison = c.read(&[POISON; 8], 0);
        for (e, &w) in written.iter().enumerate() {
            if !w {
                assert_eq!(
                    c.read(&got, e),
                    poison,
                    "{c:?}: element {e} was not in the view"
                );
            }
        }
        assert_eq!(written.iter().filter(|&&w| w).count(), n);
        // in place: out is x's own view, rows of 150 (chunks of 3 elements, so
        // two invocations, and at 150 two rows, share a 1-byte C's words)
        let own: Vec<i64> = (0..2 * 150)
            .map(|i| base[(i * 5 + 1) % base.len()])
            .collect();
        let ib = dev.upload(&c.encode(&own), 0);
        let mut r = flat(IntOp::Cumsum, c, c, c, c, 0);
        r.ndim = 2;
        r.shape = [2, 150, 0, 0];
        r.x_strides = [150, 1, 0, 0];
        r.out_strides = [150, 1, 0, 0];
        assert_eq!(dev.intwise(ib, ib, ib, &r), ffi::ACK_OK, "{}", last_error());
        let got = dev.download(ib, 300 * c.bytes());
        for row in 0..2 {
            let want = scan_reference(c, &own[row * 150..(row + 1) * 150]);
            for k in 0..150 {
                assert_eq!(
                    c.read(&got, row * 150 + k),
                    want[k],
                    "{c:?} in place [{row},{k}]"
                );
            }
        }
    }
    // more rows than one grid axis holds (the second axis carries the rest)
    let rows = 70_000usize;
    check_scan(&mut dev, (T::I32, T::I32, T::I32), rows, 2, 0xB16);
    check_scan(&mut dev, (T::U8, T::I64, T::I64), rows, 3, 0xB17);
}

// ---- the triangles: tril and triu (schema 4) ---------------------------------------

/// Whether (i, j) is on the kept side of `diag`: j - i <= diag for tril, >= for triu.
fn kept(op: IntOp, i: usize, j: usize, diag: i64) -> bool {
    let d = j as i64 - i as i64;
    if op == IntOp::Tril {
        d <= diag
    } else {
        d >= diag
    }
}

#[test]
#[allow(clippy::needless_range_loop)]
fn tril_and_triu_keep_x_on_their_side_of_the_diagonal() {
    let Some(mut dev) = open_or_unmeasured("intwise triangles") else {
        return;
    };
    // a [2, 5, 7] block (batch, rows, columns) of every code, diagonals inside,
    // at and past both edges, and far outside every C (the diagonal is an
    // int64 whatever C is), into a poisoned contiguous output
    let (batch, rows, cols) = (2usize, 5usize, 7usize);
    let n = batch * rows * cols;
    for c in TYPES {
        let base = c.inputs(80, 0x7121);
        let xs: Vec<i64> = (0..n).map(|i| base[(i * 3 + 1) % base.len()]).collect();
        let xb = dev.upload(&c.encode(&xs), 0);
        for op in [IntOp::Tril, IntOp::Triu] {
            for diag in [
                0i64,
                1,
                -1,
                3,
                -4,
                6,
                -5,
                7,
                -8,
                1 << 40,
                -(1 << 40),
                i64::MIN,
                i64::MAX,
            ] {
                let ob = dev.poisoned(n * c.bytes() + 3);
                let mut r = flat(op, c, c, c, c, 0);
                r.ndim = 3;
                r.shape = [batch as u64, rows as u64, cols as u64, 0];
                r.x_strides = [(rows * cols) as u64, cols as u64, 1, 0];
                r.out_strides = r.x_strides;
                r.scalar = diag;
                assert_eq!(
                    dev.intwise(xb, ob, ob, &r),
                    ffi::ACK_OK,
                    "{op:?} {c:?} {diag}: {}",
                    last_error()
                );
                let got = dev.download(ob, n * c.bytes() + 3);
                for b in 0..batch {
                    for i in 0..rows {
                        for j in 0..cols {
                            let e = (b * rows + i) * cols + j;
                            let want = if kept(op, i, j, diag) { xs[e] } else { 0 };
                            assert_eq!(c.read(&got, e), want, "{op:?} {c:?} {diag} [{b},{i},{j}]");
                        }
                    }
                }
                assert!(
                    got[n * c.bytes()..].iter().all(|&byte| byte == POISON),
                    "{op:?} {c:?} {diag}: the bytes past the last element changed"
                );
            }
        }
    }
    // in place through the output's own view (a dense u8 block: whole words),
    // and a transposed i16 view into a strided output (atomics)
    let own: Vec<i64> = (0..35).map(|i| 1 + i as i64).collect();
    let ib = dev.upload(&T::U8.encode(&own), 0);
    let mut r = flat(IntOp::Tril, T::U8, T::U8, T::U8, T::U8, 0);
    r.ndim = 2;
    r.shape = [5, 7, 0, 0];
    r.x_strides = [7, 1, 0, 0];
    r.out_strides = [7, 1, 0, 0];
    r.scalar = 1;
    assert_eq!(dev.intwise(ib, ib, ib, &r), ffi::ACK_OK, "{}", last_error());
    let got = dev.download(ib, 35);
    for i in 0..5 {
        for j in 0..7 {
            let want = if kept(IntOp::Tril, i, j, 1) {
                own[i * 7 + j]
            } else {
                0
            };
            assert_eq!(T::U8.read(&got, i * 7 + j), want, "in place [{i},{j}]");
        }
    }
    let src: Vec<i64> = (0..24).map(|i| i as i64 * 37 - 400).collect();
    let sb = dev.upload(&T::I16.encode(&src), 0);
    let ob = dev.poisoned(2 * 49);
    let mut r = flat(IntOp::Triu, T::I16, T::I16, T::I16, T::I16, 0);
    r.ndim = 2;
    r.shape = [4, 6, 0, 0];
    r.x_strides = [1, 4, 0, 0]; // the [4, 6] transpose of a [6, 4] block
    r.out_offset = 1;
    r.out_strides = [12, 2, 0, 0]; // every other element of a [4, 12] block, one in
    assert_eq!(dev.intwise(sb, ob, ob, &r), ffi::ACK_OK, "{}", last_error());
    let got = dev.download(ob, 2 * 49);
    let mut written = [false; 49];
    for i in 0..4 {
        for j in 0..6 {
            let e = 1 + 12 * i + 2 * j;
            written[e] = true;
            let want = if kept(IntOp::Triu, i, j, 0) {
                src[i + 4 * j]
            } else {
                0
            };
            assert_eq!(T::I16.read(&got, e), want, "strided triu [{i},{j}]");
        }
    }
    let poison = T::I16.read(&[POISON; 2], 0);
    for (e, &w) in written.iter().enumerate() {
        if !w {
            assert_eq!(
                T::I16.read(&got, e),
                poison,
                "element {e} was not in the view"
            );
        }
    }
    // an f32 block sent as its i32 bits: a NaN's payload and a -0.0 kept bit
    // for bit, the zeroed side written as +0.0
    let bits: Vec<i64> = [
        0x7FC0_0123u32,
        0x8000_0000,
        0x3F80_0000,
        0xFF80_0001,
        0x4049_0FDB,
        0xC000_0000,
    ]
    .iter()
    .map(|&b| b as i32 as i64)
    .collect();
    let fb = dev.upload(&T::I32.encode(&bits), 0);
    let ob = dev.poisoned(4 * 6);
    let mut r = flat(IntOp::Triu, T::I32, T::I32, T::I32, T::I32, 0);
    r.ndim = 2;
    r.shape = [2, 3, 0, 0];
    r.x_strides = [3, 1, 0, 0];
    r.out_strides = [3, 1, 0, 0];
    assert_eq!(dev.intwise(fb, ob, ob, &r), ffi::ACK_OK, "{}", last_error());
    let got = dev.download(ob, 4 * 6);
    for i in 0..2 {
        for j in 0..3 {
            let want = if j >= i { bits[i * 3 + j] } else { 0 };
            assert_eq!(T::I32.read(&got, i * 3 + j), want, "f32 bits [{i},{j}]");
        }
    }
}

// ---- the ranks (schema 5) ----------------------------------------------------------

/// An element as torch's sort compares it: an integer by value, a float with
/// every NaN equal and above every number and -0.0 equal to +0.0.
#[derive(Clone, Copy, Debug)]
enum Key {
    Int(i64),
    Float(f64),
}

fn key_cmp(a: Key, b: Key) -> std::cmp::Ordering {
    use std::cmp::Ordering::{Equal, Greater, Less};
    match (a, b) {
        (Key::Int(x), Key::Int(y)) => x.cmp(&y),
        (Key::Float(x), Key::Float(y)) => match (x.is_nan(), y.is_nan()) {
            (true, true) => Equal,
            (true, false) => Greater,
            (false, true) => Less,
            _ => x.partial_cmp(&y).expect("no NaN left"), // -0.0 == +0.0
        },
        _ => unreachable!("one code a row"),
    }
}

/// A half's value, exactly (a subnormal, an infinity and a NaN included).
fn half_value(h: u16) -> f64 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = i32::from((h >> 10) & 0x1F);
    let mant = f64::from(h & 0x3FF);
    sign * match exp {
        0 => mant * 2f64.powi(-24),
        31 if mant == 0.0 => f64::INFINITY,
        31 => f64::NAN,
        _ => (1.0 + mant / 1024.0) * 2f64.powi(exp - 15),
    }
}

/// Each element's position in a STABLE sort of its row (Rust's sort_by is
/// stable), rows of `len` laid out one after another.
fn ref_ranks(keys: &[Key], len: usize, descending: bool) -> Vec<i64> {
    let mut ranks = vec![0i64; keys.len()];
    for (r, row) in keys.chunks(len).enumerate() {
        let mut order: Vec<usize> = (0..row.len()).collect();
        if descending {
            order.sort_by(|&a, &b| key_cmp(row[b], row[a]));
        } else {
            order.sort_by(|&a, &b| key_cmp(row[a], row[b]));
        }
        for (position, &i) in order.iter().enumerate() {
            ranks[r * len + i] = position as i64;
        }
    }
    ranks
}

/// One rank request over a contiguous [rows, len] block of `code` elements
/// (`bytes` their little-endian storage), into a poisoned i64 output.
fn run_ranks(
    dev: &mut Device,
    code: Code,
    bytes: &[u8],
    rows: usize,
    len: usize,
    descending: bool,
) -> Vec<i64> {
    let xb = dev.upload(bytes, 0);
    let ob = dev.poisoned(rows * len * 8 + 3);
    let op = if descending {
        IntOp::RankDesc
    } else {
        IntOp::RankAsc
    };
    let mut r = flat(op, T::I64, T::I64, T::I64, T::I64, 0);
    r.x_code = code as u32;
    r.ndim = 2;
    r.shape = [rows as u64, len as u64, 0, 0];
    r.x_strides = [len as u64, 1, 0, 0];
    r.out_strides = [len as u64, 1, 0, 0];
    assert_eq!(
        dev.intwise(xb, ob, ob, &r),
        ffi::ACK_OK,
        "{op:?} {code:?}: {}",
        last_error()
    );
    let got = dev.download(ob, rows * len * 8 + 3);
    assert!(got[rows * len * 8..].iter().all(|&b| b == POISON));
    (0..rows * len).map(|i| T::I64.read(&got, i)).collect()
}

#[test]
fn ranks_follow_torchs_stable_sort_for_every_storage_code() {
    let Some(mut dev) = open_or_unmeasured("intwise ranks") else {
        return;
    };
    // floats: every special class, with ties between numbers, between +0.0
    // and -0.0 and between NaNs of different payloads and signs
    let f32s: Vec<u32> = vec![
        0x0000_0000,
        0x8000_0000,
        0x3F80_0000,
        0xBF80_0000,
        0x4060_0000,
        0x7FC0_0000,
        0xFFC0_0001,
        0x7F80_0000,
        0xFF80_0000,
        0x3F80_0000,
        0x4060_0000,
        0x8000_0000,
        0x0000_0001,
        0x8000_0001,
        0x0080_0000,
        0x7F7F_FFFF,
        0xFF7F_FFFF,
        0x7FA0_0123,
        0x40E0_0000,
        0x40E0_0000,
        0xC0E0_0000,
    ];
    let f16s: Vec<u16> = vec![
        0x0000, 0x8000, 0x3C00, 0xBC00, 0x7E00, 0xFE01, 0x7C00, 0xFC00, 0x0001, 0x8001, 0x3C00,
        0x7BFF, 0xFBFF, 0x4700, 0x4700,
    ];
    let bf16s: Vec<u16> = vec![
        0x0000, 0x8000, 0x3F80, 0xBF80, 0x7FC0, 0xFFC1, 0x7F80, 0xFF80, 0x0001, 0x3F80, 0x40E0,
        0x40E0, 0xC0E0,
    ];
    let f64s: Vec<u64> = vec![
        0x0000_0000_0000_0000,
        0x8000_0000_0000_0000,
        0x3FF0_0000_0000_0000,
        0xBFF0_0000_0000_0000,
        0x7FF8_0000_0000_0000,
        0xFFF8_0000_0000_0001,
        0x7FF0_0000_0000_0000,
        0xFFF0_0000_0000_0000,
        0x0000_0000_0000_0001,
        0x3FF0_0000_0000_0000,
        0x7FEF_FFFF_FFFF_FFFF,
        0x3FF0_0000_0000_0001,
    ];
    // a [3, len] block of each: the list in three rotations
    let rotate = |len: usize, r: usize, i: usize| (i * 7 + r * 5) % len;
    for descending in [false, true] {
        let len = f32s.len();
        let (mut keys, mut bytes) = (vec![], vec![]);
        for r in 0..3 {
            for i in 0..len {
                let b = f32s[rotate(len, r, i)];
                keys.push(Key::Float(f64::from(f32::from_bits(b))));
                bytes.extend_from_slice(&b.to_le_bytes());
            }
        }
        let got = run_ranks(&mut dev, Code::F32, &bytes, 3, len, descending);
        assert_eq!(
            got,
            ref_ranks(&keys, len, descending),
            "f32 descending={descending}"
        );
        for (code, list) in [(Code::F16, &f16s), (Code::Bf16, &bf16s)] {
            let len = list.len();
            let (mut keys, mut bytes) = (vec![], vec![]);
            for r in 0..3 {
                for i in 0..len {
                    let h = list[rotate(len, r, i)];
                    let v = if code == Code::F16 {
                        half_value(h)
                    } else {
                        f64::from(f32::from_bits(u32::from(h) << 16))
                    };
                    keys.push(Key::Float(v));
                    bytes.extend_from_slice(&h.to_le_bytes());
                }
            }
            let got = run_ranks(&mut dev, code, &bytes, 3, len, descending);
            assert_eq!(
                got,
                ref_ranks(&keys, len, descending),
                "{code:?} descending={descending}"
            );
        }
        let len = f64s.len();
        let (mut keys, mut bytes) = (vec![], vec![]);
        for r in 0..3 {
            for i in 0..len {
                let b = f64s[rotate(len, r, i)];
                keys.push(Key::Float(f64::from_bits(b)));
                bytes.extend_from_slice(&b.to_le_bytes());
            }
        }
        let got = run_ranks(&mut dev, Code::F64, &bytes, 3, len, descending);
        assert_eq!(
            got,
            ref_ranks(&keys, len, descending),
            "f64 descending={descending}"
        );
        // every integer code, its boundary values repeated for ties, rows of 70
        for t in TYPES {
            let base = t.inputs(40, 0x5EED_0005);
            let len = 70usize;
            let values: Vec<i64> = (0..2 * len).map(|i| base[(i * 3) % base.len()]).collect();
            let keys: Vec<Key> = values.iter().map(|&v| Key::Int(t.cast(v))).collect();
            let got = run_ranks(&mut dev, t.code(), &t.encode(&values), 2, len, descending);
            assert_eq!(
                got,
                ref_ranks(&keys, len, descending),
                "{t:?} descending={descending}"
            );
        }
    }
    // a strided row: the columns of a [5, 9] block, ranked along the rows' axis
    let values: Vec<i64> = (0..45).map(|i| (i * 37 % 11) as i64 - 5).collect();
    let xb = dev.upload(&T::I32.encode(&values), 0);
    let ob = dev.poisoned(45 * 8);
    let mut r = flat(IntOp::RankAsc, T::I32, T::I64, T::I64, T::I64, 0);
    r.ndim = 2;
    r.shape = [9, 5, 0, 0]; // row c of the request is column c of the block
    r.x_strides = [1, 9, 0, 0];
    r.out_strides = [5, 1, 0, 0];
    assert_eq!(dev.intwise(xb, ob, ob, &r), ffi::ACK_OK, "{}", last_error());
    let got = dev.download(ob, 45 * 8);
    let keys: Vec<Key> = (0..9)
        .flat_map(|c| (0..5).map(move |i| (c, i)))
        .map(|(c, i)| Key::Int(values[i * 9 + c]))
        .collect();
    let want = ref_ranks(&keys, 5, false);
    for (e, &w) in want.iter().enumerate() {
        assert_eq!(T::I64.read(&got, e), w, "strided row element {e}");
    }
    // refused: in place (the rank reads its whole row), a float output, and a
    // row past the walk's limit with an int16 output (schema 6: an int64 or
    // int32 output that long is the radix module's, see below)
    let ib = dev.upload(&T::I64.encode(&[3, 1, 2, 1]), 0);
    let before = dev.download(ib, 32);
    for (req, x) in [
        (flat(IntOp::RankAsc, T::I64, T::I64, T::I64, T::I64, 4), ib),
        (
            ffi::AckIntwiseRequest {
                out_code: Code::F32 as u32,
                ..flat(IntOp::RankAsc, T::I64, T::I64, T::I64, T::I64, 4)
            },
            xb,
        ),
        (
            flat(
                IntOp::RankDesc,
                T::I64,
                T::I64,
                T::I16,
                T::I16,
                (1 << 16) + 1,
            ),
            xb,
        ),
    ] {
        assert_eq!(dev.intwise(x, ib, ib, &req), ffi::ACK_ERR_SHAPE, "{req:?}");
    }
    assert_eq!(dev.download(ib, 32), before, "a refusal wrote");
}

// ---- long rows: the radix module (schema 6) ---------------------------------------

/// A deterministic stream of 64-bit values (xorshift64).
fn stream(seed: u64) -> impl FnMut() -> u64 {
    let mut state = seed | 1;
    move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    }
}

/// One rank request over a [rows, len] block of `code` elements laid out with
/// strides `x_strides` (in elements), into a poisoned output of type `out`
/// (also C) laid out with `out_strides` over `out_elems` elements; returns
/// every element's rank in the request's row-major order.
fn run_long_ranks(
    dev: &mut Device,
    code: Code,
    bytes: &[u8],
    (rows, len): (usize, usize),
    x_strides: [u64; 2],
    (out, out_strides, out_elems): (T, [u64; 2], usize),
    descending: bool,
) -> Vec<i64> {
    let xb = dev.upload(bytes, 0);
    let ob = dev.poisoned(out_elems * out.bytes() + 3);
    let op = if descending {
        IntOp::RankDesc
    } else {
        IntOp::RankAsc
    };
    let mut r = flat(op, out, out, out, out, 0);
    r.x_code = code as u32;
    r.ndim = 2;
    r.shape = [rows as u64, len as u64, 0, 0];
    r.x_strides = [x_strides[0], x_strides[1], 0, 0];
    r.out_strides = [out_strides[0], out_strides[1], 0, 0];
    assert_eq!(
        dev.intwise(xb, ob, ob, &r),
        ffi::ACK_OK,
        "{op:?} {code:?} [{rows}, {len}]: {}",
        last_error()
    );
    let got = dev.download(ob, out_elems * out.bytes() + 3);
    assert!(
        got[out_elems * out.bytes()..out_elems * out.bytes() + 3]
            .iter()
            .all(|&b| b == POISON),
        "the bytes past the output"
    );
    let mut ranks = Vec::with_capacity(rows * len);
    for i in 0..rows {
        for j in 0..len {
            let e = i as u64 * out_strides[0] + j as u64 * out_strides[1];
            ranks.push(out.read(&got, e as usize));
        }
    }
    ranks
}

/// Schema 6: a rank row longer than 1,024 with an int64 or int32 output is the
/// radix module's. Every storage code, both directions, ties everywhere (the
/// MoE routing's 65,536 expert ids in [0, 128) first), the special float
/// classes, several rows (the row index's digit passes), strided views, and a
/// row past the walk's own limit, each against a stable host sort.
#[test]
fn a_long_rows_ranks_are_torchs_stable_sort_through_the_radix_module() {
    let Some(mut dev) = open_or_unmeasured("intwise radix ranks") else {
        return;
    };
    let mut next = stream(0x5EED_0006);
    // the MoE routing's argsort: int64 ids in [0, 128), both output widths
    let n = 65_536usize;
    let ids: Vec<i64> = (0..n).map(|_| (next() % 128) as i64).collect();
    let keys: Vec<Key> = ids.iter().map(|&v| Key::Int(v)).collect();
    for descending in [false, true] {
        let want = ref_ranks(&keys, n, descending);
        for out in [T::I64, T::I32] {
            let got = run_long_ranks(
                &mut dev,
                Code::I64,
                &T::I64.encode(&ids),
                (1, n),
                [n as u64, 1],
                (out, [n as u64, 1], n),
                descending,
            );
            assert_eq!(got, want, "ids descending={descending} out={out:?}");
        }
    }
    // every integer code: boundary values and a stream, 3 rows of 3,000
    let (rows, len) = (3usize, 3_000usize);
    for t in TYPES {
        let base = t.inputs(64, 0x5EED_0007);
        let values: Vec<i64> = (0..rows * len)
            .map(|i| {
                if i % 3 == 0 {
                    base[i % base.len()]
                } else {
                    t.cast(next() as i64)
                }
            })
            .collect();
        let keys: Vec<Key> = values.iter().map(|&v| Key::Int(t.cast(v))).collect();
        for descending in [false, true] {
            let got = run_long_ranks(
                &mut dev,
                t.code(),
                &t.encode(&values),
                (rows, len),
                [len as u64, 1],
                (T::I64, [len as u64, 1], rows * len),
                descending,
            );
            assert_eq!(
                got,
                ref_ranks(&keys, len, descending),
                "{t:?} descending={descending}"
            );
        }
    }
    // every float code: the special classes (both zeros, NaNs of either sign
    // and any payload, infinities, subnormals) mixed into a stream of bits
    let f32_specials: [u32; 12] = [
        0x0000_0000,
        0x8000_0000,
        0x7FC0_0000,
        0xFFC0_0001,
        0x7F80_0000,
        0xFF80_0000,
        0x0000_0001,
        0x8000_0001,
        0x7FA0_0123,
        0x3F80_0000,
        0xBF80_0000,
        0x3F80_0000,
    ];
    let half_specials: [u16; 10] = [
        0x0000, 0x8000, 0x7E00, 0xFE01, 0x7C00, 0xFC00, 0x0001, 0x8001, 0x3C00, 0x3C00,
    ];
    let f64_specials: [u64; 9] = [
        0x0000_0000_0000_0000,
        0x8000_0000_0000_0000,
        0x7FF8_0000_0000_0000,
        0xFFF8_0000_0000_0001,
        0x7FF0_0000_0000_0000,
        0xFFF0_0000_0000_0000,
        0x0000_0000_0000_0001,
        0x3FF0_0000_0000_0000,
        0x3FF0_0000_0000_0000,
    ];
    for descending in [false, true] {
        // f32: 64 magnitudes of either sign, so ties are common
        let bits: Vec<u32> = (0..rows * len)
            .map(|i| {
                if i % 4 == 0 {
                    f32_specials[i % f32_specials.len()]
                } else {
                    let v = next();
                    (v as u32 & 0x8000_0000) | (0x3F80_0000 + ((v >> 32) as u32 % 64) * 0x0010_0000)
                }
            })
            .collect();
        let keys: Vec<Key> = bits
            .iter()
            .map(|&b| Key::Float(f64::from(f32::from_bits(b))))
            .collect();
        let bytes: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
        let got = run_long_ranks(
            &mut dev,
            Code::F32,
            &bytes,
            (rows, len),
            [len as u64, 1],
            (T::I64, [len as u64, 1], rows * len),
            descending,
        );
        assert_eq!(
            got,
            ref_ranks(&keys, len, descending),
            "f32 descending={descending}"
        );
        for code in [Code::F16, Code::Bf16] {
            let halves: Vec<u16> = (0..rows * len)
                .map(|i| {
                    if i % 4 == 0 {
                        half_specials[i % half_specials.len()]
                    } else {
                        (next() >> 48) as u16
                    }
                })
                .collect();
            let keys: Vec<Key> = halves
                .iter()
                .map(|&h| {
                    Key::Float(if code == Code::F16 {
                        half_value(h)
                    } else {
                        f64::from(f32::from_bits(u32::from(h) << 16))
                    })
                })
                .collect();
            let bytes: Vec<u8> = halves.iter().flat_map(|h| h.to_le_bytes()).collect();
            let got = run_long_ranks(
                &mut dev,
                code,
                &bytes,
                (rows, len),
                [len as u64, 1],
                (T::I32, [len as u64, 1], rows * len),
                descending,
            );
            assert_eq!(
                got,
                ref_ranks(&keys, len, descending),
                "{code:?} descending={descending}"
            );
        }
        let doubles: Vec<u64> = (0..rows * len)
            .map(|i| {
                if i % 4 == 0 {
                    f64_specials[i % f64_specials.len()]
                } else {
                    let v = next();
                    (v & 0x8000_0000_0000_0000)
                        | (0x3FF0_0000_0000_0000 + (v % 32) * 0x0008_0000_0000_0000)
                }
            })
            .collect();
        let keys: Vec<Key> = doubles
            .iter()
            .map(|&b| Key::Float(f64::from_bits(b)))
            .collect();
        let bytes: Vec<u8> = doubles.iter().flat_map(|b| b.to_le_bytes()).collect();
        let got = run_long_ranks(
            &mut dev,
            Code::F64,
            &bytes,
            (rows, len),
            [len as u64, 1],
            (T::I64, [len as u64, 1], rows * len),
            descending,
        );
        assert_eq!(
            got,
            ref_ranks(&keys, len, descending),
            "f64 descending={descending}"
        );
    }
    // 300 rows (two digit passes of the row index), and strided views: the
    // request's rows are the columns of a [1100, 300] block, the output too
    let (rows, len) = (300usize, 1_100usize);
    let values: Vec<i64> = (0..rows * len).map(|_| (next() % 50) as i64 - 25).collect();
    let block: Vec<i64> = (0..len)
        .flat_map(|i| (0..rows).map(move |c| (i, c)))
        .map(|(i, c)| values[c * len + i])
        .collect();
    let keys: Vec<Key> = values.iter().map(|&v| Key::Int(v)).collect();
    let got = run_long_ranks(
        &mut dev,
        Code::I32,
        &T::I32.encode(&block),
        (rows, len),
        [1, rows as u64],
        (T::I64, [1, rows as u64], rows * len),
        true,
    );
    assert_eq!(got, ref_ranks(&keys, len, true), "strided, 300 rows");
    // past the walk's limit: one row of 70,000 f32 (schema 5 refused it)
    let n = 70_000usize;
    let bits: Vec<u32> = (0..n)
        .map(|_| 0x4000_0000 + (next() as u32 % 1000) * 0x100)
        .collect();
    let keys: Vec<Key> = bits
        .iter()
        .map(|&b| Key::Float(f64::from(f32::from_bits(b))))
        .collect();
    let bytes: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
    let got = run_long_ranks(
        &mut dev,
        Code::F32,
        &bytes,
        (1, n),
        [n as u64, 1],
        (T::I64, [n as u64, 1], n),
        false,
    );
    assert_eq!(got, ref_ranks(&keys, n, false), "a row of 70,000");
}

#[test]
fn a_refused_request_writes_nothing() {
    let Some(mut dev) = open_or_unmeasured("intwise refusals") else {
        return;
    };
    let xb = dev.upload(&T::I32.encode(&[1, 2, 3, 4]), 0);
    let ob = dev.poisoned(16);
    for (r, code) in [
        // sub on bool: torch refuses it
        (
            flat(IntOp::Sub, T::Bool, T::Bool, T::Bool, T::Bool, 4),
            ffi::ACK_ERR_SHAPE,
        ),
        // a view past the input
        (
            flat(IntOp::Add, T::I64, T::I64, T::I64, T::I64, 4),
            ffi::ACK_ERR_SIZE,
        ),
        // the output overlapping x through another view
        (
            ffi::AckIntwiseRequest {
                x_offset: 1,
                ..flat(IntOp::Neg, T::I32, T::I32, T::I32, T::I32, 3)
            },
            ffi::ACK_ERR_SHAPE,
        ),
        // a division by a tensor, by zero, and by -1 in i32
        (
            flat(IntOp::DivTrunc, T::I32, T::I32, T::I32, T::I32, 4),
            ffi::ACK_ERR_SHAPE,
        ),
        (
            ffi::AckIntwiseRequest {
                flags: 1,
                scalar: 0,
                ..flat(IntOp::DivFloor, T::I32, T::I32, T::I32, T::I32, 4)
            },
            ffi::ACK_ERR_SHAPE,
        ),
        (
            ffi::AckIntwiseRequest {
                flags: 1,
                scalar: -1,
                ..flat(IntOp::Remainder, T::I32, T::I32, T::I32, T::I32, 4)
            },
            ffi::ACK_ERR_SHAPE,
        ),
        // cumsum on a bool C, which torch refuses, and a row past the scan's
        // limit, refused before any capacity is read
        (
            flat(IntOp::Cumsum, T::Bool, T::Bool, T::Bool, T::Bool, 4),
            ffi::ACK_ERR_SHAPE,
        ),
        (
            flat(IntOp::Cumsum, T::I32, T::I32, T::I32, T::I32, (1 << 24) + 1),
            ffi::ACK_ERR_SHAPE,
        ),
    ] {
        let x = if r.x_offset == 1 { ob } else { xb };
        assert_eq!(dev.intwise(x, xb, ob, &r), code, "{r:?}");
    }
    assert!(
        dev.download(ob, 16).iter().all(|&b| b == POISON),
        "a refusal wrote"
    );
}
