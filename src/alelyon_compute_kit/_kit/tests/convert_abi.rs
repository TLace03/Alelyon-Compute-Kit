//! `ack_convert` (ABI 13): every storage pair the family converts, on a real
//! device, through the C ABI, bit for bit against host references.
//!
//! THE REFERENCES SHARE NO CODE WITH THE SHADER'S INTEGER ROUNDING. f32 ->
//! f16 finds the nearest value by searching every finite fp16 (ties to the
//! even significand, 65520 and above to infinity); f32 -> bf16 compares the
//! two neighbouring bf16 values' distances exactly in f64; f64 -> f32, f32 ->
//! f64 and int -> float are Rust's `as`, which rounds to nearest even; int ->
//! int is `as` truncation. What c10 adds on top, the references state
//! explicitly: a float -> float conversion goes through f32 (c10's Half and
//! BFloat16 are built from float).
//!
//! A NaN IS HELD TO NaN AND ITS SIGN, NOT TO ITS PAYLOAD BITS, because torch's
//! own payloads are not one function of the input: c10's scalar f32 -> f16
//! writes sign | 0x7E00, while the vectorised copy on an x86 host with F16C
//! (`_mm256_cvtps_ph`) keeps the top payload bits, so which one an element
//! gets depends on its position in the tensor; and F16C's f16 -> f32 quiets a
//! signalling NaN where a bf16 -> f32 shift does not; c10's scalar f32 ->
//! bf16 writes 0x7FC0 and its AVX2 path 0xFFFF, whatever the sign. The kernel
//! writes bf16 as cast_f32_bf16.comp does ((bits >> 16) | 0x40), f16 as
//! c10's scalar sign | 0x7E00, and x86's cvt forms for f64 <-> f32 (quiet bit
//! set, top payload bits kept); the test asserts a NaN of the same sign.
//!
//! The inputs are the special values of every type (zeros, infinities, NaNs
//! with payloads, subnormals, the rounding ties and the overflow boundaries of
//! each target) and a fixed pseudo-random set of bit patterns. Outputs are
//! poisoned first, so an element the kernel never wrote cannot pass, and the
//! bytes past a partly filled last output word must come back as they were.
//!
//! Without a device every test prints UNMEASURED (`ACK_REQUIRE_DEVICE=1`
//! turns that into a failure).
use alelyon_compute_kit::convert_ops::Code;
use alelyon_compute_kit::ffi;
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

    fn download(&self, buf: Buf, len: usize) -> Vec<u8> {
        let mut out = vec![0u8; len];
        let rc = unsafe { ffi::ack_download(self.dev, buf, out.as_mut_ptr(), len) };
        assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
        out
    }

    fn convert(&self, src: Buf, out: Buf, r: &ffi::AckConvertRequest) -> i32 {
        let mut ms = 0.0f64;
        unsafe {
            ffi::ack_convert(
                self.dev,
                src,
                out,
                r,
                std::mem::size_of::<ffi::AckConvertRequest>(),
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

/// A contiguous 1-D conversion of `n` elements.
fn flat(from: Code, to: Code, n: u64) -> ffi::AckConvertRequest {
    ffi::AckConvertRequest {
        src_code: from as u32,
        out_code: to as u32,
        ndim: 1,
        flags: 0,
        shape: [n, 0, 0, 0],
        in_offset: 0,
        in_strides: [1, 0, 0, 0],
        out_offset: 0,
        out_strides: [1, 0, 0, 0],
    }
}

// ---- values ------------------------------------------------------------------

/// A value on the host: the float types as their bits widened to u64, the
/// integer types sign- or zero-extended, bool as 0/1.
#[derive(Clone, Copy, Debug)]
enum V {
    F32(u32),
    Bf16(u16),
    F16(u16),
    F64(u64),
    Int(i64),
}

fn decode(code: Code, raw: u64) -> V {
    match code {
        Code::F32 => V::F32(raw as u32),
        Code::Bf16 => V::Bf16(raw as u16),
        Code::F16 => V::F16(raw as u16),
        Code::F64 => V::F64(raw),
        Code::I64 => V::Int(raw as i64),
        Code::I32 => V::Int(raw as u32 as i32 as i64),
        Code::I16 => V::Int(raw as u16 as i16 as i64),
        Code::I8 => V::Int(raw as u8 as i8 as i64),
        Code::U8 => V::Int(raw as u8 as i64),
        Code::Bool => V::Int(i64::from(raw as u8 != 0)),
    }
}

fn f16_value(h: u16) -> f64 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = (h >> 10) & 0x1f;
    let m = f64::from(h & 0x3ff);
    sign * if e == 0 {
        m * 2f64.powi(-24)
    } else {
        (1.0 + m / 1024.0) * 2f64.powi(i32::from(e) - 15)
    }
}

/// Every finite non-negative fp16, ascending (bits ascend with value).
fn f16_table() -> Vec<(f64, u16)> {
    (0u16..0x7c00).map(|h| (f16_value(h), h)).collect()
}

/// f32 -> f16, nearest with ties to the even significand, by search.
fn ref_f32_to_f16(bits: u32, table: &[(f64, u16)]) -> u16 {
    let x = f32::from_bits(bits);
    let sign = ((bits >> 16) & 0x8000) as u16;
    if x.is_nan() {
        return sign | 0x7e00;
    }
    let a = f64::from(x.abs());
    // 65520 is halfway from 65504 to 65536, which is even: infinity
    if a >= 65520.0 {
        return sign | 0x7c00;
    }
    let i = table.partition_point(|&(v, _)| v < a);
    if i < table.len() && table[i].0 == a {
        return sign | table[i].1;
    }
    // past 65504 the upper neighbour is 65536, whose bits are infinity's
    let hi = table.get(i).copied().unwrap_or((65536.0, 0x7c00));
    let lo = table[i - 1];
    let (dl, dh) = (a - lo.0, hi.0 - a);
    let pick = if dl < dh || (dl == dh && lo.1 & 1 == 0) {
        lo.1
    } else {
        hi.1
    };
    sign | pick
}

/// f32 -> bf16, nearest with ties to even, comparing the neighbours exactly.
fn ref_f32_to_bf16(bits: u32) -> u16 {
    let x = f32::from_bits(bits);
    if x.is_nan() {
        return ((bits >> 16) | 0x40) as u16;
    }
    let lo = bits & 0xffff_0000;
    if lo == bits || x.is_infinite() {
        return (lo >> 16) as u16;
    }
    let hi = lo + 0x1_0000; // one bf16 ulp further from zero
    let a = f64::from(f32::from_bits(bits & 0x7fff_ffff));
    let vl = f64::from(f32::from_bits(lo & 0x7fff_ffff));
    // above the largest finite bf16 the next value is 2^128 in the arithmetic
    // (the exponent field's carry), which rounds to infinity
    let vh = if hi & 0x7fff_ffff == 0x7f80_0000 {
        2f64.powi(128)
    } else {
        f64::from(f32::from_bits(hi & 0x7fff_ffff))
    };
    let (dl, dh) = (a - vl, vh - a);
    let pick = if dl < dh || (dl == dh && (lo >> 16) & 1 == 0) {
        lo
    } else {
        hi
    };
    (pick >> 16) as u16
}

fn nan_f64_to_f32(d: u64) -> u32 {
    ((d >> 32) as u32 & 0x8000_0000) | 0x7fc0_0000 | ((d & 0x000f_ffff_ffff_ffff) >> 29) as u32
}

fn nan_f32_to_f64(w: u32) -> u64 {
    (u64::from(w & 0x8000_0000) << 32) | 0x7ff8_0000_0000_0000 | (u64::from(w & 0x007f_ffff) << 29)
}

/// The value as f32 bits, as c10 reaches float (through f32 for f16 and bf16).
fn to_f32_bits(v: V) -> u32 {
    match v {
        V::F32(w) => w,
        V::Bf16(h) => u32::from(h) << 16,
        V::F16(h) => {
            let x = f16_value(h);
            if h & 0x7c00 == 0x7c00 {
                let sign = u32::from(h & 0x8000) << 16;
                return sign | 0x7f80_0000 | (u32::from(h & 0x3ff) << 13);
            }
            let w = (x as f32).to_bits(); // exact: every fp16 is an f32
            if x == 0.0 && h & 0x8000 != 0 {
                0x8000_0000
            } else {
                w
            }
        }
        V::F64(d) => {
            let x = f64::from_bits(d);
            if x.is_nan() {
                nan_f64_to_f32(d)
            } else {
                (x as f32).to_bits()
            }
        }
        V::Int(i) => (i as f32).to_bits(),
    }
}

/// The reference output bits for `v` in storage `to` (never float -> integer).
fn reference(v: V, to: Code, table: &[(f64, u16)]) -> u64 {
    match to {
        Code::Bool => {
            let nonzero = match v {
                V::F32(w) => w & 0x7fff_ffff != 0,
                V::Bf16(h) | V::F16(h) => h & 0x7fff != 0,
                V::F64(d) => d & 0x7fff_ffff_ffff_ffff != 0,
                V::Int(i) => i != 0,
            };
            u64::from(nonzero)
        }
        Code::F64 => match v {
            V::F64(d) => d,
            V::Int(i) => (i as f64).to_bits(),
            other => {
                let w = to_f32_bits(other);
                let x = f32::from_bits(w);
                if x.is_nan() {
                    nan_f32_to_f64(w)
                } else {
                    f64::from(x).to_bits()
                }
            }
        },
        Code::F32 => u64::from(to_f32_bits(v)),
        Code::Bf16 => u64::from(ref_f32_to_bf16(to_f32_bits(v))),
        Code::F16 => u64::from(ref_f32_to_f16(to_f32_bits(v), table)),
        integer => {
            let V::Int(i) = v else {
                unreachable!("float -> integer is refused")
            };
            match integer {
                Code::I64 => i as u64,
                Code::I32 => u64::from(i as u32),
                Code::I16 => u64::from(i as u16),
                _ => u64::from(i as u8),
            }
        }
    }
}

/// The special values of `code`'s type, then `random` pseudo-random patterns.
fn inputs(code: Code, random: usize) -> Vec<u64> {
    let mut v: Vec<u64> = match code {
        Code::F32 => [
            0u32,
            0x8000_0000,
            0x7f80_0000,
            0xff80_0000,
            0x7fc0_0000,
            0xffc0_0001,
            0x7f80_0001,
            0x7fbf_ffff,
            1,
            0x8000_0001,
            0x007f_ffff,
            0x0080_0000,
            0x3f80_0000,
            0xbf80_0000,
            0x7f7f_ffff,
            0xff7f_ffff,
            // f16: 65504, 65519.996, 65520 (tie to inf), 65536, min subnormal,
            // half of it (tie to zero), just above half, a normal-range tie
            0x477f_e000,
            0x477f_efff,
            0x477f_f000,
            0x4780_0000,
            0x3380_0000,
            0x3300_0000,
            0x3300_0001,
            0x3f80_1000,
            0x3f80_3000,
            0xc000_0800,
            // bf16: exact ties (to even down and up), just above a tie, near max
            0x3f80_8000,
            0x3f81_8000,
            0x3f80_8001,
            0x7f7f_8000,
            0x7f7f_7fff,
        ]
        .iter()
        .map(|&w| u64::from(w))
        .collect(),
        Code::F64 => [
            0u64,
            0x8000_0000_0000_0000,
            0x7ff0_0000_0000_0000,
            0xfff0_0000_0000_0000,
            0x7ff8_0000_0000_0000,
            0x7ff0_0000_2000_0001,
            0xfff4_0000_0000_0000,
            1,
            0x000f_ffff_ffff_ffff,
            f64::MAX.to_bits(),
            f64::from(f32::MAX).to_bits(),
            // halfway from f32::MAX to 2^128: a tie, to infinity (even)
            (f64::from(f32::MAX) + 2f64.powi(103)).to_bits(),
            (f64::from(f32::MAX) + 2f64.powi(103)).to_bits() - 1,
            // the f32 subnormal range: 2^-149, 2^-150 (a tie, to zero), 3 * 2^-150
            // (a tie, to 2^-148), just above 2^-150, 2^-126 - 2^-150
            2f64.powi(-149).to_bits(),
            2f64.powi(-150).to_bits(),
            (3.0 * 2f64.powi(-150)).to_bits(),
            2f64.powi(-150).to_bits() + 1,
            (2f64.powi(-126) - 2f64.powi(-150)).to_bits(),
            1.0f64.to_bits() + 1,
            (1.0f64 + 2f64.powi(-24)).to_bits(),
            (1.0f64 + 3.0 * 2f64.powi(-24)).to_bits(),
            65519.99f64.to_bits(),
            (-0.1f64).to_bits(),
        ]
        .to_vec(),
        Code::F16 => [
            0u16, 0x8000, 0x7c00, 0xfc00, 0x7e00, 0xfd01, 0x7c01, 1, 0x8001, 0x03ff, 0x0400,
            0x3c00, 0x7bff, 0xfbff,
        ]
        .iter()
        .map(|&h| u64::from(h))
        .collect(),
        Code::Bf16 => [
            0u16, 0x8000, 0x7f80, 0xff80, 0x7fc0, 0xffc1, 0x7f81, 1, 0x8001, 0x007f, 0x0080,
            0x3f80, 0x7f7f,
        ]
        .iter()
        .map(|&h| u64::from(h))
        .collect(),
        Code::I64 => [
            0i64,
            1,
            -1,
            i64::MIN,
            i64::MAX,
            (1 << 24) + 1,
            (1 << 24) + 3,
            -((1 << 24) + 1),
            (1 << 53) + 1,
            (1 << 53) + 3,
            -((1 << 53) + 1),
            (1 << 62) + (1 << 38) + 1,
            i64::from(i32::MIN),
            i64::from(i32::MAX) + 1,
            65535,
            -32769,
            255,
            256,
            -129,
        ]
        .iter()
        .map(|&i| i as u64)
        .collect(),
        Code::I32 => [
            0i32,
            1,
            -1,
            i32::MIN,
            i32::MAX,
            (1 << 24) + 1,
            65536,
            -32769,
            255,
            256,
            -129,
        ]
        .iter()
        .map(|&i| u64::from(i as u32))
        .collect(),
        Code::I16 => [0i16, 1, -1, i16::MIN, i16::MAX, 255, 256, -129, 127]
            .iter()
            .map(|&i| u64::from(i as u16))
            .collect(),
        Code::I8 => [0i8, 1, -1, i8::MIN, i8::MAX]
            .iter()
            .map(|&i| u64::from(i as u8))
            .collect(),
        Code::U8 => vec![0, 1, 127, 128, 255],
        Code::Bool => vec![0, 1],
    };
    // a fixed LCG: the same patterns on every run and every device
    let mut state = 0x9e37_79b9_7f4a_7c15u64 ^ (code as u64);
    let bits = code.bytes() * 8;
    for _ in 0..random {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let r = state ^ (state >> 29);
        v.push(match code {
            Code::Bool => r & 1,
            _ if bits == 64 => r,
            _ => r & ((1u64 << bits) - 1),
        });
    }
    v
}

/// For a NaN of `code`'s float type, its sign bit; `None` for anything else.
fn nan_sign(code: Code, bits: u64) -> Option<bool> {
    match code {
        Code::F32 => f32::from_bits(bits as u32)
            .is_nan()
            .then_some(bits & 0x8000_0000 != 0),
        Code::F64 => f64::from_bits(bits).is_nan().then_some(bits >> 63 != 0),
        Code::Bf16 => ((bits & 0x7f80) == 0x7f80 && bits & 0x7f != 0).then_some(bits & 0x8000 != 0),
        Code::F16 => ((bits & 0x7c00) == 0x7c00 && bits & 0x3ff != 0).then_some(bits & 0x8000 != 0),
        _ => None,
    }
}

/// `have` is `want`, or both are NaNs of one sign (see the module note).
fn same(to: Code, have: u64, want: u64) -> bool {
    match nan_sign(to, want) {
        Some(sign) => nan_sign(to, have) == Some(sign),
        None => have == want,
    }
}

fn encode(code: Code, values: &[u64]) -> Vec<u8> {
    let es = code.bytes() as usize;
    values
        .iter()
        .flat_map(|&v| v.to_le_bytes()[..es].to_vec())
        .collect()
}

fn read(code: Code, bytes: &[u8], i: usize) -> u64 {
    let es = code.bytes() as usize;
    let mut word = [0u8; 8];
    word[..es].copy_from_slice(&bytes[i * es..(i + 1) * es]);
    u64::from_le_bytes(word)
}

#[test]
fn every_pair_converts_bit_for_bit_as_c10_does() {
    let Some(mut device) = open_or_unmeasured("convert: every pair") else {
        return;
    };
    let table = f16_table();
    let mut checked = 0usize;
    for from in Code::ALL {
        let values = inputs(from, 4096);
        let n = values.len();
        let src = device.upload(&encode(from, &values), 0);
        for to in Code::ALL {
            let r = flat(from, to, n as u64);
            let out_bytes = n * to.bytes() as usize;
            // poisoned, with a known pad past the last element
            let out = device.upload(&vec![0xa5u8; out_bytes], 0x5a);
            let rc = device.convert(src, out, &r);
            if from.is_float() && !to.is_float() && to != Code::Bool {
                assert_eq!(rc, ffi::ACK_ERR_SHAPE, "{from:?} -> {to:?} is refused");
                assert!(
                    last_error().contains("convert-float-to-integer"),
                    "{}",
                    last_error()
                );
                continue;
            }
            assert_eq!(rc, ffi::ACK_OK, "{from:?} -> {to:?}: {}", last_error());
            let padded = out_bytes.next_multiple_of(4).max(4);
            let got = device.download(out, padded);
            for (i, &raw) in values.iter().enumerate() {
                let want = reference(decode(from, raw), to, &table);
                let have = read(to, &got, i);
                assert!(
                    same(to, have, want),
                    "{from:?} -> {to:?}, input {raw:#x} (element {i}): got {have:#x}, want {want:#x}"
                );
            }
            assert!(
                got[out_bytes..].iter().all(|&b| b == 0x5a),
                "{from:?} -> {to:?}: the bytes past the last element came back changed"
            );
            checked += n;
        }
    }
    assert!(checked > 80 * 4096, "{checked} conversions checked");
}

#[test]
fn a_strided_input_and_a_strided_output_convert_in_place() {
    use alelyon_compute_kit::convert_ops::{
        Bindings, ConvertPlan, ConvertRequest, Storage, View, FLAG_ATOMIC,
    };
    let Some(mut device) = open_or_unmeasured("convert: strided") else {
        return;
    };
    let table = f16_table();
    // a [3, 5, 4, 6] view of a larger i32 buffer: permuted strides, an offset
    let shape = [3u64, 5, 4, 6];
    let base: Vec<u64> = (0..1200u64)
        .map(|i| (i * 2_654_435_761) & 0xffff_ffff)
        .collect();
    let src = device.upload(&encode(Code::I32, &base), 0);
    let in_strides = [1u64, 3, 15 * 4, 15];
    let offset = 17u64;
    let at = |c: [u64; 4]| (offset + (0..4).map(|k| c[k] * in_strides[k]).sum::<u64>()) as usize;
    let n = shape.iter().product::<u64>() as usize;
    let coords = |l: usize| {
        let mut c = [0u64; 4];
        let mut rest = l as u64;
        for k in (0..4).rev() {
            c[k] = rest % shape[k];
            rest /= shape[k];
        }
        c
    };
    // (a) into a dense f16 output: sub-word, whole words per invocation
    let mut r = flat(Code::I32, Code::F16, 0);
    r.ndim = 4;
    r.shape = shape;
    r.in_offset = offset;
    r.in_strides = in_strides;
    r.out_strides = [120, 24, 6, 1];
    let out = device.upload(&vec![0xa5u8; n * 2], 0x5a);
    assert_eq!(
        device.convert(src, out, &r),
        ffi::ACK_OK,
        "{}",
        last_error()
    );
    let got = device.download(out, n * 2);
    for l in 0..n {
        let want = reference(decode(Code::I32, base[at(coords(l))]), Code::F16, &table);
        assert_eq!(read(Code::F16, &got, l), want, "element {l}");
    }
    // (b) into a transposed f64 output at an offset: whole-word elements

    r.out_code = Code::F64 as u32;
    r.out_offset = 3;
    r.out_strides = [1, 3, 15, 60];
    let out = device.upload(&vec![0xa5u8; (3 + n) * 8], 0);
    assert_eq!(
        device.convert(src, out, &r),
        ffi::ACK_OK,
        "{}",
        last_error()
    );
    let got = device.download(out, (3 + n) * 8);
    assert!(
        got[..24].iter().all(|&b| b == 0xa5),
        "the offset's elements are untouched"
    );
    for l in 0..n {
        let c = coords(l);
        let e = (3 + c[0] + 3 * c[1] + 15 * c[2] + 60 * c[3]) as usize;
        let want = reference(decode(Code::I32, base[at(c)]), Code::F64, &table);
        assert_eq!(read(Code::F64, &got, e), want, "element {l}");
    }
    // (c) strided sub-word outputs (schema 2). Since 2026-10-03 the planner
    // reads a view in its output's memory order, where the transposed bf16 view
    // at an odd offset is dense: whole words, the first shared with an element
    // before the offset. The padded bf16 view (a gap after every third element)
    // and the stride-2 u8 view are dense in no order and share their words with
    // their own other elements and with bytes nobody writes, so they take
    // atomics. Every byte a view does not address must come back as it was.
    for (to, off, strides, path) in [
        (Code::Bf16, 3u64, [1u64, 3, 15, 60], 0),
        (Code::Bf16, 3, [1, 4, 20, 80], FLAG_ATOMIC),
        (Code::U8, 1, [2, 6, 30, 120], FLAG_ATOMIC),
    ] {
        let request = ConvertRequest {
            from: Code::I32,
            to,
            ndim: 4,
            shape,
            input: View::new(offset, in_strides),
            out: View::new(off, strides),
        };
        let unbounded = Bindings {
            input: Storage::new(1, 1 << 40),
            out: Storage::new(2, 1 << 40),
        };
        let plan = ConvertPlan::new(request, unbounded).unwrap();
        assert_eq!(
            plan.words()[1],
            path,
            "{to:?} {strides:?}: the planner's path"
        );
        let span = 1 + (off + (0..4).map(|k| (shape[k] - 1) * strides[k]).sum::<u64>()) as usize;
        let es = to.bytes() as usize;
        let out = device.upload(&vec![0xa5u8; span * es], 0x5a);
        let mut r = flat(Code::I32, to, 0);
        r.ndim = 4;
        r.shape = shape;
        r.in_offset = offset;
        r.in_strides = in_strides;
        r.out_offset = off;
        r.out_strides = strides;
        assert_eq!(
            device.convert(src, out, &r),
            ffi::ACK_OK,
            "{to:?}: {}",
            last_error()
        );
        // the buffer is a whole number of words, and a download names its whole length
        let got = device.download(out, (span * es).div_ceil(4) * 4);
        let mut written = vec![false; span];
        for l in 0..n {
            let c = coords(l);
            let e = (off + (0..4).map(|k| c[k] * strides[k]).sum::<u64>()) as usize;
            written[e] = true;
            let want = reference(decode(Code::I32, base[at(c)]), to, &table);
            assert!(same(to, read(to, &got, e), want), "{to:?} element {l}");
        }
        for (e, w) in written.iter().enumerate() {
            if !w {
                assert!(
                    got[e * es..(e + 1) * es].iter().all(|&byte| byte == 0xa5),
                    "{to:?}: element {e}, which the view does not address, changed"
                );
            }
        }
    }
    // (d) an output that is the input's buffer is refused
    let r = flat(Code::I32, Code::F32, 16);
    assert_eq!(device.convert(src, src, &r), ffi::ACK_ERR_SHAPE);
    assert!(
        last_error().contains("convert-output-overlaps-input"),
        "{}",
        last_error()
    );
}

#[test]
fn an_odd_length_sub_word_output_keeps_the_rest_of_its_last_word() {
    let Some(mut device) = open_or_unmeasured("convert: last word") else {
        return;
    };
    let table = f16_table();
    for (to, n) in [
        (Code::U8, 5usize),
        (Code::Bool, 7),
        (Code::I8, 1),
        (Code::F16, 3),
        (Code::I16, 1),
    ] {
        let values: Vec<u64> = (0..n as u64).map(|i| i * 77 + 3).collect();
        let src = device.upload(&encode(Code::I64, &values), 0);
        // the output buffer is a whole number of words; everything poisoned
        let words = (n * to.bytes() as usize).div_ceil(4);
        let out = device.upload(&vec![0xc3u8; words * 4], 0);
        let rc = device.convert(src, out, &flat(Code::I64, to, n as u64));
        assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
        let got = device.download(out, words * 4);
        let end = n * to.bytes() as usize;
        for (i, &v) in values.iter().enumerate() {
            assert_eq!(
                read(to, &got, i),
                reference(V::Int(v as i64), to, &table),
                "{to:?} element {i}"
            );
        }
        assert!(
            got[end..].iter().all(|&b| b == 0xc3),
            "{to:?}: bytes past {end} changed: {:?}",
            &got[end..]
        );
    }
}

/// The elements a strided output writes, each byte of the buffer marked: element index or None.
fn owners(es: usize, buffer_bytes: usize, addresses: &[usize]) -> Vec<Option<usize>> {
    let mut owner = vec![None; buffer_bytes];
    for (i, &a) in addresses.iter().enumerate() {
        for b in 0..es {
            owner[a * es + b] = Some(i);
        }
    }
    owner
}

#[test]
fn a_dense_sub_word_output_at_any_offset_takes_whole_words_and_keeps_its_neighbours() {
    // 2026-10-01: a dense 1- or 2-byte output that does not start on a word boundary is written whole words,
    // its first and last words read-modify-written by the one invocation that owns each (it took atomics for
    // every element before). Every byte outside the output must come back as it was.
    let Some(mut device) = open_or_unmeasured("convert: dense output at an offset") else {
        return;
    };
    let table = f16_table();
    for (to, n, offset) in [
        (Code::U8, 5usize, 1usize),
        (Code::U8, 9, 3),
        (Code::I8, 13, 2),
        (Code::Bool, 6, 2),
        (Code::I16, 3, 1),
        (Code::I16, 8, 1),
        (Code::F16, 5, 1),
        (Code::I16, 1, 1),
        (Code::U8, 1, 3),
    ] {
        let es = to.bytes() as usize;
        let values: Vec<u64> = (0..n as u64).map(|i| i * 131 + 7).collect();
        let src = device.upload(&encode(Code::I64, &values), 0);
        let words = ((offset + n) * es).div_ceil(4) + 1;
        let out = device.upload(&vec![0xc3u8; words * 4], 0);
        let mut r = flat(Code::I64, to, n as u64);
        r.out_offset = offset as u64;
        let rc = device.convert(src, out, &r);
        assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
        let got = device.download(out, words * 4);
        for (i, &v) in values.iter().enumerate() {
            assert_eq!(
                read(to, &got, offset + i),
                reference(V::Int(v as i64), to, &table),
                "{to:?} n {n} offset {offset}: element {i}"
            );
        }
        let (start, end) = (offset * es, (offset + n) * es);
        assert!(
            got[..start].iter().chain(&got[end..]).all(|&b| b == 0xc3),
            "{to:?} n {n} offset {offset}: a byte outside [{start}, {end}) changed: {got:?}"
        );
    }
    // a two-dimension dense output at an odd offset: shape (3, 5), row-major from element 1
    let to = Code::I16;
    let values: Vec<u64> = (0..15u64).map(|i| 1000 + i * 3).collect();
    let src = device.upload(&encode(Code::I64, &values), 0);
    let out = device.upload(&[0xc3u8; 40], 0);
    let r = ffi::AckConvertRequest {
        src_code: Code::I64 as u32,
        out_code: to as u32,
        ndim: 2,
        flags: 0,
        shape: [3, 5, 0, 0],
        in_offset: 0,
        in_strides: [5, 1, 0, 0],
        out_offset: 1,
        out_strides: [5, 1, 0, 0],
    };
    assert_eq!(
        device.convert(src, out, &r),
        ffi::ACK_OK,
        "{}",
        last_error()
    );
    let got = device.download(out, 40);
    for (i, &v) in values.iter().enumerate() {
        assert_eq!(
            read(to, &got, 1 + i),
            reference(V::Int(v as i64), to, &table),
            "2-D element {i}"
        );
    }
    assert!(
        got[..2].iter().chain(&got[32..]).all(|&b| b == 0xc3),
        "2-D: outside bytes changed: {got:?}"
    );
}

#[test]
fn a_strided_sub_word_output_with_a_word_to_each_element_keeps_the_bytes_between() {
    // 2026-10-01: a 1- or 2-byte output whose every stride is a multiple of the slots a word has puts each
    // element in a word of its own, and is written with a plain read-modify-write of that word (it took two
    // atomics an element before). The bytes between the elements must come back as they were.
    let Some(mut device) = open_or_unmeasured("convert: word-exclusive output") else {
        return;
    };
    let table = f16_table();
    for (to, n, stride, offset) in [
        (Code::I16, 7usize, 2usize, 0usize),
        (Code::I16, 6, 2, 1),
        (Code::F16, 5, 4, 1),
        (Code::I16, 4, 6, 3),
        (Code::U8, 5, 4, 0),
        (Code::U8, 5, 4, 3),
        (Code::Bool, 6, 8, 2),
        (Code::I8, 9, 4, 1),
    ] {
        let es = to.bytes() as usize;
        let values: Vec<u64> = (0..n as u64).map(|i| i * 57 + 11).collect();
        let src = device.upload(&encode(Code::I64, &values), 0);
        let addresses: Vec<usize> = (0..n).map(|i| offset + i * stride).collect();
        let words = ((addresses[n - 1] + 1) * es).div_ceil(4) + 1;
        let out = device.upload(&vec![0xc3u8; words * 4], 0);
        let mut r = flat(Code::I64, to, n as u64);
        r.out_offset = offset as u64;
        r.out_strides = [stride as u64, 0, 0, 0];
        let rc = device.convert(src, out, &r);
        assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
        let got = device.download(out, words * 4);
        for (i, &v) in values.iter().enumerate() {
            assert_eq!(
                read(to, &got, addresses[i]),
                reference(V::Int(v as i64), to, &table),
                "{to:?} stride {stride} offset {offset}: element {i}"
            );
        }
        let owner = owners(es, words * 4, &addresses);
        for (b, o) in owner.iter().enumerate() {
            if o.is_none() {
                assert_eq!(
                    got[b], 0xc3,
                    "{to:?} stride {stride} offset {offset}: byte {b} between elements changed"
                );
            }
        }
    }
}

/// A transposed read of two dimensions (`convert_ops::FLAG_TILED`, 2026-10-03):
/// the planner takes the tiled module, and every element is the reference's, bit
/// for bit, with tiles cut short at both edges and the bytes on either side of
/// the output kept. The input is a [cols, ld] buffer read through its transpose
/// (strides 1 and ld) at an offset; its values cycle through the type's special
/// values and a pseudo-random set, so a tile read or written in the wrong order
/// lands a different value on most elements.
#[test]
fn a_transposed_read_is_tiled_and_converts_bit_for_bit() {
    use alelyon_compute_kit::convert_ops::{Bindings, ConvertPlan, ConvertRequest, Storage, View};
    let Some(mut device) = open_or_unmeasured("convert: tiled transposed read") else {
        return;
    };
    let table = f16_table();
    // (from, to, rows, cols, ld, in_offset, out_offset)
    let cases: [(Code, Code, u64, u64, u64, u64, u64); 9] = [
        (Code::I8, Code::F16, 67, 130, 70, 5, 0), // the stage-E head's dtype pair
        (Code::I8, Code::F32, 64, 129, 64, 0, 3), // a 4-byte output: any row length and offset
        (Code::U8, Code::Bf16, 100, 66, 128, 1, 2),
        (Code::I16, Code::F16, 70, 64, 96, 3, 0),  // rounds
        (Code::F32, Code::F16, 65, 132, 80, 0, 0), // rounds, and the specials
        (Code::F16, Code::F32, 33, 200, 40, 1, 0),
        (Code::Bool, Code::F16, 64, 64, 64, 0, 0),
        (Code::I32, Code::I8, 40, 132, 50, 2, 4), // a 1-byte output: rows of whole words
        (Code::F32, Code::Bool, 31, 68, 33, 0, 4),
    ];
    for (from, to, rows, cols, ld, in_offset, out_offset) in cases {
        let what = format!("{from:?} -> {to:?} {rows}x{cols} ld {ld}");
        let request = ConvertRequest {
            from,
            to,
            ndim: 2,
            shape: [rows, cols, 0, 0],
            input: View::new(in_offset, [1, ld, 0, 0]),
            out: View::new(out_offset, [cols, 1, 0, 0]),
        };
        let unbounded = Bindings {
            input: Storage::new(1, 1 << 40),
            out: Storage::new(2, 1 << 40),
        };
        let plan = ConvertPlan::new(request, unbounded).expect(&what);
        assert!(
            plan.tiled(),
            "{what}: the planner must take the tiled module"
        );
        let pool = inputs(from, 509);
        let in_len = (in_offset + ld * cols) as usize;
        let values: Vec<u64> = (0..in_len)
            .map(|i| pool[(i * 7 + 3) % pool.len()])
            .collect();
        let src = device.upload(&encode(from, &values), 0);
        let es = to.bytes() as usize;
        let n = (rows * cols) as usize;
        // whole words: the kit's download takes the buffer's whole (word-rounded) length
        let out_len = ((out_offset as usize + n + 9) * es).div_ceil(4) * 4;
        let out = device.upload(&vec![0xa5u8; out_len], 0x5a);
        let mut r = flat(from, to, 0);
        r.ndim = 2;
        r.shape = [rows, cols, 0, 0];
        r.in_offset = in_offset;
        r.in_strides = [1, ld, 0, 0];
        r.out_offset = out_offset;
        r.out_strides = [cols, 1, 0, 0];
        assert_eq!(
            device.convert(src, out, &r),
            ffi::ACK_OK,
            "{what}: {}",
            last_error()
        );
        let got = device.download(out, out_len);
        let first = out_offset as usize * es;
        let last = (out_offset as usize + n) * es;
        assert!(
            got[..first].iter().all(|&b| b == 0xa5),
            "{what}: a byte before the output changed"
        );
        assert!(
            got[last..].iter().all(|&b| b == 0xa5),
            "{what}: a byte after the output changed"
        );
        for row in 0..rows as usize {
            for col in 0..cols as usize {
                let raw = values[in_offset as usize + row + col * ld as usize];
                let want = reference(decode(from, raw), to, &table);
                let have = read(to, &got, out_offset as usize + row * cols as usize + col);
                assert!(
                    same(to, have, want),
                    "{what}: ({row}, {col}) is {have:#x}, the reference {want:#x}"
                );
            }
        }
    }
}

/// A transposed view converted into the SAME layout (2026-10-03): the stage-E
/// head's cast as torch hands it to the kit, since `x.t().to(dtype)` gives the
/// output x's strides. In the output's memory order, where the planner reads it,
/// input and output are one dense run, so the planner takes neither atomics nor
/// the tiled module, a 1- or 2-byte output takes whole words, and every element
/// is the reference's, bit for bit, with the bytes on either side of the output
/// kept. The input is a [cols, rows] buffer read through its transpose (strides
/// 1 and rows) at an offset, and the output is the same view of its own buffer.
#[test]
fn a_transposed_view_into_the_same_layout_converts_as_one_dense_run() {
    use alelyon_compute_kit::convert_ops::{Bindings, ConvertPlan, ConvertRequest, Storage, View};
    let Some(mut device) = open_or_unmeasured("convert: the output's memory order") else {
        return;
    };
    let table = f16_table();
    // (from, to, rows, cols, in_offset, out_offset)
    let cases: [(Code, Code, u64, u64, u64, u64); 6] = [
        (Code::I8, Code::F16, 67, 130, 5, 0), // the stage-E head's dtype pair
        (Code::I8, Code::F16, 67, 130, 0, 1), // an odd offset: part words at both ends
        (Code::F32, Code::Bf16, 33, 70, 2, 3), // rounds
        (Code::U8, Code::I8, 9, 31, 1, 2),    // four elements a word
        (Code::I16, Code::F32, 40, 50, 0, 0), // a 4-byte output
        (Code::F64, Code::F16, 17, 24, 3, 0), // an 8-byte input
    ];
    for (from, to, rows, cols, in_offset, out_offset) in cases {
        let what = format!("{from:?} -> {to:?} {rows}x{cols}");
        let n = rows * cols;
        let request = ConvertRequest {
            from,
            to,
            ndim: 2,
            shape: [rows, cols, 0, 0],
            input: View::new(in_offset, [1, rows, 0, 0]),
            out: View::new(out_offset, [1, rows, 0, 0]),
        };
        let unbounded = Bindings {
            input: Storage::new(1, 1 << 40),
            out: Storage::new(2, 1 << 40),
        };
        let plan = ConvertPlan::new(request, unbounded).expect(&what);
        let w = plan.words();
        assert_eq!(w[1], 0, "{what}: neither atomics nor the tiled module");
        assert_eq!(&w[2..6], &[1, 1, 1, n as u32], "{what}: one run");
        let es = to.bytes();
        let units = if es < 4 {
            (out_offset % (4 / es) + n).div_ceil(4 / es)
        } else {
            n
        };
        assert_eq!(u64::from(plan.units()), units, "{what}: whole words");
        let pool = inputs(from, 509);
        let values: Vec<u64> = (0..in_offset + n)
            .map(|i| pool[(i as usize * 7 + 3) % pool.len()])
            .collect();
        let src = device.upload(&encode(from, &values), 0);
        let es = es as usize;
        // whole words: the kit's download takes the buffer's whole (word-rounded) length
        let out_len = ((out_offset + n + 9) as usize * es).div_ceil(4) * 4;
        let out = device.upload(&vec![0xa5u8; out_len], 0x5a);
        let mut r = flat(from, to, 0);
        r.ndim = 2;
        r.shape = [rows, cols, 0, 0];
        r.in_offset = in_offset;
        r.in_strides = [1, rows, 0, 0];
        r.out_offset = out_offset;
        r.out_strides = [1, rows, 0, 0];
        assert_eq!(
            device.convert(src, out, &r),
            ffi::ACK_OK,
            "{what}: {}",
            last_error()
        );
        let got = device.download(out, out_len);
        let first = out_offset as usize * es;
        let last = (out_offset + n) as usize * es;
        assert!(
            got[..first].iter().all(|&b| b == 0xa5),
            "{what}: a byte before the output changed"
        );
        assert!(
            got[last..].iter().all(|&b| b == 0xa5),
            "{what}: a byte after the output changed"
        );
        for row in 0..rows {
            for col in 0..cols {
                let at = row + col * rows;
                let want = reference(decode(from, values[(in_offset + at) as usize]), to, &table);
                let have = read(to, &got, (out_offset + at) as usize);
                assert!(
                    same(to, have, want),
                    "{what}: ({row}, {col}) is {have:#x}, the reference {want:#x}"
                );
            }
        }
    }
}

/// SAME-TYPE COPIES (2026-10-05, `kernels/convert.comp`): a conversion between a
/// code and itself whose routine is already the identity -- the integer codes,
/// f32 and f64, every code the torch adapter's same-dtype bits copies plan --
/// moves each element's bits through no routine, and stores a 2-byte element of
/// an output that is not dense on its own (no read-modify-write, no atomics).
/// The result must still be the routines' result, which for these codes is the
/// input's bits exactly (a NaN keeps its payload), on every path the planner
/// can take: dense at offsets 0 to 3, a word to each element, words shared
/// between elements, and 4- and 8-byte strided outputs. The whole output buffer
/// is compared with the expected image, so every byte the output does not own
/// must come back as it was.
#[test]
fn a_same_type_copy_moves_every_bit_on_every_path_and_keeps_its_neighbours() {
    use alelyon_compute_kit::convert_ops::{
        Bindings, ConvertPlan, ConvertRequest, Storage, View, FLAG_ATOMIC, FLAG_EXCLUSIVE,
    };
    let Some(mut device) = open_or_unmeasured("convert: same-type copies") else {
        return;
    };
    /// One output layout, and the flags word the planner must choose for it.
    struct Layout {
        shape: [u64; 2],
        in_strides: [u64; 2],
        out_offset: u64,
        out_strides: [u64; 2],
        flags: u32,
    }
    let mut checked = 0usize;
    for code in [
        Code::I64,
        Code::I32,
        Code::I16,
        Code::I8,
        Code::U8,
        Code::F32,
        Code::F64,
    ] {
        let es = code.bytes();
        let slots = if es < 4 { 4 / es } else { 1 };
        // dense at offsets 0 to 3
        let mut layouts: Vec<Layout> = (0..4)
            .map(|out_offset| Layout {
                shape: [1, 37],
                in_strides: [37, 1],
                out_offset,
                out_strides: [37, 1],
                flags: 0,
            })
            .collect();
        let stride = slots.max(2);
        // a word to each element
        layouts.push(Layout {
            shape: [1, 29],
            in_strides: [29, 1],
            out_offset: 1,
            out_strides: [29 * stride, stride],
            flags: if es < 4 { FLAG_EXCLUSIVE } else { 0 },
        });
        // rows of 7, two elements apart: words shared between elements
        layouts.push(Layout {
            shape: [5, 7],
            in_strides: [7, 1],
            out_offset: 1,
            out_strides: [9, 1],
            flags: if es < 4 { FLAG_ATOMIC } else { 0 },
        });
        let pool = inputs(code, 300);
        for Layout {
            shape,
            in_strides,
            out_offset,
            out_strides,
            flags,
        } in layouts
        {
            let what =
                format!("{code:?} {shape:?} out offset {out_offset} strides {out_strides:?}");
            let request = ConvertRequest {
                from: code,
                to: code,
                ndim: 2,
                shape: [shape[0], shape[1], 0, 0],
                input: View::new(0, [in_strides[0], in_strides[1], 0, 0]),
                out: View::new(out_offset, [out_strides[0], out_strides[1], 0, 0]),
            };
            let unbounded = Bindings {
                input: Storage::new(1, 1 << 40),
                out: Storage::new(2, 1 << 40),
            };
            let plan = ConvertPlan::new(request, unbounded).expect(&what);
            assert_eq!(plan.words()[1], flags, "{what}: the path");
            let n_in = (shape[0] * in_strides[0]) as usize;
            let values: Vec<u64> = (0..n_in).map(|i| pool[(i * 7 + 3) % pool.len()]).collect();
            let src = device.upload(&encode(code, &values), 0);
            let last =
                out_offset + (shape[0] - 1) * out_strides[0] + (shape[1] - 1) * out_strides[1];
            let out_len = ((last + 1) * es).div_ceil(4) as usize * 4 + 4;
            let mut want = vec![0xc3u8; out_len];
            let out = device.upload(&want, 0);
            for r in 0..shape[0] {
                for c in 0..shape[1] {
                    let i = (r * in_strides[0] + c * in_strides[1]) as usize;
                    let o = (out_offset + r * out_strides[0] + c * out_strides[1]) as usize;
                    let bytes = values[i].to_le_bytes();
                    want[o * es as usize..(o + 1) * es as usize]
                        .copy_from_slice(&bytes[..es as usize]);
                }
            }
            let mut r = flat(code, code, 0);
            r.ndim = 2;
            r.shape = [shape[0], shape[1], 0, 0];
            r.in_strides = [in_strides[0], in_strides[1], 0, 0];
            r.out_offset = out_offset;
            r.out_strides = [out_strides[0], out_strides[1], 0, 0];
            assert_eq!(
                device.convert(src, out, &r),
                ffi::ACK_OK,
                "{what}: {}",
                last_error()
            );
            let got = device.download(out, out_len);
            if let Some(b) = (0..out_len).find(|&b| got[b] != want[b]) {
                panic!(
                    "{what}: byte {b} (element {}) is {:#04x}, want {:#04x}",
                    b / es as usize,
                    got[b],
                    want[b]
                );
            }
            checked += (shape[0] * shape[1]) as usize;
        }
    }
    assert!(checked > 7 * 200, "{checked} elements checked");
}

/// The codes SAME-TYPE COPIES leaves out keep their routines' answers, dense
/// and a word to each element: bf16 -> bf16 quiets a signalling NaN as
/// cast_f32_bf16.comp writes one, f16 -> f16 writes c10's sign | 0x7E00, and
/// bool -> bool writes 1 for any nonzero byte. A same-type path that took them
/// would return the input unchanged instead.
#[test]
fn a_same_type_bf16_f16_or_bool_conversion_keeps_its_routine() {
    let Some(mut device) = open_or_unmeasured("convert: same-type routines kept") else {
        return;
    };
    for (code, input, want) in [
        (Code::Bf16, 0x7f81u64, 0x7fc1u64),
        (Code::Bf16, 0xff83, 0xffc3),
        (Code::F16, 0x7c01, 0x7e00),
        (Code::F16, 0xfc01, 0xfe00),
        (Code::Bool, 2, 1),
        (Code::Bool, 0x80, 1),
    ] {
        let es = code.bytes() as usize;
        for stride in [1usize, 4 / es] {
            let src = device.upload(&encode(code, &[input; 3]), 0);
            let words = (3 * stride * es).div_ceil(4) + 1;
            let out = device.upload(&vec![0u8; words * 4], 0);
            let mut r = flat(code, code, 3);
            r.out_strides = [stride as u64, 0, 0, 0];
            assert_eq!(
                device.convert(src, out, &r),
                ffi::ACK_OK,
                "{}",
                last_error()
            );
            let got = device.download(out, words * 4);
            for i in 0..3 {
                assert_eq!(
                    read(code, &got, i * stride),
                    want,
                    "{code:?} {input:#x} stride {stride}: element {i}"
                );
            }
        }
    }
}
