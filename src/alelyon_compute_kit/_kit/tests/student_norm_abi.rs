//! `ack_student_norm_f16` (student-norm-f16/v1, 2026-09-29): the sub-bit
//! student's attention-block RMSNorms, the residual add before the
//! post-attention one, and its q/k norms with RoPE, forward and backward, on a
//! real device.
//!
//! THE LAW is `student_norm_ops::reference`, the operations in f64 with the
//! kernel's fp16 roundings at the kernel's sites:
//! - the residual sum `h = x + delta` BIT FOR BIT (one f32 add);
//! - the f32 outputs to 2e-6 of their value (op 0) or 1e-5 of their terms'
//!   magnitudes (op 1: the backward's two terms can cancel);
//! - every value an fp16 rounding reaches to 2^-9 of its magnitude (one fp16
//!   ulp where the kernel's f32 value and the law's fall either side of a
//!   rounding boundary), plus the output's own rounding.
//!
//! The shapes cover the widest norm row (4,096) and q/k row (256), rows of a
//! width that is not a multiple of the 64 lanes' pairs, a backward group that
//! ends inside the last rows, both q/k layouts, and grids folded past 65,535
//! workgroups. Every output is poisoned first, and an output the flags do not
//! ask for must keep its poison. Refusals are checked by name. Skips as
//! UNMEASURED without a device unless ACK_REQUIRE_DEVICE is set.
use alelyon_compute_kit::ffi;
use alelyon_compute_kit::head_loss_ops::{f16_to_f32, f32_to_f16};
use alelyon_compute_kit::student_norm_ops::{
    reference, NormOp, ADD, AUX16, DH, DY16, DY_ABSENT, SAME_LAYOUT, X16, Y16, Y32,
};
use std::ffi::{c_char, CStr};

const EPS: f32 = 1e-6;
const POISON32: u32 = 0xA5A5_A5A5;
const POISON16: u16 = 0xA5A5;

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

type Buf = *mut alelyon_compute_kit::Buffer;

fn alloc(dev: *mut ffi::AckDevice, bytes: &[u8]) -> Buf {
    let b = ffi::ack_buffer_alloc(dev, bytes.len().max(4) as u64);
    assert!(!b.is_null(), "{}", last_error());
    let rc = unsafe { ffi::ack_upload(dev, b, bytes.as_ptr(), bytes.len()) };
    assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
    b
}

fn poisoned(dev: *mut ffi::AckDevice, bytes: usize) -> Buf {
    alloc(dev, &vec![0xA5u8; bytes])
}

fn download(dev: *mut ffi::AckDevice, buf: Buf, bytes: usize) -> Vec<u8> {
    let mut out = vec![0u8; bytes];
    let rc = unsafe { ffi::ack_download(dev, buf, out.as_mut_ptr(), bytes) };
    assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
    out
}

fn free(dev: *mut ffi::AckDevice, bufs: &[Buf]) {
    for &b in bufs {
        assert_eq!(ffi::ack_buffer_free(dev, b), ffi::ACK_OK);
    }
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn f16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|x| f32_to_f16(*x).to_le_bytes())
        .collect()
}

fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn u16s(b: &[u8]) -> Vec<u16> {
    b.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn unit(i: usize, salt: u32) -> f32 {
    let hash = (i as u32).wrapping_mul(2_654_435_761) ^ salt.wrapping_mul(40_503);
    (hash % 20_011) as f32 / 10_000.0 - 1.0
}

/// A value fp16 holds exactly.
fn half(v: f32) -> f32 {
    f16_to_f32(f32_to_f16(v))
}

/// The rope tables of positions 0..t, (t, d) each: angle t * base^(-2j/d)
/// for pair j of the half, repeated over both halves, as the student builds
/// them (layer.rotary_cos_sin, base 1e7).
fn rope_tables(t: usize, d: usize) -> (Vec<f32>, Vec<f32>) {
    let half_d = d / 2;
    let (mut cos, mut sin) = (Vec::with_capacity(t * d), Vec::with_capacity(t * d));
    for pos in 0..t {
        for c in 0..d {
            let j = (c % half_d) as f64;
            let angle = pos as f64 * 1e7f64.powf(-2.0 * j / d as f64);
            cos.push(angle.cos() as f32);
            sin.push(angle.sin() as f32);
        }
    }
    (cos, sin)
}

/// How many of `got` lie farther from `want` than `tol(i)` (a NaN always
/// does), and the first of them.
fn beyond(
    got: &[f64],
    want: &[f64],
    tol: impl Fn(usize) -> f64,
) -> (usize, Option<(usize, f64, f64)>) {
    let mut count = 0;
    let mut first = None;
    for i in 0..want.len() {
        // written so that a NaN on either side is never within
        let within = (got[i] - want[i]).abs() <= tol(i);
        if !within {
            count += 1;
            first.get_or_insert((i, got[i], want[i]));
        }
    }
    (count, first)
}

fn widen(v: &[f32]) -> Vec<f64> {
    v.iter().map(|x| f64::from(*x)).collect()
}

fn widen16(v: &[u16]) -> Vec<f64> {
    v.iter().map(|x| f64::from(f16_to_f32(*x))).collect()
}

#[allow(clippy::too_many_arguments)]
fn call(
    dev: *mut ffi::AckDevice,
    b: [Buf; 8],
    op: u32,
    flags: u32,
    rows: usize,
    d: usize,
    eps: f32,
    t: usize,
    h: usize,
    rows_per_group: usize,
) -> i32 {
    unsafe {
        ffi::ack_student_norm_f16(
            dev,
            b[0],
            b[1],
            b[2],
            b[3],
            b[4],
            b[5],
            b[6],
            b[7],
            op,
            flags,
            rows as u32,
            d as u32,
            eps,
            t as u32,
            h as u32,
            rows_per_group as u32,
            std::ptr::null_mut(),
        )
    }
}

#[test]
fn the_norm_and_the_residual_add_are_the_law() {
    let Some(dev) = open_or_unmeasured("student norm forward") else {
        return;
    };
    let spare = alloc(dev, &[0u8; 16]);
    for (rows, d, flags) in [
        (3_usize, 4096_usize, Y32 | Y16),
        (37, 130, X16 | Y32 | Y16),
        (5, 2048, ADD | Y32 | Y16),
        (5, 2048, ADD | AUX16 | Y16),
        (66_000, 2, Y32),
    ] {
        let n = rows * d;
        let narrow = flags & X16 != 0;
        let x: Vec<f32> = (0..n)
            .map(|i| {
                let v = unit(i, 3) * 2.0;
                if narrow {
                    half(v)
                } else {
                    v
                }
            })
            .collect();
        let delta: Option<Vec<f32>> = (flags & ADD != 0).then(|| {
            (0..n)
                .map(|i| {
                    let v = unit(i, 5) * 0.5;
                    if flags & AUX16 != 0 {
                        half(v)
                    } else {
                        v
                    }
                })
                .collect()
        });
        let w: Vec<f32> = (0..d).map(|c| 0.75 + 0.5 * unit(c, 7)).collect();
        let (h, y) = reference::norm(&x, delta.as_deref(), &w, d, EPS, narrow);
        let bx = alloc(dev, &if narrow { f16_bytes(&x) } else { f32_bytes(&x) });
        let baux = match &delta {
            Some(dl) if flags & AUX16 != 0 => alloc(dev, &f16_bytes(dl)),
            Some(dl) => alloc(dev, &f32_bytes(dl)),
            None => alloc(dev, &[0u8; 16]),
        };
        let bw = alloc(dev, &f32_bytes(&w));
        let (out0, out1, io2) = (
            poisoned(dev, n * 4),
            poisoned(dev, n * 2),
            poisoned(dev, n * 4),
        );
        let rc = call(
            dev,
            [bx, baux, bw, spare, out0, out1, spare, io2],
            NormOp::Norm as u32,
            flags,
            rows,
            d,
            EPS,
            1,
            1,
            1,
        );
        let what = format!("{rows} rows of {d}, flags {flags}");
        assert_eq!(rc, ffi::ACK_OK, "{what}: {}", last_error());
        let y32 = f32s(&download(dev, out0, n * 4));
        let y16 = u16s(&download(dev, out1, n * 2));
        let h_out = f32s(&download(dev, io2, n * 4));
        if flags & Y32 != 0 {
            let tol = if narrow { 2f64.powi(-9) } else { 2e-6 };
            let (bad, first) = beyond(&widen(&y32), &y, |i| tol * y[i].abs() + 1e-30);
            assert_eq!(
                bad, 0,
                "{what}: y (f32) beyond {tol:e} of itself; first {first:?}"
            );
        } else {
            assert!(
                y32.iter().all(|v| v.to_bits() == POISON32),
                "{what}: out0 written without Y32"
            );
        }
        if flags & Y16 != 0 {
            let tol = if narrow {
                2f64.powi(-9)
            } else {
                2f64.powi(-10)
            };
            let (bad, first) = beyond(&widen16(&y16), &y, |i| tol * y[i].abs() + 2f64.powi(-24));
            assert_eq!(
                bad, 0,
                "{what}: y (fp16) beyond {tol:e} of itself; first {first:?}"
            );
        } else {
            assert!(
                y16.iter().all(|&v| v == POISON16),
                "{what}: out1 written without Y16"
            );
        }
        if flags & ADD != 0 {
            let bad = (0..n)
                .filter(|&i| h_out[i].to_bits() != h[i].to_bits())
                .count();
            assert_eq!(
                bad, 0,
                "{what}: {bad} of {n} residual sums are not the f32 add's"
            );
        } else {
            assert!(
                h_out.iter().all(|v| v.to_bits() == POISON32),
                "{what}: io2 written without ADD"
            );
        }
        free(dev, &[bx, baux, bw, out0, out1, io2]);
    }
    free(dev, &[spare]);
    assert_eq!(ffi::ack_close(dev), ffi::ACK_OK, "{}", last_error());
}

#[test]
fn the_norm_backward_is_the_law() {
    let Some(dev) = open_or_unmeasured("student norm backward") else {
        return;
    };
    let spare = alloc(dev, &[0u8; 16]);
    for (rows, d, flags, rows_per_group) in [
        (37_usize, 130_usize, 0, 8_usize),
        (37, 130, DY16 | DH, 8),
        (3, 4096, DY16, 64),
        (70, 2048, DH, 64),
        (37, 130, DY16 | DY_ABSENT, 8),
    ] {
        let n = rows * d;
        let groups = rows.div_ceil(rows_per_group);
        let h: Vec<f32> = (0..n).map(|i| unit(i, 11) * 3.0).collect();
        let absent = flags & DY_ABSENT != 0;
        let dy: Vec<f32> = (0..n).map(|i| unit(i, 13)).collect();
        let dy16: Option<Vec<f32>> =
            (flags & DY16 != 0).then(|| (0..n).map(|i| half(unit(i, 17) * 0.5)).collect());
        let dh: Option<Vec<f32>> =
            (flags & DH != 0).then(|| (0..n).map(|i| unit(i, 19) * 0.25).collect());
        let w: Vec<f32> = (0..d).map(|c| 0.75 + 0.5 * unit(c, 7)).collect();
        let (dx, sx, dw, sw) = reference::norm_backward(
            &h,
            (!absent).then_some(&dy[..]),
            dy16.as_deref(),
            dh.as_deref(),
            &w,
            d,
            EPS,
            rows_per_group,
        );
        let bh = alloc(dev, &f32_bytes(&h));
        let baux = match &dy16 {
            Some(v) => alloc(dev, &f16_bytes(v)),
            None => alloc(dev, &[0u8; 16]),
        };
        let bw = alloc(dev, &f32_bytes(&w));
        // with DY_ABSENT the role is bound to a 16-byte buffer the kernel must not read
        let bdy = alloc(
            dev,
            &if absent {
                vec![0u8; 16]
            } else {
                f32_bytes(&dy)
            },
        );
        let bdh = match &dh {
            Some(v) => alloc(dev, &f32_bytes(v)),
            None => alloc(dev, &[0u8; 16]),
        };
        let (out0, out1) = (poisoned(dev, n * 4), poisoned(dev, groups * d * 4));
        let rc = call(
            dev,
            [bh, baux, bw, spare, out0, out1, bdy, bdh],
            NormOp::NormBackward as u32,
            flags,
            rows,
            d,
            EPS,
            1,
            1,
            rows_per_group,
        );
        let what = format!("{rows} rows of {d}, flags {flags}, {rows_per_group} rows a group");
        assert_eq!(rc, ffi::ACK_OK, "{what}: {}", last_error());
        let got_dx = widen(&f32s(&download(dev, out0, n * 4)));
        let got_dw = widen(&f32s(&download(dev, out1, groups * d * 4)));
        let (bad, first) = beyond(&got_dx, &dx, |i| 1e-5 * sx[i] + 1e-30);
        assert_eq!(
            bad, 0,
            "{what}: dx beyond 1e-5 of its terms; first {first:?}"
        );
        let (bad, first) = beyond(&got_dw, &dw, |i| 1e-5 * sw[i] + 1e-30);
        assert_eq!(
            bad, 0,
            "{what}: the dw partials beyond 1e-5 of their terms; first {first:?}"
        );
        free(dev, &[bh, baux, bw, bdy, bdh, out0, out1]);
    }
    free(dev, &[spare]);
    assert_eq!(ffi::ack_close(dev), ffi::ACK_OK, "{}", last_error());
}

#[test]
fn the_qk_norm_with_rope_and_its_backward_are_the_law() {
    let Some(dev) = open_or_unmeasured("student q/k norm with rope") else {
        return;
    };
    let spare = alloc(dev, &[0u8; 16]);
    let rows_per_group: usize = 4;
    for (b, t, heads, d, same_layout) in [
        (2_usize, 5_usize, 3_usize, 64_usize, false),
        (2, 5, 3, 64, true),
        (2, 16, 4, 128, false),
        (1, 7, 2, 256, false),
        (1, 2100, 32, 2, false),
    ] {
        let rows = b * t * heads;
        let n = rows * d;
        let groups = rows.div_ceil(rows_per_group);
        let flags = if same_layout { SAME_LAYOUT } else { 0 };
        let x: Vec<f32> = (0..n).map(|i| half(unit(i, 23) * 4.0)).collect();
        let w: Vec<f32> = (0..d).map(|c| 0.75 + 0.5 * unit(c, 29)).collect();
        let (cos, sin) = rope_tables(t, d);
        let tab: Vec<f32> = cos.iter().chain(&sin).copied().collect();
        let (bx, bw, btab) = (
            alloc(dev, &f16_bytes(&x)),
            alloc(dev, &f32_bytes(&w)),
            alloc(dev, &f32_bytes(&tab)),
        );
        let what = format!("({b}, {t}, {heads}, {d}), same layout {same_layout}");

        let (out, mag) = reference::qk_norm_rope(&x, &w, &cos, &sin, t, heads, d, EPS, same_layout);
        let out0 = poisoned(dev, n * 2);
        let rc = call(
            dev,
            [bx, spare, bw, btab, out0, spare, spare, spare],
            NormOp::QkNormRope as u32,
            flags,
            rows,
            d,
            EPS,
            t,
            heads,
            1,
        );
        assert_eq!(rc, ffi::ACK_OK, "{what}: {}", last_error());
        let got = widen16(&u16s(&download(dev, out0, n * 2)));
        let (bad, first) = beyond(&got, &out, |i| {
            2f64.powi(-9) * mag[i] + 2f64.powi(-10) * out[i].abs() + 2f64.powi(-24)
        });
        assert_eq!(
            bad, 0,
            "{what}: the rotated norm beyond its tolerance; first {first:?}"
        );

        let dout: Vec<f32> = (0..n).map(|i| half(unit(i, 31) * 0.5)).collect();
        let (dx, sx, dw, sw) = reference::qk_norm_rope_backward(
            &x,
            &dout,
            &w,
            &cos,
            &sin,
            t,
            heads,
            d,
            EPS,
            same_layout,
            rows_per_group,
        );
        let bdout = alloc(dev, &f16_bytes(&dout));
        let (gx, gw) = (poisoned(dev, n * 2), poisoned(dev, groups * d * 4));
        let rc = call(
            dev,
            [bx, bdout, bw, btab, gx, gw, spare, spare],
            NormOp::QkNormRopeBackward as u32,
            flags,
            rows,
            d,
            EPS,
            t,
            heads,
            rows_per_group,
        );
        assert_eq!(rc, ffi::ACK_OK, "{what} backward: {}", last_error());
        let got_dx = widen16(&u16s(&download(dev, gx, n * 2)));
        let got_dw = widen(&f32s(&download(dev, gw, groups * d * 4)));
        let (bad, first) = beyond(&got_dx, &dx, |i| {
            2f64.powi(-9) * sx[i] + 2f64.powi(-10) * dx[i].abs() + 2f64.powi(-24)
        });
        assert_eq!(bad, 0, "{what}: dx beyond its tolerance; first {first:?}");
        let (bad, first) = beyond(&got_dw, &dw, |i| 2f64.powi(-9) * sw[i] + 1e-30);
        assert_eq!(
            bad, 0,
            "{what}: the dw partials beyond 2^-9 of their terms; first {first:?}"
        );
        free(dev, &[bx, bw, btab, out0, bdout, gx, gw]);
    }
    free(dev, &[spare]);
    assert_eq!(ffi::ack_close(dev), ffi::ACK_OK, "{}", last_error());
}

#[test]
fn what_the_entry_does_not_address_is_refused_by_name() {
    let Some(dev) = open_or_unmeasured("student norm refusals") else {
        return;
    };
    let (rows, d) = (4, 16);
    let x = alloc(dev, &f32_bytes(&vec![1.0; rows * d]));
    let w = alloc(dev, &f32_bytes(&vec![1.0; d]));
    let (out0, out1, small) = (
        poisoned(dev, rows * d * 4),
        poisoned(dev, rows * d * 4),
        alloc(dev, &[0u8; 16]),
    );
    let norm = NormOp::Norm as u32;
    let refused = |rc: i32, code: i32, name: &str| {
        assert_eq!(rc, code, "{name}: {}", last_error());
        assert!(
            last_error().ends_with(name),
            "{:?} names {name}",
            last_error()
        );
    };
    let all = |o0: Buf, o1: Buf| [x, x, w, x, o0, o1, x, x];
    refused(
        call(dev, all(out0, out1), norm, Y32, rows, 8192, EPS, 1, 1, 1),
        ffi::ACK_ERR_SHAPE,
        "student-norm-width-above-limit",
    );
    refused(
        call(
            dev,
            all(out0, out1),
            NormOp::QkNormRope as u32,
            0,
            rows,
            512,
            EPS,
            2,
            2,
            1,
        ),
        ffi::ACK_ERR_SHAPE,
        "student-norm-width-above-limit",
    );
    refused(
        call(dev, all(out0, out1), norm, Y32, rows, 15, EPS, 1, 1, 1),
        ffi::ACK_ERR_SHAPE,
        "student-norm-odd-width",
    );
    refused(
        call(dev, all(out0, out1), norm, 0, rows, d, EPS, 1, 1, 1),
        ffi::ACK_ERR_SHAPE,
        "student-norm-flags-invalid",
    );
    refused(
        call(dev, all(out0, out1), norm, Y32 | DH, rows, d, EPS, 1, 1, 1),
        ffi::ACK_ERR_SHAPE,
        "student-norm-flags-invalid",
    );
    refused(
        call(dev, all(out0, out1), norm, Y32, rows, d, 0.0, 1, 1, 1),
        ffi::ACK_ERR_SHAPE,
        "student-norm-epsilon-invalid",
    );
    refused(
        call(
            dev,
            all(out0, out1),
            NormOp::QkNormRope as u32,
            0,
            6,
            d,
            EPS,
            4,
            1,
            1,
        ),
        ffi::ACK_ERR_SHAPE,
        "student-norm-positions-invalid",
    );
    refused(
        call(
            dev,
            all(out0, out1),
            NormOp::NormBackward as u32,
            0,
            rows,
            d,
            EPS,
            1,
            1,
            0,
        ),
        ffi::ACK_ERR_SHAPE,
        "student-norm-rows-per-group-out-of-range",
    );
    refused(
        call(dev, all(out0, out1), 4, Y32, rows, d, EPS, 1, 1, 1),
        ffi::ACK_ERR_SHAPE,
        "student-norm-operation-out-of-range",
    );
    refused(
        call(dev, all(out0, out1), norm, Y32, 1 << 20, 4096, EPS, 1, 1, 1),
        ffi::ACK_ERR_SIZE,
        "student-norm-element-limit-exceeded",
    );
    refused(
        call(
            dev,
            [small, x, w, x, out0, out1, x, x],
            norm,
            Y32,
            rows,
            d,
            EPS,
            1,
            1,
            1,
        ),
        ffi::ACK_ERR_SIZE,
        "student-norm-x-too-small",
    );
    refused(
        call(dev, all(small, out1), norm, Y32, rows, d, EPS, 1, 1, 1),
        ffi::ACK_ERR_SIZE,
        "student-norm-out0-too-small",
    );
    refused(
        call(dev, all(x, out1), norm, Y32, rows, d, EPS, 1, 1, 1),
        ffi::ACK_ERR_SHAPE,
        "student-norm-output-aliases-input",
    );
    let rc = call(
        dev,
        [x, x, w, x, out0, out0, x, x],
        NormOp::NormBackward as u32,
        0,
        rows,
        d,
        EPS,
        1,
        1,
        64,
    );
    assert_ne!(rc, ffi::ACK_OK, "two outputs on one buffer must be refused");
    // the refusals recorded nothing: a plain call still runs
    assert_eq!(
        call(dev, all(out0, out1), norm, Y32, rows, d, EPS, 1, 1, 1),
        ffi::ACK_OK,
        "{}",
        last_error()
    );
    free(dev, &[x, w, out0, out1, small]);
    assert_eq!(ffi::ack_close(dev), ffi::ACK_OK, "{}", last_error());
}
