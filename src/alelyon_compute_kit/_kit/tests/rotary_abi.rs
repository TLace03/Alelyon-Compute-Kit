//! `ack_rotary` (rotary-f32/v1, 2026-09-29): the rotate-half rotary position
//! embedding and its backward, on a real device.
//!
//! THE LAW IS BIT IDENTITY with `rotary_ops::reference`, the host statement of
//! the kernel's arithmetic: each product rounded, then their sum, in the order
//! torch's decomposition `x * cos + cat(-x[h:], x[:h]) * sin` and its autograd
//! backward use. Rust never contracts an f32 multiply and add, so the host
//! reference is that order exactly, and the device must reproduce every bit,
//! signed zeros included (the inputs carry exact zeros of both signs). The
//! shapes cover the two layouts a model hands over:
//! - the transposed view of a dense (B, T, H, D) projection, row position
//!   `(r / H) % T` (the A3r-X2 queries, 4 x 1024 x 8 x 16);
//! - a (B, H, T, D) tensor stored in that order, position `r % T`.
//!
//! They also cover a single pair per row, an odd row count, and one row. The
//! output is poisoned first, so an element the kernel never writes cannot pass.
//! Refusals are checked by name.
//!
//! Skips as UNMEASURED without a device unless ACK_REQUIRE_DEVICE is set.
use alelyon_compute_kit::ffi;
use alelyon_compute_kit::rotary_ops::{self, RotaryOp, RotaryPlan};
use std::ffi::{c_char, CStr};

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

fn alloc(dev: *mut ffi::AckDevice, bytes: usize) -> *mut alelyon_compute_kit::Buffer {
    let b = ffi::ack_buffer_alloc(dev, bytes as u64);
    assert!(!b.is_null(), "{}", last_error());
    b
}

fn upload(dev: *mut ffi::AckDevice, buf: *const alelyon_compute_kit::Buffer, bytes: &[u8]) {
    let rc = unsafe { ffi::ack_upload(dev, buf, bytes.as_ptr(), bytes.len()) };
    assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
}

fn download(
    dev: *mut ffi::AckDevice,
    buf: *const alelyon_compute_kit::Buffer,
    bytes: usize,
) -> Vec<u8> {
    let mut out = vec![0u8; bytes];
    let rc = unsafe { ffi::ack_download(dev, buf, out.as_mut_ptr(), bytes) };
    assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
    out
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// A spread of magnitudes, with exact zeros of both signs every so often.
fn values(n: usize, salt: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let h = (i as u32).wrapping_mul(2_654_435_761) ^ salt;
            match h % 97 {
                0 => 0.0,
                1 => -0.0,
                _ => {
                    let mag = [1.0f32, 0.25, 8.0, 1.0 / 128.0][(h >> 29) as usize % 4];
                    ((h % 20_011) as f32 / 10_000.0 - 1.0) * mag
                }
            }
        })
        .collect()
}

/// cos and sin tables of `positions` rows of `cols`, as a model builds them:
/// angle p * theta^(-2i/cols), the two halves of a row equal.
fn tables(positions: usize, cols: usize) -> (Vec<f32>, Vec<f32>) {
    let half = cols / 2;
    let mut cos = vec![0.0f32; positions * cols];
    let mut sin = vec![0.0f32; positions * cols];
    for p in 0..positions {
        for i in 0..half {
            let inv = 10_000f64.powf(-2.0 * i as f64 / cols as f64);
            let angle = (p as f64 * inv) as f32;
            for j in [i, i + half] {
                cos[p * cols + j] = angle.cos();
                sin[p * cols + j] = angle.sin();
            }
        }
    }
    (cos, sin)
}

#[test]
fn the_forward_and_the_backward_are_the_law_bit_for_bit() {
    let Some(dev) = open_or_unmeasured("rotary") else {
        return;
    };
    // (rows, cols, pos_div, positions)
    let shapes = [
        (4 * 1024 * 8, 16u32, 8u32, 1024u32),
        (2 * 4 * 512, 64, 1, 512),
        (130, 2, 13, 5),
        (7, 6, 1, 7),
        (1, 128, 1, 1),
    ];
    let mut checked = 0usize;
    for (si, &(rows, cols, pos_div, positions)) in shapes.iter().enumerate() {
        let (cos, sin) = tables(positions as usize, cols as usize);
        let c = alloc(dev, cos.len() * 4);
        let s = alloc(dev, sin.len() * 4);
        upload(dev, c, &f32_bytes(&cos));
        upload(dev, s, &f32_bytes(&sin));
        for op in [RotaryOp::Forward, RotaryOp::Backward] {
            let plan = RotaryPlan::new(rows, cols, op, pos_div, positions).unwrap();
            let x = values(plan.elements(), 31 + si as u32 * 7 + op as u32);
            let want = rotary_ops::reference(&x, &cos, &sin, &plan).unwrap();
            let xb = alloc(dev, x.len() * 4);
            let ob = alloc(dev, x.len() * 4);
            upload(dev, xb, &f32_bytes(&x));
            upload(dev, ob, &vec![0xA5u8; x.len() * 4]);
            let rc = unsafe {
                ffi::ack_rotary(
                    dev,
                    xb,
                    c,
                    s,
                    ob,
                    rows,
                    cols,
                    op as u32,
                    pos_div,
                    positions,
                    std::ptr::null_mut(),
                )
            };
            assert_eq!(rc, ffi::ACK_OK, "{rows}x{cols} {op:?}: {}", last_error());
            let got: Vec<u32> = download(dev, ob, x.len() * 4)
                .chunks_exact(4)
                .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            let bad = want
                .iter()
                .zip(&got)
                .filter(|(w, g)| w.to_bits() != **g)
                .count();
            assert_eq!(
                bad,
                0,
                "{rows}x{cols} (div {pos_div}, {positions} positions) {op:?}: {bad} of {} elements differ from the law (first: {:?})",
                x.len(),
                want.iter().zip(&got).position(|(w, g)| w.to_bits() != *g)
            );
            checked += x.len();
            for buf in [xb, ob] {
                assert_eq!(ffi::ack_buffer_free(dev, buf), ffi::ACK_OK);
            }
        }
        for buf in [c, s] {
            assert_eq!(ffi::ack_buffer_free(dev, buf), ffi::ACK_OK);
        }
    }
    eprintln!(
        "{checked} elements bit-identical across {} shapes x 2 operations",
        shapes.len()
    );
    assert_eq!(ffi::ack_close(dev), ffi::ACK_OK, "{}", last_error());
}

fn refused(what: &str, rc: i32, code: i32, name: &str) {
    assert_eq!(rc, code, "{what}: code");
    let e = last_error();
    assert!(e.contains(name), "{what}: {e:?} names {name:?}");
}

#[test]
fn what_the_entry_does_not_address_is_refused_by_name() {
    let Some(dev) = open_or_unmeasured("rotary refusals") else {
        return;
    };
    let x = alloc(dev, 64 * 16 * 4);
    let t = alloc(dev, 64 * 16 * 4);
    let o = alloc(dev, 64 * 16 * 4);
    let small = alloc(dev, 16);
    let call = |x, c, s, o, rows: u32, cols: u32, op: u32| unsafe {
        ffi::ack_rotary(dev, x, c, s, o, rows, cols, op, 1, 64, std::ptr::null_mut())
    };
    refused(
        "odd cols",
        call(x, t, t, o, 64, 15, 0),
        ffi::ACK_ERR_SHAPE,
        "rotary-odd-columns",
    );
    refused(
        "zero rows",
        call(x, t, t, o, 0, 16, 0),
        ffi::ACK_ERR_SHAPE,
        "rotary-dimension-out-of-range",
    );
    refused(
        "op 2",
        call(x, t, t, o, 64, 16, 2),
        ffi::ACK_ERR_SHAPE,
        "rotary-operation-out-of-range",
    );
    refused(
        "short output",
        call(x, t, t, small, 64, 16, 0),
        ffi::ACK_ERR_SIZE,
        "rotary-output-too-small",
    );
    refused(
        "short cos",
        call(x, small, t, o, 64, 16, 0),
        ffi::ACK_ERR_SIZE,
        "rotary-cos-too-small",
    );
    refused(
        "output is x",
        call(x, t, t, x, 64, 16, 1),
        ffi::ACK_ERR_SHAPE,
        "rotary-output-aliases-input",
    );
    // the refusals recorded nothing: a plain call still runs
    assert_eq!(call(x, t, t, o, 64, 16, 0), ffi::ACK_OK, "{}", last_error());
    for buf in [x, t, o, small] {
        assert_eq!(ffi::ack_buffer_free(dev, buf), ffi::ACK_OK);
    }
    assert_eq!(ffi::ack_close(dev), ffi::ACK_OK, "{}", last_error());
}
