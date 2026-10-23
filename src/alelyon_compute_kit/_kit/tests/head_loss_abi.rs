//! `ack_head_lse_f16` / `ack_head_grad_f16` (head-kl-ce-f16/v1, 2026-09-29):
//! the sub-bit student's head loss over an fp16 head output, on a real device.
//!
//! THE LAW is `head_loss_ops::reference_lse` / `reference_grad`, the kernels'
//! arithmetic in f64. The kernels round each logit z = f32(yp) * h exactly as
//! the reference does; they differ from it through exp's own rounding and the
//! order of their sums. So the forward's log-sum-exps are held to 2e-5
//! absolute plus 2e-5 relative, as the loss family's probe is. The backward's
//! fp16 gradients are held to one fp16 rounding of the law plus 2^-12 relative,
//! and its d/dh to 2e-5 of the sum of the magnitudes it adds.
//!
//! The shapes cover:
//! - a row wider than one of the forward's 32,768-column segments, its top-k
//!   ids on both sides of the boundary;
//! - a single pair of columns;
//! - repeated top-k ids and a target among the top-k;
//! - a row whose every column is in its top-k (its second sum is -inf);
//! - a row whose top-k logits stand 100 above the rest, where a second sum
//!   taken against the row's overall maximum would underflow;
//! - a top-k id out of range (its row is NaN).
//!
//! The outputs are poisoned first, so an element the kernels never write cannot
//! pass. Refusals are checked by name. Skips as UNMEASURED without a device
//! unless ACK_REQUIRE_DEVICE is set.
use alelyon_compute_kit::ffi;
use alelyon_compute_kit::head_loss_ops::{self, f16_to_f32, f32_to_f16, HeadGradPlan, HeadLsePlan};
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
    let b = ffi::ack_buffer_alloc(dev, bytes.max(4) as u64);
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

fn u16_bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn i64_bytes(v: &[i64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn hash(i: usize, salt: u32) -> u32 {
    (i as u32).wrapping_mul(2_654_435_761) ^ salt.wrapping_mul(40_503)
}

/// A head output of `rows` x `cols`: logits of a few units, a few rows' worth of
/// peaks, stored at fp16.
fn head_output(rows: usize, cols: usize, salt: u32) -> Vec<u16> {
    (0..rows * cols)
        .map(|i| {
            let h = hash(i, salt);
            let v = ((h % 20_011) as f32 / 10_000.0 - 1.0) * 3.0;
            f32_to_f16(if h % 997 == 0 { v + 8.0 } else { v })
        })
        .collect()
}

fn scale(cols: usize, salt: u32) -> Vec<f32> {
    (0..cols)
        .map(|c| 0.5 + (hash(c, salt) % 1000) as f32 / 1000.0)
        .collect()
}

/// `k` top ids a row, spread over the row (a segment boundary included where the
/// row is wide), with one repeat in every row of at least two ids.
fn top_ids(rows: usize, cols: usize, k: usize, salt: u32) -> Vec<i64> {
    let mut out = Vec::with_capacity(rows * k);
    for r in 0..rows {
        for i in 0..k {
            let id = if i == k - 1 && k >= 2 {
                out[r * k] // a repeat of the row's first id
            } else if i == 1 && cols > 32_768 {
                32_768 // the forward's first segment boundary
            } else {
                (hash(r * k + i, salt) as usize % cols) as i64
            };
            out.push(id);
        }
    }
    out
}

#[test]
fn a_malformed_request_is_refused_before_any_device_lookup() {
    let dev: *const ffi::AckDevice = std::ptr::null();
    let none: *const alelyon_compute_kit::Buffer = std::ptr::null();
    let lse = |rows: u32, cols: u32, k: u32| unsafe {
        ffi::ack_head_lse_f16(
            dev,
            none,
            none,
            none,
            none,
            rows,
            cols,
            k,
            std::ptr::null_mut(),
        )
    };
    let grad = |rows: u32, cols: u32, k: u32, rpg: u32| unsafe {
        ffi::ack_head_grad_f16(
            dev,
            none,
            none,
            none,
            none,
            none,
            none,
            none,
            none,
            rows,
            cols,
            k,
            rpg,
            std::ptr::null_mut(),
        )
    };
    // Each case is called INSIDE the loop: an array of results would make every
    // call first and leave ack_last_error holding only the last message.
    for (rows, cols, k, rpg, name) in [
        (0u32, 16u32, 4u32, None, "head-loss-dimension-out-of-range"),
        (4, 15, 4, None, "head-loss-odd-columns"),
        (4, 16, 300, None, "head-loss-top-k-above-limit"),
        (
            4,
            16,
            4,
            Some(0u32),
            "head-loss-rows-per-group-out-of-range",
        ),
        (4, 16, 4, Some(65), "head-loss-rows-per-group-out-of-range"),
    ] {
        let rc = match rpg {
            None => lse(rows, cols, k),
            Some(rpg) => grad(rows, cols, k, rpg),
        };
        assert_eq!(rc, ffi::ACK_ERR_SHAPE, "{name}");
        assert!(
            last_error().ends_with(name),
            "{:?} names {name}",
            last_error()
        );
    }
    // a well-formed request with a null device goes on to the device lookup and fails there
    assert_eq!(lse(4, 16, 4), ffi::ACK_ERR_NULL);
}

struct Case {
    rows: usize,
    cols: usize,
    k: usize,
    rows_per_group: u32,
}

#[test]
fn the_forward_and_the_backward_hold_the_law() {
    let Some(dev) = open_or_unmeasured("head loss") else {
        return;
    };
    let cases = [
        Case {
            rows: 5,
            cols: 70_000,
            k: 32,
            rows_per_group: 2,
        },
        Case {
            rows: 130,
            cols: 1_000,
            k: 8,
            rows_per_group: 64,
        },
        Case {
            rows: 3,
            cols: 2,
            k: 1,
            rows_per_group: 1,
        },
        Case {
            rows: 7,
            cols: 514,
            k: 4,
            rows_per_group: 3,
        },
        Case {
            rows: 2,
            cols: 2,
            k: 2,
            rows_per_group: 1,
        },
    ];
    for (ci, case) in cases.iter().enumerate() {
        let (rows, cols, k) = (case.rows, case.cols, case.k);
        let salt = 17 + ci as u32 * 101;
        let mut yp = head_output(rows, cols, salt);
        let h = scale(cols, salt);
        let mut top = top_ids(rows, cols, k, salt);
        if rows >= 3 && k >= 2 {
            top[2 * k] = cols as i64; // row 2: an id out of range
        }
        if cols == 2 && k == 2 {
            top = vec![0, 1, 1, 0]; // every column of each row in its top-k
        }
        if rows >= 2 && cols > 2 {
            // row 1: its top-k logits 100 above the rest, so a sum over the rest taken
            // against the row's overall maximum would underflow to zero
            for i in 0..k {
                let id = top[k + i];
                if id >= 0 && (id as usize) < cols {
                    yp[cols + id as usize] = f32_to_f16(100.0);
                }
            }
        }
        let targets: Vec<i64> = (0..rows)
            .map(|r| {
                if r % 2 == 0 {
                    top[r * k]
                } else {
                    (hash(r, salt) as usize % cols) as i64
                }
            })
            .collect();
        let plan = HeadLsePlan::new(rows as u32, cols as u32, k as u32).unwrap();
        let want = head_loss_ops::reference_lse(&yp, &h, &top, &plan);

        let b_yp = alloc(dev, yp.len() * 2);
        let b_h = alloc(dev, h.len() * 4);
        let b_top = alloc(dev, top.len() * 8);
        let b_out = alloc(dev, 2 * rows * 4);
        upload(dev, b_yp, &u16_bytes(&yp));
        upload(dev, b_h, &f32_bytes(&h));
        upload(dev, b_top, &i64_bytes(&top));
        upload(dev, b_out, &vec![0xA5u8; 2 * rows * 4]);
        let rc = unsafe {
            ffi::ack_head_lse_f16(
                dev,
                b_yp,
                b_h,
                b_top,
                b_out,
                rows as u32,
                cols as u32,
                k as u32,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, ffi::ACK_OK, "case {ci}: {}", last_error());
        let got = f32s(&download(dev, b_out, 2 * rows * 4));
        for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
            let ok = if w.is_nan() {
                g.is_nan()
            } else if w.is_infinite() {
                f64::from(g) == w
            } else {
                (f64::from(g) - w).abs() <= 2e-5 + 2e-5 * w.abs()
            };
            assert!(ok, "case {ci} out[{i}]: device {g} against the law {w}");
        }

        // the backward, from the forward's own log-sum-exps and a teacher's coefficients
        let back =
            HeadGradPlan::new(rows as u32, cols as u32, k as u32, case.rows_per_group).unwrap();
        let mut row_values = vec![0.0f32; 3 * rows + 1];
        for r in 0..rows {
            // a row the forward made NaN (its out-of-range id) takes its own log-sum-exp over
            // every column, so its gradient stays in fp16's range
            row_values[r] = if got[r].is_finite() {
                got[r]
            } else {
                let z = (0..cols).map(|c| f64::from(f16_to_f32(yp[r * cols + c]) * h[c]));
                let big = z.clone().fold(f64::NEG_INFINITY, f64::max);
                (big + z.map(|v| (v - big).exp()).sum::<f64>().ln()) as f32
            };
            row_values[rows + r] = 0.75 + (hash(r, salt + 3) % 100) as f32 / 400.0;
            row_values[2 * rows + r] = 0.5 + (hash(r, salt + 5) % 100) as f32 / 400.0;
        }
        row_values[3 * rows] = 0.5;
        let gkp: Vec<f32> = (0..rows * k)
            .map(|i| (hash(i, salt + 7) % 1000) as f32 / 30_000.0)
            .collect();
        let (want_grad, want_dh) =
            head_loss_ops::reference_grad(&yp, &h, &top, &targets, &row_values, &gkp, &back);
        let groups = back.groups() as usize;
        let b_tgt = alloc(dev, rows * 8);
        let b_rv = alloc(dev, row_values.len() * 4);
        let b_gkp = alloc(dev, gkp.len() * 4);
        let b_grad = alloc(dev, rows * cols * 2);
        let b_part = alloc(dev, groups * cols * 4);
        upload(dev, b_tgt, &i64_bytes(&targets));
        upload(dev, b_rv, &f32_bytes(&row_values));
        upload(dev, b_gkp, &f32_bytes(&gkp));
        upload(dev, b_grad, &vec![0xA5u8; rows * cols * 2]);
        upload(dev, b_part, &vec![0xA5u8; groups * cols * 4]);
        let rc = unsafe {
            ffi::ack_head_grad_f16(
                dev,
                b_yp,
                b_h,
                b_top,
                b_tgt,
                b_rv,
                b_gkp,
                b_grad,
                b_part,
                rows as u32,
                cols as u32,
                k as u32,
                case.rows_per_group,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, ffi::ACK_OK, "case {ci} backward: {}", last_error());
        let grad: Vec<u16> = download(dev, b_grad, rows * cols * 2)
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect();
        for (i, (&g, &w)) in grad.iter().zip(&want_grad).enumerate() {
            let dev_value = f64::from(f16_to_f32(g));
            // a law past fp16's range rounds to an infinity of its sign
            let w16 = f64::from(f16_to_f32(f32_to_f16(w as f32)));
            // otherwise one fp16 rounding of the law (half an ulp, 2^-11 relative), 2^-12
            // relative for exp's rounding carried through, and 2^-20 absolute for the f32
            // rounding of the terms where a top-k target's three terms cancel
            let tol = w.abs() * (2f64.powi(-11) + 2f64.powi(-12)) + 2f64.powi(-20);
            assert!(
                if w16.is_infinite() {
                    dev_value == w16
                } else {
                    (dev_value - w).abs() <= tol
                },
                "case {ci} grad[{i}] (row {}, column {}): device {dev_value} against the law {w}",
                i / cols,
                i % cols
            );
        }
        let part = f32s(&download(dev, b_part, groups * cols * 4));
        for c in 0..cols {
            let dev_dh: f64 = (0..groups).map(|g| f64::from(part[g * cols + c])).sum();
            let magnitude: f64 = (0..rows)
                .map(|r| {
                    let y = f64::from(f16_to_f32(yp[r * cols + c]));
                    let hc = f64::from(h[c]);
                    (y * want_grad[r * cols + c] / hc).abs()
                })
                .sum();
            let y_sum: f64 = (0..rows)
                .map(|r| f64::from(f16_to_f32(yp[r * cols + c])).abs())
                .sum();
            assert!(
                (dev_dh - want_dh[c]).abs() <= 2e-5 * magnitude + 2f64.powi(-18) * y_sum,
                "case {ci} d/dh[{c}]: device {dev_dh} against the law {}",
                want_dh[c]
            );
        }
        for buf in [b_yp, b_h, b_top, b_out, b_tgt, b_rv, b_gkp, b_grad, b_part] {
            assert_eq!(ffi::ack_buffer_free(dev, buf), ffi::ACK_OK);
        }
    }
    assert_eq!(ffi::ack_close(dev), ffi::ACK_OK, "{}", last_error());
}

fn refused(what: &str, rc: i32, code: i32, name: &str) {
    assert_eq!(rc, code, "{what}: code");
    let e = last_error();
    assert!(e.contains(name), "{what}: {e:?} names {name:?}");
}

#[test]
fn what_the_entries_do_not_address_is_refused_by_name() {
    let Some(dev) = open_or_unmeasured("head loss refusals") else {
        return;
    };
    let (rows, cols, k) = (4u32, 64u32, 2u32);
    let yp = alloc(dev, (rows * cols * 2) as usize);
    let h = alloc(dev, (cols * 4) as usize);
    let top = alloc(dev, (rows * k * 8) as usize);
    let out = alloc(dev, (2 * rows * 4) as usize);
    let small = alloc(dev, 4);
    upload(dev, top, &i64_bytes(&[0, 1, 2, 3, 4, 5, 6, 7]));
    let lse = |y, s, t, o| unsafe {
        ffi::ack_head_lse_f16(dev, y, s, t, o, rows, cols, k, std::ptr::null_mut())
    };
    refused(
        "short yp",
        lse(small, h, top, out),
        ffi::ACK_ERR_SIZE,
        "head-loss-yp-too-small",
    );
    refused(
        "short scale",
        lse(yp, small, top, out),
        ffi::ACK_ERR_SIZE,
        "head-loss-scale-too-small",
    );
    refused(
        "short top",
        lse(yp, h, small, out),
        ffi::ACK_ERR_SIZE,
        "head-loss-top-too-small",
    );
    refused(
        "short output",
        lse(yp, h, top, small),
        ffi::ACK_ERR_SIZE,
        "head-loss-output-too-small",
    );
    refused(
        "output is yp",
        lse(yp, h, top, yp),
        ffi::ACK_ERR_SHAPE,
        "head-loss-output-aliases-input",
    );
    // the refusals recorded nothing: a plain call still runs
    assert_eq!(lse(yp, h, top, out), ffi::ACK_OK, "{}", last_error());
    let tgt = alloc(dev, (rows * 8) as usize);
    let rv = alloc(dev, ((3 * rows + 1) * 4) as usize);
    let gkp = alloc(dev, (rows * k * 4) as usize);
    let grad = alloc(dev, (rows * cols * 2) as usize);
    let part = alloc(dev, (rows * cols * 4) as usize);
    let back = |g, p| unsafe {
        ffi::ack_head_grad_f16(
            dev,
            yp,
            h,
            top,
            tgt,
            rv,
            gkp,
            g,
            p,
            rows,
            cols,
            k,
            1,
            std::ptr::null_mut(),
        )
    };
    refused(
        "short partial",
        back(grad, small),
        ffi::ACK_ERR_SIZE,
        "head-loss-partial-too-small",
    );
    refused(
        "grad is partial",
        back(part, part),
        ffi::ACK_ERR_SHAPE,
        "head-loss-output-aliases-input",
    );
    refused(
        "grad is yp",
        back(yp, part),
        ffi::ACK_ERR_SHAPE,
        "head-loss-output-aliases-input",
    );
    assert_eq!(back(grad, part), ffi::ACK_OK, "{}", last_error());
    for buf in [yp, h, top, out, small, tgt, rv, gkp, grad, part] {
        assert_eq!(ffi::ack_buffer_free(dev, buf), ffi::ACK_OK);
    }
    assert_eq!(ffi::ack_close(dev), ffi::ACK_OK, "{}", last_error());
}
