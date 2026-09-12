//! indexed-cross-entropy-f32/v2 at the C ABI (`ack_loss`, ABI 9): the entry
//! validates the caller's host targets through `loss_ops::LossPlan`, uploads
//! the plan's own canonical i32 image into the targets buffer, and dispatches.
//! The host-only checks need no device; the operations need one and print
//! UNMEASURED without it unless ACK_REQUIRE_DEVICE is set. Fixed-fixture
//! evidence only.
use alelyon_compute_kit::ffi;
use alelyon_compute_kit::loss_ops::{LossOp, LossReduction};
use std::ffi::{c_char, CStr};

/// The loss probe's declared tolerances (src/bin/loss_probe.rs).
const ABS_TOL: f64 = 2.0e-5;
const REL_TOL: f64 = 2.0e-5;
/// A byte pattern no canonical target image contains, so a byte still holding
/// it was not written by the upload.
const POISON: i32 = 0x5A5A_5A5Au32 as i32;

fn last_error() -> String {
    let mut buf = vec![0 as c_char; 1024];
    let rc = unsafe { ffi::ack_last_error(buf.as_mut_ptr(), buf.len()) };
    assert!(rc >= 0, "ack_last_error returned {rc}");
    unsafe { CStr::from_ptr(buf.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

fn within(got: f32, want: f64) -> bool {
    (f64::from(got) - want).abs() <= ABS_TOL + REL_TOL * want.abs()
}

/// Row-wise log-softmax in f64: the log probabilities the NLL operations take.
fn log_softmax(x: &[f32], cols: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(x.len());
    for row in x.chunks_exact(cols) {
        let largest = row
            .iter()
            .fold(f64::NEG_INFINITY, |m, &v| m.max(f64::from(v)));
        let total: f64 = row.iter().map(|&v| (f64::from(v) - largest).exp()).sum();
        out.extend(
            row.iter()
                .map(|&v| ((f64::from(v) - largest) - total.ln()) as f32),
        );
    }
    out
}

#[test]
fn a_malformed_request_is_refused_before_any_device_lookup() {
    let dev = std::ptr::null();
    let none = std::ptr::null();
    let targets = [0i64, 1];
    let call = |rows: u32, cols: u32, op: u32, red: u32, ignore: i64, t: &[i64]| unsafe {
        ffi::ack_loss(
            dev,
            none,
            none,
            none,
            none,
            rows,
            cols,
            op,
            red,
            ignore,
            t.as_ptr(),
            t.len(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    };
    // a null target pointer with a nonzero length is refused first of all
    assert_eq!(
        unsafe {
            ffi::ack_loss(
                dev,
                none,
                none,
                none,
                none,
                2,
                4,
                3,
                0,
                -100,
                std::ptr::null(),
                2,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        },
        ffi::ACK_ERR_NULL
    );
    // Each case is called INSIDE the loop: an array literal would evaluate every
    // call first and leave ack_last_error holding only the last message.
    for (rows, cols, op, red, ignore, t, name) in [
        (
            2u32,
            4u32,
            5u32,
            0u32,
            -100i64,
            &targets[..],
            "loss-operation-out-of-range",
        ),
        (
            2,
            4,
            3,
            3,
            -100,
            &targets[..],
            "loss-reduction-out-of-range",
        ),
        (
            0,
            4,
            3,
            0,
            -100,
            &targets[..],
            "loss-dimension-out-of-range",
        ),
        (
            2,
            4,
            2,
            0,
            -100,
            &targets[..],
            "loss-invalid-reduction-for-operation",
        ),
        (3, 4, 3, 0, -100, &targets[..], "loss-target-shape-mismatch"),
        (2, 4, 3, 0, -100, &[0i64, 9][..], "loss-target-out-of-range"),
        (
            2,
            4,
            3,
            2,
            0,
            &[0i64, 0][..],
            "loss-mean-has-no-valid-targets",
        ),
    ] {
        assert_eq!(
            call(rows, cols, op, red, ignore, t),
            ffi::ACK_ERR_SHAPE,
            "{name}"
        );
        assert!(last_error().contains(name), "{name}: {}", last_error());
    }
    // with a well-formed request the null device handle is what refuses next
    assert_eq!(call(2, 4, 3, 0, -100, &targets), ffi::ACK_ERR_NULL);
}

#[test]
fn the_nll_operations_match_a_host_reference_and_the_upload_is_reported_exactly() {
    let dev = ffi::ack_open();
    if dev.is_null() {
        if std::env::var_os("ACK_REQUIRE_DEVICE").is_some() {
            panic!(
                "loss ABI: ACK_REQUIRE_DEVICE is set and ack_open failed: {}",
                last_error()
            );
        }
        eprintln!("UNMEASURED here (loss ABI): {}", last_error());
        return;
    }
    let rows = 6usize;
    let cols = 9usize;
    let n = rows * cols;
    let logits: Vec<f32> = (0..n)
        .map(|i| ((i * 7919) % 61) as f32 / 7.0 - 4.0)
        .collect();
    let lp = log_softmax(&logits, cols);
    // one row ignored, to exercise the -1 canonical target and the mean's
    // denominator being the count of the rest
    let targets: Vec<i64> = vec![3, 0, 8, -100, 5, 1];
    let upstream: Vec<f32> = (0..rows).map(|i| 0.25 + i as f32 * 0.125).collect();

    let alloc = |elements: usize| {
        let b = ffi::ack_buffer_alloc(dev, (elements * 4) as u64);
        assert!(!b.is_null(), "{}", last_error());
        b
    };
    let upload_f32 = |buf, data: &[f32]| {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let rc = unsafe { ffi::ack_upload(dev, buf, bytes.as_ptr(), bytes.len()) };
        assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
    };
    let upload_i32 = |buf, data: &[i32]| {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let rc = unsafe { ffi::ack_upload(dev, buf, bytes.as_ptr(), bytes.len()) };
        assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
    };
    let download_f32 = |buf, elements: usize| {
        let mut raw = vec![0u8; elements * 4];
        let rc = unsafe { ffi::ack_download(dev, buf, raw.as_mut_ptr(), raw.len()) };
        assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
        raw.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect::<Vec<f32>>()
    };
    let download_i32 = |buf, elements: usize| {
        let mut raw = vec![0u8; elements * 4];
        let rc = unsafe { ffi::ack_download(dev, buf, raw.as_mut_ptr(), raw.len()) };
        assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
        raw.chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect::<Vec<i32>>()
    };

    let b_lp = alloc(n);
    let b_up = alloc(rows);
    let b_out_rows = alloc(rows);
    let b_out_full = alloc(n);
    // Exactly the plan's image, and poisoned before the call. The kit uploads a
    // WHOLE buffer, so this slot cannot be over-allocated; that is refused by
    // name below. Poisoning it is still what makes the reported byte count a
    // measurement rather than a restatement: had the entry written fewer bytes
    // than it reported, a poisoned word would have survived.
    let targets_capacity = rows;
    let b_targets = alloc(targets_capacity);
    upload_f32(b_lp, &lp);
    upload_f32(b_up, &upstream);
    upload_i32(b_targets, &vec![POISON; targets_capacity]);

    let mut ms = 0f64;
    let mut uploaded = 0u64;
    let rc = unsafe {
        ffi::ack_loss(
            dev,
            b_lp,
            b_targets,
            b_up,
            b_out_rows,
            rows as u32,
            cols as u32,
            LossOp::NllForward as u32,
            LossReduction::None as u32,
            -100,
            targets.as_ptr(),
            targets.len(),
            &mut uploaded,
            &mut ms,
        )
    };
    assert_eq!(rc, ffi::ACK_OK, "{}", last_error());

    // THE REPORTED UPLOAD IS A MEASUREMENT, and this is what makes it one: the
    // entry says how many bytes it wrote, and the buffer says which bytes
    // changed. A reported figure that disagreed with the device -- an image
    // that grew, or a padded write -- would leave either a poisoned word inside
    // the reported prefix or an overwritten one beyond it. Asserting the figure
    // against `rows * 4` instead would only restate the arithmetic the caller
    // was told not to trust.
    let image = download_i32(b_targets, targets_capacity);
    assert_eq!(uploaded % 4, 0, "a whole number of i32 targets");
    let written = (uploaded / 4) as usize;
    assert_eq!(written, rows, "the canonical image is one word per row");
    for (i, value) in image.iter().enumerate() {
        assert_ne!(
            *value, POISON,
            "target word {i} still holds the poison, so the reported upload did not write it"
        );
    }
    // and the image is the plan's canonicalisation, with the ignored row at -1
    assert_eq!(image, [3, 0, 8, -1, 5, 1]);

    let losses = download_f32(b_out_rows, rows);
    for (row, value) in losses.iter().enumerate() {
        let want = if targets[row] == -100 {
            0.0
        } else {
            -f64::from(lp[row * cols + targets[row] as usize])
        };
        assert!(within(*value, want), "row {row}: got {value}, want {want}");
        // range, independent of that reference: a negative log-likelihood of a
        // log probability is never negative, because a log probability is <= 0
        assert!(
            value.is_finite() && *value >= 0.0,
            "row {row}: {value} is not a negative log-likelihood"
        );
    }

    // the mean over the NON-ignored rows, through the reduction operation
    let b_one = alloc(1);
    let rc = unsafe {
        ffi::ack_loss(
            dev,
            b_out_rows,
            b_targets,
            b_up,
            b_one,
            rows as u32,
            cols as u32,
            LossOp::Reduce as u32,
            LossReduction::Mean as u32,
            -100,
            targets.as_ptr(),
            targets.len(),
            std::ptr::null_mut(),
            &mut ms,
        )
    };
    assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
    let valid: Vec<f64> = losses
        .iter()
        .enumerate()
        .filter(|(row, _)| targets[*row] != -100)
        .map(|(_, v)| f64::from(*v))
        .collect();
    let want = valid.iter().sum::<f64>() / valid.len() as f64;
    let got = download_f32(b_one, 1)[0];
    assert!(within(got, want), "mean: got {got}, want {want}");

    // the backward: -upstream at the target column, zero elsewhere, and zero
    // for an ignored row. It reads the target and the upstream gradient and
    // never the logits, so the primary slot is bound to a buffer it must not
    // read -- the one-element output of the reduction above.
    let rc = unsafe {
        ffi::ack_loss(
            dev,
            b_one,
            b_targets,
            b_up,
            b_out_full,
            rows as u32,
            cols as u32,
            LossOp::NllBackward as u32,
            LossReduction::None as u32,
            -100,
            targets.as_ptr(),
            targets.len(),
            std::ptr::null_mut(),
            &mut ms,
        )
    };
    assert_eq!(
        rc,
        ffi::ACK_OK,
        "the NLL backward must not require a primary it never reads: {}",
        last_error()
    );
    let grad = download_f32(b_out_full, n);
    for row in 0..rows {
        for col in 0..cols {
            let want = if targets[row] == -100 {
                0.0
            } else if col == targets[row] as usize {
                -f64::from(upstream[row])
            } else {
                0.0
            };
            let got = grad[row * cols + col];
            assert!(
                within(got, want),
                "grad[{row},{col}]: got {got}, want {want}"
            );
        }
    }

    // refusals by name, nothing dispatched
    let before = download_f32(b_out_full, n);
    // a targets buffer that is not exactly the image, in BOTH directions, and
    // by name rather than as the kit's opaque whole-buffer upload error
    for wrong in [alloc(1), alloc(rows + 5)] {
        let rc = unsafe {
            ffi::ack_loss(
                dev,
                b_lp,
                wrong,
                b_up,
                b_out_full,
                rows as u32,
                cols as u32,
                LossOp::NllForward as u32,
                LossReduction::None as u32,
                -100,
                targets.as_ptr(),
                targets.len(),
                std::ptr::null_mut(),
                &mut ms,
            )
        };
        assert_eq!(rc, ffi::ACK_ERR_SIZE, "{}", last_error());
        assert!(
            last_error().contains("loss-targets-size-mismatch"),
            "{}",
            last_error()
        );
        assert_eq!(
            ffi::ack_buffer_free(dev, wrong),
            ffi::ACK_OK,
            "{}",
            last_error()
        );
    }
    // the output through a buffer the operation reads
    let rc = unsafe {
        ffi::ack_loss(
            dev,
            b_lp,
            b_targets,
            b_up,
            b_targets,
            rows as u32,
            cols as u32,
            LossOp::NllForward as u32,
            LossReduction::None as u32,
            -100,
            targets.as_ptr(),
            targets.len(),
            std::ptr::null_mut(),
            &mut ms,
        )
    };
    assert_eq!(rc, ffi::ACK_ERR_SHAPE, "{}", last_error());
    assert!(
        last_error().contains("loss-output-aliases-input"),
        "{}",
        last_error()
    );
    assert_eq!(
        download_f32(b_out_full, n),
        before,
        "a refusal dispatches nothing"
    );

    for buf in [b_lp, b_up, b_out_rows, b_out_full, b_targets, b_one] {
        assert_eq!(
            ffi::ack_buffer_free(dev, buf),
            ffi::ACK_OK,
            "{}",
            last_error()
        );
    }
    assert_eq!(ffi::ack_close(dev), ffi::ACK_OK, "{}", last_error());
}
