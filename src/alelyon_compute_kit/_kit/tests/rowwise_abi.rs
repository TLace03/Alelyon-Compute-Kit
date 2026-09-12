//! rowwise-f32/v2 at the C ABI (`ack_rowwise`, ABI 8): the caller's 16-byte
//! request block (the kernel's own push layout) is re-derived by
//! `row_ops::RowPlan` before any device lookup, the buffers are checked
//! against the plan's lengths and aliasing rules by name, and all seven
//! operations (softmax, log-softmax and RMSNorm, including packed RMSNorm
//! backward and gamma reduction) match a host FP64
//! reference within the row probe's declared tolerances. The host-only checks
//! need no device; the operations need one and print UNMEASURED without it
//! unless ACK_REQUIRE_DEVICE is set. Fixed-fixture evidence only.
use alelyon_compute_kit::ffi;
use alelyon_compute_kit::row_ops::{RowOp, MAX_COLS};
use std::ffi::{c_char, CStr};

/// The row probe's declared tolerances (src/bin/row_probe.rs).
const SOFTMAX_ABS_TOL: f64 = 2.0e-6;
const SOFTMAX_REL_TOL: f64 = 2.0e-5;
const LOG_SOFTMAX_ABS_TOL: f64 = 2.0e-5;
const LOG_SOFTMAX_REL_TOL: f64 = 2.0e-5;
const BACKWARD_ABS_TOL: f64 = 2.0e-5;
const BACKWARD_REL_TOL: f64 = 2.0e-5;
const EPSILON: f32 = 1.0e-5;

fn last_error() -> String {
    let mut buf = vec![0 as c_char; 1024];
    let rc = unsafe { ffi::ack_last_error(buf.as_mut_ptr(), buf.len()) };
    assert!(rc >= 0, "ack_last_error returned {rc}");
    unsafe { CStr::from_ptr(buf.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

fn request(rows: u32, cols: u32, op: RowOp) -> [u8; ffi::ROWWISE_PUSH_BYTES] {
    let mut block = [0u8; ffi::ROWWISE_PUSH_BYTES];
    for (i, w) in [rows, cols, op as u32, EPSILON.to_bits()]
        .iter()
        .enumerate()
    {
        block[4 * i..4 * i + 4].copy_from_slice(&w.to_le_bytes());
    }
    block
}

fn within(got: f32, want: f64, abs_tol: f64, rel_tol: f64) -> bool {
    if want.is_infinite() || got.is_infinite() {
        return f64::from(got) == want;
    }
    (f64::from(got) - want).abs() <= abs_tol + rel_tol * want.abs()
}

fn softmax_rows(x: &[f32], cols: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(x.len());
    for row in x.chunks_exact(cols) {
        let largest = row
            .iter()
            .fold(f64::NEG_INFINITY, |m, &v| m.max(f64::from(v)));
        let total: f64 = row.iter().map(|&v| (f64::from(v) - largest).exp()).sum();
        out.extend(row.iter().map(|&v| (f64::from(v) - largest).exp() / total));
    }
    out
}

fn log_softmax_rows(x: &[f32], cols: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(x.len());
    for row in x.chunks_exact(cols) {
        let largest = row
            .iter()
            .fold(f64::NEG_INFINITY, |m, &v| m.max(f64::from(v)));
        let total: f64 = row.iter().map(|&v| (f64::from(v) - largest).exp()).sum();
        out.extend(row.iter().map(|&v| (f64::from(v) - largest) - total.ln()));
    }
    out
}

fn softmax_backward_rows(p: &[f32], g: &[f32], cols: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(p.len());
    for (prow, grow) in p.chunks_exact(cols).zip(g.chunks_exact(cols)) {
        let dot: f64 = prow
            .iter()
            .zip(grow)
            .map(|(&a, &b)| f64::from(a) * f64::from(b))
            .sum();
        out.extend(
            prow.iter()
                .zip(grow)
                .map(|(&a, &b)| f64::from(a) * (f64::from(b) - dot)),
        );
    }
    out
}

fn log_softmax_backward_rows(lp: &[f32], g: &[f32], cols: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(lp.len());
    for (lrow, grow) in lp.chunks_exact(cols).zip(g.chunks_exact(cols)) {
        let total: f64 = grow.iter().map(|&b| f64::from(b)).sum();
        out.extend(
            lrow.iter()
                .zip(grow)
                .map(|(&a, &b)| f64::from(b) - f64::from(a).exp() * total),
        );
    }
    out
}

/// Direct FP64 formula, without the shader's scaling or reduction tree. The
/// square of every finite FP32 input is representable in FP64.
fn rmsnorm_reference(
    x: &[f32],
    g: &[f32],
    gamma: &[f32],
    epsilon: f32,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let cols = gamma.len();
    let mut forward = vec![0.0; x.len()];
    let mut packed = vec![0.0; 2 * x.len()];
    let mut dgamma = vec![0.0; cols];
    for (r, row) in x.chunks_exact(cols).enumerate() {
        let rms = (row.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / cols as f64
            + f64::from(epsilon))
        .sqrt();
        let dot = row
            .iter()
            .enumerate()
            .map(|(c, &v)| f64::from(g[r * cols + c]) * f64::from(gamma[c]) * f64::from(v) / rms)
            .sum::<f64>()
            / cols as f64;
        for (c, &v) in row.iter().enumerate() {
            let i = r * cols + c;
            let z = f64::from(v) / rms;
            forward[i] = f64::from(gamma[c]) * z;
            packed[i] = (f64::from(g[i]) * f64::from(gamma[c]) - z * dot) / rms;
            packed[x.len() + i] = f64::from(g[i]) * z;
            dgamma[c] += packed[x.len() + i];
        }
    }
    (forward, packed, dgamma)
}

/// Assert successful close on normal paths. Drop attempts best-effort close
/// during assertion unwinding; it does not establish successful cleanup if
/// an earlier failure leaves resources live or close itself fails.
struct RmsDevice(*mut ffi::AckDevice);

impl RmsDevice {
    fn close(mut self) {
        let dev = std::mem::replace(&mut self.0, std::ptr::null_mut());
        assert_eq!(ffi::ack_close(dev), ffi::ACK_OK, "{}", last_error());
    }
}

impl Drop for RmsDevice {
    fn drop(&mut self) {
        if !self.0.is_null() {
            let _ = ffi::ack_close(self.0);
        }
    }
}

fn rms_device() -> Option<RmsDevice> {
    let dev = ffi::ack_open();
    if dev.is_null() {
        assert!(
            std::env::var_os("ACK_REQUIRE_DEVICE").is_none(),
            "RMSNorm ABI device required: {}",
            last_error()
        );
        eprintln!("UNMEASURED here (RMSNorm ABI): {}", last_error());
        None
    } else {
        Some(RmsDevice(dev))
    }
}

fn check_rmsnorm_case(
    device: &RmsDevice,
    name: &str,
    x: &[f32],
    g: &[f32],
    gamma: &[f32],
    epsilon: f32,
) {
    let dev = device.0;
    let cols = gamma.len();
    let rows = x.len() / cols;
    assert_eq!(x.len(), g.len());
    assert_eq!(x.len() % cols, 0);
    let alloc = |values: &[f32]| {
        let buf = ffi::ack_buffer_alloc(dev, (values.len() * 4) as u64);
        assert!(!buf.is_null(), "{}", last_error());
        assert_eq!(
            unsafe { ffi::ack_upload(dev, buf, values.as_ptr().cast(), values.len() * 4) },
            ffi::ACK_OK,
            "{}",
            last_error()
        );
        buf
    };
    let download = |buf, len: usize| {
        let mut values = vec![0.0f32; len];
        assert_eq!(
            unsafe { ffi::ack_download(dev, buf, values.as_mut_ptr().cast(), len * 4) },
            ffi::ACK_OK,
            "{}",
            last_error()
        );
        values
    };
    let bx = alloc(x);
    let bg = alloc(g);
    let bgamma = alloc(gamma);
    let tiny = alloc(&[123.0]);
    // Poison every output and guard its tail. In particular, op4 must read
    // the contributions in the SECOND half of op3's packed output.
    let poison = |len: usize| {
        let mut values = vec![f32::NAN; len + 1];
        values[len] = 12_345.25;
        alloc(&values)
    };
    let forward = poison(x.len());
    let packed = poison(2 * x.len());
    let reduction = poison(cols);
    for (op, primary, upstream, weights, output) in [
        (RowOp::RmsNormForward, bx, tiny, bgamma, forward),
        (RowOp::RmsNormBackward, bx, bg, bgamma, packed),
        (RowOp::RmsNormGammaReduction, packed, tiny, tiny, reduction),
    ] {
        let mut block = request(rows as u32, cols as u32, op);
        block[12..16].copy_from_slice(&epsilon.to_bits().to_le_bytes());
        assert_eq!(
            unsafe {
                ffi::ack_rowwise(
                    dev,
                    primary,
                    upstream,
                    weights,
                    output,
                    block.as_ptr(),
                    block.len(),
                    std::ptr::null_mut(),
                )
            },
            ffi::ACK_OK,
            "{name} {op:?}: {}",
            last_error()
        );
    }
    let expected = rmsnorm_reference(x, g, gamma, epsilon);
    let mut failures = Vec::new();
    for (label, buf, want, abs_tol) in [
        ("forward", forward, &expected.0, 2.0e-5),
        ("packed dx/contribution", packed, &expected.1, 2.0e-5),
        ("dgamma", reduction, &expected.2, 1.0e-4),
    ] {
        let got = download(buf, want.len() + 1);
        if name == "ill-conditioned gamma diagnostic" && label == "dgamma" {
            eprintln!("{name}: got {:?}; FP64 {:?}", &got[..want.len()], want);
        }
        assert_eq!(
            got[want.len()],
            12_345.25,
            "{name} {label}: output tail overwritten"
        );
        for (i, (&a, &b)) in got[..want.len()].iter().zip(want).enumerate() {
            assert!(
                b.is_finite() && b.abs() <= f64::from(f32::MAX),
                "fixture has unrepresentable {label}[{i}]={b}"
            );
            // A normal tiny result must not silently become zero under the
            // usual absolute tolerance. Stay away from the denormal boundary.
            let tiny_normal = b.abs() >= 2.0 * f64::from(f32::MIN_POSITIVE) && b.abs() < 1.0e-30;
            let effective_abs = if tiny_normal { 0.0 } else { abs_tol };
            if (!a.is_finite() || !within(a, b, effective_abs, 2.0e-5))
                && failures
                    .iter()
                    .filter(|line: &&String| line.starts_with(label))
                    .count()
                    < 3
            {
                failures.push(format!("{label}[{i}]: got {a}, want {b}"));
            }
        }
    }
    assert_eq!(download(bx, x.len()), x, "{name}: primary changed");
    assert_eq!(download(bg, g.len()), g, "{name}: upstream changed");
    assert_eq!(
        download(bgamma, gamma.len()),
        gamma,
        "{name}: gamma changed"
    );
    for buf in [bx, bg, bgamma, tiny, forward, packed, reduction] {
        assert_eq!(
            ffi::ack_buffer_free(dev, buf),
            ffi::ACK_OK,
            "{}",
            last_error()
        );
    }
    assert!(failures.is_empty(), "{name}: {}", failures.join("; "));
}

#[test]
fn rmsnorm_sparse_finite_extremes_preserve_forward_and_normal_backward_results() {
    let Some(device) = rms_device() else { return };
    for (peak, upstream) in [
        (1.0e20f32, 1.0e16f32),
        (1.0e38, 1.0e34),
        (1.0e38, 0.5),
        (f32::MAX, 1.0e34),
        (-f32::MAX, -1.0e34),
    ] {
        let mut x = vec![0.0; 64];
        let mut g = vec![0.0; 64];
        x[0] = peak;
        g[1] = upstream;
        // This makes dx[1] about 8e-4 for the recorded 1e38 failure,
        // clearly above the absolute tolerance, while gamma's gradient is 0.
        check_rmsnorm_case(
            &device,
            &format!("sparse peak={peak}"),
            &x,
            &g,
            &[1.0; 64],
            1.0e-6,
        );
    }
    device.close();
}

#[test]
fn rmsnorm_dense_finite_extremes_preserve_nonzero_backward_results() {
    let Some(device) = rms_device() else { return };
    let cols = 65;
    let x: Vec<f32> = (0..3 * cols)
        .map(|i| match i / cols {
            0 => {
                if i % 2 == 0 {
                    f32::MAX
                } else {
                    -f32::MAX
                }
            }
            1 => 1.0e38,
            _ => -f32::MAX,
        })
        .collect();
    // Align the signs so the reduction has nonzero contributions without
    // near-cancellation of ~1e34 values; this is a scaling regression, not a
    // relative-error claim for arbitrarily ill-conditioned gamma sums.
    let g: Vec<f32> = (0..x.len())
        .map(|i| x[i].signum() * (1 + i % 7) as f32 * 1.0e34)
        .collect();
    check_rmsnorm_case(
        &device,
        "dense extremes 3x65",
        &x,
        &g,
        &vec![1.0; cols],
        EPSILON,
    );
    device.close();
}

#[test]
#[ignore = "Known FP32 gamma-sum cancellation limitation; run explicitly to reproduce, not acceptance evidence"]
fn rmsnorm_ill_conditioned_gamma_sum_diagnostic() {
    let Some(device) = rms_device() else { return };
    let cols = 65;
    let x: Vec<f32> = (0..3 * cols)
        .map(|i| match i / cols {
            0 => {
                if i % 2 == 0 {
                    f32::MAX
                } else {
                    -f32::MAX
                }
            }
            1 => 1.0e38,
            _ => -f32::MAX,
        })
        .collect();
    let g: Vec<f32> = (0..x.len()).map(|i| (1 + i % 7) as f32 * 1.0e33).collect();
    check_rmsnorm_case(
        &device,
        "ill-conditioned gamma diagnostic",
        &x,
        &g,
        &vec![1.0; cols],
        EPSILON,
    );
    device.close();
}

#[test]
fn rmsnorm_large_and_small_scales_keep_representable_gradients() {
    let Some(device) = rms_device() else { return };
    for (peak, upstream, epsilon) in [(f32::MAX, 1.0e35, 1.0e-6), (0.1, 4.0e37, 0.01)] {
        let mut x = vec![peak; 64];
        let mut g = vec![0.0; 64];
        x[0] = 0.0;
        g[0] = upstream;
        // The dot is exactly zero. Expected dx[0] is approximately 2.96e-4
        // at the large scale and 2.84e38 at the small scale: both normal.
        // Forming 1/rms first loses the first; dividing g/scale first can
        // overflow the second even though its final result is representable.
        check_rmsnorm_case(
            &device,
            &format!("gradient scale={peak}"),
            &x,
            &g,
            &[1.0; 64],
            epsilon,
        );
    }
    device.close();
}

#[test]
fn rmsnorm_mixed_rows_and_nonzero_gamma_reduction_match_fp64() {
    let Some(device) = rms_device() else { return };
    // Both the 65-column and 67-row tails cross a workgroup. Each gamma
    // column receives nonzero contributions from multiple, distinct rows.
    let (rows, cols) = (67, 65);
    let mut x = vec![0.0; rows * cols];
    let mut g = vec![0.0; rows * cols];
    let gamma: Vec<f32> = (0..cols).map(|c| 0.75 + (c % 7) as f32 / 8.0).collect();
    for r in 0..rows {
        for c in 0..cols {
            let i = r * cols + c;
            x[i] = match r % 5 {
                0 => 0.0,
                1 => (c as f32 - 32.0) / 8.0,
                2 => {
                    if c == r % cols {
                        1.0e38
                    } else {
                        0.0
                    }
                }
                3 => {
                    if c % 2 == 0 {
                        f32::MAX
                    } else {
                        -f32::MAX
                    }
                }
                _ => -1.0e38,
            };
            g[i] = 0.125 + ((r + c) % 11) as f32 / 16.0;
        }
    }
    check_rmsnorm_case(&device, "mixed 67x65", &x, &g, &gamma, EPSILON);
    device.close();
}

#[test]
fn rmsnorm_epsilon_endpoints_cover_zero_tiny_and_ordinary_rows() {
    let Some(device) = rms_device() else { return };
    let x = [0.0, 0.0, 0.0, 1.0e-30, -1.0e-30, 0.0, -3.0, 2.0, 1.0];
    let g = [1.0e-7; 9];
    for epsilon in [1.0e-12, 1.0e12] {
        check_rmsnorm_case(
            &device,
            &format!("epsilon={epsilon}"),
            &x,
            &g,
            &[0.5, -1.5, 2.0],
            epsilon,
        );
    }
    device.close();
}

#[test]
fn rmsnorm_buffer_refusals_preserve_every_operand() {
    let Some(device) = rms_device() else { return };
    let dev = device.0;
    let alloc = |elements: usize| {
        let buf = ffi::ack_buffer_alloc(dev, (elements * 4) as u64);
        assert!(!buf.is_null(), "{}", last_error());
        let data = vec![123.25f32; elements];
        assert_eq!(
            unsafe { ffi::ack_upload(dev, buf, data.as_ptr().cast(), elements * 4) },
            ffi::ACK_OK,
            "{}",
            last_error()
        );
        buf
    };
    let primary = alloc(12);
    let upstream = alloc(12);
    let gamma = alloc(12);
    let output = alloc(12);
    let short_packed = alloc(6);
    let tiny = alloc(1);
    for (op, x, g, w, out, code, reason) in [
        (
            RowOp::RmsNormForward,
            primary,
            tiny,
            tiny,
            output,
            ffi::ACK_ERR_SIZE,
            "rowwise-gamma-too-small",
        ),
        (
            RowOp::RmsNormBackward,
            primary,
            tiny,
            gamma,
            output,
            ffi::ACK_ERR_SIZE,
            "rowwise-upstream-too-small",
        ),
        (
            RowOp::RmsNormForward,
            primary,
            tiny,
            gamma,
            gamma,
            ffi::ACK_ERR_SHAPE,
            "rowwise-output-aliases-input",
        ),
        (
            RowOp::RmsNormBackward,
            primary,
            upstream,
            gamma,
            upstream,
            ffi::ACK_ERR_SHAPE,
            "rowwise-output-aliases-input",
        ),
        (
            RowOp::RmsNormBackward,
            primary,
            upstream,
            gamma,
            short_packed,
            ffi::ACK_ERR_SIZE,
            "rowwise-output-too-small",
        ),
        (
            RowOp::RmsNormGammaReduction,
            short_packed,
            tiny,
            tiny,
            output,
            ffi::ACK_ERR_SIZE,
            "rowwise-primary-too-small",
        ),
        (
            RowOp::RmsNormGammaReduction,
            primary,
            tiny,
            tiny,
            primary,
            ffi::ACK_ERR_SHAPE,
            "rowwise-output-aliases-input",
        ),
    ] {
        let block = request(2, 3, op);
        assert_eq!(
            unsafe {
                ffi::ack_rowwise(
                    dev,
                    x,
                    g,
                    w,
                    out,
                    block.as_ptr(),
                    block.len(),
                    std::ptr::null_mut(),
                )
            },
            code,
            "{op:?}: {}",
            last_error()
        );
        assert!(last_error().contains(reason), "{op:?}: {}", last_error());
    }
    for (buf, elements) in [
        (primary, 12),
        (upstream, 12),
        (gamma, 12),
        (output, 12),
        (short_packed, 6),
        (tiny, 1),
    ] {
        let mut got = vec![0.0f32; elements];
        assert_eq!(
            unsafe { ffi::ack_download(dev, buf, got.as_mut_ptr().cast(), elements * 4) },
            ffi::ACK_OK,
            "{}",
            last_error()
        );
        assert_eq!(got, vec![123.25; elements], "a refusal dispatched work");
        assert_eq!(
            ffi::ack_buffer_free(dev, buf),
            ffi::ACK_OK,
            "{}",
            last_error()
        );
    }
    device.close();
}

#[test]
fn rmsnorm_invalid_epsilon_is_refused_before_device_lookup() {
    for op in [
        RowOp::RmsNormForward,
        RowOp::RmsNormBackward,
        RowOp::RmsNormGammaReduction,
    ] {
        for epsilon in [0.0f32, -1.0, 1.0e-13, 1.0e13, f32::NAN, f32::INFINITY] {
            let mut block = request(2, 3, op);
            block[12..16].copy_from_slice(&epsilon.to_bits().to_le_bytes());
            assert_eq!(
                unsafe {
                    ffi::ack_rowwise(
                        std::ptr::null(),
                        std::ptr::null(),
                        std::ptr::null(),
                        std::ptr::null(),
                        std::ptr::null(),
                        block.as_ptr(),
                        block.len(),
                        std::ptr::null_mut(),
                    )
                },
                ffi::ACK_ERR_SHAPE,
                "{op:?} eps={epsilon}: {}",
                last_error()
            );
            assert!(
                last_error().contains("rowwise-invalid-epsilon"),
                "{}",
                last_error()
            );
        }
    }
}

#[test]
fn a_null_short_or_malformed_request_is_refused_before_any_device_lookup() {
    let dev = std::ptr::null();
    let none = std::ptr::null();
    let call = |block: *const u8, len: usize| unsafe {
        ffi::ack_rowwise(
            dev,
            none,
            none,
            none,
            none,
            block,
            len,
            std::ptr::null_mut(),
        )
    };
    assert_eq!(
        call(std::ptr::null(), ffi::ROWWISE_PUSH_BYTES),
        ffi::ACK_ERR_NULL,
        "a null request block"
    );
    let short = [0u8; 8];
    assert_eq!(
        call(short.as_ptr(), short.len()),
        ffi::ACK_ERR_SIZE,
        "an 8-byte block: {}",
        last_error()
    );
    let mut unknown = request(4, 8, RowOp::SoftmaxForward);
    unknown[8..12].copy_from_slice(&9u32.to_le_bytes());
    assert_eq!(
        call(unknown.as_ptr(), unknown.len()),
        ffi::ACK_ERR_SHAPE,
        "{}",
        last_error()
    );
    assert!(
        last_error().contains("rowwise-operation-out-of-range"),
        "{}",
        last_error()
    );
    let zero = request(0, 8, RowOp::SoftmaxForward);
    assert_eq!(call(zero.as_ptr(), zero.len()), ffi::ACK_ERR_SHAPE);
    assert!(
        last_error().contains("rowwise-dimension-out-of-range"),
        "{}",
        last_error()
    );
    let wide = request(1, MAX_COLS + 1, RowOp::LogSoftmaxForward);
    assert_eq!(call(wide.as_ptr(), wide.len()), ffi::ACK_ERR_SHAPE);
    let huge = request(1 << 20, MAX_COLS, RowOp::SoftmaxForward);
    assert_eq!(call(huge.as_ptr(), huge.len()), ffi::ACK_ERR_SIZE);
    assert!(
        last_error().contains("rowwise-element-limit-exceeded"),
        "{}",
        last_error()
    );
    // with a well-formed block the null device handle is what refuses next
    let good = request(4, 8, RowOp::SoftmaxForward);
    assert_eq!(
        call(good.as_ptr(), good.len()),
        ffi::ACK_ERR_NULL,
        "a null device handle is refused as null before any registry lookup: {}",
        last_error()
    );
}

#[test]
fn softmax_and_log_softmax_forward_and_backward_match_the_reference_and_refusals_are_by_name() {
    let dev = ffi::ack_open();
    if dev.is_null() {
        if std::env::var_os("ACK_REQUIRE_DEVICE").is_some() {
            panic!(
                "rowwise ABI: ACK_REQUIRE_DEVICE is set and ack_open failed: {}",
                last_error()
            );
        }
        eprintln!("UNMEASURED here (rowwise ABI): {}", last_error());
        return;
    }
    // five rows of 130 columns: more than the 64 lanes of a workgroup, and not
    // a multiple of them, so the column tails are exercised
    let rows = 5usize;
    let cols = 130usize;
    let n = rows * cols;
    let x: Vec<f32> = (0..n)
        .map(|i| ((i * 7919) % 97) as f32 / 13.0 - 3.5)
        .collect();
    let g: Vec<f32> = (0..n)
        .map(|i| ((i * 104_729) % 89) as f32 / 11.0 - 4.0)
        .collect();
    let alloc = |elements: usize| {
        let b = ffi::ack_buffer_alloc(dev, (elements * 4) as u64);
        assert!(!b.is_null(), "{}", last_error());
        b
    };
    let upload = |buf, data: &[f32]| {
        let rc = unsafe { ffi::ack_upload(dev, buf, data.as_ptr().cast::<u8>(), data.len() * 4) };
        assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
    };
    let download = |buf, elements: usize| {
        let mut out = vec![0f32; elements];
        let rc =
            unsafe { ffi::ack_download(dev, buf, out.as_mut_ptr().cast::<u8>(), out.len() * 4) };
        assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
        out
    };
    let poison = |buf, elements: usize| upload(buf, &vec![f32::NAN; elements]);
    let bx = alloc(n);
    let bg = alloc(n);
    let bp = alloc(n); // the saved probabilities (the kit's own forward)
    let blp = alloc(n); // the saved log-probabilities
    let bout = alloc(n);
    let tiny = alloc(4);
    upload(bx, &x);
    upload(bg, &g);
    let mut ms = 0f64;
    let (r, c) = (rows as u32, cols as u32);

    // 1. softmax forward: the unread upstream and gamma slots are bound to the input
    poison(bp, n);
    let block = request(r, c, RowOp::SoftmaxForward);
    let rc = unsafe { ffi::ack_rowwise(dev, bx, bx, bx, bp, block.as_ptr(), block.len(), &mut ms) };
    assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
    assert!(
        ms.is_nan() || ms >= 0.0,
        "a duration or NaN, never negative: {ms}"
    );
    let p = download(bp, n);
    // range and normalisation, independent of the f64 reference below: a
    // probability lies in [0, 1] and a row sums to one. The reference is a
    // formula written by the same hand as the entry, so a shared
    // misderivation would agree with itself; these two properties would not.
    for (i, value) in p.iter().enumerate() {
        assert!(
            value.is_finite() && (0.0..=1.0).contains(value),
            "softmax[{i}] = {value} is not a probability"
        );
    }
    for (r, row) in p.chunks_exact(cols).enumerate() {
        let total: f64 = row.iter().map(|&v| f64::from(v)).sum();
        assert!(
            (total - 1.0).abs() <= 2.0e-5,
            "softmax row {r} sums to {total}, not to one"
        );
    }
    let want = softmax_rows(&x, cols);
    for (i, (got, w)) in p.iter().zip(&want).enumerate() {
        assert!(
            within(*got, *w, SOFTMAX_ABS_TOL, SOFTMAX_REL_TOL),
            "softmax[{i}]: got {got}, want {w}"
        );
    }
    eprintln!("rowwise softmax forward {rows}x{cols}: {ms:.3} ms");

    // 2. log-softmax forward with masked logits: two -inf entries in row 1
    //    give -inf outputs and no NaN anywhere
    let mut masked = x.clone();
    masked[cols + 3] = f32::NEG_INFINITY;
    masked[cols + 77] = f32::NEG_INFINITY;
    upload(bx, &masked);
    poison(blp, n);
    let block = request(r, c, RowOp::LogSoftmaxForward);
    let rc =
        unsafe { ffi::ack_rowwise(dev, bx, bx, bx, blp, block.as_ptr(), block.len(), &mut ms) };
    assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
    let lp = download(blp, n);
    // the same, for log probabilities: never above zero, never NaN even where
    // a class is masked to -inf, and the exponentials still sum to one
    for (i, value) in lp.iter().enumerate() {
        assert!(
            !value.is_nan() && *value <= 0.0,
            "log-softmax[{i}] = {value} is not a log probability"
        );
    }
    for (r, row) in lp.chunks_exact(cols).enumerate() {
        let total: f64 = row.iter().map(|&v| f64::from(v).exp()).sum();
        assert!(
            (total - 1.0).abs() <= 2.0e-5,
            "exp(log-softmax) row {r} sums to {total}, not to one"
        );
    }
    let want = log_softmax_rows(&masked, cols);
    for (i, (got, w)) in lp.iter().zip(&want).enumerate() {
        assert!(!got.is_nan(), "log-softmax[{i}] is NaN");
        assert!(
            within(*got, *w, LOG_SOFTMAX_ABS_TOL, LOG_SOFTMAX_REL_TOL),
            "log-softmax[{i}]: got {got}, want {w}"
        );
    }
    assert_eq!(lp[cols + 3], f32::NEG_INFINITY);
    assert_eq!(lp[cols + 77], f32::NEG_INFINITY);
    // a tiny buffer in an unread slot is admitted: the slot law
    poison(bout, n);
    let rc = unsafe {
        ffi::ack_rowwise(
            dev,
            bx,
            tiny,
            tiny,
            bout,
            block.as_ptr(),
            block.len(),
            &mut ms,
        )
    };
    assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
    assert_eq!(
        download(bout, n),
        lp,
        "the same forward through the unread slots"
    );
    upload(bx, &x);

    // 3. softmax backward from the kit's own probabilities and the upstream gradient
    poison(bout, n);
    let block = request(r, c, RowOp::SoftmaxBackward);
    let rc =
        unsafe { ffi::ack_rowwise(dev, bp, bg, bp, bout, block.as_ptr(), block.len(), &mut ms) };
    assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
    let got = download(bout, n);
    let want = softmax_backward_rows(&p, &g, cols);
    for (i, (v, w)) in got.iter().zip(&want).enumerate() {
        assert!(
            within(*v, *w, BACKWARD_ABS_TOL, BACKWARD_REL_TOL),
            "softmax backward[{i}]: got {v}, want {w}"
        );
    }

    // 4. log-softmax backward from the unmasked log-probabilities
    poison(blp, n);
    let block = request(r, c, RowOp::LogSoftmaxForward);
    let rc =
        unsafe { ffi::ack_rowwise(dev, bx, bx, bx, blp, block.as_ptr(), block.len(), &mut ms) };
    assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
    let lp = download(blp, n);
    poison(bout, n);
    let block = request(r, c, RowOp::LogSoftmaxBackward);
    let rc = unsafe {
        ffi::ack_rowwise(
            dev,
            blp,
            bg,
            blp,
            bout,
            block.as_ptr(),
            block.len(),
            &mut ms,
        )
    };
    assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
    let got = download(bout, n);
    let want = log_softmax_backward_rows(&lp, &g, cols);
    for (i, (v, w)) in got.iter().zip(&want).enumerate() {
        assert!(
            within(*v, *w, BACKWARD_ABS_TOL, BACKWARD_REL_TOL),
            "log-softmax backward[{i}]: got {v}, want {w}"
        );
    }

    // refusals by name, nothing dispatched: the previous result survives each one
    let before = download(bout, n);
    let forward = request(r, c, RowOp::SoftmaxForward);
    // the output through the input's buffer
    let rc = unsafe {
        ffi::ack_rowwise(
            dev,
            bx,
            bx,
            bx,
            bx,
            forward.as_ptr(),
            forward.len(),
            &mut ms,
        )
    };
    assert_eq!(rc, ffi::ACK_ERR_SHAPE, "{}", last_error());
    assert!(
        last_error().contains("rowwise-output-aliases-input"),
        "{}",
        last_error()
    );
    // the output through the upstream buffer of a backward
    let rc = unsafe { ffi::ack_rowwise(dev, bp, bg, bp, bg, block.as_ptr(), block.len(), &mut ms) };
    assert_eq!(rc, ffi::ACK_ERR_SHAPE, "{}", last_error());
    assert!(
        last_error().contains("rowwise-output-aliases-input"),
        "{}",
        last_error()
    );
    // a primary buffer too small for the rows
    let rc = unsafe {
        ffi::ack_rowwise(
            dev,
            tiny,
            bx,
            bx,
            bout,
            forward.as_ptr(),
            forward.len(),
            &mut ms,
        )
    };
    assert_eq!(rc, ffi::ACK_ERR_SIZE, "{}", last_error());
    assert!(
        last_error().contains("rowwise-primary-too-small"),
        "{}",
        last_error()
    );
    // an upstream buffer too small for a backward (read by that operation)
    let rc = unsafe {
        ffi::ack_rowwise(
            dev,
            bp,
            tiny,
            bp,
            bout,
            block.as_ptr(),
            block.len(),
            &mut ms,
        )
    };
    assert_eq!(rc, ffi::ACK_ERR_SIZE, "{}", last_error());
    assert!(
        last_error().contains("rowwise-upstream-too-small"),
        "{}",
        last_error()
    );
    // an output buffer too small for the rows
    let rc = unsafe {
        ffi::ack_rowwise(
            dev,
            bx,
            bx,
            bx,
            tiny,
            forward.as_ptr(),
            forward.len(),
            &mut ms,
        )
    };
    assert_eq!(rc, ffi::ACK_ERR_SIZE, "{}", last_error());
    assert!(
        last_error().contains("rowwise-output-too-small"),
        "{}",
        last_error()
    );
    // a freed buffer is refused by the registry before the plan is checked against it
    let spare = alloc(n);
    assert_eq!(
        ffi::ack_buffer_free(dev, spare),
        ffi::ACK_OK,
        "{}",
        last_error()
    );
    let rc = unsafe {
        ffi::ack_rowwise(
            dev,
            spare,
            bx,
            bx,
            bout,
            forward.as_ptr(),
            forward.len(),
            &mut ms,
        )
    };
    assert_eq!(rc, ffi::ACK_ERR_FREED, "{}", last_error());
    assert_eq!(download(bout, n), before, "a refusal dispatches nothing");

    for buf in [bx, bg, bp, blp, bout, tiny] {
        assert_eq!(
            ffi::ack_buffer_free(dev, buf),
            ffi::ACK_OK,
            "{}",
            last_error()
        );
    }
    assert_eq!(ffi::ack_close(dev), ffi::ACK_OK, "{}", last_error());
}
