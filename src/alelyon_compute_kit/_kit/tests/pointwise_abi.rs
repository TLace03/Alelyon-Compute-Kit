//! pointwise-f32/v1 at the C ABI (`ack_pointwise`, ABI 6, operation schema
//! 2): the caller's 120-byte plan block is re-validated by
//! `pointwise_ops::PointwisePlan` against the buffers' real capacities and
//! identities, refusals are by name, three small schema-1 operations (a
//! contiguous add, a broadcast multiply and an in-place division by a scalar)
//! and the schema-2 operations (sigmoid and silu, pow with the CPU's special
//! exponents, the general path and the odd-integer boundary at 2^24, sin and
//! cos at the AKV RoPE angles, at 1000 rad and across the reduction's proven
//! domain edge (NaN above it), addcdiv and addcmul, lerp, sigmoid_backward, clamp, iota, triu
//! and the unary set) match a host FP64 reference within the family's own
//! tolerance, bitwise where the operation is a chain of correctly rounded
//! IEEE steps. The host-only checks need no device; the operations need one
//! and print UNMEASURED without it unless ACK_REQUIRE_DEVICE is set.
//! Fixed-fixture evidence only, on the card that ran it.
use alelyon_compute_kit::ffi;
use alelyon_compute_kit::pointwise_ops::{
    PointwiseOp, PointwisePlan, Scalars, Storage, StridedView, LAST_OP, MAX_NDIM, OPERANDS,
    PUSH_BYTES,
};
use alelyon_compute_kit::Buffer;
use std::ffi::{c_char, CStr};

fn last_error() -> String {
    let mut buf = vec![0 as c_char; 1024];
    let rc = unsafe { ffi::ack_last_error(buf.as_mut_ptr(), buf.len()) };
    assert!(rc >= 0, "ack_last_error returned {rc}");
    unsafe { CStr::from_ptr(buf.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

/// A block built by the plan module itself over declared storages that are
/// large enough, so what the C entry re-validates is the real buffers.
fn block(
    op: PointwiseOp,
    shape: &[u64],
    views: [StridedView; OPERANDS],
    scalars: Scalars,
) -> [u8; PUSH_BYTES] {
    let roomy = [Storage::new(1, u64::MAX / 4); OPERANDS];
    // distinct declared identities: aliasing is decided by the C entry, not here
    let storage = [
        roomy[0],
        Storage::new(2, u64::MAX / 4),
        Storage::new(3, u64::MAX / 4),
        Storage::new(4, u64::MAX / 4),
    ];
    PointwisePlan::new(op, shape, views, scalars, storage)
        .unwrap()
        .push_constants()
}

fn within_tolerance(got: f32, want: f64) -> bool {
    (f64::from(got) - want).abs() <= 1e-7 + 1e-4 * want.abs()
}

/// The largest `|x|` the shader's five-word split of pi/2 reduces exactly,
/// derived and measured in `kernels/pointwise_f32.comp`; above it sin and cos
/// write NaN rather than a plausible number.
// the exact decimal of the float32 0x48c90fc1, written out as the shader does
#[allow(clippy::excessive_precision)]
const SINCOS_MAX: f32 = 411774.03125;

#[test]
fn a_null_short_or_unknown_plan_is_refused_before_any_device_lookup() {
    let dev = std::ptr::null();
    let none = std::ptr::null();
    let rc = unsafe {
        ffi::ack_pointwise(
            dev,
            none,
            none,
            none,
            none,
            std::ptr::null(),
            PUSH_BYTES,
            std::ptr::null_mut(),
        )
    };
    assert_eq!(rc, ffi::ACK_ERR_NULL, "a null plan block");
    let short = [0u8; 64];
    let rc = unsafe {
        ffi::ack_pointwise(
            dev,
            none,
            none,
            none,
            none,
            short.as_ptr(),
            short.len(),
            std::ptr::null_mut(),
        )
    };
    assert_eq!(rc, ffi::ACK_ERR_SIZE, "a 64-byte block: {}", last_error());
    let mut unknown = [0u8; PUSH_BYTES];
    unknown[..4].copy_from_slice(&(LAST_OP + 1).to_le_bytes());
    let rc = unsafe {
        ffi::ack_pointwise(
            dev,
            none,
            none,
            none,
            none,
            unknown.as_ptr(),
            unknown.len(),
            std::ptr::null_mut(),
        )
    };
    assert_eq!(
        rc,
        ffi::ACK_ERR_SHAPE,
        "operation {}: {}",
        LAST_OP + 1,
        last_error()
    );
    assert!(
        last_error().contains("pointwise-operation-out-of-range"),
        "{}",
        last_error()
    );
    // with a well-formed block the null device handle is what refuses next
    let full = block(
        PointwiseOp::Copy,
        &[4],
        [StridedView::contiguous(&[4]).unwrap(); OPERANDS],
        Scalars::new(0.0),
    );
    let rc = unsafe {
        ffi::ack_pointwise(
            dev,
            none,
            none,
            none,
            none,
            full.as_ptr(),
            full.len(),
            std::ptr::null_mut(),
        )
    };
    assert_eq!(
        rc,
        ffi::ACK_ERR_NULL,
        "a null device handle is refused as null before any registry lookup, got {rc}: {}",
        last_error()
    );
}

#[test]
fn three_operations_match_a_host_f64_reference_and_refusals_are_by_name() {
    let dev = ffi::ack_open();
    if dev.is_null() {
        if std::env::var_os("ACK_REQUIRE_DEVICE").is_some() {
            panic!(
                "pointwise ABI: ACK_REQUIRE_DEVICE is set and ack_open failed: {}",
                last_error()
            );
        }
        eprintln!("UNMEASURED here (pointwise ABI): {}", last_error());
        return;
    }
    // a 3-D shape whose element count is not a multiple of the 1,024 elements
    // one workgroup handles, so the last group's tail lanes are exercised
    let shape = [3u64, 5, 7];
    let n = 105usize;
    let x: Vec<f32> = (0..n)
        .map(|i| ((i * 7919) % 97) as f32 / 13.0 - 3.5)
        .collect();
    let y: Vec<f32> = (0..n)
        .map(|i| ((i * 104_729) % 89) as f32 / 11.0 - 4.0)
        .collect();
    let w: Vec<f32> = (0..7).map(|i| (i as f32 + 1.0) * 0.375 - 1.0).collect();
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
    let bx = alloc(n);
    let by = alloc(n);
    let bw = alloc(7);
    let bout = alloc(n);
    upload(bx, &x);
    upload(by, &y);
    upload(bw, &w);
    let poison = vec![f32::NAN; n];
    let contiguous = StridedView::contiguous(&shape).unwrap();
    let mut ms = 0f64;

    // 1. contiguous add: out = x + 1.0 * y, z bound to x
    upload(bout, &poison);
    let plan = block(
        PointwiseOp::AddScaled,
        &shape,
        [contiguous; OPERANDS],
        Scalars::new(1.0),
    );
    let rc =
        unsafe { ffi::ack_pointwise(dev, bx, by, bx, bout, plan.as_ptr(), plan.len(), &mut ms) };
    assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
    assert!(
        ms.is_nan() || ms >= 0.0,
        "a duration or NaN, never negative: {ms}"
    );
    let got = download(bout, n);
    for (i, value) in got.iter().enumerate() {
        let want = f64::from(x[i]) + f64::from(y[i]);
        assert!(
            within_tolerance(*value, want),
            "add[{i}]: got {value}, want {want}"
        );
        // both sides round one IEEE addition once: the f32 host sum is bitwise the kit's
        assert_eq!(
            value.to_bits(),
            (x[i] + y[i]).to_bits(),
            "add[{i}] is not bitwise the host's"
        );
    }
    eprintln!("pointwise add {shape:?}: {ms:.3} ms");

    // 2. broadcast multiply: out = w[7] * x[3,5,7], the weight row read through stride 0
    upload(bout, &poison);
    let weight = StridedView::new(0, [0, 0, 1, 0]);
    let plan = block(
        PointwiseOp::Mul,
        &shape,
        [contiguous, weight, contiguous, contiguous],
        Scalars::new(1.0),
    );
    let rc =
        unsafe { ffi::ack_pointwise(dev, bx, bw, bx, bout, plan.as_ptr(), plan.len(), &mut ms) };
    assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
    let got = download(bout, n);
    for (i, value) in got.iter().enumerate() {
        let want = f64::from(x[i]) * f64::from(w[i % 7]);
        assert!(
            within_tolerance(*value, want),
            "mul[{i}]: got {value}, want {want}"
        );
        assert_eq!(
            value.to_bits(),
            (x[i] * w[i % 7]).to_bits(),
            "mul[{i}] is not bitwise the host's"
        );
    }

    // 3. in-place division by a scalar: y /= 4.0, out is y's buffer through y's view,
    //    and the unused y and z slots are bound to the same buffer with the same view
    let mut expected = y.clone();
    let plan = block(
        PointwiseOp::Div,
        &shape,
        [contiguous; OPERANDS],
        Scalars::with_scalar_y(1.0, 4.0),
    );
    let rc = unsafe { ffi::ack_pointwise(dev, by, by, by, by, plan.as_ptr(), plan.len(), &mut ms) };
    assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
    let got = download(by, n);
    for (i, value) in got.iter().enumerate() {
        let want = f64::from(y[i]) / 4.0;
        assert!(
            within_tolerance(*value, want),
            "div[{i}]: got {value}, want {want}"
        );
        expected[i] = y[i] / 4.0;
    }
    // Vulkan does not require a correctly rounded division (DECLARED), so the
    // tolerance above is the claim; a power-of-two divisor is exact on any
    // conforming implementation that divides at all, reported here as observed
    let exact = got
        .iter()
        .zip(&expected)
        .all(|(g, e)| g.to_bits() == e.to_bits());
    eprintln!("pointwise div by 4.0 in place: bitwise the host's = {exact}");

    // refusals by name, nothing dispatched: the previous result must survive each one
    let before = download(by, n);
    // the output through another view of an input's buffer (y transposed in its last two dims)
    let transposed = StridedView::new(0, [35, 1, 5, 0]);
    let plan = block(
        PointwiseOp::Mul,
        &shape,
        [contiguous, contiguous, contiguous, transposed],
        Scalars::new(1.0),
    );
    let rc = unsafe { ffi::ack_pointwise(dev, bx, by, bx, by, plan.as_ptr(), plan.len(), &mut ms) };
    assert_eq!(rc, ffi::ACK_ERR_SHAPE, "{}", last_error());
    assert!(
        last_error().contains("pointwise-output-overlaps-input"),
        "{}",
        last_error()
    );
    // a view past its buffer (the weight buffer holds 7 elements, the view reads 105)
    let plan = block(
        PointwiseOp::Mul,
        &shape,
        [contiguous; OPERANDS],
        Scalars::new(1.0),
    );
    let rc =
        unsafe { ffi::ack_pointwise(dev, bx, bw, bx, bout, plan.as_ptr(), plan.len(), &mut ms) };
    assert_eq!(rc, ffi::ACK_ERR_SIZE, "{}", last_error());
    assert!(
        last_error().contains("pointwise-address-overflow"),
        "{}",
        last_error()
    );
    // an output overlapping itself (a broadcast output view) and a reserved flag
    let plan = block(
        PointwiseOp::Copy,
        &shape,
        [contiguous, contiguous, contiguous, contiguous],
        Scalars::new(0.0),
    );
    let mut self_overlap = plan;
    self_overlap[23 * 4..24 * 4].copy_from_slice(&0u32.to_le_bytes()); // out stride[0] = 0
    let rc = unsafe {
        ffi::ack_pointwise(
            dev,
            bx,
            by,
            bx,
            bout,
            self_overlap.as_ptr(),
            self_overlap.len(),
            &mut ms,
        )
    };
    assert_eq!(rc, ffi::ACK_ERR_SHAPE, "{}", last_error());
    assert!(
        last_error().contains("pointwise-output-self-overlap"),
        "{}",
        last_error()
    );
    let mut flagged = plan;
    flagged[12..16].copy_from_slice(&2u32.to_le_bytes()); // flags bit 1: the reserved scalar z
    let rc = unsafe {
        ffi::ack_pointwise(
            dev,
            bx,
            by,
            bx,
            bout,
            flagged.as_ptr(),
            flagged.len(),
            &mut ms,
        )
    };
    assert_eq!(rc, ffi::ACK_ERR_SHAPE, "{}", last_error());
    assert!(
        last_error().contains("pointwise-flags-out-of-range"),
        "{}",
        last_error()
    );
    // a freed buffer is refused by the registry before the plan is read
    let spare = alloc(n);
    assert_eq!(
        ffi::ack_buffer_free(dev, spare),
        ffi::ACK_OK,
        "{}",
        last_error()
    );
    let rc =
        unsafe { ffi::ack_pointwise(dev, bx, spare, bx, bout, plan.as_ptr(), plan.len(), &mut ms) };
    assert_eq!(rc, ffi::ACK_ERR_FREED, "{}", last_error());
    assert_eq!(download(by, n), before, "a refusal dispatches nothing");

    // the padded dimensions are not read: a 4-D shape of extent-1 leading dimensions
    // addresses the same elements, and copy(x) writes them all
    upload(bout, &poison);
    let four = [1u64, 3, 5, 7];
    let padded = StridedView::new(0, [0, 35, 7, 1]);
    let plan = block(
        PointwiseOp::Copy,
        &four,
        [padded; OPERANDS],
        Scalars::new(0.0),
    );
    let rc =
        unsafe { ffi::ack_pointwise(dev, bx, bx, bx, bout, plan.as_ptr(), plan.len(), &mut ms) };
    assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
    assert_eq!(download(bout, n), x, "copy through a padded 4-D view");
    let _ = MAX_NDIM;

    for buf in [bx, by, bw, bout] {
        assert_eq!(
            ffi::ack_buffer_free(dev, buf),
            ffi::ACK_OK,
            "{}",
            last_error()
        );
    }
    assert_eq!(ffi::ack_close(dev), ffi::ACK_OK, "{}", last_error());
}

/// A device result against a host FP64 reference: a NaN reference wants a
/// NaN, an infinite reference wants that infinity, and everything else is
/// held to the family tolerance.
fn matches_reference(got: f32, want: f64) -> bool {
    if want.is_nan() {
        got.is_nan()
    } else if want.is_infinite() {
        f64::from(got) == want
    } else {
        within_tolerance(got, want)
    }
}

/// torch's `std::min(std::max(x, low), high)` with the CPU's comparisons: a
/// NaN `x` propagates, a lower bound above the upper bound yields the upper.
fn clamp_reference(x: f32, low: f32, high: f32) -> f32 {
    let raised = if x < low { low } else { x };
    if raised > high {
        high
    } else {
        raised
    }
}

#[test]
fn schema_two_operations_match_a_host_f64_reference() {
    let dev = ffi::ack_open();
    if dev.is_null() {
        if std::env::var_os("ACK_REQUIRE_DEVICE").is_some() {
            panic!(
                "pointwise ABI (schema 2): ACK_REQUIRE_DEVICE is set and ack_open failed: {}",
                last_error()
            );
        }
        eprintln!(
            "UNMEASURED here (pointwise ABI, schema 2): {}",
            last_error()
        );
        return;
    }
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
    let run = |op: PointwiseOp,
               shape: &[u64],
               views: [StridedView; OPERANDS],
               scalars: Scalars,
               buffers: [*const Buffer; OPERANDS]|
     -> i32 {
        let plan = block(op, shape, views, scalars);
        let mut ms = 0f64;
        unsafe {
            ffi::ack_pointwise(
                dev,
                buffers[0],
                buffers[1],
                buffers[2],
                buffers[3],
                plan.as_ptr(),
                plan.len(),
                &mut ms,
            )
        }
    };

    // the fixture of the schema-1 test: 105 elements in a 3-D shape, x in
    // [-3.5, 3.9] with no zero, y in [-4, 4]
    let shape = [3u64, 5, 7];
    let n = 105usize;
    let x: Vec<f32> = (0..n)
        .map(|i| ((i * 7919) % 97) as f32 / 13.0 - 3.5)
        .collect();
    let y: Vec<f32> = (0..n)
        .map(|i| ((i * 104_729) % 89) as f32 / 11.0 - 4.0)
        .collect();
    let z: Vec<f32> = y.iter().map(|v| v.abs() + 0.5).collect();
    let contiguous = StridedView::contiguous(&shape).unwrap();
    let bx = alloc(n);
    let by = alloc(n);
    let bz = alloc(n);
    let bout = alloc(n);
    upload(bx, &x);
    upload(by, &y);
    upload(bz, &z);
    let poison = vec![f32::NAN; n];
    // out = op(x [, y, z]) over the fixture, the unused slots bound to x
    let unary = |op: PointwiseOp, scalars: Scalars| -> Vec<f32> {
        upload(bout, &poison);
        let rc = run(
            op,
            &shape,
            [contiguous; OPERANDS],
            scalars,
            [bx, bx, bx, bout],
        );
        assert_eq!(rc, ffi::ACK_OK, "{op:?}: {}", last_error());
        download(bout, n)
    };
    let binary = |op: PointwiseOp, scalars: Scalars| -> Vec<f32> {
        upload(bout, &poison);
        let rc = run(
            op,
            &shape,
            [contiguous; OPERANDS],
            scalars,
            [bx, by, bx, bout],
        );
        assert_eq!(rc, ffi::ACK_OK, "{op:?}: {}", last_error());
        download(bout, n)
    };
    let ternary = |op: PointwiseOp, scalars: Scalars| -> Vec<f32> {
        upload(bout, &poison);
        let rc = run(
            op,
            &shape,
            [contiguous; OPERANDS],
            scalars,
            [bx, by, bz, bout],
        );
        assert_eq!(rc, ffi::ACK_OK, "{op:?}: {}", last_error());
        download(bout, n)
    };

    // 1. the unary set: neg and abs are bitwise (a sign operation), the
    //    rest within tolerance (Vulkan requires no correctly rounded
    //    division or square root), sqrt and rsqrt NaN below zero
    for (i, got) in unary(PointwiseOp::Neg, Scalars::new(0.0))
        .iter()
        .enumerate()
    {
        assert_eq!(got.to_bits(), (-x[i]).to_bits(), "neg[{i}]");
    }
    for (i, got) in unary(PointwiseOp::Abs, Scalars::new(0.0))
        .iter()
        .enumerate()
    {
        assert_eq!(got.to_bits(), x[i].abs().to_bits(), "abs[{i}]");
    }
    for (i, got) in unary(PointwiseOp::Reciprocal, Scalars::new(0.0))
        .iter()
        .enumerate()
    {
        let want = 1.0 / f64::from(x[i]);
        assert!(
            matches_reference(*got, want),
            "reciprocal[{i}]: got {got}, want {want}"
        );
    }
    for (op, name) in [(PointwiseOp::Sqrt, "sqrt"), (PointwiseOp::Rsqrt, "rsqrt")] {
        let got = unary(op, Scalars::new(0.0));
        for (i, value) in got.iter().enumerate() {
            let root = f64::from(x[i]).sqrt();
            let want = if op == PointwiseOp::Sqrt {
                root
            } else {
                1.0 / root
            };
            assert!(
                matches_reference(*value, want),
                "{name}[{i}]: x {}, got {value}, want {want}",
                x[i]
            );
        }
        assert!(
            x.iter().zip(&got).any(|(x, g)| *x < 0.0 && g.is_nan()),
            "{name}: a negative input is NaN"
        );
    }

    // 2. sigmoid and silu over 1,005 points: [-30, 30] and the extremes, where
    //    exp overflows to infinity and the quotient must be 0 (or -0 for silu)
    let mut s: Vec<f32> = (0..1000).map(|i| -30.0 + 60.0 * i as f32 / 999.0).collect();
    s.extend([-200.0, 200.0, 0.0, -0.0, 1e-6]);
    let ns = s.len();
    let bs = alloc(ns);
    let bs_out = alloc(ns);
    upload(bs, &s);
    let line = StridedView::contiguous(&[ns as u64]).unwrap();
    let mut worst_sigmoid = 0f64;
    for (op, name) in [
        (PointwiseOp::Sigmoid, "sigmoid"),
        (PointwiseOp::Silu, "silu"),
    ] {
        upload(bs_out, &vec![f32::NAN; ns]);
        let rc = run(
            op,
            &[ns as u64],
            [line; OPERANDS],
            Scalars::new(0.0),
            [bs, bs, bs, bs_out],
        );
        assert_eq!(rc, ffi::ACK_OK, "{name}: {}", last_error());
        let got = download(bs_out, ns);
        for (i, value) in got.iter().enumerate() {
            let xd = f64::from(s[i]);
            let sigma = 1.0 / (1.0 + (-xd).exp());
            let want = if op == PointwiseOp::Sigmoid {
                sigma
            } else {
                xd * sigma
            };
            assert!(
                matches_reference(*value, want),
                "{name}[{i}]: x {}, got {value}, want {want}",
                s[i]
            );
            if op == PointwiseOp::Sigmoid {
                worst_sigmoid = worst_sigmoid.max((f64::from(*value) - want).abs());
            }
        }
        // the extremes: sigmoid(-200) is 0 and sigmoid(200) is 1 exactly on
        // the host; silu(-200) is -0 (a finite over an infinity)
        let at = |value: f32| {
            got[s
                .iter()
                .position(|v| v.to_bits() == value.to_bits())
                .unwrap()]
        };
        if op == PointwiseOp::Sigmoid {
            assert_eq!(at(-200.0), 0.0, "sigmoid(-200)");
            assert_eq!(at(200.0), 1.0, "sigmoid(200)");
            assert_eq!(at(0.0), 0.5, "sigmoid(0)");
        } else {
            assert_eq!(
                at(-200.0).to_bits(),
                (-0.0f32).to_bits(),
                "silu(-200) is -0"
            );
            assert_eq!(at(200.0), 200.0, "silu(200)");
        }
    }
    eprintln!("pointwise sigmoid: max abs error against f64 {worst_sigmoid:.3e}");

    // 3. pow with a scalar exponent: 2 and 3 are the CPU's x*x and x*x*x
    //    (bitwise, one rounding per multiply on both sides), 0.5 is sqrt, -1
    //    is the reciprocal, 1.5 takes the general path (NaN for a negative
    //    base); and a scalar base of 2 through PowScalarBase
    let squared = binary(PointwiseOp::Pow, Scalars::with_scalar_y(0.0, 2.0));
    let cubed = binary(PointwiseOp::Pow, Scalars::with_scalar_y(0.0, 3.0));
    for (i, (square, cube)) in squared.iter().zip(&cubed).enumerate() {
        assert_eq!(square.to_bits(), (x[i] * x[i]).to_bits(), "pow2[{i}]");
        assert_eq!(cube.to_bits(), (x[i] * x[i] * x[i]).to_bits(), "pow3[{i}]");
    }
    for (exponent, name) in [
        (0.5f32, "pow0.5"),
        (-1.0, "pow-1"),
        (1.5, "pow1.5"),
        (-2.0, "pow-2"),
    ] {
        let got = binary(PointwiseOp::Pow, Scalars::with_scalar_y(0.0, exponent));
        for (i, value) in got.iter().enumerate() {
            let want = f64::from(x[i]).powf(f64::from(exponent));
            assert!(
                matches_reference(*value, want),
                "{name}[{i}]: x {}, got {value}, want {want}",
                x[i]
            );
        }
    }
    // the exponent from a buffer takes the same path: x ^ y elementwise
    let tensor_exponent = binary(PointwiseOp::Pow, Scalars::new(0.0));
    for (i, value) in tensor_exponent.iter().enumerate() {
        let want = f64::from(x[i]).powf(f64::from(y[i]));
        assert!(
            matches_reference(*value, want),
            "pow_tensor[{i}]: x {}, y {}, got {value}, want {want}",
            x[i],
            y[i]
        );
    }
    let base_two = unary(PointwiseOp::PowScalarBase, Scalars::new(2.0));
    for (i, value) in base_two.iter().enumerate() {
        let want = 2f64.powf(f64::from(x[i]));
        assert!(
            matches_reference(*value, want),
            "2^x[{i}]: x {}, got {value}, want {want}",
            x[i]
        );
    }

    // 3b. pow's odd-integer test, pinned from both sides of 2^24. Every float
    //     at or above 16777216 is an even integer, so an odd exponent is one
    //     strictly below it; the shader wrote that guard at 2^23 = 8388608
    //     until 2026-09-07, which silently dropped a negative base's sign for
    //     every odd exponent in [2^23, 2^24) -- (-1)^8388609 came back +1.
    //     The fixture's |x| is never 0 or 1, so each result is exactly +-0
    //     (|x| < 1) or +-inf (|x| > 1) and the sign is the whole assertion.
    for (exponent, odd) in [
        (4_194_305.0f32, true),
        (8_388_607.0, true),
        (8_388_609.0, true),
        (8_388_611.0, true),
        (16_777_215.0, true),
        (8_388_608.0, false),
        (16_777_216.0, false),
        (16_777_218.0, false),
    ] {
        let got = binary(PointwiseOp::Pow, Scalars::with_scalar_y(0.0, exponent));
        for (i, value) in got.iter().enumerate() {
            let magnitude = if x[i].abs() > 1.0 { f32::INFINITY } else { 0.0 };
            let want = if x[i] < 0.0 && odd {
                -magnitude
            } else {
                magnitude
            };
            assert_eq!(
                value.to_bits(),
                want.to_bits(),
                "pow({}, {exponent}) with an {} exponent: got {value}, want {want}",
                x[i],
                if odd { "odd" } else { "even" }
            );
        }
    }

    // 4. sigmoid_backward, addcmul, addcdiv and lerp: the products and sums
    //    are chains of correctly rounded steps in the CPU's order, so
    //    sigmoid_backward and addcmul are held bitwise to an f32 evaluation
    //    in that order; addcdiv (a division) and lerp (the CPU fuses its
    //    multiply-add) to the tolerance
    let backward = binary(PointwiseOp::SigmoidBackward, Scalars::new(0.0));
    let cmul = ternary(PointwiseOp::Addcmul, Scalars::new(0.7));
    let cdiv = ternary(PointwiseOp::Addcdiv, Scalars::new(0.7));
    for (i, ((grad, mul), div)) in backward.iter().zip(&cmul).zip(&cdiv).enumerate() {
        assert_eq!(
            grad.to_bits(),
            (x[i] * (1.0 - y[i]) * y[i]).to_bits(),
            "sigmoid_backward[{i}]"
        );
        assert_eq!(
            mul.to_bits(),
            (x[i] + 0.7 * y[i] * z[i]).to_bits(),
            "addcmul[{i}]"
        );
        // the f32 alpha 0.7 is what the block carries; the reference uses its exact value
        let want = f64::from(x[i]) + f64::from(0.7f32) * f64::from(y[i]) / f64::from(z[i]);
        assert!(
            matches_reference(*div, want),
            "addcdiv[{i}]: got {div}, want {want}"
        );
    }
    for weight in [0.3f32, 0.8, 0.0, 1.0] {
        let got = binary(PointwiseOp::Lerp, Scalars::new(weight));
        for (i, value) in got.iter().enumerate() {
            let (start, end, w) = (f64::from(x[i]), f64::from(y[i]), f64::from(weight));
            let want = if w.abs() < 0.5 {
                start + w * (end - start)
            } else {
                end - (end - start) * (1.0 - w)
            };
            assert!(
                matches_reference(*value, want),
                "lerp({weight})[{i}]: got {value}, want {want}"
            );
        }
        // the two branches are exact at the ends: weight 0 is x, weight 1 is y
        if weight == 0.0 {
            assert_eq!(got, x, "lerp(0) is the start");
        }
        if weight == 1.0 {
            assert_eq!(got, y, "lerp(1) is the end");
        }
    }

    // 5. sin and cos at the AKV RoPE angles (T = 1024 positions by the 32
    //    inverse frequencies of head dimension 64 at theta 1e4, as
    //    akv_model._rope_tables builds them in f32: 32,768 angles up to
    //    1023 rad, 17,532 of them above pi), plus 1000 rad, the old split's
    //    edge at 12867, 1e5 and the top of the domain at SINCOS_MAX, against
    //    f64 sin/cos of the same f32 inputs; no result may leave [-1, 1]
    let inverse: Vec<f32> = (0..32)
        .map(|i| 1.0f32 / 10000f32.powf((2 * i) as f32 / 64.0))
        .collect();
    let mut angles: Vec<f32> = (0..1024u32)
        .flat_map(|t| inverse.iter().map(move |f| t as f32 * f))
        .collect();
    assert_eq!(angles.len(), 32_768);
    assert!(angles.iter().filter(|a| **a > std::f32::consts::PI).count() > 17_000);
    let fixed = [
        1000.0f32,
        -1000.0,
        1023.0,
        1_011.592_83, // the f32 nearest a multiple of pi below 1024 (1011.5928344726562): sin is 1.7e-8
        1e-3,
        -1e-3,
        0.0,
        -0.0,
        std::f32::consts::PI,
        std::f32::consts::FRAC_PI_2,
        // the old three-word split's edge, and the top of the five-word one's
        // domain: 12867 was where k*DP2 stopped being exact until 2026-09-07,
        // and SINCOS_MAX is the largest float32 the widened split reduces
        12_867.0,
        100_000.0,
        -100_000.0,
        SINCOS_MAX,
        -SINCOS_MAX,
    ];
    angles.extend(fixed);
    let na = angles.len();
    let ba = alloc(na);
    let ba_out = alloc(na);
    upload(ba, &angles);
    let ray = StridedView::contiguous(&[na as u64]).unwrap();
    let mut report = Vec::new();
    for (op, name) in [(PointwiseOp::Sin, "sin"), (PointwiseOp::Cos, "cos")] {
        upload(ba_out, &vec![f32::NAN; na]);
        let rc = run(
            op,
            &[na as u64],
            [ray; OPERANDS],
            Scalars::new(0.0),
            [ba, ba, ba, ba_out],
        );
        assert_eq!(rc, ffi::ACK_OK, "{name}: {}", last_error());
        let got = download(ba_out, na);
        let mut worst_abs = 0f64;
        let mut worst_rel = 0f64;
        let mut worst_ratio = 0f64;
        let mut host_f32_bitwise = 0usize;
        for (i, value) in got.iter().enumerate() {
            let xd = f64::from(angles[i]);
            let want = if op == PointwiseOp::Sin {
                xd.sin()
            } else {
                xd.cos()
            };
            let error = (f64::from(*value) - want).abs();
            assert!(
                within_tolerance(*value, want),
                "{name}[{i}]: x {}, got {value}, want {want}, error {error:.3e}",
                angles[i]
            );
            assert!(
                (-1.0..=1.0).contains(value),
                "{name}[{i}]: x {} gave {value}, which is not a {name} value",
                angles[i]
            );
            worst_abs = worst_abs.max(error);
            worst_ratio = worst_ratio.max(error / (1e-7 + 1e-4 * want.abs()));
            if want.abs() > 1e-3 {
                worst_rel = worst_rel.max(error / want.abs());
            }
            let host = if op == PointwiseOp::Sin {
                angles[i].sin()
            } else {
                angles[i].cos()
            };
            host_f32_bitwise += usize::from(host.to_bits() == value.to_bits());
        }
        // the header's claims at the fixed points: sin(+-0) keeps its sign,
        // cos(0) is 1 exactly
        let at = |value: f32| {
            got[32_768
                + fixed
                    .iter()
                    .position(|v| v.to_bits() == value.to_bits())
                    .unwrap()]
        };
        if op == PointwiseOp::Sin {
            assert_eq!(at(0.0).to_bits(), 0.0f32.to_bits(), "sin(0)");
            assert_eq!(at(-0.0).to_bits(), (-0.0f32).to_bits(), "sin(-0)");
            eprintln!(
                "pointwise sin(1000 rad) on the kit: {} (f64 reference {})",
                at(1000.0),
                f64::from(1000.0f32).sin()
            );
        } else {
            assert_eq!(at(0.0), 1.0, "cos(0)");
            eprintln!(
                "pointwise cos(1000 rad) on the kit: {} (f64 reference {})",
                at(1000.0),
                f64::from(1000.0f32).cos()
            );
        }
        report.push(format!(
            "{name} over {na} angles: max abs error {worst_abs:.3e}, max relative error {worst_rel:.3e} where |ref| > 1e-3, worst share of the family tolerance {worst_ratio:.4}, bitwise this host's f32 {name} at {host_f32_bitwise}/{na}"
        ));
    }
    for line in &report {
        eprintln!("pointwise {line}");
    }

    // 5b. beyond the reduction's domain, and for an infinity or a NaN, both
    //     functions must return NaN -- loud, never a plausible number. The
    //     three-word split this replaced returned -26.17 for sin(2^28),
    //     -1.59e34 for sin(1e12) and -inf for sin(1e15), every one of them
    //     outside sin's own range and counted as a kit product (2026-09-07).
    #[allow(clippy::excessive_precision)] // the exact decimal of 0x48c90fc2
    let beyond = [
        411774.0625f32, // the first float32 above SINCOS_MAX
        411_775.0,
        1e6,
        268_435_456.0, // 2^28
        1e12,
        1e15,
        3.4e38,
        -1e6,
        -3.4e38,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ];
    assert!(
        beyond[..9].iter().all(|v| v.abs() > SINCOS_MAX),
        "every finite entry is outside the domain"
    );
    let nb = beyond.len();
    let bb = alloc(nb);
    let bb_out = alloc(nb);
    upload(bb, &beyond);
    let bline = StridedView::contiguous(&[nb as u64]).unwrap();
    for (op, name) in [(PointwiseOp::Sin, "sin"), (PointwiseOp::Cos, "cos")] {
        upload(bb_out, &vec![0.0f32; nb]);
        let rc = run(
            op,
            &[nb as u64],
            [bline; OPERANDS],
            Scalars::new(0.0),
            [bb, bb, bb, bb_out],
        );
        assert_eq!(
            rc,
            ffi::ACK_OK,
            "{name} beyond the domain: {}",
            last_error()
        );
        for (i, value) in download(bb_out, nb).iter().enumerate() {
            assert!(
                value.is_nan(),
                "{name}({}) is outside the reduction's domain and must be NaN, got {value}",
                beyond[i]
            );
        }
    }
    eprintln!(
        "pointwise sin/cos domain: exact to |x| <= {SINCOS_MAX}, NaN above it ({} arguments checked)",
        beyond.len()
    );

    // 6. clamp: bounds, an absent bound, a lower bound above the upper (every
    //    element becomes the upper, as torch), a NaN kept, infinities bounded
    let c = [
        -1.0f32,
        -0.4,
        0.0,
        0.2,
        0.7,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
    ];
    let bc = alloc(c.len());
    let bc_out = alloc(c.len());
    upload(bc, &c);
    let short = StridedView::contiguous(&[c.len() as u64]).unwrap();
    for (low, high) in [
        (Some(-0.5f32), None),
        (None, Some(0.25f32)),
        (Some(-0.5), Some(0.25)),
        (Some(0.3), Some(0.1)),
        (None, None),
    ] {
        upload(bc_out, &vec![7.0; c.len()]);
        let rc = run(
            PointwiseOp::Clamp,
            &[c.len() as u64],
            [short; OPERANDS],
            Scalars::clamp(low, high),
            [bc, bc, bc, bc_out],
        );
        assert_eq!(rc, ffi::ACK_OK, "clamp: {}", last_error());
        let got = download(bc_out, c.len());
        let lo = low.unwrap_or(f32::NEG_INFINITY);
        let hi = high.unwrap_or(f32::INFINITY);
        for (i, value) in got.iter().enumerate() {
            let want = clamp_reference(c[i], lo, hi);
            assert!(
                value.to_bits() == want.to_bits() || (value.is_nan() && want.is_nan()),
                "clamp({low:?}, {high:?})[{i}]: x {}, got {value}, want {want}",
                c[i]
            );
        }
        if low == Some(0.3) {
            assert!(
                got.iter().take(5).all(|v| *v == 0.1),
                "min above max gives max"
            );
        }
    }

    // 7. iota over 1,000 elements (start 0.5, step 0.25: every value exact on
    //    both sides) and triu over the fixture's first 5x7 matrix and the
    //    whole 3x5x7 batch (the last two dimensions are the matrix)
    upload(bs_out, &vec![f32::NAN; ns]);
    let rc = run(
        PointwiseOp::Iota,
        &[1000],
        [StridedView::contiguous(&[1000]).unwrap(); OPERANDS],
        Scalars::iota(0.5, 0.25),
        [bs, bs, bs, bs_out],
    );
    assert_eq!(rc, ffi::ACK_OK, "iota: {}", last_error());
    // a download is the whole buffer; the ramp is its first 1,000 elements
    let ramp = download(bs_out, ns);
    for (i, value) in ramp.iter().take(1000).enumerate() {
        assert_eq!(
            value.to_bits(),
            (0.5 + i as f32 * 0.25).to_bits(),
            "iota[{i}]"
        );
    }
    for diagonal in [-1i32, 0, 2] {
        upload(bout, &poison);
        let matrix = StridedView::contiguous(&[5, 7]).unwrap();
        let rc = run(
            PointwiseOp::Triu,
            &[5, 7],
            [matrix; OPERANDS],
            Scalars::triu(diagonal),
            [bx, bx, bx, bout],
        );
        assert_eq!(rc, ffi::ACK_OK, "triu({diagonal}): {}", last_error());
        let got = download(bout, n);
        assert!(
            got[35..].iter().all(|v| v.is_nan()),
            "the rest is untouched"
        );
        for (i, value) in got.iter().take(35).enumerate() {
            let (row, column) = ((i / 7) as i64, (i % 7) as i64);
            let want = if column - row >= i64::from(diagonal) {
                x[i]
            } else {
                0.0
            };
            assert_eq!(
                value.to_bits(),
                want.to_bits(),
                "triu({diagonal})[{row}][{column}]"
            );
        }
    }
    let batched = unary(PointwiseOp::Triu, Scalars::triu(0));
    for (i, value) in batched.iter().enumerate() {
        let (row, column) = (((i % 35) / 7) as i64, (i % 7) as i64);
        let want = if column >= row { x[i] } else { 0.0 };
        assert_eq!(value.to_bits(), want.to_bits(), "batched triu[{i}]");
    }
    // triu over one dimension is refused by name before any dispatch
    let before = download(bs_out, ns);
    let mut one_dimensional = block(
        PointwiseOp::Iota,
        &[1000],
        [StridedView::contiguous(&[1000]).unwrap(); OPERANDS],
        Scalars::triu(0),
    );
    one_dimensional[..4].copy_from_slice(&(PointwiseOp::Triu as u32).to_le_bytes());
    let mut ms = 0f64;
    let rc = unsafe {
        ffi::ack_pointwise(
            dev,
            bs,
            bs,
            bs,
            bs_out,
            one_dimensional.as_ptr(),
            one_dimensional.len(),
            &mut ms,
        )
    };
    assert_eq!(rc, ffi::ACK_ERR_SHAPE, "{}", last_error());
    assert!(
        last_error().contains("pointwise-triu-needs-two-dimensions"),
        "{}",
        last_error()
    );
    // compared as bits: the buffer's tail is the NaN poison, which is not equal to itself
    let after = download(bs_out, ns);
    assert_eq!(
        after.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        before.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "a refusal dispatches nothing"
    );

    for buf in [
        bx, by, bz, bout, bs, bs_out, ba, ba_out, bb, bb_out, bc, bc_out,
    ] {
        assert_eq!(
            ffi::ack_buffer_free(dev, buf),
            ffi::ACK_OK,
            "{}",
            last_error()
        );
    }
    assert_eq!(ffi::ack_close(dev), ffi::ACK_OK, "{}", last_error());
}
