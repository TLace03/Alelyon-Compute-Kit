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
//! IEEE steps. Schema 4 (2026-09-22) adds exp within the tolerance and at its
//! overflow and underflow edges, round, the six comparisons into bool storage,
//! the logical operations over bool storage, where with a broadcast zero, and
//! the two storage refusals. Schema 6 (2026-09-28) adds tanh, log and xlogy,
//! held BIT FOR BIT to a host emulation of the shader's own steps and to their
//! measured ulp bounds against float64, over every 1,024th float32 of both
//! signs. Schema 8 (2026-10-16) adds minimum, maximum, relu and
//! threshold_backward, held BIT FOR BIT to a host emulation of their rules over
//! every ordered pair of an edge set (NaN payloads, signed zeros, subnormals,
//! infinities). The host-only checks need no device; the operations need one
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

/// A block over roomy declared storages, as `block`, with schema 4's storage
/// bits (bf16 or bool per slot) in its flags word.
fn block_with_storage(
    op: PointwiseOp,
    shape: &[u64],
    views: [StridedView; OPERANDS],
    scalars: Scalars,
    storage_flags: u32,
) -> [u8; PUSH_BYTES] {
    let storage = [
        Storage::new(1, u64::MAX / 4),
        Storage::new(2, u64::MAX / 4),
        Storage::new(3, u64::MAX / 4),
        Storage::new(4, u64::MAX / 4),
    ];
    PointwisePlan::new_with_storage_flags(op, shape, views, scalars, storage, storage_flags)
        .unwrap()
        .push_constants()
}

#[test]
fn schema_four_operations_match_the_host_and_bool_storage_is_one_byte_per_element() {
    use alelyon_compute_kit::pointwise_ops::{FLAG_BF16, FLAG_U8};
    let dev = ffi::ack_open();
    if dev.is_null() {
        if std::env::var_os("ACK_REQUIRE_DEVICE").is_some() {
            panic!(
                "pointwise ABI (schema 4): ACK_REQUIRE_DEVICE is set and ack_open failed: {}",
                last_error()
            );
        }
        eprintln!(
            "UNMEASURED here (pointwise ABI, schema 4): {}",
            last_error()
        );
        return;
    }
    let alloc = |bytes: usize| {
        let b = ffi::ack_buffer_alloc(dev, bytes as u64);
        assert!(!b.is_null(), "{}", last_error());
        b
    };
    let upload = |buf, data: &[u8]| {
        let rc = unsafe { ffi::ack_upload(dev, buf, data.as_ptr(), data.len()) };
        assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
    };
    let download = |buf, bytes: usize| {
        let mut out = vec![0u8; bytes];
        let rc = unsafe { ffi::ack_download(dev, buf, out.as_mut_ptr(), out.len()) };
        assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
        out
    };
    let f32_bytes = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    let f32_from = |b: &[u8]| {
        b.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect::<Vec<f32>>()
    };
    let run = |op: PointwiseOp,
               shape: &[u64],
               scalars: Scalars,
               flags: u32,
               buffers: [*const Buffer; OPERANDS]|
     -> i32 {
        let contiguous = StridedView::contiguous(shape).unwrap();
        let plan = block_with_storage(op, shape, [contiguous; OPERANDS], scalars, flags);
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

    // exp: the family tolerance everywhere, and the shader's edges exactly
    #[allow(clippy::excessive_precision)]
    let mut x: Vec<f32> = vec![
        -200.0,
        -104.0,
        -103.97208404541016,
        -103.97207641601562,
        -87.0,
        -1.0,
        -0.0,
        0.0,
        1e-8,
        1.0,
        10.0,
        88.0,
        88.72283172607422,
        88.72283935546875,
        100.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ];
    x.extend((0..4078).map(|i| -104.0 + (i as f32) * (193.0 / 4078.0)));
    let n = x.len();
    let bx = alloc(n * 4);
    let bout = alloc(n * 4);
    upload(bx, &f32_bytes(&x));
    let rc = run(
        PointwiseOp::Exp,
        &[n as u64],
        Scalars::new(0.0),
        0,
        [bx, bx, bx, bout],
    );
    assert_eq!(rc, ffi::ACK_OK, "exp: {}", last_error());
    let got = f32_from(&download(bout, n * 4));
    for (i, (&xi, &gi)) in x.iter().zip(&got).enumerate() {
        let want = f64::from(xi).exp();
        if xi.is_nan() {
            assert!(gi.is_nan(), "exp(NaN) at {i}: {gi}");
        } else if (want as f32).is_infinite() {
            assert!(
                gi.is_infinite() && gi > 0.0,
                "exp({xi}) at {i}: {gi}, want +inf"
            );
        } else {
            assert!(
                within_tolerance(gi, want),
                "exp({xi}) at {i}: {gi}, want {want}"
            );
        }
    }
    assert!(
        got[12].is_finite(),
        "88.72283172607422 is the largest float32 with a finite exp: {}",
        got[12]
    );
    assert!(got[13].is_infinite(), "88.72283935546875 overflows");
    assert_eq!(got[2], 0.0, "exp rounds to 0 at -103.97208404541016");

    // round: halves to even, bitwise against the host's ties-to-even
    let r: Vec<f32> = vec![
        -2.5,
        -1.5,
        -0.5,
        -0.4,
        -0.0,
        0.0,
        0.4,
        0.5,
        1.5,
        2.5,
        3.5,
        8388607.5,
        8388608.0,
        1e20,
        f32::INFINITY,
        f32::NEG_INFINITY,
    ];
    let nr = r.len();
    let br = alloc(nr * 4);
    let br_out = alloc(nr * 4);
    upload(br, &f32_bytes(&r));
    let rc = run(
        PointwiseOp::Round,
        &[nr as u64],
        Scalars::new(0.0),
        0,
        [br, br, br, br_out],
    );
    assert_eq!(rc, ffi::ACK_OK, "round: {}", last_error());
    let got = f32_from(&download(br_out, nr * 4));
    for (&ri, &gi) in r.iter().zip(&got) {
        assert_eq!(
            gi.to_bits(),
            ri.round_ties_even().to_bits(),
            "round({ri}) = {gi}"
        );
    }

    // comparisons into bool storage: one byte per element, 1 or 0, IEEE on a NaN
    let a: Vec<f32> = vec![f32::NAN, -1.0, 0.0, -0.0, 0.1, 7.0, -8.0, 3.0];
    let b: Vec<f32> = vec![f32::NAN, 1.0, -0.0, 0.0, 0.1, 6.0, -8.0, f32::NAN];
    let na = a.len();
    let ba = alloc(na * 4);
    let bb = alloc(na * 4);
    let bmask = alloc(na);
    upload(ba, &f32_bytes(&a));
    upload(bb, &f32_bytes(&b));
    type Comparison = fn(f32, f32) -> bool;
    let cases: [(PointwiseOp, Comparison); 6] = [
        (PointwiseOp::Gt, |p, q| p > q),
        (PointwiseOp::Lt, |p, q| p < q),
        (PointwiseOp::Ge, |p, q| p >= q),
        (PointwiseOp::Le, |p, q| p <= q),
        (PointwiseOp::Eq, |p, q| p == q),
        (PointwiseOp::Ne, |p, q| p != q),
    ];
    for (op, host) in cases {
        upload(bmask, &vec![0xAB; na]);
        let rc = run(
            op,
            &[na as u64],
            Scalars::new(0.0),
            FLAG_U8[3],
            [ba, bb, ba, bmask],
        );
        assert_eq!(rc, ffi::ACK_OK, "{op:?}: {}", last_error());
        let got = download(bmask, na);
        let want: Vec<u8> = a
            .iter()
            .zip(&b)
            .map(|(&p, &q)| u8::from(host(p, q)))
            .collect();
        assert_eq!(got, want, "{op:?} tensor-tensor");
        upload(bmask, &vec![0xAB; na]);
        let rc = run(
            op,
            &[na as u64],
            Scalars::with_scalar_y(0.0, 0.1),
            FLAG_U8[3],
            [ba, ba, ba, bmask],
        );
        assert_eq!(rc, ffi::ACK_OK, "{op:?} scalar: {}", last_error());
        let got = download(bmask, na);
        let want: Vec<u8> = a.iter().map(|&p| u8::from(host(p, 0.1))).collect();
        assert_eq!(got, want, "{op:?} against the scalar 0.1");
    }

    // logical operations on bool storage in every slot, the unread z bound to x
    let p8: Vec<u8> = vec![0, 1, 0, 1, 1, 0, 1, 0];
    let q8: Vec<u8> = vec![0, 0, 1, 1, 1, 1, 0, 0];
    let bp = alloc(na);
    let bq = alloc(na);
    let bl = alloc(na);
    upload(bp, &p8);
    upload(bq, &q8);
    let all_bool = FLAG_U8[0] | FLAG_U8[1] | FLAG_U8[2] | FLAG_U8[3];
    type Logical = fn(bool, bool) -> bool;
    let logical: [(PointwiseOp, Logical); 3] = [
        (PointwiseOp::LogicalAnd, |p, q| p && q),
        (PointwiseOp::LogicalOr, |p, q| p || q),
        (PointwiseOp::LogicalXor, |p, q| p != q),
    ];
    for (op, host) in logical {
        let rc = run(
            op,
            &[na as u64],
            Scalars::new(0.0),
            all_bool,
            [bp, bq, bp, bl],
        );
        assert_eq!(rc, ffi::ACK_OK, "{op:?}: {}", last_error());
        let want: Vec<u8> = p8
            .iter()
            .zip(&q8)
            .map(|(&p, &q)| u8::from(host(p != 0, q != 0)))
            .collect();
        assert_eq!(download(bl, na), want, "{op:?}");
    }
    let rc = run(
        PointwiseOp::LogicalNot,
        &[na as u64],
        Scalars::new(0.0),
        all_bool,
        [bp, bp, bp, bl],
    );
    assert_eq!(rc, ffi::ACK_OK, "not: {}", last_error());
    assert_eq!(
        download(bl, na),
        p8.iter().map(|&p| u8::from(p == 0)).collect::<Vec<u8>>()
    );

    // where: a bool condition selecting between f32 values; a clamp's backward
    // is exactly this, with z the broadcast zero
    let bw = alloc(na * 4);
    let bzero = alloc(4);
    upload(bzero, &f32_bytes(&[0.0]));
    let broadcast = StridedView::new(0, [0; MAX_NDIM]);
    let contiguous = StridedView::contiguous(&[na as u64]).unwrap();
    let plan = block_with_storage(
        PointwiseOp::Where,
        &[na as u64],
        [contiguous, contiguous, broadcast, contiguous],
        Scalars::new(0.0),
        FLAG_U8[0],
    );
    let mut ms = 0f64;
    let rc =
        unsafe { ffi::ack_pointwise(dev, bp, ba, bzero, bw, plan.as_ptr(), plan.len(), &mut ms) };
    assert_eq!(rc, ffi::ACK_OK, "where: {}", last_error());
    let got = f32_from(&download(bw, na * 4));
    for i in 0..na {
        let want = if p8[i] != 0 { a[i] } else { 0.0 };
        assert_eq!(got[i].to_bits(), want.to_bits(), "where at {i}");
    }

    // refusals by name: a slot with two storages, and a combination with no module
    let contiguous_all = [contiguous; OPERANDS];
    for (flags, name) in [
        (FLAG_BF16[0] | FLAG_U8[0], "pointwise-storage"),
        (FLAG_U8[1], "pointwise-storage"),
    ] {
        let mut words = block_with_storage(
            PointwiseOp::Gt,
            &[na as u64],
            contiguous_all,
            Scalars::new(0.0),
            0,
        );
        words[12..16].copy_from_slice(&flags.to_le_bytes());
        let rc = unsafe {
            ffi::ack_pointwise(dev, ba, bb, ba, bmask, words.as_ptr(), words.len(), &mut ms)
        };
        assert_eq!(rc, ffi::ACK_ERR_SHAPE, "flags {flags:#x}: {}", last_error());
        assert!(last_error().contains(name), "{}", last_error());
    }

    for buf in [bx, bout, br, br_out, ba, bb, bmask, bp, bq, bl, bw, bzero] {
        assert_eq!(
            ffi::ack_buffer_free(dev, buf),
            ffi::ACK_OK,
            "{}",
            last_error()
        );
    }
    assert_eq!(ffi::ack_close(dev), ffi::ACK_OK, "{}", last_error());
}

/// torch's `silu_backward` in f64: `s = 1 / (1 + exp(-x))`, `dy * s * (1 + x * (1 - s))`
fn silu_backward_reference(dy: f32, x: f32) -> f64 {
    let (dy, x) = (f64::from(dy), f64::from(x));
    let s = 1.0 / (1.0 + (-x).exp());
    dy * s * (1.0 + x * (1.0 - s))
}

#[test]
fn schema_five_operations_match_a_host_f64_reference_through_packed_halves() {
    let dev = ffi::ack_open();
    if dev.is_null() {
        if std::env::var_os("ACK_REQUIRE_DEVICE").is_some() {
            panic!(
                "pointwise ABI (schema 5): ACK_REQUIRE_DEVICE is set and ack_open failed: {}",
                last_error()
            );
        }
        eprintln!(
            "UNMEASURED here (pointwise ABI, schema 5): {}",
            last_error()
        );
        return;
    }
    assert_eq!(PointwiseOp::try_from(39), Ok(PointwiseOp::SwigluGradGate));
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
               buffers: [*const Buffer; OPERANDS]|
     -> i32 {
        let plan = block(op, shape, views, Scalars::new(0.0));
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

    // The gated linear unit's layout: ONE packed [rows, 2d] projection whose
    // first half is u and second half the gate g, read through strided views
    // (offset 0 and offset d, row stride 2d), and one packed [rows, 2d]
    // gradient whose halves are written the same way, so no cat is needed.
    // The gate covers [-30, 30] and the extremes where exp overflows.
    let (rows, d) = (9usize, 128usize);
    let mut gate: Vec<f32> = (0..rows * d)
        .map(|i| -30.0 + 60.0 * i as f32 / (rows * d - 1) as f32)
        .collect();
    gate[..8].copy_from_slice(&[-200.0, 200.0, -88.5, 88.5, 0.0, -0.0, 1e-6, -1e-6]);
    let up: Vec<f32> = (0..rows * d)
        .map(|i| ((i * 7919) % 97) as f32 / 13.0 - 3.5)
        .collect();
    let grad: Vec<f32> = (0..rows * d)
        .map(|i| ((i * 104_729) % 89) as f32 / 11.0 - 4.0)
        .collect();
    let mut packed = vec![0f32; rows * 2 * d];
    for r in 0..rows {
        packed[r * 2 * d..r * 2 * d + d].copy_from_slice(&up[r * d..(r + 1) * d]);
        packed[r * 2 * d + d..(r + 1) * 2 * d].copy_from_slice(&gate[r * d..(r + 1) * d]);
    }
    let bpacked = alloc(rows * 2 * d);
    let bgrad = alloc(rows * d);
    let bout = alloc(rows * d);
    let bpacked_grad = alloc(rows * 2 * d);
    upload(bpacked, &packed);
    upload(bgrad, &grad);
    let shape = [rows as u64, d as u64];
    let half = |offset: u64| StridedView::new(offset, [2 * d as u64, 1, 0, 0]);
    let dense = StridedView::contiguous(&shape).unwrap();

    // 38: u * silu(g), both read from the packed projection
    upload(bout, &vec![f32::NAN; rows * d]);
    let rc = run(
        PointwiseOp::SiluMul,
        &shape,
        [half(0), half(d as u64), dense, dense],
        [bpacked, bpacked, bpacked, bout],
    );
    assert_eq!(rc, ffi::ACK_OK, "silu_mul: {}", last_error());
    let got = download(bout, rows * d);
    for i in 0..rows * d {
        let g = f64::from(gate[i]);
        let want = f64::from(up[i]) * (g / (1.0 + (-g).exp()));
        assert!(
            matches_reference(got[i], want),
            "silu_mul[{i}]: u {} g {}, got {}, want {want}",
            up[i],
            gate[i],
            got[i]
        );
    }

    // 37: silu_backward(grad, g), dense operands
    let bgate = alloc(rows * d);
    upload(bgate, &gate);
    upload(bout, &vec![f32::NAN; rows * d]);
    let rc = run(
        PointwiseOp::SiluBackward,
        &shape,
        [dense; OPERANDS],
        [bgrad, bgate, bgrad, bout],
    );
    assert_eq!(rc, ffi::ACK_OK, "silu_backward: {}", last_error());
    let got = download(bout, rows * d);
    for i in 0..rows * d {
        let want = silu_backward_reference(grad[i], gate[i]);
        assert!(
            matches_reference(got[i], want),
            "silu_backward[{i}]: dy {} x {}, got {}, want {want}",
            grad[i],
            gate[i],
            got[i]
        );
    }

    // The packed gradient: its u half is grad * silu(g) (38 again, the gate read
    // from the packed projection) and its gate half grad * u * dsilu(g) (39),
    // each WRITTEN through a half view of the one packed output
    upload(bpacked_grad, &vec![f32::NAN; rows * 2 * d]);
    let rc = run(
        PointwiseOp::SiluMul,
        &shape,
        [dense, half(d as u64), dense, half(0)],
        [bgrad, bpacked, bgrad, bpacked_grad],
    );
    assert_eq!(rc, ffi::ACK_OK, "grad_u: {}", last_error());
    let rc = run(
        PointwiseOp::SwigluGradGate,
        &shape,
        [dense, half(0), half(d as u64), half(d as u64)],
        [bgrad, bpacked, bpacked, bpacked_grad],
    );
    assert_eq!(rc, ffi::ACK_OK, "swiglu_grad_gate: {}", last_error());
    let got = download(bpacked_grad, rows * 2 * d);
    for r in 0..rows {
        for c in 0..d {
            let i = r * d + c;
            let g = f64::from(gate[i]);
            let want_u = f64::from(grad[i]) * (g / (1.0 + (-g).exp()));
            let want_g = f64::from(up[i]) * silu_backward_reference(grad[i], gate[i]);
            let (got_u, got_g) = (got[r * 2 * d + c], got[r * 2 * d + d + c]);
            assert!(
                matches_reference(got_u, want_u),
                "grad_u[{r},{c}]: got {got_u}, want {want_u}"
            );
            assert!(
                matches_reference(got_g, want_g),
                "grad_gate[{r},{c}]: got {got_g}, want {want_g}"
            );
        }
    }
    // the extremes are finite: a saturated gate passes the gradient through
    // (g = 200) or stops it (g = -200), and never makes a NaN
    let row0 = &got[..2 * d];
    assert!(
        row0.iter().all(|v| v.is_finite()),
        "row 0: {:?}",
        &row0[..8]
    );
    assert_eq!(row0[d], 0.0, "grad_gate at g = -200 is zero");

    for b in [bpacked, bgrad, bout, bpacked_grad, bgate] {
        assert_eq!(
            ffi::ack_buffer_free(dev, b),
            ffi::ACK_OK,
            "{}",
            last_error()
        );
    }
    assert_eq!(ffi::ack_close(dev), ffi::ACK_OK, "{}", last_error());
}

// ---- schema 6: tanh, log and xlogy (2026-09-28) ------------------------------------

/// The shader's exp_torch, step for step in f32 (Rust never contracts a
/// multiply and an add), for arguments inside its finite range.
fn exp_torch_emulated(x: f32) -> f32 {
    // the shader's constants, by their bits
    let log2e = f32::from_bits(0x3fb8_aa3b);
    let c1 = f32::from_bits(0x3f31_8000);
    let c2 = f32::from_bits(0xb95e_8083);
    let p_coeffs = [
        0x3950_6967u32,
        0x3ab7_43ce,
        0x3c08_8908,
        0x3d2a_a9c1,
        0x3e2a_aaaa,
        0x3f00_0000,
    ]
    .map(f32::from_bits);
    let t = log2e * x + 0.5;
    let k = t.floor();
    let r = (x - k * c1) - k * c2;
    let z = r * r;
    let mut p = p_coeffs[0];
    for c in &p_coeffs[1..] {
        p = p * r + c;
    }
    let p = (p * z + r) + 1.0;
    let ki = k as i32;
    let k1 = ki >> 1;
    ldexp(ldexp(p, k1), ki - k1)
}

/// x * 2^e for a normal x and a result that stays normal: exact.
fn ldexp(x: f32, e: i32) -> f32 {
    x * f32::from_bits(((127 + e) as u32) << 23)
}

fn recip_newton_emulated(d: f32) -> f32 {
    let mut r = f32::from_bits(0x7EF3_11C3u32.wrapping_sub(d.to_bits()));
    for _ in 0..3 {
        r = r * (2.0 - d * r);
    }
    r
}

fn tanh_emulated(x: f32) -> f32 {
    let p_coeffs = [
        0xbbba_f0eau32,
        0x3ca9_134e,
        0xbd5c_1e2d,
        0x3e08_8393,
        0xbeaa_aa99,
    ]
    .map(f32::from_bits);
    if x.is_nan() {
        return x;
    }
    let bits = x.to_bits();
    let ax = f32::from_bits(bits & 0x7FFF_FFFF);
    if ax < f32::from_bits(0x3980_0000) {
        return x;
    }
    if ax >= TANH_SAT {
        return f32::from_bits((bits & 0x8000_0000) | 0x3F80_0000);
    }
    if ax < 0.625 {
        let s = x * x;
        let mut p = p_coeffs[0];
        for c in &p_coeffs[1..] {
            p = p * s + c;
        }
        return p * s * x + x;
    }
    let e = exp_torch_emulated(ax + ax);
    let q = recip_newton_emulated(e + 1.0) * 2.0;
    let m = 1.0 - q;
    f32::from_bits((bits & 0x8000_0000) | m.to_bits())
}

fn log_emulated(x: f32) -> f32 {
    let p_coeffs = [
        0x3d90_21bbu32,
        0xbdeb_d1b8,
        0x3def_251a,
        0xbdfe_5d4f,
        0x3e11_e9bf,
        0xbe2a_ae50,
        0x3e4c_ceac,
        0xbe7f_fffc,
        0x3eaa_aaaa,
    ]
    .map(f32::from_bits);
    if x.is_nan() {
        return x;
    }
    let bits = x.to_bits();
    if bits & 0x7FFF_FFFF == 0 {
        return f32::NEG_INFINITY;
    }
    if bits & 0x8000_0000 != 0 {
        return f32::NAN;
    }
    if bits == 0x7F80_0000 {
        return x;
    }
    let mut e = (bits >> 23) as i32;
    let mut man = bits & 0x7F_FFFF;
    if e == 0 {
        let shift = man.leading_zeros() as i32 - 8;
        man = (man << shift) & 0x7F_FFFF;
        e = -125 - shift;
    } else {
        e -= 126;
    }
    let m = f32::from_bits(0x3F00_0000 | man);
    let z = if m < f32::from_bits(0x3f35_04f3) {
        e -= 1;
        (m + m) - 1.0
    } else {
        m - 1.0
    };
    let zz = z * z;
    let mut p = p_coeffs[0];
    for c in &p_coeffs[1..] {
        p = p * z + c;
    }
    let mut y = p * z * zz;
    let fe = e as f32;
    if e != 0 {
        y += f32::from_bits(0xb95e_8083) * fe;
    }
    y += -0.5 * zz;
    let mut r = z + y;
    if e != 0 {
        r += f32::from_bits(0x3f31_8000) * fe;
    }
    r
}

fn xlogy_emulated(x: f32, y: f32) -> f32 {
    if y.is_nan() {
        return f32::NAN;
    }
    if x.to_bits() & 0x7FFF_FFFF == 0 {
        return 0.0;
    }
    x * log_emulated(y)
}

/// The smallest float32 whose tanh rounds to 1 (the shader's TANH_SAT,
/// 9.010913848876953).
const TANH_SAT: f32 = f32::from_bits(0x4110_2cb4);

/// |got - want| in units of the float32 ulp of the binade of |want|.
fn ulps(got: f32, want: f64) -> f64 {
    let e = want.abs().max(f64::from(f32::MIN_POSITIVE)).log2().floor();
    ((f64::from(got) - want).abs()) / 2f64.powf(e - 23.0)
}

#[test]
fn schema_six_tanh_log_and_xlogy_are_their_emulation_bit_for_bit_and_within_their_ulp_bounds() {
    let dev = ffi::ack_open();
    if dev.is_null() {
        if std::env::var_os("ACK_REQUIRE_DEVICE").is_some() {
            panic!(
                "pointwise ABI (schema 6): ACK_REQUIRE_DEVICE is set and ack_open failed: {}",
                last_error()
            );
        }
        eprintln!(
            "UNMEASURED here (pointwise ABI, schema 6): {}",
            last_error()
        );
        return;
    }
    // schema 6's last word; schema 8 appends 43-46 (its own test below)
    const { assert!(LAST_OP >= 42) };
    // Every 1,024th float32 bit pattern of both signs, which visits every
    // binade (subnormals, the infinities and NaNs included), plus each region
    // boundary and its neighbours, and the specials.
    let mut xs: Vec<f32> = (0u64..(1 << 22))
        .map(|i| f32::from_bits((i << 10) as u32 | ((i * 37) & 0x3FF) as u32))
        .collect();
    for edge in [
        f32::from_bits(0x3980_0000),
        0.625,
        TANH_SAT,
        1.0,
        f32::from_bits(0x3f35_04f3),
        2.0,
        0.5,
    ] {
        for k in -3i32..=3 {
            let b = (edge.to_bits() as i32 + k) as u32;
            xs.push(f32::from_bits(b));
            xs.push(-f32::from_bits(b));
        }
    }
    xs.extend([
        0.0,
        -0.0,
        f32::from_bits(1),
        -f32::from_bits(1),
        f32::MIN_POSITIVE,
        f32::MAX,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(0x7FC0_0ABC),
        f32::from_bits(0xFF80_0DEF),
    ]);
    let n = xs.len();
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
    let run = |op: PointwiseOp, scalars: Scalars, buffers: [*const Buffer; OPERANDS]| -> i32 {
        let shape = [n as u64];
        let dense = StridedView::contiguous(&shape).unwrap();
        let plan = block(op, &shape, [dense; OPERANDS], scalars);
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
    let bx = alloc(n);
    let by = alloc(n);
    let bout = alloc(n);
    upload(bx, &xs);
    // the same bits, or both NaN (a NaN's payload is the device's to choose
    // only where the operation makes a new one)
    let same =
        |got: f32, want: f32| got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan());

    // 40 tanh: the emulation's bits, and within 1.84 ulp of float64's tanh
    upload(bout, &vec![f32::NAN; n]);
    let rc = run(PointwiseOp::Tanh, Scalars::new(0.0), [bx, bx, bx, bout]);
    assert_eq!(rc, ffi::ACK_OK, "tanh: {}", last_error());
    let got = download(bout, n);
    let mut worst = 0f64;
    for (i, (&x, &g)) in xs.iter().zip(&got).enumerate() {
        let want = tanh_emulated(x);
        assert!(
            same(g, want),
            "tanh[{i}]({x:e} = {:#010x}): device {g:e} = {:#010x}, emulation {want:e} = {:#010x}",
            x.to_bits(),
            g.to_bits(),
            want.to_bits()
        );
        if x.is_finite() {
            let err = ulps(g, f64::from(x).tanh());
            worst = worst.max(err);
            assert!(err <= 1.84, "tanh({x:e}) = {g:e}: {err} ulp");
        }
    }
    eprintln!(
        "tanh over {n} values: max {worst:.3} ulp from float64, every value the emulation's bits"
    );
    assert_eq!(tanh_emulated(-0.0).to_bits(), 0x8000_0000);
    assert_eq!(tanh_emulated(f32::NEG_INFINITY), -1.0);

    // 41 log: the emulation's bits; within 0.83 ulp of float64's ln where the
    // result is finite; -inf at +-0, NaN below 0, +inf at +inf
    upload(bout, &vec![f32::NAN; n]);
    let rc = run(PointwiseOp::Log, Scalars::new(0.0), [bx, bx, bx, bout]);
    assert_eq!(rc, ffi::ACK_OK, "log: {}", last_error());
    let got = download(bout, n);
    let mut worst = 0f64;
    for (i, (&x, &g)) in xs.iter().zip(&got).enumerate() {
        let want = log_emulated(x);
        assert!(
            same(g, want),
            "log[{i}]({x:e} = {:#010x}): device {g:e} = {:#010x}, emulation {want:e} = {:#010x}",
            x.to_bits(),
            g.to_bits(),
            want.to_bits()
        );
        if x.is_finite() && x > 0.0 {
            let err = ulps(g, f64::from(x).ln());
            worst = worst.max(err);
            assert!(err <= 0.83, "log({x:e}) = {g:e}: {err} ulp");
        }
    }
    eprintln!(
        "log over {n} values: max {worst:.3} ulp from float64, every value the emulation's bits"
    );
    assert_eq!(log_emulated(1.0).to_bits(), 0, "log(1) is +0");
    assert_eq!(log_emulated(-0.0), f32::NEG_INFINITY);
    assert!(log_emulated(-1.0).is_nan() && log_emulated(f32::NEG_INFINITY).is_nan());
    assert_eq!(log_emulated(f32::INFINITY), f32::INFINITY);

    // 42 xlogy, y a tensor: x runs over moderate magnitudes of both signs and
    // the zeros (a product with a tiny x would test the device's subnormal
    // flushing, not the formula), y over the whole sweep
    let x2: Vec<f32> = (0..n)
        .map(|i| match i % 7 {
            0 => 0.0,
            1 => -0.0,
            k => (k as f32 - 3.5) * (1.0 + (i % 1013) as f32 / 1013.0),
        })
        .collect();
    upload(by, &xs);
    upload(bx, &x2);
    upload(bout, &vec![f32::NAN; n]);
    let rc = run(PointwiseOp::Xlogy, Scalars::new(0.0), [bx, by, bx, bout]);
    assert_eq!(rc, ffi::ACK_OK, "xlogy: {}", last_error());
    let got = download(bout, n);
    for (i, ((&x, &y), &g)) in x2.iter().zip(&xs).zip(&got).enumerate() {
        let want = xlogy_emulated(x, y);
        assert!(
            same(g, want),
            "xlogy[{i}]({x:e}, {y:e}): device {g:e} = {:#010x}, emulation {want:e} = {:#010x}",
            g.to_bits(),
            want.to_bits()
        );
    }
    // torch's rules at the edges
    assert!(
        xlogy_emulated(0.0, f32::NAN).is_nan(),
        "a NaN y wins over x = 0"
    );
    assert_eq!(xlogy_emulated(-0.0, 5.0).to_bits(), 0, "x = -0 gives +0");
    assert_eq!(
        xlogy_emulated(0.0, 0.0).to_bits(),
        0,
        "x = 0 gives 0 even at y = 0"
    );
    assert_eq!(xlogy_emulated(2.0, 0.0), f32::NEG_INFINITY);
    assert!(xlogy_emulated(1.0, -1.0).is_nan());
    // and y as the scalar b (flags bit 0)
    upload(bout, &vec![f32::NAN; n]);
    let rc = run(
        PointwiseOp::Xlogy,
        Scalars::with_scalar_y(0.0, 0.25),
        [bx, bx, bx, bout],
    );
    assert_eq!(rc, ffi::ACK_OK, "xlogy, scalar y: {}", last_error());
    let got = download(bout, n);
    for (i, (&x, &g)) in x2.iter().zip(&got).enumerate() {
        let want = xlogy_emulated(x, 0.25);
        assert!(
            same(g, want),
            "xlogy[{i}]({x:e}, 0.25): device {g:e}, emulation {want:e}"
        );
    }

    for b in [bx, by, bout] {
        assert_eq!(
            ffi::ack_buffer_free(dev, b),
            ffi::ACK_OK,
            "{}",
            last_error()
        );
    }
    assert_eq!(ffi::ack_close(dev), ffi::ACK_OK, "{}", last_error());
}

/// The schema-8 rules, on the bits (AKV MoE plan Amendment 27): a NaN is an
/// all-ones exponent with a nonzero fraction; `select_key` orders the other
/// words as their values, -0 and +0 one key.
fn nan_bits(u: u32) -> bool {
    (u & 0x7FFF_FFFF) > 0x7F80_0000
}

fn select_key(u: u32) -> u32 {
    if u & 0x7FFF_FFFF == 0 {
        0x8000_0000
    } else if u & 0x8000_0000 != 0 {
        !u
    } else {
        u | 0x8000_0000
    }
}

/// minimum / maximum as torch's CPU vectorised body answers them
fn min_max_rule(x: u32, y: u32, larger: bool) -> u32 {
    if nan_bits(x) || nan_bits(y) {
        return 0xFFFF_FFFF;
    }
    let (kx, ky) = (select_key(x), select_key(y));
    if (larger && kx > ky) || (!larger && kx < ky) {
        x
    } else {
        y
    }
}

fn relu_rule(x: u32) -> u32 {
    if x & 0x8000_0000 != 0 && x & 0x7FFF_FFFF != 0 && !nan_bits(x) {
        0
    } else {
        x
    }
}

fn threshold_backward_rule(grad: u32, input: u32, threshold: u32) -> u32 {
    if !nan_bits(input) && !nan_bits(threshold) && select_key(input) <= select_key(threshold) {
        0
    } else {
        grad
    }
}

#[test]
fn schema_eight_selections_are_their_rules_bit_for_bit() {
    // the rules against words torch's CPU was MEASURED to give (Amendment 27, K0)
    assert_eq!(min_max_rule(0x0000_0001, 0x8000_0001, false), 0x8000_0001);
    assert_eq!(min_max_rule(0xFF80_0000, 0x0000_0001, true), 0x0000_0001);
    assert_eq!(min_max_rule(0x8000_0000, 0x0000_0000, false), 0x0000_0000);
    assert_eq!(min_max_rule(0x0000_0000, 0x8000_0000, true), 0x8000_0000);
    assert_eq!(min_max_rule(0x7FA0_0001, 0x3F80_0000, false), 0xFFFF_FFFF);
    assert_eq!(relu_rule(0x8000_0000), 0x8000_0000);
    assert_eq!(relu_rule(0x8000_0001), 0);
    assert_eq!(relu_rule(0xFFA0_0002), 0xFFA0_0002);
    assert_eq!(
        threshold_backward_rule(0x3F80_0000, 0x0000_0001, 0),
        0x3F80_0000
    );
    assert_eq!(threshold_backward_rule(0x7FC0_0001, 0x8000_0000, 0), 0);
    assert_eq!(
        threshold_backward_rule(0xFFA0_0002, 0x7FC0_0005, 0),
        0xFFA0_0002
    );

    let dev = ffi::ack_open();
    if dev.is_null() {
        if std::env::var_os("ACK_REQUIRE_DEVICE").is_some() {
            panic!(
                "pointwise ABI (schema 8): ACK_REQUIRE_DEVICE is set and ack_open failed: {}",
                last_error()
            );
        }
        eprintln!(
            "UNMEASURED here (pointwise ABI, schema 8): {}",
            last_error()
        );
        return;
    }
    assert_eq!(LAST_OP, 46);
    let edges: [u32; 24] = [
        0x0000_0000,
        0x8000_0000,
        0x0000_0001,
        0x8000_0001,
        0x007F_FFFF,
        0x807F_FFFF,
        0x0080_0000,
        0x8080_0000,
        0x3F80_0000,
        0xBF80_0000,
        0x3F00_0000,
        0xBF00_0000,
        0x7F7F_FFFF,
        0xFF7F_FFFF,
        0x7F80_0000,
        0xFF80_0000,
        0x7FC0_0000,
        0xFFC0_0000,
        0x7FC0_0001,
        0xFFC0_0ABC,
        0x7FA0_0001,
        0xFFA0_0002,
        0x4049_0FDB,
        0xC049_0FDB,
    ];
    // every ordered pair: x runs over the edges, y over the edges again
    let xs: Vec<u32> = edges
        .iter()
        .flat_map(|&x| edges.iter().map(move |_| x))
        .collect();
    let ys: Vec<u32> = edges.iter().flat_map(|_| edges.iter().copied()).collect();
    let n = xs.len();
    let alloc = || {
        let b = ffi::ack_buffer_alloc(dev, (n * 4) as u64);
        assert!(!b.is_null(), "{}", last_error());
        b
    };
    let upload = |buf, data: &[u32]| {
        let rc = unsafe { ffi::ack_upload(dev, buf, data.as_ptr().cast::<u8>(), data.len() * 4) };
        assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
    };
    let download = |buf| {
        let mut out = vec![0u32; n];
        let rc =
            unsafe { ffi::ack_download(dev, buf, out.as_mut_ptr().cast::<u8>(), out.len() * 4) };
        assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
        out
    };
    let run = |op: PointwiseOp, scalars: Scalars, buffers: [*const Buffer; OPERANDS]| -> i32 {
        let shape = [n as u64];
        let dense = StridedView::contiguous(&shape).unwrap();
        let plan = block(op, &shape, [dense; OPERANDS], scalars);
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
    let (bx, by, bout) = (alloc(), alloc(), alloc());
    upload(bx, &xs);
    upload(by, &ys);
    let mut checked = 0usize;
    let mut check = |name: &str, got: Vec<u32>, want: &dyn Fn(usize) -> u32| {
        for (i, &g) in got.iter().enumerate() {
            let w = want(i);
            assert_eq!(
                g, w,
                "{name}[{i}]({:#010x}, {:#010x}): device {g:#010x}, rule {w:#010x}",
                xs[i], ys[i]
            );
        }
        checked += got.len();
    };
    for (op, larger) in [(PointwiseOp::Minimum, false), (PointwiseOp::Maximum, true)] {
        upload(bout, &vec![0x7F80_0001; n]);
        let rc = run(op, Scalars::new(0.0), [bx, by, bx, bout]);
        assert_eq!(rc, ffi::ACK_OK, "{op:?}: {}", last_error());
        check(&format!("{op:?}"), download(bout), &|i| {
            min_max_rule(xs[i], ys[i], larger)
        });
    }
    upload(bout, &vec![0x7F80_0001; n]);
    let rc = run(PointwiseOp::Relu, Scalars::new(0.0), [bx, bx, bx, bout]);
    assert_eq!(rc, ffi::ACK_OK, "relu: {}", last_error());
    check("relu", download(bout), &|i| relu_rule(xs[i]));
    // x the gradient, y the input; thresholds 0, 0.5, -1, the smallest
    // subnormal, -0 and a NaN
    for threshold in [
        0x0000_0000u32,
        0x3F00_0000,
        0xBF80_0000,
        0x0000_0001,
        0x8000_0000,
        0x7FC0_0000,
    ] {
        upload(bout, &vec![0x7F80_0001; n]);
        let rc = run(
            PointwiseOp::ThresholdBackward,
            Scalars::new(f32::from_bits(threshold)),
            [bx, by, bx, bout],
        );
        assert_eq!(rc, ffi::ACK_OK, "threshold_backward: {}", last_error());
        check(
            &format!("threshold_backward(threshold {threshold:#010x})"),
            download(bout),
            &|i| threshold_backward_rule(xs[i], ys[i], threshold),
        );
    }
    eprintln!("schema 8: {checked} selections over {n} ordered edge pairs, every word the rule's");
    for b in [bx, by, bout] {
        assert_eq!(
            ffi::ack_buffer_free(dev, b),
            ffi::ACK_OK,
            "{}",
            last_error()
        );
    }
    assert_eq!(ffi::ack_close(dev), ffi::ACK_OK, "{}", last_error());
}

/// The audit words of AKV MoE plan Amendment 29: signed zeros, four
/// subnormals of each sign, the smallest normals, +-1, the largest finites,
/// the infinities and four NaNs.
const SUBNORMAL_AUDIT_WORDS: [u32; 22] = [
    0x0000_0000,
    0x8000_0000,
    0x0000_0001,
    0x8000_0001,
    0x0000_0002,
    0x8000_0002,
    0x0040_0000,
    0x8040_0000,
    0x007F_FFFF,
    0x807F_FFFF,
    0x0080_0000,
    0x8080_0000,
    0x3F80_0000,
    0xBF80_0000,
    0x7F7F_FFFF,
    0xFF7F_FFFF,
    0x7F80_0000,
    0xFF80_0000,
    0x7FC0_0000,
    0xFFC0_0000,
    0x7FA0_0001,
    0xFFC0_0ABC,
];

/// The comparisons (26-31), the logical operations (32-35), `where` (36),
/// `clamp` (13) and `sqrt`/`rsqrt`'s NaN (8, 9) at subnormal operands
/// (Amendment 29, the owner's audit of 2026-10-16). The rule is the host's own
/// f32 arithmetic, which does not flush: a comparison is Rust's, "nonzero" is
/// `!= 0.0`, `clamp` is `clamp_reference`, and a negative subnormal's square
/// root is NaN. A device whose float comparison flushes a subnormal (the RX
/// 9070 XT's did, Amendment 28) fails here unless the shader decides on the
/// bits.
#[test]
fn comparisons_and_selections_decide_subnormals_as_the_host_does() {
    let one = 1.0f32.to_bits();
    let truth = |b: bool| if b { one } else { 0 };
    // the rule's own anchors: the host does not flush
    let sub = f32::from_bits(1);
    assert!(sub > 0.0 && sub != 0.0 && sub.partial_cmp(&0.0) == Some(std::cmp::Ordering::Greater));
    assert_eq!(clamp_reference(-sub, 0.0, f32::INFINITY).to_bits(), 0);

    let dev = ffi::ack_open();
    if dev.is_null() {
        if std::env::var_os("ACK_REQUIRE_DEVICE").is_some() {
            panic!(
                "pointwise ABI (subnormal audit): ACK_REQUIRE_DEVICE is set and ack_open failed: {}",
                last_error()
            );
        }
        eprintln!(
            "UNMEASURED here (pointwise ABI, subnormal audit): {}",
            last_error()
        );
        return;
    }
    let words = SUBNORMAL_AUDIT_WORDS;
    let xs: Vec<u32> = words
        .iter()
        .flat_map(|&x| words.iter().map(move |_| x))
        .collect();
    let ys: Vec<u32> = words.iter().flat_map(|_| words.iter().copied()).collect();
    let n = xs.len();
    let fx = |i: usize| f32::from_bits(xs[i]);
    let fy = |i: usize| f32::from_bits(ys[i]);
    let alloc = || {
        let b = ffi::ack_buffer_alloc(dev, (n * 4) as u64);
        assert!(!b.is_null(), "{}", last_error());
        b
    };
    let upload = |buf, data: &[u32]| {
        let rc = unsafe { ffi::ack_upload(dev, buf, data.as_ptr().cast::<u8>(), data.len() * 4) };
        assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
    };
    let download = |buf| {
        let mut out = vec![0u32; n];
        let rc =
            unsafe { ffi::ack_download(dev, buf, out.as_mut_ptr().cast::<u8>(), out.len() * 4) };
        assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
        out
    };
    let run = |op: PointwiseOp, scalars: Scalars, buffers: [*const Buffer; OPERANDS]| -> i32 {
        let shape = [n as u64];
        let dense = StridedView::contiguous(&shape).unwrap();
        let plan = block(op, &shape, [dense; OPERANDS], scalars);
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
    let (bx, by, bout) = (alloc(), alloc(), alloc());
    upload(bx, &xs);
    upload(by, &ys);
    // every differing element across every operation, so one run reports them all
    let mut failures: Vec<String> = Vec::new();
    let mut checked = 0usize;
    let mut check = |name: &str, got: Vec<u32>, want: &dyn Fn(usize) -> u32| {
        for (i, &g) in got.iter().enumerate() {
            let w = want(i);
            if g != w {
                failures.push(format!(
                    "{name}[{i}]({:#010x}, {:#010x}): device {g:#010x}, host {w:#010x}",
                    xs[i], ys[i]
                ));
            }
        }
        checked += got.len();
    };
    type Rule = fn(f32, f32) -> bool;
    let binary: [(PointwiseOp, Rule); 9] = [
        (PointwiseOp::Gt, |p, q| p > q),
        (PointwiseOp::Lt, |p, q| p < q),
        (PointwiseOp::Ge, |p, q| p >= q),
        (PointwiseOp::Le, |p, q| p <= q),
        (PointwiseOp::Eq, |p, q| p == q),
        (PointwiseOp::Ne, |p, q| p != q),
        (PointwiseOp::LogicalAnd, |p, q| p != 0.0 && q != 0.0),
        (PointwiseOp::LogicalOr, |p, q| p != 0.0 || q != 0.0),
        (PointwiseOp::LogicalXor, |p, q| (p != 0.0) != (q != 0.0)),
    ];
    for (op, rule) in binary {
        upload(bout, &vec![0x7F80_0001; n]);
        let rc = run(op, Scalars::new(0.0), [bx, by, bx, bout]);
        assert_eq!(rc, ffi::ACK_OK, "{op:?}: {}", last_error());
        check(&format!("{op:?}"), download(bout), &|i| {
            truth(rule(fx(i), fy(i)))
        });
    }
    upload(bout, &vec![0x7F80_0001; n]);
    let rc = run(
        PointwiseOp::LogicalNot,
        Scalars::new(0.0),
        [bx, bx, bx, bout],
    );
    assert_eq!(rc, ffi::ACK_OK, "LogicalNot: {}", last_error());
    check("LogicalNot", download(bout), &|i| truth(fx(i) == 0.0));
    // where(x, y, x): the condition x, and the chosen word moved as it is
    upload(bout, &vec![0x7F80_0001; n]);
    let rc = run(PointwiseOp::Where, Scalars::new(0.0), [bx, by, bx, bout]);
    assert_eq!(rc, ffi::ACK_OK, "Where: {}", last_error());
    check("Where", download(bout), &|i| {
        if fx(i) != 0.0 {
            ys[i]
        } else {
            xs[i]
        }
    });
    // clamp at and around the subnormals (no NaN bound: the host refuses one)
    for (low, high) in [
        (Some(0.0f32), None),
        (None, Some(0.0f32)),
        (Some(-0.0), None),
        (Some(sub), None),
        (Some(-sub), Some(sub)),
        (Some(-sub), None),
        (None, Some(-sub)),
        (Some(-1.0), Some(1.0)),
        (Some(f32::MIN_POSITIVE), Some(1.0)),
    ] {
        upload(bout, &vec![0x7F80_0001; n]);
        let rc = run(
            PointwiseOp::Clamp,
            Scalars::clamp(low, high),
            [bx, bx, bx, bout],
        );
        assert_eq!(rc, ffi::ACK_OK, "Clamp: {}", last_error());
        let (lo, hi) = (
            low.unwrap_or(f32::NEG_INFINITY),
            high.unwrap_or(f32::INFINITY),
        );
        check(&format!("Clamp({low:?}, {high:?})"), download(bout), &|i| {
            clamp_reference(fx(i), lo, hi).to_bits()
        });
    }
    // sqrt and rsqrt: the NaN class only (their arithmetic is the family's
    // tolerance law); 1 where the result is NaN
    for op in [PointwiseOp::Sqrt, PointwiseOp::Rsqrt] {
        upload(bout, &vec![0x7F80_0001; n]);
        let rc = run(op, Scalars::new(0.0), [bx, bx, bx, bout]);
        assert_eq!(rc, ffi::ACK_OK, "{op:?}: {}", last_error());
        let got: Vec<u32> = download(bout)
            .into_iter()
            .map(|g| u32::from(f32::from_bits(g).is_nan()))
            .collect();
        check(&format!("{op:?} is NaN"), got, &|i| {
            u32::from(fx(i).sqrt().is_nan())
        });
    }
    eprintln!(
        "subnormal audit: {checked} results over {n} ordered pairs of the audit words, {} differ from the host",
        failures.len()
    );
    for b in [bx, by, bout] {
        assert_eq!(
            ffi::ack_buffer_free(dev, b),
            ffi::ACK_OK,
            "{}",
            last_error()
        );
    }
    assert_eq!(ffi::ack_close(dev), ffi::ACK_OK, "{}", last_error());
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
