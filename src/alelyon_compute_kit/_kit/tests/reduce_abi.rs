//! reduce-f32/v1 at the C ABI (`ack_reduce`, ABI 7): the caller's 80-byte
//! request block is re-derived by `reduce_ops::ReducePlan` against the input
//! buffer's real capacity, refusals are by name, and the four operations the
//! torch registrations need (a sum over one dimension, a mean over two, a
//! NaN-propagating amax and a vector norm over all three) match the plan
//! module's own FP64 reference within the reduce probe's declared tolerance,
//! as does a two-pass sum and mean through a partials buffer. The host-only
//! checks need no device; the operations need one and print UNMEASURED
//! without it unless ACK_REQUIRE_DEVICE is set. Fixed-fixture evidence only.
use alelyon_compute_kit::ffi;
use alelyon_compute_kit::reduce_ops::{
    Finish, ReduceKernel, ReduceOp, ReducePlan, ReduceView, CHUNK, FLAG_IN_BF16, FLAG_OUT_BF16,
    MAX_RANK, PUSH_BYTES, REQUEST_BYTES,
};
use std::ffi::{c_char, CStr};

/// The reduce probe's declared tolerance on a sum: relative to the sum of the
/// absolute reduced values (twice the kernel's worst-case accumulation depth
/// at FP32 unit roundoff) plus an absolute floor.
const SUM_REL_TOL: f64 = 3.2e-5;
const SUM_ABS_TOL: f64 = 1.0e-6;

fn last_error() -> String {
    let mut buf = vec![0 as c_char; 1024];
    let rc = unsafe { ffi::ack_last_error(buf.as_mut_ptr(), buf.len()) };
    assert!(rc >= 0, "ack_last_error returned {rc}");
    unsafe { CStr::from_ptr(buf.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

/// The request block a caller passes: the kernel's push layout with the two
/// plan-derived words (splits, chunk) left zero.
fn request(
    op: ReduceOp,
    finish: Finish,
    shape: &[u32],
    stride: &[u32],
    offset: u32,
    reduced_mask: u32,
    scale: f32,
) -> [u8; PUSH_BYTES] {
    let mut words = [0u32; PUSH_BYTES / 4];
    words[0] = op as u32;
    words[1] = finish as u32;
    words[2] = shape.len() as u32;
    words[3] = reduced_mask;
    for d in 0..MAX_RANK {
        words[4 + d] = if d < shape.len() { shape[d] } else { 1 };
        words[10 + d] = if d < stride.len() { stride[d] } else { 0 };
    }
    words[16] = offset;
    words[19] = scale.to_bits();
    let mut block = [0u8; PUSH_BYTES];
    for (i, w) in words.iter().enumerate() {
        block[4 * i..4 * i + 4].copy_from_slice(&w.to_le_bytes());
    }
    block
}

#[allow(clippy::too_many_arguments)]
fn plan(
    op: ReduceOp,
    finish: Finish,
    shape: &[u32],
    stride: &[u32],
    offset: u32,
    reduced_mask: u32,
    scale: f32,
    capacity: u64,
) -> ReducePlan {
    let view = ReduceView::new(shape, stride, offset).unwrap();
    ReducePlan::new(view, reduced_mask, op, finish, scale, capacity).unwrap()
}

/// The per-output scale of the tolerance: the sum of |x| over the reduced
/// range, computed by the plan module's own reference walk.
fn sum_abs(shape: &[u32], stride: &[u32], offset: u32, reduced_mask: u32, x: &[f32]) -> Vec<f64> {
    let magnitudes: Vec<f32> = x.iter().map(|v| v.abs()).collect();
    plan(
        ReduceOp::Sum,
        Finish::None,
        shape,
        stride,
        offset,
        reduced_mask,
        1.0,
        x.len() as u64,
    )
    .reference(&magnitudes)
    .unwrap()
}

fn assert_within(name: &str, got: &[f32], want: &[f64], scale: &[f64]) {
    assert_eq!(got.len(), want.len(), "{name}: output length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let tol = SUM_ABS_TOL + SUM_REL_TOL * scale[i];
        assert!(
            (f64::from(*g) - w).abs() <= tol,
            "{name}[{i}]: got {g}, want {w}, tolerance {tol}"
        );
    }
}

#[test]
fn a_null_short_or_malformed_request_is_refused_before_any_device_lookup() {
    let dev = std::ptr::null();
    let none = std::ptr::null();
    let call = |block: *const u8, len: usize| unsafe {
        ffi::ack_reduce(dev, none, none, none, block, len, std::ptr::null_mut())
    };
    assert_eq!(
        call(std::ptr::null(), PUSH_BYTES),
        ffi::ACK_ERR_NULL,
        "a null request block"
    );
    let short = [0u8; 64];
    assert_eq!(
        call(short.as_ptr(), short.len()),
        ffi::ACK_ERR_SIZE,
        "a 64-byte block: {}",
        last_error()
    );
    let good = request(ReduceOp::Sum, Finish::None, &[4], &[1], 0, 1, 1.0);
    for (word, value, name) in [
        (0usize, 3u32, "reduce-operation-out-of-range"),
        (1, 3, "reduce-finish-out-of-range"),
        (2, 7, "reduce-rank-out-of-range"),
        (17, 1, "reduce-reserved-word-nonzero"),
        (18, CHUNK, "reduce-reserved-word-nonzero"),
        (5, 2, "reduce-padding-out-of-range"),
    ] {
        let mut block = good;
        block[4 * word..4 * word + 4].copy_from_slice(&value.to_le_bytes());
        assert_eq!(
            call(block.as_ptr(), block.len()),
            ffi::ACK_ERR_SHAPE,
            "{name}: {}",
            last_error()
        );
        assert!(last_error().contains(name), "{}", last_error());
    }
    // with a well-formed block the null device handle is what refuses next
    assert_eq!(
        call(good.as_ptr(), good.len()),
        ffi::ACK_ERR_NULL,
        "a null device handle is refused as null before any registry lookup: {}",
        last_error()
    );
}

#[test]
fn the_four_operations_and_a_two_pass_plan_match_the_reference_and_refusals_are_by_name() {
    let dev = ffi::ack_open();
    if dev.is_null() {
        if std::env::var_os("ACK_REQUIRE_DEVICE").is_some() {
            panic!(
                "reduce ABI: ACK_REQUIRE_DEVICE is set and ack_open failed: {}",
                last_error()
            );
        }
        eprintln!("UNMEASURED here (reduce ABI): {}", last_error());
        return;
    }
    let shape = [3u32, 5, 7];
    let stride = [35u32, 7, 1];
    let n = 105usize;
    let x: Vec<f32> = (0..n)
        .map(|i| ((i * 7919) % 97) as f32 / 13.0 - 3.5)
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
    // a transfer is the whole buffer (ack_download refuses a partial one), so
    // every output has a buffer of exactly its length, poisoned before its case
    let download = |buf, elements: usize| {
        let mut out = vec![0f32; elements];
        let rc =
            unsafe { ffi::ack_download(dev, buf, out.as_mut_ptr().cast::<u8>(), out.len() * 4) };
        assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
        out
    };
    let poison = |buf, elements: usize| upload(buf, &vec![f32::NAN; elements]);
    let bx = alloc(n);
    let b15 = alloc(15);
    let b5 = alloc(5);
    let b21 = alloc(21);
    let b1 = alloc(1);
    let b7 = alloc(7);
    upload(bx, &x);
    let mut ms = 0f64;
    let none = std::ptr::null();

    // 1. sum over the last dimension: 15 outputs
    poison(b15, 15);
    let block = request(ReduceOp::Sum, Finish::None, &shape, &stride, 0, 0b100, 1.0);
    let rc = unsafe { ffi::ack_reduce(dev, bx, none, b15, block.as_ptr(), block.len(), &mut ms) };
    assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
    assert!(
        ms.is_nan() || ms >= 0.0,
        "a duration or NaN, never negative: {ms}"
    );
    let want = plan(
        ReduceOp::Sum,
        Finish::None,
        &shape,
        &stride,
        0,
        0b100,
        1.0,
        n as u64,
    )
    .reference(&x)
    .unwrap();
    assert_within(
        "sum",
        &download(b15, 15),
        &want,
        &sum_abs(&shape, &stride, 0, 0b100, &x),
    );
    eprintln!("reduce sum over [3,5,7] dim 2: {ms:.3} ms");

    // 2. mean over the first and last dimensions (21 reduced values): 5 outputs
    poison(b5, 5);
    let scale = 1.0f32 / 21.0;
    let block = request(
        ReduceOp::Sum,
        Finish::Scale,
        &shape,
        &stride,
        0,
        0b101,
        scale,
    );
    let rc = unsafe { ffi::ack_reduce(dev, bx, none, b5, block.as_ptr(), block.len(), &mut ms) };
    assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
    let want = plan(
        ReduceOp::Sum,
        Finish::Scale,
        &shape,
        &stride,
        0,
        0b101,
        scale,
        n as u64,
    )
    .reference(&x)
    .unwrap();
    let scaled: Vec<f64> = sum_abs(&shape, &stride, 0, 0b101, &x)
        .iter()
        .map(|s| s * f64::from(scale))
        .collect();
    assert_within("mean", &download(b5, 5), &want, &scaled);

    // 3. amax over the middle dimension with one NaN: 21 outputs, the NaN
    //    propagates to its own output and every other output is the exact maximum
    let mut with_nan = x.clone();
    with_nan[35 + 2 * 7 + 3] = f32::NAN; // (1, 2, 3): output (1, 3) = index 10
    upload(bx, &with_nan);
    poison(b21, 21);
    let block = request(ReduceOp::Amax, Finish::None, &shape, &stride, 0, 0b010, 1.0);
    let rc = unsafe { ffi::ack_reduce(dev, bx, none, b21, block.as_ptr(), block.len(), &mut ms) };
    assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
    let want = plan(
        ReduceOp::Amax,
        Finish::None,
        &shape,
        &stride,
        0,
        0b010,
        1.0,
        n as u64,
    )
    .reference(&with_nan)
    .unwrap();
    let got = download(b21, 21);
    // range, independent of the reference: a maximum is one of the values it
    // was taken over. A formula shared with the kernel cannot vouch for that.
    for (i, value) in got.iter().enumerate() {
        assert!(
            i == 10 || with_nan.contains(value),
            "amax[{i}] = {value} was never an input"
        );
    }
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        if i == 10 {
            assert!(
                g.is_nan() && w.is_nan(),
                "amax[10] propagates the NaN: got {g}"
            );
        } else {
            assert_eq!(f64::from(*g), *w, "amax[{i}] is the exact maximum");
        }
    }
    upload(bx, &x);

    // 4. vector norm over every dimension: one output, sqrt of the sum of squares
    poison(b1, 1);
    let block = request(
        ReduceOp::SumOfSquares,
        Finish::Sqrt,
        &shape,
        &stride,
        0,
        0b111,
        1.0,
    );
    let rc = unsafe { ffi::ack_reduce(dev, bx, none, b1, block.as_ptr(), block.len(), &mut ms) };
    assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
    let want = plan(
        ReduceOp::SumOfSquares,
        Finish::Sqrt,
        &shape,
        &stride,
        0,
        0b111,
        1.0,
        n as u64,
    )
    .reference(&x)
    .unwrap();
    let got = download(b1, 1);
    // range, independent of the reference: a Euclidean norm is finite and
    // non-negative for finite inputs, whatever any tolerance says
    assert!(
        got[0].is_finite() && got[0] >= 0.0,
        "norm {} is not a norm",
        got[0]
    );
    // a square root halves the relative error of the sum of squares; the
    // tolerance is the sum's, relative to the norm itself
    assert!(
        (f64::from(got[0]) - want[0]).abs() <= SUM_ABS_TOL + SUM_REL_TOL * want[0],
        "norm: got {}, want {}",
        got[0],
        want[0]
    );

    // 5. a two-pass sum and mean: [3, 70,000] over the second dimension needs
    //    two splits of 65,536, so a partials buffer of 3 x 2 elements
    let wide = [3u32, 70_000];
    let wide_stride = [70_000u32, 1];
    let m = 210_000usize;
    let y: Vec<f32> = (0..m)
        .map(|i| ((i * 104_729) % 89) as f32 / 11.0 - 4.0)
        .collect();
    let by = alloc(m);
    let bpartials = alloc(6);
    let b3 = alloc(3);
    upload(by, &y);
    let two = plan(
        ReduceOp::Sum,
        Finish::None,
        &wide,
        &wide_stride,
        0,
        0b10,
        1.0,
        m as u64,
    );
    assert_eq!(two.splits(), 2, "the fixture needs two passes");
    let want = two.reference(&y).unwrap();
    let scale_abs = sum_abs(&wide, &wide_stride, 0, 0b10, &y);
    poison(b3, 3);
    let block = request(
        ReduceOp::Sum,
        Finish::None,
        &wide,
        &wide_stride,
        0,
        0b10,
        1.0,
    );
    let rc =
        unsafe { ffi::ack_reduce(dev, by, bpartials, b3, block.as_ptr(), block.len(), &mut ms) };
    assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
    assert_within("two-pass sum", &download(b3, 3), &want, &scale_abs);
    eprintln!("reduce two-pass sum over [3,70000] dim 1: {ms:.3} ms");
    let mean_scale = 1.0f32 / 70_000.0;
    poison(b3, 3);
    let block = request(
        ReduceOp::Sum,
        Finish::Scale,
        &wide,
        &wide_stride,
        0,
        0b10,
        mean_scale,
    );
    let rc =
        unsafe { ffi::ack_reduce(dev, by, bpartials, b3, block.as_ptr(), block.len(), &mut ms) };
    assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
    let want = plan(
        ReduceOp::Sum,
        Finish::Scale,
        &wide,
        &wide_stride,
        0,
        0b10,
        mean_scale,
        m as u64,
    )
    .reference(&y)
    .unwrap();
    let scaled: Vec<f64> = scale_abs
        .iter()
        .map(|s| s * f64::from(mean_scale))
        .collect();
    assert_within("two-pass mean", &download(b3, 3), &want, &scaled);

    // refusals by name, nothing dispatched: the previous results survive each one
    let before = download(b3, 3);
    // a two-pass plan without a partials buffer
    let rc = unsafe { ffi::ack_reduce(dev, by, none, b3, block.as_ptr(), block.len(), &mut ms) };
    assert_eq!(rc, ffi::ACK_ERR_NULL, "{}", last_error());
    assert!(
        last_error().contains("needs a partials buffer"),
        "{}",
        last_error()
    );
    // a partials buffer too small for the six partials (a 5-element buffer)
    let rc = unsafe { ffi::ack_reduce(dev, by, b5, b3, block.as_ptr(), block.len(), &mut ms) };
    assert_eq!(rc, ffi::ACK_ERR_SIZE, "{}", last_error());
    assert!(
        last_error().contains("reduce-partials-too-small"),
        "{}",
        last_error()
    );
    // an output buffer too small for the kept count (the [3,5,7] sum over dim 2 into 3 elements)
    let sum_block = request(ReduceOp::Sum, Finish::None, &shape, &stride, 0, 0b100, 1.0);
    let rc = unsafe {
        ffi::ack_reduce(
            dev,
            bx,
            none,
            b3,
            sum_block.as_ptr(),
            sum_block.len(),
            &mut ms,
        )
    };
    assert_eq!(rc, ffi::ACK_ERR_SIZE, "{}", last_error());
    assert!(
        last_error().contains("reduce-output-too-small"),
        "{}",
        last_error()
    );
    // the output through the input's own buffer (a reduction is never in place)
    let rc = unsafe {
        ffi::ack_reduce(
            dev,
            bx,
            none,
            bx,
            sum_block.as_ptr(),
            sum_block.len(),
            &mut ms,
        )
    };
    assert_eq!(rc, ffi::ACK_ERR_SHAPE, "{}", last_error());
    assert!(
        last_error().contains("reduce-output-aliases-input"),
        "{}",
        last_error()
    );
    // partials that are the output buffer
    let rc = unsafe { ffi::ack_reduce(dev, by, b3, b3, block.as_ptr(), block.len(), &mut ms) };
    assert_eq!(rc, ffi::ACK_ERR_SHAPE, "{}", last_error());
    assert!(
        last_error().contains("reduce-partials-alias"),
        "{}",
        last_error()
    );
    // a view past its buffer (the 3-element buffer read through the [3,5,7] view)
    let rc = unsafe {
        ffi::ack_reduce(
            dev,
            b3,
            none,
            b15,
            sum_block.as_ptr(),
            sum_block.len(),
            &mut ms,
        )
    };
    assert_eq!(rc, ffi::ACK_ERR_SIZE, "{}", last_error());
    assert!(
        last_error().contains("reduce-capacity-too-small"),
        "{}",
        last_error()
    );
    // a reduced mask naming a dimension beyond the rank, and a zero extent
    let masked = request(ReduceOp::Sum, Finish::None, &shape, &stride, 0, 0b1000, 1.0);
    let rc = unsafe { ffi::ack_reduce(dev, bx, none, b15, masked.as_ptr(), masked.len(), &mut ms) };
    assert_eq!(rc, ffi::ACK_ERR_SHAPE, "{}", last_error());
    assert!(
        last_error().contains("reduce-reduced-mask-invalid"),
        "{}",
        last_error()
    );
    let empty = request(
        ReduceOp::Sum,
        Finish::None,
        &[3, 0, 7],
        &stride,
        0,
        0b010,
        1.0,
    );
    let rc = unsafe { ffi::ack_reduce(dev, bx, none, b15, empty.as_ptr(), empty.len(), &mut ms) };
    assert_eq!(rc, ffi::ACK_ERR_SHAPE, "{}", last_error());
    assert!(
        last_error().contains("reduce-zero-extent"),
        "{}",
        last_error()
    );
    // a freed buffer is refused by the registry before the plan is derived
    let spare = alloc(n);
    assert_eq!(
        ffi::ack_buffer_free(dev, spare),
        ffi::ACK_OK,
        "{}",
        last_error()
    );
    let rc = unsafe {
        ffi::ack_reduce(
            dev,
            spare,
            none,
            b15,
            sum_block.as_ptr(),
            sum_block.len(),
            &mut ms,
        )
    };
    assert_eq!(rc, ffi::ACK_ERR_FREED, "{}", last_error());
    assert_eq!(download(b3, 3), before, "a refusal dispatches nothing");

    // a strided, offset view: the transposed last two dimensions of x from
    // element 35 (the second [5,7] slab), reduced over what was the row axis
    poison(b7, 7);
    let slab = [7u32, 5];
    let slab_stride = [1u32, 7];
    let block = request(
        ReduceOp::Sum,
        Finish::None,
        &slab,
        &slab_stride,
        35,
        0b10,
        1.0,
    );
    let rc = unsafe { ffi::ack_reduce(dev, bx, none, b7, block.as_ptr(), block.len(), &mut ms) };
    assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
    let want = plan(
        ReduceOp::Sum,
        Finish::None,
        &slab,
        &slab_stride,
        35,
        0b10,
        1.0,
        n as u64,
    )
    .reference(&x)
    .unwrap();
    assert_within(
        "strided offset sum",
        &download(b7, 7),
        &want,
        &sum_abs(&slab, &slab_stride, 35, 0b10, &x),
    );

    for buf in [bx, b15, b5, b21, b1, b7, by, bpartials, b3] {
        assert_eq!(
            ffi::ack_buffer_free(dev, buf),
            ffi::ACK_OK,
            "{}",
            last_error()
        );
    }
    assert_eq!(ffi::ack_close(dev), ffi::ACK_OK, "{}", last_error());
}

/// reduce-f32/v2 (2026-09-09) through the C ABI on the reductions the
/// registered AKV pass issues: the plan selects v2 for a leading-dims sum
/// keeping 512 columns, for a repeat-dim sum with a 65,536-wide contiguous
/// kept block, and for the same sum over a transposed view whose unit stride
/// is a middle dim; every output matches the plan's own f64 reference inside
/// the family tolerance, no padding element of a poisoned output is touched,
/// and the input is unchanged. A narrow leading-dims sum keeps v1 through the
/// same entry, so the selection is the plan's and invisible to the caller.
#[test]
fn the_reductions_the_registered_pass_issues_run_on_v2_and_match_the_reference() {
    let dev = ffi::ack_open();
    if dev.is_null() {
        if std::env::var_os("ACK_REQUIRE_DEVICE").is_some() {
            panic!(
                "reduce ABI v2: ACK_REQUIRE_DEVICE is set and ack_open failed: {}",
                last_error()
            );
        }
        eprintln!("UNMEASURED here (reduce ABI v2): {}", last_error());
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
    // (name, shape, stride, capacity, mask, the kernel the plan must select)
    type Spec = (
        &'static str,
        &'static [u32],
        &'static [u32],
        usize,
        u32,
        ReduceKernel,
    );
    let specs: [Spec; 5] = [
        (
            "leading dims of [1, 1024, 512], keep 512",
            &[1, 1024, 512],
            &[524_288, 512, 1],
            524_288,
            0b011,
            ReduceKernel::V2,
        ),
        (
            "GQA repeat dim of [1, 4, 2, 1024, 64]",
            &[1, 4, 2, 1024, 64],
            &[524_288, 131_072, 65_536, 64, 1],
            524_288,
            0b00100,
            ReduceKernel::V2,
        ),
        (
            "GQA repeat dim of a transposed [1, 4, 2, 1024, 48]",
            &[1, 4, 2, 1024, 48],
            &[393_216, 98_304, 49_152, 1, 1024],
            393_216,
            0b00100,
            ReduceKernel::V2,
        ),
        (
            "the probe's repeat extent of 3",
            &[1, 4, 3, 128, 64],
            &[98_304, 24_576, 8_192, 64, 1],
            98_304,
            0b00100,
            ReduceKernel::V2,
        ),
        (
            "leading dims keeping 48 columns stays on v1",
            &[1, 1024, 8, 48],
            &[393_216, 48, 49_152, 1],
            393_216,
            0b0111,
            ReduceKernel::V1,
        ),
    ];
    let mut ms = 0f64;
    let none = std::ptr::null();
    for (name, shape, stride, capacity, mask, expected_kernel) in specs {
        let x: Vec<f32> = (0..capacity)
            .map(|i| ((i.wrapping_mul(7919) % 97) as f32) / 13.0 - 3.5)
            .collect();
        let p = plan(
            ReduceOp::Sum,
            Finish::None,
            shape,
            stride,
            0,
            mask,
            1.0,
            capacity as u64,
        );
        assert_eq!(
            p.first().kernel(),
            expected_kernel,
            "{name}: the plan's kernel"
        );
        assert!(p.second().is_none(), "{name}: one pass");
        let kept = p.output_len();
        // the output buffer carries two extra poisoned elements past the kept
        // count, so a kernel writing past its output is caught
        let bx = alloc(capacity);
        let bo = alloc(kept + 2);
        upload(bx, &x);
        upload(bo, &vec![f32::NAN; kept + 2]);
        let block = request(ReduceOp::Sum, Finish::None, shape, stride, 0, mask, 1.0);
        let rc =
            unsafe { ffi::ack_reduce(dev, bx, none, bo, block.as_ptr(), block.len(), &mut ms) };
        assert_eq!(rc, ffi::ACK_OK, "{name}: {}", last_error());
        let got = download(bo, kept + 2);
        assert!(
            got[kept].is_nan() && got[kept + 1].is_nan(),
            "{name}: the elements past the output were written"
        );
        let want = p.reference(&x).unwrap();
        let scale = sum_abs(shape, stride, 0, mask, &x);
        assert_eq!(
            got[..kept].iter().filter(|v| v.is_nan()).count(),
            0,
            "{name}: an output was left unwritten (poison survived)"
        );
        assert_within(name, &got[..kept], &want, &scale);
        assert_eq!(download(bx, capacity), x, "{name}: the input was modified");
        for buf in [bx, bo] {
            assert_eq!(
                ffi::ack_buffer_free(dev, buf),
                ffi::ACK_OK,
                "{}",
                last_error()
            );
        }
        eprintln!("reduce v2 ABI: {name}: {kept} outputs inside the family tolerance");
    }
    assert_eq!(ffi::ack_close(dev), ffi::ACK_OK, "{}", last_error());
}

/// The four storage combinations of schema 2, against the f32 path, BITWISE.
///
/// The claim this makes is narrow on purpose: bf16 storage changes how a value
/// is READ and WRITTEN and changes nothing about the arithmetic. So it is
/// tested with inputs that are EXACTLY representable in bf16 -- the low 16 bits
/// of every f32 word zeroed -- which makes the bf16 and f32 arms see the same
/// real numbers. Any difference in the result is then a difference in the
/// arithmetic, and there must not be one, so the assertion is equality of bits
/// rather than a tolerance. A tolerance here would pass just as happily if the
/// accumulation had quietly moved to bf16, which is the defect worth catching.
///
/// The output side is asserted the same way, against `narrow` of the f32
/// result: the rounding is copied from cast_f32_bf16.comp, so storing a result
/// must equal casting it.
///
/// Both a ONE-pass and a TWO-pass plan are covered, because they exercise
/// different module derivations: one pass takes both of the caller's bits,
/// while two passes take the input bit on the first (its output is f32
/// partials) and the output bit on the second (its input is those partials).
/// A shared switch would pass the one-pass rows and fail these.
#[test]
fn bf16_storage_matches_the_f32_path_bitwise_in_every_combination() {
    let dev = ffi::ack_open();
    if dev.is_null() {
        if std::env::var_os("ACK_REQUIRE_DEVICE").is_some() {
            panic!(
                "reduce bf16: ACK_REQUIRE_DEVICE is set and ack_open failed: {}",
                last_error()
            );
        }
        eprintln!("UNMEASURED here (reduce bf16): {}", last_error());
        return;
    }
    // round-to-nearest-even f32 -> bf16, the shader's own arithmetic
    fn narrow(x: f32) -> u16 {
        let bits = x.to_bits();
        if (bits & 0x7F80_0000) == 0x7F80_0000 && (bits & 0x007F_FFFF) != 0 {
            return ((bits >> 16) | 0x0040) as u16;
        }
        let lsb = (bits >> 16) & 1;
        (bits.wrapping_add(0x7FFF).wrapping_add(lsb) >> 16) as u16
    }
    let alloc = |bytes: u64| {
        let b = ffi::ack_buffer_alloc(dev, bytes);
        assert!(!b.is_null(), "{}", last_error());
        b
    };
    let up32 = |buf, data: &[f32]| {
        let rc = unsafe { ffi::ack_upload(dev, buf, data.as_ptr().cast::<u8>(), data.len() * 4) };
        assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
    };
    let up16 = |buf, data: &[u16]| {
        let rc = unsafe { ffi::ack_upload(dev, buf, data.as_ptr().cast::<u8>(), data.len() * 2) };
        assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
    };
    let down32 = |buf, elements: usize| {
        let mut out = vec![0f32; elements];
        let rc =
            unsafe { ffi::ack_download(dev, buf, out.as_mut_ptr().cast::<u8>(), out.len() * 4) };
        assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
        out
    };
    let down16 = |buf, elements: usize| {
        let mut out = vec![0u16; elements];
        let rc =
            unsafe { ffi::ack_download(dev, buf, out.as_mut_ptr().cast::<u8>(), out.len() * 2) };
        assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
        out
    };
    // `flags` appended to the 80-byte push layout is the whole of schema 2
    let with_flags = |block: &[u8; PUSH_BYTES], flags: u32| {
        let mut request = [0u8; REQUEST_BYTES];
        request[..PUSH_BYTES].copy_from_slice(block);
        request[PUSH_BYTES..].copy_from_slice(&flags.to_le_bytes());
        request
    };

    // one pass (kept 3, reduced 5) and two passes (kept 2, reduced 70,000 >
    // CHUNK, so the plan splits and the middle block is partials)
    for (shape, stride, mask, kept, passes) in [
        (
            [3u32, 5].as_slice(),
            [5u32, 1].as_slice(),
            0b10u32,
            3usize,
            1usize,
        ),
        (
            [2u32, 70_000].as_slice(),
            [70_000u32, 1].as_slice(),
            0b10u32,
            2usize,
            2usize,
        ),
    ] {
        let n: usize = shape.iter().map(|&s| s as usize).product();
        // EXACTLY representable in bf16: the low 16 bits of each word zeroed,
        // so the two arms see identical real numbers and any difference in the
        // answer is a difference in the arithmetic
        let x: Vec<f32> = (0..n)
            .map(|i| {
                let v = ((i * 7919) % 97) as f32 / 8.0 - 6.0;
                f32::from_bits(v.to_bits() & 0xFFFF_0000)
            })
            .collect();
        for (index, value) in x.iter().enumerate() {
            assert_eq!(
                f32::from_bits(u32::from(narrow(*value)) << 16),
                *value,
                "input {index} is not exact in bf16, so this test cannot separate storage from arithmetic"
            );
        }
        let x16: Vec<u16> = x.iter().map(|v| narrow(*v)).collect();

        let bx32 = alloc((n * 4) as u64);
        let bx16 = alloc((n * 2) as u64);
        let bout32 = alloc((kept * 4) as u64);
        let bout16 = alloc((kept * 2) as u64);
        let bpart = alloc((kept * 4 * 4) as u64); // f32 partials, splits <= 2 here
        up32(bx32, &x);
        up16(bx16, &x16);

        let block = request(ReduceOp::Sum, Finish::None, shape, stride, 0, mask, 1.0);
        let scratch = if passes == 2 { bpart } else { std::ptr::null() };
        let mut ms = 0f64;

        // the reference: f32 in, f32 out, the 80-byte block every caller before
        // schema 2 sent, unchanged
        let rc = unsafe {
            ffi::ack_reduce(
                dev,
                bx32,
                scratch,
                bout32,
                block.as_ptr(),
                PUSH_BYTES,
                &mut ms,
            )
        };
        assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
        let reference = down32(bout32, kept);

        // and the 84-byte block with flags 0 is the SAME call: a length is not
        // a behaviour, and this is what says so
        let zero = with_flags(&block, 0);
        let rc = unsafe {
            ffi::ack_reduce(
                dev,
                bx32,
                scratch,
                bout32,
                zero.as_ptr(),
                REQUEST_BYTES,
                &mut ms,
            )
        };
        assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
        assert_eq!(
            down32(bout32, kept),
            reference,
            "{passes}-pass: the flags word being present and zero changed the answer"
        );

        // bf16 IN, f32 out: identical inputs, so identical bits out
        let in_only = with_flags(&block, FLAG_IN_BF16);
        let rc = unsafe {
            ffi::ack_reduce(
                dev,
                bx16,
                scratch,
                bout32,
                in_only.as_ptr(),
                REQUEST_BYTES,
                &mut ms,
            )
        };
        assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
        assert_eq!(
            down32(bout32, kept),
            reference,
            "{passes}-pass: bf16 storage moved the arithmetic"
        );

        // f32 in, bf16 OUT: the stored word is the cast of the f32 answer
        let out_only = with_flags(&block, FLAG_OUT_BF16);
        let rc = unsafe {
            ffi::ack_reduce(
                dev,
                bx32,
                scratch,
                bout16,
                out_only.as_ptr(),
                REQUEST_BYTES,
                &mut ms,
            )
        };
        assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
        let want16: Vec<u16> = reference.iter().map(|v| narrow(*v)).collect();
        assert_eq!(
            down16(bout16, kept),
            want16,
            "{passes}-pass: storing a bf16 result is not the cast of the f32 one"
        );

        // both, which is the combination the torch adapter actually sends
        let both = with_flags(&block, FLAG_IN_BF16 | FLAG_OUT_BF16);
        let rc = unsafe {
            ffi::ack_reduce(
                dev,
                bx16,
                scratch,
                bout16,
                both.as_ptr(),
                REQUEST_BYTES,
                &mut ms,
            )
        };
        assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
        assert_eq!(
            down16(bout16, kept),
            want16,
            "{passes}-pass: bf16 on both sides is not bf16 in composed with bf16 out"
        );

        for b in [bx32, bx16, bout32, bout16, bpart] {
            assert_eq!(
                ffi::ack_buffer_free(dev, b),
                ffi::ACK_OK,
                "{}",
                last_error()
            );
        }
    }
    assert_eq!(ffi::ack_close(dev), ffi::ACK_OK, "{}", last_error());
}
