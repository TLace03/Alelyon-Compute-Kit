//! matmul-f32/v1 at the C ABI (`ack_matmul_f32`, ABI 4): the caller's 64-byte
//! plan block is re-validated by `matmul_ops::MatmulPlan` against the buffers'
//! real capacities, refusals are by name, and a small product matches a host
//! FP64 reference within the family's own tolerance. The host-only checks need
//! no device; the product needs one and prints UNMEASURED without it unless
//! ACK_REQUIRE_DEVICE is set. Fixed-fixture evidence only, like the family's.
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

/// The family's push block: batch, m, n, k, then (offset, batch stride, row
/// stride, column stride) for A, B and C, sixteen little-endian u32.
fn block(batch: u32, m: u32, n: u32, k: u32, a: [u32; 4], b: [u32; 4], c: [u32; 4]) -> [u8; 64] {
    let mut words = [batch, m, n, k, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    words[4..8].copy_from_slice(&a);
    words[8..12].copy_from_slice(&b);
    words[12..16].copy_from_slice(&c);
    let mut out = [0u8; 64];
    for (i, w) in words.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
    }
    out
}

#[test]
fn a_null_short_or_unknown_plan_is_refused_before_any_device_lookup() {
    let dev = std::ptr::null();
    let none = std::ptr::null();
    let rc = unsafe {
        ffi::ack_matmul_f32(
            dev,
            none,
            none,
            none,
            0,
            std::ptr::null(),
            64,
            std::ptr::null_mut(),
        )
    };
    assert_eq!(rc, ffi::ACK_ERR_NULL, "a null plan block");
    let short = [0u8; 32];
    let rc = unsafe {
        ffi::ack_matmul_f32(
            dev,
            none,
            none,
            none,
            0,
            short.as_ptr(),
            short.len(),
            std::ptr::null_mut(),
        )
    };
    assert_eq!(rc, ffi::ACK_ERR_SIZE, "a 32-byte block: {}", last_error());
    let full = [0u8; 64];
    let rc = unsafe {
        ffi::ack_matmul_f32(
            dev,
            none,
            none,
            none,
            7,
            full.as_ptr(),
            full.len(),
            std::ptr::null_mut(),
        )
    };
    assert_eq!(rc, ffi::ACK_ERR_SHAPE, "kind 7: {}", last_error());
    // with a well-formed block the null device handle is what refuses next
    let rc = unsafe {
        ffi::ack_matmul_f32(
            dev,
            none,
            none,
            none,
            0,
            full.as_ptr(),
            full.len(),
            std::ptr::null_mut(),
        )
    };
    assert!(
        [ffi::ACK_ERR_CLOSED, ffi::ACK_ERR_NULL].contains(&rc),
        "a null device handle is refused by the registry, got {rc}: {}",
        last_error()
    );
}

#[test]
fn a_small_product_matches_a_host_f64_reference_and_refusals_are_by_name() {
    let dev = ffi::ack_open();
    if dev.is_null() {
        if std::env::var_os("ACK_REQUIRE_DEVICE").is_some() {
            panic!(
                "matmul_f32 ABI: ACK_REQUIRE_DEVICE is set and ack_open failed: {}",
                last_error()
            );
        }
        eprintln!("UNMEASURED here (matmul_f32 ABI): {}", last_error());
        return;
    }
    let (m, n, k) = (24u32, 40u32, 56u32);
    let a: Vec<f32> = (0..m * k)
        .map(|i| ((i * 7919) % 97) as f32 / 13.0 - 3.5)
        .collect();
    let b: Vec<f32> = (0..k * n)
        .map(|i| ((i * 104_729) % 89) as f32 / 11.0 - 4.0)
        .collect();
    let mut reference = vec![0f64; (m * n) as usize];
    for i in 0..m as usize {
        for j in 0..n as usize {
            let mut acc = 0f64;
            for l in 0..k as usize {
                acc += f64::from(a[i * k as usize + l]) * f64::from(b[l * n as usize + j]);
            }
            reference[i * n as usize + j] = acc;
        }
    }
    let ba = ffi::ack_buffer_alloc(dev, (a.len() * 4) as u64);
    let bb = ffi::ack_buffer_alloc(dev, (b.len() * 4) as u64);
    let bc = ffi::ack_buffer_alloc(dev, u64::from(m * n) * 4);
    assert!(
        !ba.is_null() && !bb.is_null() && !bc.is_null(),
        "{}",
        last_error()
    );
    let rc = unsafe { ffi::ack_upload(dev, ba, a.as_ptr().cast::<u8>(), a.len() * 4) };
    assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
    let rc = unsafe { ffi::ack_upload(dev, bb, b.as_ptr().cast::<u8>(), b.len() * 4) };
    assert_eq!(rc, ffi::ACK_OK, "{}", last_error());

    // row-major A (m x k), B (k x n), C (m x n); batch one with a unit batch stride
    let plan = block(1, m, n, k, [0, 1, k, 1], [0, 1, n, 1], [0, 1, n, 1]);
    let mut ms = 0f64;
    let rc = unsafe { ffi::ack_matmul_f32(dev, ba, bb, bc, 0, plan.as_ptr(), plan.len(), &mut ms) };
    assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
    let mut c = vec![0f32; (m * n) as usize];
    let rc = unsafe { ffi::ack_download(dev, bc, c.as_mut_ptr().cast::<u8>(), c.len() * 4) };
    assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
    let mut worst = 0f64;
    for (got, want) in c.iter().zip(&reference) {
        let err = (f64::from(*got) - want).abs();
        assert!(
            err <= 1e-7 + 1e-4 * want.abs(),
            "got {got}, want {want}, err {err}"
        );
        worst = worst.max(err);
    }
    eprintln!("matmul_f32 {m}x{n}x{k}: worst abs error {worst:e} in {ms:.3} ms");

    // the transposed storage of the same A (k x m stored, read as m x k): row stride 1, column stride m
    let at: Vec<f32> = (0..k as usize)
        .flat_map(|l| (0..m as usize).map(move |i| (i, l)))
        .map(|(i, l)| a[i * k as usize + l])
        .collect();
    let rc = unsafe { ffi::ack_upload(dev, ba, at.as_ptr().cast::<u8>(), at.len() * 4) };
    assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
    // re-poison the output first: a dispatch that silently wrote nothing would
    // otherwise leave the previous, correct result in place
    let poison = vec![f32::NAN; (m * n) as usize];
    let rc = unsafe { ffi::ack_upload(dev, bc, poison.as_ptr().cast::<u8>(), poison.len() * 4) };
    assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
    let plan_t = block(1, m, n, k, [0, 1, 1, m], [0, 1, n, 1], [0, 1, n, 1]);
    let rc =
        unsafe { ffi::ack_matmul_f32(dev, ba, bb, bc, 0, plan_t.as_ptr(), plan_t.len(), &mut ms) };
    assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
    let rc = unsafe { ffi::ack_download(dev, bc, c.as_mut_ptr().cast::<u8>(), c.len() * 4) };
    assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
    for (got, want) in c.iter().zip(&reference) {
        assert!(
            (f64::from(*got) - want).abs() <= 1e-7 + 1e-4 * want.abs(),
            "transposed A: got {got}, want {want}"
        );
    }

    // refusals by name, nothing dispatched: the output must be a distinct buffer
    let rc = unsafe { ffi::ack_matmul_f32(dev, ba, bb, ba, 0, plan.as_ptr(), plan.len(), &mut ms) };
    assert_eq!(rc, ffi::ACK_ERR_SHAPE, "{}", last_error());
    // a view addressing past its buffer (A with twice the row stride) is a size refusal
    let over = block(1, m, n, k, [0, 1, 2 * k, 1], [0, 1, n, 1], [0, 1, n, 1]);
    let rc = unsafe { ffi::ack_matmul_f32(dev, ba, bb, bc, 0, over.as_ptr(), over.len(), &mut ms) };
    assert_eq!(rc, ffi::ACK_ERR_SIZE, "{}", last_error());
    // a zero dimension, a zero stride, and a batch the buffers do not hold
    let zero = block(1, 0, n, k, [0, 1, k, 1], [0, 1, n, 1], [0, 1, n, 1]);
    let rc = unsafe { ffi::ack_matmul_f32(dev, ba, bb, bc, 0, zero.as_ptr(), zero.len(), &mut ms) };
    assert_eq!(rc, ffi::ACK_ERR_SHAPE, "{}", last_error());
    let stride0 = block(1, m, n, k, [0, 1, 0, 1], [0, 1, n, 1], [0, 1, n, 1]);
    let rc = unsafe {
        ffi::ack_matmul_f32(dev, ba, bb, bc, 0, stride0.as_ptr(), stride0.len(), &mut ms)
    };
    assert_eq!(rc, ffi::ACK_ERR_SHAPE, "{}", last_error());
    let batched = block(
        2,
        m,
        n,
        k,
        [0, m * k, k, 1],
        [0, k * n, n, 1],
        [0, m * n, n, 1],
    );
    let rc = unsafe {
        ffi::ack_matmul_f32(dev, ba, bb, bc, 1, batched.as_ptr(), batched.len(), &mut ms)
    };
    assert_eq!(
        rc,
        ffi::ACK_ERR_SIZE,
        "two batches in one-batch buffers: {}",
        last_error()
    );
    // mm with a batch of two is the plan's own refusal, once the buffers can hold two batches
    let ba2 = ffi::ack_buffer_alloc(dev, (a.len() * 8) as u64);
    let bb2 = ffi::ack_buffer_alloc(dev, (b.len() * 8) as u64);
    let bc2 = ffi::ack_buffer_alloc(dev, u64::from(m * n) * 8);
    assert!(
        !ba2.is_null() && !bb2.is_null() && !bc2.is_null(),
        "{}",
        last_error()
    );
    let rc = unsafe {
        ffi::ack_matmul_f32(
            dev,
            ba2,
            bb2,
            bc2,
            0,
            batched.as_ptr(),
            batched.len(),
            &mut ms,
        )
    };
    assert_eq!(
        rc,
        ffi::ACK_ERR_SHAPE,
        "mm with batch two: {}",
        last_error()
    );
    assert!(last_error().contains("batch"), "{}", last_error());
    for buf in [ba2, bb2, bc2] {
        assert_eq!(
            ffi::ack_buffer_free(dev, buf),
            ffi::ACK_OK,
            "{}",
            last_error()
        );
    }

    for buf in [ba, bb, bc] {
        assert_eq!(
            ffi::ack_buffer_free(dev, buf),
            ffi::ACK_OK,
            "{}",
            last_error()
        );
    }
    assert_eq!(ffi::ack_close(dev), ffi::ACK_OK, "{}", last_error());
}

// ---------------------------------------------------------------------------
// The accepted boundary, on a real card. The parity guard in matmul_plan.rs
// proves the copies agree as TEXT; only a dispatch proves the shader agrees.
// ---------------------------------------------------------------------------

/// A finite sentinel far outside any product this test can legitimately make.
///
/// Finite on purpose, so the finiteness assertion below is not doing double
/// duty for the silent-no-op catch, and vastly outside the analytic bound, so
/// it is never a plausible answer.
const SENTINEL: f32 = -7.5e30;

/// Fill `buffer` with the sentinel and PROVE it landed.
///
/// A poison that never applied reads as a passing row: without the download the
/// sentinel check below would pass on a buffer that was never poisoned at all.
fn poison(dev: *const ffi::AckDevice, buffer: *const alelyon_compute_kit::Buffer, len: usize) {
    let fill = vec![SENTINEL; len];
    let rc = unsafe { ffi::ack_upload(dev, buffer, fill.as_ptr().cast::<u8>(), len * 4) };
    assert_eq!(rc, ffi::ACK_OK, "poison upload: {}", last_error());
    let mut back = vec![0f32; len];
    let rc = unsafe { ffi::ack_download(dev, buffer, back.as_mut_ptr().cast::<u8>(), len * 4) };
    assert_eq!(rc, ffi::ACK_OK, "poison readback: {}", last_error());
    let landed = back
        .iter()
        .filter(|v| v.to_bits() == SENTINEL.to_bits())
        .count();
    assert_eq!(
        landed,
        len,
        "the poison did not apply to {} of {len} elements, so the sentinel check below would be \
         vacuous",
        len - landed
    );
}

/// Products AT each raised bound come back written, finite, in range and
/// non-zero everywhere -- and the geometry one past a bound comes back
/// UNTOUCHED.
///
/// WHY THIS EXISTS. `matmul_f32.comp`'s bound guard returns having written
/// nothing while this ABI still reports ACK_OK (see the rustdoc on
/// `ack_matmul_f32`). If the planner and the shader disagreed by one, an
/// accepted product would return whatever was already in the output buffer and
/// every counter would say it succeeded. The guard is evaluated on push
/// constants and is uniform for the whole dispatch, so such a disagreement is
/// all-or-nothing: the entire output stays at the sentinel. There is no partial
/// -write mode for it to hide in, which is why ONE accepted geometry per bound
/// detects that bound's drift.
///
/// THE ASSERTIONS ARE SEPARATE ON PURPOSE. Written-ness (no surviving
/// sentinel), finiteness, strict positivity over the WHOLE output, and the
/// analytic range are each checked with no reference to any tolerance. An
/// all-zero output is finite and in range -- and is exactly what a silently
/// refusing shader leaves behind if the buffer happened to be zeroed -- so the
/// operands are drawn strictly positive, making every true `C[i,j] >= 0.25*k`
/// and the zero output impossible BY CONSTRUCTION rather than by luck.
///
/// THE CONTROL differs in ONE input: `m = MAX_DIM + 1`, same buffers, same
/// sentinel, same everything else. It must be refused BY NAME and must leave
/// every element of C still carrying the sentinel. That is what gives the
/// sentinel check teeth: if it could not separate "written" from "untouched",
/// the over-bound case would pass it too.
///
/// WHAT IT DOES NOT ESTABLISH: the shader's other silent returns
/// (`valid_extent`, `valid_output_layout`) on geometries it does not dispatch;
/// any device other than the one it ran on; and accuracy at `k = MAX_K` beyond
/// the worst error it prints.
#[test]
fn products_at_the_raised_bounds_are_written_and_the_geometry_past_them_is_not() {
    use alelyon_compute_kit::matmul_ops::{MAX_ADDRESSED_ELEMENTS, MAX_DIM, MAX_K};

    let dev = ffi::ack_open();
    if dev.is_null() {
        if std::env::var_os("ACK_REQUIRE_DEVICE").is_some() {
            panic!(
                "matmul_f32 bounds: ACK_REQUIRE_DEVICE is set and ack_open failed: {}",
                last_error()
            );
        }
        eprintln!(
            "UNMEASURED here (matmul_f32 raised bounds): {}",
            last_error()
        );
        return;
    }

    // Operands in [0.5, 1.5], all strictly positive: every true C[i,j] is at
    // least 0.25*k, so "every element > 0" is an assertion, not a coincidence.
    let draw = |i: usize, salt: usize| {
        0.5f32 + ((i.wrapping_mul(2_654_435_761) ^ salt) % 1_024) as f32 / 1_023.0
    };

    // (name, m, n, k, why this geometry)
    let cases: [(&str, u32, u32, u32, &str); 4] = [
        ("m at MAX_DIM", MAX_DIM, 1, 1, "the m bound exactly"),
        ("n at MAX_DIM", 1, MAX_DIM, 1, "the n bound exactly"),
        (
            "k at MAX_K",
            8,
            8,
            MAX_K,
            "the reduction bound exactly: 64 outputs, each a 65,536-term reduction",
        ),
        (
            "A view past the OLD element cap",
            2_048,
            8,
            2_048,
            "an A view of 4,194,304 elements: inside the raised cap and 4x outside the pre-raise \
             2^20. The cheap LIMIT-drift detector -- with a stale LIMIT = 1048576u this dispatch \
             would silently write nothing, for 16 MiB instead of the 1 GiB the exact cap costs",
        ),
    ];

    for (name, m, n, k, why) in cases {
        let a: Vec<f32> = (0..(m as usize) * (k as usize))
            .map(|i| draw(i, 11))
            .collect();
        let b: Vec<f32> = (0..(k as usize) * (n as usize))
            .map(|i| draw(i, 29))
            .collect();
        let out_len = (m as usize) * (n as usize);

        let ba = ffi::ack_buffer_alloc(dev, (a.len() * 4) as u64);
        let bb = ffi::ack_buffer_alloc(dev, (b.len() * 4) as u64);
        let bc = ffi::ack_buffer_alloc(dev, (out_len * 4) as u64);
        assert!(
            !ba.is_null() && !bb.is_null() && !bc.is_null(),
            "{name} ({why}): allocation: {}",
            last_error()
        );
        let rc = unsafe { ffi::ack_upload(dev, ba, a.as_ptr().cast::<u8>(), a.len() * 4) };
        assert_eq!(rc, ffi::ACK_OK, "{name}: {}", last_error());
        let rc = unsafe { ffi::ack_upload(dev, bb, b.as_ptr().cast::<u8>(), b.len() * 4) };
        assert_eq!(rc, ffi::ACK_OK, "{name}: {}", last_error());
        poison(dev, bc, out_len);

        let plan = block(1, m, n, k, [0, 1, k, 1], [0, 1, n, 1], [0, 1, n, 1]);
        let mut ms = 0f64;
        let rc =
            unsafe { ffi::ack_matmul_f32(dev, ba, bb, bc, 0, plan.as_ptr(), plan.len(), &mut ms) };
        assert_eq!(rc, ffi::ACK_OK, "{name} ({why}): {}", last_error());

        let mut c = vec![0f32; out_len];
        let rc = unsafe { ffi::ack_download(dev, bc, c.as_mut_ptr().cast::<u8>(), c.len() * 4) };
        assert_eq!(rc, ffi::ACK_OK, "{name}: {}", last_error());

        // (1) written-ness: no element still carries the sentinel bits
        let survived = c
            .iter()
            .filter(|v| v.to_bits() == SENTINEL.to_bits())
            .count();
        assert_eq!(
            survived, 0,
            "{name} ({why}): {survived} of {out_len} output elements still carry the poison. The \
             shader refused this geometry and returned having written nothing, while this call \
             reported ACK_OK -- exactly the silent refusal this test exists to catch."
        );
        // (2) finiteness, over the whole output, with no tolerance in sight
        let nonfinite = c.iter().filter(|v| !v.is_finite()).count();
        assert_eq!(nonfinite, 0, "{name}: {nonfinite} non-finite outputs");
        // (3) strictly positive everywhere: counted, not sampled
        let positive = c.iter().filter(|v| **v > 0.0).count();
        assert_eq!(
            positive, out_len,
            "{name}: only {positive} of {out_len} outputs are positive, though every operand was \
             drawn in [0.5, 1.5] and every true value is at least 0.25*k"
        );
        // (4) the analytic range: |C[i,j]| <= k * max|A| * max|B|
        let bound = f64::from(k) * 1.5 * 1.5;
        let worst = c.iter().fold(0f64, |acc, v| acc.max(f64::from(*v).abs()));
        assert!(
            worst <= bound,
            "{name}: max |C| {worst} exceeds the analytic bound {bound}"
        );

        // (5) SEPARATELY, and only now, accuracy against an f64 reference built
        // from the same f32 operands. At k = MAX_K the threshold is a DECISION,
        // not an inheritance: the k=56 case above uses 1e-7 + 1e-4*|want|, which
        // would be meaningless here. This is a relative bound of 2^-18 on a
        // reduction whose exact value is O(k), chosen against the swept
        // measurement (kit relative error was ~half an f32 ulp, flat in k, on
        // benign operands -- and these operands are benign by construction,
        // being all positive, so no cancellation occurs at all).
        let mut worst_relative = 0f64;
        let samples: Vec<usize> = if out_len <= 64 {
            (0..out_len).collect()
        } else {
            (0..64).map(|s| s * (out_len / 64)).collect()
        };
        for index in samples {
            let (i, j) = (index / n as usize, index % n as usize);
            let mut want = 0f64;
            for l in 0..k as usize {
                want += f64::from(a[i * k as usize + l]) * f64::from(b[l * n as usize + j]);
            }
            let relative = (f64::from(c[index]) - want).abs() / want.abs();
            worst_relative = worst_relative.max(relative);
        }
        assert!(
            worst_relative <= 3.815e-6,
            "{name} ({why}): worst relative error {worst_relative:e} exceeds 2^-18"
        );
        eprintln!(
            "matmul_f32 bound {name}: m={m} n={n} k={k}, {out_len} outputs, no poison survived, \
             worst relative error {worst_relative:e}"
        );

        for buf in [ba, bb, bc] {
            assert_eq!(
                ffi::ack_buffer_free(dev, buf),
                ffi::ACK_OK,
                "{}",
                last_error()
            );
        }
    }

    // THE CONTROL, differing in exactly one input: m is MAX_DIM + 1.
    let m = MAX_DIM + 1;
    let (n, k) = (1u32, 1u32);
    let a: Vec<f32> = (0..m as usize).map(|i| draw(i, 11)).collect();
    let b = [1.0f32; 1];
    let out_len = m as usize;
    let ba = ffi::ack_buffer_alloc(dev, (a.len() * 4) as u64);
    let bb = ffi::ack_buffer_alloc(dev, 4);
    let bc = ffi::ack_buffer_alloc(dev, (out_len * 4) as u64);
    assert!(
        !ba.is_null() && !bb.is_null() && !bc.is_null(),
        "{}",
        last_error()
    );
    let rc = unsafe { ffi::ack_upload(dev, ba, a.as_ptr().cast::<u8>(), a.len() * 4) };
    assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
    let rc = unsafe { ffi::ack_upload(dev, bb, b.as_ptr().cast::<u8>(), 4) };
    assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
    poison(dev, bc, out_len);

    let plan = block(1, m, n, k, [0, 1, k, 1], [0, 1, n, 1], [0, 1, n, 1]);
    let mut ms = 0f64;
    let rc = unsafe { ffi::ack_matmul_f32(dev, ba, bb, bc, 0, plan.as_ptr(), plan.len(), &mut ms) };
    assert_eq!(
        rc,
        ffi::ACK_ERR_SHAPE,
        "m = MAX_DIM + 1 must be refused by the planner: {}",
        last_error()
    );
    assert!(
        last_error().contains("matmul-dimension-out-of-range"),
        "refused, but not by name: {}",
        last_error()
    );
    let mut c = vec![0f32; out_len];
    let rc = unsafe { ffi::ack_download(dev, bc, c.as_mut_ptr().cast::<u8>(), c.len() * 4) };
    assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
    let survived = c
        .iter()
        .filter(|v| v.to_bits() == SENTINEL.to_bits())
        .count();
    assert_eq!(
        survived,
        out_len,
        "the refused geometry left {} of {out_len} elements NOT carrying the sentinel. The \
         sentinel check in the accepted cases above cannot separate written from untouched, so \
         those rows prove nothing.",
        out_len - survived
    );
    eprintln!(
        "matmul_f32 bound control: m={m} (MAX_DIM + 1) refused by name, all {out_len} sentinel \
         elements intact"
    );
    for buf in [ba, bb, bc] {
        assert_eq!(
            ffi::ack_buffer_free(dev, buf),
            ffi::ACK_OK,
            "{}",
            last_error()
        );
    }

    // The one geometry that touches MAX_ADDRESSED_ELEMENTS exactly is
    // m = MAX_DIM, k = 4_096, n = 1: an A view of exactly 268,435,456 f32,
    // which is 1.00 GiB however it is shaped. It is not attempted here.
    eprintln!(
        "UNMEASURED (matmul_f32 raised bounds): a dispatch with one operand view at exactly \
         MAX_ADDRESSED_ELEMENTS ({MAX_ADDRESSED_ELEMENTS} f32 = 1.00 GiB) was NOT run. The host \
         half of that boundary is in tests/matmul_plan.rs; the device half is unmeasured."
    );

    assert_eq!(ffi::ack_close(dev), ffi::ACK_OK, "{}", last_error());
}
