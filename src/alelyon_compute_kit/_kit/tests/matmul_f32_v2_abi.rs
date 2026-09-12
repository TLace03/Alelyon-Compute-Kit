//! matmul-f32/v2 through the C ABI: kinds 2 and 3 of `ack_matmul_f32` on a
//! real device, against a host f64 reference under the forward error bound of
//! plain fp32 accumulation, over row-major, transposed, batched, offset and
//! edge-shaped views; the poison-at-the-bounds law; and the refusals the plain
//! kinds share with the exact ones. Skips as UNMEASURED without a device
//! unless ACK_REQUIRE_DEVICE is set. Fixed-fixture evidence only.
//!
//! THE TOLERANCE IS A BOUND, NOT A NUMBER SOMEBODY LIKED. For c = sum over k
//! of a*b accumulated in fp32 in one order, with or without fused
//! multiply-adds, |c_computed - c_exact| <= gamma_k * sum |a*b| with
//! gamma_k ~ k * 2^-24 (the standard forward bound for recursive summation;
//! an fma contracts one rounding and cannot make it worse). The assertion
//! below uses 2 * k * 2^-24 * sum|terms| + 1e-7, which is that bound with a
//! factor of two of slack for the f64 reference's own rounding and the
//! product roundings. It is derived from the arithmetic the kernel declares,
//! so a kernel that quietly summed in a different precision would still pass
//! it -- what pins the precision is the Rust unit tests on the plan and the
//! SPIR-V constant pool, not this bound.
use alelyon_compute_kit::ffi;
use std::ffi::{c_char, CStr};

/// `ack_last_error` returns ACK_OK and fills a NUL-terminated buffer; the
/// text is what carries the refusal's name.
fn last_error() -> String {
    let mut buf = vec![0 as c_char; 1024];
    let rc = unsafe { ffi::ack_last_error(buf.as_mut_ptr(), buf.len()) };
    assert!(rc >= 0, "ack_last_error returned {rc}");
    unsafe { CStr::from_ptr(buf.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

fn block(batch: u32, m: u32, n: u32, k: u32, a: [u32; 4], b: [u32; 4], c: [u32; 4]) -> [u8; 64] {
    let words = [
        batch, m, n, k, a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3], c[0], c[1], c[2], c[3],
    ];
    let mut out = [0u8; 64];
    for (i, w) in words.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
    }
    out
}

const SENTINEL: f32 = -7.5e30;
const KIND_MM_PLAIN: u32 = 2;
const KIND_BMM_PLAIN: u32 = 3;
/// (label, batch, m, n, k, A transposed, B transposed, C column-major)
type LayoutCase = (&'static str, usize, usize, usize, usize, bool, bool, bool);

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

fn upload(dev: *mut ffi::AckDevice, buffer: *const alelyon_compute_kit::Buffer, data: &[f32]) {
    let rc = unsafe { ffi::ack_upload(dev, buffer, data.as_ptr().cast::<u8>(), data.len() * 4) };
    assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
}

fn download(
    dev: *mut ffi::AckDevice,
    buffer: *const alelyon_compute_kit::Buffer,
    len: usize,
) -> Vec<f32> {
    let mut out = vec![0f32; len];
    let rc = unsafe { ffi::ack_download(dev, buffer, out.as_mut_ptr().cast::<u8>(), len * 4) };
    assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
    out
}

/// A padded, strided operand image: `capacity` floats of sentinel with the
/// logical `[batch, rows, cols]` view written at `offset + b*bs + r*rs + c*cs`.
struct Image {
    data: Vec<f32>,
    offset: u32,
    batch_stride: u32,
    row_stride: u32,
    col_stride: u32,
}

impl Image {
    fn words(&self) -> [u32; 4] {
        [
            self.offset,
            self.batch_stride,
            self.row_stride,
            self.col_stride,
        ]
    }
    fn address(&self, b: usize, r: usize, c: usize) -> usize {
        self.offset as usize
            + b * self.batch_stride as usize
            + r * self.row_stride as usize
            + c * self.col_stride as usize
    }
}

/// `value(b, r, c)` laid out with the given strides and a padding prefix.
fn image(
    batch: usize,
    rows: usize,
    cols: usize,
    offset: u32,
    transposed: bool,
    pad_between_batches: u32,
    value: impl Fn(usize, usize, usize) -> f32,
) -> Image {
    let (row_stride, col_stride) = if transposed {
        (1u32, rows as u32)
    } else {
        (cols as u32, 1u32)
    };
    let span = (rows as u32 - 1) * row_stride + (cols as u32 - 1) * col_stride + 1;
    let batch_stride = span + pad_between_batches;
    let capacity = offset + (batch as u32 - 1) * batch_stride + span + 3;
    let mut img = Image {
        data: vec![SENTINEL; capacity as usize],
        offset,
        batch_stride,
        row_stride,
        col_stride,
    };
    for b in 0..batch {
        for r in 0..rows {
            for c in 0..cols {
                let at = img.address(b, r, c);
                img.data[at] = value(b, r, c);
            }
        }
    }
    img
}

/// The f64 reference and, beside every element, the sum of |terms| the fp32
/// forward bound is stated against.
fn reference(
    a: &Image,
    b: &Image,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut want = vec![0f64; batch * m * n];
    let mut mass = vec![0f64; batch * m * n];
    for bi in 0..batch {
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0f64;
                let mut abs = 0f64;
                for l in 0..k {
                    let t = f64::from(a.data[a.address(bi, i, l)])
                        * f64::from(b.data[b.address(bi, l, j)]);
                    acc += t;
                    abs += t.abs();
                }
                want[(bi * m + i) * n + j] = acc;
                mass[(bi * m + i) * n + j] = abs;
            }
        }
    }
    (want, mass)
}

fn fp32_bound(k: usize, mass: f64) -> f64 {
    2.0 * k as f64 * f64::from(f32::EPSILON) / 2.0 * mass + 1e-7
}

/// One product on the plain kernel through the C ABI: allocate, upload,
/// dispatch, download; every element inside the fp32 bound, every padding
/// element of the output untouched, both inputs unchanged.
#[allow(clippy::too_many_arguments)]
fn run_plain(
    dev: *mut ffi::AckDevice,
    label: &str,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    a_t: bool,
    b_t: bool,
    c_t: bool,
) -> f64 {
    let a = image(batch, m, k, 5, a_t, 7, |b, r, c| {
        (((b * 131 + r * 17 + c * 7919) % 97) as f32) / 13.0 - 3.5
    });
    let b = image(batch, k, n, 3, b_t, 11, |b, r, c| {
        (((b * 7 + r * 104_729 + c * 31) % 89) as f32) / 11.0 - 4.0
    });
    let c = image(batch, m, n, 2, c_t, 5, |_, _, _| SENTINEL);
    let (want, mass) = reference(&a, &b, batch, m, n, k);

    let ba = ffi::ack_buffer_alloc(dev, (a.data.len() * 4) as u64);
    let bb = ffi::ack_buffer_alloc(dev, (b.data.len() * 4) as u64);
    let bc = ffi::ack_buffer_alloc(dev, (c.data.len() * 4) as u64);
    assert!(
        !ba.is_null() && !bb.is_null() && !bc.is_null(),
        "{label}: {}",
        last_error()
    );
    upload(dev, ba, &a.data);
    upload(dev, bb, &b.data);
    upload(dev, bc, &c.data);

    let kind = if batch == 1 {
        KIND_MM_PLAIN
    } else {
        KIND_BMM_PLAIN
    };
    let plan = block(
        batch as u32,
        m as u32,
        n as u32,
        k as u32,
        a.words(),
        b.words(),
        c.words(),
    );
    let mut ms = 0f64;
    let rc =
        unsafe { ffi::ack_matmul_f32(dev, ba, bb, bc, kind, plan.as_ptr(), plan.len(), &mut ms) };
    assert_eq!(rc, ffi::ACK_OK, "{label}: {}", last_error());

    let got = download(dev, bc, c.data.len());
    let mut worst_ratio = 0f64;
    let mut written = 0usize;
    for bi in 0..batch {
        for i in 0..m {
            for j in 0..n {
                let at = c.address(bi, i, j);
                let value = f64::from(got[at]);
                let idx = (bi * m + i) * n + j;
                let bound = fp32_bound(k, mass[idx]);
                let err = (value - want[idx]).abs();
                assert!(
                    got[at].to_bits() != SENTINEL.to_bits(),
                    "{label}: output ({bi},{i},{j}) still carries the poison: the kernel wrote nothing there"
                );
                assert!(
                    err <= bound,
                    "{label}: ({bi},{i},{j}) got {value}, want {}, err {err:e} exceeds the fp32 bound {bound:e}",
                    want[idx]
                );
                worst_ratio = worst_ratio.max(err / bound);
                written += 1;
            }
        }
    }
    assert_eq!(written, batch * m * n);
    // padding stays poison: nothing outside the view was written
    let poison_left = got
        .iter()
        .filter(|v| v.to_bits() == SENTINEL.to_bits())
        .count();
    assert_eq!(
        poison_left,
        c.data.len() - batch * m * n,
        "{label}: an element outside the output view was written"
    );
    assert_eq!(
        download(dev, ba, a.data.len()),
        a.data,
        "{label}: A was modified"
    );
    assert_eq!(
        download(dev, bb, b.data.len()),
        b.data,
        "{label}: B was modified"
    );
    for buf in [ba, bb, bc] {
        assert_eq!(
            ffi::ack_buffer_free(dev, buf),
            ffi::ACK_OK,
            "{}",
            last_error()
        );
    }
    eprintln!("matmul_f32 v2 {label}: worst error / bound = {worst_ratio:.4}");
    worst_ratio
}

#[test]
fn plain_products_match_a_host_f64_reference_within_the_fp32_bound_on_every_layout() {
    let Some(dev) = open_or_unmeasured("matmul_f32 v2 ABI") else {
        return;
    };
    // (label, batch, m, n, k, a_t, b_t, c_t): tile-aligned, unaligned in every
    // extent, k below one k-step, k across several, a batch, and the
    // transposed and column-major layouts the adapter's views produce
    let cases: [LayoutCase; 9] = [
        ("aligned 64x64x16 nn", 1, 64, 64, 16, false, false, false),
        ("tiny 1x1x1", 1, 1, 1, 1, false, false, false),
        ("edge 70x130x50 nn", 1, 70, 130, 50, false, false, false),
        ("edge 70x130x50 tn", 1, 70, 130, 50, true, false, false),
        ("edge 70x130x50 nt", 1, 70, 130, 50, false, true, false),
        (
            "edge 70x130x50 tt, column-major C",
            1,
            70,
            130,
            50,
            true,
            true,
            true,
        ),
        ("k tail 65x65x17", 1, 65, 65, 17, false, false, false),
        ("batch 3 of 40x24x56 nn", 3, 40, 24, 56, false, false, false),
        (
            "batch 3 of 40x24x56 tn, column-major C",
            3,
            40,
            24,
            56,
            true,
            false,
            true,
        ),
    ];
    let mut worst = 0f64;
    for (label, batch, m, n, k, a_t, b_t, c_t) in cases {
        worst = worst.max(run_plain(dev, label, batch, m, n, k, a_t, b_t, c_t));
    }
    assert!(worst <= 1.0);
    eprintln!("matmul_f32 v2: worst error over every case = {worst:.4} of the fp32 bound");
    assert_eq!(ffi::ack_close(dev), ffi::ACK_OK, "{}", last_error());
}

/// The bounds are enforced in v2 on the same terms as in v1, and an accepted
/// geometry at each bound is WRITTEN: the same four cheap witnesses
/// `matmul_f32_abi.rs` runs for v1, on the plain kernel, output poisoned first.
#[test]
fn plain_products_at_the_bounds_are_written_and_the_geometry_past_them_is_not() {
    use alelyon_compute_kit::matmul_ops::{MAX_DIM, MAX_K};
    let Some(dev) = open_or_unmeasured("matmul_f32 v2 bounds") else {
        return;
    };
    let draw = |i: usize, salt: usize| {
        0.5f32 + ((i.wrapping_mul(2_654_435_761) ^ salt) % 1_024) as f32 / 1_023.0
    };
    let cases: [(&str, u32, u32, u32); 4] = [
        ("m at MAX_DIM", MAX_DIM, 1, 1),
        ("n at MAX_DIM", 1, MAX_DIM, 1),
        ("k at MAX_K", 8, 8, MAX_K),
        ("A view past the pre-raise cap", 2_048, 8, 2_048),
    ];
    for (name, m, n, k) in cases {
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
            "{name}: {}",
            last_error()
        );
        upload(dev, ba, &a);
        upload(dev, bb, &b);
        upload(dev, bc, &vec![SENTINEL; out_len]);
        let plan = block(1, m, n, k, [0, 1, k, 1], [0, 1, n, 1], [0, 1, n, 1]);
        let mut ms = 0f64;
        let rc = unsafe {
            ffi::ack_matmul_f32(
                dev,
                ba,
                bb,
                bc,
                KIND_MM_PLAIN,
                plan.as_ptr(),
                plan.len(),
                &mut ms,
            )
        };
        assert_eq!(rc, ffi::ACK_OK, "{name}: {}", last_error());
        let c = download(dev, bc, out_len);
        let survived = c
            .iter()
            .filter(|v| v.to_bits() == SENTINEL.to_bits())
            .count();
        assert_eq!(survived, 0, "{name}: {survived} of {out_len} outputs still carry the poison: v2 refused a geometry v1 accepts");
        assert_eq!(
            c.iter().filter(|v| !v.is_finite()).count(),
            0,
            "{name}: non-finite outputs"
        );
        assert_eq!(
            c.iter().filter(|v| **v > 0.0).count(),
            out_len,
            "{name}: every element must be strictly positive"
        );
        // the analytic range: operands in [0.5, 1.5] give k/4 <= c <= 9k/4
        let (lo, hi) = (0.25 * k as f64, 2.25 * k as f64);
        for v in &c {
            let v = f64::from(*v);
            assert!(
                v >= lo * (1.0 - 1e-4) && v <= hi * (1.0 + 1e-4),
                "{name}: {v} outside [{lo}, {hi}]"
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
        eprintln!("matmul_f32 v2 bounds: {name} written and in range");
    }
    // past the bound: the plan refuses before any dispatch, on either kind
    let ba = ffi::ack_buffer_alloc(dev, 4 * 64);
    let bb = ffi::ack_buffer_alloc(dev, 4 * 64);
    let bc = ffi::ack_buffer_alloc(dev, 4 * 64);
    let too_wide = block(
        1,
        MAX_DIM + 1,
        1,
        1,
        [0, 1, 1, 1],
        [0, 1, 1, 1],
        [0, 1, 1, 1],
    );
    let mut ms = 0f64;
    for kind in [0u32, KIND_MM_PLAIN] {
        let rc = unsafe {
            ffi::ack_matmul_f32(
                dev,
                ba,
                bb,
                bc,
                kind,
                too_wide.as_ptr(),
                too_wide.len(),
                &mut ms,
            )
        };
        assert_eq!(rc, ffi::ACK_ERR_SHAPE, "kind {kind}: {}", last_error());
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

/// The plain kinds refuse exactly what the exact kinds refuse, by the same
/// codes, before any device lookup, and an unknown kind is refused too.
#[test]
fn plain_kinds_share_the_exact_kinds_refusals_and_kind_four_is_refused() {
    let Some(dev) = open_or_unmeasured("matmul_f32 v2 refusals") else {
        return;
    };
    let (m, n, k) = (24u32, 40u32, 56u32);
    let ba = ffi::ack_buffer_alloc(dev, u64::from(m * k) * 4);
    let bb = ffi::ack_buffer_alloc(dev, u64::from(k * n) * 4);
    let bc = ffi::ack_buffer_alloc(dev, u64::from(m * n) * 4);
    assert!(
        !ba.is_null() && !bb.is_null() && !bc.is_null(),
        "{}",
        last_error()
    );
    let plan = block(1, m, n, k, [0, 1, k, 1], [0, 1, n, 1], [0, 1, n, 1]);
    let mut ms = 0f64;
    for (exact, plain) in [(0u32, KIND_MM_PLAIN), (1u32, KIND_BMM_PLAIN)] {
        // output aliasing an input
        let e = unsafe {
            ffi::ack_matmul_f32(dev, ba, bb, ba, exact, plan.as_ptr(), plan.len(), &mut ms)
        };
        let p = unsafe {
            ffi::ack_matmul_f32(dev, ba, bb, ba, plain, plan.as_ptr(), plan.len(), &mut ms)
        };
        assert_eq!(
            (e, p),
            (ffi::ACK_ERR_SHAPE, ffi::ACK_ERR_SHAPE),
            "aliasing, kinds {exact}/{plain}"
        );
        // a view past its buffer
        let over = block(1, m, n, k, [0, 1, 2 * k, 1], [0, 1, n, 1], [0, 1, n, 1]);
        let e = unsafe {
            ffi::ack_matmul_f32(dev, ba, bb, bc, exact, over.as_ptr(), over.len(), &mut ms)
        };
        let p = unsafe {
            ffi::ack_matmul_f32(dev, ba, bb, bc, plain, over.as_ptr(), over.len(), &mut ms)
        };
        assert_eq!(
            (e, p),
            (ffi::ACK_ERR_SIZE, ffi::ACK_ERR_SIZE),
            "capacity, kinds {exact}/{plain}"
        );
        // a zero stride
        let stride0 = block(1, m, n, k, [0, 1, 0, 1], [0, 1, n, 1], [0, 1, n, 1]);
        let e = unsafe {
            ffi::ack_matmul_f32(
                dev,
                ba,
                bb,
                bc,
                exact,
                stride0.as_ptr(),
                stride0.len(),
                &mut ms,
            )
        };
        let p = unsafe {
            ffi::ack_matmul_f32(
                dev,
                ba,
                bb,
                bc,
                plain,
                stride0.as_ptr(),
                stride0.len(),
                &mut ms,
            )
        };
        assert_eq!(
            (e, p),
            (ffi::ACK_ERR_SHAPE, ffi::ACK_ERR_SHAPE),
            "zero stride, kinds {exact}/{plain}"
        );
    }
    // a single-product kind with a batch of two is the plan's own refusal under
    // both contracts, once the buffers can hold two batches (in one-batch
    // buffers the capacity refusal fires first, as it does for v1)
    let two = block(
        2,
        m,
        n,
        k,
        [0, m * k, k, 1],
        [0, k * n, n, 1],
        [0, m * n, n, 1],
    );
    let ba2 = ffi::ack_buffer_alloc(dev, u64::from(m * k) * 8);
    let bb2 = ffi::ack_buffer_alloc(dev, u64::from(k * n) * 8);
    let bc2 = ffi::ack_buffer_alloc(dev, u64::from(m * n) * 8);
    assert!(
        !ba2.is_null() && !bb2.is_null() && !bc2.is_null(),
        "{}",
        last_error()
    );
    for kind in [0u32, KIND_MM_PLAIN] {
        let rc = unsafe {
            ffi::ack_matmul_f32(dev, ba2, bb2, bc2, kind, two.as_ptr(), two.len(), &mut ms)
        };
        assert_eq!(
            rc,
            ffi::ACK_ERR_SHAPE,
            "kind {kind} with batch 2: {}",
            last_error()
        );
        assert!(
            last_error().contains("matmul-mm-requires-single-batch"),
            "kind {kind}: refused, but not by name: {}",
            last_error()
        );
    }
    // and the batched kinds accept the same block on the same buffers
    for kind in [1u32, KIND_BMM_PLAIN] {
        let rc = unsafe {
            ffi::ack_matmul_f32(dev, ba2, bb2, bc2, kind, two.as_ptr(), two.len(), &mut ms)
        };
        assert_eq!(
            rc,
            ffi::ACK_OK,
            "kind {kind} with batch 2: {}",
            last_error()
        );
    }
    for buf in [ba2, bb2, bc2] {
        assert_eq!(
            ffi::ack_buffer_free(dev, buf),
            ffi::ACK_OK,
            "{}",
            last_error()
        );
    }
    let rc = unsafe { ffi::ack_matmul_f32(dev, ba, bb, bc, 4, plan.as_ptr(), plan.len(), &mut ms) };
    assert_eq!(
        rc,
        ffi::ACK_ERR_SHAPE,
        "kind 4 is outside schema 2: {}",
        last_error()
    );
    assert!(
        last_error().contains("matmul-operation-out-of-range"),
        "kind 4: refused, but not by name: {}",
        last_error()
    );
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
