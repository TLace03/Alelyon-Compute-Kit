//! `ack_matmul_f16` and `ack_matmul_f16_z` (ABI 11): the coopmat product with
//! fp16 operands, on a real device, through the C ABI.
//!
//! EXACT, NOT BOUNDED. Every operand is a multiple of 1/8 in [-1, 1], which
//! fp16 holds exactly; every product of two is a multiple of 1/64 that f32
//! holds exactly; and every sum here stays below 2^7 in magnitude, a multiple
//! of 1/64, so f32 holds it exactly whatever order the cooperative matrices add
//! in. A correct product therefore equals the f64 host reference BIT FOR BIT,
//! and the output is poisoned with NaN first, so a fragment that is never
//! written cannot pass.
//!
//! THE CONTROL. The same fp16 bits through `ack_matmul_bf16` must come out
//! wrong: bf16 reads the high byte as sign and exponent, so 0.125 (0x3000) is
//! about 4.7e-10 to it. A check that passed both would not be telling the two
//! families apart.
//!
//! The entry runs exactly when the device record advertises both
//! `ACK_FEATURE_SHADER_FLOAT16` and `ACK_FEATURE_COOPMAT_F16_16X16X16_SUBGROUP`
//! (and subgroup size control); otherwise it must refuse with
//! `ACK_ERR_UNSUPPORTED_OP`, which is checked too. Without a device every test
//! prints UNMEASURED (`ACK_REQUIRE_DEVICE=1` turns that into a failure).
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

/// Whether the device record says the fp16 product can run here.
fn advertises_f16(dev: *mut ffi::AckDevice) -> bool {
    let mut info: ffi::AckDeviceInfo = unsafe { std::mem::zeroed() };
    let rc =
        unsafe { ffi::ack_device_info(dev, &mut info, std::mem::size_of::<ffi::AckDeviceInfo>()) };
    assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
    let needed = ffi::ACK_FEATURE_SHADER_FLOAT16
        | ffi::ACK_FEATURE_COOPMAT_F16_16X16X16_SUBGROUP
        | ffi::ACK_FEATURE_SUBGROUP_SIZE_CONTROL;
    info.features & needed == needed
}

/// fp16 bits of an f32 that fp16 holds exactly (asserted).
fn f16_bits(v: f32) -> u16 {
    if v == 0.0 {
        return if v.is_sign_negative() { 0x8000 } else { 0 };
    }
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = bits & 0x7f_ffff;
    assert!(
        (1..=30).contains(&exp) && mant & 0x1fff == 0,
        "{v} is not an exact normal fp16"
    );
    sign | ((exp as u16) << 10) | (mant >> 13) as u16
}

/// A multiple of 1/8 in [-1, 1], drawn per element.
fn eighth(i: usize, salt: u32) -> f32 {
    ((((i as u32).wrapping_mul(2_654_435_761) ^ salt) % 17) as f32) / 8.0 - 1.0
}

/// `z` products with dense per-product blocks: A stored M x K or (a_t) K x M,
/// B stored K x N or (b_t) N x K. `a` and `b` are the logical values.
struct Batch {
    z: usize,
    m: usize,
    n: usize,
    k: usize,
    a_t: bool,
    b_t: bool,
    a: Vec<f32>,
    b: Vec<f32>,
}

impl Batch {
    #[allow(clippy::too_many_arguments)]
    fn new(z: usize, m: usize, n: usize, k: usize, a_t: bool, b_t: bool, salt: u32) -> Self {
        let a = (0..z * m * k).map(|i| eighth(i, salt)).collect();
        let b = (0..z * k * n).map(|i| eighth(i + 7_919, salt)).collect();
        Batch {
            z,
            m,
            n,
            k,
            a_t,
            b_t,
            a,
            b,
        }
    }

    fn a_bytes(&self) -> Vec<u8> {
        let (m, k) = (self.m, self.k);
        let mut out = Vec::with_capacity(self.a.len() * 2);
        for bi in 0..self.z {
            for s in 0..m * k {
                let (i, l) = if self.a_t {
                    (s % m, s / m)
                } else {
                    (s / k, s % k)
                };
                let v = self.a[(bi * m + i) * k + l];
                out.extend_from_slice(&f16_bits(v).to_le_bytes());
            }
        }
        out
    }

    fn b_bytes(&self) -> Vec<u8> {
        let (k, n) = (self.k, self.n);
        let mut out = Vec::with_capacity(self.b.len() * 2);
        for bi in 0..self.z {
            for s in 0..k * n {
                let (l, j) = if self.b_t {
                    (s % k, s / k)
                } else {
                    (s / n, s % n)
                };
                let v = self.b[(bi * k + l) * n + j];
                out.extend_from_slice(&f16_bits(v).to_le_bytes());
            }
        }
        out
    }

    /// The exact product of block `bi`, row-major M x N, over k in `ks`.
    fn reference(&self, bi: usize, ks: std::ops::Range<usize>) -> Vec<f64> {
        let (m, n, k) = (self.m, self.n, self.k);
        let mut out = vec![0f64; m * n];
        for i in 0..m {
            for j in 0..n {
                out[i * n + j] = ks
                    .clone()
                    .map(|l| {
                        f64::from(self.a[(bi * m + i) * k + l])
                            * f64::from(self.b[(bi * k + l) * n + j])
                    })
                    .sum();
            }
        }
        out
    }
}

struct Buffers {
    dev: *mut ffi::AckDevice,
    a: *mut alelyon_compute_kit::Buffer,
    b: *mut alelyon_compute_kit::Buffer,
    c: *mut alelyon_compute_kit::Buffer,
    c_len: usize,
}

impl Buffers {
    fn new(dev: *mut ffi::AckDevice, batch: &Batch, c_len: usize) -> Self {
        let upload = |bytes: &[u8]| {
            let buf = ffi::ack_buffer_alloc(dev, bytes.len() as u64);
            assert!(!buf.is_null(), "{}", last_error());
            let rc = unsafe { ffi::ack_upload(dev, buf, bytes.as_ptr(), bytes.len()) };
            assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
            buf
        };
        let a = upload(&batch.a_bytes());
        let b = upload(&batch.b_bytes());
        let c = upload(&f32::NAN.to_le_bytes().repeat(c_len));
        Buffers {
            dev,
            a,
            b,
            c,
            c_len,
        }
    }

    fn c(&self) -> Vec<f32> {
        let mut out = vec![0f32; self.c_len];
        let rc = unsafe {
            ffi::ack_download(
                self.dev,
                self.c,
                out.as_mut_ptr().cast::<u8>(),
                self.c_len * 4,
            )
        };
        assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
        out
    }
}

impl Drop for Buffers {
    fn drop(&mut self) {
        for buf in [self.a, self.b, self.c] {
            assert_eq!(ffi::ack_buffer_free(self.dev, buf), ffi::ACK_OK);
        }
    }
}

fn close(dev: *mut ffi::AckDevice) {
    assert_eq!(ffi::ack_close(dev), ffi::ACK_OK, "{}", last_error());
}

/// Every element equal to the reference, bit for bit; `what` names the case.
fn assert_exact(got: &[f32], want: &[f64], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}");
    let wrong: Vec<usize> = (0..got.len())
        .filter(|&i| f64::from(got[i]) != want[i])
        .collect();
    assert!(
        wrong.is_empty(),
        "{what}: {} of {} differ, first at {}: got {}, want {}",
        wrong.len(),
        got.len(),
        wrong[0],
        got[wrong[0]],
        want[wrong[0]]
    );
}

#[test]
fn fp16_products_are_exact_in_every_layout_aligned_and_off_the_tile() {
    let Some(dev) = open_or_unmeasured("fp16 single products") else {
        return;
    };
    let supported = advertises_f16(dev);
    for (m, n, k) in [(256, 128, 96), (136, 72, 40)] {
        for (a_t, b_t) in [(false, false), (false, true), (true, false), (true, true)] {
            let batch = Batch::new(1, m, n, k, a_t, b_t, 0x5eed ^ (m as u32));
            let bufs = Buffers::new(dev, &batch, m * n);
            let mut ms = 0.0f64;
            let rc = unsafe {
                ffi::ack_matmul_f16(
                    dev, bufs.a, bufs.b, bufs.c, m as u32, n as u32, k as u32, a_t as i32,
                    b_t as i32, &mut ms,
                )
            };
            if !supported {
                assert_eq!(rc, ffi::ACK_ERR_UNSUPPORTED_OP, "{}", last_error());
                eprintln!("UNMEASURED here (fp16 products): the device record advertises no fp16 cooperative matrices, and the entry refused by code");
                drop(bufs);
                close(dev);
                return;
            }
            assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
            assert!(ms.is_nan(), "a recorded dispatch reports no duration");
            let what = format!("{m}x{n}x{k} a_t={a_t} b_t={b_t}");
            assert_exact(&bufs.c(), &batch.reference(0, 0..k), &what);
        }
    }
    close(dev);
}

#[test]
fn the_fp16_z_grid_batches_and_splits_as_the_single_product_does() {
    let Some(dev) = open_or_unmeasured("fp16 z grid") else {
        return;
    };
    if !advertises_f16(dev) {
        eprintln!("UNMEASURED here (fp16 z grid): no fp16 cooperative matrices on this device");
        close(dev);
        return;
    }
    let (z, m, n, k) = (3usize, 128usize, 128usize, 64usize);
    // batch mode: three dense products in one dispatch
    let batch = Batch::new(z, m, n, k, false, true, 0xba7c);
    let bufs = Buffers::new(dev, &batch, z * m * n);
    let mut ms = 0.0f64;
    let rc = unsafe {
        ffi::ack_matmul_f16_z(
            dev,
            bufs.a,
            bufs.b,
            bufs.c,
            m as u32,
            n as u32,
            k as u32,
            0,
            1,
            z as u32,
            0,
            (m * k) as u32,
            (k * n) as u32,
            (m * n) as u32,
            0,
            0,
            0,
            &mut ms,
        )
    };
    assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
    let out = bufs.c();
    for bi in 0..z {
        let got = &out[bi * m * n..(bi + 1) * m * n];
        assert_exact(got, &batch.reference(bi, 0..k), &format!("batch {bi}"));
    }
    drop(bufs);
    // K-split mode: two splits of 32 whose partials, summed on the host, are
    // the one product; each partial is itself exact over its own k range
    let single = Batch::new(1, m, n, k, false, false, 0x5917);
    let splits = 2usize;
    let kz = k / splits;
    let bufs = Buffers::new(dev, &single, splits * m * n);
    let rc = unsafe {
        ffi::ack_matmul_f16_z(
            dev,
            bufs.a,
            bufs.b,
            bufs.c,
            m as u32,
            n as u32,
            k as u32,
            0,
            0,
            splits as u32,
            1,
            0,
            0,
            (m * n) as u32,
            kz as u32,
            0,
            0,
            &mut ms,
        )
    };
    assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
    let out = bufs.c();
    let mut summed = vec![0f64; m * n];
    for s in 0..splits {
        let part = &out[s * m * n..(s + 1) * m * n];
        assert_exact(
            part,
            &single.reference(0, s * kz..(s + 1) * kz),
            &format!("split {s}"),
        );
        for (acc, &v) in summed.iter_mut().zip(part) {
            *acc += f64::from(v);
        }
    }
    assert_eq!(
        summed,
        single.reference(0, 0..k),
        "the partials sum to the product"
    );
    drop(bufs);
    close(dev);
}

#[test]
fn the_same_bits_through_the_bf16_entry_are_wrong() {
    let Some(dev) = open_or_unmeasured("the bf16 control") else {
        return;
    };
    let (m, n, k) = (128usize, 128usize, 32usize);
    let batch = Batch::new(1, m, n, k, false, false, 0xc0de);
    let bufs = Buffers::new(dev, &batch, m * n);
    let mut ms = 0.0f64;
    let rc = unsafe {
        ffi::ack_matmul_bf16(
            dev, bufs.a, bufs.b, bufs.c, m as u32, n as u32, k as u32, 0, 0, &mut ms,
        )
    };
    if rc == ffi::ACK_ERR_UNSUPPORTED_OP {
        eprintln!("UNMEASURED here (bf16 control): no bf16 cooperative matrices on this device");
        drop(bufs);
        close(dev);
        return;
    }
    assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
    let got = bufs.c();
    let want = batch.reference(0, 0..k);
    let differ = (0..got.len())
        .filter(|&i| f64::from(got[i]) != want[i])
        .count();
    // an all-zero reference element would agree by accident; nearly all differ
    assert!(
        differ * 10 > got.len() * 9,
        "the bf16 modules read fp16 bits as a different number: only {differ} of {} differ",
        got.len()
    );
    drop(bufs);
    close(dev);
}
