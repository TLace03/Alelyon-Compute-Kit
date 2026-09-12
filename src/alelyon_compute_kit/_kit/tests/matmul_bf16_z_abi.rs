//! `ack_matmul_bf16_z` (ABI 10): the coopmat product with a z grid, on a real
//! device. Batch mode against a host reference on aligned and edge-shaped
//! batches with poisoned outputs; K-split mode whose partials, summed on the
//! host, equal the one-dispatch product; and the single-product entry
//! bit-identical to a z = 1 call. Skips as UNMEASURED without a device unless
//! ACK_REQUIRE_DEVICE is set. Fixed-fixture evidence only.
//!
//! THE TOLERANCE. bf16 x bf16 products are exact in f32 (two 8-bit
//! significands fit in 24 bits), so the only rounding is the accumulation:
//! |c - c_exact| <= 2 * k * 2^-24 * sum|a*b| + 1e-6, the plain-fp32 forward
//! bound on the bf16-rounded operands, whatever order the cooperative
//! matrices add in. The reference is computed from the SAME bf16-rounded
//! operands in f64, so the bf16 rounding of the inputs is not in the error.
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

/// Round to nearest even, the cast kernel's own law.
fn to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    if (bits & 0x7f80_0000) == 0x7f80_0000 && (bits & 0x007f_ffff) != 0 {
        return ((bits >> 16) | 0x40) as u16;
    }
    let lsb = (bits >> 16) & 1;
    ((bits.wrapping_add(0x7fff + lsb)) >> 16) as u16
}

fn from_bf16(v: u16) -> f32 {
    f32::from_bits(u32::from(v) << 16)
}

const SENTINEL: f32 = -7.5e30;

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

fn download_f32(
    dev: *mut ffi::AckDevice,
    buf: *const alelyon_compute_kit::Buffer,
    len: usize,
) -> Vec<f32> {
    let mut out = vec![0f32; len];
    let rc = unsafe { ffi::ack_download(dev, buf, out.as_mut_ptr().cast::<u8>(), len * 4) };
    assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
    out
}

/// A batch of `z` products with dense per-product blocks, A stored M x K or
/// (transposed) K x M, B stored K x N or N x K; values drawn per element.
struct Batch {
    z: usize,
    m: usize,
    n: usize,
    k: usize,
    a_t: bool,
    b_t: bool,
    a: Vec<u16>, // z blocks of m*k in stored order
    b: Vec<u16>, // z blocks of k*n in stored order
}

impl Batch {
    fn new(z: usize, m: usize, n: usize, k: usize, a_t: bool, b_t: bool, salt: u32) -> Self {
        let draw = |i: usize| {
            ((((i as u32).wrapping_mul(2_654_435_761) ^ salt) % 2_001) as f32) / 1_000.0 - 1.0
        };
        let mut a = vec![0u16; z * m * k];
        let mut b = vec![0u16; z * k * n];
        for bi in 0..z {
            for i in 0..m {
                for l in 0..k {
                    let logical = (bi * m + i) * k + l;
                    let stored = if a_t {
                        bi * m * k + l * m + i
                    } else {
                        bi * m * k + i * k + l
                    };
                    a[stored] = to_bf16(draw(logical));
                }
            }
            for l in 0..k {
                for j in 0..n {
                    let logical = (bi * k + l) * n + j + 7_919;
                    let stored = if b_t {
                        bi * k * n + j * k + l
                    } else {
                        bi * k * n + l * n + j
                    };
                    b[stored] = to_bf16(draw(logical));
                }
            }
        }
        Self {
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
    fn a_at(&self, bi: usize, i: usize, l: usize) -> f64 {
        let stored = if self.a_t {
            bi * self.m * self.k + l * self.m + i
        } else {
            bi * self.m * self.k + i * self.k + l
        };
        f64::from(from_bf16(self.a[stored]))
    }
    fn b_at(&self, bi: usize, l: usize, j: usize) -> f64 {
        let stored = if self.b_t {
            bi * self.k * self.n + j * self.k + l
        } else {
            bi * self.k * self.n + l * self.n + j
        };
        f64::from(from_bf16(self.b[stored]))
    }
    /// (reference, sum of |terms|) per output element, batch-major.
    fn reference(&self) -> (Vec<f64>, Vec<f64>) {
        let mut want = vec![0f64; self.z * self.m * self.n];
        let mut mass = vec![0f64; self.z * self.m * self.n];
        for bi in 0..self.z {
            for i in 0..self.m {
                for j in 0..self.n {
                    let (mut acc, mut abs) = (0f64, 0f64);
                    for l in 0..self.k {
                        let t = self.a_at(bi, i, l) * self.b_at(bi, l, j);
                        acc += t;
                        abs += t.abs();
                    }
                    want[(bi * self.m + i) * self.n + j] = acc;
                    mass[(bi * self.m + i) * self.n + j] = abs;
                }
            }
        }
        (want, mass)
    }
    fn bytes(v: &[u16]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }
}

/// (label, m, n, k, splits, kz, A transposed, B transposed)
type SplitCase = (&'static str, usize, usize, usize, usize, usize, bool, bool);

fn bound(k: usize, mass: f64) -> f64 {
    2.0 * k as f64 * 2f64.powi(-24) * mass + 1e-6
}

/// Batch mode on the aligned and the edge modules, every output inside the
/// bound, the two poisoned elements past the output untouched, the products
/// independent (a batch stride wider than a block leaves the gap untouched).
#[test]
fn batch_mode_matches_the_reference_on_aligned_and_edge_batches() {
    let Some(dev) = open_or_unmeasured("matmul_bf16_z batch") else {
        return;
    };
    // (label, z, m, n, k, a_t, b_t)
    let cases: [(&str, usize, usize, usize, usize, bool, bool); 6] = [
        ("aligned nn", 3, 128, 128, 64, false, false),
        ("aligned nt", 2, 256, 128, 32, false, true),
        ("aligned tn", 2, 128, 256, 64, true, false),
        (
            "edge nn (attention 100x48x40)",
            3,
            100,
            48,
            40,
            false,
            false,
        ),
        ("edge nt", 2, 130, 64, 96, false, true),
        ("edge tt", 2, 64, 130, 40, true, true),
    ];
    for (label, z, m, n, k, a_t, b_t) in cases {
        let batch = Batch::new(z, m, n, k, a_t, b_t, 17);
        let (want, mass) = batch.reference();
        let out_len = z * m * n;
        let ba = alloc(dev, batch.a.len() * 2);
        let bb = alloc(dev, batch.b.len() * 2);
        let bc = alloc(dev, (out_len + 2) * 4);
        upload(dev, ba, &Batch::bytes(&batch.a));
        upload(dev, bb, &Batch::bytes(&batch.b));
        let poison: Vec<u8> = vec![SENTINEL; out_len + 2]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        upload(dev, bc, &poison);
        let mut ms = 0f64;
        let rc = unsafe {
            ffi::ack_matmul_bf16_z(
                dev,
                ba,
                bb,
                bc,
                m as u32,
                n as u32,
                k as u32,
                a_t as i32,
                b_t as i32,
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
        assert_eq!(rc, ffi::ACK_OK, "{label}: {}", last_error());
        let got = download_f32(dev, bc, out_len + 2);
        assert!(
            got[out_len].to_bits() == SENTINEL.to_bits()
                && got[out_len + 1].to_bits() == SENTINEL.to_bits(),
            "{label}: the elements past the last product were written"
        );
        let mut worst = 0f64;
        for (idx, (&g, &w)) in got[..out_len].iter().zip(&want).enumerate() {
            assert!(
                g.to_bits() != SENTINEL.to_bits(),
                "{label}: output {idx} still carries the poison"
            );
            let err = (f64::from(g) - w).abs();
            let b = bound(k, mass[idx]);
            assert!(
                err <= b,
                "{label}: output {idx} got {g}, want {w}, err {err:e} > bound {b:e}"
            );
            worst = worst.max(err / b);
        }
        eprintln!("matmul_bf16_z {label}: z={z} {m}x{n}x{k}, worst error / bound = {worst:.4}");
        for buf in [ba, bb, bc] {
            assert_eq!(
                ffi::ack_buffer_free(dev, buf),
                ffi::ACK_OK,
                "{}",
                last_error()
            );
        }
    }
    // A head-interleaved batch read IN PLACE: A is stored [m][z][k] (row pitch
    // z*k, batch stride k), the layout a (tokens, heads, head_dim) activation
    // has when viewed per head; B is dense per batch. This is the layout the
    // adapter's batched arm meets on the attention products and used to refuse.
    {
        let (z, m, n, k) = (8usize, 128usize, 64usize, 64usize);
        let draw = |i: usize| {
            (((i as u32).wrapping_mul(2_654_435_761) ^ 5) % 2_001) as f32 / 1_000.0 - 1.0
        };
        let pitch = z * k;
        let mut a = vec![0u16; m * pitch];
        for bi in 0..z {
            for i in 0..m {
                for l in 0..k {
                    a[i * pitch + bi * k + l] = to_bf16(draw((bi * m + i) * k + l));
                }
            }
        }
        let dense = Batch::new(z, m, n, k, false, false, 5);
        let mut want = vec![0f64; z * m * n];
        let mut mass = vec![0f64; z * m * n];
        for bi in 0..z {
            for i in 0..m {
                for j in 0..n {
                    let (mut acc, mut abs) = (0f64, 0f64);
                    for l in 0..k {
                        let t =
                            f64::from(from_bf16(a[i * pitch + bi * k + l])) * dense.b_at(bi, l, j);
                        acc += t;
                        abs += t.abs();
                    }
                    want[(bi * m + i) * n + j] = acc;
                    mass[(bi * m + i) * n + j] = abs;
                }
            }
        }
        let ba = alloc(dev, a.len() * 2);
        let bb = alloc(dev, dense.b.len() * 2);
        let bc = alloc(dev, z * m * n * 4);
        upload(dev, ba, &Batch::bytes(&a));
        upload(dev, bb, &Batch::bytes(&dense.b));
        let mut ms = 0f64;
        let rc = unsafe {
            ffi::ack_matmul_bf16_z(
                dev,
                ba,
                bb,
                bc,
                m as u32,
                n as u32,
                k as u32,
                0,
                0,
                z as u32,
                0,
                k as u32,
                (k * n) as u32,
                (m * n) as u32,
                0,
                pitch as u32,
                0,
                &mut ms,
            )
        };
        assert_eq!(rc, ffi::ACK_OK, "interleaved: {}", last_error());
        let got = download_f32(dev, bc, z * m * n);
        let mut worst = 0f64;
        for (idx, (&g, &w)) in got.iter().zip(&want).enumerate() {
            let err = (f64::from(g) - w).abs();
            let b = bound(k, mass[idx]);
            assert!(
                err <= b,
                "interleaved: output {idx} got {g}, want {w}, err {err:e} > bound {b:e}"
            );
            worst = worst.max(err / b);
        }
        eprintln!("matmul_bf16_z head-interleaved A (pitch {pitch}, z stride {k}): worst error / bound = {worst:.4}");
        // a pitch below the stored width is refused by shape
        let rc = unsafe {
            ffi::ack_matmul_bf16_z(
                dev,
                ba,
                bb,
                bc,
                m as u32,
                n as u32,
                k as u32,
                0,
                0,
                z as u32,
                0,
                k as u32,
                (k * n) as u32,
                (m * n) as u32,
                0,
                (k - 1) as u32,
                0,
                &mut ms,
            )
        };
        assert_eq!(
            rc,
            ffi::ACK_ERR_SHAPE,
            "pitch below width: {}",
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
    }
    assert_eq!(ffi::ack_close(dev), ffi::ACK_OK, "{}", last_error());
}

/// K-split mode: the partials summed on the host equal the one-dispatch
/// product within the bound, on the pass's own 48 x 8192 x 48 shape (edge
/// modules) and on an aligned shape; and a z = 1 call of the new entry is
/// bit-identical to `ack_matmul_bf16`.
#[test]
fn split_k_partials_sum_to_the_product_and_a_single_z_is_the_old_entry() {
    let Some(dev) = open_or_unmeasured("matmul_bf16_z split") else {
        return;
    };
    // (label, m, n, k, splits, kz, a_t, b_t)
    let cases: [SplitCase; 3] = [
        (
            "edge 48x8192x48 in 32 splits",
            48,
            48,
            8192,
            32,
            256,
            false,
            false,
        ),
        (
            "aligned 128x128x4096 in 8 splits",
            128,
            128,
            4096,
            8,
            512,
            false,
            false,
        ),
        (
            "edge tn 48x48x1000 in 4 uneven splits",
            48,
            48,
            1000,
            4,
            256,
            true,
            false,
        ),
    ];
    for (label, m, n, k, splits, kz, a_t, b_t) in cases {
        let single = Batch::new(1, m, n, k, a_t, b_t, 29);
        let (want, mass) = single.reference();
        let ba = alloc(dev, single.a.len() * 2);
        let bb = alloc(dev, single.b.len() * 2);
        upload(dev, ba, &Batch::bytes(&single.a));
        upload(dev, bb, &Batch::bytes(&single.b));
        // partials: splits x (m*n), poisoned
        let part_len = splits * m * n;
        let bp = alloc(dev, (part_len + 2) * 4);
        let poison: Vec<u8> = vec![SENTINEL; part_len + 2]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        upload(dev, bp, &poison);
        let mut ms = 0f64;
        let rc = unsafe {
            ffi::ack_matmul_bf16_z(
                dev,
                ba,
                bb,
                bp,
                m as u32,
                n as u32,
                k as u32,
                a_t as i32,
                b_t as i32,
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
        assert_eq!(rc, ffi::ACK_OK, "{label}: {}", last_error());
        let partials = download_f32(dev, bp, part_len + 2);
        assert!(
            partials[part_len].to_bits() == SENTINEL.to_bits(),
            "{label}: past the partials was written"
        );
        assert_eq!(
            partials[..part_len]
                .iter()
                .filter(|v| v.to_bits() == SENTINEL.to_bits())
                .count(),
            0,
            "{label}: a partial was left unwritten"
        );
        let mut worst = 0f64;
        for idx in 0..m * n {
            let sum: f64 = (0..splits)
                .map(|s| f64::from(partials[s * m * n + idx]))
                .sum();
            let err = (sum - want[idx]).abs();
            let b = bound(k, mass[idx]);
            assert!(
                err <= b,
                "{label}: output {idx} summed {sum}, want {}, err {err:e} > bound {b:e}",
                want[idx]
            );
            worst = worst.max(err / b);
        }
        eprintln!("matmul_bf16_z {label}: worst error / bound = {worst:.4}");
        // the one-dispatch product through both entries, bit-identical
        let bc1 = alloc(dev, m * n * 4);
        let bc2 = alloc(dev, m * n * 4);
        let rc = unsafe {
            ffi::ack_matmul_bf16(
                dev, ba, bb, bc1, m as u32, n as u32, k as u32, a_t as i32, b_t as i32, &mut ms,
            )
        };
        assert_eq!(rc, ffi::ACK_OK, "{label}: {}", last_error());
        let rc = unsafe {
            ffi::ack_matmul_bf16_z(
                dev, ba, bb, bc2, m as u32, n as u32, k as u32, a_t as i32, b_t as i32, 1, 0, 0, 0,
                0, 0, 0, 0, &mut ms,
            )
        };
        assert_eq!(rc, ffi::ACK_OK, "{label}: {}", last_error());
        let one = download_f32(dev, bc1, m * n);
        let two = download_f32(dev, bc2, m * n);
        assert!(
            one.iter()
                .zip(&two)
                .all(|(x, y)| x.to_bits() == y.to_bits()),
            "{label}: a z = 1 call of the new entry is not the old entry, bit for bit"
        );
        for buf in [ba, bb, bp, bc1, bc2] {
            assert_eq!(
                ffi::ack_buffer_free(dev, buf),
                ffi::ACK_OK,
                "{}",
                last_error()
            );
        }
    }
    // refusals by code, nothing dispatched: an over-narrow z stride, a split set
    // that does not cover k, a z of zero, and an output aliasing an input
    let ba = alloc(dev, 128 * 64 * 2);
    let bb = alloc(dev, 64 * 128 * 2);
    let bc = alloc(dev, 2 * 128 * 128 * 4);
    let mut ms = 0f64;
    let rc = unsafe {
        ffi::ack_matmul_bf16_z(
            dev,
            ba,
            bb,
            bc,
            128,
            128,
            64,
            0,
            0,
            2,
            0,
            32,
            64 * 128,
            128 * 128,
            0,
            0,
            0,
            &mut ms,
        )
    };
    // a z stride below the row width (32 < 64) makes two products read overlapping rows: by shape
    assert_eq!(rc, ffi::ACK_ERR_SHAPE, "narrow za: {}", last_error());
    let rc = unsafe {
        ffi::ack_matmul_bf16_z(
            dev,
            ba,
            bb,
            bc,
            128,
            128,
            64,
            0,
            0,
            2,
            1,
            0,
            0,
            128 * 128,
            16,
            0,
            0,
            &mut ms,
        )
    };
    assert_eq!(
        rc,
        ffi::ACK_ERR_SHAPE,
        "splits not covering k: {}",
        last_error()
    );
    let rc = unsafe {
        ffi::ack_matmul_bf16_z(
            dev, ba, bb, bc, 128, 128, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &mut ms,
        )
    };
    assert_eq!(rc, ffi::ACK_ERR_SHAPE, "z = 0: {}", last_error());
    let rc = unsafe {
        ffi::ack_matmul_bf16_z(
            dev, ba, bb, ba, 128, 128, 64, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, &mut ms,
        )
    };
    assert_eq!(rc, ffi::ACK_ERR_SHAPE, "aliasing: {}", last_error());
    // batch mode past the buffer: two products in one-product buffers
    let rc = unsafe {
        ffi::ack_matmul_bf16_z(
            dev,
            ba,
            bb,
            bc,
            128,
            128,
            64,
            0,
            0,
            2,
            0,
            128 * 64,
            64 * 128,
            128 * 128,
            0,
            0,
            0,
            &mut ms,
        )
    };
    assert_eq!(
        rc,
        ffi::ACK_ERR_SIZE,
        "two products in one-product buffers: {}",
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
