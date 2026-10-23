//! `ack_swiglu` (swiglu/v1, 2026-09-29): the SiLU-gated activation of one
//! packed projection and its backward, on a real device.
//!
//! THE LAW IS BIT IDENTITY, for every result that is not a NaN, with the
//! decomposition torch runs for the storage dtype, computed on the SAME device
//! through the kit's own pointwise family, every intermediate tensor stored in
//! that dtype as torch stores it:
//! - forward: `s = silu(g)` (operation 14), then `u * s` (operation 2);
//! - backward: MulBackward's two products `grad * s` and `t = grad * u`
//!   (operation 2), then `silu_backward(t, g)` (operation 37), the two halves
//!   side by side as chunk's backward concatenates them.
//!
//! `u` and `g` are read through the packed projection's strided halves, as the
//! model's `chunk(2, -1)` hands them over. The words are compared, signed
//! zeros included. A NaN must be a NaN in both, and its sign and payload are
//! free: a multiply of two NaNs returns one of them, and which one follows the
//! operand order each compilation gives it (measured 2026-09-29: one word of
//! the edges' FP32 gradient differed only there). Every output is poisoned
//! first with a signalling NaN, and a poisoned word never counts as a NaN
//! result, so an element never written cannot pass. The gate covers the
//! saturation edges (exp overflows past |g| = 88),
//! exact zeros of both signs, infinities and a NaN. A host check in float64
//! holds the forward and both gradients to the law they round, so the chain
//! and the kernel cannot agree on a wrong law. One shape folds the grid into a
//! second dimension.
//!
//! Skips as UNMEASURED without a device unless ACK_REQUIRE_DEVICE is set.
use alelyon_compute_kit::ffi;
use alelyon_compute_kit::pointwise_ops::{
    PointwiseOp, PointwisePlan, Scalars, Storage, StridedView, FLAG_BF16, OPERANDS,
};
use alelyon_compute_kit::swiglu_ops::{SwigluOp, SwigluPlan, SwigluStorage, MAX_GROUPS_X};
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

/// f32 -> bf16 bits, round to nearest even, NaN kept quiet: the kit's `narrow`.
fn bf16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    if x.is_nan() {
        return ((bits >> 16) | 0x0040) as u16;
    }
    ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) as u16
}

fn widen(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// One storage's words: 4-byte f32 words or 2-byte bf16 words, as u32.
struct Words {
    storage: SwigluStorage,
}

impl Words {
    fn width(&self) -> usize {
        self.storage.bytes() as usize
    }
    /// The storage's value of `x`: itself, or its bf16 rounding.
    fn value(&self, x: f32) -> f32 {
        match self.storage {
            SwigluStorage::F32 => x,
            SwigluStorage::Bf16 => widen(bf16_bits(x)),
        }
    }
    fn bytes(&self, v: &[f32]) -> Vec<u8> {
        match self.storage {
            SwigluStorage::F32 => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            SwigluStorage::Bf16 => v.iter().flat_map(|x| bf16_bits(*x).to_le_bytes()).collect(),
        }
    }
    fn words(&self, bytes: &[u8]) -> Vec<u32> {
        match self.storage {
            SwigluStorage::F32 => bytes
                .chunks_exact(4)
                .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
            SwigluStorage::Bf16 => bytes
                .chunks_exact(2)
                .map(|b| u16::from_le_bytes([b[0], b[1]]) as u32)
                .collect(),
        }
    }
    fn float(&self, word: u32) -> f32 {
        match self.storage {
            SwigluStorage::F32 => f32::from_bits(word),
            SwigluStorage::Bf16 => widen(word as u16),
        }
    }
    /// A signalling NaN: every step's NaN is quiet, so no write leaves this.
    fn poison(&self, n: usize) -> Vec<u8> {
        match self.storage {
            SwigluStorage::F32 => (0..n).flat_map(|_| 0x7FA5_A5A5u32.to_le_bytes()).collect(),
            SwigluStorage::Bf16 => (0..n).flat_map(|_| 0x7FA5u16.to_le_bytes()).collect(),
        }
    }
    fn flags(&self, bf16_slots: &[usize]) -> u32 {
        match self.storage {
            SwigluStorage::F32 => 0,
            SwigluStorage::Bf16 => bf16_slots.iter().map(|&i| FLAG_BF16[i]).sum(),
        }
    }
}

struct Device(*mut ffi::AckDevice);

impl Device {
    fn alloc(&self, bytes: usize) -> *mut Buffer {
        let b = ffi::ack_buffer_alloc(self.0, bytes as u64);
        assert!(!b.is_null(), "{}", last_error());
        b
    }
    fn upload(&self, buf: *const Buffer, bytes: &[u8]) {
        let rc = unsafe { ffi::ack_upload(self.0, buf, bytes.as_ptr(), bytes.len()) };
        assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
    }
    fn download(&self, buf: *const Buffer, bytes: usize) -> Vec<u8> {
        let mut out = vec![0u8; bytes];
        let rc = unsafe { ffi::ack_download(self.0, buf, out.as_mut_ptr(), bytes) };
        assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
        out
    }
    fn free(&self, bufs: &[*mut Buffer]) {
        for &b in bufs {
            assert_eq!(ffi::ack_buffer_free(self.0, b), ffi::ACK_OK);
        }
    }
    /// One pointwise dispatch over `[rows, cols]`: x (and y) through `views`,
    /// a dense output, the operands in `bf16_slots` stored as bf16.
    #[allow(clippy::too_many_arguments)]
    fn pointwise(
        &self,
        op: PointwiseOp,
        rows: usize,
        cols: usize,
        x: (*const Buffer, StridedView),
        y: Option<(*const Buffer, StridedView)>,
        out: *const Buffer,
        flags: u32,
    ) {
        let shape = [rows as u64, cols as u64];
        let dense = StridedView::contiguous(&shape).unwrap();
        let (yb, yv) = y.unwrap_or((x.0, x.1));
        let views: [StridedView; OPERANDS] = [x.1, yv, x.1, dense];
        let storage = [
            Storage::new(1, u64::MAX / 4),
            Storage::new(2, u64::MAX / 4),
            Storage::new(3, u64::MAX / 4),
            Storage::new(4, u64::MAX / 4),
        ];
        let plan = PointwisePlan::new_with_storage_flags(
            op,
            &shape,
            views,
            Scalars::new(0.0),
            storage,
            flags,
        )
        .unwrap()
        .push_constants();
        let rc = unsafe {
            ffi::ack_pointwise(
                self.0,
                x.0,
                yb,
                x.0,
                out,
                plan.as_ptr(),
                plan.len(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, ffi::ACK_OK, "pointwise {op:?}: {}", last_error());
    }
    #[allow(clippy::too_many_arguments)]
    fn swiglu(
        &self,
        packed: *const Buffer,
        upstream: *const Buffer,
        out: *const Buffer,
        rows: usize,
        cols: usize,
        op: SwigluOp,
        storage: SwigluStorage,
    ) -> i32 {
        unsafe {
            ffi::ack_swiglu(
                self.0,
                packed,
                upstream,
                out,
                rows as u32,
                cols as u32,
                op as u32,
                storage as u32,
                std::ptr::null_mut(),
            )
        }
    }
}

/// A spread of magnitudes with exact zeros of both signs; `edges` puts the
/// gate's saturation points, infinities and a NaN at the start.
fn values(n: usize, salt: u32, span: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let h = (i as u32).wrapping_mul(2_654_435_761) ^ salt;
            match h % 89 {
                0 => 0.0,
                1 => -0.0,
                _ => {
                    let mag = [1.0f32, 0.125, 4.0, 1.0 / 64.0][(h >> 29) as usize % 4];
                    ((h % 20_011) as f32 / 10_000.0 - 1.0) * mag * span
                }
            }
        })
        .collect()
}

const EDGES: [f32; 12] = [
    -200.0,
    200.0,
    -88.5,
    88.5,
    -1.2785, // near silu's minimum, where 1 + g (1 - s) crosses zero
    0.0,
    -0.0,
    1e-6,
    -1e-6,
    f32::INFINITY,
    f32::NEG_INFINITY,
    f32::NAN,
];

fn silu64(g: f64) -> f64 {
    g / (1.0 + (-g).exp())
}

/// |got - want| within `rel` of the magnitudes the rounded steps carry, NaN
/// only where the law is NaN.
fn near(got: f32, want: f64, scale: f64, rel: f64) -> bool {
    if want.is_nan() || got.is_nan() {
        return want.is_nan() && got.is_nan();
    }
    if want.is_infinite() || got.is_infinite() {
        return f64::from(got) == want || (f64::from(got) - want).abs() <= rel * scale;
    }
    (f64::from(got) - want).abs() <= rel * (scale + want.abs()) + 1e-30
}

#[test]
fn the_forward_and_the_backward_are_the_pointwise_decomposition_bit_for_bit() {
    let Some(dev) = open_or_unmeasured("swiglu") else {
        return;
    };
    let d = Device(dev);
    // (rows, cols, storages): the A3r-X2 FFN (4 x 1024 tokens, halves of
    // 1,628), the edges, odd shapes, and past one dimension's 65,535 groups
    let both = [SwigluStorage::F32, SwigluStorage::Bf16];
    let shapes: [(usize, usize, &[SwigluStorage]); 6] = [
        (4096, 1628, &both),
        (9, 128, &both),
        (257, 3, &both),
        (3, 5, &both),
        (1, 1, &both),
        (8200, 2048, &[SwigluStorage::Bf16]),
    ];
    let mut checked = 0usize;
    let (mut nan_words, mut payload_free) = (0usize, 0usize);
    for (si, &(rows, cols, storages)) in shapes.iter().enumerate() {
        let n = rows * cols;
        for &storage in storages {
            let w = Words { storage };
            let bytes = |elements: usize| elements * w.width();
            let plan =
                SwigluPlan::new(rows as u32, cols as u32, SwigluOp::Backward, storage).unwrap();
            if si == 5 {
                assert_eq!(
                    plan.dispatch_groups()[0],
                    MAX_GROUPS_X,
                    "the fold shape folds"
                );
                assert!(plan.dispatch_groups()[1] > 1);
            }
            let mut u = values(n, 17 + si as u32, 3.0);
            let mut g = values(n, 91 + si as u32, 8.0);
            let grad = values(n, 211 + si as u32, 1.0);
            if si == 1 {
                g[..EDGES.len()].copy_from_slice(&EDGES);
                u[EDGES.len()..EDGES.len() + 2].copy_from_slice(&[f32::INFINITY, -0.0]);
            }
            let mut packed = vec![0f32; 2 * n];
            for r in 0..rows {
                packed[r * 2 * cols..r * 2 * cols + cols]
                    .copy_from_slice(&u[r * cols..(r + 1) * cols]);
                packed[r * 2 * cols + cols..(r + 1) * 2 * cols]
                    .copy_from_slice(&g[r * cols..(r + 1) * cols]);
            }
            let bpacked = d.alloc(bytes(2 * n));
            let bgrad = d.alloc(bytes(n));
            d.upload(bpacked, &w.bytes(&packed));
            d.upload(bgrad, &w.bytes(&grad));
            // the decomposition through the pointwise family
            let half = |offset: usize| StridedView::new(offset as u64, [2 * cols as u64, 1, 0, 0]);
            let dense = StridedView::contiguous(&[rows as u64, cols as u64]).unwrap();
            let [bs, bout_ref, bdu_ref, bt, bdg_ref] = [(); 5].map(|_| d.alloc(bytes(n)));
            // every operand of a bf16 call, read or not, is bf16 storage
            let flags = w.flags(&[0, 1, 2, 3]);
            d.pointwise(
                PointwiseOp::Silu,
                rows,
                cols,
                (bpacked, half(cols)),
                None,
                bs,
                flags,
            );
            d.pointwise(
                PointwiseOp::Mul,
                rows,
                cols,
                (bpacked, half(0)),
                Some((bs, dense)),
                bout_ref,
                flags,
            );
            d.pointwise(
                PointwiseOp::Mul,
                rows,
                cols,
                (bgrad, dense),
                Some((bs, dense)),
                bdu_ref,
                flags,
            );
            d.pointwise(
                PointwiseOp::Mul,
                rows,
                cols,
                (bgrad, dense),
                Some((bpacked, half(0))),
                bt,
                flags,
            );
            d.pointwise(
                PointwiseOp::SiluBackward,
                rows,
                cols,
                (bt, dense),
                Some((bpacked, half(cols))),
                bdg_ref,
                flags,
            );
            let read = |b: *const Buffer| w.words(&d.download(b, bytes(n)));
            let (out_ref, du_ref, t_ref, dg_ref) =
                (read(bout_ref), read(bdu_ref), read(bt), read(bdg_ref));
            // the fused kernel, both operations
            let bout = d.alloc(bytes(n));
            let bpacked_grad = d.alloc(bytes(2 * n));
            d.upload(bout, &w.poison(n));
            d.upload(bpacked_grad, &w.poison(2 * n));
            let rc = d.swiglu(
                bpacked,
                std::ptr::null(),
                bout,
                rows,
                cols,
                SwigluOp::Forward,
                storage,
            );
            assert_eq!(
                rc,
                ffi::ACK_OK,
                "{rows}x{cols} {storage:?} forward: {}",
                last_error()
            );
            let rc = d.swiglu(
                bpacked,
                bgrad,
                bpacked_grad,
                rows,
                cols,
                SwigluOp::Backward,
                storage,
            );
            assert_eq!(
                rc,
                ffi::ACK_OK,
                "{rows}x{cols} {storage:?} backward: {}",
                last_error()
            );
            let out = read(bout);
            let packed_grad = w.words(&d.download(bpacked_grad, bytes(2 * n)));
            let mut du = Vec::with_capacity(n);
            let mut dg = Vec::with_capacity(n);
            for r in 0..rows {
                du.extend_from_slice(&packed_grad[r * 2 * cols..r * 2 * cols + cols]);
                dg.extend_from_slice(&packed_grad[r * 2 * cols + cols..(r + 1) * 2 * cols]);
            }
            let poison = w.words(&w.poison(1))[0];
            // equal words, or a NaN for a NaN; the poison is never a result
            let same = |a: &u32, b: &u32| {
                *a != poison && (a == b || (w.float(*a).is_nan() && w.float(*b).is_nan()))
            };
            for (name, got, want) in [
                ("forward", &out, &out_ref),
                ("d_u", &du, &du_ref),
                ("d_g", &dg, &dg_ref),
            ] {
                let bad = got
                    .iter()
                    .zip(want.iter())
                    .filter(|(a, b)| !same(a, b))
                    .count();
                let first = got.iter().zip(want.iter()).position(|(a, b)| !same(a, b));
                assert_eq!(
                    bad,
                    0,
                    "{rows}x{cols} {storage:?} {name}: {bad} of {n} words differ from the pointwise decomposition \
                     (first at {first:?}: fused {:?} vs decomposition {:?}, u {:?} g {:?} grad {:?})",
                    first.map(|i| w.float(got[i])),
                    first.map(|i| w.float(want[i])),
                    first.map(|i| u[i]),
                    first.map(|i| g[i]),
                    first.map(|i| grad[i]),
                );
                nan_words += got.iter().filter(|a| w.float(**a).is_nan()).count();
                payload_free += got.iter().zip(want.iter()).filter(|(a, b)| a != b).count();
            }
            // the host holds both to the law they round, in float64
            let rel = match storage {
                SwigluStorage::F32 => 1e-5,
                SwigluStorage::Bf16 => 1.0 / 64.0,
            };
            for i in 0..n {
                let (uv, gv, gr) = (
                    w.value(u[i]) as f64,
                    w.value(g[i]) as f64,
                    w.value(grad[i]) as f64,
                );
                let s = f64::from(w.value(silu64(gv) as f32));
                let t = w.float(t_ref[i]) as f64;
                let sig = 1.0 / (1.0 + (-gv).exp());
                let checks = [
                    ("forward", w.float(out[i]), uv * s, (uv * s).abs()),
                    ("d_u", w.float(du[i]), gr * s, (gr * s).abs()),
                    (
                        "d_g",
                        w.float(dg[i]),
                        t * sig * (1.0 + gv * (1.0 - sig)),
                        t.abs() * (1.0 + gv.abs()),
                    ),
                ];
                for (name, got, want, scale) in checks {
                    assert!(
                        near(got, want, scale, rel),
                        "{rows}x{cols} {storage:?} {name}[{i}]: u {uv} g {gv} grad {gr}: got {got}, the law {want}"
                    );
                }
            }
            checked += 3 * n;
            d.free(&[
                bpacked,
                bgrad,
                bs,
                bout_ref,
                bdu_ref,
                bt,
                bdg_ref,
                bout,
                bpacked_grad,
            ]);
        }
    }
    eprintln!(
        "{checked} words checked across {} shapes, FP32 and bf16: every non-NaN word bit-identical to the \
         pointwise decomposition; {nan_words} NaN results, NaN in both, {payload_free} of them with another \
         sign or payload",
        shapes.len()
    );
    // the edges really carry NaN results, so the rule above is not vacuous
    assert!(nan_words > 0);
    assert_eq!(ffi::ack_close(dev), ffi::ACK_OK, "{}", last_error());
}

fn refused(what: &str, rc: i32, code: i32, name: &str) {
    assert_eq!(rc, code, "{what}: code");
    let e = last_error();
    assert!(e.contains(name), "{what}: {e:?} names {name:?}");
}

#[test]
fn what_the_entry_does_not_address_is_refused_by_name() {
    let Some(dev) = open_or_unmeasured("swiglu refusals") else {
        return;
    };
    let d = Device(dev);
    let (rows, cols) = (64usize, 16usize);
    let packed = d.alloc(rows * 2 * cols * 4);
    let upstream = d.alloc(rows * cols * 4);
    let out = d.alloc(rows * 2 * cols * 4);
    let small = d.alloc(16);
    let call = |p, u, o, rows: usize, cols: usize, op: u32, storage: u32| unsafe {
        ffi::ack_swiglu(
            dev,
            p,
            u,
            o,
            rows as u32,
            cols as u32,
            op,
            storage,
            std::ptr::null_mut(),
        )
    };
    refused(
        "zero rows",
        call(packed, upstream, out, 0, cols, 0, 0),
        ffi::ACK_ERR_SHAPE,
        "swiglu-dimension-out-of-range",
    );
    refused(
        "zero cols",
        call(packed, upstream, out, rows, 0, 1, 1),
        ffi::ACK_ERR_SHAPE,
        "swiglu-dimension-out-of-range",
    );
    refused(
        "past the element bound",
        call(packed, upstream, out, 1 << 20, 1 << 10, 0, 1),
        ffi::ACK_ERR_SIZE,
        "swiglu-element-limit-exceeded",
    );
    refused(
        "op 2",
        call(packed, upstream, out, rows, cols, 2, 0),
        ffi::ACK_ERR_SHAPE,
        "swiglu-operation-out-of-range",
    );
    refused(
        "storage 2",
        call(packed, upstream, out, rows, cols, 0, 2),
        ffi::ACK_ERR_SHAPE,
        "swiglu-storage-out-of-range",
    );
    refused(
        "short packed",
        call(small, upstream, out, rows, cols, 0, 0),
        ffi::ACK_ERR_SIZE,
        "swiglu-packed-too-small",
    );
    refused(
        "short upstream",
        call(packed, small, out, rows, cols, 1, 0),
        ffi::ACK_ERR_SIZE,
        "swiglu-upstream-too-small",
    );
    refused(
        "output is the upstream",
        call(packed, upstream, upstream, rows, cols, 1, 1),
        ffi::ACK_ERR_SHAPE,
        "swiglu-output-aliases-input",
    );
    let forward_sized = d.alloc(rows * cols * 4);
    refused(
        "backward output of the forward's size",
        call(packed, upstream, forward_sized, rows, cols, 1, 0),
        ffi::ACK_ERR_SIZE,
        "swiglu-output-too-small",
    );
    refused(
        "output is the packed operand",
        call(packed, upstream, packed, rows, cols, 0, 0),
        ffi::ACK_ERR_SHAPE,
        "swiglu-output-aliases-input",
    );
    // the backward needs its upstream gradient; the forward reads none
    assert_eq!(
        call(packed, std::ptr::null(), out, rows, cols, 1, 0),
        ffi::ACK_ERR_NULL
    );
    // the refusals recorded nothing: plain calls still run, the forward with no upstream
    assert_eq!(
        call(packed, std::ptr::null(), out, rows, cols, 0, 0),
        ffi::ACK_OK,
        "{}",
        last_error()
    );
    assert_eq!(
        call(packed, upstream, out, rows, cols, 1, 1),
        ffi::ACK_OK,
        "{}",
        last_error()
    );
    d.free(&[packed, upstream, out, small, forward_sized]);
    assert_eq!(ffi::ack_close(dev), ffi::ACK_OK, "{}", last_error());
}
