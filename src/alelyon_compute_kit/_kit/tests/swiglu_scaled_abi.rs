//! `ack_swiglu_scaled` (swiglu-scaled/v1, 2026-09-30): the sub-bit student's
//! experts' activation `out = silu(gate) * (up * g)` and its backward, on a
//! real device.
//!
//! THE LAW IS BIT IDENTITY, for every result that is not a NaN, with the
//! decomposition torch runs for the storage dtype, computed on the SAME device
//! through the kit's own operations, every intermediate tensor stored in that
//! dtype as torch stores it:
//! - forward: `s = silu(gate)` (pointwise 14), `b = up * g` (2), `s * b` (2);
//! - backward: `grad_s = grad * b`, `grad_b = grad * s` (MulBackward),
//!   `silu_backward(grad_s, gate)` (37), `grad_b * g` and `grad_b * up`, each
//!   of the last three written to its own buffer.
//!
//! g is one row a `width` rows, read through a broadcast view (stride 0 along
//! the width), as the experts' (experts, 1, FF) scale broadcasts. For f32 and
//! bf16 each step is one pointwise dispatch (bf16 through its storage flags);
//! for fp16 each step is the kit's composed path, as the torch adapter runs it:
//! every operand widened by convert/v1, the f32 operation, the result narrowed
//! by convert/v1. The words are compared, signed zeros included; a NaN must be
//! a NaN in both, its sign and payload free. Every output is poisoned first
//! with a signalling NaN, which never counts as a result. The gate covers
//! silu's saturation edges, exact zeros of both signs, infinities and a NaN;
//! one shape folds the grid into a second dimension.
//!
//! Skips as UNMEASURED without a device unless ACK_REQUIRE_DEVICE is set.
use alelyon_compute_kit::ffi;
use alelyon_compute_kit::pointwise_ops::{
    PointwiseOp, PointwisePlan, Scalars, Storage, StridedView, FLAG_BF16, OPERANDS,
};
use alelyon_compute_kit::swiglu_ops::{ScaledStorage, SwigluOp, SwigluScaledPlan, MAX_GROUPS_X};
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

/// f32 -> f16 bits, round to nearest even; NaN -> sign | 0x7E00 (c10's).
fn f16_bits(x: f32) -> u16 {
    let w = x.to_bits();
    let sign = ((w >> 16) & 0x8000) as u16;
    let e = (w >> 23) & 0xFF;
    let m = w & 0x7F_FFFF;
    if e == 0xFF {
        return sign | if m != 0 { 0x7E00 } else { 0x7C00 };
    }
    let he = e as i32 - 112;
    if he >= 31 {
        return sign | 0x7C00;
    }
    if he <= 0 {
        if he < -10 {
            return sign;
        }
        let mf = m | 0x80_0000;
        let shift = (14 - he) as u32;
        let mut q = mf >> shift;
        let rem = mf & ((1 << shift) - 1);
        let half = 1 << (shift - 1);
        if rem > half || (rem == half && q & 1 != 0) {
            q += 1;
        }
        return sign | q as u16;
    }
    let mut h = ((he as u32) << 10) | (m >> 13);
    let rem = m & 0x1FFF;
    if rem > 0x1000 || (rem == 0x1000 && h & 1 != 0) {
        h += 1;
    }
    sign | h as u16
}

fn f16_value(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let e = ((h >> 10) & 0x1F) as i32;
    let m = (h & 0x3FF) as f32;
    sign * match e {
        0 => m * 2f32.powi(-24),
        31 if m == 0.0 => f32::INFINITY,
        31 => f32::NAN,
        _ => (1.0 + m / 1024.0) * 2f32.powi(e - 15),
    }
}

/// One storage's words as u32: 4-byte f32 words, or 2-byte bf16/fp16 words.
struct Words {
    storage: ScaledStorage,
}

impl Words {
    fn width(&self) -> usize {
        self.storage.bytes() as usize
    }
    fn bytes(&self, v: &[f32]) -> Vec<u8> {
        match self.storage {
            ScaledStorage::F32 => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            ScaledStorage::Bf16 => v.iter().flat_map(|x| bf16_bits(*x).to_le_bytes()).collect(),
            ScaledStorage::F16 => v.iter().flat_map(|x| f16_bits(*x).to_le_bytes()).collect(),
        }
    }
    fn words(&self, bytes: &[u8]) -> Vec<u32> {
        match self.storage {
            ScaledStorage::F32 => bytes
                .chunks_exact(4)
                .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
            _ => bytes
                .chunks_exact(2)
                .map(|b| u16::from_le_bytes([b[0], b[1]]) as u32)
                .collect(),
        }
    }
    fn float(&self, word: u32) -> f32 {
        match self.storage {
            ScaledStorage::F32 => f32::from_bits(word),
            ScaledStorage::Bf16 => f32::from_bits(word << 16),
            ScaledStorage::F16 => f16_value(word as u16),
        }
    }
    /// A signalling NaN: every step's NaN is quiet, so no write leaves this.
    fn poison(&self, n: usize) -> Vec<u8> {
        match self.storage {
            ScaledStorage::F32 => (0..n).flat_map(|_| 0x7FA5_A5A5u32.to_le_bytes()).collect(),
            ScaledStorage::Bf16 => (0..n).flat_map(|_| 0x7FA5u16.to_le_bytes()).collect(),
            ScaledStorage::F16 => (0..n).flat_map(|_| 0x7D55u16.to_le_bytes()).collect(),
        }
    }
}

struct Device(*mut ffi::AckDevice);

impl Device {
    /// Whole words: `download` reads whole words, so a buffer is never smaller
    /// than its rounded read (a 105-element bf16 operand is 210 bytes).
    fn alloc(&self, bytes: usize) -> *mut Buffer {
        let b = ffi::ack_buffer_alloc(self.0, (bytes.div_ceil(4).max(1) * 4) as u64);
        assert!(!b.is_null(), "{}", last_error());
        b
    }
    /// `bytes`, padded with zeros to the buffer's whole words (ack_upload fills
    /// the buffer whole); the pad is never read as a value.
    fn upload(&self, buf: *const Buffer, bytes: &[u8]) {
        let mut padded = bytes.to_vec();
        padded.resize(bytes.len().div_ceil(4).max(1) * 4, 0);
        let rc = unsafe { ffi::ack_upload(self.0, buf, padded.as_ptr(), padded.len()) };
        assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
    }
    /// `bytes` of `buf`, read as whole words (ack_download takes whole words).
    fn download(&self, buf: *const Buffer, bytes: usize) -> Vec<u8> {
        let mut out = vec![0u8; bytes.div_ceil(4) * 4];
        let rc = unsafe { ffi::ack_download(self.0, buf, out.as_mut_ptr(), out.len()) };
        assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
        out.truncate(bytes);
        out
    }
    fn free(&self, bufs: &[*mut Buffer]) {
        for &b in bufs {
            assert_eq!(ffi::ack_buffer_free(self.0, b), ffi::ACK_OK);
        }
    }
    /// One pointwise dispatch over `shape`, x (and y) through their views, a
    /// dense output; `flags` the bf16 storage bits.
    fn pointwise(
        &self,
        op: PointwiseOp,
        shape: &[u64],
        x: (*const Buffer, StridedView),
        y: Option<(*const Buffer, StridedView)>,
        out: *const Buffer,
        flags: u32,
    ) {
        let dense = StridedView::contiguous(shape).unwrap();
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
            shape,
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
    /// convert/v1 over `n` dense elements: storage code `from` -> `to`.
    fn convert(&self, src: *const Buffer, out: *const Buffer, n: usize, from: u32, to: u32) {
        let r = ffi::AckConvertRequest {
            src_code: from,
            out_code: to,
            ndim: 1,
            flags: 0,
            shape: [n as u64, 0, 0, 0],
            in_offset: 0,
            in_strides: [1, 0, 0, 0],
            out_offset: 0,
            out_strides: [1, 0, 0, 0],
        };
        let rc = unsafe {
            ffi::ack_convert(
                self.0,
                src,
                out,
                &r,
                std::mem::size_of::<ffi::AckConvertRequest>(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, ffi::ACK_OK, "convert {from} -> {to}: {}", last_error());
    }
    #[allow(clippy::too_many_arguments)]
    fn scaled(
        &self,
        [gate, up, scale, upstream, out, out_up, out_t]: [*const Buffer; 7],
        (rows, cols, width): (usize, usize, usize),
        op: SwigluOp,
        storage: ScaledStorage,
    ) -> i32 {
        unsafe {
            ffi::ack_swiglu_scaled(
                self.0,
                gate,
                up,
                scale,
                upstream,
                out,
                out_up,
                out_t,
                rows as u32,
                cols as u32,
                width as u32,
                op as u32,
                storage as u32,
                std::ptr::null_mut(),
            )
        }
    }
}

/// A spread of magnitudes with exact zeros of both signs.
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
    -1.2785,
    0.0,
    -0.0,
    1e-6,
    -1e-6,
    f32::INFINITY,
    f32::NEG_INFINITY,
    f32::NAN,
];

/// The decomposition on the device, one storage: `step(op, x, y) -> stored
/// result`, each operand a (buffer, view) over the [experts, width, cols] shape.
struct Decomposition<'a> {
    d: &'a Device,
    storage: ScaledStorage,
    shape: [u64; 3],
    /// every buffer this decomposition allocated, freed at the end
    live: Vec<*mut Buffer>,
}

impl Decomposition<'_> {
    fn n(&self) -> usize {
        (self.shape[0] * self.shape[1] * self.shape[2]) as usize
    }
    fn fresh(&mut self, bytes: usize) -> *mut Buffer {
        let b = self.d.alloc(bytes);
        self.live.push(b);
        b
    }
    /// One step of the law: `op` over x (and y), rounded to the storage. An
    /// fp16 step is the composed path: both operands widened by convert/v1
    /// (over the elements their views reach, `x_len`/`y_len`), the f32
    /// operation, the result narrowed by convert/v1.
    #[allow(clippy::too_many_arguments)]
    fn step(
        &mut self,
        op: PointwiseOp,
        x: (*const Buffer, StridedView, usize),
        y: Option<(*const Buffer, StridedView, usize)>,
    ) -> *mut Buffer {
        let n = self.n();
        let shape = self.shape;
        match self.storage {
            ScaledStorage::F32 => {
                let out = self.fresh(4 * n);
                self.d
                    .pointwise(op, &shape, (x.0, x.1), y.map(|y| (y.0, y.1)), out, 0);
                out
            }
            ScaledStorage::Bf16 => {
                let out = self.fresh(2 * n);
                let flags: u32 = FLAG_BF16.iter().sum();
                self.d
                    .pointwise(op, &shape, (x.0, x.1), y.map(|y| (y.0, y.1)), out, flags);
                out
            }
            ScaledStorage::F16 => {
                let wide_x = self.fresh(4 * x.2);
                self.d.convert(x.0, wide_x, x.2, 2, 0);
                let wide_y = y.map(|y| {
                    let b = self.fresh(4 * y.2);
                    self.d.convert(y.0, b, y.2, 2, 0);
                    (b as *const Buffer, y.1)
                });
                let wide_out = self.fresh(4 * n);
                self.d
                    .pointwise(op, &shape, (wide_x, x.1), wide_y, wide_out, 0);
                let out = self.fresh(2 * n);
                self.d.convert(wide_out, out, n, 0, 2);
                out
            }
        }
    }
}

#[test]
fn the_forward_and_the_backward_are_the_kits_decomposition_bit_for_bit() {
    let Some(dev) = open_or_unmeasured("swiglu-scaled") else {
        return;
    };
    let d = Device(dev);
    let all = [ScaledStorage::F32, ScaledStorage::Bf16, ScaledStorage::F16];
    // (experts, width, cols): a stage-E group (12 experts of 640 rows, FF 768),
    // the edges, odd shapes, and past one dimension's 65,535 groups
    let shapes: [(usize, usize, usize, &[ScaledStorage]); 5] = [
        (12, 640, 768, &all),
        (1, 37, 12, &all),
        (3, 5, 7, &all),
        (1, 1, 1, &all),
        (4, 4096, 1100, &[ScaledStorage::F16]),
    ];
    let (mut checked, mut nan_words, mut payload_free) = (0usize, 0usize, 0usize);
    for (si, &(experts, width, cols, storages)) in shapes.iter().enumerate() {
        let rows = experts * width;
        let n = rows * cols;
        let ns = experts * cols;
        for &storage in storages {
            let w = Words { storage };
            let bytes = |elements: usize| elements * w.width();
            let plan = SwigluScaledPlan::new(
                rows as u32,
                cols as u32,
                width as u32,
                SwigluOp::Backward,
                storage,
            )
            .unwrap();
            if si == 4 {
                assert_eq!(
                    plan.dispatch_groups()[0],
                    MAX_GROUPS_X,
                    "the fold shape folds"
                );
                assert!(plan.dispatch_groups()[1] > 1);
            }
            let mut gate = values(n, 17 + si as u32, 8.0);
            let mut up = values(n, 91 + si as u32, 3.0);
            let scale = values(ns, 57 + si as u32, 2.0);
            let grad = values(n, 211 + si as u32, 1.0);
            if si == 1 {
                gate[..EDGES.len()].copy_from_slice(&EDGES);
                up[EDGES.len()..EDGES.len() + 2].copy_from_slice(&[f32::INFINITY, -0.0]);
            }
            let [bgate, bup, bgrad] = [&gate, &up, &grad].map(|v| {
                let b = d.alloc(bytes(n));
                d.upload(b, &w.bytes(v));
                b
            });
            let bscale = d.alloc(bytes(ns));
            d.upload(bscale, &w.bytes(&scale));
            // the decomposition on the device
            let shape = [experts as u64, width as u64, cols as u64];
            let dense = StridedView::contiguous(&shape).unwrap();
            let bcast = StridedView::new(0, [cols as u64, 0, 1, 0]);
            let mut dec = Decomposition {
                d: &d,
                storage,
                shape,
                live: vec![],
            };
            let s = dec.step(PointwiseOp::Silu, (bgate, dense, n), None);
            let b = dec.step(PointwiseOp::Mul, (bup, dense, n), Some((bscale, bcast, ns)));
            let out_ref = dec.step(PointwiseOp::Mul, (s, dense, n), Some((b, dense, n)));
            let grad_s = dec.step(PointwiseOp::Mul, (bgrad, dense, n), Some((b, dense, n)));
            let grad_b = dec.step(PointwiseOp::Mul, (bgrad, dense, n), Some((s, dense, n)));
            let dgate_ref = dec.step(
                PointwiseOp::SiluBackward,
                (grad_s, dense, n),
                Some((bgate, dense, n)),
            );
            let dup_ref = dec.step(
                PointwiseOp::Mul,
                (grad_b, dense, n),
                Some((bscale, bcast, ns)),
            );
            let t_ref = dec.step(PointwiseOp::Mul, (grad_b, dense, n), Some((bup, dense, n)));
            let read = |b: *const Buffer, elements: usize| w.words(&d.download(b, bytes(elements)));
            let (out_ref, dgate_ref, dup_ref, t_ref) = (
                read(out_ref, n),
                read(dgate_ref, n),
                read(dup_ref, n),
                read(t_ref, n),
            );
            // the fused kernel, both operations, every output poisoned first
            let [bout, bdgate, bdup, bt] = [0; 4].map(|_| {
                let b = d.alloc(bytes(n));
                d.upload(b, &w.poison(n));
                b
            });
            let dims = (rows, cols, width);
            let none = std::ptr::null();
            let rc = d.scaled(
                [bgate, bup, bscale, none, bout, none, none],
                dims,
                SwigluOp::Forward,
                storage,
            );
            assert_eq!(
                rc,
                ffi::ACK_OK,
                "{shape:?} {storage:?} forward: {}",
                last_error()
            );
            let rc = d.scaled(
                [bgate, bup, bscale, bgrad, bdgate, bdup, bt],
                dims,
                SwigluOp::Backward,
                storage,
            );
            assert_eq!(
                rc,
                ffi::ACK_OK,
                "{shape:?} {storage:?} backward: {}",
                last_error()
            );
            let (out, dgate, dup, t) = (read(bout, n), read(bdgate, n), read(bdup, n), read(bt, n));
            let poison = w.words(&w.poison(1))[0];
            let same = |a: &u32, b: &u32| {
                *a != poison && (a == b || (w.float(*a).is_nan() && w.float(*b).is_nan()))
            };
            for (name, got, want) in [
                ("forward", &out[..], &out_ref[..]),
                ("d_gate", &dgate[..], &dgate_ref[..]),
                ("d_up", &dup[..], &dup_ref[..]),
                ("t (g's gradient before its sum)", &t[..], &t_ref[..]),
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
                    "{shape:?} {storage:?} {name}: {bad} of {n} words differ from the decomposition \
                     (first at {first:?}: fused {:?} vs decomposition {:?}, gate {:?} up {:?} grad {:?})",
                    first.map(|i| w.float(got[i])),
                    first.map(|i| w.float(want[i])),
                    first.map(|i| gate[i]),
                    first.map(|i| up[i]),
                    first.map(|i| grad[i]),
                );
                nan_words += got.iter().filter(|a| w.float(**a).is_nan()).count();
                payload_free += got.iter().zip(want.iter()).filter(|(a, b)| a != b).count();
                checked += n;
            }
            let mut owned = vec![bgate, bup, bgrad, bscale, bout, bdgate, bdup, bt];
            owned.append(&mut dec.live);
            d.free(&owned);
        }
    }
    eprintln!(
        "swiglu-scaled: {checked} words bit-identical to the decomposition; {nan_words} NaN results, \
         {payload_free} of them with another sign or payload"
    );
    assert!(nan_words > 0, "the edges reach NaN results");
    assert_eq!(ffi::ack_close(dev), ffi::ACK_OK, "{}", last_error());
}

#[test]
fn a_call_the_family_does_not_address_is_refused_by_name_before_it_runs() {
    let Some(dev) = open_or_unmeasured("swiglu-scaled refusals") else {
        return;
    };
    let d = Device(dev);
    let b = d.alloc(4 * 64);
    let refused = |rc: i32, code: i32, name: &str| {
        assert_eq!(rc, code, "{name}");
        assert!(last_error().contains(name), "{name}: {}", last_error());
    };
    let f32s = ScaledStorage::F32;
    let fwd = SwigluOp::Forward;
    let none = std::ptr::null();
    let bufs = [b as *const Buffer, b, b, none, b, none, none];
    // a width that does not divide the rows, a zero dimension
    refused(
        d.scaled(bufs, (9, 4, 2), fwd, f32s),
        ffi::ACK_ERR_SHAPE,
        "swiglu-dimension-out-of-range",
    );
    refused(
        d.scaled(bufs, (0, 4, 1), fwd, f32s),
        ffi::ACK_ERR_SHAPE,
        "swiglu-dimension-out-of-range",
    );
    // an unknown storage and operation
    let rc = unsafe {
        ffi::ack_swiglu_scaled(
            dev,
            b,
            b,
            b,
            none,
            b,
            none,
            none,
            4,
            4,
            1,
            0,
            3,
            std::ptr::null_mut(),
        )
    };
    refused(rc, ffi::ACK_ERR_SHAPE, "swiglu-storage-out-of-range");
    // the output is an input; the backward's outputs are one buffer
    let [other, o1, o2] = [0; 3].map(|_| d.alloc(4 * 64));
    refused(
        d.scaled([b, other, other, none, b, none, none], (4, 4, 1), fwd, f32s),
        ffi::ACK_ERR_SHAPE,
        "swiglu-output-aliases-input",
    );
    let bwd = SwigluOp::Backward;
    refused(
        d.scaled([b, b, b, b, o1, o2, b], (4, 4, 1), bwd, f32s),
        ffi::ACK_ERR_SHAPE,
        "swiglu-output-aliases-input",
    );
    refused(
        d.scaled([b, b, b, b, o1, o2, o1], (4, 4, 1), bwd, f32s),
        ffi::ACK_ERR_SHAPE,
        "swiglu-output-aliases-output",
    );
    // a buffer smaller than its role
    let small = d.alloc(4);
    refused(
        d.scaled(
            [small, other, other, none, b, none, none],
            (4, 4, 1),
            fwd,
            f32s,
        ),
        ffi::ACK_ERR_SIZE,
        "swiglu-gate-too-small",
    );
    refused(
        d.scaled([b, b, b, b, o1, small, o2], (4, 4, 1), bwd, f32s),
        ffi::ACK_ERR_SIZE,
        "swiglu-up-grad-too-small",
    );
    d.free(&[b, other, o1, o2, small]);
    assert_eq!(ffi::ack_close(dev), ffi::ACK_OK, "{}", last_error());
}
