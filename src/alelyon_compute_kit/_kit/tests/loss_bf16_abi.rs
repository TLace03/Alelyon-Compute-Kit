//! `ack_loss_bf16` (2026-09-28): the cross-entropy over bf16 storage, on a real
//! device.
//!
//! THE LAW IS BIT IDENTITY with the f32 entry. The bf16 entry exists to drop the
//! caller's upcast of the logits and its downcast of the gradient, and nothing
//! else. So the forward and backward over bf16 logits must equal, bit for bit,
//! `ack_loss` over the same logits widened to f32; and the bf16 gradient must
//! equal `ack_loss`'s f32 gradient rounded to nearest even as the cast kernel
//! rounds. Every output is poisoned first, so an element the kernel never writes
//! cannot pass. The operands have the vocabulary head's real width (32,000
//! classes) plus a narrow case, logits of the size a trained head produces, and
//! an ignored row. Refusals are checked by name.
//!
//! Skips as UNMEASURED without a device unless ACK_REQUIRE_DEVICE is set.
use alelyon_compute_kit::ffi;
use alelyon_compute_kit::loss_ops::{LossOp, LossReduction};
use std::ffi::{c_char, CStr};

fn last_error() -> String {
    let mut buf = vec![0 as c_char; 1024];
    let rc = unsafe { ffi::ack_last_error(buf.as_mut_ptr(), buf.len()) };
    assert!(rc >= 0, "ack_last_error returned {rc}");
    unsafe { CStr::from_ptr(buf.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

/// Round to nearest even with a quiet NaN, the cast kernel's own law.
fn to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    if (bits & 0x7f80_0000) == 0x7f80_0000 && (bits & 0x007f_ffff) != 0 {
        return ((bits >> 16) | 0x40) as u16;
    }
    let lsb = (bits >> 16) & 1;
    ((bits.wrapping_add(0x7fff + lsb)) >> 16) as u16
}

fn widen(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

struct Dev(*mut ffi::AckDevice);

impl Dev {
    fn alloc(&self, bytes: usize) -> *mut alelyon_compute_kit::Buffer {
        let b = ffi::ack_buffer_alloc(self.0, bytes as u64);
        assert!(!b.is_null(), "{}", last_error());
        b
    }
    fn upload(&self, buf: *const alelyon_compute_kit::Buffer, bytes: &[u8]) {
        let rc = unsafe { ffi::ack_upload(self.0, buf, bytes.as_ptr(), bytes.len()) };
        assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
    }
    fn download(&self, buf: *const alelyon_compute_kit::Buffer, bytes: usize) -> Vec<u8> {
        let mut out = vec![0u8; bytes];
        let rc = unsafe { ffi::ack_download(self.0, buf, out.as_mut_ptr(), bytes) };
        assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
        out
    }
}

fn open_or_unmeasured() -> Option<Dev> {
    let dev = ffi::ack_open();
    if dev.is_null() {
        if std::env::var_os("ACK_REQUIRE_DEVICE").is_some() {
            panic!(
                "loss bf16 ABI: ACK_REQUIRE_DEVICE is set and ack_open failed: {}",
                last_error()
            );
        }
        eprintln!("UNMEASURED here (loss bf16 ABI): {}", last_error());
        return None;
    }
    Some(Dev(dev))
}

#[allow(clippy::too_many_arguments)]
fn call(
    dev: &Dev,
    primary: *const alelyon_compute_kit::Buffer,
    targets_buf: *const alelyon_compute_kit::Buffer,
    upstream: *const alelyon_compute_kit::Buffer,
    out: *const alelyon_compute_kit::Buffer,
    rows: usize,
    cols: usize,
    op: LossOp,
    targets: &[i64],
    flags: Option<u32>,
) -> i32 {
    let mut ms = 0f64;
    let mut uploaded = 0u64;
    unsafe {
        match flags {
            None => ffi::ack_loss(
                dev.0,
                primary,
                targets_buf,
                upstream,
                out,
                rows as u32,
                cols as u32,
                op as u32,
                LossReduction::Sum as u32,
                -100,
                targets.as_ptr(),
                targets.len(),
                &mut uploaded,
                &mut ms,
            ),
            Some(f) => ffi::ack_loss_bf16(
                dev.0,
                primary,
                targets_buf,
                upstream,
                out,
                rows as u32,
                cols as u32,
                op as u32,
                LossReduction::Sum as u32,
                -100,
                targets.as_ptr(),
                targets.len(),
                &mut uploaded,
                &mut ms,
                f,
            ),
        }
    }
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn bf16_storage_is_the_f32_entry_bit_for_bit_and_its_refusals_are_named() {
    let Some(dev) = open_or_unmeasured() else {
        return;
    };
    for &(rows, cols) in &[(9usize, 32_000usize), (37, 1_000)] {
        let n = rows * cols;
        // logits of a trained head's size: a spread up to about +-24, with the
        // hashed draw making ties (and so exact rounding points) likely
        let logits16: Vec<u16> = (0..n)
            .map(|i| {
                let h = (i as u32).wrapping_mul(2_654_435_761) ^ 0x5bd1_e995;
                to_bf16(((h % 4_801) as f32 / 100.0 - 24.0) * if h & 8 == 0 { 1.0 } else { 0.25 })
            })
            .collect();
        let logits32: Vec<f32> = logits16.iter().map(|&h| widen(h)).collect();
        let targets: Vec<i64> = (0..rows)
            .map(|r| {
                if r == 2 {
                    -100
                } else {
                    ((r * 7_919) % cols) as i64
                }
            })
            .collect();
        let dy = [0.37f32];

        let b16 = dev.alloc(n * 2);
        let b32 = dev.alloc(n * 4);
        let bt = dev.alloc(rows * 4);
        let bu = dev.alloc(4);
        let brow = dev.alloc(rows * 4);
        let bgrad32 = dev.alloc(n * 4);
        let bgrad16 = dev.alloc(n * 2);
        dev.upload(
            b16,
            &logits16
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>(),
        );
        dev.upload(
            b32,
            &logits32
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>(),
        );
        dev.upload(bu, &dy[0].to_le_bytes());
        let poison = |buf, bytes| dev.upload(buf, &vec![0xA5u8; bytes]);

        // the forward: per-row losses from f32 logits, then from the bf16 ones
        poison(brow, rows * 4);
        assert_eq!(
            call(
                &dev,
                b32,
                bt,
                bu,
                brow,
                rows,
                cols,
                LossOp::Forward,
                &targets,
                None
            ),
            ffi::ACK_OK,
            "{}",
            last_error()
        );
        let want_rows = dev.download(brow, rows * 4);
        poison(brow, rows * 4);
        assert_eq!(
            call(
                &dev,
                b16,
                bt,
                bu,
                brow,
                rows,
                cols,
                LossOp::Forward,
                &targets,
                Some(ffi::ACK_LOSS_PRIMARY_BF16)
            ),
            ffi::ACK_OK,
            "{}",
            last_error()
        );
        let got_rows = dev.download(brow, rows * 4);
        assert_eq!(
            got_rows, want_rows,
            "{rows}x{cols}: the bf16-logit forward differs from the f32 entry"
        );
        assert!(f32s(&got_rows).iter().enumerate().all(|(r, v)| if r == 2 {
            *v == 0.0
        } else {
            v.is_finite() && *v > 0.0
        }));

        // the backward: the f32 gradient, from f32 logits and from bf16 ones
        poison(bgrad32, n * 4);
        assert_eq!(
            call(
                &dev,
                b32,
                bt,
                bu,
                bgrad32,
                rows,
                cols,
                LossOp::Backward,
                &targets,
                None
            ),
            ffi::ACK_OK,
            "{}",
            last_error()
        );
        let want_grad = dev.download(bgrad32, n * 4);
        poison(bgrad32, n * 4);
        assert_eq!(
            call(
                &dev,
                b16,
                bt,
                bu,
                bgrad32,
                rows,
                cols,
                LossOp::Backward,
                &targets,
                Some(ffi::ACK_LOSS_PRIMARY_BF16)
            ),
            ffi::ACK_OK,
            "{}",
            last_error()
        );
        assert_eq!(
            dev.download(bgrad32, n * 4),
            want_grad,
            "{rows}x{cols}: the bf16-logit backward (f32 gradient) differs"
        );

        // the bf16 gradient: the f32 one rounded as the cast rounds
        poison(bgrad16, n * 2);
        let both = ffi::ACK_LOSS_PRIMARY_BF16 | ffi::ACK_LOSS_OUTPUT_BF16;
        assert_eq!(
            call(
                &dev,
                b16,
                bt,
                bu,
                bgrad16,
                rows,
                cols,
                LossOp::Backward,
                &targets,
                Some(both)
            ),
            ffi::ACK_OK,
            "{}",
            last_error()
        );
        let got16 = dev.download(bgrad16, n * 2);
        let want16: Vec<u8> = f32s(&want_grad)
            .iter()
            .flat_map(|&v| to_bf16(v).to_le_bytes())
            .collect();
        let differing = got16
            .chunks_exact(2)
            .zip(want16.chunks_exact(2))
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(differing, 0, "{rows}x{cols}: {differing} of {n} bf16 gradient elements differ from the cast of the f32 gradient");
        // the ignored row's gradient is exactly zero
        assert!(got16[2 * cols * 2..3 * cols * 2].iter().all(|&b| b == 0));

        // refusals, by name, before any dispatch; the plan validates the targets
        // first, so each case's targets are in range for its own width
        let refused = |flags: u32, op: LossOp, primary, out, c: usize, name: &str| {
            let fit: Vec<i64> = targets
                .iter()
                .map(|&t| if t < 0 { t } else { t % c as i64 })
                .collect();
            let rc = call(&dev, primary, bt, bu, out, rows, c, op, &fit, Some(flags));
            assert_ne!(rc, ffi::ACK_OK, "{name}: accepted");
            assert!(last_error().ends_with(name), "{name}: {}", last_error());
        };
        refused(
            4,
            LossOp::Forward,
            b16,
            brow,
            cols,
            "loss-bf16-unknown-flag",
        );
        refused(
            ffi::ACK_LOSS_OUTPUT_BF16,
            LossOp::Backward,
            b32,
            bgrad16,
            cols,
            "loss-bf16-output-op",
        );
        refused(
            both,
            LossOp::Forward,
            b16,
            brow,
            cols,
            "loss-bf16-output-op",
        );
        refused(
            ffi::ACK_LOSS_PRIMARY_BF16,
            LossOp::Reduce,
            b16,
            brow,
            cols,
            "loss-bf16-logits-op",
        );
        // an odd width cannot be written in bf16 pairs
        refused(
            both,
            LossOp::Backward,
            b16,
            bgrad16,
            cols - 1,
            "loss-bf16-output-odd-cols",
        );
        // a bf16 primary one element short
        let short = dev.alloc(n * 2 - 2);
        refused(
            ffi::ACK_LOSS_PRIMARY_BF16,
            LossOp::Forward,
            short,
            brow,
            cols,
            "loss-primary-too-small",
        );
        for b in [b16, b32, bt, bu, brow, bgrad32, bgrad16, short] {
            assert_eq!(
                ffi::ack_buffer_free(dev.0, b),
                ffi::ACK_OK,
                "{}",
                last_error()
            );
        }
    }
    assert_eq!(ffi::ack_close(dev.0), ffi::ACK_OK, "{}", last_error());
}
