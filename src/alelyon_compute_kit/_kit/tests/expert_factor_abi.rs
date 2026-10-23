//! `ack_expert_factor_f16` (expert-factor-f16/v1, 2026-09-29): the sub-bit
//! student's expert factors from their signs, and their backward, on a real
//! device.
//!
//! THE LAW is `expert_factor_ops::Reference`, the kernel's arithmetic on the
//! host:
//! - the forward BIT FOR BIT: one f32 scale product, one fp16 rounding, an
//!   exact sign;
//! - `g * sign` bit for bit;
//! - the latent's gradient to 2e-6 relative (tanh's own rounding);
//! - the scales' sums to 1e-5 of the magnitudes they add.
//!
//! The cases cover a latent part with both scales at a column offset, a
//! packed part with a row scale only, an unscaled latent part, routed ids out
//! of order, and an id out of range (its rows NaN, no latent row written).
//! The int8 sign cache (2026-09-30, the student's frozen attention factors)
//! takes the cache ops' forward and both scale gradients, and a byte that is
//! not +1 or -1 makes exactly its own entries NaN.
//! Every output is poisoned first, and the latent gradient's unrouted rows
//! must keep their poison. Refusals are checked by name. Skips as UNMEASURED
//! without a device unless ACK_REQUIRE_DEVICE is set.
use alelyon_compute_kit::expert_factor_ops::{
    ExpertFactorOp, ExpertFactorPlan, Reference, HAS_COL, HAS_ROW, INT8_SIGNS,
};
use alelyon_compute_kit::ffi;
use alelyon_compute_kit::head_loss_ops::{f16_to_f32, f32_to_f16};
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

type Buf = *mut alelyon_compute_kit::Buffer;

fn alloc(dev: *mut ffi::AckDevice, bytes: &[u8]) -> Buf {
    let b = ffi::ack_buffer_alloc(dev, bytes.len().max(4) as u64);
    assert!(!b.is_null(), "{}", last_error());
    let rc = unsafe { ffi::ack_upload(dev, b, bytes.as_ptr(), bytes.len()) };
    assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
    b
}

fn download(dev: *mut ffi::AckDevice, buf: Buf, bytes: usize) -> Vec<u8> {
    let mut out = vec![0u8; bytes];
    let rc = unsafe { ffi::ack_download(dev, buf, out.as_mut_ptr(), bytes) };
    assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
    out
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn u16s(b: &[u8]) -> Vec<u16> {
    b.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn hash(i: usize, salt: u32) -> u32 {
    (i as u32).wrapping_mul(2_654_435_761) ^ salt.wrapping_mul(40_503)
}

fn unit(i: usize, salt: u32) -> f32 {
    (hash(i, salt) % 20_011) as f32 / 10_000.0 - 1.0
}

const E: usize = 6;
const M: usize = 70;
const R: usize = 41;
const R8: usize = 48;
const LD: usize = 96;
const K: f32 = 3.0;

struct Inputs {
    latent: Vec<f32>,
    packed: Vec<u8>,
    int8: Vec<i8>,
    ids: Vec<i64>,
    row: Vec<f32>,
    col: Vec<f32>,
}

fn inputs() -> Inputs {
    let nb = R.div_ceil(8);
    let mut latent: Vec<f32> = (0..E * M * R).map(|i| unit(i, 11) * 0.5).collect();
    latent[3] = -0.0;
    latent[7] = 0.0;
    Inputs {
        latent,
        packed: (0..E * M * nb).map(|i| hash(i, 13) as u8).collect(),
        // _sign_cache's form: +1 or -1 a byte
        int8: (0..E * M * R)
            .map(|i| if hash(i, 29) & 4 != 0 { 1 } else { -1 })
            .collect(),
        ids: vec![5, 0, 3, 2],
        row: (0..4 * M).map(|i| unit(i, 17) * 0.01).collect(),
        col: (0..4 * R).map(|i| 0.5 + unit(i, 19)).collect(),
    }
}

fn part(factor: &[u16], p: usize) -> Vec<u16> {
    let mut out = Vec::with_capacity(p * M * R8);
    for row in factor.chunks_exact(LD).take(p * M) {
        out.extend_from_slice(&row[..R8]);
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn call(
    dev: *mut ffi::AckDevice,
    bufs: [Buf; 7],
    op: ExpertFactorOp,
    flags: u32,
    p: usize,
    off: u32,
) -> i32 {
    unsafe {
        ffi::ack_expert_factor_f16(
            dev,
            bufs[0],
            bufs[1],
            bufs[2],
            bufs[3],
            bufs[4],
            bufs[5],
            bufs[6],
            op as u32,
            flags,
            E as u32,
            p as u32,
            M as u32,
            R as u32,
            R8 as u32,
            LD as u32,
            off,
            K,
            std::ptr::null_mut(),
        )
    }
}

#[test]
fn the_forward_is_the_law_bit_for_bit_and_the_backward_within_its_tolerances() {
    let Some(dev) = open_or_unmeasured("expert factor") else {
        return;
    };
    let x = inputs();
    let p = x.ids.len();
    let i64b: Vec<u8> = x.ids.iter().flat_map(|v| v.to_le_bytes()).collect();
    let b_lat = alloc(dev, &f32_bytes(&x.latent));
    let b_pack = alloc(dev, &x.packed);
    let int8_bytes: Vec<u8> = x.int8.iter().map(|&v| v as u8).collect();
    let b_i8 = alloc(dev, &int8_bytes);
    let b_ids = alloc(dev, &i64b);
    let b_row = alloc(dev, &f32_bytes(&x.row));
    let b_col = alloc(dev, &f32_bytes(&x.col));
    let spare = alloc(dev, &[0u8; 16]);
    let spare2 = alloc(dev, &[0u8; 16]);
    let checked_bits = |factor: &[u16], want: &[u16], what: &str| {
        let got = part(factor, p);
        let bad = got.iter().zip(want).filter(|(g, w)| g != w).count();
        assert_eq!(
            bad,
            0,
            "{what}: {bad} of {} fp16 entries differ from the law",
            want.len()
        );
    };
    for (op, flags, src, off) in [
        (
            ExpertFactorOp::LatentForward,
            HAS_ROW | HAS_COL,
            b_lat,
            0u32,
        ),
        (ExpertFactorOp::LatentForward, 0, b_lat, 48),
        (ExpertFactorOp::PackedForward, HAS_ROW, b_pack, 48),
        (
            ExpertFactorOp::PackedForward,
            HAS_ROW | HAS_COL | INT8_SIGNS,
            b_i8,
            0,
        ),
        (
            ExpertFactorOp::PackedForward,
            HAS_ROW | INT8_SIGNS,
            b_i8,
            48,
        ),
    ] {
        let plan = ExpertFactorPlan::new(
            op, flags, E as u32, p as u32, M as u32, R as u32, R8 as u32, LD as u32, off, K,
        )
        .unwrap();
        let reference = Reference {
            plan: &plan,
            latent: &x.latent,
            packed: &x.packed,
            int8: &x.int8,
            ids: &x.ids,
            row: &x.row,
            col: &x.col,
        };
        let factor = alloc(dev, &vec![0xA5u8; p * M * LD * 2]);
        let rc = call(
            dev,
            [src, b_ids, b_row, b_col, factor, spare, spare2],
            op,
            flags,
            p,
            off,
        );
        assert_eq!(rc, ffi::ACK_OK, "{op:?}: {}", last_error());
        let all = u16s(&download(dev, factor, p * M * LD * 2));
        let shifted: Vec<u16> = all
            .chunks_exact(LD)
            .flat_map(|r| {
                let mut row = r[off as usize..].to_vec();
                row.resize(LD, 0);
                row
            })
            .collect();
        checked_bits(
            &shifted,
            &reference.forward(),
            &format!("{op:?} flags {flags} at {off}"),
        );
        // the columns outside this part keep their poison
        let outside = all.chunks_exact(LD).all(|r| {
            (0..LD)
                .filter(|c| (*c as u32) < off || *c as u32 >= off + R8 as u32)
                .all(|c| r[c] == 0xA5A5)
        });
        assert!(
            outside,
            "{op:?}: a column outside [{off}, {}) was written",
            off as usize + R8
        );
        assert_eq!(ffi::ack_buffer_free(dev, factor), ffi::ACK_OK);
    }

    // the backward from an upstream gradient at fp16
    let grad: Vec<u16> = (0..p * M * LD)
        .map(|i| f32_to_f16(unit(i, 23) * 0.25))
        .collect();
    let grad_part = part(&grad, p);
    let b_grad = alloc(
        dev,
        &grad
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>(),
    );
    for (op, flags, src) in [
        (ExpertFactorOp::LatentBackward, HAS_ROW | HAS_COL, b_lat),
        (ExpertFactorOp::LatentBackward, HAS_ROW, b_lat),
        (ExpertFactorOp::PackedBackwardRow, HAS_ROW | HAS_COL, b_pack),
        (ExpertFactorOp::PackedBackwardCol, HAS_ROW | HAS_COL, b_pack),
        (
            ExpertFactorOp::PackedBackwardRow,
            HAS_ROW | HAS_COL | INT8_SIGNS,
            b_i8,
        ),
        (
            ExpertFactorOp::PackedBackwardCol,
            HAS_ROW | HAS_COL | INT8_SIGNS,
            b_i8,
        ),
    ] {
        let plan = ExpertFactorPlan::new(
            op, flags, E as u32, p as u32, M as u32, R as u32, R8 as u32, LD as u32, 0, K,
        )
        .unwrap();
        let reference = Reference {
            plan: &plan,
            latent: &x.latent,
            packed: &x.packed,
            int8: &x.int8,
            ids: &x.ids,
            row: &x.row,
            col: &x.col,
        };
        let (gw, gs, g_row, g_col) = reference.backward(&grad_part, K);
        let a_bytes = (plan.out_a_bytes() as usize).max(4);
        let b_bytes = (plan.out_b_bytes() as usize).max(4);
        let out_a = alloc(dev, &vec![0xA5u8; a_bytes]);
        let out_b = alloc(dev, &vec![0xA5u8; b_bytes]);
        let rc = call(
            dev,
            [src, b_ids, b_row, b_col, b_grad, out_a, out_b],
            op,
            flags,
            p,
            0,
        );
        assert_eq!(rc, ffi::ACK_OK, "{op:?} flags {flags}: {}", last_error());
        let a = f32s(&download(dev, out_a, a_bytes));
        let b = f32s(&download(dev, out_b, b_bytes));
        let near = |got: f32, want: f64, scale: f64, tol: f64| {
            (f64::from(got) - want).abs() <= tol * scale + 1e-30
        };
        match op {
            ExpertFactorOp::LatentBackward => {
                for (q, &e) in x.ids.iter().enumerate() {
                    for i in 0..M * R {
                        let want = gw[q * M * R + i];
                        let got = a[e as usize * M * R + i];
                        assert!(
                            near(got, want, want.abs(), 2e-6),
                            "gW[{e}][{i}]: {got} against {want}"
                        );
                    }
                }
                let routed: Vec<usize> = x.ids.iter().map(|&e| e as usize).collect();
                for e in (0..E).filter(|e| !routed.contains(e)) {
                    assert!(
                        a[e * M * R..(e + 1) * M * R]
                            .iter()
                            .all(|v| v.to_bits() == 0xA5A5_A5A5),
                        "unrouted {e} written"
                    );
                }
                if flags == HAS_ROW | HAS_COL {
                    let bad = b
                        .iter()
                        .zip(&gs)
                        .filter(|(g, w)| f64::from(**g) != **w)
                        .count();
                    assert_eq!(bad, 0, "gs: {bad} entries differ from sign * g");
                } else {
                    for (i, &want) in g_row.iter().enumerate() {
                        let mag: f64 = (0..R).map(|c| gs[i * R + c].abs()).sum();
                        assert!(
                            near(b[i], want, mag, 1e-5),
                            "g_row[{i}]: {} against {want}",
                            b[i]
                        );
                    }
                }
            }
            ExpertFactorOp::PackedBackwardRow => {
                for (i, &want) in g_row.iter().enumerate() {
                    let mag: f64 = (0..R)
                        .map(|c| (gs[i * R + c] * f64::from(x.col[(i / M) * R + c])).abs())
                        .sum();
                    assert!(
                        near(a[i], want, mag, 1e-5),
                        "g_row[{i}]: {} against {want}",
                        a[i]
                    );
                }
            }
            _ => {
                for (i, &want) in g_col.iter().enumerate() {
                    let (q, c) = (i / R, i % R);
                    let mag: f64 = (0..M)
                        .map(|m| (gs[(q * M + m) * R + c] * f64::from(x.row[q * M + m])).abs())
                        .sum();
                    assert!(
                        near(a[i], want, mag, 1e-5),
                        "g_col[{i}]: {} against {want}",
                        a[i]
                    );
                }
            }
        }
        for buf in [out_a, out_b] {
            assert_eq!(ffi::ack_buffer_free(dev, buf), ffi::ACK_OK);
        }
    }
    // an id out of range: its rows NaN in the forward, and never an address
    let bad_ids: Vec<u8> = [5i64, 6, 0, 1]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let b_bad = alloc(dev, &bad_ids);
    let factor = alloc(dev, &vec![0u8; p * M * LD * 2]);
    let rc = call(
        dev,
        [b_lat, b_bad, b_row, b_col, factor, spare, spare2],
        ExpertFactorOp::LatentForward,
        HAS_ROW | HAS_COL,
        p,
        0,
    );
    assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
    let all = u16s(&download(dev, factor, p * M * LD * 2));
    assert!(all[M * LD..2 * M * LD]
        .chunks_exact(LD)
        .all(|r| r[..R8].iter().all(|&v| f16_to_f32(v).is_nan())));
    assert!(all[..M * LD]
        .chunks_exact(LD)
        .all(|r| r[..R].iter().all(|&v| !f16_to_f32(v).is_nan())));
    // an int8 byte that is not a sign (0 and 2): exactly its entries NaN in the factor, and exactly its rows'
    // and columns' scale gradients NaN; everything else is the law
    let mut odd = x.int8.clone();
    let (z, t) = ((5 * M + 2) * R + 1, (3 * M + 9) * R + 30);
    odd[z] = 0;
    odd[t] = 2;
    let odd_bytes: Vec<u8> = odd.iter().map(|&v| v as u8).collect();
    let b_odd = alloc(dev, &odd_bytes);
    let flags = HAS_ROW | HAS_COL | INT8_SIGNS;
    let plan = ExpertFactorPlan::new(
        ExpertFactorOp::PackedForward,
        flags,
        E as u32,
        p as u32,
        M as u32,
        R as u32,
        R8 as u32,
        LD as u32,
        0,
        K,
    )
    .unwrap();
    let reference = Reference {
        plan: &plan,
        latent: &x.latent,
        packed: &x.packed,
        int8: &odd,
        ids: &x.ids,
        row: &x.row,
        col: &x.col,
    };
    let want = reference.forward();
    assert_eq!(
        want.iter().filter(|&&v| v == 0x7E00).count(),
        2,
        "the routed experts 5 and 3 carry one odd byte each"
    );
    let factor_odd = alloc(dev, &vec![0u8; p * M * LD * 2]);
    let rc = call(
        dev,
        [b_odd, b_ids, b_row, b_col, factor_odd, spare, spare2],
        ExpertFactorOp::PackedForward,
        flags,
        p,
        0,
    );
    assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
    checked_bits(
        &u16s(&download(dev, factor_odd, p * M * LD * 2)),
        &want,
        "odd int8 bytes",
    );
    let g_row_odd = alloc(dev, &vec![0u8; p * M * 4]);
    let rc = call(
        dev,
        [b_odd, b_ids, b_row, b_col, b_grad, g_row_odd, spare2],
        ExpertFactorOp::PackedBackwardRow,
        flags,
        p,
        0,
    );
    assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
    let g = f32s(&download(dev, g_row_odd, p * M * 4));
    // routed slot 0 is expert 5 (row 2 holds the 0), slot 2 is expert 3 (row 9 holds the 2)
    let nan_rows: Vec<usize> = (0..p * M).filter(|&i| g[i].is_nan()).collect();
    assert_eq!(
        nan_rows,
        vec![2, 2 * M + 9],
        "exactly the odd bytes' rows are NaN"
    );
    for buf in [
        b_lat, b_pack, b_i8, b_odd, b_ids, b_row, b_col, spare, spare2, b_grad, b_bad, factor,
        factor_odd, g_row_odd,
    ] {
        assert_eq!(ffi::ack_buffer_free(dev, buf), ffi::ACK_OK);
    }
    assert_eq!(ffi::ack_close(dev), ffi::ACK_OK, "{}", last_error());
}

#[test]
fn what_the_entry_does_not_address_is_refused_by_name() {
    let Some(dev) = open_or_unmeasured("expert factor refusals") else {
        return;
    };
    let x = inputs();
    let p = x.ids.len();
    let ids: Vec<u8> = x.ids.iter().flat_map(|v| v.to_le_bytes()).collect();
    let (lat, idb) = (alloc(dev, &f32_bytes(&x.latent)), alloc(dev, &ids));
    let (row, col) = (
        alloc(dev, &f32_bytes(&x.row)),
        alloc(dev, &f32_bytes(&x.col)),
    );
    let factor = alloc(dev, &vec![0u8; p * M * LD * 2]);
    let (a, b, small) = (
        alloc(dev, &[0u8; 16]),
        alloc(dev, &[0u8; 16]),
        alloc(dev, &[0u8; 4]),
    );
    let f = ExpertFactorOp::LatentForward;
    let refused = |rc: i32, code: i32, name: &str| {
        assert_eq!(rc, code, "{name}");
        assert!(
            last_error().ends_with(name),
            "{:?} names {name}",
            last_error()
        );
    };
    refused(
        call(dev, [lat, idb, row, col, factor, a, b], f, HAS_COL, p, 0),
        ffi::ACK_ERR_SHAPE,
        "expert-factor-scale-missing",
    );
    refused(
        call(dev, [lat, idb, row, col, factor, a, b], f, HAS_ROW, p, 49),
        ffi::ACK_ERR_SHAPE,
        "expert-factor-layout-invalid",
    );
    refused(
        call(dev, [lat, idb, small, col, factor, a, b], f, HAS_ROW, p, 0),
        ffi::ACK_ERR_SIZE,
        "expert-factor-row-too-small",
    );
    refused(
        call(dev, [lat, idb, row, col, small, a, b], f, HAS_ROW, p, 0),
        ffi::ACK_ERR_SIZE,
        "expert-factor-factor-too-small",
    );
    refused(
        call(dev, [lat, idb, row, col, lat, a, b], f, HAS_ROW, p, 0),
        ffi::ACK_ERR_SHAPE,
        "expert-factor-output-aliases-input",
    );
    refused(
        call(
            dev,
            [lat, idb, row, col, factor, a, b],
            f,
            HAS_ROW | INT8_SIGNS,
            p,
            0,
        ),
        ffi::ACK_ERR_SHAPE,
        "expert-factor-int8-signs-with-latent",
    );
    // an int8 cache holds a byte a sign: a packed-sized one is too small for it
    let packed_sized = alloc(dev, &vec![0u8; E * M * R.div_ceil(8)]);
    refused(
        call(
            dev,
            [packed_sized, idb, row, col, factor, a, b],
            ExpertFactorOp::PackedForward,
            HAS_ROW | INT8_SIGNS,
            p,
            0,
        ),
        ffi::ACK_ERR_SIZE,
        "expert-factor-source-too-small",
    );
    assert_eq!(ffi::ack_buffer_free(dev, packed_sized), ffi::ACK_OK);
    let rc = call(
        dev,
        [lat, idb, row, col, factor, a, a],
        ExpertFactorOp::LatentBackward,
        HAS_ROW,
        p,
        0,
    );
    assert_ne!(rc, ffi::ACK_OK, "two outputs on one buffer must be refused");
    // the refusals recorded nothing: a plain call still runs
    assert_eq!(
        call(
            dev,
            [lat, idb, row, col, factor, a, b],
            f,
            HAS_ROW | HAS_COL,
            p,
            0
        ),
        ffi::ACK_OK,
        "{}",
        last_error()
    );
    for buf in [lat, idb, row, col, factor, a, b, small] {
        assert_eq!(ffi::ack_buffer_free(dev, buf), ffi::ACK_OK);
    }
    assert_eq!(ffi::ack_close(dev), ffi::ACK_OK, "{}", last_error());
}
