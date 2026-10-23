//! The grouped WORKGROUP ORDER and the f32-output edge modules' fast loops
//! (2026-10-01; the shader's WORKGROUP ORDER, TWO LOOPS, NARROW and MASKED
//! notes, and `rast_index` in src/ffi.rs), on a real device. Each claim is
//! against a module this change did not touch:
//!
//! 1. THE SAME BITS. A wide aligned product (66 tile columns: the 64-lane sets
//!    take their RAST modules from 64) equals, bit for bit, the same product
//!    computed as three column strips of 22 tile columns, which keep the row
//!    order -- for every entry with RAST modules (the bf16 and fp16 f32-output
//!    products and their 16-bit-output twins) and every layout, on 5 tile rows,
//!    so the last band of four is partial.
//! 2. IT RUNS. With device timestamps on, each entry's wide product (2048 x
//!    32768 x 1024, in every layout) is timed on the default device and on one
//!    opened with `ACK_MATMUL_ORDER=rows`: the two alternate, seven products
//!    each, and each keeps its least. Every f32-output layout must be at least
//!    1.15x faster (the bf16 modules measured 1.68-1.94x there). A 16-bit-output
//!    entry must be no slower in any layout (0.95x) and at least 1.15x faster in
//!    its best (bf16 output measured 1.08x nn and 1.29x nt). A value test alone
//!    passes whether the RAST modules ran or not; this one fails if they do not.
//!    The two C are compared too, bit for bit, on a grid of whole bands.
//! 3. THE EDGE LOOPS. An off-tile f32-output product (the edge modules, which
//!    now take COMBINED, NARROW and MASKED) equals, bit for bit, the aligned
//!    product of the same operands zero-padded to the tile, cropped: the edge
//!    gather stages exact zeros outside the extent, which is what the padding
//!    holds, and every loop stages the same K tiles in the same order.
//!
//! The operands are random words (random signs and mantissas, magnitudes from
//! 1/8 to 2), so the f32 sums round: the same bits means the same values
//! staged AND the same order of sums, which is what each claim says.
//!
//! Needs a Vulkan device; without one every test prints UNMEASURED and returns
//! (`ACK_REQUIRE_DEVICE=1` turns that into a failure). An entry the device
//! refuses (ACK_ERR_UNSUPPORTED_OP: no cooperative matrices of its type) is
//! UNMEASURED for that entry alone. The RAST modules exist in the 64-lane sets
//! only (a device whose subgroup size is 64, without ACK_COOPMAT_LANES=32) and
//! run by default only (without ACK_MATMUL_ORDER): elsewhere claim 2 prints
//! UNMEASURED, and claim 1 says so and still compares the strips.
use alelyon_compute_kit::ffi;
use std::ffi::{c_char, CStr};
use std::io::Write;
use std::sync::Mutex;

/// Every test here opens devices, and claim 2 opens one with ACK_MATMUL_ORDER
/// set: the environment is process-wide, so the tests take turns.
static ENV: Mutex<()> = Mutex::new(());

fn last_error() -> String {
    let mut buf = vec![0 as c_char; 1024];
    let rc = unsafe { ffi::ack_last_error(buf.as_mut_ptr(), buf.len()) };
    assert!(rc >= 0, "ack_last_error returned {rc}");
    unsafe { CStr::from_ptr(buf.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

/// Open a device with `vars` set for the open alone (the sets read them then),
/// restoring what the environment held.
fn open_with(what: &str, vars: &[(&str, &str)]) -> Option<*mut ffi::AckDevice> {
    let held: Vec<_> = vars
        .iter()
        .map(|(k, _)| (*k, std::env::var_os(k)))
        .collect();
    for (k, v) in vars {
        std::env::set_var(k, v);
    }
    let dev = ffi::ack_open();
    for (k, was) in held {
        match was {
            Some(v) => std::env::set_var(k, v),
            None => std::env::remove_var(k),
        }
    }
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

/// Why a wide aligned product on `dev` would not take its RAST modules, if it
/// would not: the rule `rast_index` applies, from outside.
fn no_rast_modules(dev: *mut ffi::AckDevice) -> Option<String> {
    let mut info: ffi::AckDeviceInfo = unsafe { std::mem::zeroed() };
    let size = std::mem::size_of::<ffi::AckDeviceInfo>();
    let rc = unsafe { ffi::ack_device_info(dev, &mut info, size) };
    assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
    if std::env::var_os("ACK_MATMUL_ORDER").is_some() {
        Some("ACK_MATMUL_ORDER is set in the environment".into())
    } else if std::env::var_os("ACK_COOPMAT_LANES").is_some_and(|v| v == "32") {
        Some("ACK_COOPMAT_LANES=32 opens the 32-lane sets, which have no RAST modules".into())
    } else if info.subgroup_size != 64 {
        Some(format!(
            "subgroup size {}: the 32-lane sets have no RAST modules",
            info.subgroup_size
        ))
    } else {
        None
    }
}

fn alloc(dev: *mut ffi::AckDevice, bytes: usize) -> *mut alelyon_compute_kit::Buffer {
    let b = ffi::ack_buffer_alloc(dev, bytes as u64);
    assert!(!b.is_null(), "{}", last_error());
    b
}

fn upload(dev: *mut ffi::AckDevice, buf: *const alelyon_compute_kit::Buffer, words: &[u16]) {
    let rc = unsafe {
        ffi::ack_upload(
            dev,
            buf,
            words.as_ptr().cast::<u8>(),
            std::mem::size_of_val(words),
        )
    };
    assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
}

fn download(
    dev: *mut ffi::AckDevice,
    buf: *const alelyon_compute_kit::Buffer,
    bytes: usize,
) -> Vec<u8> {
    let mut out = vec![0u8; bytes];
    let rc = unsafe { ffi::ack_download(dev, buf, out.as_mut_ptr(), bytes) };
    assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
    out
}

fn free(dev: *mut ffi::AckDevice, bufs: &[*mut alelyon_compute_kit::Buffer]) {
    for b in bufs {
        assert_eq!(
            ffi::ack_buffer_free(dev, *b),
            ffi::ACK_OK,
            "{}",
            last_error()
        );
    }
}

/// A 64-bit hash of (i, salt), for operand words.
fn hash(i: usize, salt: u64) -> u64 {
    let h = (i as u64 ^ salt.rotate_left(32)).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    h ^ (h >> 29)
}

#[derive(Clone, Copy, Debug)]
enum Entry {
    Bf16F32,
    F16F32,
    Bf16Bf16,
    F16F16,
}

impl Entry {
    const ALL: [Entry; 4] = [
        Entry::Bf16F32,
        Entry::F16F32,
        Entry::Bf16Bf16,
        Entry::F16F16,
    ];

    fn out_bytes(self) -> usize {
        match self {
            Entry::Bf16F32 | Entry::F16F32 => 4,
            Entry::Bf16Bf16 | Entry::F16F16 => 2,
        }
    }

    /// An operand word of this entry's type: a random sign and mantissa, with
    /// a magnitude in [1/8, 2). The products' sums round in f32, so C's bits
    /// depend on the order of the sums as well as on the values staged.
    fn word(self, h: u64) -> u16 {
        let sign = ((h >> 63) as u16) << 15;
        let octave = ((h >> 40) % 4) as u16;
        match self {
            Entry::Bf16F32 | Entry::Bf16Bf16 => sign | (124 + octave) << 7 | (h & 0x7f) as u16,
            Entry::F16F32 | Entry::F16F16 => sign | (12 + octave) << 10 | (h & 0x3ff) as u16,
        }
    }

    /// ACK_OK, or the entry's own refusal (an fp16 entry on a device without
    /// fp16 cooperative matrices refuses with ACK_ERR_UNSUPPORTED_OP).
    #[allow(clippy::too_many_arguments)]
    fn run(
        self,
        dev: *mut ffi::AckDevice,
        a: *const alelyon_compute_kit::Buffer,
        b: *const alelyon_compute_kit::Buffer,
        c: *const alelyon_compute_kit::Buffer,
        (m, n, k): (usize, usize, usize),
        a_t: i32,
        b_t: i32,
    ) -> i32 {
        let (m, n, k) = (m as u32, n as u32, k as u32);
        let ms = std::ptr::null_mut();
        unsafe {
            match self {
                Entry::Bf16F32 => ffi::ack_matmul_bf16(dev, a, b, c, m, n, k, a_t, b_t, ms),
                Entry::F16F32 => ffi::ack_matmul_f16(dev, a, b, c, m, n, k, a_t, b_t, ms),
                Entry::Bf16Bf16 => {
                    ffi::ack_matmul_bf16_to_bf16(dev, a, b, c, m, n, k, a_t, b_t, ms)
                }
                Entry::F16F16 => ffi::ack_matmul_f16_to_f16(dev, a, b, c, m, n, k, a_t, b_t, ms),
            }
        }
    }
}

/// A (logical M x K) and B (logical K x N) in their stored orders, as words of
/// `entry`'s operand type; A is drawn from salt 1, B from salt 2.
fn operands(
    entry: Entry,
    (m, n, k): (usize, usize, usize),
    a_t: bool,
    b_t: bool,
) -> (Vec<u16>, Vec<u16>) {
    let mut a = vec![0u16; m * k];
    let mut b = vec![0u16; k * n];
    for i in 0..m {
        for l in 0..k {
            let at = if a_t { l * m + i } else { i * k + l };
            a[at] = entry.word(hash(i * k + l, 1));
        }
    }
    for l in 0..k {
        for j in 0..n {
            let at = if b_t { j * k + l } else { l * n + j };
            b[at] = entry.word(hash(l * n + j, 2));
        }
    }
    (a, b)
}

/// B's columns [from, from + width) in the stored order, as a dense operand.
fn columns(b: &[u16], (k, n): (usize, usize), b_t: bool, from: usize, width: usize) -> Vec<u16> {
    let mut out = vec![0u16; k * width];
    for l in 0..k {
        for j in 0..width {
            let (src, dst) = if b_t {
                ((from + j) * k + l, j * k + l)
            } else {
                (l * n + from + j, l * width + j)
            };
            out[dst] = b[src];
        }
    }
    out
}

const LAYOUTS: [(i32, i32); 4] = [(0, 0), (0, 1), (1, 0), (1, 1)];

#[test]
fn a_wide_product_in_the_grouped_order_is_its_row_order_strips_bit_for_bit() {
    let _env = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let Some(dev) = open_with("grouped order, same bits", &[]) else {
        return;
    };
    if let Some(why) = no_rast_modules(dev) {
        eprintln!("UNMEASURED here (the grouped order): {why}; the strips are still compared");
    }
    // 5 tile rows (the last band of four is partial) x 66 tile columns, K two tiles
    let (m, n, k) = (5 * 128, 66 * 128, 64);
    let strip = 22 * 128;
    let mut checked = 0usize;
    for entry in Entry::ALL {
        for (a_t, b_t) in LAYOUTS {
            let (a, b) = operands(entry, (m, n, k), a_t != 0, b_t != 0);
            let width = entry.out_bytes();
            let a_buf = alloc(dev, a.len() * 2);
            let b_buf = alloc(dev, b.len() * 2);
            let c_buf = alloc(dev, m * n * width);
            upload(dev, a_buf, &a);
            upload(dev, b_buf, &b);
            let rc = entry.run(dev, a_buf, b_buf, c_buf, (m, n, k), a_t, b_t);
            if rc == ffi::ACK_ERR_UNSUPPORTED_OP {
                // a device feature, so the first layout's refusal or none
                assert_eq!((a_t, b_t), (0, 0), "{entry:?}: {}", last_error());
                eprintln!("UNMEASURED here ({entry:?}): {}", last_error());
                free(dev, &[a_buf, b_buf, c_buf]);
                break;
            }
            assert_eq!(
                rc,
                ffi::ACK_OK,
                "{entry:?} {a_t}{b_t} wide: {}",
                last_error()
            );
            let wide = download(dev, c_buf, m * n * width);
            for s in 0..n / strip {
                let bs = columns(&b, (k, n), b_t != 0, s * strip, strip);
                let bs_buf = alloc(dev, bs.len() * 2);
                let cs_buf = alloc(dev, m * strip * width);
                upload(dev, bs_buf, &bs);
                let rc = entry.run(dev, a_buf, bs_buf, cs_buf, (m, strip, k), a_t, b_t);
                assert_eq!(
                    rc,
                    ffi::ACK_OK,
                    "{entry:?} {a_t}{b_t} strip {s}: {}",
                    last_error()
                );
                let part = download(dev, cs_buf, m * strip * width);
                let mut differing = 0usize;
                for i in 0..m {
                    let row = &part[i * strip * width..(i + 1) * strip * width];
                    let at = (i * n + s * strip) * width;
                    differing += row
                        .chunks(width)
                        .zip(wide[at..at + strip * width].chunks(width))
                        .filter(|(x, y)| x != y)
                        .count();
                }
                assert_eq!(
                    differing, 0,
                    "{entry:?} a_t={a_t} b_t={b_t}: {differing} words of strip {s} differ from the wide product"
                );
                free(dev, &[bs_buf, cs_buf]);
            }
            free(dev, &[a_buf, b_buf, c_buf]);
            checked += 1;
        }
    }
    if checked == 0 {
        eprintln!("UNMEASURED here (grouped order, same bits): the device refused every entry");
    }
    assert_eq!(ffi::ack_close(dev), ffi::ACK_OK);
}

/// One arm of claim 2: a device opened with the timestamp instrument on (and
/// `vars`).
struct Arm {
    dev: *mut ffi::AckDevice,
    /// 16 bytes no product touches. `ack_download` reads whole buffers only,
    /// and any readback flushes and waits for the whole open batch first
    /// (`Context::download_into`), so reading this drains a product cheaply.
    drain: *mut alelyon_compute_kit::Buffer,
}

/// The wide product claim 2 times, nt: 16 tile rows (four whole bands) x 256
/// tile columns.
const TIMED: (usize, usize, usize) = (2048, 32768, 1024);

impl Arm {
    /// None (UNMEASURED, said here) without a device or without RAST modules.
    fn open(vars: &[(&str, &str)]) -> Option<Self> {
        let mut all: Vec<(&str, &str)> = vec![("ACK_DISPATCH_TIMESTAMPS", "1")];
        all.extend_from_slice(vars);
        let dev = open_with("grouped order, device time", &all)?;
        if let Some(why) = no_rast_modules(dev) {
            eprintln!("UNMEASURED here (the grouped order's device time): {why}");
            assert_eq!(ffi::ack_close(dev), ffi::ACK_OK);
            return None;
        }
        let drain = alloc(dev, 16);
        Some(Arm { dev, drain })
    }

    /// A, B and C for `entry`'s timed product, A and B uploaded.
    fn buffers(
        &self,
        entry: Entry,
        (a, b): (&[u16], &[u16]),
    ) -> [*mut alelyon_compute_kit::Buffer; 3] {
        let (m, n, _) = TIMED;
        let bufs = [
            alloc(self.dev, a.len() * 2),
            alloc(self.dev, b.len() * 2),
            alloc(self.dev, m * n * entry.out_bytes()),
        ];
        upload(self.dev, bufs[0], a);
        upload(self.dev, bufs[1], b);
        bufs
    }

    /// One product's device nanoseconds; None if the device refuses `entry`.
    fn product_ns(
        &self,
        entry: Entry,
        [a, b, c]: [*mut alelyon_compute_kit::Buffer; 3],
        (a_t, b_t): (i32, i32),
    ) -> Option<u64> {
        let stats = std::mem::size_of::<ffi::AckDispatchStats>();
        let mut before: ffi::AckDispatchStats = unsafe { std::mem::zeroed() };
        let rc = unsafe { ffi::ack_dispatch_stats(self.dev, &mut before, stats) };
        assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
        let rc = entry.run(self.dev, a, b, c, TIMED, a_t, b_t);
        if rc == ffi::ACK_ERR_UNSUPPORTED_OP {
            return None;
        }
        assert_eq!(rc, ffi::ACK_OK, "{entry:?}: {}", last_error());
        // a readback waits for the batch and folds its timestamps in
        download(self.dev, self.drain, 16);
        let mut after: ffi::AckDispatchStats = unsafe { std::mem::zeroed() };
        let rc = unsafe { ffi::ack_dispatch_stats(self.dev, &mut after, stats) };
        assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
        assert_eq!(
            after.timestamps_enabled, 1,
            "the timestamp instrument is off"
        );
        assert_eq!(
            after.items - before.items,
            1,
            "{entry:?}: one product, one item"
        );
        assert_eq!(
            after.timed_items - before.timed_items,
            1,
            "{entry:?} was not timed"
        );
        assert_eq!(
            after.unresolved_flushes, before.unresolved_flushes,
            "a flush's span was not folded in, so device_ns covers less than the product"
        );
        Some(after.device_ns - before.device_ns)
    }

    fn close(self) {
        free(self.dev, &[self.drain]);
        assert_eq!(ffi::ack_close(self.dev), ffi::ACK_OK);
    }
}

#[test]
fn the_grouped_order_is_what_runs_a_wide_product_and_it_is_faster() {
    let _env = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let Some(row_order) = Arm::open(&[("ACK_MATMUL_ORDER", "rows")]) else {
        return;
    };
    let grouped_order = Arm::open(&[]).expect("the same device, a moment later");
    let (m, n, _) = TIMED;
    let mut measured = 0usize;
    for entry in Entry::ALL {
        // one draw of operand words serves every layout: each stored order
        // holds the same number of words, and both arms read the same ones
        let (a, b) = operands(entry, TIMED, false, true);
        let (row_bufs, grouped_bufs) = (
            row_order.buffers(entry, (&a, &b)),
            grouped_order.buffers(entry, (&a, &b)),
        );
        let mut best = 0f64;
        for (a_t, b_t) in LAYOUTS {
            let layout = format!("{}{}", ["n", "t"][a_t as usize], ["n", "t"][b_t as usize]);
            // the arms alternate, so a clock still ramping up, or another
            // process's work on the card, falls on both; each keeps its least
            let (mut rows, mut grouped) = (u64::MAX, u64::MAX);
            for round in 0..7 {
                let Some(r) = row_order.product_ns(entry, row_bufs, (a_t, b_t)) else {
                    // a device feature, so the first product's refusal or none
                    assert_eq!((round, a_t, b_t), (0, 0, 0), "{entry:?}: {}", last_error());
                    break;
                };
                rows = rows.min(r);
                let g = grouped_order.product_ns(entry, grouped_bufs, (a_t, b_t));
                grouped = grouped.min(g.expect("the same device refused what it ran"));
            }
            if rows == u64::MAX {
                eprintln!("UNMEASURED here ({entry:?}, device time): {}", last_error());
                break;
            }
            let bytes = m * n * entry.out_bytes();
            let (c_rows, c_grouped) = (
                download(row_order.dev, row_bufs[2], bytes),
                download(grouped_order.dev, grouped_bufs[2], bytes),
            );
            if c_rows != c_grouped {
                let differing = c_rows
                    .chunks(entry.out_bytes())
                    .zip(c_grouped.chunks(entry.out_bytes()))
                    .filter(|(x, y)| x != y)
                    .count();
                panic!("{entry:?} {layout}: {differing} words of C differ between the two orders");
            }
            let ratio = rows as f64 / grouped as f64;
            // straight to the process's stderr, past libtest's capture: a
            // verify that runs this without --nocapture keeps only a stderr
            // tail, and these lines are what a failure there must show
            let _ = writeln!(
                std::io::stderr(),
                "matmul_order_abi {entry:?} 2048 x 32768 x 1024 {layout}: row order {rows} ns, grouped order {grouped} ns, {ratio:.2}x"
            );
            if entry.out_bytes() == 4 {
                assert!(
                    ratio >= 1.15,
                    "{entry:?} {layout}: the default device's wide product ({grouped} ns) is not clearly faster \
                     than the row order's ({rows} ns): {ratio:.2}x, where the bf16 f32-output modules measured \
                     1.68-1.94x in every layout -- the RAST modules may not be running"
                );
            } else {
                assert!(
                    ratio >= 0.95,
                    "{entry:?} {layout}: the grouped order made a wide 16-bit product slower: {grouped} ns \
                     against the row order's {rows} ns ({ratio:.2}x)"
                );
            }
            best = best.max(ratio);
            measured += 1;
        }
        // a set is built with all four of its RAST modules or none, and claim
        // 1 holds each layout's module to its own bits, so one clearly faster
        // layout shows the set's RAST modules run (bf16 output measured 1.08x
        // nn and 1.29x nt at this shape: its C is half the f32 traffic)
        if entry.out_bytes() == 2 && best > 0.0 {
            assert!(
                best >= 1.15,
                "{entry:?}: no layout of the wide 16-bit product is clearly faster in the grouped order (best \
                 {best:.2}x, where bf16 output measured 1.29x nt) -- its RAST modules may not be running"
            );
        }
        free(row_order.dev, &row_bufs);
        free(grouped_order.dev, &grouped_bufs);
    }
    if measured == 0 {
        eprintln!(
            "UNMEASURED here (the grouped order's device time): the device refused every entry"
        );
    }
    row_order.close();
    grouped_order.close();
}

/// The aligned product of `a` and `b` zero-padded to the tile, cropped to M x N.
fn zero_padded_product(
    dev: *mut ffi::AckDevice,
    entry: Entry,
    (a, b): (&[u16], &[u16]),
    (m, n, k): (usize, usize, usize),
    (a_t, b_t): (i32, i32),
) -> Vec<u8> {
    let up = |x: usize, t: usize| x.div_ceil(t) * t;
    let (mp, np, kp) = (up(m, 128), up(n, 128), up(k, 32));
    let mut ap = vec![0u16; mp * kp];
    let mut bp = vec![0u16; kp * np];
    for i in 0..m {
        for l in 0..k {
            if a_t != 0 {
                ap[l * mp + i] = a[l * m + i];
            } else {
                ap[i * kp + l] = a[i * k + l];
            }
        }
    }
    for l in 0..k {
        for j in 0..n {
            if b_t != 0 {
                bp[j * kp + l] = b[j * k + l];
            } else {
                bp[l * np + j] = b[l * n + j];
            }
        }
    }
    let width = entry.out_bytes();
    let (ab, bb, cb) = (
        alloc(dev, ap.len() * 2),
        alloc(dev, bp.len() * 2),
        alloc(dev, mp * np * width),
    );
    upload(dev, ab, &ap);
    upload(dev, bb, &bp);
    let rc = entry.run(dev, ab, bb, cb, (mp, np, kp), a_t, b_t);
    assert_eq!(rc, ffi::ACK_OK, "padded {entry:?}: {}", last_error());
    let full = download(dev, cb, mp * np * width);
    free(dev, &[ab, bb, cb]);
    let mut out = Vec::with_capacity(m * n * width);
    for i in 0..m {
        out.extend_from_slice(&full[i * np * width..(i * np + n) * width]);
    }
    out
}

#[test]
fn an_edge_product_is_the_zero_padded_aligned_product_bit_for_bit() {
    let _env = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let Some(dev) = open_with("edge loops, same bits", &[]) else {
        return;
    };
    // every loop across the layouts: tiles wholly inside on pitches of 8 (the
    // uvec4 loop) and of 4 mod 8 (NARROW's uvec2: 300, and the 1628-wide FFN
    // extent), ragged tiles (MASKED), odd pitches (517, 257: the gather loop),
    // and a K tail on each shape
    let shapes = [
        (300, 1628, 200),
        (4 * 128 + 5, 2 * 128, 96 + 16),
        (257, 640, 3256),
    ];
    let mut checked = 0usize;
    for entry in [Entry::Bf16F32, Entry::F16F32] {
        'layouts: for (a_t, b_t) in LAYOUTS {
            for (m, n, k) in shapes {
                let (a, b) = operands(entry, (m, n, k), a_t != 0, b_t != 0);
                let width = entry.out_bytes();
                let (ab, bb, cb) = (
                    alloc(dev, a.len() * 2),
                    alloc(dev, b.len() * 2),
                    alloc(dev, m * n * width),
                );
                upload(dev, ab, &a);
                upload(dev, bb, &b);
                let rc = entry.run(dev, ab, bb, cb, (m, n, k), a_t, b_t);
                if rc == ffi::ACK_ERR_UNSUPPORTED_OP {
                    // a device feature, so the first product's refusal or none
                    assert_eq!(checked % 12, 0, "{entry:?}: {}", last_error());
                    eprintln!("UNMEASURED here ({entry:?}): {}", last_error());
                    free(dev, &[ab, bb, cb]);
                    break 'layouts;
                }
                assert_eq!(rc, ffi::ACK_OK, "{entry:?} {m}x{n}x{k}: {}", last_error());
                let edge = download(dev, cb, m * n * width);
                free(dev, &[ab, bb, cb]);
                let padded = zero_padded_product(dev, entry, (&a, &b), (m, n, k), (a_t, b_t));
                let differing = edge
                    .chunks(width)
                    .zip(padded.chunks(width))
                    .filter(|(x, y)| x != y)
                    .count();
                assert_eq!(
                    differing, 0,
                    "{entry:?} {m}x{n}x{k} a_t={a_t} b_t={b_t}: {differing} words differ from the padded aligned product"
                );
                checked += 1;
            }
        }
    }
    if checked == 0 {
        eprintln!("UNMEASURED here (edge loops, same bits): the device refused both entries");
    }
    assert_eq!(ffi::ack_close(dev), ffi::ACK_OK);
}
