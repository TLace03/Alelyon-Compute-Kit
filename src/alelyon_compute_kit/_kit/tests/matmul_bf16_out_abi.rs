//! `ack_matmul_bf16_to_bf16` (2026-09-27): the coopmat product writing C as
//! bf16 in its own epilogue, on a real device.
//!
//! THE LAW IS BIT IDENTITY. The entry exists to drop the f32 block and the
//! cast dispatch that every bf16 product paid, and nothing else: for every
//! layout and every shape its bf16 C must equal, bit for bit, what
//! `ack_matmul_bf16` writes in f32 rounded to nearest even as the cast kernel
//! rounds. Both products run the same staging and the same accumulation order
//! (the aligned modules for a shape on the 128 x 128 x 32 tile, since
//! 2026-09-28, and the edge modules otherwise; each entry picks by the same
//! rule), so any difference is the epilogue's. The aligned shapes below are
//! what reach the aligned bf16-output modules. Since TWO LOOPS (2026-09-28) the
//! 64-lane bf16-output edge modules run, for a tile wholly inside the extent
//! with pitches that are multiples of 8, the aligned module's loop over the
//! whole K tiles and one gathered step for a K tail, and the gather loop for
//! every other tile. `ack_matmul_bf16`'s f32 edge modules keep the gather loop
//! over every tile, so this law compares the two loops against an unchanged
//! path: it holds because each tile stages the same values for the same K tiles
//! in the same order. The shapes below reach the two-loop path in one or all
//! four layouts: off the tile in M and N, the A3r-X2 pass's 3256-wide M and N
//! at a shorter K, and K tails (72, and the pass's 3256). Since NARROW (the
//! same day) an interior tile whose pitches are multiples of 4 but not all of 8
//! takes a third loop of uvec2 reads: 264 x 132 (B's pitch 132 unless B is stored
//! transposed) and 512 x 256 x 1628 (the pass's 1628-wide K as a pitch, with a
//! K tail of 28) reach it, and 264 x 130 (a pitch of 2 mod 4) keeps its interior
//! tiles on the gather loop. Since MASKED (2026-09-29) a ragged tile whose reads
//! each lie wholly inside or wholly outside the extent takes a masked form of
//! those loops. Most of the shapes above have such tiles. 260 x 136 (M of 4 mod 8)
//! is the shape where the upper uvec2 half of a transposed A's last read lies
//! outside. The output is poisoned first, so an element
//! the epilogue never writes cannot pass. Refusals are checked by name: an odd
//! n (a bf16 pair would straddle two rows), a zero extent, short buffers, and
//! an output that aliases an operand.
//!
//! Skips as UNMEASURED without a device unless ACK_REQUIRE_DEVICE is set.
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

fn bf16_bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Operands of an m x n x k product, stored as the layout flags say, drawn
/// with a spread of magnitudes (and ties at the bf16 rounding point are
/// likely in the f32 sums, which is what an RNE epilogue must get right).
fn operands(m: usize, n: usize, k: usize, salt: u32) -> (Vec<u16>, Vec<u16>) {
    let draw = |i: usize| {
        let h = (i as u32).wrapping_mul(2_654_435_761) ^ salt;
        let mag = [1.0f32, 0.25, 4.0, 1.0 / 64.0][(h >> 29) as usize % 4];
        ((h % 2_001) as f32 / 1_000.0 - 1.0) * mag
    };
    let a = (0..m * k).map(|i| to_bf16(draw(i))).collect();
    let b = (0..k * n).map(|i| to_bf16(draw(i + 1_000_003))).collect();
    (a, b)
}

#[test]
fn the_bf16_output_is_bit_identical_to_the_f32_product_rounded_as_the_cast_rounds() {
    let Some(dev) = open_or_unmeasured("matmul bf16 -> bf16") else {
        return;
    };
    // aligned; off the tile in every extent; the pass's 4096 x 512 x 512;
    // a tall, narrow output; and one fragment column (n = 2)
    let shapes = [
        (256usize, 256usize, 128usize),
        (130, 98, 77),
        (4096, 512, 512),
        (200, 2, 300),
        (48, 1000, 64),
        // two loops: off the tile in both M and N; the A3r-X2 pass's 3256-wide M
        // and N at a shorter K; K tails of 8 and of 24 (the pass's 3256) behind
        // whole K tiles; a pitch of 4 mod 8 (the narrow loop unless the
        // transposed layout makes the pitch K); the pass's 1628 as a pitch with a
        // K tail of 28; and a pitch of 2 mod 4 (the gather loop)
        (264, 136, 64),
        (3256, 512, 256),
        (512, 3256, 128),
        (264, 136, 72),
        (512, 256, 3256),
        (264, 132, 64),
        (512, 256, 1628),
        (264, 130, 64),
        // masked: a ragged row with M = 4 mod 8 (the upper uvec2 half of a
        // transposed A's last read lies outside M)
        (260, 136, 64),
    ];
    let mut checked = 0usize;
    for (si, &(m, n, k)) in shapes.iter().enumerate() {
        for (a_t, b_t) in [(0i32, 0i32), (0, 1), (1, 0), (1, 1)] {
            let (a16, b16) = operands(m, n, k, 97 + si as u32 * 13 + (a_t * 2 + b_t) as u32);
            let a = alloc(dev, a16.len() * 2);
            let b = alloc(dev, b16.len() * 2);
            let c32 = alloc(dev, m * n * 4);
            let c16 = alloc(dev, m * n * 2);
            upload(dev, a, &bf16_bytes(&a16));
            upload(dev, b, &bf16_bytes(&b16));
            upload(dev, c16, &vec![0xA5u8; m * n * 2]);
            let rc = unsafe {
                ffi::ack_matmul_bf16(
                    dev,
                    a,
                    b,
                    c32,
                    m as u32,
                    n as u32,
                    k as u32,
                    a_t,
                    b_t,
                    std::ptr::null_mut(),
                )
            };
            assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
            let rc = unsafe {
                ffi::ack_matmul_bf16_to_bf16(
                    dev,
                    a,
                    b,
                    c16,
                    m as u32,
                    n as u32,
                    k as u32,
                    a_t,
                    b_t,
                    std::ptr::null_mut(),
                )
            };
            assert_eq!(
                rc,
                ffi::ACK_OK,
                "{m}x{n}x{k} a_t {a_t} b_t {b_t}: {}",
                last_error()
            );
            let want: Vec<u16> = download(dev, c32, m * n * 4)
                .chunks_exact(4)
                .map(|c| to_bf16(f32::from_le_bytes([c[0], c[1], c[2], c[3]])))
                .collect();
            let got: Vec<u16> = download(dev, c16, m * n * 2)
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect();
            let bad = want.iter().zip(&got).filter(|(w, g)| w != g).count();
            assert_eq!(
                bad, 0,
                "{m}x{n}x{k} a_t {a_t} b_t {b_t}: {bad} of {} elements differ from the f32 product rounded to bf16 (first: {:?})",
                m * n,
                want.iter().zip(&got).position(|(w, g)| w != g)
            );
            checked += m * n;
            for buf in [a, b, c32, c16] {
                assert_eq!(ffi::ack_buffer_free(dev, buf), ffi::ACK_OK);
            }
        }
    }
    eprintln!(
        "{checked} elements bit-identical across {} shapes x 4 layouts",
        shapes.len()
    );
    assert_eq!(ffi::ack_close(dev), ffi::ACK_OK, "{}", last_error());
}

fn refused(what: &str, rc: i32, code: i32, name: &str) {
    assert_eq!(rc, code, "{what}: code");
    let e = last_error();
    assert!(e.contains(name), "{what}: {e:?} names {name:?}");
}

#[test]
fn what_the_entry_does_not_address_is_refused_by_name() {
    let Some(dev) = open_or_unmeasured("matmul bf16 -> bf16 refusals") else {
        return;
    };
    let a = alloc(dev, 64 * 64 * 2);
    let b = alloc(dev, 64 * 64 * 2);
    let c = alloc(dev, 64 * 64 * 2);
    let small = alloc(dev, 16);
    let call = |a, b, c, m: u32, n: u32, k: u32| unsafe {
        ffi::ack_matmul_bf16_to_bf16(dev, a, b, c, m, n, k, 0, 0, std::ptr::null_mut())
    };
    refused(
        "odd n",
        call(a, b, c, 64, 63, 64),
        ffi::ACK_ERR_SHAPE,
        "matmul-bf16-out-odd-n",
    );
    refused(
        "zero m",
        call(a, b, c, 0, 64, 64),
        ffi::ACK_ERR_SHAPE,
        "zero dimension",
    );
    refused(
        "short C",
        call(a, b, small, 64, 64, 64),
        ffi::ACK_ERR_SIZE,
        "buffers too small",
    );
    refused(
        "C is A",
        call(a, b, a, 64, 64, 64),
        ffi::ACK_ERR_SHAPE,
        "matmul-bf16-out-output-aliases-input",
    );
    // the refusals recorded nothing: a plain product still runs and is right-sized
    assert_eq!(call(a, b, c, 64, 64, 64), ffi::ACK_OK, "{}", last_error());
    for buf in [a, b, c, small] {
        assert_eq!(ffi::ack_buffer_free(dev, buf), ffi::ACK_OK);
    }
    assert_eq!(ffi::ack_close(dev), ffi::ACK_OK, "{}", last_error());
}
