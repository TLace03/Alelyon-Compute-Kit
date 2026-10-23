//! `ack_matmul_f16_to_f16` (2026-09-30): the fp16 coopmat product writing C
//! as fp16 in its own epilogue, on a real device.
//!
//! THE LAW IS BIT IDENTITY FOR EVERY WORD THAT IS NOT A NaN. The entry exists to
//! drop the f32 block and the convert dispatch that every fp16 product paid (the
//! torch adapter ran `ack_matmul_f16`, then convert/v1 over its f32 C), and
//! nothing else. It rounds with the device's own conversion under DECLARED float
//! controls (to nearest even, subnormals kept), so for every layout and every
//! shape each of its words must equal, bit for bit, the word of both of these
//! wherever that word is not a NaN, and must be a NaN wherever it is:
//! - `ack_matmul_f16`'s f32 C converted by the kit's own convert/v1 (the
//!   adapter's two dispatches);
//! - the same f32 C rounded on the host by a port of convert/v1's f32 -> f16
//!   (to nearest even, overflow to infinity, subnormals exact, NaN -> sign |
//!   0x7E00), and the device's convert is held to that port on every word.
//!
//! A NaN's sign and payload are free: the device writes its own NaN, where
//! convert/v1 writes sign | 0x7E00 (a per-word NaN rule made the module spill;
//! the kernel's FP16 OUTPUT note). swiglu/v1 scopes its NaN words the same way.
//!
//! Both products run the same staging and the same accumulation order: the
//! aligned modules for a shape on the 128 x 128 x 32 tile and the edge modules
//! otherwise, each entry picking by the same rule. The 64-lane fp16-output edge
//! modules take the bf16-output set's TWO LOOPS, NARROW and MASKED paths
//! (the shader's notes), which `ack_matmul_f16`'s f32 edge modules do not, so
//! this law also compares those loops against the plain gather loop for fp16.
//! The shapes are the bf16-output test's, which reach every one of those paths,
//! and three of the student's own: its head at K = 1,002 (a pitch of 2 mod 4:
//! the gather loop) with B stored transposed as the head's weight is, an
//! FFN-sized K of 112 (a K tail of 16 behind three whole K tiles), and a K of
//! 32. Three more draw operands for fp16's own edges: a range whose sums pass
//! 65,504 (C rounds to infinity), one whose sums fall below 2^-14 (subnormal C),
//! and a NaN operand (its row of C is NaN). Each of those is REQUIRED to occur,
//! so the test cannot pass without exercising it. The output is poisoned first,
//! so an element the epilogue never writes cannot pass. Refusals are checked by
//! name: an odd n (an fp16 pair would straddle two rows), a zero extent, short
//! buffers, and an output that aliases an operand.
//!
//! The batched twin `ack_matmul_f16_z_to_f16` (the same day) is held to the same
//! law, every word that is not a NaN bit for bit and every NaN a NaN, against
//! `ack_matmul_f16_z` and the two conversions, batch by batch. Its
//! batches are strided with gaps and its pitches are wider than their widths, and
//! every gap and pad holds NaN, so a read outside a slab would show. Its layouts,
//! shapes and fp16 edges are the 2-D test's, with the student's attention bmm
//! (640 x 48 @ 48 x 2,048). The gaps of C must keep their poison. Its refusals
//! add a K-split (zmode 1) and an odd batch stride of C.
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

/// convert/v1's f32 -> f16 (kernels/convert.comp), ported: to nearest even,
/// overflow to infinity, subnormals exact, NaN -> sign | 0x7E00.
fn to_f16(x: f32) -> u16 {
    let w = x.to_bits();
    let sign = (w >> 16) & 0x8000;
    let e = (w >> 23) & 0xff;
    let m = w & 0x7f_ffff;
    if e == 0xff {
        return (sign | if m != 0 { 0x7e00 } else { 0x7c00 }) as u16;
    }
    let he = e as i32 - 112;
    if he >= 31 {
        return (sign | 0x7c00) as u16;
    }
    if he <= 0 {
        if he < -10 {
            return sign as u16;
        }
        let mf = m | 0x80_0000;
        let shift = (14 - he) as u32;
        let mut q = mf >> shift;
        let rem = mf & ((1u32 << shift) - 1);
        let half = 1u32 << (shift - 1);
        if rem > half || (rem == half && (q & 1) != 0) {
            q += 1;
        }
        return (sign | q) as u16;
    }
    let mut h = ((he as u32) << 10) | (m >> 13);
    let rem = m & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (h & 1) != 0) {
        h += 1;
    }
    (sign | h) as u16
}

/// fp16 NaN: exponent all ones, mantissa not zero.
fn is_nan16(h: u16) -> bool {
    h & 0x7c00 == 0x7c00 && h & 0x03ff != 0
}

/// The law's comparison: the same word, or a NaN where the reference has a NaN
/// (its sign and payload free).
fn same_word(want: u16, got: u16) -> bool {
    if is_nan16(want) {
        is_nan16(got)
    } else {
        want == got
    }
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

fn halves(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn words16(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

/// The kit's convert/v1 over `n` dense f32 elements into fp16 (storage codes
/// 0 -> 2), as the torch adapter converted an fp16 product's f32 C.
fn convert_f32_to_f16(
    dev: *mut ffi::AckDevice,
    src: *const alelyon_compute_kit::Buffer,
    out: *const alelyon_compute_kit::Buffer,
    n: usize,
) {
    let r = ffi::AckConvertRequest {
        src_code: 0,
        out_code: 2,
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
            dev,
            src,
            out,
            &r,
            std::mem::size_of::<ffi::AckConvertRequest>(),
            std::ptr::null_mut(),
        )
    };
    assert_eq!(rc, ffi::ACK_OK, "convert f32 -> f16: {}", last_error());
}

/// How a shape's operands are drawn.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Draw {
    /// the bf16-output test's spread of magnitudes (1, 1/4, 4, 1/64)
    Spread,
    /// magnitudes of 256: most sums pass fp16's 65,504, so C rounds to infinity
    Huge,
    /// magnitudes of 1/256: sums below 2^-14, so C is subnormal
    Tiny,
    /// the spread, with A's first stored element NaN
    Nan,
}

/// Operands of an m x n x k product, stored as the layout flags say (each
/// operand's elements in storage order), as fp16 words.
fn operands(m: usize, n: usize, k: usize, salt: u32, draw: Draw) -> (Vec<u16>, Vec<u16>) {
    let value = |i: usize| {
        let h = (i as u32).wrapping_mul(2_654_435_761) ^ salt;
        let mag = match draw {
            Draw::Spread | Draw::Nan => [1.0f32, 0.25, 4.0, 1.0 / 64.0][(h >> 29) as usize % 4],
            Draw::Huge => 256.0,
            Draw::Tiny => 1.0 / 256.0,
        };
        ((h % 2_001) as f32 / 1_000.0 - 1.0) * mag
    };
    let mut a: Vec<u16> = (0..m * k).map(|i| to_f16(value(i))).collect();
    let b = (0..k * n).map(|i| to_f16(value(i + 1_000_003))).collect();
    if draw == Draw::Nan {
        a[0] = 0x7e00;
    }
    (a, b)
}

#[test]
fn the_fp16_output_is_bit_identical_to_the_f32_product_converted_as_convert_v1_converts() {
    let Some(dev) = open_or_unmeasured("matmul f16 -> f16") else {
        return;
    };
    // (m, n, k, draw): the bf16-output test's shapes (aligned; off the tile in
    // every extent; two loops; narrow; masked), then the student's own and fp16's
    // edges (the module header)
    let shapes = [
        (256usize, 256usize, 128usize, Draw::Spread),
        (130, 98, 77, Draw::Spread),
        (4096, 512, 512, Draw::Spread),
        (200, 2, 300, Draw::Spread),
        (48, 1000, 64, Draw::Spread),
        (264, 136, 64, Draw::Spread),
        (3256, 512, 256, Draw::Spread),
        (512, 3256, 128, Draw::Spread),
        (264, 136, 72, Draw::Spread),
        (512, 256, 3256, Draw::Spread),
        (264, 132, 64, Draw::Spread),
        (512, 256, 1628, Draw::Spread),
        (264, 130, 64, Draw::Spread),
        (260, 136, 64, Draw::Spread),
        // the student's head (2,048 x 1,002 @ 1,002 x 151,936), 256 of its rows and
        // 2,048 of its columns; an FFN-sized K of 112; a K of 32
        (256, 2048, 1002, Draw::Spread),
        (512, 512, 112, Draw::Spread),
        (512, 2048, 32, Draw::Spread),
        // fp16's own edges
        (136, 264, 64, Draw::Huge),
        (136, 264, 64, Draw::Tiny),
        (130, 98, 77, Draw::Nan),
    ];
    let (mut checked, mut infinite, mut subnormal, mut nan) = (0usize, 0usize, 0usize, 0usize);
    for (si, &(m, n, k, draw)) in shapes.iter().enumerate() {
        for (a_t, b_t) in [(0i32, 0i32), (0, 1), (1, 0), (1, 1)] {
            let (a16, b16) = operands(m, n, k, 131 + si as u32 * 17 + (a_t * 2 + b_t) as u32, draw);
            let a = alloc(dev, a16.len() * 2);
            let b = alloc(dev, b16.len() * 2);
            let c32 = alloc(dev, m * n * 4);
            let c16 = alloc(dev, m * n * 2);
            let conv = alloc(dev, m * n * 2);
            upload(dev, a, &halves(&a16));
            upload(dev, b, &halves(&b16));
            upload(dev, c16, &vec![0xA5u8; m * n * 2]);
            upload(dev, conv, &vec![0x5Au8; m * n * 2]);
            let rc = unsafe {
                ffi::ack_matmul_f16(
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
            convert_f32_to_f16(dev, c32, conv, m * n);
            let rc = unsafe {
                ffi::ack_matmul_f16_to_f16(
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
            let host: Vec<u16> = download(dev, c32, m * n * 4)
                .chunks_exact(4)
                .map(|c| to_f16(f32::from_le_bytes([c[0], c[1], c[2], c[3]])))
                .collect();
            let device_convert = words16(&download(dev, conv, m * n * 2));
            let got = words16(&download(dev, c16, m * n * 2));
            let bad_convert = host
                .iter()
                .zip(&device_convert)
                .filter(|(w, g)| w != g)
                .count();
            assert_eq!(
                bad_convert, 0,
                "{m}x{n}x{k} {draw:?} a_t {a_t} b_t {b_t}: the kit's convert/v1 differs from its host port at {bad_convert} of {} elements (first: {:?})",
                m * n,
                host.iter().zip(&device_convert).position(|(w, g)| w != g)
            );
            let bad = host
                .iter()
                .zip(&got)
                .filter(|(w, g)| !same_word(**w, **g))
                .count();
            assert_eq!(
                bad, 0,
                "{m}x{n}x{k} {draw:?} a_t {a_t} b_t {b_t}: {bad} of {} elements differ from the f32 product converted to fp16 (first: {:?})",
                m * n,
                host.iter().zip(&got).position(|(w, g)| !same_word(*w, *g))
            );
            checked += m * n;
            infinite += got.iter().filter(|&&h| h & 0x7fff == 0x7c00).count();
            subnormal += got
                .iter()
                .filter(|&&h| h & 0x7c00 == 0 && h & 0x03ff != 0)
                .count();
            nan += got
                .iter()
                .filter(|&&h| h & 0x7c00 == 0x7c00 && h & 0x03ff != 0)
                .count();
            for buf in [a, b, c32, c16, conv] {
                assert_eq!(ffi::ack_buffer_free(dev, buf), ffi::ACK_OK);
            }
        }
    }
    eprintln!(
        "{checked} elements lawful (every non-NaN word bit-identical, every NaN a NaN) across {} shapes x 4 layouts ({infinite} infinite, {subnormal} subnormal, {nan} NaN)",
        shapes.len()
    );
    // fp16's own edges were exercised, not only drawn
    assert!(infinite > 0, "no product rounded to infinity");
    assert!(subnormal > 0, "no product rounded to a subnormal");
    assert!(nan > 0, "no NaN product");
    assert_eq!(ffi::ack_close(dev), ffi::ACK_OK, "{}", last_error());
}

fn refused(what: &str, rc: i32, code: i32, name: &str) {
    assert_eq!(rc, code, "{what}: code");
    let e = last_error();
    assert!(e.contains(name), "{what}: {e:?} names {name:?}");
}

#[test]
fn what_the_entry_does_not_address_is_refused_by_name() {
    let Some(dev) = open_or_unmeasured("matmul f16 -> f16 refusals") else {
        return;
    };
    let a = alloc(dev, 64 * 64 * 2);
    let b = alloc(dev, 64 * 64 * 2);
    let c = alloc(dev, 64 * 64 * 2);
    let small = alloc(dev, 16);
    let call = |a, b, c, m: u32, n: u32, k: u32| unsafe {
        ffi::ack_matmul_f16_to_f16(dev, a, b, c, m, n, k, 0, 0, std::ptr::null_mut())
    };
    refused(
        "odd n",
        call(a, b, c, 64, 63, 64),
        ffi::ACK_ERR_SHAPE,
        "matmul-f16-out-odd-n",
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
        "matmul-f16-out-output-aliases-input",
    );
    // the refusals recorded nothing: a plain product still runs and is right-sized
    assert_eq!(call(a, b, c, 64, 64, 64), ffi::ACK_OK, "{}", last_error());
    for buf in [a, b, c, small] {
        assert_eq!(ffi::ack_buffer_free(dev, buf), ffi::ACK_OK);
    }
    assert_eq!(ffi::ack_close(dev), ffi::ACK_OK, "{}", last_error());
}

/// A strided batch of one operand: `z` slabs of `rows` stored rows of `cols`, a pitch
/// of `cols + pad` and a batch stride of `rows * pitch + gap`, every pad and gap
/// NaN. Returns the words and (pitch, batch stride).
fn slabs(
    words: &[u16],
    z: usize,
    rows: usize,
    cols: usize,
    pad: usize,
    gap: usize,
) -> (Vec<u16>, usize, usize) {
    let pitch = cols + pad;
    let stride = rows * pitch + gap;
    let mut out = vec![0x7e00u16; (z - 1) * stride + (rows - 1) * pitch + cols];
    for bz in 0..z {
        for r in 0..rows {
            for c in 0..cols {
                out[bz * stride + r * pitch + c] = words[(bz * rows + r) * cols + c];
            }
        }
    }
    (out, pitch, stride)
}

#[test]
fn the_batched_fp16_output_is_bit_identical_to_the_batched_product_converted() {
    let Some(dev) = open_or_unmeasured("matmul f16 z -> f16") else {
        return;
    };
    // (z, m, n, k, pad, gap, draw): dense batches off the tile; aligned; aligned with
    // pitches and gaps that keep the aligned modules (multiples of 8); off the tile with
    // odd pads and gaps (the edge modules); the student's attention bmm; fp16's edges
    let shapes = [
        (
            3usize,
            64usize,
            40usize,
            256usize,
            0usize,
            0usize,
            Draw::Spread,
        ),
        (2, 256, 256, 128, 0, 0, Draw::Spread),
        (2, 256, 256, 128, 8, 16, Draw::Spread),
        (3, 130, 98, 77, 3, 5, Draw::Spread),
        (5, 640, 2048, 48, 0, 0, Draw::Spread),
        (2, 136, 264, 64, 0, 0, Draw::Huge),
        (2, 136, 264, 64, 0, 0, Draw::Tiny),
        (2, 130, 98, 77, 0, 0, Draw::Nan),
    ];
    let (mut checked, mut infinite, mut subnormal, mut nan) = (0usize, 0usize, 0usize, 0usize);
    for (si, &(z, m, n, k, pad, gap, draw)) in shapes.iter().enumerate() {
        for (a_t, b_t) in [(0i32, 0i32), (0, 1), (1, 0), (1, 1)] {
            // each batch's operands, drawn as the 2-D test draws them
            let (mut a_all, mut b_all) = (Vec::new(), Vec::new());
            for bz in 0..z {
                let (a16, b16) = operands(
                    m,
                    n,
                    k,
                    977 + si as u32 * 31 + (a_t * 2 + b_t) as u32 * 7 + bz as u32,
                    if bz == 0 {
                        draw
                    } else if draw == Draw::Nan {
                        Draw::Spread
                    } else {
                        draw
                    },
                );
                a_all.extend(a16);
                b_all.extend(b16);
            }
            let (a_rows, a_cols) = if a_t != 0 { (k, m) } else { (m, k) };
            let (b_rows, b_cols) = if b_t != 0 { (n, k) } else { (k, n) };
            let (a_words, lda, za) = slabs(&a_all, z, a_rows, a_cols, pad, gap);
            let (b_words, ldb, zb) = slabs(&b_all, z, b_rows, b_cols, pad, gap);
            // C's batch stride: the block, plus an even gap when the operands have one
            let zc = m * n + if gap > 0 { 2 * gap } else { 0 };
            let c_len = (z - 1) * zc + m * n;
            let a = alloc(dev, a_words.len() * 2);
            let b = alloc(dev, b_words.len() * 2);
            let c32 = alloc(dev, c_len * 4);
            let c16 = alloc(dev, c_len * 2);
            let conv = alloc(dev, c_len * 2);
            upload(dev, a, &halves(&a_words));
            upload(dev, b, &halves(&b_words));
            upload(dev, c32, &vec![0u8; c_len * 4]);
            upload(dev, c16, &vec![0xA5u8; c_len * 2]);
            let z_call = |f: unsafe extern "C" fn(
                *const ffi::AckDevice,
                *const alelyon_compute_kit::Buffer,
                *const alelyon_compute_kit::Buffer,
                *const alelyon_compute_kit::Buffer,
                u32,
                u32,
                u32,
                i32,
                i32,
                u32,
                u32,
                u32,
                u32,
                u32,
                u32,
                u32,
                u32,
                *mut f64,
            ) -> i32,
                          c| unsafe {
                f(
                    dev,
                    a,
                    b,
                    c,
                    m as u32,
                    n as u32,
                    k as u32,
                    a_t,
                    b_t,
                    z as u32,
                    0,
                    za as u32,
                    zb as u32,
                    zc as u32,
                    0,
                    lda as u32,
                    ldb as u32,
                    std::ptr::null_mut(),
                )
            };
            let case =
                format!("z {z} {m}x{n}x{k} pad {pad} gap {gap} {draw:?} a_t {a_t} b_t {b_t}");
            assert_eq!(
                z_call(ffi::ack_matmul_f16_z, c32),
                ffi::ACK_OK,
                "{case}: {}",
                last_error()
            );
            convert_f32_to_f16(dev, c32, conv, c_len);
            assert_eq!(
                z_call(ffi::ack_matmul_f16_z_to_f16, c16),
                ffi::ACK_OK,
                "{case}: {}",
                last_error()
            );
            let f32s: Vec<f32> = download(dev, c32, c_len * 4)
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let device_convert = words16(&download(dev, conv, c_len * 2));
            let got = words16(&download(dev, c16, c_len * 2));
            for bz in 0..z {
                for i in 0..m * n {
                    let at = bz * zc + i;
                    let want = to_f16(f32s[at]);
                    if draw != Draw::Nan {
                        assert!(
                            !f32s[at].is_nan(),
                            "{case}: batch {bz} element {i} read a NaN pad or gap"
                        );
                    }
                    assert_eq!(
                        device_convert[at], want,
                        "{case}: batch {bz} element {i}: the kit's convert/v1 differs from its host port"
                    );
                    assert!(
                        same_word(want, got[at]),
                        "{case}: batch {bz} element {i}: {:#06x} differs from the batched product converted to fp16 ({want:#06x})",
                        got[at]
                    );
                    infinite += (got[at] & 0x7fff == 0x7c00) as usize;
                    subnormal += (got[at] & 0x7c00 == 0 && got[at] & 0x03ff != 0) as usize;
                    nan += (got[at] & 0x7c00 == 0x7c00 && got[at] & 0x03ff != 0) as usize;
                }
                // the gap after each batch's block keeps its poison
                if bz + 1 < z {
                    let gap_words = &got[bz * zc + m * n..(bz + 1) * zc];
                    assert!(
                        gap_words.iter().all(|&w| w == 0xA5A5),
                        "{case}: the gap after batch {bz} was written (first at {:?})",
                        gap_words.iter().position(|&w| w != 0xA5A5)
                    );
                }
            }
            checked += z * m * n;
            for buf in [a, b, c32, c16, conv] {
                assert_eq!(ffi::ack_buffer_free(dev, buf), ffi::ACK_OK);
            }
        }
    }
    eprintln!(
        "{checked} batched elements lawful (every non-NaN word bit-identical, every NaN a NaN) across {} shapes x 4 layouts ({infinite} infinite, {subnormal} subnormal, {nan} NaN)",
        shapes.len()
    );
    assert!(
        infinite > 0 && subnormal > 0 && nan > 0,
        "an fp16 edge did not occur"
    );
    assert_eq!(ffi::ack_close(dev), ffi::ACK_OK, "{}", last_error());
}

#[test]
fn what_the_batched_entry_does_not_address_is_refused_by_name() {
    let Some(dev) = open_or_unmeasured("matmul f16 z -> f16 refusals") else {
        return;
    };
    let a = alloc(dev, 2 * 64 * 64 * 2);
    let b = alloc(dev, 2 * 64 * 64 * 2);
    let c = alloc(dev, 2 * 64 * 64 * 2);
    let small = alloc(dev, 16);
    #[allow(clippy::too_many_arguments)]
    let call = |a, b, c, n: u32, z: u32, zmode: u32, zc: u32, kz: u32| unsafe {
        ffi::ack_matmul_f16_z_to_f16(
            dev,
            a,
            b,
            c,
            64,
            n,
            64,
            0,
            0,
            z,
            zmode,
            64 * 64,
            64 * 64,
            zc,
            kz,
            0,
            0,
            std::ptr::null_mut(),
        )
    };
    refused(
        "a K-split",
        call(a, b, c, 64, 2, 1, 64 * 64, 32),
        ffi::ACK_ERR_SHAPE,
        "matmul-f16-out-k-split",
    );
    refused(
        "an odd batch stride of C",
        call(a, b, c, 64, 2, 0, 64 * 64 + 1, 0),
        ffi::ACK_ERR_SHAPE,
        "matmul-f16-out-odd-n",
    );
    refused(
        "odd n",
        call(a, b, c, 63, 1, 0, 0, 0),
        ffi::ACK_ERR_SHAPE,
        "matmul-f16-out-odd-n",
    );
    refused(
        "z = 0",
        call(a, b, c, 64, 0, 0, 0, 0),
        ffi::ACK_ERR_SHAPE,
        "outside",
    );
    refused(
        "short C",
        call(a, b, small, 64, 2, 0, 64 * 64, 0),
        ffi::ACK_ERR_SIZE,
        "buffers too small",
    );
    refused(
        "C is A",
        call(a, b, a, 64, 2, 0, 64 * 64, 0),
        ffi::ACK_ERR_SHAPE,
        "matmul-f16-out-output-aliases-input",
    );
    // the refusals recorded nothing: a batched product still runs
    assert_eq!(
        call(a, b, c, 64, 2, 0, 64 * 64, 0),
        ffi::ACK_OK,
        "{}",
        last_error()
    );
    for buf in [a, b, c, small] {
        assert_eq!(ffi::ack_buffer_free(dev, buf), ffi::ACK_OK);
    }
    assert_eq!(ffi::ack_close(dev), ffi::ACK_OK, "{}", last_error());
}
