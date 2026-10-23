//! `ack_cast` (f32 <-> bf16 over whole blocks) on a real device, 2026-09-27:
//! the kernel moves each invocation's four elements as one vector load and one
//! store, and takes a group that `n` cuts short element by element.
//!
//! THE LAW IS BIT IDENTITY with the host's rounding: f32 -> bf16 rounds to
//! nearest even with a NaN kept quiet, and bf16 -> f32 is exact. The counts
//! sweep every remainder mod 4 around the vector and workgroup boundaries
//! (1,024 elements a workgroup). The destinations are poisoned, and the
//! elements past `n` must keep the poison: the kernel writes exactly `n`. The
//! values include ties at the rounding point, subnormals, infinities, signed
//! zeros and NaNs with payloads.
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

/// f32 values with the cases rounding can get wrong, then a spread.
fn values(n: usize) -> Vec<f32> {
    let special = [
        0.0f32,
        -0.0,
        1.0,
        -1.0,
        f32::from_bits(0x3f80_8000), // a tie at the rounding point: to even (down)
        f32::from_bits(0x3f81_8000), // a tie: to even (up)
        f32::from_bits(0x3f80_8001), // just above a tie: up
        f32::from_bits(0x0000_0001), // the smallest subnormal
        f32::from_bits(0x807f_ffff), // the largest negative subnormal
        f32::MAX,                    // rounds to +inf
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(0x7fc0_0001), // a quiet NaN with a low payload
        f32::from_bits(0xff80_0001), // a signalling NaN: kept quiet
        f32::from_bits(0x7fbf_ffff), // a NaN whose top half alone would read as inf
    ];
    let mut state = 0x9e37_79b9_7f4a_7c15u64 ^ n as u64;
    (0..n)
        .map(|i| {
            if i < special.len() {
                return special[i];
            }
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let exponent = (state % 60) as i32 - 30;
            let mantissa = ((state >> 8) & 0x7f_ffff) as f32 / (1u32 << 23) as f32 + 1.0;
            let sign = if state & 1 == 0 { 1.0 } else { -1.0 };
            sign * mantissa * 2f32.powi(exponent)
        })
        .collect()
}

#[test]
fn casts_are_bit_identical_to_the_host_law_at_every_remainder_and_write_exactly_n() {
    let Some(dev) = open_or_unmeasured("ack_cast") else {
        return;
    };
    const POISON16: u16 = 0xdead;
    const POISON32: u32 = 0xdead_beef;
    for n in [
        1usize, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 4096,
        4099, 65_537,
    ] {
        let spare = 8; // elements past n that must keep the poison
        let xs = values(n);
        // f32 -> bf16
        let src = alloc(dev, (n + spare) * 4);
        let dst = alloc(dev, (n + spare) * 2);
        let src_bytes: Vec<u8> = xs
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .chain(std::iter::repeat_n(0u8, spare * 4))
            .collect();
        upload(dev, src, &src_bytes);
        let poison: Vec<u8> = std::iter::repeat_n(POISON16.to_le_bytes(), n + spare)
            .flatten()
            .collect();
        upload(dev, dst, &poison);
        assert_eq!(
            ffi::ack_cast(dev, src, dst, n as u64, 0),
            ffi::ACK_OK,
            "{}",
            last_error()
        );
        let got = download(dev, dst, (n + spare) * 2);
        for i in 0..n + spare {
            let word = u16::from_le_bytes([got[2 * i], got[2 * i + 1]]);
            let want = if i < n { to_bf16(xs[i]) } else { POISON16 };
            assert_eq!(
                word,
                want,
                "f32 -> bf16, n {n}, element {i} (input {:#010x})",
                xs.get(i).map_or(0, |x| x.to_bits())
            );
        }
        // bf16 -> f32, from the bf16 just made
        let back = alloc(dev, (n + spare) * 4);
        let poison32: Vec<u8> = std::iter::repeat_n(POISON32.to_le_bytes(), n + spare)
            .flatten()
            .collect();
        upload(dev, back, &poison32);
        assert_eq!(
            ffi::ack_cast(dev, dst, back, n as u64, 1),
            ffi::ACK_OK,
            "{}",
            last_error()
        );
        let widened = download(dev, back, (n + spare) * 4);
        for i in 0..n + spare {
            let word = u32::from_le_bytes([
                widened[4 * i],
                widened[4 * i + 1],
                widened[4 * i + 2],
                widened[4 * i + 3],
            ]);
            let want = if i < n {
                u32::from(to_bf16(xs[i])) << 16
            } else {
                POISON32
            };
            assert_eq!(word, want, "bf16 -> f32, n {n}, element {i}");
        }
        for buf in [src, dst, back] {
            assert_eq!(ffi::ack_buffer_free(dev, buf), ffi::ACK_OK);
        }
    }
    assert_eq!(ffi::ack_close(dev), ffi::ACK_OK, "{}", last_error());
}
