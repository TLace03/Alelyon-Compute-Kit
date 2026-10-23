//! amp-unscale-f32/v1 at the C ABI (`ack_amp_unscale`, OPTIONAL, 2026-09-29):
//! GradScaler's non-finite check and unscale of one dense float32 gradient,
//! held BIT FOR BIT to an independent host reference -- this host's own
//! binary32 multiply (x86-64 SSE: IEEE, round to nearest even, subnormals
//! kept; asserted below) with torch's CPU NaN rules applied around it: the
//! scale's NaN first, then the element's, each quieted, and an invalid product
//! (0 * inf) x86's default NaN 0xFFC00000. The kernel computes the same product
//! in integer instructions, so the two share no arithmetic.
//!
//! Covered: every class of float32 as the element (zeros, subnormals, normals,
//! the extremes, infinities, quiet and signalling NaNs of both signs, and
//! random bit patterns) against scales of every class; found_inf's semantics
//! (1.0 stored by any non-finite element, never cleared, a 2.0 overwritten);
//! a scale of exactly 1.0 keeping every bit, a signalling NaN's included; a
//! view at an offset leaving its neighbours untouched; a gradient longer than
//! the grid (the kernel's grid-stride loop); an empty gradient; the refusals
//! by name. The host-only checks need no device; the rest needs one and prints
//! UNMEASURED without it unless ACK_REQUIRE_DEVICE is set.
//!
//! amp-update-scale-f32/v1 (`ack_amp_update_scale`, OPTIONAL, 2026-09-29):
//! GradScaler's scale update, held BIT FOR BIT to torch's CPU kernel as this
//! host executes it -- the float scale widened to double, multiplied by the
//! double factor, rounded to float once (x86-64 SSE) -- over scales of every
//! class, trackers at and around the interval (int32's wrap included), found
//! values of every truth class (-0.0 false, a NaN true), factor pairs that make
//! the product overflow, underflow, go invalid (inf * 0) or stay exact, and
//! intervals of every sign; plus the refusals by name.
use alelyon_compute_kit::amp_ops::{
    AmpUnscaleRequest, AmpUpdateRequest, GROUP, MAX_COUNT, MAX_GROUPS, REQUEST_BYTES,
    UPDATE_REQUEST_BYTES,
};
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

fn is_nan(x: u32) -> bool {
    (x & 0x7F80_0000) == 0x7F80_0000 && (x & 0x7F_FFFF) != 0
}

fn non_finite(x: u32) -> bool {
    (x & 0x7F80_0000) == 0x7F80_0000
}

/// torch's CPU product of one element (its vectorised body, MEASURED on torch
/// 2.14 over 480,000 random bit patterns): the host's hardware product around
/// which torch's NaN precedence is applied explicitly.
fn reference(scale: u32, v: u32) -> u32 {
    if is_nan(scale) {
        return scale | 0x40_0000;
    }
    if is_nan(v) {
        return v | 0x40_0000;
    }
    let invalid = |a: u32, b: u32| (a & 0x7FFF_FFFF) == 0x7F80_0000 && (b & 0x7FFF_FFFF) == 0;
    if invalid(scale, v) || invalid(v, scale) {
        return 0xFFC0_0000;
    }
    (f32::from_bits(scale) * f32::from_bits(v)).to_bits()
}

/// The whole call on the host: the new bits and whether `found` is set.
fn reference_call(scale: u32, data: &[u32]) -> (Vec<u32>, bool) {
    let found = data.iter().any(|&v| non_finite(v));
    if scale == 0x3F80_0000 {
        return (data.to_vec(), found);
    }
    (data.iter().map(|&v| reference(scale, v)).collect(), found)
}

/// A small xorshift, so the bit patterns are fixed without a dependency.
fn patterns(n: usize, seed: u64) -> Vec<u32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 16) as u32
        })
        .collect()
}

const SPECIALS: [u32; 24] = [
    0x0000_0000,
    0x8000_0000,
    0x0000_0001,
    0x8000_0001,
    0x007F_FFFF,
    0x0080_0000,
    0x0040_0000,
    0x7F7F_FFFF,
    0xFF7F_FFFF,
    0x7F80_0000,
    0xFF80_0000,
    0x7FC0_0000,
    0xFFC0_0000,
    0x7F80_0001,
    0xFF80_0001,
    0x7FC1_2345,
    0x7F81_2345,
    0x3F80_0000,
    0xBF80_0000,
    0x3F80_0001,
    0x1234_ABCD,
    0x0080_0001,
    0x0100_0000,
    0x3E99_999A,
];

fn request(
    data_offset: u64,
    count: u64,
    scale_offset: u64,
    found_offset: u64,
) -> [u8; REQUEST_BYTES] {
    AmpUnscaleRequest {
        data_offset,
        count,
        scale_offset,
        found_offset,
    }
    .to_bytes()
}

#[test]
fn a_null_or_short_request_is_refused_before_any_device_lookup() {
    let mut ms = 0f64;
    let rc = unsafe {
        ffi::ack_amp_unscale(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            0,
            &mut ms,
        )
    };
    assert_eq!(rc, ffi::ACK_ERR_NULL);
    let short = [0u8; REQUEST_BYTES - 1];
    let rc = unsafe {
        ffi::ack_amp_unscale(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            short.as_ptr(),
            short.len(),
            &mut ms,
        )
    };
    assert_eq!(
        rc,
        ffi::ACK_ERR_SHAPE,
        "the request is decoded before the null device is looked at"
    );
    assert!(
        last_error().ends_with("amp-request-size"),
        "{}",
        last_error()
    );
}

#[test]
fn the_host_reference_is_ieee_binary32_with_subnormals_on_this_host() {
    // the premises the reference rests on, asserted rather than assumed
    let half_min_normal = f32::from_bits(0x0080_0000) * 0.5;
    assert_eq!(
        half_min_normal.to_bits(),
        0x0040_0000,
        "this host flushes float32 subnormals"
    );
    assert_eq!(
        (f32::from_bits(0x3F80_0001) * f32::from_bits(0x3F80_0001)).to_bits(),
        0x3F80_0002
    );
    assert_eq!(reference(0x0000_0000, 0x7F80_0000), 0xFFC0_0000);
    assert_eq!(
        reference(0x7FC0_0000, 0x7F80_0001),
        0x7FC0_0000,
        "the scale's NaN first"
    );
    assert_eq!(
        reference(0x3F00_0000, 0x7F81_2345),
        0x7FC1_2345,
        "the element's NaN, quieted"
    );
}

#[test]
fn unscale_is_torchs_cpu_kernel_bit_for_bit_and_found_follows_its_rules() {
    let dev = ffi::ack_open();
    if dev.is_null() {
        if std::env::var_os("ACK_REQUIRE_DEVICE").is_some() {
            panic!(
                "amp ABI: ACK_REQUIRE_DEVICE is set and ack_open failed: {}",
                last_error()
            );
        }
        eprintln!("UNMEASURED here (amp ABI): {}", last_error());
        return;
    }
    let alloc = |elements: usize| {
        let b = ffi::ack_buffer_alloc(dev, (elements * 4) as u64);
        assert!(!b.is_null(), "{}", last_error());
        b
    };
    let upload = |buf, words: &[u32]| {
        let rc = unsafe { ffi::ack_upload(dev, buf, words.as_ptr().cast::<u8>(), words.len() * 4) };
        assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
    };
    let download = |buf, elements: usize| {
        let mut out = vec![0u32; elements];
        let rc =
            unsafe { ffi::ack_download(dev, buf, out.as_mut_ptr().cast::<u8>(), out.len() * 4) };
        assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
        out
    };
    let call = |data, scale, found, block: &[u8]| {
        let mut ms = 0f64;
        unsafe {
            ffi::ack_amp_unscale(
                dev,
                data,
                scale,
                found,
                block.as_ptr(),
                block.len(),
                &mut ms,
            )
        }
    };

    // every element class against every scale class, in a view at an offset
    // with untouched neighbours, and scale / found at offsets in their buffers
    let mut data: Vec<u32> = SPECIALS.to_vec();
    data.extend(patterns(70_000, 0x9E37_79B9_7F4A_7C15));
    let n = data.len();
    let guard_words = [
        0x1357_9BDFu32,
        0x2468_ACE0,
        0xDEAD_BEEF,
        0x7F80_0001,
        0x0000_0001,
    ];
    let mut image = guard_words.to_vec();
    image.extend(&data);
    image.extend(guard_words);
    let bdata = alloc(image.len());
    let bscale = alloc(3);
    let bfound = alloc(2);
    let scales: [u32; 16] = [
        0x3F80_0000, // 1.0: every bit kept
        0x3F00_0000, // 0.5
        0x3A80_0000, // 2^-10
        0x3EBD_70A4, // 0.37
        0x3780_0000, // 2^-16
        0x4000_0000, // 2.0
        0x7F31_1D7C, // 2.35e38: overflow to infinity
        0x0DA2_4260, // 1e-30: underflow into the subnormals
        0x0000_0001, // the smallest subnormal
        0x0000_0000, // 0: 0 * inf is invalid
        0x8000_0000, // -0
        0x7F80_0000, // inf
        0xFF80_0000, // -inf
        0x7FC0_0000, // a quiet NaN
        0x7F81_2345, // a signalling NaN
        0x3EAA_AAAB, // 1/3
    ];
    let mut checked = 0usize;
    for &s in &scales {
        for initial_found in [0x0000_0000u32, 0x4000_0000] {
            upload(bdata, &image);
            upload(bscale, &[0xFFFF_FFFF, 0xFFFF_FFFF, s]);
            upload(bfound, &[0xAAAA_AAAA, initial_found]);
            let rc = call(
                bdata,
                bscale,
                bfound,
                &request(guard_words.len() as u64, n as u64, 2, 1),
            );
            assert_eq!(rc, ffi::ACK_OK, "scale {s:#010x}: {}", last_error());
            let got = download(bdata, image.len());
            let (want, set) = reference_call(s, &data);
            assert_eq!(
                &got[..guard_words.len()],
                &guard_words,
                "the words before the view moved"
            );
            assert_eq!(
                &got[guard_words.len() + n..],
                &guard_words,
                "the words after the view moved"
            );
            for (i, (&g, &w)) in got[guard_words.len()..guard_words.len() + n]
                .iter()
                .zip(&want)
                .enumerate()
            {
                assert_eq!(
                    g, w,
                    "scale {s:#010x}, element {i} {:#010x}: kit {g:#010x}, reference {w:#010x}",
                    data[i]
                );
            }
            checked += n;
            let f = download(bfound, 2);
            assert_eq!(f[0], 0xAAAA_AAAA, "found's neighbour moved");
            assert!(set, "the specials hold non-finite elements");
            assert_eq!(
                f[1], 0x3F80_0000,
                "any non-finite element stores 1.0 (over {initial_found:#x})"
            );
            assert_eq!(
                download(bscale, 3)[2],
                s,
                "the scale is read, never written"
            );
        }
    }
    eprintln!("amp ABI: {checked} elements bit for bit against the host reference");

    // all finite: found is left exactly as it was, 0.0, 1.0 or 2.0
    let finite: Vec<u32> = patterns(5_000, 7)
        .into_iter()
        .map(|v| if non_finite(v) { v & 0xBFFF_FFFF } else { v })
        .collect();
    let bfinite = alloc(finite.len());
    for initial_found in [0x0000_0000u32, 0x3F80_0000, 0x4000_0000] {
        upload(bfinite, &finite);
        upload(bscale, &[0, 0, 0x3F00_0000]);
        upload(bfound, &[0, initial_found]);
        let rc = call(
            bfinite,
            bscale,
            bfound,
            &request(0, finite.len() as u64, 2, 1),
        );
        assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
        assert_eq!(
            download(bfound, 2)[1],
            initial_found,
            "found is never cleared or rewritten"
        );
        let (want, set) = reference_call(0x3F00_0000, &finite);
        assert!(!set);
        assert_eq!(download(bfinite, finite.len()), want);
    }

    // a gradient longer than the grid: the grid-stride loop's second lap
    let long = (GROUP as usize) * (MAX_GROUPS as usize) + 4_099;
    let long_data = patterns(long, 11);
    let blong = alloc(long);
    upload(blong, &long_data);
    upload(bscale, &[0, 0, 0x3EBD_70A4]);
    upload(bfound, &[0, 0]);
    let rc = call(blong, bscale, bfound, &request(0, long as u64, 2, 1));
    assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
    let (want, set) = reference_call(0x3EBD_70A4, &long_data);
    let got = download(blong, long);
    let first_bad = got.iter().zip(&want).position(|(g, w)| g != w);
    assert_eq!(
        first_bad, None,
        "the long gradient differs from the reference"
    );
    assert_eq!(download(bfound, 2)[1], if set { 0x3F80_0000 } else { 0 });

    // an empty gradient dispatches nothing and changes nothing
    upload(bfound, &[0, 0]);
    assert_eq!(
        call(bfinite, bscale, bfound, &request(0, 0, 2, 1)),
        ffi::ACK_OK,
        "{}",
        last_error()
    );
    assert_eq!(download(bfound, 2), vec![0, 0]);

    // the refusals, by name, with nothing dispatched
    let refused = |rc: i32, code: i32, name: &str| {
        assert_eq!(rc, code, "{name}: {}", last_error());
        assert!(last_error().ends_with(name), "{name}: {}", last_error());
    };
    refused(
        call(
            bfinite,
            bscale,
            bfound,
            &request(1, finite.len() as u64, 2, 1),
        ),
        ffi::ACK_ERR_SIZE,
        "amp-data-too-small",
    );
    refused(
        call(bfinite, bscale, bfound, &request(0, 1, 3, 1)),
        ffi::ACK_ERR_SIZE,
        "amp-scale-too-small",
    );
    refused(
        call(bfinite, bscale, bfound, &request(0, 1, 2, 2)),
        ffi::ACK_ERR_SIZE,
        "amp-found-too-small",
    );
    refused(
        call(bfinite, bscale, bfound, &request(0, MAX_COUNT + 1, 2, 1)),
        ffi::ACK_ERR_SIZE,
        "amp-count-above-limit",
    );
    refused(
        call(bfinite, bscale, bfinite, &request(0, 1, 2, 1)),
        ffi::ACK_ERR_SHAPE,
        "amp-buffers-alias",
    );
    refused(
        call(bfinite, bfinite, bfound, &request(0, 1, 2, 1)),
        ffi::ACK_ERR_SHAPE,
        "amp-buffers-alias",
    );
    refused(
        call(bfinite, bscale, bscale, &request(0, 1, 2, 1)),
        ffi::ACK_ERR_SHAPE,
        "amp-buffers-alias",
    );
    let long_block = [0u8; REQUEST_BYTES + 1];
    refused(
        call(bfinite, bscale, bfound, &long_block),
        ffi::ACK_ERR_SHAPE,
        "amp-request-size",
    );

    for b in [bdata, bscale, bfound, bfinite, blong] {
        assert_eq!(
            ffi::ack_buffer_free(dev, b),
            ffi::ACK_OK,
            "{}",
            last_error()
        );
    }
    assert_eq!(ffi::ack_close(dev), ffi::ACK_OK, "{}", last_error());
}

/// torch's CPU `_amp_update_scale_` as this host executes it
/// (aten/src/ATen/native/cpu/AmpGradScalerKernels.cpp): a C++ truth test on
/// the float `found`, then `scale * factor` with the float widened to double
/// and the double product rounded to float once, the int tracker's increment
/// wrapping as x86's does.
fn update_reference(
    scale: u32,
    tracker: i32,
    found: u32,
    growth: f64,
    backoff: f64,
    interval: i64,
) -> (u32, i32) {
    let s = f64::from(f32::from_bits(scale));
    if f32::from_bits(found) != 0.0 {
        return (((s * backoff) as f32).to_bits(), 0);
    }
    let successful = tracker.wrapping_add(1);
    if i64::from(successful) == interval {
        let grown = (s * growth) as f32;
        return (
            if grown.is_finite() {
                grown.to_bits()
            } else {
                scale
            },
            0,
        );
    }
    (scale, successful)
}

fn update_request(
    scale_offset: u64,
    tracker_offset: u64,
    found_offset: u64,
    growth_factor: f64,
    backoff_factor: f64,
    growth_interval: i64,
) -> [u8; UPDATE_REQUEST_BYTES] {
    AmpUpdateRequest {
        scale_offset,
        tracker_offset,
        found_offset,
        growth_factor,
        backoff_factor,
        growth_interval,
    }
    .to_bytes()
}

#[test]
fn a_null_or_short_update_request_is_refused_before_any_device_lookup() {
    let mut ms = 0f64;
    let rc = unsafe {
        ffi::ack_amp_update_scale(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            0,
            &mut ms,
        )
    };
    assert_eq!(rc, ffi::ACK_ERR_NULL);
    let short = [0u8; UPDATE_REQUEST_BYTES - 1];
    let rc = unsafe {
        ffi::ack_amp_update_scale(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            short.as_ptr(),
            short.len(),
            &mut ms,
        )
    };
    assert_eq!(
        rc,
        ffi::ACK_ERR_SHAPE,
        "the request is decoded before the null device is looked at"
    );
    assert!(
        last_error().ends_with("amp-request-size"),
        "{}",
        last_error()
    );
}

#[test]
fn the_update_reference_rests_on_this_hosts_double_round_trip() {
    // the premises the reference rests on, asserted rather than assumed
    assert_eq!(
        update_reference(0x4780_0000, 7, 0x3F80_0000, 2.0, 0.5, 2000),
        (0x4700_0000, 0),
        "a found step halves 65536 and clears the tracker"
    );
    assert_eq!(
        update_reference(0x7F81_2345, 0, 0x3F80_0000, 2.0, 0.5, 2000).0,
        0x7FC1_2345,
        "a signalling NaN scale comes back quieted with its payload"
    );
    assert_eq!(
        update_reference(0x7F80_0000, 0, 0x3F80_0000, 2.0, 0.0, 2000).0,
        0xFFC0_0000,
        "inf * 0 is x86's default NaN"
    );
    assert_eq!(
        update_reference(0x0000_0001, 0, 0x3F80_0000, 2.0, 0.5, 2000).0,
        0x0000_0000,
        "half the smallest subnormal rounds to even: zero, not flushed early"
    );
    assert_eq!(
        update_reference(0x0000_0003, 0, 0x3F80_0000, 2.0, 0.5, 2000).0,
        0x0000_0002,
        "1.5 x the smallest subnormal rounds to even"
    );
    assert_eq!(
        update_reference(0x7F00_0000, 1999, 0, 2.0, 0.5, 2000),
        (0x7F00_0000, 0),
        "a growth past float32's range keeps the scale and still clears the tracker"
    );
    assert_eq!(
        update_reference(0x3F80_0000, 0, 0x8000_0000, 2.0, 0.5, 2000),
        (0x3F80_0000, 1),
        "found -0.0 is false"
    );
    assert_eq!(
        update_reference(0x3F80_0000, i32::MAX, 0, 2.0, 0.5, 2000),
        (0x3F80_0000, i32::MIN),
        "the tracker wraps"
    );
}

#[test]
fn update_scale_is_torchs_cpu_kernel_bit_for_bit_and_refuses_by_name() {
    let dev = ffi::ack_open();
    if dev.is_null() {
        if std::env::var_os("ACK_REQUIRE_DEVICE").is_some() {
            panic!(
                "amp update ABI: ACK_REQUIRE_DEVICE is set and ack_open failed: {}",
                last_error()
            );
        }
        eprintln!("UNMEASURED here (amp update ABI): {}", last_error());
        return;
    }
    let alloc = |elements: usize| {
        let b = ffi::ack_buffer_alloc(dev, (elements * 4) as u64);
        assert!(!b.is_null(), "{}", last_error());
        b
    };
    let upload = |buf, words: &[u32]| {
        let rc = unsafe { ffi::ack_upload(dev, buf, words.as_ptr().cast::<u8>(), words.len() * 4) };
        assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
    };
    let download = |buf, elements: usize| {
        let mut out = vec![0u32; elements];
        let rc =
            unsafe { ffi::ack_download(dev, buf, out.as_mut_ptr().cast::<u8>(), out.len() * 4) };
        assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
        out
    };
    let call = |scale, tracker, found, block: &[u8]| {
        let mut ms = 0f64;
        unsafe {
            ffi::ack_amp_update_scale(
                dev,
                scale,
                tracker,
                found,
                block.as_ptr(),
                block.len(),
                &mut ms,
            )
        }
    };

    let scales: [u32; 16] = [
        0x4780_0000, // 65536.0, GradScaler's initial scale
        0x3F80_0000, // 1.0
        0x7F7F_FFFF, // the largest float: any growth overflows
        0x7F00_0000, // 2^127
        0x0000_0001, // the smallest subnormal
        0x0000_0003,
        0x007F_FFFF, // the largest subnormal
        0x0080_0000, // the smallest normal
        0x0000_0000,
        0x8000_0000,
        0xC2F6_0000, // -123.0
        0x7F80_0000,
        0xFF80_0000,
        0x7FC0_0000,
        0x7F81_2345, // a signalling NaN
        0x3EAA_AAAB, // 1/3
    ];
    let founds: [u32; 7] = [
        0x0000_0000,
        0x8000_0000, // -0.0: false
        0x3F80_0000,
        0x4000_0000,
        0x0000_0001, // a subnormal: true
        0x7F80_0000,
        0x7FC0_0000, // a NaN: true
    ];
    let trackers: [i32; 8] = [0, 1, 1998, 1999, 2000, -1, -2, i32::MAX];
    let factors: [(f64, f64); 5] = [
        (2.0, 0.5),
        (1.5, 0.25),
        (f64::from(1.1f32), f64::from(0.9f32)),
        (f64::INFINITY, 0.0),
        (3.0, f64::NEG_INFINITY),
    ];
    let intervals: [i64; 5] = [2000, 1, 0, -1, i64::from(i32::MIN)];

    // every case is one element of three buffers and one recorded call; one
    // download each at the end
    let mut cases = Vec::new();
    for &s in &scales {
        for &f in &founds {
            for &t in &trackers {
                for &(g, b) in &factors {
                    for &i in &intervals {
                        cases.push((s, t, f, g, b, i));
                    }
                }
            }
        }
    }
    let n = cases.len();
    let bscale = alloc(n + 1);
    let btracker = alloc(n + 1);
    let bfound = alloc(n + 1);
    let sentinel = 0x5A5A_5A5Au32;
    let mut scale_words: Vec<u32> = cases.iter().map(|c| c.0).collect();
    let mut tracker_words: Vec<u32> = cases.iter().map(|c| c.1 as u32).collect();
    let mut found_words: Vec<u32> = cases.iter().map(|c| c.2).collect();
    scale_words.push(sentinel);
    tracker_words.push(sentinel);
    found_words.push(sentinel);
    upload(bscale, &scale_words);
    upload(btracker, &tracker_words);
    upload(bfound, &found_words);
    for (k, &(_, _, _, g, b, i)) in cases.iter().enumerate() {
        let k = k as u64;
        let rc = call(bscale, btracker, bfound, &update_request(k, k, k, g, b, i));
        assert_eq!(rc, ffi::ACK_OK, "case {k}: {}", last_error());
    }
    let got_scale = download(bscale, n + 1);
    let got_tracker = download(btracker, n + 1);
    let got_found = download(bfound, n + 1);
    assert_eq!(got_scale[n], sentinel, "the word after the last case moved");
    assert_eq!(
        got_tracker[n], sentinel,
        "the word after the last case moved"
    );
    assert_eq!(got_found, found_words, "found is read, never written");
    for (k, &(s, t, f, g, b, i)) in cases.iter().enumerate() {
        let (want_scale, want_tracker) = update_reference(s, t, f, g, b, i);
        assert_eq!(
            (got_scale[k], got_tracker[k] as i32),
            (want_scale, want_tracker),
            "case {k}: scale {s:#010x}, tracker {t}, found {f:#010x}, factors {g} / {b}, interval {i}"
        );
    }
    eprintln!("amp update ABI: {n} updates bit for bit against the host reference");

    // the refusals, by name, with nothing dispatched
    let refused = |rc: i32, code: i32, name: &str| {
        assert_eq!(rc, code, "{name}: {}", last_error());
        assert!(last_error().ends_with(name), "{name}: {}", last_error());
    };
    let before = download(bscale, n + 1);
    let size = ffi::ACK_ERR_SIZE;
    let n64 = n as u64;
    refused(
        call(
            bscale,
            btracker,
            bfound,
            &update_request(n64 + 1, 0, 0, 2.0, 0.5, 1),
        ),
        size,
        "amp-scale-too-small",
    );
    refused(
        call(
            bscale,
            btracker,
            bfound,
            &update_request(0, n64 + 1, 0, 2.0, 0.5, 1),
        ),
        size,
        "amp-tracker-too-small",
    );
    refused(
        call(
            bscale,
            btracker,
            bfound,
            &update_request(0, 0, n64 + 1, 2.0, 0.5, 1),
        ),
        size,
        "amp-found-too-small",
    );
    refused(
        call(
            bscale,
            btracker,
            bfound,
            &update_request(0, 0, 0, 0.37, 0.5, 1),
        ),
        size,
        "amp-update-factor-not-f32",
    );
    refused(
        call(
            bscale,
            btracker,
            bfound,
            &update_request(0, 0, 0, 2.0, f64::NAN, 1),
        ),
        size,
        "amp-update-factor-not-f32",
    );
    refused(
        call(
            bscale,
            btracker,
            bfound,
            &update_request(0, 0, 0, 2.0, 0.5, i64::from(i32::MAX) + 1),
        ),
        size,
        "amp-update-interval-above-i32",
    );
    refused(
        call(
            bscale,
            btracker,
            bfound,
            &update_request(u64::from(u32::MAX) + 1, 0, 0, 2.0, 0.5, 1),
        ),
        size,
        "amp-word-above-u32",
    );
    let shape = ffi::ACK_ERR_SHAPE;
    let ok_block = update_request(0, 0, 0, 2.0, 0.5, 1);
    refused(
        call(bscale, bscale, bfound, &ok_block),
        shape,
        "amp-buffers-alias",
    );
    refused(
        call(bscale, btracker, bscale, &ok_block),
        shape,
        "amp-buffers-alias",
    );
    refused(
        call(bscale, btracker, btracker, &ok_block),
        shape,
        "amp-buffers-alias",
    );
    let long_block = [0u8; UPDATE_REQUEST_BYTES + 1];
    refused(
        call(bscale, btracker, bfound, &long_block),
        shape,
        "amp-request-size",
    );
    assert_eq!(
        download(bscale, n + 1),
        before,
        "a refused call wrote nothing"
    );

    for b in [bscale, btracker, bfound] {
        assert_eq!(
            ffi::ack_buffer_free(dev, b),
            ffi::ACK_OK,
            "{}",
            last_error()
        );
    }
    assert_eq!(ffi::ack_close(dev), ffi::ACK_OK, "{}", last_error());
}
