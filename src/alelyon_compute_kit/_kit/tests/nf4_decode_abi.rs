//! `ack_nf4_decode` (nf4-decode/v1, 2026-09-30): bitsandbytes' NF4 with double
//! quantisation, decoded from its stored form on a real device.
//!
//! THE LAW IS BIT IDENTITY, for every result that is not a NaN, with the
//! reference decode (`research/subbit_expert/moe/nf4.py`) and bitsandbytes':
//! value `s` of member `m` in block `b = s / 64` is
//! `code[index(s)] * fl(fl(map[absmax_q[b]] * absmax2[b / 256]) + offsets[m])`,
//! each product and sum one FP32 rounding, then one rounding to the output
//! storage (round to nearest even for bf16 and fp16, c10's NaN for fp16). The
//! host computes it here with Rust's FP32 arithmetic, which is IEEE's, and the
//! roundings the kit's other tests use. Indices sit two to a byte, the first
//! value in the HIGH nibble.
//!
//! Held: a whole group of three members of different sizes, the last ending in
//! a partial block, an odd count (a pad nibble) and a partial word; a selection
//! of experts by int64 ids, repeats included, and ids outside the members
//! (negative, too large), whose values must be NaN; one member decoded alone
//! from its place in the group (`src_base`); one registered teacher projection
//! (2,048 x 768). The code is bitsandbytes' NF4 table as torch holds it, bit for
//! bit; the map is any 256 values (the law does not depend on them). Every
//! output is poisoned first with a signalling NaN, which never counts as a
//! result. The nested scales cover zero, magnitudes whose values overflow fp16,
//! an infinity and a NaN; results stay in FP32's normal range or at zero.
//!
//! Skips as UNMEASURED without a device unless ACK_REQUIRE_DEVICE is set.
use alelyon_compute_kit::ffi;
use alelyon_compute_kit::nf4_ops::{Nf4Storage, NESTED_VALUES, TABLE_ENTRIES};
use alelyon_compute_kit::Buffer;
use std::ffi::{c_char, CStr};

/// bitsandbytes' NF4 values (`functional.get_4bit_type("nf4")`) as torch's
/// float32 tensor holds them (`nf4.NF4_CODE.view(torch.int32)`).
const NF4_BITS: [u32; 16] = [
    0xbf80_0000,
    0xbf32_39b1,
    0xbf06_6b30,
    0xbeca_32a0,
    0xbe91_a24d,
    0xbe3d_353f,
    0xbdba_7871,
    0x0000_0000,
    0x3da2_faff,
    0x3e24_cae3,
    0x3e7c_04dd,
    0x3ead_033a,
    0x3ee1_a4b8,
    0x3f10_07ab,
    0x3f39_13b3,
    0x3f80_0000,
];

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

fn word_of(storage: Nf4Storage, x: f32) -> u32 {
    match storage {
        Nf4Storage::F32 => x.to_bits(),
        Nf4Storage::Bf16 => bf16_bits(x) as u32,
        Nf4Storage::F16 => f16_bits(x) as u32,
    }
}

fn is_nan_word(storage: Nf4Storage, w: u32) -> bool {
    match storage {
        Nf4Storage::F32 => f32::from_bits(w).is_nan(),
        Nf4Storage::Bf16 => (w & 0x7F80) == 0x7F80 && (w & 0x7F) != 0,
        Nf4Storage::F16 => (w & 0x7C00) == 0x7C00 && (w & 0x3FF) != 0,
    }
}

fn poison_word(storage: Nf4Storage) -> u32 {
    match storage {
        Nf4Storage::F32 => 0x7FA5_A5A5,
        Nf4Storage::Bf16 => 0x7FA5,
        Nf4Storage::F16 => 0x7D55,
    }
}

/// A deterministic byte stream: every value appears.
fn stream(n: usize, salt: u32) -> Vec<u8> {
    (0..n)
        .map(|i| ((i as u32).wrapping_mul(2_654_435_761) ^ salt).rotate_left(11) as u8)
        .collect()
}

fn le_bytes<T: Copy, const N: usize>(values: &[T], to: impl Fn(T) -> [u8; N]) -> Vec<u8> {
    values.iter().flat_map(|v| to(*v)).collect()
}

/// A stored source: members back to back, each starting on a whole nested block.
struct Stored {
    packed: Vec<u8>,
    absmax_q: Vec<u8>,
    absmax2: Vec<f32>,
    offsets: Vec<f32>,
    starts: Vec<i64>,
    values: usize,
}

impl Stored {
    /// `sizes`: each member's values; all but the last whole nested blocks.
    fn new(sizes: &[usize], salt: u32) -> Self {
        let mut starts = vec![];
        let mut at = 0usize;
        for (i, &size) in sizes.iter().enumerate() {
            starts.push(at as i64);
            at += if i + 1 < sizes.len() {
                assert_eq!(size % NESTED_VALUES as usize, 0);
                size
            } else {
                size
            };
        }
        let values = at;
        let blocks = values.div_ceil(64);
        let nested = blocks.div_ceil(256);
        let mut absmax2: Vec<f32> = (0..nested)
            .map(|i| 0.002 + ((i as u32 * 7919 + salt) % 1000) as f32 * 0.0003)
            .collect();
        // the edges: zero, a scale whose values overflow fp16, an infinity, a NaN
        for (slot, e) in absmax2
            .iter_mut()
            .skip(1)
            .zip([0.0f32, 9.0e4, f32::INFINITY, f32::NAN])
        {
            *slot = e;
        }
        let offsets = (0..sizes.len())
            .map(|m| [0.011f32, -0.004, 0.0, 0.0073, -0.0][m % 5])
            .collect();
        Self {
            packed: stream(values.div_ceil(2), salt),
            absmax_q: stream(blocks, salt ^ 0x5A5A),
            absmax2,
            offsets,
            starts,
            values,
        }
    }

    /// Value `s` of the source (from its front), in member `m`, rounded to `storage`.
    fn value(&self, tables: &[f32], m: usize, s: usize, storage: Nf4Storage) -> u32 {
        let byte = self.packed[s / 2];
        let index = if s % 2 == 0 { byte >> 4 } else { byte & 0xF } as usize;
        let b = s / 64;
        let scale = tables[16 + self.absmax_q[b] as usize] * self.absmax2[b / 256];
        let absmax = scale + self.offsets[m];
        word_of(storage, tables[index] * absmax)
    }
}

fn tables() -> Vec<f32> {
    let mut t: Vec<f32> = NF4_BITS.iter().map(|b| f32::from_bits(*b)).collect();
    t.extend((0..256).map(|i| (i as f32 - 127.5) / 128.0 + if i == 200 { 0.0 } else { 1.0e-3 }));
    assert_eq!(t.len(), TABLE_ENTRIES);
    t
}

struct Device(*mut ffi::AckDevice);

impl Device {
    fn upload_new(&self, bytes: &[u8]) -> *mut Buffer {
        let b = ffi::ack_buffer_alloc(self.0, bytes.len().div_ceil(4).max(1) as u64 * 4);
        assert!(!b.is_null(), "{}", last_error());
        let mut padded = bytes.to_vec();
        padded.resize(bytes.len().div_ceil(4).max(1) * 4, 0);
        let rc = unsafe { ffi::ack_upload(self.0, b, padded.as_ptr(), padded.len()) };
        assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
        b
    }
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
}

/// The stored source on the device: [packed, absmax_q, absmax2, tables, starts, offsets].
fn upload(d: &Device, s: &Stored, tables: &[f32]) -> [*mut Buffer; 6] {
    [
        d.upload_new(&s.packed),
        d.upload_new(&s.absmax_q),
        d.upload_new(&le_bytes(&s.absmax2, f32::to_le_bytes)),
        d.upload_new(&le_bytes(tables, f32::to_le_bytes)),
        d.upload_new(&le_bytes(&s.starts, i64::to_le_bytes)),
        d.upload_new(&le_bytes(&s.offsets, f32::to_le_bytes)),
    ]
}

/// One call; `ids` null for the whole source.
#[allow(clippy::too_many_arguments)]
fn decode(
    d: &Device,
    src: [*const Buffer; 6],
    ids: *const Buffer,
    out: *const Buffer,
    n: usize,
    (members, selected, member_values, src_base): (usize, usize, usize, usize),
    storage: u32,
) -> i32 {
    unsafe {
        ffi::ack_nf4_decode(
            d.0,
            src[0],
            src[1],
            src[2],
            src[3],
            src[4],
            src[5],
            ids,
            out,
            n as u32,
            members as u32,
            selected as u32,
            member_values as u32,
            src_base as u32,
            storage,
            std::ptr::null_mut(),
        )
    }
}

/// Decodes `n` values into a poisoned output and compares every word with
/// `want(i)` (None: a NaN is required). Returns (words checked, NaN words).
#[allow(clippy::too_many_arguments)]
fn check(
    d: &Device,
    what: &str,
    src: [*const Buffer; 6],
    ids: *const Buffer,
    n: usize,
    shape: (usize, usize, usize, usize),
    storage: Nf4Storage,
    want: impl Fn(usize) -> Option<u32>,
) -> (usize, usize) {
    let width = storage.bytes() as usize;
    let poison: Vec<u8> = (0..n)
        .flat_map(|_| poison_word(storage).to_le_bytes()[..width].to_vec())
        .collect();
    let out = d.upload_new(&poison);
    let rc = decode(d, src, ids, out, n, shape, storage as u32);
    assert_eq!(rc, ffi::ACK_OK, "{what} {storage:?}: {}", last_error());
    let got = d.download(out, n * width);
    let (mut bad, mut first, mut nans) = (0usize, None, 0usize);
    for i in 0..n {
        let have = match storage {
            Nf4Storage::F32 => u32::from_le_bytes(got[4 * i..4 * i + 4].try_into().unwrap()),
            _ => u16::from_le_bytes([got[2 * i], got[2 * i + 1]]) as u32,
        };
        let ok = have != poison_word(storage)
            && match want(i) {
                None => is_nan_word(storage, have),
                Some(w) => have == w || (is_nan_word(storage, have) && is_nan_word(storage, w)),
            };
        if !ok {
            bad += 1;
            first.get_or_insert((i, have, want(i)));
        }
        nans += usize::from(is_nan_word(storage, have));
    }
    assert_eq!(
        bad, 0,
        "{what} {storage:?}: {bad} of {n} words differ from the reference decode; first (index, device, host) {first:x?}"
    );
    d.free(&[out]);
    (n, nans)
}

#[test]
fn every_value_is_the_reference_decode_bit_for_bit() {
    let Some(dev) = open_or_unmeasured("nf4-decode") else {
        return;
    };
    let d = Device(dev);
    let t = tables();
    let nv = NESTED_VALUES as usize;
    let storages = [Nf4Storage::F32, Nf4Storage::Bf16, Nf4Storage::F16];
    let (mut checked, mut nans) = (0usize, 0usize);
    let mut tally = |(c, x): (usize, usize)| {
        checked += c;
        nans += x;
    };
    // a whole group: three members, the last ending in a partial block, an odd count and a partial word
    let group = Stored::new(&[nv, 2 * nv, nv + 64 * 3 + 13], 0xC0DE);
    let src = upload(&d, &group, &t);
    let srcc = src.map(|b| b as *const Buffer);
    let member_of = |s: usize| {
        group
            .starts
            .iter()
            .rposition(|&st| st as usize <= s)
            .unwrap()
    };
    for storage in storages {
        let shape = (3, 0, 0, 0);
        tally(check(
            &d,
            "whole group",
            srcc,
            std::ptr::null(),
            group.values,
            shape,
            storage,
            |i| Some(group.value(&t, member_of(i), i, storage)),
        ));
        // member 1 decoded alone from its place in the group: its own start and offset, the group's buffers
        let one = [0i64];
        let off = [group.offsets[1]];
        let starts1 = d.upload_new(&le_bytes(&one, i64::to_le_bytes));
        let offsets1 = d.upload_new(&le_bytes(&off, f32::to_le_bytes));
        let view = [
            srcc[0],
            srcc[1],
            srcc[2],
            srcc[3],
            starts1 as *const Buffer,
            offsets1 as *const Buffer,
        ];
        tally(check(
            &d,
            "member view",
            view,
            std::ptr::null(),
            2 * nv,
            (1, 0, 0, nv),
            storage,
            |i| Some(group.value(&t, 1, nv + i, storage)),
        ));
        d.free(&[starts1, offsets1]);
    }
    d.free(&src);
    // a selection of experts of one size, repeats included, then ids outside the members
    let experts = Stored::new(&[nv; 6], 0xE4);
    let src = upload(&d, &experts, &t);
    let srcc = src.map(|b| b as *const Buffer);
    for (what, sel) in [
        ("selection", vec![4i64, 0, 5, 5]),
        ("bad ids", vec![1i64, 99, -1]),
    ] {
        let ids = d.upload_new(&le_bytes(&sel, i64::to_le_bytes));
        for storage in storages {
            let shape = (6, sel.len(), nv, 0);
            tally(check(
                &d,
                what,
                srcc,
                ids,
                sel.len() * nv,
                shape,
                storage,
                |i| {
                    let m = sel[i / nv];
                    (0..6)
                        .contains(&m)
                        .then(|| experts.value(&t, m as usize, m as usize * nv + i % nv, storage))
                },
            ));
        }
        d.free(&[ids]);
    }
    d.free(&src);
    // one registered teacher projection, 2,048 x 768: 96 nested blocks
    let proj = Stored::new(&[2048 * 768], 0x7EAC);
    let src = upload(&d, &proj, &t);
    tally(check(
        &d,
        "teacher projection",
        src.map(|b| b as *const Buffer),
        std::ptr::null(),
        proj.values,
        (1, 0, 0, 0),
        Nf4Storage::F16,
        |i| Some(proj.value(&t, 0, i, Nf4Storage::F16)),
    ));
    d.free(&src);
    eprintln!(
        "nf4-decode: {checked} words bit-identical to the reference decode; {nans} NaN results"
    );
    assert!(nans > 0, "the edges and the bad ids reach NaN results");
    assert_eq!(ffi::ack_close(dev), ffi::ACK_OK, "{}", last_error());
}

#[test]
fn a_call_the_family_does_not_address_is_refused_by_name() {
    let Some(dev) = open_or_unmeasured("nf4-decode refusals") else {
        return;
    };
    let d = Device(dev);
    let t = tables();
    let nv = NESTED_VALUES as usize;
    let s = Stored::new(&[nv, nv], 0x51);
    let src = upload(&d, &s, &t);
    let srcc = src.map(|b| b as *const Buffer);
    let out = d.upload_new(&vec![0u8; 4 * 2 * nv]);
    let ids = d.upload_new(&le_bytes(&[1i64, 0], i64::to_le_bytes));
    let null = std::ptr::null();
    let f32s = Nf4Storage::F32 as u32;
    let refused = |rc: i32, want: i32, name: &str| {
        assert_eq!(rc, want, "{name}: {}", last_error());
        assert!(last_error().contains(name), "{name}: {}", last_error());
    };
    refused(
        decode(&d, srcc, null, out, 64, (2, 0, 0, 0), 3),
        ffi::ACK_ERR_SHAPE,
        "nf4-storage-out-of-range",
    );
    refused(
        decode(&d, srcc, null, out, 0, (2, 0, 0, 0), f32s),
        ffi::ACK_ERR_SHAPE,
        "nf4-zero-values",
    );
    refused(
        decode(&d, srcc, null, out, 64, (0, 0, 0, 0), f32s),
        ffi::ACK_ERR_SHAPE,
        "nf4-members-out-of-range",
    );
    refused(
        decode(&d, srcc, ids, out, 192, (2, 2, 96, 0), f32s),
        ffi::ACK_ERR_SHAPE,
        "nf4-selection-out-of-range",
    );
    refused(
        decode(&d, srcc, ids, out, 64, (2, 2, nv, 0), f32s),
        ffi::ACK_ERR_SHAPE,
        "nf4-selection-out-of-range",
    );
    refused(
        decode(&d, srcc, null, out, 64, (2, 0, 0, 64), f32s),
        ffi::ACK_ERR_SHAPE,
        "nf4-base-out-of-range",
    );
    refused(
        decode(&d, srcc, null, out, 64, (2, 0, 0, 2 * nv), f32s),
        ffi::ACK_ERR_SHAPE,
        "nf4-base-out-of-range",
    );
    refused(
        decode(&d, srcc, null, out, 2 * nv + 8, (2, 0, 0, 0), f32s),
        ffi::ACK_ERR_SIZE,
        "nf4-source-too-small",
    );
    refused(
        decode(&d, srcc, null, out, 64, (3, 0, 0, 0), f32s),
        ffi::ACK_ERR_SIZE,
        "nf4-starts-too-small",
    );
    refused(
        decode(&d, srcc, ids, out, 3 * nv, (2, 3, nv, 0), f32s),
        ffi::ACK_ERR_SIZE,
        "nf4-ids-too-small",
    );
    let small = d.upload_new(&[0u8; 4]);
    let short_tables = [
        srcc[0],
        srcc[1],
        srcc[2],
        small as *const Buffer,
        srcc[4],
        srcc[5],
    ];
    refused(
        decode(&d, short_tables, null, out, 64, (2, 0, 0, 0), f32s),
        ffi::ACK_ERR_SIZE,
        "nf4-tables-too-small",
    );
    refused(
        decode(&d, srcc, null, small, 64, (2, 0, 0, 0), f32s),
        ffi::ACK_ERR_SIZE,
        "nf4-output-too-small",
    );
    refused(
        decode(&d, srcc, null, srcc[3], 16, (2, 0, 0, 0), f32s),
        ffi::ACK_ERR_SHAPE,
        "nf4-output-aliases-input",
    );
    // the positive control: the same buffers carry calls that fit them, so each refusal above is its own bound's
    assert_eq!(
        decode(&d, srcc, null, out, 2 * nv, (2, 0, 0, 0), f32s),
        ffi::ACK_OK,
        "{}",
        last_error()
    );
    assert_eq!(
        decode(&d, srcc, ids, out, 2 * nv, (2, 2, nv, 0), f32s),
        ffi::ACK_OK,
        "{}",
        last_error()
    );
    d.free(&[out, ids, small]);
    d.free(&src);
    assert_eq!(ffi::ack_close(dev), ffi::ACK_OK, "{}", last_error());
}
