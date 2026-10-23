//! `ack_matmul_nf4` (nf4-matmul/v1, 2026-09-30): the kit's own cooperative-
//! matrix product with its B operand held in NF4 with double quantisation and
//! read in its stored form, on a real device.
//!
//! THE LAW IS BIT IDENTITY with nf4-decode/v1 followed by the kit's dense
//! product, computed on the same device through the kit's own entries: each
//! group's weight is decoded by `ack_nf4_decode` (a selection of its members)
//! into bf16 or fp16, and that weight times the group's rows of A goes through
//! `ack_matmul_bf16` / `ack_matmul_f16`. The f32 words of C are compared,
//! signed zeros included; C is poisoned first, so a word the kernel never
//! wrote fails. A NaN must be a NaN in both, its sign and payload free.
//!
//! Held: a grouped call over three routed experts in a permuted order, each
//! weight two members with different offsets stacked (the switch falls
//! mid-weight), one group with no rows, the source a nested block into its
//! buffers; the transposed form (C = A . W, the backward's); an attention-sized
//! single weight at one row (decode) and at a ragged row count; and the kernel's
//! guards against a groups table that points past the rows or the source, puts a
//! weight base off a packed word, or counts more rows than the call's largest
//! (the call writes nothing for such a group). Refusals by name, with positive
//! controls.
//!
//! fp16 SUBNORMALS (2026-09-30): weights and rows below fp16's smallest normal,
//! forward and transposed, two members a weight. Since 2026-09-30 a device that
//! reports shaderFloat16 and the float controls takes the fp16 modules that round
//! each decoded weight with its own conversion (NF4_HW), and
//! ACK_NF4_MATMUL_F16_HW=0 keeps the integer rounding: run this test both ways.
//!
//! A 16-BIT C (2026-10-01): `ack_matmul_nf4_to_16` writes C in the operands'
//! type from the product's own epilogue. Every case below also runs it into a
//! poisoned C: each word must be the f32 C's word rounded once (fp16 to nearest
//! even with subnormals kept and overflow to infinity, bf16 as cast_f32_bf16
//! rounds; a NaN a NaN), and each group's rows must be, word for word, the kit's
//! dense 16-bit-output product of the decoded weight (`ack_matmul_f16_to_f16`,
//! `ack_matmul_bf16_to_bf16`). The guards hold for it too, and its own refusals
//! (an odd N, a C too small for 16-bit words) are named.
//!
//! Skips as UNMEASURED without a device unless ACK_REQUIRE_DEVICE is set; each
//! element type the device's cooperative matrices lack is UNMEASURED by name,
//! and so is a 16-bit C the device cannot round (fp16 without the float
//! controls).
use alelyon_compute_kit::ffi;
use alelyon_compute_kit::nf4_ops::{Nf4Elem, NESTED_VALUES, TABLE_ENTRIES};
use alelyon_compute_kit::Buffer;
use std::ffi::{c_char, CStr};

/// bitsandbytes' NF4 values as torch's float32 tensor holds them.
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
const POISON: u32 = 0x7FA5_A5A5;

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

fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    if x.is_nan() {
        return ((bits >> 16) | 0x0040) as u16;
    }
    ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) as u16
}

/// f32 -> f16, round to nearest even (the operands here are finite and in range).
fn f32_to_f16(x: f32) -> u16 {
    let w = x.to_bits();
    let sign = ((w >> 16) & 0x8000) as u16;
    let e = (w >> 23) & 0xFF;
    let m = w & 0x7F_FFFF;
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

fn stream(n: usize, salt: u32) -> Vec<u8> {
    (0..n)
        .map(|i| ((i as u32).wrapping_mul(2_654_435_761) ^ salt).rotate_left(11) as u8)
        .collect()
}

fn le<T: Copy, const N: usize>(values: &[T], to: impl Fn(T) -> [u8; N]) -> Vec<u8> {
    values.iter().flat_map(|v| to(*v)).collect()
}

struct Device(*mut ffi::AckDevice);

impl Device {
    /// A buffer of whole words holding `bytes`, the pad zero.
    fn upload(&self, bytes: &[u8]) -> *mut Buffer {
        let words = bytes.len().div_ceil(4).max(1);
        let b = ffi::ack_buffer_alloc(self.0, (words * 4) as u64);
        assert!(!b.is_null(), "{}", last_error());
        let mut padded = bytes.to_vec();
        padded.resize(words * 4, 0);
        let rc = unsafe { ffi::ack_upload(self.0, b, padded.as_ptr(), padded.len()) };
        assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
        b
    }
    fn words(&self, buf: *const Buffer, count: usize) -> Vec<u32> {
        let mut out = vec![0u8; count * 4];
        let rc = unsafe { ffi::ack_download(self.0, buf, out.as_mut_ptr(), out.len()) };
        assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
        out.chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
    fn free(&self, bufs: &[*mut Buffer]) {
        for &b in bufs {
            assert_eq!(ffi::ack_buffer_free(self.0, b), ffi::ACK_OK);
        }
    }
}

/// A group of `members` NF4 members of `member_values` each (a whole number of
/// nested blocks), back to back, the source `base_blocks` nested blocks into
/// its buffers (whose leading values are filler the kernel must skip).
struct Group {
    packed: Vec<u8>,
    absmax_q: Vec<u8>,
    absmax2: Vec<f32>,
    starts: Vec<i64>,
    offsets: Vec<f32>,
    member_values: usize,
    base: usize,
}

impl Group {
    fn new(members: usize, member_values: usize, base_blocks: usize, salt: u32) -> Self {
        let nv = NESTED_VALUES as usize;
        assert_eq!(member_values % nv, 0);
        let base = base_blocks * nv;
        let values = base + members * member_values;
        let blocks = values / 64;
        let nested = blocks / 256;
        Self {
            packed: stream(values / 2, salt),
            absmax_q: stream(blocks, salt ^ 0x77),
            absmax2: (0..nested)
                .map(|i| 0.004 + ((i as u32 * 7919 + salt) % 997) as f32 * 0.0002)
                .collect(),
            starts: (0..members).map(|m| (m * member_values) as i64).collect(),
            offsets: (0..members)
                .map(|m| [0.011f32, -0.004, 0.0073, 0.0, -0.0021][m % 5])
                .collect(),
            member_values,
            base,
        }
    }
}

fn tables() -> Vec<f32> {
    let mut t: Vec<f32> = NF4_BITS.iter().map(|b| f32::from_bits(*b)).collect();
    t.extend((0..256).map(|i| (i as f32 - 127.5) / 128.0));
    assert_eq!(t.len(), TABLE_ENTRIES);
    t
}

/// The group's stored form on the device: [packed, absmax_q, absmax2, tables, starts, offsets].
fn stored(d: &Device, g: &Group) -> [*mut Buffer; 6] {
    [
        d.upload(&g.packed),
        d.upload(&g.absmax_q),
        d.upload(&le(&g.absmax2, f32::to_le_bytes)),
        d.upload(&le(&tables(), f32::to_le_bytes)),
        d.upload(&le(&g.starts, i64::to_le_bytes)),
        d.upload(&le(&g.offsets, f32::to_le_bytes)),
    ]
}

#[allow(clippy::too_many_arguments)]
fn matmul_nf4(
    d: &Device,
    a: *const Buffer,
    src: &[*mut Buffer; 6],
    groups: *const Buffer,
    c: *const Buffer,
    (rows, max_rows, n, k): (usize, usize, usize, usize),
    b_t: bool,
    (ngroups, members, base): (usize, usize, usize),
    elem: u32,
) -> i32 {
    unsafe {
        ffi::ack_matmul_nf4(
            d.0,
            a,
            src[0],
            src[1],
            src[2],
            src[3],
            src[4],
            src[5],
            groups,
            c,
            rows as u32,
            max_rows as u32,
            n as u32,
            k as u32,
            u32::from(b_t),
            0,
            ngroups as u32,
            members as u32,
            base as u32,
            elem,
            std::ptr::null_mut(),
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn matmul_nf4_16(
    d: &Device,
    a: *const Buffer,
    src: &[*mut Buffer; 6],
    groups: *const Buffer,
    c: *const Buffer,
    (rows, max_rows, n, k): (usize, usize, usize, usize),
    b_t: bool,
    (ngroups, members, base): (usize, usize, usize),
    elem: u32,
) -> i32 {
    unsafe {
        ffi::ack_matmul_nf4_to_16(
            d.0,
            a,
            src[0],
            src[1],
            src[2],
            src[3],
            src[4],
            src[5],
            groups,
            c,
            rows as u32,
            max_rows as u32,
            n as u32,
            k as u32,
            u32::from(b_t),
            0,
            ngroups as u32,
            members as u32,
            base as u32,
            elem,
            std::ptr::null_mut(),
        )
    }
}

/// The 16-bit halves of downloaded words, low half first: element order.
fn halves(words: &[u32]) -> Vec<u16> {
    words
        .iter()
        .flat_map(|w| [(*w & 0xFFFF) as u16, (*w >> 16) as u16])
        .collect()
}

/// Whether a 16-bit word of C is the f32 word `f` rounded once: fp16 to nearest
/// even (subnormals kept, overflow to infinity) or bf16 as cast_f32_bf16 rounds;
/// for a NaN, an fp16 NaN of any sign and payload, or cast_f32_bf16's own word.
fn rounded_once(h: u16, f: u32, elem: Nf4Elem) -> bool {
    let x = f32::from_bits(f);
    match elem {
        Nf4Elem::Bf16 => h == f32_to_bf16(x),
        Nf4Elem::F16 if x.is_nan() => h & 0x7C00 == 0x7C00 && h & 0x03FF != 0,
        Nf4Elem::F16 => h == f32_to_f16(x),
    }
}

/// One routed product checked against its decomposition: every group's rows of
/// C against K1's decode of its weight followed by the kit's dense product.
/// `experts` lists (expert, rows) in dispatch order; each weight is `mpe`
/// consecutive members, `n x k` (b_t) or `k x n`.
#[allow(clippy::too_many_arguments)]
fn check(
    d: &Device,
    what: &str,
    g: &Group,
    experts: &[(usize, usize)],
    mpe: usize,
    (n, k): (usize, usize),
    b_t: bool,
    elem: Nf4Elem,
    a_scale: f32,
) -> (usize, usize) {
    let dev = d.0;
    let rows: usize = experts.iter().map(|e| e.1).sum();
    let max_rows = experts.iter().map(|e| e.1).max().unwrap();
    assert_eq!(
        mpe * g.member_values,
        n * k,
        "{what}: a weight is its members"
    );
    // A: rows x k in the element type, a spread of magnitudes (times a_scale) and exact zeros
    let a_vals: Vec<f32> = (0..rows * k)
        .map(|i| {
            let h = (i as u32).wrapping_mul(2_246_822_519) ^ 0xA11CE;
            if h % 53 == 0 {
                0.0
            } else {
                ((h % 20_011) as f32 / 10_000.0 - 1.0)
                    * [1.0f32, 0.25, 3.0][(h >> 30) as usize % 3]
                    * a_scale
            }
        })
        .collect();
    let a16: Vec<u16> = a_vals
        .iter()
        .map(|&x| match elem {
            Nf4Elem::Bf16 => f32_to_bf16(x),
            Nf4Elem::F16 => f32_to_f16(x),
        })
        .collect();
    let src = stored(d, g);
    let a = d.upload(&le(&a16, u16::to_le_bytes));
    let mut table = vec![];
    let mut at = 0usize;
    for &(e, r) in experts {
        let base = g.starts[e * mpe] as usize;
        table.extend([at as u32, r as u32, base as u32]);
        at += r;
    }
    let groups = d.upload(&le(&table, u32::to_le_bytes));
    let c = d.upload(&le(&vec![POISON; rows * n], u32::to_le_bytes));
    let members = g.starts.len();
    let rc = matmul_nf4(
        d,
        a,
        &src,
        groups,
        c,
        (rows, max_rows, n, k),
        b_t,
        (experts.len(), members, g.base),
        elem as u32,
    );
    assert_eq!(rc, ffi::ACK_OK, "{what} {elem:?}: {}", last_error());
    let got = d.words(c, rows * n);
    // the same call with a 16-bit C, poisoned first: every word the f32 C's rounded once
    assert_eq!(n % 2, 0, "{what}: a 16-bit C needs an even N");
    let c16 = d.upload(&le(&vec![POISON; rows * n / 2], u32::to_le_bytes));
    let rc = matmul_nf4_16(
        d,
        a,
        &src,
        groups,
        c16,
        (rows, max_rows, n, k),
        b_t,
        (experts.len(), members, g.base),
        elem as u32,
    );
    let got16 = if rc == ffi::ACK_ERR_UNSUPPORTED_OP {
        eprintln!(
            "UNMEASURED here: nf4-matmul 16-bit C {elem:?}: {}",
            last_error()
        );
        None
    } else {
        assert_eq!(
            rc,
            ffi::ACK_OK,
            "{what} {elem:?} 16-bit C: {}",
            last_error()
        );
        let h = halves(&d.words(c16, rows * n / 2));
        let bad: Vec<usize> = (0..rows * n)
            .filter(|&i| !rounded_once(h[i], got[i], elem))
            .collect();
        assert!(
            bad.is_empty(),
            "{what} {elem:?} 16-bit C: {} of {} words are not the f32 C rounded once; first {:?}",
            bad.len(),
            rows * n,
            bad.first().map(|&i| (i, h[i], f32::from_bits(got[i])))
        );
        Some(h)
    };
    d.free(&[c16]);
    // the decomposition, group by group
    let (mut checked, mut checked16, mut at) = (0usize, 0usize, 0usize);
    for &(e, r) in experts {
        if r == 0 {
            continue;
        }
        let ids: Vec<i64> = (0..mpe).map(|j| (e * mpe + j) as i64).collect();
        let idb = d.upload(&le(&ids, i64::to_le_bytes));
        let w16 = d.upload(&vec![0u8; n * k * 2]);
        let storage = match elem {
            Nf4Elem::Bf16 => 1,
            Nf4Elem::F16 => 2,
        };
        let rc = unsafe {
            ffi::ack_nf4_decode(
                dev,
                src[0],
                src[1],
                src[2],
                src[3],
                src[4],
                src[5],
                idb,
                w16,
                (n * k) as u32,
                members as u32,
                mpe as u32,
                g.member_values as u32,
                g.base as u32,
                storage,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, ffi::ACK_OK, "{what}: decode {}", last_error());
        // the group's rows from the host's own copy of A: ack_download takes a buffer whole, never a prefix
        let rows_z = d.upload(&le(&a16[at * k..(at + r) * k], u16::to_le_bytes));
        let c_ref = d.upload(&vec![0u8; r * n * 4]);
        let product = match elem {
            Nf4Elem::Bf16 => ffi::ack_matmul_bf16,
            Nf4Elem::F16 => ffi::ack_matmul_f16,
        };
        let rc = unsafe {
            product(
                dev,
                rows_z,
                w16,
                c_ref,
                r as u32,
                n as u32,
                k as u32,
                0,
                i32::from(b_t),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, ffi::ACK_OK, "{what}: dense product {}", last_error());
        let want = d.words(c_ref, r * n);
        let mine = &got[at * n..(at + r) * n];
        let same =
            |x: u32, y: u32| x == y || (f32::from_bits(x).is_nan() && f32::from_bits(y).is_nan());
        let bad = mine
            .iter()
            .zip(&want)
            .filter(|(x, y)| !same(**x, **y))
            .count();
        let first = mine.iter().zip(&want).position(|(x, y)| !same(*x, *y));
        assert_eq!(
            bad, 0,
            "{what} {elem:?} expert {e} ({r} rows): {bad} of {} words differ from decode + product; first {:?}",
            r * n,
            first.map(|i| (i, f32::from_bits(mine[i]), f32::from_bits(want[i])))
        );
        checked += r * n;
        // the 16-bit C's rows: the kit's dense 16-bit-output product of the same operands, word for word
        if let Some(h) = &got16 {
            let product16 = match elem {
                Nf4Elem::Bf16 => ffi::ack_matmul_bf16_to_bf16,
                Nf4Elem::F16 => ffi::ack_matmul_f16_to_f16,
            };
            let c16_ref = d.upload(&vec![0u8; r * n * 2]);
            let rc = unsafe {
                product16(
                    dev,
                    rows_z,
                    w16,
                    c16_ref,
                    r as u32,
                    n as u32,
                    k as u32,
                    0,
                    i32::from(b_t),
                    std::ptr::null_mut(),
                )
            };
            assert_eq!(
                rc,
                ffi::ACK_OK,
                "{what}: dense 16-bit product {}",
                last_error()
            );
            let want16 = halves(&d.words(c16_ref, r * n / 2));
            let mine16 = &h[at * n..(at + r) * n];
            let nan16 = |x: u16| match elem {
                Nf4Elem::Bf16 => x & 0x7F80 == 0x7F80 && x & 0x007F != 0,
                Nf4Elem::F16 => x & 0x7C00 == 0x7C00 && x & 0x03FF != 0,
            };
            let bad = mine16
                .iter()
                .zip(&want16)
                .filter(|(x, y)| x != y && !(nan16(**x) && nan16(**y)))
                .count();
            assert_eq!(
                bad,
                0,
                "{what} {elem:?} expert {e} ({r} rows): {bad} of {} 16-bit words differ from decode + the 16-bit product",
                r * n
            );
            checked16 += r * n;
            d.free(&[c16_ref]);
        }
        at += r;
        d.free(&[idb, w16, rows_z, c_ref]);
    }
    d.free(&[a, groups, c]);
    d.free(&src);
    (checked, checked16)
}

fn elems(dev: *mut ffi::AckDevice) -> Vec<Nf4Elem> {
    // an element type the device has no cooperative matrices for is UNMEASURED by name
    let d = Device(dev);
    let mut out = vec![];
    for elem in [Nf4Elem::F16, Nf4Elem::Bf16] {
        let g = Group::new(1, NESTED_VALUES as usize, 0, 1);
        let src = stored(&d, &g);
        let a = d.upload(&[0u8; 256 * 2]);
        let groups = d.upload(&le(&[0u32, 1, 0], u32::to_le_bytes));
        let c = d.upload(&[0u8; 64 * 4]);
        let rc = matmul_nf4(
            &d,
            a,
            &src,
            groups,
            c,
            (1, 1, 64, 256),
            true,
            (1, 1, 0),
            elem as u32,
        );
        if rc == ffi::ACK_ERR_UNSUPPORTED_OP {
            eprintln!("UNMEASURED here: nf4-matmul {elem:?}: {}", last_error());
        } else {
            assert_eq!(rc, ffi::ACK_OK, "{elem:?}: {}", last_error());
            out.push(elem);
        }
        d.free(&[a, groups, c]);
        d.free(&src);
    }
    out
}

#[test]
fn every_group_is_decode_then_the_kits_product_bit_for_bit() {
    let Some(dev) = open_or_unmeasured("nf4-matmul") else {
        return;
    };
    let d = Device(dev);
    let nv = NESTED_VALUES as usize;
    let (mut checked, mut checked16) = (0usize, 0usize);
    let mut tally = |(w, h): (usize, usize)| {
        checked += w;
        checked16 += h;
    };
    for elem in elems(dev) {
        // the routed gate_up call: four experts' stacked weights, 2 members of 64 x 256 each
        // (W = 128 x 256), experts 2, 0 and 3 routed with 5, 131 and 1 rows, expert 1 with none,
        // the source one nested block into its buffers
        let gu = Group::new(8, nv, 1, 0xC0DE);
        tally(check(
            &d,
            "gate_up",
            &gu,
            &[(2, 5), (1, 0), (0, 131), (3, 1)],
            2,
            (128, 256),
            true,
            elem,
            1.0,
        ));
        // the backward's rows @ W over the same weights read as 256 x 128 (k x n)
        tally(check(
            &d,
            "gate_up backward",
            &gu,
            &[(3, 70), (0, 1)],
            2,
            (128, 256),
            false,
            elem,
            1.0,
        ));
        // an attention-sized weight, 512 x 2,048 (64 nested blocks), at decode and at a ragged row count
        let attn = Group::new(1, 64 * nv, 0, 0xA77E);
        tally(check(
            &d,
            "attention decode",
            &attn,
            &[(0, 1)],
            1,
            (512, 2048),
            true,
            elem,
            1.0,
        ));
        tally(check(
            &d,
            "attention ragged",
            &attn,
            &[(0, 300)],
            1,
            (512, 2048),
            true,
            elem,
            1.0,
        ));
        tally(check(
            &d,
            "attention backward",
            &attn,
            &[(0, 17)],
            1,
            (2048, 512),
            false,
            elem,
            1.0,
        ));
    }
    // fp16 SUBNORMALS (2026-09-30): every absmax below fp16's smallest normal (no offset, absmax2 at most
    // 4.4e-5), so each nonzero weight decodes to a subnormal, times rows scaled into the subnormals too; two
    // members a weight, forward and transposed. The staging's rounding of subnormal weights, and the cooperative
    // matrices' reading of subnormal operands, must match the dense reference's; a build that flushes subnormals
    // or rounds other than to nearest even writes other words here
    if elems(dev).contains(&Nf4Elem::F16) {
        let mut tiny = Group::new(8, nv, 1, 0xDE41);
        for (i, v) in tiny.absmax2.iter_mut().enumerate() {
            *v = 2.0e-6 + (i % 7) as f32 * 7.0e-6;
        }
        tiny.offsets.fill(0.0);
        for (what, experts, b_t) in [
            ("fp16 subnormals", vec![(2usize, 5usize), (0, 131)], true),
            ("fp16 subnormals backward", vec![(3, 70), (1, 1)], false),
        ] {
            tally(check(
                &d,
                what,
                &tiny,
                &experts,
                2,
                (128, 256),
                b_t,
                Nf4Elem::F16,
                1.0e-5,
            ));
        }
    }
    eprintln!(
        "nf4-matmul: {checked} words bit-identical to nf4-decode + the kit's product; \
         {checked16} 16-bit words the f32 C rounded once and the kit's 16-bit-output product"
    );
    assert!(checked > 0 || std::env::var_os("ACK_REQUIRE_DEVICE").is_none());
    assert_eq!(ffi::ack_close(dev), ffi::ACK_OK, "{}", last_error());
}

#[test]
fn a_groups_table_the_call_does_not_cover_writes_nothing_for_that_group() {
    let Some(dev) = open_or_unmeasured("nf4-matmul guards") else {
        return;
    };
    let d = Device(dev);
    let nv = NESTED_VALUES as usize;
    let g = Group::new(2, nv, 0, 0x6A);
    let src = stored(&d, &g);
    let (n, k) = (64usize, 256usize);
    let a = d.upload(&vec![0x3Cu8; 12 * k * 2]);
    // group 0 is honest (rows 0-3, member 0); group 1 runs past the 12 rows; group 2 points past the
    // source; group 3's weight base is off a packed word (4); group 4 has 5 rows, more than the call's 4
    #[rustfmt::skip]
    let table = [
        0u32, 4, 0,
        10, 4, nv as u32,
        4, 4, (2 * nv) as u32,
        4, 4, 4,
        4, 5, 0,
    ];
    let groups = d.upload(&le(&table, u32::to_le_bytes));
    // the f32 C, then the 16-bit C (2026-10-01), whose rows are half as many words
    for out16 in [false, true] {
        let row_words = if out16 { n / 2 } else { n };
        let c = d.upload(&le(&vec![POISON; 12 * row_words], u32::to_le_bytes));
        let call = if out16 { matmul_nf4_16 } else { matmul_nf4 };
        let rc = call(
            &d,
            a,
            &src,
            groups,
            c,
            (12, 4, n, k),
            true,
            (5, 2, 0),
            Nf4Elem::F16 as u32,
        );
        if rc == ffi::ACK_ERR_UNSUPPORTED_OP {
            eprintln!(
                "UNMEASURED here: nf4-matmul fp16 (16-bit C: {out16}): {}",
                last_error()
            );
            d.free(&[c]);
            continue;
        }
        assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
        let got = d.words(c, 12 * row_words);
        assert!(
            got[..4 * row_words].iter().all(|&w| w != POISON),
            "the honest group wrote its rows (16-bit C: {out16})"
        );
        assert!(
            got[4 * row_words..].iter().all(|&w| w == POISON),
            "the groups the call does not cover wrote nothing (16-bit C: {out16})"
        );
        d.free(&[c]);
    }
    d.free(&[a, groups]);
    d.free(&src);
    assert_eq!(ffi::ack_close(dev), ffi::ACK_OK, "{}", last_error());
}

#[test]
fn a_call_the_family_does_not_address_is_refused_by_name() {
    let Some(dev) = open_or_unmeasured("nf4-matmul refusals") else {
        return;
    };
    let d = Device(dev);
    let nv = NESTED_VALUES as usize;
    let g = Group::new(2, nv, 0, 0x51);
    let src = stored(&d, &g);
    let (n, k) = (64usize, 256usize);
    let a = d.upload(&vec![0u8; 8 * k * 2]);
    let groups = d.upload(&le(&[0u32, 8, 0], u32::to_le_bytes));
    let c = d.upload(&vec![0u8; 8 * n * 4]);
    let f16 = Nf4Elem::F16 as u32;
    let refused = |rc: i32, want: i32, name: &str| {
        assert_eq!(rc, want, "{name}: {}", last_error());
        assert!(last_error().contains(name), "{name}: {}", last_error());
    };
    let call =
        |a: *const Buffer, groups: *const Buffer, c: *const Buffer, shape, b_t, grp, elem| {
            matmul_nf4(&d, a, &src, groups, c, shape, b_t, grp, elem)
        };
    // the positive control first: the same buffers carry a call that fits them
    let rc = call(a, groups, c, (8, 8, n, k), true, (1, 2, 0), f16);
    if rc == ffi::ACK_ERR_UNSUPPORTED_OP {
        eprintln!("UNMEASURED here: nf4-matmul fp16: {}", last_error());
        return;
    }
    assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
    refused(
        call(a, groups, c, (8, 8, n, k), true, (1, 2, 0), 2),
        ffi::ACK_ERR_SHAPE,
        "nf4-matmul-elem-out-of-range",
    );
    refused(
        call(a, groups, c, (0, 1, n, k), true, (1, 2, 0), f16),
        ffi::ACK_ERR_SHAPE,
        "nf4-matmul-zero-extent",
    );
    refused(
        call(a, groups, c, (8, 8, n, k), true, (0, 2, 0), f16),
        ffi::ACK_ERR_SHAPE,
        "nf4-matmul-groups-out-of-range",
    );
    refused(
        call(a, groups, c, (8, 9, n, k), true, (1, 2, 0), f16),
        ffi::ACK_ERR_SHAPE,
        "nf4-matmul-rows-out-of-range",
    );
    refused(
        call(a, groups, c, (8, 8, 65, 252), true, (1, 2, 0), f16),
        ffi::ACK_ERR_SHAPE,
        "nf4-matmul-row-not-whole-words",
    );
    refused(
        call(a, groups, c, (8, 8, n, k), true, (1, 0, 0), f16),
        ffi::ACK_ERR_SHAPE,
        "nf4-members-out-of-range",
    );
    refused(
        call(a, groups, c, (8, 8, n, k), true, (1, 2, 64), f16),
        ffi::ACK_ERR_SHAPE,
        "nf4-base-out-of-range",
    );
    refused(
        call(a, groups, c, (8, 8, 256, 256), true, (1, 2, 0), f16),
        ffi::ACK_ERR_SIZE,
        "nf4-source-too-small",
    );
    refused(
        call(a, groups, c, (9, 8, n, k), true, (1, 2, 0), f16),
        ffi::ACK_ERR_SIZE,
        "nf4-matmul-a-too-small",
    );
    refused(
        call(a, groups, c, (8, 8, n, k), true, (2, 2, 0), f16),
        ffi::ACK_ERR_SIZE,
        "nf4-matmul-groups-too-small",
    );
    refused(
        call(a, groups, c, (8, 8, 128, k), true, (1, 2, 0), f16),
        ffi::ACK_ERR_SIZE,
        "nf4-matmul-output-too-small",
    );
    refused(
        call(a, groups, a, (8, 8, n, 64), true, (1, 2, 0), f16),
        ffi::ACK_ERR_SHAPE,
        "nf4-matmul-output-aliases-input",
    );
    // the 16-bit C (2026-10-01): a C of exactly 8 x 64 16-bit words carries the call; an odd N is refused
    // before any device lookup, a C one word short is too small, and C cannot be A
    let c16 = d.upload(&vec![0u8; 8 * n * 2]);
    let short16 = d.upload(&vec![0u8; 8 * n * 2 - 4]);
    let call16 = |a: *const Buffer, c: *const Buffer, shape, elem| {
        matmul_nf4_16(&d, a, &src, groups, c, shape, true, (1, 2, 0), elem)
    };
    let rc = call16(a, c16, (8, 8, n, k), f16);
    if rc == ffi::ACK_ERR_UNSUPPORTED_OP {
        eprintln!(
            "UNMEASURED here: nf4-matmul fp16 16-bit C: {}",
            last_error()
        );
    } else {
        assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
        refused(
            call16(a, c16, (8, 8, 63, k), f16),
            ffi::ACK_ERR_SHAPE,
            "nf4-matmul-out16-odd-n",
        );
        refused(
            call16(a, short16, (8, 8, n, k), f16),
            ffi::ACK_ERR_SIZE,
            "nf4-matmul-output-too-small",
        );
        refused(
            call16(a, a, (8, 8, n, 64), f16),
            ffi::ACK_ERR_SHAPE,
            "nf4-matmul-output-aliases-input",
        );
        refused(
            call16(a, c16, (8, 8, n, k), 2),
            ffi::ACK_ERR_SHAPE,
            "nf4-matmul-elem-out-of-range",
        );
    }
    // the odd N is refused by name before any device lookup: a null device does not change the answer
    let rc = unsafe {
        ffi::ack_matmul_nf4_to_16(
            std::ptr::null(),
            a,
            src[0],
            src[1],
            src[2],
            src[3],
            src[4],
            src[5],
            groups,
            c16,
            8,
            8,
            63,
            k as u32,
            1,
            0,
            1,
            2,
            0,
            f16,
            std::ptr::null_mut(),
        )
    };
    refused(rc, ffi::ACK_ERR_SHAPE, "nf4-matmul-out16-odd-n");
    d.free(&[a, groups, c, c16, short16]);
    d.free(&src);
    assert_eq!(ffi::ack_close(dev), ffi::ACK_OK, "{}", last_error());
}
