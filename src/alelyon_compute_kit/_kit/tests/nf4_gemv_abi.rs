//! `ack_gemv_nf4` (nf4-gemv/v1, 2026-09-30): the NF4 product at one to 16 rows
//! a group, as decoding runs it, on a real device.
//!
//! THE LAW is the kernel header's documented order, so this test computes every
//! word of C on the host and compares bit for bit. Each weight is the word
//! `ack_nf4_decode` writes for it from the same stored form on the same device
//! (nf4-decode/v1, itself bitsandbytes' decode bit for bit), widened exactly;
//! each product is exact in f32; invocation l sums its packed words j = l,
//! l + 64, ... in ascending order, eight values each, in IEEE single precision;
//! then the pairwise tree s = 32, 16, ..., 1 over the 64 sums. C is poisoned
//! first and the data holds no NaN, so the comparison is of bits, with no NaN
//! leniency: a word the kernel never wrote fails. Each word is also held within
//! that order's error bound of the float64 sum of the same products, so the
//! emulation is itself a sum of the right terms.
//!
//! Held: a grouped decode over four routed experts in a permuted order (one
//! group of no rows, one of 16), each weight two members with different offsets
//! (the switch falls mid-weight), with fewer packed words a column than
//! invocations and the source a nested block into its buffers; an
//! attention-sized weight (four words an invocation) at one row and at 16 rows
//! with a pitch wider than K (the pad holds NaN, so a read of it fails); a
//! weight of more columns than one grid dimension holds (folded into y); the
//! kernel's guards against a groups table the call does not cover; and the
//! refusals by name, with a positive control. The calls between them run each
//! of the kernel's modules (1, 4 and 16 rows a group, 2026-09-30), and a weight
//! whose columns straddle a member boundary takes the per-word member lookup
//! that a column inside one member skips.
//!
//! The kernel needs no cooperative matrices, no 16-bit arithmetic and no
//! subgroup size, so both element types run on every device. For fp16 a device
//! reporting shaderFloat16 and the float controls takes the modules that
//! convert with its own instructions (F16_HW, 2026-09-30), and
//! ACK_NF4_GEMV_F16_HW=0 keeps convert/v1's integer modules: run this test both
//! ways, since both are held to the same words (no NaN is involved here). The
//! fp16 subnormal case (weights and activations below fp16's smallest normal,
//! at 1, 3 and 16 rows) is where a build that flushes subnormals or rounds other
//! than to nearest even writes other words. Skips as UNMEASURED without a device
//! unless ACK_REQUIRE_DEVICE is set.
use alelyon_compute_kit::ffi;
use alelyon_compute_kit::nf4_ops::{Nf4Elem, GEMV_LANES, NESTED_VALUES, TABLE_ENTRIES};
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

/// f16 -> f32, exact (subnormals, infinities and NaN included).
fn f16_to_f32(h: u16) -> f32 {
    let h = u32::from(h);
    let sign = (h & 0x8000) << 16;
    let e = (h >> 10) & 0x1F;
    let m = h & 0x3FF;
    let bits = if e == 0x1F {
        sign | 0x7F80_0000 | (m << 13)
    } else if e == 0 {
        if m == 0 {
            sign
        } else {
            let t = 31 - m.leading_zeros();
            sign | ((t + 103) << 23) | ((m << (23 - t)) & 0x7F_FFFF)
        }
    } else {
        sign | ((e + 112) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}

fn widen(elem: Nf4Elem, h: u16) -> f32 {
    match elem {
        Nf4Elem::Bf16 => f32::from_bits(u32::from(h) << 16),
        Nf4Elem::F16 => f16_to_f32(h),
    }
}

fn narrow(elem: Nf4Elem, x: f32) -> u16 {
    match elem {
        Nf4Elem::Bf16 => f32_to_bf16(x),
        Nf4Elem::F16 => f32_to_f16(x),
    }
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
    fn halves(&self, buf: *const Buffer, count: usize) -> Vec<u16> {
        let mut out = vec![0u8; count.div_ceil(2) * 4];
        let rc = unsafe { ffi::ack_download(self.0, buf, out.as_mut_ptr(), out.len()) };
        assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
        out.chunks_exact(2)
            .take(count)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect()
    }
    fn free(&self, bufs: &[*mut Buffer]) {
        for &b in bufs {
            assert_eq!(ffi::ack_buffer_free(self.0, b), ffi::ACK_OK);
        }
    }
}

/// A source of `members` NF4 members of `member_values` each (a whole number of
/// nested blocks), back to back, `base_blocks` nested blocks into its buffers
/// (whose leading values are filler the kernel must skip).
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
    fn values(&self) -> usize {
        self.starts.len() * self.member_values
    }
}

fn tables() -> Vec<f32> {
    let mut t: Vec<f32> = NF4_BITS.iter().map(|b| f32::from_bits(*b)).collect();
    t.extend((0..256).map(|i| (i as f32 - 127.5) / 128.0));
    assert_eq!(t.len(), TABLE_ENTRIES);
    t
}

/// The source's stored form on the device: [packed, absmax_q, absmax2, tables, starts, offsets].
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

/// The whole source as nf4-decode/v1 writes it in `elem`, value by value.
fn decoded(d: &Device, g: &Group, src: &[*mut Buffer; 6], elem: Nf4Elem) -> Vec<u16> {
    let n = g.values();
    let out = d.upload(&vec![0u8; n * 2]);
    let storage = match elem {
        Nf4Elem::Bf16 => 1,
        Nf4Elem::F16 => 2,
    };
    let rc = unsafe {
        ffi::ack_nf4_decode(
            d.0,
            src[0],
            src[1],
            src[2],
            src[3],
            src[4],
            src[5],
            std::ptr::null(),
            out,
            n as u32,
            g.starts.len() as u32,
            0,
            0,
            g.base as u32,
            storage,
            std::ptr::null_mut(),
        )
    };
    assert_eq!(rc, ffi::ACK_OK, "decode: {}", last_error());
    let w = d.halves(out, n);
    d.free(&[out]);
    w
}

#[allow(clippy::too_many_arguments)]
fn gemv_nf4(
    d: &Device,
    a: *const Buffer,
    src: &[*mut Buffer; 6],
    groups: *const Buffer,
    c: *const Buffer,
    (rows, max_rows, n, k, lda): (usize, usize, usize, usize, usize),
    (ngroups, members, base): (usize, usize, usize),
    elem: u32,
) -> i32 {
    unsafe {
        ffi::ack_gemv_nf4(
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
            lda as u32,
            ngroups as u32,
            members as u32,
            base as u32,
            elem,
            std::ptr::null_mut(),
        )
    }
}

/// The kernel header's order for one column of one group: row m's value is the
/// 64 invocations' ascending sums over their packed words, then the tree. `a`
/// holds the group's rows at pitch `lda`; `w` the column's K weights.
fn documented_order(elem: Nf4Elem, a: &[u16], lda: usize, rows: usize, w: &[u16]) -> Vec<f32> {
    let lanes = GEMV_LANES as usize;
    let k = w.len();
    let words = k / 8;
    let wf: Vec<f32> = w.iter().map(|&h| widen(elem, h)).collect();
    (0..rows)
        .map(|m| {
            let row = &a[m * lda..m * lda + k];
            let mut partial: Vec<f32> = (0..lanes)
                .map(|lane| {
                    let mut acc = 0.0f32;
                    for j in (lane..words).step_by(lanes) {
                        for e in 0..8 {
                            // exact in f32 for these operands: no rounding before the sum's
                            acc += widen(elem, row[8 * j + e]) * wf[8 * j + e];
                        }
                    }
                    acc
                })
                .collect();
            let mut s = lanes / 2;
            while s > 0 {
                for l in 0..s {
                    partial[l] += partial[l + s];
                }
                s /= 2;
            }
            partial[0]
        })
        .collect()
}

/// A grouped decode checked against the documented order: `experts` lists
/// (expert, rows) in dispatch order; each weight is `mpe` consecutive members,
/// `n x k` stored. Returns the words compared.
#[allow(clippy::too_many_arguments)]
fn check(
    d: &Device,
    what: &str,
    g: &Group,
    experts: &[(usize, usize)],
    mpe: usize,
    (n, k, lda): (usize, usize, usize),
    elem: Nf4Elem,
    a_scale: f32,
) -> usize {
    let rows: usize = experts.iter().map(|e| e.1).sum();
    let max_rows = experts.iter().map(|e| e.1).max().unwrap();
    assert!(
        n * k <= mpe * g.member_values,
        "{what}: a weight lies in its members"
    );
    let src = stored(d, g);
    let w16 = decoded(d, g, &src, elem);
    // A: rows x k at pitch lda in the element type, a spread of magnitudes (times a_scale) and exact zeros; the
    // pad is NaN
    let nan = match elem {
        Nf4Elem::Bf16 => 0x7FC0u16,
        Nf4Elem::F16 => 0x7E00u16,
    };
    let a16: Vec<u16> = (0..rows * lda)
        .map(|i| {
            if i % lda >= k {
                return nan;
            }
            let h = (i as u32).wrapping_mul(2_246_822_519) ^ 0xA11CE;
            if h % 53 == 0 {
                0
            } else {
                let x = ((h % 20_011) as f32 / 10_000.0 - 1.0)
                    * [1.0f32, 0.25, 3.0][(h >> 30) as usize % 3];
                narrow(elem, x * a_scale)
            }
        })
        .collect();
    let a = d.upload(&le(&a16, u16::to_le_bytes));
    let mut table = vec![];
    let mut at = 0usize;
    for &(e, r) in experts {
        table.extend([at as u32, r as u32, g.starts[e * mpe] as u32]);
        at += r;
    }
    let groups = d.upload(&le(&table, u32::to_le_bytes));
    let c = d.upload(&le(&vec![POISON; rows * n], u32::to_le_bytes));
    let rc = gemv_nf4(
        d,
        a,
        &src,
        groups,
        c,
        (rows, max_rows, n, k, lda),
        (experts.len(), g.starts.len(), g.base),
        elem as u32,
    );
    assert_eq!(rc, ffi::ACK_OK, "{what} {elem:?}: {}", last_error());
    let got = d.words(c, rows * n);
    // the order's bound on |fl(sum) - sum|: the longest chain of roundings is 8 a word an invocation plus the tree
    let lanes = GEMV_LANES as usize;
    let depth = 8 * (k / 8).div_ceil(lanes) + lanes.trailing_zeros() as usize;
    let u = f64::EPSILON * f64::from(1u32 << 28); // 2^-24
    let gamma = depth as f64 * u / (1.0 - depth as f64 * u);
    let (mut checked, mut at) = (0usize, 0usize);
    for &(e, r) in experts {
        let w_base = g.starts[e * mpe] as usize;
        let a_rows = &a16[at * lda..(at + r) * lda];
        for col in 0..n {
            let w = &w16[w_base + col * k..w_base + (col + 1) * k];
            let want = documented_order(elem, a_rows, lda, r, w);
            for (m, value) in want.iter().enumerate() {
                let word = got[(at + m) * n + col];
                assert!(value.is_finite(), "{what}: the data is finite");
                assert_eq!(
                    word,
                    value.to_bits(),
                    "{what} {elem:?} expert {e} row {m} column {col}: {} from the device, {value} by the documented order",
                    f32::from_bits(word)
                );
                let (mut sum, mut magnitude) = (0.0f64, 0.0f64);
                for i in 0..k {
                    let p =
                        f64::from(widen(elem, a_rows[m * lda + i])) * f64::from(widen(elem, w[i]));
                    sum += p;
                    magnitude += p.abs();
                }
                let bound = gamma * magnitude + 4.0 * k as f64 * f64::EPSILON * magnitude;
                assert!(
                    (f64::from(*value) - sum).abs() <= bound,
                    "{what} {elem:?} expert {e} row {m} column {col}: {value} is {} from the float64 sum {sum}, past the order's bound {bound}",
                    (f64::from(*value) - sum).abs()
                );
            }
        }
        checked += r * n;
        at += r;
    }
    d.free(&[a, groups, c]);
    d.free(&src);
    checked
}

#[test]
fn every_word_follows_the_documented_order_bit_for_bit() {
    let Some(dev) = open_or_unmeasured("nf4-gemv") else {
        return;
    };
    let d = Device(dev);
    let nv = NESTED_VALUES as usize;
    let mut checked = 0usize;
    for elem in [Nf4Elem::F16, Nf4Elem::Bf16] {
        // decoding through four routed experts' gate_up weights, 2 members of 64 x 256 each (W = 128 x 256,
        // 32 packed words a column: fewer than the invocations), experts 2, 0 and 3 with 5, 16 and 1 rows,
        // expert 1 with none, the source one nested block into its buffers
        let gu = Group::new(8, nv, 1, 0xC0DE);
        checked += check(
            &d,
            "gate_up",
            &gu,
            &[(2, 5), (1, 0), (0, 16), (3, 1)],
            2,
            (128, 256, 256),
            elem,
            1.0,
        );
        // an attention-sized weight, 512 x 2,048 (four words an invocation), at one row and at 16 rows
        // whose pitch is wider than K
        let attn = Group::new(1, 64 * nv, 0, 0xA77E);
        checked += check(
            &d,
            "attention decode",
            &attn,
            &[(0, 1)],
            1,
            (512, 2048, 2048),
            elem,
            1.0,
        );
        checked += check(
            &d,
            "attention 16 rows",
            &attn,
            &[(0, 16)],
            1,
            (512, 2048, 2056),
            elem,
            1.0,
        );
        checked += check(
            &d,
            "attention 3 rows",
            &attn,
            &[(0, 3)],
            1,
            (512, 2048, 2048),
            elem,
            1.0,
        );
        // columns of 24,576 values over members of 16,384, each member with its own offset: every column
        // straddles a member boundary, so its members differ and each word looks its own up
        let straddle = Group::new(3, nv, 0, 0x57AD);
        checked += check(
            &d,
            "straddling members",
            &straddle,
            &[(0, 2)],
            3,
            (2, 3 * nv / 2, 3 * nv / 2),
            elem,
            1.0,
        );
    }
    // more columns than one grid dimension holds: 70,000 x 64, folded into y (the weight is a prefix of
    // its member)
    let wide = Group::new(1, 274 * nv, 0, 0x71DE);
    checked += check(
        &d,
        "wide",
        &wide,
        &[(0, 2)],
        1,
        (70_000, 64, 64),
        Nf4Elem::F16,
        1.0,
    );
    // fp16 SUBNORMALS (2026-09-30), weights and activations alike: every absmax below fp16's smallest normal
    // (no offset, absmax2 at most 4.4e-5), so each nonzero weight rounds to a subnormal, times activations scaled
    // into the subnormals too, at 1, 3 and 16 rows (each of the three modules). Every product of two fp16
    // subnormals is an exact normal f32 value, so the law holds on every device. The fp16 modules that convert
    // with the device's own instructions keep subnormals only under their declared DenormPreserve: a build or a
    // device that flushes them writes other words here, and so does any rounding but to nearest even (weights
    // near half a subnormal step round differently)
    let mut tiny = Group::new(1, 64 * nv, 0, 0xDE40);
    for (i, v) in tiny.absmax2.iter_mut().enumerate() {
        *v = 2.0e-6 + (i % 7) as f32 * 7.0e-6;
    }
    tiny.offsets.fill(0.0);
    for rows in [1, 3, 16] {
        checked += check(
            &d,
            "fp16 subnormals",
            &tiny,
            &[(0, rows)],
            1,
            (512, 2048, 2048),
            Nf4Elem::F16,
            1.0e-5,
        );
    }
    eprintln!("nf4-gemv: {checked} words bit-identical to the documented order");
    assert!(checked > 0);
    assert_eq!(ffi::ack_close(dev), ffi::ACK_OK, "{}", last_error());
}

#[test]
fn a_groups_table_the_call_does_not_cover_writes_nothing_for_that_group() {
    let Some(dev) = open_or_unmeasured("nf4-gemv guards") else {
        return;
    };
    let d = Device(dev);
    let nv = NESTED_VALUES as usize;
    let g = Group::new(2, nv, 0, 0x6A);
    let src = stored(&d, &g);
    let (n, k) = (64usize, 256usize);
    let a = d.upload(&vec![0x3Cu8; 24 * k * 2]);
    // group 0 is honest (rows 0-3, member 0); group 1 has 17 rows, more than the kernel holds; group 2
    // runs past the 24 rows; group 3 points past the source; group 4's weight base is off a packed word
    // (4); group 5 has no rows
    #[rustfmt::skip]
    let table = [
        0u32, 4, 0,
        4, 17, 0,
        21, 4, nv as u32,
        4, 4, (2 * nv) as u32,
        4, 4, 4,
        8, 0, 0,
    ];
    let groups = d.upload(&le(&table, u32::to_le_bytes));
    let c = d.upload(&le(&vec![POISON; 24 * n], u32::to_le_bytes));
    let rc = gemv_nf4(
        &d,
        a,
        &src,
        groups,
        c,
        (24, 16, n, k, k),
        (6, 2, 0),
        Nf4Elem::F16 as u32,
    );
    assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
    let got = d.words(c, 24 * n);
    assert!(
        got[..4 * n].iter().all(|&w| w != POISON),
        "the honest group wrote its rows"
    );
    assert!(
        got[4 * n..].iter().all(|&w| w == POISON),
        "the groups the call does not cover wrote nothing"
    );
    d.free(&[a, groups, c]);
    d.free(&src);
    assert_eq!(ffi::ack_close(dev), ffi::ACK_OK, "{}", last_error());
}

#[test]
fn a_call_the_family_does_not_address_is_refused_by_name() {
    let Some(dev) = open_or_unmeasured("nf4-gemv refusals") else {
        return;
    };
    let d = Device(dev);
    let nv = NESTED_VALUES as usize;
    let g = Group::new(2, nv, 0, 0x51);
    let src = stored(&d, &g);
    let (n, k) = (64usize, 256usize);
    let a = d.upload(&vec![0u8; 16 * k * 2]);
    let groups = d.upload(&le(&[0u32, 16, 0], u32::to_le_bytes));
    let c = d.upload(&vec![0u8; 16 * n * 4]);
    let f16 = Nf4Elem::F16 as u32;
    let refused = |rc: i32, want: i32, name: &str| {
        assert_eq!(rc, want, "{name}: {}", last_error());
        let message = last_error();
        assert!(
            message.contains(name) && message.starts_with("nf4-gemv"),
            "{name}: {message}"
        );
    };
    let call = |a: *const Buffer, groups: *const Buffer, c: *const Buffer, shape, grp, elem| {
        gemv_nf4(&d, a, &src, groups, c, shape, grp, elem)
    };
    // the positive control first: the same buffers carry a call that fits them
    assert_eq!(
        call(a, groups, c, (16, 16, n, k, k), (1, 2, 0), f16),
        ffi::ACK_OK,
        "{}",
        last_error()
    );
    refused(
        call(a, groups, c, (16, 16, n, k, k), (1, 2, 0), 2),
        ffi::ACK_ERR_SHAPE,
        "nf4-matmul-elem-out-of-range",
    );
    refused(
        call(a, groups, c, (17, 17, n, k, k), (1, 2, 0), f16),
        ffi::ACK_ERR_SHAPE,
        "nf4-gemv-rows-above-16",
    );
    refused(
        call(a, groups, c, (8, 8, n, k, k + 4), (1, 2, 0), f16),
        ffi::ACK_ERR_SHAPE,
        "nf4-gemv-pitch-not-whole-words",
    );
    refused(
        call(a, groups, c, (8, 8, n, k, k - 8), (1, 2, 0), f16),
        ffi::ACK_ERR_SHAPE,
        "nf4-matmul-pitch-below-width",
    );
    refused(
        call(a, groups, c, (0, 1, n, k, k), (1, 2, 0), f16),
        ffi::ACK_ERR_SHAPE,
        "nf4-matmul-zero-extent",
    );
    refused(
        call(a, groups, c, (16, 16, n, k, k), (0, 2, 0), f16),
        ffi::ACK_ERR_SHAPE,
        "nf4-matmul-groups-out-of-range",
    );
    refused(
        call(a, groups, c, (8, 9, n, k, k), (1, 2, 0), f16),
        ffi::ACK_ERR_SHAPE,
        "nf4-matmul-rows-out-of-range",
    );
    refused(
        call(a, groups, c, (8, 8, 65, 252, 256), (1, 2, 0), f16),
        ffi::ACK_ERR_SHAPE,
        "nf4-matmul-row-not-whole-words",
    );
    refused(
        call(a, groups, c, (16, 16, n, k, k), (1, 0, 0), f16),
        ffi::ACK_ERR_SHAPE,
        "nf4-members-out-of-range",
    );
    refused(
        call(a, groups, c, (16, 16, n, k, k), (1, 2, 64), f16),
        ffi::ACK_ERR_SHAPE,
        "nf4-base-out-of-range",
    );
    refused(
        call(a, groups, c, (16, 16, 256, 256, 256), (1, 2, 0), f16),
        ffi::ACK_ERR_SIZE,
        "nf4-source-too-small",
    );
    refused(
        call(a, groups, c, (17, 16, n, k, k), (1, 2, 0), f16),
        ffi::ACK_ERR_SIZE,
        "nf4-matmul-a-too-small",
    );
    refused(
        call(a, groups, c, (16, 16, n, k, k), (2, 2, 0), f16),
        ffi::ACK_ERR_SIZE,
        "nf4-matmul-groups-too-small",
    );
    refused(
        call(a, groups, c, (16, 16, 128, k, k), (1, 2, 0), f16),
        ffi::ACK_ERR_SIZE,
        "nf4-matmul-output-too-small",
    );
    refused(
        call(a, groups, a, (16, 16, n, k, k), (1, 2, 0), f16),
        ffi::ACK_ERR_SHAPE,
        "nf4-matmul-output-aliases-input",
    );
    d.free(&[a, groups, c]);
    d.free(&src);
    assert_eq!(ffi::ack_close(dev), ffi::ACK_OK, "{}", last_error());
}
