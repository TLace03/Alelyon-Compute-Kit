//! `ack_index` (ABI 12): gather, scatter and scatter-add on a real device,
//! through the C ABI, against host references.
//!
//! EXACT, NOT BOUNDED. A gather and a scatter move bits, so they are compared
//! bit for bit. A scatter-add walks each fiber in index order and rounds each
//! step in the element's own type, so it is compared bit for bit against a
//! host loop in that order: f32 in f32, bf16 and f16 through f32 and back with
//! round-to-nearest-even (the f16 reference finds the nearest value by
//! searching every finite fp16, so it shares no code with the shader's integer
//! rounding), int64 with its carries. For f32 and integers that loop is also
//! torch's CPU result; for bf16 and f16 torch's CPU kernels accumulate in f32
//! and round once (measured), which is why the torch adapter refuses those.
//!
//! Outputs are poisoned first (NaN words, or a byte pattern), so an element
//! the kernel never wrote cannot pass, and bytes past a sub-word output must
//! come back as they were. Every case also reads the status word.
//!
//! Since schema 2, sub-word scatters whose fibers share words (written with
//! integer atomics) are checked the same way, whole output against a host
//! loop, including 1,001 neighbouring fibers that run at once; and a float64
//! scatter-add is checked bit for bit against the host's IEEE add under
//! torch's measured NaN rule, over every pair of special values and long
//! accumulations across every exponent range.
//!
//! Since schema 3, the long fibers' element-parallel paths are checked the
//! same way: a scatter of 65,536 elements with repeated indices whose last
//! writer the plan elects, one fiber and several, whole-word and sub-word
//! storage, a scalar source, an index out of range; the int32 and int64
//! scatter-adds added with integer atomics, against the host's wrapping sums;
//! and the float and narrow scatter-adds grouped by destination with the
//! radix sort and added in the fiber's order, against host walks in that
//! order (float64 with torch's NaN rule, f32 with alpha, bf16, uint8).
//! Each case first asserts the plan's strategy, so a value test cannot pass
//! on the fiber walk it replaces.
//!
//! Without a device every test prints UNMEASURED (`ACK_REQUIRE_DEVICE=1`
//! turns that into a failure).
use alelyon_compute_kit::ffi;
use alelyon_compute_kit::index_ops::{
    Bindings, Dtype, IndexOp, IndexPlan, IndexRequest, IndexStrategy, Storage, View, FLAG_ALPHA,
    FLAG_I32_INDEX, FLAG_VALUE, FLAG_WRAP, STATUS_INDEX,
};
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

struct Device {
    dev: *mut ffi::AckDevice,
    live: Vec<Buf>,
}

impl Device {
    /// A buffer holding `bytes`, padded with `pad` to a whole word.
    fn upload(&mut self, bytes: &[u8], pad: u8) -> Buf {
        let mut data = bytes.to_vec();
        while data.len() % 4 != 0 || data.is_empty() {
            data.push(pad);
        }
        let buf = ffi::ack_buffer_alloc(self.dev, data.len() as u64);
        assert!(!buf.is_null(), "{}", last_error());
        let rc = unsafe { ffi::ack_upload(self.dev, buf, data.as_ptr(), data.len()) };
        assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
        self.live.push(buf);
        buf
    }

    fn download(&self, buf: Buf, len: usize) -> Vec<u8> {
        let mut out = vec![0u8; len];
        let rc = unsafe { ffi::ack_download(self.dev, buf, out.as_mut_ptr(), len) };
        assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
        out
    }

    /// Runs one request; returns the entry's code.
    fn run(&self, r: IndexRequest, src: Buf, index: Buf, out: Buf, status: Buf) -> i32 {
        let abi = to_abi(&r);
        let mut ms = 0.0f64;
        unsafe {
            ffi::ack_index(
                self.dev,
                src,
                index,
                out,
                status,
                &abi,
                std::mem::size_of::<ffi::AckIndexRequest>(),
                &mut ms,
            )
        }
    }
}

/// The C ABI's words for a request.
fn to_abi(r: &IndexRequest) -> ffi::AckIndexRequest {
    let mut flags = 0;
    if r.index_i32 {
        flags |= FLAG_I32_INDEX;
    }
    if r.wrap {
        flags |= FLAG_WRAP;
    }
    if r.value.is_some() {
        flags |= FLAG_VALUE;
    }
    if r.alpha.is_some() {
        flags |= FLAG_ALPHA;
    }
    ffi::AckIndexRequest {
        op: r.op as u32,
        dtype: r.dtype as u32,
        ndim: r.ndim as u32,
        dim: r.dim as u32,
        flags,
        status_slot: r.status_slot,
        shape: r.shape,
        index_offset: r.index.offset,
        index_strides: r.index.strides,
        src_offset: r.src.offset,
        src_strides: r.src.strides,
        out_offset: r.out.offset,
        out_strides: r.out.strides,
        bound: r.bound,
        value: r.value.unwrap_or(0),
        alpha_bits: r.alpha.unwrap_or(0),
        reserved: 0,
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        for buf in self.live.drain(..) {
            assert_eq!(ffi::ack_buffer_free(self.dev, buf), ffi::ACK_OK);
        }
        assert_eq!(ffi::ack_close(self.dev), ffi::ACK_OK, "{}", last_error());
    }
}

fn i64_bytes(values: &[i64]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn bf16_round(x: f32) -> u16 {
    if x.is_nan() {
        return 0x7FC0;
    }
    let b = x.to_bits();
    ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) as u16
}

fn bf16_value(h: u16) -> f32 {
    f32::from_bits(u32::from(h) << 16)
}

/// Every finite non-negative fp16 as (value, bits), ascending.
fn f16_table() -> Vec<(f64, u16)> {
    let mut table: Vec<(f64, u16)> = (0u16..0x7C00)
        .map(|h| {
            let e = i32::from((h >> 10) & 0x1F);
            let m = f64::from(h & 0x3FF);
            let v = if e == 0 {
                m * 2f64.powi(-24)
            } else {
                (1.0 + m / 1024.0) * 2f64.powi(e - 15)
            };
            (v, h)
        })
        .collect();
    table.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    table
}

fn f16_value(table: &[(f64, u16)], h: u16) -> f32 {
    let magnitude = h & 0x7FFF;
    let v = if magnitude >= 0x7C00 {
        if magnitude == 0x7C00 {
            f64::INFINITY
        } else {
            f64::NAN
        }
    } else {
        table
            .iter()
            .find(|&&(_, bits)| bits == magnitude)
            .unwrap()
            .0
    };
    (if h & 0x8000 != 0 { -v } else { v }) as f32
}

/// Round to the nearest fp16, ties to the even encoding, overflow to infinity
/// at 65520 and above: by search, not by bit manipulation.
fn f16_round(table: &[(f64, u16)], x: f32) -> u16 {
    if x.is_nan() {
        return if x.is_sign_negative() { 0xFE00 } else { 0x7E00 };
    }
    let sign = if x.is_sign_negative() { 0x8000 } else { 0 };
    let a = f64::from(x).abs();
    if a >= 65520.0 {
        return sign | 0x7C00;
    }
    let pos = table.partition_point(|&(v, _)| v < a);
    let hi = table[pos.min(table.len() - 1)];
    if pos == 0 || hi.0 == a {
        return sign | hi.1;
    }
    let lo = table[pos - 1];
    let pick = if a - lo.0 < hi.0 - a {
        lo
    } else if a - lo.0 > hi.0 - a {
        hi
    } else if lo.1 % 2 == 0 {
        lo
    } else {
        hi
    };
    sign | pick.1
}

fn request(op: IndexOp, dtype: Dtype, ndim: usize, shape: [u64; 4], dim: usize) -> IndexRequest {
    IndexRequest {
        op,
        dtype,
        ndim,
        shape,
        dim,
        index: View::new(0, [0; 4]),
        index_i32: false,
        src: View::new(0, [0; 4]),
        out: View::new(0, [0; 4]),
        bound: 1,
        wrap: false,
        value: None,
        alpha: None,
        status_slot: 0,
    }
}

#[test]
fn a_gather_copies_bits_for_every_element_size_and_keeps_the_bytes_past_its_output() {
    let Some(dev) = open_or_unmeasured("index gather") else {
        return;
    };
    let mut d = Device { dev, live: vec![] };
    // torch.gather(src [3, 7], 1, index [3, 5]): the output is 15 elements
    let (rows, cols, take) = (3usize, 7usize, 5usize);
    let index: Vec<i64> = (0..rows * take)
        .map(|i| ((i * 5 + 3) % cols) as i64)
        .collect();
    let idx = d.upload(&i64_bytes(&index), 0);
    for (dtype, es) in [
        (Dtype::U8, 1usize),
        (Dtype::Bf16, 2),
        (Dtype::F32, 4),
        (Dtype::I64, 8),
    ] {
        let src_bytes: Vec<u8> = (0..rows * cols * es).map(|i| (i * 37 + 11) as u8).collect();
        let src = d.upload(&src_bytes, 0);
        // poisoned output: 0xA5 everywhere, including the padding past the tensor
        let out_len = (rows * take * es).div_ceil(4) * 4 + 4;
        let out = d.upload(&vec![0xA5u8; out_len], 0xA5);
        let status = d.upload(&[0u8; 16], 0);
        let mut r = request(
            IndexOp::Gather,
            dtype,
            2,
            [rows as u64, take as u64, 0, 0],
            1,
        );
        r.index = View::new(0, [take as u64, 1, 0, 0]);
        r.src = View::new(0, [cols as u64, 1, 0, 0]);
        r.out = View::new(0, [take as u64, 1, 0, 0]);
        r.bound = cols as u64;
        assert_eq!(
            d.run(r, src, idx, out, status),
            ffi::ACK_OK,
            "{}",
            last_error()
        );
        let got = d.download(out, out_len);
        for row in 0..rows {
            for j in 0..take {
                let t = index[row * take + j] as usize;
                let want = &src_bytes[(row * cols + t) * es..(row * cols + t + 1) * es];
                let at = (row * take + j) * es;
                assert_eq!(&got[at..at + es], want, "{dtype:?} ({row}, {j})");
            }
        }
        assert!(
            got[rows * take * es..].iter().all(|&b| b == 0xA5),
            "{dtype:?}: a byte past the output changed"
        );
        assert_eq!(words(&d.download(status, 16)), vec![0; 4], "{dtype:?}");
    }
}

#[test]
fn an_index_select_broadcasts_a_one_dimensional_index_along_either_dimension() {
    let Some(dev) = open_or_unmeasured("index select") else {
        return;
    };
    let mut d = Device { dev, live: vec![] };
    // x [4, 6] f32; index [5] with a repeat
    let x: Vec<f32> = (0..24).map(|i| i as f32 + 0.5).collect();
    let src = d.upload(&f32_bytes(&x), 0);
    let index = [3i64, 0, 3, 1, 2];
    let idx = d.upload(&i64_bytes(&index), 0);
    let status = d.upload(&[0u8; 16], 0);
    // dim 0: out [5, 6] = x[index, :]
    let out = d.upload(&f32_bytes(&[f32::NAN; 30]), 0);
    let mut r = request(IndexOp::Gather, Dtype::F32, 2, [5, 6, 0, 0], 0);
    r.index = View::new(0, [1, 0, 0, 0]);
    r.src = View::new(0, [6, 1, 0, 0]);
    r.out = View::new(0, [6, 1, 0, 0]);
    r.bound = 4;
    assert_eq!(
        d.run(r, src, idx, out, status),
        ffi::ACK_OK,
        "{}",
        last_error()
    );
    let got = d.download(out, 120);
    let want: Vec<f32> = index
        .iter()
        .flat_map(|&t| x[t as usize * 6..t as usize * 6 + 6].to_vec())
        .collect();
    assert_eq!(got, f32_bytes(&want));
    // dim 1: out [4, 5] = x[:, index]
    let out = d.upload(&f32_bytes(&[f32::NAN; 20]), 0);
    let mut r = request(IndexOp::Gather, Dtype::F32, 2, [4, 5, 0, 0], 1);
    r.index = View::new(0, [0, 1, 0, 0]);
    r.src = View::new(0, [6, 1, 0, 0]);
    r.out = View::new(0, [5, 1, 0, 0]);
    r.bound = 6;
    assert_eq!(
        d.run(r, src, idx, out, status),
        ffi::ACK_OK,
        "{}",
        last_error()
    );
    let got = d.download(out, 80);
    let want: Vec<f32> = (0..4)
        .flat_map(|row| {
            index
                .iter()
                .map(move |&t| (row * 6 + t as usize) as f32 + 0.5)
        })
        .collect();
    assert_eq!(got, f32_bytes(&want));
    assert_eq!(words(&d.download(status, 16)), vec![0; 4]);
}

#[test]
fn a_scatter_leaves_the_last_of_repeated_indices_as_the_cpu_does() {
    let Some(dev) = open_or_unmeasured("index scatter") else {
        return;
    };
    let mut d = Device { dev, live: vec![] };
    // out [6, 8] uint8, index_copy_ along dim 0 from src [5, 8] with index [2, 4, 2, 0, 4]
    let index = [2i64, 4, 2, 0, 4];
    let idx = d.upload(&i64_bytes(&index), 0);
    let src_bytes: Vec<u8> = (0..40).map(|i| (i * 3 + 1) as u8).collect();
    let src = d.upload(&src_bytes, 0);
    let before: Vec<u8> = (0..48).map(|i| 200 + (i % 50) as u8).collect();
    let out = d.upload(&before, 0);
    let status = d.upload(&[0u8; 16], 0);
    let mut r = request(IndexOp::Scatter, Dtype::U8, 2, [5, 8, 0, 0], 0);
    r.index = View::new(0, [1, 0, 0, 0]);
    r.src = View::new(0, [8, 1, 0, 0]);
    r.out = View::new(0, [8, 1, 0, 0]);
    r.bound = 6;
    assert_eq!(
        d.run(r, src, idx, out, status),
        ffi::ACK_OK,
        "{}",
        last_error()
    );
    let mut want = before.clone();
    for (i, &t) in index.iter().enumerate() {
        let t = t as usize;
        want[t * 8..t * 8 + 8].copy_from_slice(&src_bytes[i * 8..i * 8 + 8]);
    }
    assert_eq!(
        d.download(out, 48),
        want,
        "the later of each repeat wins, untouched rows stay"
    );
    assert_eq!(words(&d.download(status, 16)), vec![0; 4]);
}

#[test]
fn a_value_scatter_writes_the_scalar_into_a_bool_mask() {
    let Some(dev) = open_or_unmeasured("index value scatter") else {
        return;
    };
    let mut d = Device { dev, live: vec![] };
    // torch.zeros(3, 12, dtype=bool).scatter(-1, idx [3, 2], True): each row
    // starts on a word, so each row's invocation owns its words
    let index = [11i64, 0, 4, 4, 1, 7];
    let idx = d.upload(&i64_bytes(&index), 0);
    let out = d.upload(&[0u8; 36], 0);
    let status = d.upload(&[0u8; 16], 0);
    let mut r = request(IndexOp::Scatter, Dtype::U8, 2, [3, 2, 0, 0], 1);
    r.index = View::new(0, [2, 1, 0, 0]);
    r.out = View::new(0, [12, 1, 0, 0]);
    r.bound = 12;
    r.value = Some(1);
    // the unread source may be bound to the output
    assert_eq!(
        d.run(r, out, idx, out, status),
        ffi::ACK_OK,
        "{}",
        last_error()
    );
    let mut want = vec![0u8; 36];
    for row in 0..3 {
        for j in 0..2 {
            want[row * 12 + index[row * 2 + j] as usize] = 1;
        }
    }
    assert_eq!(d.download(out, 36), want);
    assert_eq!(words(&d.download(status, 16)), vec![0; 4]);
}

#[test]
fn a_scatter_add_rounds_each_step_as_the_cpu_does_in_every_float_storage() {
    let Some(dev) = open_or_unmeasured("index scatter-add floats") else {
        return;
    };
    let mut d = Device { dev, live: vec![] };
    let table = f16_table();
    // index_add_(0, index [9], src [9, 6]) into out [4, 6], heavy on repeats,
    // with magnitudes far apart so each step's rounding matters
    let index = [1i64, 3, 1, 1, 0, 3, 1, 2, 1];
    let idx = d.upload(&i64_bytes(&index), 0);
    let status = d.upload(&[0u8; 16], 0);
    let value = |i: usize, salt: f32| -> f32 {
        let s = ((i * 2_654_435_761) % 1000) as f32 / 999.0 - 0.5;
        s * [1.0, 300.0, 0.003, 70.0][i % 4] + salt
    };
    // f32, with and without alpha
    for alpha in [None, Some(0.3f32)] {
        let src_v: Vec<f32> = (0..54).map(|i| value(i, 0.0)).collect();
        let out_v: Vec<f32> = (0..24).map(|i| value(i + 100, 0.25)).collect();
        let src = d.upload(&f32_bytes(&src_v), 0);
        let out = d.upload(&f32_bytes(&out_v), 0);
        let mut r = request(IndexOp::ScatterAdd, Dtype::F32, 2, [9, 6, 0, 0], 0);
        r.index = View::new(0, [1, 0, 0, 0]);
        r.src = View::new(0, [6, 1, 0, 0]);
        r.out = View::new(0, [6, 1, 0, 0]);
        r.bound = 4;
        r.alpha = alpha.map(f32::to_bits);
        assert_eq!(
            d.run(r, src, idx, out, status),
            ffi::ACK_OK,
            "{}",
            last_error()
        );
        let mut want = out_v.clone();
        for (i, &t) in index.iter().enumerate() {
            for j in 0..6 {
                let y = match alpha {
                    Some(a) => src_v[i * 6 + j] * a,
                    None => src_v[i * 6 + j],
                };
                want[t as usize * 6 + j] += y;
            }
        }
        assert_eq!(d.download(out, 96), f32_bytes(&want), "f32 alpha {alpha:?}");
    }
    // bf16 and f16: each step widened, added in f32, rounded to nearest even
    for dtype in [Dtype::Bf16, Dtype::F16] {
        let round = |x: f32| -> u16 {
            if dtype == Dtype::Bf16 {
                bf16_round(x)
            } else {
                f16_round(&table, x)
            }
        };
        let widen = |h: u16| -> f32 {
            if dtype == Dtype::Bf16 {
                bf16_value(h)
            } else {
                f16_value(&table, h)
            }
        };
        let src_h: Vec<u16> = (0..54).map(|i| round(value(i, 0.0))).collect();
        let out_h: Vec<u16> = (0..24).map(|i| round(value(i + 100, 0.25))).collect();
        let bytes = |v: &[u16]| -> Vec<u8> { v.iter().flat_map(|h| h.to_le_bytes()).collect() };
        let src = d.upload(&bytes(&src_h), 0);
        let out = d.upload(&bytes(&out_h), 0);
        let mut r = request(IndexOp::ScatterAdd, dtype, 2, [9, 6, 0, 0], 0);
        r.index = View::new(0, [1, 0, 0, 0]);
        r.src = View::new(0, [6, 1, 0, 0]);
        r.out = View::new(0, [6, 1, 0, 0]);
        r.bound = 4;
        assert_eq!(
            d.run(r, src, idx, out, status),
            ffi::ACK_OK,
            "{}",
            last_error()
        );
        let mut want = out_h.clone();
        for (i, &t) in index.iter().enumerate() {
            for j in 0..6 {
                let at = t as usize * 6 + j;
                want[at] = round(widen(want[at]) + widen(src_h[i * 6 + j]));
            }
        }
        assert_eq!(d.download(out, 48), bytes(&want), "{dtype:?}");
    }
    assert_eq!(words(&d.download(status, 16)), vec![0; 4]);
}

#[test]
fn an_int64_scatter_add_carries_across_the_word_boundary() {
    let Some(dev) = open_or_unmeasured("index scatter-add int64") else {
        return;
    };
    let mut d = Device { dev, live: vec![] };
    // counts.scatter_add_(0, flat [8], ones-ish [8]) into counts [3], with values
    // chosen so the low words wrap
    let flat = [2i64, 0, 2, 2, 1, 0, 2, 1];
    let vals = [
        0xFFFF_FFFFi64,
        -5,
        3,
        0x1_0000_0001,
        -0x7FFF_FFFF_FFFF,
        7,
        0xFFFF_FFFF,
        1,
    ];
    let before = [0x7FFF_FFFFi64, -1, 0xFFFF_FFF0];
    let idx = d.upload(&i64_bytes(&flat), 0);
    let src = d.upload(&i64_bytes(&vals), 0);
    let out = d.upload(&i64_bytes(&before), 0);
    let status = d.upload(&[0u8; 16], 0);
    let mut r = request(IndexOp::ScatterAdd, Dtype::I64, 1, [8, 0, 0, 0], 0);
    r.index = View::new(0, [1, 0, 0, 0]);
    r.src = View::new(0, [1, 0, 0, 0]);
    r.out = View::new(0, [1, 0, 0, 0]);
    r.bound = 3;
    assert_eq!(
        d.run(r, src, idx, out, status),
        ffi::ACK_OK,
        "{}",
        last_error()
    );
    let mut want = before;
    for (i, &t) in flat.iter().enumerate() {
        want[t as usize] = want[t as usize].wrapping_add(vals[i]);
    }
    assert_eq!(d.download(out, 24), i64_bytes(&want));
}

#[test]
fn an_index_out_of_range_is_never_used_and_is_reported_in_the_status_word() {
    let Some(dev) = open_or_unmeasured("index out of range") else {
        return;
    };
    let mut d = Device { dev, live: vec![] };
    let x: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let src = d.upload(&f32_bytes(&x), 0);
    // bound 8: 8 and -1 are out of range for a gather, and -1 wraps for index.Tensor
    let index = [3i64, 8, -1, 7];
    let idx = d.upload(&i64_bytes(&index), 0);
    for (wrap, slot) in [(false, 1u32), (true, 2u32)] {
        let out = d.upload(&f32_bytes(&[1234.0; 4]), 0);
        let status = d.upload(&[0u8; 16], 0);
        let mut r = request(IndexOp::Gather, Dtype::F32, 1, [4, 0, 0, 0], 0);
        r.index = View::new(0, [1, 0, 0, 0]);
        r.src = View::new(0, [1, 0, 0, 0]);
        r.out = View::new(0, [1, 0, 0, 0]);
        r.bound = 8;
        r.wrap = wrap;
        r.status_slot = slot;
        assert_eq!(
            d.run(r, src, idx, out, status),
            ffi::ACK_OK,
            "{}",
            last_error()
        );
        let got = words(&d.download(out, 16));
        assert_eq!(got[0], 3.0f32.to_bits());
        assert_eq!(got[1], 0xFFFF_FFFF, "an index of 8 reads nothing: all ones");
        assert_eq!(got[3], 7.0f32.to_bits());
        let st = words(&d.download(status, 16));
        if wrap {
            assert_eq!(got[2], 7.0f32.to_bits(), "-1 counts from the end");
        } else {
            assert_eq!(got[2], 0xFFFF_FFFF);
        }
        assert_eq!(st[slot as usize], STATUS_INDEX, "wrap {wrap}");
        for (k, &w) in st.iter().enumerate() {
            if k != slot as usize {
                assert_eq!(w, 0, "only the call's own slot is written");
            }
        }
    }
    // a scatter skips an index out of range and still writes the others
    let out = d.upload(&f32_bytes(&[0.0; 8]), 0);
    let status = d.upload(&[0u8; 16], 0);
    let mut r = request(IndexOp::Scatter, Dtype::F32, 1, [4, 0, 0, 0], 0);
    r.index = View::new(0, [1, 0, 0, 0]);
    r.src = View::new(0, [1, 0, 0, 0]);
    r.out = View::new(0, [1, 0, 0, 0]);
    r.bound = 8;
    assert_eq!(
        d.run(r, src, idx, out, status),
        ffi::ACK_OK,
        "{}",
        last_error()
    );
    let got = words(&d.download(out, 32));
    let mut want = [0.0f32.to_bits(); 8];
    want[3] = 0.0f32.to_bits(); // x[0] = 0.0 lands on 3
    want[7] = 3.0f32.to_bits(); // x[3] = 3.0 lands on 7
    assert_eq!(got, want.to_vec());
    assert_eq!(words(&d.download(status, 16))[0], STATUS_INDEX);
}

#[test]
fn the_entry_plans_against_the_real_buffers_and_refuses_by_name() {
    let Some(dev) = open_or_unmeasured("index refusals") else {
        return;
    };
    let mut d = Device { dev, live: vec![] };
    let src = d.upload(&[0u8; 64], 0);
    let idx = d.upload(&i64_bytes(&[0, 1, 2, 3]), 0);
    let out = d.upload(&[0u8; 16], 0);
    let status = d.upload(&[0u8; 16], 0);
    // a gather whose source rows run to 17 f32s from a 64-byte buffer
    let mut r = request(IndexOp::Gather, Dtype::F32, 1, [4, 0, 0, 0], 0);
    r.index = View::new(0, [1, 0, 0, 0]);
    r.src = View::new(0, [1, 0, 0, 0]);
    r.out = View::new(0, [1, 0, 0, 0]);
    r.bound = 17;
    let rc = d.run(r, src, idx, out, status);
    assert_eq!(rc, ffi::ACK_ERR_SIZE);
    assert!(
        last_error().ends_with("index-address-overflow"),
        "{}",
        last_error()
    );
    // the status word bound to the output's buffer
    r.bound = 4;
    let rc = d.run(r, src, idx, out, out);
    assert_eq!(rc, ffi::ACK_ERR_SHAPE);
    assert!(
        last_error().ends_with("index-status-overlaps-operand"),
        "{}",
        last_error()
    );
    // a status slot past its 16-byte buffer
    r.status_slot = 4;
    let rc = d.run(r, src, idx, out, status);
    assert_eq!(rc, ffi::ACK_ERR_SIZE);
    assert!(
        last_error().ends_with("index-status-slot-out-of-range"),
        "{}",
        last_error()
    );
    // a bool mask of ten-byte rows shares words between rows: schema 1 refused
    // it (index-scatter-output-misaligned), schema 2 writes it with atomics
    // (bit for bit in `sub_word_scatters_whose_fibers_share_words_...`)
    let mut m = request(IndexOp::Scatter, Dtype::U8, 2, [3, 2, 0, 0], 1);
    m.index = View::new(0, [2, 1, 0, 0]);
    m.out = View::new(0, [10, 1, 0, 0]);
    m.bound = 10;
    m.value = Some(1);
    let mask = d.upload(&[0u8; 32], 0);
    let six = d.upload(&i64_bytes(&[0, 1, 2, 3, 4, 5]), 0);
    let rc = d.run(m, mask, six, mask, status);
    assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
}

/// `combine(old, new)`: the bytes a host reference writes for one element.
type Combine = dyn Fn(&[u8], &[u8]) -> Vec<u8>;

/// A host scatter over a 2-D iteration space in row-major order, which walks
/// every fiber along `dim` in index order: `combine(old, new)` gives the bytes
/// written (the source's for a scatter, the sum rounded in the element's own
/// type for a scatter-add).
fn scatter_reference(
    r: &IndexRequest,
    index: &[i64],
    src: &[u8],
    out: &mut [u8],
    es: usize,
    combine: &Combine,
) {
    let at = |v: &View, a: u64, b: u64| (v.offset + a * v.strides[0] + b * v.strides[1]) as usize;
    for a in 0..r.shape[0] {
        for b in 0..r.shape[1] {
            let t = index[at(&r.index, a, b)] as u64;
            let (oa, ob) = if r.dim == 0 { (t, b) } else { (a, t) };
            let o = at(&r.out, oa, ob) * es;
            let s = match r.value {
                Some(bits) => bits.to_le_bytes()[..es].to_vec(),
                None => {
                    let s = at(&r.src, a, b) * es;
                    src[s..s + es].to_vec()
                }
            };
            let new = combine(&out[o..o + es], &s);
            out[o..o + es].copy_from_slice(&new);
        }
    }
}

#[test]
fn sub_word_scatters_whose_fibers_share_words_are_written_with_atomics_bit_for_bit() {
    let Some(dev) = open_or_unmeasured("index atomic sub-word scatter") else {
        return;
    };
    let mut d = Device { dev, live: vec![] };
    // Every case below was refused as index-scatter-output-misaligned under
    // schema 1: neighbouring fibers write different bytes of one word, so a
    // plain read-modify-write would lose one of them. The output starts as a
    // byte pattern and is compared whole, so a lost neighbour, a byte the
    // tensor keeps, or a byte past the tensor that changed all fail.
    let pattern = |n: usize| -> Vec<u8> { (0..n).map(|i| (i * 37 + 11) as u8).collect() };
    let replace = |_: &[u8], s: &[u8]| s.to_vec();
    let add_i16 = |o: &[u8], s: &[u8]| {
        let sum = i16::from_le_bytes([o[0], o[1]]).wrapping_add(i16::from_le_bytes([s[0], s[1]]));
        sum.to_le_bytes().to_vec()
    };
    let add_i8 = |o: &[u8], s: &[u8]| vec![(o[0] as i8).wrapping_add(s[0] as i8) as u8];
    let add_bf16 = |o: &[u8], s: &[u8]| {
        let sum = bf16_value(u16::from_le_bytes([o[0], o[1]]))
            + bf16_value(u16::from_le_bytes([s[0], s[1]]));
        bf16_round(sum).to_le_bytes().to_vec()
    };
    let mut check = |name: &str,
                     r: IndexRequest,
                     es: usize,
                     index: &[i64],
                     src_bytes: &[u8],
                     out_len: usize,
                     combine: &Combine| {
        let out_len = out_len.div_ceil(4) * 4;
        let before = pattern(out_len);
        let idx = d.upload(&i64_bytes(index), 0);
        let out = d.upload(&before, 0);
        // a value scatter leaves its source unread, so it may be the output
        let src = if r.value.is_some() {
            out
        } else {
            d.upload(src_bytes, 0)
        };
        let status = d.upload(&[0u8; 16], 0);
        assert_eq!(
            d.run(r, src, idx, out, status),
            ffi::ACK_OK,
            "{name}: {}",
            last_error()
        );
        let mut want = before.clone();
        scatter_reference(&r, index, src_bytes, &mut want, es, combine);
        assert_ne!(want, before, "{name}: the case writes something");
        assert_eq!(d.download(out, out_len), want, "{name}");
        assert_eq!(
            words(&d.download(status, 16)),
            vec![0; 4],
            "{name}: no index error, and no plan error (2) from the shader's guards"
        );
    };

    // (a) torch.zeros(257, 10, dtype=bool).scatter(-1, idx [257, 3], True):
    // ten-byte rows, so each row's last word is the next row's first, the
    // toy stage-E step's last scatter.value refusals (MEASURED 2026-09-28)
    let mut r = request(IndexOp::Scatter, Dtype::U8, 2, [257, 3, 0, 0], 1);
    r.index = View::new(0, [3, 1, 0, 0]);
    r.out = View::new(0, [10, 1, 0, 0]);
    r.bound = 10;
    r.value = Some(1);
    let index: Vec<i64> = (0..257)
        .flat_map(|a: i64| [(a * 7) % 10, (a * 3 + 1) % 10, (a * 7) % 10])
        .collect();
    check(
        "bool mask of ten-byte rows",
        r,
        1,
        &index,
        &[],
        2570,
        &replace,
    );

    // (b) a uint8 scatter along dim 0 of [5, 1001] into [6, 1001]: 1,001
    // column fibers, each sharing words with its neighbours, all running at
    // once; every column repeats an index, and the later one must win
    let (rows, cols) = (5u64, 1001u64);
    let mut r = request(IndexOp::Scatter, Dtype::U8, 2, [rows, cols, 0, 0], 0);
    r.index = View::new(0, [cols, 1, 0, 0]);
    r.src = View::new(0, [cols, 1, 0, 0]);
    r.out = View::new(0, [cols, 1, 0, 0]);
    r.bound = 6;
    let index: Vec<i64> = (0..rows * cols)
        .map(|i| ((i / cols * 5 + i % cols * 3) % 4) as i64)
        .collect();
    let src: Vec<u8> = (0..rows * cols).map(|i| (i * 91 + 7) as u8).collect();
    check(
        "uint8 columns of [5, 1001]",
        r,
        1,
        &index,
        &src,
        6006,
        &replace,
    );

    // (c) index_add_ in int16 over five columns (an odd count: the pairs of
    // columns a word holds cannot be grouped), wrapping, with repeats
    let add_index = [1i64, 3, 1, 1, 0, 3, 1, 2, 1];
    let mut r = request(IndexOp::ScatterAdd, Dtype::I16, 2, [9, 5, 0, 0], 0);
    r.index = View::new(0, [1, 0, 0, 0]);
    r.src = View::new(0, [5, 1, 0, 0]);
    r.out = View::new(0, [5, 1, 0, 0]);
    r.bound = 4;
    let src: Vec<u8> = (0..45i32)
        .flat_map(|i| ((i * 12_345 - 20_000) as i16).to_le_bytes())
        .collect();
    check(
        "int16 index_add_ over five columns",
        r,
        2,
        &add_index,
        &src,
        40,
        &add_i16,
    );

    // (d) the same in bf16 at an odd element offset, each step rounded to
    // bf16 (the kit's add; torch's CPU rounds once, so the adapter refuses
    // 16-bit float adds and this is the kit's own contract)
    let mut r = request(IndexOp::ScatterAdd, Dtype::Bf16, 2, [9, 6, 0, 0], 0);
    r.index = View::new(0, [1, 0, 0, 0]);
    r.src = View::new(0, [6, 1, 0, 0]);
    r.out = View::new(1, [6, 1, 0, 0]);
    r.bound = 4;
    let src: Vec<u8> = (0..54usize)
        .flat_map(|i| {
            let s = ((i * 2_654_435_761) % 1000) as f32 / 999.0 - 0.5;
            bf16_round(s * [1.0, 300.0, 0.003, 70.0][i % 4]).to_le_bytes()
        })
        .collect();
    // read as bf16, the pattern's 26 elements are all finite (no high byte of
    // 0x7F or 0xFF); one is a subnormal, which adds as zero to a normal source
    // whether or not the device flushes it
    check(
        "bf16 index_add_ at offset 1",
        r,
        2,
        &add_index,
        &src,
        52,
        &add_bf16,
    );

    // (e) int8 scatter-add into every other byte (output strides [16, 2]):
    // the bytes between the tensor's must come back as they were
    let mut r = request(IndexOp::ScatterAdd, Dtype::I8, 2, [9, 8, 0, 0], 0);
    r.index = View::new(0, [8, 1, 0, 0]);
    r.src = View::new(0, [8, 1, 0, 0]);
    r.out = View::new(0, [16, 2, 0, 0]);
    r.bound = 4;
    let index: Vec<i64> = (0..72i64).map(|i| (i * 5 + i / 8) % 4).collect();
    let src: Vec<u8> = (0..72i32).map(|i| (i * 57 - 100) as i8 as u8).collect();
    check(
        "int8 scatter-add into every other byte",
        r,
        1,
        &index,
        &src,
        64,
        &add_i8,
    );
}

#[test]
fn a_three_byte_output_is_gathered_through_its_word_rounded_buffer() {
    let Some(dev) = open_or_unmeasured("index three-byte output") else {
        return;
    };
    let mut d = Device { dev, live: vec![] };
    let src = d.upload(&[10u8, 11, 12, 13, 14, 15, 16], 0);
    let idx = d.upload(&i64_bytes(&[6, 0, 3]), 0);
    let status = d.upload(&[0u8; 16], 0);
    // a buffer whose LOGICAL size is three bytes, exactly as torch allocates a
    // three-element uint8 tensor; the kit creates its VkBuffer a word long
    let out = ffi::ack_buffer_alloc(d.dev, 3);
    assert!(!out.is_null(), "{}", last_error());
    d.live.push(out);
    let rc = unsafe { ffi::ack_upload(d.dev, out, [0xEEu8; 3].as_ptr(), 3) };
    assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
    let mut r = request(IndexOp::Gather, Dtype::U8, 1, [3, 0, 0, 0], 0);
    r.index = View::new(0, [1, 0, 0, 0]);
    r.src = View::new(0, [1, 0, 0, 0]);
    r.out = View::new(0, [1, 0, 0, 0]);
    r.bound = 7;
    assert_eq!(
        d.run(r, src, idx, out, status),
        ffi::ACK_OK,
        "{}",
        last_error()
    );
    assert_eq!(d.download(out, 3), vec![16, 10, 13]);
    // four elements do not fit three bytes
    r.shape = [4, 0, 0, 0];
    let idx4 = d.upload(&i64_bytes(&[6, 0, 3, 1]), 0);
    assert_eq!(d.run(r, src, idx4, out, status), ffi::ACK_ERR_SIZE);
    assert!(
        last_error().ends_with("index-address-overflow"),
        "{}",
        last_error()
    );
}

/// One step of torch's CPU float64 scatter-add, `out += src` (MEASURED
/// 2026-09-28 on torch 2.14, scatter_add_ and index_add_ alike): the source's
/// NaN if it is one, else the output's, quieted; inf + -inf is x86's default
/// NaN; otherwise IEEE round to nearest even, which the host's own add is.
fn f64_step(old: u64, src: u64) -> u64 {
    const QUIET: u64 = 0x0008_0000_0000_0000;
    let (o, s) = (f64::from_bits(old), f64::from_bits(src));
    if s.is_nan() {
        return src | QUIET;
    }
    if o.is_nan() {
        return old | QUIET;
    }
    let sum = std::hint::black_box(o) + std::hint::black_box(s);
    if sum.is_nan() {
        0xFFF8_0000_0000_0000
    } else {
        sum.to_bits()
    }
}

fn u64_bytes(values: &[u64]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

#[test]
fn a_float64_scatter_add_is_torchs_cpu_add_bit_for_bit_nans_included() {
    let Some(dev) = open_or_unmeasured("index scatter-add float64") else {
        return;
    };
    let mut d = Device { dev, live: vec![] };
    // (a) every ordered pair of special values, one step each: out[i] = a,
    // src[i] = b, a 1-D scatter-add through the identity index
    let specials: [u64; 24] = [
        0x0000_0000_0000_0000, // +0
        0x8000_0000_0000_0000, // -0
        0x0000_0000_0000_0001, // the smallest subnormal
        0x8000_0000_0000_0001,
        0x000F_FFFF_FFFF_FFFF, // the largest subnormal
        0x0010_0000_0000_0000, // the smallest normal
        0x8010_0000_0000_0000,
        0x3FF0_0000_0000_0000, // 1
        0xBFF0_0000_0000_0000, // -1
        0x3FF0_0000_0000_0001, // 1 + ulp
        0x3CA0_0000_0000_0000, // 2^-53: 1 + it ties to even
        0x3CA0_0000_0000_0001, // just past the tie
        0x4330_0000_0000_0001, // 2^52 + 1
        0x3FE0_0000_0000_0000, // 0.5
        0x7FEF_FFFF_FFFF_FFFF, // the largest finite: doubled, it overflows
        0xFFEF_FFFF_FFFF_FFFF,
        0x7CA0_0000_0000_0000, // half an ulp of the largest finite
        0x7FF0_0000_0000_0000, // +inf
        0xFFF0_0000_0000_0000, // -inf
        0x7FF8_0000_0000_0AAA, // quiet NaN, positive
        0xFFF8_0000_0000_0BBB, // quiet NaN, negative
        0x7FF0_0000_0000_0CCC, // signaling NaN, positive
        0xFFF0_0000_0000_0DDD, // signaling NaN, negative
        0x0000_0000_0000_0003, // a subnormal: sums of subnormals stay exact
    ];
    let n = specials.len();
    let olds: Vec<u64> = (0..n * n).map(|i| specials[i / n]).collect();
    let srcs: Vec<u64> = (0..n * n).map(|i| specials[i % n]).collect();
    let identity: Vec<i64> = (0..(n * n) as i64).collect();
    let idx = d.upload(&i64_bytes(&identity), 0);
    let src = d.upload(&u64_bytes(&srcs), 0);
    let out = d.upload(&u64_bytes(&olds), 0);
    let status = d.upload(&[0u8; 16], 0);
    let mut r = request(
        IndexOp::ScatterAdd,
        Dtype::F64,
        1,
        [(n * n) as u64, 0, 0, 0],
        0,
    );
    r.index = View::new(0, [1, 0, 0, 0]);
    r.src = View::new(0, [1, 0, 0, 0]);
    r.out = View::new(0, [1, 0, 0, 0]);
    r.bound = (n * n) as u64;
    assert_eq!(
        d.run(r, src, idx, out, status),
        ffi::ACK_OK,
        "{}",
        last_error()
    );
    let got = d.download(out, n * n * 8);
    for (i, (&old, &new)) in olds.iter().zip(&srcs).enumerate() {
        let bits = u64::from_le_bytes(got[i * 8..i * 8 + 8].try_into().unwrap());
        let want = f64_step(old, new);
        assert_eq!(
            bits, want,
            "{old:016X} + {new:016X}: got {bits:016X}, want {want:016X}"
        );
    }
    assert_eq!(words(&d.download(status, 16)), vec![0; 4]);

    // (b) index_add_ of 97 rows into 8, over 16 columns: every column walks
    // its rows in index order, so each output element takes a dozen rounded
    // steps; values from every exponent range, specials mixed in
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut value = move || -> u64 {
        let r = next();
        let sign = r & 0x8000_0000_0000_0000;
        let mantissa = next() & 0x000F_FFFF_FFFF_FFFF;
        let exponent = match r % 8 {
            0 => 0,                    // a subnormal (or zero)
            1 => 1 + (r >> 8) % 60,    // tiny normals
            2 => 2046 - (r >> 8) % 60, // huge: sums overflow
            3 => return specials[((r >> 8) % specials.len() as u64) as usize],
            _ => 1023 - 30 + (r >> 8) % 60, // around 1, so rounding cancels
        };
        sign | (exponent << 52) | mantissa
    };
    let (rows, cols, bound) = (97usize, 16usize, 8usize);
    let src_v: Vec<u64> = (0..rows * cols).map(|_| value()).collect();
    let out_v: Vec<u64> = (0..bound * cols).map(|_| value()).collect();
    let rows_index: Vec<i64> = (0..rows as i64)
        .map(|i| (i * 5 + i / 7) % bound as i64)
        .collect();
    let idx = d.upload(&i64_bytes(&rows_index), 0);
    let src = d.upload(&u64_bytes(&src_v), 0);
    let out = d.upload(&u64_bytes(&out_v), 0);
    let mut r = request(
        IndexOp::ScatterAdd,
        Dtype::F64,
        2,
        [rows as u64, cols as u64, 0, 0],
        0,
    );
    r.index = View::new(0, [1, 0, 0, 0]);
    r.src = View::new(0, [cols as u64, 1, 0, 0]);
    r.out = View::new(0, [cols as u64, 1, 0, 0]);
    r.bound = bound as u64;
    assert_eq!(
        d.run(r, src, idx, out, status),
        ffi::ACK_OK,
        "{}",
        last_error()
    );
    let mut want = out_v.clone();
    for (i, &t) in rows_index.iter().enumerate() {
        for j in 0..cols {
            let at = t as usize * cols + j;
            want[at] = f64_step(want[at], src_v[i * cols + j]);
        }
    }
    let got = d.download(out, bound * cols * 8);
    for (k, &w) in want.iter().enumerate() {
        let bits = u64::from_le_bytes(got[k * 8..k * 8 + 8].try_into().unwrap());
        assert_eq!(bits, w, "element {k}: got {bits:016X}, want {w:016X}");
    }
    assert_eq!(words(&d.download(status, 16)), vec![0; 4]);
}

// ---- long fibers (schema 3) --------------------------------------------------------

/// A deterministic stream of 64-bit values (xorshift64).
fn stream(seed: u64) -> impl FnMut() -> u64 {
    let mut state = seed | 1;
    move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    }
}

/// The strategy the kit's planner picks for `r`, planned against buffers
/// large enough for it: what the device test below then runs.
fn strategy_of(r: IndexRequest) -> IndexStrategy {
    let big = |id| Storage::new(id, 1 << 30);
    IndexPlan::new(
        r,
        Bindings {
            src: big(1),
            index: big(2),
            out: big(3),
            status: Storage::new(4, 64),
        },
    )
    .expect("the request plans")
    .strategy()
}

/// A 1-D request over `n` elements into `bound` destinations, contiguous.
fn line(op: IndexOp, dtype: Dtype, n: u64, bound: u64) -> IndexRequest {
    let mut r = request(op, dtype, 1, [n, 0, 0, 0], 0);
    r.index = View::new(0, [1, 0, 0, 0]);
    r.src = View::new(0, [1, 0, 0, 0]);
    r.out = View::new(0, [1, 0, 0, 0]);
    r.bound = bound;
    r
}

#[test]
fn a_long_fibers_scatter_elects_the_cpus_last_writer_bit_for_bit() {
    let Some(dev) = open_or_unmeasured("index elected scatter") else {
        return;
    };
    let mut d = Device { dev, live: vec![] };
    let mut next = stream(0x5EED_0301);
    // int64, the MoE routing's form (padded_src.scatter_ and inv.index_copy_):
    // 65,536 elements into 70,000 destinations, about a third of the indices
    // repeated, one out of range at element 1,234
    let (n, bound) = (65_536usize, 70_000u64);
    let mut index: Vec<i64> = (0..n).map(|_| (next() % bound) as i64).collect();
    for i in (0..n).step_by(3) {
        index[i] = index[(i * 7919) % n];
    }
    index[1_234] = bound as i64;
    let src: Vec<i64> = (0..n)
        .map(|i| ((next() as i64) & !0xFFFF) | i as i64)
        .collect();
    let before: Vec<i64> = (0..bound).map(|j| -(j as i64) - 1).collect();
    let r = line(IndexOp::Scatter, Dtype::I64, n as u64, bound);
    assert_eq!(strategy_of(r), IndexStrategy::Elected);
    let (sb, ib) = (
        d.upload(&i64_bytes(&src), 0),
        d.upload(&i64_bytes(&index), 0),
    );
    let ob = d.upload(&i64_bytes(&before), 0);
    let status = d.upload(&[0u8; 16], 0);
    assert_eq!(
        d.run(r, sb, ib, ob, status),
        ffi::ACK_OK,
        "{}",
        last_error()
    );
    let mut want = before.clone();
    for (i, &t) in index.iter().enumerate() {
        if (0..bound as i64).contains(&t) {
            want[t as usize] = src[i];
        }
    }
    assert_eq!(d.download(ob, bound as usize * 8), i64_bytes(&want));
    assert_eq!(words(&d.download(status, 16)), vec![STATUS_INDEX, 0, 0, 0]);

    // f32, four fibers of 2,048 along dim 1 of out [4, 3000], int32 indices
    let (rows, len, width) = (4usize, 2_048usize, 3_000u64);
    let idx32: Vec<i32> = (0..rows * len).map(|_| (next() % 2_000) as i32).collect();
    let vals: Vec<f32> = (0..rows * len).map(|i| i as f32 * 0.5 - 7.0).collect();
    let before: Vec<f32> = (0..rows * width as usize).map(|j| -(j as f32)).collect();
    let mut r = request(
        IndexOp::Scatter,
        Dtype::F32,
        2,
        [rows as u64, len as u64, 0, 0],
        1,
    );
    r.index = View::new(0, [len as u64, 1, 0, 0]);
    r.index_i32 = true;
    r.src = View::new(0, [len as u64, 1, 0, 0]);
    r.out = View::new(0, [width, 1, 0, 0]);
    r.bound = width;
    assert_eq!(strategy_of(r), IndexStrategy::Elected);
    let ib = d.upload(
        &idx32
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>(),
        0,
    );
    let (sb, ob) = (
        d.upload(&f32_bytes(&vals), 0),
        d.upload(&f32_bytes(&before), 0),
    );
    let status = d.upload(&[0u8; 16], 0);
    assert_eq!(
        d.run(r, sb, ib, ob, status),
        ffi::ACK_OK,
        "{}",
        last_error()
    );
    let mut want = before.clone();
    for row in 0..rows {
        for j in 0..len {
            want[row * width as usize + idx32[row * len + j] as usize] = vals[row * len + j];
        }
    }
    assert_eq!(d.download(ob, before.len() * 4), f32_bytes(&want));
    assert_eq!(words(&d.download(status, 16)), vec![0; 4]);

    // uint8 with a scalar: a bool mask of 7,001 bytes, 5,000 writes of 1, the
    // byte past the mask kept (neighbouring winners share words: atomics)
    let (n, bound) = (5_000usize, 7_001u64);
    let index: Vec<i64> = (0..n).map(|_| (next() % bound) as i64).collect();
    let mut r = line(IndexOp::Scatter, Dtype::U8, n as u64, bound);
    r.value = Some(1);
    assert_eq!(strategy_of(r), IndexStrategy::Elected);
    let ib = d.upload(&i64_bytes(&index), 0);
    let mask = vec![0u8; bound as usize + 1];
    let ob = d.upload(&mask, 0xEE);
    let status = d.upload(&[0u8; 16], 0);
    assert_eq!(
        d.run(r, ob, ib, ob, status),
        ffi::ACK_OK,
        "{}",
        last_error()
    );
    let mut want = mask.clone();
    for &t in &index {
        want[t as usize] = 1;
    }
    assert_eq!(
        d.download(ob, mask.len().div_ceil(4) * 4)[..mask.len()].to_vec(),
        want,
        "the mask, and the byte past it"
    );
    assert_eq!(words(&d.download(status, 16)), vec![0; 4]);

    // int16 from a source, repeats and all: sub-word winners with values
    let (n, bound) = (3_000usize, 1_001u64);
    let index: Vec<i64> = (0..n).map(|_| (next() % bound) as i64).collect();
    let src: Vec<i16> = (0..n).map(|i| (i as i16).wrapping_mul(97)).collect();
    let r = line(IndexOp::Scatter, Dtype::I16, n as u64, bound);
    assert_eq!(strategy_of(r), IndexStrategy::Elected);
    let ib = d.upload(&i64_bytes(&index), 0);
    let sb = d.upload(
        &src.iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>(),
        0,
    );
    let before: Vec<u8> = (0..bound as usize * 2 + 2).map(|j| j as u8).collect();
    let ob = d.upload(&before, 0);
    let status = d.upload(&[0u8; 16], 0);
    assert_eq!(
        d.run(r, sb, ib, ob, status),
        ffi::ACK_OK,
        "{}",
        last_error()
    );
    let mut want = before.clone();
    for (i, &t) in index.iter().enumerate() {
        want[t as usize * 2..t as usize * 2 + 2].copy_from_slice(&src[i].to_le_bytes());
    }
    assert_eq!(d.download(ob, before.len()), want);
}

#[test]
fn a_long_fibers_integer_scatter_add_is_the_wrapping_sum_with_atomics() {
    let Some(dev) = open_or_unmeasured("index atomic scatter-add") else {
        return;
    };
    let mut d = Device { dev, live: vec![] };
    let mut next = stream(0x5EED_0302);
    // counts.scatter_add_(0, flat [65536], ones) into 128 experts, then full
    // 64-bit sources whose low words carry into the high words
    let (n, bound) = (65_536usize, 128u64);
    let index: Vec<i64> = (0..n).map(|_| (next() % bound) as i64).collect();
    for ones in [true, false] {
        let src: Vec<i64> = (0..n)
            .map(|_| if ones { 1 } else { next() as i64 })
            .collect();
        let before: Vec<i64> = (0..bound).map(|_| next() as i64).collect();
        let r = line(IndexOp::ScatterAdd, Dtype::I64, n as u64, bound);
        assert_eq!(strategy_of(r), IndexStrategy::AtomicAdd);
        let (sb, ib) = (
            d.upload(&i64_bytes(&src), 0),
            d.upload(&i64_bytes(&index), 0),
        );
        let ob = d.upload(&i64_bytes(&before), 0);
        let status = d.upload(&[0u8; 16], 0);
        assert_eq!(
            d.run(r, sb, ib, ob, status),
            ffi::ACK_OK,
            "{}",
            last_error()
        );
        let mut want = before.clone();
        for (i, &t) in index.iter().enumerate() {
            want[t as usize] = want[t as usize].wrapping_add(src[i]);
        }
        assert_eq!(
            d.download(ob, bound as usize * 8),
            i64_bytes(&want),
            "int64 ones={ones}"
        );
        assert_eq!(words(&d.download(status, 16)), vec![0; 4]);
    }
    // int32 with int32 indices, one out of range (skipped and reported)
    let (n, bound) = (40_000usize, 1_000u64);
    let mut idx32: Vec<i32> = (0..n).map(|_| (next() % bound) as i32).collect();
    idx32[777] = -1;
    let src: Vec<i32> = (0..n).map(|_| next() as i32).collect();
    let before: Vec<i32> = (0..bound).map(|j| j as i32 * 3).collect();
    let mut r = line(IndexOp::ScatterAdd, Dtype::I32, n as u64, bound);
    r.index_i32 = true;
    r.status_slot = 2;
    assert_eq!(strategy_of(r), IndexStrategy::AtomicAdd);
    let bytes32 = |v: &[i32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    let (sb, ib) = (d.upload(&bytes32(&src), 0), d.upload(&bytes32(&idx32), 0));
    let ob = d.upload(&bytes32(&before), 0);
    let status = d.upload(&[0u8; 16], 0);
    assert_eq!(
        d.run(r, sb, ib, ob, status),
        ffi::ACK_OK,
        "{}",
        last_error()
    );
    let mut want = before.clone();
    for (i, &t) in idx32.iter().enumerate() {
        if t >= 0 {
            want[t as usize] = want[t as usize].wrapping_add(src[i]);
        }
    }
    assert_eq!(d.download(ob, bound as usize * 4), bytes32(&want));
    assert_eq!(words(&d.download(status, 16)), vec![0, 0, STATUS_INDEX, 0]);
}

#[test]
fn a_long_fibers_float_scatter_add_is_grouped_and_adds_in_the_cpus_order() {
    let Some(dev) = open_or_unmeasured("index grouped scatter-add") else {
        return;
    };
    let mut d = Device { dev, live: vec![] };
    let mut next = stream(0x5EED_0303);
    // float64 into 128 experts, as the load-balancing loss counts: ones, then
    // values whose order matters, NaNs of either sign among them; one index
    // out of range (skipped and reported)
    let (n, bound) = (65_536usize, 128u64);
    let mut index: Vec<i64> = (0..n).map(|_| (next() % bound) as i64).collect();
    index[4_321] = bound as i64 + 7;
    for ones in [true, false] {
        let src: Vec<u64> = (0..n)
            .map(|i| {
                if ones {
                    1.0f64.to_bits()
                } else if i % 9_973 == 0 {
                    0x7FF8_0000_0000_0123 | ((i as u64 & 1) << 63)
                } else {
                    let v = next();
                    ((v >> 12) & 0x000F_FFFF_FFFF_FFFF)
                        | ((0x3F0 + (v % 64)) << 52)
                        | (v & (1 << 63))
                }
            })
            .collect();
        let before: Vec<u64> = (0..bound).map(|j| (j as f64 * 0.25).to_bits()).collect();
        let r = line(IndexOp::ScatterAdd, Dtype::F64, n as u64, bound);
        assert_eq!(strategy_of(r), IndexStrategy::Grouped);
        let (sb, ib) = (
            d.upload(&u64_bytes(&src), 0),
            d.upload(&i64_bytes(&index), 0),
        );
        let ob = d.upload(&u64_bytes(&before), 0);
        let status = d.upload(&[0u8; 16], 0);
        assert_eq!(
            d.run(r, sb, ib, ob, status),
            ffi::ACK_OK,
            "{}",
            last_error()
        );
        let mut want = before.clone();
        for (i, &t) in index.iter().enumerate() {
            if (0..bound as i64).contains(&t) {
                want[t as usize] = f64_step(want[t as usize], src[i]);
            }
        }
        assert_eq!(
            d.download(ob, bound as usize * 8),
            u64_bytes(&want),
            "float64 ones={ones}"
        );
        assert_eq!(words(&d.download(status, 16)), vec![STATUS_INDEX, 0, 0, 0]);
    }
    // f32 with alpha into 300 destinations (two-byte keys), int32 indices
    let (n, bound) = (20_000usize, 300u64);
    let idx32: Vec<i32> = (0..n).map(|_| (next() % bound) as i32).collect();
    let src: Vec<f32> = (0..n)
        .map(|i| ((next() % 2_001) as f32 - 1_000.0) * [1.0, 1e-3, 37.0][i % 3])
        .collect();
    let before: Vec<f32> = (0..bound).map(|j| j as f32 * 0.5).collect();
    let mut r = line(IndexOp::ScatterAdd, Dtype::F32, n as u64, bound);
    r.index_i32 = true;
    r.alpha = Some(0.3f32.to_bits());
    assert_eq!(strategy_of(r), IndexStrategy::Grouped);
    let ib = d.upload(
        &idx32
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>(),
        0,
    );
    let (sb, ob) = (
        d.upload(&f32_bytes(&src), 0),
        d.upload(&f32_bytes(&before), 0),
    );
    let status = d.upload(&[0u8; 16], 0);
    assert_eq!(
        d.run(r, sb, ib, ob, status),
        ffi::ACK_OK,
        "{}",
        last_error()
    );
    let mut want = before.clone();
    for (i, &t) in idx32.iter().enumerate() {
        let y = std::hint::black_box(src[i]) * std::hint::black_box(0.3f32);
        want[t as usize] += y;
    }
    assert_eq!(
        d.download(ob, bound as usize * 4),
        f32_bytes(&want),
        "f32 with alpha"
    );
    // bf16, three fibers of 2,048 along dim 1 of out [3, 500]: every step
    // rounded to bf16, as the kit's add of that storage does
    let (rows, len, width) = (3usize, 2_048usize, 500u64);
    let index: Vec<i64> = (0..rows * len).map(|_| (next() % width) as i64).collect();
    let src_h: Vec<u16> = (0..rows * len)
        .map(|_| bf16_round(((next() % 4_001) as f32 - 2_000.0) / 64.0))
        .collect();
    let out_h: Vec<u16> = (0..rows * width as usize)
        .map(|j| bf16_round(j as f32))
        .collect();
    let halves = |v: &[u16]| -> Vec<u8> { v.iter().flat_map(|h| h.to_le_bytes()).collect() };
    let mut r = request(
        IndexOp::ScatterAdd,
        Dtype::Bf16,
        2,
        [rows as u64, len as u64, 0, 0],
        1,
    );
    r.index = View::new(0, [len as u64, 1, 0, 0]);
    r.src = View::new(0, [len as u64, 1, 0, 0]);
    r.out = View::new(0, [width, 1, 0, 0]);
    r.bound = width;
    assert_eq!(strategy_of(r), IndexStrategy::Grouped);
    let (ib, sb) = (
        d.upload(&i64_bytes(&index), 0),
        d.upload(&halves(&src_h), 0),
    );
    let ob = d.upload(&halves(&out_h), 0);
    let status = d.upload(&[0u8; 16], 0);
    assert_eq!(
        d.run(r, sb, ib, ob, status),
        ffi::ACK_OK,
        "{}",
        last_error()
    );
    let mut want = out_h.clone();
    for row in 0..rows {
        for j in 0..len {
            let at = row * width as usize + index[row * len + j] as usize;
            want[at] = bf16_round(bf16_value(want[at]) + bf16_value(src_h[row * len + j]));
        }
    }
    assert_eq!(
        d.download(ob, out_h.len() * 2),
        halves(&want),
        "bf16, three fibers"
    );
    // uint8: neighbouring destinations share words (the runs write with atomics)
    let (n, bound) = (9_000usize, 777u64);
    let index: Vec<i64> = (0..n).map(|_| (next() % bound) as i64).collect();
    let src: Vec<u8> = (0..n).map(|_| next() as u8).collect();
    let before: Vec<u8> = (0..bound as usize + 1).map(|j| j as u8).collect();
    let r = line(IndexOp::ScatterAdd, Dtype::U8, n as u64, bound);
    assert_eq!(strategy_of(r), IndexStrategy::Grouped);
    let (ib, sb) = (d.upload(&i64_bytes(&index), 0), d.upload(&src, 0));
    let ob = d.upload(&before, 0);
    let status = d.upload(&[0u8; 16], 0);
    assert_eq!(
        d.run(r, sb, ib, ob, status),
        ffi::ACK_OK,
        "{}",
        last_error()
    );
    let mut want = before.clone();
    for (i, &t) in index.iter().enumerate() {
        want[t as usize] = want[t as usize].wrapping_add(src[i]);
    }
    assert_eq!(
        d.download(ob, before.len().div_ceil(4) * 4)[..before.len()].to_vec(),
        want,
        "uint8, and the byte past it"
    );
    assert_eq!(words(&d.download(status, 16)), vec![0; 4]);
}
