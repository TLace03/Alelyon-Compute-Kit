//! pointwise-f32/v1 over BINARY16 storage (operation schema 7, 2026-09-29) at
//! the C ABI (`ack_pointwise`), on the device.
//!
//! The law is exact: an operation over binary16 slots is the SAME f32
//! operation over the widened operands, narrowed once. So every case runs
//! twice on the device and the two must agree in EVERY bit, NaNs included:
//! * natively, over binary16 storage (the schema-7 module the flags select);
//! * over f32 storage (the f32 module, or the bool module with the same bool
//!   slots), with each binary16 operand widened on the host and the result
//!   narrowed on the host, by kernels/convert.comp's two integer routines
//!   ported below.
//!
//! That is the adapter's composed path done on the host, so the claim does not
//! depend on how close any operation is to real arithmetic: it holds for all
//! 43 operations, the transcendental ones included. Every module
//! `pointwise_ops::F16_MODULES` and `F16_BOOL_MODULES` builds is run, the
//! non-bool ones both four at a time (the vector module) and one at a time.
//!
//! Needs a Vulkan device; without one each test prints UNMEASURED and
//! returns (`ACK_REQUIRE_DEVICE=1` makes that a failure). A GREEN RUN WITHOUT
//! THAT VARIABLE IS NOT DEVICE EVIDENCE.
use alelyon_compute_kit::ffi;
use alelyon_compute_kit::pointwise_ops::{
    PointwiseOp, PointwisePlan, Scalars, Storage, StridedView, F16_BOOL_MODULES, F16_MODULES,
    FLAG_F16, FLAG_U8, LAST_OP, OPERANDS, PUSH_BYTES,
};
use alelyon_compute_kit::Buffer;
use std::ffi::{c_char, CStr};

fn last_error() -> String {
    let mut buf = vec![0 as c_char; 1024];
    let rc = unsafe { ffi::ack_last_error(buf.as_mut_ptr(), buf.len()) };
    assert!(rc >= 0, "ack_last_error returned {rc}");
    unsafe { CStr::from_ptr(buf.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

/// kernels/convert.comp's `f16_to_f32`, ported: binary16 -> f32, exact.
fn widen(h: u16) -> f32 {
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

/// kernels/convert.comp's `f32_to_f16`, ported: round to nearest even,
/// subnormals kept, an overflow to infinity, a NaN written sign | 0x7E00.
fn narrow(x: f32) -> u16 {
    let w = x.to_bits();
    let sign = (w >> 16) & 0x8000;
    let e = (w >> 23) & 0xFF;
    let m = w & 0x7F_FFFF;
    if e == 0xFF {
        return (sign | if m != 0 { 0x7E00 } else { 0x7C00 }) as u16;
    }
    let he = e as i32 - 112;
    if he >= 31 {
        return (sign | 0x7C00) as u16;
    }
    if he <= 0 {
        if he < -10 {
            return sign as u16;
        }
        let mf = m | 0x80_0000;
        let shift = (14 - he) as u32;
        let mut q = mf >> shift;
        let rem = mf & ((1 << shift) - 1);
        let half_ulp = 1 << (shift - 1);
        if rem > half_ulp || (rem == half_ulp && q & 1 != 0) {
            q += 1;
        }
        return (sign | q) as u16;
    }
    let mut h = ((he as u32) << 10) | (m >> 13);
    let rem = m & 0x1FFF;
    if rem > 0x1000 || (rem == 0x1000 && h & 1 != 0) {
        h += 1;
    }
    (sign | h) as u16
}

/// binary16 patterns the rounding and the NaN rules are about, then a
/// deterministic spread of bit patterns (every class appears among 2^16)
fn halves(n: usize, seed: u64) -> Vec<u16> {
    let mut v: Vec<u16> = vec![
        0x0000, 0x8000, 0x0001, 0x8001, 0x0002, 0x03FF, 0x83FF, 0x0400, 0x8400, 0x3C00, 0xBC00,
        0x3C01, 0x3BFF, 0x7BFF, 0xFBFF, 0x7BFE, 0x7C00, 0xFC00, 0x7E00, 0xFE00, 0x7C01, 0xFC01,
        0x7DFF, 0x3555, 0x2E66, 0x4248, 0xC248, 0x5640, 0x1000, 0x0200, 0x4900, 0x6BFF,
    ];
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    while v.len() < n {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        // mostly ordinary magnitudes, the full bit space one time in four
        let r = (s >> 16) as u16;
        v.push(if s & 3 == 0 {
            r
        } else {
            (r & 0x83FF) | (0x3000 + ((r >> 4) & 0x0F00))
        });
    }
    v.truncate(n);
    v
}

fn floats(n: usize, seed: u64) -> Vec<f32> {
    let mut v: Vec<f32> = vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        -f32::NAN,
        1e-30,
        -3.0e-8,
        65504.0,
        70000.0,
        0.1,
        1.0 / 3.0,
    ];
    let mut s = seed.wrapping_mul(0xD6E8_FEB8_6659_FD93) | 1;
    while v.len() < n {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let unit = (s >> 40) as f32 / (1u64 << 24) as f32; // [0, 1)
        v.push((unit - 0.5) * 12.0);
    }
    v.truncate(n);
    v
}

/// One slot's contents as the host holds them, per its storage code.
enum Column {
    F32(Vec<f32>),
    Bool(Vec<u8>),
    F16(Vec<u16>),
}

impl Column {
    fn native_bytes(&self) -> Vec<u8> {
        match self {
            Column::F32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Column::Bool(v) => v.clone(),
            Column::F16(v) => v.iter().flat_map(|h| h.to_le_bytes()).collect(),
        }
    }
    /// The reference arm's storage: binary16 widened on the host, the rest as is.
    fn reference_bytes(&self) -> Vec<u8> {
        match self {
            Column::F16(v) => v.iter().flat_map(|&h| widen(h).to_le_bytes()).collect(),
            other => other.native_bytes(),
        }
    }
}

fn column(code: u8, n: usize, seed: u64) -> Column {
    match code {
        0 => Column::F32(floats(n, seed)),
        2 => Column::Bool(
            (0..n)
                .map(|i| ((i * 7 + seed as usize) % 3 == 0) as u8)
                .collect(),
        ),
        3 => Column::F16(halves(n, seed)),
        _ => unreachable!("no bf16 slot here"),
    }
}

fn flags_of(codes: [u8; OPERANDS]) -> u32 {
    (0..OPERANDS)
        .map(|i| match codes[i] {
            2 => FLAG_U8[i],
            3 => FLAG_F16[i],
            _ => 0,
        })
        .sum()
}

/// A block over roomy declared storage; the C entry re-validates it against
/// the buffers really bound.
fn block(op: PointwiseOp, shape: &[u64], scalars: Scalars, flags: u32) -> PointwisePlan {
    let storage = [1, 2, 3, 4].map(|id| Storage::new(id, u64::MAX / 8));
    let dense = StridedView::contiguous(shape).unwrap();
    PointwisePlan::new_with_storage_flags(op, shape, [dense; OPERANDS], scalars, storage, flags)
        .unwrap()
}

fn scalars_for(op: PointwiseOp) -> Scalars {
    match op {
        PointwiseOp::Clamp => Scalars::clamp(Some(-0.5), Some(0.625)),
        PointwiseOp::Iota => Scalars::iota(-3.0, 0.125),
        PointwiseOp::Triu => Scalars::triu(1),
        _ => Scalars::pair(0.75, -1.25),
    }
}

struct Device(*mut ffi::AckDevice);

impl Device {
    fn open(what: &str) -> Option<Device> {
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
        Some(Device(dev))
    }
    fn buffer(&self, bytes: &[u8]) -> *mut Buffer {
        let b = ffi::ack_buffer_alloc(self.0, bytes.len() as u64);
        assert!(!b.is_null(), "{}", last_error());
        let rc = unsafe { ffi::ack_upload(self.0, b, bytes.as_ptr(), bytes.len()) };
        assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
        b
    }
    fn download(&self, b: *mut Buffer, bytes: usize) -> Vec<u8> {
        let mut out = vec![0u8; bytes];
        let rc = unsafe { ffi::ack_download(self.0, b, out.as_mut_ptr(), out.len()) };
        assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
        out
    }
    fn free(&self, b: *mut Buffer) {
        assert_eq!(
            ffi::ack_buffer_free(self.0, b),
            ffi::ACK_OK,
            "{}",
            last_error()
        );
    }
    /// One dispatch; the four slots bound to the four buffers given.
    fn dispatch(&self, plan: &[u8; PUSH_BYTES], slots: [*mut Buffer; OPERANDS]) -> i32 {
        let mut ms = 0f64;
        unsafe {
            ffi::ack_pointwise(
                self.0,
                slots[0],
                slots[1],
                slots[2],
                slots[3],
                plan.as_ptr(),
                plan.len(),
                &mut ms,
            )
        }
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        assert_eq!(ffi::ack_close(self.0), ffi::ACK_OK, "{}", last_error());
    }
}

fn element_bytes(code: u8) -> usize {
    match code {
        2 => 1,
        3 => 2,
        _ => 4,
    }
}

/// The output of one case on both arms, as comparable bits (binary16 outputs
/// of the reference arm narrowed on the host).
fn both_arms(
    dev: &Device,
    op: PointwiseOp,
    shape: &[u64],
    scalars: Scalars,
    codes: [u8; OPERANDS],
    seed: u64,
) -> (Vec<u32>, Vec<u32>, bool) {
    let n: usize = shape.iter().product::<u64>() as usize;
    let inputs: Vec<Column> = (0..3)
        .map(|k| column(codes[k], n, seed + k as u64))
        .collect();
    let reference_codes = codes.map(|c| if c == 3 { 0 } else { c });
    let native = block(op, shape, scalars, flags_of(codes));
    let reference = block(op, shape, scalars, flags_of(reference_codes));
    let out_bytes = n * element_bytes(codes[3]);
    let ref_out_bytes = n * element_bytes(reference_codes[3]);
    let run = |plan: &PointwisePlan, bytes: Vec<Vec<u8>>, out_len: usize| -> Vec<u8> {
        let mut bufs: Vec<*mut Buffer> = bytes.iter().map(|b| dev.buffer(b)).collect();
        bufs.push(dev.buffer(&vec![0xA5u8; out_len]));
        let rc = dev.dispatch(&plan.push_constants(), [bufs[0], bufs[1], bufs[2], bufs[3]]);
        assert_eq!(rc, ffi::ACK_OK, "{op:?} {codes:?}: {}", last_error());
        let out = dev.download(bufs[3], out_len);
        for b in bufs {
            dev.free(b);
        }
        out
    };
    let got = run(
        &native,
        inputs.iter().map(Column::native_bytes).collect(),
        out_bytes,
    );
    let want = run(
        &reference,
        inputs.iter().map(Column::reference_bytes).collect(),
        ref_out_bytes,
    );
    let as_words = |bytes: &[u8], code: u8| -> Vec<u32> {
        match code {
            2 => bytes.iter().map(|&b| u32::from(b)).collect(),
            3 => bytes
                .chunks_exact(2)
                .map(|c| u32::from(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
            _ => bytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        }
    };
    let got_words = as_words(&got, codes[3]);
    let want_words: Vec<u32> = if codes[3] == 3 {
        as_words(&want, 0)
            .into_iter()
            .map(|w| u32::from(narrow(f32::from_bits(w))))
            .collect()
    } else {
        as_words(&want, reference_codes[3])
    };
    (got_words, want_words, native.vector_eligible())
}

fn assert_equal(op: PointwiseOp, codes: [u8; OPERANDS], shape: &[u64], got: &[u32], want: &[u32]) {
    assert_eq!(got.len(), want.len());
    let differ: Vec<usize> = (0..got.len()).filter(|&i| got[i] != want[i]).collect();
    assert!(
        differ.is_empty(),
        "{op:?} over storage {codes:?} at {shape:?}: {} of {} differ, first at {}: native {:#x}, \
         widen/f32/narrow {:#x}",
        differ.len(),
        got.len(),
        differ[0],
        got[differ[0]],
        want[differ[0]]
    );
}

#[test]
fn the_host_routines_are_convert_comp_s_on_every_binary16_and_on_rounding_edges() {
    // widening then narrowing is the identity on every non-NaN binary16, and a
    // NaN comes back sign | 0x7E00, c10's rule
    for h in 0..=u16::MAX {
        let back = narrow(widen(h));
        let nan = (h & 0x7C00) == 0x7C00 && (h & 0x03FF) != 0;
        if nan {
            assert_eq!(back, (h & 0x8000) | 0x7E00, "{h:#06x}");
        } else {
            assert_eq!(back, h, "{h:#06x}");
        }
    }
    // ties to even, the subnormal boundary, overflow
    assert_eq!(narrow(1.0 + 2f32.powi(-11)), 0x3C00); // exact tie: to even
    assert_eq!(narrow(1.0 + 3.0 * 2f32.powi(-11)), 0x3C02); // tie up to even
    assert_eq!(narrow(2f32.powi(-25)), 0x0000); // half the smallest subnormal: to even 0
    assert_eq!(narrow(1.5 * 2f32.powi(-25)), 0x0001);
    assert_eq!(narrow(65519.0), 0x7BFF);
    assert_eq!(narrow(65520.0), 0x7C00);
    assert_eq!(narrow(-0.0), 0x8000);
}

#[test]
fn every_operation_over_every_binary16_module_is_the_widened_f32_one_narrowed_bit_for_bit() {
    let Some(dev) = Device::open("pointwise schema 7, binary16 storage") else {
        return;
    };
    let (mut cases, mut values, mut vector) = (0usize, 0usize, 0usize);
    let modules: Vec<[u8; OPERANDS]> = F16_MODULES
        .iter()
        .chain(F16_BOOL_MODULES.iter())
        .copied()
        .collect();
    for codes in modules {
        for word in 0..=LAST_OP {
            let op = PointwiseOp::try_from(word).unwrap();
            if op == PointwiseOp::Fill {
                continue; // no operand to widen: its own test below
            }
            // 64 x 64 (the vector module where the storage has one) and 67 x 61
            // (one at a time); triu needs its two dimensions, which both have
            for (k, shape) in [[64u64, 64], [67, 61]].iter().enumerate() {
                let (got, want, eligible) = both_arms(
                    &dev,
                    op,
                    shape,
                    scalars_for(op),
                    codes,
                    17 * word as u64 + k as u64,
                );
                assert_equal(op, codes, shape, &got, &want);
                cases += 1;
                values += got.len();
                vector += usize::from(eligible);
            }
            // y the scalar b instead of a buffer, where the operation reads y
            if op.reads_y() && codes[1] == codes[0] {
                let scalars = Scalars::with_scalar_y(0.75, 0.1);
                let (got, want, _) =
                    both_arms(&dev, op, &[64, 64], scalars, codes, 999 + word as u64);
                assert_equal(op, codes, &[64, 64], &got, &want);
                cases += 1;
                values += got.len();
            }
        }
    }
    assert!(vector > 0, "no case reached a vector module");
    eprintln!(
        "binary16 storage (schema 7): {cases} cases over {} modules, {values} results equal to the \
         widened f32 operation narrowed, bit for bit; {vector} of the dispatches vector-eligible",
        F16_MODULES.len() + F16_BOOL_MODULES.len()
    );
}

#[test]
fn a_binary16_fill_is_the_f32_fill_narrowed() {
    let Some(dev) = Device::open("pointwise schema 7, binary16 fill") else {
        return;
    };
    for value in [
        0.1f32,
        -0.0,
        f32::INFINITY,
        f32::NAN,
        65519.0,
        70000.0,
        1e-8,
        6.0e-5,
        1.0 / 3.0,
    ] {
        for shape in [[64u64, 64], [67, 61]] {
            let (got, want, _) = both_arms(
                &dev,
                PointwiseOp::Fill,
                &shape,
                Scalars::new(value),
                [3, 3, 3, 3],
                5,
            );
            assert_equal(PointwiseOp::Fill, [3, 3, 3, 3], &shape, &got, &want);
        }
    }
}
