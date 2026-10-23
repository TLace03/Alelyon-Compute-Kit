//! Fused attention (attention-v2, 2026-09-27; attention-bf16/v1 of 2026-09-22
//! before it, with the same push blocks and grids): the plan the forward kernel
//! `kernels/attention_fwd_v2.comp` runs under. Pure validation, the push block
//! and the dispatch grid; no Vulkan here.
//!
//! The operation is `O = softmax(scale * Q K^T + causal mask) V` with the
//! row-wise log-sum-exp `LSE`, per (batch, query head), over bf16 Q, K, V of
//! head dimension 64 or 128 and bf16 O, f32 LSE. Query head `h` reads kv head
//! `h / (heads / kv_heads)`. Each operand is addressed through its own
//! (batch, head, row) strides in ELEMENTS with the head dimension contiguous, so
//! a `[B, T, H, D]` tensor viewed as `[B, H, T, D]` -- what the model's
//! `.transpose(1, 2)` hands over -- is read in place.
//!
//! What is refused, by name, before anything is dispatched: a head dimension
//! other than 64 or 128 (one module each; the shared geometry is built for
//! them), a zero
//! extent, query heads that are not a multiple of the kv heads, an odd stride
//! (the kernel moves bf16 in pairs), an operand too small for the addresses the
//! shape reaches, an address past `u32` (the kernel computes in `uint`), a grid
//! dimension past Vulkan's guaranteed 65,535, and a scale that is not finite.

use std::fmt;

/// The head dimension attention-bf16/v1 is built for.
pub const HEAD_DIM: u32 = 64;
/// The head dimensions attention-v2 (forward, dQ and dK/dV) is built for, one
/// module each (2026-09-27). A plan accepts these.
pub const HEAD_DIMS: [u32; 2] = [64, 128];
/// Query rows per workgroup (`BR` in the kernel).
pub const BLOCK_ROWS: u32 = 64;
/// q, k, v, o, lse.
pub const OPERANDS: usize = 5;
/// Sixteen `u32` and one `f32`, in the kernel's push block order.
pub const PUSH_BYTES: usize = 68;
/// Vulkan guarantees at least this many workgroups per grid dimension.
pub const MAX_GRID: u64 = 65_535;

/// One operand's strides in elements; the head dimension is contiguous.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Strides {
    pub batch: u64,
    pub head: u64,
    pub row: u64,
}

impl Strides {
    /// The strides of a dense `[batch, heads, seq, HEAD_DIM]` tensor.
    pub fn dense_bhtd(heads: u64, seq: u64) -> Self {
        Self::dense_bhtd_d(heads, seq, HEAD_DIM as u64)
    }

    /// The strides of a dense `[batch, seq, heads, HEAD_DIM]` tensor viewed as
    /// `[batch, heads, seq, HEAD_DIM]`.
    pub fn dense_bthd(heads: u64, seq: u64) -> Self {
        Self::dense_bthd_d(heads, seq, HEAD_DIM as u64)
    }

    /// [`Strides::dense_bhtd`] at head dimension `d`.
    pub fn dense_bhtd_d(heads: u64, seq: u64, d: u64) -> Self {
        Strides {
            batch: heads * seq * d,
            head: seq * d,
            row: d,
        }
    }

    /// [`Strides::dense_bthd`] at head dimension `d`.
    pub fn dense_bthd_d(heads: u64, seq: u64, d: u64) -> Self {
        Strides {
            batch: seq * heads * d,
            head: d,
            row: heads * d,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AttentionShape {
    pub batch: u32,
    pub heads: u32,
    pub kv_heads: u32,
    pub seq: u32,
    pub head_dim: u32,
    pub causal: bool,
    pub scale: f32,
    pub q: Strides,
    pub k: Strides,
    pub v: Strides,
    pub o: Strides,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AttentionPlanError {
    HeadDim(u32),
    ZeroExtent,
    KvHeads {
        heads: u32,
        kv_heads: u32,
    },
    OddStride(&'static str),
    AddressPastU32(&'static str),
    OperandTooSmall {
        operand: &'static str,
        needed: u64,
        have: u64,
    },
    GridTooLarge(&'static str, u64),
    ScaleNotFinite,
}

impl AttentionPlanError {
    /// The refusal's stable name, as the kit's other families name theirs.
    pub fn name(&self) -> &'static str {
        match self {
            Self::HeadDim(_) => "attention-head-dim-unsupported",
            Self::ZeroExtent => "attention-zero-extent",
            Self::KvHeads { .. } => "attention-kv-heads-do-not-divide",
            Self::OddStride(_) => "attention-odd-stride",
            Self::AddressPastU32(_) => "attention-address-past-u32",
            Self::OperandTooSmall { .. } => "attention-operand-too-small",
            Self::GridTooLarge(..) => "attention-grid-too-large",
            Self::ScaleNotFinite => "attention-scale-not-finite",
        }
    }
}

impl fmt::Display for AttentionPlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HeadDim(d) => write!(
                f,
                "{}: head dimension {d}; the kernels are built for {HEAD_DIMS:?}",
                self.name()
            ),
            Self::ZeroExtent => write!(
                f,
                "{}: a batch, head or sequence extent is zero",
                self.name()
            ),
            Self::KvHeads { heads, kv_heads } => write!(
                f,
                "{}: {heads} query heads are not a multiple of {kv_heads} kv heads",
                self.name()
            ),
            Self::OddStride(what) => write!(
                f,
                "{}: a {what} stride is odd (bf16 moves in pairs)",
                self.name()
            ),
            Self::AddressPastU32(what) => write!(
                f,
                "{}: {what} reaches an element index past u32",
                self.name()
            ),
            Self::OperandTooSmall {
                operand,
                needed,
                have,
            } => write!(
                f,
                "{}: {operand} needs {needed} elements and has {have}",
                self.name()
            ),
            Self::GridTooLarge(what, n) => {
                write!(f, "{}: {n} workgroups along {what}", self.name())
            }
            Self::ScaleNotFinite => write!(f, "{}: the scale is not finite", self.name()),
        }
    }
}

impl std::error::Error for AttentionPlanError {}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AttentionPlan {
    shape: AttentionShape,
    groups: [u32; 3],
}

/// The largest element index `(batch-1)*B + (heads-1)*H + (seq-1)*R + D - 1`.
fn last_index(s: Strides, batch: u32, heads: u32, seq: u32, d: u32) -> Option<u64> {
    let b = (batch as u64 - 1).checked_mul(s.batch)?;
    let h = (heads as u64 - 1).checked_mul(s.head)?;
    let r = (seq as u64 - 1).checked_mul(s.row)?;
    b.checked_add(h)?.checked_add(r)?.checked_add(d as u64 - 1)
}

impl AttentionPlan {
    /// Validate `shape` against operands of `lens` elements (q, k, v and o in
    /// bf16 elements, lse in f32 elements).
    pub fn new(shape: AttentionShape, lens: [u64; OPERANDS]) -> Result<Self, AttentionPlanError> {
        use AttentionPlanError as E;
        if !HEAD_DIMS.contains(&shape.head_dim) {
            return Err(E::HeadDim(shape.head_dim));
        }
        if shape.batch == 0 || shape.heads == 0 || shape.kv_heads == 0 || shape.seq == 0 {
            return Err(E::ZeroExtent);
        }
        if shape.heads % shape.kv_heads != 0 {
            return Err(E::KvHeads {
                heads: shape.heads,
                kv_heads: shape.kv_heads,
            });
        }
        if !shape.scale.is_finite() {
            return Err(E::ScaleNotFinite);
        }
        let operands = [
            ("q", shape.q, shape.heads, lens[0]),
            ("k", shape.k, shape.kv_heads, lens[1]),
            ("v", shape.v, shape.kv_heads, lens[2]),
            ("o", shape.o, shape.heads, lens[3]),
        ];
        for (name, s, heads, have) in operands {
            if s.batch % 2 != 0 || s.head % 2 != 0 || s.row % 2 != 0 {
                return Err(E::OddStride(name));
            }
            if s.batch > u32::MAX as u64 || s.head > u32::MAX as u64 || s.row > u32::MAX as u64 {
                return Err(E::AddressPastU32(name));
            }
            let last = last_index(s, shape.batch, heads, shape.seq, shape.head_dim)
                .ok_or(E::AddressPastU32(name))?;
            if last > u32::MAX as u64 {
                return Err(E::AddressPastU32(name));
            }
            if last >= have {
                return Err(E::OperandTooSmall {
                    operand: name,
                    needed: last + 1,
                    have,
                });
            }
        }
        let lse = shape.batch as u64 * shape.heads as u64 * shape.seq as u64;
        if lse > u32::MAX as u64 {
            return Err(E::AddressPastU32("lse"));
        }
        if lse > lens[4] {
            return Err(E::OperandTooSmall {
                operand: "lse",
                needed: lse,
                have: lens[4],
            });
        }
        let blocks = (shape.seq as u64).div_ceil(BLOCK_ROWS as u64);
        for (what, n) in [
            ("query blocks", blocks),
            ("heads", shape.heads as u64),
            ("batch", shape.batch as u64),
        ] {
            if n > MAX_GRID {
                return Err(E::GridTooLarge(what, n));
            }
        }
        Ok(AttentionPlan {
            shape,
            groups: [blocks as u32, shape.heads, shape.batch],
        })
    }

    pub fn shape(&self) -> AttentionShape {
        self.shape
    }

    /// Workgroups: (query blocks, heads, batch).
    pub fn dispatch_groups(&self) -> [u32; 3] {
        self.groups
    }

    /// The kernel's push block: T, H, HKV, causal, then (batch, head, row)
    /// strides of q, k, v and o, then the scale.
    pub fn push_constants(&self) -> [u8; PUSH_BYTES] {
        let s = &self.shape;
        let words: [u32; 16] = [
            s.seq,
            s.heads,
            s.kv_heads,
            s.causal as u32,
            s.q.batch as u32,
            s.q.head as u32,
            s.q.row as u32,
            s.k.batch as u32,
            s.k.head as u32,
            s.k.row as u32,
            s.v.batch as u32,
            s.v.head as u32,
            s.v.row as u32,
            s.o.batch as u32,
            s.o.head as u32,
            s.o.row as u32,
        ];
        let mut out = [0u8; PUSH_BYTES];
        for (i, w) in words.iter().enumerate() {
            out[4 * i..4 * i + 4].copy_from_slice(&w.to_le_bytes());
        }
        out[64..68].copy_from_slice(&s.scale.to_le_bytes());
        out
    }

    /// Multiply-adds of the forward (both products), counting only the key
    /// blocks the kernel visits: the measure a TFLOPS figure is quoted against.
    pub fn flops(&self) -> f64 {
        let s = &self.shape;
        let blocks = (s.seq as u64).div_ceil(BLOCK_ROWS as u64);
        let visited_pairs: u64 = if s.causal {
            (1..=blocks).sum::<u64>()
        } else {
            blocks * blocks
        };
        // per visited (query block, key block) pair: two 64x64xD products
        let per_pair = 2.0 * 2.0 * (BLOCK_ROWS as f64).powi(2) * s.head_dim as f64;
        per_pair * visited_pairs as f64 * s.batch as f64 * s.heads as f64
    }
}

/// Push block of the dQ kernel: T, H, HKV, causal, the strides of q, k, v, o,
/// dO and dQ, then the scale.
pub const DQ_PUSH_BYTES: usize = 92;
/// Push block of the dK/dV kernel: T, H, HKV, causal, the strides of q, k, v,
/// o, dO, dK and dV, then the scale.
pub const DKDV_PUSH_BYTES: usize = 104;
/// q, k, v, o, dO, lse, dQ.
pub const DQ_OPERANDS: usize = 7;
/// q, k, v, o, dO, lse, dK, dV.
pub const DKDV_OPERANDS: usize = 8;

/// The backward's own operands: dO (shaped as o) and the three gradients
/// (dQ as q, dK and dV as k and v).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BackwardStrides {
    pub dout: Strides,
    pub dq: Strides,
    pub dk: Strides,
    pub dv: Strides,
}

/// The backward of one validated forward: FlashAttention-2's two passes,
/// `kernels/attention_bwd_dq_v2.comp` over query blocks and
/// `kernels/attention_bwd_dkdv_v2.comp` over key blocks, both reading the
/// forward's o and lse.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AttentionBackward {
    forward: AttentionPlan,
    strides: BackwardStrides,
}

impl AttentionBackward {
    /// Validate the backward's operands of `lens` elements (dO, dQ, dK, dV,
    /// all bf16) against `forward`, which has already validated q, k, v, o and
    /// lse.
    pub fn new(
        forward: AttentionPlan,
        strides: BackwardStrides,
        lens: [u64; 4],
    ) -> Result<Self, AttentionPlanError> {
        use AttentionPlanError as E;
        let s = forward.shape;
        let operands = [
            ("dO", strides.dout, s.heads, lens[0]),
            ("dQ", strides.dq, s.heads, lens[1]),
            ("dK", strides.dk, s.kv_heads, lens[2]),
            ("dV", strides.dv, s.kv_heads, lens[3]),
        ];
        for (name, st, heads, have) in operands {
            if st.batch % 2 != 0 || st.head % 2 != 0 || st.row % 2 != 0 {
                return Err(E::OddStride(name));
            }
            if st.batch > u32::MAX as u64 || st.head > u32::MAX as u64 || st.row > u32::MAX as u64 {
                return Err(E::AddressPastU32(name));
            }
            let last =
                last_index(st, s.batch, heads, s.seq, s.head_dim).ok_or(E::AddressPastU32(name))?;
            if last > u32::MAX as u64 {
                return Err(E::AddressPastU32(name));
            }
            if last >= have {
                return Err(E::OperandTooSmall {
                    operand: name,
                    needed: last + 1,
                    have,
                });
            }
        }
        Ok(AttentionBackward { forward, strides })
    }

    fn words(&self, tail: &[Strides]) -> Vec<u32> {
        let s = &self.forward.shape;
        let mut w = vec![s.seq, s.heads, s.kv_heads, s.causal as u32];
        for st in [s.q, s.k, s.v, s.o, self.strides.dout].iter().chain(tail) {
            w.extend([st.batch as u32, st.head as u32, st.row as u32]);
        }
        w
    }

    fn pack<const N: usize>(&self, words: Vec<u32>) -> [u8; N] {
        let mut out = [0u8; N];
        for (i, w) in words.iter().enumerate() {
            out[4 * i..4 * i + 4].copy_from_slice(&w.to_le_bytes());
        }
        out[N - 4..].copy_from_slice(&self.forward.shape.scale.to_le_bytes());
        out
    }

    /// The dQ kernel's push block, in its order.
    pub fn dq_push_constants(&self) -> [u8; DQ_PUSH_BYTES] {
        self.pack(self.words(&[self.strides.dq]))
    }

    /// The dK/dV kernel's push block, in its order.
    pub fn dkdv_push_constants(&self) -> [u8; DKDV_PUSH_BYTES] {
        self.pack(self.words(&[self.strides.dk, self.strides.dv]))
    }

    /// dQ workgroups: (query blocks, query heads, batch), as the forward's.
    pub fn dq_groups(&self) -> [u32; 3] {
        self.forward.groups
    }

    /// dK/dV workgroups: (key blocks, kv heads, batch).
    pub fn dkdv_groups(&self) -> [u32; 3] {
        let s = &self.forward.shape;
        [self.forward.groups[0], s.kv_heads, s.batch]
    }

    /// Multiply-adds of both backward passes over the visited blocks: dQ makes
    /// three 64x64x64 products per visited pair (S, dP, dS K), dK/dV four
    /// (S^T, dP^T, P^T dO, dS^T Q).
    pub fn flops(&self) -> f64 {
        self.forward.flops() / 2.0 * 7.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape(b: u32, h: u32, hkv: u32, t: u32) -> AttentionShape {
        AttentionShape {
            batch: b,
            heads: h,
            kv_heads: hkv,
            seq: t,
            head_dim: HEAD_DIM,
            causal: true,
            scale: 0.125,
            q: Strides::dense_bhtd(h as u64, t as u64),
            k: Strides::dense_bhtd(hkv as u64, t as u64),
            v: Strides::dense_bhtd(hkv as u64, t as u64),
            o: Strides::dense_bhtd(h as u64, t as u64),
        }
    }

    fn lens(b: u32, h: u32, hkv: u32, t: u32) -> [u64; OPERANDS] {
        let q = b as u64 * h as u64 * t as u64 * HEAD_DIM as u64;
        let kv = b as u64 * hkv as u64 * t as u64 * HEAD_DIM as u64;
        [q, kv, kv, q, b as u64 * h as u64 * t as u64]
    }

    #[test]
    fn the_registered_geometry_plans_and_its_push_block_is_in_kernel_order() {
        let plan = AttentionPlan::new(shape(4, 8, 4, 1024), lens(4, 8, 4, 1024)).expect("plan");
        assert_eq!(plan.dispatch_groups(), [16, 8, 4]);
        let push = plan.push_constants();
        let word = |i: usize| u32::from_le_bytes(push[4 * i..4 * i + 4].try_into().unwrap());
        assert_eq!((word(0), word(1), word(2), word(3)), (1024, 8, 4, 1));
        assert_eq!((word(4), word(5), word(6)), (8 * 1024 * 64, 1024 * 64, 64));
        assert_eq!(f32::from_le_bytes(push[64..68].try_into().unwrap()), 0.125);
    }

    #[test]
    fn a_transposed_view_is_read_in_place() {
        let mut s = shape(2, 8, 4, 100);
        s.q = Strides::dense_bthd(8, 100);
        s.o = Strides::dense_bthd(8, 100);
        AttentionPlan::new(s, lens(2, 8, 4, 100)).expect("a [B, T, H, D] view plans");
    }

    #[test]
    fn every_refusal_is_named() {
        let ok = shape(1, 2, 1, 64);
        let l = lens(1, 2, 1, 64);
        let mut s = ok;
        s.head_dim = 96;
        assert_eq!(
            AttentionPlan::new(s, l).unwrap_err().name(),
            "attention-head-dim-unsupported"
        );
        let mut s = ok;
        s.seq = 0;
        assert_eq!(
            AttentionPlan::new(s, l).unwrap_err().name(),
            "attention-zero-extent"
        );
        let mut s = ok;
        s.heads = 3;
        s.kv_heads = 2;
        assert_eq!(
            AttentionPlan::new(s, l).unwrap_err().name(),
            "attention-kv-heads-do-not-divide"
        );
        let mut s = ok;
        s.q.row = 65;
        assert_eq!(
            AttentionPlan::new(s, l).unwrap_err().name(),
            "attention-odd-stride"
        );
        let mut s = ok;
        s.scale = f32::NAN;
        assert_eq!(
            AttentionPlan::new(s, l).unwrap_err().name(),
            "attention-scale-not-finite"
        );
        let mut short = l;
        short[1] -= 1;
        assert_eq!(
            AttentionPlan::new(ok, short).unwrap_err().name(),
            "attention-operand-too-small"
        );
        let mut short = l;
        short[4] -= 1;
        assert_eq!(
            AttentionPlan::new(ok, short).unwrap_err().name(),
            "attention-operand-too-small"
        );
        let big = shape(1, 1, 1, 65_536 * 64);
        let bl = lens(1, 1, 1, 65_536 * 64);
        assert_eq!(
            AttentionPlan::new(big, bl).unwrap_err().name(),
            "attention-grid-too-large"
        );
    }

    #[test]
    fn the_backward_push_blocks_are_in_kernel_order_and_its_operands_are_checked() {
        let f = AttentionPlan::new(shape(4, 8, 4, 1024), lens(4, 8, 4, 1024)).unwrap();
        let q = Strides::dense_bhtd(8, 1024);
        let kv = Strides::dense_bhtd(4, 1024);
        let bs = BackwardStrides {
            dout: q,
            dq: q,
            dk: kv,
            dv: kv,
        };
        let l = lens(4, 8, 4, 1024);
        let bw = AttentionBackward::new(f, bs, [l[0], l[0], l[1], l[1]]).expect("backward");
        let dq = bw.dq_push_constants();
        let word = |b: &[u8], i: usize| u32::from_le_bytes(b[4 * i..4 * i + 4].try_into().unwrap());
        // T H HKV causal, then q k v o dO dQ: dQ's row stride is word 21
        assert_eq!((word(&dq, 0), word(&dq, 2), word(&dq, 21)), (1024, 4, 64));
        assert_eq!(f32::from_le_bytes(dq[88..92].try_into().unwrap()), 0.125);
        let dkdv = bw.dkdv_push_constants();
        // ... then dK and dV: dV's head stride is word 23
        assert_eq!(word(&dkdv, 23), 1024 * 64);
        assert_eq!(
            f32::from_le_bytes(dkdv[100..104].try_into().unwrap()),
            0.125
        );
        assert_eq!(bw.dkdv_groups(), [16, 4, 4]);
        let short = AttentionBackward::new(f, bs, [l[0], l[0], l[1] - 1, l[1]]).unwrap_err();
        assert_eq!(short.name(), "attention-operand-too-small");
    }

    #[test]
    fn the_causal_flop_count_visits_the_lower_triangle_of_blocks() {
        let causal = AttentionPlan::new(shape(1, 1, 1, 256), lens(1, 1, 1, 256)).unwrap();
        let mut full = shape(1, 1, 1, 256);
        full.causal = false;
        let full = AttentionPlan::new(full, lens(1, 1, 1, 256)).unwrap();
        // 4 query blocks: 10 visited pairs causal, 16 without
        assert_eq!(causal.flops() * 16.0, full.flops() * 10.0);
    }
}
