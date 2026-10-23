//! Pure geometry, capacity, aliasing and dispatch planning for `intwise/v1`
//! (operation schema 2): integer, bool and float64 pointwise operations over
//! strided views of at most four dimensions, computed as torch's CPU kernels
//! compute them. `kernels/intwise.comp` is the shader; its header states the
//! arithmetic (every value a signed 64-bit integer in 32-bit words, wrapped to
//! the operation's common dtype C) and each operation's c10 rule.
//!
//! With an integer C (schema 1) the family takes the integer storage codes of
//! `convert_ops::Code` (i64, i32, i16, i8, u8 and bool) for every operand, the
//! output and C. What torch itself refuses on bool is refused here by name
//! ([`IntwisePlanError::BoolOperation`]): sub, neg, abs, the shifts, arange
//! and the divisions. With C = f64 (schema 2, the binary64 module built from
//! the same source with `-DF64=1`) x and y may be any storage code, the output
//! is f64 or, for a comparison, bool, and the scalar and alpha words carry an
//! f64's bits ([`check_codes`], [`check_alpha`]).
//!
//! The divisions (div_trunc, div_floor, remainder, fmod) take a SCALAR
//! divisor only: c10 raises ZeroDivisionError for a zero divisor element,
//! which a shader cannot, so a tensor divisor is refused
//! ([`IntwisePlanError::DivisorNotScalar`]), as are a scalar that wraps to 0
//! in C ([`IntwisePlanError::DivideByZero`]) and -1 in an i32 or i64 C, where
//! MIN / -1 overflows C++'s division ([`IntwisePlanError::DivideOverflow`]).
//!
//! An operand the operation does not read (y of a unary operation or of a
//! scalar one, both inputs of fill and arange) is ignored: its code in the
//! block is C, and its binding is never capacity-checked or read. An output
//! may share an input's storage only through the same view and code (in
//! place); any other sharing is refused. A dense 1- or 2-byte output is
//! written a whole 32-bit word per invocation, any other with 32-bit atomics
//! ([`FLAG_ATOMIC`]), as convert/v1 wrote one until 2026-10-01.
//!
//! SCHEMA 2 (2026-09-28) adds the BINARY64 module, `kernels/intwise.comp` built
//! with `-DF64=1`: torch's float64 pointwise arithmetic (C = f64) computed in the
//! same 32-bit integer words, bit for bit as torch's x86 CPU kernels compute it
//! (IEEE 754 binary64, round to nearest even, subnormals kept, x86 SSE's NaN
//! rule), with no float instruction and no shaderFloat64. Its operations are
//! add and sub (alpha +1.0 or -1.0 only, [`IntwisePlanError::FloatAlpha`]), mul,
//! div, reciprocal, neg, abs, fill and the comparisons
//! ([`IntOp::binary64`]); x and y may be any storage code, each widened as torch
//! casts it to f64, and the output is f64, or bool for a comparison
//! ([`IntwisePlanError::FloatOutput`]). The scalar and alpha words carry an
//! f64's bits. div and reciprocal are the binary64 module's alone, and the
//! bitwise operations, the shifts, arange and the integer divisions the
//! integer module's ([`IntwisePlanError::OpNotForCompute`]).
//!
//! SCHEMA 3 (2026-09-28) adds the integer module's SCAN, [`IntOp::Cumsum`]:
//! torch's CPU cumsum along the request's LAST dimension (the row), element k
//! of a row the sum of its elements 0..=k in C, as `kernels/intwise.comp`'s
//! header derives. It is not elementwise: the plan dispatches ONE WORKGROUP A
//! ROW ([`IntwisePlan::units`] is the row count), writes a 1- or 2-byte output
//! with atomics whatever its layout, refuses a bool C
//! ([`IntwisePlanError::BoolOperation`]: torch's CPU kernel is not built for
//! bool) and a row longer than [`SCAN_LIMIT`]
//! ([`IntwisePlanError::ScanTooLong`]), which bounds one dispatch's duration.
//!
//! SCHEMA 4 (2026-09-28) adds the TRIANGLES, [`IntOp::Tril`] and
//! [`IntOp::Triu`]: torch's tril and triu over the request's last two
//! dimensions, elementwise. Element (.., i, j) is x's where j - i <= s (tril)
//! or where j - i >= s (triu), and 0 elsewhere. The request's `scalar` is the
//! diagonal, a signed 64-bit value never wrapped to C, and y is not read.
//! Every integer C takes them, bool included; the adapter sends any other 1-,
//! 2-, 4- or 8-byte dtype as the integer code of its width, since a triangle
//! moves bits and computes none.
//!
//! SCHEMA 5 (2026-09-28) adds the RANKS, [`IntOp::RankAsc`] and
//! [`IntOp::RankDesc`]: each element's position in torch's STABLE sort of its
//! row (the request's last dimension), the count of elements that sort before
//! it plus the equal ones at a lower index, wrapped to an integer C. x may be
//! any storage code, compared as torch compares (floats: -0.0 equal to +0.0,
//! every NaN equal and above +infinity; `kernels/intwise.comp`'s header). Each
//! invocation walks its row, so the plan bounds the row length and the total
//! work ([`RANK_ROW_LIMIT`], [`RANK_WORK_LIMIT`];
//! [`IntwisePlanError::RankTooLong`]), and it refuses an output sharing x's
//! storage at all ([`IntwisePlanError::OutputOverlapsInput`]): every
//! invocation reads the whole row while the others write.
//!
//! This module has no native `Context` or `Buffer` dependency: storage
//! identities and capacities are the caller's declaration, and the native
//! entry takes them from the real buffers immediately before binding.
//!
//! Refusals are by name, in this order: the operation, codes and flags (at
//! decode: an integer C with integer codes, or C = f64 with its operation and
//! output), a bool operation torch refuses, a division's divisor, the binary64
//! module's alpha, the number of dimensions, zero
//! elements, the element count, a scan's row length, a rank's row length and
//! work, capacity for x, y and the output, the output's distinct addresses,
//! and last the output sharing an input's storage.

use std::fmt;

use crate::convert_ops::{addresses_are_distinct, max_element, Code, Storage, View};
use crate::radix_ops::RadixPlan;

pub const ABI: &str = "intwise/v1";
/// The operation schema the C ABI reports for the family (`ffi::ACK_OP_SCHEMA_INTWISE`):
/// 2 since the binary64 module, 3 since the scan, 4 since the triangles, 5 since
/// the ranks (all 2026-09-28), 6 since a long row's ranks are a radix sort
/// (`radix_ops`, 2026-09-29).
pub const SCHEMA: u32 = 6;
pub const MAX_NDIM: usize = 4;
/// `x`, `y`, `out`, in binding order.
pub const BINDINGS: u32 = 3;
pub const WORKGROUP_INVOCATIONS: u32 = 64;
/// Every device offers at least this many workgroups on each dispatch axis.
pub const MAX_GROUPS_X: u32 = 65_535;
/// Every element and word index the shader forms is below this, so its u32
/// arithmetic cannot wrap and an index fits a signed 32-bit int.
pub const WORD_LIMIT: u64 = 1 << 31;
/// The longest row the scan takes. Each of a workgroup's 64 invocations walks
/// at most 2^18 elements of it, twice: the bound keeps one dispatch far below
/// a driver's watchdog, and a longer row is the host's.
pub const SCAN_LIMIT: u64 = 1 << 24;
/// The longest row a rank takes: each invocation compares its element with
/// every element of its row, so a row of 2^16 costs 2^16 steps an invocation
/// (stage E's routing argsort is 8,192 tokens x 8 = 2^16).
pub const RANK_ROW_LIMIT: u64 = 1 << 16;
/// The most comparisons one rank dispatch makes (elements x row length): at
/// the RX 9070 XT's rate this is tens of milliseconds, far below a driver's
/// watchdog, and a larger rank is the host's (DERIVED from the lane count, not
/// measured at the bound).
pub const RANK_WORK_LIMIT: u64 = 1 << 36;
/// Schema 6: a rank row longer than this, with an int64 or int32 output, is
/// ranked by the radix module (`radix_ops`, `kernels/radix.comp`) in a number
/// of passes over its elements, not by the walk's comparison of every pair;
/// [`RANK_ROW_LIMIT`] and [`RANK_WORK_LIMIT`] bound the walk alone. A walk of
/// a row this long takes about a thousand steps an invocation.
pub const RADIX_MIN_ROW: u64 = 1024;

/// Schema 6: whether a request's words alone allow a radix plan -- a rank
/// whose last dimension is longer than [`RADIX_MIN_ROW`], with an int64 or
/// int32 output that is C -- so the native entry can build the radix module
/// before it borrows any buffer. The planner decides; this never says no
/// where the planner could say yes.
pub fn may_plan_radix(
    op: u32,
    ndim: u32,
    shape: [u64; MAX_NDIM],
    out_code: u32,
    compute_code: u32,
) -> bool {
    let rank = op == IntOp::RankAsc as u32 || op == IntOp::RankDesc as u32;
    let wide =
        out_code == compute_code && (out_code == Code::I64 as u32 || out_code == Code::I32 as u32);
    let row = match ndim {
        1..=4 => shape[ndim as usize - 1],
        _ => 0,
    };
    rank && wide && row > RADIX_MIN_ROW
}
pub const PUSH_WORDS: usize = 27;
pub const PUSH_BYTES: usize = PUSH_WORDS * 4;
/// Push word 1, bit 0: a 1- or 2-byte output that is not dense, written one
/// element per invocation with 32-bit atomics (`units == n`). Set by the planner.
pub const FLAG_ATOMIC: u32 = 1;
/// Push word 1, bit 1: y is the scalar in words 21-22, not a tensor.
pub const FLAG_Y_SCALAR: u32 = 2;
/// The C request's own flags: bit 0 asks for y as the scalar.
pub const REQUEST_Y_SCALAR: u32 = 1;

/// The operations, numbered as the shader's `op` field.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum IntOp {
    /// x + alpha*y
    Add = 0,
    /// x - alpha*y
    Sub = 1,
    Mul = 2,
    And = 3,
    Or = 4,
    Xor = 5,
    Shl = 6,
    Shr = 7,
    Eq = 8,
    Ne = 9,
    Lt = 10,
    Le = 11,
    Gt = 12,
    Ge = 13,
    Neg = 14,
    Abs = 15,
    /// every element the scalar
    Fill = 16,
    /// element l (row-major over the shape) is scalar + l*alpha
    Arange = 17,
    /// C++'s truncating x / scalar
    DivTrunc = 18,
    /// c10's div_floor_integer: x / scalar rounded toward negative infinity
    DivFloor = 19,
    /// Python's x % scalar: the result takes the divisor's sign
    Remainder = 20,
    /// C++'s x % scalar: the result takes the dividend's sign
    Fmod = 21,
    /// x / y, IEEE binary64 (the binary64 module only)
    Div = 22,
    /// 1 / x, IEEE binary64 (the binary64 module only)
    Reciprocal = 23,
    /// the running sum along the last dimension (the integer module's scan,
    /// schema 3): element k of each row is the sum of its elements 0..=k in C
    Cumsum = 24,
    /// x where column - row <= scalar over the last two dimensions, else 0 (schema 4)
    Tril = 25,
    /// x where column - row >= scalar over the last two dimensions, else 0 (schema 4)
    Triu = 26,
    /// each element's position in its row's stable ascending sort (schema 5)
    RankAsc = 27,
    /// each element's position in its row's stable descending sort (schema 5)
    RankDesc = 28,
}

impl IntOp {
    pub const ALL: [IntOp; 29] = [
        Self::Add,
        Self::Sub,
        Self::Mul,
        Self::And,
        Self::Or,
        Self::Xor,
        Self::Shl,
        Self::Shr,
        Self::Eq,
        Self::Ne,
        Self::Lt,
        Self::Le,
        Self::Gt,
        Self::Ge,
        Self::Neg,
        Self::Abs,
        Self::Fill,
        Self::Arange,
        Self::DivTrunc,
        Self::DivFloor,
        Self::Remainder,
        Self::Fmod,
        Self::Div,
        Self::Reciprocal,
        Self::Cumsum,
        Self::Tril,
        Self::Triu,
        Self::RankAsc,
        Self::RankDesc,
    ];

    /// Whether the operation reads x.
    pub fn reads_x(self) -> bool {
        !matches!(self, Self::Fill | Self::Arange)
    }

    /// Whether the operation has a y operand (a tensor or the scalar). The
    /// triangles' scalar is their diagonal, not a y.
    pub fn has_y(self) -> bool {
        self.reads_x()
            && !matches!(
                self,
                Self::Neg
                    | Self::Abs
                    | Self::Reciprocal
                    | Self::Cumsum
                    | Self::Tril
                    | Self::Triu
                    | Self::RankAsc
                    | Self::RankDesc
            )
    }

    /// The scan, dispatched one workgroup a row rather than one invocation an element.
    pub fn is_scan(self) -> bool {
        matches!(self, Self::Cumsum)
    }

    /// The ranks, whose x may be any storage code and whose invocations each
    /// read a whole row.
    pub fn is_rank(self) -> bool {
        matches!(self, Self::RankAsc | Self::RankDesc)
    }

    /// The comparisons, whose result is 0 or 1 (bool out of the binary64 module).
    pub fn is_comparison(self) -> bool {
        matches!(
            self,
            Self::Eq | Self::Ne | Self::Lt | Self::Le | Self::Gt | Self::Ge
        )
    }

    /// The operations of the binary64 module (C = f64, schema 2).
    pub fn binary64(self) -> bool {
        self.is_comparison()
            || matches!(
                self,
                Self::Add
                    | Self::Sub
                    | Self::Mul
                    | Self::Div
                    | Self::Reciprocal
                    | Self::Neg
                    | Self::Abs
                    | Self::Fill
            )
    }

    /// The operations of the integer module (an integer or bool C).
    pub fn integer(self) -> bool {
        !matches!(self, Self::Div | Self::Reciprocal)
    }

    /// The operations torch refuses on a bool C (cumsum: its CPU kernel is not
    /// built for bool and raises).
    pub fn refused_on_bool(self) -> bool {
        self.is_division()
            || matches!(
                self,
                Self::Sub
                    | Self::Neg
                    | Self::Abs
                    | Self::Shl
                    | Self::Shr
                    | Self::Arange
                    | Self::Cumsum
            )
    }

    /// The divisions, whose y must be a nonzero scalar.
    pub fn is_division(self) -> bool {
        matches!(
            self,
            Self::DivTrunc | Self::DivFloor | Self::Remainder | Self::Fmod
        )
    }
}

impl TryFrom<u32> for IntOp {
    type Error = IntwisePlanError;

    fn try_from(word: u32) -> Result<Self, Self::Error> {
        Self::ALL
            .get(word as usize)
            .copied()
            .ok_or(IntwisePlanError::OpOutOfRange)
    }
}

/// `v` as C holds it: truncated to C's width and extended back, bool nonzero.
/// The shader wraps the scalar the same way before it reads it.
pub fn wrap_to(code: Code, v: i64) -> i64 {
    match code {
        Code::I32 => v as i32 as i64,
        Code::I16 => v as i16 as i64,
        Code::I8 => v as i8 as i64,
        Code::U8 => v as u8 as i64,
        Code::Bool => (v != 0) as i64,
        _ => v,
    }
}

/// A division's divisor: the scalar (not a tensor), nonzero in C, and not -1
/// in an i32 or i64 C. Any other operation passes. The C entry decodes this
/// before any device lookup, and the plan checks it again.
pub fn check_divisor(
    op: IntOp,
    compute: Code,
    y_scalar: bool,
    scalar: i64,
) -> Result<(), IntwisePlanError> {
    if !op.is_division() {
        return Ok(());
    }
    if !y_scalar {
        return Err(IntwisePlanError::DivisorNotScalar);
    }
    let divisor = wrap_to(compute, scalar);
    if divisor == 0 {
        return Err(IntwisePlanError::DivideByZero);
    }
    if divisor == -1 && matches!(compute, Code::I32 | Code::I64) {
        return Err(IntwisePlanError::DivideOverflow);
    }
    Ok(())
}

/// +1.0 and -1.0 as the binary64 module's alpha word carries them: an f64's bits.
pub const F64_PLUS_ONE: i64 = 0x3FF0_0000_0000_0000;
pub const F64_MINUS_ONE: i64 = 0xBFF0_0000_0000_0000_u64 as i64;

/// The storage codes a request may carry: an integer C with every code an
/// integer one (schema 1), or C = f64 (schema 2), the binary64 module, whose x
/// and y may be any storage code (each widened exactly as torch casts it) and
/// whose output is f64 for an arithmetic operation and bool for a comparison.
/// A rank (schema 5) is the integer module's and compares its x in any storage
/// code, so only its y, output and C must be integer codes. An operation the
/// C's module does not have is refused by name.
pub fn check_codes(
    op: IntOp,
    x: Code,
    y: Code,
    out: Code,
    compute: Code,
) -> Result<(), IntwisePlanError> {
    if compute == Code::F64 {
        if !op.binary64() {
            return Err(IntwisePlanError::OpNotForCompute);
        }
        let wanted = if op.is_comparison() {
            Code::Bool
        } else {
            Code::F64
        };
        if out != wanted {
            return Err(IntwisePlanError::FloatOutput);
        }
        // every storage code is one the module widens: nothing to refuse in x or y
        let _ = (x, y);
        return Ok(());
    }
    // x is the first: a rank's x may be any storage code
    for (slot, code) in [x, y, out, compute].into_iter().enumerate() {
        if !is_integer(code) && !(slot == 0 && op.is_rank()) {
            return Err(IntwisePlanError::CodeNotInteger);
        }
    }
    if !op.integer() {
        return Err(IntwisePlanError::OpNotForCompute);
    }
    Ok(())
}

/// The binary64 module's add and sub take alpha +1.0 or -1.0 only. torch's CPU
/// computes x + alpha*y as ONE fused multiply-add in its vectorized loop and
/// with two roundings in its scalar one, so for another alpha its own answer
/// depends on the path, and there is no single answer to match.
pub fn check_alpha(op: IntOp, compute: Code, alpha: i64) -> Result<(), IntwisePlanError> {
    if compute == Code::F64
        && matches!(op, IntOp::Add | IntOp::Sub)
        && alpha != F64_PLUS_ONE
        && alpha != F64_MINUS_ONE
    {
        return Err(IntwisePlanError::FloatAlpha);
    }
    Ok(())
}

/// Whether a storage code is one the integer module computes on.
pub fn is_integer(code: Code) -> bool {
    matches!(
        code,
        Code::I64 | Code::I32 | Code::I16 | Code::I8 | Code::U8 | Code::Bool
    )
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IntwisePlanError {
    OpOutOfRange,
    CodeOutOfRange,
    /// A float storage code where C is an integer type, or a float C other than
    /// f64: the f32 family and the adapter's composition own those.
    CodeNotInteger,
    /// An operation the C's module does not have: div or reciprocal with an
    /// integer C, or an integer-only operation (bitwise, shifts, arange, the
    /// integer divisions) with C = f64.
    OpNotForCompute,
    /// C = f64 with an output that is not f64 (arithmetic) or bool (a comparison).
    FloatOutput,
    /// C = f64 add or sub with an alpha other than +1.0 or -1.0.
    FloatAlpha,
    FlagsOutOfRange,
    /// sub, neg, abs, a shift, arange, a division or cumsum with a bool C,
    /// which torch refuses.
    BoolOperation,
    /// A division whose divisor is a tensor, which could hold a zero.
    DivisorNotScalar,
    /// A division by a scalar that is 0 in C: c10 raises ZeroDivisionError.
    DivideByZero,
    /// A division by -1 in an i32 or i64 C, where MIN / -1 overflows.
    DivideOverflow,
    NdimOutOfRange,
    ZeroElements,
    /// A scan row (the last dimension) longer than [`SCAN_LIMIT`].
    ScanTooLong,
    /// A rank row longer than [`RANK_ROW_LIMIT`], or more comparisons than
    /// [`RANK_WORK_LIMIT`].
    RankTooLong,
    AddressOverflow,
    OutputSelfOverlap,
    OutputOverlapsInput,
}

impl fmt::Display for IntwisePlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::OpOutOfRange => "intwise-op-out-of-range",
            Self::CodeOutOfRange => "intwise-code-out-of-range",
            Self::CodeNotInteger => "intwise-code-not-integer",
            Self::OpNotForCompute => "intwise-op-not-for-compute",
            Self::FloatOutput => "intwise-float-output",
            Self::FloatAlpha => "intwise-float-alpha",
            Self::FlagsOutOfRange => "intwise-flags-out-of-range",
            Self::BoolOperation => "intwise-bool-operation",
            Self::DivisorNotScalar => "intwise-divisor-not-scalar",
            Self::DivideByZero => "intwise-divide-by-zero",
            Self::DivideOverflow => "intwise-divide-overflow",
            Self::NdimOutOfRange => "intwise-ndim-out-of-range",
            Self::ZeroElements => "intwise-zero-elements",
            Self::ScanTooLong => "intwise-scan-too-long",
            Self::RankTooLong => "intwise-rank-too-long",
            Self::AddressOverflow => "intwise-address-overflow",
            Self::OutputSelfOverlap => "intwise-output-self-overlap",
            Self::OutputOverlapsInput => "intwise-output-overlaps-input",
        })
    }
}

impl std::error::Error for IntwisePlanError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IntwiseRequest {
    pub op: IntOp,
    pub x: Code,
    pub y: Code,
    pub out: Code,
    /// C, the common dtype torch computes in.
    pub compute: Code,
    /// y is `scalar`, not a tensor.
    pub y_scalar: bool,
    pub ndim: usize,
    pub shape: [u64; MAX_NDIM],
    pub x_view: View,
    pub y_view: View,
    pub out_view: View,
    /// y when `y_scalar`; fill's value; arange's start. With C = f64, an f64's bits.
    pub scalar: i64,
    /// add and sub's alpha; arange's step. With C = f64, an f64's bits (+-1.0).
    pub alpha: i64,
}

/// Storage in binding order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Bindings {
    pub x: Storage,
    pub y: Storage,
    pub out: Storage,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IntwisePlan {
    request: IntwiseRequest,
    words: [u32; PUSH_WORDS],
    groups: [u32; 3],
    /// Schema 6: a long row's ranks, planned for the radix module instead of
    /// this family's own dispatch ([`IntwisePlan::radix`]).
    radix: Option<RadixPlan>,
}

/// Checks `view` over `extents` against the word limit and `storage`'s size:
/// every element the view reaches must end inside it.
fn check_capacity(
    view: &View,
    extents: &[u64; MAX_NDIM],
    element_bytes: u64,
    storage: &Storage,
) -> Result<(), IntwisePlanError> {
    let max = max_element(view, extents).ok_or(IntwisePlanError::AddressOverflow)?;
    let end_bytes = max
        .checked_add(1)
        .and_then(|elements| elements.checked_mul(element_bytes))
        .ok_or(IntwisePlanError::AddressOverflow)?;
    let last_word = (end_bytes - 1) / 4;
    if max >= WORD_LIMIT || last_word >= WORD_LIMIT || end_bytes > storage.capacity_bytes {
        return Err(IntwisePlanError::AddressOverflow);
    }
    Ok(())
}

/// The strides of `view` for the words: zero on an extent-1 dimension, so a
/// decoded block re-encodes to the same words.
fn stride_words(strides: &[u64; MAX_NDIM], extents: &[u64; MAX_NDIM]) -> [u32; MAX_NDIM] {
    let mut words = [0u32; MAX_NDIM];
    for k in 0..MAX_NDIM {
        if extents[k] > 1 {
            words[k] = strides[k] as u32;
        }
    }
    words
}

impl IntwisePlan {
    pub fn new(request: IntwiseRequest, bindings: Bindings) -> Result<Self, IntwisePlanError> {
        let op = request.op;
        let c = request.compute;
        check_codes(op, request.x, request.y, request.out, c)?;
        if c == Code::Bool && op.refused_on_bool() {
            return Err(IntwisePlanError::BoolOperation);
        }
        check_divisor(op, c, request.y_scalar, request.scalar)?;
        check_alpha(op, c, request.alpha)?;
        let ndim = request.ndim;
        if ndim == 0 || ndim > MAX_NDIM {
            return Err(IntwisePlanError::NdimOutOfRange);
        }
        let reads_x = op.reads_x();
        let y_scalar = op.has_y() && request.y_scalar;
        let reads_y = op.has_y() && !request.y_scalar;
        // pad at the front, so the row-major order of the coordinates is unchanged
        let pad = MAX_NDIM - ndim;
        let mut extents = [1u64; MAX_NDIM];
        let mut x_st = [0u64; MAX_NDIM];
        let mut y_st = [0u64; MAX_NDIM];
        let mut out_st = [0u64; MAX_NDIM];
        extents[pad..].copy_from_slice(&request.shape[..ndim]);
        x_st[pad..].copy_from_slice(&request.x_view.strides[..ndim]);
        y_st[pad..].copy_from_slice(&request.y_view.strides[..ndim]);
        out_st[pad..].copy_from_slice(&request.out_view.strides[..ndim]);
        if extents.contains(&0) {
            return Err(IntwisePlanError::ZeroElements);
        }
        let n = extents
            .iter()
            .try_fold(1u64, |acc, &extent| acc.checked_mul(extent))
            .filter(|&n| n < WORD_LIMIT)
            .ok_or(IntwisePlanError::AddressOverflow)?;
        // the scan's row is the last dimension, padded or not
        let scan = op.is_scan();
        if scan && extents[MAX_NDIM - 1] > SCAN_LIMIT {
            return Err(IntwisePlanError::ScanTooLong);
        }
        let x_view = View::new(request.x_view.offset, x_st);
        let y_view = View::new(request.y_view.offset, y_st);
        let out_view = View::new(request.out_view.offset, out_st);
        // A rank's row is the last dimension too. A long row with an int64 or
        // int32 output is the radix module's (schema 6), which costs passes
        // over its elements; any other rank walks its row from every element,
        // so its row and total work are bounded.
        let radix = if op.is_rank() && extents[MAX_NDIM - 1] > RADIX_MIN_ROW && request.out == c {
            RadixPlan::new(
                extents,
                request.x,
                x_view,
                request.out,
                out_view,
                op == IntOp::RankDesc,
            )
        } else {
            None
        };
        if op.is_rank()
            && radix.is_none()
            && (extents[MAX_NDIM - 1] > RANK_ROW_LIMIT
                || n.saturating_mul(extents[MAX_NDIM - 1]) > RANK_WORK_LIMIT)
        {
            return Err(IntwisePlanError::RankTooLong);
        }
        if reads_x {
            check_capacity(&x_view, &extents, request.x.bytes(), &bindings.x)?;
        }
        if reads_y {
            check_capacity(&y_view, &extents, request.y.bytes(), &bindings.y)?;
        }
        check_capacity(&out_view, &extents, request.out.bytes(), &bindings.out)?;
        let es = request.out.bytes();
        let dense = View::contiguous(0, &extents);
        let row_major = (0..MAX_NDIM).all(|k| extents[k] == 1 || out_st[k] == dense.strides[k]);
        // a dense sub-word output takes whole words; any other takes atomics,
        // and so does every sub-word output of the scan, whose chunks of a row
        // (and whose rows) can meet inside one word
        let whole_words = !scan && es < 4 && row_major && request.out_view.offset % (4 / es) == 0;
        if !whole_words && !addresses_are_distinct(&out_view, &extents) {
            return Err(IntwisePlanError::OutputSelfOverlap);
        }
        // An input the output shares storage with must be the output itself:
        // the same code and the same view. The shader reads every element an
        // invocation writes before writing it, and only those.
        let same_view = |code: Code, view: &View| {
            code == request.out
                && view.offset == out_view.offset
                && (0..MAX_NDIM).all(|k| extents[k] == 1 || view.strides[k] == out_st[k])
        };
        // A rank's invocation reads its whole row while the others write, so its
        // output shares nothing with x, not even through x's own view.
        if reads_x
            && bindings.x.identity == bindings.out.identity
            && (op.is_rank() || !same_view(request.x, &x_view))
        {
            return Err(IntwisePlanError::OutputOverlapsInput);
        }
        if reads_y && bindings.y.identity == bindings.out.identity && !same_view(request.y, &y_view)
        {
            return Err(IntwisePlanError::OutputOverlapsInput);
        }
        let atomic = es < 4 && !whole_words;
        let units = if scan {
            n / extents[MAX_NDIM - 1] // the rows
        } else if whole_words {
            n.div_ceil(4 / es)
        } else {
            n
        };
        // A whole-word sub-word output carries its row-major strides on every
        // dimension (the shader checks them all).
        let out_words = if whole_words {
            let mut w = [0u32; MAX_NDIM];
            for (k, word) in w.iter_mut().enumerate() {
                *word = dense.strides[k] as u32;
            }
            w
        } else {
            stride_words(&out_st, &extents)
        };
        let x_code = if reads_x { request.x } else { c };
        let y_code = if reads_y { request.y } else { c };
        let x_words = if reads_x {
            stride_words(&x_st, &extents)
        } else {
            [0; MAX_NDIM]
        };
        let y_words = if reads_y {
            stride_words(&y_st, &extents)
        } else {
            [0; MAX_NDIM]
        };
        let mut words = [0u32; PUSH_WORDS];
        words[0] = x_code as u32
            | (y_code as u32) << 4
            | (request.out as u32) << 8
            | (c as u32) << 12
            | (op as u32) << 16;
        words[1] =
            (if atomic { FLAG_ATOMIC } else { 0 }) | (if y_scalar { FLAG_Y_SCALAR } else { 0 });
        for k in 0..MAX_NDIM {
            words[2 + k] = extents[k] as u32;
            words[7 + k] = x_words[k];
            words[12 + k] = y_words[k];
            words[17 + k] = out_words[k];
        }
        words[6] = if reads_x {
            request.x_view.offset as u32
        } else {
            0
        };
        words[11] = if reads_y {
            request.y_view.offset as u32
        } else {
            0
        };
        words[16] = request.out_view.offset as u32;
        let scalar = request.scalar as u64;
        let alpha = request.alpha as u64;
        words[21] = scalar as u32;
        words[22] = (scalar >> 32) as u32;
        words[23] = alpha as u32;
        words[24] = (alpha >> 32) as u32;
        words[25] = units as u32;
        words[26] = n as u32;
        // one workgroup a row for the scan, else one invocation a unit
        let total_groups = if scan {
            units
        } else {
            units.div_ceil(u64::from(WORKGROUP_INVOCATIONS))
        };
        let gx = total_groups.min(u64::from(MAX_GROUPS_X)).max(1);
        let gy = total_groups.div_ceil(gx).max(1);
        Ok(Self {
            request,
            words,
            groups: [gx as u32, gy as u32, 1],
            radix,
        })
    }

    /// A request in the C ABI's words (`ffi::AckIntwiseRequest`): the
    /// operation, storage codes and flags decoded by name, then planned
    /// against the bindings.
    #[allow(clippy::too_many_arguments)]
    pub fn from_abi(
        op: u32,
        codes: [u32; 4],
        ndim: u32,
        flags: u32,
        shape: [u64; MAX_NDIM],
        views: [View; 3],
        scalar: i64,
        alpha: i64,
        bindings: Bindings,
    ) -> Result<Self, IntwisePlanError> {
        let op = IntOp::try_from(op)?;
        let code = |word: u32| Code::try_from(word).map_err(|_| IntwisePlanError::CodeOutOfRange);
        let [x, y, out, compute] = [
            code(codes[0])?,
            code(codes[1])?,
            code(codes[2])?,
            code(codes[3])?,
        ];
        if flags & !REQUEST_Y_SCALAR != 0 {
            return Err(IntwisePlanError::FlagsOutOfRange);
        }
        let request = IntwiseRequest {
            op,
            x,
            y,
            out,
            compute,
            y_scalar: flags & REQUEST_Y_SCALAR != 0,
            ndim: usize::try_from(ndim).map_err(|_| IntwisePlanError::NdimOutOfRange)?,
            shape,
            x_view: views[0],
            y_view: views[1],
            out_view: views[2],
            scalar,
            alpha,
        };
        Self::new(request, bindings)
    }

    pub fn request(&self) -> &IntwiseRequest {
        &self.request
    }

    /// Invocations with work: output words for a dense 1- or 2-byte output,
    /// else output elements; for the scan, its rows, one workgroup each.
    pub fn units(&self) -> u32 {
        self.words[25]
    }

    pub fn words(&self) -> [u32; PUSH_WORDS] {
        self.words
    }

    pub fn push_constants(&self) -> [u8; PUSH_BYTES] {
        let mut bytes = [0u8; PUSH_BYTES];
        for (chunk, word) in bytes.chunks_exact_mut(4).zip(self.words) {
            chunk.copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }

    pub fn dispatch_groups(&self) -> [u32; 3] {
        self.groups
    }

    /// Schema 6: a rank over rows longer than [`RADIX_MIN_ROW`] with an int64
    /// or int32 output runs as this radix plan's passes (over the native
    /// entry's scratch) instead of this plan's one dispatch.
    pub fn radix(&self) -> Option<&RadixPlan> {
        self.radix.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const X: Storage = Storage::new(1, 1 << 20);
    const Y: Storage = Storage::new(2, 1 << 20);
    const OUT: Storage = Storage::new(3, 1 << 20);

    fn bindings() -> Bindings {
        Bindings {
            x: X,
            y: Y,
            out: OUT,
        }
    }

    /// x [4, 6] transposed (a view of a [6, 4] tensor), y broadcast along the
    /// rows, into a contiguous out.
    fn request(op: IntOp, c: Code) -> IntwiseRequest {
        IntwiseRequest {
            op,
            x: c,
            y: c,
            out: c,
            compute: c,
            y_scalar: false,
            ndim: 2,
            shape: [4, 6, 0, 0],
            x_view: View::new(0, [1, 4, 0, 0]),
            y_view: View::new(3, [0, 1, 0, 0]),
            out_view: View::new(0, [6, 1, 0, 0]),
            scalar: -5,
            alpha: 2,
        }
    }

    const INTEGERS: [Code; 6] = [
        Code::I64,
        Code::I32,
        Code::I16,
        Code::I8,
        Code::U8,
        Code::Bool,
    ];

    #[test]
    fn every_operation_is_planned_on_every_integer_code_but_what_torch_refuses_on_bool() {
        for op in IntOp::ALL {
            for c in INTEGERS {
                let mut r = request(op, c);
                // a division's divisor is the scalar (-5: nonzero in every C)
                r.y_scalar = op.is_division();
                let planned = IntwisePlan::new(r, bindings());
                if !op.integer() {
                    assert_eq!(
                        planned,
                        Err(IntwisePlanError::OpNotForCompute),
                        "{op:?} {c:?}"
                    );
                } else if c == Code::Bool && op.refused_on_bool() {
                    assert_eq!(
                        planned,
                        Err(IntwisePlanError::BoolOperation),
                        "{op:?} {c:?}"
                    );
                } else {
                    let plan = planned.unwrap_or_else(|e| panic!("{op:?} {c:?}: {e}"));
                    assert_eq!(plan.words()[0] >> 16, op as u32);
                    assert_eq!((plan.words()[0] >> 12) & 0xF, c as u32);
                }
            }
        }
    }

    #[test]
    fn a_division_takes_a_nonzero_scalar_divisor_that_cannot_overflow() {
        for op in [
            IntOp::DivTrunc,
            IntOp::DivFloor,
            IntOp::Remainder,
            IntOp::Fmod,
        ] {
            // a tensor divisor could hold a zero, which c10 raises for
            assert_eq!(
                IntwisePlan::new(request(op, Code::I64), bindings()).unwrap_err(),
                IntwisePlanError::DivisorNotScalar,
                "{op:?}"
            );
            for (c, divisor, refused) in [
                (Code::I64, 0, Some(IntwisePlanError::DivideByZero)),
                // zero once wrapped to C: 256 is 0 in u8, 65536 in i16
                (Code::U8, 256, Some(IntwisePlanError::DivideByZero)),
                (Code::I16, 65536, Some(IntwisePlanError::DivideByZero)),
                // MIN / -1 overflows C++'s division in i32 and i64 only
                (Code::I64, -1, Some(IntwisePlanError::DivideOverflow)),
                (Code::I32, -1, Some(IntwisePlanError::DivideOverflow)),
                (
                    Code::I32,
                    (1i64 << 32) - 1,
                    Some(IntwisePlanError::DivideOverflow),
                ),
                (Code::I16, -1, None),
                (Code::I8, -1, None),
                (Code::U8, -1, None),
                (Code::I64, i64::MIN, None),
                (Code::I32, 7, None),
                (Code::Bool, 1, Some(IntwisePlanError::BoolOperation)),
            ] {
                let mut r = request(op, c);
                r.y_scalar = true;
                r.scalar = divisor;
                let planned = IntwisePlan::new(r, bindings());
                match refused {
                    Some(name) => assert_eq!(planned.unwrap_err(), name, "{op:?} {c:?} {divisor}"),
                    None => {
                        let plan =
                            planned.unwrap_or_else(|e| panic!("{op:?} {c:?} {divisor}: {e}"));
                        assert_eq!(plan.words()[1] & FLAG_Y_SCALAR, FLAG_Y_SCALAR);
                    }
                }
            }
        }
        assert_eq!(IntOp::try_from(21), Ok(IntOp::Fmod));
        assert_eq!(IntOp::try_from(23), Ok(IntOp::Reciprocal));
        assert_eq!(IntOp::try_from(24), Ok(IntOp::Cumsum));
        assert_eq!(IntOp::try_from(26), Ok(IntOp::Triu));
        assert_eq!(IntOp::try_from(28), Ok(IntOp::RankDesc));
        assert_eq!(IntOp::try_from(29), Err(IntwisePlanError::OpOutOfRange));
    }

    /// Schema 6: a rank row longer than RADIX_MIN_ROW whose output is C, int64
    /// or int32, is the radix module's, past the walk's row and work limits
    /// too, up to radix_ops::MAX_ELEMENTS elements; everything else walks.
    #[test]
    fn a_long_rank_row_with_a_wide_output_is_the_radix_modules() {
        let big = Storage::new(4, 1 << 40);
        let b = Bindings {
            x: big,
            y: big,
            out: Storage::new(5, 1 << 40),
        };
        for op in [IntOp::RankAsc, IntOp::RankDesc] {
            for c in [Code::I64, Code::I32] {
                let mut r = request(op, c);
                r.x = Code::F32;
                r.ndim = 1;
                r.x_view = View::new(0, [1, 0, 0, 0]);
                r.out_view = View::new(0, [1, 0, 0, 0]);
                r.shape = [RADIX_MIN_ROW, 0, 0, 0];
                let plan = IntwisePlan::new(r, b).unwrap();
                assert!(plan.radix().is_none(), "the walk up to RADIX_MIN_ROW");
                r.shape = [RADIX_MIN_ROW + 1, 0, 0, 0];
                let plan = IntwisePlan::new(r, b).unwrap();
                let radix = plan.radix().expect("the radix module above RADIX_MIN_ROW");
                assert_eq!(radix.digit_passes(), 4, "an f32 key is four bytes");
                let codes = radix.passes()[0].words()[5];
                let descending = codes & crate::radix_ops::FLAG_DESCENDING != 0;
                assert_eq!(descending, op == IntOp::RankDesc);
                assert_eq!((codes >> 4) & 0xF, c as u32, "the output's code");
                // past the walk's row limit, as long as the module takes it
                r.shape = [RANK_ROW_LIMIT * 4, 0, 0, 0];
                assert!(IntwisePlan::new(r, b).unwrap().radix().is_some());
                // 2^30 elements are past radix_ops::MAX_ELEMENTS: the walk's work limit refuses
                r.ndim = 2;
                r.shape = [1 << 15, 1 << 15, 0, 0];
                r.x_view = View::new(0, [1 << 15, 1, 0, 0]);
                r.out_view = View::new(0, [1 << 15, 1, 0, 0]);
                assert_eq!(
                    IntwisePlan::new(r, b).unwrap_err(),
                    IntwisePlanError::RankTooLong
                );
                // 64 rows of 2^15: the module, with a row pass
                r.shape = [1 << 6, 1 << 15, 0, 0];
                let plan = IntwisePlan::new(r, b).unwrap();
                assert_eq!(plan.radix().map(|p| p.digit_passes()), Some(5));
            }
            // an output other than C, or a narrow C, walks
            let mut r = request(op, Code::I64);
            r.ndim = 1;
            r.x_view = View::new(0, [1, 0, 0, 0]);
            r.out_view = View::new(0, [1, 0, 0, 0]);
            r.shape = [RADIX_MIN_ROW * 4, 0, 0, 0];
            r.compute = Code::I16;
            if let Ok(plan) = IntwisePlan::new(r, b) {
                assert!(plan.radix().is_none());
            }
            let mut r = request(op, Code::I16);
            r.ndim = 1;
            r.x_view = View::new(0, [1, 0, 0, 0]);
            r.out_view = View::new(0, [1, 0, 0, 0]);
            r.shape = [RADIX_MIN_ROW * 4, 0, 0, 0];
            assert!(IntwisePlan::new(r, b).unwrap().radix().is_none());
        }
    }

    /// The ranks (schema 5): elementwise, x of ANY storage code while C and
    /// the output stay integer codes, no y, the row and the total work bounded,
    /// and no sharing of x's storage at all, not even in place.
    #[test]
    fn a_rank_takes_any_x_code_bounds_its_work_and_never_shares_x() {
        let every = [
            Code::F32,
            Code::Bf16,
            Code::F16,
            Code::F64,
            Code::I64,
            Code::I32,
            Code::I16,
            Code::I8,
            Code::U8,
            Code::Bool,
        ];
        for op in [IntOp::RankAsc, IntOp::RankDesc] {
            for x in every {
                let mut r = request(op, Code::I64);
                r.x = x;
                let plan =
                    IntwisePlan::new(r, bindings()).unwrap_or_else(|e| panic!("{op:?} {x:?}: {e}"));
                let w = plan.words();
                assert_eq!(w[0] & 0xF, x as u32, "x's own code");
                assert_eq!((w[0] >> 4) & 0xF, Code::I64 as u32, "the unread y is C");
                assert_eq!(w[1], 0, "no y, no atomics");
                assert_eq!(plan.units(), 24, "one invocation an element");
            }
            // the output and C stay integer codes
            let mut r = request(op, Code::I64);
            r.out = Code::F32;
            assert_eq!(
                IntwisePlan::new(r, bindings()).unwrap_err(),
                IntwisePlanError::CodeNotInteger
            );
            // never in place: every invocation reads the whole row
            let mut r = request(op, Code::I64);
            r.x_view = r.out_view;
            let in_place = Bindings {
                x: OUT,
                y: Y,
                out: OUT,
            };
            assert_eq!(
                IntwisePlan::new(r, in_place).unwrap_err(),
                IntwisePlanError::OutputOverlapsInput
            );
            // the row length and the total work bound the WALK: an int16 C
            // is never the radix module's (schema 6), so it keeps schema 5's
            // limits (a_long_rank_row_with_a_wide_output_is_the_radix_modules)
            let big = Storage::new(4, 1 << 40);
            let b = Bindings {
                x: big,
                y: big,
                out: Storage::new(5, 1 << 40),
            };
            let mut r = request(op, Code::I16);
            r.ndim = 1;
            r.shape = [RANK_ROW_LIMIT, 0, 0, 0];
            r.x_view = View::new(0, [1, 0, 0, 0]);
            r.out_view = View::new(0, [1, 0, 0, 0]);
            assert!(IntwisePlan::new(r, b).is_ok());
            r.shape = [RANK_ROW_LIMIT + 1, 0, 0, 0];
            assert_eq!(
                IntwisePlan::new(r, b).unwrap_err(),
                IntwisePlanError::RankTooLong
            );
            // 2^15 rows of 2^15: each row allowed, the work (2^45) not
            r.ndim = 2;
            r.shape = [1 << 15, 1 << 15, 0, 0];
            r.x_view = View::new(0, [1 << 15, 1, 0, 0]);
            r.out_view = View::new(0, [1 << 15, 1, 0, 0]);
            assert_eq!(
                IntwisePlan::new(r, b).unwrap_err(),
                IntwisePlanError::RankTooLong
            );
            r.shape = [1 << 6, 1 << 15, 0, 0];
            assert!(IntwisePlan::new(r, b).is_ok(), "2^36 comparisons exactly");
            assert!(IntwisePlan::new(r, b).unwrap().radix().is_none());
        }
        // a float x is a rank's alone: another integer operation still refuses it
        let mut r = request(IntOp::Add, Code::I64);
        r.x = Code::F32;
        assert_eq!(
            IntwisePlan::new(r, bindings()).unwrap_err(),
            IntwisePlanError::CodeNotInteger
        );
    }

    /// The triangles (schema 4): elementwise over the request's shape, the
    /// diagonal in the scalar words unwrapped (an int64 whatever C is), no y,
    /// every integer C (bool included), in place through the output's view.
    #[test]
    fn the_triangles_carry_their_diagonal_unwrapped_and_read_no_y() {
        for op in [IntOp::Tril, IntOp::Triu] {
            for c in INTEGERS {
                let mut r = request(op, c);
                r.scalar = -(1i64 << 40) - 3;
                let plan =
                    IntwisePlan::new(r, bindings()).unwrap_or_else(|e| panic!("{op:?} {c:?}: {e}"));
                let w = plan.words();
                assert_eq!(w[0] >> 16, op as u32);
                assert_eq!(w[1] & FLAG_Y_SCALAR, 0, "the diagonal is not a y");
                assert_eq!(&w[11..16], &[0; 5], "y is not read");
                assert_eq!(
                    (w[21], w[22]),
                    (r.scalar as u64 as u32, (r.scalar as u64 >> 32) as u32),
                    "{op:?} {c:?}: the diagonal's own 64 bits"
                );
                // elementwise: 24 elements, or the dense sub-word output's whole words
                let units = if c.bytes() < 4 {
                    24 / (4 / c.bytes())
                } else {
                    24
                };
                assert_eq!(u64::from(w[25]), units, "{op:?} {c:?}: elementwise units");
            }
            assert_eq!(
                IntwisePlan::new(request(op, Code::F64), bindings()).unwrap_err(),
                IntwisePlanError::OpNotForCompute
            );
            let mut r = request(op, Code::U8);
            r.x_view = r.out_view;
            let in_place = Bindings {
                x: OUT,
                y: Y,
                out: OUT,
            };
            assert!(IntwisePlan::new(r, in_place).is_ok());
        }
    }

    /// The scan (schema 3): one workgroup a row, the row the last dimension; a
    /// sub-word output with atomics whatever its layout; a bool C refused as
    /// torch refuses it, and a row past SCAN_LIMIT by name.
    #[test]
    fn the_scan_takes_one_workgroup_a_row() {
        // x [4, 6] transposed into a contiguous out: four rows of six
        let plan = IntwisePlan::new(request(IntOp::Cumsum, Code::I64), bindings()).unwrap();
        let w = plan.words();
        assert_eq!(w[0] >> 16, IntOp::Cumsum as u32);
        assert_eq!(
            w[1], 0,
            "no scalar y, and an 8-byte output takes no atomics"
        );
        assert_eq!(&w[11..16], &[0; 5], "y is not read");
        assert_eq!((w[25], w[26]), (4, 24), "units are the rows");
        assert_eq!(plan.dispatch_groups(), [4, 1, 1], "a workgroup a row");
        // a DENSE sub-word output takes atomics all the same: two chunks of a
        // row, or two rows, can meet inside one word
        for c in [Code::I16, Code::I8, Code::U8] {
            let plan = IntwisePlan::new(request(IntOp::Cumsum, c), bindings()).unwrap();
            assert_eq!((plan.words()[1], plan.units()), (FLAG_ATOMIC, 4), "{c:?}");
        }
        assert_eq!(
            IntwisePlan::new(request(IntOp::Cumsum, Code::Bool), bindings()).unwrap_err(),
            IntwisePlanError::BoolOperation
        );
        assert_eq!(
            IntwisePlan::new(request(IntOp::Cumsum, Code::F64), bindings()).unwrap_err(),
            IntwisePlanError::OpNotForCompute
        );
        // a bool x summed in an i64 C, and in place through x's own view
        let mut r = request(IntOp::Cumsum, Code::I64);
        r.x = Code::Bool;
        assert!(IntwisePlan::new(r, bindings()).is_ok());
        let mut r = request(IntOp::Cumsum, Code::I32);
        r.x_view = r.out_view;
        let in_place = Bindings {
            x: OUT,
            y: Y,
            out: OUT,
        };
        assert!(IntwisePlan::new(r, in_place).is_ok());
        // more rows than one grid axis holds wrap into the second
        let big = Storage::new(4, 1 << 40);
        let b = Bindings {
            x: big,
            y: big,
            out: Storage::new(5, 1 << 40),
        };
        let mut r = request(IntOp::Cumsum, Code::I32);
        r.shape = [100_000, 3, 0, 0];
        r.x_view = View::new(0, [3, 1, 0, 0]);
        r.out_view = View::new(0, [3, 1, 0, 0]);
        let plan = IntwisePlan::new(r, b).unwrap();
        assert_eq!(plan.units(), 100_000);
        assert_eq!(plan.dispatch_groups(), [65_535, 2, 1]);
        // the row's length, not the element count, is bounded
        r.ndim = 1;
        r.shape = [SCAN_LIMIT, 0, 0, 0];
        r.x_view = View::new(0, [1, 0, 0, 0]);
        r.out_view = View::new(0, [1, 0, 0, 0]);
        assert_eq!(IntwisePlan::new(r, b).unwrap().units(), 1);
        r.shape = [SCAN_LIMIT + 1, 0, 0, 0];
        assert_eq!(
            IntwisePlan::new(r, b).unwrap_err(),
            IntwisePlanError::ScanTooLong
        );
        r.ndim = 2;
        r.shape = [64, SCAN_LIMIT, 0, 0];
        r.x_view = View::new(0, [SCAN_LIMIT, 1, 0, 0]);
        r.out_view = View::new(0, [SCAN_LIMIT, 1, 0, 0]);
        assert_eq!(IntwisePlan::new(r, b).unwrap().units(), 64);
    }

    /// C = f64: every binary64 operation is planned with operands of any storage
    /// code, the output f64 (bool for a comparison), and the words carry the
    /// f64 bits; what the module does not have is refused by name.
    #[test]
    fn the_binary64_module_plans_its_operations_on_any_operand_code() {
        let every = [
            Code::F32,
            Code::Bf16,
            Code::F16,
            Code::I64,
            Code::I32,
            Code::I16,
            Code::I8,
            Code::U8,
            Code::F64,
            Code::Bool,
        ];
        for op in IntOp::ALL {
            for code in every {
                let mut r = request(op, Code::F64);
                r.x = code;
                r.y = code;
                r.out = if op.is_comparison() {
                    Code::Bool
                } else {
                    Code::F64
                };
                r.alpha = F64_MINUS_ONE;
                let planned = IntwisePlan::new(r, bindings());
                if !op.binary64() {
                    assert_eq!(planned, Err(IntwisePlanError::OpNotForCompute), "{op:?}");
                    continue;
                }
                let plan = planned.unwrap_or_else(|e| panic!("{op:?} {code:?}: {e}"));
                let w = plan.words();
                assert_eq!(w[0] >> 16, op as u32);
                assert_eq!((w[0] >> 12) & 0xF, Code::F64 as u32);
                if op.reads_x() {
                    assert_eq!(w[0] & 0xF, code as u32, "{op:?}");
                }
                assert_eq!((w[23], w[24]), (0, 0xBFF0_0000), "alpha's f64 bits");
            }
        }
        // the output: f64 for arithmetic, bool for a comparison
        let mut r = request(IntOp::Mul, Code::F64);
        r.out = Code::F32;
        assert_eq!(
            IntwisePlan::new(r, bindings()).unwrap_err(),
            IntwisePlanError::FloatOutput
        );
        let mut r = request(IntOp::Lt, Code::F64);
        r.out = Code::F64;
        assert_eq!(
            IntwisePlan::new(r, bindings()).unwrap_err(),
            IntwisePlanError::FloatOutput
        );
        // alpha +1.0 or -1.0 only; mul ignores it
        for (op, alpha, ok) in [
            (IntOp::Add, F64_PLUS_ONE, true),
            (IntOp::Sub, F64_MINUS_ONE, true),
            (IntOp::Add, 0x3FE0_0000_0000_0000, false), // 0.5
            (IntOp::Sub, 1, false),                     // the integer 1 is not 1.0's bits
            (IntOp::Mul, 7, true),
        ] {
            let mut r = request(op, Code::F64);
            r.alpha = alpha;
            let planned = IntwisePlan::new(r, bindings());
            if ok {
                assert!(planned.is_ok(), "{op:?} {alpha:#x}");
            } else {
                assert_eq!(planned.unwrap_err(), IntwisePlanError::FloatAlpha);
            }
        }
    }

    #[test]
    fn a_float_code_is_refused_where_c_is_an_integer_or_a_float_other_than_f64() {
        for float in [Code::F32, Code::Bf16, Code::F16, Code::F64] {
            let mut r = request(IntOp::Add, Code::I32);
            r.x = float;
            assert_eq!(
                IntwisePlan::new(r, bindings()).unwrap_err(),
                IntwisePlanError::CodeNotInteger
            );
            if float != Code::F64 {
                // C = f64 is the binary64 module's; another float C is no module's
                let mut r = request(IntOp::Add, Code::I32);
                r.compute = float;
                assert_eq!(
                    IntwisePlan::new(r, bindings()).unwrap_err(),
                    IntwisePlanError::CodeNotInteger
                );
            }
            let mut r = request(IntOp::Eq, Code::I32);
            r.out = float;
            assert_eq!(
                IntwisePlan::new(r, bindings()).unwrap_err(),
                IntwisePlanError::CodeNotInteger
            );
        }
    }

    #[test]
    fn the_words_mirror_the_shader_block() {
        let mut r = request(IntOp::Sub, Code::I64);
        r.x = Code::I16;
        r.out = Code::I32;
        let plan = IntwisePlan::new(r, bindings()).unwrap();
        let w = plan.words();
        assert_eq!(w[0], 5 | 3 << 4 | 4 << 8 | 3 << 12 | 1 << 16);
        assert_eq!(w[1], 0);
        assert_eq!(&w[2..6], &[1, 1, 4, 6], "padded at the front");
        assert_eq!(&w[6..11], &[0, 0, 0, 1, 4], "x offset then strides");
        assert_eq!(
            &w[11..16],
            &[3, 0, 0, 0, 1],
            "y offset then strides, broadcast rows"
        );
        assert_eq!(&w[16..21], &[0, 0, 0, 6, 1], "out offset then strides");
        assert_eq!(
            (w[21], w[22]),
            (-5i64 as u64 as u32, u32::MAX),
            "the scalar, two's complement"
        );
        assert_eq!((w[23], w[24]), (2, 0), "alpha");
        assert_eq!((w[25], w[26]), (24, 24), "units, elements");
        assert_eq!(plan.dispatch_groups(), [1, 1, 1]);
        assert_eq!(plan.push_constants().len(), PUSH_BYTES);
    }

    #[test]
    fn an_operand_the_operation_does_not_read_is_ignored() {
        // y of a scalar operation, y of a unary one, x and y of fill and arange:
        // their codes become C, their words zero, and their bindings are not checked
        let tiny = Storage::new(1, 0);
        let b = Bindings {
            x: tiny,
            y: tiny,
            out: OUT,
        };
        let mut r = request(IntOp::Fill, Code::I16);
        r.x = Code::U8;
        r.y = Code::I64;
        let plan = IntwisePlan::new(r, b).unwrap();
        assert_eq!(plan.words()[0] & 0xFF, 5 | 5 << 4);
        assert_eq!(&plan.words()[6..16], &[0; 10]);
        let arange = IntwisePlan::new(request(IntOp::Arange, Code::I64), b).unwrap();
        assert_eq!(arange.words()[1], 0);
        let mut r = request(IntOp::Neg, Code::I32);
        r.y = Code::U8;
        let b = Bindings {
            x: X,
            y: tiny,
            out: OUT,
        };
        assert_eq!(IntwisePlan::new(r, b).unwrap().words()[0] & 0xF0, 4 << 4);
        let mut r = request(IntOp::Mul, Code::I32);
        r.y_scalar = true;
        let plan = IntwisePlan::new(r, b).unwrap();
        assert_eq!(plan.words()[1], FLAG_Y_SCALAR);
        assert_eq!(&plan.words()[11..16], &[0; 5]);
        r.y_scalar = false;
        assert_eq!(
            IntwisePlan::new(r, b).unwrap_err(),
            IntwisePlanError::AddressOverflow
        );
    }

    #[test]
    fn a_sub_word_output_is_whole_words_when_dense_and_atomics_otherwise() {
        let mut r = request(IntOp::Eq, Code::I32);
        r.out = Code::Bool;
        let dense = IntwisePlan::new(r, bindings()).unwrap();
        assert_eq!(
            (dense.words()[1], dense.units()),
            (0, 6),
            "24 bools are six words"
        );
        assert_eq!(
            &dense.words()[17..21],
            &[24, 24, 6, 1],
            "every row-major stride"
        );
        r.out_view = View::new(1, [6, 1, 0, 0]);
        let odd = IntwisePlan::new(r, bindings()).unwrap();
        assert_eq!(
            (odd.words()[1], odd.units()),
            (FLAG_ATOMIC, 24),
            "an odd offset: atomics"
        );
        r.out_view = View::new(0, [1, 4, 0, 0]);
        assert_eq!(
            IntwisePlan::new(r, bindings()).unwrap().words()[1],
            FLAG_ATOMIC,
            "transposed"
        );
        r.out_view = View::new(0, [1, 0, 0, 0]);
        assert_eq!(
            IntwisePlan::new(r, bindings()).unwrap_err(),
            IntwisePlanError::OutputSelfOverlap
        );
        let mut r = request(IntOp::Add, Code::I64);
        r.out_view = View::new(0, [1, 0, 0, 0]);
        assert_eq!(
            IntwisePlan::new(r, bindings()).unwrap_err(),
            IntwisePlanError::OutputSelfOverlap
        );
    }

    #[test]
    fn an_output_shares_an_input_only_as_that_input_itself() {
        // in place: x's own view and code
        let mut r = request(IntOp::Add, Code::I16);
        r.x_view = r.out_view;
        let same_x = Bindings {
            x: OUT,
            y: Y,
            out: OUT,
        };
        assert!(IntwisePlan::new(r, same_x).is_ok());
        // another view of the output's storage
        let mut r = request(IntOp::Add, Code::I16);
        assert_eq!(
            IntwisePlan::new(r, same_x).unwrap_err(),
            IntwisePlanError::OutputOverlapsInput
        );
        // the same view read as another code
        r.x_view = r.out_view;
        r.x = Code::U8;
        assert_eq!(
            IntwisePlan::new(r, same_x).unwrap_err(),
            IntwisePlanError::OutputOverlapsInput
        );
        // y broadcast from the output's storage
        let r = request(IntOp::Mul, Code::I32);
        let same_y = Bindings {
            x: X,
            y: OUT,
            out: OUT,
        };
        assert_eq!(
            IntwisePlan::new(r, same_y).unwrap_err(),
            IntwisePlanError::OutputOverlapsInput
        );
        // a scalar y never shares anything
        let mut r = request(IntOp::Mul, Code::I32);
        r.y_scalar = true;
        assert!(IntwisePlan::new(r, same_y).is_ok());
        // x and y may share their own storage
        let r = request(IntOp::Mul, Code::I32);
        assert!(IntwisePlan::new(
            r,
            Bindings {
                x: X,
                y: X,
                out: OUT
            }
        )
        .is_ok());
    }

    #[test]
    fn capacity_is_checked_for_each_operand_in_its_own_element_size() {
        // x's view of [4, 6] transposed reaches element 23: 24 i64 are 192 bytes
        let r = request(IntOp::Add, Code::I64);
        let b = |x: u64| Bindings {
            x: Storage::new(1, x),
            y: Y,
            out: OUT,
        };
        assert_eq!(
            IntwisePlan::new(r, b(191)).unwrap_err(),
            IntwisePlanError::AddressOverflow
        );
        assert!(IntwisePlan::new(r, b(192)).is_ok());
        let mut r = request(IntOp::Eq, Code::I64);
        r.out = Code::U8;
        let o = |out: u64| Bindings {
            x: X,
            y: Y,
            out: Storage::new(3, out),
        };
        assert_eq!(
            IntwisePlan::new(r, o(23)).unwrap_err(),
            IntwisePlanError::AddressOverflow
        );
        assert!(IntwisePlan::new(r, o(24)).is_ok());
    }

    #[test]
    fn the_abi_decode_refuses_by_name() {
        let shape = [4, 6, 0, 0];
        let views = [
            View::new(0, [1, 4, 0, 0]),
            View::new(0, [0, 1, 0, 0]),
            View::new(0, [6, 1, 0, 0]),
        ];
        let plan = |op, codes, ndim, flags| {
            IntwisePlan::from_abi(op, codes, ndim, flags, shape, views, 0, 1, bindings())
        };
        let i32s = [4, 4, 4, 4];
        assert_eq!(
            plan(29, i32s, 2, 0).unwrap_err(),
            IntwisePlanError::OpOutOfRange
        );
        assert!(plan(24, i32s, 2, 0).is_ok(), "cumsum (schema 3)");
        assert!(
            plan(28, [0, 3, 3, 3], 2, 0).is_ok(),
            "a descending rank of f32 in i64 (schema 5)"
        );
        assert!(
            plan(25, [9, 9, 9, 9], 2, 0).is_ok(),
            "tril of bool (schema 4)"
        );
        assert_eq!(
            plan(22, i32s, 2, 0).unwrap_err(),
            IntwisePlanError::OpNotForCompute
        );
        assert_eq!(
            plan(0, [10, 4, 4, 4], 2, 0).unwrap_err(),
            IntwisePlanError::CodeOutOfRange
        );
        assert_eq!(
            plan(0, [4, 4, 4, 0], 2, 0).unwrap_err(),
            IntwisePlanError::CodeNotInteger
        );
        assert_eq!(
            plan(0, i32s, 2, 2).unwrap_err(),
            IntwisePlanError::FlagsOutOfRange
        );
        assert_eq!(
            plan(0, i32s, 0, 0).unwrap_err(),
            IntwisePlanError::NdimOutOfRange
        );
        assert_eq!(
            plan(0, i32s, 5, 0).unwrap_err(),
            IntwisePlanError::NdimOutOfRange
        );
        assert_eq!(
            plan(1, [9, 9, 9, 9], 2, 0).unwrap_err(),
            IntwisePlanError::BoolOperation
        );
        assert!(plan(0, i32s, 2, REQUEST_Y_SCALAR).is_ok());
        let zero = IntwisePlan::from_abi(0, i32s, 2, 0, [4, 0, 0, 0], views, 0, 1, bindings());
        assert_eq!(zero.unwrap_err(), IntwisePlanError::ZeroElements);
    }

    #[test]
    fn element_counts_past_the_word_limit_are_refused() {
        let big = Storage::new(4, u64::MAX);
        let mut r = request(IntOp::Fill, Code::U8);
        r.ndim = 1;
        r.shape = [WORD_LIMIT, 0, 0, 0];
        r.out_view = View::new(0, [1, 0, 0, 0]);
        let b = Bindings {
            x: big,
            y: big,
            out: Storage::new(5, u64::MAX),
        };
        assert_eq!(
            IntwisePlan::new(r, b).unwrap_err(),
            IntwisePlanError::AddressOverflow
        );
    }
}
