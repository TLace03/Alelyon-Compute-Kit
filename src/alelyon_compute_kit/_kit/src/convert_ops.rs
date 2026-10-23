//! Pure geometry, capacity, aliasing and dispatch planning for `convert/v1`
//! (operation schema 2): `out[c] = T_out(in[c])` over strided views of at most
//! four dimensions, between the storage types in [`Code`]. `kernels/convert.comp`
//! is the shader; its header states every conversion's rounding and what c10
//! does that it matches.
//!
//! FLOAT -> INTEGER IS REFUSED BY NAME ([`ConvertPlanError::FloatToInteger`]):
//! c10's `static_cast` from a float to an integer type is undefined behaviour
//! out of range, so there is no reference answer to hold the kernel to. A
//! float -> bool conversion (nonzero) is defined and admitted.
//!
//! A dense 1- or 2-byte output (row-major over the shape, at any element
//! offset since 2026-10-01) is written a whole 32-bit word per unit; its first
//! and last words, which it may fill only partly, keep their other bytes. Any
//! other 1- or 2-byte output is written one element per unit: with a plain
//! read-modify-write of its word when no two elements share a word
//! ([`FLAG_EXCLUSIVE`], 2026-10-01: every stride over an extent above 1 a
//! multiple of 4 / bytes), else with 32-bit atomics ([`FLAG_ATOMIC`], schema
//! 2). Like a 4- or 8-byte output, it may be any view whose coordinates reach
//! distinct addresses.
//!
//! A SAME-TYPE COPY (2026-10-05) is a conversion between one integer code, f32
//! or f64 and itself, which is the identity on every bit pattern. The shader
//! moves its bits through no conversion routine, and stores a 2-byte element of
//! an output that is not dense on its own, with a 16-bit store, instead of the
//! read-modify-write or the atomics its flag names. The plan, its flags and
//! every result are unchanged (`kernels/convert.comp`, SAME-TYPE COPIES).
//!
//! A workgroup of [`WORKGROUP_INVOCATIONS`] invocations takes
//! [`UNITS_PER_WORKGROUP`] consecutive units, [`UNITS_PER_INVOCATION`] an
//! invocation, `WORKGROUP_INVOCATIONS` apart (2026-10-01: the pointwise
//! family's layout; see `kernels/convert.comp`, WORK LAYOUT).
//!
//! A TRANSPOSED READ is tiled ([`FLAG_TILED`], 2026-10-03): a view of two
//! dimensions whose input runs along the output's rows (input stride 1 on the
//! row axis, at least [`MIN_TILED_STRIDE`] on the column axis, at least
//! [`MIN_TILED_ROWS`] rows) into a dense row-major output,
//! with 1- to 4-byte elements on both sides, is dispatched to
//! `kernels/convert_tiled.spv` as a grid of [`TILE`] x [`TILE`] tiles. A 1- or
//! 2-byte output takes it only when its offset and row length are multiples of
//! 4 / bytes, so no output word is shared by two tiles. Every element is the
//! same conversion as on the other paths; only the order of the memory traffic
//! changes (see `kernels/convert.comp`, TRANSPOSED READS).
//!
//! THE OUTPUT'S MEMORY ORDER (2026-10-03). Before it chooses a path, the
//! planner puts the dimensions in the output's memory order (its strides,
//! largest first; extent-1 dimensions dropped) and merges each adjacent pair
//! that composes in both views, so every rule above reads the view in that
//! order. A conversion pairs each coordinate's input element with the same
//! coordinate's output element, and neither step changes a pair; only the order
//! the units take them in changes. What it changes is the path: a transposed
//! output whose input has the same layout -- `x.t().to(dtype)`, to which torch
//! gives x's strides -- is one dense run, written whole words at a time with
//! reads in order, where in the request's order a 1- or 2-byte output took two
//! atomics an element with both its reads and its writes a row apart; and an
//! output transposed against a dense input is a transposed read, tiled above.
//!
//! This module has no native `Context` or `Buffer` dependency: storage
//! identities and capacities are the caller's declaration, and the native
//! entry takes them from the real buffers immediately before binding.
//!
//! Refusals are by name, in this order: the storage codes and flags (at
//! decode), the conversion, the number of dimensions, zero elements, address
//! arithmetic and capacity for the input and then the output, the output's
//! distinct addresses, and last the output sharing the input's storage.

use std::fmt;

pub const ABI: &str = "convert/v1";
/// The operation schema the C ABI reports for the family
/// (`ffi::ACK_OP_SCHEMA_CONVERT`): 1 refused a 1- or 2-byte output that was
/// not dense (`convert-output-not-dense`); 2 (2026-09-27) writes it with
/// atomics through the same entry and request. The work layout and the
/// sub-word paths of 2026-10-01 keep 2: the request, the refusals and every
/// result are unchanged, only how the dispatch reaches them.
pub const SCHEMA: u32 = 2;
pub const MAX_NDIM: usize = 4;
/// `in`, `out`, in binding order.
pub const BINDINGS: u32 = 2;
pub const WORKGROUP_INVOCATIONS: u32 = 256;
/// Units each invocation takes, `WORKGROUP_INVOCATIONS` apart (2026-10-01).
pub const UNITS_PER_INVOCATION: u32 = 4;
/// Consecutive units a workgroup takes: the dispatch is `units / UNITS_PER_WORKGROUP` groups, rounded up.
pub const UNITS_PER_WORKGROUP: u32 = WORKGROUP_INVOCATIONS * UNITS_PER_INVOCATION;
/// Every device offers at least this many workgroups on each dispatch axis.
pub const MAX_GROUPS_X: u32 = 65_535;
/// Every element and word index the shader forms is below this, so its u32
/// arithmetic cannot wrap and an index fits a signed 32-bit int.
pub const WORD_LIMIT: u64 = 1 << 31;
pub const PUSH_WORDS: usize = 18;
pub const PUSH_BYTES: usize = PUSH_WORDS * 4;
/// Push word 1, bit 0: a 1- or 2-byte output that is not dense and whose words
/// elements share, written one element per unit with 32-bit atomics
/// (`units == n`; a same-type copy of 2-byte elements stores each with a 16-bit
/// store instead, 2026-10-05). Set by the planner, never by a caller: the C
/// request's own flags word stays zero.
pub const FLAG_ATOMIC: u32 = 1;
/// Push word 1, bit 1 (2026-10-01): a 1- or 2-byte output that is not dense and
/// in which no two elements share a word, written one element per unit with a
/// plain read-modify-write of its word (`units == n`; a same-type copy of
/// 2-byte elements stores each with a 16-bit store instead, 2026-10-05). Set by
/// the planner only.
pub const FLAG_EXCLUSIVE: u32 = 2;
/// Push word 1, bit 2 (2026-10-03): a transposed read of two dimensions, tiled
/// (`units == n`, the dispatch a grid of [`TILE`] x [`TILE`] tiles, columns on
/// x and rows on y). Set by the planner only, ALONE, and only the tiled module
/// (`kernels/convert_tiled.spv`) accepts it: the other module poisons a block
/// that carries it, as the tiled one poisons a block without it.
pub const FLAG_TILED: u32 = 4;
/// The edge of a tiled workgroup's square of elements.
pub const TILE: u32 = 64;
/// The fewest rows, and the smallest input column stride in elements, a
/// transposed read is tiled at (see `ConvertPlan::new`).
pub const MIN_TILED_ROWS: u64 = 16;
pub const MIN_TILED_STRIDE: u64 = 16;

/// Storage codes, the index family's numbering plus `Bool` (whose output is
/// 0 or 1: a nonzero value, a NaN included, converts to 1).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Code {
    F32 = 0,
    Bf16 = 1,
    F16 = 2,
    I64 = 3,
    I32 = 4,
    I16 = 5,
    I8 = 6,
    U8 = 7,
    F64 = 8,
    Bool = 9,
}

impl Code {
    pub const ALL: [Code; 10] = [
        Self::F32,
        Self::Bf16,
        Self::F16,
        Self::I64,
        Self::I32,
        Self::I16,
        Self::I8,
        Self::U8,
        Self::F64,
        Self::Bool,
    ];

    pub fn bytes(self) -> u64 {
        match self {
            Self::I64 | Self::F64 => 8,
            Self::F32 | Self::I32 => 4,
            Self::Bf16 | Self::F16 | Self::I16 => 2,
            Self::I8 | Self::U8 | Self::Bool => 1,
        }
    }

    pub fn is_float(self) -> bool {
        matches!(self, Self::F32 | Self::Bf16 | Self::F16 | Self::F64)
    }
}

impl TryFrom<u32> for Code {
    type Error = ConvertPlanError;

    fn try_from(word: u32) -> Result<Self, Self::Error> {
        Ok(match word {
            0 => Self::F32,
            1 => Self::Bf16,
            2 => Self::F16,
            3 => Self::I64,
            4 => Self::I32,
            5 => Self::I16,
            6 => Self::I8,
            7 => Self::U8,
            8 => Self::F64,
            9 => Self::Bool,
            _ => return Err(ConvertPlanError::CodeOutOfRange),
        })
    }
}

/// Whether the family converts `from` into `to`: every pair but a float into
/// an integer type ([`ConvertPlanError::FloatToInteger`]).
pub fn converts(from: Code, to: Code) -> bool {
    !(from.is_float() && !to.is_float() && to != Code::Bool)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConvertPlanError {
    CodeOutOfRange,
    FlagsOutOfRange,
    /// A float into an integer type: undefined out of range in c10, so refused.
    FloatToInteger,
    NdimOutOfRange,
    ZeroElements,
    AddressOverflow,
    OutputSelfOverlap,
    OutputOverlapsInput,
}

impl fmt::Display for ConvertPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::CodeOutOfRange => "convert-code-out-of-range",
            Self::FlagsOutOfRange => "convert-flags-out-of-range",
            Self::FloatToInteger => "convert-float-to-integer",
            Self::NdimOutOfRange => "convert-ndim-out-of-range",
            Self::ZeroElements => "convert-zero-elements",
            Self::AddressOverflow => "convert-address-overflow",
            Self::OutputSelfOverlap => "convert-output-self-overlap",
            Self::OutputOverlapsInput => "convert-output-overlaps-input",
        })
    }
}

impl std::error::Error for ConvertPlanError {}

/// One operand's element offset and per-dimension element strides, in its own
/// element size, indexed like the plan's shape.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct View {
    pub offset: u64,
    pub strides: [u64; MAX_NDIM],
}

impl View {
    pub const fn new(offset: u64, strides: [u64; MAX_NDIM]) -> Self {
        Self { offset, strides }
    }

    /// Row-major strides for `shape` at `offset`.
    pub fn contiguous(offset: u64, shape: &[u64]) -> Self {
        let mut strides = [0u64; MAX_NDIM];
        let mut stride = 1u64;
        for (dimension, &extent) in shape.iter().enumerate().rev() {
            strides[dimension] = stride;
            stride = stride.saturating_mul(extent);
        }
        Self { offset, strides }
    }
}

/// The buffer an operand is bound to, as the caller declares it: an identity
/// equal exactly when two operands share one allocation, and its size in bytes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Storage {
    pub identity: u64,
    pub capacity_bytes: u64,
}

impl Storage {
    pub const fn new(identity: u64, capacity_bytes: u64) -> Self {
        Self {
            identity,
            capacity_bytes,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConvertRequest {
    pub from: Code,
    pub to: Code,
    pub ndim: usize,
    pub shape: [u64; MAX_NDIM],
    pub input: View,
    pub out: View,
}

/// Storage in binding order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Bindings {
    pub input: Storage,
    pub out: Storage,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConvertPlan {
    request: ConvertRequest,
    words: [u32; PUSH_WORDS],
    groups: [u32; 3],
}

/// The highest element `view` addresses over `extents`; `None` on overflow.
/// Shared with `intwise_ops`.
pub(crate) fn max_element(view: &View, extents: &[u64; MAX_NDIM]) -> Option<u64> {
    let mut max_index = view.offset;
    for (k, &extent) in extents.iter().enumerate() {
        if extent > 1 {
            let span = (extent - 1).checked_mul(view.strides[k])?;
            max_index = max_index.checked_add(span)?;
        }
    }
    Some(max_index)
}

/// Every coordinate reaches a distinct address: ordered by stride, each stride
/// is at least the span of the dimensions inside it (the pointwise family's rule).
/// Shared with `intwise_ops`.
pub(crate) fn addresses_are_distinct(view: &View, extents: &[u64; MAX_NDIM]) -> bool {
    let mut dimensions: Vec<(u64, u64)> = extents
        .iter()
        .zip(view.strides)
        .filter(|(&extent, _)| extent > 1)
        .map(|(&extent, stride)| (stride, extent))
        .collect();
    dimensions.sort_unstable();
    let mut span = 1u64;
    for (stride, extent) in dimensions {
        if stride < span {
            return false;
        }
        span = span.saturating_add(stride.saturating_mul(extent - 1));
    }
    true
}

/// Checks `view` over `extents` against the word limit and `storage`'s LOGICAL
/// size, as the index family does: every element the view reaches must end
/// inside it (the kit creates every VkBuffer a whole word long, so the word
/// that holds a final sub-word element is always there).
fn check_capacity(
    view: &View,
    extents: &[u64; MAX_NDIM],
    element_bytes: u64,
    storage: &Storage,
) -> Result<(), ConvertPlanError> {
    let max = max_element(view, extents).ok_or(ConvertPlanError::AddressOverflow)?;
    let end_bytes = max
        .checked_add(1)
        .and_then(|elements| elements.checked_mul(element_bytes))
        .ok_or(ConvertPlanError::AddressOverflow)?;
    let last_word = (end_bytes - 1) / 4;
    if max >= WORD_LIMIT || last_word >= WORD_LIMIT {
        return Err(ConvertPlanError::AddressOverflow);
    }
    if end_bytes > storage.capacity_bytes {
        return Err(ConvertPlanError::AddressOverflow);
    }
    Ok(())
}

/// `extents` and both views' strides in the output's memory order (module doc,
/// THE OUTPUT'S MEMORY ORDER): the dimensions of extent above 1 ordered by the
/// output's stride, largest first (a stable sort: equal strides, which only a
/// self-overlapping output has, keep their order), each adjacent pair whose
/// outer strides are the inner ones times the inner extent in BOTH views merged
/// into one, then padded at the front with extent-1 dimensions of stride 0.
/// Every coordinate keeps its input and output address. The caller has checked
/// both views' capacity, so every stride times its extent is below 2^32 and the
/// merged extents multiply to the element count.
fn memory_order(
    extents: &[u64; MAX_NDIM],
    in_st: &[u64; MAX_NDIM],
    out_st: &[u64; MAX_NDIM],
) -> ([u64; MAX_NDIM], [u64; MAX_NDIM], [u64; MAX_NDIM]) {
    let mut order: Vec<usize> = (0..MAX_NDIM).filter(|&k| extents[k] > 1).collect();
    order.sort_by_key(|&k| std::cmp::Reverse(out_st[k]));
    // (extent, input stride, output stride), innermost first
    let mut runs: Vec<(u64, u64, u64)> = Vec::with_capacity(MAX_NDIM);
    for &k in order.iter().rev() {
        if let Some(inner) = runs.last_mut() {
            if inner.1.checked_mul(inner.0) == Some(in_st[k])
                && inner.2.checked_mul(inner.0) == Some(out_st[k])
            {
                inner.0 *= extents[k];
                continue;
            }
        }
        runs.push((extents[k], in_st[k], out_st[k]));
    }
    let mut ordered = ([1u64; MAX_NDIM], [0u64; MAX_NDIM], [0u64; MAX_NDIM]);
    for (inner, &(extent, input, output)) in runs.iter().enumerate() {
        let k = MAX_NDIM - 1 - inner;
        ordered.0[k] = extent;
        ordered.1[k] = input;
        ordered.2[k] = output;
    }
    ordered
}

impl ConvertPlan {
    pub fn new(request: ConvertRequest, bindings: Bindings) -> Result<Self, ConvertPlanError> {
        let ConvertRequest {
            from,
            to,
            ndim,
            shape,
            input,
            out,
        } = request;
        if !converts(from, to) {
            return Err(ConvertPlanError::FloatToInteger);
        }
        if ndim == 0 || ndim > MAX_NDIM {
            return Err(ConvertPlanError::NdimOutOfRange);
        }
        // pad at the front, so the row-major order of the coordinates is unchanged
        let pad = MAX_NDIM - ndim;
        let mut extents = [1u64; MAX_NDIM];
        let mut in_st = [0u64; MAX_NDIM];
        let mut out_st = [0u64; MAX_NDIM];
        extents[pad..].copy_from_slice(&shape[..ndim]);
        in_st[pad..].copy_from_slice(&input.strides[..ndim]);
        out_st[pad..].copy_from_slice(&out.strides[..ndim]);
        if extents.contains(&0) {
            return Err(ConvertPlanError::ZeroElements);
        }
        let n = extents
            .iter()
            .try_fold(1u64, |acc, &extent| acc.checked_mul(extent))
            .filter(|&n| n < WORD_LIMIT)
            .ok_or(ConvertPlanError::AddressOverflow)?;
        let in_view = View::new(input.offset, in_st);
        let out_view = View::new(out.offset, out_st);
        check_capacity(&in_view, &extents, from.bytes(), &bindings.input)?;
        check_capacity(&out_view, &extents, to.bytes(), &bindings.out)?;
        // the output's memory order (module doc): every rule below reads the view in it
        let (extents, in_st, out_st) = memory_order(&extents, &in_st, &out_st);
        let out_view = View::new(out.offset, out_st);
        let es = to.bytes();
        let dense = View::contiguous(0, &extents);
        let row_major = (0..MAX_NDIM).all(|k| extents[k] == 1 || out_st[k] == dense.strides[k]);
        // A dense sub-word output takes whole words, at any offset (2026-10-01): its
        // first and last words may hold fewer of its elements, and the shader
        // keeps their other bytes. Any other sub-word output takes one element a
        // unit: with a plain read-modify-write when no two elements share a word
        // (every stride over an extent above 1 a multiple of the slots a word has,
        // so every element sits at the same slot of a word of its own), else with
        // atomics.
        let slots = if es < 4 { 4 / es } else { 1 };
        let whole_words = es < 4 && row_major;
        if !whole_words && !addresses_are_distinct(&out_view, &extents) {
            return Err(ConvertPlanError::OutputSelfOverlap);
        }
        let exclusive = es < 4
            && !whole_words
            && (0..MAX_NDIM).all(|k| extents[k] == 1 || out_st[k] % slots == 0);
        let atomic = es < 4 && !whole_words && !exclusive;
        // A TRANSPOSED READ of two dimensions is tiled (2026-10-03; module doc).
        // Below MIN_TILED_ROWS rows a tile column is too short to fill a cache
        // line, and below MIN_TILED_STRIDE elements of input stride the untiled
        // path's reads already share lines, so neither is tiled.
        let (rows, cols) = (extents[2], extents[3]);
        let tile = u64::from(TILE);
        let tiled = extents[0] == 1
            && extents[1] == 1
            && rows >= MIN_TILED_ROWS
            && cols > 1
            && from.bytes() <= 4
            && es <= 4
            && in_st[2] == 1
            && in_st[3] >= MIN_TILED_STRIDE
            && out_st[3] == 1
            && out_st[2] == cols
            && (es == 4 || (out.offset % slots == 0 && cols % slots == 0))
            && rows.div_ceil(tile) <= u64::from(MAX_GROUPS_X)
            && cols.div_ceil(tile) <= u64::from(MAX_GROUPS_X);
        let lead = if whole_words { out.offset % slots } else { 0 };
        let units = if tiled {
            n
        } else if whole_words {
            (lead + n).div_ceil(slots)
        } else {
            n
        };
        // the output is written while the input is read
        if bindings.out.identity == bindings.input.identity {
            return Err(ConvertPlanError::OutputOverlapsInput);
        }
        // A stride on an extent-1 dimension never contributes; zeroing it makes
        // a decoded block re-encode to the same words. A whole-word sub-word
        // output carries its row-major strides on every dimension (the shader
        // checks them all), so there an extent-1 dimension gets the dense stride.
        let mut in_words = [0u32; MAX_NDIM];
        let mut out_words = [0u32; MAX_NDIM];
        for k in 0..MAX_NDIM {
            if extents[k] > 1 {
                in_words[k] = in_st[k] as u32;
            }
            out_words[k] = if whole_words {
                dense.strides[k] as u32
            } else if extents[k] > 1 {
                out_st[k] as u32
            } else {
                0
            };
        }
        let mut words = [0u32; PUSH_WORDS];
        words[0] = from as u32 | (to as u32) << 4;
        words[1] = if tiled {
            FLAG_TILED
        } else if atomic {
            FLAG_ATOMIC
        } else if exclusive {
            FLAG_EXCLUSIVE
        } else {
            0
        };
        for k in 0..MAX_NDIM {
            words[2 + k] = extents[k] as u32;
            words[7 + k] = in_words[k];
            words[12 + k] = out_words[k];
        }
        words[6] = input.offset as u32;
        words[11] = out.offset as u32;
        words[16] = units as u32;
        words[17] = n as u32;
        let groups = if tiled {
            // columns on x, rows on y: one workgroup a tile
            [cols.div_ceil(tile) as u32, rows.div_ceil(tile) as u32, 1]
        } else {
            let total_groups = units.div_ceil(u64::from(UNITS_PER_WORKGROUP));
            let x = total_groups.min(u64::from(MAX_GROUPS_X)).max(1);
            let y = total_groups.div_ceil(x).max(1);
            [x as u32, y as u32, 1]
        };
        Ok(Self {
            request,
            words,
            groups,
        })
    }

    /// Whether this plan is a tiled transposed read ([`FLAG_TILED`]), which
    /// only `kernels/convert_tiled.spv` runs.
    pub fn tiled(&self) -> bool {
        self.words[1] & FLAG_TILED != 0
    }

    /// A request in the C ABI's words (`ffi::AckConvertRequest`): the storage
    /// codes and flags decoded by name, then planned against the bindings.
    #[allow(clippy::too_many_arguments)]
    pub fn from_abi(
        from: u32,
        to: u32,
        ndim: u32,
        flags: u32,
        shape: [u64; MAX_NDIM],
        input: View,
        out: View,
        bindings: Bindings,
    ) -> Result<Self, ConvertPlanError> {
        let from = Code::try_from(from)?;
        let to = Code::try_from(to)?;
        if flags != 0 {
            return Err(ConvertPlanError::FlagsOutOfRange);
        }
        let request = ConvertRequest {
            from,
            to,
            ndim: usize::try_from(ndim).map_err(|_| ConvertPlanError::NdimOutOfRange)?,
            shape,
            input,
            out,
        };
        Self::new(request, bindings)
    }

    pub fn request(&self) -> &ConvertRequest {
        &self.request
    }

    /// Invocations with work: output words for a dense 1- or 2-byte output,
    /// else output elements.
    pub fn units(&self) -> u32 {
        self.words[16]
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
}

#[cfg(test)]
mod tests {
    use super::*;

    const IN: Storage = Storage::new(1, 1 << 20);
    const OUT: Storage = Storage::new(2, 1 << 20);

    fn bindings() -> Bindings {
        Bindings {
            input: IN,
            out: OUT,
        }
    }

    /// x [4, 6] (a transposed view of a [6, 4] tensor) into a contiguous out.
    fn request(from: Code, to: Code) -> ConvertRequest {
        ConvertRequest {
            from,
            to,
            ndim: 2,
            shape: [4, 6, 0, 0],
            input: View::new(0, [1, 4, 0, 0]),
            out: View::new(0, [6, 1, 0, 0]),
        }
    }

    #[test]
    fn every_pair_but_float_to_integer_is_planned() {
        for from in Code::ALL {
            for to in Code::ALL {
                let planned = ConvertPlan::new(request(from, to), bindings());
                if from.is_float() && !to.is_float() && to != Code::Bool {
                    assert_eq!(
                        planned,
                        Err(ConvertPlanError::FloatToInteger),
                        "{from:?} -> {to:?}"
                    );
                } else {
                    let plan = planned.unwrap_or_else(|e| panic!("{from:?} -> {to:?}: {e}"));
                    assert_eq!(plan.words()[0], from as u32 | (to as u32) << 4);
                }
            }
        }
    }

    #[test]
    fn the_words_mirror_the_shader_block() {
        let plan = ConvertPlan::new(request(Code::F32, Code::F16), bindings()).unwrap();
        let w = plan.words();
        // padded at the front: [1, 1, 4, 6]
        assert_eq!(&w[2..6], &[1, 1, 4, 6]);
        assert_eq!(&w[6..11], &[0, 0, 0, 1, 4], "input offset then strides");
        assert_eq!(
            &w[11..16],
            &[0, 24, 24, 6, 1],
            "a dense sub-word output carries every row-major stride"
        );
        assert_eq!(w[17], 24, "elements");
        assert_eq!(w[16], 12, "two f16 elements per word");
        assert_eq!(plan.dispatch_groups(), [1, 1, 1]);
        assert_eq!(plan.push_constants().len(), PUSH_BYTES);

        let plan = ConvertPlan::new(request(Code::I8, Code::F32), bindings()).unwrap();
        assert_eq!(plan.units(), 24, "one invocation per four-byte element");
        assert_eq!(
            &plan.words()[11..16],
            &[0, 0, 0, 6, 1],
            "extent-1 strides are zeroed"
        );
    }

    #[test]
    fn units_round_up_to_whole_output_words() {
        let mut r = request(Code::F32, Code::U8);
        r.shape = [5, 3, 0, 0];
        r.input = View::new(0, [3, 1, 0, 0]);
        r.out = View::new(0, [3, 1, 0, 0]);
        assert_eq!(
            ConvertPlan::new(r, bindings()).unwrap_err(),
            ConvertPlanError::FloatToInteger
        );
        r.from = Code::I32;
        let plan = ConvertPlan::new(r, bindings()).unwrap();
        assert_eq!(plan.units(), 4, "15 one-byte elements are four words");
        r.to = Code::I16;
        assert_eq!(
            ConvertPlan::new(r, bindings()).unwrap().units(),
            8,
            "15 two-byte elements are eight words"
        );
    }

    #[test]
    fn a_sub_word_output_is_whole_words_when_dense_and_one_element_a_unit_otherwise() {
        let mut r = request(Code::F32, Code::Bf16);
        let dense = ConvertPlan::new(r, bindings()).unwrap();
        assert_eq!(
            (dense.words()[1], dense.units()),
            (0, 12),
            "dense: whole words"
        );
        // 2026-10-01: a dense output at an odd offset takes whole words too. Its
        // 24 elements start at slot 1 of their first word, so 13 words: the first
        // and the last hold one element each and keep their other bytes.
        r.out = View::new(1, [6, 1, 0, 0]);
        let odd = ConvertPlan::new(r, bindings()).unwrap();
        assert_eq!(
            (odd.words()[1], odd.units()),
            (0, 13),
            "an odd offset: whole words, one more than the even offset's"
        );
        assert_eq!(
            &odd.words()[11..16],
            &[1, 24, 24, 6, 1],
            "a whole-word output carries its offset and its row-major strides"
        );
        r.out = View::new(2, [6, 1, 0, 0]);
        assert_eq!(
            (
                ConvertPlan::new(r, bindings()).unwrap().words()[1],
                ConvertPlan::new(r, bindings()).unwrap().units()
            ),
            (0, 12),
            "an even offset starts a word"
        );
        // 2026-10-03: a transposed output is dense in its own memory order, where
        // the planner puts it; here the input has the same layout, so the two are
        // one run of 24. Until then its neighbours shared words in the request's
        // order, so atomics.
        r.out = View::new(0, [1, 4, 0, 0]);
        let transposed = ConvertPlan::new(r, bindings()).unwrap();
        assert_eq!(
            (transposed.words()[1], transposed.units()),
            (0, 12),
            "a transposed output: whole words in its memory order"
        );
        assert_eq!(&transposed.words()[2..6], &[1, 1, 1, 24]);
        // a gap after every fourth element: dense in no order, and neighbours share words
        r.out = View::new(0, [1, 5, 0, 0]);
        assert_eq!(
            ConvertPlan::new(r, bindings()).unwrap().words()[1],
            FLAG_ATOMIC,
            "a padded transposed output: neighbours share words, so atomics"
        );
        // every stride even: each element at slot 0 of a word of its own
        r.out = View::new(0, [12, 2, 0, 0]);
        let spread = ConvertPlan::new(r, bindings()).unwrap();
        assert_eq!(
            (spread.words()[1], spread.units()),
            (FLAG_EXCLUSIVE, 24),
            "a stride-2 output: a word to each element, no atomics"
        );
        assert_eq!(&spread.words()[11..16], &[0, 0, 0, 12, 2]);
        r.out = View::new(1, [12, 2, 0, 0]);
        assert_eq!(
            ConvertPlan::new(r, bindings()).unwrap().words()[1],
            FLAG_EXCLUSIVE,
            "the same at an odd offset: every element at slot 1"
        );
        r.out = View::new(0, [1, 0, 0, 0]);
        assert_eq!(
            ConvertPlan::new(r, bindings()).unwrap_err(),
            ConvertPlanError::OutputSelfOverlap
        );
        // a byte output has four slots a word: a word of its own needs every stride a multiple of 4
        r.to = Code::U8;
        r.from = Code::U8;
        r.out = View::new(0, [24, 4, 0, 0]);
        assert_eq!(
            ConvertPlan::new(r, bindings()).unwrap().words()[1],
            FLAG_EXCLUSIVE
        );
        r.out = View::new(0, [12, 2, 0, 0]);
        assert_eq!(
            ConvertPlan::new(r, bindings()).unwrap().words()[1],
            FLAG_ATOMIC,
            "stride 2 bytes: two elements share each word"
        );
        r.out = View::new(3, [6, 1, 0, 0]);
        assert_eq!(
            (
                ConvertPlan::new(r, bindings()).unwrap().words()[1],
                ConvertPlan::new(r, bindings()).unwrap().units()
            ),
            (0, 7),
            "24 dense bytes from slot 3: 1 + 4 x 5 + 3 in 7 words"
        );
        r.from = Code::F32;
        r.to = Code::F32;
        r.out = View::new(0, [1, 4, 0, 0]);
        let wide = ConvertPlan::new(r, bindings()).unwrap();
        assert_eq!(
            (wide.words()[1], wide.units()),
            (0, 24),
            "a four-byte output never takes atomics"
        );
        r.out = View::new(0, [1, 0, 0, 0]);
        assert_eq!(
            ConvertPlan::new(r, bindings()).unwrap_err(),
            ConvertPlanError::OutputSelfOverlap
        );
    }

    #[test]
    fn a_workgroup_takes_1024_units_and_the_grid_folds_above_65535_groups() {
        // 2026-10-01: 256 invocations a workgroup, four units each
        assert_eq!(UNITS_PER_WORKGROUP, 1024);
        let small = ConvertPlan::new(request(Code::F32, Code::Bf16), bindings()).unwrap();
        assert_eq!(small.dispatch_groups(), [1, 1, 1]);
        let roomy = Bindings {
            input: Storage::new(1, 1 << 30),
            out: Storage::new(2, 1 << 30),
        };
        let flat = |n: u64| ConvertRequest {
            from: Code::F32,
            to: Code::F32,
            ndim: 1,
            shape: [n, 0, 0, 0],
            input: View::new(0, [1, 0, 0, 0]),
            out: View::new(0, [1, 0, 0, 0]),
        };
        let one = ConvertPlan::new(flat(1024), roomy).unwrap();
        assert_eq!(
            one.dispatch_groups(),
            [1, 1, 1],
            "1,024 units: one workgroup"
        );
        let two = ConvertPlan::new(flat(1025), roomy).unwrap();
        assert_eq!(two.dispatch_groups(), [2, 1, 1], "1,025 units: two");
        // 2^27 units are 131,072 workgroups: the x axis holds 65,535, so three rows
        let big = ConvertPlan::new(flat(1 << 27), roomy).unwrap();
        assert_eq!(big.dispatch_groups(), [65_535, 3, 1]);
    }

    #[test]
    fn capacity_is_checked_for_each_operand_in_its_own_element_size() {
        // 24 f64 elements are 192 bytes, 24 u8 elements 24 bytes
        let small_in = Bindings {
            input: Storage::new(1, 191),
            out: OUT,
        };
        let mut r = request(Code::F64, Code::F32);
        assert_eq!(
            ConvertPlan::new(r, small_in).unwrap_err(),
            ConvertPlanError::AddressOverflow
        );
        let exact_in = Bindings {
            input: Storage::new(1, 192),
            out: OUT,
        };
        assert!(ConvertPlan::new(r, exact_in).is_ok());
        r = request(Code::I32, Code::U8);
        let small_out = Bindings {
            input: IN,
            out: Storage::new(2, 23),
        };
        assert_eq!(
            ConvertPlan::new(r, small_out).unwrap_err(),
            ConvertPlanError::AddressOverflow
        );
        let exact_out = Bindings {
            input: IN,
            out: Storage::new(2, 24),
        };
        assert!(ConvertPlan::new(r, exact_out).is_ok());
    }

    #[test]
    fn the_output_may_not_share_the_input_storage() {
        let same = Bindings {
            input: IN,
            out: Storage::new(1, 1 << 20),
        };
        assert_eq!(
            ConvertPlan::new(request(Code::F16, Code::F32), same).unwrap_err(),
            ConvertPlanError::OutputOverlapsInput
        );
    }

    #[test]
    fn the_abi_decode_refuses_by_name() {
        let shape = [4, 6, 0, 0];
        let v_in = View::new(0, [1, 4, 0, 0]);
        let v_out = View::new(0, [6, 1, 0, 0]);
        let plan = |from, to, ndim, flags| {
            ConvertPlan::from_abi(from, to, ndim, flags, shape, v_in, v_out, bindings())
        };
        assert_eq!(
            plan(10, 0, 2, 0).unwrap_err(),
            ConvertPlanError::CodeOutOfRange
        );
        assert_eq!(
            plan(0, 10, 2, 0).unwrap_err(),
            ConvertPlanError::CodeOutOfRange
        );
        assert_eq!(
            plan(0, 1, 2, 1).unwrap_err(),
            ConvertPlanError::FlagsOutOfRange
        );
        assert_eq!(
            plan(0, 1, 0, 0).unwrap_err(),
            ConvertPlanError::NdimOutOfRange
        );
        assert_eq!(
            plan(0, 1, 5, 0).unwrap_err(),
            ConvertPlanError::NdimOutOfRange
        );
        assert_eq!(
            plan(0, 3, 2, 0).unwrap_err(),
            ConvertPlanError::FloatToInteger
        );
        assert!(plan(3, 0, 2, 0).is_ok(), "int64 -> f32");
        let zero = ConvertPlan::from_abi(0, 1, 2, 0, [4, 0, 0, 0], v_in, v_out, bindings());
        assert_eq!(zero.unwrap_err(), ConvertPlanError::ZeroElements);
    }

    #[test]
    fn element_counts_past_the_word_limit_are_refused() {
        let big = Storage::new(3, u64::MAX);
        let r = ConvertRequest {
            from: Code::I8,
            to: Code::I16,
            ndim: 1,
            shape: [WORD_LIMIT, 0, 0, 0],
            input: View::new(0, [1, 0, 0, 0]),
            out: View::new(0, [1, 0, 0, 0]),
        };
        let b = Bindings {
            input: big,
            out: Storage::new(4, u64::MAX),
        };
        assert_eq!(
            ConvertPlan::new(r, b).unwrap_err(),
            ConvertPlanError::AddressOverflow
        );
    }

    /// `rows` x `cols` read from a [cols, ld] tensor's transpose (input
    /// strides 1 and `ld`) into a dense row-major output at `out_offset`.
    fn transposed(
        from: Code,
        to: Code,
        rows: u64,
        cols: u64,
        ld: u64,
        out_offset: u64,
    ) -> ConvertRequest {
        ConvertRequest {
            from,
            to,
            ndim: 2,
            shape: [rows, cols, 0, 0],
            input: View::new(0, [1, ld, 0, 0]),
            out: View::new(out_offset, [cols, 1, 0, 0]),
        }
    }

    fn big() -> Bindings {
        Bindings {
            input: Storage::new(1, 1 << 34),
            out: Storage::new(2, 1 << 34),
        }
    }

    #[test]
    fn a_transposed_read_of_two_dimensions_is_tiled() {
        // the stage-E head's sign cache read through its transpose into a dense
        // f16 tensor (MEASURED 2026-10-03 on the RX 9070 XT: 8.27 ms untiled,
        // 2.66 ms tiled); stage E's own cast keeps the transposed layout instead
        // (the_dimensions_are_put_in_the_outputs_memory_order)
        let plan = ConvertPlan::new(
            transposed(Code::I8, Code::F16, 1002, 151_936, 1002, 0),
            big(),
        )
        .unwrap();
        assert!(plan.tiled());
        let w = plan.words();
        assert_eq!(w[1], FLAG_TILED, "the tiled flag ALONE");
        assert_eq!(&w[2..6], &[1, 1, 1002, 151_936]);
        assert_eq!(
            &w[7..11],
            &[0, 0, 1, 1002],
            "input strides: 1 along the rows"
        );
        assert_eq!(&w[14..16], &[151_936, 1], "output: dense row-major");
        assert_eq!(w[16], w[17], "units == n");
        assert_eq!(w[17], 1002 * 151_936);
        assert_eq!(
            plan.dispatch_groups(),
            [151_936u32.div_ceil(TILE), 1002u32.div_ceil(TILE), 1],
            "one workgroup a tile: columns on x, rows on y"
        );
        // a 4-byte output needs no word alignment of its rows
        assert!(
            ConvertPlan::new(transposed(Code::I8, Code::F32, 100, 1001, 300, 3), big())
                .unwrap()
                .tiled()
        );
        // 1-byte input and output, rows and offset on words
        assert!(
            ConvertPlan::new(transposed(Code::U8, Code::I8, 64, 128, 64, 4), big())
                .unwrap()
                .tiled()
        );
    }

    #[test]
    fn a_read_tiling_cannot_help_or_a_shared_output_word_is_not_tiled() {
        let untiled = |r: ConvertRequest, why: &str| {
            let plan = ConvertPlan::new(r, big()).unwrap();
            assert!(!plan.tiled(), "{why}");
            assert_eq!(plan.words()[1] & FLAG_TILED, 0, "{why}");
        };
        let contiguous = ConvertRequest {
            input: View::new(0, [2048, 1, 0, 0]),
            ..transposed(Code::I8, Code::F16, 1002, 2048, 0, 0)
        };
        untiled(contiguous, "a contiguous input reads in order already");
        untiled(
            transposed(Code::I8, Code::F16, 1002, 2047, 1002, 0),
            "f16 rows of an odd length share words",
        );
        untiled(
            transposed(Code::I8, Code::F16, 1002, 2048, 1002, 1),
            "an f16 output off a word boundary",
        );
        untiled(
            transposed(Code::I8, Code::U8, 1002, 2050, 1002, 0),
            "u8 rows not a multiple of 4",
        );
        untiled(
            transposed(Code::I64, Code::F32, 1002, 2048, 1002, 0),
            "an 8-byte input",
        );
        untiled(
            transposed(Code::F32, Code::F64, 1002, 2048, 1002, 0),
            "an 8-byte output",
        );
        untiled(
            transposed(Code::I8, Code::F16, MIN_TILED_ROWS - 1, 2048, 1002, 0),
            "too few rows",
        );
        let narrow = ConvertRequest {
            input: View::new(0, [1, MIN_TILED_STRIDE - 1, 0, 0]),
            ..transposed(Code::I8, Code::F16, 1002, 2048, 1002, 0)
        };
        untiled(
            narrow,
            "an input stride the untiled reads already share lines at",
        );
        let three = ConvertRequest {
            ndim: 3,
            shape: [2, 1002, 2048, 0],
            input: View::new(0, [1002 * 2048, 1, 1002, 0]),
            out: View::new(0, [1002 * 2048, 2048, 1, 0]),
            ..transposed(Code::I8, Code::F16, 1, 1, 1, 0)
        };
        untiled(three, "three dimensions");
    }

    #[test]
    fn the_dimensions_are_put_in_the_outputs_memory_order() {
        // the stage-E head's cast as torch hands it over (2026-10-03): x.t().to(f16)
        // keeps x's strides, so the input and the output are the same transposed
        // view of their buffers, which in the output's order is one dense run
        let n = 1002 * 151_936;
        let same = ConvertRequest {
            input: View::new(0, [1, 1002, 0, 0]),
            out: View::new(0, [1, 1002, 0, 0]),
            ..transposed(Code::I8, Code::F16, 1002, 151_936, 1002, 0)
        };
        let w = ConvertPlan::new(same, big()).unwrap().words();
        assert_eq!(w[1], 0, "no flag: neither atomics nor the tiled module");
        assert_eq!(&w[2..6], &[1, 1, 1, n]);
        assert_eq!(&w[6..11], &[0, 0, 0, 0, 1], "the input read in order");
        assert_eq!(&w[11..16], &[0, n, n, n, 1], "a dense output");
        assert_eq!(
            (w[16], w[17]),
            (n / 2, n),
            "whole words, two f16 elements each"
        );
        // an output transposed against a dense input (z.t().copy_(y)) is a
        // transposed read in the output's order, so tiled
        let written = ConvertRequest {
            shape: [1002, 2048, 0, 0],
            input: View::new(0, [2048, 1, 0, 0]),
            out: View::new(0, [1, 1002, 0, 0]),
            ..transposed(Code::I8, Code::F16, 1, 1, 1, 0)
        };
        let plan = ConvertPlan::new(written, big()).unwrap();
        assert!(plan.tiled());
        let w = plan.words();
        assert_eq!(
            &w[2..6],
            &[1, 1, 2048, 1002],
            "the rows and columns swapped"
        );
        assert_eq!(&w[7..11], &[0, 0, 1, 2048]);
        assert_eq!(&w[14..16], &[1002, 1]);
        // an extent-1 dimension is dropped wherever it sits, and dimensions that
        // compose in both views merge: [3, 1, 5], dense in both, is one run of 15
        let r = ConvertRequest {
            from: Code::I32,
            to: Code::F32,
            ndim: 3,
            shape: [3, 1, 5, 0],
            input: View::new(0, [5, 7, 1, 0]),
            out: View::new(0, [5, 9, 1, 0]),
        };
        let w = ConvertPlan::new(r, bindings()).unwrap().words();
        assert_eq!(&w[2..6], &[1, 1, 1, 15]);
        assert_eq!(&w[7..11], &[0, 0, 0, 1]);
        assert_eq!(&w[12..16], &[0, 0, 0, 1]);
        // a broadcast input dimension (stride 0) merges with a broadcast
        // neighbour, never with a strided one
        let r = ConvertRequest {
            from: Code::I32,
            to: Code::F32,
            ndim: 3,
            shape: [2, 4, 6, 0],
            input: View::new(5, [0, 0, 1, 0]),
            out: View::new(0, [24, 6, 1, 0]),
        };
        let w = ConvertPlan::new(r, bindings()).unwrap().words();
        assert_eq!(&w[2..6], &[1, 1, 8, 6]);
        assert_eq!(&w[6..11], &[5, 0, 0, 0, 1]);
        assert_eq!(&w[12..16], &[0, 0, 6, 1]);
    }

    /// Every coordinate's (input, output) element address over `extents`, sorted.
    fn pairs(
        extents: &[u64],
        in_offset: u64,
        in_st: &[u64],
        out_offset: u64,
        out_st: &[u64],
    ) -> Vec<(u64, u64)> {
        let n: u64 = extents.iter().product();
        let mut all: Vec<(u64, u64)> = (0..n)
            .map(|l| {
                let (mut rest, mut i, mut o) = (l, in_offset, out_offset);
                for k in (0..extents.len()).rev() {
                    let c = rest % extents[k];
                    rest /= extents[k];
                    i += c * in_st[k];
                    o += c * out_st[k];
                }
                (i, o)
            })
            .collect();
        all.sort_unstable();
        all
    }

    #[test]
    fn the_memory_order_keeps_every_coordinates_pair_of_addresses() {
        // A fixed pseudo-random walk (xorshift) over 1 to 4 dimensions, of
        // extent 1 to 24 at one or two dimensions and 1 to 5 above. Each
        // operand's layout is a random order of the dimensions laid out densely
        // from a stride of 1 or 2, with a gap after some of them; the input's may
        // also broadcast a dimension (stride 0). The plan's words, unravelled as
        // the shader unravels them, must reach exactly the request's pairs,
        // whatever path the planner took.
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = move |bound: u64| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % bound
        };
        let layout = |shape: &[u64], broadcast: bool, next: &mut dyn FnMut(u64) -> u64| {
            let mut order: Vec<usize> = (0..shape.len()).collect();
            for k in (1..order.len()).rev() {
                order.swap(k, next(k as u64 + 1) as usize);
            }
            let mut strides = [0u64; MAX_NDIM];
            let mut stride = if next(4) == 0 { 2 } else { 1 };
            for &k in &order {
                strides[k] = if broadcast && next(4) == 0 { 0 } else { stride };
                stride *= shape[k] + if next(3) == 0 { 1 } else { 0 };
            }
            strides
        };
        let mut paths = [0usize; 8];
        for case in 0..3000 {
            let ndim = 1 + (case % 4);
            let top = if ndim <= 2 { 24 } else { 5 };
            let mut shape = [0u64; MAX_NDIM];
            for extent in shape.iter_mut().take(ndim) {
                *extent = 1 + next(top);
            }
            let input = View::new(next(3), layout(&shape[..ndim], true, &mut next));
            let out = View::new(next(3), layout(&shape[..ndim], false, &mut next));
            let to = if case % 2 == 0 { Code::F32 } else { Code::F16 };
            let r = ConvertRequest {
                from: Code::I8,
                to,
                ndim,
                shape,
                input,
                out,
            };
            let w = ConvertPlan::new(r, bindings()).unwrap().words();
            paths[w[1] as usize] += 1;
            let words = |at: usize| -> Vec<u64> { (at..at + 4).map(|k| u64::from(w[k])).collect() };
            assert_eq!(
                pairs(
                    &words(2),
                    u64::from(w[6]),
                    &words(7),
                    u64::from(w[11]),
                    &words(12)
                ),
                pairs(
                    &shape[..ndim],
                    input.offset,
                    &input.strides[..ndim],
                    out.offset,
                    &out.strides[..ndim]
                ),
                "case {case}: {r:?}"
            );
        }
        // the walk reached the dense path, both one-element paths and the tiled one
        for path in [0, FLAG_ATOMIC, FLAG_EXCLUSIVE, FLAG_TILED] {
            assert!(
                paths[path as usize] > 0,
                "no case took path {path}: {paths:?}"
            );
        }
    }
}
