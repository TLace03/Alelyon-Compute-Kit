//! Pure plans for `kernels/reduce_f32.comp`: a fixed-order reduction of a
//! strided FP32 view over any subset of its dimensions.
//!
//! The plan validates every address the kernel will touch against the
//! declared input capacity, chooses the split of the reduced index space into
//! fixed contiguous ranges, and exports the push constants and dispatch grids
//! of the one or two passes. It has no Vulkan dependency. The order of
//! combination is a function of the plan alone, so two dispatches of one plan
//! produce bit-identical outputs; it is not the order PyTorch's CPU kernels
//! use, and the difference is bounded by the probe's declared tolerances,
//! never zero.
//!
//! Semantics follow `aten::sum.dim_IntList`, `aten::mean.dim`,
//! `aten::amax` (NaN-propagating) and `aten::linalg_vector_norm` with `ord=2`
//! (the sum of squares, then its root) or `ord=inf` (`AbsMax`, the largest
//! |x|, NaN-propagating like amax)
//! over the reduced dims. `keepdim` is the caller's output view: the output is
//! dense over the kept dims in their original order.

use std::fmt;

pub const MAX_RANK: usize = 6;
/// Reduced elements per split of the first pass. A function of nothing but
/// this constant and the reduced count, so the combination order is fixed.
pub const CHUNK: u32 = 65_536;
/// The kernel addresses elements with 32-bit unsigned offsets below this bound.
pub const MAX_ADDRESS: u64 = 1 << 31;
pub const MAX_WORKGROUPS: u64 = 1 << 31;
pub const PUSH_BYTES: usize = 80;

/// Storage flags (schema 2, 2026-09-10). Bit 0: the INPUT buffer holds
/// bf16 rather than f32. Bit 1: the OUTPUT buffer does. They decide only
/// how a value is READ and WRITTEN -- every accumulation, split
/// combination and finish stays f32 -- so the family's fixed-order
/// guarantee is unaffected and two dispatches of one plan remain
/// bit-identical.
///
/// ONE BIT PER BUFFER rather than one for the pair, because a two-pass
/// plan's middle block is PARTIALS and is always f32: a bf16 reduction is
/// pass 1 (bf16 in, f32 out) then pass 2 (f32 in, bf16 out), and a shared
/// bit could express neither.
pub const FLAG_IN_BF16: u32 = 1;
pub const FLAG_OUT_BF16: u32 = 2;
/// Storage flag bit 2 (schema 4, 2026-09-29): the INPUT, any PARTIALS and the
/// OUTPUT are float64, reduced by `kernels/reduce_f64.comp` -- v1's plan, its
/// fixed order, with a binary64 add in 32-bit words (no shaderFloat64). For
/// `ReduceOp::Sum` with `Finish::None` alone, and beside no bf16 bit; anything
/// else is `ReducePlanError::F64Unsupported`.
pub const FLAG_F64: u32 = 4;
pub const STORAGE_FLAGS: u32 = FLAG_IN_BF16 | FLAG_OUT_BF16 | FLAG_F64;

/// The request block `ack_reduce` accepts: the 80-byte push layout, which
/// is what every caller before 2026-09-10 sends and still means f32 on
/// both sides, or that layout followed by ONE storage-flags word. The
/// kernel's own push block is unchanged at 80/96 bytes either way; the
/// flags word never reaches a shader, it selects which shader.
pub const REQUEST_BYTES: usize = PUSH_BYTES + 4;

/// Bytes one element of a pass's input (`input`) or output occupies under a
/// storage word: 8 under `FLAG_F64`, else by that side's bf16 bit.
pub fn storage_element_bytes(storage: u32, input: bool) -> u64 {
    if storage & FLAG_F64 != 0 {
        return 8;
    }
    let bit = if input { FLAG_IN_BF16 } else { FLAG_OUT_BF16 };
    element_bytes(storage & bit != 0)
}

/// Bytes one element of a buffer occupies under a storage bit.
pub fn element_bytes(bf16: bool) -> u64 {
    if bf16 {
        2
    } else {
        4
    }
}
pub const WORKGROUP_SIZE: u32 = 256;
/// reduce-f32/v2 (2026-09-09, `kernels/reduce_f32_v2.comp`): the kept index
/// space across the lanes. Its push block is v1's 80 bytes followed by four
/// words: `cdim` (the kept dim the lanes run along, stride 1), `cdim2` (a
/// second kept dim flattened onto it, or `NONE_DIM`), `tile_cols` (kept
/// elements per workgroup, a power of two in 8..=256) and `outer_kept` (the
/// product of the other kept extents, the z grid).
pub const PUSH_BYTES_V2: usize = 96;
pub const TILE_COLS_MIN: u32 = 8;
pub const TILE_COLS_MAX: u32 = 256;
pub const NONE_DIM: u32 = u32::MAX;
/// Vulkan guarantees `maxComputeWorkGroupCount` of at least 65,535 in y and
/// z; a v2 grid that would exceed it falls back to v1's one-dimensional grid.
pub const MAX_GRID_YZ: u64 = 65_535;
/// v2 is selected when its kept span across the lanes is at least
/// `V2_MIN_KEPT` wide and its grid reaches `V2_MIN_WORKGROUPS`, or when the
/// span is at least `V2_MIN_KEPT_FOR_ANY_GRID` wide whatever the grid.
pub const V2_MIN_KEPT: u64 = 64;
pub const V2_MIN_WORKGROUPS: u64 = 32;
pub const V2_MIN_KEPT_FOR_ANY_GRID: u64 = 4_096;

/// reduce-f32/v3 (2026-09-26, `kernels/reduce_f32_v3.comp`): rows own a few
/// lanes. Selected when the REDUCED dims form one dense row-major span of
/// unit stride (the element at reduced index r is the row's base + r), which
/// is every last-dim statistic the AKV pass takes and every second pass over
/// partials. Its push block is v1's 80 bytes followed by `lanes` (per row, a
/// power of two in 1..=256), `kept_count`, `reduced_count` and a zero word.
pub const PUSH_BYTES_V3: usize = 96;
/// Rows up to this length are one pass whatever the grid.
pub const V3_ROW_MAX: u64 = 4_096;
/// A longer contiguous span splits into chunks of at least this many elements,
/// and of at least `reduced_count / V3_SPLITS_MAX`, so its partials are one
/// v3 row per kept index in the second pass.
pub const V3_CHUNK_MIN: u64 = 2_048;
pub const V3_SPLITS_MAX: u64 = 4_096;
/// Elements each lane of a row accumulates before the tree, at least: the
/// lane count starts at the power of two covering `row_len / V3_PER_LANE`
/// and doubles while the grid is short of `TARGET_GROUPS` (a small problem
/// wants lanes more than it wants efficiency per lane), up to 256 (one row
/// per workgroup) and never past the row's own length.
pub const V3_PER_LANE: u64 = 8;
pub const V3_MAX_LANES: u64 = 256;
/// First-pass workgroups the split laws aim for. A plan below it on one split
/// splits its reduced rows until it reaches it (or runs out of rows); a
/// function of the shape alone, so the combination order stays fixed.
pub const TARGET_GROUPS: u64 = 256;
/// Reduced rows per split, at least, when a column plan splits for occupancy.
pub const MIN_SPLIT_ROWS: u64 = 32;
/// A v1 plan splits for occupancy from this many reduced elements, in chunks
/// of at least `V1_CHUNK_MIN` (eight per lane of v1's 256).
pub const V1_SPLIT_MIN: u64 = 4_096;
pub const V1_CHUNK_MIN: u64 = 2_048;
/// A column plan splits for occupancy only from this many reduced rows: below
/// it the second pass costs more than the split saves (measured 2026-09-26: a
/// 128-row gamma gradient took 6.7 us in two passes against 4.9 in one).
pub const V2_SPLIT_MIN_ROWS: u64 = 512;
/// Kept columns per workgroup when a column plan splits for occupancy: 256
/// contiguous bytes of f32 per row read, four row lanes per column.
pub const V2_SPLIT_TILE: u64 = 64;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReduceOp {
    Sum = 0,
    Amax = 1,
    SumOfSquares = 2,
    /// The largest |x| over the reduced values, NaN-propagating:
    /// `linalg_vector_norm` with `ord=inf`. The kernel clears the sign bit and
    /// takes the max on the BITS (non-negative floats order as their bits do),
    /// so no float instruction sees a subnormal it could flush.
    AbsMax = 3,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Finish {
    None = 0,
    Scale = 1,
    Sqrt = 2,
}

/// Which shader a pass binds. v1 strides the reduced index space across a
/// workgroup's lanes (one workgroup per output and split); v2 lays the kept
/// index space across the lanes; v3 gives each row of a contiguous reduced
/// span a few lanes, several rows per workgroup. The plan chooses; the
/// caller's request block and the family's results contract do not change.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReduceKernel {
    V1,
    V2,
    V3,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReducePlanError {
    RankOutOfRange,
    ZeroExtent,
    AddressOutOfRange,
    ReducedMaskInvalid,
    InvalidScale,
    TooManyWorkgroups,
    CapacityTooSmall,
    /// A bit outside `STORAGE_FLAGS` was set in the request's flags word.
    StorageFlagsOutOfRange,
    /// `FLAG_F64` with an operation other than `Sum`, a finish other than
    /// `None`, or a bf16 bit beside it (schema 4).
    F64Unsupported,
}

impl fmt::Display for ReducePlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::RankOutOfRange => "reduce-rank-out-of-range",
            Self::ZeroExtent => "reduce-zero-extent",
            Self::AddressOutOfRange => "reduce-address-out-of-range",
            Self::ReducedMaskInvalid => "reduce-reduced-mask-invalid",
            Self::InvalidScale => "reduce-invalid-scale",
            Self::TooManyWorkgroups => "reduce-too-many-workgroups",
            Self::CapacityTooSmall => "reduce-capacity-too-small",
            Self::StorageFlagsOutOfRange => "reduce-storage-flags-out-of-range",
            Self::F64Unsupported => "reduce-f64-unsupported",
        })
    }
}

impl std::error::Error for ReducePlanError {}

/// A strided view of FP32 elements inside a buffer. Strides are element
/// strides and may be zero (broadcast) in any order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReduceView {
    pub shape: [u32; MAX_RANK],
    pub stride: [u32; MAX_RANK],
    pub ndim: usize,
    pub offset: u32,
}

impl ReduceView {
    pub fn new(shape: &[u32], stride: &[u32], offset: u32) -> Result<Self, ReducePlanError> {
        if shape.is_empty() || shape.len() > MAX_RANK || stride.len() != shape.len() {
            return Err(ReducePlanError::RankOutOfRange);
        }
        if shape.contains(&0) {
            return Err(ReducePlanError::ZeroExtent);
        }
        let mut view = Self {
            shape: [1; MAX_RANK],
            stride: [0; MAX_RANK],
            ndim: shape.len(),
            offset,
        };
        view.shape[..shape.len()].copy_from_slice(shape);
        view.stride[..stride.len()].copy_from_slice(stride);
        if view.last_address() >= MAX_ADDRESS {
            return Err(ReducePlanError::AddressOutOfRange);
        }
        Ok(view)
    }

    /// The largest element offset the view touches (inclusive).
    pub fn last_address(&self) -> u64 {
        let mut last = self.offset as u64;
        for d in 0..self.ndim {
            last += (self.shape[d] as u64 - 1) * self.stride[d] as u64;
        }
        last
    }

    pub fn elements(&self) -> u64 {
        self.shape[..self.ndim].iter().map(|&x| x as u64).product()
    }
}

/// One dispatch of the kernel: its push constants and grid.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReducePass {
    kernel: ReduceKernel,
    storage: u32,
    push: [u8; PUSH_BYTES_V2],
    push_len: usize,
    tile_cols: u32,
    lanes: u32,
    groups: [u32; 3],
    input_len: usize,
    output_len: usize,
}

impl ReducePass {
    /// Lanes per row under v3 (1..=256); zero under v1 and v2.
    pub fn lanes(&self) -> u32 {
        self.lanes
    }
    /// The push block the pass's kernel expects: 80 bytes for v1, 96 for v2 and v3.
    pub fn push_constants(&self) -> &[u8] {
        &self.push[..self.push_len]
    }
    pub fn kernel(&self) -> ReduceKernel {
        self.kernel
    }
    /// This pass's own storage bits: bit 0 the buffer it reads, bit 1 the
    /// buffer it writes. Derived by the plan, never the caller's directly.
    pub fn storage(&self) -> u32 {
        self.storage
    }
    /// The module this pass selects among the four each kernel has:
    /// 0 f32/f32 (the module every device builds at open), 1 bf16 in,
    /// 2 bf16 out, 3 both; and 4, `FLAG_F64`, v1's float64 module (schema 4),
    /// which no other kernel has.
    pub fn storage_module(&self) -> usize {
        self.storage as usize
    }
    /// Kept elements per workgroup under v2; zero under v1.
    pub fn tile_cols(&self) -> u32 {
        self.tile_cols
    }
    pub fn dispatch_groups(&self) -> [u32; 3] {
        self.groups
    }
    /// Minimum element count of the buffer bound at binding 0.
    pub fn input_len(&self) -> usize {
        self.input_len
    }
    /// Exact element count written at binding 1 (one writer per element).
    pub fn output_len(&self) -> usize {
        self.output_len
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReducePlan {
    view: ReduceView,
    reduced_mask: u32,
    op: ReduceOp,
    finish: Finish,
    scale: f32,
    kept_count: u64,
    reduced_count: u64,
    splits: u32,
    first: ReducePass,
    second: Option<ReducePass>,
    storage: u32,
}

fn push_bytes(
    op: ReduceOp,
    finish: Finish,
    view: &ReduceView,
    reduced_mask: u32,
    splits: u32,
    chunk: u32,
    scale: f32,
) -> [u8; PUSH_BYTES] {
    let mut words = [0u32; PUSH_BYTES / 4];
    words[0] = op as u32;
    words[1] = finish as u32;
    words[2] = view.ndim as u32;
    words[3] = reduced_mask;
    words[4..10].copy_from_slice(&view.shape);
    words[10..16].copy_from_slice(&view.stride);
    words[16] = view.offset;
    words[17] = splits;
    words[18] = chunk;
    words[19] = scale.to_bits();
    let mut bytes = [0u8; PUSH_BYTES];
    for (index, word) in words.iter().enumerate() {
        bytes[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
    }
    bytes
}

/// The geometry of a v2 first pass, once the plan has selected it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct V2Geometry {
    cdim: usize,
    cdim2: Option<usize>,
    tile_cols: u32,
    outer_kept: u64,
    k_flat: u64,
}

/// v2's block: v1's words, then cdim, cdim2, tile_cols, outer_kept.
fn push_bytes_v2(v1: &[u8; PUSH_BYTES], g: &V2Geometry) -> [u8; PUSH_BYTES_V2] {
    let mut bytes = [0u8; PUSH_BYTES_V2];
    bytes[..PUSH_BYTES].copy_from_slice(v1);
    let tail = [
        g.cdim as u32,
        g.cdim2.map_or(NONE_DIM, |d| d as u32),
        g.tile_cols,
        g.outer_kept as u32,
    ];
    for (index, word) in tail.iter().enumerate() {
        let at = PUSH_BYTES + index * 4;
        bytes[at..at + 4].copy_from_slice(&word.to_le_bytes());
    }
    bytes
}

/// Whether, and how, the first pass runs on v2.
///
/// The lanes run along the highest kept dim with unit stride and an extent
/// above one (`cdim`); a second kept dim whose stride is `cdim`'s extent is
/// flattened onto it (`cdim2`), so a contiguous [T, C] kept block is one
/// span of T x C consecutive elements. v2 is refused when the REDUCED side
/// owns the unit stride (v1's lanes coalesce there already), when the kept
/// span is narrower than `V2_MIN_KEPT`, when the y or z grid would exceed
/// `MAX_GRID_YZ`, or when no tile width reaches `V2_MIN_WORKGROUPS`
/// workgroups and the span is below `V2_MIN_KEPT_FOR_ANY_GRID`. The tile
/// width starts at the largest power of two not above the span (at most
/// 256) and halves until the grid reaches `V2_MIN_WORKGROUPS` or until a
/// row lane would have no row of its split to read.
fn select_v2(view: &ReduceView, reduced_mask: u32, splits: u32, chunk: u32) -> Option<V2Geometry> {
    let n = view.ndim;
    let kept = |d: usize| reduced_mask & (1 << d) == 0;
    if (0..n).any(|d| !kept(d) && view.stride[d] == 1 && view.shape[d] > 1) {
        return None;
    }
    let cdim = (0..n)
        .rev()
        .find(|&d| kept(d) && view.stride[d] == 1 && view.shape[d] > 1)?;
    let k_inner = view.shape[cdim] as u64;
    let cdim2 = (0..n)
        .rev()
        .find(|&d| d != cdim && kept(d) && view.shape[d] > 1 && view.stride[d] as u64 == k_inner);
    let k_flat = k_inner * cdim2.map_or(1, |d| view.shape[d] as u64);
    let outer_kept: u64 = (0..n)
        .filter(|&d| kept(d) && d != cdim && Some(d) != cdim2)
        .map(|d| view.shape[d] as u64)
        .product();
    if k_flat < V2_MIN_KEPT || outer_kept > MAX_GRID_YZ || u64::from(splits) > MAX_GRID_YZ {
        return None;
    }
    let cap = k_flat.min(u64::from(TILE_COLS_MAX));
    let mut tile = 1u64 << (63 - cap.leading_zeros());
    let rows_per_split = u64::from(chunk).max(1);
    let t_min = (256 / rows_per_split)
        .max(u64::from(TILE_COLS_MIN))
        .next_power_of_two()
        .min(u64::from(TILE_COLS_MAX));
    let groups = |t: u64| k_flat.div_ceil(t) * outer_kept * u64::from(splits);
    while tile > t_min && groups(tile) < V2_MIN_WORKGROUPS {
        tile /= 2;
    }
    if groups(tile) < V2_MIN_WORKGROUPS && k_flat < V2_MIN_KEPT_FOR_ANY_GRID {
        return None;
    }
    if k_flat.div_ceil(tile) > MAX_WORKGROUPS {
        return None;
    }
    Some(V2Geometry {
        cdim,
        cdim2,
        tile_cols: tile as u32,
        outer_kept,
        k_flat,
    })
}

/// A v2 column plan that is short of `TARGET_GROUPS` on one split, re-planned
/// to split its reduced rows (2026-09-26). On one split, v2 halves its tile
/// until 32 workgroups exist and then leaves each row lane a long serial walk:
/// a [4096, 512] gamma gradient ran 32 workgroups of 16 columns, each lane
/// summing 256 rows (81 us on an RX 9070 XT, against ROCm's 11). Here the
/// tile is `V2_SPLIT_TILE` columns (256 contiguous bytes per f32 row read)
/// and the rows split until the first pass reaches `TARGET_GROUPS`, keeping
/// at least `MIN_SPLIT_ROWS` rows per split. `None` when the plan needs no
/// split or cannot take one; the caller keeps its one-split plan.
fn select_v2_split(
    view: &ReduceView,
    reduced_mask: u32,
    reduced_count: u64,
) -> Option<(V2Geometry, u32, u32)> {
    let one = select_v2(view, reduced_mask, 1, u32::try_from(reduced_count).ok()?)?;
    let tiles_at = |t: u64| one.k_flat.div_ceil(t) * one.outer_kept;
    if tiles_at(u64::from(one.tile_cols)) >= TARGET_GROUPS || reduced_count < V2_SPLIT_MIN_ROWS {
        return None;
    }
    let cap = one.k_flat.min(u64::from(TILE_COLS_MAX));
    let tile = (1u64 << (63 - cap.leading_zeros())).min(V2_SPLIT_TILE);
    let tiles = tiles_at(tile);
    let want = TARGET_GROUPS.div_ceil(tiles);
    let most = reduced_count / MIN_SPLIT_ROWS;
    let splits = want.min(most).max(1);
    if splits < 2 {
        return None;
    }
    let chunk = reduced_count.div_ceil(splits);
    let splits = reduced_count.div_ceil(chunk);
    // every row lane of a split must own at least one row of it
    if 256 / tile > chunk || splits > MAX_GRID_YZ || one.outer_kept > MAX_GRID_YZ {
        return None;
    }
    Some((
        V2Geometry {
            tile_cols: tile as u32,
            ..one
        },
        splits as u32,
        chunk as u32,
    ))
}

/// Whether the reduced dims form one dense row-major span of unit stride, so
/// reduced index r sits at the row's base + r (extent-1 dims are ignored).
fn contiguous_reduced_span(view: &ReduceView, reduced_mask: u32) -> bool {
    let mut expected = 1u64;
    for d in (0..view.ndim).rev() {
        if reduced_mask & (1 << d) != 0 && view.shape[d] > 1 {
            if u64::from(view.stride[d]) != expected {
                return false;
            }
            expected *= u64::from(view.shape[d]);
        }
    }
    true
}

/// Lanes per v3 row for `rows` rows of `row_len` elements: see `V3_PER_LANE`.
fn v3_lanes(row_len: u64, rows: u64) -> u32 {
    let most = row_len.next_power_of_two().min(V3_MAX_LANES);
    let mut lanes = row_len
        .div_ceil(V3_PER_LANE)
        .max(1)
        .next_power_of_two()
        .min(most);
    while lanes < most && rows.div_ceil(256 / lanes) < TARGET_GROUPS {
        lanes *= 2;
    }
    lanes as u32
}

/// v3's (splits, chunk) for a contiguous span of `reduced_count` elements over
/// `kept_count` rows: one pass when the rows are short, or long but numerous
/// enough to fill `TARGET_GROUPS` workgroups; otherwise chunks of at least
/// `V3_CHUNK_MIN` elements and at most `V3_SPLITS_MAX` of them.
fn v3_split_law(kept_count: u64, reduced_count: u64) -> (u64, u64) {
    let rows_per_group = 256 / u64::from(v3_lanes(reduced_count, kept_count));
    let groups = kept_count.div_ceil(rows_per_group);
    if reduced_count <= V3_ROW_MAX || (reduced_count <= u64::from(CHUNK) && groups >= TARGET_GROUPS)
    {
        return (1, reduced_count);
    }
    let chunk = V3_CHUNK_MIN.max(reduced_count.div_ceil(V3_SPLITS_MAX));
    (reduced_count.div_ceil(chunk), chunk)
}

/// A v1 plan (a strided reduction neither v2 nor v3 takes) short of
/// `TARGET_GROUPS` workgroups splits its reduced range further, from
/// `V1_SPLIT_MIN` reduced elements, in chunks of at least `V1_CHUNK_MIN`
/// (2026-09-26). v1 gives each (output, split) one workgroup, so a reduction
/// to a few outputs ran a few workgroups: the per-head gradient sums of the
/// AKV pass (a [4, 8, 1024, 48] tensor kept to its 8 heads) took 24 workgroups
/// and 165 us on an RX 9070 XT. A function of the counts alone; the law never
/// splits LESS than CHUNK's law did.
fn v1_split_law(kept_count: u64, reduced_count: u64, splits: u32, chunk: u32) -> (u32, u32) {
    let splits64 = u64::from(splits);
    if kept_count * splits64 >= TARGET_GROUPS || reduced_count < V1_SPLIT_MIN {
        return (splits, chunk);
    }
    let want = TARGET_GROUPS.div_ceil(kept_count);
    let most = reduced_count / V1_CHUNK_MIN;
    let target = want.min(most).max(splits64);
    if target <= splits64 {
        return (splits, chunk);
    }
    let chunk = reduced_count.div_ceil(target);
    (reduced_count.div_ceil(chunk) as u32, chunk as u32)
}

/// v3's block: v1's words, then lanes, kept_count, reduced_count and a zero.
fn push_bytes_v3(
    v1: &[u8; PUSH_BYTES],
    lanes: u32,
    kept_count: u32,
    reduced_count: u32,
) -> [u8; PUSH_BYTES_V2] {
    let mut bytes = [0u8; PUSH_BYTES_V2];
    bytes[..PUSH_BYTES].copy_from_slice(v1);
    for (index, word) in [lanes, kept_count, reduced_count, 0].iter().enumerate() {
        let at = PUSH_BYTES + index * 4;
        bytes[at..at + 4].copy_from_slice(&word.to_le_bytes());
    }
    bytes
}

/// v3's grid for `rows` rows of `lanes` lanes: x up to 65,535, then y.
fn v3_groups(rows: u64, lanes: u32) -> [u32; 3] {
    let groups = rows.div_ceil(256 / u64::from(lanes));
    let x = groups.min(MAX_GRID_YZ);
    [x as u32, groups.div_ceil(x.max(1)) as u32, 1]
}

impl ReducePlan {
    /// `reduced_mask` bit `d` set means dimension `d` of `view` is reduced.
    /// `scale` is the finishing multiplier for `Finish::Scale` (a mean passes
    /// the reciprocal of the reduced count); it must be finite. `capacity` is
    /// the element count of the input buffer.
    pub fn new(
        view: ReduceView,
        reduced_mask: u32,
        op: ReduceOp,
        finish: Finish,
        scale: f32,
        capacity: u64,
    ) -> Result<Self, ReducePlanError> {
        Self::with_v2(view, reduced_mask, op, finish, scale, capacity, true)
    }

    /// As `new`, with v2 selection switched off when `allow_v2` is false:
    /// the same plan on v1's kernel for every pass, for a measured A/B and
    /// as the fallback a device that cannot run v2 would take.
    pub fn with_v2(
        view: ReduceView,
        reduced_mask: u32,
        op: ReduceOp,
        finish: Finish,
        scale: f32,
        capacity: u64,
        allow_v2: bool,
    ) -> Result<Self, ReducePlanError> {
        Self::with_storage(view, reduced_mask, op, finish, scale, capacity, allow_v2, 0)
    }

    /// As `with_v2`, with the caller's storage bits (`FLAG_IN_BF16`,
    /// `FLAG_OUT_BF16`). Each PASS gets its own pair derived from them:
    /// a one-pass plan takes both, and a two-pass plan takes the input bit
    /// on the first pass and the output bit on the second, because the
    /// block between them is partials and is always f32.
    ///
    /// `capacity` is in ELEMENTS of the input buffer, so a bf16 input's
    /// capacity is its byte length halved -- the caller converts, because
    /// only the caller knows the buffer.
    ///
    /// Eight arguments, which is one past clippy's threshold and is `with_v2`
    /// plus one word. Every one of them is REQUIRED to plan, none has a
    /// meaningful default, and a builder or an options struct would hide that
    /// by making each look optional -- the same reason `matmul_ops` and the
    /// probes carry this allow.
    #[allow(clippy::too_many_arguments)]
    pub fn with_storage(
        view: ReduceView,
        reduced_mask: u32,
        op: ReduceOp,
        finish: Finish,
        scale: f32,
        capacity: u64,
        allow_v2: bool,
        storage: u32,
    ) -> Result<Self, ReducePlanError> {
        if storage & !STORAGE_FLAGS != 0 {
            return Err(ReducePlanError::StorageFlagsOutOfRange);
        }
        // float64 (schema 4): a sum with no finish, on v1's plan alone (its
        // module is v1's), with float64 partials between two passes
        let f64_storage = storage & FLAG_F64 != 0;
        if f64_storage && (storage != FLAG_F64 || op != ReduceOp::Sum || finish != Finish::None) {
            return Err(ReducePlanError::F64Unsupported);
        }
        let allow_v2 = allow_v2 && !f64_storage;
        if reduced_mask == 0 || (reduced_mask >> view.ndim) != 0 {
            return Err(ReducePlanError::ReducedMaskInvalid);
        }
        if !scale.is_finite() {
            return Err(ReducePlanError::InvalidScale);
        }
        if view.last_address() >= capacity {
            return Err(ReducePlanError::CapacityTooSmall);
        }
        let mut kept_count = 1u64;
        let mut reduced_count = 1u64;
        for d in 0..view.ndim {
            if reduced_mask & (1 << d) != 0 {
                reduced_count *= view.shape[d] as u64;
            } else {
                kept_count *= view.shape[d] as u64;
            }
        }
        // THE SPLIT LAW (2026-09-26). Three kinds of plan, each a function of
        // the view alone, so the combination order stays fixed:
        // * the reduced dims are one contiguous span (v3): one pass for short
        //   rows, or for long rows numerous enough to fill `TARGET_GROUPS`
        //   workgroups; otherwise chunks (`v3_split_law`);
        // * the kept side owns a unit-stride dim (v2): CHUNK's law, and a
        //   one-split plan short of `TARGET_GROUPS` splits its rows for
        //   occupancy (`select_v2_split`);
        // * anything else (v1): CHUNK's law, and a plan short of
        //   `TARGET_GROUPS` splits further for occupancy (`v1_split_law`).
        // `allow_v2 = false` is the v1-everywhere arm for A/B and fallback,
        // and keeps CHUNK's law alone.
        let v3_first = allow_v2 && contiguous_reduced_span(&view, reduced_mask);
        let mut v2_split = None;
        let (splits, chunk) = if v3_first {
            let (splits, chunk) = v3_split_law(kept_count, reduced_count);
            (splits, chunk as u32)
        } else if reduced_count > CHUNK as u64 {
            (reduced_count.div_ceil(CHUNK as u64), CHUNK)
        } else {
            if allow_v2 {
                v2_split = select_v2_split(&view, reduced_mask, reduced_count);
            }
            match v2_split {
                Some((_, splits, chunk)) => (u64::from(splits), chunk),
                None => (1, reduced_count as u32),
            }
        };
        if splits > u32::MAX as u64
            || kept_count > u32::MAX as u64
            || kept_count * splits > MAX_WORKGROUPS
        {
            return Err(ReducePlanError::TooManyWorkgroups);
        }
        let splits = splits as u32;
        let v2 = if v3_first || !allow_v2 {
            None
        } else {
            v2_split
                .map(|(g, _, _)| g)
                .or_else(|| select_v2(&view, reduced_mask, splits, chunk))
        };
        // A plan left on v1 splits for occupancy too (`v1_split_law`).
        let (splits, chunk) = if allow_v2 && !v3_first && v2.is_none() {
            v1_split_law(kept_count, reduced_count, splits, chunk)
        } else {
            (splits, chunk)
        };
        let rows = kept_count * u64::from(splits);
        // a one-pass plan writes the caller's output; a first pass of two
        // writes PARTIALS, which are f32 whatever the caller's output is
        let first_storage = if splits == 1 {
            storage
        } else if f64_storage {
            FLAG_F64
        } else {
            storage & FLAG_IN_BF16
        };
        // the second pass reads the partials (f32, or float64 under FLAG_F64)
        // and writes the caller's output
        let second_storage = if f64_storage {
            FLAG_F64
        } else {
            storage & FLAG_OUT_BF16
        };
        let first_finish = if splits == 1 { finish } else { Finish::None };
        let v1_block = push_bytes(op, first_finish, &view, reduced_mask, splits, chunk, scale);
        let mut push = [0u8; PUSH_BYTES_V2];
        push[..PUSH_BYTES].copy_from_slice(&v1_block);
        let first = if v3_first {
            let lanes = v3_lanes(u64::from(chunk).min(reduced_count), rows);
            ReducePass {
                kernel: ReduceKernel::V3,
                storage: first_storage,
                push: push_bytes_v3(&v1_block, lanes, kept_count as u32, reduced_count as u32),
                push_len: PUSH_BYTES_V3,
                tile_cols: 0,
                lanes,
                groups: v3_groups(rows, lanes),
                input_len: (view.last_address() + 1) as usize,
                output_len: rows as usize,
            }
        } else {
            match v2 {
                Some(g) => ReducePass {
                    kernel: ReduceKernel::V2,
                    storage: first_storage,
                    push: push_bytes_v2(&v1_block, &g),
                    push_len: PUSH_BYTES_V2,
                    tile_cols: g.tile_cols,
                    lanes: 0,
                    groups: [
                        g.k_flat.div_ceil(u64::from(g.tile_cols)) as u32,
                        splits,
                        g.outer_kept as u32,
                    ],
                    input_len: (view.last_address() + 1) as usize,
                    output_len: rows as usize,
                },
                None => ReducePass {
                    kernel: ReduceKernel::V1,
                    storage: first_storage,
                    push,
                    push_len: PUSH_BYTES,
                    tile_cols: 0,
                    lanes: 0,
                    groups: [rows as u32, 1, 1],
                    input_len: (view.last_address() + 1) as usize,
                    output_len: rows as usize,
                },
            }
        };
        let second = if splits == 1 {
            None
        } else {
            let partial = ReduceView::new(&[kept_count as u32, splits], &[splits, 1], 0)?;
            let combine = match op {
                ReduceOp::Amax => ReduceOp::Amax,
                // the partials are magnitudes already; AbsMax combines them on
                // their bits, where Amax's float max could flush a subnormal
                ReduceOp::AbsMax => ReduceOp::AbsMax,
                ReduceOp::Sum | ReduceOp::SumOfSquares => ReduceOp::Sum,
            };
            let block = push_bytes(combine, finish, &partial, 0b10, 1, splits, scale);
            // the second pass reads `splits` CONTIGUOUS partials per output: a
            // contiguous span, so v3 whenever it is one row (v1 on the v1 arm,
            // or when a CHUNK-law plan has more partials than v3 takes)
            if allow_v2 && u64::from(splits) <= V3_ROW_MAX {
                let lanes = v3_lanes(u64::from(splits), kept_count);
                Some(ReducePass {
                    kernel: ReduceKernel::V3,
                    storage: second_storage,
                    push: push_bytes_v3(&block, lanes, kept_count as u32, splits),
                    push_len: PUSH_BYTES_V3,
                    tile_cols: 0,
                    lanes,
                    groups: v3_groups(kept_count, lanes),
                    input_len: rows as usize,
                    output_len: kept_count as usize,
                })
            } else {
                let mut push = [0u8; PUSH_BYTES_V2];
                push[..PUSH_BYTES].copy_from_slice(&block);
                Some(ReducePass {
                    kernel: ReduceKernel::V1,
                    storage: second_storage,
                    push,
                    push_len: PUSH_BYTES,
                    tile_cols: 0,
                    lanes: 0,
                    groups: [kept_count as u32, 1, 1],
                    input_len: rows as usize,
                    output_len: kept_count as usize,
                })
            }
        };
        Ok(Self {
            storage,
            view,
            reduced_mask,
            op,
            finish,
            scale,
            kept_count,
            reduced_count,
            splits,
            first,
            second,
        })
    }

    pub fn view(&self) -> &ReduceView {
        &self.view
    }
    pub fn reduced_mask(&self) -> u32 {
        self.reduced_mask
    }
    pub fn op(&self) -> ReduceOp {
        self.op
    }
    pub fn finish(&self) -> Finish {
        self.finish
    }
    pub fn scale(&self) -> f32 {
        self.scale
    }
    pub fn kept_count(&self) -> u64 {
        self.kept_count
    }
    pub fn reduced_count(&self) -> u64 {
        self.reduced_count
    }
    pub fn splits(&self) -> u32 {
        self.splits
    }
    /// The first dispatch: the input view to `kept_count * splits` partials,
    /// or the finished output when `splits == 1`.
    pub fn first(&self) -> &ReducePass {
        &self.first
    }
    /// The second dispatch over the partial buffer, present when `splits > 1`.
    pub fn second(&self) -> Option<&ReducePass> {
        self.second.as_ref()
    }
    /// Element count of the finished output: one per kept index.
    pub fn output_len(&self) -> usize {
        self.kept_count as usize
    }

    /// Reference computation in f64, in natural (row-major) order of the
    /// reduced index space. `input` must hold at least `first().input_len()`.
    pub fn reference(&self, input: &[f32]) -> Result<Vec<f64>, ReducePlanError> {
        if input.len() < self.first.input_len {
            return Err(ReducePlanError::CapacityTooSmall);
        }
        let v = &self.view;
        let mut out = Vec::with_capacity(self.kept_count as usize);
        for kept in 0..self.kept_count {
            let mut base = v.offset as u64;
            let mut rem = kept;
            for d in (0..v.ndim).rev() {
                if self.reduced_mask & (1 << d) == 0 {
                    let idx = rem % v.shape[d] as u64;
                    rem /= v.shape[d] as u64;
                    base += idx * v.stride[d] as u64;
                }
            }
            let mut acc = if matches!(self.op, ReduceOp::Amax | ReduceOp::AbsMax) {
                f64::NEG_INFINITY
            } else {
                0.0
            };
            let mut saw_nan = false;
            for r in 0..self.reduced_count {
                let mut address = base;
                let mut q = r;
                for d in (0..v.ndim).rev() {
                    if self.reduced_mask & (1 << d) != 0 {
                        let idx = q % v.shape[d] as u64;
                        q /= v.shape[d] as u64;
                        address += idx * v.stride[d] as u64;
                    }
                }
                let x = input[address as usize] as f64;
                match self.op {
                    ReduceOp::Sum => acc += x,
                    ReduceOp::SumOfSquares => acc += x * x,
                    ReduceOp::Amax => {
                        if x.is_nan() {
                            saw_nan = true;
                        }
                        acc = acc.max(x);
                    }
                    ReduceOp::AbsMax => {
                        if x.is_nan() {
                            saw_nan = true;
                        }
                        acc = acc.max(x.abs());
                    }
                }
            }
            if saw_nan {
                acc = f64::NAN;
            }
            out.push(match self.finish {
                Finish::None => acc,
                Finish::Scale => acc * self.scale as f64,
                Finish::Sqrt => acc.sqrt(),
            });
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(pass: &ReducePass) -> Vec<u32> {
        pass.push_constants()
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()
    }

    #[test]
    fn last_dim_sum_is_one_pass_with_dense_kept_output() {
        let view = ReduceView::new(&[1, 128, 768], &[98_304, 768, 1], 0).unwrap();
        let plan = ReducePlan::new(view, 0b100, ReduceOp::Sum, Finish::None, 1.0, 98_304).unwrap();
        assert_eq!(
            (plan.kept_count(), plan.reduced_count(), plan.splits()),
            (128, 768, 1)
        );
        assert!(plan.second().is_none());
        assert_eq!(plan.first().dispatch_groups(), [128, 1, 1]);
        assert_eq!(plan.first().output_len(), 128);
        assert_eq!(plan.first().input_len(), 98_304);
        let w = words(plan.first());
        assert_eq!(&w[..4], &[0, 0, 3, 0b100]);
        assert_eq!(&w[4..7], &[1, 128, 768]);
        assert_eq!(&w[10..13], &[98_304, 768, 1]);
        assert_eq!(&w[16..19], &[0, 1, 768]);
    }

    #[test]
    fn leading_dims_sum_keeps_the_trailing_dim_on_v2() {
        let view = ReduceView::new(&[1, 128, 768], &[98_304, 768, 1], 0).unwrap();
        let plan = ReducePlan::new(view, 0b011, ReduceOp::Sum, Finish::None, 1.0, 98_304).unwrap();
        assert_eq!((plan.kept_count(), plan.reduced_count()), (768, 128));
        // 2026-09-09: the kept span (768, unit stride) goes across the lanes; the
        // tile halves from 256 until 32 workgroups are reached: 16 columns x 48
        let first = plan.first();
        assert_eq!(first.kernel(), ReduceKernel::V2);
        assert_eq!(first.tile_cols(), 16);
        assert_eq!(first.dispatch_groups(), [48, 1, 1]);
        assert_eq!(first.push_constants().len(), PUSH_BYTES_V2);
        let w = words(first);
        assert_eq!(&w[..20], &words_v1(&view, 0b011, 1, 128)[..]);
        assert_eq!(&w[20..24], &[2, NONE_DIM, 16, 1]);
    }

    fn words_v1(view: &ReduceView, mask: u32, splits: u32, chunk: u32) -> Vec<u32> {
        push_bytes(ReduceOp::Sum, Finish::None, view, mask, splits, chunk, 1.0)
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()
    }

    /// The registered AKV pass's reductions, from a host census of one pass
    /// (2026-09-09): which kernel each selects and why, under the split law
    /// of 2026-09-26 (v3 rows for contiguous reduced spans, occupancy splits
    /// for v2 column plans).
    #[test]
    fn the_registered_passes_reductions_select_the_kernel_the_law_says() {
        let sum = |shape: &[u32], stride: &[u32], mask: u32| {
            let view = ReduceView::new(shape, stride, 0).unwrap();
            let cap = view.last_address() + 1;
            ReducePlan::new(view, mask, ReduceOp::Sum, Finish::None, 1.0, cap).unwrap()
        };
        // 25 per pass: the leading dims of a [1, 1024, 512] activation, keep 512.
        // One split would be 32 workgroups each walking 64 rows per lane; the
        // occupancy law takes 64-column tiles (8 of them) and 32 splits of 32
        // rows, 256 workgroups, then one v3 row of 32 partials per column.
        let p = sum(&[1, 1024, 512], &[524_288, 512, 1], 0b011);
        assert_eq!(
            (p.first().kernel(), p.first().tile_cols()),
            (ReduceKernel::V2, 64)
        );
        assert_eq!(p.splits(), 32);
        assert_eq!(p.first().dispatch_groups(), [8, 32, 1]);
        assert_eq!(p.first().output_len(), 512 * 32);
        let second = p.second().unwrap();
        assert_eq!((second.kernel(), second.lanes()), (ReduceKernel::V3, 32));
        assert_eq!(second.dispatch_groups(), [64, 1, 1]);
        // 25 per pass: the LAST dim of the same activation: the reduced side owns
        // the unit stride, so v3 rows: 64 lanes per 512-wide row, 4 rows per
        // workgroup, one pass (v1 gave every row its own 256-lane workgroup)
        let p = sum(&[1, 1024, 512], &[524_288, 512, 1], 0b100);
        assert_eq!(
            (p.first().kernel(), p.first().lanes()),
            (ReduceKernel::V3, 64)
        );
        assert_eq!(p.first().dispatch_groups(), [256, 1, 1]);
        assert!(p.second().is_none());
        // 12 per pass: the GQA repeat dim (extent 2) with a contiguous [1024, 64] kept block
        let p = sum(
            &[1, 4, 2, 1024, 64],
            &[524_288, 131_072, 65_536, 64, 1],
            0b00100,
        );
        let f = p.first();
        assert_eq!((f.kernel(), f.tile_cols()), (ReduceKernel::V2, 256));
        assert_eq!(
            f.dispatch_groups(),
            [256, 1, 4],
            "65,536 kept across 256 tiles, 4 outer"
        );
        assert_eq!(&words(f)[20..24], &[4, 3, 256, 4]);
        // 24 per pass: the same reduction over a transposed view, unit stride in the middle
        let p = sum(
            &[1, 4, 2, 1024, 48],
            &[393_216, 98_304, 49_152, 1, 1024],
            0b00100,
        );
        let f = p.first();
        assert_eq!((f.kernel(), f.tile_cols()), (ReduceKernel::V2, 256));
        assert_eq!(f.dispatch_groups(), [192, 1, 4]);
        assert_eq!(
            &words(f)[20..24],
            &[3, 4, 256, 4],
            "dim 3 across the lanes, dim 4 flattened on it"
        );
        let p = sum(
            &[1, 4, 2, 1024, 16],
            &[131_072, 32_768, 16_384, 1, 1024],
            0b00100,
        );
        assert_eq!(p.first().dispatch_groups(), [64, 1, 4]);
        // 36 per pass: leading dims keeping only 16 or 48 columns: too narrow
        // for v2, and the reduced dims are not one span, so v1
        let p = sum(&[1, 1024, 8, 48], &[393_216, 48, 49_152, 1], 0b0111);
        assert_eq!(p.first().kernel(), ReduceKernel::V1);
        let p = sum(&[1, 1024, 4, 16], &[65_536, 16, 16_384, 1], 0b0111);
        assert_eq!(p.first().kernel(), ReduceKernel::V1);
        // 12 per pass: an attention matrix summed over both of its last dims:
        // one contiguous 1,048,576-element span per kept index, chunked into
        // 512 rows of 2,048 on v3, then one v3 row of 512 partials per head
        let p = sum(
            &[1, 8, 1024, 1024],
            &[8_388_608, 1_048_576, 1024, 1],
            0b1100,
        );
        assert_eq!(p.first().kernel(), ReduceKernel::V3);
        assert_eq!(p.splits(), 512);
        assert_eq!(p.second().unwrap().kernel(), ReduceKernel::V3);
        // the probe's transposed heads-4 case: a 64-wide kept span whose grid never
        // reaches 32 and whose reduced dims are not one span: v1
        let p = sum(&[1, 128, 4, 64], &[32_768, 64, 8_192, 1], 0b0111);
        assert_eq!(p.first().kernel(), ReduceKernel::V1);
        // the probe's GQA case with a repeat extent of 3: 8,192 kept, one row lane per column
        let p = sum(
            &[1, 4, 3, 128, 64],
            &[98_304, 24_576, 8_192, 64, 1],
            0b00100,
        );
        assert_eq!(
            (p.first().kernel(), p.first().tile_cols()),
            (ReduceKernel::V2, 256)
        );
        assert_eq!(p.first().dispatch_groups(), [32, 1, 4]);
        // a full reduction to one element: one contiguous span, v3 in chunks
        let p = sum(&[4096, 4096], &[4096, 1], 0b11);
        assert_eq!(p.first().kernel(), ReduceKernel::V3);
        assert_eq!(p.second().unwrap().kernel(), ReduceKernel::V3);
    }

    /// A two-pass v2 plan writes [kept, splits] partials, the layout v1's
    /// first pass writes, and the second pass combines them as one contiguous
    /// row per kept index (v3 since 2026-09-26).
    #[test]
    fn a_two_pass_v2_plan_hands_its_partials_on_as_rows() {
        // [70_000, 512] reduced over dim 0: 512 kept, 70,000 reduced -> 2 splits
        let view = ReduceView::new(&[70_000, 512], &[512, 1], 0).unwrap();
        let plan =
            ReducePlan::new(view, 0b01, ReduceOp::Sum, Finish::Scale, 0.5, 70_000 * 512).unwrap();
        assert_eq!(plan.splits(), 2);
        let first = plan.first();
        assert_eq!(first.kernel(), ReduceKernel::V2);
        assert_eq!(first.dispatch_groups(), [512 / first.tile_cols(), 2, 1]);
        assert_eq!(first.output_len(), 1024, "kept x splits partials");
        assert_eq!(
            words(first)[1],
            Finish::None as u32,
            "the finish waits for the second pass"
        );
        let second = plan.second().unwrap();
        assert_eq!(second.kernel(), ReduceKernel::V3);
        assert_eq!(second.push_constants().len(), PUSH_BYTES_V3);
        let w = words(second);
        assert_eq!(
            &w[..4],
            &[ReduceOp::Sum as u32, Finish::Scale as u32, 2, 0b10]
        );
        assert_eq!(&w[4..6], &[512, 2], "a [kept, splits] partial view");
        assert_eq!(&w[10..12], &[2, 1]);
        assert_eq!(&w[20..24], &[2, 512, 2, 0], "lanes, kept, reduced, zero");
        assert_eq!(second.dispatch_groups(), [4, 1, 1]);
        // the v1-everywhere arm keeps v1's second pass
        let v1 = ReducePlan::with_v2(
            view,
            0b01,
            ReduceOp::Sum,
            Finish::Scale,
            0.5,
            70_000 * 512,
            false,
        )
        .unwrap();
        assert_eq!(v1.second().unwrap().kernel(), ReduceKernel::V1);
        assert_eq!(v1.second().unwrap().push_constants().len(), PUSH_BYTES);
    }

    #[test]
    fn a_large_norm_splits_deterministically_into_two_passes() {
        let view = ReduceView::new(&[32_768, 768], &[768, 1], 0).unwrap();
        let n = 32_768u64 * 768;
        let plan =
            ReducePlan::new(view, 0b11, ReduceOp::SumOfSquares, Finish::Sqrt, 1.0, n).unwrap();
        assert_eq!(plan.kept_count(), 1);
        assert_eq!(plan.reduced_count(), n);
        // one contiguous span: chunks of max(V3_CHUNK_MIN, n / V3_SPLITS_MAX)
        let chunk = V3_CHUNK_MIN.max(n.div_ceil(V3_SPLITS_MAX));
        assert_eq!(plan.splits() as u64, n.div_ceil(chunk));
        assert!(u64::from(plan.splits()) <= V3_SPLITS_MAX);
        assert_eq!(plan.first().kernel(), ReduceKernel::V3);
        assert_eq!(words(plan.first())[18] as u64, chunk);
        assert_eq!(
            words(plan.first())[1],
            Finish::None as u32,
            "the finish waits for the second pass"
        );
        let second = plan.second().expect("two passes");
        let w = words(second);
        assert_eq!(
            &w[..4],
            &[ReduceOp::Sum as u32, Finish::Sqrt as u32, 2, 0b10]
        );
        assert_eq!(&w[4..6], &[1, plan.splits()]);
        assert_eq!(&w[10..12], &[plan.splits(), 1]);
        assert_eq!(second.kernel(), ReduceKernel::V3);
        assert_eq!(second.dispatch_groups(), [1, 1, 1]);
        assert_eq!(second.input_len(), plan.splits() as usize);
        assert_eq!(second.output_len(), 1);
        // the same view again gives the same plan: the law reads nothing else
        let again =
            ReducePlan::new(view, 0b11, ReduceOp::SumOfSquares, Finish::Sqrt, 1.0, n).unwrap();
        assert_eq!(again, plan);
    }

    #[test]
    fn amax_partials_combine_with_amax() {
        // [2, 70_000] amax over the last dim: two contiguous rows of 70,000,
        // chunked into 35 rows of 2,048 each, then combined with amax
        let view = ReduceView::new(&[2, 70_000], &[70_000, 1], 0).unwrap();
        let plan = ReducePlan::new(view, 0b10, ReduceOp::Amax, Finish::None, 1.0, 140_000).unwrap();
        assert_eq!(plan.splits(), 35);
        assert_eq!(words(plan.second().unwrap())[0], ReduceOp::Amax as u32);
        assert_eq!(plan.first().output_len(), 70);
        assert_eq!(plan.second().unwrap().kernel(), ReduceKernel::V3);
    }

    #[test]
    fn a_float64_sum_plans_on_v1_with_float64_partials_and_nothing_else_is_admitted() {
        // schema 4: FLAG_F64 keeps v1 for every pass, whatever the view (a
        // contiguous last-dim span would take v3 in f32), and a two-pass plan's
        // partials are float64 on both sides of the split
        let view = ReduceView::new(&[2, 70_000], &[70_000, 1], 0).unwrap();
        let f64_plan = |op, finish, storage| {
            ReducePlan::with_storage(view, 0b10, op, finish, 1.0, 140_000, true, storage)
        };
        let plan = f64_plan(ReduceOp::Sum, Finish::None, FLAG_F64).unwrap();
        assert_eq!(plan.first().kernel(), ReduceKernel::V1);
        assert_eq!(plan.first().storage_module(), 4);
        // CHUNK's law alone, as on the v1-everywhere arm
        assert_eq!(plan.splits(), 70_000u32.div_ceil(CHUNK));
        let second = plan.second().expect("two passes");
        assert_eq!(second.kernel(), ReduceKernel::V1);
        assert_eq!(
            second.storage_module(),
            4,
            "float64 partials in, float64 out"
        );
        assert_eq!(words(second)[0], ReduceOp::Sum as u32);
        // a short reduction is one v1 pass, float64 in and out
        let short = ReduceView::new(&[3, 8], &[8, 1], 0).unwrap();
        let one = ReducePlan::with_storage(
            short,
            0b10,
            ReduceOp::Sum,
            Finish::None,
            1.0,
            24,
            true,
            FLAG_F64,
        )
        .unwrap();
        assert!(one.second().is_none());
        assert_eq!(
            (one.first().kernel(), one.first().storage_module()),
            (ReduceKernel::V1, 4)
        );
        // every other request under the bit is refused by name
        for (op, finish, storage) in [
            (ReduceOp::Amax, Finish::None, FLAG_F64),
            (ReduceOp::SumOfSquares, Finish::None, FLAG_F64),
            (ReduceOp::AbsMax, Finish::None, FLAG_F64),
            (ReduceOp::Sum, Finish::Scale, FLAG_F64),
            (ReduceOp::Sum, Finish::Sqrt, FLAG_F64),
            (ReduceOp::Sum, Finish::None, FLAG_F64 | FLAG_IN_BF16),
            (ReduceOp::Sum, Finish::None, FLAG_F64 | FLAG_OUT_BF16),
        ] {
            assert_eq!(
                f64_plan(op, finish, storage),
                Err(ReducePlanError::F64Unsupported),
                "{op:?} {finish:?} {storage:#b}"
            );
        }
        assert_eq!(
            ReducePlanError::F64Unsupported.to_string(),
            "reduce-f64-unsupported"
        );
        assert_eq!(
            f64_plan(ReduceOp::Sum, Finish::None, 0b1000),
            Err(ReducePlanError::StorageFlagsOutOfRange)
        );
        assert_eq!(storage_element_bytes(FLAG_F64, true), 8);
        assert_eq!(storage_element_bytes(FLAG_F64, false), 8);
        assert_eq!(storage_element_bytes(FLAG_IN_BF16, true), 2);
        assert_eq!(storage_element_bytes(FLAG_IN_BF16, false), 4);
    }

    #[test]
    fn abs_max_passes_both_carry_its_op() {
        // the partials are magnitudes already, and the second pass combines
        // them with the same bitwise max (a float max could flush a subnormal)
        let view = ReduceView::new(&[2, 70_000], &[70_000, 1], 0).unwrap();
        let plan =
            ReducePlan::new(view, 0b10, ReduceOp::AbsMax, Finish::None, 1.0, 140_000).unwrap();
        assert_eq!(words(plan.first())[0], ReduceOp::AbsMax as u32);
        assert_eq!(words(plan.second().unwrap())[0], ReduceOp::AbsMax as u32);
        // the host reference: the largest magnitude, a negative input's
        let mut x = vec![0.5f32; 140_000];
        x[69_999] = -7.25;
        x[70_000] = -0.0;
        let want = plan.reference(&x).unwrap();
        assert_eq!(want, vec![7.25, 0.5]);
        x[3] = f32::NAN;
        assert!(plan.reference(&x).unwrap()[0].is_nan());
    }

    /// v3's lane law: short rows share a workgroup; a small grid takes more
    /// lanes per row (up to one row per workgroup), never more than the row.
    #[test]
    fn v3_lanes_follow_the_row_and_the_grid() {
        assert_eq!(
            v3_lanes(512, 4096),
            64,
            "the AKV norm statistic: 4 rows per workgroup"
        );
        assert_eq!(
            v3_lanes(16, 32_768),
            2,
            "a 16-wide head norm: 128 rows per workgroup"
        );
        assert_eq!(v3_lanes(768, 128), 256, "few rows: one per workgroup");
        assert_eq!(v3_lanes(8, 1), 8, "never more lanes than the row");
        assert_eq!(v3_lanes(1, 1), 1);
        assert_eq!(
            v3_split_law(1024, 32_000),
            (1, 32_000),
            "long rows, numerous enough: one pass"
        );
        assert_eq!(
            v3_split_law(1, 1 << 24),
            (4096, 4096),
            "one long span: chunks, at most 4,096 of them"
        );
    }

    /// v1's occupancy law: a strided reduction to a few outputs splits until
    /// the first pass fills `TARGET_GROUPS`, in chunks of at least 2,048, and
    /// never below CHUNK's own split; a small reduction is left alone.
    #[test]
    fn v1_splits_a_strided_reduction_to_a_few_outputs() {
        // the per-head gradient sum: [4, 8, 1024, 48] kept to its 8 heads
        let view = ReduceView::new(&[4, 8, 1024, 48], &[393_216, 48, 384, 1], 0).unwrap();
        let p = ReducePlan::new(view, 0b1101, ReduceOp::Sum, Finish::None, 1.0, 1_572_864).unwrap();
        assert_eq!(p.first().kernel(), ReduceKernel::V1);
        assert_eq!((p.kept_count(), p.reduced_count()), (8, 196_608));
        assert_eq!(
            p.splits(),
            32,
            "8 x 32 = 256 workgroups (CHUNK's law gave 3)"
        );
        assert_eq!(p.first().dispatch_groups(), [256, 1, 1]);
        assert_eq!(p.second().unwrap().kernel(), ReduceKernel::V3);
        assert_eq!(v1_split_law(8, 196_608, 3, 65_536), (32, 6_144));
        assert_eq!(
            v1_split_law(48, 8_192, 1, 8_192),
            (4, 2_048),
            "as many as 2,048-chunks allow"
        );
        assert_eq!(
            v1_split_law(64, 512, 1, 512),
            (1, 512),
            "below 4,096 reduced: left alone"
        );
        assert_eq!(
            v1_split_law(300, 100_000, 2, 65_536),
            (2, 65_536),
            "a full grid: left alone"
        );
        // the v1-everywhere arm keeps CHUNK's law
        let v1 = ReducePlan::with_v2(
            view,
            0b1101,
            ReduceOp::Sum,
            Finish::None,
            1.0,
            1_572_864,
            false,
        )
        .unwrap();
        assert_eq!(v1.splits(), 3);
    }

    /// Only a dense row-major reduced block is one span; a broadcast or a
    /// kept dim between reduced dims is not.
    #[test]
    fn a_contiguous_reduced_span_is_recognised_exactly() {
        let view = ReduceView::new(&[4, 8, 16], &[128, 16, 1], 0).unwrap();
        assert!(contiguous_reduced_span(&view, 0b110));
        assert!(contiguous_reduced_span(&view, 0b100));
        assert!(!contiguous_reduced_span(&view, 0b101), "a kept dim between");
        assert!(
            !contiguous_reduced_span(&view, 0b011),
            "the unit stride is kept"
        );
        let broadcast = ReduceView::new(&[4, 2], &[0, 1], 0).unwrap();
        assert!(!contiguous_reduced_span(&broadcast, 0b01));
    }

    #[test]
    fn refusals_are_named() {
        assert_eq!(
            ReduceView::new(&[], &[], 0),
            Err(ReducePlanError::RankOutOfRange)
        );
        assert_eq!(
            ReduceView::new(&[1; 7], &[1; 7], 0),
            Err(ReducePlanError::RankOutOfRange)
        );
        assert_eq!(
            ReduceView::new(&[1, 2], &[1], 0),
            Err(ReducePlanError::RankOutOfRange)
        );
        assert_eq!(
            ReduceView::new(&[0], &[1], 0),
            Err(ReducePlanError::ZeroExtent)
        );
        assert_eq!(
            ReduceView::new(&[1 << 20, 1 << 12], &[1 << 12, 1], 0),
            Err(ReducePlanError::AddressOutOfRange)
        );
        let view = ReduceView::new(&[4, 4], &[4, 1], 0).unwrap();
        assert_eq!(
            ReducePlan::new(view, 0, ReduceOp::Sum, Finish::None, 1.0, 16),
            Err(ReducePlanError::ReducedMaskInvalid)
        );
        assert_eq!(
            ReducePlan::new(view, 0b100, ReduceOp::Sum, Finish::None, 1.0, 16),
            Err(ReducePlanError::ReducedMaskInvalid)
        );
        assert_eq!(
            ReducePlan::new(view, 0b1, ReduceOp::Sum, Finish::Scale, f32::NAN, 16),
            Err(ReducePlanError::InvalidScale)
        );
        assert_eq!(
            ReducePlan::new(view, 0b1, ReduceOp::Sum, Finish::None, 1.0, 15),
            Err(ReducePlanError::CapacityTooSmall)
        );
    }

    #[test]
    fn reference_follows_torch_semantics_on_a_transposed_view() {
        // A [2, 3] tensor viewed transposed as [3, 2] with strides [1, 3].
        let input: Vec<f32> = (1..=6).map(|x| x as f32).collect();
        let view = ReduceView::new(&[3, 2], &[1, 3], 0).unwrap();
        let sum_rows = ReducePlan::new(view, 0b10, ReduceOp::Sum, Finish::None, 1.0, 6).unwrap();
        assert_eq!(sum_rows.reference(&input).unwrap(), vec![5.0, 7.0, 9.0]);
        let mean_cols =
            ReducePlan::new(view, 0b01, ReduceOp::Sum, Finish::Scale, 1.0 / 3.0, 6).unwrap();
        let got = mean_cols.reference(&input).unwrap();
        assert!((got[0] - 2.0).abs() < 1e-6 && (got[1] - 5.0).abs() < 1e-6);
        let amax = ReducePlan::new(view, 0b11, ReduceOp::Amax, Finish::None, 1.0, 6).unwrap();
        assert_eq!(amax.reference(&input).unwrap(), vec![6.0]);
        let norm =
            ReducePlan::new(view, 0b11, ReduceOp::SumOfSquares, Finish::Sqrt, 1.0, 6).unwrap();
        assert!((norm.reference(&input).unwrap()[0] - 91f64.sqrt()).abs() < 1e-12);
        let nan_input = vec![1.0, f32::NAN, 3.0, 4.0, 5.0, 6.0];
        assert!(amax.reference(&nan_input).unwrap()[0].is_nan());
    }

    #[test]
    fn broadcast_strides_count_repeats() {
        let input = vec![2.0f32, 3.0];
        let view = ReduceView::new(&[4, 2], &[0, 1], 0).unwrap();
        let plan = ReducePlan::new(view, 0b01, ReduceOp::Sum, Finish::None, 1.0, 2).unwrap();
        assert_eq!(plan.reference(&input).unwrap(), vec![8.0, 12.0]);
    }
}
