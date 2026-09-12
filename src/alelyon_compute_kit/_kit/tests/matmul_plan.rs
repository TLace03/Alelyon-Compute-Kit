#[path = "../src/matmul_ops.rs"]
mod matmul_ops;

use matmul_ops::{
    MatmulKind, MatmulPlan, MatmulPlanError, MatrixView, ABI, ABI_V2, MAX_ADDRESSED_ELEMENTS,
    MAX_BATCH, MAX_DIM, MAX_K, PUSH_BYTES, SCHEMA, TILE, V2_TILE_K, V2_TILE_M, V2_TILE_N,
    V2_WORKGROUP_INVOCATIONS, WORKGROUP_INVOCATIONS,
};
use std::collections::BTreeMap;

#[derive(Clone, Copy)]
struct Profile {
    name: &'static str,
    rows: u32,
    model: u32,
    kv: u32,
    feed_forward: u32,
    vocab: u32,
    attention_batches: u32,
    seq: u32,
    head_dim: u32,
    layers: u32,
}

const PROFILES: [Profile; 3] = [
    Profile {
        name: "singleton",
        rows: 1,
        model: 32,
        kv: 16,
        feed_forward: 64,
        vocab: 64,
        attention_batches: 2,
        seq: 1,
        head_dim: 16,
        layers: 1,
    },
    Profile {
        name: "tiny",
        rows: 16,
        model: 32,
        kv: 16,
        feed_forward: 64,
        vocab: 64,
        attention_batches: 4,
        seq: 8,
        head_dim: 16,
        layers: 1,
    },
    Profile {
        name: "smoke",
        rows: 512,
        model: 128,
        kv: 64,
        feed_forward: 256,
        vocab: 2_048,
        attention_batches: 16,
        seq: 128,
        head_dim: 32,
        layers: 2,
    },
];

fn add_case(
    cases: &mut BTreeMap<(u32, u32, u32), (Vec<&'static str>, u32)>,
    label: &'static str,
    shape: (u32, u32, u32),
    count: u32,
) {
    let entry = cases.entry(shape).or_default();
    entry.0.push(label);
    entry.1 += count;
}

fn mm_cases(profile: Profile) -> BTreeMap<(u32, u32, u32), (Vec<&'static str>, u32)> {
    let r = profile.rows;
    let d = profile.model;
    let kv = profile.kv;
    let f = profile.feed_forward;
    let v = profile.vocab;
    let layers = profile.layers;
    let mut cases = BTreeMap::new();
    add_case(
        &mut cases,
        "q_o_forward_and_input_grad",
        (r, d, d),
        4 * layers,
    );
    add_case(&mut cases, "q_o_weight_grad", (d, d, r), 2 * layers);
    add_case(&mut cases, "kv_forward", (r, kv, d), 2 * layers);
    add_case(&mut cases, "kv_input_grad", (r, d, kv), 2 * layers);
    add_case(&mut cases, "kv_weight_grad", (kv, d, r), 2 * layers);
    add_case(&mut cases, "up_forward", (r, 2 * f, d), layers);
    add_case(&mut cases, "up_input_grad", (r, d, 2 * f), layers);
    add_case(&mut cases, "up_weight_grad", (2 * f, d, r), layers);
    add_case(&mut cases, "down_forward", (r, d, f), layers);
    add_case(&mut cases, "down_input_grad", (r, f, d), layers);
    add_case(&mut cases, "down_weight_grad", (d, f, r), layers);
    add_case(&mut cases, "head_forward_and_recompute", (r, v, d), 2);
    add_case(&mut cases, "head_input_grad", (r, d, v), 1);
    add_case(&mut cases, "head_weight_grad", (v, d, r), 1);
    cases
}

fn bmm_cases(profile: Profile) -> BTreeMap<(u32, u32, u32, u32), u32> {
    let mut cases = BTreeMap::new();
    cases.insert(
        (
            profile.attention_batches,
            profile.seq,
            profile.seq,
            profile.head_dim,
        ),
        2 * profile.layers,
    );
    cases.insert(
        (
            profile.attention_batches,
            profile.seq,
            profile.head_dim,
            profile.seq,
        ),
        3 * profile.layers,
    );
    cases.insert(
        (
            profile.attention_batches,
            profile.head_dim,
            profile.seq,
            profile.seq,
        ),
        profile.layers,
    );
    cases
}

fn compact(batch: u32, rows: u32, cols: u32) -> MatrixView {
    let elements = u64::from(batch) * u64::from(rows) * u64::from(cols);
    MatrixView::contiguous(batch, rows, cols, 0, elements).unwrap()
}

fn transposed(batch: u32, rows: u32, cols: u32) -> MatrixView {
    let elements = u64::from(batch) * u64::from(rows) * u64::from(cols);
    MatrixView::transposed_storage(batch, rows, cols, 0, elements).unwrap()
}

fn make_plan(
    kind: MatmulKind,
    batch: u32,
    m: u32,
    n: u32,
    k: u32,
    a_transposed: bool,
    b_transposed: bool,
) -> MatmulPlan {
    let a = if a_transposed {
        transposed(batch, m, k)
    } else {
        compact(batch, m, k)
    };
    let b = if b_transposed {
        transposed(batch, k, n)
    } else {
        compact(batch, k, n)
    };
    let output = compact(batch, m, n);
    MatmulPlan::new(kind, a, b, output).unwrap()
}

#[test]
fn all_gate_geometry_is_admitted_with_exact_source_derived_multiplicity() {
    let mut mm_unique = 0;
    let mut bmm_unique = 0;
    let expected_mm = [("singleton", 12, 22), ("tiny", 11, 22), ("smoke", 13, 40)];
    let expected_bmm = [("singleton", 3, 6), ("tiny", 3, 6), ("smoke", 3, 12)];

    for (profile_index, profile) in PROFILES.into_iter().enumerate() {
        let mm = mm_cases(profile);
        let bmm = bmm_cases(profile);
        assert_eq!(
            (
                profile.name,
                mm.len(),
                mm.values().map(|entry| entry.1).sum::<u32>()
            ),
            expected_mm[profile_index]
        );
        assert_eq!(
            (profile.name, bmm.len(), bmm.values().sum::<u32>()),
            expected_bmm[profile_index]
        );
        for (case_index, (&(m, n, k), (labels, multiplicity))) in mm.iter().enumerate() {
            assert!(!labels.is_empty() && *multiplicity > 0);
            let plan = make_plan(
                MatmulKind::Mm,
                1,
                m,
                n,
                k,
                case_index & 1 != 0,
                case_index & 2 != 0,
            );
            assert_eq!((plan.batch(), plan.m(), plan.n(), plan.k()), (1, m, n, k));
            assert_eq!(plan.kind(), MatmulKind::Mm);
            assert_eq!(
                plan.dispatch_groups(),
                [n.div_ceil(TILE), m.div_ceil(TILE), 1]
            );
        }
        for (&(batch, m, n, k), &multiplicity) in &bmm {
            assert!(multiplicity > 0);
            let plan = make_plan(MatmulKind::Bmm, batch, m, n, k, true, true);
            assert_eq!(
                (plan.batch(), plan.m(), plan.n(), plan.k()),
                (batch, m, n, k)
            );
            assert_eq!(plan.kind(), MatmulKind::Bmm);
            assert_eq!(
                plan.dispatch_groups(),
                [n.div_ceil(TILE), m.div_ceil(TILE), batch]
            );
        }
        mm_unique += mm.len();
        bmm_unique += bmm.len();
    }
    // Tiny's (R,D,KV) input-gradient and (KV,D,R) weight-gradient shapes
    // coincide at (16,32,16), so 22 calls reduce to 11 unique geometries.
    assert_eq!((mm_unique, bmm_unique), (36, 9));
}

#[test]
fn all_operand_layout_pairs_and_tile_tails_share_one_contract() {
    for (m, n, k) in [
        (1, 1, 1),
        (1, 3, 1),
        (3, 1, 5),
        (7, 9, 7),
        (8, 8, 8),
        (9, 15, 9),
        (17, 31, 31),
        (31, 33, 32),
        (33, 65, 33),
    ] {
        for (a_transposed, b_transposed) in
            [(false, false), (false, true), (true, false), (true, true)]
        {
            let plan = make_plan(MatmulKind::Mm, 1, m, n, k, a_transposed, b_transposed);
            assert_eq!(plan.dispatch_groups(), [n.div_ceil(8), m.div_ceil(8), 1]);
        }
    }
}

#[test]
fn padded_offsets_and_batched_row_or_column_major_outputs_are_non_overlapping() {
    let a = MatrixView::new(3, 7, 9, 5, 100, 12, 1, 300).unwrap();
    let b = MatrixView::new(3, 9, 5, 7, 90, 1, 11, 240).unwrap();
    let row_output = MatrixView::new(3, 7, 5, 11, 64, 8, 1, 192).unwrap();
    let column_output = MatrixView::new(3, 7, 5, 13, 70, 1, 10, 205).unwrap();

    for output in [row_output, column_output] {
        let plan = MatmulPlan::bmm(a, b, output).unwrap();
        assert_eq!(plan.dispatch_groups(), [1, 1, 3]);
        assert!(plan.a().required_elements() <= plan.a().capacity_elements());
        assert!(plan.b().required_elements() <= plan.b().capacity_elements());
        assert!(plan.output().required_elements() <= plan.output().capacity_elements());
    }
}

/// The element cap is REACHABLE EXACTLY, and neither an offset nor a row pad
/// can hide inside it.
///
/// The shape is `MAX_BATCH x MAX_DIM`, whose product is exactly
/// `MAX_ADDRESSED_ELEMENTS`. Both extents are named constants rather than
/// literals because a hand-derived literal is precisely what went stale when
/// the cap was raised: this test used to be spelled `512 x 2_048`, the AKV full
/// smoke head, because 512*2048 was the old cap of 2^20. That head no longer
/// reaches the cap, and the property being pinned was never about the head --
/// it is that the cap is touchable exactly and that nothing hides underneath.
///
/// This is also the host half of the boundary the device test in
/// `matmul_f32_abi.rs` cannot afford: any shaping of 2^28 f32 costs 1.00 GiB,
/// so only the arithmetic is established here.
#[test]
fn the_address_cap_is_reachable_exactly_and_cannot_hide_padding_or_an_offset() {
    assert_eq!(
        u64::from(MAX_BATCH) * u64::from(MAX_DIM),
        MAX_ADDRESSED_ELEMENTS,
        "the exact-cap fixture below is only exact while this identity holds"
    );
    let exact = MatrixView::contiguous(1, MAX_BATCH, MAX_DIM, 0, MAX_ADDRESSED_ELEMENTS).unwrap();
    assert_eq!(
        (exact.logical_elements(), exact.required_elements()),
        (MAX_ADDRESSED_ELEMENTS, MAX_ADDRESSED_ELEMENTS)
    );
    // the same shape, one element further along the buffer
    assert_eq!(
        MatrixView::contiguous(1, MAX_BATCH, MAX_DIM, 1, MAX_ADDRESSED_ELEMENTS + 1),
        Err(MatmulPlanError::AddressLimitExceeded)
    );
    // the same shape with one element of row padding: the LOGICAL count still
    // fits, the SPANNED count does not, and the cap governs both
    assert_eq!(
        MatrixView::new(
            1,
            MAX_BATCH,
            MAX_DIM,
            0,
            MAX_ADDRESSED_ELEMENTS,
            u64::from(MAX_DIM) + 1,
            1,
            MAX_ADDRESSED_ELEMENTS + u64::from(MAX_BATCH),
        ),
        Err(MatmulPlanError::AddressLimitExceeded)
    );
}

#[test]
fn capacity_and_full_width_address_arithmetic_fail_closed() {
    assert_eq!(
        MatrixView::contiguous(1, 2, 3, 4, 9),
        Err(MatmulPlanError::CapacityTooSmall)
    );
    assert_eq!(
        MatrixView::new(2, 2, 2, u64::MAX - 1, u64::MAX, 2, 1, u64::MAX),
        Err(MatmulPlanError::AddressOverflow)
    );
    // one row past MAX_DIM, written as MAX_DIM + 1 rather than 65_537: the
    // literal 4_097 that used to stand here is exactly what went stale when
    // the bound was raised, and it stopped refusing without saying so.
    assert_eq!(
        MatrixView::contiguous(1, MAX_DIM + 1, 1, 0, u64::from(MAX_DIM) + 1),
        Err(MatmulPlanError::DimensionOutOfRange)
    );
    // both extents inside MAX_DIM, their product one whole row past the
    // element cap: MAX_DIM x (MAX_BATCH + 1) is 268,500,992 against 2^28
    assert_eq!(
        MatrixView::contiguous(
            1,
            MAX_DIM,
            MAX_BATCH + 1,
            0,
            u64::from(MAX_DIM) * u64::from(MAX_BATCH + 1)
        ),
        Err(MatmulPlanError::AddressLimitExceeded)
    );
}

#[test]
fn zero_stride_and_implicit_batch_broadcast_are_refused() {
    assert_eq!(
        MatrixView::new(2, 3, 4, 0, 0, 4, 1, 24),
        Err(MatmulPlanError::ZeroStride)
    );
    let a = compact(2, 3, 4);
    let b_one_batch = compact(1, 4, 5);
    let output = compact(2, 3, 5);
    assert_eq!(
        MatmulPlan::bmm(a, b_one_batch, output),
        Err(MatmulPlanError::BatchMismatch)
    );
    let b = compact(2, 4, 5);
    assert_eq!(
        MatmulPlan::mm(a, b, output),
        Err(MatmulPlanError::MmRequiresSingleBatch)
    );
}

#[test]
fn output_overlap_is_refused_while_read_only_input_overlap_is_well_defined() {
    let overlapping_input = MatrixView::new(1, 2, 2, 0, 4, 1, 1, 3).unwrap();
    let b = compact(1, 2, 2);
    let output = compact(1, 2, 2);
    assert!(MatmulPlan::mm(overlapping_input, b, output).is_ok());

    let overlapping_output = MatrixView::new(1, 2, 2, 0, 4, 1, 1, 3).unwrap();
    assert_eq!(
        MatmulPlan::mm(compact(1, 2, 2), b, overlapping_output),
        Err(MatmulPlanError::OutputOverlap)
    );
    let cross_batch_overlap = MatrixView::new(2, 2, 2, 0, 3, 2, 1, 7).unwrap();
    assert_eq!(
        MatmulPlan::bmm(compact(2, 2, 2), compact(2, 2, 2), cross_batch_overlap),
        Err(MatmulPlanError::OutputOverlap)
    );
}

#[test]
fn matrix_shape_mismatches_refuse_before_push_bytes_exist() {
    let a = compact(1, 2, 3);
    let bad_reduction = compact(1, 4, 5);
    let output = compact(1, 2, 5);
    assert_eq!(
        MatmulPlan::mm(a, bad_reduction, output),
        Err(MatmulPlanError::DimensionMismatch)
    );

    let b = compact(1, 3, 5);
    let bad_output = compact(1, 3, 5);
    assert_eq!(
        MatmulPlan::mm(a, b, bad_output),
        Err(MatmulPlanError::DimensionMismatch)
    );
}

#[test]
fn push_layout_and_public_getters_are_exact_and_little_endian() {
    let a = MatrixView::new(2, 3, 4, 5, 40, 6, 1, 70).unwrap();
    let b = MatrixView::new(2, 4, 5, 7, 50, 1, 8, 93).unwrap();
    let output = MatrixView::new(2, 3, 5, 11, 60, 7, 1, 95).unwrap();
    let plan = MatmulPlan::bmm(a, b, output).unwrap();
    let bytes = plan.push_constants();
    let words: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
        .collect();
    assert_eq!(bytes.len(), PUSH_BYTES);
    assert_eq!(words, [2, 3, 5, 4, 5, 40, 6, 1, 7, 50, 1, 8, 11, 60, 7, 1,]);
    assert_eq!(
        (
            plan.a().batch(),
            plan.a().rows(),
            plan.a().cols(),
            plan.a().offset(),
            plan.a().batch_stride(),
            plan.a().row_stride(),
            plan.a().col_stride(),
        ),
        (2, 3, 4, 5, 40, 6, 1)
    );
    assert_eq!((ABI, TILE, WORKGROUP_INVOCATIONS), ("matmul-f32/v1", 8, 64));
}

#[test]
fn operation_tags_and_named_errors_are_closed() {
    assert_eq!(MatmulKind::try_from(0), Ok(MatmulKind::Mm));
    assert_eq!(MatmulKind::try_from(1), Ok(MatmulKind::Bmm));
    // schema 2 (2026-09-09): the plain-fp32 v2 kernel behind kinds 2 and 3
    assert_eq!(MatmulKind::try_from(2), Ok(MatmulKind::MmPlain));
    assert_eq!(MatmulKind::try_from(3), Ok(MatmulKind::BmmPlain));
    assert_eq!(
        MatmulKind::try_from(4),
        Err(MatmulPlanError::OperationOutOfRange)
    );
    assert_eq!(
        SCHEMA, 2,
        "a new kind value is a schema move; this pin moves with it"
    );
    assert_eq!(ABI_V2, "matmul-f32/v2");
    assert!(MatmulKind::MmPlain.is_plain() && MatmulKind::BmmPlain.is_plain());
    assert!(!MatmulKind::Mm.is_plain() && !MatmulKind::Bmm.is_plain());
    assert!(MatmulKind::Bmm.is_batched() && MatmulKind::BmmPlain.is_batched());
    assert!(!MatmulKind::Mm.is_batched() && !MatmulKind::MmPlain.is_batched());
    assert_eq!(
        MatmulPlanError::OutputOverlap.to_string(),
        "matmul-output-overlap"
    );
    assert_eq!(
        MatmulPlanError::CapacityTooSmall.to_string(),
        "matmul-capacity-too-small"
    );
}

/// The plain kinds cut the dispatch grid to v2's 64 x 64 tile and refuse
/// exactly what the exact kinds refuse: same views, same block, same law.
///
/// The grid is the one thing the plan changes per kind. A grid cut to v1's
/// 8 x 8 tile and handed to v2 would launch 64x too many workgroups (slow but
/// harmless); a grid cut to v2's tile and handed to v1 would leave 63 of every
/// 64 output rows and columns UNWRITTEN while the call reported ACK_OK. So the
/// pairing is pinned here, and `ack_matmul_f32` binds the kernel by the same
/// `is_plain()` the plan cut its grid by.
#[test]
fn plain_kinds_cut_the_grid_to_the_v2_tile_and_share_every_refusal() {
    assert_eq!(V2_TILE_M, 64);
    assert_eq!(V2_TILE_N, 64);
    assert_eq!(V2_TILE_K, 16);
    assert_eq!(
        V2_WORKGROUP_INVOCATIONS, 256,
        "16 x 16 invocations own a 64 x 64 tile"
    );
    for (batch, m, n, k) in [
        (1, 1, 1, 1),
        (1, 64, 64, 16),
        (1, 65, 130, 17),
        (3, 70, 130, 50),
        (8, 1024, 1024, 64),
    ] {
        let exact = make_plan(MatmulKind::Bmm, batch, m, n, k, false, false);
        let plain = make_plan(MatmulKind::BmmPlain, batch, m, n, k, false, false);
        assert_eq!(
            exact.dispatch_groups(),
            [n.div_ceil(TILE), m.div_ceil(TILE), batch],
            "v1 keeps its 8 x 8 grid"
        );
        assert_eq!(
            plain.dispatch_groups(),
            [n.div_ceil(V2_TILE_N), m.div_ceil(V2_TILE_M), batch],
            "v2's grid: one workgroup per 64 x 64 output tile per batch"
        );
        assert_eq!(
            exact.push_constants(),
            plain.push_constants(),
            "the same 64-byte block"
        );
        assert_eq!(plain.kind(), MatmulKind::BmmPlain);
        // the named constructors are the kinds they name
        let (a, b, c) = (
            compact(batch, m, k),
            compact(batch, k, n),
            compact(batch, m, n),
        );
        assert_eq!(
            MatmulPlan::bmm_plain(a, b, c).map(|p| (p.kind(), p.dispatch_groups())),
            Ok((MatmulKind::BmmPlain, plain.dispatch_groups()))
        );
        if batch == 1 {
            assert_eq!(
                MatmulPlan::mm_plain(a, b, c).map(|p| p.kind()),
                Ok(MatmulKind::MmPlain)
            );
        }
        assert_eq!(
            make_plan(MatmulKind::Mm, 1, m, n, k, true, true).push_constants(),
            make_plan(MatmulKind::MmPlain, 1, m, n, k, true, true).push_constants(),
            "transposed operands: the same block under either contract"
        );
    }
    // a single-product kind with a batch above one is refused under either contract
    for kind in [MatmulKind::Mm, MatmulKind::MmPlain] {
        assert_eq!(
            MatmulPlan::new(kind, compact(2, 8, 8), compact(2, 8, 8), compact(2, 8, 8))
                .map(|p| p.kind()),
            Err(MatmulPlanError::MmRequiresSingleBatch),
            "{kind:?}"
        );
    }
    // a reduction-length mismatch is the shared view law, whatever the kind
    for kind in [MatmulKind::Bmm, MatmulKind::BmmPlain] {
        assert_eq!(
            MatmulPlan::new(kind, compact(1, 8, 8), compact(1, 9, 8), compact(1, 8, 8))
                .map(|p| p.kind()),
            Err(MatmulPlanError::DimensionMismatch),
            "{kind:?}"
        );
    }
}

/// The reduction length is bounded SEPARATELY from the other two extents.
///
/// `MatrixView` cannot see which of its extents is `k` -- for A it is the
/// column count, for B the row count -- so only `MatmulPlan` can enforce
/// `MAX_K`. The two constants are equal today, so this test cannot construct a
/// case that only `MAX_K` refuses; what it pins is that a plan at the bound is
/// accepted and carries `k` through unchanged, and that the relationship a
/// future raise has to respect is written where that raise will read it.
#[test]
fn the_reduction_length_is_bounded_by_its_own_constant() {
    // k at the bound exactly, m and n small enough to stay inside the cap
    let a = MatrixView::contiguous(1, 8, MAX_K, 0, 8 * u64::from(MAX_K)).unwrap();
    let b = MatrixView::contiguous(1, MAX_K, 8, 0, u64::from(MAX_K) * 8).unwrap();
    let c = MatrixView::contiguous(1, 8, 8, 0, 64).unwrap();
    let plan = MatmulPlan::mm(a, b, c).unwrap();
    assert_eq!((plan.m(), plan.n(), plan.k()), (8, 8, MAX_K));
}

// ---------------------------------------------------------------------------
// The capacity bounds live in several copies. This is the in-crate half of the
// parity guard, and the half that reaches the public SDK export.
// ---------------------------------------------------------------------------

/// The `.comp` source and the `.spv` that gets dispatched, as the crate itself
/// sees them: `src/ffi.rs` embeds these exact bytes with `include_bytes!`, so
/// this test cannot be fooled by another worktree's kernels directory.
const MATMUL_COMP: &str = include_str!("../kernels/matmul_f32.comp");
const MATMUL_SPV: &[u8] = include_bytes!("../kernels/matmul_f32.spv");
/// matmul-f32/v2 carries the same capacity bounds as copies six and seven, so
/// it is held to the same parity below.
const MATMUL_COMP_V2: &str = include_str!("../kernels/matmul_f32_v2.comp");
const MATMUL_SPV_V2: &[u8] = include_bytes!("../kernels/matmul_f32_v2.spv");

/// The single NUMERIC literal that follows `needle` in `matmul_f32.comp`.
///
/// Occurrences of `needle` that are not followed by a digit are skipped rather
/// than counted: the same comparisons appear against `LIMIT` in the shader's
/// staged division guards (`p.batch > LIMIT / p.m`), and those are not bound
/// literals. What must be exactly one is the count of NUMERIC ones.
///
/// Deliberately hand-written: this crate has ONE dependency (ash), and taking
/// on regex to read five numbers would be the worse trade. It is fail-closed on
/// both sides -- zero numeric occurrences and more than one are equally a
/// failure, because a guard that silently stops matching manufactures the
/// agreement it exists to check.
fn only_literal_after(needle: &str) -> u64 {
    only_literal_in(MATMUL_COMP, "matmul_f32.comp", needle)
}

/// The same reader over any shader text, named for its messages.
fn only_literal_in(text: &str, file: &str, needle: &str) -> u64 {
    let mut found: Option<u64> = None;
    for (index, _) in text.match_indices(needle) {
        let rest = &text[index + needle.len()..];
        let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
        if digits.is_empty() {
            continue; // a comparison against LIMIT, not against a literal
        }
        assert!(
            found.is_none(),
            "{needle:?} is followed by a number more than once in {file}: a second copy of a \
             capacity bound appeared that this parity table does not cover"
        );
        found = Some(digits.parse().expect("a bound literal that fits u64"));
    }
    found.unwrap_or_else(|| {
        panic!(
            "{needle:?} is never followed by a number in {file}. The shader was refactored and \
             this guard stopped guarding; fix the guard rather than deleting it."
        )
    })
}

/// Every OpConstant carrying a one-word literal, from a real instruction walk.
///
/// The magic number is checked FIRST: a file that is not SPIR-V must fail here
/// rather than yield an empty set that would read as agreement.
fn spirv_single_word_constants(blob: &[u8]) -> Vec<u32> {
    assert!(
        blob.len() >= 20 && blob.len() % 4 == 0,
        "matmul_f32.spv is not a SPIR-V module: {} bytes",
        blob.len()
    );
    let words: Vec<u32> = blob
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(words[0], 0x0723_0203, "bad SPIR-V magic in matmul_f32.spv");
    let mut constants = Vec::new();
    let mut index = 5; // skip the five-word header
    while index < words.len() {
        let opcode = words[index] & 0xffff;
        let count = (words[index] >> 16) as usize;
        assert!(count > 0, "zero-length SPIR-V instruction at word {index}");
        assert!(
            index + count <= words.len(),
            "SPIR-V instruction at word {index} runs past the module"
        );
        // OpConstant is 43: result type, result id, then the literal words
        if opcode == 43 && count == 4 {
            constants.push(words[index + 3]);
        }
        index += count;
    }
    constants
}

/// The shader's capacity literals are the same numbers as this module's
/// constants, and the compiled module still carries them.
///
/// WHY THIS EXISTS. `matmul_f32.comp`'s bound guard `return`s having written
/// NOTHING, while `ack_matmul_f32` still reports ACK_OK and the torch backend
/// still counts the product as carried. A shader copy sitting BELOW the
/// planner's therefore makes every product in the gap return whatever was
/// already in the output buffer, with every counter reporting success. Nothing
/// enforced the copies against each other before this test existed.
///
/// WHAT IT DOES NOT ESTABLISH. It is a DRIFT detector, not a justification: it
/// will happily certify agreement at a bound nobody earned. It reads a literal
/// beside a fixed pattern, so a `p.m`/`p.n` swap, or `>` becoming `>=`, is
/// invisible to it. The SPIR-V half is a PRESENCE check on the constant pool --
/// it cannot say which comparison consumes a constant, and an optimiser rewrite
/// of `> 65536u` into `>= 65537u` would fail it with no defect present. And it
/// says nothing about the shader's OTHER silent returns (`valid_extent`,
/// `valid_output_layout`, `batch_index >= p.batch`), which write nothing on
/// exactly the same terms and are not bound literals at all.
///
/// The upstream repository runs a wider version of this over the Python and C++
/// mirrors as well; those files are not part of this crate.
#[test]
fn the_capacity_bounds_are_identical_in_every_copy() {
    let table: [(&str, u64, &str, u64); 5] = [
        (
            "p.batch",
            only_literal_after("p.batch > "),
            "MAX_BATCH",
            u64::from(MAX_BATCH),
        ),
        (
            "p.m",
            only_literal_after("p.m > "),
            "MAX_DIM",
            u64::from(MAX_DIM),
        ),
        (
            "p.n",
            only_literal_after("p.n > "),
            "MAX_DIM",
            u64::from(MAX_DIM),
        ),
        (
            "p.k",
            only_literal_after("p.k > "),
            "MAX_K",
            u64::from(MAX_K),
        ),
        (
            "LIMIT",
            only_literal_after("const uint LIMIT = "),
            "MAX_ADDRESSED_ELEMENTS",
            MAX_ADDRESSED_ELEMENTS,
        ),
    ];
    for (which, got, name, want) in table {
        assert_eq!(
            got, want,
            "matmul_f32.comp's {which} bound is {got}, but matmul_ops::{name} is {want}. A shader \
             copy below the planner's is SILENT: the dispatch writes nothing and the host still \
             reports ACK_OK."
        );
    }

    // the artifact that actually runs still carries every one of them
    let constants = spirv_single_word_constants(MATMUL_SPV);
    assert!(
        !constants.is_empty(),
        "the OpConstant walk over matmul_f32.spv found nothing: a scan that finds nothing must \
         never read as agreement"
    );
    for (name, value) in [
        ("MAX_BATCH", MAX_BATCH),
        ("MAX_DIM", MAX_DIM),
        ("MAX_K", MAX_K),
        (
            "MAX_ADDRESSED_ELEMENTS",
            u32::try_from(MAX_ADDRESSED_ELEMENTS).expect("MAX_ADDRESSED_ELEMENTS must fit u32"),
        ),
    ] {
        assert!(
            constants.contains(&value),
            "matmul_f32.spv's constant pool does not contain {name} = {value}. The shipped shader \
             was not rebuilt from the current bounds, and every product in the gap would silently \
             write nothing while reporting success."
        );
    }
}

// The raised bounds still fit the arithmetic the SHADER does in 32 bits.
//
// These are facts about constants, so they are checked at COMPILE TIME rather
// than in a test: a `#[test]` asserting them would be one clippy lint away from
// being deleted, and a build that cannot produce a wrong-bounds binary at all
// is the stronger guarantee. Each one fails the build with its own message.
//
// `MAX_BATCH * MAX_DIM * MAX_DIM` is deliberately NOT required to fit. The
// shader's staged divisions (`p.batch > LIMIT / p.m`, then
// `p.batch * p.m > LIMIT / p.k`) exist precisely so that product is never
// formed.

/// The shader holds `LIMIT` in a `uint`.
const _: () = assert!(
    MAX_ADDRESSED_ELEMENTS <= u32::MAX as u64,
    "MAX_ADDRESSED_ELEMENTS does not fit the shader's 32-bit LIMIT"
);

/// The shader computes `p.batch * p.m` in `uint`; the planner does the same
/// arithmetic in u64 and would not notice a wrap.
const _: () = assert!(
    MAX_BATCH as u64 * MAX_DIM as u64 <= u32::MAX as u64,
    "MAX_BATCH * MAX_DIM wraps the shader's 32-bit `p.batch * p.m`"
);

/// `MAX_BATCH * MAX_DIM` is EXACTLY the element cap, so there is room against
/// u32 but none at all against the cap: the next raise of either constant has
/// to move the other down or change the shader's staged arithmetic. The
/// exact-cap fixture above and the headroom note in matmul_ops.rs both assume
/// this identity holds.
const _: () = assert!(
    MAX_BATCH as u64 * MAX_DIM as u64 == MAX_ADDRESSED_ELEMENTS,
    "MAX_BATCH * MAX_DIM no longer equals MAX_ADDRESSED_ELEMENTS"
);

/// `MatrixView` refuses an extent above `MAX_DIM` before `MatmulPlan`'s own
/// `MAX_K` check can be reached, so raising `MAX_K` past `MAX_DIM` would do
/// nothing without also widening `MatrixView`.
const _: () = assert!(
    MAX_K <= MAX_DIM,
    "MAX_K exceeds MAX_DIM, so the reduction bound is unreachable"
);

/// matmul-f32/v2 mirrors every capacity literal v1 mirrors, its tile constants
/// are the plan's grid constants, and the shipped v2 module carries the bounds:
/// a v2 copy below the planner's would refuse silently on exactly the terms
/// v1's would, for the products the registered policy now routes through v2.
#[test]
fn the_capacity_bounds_are_identical_in_every_copy_of_v2_too() {
    const FILE: &str = "matmul_f32_v2.comp";
    let table: [(&str, u64, &str, u64); 5] = [
        (
            "p.batch",
            only_literal_in(MATMUL_COMP_V2, FILE, "p.batch > "),
            "MAX_BATCH",
            u64::from(MAX_BATCH),
        ),
        (
            "p.m",
            only_literal_in(MATMUL_COMP_V2, FILE, "p.m > "),
            "MAX_DIM",
            u64::from(MAX_DIM),
        ),
        (
            "p.n",
            only_literal_in(MATMUL_COMP_V2, FILE, "p.n > "),
            "MAX_DIM",
            u64::from(MAX_DIM),
        ),
        (
            "p.k",
            only_literal_in(MATMUL_COMP_V2, FILE, "p.k > "),
            "MAX_K",
            u64::from(MAX_K),
        ),
        (
            "LIMIT",
            only_literal_in(MATMUL_COMP_V2, FILE, "const uint LIMIT = "),
            "MAX_ADDRESSED_ELEMENTS",
            MAX_ADDRESSED_ELEMENTS,
        ),
    ];
    for (which, got, name, want) in table {
        assert_eq!(
            got, want,
            "matmul_f32_v2.comp's {which} bound is {got}, but matmul_ops::{name} is {want}: a \
             shader copy below the planner's writes nothing while the host reports ACK_OK."
        );
    }
    // the shader's tile constants are the plan's grid constants
    for (needle, want) in [
        ("const uint BM = ", V2_TILE_M),
        ("const uint BN = ", V2_TILE_N),
        ("const uint BK = ", V2_TILE_K),
        ("const uint INVOCATIONS = ", V2_WORKGROUP_INVOCATIONS),
    ] {
        assert_eq!(
            only_literal_in(MATMUL_COMP_V2, FILE, needle),
            u64::from(want),
            "{needle:?} in matmul_f32_v2.comp does not match the plan's grid constant"
        );
    }
    assert!(
        MATMUL_COMP_V2.contains("local_size_x = 16, local_size_y = 16, local_size_z = 1"),
        "the v2 workgroup is 16 x 16 x 1 invocations"
    );
    let constants = spirv_single_word_constants(MATMUL_SPV_V2);
    assert!(
        !constants.is_empty(),
        "the OpConstant walk over matmul_f32_v2.spv found nothing"
    );
    for (name, value) in [
        ("MAX_BATCH", MAX_BATCH),
        ("MAX_DIM", MAX_DIM),
        ("MAX_K", MAX_K),
        (
            "MAX_ADDRESSED_ELEMENTS",
            u32::try_from(MAX_ADDRESSED_ELEMENTS).expect("MAX_ADDRESSED_ELEMENTS must fit u32"),
        ),
        ("V2_TILE_M", V2_TILE_M),
        ("V2_TILE_K", V2_TILE_K),
    ] {
        assert!(
            constants.contains(&value),
            "matmul_f32_v2.spv's constant pool does not contain {name} = {value}: the shipped v2 \
             module was not rebuilt from the current bounds"
        );
    }
}
