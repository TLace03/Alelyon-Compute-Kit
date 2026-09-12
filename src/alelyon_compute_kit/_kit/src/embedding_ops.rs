//! Pure validation and dispatch planning for the experimental embedding-f32/v1 ABI.
//! Bounds describe this implementation, not Vulkan or a selected GPU. Gather
//! binds weights, token IDs, a dummy offset word, and token vectors. Backward
//! binds upstream token vectors, stable token positions grouped by vocabulary
//! ID, CSR offsets, and the complete vocabulary gradient. The grouped positions
//! retain original token order so each gradient lane can sum without atomics.

use std::fmt;

/// v3 (2026-09-08): raised for the 1B `xl` geometry, which the v2 bounds
/// refused outright. THESE ARE POLICY NUMBERS, NOT HARDWARE ONES, and the
/// hardware was measured before they moved rather than after.
///
/// What v2 refused, and by how much:
///
/// | bound | v2 | `xl` needs | |
/// |---|---:|---:|---|
/// | `MAX_DIM` | 1,024 | 2,048 (`d_model`) | 2x over |
/// | `MAX_ELEMENTS` | 1<<26 | 67,108,864 (32,768 x 2,048) | **exactly 2^26** — passed by zero margin |
/// | `MAX_N` | 4,096 | 4,096 (micro 4 x seq 1024) | exactly at the bound, zero headroom |
///
/// `MAX_ELEMENTS` was also sixteen times smaller than its siblings, which went
/// to 1<<30 on 2026-09-07 (`loss_ops`, `row_ops`); this one was not lifted with
/// them. A bound that a supported geometry meets EXACTLY is not a bound with
/// margin, it is a bound about to be crossed by the next size.
///
/// MEASURED ON THE RX 9070 XT before raising anything, because a raised
/// constant the kernel cannot address is a wrong answer rather than a bigger
/// capability:
///
/// * `max_workgroup_count` = `[4294967295, 65535, 65535]`. The backward
///   dispatches `output_len / WORKGROUP_SIZE` groups in **x**, so the largest
///   table these bounds now allow needs 16,777,216 groups against 4.29e9
///   available — three orders of margin.
/// * `max_storage_buffer_bytes` = 4,294,967,295, which at f32 is
///   **1,073,741,823 elements — one short of 1<<30**. So the effective ceiling
///   is `min(MAX_ELEMENTS, max_storage_buffer_bytes / 4)` and the DEVICE refuses
///   first, by one element, with its own named error from `Context::buffer`.
///   That layering is deliberate rather than sloppy: this constant is a POLICY
///   ceiling the plan can reason about, and the buffer limit is a PHYSICAL one
///   that belongs to the device and should refuse in the device's own words.
///   `MAX_ELEMENTS` is not raised past 1<<30 because past it every value is
///   unreachable on this hardware, and a bound nothing can reach is not a bound.
///
/// v4 (2026-09-08) RETRACTS the v3 note that used to stand here. It read:
/// "`MAX_VOCAB` stays at 1<<16. The registered vocabulary is 32,768 and nothing
/// measured needs more; raising a bound nothing exercises would be a claim
/// without a test behind it." The reasoning was sound and the number was wrong.
/// The registered training contract's vocabulary is **151,936**,
/// not 32,768, so 1<<16 refused the gather and the dense backward on every
/// microbatch pass of the campaign this crate exists to serve. Measured at
/// 4a95f3d3: `index_select.out` 5 and `embedding_dense_backward.out` 5 among 25
/// fallbacks, part of 24,914,370,580 download bytes over five passes.
///
/// `MAX_VOCAB` is now 1<<18 = 262,144, matching `row_ops::MAX_COLS` and
/// `loss_ops::MAX_COLS`, and covering Gemma's 256,000 as well as Qwen's 151,936.
///
/// THE EQUALITY `MAX_ELEMENTS == MAX_VOCAB * MAX_DIM` IS GONE, DELIBERATELY, and
/// the property it used to encode is gone with it. It said: no in-range
/// dimension triple can exceed the element bound, so the element check is
/// unreachable from in-range dims. At 1<<18 x 1<<14 the largest in-range triple
/// is 2^32, four times `MAX_ELEMENTS`, so the element check is now REACHABLE and
/// load-bearing rather than decorative -- `dimensions_and_both_element_bounds_precede_id_validation`
/// exhibits a concrete in-range triple that it refuses.
///
/// WHAT REPLACES IT is the property that actually keeps the kernel safe:
/// `MAX_VOCAB as u64 * MAX_DIM as u64 <= u32::MAX`, i.e. the largest in-range
/// product a caller can name must not WRAP the u32 the shader computes it in.
/// 2^18 x 2^14 = 2^32 exceeds u32::MAX by one, so the shader's guard cannot be a
/// product any more; `embedding_f32.comp` now divides instead of multiplying
/// (`p.vocab > MAX_ELEMENTS / p.dim`), which is exactly equivalent for positive
/// `dim` and cannot overflow at any bound. Had the guard stayed a product, a
/// raw-ABI call at vocab 262,144 x dim 16,384 would have wrapped to 0, passed
/// the guard, set `count` to 0 and returned from every invocation having written
/// NOTHING, while the C entry reported ACK_OK -- the same silent-no-op that
/// PR #1006 fixed in the matmul family and PR #1011 fixed in this one.
///
/// THE FOUR ARE STILL MUTUALLY CONSTRAINED, and the bound test still catches it.
/// Both of the plan's outputs must fit `MAX_ELEMENTS`: the backward's is
/// `vocab * dim` and **the gather's is `n * dim`**. `MAX_N` at 65,536 is sixteen
/// times the registered micro 4 x 1024 and eight times `xl` at micro 4 x 2048.
/// The registered geometry itself is comfortable: vocab 151,936 x dim 512 is
/// 77,791,232 elements, 7.2% of `MAX_ELEMENTS`.
pub const MAX_N: u32 = 1 << 16;
pub const MAX_VOCAB: u32 = 1 << 18;
pub const MAX_DIM: u32 = 1 << 14;
pub const MAX_ELEMENTS: usize = 1 << 30;
pub const WORKGROUP_SIZE: u32 = 64;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum EmbeddingOp {
    Gather = 0,
    Backward = 1,
}

impl TryFrom<u32> for EmbeddingOp {
    type Error = EmbeddingPlanError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Gather),
            1 => Ok(Self::Backward),
            _ => Err(EmbeddingPlanError::OperationOutOfRange),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EmbeddingPlanError {
    DimensionOutOfRange,
    ElementLimitExceeded,
    OperationOutOfRange,
    IndexShapeMismatch,
    IndexOutOfRange,
    ShapeMismatch,
    NonfiniteInput,
}

impl fmt::Display for EmbeddingPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::DimensionOutOfRange => "embedding-dimension-out-of-range",
            Self::ElementLimitExceeded => "embedding-element-limit-exceeded",
            Self::OperationOutOfRange => "embedding-operation-out-of-range",
            Self::IndexShapeMismatch => "embedding-index-shape-mismatch",
            Self::IndexOutOfRange => "embedding-index-out-of-range",
            Self::ShapeMismatch => "embedding-input-shape-mismatch",
            Self::NonfiniteInput => "embedding-nonfinite-input",
        })
    }
}

impl std::error::Error for EmbeddingPlanError {}

/// Immutable dimensions, owned token IDs, and stable CSR dispatch metadata.
/// Backward represents an unscaled sum into a dense vocabulary gradient: no
/// padding index, frequency scaling, or sparse-gradient semantics are implied.
/// Upload the plan's own index/offset bytes to keep the dispatched lookup data
/// consistent with the dimensions that were validated.
///
/// Callers cannot mutate validated IDs or CSR metadata:
/// ```compile_fail
/// use alelyon_compute_kit::embedding_ops::{EmbeddingOp, EmbeddingPlan};
/// let mut plan = EmbeddingPlan::new(1, 2, 3, EmbeddingOp::Gather, &[0]).unwrap();
/// plan.ids[0] = 9;
/// ```
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EmbeddingPlan {
    n: u32,
    vocab: u32,
    dim: u32,
    op: EmbeddingOp,
    ids: Vec<u32>,
    offsets: Vec<u32>,
    positions: Vec<u32>,
    primary_len: usize,
    output_len: usize,
    dispatch_groups: [u32; 3],
}

impl EmbeddingPlan {
    pub fn new(
        n: u32,
        vocab: u32,
        dim: u32,
        op: EmbeddingOp,
        ids: &[i64],
    ) -> Result<Self, EmbeddingPlanError> {
        if n == 0 || n > MAX_N || vocab == 0 || vocab > MAX_VOCAB || dim == 0 || dim > MAX_DIM {
            return Err(EmbeddingPlanError::DimensionOutOfRange);
        }
        let token_elements = checked_elements(n, dim)?;
        let table_elements = checked_elements(vocab, dim)?;
        if ids.len() != n as usize {
            return Err(EmbeddingPlanError::IndexShapeMismatch);
        }
        // Validate in the caller's i64 domain before any narrowing or lookup.
        if ids.iter().any(|&id| id < 0 || id >= i64::from(vocab)) {
            return Err(EmbeddingPlanError::IndexOutOfRange);
        }
        let ids: Vec<u32> = ids
            .iter()
            .map(|&id| u32::try_from(id).map_err(|_| EmbeddingPlanError::IndexOutOfRange))
            .collect::<Result<_, _>>()?;

        let mut offsets = vec![0u32; vocab as usize + 1];
        for &id in &ids {
            offsets[id as usize + 1] += 1;
        }
        for index in 1..offsets.len() {
            offsets[index] += offsets[index - 1];
        }
        let mut cursors = offsets[..vocab as usize].to_vec();
        let mut positions = vec![0u32; n as usize];
        // Appending in token order establishes a stable ordering within every
        // vocabulary bucket. All counts and positions are bounded by MAX_N.
        for (position, &id) in ids.iter().enumerate() {
            positions[cursors[id as usize] as usize] = position as u32;
            cursors[id as usize] += 1;
        }
        let (primary_len, output_len) = match op {
            EmbeddingOp::Gather => (table_elements, token_elements),
            EmbeddingOp::Backward => (token_elements, table_elements),
        };
        Ok(Self {
            n,
            vocab,
            dim,
            op,
            ids,
            offsets,
            positions,
            primary_len,
            output_len,
            dispatch_groups: [(output_len as u32).div_ceil(WORKGROUP_SIZE), 1, 1],
        })
    }

    pub fn validate_inputs(&self, primary: &[f32]) -> Result<(), EmbeddingPlanError> {
        if primary.len() != self.primary_len {
            return Err(EmbeddingPlanError::ShapeMismatch);
        }
        if primary.iter().any(|value| !value.is_finite()) {
            return Err(EmbeddingPlanError::NonfiniteInput);
        }
        Ok(())
    }

    pub fn n(&self) -> u32 {
        self.n
    }
    pub fn vocab(&self) -> u32 {
        self.vocab
    }
    pub fn dim(&self) -> u32 {
        self.dim
    }
    pub fn op(&self) -> EmbeddingOp {
        self.op
    }
    pub fn ids(&self) -> &[u32] {
        &self.ids
    }
    pub fn offsets(&self) -> &[u32] {
        &self.offsets
    }
    pub fn positions(&self) -> &[u32] {
        &self.positions
    }
    pub fn primary_len(&self) -> usize {
        self.primary_len
    }
    pub fn output_len(&self) -> usize {
        self.output_len
    }
    pub fn dispatch_groups(&self) -> [u32; 3] {
        self.dispatch_groups
    }

    /// Gather IDs or backward token positions, from this plan's validated data.
    pub fn index_bytes(&self) -> Vec<u8> {
        words_to_bytes(match self.op {
            EmbeddingOp::Gather => &self.ids,
            EmbeddingOp::Backward => &self.positions,
        })
    }

    /// One zero dummy word for gather, or the complete backward CSR offsets.
    pub fn offset_bytes(&self) -> Vec<u8> {
        words_to_bytes(match self.op {
            EmbeddingOp::Gather => &[0],
            EmbeddingOp::Backward => &self.offsets,
        })
    }

    pub fn push_constants(&self) -> [u8; 16] {
        let mut bytes = [0u8; 16];
        for (index, word) in [self.n, self.vocab, self.dim, self.op as u32]
            .iter()
            .enumerate()
        {
            bytes[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }
}

fn checked_elements(rows: u32, dim: u32) -> Result<usize, EmbeddingPlanError> {
    (rows as usize)
        .checked_mul(dim as usize)
        .filter(|&count| count <= MAX_ELEMENTS)
        .ok_or(EmbeddingPlanError::ElementLimitExceeded)
}

fn words_to_bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn operation_codes_refuse_unknown_values() {
        assert_eq!(EmbeddingOp::try_from(0), Ok(EmbeddingOp::Gather));
        assert_eq!(EmbeddingOp::try_from(1), Ok(EmbeddingOp::Backward));
        for value in [2, u32::MAX] {
            assert_eq!(
                EmbeddingOp::try_from(value),
                Err(EmbeddingPlanError::OperationOutOfRange)
            );
        }
    }

    #[test]
    fn dimensions_and_both_element_bounds_precede_id_validation() {
        for (n, vocab, dim) in [
            (0, 1, 1),
            (1, 0, 1),
            (1, 1, 0),
            (MAX_N + 1, 1, 1),
            (1, MAX_VOCAB + 1, 1),
            (1, 1, MAX_DIM + 1),
            (u32::MAX, u32::MAX, u32::MAX),
        ] {
            assert_eq!(
                EmbeddingPlan::new(n, vocab, dim, EmbeddingOp::Gather, &[]),
                Err(EmbeddingPlanError::DimensionOutOfRange)
            );
        }
        // RE-HOMED 2026-09-08 with the vocabulary raise. This assertion used to
        // read `assert_eq!(MAX_ELEMENTS, MAX_VOCAB as usize * MAX_DIM as usize)`
        // and the comment above it said the element check was unreachable from
        // in-range dimensions. Both died when MAX_VOCAB went to 1<<18: the
        // largest in-range pair is now 2^32, FOUR TIMES the element bound. The
        // assertion is not deleted and not relaxed -- it is replaced by the two
        // properties that are actually load-bearing at the new bounds.
        //
        // PROPERTY ONE: the element check is REACHABLE from an in-range triple,
        // so it is live validation rather than decoration. Here is one.
        let reachable = EmbeddingPlan::new(1, MAX_VOCAB, MAX_DIM, EmbeddingOp::Backward, &[0]);
        assert_eq!(reachable, Err(EmbeddingPlanError::ElementLimitExceeded));
        assert!(MAX_VOCAB as usize * MAX_DIM as usize > MAX_ELEMENTS);
        // PROPERTY TWO, and this is the one that keeps the SHADER honest: the
        // largest product an in-range triple can name must not wrap the u32 the
        // kernel computes it in. 2^18 * 2^14 is 2^32, which is one past
        // u32::MAX, so `embedding_f32.comp` cannot guard with a multiply and
        // divides instead. If either bound rises again and this fails, the GLSL
        // guard must be re-examined BEFORE the constant lands.
        assert!(MAX_VOCAB as u64 * MAX_DIM as u64 <= u32::MAX as u64 + 1);
        assert!(MAX_N as u64 * MAX_DIM as u64 <= u32::MAX as u64 + 1);
        assert_eq!(
            checked_elements(MAX_VOCAB, MAX_DIM + 1),
            Err(EmbeddingPlanError::ElementLimitExceeded)
        );
        // The largest table that still FITS the element bound, exercised from
        // both directions, replacing the old MAX_VOCAB x MAX_DIM corner.
        let widest_vocab = MAX_ELEMENTS as u32 / MAX_DIM; // 65,536 at 1<<30 / 1<<14
        let ids = vec![i64::from(widest_vocab) - 1; MAX_N as usize];
        let largest_table =
            EmbeddingPlan::new(MAX_N, widest_vocab, MAX_DIM, EmbeddingOp::Gather, &ids).unwrap();
        assert_eq!(largest_table.primary_len(), MAX_ELEMENTS);
        assert_eq!(
            largest_table.output_len(),
            MAX_N as usize * MAX_DIM as usize
        );
        let densest_gradient =
            EmbeddingPlan::new(MAX_N, widest_vocab, MAX_DIM, EmbeddingOp::Backward, &ids).unwrap();
        assert_eq!(densest_gradient.output_len(), MAX_ELEMENTS);
        assert_eq!(
            densest_gradient.dispatch_groups(),
            [(MAX_ELEMENTS as u32).div_ceil(WORKGROUP_SIZE), 1, 1]
        );
        // THE REGISTERED GEOMETRY, which is the whole reason for the raise and
        // which the 1<<16 bound refused: vocab 151,936 at d_model 512.
        assert!(
            EmbeddingPlan::new(1_024, 151_936, 512, EmbeddingOp::Backward, &[7; 1_024]).is_ok()
        );
        assert!(EmbeddingPlan::new(1_024, 151_936, 512, EmbeddingOp::Gather, &[7; 1_024]).is_ok());
        // The older 32,768 geometry keeps working -- the control, in unit form.
        assert!(EmbeddingPlan::new(4_096, 32_768, 768, EmbeddingOp::Backward, &[7; 4_096]).is_ok());
        assert!(EmbeddingPlan::new(1, 1, MAX_DIM, EmbeddingOp::Gather, &[0]).is_ok());
    }

    #[test]
    fn indices_require_exact_length_and_original_i64_range() {
        for ids in [&[][..], &[0, 1][..]] {
            assert_eq!(
                EmbeddingPlan::new(1, 3, 2, EmbeddingOp::Gather, ids),
                Err(EmbeddingPlanError::IndexShapeMismatch)
            );
        }
        for bad in [-1, 3, i64::MIN, i64::MAX, 1_i64 << 32, (1_i64 << 32) + 1] {
            assert_eq!(
                EmbeddingPlan::new(1, 3, 2, EmbeddingOp::Backward, &[bad]),
                Err(EmbeddingPlanError::IndexOutOfRange)
            );
        }
        assert_eq!(
            EmbeddingPlan::new(2, 3, 2, EmbeddingOp::Gather, &[0, 2])
                .unwrap()
                .ids(),
            &[0, 2]
        );
    }

    #[test]
    fn repeated_ids_form_exact_stable_csr_with_empty_buckets() {
        let plan = EmbeddingPlan::new(6, 5, 3, EmbeddingOp::Backward, &[3, 1, 3, 0, 1, 3]).unwrap();
        assert_eq!(plan.ids(), &[3, 1, 3, 0, 1, 3]);
        assert_eq!(plan.offsets(), &[0, 1, 3, 3, 6, 6]);
        assert_eq!(plan.positions(), &[3, 1, 4, 0, 2, 5]);
        for vocab_id in 0..plan.vocab() as usize {
            let bucket = &plan.positions()
                [plan.offsets()[vocab_id] as usize..plan.offsets()[vocab_id + 1] as usize];
            assert!(bucket.windows(2).all(|pair| pair[0] < pair[1]));
            assert!(bucket
                .iter()
                .all(|&position| plan.ids()[position as usize] == vocab_id as u32));
        }
        let mut all_positions = plan.positions().to_vec();
        all_positions.sort_unstable();
        assert_eq!(all_positions, [0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn one_repeated_id_keeps_every_position_in_original_order() {
        let plan = EmbeddingPlan::new(5, 4, 1, EmbeddingOp::Backward, &[2; 5]).unwrap();
        assert_eq!(plan.offsets(), &[0, 0, 0, 5, 5]);
        assert_eq!(plan.positions(), &[0, 1, 2, 3, 4]);
        let one = EmbeddingPlan::new(1, 1, 1, EmbeddingOp::Backward, &[0]).unwrap();
        assert_eq!(one.offsets(), &[0, 1]);
        assert_eq!(one.positions(), &[0]);
    }

    #[test]
    fn caller_mutation_cannot_change_owned_ids_or_csr() {
        let mut ids = [2, 0, 2, 1];
        let plan = EmbeddingPlan::new(4, 3, 1, EmbeddingOp::Gather, &ids).unwrap();
        ids.fill(0);
        assert_eq!(ids, [0; 4]);
        assert_eq!(plan.ids(), &[2, 0, 2, 1]);
        assert_eq!(plan.offsets(), &[0, 1, 2, 4]);
        assert_eq!(plan.positions(), &[1, 3, 0, 2]);
        let mut upload = plan.index_bytes();
        upload.fill(255);
        assert_ne!(upload, plan.index_bytes());
    }

    #[test]
    fn byte_images_and_metadata_match_each_operation() {
        for op in [EmbeddingOp::Gather, EmbeddingOp::Backward] {
            let plan = EmbeddingPlan::new(3, 2, 257, op, &[1, 0, 1]).unwrap();
            assert_eq!(
                (plan.n(), plan.vocab(), plan.dim(), plan.op()),
                (3, 2, 257, op)
            );
            assert_eq!(&plan.push_constants()[8..12], &[1, 1, 0, 0]);
            assert_eq!(
                plan.push_constants(),
                [3, 0, 0, 0, 2, 0, 0, 0, 1, 1, 0, 0, op as u8, 0, 0, 0]
            );
            match op {
                EmbeddingOp::Gather => {
                    assert_eq!((plan.primary_len(), plan.output_len()), (514, 771));
                    assert_eq!(plan.dispatch_groups(), [13, 1, 1]);
                    assert_eq!(plan.index_bytes(), [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]);
                    assert_eq!(plan.offset_bytes(), [0, 0, 0, 0]);
                }
                EmbeddingOp::Backward => {
                    assert_eq!((plan.primary_len(), plan.output_len()), (771, 514));
                    assert_eq!(plan.dispatch_groups(), [9, 1, 1]);
                    assert_eq!(plan.index_bytes(), [1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0]);
                    assert_eq!(plan.offset_bytes(), [0, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0]);
                }
            }
        }
    }

    #[test]
    fn grids_cover_exact_and_partial_workgroups() {
        for (len, groups) in [(1, 1), (63, 1), (64, 1), (65, 2)] {
            let gather =
                EmbeddingPlan::new(len, 1, 1, EmbeddingOp::Gather, &vec![0; len as usize]).unwrap();
            let backward = EmbeddingPlan::new(1, len, 1, EmbeddingOp::Backward, &[0]).unwrap();
            assert_eq!(gather.dispatch_groups(), [groups, 1, 1]);
            assert_eq!(backward.dispatch_groups(), [groups, 1, 1]);
        }
    }

    #[test]
    fn input_lengths_and_every_nonfinite_value_are_refused() {
        for op in [EmbeddingOp::Gather, EmbeddingOp::Backward] {
            let plan = EmbeddingPlan::new(2, 3, 2, op, &[0, 2]).unwrap();
            assert_eq!(plan.validate_inputs(&vec![0.0; plan.primary_len()]), Ok(()));
            for len in [plan.primary_len() - 1, plan.primary_len() + 1] {
                assert_eq!(
                    plan.validate_inputs(&vec![0.0; len]),
                    Err(EmbeddingPlanError::ShapeMismatch)
                );
            }
            for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let mut input = vec![0.0; plan.primary_len()];
                *input.last_mut().unwrap() = bad;
                assert_eq!(
                    plan.validate_inputs(&input),
                    Err(EmbeddingPlanError::NonfiniteInput)
                );
            }
        }
    }
}
