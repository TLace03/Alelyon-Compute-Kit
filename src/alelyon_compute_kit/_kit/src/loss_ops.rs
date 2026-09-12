//! Pure validation and dispatch planning for the experimental indexed-cross-entropy-f32/v1 ABI.
//! Bounds describe this implementation contract, not Vulkan or a selected GPU.
//! Four buffers are bound in order: primary, canonical targets, upstream, output.
//! Primary holds logits for forward/backward and raw per-row losses for reduction.
//! Forward always produces raw row losses; reduction and backward apply the
//! normalizer. Dummy upstream buffers remain required for operations that do not
//! consume them, so every dispatch has the same descriptor layout.

use std::fmt;

/// v2 (2026-09-07): rows lifted to the registered AKV geometry (4,096 rows at
/// micro-batch 4 x sequence 1,024) and elements to 2^30 so every addressed
/// buffer stays below 2^31 elements.
pub const MAX_ROWS: u32 = 1 << 20;
pub const MAX_COLS: u32 = 262_144;
pub const MAX_ELEMENTS: usize = 1 << 30;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum LossOp {
    Forward = 0,
    Backward = 1,
    Reduce = 2,
    /// Negative log-likelihood of log probabilities: `-primary[row, target]`.
    NllForward = 3,
    /// Its gradient: `-upstream * normalizer` at the target column, 0 elsewhere.
    NllBackward = 4,
}

impl TryFrom<u32> for LossOp {
    type Error = LossPlanError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Forward),
            1 => Ok(Self::Backward),
            2 => Ok(Self::Reduce),
            3 => Ok(Self::NllForward),
            4 => Ok(Self::NllBackward),
            _ => Err(LossPlanError::OperationOutOfRange),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum LossReduction {
    None = 0,
    Sum = 1,
    Mean = 2,
}

impl TryFrom<u32> for LossReduction {
    type Error = LossPlanError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::None),
            1 => Ok(Self::Sum),
            2 => Ok(Self::Mean),
            _ => Err(LossPlanError::ReductionOutOfRange),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LossPlanError {
    DimensionOutOfRange,
    ElementLimitExceeded,
    OperationOutOfRange,
    ReductionOutOfRange,
    InvalidReduction,
    TargetShapeMismatch,
    TargetOutOfRange,
    MeanHasNoValidTargets,
    ShapeMismatch,
    NonfiniteInput,
}

impl fmt::Display for LossPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::DimensionOutOfRange => "loss-dimension-out-of-range",
            Self::ElementLimitExceeded => "loss-element-limit-exceeded",
            Self::OperationOutOfRange => "loss-operation-out-of-range",
            Self::ReductionOutOfRange => "loss-reduction-out-of-range",
            Self::InvalidReduction => "loss-invalid-reduction-for-operation",
            Self::TargetShapeMismatch => "loss-target-shape-mismatch",
            Self::TargetOutOfRange => "loss-target-out-of-range",
            Self::MeanHasNoValidTargets => "loss-mean-has-no-valid-targets",
            Self::ShapeMismatch => "loss-input-shape-mismatch",
            Self::NonfiniteInput => "loss-nonfinite-input",
        })
    }
}

impl std::error::Error for LossPlanError {}

/// Immutable validated dimensions, owned target indices, and dispatch metadata.
/// Mean divides by the count of non-ignored targets. Adapters that need a
/// different denominator must use Sum and explicitly apply their outer scale.
///
/// A caller cannot mutate canonical targets after validation:
/// ```compile_fail
/// use alelyon_compute_kit::loss_ops::{LossOp, LossPlan, LossReduction};
/// let mut plan = LossPlan::new(1, 2, LossOp::Forward, LossReduction::None, -100, &[0]).unwrap();
/// plan.targets[0] = 9;
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct LossPlan {
    rows: u32,
    cols: u32,
    op: LossOp,
    reduction: LossReduction,
    ignore_index: i64,
    targets: Vec<i32>,
    valid_count: u32,
    elements: usize,
    primary_len: usize,
    upstream_len: usize,
    output_len: usize,
    normalizer: f32,
    dispatch_groups: [u32; 3],
}

impl LossPlan {
    pub fn new(
        rows: u32,
        cols: u32,
        op: LossOp,
        reduction: LossReduction,
        ignore_index: i64,
        targets: &[i64],
    ) -> Result<Self, LossPlanError> {
        if rows == 0 || rows > MAX_ROWS || cols == 0 || cols > MAX_COLS {
            return Err(LossPlanError::DimensionOutOfRange);
        }
        let elements = (rows as usize)
            .checked_mul(cols as usize)
            .filter(|&count| count <= MAX_ELEMENTS)
            .ok_or(LossPlanError::ElementLimitExceeded)?;
        if op == LossOp::Reduce && reduction == LossReduction::None {
            return Err(LossPlanError::InvalidReduction);
        }
        if targets.len() != rows as usize {
            return Err(LossPlanError::TargetShapeMismatch);
        }
        let mut canonical_targets = Vec::with_capacity(targets.len());
        let mut valid_count = 0;
        for &target in targets {
            // Compare in the original i64 domain, even when the ignored index
            // is an in-range class or lies outside the GPU index type.
            if target == ignore_index {
                canonical_targets.push(-1);
            } else {
                if target < 0 || target >= i64::from(cols) {
                    return Err(LossPlanError::TargetOutOfRange);
                }
                canonical_targets
                    .push(i32::try_from(target).map_err(|_| LossPlanError::TargetOutOfRange)?);
                valid_count += 1;
            }
        }
        if reduction == LossReduction::Mean && valid_count == 0 {
            return Err(LossPlanError::MeanHasNoValidTargets);
        }
        let normalizer = if reduction == LossReduction::Mean {
            1.0 / valid_count as f32
        } else {
            1.0
        };
        let primary_len = if op == LossOp::Reduce {
            rows as usize
        } else {
            elements
        };
        let output_len = match op {
            LossOp::Forward | LossOp::NllForward => rows as usize,
            LossOp::Backward | LossOp::NllBackward => elements,
            LossOp::Reduce => 1,
        };
        Ok(Self {
            rows,
            cols,
            op,
            reduction,
            ignore_index,
            targets: canonical_targets,
            valid_count,
            elements,
            primary_len,
            upstream_len: if reduction == LossReduction::None {
                rows as usize
            } else {
                1
            },
            output_len,
            normalizer,
            dispatch_groups: if op == LossOp::Reduce {
                [1, 1, 1]
            } else {
                [rows, 1, 1]
            },
        })
    }

    /// Validate every element, including ignored rows and unused dummy upstream
    /// entries. This prevents a successful plan from hiding nonfinite inputs.
    pub fn validate_inputs(&self, primary: &[f32], upstream: &[f32]) -> Result<(), LossPlanError> {
        if primary.len() != self.primary_len || upstream.len() != self.upstream_len {
            return Err(LossPlanError::ShapeMismatch);
        }
        if upstream.iter().any(|value| !value.is_finite()) {
            return Err(LossPlanError::NonfiniteInput);
        }
        // Log probabilities may be -inf (a masked class); +inf and NaN never.
        let log_probabilities = matches!(self.op, LossOp::NllForward | LossOp::NllBackward);
        if primary.iter().any(|value| {
            if log_probabilities {
                value.is_nan() || *value == f32::INFINITY
            } else {
                !value.is_finite()
            }
        }) {
            return Err(LossPlanError::NonfiniteInput);
        }
        Ok(())
    }

    pub fn rows(&self) -> u32 {
        self.rows
    }
    pub fn cols(&self) -> u32 {
        self.cols
    }
    pub fn op(&self) -> LossOp {
        self.op
    }
    pub fn reduction(&self) -> LossReduction {
        self.reduction
    }
    pub fn ignore_index(&self) -> i64 {
        self.ignore_index
    }
    pub fn targets(&self) -> &[i32] {
        &self.targets
    }
    pub fn targets_len(&self) -> usize {
        self.targets.len()
    }
    /// An owned upload image of the validated signed indices, in ABI byte order.
    pub fn target_bytes(&self) -> Vec<u8> {
        self.targets
            .iter()
            .flat_map(|target| target.to_le_bytes())
            .collect()
    }
    pub fn valid_count(&self) -> u32 {
        self.valid_count
    }
    pub fn elements(&self) -> usize {
        self.elements
    }
    pub fn primary_len(&self) -> usize {
        self.primary_len
    }
    pub fn upstream_len(&self) -> usize {
        self.upstream_len
    }
    pub fn output_len(&self) -> usize {
        self.output_len
    }
    pub fn normalizer(&self) -> f32 {
        self.normalizer
    }
    pub fn dispatch_groups(&self) -> [u32; 3] {
        self.dispatch_groups
    }

    pub fn push_constants(&self) -> [u8; 20] {
        let mut bytes = [0u8; 20];
        for (index, word) in [
            self.rows,
            self.cols,
            self.op as u32,
            self.reduction as u32,
            self.normalizer.to_bits(),
        ]
        .iter()
        .enumerate()
        {
            bytes[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_enums_are_checked() {
        for (wire, op) in [
            LossOp::Forward,
            LossOp::Backward,
            LossOp::Reduce,
            LossOp::NllForward,
            LossOp::NllBackward,
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(LossOp::try_from(wire as u32), Ok(op));
            assert_eq!(op as u32, wire as u32);
        }
        for (wire, reduction) in [LossReduction::None, LossReduction::Sum, LossReduction::Mean]
            .into_iter()
            .enumerate()
        {
            assert_eq!(LossReduction::try_from(wire as u32), Ok(reduction));
            assert_eq!(reduction as u32, wire as u32);
        }
        for bad in [5, u32::MAX] {
            assert_eq!(
                LossOp::try_from(bad),
                Err(LossPlanError::OperationOutOfRange)
            );
        }
        for bad in [3, u32::MAX] {
            assert_eq!(
                LossReduction::try_from(bad),
                Err(LossPlanError::ReductionOutOfRange)
            );
        }
    }

    #[test]
    fn bounds_precede_target_allocation_and_support_wide_vocabulary() {
        for (rows, cols) in [
            (0, 1),
            (1, 0),
            (MAX_ROWS + 1, 1),
            (1, MAX_COLS + 1),
            (u32::MAX, u32::MAX),
        ] {
            assert_eq!(
                LossPlan::new(rows, cols, LossOp::Forward, LossReduction::None, -100, &[]),
                Err(LossPlanError::DimensionOutOfRange)
            );
        }
        let too_many_rows = (MAX_ELEMENTS / MAX_COLS as usize + 1) as u32;
        assert_eq!(
            LossPlan::new(
                too_many_rows,
                MAX_COLS,
                LossOp::Forward,
                LossReduction::None,
                -100,
                &[]
            ),
            Err(LossPlanError::ElementLimitExceeded)
        );
        let widest_rows = (MAX_ELEMENTS / MAX_COLS as usize) as u32;
        let maximum = LossPlan::new(
            widest_rows,
            MAX_COLS,
            LossOp::Backward,
            LossReduction::Sum,
            -100,
            &vec![i64::from(MAX_COLS) - 1; widest_rows as usize],
        )
        .unwrap();
        assert_eq!(maximum.elements(), MAX_ELEMENTS);
        assert_eq!(maximum.output_len(), MAX_ELEMENTS);
        assert!(maximum.targets().iter().all(|&target| target == 262_143));
        assert_eq!(maximum.targets_len(), widest_rows as usize);
        assert!(LossPlan::new(
            MAX_ROWS,
            1_024,
            LossOp::Forward,
            LossReduction::None,
            -100,
            &[0; MAX_ROWS as usize],
        )
        .is_ok());
        assert!(LossPlan::new(
            1,
            151_936,
            LossOp::Forward,
            LossReduction::None,
            -100,
            &[151_935],
        )
        .is_ok());
    }

    #[test]
    fn target_shape_and_index_narrowing_fail_closed() {
        for targets in [&[][..], &[0, 1][..]] {
            assert_eq!(
                LossPlan::new(1, 3, LossOp::Forward, LossReduction::None, -100, targets),
                Err(LossPlanError::TargetShapeMismatch)
            );
        }
        for bad in [
            -2,
            -1,
            3,
            i64::from(i32::MAX),
            1_i64 << 32,
            i64::MIN,
            i64::MAX,
        ] {
            assert_eq!(
                LossPlan::new(1, 3, LossOp::Forward, LossReduction::None, -100, &[bad]),
                Err(LossPlanError::TargetOutOfRange)
            );
        }
    }

    #[test]
    fn ignore_is_compared_before_narrowing_even_for_an_in_range_class() {
        for ignored in [i64::MIN, i64::MAX, -1, -100, 0] {
            let mut source = [ignored, 1, ignored, 2];
            let plan = LossPlan::new(
                4,
                3,
                LossOp::Backward,
                LossReduction::Mean,
                ignored,
                &source,
            )
            .unwrap();
            source[0] = 2;
            assert_eq!(source[0], 2);
            assert_eq!(plan.targets(), &[-1, 1, -1, 2]);
            assert_eq!(plan.ignore_index(), ignored);
            assert_eq!(plan.valid_count(), 2);
            assert_eq!(plan.normalizer(), 0.5);
            assert_eq!(
                plan.target_bytes(),
                [255, 255, 255, 255, 1, 0, 0, 0, 255, 255, 255, 255, 2, 0, 0, 0]
            );
        }
    }

    #[test]
    fn all_ignored_mean_is_refused_but_none_and_sum_have_defined_zero_paths() {
        for op in [LossOp::Forward, LossOp::Backward, LossOp::Reduce] {
            assert_eq!(
                LossPlan::new(2, 3, op, LossReduction::Mean, -100, &[-100; 2]),
                Err(LossPlanError::MeanHasNoValidTargets)
            );
            for reduction in [LossReduction::None, LossReduction::Sum] {
                if op == LossOp::Reduce && reduction == LossReduction::None {
                    continue;
                }
                let plan = LossPlan::new(2, 3, op, reduction, -100, &[-100; 2]).unwrap();
                assert_eq!(plan.targets(), &[-1, -1]);
                assert_eq!(plan.valid_count(), 0);
                assert_eq!(plan.normalizer(), 1.0);
            }
        }
        assert_eq!(
            LossPlan::new(1, 3, LossOp::Reduce, LossReduction::None, -100, &[0]),
            Err(LossPlanError::InvalidReduction)
        );
    }

    #[test]
    fn each_operation_has_exact_lengths_grid_and_readonly_metadata() {
        for reduction in [LossReduction::None, LossReduction::Sum, LossReduction::Mean] {
            for op in [LossOp::Forward, LossOp::Backward, LossOp::Reduce] {
                if op == LossOp::Reduce && reduction == LossReduction::None {
                    continue;
                }
                let plan = LossPlan::new(3, 5, op, reduction, -100, &[0, -100, 4]).unwrap();
                assert_eq!((plan.rows(), plan.cols(), plan.elements()), (3, 5, 15));
                assert_eq!((plan.op(), plan.reduction()), (op, reduction));
                assert_eq!((plan.targets_len(), plan.valid_count()), (3, 2));
                assert_eq!(
                    plan.upstream_len(),
                    if reduction == LossReduction::None {
                        3
                    } else {
                        1
                    }
                );
                assert_eq!(
                    plan.primary_len(),
                    if op == LossOp::Reduce { 3 } else { 15 }
                );
                assert_eq!(
                    plan.output_len(),
                    match op {
                        LossOp::Forward | LossOp::NllForward => 3,
                        LossOp::Backward | LossOp::NllBackward => 15,
                        LossOp::Reduce => 1,
                    }
                );
                assert_eq!(
                    plan.dispatch_groups(),
                    if op == LossOp::Reduce {
                        [1, 1, 1]
                    } else {
                        [3, 1, 1]
                    }
                );
                assert_eq!(
                    plan.normalizer(),
                    if reduction == LossReduction::Mean {
                        0.5
                    } else {
                        1.0
                    }
                );
            }
        }
    }

    #[test]
    fn push_constants_are_exact_little_endian_words_for_every_operation() {
        for op in [LossOp::Forward, LossOp::Backward, LossOp::Reduce] {
            let plan =
                LossPlan::new(4, 257, op, LossReduction::Mean, -100, &[0, 1, -100, 256]).unwrap();
            let bytes = plan.push_constants();
            assert_eq!(bytes.len(), 20);
            let words: Vec<u32> = bytes
                .chunks_exact(4)
                .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
                .collect();
            assert_eq!(words, [4, 257, op as u32, 2, (1.0_f32 / 3.0).to_bits()]);
            assert_eq!(&bytes[4..8], &[1, 1, 0, 0]);
        }
    }

    #[test]
    fn validate_inputs_refuses_short_long_and_nonfinite_including_dummy_values() {
        for (op, reduction) in [
            (LossOp::Forward, LossReduction::None),
            (LossOp::Backward, LossReduction::Mean),
            (LossOp::Reduce, LossReduction::Sum),
        ] {
            let plan = LossPlan::new(2, 3, op, reduction, -100, &[0, -100]).unwrap();
            let primary = vec![0.0; plan.primary_len()];
            let upstream = vec![1.0; plan.upstream_len()];
            assert_eq!(plan.validate_inputs(&primary, &upstream), Ok(()));
            for len in [primary.len() - 1, primary.len() + 1] {
                assert_eq!(
                    plan.validate_inputs(&vec![0.0; len], &upstream),
                    Err(LossPlanError::ShapeMismatch)
                );
            }
            for len in [upstream.len() - 1, upstream.len() + 1] {
                assert_eq!(
                    plan.validate_inputs(&primary, &vec![1.0; len]),
                    Err(LossPlanError::ShapeMismatch)
                );
            }
            for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let mut bad_primary = primary.clone();
                *bad_primary.last_mut().unwrap() = invalid;
                assert_eq!(
                    plan.validate_inputs(&bad_primary, &upstream),
                    Err(LossPlanError::NonfiniteInput)
                );
                let mut bad_upstream = upstream.clone();
                *bad_upstream.last_mut().unwrap() = invalid;
                assert_eq!(
                    plan.validate_inputs(&primary, &bad_upstream),
                    Err(LossPlanError::NonfiniteInput)
                );
            }
        }
    }

    #[test]
    fn nll_plans_read_only_the_target_column_and_admit_masked_log_probabilities() {
        let forward = LossPlan::new(
            3,
            4,
            LossOp::NllForward,
            LossReduction::None,
            -100,
            &[1, -100, 3],
        )
        .unwrap();
        assert_eq!(
            (
                forward.primary_len(),
                forward.upstream_len(),
                forward.output_len()
            ),
            (12, 3, 3)
        );
        assert_eq!(forward.dispatch_groups(), [3, 1, 1]);
        assert_eq!(forward.targets(), &[1, -1, 3]);
        let backward = LossPlan::new(
            3,
            4,
            LossOp::NllBackward,
            LossReduction::Mean,
            -100,
            &[1, -100, 3],
        )
        .unwrap();
        assert_eq!((backward.upstream_len(), backward.output_len()), (1, 12));
        assert_eq!(backward.normalizer(), 0.5);
        let neg = f32::NEG_INFINITY;
        let masked = [
            0.0, neg, neg, neg, -1.0, -1.0, -1.0, -1.0, neg, neg, neg, 0.0,
        ];
        assert!(forward.validate_inputs(&masked, &[0.0; 3]).is_ok());
        let positive = [
            0.0,
            f32::INFINITY,
            0.0,
            0.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ];
        assert_eq!(
            forward.validate_inputs(&positive, &[0.0; 3]),
            Err(LossPlanError::NonfiniteInput)
        );
        let plain = LossPlan::new(
            3,
            4,
            LossOp::Forward,
            LossReduction::None,
            -100,
            &[1, -100, 3],
        )
        .unwrap();
        assert_eq!(
            plain.validate_inputs(&masked, &[0.0; 3]),
            Err(LossPlanError::NonfiniteInput)
        );
        assert!(LossPlan::new(
            4_096,
            32_768,
            LossOp::NllBackward,
            LossReduction::Sum,
            -100,
            &[0; 4_096]
        )
        .is_ok());
        assert_eq!(
            u32::from_le_bytes(backward.push_constants()[8..12].try_into().unwrap()),
            4
        );
    }
}
