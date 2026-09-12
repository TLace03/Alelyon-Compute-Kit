//! Pure validation and dispatch planning for the rowwise-f32/v2 ABI.
//! These bounds limit the current implementation contract; they are not limits
//! of Vulkan or of any selected GPU. v2 (2026-09-07) lifted them to the
//! registered AKV geometry (12,288 attention rows of 1,024; 1,024 vocabulary
//! rows of 32,768), added log-softmax forward and backward, and admits masked
//! (-inf) logits on the softmax forwards.
//!
//! v3 (2026-09-08) lifts `MAX_COLS` from 1<<16 to 1<<18. v2's "1,024
//! vocabulary rows of 32,768" was calibrated to an experiment, not to the
//! registered training contract, whose vocabulary is **151,936**.
//! `log_softmax` reduces over the vocabulary as COLUMNS, so 1<<16 refused the
//! registered logits tensor and its backward, and both fell back to the host:
//! measured at 25 fallbacks and 24,914,370,580 download bytes for five
//! microbatch passes, against 0 and 204,820 at vocab 32,768.
//!
//! 1<<18 = 262,144 is chosen to EQUAL `loss_ops::MAX_COLS`. That equality is
//! the point, not a coincidence: `log_softmax` and `nll_loss` consume the same
//! `[rows, vocab]` tensor in the same step, so a column bound that differs
//! between the two families means one refuses what the other accepts, which is
//! precisely how this defect survived. `tests/languages/test_compute_kit_row_bounds.py`
//! pins the equality. It also covers every shipped vocabulary this program can
//! name: Llama-3 128,256, DeepSeek-V3 129,280, Qwen 151,936, Gemma 256,000.
//!
//! The first TRUE ceiling above 1<<18 is not a policy number: `rowwise_f32.comp`
//! forms `float(p.cols)` for the RMSNorm normalisation, exact in f32 only to
//! 2^24. Nothing sits between 1<<18 and that. `MAX_ELEMENTS` is unchanged and
//! still governs: at the registered 1,024 rows it permits cols up to 1,048,576,
//! and the registered 1,024 x 151,936 is 155,582,464 elements, 14.5% of it.

use std::fmt;

pub const MAX_ROWS: u32 = 1 << 24;
/// Equal to `loss_ops::MAX_COLS` BY CONSTRUCTION -- see the module note. The
/// row family and the loss family reduce over the same vocabulary axis of the
/// same tensor, so these two bounds must not disagree.
pub const MAX_COLS: u32 = 1 << 18;
/// Every buffer the kernel addresses is at most twice this many elements, so
/// element offsets stay below 2^31.
pub const MAX_ELEMENTS: usize = 1 << 30;
pub const MIN_RMS_EPSILON: f32 = 1.0e-12;
pub const MAX_RMS_EPSILON: f32 = 1.0e12;
pub const SOFTMAX_PROBABILITY_SUM_TOLERANCE: f64 = 2.0e-5;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum RowOp {
    SoftmaxForward = 0,
    SoftmaxBackward = 1,
    RmsNormForward = 2,
    RmsNormBackward = 3,
    RmsNormGammaReduction = 4,
    LogSoftmaxForward = 5,
    LogSoftmaxBackward = 6,
}

impl TryFrom<u32> for RowOp {
    type Error = RowPlanError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::SoftmaxForward),
            1 => Ok(Self::SoftmaxBackward),
            2 => Ok(Self::RmsNormForward),
            3 => Ok(Self::RmsNormBackward),
            4 => Ok(Self::RmsNormGammaReduction),
            5 => Ok(Self::LogSoftmaxForward),
            6 => Ok(Self::LogSoftmaxBackward),
            _ => Err(RowPlanError::OperationOutOfRange),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RowPlanError {
    DimensionOutOfRange,
    ElementLimitExceeded,
    OperationOutOfRange,
    InvalidEpsilon,
    ShapeMismatch,
    NonfiniteInput,
    InvalidProbability,
    FullyMaskedRow,
    InvalidLogProbability,
}

impl fmt::Display for RowPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::DimensionOutOfRange => "rowwise-dimension-out-of-range",
            Self::ElementLimitExceeded => "rowwise-element-limit-exceeded",
            Self::OperationOutOfRange => "rowwise-operation-out-of-range",
            Self::InvalidEpsilon => "rowwise-invalid-epsilon",
            Self::ShapeMismatch => "rowwise-shape-mismatch",
            Self::NonfiniteInput => "rowwise-nonfinite-input",
            Self::InvalidProbability => "rowwise-invalid-softmax-probability",
            Self::FullyMaskedRow => "rowwise-fully-masked-row",
            Self::InvalidLogProbability => "rowwise-invalid-log-probability",
        })
    }
}

impl std::error::Error for RowPlanError {}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RowPlan {
    rows: u32,
    cols: u32,
    op: RowOp,
    epsilon: f32,
    elements: usize,
    primary_len: usize,
    upstream_len: usize,
    gamma_len: usize,
    output_len: usize,
    dispatch_groups: [u32; 3],
}

impl RowPlan {
    pub fn new(rows: u32, cols: u32, op: RowOp, epsilon: f32) -> Result<Self, RowPlanError> {
        if rows == 0 || rows > MAX_ROWS || cols == 0 || cols > MAX_COLS {
            return Err(RowPlanError::DimensionOutOfRange);
        }
        let elements = rows as usize * cols as usize;
        if elements > MAX_ELEMENTS {
            return Err(RowPlanError::ElementLimitExceeded);
        }
        if !epsilon.is_finite()
            || matches!(
                op,
                RowOp::RmsNormForward | RowOp::RmsNormBackward | RowOp::RmsNormGammaReduction
            ) && !(MIN_RMS_EPSILON..=MAX_RMS_EPSILON).contains(&epsilon)
        {
            return Err(RowPlanError::InvalidEpsilon);
        }
        let primary_len = if op == RowOp::RmsNormGammaReduction {
            2 * elements
        } else {
            elements
        };
        let output_len = match op {
            RowOp::RmsNormBackward => 2 * elements,
            RowOp::RmsNormGammaReduction => cols as usize,
            _ => elements,
        };
        let dispatch_groups = if op == RowOp::RmsNormGammaReduction {
            [cols, 1, 1]
        } else {
            [rows, 1, 1]
        };
        Ok(Self {
            rows,
            cols,
            op,
            epsilon,
            elements,
            primary_len,
            upstream_len: elements,
            gamma_len: cols as usize,
            output_len,
            dispatch_groups,
        })
    }

    pub fn validate_inputs(
        &self,
        primary: &[f32],
        upstream: &[f32],
        gamma: &[f32],
    ) -> Result<(), RowPlanError> {
        if primary.len() != self.primary_len
            || upstream.len() != self.upstream_len
            || gamma.len() != self.gamma_len
        {
            return Err(RowPlanError::ShapeMismatch);
        }
        if upstream.iter().chain(gamma).any(|value| !value.is_finite()) {
            return Err(RowPlanError::NonfiniteInput);
        }
        let masked_forward = matches!(self.op, RowOp::SoftmaxForward | RowOp::LogSoftmaxForward);
        let log_backward = self.op == RowOp::LogSoftmaxBackward;
        if primary.iter().any(|value| {
            if masked_forward || log_backward {
                value.is_nan() || *value == f32::INFINITY
            } else {
                !value.is_finite()
            }
        }) {
            return Err(RowPlanError::NonfiniteInput);
        }
        if masked_forward
            && primary
                .chunks_exact(self.cols as usize)
                .any(|row| row.iter().all(|value| *value == f32::NEG_INFINITY))
        {
            return Err(RowPlanError::FullyMaskedRow);
        }
        if log_backward {
            if primary.iter().any(|&value| value > 0.0) {
                return Err(RowPlanError::InvalidLogProbability);
            }
            for row in primary.chunks_exact(self.cols as usize) {
                let sum: f64 = row.iter().map(|&value| (value as f64).exp()).sum();
                if (sum - 1.0).abs() > SOFTMAX_PROBABILITY_SUM_TOLERANCE {
                    return Err(RowPlanError::InvalidLogProbability);
                }
            }
        }
        if self.op == RowOp::SoftmaxBackward {
            if primary
                .iter()
                .any(|&probability| !(0.0..=1.0).contains(&probability))
            {
                return Err(RowPlanError::InvalidProbability);
            }
            for row in primary.chunks_exact(self.cols as usize) {
                let sum: f64 = row.iter().map(|&probability| probability as f64).sum();
                if (sum - 1.0).abs() > SOFTMAX_PROBABILITY_SUM_TOLERANCE {
                    return Err(RowPlanError::InvalidProbability);
                }
            }
        }
        Ok(())
    }

    pub fn rows(&self) -> u32 {
        self.rows
    }
    pub fn cols(&self) -> u32 {
        self.cols
    }
    pub fn op(&self) -> RowOp {
        self.op
    }
    pub fn epsilon(&self) -> f32 {
        self.epsilon
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
    pub fn gamma_len(&self) -> usize {
        self.gamma_len
    }
    pub fn output_len(&self) -> usize {
        self.output_len
    }
    pub fn dispatch_groups(&self) -> [u32; 3] {
        self.dispatch_groups
    }

    pub fn push_constants(&self) -> [u8; 16] {
        let mut bytes = [0u8; 16];
        for (index, word) in [self.rows, self.cols, self.op as u32, self.epsilon.to_bits()]
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
    fn plans_exact_abi_lengths_constants_and_grids() {
        let backward = RowPlan::new(65, 63, RowOp::RmsNormBackward, 1.0e-5).unwrap();
        assert_eq!(backward.elements(), 4_095);
        assert_eq!(
            (
                backward.primary_len(),
                backward.upstream_len(),
                backward.gamma_len()
            ),
            (4_095, 4_095, 63)
        );
        assert_eq!(backward.output_len(), 8_190);
        assert_eq!(backward.dispatch_groups(), [65, 1, 1]);
        assert_eq!(
            (
                backward.rows(),
                backward.cols(),
                backward.op(),
                backward.epsilon()
            ),
            (65, 63, RowOp::RmsNormBackward, 1.0e-5)
        );
        assert_eq!(
            u32::from_le_bytes(backward.push_constants()[8..12].try_into().unwrap()),
            3
        );

        let reduction = RowPlan::new(65, 63, RowOp::RmsNormGammaReduction, 1.0e-5).unwrap();
        assert_eq!(reduction.primary_len(), 8_190);
        assert_eq!(reduction.output_len(), 63);
        assert_eq!(reduction.dispatch_groups(), [63, 1, 1]);
    }

    #[test]
    fn experimental_bounds_and_epsilon_fail_closed() {
        assert_eq!(
            RowPlan::new(0, 1, RowOp::SoftmaxForward, 0.0),
            Err(RowPlanError::DimensionOutOfRange)
        );
        assert_eq!(
            RowPlan::new(MAX_ROWS + 1, 1, RowOp::SoftmaxForward, 0.0),
            Err(RowPlanError::DimensionOutOfRange)
        );
        assert_eq!(
            RowPlan::new(1, MAX_COLS + 1, RowOp::SoftmaxForward, 0.0),
            Err(RowPlanError::DimensionOutOfRange)
        );
        assert_eq!(
            RowPlan::new(MAX_ROWS, 65, RowOp::SoftmaxForward, 0.0),
            Err(RowPlanError::ElementLimitExceeded)
        );
        // The registered AKV geometry is inside the contract.
        assert!(RowPlan::new(12_288, 1_024, RowOp::SoftmaxForward, 0.0).is_ok());
        assert!(RowPlan::new(1_024, 32_768, RowOp::LogSoftmaxForward, 0.0).is_ok());
        let widest = (MAX_ELEMENTS / MAX_ROWS as usize) as u32;
        let maximum =
            RowPlan::new(MAX_ROWS, widest, RowOp::RmsNormBackward, MIN_RMS_EPSILON).unwrap();
        assert_eq!(maximum.elements(), MAX_ELEMENTS);
        assert_eq!(maximum.output_len(), 2 * MAX_ELEMENTS);
        assert_eq!(
            RowPlan::new(1, 1, RowOp::RmsNormForward, -MIN_RMS_EPSILON),
            Err(RowPlanError::InvalidEpsilon)
        );
        assert_eq!(
            RowPlan::new(1, 1, RowOp::RmsNormForward, MIN_RMS_EPSILON / 2.0),
            Err(RowPlanError::InvalidEpsilon)
        );
        assert!(RowPlan::new(1, 1, RowOp::RmsNormForward, MIN_RMS_EPSILON).is_ok());
        assert!(RowPlan::new(1, 1, RowOp::RmsNormForward, MAX_RMS_EPSILON).is_ok());
        assert_eq!(
            RowPlan::new(1, 1, RowOp::RmsNormForward, f32::INFINITY),
            Err(RowPlanError::InvalidEpsilon)
        );
        assert_eq!(
            RowPlan::new(1, 1, RowOp::RmsNormForward, MAX_RMS_EPSILON * 2.0),
            Err(RowPlanError::InvalidEpsilon)
        );
        assert_eq!(
            RowPlan::new(1, 1, RowOp::RmsNormForward, 0.0),
            Err(RowPlanError::InvalidEpsilon)
        );
        assert_eq!(
            RowPlan::new(1, 1, RowOp::RmsNormForward, f32::NAN),
            Err(RowPlanError::InvalidEpsilon)
        );
        assert_eq!(RowOp::try_from(7), Err(RowPlanError::OperationOutOfRange));
        assert_eq!(RowOp::try_from(5), Ok(RowOp::LogSoftmaxForward));
        assert_eq!(RowOp::try_from(6), Ok(RowOp::LogSoftmaxBackward));
    }

    #[test]
    fn input_validation_checks_exact_shapes_and_finiteness() {
        let plan = RowPlan::new(2, 3, RowOp::SoftmaxForward, 0.0).unwrap();
        assert!(plan
            .validate_inputs(&[0.0; 6], &[0.0; 6], &[1.0; 3])
            .is_ok());
        assert_eq!(
            plan.validate_inputs(&[0.0; 5], &[0.0; 6], &[1.0; 3]),
            Err(RowPlanError::ShapeMismatch)
        );
        let mut primary = [0.0; 6];
        primary[2] = f32::INFINITY;
        assert_eq!(
            plan.validate_inputs(&primary, &[0.0; 6], &[1.0; 3]),
            Err(RowPlanError::NonfiniteInput)
        );
    }

    #[test]
    fn softmax_backward_rejects_probability_outside_unit_interval() {
        let plan = RowPlan::new(2, 3, RowOp::SoftmaxBackward, 0.0).unwrap();
        let upstream = [0.0; 6];
        let gamma = [1.0; 3];
        assert!(plan
            .validate_inputs(&[0.2, 0.3, 0.5, 0.1, 0.2, 0.7], &upstream, &gamma)
            .is_ok());
        assert_eq!(
            plan.validate_inputs(&[-0.1, 0.6, 0.5, 0.1, 0.2, 0.7], &upstream, &gamma)
                .unwrap_err()
                .to_string(),
            "rowwise-invalid-softmax-probability"
        );
    }

    #[test]
    fn softmax_backward_rejects_wrong_probability_row_sum() {
        let plan = RowPlan::new(2, 3, RowOp::SoftmaxBackward, 0.0).unwrap();
        let upstream = [0.0; 6];
        let gamma = [1.0; 3];
        assert_eq!(
            plan.validate_inputs(&[0.2, 0.3, 0.4, 0.1, 0.2, 0.7], &upstream, &gamma)
                .unwrap_err()
                .to_string(),
            "rowwise-invalid-softmax-probability"
        );
    }

    #[test]
    fn softmax_forwards_admit_masked_logits_but_not_a_fully_masked_row() {
        let upstream = [0.0; 6];
        let gamma = [1.0; 3];
        let neg = f32::NEG_INFINITY;
        for op in [RowOp::SoftmaxForward, RowOp::LogSoftmaxForward] {
            let plan = RowPlan::new(2, 3, op, 0.0).unwrap();
            let masked = [0.5, neg, neg, -1.0, 2.0, neg];
            assert!(plan.validate_inputs(&masked, &upstream, &gamma).is_ok());
            let fully = [0.5, 0.1, 0.2, neg, neg, neg];
            assert_eq!(
                plan.validate_inputs(&fully, &upstream, &gamma),
                Err(RowPlanError::FullyMaskedRow)
            );
            let positive = [0.5, f32::INFINITY, 0.2, 0.1, 0.2, 0.7];
            assert_eq!(
                plan.validate_inputs(&positive, &upstream, &gamma),
                Err(RowPlanError::NonfiniteInput)
            );
        }
        // RMSNorm keeps refusing every nonfinite value.
        let rms = RowPlan::new(2, 3, RowOp::RmsNormForward, 1.0e-5).unwrap();
        let masked = [0.5, neg, 0.2, 0.1, 0.2, 0.7];
        assert_eq!(
            rms.validate_inputs(&masked, &upstream, &gamma),
            Err(RowPlanError::NonfiniteInput)
        );
    }

    #[test]
    fn log_softmax_backward_requires_normalised_log_probabilities() {
        let plan = RowPlan::new(2, 3, RowOp::LogSoftmaxBackward, 0.0).unwrap();
        let upstream = [0.0; 6];
        let gamma = [1.0; 3];
        let third = (1.0f32 / 3.0).ln();
        let ok = [third; 6];
        assert!(plan.validate_inputs(&ok, &upstream, &gamma).is_ok());
        let neg = f32::NEG_INFINITY;
        let masked = [0.0, neg, neg, third, third, third];
        assert!(plan.validate_inputs(&masked, &upstream, &gamma).is_ok());
        let positive = [0.1, third, third, third, third, third];
        assert_eq!(
            plan.validate_inputs(&positive, &upstream, &gamma),
            Err(RowPlanError::InvalidLogProbability)
        );
        let unnormalised = [third, third, 0.0, third, third, third];
        assert_eq!(
            plan.validate_inputs(&unnormalised, &upstream, &gamma),
            Err(RowPlanError::InvalidLogProbability)
        );
        let lengths = RowPlan::new(2, 3, RowOp::LogSoftmaxForward, 0.0).unwrap();
        assert_eq!(
            (
                lengths.primary_len(),
                lengths.upstream_len(),
                lengths.output_len()
            ),
            (6, 6, 6)
        );
        assert_eq!(lengths.dispatch_groups(), [2, 1, 1]);
    }
}
