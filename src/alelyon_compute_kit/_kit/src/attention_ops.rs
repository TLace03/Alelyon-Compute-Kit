//! Pure validation and dispatch planning for the causal-softmax-f32/v1 ABI.
//! Each dense [batch, heads, sequence, sequence] row includes keys up to its
//! query index. These bounds describe this implementation, not Vulkan limits.

use std::fmt;

pub use crate::row_ops::SOFTMAX_PROBABILITY_SUM_TOLERANCE;

pub const MAX_SEQ: u32 = 1_024;
pub const MAX_ROWS: u32 = 4_096;
pub const MAX_ELEMENTS: usize = 1_048_576;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum CausalSoftmaxOp {
    Forward = 0,
    Backward = 1,
}

impl TryFrom<u32> for CausalSoftmaxOp {
    type Error = CausalSoftmaxPlanError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Forward),
            1 => Ok(Self::Backward),
            _ => Err(CausalSoftmaxPlanError::OperationOutOfRange),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CausalSoftmaxPlanError {
    DimensionOutOfRange,
    DimensionOverflow,
    ElementLimitExceeded,
    OperationOutOfRange,
    ShapeMismatch,
    NonfiniteInput,
    InvalidProbability,
    InvalidMaskedProbability,
}

impl fmt::Display for CausalSoftmaxPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::DimensionOutOfRange => "causal-softmax-dimension-out-of-range",
            Self::DimensionOverflow => "causal-softmax-dimension-overflow",
            Self::ElementLimitExceeded => "causal-softmax-element-limit-exceeded",
            Self::OperationOutOfRange => "causal-softmax-operation-out-of-range",
            Self::ShapeMismatch => "causal-softmax-input-shape-mismatch",
            Self::NonfiniteInput => "causal-softmax-nonfinite-input",
            Self::InvalidProbability => "causal-softmax-invalid-probability",
            Self::InvalidMaskedProbability => "causal-softmax-invalid-masked-probability",
        })
    }
}

impl std::error::Error for CausalSoftmaxPlanError {}

/// Immutable dimensions and derived lengths for dense causal softmax.
///
/// Bind four distinct buffers: primary logits (forward) or saved probabilities
/// (backward), upstream, auxiliary, and output. Forward upstream and auxiliary
/// are finite one-element dummy buffers; backward upstream has one value per
/// probability and auxiliary remains a finite one-element dummy. This pure
/// plan validates slices, but cannot establish GPU buffer identity, lifetime,
/// or whether a later upload still contains the values validated here.
///
/// Validated geometry cannot be changed by a caller:
/// ```compile_fail
/// use alelyon_compute_kit::attention_ops::{CausalSoftmaxOp, CausalSoftmaxPlan};
/// let mut plan = CausalSoftmaxPlan::new(1, 1, 2, CausalSoftmaxOp::Forward).unwrap();
/// plan.seq = 0;
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CausalSoftmaxPlan {
    batch: u32,
    heads: u32,
    seq: u32,
    op: CausalSoftmaxOp,
    rows: u32,
    elements: usize,
}

impl CausalSoftmaxPlan {
    pub fn new(
        batch: u32,
        heads: u32,
        seq: u32,
        op: CausalSoftmaxOp,
    ) -> Result<Self, CausalSoftmaxPlanError> {
        if batch == 0 || heads == 0 || seq == 0 || seq > MAX_SEQ {
            return Err(CausalSoftmaxPlanError::DimensionOutOfRange);
        }
        // Even u64 can overflow for valid u32 factors. Check the complete
        // geometry before narrowing, allocating, or deriving dispatch values.
        let rows = u64::from(batch)
            .checked_mul(u64::from(heads))
            .and_then(|planes| planes.checked_mul(u64::from(seq)))
            .ok_or(CausalSoftmaxPlanError::DimensionOverflow)?;
        let elements = rows
            .checked_mul(u64::from(seq))
            .ok_or(CausalSoftmaxPlanError::DimensionOverflow)?;
        if rows > u64::from(MAX_ROWS) {
            return Err(CausalSoftmaxPlanError::DimensionOutOfRange);
        }
        if elements > MAX_ELEMENTS as u64 {
            return Err(CausalSoftmaxPlanError::ElementLimitExceeded);
        }
        Ok(Self {
            batch,
            heads,
            seq,
            op,
            rows: u32::try_from(rows).map_err(|_| CausalSoftmaxPlanError::DimensionOverflow)?,
            elements: usize::try_from(elements)
                .map_err(|_| CausalSoftmaxPlanError::DimensionOverflow)?,
        })
    }

    pub fn validate_inputs(
        &self,
        primary: &[f32],
        upstream: &[f32],
        auxiliary: &[f32],
    ) -> Result<(), CausalSoftmaxPlanError> {
        if primary.len() != self.primary_len()
            || upstream.len() != self.upstream_len()
            || auxiliary.len() != self.auxiliary_len()
        {
            return Err(CausalSoftmaxPlanError::ShapeMismatch);
        }
        // Masked values and unused dummy bindings remain part of the finite
        // input contract; a mask does not license NaN or infinity inputs.
        if primary
            .iter()
            .chain(upstream)
            .chain(auxiliary)
            .any(|value| !value.is_finite())
        {
            return Err(CausalSoftmaxPlanError::NonfiniteInput);
        }
        if self.op == CausalSoftmaxOp::Backward {
            if primary.iter().any(|value| !(0.0..=1.0).contains(value)) {
                return Err(CausalSoftmaxPlanError::InvalidProbability);
            }
            let seq = self.seq as usize;
            for (row_index, row) in primary.chunks_exact(seq).enumerate() {
                let prefix_len = row_index % seq + 1;
                if row[prefix_len..].iter().any(|value| value.to_bits() != 0) {
                    return Err(CausalSoftmaxPlanError::InvalidMaskedProbability);
                }
                let sum: f64 = row[..prefix_len]
                    .iter()
                    .map(|&probability| f64::from(probability))
                    .sum();
                if (sum - 1.0).abs() > SOFTMAX_PROBABILITY_SUM_TOLERANCE {
                    return Err(CausalSoftmaxPlanError::InvalidProbability);
                }
            }
        }
        Ok(())
    }

    pub fn batch(&self) -> u32 {
        self.batch
    }

    pub fn heads(&self) -> u32 {
        self.heads
    }

    pub fn seq(&self) -> u32 {
        self.seq
    }

    pub fn op(&self) -> CausalSoftmaxOp {
        self.op
    }

    pub fn rows(&self) -> u32 {
        self.rows
    }

    pub fn elements(&self) -> usize {
        self.elements
    }

    pub fn primary_len(&self) -> usize {
        self.elements
    }

    pub fn upstream_len(&self) -> usize {
        match self.op {
            CausalSoftmaxOp::Forward => 1,
            CausalSoftmaxOp::Backward => self.elements,
        }
    }

    pub fn auxiliary_len(&self) -> usize {
        1
    }

    pub fn output_len(&self) -> usize {
        self.elements
    }

    pub fn dispatch_groups(&self) -> [u32; 3] {
        [self.rows, 1, 1]
    }

    pub fn push_constants(&self) -> [u8; 16] {
        let mut bytes = [0u8; 16];
        for (index, word) in [self.batch, self.heads, self.seq, self.op as u32]
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

    fn probabilities(plan: &CausalSoftmaxPlan) -> Vec<f32> {
        let mut values = vec![0.0; plan.elements()];
        for (row_index, row) in values.chunks_exact_mut(plan.seq() as usize).enumerate() {
            let prefix_len = row_index % plan.seq() as usize + 1;
            row[..prefix_len].fill(1.0 / prefix_len as f32);
        }
        values
    }

    #[test]
    fn plans_exact_lengths_geometry_and_little_endian_push_bytes() {
        for op in [CausalSoftmaxOp::Forward, CausalSoftmaxOp::Backward] {
            let plan = CausalSoftmaxPlan::new(2, 3, 7, op).unwrap();
            assert_eq!(
                (plan.batch(), plan.heads(), plan.seq(), plan.op()),
                (2, 3, 7, op)
            );
            assert_eq!((plan.rows(), plan.elements()), (42, 294));
            assert_eq!((plan.primary_len(), plan.output_len()), (294, 294));
            assert_eq!(plan.auxiliary_len(), 1);
            assert_eq!(
                plan.upstream_len(),
                if op == CausalSoftmaxOp::Forward {
                    1
                } else {
                    294
                }
            );
            assert_eq!(plan.dispatch_groups(), [42, 1, 1]);
            assert_eq!(
                plan.push_constants(),
                [2, 0, 0, 0, 3, 0, 0, 0, 7, 0, 0, 0, op as u8, 0, 0, 0]
            );
        }
        let plan = CausalSoftmaxPlan::new(1, 1, 1, CausalSoftmaxOp::Backward).unwrap();
        assert_eq!(plan.dispatch_groups(), [1, 1, 1]);
        assert_eq!(
            (plan.primary_len(), plan.upstream_len(), plan.output_len()),
            (1, 1, 1)
        );
        assert!(plan.validate_inputs(&[1.0], &[-17.0], &[2.0]).is_ok());
    }

    #[test]
    fn operation_tags_are_closed_and_errors_are_named() {
        assert_eq!(CausalSoftmaxOp::try_from(0), Ok(CausalSoftmaxOp::Forward));
        assert_eq!(CausalSoftmaxOp::try_from(1), Ok(CausalSoftmaxOp::Backward));
        for tag in [2, 3, u32::MAX] {
            assert_eq!(
                CausalSoftmaxOp::try_from(tag),
                Err(CausalSoftmaxPlanError::OperationOutOfRange)
            );
        }
        assert_eq!(
            CausalSoftmaxPlanError::DimensionOverflow.to_string(),
            "causal-softmax-dimension-overflow"
        );
        assert_eq!(
            CausalSoftmaxPlanError::InvalidMaskedProbability.to_string(),
            "causal-softmax-invalid-masked-probability"
        );
    }

    #[test]
    fn dimensions_caps_and_full_width_products_fail_closed() {
        let op = CausalSoftmaxOp::Forward;
        for (batch, heads, seq) in [
            (0, 1, 1),
            (1, 0, 1),
            (1, 1, 0),
            (1, 1, MAX_SEQ + 1),
            (MAX_ROWS + 1, 1, 1),
            (1, MAX_ROWS + 1, 1),
            (u32::MAX, 1, 1),
        ] {
            assert_eq!(
                CausalSoftmaxPlan::new(batch, heads, seq, op),
                Err(CausalSoftmaxPlanError::DimensionOutOfRange)
            );
        }
        assert_eq!(
            CausalSoftmaxPlan::new(1, 2, MAX_SEQ, op),
            Err(CausalSoftmaxPlanError::ElementLimitExceeded)
        );
        // Row multiplication overflows u64 in the first case; the second has
        // representable rows but overflows only when multiplied by seq again.
        for (batch, heads, seq) in [(u32::MAX, u32::MAX, 2), (2_000_000_000, 2_000_000_000, 3)] {
            assert_eq!(
                CausalSoftmaxPlan::new(batch, heads, seq, op),
                Err(CausalSoftmaxPlanError::DimensionOverflow)
            );
        }
        for (batch, heads, seq) in [
            (MAX_ROWS, 1, 1),
            (1, MAX_ROWS, 1),
            (4, 4, 256),
            (1, 1, MAX_SEQ),
        ] {
            let plan = CausalSoftmaxPlan::new(batch, heads, seq, op).unwrap();
            assert!(plan.rows() <= MAX_ROWS && plan.elements() <= MAX_ELEMENTS);
        }
        let maximum = CausalSoftmaxPlan::new(4, 4, 256, op).unwrap();
        assert_eq!(
            (maximum.rows(), maximum.elements()),
            (MAX_ROWS, MAX_ELEMENTS)
        );
        assert_eq!(
            CausalSoftmaxPlan::new(1, 1, MAX_SEQ, op)
                .unwrap()
                .elements(),
            MAX_ELEMENTS
        );
    }

    #[test]
    fn every_binding_requires_its_exact_length_before_value_checks() {
        for op in [CausalSoftmaxOp::Forward, CausalSoftmaxOp::Backward] {
            let plan = CausalSoftmaxPlan::new(1, 1, 3, op).unwrap();
            let primary = if op == CausalSoftmaxOp::Backward {
                probabilities(&plan)
            } else {
                vec![0.0; plan.primary_len()]
            };
            let upstream = vec![0.0; plan.upstream_len()];
            assert!(plan.validate_inputs(&primary, &upstream, &[0.0]).is_ok());
            for binding in 0..3 {
                let bindings = [primary.clone(), upstream.clone(), vec![0.0]];
                for length in [bindings[binding].len() - 1, bindings[binding].len() + 1] {
                    let mut malformed = bindings.clone();
                    malformed[binding] = vec![f32::NAN; length];
                    assert_eq!(
                        plan.validate_inputs(&malformed[0], &malformed[1], &malformed[2]),
                        Err(CausalSoftmaxPlanError::ShapeMismatch)
                    );
                }
            }
        }
    }

    #[test]
    fn nonfinite_values_are_refused_including_masked_and_dummy_inputs() {
        for op in [CausalSoftmaxOp::Forward, CausalSoftmaxOp::Backward] {
            let plan = CausalSoftmaxPlan::new(1, 1, 2, op).unwrap();
            for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                for binding in 0..3 {
                    let primary = if op == CausalSoftmaxOp::Backward {
                        probabilities(&plan)
                    } else {
                        vec![0.0; plan.primary_len()]
                    };
                    let mut values = [primary, vec![0.0; plan.upstream_len()], vec![0.0]];
                    // primary[1] and backward upstream[1] are masked entries.
                    let index = usize::from(values[binding].len() > 1);
                    values[binding][index] = bad;
                    assert_eq!(
                        plan.validate_inputs(&values[0], &values[1], &values[2]),
                        Err(CausalSoftmaxPlanError::NonfiniteInput)
                    );
                }
            }
        }
    }

    #[test]
    fn backward_causal_geometry_restarts_for_every_batch_and_head() {
        let plan = CausalSoftmaxPlan::new(2, 3, 4, CausalSoftmaxOp::Backward).unwrap();
        let values = probabilities(&plan);
        let upstream = vec![2.0; plan.upstream_len()];
        assert!(plan.validate_inputs(&values, &upstream, &[0.0]).is_ok());
        for row_index in 0..plan.rows() as usize {
            let query = row_index % plan.seq() as usize;
            let base = row_index * plan.seq() as usize;
            if query + 1 < plan.seq() as usize {
                let mut wrong_mask = values.clone();
                wrong_mask[base + query + 1] = f32::from_bits(1);
                assert_eq!(
                    plan.validate_inputs(&wrong_mask, &upstream, &[0.0]),
                    Err(CausalSoftmaxPlanError::InvalidMaskedProbability)
                );
            }
            let mut wrong_mass = values.clone();
            wrong_mass[base] *= 0.5;
            assert_eq!(
                plan.validate_inputs(&wrong_mass, &upstream, &[0.0]),
                Err(CausalSoftmaxPlanError::InvalidProbability)
            );
        }
    }

    #[test]
    fn backward_mask_requires_positive_zero_bits() {
        let plan = CausalSoftmaxPlan::new(1, 1, 2, CausalSoftmaxOp::Backward).unwrap();
        for bad in [-0.0, f32::from_bits(1), 0.25] {
            assert_eq!(
                plan.validate_inputs(&[1.0, bad, 0.5, 0.5], &[0.0; 4], &[0.0]),
                Err(CausalSoftmaxPlanError::InvalidMaskedProbability)
            );
        }
        // Only the masked suffix requires a particular zero sign. A valid
        // prefix may contain signed zero under the declared [0, 1] contract.
        assert!(plan
            .validate_inputs(&[1.0, 0.0, -0.0, 1.0], &[0.0; 4], &[0.0])
            .is_ok());
    }

    #[test]
    fn backward_probability_range_and_prefix_mass_are_enforced() {
        let plan = CausalSoftmaxPlan::new(1, 1, 2, CausalSoftmaxOp::Backward).unwrap();
        for last in [
            [-f32::EPSILON, 1.0],
            [1.0 + f32::EPSILON, 0.0],
            [0.0, 0.0],
            [0.5, 0.5 - 3.0e-5],
        ] {
            assert_eq!(
                plan.validate_inputs(&[1.0, 0.0, last[0], last[1]], &[0.0; 4], &[0.0]),
                Err(CausalSoftmaxPlanError::InvalidProbability)
            );
        }
        for delta in [-1.0e-5, 0.0, 1.0e-5] {
            assert!(plan
                .validate_inputs(&[1.0, 0.0, 0.5, 0.5 + delta], &[0.0; 4], &[0.0])
                .is_ok());
        }
        let largest = CausalSoftmaxPlan::new(1, 1, MAX_SEQ, CausalSoftmaxOp::Backward).unwrap();
        assert!(largest
            .validate_inputs(
                &probabilities(&largest),
                &vec![0.0; largest.upstream_len()],
                &[0.0]
            )
            .is_ok());
    }

    #[test]
    fn forward_accepts_finite_extremes_without_probability_or_mask_semantics() {
        let plan = CausalSoftmaxPlan::new(1, 1, 2, CausalSoftmaxOp::Forward).unwrap();
        assert!(plan
            .validate_inputs(
                &[f32::MAX, -f32::MAX, -0.0, f32::from_bits(1)],
                &[f32::MIN],
                &[1.0]
            )
            .is_ok());
    }
}
