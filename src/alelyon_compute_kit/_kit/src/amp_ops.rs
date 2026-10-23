//! amp-unscale-f32/v1 (2026-09-29): GradScaler's non-finite check and unscale
//! of one dense float32 gradient on the kit, `kernels/amp_unscale_f32.comp`,
//! through the OPTIONAL entry `ack_amp_unscale` (resolved by name, in no
//! pointer table; `ACK_ABI_VERSION` does not move).
//!
//! torch's `_amp_foreach_non_finite_check_and_unscale_(grads, found_inf,
//! inv_scale)` stores 1.0 into `found_inf` when any element of any gradient is
//! an infinity or a NaN (it never clears it) and multiplies every element by
//! `inv_scale` in place, unless `inv_scale` is exactly 1.0, which keeps every
//! bit. The torch adapter issues one call of this entry per gradient. Before it
//! (MEASURED 2026-09-29) the fp16-autocast toy stage-E step ran the whole
//! operator on the host: every gradient downloaded and uploaded, twice a step.
//!
//! The request is four element offsets and a count over three buffers: the
//! gradient's dense view `[data_offset, data_offset + count)`, the scale's one
//! element and `found`'s one element. This module re-derives every word against
//! the buffers' real capacities and refuses by name before any dispatch.

use std::fmt;

/// The kernel's push block: offset, count, scale_offset, found_offset (u32 each).
pub const PUSH_BYTES: usize = 16;
/// The caller's request block, `AckAmpUnscaleRequest`: four u64 words.
pub const REQUEST_BYTES: usize = 32;
/// Invocations per workgroup (`local_size_x`).
pub const GROUP: u32 = 256;
/// The x grid is capped at the Vulkan-guaranteed 65,535 groups; a longer
/// gradient is covered by the kernel's grid-stride loop.
pub const MAX_GROUPS: u32 = 65_535;
/// The largest gradient one call takes: 2^31 elements, so the kernel's
/// grid-stride index cannot wrap.
pub const MAX_COUNT: u64 = 1 << 31;

/// A request as the caller passes it (`AckAmpUnscaleRequest`, 32 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AmpUnscaleRequest {
    /// The gradient's first element in its buffer.
    pub data_offset: u64,
    /// The gradient's elements (dense).
    pub count: u64,
    /// `inv_scale`'s element in its buffer.
    pub scale_offset: u64,
    /// `found_inf`'s element in its buffer.
    pub found_offset: u64,
}

impl AmpUnscaleRequest {
    /// Decode a request block (little-endian u64 words).
    pub fn from_bytes(block: &[u8]) -> Option<Self> {
        if block.len() != REQUEST_BYTES {
            return None;
        }
        let word = |i: usize| u64::from_le_bytes(block[8 * i..8 * i + 8].try_into().unwrap());
        Some(Self {
            data_offset: word(0),
            count: word(1),
            scale_offset: word(2),
            found_offset: word(3),
        })
    }

    /// Encode a request block.
    pub fn to_bytes(self) -> [u8; REQUEST_BYTES] {
        let mut block = [0u8; REQUEST_BYTES];
        for (i, w) in [
            self.data_offset,
            self.count,
            self.scale_offset,
            self.found_offset,
        ]
        .iter()
        .enumerate()
        {
            block[8 * i..8 * i + 8].copy_from_slice(&w.to_le_bytes());
        }
        block
    }
}

/// Why a request is refused, each with the name a refusal histogram carries.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AmpPlanError {
    /// More than `MAX_COUNT` elements.
    CountAboveLimit,
    /// An offset or the view's end does not fit the kernel's 32-bit words.
    WordAboveU32,
    /// The gradient's view runs past its buffer.
    DataTooSmall,
    /// The scale's element lies past its buffer.
    ScaleTooSmall,
    /// `found`'s element lies past its buffer.
    FoundTooSmall,
}

impl AmpPlanError {
    /// The refusal's name.
    pub fn name(self) -> &'static str {
        match self {
            AmpPlanError::CountAboveLimit => "amp-count-above-limit",
            AmpPlanError::WordAboveU32 => "amp-word-above-u32",
            AmpPlanError::DataTooSmall => "amp-data-too-small",
            AmpPlanError::ScaleTooSmall => "amp-scale-too-small",
            AmpPlanError::FoundTooSmall => "amp-found-too-small",
        }
    }
}

impl fmt::Display for AmpPlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A validated call: the kernel's words and its grid.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AmpPlan {
    offset: u32,
    count: u32,
    scale_offset: u32,
    found_offset: u32,
}

impl AmpPlan {
    /// Validate `r` against the three buffers' capacities in float32 elements.
    pub fn new(
        r: &AmpUnscaleRequest,
        data_elements: u64,
        scale_elements: u64,
        found_elements: u64,
    ) -> Result<Self, AmpPlanError> {
        if r.count > MAX_COUNT {
            return Err(AmpPlanError::CountAboveLimit);
        }
        let end = r
            .data_offset
            .checked_add(r.count)
            .ok_or(AmpPlanError::WordAboveU32)?;
        if end > u64::from(u32::MAX)
            || r.scale_offset > u64::from(u32::MAX)
            || r.found_offset > u64::from(u32::MAX)
        {
            return Err(AmpPlanError::WordAboveU32);
        }
        if end > data_elements {
            return Err(AmpPlanError::DataTooSmall);
        }
        if r.scale_offset >= scale_elements {
            return Err(AmpPlanError::ScaleTooSmall);
        }
        if r.found_offset >= found_elements {
            return Err(AmpPlanError::FoundTooSmall);
        }
        Ok(Self {
            offset: r.data_offset as u32,
            count: r.count as u32,
            scale_offset: r.scale_offset as u32,
            found_offset: r.found_offset as u32,
        })
    }

    /// Whether the call has nothing to do (an empty gradient): no dispatch.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// The push block.
    pub fn push_constants(&self) -> [u8; PUSH_BYTES] {
        let mut block = [0u8; PUSH_BYTES];
        for (i, w) in [
            self.offset,
            self.count,
            self.scale_offset,
            self.found_offset,
        ]
        .iter()
        .enumerate()
        {
            block[4 * i..4 * i + 4].copy_from_slice(&w.to_le_bytes());
        }
        block
    }

    /// The grid: one invocation per element up to `MAX_GROUPS` groups.
    pub fn dispatch_groups(&self) -> [u32; 3] {
        let groups = self.count.div_ceil(GROUP).clamp(1, MAX_GROUPS);
        [groups, 1, 1]
    }
}

// ---- amp-update-scale-f32/v1 (2026-09-29) ------------------------------------
//
// GradScaler's scale update, torch's `_amp_update_scale_(scale, growth_tracker,
// found_inf, growth_factor, backoff_factor, growth_interval)`, in ONE invocation
// of `kernels/amp_update_scale_f32.comp` through the OPTIONAL entry
// `ack_amp_update_scale` (resolved by name; `ACK_ABI_VERSION` does not move).
// Until it (MEASURED 2026-09-29) this was the one host fallback left in the
// fp16-autocast stage-E step, at the toy size and at the registered geometry.
//
// The CPU multiplies the float scale by a DOUBLE factor and rounds the product
// to float once. A factor that is a float32 exactly (any finite float32, or an
// infinity) makes that double product exact, so its rounding is the correctly
// rounded binary32 product the kernel computes in integer instructions; any
// other factor (a NaN, or a double between two float32 values) is refused by
// name here and left to the host.

/// The update kernel's push block: three offsets, two factor words, the interval.
pub const UPDATE_PUSH_BYTES: usize = 24;
/// The caller's request block, `AckAmpUpdateRequest`: six 8-byte words.
pub const UPDATE_REQUEST_BYTES: usize = 48;

/// An update request as the caller passes it (`AckAmpUpdateRequest`, 48 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AmpUpdateRequest {
    /// The scale's element in its buffer (float32).
    pub scale_offset: u64,
    /// The growth tracker's element in its buffer (int32).
    pub tracker_offset: u64,
    /// `found_inf`'s element in its buffer (float32).
    pub found_offset: u64,
    /// `growth_factor`, as torch passes it (a double).
    pub growth_factor: f64,
    /// `backoff_factor`, as torch passes it (a double).
    pub backoff_factor: f64,
    /// `growth_interval`, as torch passes it (an int64).
    pub growth_interval: i64,
}

impl AmpUpdateRequest {
    /// Decode a request block (little-endian 8-byte words).
    pub fn from_bytes(block: &[u8]) -> Option<Self> {
        if block.len() != UPDATE_REQUEST_BYTES {
            return None;
        }
        let word = |i: usize| u64::from_le_bytes(block[8 * i..8 * i + 8].try_into().unwrap());
        Some(Self {
            scale_offset: word(0),
            tracker_offset: word(1),
            found_offset: word(2),
            growth_factor: f64::from_bits(word(3)),
            backoff_factor: f64::from_bits(word(4)),
            growth_interval: word(5) as i64,
        })
    }

    /// Encode a request block.
    pub fn to_bytes(self) -> [u8; UPDATE_REQUEST_BYTES] {
        let mut block = [0u8; UPDATE_REQUEST_BYTES];
        for (i, w) in [
            self.scale_offset,
            self.tracker_offset,
            self.found_offset,
            self.growth_factor.to_bits(),
            self.backoff_factor.to_bits(),
            self.growth_interval as u64,
        ]
        .iter()
        .enumerate()
        {
            block[8 * i..8 * i + 8].copy_from_slice(&w.to_le_bytes());
        }
        block
    }
}

/// Why an update request is refused, each with the name a refusal histogram carries.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AmpUpdateError {
    /// An offset does not fit the kernel's 32-bit words.
    WordAboveU32,
    /// The scale's element lies past its buffer.
    ScaleTooSmall,
    /// The tracker's element lies past its buffer.
    TrackerTooSmall,
    /// `found`'s element lies past its buffer.
    FoundTooSmall,
    /// A factor that is not a float32 exactly (a NaN, or a double between two
    /// float32 values): the CPU's double product would round twice here.
    FactorNotF32,
    /// A growth interval outside int32: the CPU compares it with an int32
    /// tracker, which the kernel keeps as one.
    IntervalAboveI32,
}

impl AmpUpdateError {
    /// The refusal's name.
    pub fn name(self) -> &'static str {
        match self {
            AmpUpdateError::WordAboveU32 => "amp-word-above-u32",
            AmpUpdateError::ScaleTooSmall => "amp-scale-too-small",
            AmpUpdateError::TrackerTooSmall => "amp-tracker-too-small",
            AmpUpdateError::FoundTooSmall => "amp-found-too-small",
            AmpUpdateError::FactorNotF32 => "amp-update-factor-not-f32",
            AmpUpdateError::IntervalAboveI32 => "amp-update-interval-above-i32",
        }
    }
}

impl fmt::Display for AmpUpdateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// The float32 a double factor IS, exactly, or `None`: any finite float32 and
/// both infinities qualify; a NaN and every double between two float32 values
/// do not.
pub fn exact_f32(x: f64) -> Option<f32> {
    let f = x as f32;
    (f64::from(f) == x).then_some(f)
}

/// A validated update: the kernel's words.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AmpUpdatePlan {
    scale_offset: u32,
    tracker_offset: u32,
    found_offset: u32,
    growth_bits: u32,
    backoff_bits: u32,
    interval: i32,
}

impl AmpUpdatePlan {
    /// Validate `r` against the three buffers' capacities in 4-byte elements.
    pub fn new(
        r: &AmpUpdateRequest,
        scale_elements: u64,
        tracker_elements: u64,
        found_elements: u64,
    ) -> Result<Self, AmpUpdateError> {
        let growth = exact_f32(r.growth_factor).ok_or(AmpUpdateError::FactorNotF32)?;
        let backoff = exact_f32(r.backoff_factor).ok_or(AmpUpdateError::FactorNotF32)?;
        let interval =
            i32::try_from(r.growth_interval).map_err(|_| AmpUpdateError::IntervalAboveI32)?;
        let word = |w: u64| u32::try_from(w).map_err(|_| AmpUpdateError::WordAboveU32);
        let (scale_offset, tracker_offset, found_offset) = (
            word(r.scale_offset)?,
            word(r.tracker_offset)?,
            word(r.found_offset)?,
        );
        if r.scale_offset >= scale_elements {
            return Err(AmpUpdateError::ScaleTooSmall);
        }
        if r.tracker_offset >= tracker_elements {
            return Err(AmpUpdateError::TrackerTooSmall);
        }
        if r.found_offset >= found_elements {
            return Err(AmpUpdateError::FoundTooSmall);
        }
        Ok(Self {
            scale_offset,
            tracker_offset,
            found_offset,
            growth_bits: growth.to_bits(),
            backoff_bits: backoff.to_bits(),
            interval,
        })
    }

    /// The push block.
    pub fn push_constants(&self) -> [u8; UPDATE_PUSH_BYTES] {
        let mut block = [0u8; UPDATE_PUSH_BYTES];
        for (i, w) in [
            self.scale_offset,
            self.tracker_offset,
            self.found_offset,
            self.growth_bits,
            self.backoff_bits,
            self.interval as u32,
        ]
        .iter()
        .enumerate()
        {
            block[4 * i..4 * i + 4].copy_from_slice(&w.to_le_bytes());
        }
        block
    }

    /// One invocation.
    pub fn dispatch_groups(&self) -> [u32; 3] {
        [1, 1, 1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(
        data_offset: u64,
        count: u64,
        scale_offset: u64,
        found_offset: u64,
    ) -> AmpUnscaleRequest {
        AmpUnscaleRequest {
            data_offset,
            count,
            scale_offset,
            found_offset,
        }
    }

    #[test]
    fn a_request_round_trips_and_a_block_of_another_size_is_no_request() {
        let r = req(7, 1000, 3, 9);
        assert_eq!(AmpUnscaleRequest::from_bytes(&r.to_bytes()), Some(r));
        assert_eq!(AmpUnscaleRequest::from_bytes(&[0u8; 31]), None);
        assert_eq!(AmpUnscaleRequest::from_bytes(&[0u8; 33]), None);
        assert_eq!(std::mem::size_of::<AmpUnscaleRequest>(), REQUEST_BYTES);
    }

    #[test]
    fn the_plan_addresses_the_view_and_refuses_by_name() {
        let plan = AmpPlan::new(&req(5, 100, 0, 2), 105, 1, 3).unwrap();
        assert_eq!(plan.push_constants()[..8], [5, 0, 0, 0, 100, 0, 0, 0]);
        assert_eq!(plan.push_constants()[8..], [0, 0, 0, 0, 2, 0, 0, 0]);
        assert_eq!(plan.dispatch_groups(), [1, 1, 1]);
        assert_eq!(
            AmpPlan::new(&req(5, 101, 0, 0), 105, 1, 1),
            Err(AmpPlanError::DataTooSmall)
        );
        assert_eq!(
            AmpPlan::new(&req(0, 1, 1, 0), 1, 1, 1),
            Err(AmpPlanError::ScaleTooSmall)
        );
        assert_eq!(
            AmpPlan::new(&req(0, 1, 0, 1), 1, 1, 1),
            Err(AmpPlanError::FoundTooSmall)
        );
        assert_eq!(
            AmpPlan::new(&req(0, MAX_COUNT + 1, 0, 0), u64::MAX, 1, 1),
            Err(AmpPlanError::CountAboveLimit)
        );
        assert_eq!(
            AmpPlan::new(&req(u64::from(u32::MAX), 1, 0, 0), u64::MAX, 1, 1),
            Err(AmpPlanError::WordAboveU32)
        );
        assert_eq!(
            AmpPlan::new(&req(u64::MAX, 1, 0, 0), u64::MAX, 1, 1),
            Err(AmpPlanError::WordAboveU32)
        );
        assert_eq!(
            AmpPlan::new(&req(0, 1, u64::from(u32::MAX) + 1, 0), 1, u64::MAX, 1),
            Err(AmpPlanError::WordAboveU32)
        );
        for e in [
            AmpPlanError::CountAboveLimit,
            AmpPlanError::WordAboveU32,
            AmpPlanError::DataTooSmall,
            AmpPlanError::ScaleTooSmall,
            AmpPlanError::FoundTooSmall,
        ] {
            assert!(e.name().starts_with("amp-"), "{e}");
        }
    }

    #[test]
    fn the_grid_covers_the_view_and_is_capped() {
        let groups = |count: u64| {
            AmpPlan::new(&req(0, count, 0, 0), count, 1, 1)
                .unwrap()
                .dispatch_groups()[0]
        };
        assert!(AmpPlan::new(&req(0, 0, 0, 0), 0, 1, 1).unwrap().is_empty());
        assert_eq!(groups(1), 1);
        assert_eq!(groups(256), 1);
        assert_eq!(groups(257), 2);
        assert_eq!(groups(256 * 65_535), 65_535);
        assert_eq!(
            groups(256 * 65_535 + 1),
            65_535,
            "the rest is the grid-stride loop's"
        );
        assert_eq!(groups(MAX_COUNT), 65_535);
    }

    fn update(
        scale_offset: u64,
        tracker_offset: u64,
        found_offset: u64,
        growth_factor: f64,
        backoff_factor: f64,
        growth_interval: i64,
    ) -> AmpUpdateRequest {
        AmpUpdateRequest {
            scale_offset,
            tracker_offset,
            found_offset,
            growth_factor,
            backoff_factor,
            growth_interval,
        }
    }

    #[test]
    fn an_update_request_round_trips_and_a_block_of_another_size_is_no_request() {
        let r = update(1, 2, 3, 2.0, 0.5, 2000);
        assert_eq!(AmpUpdateRequest::from_bytes(&r.to_bytes()), Some(r));
        let odd = update(0, 0, 0, -1.5e-7, f64::INFINITY, -7);
        assert_eq!(AmpUpdateRequest::from_bytes(&odd.to_bytes()), Some(odd));
        assert_eq!(AmpUpdateRequest::from_bytes(&[0u8; 47]), None);
        assert_eq!(AmpUpdateRequest::from_bytes(&[0u8; 49]), None);
        assert_eq!(
            std::mem::size_of::<AmpUpdateRequest>(),
            UPDATE_REQUEST_BYTES
        );
    }

    #[test]
    fn a_factor_is_admitted_only_when_it_is_a_float32_exactly() {
        assert_eq!(exact_f32(2.0), Some(2.0));
        assert_eq!(exact_f32(0.5), Some(0.5));
        assert_eq!(exact_f32(f64::from(0.37f32)), Some(0.37f32));
        assert_eq!(exact_f32(0.37), None, "0.37 lies between two float32");
        assert_eq!(exact_f32(f64::INFINITY), Some(f32::INFINITY));
        assert_eq!(exact_f32(f64::NEG_INFINITY), Some(f32::NEG_INFINITY));
        assert_eq!(exact_f32(f64::NAN), None);
        assert_eq!(exact_f32(1e300), None, "past float32's range");
        assert_eq!(
            exact_f32(f64::from(f32::from_bits(1))),
            Some(f32::from_bits(1)),
            "a float32 subnormal"
        );
        assert_eq!(exact_f32(1e-46), None, "below float32's smallest subnormal");
        assert_eq!(exact_f32(-0.0).map(f32::to_bits), Some(0x8000_0000));
    }

    #[test]
    fn the_update_plan_carries_the_words_and_refuses_by_name() {
        let plan = AmpUpdatePlan::new(&update(4, 5, 6, 2.0, 0.5, 2000), 5, 6, 7).unwrap();
        let push = plan.push_constants();
        let word = |i: usize| u32::from_le_bytes(push[4 * i..4 * i + 4].try_into().unwrap());
        assert_eq!(
            [word(0), word(1), word(2), word(3), word(4), word(5)],
            [4, 5, 6, 0x4000_0000, 0x3F00_0000, 2000]
        );
        assert_eq!(plan.dispatch_groups(), [1, 1, 1]);
        let negative = AmpUpdatePlan::new(&update(0, 0, 0, 2.0, 0.5, -3), 1, 1, 1).unwrap();
        assert_eq!(
            negative.push_constants()[20..],
            (-3i32).to_le_bytes(),
            "the interval is a signed word"
        );
        let refused = |r: AmpUpdateRequest, e: AmpUpdateError| {
            assert_eq!(AmpUpdatePlan::new(&r, 1, 1, 1), Err(e), "{r:?}");
        };
        refused(update(1, 0, 0, 2.0, 0.5, 1), AmpUpdateError::ScaleTooSmall);
        refused(
            update(0, 1, 0, 2.0, 0.5, 1),
            AmpUpdateError::TrackerTooSmall,
        );
        refused(update(0, 0, 1, 2.0, 0.5, 1), AmpUpdateError::FoundTooSmall);
        refused(update(0, 0, 0, 0.37, 0.5, 1), AmpUpdateError::FactorNotF32);
        refused(
            update(0, 0, 0, 2.0, f64::NAN, 1),
            AmpUpdateError::FactorNotF32,
        );
        refused(
            update(0, 0, 0, 2.0, 0.5, i64::from(i32::MAX) + 1),
            AmpUpdateError::IntervalAboveI32,
        );
        refused(
            update(u64::from(u32::MAX) + 1, 0, 0, 2.0, 0.5, 1),
            AmpUpdateError::WordAboveU32,
        );
        for e in [
            AmpUpdateError::WordAboveU32,
            AmpUpdateError::ScaleTooSmall,
            AmpUpdateError::TrackerTooSmall,
            AmpUpdateError::FoundTooSmall,
            AmpUpdateError::FactorNotF32,
            AmpUpdateError::IntervalAboveI32,
        ] {
            assert!(e.name().starts_with("amp-"), "{e}");
        }
    }

    /// The text of `uint f32_mul(...) { ... }` in a kernel source, through the
    /// closing brace at the start of a line.
    fn f32_mul_of(source: &str) -> &str {
        let start = source
            .find("uint f32_mul(uint s, uint v) {")
            .expect("f32_mul is defined");
        let end = start + source[start..].find("\n}\n").expect("f32_mul ends") + 3;
        &source[start..end]
    }

    #[test]
    fn the_update_kernel_multiplies_with_the_unscale_kernels_routine_character_for_character() {
        let unscale = include_str!("../kernels/amp_unscale_f32.comp");
        let update = include_str!("../kernels/amp_update_scale_f32.comp");
        assert_eq!(f32_mul_of(unscale), f32_mul_of(update));
        assert!(
            f32_mul_of(update).len() > 2000,
            "the whole routine is compared"
        );
    }
}
