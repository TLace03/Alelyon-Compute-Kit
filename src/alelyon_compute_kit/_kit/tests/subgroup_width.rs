//! The subgroup width a pipeline may require, on a real device. A width
//! outside the device's [minSubgroupSize, maxSubgroupSize] is refused by name
//! before any object exists, because a driver need not reject it: NVIDIA
//! 580.173.02 on an L4 (range [32, 32]) accepted a required width of 64, and
//! the 64-lane BF16 matmul then returned wrong products with no error
//! (2026-09-22). Without a device every test prints UNMEASURED and returns
//! (`ACK_REQUIRE_DEVICE=1` turns that into a failure).

use alelyon_compute_kit::{AckError, Context};

/// A 256-invocation kernel with no subgroup operations: a multiple of every
/// width tried here, so REQUIRE_FULL_SUBGROUPS is satisfiable at each of them.
const CAST: &[u8] = include_bytes!("../kernels/cast_f32_bf16.spv");
const WIDTHS: [u32; 9] = [1, 2, 4, 8, 16, 32, 64, 128, 256];

fn open_or_unmeasured(what: &str) -> Option<Context> {
    match Context::open() {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            if std::env::var_os("ACK_REQUIRE_DEVICE").is_some() {
                panic!("{what}: ACK_REQUIRE_DEVICE is set and no device opened: {e}");
            }
            eprintln!("UNMEASURED here ({what}): {e}");
            None
        }
    }
}

#[test]
fn the_admitted_widths_are_the_powers_of_two_in_the_device_range() {
    let Some(ctx) = open_or_unmeasured("admitted widths") else {
        return;
    };
    let (min, max) = ctx.subgroup_size_range;
    for size in WIDTHS {
        let expected = ctx.subgroup_size_control
            && ctx.required_subgroup_size_compute
            && min <= size
            && size <= max;
        assert_eq!(
            ctx.can_require_subgroup_size(size),
            expected,
            "width {size} on a device with range [{min}, {max}]"
        );
    }
    for size in [0u32, 3, 48, 96] {
        assert!(
            !ctx.can_require_subgroup_size(size),
            "{size} is not a power of two and is never a subgroup width"
        );
    }
    if ctx.subgroup_size_control {
        assert!(
            min.is_power_of_two() && max.is_power_of_two() && min <= max,
            "a device with subgroup size control reports a power-of-two range, got [{min}, {max}]"
        );
        let default = ctx.report.subgroup_size;
        assert!(
            min <= default && default <= max,
            "the default width {default} lies outside the reported range [{min}, {max}]"
        );
    }
}

#[test]
fn a_width_the_device_cannot_guarantee_is_refused_before_any_object_exists() {
    let Some(ctx) = open_or_unmeasured("refused width") else {
        return;
    };
    let before = ctx.live_objects();
    let mut refused = 0;
    for size in WIDTHS {
        if ctx.can_require_subgroup_size(size) {
            continue;
        }
        match ctx.kernel_with_subgroup(CAST, 2, 4, Some(size)) {
            Err(AckError::Unsupported(_)) => refused += 1,
            Err(e) => panic!("width {size}: refused for the wrong reason: {e}"),
            Ok(k) => {
                ctx.destroy_kernel(k);
                panic!(
                    "width {size} is outside [{}, {}] and was built anyway",
                    ctx.subgroup_size_range.0, ctx.subgroup_size_range.1
                );
            }
        }
    }
    assert!(
        refused > 0,
        "no width in {WIDTHS:?} lies outside the device's range, so no refusal was exercised"
    );
    assert_eq!(
        ctx.live_objects(),
        before,
        "a refused width created Vulkan objects"
    );
}

#[test]
fn every_admitted_width_builds_a_pipeline() {
    let Some(ctx) = open_or_unmeasured("admitted width builds") else {
        return;
    };
    let before = ctx.live_objects();
    let mut built = 0;
    for size in WIDTHS {
        if !ctx.can_require_subgroup_size(size) {
            continue;
        }
        let k = ctx
            .kernel_with_subgroup(CAST, 2, 4, Some(size))
            .unwrap_or_else(|e| panic!("width {size} is admitted and failed to build: {e}"));
        ctx.destroy_kernel(k);
        built += 1;
    }
    if ctx.subgroup_size_control && ctx.required_subgroup_size_compute {
        assert!(
            built > 0,
            "the device admits required widths and none built"
        );
    }
    assert_eq!(
        ctx.live_objects(),
        before,
        "a built and destroyed pipeline leaked"
    );
}
