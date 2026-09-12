//! ACK-L0-05: nothing a `Context` method created before a later step failed
//! may stay allocated. Every Vulkan object a method creates goes through one
//! counting wrapper and every destroy through its twin, so `live_objects()`
//! is the number of objects the context has not destroyed, counted by the
//! calls themselves. With the `fault-injection` feature a creation step can
//! be made to fail AFTER the real call succeeded (the wrapper destroys the
//! real object and reports the injected result), which drives every cleanup
//! path on a healthy card. Needs a device; without one every test prints
//! UNMEASURED and returns, unless ACK_REQUIRE_DEVICE is set (the print is
//! visible only with --nocapture: a green run without the variable is not
//! device evidence).
//!
//! EVERY CONTEXT HERE RUNS WITH THE BUFFER POOL OFF, and the switch is in
//! `open_or_unmeasured` so no test can forget it. Two separate reasons, both
//! of which would otherwise make these tests say something they do not mean:
//!
//! 1. THE COUNT. A pooled pair is a LIVE object and `live_objects()` counts it
//!    (correctly: the pool is holding it, not leaking it), so "back to the
//!    baseline" after a free would read base+2 with the pool on. That is not a
//!    leak and the assertion is not wrong; it is measuring a different
//!    quantity. The pooled counterpart --  that `live_objects()` equals the
//!    baseline plus exactly `pooled_objects()`, and returns to the baseline
//!    when the pool is drained -- is
//!    `the_pool_holds_its_objects_and_the_leak_law_still_closes` in
//!    tests/buffer_pool.rs.
//!
//! 2. THE FAULTS, which matters more. `take_create_fault` fires inside the
//!    creation wrappers, and a request served from the pool reaches none of
//!    them. `a_failure_at_any_step_of_a_buffer_leaves_no_object_behind` drives
//!    four injected creation failures through `ctx.buffer(4096)` and requires
//!    each to FAIL; a pool hit would satisfy the request instead, leave the
//!    fault armed, and turn a cleanup-path test into a test of nothing. The
//!    ordering these tests depend on is preserved in the code -- the zero-size
//!    and over-limit refusals stay above the pool lookup in `Context::buffer`
//!    -- but the hit itself cannot be allowed here.
#![cfg(feature = "fault-injection")]

use alelyon_compute_kit::{AckError, Context, CreateStep, Fault};
use ash::vk;

const CAST_F32_BF16: &[u8] = include_bytes!("../kernels/cast_f32_bf16.spv");

fn open_or_unmeasured(what: &str) -> Option<Context> {
    match Context::open() {
        Ok(ctx) => {
            // see the module doc: the pool is off for every test in this file
            ctx.set_buffer_pool_enabled(false);
            Some(ctx)
        }
        Err(e) => {
            if std::env::var_os("ACK_REQUIRE_DEVICE").is_some() {
                panic!("{what}: ACK_REQUIRE_DEVICE is set and no device opened: {e}");
            }
            eprintln!("UNMEASURED here ({what}): {e}");
            None
        }
    }
}

fn injected(ctx: &Context, step: CreateStep) -> vk::Result {
    let r = vk::Result::ERROR_OUT_OF_DEVICE_MEMORY;
    ctx.inject_fault(Fault::CreateFails(step, r));
    r
}

#[test]
fn every_object_a_method_creates_is_destroyed_again_and_the_count_says_so() {
    let Some(ctx) = open_or_unmeasured("live objects") else {
        return;
    };
    assert_eq!(
        ctx.live_objects(),
        0,
        "a fresh context counts nothing: open's own objects are outside the count"
    );
    // an observation, not a falsifier: after open() this field is true by construction
    // on any card that opened; the refusal path needs a device without shaderInt16
    eprintln!(
        "observed: shader_int16 {} timestamp_valid_bits {}",
        ctx.report.shader_int16, ctx.timestamp_valid_bits
    );
    let base = ctx.live_objects();
    let buf = ctx.buffer(4096).expect("buffer");
    assert_eq!(ctx.live_objects(), base + 2, "a buffer and its memory");
    ctx.upload(&buf, &vec![1u8; 4096]).expect("upload");
    let back = ctx.download(&buf).expect("download");
    assert!(back.iter().all(|&x| x == 1));
    assert_eq!(
        ctx.live_objects(),
        base + 2,
        "staging pairs, fences and command buffers are transient"
    );
    let kernel = ctx.kernel(CAST_F32_BF16, 2, 4).expect("kernel");
    assert_eq!(
        ctx.live_objects(),
        base + 6,
        "a kernel is four objects; its shader module is transient"
    );
    let out = ctx.buffer(2048).expect("bf16 output");
    ctx.bind(&kernel, &[&buf, &out]).expect("bind");
    let count: u32 = 1024;
    ctx.dispatch_timed(&kernel, &count.to_le_bytes(), [4, 1, 1], 1)
        .expect("dispatch");
    assert_eq!(
        ctx.live_objects(),
        base + 8,
        "the timestamp query pool is transient"
    );
    ctx.destroy_kernel(kernel);
    assert_eq!(ctx.live_objects(), base + 4);
    ctx.destroy_buffer(out);
    ctx.destroy_buffer(buf);
    assert_eq!(ctx.live_objects(), base, "back to the baseline");
    assert_eq!(ctx.leaked_on_poison(), 0, "a healthy context leaks nothing");
}

#[test]
fn a_failure_at_any_step_of_a_buffer_leaves_no_object_behind() {
    let Some(ctx) = open_or_unmeasured("buffer steps") else {
        return;
    };
    let base = ctx.live_objects();
    for step in [
        CreateStep::Buffer,
        CreateStep::MemoryType,
        CreateStep::Memory,
        CreateStep::BindMemory,
    ] {
        let r = injected(&ctx, step);
        let err = match ctx.buffer(4096) {
            Ok(b) => {
                ctx.destroy_buffer(b);
                panic!("{step:?}: the buffer must fail");
            }
            Err(e) => e,
        };
        assert!(
            matches!(err, AckError::Vulkan(e) if e == r),
            "{step:?}: {err}"
        );
        assert_eq!(
            ctx.live_objects(),
            base,
            "{step:?}: nothing stays allocated"
        );
        assert_eq!(
            ctx.armed_faults(),
            0,
            "{step:?}: the fault was consumed by the step it names"
        );
    }
    // the context works afterwards
    let buf = ctx.buffer(64).expect("a later buffer");
    ctx.upload(&buf, &[5u8; 64]).expect("upload");
    assert_eq!(ctx.download(&buf).expect("download"), vec![5u8; 64]);
    ctx.destroy_buffer(buf);
    assert_eq!(ctx.live_objects(), base);
}

#[test]
fn a_failure_at_any_step_of_a_kernel_leaves_no_object_behind() {
    let Some(ctx) = open_or_unmeasured("kernel steps") else {
        return;
    };
    let base = ctx.live_objects();
    for step in [
        CreateStep::ShaderModule,
        CreateStep::DescriptorSetLayout,
        CreateStep::PipelineLayout,
        CreateStep::Pipeline,
        CreateStep::DescriptorPool,
        CreateStep::DescriptorSet,
    ] {
        let r = injected(&ctx, step);
        let err = match ctx.kernel(CAST_F32_BF16, 2, 4) {
            Ok(k) => {
                ctx.destroy_kernel(k);
                panic!("{step:?}: the kernel must fail");
            }
            Err(e) => e,
        };
        assert!(
            matches!(err, AckError::Vulkan(e) if e == r),
            "{step:?}: {err}"
        );
        assert_eq!(
            ctx.live_objects(),
            base,
            "{step:?}: nothing stays allocated"
        );
        assert_eq!(
            ctx.armed_faults(),
            0,
            "{step:?}: the fault was consumed by the step it names"
        );
    }
    let kernel = ctx.kernel(CAST_F32_BF16, 2, 4).expect("a later kernel");
    assert_eq!(ctx.live_objects(), base + 4);
    ctx.destroy_kernel(kernel);
    assert_eq!(ctx.live_objects(), base);
}

#[test]
fn a_failure_inside_a_submission_or_a_timed_dispatch_leaves_no_object_behind() {
    let Some(ctx) = open_or_unmeasured("submission steps") else {
        return;
    };
    let buf = ctx.buffer(4096).expect("buffer");
    let base = ctx.live_objects();
    // an upload creates a staging pair, a command buffer and a fence; a failure
    // at the command buffer or the fence releases all of them, and so does the
    // same failure inside a download (its own pre-submission release path)
    for step in [CreateStep::CommandBuffer, CreateStep::Fence] {
        let r = injected(&ctx, step);
        let err = ctx
            .upload(&buf, &vec![2u8; 4096])
            .expect_err("the upload must fail");
        assert!(
            matches!(err, AckError::Vulkan(e) if e == r),
            "upload {step:?}: {err}"
        );
        assert_eq!(
            ctx.live_objects(),
            base,
            "upload {step:?}: staging, command buffer and fence released"
        );
        assert!(
            !ctx.is_poisoned(),
            "a failure before the submission is not a poison"
        );
        let r = injected(&ctx, step);
        let err = ctx.download(&buf).expect_err("the download must fail");
        assert!(
            matches!(err, AckError::Vulkan(e) if e == r),
            "download {step:?}: {err}"
        );
        assert_eq!(
            ctx.live_objects(),
            base,
            "download {step:?}: staging, command buffer and fence released"
        );
    }
    let kernel = ctx.kernel(CAST_F32_BF16, 2, 4).expect("kernel");
    let out = ctx.buffer(2048).expect("bf16 output");
    ctx.bind(&kernel, &[&buf, &out]).expect("bind");
    let after_kernel = ctx.live_objects();
    let count: u32 = 1024;
    // the query pool itself failing (the wrapper's own cleanup), then a failure
    // AFTER the pool exists (dispatch_timed's own release of the pool)
    for step in [
        CreateStep::QueryPool,
        CreateStep::CommandBuffer,
        CreateStep::Fence,
    ] {
        let r = injected(&ctx, step);
        let err = ctx
            .dispatch_timed(&kernel, &count.to_le_bytes(), [4, 1, 1], 1)
            .expect_err("the dispatch must fail");
        assert!(
            matches!(err, AckError::Vulkan(e) if e == r),
            "dispatch {step:?}: {err}"
        );
        assert_eq!(
            ctx.live_objects(),
            after_kernel,
            "dispatch {step:?}: the query pool did not stay allocated"
        );
        assert!(!ctx.is_poisoned());
    }
    ctx.dispatch_timed(&kernel, &count.to_le_bytes(), [4, 1, 1], 1)
        .expect("a later dispatch succeeds");
    ctx.destroy_kernel(kernel);
    ctx.destroy_buffer(out);
    ctx.destroy_buffer(buf);
    assert_eq!(ctx.live_objects(), base - 2);
}

#[test]
fn a_refusal_before_any_object_exists_creates_nothing() {
    let Some(mut ctx) = open_or_unmeasured("early refusals") else {
        return;
    };
    let base = ctx.live_objects();
    // a fault armed on the FIRST creation step stays armed if the refusal
    // really precedes every creation; a refusal that created and destroyed
    // objects first would have consumed it (and the count alone cannot tell
    // those two apart)
    let _ = injected(&ctx, CreateStep::ShaderModule);
    assert!(
        matches!(
            ctx.kernel(CAST_F32_BF16, 2, 6),
            Err(AckError::Unsupported(_))
        ),
        "push bytes not a multiple of 4"
    );
    assert_eq!(
        ctx.armed_faults(),
        1,
        "the push-range refusal created no shader module"
    );
    // the fixed-subgroup refusal, driven on this card by clearing the flag the
    // check reads (the field is plain data, set once at open)
    let had_control = ctx.subgroup_size_control;
    ctx.subgroup_size_control = false;
    assert!(matches!(
        ctx.kernel_with_subgroup(CAST_F32_BF16, 2, 4, Some(64)),
        Err(AckError::Unsupported(_))
    ));
    ctx.subgroup_size_control = had_control;
    assert_eq!(
        ctx.armed_faults(),
        1,
        "the subgroup refusal created no shader module either"
    );
    let _ = injected(&ctx, CreateStep::Buffer);
    assert!(matches!(ctx.buffer(0), Err(AckError::Unsupported(_))));
    assert_eq!(
        ctx.armed_faults(),
        2,
        "the zero-size refusal created no buffer"
    );
    ctx.clear_faults();
    assert_eq!(ctx.live_objects(), base);
    // and with the faults cleared the same calls succeed
    let k = ctx.kernel(CAST_F32_BF16, 2, 4).expect("kernel");
    ctx.destroy_kernel(k);
    assert_eq!(ctx.live_objects(), base);
}
