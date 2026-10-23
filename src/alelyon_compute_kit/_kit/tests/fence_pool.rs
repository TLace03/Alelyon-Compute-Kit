//! The fence pool: a completed submission's fence is reset and kept for the
//! next submission instead of destroyed.
//!
//! Measured on an NVIDIA L4 (driver 580.173.02) under the torch adapter, one
//! A0 training update made 2,186 `vkCreateFence` and 2,187 `vkDestroyFence`
//! calls, 2.03 s of host time between them: one pair per batch and per
//! one-shot submission (docs/audits/2026-09-22-ack-nvidia-l4-step-profile.md).
//!
//! What may go wrong is narrow, and each test here pins one piece of it:
//!
//! - A fence is pooled only after a wait on it SUCCEEDED, so nothing can still
//!   be using it, and it is reset when it is pooled, so the next submission is
//!   handed an unsignalled fence. A failed wait destroys it, as before.
//! - A pooled fence is a LIVE object. `live_objects()` counts it and
//!   `pooled_objects()` names it, exactly as a pooled buffer pair; turning the
//!   pool off drains it and the count closes on the baseline.
//! - A pool hit still consults the fault injector at `CreateStep::Fence`, so a
//!   cleanup-path test that arms that fault cannot become a test of nothing
//!   because a pooled fence happened to be available.
//!
//! EVERY TEST ASSERTS THE POOL'S OWN COUNT as well as the arithmetic. A
//! correctness probe cannot tell a reused fence from a fresh one: a pool that
//! silently created a fence every time would pass every value assertion.
//!
//! The pool is OFF by default in the crate (the leak suites pin a fresh fence
//! per submission) and each test turns it on for its own context. Needs a
//! Vulkan device. Without one every test here prints UNMEASURED and returns,
//! which is a skip with its reason, never a pass; set `ACK_REQUIRE_DEVICE=1`
//! to turn that into a failure on a machine that is supposed to have the
//! device. A GREEN RUN WITHOUT THAT VARIABLE IS NOT DEVICE EVIDENCE.

use alelyon_compute_kit::batch::{env_flag, FENCE_POOL_MAX};
use alelyon_compute_kit::pointwise_ops::{
    PointwiseOp, PointwisePlan, Scalars, Storage, StridedView, OPERANDS, PUSH_BYTES,
};
use alelyon_compute_kit::{Access, Buffer, Context, Kernel};

const POINTWISE_F32: &[u8] = include_bytes!("../kernels/pointwise_f32.spv");

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

/// A context with the fence pool on and the buffer pool off, so that every
/// object delta below is a fence's or a buffer's own and not a pooled pair's.
fn pooled_fences_context(what: &str) -> Option<Context> {
    let ctx = open_or_unmeasured(what)?;
    ctx.set_buffer_pool_enabled(false);
    ctx.set_fence_pool_enabled(true);
    assert!(ctx.fence_pool_enabled());
    Some(ctx)
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn from_bytes(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn contiguous(shape: &[u64]) -> StridedView {
    StridedView::contiguous(shape).expect("contiguous view")
}

fn storage_of(b: &Buffer) -> Storage {
    Storage::new(b.id(), b.bytes / 4)
}

fn pointwise_kernel(ctx: &Context) -> Kernel {
    ctx.kernel(POINTWISE_F32, OPERANDS as u32, PUSH_BYTES as u32)
        .expect("pointwise kernel")
}

/// `dst = src + 1 * src` over `shape`: a doubling, so a link that ran early,
/// twice, or not at all leaves a different value at the chain's end (a copy
/// chain reads the same either way once the first link has landed). Doubling
/// is exact in f32 at every magnitude the chains below reach.
fn double_plan(shape: &[u64], src: &Buffer, dst: &Buffer) -> (PointwisePlan, [u8; PUSH_BYTES]) {
    let view = contiguous(shape);
    let views: [StridedView; OPERANDS] = [view, view, view, view];
    // x and y both read `src`; z, which `AddScaled` does not read, is bound to
    // the output, the plan's own rule for an unread slot
    let storage: [Storage; OPERANDS] = [
        storage_of(src),
        storage_of(src),
        storage_of(dst),
        storage_of(dst),
    ];
    let plan = PointwisePlan::new(
        PointwiseOp::AddScaled,
        shape,
        views,
        Scalars::new(1.0),
        storage,
    )
    .expect("doubling plan");
    let push = plan.push_constants();
    (plan, push)
}

fn access_for(src: &Buffer, dst: &Buffer) -> [Access; OPERANDS] {
    [
        Access::read(src.id()),
        Access::read(src.id()),
        Access::read(dst.id()),
        Access::write(dst.id()),
    ]
}

fn timestamps_off() -> bool {
    match std::env::var_os("ACK_DISPATCH_TIMESTAMPS") {
        None => true,
        Some(v) => v == "0",
    }
}

/// `links` doublings, each reading the previous one's output, at batch
/// depth `depth`; returns what the host expects at the end and what the
/// device read back. Every buffer and the kernel are released before return.
fn doubling_chain(ctx: &Context, depth: u32, links: usize) -> (Vec<f32>, Vec<f32>) {
    ctx.set_max_recorded_items(depth);
    let n = 4096u64;
    let start: Vec<f32> = (0..n).map(|i| (i % 97) as f32 * 0.25 - 3.0).collect();
    let mut buffers = Vec::with_capacity(links + 1);
    for _ in 0..=links {
        buffers.push(ctx.buffer(n * 4).expect("link"));
    }
    ctx.upload(&buffers[0], &f32_bytes(&start)).expect("upload");
    let kernel = pointwise_kernel(ctx);
    let shape = [n];
    for i in 0..links {
        let (src, dst) = (&buffers[i], &buffers[i + 1]);
        let (plan, push) = double_plan(&shape, src, dst);
        ctx.bind(&kernel, &[src, src, dst, dst]).expect("bind");
        ctx.dispatch(
            &kernel,
            &push,
            plan.dispatch_groups(),
            1,
            &access_for(src, dst),
        )
        .expect("record");
    }
    let back = from_bytes(&ctx.download(&buffers[links]).expect("download"));
    // the host's own chain: |start| <= 21 and 2^40 * 21 is far inside f32's
    // range, so every doubling is exact on both sides
    let expected: Vec<f32> = start
        .iter()
        .map(|&x| (0..links).fold(x, |v, _| v + v))
        .collect();
    ctx.destroy_kernel(kernel);
    for b in buffers {
        ctx.destroy_buffer(b);
    }
    (expected, back)
}

#[test]
fn the_crate_opens_with_the_pool_off_unless_the_environment_turns_it_on() {
    let Some(ctx) = open_or_unmeasured("fence pool default") else {
        return;
    };
    // the kit's own reading of the variable, so a value it takes as "on" is
    // the one this expects
    let from_env = env_flag("ACK_FENCE_POOL", false);
    assert_eq!(
        ctx.fence_pool_enabled(),
        from_env,
        "the crate's default is a fresh fence per submission; ACK_FENCE_POOL=1 at open is \
         the switch the torch adapter uses"
    );
    assert_eq!(ctx.pooled_fences(), 0, "a fresh context holds no fence");
}

#[test]
fn a_completed_submission_keeps_its_fence_and_the_next_submission_takes_it() {
    let Some(ctx) = pooled_fences_context("fence pool hit") else {
        return;
    };
    let base = ctx.live_objects();
    let n = 1024usize;
    let b = ctx.buffer((n * 4) as u64).expect("b");
    assert_eq!(ctx.live_objects(), base + 2, "a buffer is two objects");
    assert_eq!(ctx.pooled_fences(), 0, "nothing has been submitted yet");

    let first = f32_bytes(&vec![1.5f32; n]);
    ctx.upload(&b, &first).expect("upload");
    assert_eq!(
        ctx.pooled_fences(),
        1,
        "the upload's submission completed, so its fence was reset and kept"
    );
    assert_eq!(
        ctx.live_objects(),
        base + 3,
        "a pooled fence is a LIVE object: live_objects() counts it"
    );
    assert_eq!(
        ctx.pooled_objects(),
        1,
        "and pooled_objects() names it as the pools' share of that count"
    );

    let torn_before = ctx.submissions_torn_down();
    for round in 0..5u32 {
        let v = f32_bytes(&vec![round as f32 + 0.25; n]);
        ctx.upload(&b, &v).expect("upload");
        assert_eq!(ctx.download(&b).expect("download"), v, "round {round}");
    }
    assert_eq!(
        ctx.submissions_torn_down(),
        torn_before + 10,
        "ten submissions, each released once"
    );
    assert_eq!(
        ctx.pooled_fences(),
        1,
        "ten sequential submissions took the one pooled fence and gave it back: a fresh \
         fence per submission would have left the pool holding more than one"
    );
    assert_eq!(ctx.live_objects(), base + 3, "and created nothing");

    ctx.destroy_buffer(b);
    assert_eq!(
        ctx.live_objects() - ctx.pooled_objects(),
        base,
        "back to the baseline, once what the pool holds is named"
    );
    assert_eq!(ctx.leaked_on_poison(), 0, "HELD is not LEAKED");
}

#[test]
fn turning_the_fence_pool_off_drains_it_and_restores_a_fence_per_submission() {
    let Some(ctx) = pooled_fences_context("fence pool off") else {
        return;
    };
    let base = ctx.live_objects();
    let b = ctx.buffer(4096).expect("b");
    ctx.upload(&b, &vec![0u8; 4096]).expect("upload");
    assert_eq!(ctx.pooled_fences(), 1);
    assert_eq!(ctx.live_objects(), base + 3);

    ctx.set_fence_pool_enabled(false);
    assert!(!ctx.fence_pool_enabled());
    assert_eq!(
        ctx.pooled_fences(),
        0,
        "turning the pool off destroys what it holds"
    );
    assert_eq!(
        ctx.live_objects(),
        base + 2,
        "and the count closes: only the buffer is left"
    );
    ctx.upload(&b, &vec![1u8; 4096]).expect("upload");
    assert_eq!(
        ctx.live_objects(),
        base + 2,
        "with the pool off a submission creates its fence and destroys it, exactly as it \
         did before the pool existed"
    );
    assert_eq!(ctx.pooled_fences(), 0);
    ctx.destroy_buffer(b);
    assert_eq!(ctx.live_objects(), base);
}

#[test]
fn batches_in_flight_draw_on_the_pool_and_a_dependent_chain_still_arrives_intact() {
    let Some(ctx) = pooled_fences_context("fence pool across batches") else {
        return;
    };
    if !timestamps_off() {
        eprintln!("UNMEASURED here: the timestamps instrument completes every flush");
        return;
    }
    let base = ctx.live_objects();
    // depth 1: every link is its own batch, submitted while the previous one
    // may still be running, so the in-flight batch, the open batch and the
    // download's one-shot each hold a fence at once; depth 4: links straddle
    // batches. 40 links at depth 1 is 40 batches: far more submissions than
    // the pool may hold.
    for depth in [1u32, 4] {
        let torn_before = ctx.submissions_torn_down();
        let (expected, back) = doubling_chain(&ctx, depth, 40);
        assert_eq!(
            back, expected,
            "depth {depth}: every link ran once, in order"
        );
        let submissions = ctx.submissions_torn_down() - torn_before;
        assert!(
            submissions >= 40 / depth as u64,
            "depth {depth}: {submissions} submissions released, fewer than the batches the \
             chain needs"
        );
        let held = ctx.pooled_fences();
        assert!(
            held >= 1 && (held as usize) < FENCE_POOL_MAX,
            "depth {depth}: {submissions} submissions left {held} fences pooled; the pool \
             must be drawn on (a fence created per submission would fill it to \
             {FENCE_POOL_MAX}) and hold at least the last one"
        );
        assert_eq!(
            ctx.batches_in_flight(),
            0,
            "the readback finished the last batch"
        );
        assert_eq!(
            ctx.live_objects() - ctx.pooled_objects(),
            base,
            "depth {depth}: nothing but pooled fences outlives the chain"
        );
    }

    // the control arm on the same context: the pool off, the same chain, the
    // same bytes
    ctx.set_fence_pool_enabled(false);
    let (expected, back) = doubling_chain(&ctx, 1, 40);
    assert_eq!(
        back, expected,
        "pool off: the same chain arrives the same way"
    );
    assert_eq!(
        ctx.live_objects(),
        base,
        "and the unpooled count closes exactly"
    );
}

#[cfg(feature = "fault-injection")]
#[test]
fn a_fence_fault_fires_on_a_pool_hit_and_leaves_the_pool_as_it_was() {
    use alelyon_compute_kit::{AckError, CreateStep, Fault};
    let Some(ctx) = pooled_fences_context("fence pool: create fault") else {
        return;
    };
    let b = ctx.buffer(4096).expect("b");
    ctx.upload(&b, &vec![0u8; 4096]).expect("upload");
    assert_eq!(
        ctx.pooled_fences(),
        1,
        "a pooled fence is available for the next call"
    );
    let live = ctx.live_objects();

    ctx.inject_fault(Fault::CreateFails(
        CreateStep::Fence,
        ash::vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
    ));
    let err = ctx
        .upload(&b, &vec![1u8; 4096])
        .expect_err("the armed fault fires although the pool could serve the fence");
    assert!(
        matches!(
            err,
            AckError::Vulkan(ash::vk::Result::ERROR_OUT_OF_DEVICE_MEMORY)
        ),
        "{err}"
    );
    assert_eq!(ctx.armed_faults(), 0, "the pool hit consumed the fault");
    assert_eq!(
        ctx.pooled_fences(),
        1,
        "the fence the failed call was handed went back to the pool"
    );
    assert_eq!(
        ctx.live_objects(),
        live,
        "and the command buffer the call allocated before its fence was freed"
    );
    assert!(!ctx.is_poisoned(), "a creation failure is not a poison");

    ctx.upload(&b, &vec![2u8; 4096])
        .expect("the next call succeeds on the pooled fence");
    assert_eq!(ctx.download(&b).expect("download"), vec![2u8; 4096]);
    assert_eq!(ctx.pooled_fences(), 1);
    ctx.destroy_buffer(b);
}

#[cfg(feature = "fault-injection")]
#[test]
fn a_failed_wait_destroys_the_fence_rather_than_pooling_it() {
    use alelyon_compute_kit::{AckError, Fault};
    let Some(ctx) = pooled_fences_context("fence pool: failed wait") else {
        return;
    };
    let b = ctx.buffer(4096).expect("b");
    ctx.upload(&b, &vec![0u8; 4096]).expect("upload");
    assert_eq!(ctx.pooled_fences(), 1);
    let live = ctx.live_objects();
    let torn_before = ctx.submissions_torn_down();

    // the device quiesces after the injected failure, so the submission is
    // released rather than leaked -- but only a wait that SUCCEEDED may pool
    ctx.inject_fault(Fault::WaitFails(ash::vk::Result::ERROR_OUT_OF_HOST_MEMORY));
    let err = ctx
        .upload(&b, &vec![1u8; 4096])
        .expect_err("the injected wait failure surfaces");
    assert!(
        matches!(
            err,
            AckError::Vulkan(ash::vk::Result::ERROR_OUT_OF_HOST_MEMORY)
        ),
        "{err}"
    );
    assert!(!ctx.is_poisoned(), "the device quiesced");
    assert_eq!(
        ctx.submissions_torn_down(),
        torn_before + 1,
        "released once"
    );
    assert_eq!(
        ctx.pooled_fences(),
        0,
        "the failed submission took the pooled fence and DESTROYED it: a wait that did not \
         succeed does not establish that the fence is free to reset"
    );
    assert_eq!(
        ctx.live_objects(),
        live - 1,
        "one fence fewer, nothing leaked"
    );
    assert_eq!(ctx.leaked_on_poison(), 0);

    ctx.upload(&b, &vec![3u8; 4096])
        .expect("the context is still usable");
    assert_eq!(
        ctx.pooled_fences(),
        1,
        "and a completed submission pools again"
    );
    ctx.destroy_buffer(b);
}
